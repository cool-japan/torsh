use anyhow;
use torsh_core::dtype::DType;
use torsh_core::{Device, DeviceType};
use torsh_core::{Result, TorshError};
use torsh_nn::prelude::{Conv2d, Sequential};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

use super::{ModelConfig, VisionModel};
use crate::ops::nms;

/// YOLOv5 Detection Model
/// Implements the YOLOv5 architecture for object detection
pub struct YOLOv5 {
    backbone: Sequential,
    neck: Sequential,
    head: DetectionHead,
    num_classes: usize,
    input_size: (usize, usize),
    anchors: Tensor,
}

impl std::fmt::Debug for YOLOv5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("YOLOv5")
            .field("backbone", &"<Sequential>")
            .field("neck", &"<Sequential>")
            .field("head", &self.head)
            .field("num_classes", &self.num_classes)
            .field("input_size", &self.input_size)
            .field("anchors", &self.anchors)
            .finish()
    }
}

impl YOLOv5 {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let backbone = Self::build_backbone()?;
        let neck = Self::build_neck()?;
        let head = DetectionHead::new(config.num_classes, 3)?; // 3 detection scales

        // YOLO anchor boxes for 3 scales (small, medium, large objects)
        let anchors = Tensor::from_data(
            vec![
                // P3/8 (small objects)
                10.0, 13.0, 16.0, 30.0, 33.0, 23.0, // P4/16 (medium objects)
                30.0, 61.0, 62.0, 45.0, 59.0, 119.0, // P5/32 (large objects)
                116.0, 90.0, 156.0, 198.0, 373.0, 326.0,
            ],
            vec![3, 3, 2], // [scales, anchors_per_scale, (width, height)]
            torsh_core::DeviceType::Cpu,
        )?;

        Ok(Self {
            backbone,
            neck,
            head,
            num_classes: config.num_classes,
            input_size: (640, 640),
            anchors,
        })
    }

    fn build_backbone() -> Result<Sequential> {
        let mut builder = Sequential::new();

        // Focus layer - splits input into 4 slices and concatenates
        builder = builder.add(Conv2d::new(12, 32, (3, 3), (1, 1), (1, 1), (1, 1), true, 1)); // 3*4=12 input channels
        builder = builder.add(Conv2d::new(32, 64, (3, 3), (2, 2), (1, 1), (1, 1), true, 1));
        builder = builder.add(Conv2d::new(64, 64, (1, 1), (1, 1), (0, 0), (1, 1), true, 1));

        // C3 blocks (CSP bottleneck)
        builder = builder.add(Conv2d::new(
            64,
            128,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            128,
            128,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            true,
            1,
        ));

        builder = builder.add(Conv2d::new(
            128,
            256,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            256,
            256,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            true,
            1,
        ));

        builder = builder.add(Conv2d::new(
            256,
            512,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            512,
            512,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            true,
            1,
        ));

        builder = builder.add(Conv2d::new(
            512,
            1024,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            1024,
            1024,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            true,
            1,
        ));

        Ok(builder)
    }

    fn build_neck() -> Result<Sequential> {
        // SPP (Spatial Pyramid Pooling) + PANet (Path Aggregation Network)
        let mut builder = Sequential::new();

        // SPP layer with multiple pooling scales
        builder = builder.add(Conv2d::new(
            1024,
            512,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            true,
            1,
        ));

        // PANet upsampling and concatenation layers
        builder = builder.add(Conv2d::new(
            512,
            256,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            256,
            256,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            true,
            1,
        ));

        Ok(builder)
    }

    pub fn detect(
        &self,
        input: &Tensor,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let predictions = self.forward(input)?;
        self.post_process(predictions, conf_threshold, iou_threshold)
    }

    fn post_process(
        &self,
        predictions: Tensor,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();

        // predictions shape: [batch, anchors, 5 + num_classes]
        // 5 = x, y, w, h, confidence
        let batch_size = predictions.size(0)?;

        for batch_idx in 0..batch_size {
            let batch_preds = predictions.narrow(0, batch_idx as i64, 1)?.squeeze(0)?;

            // Filter by confidence threshold
            let confidences = batch_preds.narrow(1, 4, 1)?;
            let conf_threshold_tensor =
                Tensor::from_data(vec![conf_threshold], vec![1], DeviceType::Cpu)?;
            let conf_mask = confidences.gt(&conf_threshold_tensor)?;

            // Apply confidence filter
            let filtered_preds = batch_preds.masked_select(&conf_mask)?;

            if filtered_preds.numel() == 0 {
                continue;
            }

            // Extract boxes and scores
            let boxes = filtered_preds.narrow(1, 0, 4)?;
            let scores = filtered_preds.narrow(1, 4, 1)?;
            let class_probs = filtered_preds.narrow(1, 5, self.num_classes)?;

            // Get class predictions
            let class_scores = class_probs.max(Some(1), true)?;
            let class_indices = class_probs.argmax(Some(1))?;
            let final_scores = scores.squeeze(1)?.mul(&class_scores)?;

            // Create detection objects first
            let mut all_detections = Vec::new();
            let num_boxes = boxes.size(0)?;

            for i in 0..num_boxes {
                let box_coords = boxes.narrow(0, i as i64, 1)?.squeeze(0)?;
                let score = final_scores.narrow(0, i as i64, 1)?.item()?;
                let class_id = class_indices.narrow(0, i as i64, 1)?.item()? as usize;

                let bbox = [
                    box_coords.narrow(0, 0, 1)?.item()?,
                    box_coords.narrow(0, 1, 1)?.item()?,
                    box_coords.narrow(0, 2, 1)?.item()?,
                    box_coords.narrow(0, 3, 1)?.item()?,
                ];

                let detection = crate::ops::detection::Detection::new(bbox, score, class_id);
                all_detections.push(detection);
            }

            // Apply NMS
            let nms_config = crate::ops::detection::NMSConfig {
                iou_threshold,
                confidence_threshold: conf_threshold,
                max_detections: None,
                per_class: false,
            };

            let filtered_detections = crate::ops::detection::nms(all_detections, nms_config)
                .map_err(|e| TorshError::Other(e.to_string()))?;

            // Convert from ops::detection::Detection to models::detection::Detection
            for ops_detection in filtered_detections {
                let detection = Detection {
                    bbox: BoundingBox {
                        x: ops_detection.bbox[0],
                        y: ops_detection.bbox[1],
                        width: ops_detection.bbox[2] - ops_detection.bbox[0],
                        height: ops_detection.bbox[3] - ops_detection.bbox[1],
                    },
                    confidence: ops_detection.confidence,
                    class_id: ops_detection.class_id,
                };
                detections.push(detection);
            }
        }

        Ok(detections)
    }
}

impl Module for YOLOv5 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Focus operation - rearrange input
        let focused = self.focus_transform(x)?;

        // Backbone feature extraction
        let backbone_out = self.backbone.forward(&focused)?;

        // Neck processing
        let neck_out = self.neck.forward(&backbone_out)?;

        // Detection head
        self.head.forward(&neck_out)
    }
}

impl YOLOv5 {
    fn focus_transform(&self, x: &Tensor) -> Result<Tensor> {
        // Split image into 4 quadrants and stack as channels
        // Input: [B, 3, H, W] -> Output: [B, 12, H/2, W/2]
        let b = x.size(0)?;
        let c = x.size(1)?;
        let h = x.size(2)?;
        let w = x.size(3)?;

        // For now, use basic slicing - step slicing would need custom implementation
        let top_left = x.narrow(2, 0, h / 2)?.narrow(3, 0, w / 2)?;
        let top_right = x.narrow(2, 0, h / 2)?.narrow(3, 1, w / 2)?;
        let bottom_left = x.narrow(2, 1, h / 2)?.narrow(3, 0, w / 2)?;
        let bottom_right = x.narrow(2, 1, h / 2)?.narrow(3, 1, w / 2)?;

        Tensor::cat(&[&top_left, &top_right, &bottom_left, &bottom_right], 1)
    }
}

impl VisionModel for YOLOv5 {
    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn input_size(&self) -> (usize, usize) {
        self.input_size
    }

    fn name(&self) -> &str {
        "YOLOv5"
    }
}

/// RetinaNet Detection Model  
/// Single-stage detector with Feature Pyramid Network
pub struct RetinaNet {
    backbone: Sequential,
    fpn: FeaturePyramidNetwork,
    classification_head: ClassificationHead,
    regression_head: RegressionHead,
    num_classes: usize,
    anchors: AnchorGenerator,
}

impl std::fmt::Debug for RetinaNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetinaNet")
            .field("backbone", &"<Sequential>")
            .field("fpn", &self.fpn)
            .field("classification_head", &self.classification_head)
            .field("regression_head", &self.regression_head)
            .field("num_classes", &self.num_classes)
            .field("anchors", &self.anchors)
            .finish()
    }
}

impl RetinaNet {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let backbone = Self::build_resnet_backbone()?;
        let fpn = FeaturePyramidNetwork::new()?;
        let classification_head = ClassificationHead::new(config.num_classes)?;
        let regression_head = RegressionHead::new()?;
        let anchors = AnchorGenerator::new(vec![32, 64, 128, 256, 512], vec![0.5, 1.0, 2.0])?;

        Ok(Self {
            backbone,
            fpn,
            classification_head,
            regression_head,
            num_classes: config.num_classes,
            anchors,
        })
    }

    fn build_resnet_backbone() -> Result<Sequential> {
        // ResNet-50 backbone for RetinaNet
        let mut builder = Sequential::new();

        // Initial conv block
        builder = builder.add(Conv2d::new(3, 64, (7, 7), (2, 2), (3, 3), (1, 1), true, 1));

        // ResNet blocks - simplified version
        builder = builder.add(Conv2d::new(
            64,
            256,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            256,
            512,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            512,
            1024,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            1024,
            2048,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        ));

        Ok(builder)
    }

    pub fn detect(
        &self,
        input: &Tensor,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let features = self.backbone.forward(input)?;
        let fpn_features = self.fpn.forward(&features)?;

        let classifications = self.classification_head.forward(&fpn_features)?;
        let regressions = self.regression_head.forward(&fpn_features)?;

        // Generate anchors
        let input_size = vec![
            input.size(0)?,
            input.size(1)?,
            input.size(2)?,
            input.size(3)?,
        ];
        let anchors = self.anchors.generate(&input_size)?;

        // Apply regression to anchors to get final boxes
        let boxes = self.apply_regression(&anchors, &regressions)?;

        // Post-process detections
        self.post_process_detections(boxes, classifications, score_threshold, nms_threshold)
    }

    fn apply_regression(&self, anchors: &Tensor, regressions: &Tensor) -> Result<Tensor> {
        // Convert anchor-relative predictions to absolute coordinates
        let anchor_centers = anchors.narrow(-1, 0, 2)?;
        let anchor_sizes = anchors.narrow(-1, 2, 2)?;

        let dx_dy = regressions.narrow(-1, 0, 2)?;
        let dw_dh = regressions.narrow(-1, 2, 2)?;

        let predicted_centers = anchor_centers.add(&dx_dy.mul(&anchor_sizes)?)?;
        let predicted_sizes = anchor_sizes.mul(&dw_dh.exp()?)?;

        Tensor::cat(&[&predicted_centers, &predicted_sizes], -1)
    }

    fn post_process_detections(
        &self,
        boxes: Tensor,
        classifications: Tensor,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<Detection>> {
        // Similar to YOLOv5 post-processing but adapted for RetinaNet output format
        let mut detections = Vec::new();

        // Apply score threshold
        let scores = classifications.sigmoid()?;
        let max_scores = scores.max_dim(-1, false)?;
        let score_mask = max_scores.gt(&Tensor::scalar(score_threshold)?)?;

        let filtered_boxes = boxes.masked_select(&score_mask)?;
        let filtered_scores = max_scores.masked_select(&score_mask)?;

        if filtered_boxes.numel() == 0 {
            return Ok(detections);
        }

        // Create detection objects first
        let mut all_detections = Vec::new();
        let num_boxes = filtered_boxes.size(0)?;

        for i in 0..num_boxes {
            let box_coords = filtered_boxes.select(0, i as i64)?;
            let score = filtered_scores.select(0, i as i64)?.item()?;

            let bbox = [
                box_coords.select(0, 0)?.item()?,
                box_coords.select(0, 1)?.item()?,
                box_coords.select(0, 2)?.item()?,
                box_coords.select(0, 3)?.item()?,
            ];

            let detection = crate::ops::detection::Detection::new(bbox, score, 0); // Simplified class_id
            all_detections.push(detection);
        }

        // Apply NMS
        let nms_config = crate::ops::detection::NMSConfig {
            iou_threshold: nms_threshold,
            confidence_threshold: 0.0, // Already filtered by confidence
            max_detections: None,
            per_class: false,
        };

        let filtered_detections = crate::ops::detection::nms(all_detections, nms_config)
            .map_err(|e| TorshError::Other(format!("NMS failed: {}", e)))?;

        // Convert from ops::detection::Detection to models::detection::Detection
        for ops_detection in filtered_detections {
            let detection = Detection {
                bbox: BoundingBox {
                    x: ops_detection.bbox[0],
                    y: ops_detection.bbox[1],
                    width: ops_detection.bbox[2] - ops_detection.bbox[0],
                    height: ops_detection.bbox[3] - ops_detection.bbox[1],
                },
                confidence: ops_detection.confidence,
                class_id: ops_detection.class_id,
            };
            detections.push(detection);
        }

        Ok(detections)
    }
}

impl Module for RetinaNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let features = self.backbone.forward(x)?;
        let fpn_features = self.fpn.forward(&features)?;

        let classifications = self.classification_head.forward(&fpn_features)?;
        let regressions = self.regression_head.forward(&fpn_features)?;

        // Combine classification and regression outputs
        Tensor::cat(&[&classifications, &regressions], -1)
    }
}

impl VisionModel for RetinaNet {
    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn input_size(&self) -> (usize, usize) {
        (800, 800)
    }

    fn name(&self) -> &str {
        "RetinaNet"
    }
}

/// Single Shot Detector (SSD) Model
pub struct SSD {
    backbone: Sequential,
    extra_layers: Sequential,
    classification_heads: Vec<Conv2d>,
    regression_heads: Vec<Conv2d>,
    num_classes: usize,
    default_boxes: AnchorGenerator,
}

impl SSD {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let backbone = Self::build_vgg_backbone()?;
        let extra_layers = Self::build_extra_layers()?;

        // Multiple detection heads for different feature map scales
        let mut classification_heads = Vec::new();
        let mut regression_heads = Vec::new();

        let feature_sizes = vec![512, 1024, 512, 256, 256, 256];
        let num_boxes = vec![4, 6, 6, 6, 4, 4]; // Default boxes per location

        for (i, &feature_size) in feature_sizes.iter().enumerate() {
            classification_heads.push(Conv2d::new(
                feature_size,
                num_boxes[i] * config.num_classes,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                false,
                1,
            ));
            regression_heads.push(Conv2d::new(
                feature_size,
                num_boxes[i] * 4, // 4 for bbox coordinates
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                false,
                1,
            ));
        }

        let default_boxes = AnchorGenerator::new(
            vec![30, 60, 111, 162, 213, 264, 315], // scales
            vec![1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],   // aspect ratios
        )?;

        Ok(Self {
            backbone,
            extra_layers,
            classification_heads,
            regression_heads,
            num_classes: config.num_classes,
            default_boxes,
        })
    }

    fn build_vgg_backbone() -> Result<Sequential> {
        // VGG-16 backbone modified for SSD
        let mut builder = Sequential::new();

        // VGG-16 layers up to conv5_3
        builder = builder.add(Conv2d::new(3, 64, (3, 3), (1, 1), (1, 1), (1, 1), false, 1));
        builder = builder.add(Conv2d::new(
            64,
            64,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            false,
            1,
        ));
        builder = builder.add(Conv2d::new(
            64,
            128,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        )); // pool1

        builder = builder.add(Conv2d::new(
            128,
            128,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            false,
            1,
        ));
        builder = builder.add(Conv2d::new(
            128,
            256,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        )); // pool2

        builder = builder.add(Conv2d::new(
            256,
            256,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            false,
            1,
        ));
        builder = builder.add(Conv2d::new(
            256,
            256,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            false,
            1,
        ));
        builder = builder.add(Conv2d::new(
            256,
            512,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        )); // pool3

        builder = builder.add(Conv2d::new(
            512,
            512,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            false,
            1,
        ));
        builder = builder.add(Conv2d::new(
            512,
            512,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            512,
            512,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        )); // pool4

        builder = builder.add(Conv2d::new(
            512,
            512,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            512,
            512,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            512,
            512,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            true,
            1,
        ));

        Ok(builder)
    }

    fn build_extra_layers() -> Result<Sequential> {
        // Additional layers for multi-scale feature extraction
        let mut builder = Sequential::new();

        builder = builder.add(Conv2d::new(
            512,
            1024,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            1024,
            1024,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            true,
            1,
        ));

        builder = builder.add(Conv2d::new(
            1024,
            256,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            256,
            512,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        ));

        builder = builder.add(Conv2d::new(
            512,
            128,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            true,
            1,
        ));
        builder = builder.add(Conv2d::new(
            128,
            256,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            true,
            1,
        ));

        Ok(builder)
    }
}

impl Module for SSD {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let backbone_features = self.backbone.forward(x)?;
        let extra_features = self.extra_layers.forward(&backbone_features)?;

        // Apply classification and regression heads to each feature map
        let mut classifications = Vec::new();
        let mut regressions = Vec::new();

        // Apply heads to backbone_features and extra_features
        for (i, (cls_head, reg_head)) in self
            .classification_heads
            .iter()
            .zip(self.regression_heads.iter())
            .enumerate()
        {
            let features = if i == 0 {
                &backbone_features
            } else {
                &extra_features
            };

            let cls_output = cls_head.forward(features)?;
            let reg_output = reg_head.forward(features)?;

            classifications.push(cls_output);
            regressions.push(reg_output);
        }

        // Concatenate all outputs
        let classification_refs: Vec<&Tensor> = classifications.iter().collect();
        let regression_refs: Vec<&Tensor> = regressions.iter().collect();
        let all_classifications = Tensor::cat(&classification_refs, 1)?;
        let all_regressions = Tensor::cat(&regression_refs, 1)?;

        Tensor::cat(&[&all_classifications, &all_regressions], -1)
    }
}

impl VisionModel for SSD {
    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn input_size(&self) -> (usize, usize) {
        (300, 300)
    }

    fn name(&self) -> &str {
        "SSD"
    }
}

/// Detection output structure
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub confidence: f32,
    pub class_id: usize,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox {
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }
}

/// Detection Head for YOLO-style models
#[derive(Debug)]
struct DetectionHead {
    convs: Vec<Conv2d>,
    num_classes: usize,
    num_scales: usize,
}

impl DetectionHead {
    fn new(num_classes: usize, num_scales: usize) -> Result<Self> {
        let mut convs = Vec::new();

        // Create detection heads for each scale
        for _ in 0..num_scales {
            // Each anchor predicts: x, y, w, h, confidence, class_probs
            let output_size = 3 * (5 + num_classes); // 3 anchors per scale
            convs.push(Conv2d::new(
                256,
                output_size,
                (1, 1),
                (1, 1),
                (0, 0),
                (1, 1),
                false,
                1,
            ));
        }

        Ok(Self {
            convs,
            num_classes,
            num_scales,
        })
    }
}

impl Module for DetectionHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut outputs = Vec::new();

        for conv in &self.convs {
            let output = conv.forward(x)?;
            outputs.push(output);
        }

        // Concatenate outputs from all scales
        let output_refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&output_refs, 1)
    }
}

/// Feature Pyramid Network for RetinaNet
#[derive(Debug)]
struct FeaturePyramidNetwork {
    lateral_convs: Vec<Conv2d>,
    fpn_convs: Vec<Conv2d>,
}

impl FeaturePyramidNetwork {
    fn new() -> Result<Self> {
        // Lateral connections from backbone to FPN
        let lateral_convs = vec![
            Conv2d::new(256, 256, (1, 1), (1, 1), (0, 0), (1, 1), false, 1),
            Conv2d::new(512, 256, (1, 1), (1, 1), (0, 0), (1, 1), false, 1),
            Conv2d::new(1024, 256, (1, 1), (1, 1), (0, 0), (1, 1), false, 1),
            Conv2d::new(2048, 256, (1, 1), (1, 1), (0, 0), (1, 1), false, 1),
        ];

        // FPN output convolutions
        let fpn_convs = vec![
            Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
        ];

        Ok(Self {
            lateral_convs,
            fpn_convs,
        })
    }
}

impl Module for FeaturePyramidNetwork {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Simplified FPN - in practice, this would process multiple feature maps
        let lateral = self.lateral_convs[0].forward(x)?;
        self.fpn_convs[0].forward(&lateral)
    }
}

/// Classification Head for RetinaNet
#[derive(Debug)]
struct ClassificationHead {
    convs: Vec<Conv2d>,
    output_conv: Conv2d,
}

impl ClassificationHead {
    fn new(num_classes: usize) -> Result<Self> {
        let convs = vec![
            Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
        ];

        // A anchors * K classes
        let output_conv = Conv2d::new(
            256,
            9 * num_classes,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            false,
            1,
        );

        Ok(Self { convs, output_conv })
    }
}

impl Module for ClassificationHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();

        for conv in &self.convs {
            out = conv.forward(&out)?;
            out = out.relu()?;
        }

        self.output_conv.forward(&out)
    }
}

/// Regression Head for RetinaNet  
#[derive(Debug)]
struct RegressionHead {
    convs: Vec<Conv2d>,
    output_conv: Conv2d,
}

impl RegressionHead {
    fn new() -> Result<Self> {
        let convs = vec![
            Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
        ];

        // A anchors * 4 coordinates
        let output_conv = Conv2d::new(256, 9 * 4, (3, 3), (1, 1), (1, 1), (1, 1), false, 1);

        Ok(Self { convs, output_conv })
    }
}

impl Module for RegressionHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();

        for conv in &self.convs {
            out = conv.forward(&out)?;
            out = out.relu()?;
        }

        self.output_conv.forward(&out)
    }
}

/// Anchor Generator for detection models
#[derive(Debug)]
struct AnchorGenerator {
    scales: Vec<f32>,
    aspect_ratios: Vec<f32>,
}

impl AnchorGenerator {
    fn new(scales: Vec<i32>, aspect_ratios: Vec<f32>) -> Result<Self> {
        let scales = scales.into_iter().map(|s| s as f32).collect();
        Ok(Self {
            scales,
            aspect_ratios,
        })
    }

    fn generate(&self, input_size: &[usize]) -> Result<Tensor> {
        // Generate anchor boxes for the given input size
        // This is a simplified version - real implementation would be more complex
        let height = input_size[2] as f32;
        let width = input_size[3] as f32;

        let mut anchors = Vec::new();

        for &scale in &self.scales {
            for &ratio in &self.aspect_ratios {
                let anchor_height = scale * ratio.sqrt();
                let anchor_width = scale / ratio.sqrt();

                // Center the anchor at (width/2, height/2) for simplicity
                let x = width / 2.0 - anchor_width / 2.0;
                let y = height / 2.0 - anchor_height / 2.0;

                anchors.extend_from_slice(&[x, y, anchor_width, anchor_height]);
            }
        }

        Tensor::from_data(
            anchors,
            vec![self.scales.len() * self.aspect_ratios.len(), 4],
            torsh_core::DeviceType::Cpu,
        )
    }
}

/// Factory functions for easy model creation
pub fn yolo_v5_small(num_classes: usize) -> Result<YOLOv5> {
    YOLOv5::new(ModelConfig {
        num_classes,
        dropout: 0.1,
        pretrained: false,
    })
}

pub fn yolo_v5_medium(num_classes: usize) -> Result<YOLOv5> {
    YOLOv5::new(ModelConfig {
        num_classes,
        dropout: 0.2,
        pretrained: false,
    })
}

pub fn retina_net_resnet50(num_classes: usize) -> Result<RetinaNet> {
    RetinaNet::new(ModelConfig {
        num_classes,
        dropout: 0.1,
        pretrained: false,
    })
}

pub fn ssd_300(num_classes: usize) -> Result<SSD> {
    SSD::new(ModelConfig {
        num_classes,
        dropout: 0.1,
        pretrained: false,
    })
}

pub fn ssd_512(num_classes: usize) -> Result<SSD> {
    SSD::new(ModelConfig {
        num_classes,
        dropout: 0.2,
        pretrained: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yolo_v5_creation() {
        let model = yolo_v5_small(80).unwrap(); // COCO classes
        assert_eq!(model.num_classes(), 80);
        assert_eq!(model.input_size(), (640, 640));
        assert_eq!(VisionModel::name(&model), "YOLOv5");
    }

    #[test]
    fn test_retina_net_creation() {
        let model = retina_net_resnet50(80).unwrap();
        assert_eq!(model.num_classes(), 80);
        assert_eq!(model.input_size(), (800, 800));
        assert_eq!(VisionModel::name(&model), "RetinaNet");
    }

    #[test]
    fn test_ssd_creation() {
        let model = ssd_300(21).unwrap(); // VOC classes
        assert_eq!(model.num_classes(), 21);
        assert_eq!(model.input_size(), (300, 300));
        assert_eq!(VisionModel::name(&model), "SSD");
    }

    #[test]
    fn test_bounding_box() {
        let bbox = BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 40.0,
        };

        assert_eq!(bbox.area(), 1200.0);
        assert_eq!(bbox.center(), (25.0, 40.0));
    }

    #[test]
    fn test_anchor_generator() {
        let generator = AnchorGenerator::new(vec![32, 64], vec![0.5, 1.0, 2.0]).unwrap();
        let anchors = generator.generate(&[1, 3, 224, 224]).unwrap();

        // Should generate 2 scales * 3 ratios = 6 anchors
        assert_eq!(anchors.shape().dims(), &[6, 4]);
    }

    #[test]
    fn test_detection_structure() {
        let detection = Detection {
            bbox: BoundingBox {
                x: 100.0,
                y: 150.0,
                width: 50.0,
                height: 75.0,
            },
            confidence: 0.85,
            class_id: 0,
        };

        assert_eq!(detection.confidence, 0.85);
        assert_eq!(detection.class_id, 0);
        assert_eq!(detection.bbox.area(), 3750.0);
    }
}
