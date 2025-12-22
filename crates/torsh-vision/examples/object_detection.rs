//! Object Detection Example
//!
//! This example demonstrates:
//! - Loading object detection datasets (VOC, COCO)
//! - Using detection models (YOLO, RetinaNet, SSD)
//! - Non-Maximum Suppression (NMS)
//! - Bounding box operations and IoU calculation
//! - Evaluation metrics for detection
//! - Visualization of detection results
//!
//! Run with: cargo run --example object_detection --features pretrained

use std::path::PathBuf;
use std::sync::Arc;
use torsh_core::device::CpuDevice;
use torsh_tensor::{creation, Tensor};
use torsh_vision::{
    calculate_iou, generate_anchors, nms, retina_net_resnet50, ssd_300, yolo_v5_small, Normalize,
    Resize, Result, VisionModel, VocDataset,
};
// Use types from ops::detection module
use torsh_vision::ops::detection::{AnchorConfig, Detection, NMSConfig};

/// Configuration for object detection
#[derive(Debug, Clone)]
struct DetectionConfig {
    confidence_threshold: f32,
    nms_threshold: f32,
    max_detections: usize,
    input_size: (usize, usize),
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            nms_threshold: 0.4,
            max_detections: 100,
            input_size: (640, 640),
        }
    }
}

/// Run inference with a detection model
fn run_detection(
    model: &dyn VisionModel,
    image: &Tensor,
    config: &DetectionConfig,
) -> Result<Vec<Detection>> {
    // Preprocess image
    let preprocessed = preprocess_image(image, config.input_size)?;

    // Run model inference
    let output = model.forward(&preprocessed)?;

    // Post-process predictions
    let detections = postprocess_predictions(output, config)?;

    Ok(detections)
}

/// Preprocess image for detection
fn preprocess_image(image: &Tensor, target_size: (usize, usize)) -> Result<Tensor> {
    // Resize to target size
    let resize = Resize::new(target_size);
    let resized = resize.apply(image)?;

    // Normalize with ImageNet statistics
    let normalize = Normalize::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225]);
    let normalized = normalize.apply(&resized)?;

    // Add batch dimension
    let batched = normalized.unsqueeze(0)?;

    Ok(batched)
}

/// Post-process model predictions
fn postprocess_predictions(
    predictions: Tensor,
    config: &DetectionConfig,
) -> Result<Vec<Detection>> {
    // Extract boxes, scores, and class predictions
    // Note: Actual implementation depends on model output format

    let batch_size = predictions.shape().dims()[0] as usize;
    let num_predictions = predictions.shape().dims()[1] as usize;

    let mut all_detections = Vec::new();

    for batch_idx in 0..batch_size {
        let batch_pred = predictions.narrow(0, batch_idx as i64, 1)?;

        // Extract boxes and scores (simplified)
        // In real implementation, this depends on model architecture
        let boxes = batch_pred.narrow(2, 0, 4)?; // [x1, y1, x2, y2]
        let scores = batch_pred.narrow(2, 4, 1)?;
        let class_scores = batch_pred.narrow(2, 5, batch_pred.shape().dims()[2] as i64 - 5)?;

        // Apply confidence threshold
        let mut detections = Vec::new();
        for i in 0..num_predictions {
            let score: f32 = scores.narrow(1, i as i64, 1)?.item();

            if score >= config.confidence_threshold {
                let box_coords = boxes.narrow(1, i as i64, 1)?;
                let x1: f32 = box_coords.narrow(2, 0, 1)?.item();
                let y1: f32 = box_coords.narrow(2, 1, 1)?.item();
                let x2: f32 = box_coords.narrow(2, 2, 1)?.item();
                let y2: f32 = box_coords.narrow(2, 3, 1)?.item();

                let class_probs = class_scores.narrow(1, i as i64, 1)?;
                let class_id: usize = class_probs.argmax(Some(2), false)?.item() as usize;

                detections.push(Detection {
                    bbox: BoundingBox { x1, y1, x2, y2 },
                    class_id,
                    confidence: score,
                });
            }
        }

        // Apply NMS per class
        let filtered_detections = apply_nms_per_class(detections, config.nms_threshold)?;

        // Keep top-k detections
        let mut sorted = filtered_detections;
        sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        sorted.truncate(config.max_detections);

        all_detections.extend(sorted);
    }

    Ok(all_detections)
}

/// Apply NMS per class
fn apply_nms_per_class(detections: Vec<Detection>, threshold: f32) -> Result<Vec<Detection>> {
    // Group detections by class
    let mut class_detections: std::collections::HashMap<usize, Vec<Detection>> =
        std::collections::HashMap::new();

    for det in detections {
        class_detections
            .entry(det.class_id)
            .or_insert_with(Vec::new)
            .push(det);
    }

    let mut final_detections = Vec::new();

    // Apply NMS per class
    for (_class_id, mut class_dets) in class_detections {
        // Extract boxes and scores for NMS
        let boxes: Vec<Vec<f32>> = class_dets
            .iter()
            .map(|d| vec![d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2])
            .collect();

        let scores: Vec<f32> = class_dets.iter().map(|d| d.confidence).collect();

        // Convert to tensors
        let boxes_flat: Vec<f32> = boxes.iter().flat_map(|b| b.clone()).collect();
        let boxes_tensor = creation::tensor(
            &boxes_flat,
            &[boxes.len() as i64, 4],
            torsh_core::dtype::DType::Float32,
        )?;

        let scores_tensor = creation::tensor(
            &scores,
            &[scores.len() as i64],
            torsh_core::dtype::DType::Float32,
        )?;

        // Apply NMS
        let keep_indices = nms(&boxes_tensor, &scores_tensor, threshold)?;

        // Keep selected detections
        for idx in keep_indices {
            if idx < class_dets.len() {
                final_detections.push(class_dets[idx].clone());
            }
        }
    }

    Ok(final_detections)
}

/// Calculate detection metrics
fn calculate_metrics(
    predictions: &[Detection],
    ground_truth: &[Detection],
    iou_threshold: f32,
) -> DetectionMetrics {
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    let mut matched_gt = vec![false; ground_truth.len()];

    // Match predictions to ground truth
    for pred in predictions {
        let mut best_iou = 0.0;
        let mut best_match = None;

        for (gt_idx, gt) in ground_truth.iter().enumerate() {
            if matched_gt[gt_idx] || pred.class_id != gt.class_id {
                continue;
            }

            let iou = calculate_bbox_iou(&pred.bbox, &gt.bbox);
            if iou > best_iou {
                best_iou = iou;
                best_match = Some(gt_idx);
            }
        }

        if let Some(idx) = best_match {
            if best_iou >= iou_threshold {
                true_positives += 1;
                matched_gt[idx] = true;
            } else {
                false_positives += 1;
            }
        } else {
            false_positives += 1;
        }
    }

    false_negatives = matched_gt.iter().filter(|&&m| !m).count();

    let precision = if true_positives + false_positives > 0 {
        true_positives as f32 / (true_positives + false_positives) as f32
    } else {
        0.0
    };

    let recall = if true_positives + false_negatives > 0 {
        true_positives as f32 / (true_positives + false_negatives) as f32
    } else {
        0.0
    };

    let f1_score = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    DetectionMetrics {
        precision,
        recall,
        f1_score,
        true_positives,
        false_positives,
        false_negatives,
    }
}

/// Calculate IoU between two bounding boxes
fn calculate_bbox_iou(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    let x1 = box1.x1.max(box2.x1);
    let y1 = box1.y1.max(box2.y1);
    let x2 = box1.x2.min(box2.x2);
    let y2 = box1.y2.min(box2.y2);

    let intersection = ((x2 - x1).max(0.0)) * ((y2 - y1).max(0.0));

    let area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    let area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

    let union = area1 + area2 - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

#[derive(Debug, Clone)]
struct DetectionMetrics {
    precision: f32,
    recall: f32,
    f1_score: f32,
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
}

/// Demonstrate anchor generation
fn demonstrate_anchor_generation() -> Result<()> {
    println!("\n⚓ Anchor Generation");
    println!("════════════════════\n");

    let feature_sizes = vec![(80, 80), (40, 40), (20, 20)];
    let scales = vec![32.0, 64.0, 128.0];

    for (idx, (&feat_size, &scale)) in feature_sizes.iter().zip(scales.iter()).enumerate() {
        let config = AnchorConfig {
            base_size: scale,
            aspect_ratios: vec![0.5, 1.0, 2.0],
            scales: vec![1.0], // Scale multipliers
            stride: 8.0 * (2_f32.powi(idx as i32)),
        };

        let anchors = generate_anchors(feat_size.0, feat_size.1, config)?;

        println!("Level {}: Feature size {:?}", idx, feat_size);
        println!("  Scale: {}", scale);
        println!("  Number of anchors: {}\n", anchors.len());
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("🎯 ToRSh Vision - Object Detection Example");
    println!("===========================================\n");

    let config = DetectionConfig::default();
    let _device = Arc::new(CpuDevice::new());

    println!("📊 Detection Configuration:");
    println!("  Confidence threshold: {}", config.confidence_threshold);
    println!("  NMS threshold: {}", config.nms_threshold);
    println!("  Max detections: {}", config.max_detections);
    println!("  Input size: {:?}\n", config.input_size);

    // Demonstrate anchor generation
    demonstrate_anchor_generation()?;

    // Create sample image
    println!("📸 Creating sample image...");
    let sample_image: Tensor<f32> = creation::randn(&[3, 640, 640])?;
    println!("  Image shape: {:?}\n", sample_image.shape());

    // Demonstrate different detection models
    println!("🏗️  Detection Models:");
    println!("════════════════════\n");

    println!("1️⃣  YOLOv5 (Single-Stage Detector):");
    println!("  - Fast inference speed");
    println!("  - Good balance of speed and accuracy");
    println!("  - Suitable for real-time applications\n");

    let _yolo = yolo_v5_small(80)?;
    println!("  ✓ YOLOv5-small created\n");

    println!("2️⃣  RetinaNet (FPN-based Detector):");
    println!("  - Feature Pyramid Network");
    println!("  - Focal loss for class imbalance");
    println!("  - Better for small objects\n");

    let _retinanet = retina_net_resnet50(80)?;
    println!("  ✓ RetinaNet-ResNet50 created\n");

    println!("3️⃣  SSD (Multi-Scale Detector):");
    println!("  - Multiple detection scales");
    println!("  - VGG backbone");
    println!("  - Efficient inference\n");

    let _ssd = ssd_300(80)?;
    println!("  ✓ SSD-300 created\n");

    // Demonstrate NMS
    println!("🔍 Non-Maximum Suppression Demo:");
    println!("══════════════════════════════════\n");

    // Create sample detections (BoundingBox is [x1, y1, x2, y2])
    let sample_detections = vec![
        Detection::new([100.0, 100.0, 200.0, 200.0], 0.9, 1),
        Detection::new([110.0, 110.0, 210.0, 210.0], 0.8, 1), // Overlaps with first box
        Detection::new([300.0, 300.0, 400.0, 400.0], 0.95, 1), // Separate box
    ];

    let nms_config = NMSConfig {
        iou_threshold: 0.5,
        confidence_threshold: 0.5,
        max_detections: Some(100),
        per_class: true, // Apply NMS per class
    };

    let kept_detections = nms(sample_detections.clone(), nms_config)?;
    println!("  Input boxes: 3");
    println!("  Kept after NMS: {}", kept_detections.len());
    println!(
        "  Removed overlapping boxes: {}\n",
        3 - kept_detections.len()
    );

    // Best practices
    println!("📚 Object Detection Best Practices:");
    println!("════════════════════════════════════\n");

    println!("1. Model Selection:");
    println!("   - YOLOv5: Real-time applications, embedded systems");
    println!("   - RetinaNet: Better accuracy, small object detection");
    println!("   - SSD: Balance of speed and accuracy\n");

    println!("2. Data Preparation:");
    println!("   - Augmentation: Flip, crop, color jitter");
    println!("   - Scale variation: Multi-scale training");
    println!("   - Mosaic augmentation for better small object detection\n");

    println!("3. Training Tips:");
    println!("   - Warm-up learning rate schedule");
    println!("   - Focal loss for class imbalance");
    println!("   - Multi-scale training\n");

    println!("4. Post-Processing:");
    println!("   - Confidence threshold: 0.3-0.5");
    println!("   - NMS threshold: 0.4-0.5");
    println!("   - Per-class NMS for better results\n");

    println!("5. Evaluation:");
    println!("   - mAP (mean Average Precision) @ IoU 0.5");
    println!("   - mAP @ IoU 0.5:0.95 for stricter evaluation");
    println!("   - Consider both precision and recall\n");

    println!("✅ Example completed successfully!");
    println!("\nNext steps:");
    println!("  - Train on your custom dataset (VOC or COCO format)");
    println!("  - Fine-tune hyperparameters");
    println!("  - Experiment with different architectures");
    println!("  - Optimize inference speed for deployment\n");

    Ok(())
}
