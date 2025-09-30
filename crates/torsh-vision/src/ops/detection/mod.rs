//! Detection operations for object detection and localization
//!
//! This module provides comprehensive utilities for object detection tasks including:
//! - Non-Maximum Suppression (NMS) for duplicate detection removal
//! - Region of Interest (ROI) pooling for feature extraction
//! - Bounding box operations and transformations
//! - Intersection over Union (IoU) calculations
//! - Anchor generation and processing utilities
//! - Post-processing functions for detection models

use crate::ops::common::{utils, InterpolationMode};
use crate::{Result, VisionError};
use torsh_tensor::creation::{full, ones, zeros};
use torsh_tensor::Tensor;

/// Bounding box representation [x1, y1, x2, y2]
pub type BoundingBox = [f32; 4];

/// Detection result with bounding box, confidence, and class
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box coordinates [x1, y1, x2, y2]
    pub bbox: BoundingBox,
    /// Confidence score [0.0, 1.0]
    pub confidence: f32,
    /// Class index
    pub class_id: usize,
    /// Optional class label
    pub class_label: Option<String>,
}

impl Detection {
    /// Create a new detection
    pub fn new(bbox: BoundingBox, confidence: f32, class_id: usize) -> Self {
        Self {
            bbox,
            confidence,
            class_id,
            class_label: None,
        }
    }

    /// Create detection with class label
    pub fn with_label(bbox: BoundingBox, confidence: f32, class_id: usize, label: String) -> Self {
        Self {
            bbox,
            confidence,
            class_id,
            class_label: Some(label),
        }
    }

    /// Get bounding box area
    pub fn area(&self) -> f32 {
        let [x1, y1, x2, y2] = self.bbox;
        (x2 - x1) * (y2 - y1)
    }

    /// Get bounding box center
    pub fn center(&self) -> (f32, f32) {
        let [x1, y1, x2, y2] = self.bbox;
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    }

    /// Get bounding box width and height
    pub fn size(&self) -> (f32, f32) {
        let [x1, y1, x2, y2] = self.bbox;
        (x2 - x1, y2 - y1)
    }
}

/// Bounding box format for different coordinate systems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BBoxFormat {
    /// [x1, y1, x2, y2] - Top-left and bottom-right corners
    XYXY,
    /// [x_center, y_center, width, height] - Center and dimensions
    XYWH,
    /// [x1, y1, width, height] - Top-left corner and dimensions
    LTWH,
}

/// NMS (Non-Maximum Suppression) configuration
#[derive(Debug, Clone)]
pub struct NMSConfig {
    /// IoU threshold for suppression
    pub iou_threshold: f32,
    /// Confidence threshold for keeping detections
    pub confidence_threshold: f32,
    /// Maximum number of detections to keep
    pub max_detections: Option<usize>,
    /// Whether to apply NMS per class separately
    pub per_class: bool,
}

impl Default for NMSConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.5,
            confidence_threshold: 0.5,
            max_detections: None,
            per_class: true,
        }
    }
}

impl NMSConfig {
    /// Create NMS config with custom thresholds
    pub fn new(iou_threshold: f32, confidence_threshold: f32) -> Self {
        Self {
            iou_threshold,
            confidence_threshold,
            ..Default::default()
        }
    }

    /// Set maximum number of detections
    pub fn with_max_detections(mut self, max_detections: usize) -> Self {
        self.max_detections = Some(max_detections);
        self
    }

    /// Set per-class NMS behavior
    pub fn with_per_class(mut self, per_class: bool) -> Self {
        self.per_class = per_class;
        self
    }
}

/// ROI (Region of Interest) pooling configuration
#[derive(Debug, Clone)]
pub struct ROIPoolConfig {
    /// Output size (height, width)
    pub output_size: (usize, usize),
    /// Spatial scale factor
    pub spatial_scale: f32,
    /// Interpolation mode for resizing
    pub interpolation: InterpolationMode,
}

impl Default for ROIPoolConfig {
    fn default() -> Self {
        Self {
            output_size: (7, 7),
            spatial_scale: 1.0,
            interpolation: InterpolationMode::Bilinear,
        }
    }
}

impl ROIPoolConfig {
    /// Create ROI pool config with output size
    pub fn new(output_size: (usize, usize), spatial_scale: f32) -> Self {
        Self {
            output_size,
            spatial_scale,
            ..Default::default()
        }
    }
}

/// Anchor generation configuration
#[derive(Debug, Clone)]
pub struct AnchorConfig {
    /// Base anchor size
    pub base_size: f32,
    /// Aspect ratios for anchors
    pub aspect_ratios: Vec<f32>,
    /// Scale factors for anchors
    pub scales: Vec<f32>,
    /// Anchor stride
    pub stride: f32,
}

impl Default for AnchorConfig {
    fn default() -> Self {
        Self {
            base_size: 16.0,
            aspect_ratios: vec![0.5, 1.0, 2.0],
            scales: vec![1.0, 1.26, 1.59], // 2^(0/3), 2^(1/3), 2^(2/3)
            stride: 16.0,
        }
    }
}

impl AnchorConfig {
    /// Create anchor config with custom parameters
    pub fn new(base_size: f32, aspect_ratios: Vec<f32>, scales: Vec<f32>, stride: f32) -> Self {
        Self {
            base_size,
            aspect_ratios,
            scales,
            stride,
        }
    }
}

/// Apply Non-Maximum Suppression to filter overlapping detections
pub fn nms(detections: Vec<Detection>, config: NMSConfig) -> Result<Vec<Detection>> {
    if detections.is_empty() {
        return Ok(Vec::new());
    }

    // Filter by confidence threshold
    let mut filtered: Vec<Detection> = detections
        .into_iter()
        .filter(|det| det.confidence >= config.confidence_threshold)
        .collect();

    if filtered.is_empty() {
        return Ok(Vec::new());
    }

    // Sort by confidence (descending)
    filtered.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    if config.per_class {
        apply_per_class_nms(filtered, &config)
    } else {
        apply_global_nms(filtered, &config)
    }
}

/// Calculate Intersection over Union (IoU) between two bounding boxes
pub fn calculate_iou(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    utils::calculate_box_iou(box1, box2)
}

/// Calculate IoU between multiple boxes efficiently
pub fn calculate_iou_matrix(boxes: &[BoundingBox]) -> Result<Tensor<f32>> {
    let n = boxes.len();
    let mut iou_matrix = zeros(&[n, n])?;

    for i in 0..n {
        for j in 0..n {
            let iou = if i == j {
                1.0
            } else {
                calculate_iou(&boxes[i], &boxes[j])
            };
            iou_matrix.set(&[i, j], iou.into())?;
        }
    }

    Ok(iou_matrix)
}

/// Convert bounding box between different formats
pub fn convert_bbox_format(bbox: &BoundingBox, from: BBoxFormat, to: BBoxFormat) -> BoundingBox {
    if from == to {
        return *bbox;
    }

    match (from, to) {
        (BBoxFormat::XYXY, BBoxFormat::XYWH) => xyxy_to_xywh(bbox),
        (BBoxFormat::XYXY, BBoxFormat::LTWH) => xyxy_to_ltwh(bbox),
        (BBoxFormat::XYWH, BBoxFormat::XYXY) => xywh_to_xyxy(bbox),
        (BBoxFormat::XYWH, BBoxFormat::LTWH) => xywh_to_ltwh(bbox),
        (BBoxFormat::LTWH, BBoxFormat::XYXY) => ltwh_to_xyxy(bbox),
        (BBoxFormat::LTWH, BBoxFormat::XYWH) => ltwh_to_xywh(bbox),
        // Same format cases should be handled by early return above
        _ => *bbox,
    }
}

/// Scale bounding box coordinates
pub fn scale_bbox(bbox: &BoundingBox, scale_x: f32, scale_y: f32) -> BoundingBox {
    let [x1, y1, x2, y2] = *bbox;
    [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
}

/// Clip bounding box to image boundaries
pub fn clip_bbox(bbox: &BoundingBox, image_width: f32, image_height: f32) -> BoundingBox {
    let [x1, y1, x2, y2] = *bbox;
    [
        x1.max(0.0).min(image_width),
        y1.max(0.0).min(image_height),
        x2.max(0.0).min(image_width),
        y2.max(0.0).min(image_height),
    ]
}

/// Apply ROI (Region of Interest) pooling
pub fn roi_pool(
    features: &Tensor<f32>,
    rois: &[BoundingBox],
    config: ROIPoolConfig,
) -> Result<Tensor<f32>> {
    let feature_shape = features.shape();
    let feature_dims = feature_shape.dims();

    if feature_dims.len() != 4 {
        return Err(VisionError::InvalidShape(
            "Features must be 4D tensor (N, C, H, W)".to_string(),
        ));
    }

    let (_batch_size, channels, feature_height, feature_width) = (
        feature_dims[0],
        feature_dims[1],
        feature_dims[2],
        feature_dims[3],
    );

    let num_rois = rois.len();
    let (pool_height, pool_width) = config.output_size;

    let mut pooled_features = zeros(&[num_rois, channels, pool_height, pool_width])?;

    for (roi_idx, roi) in rois.iter().enumerate() {
        let scaled_roi = scale_bbox(roi, config.spatial_scale, config.spatial_scale);
        let clipped_roi = clip_bbox(&scaled_roi, feature_width as f32, feature_height as f32);

        let [x1, y1, x2, y2] = clipped_roi;

        // Compute roi dimensions
        let roi_width = (x2 - x1).max(1.0);
        let roi_height = (y2 - y1).max(1.0);

        // Compute bin size
        let bin_size_h = roi_height / pool_height as f32;
        let bin_size_w = roi_width / pool_width as f32;

        for c in 0..channels {
            for ph in 0..pool_height {
                for pw in 0..pool_width {
                    let h_start = y1 + ph as f32 * bin_size_h;
                    let w_start = x1 + pw as f32 * bin_size_w;
                    let h_end = h_start + bin_size_h;
                    let w_end = w_start + bin_size_w;

                    let pooled_val = roi_pool_single_bin(
                        features,
                        c,
                        h_start,
                        h_end,
                        w_start,
                        w_end,
                        feature_height,
                        feature_width,
                    )?;

                    pooled_features.set(&[roi_idx, c, ph, pw], pooled_val.into())?;
                }
            }
        }
    }

    Ok(pooled_features)
}

/// Generate anchor boxes for object detection
pub fn generate_anchors(
    feature_height: usize,
    feature_width: usize,
    config: AnchorConfig,
) -> Result<Vec<BoundingBox>> {
    let mut anchors = Vec::new();

    for y in 0..feature_height {
        for x in 0..feature_width {
            let center_x = (x as f32 + 0.5) * config.stride;
            let center_y = (y as f32 + 0.5) * config.stride;

            for &scale in &config.scales {
                for &aspect_ratio in &config.aspect_ratios {
                    let anchor_size = config.base_size * scale;
                    let anchor_width = anchor_size * aspect_ratio.sqrt();
                    let anchor_height = anchor_size / aspect_ratio.sqrt();

                    let x1 = center_x - anchor_width / 2.0;
                    let y1 = center_y - anchor_height / 2.0;
                    let x2 = center_x + anchor_width / 2.0;
                    let y2 = center_y + anchor_height / 2.0;

                    anchors.push([x1, y1, x2, y2]);
                }
            }
        }
    }

    Ok(anchors)
}

/// Apply bounding box regression deltas to anchors
pub fn apply_bbox_deltas(
    anchors: &[BoundingBox],
    deltas: &Tensor<f32>,
) -> Result<Vec<BoundingBox>> {
    let delta_shape = deltas.shape();
    let delta_dims = delta_shape.dims();

    if delta_dims.len() != 2 || delta_dims[1] != 4 {
        return Err(VisionError::InvalidShape(
            "Deltas must be Nx4 tensor".to_string(),
        ));
    }

    if delta_dims[0] != anchors.len() {
        return Err(VisionError::InvalidArgument(
            "Number of deltas must match number of anchors".to_string(),
        ));
    }

    let mut regressed_boxes = Vec::with_capacity(anchors.len());

    for (i, anchor) in anchors.iter().enumerate() {
        let [xa, ya, xa2, ya2] = *anchor;
        let wa = xa2 - xa;
        let ha = ya2 - ya;
        let cxa = xa + wa / 2.0;
        let cya = ya + ha / 2.0;

        let dx: f32 = deltas.get(&[i, 0])?.clone().into();
        let dy: f32 = deltas.get(&[i, 1])?.clone().into();
        let dw: f32 = deltas.get(&[i, 2])?.clone().into();
        let dh: f32 = deltas.get(&[i, 3])?.clone().into();

        let cxp = dx * wa + cxa;
        let cyp = dy * ha + cya;
        let wp = dw.exp() * wa;
        let hp = dh.exp() * ha;

        let x1 = cxp - wp / 2.0;
        let y1 = cyp - hp / 2.0;
        let x2 = cxp + wp / 2.0;
        let y2 = cyp + hp / 2.0;

        regressed_boxes.push([x1, y1, x2, y2]);
    }

    Ok(regressed_boxes)
}

/// Filter boxes by size constraints
pub fn filter_boxes_by_size(
    boxes: Vec<BoundingBox>,
    min_size: f32,
    max_size: Option<f32>,
) -> Vec<BoundingBox> {
    boxes
        .into_iter()
        .filter(|bbox| {
            let [x1, y1, x2, y2] = *bbox;
            let width = x2 - x1;
            let height = y2 - y1;
            let size = width.min(height);

            size >= min_size && max_size.map_or(true, |max| size <= max)
        })
        .collect()
}

/// Compute bounding box regression targets
pub fn compute_bbox_targets(
    anchors: &[BoundingBox],
    ground_truth: &[BoundingBox],
    iou_threshold: f32,
) -> Result<(Vec<i32>, Tensor<f32>)> {
    let num_anchors = anchors.len();
    let mut labels = vec![-1i32; num_anchors]; // -1: ignore, 0: negative, 1: positive
    let mut bbox_targets = zeros(&[num_anchors, 4])?;

    if ground_truth.is_empty() {
        // All anchors are negative when no ground truth
        for label in &mut labels {
            *label = 0;
        }
        return Ok((labels, bbox_targets));
    }

    // Compute IoU matrix between anchors and ground truth
    let mut max_iou_per_anchor = vec![0.0f32; num_anchors];
    let mut best_gt_per_anchor = vec![0usize; num_anchors];

    for (anchor_idx, anchor) in anchors.iter().enumerate() {
        for (gt_idx, gt_box) in ground_truth.iter().enumerate() {
            let iou = calculate_iou(anchor, gt_box);
            if iou > max_iou_per_anchor[anchor_idx] {
                max_iou_per_anchor[anchor_idx] = iou;
                best_gt_per_anchor[anchor_idx] = gt_idx;
            }
        }
    }

    // Assign labels based on IoU
    for (anchor_idx, &max_iou) in max_iou_per_anchor.iter().enumerate() {
        if max_iou >= iou_threshold {
            labels[anchor_idx] = 1; // Positive
            let gt_idx = best_gt_per_anchor[anchor_idx];
            let targets = compute_single_bbox_target(&anchors[anchor_idx], &ground_truth[gt_idx]);

            bbox_targets.set(&[anchor_idx, 0], targets[0].into())?;
            bbox_targets.set(&[anchor_idx, 1], targets[1].into())?;
            bbox_targets.set(&[anchor_idx, 2], targets[2].into())?;
            bbox_targets.set(&[anchor_idx, 3], targets[3].into())?;
        } else if max_iou < 0.3 {
            labels[anchor_idx] = 0; // Negative
        }
        // IoU between 0.3 and threshold remains as ignore (-1)
    }

    Ok((labels, bbox_targets))
}

// Internal helper functions

fn apply_per_class_nms(detections: Vec<Detection>, config: &NMSConfig) -> Result<Vec<Detection>> {
    let mut result = Vec::new();

    // Group detections by class
    let mut class_groups: std::collections::HashMap<usize, Vec<Detection>> =
        std::collections::HashMap::new();

    for detection in detections {
        class_groups
            .entry(detection.class_id)
            .or_default()
            .push(detection);
    }

    // Apply NMS per class
    for (_, mut class_detections) in class_groups {
        class_detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        let nms_result = apply_nms_to_group(class_detections, config)?;
        result.extend(nms_result);
    }

    // Sort final result by confidence and apply global max detections limit
    result.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    if let Some(max_dets) = config.max_detections {
        result.truncate(max_dets);
    }

    Ok(result)
}

fn apply_global_nms(detections: Vec<Detection>, config: &NMSConfig) -> Result<Vec<Detection>> {
    let result = apply_nms_to_group(detections, config)?;

    Ok(if let Some(max_dets) = config.max_detections {
        result.into_iter().take(max_dets).collect()
    } else {
        result
    })
}

fn apply_nms_to_group(
    mut detections: Vec<Detection>,
    config: &NMSConfig,
) -> Result<Vec<Detection>> {
    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }

        keep.push(detections[i].clone());

        // Suppress overlapping detections
        for j in (i + 1)..detections.len() {
            if !suppressed[j] {
                let iou = calculate_iou(&detections[i].bbox, &detections[j].bbox);
                if iou > config.iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    Ok(keep)
}

fn xyxy_to_xywh(bbox: &BoundingBox) -> BoundingBox {
    let [x1, y1, x2, y2] = *bbox;
    let cx = (x1 + x2) / 2.0;
    let cy = (y1 + y2) / 2.0;
    let w = x2 - x1;
    let h = y2 - y1;
    [cx, cy, w, h]
}

fn xyxy_to_ltwh(bbox: &BoundingBox) -> BoundingBox {
    let [x1, y1, x2, y2] = *bbox;
    let w = x2 - x1;
    let h = y2 - y1;
    [x1, y1, w, h]
}

fn xywh_to_xyxy(bbox: &BoundingBox) -> BoundingBox {
    let [cx, cy, w, h] = *bbox;
    let x1 = cx - w / 2.0;
    let y1 = cy - h / 2.0;
    let x2 = cx + w / 2.0;
    let y2 = cy + h / 2.0;
    [x1, y1, x2, y2]
}

fn xywh_to_ltwh(bbox: &BoundingBox) -> BoundingBox {
    let [cx, cy, w, h] = *bbox;
    let x1 = cx - w / 2.0;
    let y1 = cy - h / 2.0;
    [x1, y1, w, h]
}

fn ltwh_to_xyxy(bbox: &BoundingBox) -> BoundingBox {
    let [x1, y1, w, h] = *bbox;
    let x2 = x1 + w;
    let y2 = y1 + h;
    [x1, y1, x2, y2]
}

fn ltwh_to_xywh(bbox: &BoundingBox) -> BoundingBox {
    let [x1, y1, w, h] = *bbox;
    let cx = x1 + w / 2.0;
    let cy = y1 + h / 2.0;
    [cx, cy, w, h]
}

fn roi_pool_single_bin(
    features: &Tensor<f32>,
    channel: usize,
    h_start: f32,
    h_end: f32,
    w_start: f32,
    w_end: f32,
    feature_height: usize,
    feature_width: usize,
) -> Result<f32> {
    let h_start_int = h_start.floor() as usize;
    let h_end_int = (h_end.ceil() as usize).min(feature_height);
    let w_start_int = w_start.floor() as usize;
    let w_end_int = (w_end.ceil() as usize).min(feature_width);

    let mut max_val = f32::NEG_INFINITY;
    let mut found_any = false;

    for h in h_start_int..h_end_int {
        for w in w_start_int..w_end_int {
            if h < feature_height && w < feature_width {
                let val: f32 = features.get(&[0, channel, h, w])?.clone().into();
                max_val = max_val.max(val);
                found_any = true;
            }
        }
    }

    Ok(if found_any { max_val } else { 0.0 })
}

fn compute_single_bbox_target(anchor: &BoundingBox, ground_truth: &BoundingBox) -> [f32; 4] {
    let [xa, ya, xa2, ya2] = *anchor;
    let [xg, yg, xg2, yg2] = *ground_truth;

    let wa = xa2 - xa;
    let ha = ya2 - ya;
    let wg = xg2 - xg;
    let hg = yg2 - yg;

    let cxa = xa + wa / 2.0;
    let cya = ya + ha / 2.0;
    let cxg = xg + wg / 2.0;
    let cyg = yg + hg / 2.0;

    let dx = (cxg - cxa) / wa;
    let dy = (cyg - cya) / ha;
    let dw = (wg / wa).ln();
    let dh = (hg / ha).ln();

    [dx, dy, dw, dh]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_creation() {
        let detection = Detection::new([10.0, 20.0, 50.0, 80.0], 0.9, 1);
        assert_eq!(detection.bbox, [10.0, 20.0, 50.0, 80.0]);
        assert_eq!(detection.confidence, 0.9);
        assert_eq!(detection.class_id, 1);

        let area = detection.area();
        assert_eq!(area, 2400.0); // (50-10) * (80-20)

        let center = detection.center();
        assert_eq!(center, (30.0, 50.0));

        let size = detection.size();
        assert_eq!(size, (40.0, 60.0));
    }

    #[test]
    fn test_iou_calculation() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [5.0, 5.0, 15.0, 15.0];

        let iou = calculate_iou(&box1, &box2);
        assert!((iou - 0.142857).abs() < 1e-5); // 25 / 175

        let identical_boxes = [0.0, 0.0, 10.0, 10.0];
        let iou_identical = calculate_iou(&box1, &identical_boxes);
        assert!((iou_identical - 1.0).abs() < 1e-6);

        let non_overlapping = [20.0, 20.0, 30.0, 30.0];
        let iou_zero = calculate_iou(&box1, &non_overlapping);
        assert_eq!(iou_zero, 0.0);
    }

    #[test]
    fn test_bbox_format_conversion() {
        let xyxy_box = [10.0, 20.0, 30.0, 40.0];

        let xywh_box = convert_bbox_format(&xyxy_box, BBoxFormat::XYXY, BBoxFormat::XYWH);
        assert_eq!(xywh_box, [20.0, 30.0, 20.0, 20.0]); // center_x, center_y, width, height

        let ltwh_box = convert_bbox_format(&xyxy_box, BBoxFormat::XYXY, BBoxFormat::LTWH);
        assert_eq!(ltwh_box, [10.0, 20.0, 20.0, 20.0]); // left, top, width, height

        // Test round-trip conversion
        let back_to_xyxy = convert_bbox_format(&xywh_box, BBoxFormat::XYWH, BBoxFormat::XYXY);
        assert_eq!(back_to_xyxy, xyxy_box);
    }

    #[test]
    fn test_bbox_operations() {
        let bbox = [10.0, 20.0, 30.0, 40.0];

        // Test scaling
        let scaled = scale_bbox(&bbox, 2.0, 1.5);
        assert_eq!(scaled, [20.0, 30.0, 60.0, 60.0]);

        // Test clipping
        let large_bbox = [-5.0, -10.0, 105.0, 110.0];
        let clipped = clip_bbox(&large_bbox, 100.0, 100.0);
        assert_eq!(clipped, [0.0, 0.0, 100.0, 100.0]);
    }

    #[test]
    fn test_nms_basic() -> Result<()> {
        let detections = vec![
            Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            Detection::new([5.0, 5.0, 15.0, 15.0], 0.8, 0), // Overlapping
            Detection::new([20.0, 20.0, 30.0, 30.0], 0.7, 0), // Non-overlapping
        ];

        let config = NMSConfig::new(0.5, 0.5);
        let result = nms(detections, config)?;

        // Should keep all 3 detections since IoU between boxes 1 and 2 is only ~0.143 < 0.5 threshold
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].confidence, 0.9); // Highest confidence
        assert_eq!(result[1].confidence, 0.8); // Second highest
        assert_eq!(result[2].confidence, 0.7); // Lowest confidence

        Ok(())
    }

    #[test]
    fn test_anchor_generation() -> Result<()> {
        let config = AnchorConfig::new(16.0, vec![1.0], vec![1.0], 16.0);
        let anchors = generate_anchors(2, 2, config)?;

        // Should generate 4 anchors (2x2 grid)
        assert_eq!(anchors.len(), 4);

        // Check first anchor position (center at 8, 8 with size 16x16)
        let expected_first = [0.0, 0.0, 16.0, 16.0];
        assert_eq!(anchors[0], expected_first);

        Ok(())
    }

    #[test]
    fn test_bbox_filtering() {
        let boxes = vec![
            [0.0, 0.0, 5.0, 5.0],   // Small box (area = 25)
            [0.0, 0.0, 10.0, 10.0], // Medium box (area = 100)
            [0.0, 0.0, 20.0, 20.0], // Large box (area = 400)
        ];

        let filtered = filter_boxes_by_size(boxes, 8.0, Some(15.0));
        assert_eq!(filtered.len(), 1); // Only medium box should remain
        assert_eq!(filtered[0], [0.0, 0.0, 10.0, 10.0]);
    }

    #[test]
    fn test_nms_configs() {
        let config = NMSConfig::default();
        assert_eq!(config.iou_threshold, 0.5);
        assert_eq!(config.confidence_threshold, 0.5);
        assert!(config.per_class);

        let custom_config = NMSConfig::new(0.3, 0.7)
            .with_max_detections(100)
            .with_per_class(false);
        assert_eq!(custom_config.iou_threshold, 0.3);
        assert_eq!(custom_config.confidence_threshold, 0.7);
        assert_eq!(custom_config.max_detections, Some(100));
        assert!(!custom_config.per_class);
    }

    #[test]
    fn test_roi_pool_config() {
        let config = ROIPoolConfig::default();
        assert_eq!(config.output_size, (7, 7));
        assert_eq!(config.spatial_scale, 1.0);

        let custom_config = ROIPoolConfig::new((14, 14), 0.5);
        assert_eq!(custom_config.output_size, (14, 14));
        assert_eq!(custom_config.spatial_scale, 0.5);
    }

    #[test]
    fn test_anchor_config() {
        let config = AnchorConfig::default();
        assert_eq!(config.base_size, 16.0);
        assert_eq!(config.aspect_ratios, vec![0.5, 1.0, 2.0]);

        let custom_config = AnchorConfig::new(32.0, vec![1.0, 2.0], vec![1.0, 1.5], 32.0);
        assert_eq!(custom_config.base_size, 32.0);
        assert_eq!(custom_config.aspect_ratios, vec![1.0, 2.0]);
    }

    #[test]
    fn test_detection_with_label() {
        let detection =
            Detection::with_label([10.0, 20.0, 30.0, 40.0], 0.85, 2, "person".to_string());

        assert_eq!(detection.class_label, Some("person".to_string()));
        assert_eq!(detection.class_id, 2);
    }
}
