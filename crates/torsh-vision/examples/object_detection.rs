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

use std::sync::Arc;
use torsh_core::device::CpuDevice;
use torsh_tensor::{creation, Tensor};
use torsh_vision::{
    calculate_iou, generate_anchors, nms, retina_net_resnet50, ssd_300, yolo_v5_small, Result,
};
// Use types from ops::detection module
use torsh_vision::ops::detection::{AnchorConfig, BoundingBox, Detection, NMSConfig};

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

/// Calculate detection metrics
fn calculate_metrics(
    predictions: &[Detection],
    ground_truth: &[Detection],
    iou_threshold: f32,
) -> DetectionMetrics {
    let mut true_positives = 0;
    let mut false_positives = 0;

    let mut matched_gt = vec![false; ground_truth.len()];

    // Match predictions to ground truth
    for pred in predictions {
        let mut best_iou = 0.0f32;
        let mut best_match = None;

        for (gt_idx, gt) in ground_truth.iter().enumerate() {
            if matched_gt[gt_idx] || pred.class_id != gt.class_id {
                continue;
            }

            let iou = calculate_iou(&pred.bbox, &gt.bbox);
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

    let false_negatives = matched_gt.iter().filter(|&&m| !m).count();

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

#[derive(Debug, Clone)]
struct DetectionMetrics {
    precision: f32,
    recall: f32,
    f1_score: f32,
    #[allow(dead_code)]
    true_positives: usize,
    #[allow(dead_code)]
    false_positives: usize,
    #[allow(dead_code)]
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

    // Create sample detections using the correct API (bbox is [f32; 4])
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

    // Demonstrate IoU calculation
    println!("📐 IoU Calculation Demo:");
    println!("════════════════════════\n");

    let box1: BoundingBox = [100.0, 100.0, 200.0, 200.0];
    let box2: BoundingBox = [150.0, 150.0, 250.0, 250.0];
    let iou = calculate_iou(&box1, &box2);
    println!("  Box 1: {:?}", box1);
    println!("  Box 2: {:?}", box2);
    println!("  IoU: {:.4}\n", iou);

    // Demonstrate metrics calculation
    println!("📊 Detection Metrics Demo:");
    println!("═══════════════════════════\n");

    let predictions = vec![
        Detection::new([100.0, 100.0, 200.0, 200.0], 0.9, 0),
        Detection::new([300.0, 300.0, 400.0, 400.0], 0.85, 1),
    ];

    let ground_truth = vec![
        Detection::new([105.0, 105.0, 195.0, 195.0], 1.0, 0),
        Detection::new([310.0, 310.0, 390.0, 390.0], 1.0, 1),
    ];

    let metrics = calculate_metrics(&predictions, &ground_truth, 0.5);
    println!("  Precision: {:.4}", metrics.precision);
    println!("  Recall: {:.4}", metrics.recall);
    println!("  F1 Score: {:.4}\n", metrics.f1_score);

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
