//! Analysis and loss operations for computer vision tasks
//!
//! This module provides comprehensive loss functions, evaluation metrics, and analysis
//! utilities for computer vision tasks including:
//! - Loss functions (cross-entropy, focal loss, dice loss, IoU loss, etc.)
//! - Classification metrics (accuracy, precision, recall, F1-score)
//! - Detection metrics (mAP, IoU, precision-recall curves)
//! - Segmentation metrics (IoU, Dice coefficient, pixel accuracy)
//! - Statistical analysis and visualization utilities

use crate::ops::common::utils;
use crate::ops::detection::{calculate_iou, BoundingBox, Detection};
use crate::{Result, VisionError};
use torsh_tensor::creation::{full, ones, zeros};
use torsh_tensor::Tensor;

/// Loss reduction strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// No reduction, return per-sample losses
    None,
    /// Mean reduction across all elements
    Mean,
    /// Sum reduction across all elements
    Sum,
    /// Mean reduction across batch dimension only
    BatchMean,
}

impl Default for Reduction {
    fn default() -> Self {
        Reduction::Mean
    }
}

/// Configuration for loss functions
#[derive(Debug, Clone)]
pub struct LossConfig {
    /// Reduction strategy
    pub reduction: Reduction,
    /// Label smoothing factor [0.0, 1.0]
    pub label_smoothing: f32,
    /// Class weights for imbalanced datasets
    pub class_weights: Option<Vec<f32>>,
    /// Ignore index for loss computation
    pub ignore_index: Option<usize>,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            reduction: Reduction::Mean,
            label_smoothing: 0.0,
            class_weights: None,
            ignore_index: None,
        }
    }
}

impl LossConfig {
    /// Create config with label smoothing
    pub fn with_label_smoothing(smoothing: f32) -> Self {
        Self {
            label_smoothing: smoothing,
            ..Default::default()
        }
    }

    /// Create config with class weights
    pub fn with_class_weights(weights: Vec<f32>) -> Self {
        Self {
            class_weights: Some(weights),
            ..Default::default()
        }
    }

    /// Set reduction strategy
    pub fn with_reduction(mut self, reduction: Reduction) -> Self {
        self.reduction = reduction;
        self
    }

    /// Set ignore index
    pub fn with_ignore_index(mut self, ignore_index: usize) -> Self {
        self.ignore_index = Some(ignore_index);
        self
    }
}

/// Focal loss configuration for addressing class imbalance
#[derive(Debug, Clone)]
pub struct FocalLossConfig {
    /// Alpha parameter for class balancing
    pub alpha: f32,
    /// Gamma parameter for focusing on hard examples
    pub gamma: f32,
    /// Base loss configuration
    pub base_config: LossConfig,
}

impl Default for FocalLossConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            gamma: 2.0,
            base_config: LossConfig::default(),
        }
    }
}

impl FocalLossConfig {
    /// Create focal loss config with custom parameters
    pub fn new(alpha: f32, gamma: f32) -> Self {
        Self {
            alpha,
            gamma,
            base_config: LossConfig::default(),
        }
    }
}

/// Dice loss configuration for segmentation tasks
#[derive(Debug, Clone)]
pub struct DiceLossConfig {
    /// Smoothing parameter to avoid division by zero
    pub smooth: f32,
    /// Base loss configuration
    pub base_config: LossConfig,
}

impl Default for DiceLossConfig {
    fn default() -> Self {
        Self {
            smooth: 1.0,
            base_config: LossConfig::default(),
        }
    }
}

/// Classification evaluation metrics
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    /// Overall accuracy
    pub accuracy: f32,
    /// Precision per class
    pub precision: Vec<f32>,
    /// Recall per class
    pub recall: Vec<f32>,
    /// F1-score per class
    pub f1_score: Vec<f32>,
    /// Macro-averaged precision
    pub macro_precision: f32,
    /// Macro-averaged recall
    pub macro_recall: f32,
    /// Macro-averaged F1-score
    pub macro_f1: f32,
    /// Weighted-averaged precision
    pub weighted_precision: f32,
    /// Weighted-averaged recall
    pub weighted_recall: f32,
    /// Weighted-averaged F1-score
    pub weighted_f1: f32,
}

/// Detection evaluation metrics
#[derive(Debug, Clone)]
pub struct DetectionMetrics {
    /// Mean Average Precision (mAP)
    pub map: f32,
    /// Average Precision per class
    pub ap_per_class: Vec<f32>,
    /// Precision at different IoU thresholds
    pub precision_at_iou: Vec<f32>,
    /// Recall at different IoU thresholds
    pub recall_at_iou: Vec<f32>,
    /// F1-score at different IoU thresholds
    pub f1_at_iou: Vec<f32>,
}

/// Segmentation evaluation metrics
#[derive(Debug, Clone)]
pub struct SegmentationMetrics {
    /// Pixel accuracy
    pub pixel_accuracy: f32,
    /// Mean pixel accuracy per class
    pub mean_pixel_accuracy: f32,
    /// Mean Intersection over Union (mIoU)
    pub mean_iou: f32,
    /// IoU per class
    pub iou_per_class: Vec<f32>,
    /// Dice coefficient per class
    pub dice_per_class: Vec<f32>,
    /// Mean Dice coefficient
    pub mean_dice: f32,
    /// Frequency weighted IoU
    pub frequency_weighted_iou: f32,
}

/// Cross-entropy loss for classification
pub fn cross_entropy_loss(
    predictions: &Tensor<f32>,
    targets: &Tensor<f32>,
    config: LossConfig,
) -> Result<Tensor<f32>> {
    let pred_shape = predictions.shape();
    let pred_dims = pred_shape.dims();
    let target_shape = targets.shape();
    let target_dims = target_shape.dims();

    if pred_dims.len() != 2 {
        return Err(VisionError::InvalidShape(
            "Predictions must be 2D tensor (N, C)".to_string(),
        ));
    }

    if target_dims.len() != 1 {
        return Err(VisionError::InvalidShape(
            "Targets must be 1D tensor (N,)".to_string(),
        ));
    }

    let (batch_size, num_classes) = (pred_dims[0], pred_dims[1]);

    if target_dims[0] != batch_size {
        return Err(VisionError::InvalidShape(
            "Predictions and targets must have same batch size".to_string(),
        ));
    }

    // Apply log softmax to predictions
    let log_probs = log_softmax(predictions)?;

    // Compute per-sample losses
    let mut losses = zeros(&[batch_size])?;

    for i in 0..batch_size {
        let target_class: f32 = targets.get(&[i])?.clone().into();
        let target_class = target_class as usize;

        // Skip ignored indices
        if let Some(ignore_idx) = config.ignore_index {
            if target_class == ignore_idx {
                continue;
            }
        }

        if target_class >= num_classes {
            return Err(VisionError::InvalidArgument(format!(
                "Target class {} exceeds number of classes {}",
                target_class, num_classes
            )));
        }

        let mut loss: f32 = log_probs.get(&[i, target_class])?.clone().into();
        loss = -loss; // Negative log likelihood

        // Apply label smoothing
        if config.label_smoothing > 0.0 {
            let smooth_loss = apply_label_smoothing(
                &log_probs,
                i,
                target_class,
                num_classes,
                config.label_smoothing,
            )?;
            loss = (1.0 - config.label_smoothing) * loss + config.label_smoothing * smooth_loss;
        }

        // Apply class weights
        if let Some(ref weights) = config.class_weights {
            if target_class < weights.len() {
                loss *= weights[target_class];
            }
        }

        losses.set(&[i], loss.into())?;
    }

    apply_reduction(&losses, config.reduction)
}

/// Focal loss for addressing class imbalance
pub fn focal_loss(
    predictions: &Tensor<f32>,
    targets: &Tensor<f32>,
    config: FocalLossConfig,
) -> Result<Tensor<f32>> {
    let pred_shape = predictions.shape();
    let pred_dims = pred_shape.dims();
    let (batch_size, num_classes) = (pred_dims[0], pred_dims[1]);

    // Apply softmax to get probabilities
    let probs = softmax(predictions)?;
    let log_probs = log_softmax(predictions)?;

    let mut losses = zeros(&[batch_size])?;

    for i in 0..batch_size {
        let target_class: f32 = targets.get(&[i])?.clone().into();
        let target_class = target_class as usize;

        let prob: f32 = probs.get(&[i, target_class])?.clone().into();
        let log_prob: f32 = log_probs.get(&[i, target_class])?.clone().into();

        // Focal loss formula: -alpha * (1 - p)^gamma * log(p)
        let focal_weight = config.alpha * (1.0 - prob).powf(config.gamma);
        let loss = -focal_weight * log_prob;

        losses.set(&[i], loss.into())?;
    }

    apply_reduction(&losses, config.base_config.reduction)
}

/// Dice loss for segmentation tasks
pub fn dice_loss(
    predictions: &Tensor<f32>,
    targets: &Tensor<f32>,
    config: DiceLossConfig,
) -> Result<Tensor<f32>> {
    let pred_shape = predictions.shape();
    let pred_dims = pred_shape.dims();
    let target_shape = targets.shape();
    let target_dims = target_shape.dims();

    if pred_dims != target_dims {
        return Err(VisionError::InvalidShape(
            "Predictions and targets must have same shape for Dice loss".to_string(),
        ));
    }

    // Apply sigmoid to predictions
    let pred_probs = sigmoid(predictions)?;

    // Compute Dice coefficient
    let intersection = compute_intersection(&pred_probs, targets)?;
    let pred_sum = compute_tensor_sum(&pred_probs)?;
    let target_sum = compute_tensor_sum(targets)?;

    let dice_coeff = (2.0 * intersection + config.smooth) / (pred_sum + target_sum + config.smooth);
    let dice_loss = 1.0 - dice_coeff;

    let loss_tensor = full(&[1], dice_loss)?;
    apply_reduction(&loss_tensor, config.base_config.reduction)
}

/// IoU (Intersection over Union) loss for segmentation
pub fn iou_loss(predictions: &Tensor<f32>, targets: &Tensor<f32>) -> Result<Tensor<f32>> {
    let pred_probs = sigmoid(predictions)?;

    let intersection = compute_intersection(&pred_probs, targets)?;
    let union = compute_union(&pred_probs, targets)?;

    let iou = intersection / (union + 1e-8);
    let loss = 1.0 - iou;

    Ok(full(&[1], loss)?)
}

/// Compute classification metrics
pub fn compute_classification_metrics(
    predictions: &Tensor<f32>,
    targets: &Tensor<f32>,
    num_classes: usize,
) -> Result<ClassificationMetrics> {
    let pred_shape = predictions.shape();
    let pred_dims = pred_shape.dims();
    let target_shape = targets.shape();
    let target_dims = target_shape.dims();

    if pred_dims.len() != 2 || target_dims.len() != 1 {
        return Err(VisionError::InvalidShape(
            "Invalid shape for classification metrics".to_string(),
        ));
    }

    let batch_size = pred_dims[0];

    // Get predicted classes
    let predicted_classes = get_predicted_classes(predictions)?;

    // Compute confusion matrix
    let mut confusion_matrix = vec![vec![0usize; num_classes]; num_classes];

    for i in 0..batch_size {
        let pred_class: f32 = predicted_classes.get(&[i])?.clone().into();
        let true_class: f32 = targets.get(&[i])?.clone().into();

        let pred_idx = pred_class as usize;
        let true_idx = true_class as usize;

        if pred_idx < num_classes && true_idx < num_classes {
            confusion_matrix[true_idx][pred_idx] += 1;
        }
    }

    // Compute metrics from confusion matrix
    let mut precision = vec![0.0; num_classes];
    let mut recall = vec![0.0; num_classes];
    let mut f1_score = vec![0.0; num_classes];
    let mut class_counts = vec![0usize; num_classes];

    let mut total_correct = 0;
    let mut total_samples = 0;

    for i in 0..num_classes {
        let tp = confusion_matrix[i][i] as f32;
        let fp: f32 = (0..num_classes)
            .map(|j| if j != i { confusion_matrix[j][i] } else { 0 })
            .sum::<usize>() as f32;
        let fn_count: f32 = (0..num_classes)
            .map(|j| if j != i { confusion_matrix[i][j] } else { 0 })
            .sum::<usize>() as f32;

        class_counts[i] = (tp + fn_count) as usize;
        total_correct += tp as usize;
        total_samples += class_counts[i];

        precision[i] = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        recall[i] = if tp + fn_count > 0.0 {
            tp / (tp + fn_count)
        } else {
            0.0
        };
        f1_score[i] = if precision[i] + recall[i] > 0.0 {
            2.0 * precision[i] * recall[i] / (precision[i] + recall[i])
        } else {
            0.0
        };
    }

    let accuracy = total_correct as f32 / total_samples as f32;

    // Compute macro averages
    let macro_precision = precision.iter().sum::<f32>() / num_classes as f32;
    let macro_recall = recall.iter().sum::<f32>() / num_classes as f32;
    let macro_f1 = f1_score.iter().sum::<f32>() / num_classes as f32;

    // Compute weighted averages
    let total_weight: f32 = class_counts.iter().sum::<usize>() as f32;
    let weighted_precision = precision
        .iter()
        .zip(class_counts.iter())
        .map(|(&p, &c)| p * c as f32)
        .sum::<f32>()
        / total_weight;
    let weighted_recall = recall
        .iter()
        .zip(class_counts.iter())
        .map(|(&r, &c)| r * c as f32)
        .sum::<f32>()
        / total_weight;
    let weighted_f1 = f1_score
        .iter()
        .zip(class_counts.iter())
        .map(|(&f, &c)| f * c as f32)
        .sum::<f32>()
        / total_weight;

    Ok(ClassificationMetrics {
        accuracy,
        precision,
        recall,
        f1_score,
        macro_precision,
        macro_recall,
        macro_f1,
        weighted_precision,
        weighted_recall,
        weighted_f1,
    })
}

/// Compute detection metrics (mAP)
pub fn compute_detection_metrics(
    predictions: &[Detection],
    ground_truth: &[Detection],
    iou_thresholds: &[f32],
    num_classes: usize,
) -> Result<DetectionMetrics> {
    let mut ap_per_class = vec![0.0; num_classes];
    let mut precision_at_iou = vec![0.0; iou_thresholds.len()];
    let mut recall_at_iou = vec![0.0; iou_thresholds.len()];
    let mut f1_at_iou = vec![0.0; iou_thresholds.len()];

    // Compute AP for each class
    for class_id in 0..num_classes {
        let class_predictions: Vec<&Detection> = predictions
            .iter()
            .filter(|det| det.class_id == class_id)
            .collect();

        let class_ground_truth: Vec<&Detection> = ground_truth
            .iter()
            .filter(|det| det.class_id == class_id)
            .collect();

        if !class_ground_truth.is_empty() {
            ap_per_class[class_id] =
                compute_average_precision(&class_predictions, &class_ground_truth, 0.5)?;
        }
    }

    // Compute metrics at different IoU thresholds
    for (i, &iou_threshold) in iou_thresholds.iter().enumerate() {
        let (precision, recall, f1) =
            compute_precision_recall_at_iou(predictions, ground_truth, iou_threshold)?;

        precision_at_iou[i] = precision;
        recall_at_iou[i] = recall;
        f1_at_iou[i] = f1;
    }

    let map = ap_per_class.iter().sum::<f32>() / num_classes as f32;

    Ok(DetectionMetrics {
        map,
        ap_per_class,
        precision_at_iou,
        recall_at_iou,
        f1_at_iou,
    })
}

/// Compute segmentation metrics
pub fn compute_segmentation_metrics(
    predictions: &Tensor<f32>,
    targets: &Tensor<f32>,
    num_classes: usize,
) -> Result<SegmentationMetrics> {
    let pred_classes = get_predicted_classes(predictions)?;
    let pred_shape = pred_classes.shape();
    let pred_dims = pred_shape.dims();

    if pred_dims.len() != 3 {
        return Err(VisionError::InvalidShape(
            "Predictions must be 3D tensor (H, W, C) or (C, H, W)".to_string(),
        ));
    }

    let total_pixels = pred_dims.iter().product::<usize>();

    // Compute confusion matrix for segmentation
    let mut confusion_matrix = vec![vec![0usize; num_classes]; num_classes];
    let mut correct_pixels = 0;

    for i in 0..total_pixels {
        let indices = linear_to_indices(i, &pred_dims);
        let pred_class: f32 = pred_classes.get(&indices)?.clone().into();
        let true_class: f32 = targets.get(&indices)?.clone().into();

        let pred_idx = pred_class as usize;
        let true_idx = true_class as usize;

        if pred_idx < num_classes && true_idx < num_classes {
            confusion_matrix[true_idx][pred_idx] += 1;
            if pred_idx == true_idx {
                correct_pixels += 1;
            }
        }
    }

    let pixel_accuracy = correct_pixels as f32 / total_pixels as f32;

    // Compute per-class metrics
    let mut iou_per_class = vec![0.0; num_classes];
    let mut dice_per_class = vec![0.0; num_classes];
    let mut class_pixel_accuracies = vec![0.0; num_classes];

    for i in 0..num_classes {
        let tp = confusion_matrix[i][i] as f32;
        let fp: f32 = (0..num_classes)
            .map(|j| if j != i { confusion_matrix[j][i] } else { 0 })
            .sum::<usize>() as f32;
        let fn_count: f32 = (0..num_classes)
            .map(|j| if j != i { confusion_matrix[i][j] } else { 0 })
            .sum::<usize>() as f32;

        let class_total = tp + fn_count;
        class_pixel_accuracies[i] = if class_total > 0.0 {
            tp / class_total
        } else {
            0.0
        };

        let union = tp + fp + fn_count;
        iou_per_class[i] = if union > 0.0 { tp / union } else { 0.0 };

        let dice_denominator = 2.0 * tp + fp + fn_count;
        dice_per_class[i] = if dice_denominator > 0.0 {
            2.0 * tp / dice_denominator
        } else {
            0.0
        };
    }

    let mean_pixel_accuracy = class_pixel_accuracies.iter().sum::<f32>() / num_classes as f32;
    let mean_iou = iou_per_class.iter().sum::<f32>() / num_classes as f32;
    let mean_dice = dice_per_class.iter().sum::<f32>() / num_classes as f32;

    // Compute frequency weighted IoU
    let class_frequencies: Vec<f32> = (0..num_classes)
        .map(|i| {
            let class_count: usize = confusion_matrix[i].iter().sum();
            class_count as f32 / total_pixels as f32
        })
        .collect();

    let frequency_weighted_iou = iou_per_class
        .iter()
        .zip(class_frequencies.iter())
        .map(|(&iou, &freq)| iou * freq)
        .sum::<f32>();

    Ok(SegmentationMetrics {
        pixel_accuracy,
        mean_pixel_accuracy,
        mean_iou,
        iou_per_class,
        dice_per_class,
        mean_dice,
        frequency_weighted_iou,
    })
}

// Helper functions

fn log_softmax(predictions: &Tensor<f32>) -> Result<Tensor<f32>> {
    // Simplified log softmax implementation
    let softmax_result = softmax(predictions)?;
    apply_log(&softmax_result)
}

fn softmax(predictions: &Tensor<f32>) -> Result<Tensor<f32>> {
    let pred_shape = predictions.shape();
    let pred_dims = pred_shape.dims();
    let (batch_size, num_classes) = (pred_dims[0], pred_dims[1]);

    let mut result = zeros(&pred_dims)?;

    for i in 0..batch_size {
        // Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..num_classes {
            let val: f32 = predictions.get(&[i, j])?.clone().into();
            max_val = max_val.max(val);
        }

        // Compute exponentials and sum
        let mut exp_sum = 0.0;
        let mut exp_vals = vec![0.0; num_classes];
        for j in 0..num_classes {
            let val: f32 = predictions.get(&[i, j])?.clone().into();
            exp_vals[j] = (val - max_val).exp();
            exp_sum += exp_vals[j];
        }

        // Normalize
        for j in 0..num_classes {
            let softmax_val = exp_vals[j] / exp_sum;
            result.set(&[i, j], softmax_val.into())?;
        }
    }

    Ok(result)
}

fn sigmoid(tensor: &Tensor<f32>) -> Result<Tensor<f32>> {
    let shape = tensor.shape();
    let dims = shape.dims();
    let mut result = zeros(&dims)?;

    let total_elements = dims.iter().product::<usize>();

    for i in 0..total_elements {
        let indices = linear_to_indices(i, &dims);
        let val: f32 = tensor.get(&indices)?.clone().into();
        let sigmoid_val = 1.0 / (1.0 + (-val).exp());
        result.set(&indices, sigmoid_val.into())?;
    }

    Ok(result)
}

fn apply_log(tensor: &Tensor<f32>) -> Result<Tensor<f32>> {
    let shape = tensor.shape();
    let dims = shape.dims();
    let mut result = zeros(&dims)?;

    let total_elements = dims.iter().product::<usize>();

    for i in 0..total_elements {
        let indices = linear_to_indices(i, &dims);
        let val: f32 = tensor.get(&indices)?.clone().into();
        let log_val = (val + 1e-8).ln(); // Add small epsilon for numerical stability
        result.set(&indices, log_val.into())?;
    }

    Ok(result)
}

fn apply_label_smoothing(
    log_probs: &Tensor<f32>,
    sample_idx: usize,
    target_class: usize,
    num_classes: usize,
    smoothing: f32,
) -> Result<f32> {
    let mut smooth_loss = 0.0;
    for j in 0..num_classes {
        let log_prob: f32 = log_probs.get(&[sample_idx, j])?.clone().into();
        smooth_loss -= log_prob;
    }
    smooth_loss /= num_classes as f32;
    Ok(smooth_loss)
}

fn apply_reduction(losses: &Tensor<f32>, reduction: Reduction) -> Result<Tensor<f32>> {
    match reduction {
        Reduction::None => Ok(losses.clone()),
        Reduction::Mean | Reduction::BatchMean => {
            let mean_loss = compute_tensor_mean(losses)?;
            Ok(full(&[1], mean_loss)?)
        }
        Reduction::Sum => {
            let sum_loss = compute_tensor_sum(losses)?;
            Ok(full(&[1], sum_loss)?)
        }
    }
}

fn get_predicted_classes(predictions: &Tensor<f32>) -> Result<Tensor<f32>> {
    let pred_shape = predictions.shape();
    let pred_dims = pred_shape.dims();

    if pred_dims.len() == 2 {
        // Classification case: (N, C)
        let (batch_size, num_classes) = (pred_dims[0], pred_dims[1]);
        let mut result = zeros(&[batch_size])?;

        for i in 0..batch_size {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0;

            for j in 0..num_classes {
                let val: f32 = predictions.get(&[i, j])?.clone().into();
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }

            result.set(&[i], (max_idx as f32).into())?;
        }

        Ok(result)
    } else {
        // Segmentation case: apply argmax along channel dimension
        // This is a simplified implementation
        Ok(predictions.clone())
    }
}

fn compute_intersection(pred: &Tensor<f32>, target: &Tensor<f32>) -> Result<f32> {
    let shape = pred.shape();
    let dims = shape.dims();
    let total_elements = dims.iter().product::<usize>();

    let mut intersection = 0.0;

    for i in 0..total_elements {
        let indices = linear_to_indices(i, &dims);
        let pred_val: f32 = pred.get(&indices)?.clone().into();
        let target_val: f32 = target.get(&indices)?.clone().into();
        intersection += pred_val * target_val;
    }

    Ok(intersection)
}

fn compute_union(pred: &Tensor<f32>, target: &Tensor<f32>) -> Result<f32> {
    let pred_sum = compute_tensor_sum(pred)?;
    let target_sum = compute_tensor_sum(target)?;
    let intersection = compute_intersection(pred, target)?;

    Ok(pred_sum + target_sum - intersection)
}

fn compute_tensor_sum(tensor: &Tensor<f32>) -> Result<f32> {
    let shape = tensor.shape();
    let dims = shape.dims();
    let total_elements = dims.iter().product::<usize>();

    let mut sum = 0.0;

    for i in 0..total_elements {
        let indices = linear_to_indices(i, &dims);
        let val: f32 = tensor.get(&indices)?.clone().into();
        sum += val;
    }

    Ok(sum)
}

fn compute_tensor_mean(tensor: &Tensor<f32>) -> Result<f32> {
    let shape = tensor.shape();
    let dims = shape.dims();
    let total_elements = dims.iter().product::<usize>();

    let sum = compute_tensor_sum(tensor)?;
    Ok(sum / total_elements as f32)
}

fn compute_average_precision(
    predictions: &[&Detection],
    ground_truth: &[&Detection],
    iou_threshold: f32,
) -> Result<f32> {
    if predictions.is_empty() || ground_truth.is_empty() {
        return Ok(0.0);
    }

    // Sort predictions by confidence (descending)
    let mut sorted_predictions = predictions.to_vec();
    sorted_predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut tp = vec![false; sorted_predictions.len()];
    let mut gt_matched = vec![false; ground_truth.len()];

    // Match predictions to ground truth
    for (pred_idx, pred) in sorted_predictions.iter().enumerate() {
        let mut best_iou = 0.0;
        let mut best_gt_idx = None;

        for (gt_idx, gt) in ground_truth.iter().enumerate() {
            if gt_matched[gt_idx] {
                continue;
            }

            let iou = calculate_iou(&pred.bbox, &gt.bbox);
            if iou > best_iou {
                best_iou = iou;
                best_gt_idx = Some(gt_idx);
            }
        }

        if let Some(gt_idx) = best_gt_idx {
            if best_iou >= iou_threshold {
                tp[pred_idx] = true;
                gt_matched[gt_idx] = true;
            }
        }
    }

    // Compute precision and recall at each prediction
    let mut precisions = Vec::new();
    let mut recalls = Vec::new();
    let mut tp_count = 0;

    for (i, &is_tp) in tp.iter().enumerate() {
        if is_tp {
            tp_count += 1;
        }

        let precision = tp_count as f32 / (i + 1) as f32;
        let recall = tp_count as f32 / ground_truth.len() as f32;

        precisions.push(precision);
        recalls.push(recall);
    }

    // Compute AP using trapezoidal rule
    let mut ap = 0.0;
    for i in 1..recalls.len() {
        let recall_diff = recalls[i] - recalls[i - 1];
        let avg_precision = (precisions[i] + precisions[i - 1]) / 2.0;
        ap += recall_diff * avg_precision;
    }

    Ok(ap)
}

fn compute_precision_recall_at_iou(
    predictions: &[Detection],
    ground_truth: &[Detection],
    iou_threshold: f32,
) -> Result<(f32, f32, f32)> {
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_count = 0;

    let mut gt_matched = vec![false; ground_truth.len()];

    for pred in predictions {
        let mut matched = false;

        for (gt_idx, gt) in ground_truth.iter().enumerate() {
            if gt_matched[gt_idx] || pred.class_id != gt.class_id {
                continue;
            }

            let iou = calculate_iou(&pred.bbox, &gt.bbox);
            if iou >= iou_threshold {
                tp += 1;
                gt_matched[gt_idx] = true;
                matched = true;
                break;
            }
        }

        if !matched {
            fp += 1;
        }
    }

    fn_count = gt_matched.iter().filter(|&&matched| !matched).count();

    let precision = if tp + fp > 0 {
        tp as f32 / (tp + fp) as f32
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f32 / (tp + fn_count) as f32
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    Ok((precision, recall, f1))
}

fn linear_to_indices(linear_index: usize, dims: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; dims.len()];
    let mut remaining = linear_index;

    for i in (0..dims.len()).rev() {
        indices[i] = remaining % dims[i];
        remaining /= dims[i];
    }

    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::{ones, zeros};

    #[test]
    fn test_cross_entropy_loss() -> Result<()> {
        let predictions = zeros(&[2, 3])?; // 2 samples, 3 classes
        let targets = zeros(&[2])?; // Target classes

        let config = LossConfig::default();
        let loss = cross_entropy_loss(&predictions, &targets, config)?;

        assert_eq!(loss.shape().dims(), &[1]);
        Ok(())
    }

    #[test]
    fn test_focal_loss() -> Result<()> {
        let predictions = zeros(&[2, 3])?;
        let targets = zeros(&[2])?;

        let config = FocalLossConfig::default();
        let loss = focal_loss(&predictions, &targets, config)?;

        assert_eq!(loss.shape().dims(), &[1]);
        Ok(())
    }

    #[test]
    fn test_dice_loss() -> Result<()> {
        let predictions = zeros(&[1, 32, 32])?;
        let targets = zeros(&[1, 32, 32])?;

        let config = DiceLossConfig::default();
        let loss = dice_loss(&predictions, &targets, config)?;

        assert_eq!(loss.shape().dims(), &[1]);
        Ok(())
    }

    #[test]
    fn test_classification_metrics() -> Result<()> {
        let predictions = zeros(&[10, 3])?; // 10 samples, 3 classes
        let targets = zeros(&[10])?; // Target classes

        let metrics = compute_classification_metrics(&predictions, &targets, 3)?;

        assert_eq!(metrics.precision.len(), 3);
        assert_eq!(metrics.recall.len(), 3);
        assert_eq!(metrics.f1_score.len(), 3);
        assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);

        Ok(())
    }

    #[test]
    fn test_segmentation_metrics() -> Result<()> {
        let predictions = zeros(&[32, 32, 1])?; // H x W x C
        let targets = zeros(&[32, 32, 1])?;

        let metrics = compute_segmentation_metrics(&predictions, &targets, 2)?;

        assert_eq!(metrics.iou_per_class.len(), 2);
        assert_eq!(metrics.dice_per_class.len(), 2);
        assert!(metrics.pixel_accuracy >= 0.0 && metrics.pixel_accuracy <= 1.0);
        assert!(metrics.mean_iou >= 0.0 && metrics.mean_iou <= 1.0);

        Ok(())
    }

    #[test]
    fn test_detection_metrics() -> Result<()> {
        let predictions = vec![
            Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            Detection::new([20.0, 20.0, 30.0, 30.0], 0.8, 1),
        ];

        let ground_truth = vec![
            Detection::new([1.0, 1.0, 9.0, 9.0], 1.0, 0),
            Detection::new([21.0, 21.0, 29.0, 29.0], 1.0, 1),
        ];

        let iou_thresholds = vec![0.3, 0.5, 0.7];
        let metrics = compute_detection_metrics(&predictions, &ground_truth, &iou_thresholds, 2)?;

        assert_eq!(metrics.ap_per_class.len(), 2);
        assert_eq!(metrics.precision_at_iou.len(), 3);
        assert!(metrics.map >= 0.0 && metrics.map <= 1.0);

        Ok(())
    }

    #[test]
    fn test_loss_configs() {
        let config = LossConfig::with_label_smoothing(0.1);
        assert_eq!(config.label_smoothing, 0.1);

        let config = LossConfig::with_class_weights(vec![1.0, 2.0, 0.5]);
        assert!(config.class_weights.is_some());

        let focal_config = FocalLossConfig::new(0.25, 2.0);
        assert_eq!(focal_config.alpha, 0.25);
        assert_eq!(focal_config.gamma, 2.0);
    }

    #[test]
    fn test_softmax() -> Result<()> {
        let input = zeros(&[2, 3])?;
        let result = softmax(&input)?;

        assert_eq!(result.shape().dims(), &[2, 3]);

        // Check that softmax sums to 1 for each sample
        for i in 0..2 {
            let mut sum = 0.0;
            for j in 0..3 {
                let val: f32 = result.get(&[i, j])?.clone().into();
                sum += val;
            }
            assert!((sum - 1.0).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_sigmoid() -> Result<()> {
        let input = zeros(&[2, 2])?;
        let result = sigmoid(&input)?;

        assert_eq!(result.shape().dims(), &[2, 2]);

        // Check that all values are in [0, 1]
        let total_elements = 4;
        for i in 0..total_elements {
            let indices = linear_to_indices(i, &[2, 2]);
            let val: f32 = result.get(&indices)?.clone().into();
            assert!(val >= 0.0 && val <= 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_reduction_strategies() -> Result<()> {
        let losses = ones(&[4])?; // All losses = 1.0

        let none_result = apply_reduction(&losses, Reduction::None)?;
        assert_eq!(none_result.shape().dims(), &[4]);

        let mean_result = apply_reduction(&losses, Reduction::Mean)?;
        assert_eq!(mean_result.shape().dims(), &[1]);
        let mean_val: f32 = mean_result.get(&[0])?.clone().into();
        assert!((mean_val - 1.0).abs() < 1e-6);

        let sum_result = apply_reduction(&losses, Reduction::Sum)?;
        let sum_val: f32 = sum_result.get(&[0])?.clone().into();
        assert!((sum_val - 4.0).abs() < 1e-6);

        Ok(())
    }
}
