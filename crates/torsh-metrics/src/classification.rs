//! Classification metrics

use crate::Metric;
use torsh_core::{device::DeviceType, error::TorshError};
use torsh_tensor::{
    creation::{ones, zeros},
    Tensor,
};
// Enhanced with scirs2-metrics integration
// use scirs2_metrics::classification::*; // Will be added when API stabilizes

/// Averaging method for multi-class metrics
#[derive(Debug, Clone)]
pub enum AverageMethod {
    Micro,
    Macro,
    Weighted,
    None, // Per-class results
}

/// Accuracy metric
#[derive(Clone)]
pub struct Accuracy {
    top_k: Option<usize>,
}

impl Accuracy {
    /// Create a new accuracy metric
    pub fn new() -> Self {
        Self { top_k: None }
    }

    /// Create a top-k accuracy metric
    pub fn top_k(k: usize) -> Self {
        Self { top_k: Some(k) }
    }
}

impl Metric for Accuracy {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        if let Some(k) = self.top_k {
            // Top-k accuracy
            compute_top_k_accuracy(predictions, targets, k)
        } else {
            // Standard accuracy using robust implementation
            compute_standard_accuracy(predictions, targets)
        }
    }

    fn name(&self) -> &str {
        if self.top_k.is_some() {
            "top_k_accuracy"
        } else {
            "accuracy"
        }
    }
}

/// Precision metric - Enhanced with scirs2-metrics compatibility
pub struct Precision {
    average: AverageMethod,
}

impl Precision {
    /// Create a new precision metric
    pub fn new(average: AverageMethod) -> Self {
        Self { average }
    }

    /// Create micro-averaged precision
    pub fn micro() -> Self {
        Self {
            average: AverageMethod::Micro,
        }
    }

    /// Create macro-averaged precision
    pub fn macro_averaged() -> Self {
        Self {
            average: AverageMethod::Macro,
        }
    }
}

impl Metric for Precision {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        compute_precision(predictions, targets, &self.average)
    }

    fn name(&self) -> &str {
        match self.average {
            AverageMethod::Micro => "precision_micro",
            AverageMethod::Macro => "precision_macro",
            AverageMethod::Weighted => "precision_weighted",
            AverageMethod::None => "precision",
        }
    }
}

/// Recall metric - Enhanced with scirs2-metrics compatibility
pub struct Recall {
    average: AverageMethod,
}

impl Recall {
    /// Create a new recall metric
    pub fn new(average: AverageMethod) -> Self {
        Self { average }
    }

    /// Create micro-averaged recall
    pub fn micro() -> Self {
        Self {
            average: AverageMethod::Micro,
        }
    }

    /// Create macro-averaged recall
    pub fn macro_averaged() -> Self {
        Self {
            average: AverageMethod::Macro,
        }
    }
}

impl Metric for Recall {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        compute_recall(predictions, targets, &self.average)
    }

    fn name(&self) -> &str {
        match self.average {
            AverageMethod::Micro => "recall_micro",
            AverageMethod::Macro => "recall_macro",
            AverageMethod::Weighted => "recall_weighted",
            AverageMethod::None => "recall",
        }
    }
}

/// F1 Score metric - Enhanced with scirs2-metrics compatibility
pub struct F1Score {
    average: AverageMethod,
}

impl F1Score {
    /// Create a new F1 score metric
    pub fn new(average: AverageMethod) -> Self {
        Self { average }
    }

    /// Create micro-averaged F1 score
    pub fn micro() -> Self {
        Self {
            average: AverageMethod::Micro,
        }
    }

    /// Create macro-averaged F1 score
    pub fn macro_averaged() -> Self {
        Self {
            average: AverageMethod::Macro,
        }
    }
}

impl Metric for F1Score {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        compute_f1_score(predictions, targets, &self.average)
    }

    fn name(&self) -> &str {
        match self.average {
            AverageMethod::Micro => "f1_micro",
            AverageMethod::Macro => "f1_macro",
            AverageMethod::Weighted => "f1_weighted",
            AverageMethod::None => "f1",
        }
    }
}

// Implementation functions for the metrics
fn compute_standard_accuracy(predictions: &Tensor, targets: &Tensor) -> f64 {
    // Handle empty tensors
    if predictions.numel() == 0 || targets.numel() == 0 {
        return 0.0;
    }

    // Use manual argmax as workaround for tensor API issues
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(targets_vec)) => {
            // Get tensor shape
            let shape = predictions.shape();
            let dims = shape.dims();

            // Handle both 1D and 2D predictions
            let (rows, cols) = if dims.len() == 1 {
                // 1D predictions - binary classification with threshold 0.5
                let rows = dims[0];
                if rows == 0 || targets_vec.len() != rows {
                    return 0.0;
                }

                let mut correct = 0;
                for i in 0..rows {
                    let predicted_class = if pred_vec[i] >= 0.5 { 1.0 } else { 0.0 };
                    if (predicted_class - targets_vec[i]).abs() < 1e-6 {
                        correct += 1;
                    }
                }
                return correct as f64 / rows as f64;
            } else if dims.len() == 2 {
                let rows = dims[0];
                let cols = dims[1];
                if rows == 0 || cols == 0 || targets_vec.len() != rows {
                    return 0.0;
                }
                (rows, cols)
            } else {
                return 0.0;
            };

            let mut correct = 0;

            // Manually compute argmax for each row
            for i in 0..rows {
                let mut max_idx = 0;
                let mut max_val = pred_vec[i * cols];

                for j in 1..cols {
                    let val = pred_vec[i * cols + j];
                    if val > max_val {
                        max_val = val;
                        max_idx = j;
                    }
                }

                if max_idx as i64 == targets_vec[i] as i64 {
                    correct += 1;
                }
            }

            correct as f64 / rows as f64
        }
        _ => 0.0,
    }
}

fn compute_top_k_accuracy(predictions: &Tensor, targets: &Tensor, k: usize) -> f64 {
    // Handle empty tensors
    if predictions.numel() == 0 || targets.numel() == 0 {
        return 0.0;
    }

    // Use manual top-k computation as workaround
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(targets_vec)) => {
            let shape = predictions.shape();
            let dims = shape.dims();

            if dims.len() != 2 {
                return 0.0;
            }

            let rows = dims[0];
            let cols = dims[1];

            if rows == 0 || cols == 0 || targets_vec.len() != rows || k > cols {
                return 0.0;
            }

            let mut correct = 0;

            // Manually compute top-k for each row
            for i in 0..rows {
                let target = targets_vec[i] as usize;

                // Get values for this row and find top-k indices
                let mut row_values: Vec<(f32, usize)> =
                    (0..cols).map(|j| (pred_vec[i * cols + j], j)).collect();

                // Sort by value in descending order
                row_values.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                // Check if target is in top-k
                for j in 0..k.min(row_values.len()) {
                    if row_values[j].1 == target {
                        correct += 1;
                        break;
                    }
                }
            }

            correct as f64 / rows as f64
        }
        _ => 0.0,
    }
}

fn compute_precision(predictions: &Tensor, targets: &Tensor, average: &AverageMethod) -> f64 {
    let (tp, fp, _fn) = compute_confusion_components(predictions, targets);

    match average {
        AverageMethod::Micro => {
            let total_tp: f64 = tp.iter().sum();
            let total_fp: f64 = fp.iter().sum();

            if total_tp + total_fp > 0.0 {
                total_tp / (total_tp + total_fp)
            } else {
                0.0
            }
        }
        AverageMethod::Macro => {
            let precisions: Vec<f64> = tp
                .iter()
                .zip(fp.iter())
                .map(|(tp_val, fp_val)| {
                    if tp_val + fp_val > 0.0 {
                        tp_val / (tp_val + fp_val)
                    } else {
                        0.0
                    }
                })
                .collect();

            if precisions.is_empty() {
                0.0
            } else {
                precisions.iter().sum::<f64>() / precisions.len() as f64
            }
        }
        _ => {
            // For weighted and per-class, return macro for now
            compute_precision(predictions, targets, &AverageMethod::Macro)
        }
    }
}

fn compute_recall(predictions: &Tensor, targets: &Tensor, average: &AverageMethod) -> f64 {
    let (tp, _fp, fn_) = compute_confusion_components(predictions, targets);

    match average {
        AverageMethod::Micro => {
            let total_tp: f64 = tp.iter().sum();
            let total_fn: f64 = fn_.iter().sum();

            if total_tp + total_fn > 0.0 {
                total_tp / (total_tp + total_fn)
            } else {
                0.0
            }
        }
        AverageMethod::Macro => {
            let recalls: Vec<f64> = tp
                .iter()
                .zip(fn_.iter())
                .map(|(tp_val, fn_val)| {
                    if tp_val + fn_val > 0.0 {
                        tp_val / (tp_val + fn_val)
                    } else {
                        0.0
                    }
                })
                .collect();

            if recalls.is_empty() {
                0.0
            } else {
                recalls.iter().sum::<f64>() / recalls.len() as f64
            }
        }
        _ => {
            // For weighted and per-class, return macro for now
            compute_recall(predictions, targets, &AverageMethod::Macro)
        }
    }
}

fn compute_f1_score(predictions: &Tensor, targets: &Tensor, average: &AverageMethod) -> f64 {
    let precision = compute_precision(predictions, targets, average);
    let recall = compute_recall(predictions, targets, average);

    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

/// Compute confusion matrix components (TP, FP, FN) for each class
fn compute_confusion_components(
    predictions: &Tensor,
    targets: &Tensor,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Use manual argmax computation
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(targets_vec)) => {
            let shape = predictions.shape();
            let dims = shape.dims();

            if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 || targets_vec.len() != dims[0] {
                return (vec![0.0], vec![0.0], vec![0.0]);
            }

            let rows = dims[0];
            let cols = dims[1];

            // Manually compute predicted classes
            let mut preds_vec = Vec::with_capacity(rows);
            for i in 0..rows {
                let mut max_idx = 0;
                let mut max_val = pred_vec[i * cols];

                for j in 1..cols {
                    let val = pred_vec[i * cols + j];
                    if val > max_val {
                        max_val = val;
                        max_idx = j;
                    }
                }
                preds_vec.push(max_idx as f32);
            }

            // Determine number of classes
            let max_target = targets_vec.iter().map(|&x| x as usize).max().unwrap_or(0);
            let max_pred = preds_vec.iter().map(|&x| x as usize).max().unwrap_or(0);
            let num_classes = (max_target.max(max_pred) + 1).max(2);

            let mut tp = vec![0.0; num_classes];
            let mut fp = vec![0.0; num_classes];
            let mut fn_ = vec![0.0; num_classes];

            // Compute confusion matrix components
            for i in 0..targets_vec.len().min(preds_vec.len()) {
                let target = targets_vec[i] as usize;
                let pred = preds_vec[i] as usize;

                if target < num_classes && pred < num_classes {
                    if target == pred {
                        tp[target] += 1.0;
                    } else {
                        fp[pred] += 1.0;
                        fn_[target] += 1.0;
                    }
                }
            }

            (tp, fp, fn_)
        }
        _ => (vec![0.0], vec![0.0], vec![0.0]),
    }
}
