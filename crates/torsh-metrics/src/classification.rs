//! Classification metrics

use crate::Metric;
use torsh_tensor::Tensor;
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
                row_values.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .expect("row values should be comparable")
                });

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

/// Multi-class classification metrics with per-class statistics
#[derive(Debug, Clone)]
pub struct MultiClassMetrics {
    /// Per-class precision scores
    pub per_class_precision: Vec<f64>,
    /// Per-class recall scores
    pub per_class_recall: Vec<f64>,
    /// Per-class F1 scores
    pub per_class_f1: Vec<f64>,
    /// Macro-averaged F1 score (unweighted mean across classes)
    pub macro_avg: f64,
    /// Weighted-averaged F1 score (weighted by support)
    pub weighted_avg: f64,
    /// Per-class support (number of true instances for each class)
    pub support: Vec<usize>,
}

impl MultiClassMetrics {
    /// Compute multi-class metrics from predictions and targets
    pub fn compute(predictions: &Tensor, targets: &Tensor) -> Self {
        let (tp, fp, fn_) = compute_confusion_components(predictions, targets);

        let num_classes = tp.len();
        let mut per_class_precision = Vec::with_capacity(num_classes);
        let mut per_class_recall = Vec::with_capacity(num_classes);
        let mut per_class_f1 = Vec::with_capacity(num_classes);
        let mut support = Vec::with_capacity(num_classes);

        // Compute per-class metrics
        for i in 0..num_classes {
            let precision = if tp[i] + fp[i] > 0.0 {
                tp[i] / (tp[i] + fp[i])
            } else {
                0.0
            };

            let recall = if tp[i] + fn_[i] > 0.0 {
                tp[i] / (tp[i] + fn_[i])
            } else {
                0.0
            };

            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            per_class_precision.push(precision);
            per_class_recall.push(recall);
            per_class_f1.push(f1);
            support.push((tp[i] + fn_[i]) as usize);
        }

        // Compute macro average (simple mean)
        let macro_avg = if !per_class_f1.is_empty() {
            per_class_f1.iter().sum::<f64>() / per_class_f1.len() as f64
        } else {
            0.0
        };

        // Compute weighted average (weighted by support)
        let total_support: usize = support.iter().sum();
        let weighted_avg = if total_support > 0 {
            per_class_f1
                .iter()
                .zip(support.iter())
                .map(|(f1, sup)| f1 * (*sup as f64))
                .sum::<f64>()
                / total_support as f64
        } else {
            0.0
        };

        MultiClassMetrics {
            per_class_precision,
            per_class_recall,
            per_class_f1,
            macro_avg,
            weighted_avg,
            support,
        }
    }

    /// Get the number of classes
    pub fn num_classes(&self) -> usize {
        self.per_class_f1.len()
    }

    /// Get metrics for a specific class
    pub fn class_metrics(&self, class_idx: usize) -> Option<(f64, f64, f64, usize)> {
        if class_idx < self.num_classes() {
            Some((
                self.per_class_precision[class_idx],
                self.per_class_recall[class_idx],
                self.per_class_f1[class_idx],
                self.support[class_idx],
            ))
        } else {
            None
        }
    }

    /// Format metrics as a human-readable string
    pub fn format(&self) -> String {
        let mut result = String::new();
        result.push_str("Multi-Class Metrics:\n");
        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for i in 0..self.num_classes() {
            result.push_str(&format!(
                "Class {}: Precision={:.4}, Recall={:.4}, F1={:.4}, Support={}\n",
                i,
                self.per_class_precision[i],
                self.per_class_recall[i],
                self.per_class_f1[i],
                self.support[i]
            ));
        }

        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        result.push_str(&format!("Macro Avg F1: {:.4}\n", self.macro_avg));
        result.push_str(&format!("Weighted Avg F1: {:.4}\n", self.weighted_avg));

        result
    }
}

/// Confusion Matrix for multi-class classification
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// The confusion matrix as a 2D array (rows=true labels, cols=predicted labels)
    pub matrix: Vec<Vec<usize>>,
    /// Number of classes
    pub num_classes: usize,
    /// Class labels (optional)
    pub labels: Option<Vec<String>>,
}

impl ConfusionMatrix {
    /// Create a confusion matrix from predictions and targets
    pub fn compute(predictions: &Tensor, targets: &Tensor) -> Self {
        let (matrix, num_classes) = compute_confusion_matrix(predictions, targets);

        ConfusionMatrix {
            matrix,
            num_classes,
            labels: None,
        }
    }

    /// Create a confusion matrix with custom class labels
    pub fn compute_with_labels(
        predictions: &Tensor,
        targets: &Tensor,
        labels: Vec<String>,
    ) -> Self {
        let (matrix, num_classes) = compute_confusion_matrix(predictions, targets);

        ConfusionMatrix {
            matrix,
            num_classes,
            labels: Some(labels),
        }
    }

    /// Get the value at position (true_class, predicted_class)
    pub fn get(&self, true_class: usize, predicted_class: usize) -> Option<usize> {
        if true_class < self.num_classes && predicted_class < self.num_classes {
            Some(self.matrix[true_class][predicted_class])
        } else {
            None
        }
    }

    /// Get the total number of samples
    pub fn total(&self) -> usize {
        self.matrix
            .iter()
            .map(|row| row.iter().sum::<usize>())
            .sum()
    }

    /// Get accuracy from the confusion matrix
    pub fn accuracy(&self) -> f64 {
        let correct: usize = (0..self.num_classes).map(|i| self.matrix[i][i]).sum();
        let total = self.total();

        if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get the diagonal (true positives for each class)
    pub fn diagonal(&self) -> Vec<usize> {
        (0..self.num_classes).map(|i| self.matrix[i][i]).collect()
    }

    /// Normalize the confusion matrix by true labels (row-wise)
    pub fn normalize_by_true(&self) -> Vec<Vec<f64>> {
        self.matrix
            .iter()
            .map(|row| {
                let sum: usize = row.iter().sum();
                if sum > 0 {
                    row.iter().map(|&val| val as f64 / sum as f64).collect()
                } else {
                    vec![0.0; self.num_classes]
                }
            })
            .collect()
    }

    /// Normalize the confusion matrix by predicted labels (column-wise)
    pub fn normalize_by_pred(&self) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.num_classes]; self.num_classes];

        // Compute column sums
        let mut col_sums = vec![0; self.num_classes];
        for row in &self.matrix {
            for (j, &val) in row.iter().enumerate() {
                col_sums[j] += val;
            }
        }

        // Normalize
        for i in 0..self.num_classes {
            for j in 0..self.num_classes {
                if col_sums[j] > 0 {
                    result[i][j] = self.matrix[i][j] as f64 / col_sums[j] as f64;
                }
            }
        }

        result
    }

    /// Normalize the confusion matrix by all samples
    pub fn normalize_all(&self) -> Vec<Vec<f64>> {
        let total = self.total();
        if total > 0 {
            self.matrix
                .iter()
                .map(|row| row.iter().map(|&val| val as f64 / total as f64).collect())
                .collect()
        } else {
            vec![vec![0.0; self.num_classes]; self.num_classes]
        }
    }

    /// Format the confusion matrix as a string
    pub fn format(&self) -> String {
        let mut result = String::new();
        result.push_str("Confusion Matrix:\n");

        // Header
        result.push_str("        ");
        for j in 0..self.num_classes {
            if let Some(ref labels) = self.labels {
                if j < labels.len() {
                    result.push_str(&format!("{:>8} ", labels[j]));
                } else {
                    result.push_str(&format!("{:>8} ", j));
                }
            } else {
                result.push_str(&format!("{:>8} ", j));
            }
        }
        result.push('\n');

        // Matrix rows
        for i in 0..self.num_classes {
            if let Some(ref labels) = self.labels {
                if i < labels.len() {
                    result.push_str(&format!("{:>8} ", labels[i]));
                } else {
                    result.push_str(&format!("{:>8} ", i));
                }
            } else {
                result.push_str(&format!("{:>8} ", i));
            }

            for j in 0..self.num_classes {
                result.push_str(&format!("{:>8} ", self.matrix[i][j]));
            }
            result.push('\n');
        }

        result
    }

    /// Format normalized confusion matrix (by true labels)
    pub fn format_normalized(&self) -> String {
        let normalized = self.normalize_by_true();
        let mut result = String::new();
        result.push_str("Normalized Confusion Matrix (by true labels):\n");

        // Header
        result.push_str("        ");
        for j in 0..self.num_classes {
            if let Some(ref labels) = self.labels {
                if j < labels.len() {
                    result.push_str(&format!("{:>8} ", labels[j]));
                } else {
                    result.push_str(&format!("{:>8} ", j));
                }
            } else {
                result.push_str(&format!("{:>8} ", j));
            }
        }
        result.push('\n');

        // Matrix rows
        for i in 0..self.num_classes {
            if let Some(ref labels) = self.labels {
                if i < labels.len() {
                    result.push_str(&format!("{:>8} ", labels[i]));
                } else {
                    result.push_str(&format!("{:>8} ", i));
                }
            } else {
                result.push_str(&format!("{:>8} ", i));
            }

            for j in 0..self.num_classes {
                result.push_str(&format!("{:>8.4} ", normalized[i][j]));
            }
            result.push('\n');
        }

        result
    }
}

/// Helper function to compute the confusion matrix
fn compute_confusion_matrix(predictions: &Tensor, targets: &Tensor) -> (Vec<Vec<usize>>, usize) {
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(targets_vec)) => {
            let shape = predictions.shape();
            let dims = shape.dims();

            if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 || targets_vec.len() != dims[0] {
                return (vec![vec![0; 2]; 2], 2);
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
                preds_vec.push(max_idx);
            }

            // Determine number of classes
            let max_target = targets_vec.iter().map(|&x| x as usize).max().unwrap_or(0);
            let max_pred = preds_vec.iter().map(|&x| x).max().unwrap_or(0);
            let num_classes = (max_target.max(max_pred) + 1).max(2);

            // Initialize confusion matrix
            let mut matrix = vec![vec![0; num_classes]; num_classes];

            // Populate confusion matrix
            for i in 0..targets_vec.len().min(preds_vec.len()) {
                let target = targets_vec[i] as usize;
                let pred = preds_vec[i];

                if target < num_classes && pred < num_classes {
                    matrix[target][pred] += 1;
                }
            }

            (matrix, num_classes)
        }
        _ => (vec![vec![0; 2]; 2], 2),
    }
}

/// Threshold-dependent metrics for binary classification
#[derive(Debug, Clone)]
pub struct ThresholdMetrics {
    /// Optimal threshold for classification
    pub optimal_threshold: f64,
    /// Precision-recall curve (precision, recall) pairs at different thresholds
    pub precision_recall_curve: (Vec<f64>, Vec<f64>),
    /// ROC curve (false positive rate, true positive rate) pairs
    pub roc_curve: (Vec<f64>, Vec<f64>),
    /// Thresholds used for the curves
    pub thresholds: Vec<f64>,
}

impl ThresholdMetrics {
    /// Compute threshold metrics for binary classification
    /// predictions: probabilities for the positive class (shape: \[n\])
    /// targets: binary labels (0 or 1) (shape: \[n\])
    pub fn compute(predictions: &Tensor, targets: &Tensor) -> Self {
        let (pred_vec, targets_vec) = match (predictions.to_vec(), targets.to_vec()) {
            (Ok(p), Ok(t)) => (p, t),
            _ => {
                return ThresholdMetrics {
                    optimal_threshold: 0.5,
                    precision_recall_curve: (vec![0.0], vec![0.0]),
                    roc_curve: (vec![0.0], vec![0.0]),
                    thresholds: vec![0.5],
                }
            }
        };

        if pred_vec.is_empty() || targets_vec.is_empty() || pred_vec.len() != targets_vec.len() {
            return ThresholdMetrics {
                optimal_threshold: 0.5,
                precision_recall_curve: (vec![0.0], vec![0.0]),
                roc_curve: (vec![0.0], vec![0.0]),
                thresholds: vec![0.5],
            };
        }

        // Generate thresholds
        let mut thresholds: Vec<f64> = pred_vec.iter().map(|&x| x as f64).collect();
        thresholds.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("threshold values should be comparable")
        });
        thresholds.dedup();

        // Add boundary thresholds
        if !thresholds.contains(&0.0) {
            thresholds.insert(0, 0.0);
        }
        if !thresholds.contains(&1.0) {
            thresholds.push(1.0);
        }

        let mut precisions = Vec::new();
        let mut recalls = Vec::new();
        let mut fprs = Vec::new();
        let mut tprs = Vec::new();

        // Compute metrics at each threshold
        for &threshold in &thresholds {
            let (tp, fp, tn, fn_) =
                compute_binary_confusion_matrix(&pred_vec, &targets_vec, threshold as f32);

            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 };

            let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };

            let fpr = if fp + tn > 0.0 { fp / (fp + tn) } else { 0.0 };

            let tpr = recall;

            precisions.push(precision);
            recalls.push(recall);
            fprs.push(fpr);
            tprs.push(tpr);
        }

        // Find optimal threshold (maximize F1 score)
        let mut best_f1 = 0.0;
        let mut optimal_threshold = 0.5;

        for (i, &threshold) in thresholds.iter().enumerate() {
            let precision = precisions[i];
            let recall = recalls[i];

            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            if f1 > best_f1 {
                best_f1 = f1;
                optimal_threshold = threshold;
            }
        }

        ThresholdMetrics {
            optimal_threshold,
            precision_recall_curve: (precisions, recalls),
            roc_curve: (fprs, tprs),
            thresholds,
        }
    }

    /// Calculate Area Under the ROC Curve (AUC-ROC)
    pub fn auc_roc(&self) -> f64 {
        let (fprs, tprs) = &self.roc_curve;
        calculate_auc(fprs, tprs)
    }

    /// Calculate Area Under the Precision-Recall Curve (AUC-PR)
    pub fn auc_pr(&self) -> f64 {
        let (precisions, recalls) = &self.precision_recall_curve;
        calculate_auc(recalls, precisions)
    }

    /// Get precision and recall at the optimal threshold
    pub fn optimal_metrics(&self) -> (f64, f64) {
        if let Some(idx) = self
            .thresholds
            .iter()
            .position(|&t| (t - self.optimal_threshold).abs() < 1e-9)
        {
            (
                self.precision_recall_curve.0[idx],
                self.precision_recall_curve.1[idx],
            )
        } else {
            (0.0, 0.0)
        }
    }
}

/// Helper function to compute binary confusion matrix at a threshold
fn compute_binary_confusion_matrix(
    predictions: &[f32],
    targets: &[f32],
    threshold: f32,
) -> (f64, f64, f64, f64) {
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut tn = 0.0;
    let mut fn_ = 0.0;

    for (&pred, &target) in predictions.iter().zip(targets.iter()) {
        let pred_class = if pred >= threshold { 1.0 } else { 0.0 };

        if target > 0.5 {
            // Positive class
            if pred_class > 0.5 {
                tp += 1.0;
            } else {
                fn_ += 1.0;
            }
        } else {
            // Negative class
            if pred_class > 0.5 {
                fp += 1.0;
            } else {
                tn += 1.0;
            }
        }
    }

    (tp, fp, tn, fn_)
}

/// Calculate Area Under Curve using trapezoidal rule
fn calculate_auc(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let mut auc = 0.0;
    for i in 1..x.len() {
        let dx = x[i] - x[i - 1];
        let avg_y = (y[i] + y[i - 1]) / 2.0;
        auc += dx.abs() * avg_y;
    }

    auc
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_multi_class_metrics() {
        // Create sample predictions and targets
        let predictions = from_vec(
            vec![
                0.9, 0.1, 0.0, // Class 0
                0.2, 0.7, 0.1, // Class 1
                0.1, 0.2, 0.7, // Class 2
                0.8, 0.1, 0.1, // Class 0
                0.1, 0.8, 0.1, // Class 1
            ],
            &[5, 3],
            DeviceType::Cpu,
        )
        .unwrap();
        let targets = from_vec(vec![0.0, 1.0, 2.0, 0.0, 1.0], &[5], DeviceType::Cpu).unwrap();

        let metrics = MultiClassMetrics::compute(&predictions, &targets);

        // Perfect predictions, so all metrics should be 1.0
        assert_eq!(metrics.num_classes(), 3);
        assert!((metrics.macro_avg - 1.0).abs() < 1e-6);
        assert!((metrics.weighted_avg - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_confusion_matrix() {
        let predictions = from_vec(
            vec![
                0.9, 0.1, 0.0, // Predicts 0, actually 0 (TP)
                0.2, 0.7, 0.1, // Predicts 1, actually 1 (TP)
                0.1, 0.2, 0.7, // Predicts 2, actually 2 (TP)
            ],
            &[3, 3],
            DeviceType::Cpu,
        )
        .unwrap();
        let targets = from_vec(vec![0.0, 1.0, 2.0], &[3], DeviceType::Cpu).unwrap();

        let cm = ConfusionMatrix::compute(&predictions, &targets);

        assert_eq!(cm.num_classes, 3);
        assert_eq!(cm.total(), 3);
        assert!((cm.accuracy() - 1.0).abs() < 1e-6);
        assert_eq!(cm.diagonal(), vec![1, 1, 1]);
    }

    #[test]
    fn test_threshold_metrics() {
        // Binary classification probabilities
        let predictions = from_vec(vec![0.9, 0.8, 0.3, 0.2, 0.7], &[5], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 1.0, 0.0, 0.0, 1.0], &[5], DeviceType::Cpu).unwrap();

        let metrics = ThresholdMetrics::compute(&predictions, &targets);

        // Should find a reasonable threshold
        assert!(metrics.optimal_threshold > 0.0);
        assert!(metrics.optimal_threshold < 1.0);

        // AUC should be high for this good prediction
        let auc = metrics.auc_roc();
        assert!(auc > 0.8);
    }

    #[test]
    fn test_confusion_matrix_normalization() {
        let predictions = from_vec(
            vec![
                0.9, 0.1, // Predicts 0, actually 0
                0.2, 0.8, // Predicts 1, actually 1
                0.7, 0.3, // Predicts 0, actually 0
                0.3, 0.7, // Predicts 1, actually 1
            ],
            &[4, 2],
            DeviceType::Cpu,
        )
        .unwrap();
        let targets = from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4], DeviceType::Cpu).unwrap();

        let cm = ConfusionMatrix::compute(&predictions, &targets);

        let normalized = cm.normalize_by_true();
        // Each row should sum to 1.0
        for row in &normalized {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
