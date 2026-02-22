//! Scikit-learn compatibility layer for torsh-metrics
//!
//! This module provides utilities for validating torsh-metrics against scikit-learn
//! reference implementations and for creating scikit-learn compatible API interfaces.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Scikit-learn compatible metric interface
pub trait SklearnMetric {
    /// Compute the metric in a scikit-learn compatible way
    fn sklearn_compute(&self, y_true: &[f64], y_pred: &[f64]) -> f64;

    /// Get the metric name (matching scikit-learn naming)
    fn sklearn_name(&self) -> &str;

    /// Get the metric parameters (matching scikit-learn parameters)
    fn sklearn_params(&self) -> HashMap<String, String>;
}

/// Accuracy metric (scikit-learn compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SklearnAccuracy {
    /// Whether to normalize the accuracy
    pub normalize: bool,
}

impl SklearnAccuracy {
    /// Create a new sklearn-compatible accuracy metric
    pub fn new() -> Self {
        Self { normalize: true }
    }

    /// Set normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Compute accuracy
    pub fn compute(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "y_true and y_pred must have same length"
        );

        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| t == p)
            .count();

        if self.normalize {
            correct as f64 / y_true.len() as f64
        } else {
            correct as f64
        }
    }
}

impl Default for SklearnAccuracy {
    fn default() -> Self {
        Self::new()
    }
}

/// Precision metric (scikit-learn compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SklearnPrecision {
    /// Average type: 'binary', 'micro', 'macro', 'weighted'
    pub average: String,
    /// Positive class label for binary classification
    pub pos_label: usize,
    /// Whether to include zero division handling
    pub zero_division: f64,
}

impl SklearnPrecision {
    /// Create a new sklearn-compatible precision metric
    pub fn new() -> Self {
        Self {
            average: "binary".to_string(),
            pos_label: 1,
            zero_division: 0.0,
        }
    }

    /// Set average type
    pub fn with_average(mut self, average: impl Into<String>) -> Self {
        self.average = average.into();
        self
    }

    /// Set positive label
    pub fn with_pos_label(mut self, pos_label: usize) -> Self {
        self.pos_label = pos_label;
        self
    }

    /// Set zero division handling
    pub fn with_zero_division(mut self, zero_division: f64) -> Self {
        self.zero_division = zero_division;
        self
    }

    /// Compute precision
    pub fn compute(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "y_true and y_pred must have same length"
        );

        match self.average.as_str() {
            "binary" => self.compute_binary_precision(y_true, y_pred),
            "micro" => self.compute_micro_precision(y_true, y_pred),
            "macro" => self.compute_macro_precision(y_true, y_pred),
            "weighted" => self.compute_weighted_precision(y_true, y_pred),
            _ => panic!("Unknown average type: {}", self.average),
        }
    }

    fn compute_binary_precision(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        let tp = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| **t == self.pos_label && **p == self.pos_label)
            .count();
        let fp = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| **t != self.pos_label && **p == self.pos_label)
            .count();

        if tp + fp == 0 {
            self.zero_division
        } else {
            tp as f64 / (tp + fp) as f64
        }
    }

    fn compute_micro_precision(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        // Micro average is the same as accuracy for multi-class
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| t == p)
            .count();
        correct as f64 / y_true.len() as f64
    }

    fn compute_macro_precision(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        let unique_labels = get_unique_labels(y_true, y_pred);
        let n_labels = unique_labels.len();

        let mut total_precision = 0.0;
        for &label in &unique_labels {
            let tp = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(t, p)| **t == label && **p == label)
                .count();
            let fp = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(t, p)| **t != label && **p == label)
                .count();

            let precision = if tp + fp == 0 {
                self.zero_division
            } else {
                tp as f64 / (tp + fp) as f64
            };
            total_precision += precision;
        }

        total_precision / n_labels as f64
    }

    fn compute_weighted_precision(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        let unique_labels = get_unique_labels(y_true, y_pred);
        let n_samples = y_true.len();

        let mut weighted_precision = 0.0;
        for &label in &unique_labels {
            let support = y_true.iter().filter(|&&t| t == label).count();
            let tp = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(t, p)| **t == label && **p == label)
                .count();
            let fp = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(t, p)| **t != label && **p == label)
                .count();

            let precision = if tp + fp == 0 {
                self.zero_division
            } else {
                tp as f64 / (tp + fp) as f64
            };
            weighted_precision += precision * support as f64;
        }

        weighted_precision / n_samples as f64
    }
}

impl Default for SklearnPrecision {
    fn default() -> Self {
        Self::new()
    }
}

/// Recall metric (scikit-learn compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SklearnRecall {
    /// Average type: 'binary', 'micro', 'macro', 'weighted'
    pub average: String,
    /// Positive class label for binary classification
    pub pos_label: usize,
    /// Whether to include zero division handling
    pub zero_division: f64,
}

impl SklearnRecall {
    /// Create a new sklearn-compatible recall metric
    pub fn new() -> Self {
        Self {
            average: "binary".to_string(),
            pos_label: 1,
            zero_division: 0.0,
        }
    }

    /// Set average type
    pub fn with_average(mut self, average: impl Into<String>) -> Self {
        self.average = average.into();
        self
    }

    /// Set positive label
    pub fn with_pos_label(mut self, pos_label: usize) -> Self {
        self.pos_label = pos_label;
        self
    }

    /// Set zero division handling
    pub fn with_zero_division(mut self, zero_division: f64) -> Self {
        self.zero_division = zero_division;
        self
    }

    /// Compute recall
    pub fn compute(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "y_true and y_pred must have same length"
        );

        match self.average.as_str() {
            "binary" => self.compute_binary_recall(y_true, y_pred),
            "micro" => self.compute_micro_recall(y_true, y_pred),
            "macro" => self.compute_macro_recall(y_true, y_pred),
            "weighted" => self.compute_weighted_recall(y_true, y_pred),
            _ => panic!("Unknown average type: {}", self.average),
        }
    }

    fn compute_binary_recall(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        let tp = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| **t == self.pos_label && **p == self.pos_label)
            .count();
        let fn_count = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| **t == self.pos_label && **p != self.pos_label)
            .count();

        if tp + fn_count == 0 {
            self.zero_division
        } else {
            tp as f64 / (tp + fn_count) as f64
        }
    }

    fn compute_micro_recall(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        // Micro average is the same as accuracy for multi-class
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| t == p)
            .count();
        correct as f64 / y_true.len() as f64
    }

    fn compute_macro_recall(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        let unique_labels = get_unique_labels(y_true, y_pred);
        let n_labels = unique_labels.len();

        let mut total_recall = 0.0;
        for &label in &unique_labels {
            let tp = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(t, p)| **t == label && **p == label)
                .count();
            let fn_count = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(t, p)| **t == label && **p != label)
                .count();

            let recall = if tp + fn_count == 0 {
                self.zero_division
            } else {
                tp as f64 / (tp + fn_count) as f64
            };
            total_recall += recall;
        }

        total_recall / n_labels as f64
    }

    fn compute_weighted_recall(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        let unique_labels = get_unique_labels(y_true, y_pred);
        let n_samples = y_true.len();

        let mut weighted_recall = 0.0;
        for &label in &unique_labels {
            let support = y_true.iter().filter(|&&t| t == label).count();
            let tp = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(t, p)| **t == label && **p == label)
                .count();
            let fn_count = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(t, p)| **t == label && **p != label)
                .count();

            let recall = if tp + fn_count == 0 {
                self.zero_division
            } else {
                tp as f64 / (tp + fn_count) as f64
            };
            weighted_recall += recall * support as f64;
        }

        weighted_recall / n_samples as f64
    }
}

impl Default for SklearnRecall {
    fn default() -> Self {
        Self::new()
    }
}

/// F1 score metric (scikit-learn compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SklearnF1Score {
    /// Average type: 'binary', 'micro', 'macro', 'weighted'
    pub average: String,
    /// Positive class label for binary classification
    pub pos_label: usize,
    /// Whether to include zero division handling
    pub zero_division: f64,
}

impl SklearnF1Score {
    /// Create a new sklearn-compatible F1 score metric
    pub fn new() -> Self {
        Self {
            average: "binary".to_string(),
            pos_label: 1,
            zero_division: 0.0,
        }
    }

    /// Set average type
    pub fn with_average(mut self, average: impl Into<String>) -> Self {
        self.average = average.into();
        self
    }

    /// Set positive label
    pub fn with_pos_label(mut self, pos_label: usize) -> Self {
        self.pos_label = pos_label;
        self
    }

    /// Set zero division handling
    pub fn with_zero_division(mut self, zero_division: f64) -> Self {
        self.zero_division = zero_division;
        self
    }

    /// Compute F1 score
    pub fn compute(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        let precision = SklearnPrecision::new()
            .with_average(&self.average)
            .with_pos_label(self.pos_label)
            .with_zero_division(self.zero_division)
            .compute(y_true, y_pred);

        let recall = SklearnRecall::new()
            .with_average(&self.average)
            .with_pos_label(self.pos_label)
            .with_zero_division(self.zero_division)
            .compute(y_true, y_pred);

        if precision + recall == 0.0 {
            self.zero_division
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }
}

impl Default for SklearnF1Score {
    fn default() -> Self {
        Self::new()
    }
}

/// Mean Squared Error (scikit-learn compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SklearnMeanSquaredError {
    /// Whether to compute squared=False (i.e., RMSE)
    pub squared: bool,
}

impl SklearnMeanSquaredError {
    /// Create a new sklearn-compatible MSE metric
    pub fn new() -> Self {
        Self { squared: true }
    }

    /// Set squared parameter
    pub fn with_squared(mut self, squared: bool) -> Self {
        self.squared = squared;
        self
    }

    /// Compute MSE or RMSE
    pub fn compute(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "y_true and y_pred must have same length"
        );

        let mse = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>()
            / y_true.len() as f64;

        if self.squared {
            mse
        } else {
            mse.sqrt()
        }
    }
}

impl Default for SklearnMeanSquaredError {
    fn default() -> Self {
        Self::new()
    }
}

/// Mean Absolute Error (scikit-learn compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SklearnMeanAbsoluteError {}

impl SklearnMeanAbsoluteError {
    /// Create a new sklearn-compatible MAE metric
    pub fn new() -> Self {
        Self {}
    }

    /// Compute MAE
    pub fn compute(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "y_true and y_pred must have same length"
        );

        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / y_true.len() as f64
    }
}

impl Default for SklearnMeanAbsoluteError {
    fn default() -> Self {
        Self::new()
    }
}

/// R² score (scikit-learn compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SklearnR2Score {}

impl SklearnR2Score {
    /// Create a new sklearn-compatible R² score metric
    pub fn new() -> Self {
        Self {}
    }

    /// Compute R² score
    pub fn compute(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "y_true and y_pred must have same length"
        );

        let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;

        let ss_tot = y_true.iter().map(|t| (t - mean).powi(2)).sum::<f64>();
        let ss_res = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>();

        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }
}

impl Default for SklearnR2Score {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions

/// Get unique labels from y_true and y_pred
fn get_unique_labels(y_true: &[usize], y_pred: &[usize]) -> Vec<usize> {
    let mut labels: Vec<usize> = y_true.iter().chain(y_pred.iter()).copied().collect();
    labels.sort_unstable();
    labels.dedup();
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sklearn_accuracy() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 2, 1, 0, 0, 1];

        let accuracy = SklearnAccuracy::new().compute(&y_true, &y_pred);
        assert!((accuracy - 0.3333).abs() < 0.01);

        let accuracy_unnormalized = SklearnAccuracy::new()
            .with_normalize(false)
            .compute(&y_true, &y_pred);
        assert_eq!(accuracy_unnormalized, 2.0);
    }

    #[test]
    fn test_sklearn_precision_binary() {
        let y_true = vec![0, 1, 1, 0, 1, 1];
        let y_pred = vec![0, 1, 0, 0, 1, 0];

        let precision = SklearnPrecision::new().compute(&y_true, &y_pred);
        assert_eq!(precision, 1.0); // TP=2, FP=0
    }

    #[test]
    fn test_sklearn_precision_macro() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 2, 1, 0, 0, 1];

        let precision = SklearnPrecision::new()
            .with_average("macro")
            .compute(&y_true, &y_pred);

        // Class 0: TP=2 (pred 0 & true 0 at indices 0,3), FP=1 (pred 0 & true 1 at index 4)
        //          precision = 2/(2+1) = 0.6667
        // Class 1: TP=0, FP=2 (pred 1 at indices 2,5)
        //          precision = 0.0
        // Class 2: TP=0, FP=2 (pred 2 at indices 1)
        //          precision = 0.0
        // Macro avg = (0.6667 + 0.0 + 0.0) / 3 = 0.2222...
        assert!((precision - 0.2222).abs() < 0.01);
    }

    #[test]
    fn test_sklearn_recall_binary() {
        let y_true = vec![0, 1, 1, 0, 1, 1];
        let y_pred = vec![0, 1, 0, 0, 1, 0];

        let recall = SklearnRecall::new().compute(&y_true, &y_pred);
        assert_eq!(recall, 0.5); // TP=2, FN=2
    }

    #[test]
    fn test_sklearn_f1_binary() {
        let y_true = vec![0, 1, 1, 0, 1, 1];
        let y_pred = vec![0, 1, 0, 0, 1, 0];

        let f1 = SklearnF1Score::new().compute(&y_true, &y_pred);
        // Precision = 1.0, Recall = 0.5, F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 0.6667
        assert!((f1 - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_sklearn_mse() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];

        let mse = SklearnMeanSquaredError::new().compute(&y_true, &y_pred);
        // ((3-2.5)^2 + (-0.5-0)^2 + (2-2)^2 + (7-8)^2) / 4
        // (0.25 + 0.25 + 0 + 1) / 4 = 0.375
        assert!((mse - 0.375).abs() < 1e-6);

        let rmse = SklearnMeanSquaredError::new()
            .with_squared(false)
            .compute(&y_true, &y_pred);
        assert!((rmse - 0.375_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_sklearn_mae() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];

        let mae = SklearnMeanAbsoluteError::new().compute(&y_true, &y_pred);
        // (|3-2.5| + |-0.5-0| + |2-2| + |7-8|) / 4
        // (0.5 + 0.5 + 0 + 1) / 4 = 0.5
        assert!((mae - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sklearn_r2() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];

        let r2 = SklearnR2Score::new().compute(&y_true, &y_pred);
        assert!(r2 > 0.0 && r2 < 1.0);
    }
}
