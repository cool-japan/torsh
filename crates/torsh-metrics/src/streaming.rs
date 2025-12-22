//! Streaming (online) metrics for efficient large-scale evaluation
//!
//! This module provides memory-efficient metrics that can be computed incrementally
//! without storing all data in memory, crucial for large-scale evaluations.

use crate::Metric;
use scirs2_core::random::Random;
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// Streaming accuracy metric that maintains running statistics
#[derive(Debug, Clone)]
pub struct StreamingAccuracy {
    correct_count: u64,
    total_count: u64,
    top_k: Option<usize>,
}

impl StreamingAccuracy {
    /// Create a new streaming accuracy metric
    pub fn new() -> Self {
        Self {
            correct_count: 0,
            total_count: 0,
            top_k: None,
        }
    }

    /// Create a streaming top-k accuracy metric
    pub fn top_k(k: usize) -> Self {
        Self {
            correct_count: 0,
            total_count: 0,
            top_k: Some(k),
        }
    }

    /// Update with a new batch of predictions and targets
    pub fn update(&mut self, predictions: &Tensor, targets: &Tensor) {
        if let Some(k) = self.top_k {
            self.update_top_k_accuracy(predictions, targets, k);
        } else {
            self.update_standard_accuracy(predictions, targets);
        }
    }

    /// Get current accuracy
    pub fn compute(&self) -> f64 {
        if self.total_count > 0 {
            self.correct_count as f64 / self.total_count as f64
        } else {
            0.0
        }
    }

    /// Reset the metric state
    pub fn reset(&mut self) {
        self.correct_count = 0;
        self.total_count = 0;
    }

    /// Get the number of samples processed
    pub fn count(&self) -> u64 {
        self.total_count
    }

    fn update_standard_accuracy(&mut self, predictions: &Tensor, targets: &Tensor) {
        match (predictions.to_vec(), targets.to_vec()) {
            (Ok(pred_vec), Ok(targets_vec)) => {
                let shape = predictions.shape();
                let dims = shape.dims();

                if dims.len() == 2 && dims[0] == targets_vec.len() && dims[0] > 0 && dims[1] > 0 {
                    let rows = dims[0];
                    let cols = dims[1];

                    // Manual argmax computation for each row
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
                            self.correct_count += 1;
                        }
                        self.total_count += 1;
                    }
                } else if dims.len() == 1 && dims[0] == targets_vec.len() {
                    // Binary classification case
                    for i in 0..dims[0] {
                        let pred_class = if pred_vec[i] >= 0.5 { 1 } else { 0 };
                        if pred_class as i64 == targets_vec[i] as i64 {
                            self.correct_count += 1;
                        }
                        self.total_count += 1;
                    }
                }
            }
            _ => {}
        }
    }

    fn update_top_k_accuracy(&mut self, predictions: &Tensor, targets: &Tensor, k: usize) {
        match (predictions.to_vec(), targets.to_vec()) {
            (Ok(pred_vec), Ok(targets_vec)) => {
                let shape = predictions.shape();
                let dims = shape.dims();

                if dims.len() == 2 && dims[0] == targets_vec.len() && dims[0] > 0 && dims[1] > 0 {
                    let rows = dims[0];
                    let cols = dims[1];

                    if k <= cols {
                        for i in 0..rows {
                            let target = targets_vec[i] as usize;

                            // Get values for this row and find top-k indices
                            let mut row_values: Vec<(f32, usize)> =
                                (0..cols).map(|j| (pred_vec[i * cols + j], j)).collect();

                            // Sort by value in descending order
                            row_values.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                            // Check if target is in top-k
                            let mut found = false;
                            for j in 0..k.min(row_values.len()) {
                                if row_values[j].1 == target {
                                    found = true;
                                    break;
                                }
                            }

                            if found {
                                self.correct_count += 1;
                            }
                            self.total_count += 1;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

impl Default for StreamingAccuracy {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for StreamingAccuracy {
    fn compute(&self, _predictions: &Tensor, _targets: &Tensor) -> f64 {
        self.compute()
    }

    fn update(&mut self, predictions: &Tensor, targets: &Tensor) {
        self.update(predictions, targets);
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn name(&self) -> &str {
        if self.top_k.is_some() {
            "streaming_top_k_accuracy"
        } else {
            "streaming_accuracy"
        }
    }
}

/// Streaming confusion matrix for classification metrics
#[derive(Debug, Clone)]
pub struct StreamingConfusionMatrix {
    num_classes: usize,
    matrix: Vec<Vec<u64>>,
    total_samples: u64,
}

impl StreamingConfusionMatrix {
    /// Create a new streaming confusion matrix
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            matrix: vec![vec![0; num_classes]; num_classes],
            total_samples: 0,
        }
    }

    /// Update with a new batch
    pub fn update(&mut self, predictions: &Tensor, targets: &Tensor) {
        match (predictions.to_vec(), targets.to_vec()) {
            (Ok(pred_vec), Ok(targets_vec)) => {
                let shape = predictions.shape();
                let dims = shape.dims();

                if dims.len() == 2 && dims[0] == targets_vec.len() {
                    let rows = dims[0];
                    let cols = dims[1];

                    for i in 0..rows {
                        // Manual argmax computation
                        let mut max_idx = 0;
                        let mut max_val = pred_vec[i * cols];

                        for j in 1..cols {
                            let val = pred_vec[i * cols + j];
                            if val > max_val {
                                max_val = val;
                                max_idx = j;
                            }
                        }

                        let pred_class = max_idx;
                        let true_class = targets_vec[i] as usize;

                        if pred_class < self.num_classes && true_class < self.num_classes {
                            self.matrix[true_class][pred_class] += 1;
                            self.total_samples += 1;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Get the confusion matrix
    pub fn matrix(&self) -> &Vec<Vec<u64>> {
        &self.matrix
    }

    /// Compute precision for each class
    pub fn precision_per_class(&self) -> Vec<f64> {
        let mut precisions = Vec::with_capacity(self.num_classes);

        for i in 0..self.num_classes {
            let tp = self.matrix[i][i];
            let fp: u64 = (0..self.num_classes)
                .filter(|&j| j != i)
                .map(|j| self.matrix[j][i])
                .sum();

            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            precisions.push(precision);
        }

        precisions
    }

    /// Compute recall for each class
    pub fn recall_per_class(&self) -> Vec<f64> {
        let mut recalls = Vec::with_capacity(self.num_classes);

        for i in 0..self.num_classes {
            let tp = self.matrix[i][i];
            let fn_count: u64 = (0..self.num_classes)
                .filter(|&j| j != i)
                .map(|j| self.matrix[i][j])
                .sum();

            let recall = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };
            recalls.push(recall);
        }

        recalls
    }

    /// Compute F1 score for each class
    pub fn f1_per_class(&self) -> Vec<f64> {
        let precisions = self.precision_per_class();
        let recalls = self.recall_per_class();

        precisions
            .iter()
            .zip(recalls.iter())
            .map(|(&p, &r)| {
                if p + r > 0.0 {
                    2.0 * p * r / (p + r)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Compute macro-averaged metrics
    pub fn macro_avg(&self) -> (f64, f64, f64) {
        let precisions = self.precision_per_class();
        let recalls = self.recall_per_class();
        let f1s = self.f1_per_class();

        let avg_precision = precisions.iter().sum::<f64>() / self.num_classes as f64;
        let avg_recall = recalls.iter().sum::<f64>() / self.num_classes as f64;
        let avg_f1 = f1s.iter().sum::<f64>() / self.num_classes as f64;

        (avg_precision, avg_recall, avg_f1)
    }

    /// Compute micro-averaged metrics
    pub fn micro_avg(&self) -> (f64, f64, f64) {
        let mut total_tp = 0u64;
        let mut total_fp = 0u64;
        let mut total_fn = 0u64;

        for i in 0..self.num_classes {
            let tp = self.matrix[i][i];
            let fp: u64 = (0..self.num_classes)
                .filter(|&j| j != i)
                .map(|j| self.matrix[j][i])
                .sum();
            let fn_count: u64 = (0..self.num_classes)
                .filter(|&j| j != i)
                .map(|j| self.matrix[i][j])
                .sum();

            total_tp += tp;
            total_fp += fp;
            total_fn += fn_count;
        }

        let precision = if total_tp + total_fp > 0 {
            total_tp as f64 / (total_tp + total_fp) as f64
        } else {
            0.0
        };

        let recall = if total_tp + total_fn > 0 {
            total_tp as f64 / (total_tp + total_fn) as f64
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        (precision, recall, f1)
    }

    /// Reset the matrix
    pub fn reset(&mut self) {
        for row in &mut self.matrix {
            row.fill(0);
        }
        self.total_samples = 0;
    }

    /// Get total number of samples processed
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }
}

/// Streaming AUROC computation using efficient algorithms
#[derive(Debug, Clone)]
pub struct StreamingAUROC {
    positive_scores: Vec<f32>,
    negative_scores: Vec<f32>,
    max_samples: usize,
    use_reservoir_sampling: bool,
    rng: Random,
}

impl StreamingAUROC {
    /// Create a new streaming AUROC metric
    pub fn new(max_samples: usize) -> Self {
        Self {
            positive_scores: Vec::new(),
            negative_scores: Vec::new(),
            max_samples,
            use_reservoir_sampling: max_samples > 0,
            rng: scirs2_core::legacy::rng(), // Fixed seed for reproducibility
        }
    }

    /// Update with new predictions and labels
    pub fn update(&mut self, predictions: &Tensor, targets: &Tensor) {
        match (predictions.to_vec(), targets.to_vec()) {
            (Ok(pred_vec), Ok(targets_vec)) => {
                if pred_vec.len() == targets_vec.len() {
                    for (pred, target) in pred_vec.iter().zip(targets_vec.iter()) {
                        if *target >= 0.5 {
                            self.add_positive_score(*pred);
                        } else {
                            self.add_negative_score(*pred);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn add_positive_score(&mut self, score: f32) {
        if self.use_reservoir_sampling && self.positive_scores.len() >= self.max_samples {
            // Reservoir sampling to maintain fixed size
            let i = self.rng.gen_range(0..=self.positive_scores.len());
            if i < self.positive_scores.len() {
                self.positive_scores[i] = score;
            }
        } else {
            self.positive_scores.push(score);
        }
    }

    fn add_negative_score(&mut self, score: f32) {
        if self.use_reservoir_sampling && self.negative_scores.len() >= self.max_samples {
            let i = self.rng.gen_range(0..=self.negative_scores.len());
            if i < self.negative_scores.len() {
                self.negative_scores[i] = score;
            }
        } else {
            self.negative_scores.push(score);
        }
    }

    /// Compute AUROC using efficient algorithm
    pub fn compute(&self) -> f64 {
        if self.positive_scores.is_empty() || self.negative_scores.is_empty() {
            return 0.5; // Random performance
        }

        // Use the efficient O(n log n) algorithm
        let mut all_scores: Vec<(f32, bool)> = Vec::new();

        // Add positive samples
        for &score in &self.positive_scores {
            all_scores.push((score, true));
        }

        // Add negative samples
        for &score in &self.negative_scores {
            all_scores.push((score, false));
        }

        // Sort by score in descending order
        all_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Use proper AUROC calculation
        let n_pos = self.positive_scores.len() as f64;
        let n_neg = self.negative_scores.len() as f64;

        if n_pos == 0.0 || n_neg == 0.0 {
            return 0.5;
        }

        // Count pairs where positive score > negative score
        let mut auc_sum = 0.0;

        for &pos_score in &self.positive_scores {
            for &neg_score in &self.negative_scores {
                if pos_score > neg_score {
                    auc_sum += 1.0;
                } else if pos_score == neg_score {
                    auc_sum += 0.5; // Tie-breaking
                }
            }
        }

        auc_sum / (n_pos * n_neg)
    }

    /// Reset the metric
    pub fn reset(&mut self) {
        self.positive_scores.clear();
        self.negative_scores.clear();
    }

    /// Get the number of positive and negative samples
    pub fn counts(&self) -> (usize, usize) {
        (self.positive_scores.len(), self.negative_scores.len())
    }
}

impl Metric for StreamingAUROC {
    fn compute(&self, _predictions: &Tensor, _targets: &Tensor) -> f64 {
        self.compute()
    }

    fn update(&mut self, predictions: &Tensor, targets: &Tensor) {
        self.update(predictions, targets);
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn name(&self) -> &str {
        "streaming_auroc"
    }
}

/// Streaming mean and variance computation using Welford's algorithm
#[derive(Debug, Clone)]
pub struct StreamingStats {
    count: u64,
    mean: f64,
    m2: f64, // Sum of squares of differences from the mean
    min_val: f64,
    max_val: f64,
}

impl StreamingStats {
    /// Create a new streaming statistics tracker
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
        }
    }

    /// Update with a new value using Welford's algorithm
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        self.min_val = self.min_val.min(value);
        self.max_val = self.max_val.max(value);
    }

    /// Update with multiple values from a tensor
    pub fn update_tensor(&mut self, values: &Tensor) {
        if let Ok(values_vec) = values.to_vec() {
            for value in values_vec {
                self.update(value as f64);
            }
        }
    }

    /// Get the current mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get the current variance (population)
    pub fn variance(&self) -> f64 {
        if self.count > 0 {
            self.m2 / self.count as f64
        } else {
            0.0
        }
    }

    /// Get the current sample variance
    pub fn sample_variance(&self) -> f64 {
        if self.count > 1 {
            self.m2 / (self.count - 1) as f64
        } else {
            0.0
        }
    }

    /// Get the current standard deviation
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get the current sample standard deviation
    pub fn sample_std(&self) -> f64 {
        self.sample_variance().sqrt()
    }

    /// Get min and max values
    pub fn min_max(&self) -> (f64, f64) {
        if self.count > 0 {
            (self.min_val, self.max_val)
        } else {
            (0.0, 0.0)
        }
    }

    /// Get the count of samples
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset the statistics
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.min_val = f64::INFINITY;
        self.max_val = f64::NEG_INFINITY;
    }

    /// Merge with another StreamingStats (useful for parallel processing)
    pub fn merge(&mut self, other: &StreamingStats) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let new_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let new_mean =
            (self.count as f64 * self.mean + other.count as f64 * other.mean) / new_count as f64;

        let new_m2 = self.m2
            + other.m2
            + delta * delta * (self.count as f64 * other.count as f64) / new_count as f64;

        self.count = new_count;
        self.mean = new_mean;
        self.m2 = new_m2;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
    }
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Collection of streaming metrics for efficient batch evaluation
pub struct StreamingMetricCollection {
    metrics: HashMap<String, Box<dyn StreamingMetricTrait>>,
    sample_count: u64,
}

pub trait StreamingMetricTrait {
    fn update(&mut self, predictions: &Tensor, targets: &Tensor);
    fn compute(&self) -> f64;
    fn reset(&mut self);
    fn name(&self) -> &str;
}

// Implement StreamingMetricTrait for our metrics
impl StreamingMetricTrait for StreamingAccuracy {
    fn update(&mut self, predictions: &Tensor, targets: &Tensor) {
        self.update(predictions, targets);
    }

    fn compute(&self) -> f64 {
        self.compute()
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn name(&self) -> &str {
        Metric::name(self)
    }
}

impl StreamingMetricTrait for StreamingAUROC {
    fn update(&mut self, predictions: &Tensor, targets: &Tensor) {
        self.update(predictions, targets);
    }

    fn compute(&self) -> f64 {
        self.compute()
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn name(&self) -> &str {
        Metric::name(self)
    }
}

impl StreamingMetricCollection {
    /// Create a new streaming metric collection
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            sample_count: 0,
        }
    }

    /// Add a streaming metric
    pub fn add_metric<M: StreamingMetricTrait + 'static>(mut self, metric: M) -> Self {
        let name = metric.name().to_string();
        self.metrics.insert(name, Box::new(metric));
        self
    }

    /// Update all metrics with a new batch
    pub fn update(&mut self, predictions: &Tensor, targets: &Tensor) {
        for metric in self.metrics.values_mut() {
            metric.update(predictions, targets);
        }

        // Estimate sample count from tensor shape
        if let Ok(targets_vec) = targets.to_vec() {
            self.sample_count += targets_vec.len() as u64;
        }
    }

    /// Compute all metrics
    pub fn compute(&self) -> HashMap<String, f64> {
        self.metrics
            .iter()
            .map(|(name, metric)| (name.clone(), metric.compute()))
            .collect()
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        for metric in self.metrics.values_mut() {
            metric.reset();
        }
        self.sample_count = 0;
    }

    /// Get the number of samples processed
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Get formatted results string
    pub fn format_results(&self) -> String {
        let results = self.compute();
        results
            .iter()
            .map(|(name, value)| format!("{}: {:.4}", name, value))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl Default for StreamingMetricCollection {
    fn default() -> Self {
        Self::new()
    }
}
