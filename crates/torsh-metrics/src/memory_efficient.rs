//! Memory-efficient large dataset evaluation
//!
//! This module provides streaming and chunked evaluation for datasets
//! that don't fit in memory.

use torsh_core::error::TorshError;
use torsh_tensor::Tensor;

/// Chunk-based metric evaluator for large datasets
pub struct ChunkedEvaluator {
    #[allow(dead_code)] // Reserved for future chunked processing optimization
    chunk_size: usize,
    metrics: Vec<Box<dyn StreamingMetric>>,
}

impl ChunkedEvaluator {
    /// Create a new chunked evaluator
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            metrics: Vec::new(),
        }
    }

    /// Add a streaming metric
    pub fn add_metric<M: StreamingMetric + 'static>(mut self, metric: M) -> Self {
        self.metrics.push(Box::new(metric));
        self
    }

    /// Evaluate in chunks
    pub fn evaluate(
        &mut self,
        predictions: &[Tensor],
        targets: &[Tensor],
    ) -> Result<Vec<(String, f64)>, TorshError> {
        if predictions.len() != targets.len() {
            return Err(TorshError::InvalidArgument(
                "Predictions and targets must have same length".to_string(),
            ));
        }

        // Reset all metrics
        for metric in &mut self.metrics {
            metric.reset();
        }

        // Process chunks
        for (pred, targ) in predictions.iter().zip(targets.iter()) {
            for metric in &mut self.metrics {
                metric.update(pred, targ);
            }
        }

        // Compute final results
        Ok(self
            .metrics
            .iter()
            .map(|m| (m.name().to_string(), m.compute()))
            .collect())
    }
}

/// Trait for streaming/incremental metrics
pub trait StreamingMetric {
    /// Update with new batch
    fn update(&mut self, predictions: &Tensor, targets: &Tensor);

    /// Compute current metric value
    fn compute(&self) -> f64;

    /// Reset internal state
    fn reset(&mut self);

    /// Get metric name
    fn name(&self) -> &str;
}

/// Memory-efficient accuracy metric
pub struct MemoryEfficientAccuracy {
    correct: u64,
    total: u64,
}

impl MemoryEfficientAccuracy {
    /// Create a new memory-efficient accuracy
    pub fn new() -> Self {
        Self {
            correct: 0,
            total: 0,
        }
    }

    /// Get count statistics
    pub fn counts(&self) -> (u64, u64) {
        (self.correct, self.total)
    }
}

impl Default for MemoryEfficientAccuracy {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingMetric for MemoryEfficientAccuracy {
    fn update(&mut self, predictions: &Tensor, targets: &Tensor) {
        if let (Ok(pred_vec), Ok(targ_vec)) = (predictions.to_vec(), targets.to_vec()) {
            let shape = predictions.shape();
            let dims = shape.dims();

            if dims.len() == 2 && dims[0] == targ_vec.len() {
                let rows = dims[0];
                let cols = dims[1];

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

                    if max_idx as i64 == targ_vec[i] as i64 {
                        self.correct += 1;
                    }
                    self.total += 1;
                }
            }
        }
    }

    fn compute(&self) -> f64 {
        if self.total > 0 {
            self.correct as f64 / self.total as f64
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    fn name(&self) -> &str {
        "memory_efficient_accuracy"
    }
}

/// Memory-efficient MSE metric
pub struct MemoryEfficientMSE {
    sum_squared_error: f64,
    count: u64,
}

impl MemoryEfficientMSE {
    /// Create a new memory-efficient MSE
    pub fn new() -> Self {
        Self {
            sum_squared_error: 0.0,
            count: 0,
        }
    }
}

impl Default for MemoryEfficientMSE {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingMetric for MemoryEfficientMSE {
    fn update(&mut self, predictions: &Tensor, targets: &Tensor) {
        if let (Ok(pred_vec), Ok(targ_vec)) = (predictions.to_vec(), targets.to_vec()) {
            if pred_vec.len() == targ_vec.len() {
                for (p, t) in pred_vec.iter().zip(targ_vec.iter()) {
                    let error = (*p as f64) - (*t as f64);
                    self.sum_squared_error += error * error;
                    self.count += 1;
                }
            }
        }
    }

    fn compute(&self) -> f64 {
        if self.count > 0 {
            self.sum_squared_error / self.count as f64
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        self.sum_squared_error = 0.0;
        self.count = 0;
    }

    fn name(&self) -> &str {
        "memory_efficient_mse"
    }
}

/// Memory-efficient MAE metric
pub struct MemoryEfficientMAE {
    sum_absolute_error: f64,
    count: u64,
}

impl MemoryEfficientMAE {
    /// Create a new memory-efficient MAE
    pub fn new() -> Self {
        Self {
            sum_absolute_error: 0.0,
            count: 0,
        }
    }
}

impl Default for MemoryEfficientMAE {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingMetric for MemoryEfficientMAE {
    fn update(&mut self, predictions: &Tensor, targets: &Tensor) {
        if let (Ok(pred_vec), Ok(targ_vec)) = (predictions.to_vec(), targets.to_vec()) {
            if pred_vec.len() == targ_vec.len() {
                for (p, t) in pred_vec.iter().zip(targ_vec.iter()) {
                    let error = (*p as f64) - (*t as f64);
                    self.sum_absolute_error += error.abs();
                    self.count += 1;
                }
            }
        }
    }

    fn compute(&self) -> f64 {
        if self.count > 0 {
            self.sum_absolute_error / self.count as f64
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        self.sum_absolute_error = 0.0;
        self.count = 0;
    }

    fn name(&self) -> &str {
        "memory_efficient_mae"
    }
}

/// Online confusion matrix for classification
pub struct OnlineConfusionMatrix {
    num_classes: usize,
    matrix: Vec<Vec<u64>>,
}

impl OnlineConfusionMatrix {
    /// Create a new online confusion matrix
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            matrix: vec![vec![0u64; num_classes]; num_classes],
        }
    }

    /// Get the confusion matrix
    pub fn matrix(&self) -> &Vec<Vec<u64>> {
        &self.matrix
    }

    /// Compute precision for each class
    pub fn precision_per_class(&self) -> Vec<f64> {
        let mut precisions = Vec::with_capacity(self.num_classes);

        for class in 0..self.num_classes {
            let mut tp = 0u64;
            let mut fp = 0u64;

            for i in 0..self.num_classes {
                if i == class {
                    tp = self.matrix[i][class];
                } else {
                    fp += self.matrix[i][class];
                }
            }

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

        for class in 0..self.num_classes {
            let row_sum: u64 = self.matrix[class].iter().sum();

            let recall = if row_sum > 0 {
                self.matrix[class][class] as f64 / row_sum as f64
            } else {
                0.0
            };

            recalls.push(recall);
        }

        recalls
    }
}

impl StreamingMetric for OnlineConfusionMatrix {
    fn update(&mut self, predictions: &Tensor, targets: &Tensor) {
        if let (Ok(pred_vec), Ok(targ_vec)) = (predictions.to_vec(), targets.to_vec()) {
            let shape = predictions.shape();
            let dims = shape.dims();

            if dims.len() == 2 && dims[0] == targ_vec.len() {
                let rows = dims[0];
                let cols = dims[1];

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

                    let target = targ_vec[i] as usize;
                    if max_idx < self.num_classes && target < self.num_classes {
                        self.matrix[target][max_idx] += 1;
                    }
                }
            }
        }
    }

    fn compute(&self) -> f64 {
        // Return overall accuracy
        let mut correct = 0u64;
        let mut total = 0u64;

        for i in 0..self.num_classes {
            correct += self.matrix[i][i];
            for j in 0..self.num_classes {
                total += self.matrix[i][j];
            }
        }

        if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        for row in &mut self.matrix {
            for cell in row {
                *cell = 0;
            }
        }
    }

    fn name(&self) -> &str {
        "online_confusion_matrix"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_memory_efficient_accuracy() {
        let mut acc = MemoryEfficientAccuracy::new();

        let preds1 = from_vec(vec![0.1, 0.9, 0.8, 0.2], &[2, 2], DeviceType::Cpu).unwrap();
        let targs1 = from_vec(vec![1.0, 0.0], &[2], DeviceType::Cpu).unwrap();

        acc.update(&preds1, &targs1);

        let result = acc.compute();
        assert!((result - 1.0).abs() < 1e-6);

        let (correct, total) = acc.counts();
        assert_eq!(total, 2);
        assert_eq!(correct, 2);
    }

    #[test]
    fn test_memory_efficient_mse() {
        let mut mse = MemoryEfficientMSE::new();

        let preds = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu).unwrap();
        let targs = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu).unwrap();

        mse.update(&preds, &targs);

        let result = mse.compute();
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn test_chunked_evaluator() {
        let mut evaluator = ChunkedEvaluator::new(100)
            .add_metric(MemoryEfficientAccuracy::new())
            .add_metric(MemoryEfficientMSE::new());

        let preds1 = from_vec(vec![0.1, 0.9, 0.8, 0.2], &[2, 2], DeviceType::Cpu).unwrap();
        let targs1 = from_vec(vec![1.0, 0.0], &[2], DeviceType::Cpu).unwrap();

        let preds2 = from_vec(vec![0.3, 0.7, 0.9, 0.1], &[2, 2], DeviceType::Cpu).unwrap();
        let targs2 = from_vec(vec![1.0, 0.0], &[2], DeviceType::Cpu).unwrap();

        let results = evaluator
            .evaluate(&vec![preds1, preds2], &vec![targs1, targs2])
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_online_confusion_matrix() {
        let mut conf = OnlineConfusionMatrix::new(2);

        let preds = from_vec(vec![0.1, 0.9, 0.8, 0.2, 0.3, 0.7], &[3, 2], DeviceType::Cpu).unwrap();
        let targs = from_vec(vec![1.0, 0.0, 1.0], &[3], DeviceType::Cpu).unwrap();

        conf.update(&preds, &targs);

        let matrix = conf.matrix();
        assert_eq!(matrix.len(), 2);

        let precisions = conf.precision_per_class();
        assert_eq!(precisions.len(), 2);

        let recalls = conf.recall_per_class();
        assert_eq!(recalls.len(), 2);
    }
}
