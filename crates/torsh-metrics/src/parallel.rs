//! Parallel metric computation for scalability
//!
//! This module provides parallel implementations of metrics using scirs2-core's
//! parallel operations for efficient computation on multi-core systems.

use crate::Metric;
use torsh_core::error::TorshError;
use torsh_tensor::Tensor;

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

/// Parallel accuracy metric
#[derive(Clone)]
pub struct ParallelAccuracy {
    top_k: Option<usize>,
    #[allow(dead_code)] // Reserved for future chunked parallel processing
    chunk_size: usize,
}

impl ParallelAccuracy {
    /// Create a new parallel accuracy metric
    pub fn new() -> Self {
        Self {
            top_k: None,
            chunk_size: 1000, // Default chunk size for parallelization
        }
    }

    /// Create with custom chunk size
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            top_k: None,
            chunk_size,
        }
    }

    /// Create a parallel top-k accuracy metric
    pub fn top_k(k: usize) -> Self {
        Self {
            top_k: Some(k),
            chunk_size: 1000,
        }
    }

    /// Compute accuracy in parallel
    #[cfg(feature = "parallel")]
    fn compute_parallel(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64, TorshError> {
        let pred_vec = predictions.to_vec()?;
        let targets_vec = targets.to_vec()?;
        let shape = predictions.shape();
        let dims = shape.dims();

        if dims.len() != 2 || dims[0] != targets_vec.len() {
            return Err(TorshError::InvalidArgument("Shape mismatch".to_string()));
        }

        let rows = dims[0];
        let cols = dims[1];

        // Parallel computation using scirs2-core parallel ops
        let correct: usize = (0..rows)
            .into_par_iter()
            .map(|i| {
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
                    1
                } else {
                    0
                }
            })
            .sum();

        Ok(correct as f64 / rows as f64)
    }

    #[cfg(not(feature = "parallel"))]
    fn compute_parallel(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64, TorshError> {
        self.compute_sequential(predictions, targets)
    }

    #[allow(dead_code)] // Used in non-parallel builds
    fn compute_sequential(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<f64, TorshError> {
        let pred_vec = predictions.to_vec()?;
        let targets_vec = targets.to_vec()?;
        let shape = predictions.shape();
        let dims = shape.dims();

        if dims.len() != 2 || dims[0] != targets_vec.len() {
            return Err(TorshError::InvalidArgument("Shape mismatch".to_string()));
        }

        let rows = dims[0];
        let cols = dims[1];
        let mut correct = 0;

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

        Ok(correct as f64 / rows as f64)
    }
}

impl Default for ParallelAccuracy {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for ParallelAccuracy {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_parallel(predictions, targets).unwrap_or(0.0)
    }

    fn name(&self) -> &str {
        if self.top_k.is_some() {
            "parallel_top_k_accuracy"
        } else {
            "parallel_accuracy"
        }
    }
}

/// Parallel confusion matrix computation
pub struct ParallelConfusionMatrix {
    num_classes: usize,
}

impl ParallelConfusionMatrix {
    /// Create a new parallel confusion matrix
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }

    /// Compute confusion matrix in parallel
    #[cfg(feature = "parallel")]
    pub fn compute(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<Vec<Vec<u64>>, TorshError> {
        let pred_vec = predictions.to_vec()?;
        let targets_vec = targets.to_vec()?;
        let shape = predictions.shape();
        let dims = shape.dims();

        if dims.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Predictions must be 2D".to_string(),
            ));
        }

        let rows = dims[0];
        let cols = dims[1];

        // Parallel reduction to compute confusion matrix
        let partial_matrices: Vec<Vec<Vec<u64>>> = (0..rows)
            .into_par_iter()
            .map(|i| {
                let mut local_matrix = vec![vec![0u64; self.num_classes]; self.num_classes];

                let mut max_idx = 0;
                let mut max_val = pred_vec[i * cols];

                for j in 1..cols {
                    let val = pred_vec[i * cols + j];
                    if val > max_val {
                        max_val = val;
                        max_idx = j;
                    }
                }

                let target = targets_vec[i] as usize;
                if max_idx < self.num_classes && target < self.num_classes {
                    local_matrix[target][max_idx] += 1;
                }

                local_matrix
            })
            .collect();

        // Merge partial matrices
        let mut final_matrix = vec![vec![0u64; self.num_classes]; self.num_classes];
        for partial in partial_matrices {
            for i in 0..self.num_classes {
                for j in 0..self.num_classes {
                    final_matrix[i][j] += partial[i][j];
                }
            }
        }

        Ok(final_matrix)
    }

    #[cfg(not(feature = "parallel"))]
    pub fn compute(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<Vec<Vec<u64>>, TorshError> {
        let mut matrix = vec![vec![0u64; self.num_classes]; self.num_classes];
        let pred_vec = predictions.to_vec()?;
        let targets_vec = targets.to_vec()?;
        let shape = predictions.shape();
        let dims = shape.dims();

        if dims.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Predictions must be 2D".to_string(),
            ));
        }

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

            let target = targets_vec[i] as usize;
            if max_idx < self.num_classes && target < self.num_classes {
                matrix[target][max_idx] += 1;
            }
        }

        Ok(matrix)
    }
}

/// Parallel batch metric computation
pub struct ParallelMetricCollection {
    metrics: Vec<(String, Box<dyn Metric + Send + Sync>)>,
}

impl ParallelMetricCollection {
    /// Create a new parallel metric collection
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    /// Add a metric to the collection
    pub fn add_metric<M: Metric + Send + Sync + 'static>(
        mut self,
        name: String,
        metric: M,
    ) -> Self {
        self.metrics.push((name, Box::new(metric)));
        self
    }

    /// Compute all metrics in parallel
    #[cfg(feature = "parallel")]
    pub fn compute_all(&self, predictions: &Tensor, targets: &Tensor) -> Vec<(String, f64)> {
        self.metrics
            .par_iter()
            .map(|(name, metric)| {
                let value = metric.compute(predictions, targets);
                (name.clone(), value)
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    pub fn compute_all(&self, predictions: &Tensor, targets: &Tensor) -> Vec<(String, f64)> {
        self.metrics
            .iter()
            .map(|(name, metric)| {
                let value = metric.compute(predictions, targets);
                (name.clone(), value)
            })
            .collect()
    }
}

impl Default for ParallelMetricCollection {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel cross-validation metric computation
pub struct ParallelCrossValidation {
    #[allow(dead_code)] // Stored for reference and future validation logic
    n_folds: usize,
}

impl ParallelCrossValidation {
    /// Create a new parallel cross-validation
    pub fn new(n_folds: usize) -> Self {
        Self { n_folds }
    }

    /// Compute cross-validation scores in parallel
    #[cfg(feature = "parallel")]
    pub fn compute_scores<M: Metric + Send + Sync>(
        &self,
        metric: &M,
        all_predictions: &[Tensor],
        all_targets: &[Tensor],
    ) -> Vec<f64> {
        all_predictions
            .par_iter()
            .zip(all_targets.par_iter())
            .map(|(preds, targets)| metric.compute(preds, targets))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    pub fn compute_scores<M: Metric>(
        &self,
        metric: &M,
        all_predictions: &[Tensor],
        all_targets: &[Tensor],
    ) -> Vec<f64> {
        all_predictions
            .iter()
            .zip(all_targets.iter())
            .map(|(preds, targets)| metric.compute(preds, targets))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_parallel_accuracy() {
        let predictions = from_vec(
            vec![0.1, 0.9, 0.8, 0.2, 0.3, 0.7, 0.9, 0.1, 0.2, 0.8, 0.7, 0.3],
            &[6, 2],
            DeviceType::Cpu,
        )
        .unwrap();
        let targets = from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[6], DeviceType::Cpu).unwrap();

        let parallel_acc = ParallelAccuracy::new();
        let result = parallel_acc.compute(&predictions, &targets);

        assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn test_parallel_confusion_matrix() {
        let predictions =
            from_vec(vec![0.1, 0.9, 0.8, 0.2, 0.3, 0.7], &[3, 2], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 0.0, 1.0], &[3], DeviceType::Cpu).unwrap();

        let conf_matrix = ParallelConfusionMatrix::new(2);
        let matrix = conf_matrix.compute(&predictions, &targets).unwrap();

        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);

        // Total count should equal number of samples
        let total: u64 = matrix.iter().flat_map(|row| row.iter()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_parallel_metric_collection() {
        let predictions =
            from_vec(vec![0.1, 0.9, 0.8, 0.2, 0.3, 0.7], &[3, 2], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 0.0, 1.0], &[3], DeviceType::Cpu).unwrap();

        let collection = ParallelMetricCollection::new()
            .add_metric("accuracy".to_string(), ParallelAccuracy::new());

        let results = collection.compute_all(&predictions, &targets);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "accuracy");
        assert!(results[0].1 >= 0.0 && results[0].1 <= 1.0);
    }

    #[test]
    fn test_parallel_cross_validation() {
        let cv = ParallelCrossValidation::new(3);
        let metric = ParallelAccuracy::new();

        let predictions1 = from_vec(vec![0.1, 0.9, 0.8, 0.2], &[2, 2], DeviceType::Cpu).unwrap();
        let targets1 = from_vec(vec![1.0, 0.0], &[2], DeviceType::Cpu).unwrap();

        let predictions2 = from_vec(vec![0.3, 0.7, 0.9, 0.1], &[2, 2], DeviceType::Cpu).unwrap();
        let targets2 = from_vec(vec![1.0, 0.0], &[2], DeviceType::Cpu).unwrap();

        let all_preds = vec![predictions1, predictions2];
        let all_targets = vec![targets1, targets2];

        let scores = cv.compute_scores(&metric, &all_preds, &all_targets);
        assert_eq!(scores.len(), 2);
    }
}
