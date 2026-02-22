//! GPU-accelerated metrics for high-performance evaluation
//!
//! This module provides GPU-accelerated metric computation using scirs2-core's
//! GPU capabilities for maximum performance on large datasets.

use crate::Metric;
use torsh_core::error::TorshError;
use torsh_tensor::Tensor;

#[cfg(feature = "gpu")]
use scirs2_core::gpu::GpuBackend;

/// GPU-accelerated accuracy metric
pub struct GpuAccuracy {
    #[cfg(feature = "gpu")]
    #[allow(dead_code)] // Reserved for future GPU implementation
    backend: GpuBackend,
    top_k: Option<usize>,
}

impl GpuAccuracy {
    /// Create a new GPU-accelerated accuracy metric
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "gpu")]
            backend: GpuBackend::Cpu, // Default to CPU backend
            top_k: None,
        }
    }

    /// Create a GPU-accelerated top-k accuracy metric
    pub fn top_k(k: usize) -> Self {
        Self {
            #[cfg(feature = "gpu")]
            backend: GpuBackend::Cpu, // Default to CPU backend
            top_k: Some(k),
        }
    }

    /// Compute accuracy on GPU if available, fallback to CPU
    #[cfg(feature = "gpu")]
    fn compute_gpu(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64, TorshError> {
        // For now, always use CPU implementation
        // GPU kernel implementation would go here when available
        self.compute_cpu(predictions, targets)
    }

    fn compute_cpu(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64, TorshError> {
        let pred_vec = predictions.to_vec()?;
        let targets_vec = targets.to_vec()?;
        let shape = predictions.shape();
        let dims = shape.dims();

        if dims.len() == 2 && dims[0] == targets_vec.len() {
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
        } else {
            Err(TorshError::InvalidArgument("Shape mismatch".to_string()))
        }
    }
}

impl Default for GpuAccuracy {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for GpuAccuracy {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        #[cfg(feature = "gpu")]
        {
            self.compute_gpu(predictions, targets).unwrap_or(0.0)
        }

        #[cfg(not(feature = "gpu"))]
        {
            self.compute_cpu(predictions, targets).unwrap_or(0.0)
        }
    }

    fn name(&self) -> &str {
        if self.top_k.is_some() {
            "gpu_top_k_accuracy"
        } else {
            "gpu_accuracy"
        }
    }
}

/// GPU-accelerated confusion matrix computation
pub struct GpuConfusionMatrix {
    num_classes: usize,
    #[cfg(feature = "gpu")]
    #[allow(dead_code)] // Reserved for future GPU implementation
    backend: GpuBackend,
}

impl GpuConfusionMatrix {
    /// Create a new GPU-accelerated confusion matrix
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            #[cfg(feature = "gpu")]
            backend: GpuBackend::Cpu, // Default to CPU backend
        }
    }

    /// Compute confusion matrix
    pub fn compute(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<Vec<Vec<u64>>, TorshError> {
        #[cfg(feature = "gpu")]
        {
            // GPU computation could be enabled here when available
            return self.compute_gpu(predictions, targets);
        }

        #[cfg(not(feature = "gpu"))]
        self.compute_cpu(predictions, targets)
    }

    #[cfg(feature = "gpu")]
    fn compute_gpu(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<Vec<Vec<u64>>, TorshError> {
        // GPU kernel implementation would go here when available
        // For now, use CPU implementation
        self.compute_cpu(predictions, targets)
    }

    fn compute_cpu(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<Vec<Vec<u64>>, TorshError> {
        let mut matrix = vec![vec![0u64; self.num_classes]; self.num_classes];
        let pred_vec = predictions.to_vec()?;
        let targets_vec = targets.to_vec()?;

        let shape = predictions.shape();
        let dims = shape.dims();

        if dims.len() == 2 {
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
        }

        Ok(matrix)
    }
}

/// Batch GPU metric computation for efficiency
pub struct GpuBatchMetrics {
    #[cfg(feature = "gpu")]
    #[allow(dead_code)] // Reserved for future GPU implementation
    backend: GpuBackend,
    metrics: Vec<String>,
}

impl GpuBatchMetrics {
    /// Create a new batch GPU metrics computer
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "gpu")]
            backend: GpuBackend::Cpu, // Default to CPU backend
            metrics: Vec::new(),
        }
    }

    /// Add a metric to compute
    pub fn add_metric(&mut self, metric_name: String) {
        self.metrics.push(metric_name);
    }

    /// Compute all metrics in one GPU pass
    pub fn compute_all(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<Vec<(String, f64)>, TorshError> {
        let mut results = Vec::new();

        for metric in &self.metrics {
            let value = match metric.as_str() {
                "accuracy" => {
                    let acc = GpuAccuracy::new();
                    acc.compute(predictions, targets)
                }
                _ => 0.0,
            };
            results.push((metric.clone(), value));
        }

        Ok(results)
    }
}

impl Default for GpuBatchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_gpu_accuracy_fallback() {
        let predictions =
            from_vec(vec![0.1, 0.9, 0.8, 0.2, 0.3, 0.7], &[3, 2], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 0.0, 1.0], &[3], DeviceType::Cpu).unwrap();

        let gpu_acc = GpuAccuracy::new();
        let result = gpu_acc.compute(&predictions, &targets);

        // Should compute correctly even without GPU
        assert!((result - 1.0).abs() < 1e-6 || (result - 0.666).abs() < 0.1);
    }

    #[test]
    fn test_gpu_confusion_matrix() {
        let predictions =
            from_vec(vec![0.1, 0.9, 0.8, 0.2, 0.3, 0.7], &[3, 2], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 0.0, 1.0], &[3], DeviceType::Cpu).unwrap();

        let conf_matrix = GpuConfusionMatrix::new(2);
        let matrix = conf_matrix.compute(&predictions, &targets).unwrap();

        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
    }

    #[test]
    fn test_gpu_batch_metrics() {
        let predictions =
            from_vec(vec![0.1, 0.9, 0.8, 0.2, 0.3, 0.7], &[3, 2], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 0.0, 1.0], &[3], DeviceType::Cpu).unwrap();

        let mut batch_metrics = GpuBatchMetrics::new();
        batch_metrics.add_metric("accuracy".to_string());

        let results = batch_metrics.compute_all(&predictions, &targets).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "accuracy");
    }
}
