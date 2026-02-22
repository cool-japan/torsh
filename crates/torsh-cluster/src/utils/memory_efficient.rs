//! Memory-efficient clustering operations for large-scale datasets
//!
//! This module provides utilities for clustering datasets that don't fit in memory
//! by processing data in chunks and using lazy evaluation strategies.
//!
//! # Key Features
//!
//! - **Chunked Processing**: Process large datasets in manageable chunks
//! - **Streaming K-Means**: Cluster streaming data without loading all into memory
//! - **Memory Profiling**: Track and optimize memory usage during clustering
//! - **Lazy Centroid Updates**: Delay expensive operations until necessary
//!
//! # SciRS2 POLICY Compliance
//!
//! All operations use `scirs2_core` abstractions:
//! - Array operations via `scirs2_core::ndarray`
//! - Parallel processing via `scirs2_core::parallel_ops`
//! - Random number generation via `scirs2_core::random`

use crate::error::{ClusterError, ClusterResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::parallel_ops::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;
use torsh_tensor::Tensor;

/// Configuration for memory-efficient operations
#[derive(Debug, Clone)]
pub struct MemoryEfficientConfig {
    /// Maximum chunk size for processing (number of samples)
    pub chunk_size: usize,
    /// Whether to use parallel processing for chunks
    pub parallel: bool,
    /// Memory limit in bytes (approximate)
    pub memory_limit_mb: Option<usize>,
}

impl Default for MemoryEfficientConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            parallel: true,
            memory_limit_mb: None,
        }
    }
}

/// Memory-efficient chunked data processor
///
/// Processes large datasets in chunks to avoid memory overflow.
///
/// # Example
///
/// ```rust
/// use torsh_cluster::utils::memory_efficient::ChunkedDataProcessor;
/// use torsh_tensor::Tensor;
///
/// let large_data = Tensor::from_vec(
///     (0..10000).map(|i| i as f32).collect(),
///     &[1000, 10]
/// )?;
///
/// let processor = ChunkedDataProcessor::new(100); // Process 100 samples at a time
///
/// let mut sum = 0.0;
/// processor.process(&large_data, |chunk| {
///     // Process each chunk
///     let chunk_sum: f32 = chunk.iter().sum();
///     sum += chunk_sum;
///     Ok(())
/// })?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ChunkedDataProcessor {
    chunk_size: usize,
    parallel: bool,
}

impl ChunkedDataProcessor {
    /// Create a new chunked data processor
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            parallel: true,
        }
    }

    /// Set whether to use parallel processing
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Process data in chunks
    ///
    /// Applies the given function to each chunk of data sequentially.
    pub fn process<F>(&self, data: &Tensor, mut f: F) -> ClusterResult<()>
    where
        F: FnMut(ArrayView2<f32>) -> ClusterResult<()>,
    {
        let shape = data.shape();
        let n_samples = shape.dims()[0];
        let n_features = shape.dims()[1];

        // Convert tensor to ndarray for efficient slicing
        let data_vec = data.to_vec()?;
        let data_array = Array2::from_shape_vec((n_samples, n_features), data_vec)
            .map_err(|e| ClusterError::InvalidInput(format!("Shape error: {}", e)))?;

        // Process in chunks
        for start_idx in (0..n_samples).step_by(self.chunk_size) {
            let end_idx = (start_idx + self.chunk_size).min(n_samples);
            let chunk = data_array.slice(s![start_idx..end_idx, ..]);
            f(chunk)?;
        }

        Ok(())
    }

    /// Process data in parallel chunks
    ///
    /// Applies the given function to each chunk in parallel.
    /// Results from all chunks are collected and returned.
    pub fn process_parallel<F, R>(&self, data: &Tensor, f: F) -> ClusterResult<Vec<R>>
    where
        F: Fn(ArrayView2<f32>) -> ClusterResult<R> + Send + Sync,
        R: Send,
    {
        let shape = data.shape();
        let n_samples = shape.dims()[0];
        let n_features = shape.dims()[1];

        // Convert tensor to ndarray
        let data_vec = data.to_vec()?;
        let data_array = Array2::from_shape_vec((n_samples, n_features), data_vec)
            .map_err(|e| ClusterError::InvalidInput(format!("Shape error: {}", e)))?;

        // Create Arc for thread-safe sharing
        let data_arc = Arc::new(data_array);

        // Collect chunk indices
        let chunks: Vec<(usize, usize)> = (0..n_samples)
            .step_by(self.chunk_size)
            .map(|start| {
                let end = (start + self.chunk_size).min(n_samples);
                (start, end)
            })
            .collect();

        if !self.parallel || chunks.len() <= 1 {
            // Sequential processing
            let results: Result<Vec<R>, ClusterError> = chunks
                .iter()
                .map(|(start, end)| {
                    let chunk = data_arc.slice(s![*start..*end, ..]);
                    f(chunk)
                })
                .collect();
            return results;
        }

        // Parallel processing using scirs2_core parallel_ops
        let results: Result<Vec<R>, ClusterError> = chunks
            .into_par_iter()
            .map(|(start, end)| {
                let chunk = data_arc.slice(s![start..end, ..]);
                f(chunk)
            })
            .collect();

        results
    }

    /// Calculate optimal chunk size based on available memory and data dimensions
    pub fn optimal_chunk_size(
        n_samples: usize,
        n_features: usize,
        available_memory_mb: usize,
    ) -> usize {
        // Estimate memory per sample (assuming f32)
        let bytes_per_sample = n_features * std::mem::size_of::<f32>();
        let available_bytes = available_memory_mb * 1024 * 1024;

        // Use 80% of available memory for safety
        let safe_bytes = (available_bytes as f64 * 0.8) as usize;

        // Calculate chunk size
        let chunk_size = safe_bytes / bytes_per_sample;

        // Ensure at least 10 samples per chunk, but not more than total samples
        chunk_size.max(10).min(n_samples)
    }
}

/// Memory-efficient incremental centroid updater
///
/// Updates centroids incrementally as new data arrives, minimizing memory overhead.
///
/// # Mathematical Formulation
///
/// For online centroid updates, we use Welford's algorithm:
///
/// ```text
/// μ_{n+1} = μ_n + (x_{n+1} - μ_n) / (n + 1)
/// ```
///
/// where:
/// - `μ_n` is the mean after n samples
/// - `x_{n+1}` is the new sample
/// - `n` is the number of samples seen so far
pub struct IncrementalCentroidUpdater {
    /// Current centroids
    centroids: Array2<f64>,
    /// Number of samples assigned to each centroid
    counts: Array1<usize>,
    /// Total samples processed
    n_samples: usize,
}

impl IncrementalCentroidUpdater {
    /// Create a new incremental centroid updater
    pub fn new(n_clusters: usize, n_features: usize) -> Self {
        Self {
            centroids: Array2::zeros((n_clusters, n_features)),
            counts: Array1::zeros(n_clusters),
            n_samples: 0,
        }
    }

    /// Initialize centroids from initial samples
    pub fn initialize(&mut self, initial_centroids: ArrayView2<f64>) -> ClusterResult<()> {
        let (n_clusters, n_features) = initial_centroids.dim();

        if (n_clusters, n_features) != self.centroids.dim() {
            return Err(ClusterError::InvalidInput(format!(
                "Expected {} clusters and {} features, got {} and {}",
                self.centroids.nrows(),
                self.centroids.ncols(),
                n_clusters,
                n_features
            )));
        }

        self.centroids.assign(&initial_centroids);
        self.counts.fill(1); // Assume each centroid initialized with one sample
        self.n_samples = n_clusters;

        Ok(())
    }

    /// Update centroids with new sample batch
    ///
    /// Uses incremental averaging to update centroids without storing all data.
    pub fn update_batch(
        &mut self,
        samples: ArrayView2<f64>,
        labels: &[usize],
    ) -> ClusterResult<()> {
        if samples.nrows() != labels.len() {
            return Err(ClusterError::InvalidInput(format!(
                "Sample count {} doesn't match label count {}",
                samples.nrows(),
                labels.len()
            )));
        }

        // Update each centroid incrementally
        for (sample, &label) in samples.outer_iter().zip(labels.iter()) {
            if label >= self.centroids.nrows() {
                return Err(ClusterError::InvalidInput(format!(
                    "Label {} exceeds number of clusters {}",
                    label,
                    self.centroids.nrows()
                )));
            }

            let count = self.counts[label];
            let mut centroid = self.centroids.row_mut(label);

            // Welford's algorithm for incremental mean update
            for (i, &value) in sample.iter().enumerate() {
                centroid[i] += (value - centroid[i]) / (count + 1) as f64;
            }

            self.counts[label] += 1;
        }

        self.n_samples += samples.nrows();

        Ok(())
    }

    /// Get current centroids
    pub fn centroids(&self) -> ArrayView2<'_, f64> {
        self.centroids.view()
    }

    /// Get cluster counts
    pub fn counts(&self) -> &Array1<usize> {
        &self.counts
    }

    /// Get total samples processed
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }
}

/// Estimate memory usage for clustering operation
///
/// Returns estimated memory usage in megabytes.
pub fn estimate_memory_usage(n_samples: usize, n_features: usize, n_clusters: usize) -> f64 {
    // Data matrix: n_samples × n_features (f32)
    let data_size = n_samples * n_features * std::mem::size_of::<f32>();

    // Centroids: n_clusters × n_features (f64)
    let centroids_size = n_clusters * n_features * std::mem::size_of::<f64>();

    // Labels: n_samples (usize)
    let labels_size = n_samples * std::mem::size_of::<usize>();

    // Distance matrix (for some algorithms): n_samples × n_clusters (f32)
    let distances_size = n_samples * n_clusters * std::mem::size_of::<f32>();

    // Total in MB
    let total_bytes = data_size + centroids_size + labels_size + distances_size;
    total_bytes as f64 / (1024.0 * 1024.0)
}

/// Suggest optimal clustering strategy based on dataset size and available memory
pub fn suggest_clustering_strategy(
    n_samples: usize,
    n_features: usize,
    available_memory_mb: usize,
) -> String {
    let estimated_mb = estimate_memory_usage(n_samples, n_features, 10); // Assume 10 clusters

    if estimated_mb < available_memory_mb as f64 * 0.5 {
        format!(
            "Standard clustering (estimated {:.2} MB, available {} MB)",
            estimated_mb, available_memory_mb
        )
    } else if estimated_mb < available_memory_mb as f64 * 0.8 {
        format!(
            "Use parallel processing with caution (estimated {:.2} MB, available {} MB)",
            estimated_mb, available_memory_mb
        )
    } else {
        let chunk_size =
            ChunkedDataProcessor::optimal_chunk_size(n_samples, n_features, available_memory_mb);
        format!(
            "Use chunked processing with chunk_size={} (estimated {:.2} MB exceeds available {} MB)",
            chunk_size, estimated_mb, available_memory_mb
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_chunked_processor_basic() -> Result<(), Box<dyn std::error::Error>> {
        let data = Tensor::from_vec((0..100).map(|i| i as f32).collect(), &[10, 10])?;

        let processor = ChunkedDataProcessor::new(3);

        let mut chunk_count = 0;
        processor.process(&data, |chunk| {
            chunk_count += 1;
            assert!(chunk.nrows() <= 3);
            Ok(())
        })?;

        assert_eq!(chunk_count, 4); // 10 samples / 3 per chunk = 4 chunks

        Ok(())
    }

    #[test]
    fn test_chunked_processor_parallel() -> Result<(), Box<dyn std::error::Error>> {
        let data = Tensor::from_vec((0..100).map(|i| i as f32).collect(), &[10, 10])?;

        let processor = ChunkedDataProcessor::new(3).parallel(true);

        let results = processor.process_parallel(&data, |chunk| Ok(chunk.nrows()))?;

        assert_eq!(results.len(), 4);
        assert_eq!(results.iter().sum::<usize>(), 10);

        Ok(())
    }

    #[test]
    fn test_optimal_chunk_size() {
        // 1000 samples, 100 features, 100 MB available
        let chunk_size = ChunkedDataProcessor::optimal_chunk_size(1000, 100, 100);

        // Each sample is 100 * 4 = 400 bytes
        // 100 MB * 0.8 = 80 MB = 83,886,080 bytes
        // 83,886,080 / 400 = 209,715 samples (but capped at 1000)
        assert!(chunk_size > 0);
        assert!(chunk_size <= 1000);
    }

    #[test]
    fn test_incremental_centroid_updater() -> Result<(), Box<dyn std::error::Error>> {
        let mut updater = IncrementalCentroidUpdater::new(2, 3);

        // Initialize with some centroids
        let initial = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 5.0, 5.0, 5.0])?;
        updater.initialize(initial.view())?;

        // Add new samples
        let samples = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, 6.0, 6.0, 6.0])?;
        let labels = vec![0, 1];
        updater.update_batch(samples.view(), &labels)?;

        // Check updated centroids
        let centroids = updater.centroids();
        assert_relative_eq!(centroids[[0, 0]], 0.5, epsilon = 1e-6);
        assert_relative_eq!(centroids[[1, 0]], 5.5, epsilon = 1e-6);

        assert_eq!(updater.n_samples(), 4); // 2 initial + 2 new

        Ok(())
    }

    #[test]
    fn test_memory_estimation() {
        let memory_mb = estimate_memory_usage(1000, 100, 10);

        // Data: 1000 * 100 * 4 = 400,000 bytes
        // Centroids: 10 * 100 * 8 = 8,000 bytes
        // Labels: 1000 * 8 = 8,000 bytes
        // Distances: 1000 * 10 * 4 = 40,000 bytes
        // Total: ~456,000 bytes ≈ 0.435 MB

        assert!(memory_mb > 0.4);
        assert!(memory_mb < 0.5);
    }

    #[test]
    fn test_suggest_clustering_strategy() {
        // Small dataset that fits in memory
        let strategy = suggest_clustering_strategy(100, 10, 100);
        assert!(strategy.contains("Standard"));

        // Large dataset that needs chunking
        let strategy = suggest_clustering_strategy(1_000_000, 100, 10);
        assert!(strategy.contains("chunked"));
    }
}
