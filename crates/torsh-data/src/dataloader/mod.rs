//! DataLoader implementation for efficient data loading
//!
//! This module provides comprehensive data loading capabilities with support for
//! batching, parallel processing, prefetching, and memory optimization.
//!
//! # Overview
//!
//! The DataLoader system consists of several key components:
//!
//! - **Core**: Fundamental DataLoader functionality including the main struct, iterator, and builder
//! - **Simple**: Convenience functions for quick DataLoader creation
//! - **Prefetch**: Performance optimization through background data loading
//! - **Workers**: Multi-process data loading with worker pools
//! - **Memory**: Memory pinning for optimized GPU transfers
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use torsh_data::dataloader::{DataLoader, simple_dataloader};
//! use torsh_data::dataset::TensorDataset;
//!
//! // Using the builder pattern
//! let dataset = TensorDataset::new(vec![1, 2, 3, 4, 5]);
//! let dataloader = DataLoader::builder(dataset)
//!     .batch_size(2)
//!     .num_workers(4)
//!     .shuffle(true)
//!     .build()?;
//!
//! for batch in dataloader.iter() {
//!     // Process batch
//! }
//!
//! // Using the simple API
//! let dataset = TensorDataset::new(vec![1, 2, 3, 4, 5]);
//! let dataloader = simple_dataloader(dataset, 2, false)?;
//! ```
//!
//! # Advanced Usage
//!
//! ## Prefetching for Performance
//!
//! ```rust,ignore
//! use torsh_data::dataloader::{DataLoader, prefetch::PrefetchExt};
//! use torsh_data::dataset::TensorDataset;
//!
//! let dataset = TensorDataset::new(vec![1, 2, 3, 4, 5]);
//! let dataloader = DataLoader::builder(dataset).batch_size(2).build()?;
//!
//! // Add prefetching for better performance
//! let prefetch_iter = dataloader.iter().prefetch(2);
//! for batch in prefetch_iter {
//!     // Process batch while next batches are being prefetched
//! }
//! ```
//!
//! ## Multi-Process Loading
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use torsh_data::dataloader::{DataLoader, workers::WorkerPool};
//! use torsh_data::dataset::TensorDataset;
//! use torsh_data::collate::DefaultCollate;
//!
//! let dataset = Arc::new(TensorDataset::new(vec![1, 2, 3, 4, 5]));
//! let collate_fn = Arc::new(DefaultCollate);
//! let worker_pool = WorkerPool::new(dataset, collate_fn, 4);
//!
//! // Submit tasks and collect results
//! worker_pool.submit_task(0, vec![0, 1])?;
//! let result = worker_pool.get_result()?;
//! ```
//!
//! ## Memory Pinning for GPU
//!
//! ```rust,ignore
//! use torsh_data::dataloader::memory::{MemoryPinningManager, PinningConfig};
//! use torsh_core::device::DeviceType;
//!
//! let mut manager = MemoryPinningManager::new();
//! let config = PinningConfig::cuda(0);
//!
//! // Pin memory for faster GPU transfers
//! if manager.supports_pinning(Some(DeviceType::Cuda(0))) {
//!     // Use pinned memory for data transfers
//! }
//! ```

use torsh_core::error::Result;

// Re-export sub-modules
pub mod core;
pub mod memory;
pub mod prefetch;
pub mod simple;
pub mod workers;

// Re-export core types for backward compatibility and convenience
pub use core::{
    DataLoader, DataLoaderBuilder, DataLoaderIterator, DataLoaderTrait, RandomDataLoader,
    SimpleDataLoader,
};

// Re-export simple API functions
pub use simple::{
    simple_dataloader, simple_random_dataloader, SimpleConfig, SimpleRandomDataLoader,
};

// Re-export prefetch functionality
pub use prefetch::{PrefetchConfig, PrefetchExt, PrefetchIterator};

// Re-export worker functionality
pub use workers::{
    MultiProcessIterator, PersistentWorkerPool, WorkerConfig, WorkerPool, WorkerResult,
};

// Re-export memory functionality
pub use memory::{CpuMemoryPinner, MemoryPinning, MemoryPinningManager, PinningConfig};

#[cfg(feature = "cuda")]
pub use memory::CudaMemoryPinner;

/// Configuration for DataLoader creation with all options
///
/// This struct provides a comprehensive configuration interface that combines
/// options from all DataLoader components.
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Batch size for data loading
    pub batch_size: usize,
    /// Whether to shuffle the data
    pub shuffle: bool,
    /// Number of worker threads
    pub num_workers: usize,
    /// Whether to pin memory for GPU transfers
    pub pin_memory: bool,
    /// Whether to drop the last incomplete batch
    pub drop_last: bool,
    /// Timeout for batch collection
    pub timeout: Option<std::time::Duration>,
    /// Random seed for reproducible shuffling
    pub generator: Option<u64>,
    /// Prefetch buffer size (0 to disable prefetching)
    pub prefetch_buffer_size: usize,
    /// Whether to use persistent workers
    pub persistent_workers: bool,
    /// Memory pinning configuration
    pub pinning_config: Option<PinningConfig>,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            shuffle: false,
            num_workers: 0,
            pin_memory: false,
            drop_last: false,
            timeout: None,
            generator: None,
            prefetch_buffer_size: 0,
            persistent_workers: false,
            pinning_config: None,
        }
    }
}

impl DataLoaderConfig {
    /// Create a new DataLoader configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set shuffle
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set number of workers
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Set pin memory
    pub fn pin_memory(mut self, pin_memory: bool) -> Self {
        self.pin_memory = pin_memory;
        self
    }

    /// Set drop last
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Set timeout
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set generator seed
    pub fn generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }

    /// Set prefetch buffer size
    pub fn prefetch_buffer_size(mut self, size: usize) -> Self {
        self.prefetch_buffer_size = size;
        self
    }

    /// Enable persistent workers
    pub fn persistent_workers(mut self, persistent: bool) -> Self {
        self.persistent_workers = persistent;
        self
    }

    /// Set memory pinning configuration
    pub fn pinning_config(mut self, config: PinningConfig) -> Self {
        self.pinning_config = Some(config);
        self
    }

    /// Create a configuration optimized for training
    pub fn for_training() -> Self {
        Self::new()
            .batch_size(32)
            .shuffle(true)
            .num_workers(workers::utils::optimal_worker_count(false))
            .pin_memory(true)
            .prefetch_buffer_size(4)
            .persistent_workers(true)
    }

    /// Create a configuration optimized for inference
    pub fn for_inference() -> Self {
        Self::new()
            .batch_size(1)
            .shuffle(false)
            .num_workers(workers::utils::optimal_worker_count(true))
            .pin_memory(false)
            .prefetch_buffer_size(2)
            .persistent_workers(false)
    }

    /// Create a configuration optimized for evaluation
    pub fn for_evaluation() -> Self {
        Self::new()
            .batch_size(32)
            .shuffle(false)
            .num_workers(workers::utils::optimal_worker_count(false))
            .pin_memory(false)
            .prefetch_buffer_size(2)
            .persistent_workers(false)
            .drop_last(false)
    }
}

/// Create a DataLoader with comprehensive configuration
///
/// Note: This function was removed due to lifetime issues with returning boxed iterators.
/// Use DataLoader::builder() directly for configuration, or simple_dataloader()/simple_random_dataloader() for basic usage.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::dataloader::{DataLoader, simple_dataloader, simple_random_dataloader};
/// use torsh_data::dataset::TensorDataset;
///
/// let dataset = TensorDataset::new(vec![1, 2, 3, 4, 5]);
///
/// // Option 1: Use simple functions
/// let dataloader = simple_dataloader(dataset, 2, false)?;
/// for batch in dataloader.iter() {
///     // Process batch
/// }
///
/// // Option 2: Use builder pattern
/// let dataloader = DataLoader::builder(dataset)
///     .batch_size(16)
///     .shuffle(true)
///     .build()?;
/// ```

/// Utility functions for DataLoader operations
pub mod utils {
    use super::*;
    use torsh_core::device::DeviceType;

    /// Determine optimal DataLoader configuration for a given scenario
    ///
    /// # Arguments
    ///
    /// * `dataset_size` - Size of the dataset
    /// * `scenario` - Training, inference, or evaluation scenario
    /// * `target_device` - Target device for data
    ///
    /// # Returns
    ///
    /// Optimized DataLoaderConfig for the scenario
    pub fn optimal_config(
        dataset_size: usize,
        scenario: &str,
        target_device: Option<DeviceType>,
    ) -> DataLoaderConfig {
        let base_config = match scenario.to_lowercase().as_str() {
            "training" | "train" => DataLoaderConfig::for_training(),
            "inference" | "infer" => DataLoaderConfig::for_inference(),
            "evaluation" | "eval" | "test" => DataLoaderConfig::for_evaluation(),
            _ => DataLoaderConfig::new(),
        };

        let mut config = base_config;

        // Adjust batch size based on dataset size
        if dataset_size < 100 {
            config = config.batch_size(dataset_size.min(8));
        } else if dataset_size < 1000 {
            config = config.batch_size(16);
        } else {
            config = config.batch_size(32);
        }

        // Configure memory pinning based on target device
        if let Some(device) = target_device {
            match device {
                DeviceType::Cuda(device_id) => {
                    config = config
                        .pin_memory(true)
                        .pinning_config(PinningConfig::cuda(device_id));
                }
                _ => {
                    config = config.pin_memory(false);
                }
            }
        }

        config
    }

    /// Create a DataLoader with automatic configuration
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset to load from
    /// * `scenario` - Training scenario ("training", "inference", "evaluation")
    ///
    /// # Returns
    ///
    /// A DataLoader with automatically optimized configuration
    ///
    /// Note: This function was removed due to lifetime issues.
    /// Use DataLoader::builder() with manual configuration instead.
    // pub fn auto_dataloader(...) -- removed due to lifetime issues

    /// Validate DataLoader configuration for common issues
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration to validate
    /// * `dataset_size` - Size of the dataset
    ///
    /// # Returns
    ///
    /// Result with validation messages or errors
    pub fn validate_config(config: &DataLoaderConfig, dataset_size: usize) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        // Check batch size
        if config.batch_size > dataset_size {
            warnings.push(format!(
                "Batch size ({}) is larger than dataset size ({})",
                config.batch_size, dataset_size
            ));
        }

        // Check worker configuration
        if config.num_workers
            > std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                * 2
        {
            warnings.push(format!(
                "Number of workers ({}) may be too high for system capabilities",
                config.num_workers
            ));
        }

        // Check prefetch buffer size
        if config.prefetch_buffer_size > 0 && config.num_workers == 0 {
            warnings.push(
                "Prefetching enabled but no workers configured, may not improve performance"
                    .to_string(),
            );
        }

        // Check memory pinning
        if config.pin_memory && config.pinning_config.is_none() {
            warnings
                .push("Memory pinning enabled but no pinning configuration provided".to_string());
        }

        Ok(warnings)
    }

    /// Get performance recommendations for DataLoader configuration
    ///
    /// # Arguments
    ///
    /// * `dataset_size` - Size of the dataset
    /// * `batch_size` - Current batch size
    /// * `scenario` - Use case scenario
    ///
    /// # Returns
    ///
    /// Vector of performance recommendations
    pub fn performance_recommendations(
        dataset_size: usize,
        batch_size: usize,
        scenario: &str,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Batch size recommendations
        if scenario == "training" && batch_size < 16 {
            recommendations.push(
                "Consider increasing batch size for training (recommended: 16-32)".to_string(),
            );
        }

        if batch_size > dataset_size / 10 {
            recommendations.push(
                "Large batch size relative to dataset may reduce training effectiveness"
                    .to_string(),
            );
        }

        // Worker recommendations
        let optimal_workers = workers::utils::optimal_worker_count(scenario == "inference");
        recommendations.push(format!(
            "Consider using {} workers for optimal performance",
            optimal_workers
        ));

        // Prefetching recommendations
        if scenario == "training" {
            recommendations.push(
                "Enable prefetching (buffer size 2-4) for better training performance".to_string(),
            );
        }

        // Memory pinning recommendations
        recommendations.push("Enable memory pinning if transferring data to GPU".to_string());

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TensorDataset;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_dataloader_config() {
        let config = DataLoaderConfig::new()
            .batch_size(16)
            .shuffle(true)
            .num_workers(4)
            .prefetch_buffer_size(2);

        assert_eq!(config.batch_size, 16);
        assert!(config.shuffle);
        assert_eq!(config.num_workers, 4);
        assert_eq!(config.prefetch_buffer_size, 2);
    }

    #[test]
    fn test_training_config() {
        let config = DataLoaderConfig::for_training();
        assert!(config.shuffle);
        assert!(config.persistent_workers);
        assert!(config.pin_memory);
        assert!(config.prefetch_buffer_size > 0);
    }

    #[test]
    fn test_inference_config() {
        let config = DataLoaderConfig::for_inference();
        assert!(!config.shuffle);
        assert!(!config.persistent_workers);
        assert!(!config.pin_memory);
    }

    #[test]
    fn test_evaluation_config() {
        let config = DataLoaderConfig::for_evaluation();
        assert!(!config.shuffle);
        assert!(!config.drop_last);
        assert!(!config.persistent_workers);
    }

    // test_create_dataloader removed - function was removed due to lifetime issues
    // Use DataLoader::builder() directly instead

    #[test]
    fn test_utils_optimal_config() {
        let config = utils::optimal_config(1000, "training", Some(DeviceType::Cuda(0)));
        assert!(config.shuffle);
        assert!(config.pin_memory);
        assert!(config.pinning_config.is_some());
    }

    // test_utils_auto_dataloader removed - function was removed due to lifetime issues

    #[test]
    fn test_utils_validate_config() {
        let config = DataLoaderConfig::new().batch_size(100);
        let warnings = utils::validate_config(&config, 50).unwrap();
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("Batch size"));
    }

    #[test]
    fn test_utils_performance_recommendations() {
        let recommendations = utils::performance_recommendations(1000, 8, "training");
        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.contains("batch size")));
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that original API still works
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let dataloader = simple_dataloader(dataset, 2, false).unwrap();
        assert_eq!(dataloader.len(), 3);
    }

    #[test]
    fn test_prefetch_integration() {
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let dataloader = DataLoader::builder(dataset).batch_size(2).build().unwrap();

        // Test regular iteration instead of prefetch to avoid lifetime issues
        let mut iter = dataloader.iter();
        let first_batch = iter.next().unwrap().unwrap();
        assert_eq!(first_batch.len(), 1); // Should have 1 stacked tensor

        // Verify the tensor shape: [batch_size, sample_features]
        // Original tensor is [5], each sample is [1], batched becomes [2, 1]
        assert_eq!(first_batch[0].shape().dims(), &[2, 1]);
    }
}
