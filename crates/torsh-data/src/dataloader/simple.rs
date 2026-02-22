//! Simple DataLoader API
//!
//! This module provides convenient functions for creating DataLoaders with common
//! configurations without needing to use the builder pattern.

use super::core::DataLoader;
use crate::{
    collate::DefaultCollate,
    dataset::Dataset,
    sampler::{BatchingSampler, RandomSampler, SequentialSampler},
};
use torsh_core::error::Result;

/// Simplified DataLoader type for common use cases with sequential sampling
pub type SimpleDataLoader<D> = DataLoader<D, BatchingSampler<SequentialSampler>, DefaultCollate>;

/// Simplified DataLoader type for common use cases with random sampling
pub type SimpleRandomDataLoader<D> = DataLoader<D, BatchingSampler<RandomSampler>, DefaultCollate>;

/// Create a simple DataLoader with basic settings (sequential sampling)
///
/// This is a convenience function for quickly creating a DataLoader with sequential
/// sampling, which is useful for evaluation or when deterministic order is desired.
///
/// # Arguments
///
/// * `dataset` - The dataset to iterate over
/// * `batch_size` - Number of samples per batch
/// * `_shuffle` - Currently ignored (for API compatibility), always uses sequential sampling
///
/// # Returns
///
/// A Result containing the configured DataLoader or an error
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::dataloader::simple::simple_dataloader;
/// use torsh_data::dataset::TensorDataset;
///
/// let dataset = TensorDataset::new(vec![1, 2, 3, 4, 5]);
/// let dataloader = simple_dataloader(dataset, 2, false)?;
///
/// for batch in dataloader.iter() {
///     // Process batch sequentially
/// }
/// ```
///
/// # Note
///
/// This function always creates a DataLoader with sequential sampling for type consistency.
/// If you need random sampling, use `simple_random_dataloader` instead.
pub fn simple_dataloader<D: Dataset>(
    dataset: D,
    batch_size: usize,
    _shuffle: bool, // Note: This function always uses sequential sampling for type consistency
) -> Result<SimpleDataLoader<D>> {
    DataLoader::builder(dataset)
        .batch_size(batch_size)
        .shuffle(false) // Sequential sampling
        .build()
}

/// Create a simple DataLoader with random sampling (shuffled)
///
/// This is a convenience function for quickly creating a DataLoader with random
/// sampling, which is useful for training scenarios where data order randomization
/// is important.
///
/// # Arguments
///
/// * `dataset` - The dataset to iterate over
/// * `batch_size` - Number of samples per batch
/// * `generator` - Optional random seed for reproducible shuffling
///
/// # Returns
///
/// A Result containing the configured DataLoader with random sampling or an error
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::dataloader::simple::simple_random_dataloader;
/// use torsh_data::dataset::TensorDataset;
///
/// let dataset = TensorDataset::new(vec![1, 2, 3, 4, 5]);
/// let dataloader = simple_random_dataloader(dataset, 2, Some(42))?;
///
/// for batch in dataloader.iter() {
///     // Process batch in random order
/// }
/// ```
pub fn simple_random_dataloader<D: Dataset>(
    dataset: D,
    batch_size: usize,
    generator: Option<u64>,
) -> Result<SimpleRandomDataLoader<D>> {
    let mut builder = DataLoader::builder(dataset)
        .batch_size(batch_size)
        .shuffle(true); // Random sampling

    if let Some(seed) = generator {
        builder = builder.generator(seed);
    }

    builder.build_with_random_sampling()
}

/// Create a simple DataLoader with automatic sampling strategy
///
/// This function automatically chooses between sequential and random sampling
/// based on the shuffle parameter.
///
/// # Arguments
///
/// * `dataset` - The dataset to iterate over
/// * `batch_size` - Number of samples per batch
/// * `shuffle` - Whether to use random sampling (true) or sequential sampling (false)
/// * `generator` - Optional random seed for reproducible shuffling (only used when shuffle=true)
///
/// # Returns
///
/// Either a SimpleDataLoader or SimpleRandomDataLoader depending on shuffle setting
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::dataloader::simple::{simple_dataloader, simple_random_dataloader};
/// use torsh_data::dataset::TensorDataset;
///
/// let dataset = TensorDataset::new(vec![1, 2, 3, 4, 5]);
///
/// // Sequential sampling
/// let sequential_loader = simple_dataloader(dataset, 2, false)?;
/// for batch in sequential_loader.iter() {
///     // Process batch
/// }
/// ```
// Note: This function was removed due to lifetime issues with returning boxed iterators
// Use simple_dataloader() or simple_random_dataloader() directly instead

/// Configuration for simple DataLoader creation
///
/// This struct provides a more structured way to configure simple DataLoaders
/// while maintaining the convenience of the simple API.
#[derive(Debug, Clone)]
pub struct SimpleConfig {
    /// Number of samples per batch
    pub batch_size: usize,
    /// Whether to shuffle the data
    pub shuffle: bool,
    /// Number of worker threads (0 for single-threaded)
    pub num_workers: usize,
    /// Whether to drop the last incomplete batch
    pub drop_last: bool,
    /// Optional random seed for reproducible results
    pub generator: Option<u64>,
}

impl Default for SimpleConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            shuffle: false,
            num_workers: 0,
            drop_last: false,
            generator: None,
        }
    }
}

impl SimpleConfig {
    /// Create a new simple configuration
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

    /// Set drop last
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Set random generator seed
    pub fn generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }
}

/// Create a DataLoader with simple configuration
///
/// This function provides a middle ground between the simple functions and the full builder,
/// allowing for more configuration while maintaining simplicity.
///
/// # Arguments
///
/// * `dataset` - The dataset to iterate over
/// * `config` - Configuration for the DataLoader
///
/// # Returns
///
/// A boxed DataLoader trait object configured according to the provided config
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::dataloader::simple::{simple_configured_dataloader, SimpleConfig};
/// use torsh_data::dataset::TensorDataset;
///
/// let dataset = TensorDataset::new(vec![1, 2, 3, 4, 5]);
/// let config = SimpleConfig::new()
///     .batch_size(2)
///     .shuffle(true)
///     .num_workers(2)
///     .generator(42);
///
/// let dataloader = simple_configured_dataloader(dataset, config)?;
/// ```
// Note: simple_configured_dataloader was removed due to lifetime issues
// Use DataLoader::builder() for advanced configuration or simple_dataloader()/simple_random_dataloader() for basic usage

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TensorDataset;

    #[test]
    fn test_simple_dataloader() {
        // Create a tensor with 5 samples (first dimension is number of samples)
        let tensor = torsh_tensor::creation::ones::<f32>(&[5]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let dataloader = simple_dataloader(dataset, 2, false).unwrap();

        assert_eq!(dataloader.len(), 3); // 5 items, batch size 2 = 3 batches
        assert!(!dataloader.is_empty());
    }

    #[test]
    fn test_simple_random_dataloader() {
        // Create a tensor with 5 samples (first dimension is number of samples)
        let tensor = torsh_tensor::creation::ones::<f32>(&[5]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let dataloader = simple_random_dataloader(dataset, 2, Some(42)).unwrap();

        assert_eq!(dataloader.len(), 3);
        assert!(!dataloader.is_empty());
    }

    #[test]
    fn test_simple_random_dataloader_no_seed() {
        // Create a tensor with 5 samples (first dimension is number of samples)
        let tensor = torsh_tensor::creation::ones::<f32>(&[5]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let dataloader = simple_random_dataloader(dataset, 2, None).unwrap();

        assert_eq!(dataloader.len(), 3);
        assert!(!dataloader.is_empty());
    }

    // Tests for removed functions - use simple_dataloader/simple_random_dataloader directly

    #[test]
    fn test_simple_config() {
        let config = SimpleConfig::new()
            .batch_size(4)
            .shuffle(true)
            .num_workers(2)
            .drop_last(true)
            .generator(42);

        assert_eq!(config.batch_size, 4);
        assert!(config.shuffle);
        assert_eq!(config.num_workers, 2);
        assert!(config.drop_last);
        assert_eq!(config.generator, Some(42));
    }

    #[test]
    fn test_simple_config_defaults() {
        let config = SimpleConfig::new();

        assert_eq!(config.batch_size, 1);
        assert!(!config.shuffle);
        assert_eq!(config.num_workers, 0);
        assert!(!config.drop_last);
        assert_eq!(config.generator, None);
    }

    #[test]
    fn test_simple_configured_dataloader_sequential() {
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let _config = SimpleConfig::new()
            .batch_size(2)
            .shuffle(false)
            .drop_last(false);

        let dataloader = simple_dataloader(dataset, 2, false).unwrap();
        assert_eq!(dataloader.len(), 3);
    }

    #[test]
    fn test_simple_configured_dataloader_random() {
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();
        let _dataset = TensorDataset::from_tensor(tensor);
        let config = SimpleConfig::new()
            .batch_size(2)
            .shuffle(true)
            .generator(42);

        // Test that config is built correctly since simple_configured_dataloader was removed
        // Use DataLoader::builder() for configuration instead
        assert_eq!(config.batch_size, 2);
    }

    #[test]
    fn test_empty_dataset_simple() {
        let dataset: TensorDataset<f32> = TensorDataset::new(vec![]);
        let dataloader = simple_dataloader(dataset, 2, false).unwrap();

        assert_eq!(dataloader.len(), 0);
        assert!(dataloader.is_empty());
    }
}
