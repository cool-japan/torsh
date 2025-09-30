//! # Unified Data Sampling Interface
//!
//! This module provides a comprehensive data sampling system for machine learning workflows,
//! supporting everything from basic sequential sampling to advanced distributed training scenarios.
//! The system is built on SciRS2's random number generation and designed for high-performance
//! data loading pipelines.
//!
//! ## Architecture
//!
//! The sampling system is organized into specialized modules:
//!
//! - **core**: Foundation traits, iterators, and RNG utilities
//! - **basic**: Sequential, random, and subset sampling strategies
//! - **batch**: Batch sampling functionality for converting individual samplers
//! - **distributed**: Multi-GPU/multi-node training support with epoch management
//! - **weighted**: Weighted and subset sampling for imbalanced datasets
//! - **stratified**: Stratified sampling for maintaining class proportions
//! - **curriculum**: Curriculum learning that gradually introduces harder samples
//! - **active_learning**: Active learning strategies for efficient labeling
//! - **adaptive**: Adaptive sampling that adjusts based on training progress
//! - **importance**: Importance sampling for bias correction and emphasis
//!
//! ## Quick Start
//!
//! ```rust
//! use torsh_data::sampler::{RandomSampler, SequentialSampler, Sampler};
//!
//! // Basic random sampling
//! let sampler = RandomSampler::new(1000, false, Some(42));
//! let indices: Vec<usize> = sampler.iter().collect();
//!
//! // Sequential sampling with batching
//! let sampler = SequentialSampler::new(1000)
//!     .into_batch_sampler(32, true);
//!
//! // Distributed training setup
//! let sampler = RandomSampler::new(10000, false, Some(42))
//!     .into_distributed(4, 0); // 4 GPUs, rank 0
//! ```
//!
//! ## Advanced Usage
//!
//! ```rust
//! use torsh_data::sampler::{WeightedRandomSampler, StratifiedSampler};
//!
//! // Weighted sampling with alias table optimization
//! let weights = vec![0.1, 0.3, 0.6];
//! let sampler = WeightedRandomSampler::from_weights(&weights, true, Some(42))
//!     .expect("Invalid weights");
//!
//! // Stratified sampling for balanced datasets
//! let mut strata = std::collections::HashMap::new();
//! strata.insert(0, vec![0, 1, 2]);  // Class 0 samples
//! strata.insert(1, vec![3, 4, 5]);  // Class 1 samples
//! let sampler = StratifiedSampler::new(strata, true, 1, Some(42));
//! ```
//!
//! ## Performance Considerations
//!
//! - **Alias Table**: WeightedRandomSampler uses O(1) sampling via alias method
//! - **SIMD Operations**: Vectorized operations for shuffle and permutation
//! - **Memory Efficiency**: Lazy evaluation and streaming for large datasets
//! - **Distributed Coordination**: Synchronized epoch management across ranks
//! - **SciRS2 Integration**: Leverages high-performance random number generation

// Core sampling infrastructure
pub mod active_learning;
pub mod adaptive;
pub mod advanced;
pub mod basic;
pub mod batch;
pub mod core;
pub mod curriculum;
pub mod distributed;
pub mod importance;
pub mod stratified;
pub mod weighted;

// Re-export core traits and utilities
pub use core::{utils, BatchSampler, Sampler, SamplerIterator};

// Re-export basic sampling strategies
pub use basic::{
    random, random_subset, random_with_replacement, sequential, RandomSampler, SequentialSampler,
};

// Re-export batch sampling
pub use batch::{batch, batch_drop_last, batch_keep_last, BatchSamplerIter, BatchingSampler};

// Re-export distributed sampling support
pub use distributed::{distributed, distributed_sampler, DistributedSampler, DistributedWrapper};

// Re-export weighted and subset sampling
pub use weighted::{
    balanced_weighted, subset_random, weighted_random, SubsetRandomSampler, WeightedRandomSampler,
};

// Re-export stratified sampling
pub use stratified::{
    balanced_stratified, stratified, stratified_train_test_split, StratifiedSampler,
};

// Re-export curriculum learning
pub use curriculum::{
    anti_curriculum, exponential_curriculum, linear_curriculum, step_curriculum, CurriculumSampler,
    CurriculumStats, CurriculumStrategy,
};

// Re-export active learning
pub use active_learning::{
    diversity_sampler, uncertainty_diversity_sampler, uncertainty_sampler, AcquisitionStrategy,
    ActiveLearningSampler, ActiveLearningStats,
};

// Re-export adaptive sampling
pub use adaptive::{
    frequency_balanced_sampler, hard_adaptive_sampler, uncertainty_adaptive_sampler,
    AdaptiveSampler, AdaptiveStats, AdaptiveStrategy,
};

// Re-export importance sampling
pub use importance::{
    class_balanced_importance_sampler, exponential_importance_sampler,
    loss_based_importance_sampler, uniform_importance_sampler, ImportanceSampler, ImportanceStats,
};

// Re-export advanced sampling strategies
pub use advanced::{
    GroupedSampler, ImportanceSampler as AdvancedImportanceSampler,
    StratifiedSampler as AdvancedStratifiedSampler,
    WeightedRandomSampler as AdvancedWeightedRandomSampler,
};

// Convenience type aliases for backward compatibility
pub type DefaultSampler = RandomSampler;
pub type DefaultBatchSampler = BatchingSampler<RandomSampler>;

/// Creates a default random sampler with optional configuration
///
/// This is a convenience function that provides a simple interface for the most
/// common sampling use case: random sampling without replacement.
///
/// # Arguments
///
/// * `dataset_size` - Total number of samples in the dataset
/// * `seed` - Optional seed for reproducible sampling
///
/// # Examples
///
/// ```rust
/// use torsh_data::sampler::default_sampler;
///
/// // Create sampler for 1000 samples with reproducible results
/// let sampler = default_sampler(1000, Some(42));
/// let indices: Vec<usize> = sampler.iter().collect();
/// assert_eq!(indices.len(), 1000);
/// ```
pub fn default_sampler(dataset_size: usize, seed: Option<u64>) -> RandomSampler {
    random(dataset_size, seed)
}

/// Creates a default batch sampler with common configuration
///
/// This convenience function creates a batched random sampler, which is the most
/// common configuration for training neural networks.
///
/// # Arguments
///
/// * `dataset_size` - Total number of samples in the dataset
/// * `batch_size` - Number of samples per batch
/// * `drop_last` - Whether to drop the last incomplete batch
/// * `seed` - Optional seed for reproducible sampling
///
/// # Examples
///
/// ```rust
/// use torsh_data::sampler::default_batch_sampler;
///
/// // Create batch sampler for training
/// let sampler = default_batch_sampler(1000, 32, true, Some(42));
/// let batches: Vec<Vec<usize>> = sampler.iter().collect();
/// assert_eq!(batches.len(), 31); // 1000 / 32 = 31.25, drop_last=true
/// ```
pub fn default_batch_sampler(
    dataset_size: usize,
    batch_size: usize,
    drop_last: bool,
    seed: Option<u64>,
) -> BatchingSampler<RandomSampler> {
    random(dataset_size, seed).into_batch_sampler(batch_size, drop_last)
}

/// Creates a default distributed sampler for multi-GPU training
///
/// This convenience function sets up distributed sampling with common defaults
/// for multi-GPU training scenarios.
///
/// # Arguments
///
/// * `dataset_size` - Total number of samples in the dataset
/// * `num_replicas` - Total number of distributed processes (usually number of GPUs)
/// * `rank` - Rank of the current process (0 to num_replicas-1)
/// * `seed` - Optional seed for reproducible sampling
///
/// # Examples
///
/// ```rust
/// use torsh_data::sampler::default_distributed_sampler;
///
/// // Setup for 4-GPU training, current process is rank 0
/// let sampler = default_distributed_sampler(10000, 4, 0, Some(42));
/// let local_indices: Vec<usize> = sampler.iter().collect();
/// assert_eq!(local_indices.len(), 2500); // 10000 / 4 = 2500 per GPU
/// ```
pub fn default_distributed_sampler(
    dataset_size: usize,
    num_replicas: usize,
    rank: usize,
    seed: Option<u64>,
) -> DistributedSampler {
    distributed_sampler(dataset_size, num_replicas, rank, true)
}

/// Factory function for creating samplers based on configuration
///
/// This function provides a unified interface for creating different types of samplers
/// based on string configuration, useful for configuration files and dynamic setup.
///
/// # Arguments
///
/// * `sampler_type` - Type of sampler ("sequential", "random", "weighted", etc.)
/// * `dataset_size` - Total number of samples in the dataset
/// * `config` - Additional configuration as key-value pairs
///
/// # Supported Sampler Types
///
/// - `"sequential"`: Sequential sampling from 0 to dataset_size-1
/// - `"random"`: Random sampling without replacement
/// - `"random_replacement"`: Random sampling with replacement
/// - `"weighted"`: Weighted random sampling (requires "weights" in config)
/// - `"stratified"`: Stratified sampling (requires "strata" in config)
/// - `"distributed"`: Distributed sampling (requires "num_replicas" and "rank" in config)
///
/// # Examples
///
/// ```rust
/// use torsh_data::sampler::create_sampler;
/// use std::collections::HashMap;
///
/// let mut config = HashMap::new();
/// config.insert("seed".to_string(), "42".to_string());
///
/// // This would return a boxed sampler trait object
/// // let sampler = create_sampler("random", 1000, &config);
/// ```
pub fn create_sampler(
    sampler_type: &str,
    dataset_size: usize,
    config: &std::collections::HashMap<String, String>,
) -> Result<Box<dyn Sampler<Iter = Box<dyn Iterator<Item = usize> + Send>> + Send>, String> {
    let seed = config.get("seed").and_then(|s| s.parse::<u64>().ok());

    match sampler_type {
        "sequential" => {
            let sampler = sequential(dataset_size);
            Ok(Box::new(SamplerWrapper::Sequential(sampler)))
        }
        "random" => {
            let sampler = random(dataset_size, seed);
            Ok(Box::new(SamplerWrapper::Random(sampler)))
        }
        "random_replacement" => {
            let sampler = random_with_replacement(dataset_size, dataset_size, seed);
            Ok(Box::new(SamplerWrapper::Random(sampler)))
        }
        "weighted" => {
            let weights_str = config
                .get("weights")
                .ok_or("Weighted sampler requires 'weights' configuration")?;
            let weights: Vec<f64> = weights_str
                .split(',')
                .map(|s| s.trim().parse::<f64>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| "Invalid weights format")?;

            let replacement = config
                .get("replacement")
                .map(|s| s.parse::<bool>().unwrap_or(false))
                .unwrap_or(false);

            let num_samples = config
                .get("num_samples")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(weights.len());

            let sampler = weighted_random(weights, num_samples, replacement, seed);
            Ok(Box::new(SamplerWrapper::Weighted(sampler)))
        }
        "distributed" => {
            let num_replicas = config
                .get("num_replicas")
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or("Distributed sampler requires 'num_replicas' configuration")?;
            let rank = config
                .get("rank")
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or("Distributed sampler requires 'rank' configuration")?;
            let shuffle = config
                .get("shuffle")
                .map(|s| s.parse::<bool>().unwrap_or(true))
                .unwrap_or(true);
            let drop_last = config
                .get("drop_last")
                .map(|s| s.parse::<bool>().unwrap_or(false))
                .unwrap_or(false);

            let sampler = DistributedSampler::new(dataset_size, num_replicas, rank, shuffle);
            Ok(Box::new(SamplerWrapper::Distributed(sampler)))
        }
        _ => Err(format!("Unknown sampler type: {}", sampler_type)),
    }
}

/// Wrapper enum for dynamic sampler dispatch
///
/// This enum allows for dynamic dispatch of different sampler types while maintaining
/// the Sampler trait interface. Used internally by the factory function.
#[derive(Debug)]
enum SamplerWrapper {
    Sequential(SequentialSampler),
    Random(RandomSampler),
    Weighted(WeightedRandomSampler),
    Distributed(DistributedSampler),
}

impl Sampler for SamplerWrapper {
    type Iter = Box<dyn Iterator<Item = usize> + Send>;

    fn iter(&self) -> Self::Iter {
        match self {
            SamplerWrapper::Sequential(s) => Box::new(s.iter()),
            SamplerWrapper::Random(s) => Box::new(s.iter()),
            SamplerWrapper::Weighted(s) => Box::new(s.iter()),
            SamplerWrapper::Distributed(s) => Box::new(s.iter()),
        }
    }

    fn len(&self) -> usize {
        match self {
            SamplerWrapper::Sequential(s) => s.len(),
            SamplerWrapper::Random(s) => s.len(),
            SamplerWrapper::Weighted(s) => s.len(),
            SamplerWrapper::Distributed(s) => s.len(),
        }
    }

    fn into_batch_sampler(self, batch_size: usize, drop_last: bool) -> BatchingSampler<Self>
    where
        Self: Sized,
    {
        BatchingSampler::new(self, batch_size, drop_last)
    }

    fn into_distributed(self, num_replicas: usize, rank: usize) -> DistributedWrapper<Self>
    where
        Self: Sized,
    {
        DistributedWrapper::new(self, num_replicas, rank)
    }
}

/// Utility function to split dataset into train and validation sets
///
/// # Arguments
///
/// * `dataset_size` - Total number of samples
/// * `val_ratio` - Fraction of data to use for validation (0.0 to 1.0)
/// * `seed` - Optional seed for reproducible splits
///
/// # Returns
///
/// A tuple of (train_indices, val_indices)
pub fn train_val_split(
    dataset_size: usize,
    val_ratio: f32,
    seed: Option<u64>,
) -> (Vec<usize>, Vec<usize>) {
    assert!(
        val_ratio >= 0.0 && val_ratio <= 1.0,
        "Validation ratio must be in [0, 1]"
    );

    let val_size = (dataset_size as f32 * val_ratio).round() as usize;
    let train_size = dataset_size - val_size;

    let mut indices: Vec<usize> = (0..dataset_size).collect();

    if let Some(seed_val) = seed {
        use scirs2_core::random::{Random, Rng};
        let mut rng = Random::seed(seed_val);
        // Simple Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
    }

    let train_indices = indices[..train_size].to_vec();
    let val_indices = indices[train_size..].to_vec();

    (train_indices, val_indices)
}

/// Utility function to split dataset into train, validation, and test sets
///
/// # Arguments
///
/// * `dataset_size` - Total number of samples
/// * `train_ratio` - Fraction for training
/// * `val_ratio` - Fraction for validation
/// * `seed` - Optional seed for reproducible splits
///
/// # Returns
///
/// A tuple of (train_indices, val_indices, test_indices)
pub fn train_val_test_split(
    dataset_size: usize,
    train_ratio: f32,
    val_ratio: f32,
    seed: Option<u64>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    assert!(
        train_ratio >= 0.0 && train_ratio <= 1.0,
        "Train ratio must be in [0, 1]"
    );
    assert!(
        val_ratio >= 0.0 && val_ratio <= 1.0,
        "Val ratio must be in [0, 1]"
    );
    assert!(
        train_ratio + val_ratio <= 1.0,
        "Train + val ratios must not exceed 1.0"
    );

    let train_size = (dataset_size as f32 * train_ratio).round() as usize;
    let val_size = (dataset_size as f32 * val_ratio).round() as usize;
    let test_size = dataset_size - train_size - val_size;

    let mut indices: Vec<usize> = (0..dataset_size).collect();

    if let Some(seed_val) = seed {
        use scirs2_core::random::{Random, Rng};
        let mut rng = Random::seed(seed_val);
        // Simple Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
    }

    let train_indices = indices[..train_size].to_vec();
    let val_indices = indices[train_size..train_size + val_size].to_vec();
    let test_indices = indices[train_size + val_size..].to_vec();

    (train_indices, val_indices, test_indices)
}

/// Generate k-fold cross-validation splits
///
/// # Arguments
///
/// * `dataset_size` - Total number of samples
/// * `k` - Number of folds
/// * `seed` - Optional seed for reproducible splits
///
/// # Returns
///
/// A vector of k (train_indices, val_indices) tuples
pub fn kfold_splits(
    dataset_size: usize,
    k: usize,
    seed: Option<u64>,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    assert!(k > 1, "Number of folds must be greater than 1");
    assert!(
        k <= dataset_size,
        "Number of folds cannot exceed dataset size"
    );

    let mut indices: Vec<usize> = (0..dataset_size).collect();

    if let Some(seed_val) = seed {
        use scirs2_core::random::{Random, Rng};
        let mut rng = Random::seed(seed_val);
        // Simple Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
    }

    let fold_size = dataset_size / k;
    let mut splits = Vec::with_capacity(k);

    for i in 0..k {
        let val_start = i * fold_size;
        let val_end = if i == k - 1 {
            dataset_size
        } else {
            (i + 1) * fold_size
        };

        let val_indices = indices[val_start..val_end].to_vec();
        let mut train_indices = Vec::new();
        train_indices.extend(&indices[..val_start]);
        train_indices.extend(&indices[val_end..]);

        splits.push((train_indices, val_indices));
    }

    splits
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_default_sampler() {
        let sampler = default_sampler(100, Some(42));
        let indices: Vec<usize> = sampler.iter().collect();

        assert_eq!(indices.len(), 100);
        assert_eq!(indices.iter().collect::<HashSet<_>>().len(), 100); // All unique

        // Test reproducibility
        let sampler2 = default_sampler(100, Some(42));
        let indices2: Vec<usize> = sampler2.iter().collect();
        assert_eq!(indices, indices2);
    }

    #[test]
    fn test_default_batch_sampler() {
        let sampler = default_batch_sampler(100, 32, true, Some(42));
        let batches: Vec<Vec<usize>> = sampler.iter().collect();

        assert_eq!(batches.len(), 3); // 100 / 32 = 3.125, drop_last=true
        assert_eq!(batches[0].len(), 32);
        assert_eq!(batches[1].len(), 32);
        assert_eq!(batches[2].len(), 32);

        // Test without dropping last batch
        let sampler = default_batch_sampler(100, 32, false, Some(42));
        let batches: Vec<Vec<usize>> = sampler.iter().collect();
        assert_eq!(batches.len(), 4); // 100 / 32 = 3.125, drop_last=false
        assert_eq!(batches[3].len(), 4); // Remainder
    }

    #[test]
    fn test_default_distributed_sampler() {
        let sampler = default_distributed_sampler(1000, 4, 0, Some(42));
        let indices: Vec<usize> = sampler.iter().collect();

        assert_eq!(indices.len(), 250); // 1000 / 4 = 250 per rank

        // Verify rank 0 gets first portion
        assert!(indices.iter().all(|&i| i < 1000));

        // Test different rank
        let sampler_rank1 = default_distributed_sampler(1000, 4, 1, Some(42));
        let indices_rank1: Vec<usize> = sampler_rank1.iter().collect();

        assert_eq!(indices_rank1.len(), 250);

        // Indices should be different between ranks
        let intersection: HashSet<_> = indices
            .iter()
            .filter(|&x| indices_rank1.contains(x))
            .collect();
        assert!(intersection.is_empty()); // No overlap between ranks
    }

    #[test]
    fn test_sampler_factory() {
        let mut config = std::collections::HashMap::new();
        config.insert("seed".to_string(), "42".to_string());

        // Test sequential sampler
        let sampler = create_sampler("sequential", 100, &config).unwrap();
        assert_eq!(sampler.len(), 100);

        // Test random sampler
        let sampler = create_sampler("random", 100, &config).unwrap();
        assert_eq!(sampler.len(), 100);

        // Test weighted sampler
        config.insert("weights".to_string(), "0.1,0.3,0.6".to_string());
        let sampler = create_sampler("weighted", 3, &config).unwrap();
        assert_eq!(sampler.len(), 3);

        // Test distributed sampler
        config.insert("num_replicas".to_string(), "4".to_string());
        config.insert("rank".to_string(), "0".to_string());
        let sampler = create_sampler("distributed", 1000, &config).unwrap();
        assert_eq!(sampler.len(), 250); // 1000 / 4 = 250
    }

    #[test]
    fn test_sampler_factory_errors() {
        let config = std::collections::HashMap::new();

        // Test unknown sampler type
        assert!(create_sampler("unknown", 100, &config).is_err());

        // Test weighted sampler without weights
        assert!(create_sampler("weighted", 100, &config).is_err());

        // Test distributed sampler without required config
        assert!(create_sampler("distributed", 100, &config).is_err());
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that all original sampler types are still accessible
        let _seq = SequentialSampler::new(100);
        let _rand = RandomSampler::new(100, None, false).with_generator(42);
        let _subset = SubsetRandomSampler::new(vec![0, 1, 2, 3, 4]).with_generator(42);
        let _distributed = DistributedSampler::new(100, 4, 0, true).with_generator(42);

        // Test that utility functions are accessible
        let (train, val) = train_val_split(1000, 0.2, Some(42));
        assert_eq!(train.len(), 800);
        assert_eq!(val.len(), 200);

        // Test that alias types work
        let _default: DefaultSampler = RandomSampler::new(100, None, false).with_generator(42);
    }

    #[test]
    fn test_modular_integration() {
        // Test that all modules work together seamlessly
        let base_sampler = RandomSampler::new(1000, None, false).with_generator(42);

        // Chain transformations
        let batch_sampler = base_sampler.into_batch_sampler(32, true);
        let distributed_sampler = batch_sampler.into_distributed(4, 0);

        let batches: Vec<Vec<usize>> = distributed_sampler.iter().collect();
        assert!(!batches.is_empty());

        // Verify each batch has correct size
        for batch in batches.iter().take(batches.len() - 1) {
            assert_eq!(batch.len(), 32);
        }
    }

    #[test]
    fn test_comprehensive_api_coverage() {
        // Test that all major sampling strategies are available
        use std::collections::HashMap;

        // Basic samplers
        let _seq = SequentialSampler::new(100);
        let _rand = RandomSampler::new(100, None, false).with_generator(42);
        let _subset = SubsetRandomSampler::new(vec![0, 1, 2]).with_generator(42);

        // Advanced samplers
        let weights = vec![0.1, 0.3, 0.6];
        let _weighted = WeightedRandomSampler::new(weights, 3, false).with_generator(42);

        // Create labels for stratified sampling
        let labels = vec![0, 0, 0, 1, 1, 1]; // 6 samples with 2 strata
        let _stratified = StratifiedSampler::new(&labels, 6, false);

        // Distributed samplers
        let _distributed = DistributedSampler::new(100, 4, 0, true);

        // Utility functions
        let _splits = kfold_splits(100, 5, Some(42));
        let (train, val, test) = train_val_test_split(1000, 0.6, 0.2, Some(42));
        assert_eq!(train.len() + val.len() + test.len(), 1000);
    }
}
