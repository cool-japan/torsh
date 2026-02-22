//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "std")]
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use torsh_core::error::Result;
use torsh_tensor::Tensor;

use super::functions::{Dataset, IterableDataset, StreamingDataset};

/// Iterator for PipelineStreamingDataset
pub struct PipelineStreamingDatasetIter<S: StreamingDataset, T> {
    pub(super) source_iter: S::Stream,
    pub(super) pipeline: Arc<DataPipeline<T>>,
}
/// Iterator for DatasetToStreaming
pub struct DatasetToStreamingIter<D: Dataset> {
    pub(super) dataset: D,
    pub(super) current_index: usize,
    pub(super) repeat: bool,
}
/// Convert a regular Dataset into a StreamingDataset
pub struct DatasetToStreaming<D: Dataset> {
    pub(super) dataset: D,
    pub(super) repeat: bool,
}
impl<D: Dataset> DatasetToStreaming<D> {
    /// Create a streaming version of a Dataset
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            repeat: false,
        }
    }
    /// Enable repeating the dataset infinitely
    pub fn repeat(mut self) -> Self {
        self.repeat = true;
        self
    }
}
/// Dataset statistics for a single feature
#[derive(Debug, Clone)]
pub struct FeatureStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub count: usize,
}
impl FeatureStats {
    /// Create feature statistics from data
    pub fn from_data(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                count: 0,
            };
        }
        let count = data.len();
        let sum: f32 = data.iter().sum();
        let mean = sum / count as f32;
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / count as f32;
        let std = variance.sqrt();
        Self {
            mean,
            std,
            min,
            max,
            count,
        }
    }
}
/// A buffered streaming dataset that maintains a buffer for efficient streaming
pub struct BufferedStreamingDataset<S: StreamingDataset> {
    pub(super) source: S,
    pub(super) buffer_size: usize,
    pub(super) prefetch: bool,
}
impl<S: StreamingDataset> BufferedStreamingDataset<S> {
    /// Create a new buffered streaming dataset
    pub fn new(source: S, buffer_size: usize) -> Self {
        Self {
            source,
            buffer_size,
            prefetch: true,
        }
    }
    /// Enable or disable prefetching
    pub fn with_prefetch(mut self, prefetch: bool) -> Self {
        self.prefetch = prefetch;
        self
    }
}
/// K-fold cross-validation indices generator
///
/// Generates k folds of train/validation indices for cross-validation.
/// Each fold uses (k-1)/k of the data for training and 1/k for validation.
#[derive(Debug, Clone)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    random_seed: Option<u64>,
}
impl KFold {
    /// Create a new K-fold cross-validator
    ///
    /// # Arguments
    /// * `n_splits` - Number of folds (must be >= 2)
    /// * `shuffle` - Whether to shuffle data before splitting
    /// * `random_seed` - Random seed for reproducible shuffling
    pub fn new(n_splits: usize, shuffle: bool, random_seed: Option<u64>) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        Self {
            n_splits,
            shuffle,
            random_seed,
        }
    }
    /// Generate fold indices for a dataset
    ///
    /// Returns a vector of (train_indices, val_indices) tuples, one for each fold.
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if self.shuffle {
            use scirs2_core::random::prelude::*;
            use scirs2_core::random::seq::ScientificSliceRandom;
            use scirs2_core::random::SeedableRng;
            let mut rng = if let Some(seed) = self.random_seed {
                StdRng::seed_from_u64(seed)
            } else {
                use std::time::SystemTime;
                let seed = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .expect("time should be after UNIX_EPOCH")
                    .as_secs();
                StdRng::seed_from_u64(seed)
            };
            indices.scientific_shuffle(&mut rng);
        }
        let mut folds = Vec::with_capacity(self.n_splits);
        let fold_size = n_samples / self.n_splits;
        for fold_idx in 0..self.n_splits {
            let val_start = fold_idx * fold_size;
            let val_end = if fold_idx == self.n_splits - 1 {
                n_samples
            } else {
                (fold_idx + 1) * fold_size
            };
            let val_indices: Vec<usize> = indices[val_start..val_end].to_vec();
            let mut train_indices: Vec<usize> = Vec::with_capacity(n_samples - val_indices.len());
            train_indices.extend_from_slice(&indices[0..val_start]);
            train_indices.extend_from_slice(&indices[val_end..n_samples]);
            folds.push((train_indices, val_indices));
        }
        folds
    }
}
/// Real-time data source that can be fed data from external sources
/// This is a simplified implementation for demonstration purposes
pub struct RealTimeDataset<T> {
    sender: std::sync::Arc<std::sync::Mutex<std::sync::mpsc::Sender<T>>>,
}
impl<T> RealTimeDataset<T> {
    /// Create a new real-time dataset
    pub fn new() -> (Self, std::sync::mpsc::Receiver<T>) {
        let (sender, receiver) = std::sync::mpsc::channel();
        let dataset = Self {
            sender: std::sync::Arc::new(std::sync::Mutex::new(sender)),
        };
        (dataset, receiver)
    }
    /// Get a sender handle to feed data into the dataset
    pub fn sender(&self) -> std::sync::Arc<std::sync::Mutex<std::sync::mpsc::Sender<T>>> {
        self.sender.clone()
    }
    /// Try to send data to the dataset (non-blocking)
    pub fn try_send(&self, item: T) -> std::result::Result<(), std::sync::mpsc::TrySendError<T>> {
        if let Ok(sender) = self.sender.try_lock() {
            sender
                .send(item)
                .map_err(|e| std::sync::mpsc::TrySendError::Disconnected(e.0))
        } else {
            Err(std::sync::mpsc::TrySendError::Full(item))
        }
    }
}
/// Iterator for BufferedStreamingDataset
pub struct BufferedStreamingDatasetIter<S: StreamingDataset> {
    pub(super) source_iter: S::Stream,
    pub(super) buffer: std::collections::VecDeque<Result<S::Item>>,
    pub(super) buffer_size: usize,
    pub(super) prefetch: bool,
}
/// Iterator for ChainDataset
pub struct ChainDatasetIter<D: IterableDataset + Clone> {
    pub(super) datasets: Vec<D>,
    pub(super) current_index: usize,
    pub(super) current_iter: Option<D::Iter>,
}
/// Iterator for InfiniteDataset
pub struct InfiniteDatasetIter<F, T>
where
    F: Fn() -> Result<T> + Send + Sync,
{
    pub(super) generator: F,
}
/// A simple dataset wrapping tensors
#[derive(Clone)]
pub struct TensorDataset<T = f32>
where
    T: torsh_core::dtype::TensorElement,
{
    pub(super) tensors: Vec<Tensor<T>>,
}
impl<T: torsh_core::dtype::TensorElement> TensorDataset<T> {
    /// Create a new tensor dataset from a vector of tensors
    pub fn new(tensors: Vec<Tensor<T>>) -> Self {
        if !tensors.is_empty() {
            let first_dim = tensors[0].size(0).unwrap_or(0);
            for tensor in &tensors[1..] {
                assert_eq!(
                    tensor.size(0).unwrap_or(0),
                    first_dim,
                    "All tensors must have the same first dimension"
                );
            }
        }
        Self { tensors }
    }
    /// Create from a single tensor, treating the first dimension as the dataset size
    pub fn from_tensor(tensor: Tensor<T>) -> Self {
        Self::new(vec![tensor])
    }
    /// Create from multiple tensors (e.g., features and labels)
    pub fn from_tensors(tensors: Vec<Tensor<T>>) -> Self {
        Self::new(tensors)
    }
}
/// Memory-mapped dataset for efficient large file handling
#[cfg(all(feature = "std", feature = "mmap-support"))]
pub struct MemoryMappedDataset<T: torsh_core::dtype::TensorElement> {
    /// Memory mapped file
    pub(super) mmap: Arc<memmap2::Mmap>,
    /// Metadata about tensor locations
    pub(super) tensor_offsets: Vec<usize>,
    /// Tensor shapes
    pub(super) tensor_shapes: Vec<Vec<usize>>,
    /// Number of samples
    pub(super) length: usize,
    /// Phantom data for type safety
    _phantom: std::marker::PhantomData<T>,
}
#[cfg(all(feature = "std", feature = "mmap-support"))]
impl<T: torsh_core::dtype::TensorElement> MemoryMappedDataset<T> {
    /// Create a memory-mapped dataset from a file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?
        };
        Ok(Self {
            mmap: Arc::new(mmap),
            tensor_offsets: Vec::new(),
            tensor_shapes: Vec::new(),
            length: 0,
            _phantom: std::marker::PhantomData,
        })
    }
    /// Create from raw bytes with metadata
    pub fn from_bytes_with_metadata(
        bytes: &[u8],
        offsets: Vec<usize>,
        shapes: Vec<Vec<usize>>,
        length: usize,
    ) -> Result<Self> {
        use std::io::Write;
        let mut temp_file = tempfile::NamedTempFile::new()
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;
        temp_file
            .write_all(bytes)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;
        temp_file
            .flush()
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(temp_file.as_file())
                .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?
        };
        Ok(Self {
            mmap: Arc::new(mmap),
            tensor_offsets: offsets,
            tensor_shapes: shapes,
            length,
            _phantom: std::marker::PhantomData,
        })
    }
}
/// A dataset that concatenates multiple datasets
#[derive(Clone)]
pub struct ConcatDataset<D: Dataset> {
    pub(super) datasets: Vec<D>,
    pub(super) cumulative_sizes: Vec<usize>,
}
impl<D: Dataset> ConcatDataset<D> {
    /// Create a new concatenated dataset
    pub fn new(datasets: Vec<D>) -> Self {
        let mut cumulative_sizes = Vec::with_capacity(datasets.len());
        let mut total = 0;
        for dataset in &datasets {
            total += dataset.len();
            cumulative_sizes.push(total);
        }
        Self {
            datasets,
            cumulative_sizes,
        }
    }
    /// Find which dataset an index belongs to
    pub(crate) fn dataset_idx(&self, index: usize) -> Option<(usize, usize)> {
        for (dataset_idx, &cumsum) in self.cumulative_sizes.iter().enumerate() {
            if index < cumsum {
                let dataset_offset = if dataset_idx == 0 {
                    0
                } else {
                    self.cumulative_sizes[dataset_idx - 1]
                };
                return Some((dataset_idx, index - dataset_offset));
            }
        }
        None
    }
}
/// Cached dataset wrapper for improved repeated access performance
pub struct CachedDataset<D: Dataset> {
    pub(super) dataset: D,
    pub(super) cache: Arc<RwLock<HashMap<usize, D::Item>>>,
    max_cache_size: usize,
    pub(super) access_count: Arc<RwLock<HashMap<usize, usize>>>,
}
impl<D: Dataset> CachedDataset<D>
where
    D::Item: Clone + Send + Sync,
{
    /// Create a new cached dataset
    pub fn new(dataset: D, max_cache_size: usize) -> Self {
        Self {
            dataset,
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size,
            access_count: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    /// Clear the cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().expect("lock should not be poisoned");
        let mut access_count = self
            .access_count
            .write()
            .expect("lock should not be poisoned");
        cache.clear();
        access_count.clear();
    }
    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let access_count = self
            .access_count
            .read()
            .expect("lock should not be poisoned");
        let total_accesses: usize = access_count.values().sum();
        let cache = self.cache.read().expect("lock should not be poisoned");
        let cache_hits: usize = access_count
            .iter()
            .filter_map(|(&idx, &count)| {
                if cache.contains_key(&idx) {
                    Some(count)
                } else {
                    None
                }
            })
            .sum();
        if total_accesses == 0 {
            0.0
        } else {
            cache_hits as f64 / total_accesses as f64
        }
    }
    /// Evict least recently used items if cache is full
    pub(crate) fn evict_lru(&self) {
        let mut cache = self.cache.write().expect("lock should not be poisoned");
        let access_count = self
            .access_count
            .read()
            .expect("lock should not be poisoned");
        if cache.len() >= self.max_cache_size {
            if let Some((&lru_idx, _)) = access_count.iter().min_by_key(|&(_, &count)| count) {
                cache.remove(&lru_idx);
            }
        }
    }
}
/// Iterator for RealTimeDataset
pub struct RealTimeDatasetIter<T> {
    pub(super) receiver: std::sync::mpsc::Receiver<T>,
}
/// Data pipeline for composing streaming operations
pub struct DataPipeline<T> {
    transformations: Vec<Box<dyn Fn(T) -> Result<T> + Send + Sync>>,
}
impl<T> DataPipeline<T> {
    /// Create a new data pipeline
    pub fn new() -> Self {
        Self {
            transformations: Vec::new(),
        }
    }
    /// Add a transformation to the pipeline
    pub fn add_transform<F>(mut self, transform: F) -> Self
    where
        F: Fn(T) -> Result<T> + Send + Sync + 'static,
    {
        self.transformations.push(Box::new(transform));
        self
    }
    /// Apply all transformations to an item
    pub fn apply(&self, mut item: T) -> Result<T> {
        for transform in &self.transformations {
            item = transform(item)?;
        }
        Ok(item)
    }
}
/// A subset of a dataset
#[derive(Clone)]
pub struct Subset<D: Dataset> {
    pub(super) dataset: D,
    pub(super) indices: Vec<usize>,
}
impl<D: Dataset> Subset<D> {
    /// Create a new subset with the given indices
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }
}
/// Dataset profiler for analyzing data loading performance
///
/// This utility helps diagnose performance bottlenecks in data loading pipelines
/// by tracking access patterns, timing statistics, and providing optimization hints.
#[cfg(feature = "std")]
pub struct DatasetProfiler {
    /// Total number of accesses
    access_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Sequential access count
    sequential_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Last accessed index
    last_index: std::sync::Arc<std::sync::Mutex<Option<usize>>>,
    /// Total access time in microseconds
    total_time_us: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Start time for profiling session
    start_time: std::time::Instant,
}
#[cfg(feature = "std")]
impl DatasetProfiler {
    /// Create a new dataset profiler
    pub fn new() -> Self {
        Self {
            access_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            sequential_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            last_index: std::sync::Arc::new(std::sync::Mutex::new(None)),
            total_time_us: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            start_time: std::time::Instant::now(),
        }
    }
    /// Record a dataset access
    pub fn record_access(&self, index: usize, duration: std::time::Duration) {
        use std::sync::atomic::Ordering;
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.total_time_us
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
        if let Ok(mut last) = self.last_index.lock() {
            if let Some(prev_idx) = *last {
                if index == prev_idx + 1 {
                    self.sequential_count.fetch_add(1, Ordering::Relaxed);
                }
            }
            *last = Some(index);
        }
    }
    /// Get profiling statistics
    pub fn stats(&self) -> DatasetProfileStats {
        use std::sync::atomic::Ordering;
        let access_count = self.access_count.load(Ordering::Relaxed);
        let sequential_count = self.sequential_count.load(Ordering::Relaxed);
        let total_time_us = self.total_time_us.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed();
        DatasetProfileStats {
            total_accesses: access_count,
            sequential_accesses: sequential_count,
            sequential_ratio: if access_count > 1 {
                sequential_count as f64 / (access_count - 1) as f64
            } else {
                0.0
            },
            avg_access_time_us: if access_count > 0 {
                total_time_us as f64 / access_count as f64
            } else {
                0.0
            },
            total_time_us,
            elapsed_seconds: elapsed.as_secs_f64(),
            throughput_accesses_per_sec: if elapsed.as_secs_f64() > 0.0 {
                access_count as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            },
        }
    }
    /// Get optimization hints based on profiling data
    pub fn hints(&self) -> Vec<String> {
        let stats = self.stats();
        let mut hints = Vec::new();
        if stats.sequential_ratio > 0.9 {
            hints
                .push(
                    "High sequential access detected. Consider using SequentialSampler for optimal performance."
                        .to_string(),
                );
        } else if stats.sequential_ratio < 0.1 {
            hints
                .push(
                    "Random access pattern detected. Consider using memory-mapped dataset or caching for better performance."
                        .to_string(),
                );
        }
        if stats.avg_access_time_us > 1000.0 {
            hints.push(format!(
                "Average access time is {:.2}ms. Consider prefetching or increasing num_workers.",
                stats.avg_access_time_us / 1000.0
            ));
        }
        if stats.throughput_accesses_per_sec < 100.0 && stats.total_accesses > 100 {
            hints
                .push(
                    format!(
                        "Low throughput ({:.1} accesses/sec). Consider optimizing dataset.get() implementation or using parallel loading.",
                        stats.throughput_accesses_per_sec
                    ),
                );
        }
        if hints.is_empty() {
            hints.push("Data loading performance looks good!".to_string());
        }
        hints
    }
    /// Reset profiling statistics
    pub fn reset(&self) {
        use std::sync::atomic::Ordering;
        self.access_count.store(0, Ordering::Relaxed);
        self.sequential_count.store(0, Ordering::Relaxed);
        self.total_time_us.store(0, Ordering::Relaxed);
        if let Ok(mut last) = self.last_index.lock() {
            *last = None;
        }
    }
}
/// Wrapper dataset that profiles accesses
#[cfg(feature = "std")]
pub struct ProfiledDataset<D: Dataset> {
    pub(super) dataset: D,
    pub(super) profiler: std::sync::Arc<DatasetProfiler>,
}
#[cfg(feature = "std")]
impl<D: Dataset> ProfiledDataset<D> {
    /// Wrap a dataset with profiling
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            profiler: std::sync::Arc::new(DatasetProfiler::new()),
        }
    }
    /// Get reference to the profiler
    pub fn profiler(&self) -> &std::sync::Arc<DatasetProfiler> {
        &self.profiler
    }
    /// Get profiling statistics
    pub fn stats(&self) -> DatasetProfileStats {
        self.profiler.stats()
    }
    /// Get optimization hints
    pub fn hints(&self) -> Vec<String> {
        self.profiler.hints()
    }
    /// Print a profiling report
    pub fn print_report(&self) {
        println!("{}", self.stats());
        println!("\nOptimization Hints:");
        for hint in self.hints() {
            println!("  • {}", hint);
        }
    }
}
/// Chain multiple iterable datasets sequentially
///
/// This dataset chains multiple IterableDatasets together, yielding items
/// from each dataset in sequence. When one dataset is exhausted, it moves
/// to the next one.
pub struct ChainDataset<D: IterableDataset + Clone> {
    pub(super) datasets: Vec<D>,
}
impl<D: IterableDataset + Clone> ChainDataset<D> {
    /// Create a new ChainDataset from a vector of datasets
    pub fn new(datasets: Vec<D>) -> Self {
        Self { datasets }
    }
}
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct SharedTensorMeta {
    pub(crate) offset: usize,
    pub(crate) size: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype_size: usize,
}
/// Shared memory dataset for efficient multi-process data loading
#[cfg(feature = "std")]
pub struct SharedMemoryDataset<T: torsh_core::dtype::TensorElement> {
    /// Shared memory region containing serialized tensors
    pub(super) shared_data: Arc<RwLock<Vec<u8>>>,
    /// Metadata about tensor locations and sizes
    pub(super) metadata: Arc<RwLock<Vec<SharedTensorMeta>>>,
    /// Number of samples in the dataset
    pub(super) length: usize,
    /// Phantom data for type safety
    _phantom: std::marker::PhantomData<T>,
}
#[cfg(feature = "std")]
impl<T: torsh_core::dtype::TensorElement> SharedMemoryDataset<T> {
    /// Create a new shared memory dataset from existing tensors
    pub fn from_tensors(tensors: Vec<Vec<Tensor<T>>>) -> Result<Self> {
        let length = tensors.len();
        let mut shared_data = Vec::new();
        let mut metadata = Vec::new();
        for sample_tensors in tensors {
            for tensor in sample_tensors {
                let shape = tensor.shape().dims().to_vec();
                let data = tensor.data()?;
                let size = tensor.numel() * std::mem::size_of::<T>();
                let offset = shared_data.len();
                unsafe {
                    let data_ptr = data.as_ptr() as *const u8;
                    let slice = std::slice::from_raw_parts(data_ptr, size);
                    shared_data.extend_from_slice(slice);
                }
                metadata.push(SharedTensorMeta {
                    offset,
                    size,
                    shape,
                    dtype_size: std::mem::size_of::<T>(),
                });
            }
        }
        Ok(Self {
            shared_data: Arc::new(RwLock::new(shared_data)),
            metadata: Arc::new(RwLock::new(metadata)),
            length,
            _phantom: std::marker::PhantomData,
        })
    }
    /// Create shared memory dataset with a specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            shared_data: Arc::new(RwLock::new(Vec::with_capacity(capacity))),
            metadata: Arc::new(RwLock::new(Vec::new())),
            length: 0,
            _phantom: std::marker::PhantomData,
        }
    }
    /// Add a sample to the shared memory dataset
    pub fn add_sample(&mut self, tensors: Vec<Tensor<T>>) -> Result<()> {
        let mut shared_data = self
            .shared_data
            .write()
            .expect("lock should not be poisoned");
        let mut metadata = self.metadata.write().expect("lock should not be poisoned");
        for tensor in tensors {
            let shape = tensor.shape().dims().to_vec();
            let data = tensor.data()?;
            let size = tensor.numel() * std::mem::size_of::<T>();
            let offset = shared_data.len();
            unsafe {
                let data_ptr = data.as_ptr() as *const u8;
                let slice = std::slice::from_raw_parts(data_ptr, size);
                shared_data.extend_from_slice(slice);
            }
            metadata.push(SharedTensorMeta {
                offset,
                size,
                shape,
                dtype_size: std::mem::size_of::<T>(),
            });
        }
        self.length += 1;
        Ok(())
    }
    /// Get the shared data reference for external processes
    pub fn shared_data_ref(&self) -> Arc<RwLock<Vec<u8>>> {
        self.shared_data.clone()
    }
    /// Get metadata reference for external processes
    pub fn metadata_ref(&self) -> Arc<RwLock<Vec<SharedTensorMeta>>> {
        self.metadata.clone()
    }
}
/// An infinite dataset that generates data continuously
pub struct InfiniteDataset<F, T>
where
    F: Fn() -> Result<T> + Send + Sync,
{
    pub(super) generator: F,
}
impl<F, T> InfiniteDataset<F, T>
where
    F: Fn() -> Result<T> + Send + Sync,
{
    /// Create a new infinite dataset with a generator function
    pub fn new(generator: F) -> Self {
        Self { generator }
    }
}
/// Statistics from dataset profiling
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct DatasetProfileStats {
    /// Total number of dataset accesses
    pub total_accesses: usize,
    /// Number of sequential accesses
    pub sequential_accesses: usize,
    /// Ratio of sequential to total accesses
    pub sequential_ratio: f64,
    /// Average access time in microseconds
    pub avg_access_time_us: f64,
    /// Total access time in microseconds
    pub total_time_us: u64,
    /// Elapsed time since profiling started (seconds)
    pub elapsed_seconds: f64,
    /// Throughput in accesses per second
    pub throughput_accesses_per_sec: f64,
}
/// A streaming dataset that applies a data pipeline
pub struct PipelineStreamingDataset<S: StreamingDataset, T> {
    pub(super) source: S,
    pub(super) pipeline: Arc<DataPipeline<T>>,
}
impl<S: StreamingDataset<Item = T>, T> PipelineStreamingDataset<S, T> {
    /// Create a new pipeline streaming dataset
    pub fn new(source: S, pipeline: DataPipeline<T>) -> Self {
        Self {
            source,
            pipeline: Arc::new(pipeline),
        }
    }
}
