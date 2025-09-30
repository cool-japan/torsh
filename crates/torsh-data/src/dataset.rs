//! Dataset trait and implementations

use torsh_core::error::Result;
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

// ✅ SciRS2 Policy Compliant - Import SliceRandom for shuffle functionality
use scirs2_core::rand_prelude::SliceRandom;

#[cfg(feature = "std")]
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

/// A map-style dataset
///
/// Represents a dataset that supports random access with a known length.
pub trait Dataset: Send + Sync {
    /// The type of items returned by the dataset
    type Item;

    /// Returns the number of items in the dataset
    fn len(&self) -> usize;

    /// Returns true if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a single item from the dataset
    fn get(&self, index: usize) -> Result<Self::Item>;
}

/// An iterable-style dataset
///
/// Represents a dataset that can be iterated over but may not support
/// random access or have a known length.
pub trait IterableDataset: Send + Sync {
    /// The type of items returned by the dataset
    type Item;
    /// The iterator type
    type Iter: Iterator<Item = Result<Self::Item>> + Send;

    /// Create an iterator over the dataset
    fn iter(&self) -> Self::Iter;
}

/// A simple dataset wrapping tensors
#[derive(Clone)]
pub struct TensorDataset<T = f32>
where
    T: torsh_core::dtype::TensorElement,
{
    tensors: Vec<Tensor<T>>,
}

impl<T: torsh_core::dtype::TensorElement> TensorDataset<T> {
    /// Create a new tensor dataset from a vector of tensors
    pub fn new(tensors: Vec<Tensor<T>>) -> Self {
        // Verify all tensors have the same first dimension
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

impl<T: torsh_core::dtype::TensorElement> Dataset for TensorDataset<T> {
    type Item = Vec<Tensor<T>>;

    fn len(&self) -> usize {
        if self.tensors.is_empty() {
            0
        } else {
            self.tensors[0].size(0).unwrap_or(0)
        }
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.len() {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            });
        }

        // Extract the index-th element from each tensor
        let mut items = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            // Use select to extract the index-th element along the first dimension
            let selected = tensor.select(0, index as i64)?;
            items.push(selected);
        }

        Ok(items)
    }
}

/// A dataset that concatenates multiple datasets
#[derive(Clone)]
pub struct ConcatDataset<D: Dataset> {
    datasets: Vec<D>,
    cumulative_sizes: Vec<usize>,
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
    fn dataset_idx(&self, index: usize) -> Option<(usize, usize)> {
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

impl<D: Dataset> Dataset for ConcatDataset<D> {
    type Item = D::Item;

    fn len(&self) -> usize {
        self.cumulative_sizes.last().copied().unwrap_or(0)
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if let Some((dataset_idx, sample_idx)) = self.dataset_idx(index) {
            self.datasets[dataset_idx].get(sample_idx)
        } else {
            Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            })
        }
    }
}

/// A subset of a dataset
#[derive(Clone)]
pub struct Subset<D: Dataset> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D: Dataset> Subset<D> {
    /// Create a new subset with the given indices
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }
}

impl<D: Dataset> Dataset for Subset<D> {
    type Item = D::Item;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.indices.len() {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            });
        }

        let actual_index = self.indices[index];
        self.dataset.get(actual_index)
    }
}

/// Split a dataset into train and validation sets
pub fn random_split<D>(
    dataset: D,
    lengths: &[usize],
    generator: Option<u64>,
) -> Result<Vec<Subset<D>>>
where
    D: Dataset + Clone,
{
    let total_length: usize = lengths.iter().sum();
    if total_length != dataset.len() {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "Sum of lengths {} does not equal dataset length {}",
            total_length,
            dataset.len()
        )));
    }

    // Create indices
    let mut indices: Vec<usize> = (0..dataset.len()).collect();

    // Shuffle indices if generator seed is provided
    if let Some(seed) = generator {
        // ✅ SciRS2 Policy Compliant - Using enhanced scientific shuffle
        use scirs2_core::random::prelude::*;
        use scirs2_core::random::seq::ScientificSliceRandom;

        let mut rng = thread_rng();
        indices.scientific_shuffle(&mut rng);
    }

    // Split indices according to lengths
    let mut subsets = Vec::with_capacity(lengths.len());
    let mut offset = 0;

    for &length in lengths {
        let subset_indices = indices[offset..offset + length].to_vec();
        subsets.push(Subset::new(dataset.clone(), subset_indices));
        offset += length;
    }

    Ok(subsets)
}

/// Chain multiple iterable datasets sequentially
///
/// This dataset chains multiple IterableDatasets together, yielding items
/// from each dataset in sequence. When one dataset is exhausted, it moves
/// to the next one.
pub struct ChainDataset<D: IterableDataset + Clone> {
    datasets: Vec<D>,
}

impl<D: IterableDataset + Clone> ChainDataset<D> {
    /// Create a new ChainDataset from a vector of datasets
    pub fn new(datasets: Vec<D>) -> Self {
        Self { datasets }
    }
}

/// Iterator for ChainDataset
pub struct ChainDatasetIter<D: IterableDataset + Clone> {
    datasets: Vec<D>,
    current_index: usize,
    current_iter: Option<D::Iter>,
}

impl<D: IterableDataset + Clone> Iterator for ChainDatasetIter<D> {
    type Item = Result<D::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have a current iterator, try to get the next item
            if let Some(ref mut iter) = self.current_iter {
                if let Some(item) = iter.next() {
                    return Some(item);
                }
            }

            // Current iterator is exhausted, move to the next dataset
            self.current_index += 1;
            if self.current_index >= self.datasets.len() {
                return None; // All datasets exhausted
            }

            // Create iterator for the next dataset
            self.current_iter = Some(self.datasets[self.current_index].iter());
        }
    }
}

impl<D: IterableDataset + Clone> IterableDataset for ChainDataset<D> {
    type Item = D::Item;
    type Iter = ChainDatasetIter<D>;

    fn iter(&self) -> Self::Iter {
        let current_iter = if !self.datasets.is_empty() {
            Some(self.datasets[0].iter())
        } else {
            None
        };

        ChainDatasetIter {
            datasets: self.datasets.clone(),
            current_index: 0,
            current_iter,
        }
    }
}

// Add Clone trait bound for ChainDataset if the dataset type is Clone
impl<D: IterableDataset + Clone> Clone for ChainDataset<D> {
    fn clone(&self) -> Self {
        Self {
            datasets: self.datasets.clone(),
        }
    }
}

/// Streaming dataset interface for real-time data processing
///
/// This trait represents datasets that can continuously produce data,
/// potentially from real-time sources or infinite data generators.
pub trait StreamingDataset: Send + Sync {
    /// The type of items returned by the dataset
    type Item;
    /// The streaming iterator type
    type Stream: Iterator<Item = Result<Self::Item>> + Send;

    /// Create a stream over the dataset
    fn stream(&self) -> Self::Stream;

    /// Check if the stream has more data available
    fn has_more(&self) -> bool {
        true // Default: assume infinite streams
    }

    /// Reset the stream to the beginning (if supported)
    fn reset(&self) -> Result<()> {
        Ok(()) // Default: do nothing
    }
}

/// An infinite dataset that generates data continuously
pub struct InfiniteDataset<F, T>
where
    F: Fn() -> Result<T> + Send + Sync,
{
    generator: F,
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

/// Iterator for InfiniteDataset
pub struct InfiniteDatasetIter<F, T>
where
    F: Fn() -> Result<T> + Send + Sync,
{
    generator: F,
}

impl<F, T> Iterator for InfiniteDatasetIter<F, T>
where
    F: Fn() -> Result<T> + Send + Sync,
{
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        Some((self.generator)())
    }
}

impl<F, T> StreamingDataset for InfiniteDataset<F, T>
where
    F: Fn() -> Result<T> + Send + Sync + Clone,
{
    type Item = T;
    type Stream = InfiniteDatasetIter<F, T>;

    fn stream(&self) -> Self::Stream {
        InfiniteDatasetIter {
            generator: self.generator.clone(),
        }
    }

    fn has_more(&self) -> bool {
        true
    }
}

/// A buffered streaming dataset that maintains a buffer for efficient streaming
pub struct BufferedStreamingDataset<S: StreamingDataset> {
    source: S,
    buffer_size: usize,
    prefetch: bool,
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

/// Iterator for BufferedStreamingDataset
pub struct BufferedStreamingDatasetIter<S: StreamingDataset> {
    source_iter: S::Stream,
    buffer: std::collections::VecDeque<Result<S::Item>>,
    buffer_size: usize,
    prefetch: bool,
}

impl<S: StreamingDataset> Iterator for BufferedStreamingDatasetIter<S> {
    type Item = Result<S::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        // If prefetch is enabled, try to fill buffer
        if self.prefetch {
            while self.buffer.len() < self.buffer_size {
                if let Some(item) = self.source_iter.next() {
                    self.buffer.push_back(item);
                } else {
                    break;
                }
            }
        }

        // Return from buffer or directly from source
        if let Some(item) = self.buffer.pop_front() {
            Some(item)
        } else {
            self.source_iter.next()
        }
    }
}

impl<S: StreamingDataset> StreamingDataset for BufferedStreamingDataset<S>
where
    S::Item: Send,
{
    type Item = S::Item;
    type Stream = BufferedStreamingDatasetIter<S>;

    fn stream(&self) -> Self::Stream {
        BufferedStreamingDatasetIter {
            source_iter: self.source.stream(),
            buffer: std::collections::VecDeque::with_capacity(self.buffer_size),
            buffer_size: self.buffer_size,
            prefetch: self.prefetch,
        }
    }

    fn has_more(&self) -> bool {
        self.source.has_more()
    }

    fn reset(&self) -> Result<()> {
        self.source.reset()
    }
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

impl<T> Default for DataPipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A streaming dataset that applies a data pipeline
pub struct PipelineStreamingDataset<S: StreamingDataset, T> {
    source: S,
    pipeline: Arc<DataPipeline<T>>,
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

/// Iterator for PipelineStreamingDataset
pub struct PipelineStreamingDatasetIter<S: StreamingDataset, T> {
    source_iter: S::Stream,
    pipeline: Arc<DataPipeline<T>>,
}

impl<S: StreamingDataset<Item = T>, T> Iterator for PipelineStreamingDatasetIter<S, T> {
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.source_iter.next()? {
            Ok(item) => match self.pipeline.apply(item) {
                Ok(transformed) => Some(Ok(transformed)),
                Err(e) => Some(Err(e)),
            },
            Err(e) => Some(Err(e)),
        }
    }
}

impl<S: StreamingDataset<Item = T>, T> StreamingDataset for PipelineStreamingDataset<S, T> {
    type Item = T;
    type Stream = PipelineStreamingDatasetIter<S, T>;

    fn stream(&self) -> Self::Stream {
        PipelineStreamingDatasetIter {
            source_iter: self.source.stream(),
            pipeline: self.pipeline.clone(),
        }
    }

    fn has_more(&self) -> bool {
        self.source.has_more()
    }

    fn reset(&self) -> Result<()> {
        self.source.reset()
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

/// Iterator for RealTimeDataset
pub struct RealTimeDatasetIter<T> {
    receiver: std::sync::mpsc::Receiver<T>,
}

impl<T> Iterator for RealTimeDatasetIter<T> {
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.try_recv() {
            Ok(item) => Some(Ok(item)),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
        }
    }
}

impl<T: Send + Sync + 'static> StreamingDataset for RealTimeDataset<T> {
    type Item = T;
    type Stream = RealTimeDatasetIter<T>;

    fn stream(&self) -> Self::Stream {
        // Create a new channel for this stream
        let (_, receiver) = std::sync::mpsc::channel();
        RealTimeDatasetIter { receiver }
    }

    fn has_more(&self) -> bool {
        true // Real-time data sources are considered always available
    }
}

/// Convert a regular Dataset into a StreamingDataset
pub struct DatasetToStreaming<D: Dataset> {
    dataset: D,
    repeat: bool,
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

/// Iterator for DatasetToStreaming
pub struct DatasetToStreamingIter<D: Dataset> {
    dataset: D,
    current_index: usize,
    repeat: bool,
}

impl<D: Dataset> Iterator for DatasetToStreamingIter<D> {
    type Item = Result<D::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.dataset.len() {
            if self.repeat {
                self.current_index = 0;
            } else {
                return None;
            }
        }

        let result = self.dataset.get(self.current_index);
        self.current_index += 1;
        Some(result)
    }
}

impl<D: Dataset + Clone> StreamingDataset for DatasetToStreaming<D> {
    type Item = D::Item;
    type Stream = DatasetToStreamingIter<D>;

    fn stream(&self) -> Self::Stream {
        DatasetToStreamingIter {
            dataset: self.dataset.clone(),
            current_index: 0,
            repeat: self.repeat,
        }
    }

    fn has_more(&self) -> bool {
        self.repeat || self.dataset.len() > 0
    }

    fn reset(&self) -> Result<()> {
        Ok(()) // Reset is handled by creating a new iterator
    }
}

/// Shared memory dataset for efficient multi-process data loading
#[cfg(feature = "std")]
pub struct SharedMemoryDataset<T: torsh_core::dtype::TensorElement> {
    /// Shared memory region containing serialized tensors
    shared_data: Arc<RwLock<Vec<u8>>>,
    /// Metadata about tensor locations and sizes
    metadata: Arc<RwLock<Vec<SharedTensorMeta>>>,
    /// Number of samples in the dataset
    length: usize,
    /// Phantom data for type safety
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct SharedTensorMeta {
    offset: usize,
    size: usize,
    shape: Vec<usize>,
    dtype_size: usize,
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

                // Serialize tensor data
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
        let mut shared_data = self.shared_data.write().unwrap();
        let mut metadata = self.metadata.write().unwrap();

        for tensor in tensors {
            let shape = tensor.shape().dims().to_vec();
            let data = tensor.data()?;
            let size = tensor.numel() * std::mem::size_of::<T>();
            let offset = shared_data.len();

            // Serialize tensor data
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

#[cfg(feature = "std")]
impl<T: torsh_core::dtype::TensorElement> Dataset for SharedMemoryDataset<T> {
    type Item = Vec<Tensor<T>>;

    fn len(&self) -> usize {
        self.length
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.length {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.length,
            });
        }

        let shared_data = self.shared_data.read().unwrap();
        let metadata = self.metadata.read().unwrap();

        // For simplicity, assume each sample has the same number of tensors
        // In a real implementation, you'd track this in metadata
        let tensors_per_sample = metadata.len() / self.length;
        let start_idx = index * tensors_per_sample;
        let end_idx = start_idx + tensors_per_sample;

        let mut result_tensors = Vec::new();

        for meta_idx in start_idx..end_idx {
            if meta_idx >= metadata.len() {
                break;
            }

            let meta = &metadata[meta_idx];
            let data_slice = &shared_data[meta.offset..meta.offset + meta.size];

            // Reconstruct tensor from shared memory
            unsafe {
                let data_ptr = data_slice.as_ptr() as *const T;
                let data_slice_typed =
                    std::slice::from_raw_parts(data_ptr, meta.size / meta.dtype_size);

                // Create tensor from the data
                let tensor = Tensor::from_data(
                    data_slice_typed.to_vec(),
                    meta.shape.clone(),
                    torsh_core::device::DeviceType::Cpu,
                )?;
                result_tensors.push(tensor);
            }
        }

        Ok(result_tensors)
    }
}

/// Memory-mapped dataset for efficient large file handling
#[cfg(all(feature = "std", feature = "mmap-support"))]
pub struct MemoryMappedDataset<T: torsh_core::dtype::TensorElement> {
    /// Memory mapped file
    mmap: Arc<memmap2::Mmap>,
    /// Metadata about tensor locations
    tensor_offsets: Vec<usize>,
    /// Tensor shapes
    tensor_shapes: Vec<Vec<usize>>,
    /// Number of samples
    length: usize,
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

        // For now, create a simple implementation
        // In practice, you'd read headers to get tensor metadata
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
        // Create a temporary file for mmap
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

#[cfg(all(feature = "std", feature = "mmap-support"))]
impl<T: torsh_core::dtype::TensorElement> Dataset for MemoryMappedDataset<T> {
    type Item = Tensor<T>;

    fn len(&self) -> usize {
        self.length
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.length {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.length,
            });
        }

        if index >= self.tensor_offsets.len() || index >= self.tensor_shapes.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Invalid tensor metadata".to_string(),
            ));
        }

        let offset = self.tensor_offsets[index];
        let shape = &self.tensor_shapes[index];

        unsafe {
            let data_ptr = self.mmap.as_ptr().add(offset) as *const T;
            let numel = shape.iter().product::<usize>();
            let data_slice = std::slice::from_raw_parts(data_ptr, numel);
            let data_vec = data_slice.to_vec();

            Tensor::from_data(
                data_vec,
                shape.to_vec(),
                torsh_core::device::DeviceType::Cpu,
            )
        }
    }
}

/// Cached dataset wrapper for improved repeated access performance
pub struct CachedDataset<D: Dataset> {
    dataset: D,
    cache: Arc<RwLock<HashMap<usize, D::Item>>>,
    max_cache_size: usize,
    access_count: Arc<RwLock<HashMap<usize, usize>>>,
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
        let mut cache = self.cache.write().unwrap();
        let mut access_count = self.access_count.write().unwrap();
        cache.clear();
        access_count.clear();
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let access_count = self.access_count.read().unwrap();
        let total_accesses: usize = access_count.values().sum();
        let cache = self.cache.read().unwrap();
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
    fn evict_lru(&self) {
        let mut cache = self.cache.write().unwrap();
        let access_count = self.access_count.read().unwrap();

        if cache.len() >= self.max_cache_size {
            // Find least recently used item
            if let Some((&lru_idx, _)) = access_count.iter().min_by_key(|&(_, &count)| count) {
                cache.remove(&lru_idx);
            }
        }
    }
}

impl<D: Dataset> Dataset for CachedDataset<D>
where
    D::Item: Clone + Send + Sync,
{
    type Item = D::Item;

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        // Update access count
        {
            let mut access_count = self.access_count.write().unwrap();
            *access_count.entry(index).or_insert(0) += 1;
        }

        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some(item) = cache.get(&index) {
                return Ok(item.clone());
            }
        }

        // Not in cache, fetch from dataset
        let item = self.dataset.get(index)?;

        // Add to cache
        {
            self.evict_lru();
            let mut cache = self.cache.write().unwrap();
            cache.insert(index, item.clone());
        }

        Ok(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_tensor_dataset() {
        let data = ones::<f32>(&[10, 3]).unwrap();
        let labels = zeros::<f32>(&[10]).unwrap();

        let dataset = TensorDataset::from_tensors(vec![data, labels]);
        assert_eq!(dataset.len(), 10);

        let item = dataset.get(0).unwrap();
        assert_eq!(item.len(), 2);
    }

    #[test]
    fn test_concat_dataset() {
        let ds1 = TensorDataset::from_tensor(ones::<f32>(&[5, 3]).unwrap());
        let ds2 = TensorDataset::from_tensor(zeros::<f32>(&[3, 3]).unwrap());

        let concat = ConcatDataset::new(vec![ds1, ds2]);
        assert_eq!(concat.len(), 8);

        // Test dataset index calculation
        assert_eq!(concat.dataset_idx(0), Some((0, 0)));
        assert_eq!(concat.dataset_idx(4), Some((0, 4)));
        assert_eq!(concat.dataset_idx(5), Some((1, 0)));
        assert_eq!(concat.dataset_idx(7), Some((1, 2)));
        assert_eq!(concat.dataset_idx(8), None);
    }

    #[test]
    fn test_subset() {
        let dataset = TensorDataset::from_tensor(ones::<f32>(&[10, 3]).unwrap());
        let subset = Subset::new(dataset, vec![0, 2, 4, 6, 8]);

        assert_eq!(subset.len(), 5);
        assert!(subset.get(0).is_ok());
        assert!(subset.get(5).is_err());
    }

    // Test implementation for IterableDataset trait to use in ChainDataset tests
    #[derive(Clone)]
    struct SimpleIterableDataset {
        data: Vec<i32>,
    }

    impl IterableDataset for SimpleIterableDataset {
        type Item = i32;
        type Iter = std::iter::Map<std::vec::IntoIter<i32>, fn(i32) -> Result<i32>>;

        fn iter(&self) -> Self::Iter {
            self.data.clone().into_iter().map(|x| Ok(x) as Result<i32>)
        }
    }

    #[test]
    fn test_chain_dataset() {
        let ds1 = SimpleIterableDataset {
            data: vec![1, 2, 3],
        };
        let ds2 = SimpleIterableDataset {
            data: vec![4, 5, 6],
        };
        let ds3 = SimpleIterableDataset {
            data: vec![7, 8, 9],
        };

        let chain = ChainDataset::new(vec![ds1, ds2, ds3]);
        let collected: Result<Vec<_>> = chain.iter().collect();

        assert!(collected.is_ok());
        let values = collected.unwrap();
        assert_eq!(values, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_chain_dataset_empty() {
        let chain: ChainDataset<SimpleIterableDataset> = ChainDataset::new(vec![]);
        let collected: Result<Vec<_>> = chain.iter().collect();

        assert!(collected.is_ok());
        let values = collected.unwrap();
        assert_eq!(values, Vec::<i32>::new());
    }

    #[test]
    fn test_chain_dataset_with_empty_datasets() {
        let ds1 = SimpleIterableDataset { data: vec![] };
        let ds2 = SimpleIterableDataset {
            data: vec![1, 2, 3],
        };
        let ds3 = SimpleIterableDataset { data: vec![] };
        let ds4 = SimpleIterableDataset { data: vec![4, 5] };

        let chain = ChainDataset::new(vec![ds1, ds2, ds3, ds4]);
        let collected: Result<Vec<_>> = chain.iter().collect();

        assert!(collected.is_ok());
        let values = collected.unwrap();
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_infinite_dataset() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let dataset = InfiniteDataset::new(move || {
            let val = counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(val)
        });

        assert!(dataset.has_more());

        let mut stream = dataset.stream();
        assert_eq!(stream.next().unwrap().unwrap(), 0);
        assert_eq!(stream.next().unwrap().unwrap(), 1);
        assert_eq!(stream.next().unwrap().unwrap(), 2);
    }

    #[test]
    fn test_buffered_streaming_dataset() {
        let dataset = InfiniteDataset::new(|| Ok(42i32));
        let buffered = BufferedStreamingDataset::new(dataset, 5).with_prefetch(true);

        assert!(buffered.has_more());

        let mut stream = buffered.stream();
        for _ in 0..10 {
            assert_eq!(stream.next().unwrap().unwrap(), 42);
        }
    }

    #[test]
    fn test_data_pipeline() {
        let pipeline = DataPipeline::new()
            .add_transform(|x: i32| Ok(x * 2))
            .add_transform(|x: i32| Ok(x + 1));

        let result = pipeline.apply(5).unwrap();
        assert_eq!(result, 11); // (5 * 2) + 1
    }

    #[test]
    fn test_pipeline_streaming_dataset() {
        let dataset = InfiniteDataset::new(|| Ok(5i32));
        let pipeline = DataPipeline::new()
            .add_transform(|x: i32| Ok(x * 2))
            .add_transform(|x: i32| Ok(x + 1));

        let pipeline_dataset = PipelineStreamingDataset::new(dataset, pipeline);

        assert!(pipeline_dataset.has_more());

        let mut stream = pipeline_dataset.stream();
        for _ in 0..5 {
            assert_eq!(stream.next().unwrap().unwrap(), 11); // (5 * 2) + 1
        }
    }

    #[test]
    fn test_real_time_dataset() {
        let (dataset, _receiver) = RealTimeDataset::<i32>::new();
        let sender = dataset.sender();

        // Send some data
        {
            let sender_lock = sender.lock().unwrap();
            sender_lock.send(1).unwrap();
            sender_lock.send(2).unwrap();
            sender_lock.send(3).unwrap();
        }

        assert!(dataset.has_more());

        // Note: Due to the simplified implementation of RealTimeDataset::stream(),
        // this test demonstrates the structure rather than actual functionality
        let _stream = dataset.stream();
    }

    #[test]
    fn test_dataset_to_streaming() {
        let tensor = ones::<f32>(&[5, 3]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let streaming = DatasetToStreaming::new(dataset);

        assert!(streaming.has_more());

        let stream = streaming.stream();
        let mut count = 0;
        for result in stream {
            assert!(result.is_ok());
            count += 1;
            if count >= 5 {
                break; // Don't run indefinitely
            }
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_dataset_to_streaming_repeat() {
        let tensor = ones::<f32>(&[3, 2]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let streaming = DatasetToStreaming::new(dataset).repeat();

        assert!(streaming.has_more());

        let stream = streaming.stream();
        let mut count = 0;
        for result in stream {
            assert!(result.is_ok());
            count += 1;
            if count >= 10 {
                break; // Test that it repeats beyond original dataset size (3)
            }
        }
        assert_eq!(count, 10);
    }

    #[test]
    fn test_streaming_dataset_reset() {
        let dataset = InfiniteDataset::new(|| Ok(42i32));
        let buffered = BufferedStreamingDataset::new(dataset, 3);

        // Test reset functionality
        assert!(buffered.reset().is_ok());
    }
}
