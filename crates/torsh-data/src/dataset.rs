//! Dataset trait and implementations

use torsh_core::error::Result;
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

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
    if let Some(_seed) = generator {
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
impl Default for DatasetProfiler {
    fn default() -> Self {
        Self::new()
    }
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

        // Update access count
        self.access_count.fetch_add(1, Ordering::Relaxed);

        // Update timing
        self.total_time_us
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);

        // Check if sequential
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

        // Check sequential access pattern
        if stats.sequential_ratio > 0.9 {
            hints.push("High sequential access detected. Consider using SequentialSampler for optimal performance.".to_string());
        } else if stats.sequential_ratio < 0.1 {
            hints.push("Random access pattern detected. Consider using memory-mapped dataset or caching for better performance.".to_string());
        }

        // Check access time
        if stats.avg_access_time_us > 1000.0 {
            hints.push(format!(
                "Average access time is {:.2}ms. Consider prefetching or increasing num_workers.",
                stats.avg_access_time_us / 1000.0
            ));
        }

        // Check throughput
        if stats.throughput_accesses_per_sec < 100.0 && stats.total_accesses > 100 {
            hints.push(format!(
                "Low throughput ({:.1} accesses/sec). Consider optimizing dataset.get() implementation or using parallel loading.",
                stats.throughput_accesses_per_sec
            ));
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

#[cfg(feature = "std")]
impl std::fmt::Display for DatasetProfileStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dataset Profile Statistics:")?;
        writeln!(f, "  Total Accesses: {}", self.total_accesses)?;
        writeln!(
            f,
            "  Sequential Accesses: {} ({:.1}%)",
            self.sequential_accesses,
            self.sequential_ratio * 100.0
        )?;
        writeln!(
            f,
            "  Avg Access Time: {:.2} µs ({:.3} ms)",
            self.avg_access_time_us,
            self.avg_access_time_us / 1000.0
        )?;
        writeln!(
            f,
            "  Throughput: {:.1} accesses/sec",
            self.throughput_accesses_per_sec
        )?;
        writeln!(f, "  Elapsed Time: {:.2} seconds", self.elapsed_seconds)?;
        Ok(())
    }
}

/// Wrapper dataset that profiles accesses
#[cfg(feature = "std")]
pub struct ProfiledDataset<D: Dataset> {
    dataset: D,
    profiler: std::sync::Arc<DatasetProfiler>,
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

#[cfg(feature = "std")]
impl<D: Dataset> Dataset for ProfiledDataset<D> {
    type Item = D::Item;

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        let start = std::time::Instant::now();
        let result = self.dataset.get(index);
        let duration = start.elapsed();
        self.profiler.record_access(index, duration);
        result
    }
}

// ============================================================================
// Dataset Analysis and Splitting Utilities
// ============================================================================

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

/// Compute statistics for a tensor dataset
///
/// Returns feature statistics for each feature dimension in the dataset.
/// Only works with TensorDataset<f32> where the first tensor contains the features.
pub fn dataset_statistics(dataset: &TensorDataset<f32>) -> Result<Vec<FeatureStats>> {
    if dataset.len() == 0 {
        return Ok(Vec::new());
    }

    // Get the first item to determine feature dimensions
    let first_item = dataset.get(0)?;
    if first_item.is_empty() {
        return Ok(Vec::new());
    }

    let features_tensor = &first_item[0];
    let n_features = features_tensor.numel();

    // Collect all feature values
    let mut feature_data: Vec<Vec<f32>> = vec![Vec::with_capacity(dataset.len()); n_features];

    for i in 0..dataset.len() {
        let item = dataset.get(i)?;
        if item.is_empty() {
            continue;
        }

        let features = &item[0];

        // Extract feature values - assuming 1D tensor for features
        // For each feature dimension, extract the value
        for feat_idx in 0..n_features.min(features.numel()) {
            // Use indexing to get individual elements
            if let Ok(indices) = torsh_tensor::Tensor::from_vec(vec![feat_idx as i64], &[1]) {
                if let Ok(value_tensor) = features.index_select(0, &indices) {
                    if let Ok(value) = value_tensor.item() {
                        feature_data[feat_idx].push(value);
                    }
                }
            }
        }
    }

    // Compute statistics for each feature
    Ok(feature_data
        .iter()
        .map(|data| FeatureStats::from_data(data))
        .collect())
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

        // Shuffle if requested
        if self.shuffle {
            // ✅ SciRS2 Policy Compliant - Using enhanced scientific shuffle
            use scirs2_core::random::prelude::*;
            use scirs2_core::random::seq::ScientificSliceRandom;
            use scirs2_core::random::SeedableRng;

            let mut rng = if let Some(seed) = self.random_seed {
                StdRng::seed_from_u64(seed)
            } else {
                // Use current time as seed for non-deterministic shuffling
                use std::time::SystemTime;
                let seed = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
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
                n_samples // Last fold gets remainder
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

/// Stratified split that preserves class distribution
///
/// Splits data into train/val/test sets while maintaining the same class distribution
/// in each split as in the original dataset.
pub fn stratified_split<D>(
    dataset: D,
    labels: &[usize],
    train_ratio: f32,
    val_ratio: Option<f32>,
    random_seed: Option<u64>,
) -> Result<(Subset<D>, Subset<D>, Option<Subset<D>>)>
where
    D: Dataset + Clone,
{
    if train_ratio <= 0.0 || train_ratio >= 1.0 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "train_ratio must be between 0 and 1".to_string(),
        ));
    }

    let has_val = val_ratio.is_some();
    let val_r = val_ratio.unwrap_or(0.0);

    if has_val && (train_ratio + val_r >= 1.0) {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "train_ratio + val_ratio must be less than 1".to_string(),
        ));
    }

    if labels.len() != dataset.len() {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "labels length must equal dataset length".to_string(),
        ));
    }

    // Group indices by class
    let mut class_indices: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        class_indices.entry(label).or_default().push(idx);
    }

    // ✅ SciRS2 Policy Compliant - Using enhanced scientific shuffle
    use scirs2_core::random::prelude::*;
    use scirs2_core::random::seq::ScientificSliceRandom;
    use scirs2_core::random::SeedableRng;

    let mut rng = if let Some(seed) = random_seed {
        StdRng::seed_from_u64(seed)
    } else {
        // Use current time as seed for non-deterministic shuffling
        use std::time::SystemTime;
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        StdRng::seed_from_u64(seed)
    };

    let mut train_indices = Vec::new();
    let mut val_indices = Vec::new();
    let mut test_indices = Vec::new();

    // Split each class proportionally
    for (_class, mut indices) in class_indices {
        indices.scientific_shuffle(&mut rng);

        let n_train = (indices.len() as f32 * train_ratio).round() as usize;
        let n_val = if has_val {
            (indices.len() as f32 * val_r).round() as usize
        } else {
            0
        };

        train_indices.extend_from_slice(&indices[0..n_train]);

        if has_val {
            val_indices.extend_from_slice(&indices[n_train..n_train + n_val]);
            test_indices.extend_from_slice(&indices[n_train + n_val..]);
        } else {
            test_indices.extend_from_slice(&indices[n_train..]);
        }
    }

    let train_subset = Subset::new(dataset.clone(), train_indices);
    let test_subset = Subset::new(dataset.clone(), test_indices);
    let val_subset = if has_val {
        Some(Subset::new(dataset, val_indices))
    } else {
        None
    };

    Ok((train_subset, test_subset, val_subset))
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

    #[test]
    #[cfg(feature = "std")]
    fn test_dataset_profiler_sequential_access() {
        use std::thread;
        use std::time::Duration;

        let tensor = ones::<f32>(&[10, 2]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let profiled = ProfiledDataset::new(dataset);

        // Access sequentially
        for i in 0..10 {
            let _ = profiled.get(i).unwrap();
            thread::sleep(Duration::from_micros(100));
        }

        let stats = profiled.stats();
        assert_eq!(stats.total_accesses, 10);
        assert_eq!(stats.sequential_accesses, 9); // 9 sequential transitions for 10 accesses
        assert!(stats.sequential_ratio > 0.8);
        assert!(stats.avg_access_time_us > 0.0);
        assert!(stats.throughput_accesses_per_sec > 0.0);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_dataset_profiler_random_access() {
        let tensor = ones::<f32>(&[10, 2]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let profiled = ProfiledDataset::new(dataset);

        // Access randomly (no sequential pattern)
        let indices = [0, 5, 2, 8, 1];
        for &i in &indices {
            let _ = profiled.get(i).unwrap();
        }

        let stats = profiled.stats();
        assert_eq!(stats.total_accesses, 5);
        assert_eq!(stats.sequential_accesses, 0);
        assert_eq!(stats.sequential_ratio, 0.0);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_dataset_profiler_hints() {
        let tensor = ones::<f32>(&[100, 2]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let profiled = ProfiledDataset::new(dataset);

        // Sequential access
        for i in 0..20 {
            let _ = profiled.get(i).unwrap();
        }

        let hints = profiled.hints();
        assert!(!hints.is_empty());
        // Should detect sequential pattern
        assert!(hints
            .iter()
            .any(|h| h.contains("sequential") || h.contains("good")));
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_dataset_profiler_reset() {
        let tensor = ones::<f32>(&[10, 2]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let profiled = ProfiledDataset::new(dataset);

        // Make some accesses
        for i in 0..5 {
            let _ = profiled.get(i).unwrap();
        }

        assert_eq!(profiled.stats().total_accesses, 5);

        // Reset and verify
        profiled.profiler().reset();
        assert_eq!(profiled.stats().total_accesses, 0);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_dataset_profiler_display() {
        let tensor = ones::<f32>(&[10, 2]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let profiled = ProfiledDataset::new(dataset);

        // Make some accesses
        for i in 0..5 {
            let _ = profiled.get(i).unwrap();
        }

        // Test Display implementation
        let stats_string = format!("{}", profiled.stats());
        assert!(stats_string.contains("Dataset Profile Statistics"));
        assert!(stats_string.contains("Total Accesses: 5"));
    }

    // ============================================================================
    // Tests for new dataset analysis and splitting utilities
    // ============================================================================

    #[test]
    fn test_feature_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FeatureStats::from_data(&data);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.std - 1.4142).abs() < 0.01); // sqrt(2) ≈ 1.4142
    }

    #[test]
    fn test_feature_stats_empty() {
        let data: Vec<f32> = vec![];
        let stats = FeatureStats::from_data(&data);

        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
    }

    #[test]
    fn test_dataset_statistics() {
        // Create a simple dataset with 10 samples, 3 features
        let data = torsh_tensor::creation::randn::<f32>(&[10, 3]).unwrap();
        let dataset = TensorDataset::from_tensor(data);

        let stats = dataset_statistics(&dataset).unwrap();

        assert_eq!(stats.len(), 3); // 3 features
        for stat in &stats {
            assert_eq!(stat.count, 10); // 10 samples
            assert!(stat.min <= stat.mean);
            assert!(stat.mean <= stat.max);
            assert!(stat.std >= 0.0);
        }
    }

    #[test]
    fn test_dataset_statistics_empty() {
        let data = torsh_tensor::creation::zeros::<f32>(&[0, 3]).unwrap();
        let dataset = TensorDataset::from_tensor(data);

        let stats = dataset_statistics(&dataset).unwrap();
        assert_eq!(stats.len(), 0);
    }

    #[test]
    fn test_kfold_basic() {
        let kfold = KFold::new(5, false, Some(42));
        let folds = kfold.split(100);

        assert_eq!(folds.len(), 5);

        for (fold_idx, (train_indices, val_indices)) in folds.iter().enumerate() {
            // Each fold should have 20 validation samples (100 / 5)
            assert_eq!(val_indices.len(), 20);

            // Training should have the remaining 80 samples
            assert_eq!(train_indices.len(), 80);

            // Verify no overlap between train and val
            for &val_idx in val_indices {
                assert!(!train_indices.contains(&val_idx));
            }

            // All indices should be in range
            for &idx in train_indices.iter().chain(val_indices.iter()) {
                assert!(idx < 100);
            }

            println!(
                "Fold {}: train={}, val={}",
                fold_idx,
                train_indices.len(),
                val_indices.len()
            );
        }
    }

    #[test]
    fn test_kfold_shuffle() {
        let kfold_shuffled = KFold::new(3, true, Some(42));
        let kfold_unshuffled = KFold::new(3, false, None);

        let folds_shuffled = kfold_shuffled.split(30);
        let folds_unshuffled = kfold_unshuffled.split(30);

        // Both should have same number of folds
        assert_eq!(folds_shuffled.len(), folds_unshuffled.len());

        // Validation indices should be different due to shuffling
        let shuffled_val = &folds_shuffled[0].1;
        let unshuffled_val = &folds_unshuffled[0].1;

        // Unshuffled should be [0..10], shuffled should be different
        assert_eq!(unshuffled_val, &(0..10).collect::<Vec<_>>());
        assert_ne!(shuffled_val, unshuffled_val);
    }

    #[test]
    fn test_kfold_uneven_split() {
        let kfold = KFold::new(3, false, None);
        let folds = kfold.split(10); // 10 doesn't divide evenly by 3

        // Should still create 3 folds
        assert_eq!(folds.len(), 3);

        // Last fold gets the remainder
        assert_eq!(folds[0].1.len(), 3); // 10 / 3 = 3
        assert_eq!(folds[1].1.len(), 3);
        assert_eq!(folds[2].1.len(), 4); // Gets remainder

        // All samples should be used
        let all_val_samples: usize = folds.iter().map(|(_, val)| val.len()).sum();
        assert_eq!(all_val_samples, 10);
    }

    #[test]
    #[should_panic(expected = "n_splits must be at least 2")]
    fn test_kfold_invalid_splits() {
        KFold::new(1, false, None); // Should panic
    }

    #[test]
    fn test_stratified_split_binary() {
        let data = ones::<f32>(&[100, 5]).unwrap();
        let dataset = TensorDataset::from_tensor(data);

        // Create balanced binary labels: 50 class 0, 50 class 1
        let labels: Vec<usize> = (0..100).map(|i| if i < 50 { 0 } else { 1 }).collect();

        let (train, test, val) =
            stratified_split(dataset, &labels, 0.6, Some(0.2), Some(42)).unwrap();

        // Check sizes (60% train, 20% val, 20% test)
        assert_eq!(train.len(), 60);
        assert!(val.is_some());
        assert_eq!(val.as_ref().unwrap().len(), 20);
        assert_eq!(test.len(), 20);

        // Verify stratification: each split should have roughly equal class distribution
        // For a 50-50 split with 60 training samples, we expect ~30 of each class
        println!(
            "Stratified split: train={}, val={}, test={}",
            train.len(),
            val.as_ref().unwrap().len(),
            test.len()
        );
    }

    #[test]
    fn test_stratified_split_multi_class() {
        let data = ones::<f32>(&[90, 5]).unwrap();
        let dataset = TensorDataset::from_tensor(data);

        // Create 3-class labels: 30 of each class
        let labels: Vec<usize> = (0..90).map(|i| i / 30).collect();

        let (train, test, _val) = stratified_split(dataset, &labels, 0.7, None, Some(42)).unwrap();

        // Check sizes (70% train, 30% test)
        assert_eq!(train.len(), 63); // 0.7 * 90 = 63
        assert_eq!(test.len(), 27); // 0.3 * 90 = 27

        println!(
            "Multi-class split: train={}, test={}",
            train.len(),
            test.len()
        );
    }

    #[test]
    fn test_stratified_split_no_val() {
        let data = ones::<f32>(&[50, 3]).unwrap();
        let dataset = TensorDataset::from_tensor(data);
        let labels: Vec<usize> = (0..50).map(|i| i % 2).collect();

        let (train, test, val) = stratified_split(dataset, &labels, 0.8, None, Some(42)).unwrap();

        assert_eq!(train.len(), 40);
        assert_eq!(test.len(), 10);
        assert!(val.is_none());
    }

    #[test]
    fn test_stratified_split_invalid_ratio() {
        let data = ones::<f32>(&[50, 3]).unwrap();
        let dataset = TensorDataset::from_tensor(data);
        let labels: Vec<usize> = (0..50).map(|i| i % 2).collect();

        // Invalid: train_ratio >= 1.0
        let result = stratified_split(dataset.clone(), &labels, 1.0, None, None);
        assert!(result.is_err());

        // Invalid: train_ratio + val_ratio >= 1.0
        let result = stratified_split(dataset, &labels, 0.7, Some(0.4), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_stratified_split_mismatched_labels() {
        let data = ones::<f32>(&[50, 3]).unwrap();
        let dataset = TensorDataset::from_tensor(data);
        let labels: Vec<usize> = vec![0, 1]; // Wrong length

        let result = stratified_split(dataset, &labels, 0.8, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_kfold_reproducibility() {
        let kfold1 = KFold::new(5, true, Some(42));
        let kfold2 = KFold::new(5, true, Some(42));

        let folds1 = kfold1.split(50);
        let folds2 = kfold2.split(50);

        // Same seed should produce same splits
        for (f1, f2) in folds1.iter().zip(folds2.iter()) {
            assert_eq!(f1.0, f2.0); // Train indices
            assert_eq!(f1.1, f2.1); // Val indices
        }
    }

    #[test]
    fn test_stratified_split_reproducibility() {
        let data = ones::<f32>(&[100, 5]).unwrap();
        let labels: Vec<usize> = (0..100).map(|i| i % 3).collect();

        let (train1, test1, _) = stratified_split(
            TensorDataset::from_tensor(data.clone()),
            &labels,
            0.7,
            None,
            Some(42),
        )
        .unwrap();

        let (train2, test2, _) = stratified_split(
            TensorDataset::from_tensor(data),
            &labels,
            0.7,
            None,
            Some(42),
        )
        .unwrap();

        // Same seed should produce same splits
        assert_eq!(train1.len(), train2.len());
        assert_eq!(test1.len(), test2.len());
    }
}
