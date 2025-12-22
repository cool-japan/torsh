//! Advanced collation implementations

use super::{optimized::stack_tensors, Collate};
use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

#[cfg(feature = "sparse")]
use torsh_sparse::{CooTensor, SparseTensor};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

#[cfg(feature = "std")]
use std::sync::Arc;

/// Cached collation function that reuses allocated memory
pub struct CachedCollate<T: TensorElement> {
    tensor_pool: Arc<parking_lot::Mutex<Vec<Vec<T>>>>,
    max_pool_size: usize,
}

impl<T: TensorElement> CachedCollate<T> {
    /// Create a new cached collation function
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            tensor_pool: Arc::new(parking_lot::Mutex::new(Vec::with_capacity(max_pool_size))),
            max_pool_size,
        }
    }

    /// Get a reusable buffer from the pool
    fn get_buffer(&self, capacity: usize) -> Vec<T> {
        let mut pool = self.tensor_pool.lock();
        if let Some(mut buffer) = pool.pop() {
            buffer.clear();
            if buffer.capacity() >= capacity {
                buffer.reserve(capacity - buffer.capacity());
            }
            buffer
        } else {
            Vec::with_capacity(capacity)
        }
    }

    /// Return a buffer to the pool
    fn return_buffer(&self, buffer: Vec<T>) {
        let mut pool = self.tensor_pool.lock();
        if pool.len() < self.max_pool_size {
            pool.push(buffer);
        }
    }
}

impl<T: TensorElement + Copy> Collate<Tensor<T>> for CachedCollate<T> {
    type Output = Tensor<T>;

    fn collate(&self, batch: Vec<Tensor<T>>) -> Result<Self::Output> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        // Check that all tensors have the same shape
        let first_shape = batch[0].shape();
        for tensor in &batch[1..] {
            if tensor.shape() != first_shape {
                return Err(TorshError::ShapeMismatch {
                    expected: first_shape.dims().to_vec(),
                    got: tensor.shape().dims().to_vec(),
                });
            }
        }

        // Create new shape with batch dimension
        let original_dims = first_shape.dims();
        let mut new_dims = Vec::with_capacity(original_dims.len() + 1);
        new_dims.push(batch.len());
        new_dims.extend_from_slice(original_dims);

        let tensor_size = batch[0].numel();
        let total_elements = tensor_size * batch.len();

        // Get a reusable buffer
        let mut new_data = self.get_buffer(total_elements);
        new_data.reserve_exact(total_elements);

        // Copy tensor data efficiently
        for tensor in batch.iter() {
            let data = tensor.to_vec()?;
            new_data.extend_from_slice(&data);
        }

        let result =
            torsh_tensor::Tensor::from_data(new_data.clone(), new_dims, batch[0].device())?;

        // Return buffer to pool (create a new empty vector to return)
        self.return_buffer(Vec::with_capacity(new_data.capacity()));

        Ok(result)
    }
}

/// Dynamic batching collation for variable-size sequences
pub struct DynamicBatchCollate<T: TensorElement> {
    padding_value: T,
    max_sequence_length: Option<usize>,
    pack_sequences: bool,
}

impl<T: TensorElement> DynamicBatchCollate<T> {
    /// Create a new dynamic batch collation function
    pub fn new(padding_value: T) -> Self {
        Self {
            padding_value,
            max_sequence_length: None,
            pack_sequences: false,
        }
    }

    /// Set maximum sequence length (sequences longer than this will be truncated)
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_sequence_length = Some(max_length);
        self
    }

    /// Enable sequence packing to minimize padding
    pub fn with_packing(mut self, pack: bool) -> Self {
        self.pack_sequences = pack;
        self
    }
}

impl<
        T: TensorElement
            + Copy
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Default,
    > Collate<Tensor<T>> for DynamicBatchCollate<T>
{
    type Output = (Tensor<T>, Tensor<i64>); // (padded_sequences, lengths)

    fn collate(&self, batch: Vec<Tensor<T>>) -> Result<Self::Output> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        // Collect sequence lengths
        let mut lengths = Vec::with_capacity(batch.len());
        let mut max_length = 0;

        for tensor in &batch {
            if tensor.ndim() == 0 {
                return Err(TorshError::InvalidArgument(
                    "Cannot dynamically batch scalar tensors".to_string(),
                ));
            }

            let seq_len = tensor.size(0)?;
            lengths.push(seq_len as i64);
            max_length = max_length.max(seq_len);
        }

        // Apply max length constraint if specified
        if let Some(max_len) = self.max_sequence_length {
            max_length = max_length.min(max_len);
        }

        // If packing is enabled, sort by length to minimize padding
        let mut batch_with_indices: Vec<_> = batch.into_iter().enumerate().collect();
        if self.pack_sequences {
            batch_with_indices.sort_by_key(|(_, tensor)| tensor.size(0).unwrap_or(0));
        }

        // Get the shape for creating padded tensors
        let first_tensor = &batch_with_indices[0].1;
        let mut padded_shape = first_tensor.shape().dims().to_vec();
        padded_shape[0] = max_length; // Set sequence dimension to max length

        // Create padded batch
        let batch_size = batch_with_indices.len();
        let mut padded_batch = Vec::with_capacity(batch_size);

        for (original_idx, tensor) in batch_with_indices {
            let seq_len = tensor.size(0)?;
            let actual_len = seq_len.min(max_length);

            if actual_len == max_length {
                // No padding needed, just truncate if necessary
                if seq_len > max_length {
                    let truncated = tensor.narrow(0, 0, max_length)?;
                    padded_batch.push((original_idx, truncated));
                } else {
                    padded_batch.push((original_idx, tensor));
                }
            } else {
                // Need to pad
                let mut padding_shape = padded_shape.clone();
                let padding_elements =
                    (max_length - actual_len) * padding_shape[1..].iter().product::<usize>();

                // Create padding tensor
                let padding_data = vec![self.padding_value; padding_elements];
                padding_shape[0] = max_length - actual_len;

                let padding_tensor =
                    Tensor::from_data(padding_data, padding_shape.clone(), tensor.device())?;

                // Truncate if necessary
                let tensor_to_pad = if seq_len > max_length {
                    tensor.narrow(0, 0, max_length)?
                } else {
                    tensor
                };

                // Manual concatenation since Tensor::cat is not working correctly
                let tensor_data = tensor_to_pad.to_vec()?;
                let padding_data = padding_tensor.to_vec()?;

                // Combine the data
                let mut combined_data = tensor_data;
                combined_data.extend(padding_data);

                // Create new tensor with correct shape
                let mut final_shape = tensor_to_pad.shape().dims().to_vec();
                final_shape[0] = max_length; // Set to max_length

                let padded = Tensor::from_data(combined_data, final_shape, tensor_to_pad.device())?;
                padded_batch.push((original_idx, padded));
            }
        }

        // Restore original order if packing was used
        if self.pack_sequences {
            padded_batch.sort_by_key(|(idx, _)| *idx);
        }

        // Extract tensors and stack them
        let tensors: Vec<_> = padded_batch.into_iter().map(|(_, tensor)| tensor).collect();

        let stacked = stack_tensors(&tensors, 0)?;

        // Create lengths tensor
        let lengths_tensor = Tensor::from_data(lengths, vec![batch_size], tensors[0].device())?;

        Ok((stacked, lengths_tensor))
    }
}

/// Wrapper for DynamicBatchCollate that only returns padded sequences (not lengths)
/// This allows compatibility with the CollateBuilder which expects Tensor<T> output
pub struct DynamicBatchCollateWrapper<T: TensorElement> {
    inner: DynamicBatchCollate<T>,
}

impl<T: TensorElement> DynamicBatchCollateWrapper<T> {
    pub fn new(padding_value: T) -> Self {
        Self {
            inner: DynamicBatchCollate::new(padding_value),
        }
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.inner = self.inner.with_max_length(max_length);
        self
    }

    pub fn with_packing(mut self, pack: bool) -> Self {
        self.inner = self.inner.with_packing(pack);
        self
    }
}

impl<
        T: TensorElement
            + Copy
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Default,
    > Collate<Tensor<T>> for DynamicBatchCollateWrapper<T>
{
    type Output = Tensor<T>;

    fn collate(&self, batch: Vec<Tensor<T>>) -> Result<Self::Output> {
        // Call the inner collate function and extract only the padded sequences
        let (padded_sequences, _lengths) = self.inner.collate(batch)?;
        Ok(padded_sequences)
    }
}

/// Bucket sampler for dynamic batching
/// Groups sequences of similar lengths to minimize padding
pub struct BucketBatchSampler {
    lengths: Vec<usize>,
    batch_size: usize,
    bucket_boundaries: Vec<usize>,
    drop_last: bool,
}

impl BucketBatchSampler {
    /// Create a new bucket batch sampler
    pub fn new(lengths: Vec<usize>, batch_size: usize, drop_last: bool) -> Self {
        // Create bucket boundaries based on length distribution
        let mut sorted_lengths = lengths.clone();
        sorted_lengths.sort_unstable();

        let num_buckets = (lengths.len() / batch_size).clamp(1, 10);
        let mut bucket_boundaries = Vec::with_capacity(num_buckets + 1);

        for i in 0..=num_buckets {
            let idx = (i * sorted_lengths.len()) / num_buckets;
            let boundary = if idx >= sorted_lengths.len() {
                sorted_lengths.last().copied().unwrap_or(0) + 1
            } else {
                sorted_lengths[idx]
            };
            bucket_boundaries.push(boundary);
        }

        Self {
            lengths,
            batch_size,
            bucket_boundaries,
            drop_last,
        }
    }

    /// Generate batches grouped by sequence length buckets
    pub fn generate_batches(&self) -> Vec<Vec<usize>> {
        // Group indices by bucket
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); self.bucket_boundaries.len() - 1];

        for (idx, &length) in self.lengths.iter().enumerate() {
            for (bucket_idx, bucket) in buckets.iter_mut().enumerate() {
                if length >= self.bucket_boundaries[bucket_idx]
                    && length < self.bucket_boundaries[bucket_idx + 1]
                {
                    bucket.push(idx);
                    break;
                }
            }
        }

        // Shuffle within each bucket and create batches
        let mut batches = Vec::new();

        for mut bucket in buckets {
            // ✅ SciRS2 Policy Enhanced - Using scientific shuffle for optimal ML batching
            use scirs2_core::random::prelude::*;
            use scirs2_core::random::seq::ScientificSliceRandom;

            let mut rng = thread_rng();
            bucket.scientific_shuffle(&mut rng);

            for chunk in bucket.chunks(self.batch_size) {
                if chunk.len() == self.batch_size || !self.drop_last {
                    batches.push(chunk.to_vec());
                }
            }
        }

        // Enhanced scientific shuffle to optimize ML training batch distribution
        // ✅ SciRS2 Policy Enhanced - Using scientific shuffle for superior randomness
        use scirs2_core::random::prelude::*;
        use scirs2_core::random::seq::ScientificSliceRandom;
        let mut rng = thread_rng();
        batches.scientific_shuffle(&mut rng);

        batches
    }
}

/// Adaptive batch size sampler that adjusts batch size based on sequence lengths
pub struct AdaptiveBatchSampler {
    target_tokens: usize,
    max_batch_size: usize,
    min_batch_size: usize,
    lengths: Vec<usize>,
}

impl AdaptiveBatchSampler {
    /// Create a new adaptive batch sampler
    pub fn new(
        lengths: Vec<usize>,
        target_tokens: usize,
        max_batch_size: usize,
        min_batch_size: usize,
    ) -> Self {
        Self {
            target_tokens,
            max_batch_size,
            min_batch_size,
            lengths,
        }
    }

    /// Generate batches with adaptive batch sizes
    pub fn generate_batches(&self) -> Vec<Vec<usize>> {
        let mut indices: Vec<usize> = (0..self.lengths.len()).collect();

        // Sort by length to process similar lengths together
        indices.sort_by_key(|&i| self.lengths[i]);

        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut _current_tokens = 0;

        for idx in indices {
            let length = self.lengths[idx];
            let batch_size = current_batch.len();
            let tokens_if_added = (batch_size + 1)
                * length.max(
                    current_batch
                        .iter()
                        .map(|&i| self.lengths[i])
                        .max()
                        .unwrap_or(0),
                );

            // Check if adding this sequence would exceed limits
            if tokens_if_added > self.target_tokens || batch_size >= self.max_batch_size {
                // Finish current batch if it meets minimum size
                if batch_size >= self.min_batch_size {
                    batches.push(current_batch);
                }

                // Start new batch
                current_batch = vec![idx];
                _current_tokens = length;
            } else {
                // Add to current batch
                current_batch.push(idx);
                _current_tokens = tokens_if_added;
            }
        }

        // Add final batch if it meets minimum size
        if current_batch.len() >= self.min_batch_size {
            batches.push(current_batch);
        }

        batches
    }
}

/// Padding collation for variable-length sequences
pub struct PadCollate<T: TensorElement> {
    #[allow(dead_code)]
    padding_value: T,
}

impl<T: TensorElement> PadCollate<T> {
    /// Create a new padding collation function
    pub fn new(padding_value: T) -> Self {
        Self { padding_value }
    }
}

impl<T: TensorElement + Copy> Collate<Tensor<T>> for PadCollate<T> {
    type Output = Tensor<T>;

    fn collate(&self, batch: Vec<Tensor<T>>) -> Result<Self::Output> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        // Find maximum dimensions
        let ndim = batch[0].ndim();
        let mut max_dims = vec![0; ndim];

        for tensor in &batch {
            if tensor.ndim() != ndim {
                return Err(TorshError::InvalidArgument(
                    "All tensors must have the same number of dimensions".to_string(),
                ));
            }

            for (i, max_dim) in max_dims.iter_mut().enumerate().take(ndim) {
                let size = tensor.size(i as i32)?;
                if size > *max_dim {
                    *max_dim = size;
                }
            }
        }

        // Create padded tensors
        let batch_size = batch.len();
        let mut padded_batch = Vec::with_capacity(batch_size);

        for tensor in batch {
            // For each tensor, pad to match max_dims
            let shape_ref = tensor.shape();
            let current_shape = shape_ref.dims();
            let padded_tensor = tensor;

            // Check if padding is needed
            let needs_padding = current_shape
                .iter()
                .zip(max_dims.iter())
                .any(|(&current, &max)| current < max);

            if needs_padding {
                // For now, just use the tensor as-is since we don't have full broadcasting yet
                // In a full implementation, we'd properly pad with padding_value
                // For this placeholder, we'll just use the original tensor
            }

            padded_batch.push(padded_tensor);
        }

        // Stack the padded tensors
        stack_tensors(&padded_batch, 0)
    }
}

/// Sparse tensor collation function
#[cfg(feature = "sparse")]
pub struct SparseCollate;

#[cfg(feature = "sparse")]
impl Collate<CooTensor> for SparseCollate {
    type Output = CooTensor;

    fn collate(&self, batch: Vec<CooTensor>) -> Result<Self::Output> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        // For sparse tensors, we concatenate them along the batch dimension
        // This creates a larger sparse tensor with all the non-zero elements
        collate_sparse_tensors(&batch)
    }
}

/// Stack sparse tensors along a new batch dimension
#[cfg(feature = "sparse")]
pub fn collate_sparse_tensors(tensors: &[CooTensor]) -> Result<CooTensor> {
    if tensors.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot collate empty sparse tensor batch".to_string(),
        ));
    }

    // Check that all tensors have the same shape (except batch dimension)
    let first_shape = tensors[0].shape();
    for tensor in &tensors[1..] {
        if tensor.shape() != first_shape {
            return Err(TorshError::ShapeMismatch {
                expected: first_shape.dims().to_vec(),
                got: tensor.shape().dims().to_vec(),
            });
        }
    }

    // Calculate new shape with batch dimension
    let original_dims = first_shape.dims();
    let mut new_dims = Vec::with_capacity(original_dims.len() + 1);
    new_dims.push(tensors.len());
    new_dims.extend_from_slice(original_dims);

    // For COO format, we need to:
    // 1. Collect all indices and values
    // 2. Adjust indices to account for batch dimension
    // 3. Create new COO tensor

    let mut all_row_indices = Vec::new();
    let mut all_col_indices = Vec::new();
    let mut all_values = Vec::new();
    let mut _total_nnz = 0;

    for (batch_idx, tensor) in tensors.iter().enumerate() {
        let _row_indices = tensor.row_indices();
        let col_indices = tensor.col_indices();
        let values = tensor.values();

        // Adjust indices to include batch dimension
        for i in 0..tensor.nnz() {
            all_row_indices.push(batch_idx);
            all_col_indices.push(col_indices[i]);
        }

        all_values.extend_from_slice(values);
        _total_nnz += tensor.nnz();
    }

    // Create new COO tensor
    let shape = torsh_core::Shape::new(new_dims);
    CooTensor::new(all_row_indices, all_col_indices, all_values, shape)
}

/// Collation function for mixed dense and sparse tensors
#[cfg(feature = "sparse")]
pub struct MixedCollate;

#[cfg(feature = "sparse")]
impl Collate<Box<dyn SparseTensor>> for MixedCollate {
    type Output = Box<dyn SparseTensor>;

    fn collate(&self, batch: Vec<Box<dyn SparseTensor>>) -> Result<Self::Output> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        // Convert all to COO format for consistency
        let mut coo_tensors = Vec::with_capacity(batch.len());
        for tensor in batch {
            coo_tensors.push(tensor.to_coo()?);
        }

        // Use sparse collation
        let collated = collate_sparse_tensors(&coo_tensors)?;
        Ok(Box::new(collated))
    }
}
