//! Tensor stacking utilities

use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use rayon::prelude::*;

/// Consolidated tensor stacking utility that reduces code duplication
pub struct TensorStacker {
    use_parallel: bool,
    parallel_threshold: usize,
    memory_mapped: bool,
}

impl Default for TensorStacker {
    fn default() -> Self {
        Self {
            use_parallel: cfg!(feature = "std"),
            parallel_threshold: 1000,
            memory_mapped: false,
        }
    }
}

impl TensorStacker {
    /// Create a new tensor stacker with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable/disable parallel processing
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.use_parallel = enabled && cfg!(feature = "std");
        self
    }

    /// Set threshold for parallel processing
    pub fn parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Enable memory-mapped stacking for very large batches
    pub fn with_memory_mapping(mut self, enabled: bool) -> Self {
        self.memory_mapped = enabled && cfg!(all(feature = "std", feature = "mmap-support"));
        self
    }

    /// Stack tensors along the specified dimension
    pub fn stack<T: TensorElement + Copy>(
        &self,
        tensors: &[Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        if tensors.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot stack empty tensor list".to_string(),
            ));
        }

        // Validate shapes
        self.validate_shapes(tensors)?;

        // Choose stacking strategy based on configuration and batch size
        if self.memory_mapped && tensors.len() > 100 {
            self.stack_with_mmap(tensors, dim)
        } else if self.use_parallel && self.should_use_parallel(tensors) {
            self.stack_parallel(tensors, dim)
        } else {
            self.stack_sequential(tensors, dim)
        }
    }

    /// Check if all tensors have the same shape
    fn validate_shapes<T: TensorElement>(&self, tensors: &[Tensor<T>]) -> Result<()> {
        let first_shape = tensors[0].shape();
        for tensor in tensors[1..].iter() {
            if tensor.shape() != first_shape {
                return Err(TorshError::ShapeMismatch {
                    expected: first_shape.dims().to_vec(),
                    got: tensor.shape().dims().to_vec(),
                });
            }
        }
        Ok(())
    }

    /// Determine if parallel processing should be used
    fn should_use_parallel<T: TensorElement>(&self, tensors: &[Tensor<T>]) -> bool {
        tensors.len() > 4 && tensors[0].numel() > self.parallel_threshold
    }

    /// Create new shape with additional dimension
    fn create_new_shape(
        &self,
        original_dims: &[usize],
        batch_size: usize,
        dim: usize,
    ) -> Vec<usize> {
        let mut new_dims = Vec::with_capacity(original_dims.len() + 1);

        if dim == 0 {
            new_dims.push(batch_size);
            new_dims.extend_from_slice(original_dims);
        } else {
            new_dims.extend_from_slice(&original_dims[..dim.min(original_dims.len())]);
            new_dims.push(batch_size);
            if dim < original_dims.len() {
                new_dims.extend_from_slice(&original_dims[dim..]);
            }
        }

        new_dims
    }

    /// Sequential tensor stacking
    fn stack_sequential<T: TensorElement + Copy>(
        &self,
        tensors: &[Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        let new_dims = self.create_new_shape(tensors[0].shape().dims(), tensors.len(), dim);
        let tensor_size = tensors[0].numel();
        let total_elements = new_dims.iter().product::<usize>();
        let mut new_data = vec![T::from_f64(0.0).unwrap(); total_elements];

        // Copy data sequentially
        for (i, tensor) in tensors.iter().enumerate() {
            let data = tensor.to_vec()?;
            let start_idx = i * tensor_size;
            let end_idx = start_idx + tensor_size;
            new_data[start_idx..end_idx].copy_from_slice(&data);
        }

        torsh_tensor::Tensor::from_data(new_data, new_dims, tensors[0].device())
    }

    /// Parallel tensor stacking
    #[cfg(feature = "std")]
    fn stack_parallel<T: TensorElement + Copy>(
        &self,
        tensors: &[Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        let new_dims = self.create_new_shape(tensors[0].shape().dims(), tensors.len(), dim);
        let tensor_size = tensors[0].numel();
        let total_elements = new_dims.iter().product::<usize>();
        let mut new_data = vec![T::from_f64(0.0).unwrap(); total_elements];

        // Parallel data collection
        let parallel_data: std::result::Result<Vec<Vec<T>>, TorshError> =
            tensors.par_iter().map(|tensor| tensor.to_vec()).collect();
        let parallel_data = parallel_data?;

        for (i, data) in parallel_data.into_iter().enumerate() {
            let start_idx = i * tensor_size;
            let end_idx = start_idx + tensor_size;
            new_data[start_idx..end_idx].copy_from_slice(&data);
        }

        torsh_tensor::Tensor::from_data(new_data, new_dims, tensors[0].device())
    }

    /// Fallback for no_std
    #[cfg(not(feature = "std"))]
    fn stack_parallel<T: TensorElement + Copy>(
        &self,
        tensors: &[Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        self.stack_sequential(tensors, dim)
    }

    /// Memory-mapped stacking for very large batches
    #[cfg(all(feature = "std", feature = "mmap-support"))]
    fn stack_with_mmap<T: TensorElement + Copy>(
        &self,
        tensors: &[Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        // Placeholder implementation - in practice would use memory mapping
        // For now, fallback to parallel stacking
        self.stack_parallel(tensors, dim)
    }

    /// Fallback when mmap is not available
    #[cfg(not(all(feature = "std", feature = "mmap-support")))]
    fn stack_with_mmap<T: TensorElement + Copy>(
        &self,
        tensors: &[Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        if cfg!(feature = "std") {
            self.stack_parallel(tensors, dim)
        } else {
            self.stack_sequential(tensors, dim)
        }
    }
}
