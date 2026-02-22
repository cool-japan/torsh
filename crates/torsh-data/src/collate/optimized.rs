//! Optimized collation implementations

// Used in both std and no_std feature branches
#[allow(unused_imports)]
use super::stacking::TensorStacker;
use crate::collate::Collate;
use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ✅ SciRS2 POLICY: Use scirs2_core::parallel_ops instead of rayon::prelude
#[cfg(feature = "std")]
use scirs2_core::parallel_ops::*;

/// Stack tensors along a new dimension (optimized version)
pub fn stack_tensors<T: TensorElement + Copy>(
    tensors: &[Tensor<T>],
    dim: usize,
) -> Result<Tensor<T>> {
    if tensors.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot stack empty tensor list".to_string(),
        ));
    }

    // Check that all tensors have the same shape
    let first_shape = tensors[0].shape();
    for tensor in &tensors[1..] {
        if tensor.shape() != first_shape {
            return Err(TorshError::ShapeMismatch {
                expected: first_shape.dims().to_vec(),
                got: tensor.shape().dims().to_vec(),
            });
        }
    }

    // Create new shape with additional dimension at the specified position
    let original_dims = first_shape.dims();
    let mut new_dims = Vec::with_capacity(original_dims.len() + 1);

    // Insert batch dimension at the specified position
    if dim == 0 {
        new_dims.push(tensors.len());
        new_dims.extend_from_slice(original_dims);
    } else {
        // Insert at position dim
        new_dims.extend_from_slice(&original_dims[..dim.min(original_dims.len())]);
        new_dims.push(tensors.len());
        if dim < original_dims.len() {
            new_dims.extend_from_slice(&original_dims[dim..]);
        }
    }

    // Optimized stacking: pre-allocate without unnecessary initialization
    // Use with_capacity + unsafe set_len for better performance when we know
    // we'll immediately overwrite all values
    let tensor_size = tensors[0].numel();
    let total_elements = new_dims.iter().product::<usize>();
    let mut new_data = Vec::with_capacity(total_elements);
    // SAFETY: We immediately fill all elements below, so uninitialized memory is never read
    unsafe { new_data.set_len(total_elements) };

    // Use parallel processing for large batches when std feature is available
    #[cfg(feature = "std")]
    {
        if tensors.len() > 4 && tensor_size > 1000 {
            // Parallel data collection for large tensors
            let parallel_data: std::result::Result<Vec<Vec<T>>, TorshError> =
                tensors.par_iter().map(|tensor| tensor.to_vec()).collect();
            let parallel_data = parallel_data?;
            for (i, data) in parallel_data.into_iter().enumerate() {
                let start_idx = i * tensor_size;
                let end_idx = start_idx + tensor_size;
                new_data[start_idx..end_idx].copy_from_slice(&data);
            }
        } else {
            // Sequential copy for small tensors/batches
            for (i, tensor) in tensors.iter().enumerate() {
                let data = tensor.to_vec()?;
                let start_idx = i * tensor_size;
                let end_idx = start_idx + tensor_size;
                new_data[start_idx..end_idx].copy_from_slice(&data);
            }
        }
    }

    #[cfg(not(feature = "std"))]
    {
        // Sequential copy for no_std
        for (i, tensor) in tensors.iter().enumerate() {
            let data = tensor.to_vec()?;
            let start_idx = i * tensor_size;
            let end_idx = start_idx + tensor_size;
            new_data[start_idx..end_idx].copy_from_slice(&data);
        }
    }

    let result = torsh_tensor::Tensor::from_data(new_data, new_dims, tensors[0].device())?;

    Ok(result)
}

/// Fast stack tensors using memory mapping for very large batches
#[cfg(feature = "std")]
pub fn stack_tensors_fast<T: TensorElement + Copy>(
    tensors: &[Tensor<T>],
    dim: usize,
) -> Result<Tensor<T>> {
    if tensors.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot stack empty tensor list".to_string(),
        ));
    }

    // For very large batches (>100 tensors), use memory mapped approach if available
    #[cfg(feature = "mmap-support")]
    {
        if tensors.len() > 100 {
            return stack_tensors_mmap(tensors, dim);
        }
    }

    // Otherwise use regular optimized stacking
    stack_tensors(tensors, dim)
}

/// Memory-mapped tensor stacking for very large batches
#[cfg(all(feature = "std", feature = "mmap-support"))]
pub fn stack_tensors_mmap<T: TensorElement + Copy>(
    tensors: &[Tensor<T>],
    dim: usize,
) -> Result<Tensor<T>> {
    // Check that all tensors have the same shape
    let first_shape = tensors[0].shape();
    for tensor in &tensors[1..] {
        if tensor.shape() != first_shape {
            return Err(TorshError::ShapeMismatch {
                expected: first_shape.dims().to_vec(),
                got: tensor.shape().dims().to_vec(),
            });
        }
    }

    // Create new shape
    let original_dims = first_shape.dims();
    let mut new_dims = Vec::with_capacity(original_dims.len() + 1);

    if dim == 0 {
        new_dims.push(tensors.len());
        new_dims.extend_from_slice(original_dims);
    } else {
        new_dims.extend_from_slice(&original_dims[..dim.min(original_dims.len())]);
        new_dims.push(tensors.len());
        if dim < original_dims.len() {
            new_dims.extend_from_slice(&original_dims[dim..]);
        }
    }

    let tensor_size = tensors[0].numel();
    let total_size = tensor_size * tensors.len() * std::mem::size_of::<T>();

    // Create a temporary file for memory mapping
    let mut temp_file =
        tempfile::NamedTempFile::new().map_err(|e| TorshError::IoError(e.to_string()))?;

    // Write tensor data to temp file in parallel
    temp_file
        .as_file_mut()
        .set_len(total_size as u64)
        .map_err(|e| TorshError::IoError(e.to_string()))?;

    // Use memory mapping for efficient data transfer
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .map_mut(temp_file.as_file())
            .map_err(|e| TorshError::IoError(e.to_string()))?
    };

    // Parallel collection of tensor data, then sequential copy to memory mapped region
    let all_data: std::result::Result<Vec<Vec<T>>, TorshError> =
        tensors.par_iter().map(|tensor| tensor.to_vec()).collect();
    let all_data = all_data?;

    // Sequential copy to memory mapped region for thread safety
    let mmap_ptr = mmap.as_ptr() as *mut T;
    for (i, data) in all_data.iter().enumerate() {
        unsafe {
            let dst = mmap_ptr.add(i * tensor_size);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, tensor_size);
        }
    }

    // Create tensor from memory mapped data
    unsafe {
        let data_slice =
            std::slice::from_raw_parts(mmap_ptr as *const T, tensor_size * tensors.len());
        let data_vec = data_slice.to_vec();
        let result = torsh_tensor::Tensor::from_data(data_vec, new_dims, tensors[0].device())?;
        Ok(result)
    }
}

/// Optimized collation function for high-performance scenarios
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy)]
pub struct OptimizedCollate;

#[cfg(feature = "std")]
impl<T: TensorElement + Copy> Collate<Tensor<T>> for OptimizedCollate {
    type Output = Tensor<T>;

    fn collate(&self, batch: Vec<Tensor<T>>) -> Result<Self::Output> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        // Use fast stacking with memory mapping for large batches
        stack_tensors_fast(&batch, 0)
    }
}

#[cfg(feature = "std")]
impl<T: TensorElement + Copy> Collate<Vec<Tensor<T>>> for OptimizedCollate {
    type Output = Vec<Tensor<T>>;

    fn collate(&self, batch: Vec<Vec<Tensor<T>>>) -> Result<Self::Output> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        let num_tensors = batch[0].len();
        let mut collated = Vec::with_capacity(num_tensors);

        // Process each tensor position in parallel
        (0..num_tensors)
            .into_par_iter()
            .map(|i| {
                let tensors: Vec<Tensor<T>> =
                    batch.iter().map(|sample| sample[i].clone()).collect();
                stack_tensors_fast(&tensors, 0)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .for_each(|tensor| collated.push(tensor));

        Ok(collated)
    }
}

/// Optimized collation function factory
#[cfg(feature = "std")]
pub fn optimized_collate_fn<T>() -> OptimizedCollate {
    OptimizedCollate
}

/// For no_std environments, provide a fallback OptimizedCollate that uses the TensorStacker
#[cfg(not(feature = "std"))]
#[derive(Debug, Clone, Copy)]
pub struct OptimizedCollate;

#[cfg(not(feature = "std"))]
impl<T: TensorElement + Copy> Collate<Tensor<T>> for OptimizedCollate {
    type Output = Tensor<T>;

    fn collate(&self, batch: Vec<Tensor<T>>) -> Result<Self::Output> {
        TensorStacker::new().stack(&batch, 0)
    }
}

#[cfg(not(feature = "std"))]
impl<T: TensorElement + Copy> Collate<Vec<Tensor<T>>> for OptimizedCollate {
    type Output = Vec<Tensor<T>>;

    fn collate(&self, batch: Vec<Vec<Tensor<T>>>) -> Result<Self::Output> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        let num_tensors = batch[0].len();
        let mut collated = Vec::with_capacity(num_tensors);
        let stacker = TensorStacker::new();

        // Collate each tensor position across the batch
        for i in 0..num_tensors {
            let tensors: Vec<Tensor<T>> = batch.iter().map(|sample| sample[i].clone()).collect();
            collated.push(stacker.stack(&tensors, 0)?);
        }

        Ok(collated)
    }
}

/// No_std optimized collation function factory
#[cfg(not(feature = "std"))]
pub fn optimized_collate_fn<T>() -> OptimizedCollate {
    OptimizedCollate
}
