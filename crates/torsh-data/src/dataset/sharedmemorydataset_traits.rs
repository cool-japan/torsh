//! # SharedMemoryDataset - Trait Implementations
//!
//! This module contains trait implementations for `SharedMemoryDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;
use torsh_tensor::Tensor;

use super::functions::Dataset;
use super::types::SharedMemoryDataset;

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
        let shared_data = self
            .shared_data
            .read()
            .expect("lock should not be poisoned");
        let metadata = self.metadata.read().expect("lock should not be poisoned");
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
            unsafe {
                let data_ptr = data_slice.as_ptr() as *const T;
                let data_slice_typed =
                    std::slice::from_raw_parts(data_ptr, meta.size / meta.dtype_size);
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
