//! # MemoryMappedDataset - Trait Implementations
//!
//! This module contains trait implementations for `MemoryMappedDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;
use torsh_tensor::Tensor;

use super::functions::Dataset;
use super::types::MemoryMappedDataset;

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
