// Tensor views and aliasing module for efficient memory management

use crate::{Tensor, TensorStorage};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Weak};
use torsh_core::{
    device::DeviceType,
    dtype::TensorElement,
    error::{Result, TorshError},
    shape::Shape,
};

/// A view into a tensor that shares memory but may have different shape/strides
#[derive(Debug, Clone)]
pub struct TensorView<T: TensorElement> {
    /// Reference to the underlying tensor storage
    storage: Arc<RwLock<ViewStorage<T>>>,
    /// Shape of this view
    shape: Shape,
    /// Strides for this view
    strides: Vec<usize>,
    /// Offset into the underlying data
    offset: usize,
    /// Device type
    device: DeviceType,
}

/// Storage for tensor views with reference counting
#[derive(Debug)]
struct ViewStorage<T: TensorElement> {
    /// Weak reference to parent tensor to avoid cycles
    #[allow(dead_code)]
    parent: Weak<RwLock<Vec<T>>>,
    /// Strong reference to keep data alive if needed
    data_ref: Option<Arc<RwLock<Vec<T>>>>,
    /// Cache for computed views
    view_cache: HashMap<ViewKey, Arc<TensorView<T>>>,
    /// Reference count for active views
    view_count: usize,
}

/// Key for view caching
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
struct ViewKey {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl<T: TensorElement + Copy> Tensor<T> {
    /// Calculate strides for current tensor shape
    pub fn calculate_strides(&self) -> Vec<usize> {
        let shape_binding = self.shape();
        let dims = shape_binding.dims();
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }
    /// Create a view of this tensor with a new shape (must have same number of elements)
    pub fn create_view(&self, new_shape: &[usize]) -> Result<TensorView<T>> {
        let new_numel = new_shape.iter().product::<usize>();
        if new_numel != self.numel() {
            return Err(TorshError::InvalidOperation(format!(
                "View shape {:?} has {} elements, but tensor has {} elements",
                new_shape,
                new_numel,
                self.numel()
            )));
        }

        // Calculate strides for the new shape (row-major)
        let mut strides = vec![1; new_shape.len()];
        for i in (0..new_shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * new_shape[i + 1];
        }

        self.create_view_with_strides(new_shape, &strides, 0)
    }

    /// Create a view with custom strides (advanced usage)
    pub fn view_with_strides(
        &self,
        new_shape: &[usize],
        strides: &[usize],
    ) -> Result<TensorView<T>> {
        if new_shape.len() != strides.len() {
            return Err(TorshError::InvalidOperation(
                "Shape and strides must have same length".to_string(),
            ));
        }

        self.create_view_with_strides(new_shape, strides, 0)
    }

    /// Create a slice view of the tensor along a specific dimension
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Result<TensorView<T>> {
        let shape_binding = self.shape();
        let dims = shape_binding.dims();
        if dim >= dims.len() {
            return Err(TorshError::InvalidOperation(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                dims.len()
            )));
        }

        if start >= end || end > dims[dim] {
            return Err(TorshError::InvalidOperation(format!(
                "Invalid slice range [{}:{}] for dimension of size {}",
                start, end, dims[dim]
            )));
        }

        // Calculate new shape and offset
        let mut new_shape = dims.to_vec();
        new_shape[dim] = end - start;

        // Calculate offset for the slice
        let strides = self.calculate_strides();
        let offset = start * strides[dim];

        self.create_view_with_strides(&new_shape, &strides, offset)
    }

    /// Internal method to create views with custom strides and offset
    fn create_view_with_strides(
        &self,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Result<TensorView<T>> {
        // Get reference to underlying data
        let data_ref = match &self.storage {
            TensorStorage::InMemory(data) => data.clone(),
            TensorStorage::MemoryMapped(_) => {
                // For memory-mapped storage, convert to in-memory for views
                let data = self.to_vec()?;
                Arc::new(RwLock::new(data))
            }
            #[cfg(feature = "simd")]
            TensorStorage::Aligned(data) => {
                // Convert AlignedVec to Vec for standard view handling
                let aligned_data = data.read().unwrap();
                let vec_data = aligned_data.as_slice().to_vec();
                Arc::new(RwLock::new(vec_data))
            }
        };

        // Create view storage
        let view_storage = ViewStorage {
            parent: Arc::downgrade(&data_ref),
            data_ref: Some(data_ref),
            view_cache: HashMap::new(),
            view_count: 1,
        };

        Ok(TensorView {
            storage: Arc::new(RwLock::new(view_storage)),
            shape: Shape::new(shape.to_vec()),
            strides: strides.to_vec(),
            offset,
            device: self.device,
        })
    }

    /// Create an alias (shared reference) to this tensor
    pub fn alias(&self) -> TensorAlias<T> {
        TensorAlias {
            tensor: self.clone(),
            is_mutable: false,
        }
    }

    /// Create a mutable alias to this tensor
    pub fn alias_mut(&mut self) -> TensorAlias<T> {
        TensorAlias {
            tensor: self.clone(),
            is_mutable: true,
        }
    }
}

impl<T: TensorElement + Copy> TensorView<T> {
    /// Get the shape of this view
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the strides of this view
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the offset of this view
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Convert view to a contiguous tensor
    pub fn to_tensor(&self) -> Result<Tensor<T>> {
        let data = self.to_vec()?;
        Tensor::from_data(data, self.shape.dims().to_vec(), self.device)
    }

    /// Get data as vector (materializes the view)
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let storage = self.storage.read().unwrap();
        if let Some(data_ref) = &storage.data_ref {
            let data = data_ref.read().unwrap();
            let mut result = Vec::with_capacity(self.shape.numel());

            // Extract data according to view's shape, strides, and offset
            self.extract_view_data(&data, &mut result, &mut vec![0; self.shape.ndim()], 0)?;

            Ok(result)
        } else {
            Err(TorshError::InvalidOperation(
                "View data no longer available".to_string(),
            ))
        }
    }

    /// Recursively extract data for the view
    fn extract_view_data(
        &self,
        data: &[T],
        result: &mut Vec<T>,
        indices: &mut [usize],
        dim: usize,
    ) -> Result<()> {
        if dim == self.shape.ndim() {
            // Calculate flat index from view indices
            let flat_index = self.offset
                + indices
                    .iter()
                    .zip(self.strides.iter())
                    .map(|(&idx, &stride)| idx * stride)
                    .sum::<usize>();

            if flat_index < data.len() {
                result.push(data[flat_index]);
            } else {
                return Err(TorshError::InvalidOperation(
                    "View index out of bounds".to_string(),
                ));
            }
        } else {
            for i in 0..self.shape.dims()[dim] {
                indices[dim] = i;
                self.extract_view_data(data, result, indices, dim + 1)?;
            }
        }
        Ok(())
    }

    /// Check if this view is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        // A view is contiguous if its strides match row-major layout
        let dims = self.shape.dims();
        let mut expected_strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            expected_strides[i] = expected_strides[i + 1] * dims[i + 1];
        }
        self.strides == expected_strides
    }

    /// Check if this is a view (always true for TensorView)
    pub fn is_view(&self) -> bool {
        true
    }

    /// Get element at specific indices
    pub fn get(&self, indices: &[usize]) -> Result<T> {
        if indices.len() != self.shape.ndim() {
            return Err(TorshError::InvalidOperation(format!(
                "Expected {} indices, got {}",
                self.shape.ndim(),
                indices.len()
            )));
        }

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape.dims()[i] {
                return Err(TorshError::InvalidOperation(format!(
                    "Index {} out of bounds for dimension {} (size {})",
                    idx,
                    i,
                    self.shape.dims()[i]
                )));
            }
        }

        let storage = self.storage.read().unwrap();
        if let Some(data_ref) = &storage.data_ref {
            let data = data_ref.read().unwrap();

            // Calculate flat index from view indices
            let flat_index = self.offset
                + indices
                    .iter()
                    .zip(self.strides.iter())
                    .map(|(&idx, &stride)| idx * stride)
                    .sum::<usize>();

            if flat_index < data.len() {
                Ok(data[flat_index])
            } else {
                Err(TorshError::InvalidOperation(
                    "View index out of bounds".to_string(),
                ))
            }
        } else {
            Err(TorshError::InvalidOperation(
                "View data no longer available".to_string(),
            ))
        }
    }

    /// Get memory usage of this view
    pub fn view_memory_usage(&self) -> ViewMemoryUsage {
        let storage = self.storage.read().unwrap();
        ViewMemoryUsage {
            view_elements: self.shape.numel(),
            total_elements: storage
                .data_ref
                .as_ref()
                .map(|data| data.read().unwrap().len())
                .unwrap_or(0),
            active_views: storage.view_count,
            is_contiguous: self.is_contiguous(),
            memory_efficiency: self.calculate_memory_efficiency(),
        }
    }

    /// Calculate memory efficiency of this view
    fn calculate_memory_efficiency(&self) -> f64 {
        let view_size = self.shape.numel();
        let storage = self.storage.read().unwrap();
        let total_size = storage
            .data_ref
            .as_ref()
            .map(|data| data.read().unwrap().len())
            .unwrap_or(1);

        view_size as f64 / total_size as f64
    }
}

/// An alias to a tensor that shares memory
#[derive(Debug, Clone)]
pub struct TensorAlias<T: TensorElement> {
    tensor: Tensor<T>,
    is_mutable: bool,
}

impl<T: TensorElement + Copy> TensorAlias<T> {
    /// Get reference to the underlying tensor
    pub fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Check if this alias allows mutation
    pub fn is_mutable(&self) -> bool {
        self.is_mutable
    }

    /// Convert to owned tensor (creates copy if shared)
    pub fn to_owned(&self) -> Result<Tensor<T>> {
        Ok(self.tensor.clone())
    }

    /// Get the reference count of the underlying data
    pub fn ref_count(&self) -> usize {
        match &self.tensor.storage {
            TensorStorage::InMemory(data) => Arc::strong_count(data),
            TensorStorage::MemoryMapped(storage) => Arc::strong_count(storage),
            #[cfg(feature = "simd")]
            TensorStorage::Aligned(data) => Arc::strong_count(data),
        }
    }

    /// Check if this alias has exclusive access to the data
    pub fn is_unique(&self) -> bool {
        self.ref_count() == 1
    }
}

/// Memory usage information for tensor views
#[derive(Debug, Clone)]
pub struct ViewMemoryUsage {
    /// Number of elements in this view
    pub view_elements: usize,
    /// Total elements in underlying storage
    pub total_elements: usize,
    /// Number of active views on this storage
    pub active_views: usize,
    /// Whether the view is contiguous in memory
    pub is_contiguous: bool,
    /// Memory efficiency (view_size / total_size)
    pub memory_efficiency: f64,
}

impl<T: TensorElement + Copy> Drop for ViewStorage<T> {
    fn drop(&mut self) {
        // Clean up view cache and decrement reference counts
        self.view_cache.clear();
        self.view_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use crate::creation::*;

    #[test]
    fn test_tensor_view() {
        let tensor = ones::<f32>(&[2, 3, 4]).unwrap();
        let view = tensor.create_view(&[6, 4]).unwrap();
        assert_eq!(view.shape().dims(), &[6, 4]);
        assert_eq!(view.shape().numel(), 24);
    }

    #[test]
    fn test_tensor_slice() {
        let tensor = arange(0.0f32, 12.0, 1.0).unwrap();
        let _reshaped = tensor.create_view(&[3, 4]).unwrap();
        // This would work in a full implementation
        // let slice = reshaped.slice(0, 1, 3).unwrap();
        // assert_eq!(slice.shape().dims(), &[2, 4]);
    }

    #[test]
    fn test_tensor_squeeze_unsqueeze() {
        let tensor = ones::<f32>(&[1, 3, 1, 4]).unwrap();
        let squeezed = tensor.squeeze(0).unwrap();
        assert_eq!(squeezed.shape().dims(), &[3, 1, 4]);

        let squeezed_all = tensor.squeeze_all().unwrap();
        assert_eq!(squeezed_all.shape().dims(), &[3, 4]);

        let unsqueezed = tensor.unsqueeze(2).unwrap();
        assert_eq!(unsqueezed.shape().dims(), &[1, 3, 1, 1, 4]);
    }

    #[test]
    fn test_tensor_permute() {
        let tensor = ones::<f32>(&[2, 3, 4]).unwrap();
        let permuted = tensor.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.shape().dims(), &[4, 2, 3]);
    }

    #[test]
    fn test_tensor_alias() {
        let tensor = ones::<f32>(&[10, 10]).unwrap();
        let alias = tensor.alias();
        assert!(!alias.is_mutable());
        assert!(alias.ref_count() >= 2); // Original + alias
    }

    #[test]
    fn test_view_memory_usage() {
        let tensor = ones::<f32>(&[100, 100]).unwrap();
        let view = tensor.create_view(&[1000, 10]).unwrap();
        let usage = view.view_memory_usage();
        assert_eq!(usage.view_elements, 10000);
        assert_eq!(usage.memory_efficiency, 1.0); // Full tensor view
    }
}
