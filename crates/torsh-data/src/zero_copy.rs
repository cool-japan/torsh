//! Zero-copy data loading utilities for efficient memory management
//!
//! This module provides zero-copy tensor operations and memory management
//! utilities for high-performance data processing pipelines. It enables
//! efficient handling of large datasets without unnecessary memory allocations.
//!
//! # Features
//!
//! - **Zero-copy tensors**: Work directly with existing memory without copying
//! - **Memory pools**: Reuse allocated tensors to reduce allocation overhead
//! - **Buffer management**: Efficient buffer reuse in data pipelines
//! - **Memory mapping**: Direct access to file data without loading into memory
//! - **Thread-safe operations**: Concurrent access to shared memory pools

use parking_lot::Mutex;
use std::{mem, slice};
use torsh_core::error::{Result, TorshError};

/// Zero-copy tensor wrapper that avoids unnecessary memory allocation
///
/// This struct provides a view into existing memory without copying data.
/// It can work with borrowed slices or take ownership of allocated memory.
pub struct ZeroCopyTensor<T> {
    data_ptr: *const T,
    shape: Vec<usize>,
    stride: Vec<usize>,
    capacity: usize,
    owned: bool,
}

impl<T> ZeroCopyTensor<T> {
    /// Create a zero-copy tensor from existing data without copying
    ///
    /// # Safety
    ///
    /// This function is unsafe because it directly uses raw pointers. The caller must ensure:
    /// - `data_ptr` is a valid pointer to a memory region that contains at least `capacity` elements
    /// - The memory region remains valid for the lifetime of the ZeroCopyTensor
    /// - The memory is properly aligned for type T
    /// - The shape and stride parameters correctly describe the tensor layout
    /// - No other code mutates the memory region while this tensor exists
    pub unsafe fn from_raw_parts(
        data_ptr: *const T,
        shape: Vec<usize>,
        stride: Vec<usize>,
    ) -> Self {
        let capacity = shape.iter().product();
        Self {
            data_ptr,
            shape,
            stride,
            capacity,
            owned: false,
        }
    }

    /// Create a zero-copy tensor from a slice
    ///
    /// This creates a view into the provided slice without copying data.
    /// The slice must remain valid for the lifetime of the tensor.
    pub fn from_slice(data: &[T], shape: Vec<usize>) -> Self {
        let capacity = shape.iter().product();
        assert_eq!(
            data.len(),
            capacity,
            "Data length must match tensor capacity"
        );

        let stride = Self::compute_stride(&shape);
        Self {
            data_ptr: data.as_ptr(),
            shape,
            stride,
            capacity,
            owned: false,
        }
    }

    /// Create a zero-copy tensor by taking ownership of a Vec
    ///
    /// This transfers ownership of the Vec's memory to the tensor,
    /// avoiding the need to copy data.
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        let capacity = shape.iter().product();
        assert_eq!(
            data.len(),
            capacity,
            "Data length must match tensor capacity"
        );

        let stride = Self::compute_stride(&shape);
        let data_ptr = data.as_ptr();
        mem::forget(data); // Transfer ownership to the tensor

        Self {
            data_ptr,
            shape,
            stride,
            capacity,
            owned: true,
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the stride of the tensor
    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.capacity
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.capacity == 0
    }

    /// Get data as a slice
    ///
    /// # Safety
    /// This is safe as long as the tensor was constructed properly and
    /// the underlying memory remains valid.
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data_ptr, self.capacity) }
    }

    /// Compute stride from shape (row-major order)
    ///
    /// Stride indicates how many elements to skip when moving along each dimension.
    fn compute_stride(shape: &[usize]) -> Vec<usize> {
        let mut stride = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        stride
    }

    /// Create a view into a subregion without copying data
    ///
    /// This creates a new tensor that views a slice of the current tensor.
    /// The data is not copied, only the view parameters are adjusted.
    pub fn slice_view(&self, ranges: &[(usize, usize)]) -> Result<ZeroCopyTensor<T>> {
        if ranges.len() != self.shape.len() {
            return Err(TorshError::InvalidArgument(
                "Number of slice ranges must match tensor dimensions".to_string(),
            ));
        }

        let mut new_shape = Vec::new();
        let mut offset = 0;

        for (i, &(start, end)) in ranges.iter().enumerate() {
            if start >= end || end > self.shape[i] {
                return Err(TorshError::InvalidArgument(
                    "Invalid slice range".to_string(),
                ));
            }
            new_shape.push(end - start);
            offset += start * self.stride[i];
        }

        let new_stride = self.stride.clone();
        let new_data_ptr = unsafe { self.data_ptr.add(offset) };
        let capacity = new_shape.iter().product();

        Ok(ZeroCopyTensor {
            data_ptr: new_data_ptr,
            shape: new_shape,
            stride: new_stride,
            capacity,
            owned: false,
        })
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Check if this tensor owns its memory
    pub fn is_owned(&self) -> bool {
        self.owned
    }
}

// Safety: ZeroCopyTensor can be sent between threads if T is Send
unsafe impl<T: Send> Send for ZeroCopyTensor<T> {}

// Safety: ZeroCopyTensor can be shared between threads if T is Sync
unsafe impl<T: Sync> Sync for ZeroCopyTensor<T> {}

impl<T> Drop for ZeroCopyTensor<T> {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                // Convert back to Vec and let it handle deallocation
                let _vec =
                    Vec::from_raw_parts(self.data_ptr as *mut T, self.capacity, self.capacity);
            }
        }
    }
}

/// Memory pool for reusing allocated tensors to avoid allocation/deallocation overhead
///
/// This pool maintains a collection of pre-allocated vectors that can be reused
/// to avoid the overhead of memory allocation and deallocation in tight loops.
pub struct TensorPool<T> {
    pool: Mutex<Vec<Vec<T>>>,
    max_size: usize,
}

impl<T: Clone + Default> TensorPool<T> {
    /// Create a new tensor pool
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of tensors to keep in the pool
    pub fn new(max_size: usize) -> Self {
        Self {
            pool: Mutex::new(Vec::new()),
            max_size,
        }
    }

    /// Get a tensor from the pool or allocate a new one
    ///
    /// If a suitable tensor is available in the pool, it will be reused.
    /// Otherwise, a new tensor will be allocated.
    pub fn get(&self, capacity: usize) -> Vec<T> {
        let mut pool = self.pool.lock();

        // Look for a tensor with sufficient capacity
        for i in 0..pool.len() {
            if pool[i].capacity() >= capacity {
                let mut tensor = pool.swap_remove(i);
                tensor.clear();
                tensor.resize(capacity, T::default());
                return tensor;
            }
        }

        // No suitable tensor found, allocate a new one
        vec![T::default(); capacity]
    }

    /// Return a tensor to the pool
    ///
    /// The tensor will be stored in the pool for future reuse if there's space.
    pub fn return_tensor(&self, tensor: Vec<T>) {
        let mut pool = self.pool.lock();
        if pool.len() < self.max_size {
            pool.push(tensor);
        }
        // If pool is full, the tensor will be dropped and deallocated
    }

    /// Get the number of tensors currently in the pool
    pub fn pool_size(&self) -> usize {
        self.pool.lock().len()
    }

    /// Clear all tensors from the pool
    pub fn clear(&self) {
        self.pool.lock().clear();
    }
}

/// Memory-mapped data loader for large datasets
///
/// This provides a way to access file data directly without loading it entirely into memory.
/// Useful for working with datasets larger than available RAM.
pub struct MemoryMappedLoader {
    file_path: std::path::PathBuf,
}

impl MemoryMappedLoader {
    /// Create a new memory-mapped loader
    ///
    /// # Arguments
    /// * `file_path` - Path to the file to be memory-mapped
    pub fn new<P: AsRef<std::path::Path>>(file_path: P) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();

        // Verify file exists
        if !file_path.exists() {
            return Err(TorshError::InvalidArgument(format!(
                "File does not exist: {}",
                file_path.display()
            )));
        }

        Ok(Self { file_path })
    }

    /// Get the file path
    pub fn file_path(&self) -> &std::path::Path {
        &self.file_path
    }

    /// Get file size in bytes
    pub fn file_size(&self) -> Result<u64> {
        std::fs::metadata(&self.file_path)
            .map(|metadata| metadata.len())
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to get file size: {}", e)))
    }

    /// Load data without copying (placeholder implementation)
    ///
    /// In a full implementation with memmap2 dependency, this would return
    /// a slice directly from the memory-mapped file.
    pub fn load_slice(&self, _offset: usize, _length: usize) -> Result<&[u8]> {
        // In a real implementation, this would return a slice directly from the memory map
        // using something like: &self.mmap[offset..offset + length]
        Err(TorshError::UnsupportedOperation {
            op: "memory mapping".to_string(),
            dtype: "MemoryMappedLoader".to_string(),
        })
    }

    /// Check if the file can be memory-mapped
    pub fn can_map(&self) -> bool {
        // For now, always return false since we don't have memmap2
        // In a real implementation, this would check file accessibility
        false
    }
}

/// Buffer manager for efficient buffer reuse in data pipelines
///
/// This manages a pool of pre-allocated buffers that can be acquired and released
/// by data processing operations to avoid repeated allocation/deallocation.
pub struct BufferManager<T> {
    available_buffers: Mutex<Vec<Vec<T>>>,
    max_buffers: usize,
    buffer_size: usize,
}

impl<T: Clone + Default> BufferManager<T> {
    /// Create a new buffer manager
    ///
    /// # Arguments
    /// * `max_buffers` - Maximum number of buffers to maintain
    /// * `buffer_size` - Size of each buffer in elements
    pub fn new(max_buffers: usize, buffer_size: usize) -> Self {
        let mut available_buffers = Vec::with_capacity(max_buffers);
        for _ in 0..max_buffers {
            available_buffers.push(vec![T::default(); buffer_size]);
        }

        Self {
            available_buffers: Mutex::new(available_buffers),
            max_buffers,
            buffer_size,
        }
    }

    /// Acquire a buffer from the pool
    ///
    /// Returns `Some(buffer)` if a buffer is available, `None` if all buffers are in use.
    pub fn acquire_buffer(&self) -> Option<Vec<T>> {
        let mut available = self.available_buffers.lock();
        available.pop()
    }

    /// Release a buffer back to the pool
    ///
    /// The buffer will be returned to the pool if there's space, otherwise it will be dropped.
    pub fn release_buffer(&self, buffer: Vec<T>) {
        let mut available = self.available_buffers.lock();
        if available.len() < self.max_buffers {
            available.push(buffer);
        }
    }

    /// Get number of available buffers
    pub fn available_count(&self) -> usize {
        self.available_buffers.lock().len()
    }

    /// Get number of buffers in use
    pub fn in_use_count(&self) -> usize {
        self.max_buffers - self.available_count()
    }

    /// Get the configured buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get the maximum number of buffers
    pub fn max_buffers(&self) -> usize {
        self.max_buffers
    }

    /// Reset all buffers (clear and return to pool)
    pub fn reset(&self) {
        let mut available = self.available_buffers.lock();
        available.clear();
        for _ in 0..self.max_buffers {
            available.push(vec![T::default(); self.buffer_size]);
        }
    }
}

/// Convenience function to create a zero-copy tensor from a vector
pub fn zero_copy_from_vec<T>(data: Vec<T>, shape: Vec<usize>) -> ZeroCopyTensor<T> {
    ZeroCopyTensor::from_vec(data, shape)
}

/// Convenience function to create a zero-copy tensor from a slice
pub fn zero_copy_from_slice<T>(data: &[T], shape: Vec<usize>) -> ZeroCopyTensor<T> {
    ZeroCopyTensor::from_slice(data, shape)
}

/// Convenience function to create a tensor pool
pub fn create_tensor_pool<T: Clone + Default>(max_size: usize) -> TensorPool<T> {
    TensorPool::new(max_size)
}

/// Convenience function to create a buffer manager
pub fn create_buffer_manager<T: Clone + Default>(
    max_buffers: usize,
    buffer_size: usize,
) -> BufferManager<T> {
    BufferManager::new(max_buffers, buffer_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_tensor_from_vec() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = ZeroCopyTensor::from_vec(data, shape.clone());

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.len(), 6);
        assert!(!tensor.is_empty());
        assert!(tensor.is_owned());
        assert_eq!(tensor.ndim(), 2);
    }

    #[test]
    fn test_zero_copy_tensor_from_slice() {
        let data = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let tensor = ZeroCopyTensor::from_slice(&data, shape.clone());

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        assert!(!tensor.is_owned());
        assert_eq!(tensor.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_zero_copy_tensor_slice_view() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let shape = vec![3, 3];
        let tensor = ZeroCopyTensor::from_vec(data, shape);

        // Create a 2x2 slice starting at (1,1)
        let ranges = vec![(1, 3), (1, 3)];
        let slice_view = tensor.slice_view(&ranges).unwrap();

        assert_eq!(slice_view.shape(), &[2, 2]);
        assert_eq!(slice_view.len(), 4);
        assert!(!slice_view.is_owned());
    }

    #[test]
    fn test_tensor_pool() {
        let pool = TensorPool::<f32>::new(3);
        assert_eq!(pool.pool_size(), 0);

        // Get a tensor
        let tensor1 = pool.get(10);
        assert_eq!(tensor1.len(), 10);

        // Return it to the pool
        pool.return_tensor(tensor1);
        assert_eq!(pool.pool_size(), 1);

        // Get it back (should be reused)
        let tensor2 = pool.get(10);
        assert_eq!(tensor2.len(), 10);
        assert_eq!(pool.pool_size(), 0);

        pool.return_tensor(tensor2);
        pool.clear();
        assert_eq!(pool.pool_size(), 0);
    }

    #[test]
    fn test_buffer_manager() {
        let manager = BufferManager::<u8>::new(2, 100);
        assert_eq!(manager.available_count(), 2);
        assert_eq!(manager.in_use_count(), 0);
        assert_eq!(manager.buffer_size(), 100);
        assert_eq!(manager.max_buffers(), 2);

        // Acquire buffers
        let buffer1 = manager.acquire_buffer().unwrap();
        assert_eq!(buffer1.len(), 100);
        assert_eq!(manager.available_count(), 1);

        let buffer2 = manager.acquire_buffer().unwrap();
        assert_eq!(manager.available_count(), 0);

        // No more buffers available
        assert!(manager.acquire_buffer().is_none());

        // Release buffers
        manager.release_buffer(buffer1);
        assert_eq!(manager.available_count(), 1);

        manager.release_buffer(buffer2);
        assert_eq!(manager.available_count(), 2);

        // Test reset
        manager.reset();
        assert_eq!(manager.available_count(), 2);
    }

    #[test]
    fn test_memory_mapped_loader() {
        // Test with a non-existent file
        let result = MemoryMappedLoader::new("/non/existent/file");
        assert!(result.is_err());

        // Test loading slice (should fail with unsupported operation)
        if let Ok(loader) = MemoryMappedLoader::new("/dev/null") {
            let result = loader.load_slice(0, 10);
            assert!(result.is_err());
            assert!(!loader.can_map());
        }
    }

    #[test]
    fn test_stride_computation() {
        // Test 2D tensor stride
        let stride = ZeroCopyTensor::<f32>::compute_stride(&[3, 4]);
        assert_eq!(stride, vec![4, 1]);

        // Test 3D tensor stride
        let stride = ZeroCopyTensor::<f32>::compute_stride(&[2, 3, 4]);
        assert_eq!(stride, vec![12, 4, 1]);

        // Test 1D tensor stride
        let stride = ZeroCopyTensor::<f32>::compute_stride(&[5]);
        assert_eq!(stride, vec![1]);
    }

    #[test]
    fn test_convenience_functions() {
        let data = vec![1, 2, 3, 4];
        let shape = vec![2, 2];

        let _tensor_from_vec = zero_copy_from_vec(data.clone(), shape.clone());
        let _tensor_from_slice = zero_copy_from_slice(&data, shape);
        let _pool = create_tensor_pool::<f32>(10);
        let _manager = create_buffer_manager::<u8>(5, 100);
    }

    #[test]
    #[should_panic(expected = "Data length must match tensor capacity")]
    fn test_shape_mismatch() {
        let data = vec![1, 2, 3];
        let shape = vec![2, 2]; // Requires 4 elements, but data has 3
        ZeroCopyTensor::from_vec(data, shape);
    }
}
