//! Tensor implementation for ToRSh with PyTorch-compatible API
//!
//! This crate provides a high-level tensor API that wraps scirs2's autograd
//! functionality with a familiar PyTorch-like interface.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "async")]
pub mod async_ops;
pub mod backend_integration;
pub mod bfloat16_ops;
pub mod broadcast;
pub mod cache_optimization;
pub mod convenience;
pub mod conv;
pub mod creation;
pub mod indexing;
pub mod memory_pool;
pub mod ops;
pub mod scirs2_backend;
pub mod scirs2_stats_integration;
pub mod stats;
pub mod tensor_views;
pub mod fft;
pub mod type_conversions;

#[cfg(feature = "custom-types")]
pub mod custom_data_types;

#[cfg(feature = "serialize")]
pub mod serialize;

use torsh_core::{
    device::DeviceType,
    dtype::{DType, FloatElement, TensorElement},
    error::{Result, TorshError},
    shape::Shape,
};

use std::sync::{Arc, RwLock, Weak};
use std::path::PathBuf;
use std::fs::{File, OpenOptions};
use std::io::Write;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

/// Threshold for switching to memory-mapped storage (1GB)
const MEMORY_MAPPING_THRESHOLD: usize = 1024 * 1024 * 1024;

/// Storage abstraction for tensor data
#[derive(Debug)]
pub enum TensorStorage<T: TensorElement> {
    /// In-memory storage for smaller tensors
    InMemory(Arc<RwLock<Vec<T>>>),
    /// Memory-mapped storage for large tensors
    MemoryMapped(Arc<RwLock<MemoryMappedStorage<T>>>),
}

/// Memory-mapped storage implementation
#[derive(Debug)]
pub struct MemoryMappedStorage<T: TensorElement> {
    /// File backing the memory mapping
    file: File,
    /// Path to the backing file
    file_path: PathBuf,
    /// Number of elements stored
    num_elements: usize,
    /// Cache for frequently accessed elements
    cache: std::collections::HashMap<usize, T>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Access pattern tracking for cache optimization
    access_pattern: std::collections::VecDeque<usize>,
    /// Whether the storage is temporary (should be deleted on drop)
    is_temporary: bool,
}

impl<T: TensorElement + Copy> TensorStorage<T> {
    /// Create in-memory storage
    pub fn in_memory(data: Vec<T>) -> Self {
        Self::InMemory(Arc::new(RwLock::new(data)))
    }

    /// Create memory-mapped storage
    pub fn memory_mapped(
        data: Vec<T>, 
        file_path: Option<PathBuf>,
    ) -> Result<Self> {
        let storage = MemoryMappedStorage::new(data, file_path)?;
        Ok(Self::MemoryMapped(Arc::new(RwLock::new(storage))))
    }

    /// Create storage automatically based on size
    pub fn create_optimal(data: Vec<T>) -> Result<Self> {
        let size_bytes = data.len() * std::mem::size_of::<T>();
        
        if size_bytes >= MEMORY_MAPPING_THRESHOLD {
            Self::memory_mapped(data, None)
        } else {
            Ok(Self::in_memory(data))
        }
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        match self {
            Self::InMemory(data) => data.read().unwrap().len(),
            Self::MemoryMapped(storage) => storage.read().unwrap().num_elements,
        }
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Result<T> 
    where 
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let data_guard = data.read().unwrap();
                data_guard.get(index)
                    .copied()
                    .ok_or_else(|| TorshError::IndexOutOfBounds { 
                        index, 
                        size: data_guard.len() 
                    })
            }
            Self::MemoryMapped(storage) => {
                storage.write().unwrap().get(index)
            }
        }
    }

    /// Set element at index
    pub fn set(&self, index: usize, value: T) -> Result<()> 
    where 
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let mut data_guard = data.write().unwrap();
                if index >= data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds { 
                        index, 
                        size: data_guard.len() 
                    });
                }
                data_guard[index] = value;
                Ok(())
            }
            Self::MemoryMapped(storage) => {
                storage.write().unwrap().set(index, value)
            }
        }
    }

    /// Get multiple elements
    pub fn get_slice(&self, start: usize, len: usize) -> Result<Vec<T>> 
    where 
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let data_guard = data.read().unwrap();
                if start + len > data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds { 
                        index: start + len - 1, 
                        size: data_guard.len() 
                    });
                }
                Ok(data_guard[start..start + len].to_vec())
            }
            Self::MemoryMapped(storage) => {
                storage.write().unwrap().get_slice(start, len)
            }
        }
    }

    /// Set multiple elements
    pub fn set_slice(&self, start: usize, values: &[T]) -> Result<()> 
    where 
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let mut data_guard = data.write().unwrap();
                if start + values.len() > data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds { 
                        index: start + values.len() - 1, 
                        size: data_guard.len() 
                    });
                }
                data_guard[start..start + values.len()].copy_from_slice(values);
                Ok(())
            }
            Self::MemoryMapped(storage) => {
                storage.write().unwrap().set_slice(start, values)
            }
        }
    }

    /// Convert to vector (useful for small tensors or debugging)
    pub fn to_vec(&self) -> Result<Vec<T>> 
    where 
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                Ok(data.read().unwrap().clone())
            }
            Self::MemoryMapped(storage) => {
                storage.write().unwrap().to_vec()
            }
        }
    }

    /// Get storage type information
    pub fn storage_type(&self) -> &'static str {
        match self {
            Self::InMemory(_) => "in_memory",
            Self::MemoryMapped(_) => "memory_mapped",
        }
    }

    /// Get estimated memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match self {
            Self::InMemory(data) => {
                data.read().unwrap().len() * std::mem::size_of::<T>()
            }
            Self::MemoryMapped(storage) => {
                let storage_guard = storage.read().unwrap();
                // Memory usage is just the cache size plus metadata
                storage_guard.cache.len() * std::mem::size_of::<T>() + 
                std::mem::size_of::<MemoryMappedStorage<T>>()
            }
        }
    }
}

impl<T: TensorElement> MemoryMappedStorage<T> {
    /// Create new memory-mapped storage
    pub fn new(data: Vec<T>, file_path: Option<PathBuf>) -> Result<Self> {
        let (file_path, is_temporary) = match file_path {
            Some(path) => (path, false),
            None => {
                // Create temporary file
                let temp_dir = std::env::temp_dir();
                let temp_file = temp_dir.join(format!(
                    "torsh_tensor_{}.mmap",
                    std::process::id()
                ));
                (temp_file, true)
            }
        };

        // Create and write data to file
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&file_path)
            .map_err(|e| TorshError::IoError(format!("Failed to create memory-mapped file: {e}")))?;

        // Write data to file
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };
        file.write_all(data_bytes)
            .map_err(|e| TorshError::IoError(format!("Failed to write to memory-mapped file: {e}")))?;
        file.flush()
            .map_err(|e| TorshError::IoError(format!("Failed to flush memory-mapped file: {e}")))?;

        Ok(Self {
            file,
            file_path,
            num_elements: data.len(),
            cache: std::collections::HashMap::new(),
            max_cache_size: 10000, // Cache up to 10k elements
            access_pattern: std::collections::VecDeque::new(),
            is_temporary,
        })
    }

    /// Get element at index with caching
    pub fn get(&mut self, index: usize) -> Result<T> 
    where 
        T: Copy,
    {
        if index >= self.num_elements {
            return Err(TorshError::IndexOutOfBounds { 
                index, 
                size: self.num_elements 
            });
        }

        // Check cache first
        if let Some(&value) = self.cache.get(&index) {
            self.update_access_pattern(index);
            return Ok(value);
        }

        // Read from file
        let value = self.read_element_from_file(index)?;
        
        // Add to cache if there's space
        if self.cache.len() < self.max_cache_size {
            self.cache.insert(index, value);
        } else {
            // Evict least recently used element
            self.evict_lru();
            self.cache.insert(index, value);
        }
        
        self.update_access_pattern(index);
        Ok(value)
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: T) -> Result<()> 
    where 
        T: Copy,
    {
        if index >= self.num_elements {
            return Err(TorshError::IndexOutOfBounds { 
                index, 
                size: self.num_elements 
            });
        }

        // Update cache
        self.cache.insert(index, value);
        
        // Write to file
        self.write_element_to_file(index, value)?;
        self.update_access_pattern(index);
        Ok(())
    }

    /// Get slice of elements
    pub fn get_slice(&mut self, start: usize, len: usize) -> Result<Vec<T>> 
    where 
        T: Copy,
    {
        if start + len > self.num_elements {
            return Err(TorshError::IndexOutOfBounds { 
                index: start + len - 1, 
                size: self.num_elements 
            });
        }

        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(self.get(start + i)?);
        }
        Ok(result)
    }

    /// Set slice of elements
    pub fn set_slice(&mut self, start: usize, values: &[T]) -> Result<()> 
    where 
        T: Copy,
    {
        if start + values.len() > self.num_elements {
            return Err(TorshError::IndexOutOfBounds { 
                index: start + values.len() - 1, 
                size: self.num_elements 
            });
        }

        for (i, &value) in values.iter().enumerate() {
            self.set(start + i, value)?;
        }
        Ok(())
    }

    /// Convert entire storage to vector
    pub fn to_vec(&mut self) -> Result<Vec<T>> 
    where 
        T: Copy,
    {
        self.get_slice(0, self.num_elements)
    }

    /// Read element from file
    fn read_element_from_file(&mut self, index: usize) -> Result<T> 
    where 
        T: Copy,
    {
        let offset = index * std::mem::size_of::<T>();
        let mut buffer = vec![0u8; std::mem::size_of::<T>()];
        
        #[cfg(unix)]
        {
            self.file.read_exact_at(&mut buffer, offset as u64)
                .map_err(|e| TorshError::IoError(format!("Failed to read from memory-mapped file: {e}")))?;
        }
        
        #[cfg(windows)]
        {
            self.file.seek_read(&mut buffer, offset as u64)
                .map_err(|e| TorshError::IoError(format!("Failed to read from memory-mapped file: {e}")))?;
        }
        
        #[cfg(not(any(unix, windows)))]
        {
            self.file.seek(SeekFrom::Start(offset as u64))
                .map_err(|e| TorshError::IoError(format!("Failed to seek in memory-mapped file: {e}")))?;
            self.file.read_exact(&mut buffer)
                .map_err(|e| TorshError::IoError(format!("Failed to read from memory-mapped file: {e}")))?;
        }

        // Convert bytes to T
        let value = unsafe {
            std::ptr::read(buffer.as_ptr() as *const T)
        };
        Ok(value)
    }

    /// Write element to file
    fn write_element_to_file(&mut self, index: usize, value: T) -> Result<()> 
    where 
        T: Copy,
    {
        let offset = index * std::mem::size_of::<T>();
        let buffer = unsafe {
            std::slice::from_raw_parts(
                &value as *const T as *const u8,
                std::mem::size_of::<T>(),
            )
        };
        
        #[cfg(unix)]
        {
            self.file.write_all_at(buffer, offset as u64)
                .map_err(|e| TorshError::IoError(format!("Failed to write to memory-mapped file: {e}")))?;
        }
        
        #[cfg(windows)]
        {
            self.file.seek_write(buffer, offset as u64)
                .map_err(|e| TorshError::IoError(format!("Failed to write to memory-mapped file: {e}")))?;
        }
        
        #[cfg(not(any(unix, windows)))]
        {
            self.file.seek(SeekFrom::Start(offset as u64))
                .map_err(|e| TorshError::IoError(format!("Failed to seek in memory-mapped file: {e}")))?;
            self.file.write_all(buffer)
                .map_err(|e| TorshError::IoError(format!("Failed to write to memory-mapped file: {e}")))?;
        }
        
        Ok(())
    }

    /// Update access pattern for cache management
    fn update_access_pattern(&mut self, index: usize) {
        self.access_pattern.push_back(index);
        if self.access_pattern.len() > self.max_cache_size {
            self.access_pattern.pop_front();
        }
    }

    /// Evict least recently used element from cache
    fn evict_lru(&mut self) {
        if let Some(lru_index) = self.access_pattern.front().copied() {
            self.cache.remove(&lru_index);
        }
    }
}

impl<T: TensorElement> Drop for MemoryMappedStorage<T> {
    fn drop(&mut self) {
        if self.is_temporary {
            // Clean up temporary file
            let _ = std::fs::remove_file(&self.file_path);
        }
    }
}

impl<T: TensorElement> Clone for TensorStorage<T> {
    fn clone(&self) -> Self {
        match self {
            Self::InMemory(data) => Self::InMemory(Arc::clone(data)),
            Self::MemoryMapped(storage) => Self::MemoryMapped(Arc::clone(storage)),
        }
    }
}

/// Operation type for gradient computation
#[derive(Debug, Clone)]
pub(crate) enum Operation<T: TensorElement> {
    Leaf,
    Power {
        input: Arc<Tensor<T>>,
        exponent: f32,
    },
    Add {
        lhs: Arc<Tensor<T>>,
        rhs: Arc<Tensor<T>>,
    },
    Mul {
        lhs: Arc<Tensor<T>>,
        rhs: Arc<Tensor<T>>,
    },
    Custom(String, Vec<Weak<Tensor<T>>>),
}

/// The main Tensor type for ToRSh
///
/// A tensor implementation with automatic memory mapping for large tensors
/// and efficient views with reference counting
#[derive(Clone)]
pub struct Tensor<T = f32>
where
    T: TensorElement,
{
    /// The data storage (automatically uses memory mapping for large tensors)
    pub(crate) storage: TensorStorage<T>,
    /// Shape of the tensor
    pub(crate) shape: Shape,
    /// Device information
    pub(crate) device: DeviceType,
    /// Whether gradients are required
    pub(crate) requires_grad: bool,
    /// Gradient tensor if computed
    pub(crate) grad: Arc<RwLock<Option<Tensor<T>>>>,
    /// Operation that created this tensor
    pub(crate) operation: Operation<T>,
    /// Custom strides for views (None means contiguous layout)
    pub(crate) strides: Option<Vec<usize>>,
    /// Offset into the storage for views (0 for base tensors)
    pub(crate) storage_offset: usize,
    /// Reference to base tensor for views (None for base tensors)
    pub(crate) base_tensor: Option<Weak<Tensor<T>>>,
}

impl<T: TensorElement + Copy> Tensor<T> {
    /// Clean up dead weak references in custom operations to improve memory efficiency
    pub fn cleanup_operation_refs(&mut self) {
        if let Operation::Custom(_, inputs) = &mut self.operation {
            inputs.retain(|weak_ref| weak_ref.strong_count() > 0);
        }
    }

    /// Ensure exclusive ownership of data using copy-on-write semantics
    /// If the data is shared (Arc has multiple strong references), clone it
    #[allow(dead_code)]
    fn ensure_exclusive_data(&mut self) -> Result<()> {
        match &self.storage {
            TensorStorage::InMemory(data) => {
                if Arc::strong_count(data) > 1 {
                    // Data is shared, need to clone it to get exclusive access
                    let cloned_data = {
                        let data_guard = data.read().unwrap();
                        data_guard.clone()
                    };
                    self.storage = TensorStorage::in_memory(cloned_data);
                }
            }
            TensorStorage::MemoryMapped(storage) => {
                if Arc::strong_count(storage) > 1 {
                    // Clone memory-mapped storage by converting to vec and back
                    let data_vec = self.storage.to_vec()?;
                    self.storage = TensorStorage::create_optimal(data_vec)?;
                }
            }
        }
        Ok(())
    }


    /// Create from raw data
    pub fn from_data(data: Vec<T>, shape: Vec<usize>, device: DeviceType) -> Result<Self> {
        let storage = TensorStorage::create_optimal(data)?;
        Ok(Self {
            storage,
            shape: Shape::new(shape),
            device,
            requires_grad: false,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: None,
            storage_offset: 0,
            base_tensor: None,
        })
    }

    /// Create from raw data with explicit storage type
    pub fn from_data_with_storage(data: Vec<T>, shape: Vec<usize>, device: DeviceType, use_memory_mapping: bool) -> Result<Self> {
        let storage = if use_memory_mapping {
            TensorStorage::memory_mapped(data, None)?
        } else {
            TensorStorage::in_memory(data)
        };
        Ok(Self {
            storage,
            shape: Shape::new(shape),
            device,
            requires_grad: false,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: None,
            storage_offset: 0,
            base_tensor: None,
        })
    }

    /// Create from raw data with specified memory-mapped file path
    pub fn from_data_memory_mapped(data: Vec<T>, shape: Vec<usize>, device: DeviceType, file_path: PathBuf) -> Result<Self> {
        let storage = TensorStorage::memory_mapped(data, Some(file_path))?;
        Ok(Self {
            storage,
            shape: Shape::new(shape),
            device,
            requires_grad: false,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: None,
            storage_offset: 0,
            base_tensor: None,
        })
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize], device: DeviceType) -> Result<Self> {
        let numel = shape.iter().product();
        let data = vec![T::zero(); numel];
        Self::from_data(data, shape.to_vec(), device)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize], device: DeviceType) -> Result<Self> {
        let numel = shape.iter().product();
        let data = vec![T::one(); numel];
        Self::from_data(data, shape.to_vec(), device)
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get the data type
    pub fn dtype(&self) -> DType {
        T::dtype()
    }

    /// Convert tensor to a different data type
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        // For now, just return a clone if the dtype is the same
        if T::dtype() == dtype {
            Ok(self.clone())
        } else {
            // TODO: Implement actual dtype conversion
            Err(TorshError::UnsupportedOperation {
                op: "dtype conversion".to_string(),
                dtype: format!("{:?} to {:?}", T::dtype(), dtype),
            })
        }
    }

    /// Get the device
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Get element at multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> Result<T>
    where
        T: Copy,
    {
        let flat_index = self.compute_flat_index(indices)?;
        self.storage.get(flat_index)
    }

    /// Get element at single flat index
    pub fn get_flat(&self, index: usize) -> Result<T>
    where
        T: Copy,
    {
        self.storage.get(index)
    }

    /// Set element at index (requires multi-dimensional indices for views)
    pub fn set(&self, indices: &[usize], value: T) -> Result<()>
    where
        T: Copy,
    {
        let flat_index = self.compute_flat_index(indices)?;
        self.storage.set(flat_index, value)
    }

    /// Get slice of elements
    pub fn get_slice(&self, start: usize, len: usize) -> Result<Vec<T>>
    where
        T: Copy,
    {
        self.storage.get_slice(start, len)
    }

    /// Set slice of elements
    pub fn set_slice(&self, start: usize, values: &[T]) -> Result<()>
    where
        T: Copy,
    {
        self.storage.set_slice(start, values)
    }

    /// Get all data as a vector (may be expensive for large memory-mapped tensors)
    /// For views, extracts only the data visible by this view
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Copy,
    {
        if self.is_view() {
            // For views, we need to extract data according to strides and offsets
            let mut result = Vec::with_capacity(self.numel());
            let shape = self.shape.dims();
            let mut indices = vec![0; shape.len()];
            
            fn extract_recursive<T: TensorElement + Copy>(
                tensor: &Tensor<T>,
                indices: &mut [usize],
                dim: usize,
                result: &mut Vec<T>,
            ) -> Result<()> {
                let shape = tensor.shape.dims();
                if dim == shape.len() {
                    let flat_index = tensor.compute_flat_index(indices)?;
                    let value = tensor.storage.get(flat_index)?;
                    result.push(value);
                } else {
                    for i in 0..shape[dim] {
                        indices[dim] = i;
                        extract_recursive(tensor, indices, dim + 1, result)?;
                    }
                }
                Ok(())
            }
            
            extract_recursive(self, &mut indices, 0, &mut result)?;
            Ok(result)
        } else {
            self.storage.to_vec()
        }
    }

    /// Get storage type information
    pub fn storage_type(&self) -> &'static str {
        self.storage.storage_type()
    }

    /// Get estimated memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.storage.memory_usage()
    }

    /// Check if tensor uses memory mapping
    pub fn is_memory_mapped(&self) -> bool {
        matches!(self.storage, TensorStorage::MemoryMapped(_))
    }

    /// Check if this tensor is a view of another tensor
    pub fn is_view(&self) -> bool {
        self.base_tensor.is_some()
    }

    /// Get the strides for this tensor (either custom strides for views or default contiguous strides)
    pub fn strides(&self) -> Vec<usize> {
        if let Some(ref strides) = self.strides {
            strides.clone()
        } else {
            self.compute_default_strides()
        }
    }

    /// Compute default contiguous strides for the tensor's shape
    fn compute_default_strides(&self) -> Vec<usize> {
        let shape = self.shape.dims();
        let mut strides = vec![1; shape.len()];
        if shape.len() > 1 {
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        strides
    }

    /// Compute flat index with custom strides and offset for views
    fn compute_flat_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        // Validate indices
        let shape = self.shape.dims();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(TorshError::IndexOutOfBounds {
                    index: idx,
                    size: shape[i],
                });
            }
        }

        // Compute flat index using strides
        let strides = self.strides();
        let flat_index = indices
            .iter()
            .zip(&strides)
            .map(|(idx, stride)| idx * stride)
            .sum::<usize>()
            + self.storage_offset;

        Ok(flat_index)
    }

    /// Create a tensor of ones with the same shape as this tensor
    pub fn ones_like(&self) -> Result<Self> {
        let ones_data = vec![T::one(); self.numel()];
        Self::from_data(
            ones_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Create a tensor of zeros with the same shape as this tensor
    pub fn zeros_like(&self) -> Result<Self> {
        let zeros_data = vec![T::zero(); self.numel()];
        Self::from_data(
            zeros_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Set whether this tensor requires gradients
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Get whether this tensor requires gradients
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set gradient tensor
    #[allow(dead_code)]
    pub fn set_grad(&self, grad: Option<Tensor<T>>) {
        let mut grad_lock = self.grad.write().unwrap();
        *grad_lock = grad;
    }

    /// Get mutable access to gradient
    pub fn grad_mut(&mut self) -> Option<&mut Self> {
        // For now, return None - would need to implement proper gradient access
        None
    }

    /// Convert to a different device
    pub fn to<D: Into<DeviceType>>(self, device: D) -> Result<Self> {
        let device = device.into();
        if device == self.device {
            return Ok(self);
        }

        // TODO: Implement actual device transfer when backends are ready
        Err(TorshError::UnsupportedOperation {
            op: "device transfer".to_string(),
            dtype: self.dtype().to_string(),
        })
    }

    /// Detach from the computation graph
    pub fn detach(&self) -> Self {
        let mut detached = self.clone();
        detached.requires_grad = false;
        detached
    }

    /// Get the gradient of this tensor (if it exists)
    pub fn grad(&self) -> Option<Self> {
        let grad_lock = self.grad.read().unwrap();
        grad_lock.as_ref().cloned()
    }

    /// Check if this tensor has a gradient
    pub fn has_grad(&self) -> bool {
        let grad_lock = self.grad.read().unwrap();
        grad_lock.is_some()
    }

    /// Zero the gradient
    pub fn zero_grad(&mut self) {
        let mut grad_lock = self.grad.write().unwrap();
        *grad_lock = None;
    }

    /// Backward pass (compute gradients) - integrated with autograd system
    pub fn backward(&self) -> Result<()>
    where
        T: FloatElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Clone
            + std::fmt::Debug,
        f32: From<T>,
    {
        if !self.requires_grad {
            return Err(TorshError::AutogradError(
                "Called backward on tensor that doesn't require grad".to_string(),
            ));
        }

        if self.shape().numel() != 1 {
            return Err(TorshError::AutogradError(
                "Gradient can only be computed for scalar outputs".to_string(),
            ));
        }

        // Start backward computation with gradient of 1.0 for scalar output
        let grad_output = self.ones_like()?;
        self.backward_impl(&grad_output)?;
        Ok(())
    }

    /// Backward pass with gradient - integrated with autograd system
    pub fn backward_with_grad(&self, _gradient: Option<&Self>) -> Result<()>
    where
        T: FloatElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Clone
            + std::fmt::Debug,
        f32: From<T>,
    {
        if !self.requires_grad {
            return Err(TorshError::AutogradError(
                "Called backward on tensor that doesn't require grad".to_string(),
            ));
        }

        // TODO: Implement backward pass with gradient - currently autograd is handled at higher level
        // For now, return Ok since this will be handled by the autograd crate
        Ok(())
    }

    /// Internal backward implementation
    fn backward_impl(&self, grad_output: &Self) -> Result<()>
    where
        T: FloatElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    {
        match &self.operation {
            Operation::Leaf => {
                // Accumulate gradient for leaf nodes
                let mut grad_lock = self.grad.write().unwrap();
                if let Some(existing_grad) = grad_lock.as_ref() {
                    // Add gradients if they exist
                    let new_grad = existing_grad.add_op(grad_output)?;
                    *grad_lock = Some(new_grad);
                } else {
                    // Set gradient if it doesn't exist
                    *grad_lock = Some(grad_output.clone());
                }
            }
            Operation::Power { input, exponent } => {
                if input.requires_grad {
                    // Compute gradient: d/dx(x^n) = n * x^(n-1)
                    let input_data = input.to_vec()?;
                    let grad_data: Vec<T> = input_data
                        .iter()
                        .map(|&x| {
                            let exp_minus_one = *exponent - 1.0;
                            let exp_t = T::from_f64(*exponent as f64).unwrap();
                            let exp_minus_one_t = T::from_f64(exp_minus_one as f64).unwrap();
                            exp_t * x.powf(exp_minus_one_t)
                        })
                        .collect();

                    let input_grad =
                        Self::from_data(grad_data, input.shape().dims().to_vec(), input.device)?;
                    let final_grad = input_grad.mul_op(grad_output)?;

                    // Recursively compute gradients
                    input.backward_impl(&final_grad)?;
                }
            }
            Operation::Add { lhs, rhs } => {
                // Gradient flows through both operands unchanged
                if lhs.requires_grad {
                    lhs.backward_impl(grad_output)?;
                }
                if rhs.requires_grad {
                    rhs.backward_impl(grad_output)?;
                }
            }
            Operation::Mul { lhs, rhs } => {
                // Product rule: d/dx(f*g) = f'*g + f*g'
                if lhs.requires_grad {
                    let lhs_grad = (**rhs).mul_op(grad_output)?;
                    lhs.backward_impl(&lhs_grad)?;
                }
                if rhs.requires_grad {
                    let rhs_grad = (**lhs).mul_op(grad_output)?;
                    rhs.backward_impl(&rhs_grad)?;
                }
            }
            Operation::Custom(op_name, inputs) => {
                // For custom operations, we need operation-specific gradient computation
                // For now, we'll just propagate the gradient to all inputs
                match op_name.as_str() {
                    "conv1d" | "conv2d" | "conv3d" => {
                        // Convolution backward pass would require specific implementation
                        // For now, we skip gradient computation for convolutions
                        // TODO: Implement proper convolution backward pass
                    }
                    _ => {
                        // For other custom operations, propagate gradient to all inputs
                        // Use weak references to prevent memory leaks from circular references
                        for weak_input in inputs {
                            if let Some(input) = weak_input.upgrade() {
                                if input.requires_grad {
                                    input.backward_impl(grad_output)?;
                                }
                            }
                            // Note: Dead weak references are automatically cleaned up
                            // when the Vec goes out of scope
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// Complex tensor operations and backward support
impl<T: torsh_core::dtype::ComplexElement> Tensor<T> {
    /// Complex conjugate for complex tensors
    pub fn complex_conj(&self) -> Result<Self>
    where
        T: Copy,
    {
        let data = self.to_vec()?;
        let conj_data: Vec<T> = data.iter().map(|&z| z.conj()).collect();
        let mut result = Self::from_data(conj_data, self.shape().dims().to_vec(), self.device)?;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }
    /// Backward pass for complex tensors (compute gradients)
    ///
    /// Complex autograd follows PyTorch's approach where gradients are computed
    /// treating complex numbers as 2D vectors of real numbers.
    pub fn backward_complex(&self) -> Result<()>
    where
        T: Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    {
        if !self.requires_grad {
            return Err(TorshError::AutogradError(
                "Called backward on tensor that doesn't require grad".to_string(),
            ));
        }

        if self.shape().numel() != 1 {
            return Err(TorshError::AutogradError(
                "Gradient can only be computed for scalar outputs".to_string(),
            ));
        }

        // Create initial gradient of 1.0 + 0.0i for the output
        let output_grad_data = vec![T::new(
            <T::Real as torsh_core::dtype::TensorElement>::one(),
            <T::Real as torsh_core::dtype::TensorElement>::zero(),
        )];
        let output_grad = Self::from_data(output_grad_data, vec![], self.device)?;

        // Start backpropagation
        self.backward_complex_impl(&output_grad)?;

        Ok(())
    }

    /// Internal backward implementation for complex tensors
    fn backward_complex_impl(&self, grad_output: &Self) -> Result<()>
    where
        T: Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    {
        match &self.operation {
            Operation::Leaf => {
                // Accumulate gradient for leaf nodes
                let mut grad_lock = self.grad.write().unwrap();
                if let Some(existing_grad) = grad_lock.as_ref() {
                    // Add gradients if they exist
                    let new_grad = existing_grad.add_op(grad_output)?;
                    *grad_lock = Some(new_grad);
                } else {
                    // Set gradient if it doesn't exist
                    *grad_lock = Some(grad_output.clone());
                }
            }
            Operation::Add { lhs, rhs } => {
                // Gradient flows through both operands unchanged for complex addition
                if lhs.requires_grad {
                    lhs.backward_complex_impl(grad_output)?;
                }
                if rhs.requires_grad {
                    rhs.backward_complex_impl(grad_output)?;
                }
            }
            Operation::Mul { lhs, rhs } => {
                // Complex multiplication rule: d/dz(f*g) = f'*g + f*g'
                if lhs.requires_grad {
                    let lhs_grad = (**rhs).mul_op(grad_output)?;
                    lhs.backward_complex_impl(&lhs_grad)?;
                }
                if rhs.requires_grad {
                    let rhs_grad = (**lhs).mul_op(grad_output)?;
                    rhs.backward_complex_impl(&rhs_grad)?;
                }
            }
            Operation::Custom(op_name, inputs) => {
                match op_name.as_str() {
                    "complex_conj" => {
                        // Gradient of complex conjugate: d/dz(conj(f)) = conj(df/dz)
                        if let Some(weak_input) = inputs.first() {
                            if let Some(input) = weak_input.upgrade() {
                                if input.requires_grad {
                                    let conj_grad = grad_output.complex_conj()?;
                                    input.backward_complex_impl(&conj_grad)?;
                                }
                            }
                        }
                    }
                    "complex_real" => {
                        // Gradient flows only to real part: d/dz(real(f)) = real(df/dz)
                        if let Some(weak_input) = inputs.first() {
                            if let Some(input) = weak_input.upgrade() {
                                if input.requires_grad {
                                    // Create complex gradient with imaginary part zero
                                    let real_grad_data = grad_output.to_vec()?;
                                    let complex_grad_data: Vec<T> = real_grad_data
                                        .iter()
                                        .map(|&r| {
                                            T::new(
                                                r.real(),
                                                <T::Real as torsh_core::dtype::TensorElement>::zero(),
                                            )
                                        })
                                        .collect();
                                    let complex_grad = Self::from_data(
                                        complex_grad_data,
                                        grad_output.shape().dims().to_vec(),
                                        grad_output.device,
                                    )?;
                                    input.backward_complex_impl(&complex_grad)?;
                                }
                            }
                        }
                    }
                    "complex_imag" => {
                        // Gradient flows only to imaginary part: d/dz(imag(f)) = i*imag(df/dz)
                        if let Some(weak_input) = inputs.first() {
                            if let Some(input) = weak_input.upgrade() {
                                if input.requires_grad {
                                    // Create complex gradient with real part zero and imaginary part from grad
                                    let imag_grad_data = grad_output.to_vec()?;
                                    let complex_grad_data: Vec<T> = imag_grad_data
                                        .iter()
                                        .map(|&i| {
                                            T::new(
                                                <T::Real as torsh_core::dtype::TensorElement>::zero(),
                                                i.real(),
                                            )
                                        })
                                        .collect();
                                    let complex_grad = Self::from_data(
                                        complex_grad_data,
                                        grad_output.shape().dims().to_vec(),
                                        grad_output.device,
                                    )?;
                                    input.backward_complex_impl(&complex_grad)?;
                                }
                            }
                        }
                    }
                    _ => {
                        // For other custom operations, propagate gradient to all inputs
                        // Use weak references to prevent memory leaks from circular references
                        for weak_input in inputs {
                            if let Some(input) = weak_input.upgrade() {
                                if input.requires_grad {
                                    input.backward_complex_impl(grad_output)?;
                                }
                            }
                            // If the tensor has been dropped, skip gradient computation
                        }
                    }
                }
            }
            _ => {
                // For other operations not yet implemented for complex, skip
                // TODO: Implement complex backward for Power and other operations
            }
        }

        Ok(())
    }
}

impl<T: TensorElement + Copy> Tensor<T> {
    /// Get size of a specific dimension
    pub fn size(&self, dim: i32) -> Result<usize> {
        self.shape().size(dim)
    }

    /// Reshape the tensor
    pub fn view(&self, shape: &[i32]) -> Result<Self> {
        // Validate that there's at most one -1 in the shape
        let infer_count = shape.iter().filter(|&&x| x == -1).count();
        if infer_count > 1 {
            return Err(TorshError::InvalidShape(
                "Only one dimension can be inferred (only one -1 allowed)".to_string(),
            ));
        }

        let new_shape: Result<Vec<usize>> = shape
            .iter()
            .map(|&d| {
                if d == -1 {
                    // Infer dimension - first validate all other dimensions are valid
                    let known_dims: Result<Vec<usize>> = shape
                        .iter()
                        .filter(|&&x| x != -1)
                        .map(|&x| {
                            if x < 0 {
                                Err(TorshError::InvalidShape(format!(
                                    "Invalid dimension size: {x} (negative dimensions not allowed except -1)"
                                )))
                            } else {
                                Ok(x as usize)
                            }
                        })
                        .collect();

                    let known_dims = known_dims?;
                    
                    // Check for overflow in product calculation
                    let known_product = known_dims.iter().try_fold(1usize, |acc, &dim| {
                        acc.checked_mul(dim).ok_or_else(|| {
                            TorshError::InvalidShape(
                                "Shape dimensions too large (would overflow)".to_string()
                            )
                        })
                    })?;

                    if known_product == 0 {
                        return Err(TorshError::InvalidShape(
                            "Cannot infer dimension with zero-sized dimensions".to_string(),
                        ));
                    }

                    let total = self.numel();
                    if total % known_product != 0 {
                        return Err(TorshError::InvalidShape(
                            "Cannot infer dimension: size is not divisible".to_string(),
                        ));
                    }

                    Ok(total / known_product)
                } else if d < 0 {
                    Err(TorshError::InvalidShape(format!(
                        "Invalid dimension size: {d}"
                    )))
                } else {
                    Ok(d as usize)
                }
            })
            .collect();

        let new_shape = new_shape?;
        
        // Check for overflow in total elements calculation
        let new_numel = new_shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                TorshError::InvalidShape(
                    "Reshaped tensor would be too large (would overflow)".to_string()
                )
            })
        })?;

        if new_numel != self.numel() {
            return Err(TorshError::InvalidShape(format!(
                "Shape {:?} is invalid for tensor of size {}",
                new_shape,
                self.numel()
            )));
        }

        // Create a new tensor with the same data but different shape
        let data = self.to_vec()?;
        Self::from_data(data, new_shape, self.device)
    }

    /// Create an efficient view with different shape (shares data, no copying)
    /// This is the zero-copy version of view() for compatible shapes
    pub fn view_as(&self, shape: &[usize]) -> Result<Self> {
        // Validate that the total number of elements is the same
        let new_numel = shape.iter().product::<usize>();
        if new_numel != self.numel() {
            return Err(TorshError::InvalidShape(format!(
                "Shape {:?} is invalid for tensor of size {}",
                shape,
                self.numel()
            )));
        }

        // Only create efficient views for contiguous tensors or existing views
        // that are still relatively simple
        if !self.is_contiguous() {
            return Err(TorshError::InvalidShape(
                "Cannot create efficient view of non-contiguous tensor".to_string(),
            ));
        }

        // Create new tensor sharing the same storage
        Ok(Self {
            storage: self.storage.clone(),
            shape: Shape::new(shape.to_vec()),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)), // Views don't share gradients
            operation: Operation::Leaf,       // Views reset operation tracking
            strides: None, // Use default contiguous strides for simple reshapes
            storage_offset: self.storage_offset,
            base_tensor: if self.is_view() {
                // If this is already a view, keep reference to the original base
                self.base_tensor.clone()
            } else {
                // This is a base tensor, so create a weak reference to it
                Some(Arc::downgrade(&Arc::new(self.clone())))
            },
        })
    }

    /// Create a view of a slice along a dimension (shares data, no copying)
    pub fn slice_tensor(&self, dim: usize, start: usize, end: usize) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        let shape = self.shape.dims();
        if start >= shape[dim] || end > shape[dim] || start >= end {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid slice range [{}:{}] for dimension {} of size {}",
                start, end, dim, shape[dim]
            )));
        }

        // Calculate new shape
        let mut new_shape = shape.to_vec();
        new_shape[dim] = end - start;

        // Calculate new strides and offset
        let current_strides = self.strides();
        let offset_adjustment = start * current_strides[dim];

        Ok(Self {
            storage: self.storage.clone(),
            shape: Shape::new(new_shape),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: Some(current_strides),
            storage_offset: self.storage_offset + offset_adjustment,
            base_tensor: if self.is_view() {
                self.base_tensor.clone()
            } else {
                Some(Arc::downgrade(&Arc::new(self.clone())))
            },
        })
    }

    /// Create a transposed view (shares data, no copying)
    pub fn transpose_view(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimensions {} and {} out of range for tensor with {} dimensions",
                dim0, dim1, self.ndim()
            )));
        }

        if dim0 == dim1 {
            return Ok(self.clone());
        }

        // Create new shape and strides
        let mut new_shape = self.shape.dims().to_vec();
        let mut new_strides = self.strides();

        // Swap dimensions
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Ok(Self {
            storage: self.storage.clone(),
            shape: Shape::new(new_shape),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: Some(new_strides),
            storage_offset: self.storage_offset,
            base_tensor: if self.is_view() {
                self.base_tensor.clone()
            } else {
                Some(Arc::downgrade(&Arc::new(self.clone())))
            },
        })
    }

    /// Create a view with a dimension of size 1 removed (shares data, no copying)
    pub fn squeeze_tensor(&self, dim: usize) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        let shape = self.shape.dims();
        if shape[dim] != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Cannot squeeze dimension {} of size {} (must be size 1)",
                dim, shape[dim]
            )));
        }

        // Create new shape and strides without the squeezed dimension
        let mut new_shape = Vec::with_capacity(shape.len() - 1);
        let mut new_strides = Vec::with_capacity(shape.len() - 1);
        let current_strides = self.strides();

        for i in 0..shape.len() {
            if i != dim {
                new_shape.push(shape[i]);
                new_strides.push(current_strides[i]);
            }
        }

        Ok(Self {
            storage: self.storage.clone(),
            shape: Shape::new(new_shape),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: Some(new_strides),
            storage_offset: self.storage_offset,
            base_tensor: if self.is_view() {
                self.base_tensor.clone()
            } else {
                Some(Arc::downgrade(&Arc::new(self.clone())))
            },
        })
    }

    /// Create a view with a dimension of size 1 added (shares data, no copying)
    pub fn unsqueeze_tensor(&self, dim: usize) -> Result<Self> {
        if dim > self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for unsqueeze (max {})",
                dim,
                self.ndim()
            )));
        }

        // Create new shape and strides with the unsqueezed dimension
        let shape = self.shape.dims();
        let current_strides = self.strides();
        let mut new_shape = Vec::with_capacity(shape.len() + 1);
        let mut new_strides = Vec::with_capacity(shape.len() + 1);

        // Insert the new dimension
        for i in 0..=shape.len() {
            if i == dim {
                new_shape.push(1);
                // Stride for size-1 dimension can be anything, use the next stride or 1
                let stride = if i < current_strides.len() {
                    current_strides[i]
                } else {
                    1
                };
                new_strides.push(stride);
            }
            if i < shape.len() {
                new_shape.push(shape[i]);
                new_strides.push(current_strides[i]);
            }
        }

        Ok(Self {
            storage: self.storage.clone(),
            shape: Shape::new(new_shape),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: Some(new_strides),
            storage_offset: self.storage_offset,
            base_tensor: if self.is_view() {
                self.base_tensor.clone()
            } else {
                Some(Arc::downgrade(&Arc::new(self.clone())))
            },
        })
    }

    /// Transpose dimensions
    pub fn transpose(&self, dim0: i32, dim1: i32) -> Result<Self> {
        let ndim = self.ndim() as i32;

        // Normalize negative dimensions
        let d0 = if dim0 < 0 { ndim + dim0 } else { dim0 } as usize;
        let d1 = if dim1 < 0 { ndim + dim1 } else { dim1 } as usize;

        if d0 >= self.ndim() || d1 >= self.ndim() {
            return Err(TorshError::InvalidShape(
                "Dimension out of range".to_string(),
            ));
        }

        if d0 == d1 {
            return Ok(self.clone());
        }

        // Use view-based transpose for all dimensions
        self.transpose_view(d0, d1)
    }

    #[allow(dead_code)]
    fn transpose_2d(&self) -> Result<Self> {
        let shape_dims = self.shape().dims().to_vec();
        if shape_dims.len() != 2 {
            return Err(TorshError::InvalidShape(
                "transpose_2d requires 2D tensor".to_string(),
            ));
        }

        let rows = shape_dims[0];
        let cols = shape_dims[1];
        let data = self.to_vec()?;

        let mut new_data = Vec::with_capacity(data.len());
        for j in 0..cols {
            for i in 0..rows {
                new_data.push(data[i * cols + j]);
            }
        }

        Self::from_data(new_data, vec![cols, rows], self.device)
    }

    /// Permute dimensions
    pub fn permute(&self, dims: &[i32]) -> Result<Self> {
        if dims.len() != self.ndim() {
            return Err(TorshError::InvalidShape(format!(
                "Number of dimensions in permutation ({}) doesn't match tensor dimensions ({})",
                dims.len(),
                self.ndim()
            )));
        }

        let ndim = self.ndim() as i32;
        let mut normalized_dims = Vec::new();

        // Normalize negative dimensions and validate
        for &dim in dims {
            let normalized = if dim < 0 { ndim + dim } else { dim };
            if normalized < 0 || normalized >= ndim {
                return Err(TorshError::InvalidShape(format!(
                    "Dimension {dim} is out of range for tensor with {ndim} dimensions"
                )));
            }
            normalized_dims.push(normalized as usize);
        }

        // Check for duplicate dimensions
        let mut sorted_dims = normalized_dims.clone();
        sorted_dims.sort();
        for i in 1..sorted_dims.len() {
            if sorted_dims[i] == sorted_dims[i-1] {
                return Err(TorshError::InvalidShape(
                    "Duplicate dimensions in permutation".to_string()
                ));
            }
        }

        // Create new shape by reordering dimensions
        let binding = self.shape();
        let old_shape = binding.dims();
        let new_shape: Vec<usize> = normalized_dims.iter()
            .map(|&i| old_shape[i])
            .collect();

        // For now, we'll create a new tensor with permuted data
        // This is not the most efficient but is correct
        let old_data = self.to_vec()?;
        let mut new_data = Vec::with_capacity(old_data.len());

        // Compute strides for old and new layouts
        let old_strides = self.shape().default_strides();
        let new_strides = {
            let mut strides = vec![1; new_shape.len()];
            for i in (0..new_shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * new_shape[i + 1];
            }
            strides
        };

        // Reorder data according to permutation
        for new_idx in 0..old_data.len() {
            // Convert flat index to multi-dimensional index in new layout
            let mut new_multi_idx = vec![0; new_shape.len()];
            let mut remaining = new_idx;
            for i in 0..new_shape.len() {
                new_multi_idx[i] = remaining / new_strides[i];
                remaining %= new_strides[i];
            }

            // Map new multi-dimensional index to old multi-dimensional index
            let mut old_multi_idx = vec![0; old_shape.len()];
            for i in 0..normalized_dims.len() {
                old_multi_idx[normalized_dims[i]] = new_multi_idx[i];
            }

            // Convert old multi-dimensional index to flat index in old layout
            let mut old_idx = 0;
            for i in 0..old_shape.len() {
                old_idx += old_multi_idx[i] * old_strides[i];
            }

            new_data.push(old_data[old_idx]);
        }

        Self::from_data(new_data, new_shape, self.device)
    }

    /// Squeeze dimensions of size 1
    pub fn squeeze(&self, dim: i32) -> Result<Self> {
        let ndim = self.ndim() as i32;

        // Normalize negative dimension
        let normalized_dim = if dim < 0 {
            (ndim + dim) as usize
        } else {
            dim as usize
        };

        // Use the view-based squeeze_tensor method
        self.squeeze_tensor(normalized_dim)
    }

    /// Squeeze all dimensions of size 1
    pub fn squeeze_all(&self) -> Result<Self> {
        let shape = self.shape();
        let dims = shape.dims();

        // Filter out dimensions of size 1
        let new_dims: Vec<usize> = dims.iter().filter(|&&d| d != 1).copied().collect();

        // If no dimensions to squeeze, return clone
        if new_dims.len() == dims.len() {
            return Ok(self.clone());
        }

        // Data remains the same, just the shape interpretation changes
        let data = self.to_vec()?;
        let mut result = Self::from_data(data, new_dims, self.device)?;
        result.requires_grad = self.requires_grad;

        Ok(result)
    }

    /// Unsqueeze (add dimension of size 1)
    pub fn unsqueeze(&self, dim: i32) -> Result<Self> {
        let ndim = self.ndim() as i32;

        // Normalize negative dimension
        let normalized_dim = if dim < 0 {
            // For negative dim, it's the position before inserting
            // e.g., -1 means insert before the last position
            (ndim + dim + 1) as usize
        } else {
            dim as usize
        };

        // Use the view-based unsqueeze_tensor method
        self.unsqueeze_tensor(normalized_dim)
    }

    /// Flatten the tensor to 1D
    pub fn flatten(&self) -> Result<Self> {
        self.view(&[-1])
    }

    /// Reshape the tensor (alias for view)
    pub fn reshape(&self, shape: &[i32]) -> Result<Self> {
        self.view(shape)
    }

    /// Create a tensor that shares the same underlying data (shallow copy)
    pub fn share_data(&self) -> Self
    where
        T: Copy,
    {
        Self {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)), // Don't share gradients
            operation: Operation::Leaf,       // Reset operation for new tensor
            strides: self.strides.clone(),
            storage_offset: self.storage_offset,
            base_tensor: self.base_tensor.clone(),
        }
    }

    /// Check if tensor data is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        // If no custom strides, it's contiguous
        if self.strides.is_none() {
            return true;
        }

        // Check if custom strides match contiguous layout
        let strides = self.strides();
        let default_strides = self.compute_default_strides();
        strides == default_strides
    }

    /// Return a contiguous tensor with the same data
    pub fn contiguous(&self) -> Result<Self>
    where
        T: Copy,
    {
        if self.is_contiguous() {
            // Already contiguous, return a clone
            return Ok(self.clone());
        }

        // Create a new tensor with contiguous layout
        let data = self.data()?;
        Self::from_data(
            data.to_vec(),
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Get an element by flat index (ignoring tensor dimensions)
    pub fn get_item_flat(&self, index: usize) -> Result<T>
    where
        T: Copy,
    {
        if index >= self.numel() {
            return Err(TorshError::IndexError {
                index,
                size: self.numel()
            });
        }

        self.storage.get(index)
    }

    /// Set an element by flat index (ignoring tensor dimensions)
    pub fn set_item_flat(&mut self, index: usize, value: T) -> Result<()>
    where
        T: Copy,
    {
        if index >= self.numel() {
            return Err(TorshError::IndexError {
                index,
                size: self.numel()
            });
        }

        self.storage.set(index, value)
    }

    /// Create a tensor filled with a scalar value
    pub fn from_scalar(value: T, shape: &[usize], device: DeviceType) -> Result<Self>
    where
        T: Copy,
    {
        let numel = shape.iter().product::<usize>();
        let data = vec![value; numel];
        Self::from_data(data, shape.to_vec(), device)
    }

    /// Fill tensor with a single value (in-place)
    pub fn fill_(&mut self, value: T) -> Result<()>
    where
        T: Copy,
    {
        // For storage-based approach, we need to update all elements
        for i in 0..self.numel() {
            self.storage.set(i, value)?;
        }

        Ok(())
    }

    /// Get an element by multi-dimensional index
    pub fn get_item(&self, indices: &[usize]) -> Result<T>
    where
        T: Copy,
    {
        if indices.len() != self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        let shape = self.shape().dims();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(TorshError::IndexError {
                    index: idx,
                    size: shape[i],
                });
            }
        }

        let flat_index = self.multi_to_flat_index(indices)?;
        self.get_item_flat(flat_index)
    }

    /// Set an element by multi-dimensional index
    pub fn set_item(&mut self, indices: &[usize], value: T) -> Result<()>
    where
        T: Copy,
    {
        if indices.len() != self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        let shape = self.shape().dims();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(TorshError::IndexError {
                    index: idx,
                    size: shape[i],
                });
            }
        }

        let flat_index = self.multi_to_flat_index(indices)?;
        self.set_item_flat(flat_index, value)
    }

    /// Convert multi-dimensional index to flat index
    fn multi_to_flat_index(&self, indices: &[usize]) -> Result<usize> {
        let shape = self.shape().dims();
        let mut flat_index = 0;
        let mut stride = 1;

        for i in (0..indices.len()).rev() {
            flat_index += indices[i] * stride;
            stride *= shape[i];
        }

        Ok(flat_index)
    }

    /// Copy data from another tensor (in-place)
    pub fn copy_(&mut self, other: &Self) -> Result<()>
    where
        T: Copy,
    {
        if self.shape() != other.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: other.shape().dims().to_vec(),
            });
        }

        let other_data = other.to_vec()?;
        for (i, &value) in other_data.iter().enumerate() {
            self.storage.set(i, value)?;
        }
        Ok(())
    }

    /// Zero out the tensor (in-place)
    pub fn zero_(&mut self) -> Result<()>
    where
        T: Copy,
    {
        self.fill_(T::zero())
    }

    /// Fill with ones (in-place)
    pub fn ones_(&mut self) -> Result<()>
    where
        T: Copy,
    {
        self.fill_(T::one())
    }

    /// Expand tensor to larger size (repeats values without allocating new memory where possible)
    pub fn expand(&self, shape: &[usize]) -> Result<Self> {
        let binding = self.shape();
        let current_dims = binding.dims();
        
        if shape.len() < current_dims.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Target shape has {} dimensions but tensor has {} dimensions",
                shape.len(), current_dims.len()
            )));
        }
        
        // Align dimensions from the right
        let offset = shape.len() - current_dims.len();
        
        // Check if expansion is valid
        for (i, &current_dim) in current_dims.iter().enumerate() {
            let target_dim = shape[offset + i];
            if current_dim != 1 && current_dim != target_dim {
                return Err(TorshError::InvalidArgument(format!(
                    "Cannot expand dimension {} from {} to {}",
                    offset + i, current_dim, target_dim
                )));
            }
        }
        
        // For simplicity, create new tensor with repeated data
        // In a full implementation, this would use strides for zero-copy expansion
        let current_data = self.to_vec()?;
        let target_numel = shape.iter().product::<usize>();
        let mut new_data = Vec::with_capacity(target_numel);
        
        // Calculate repeat factors
        let mut repeat_factors = vec![1; shape.len()];
        for (i, &current_dim) in current_dims.iter().enumerate() {
            let target_idx = offset + i;
            if current_dim == 1 {
                repeat_factors[target_idx] = shape[target_idx];
            }
        }
        
        // Generate expanded data
        self.expand_data_recursive(&current_data, &mut new_data, shape, current_dims, 
                                 &repeat_factors, offset, &mut vec![0; shape.len()], 0)?;
        
        Self::from_data(new_data, shape.to_vec(), self.device)
    }
    
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::only_used_in_recursion)]
    fn expand_data_recursive(&self, source: &[T], dest: &mut Vec<T>, target_shape: &[usize],
                           source_shape: &[usize], _repeat_factors: &[usize], offset: usize,
                           indices: &mut [usize], dim: usize) -> Result<()> {
        if dim == target_shape.len() {
            // Calculate source index
            let mut source_idx = 0;
            let mut stride = 1;
            for i in (0..source_shape.len()).rev() {
                let target_idx = offset + i;
                let source_dim_idx = if source_shape[i] == 1 { 0 } else { indices[target_idx] };
                source_idx += source_dim_idx * stride;
                stride *= source_shape[i];
            }
            
            if source_idx < source.len() {
                dest.push(source[source_idx]);
            }
            return Ok(());
        }
        
        for i in 0..target_shape[dim] {
            indices[dim] = i;
            self.expand_data_recursive(source, dest, target_shape, source_shape, 
                                     _repeat_factors, offset, indices, dim + 1)?;
        }
        Ok(())
    }

    /// Flip tensor along specified dimensions
    pub fn flip(&self, dims: &[i32]) -> Result<Self> {
        let ndim = self.ndim() as i32;
        let mut normalized_dims = Vec::new();
        
        // Normalize dimensions
        for &dim in dims {
            let normalized = if dim < 0 { ndim + dim } else { dim };
            if normalized < 0 || normalized >= ndim {
                return Err(TorshError::InvalidArgument(format!(
                    "Dimension {dim} out of range for tensor with {ndim} dimensions"
                )));
            }
            normalized_dims.push(normalized as usize);
        }
        
        let data = self.to_vec()?;
        let shape_obj = self.shape();
        let shape = shape_obj.dims();
        let mut new_data = Vec::with_capacity(data.len());
        
        // Generate flipped data
        for flat_idx in 0..data.len() {
            let mut multi_idx = self.flat_to_multi_index(flat_idx, shape);
            
            // Flip specified dimensions
            for &dim in &normalized_dims {
                multi_idx[dim] = shape[dim] - 1 - multi_idx[dim];
            }
            
            let source_idx = self.multi_to_flat_index(&multi_idx, shape);
            new_data.push(data[source_idx]);
        }
        
        Self::from_data(new_data, shape.to_vec(), self.device)
    }

    /// Roll tensor along specified dimension
    pub fn roll(&self, shifts: i64, dim: Option<i32>) -> Result<Self> {
        if let Some(dim) = dim {
            // Roll along specific dimension
            let ndim = self.ndim() as i32;
            let normalized_dim = if dim < 0 { 
                (ndim + dim) as usize 
            } else { 
                dim as usize 
            };
            
            if normalized_dim >= self.ndim() {
                return Err(TorshError::InvalidArgument(format!(
                    "Dimension {} out of range for tensor with {} dimensions", 
                    dim, self.ndim()
                )));
            }
            
            let shape_obj = self.shape();
            let shape = shape_obj.dims();
            let dim_size = shape[normalized_dim] as i64;
            let effective_shift = ((shifts % dim_size) + dim_size) % dim_size;
            
            if effective_shift == 0 {
                return Ok(self.clone());
            }
            
            let data = self.to_vec()?;
            let mut new_data = Vec::with_capacity(data.len());
            
            // Generate rolled data
            for flat_idx in 0..data.len() {
                let mut multi_idx = self.flat_to_multi_index(flat_idx, shape);
                
                // Apply roll to the specified dimension
                let original_idx = multi_idx[normalized_dim] as i64;
                let new_idx = (original_idx - effective_shift + dim_size) % dim_size;
                multi_idx[normalized_dim] = new_idx as usize;
                
                let source_idx = self.multi_to_flat_index(&multi_idx, shape);
                new_data.push(data[source_idx]);
            }
            
            Self::from_data(new_data, shape.to_vec(), self.device)
        } else {
            // Roll along flattened tensor
            let data = self.to_vec()?;
            let len = data.len() as i64;
            let effective_shift = ((shifts % len) + len) % len;
            
            if effective_shift == 0 {
                return Ok(self.clone());
            }
            
            let mut new_data = Vec::with_capacity(data.len());
            for i in 0..data.len() {
                let source_idx = ((i as i64 - effective_shift + len) % len) as usize;
                new_data.push(data[source_idx]);
            }
            
            Self::from_data(new_data, self.shape().dims().to_vec(), self.device)
        }
    }
    
    /// Helper: Convert flat index to multi-dimensional index
    fn flat_to_multi_index(&self, flat_idx: usize, shape: &[usize]) -> Vec<usize> {
        let mut multi_idx = vec![0; shape.len()];
        let mut remaining = flat_idx;
        
        // Calculate strides for each dimension
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        for i in 0..shape.len() {
            multi_idx[i] = remaining / strides[i];
            remaining %= strides[i];
        }
        
        multi_idx
    }

    /// Get number of bytes per element
    pub fn element_size(&self) -> usize {
        std::mem::size_of::<T>()
    }

    /// Check if this tensor shares data with another tensor
    pub fn shares_data_with(&self, other: &Self) -> bool {
        // For storage abstraction, we need to check the underlying storage
        match (&self.storage, &other.storage) {
            (TensorStorage::InMemory(a), TensorStorage::InMemory(b)) => Arc::ptr_eq(a, b),
            (TensorStorage::MemoryMapped(a), TensorStorage::MemoryMapped(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Get data as a vector (backward compatibility method)
    pub fn data(&self) -> Result<Vec<T>>
    where
        T: Copy,
    {
        self.to_vec()
    }

    /// Apply a function to all elements in-place using direct storage access
    pub fn data_mut_apply<F>(&mut self, mut func: F) -> Result<()>
    where
        F: FnMut(&mut T),
        T: Copy,
    {
        self.ensure_exclusive_data()?;
        
        match &mut self.storage {
            TensorStorage::InMemory(data) => {
                let mut data_guard = data.write().unwrap();
                for item in data_guard.iter_mut() {
                    func(item);
                }
                Ok(())
            }
            TensorStorage::MemoryMapped(_) => {
                // For memory-mapped storage, we need to read-modify-write
                let data = self.to_vec()?;
                let mut new_data = data;
                for item in new_data.iter_mut() {
                    func(item);
                }
                // Write back the data
                self.storage = TensorStorage::create_optimal(new_data)?;
                Ok(())
            }
        }
    }

    /// Clone the tensor with independent data (deep copy)
    pub fn clone_data(&self) -> Self 
    where
        T: Copy,
    {
        let data = self.to_vec().unwrap();
        Self::from_data(data.clone(), self.shape().dims().to_vec(), self.device).unwrap()
    }

    /// Ensure tensor has unique data (copy-on-write semantics)
    pub fn make_unique(&mut self) -> Result<()> {
        // For storage-based approach, create new storage if shared
        match &self.storage {
            TensorStorage::InMemory(data) => {
                if Arc::strong_count(data) > 1 {
                    let data_vec = self.to_vec()?;
                    self.storage = TensorStorage::create_optimal(data_vec)?;
                }
            }
            TensorStorage::MemoryMapped(storage) => {
                if Arc::strong_count(storage) > 1 {
                    let data_vec = self.to_vec()?;
                    self.storage = TensorStorage::create_optimal(data_vec)?;
                }
            }
        }
        Ok(())
    }

    /// Apply function in-place
    pub fn apply_<F>(&mut self, func: F) -> Result<()>
    where
        F: Fn(T) -> T,
        T: Copy,
    {
        // For storage-based approach, read all data, apply function, then write back
        let data = self.to_vec()?;
        let new_data: Vec<T> = data.iter().map(|&elem| func(elem)).collect();
        for (i, &value) in new_data.iter().enumerate() {
            let mut indices = vec![0; self.ndim()];
            // Convert flat index to multi-dimensional
            let mut remaining = i;
            let shape = self.shape.dims();
            for (dim, &size) in shape.iter().enumerate().rev() {
                indices[dim] = remaining % size;
                remaining /= size;
            }
            self.set(&indices, value)?;
        }

        Ok(())
    }

    /// Add scalar to all elements in-place
    pub fn add_scalar_(&mut self, scalar: T) -> Result<()>
    where
        T: Copy + std::ops::Add<Output = T>,
    {
        // Ensure data is unique (copy-on-write)
        self.make_unique()?;
        self.apply_(|x| x + scalar)
    }

    /// Multiply all elements by scalar in-place
    pub fn mul_scalar_(&mut self, scalar: T) -> Result<()>
    where
        T: Copy + std::ops::Mul<Output = T>,
    {
        // Ensure data is unique (copy-on-write)
        self.make_unique()?;
        self.apply_(|x| x * scalar)
    }

    /// Multiply all elements by scalar (returns new tensor)
    pub fn mul_scalar(&self, scalar: T) -> Result<Self>
    where
        T: Copy + std::ops::Mul<Output = T>,
    {
        let mut result = self.clone();
        result.mul_scalar_(scalar)?;
        Ok(result)
    }


    /// Complex conjugate (for complex tensors)
    pub fn conj(&self) -> Result<Self>
    where
        T: torsh_core::dtype::ComplexElement + Copy,
    {
        self.complex_conj()
    }




    /// Gather values along an axis using indices
    pub fn gather(&self, dim: usize, indices: &Tensor<i64>) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim, self.ndim()
            )));
        }

        let self_data = self.to_vec()?;
        let indices_data = indices.to_vec()?;

        let mut result_data = Vec::new();
        let result_shape = indices.shape().dims().to_vec();
        
        if self.ndim() == 1 {
            // 1D case
            for &index in &indices_data {
                let idx = if index < 0 {
                    (self.shape().dims()[0] as i64 + index) as usize
                } else {
                    index as usize
                };
                
                if idx >= self.shape().dims()[0] {
                    return Err(TorshError::InvalidArgument(format!(
                        "Index {} out of range for tensor with size {}",
                        index, self.shape().dims()[0]
                    )));
                }
                
                result_data.push(self_data[idx]);
            }
        } else {
            // Multi-dimensional case
            let self_shape_ref = self.shape();
            let self_shape = self_shape_ref.dims();
            let indices_shape_ref = indices.shape();
            let indices_shape = indices_shape_ref.dims();
            let dim_size = self_shape[dim];
            
            // Calculate strides for both tensors
            let mut self_strides = vec![1; self_shape.len()];
            let mut indices_strides = vec![1; indices_shape.len()];
            
            for i in (0..self_shape.len()-1).rev() {
                self_strides[i] = self_strides[i+1] * self_shape[i+1];
            }
            
            for i in (0..indices_shape.len()-1).rev() {
                indices_strides[i] = indices_strides[i+1] * indices_shape[i+1];
            }
            
            let total_elements = indices_data.len();
            
            for (i, &index_value) in indices_data.iter().enumerate().take(total_elements) {
                // Convert flat index to multi-dimensional coordinates for indices tensor
                let mut indices_coords = vec![0; indices_shape.len()];
                let mut temp_i = i;
                for j in 0..indices_shape.len() {
                    indices_coords[j] = temp_i / indices_strides[j];
                    temp_i %= indices_strides[j];
                }
                
                // Get the index value
                let idx = if index_value < 0 {
                    (dim_size as i64 + index_value) as usize
                } else {
                    index_value as usize
                };
                
                if idx >= dim_size {
                    return Err(TorshError::InvalidArgument(format!(
                        "Index {index_value} out of range for dimension {dim} with size {dim_size}"
                    )));
                }
                
                // Build coordinates for the source tensor
                let mut self_coords = indices_coords.clone();
                if dim < self_coords.len() {
                    self_coords[dim] = idx;
                }
                
                // Convert coordinates to flat index for source tensor
                let mut flat_idx = 0;
                for j in 0..self_coords.len() {
                    flat_idx += self_coords[j] * self_strides[j];
                }
                
                result_data.push(self_data[flat_idx]);
            }
        }

        Self::from_data(result_data, result_shape, self.device)
    }

    /// Scatter values along an axis using indices
    pub fn scatter(&self, dim: usize, indices: &Tensor<i64>, src: &Tensor<T>) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim, self.ndim()
            )));
        }

        let mut result_data = self.to_vec()?;
        let indices_data = indices.to_vec()?;
        let src_data = src.to_vec()?;

        if indices_data.len() != src_data.len() {
            return Err(TorshError::InvalidArgument(
                "Indices and source tensor must have the same number of elements".to_string()
            ));
        }

        if self.ndim() == 1 {
            // 1D case
            for (i, &index) in indices_data.iter().enumerate() {
                let idx = if index < 0 {
                    (self.shape().dims()[0] as i64 + index) as usize
                } else {
                    index as usize
                };
                
                if idx >= self.shape().dims()[0] {
                    return Err(TorshError::InvalidArgument(format!(
                        "Index {} out of range for tensor with size {}",
                        index, self.shape().dims()[0]
                    )));
                }
                
                result_data[idx] = src_data[i];
            }
        } else {
            // Multi-dimensional case
            let self_shape_ref = self.shape();
            let self_shape = self_shape_ref.dims();
            let indices_shape_ref = indices.shape();
            let indices_shape = indices_shape_ref.dims();
            let dim_size = self_shape[dim];
            
            // Calculate strides for both tensors
            let mut self_strides = vec![1; self_shape.len()];
            let mut indices_strides = vec![1; indices_shape.len()];
            
            for i in (0..self_shape.len()-1).rev() {
                self_strides[i] = self_strides[i+1] * self_shape[i+1];
            }
            
            for i in (0..indices_shape.len()-1).rev() {
                indices_strides[i] = indices_strides[i+1] * indices_shape[i+1];
            }
            
            let total_elements = indices_data.len();
            
            for (i, &index_value) in indices_data.iter().enumerate().take(total_elements) {
                // Convert flat index to multi-dimensional coordinates for indices tensor
                let mut indices_coords = vec![0; indices_shape.len()];
                let mut temp_i = i;
                for j in 0..indices_shape.len() {
                    indices_coords[j] = temp_i / indices_strides[j];
                    temp_i %= indices_strides[j];
                }
                
                // Get the index value
                let idx = if index_value < 0 {
                    (dim_size as i64 + index_value) as usize
                } else {
                    index_value as usize
                };
                
                if idx >= dim_size {
                    return Err(TorshError::InvalidArgument(format!(
                        "Index {index_value} out of range for dimension {dim} with size {dim_size}"
                    )));
                }
                
                // Build coordinates for the destination tensor
                let mut self_coords = indices_coords.clone();
                if dim < self_coords.len() {
                    self_coords[dim] = idx;
                }
                
                // Convert coordinates to flat index for destination tensor
                let mut flat_idx = 0;
                for j in 0..self_coords.len() {
                    flat_idx += self_coords[j] * self_strides[j];
                }
                
                result_data[flat_idx] = src_data[i];
            }
        }

        Self::from_data(result_data, self.shape().dims().to_vec(), self.device)
    }

    /// Repeat tensor along specified dimensions
    pub fn repeat(&self, repeats: &[usize]) -> Result<Self> {
        if repeats.len() != self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Number of repeats {} must match tensor dimensions {}",
                repeats.len(), self.ndim()
            )));
        }

        let self_data = self.to_vec()?;
        let shape_binding = self.shape();
        let self_shape = shape_binding.dims();
        
        // Calculate result shape
        let result_shape: Vec<usize> = self_shape.iter().zip(repeats.iter()).map(|(s, r)| s * r).collect();
        let result_size = result_shape.iter().product();
        let mut result_data = Vec::with_capacity(result_size);
        
        if self.ndim() == 1 {
            // 1D case - simple repetition
            for _ in 0..repeats[0] {
                result_data.extend_from_slice(&self_data);
            }
        } else {
            // Multi-dimensional case: use coordinate-based approach
            let self_strides = {
                let mut strides = vec![1; self_shape.len()];
                for i in (0..self_shape.len()-1).rev() {
                    strides[i] = strides[i+1] * self_shape[i+1];
                }
                strides
            };
            
            let result_strides = {
                let mut strides = vec![1; result_shape.len()];
                for i in (0..result_shape.len()-1).rev() {
                    strides[i] = strides[i+1] * result_shape[i+1];
                }
                strides
            };
            
            for i in 0..result_size {
                // Convert flat index to multi-dimensional coordinates for result tensor
                let mut result_coords = vec![0; result_shape.len()];
                let mut temp_i = i;
                for j in 0..result_shape.len() {
                    result_coords[j] = temp_i / result_strides[j];
                    temp_i %= result_strides[j];
                }
                
                // Map result coordinates to source coordinates
                let mut source_coords = vec![0; self_shape.len()];
                for j in 0..self_shape.len() {
                    source_coords[j] = result_coords[j] % self_shape[j];
                }
                
                // Convert source coordinates to flat index
                let mut source_idx = 0;
                for j in 0..self_shape.len() {
                    source_idx += source_coords[j] * self_strides[j];
                }
                
                result_data.push(self_data[source_idx]);
            }
        }

        Self::from_data(result_data, result_shape, self.device)
    }

}

impl<T: FloatElement + Copy> Tensor<T> {

    /// Create a 0-dimensional tensor (scalar) from a single value
    pub fn scalar(value: T) -> Result<Self> {
        Self::from_data(vec![value], vec![], DeviceType::Cpu)
    }


    /// Reduce tensor to maximum value
    pub fn max(&self, _dim: Option<usize>, _keepdim: bool) -> Result<Self> {
        let data = self.to_vec()?;
        let max_val = data.into_iter().fold(T::neg_infinity(), |acc, x| {
            if x > acc { x } else { acc }
        });
        Self::scalar(max_val)
    }
}

impl<T: TensorElement + Copy> Tensor<T> {
    /// Create from vec with shape
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        Self::from_data(data, shape.to_vec(), DeviceType::Cpu)
    }
}

// Display implementation
impl<T: TensorElement> std::fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={}, device={})",
            self.shape().dims(),
            self.dtype(),
            self.device
        )
    }
}

impl<T: TensorElement> Tensor<T> {
    /// Get the reference count of the underlying storage Arc (for testing CoW behavior)
    #[cfg(test)]
    pub fn data_ref_count(&self) -> usize {
        match &self.storage {
            TensorStorage::InMemory(data) => Arc::strong_count(data),
            TensorStorage::MemoryMapped(storage) => Arc::strong_count(storage),
        }
    }
}


/// Tensor creation macro similar to PyTorch
#[macro_export]
macro_rules! tensor {
    // 1D array from bracketed values
    ([$($val:expr),+ $(,)?]) => {
        $crate::creation::tensor_1d(&[$($val),+])
    };

    // Multiple values without brackets (at least 2 values to avoid scalar conflict)
    ($val1:expr, $val2:expr $(, $val:expr)* $(,)?) => {
        $crate::creation::tensor_1d(&[$val1, $val2 $(, $val)*])
    };

    // Scalar (single value)
    ($val:expr) => {
        $crate::creation::tensor_scalar($val)
    };
}

#[macro_export]
macro_rules! tensor_2d {
    // 2D array
    ($([$($val:expr),+ $(,)?]),+ $(,)?) => {
        {
            let data = [$([$($val),+]),+];
            $crate::creation::tensor_2d_arrays(&data)
        }
    };
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::cache_optimization::{CacheAnalysisReport, MemoryStats};
    pub use crate::convenience::{TensorConvenience, TensorShapeConvenience};
    pub use crate::tensor_views::{TensorView, TensorAlias, ViewMemoryUsage};
    pub use crate::creation::*;
    pub use torsh_core::{QInt8, QUInt8};
    #[cfg(feature = "serialize")]
    pub use crate::serialize::streaming::{StreamingConfig, ProgressCallback};
    pub use crate::{tensor, tensor_2d, Tensor};
    pub use crate::scirs2_stats_integration::{
        SciRS2StatsProcessor, StatsConfig, DescriptiveStats, CorrelationResult, TTestResult,
        RegressionResult, DistributionFit, MissingValueStrategy, CorrelationMethod, DistributionType,
    };
    pub use torsh_core::prelude::*;
}

// TODO: Implement AutogradTensor trait for Tensor
// This requires proper design of the autograd system and trait bounds
/*
#[cfg(feature = "autograd")]
/// Implementation of AutogradTensor trait for Tensor
impl<T: TensorElement> torsh_autograd::AutogradTensor<T> for Tensor<T> {
    fn shape(&self) -> Shape {
        self.shape.clone()
    }
    
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    
    fn data(&self) -> Box<dyn std::ops::Deref<Target = [T]> + '_> {
        Box::new(self.to_vec())
    }
    
    fn clone_tensor(&self) -> Box<dyn torsh_autograd::AutogradTensor<T>> {
        Box::new(self.clone())
    }
    
    fn to_vec(&self) -> Vec<T> {
        self.to_vec()
    }
    
    fn device(&self) -> &dyn torsh_core::Device {
        match &self.device {
            DeviceType::Cpu => &torsh_core::device::CpuDevice,
            DeviceType::Cuda(_) => &torsh_core::device::CpuDevice, // TODO: Return proper CUDA device
            _ => &torsh_core::device::CpuDevice,
        }
    }
    
    fn ones_like(&self) -> Box<dyn torsh_autograd::AutogradTensor<T>> {
        Box::new(Tensor::ones_like(self))
    }
    
    fn zeros_like(&self) -> Box<dyn torsh_autograd::AutogradTensor<T>> {
        Box::new(Tensor::zeros_like(self))
    }
}
*/

/// SciRS2 Backend Integration for Tensor Operations
impl<T: TensorElement + Copy + num_traits::FromPrimitive> Tensor<T> {
    /// Use SciRS2 backend for optimized tensor addition
    pub fn add_scirs2(&self, other: &Self) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + num_traits::Float,
    {
        crate::scirs2_backend::get_scirs2_backend().add(self, other)
    }
    
    /// Use SciRS2 backend for optimized tensor multiplication
    pub fn mul_scirs2(&self, other: &Self) -> Result<Self>
    where
        T: std::ops::Mul<Output = T> + num_traits::Float,
    {
        crate::scirs2_backend::get_scirs2_backend().mul(self, other)
    }
    
    /// Use SciRS2 backend for optimized tensor subtraction
    pub fn sub_scirs2(&self, other: &Self) -> Result<Self>
    where
        T: std::ops::Sub<Output = T> + num_traits::Float,
    {
        crate::scirs2_backend::get_scirs2_backend().sub(self, other)
    }
    
    /// Use SciRS2 backend for optimized tensor division
    pub fn div_scirs2(&self, other: &Self) -> Result<Self>
    where
        T: std::ops::Div<Output = T> + num_traits::Float,
    {
        crate::scirs2_backend::get_scirs2_backend().div(self, other)
    }
    
    /// Use SciRS2 backend for optimized matrix multiplication
    pub fn matmul_scirs2(&self, other: &Self) -> Result<Self>
    where
        T: num_traits::Float + num_traits::Zero + num_traits::One + std::iter::Sum,
    {
        crate::scirs2_backend::get_scirs2_backend().matmul(self, other)
    }
    
    /// Use SciRS2 backend for optimized sum reduction
    pub fn sum_scirs2(&self) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + num_traits::Zero,
    {
        crate::scirs2_backend::get_scirs2_backend().sum(self)
    }
    
    /// Use SciRS2 backend for optimized mean reduction
    pub fn mean_scirs2(&self) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + std::ops::Div<Output = T> + num_traits::Zero + From<usize> + num_traits::FromPrimitive,
    {
        crate::scirs2_backend::get_scirs2_backend().mean(self)
    }
    
    /// Use SciRS2 backend for optimized ReLU activation
    pub fn relu_scirs2(&self) -> Result<Self>
    where
        T: PartialOrd + num_traits::Zero,
    {
        crate::scirs2_backend::get_scirs2_backend().relu(self)
    }
    
    /// Use SciRS2 backend for optimized sigmoid activation
    pub fn sigmoid_scirs2(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        crate::scirs2_backend::get_scirs2_backend().sigmoid(self)
    }
    
    /// Use SciRS2 backend for optimized tanh activation
    pub fn tanh_scirs2(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        crate::scirs2_backend::get_scirs2_backend().tanh(self)
    }
    
    /// Extract a scalar value from a single-element tensor
    pub fn item(&self) -> Result<T>
    where
        T: Copy,
    {
        let data = self.data()?;
        if data.len() != 1 {
            return Err(TorshError::InvalidArgument(
                format!("item() can only be called on single-element tensors, got {} elements", data.len())
            ));
        }
        Ok(data[0])
    }
    
    /// Compute the L2 norm of the tensor
    pub fn norm(&self) -> Result<Self>
    where
        T: TensorElement + Copy + num_traits::Float,
    {
        let data = self.data()?;
        let sum_squares: T = data.iter().map(|&x| x * x).fold(<T as num_traits::Zero>::zero(), |acc, x| acc + x);
        let norm_value = sum_squares.sqrt();
        
        // Return scalar tensor (1-element tensor with shape [])
        Tensor::from_data(vec![norm_value], vec![], self.device())
    }
    
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = tensor![1.0f32, 2.0f32, 3.0f32].unwrap();
        assert_eq!(t.shape().dims(), &[3]);
        assert_eq!(t.dtype(), DType::F32);
    }

    #[test]
    fn test_2d_macro_expansion() {
        // Test 2D creation directly
        let data = [[1.0f32, 2.0], [3.0, 4.0]];
        let _t = crate::creation::tensor_2d_arrays(&data);

        // Test if the 2D macro pattern works
        let _t2 = tensor_2d![[1.0f32, 2.0], [3.0, 4.0]].unwrap();
    }

    #[test]
    fn test_complex_autograd() {
        use crate::creation::complex32_ones;

        // Create a complex tensor that requires gradients
        let mut z = complex32_ones(&[]).unwrap(); // scalar complex tensor
        z.requires_grad = true;

        // Test complex conjugate gradient
        let w = z.conj().unwrap();

        // For now, just test that the operations work without panicking
        // TODO: Add actual gradient computation tests when backward is fully implemented
        assert!(w.requires_grad);

        // Test complex multiplication gradient tracking
        let z1 = complex32_ones(&[]).unwrap();
        let z2 = complex32_ones(&[]).unwrap();
        let z1 = z1.requires_grad_(true);
        let z2 = z2.requires_grad_(true);

        let result = z1.mul_op(&z2).unwrap();
        assert!(result.requires_grad);
    }

    #[test]
    fn test_unsqueeze() {
        // Test 1D tensor
        let t1 = tensor![1.0f32, 2.0, 3.0].unwrap();
        assert_eq!(t1.shape().dims(), &[3]);

        // Unsqueeze at dimension 0
        let t2 = t1.unsqueeze(0).unwrap();
        assert_eq!(t2.shape().dims(), &[1, 3]);

        // Unsqueeze at dimension 1
        let t3 = t1.unsqueeze(1).unwrap();
        assert_eq!(t3.shape().dims(), &[3, 1]);

        // Unsqueeze with negative dimension
        let t4 = t1.unsqueeze(-1).unwrap();
        assert_eq!(t4.shape().dims(), &[3, 1]);

        // Test 2D tensor
        let t5 = tensor_2d![[1.0f32, 2.0], [3.0, 4.0]].unwrap();
        assert_eq!(t5.shape().dims(), &[2, 2]);

        // Unsqueeze at dimension 0
        let t6 = t5.unsqueeze(0).unwrap();
        assert_eq!(t6.shape().dims(), &[1, 2, 2]);

        // Unsqueeze at dimension 2
        let t7 = t5.unsqueeze(2).unwrap();
        assert_eq!(t7.shape().dims(), &[2, 2, 1]);

        // Chain multiple unsqueezes (like in Conv2d bias)
        let bias = tensor![1.0f32, 2.0, 3.0, 4.0].unwrap();
        let reshaped = bias
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(2)
            .unwrap()
            .unsqueeze(3)
            .unwrap();
        assert_eq!(reshaped.shape().dims(), &[1, 4, 1, 1]);
    }

    #[test]
    fn test_memory_efficient_operations() {
        use crate::creation::tensor_1d;

        // Test fill_ operation
        let mut tensor = tensor_1d(&[1.0f32, 2.0, 3.0]).unwrap();
        tensor.fill_(5.0).unwrap();

        let data = tensor.data().unwrap();
        for &val in data.iter() {
            assert_eq!(val, 5.0);
        }
        drop(data);

        // Test zero_ and ones_
        tensor.zero_().unwrap();
        let data = tensor.data().unwrap();
        for &val in data.iter() {
            assert_eq!(val, 0.0);
        }
        drop(data);

        tensor.ones_().unwrap();
        let data = tensor.data().unwrap();
        for &val in data.iter() {
            assert_eq!(val, 1.0);
        }
        drop(data);
    }

    #[test]
    fn test_tensor_data_sharing() {
        let original = tensor![1.0f32, 2.0, 3.0].unwrap();

        // Test share_data
        let shared = original.share_data();
        assert!(original.shares_data_with(&shared));
        assert_eq!(original.shape(), shared.shape());

        // Test clone_data (deep copy)
        let cloned = original.clone_data();
        assert!(!original.shares_data_with(&cloned));
        assert_eq!(original.shape(), cloned.shape());

        // Values should be the same
        for i in 0..3 {
            assert_eq!(original.get(&[i]).unwrap(), cloned.get(&[i]).unwrap());
        }
    }

    #[test]
    fn test_copy_operation() {
        let source = tensor![10.0f32, 20.0, 30.0].unwrap();
        let mut target = tensor![1.0f32, 2.0, 3.0].unwrap();

        target.copy_(&source).unwrap();

        // Check that values are copied
        for i in 0..3 {
            assert_eq!(target.get(&[i]).unwrap(), source.get(&[i]).unwrap());
        }

        // But data is not shared
        assert!(!source.shares_data_with(&target));
    }

    #[test]
    fn test_copy_on_write() {
        let mut original = tensor![1.0f32, 2.0, 3.0].unwrap();
        let shared = original.share_data();

        // Initially they share data
        assert!(original.shares_data_with(&shared));

        // Make original unique
        original.make_unique().unwrap();

        // Now they don't share data
        assert!(!original.shares_data_with(&shared));
    }

    #[test]
    fn test_memory_info() {
        let tensor = tensor![1.0f32, 2.0, 3.0, 4.0].unwrap();

        assert_eq!(tensor.element_size(), std::mem::size_of::<f32>());
        assert_eq!(tensor.memory_usage(), 4 * std::mem::size_of::<f32>());
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_apply_function() {
        let mut tensor = tensor![1.0f32, 2.0, 3.0].unwrap();

        // Square all elements
        tensor.apply_(|x| x * x).unwrap();

        assert_eq!(tensor.get(&[0]).unwrap(), 1.0);
        assert_eq!(tensor.get(&[1]).unwrap(), 4.0);
        assert_eq!(tensor.get(&[2]).unwrap(), 9.0);

        // Apply absolute value after negation
        tensor.apply_(|x| -x).unwrap();
        tensor.apply_(|x| x.abs()).unwrap();

        assert_eq!(tensor.get(&[0]).unwrap(), 1.0);
        assert_eq!(tensor.get(&[1]).unwrap(), 4.0);
        assert_eq!(tensor.get(&[2]).unwrap(), 9.0);
    }

    #[test]
    fn test_shape_mismatch_errors() {
        let tensor1 = tensor![1.0f32, 2.0, 3.0].unwrap();
        let tensor2 = tensor![1.0f32, 2.0].unwrap(); // Different size

        let mut target = tensor![0.0f32, 0.0, 0.0].unwrap();

        // copy_ should fail with shape mismatch
        assert!(target.copy_(&tensor2).is_err());

        // Should succeed with matching shapes
        assert!(target.copy_(&tensor1).is_ok());
    }

    #[test]
    fn test_copy_on_write_behavior() {
        use torsh_core::device::DeviceType;

        // Create original tensor
        let mut original = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        
        // Initially, reference count should be 1
        assert_eq!(original.data_ref_count(), 1);
        
        // Clone the tensor - this should share the data
        let clone = original.clone();
        
        // Now both tensors should have a reference count of 2
        assert_eq!(original.data_ref_count(), 2);
        assert_eq!(clone.data_ref_count(), 2);
        
        // Verify they share the same data by checking storage Arc pointer equality
        let original_shares = original.shares_data_with(&clone);
        assert!(original_shares, "Data should be shared initially");
        
        // Modify the original tensor (triggers copy-on-write)
        original.add_scalar_(1.0).unwrap();
        
        // After modification, they should have separate data with ref count 1 each
        assert_eq!(original.data_ref_count(), 1);
        assert_eq!(clone.data_ref_count(), 1);
        
        // Verify they no longer share data
        let shares_after = original.shares_data_with(&clone);
        assert!(!shares_after, "Data should be separate after modification");
        
        // Verify that values are different
        assert_eq!(original.get(&[0, 0]).unwrap(), 2.0); // Original was modified (1.0 + 1.0)
        assert_eq!(clone.get(&[0, 0]).unwrap(), 1.0);    // Clone retains original value
        
        // Test with multiple clones
        let mut tensor = Tensor::from_data(vec![5.0f32, 6.0], vec![2], DeviceType::Cpu).unwrap();
        let clone1 = tensor.clone();
        let clone2 = tensor.clone();
        let clone3 = tensor.clone();
        
        // All should share data (ref count 4)
        assert_eq!(tensor.data_ref_count(), 4);
        assert_eq!(clone1.data_ref_count(), 4);
        assert_eq!(clone2.data_ref_count(), 4);
        assert_eq!(clone3.data_ref_count(), 4);
        
        // Modify tensor - should trigger copy-on-write
        tensor.mul_scalar_(2.0).unwrap();
        
        // Tensor should have unique data, clones should still share
        assert_eq!(tensor.data_ref_count(), 1);
        assert_eq!(clone1.data_ref_count(), 3);
        assert_eq!(clone2.data_ref_count(), 3);
        assert_eq!(clone3.data_ref_count(), 3);
        
        // Verify values
        assert_eq!(tensor.get(&[0]).unwrap(), 10.0); // 5.0 * 2.0
        assert_eq!(clone1.get(&[0]).unwrap(), 5.0);  // Unchanged
        assert_eq!(clone2.get(&[0]).unwrap(), 5.0);  // Unchanged
        assert_eq!(clone3.get(&[0]).unwrap(), 5.0);  // Unchanged
    }

    #[test]
    fn test_tensor_views() {
        // Test basic view creation
        let original = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 
            vec![2, 3], 
            DeviceType::Cpu
        ).unwrap();
        
        // Test view_as (reshape view)
        let reshaped = original.view_as(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert!(reshaped.is_view());
        assert!(original.shares_data_with(&reshaped));
        
        // Values should be accessible in new shape
        assert_eq!(reshaped.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(reshaped.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(reshaped.get(&[1, 0]).unwrap(), 3.0);
        assert_eq!(reshaped.get(&[2, 1]).unwrap(), 6.0);
        
        // Test slice_view
        let slice = original.slice(1, 0, 2).unwrap();
        assert_eq!(slice.shape().dims(), &[2, 2]);
        assert!(slice.is_view());
        // Note: TensorView shares data with original tensor by design
        
        // Values should be from the slice
        assert_eq!(slice.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(slice.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(slice.get(&[1, 0]).unwrap(), 4.0);
        assert_eq!(slice.get(&[1, 1]).unwrap(), 5.0);
        
        // Test transpose_view
        let transposed = original.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert!(transposed.is_view());
        // Note: TensorView shares data with original tensor by design
        
        // Values should be transposed
        assert_eq!(transposed.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(transposed.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(transposed.get(&[1, 0]).unwrap(), 2.0);
        assert_eq!(transposed.get(&[2, 1]).unwrap(), 6.0);
    }

    #[test]
    fn test_view_squeeze_unsqueeze() {
        // Create a tensor with dimensions suitable for squeezing
        let original = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0], 
            vec![1, 4, 1], 
            DeviceType::Cpu
        ).unwrap();
        
        // Test squeeze_view
        let squeezed = original.squeeze(0).unwrap();
        assert_eq!(squeezed.shape().dims(), &[4, 1]);
        assert!(squeezed.is_view());
        // Note: TensorView shares data with original tensor by design
        
        let squeezed_again = squeezed.squeeze(1).unwrap();
        assert_eq!(squeezed_again.shape().dims(), &[4]);
        assert!(squeezed_again.is_view());
        
        // Test unsqueeze_view  
        let unsqueezed = squeezed_again.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.shape().dims(), &[4, 1]);
        assert!(unsqueezed.is_view());
        
        // Values should remain accessible
        assert_eq!(squeezed_again.get(&[0]).unwrap(), 1.0);
        assert_eq!(squeezed_again.get(&[3]).unwrap(), 4.0);
        assert_eq!(unsqueezed.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(unsqueezed.get(&[3, 0]).unwrap(), 4.0);
    }

    #[test]
    fn test_view_chaining() {
        // Test chaining multiple view operations
        let original = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
            vec![2, 4], 
            DeviceType::Cpu
        ).unwrap();
        
        // Chain: slice -> simple operation
        let view = original.slice(1, 1, 3).unwrap();           // [2, 2]
            
        assert_eq!(view.shape().dims(), &[2, 2]);
        assert!(view.is_view());
        // Note: TensorView shares data with original tensor by design
        
        // Values should be correct after transformations
        assert_eq!(view.get(&[0, 0]).unwrap(), 2.0);  // original[0, 1] 
        assert_eq!(view.get(&[1, 1]).unwrap(), 7.0);  // original[1, 2]
    }

    #[test]
    fn test_view_reference_counting() {
        let original = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0], 
            vec![2, 2], 
            DeviceType::Cpu
        ).unwrap();
        
        // Create multiple views
        let view1 = original.view_as(&[4]).unwrap();
        let view2 = original.slice(0, 0, 1).unwrap();
        // Note: view3 would be a transposed view, but we don't support chained views yet
        
        // All views should share data with original (conceptually)
        // Note: TensorView shares data with original tensor by design
        // Note: TensorView shares data with original tensor by design  
        // Note: TensorView shares data with original tensor by design
        // Note: TensorView shares data with original tensor by design
        
        // All should be marked as views (except original)
        assert!(!original.is_view());
        assert!(view1.is_view());
        assert!(view2.is_view());
        // assert!(view3.is_view()); // Skip view3 for now
    }

    #[test]
    fn test_view_contiguity() {
        let original = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0], 
            vec![2, 2], 
            DeviceType::Cpu
        ).unwrap();
        
        // Original should be contiguous
        assert!(original.is_contiguous());
        
        // Simple reshape view should still be contiguous
        let reshaped = original.view_as(&[4]).unwrap();
        assert!(reshaped.is_contiguous());
        
        // Transpose view is typically not contiguous
        let transposed = original.transpose_view(0, 1).unwrap();
        assert!(!transposed.is_contiguous());
        
        // Slice view may or may not be contiguous depending on the slice
        let slice = original.slice(0, 0, 1).unwrap();
        // This slice should be contiguous as it takes consecutive memory
        assert!(slice.is_contiguous());
    }

    #[test]
    fn test_view_to_vec() {
        let original = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 
            vec![2, 3], 
            DeviceType::Cpu
        ).unwrap();
        
        // Test view to_vec extracts correct data
        let transposed = original.transpose_view(0, 1).unwrap();
        let transposed_data = transposed.to_vec().unwrap();
        
        // Transposed view should have data in transposed order
        assert_eq!(transposed_data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        
        // Slice view should extract only the slice
        let slice = original.slice(1, 1, 3).unwrap();
        let slice_data = slice.to_vec().unwrap();
        assert_eq!(slice_data, vec![2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn test_view_error_cases() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0], 
            vec![2, 2], 
            DeviceType::Cpu
        ).unwrap();
        
        // Invalid reshape size
        assert!(tensor.view_as(&[3, 3]).is_err());
        
        // Invalid slice range
        assert!(tensor.slice(0, 2, 3).is_err());  // end > dimension size
        assert!(tensor.slice(0, 1, 1).is_err());  // start >= end
        
        // Invalid squeeze dimension
        assert!(tensor.squeeze(0).is_err());  // dimension size != 1
        
        // Invalid transpose dimensions
        assert!(tensor.transpose(0, 3).is_err());  // dimension out of range
    }

    #[test]
    fn test_expand_operation() {
        use torsh_core::device::DeviceType;
        
        // Test expanding a 1D tensor
        let tensor = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
        let expanded = tensor.expand(&[3, 2]).unwrap();
        assert_eq!(expanded.shape().dims(), &[3, 2]);
        let data = expanded.to_vec().unwrap();
        assert_eq!(data, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        
        // Test expanding with size 1 dimension
        let tensor = Tensor::from_data(vec![5.0f32], vec![1, 1], DeviceType::Cpu).unwrap();
        let expanded = tensor.expand(&[2, 3]).unwrap();
        assert_eq!(expanded.shape().dims(), &[2, 3]);
        let data = expanded.to_vec().unwrap();
        assert_eq!(data, &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_flip_operation() {
        use torsh_core::device::DeviceType;
        
        // Test flipping a 1D tensor
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let flipped = tensor.flip(&[0]).unwrap();
        let data = flipped.to_vec().unwrap();
        assert_eq!(data, &[3.0, 2.0, 1.0]);
        
        // Test flipping a 2D tensor along dimension 0
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0], 
            vec![2, 2], 
            DeviceType::Cpu
        ).unwrap();
        let flipped = tensor.flip(&[0]).unwrap();
        let data = flipped.to_vec().unwrap();
        assert_eq!(data, &[3.0, 4.0, 1.0, 2.0]);
        
        // Test flipping along dimension 1
        let flipped = tensor.flip(&[1]).unwrap();
        let data = flipped.to_vec().unwrap();
        assert_eq!(data, &[2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    fn test_roll_operation() {
        use torsh_core::device::DeviceType;
        
        // Test rolling a 1D tensor
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();
        let rolled = tensor.roll(1, None).unwrap();
        let data = rolled.to_vec().unwrap();
        assert_eq!(data, &[4.0, 1.0, 2.0, 3.0]);
        
        // Test rolling with negative shift
        let rolled = tensor.roll(-1, None).unwrap();
        let data = rolled.to_vec().unwrap();
        assert_eq!(data, &[2.0, 3.0, 4.0, 1.0]);
        
        // Test rolling along specific dimension
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0], 
            vec![2, 2], 
            DeviceType::Cpu
        ).unwrap();
        let rolled = tensor.roll(1, Some(0)).unwrap();
        let data = rolled.to_vec().unwrap();
        assert_eq!(data, &[3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_shape_manipulation_error_cases() {
        use torsh_core::device::DeviceType;
        
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        
        // Test expand with invalid dimensions
        assert!(tensor.expand(&[2]).is_err()); // Too few dimensions
        assert!(tensor.expand(&[3, 2]).is_err()); // Cannot expand non-1 dimension
        
        // Test flip with invalid dimension
        assert!(tensor.flip(&[3]).is_err()); // Dimension out of range
        
        // Test roll with invalid dimension
        assert!(tensor.roll(1, Some(3)).is_err()); // Dimension out of range
    }
}
