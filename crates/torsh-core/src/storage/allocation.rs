//! Backend allocation framework for cross-platform memory management
//!
//! This module provides a unified interface for memory allocation across different
//! backends (CPU, GPU, etc.) with support for alignment, typed memory handles,
//! and allocation strategies.

use crate::device::Device;
use crate::dtype::TensorElement;
use std::marker::PhantomData;

/// Unified memory allocation trait for all backends
///
/// This trait defines a consistent interface for memory allocation across
/// different backend implementations (CPU, CUDA, Metal, WebGPU, etc.)
/// with support for alignment requirements and specialized allocation strategies.
pub trait BackendAllocator: Send + Sync + std::fmt::Debug + 'static {
    /// The device type this allocator works with
    type Device: Device;

    /// Error type for allocation failures
    type Error: std::error::Error + Send + Sync + 'static;

    /// Allocate raw memory for the given number of bytes
    ///
    /// # Arguments
    /// * `device` - The device to allocate on
    /// * `size_bytes` - Number of bytes to allocate
    /// * `alignment` - Required memory alignment in bytes (must be power of 2)
    ///
    /// # Returns
    /// A handle to the allocated memory or an error
    fn allocate_raw(
        &self,
        device: &Self::Device,
        size_bytes: usize,
        alignment: usize,
    ) -> std::result::Result<RawMemoryHandle, Self::Error>;

    /// Deallocate raw memory
    ///
    /// # Arguments
    /// * `handle` - The memory handle to deallocate
    ///
    /// # Safety
    /// The handle must be valid and not already deallocated
    unsafe fn deallocate_raw(
        &self,
        handle: RawMemoryHandle,
    ) -> std::result::Result<(), Self::Error>;

    /// Allocate typed memory for tensor elements
    ///
    /// # Arguments
    /// * `device` - The device to allocate on
    /// * `count` - Number of elements to allocate
    /// * `alignment` - Optional alignment requirement (defaults to element alignment)
    ///
    /// # Returns
    /// A typed memory handle or an error
    fn allocate_typed<T: TensorElement>(
        &self,
        device: &Self::Device,
        count: usize,
        alignment: Option<usize>,
    ) -> std::result::Result<TypedMemoryHandle<T>, Self::Error> {
        let element_size = std::mem::size_of::<T>();
        let size_bytes = count * element_size;
        let alignment = alignment.unwrap_or(std::mem::align_of::<T>());

        let raw_handle = self.allocate_raw(device, size_bytes, alignment)?;
        Ok(TypedMemoryHandle::new(raw_handle, count))
    }

    /// Deallocate typed memory
    ///
    /// # Arguments
    /// * `handle` - The typed memory handle to deallocate
    ///
    /// # Safety
    /// The handle must be valid and not already deallocated
    unsafe fn deallocate_typed<T: TensorElement>(
        &self,
        handle: TypedMemoryHandle<T>,
    ) -> std::result::Result<(), Self::Error> {
        self.deallocate_raw(handle.into_raw())
    }

    /// Get memory information for the device
    fn memory_info(
        &self,
        device: &Self::Device,
    ) -> std::result::Result<crate::storage::memory_info::MemoryInfo, Self::Error>;

    /// Check if a specific alignment is supported
    fn supports_alignment(&self, alignment: usize) -> bool {
        alignment > 0 && alignment.is_power_of_two()
    }

    /// Get the preferred alignment for optimal performance
    fn preferred_alignment(&self) -> usize {
        64 // Common cache line size
    }

    /// Set memory allocation strategy
    fn set_strategy(
        &mut self,
        strategy: crate::storage::memory_info::AllocationStrategy,
    ) -> std::result::Result<(), Self::Error>;

    /// Get current memory allocation strategy
    fn strategy(&self) -> crate::storage::memory_info::AllocationStrategy;

    /// Allocate with specific allocation strategy (one-off override)
    fn allocate_with_strategy(
        &self,
        device: &Self::Device,
        size_bytes: usize,
        alignment: usize,
        strategy: crate::storage::memory_info::AllocationStrategy,
    ) -> std::result::Result<RawMemoryHandle, Self::Error> {
        // Default implementation ignores strategy - backends can override
        let _ = strategy;
        self.allocate_raw(device, size_bytes, alignment)
    }

    /// Batch allocate multiple memory blocks efficiently
    fn batch_allocate(
        &self,
        device: &Self::Device,
        requests: &[AllocationRequest],
    ) -> std::result::Result<Vec<RawMemoryHandle>, Self::Error> {
        // Default implementation: allocate individually
        let mut handles = Vec::with_capacity(requests.len());
        for request in requests {
            let handle = self.allocate_raw(device, request.size_bytes, request.alignment)?;
            handles.push(handle);
        }
        Ok(handles)
    }

    /// Batch deallocate multiple memory blocks efficiently
    unsafe fn batch_deallocate(
        &self,
        handles: Vec<RawMemoryHandle>,
    ) -> std::result::Result<(), Self::Error> {
        // Default implementation: deallocate individually
        for handle in handles {
            self.deallocate_raw(handle)?;
        }
        Ok(())
    }
}

/// Request for memory allocation
#[derive(Debug, Clone)]
pub struct AllocationRequest {
    pub size_bytes: usize,
    pub alignment: usize,
    pub strategy: Option<crate::storage::memory_info::AllocationStrategy>,
}

impl AllocationRequest {
    pub fn new(size_bytes: usize, alignment: usize) -> Self {
        Self {
            size_bytes,
            alignment,
            strategy: None,
        }
    }

    pub fn with_strategy(
        mut self,
        strategy: crate::storage::memory_info::AllocationStrategy,
    ) -> Self {
        self.strategy = Some(strategy);
        self
    }
}

/// Raw memory handle that can work across different backends
#[derive(Debug)]
pub struct RawMemoryHandle {
    /// Opaque pointer to the memory (backend-specific)
    pub ptr: *mut u8,
    /// Size of the allocation in bytes
    pub size_bytes: usize,
    /// Alignment used for this allocation
    pub alignment: usize,
    /// Backend-specific metadata
    pub backend_data: Box<dyn std::any::Any + Send + Sync>,
}

unsafe impl Send for RawMemoryHandle {}
unsafe impl Sync for RawMemoryHandle {}

impl RawMemoryHandle {
    /// Create a new raw memory handle
    pub fn new(
        ptr: *mut u8,
        size_bytes: usize,
        alignment: usize,
        backend_data: Box<dyn std::any::Any + Send + Sync>,
    ) -> Self {
        Self {
            ptr,
            size_bytes,
            alignment,
            backend_data,
        }
    }

    /// Create a raw memory handle with default backend data
    pub fn simple(ptr: *mut u8, size_bytes: usize, alignment: usize) -> Self {
        Self::new(ptr, size_bytes, alignment, Box::new(()))
    }

    /// Get the pointer as a specific type
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and properly aligned for T
    pub unsafe fn as_ptr<T>(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Get the number of elements that can fit in this allocation
    pub fn element_capacity<T>(&self) -> usize {
        self.size_bytes / std::mem::size_of::<T>()
    }

    /// Check if the allocation is properly aligned for type T
    pub fn is_aligned_for<T>(&self) -> bool {
        let type_align = std::mem::align_of::<T>();
        self.alignment >= type_align && (self.ptr as usize).is_multiple_of(type_align)
    }

    /// Get the end pointer (one past the last byte)
    pub fn end_ptr(&self) -> *mut u8 {
        unsafe { self.ptr.add(self.size_bytes) }
    }

    /// Check if this handle contains a given pointer
    pub fn contains_ptr(&self, ptr: *const u8) -> bool {
        let start = self.ptr as usize;
        let end = start + self.size_bytes;
        let ptr_addr = ptr as usize;
        ptr_addr >= start && ptr_addr < end
    }

    /// Get a slice view of the memory
    ///
    /// # Safety
    /// The memory must be valid and properly initialized
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        let count = self.element_capacity::<T>();
        std::slice::from_raw_parts(self.as_ptr::<T>(), count)
    }

    /// Get a mutable slice view of the memory
    ///
    /// # Safety
    /// The memory must be valid and no other references must exist
    pub unsafe fn as_mut_slice<T>(&mut self) -> &mut [T] {
        let count = self.element_capacity::<T>();
        std::slice::from_raw_parts_mut(self.as_ptr::<T>(), count)
    }

    /// Split this handle at the given byte offset
    ///
    /// # Safety
    /// The offset must be within bounds and properly aligned
    pub unsafe fn split_at(mut self, offset: usize) -> (RawMemoryHandle, RawMemoryHandle) {
        assert!(offset <= self.size_bytes, "Split offset out of bounds");

        let first_size = offset;
        let second_size = self.size_bytes - offset;
        let second_ptr = self.ptr.add(offset);

        let first = RawMemoryHandle::new(
            self.ptr,
            first_size,
            self.alignment,
            Box::new("split_first"),
        );

        let second = RawMemoryHandle::new(
            second_ptr,
            second_size,
            self.alignment,
            Box::new("split_second"),
        );

        // Prevent double-free of original handle
        self.backend_data = Box::new("consumed");

        (first, second)
    }
}

/// Typed memory handle for specific element types
#[derive(Debug)]
pub struct TypedMemoryHandle<T: TensorElement> {
    raw: RawMemoryHandle,
    element_count: usize,
    _phantom: PhantomData<T>,
}

impl<T: TensorElement> TypedMemoryHandle<T> {
    /// Create a new typed memory handle from a raw handle
    pub fn new(raw: RawMemoryHandle, element_count: usize) -> Self {
        Self {
            raw,
            element_count,
            _phantom: PhantomData,
        }
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.element_count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.element_count == 0
    }

    /// Get the raw pointer to the data
    ///
    /// # Safety
    /// The caller must ensure the pointer is used correctly
    pub unsafe fn as_ptr(&self) -> *mut T {
        self.raw.as_ptr::<T>()
    }

    /// Get the raw memory handle
    pub fn raw(&self) -> &RawMemoryHandle {
        &self.raw
    }

    /// Convert into raw memory handle
    pub fn into_raw(self) -> RawMemoryHandle {
        self.raw
    }

    /// Create a slice view of the memory
    ///
    /// # Safety
    /// The caller must ensure the memory is valid and properly initialized
    pub unsafe fn as_slice(&self) -> &[T] {
        std::slice::from_raw_parts(self.as_ptr(), self.element_count)
    }

    /// Create a mutable slice view of the memory
    ///
    /// # Safety
    /// The caller must ensure the memory is valid and no other references exist
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.as_ptr(), self.element_count)
    }

    /// Get element at index
    ///
    /// # Safety
    /// Index must be within bounds and memory must be initialized
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        &*self.as_ptr().add(index)
    }

    /// Get mutable element at index
    ///
    /// # Safety
    /// Index must be within bounds and memory must be initialized
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        &mut *self.as_ptr().add(index)
    }

    /// Split this typed handle at the given element index
    ///
    /// # Safety
    /// The index must be within bounds
    pub unsafe fn split_at(self, at: usize) -> (TypedMemoryHandle<T>, TypedMemoryHandle<T>) {
        assert!(at <= self.element_count, "Split index out of bounds");

        let byte_offset = at * std::mem::size_of::<T>();
        let (raw_first, raw_second) = self.raw.split_at(byte_offset);

        let first = TypedMemoryHandle::new(raw_first, at);
        let second = TypedMemoryHandle::new(raw_second, self.element_count - at);

        (first, second)
    }

    /// Create a sub-handle for a range of elements
    pub fn slice(&self, start: usize, count: usize) -> Result<TypedMemoryHandle<T>, &'static str> {
        if start + count > self.element_count {
            return Err("Slice range out of bounds");
        }

        let byte_offset = start * std::mem::size_of::<T>();
        let byte_size = count * std::mem::size_of::<T>();

        unsafe {
            let new_ptr = self.raw.ptr.add(byte_offset);
            let raw_handle =
                RawMemoryHandle::new(new_ptr, byte_size, self.raw.alignment, Box::new("slice"));
            Ok(TypedMemoryHandle::new(raw_handle, count))
        }
    }

    /// Fill the memory with a specific value
    ///
    /// # Safety
    /// The memory must be valid and writable
    pub unsafe fn fill(&mut self, value: T) {
        let slice = self.as_mut_slice();
        slice.fill(value);
    }

    /// Copy data from another typed handle
    ///
    /// # Safety
    /// Both handles must be valid and non-overlapping
    pub unsafe fn copy_from(&mut self, src: &TypedMemoryHandle<T>) {
        let src_slice = src.as_slice();
        let dst_slice = self.as_mut_slice();
        let copy_count = src_slice.len().min(dst_slice.len());

        std::ptr::copy_nonoverlapping(src_slice.as_ptr(), dst_slice.as_mut_ptr(), copy_count);
    }

    /// Clone data into a new Vec
    ///
    /// # Safety
    /// The memory must be valid and properly initialized
    pub unsafe fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    /// Get memory statistics for this handle
    pub fn memory_stats(&self) -> TypedMemoryStats {
        TypedMemoryStats {
            element_count: self.element_count,
            element_size: std::mem::size_of::<T>(),
            total_bytes: self.element_count * std::mem::size_of::<T>(),
            alignment: self.raw.alignment,
            type_name: std::any::type_name::<T>(),
        }
    }
}

/// Statistics for typed memory handles
#[derive(Debug, Clone)]
pub struct TypedMemoryStats {
    pub element_count: usize,
    pub element_size: usize,
    pub total_bytes: usize,
    pub alignment: usize,
    pub type_name: &'static str,
}

impl std::fmt::Display for TypedMemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TypedMemory<{}>({} elements, {} bytes, align={})",
            self.type_name, self.element_count, self.total_bytes, self.alignment
        )
    }
}

/// Utilities for working with memory handles
pub mod utils {
    use super::*;

    /// Check if two raw handles overlap
    pub fn handles_overlap(a: &RawMemoryHandle, b: &RawMemoryHandle) -> bool {
        let a_start = a.ptr as usize;
        let a_end = a_start + a.size_bytes;
        let b_start = b.ptr as usize;
        let b_end = b_start + b.size_bytes;

        !(a_end <= b_start || b_end <= a_start)
    }

    /// Calculate total memory usage of a collection of handles
    pub fn total_memory_usage(handles: &[RawMemoryHandle]) -> usize {
        handles.iter().map(|h| h.size_bytes).sum()
    }

    /// Find the handle with maximum alignment
    pub fn max_alignment(handles: &[RawMemoryHandle]) -> usize {
        handles.iter().map(|h| h.alignment).max().unwrap_or(1)
    }

    /// Check if all handles in a collection are properly aligned for type T
    pub fn all_aligned_for<T>(handles: &[RawMemoryHandle]) -> bool {
        handles.iter().all(|h| h.is_aligned_for::<T>())
    }

    /// Merge adjacent handles into a single handle (unsafe - for advanced use)
    pub unsafe fn merge_adjacent_handles(
        handles: Vec<RawMemoryHandle>,
    ) -> Result<RawMemoryHandle, Vec<RawMemoryHandle>> {
        if handles.is_empty() {
            return Err(handles);
        }

        if handles.len() == 1 {
            return Ok(handles.into_iter().next().unwrap());
        }

        // Check if all handles are adjacent and have the same alignment
        let first = &handles[0];
        let mut total_size = first.size_bytes;
        let mut current_end = first.end_ptr();

        for handle in &handles[1..] {
            if handle.ptr != current_end || handle.alignment != first.alignment {
                return Err(handles); // Not adjacent or incompatible alignment
            }
            total_size += handle.size_bytes;
            current_end = handle.end_ptr();
        }

        Ok(RawMemoryHandle::new(
            first.ptr,
            total_size,
            first.alignment,
            Box::new("merged"),
        ))
    }
}
