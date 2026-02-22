//! Memory operations for backend allocators
//!
//! This module defines traits for memory copy operations and asynchronous memory
//! management that extend the basic allocation functionality.

use super::allocation::BackendAllocator;
use crate::storage::allocation::RawMemoryHandle;

/// Trait for backends that support memory copy operations
///
/// This trait extends BackendAllocator with efficient memory transfer operations,
/// including same-device copies, cross-device transfers, and memory filling.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::storage::{BackendMemoryCopy, RawMemoryHandle};
///
/// // Copy data between memory locations
/// allocator.copy_memory(&src_handle, &dst_handle, 1024)?;
///
/// // Fill memory with a pattern
/// let pattern = [0xAA, 0xBB, 0xCC, 0xDD];
/// allocator.fill_memory(&handle, &pattern, 1024)?;
/// ```
pub trait BackendMemoryCopy: BackendAllocator {
    /// Copy memory from one location to another on the same device
    ///
    /// # Arguments
    /// * `src` - Source memory handle
    /// * `dst` - Destination memory handle
    /// * `size_bytes` - Number of bytes to copy
    ///
    /// # Returns
    /// Success or error from the copy operation
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - Both handles are valid and allocated
    /// - The memory regions do not overlap (use overlapping-safe copy if needed)
    /// - The destination has sufficient capacity
    fn copy_memory(
        &self,
        src: &RawMemoryHandle,
        dst: &RawMemoryHandle,
        size_bytes: usize,
    ) -> std::result::Result<(), Self::Error>;

    /// Copy memory between different devices (if supported)
    ///
    /// # Arguments
    /// * `src` - Source memory handle
    /// * `src_device` - Source device
    /// * `dst` - Destination memory handle
    /// * `dst_device` - Destination device
    /// * `size_bytes` - Number of bytes to copy
    ///
    /// # Returns
    /// Success or error from the cross-device copy operation
    ///
    /// # Notes
    /// This operation may be significantly slower than same-device copies
    /// and may require synchronization between devices.
    fn copy_memory_cross_device(
        &self,
        src: &RawMemoryHandle,
        src_device: &Self::Device,
        dst: &RawMemoryHandle,
        dst_device: &Self::Device,
        size_bytes: usize,
    ) -> std::result::Result<(), Self::Error>;

    /// Fill memory with a specific pattern
    ///
    /// # Arguments
    /// * `handle` - Memory handle to fill
    /// * `pattern` - Byte pattern to repeat
    /// * `size_bytes` - Number of bytes to fill
    ///
    /// # Returns
    /// Success or error from the fill operation
    ///
    /// # Examples
    /// ```ignore
    /// // Fill with zeros
    /// allocator.fill_memory(&handle, &[0], 1024)?;
    ///
    /// // Fill with a 4-byte pattern
    /// allocator.fill_memory(&handle, &[0xDE, 0xAD, 0xBE, 0xEF], 1024)?;
    /// ```
    fn fill_memory(
        &self,
        handle: &RawMemoryHandle,
        pattern: &[u8],
        size_bytes: usize,
    ) -> std::result::Result<(), Self::Error>;

    /// Copy memory with overlap safety (default implementation using memmove semantics)
    ///
    /// # Arguments
    /// * `src` - Source memory handle
    /// * `dst` - Destination memory handle
    /// * `size_bytes` - Number of bytes to copy
    ///
    /// # Returns
    /// Success or error from the copy operation
    ///
    /// # Notes
    /// This method handles overlapping memory regions safely, but may be slower
    /// than copy_memory for non-overlapping regions.
    fn copy_memory_overlapping(
        &self,
        src: &RawMemoryHandle,
        dst: &RawMemoryHandle,
        size_bytes: usize,
    ) -> std::result::Result<(), Self::Error> {
        // Default implementation: check for overlap and use appropriate copy
        if utils::handles_overlap(src, dst) {
            unsafe {
                std::ptr::copy(src.ptr, dst.ptr, size_bytes);
            }
            Ok(())
        } else {
            self.copy_memory(src, dst, size_bytes)
        }
    }

    /// Batch copy multiple memory regions efficiently
    ///
    /// # Arguments
    /// * `operations` - Vector of copy operations to perform
    ///
    /// # Returns
    /// Success or error from the batch copy operations
    fn batch_copy(&self, operations: &[CopyOperation]) -> std::result::Result<(), Self::Error> {
        // Default implementation: copy individually
        for op in operations {
            match op {
                CopyOperation::SameDevice { src, dst, size } => {
                    self.copy_memory(src, dst, *size)?;
                }
                CopyOperation::CrossDevice {
                    src,
                    src_device: _,
                    dst,
                    dst_device: _,
                    size,
                } => {
                    // For now, fall back to regular copy for cross-device operations
                    // since we can't safely downcast the device references
                    self.copy_memory(src, dst, *size)?;
                }
                CopyOperation::Fill { dst, pattern, size } => {
                    self.fill_memory(dst, pattern, *size)?;
                }
            }
        }
        Ok(())
    }

    /// Check if cross-device copy is supported between two device types
    fn supports_cross_device_copy(
        &self,
        src_device: &Self::Device,
        dst_device: &Self::Device,
    ) -> bool {
        // Default implementation: only supports same device
        std::ptr::eq(src_device, dst_device)
    }

    /// Get the optimal block size for memory operations on this backend
    fn optimal_copy_block_size(&self) -> usize {
        64 * 1024 // 64KB default
    }
}

/// Trait for backends that support asynchronous memory operations
///
/// This trait provides async/await support for memory allocation and deallocation,
/// which can be beneficial for backends that perform network operations or
/// complex resource management.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::storage::BackendAsyncMemory;
///
/// // Asynchronous allocation
/// let handle = allocator.allocate_async(&device, 1024, 8).await?;
///
/// // Asynchronous deallocation
/// allocator.deallocate_async(handle).await?;
/// ```
pub trait BackendAsyncMemory: BackendAllocator {
    /// Asynchronous allocation
    ///
    /// # Arguments
    /// * `device` - The device to allocate on
    /// * `size_bytes` - Number of bytes to allocate
    /// * `alignment` - Required memory alignment in bytes
    ///
    /// # Returns
    /// A future that resolves to a memory handle or error
    fn allocate_async(
        &self,
        device: &Self::Device,
        size_bytes: usize,
        alignment: usize,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = std::result::Result<RawMemoryHandle, Self::Error>>
                + Send,
        >,
    >;

    /// Asynchronous deallocation
    ///
    /// # Arguments
    /// * `handle` - The memory handle to deallocate
    ///
    /// # Returns
    /// A future that resolves to success or error
    ///
    /// # Safety
    /// The handle must be valid and not already deallocated
    fn deallocate_async(
        &self,
        handle: RawMemoryHandle,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = std::result::Result<(), Self::Error>> + Send>,
    >;

    /// Asynchronous batch allocation
    ///
    /// # Arguments
    /// * `device` - The device to allocate on
    /// * `requests` - Vector of allocation requests
    ///
    /// # Returns
    /// A future that resolves to a vector of memory handles or error
    fn batch_allocate_async<'a>(
        &'a self,
        device: &'a Self::Device,
        requests: &[super::allocation::AllocationRequest],
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = std::result::Result<Vec<RawMemoryHandle>, Self::Error>>
                + Send
                + 'a,
        >,
    > {
        // Default implementation: allocate individually
        let requests = requests.to_vec();
        Box::pin(async move {
            let mut handles = Vec::with_capacity(requests.len());
            for request in requests {
                let handle = self
                    .allocate_async(device, request.size_bytes, request.alignment)
                    .await?;
                handles.push(handle);
            }
            Ok(handles)
        })
    }

    /// Asynchronous batch deallocation
    ///
    /// # Arguments
    /// * `handles` - Vector of memory handles to deallocate
    ///
    /// # Returns
    /// A future that resolves to success or error
    fn batch_deallocate_async(
        &self,
        handles: Vec<RawMemoryHandle>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = std::result::Result<(), Self::Error>> + Send + '_>,
    > {
        // Default implementation: deallocate individually
        Box::pin(async move {
            for handle in handles {
                self.deallocate_async(handle).await?;
            }
            Ok(())
        })
    }

    /// Check if async operations are actually performed asynchronously
    ///
    /// Some backends may implement async traits but perform operations synchronously.
    /// This method allows clients to optimize their usage patterns accordingly.
    fn is_truly_async(&self) -> bool {
        true // Assume async by default
    }

    /// Get the recommended concurrency level for async operations
    fn recommended_concurrency(&self) -> usize {
        num_cpus::get()
    }
}

/// Memory copy operation variants for batch operations
#[derive(Clone)]
pub enum CopyOperation<'a> {
    /// Copy within the same device
    SameDevice {
        src: &'a RawMemoryHandle,
        dst: &'a RawMemoryHandle,
        size: usize,
    },
    /// Copy between different devices
    CrossDevice {
        src: &'a RawMemoryHandle,
        src_device: &'a dyn crate::device::Device,
        dst: &'a RawMemoryHandle,
        dst_device: &'a dyn crate::device::Device,
        size: usize,
    },
    /// Fill memory with pattern
    Fill {
        dst: &'a RawMemoryHandle,
        pattern: &'a [u8],
        size: usize,
    },
}

impl<'a> std::fmt::Debug for CopyOperation<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CopyOperation::SameDevice { src, dst, size } => f
                .debug_struct("SameDevice")
                .field("src_ptr", &src.ptr)
                .field("dst_ptr", &dst.ptr)
                .field("size", size)
                .finish(),
            CopyOperation::CrossDevice { src, dst, size, .. } => f
                .debug_struct("CrossDevice")
                .field("src_ptr", &src.ptr)
                .field("dst_ptr", &dst.ptr)
                .field("size", size)
                .finish(),
            CopyOperation::Fill { dst, pattern, size } => f
                .debug_struct("Fill")
                .field("dst_ptr", &dst.ptr)
                .field("pattern_len", &pattern.len())
                .field("size", size)
                .finish(),
        }
    }
}

/// Statistics for memory operations
#[derive(Debug, Clone)]
pub struct MemoryOperationStats {
    /// Total number of copy operations performed
    pub copy_operations: u64,
    /// Total bytes copied
    pub bytes_copied: u64,
    /// Total number of fill operations performed
    pub fill_operations: u64,
    /// Total bytes filled
    pub bytes_filled: u64,
    /// Total number of cross-device operations
    pub cross_device_operations: u64,
    /// Average operation latency in microseconds
    pub avg_latency_us: f64,
    /// Peak memory bandwidth achieved (bytes per second)
    pub peak_bandwidth: Option<u64>,
}

impl Default for MemoryOperationStats {
    fn default() -> Self {
        Self {
            copy_operations: 0,
            bytes_copied: 0,
            fill_operations: 0,
            bytes_filled: 0,
            cross_device_operations: 0,
            avg_latency_us: 0.0,
            peak_bandwidth: None,
        }
    }
}

impl MemoryOperationStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a copy operation
    pub fn record_copy(&mut self, bytes: u64, latency_us: f64) {
        self.copy_operations += 1;
        self.bytes_copied += bytes;
        self.update_latency(latency_us);
    }

    /// Record a fill operation
    pub fn record_fill(&mut self, bytes: u64, latency_us: f64) {
        self.fill_operations += 1;
        self.bytes_filled += bytes;
        self.update_latency(latency_us);
    }

    /// Record a cross-device operation
    pub fn record_cross_device(&mut self, bytes: u64, latency_us: f64) {
        self.cross_device_operations += 1;
        self.bytes_copied += bytes;
        self.update_latency(latency_us);
    }

    /// Update average latency calculation
    fn update_latency(&mut self, latency_us: f64) {
        let total_ops = self.copy_operations + self.fill_operations;
        if total_ops > 1 {
            self.avg_latency_us =
                (self.avg_latency_us * (total_ops - 1) as f64 + latency_us) / total_ops as f64;
        } else {
            self.avg_latency_us = latency_us;
        }
    }

    /// Calculate total operations
    pub fn total_operations(&self) -> u64 {
        self.copy_operations + self.fill_operations
    }

    /// Calculate total bytes processed
    pub fn total_bytes(&self) -> u64 {
        self.bytes_copied + self.bytes_filled
    }

    /// Calculate average bandwidth (bytes per second)
    pub fn avg_bandwidth(&self) -> Option<f64> {
        if self.avg_latency_us > 0.0 {
            Some(self.total_bytes() as f64 * 1_000_000.0 / self.avg_latency_us)
        } else {
            None
        }
    }
}

/// Utility functions for memory operations
pub mod utils {
    use super::*;

    /// Check if two memory handles overlap
    pub fn handles_overlap(a: &RawMemoryHandle, b: &RawMemoryHandle) -> bool {
        let a_start = a.ptr as usize;
        let a_end = a_start + a.size_bytes;
        let b_start = b.ptr as usize;
        let b_end = b_start + b.size_bytes;

        !(a_end <= b_start || b_end <= a_start)
    }

    /// Calculate optimal chunk size for large copy operations
    pub fn optimal_chunk_size(total_size: usize, base_chunk_size: usize) -> usize {
        if total_size < base_chunk_size {
            total_size
        } else {
            // Use chunk size that divides evenly or is close to base
            let num_chunks = total_size.div_ceil(base_chunk_size);
            total_size / num_chunks
        }
    }

    /// Create a fill pattern of a specific size from a smaller pattern
    pub fn expand_pattern(pattern: &[u8], target_size: usize) -> Vec<u8> {
        if pattern.is_empty() {
            return vec![0; target_size];
        }

        let mut result = Vec::with_capacity(target_size);
        let pattern_len = pattern.len();

        for i in 0..target_size {
            result.push(pattern[i % pattern_len]);
        }

        result
    }

    /// Validate that a memory operation is safe
    pub fn validate_copy_operation(
        src: &RawMemoryHandle,
        dst: &RawMemoryHandle,
        size: usize,
    ) -> Result<(), String> {
        if size == 0 {
            return Ok(()); // Zero-size copy is always valid
        }

        if size > src.size_bytes {
            return Err(format!(
                "Copy size {} exceeds source buffer size {}",
                size, src.size_bytes
            ));
        }

        if size > dst.size_bytes {
            return Err(format!(
                "Copy size {} exceeds destination buffer size {}",
                size, dst.size_bytes
            ));
        }

        if src.ptr.is_null() || dst.ptr.is_null() {
            return Err("Null pointer in copy operation".to_string());
        }

        Ok(())
    }

    /// Calculate memory bandwidth for an operation
    pub fn calculate_bandwidth(bytes: u64, duration_us: f64) -> f64 {
        if duration_us > 0.0 {
            bytes as f64 * 1_000_000.0 / duration_us // bytes per second
        } else {
            0.0
        }
    }

    /// Get human-readable bandwidth string
    pub fn format_bandwidth(bytes_per_second: f64) -> String {
        const UNITS: &[&str] = &["B/s", "KB/s", "MB/s", "GB/s", "TB/s"];

        let mut value = bytes_per_second;
        let mut unit_index = 0;

        while value >= 1024.0 && unit_index < UNITS.len() - 1 {
            value /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", value, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::allocation::RawMemoryHandle;

    #[test]
    fn test_handles_overlap() {
        let data1 = [1u8, 2, 3, 4];
        let data2 = [5u8, 6, 7, 8];

        let handle1 = RawMemoryHandle::simple(data1.as_ptr() as *mut u8, 4, 1);
        let handle2 = RawMemoryHandle::simple(data2.as_ptr() as *mut u8, 4, 1);

        // Non-overlapping handles
        assert!(!utils::handles_overlap(&handle1, &handle2));

        // Overlapping handles (same memory)
        let handle3 = RawMemoryHandle::simple(data1.as_ptr() as *mut u8, 4, 1);
        assert!(utils::handles_overlap(&handle1, &handle3));
    }

    #[test]
    fn test_optimal_chunk_size() {
        assert_eq!(utils::optimal_chunk_size(100, 1000), 100);
        assert_eq!(utils::optimal_chunk_size(1000, 100), 100);
        assert_eq!(utils::optimal_chunk_size(1500, 100), 100); // (1500+99)/100=15 chunks, 1500/15=100
    }

    #[test]
    fn test_expand_pattern() {
        assert_eq!(utils::expand_pattern(&[1, 2], 6), vec![1, 2, 1, 2, 1, 2]);
        assert_eq!(utils::expand_pattern(&[5], 4), vec![5, 5, 5, 5]);
        assert_eq!(utils::expand_pattern(&[], 3), vec![0, 0, 0]);
    }

    #[test]
    fn test_memory_operation_stats() {
        let mut stats = MemoryOperationStats::new();

        stats.record_copy(1024, 100.0);
        stats.record_fill(512, 50.0);

        assert_eq!(stats.total_operations(), 2);
        assert_eq!(stats.total_bytes(), 1536);
        assert_eq!(stats.avg_latency_us, 75.0); // (100 + 50) / 2
    }

    #[test]
    fn test_calculate_bandwidth() {
        let bandwidth = utils::calculate_bandwidth(1024, 1000.0); // 1KB in 1ms
        assert_eq!(bandwidth, 1_024_000.0); // 1MB/s
    }

    #[test]
    fn test_format_bandwidth() {
        assert_eq!(utils::format_bandwidth(1024.0), "1.00 KB/s");
        assert_eq!(utils::format_bandwidth(1048576.0), "1.00 MB/s");
        assert_eq!(utils::format_bandwidth(1073741824.0), "1.00 GB/s");
    }

    #[test]
    fn test_validate_copy_operation() {
        let data1 = [1u8, 2, 3, 4];
        let data2 = [5u8, 6, 7, 8];

        let handle1 = RawMemoryHandle::simple(data1.as_ptr() as *mut u8, 4, 1);
        let handle2 = RawMemoryHandle::simple(data2.as_ptr() as *mut u8, 4, 1);

        // Valid operation
        assert!(utils::validate_copy_operation(&handle1, &handle2, 4).is_ok());

        // Invalid operation - size too large
        assert!(utils::validate_copy_operation(&handle1, &handle2, 8).is_err());

        // Zero-size operation
        assert!(utils::validate_copy_operation(&handle1, &handle2, 0).is_ok());
    }
}
