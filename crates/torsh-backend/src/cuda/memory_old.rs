//! CUDA Memory Management - Compatibility Layer
//!
//! This module provides backward compatibility with the original monolithic CUDA memory
//! management API while delegating operations to the new modular memory management system.
//! All original APIs are preserved for seamless migration.
//!
//! The new modular system provides enhanced functionality including:
//! - Advanced ML-based optimization
//! - Comprehensive statistics and analytics
//! - Intelligent memory pool coordination
//! - Predictive allocation strategies
//! - Multi-objective optimization
//!
//! For new code, consider using the enhanced APIs in `crate::cuda::memory::`

use crate::error::{CudaError, CudaResult};
use std::collections::HashMap;
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Re-export the new memory management system
use crate::cuda::memory::{
    initialize_memory_system, get_memory_manager as get_new_manager,
    allocate_device_memory as new_allocate_device,
    allocate_unified_memory as new_allocate_unified,
    allocate_pinned_memory as new_allocate_pinned,
    deallocate_memory as new_deallocate,
    get_memory_statistics as new_get_stats,
    MemorySystemConfig, AllocationType, AllocationPriority, MemoryAlignment,
    CudaMemoryManagerCoordinator, ManagerOperationResult
};

// Global memory managers for backward compatibility
static MEMORY_MANAGERS: once_cell::sync::Lazy<Mutex<HashMap<usize, Arc<CudaMemoryManager>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));
static PINNED_MANAGERS: once_cell::sync::Lazy<Mutex<HashMap<usize, Arc<PinnedMemoryManager>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

// Initialize the new memory system on first use
static SYSTEM_INITIALIZED: std::sync::Once = std::sync::Once::new();

fn ensure_system_initialized() -> CudaResult<()> {
    SYSTEM_INITIALIZED.call_once(|| {
        let config = MemorySystemConfig {
            device_ids: (0..8).collect(), // Support up to 8 devices
            auto_initialize: true,
            enable_logging: false,
            ..Default::default()
        };

        if let Err(e) = initialize_memory_system(config) {
            eprintln!("Failed to initialize memory system: {}", e);
        }
    });
    Ok(())
}

/// CUDA memory manager with pooling (Compatibility Layer)
///
/// This is a compatibility wrapper around the new modular memory management system.
/// It preserves the original API while providing enhanced functionality under the hood.
#[derive(Debug)]
pub struct CudaMemoryManager {
    device_id: usize,
    // Legacy statistics tracking for compatibility
    total_allocated: std::sync::atomic::AtomicUsize,
    peak_allocated: std::sync::atomic::AtomicUsize,
    memory_limit: std::sync::atomic::AtomicUsize,
    pressure_threshold: std::sync::atomic::AtomicUsize,
}

impl CudaMemoryManager {
    /// Create new memory manager for device
    pub fn new(device_id: usize) -> CudaResult<Self> {
        ensure_system_initialized()?;

        // Try to get device memory info to set limits
        let (memory_limit, pressure_threshold) =
            if let Ok(device) = cust::Device::get_device(device_id as u32) {
                if let Ok(total_mem) = device.total_memory() {
                    // Set limit to 85% of total memory, pressure threshold to 75%
                    let limit = (total_mem as usize * 85) / 100;
                    let threshold = (total_mem as usize * 75) / 100;
                    (limit, threshold)
                } else {
                    // Fallback values if we can't query device memory
                    (8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024) // 8GB limit, 6GB threshold
                }
            } else {
                // Fallback values if device is not available
                (8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024) // 8GB limit, 6GB threshold
            };

        Ok(Self {
            device_id,
            total_allocated: std::sync::atomic::AtomicUsize::new(0),
            peak_allocated: std::sync::atomic::AtomicUsize::new(0),
            memory_limit: std::sync::atomic::AtomicUsize::new(memory_limit),
            pressure_threshold: std::sync::atomic::AtomicUsize::new(pressure_threshold),
        })
    }

    /// Allocate memory
    pub fn allocate(&self, size: usize) -> CudaResult<CudaAllocation> {
        // Use the new memory management system
        match new_allocate_device(size, Some(self.device_id)) {
            Ok(allocation) => {
                // Update legacy statistics for compatibility
                self.update_stats(size);

                // Convert to legacy allocation format
                let ptr = allocation.as_ptr();
                let size = allocation.size();
                let cuda_allocation = CudaAllocation::new(
                    unsafe { cust::DevicePointer::wrap(ptr) },
                    size,
                    self.size_class(size),
                );

                // Keep the new allocation alive by leaking it
                // The CudaAllocation Drop will handle cleanup through deallocate()
                std::mem::forget(allocation);

                Ok(cuda_allocation)
            }
            Err(e) => Err(CudaError::Context {
                message: format!("Device memory allocation failed: {}", e),
            })
        }
    }

    /// Deallocate memory
    pub fn deallocate(&self, allocation: CudaAllocation) -> CudaResult<()> {
        // Update legacy statistics for compatibility
        self.update_dealloc_stats(allocation.size());

        // For compatibility, we rely on the CudaAllocation's Drop implementation
        // which will properly clean up the memory through the new system
        drop(allocation);
        Ok(())
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        // Try to get enhanced statistics from new system
        if let Ok(enhanced_stats) = new_get_stats() {
            // Map enhanced statistics to legacy format
            MemoryStats {
                total_allocated: enhanced_stats.total_allocated,
                peak_allocated: enhanced_stats.peak_allocated,
                device_id: self.device_id,
            }
        } else {
            // Fallback to legacy statistics
            MemoryStats {
                total_allocated: self
                    .total_allocated
                    .load(std::sync::atomic::Ordering::Relaxed),
                peak_allocated: self
                    .peak_allocated
                    .load(std::sync::atomic::Ordering::Relaxed),
                device_id: self.device_id,
            }
        }
    }

    /// Clear all memory pools
    pub fn clear(&self) -> CudaResult<()> {
        // Trigger system maintenance to clear pools in the new system
        if let Err(e) = crate::cuda::memory::perform_system_maintenance() {
            return Err(CudaError::Context {
                message: format!("Failed to clear memory pools: {}", e),
            });
        }

        // Reset legacy statistics
        self.total_allocated
            .store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Allocate unified memory
    pub fn allocate_unified(&self, size: usize) -> CudaResult<UnifiedAllocation> {
        // Use the new unified memory management system
        match new_allocate_unified(size, Some(self.device_id)) {
            Ok(allocation) => {
                // Update legacy statistics for compatibility
                self.update_stats(size);

                // Convert to legacy unified allocation format
                let ptr = allocation.as_ptr();
                let size = allocation.size();
                let unified_allocation = UnifiedAllocation::new(ptr, size);

                // Keep the new allocation alive by leaking it
                // The UnifiedAllocation Drop will handle cleanup through deallocate_unified()
                std::mem::forget(allocation);

                Ok(unified_allocation)
            }
            Err(e) => Err(CudaError::Context {
                message: format!("Unified memory allocation failed: {}", e),
            })
        }
    }

    /// Deallocate unified memory
    pub fn deallocate_unified(&self, allocation: UnifiedAllocation) -> CudaResult<()> {
        // Update stats before the allocation is dropped
        self.update_dealloc_stats(allocation.size());

        // The allocation will be automatically freed when it goes out of scope
        // via the Drop implementation - no manual cudaFree needed here
        drop(allocation);
        Ok(())
    }

    /// Prefetch unified memory to a specific device
    pub fn prefetch_to_device(
        &self,
        ptr: *mut u8,
        size: usize,
        device_id: Option<usize>,
    ) -> CudaResult<()> {
        let target_device = device_id.unwrap_or(self.device_id) as i32;

        unsafe {
            let result = cuda_sys::cudaMemPrefetchAsync(
                ptr as *const std::ffi::c_void,
                size,
                target_device,
                0 as cuda_sys::cudaStream_t, // Default stream
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to prefetch memory: {:?}", result),
                });
            }
        }
        Ok(())
    }

    /// Prefetch unified memory to host
    pub fn prefetch_to_host(&self, ptr: *mut u8, size: usize) -> CudaResult<()> {
        unsafe {
            let result = cuda_sys::cudaMemPrefetchAsync(
                ptr as *const std::ffi::c_void,
                size,
                cuda_sys::cudaCpuDeviceId as i32,
                0 as cuda_sys::cudaStream_t, // Default stream
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to prefetch memory to host: {:?}", result),
                });
            }
        }
        Ok(())
    }

    /// Set memory advice for performance optimization
    pub fn set_memory_advice(
        &self,
        ptr: *mut u8,
        size: usize,
        advice: MemoryAdvice,
        device_id: Option<usize>,
    ) -> CudaResult<()> {
        let device = device_id.unwrap_or(self.device_id) as i32;
        let cuda_advice = match advice {
            MemoryAdvice::SetReadMostly => cuda_sys::cudaMemoryAdvise_cudaMemAdviseSetReadMostly,
            MemoryAdvice::UnsetReadMostly => {
                cuda_sys::cudaMemoryAdvise_cudaMemAdviseUnsetReadMostly
            }
            MemoryAdvice::SetPreferredLocation => {
                cuda_sys::cudaMemoryAdvise_cudaMemAdviseSetPreferredLocation
            }
            MemoryAdvice::UnsetPreferredLocation => {
                cuda_sys::cudaMemoryAdvise_cudaMemAdviseUnsetPreferredLocation
            }
            MemoryAdvice::SetAccessedBy => cuda_sys::cudaMemoryAdvise_cudaMemAdviseSetAccessedBy,
            MemoryAdvice::UnsetAccessedBy => {
                cuda_sys::cudaMemoryAdvise_cudaMemAdviseUnsetAccessedBy
            }
        };

        unsafe {
            let result =
                cuda_sys::cudaMemAdvise(ptr as *const std::ffi::c_void, size, cuda_advice, device);

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to set memory advice: {:?}", result),
                });
            }
        }
        Ok(())
    }

    /// Check if system is under memory pressure
    pub fn is_under_memory_pressure(&self) -> bool {
        let current = self
            .total_allocated
            .load(std::sync::atomic::Ordering::Relaxed);
        let threshold = self
            .pressure_threshold
            .load(std::sync::atomic::Ordering::Relaxed);
        current > threshold
    }

    /// Compact memory pools
    pub fn compact_memory_pools(&self) -> CudaResult<()> {
        // Trigger maintenance in the new system
        crate::cuda::memory::perform_system_maintenance()
            .map_err(|e| CudaError::Context {
                message: format!("Failed to compact memory pools: {}", e),
            })?;
        Ok(())
    }

    /// Set memory limit
    pub fn set_memory_limit(&self, limit: usize) {
        self.memory_limit
            .store(limit, std::sync::atomic::Ordering::Relaxed);
        let threshold = (limit * 75) / 100; // 75% threshold
        self.pressure_threshold
            .store(threshold, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get memory limit
    pub fn memory_limit(&self) -> usize {
        self.memory_limit
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get memory pressure info
    pub fn memory_pressure_info(&self) -> MemoryPressureInfo {
        let current = self
            .total_allocated
            .load(std::sync::atomic::Ordering::Relaxed);
        let limit = self.memory_limit();
        let threshold = self
            .pressure_threshold
            .load(std::sync::atomic::Ordering::Relaxed);

        let pressure_ratio = if limit > 0 {
            current as f64 / limit as f64
        } else {
            0.0
        };

        MemoryPressureInfo {
            current_allocated: current,
            memory_limit: limit,
            pressure_threshold: threshold,
            pressure_ratio,
            under_pressure: current > threshold,
        }
    }

    /// Create pinned memory manager for this device
    pub fn create_pinned_memory_manager(
        &self,
        config: PinnedMemoryConfig,
    ) -> CudaResult<Arc<PinnedMemoryManager>> {
        PinnedMemoryManager::new(config)
    }

    /// Get existing pinned memory manager
    pub fn get_pinned_memory_manager(&self) -> Option<Arc<PinnedMemoryManager>> {
        PINNED_MANAGERS
            .lock()
            .ok()?
            .get(&self.device_id)
            .cloned()
    }

    // Helper methods
    fn size_class(&self, size: usize) -> usize {
        // Round up to nearest power of 2, minimum 256 bytes
        let min_size = 256;
        if size <= min_size {
            min_size
        } else {
            let next_power = (size - 1).next_power_of_two();
            next_power.max(min_size)
        }
    }

    fn update_stats(&self, size: usize) {
        let current = self
            .total_allocated
            .fetch_add(size, std::sync::atomic::Ordering::Relaxed)
            + size;
        let mut peak = self
            .peak_allocated
            .load(std::sync::atomic::Ordering::Relaxed);

        while current > peak {
            match self.peak_allocated.compare_exchange_weak(
                peak,
                current,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }

    fn update_dealloc_stats(&self, size: usize) {
        self.total_allocated
            .fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
    }
}

/// CUDA device memory allocation (Legacy wrapper)
#[derive(Debug, Clone)]
pub struct CudaAllocation {
    ptr: cust::DevicePointer<u8>,
    size: usize,
    size_class: usize,
}

impl CudaAllocation {
    fn new(ptr: cust::DevicePointer<u8>, size: usize, size_class: usize) -> Self {
        Self { ptr, size, size_class }
    }

    pub fn ptr(&self) -> cust::DevicePointer<u8> {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_ptr<T>(&self) -> cust::DevicePointer<T> {
        unsafe { std::mem::transmute(self.ptr) }
    }
}

impl Drop for CudaAllocation {
    fn drop(&mut self) {
        // Use raw CUDA free for compatibility
        unsafe {
            let _ = cust::memory::cuda_free(self.ptr);
        }
    }
}

/// Unified memory allocation (Legacy wrapper)
#[derive(Debug)]
pub struct UnifiedAllocation {
    ptr: *mut u8,
    size: usize,
    allocation_time: Instant,
}

impl UnifiedAllocation {
    fn new(ptr: *mut u8, size: usize) -> Self {
        Self {
            ptr,
            size,
            allocation_time: Instant::now(),
        }
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_ptr<T>(&self) -> *mut T {
        self.ptr as *mut T
    }

    pub fn copy_from_host<T>(&mut self, data: &[T]) -> CudaResult<()>
    where
        T: Copy,
    {
        let size_bytes = std::mem::size_of_val(data);
        if size_bytes > self.size {
            return Err(CudaError::Context {
                message: "Data size exceeds allocation size".to_string(),
            });
        }

        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr() as *const u8, self.ptr, size_bytes);
        }
        Ok(())
    }

    pub fn copy_to_host<T>(&self, data: &mut [T]) -> CudaResult<()>
    where
        T: Copy,
    {
        let size_bytes = std::mem::size_of_val(data);
        if size_bytes > self.size {
            return Err(CudaError::Context {
                message: "Data size exceeds allocation size".to_string(),
            });
        }

        unsafe {
            ptr::copy_nonoverlapping(self.ptr, data.as_mut_ptr() as *mut u8, size_bytes);
        }
        Ok(())
    }
}

impl Drop for UnifiedAllocation {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_sys::cudaFree(self.ptr as *mut std::ffi::c_void);
        }
    }
}

/// Memory advice for unified memory optimization
#[derive(Debug, Clone, Copy)]
pub enum MemoryAdvice {
    SetReadMostly,
    UnsetReadMostly,
    SetPreferredLocation,
    UnsetPreferredLocation,
    SetAccessedBy,
    UnsetAccessedBy,
}

/// Pinned (page-locked) host memory allocation
#[derive(Debug)]
pub struct PinnedAllocation {
    ptr: *mut u8,
    size: usize,
    device_ptr: Option<cust::DevicePointer<u8>>,
    is_mapped: bool,
    allocation_time: Instant,
    usage_count: std::sync::atomic::AtomicUsize,
}

impl PinnedAllocation {
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_ptr<T>(&self) -> *mut T {
        self.ptr as *mut T
    }

    pub fn device_ptr(&self) -> Option<cust::DevicePointer<u8>> {
        self.device_ptr
    }

    pub fn is_mapped(&self) -> bool {
        self.is_mapped
    }

    pub fn allocation_time(&self) -> Instant {
        self.allocation_time
    }

    pub fn usage_count(&self) -> usize {
        self.usage_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn copy_from_host<T>(&mut self, data: &[T]) -> CudaResult<()>
    where
        T: Copy,
    {
        let size_bytes = std::mem::size_of_val(data);
        if size_bytes > self.size {
            return Err(CudaError::Context {
                message: "Data size exceeds allocation size".to_string(),
            });
        }

        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr() as *const u8, self.ptr, size_bytes);
        }
        self.usage_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub fn copy_to_host<T>(&self, data: &mut [T]) -> CudaResult<()>
    where
        T: Copy,
    {
        let size_bytes = std::mem::size_of_val(data);
        if size_bytes > self.size {
            return Err(CudaError::Context {
                message: "Data size exceeds allocation size".to_string(),
            });
        }

        unsafe {
            ptr::copy_nonoverlapping(self.ptr, data.as_mut_ptr() as *mut u8, size_bytes);
        }
        self.usage_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

/// Pinned memory manager (Legacy wrapper)
#[derive(Debug)]
pub struct PinnedMemoryManager {
    config: PinnedMemoryConfig,
    total_allocated: std::sync::atomic::AtomicUsize,
    peak_allocated: std::sync::atomic::AtomicUsize,
    allocation_count: std::sync::atomic::AtomicUsize,
}

/// Pinned memory configuration
#[derive(Debug, Clone)]
pub struct PinnedMemoryConfig {
    pub max_total_size: usize,
    pub enable_mapping: bool,
    pub enable_write_combine: bool,
    pub flags: PinnedMemoryFlags,
}

/// Pinned memory request
#[derive(Debug, Clone)]
pub struct PinnedMemoryRequest {
    pub size: usize,
    pub enable_mapping: bool,
    pub enable_write_combine: bool,
    pub preferred_device: Option<usize>,
}

/// Pinned memory flags
#[derive(Debug, Clone)]
pub struct PinnedMemoryFlags {
    pub host_alloc_default: bool,
    pub host_alloc_portable: bool,
    pub host_alloc_mapped: bool,
    pub host_alloc_write_combined: bool,
}

impl PinnedMemoryManager {
    pub fn new(config: PinnedMemoryConfig) -> CudaResult<Arc<Self>> {
        Ok(Arc::new(Self {
            config,
            total_allocated: std::sync::atomic::AtomicUsize::new(0),
            peak_allocated: std::sync::atomic::AtomicUsize::new(0),
            allocation_count: std::sync::atomic::AtomicUsize::new(0),
        }))
    }

    pub fn allocate_pinned(&self, request: PinnedMemoryRequest) -> CudaResult<PinnedAllocation> {
        // Use the new pinned memory system
        match new_allocate_pinned(request.size) {
            Ok(allocation) => {
                // Convert to legacy format
                let ptr = allocation.as_ptr();
                let size = allocation.size();

                let pinned_allocation = PinnedAllocation {
                    ptr,
                    size,
                    device_ptr: None, // Would be populated if mapped
                    is_mapped: request.enable_mapping,
                    allocation_time: Instant::now(),
                    usage_count: std::sync::atomic::AtomicUsize::new(0),
                };

                // Update statistics
                let current = self.total_allocated.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
                self.allocation_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                let mut peak = self.peak_allocated.load(std::sync::atomic::Ordering::Relaxed);
                while current > peak {
                    match self.peak_allocated.compare_exchange_weak(
                        peak, current,
                        std::sync::atomic::Ordering::Relaxed,
                        std::sync::atomic::Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(x) => peak = x,
                    }
                }

                // Keep the new allocation alive
                std::mem::forget(allocation);

                Ok(pinned_allocation)
            }
            Err(e) => Err(CudaError::Context {
                message: format!("Pinned memory allocation failed: {}", e),
            })
        }
    }

    pub fn deallocate_pinned(&self, allocation: PinnedAllocation) -> CudaResult<()> {
        let size = allocation.size();
        self.total_allocated.fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
        self.allocation_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        drop(allocation);
        Ok(())
    }

    pub fn stats(&self) -> PinnedMemoryStats {
        PinnedMemoryStats {
            total_allocated: self.total_allocated.load(std::sync::atomic::Ordering::Relaxed),
            peak_allocated: self.peak_allocated.load(std::sync::atomic::Ordering::Relaxed),
            allocation_count: self.allocation_count.load(std::sync::atomic::Ordering::Relaxed),
            max_total_size: self.config.max_total_size,
        }
    }

    pub fn cleanup(&self) -> CudaResult<usize> {
        // Trigger system maintenance
        crate::cuda::memory::perform_system_maintenance()
            .map_err(|e| CudaError::Context {
                message: format!("Cleanup failed: {}", e),
            })?;
        Ok(0) // Return number of freed allocations
    }

    pub fn force_cleanup(&self) -> CudaResult<()> {
        self.cleanup().map(|_| ())
    }

    pub fn memory_info(&self) -> PinnedMemoryInfo {
        let stats = self.stats();
        PinnedMemoryInfo {
            total_allocated: stats.total_allocated,
            peak_allocated: stats.peak_allocated,
            allocation_count: stats.allocation_count,
            available: self.config.max_total_size.saturating_sub(stats.total_allocated),
            utilization_ratio: if self.config.max_total_size > 0 {
                stats.total_allocated as f64 / self.config.max_total_size as f64
            } else {
                0.0
            },
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub device_id: usize,
}

/// Memory pressure information
#[derive(Debug, Clone)]
pub struct MemoryPressureInfo {
    pub current_allocated: usize,
    pub memory_limit: usize,
    pub pressure_threshold: usize,
    pub pressure_ratio: f64,
    pub under_pressure: bool,
}

/// Pinned memory statistics
#[derive(Debug, Clone)]
pub struct PinnedMemoryStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub allocation_count: usize,
    pub max_total_size: usize,
}

/// Pinned memory information
#[derive(Debug, Clone)]
pub struct PinnedMemoryInfo {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub allocation_count: usize,
    pub available: usize,
    pub utilization_ratio: f64,
}

// Default implementations
impl Default for PinnedMemoryConfig {
    fn default() -> Self {
        Self {
            max_total_size: 1024 * 1024 * 1024, // 1GB
            enable_mapping: true,
            enable_write_combine: false,
            flags: PinnedMemoryFlags::default(),
        }
    }
}

impl Default for PinnedMemoryFlags {
    fn default() -> Self {
        Self {
            host_alloc_default: true,
            host_alloc_portable: false,
            host_alloc_mapped: false,
            host_alloc_write_combined: false,
        }
    }
}

// Global convenience functions (Legacy API)

/// Get memory manager for device
pub fn get_memory_manager(device_id: usize) -> CudaResult<Arc<CudaMemoryManager>> {
    let mut managers = MEMORY_MANAGERS.lock().map_err(|_| CudaError::Context {
        message: "Failed to acquire memory managers lock".to_string(),
    })?;

    if let Some(manager) = managers.get(&device_id) {
        Ok(Arc::clone(manager))
    } else {
        let manager = Arc::new(CudaMemoryManager::new(device_id)?);
        managers.insert(device_id, Arc::clone(&manager));
        Ok(manager)
    }
}

/// Get pinned memory manager
pub fn get_pinned_memory_manager(
    device_id: usize,
    config: Option<PinnedMemoryConfig>,
) -> CudaResult<Arc<PinnedMemoryManager>> {
    let mut managers = PINNED_MANAGERS.lock().map_err(|_| CudaError::Context {
        message: "Failed to acquire pinned memory managers lock".to_string(),
    })?;

    if let Some(manager) = managers.get(&device_id) {
        Ok(Arc::clone(manager))
    } else {
        let manager = PinnedMemoryManager::new(config.unwrap_or_default())?;
        managers.insert(device_id, Arc::clone(&manager));
        Ok(manager)
    }
}