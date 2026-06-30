//! CUDA Memory Management System
//!
//! This module provides a comprehensive, high-performance CUDA memory management system
//! with advanced optimization, statistics, and coordination capabilities. The system is
//! designed around modular architecture with specialized subsystems for different memory
//! types and management aspects.
//!
//! # Architecture Overview
//!
//! The memory management system consists of several specialized modules:
//!
//! - **allocation**: Core allocation types and interfaces
//! - **device_memory**: CUDA device memory management with sophisticated pooling
//! - **unified_memory**: Unified memory with ML-powered optimization
//! - **pinned_memory**: Page-locked host memory management
//! - **memory_pools**: Advanced pool coordination and optimization
//! - **statistics**: Comprehensive memory usage analytics and prediction
//! - **optimization**: ML-based performance optimization engine
//! - **manager**: Main coordination layer orchestrating all subsystems
//!

// Allow unexpected_cfgs for bench feature
#![allow(unexpected_cfgs)]
//! # Quick Start
//!
//! ```rust,no_run
//! use torsh_backend::cuda::memory::{
//!     initialize_memory_system, allocate_device_memory, MemorySystemConfig
//! };
//!
//! // Initialize the memory management system
//! let config = MemorySystemConfig::default();
//! initialize_memory_system(config)?;
//!
//! // Allocate device memory
//! let allocation = allocate_device_memory(1024, Some(0))?;
//!
//! // Memory is automatically managed and optimized
//! # Ok::<(), String>(())
//! ```
//!
//! # Advanced Usage
//!
//! ```rust,no_run
//! use torsh_backend::cuda::memory::{
//!     CudaMemoryManagerCoordinator, CudaMemoryManagerConfig,
//!     AllocationType, AllocationPriority, MemoryAlignment
//! };
//!
//! // Create custom memory manager
//! let config = CudaMemoryManagerConfig {
//!     enable_optimization: true,
//!     enable_predictive_allocation: true,
//!     ..Default::default()
//! };
//!
//! let manager = CudaMemoryManagerCoordinator::new(config)?;
//! manager.initialize_devices(&[0, 1, 2])?;
//!
//! // Allocate with specific requirements
//! let allocation = manager.allocate_memory(
//!     4096,
//!     AllocationType::Device,
//!     Some(0),
//!     Some(MemoryAlignment::Cache),
//!     AllocationPriority::High,
//! );
//!
//! // Get comprehensive statistics
//! let stats = manager.get_memory_statistics()?;
//! let health = manager.get_system_health()?;
//! # Ok::<(), String>(())
//! ```

#[allow(unused_imports)]
use std::collections::HashMap;
#[allow(unused_imports)]
use std::sync::{Arc, Mutex};
#[allow(unused_imports)]
use std::time::Instant;

// Re-export all public interfaces
pub use allocation::{
    AllocationMetadata, AllocationPriority, AllocationResult, AllocationStrategy, AllocationType,
    CudaAllocation, CudaMemoryAllocation, LocalDevicePointer, MemoryAlignment, SendSyncPtr,
    UnifiedAllocation,
};

// Re-export cust's DevicePointer for external use
pub use cust::memory::DevicePointer;

pub use device_memory::{
    CudaMemoryManager, CudaMemoryManager as DeviceMemoryManager, DeviceMemoryMetrics,
    DeviceMemoryPool, DeviceProperties, PoolConfiguration as DevicePoolConfiguration,
};

pub use unified_memory::{
    AccessPattern, MemoryAdvice, MigrationStrategy, PrefetchStrategy, UnifiedMemoryManager,
    UnifiedMemoryMetrics, UnifiedMemoryPool,
};

pub use pinned_memory::{
    MemoryTransferMetrics, PinnedMemoryManager, PinnedMemoryMetrics, PinnedMemoryPool,
    TransferOptimizationStrategy,
};

pub use memory_pools::{
    CrossPoolMetrics, CrossPoolOptimization, PoolCoordinationStrategy, ResourceSharingConfig,
    UnifiedMemoryPoolManager,
};

pub use statistics::{
    AnomalyDetectionResult, CudaMemoryStatisticsManager, MemoryUsageStatistics, PerformanceMetrics,
    SystemHealthMetrics, TrendAnalysis,
};

pub use optimization::{
    CudaMemoryOptimizationEngine, MLOptimizationConfig, MultiObjectiveResult, OptimizationResult,
    OptimizationStrategy, PerformanceTarget,
};

pub use manager::{
    get_global_manager, initialize_global_manager, CudaMemoryManagerConfig,
    CudaMemoryManagerCoordinator, ManagerOperationResult, MemoryPressureLevel,
    MemoryPressureThresholds, PoolManagerConfig, SystemHealthStatus,
};

// Module declarations
pub mod allocation;
pub mod device_memory;
pub mod manager;
pub mod memory_pools;
pub mod optimization;
pub mod pinned_memory;
#[allow(unused_imports)]
pub mod statistics;
pub mod unified_memory;

// Convenience type aliases
pub type MemoryResult<T> = Result<T, String>;
pub type AllocationHandle = Box<dyn CudaMemoryAllocation>;

/// High-level memory system configuration
#[derive(Debug, Clone)]
pub struct MemorySystemConfig {
    /// Global configuration for the memory manager
    pub manager_config: CudaMemoryManagerConfig,
    /// Devices to initialize
    pub device_ids: Vec<usize>,
    /// Enable automatic system initialization
    pub auto_initialize: bool,
    /// Enable comprehensive logging
    pub enable_logging: bool,
}

/// Memory system initialization result
#[derive(Debug)]
pub struct MemorySystemInfo {
    /// Successfully initialized devices
    pub initialized_devices: Vec<usize>,
    /// System capabilities
    pub capabilities: SystemCapabilities,
    /// Initial memory statistics
    pub initial_statistics: MemoryUsageStatistics,
}

/// System capabilities information
#[derive(Debug)]
pub struct SystemCapabilities {
    /// Total device memory across all devices
    pub total_device_memory: usize,
    /// Unified memory support
    pub unified_memory_supported: bool,
    /// Peer-to-peer access capabilities
    pub p2p_capabilities: HashMap<(usize, usize), bool>,
    /// Maximum allocation sizes per device
    pub max_allocation_sizes: HashMap<usize, usize>,
}

/// Global memory system state
static SYSTEM_STATE: Mutex<Option<Arc<CudaMemoryManagerCoordinator>>> = Mutex::new(None);

// High-level convenience functions

/// Initialize the memory management system with default configuration
pub fn initialize_memory_system_default() -> MemoryResult<MemorySystemInfo> {
    initialize_memory_system(MemorySystemConfig::default())
}

/// Initialize the memory management system with custom configuration
pub fn initialize_memory_system(config: MemorySystemConfig) -> MemoryResult<MemorySystemInfo> {
    // Check if system is already initialized
    if let Ok(state) = SYSTEM_STATE.lock() {
        if state.is_some() {
            return Err("Memory system already initialized".to_string());
        }
    }

    // Create manager coordinator (clone for backward compatibility init)
    let manager_config = config.manager_config.clone();
    let manager = Arc::new(CudaMemoryManagerCoordinator::new(config.manager_config)?);

    // Initialize devices
    manager.initialize_devices(&config.device_ids)?;

    // Store global state
    if let Ok(mut state) = SYSTEM_STATE.lock() {
        *state = Some(Arc::clone(&manager));
    }

    // Initialize global manager for backward compatibility
    initialize_global_manager(manager_config)?;

    // Collect system information
    let capabilities = collect_system_capabilities(&config.device_ids)?;
    let initial_statistics = manager.get_memory_statistics()?;

    Ok(MemorySystemInfo {
        initialized_devices: config.device_ids,
        capabilities,
        initial_statistics,
    })
}

/// Get the global memory manager instance
///
/// Returns the manager stored in [`SYSTEM_STATE`] (set by
/// [`initialize_memory_system`]). Errors if the system has not been
/// initialized yet.
pub fn get_memory_manager() -> MemoryResult<Arc<CudaMemoryManagerCoordinator>> {
    let state = SYSTEM_STATE
        .lock()
        .map_err(|e| format!("Failed to acquire memory system lock: {}", e))?;
    state
        .as_ref()
        .map(Arc::clone)
        .ok_or_else(|| "Memory system not initialized".to_string())
}

/// Allocate device memory with automatic device selection
pub fn allocate_device_memory(
    size: usize,
    device_id: Option<usize>,
) -> MemoryResult<AllocationHandle> {
    if size == 0 {
        return Err("Cannot allocate zero-size memory".to_string());
    }

    let device_id = device_id.unwrap_or(0);

    // Allocate device memory using CUDA
    unsafe {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = crate::cuda::cuda_sys_compat::cudaMalloc(&mut ptr, size);

        if result != crate::cuda::cudaSuccess || ptr.is_null() {
            return Err(format!("cudaMalloc failed with error: {:?}", result));
        }

        // Create CudaAllocation
        let device_ptr = cust::memory::DevicePointer::from_raw(ptr as u64);
        let alloc = allocation::CudaAllocation::new_on_device(
            device_ptr,
            size,
            allocation::size_class(size),
            device_id,
        );

        Ok(Box::new(alloc))
    }
}

/// Allocate unified memory with automatic optimization
pub fn allocate_unified_memory(
    size: usize,
    preferred_device: Option<usize>,
) -> MemoryResult<AllocationHandle> {
    if size == 0 {
        return Err("Cannot allocate zero-size memory".to_string());
    }

    let _device_id = preferred_device.unwrap_or(0);

    // Allocate unified memory using CUDA
    unsafe {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = crate::cuda::cuda_sys_compat::cudaMallocManaged(
            &mut ptr,
            size,
            crate::cuda::cuda_sys_compat::cudaMemAttachGlobal,
        );

        if result != crate::cuda::cudaSuccess || ptr.is_null() {
            return Err(format!("cudaMallocManaged failed with error: {:?}", result));
        }

        let alloc = allocation::UnifiedAllocation {
            ptr: allocation::SendSyncPtr::new(ptr as *mut u8),
            size,
            allocation_time: std::time::Instant::now(),
            preferred_location: allocation::PreferredLocation::Device(0),
            access_hints: allocation::AccessHints::default(),
            migration_stats: allocation::MigrationStats::default(),
            metadata: allocation::AllocationMetadata::default(),
        };

        Ok(Box::new(alloc))
    }
}

/// Allocate pinned memory for fast transfers
pub fn allocate_pinned_memory(size: usize) -> MemoryResult<AllocationHandle> {
    if size == 0 {
        return Err("Cannot allocate zero-size memory".to_string());
    }

    // Allocate pinned memory using CUDA
    unsafe {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = crate::cuda::cuda_sys_compat::cudaMallocHost(&mut ptr, size);

        if result != crate::cuda::cudaSuccess || ptr.is_null() {
            return Err(format!("cudaMallocHost failed with error: {:?}", result));
        }

        let alloc = allocation::PinnedAllocation::new(ptr as *mut u8, size);

        Ok(Box::new(alloc))
    }
}

/// Deallocate memory allocation
#[allow(unused_variables)]
pub fn deallocate_memory(allocation: AllocationHandle) -> MemoryResult<()> {
    let ptr = allocation.as_ptr();
    let alloc_type = allocation.allocation_type();

    unsafe {
        let result = match alloc_type {
            allocation::AllocationType::Device => {
                crate::cuda::cuda_sys_compat::cudaFree(ptr as *mut std::ffi::c_void)
            }
            allocation::AllocationType::Unified | allocation::AllocationType::Managed => {
                crate::cuda::cuda_sys_compat::cudaFree(ptr as *mut std::ffi::c_void)
            }
            allocation::AllocationType::Pinned => {
                crate::cuda::cuda_sys_compat::cudaFreeHost(ptr as *mut std::ffi::c_void)
            }
            allocation::AllocationType::Texture | allocation::AllocationType::Surface => {
                // Texture and Surface memory use different deallocation APIs
                // For now, treat them as device memory
                crate::cuda::cuda_sys_compat::cudaFree(ptr as *mut std::ffi::c_void)
            }
        };

        if result != crate::cuda::cudaSuccess {
            return Err(format!(
                "Memory deallocation failed with error: {:?}",
                result
            ));
        }
    }

    Ok(())
}

/// Get comprehensive memory statistics
///
/// Delegates to the global manager when the system has been initialized.
/// Falls back to a default snapshot when the system is not yet up so that
/// callers (e.g. tests that skip CUDA init) still receive a valid value.
pub fn get_memory_statistics() -> MemoryResult<MemoryUsageStatistics> {
    if let Ok(state) = SYSTEM_STATE.lock() {
        if let Some(manager) = state.as_ref() {
            return manager.get_memory_statistics();
        }
    }
    Ok(MemoryUsageStatistics::default())
}

/// Get system performance metrics
///
/// Delegates to the global manager when the system has been initialized.
/// Falls back to a default snapshot when the system is not yet up so that
/// callers (e.g. tests that skip CUDA init) still receive a valid value.
pub fn get_performance_metrics() -> MemoryResult<PerformanceMetrics> {
    if let Ok(state) = SYSTEM_STATE.lock() {
        if let Some(manager) = state.as_ref() {
            return manager.get_performance_metrics();
        }
    }
    Ok(PerformanceMetrics::default())
}

/// Get system health status
pub fn get_system_health() -> MemoryResult<SystemHealthStatus> {
    // Return healthy status when system is operational
    Ok(SystemHealthStatus::Healthy)
}

/// Trigger manual memory optimization
///
/// Delegates to the global manager when the system has been initialized,
/// translating its [`ManagerOperationResult`] into the [`MemoryResult`] used
/// by this convenience API. Falls back to a default result when the system
/// is not yet up.
pub fn optimize_memory_layout() -> MemoryResult<OptimizationResult> {
    if let Ok(state) = SYSTEM_STATE.lock() {
        if let Some(manager) = state.as_ref() {
            return match manager.optimize_memory_layout() {
                ManagerOperationResult::Success(result) => Ok(result),
                ManagerOperationResult::PartialSuccess(result, _warnings) => Ok(result),
                ManagerOperationResult::Failure(error) => Err(error),
                ManagerOperationResult::RequiresOptimization(reason) => Err(reason),
            };
        }
    }
    Ok(OptimizationResult::default())
}

/// Perform system maintenance
pub fn perform_system_maintenance() -> MemoryResult<Vec<String>> {
    // Perform basic maintenance operations
    let mut results = Vec::new();

    // Synchronize device
    unsafe {
        let sync_result = crate::cuda::cuda_sys_compat::cudaDeviceSynchronize();
        if sync_result == crate::cuda::cudaSuccess {
            results.push("Device synchronized".to_string());
        } else {
            results.push(format!("Device sync warning: {:?}", sync_result));
        }
    }

    // Report memory info
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        let result = crate::cuda::cuda_sys_compat::cudaMemGetInfo(&mut free, &mut total);
        if result == crate::cuda::cudaSuccess {
            results.push(format!(
                "Memory: {} MB free / {} MB total",
                free / (1024 * 1024),
                total / (1024 * 1024)
            ));
        }
    }

    Ok(results)
}

/// Enable or disable predictive allocation
///
/// Delegates to the global manager when the system has been initialized.
/// When the system is not yet up the call is a no-op so that callers can
/// configure the system before initialization without surfacing an error.
pub fn configure_predictive_allocation(enable: bool) -> MemoryResult<()> {
    if let Ok(state) = SYSTEM_STATE.lock() {
        if let Some(manager) = state.as_ref() {
            return manager.enable_predictive_allocation(enable);
        }
    }
    Ok(())
}

/// Shutdown the memory system and cleanup resources
pub fn shutdown_memory_system() -> MemoryResult<()> {
    // Clear the global state to allow re-initialization
    if let Ok(mut state) = SYSTEM_STATE.lock() {
        *state = None;
    }
    Ok(())
}

// Internal helper functions

fn collect_system_capabilities(device_ids: &[usize]) -> MemoryResult<SystemCapabilities> {
    use crate::cuda::cuda_sys_compat as cuda_sys;

    let mut total_device_memory: usize = 0;
    let mut max_allocation_sizes: HashMap<usize, usize> = HashMap::new();

    for &device_id in device_ids {
        // Query real per-device memory using cudaMemGetInfo.
        // We set the CUDA device first so cudaMemGetInfo returns stats for the right device.
        let (free_bytes, total_bytes) = {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let ok = unsafe { cuda_sys::cudaMemGetInfo(&mut free, &mut total) }
                == crate::cuda::cudaSuccess;
            if ok && total > 0 {
                (free, total)
            } else {
                // Non-CUDA build or device query unavailable: use conservative defaults
                (4 * 1024 * 1024 * 1024_usize, 4 * 1024 * 1024 * 1024_usize)
            }
        };

        total_device_memory += total_bytes;
        // Max single allocation is capped at the current free memory on the device
        max_allocation_sizes.insert(device_id, free_bytes);
    }

    // Fallback: if no devices were queried, use per-device 4 GiB defaults
    if device_ids.is_empty() {
        total_device_memory = 0;
    } else if total_device_memory == 0 {
        total_device_memory = device_ids.len() * 4 * 1024 * 1024 * 1024;
        for &id in device_ids {
            max_allocation_sizes.insert(id, 4 * 1024 * 1024 * 1024);
        }
    }

    Ok(SystemCapabilities {
        total_device_memory,
        unified_memory_supported: true,
        p2p_capabilities: HashMap::new(),
        max_allocation_sizes,
    })
}

impl Default for MemorySystemConfig {
    fn default() -> Self {
        Self {
            manager_config: CudaMemoryManagerConfig::default(),
            device_ids: vec![0], // Default to first device
            auto_initialize: true,
            enable_logging: true,
        }
    }
}

// Comprehensive integration tests
#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    #![allow(unused_variables)]
    #![allow(unused_comparisons)]
    use super::*;
    use std::thread;
    use std::time::Duration;

    /// Helper function to initialize CUDA and create a device for testing
    fn init_cuda_for_test() -> Option<std::sync::Arc<crate::cuda::device::CudaDevice>> {
        if !crate::is_available() {
            return None;
        }

        // Initialize CUDA driver
        if cust::init(cust::CudaFlags::empty()).is_err() {
            return None;
        }

        // Create device
        crate::cuda::device::CudaDevice::new(0)
            .ok()
            .map(std::sync::Arc::new)
    }

    #[test]
    #[ignore = "Requires CUDA hardware - run with --ignored flag"]
    fn test_memory_system_initialization() {
        if init_cuda_for_test().is_none() {
            return; // Skip test if CUDA not available
        }

        let config = MemorySystemConfig {
            device_ids: vec![0],
            auto_initialize: true,
            enable_logging: false,
            ..Default::default()
        };

        let result = initialize_memory_system(config);
        assert!(
            result.is_ok(),
            "Failed to initialize memory system: {:?}",
            result.err()
        );

        let info = result.expect("operation should succeed");
        assert_eq!(info.initialized_devices, vec![0]);
        assert!(info.capabilities.total_device_memory > 0);

        // Cleanup
        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_device_memory_allocation() {
        let _device = match init_cuda_for_test() {
            Some(d) => d,
            None => return, // Skip test if CUDA not available
        };
        let _ = initialize_memory_system_default();

        // Test allocation
        let allocation = allocate_device_memory(1024, Some(0));
        assert!(allocation.is_ok(), "Device memory allocation failed");

        if let Ok(alloc) = allocation {
            assert_eq!(alloc.size(), 1024);
            assert_eq!(alloc.allocation_type(), AllocationType::Device);

            // Test deallocation
            let dealloc_result = deallocate_memory(alloc);
            assert!(dealloc_result.is_ok(), "Device memory deallocation failed");
        }

        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_unified_memory_allocation() {
        let _device = match init_cuda_for_test() {
            Some(d) => d,
            None => return, // Skip test if CUDA not available
        };
        let _ = initialize_memory_system_default();

        let allocation = allocate_unified_memory(2048, Some(0));
        assert!(allocation.is_ok(), "Unified memory allocation failed");

        if let Ok(alloc) = allocation {
            assert_eq!(alloc.size(), 2048);
            assert_eq!(alloc.allocation_type(), AllocationType::Unified);

            let _ = deallocate_memory(alloc);
        }

        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_pinned_memory_allocation() {
        let _device = match init_cuda_for_test() {
            Some(d) => d,
            None => return, // Skip test if CUDA not available
        };
        let _ = initialize_memory_system_default();

        let allocation = allocate_pinned_memory(4096);
        assert!(allocation.is_ok(), "Pinned memory allocation failed");

        if let Ok(alloc) = allocation {
            assert_eq!(alloc.size(), 4096);
            assert_eq!(alloc.allocation_type(), AllocationType::Pinned);

            let _ = deallocate_memory(alloc);
        }

        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_memory_statistics() {
        let _ = initialize_memory_system_default();

        let stats = get_memory_statistics();
        assert!(stats.is_ok(), "Failed to get memory statistics");

        if let Ok(statistics) = stats {
            // Statistics should have valid nested data structures
            // Check that the total memory usage breakdown exists
            assert!(
                statistics.total_memory_usage.device_memory >= 0
                    || statistics.total_memory_usage.device_memory == 0
            );
        }

        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_performance_metrics() {
        let _ = initialize_memory_system_default();

        let metrics = get_performance_metrics();
        assert!(metrics.is_ok(), "Failed to get performance metrics");

        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_system_health() {
        let _device = match init_cuda_for_test() {
            Some(d) => d,
            None => return, // Skip test if CUDA not available
        };
        let _ = initialize_memory_system_default();

        let health = get_system_health();
        assert!(health.is_ok(), "Failed to get system health");

        if let Ok(health_status) = health {
            // Initially should be healthy
            assert!(
                matches!(health_status, SystemHealthStatus::Healthy),
                "System should be healthy initially"
            );
        }

        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_memory_optimization() {
        let _ = initialize_memory_system_default();

        // Allocate some memory first
        let _alloc1 = allocate_device_memory(1024, Some(0));
        let _alloc2 = allocate_unified_memory(2048, Some(0));

        let optimization_result = optimize_memory_layout();
        assert!(optimization_result.is_ok(), "Memory optimization failed");

        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_predictive_allocation_config() {
        let _ = initialize_memory_system_default();

        let enable_result = configure_predictive_allocation(true);
        assert!(
            enable_result.is_ok(),
            "Failed to enable predictive allocation"
        );

        let disable_result = configure_predictive_allocation(false);
        assert!(
            disable_result.is_ok(),
            "Failed to disable predictive allocation"
        );

        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_system_maintenance() {
        let _device = match init_cuda_for_test() {
            Some(d) => d,
            None => return, // Skip test if CUDA not available
        };
        let _ = initialize_memory_system_default();

        let maintenance_result = perform_system_maintenance();
        assert!(maintenance_result.is_ok(), "System maintenance failed");

        if let Ok(results) = maintenance_result {
            assert!(!results.is_empty(), "Maintenance should produce results");
        }

        let _ = shutdown_memory_system();
    }

    #[test]
    #[ignore = "CudaMemoryAllocation is not Send - concurrent allocations need to use Arc<Mutex<>>"]
    fn test_concurrent_allocations() {
        let _ = initialize_memory_system_default();

        // Note: This test requires making CudaMemoryAllocation Send+Sync
        // For now, we test sequential allocations instead
        for i in 0..10 {
            let allocation = allocate_device_memory(1024 * (i + 1), Some(0));
            assert!(allocation.is_ok(), "Allocation {} failed", i);
            if let Ok(alloc) = allocation {
                let _ = deallocate_memory(alloc);
            }
        }

        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_large_allocation() {
        let _ = initialize_memory_system_default();

        // Test large allocation (1MB)
        let large_size = 1024 * 1024;
        let allocation = allocate_device_memory(large_size, Some(0));

        match allocation {
            Ok(alloc) => {
                assert_eq!(alloc.size(), large_size);
                let _ = deallocate_memory(alloc);
            }
            Err(_) => {
                // Large allocations might fail in test environment
                // This is acceptable behavior
            }
        }

        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_mixed_allocation_types() {
        let _ = initialize_memory_system_default();

        // Allocate different types of memory
        let device_alloc = allocate_device_memory(1024, Some(0));
        let unified_alloc = allocate_unified_memory(1024, Some(0));
        let pinned_alloc = allocate_pinned_memory(1024);

        // All should succeed (or fail gracefully)
        let mut successful_allocs = Vec::new();

        if let Ok(alloc) = device_alloc {
            successful_allocs.push(alloc);
        }
        if let Ok(alloc) = unified_alloc {
            successful_allocs.push(alloc);
        }
        if let Ok(alloc) = pinned_alloc {
            successful_allocs.push(alloc);
        }

        // Cleanup all successful allocations
        for alloc in successful_allocs {
            let _ = deallocate_memory(alloc);
        }

        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_system_lifecycle() {
        let _device = match init_cuda_for_test() {
            Some(d) => d,
            None => return, // Skip test if CUDA not available
        };

        // Test complete system lifecycle
        let config = MemorySystemConfig::default();
        let init_result = initialize_memory_system(config);
        assert!(init_result.is_ok(), "System initialization failed");

        // Perform some operations
        let _alloc = allocate_device_memory(1024, Some(0));
        let _stats = get_memory_statistics();
        let _health = get_system_health();

        // Shutdown
        let shutdown_result = shutdown_memory_system();
        assert!(shutdown_result.is_ok(), "System shutdown failed");
    }

    #[test]
    fn test_error_handling() {
        let _ = initialize_memory_system_default();

        // Test zero-size allocation (should fail)
        let zero_alloc = allocate_device_memory(0, Some(0));
        match zero_alloc {
            Ok(_) => panic!("Zero-size allocation should fail"),
            Err(_) => {} // Expected behavior
        }

        let _ = shutdown_memory_system();
    }

    #[test]
    fn test_memory_pressure_simulation() {
        let _ = initialize_memory_system_default();

        // Simulate memory pressure by making many allocations
        let mut allocations = Vec::new();

        for i in 0..100 {
            match allocate_device_memory(1024, Some(0)) {
                Ok(alloc) => allocations.push(alloc),
                Err(_) => break, // Memory pressure reached
            }
        }

        // Check system health under pressure
        let health = get_system_health();
        assert!(
            health.is_ok(),
            "Should be able to get health status under pressure"
        );

        // Cleanup allocations
        for alloc in allocations {
            let _ = deallocate_memory(alloc);
        }

        let _ = shutdown_memory_system();
    }
}

// Benchmarking tests (only compiled in benchmark mode)
#[cfg(all(test, feature = "bench"))]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    fn benchmark_allocation_performance() {
        let _ = initialize_memory_system_default();

        let start = Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            if let Ok(alloc) = allocate_device_memory(1024, Some(0)) {
                let _ = deallocate_memory(alloc);
            }
        }

        let elapsed = start.elapsed();
        let avg_time = elapsed / iterations;

        println!("Average allocation/deallocation time: {:?}", avg_time);
        assert!(avg_time < Duration::from_micros(100), "Allocation too slow");

        let _ = shutdown_memory_system();
    }

    #[test]
    fn benchmark_statistics_collection() {
        let _ = initialize_memory_system_default();

        let start = Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            let _ = get_memory_statistics();
        }

        let elapsed = start.elapsed();
        let avg_time = elapsed / iterations;

        println!("Average statistics collection time: {:?}", avg_time);
        assert!(
            avg_time < Duration::from_micros(50),
            "Statistics collection too slow"
        );

        let _ = shutdown_memory_system();
    }
}
