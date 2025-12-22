//! Storage system for tensor data management
//!
//! This module provides a comprehensive storage system for tensor data, including
//! memory allocation, NUMA awareness, memory mapping, pooling, and cross-device
//! operations. The system is designed to be flexible, efficient, and backend-agnostic.
//!
//! # Architecture
//!
//! The storage system is organized into several specialized modules:
//!
//! - [`core`] - Core storage interfaces and shared storage wrapper
//! - [`allocation`] - Backend allocation framework with typed memory handles
//! - [`memory_info`] - Memory information and allocation strategies
//! - [`operations`] - Memory copy operations and async support
//! - [`memory_format`] - Memory layout management for different tensor formats
//! - [`views`] - Storage view system for zero-copy tensor operations
//! - [`pooling`] - Memory pooling system with thread-local optimization
//! - [`numa`] - NUMA-aware memory allocation and topology detection
//! - [`mapped`] - Memory-mapped storage for large tensors with lazy loading
//! - [`registry`] - Registry for backend allocators and storage systems
//!
//! # Basic Usage
//!
//! ```ignore
//! use torsh_core::storage::{Storage, SharedStorage, BackendAllocator};
//! use torsh_core::device::CpuDevice;
//!
//! // Create a storage instance
//! let device = CpuDevice::new();
//! let storage = MyStorage::allocate(&device, 1000)?;
//! let shared = SharedStorage::new(storage);
//!
//! // Use with allocators
//! let allocator = MyAllocator::new();
//! let handle = allocator.allocate_raw(&device, 1024, 8)?;
//! ```
//!
//! # Memory Management
//!
//! The storage system provides several memory management strategies:
//!
//! - **Direct allocation** via [`BackendAllocator`] for simple cases
//! - **Pooled allocation** via [`pooling`] for frequent small allocations
//! - **NUMA-aware allocation** via [`numa`] for multi-socket systems
//! - **Memory-mapped storage** via [`mapped`] for large datasets
//! - **Cross-device operations** via [`operations`] for GPU/CPU transfers
//!
//! # Thread Safety
//!
//! All storage components are designed to be thread-safe and can be shared
//! across threads using [`SharedStorage`] and similar wrapper types.

// Module declarations
pub mod aligned;
pub mod allocation;
pub mod core;
pub mod mapped;
pub mod memory_format;
pub mod memory_info;
pub mod numa;
pub mod operations;
pub mod pooling;
pub mod registry;
pub mod views;

// Re-export all public items for backward compatibility
// This ensures that existing code using `use torsh_core::storage::SomeType` continues to work

// Core storage types
pub use self::core::{SharedStorage, Storage};

// Allocation system
pub use self::allocation::{
    AllocationRequest, BackendAllocator, RawMemoryHandle, TypedMemoryHandle, TypedMemoryStats,
};

// Memory information and strategies
pub use self::memory_info::{AllocationStrategy, MemoryInfo};

// Memory operations
pub use self::operations::{
    BackendAsyncMemory, BackendMemoryCopy, CopyOperation, MemoryOperationStats,
};

// Memory formats
pub use self::memory_format::{
    ConversionCost, FormatPreference, HardwareType, MemoryFormat, OperationType,
};

// Storage views
pub use self::views::{StorageView, ViewBuilder, ViewStatistics};

// Memory pooling
pub use self::pooling::{
    // Re-export pooled allocation functions
    allocate_pooled,
    allocate_pooled_with_value,
    clear_pooled_memory,
    configure_pools,
    deallocate_pooled,
    pooled_memory_stats,
    warmup_pools,
    MemoryPool,
    PoolConfig,
    PoolSizeStats,
    PoolStats,
};

// NUMA support
pub use self::numa::{
    MemoryAccessPattern, NumaAllocator, NumaMemoryHandle, NumaMetadata, NumaPolicy, NumaTopology,
    NumaTopologyStats, WorkloadType,
};

// Memory-mapped storage
pub use self::mapped::{
    AccessPatternStats, LazyLoadConfig, MappedSlice, MappedStorage, MappedStorageStats,
};

// Allocator registry
pub use self::registry::{
    AllocatorCapability, AllocatorMetadata, AllocatorRegistry, AllocatorRequirements,
    RegistryStatistics,
};

// Aligned storage for SIMD optimization
pub use self::aligned::{
    alignment, AlignedVec, AlignmentChecker, SimdLayoutAnalysis, SimdLayoutAnalyzer,
};

// Utility modules - re-export selected utilities
pub use self::allocation::utils as allocation_utils;
pub use self::memory_format::utils as memory_format_utils;
pub use self::numa::utils as numa_utils;
pub use self::operations::utils as operations_utils;
pub use self::registry::utils as registry_utils;
pub use self::views::utils as view_utils;

// Global registry functions
pub use self::registry::{global_registry, initialize_global_registry};

// Additional convenience re-exports for commonly used combinations
pub use self::allocation::TypedMemoryHandle as TensorMemoryHandle;
pub use self::core::SharedStorage as SharedTensorStorage;

/// Prelude module for common storage imports
///
/// This module provides a convenient way to import the most commonly used
/// storage types and traits.
///
/// ```ignore
/// use torsh_core::storage::prelude::*;
/// ```
pub mod prelude {
    pub use super::allocation::{BackendAllocator, RawMemoryHandle, TypedMemoryHandle};
    pub use super::core::{SharedStorage, Storage};
    pub use super::memory_format::MemoryFormat;
    pub use super::memory_info::{AllocationStrategy, MemoryInfo};
    pub use super::numa::{NumaPolicy, NumaTopology};
    pub use super::pooling::{allocate_pooled, deallocate_pooled};
    pub use super::registry::AllocatorRegistry;
    pub use super::views::StorageView;
}

/// Utility functions that operate across multiple storage modules
pub mod utils {
    use super::*;

    /// Create a default storage configuration for a given device type
    pub fn default_storage_config() -> StorageConfig {
        StorageConfig {
            memory_format: MemoryFormat::Contiguous,
            allocation_strategy: AllocationStrategy::Immediate,
            numa_policy: NumaPolicy::LocalPreferred,
            enable_pooling: true,
            enable_memory_mapping: false,
            lazy_load_config: mapped::LazyLoadConfig::default(),
        }
    }

    /// Get recommended memory format for a tensor shape and operation
    pub fn recommend_memory_format(
        shape: &[usize],
        operation: OperationType,
        hardware: HardwareType,
    ) -> MemoryFormat {
        memory_format_utils::optimal_format_for_tensor(shape, operation, hardware)
    }

    /// Calculate optimal allocation strategy based on size and access pattern
    pub fn recommend_allocation_strategy(
        size_bytes: usize,
        access_pattern: AccessPattern,
    ) -> AllocationStrategy {
        match access_pattern {
            AccessPattern::Frequent if size_bytes <= 64 * 1024 => AllocationStrategy::Pooled,
            AccessPattern::Large if size_bytes >= 1024 * 1024 * 1024 => {
                AllocationStrategy::PreAllocated
            }
            AccessPattern::Lazy => AllocationStrategy::Lazy,
            _ => AllocationStrategy::Immediate,
        }
    }

    /// Check if NUMA optimization would be beneficial
    pub fn should_use_numa(allocation_sizes: &[usize], numa_topology: &NumaTopology) -> bool {
        // Only beneficial for multi-node systems with significant allocations
        numa_topology.node_count > 1
            && allocation_sizes.iter().sum::<usize>() > 1024 * 1024 // 1MB threshold
            && numa_utils::has_numa_topology(numa_topology)
    }

    /// Create a storage view that automatically optimizes for the access pattern
    pub fn create_optimized_view<S: Storage>(
        storage: SharedStorage<S>,
        access_pattern: AccessPattern,
    ) -> Result<StorageView<S>, crate::error::TorshError> {
        let view_len = storage.get().len();
        match access_pattern {
            AccessPattern::Sequential => {
                // Create view for the entire storage for sequential access
                StorageView::new(storage, 0, view_len)
            }
            AccessPattern::Random => {
                // For random access, create a smaller view to enable better caching
                let chunk_size = std::cmp::min(view_len, 64 * 1024); // 64KB chunks
                StorageView::new(storage, 0, chunk_size)
            }
            AccessPattern::Frequent => {
                // For frequent access, use the full view
                StorageView::new(storage, 0, view_len)
            }
            AccessPattern::Large => {
                // For large data, use full view but consider memory mapping
                StorageView::new(storage, 0, view_len)
            }
            AccessPattern::Lazy => {
                // Start with a small view for lazy access
                let initial_size = std::cmp::min(view_len, 4096); // 4KB initial
                StorageView::new(storage, 0, initial_size)
            }
        }
    }

    /// Get storage statistics across all components
    pub fn storage_system_stats() -> StorageSystemStats {
        let pool_stats = pooled_memory_stats();
        let registry = global_registry();
        let registry_stats = registry.read().unwrap().statistics();

        StorageSystemStats {
            pooled_memory_types: pool_stats.len(),
            total_pooled_allocations: pool_stats
                .values()
                .map(|s| s.total_cached_allocations as u64)
                .sum(),
            registered_allocators: registry_stats.total_allocators,
            backend_types: registry_stats.backend_counts.len(),
        }
    }
}

/// Configuration for storage system behavior
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Default memory format to use
    pub memory_format: MemoryFormat,
    /// Default allocation strategy
    pub allocation_strategy: AllocationStrategy,
    /// NUMA policy for multi-node systems
    pub numa_policy: NumaPolicy,
    /// Whether to enable memory pooling for small allocations
    pub enable_pooling: bool,
    /// Whether to enable memory mapping for large data
    pub enable_memory_mapping: bool,
    /// Configuration for lazy loading when memory mapping is enabled
    pub lazy_load_config: mapped::LazyLoadConfig,
}

impl Default for StorageConfig {
    fn default() -> Self {
        utils::default_storage_config()
    }
}

/// Access pattern hints for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential access pattern
    Sequential,
    /// Random access pattern
    Random,
    /// Frequent small accesses
    Frequent,
    /// Large block access
    Large,
    /// Lazy/deferred access
    Lazy,
}

/// System-wide storage statistics
#[derive(Debug, Clone)]
pub struct StorageSystemStats {
    /// Number of different types using pooled memory
    pub pooled_memory_types: usize,
    /// Total number of pooled allocations across all types
    pub total_pooled_allocations: u64,
    /// Number of registered allocators
    pub registered_allocators: usize,
    /// Number of different backend types
    pub backend_types: usize,
}

impl std::fmt::Display for StorageSystemStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StorageSystem(pooled_types={}, pooled_allocs={}, allocators={}, backends={})",
            self.pooled_memory_types,
            self.total_pooled_allocations,
            self.registered_allocators,
            self.backend_types
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CpuDevice;

    // Simple test storage implementation
    #[derive(Debug)]
    struct TestStorage {
        data: Vec<f32>,
        device: CpuDevice,
    }

    impl Storage for TestStorage {
        type Elem = f32;
        type Device = CpuDevice;

        fn allocate(device: &Self::Device, size: usize) -> Result<Self, crate::error::TorshError> {
            Ok(TestStorage {
                data: vec![0.0; size],
                device: device.clone(),
            })
        }

        fn len(&self) -> usize {
            self.data.len()
        }

        fn device(&self) -> &Self::Device {
            &self.device
        }

        fn clone_storage(&self) -> Result<Self, crate::error::TorshError> {
            Ok(TestStorage {
                data: self.data.clone(),
                device: self.device.clone(),
            })
        }
    }

    #[test]
    fn test_storage_integration() {
        let device = CpuDevice::new();
        let storage = TestStorage::allocate(&device, 100).unwrap();
        let shared = SharedStorage::new(storage);

        assert_eq!(shared.get().len(), 100);
        assert_eq!(shared.strong_count(), 1);

        // Test cloning
        let _cloned_shared = shared.clone();
        assert_eq!(shared.strong_count(), 2);
    }

    #[test]
    fn test_storage_view_integration() {
        let device = CpuDevice::new();
        let storage = TestStorage::allocate(&device, 100).unwrap();
        let shared = SharedStorage::new(storage);

        let view = StorageView::new(shared.clone(), 10, 20).unwrap();
        assert_eq!(view.offset(), 10);
        assert_eq!(view.view_len(), 20);

        let sub_view = view.slice(5, 10).unwrap();
        assert_eq!(sub_view.offset(), 15); // 10 + 5
        assert_eq!(sub_view.view_len(), 10);
    }

    #[test]
    fn test_memory_format_integration() {
        let format = MemoryFormat::Contiguous;
        assert!(format.is_contiguous());
        assert!(!format.is_channels_last());

        let channels_last = MemoryFormat::ChannelsLast;
        assert!(channels_last.is_channels_last());
        assert_eq!(channels_last.expected_dims(), Some(4));
    }

    #[test]
    fn test_storage_config() {
        let config = StorageConfig::default();
        assert_eq!(config.memory_format, MemoryFormat::Contiguous);
        assert_eq!(config.allocation_strategy, AllocationStrategy::Immediate);
        assert!(config.enable_pooling);
    }

    #[test]
    fn test_utils_recommendations() {
        // Test memory format recommendation
        let shape = [1, 3, 224, 224]; // NCHW
        let format =
            utils::recommend_memory_format(&shape, OperationType::Convolution, HardwareType::GPU);
        assert_eq!(format, MemoryFormat::ChannelsLast); // Should recommend NHWC for GPU convolution

        // Test allocation strategy recommendation
        let strategy = utils::recommend_allocation_strategy(1024, AccessPattern::Frequent);
        assert_eq!(strategy, AllocationStrategy::Pooled);

        let strategy =
            utils::recommend_allocation_strategy(2 * 1024 * 1024 * 1024, AccessPattern::Large);
        assert_eq!(strategy, AllocationStrategy::PreAllocated);
    }

    #[test]
    fn test_optimized_view_creation() {
        let device = CpuDevice::new();
        let storage = TestStorage::allocate(&device, 1000).unwrap();
        let shared = SharedStorage::new(storage);

        // Test sequential access view
        let view = utils::create_optimized_view(shared.clone(), AccessPattern::Sequential).unwrap();
        assert_eq!(view.view_len(), 1000); // Should use full view for sequential access

        // Test random access view
        let view = utils::create_optimized_view(shared.clone(), AccessPattern::Random).unwrap();
        assert!(view.view_len() <= 64 * 1024); // Should use smaller chunk for random access
    }

    #[test]
    fn test_prelude_imports() {
        // Test that prelude imports work correctly
        use super::prelude::*;

        let device = CpuDevice::new();
        let storage = TestStorage::allocate(&device, 10).unwrap();
        let _shared = SharedStorage::new(storage);

        let _format = MemoryFormat::Contiguous;
        let _strategy = AllocationStrategy::Immediate;
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that all re-exported types are available at the module level
        let _: MemoryFormat = MemoryFormat::Contiguous;
        let _: AllocationStrategy = AllocationStrategy::Immediate;
        let _: NumaPolicy = NumaPolicy::LocalPreferred;

        // Test that utility functions are accessible
        let _config = utils::default_storage_config();
        let _stats = utils::storage_system_stats();
    }

    #[test]
    fn test_storage_system_stats() {
        let stats = utils::storage_system_stats();

        // Basic sanity checks (all fields are unsigned, so no need for >= 0 checks)
        // Just check that stats are available
        let _check = stats.registered_allocators;
        let _check = stats.backend_types;
        let _check = stats.pooled_memory_types;
    }
}
