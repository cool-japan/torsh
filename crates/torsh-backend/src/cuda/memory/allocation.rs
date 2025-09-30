//! Core CUDA memory allocation types and interfaces
//!
//! This module provides the fundamental allocation types and traits used
//! throughout the CUDA memory management system. It defines the basic
//! interfaces for different types of CUDA memory allocations.

use crate::error::{CudaError, CudaResult};
use std::time::Instant;

/// CUDA memory allocation trait
///
/// Common interface for all types of CUDA memory allocations,
/// providing basic operations and metadata access.
pub trait CudaMemoryAllocation {
    /// Get the raw pointer to allocated memory
    fn as_ptr(&self) -> *mut u8;

    /// Get the size of the allocation in bytes
    fn size(&self) -> usize;

    /// Get the allocation timestamp
    fn allocation_time(&self) -> Instant;

    /// Check if the allocation is still valid
    fn is_valid(&self) -> bool;

    /// Get allocation type identifier
    fn allocation_type(&self) -> AllocationType;
}

/// Memory allocation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocationType {
    /// Standard CUDA device memory
    Device,
    /// Unified memory accessible from both host and device
    Unified,
    /// Page-locked host memory
    Pinned,
    /// Texture memory
    Texture,
    /// Surface memory
    Surface,
    /// Managed memory with automatic migration
    Managed,
}

/// CUDA device memory allocation
///
/// Represents memory allocated on the GPU device using cudaMalloc.
/// This memory is only accessible from device code unless explicitly
/// copied to/from host memory.
#[derive(Debug, Clone)]
pub struct CudaAllocation {
    /// Device pointer to allocated memory
    pub ptr: cust::DevicePointer<u8>,

    /// Size of allocation in bytes
    pub size: usize,

    /// Size class for pooling (power of 2)
    pub size_class: usize,

    /// Timestamp when allocation was created
    pub allocation_time: Instant,

    /// Whether this allocation is currently in use
    pub in_use: bool,

    /// Device ID where memory was allocated
    pub device_id: usize,

    /// Additional metadata for tracking
    pub metadata: AllocationMetadata,
}

/// Unified memory allocation
///
/// Represents memory allocated with cudaMallocManaged that can be
/// accessed from both host and device with automatic migration.
#[derive(Debug, Clone)]
pub struct UnifiedAllocation {
    /// Pointer to unified memory
    pub ptr: *mut u8,

    /// Size of allocation in bytes
    pub size: usize,

    /// Timestamp when allocation was created
    pub allocation_time: Instant,

    /// Current preferred location (device ID or host)
    pub preferred_location: PreferredLocation,

    /// Access pattern hints for optimization
    pub access_hints: AccessHints,

    /// Migration statistics
    pub migration_stats: MigrationStats,

    /// Additional metadata
    pub metadata: AllocationMetadata,
}

/// Pinned (page-locked) host memory allocation
///
/// Represents memory allocated with cudaMallocHost that is page-locked
/// and can be accessed efficiently by the GPU for faster transfers.
#[derive(Debug, Clone)]
pub struct PinnedAllocation {
    /// Pointer to pinned host memory
    pub ptr: *mut u8,

    /// Size of allocation in bytes
    pub size: usize,

    /// Timestamp when allocation was created
    pub allocation_time: Instant,

    /// Number of times this allocation has been used
    pub usage_count: usize,

    /// Whether memory is mapped to device address space
    pub is_mapped: bool,

    /// Device pointer if mapped
    pub device_ptr: Option<cust::DevicePointer<u8>>,

    /// Mapping flags used during allocation
    pub mapping_flags: PinnedMemoryFlags,

    /// Additional metadata
    pub metadata: AllocationMetadata,
}

/// Preferred location for unified memory
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreferredLocation {
    /// Prefer host memory
    Host,
    /// Prefer specific device
    Device(usize),
    /// No preference (let driver decide)
    Auto,
}

/// Access pattern hints for unified memory optimization
#[derive(Debug, Clone)]
pub struct AccessHints {
    /// Hint that data will be read-only from GPU
    pub read_mostly: bool,

    /// Hint about access pattern frequency
    pub access_frequency: AccessFrequency,

    /// Hint about data locality
    pub locality: DataLocality,

    /// Custom optimization hints
    pub custom_hints: Vec<String>,
}

/// Access frequency patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessFrequency {
    /// Very frequent access (every few kernel launches)
    VeryHigh,
    /// High frequency access
    High,
    /// Moderate frequency access
    Medium,
    /// Low frequency access
    Low,
    /// Very rare access
    VeryLow,
}

/// Data locality hints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLocality {
    /// Data is accessed in sequential patterns
    Sequential,
    /// Data is accessed randomly
    Random,
    /// Data has temporal locality (recently accessed data likely to be accessed again)
    Temporal,
    /// Data has spatial locality (nearby data likely to be accessed together)
    Spatial,
    /// Mixed access patterns
    Mixed,
}

/// Migration statistics for unified memory
#[derive(Debug, Clone)]
pub struct MigrationStats {
    /// Number of host-to-device migrations
    pub host_to_device_migrations: u64,

    /// Number of device-to-host migrations
    pub device_to_host_migrations: u64,

    /// Total bytes migrated
    pub total_bytes_migrated: u64,

    /// Average migration time
    pub average_migration_time: std::time::Duration,

    /// Last migration timestamp
    pub last_migration: Option<Instant>,
}

/// Pinned memory allocation flags
#[derive(Debug, Clone, Copy)]
pub struct PinnedMemoryFlags {
    /// Enable device mapping
    pub enable_mapping: bool,

    /// Use portable memory (accessible from all CUDA contexts)
    pub portable: bool,

    /// Use write-combining memory for better host-to-device performance
    pub write_combining: bool,

    /// Raw CUDA flags
    pub raw_flags: u32,
}

/// General allocation metadata
#[derive(Debug, Clone)]
pub struct AllocationMetadata {
    /// Unique allocation ID
    pub id: u64,

    /// Optional name/tag for debugging
    pub tag: Option<String>,

    /// Stack trace where allocation occurred (debug builds)
    pub stack_trace: Option<String>,

    /// Thread ID that performed allocation
    pub thread_id: u64,

    /// Process ID
    pub process_id: u32,

    /// Alignment requirements
    pub alignment: usize,

    /// Whether this is a temporary allocation
    pub is_temporary: bool,

    /// Expected lifetime hint
    pub expected_lifetime: Option<std::time::Duration>,

    /// Custom user data
    pub user_data: Option<Box<dyn std::any::Any + Send + Sync>>,
}

/// Allocation request parameters
#[derive(Debug, Clone)]
pub struct AllocationRequest {
    /// Requested size in bytes
    pub size: usize,

    /// Memory alignment requirements
    pub alignment: Option<usize>,

    /// Allocation type preference
    pub allocation_type: AllocationType,

    /// Device preference (if applicable)
    pub device_id: Option<usize>,

    /// Optional tag for debugging
    pub tag: Option<String>,

    /// Whether this is a temporary allocation
    pub is_temporary: bool,

    /// Expected lifetime hint for optimization
    pub expected_lifetime: Option<std::time::Duration>,

    /// Priority level for allocation
    pub priority: AllocationPriority,
}

/// Allocation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AllocationPriority {
    /// Low priority, can be delayed or swapped out
    Low,
    /// Normal priority
    Normal,
    /// High priority, should be allocated quickly
    High,
    /// Critical priority, must be allocated immediately
    Critical,
}

/// Allocation statistics and usage information
#[derive(Debug, Clone)]
pub struct AllocationStats {
    /// Total number of allocations created
    pub total_allocations: u64,

    /// Number of currently active allocations
    pub active_allocations: u64,

    /// Total bytes allocated over lifetime
    pub total_bytes_allocated: u64,

    /// Currently allocated bytes
    pub current_bytes_allocated: u64,

    /// Peak memory usage
    pub peak_bytes_allocated: u64,

    /// Average allocation size
    pub average_allocation_size: usize,

    /// Allocation success rate
    pub success_rate: f32,

    /// Cache hit rate (for pooled allocations)
    pub cache_hit_rate: f32,

    /// Average allocation time
    pub average_allocation_time: std::time::Duration,

    /// Memory fragmentation level (0.0 to 1.0)
    pub fragmentation_level: f32,
}

// Implementation for CudaAllocation
impl CudaAllocation {
    /// Create a new CUDA device memory allocation
    pub fn new(ptr: cust::DevicePointer<u8>, size: usize, size_class: usize) -> Self {
        Self {
            ptr,
            size,
            size_class,
            allocation_time: Instant::now(),
            in_use: true,
            device_id: 0, // Default, should be set by allocator
            metadata: AllocationMetadata::new(),
        }
    }

    /// Create allocation with specific device
    pub fn new_on_device(
        ptr: cust::DevicePointer<u8>,
        size: usize,
        size_class: usize,
        device_id: usize,
    ) -> Self {
        Self {
            ptr,
            size,
            size_class,
            allocation_time: Instant::now(),
            in_use: true,
            device_id,
            metadata: AllocationMetadata::new(),
        }
    }

    /// Get device pointer as raw pointer
    pub fn as_device_ptr(&self) -> cust::DevicePointer<u8> {
        self.ptr
    }

    /// Check if allocation is in use
    pub fn is_in_use(&self) -> bool {
        self.in_use
    }

    /// Mark allocation as in use
    pub fn mark_in_use(&mut self) {
        self.in_use = true;
    }

    /// Mark allocation as free
    pub fn mark_free(&mut self) {
        self.in_use = false;
    }

    /// Get age of allocation
    pub fn age(&self) -> std::time::Duration {
        Instant::now().duration_since(self.allocation_time)
    }
}

impl CudaMemoryAllocation for CudaAllocation {
    fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_raw_mut()
    }

    fn size(&self) -> usize {
        self.size
    }

    fn allocation_time(&self) -> Instant {
        self.allocation_time
    }

    fn is_valid(&self) -> bool {
        !self.ptr.as_raw().is_null() && self.size > 0
    }

    fn allocation_type(&self) -> AllocationType {
        AllocationType::Device
    }
}

// Implementation for UnifiedAllocation
impl UnifiedAllocation {
    /// Create new unified memory allocation
    pub fn new(ptr: *mut u8, size: usize) -> Self {
        Self {
            ptr,
            size,
            allocation_time: Instant::now(),
            preferred_location: PreferredLocation::Auto,
            access_hints: AccessHints::default(),
            migration_stats: MigrationStats::default(),
            metadata: AllocationMetadata::new(),
        }
    }

    /// Create unified allocation with preferred location
    pub fn new_with_preference(
        ptr: *mut u8,
        size: usize,
        preferred_location: PreferredLocation,
    ) -> Self {
        Self {
            ptr,
            size,
            allocation_time: Instant::now(),
            preferred_location,
            access_hints: AccessHints::default(),
            migration_stats: MigrationStats::default(),
            metadata: AllocationMetadata::new(),
        }
    }

    /// Get age of allocation
    pub fn age(&self) -> std::time::Duration {
        Instant::now().duration_since(self.allocation_time)
    }

    /// Update migration statistics
    pub fn record_migration(
        &mut self,
        from_device: bool,
        bytes: usize,
        duration: std::time::Duration,
    ) {
        if from_device {
            self.migration_stats.device_to_host_migrations += 1;
        } else {
            self.migration_stats.host_to_device_migrations += 1;
        }

        self.migration_stats.total_bytes_migrated += bytes as u64;

        // Update average migration time
        let total_migrations = self.migration_stats.host_to_device_migrations
            + self.migration_stats.device_to_host_migrations;
        let total_time =
            self.migration_stats.average_migration_time * (total_migrations - 1) as u32 + duration;
        self.migration_stats.average_migration_time = total_time / total_migrations as u32;

        self.migration_stats.last_migration = Some(Instant::now());
    }
}

impl CudaMemoryAllocation for UnifiedAllocation {
    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    fn size(&self) -> usize {
        self.size
    }

    fn allocation_time(&self) -> Instant {
        self.allocation_time
    }

    fn is_valid(&self) -> bool {
        !self.ptr.is_null() && self.size > 0
    }

    fn allocation_type(&self) -> AllocationType {
        AllocationType::Unified
    }
}

// Implementation for PinnedAllocation
impl PinnedAllocation {
    /// Create new pinned memory allocation
    pub fn new(ptr: *mut u8, size: usize) -> Self {
        Self {
            ptr,
            size,
            allocation_time: Instant::now(),
            usage_count: 0,
            is_mapped: false,
            device_ptr: None,
            mapping_flags: PinnedMemoryFlags::default(),
            metadata: AllocationMetadata::new(),
        }
    }

    /// Create pinned allocation with mapping
    pub fn new_with_mapping(
        ptr: *mut u8,
        size: usize,
        device_ptr: Option<cust::DevicePointer<u8>>,
        flags: PinnedMemoryFlags,
    ) -> Self {
        Self {
            ptr,
            size,
            allocation_time: Instant::now(),
            usage_count: 0,
            is_mapped: device_ptr.is_some(),
            device_ptr,
            mapping_flags: flags,
            metadata: AllocationMetadata::new(),
        }
    }

    /// Increment usage count
    pub fn increment_usage(&mut self) {
        self.usage_count += 1;
    }

    /// Get age of allocation
    pub fn age(&self) -> std::time::Duration {
        Instant::now().duration_since(self.allocation_time)
    }

    /// Check if allocation is device mapped
    pub fn is_mapped(&self) -> bool {
        self.is_mapped
    }

    /// Get device pointer if mapped
    pub fn device_ptr(&self) -> Option<cust::DevicePointer<u8>> {
        self.device_ptr
    }
}

impl CudaMemoryAllocation for PinnedAllocation {
    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    fn size(&self) -> usize {
        self.size
    }

    fn allocation_time(&self) -> Instant {
        self.allocation_time
    }

    fn is_valid(&self) -> bool {
        !self.ptr.is_null() && self.size > 0
    }

    fn allocation_type(&self) -> AllocationType {
        AllocationType::Pinned
    }
}

// Default implementations
impl Default for AccessHints {
    fn default() -> Self {
        Self {
            read_mostly: false,
            access_frequency: AccessFrequency::Medium,
            locality: DataLocality::Mixed,
            custom_hints: Vec::new(),
        }
    }
}

impl Default for MigrationStats {
    fn default() -> Self {
        Self {
            host_to_device_migrations: 0,
            device_to_host_migrations: 0,
            total_bytes_migrated: 0,
            average_migration_time: std::time::Duration::from_secs(0),
            last_migration: None,
        }
    }
}

impl Default for PinnedMemoryFlags {
    fn default() -> Self {
        Self {
            enable_mapping: false,
            portable: false,
            write_combining: false,
            raw_flags: 0,
        }
    }
}

impl AllocationMetadata {
    /// Create new metadata with default values
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static ALLOCATION_COUNTER: AtomicU64 = AtomicU64::new(1);

        Self {
            id: ALLOCATION_COUNTER.fetch_add(1, Ordering::Relaxed),
            tag: None,
            stack_trace: None,
            thread_id: std::thread::current().id().as_u64().get(),
            process_id: std::process::id(),
            alignment: 1,
            is_temporary: false,
            expected_lifetime: None,
            user_data: None,
        }
    }

    /// Create metadata with tag
    pub fn with_tag(tag: String) -> Self {
        let mut metadata = Self::new();
        metadata.tag = Some(tag);
        metadata
    }
}

impl Default for AllocationMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AllocationRequest {
    fn default() -> Self {
        Self {
            size: 0,
            alignment: None,
            allocation_type: AllocationType::Device,
            device_id: None,
            tag: None,
            is_temporary: false,
            expected_lifetime: None,
            priority: AllocationPriority::Normal,
        }
    }
}

impl Default for AllocationStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            active_allocations: 0,
            total_bytes_allocated: 0,
            current_bytes_allocated: 0,
            peak_bytes_allocated: 0,
            average_allocation_size: 0,
            success_rate: 1.0,
            cache_hit_rate: 0.0,
            average_allocation_time: std::time::Duration::from_secs(0),
            fragmentation_level: 0.0,
        }
    }
}

// Utility functions
pub fn size_class(size: usize) -> usize {
    // Round up to nearest power of 2, minimum 256 bytes
    const MIN_SIZE: usize = 256;
    if size <= MIN_SIZE {
        MIN_SIZE
    } else {
        (size - 1).next_power_of_two().max(MIN_SIZE)
    }
}

pub fn pinned_size_class(size: usize) -> usize {
    // Round up to nearest power of 2, minimum 4KB for pinned memory
    const MIN_SIZE: usize = 4096;
    if size <= MIN_SIZE {
        MIN_SIZE
    } else {
        (size - 1).next_power_of_two().max(MIN_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_metadata() {
        let metadata1 = AllocationMetadata::new();
        let metadata2 = AllocationMetadata::new();

        // IDs should be unique
        assert_ne!(metadata1.id, metadata2.id);
        assert_eq!(metadata1.alignment, 1);
        assert!(!metadata1.is_temporary);
    }

    #[test]
    fn test_size_class_calculation() {
        assert_eq!(size_class(100), 256);
        assert_eq!(size_class(256), 256);
        assert_eq!(size_class(300), 512);
        assert_eq!(size_class(1024), 1024);
        assert_eq!(size_class(1025), 2048);
    }

    #[test]
    fn test_pinned_size_class_calculation() {
        assert_eq!(pinned_size_class(1000), 4096);
        assert_eq!(pinned_size_class(4096), 4096);
        assert_eq!(pinned_size_class(5000), 8192);
        assert_eq!(pinned_size_class(8192), 8192);
        assert_eq!(pinned_size_class(8193), 16384);
    }

    #[test]
    fn test_access_hints_default() {
        let hints = AccessHints::default();
        assert!(!hints.read_mostly);
        assert_eq!(hints.access_frequency, AccessFrequency::Medium);
        assert_eq!(hints.locality, DataLocality::Mixed);
        assert!(hints.custom_hints.is_empty());
    }

    #[test]
    fn test_allocation_priority_ordering() {
        assert!(AllocationPriority::Critical > AllocationPriority::High);
        assert!(AllocationPriority::High > AllocationPriority::Normal);
        assert!(AllocationPriority::Normal > AllocationPriority::Low);
    }

    #[test]
    fn test_migration_stats() {
        let mut stats = MigrationStats::default();
        assert_eq!(stats.host_to_device_migrations, 0);
        assert_eq!(stats.device_to_host_migrations, 0);
        assert_eq!(stats.total_bytes_migrated, 0);
        assert!(stats.last_migration.is_none());
    }

    #[test]
    fn test_pinned_memory_flags() {
        let flags = PinnedMemoryFlags::default();
        assert!(!flags.enable_mapping);
        assert!(!flags.portable);
        assert!(!flags.write_combining);
        assert_eq!(flags.raw_flags, 0);
    }

    #[test]
    fn test_allocation_request() {
        let request = AllocationRequest {
            size: 1024,
            allocation_type: AllocationType::Device,
            tag: Some("test".to_string()),
            priority: AllocationPriority::High,
            ..Default::default()
        };

        assert_eq!(request.size, 1024);
        assert_eq!(request.allocation_type, AllocationType::Device);
        assert_eq!(request.tag, Some("test".to_string()));
        assert_eq!(request.priority, AllocationPriority::High);
    }
}
