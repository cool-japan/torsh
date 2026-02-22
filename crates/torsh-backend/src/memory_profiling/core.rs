//! Core Memory Profiling Types and Tracking
//!
//! This module provides the foundational types and infrastructure for comprehensive
//! memory profiling, including allocation tracking, lifecycle management, and
//! performance hint generation.

use crate::Device;
use std::time::Instant;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

// =============================================================================
// CORE MEMORY ALLOCATION TRACKING
// =============================================================================

/// Comprehensive memory allocation tracking information
///
/// This structure contains detailed information about a memory allocation,
/// including its source, usage patterns, and performance characteristics.
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    /// Address of allocated memory
    pub ptr: usize,

    /// Size of allocation in bytes
    pub size: usize,

    /// Allocation timestamp
    pub allocated_at: Instant,

    /// Allocation source (stack trace if available)
    pub source: AllocationSource,

    /// Memory type (device, host, unified, etc.)
    pub memory_type: MemoryType,

    /// Associated device
    pub device: Option<Device>,

    /// Usage statistics
    pub usage_stats: AllocationUsageStats,

    /// Lifetime tracking events
    pub lifetime_events: Vec<LifetimeEvent>,

    /// Performance optimization hints
    pub performance_hints: Vec<PerformanceHint>,
}

impl MemoryAllocation {
    /// Creates a new memory allocation record
    pub fn new(
        ptr: usize,
        size: usize,
        source: AllocationSource,
        memory_type: MemoryType,
        device: Option<Device>,
    ) -> Self {
        Self {
            ptr,
            size,
            allocated_at: Instant::now(),
            source,
            memory_type,
            device,
            usage_stats: AllocationUsageStats::default(),
            lifetime_events: Vec::new(),
            performance_hints: Vec::new(),
        }
    }

    /// Records an access to this allocation
    pub fn record_access(&mut self, read: bool, write: bool, bytes: u64) {
        self.usage_stats.access_count += 1;
        self.usage_stats.last_accessed = Some(Instant::now());

        if read {
            self.usage_stats.bytes_read += bytes;
        }
        if write {
            self.usage_stats.bytes_written += bytes;
        }

        // Record lifetime event
        self.lifetime_events.push(LifetimeEvent {
            timestamp: Instant::now(),
            event_type: LifetimeEventType::Accessed { read, write },
            details: format!("Access: {} bytes, read: {}, write: {}", bytes, read, write),
        });

        // Update access frequency
        self.update_access_frequency();
    }

    /// Updates the access frequency calculation
    fn update_access_frequency(&mut self) {
        let elapsed = self.allocated_at.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.usage_stats.access_frequency = self.usage_stats.access_count as f64 / elapsed;
        }
    }

    /// Adds a performance hint to this allocation
    pub fn add_performance_hint(&mut self, hint: PerformanceHint) {
        // Avoid duplicate hints of the same type
        if !self.performance_hints.iter().any(|h| std::mem::discriminant(&h.hint_type) == std::mem::discriminant(&hint.hint_type)) {
            self.performance_hints.push(hint);
        }
    }

    /// Records a lifetime event
    pub fn record_lifetime_event(&mut self, event_type: LifetimeEventType, details: String) {
        self.lifetime_events.push(LifetimeEvent {
            timestamp: Instant::now(),
            event_type,
            details,
        });
    }

    /// Calculates the age of this allocation
    pub fn age(&self) -> std::time::Duration {
        self.allocated_at.elapsed()
    }

    /// Calculates bytes per second throughput
    pub fn throughput(&self) -> f64 {
        let elapsed = self.allocated_at.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            (self.usage_stats.bytes_read + self.usage_stats.bytes_written) as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Checks if this allocation appears to be unused
    pub fn is_likely_unused(&self) -> bool {
        // Consider unused if no accesses in the last 60 seconds and has been allocated for more than 10 seconds
        if let Some(last_access) = self.usage_stats.last_accessed {
            last_access.elapsed().as_secs() > 60 && self.age().as_secs() > 10
        } else {
            self.age().as_secs() > 10
        }
    }
}

/// Memory allocation source information
///
/// Tracks the source context where a memory allocation occurred,
/// including function, location, and thread information.
#[derive(Debug, Clone)]
pub struct AllocationSource {
    /// Function name where allocation occurred
    pub function: String,

    /// File and line number
    pub location: Option<(String, u32)>,

    /// Call stack depth
    pub stack_depth: usize,

    /// Thread ID
    pub thread_id: u64,

    /// Allocation context
    pub context: AllocationContext,
}

impl AllocationSource {
    /// Creates a new allocation source
    pub fn new(
        function: String,
        location: Option<(String, u32)>,
        context: AllocationContext,
    ) -> Self {
        Self {
            function,
            location,
            stack_depth: 0, // Would be filled by stack trace analysis
            thread_id: std::thread::current().id().as_u64().get(),
            context,
        }
    }

    /// Gets a human-readable description of the source
    pub fn description(&self) -> String {
        let location_str = if let Some((file, line)) = &self.location {
            format!(" ({}:{})", file, line)
        } else {
            String::new()
        };

        format!("{}{} - {}", self.function, location_str, self.context.description())
    }
}

/// Allocation context information
///
/// Provides detailed context about why and how a memory allocation occurred,
/// enabling better optimization and debugging.
#[derive(Debug, Clone)]
pub enum AllocationContext {
    /// Tensor operation allocation
    TensorOperation {
        operation_name: String,
        tensor_shape: Vec<usize>,
        data_type: String,
    },

    /// Kernel execution scratch space
    KernelScratch {
        kernel_name: String,
        scratch_type: String,
    },

    /// Intermediate computation buffer
    IntermediateBuffer {
        computation_graph_id: String,
        buffer_purpose: String,
    },

    /// Model weights or parameters
    ModelParameters {
        model_name: String,
        parameter_name: String,
    },

    /// User-requested allocation
    UserAllocation { request_id: String },

    /// Internal backend allocation
    InternalAllocation { purpose: String },

    /// Cache allocation
    CacheAllocation {
        cache_type: String,
        cache_level: usize,
    },

    /// Memory pool allocation
    PoolAllocation {
        pool_name: String,
        pool_type: String,
    },
}

impl AllocationContext {
    /// Gets a human-readable description of the context
    pub fn description(&self) -> String {
        match self {
            AllocationContext::TensorOperation { operation_name, tensor_shape, data_type } => {
                format!("Tensor {} {:?} ({})", operation_name, tensor_shape, data_type)
            },
            AllocationContext::KernelScratch { kernel_name, scratch_type } => {
                format!("Kernel {} scratch ({})", kernel_name, scratch_type)
            },
            AllocationContext::IntermediateBuffer { computation_graph_id, buffer_purpose } => {
                format!("Buffer {} ({})", computation_graph_id, buffer_purpose)
            },
            AllocationContext::ModelParameters { model_name, parameter_name } => {
                format!("Model {} parameter {}", model_name, parameter_name)
            },
            AllocationContext::UserAllocation { request_id } => {
                format!("User allocation {}", request_id)
            },
            AllocationContext::InternalAllocation { purpose } => {
                format!("Internal: {}", purpose)
            },
            AllocationContext::CacheAllocation { cache_type, cache_level } => {
                format!("Cache L{} ({})", cache_level, cache_type)
            },
            AllocationContext::PoolAllocation { pool_name, pool_type } => {
                format!("Pool {} ({})", pool_name, pool_type)
            },
        }
    }

    /// Checks if this context represents a temporary allocation
    pub fn is_temporary(&self) -> bool {
        matches!(self,
            AllocationContext::KernelScratch { .. } |
            AllocationContext::IntermediateBuffer { .. } |
            AllocationContext::CacheAllocation { .. }
        )
    }

    /// Gets the estimated lifetime category
    pub fn lifetime_category(&self) -> LifetimeCategory {
        match self {
            AllocationContext::TensorOperation { .. } => LifetimeCategory::Short,
            AllocationContext::KernelScratch { .. } => LifetimeCategory::VeryShort,
            AllocationContext::IntermediateBuffer { .. } => LifetimeCategory::Short,
            AllocationContext::ModelParameters { .. } => LifetimeCategory::Long,
            AllocationContext::UserAllocation { .. } => LifetimeCategory::Medium,
            AllocationContext::InternalAllocation { .. } => LifetimeCategory::Medium,
            AllocationContext::CacheAllocation { .. } => LifetimeCategory::Medium,
            AllocationContext::PoolAllocation { .. } => LifetimeCategory::Long,
        }
    }
}

/// Expected lifetime categories for allocations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifetimeCategory {
    VeryShort, // < 1 second
    Short,     // 1-60 seconds
    Medium,    // 1-60 minutes
    Long,      // > 1 hour
}

// =============================================================================
// MEMORY TYPE CLASSIFICATION
// =============================================================================

/// Memory type classification
///
/// Categorizes different types of memory with their characteristics
/// and access patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// Device (GPU) memory
    Device,

    /// Host (CPU) memory
    Host,

    /// Unified memory (accessible by both CPU and GPU)
    Unified,

    /// Pinned/page-locked host memory
    Pinned,

    /// Texture memory
    Texture,

    /// Constant memory
    Constant,

    /// Shared memory
    Shared,

    /// Memory-mapped file
    MemoryMapped,
}

impl MemoryType {
    /// Gets the typical bandwidth characteristics of this memory type
    pub fn bandwidth_characteristics(&self) -> BandwidthCharacteristics {
        match self {
            MemoryType::Device => BandwidthCharacteristics {
                peak_bandwidth_gbps: 1000.0, // Modern GPU memory
                typical_latency_ns: 100.0,
                access_granularity: 128, // bytes
            },
            MemoryType::Host => BandwidthCharacteristics {
                peak_bandwidth_gbps: 100.0, // DDR4/DDR5
                typical_latency_ns: 50.0,
                access_granularity: 64,
            },
            MemoryType::Unified => BandwidthCharacteristics {
                peak_bandwidth_gbps: 200.0, // Varies by implementation
                typical_latency_ns: 150.0,
                access_granularity: 128,
            },
            MemoryType::Pinned => BandwidthCharacteristics {
                peak_bandwidth_gbps: 100.0,
                typical_latency_ns: 40.0, // Slightly better than regular host
                access_granularity: 64,
            },
            MemoryType::Texture => BandwidthCharacteristics {
                peak_bandwidth_gbps: 800.0, // High bandwidth, cached
                typical_latency_ns: 200.0,
                access_granularity: 16,
            },
            MemoryType::Constant => BandwidthCharacteristics {
                peak_bandwidth_gbps: 200.0, // Broadcast bandwidth
                typical_latency_ns: 80.0,
                access_granularity: 4,
            },
            MemoryType::Shared => BandwidthCharacteristics {
                peak_bandwidth_gbps: 2000.0, // Very high, but small size
                typical_latency_ns: 20.0,
                access_granularity: 4,
            },
            MemoryType::MemoryMapped => BandwidthCharacteristics {
                peak_bandwidth_gbps: 10.0, // Storage limited
                typical_latency_ns: 1000.0,
                access_granularity: 4096, // Page size
            },
        }
    }

    /// Checks if this memory type is device-accessible
    pub fn is_device_accessible(&self) -> bool {
        matches!(self,
            MemoryType::Device |
            MemoryType::Unified |
            MemoryType::Pinned |
            MemoryType::Texture |
            MemoryType::Constant |
            MemoryType::Shared
        )
    }

    /// Checks if this memory type is host-accessible
    pub fn is_host_accessible(&self) -> bool {
        matches!(self,
            MemoryType::Host |
            MemoryType::Unified |
            MemoryType::Pinned |
            MemoryType::MemoryMapped
        )
    }
}

/// Bandwidth and latency characteristics for memory types
#[derive(Debug, Clone)]
pub struct BandwidthCharacteristics {
    /// Peak theoretical bandwidth in GB/s
    pub peak_bandwidth_gbps: f64,

    /// Typical access latency in nanoseconds
    pub typical_latency_ns: f64,

    /// Optimal access granularity in bytes
    pub access_granularity: usize,
}

// =============================================================================
// USAGE STATISTICS AND PERFORMANCE TRACKING
// =============================================================================

/// Usage statistics for individual allocations
///
/// Tracks detailed access patterns and performance metrics
/// for memory allocations.
#[derive(Debug, Clone, Default)]
pub struct AllocationUsageStats {
    /// Number of times accessed
    pub access_count: u64,

    /// Total bytes read
    pub bytes_read: u64,

    /// Total bytes written
    pub bytes_written: u64,

    /// Last access timestamp
    pub last_accessed: Option<Instant>,

    /// Access frequency (accesses per second)
    pub access_frequency: f64,

    /// Memory bandwidth utilization (0.0 to 1.0)
    pub bandwidth_utilization: f64,

    /// Cache hit/miss statistics
    pub cache_stats: CacheStats,

    /// Access pattern classification
    pub access_pattern: AccessPatternType,
}

impl AllocationUsageStats {
    /// Calculates the total I/O throughput
    pub fn total_throughput(&self) -> u64 {
        self.bytes_read + self.bytes_written
    }

    /// Calculates the read/write ratio
    pub fn read_write_ratio(&self) -> f64 {
        if self.bytes_written > 0 {
            self.bytes_read as f64 / self.bytes_written as f64
        } else if self.bytes_read > 0 {
            f64::INFINITY
        } else {
            0.0
        }
    }

    /// Checks if this allocation is read-heavy
    pub fn is_read_heavy(&self) -> bool {
        self.read_write_ratio() > 3.0
    }

    /// Checks if this allocation is write-heavy
    pub fn is_write_heavy(&self) -> bool {
        self.read_write_ratio() < 0.3
    }
}

/// Cache performance statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// L1 cache hits
    pub l1_hits: u64,

    /// L1 cache misses
    pub l1_misses: u64,

    /// L2 cache hits
    pub l2_hits: u64,

    /// L2 cache misses
    pub l2_misses: u64,

    /// TLB hits
    pub tlb_hits: u64,

    /// TLB misses
    pub tlb_misses: u64,
}

impl CacheStats {
    /// Calculates L1 cache hit rate
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l1_misses;
        if total > 0 {
            self.l1_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Calculates L2 cache hit rate
    pub fn l2_hit_rate(&self) -> f64 {
        let total = self.l2_hits + self.l2_misses;
        if total > 0 {
            self.l2_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Calculates TLB hit rate
    pub fn tlb_hit_rate(&self) -> f64 {
        let total = self.tlb_hits + self.tlb_misses;
        if total > 0 {
            self.tlb_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Calculates overall cache efficiency score
    pub fn efficiency_score(&self) -> f64 {
        (self.l1_hit_rate() * 0.5) + (self.l2_hit_rate() * 0.3) + (self.tlb_hit_rate() * 0.2)
    }
}

/// Access pattern classification
#[derive(Debug, Clone, Default)]
pub enum AccessPatternType {
    #[default]
    Unknown,
    Sequential,
    Random,
    Strided { stride: usize },
    Temporal, // Same locations accessed repeatedly
    Sparse,   // Few scattered accesses
}

// =============================================================================
// LIFETIME EVENTS AND TRACKING
// =============================================================================

/// Lifetime events for memory allocations
#[derive(Debug, Clone)]
pub struct LifetimeEvent {
    /// Event timestamp
    pub timestamp: Instant,

    /// Event type
    pub event_type: LifetimeEventType,

    /// Event details
    pub details: String,
}

impl LifetimeEvent {
    /// Creates a new lifetime event
    pub fn new(event_type: LifetimeEventType, details: String) -> Self {
        Self {
            timestamp: Instant::now(),
            event_type,
            details,
        }
    }

    /// Gets the age of this event
    pub fn age(&self) -> std::time::Duration {
        self.timestamp.elapsed()
    }
}

/// Types of lifetime events
#[derive(Debug, Clone)]
pub enum LifetimeEventType {
    /// Memory was allocated
    Allocated,

    /// Memory was accessed (read/write)
    Accessed { read: bool, write: bool },

    /// Memory was copied to/from
    Copied { source: bool, destination: bool },

    /// Memory was resized
    Resized { old_size: usize, new_size: usize },

    /// Memory was deallocated
    Deallocated,

    /// Memory pressure event occurred
    MemoryPressure { pressure_level: PressureLevel },

    /// Memory was defragmented
    Defragmented,

    /// Cache flush occurred
    CacheFlushed,

    /// Memory was swapped to storage
    SwappedOut,

    /// Memory was swapped back from storage
    SwappedIn,
}

/// Pressure level indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PressureLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
}

impl PressureLevel {
    /// Gets a numeric representation of the pressure level
    pub fn as_f64(&self) -> f64 {
        match self {
            PressureLevel::None => 0.0,
            PressureLevel::Low => 0.25,
            PressureLevel::Medium => 0.5,
            PressureLevel::High => 0.75,
            PressureLevel::Critical => 1.0,
        }
    }

    /// Creates a pressure level from a numeric value (0.0 to 1.0)
    pub fn from_f64(value: f64) -> Self {
        if value >= 0.9 {
            PressureLevel::Critical
        } else if value >= 0.7 {
            PressureLevel::High
        } else if value >= 0.4 {
            PressureLevel::Medium
        } else if value >= 0.1 {
            PressureLevel::Low
        } else {
            PressureLevel::None
        }
    }
}

// =============================================================================
// PERFORMANCE HINTS AND OPTIMIZATION
// =============================================================================

/// Performance hints for memory allocations
#[derive(Debug, Clone)]
pub struct PerformanceHint {
    /// Hint type
    pub hint_type: PerformanceHintType,

    /// Severity level
    pub severity: HintSeverity,

    /// Description
    pub description: String,

    /// Suggested action
    pub suggested_action: String,

    /// Potential performance impact (0.0 to 1.0)
    pub impact_estimate: f64,

    /// Confidence in this hint (0.0 to 1.0)
    pub confidence: f64,
}

impl PerformanceHint {
    /// Creates a new performance hint
    pub fn new(
        hint_type: PerformanceHintType,
        severity: HintSeverity,
        description: String,
        suggested_action: String,
        impact_estimate: f64,
    ) -> Self {
        Self {
            hint_type,
            severity,
            description,
            suggested_action,
            impact_estimate: impact_estimate.clamp(0.0, 1.0),
            confidence: 0.8, // Default confidence
        }
    }

    /// Gets a priority score for this hint
    pub fn priority_score(&self) -> f64 {
        let severity_weight = match self.severity {
            HintSeverity::Info => 0.3,
            HintSeverity::Warning => 0.6,
            HintSeverity::Critical => 1.0,
        };

        severity_weight * self.impact_estimate * self.confidence
    }
}

/// Types of performance hints
#[derive(Debug, Clone)]
pub enum PerformanceHintType {
    /// Memory access pattern is suboptimal
    SuboptimalAccessPattern,

    /// Allocation size is inefficient
    InefficientSize,

    /// Memory type is not optimal for usage
    SuboptimalMemoryType,

    /// Frequent allocation/deallocation detected
    ExcessiveAllocations,

    /// Memory fragmentation detected
    Fragmentation,

    /// Unused memory detected
    UnusedMemory,

    /// Cache-unfriendly access pattern
    PoorCacheLocality,

    /// Memory bandwidth underutilization
    BandwidthUnderutilization,

    /// Memory alignment issues
    AlignmentIssues,

    /// Prefetching opportunity
    PrefetchingOpportunity,

    /// Memory coalescing opportunity
    CoalescingOpportunity,

    /// Memory pool optimization
    PoolOptimization,
}

/// Severity levels for performance hints
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HintSeverity {
    Info,
    Warning,
    Critical,
}

impl HintSeverity {
    /// Gets a numeric representation of the severity
    pub fn as_f64(&self) -> f64 {
        match self {
            HintSeverity::Info => 0.3,
            HintSeverity::Warning => 0.6,
            HintSeverity::Critical => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_allocation_creation() {
        let source = AllocationSource::new(
            "test_function".to_string(),
            Some(("test.rs".to_string(), 42)),
            AllocationContext::UserAllocation { request_id: "test".to_string() },
        );

        let allocation = MemoryAllocation::new(
            0x1000,
            1024,
            source,
            MemoryType::Host,
            None,
        );

        assert_eq!(allocation.ptr, 0x1000);
        assert_eq!(allocation.size, 1024);
        assert_eq!(allocation.memory_type, MemoryType::Host);
    }

    #[test]
    fn test_allocation_access_tracking() {
        let source = AllocationSource::new(
            "test".to_string(),
            None,
            AllocationContext::InternalAllocation { purpose: "test".to_string() },
        );

        let mut allocation = MemoryAllocation::new(
            0x1000,
            1024,
            source,
            MemoryType::Host,
            None,
        );

        allocation.record_access(true, false, 512);
        assert_eq!(allocation.usage_stats.access_count, 1);
        assert_eq!(allocation.usage_stats.bytes_read, 512);
        assert_eq!(allocation.usage_stats.bytes_written, 0);
    }

    #[test]
    fn test_cache_stats() {
        let mut stats = CacheStats::default();
        stats.l1_hits = 80;
        stats.l1_misses = 20;

        assert_eq!(stats.l1_hit_rate(), 0.8);
    }

    #[test]
    fn test_pressure_level_conversion() {
        assert_eq!(PressureLevel::from_f64(0.95), PressureLevel::Critical);
        assert_eq!(PressureLevel::from_f64(0.75), PressureLevel::High);
        assert_eq!(PressureLevel::from_f64(0.5), PressureLevel::Medium);
        assert_eq!(PressureLevel::from_f64(0.2), PressureLevel::Low);
        assert_eq!(PressureLevel::from_f64(0.05), PressureLevel::None);
    }

    #[test]
    fn test_performance_hint() {
        let hint = PerformanceHint::new(
            PerformanceHintType::UnusedMemory,
            HintSeverity::Warning,
            "Memory appears unused".to_string(),
            "Consider deallocating".to_string(),
            0.6,
        );

        assert!(hint.priority_score() > 0.0);
    }

    #[test]
    fn test_memory_type_characteristics() {
        let device_chars = MemoryType::Device.bandwidth_characteristics();
        let host_chars = MemoryType::Host.bandwidth_characteristics();

        assert!(device_chars.peak_bandwidth_gbps > host_chars.peak_bandwidth_gbps);
        assert!(MemoryType::Device.is_device_accessible());
        assert!(MemoryType::Host.is_host_accessible());
    }
}