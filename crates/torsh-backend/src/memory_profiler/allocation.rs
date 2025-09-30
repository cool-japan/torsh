//! Memory allocation tracking and management
//!
//! This module provides comprehensive allocation tracking capabilities including:
//! - Individual allocation metadata and statistics
//! - Allocation source tracking and context information
//! - Access pattern analysis and performance optimization
//! - Lifetime event tracking and cache performance monitoring
//! - Performance hint generation for optimization suggestions

use crate::Device;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Memory allocation tracking information
///
/// Tracks detailed metadata for each memory allocation including size, location,
/// usage statistics, and performance characteristics.
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

    /// Lifetime tracking
    pub lifetime_events: Vec<LifetimeEvent>,

    /// Performance hints
    pub performance_hints: Vec<PerformanceHint>,
}

/// Memory allocation source information
///
/// Captures the context and origin of memory allocations for debugging
/// and optimization purposes.
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

/// Allocation context information
///
/// Provides detailed context about the purpose and nature of memory allocations
/// to enable better optimization and debugging.
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
}

/// Memory type classification
///
/// Categorizes different types of memory for optimization and tracking purposes.
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

/// Usage statistics for individual allocations
///
/// Tracks detailed usage patterns and performance characteristics
/// for each memory allocation.
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

    /// Access frequency pattern
    pub access_frequency: f64,

    /// Memory bandwidth utilization
    pub bandwidth_utilization: f64,

    /// Cache hit/miss statistics
    pub cache_stats: CacheStats,
}

/// Cache performance statistics
///
/// Provides detailed cache performance metrics for memory accesses.
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

/// Lifetime events for memory allocations
///
/// Tracks significant events in the lifetime of memory allocations
/// for debugging and optimization analysis.
#[derive(Debug, Clone)]
pub struct LifetimeEvent {
    /// Event timestamp
    pub timestamp: Instant,

    /// Event type
    pub event_type: LifetimeEventType,

    /// Event details
    pub details: String,
}

/// Types of lifetime events
///
/// Categorizes different events that can occur during an allocation's lifetime.
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
}

/// Memory pressure levels
///
/// Indicates the severity of memory pressure conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PressureLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
}

impl Default for PressureLevel {
    fn default() -> Self {
        PressureLevel::None
    }
}

/// Performance hints for memory allocations
///
/// Provides actionable performance optimization suggestions based on
/// allocation usage patterns and characteristics.
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

    /// Potential performance impact
    pub impact_estimate: f64,
}

/// Types of performance hints
///
/// Categorizes different types of performance optimization opportunities.
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
}

/// Severity levels for performance hints
///
/// Indicates the importance and urgency of performance optimization suggestions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HintSeverity {
    Info,
    Warning,
    Critical,
}

/// Memory access pattern tracking
///
/// Analyzes and tracks memory access patterns to identify optimization
/// opportunities and performance characteristics.
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Access timestamps
    pub access_times: VecDeque<Instant>,

    /// Access sizes
    pub access_sizes: VecDeque<usize>,

    /// Access types (read/write)
    pub access_types: VecDeque<AccessType>,

    /// Sequential access score
    pub sequential_score: f64,

    /// Random access score
    pub random_score: f64,

    /// Temporal locality score
    pub temporal_locality: f64,

    /// Spatial locality score
    pub spatial_locality: f64,

    /// Access frequency
    pub frequency: f64,

    /// Last pattern analysis
    pub last_analysis: Option<Instant>,
}

/// Memory access type
///
/// Categorizes different types of memory access operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
    ReadWrite,
}

/// Allocation tracker for managing multiple allocations
///
/// Provides centralized tracking and management of memory allocations
/// with support for performance analysis and optimization.
#[derive(Debug)]
pub struct AllocationTracker {
    /// Active allocations (using pointer addresses as keys)
    allocations: HashMap<usize, MemoryAllocation>,

    /// Access patterns (using pointer addresses as keys)
    access_patterns: HashMap<usize, AccessPattern>,

    /// Total number of allocations tracked
    total_allocations: u64,

    /// Total bytes allocated
    total_bytes: u64,

    /// Performance hints generated
    performance_hints: Vec<PerformanceHint>,
}

impl MemoryAllocation {
    /// Create a new memory allocation record
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
            lifetime_events: vec![LifetimeEvent {
                timestamp: Instant::now(),
                event_type: LifetimeEventType::Allocated,
                details: format!("Allocated {} bytes at {:p}", size, ptr as *const u8),
            }],
            performance_hints: Vec::new(),
        }
    }

    /// Record a memory access event
    pub fn record_access(&mut self, access_type: AccessType, bytes: usize) {
        let now = Instant::now();

        self.usage_stats.access_count += 1;
        self.usage_stats.last_accessed = Some(now);

        match access_type {
            AccessType::Read => self.usage_stats.bytes_read += bytes as u64,
            AccessType::Write => self.usage_stats.bytes_written += bytes as u64,
            AccessType::ReadWrite => {
                self.usage_stats.bytes_read += bytes as u64;
                self.usage_stats.bytes_written += bytes as u64;
            }
        }

        // Record lifetime event
        self.lifetime_events.push(LifetimeEvent {
            timestamp: now,
            event_type: LifetimeEventType::Accessed {
                read: matches!(access_type, AccessType::Read | AccessType::ReadWrite),
                write: matches!(access_type, AccessType::Write | AccessType::ReadWrite),
            },
            details: format!("Accessed {} bytes ({:?})", bytes, access_type),
        });

        // Update access frequency
        self.update_access_frequency();
    }

    /// Add a performance hint
    pub fn add_performance_hint(&mut self, hint: PerformanceHint) {
        self.performance_hints.push(hint);
    }

    /// Get allocation age
    pub fn age(&self) -> Duration {
        Instant::now().duration_since(self.allocated_at)
    }

    /// Check if allocation is active (not deallocated)
    pub fn is_active(&self) -> bool {
        !self
            .lifetime_events
            .iter()
            .any(|event| matches!(event.event_type, LifetimeEventType::Deallocated))
    }

    /// Get total bytes accessed
    pub fn total_bytes_accessed(&self) -> u64 {
        self.usage_stats.bytes_read + self.usage_stats.bytes_written
    }

    /// Update access frequency based on recent accesses
    fn update_access_frequency(&mut self) {
        let now = Instant::now();
        let age = now.duration_since(self.allocated_at).as_secs_f64();

        if age > 0.0 {
            self.usage_stats.access_frequency = self.usage_stats.access_count as f64 / age;
        }
    }
}

impl AllocationSource {
    /// Create a new allocation source
    pub fn new(
        function: String,
        location: Option<(String, u32)>,
        thread_id: u64,
        context: AllocationContext,
    ) -> Self {
        Self {
            function,
            location,
            stack_depth: 0, // Would be filled by stack trace collection
            thread_id,
            context,
        }
    }

    /// Get a human-readable description of the allocation source
    pub fn description(&self) -> String {
        let location_str = if let Some((file, line)) = &self.location {
            format!(" at {}:{}", file, line)
        } else {
            String::new()
        };

        format!("{}{}", self.function, location_str)
    }
}

impl AccessPattern {
    /// Create a new access pattern tracker
    pub fn new() -> Self {
        Self {
            access_times: VecDeque::new(),
            access_sizes: VecDeque::new(),
            access_types: VecDeque::new(),
            sequential_score: 0.0,
            random_score: 0.0,
            temporal_locality: 0.0,
            spatial_locality: 0.0,
            frequency: 0.0,
            last_analysis: None,
        }
    }

    /// Record a new memory access
    pub fn record_access(&mut self, access_type: AccessType, size: usize) {
        let now = Instant::now();

        self.access_times.push_back(now);
        self.access_sizes.push_back(size);
        self.access_types.push_back(access_type);

        // Keep only recent accesses (e.g., last 1000)
        const MAX_TRACKED_ACCESSES: usize = 1000;
        if self.access_times.len() > MAX_TRACKED_ACCESSES {
            self.access_times.pop_front();
            self.access_sizes.pop_front();
            self.access_types.pop_front();
        }

        // Update frequency
        self.update_frequency();
    }

    /// Analyze access patterns and update scores
    pub fn analyze_patterns(&mut self) {
        if self.access_times.len() < 2 {
            return;
        }

        self.analyze_sequentiality();
        self.analyze_locality();
        self.last_analysis = Some(Instant::now());
    }

    /// Check if pattern analysis is needed
    pub fn needs_analysis(&self) -> bool {
        const ANALYSIS_INTERVAL: Duration = Duration::from_secs(30);

        match self.last_analysis {
            Some(last) => Instant::now().duration_since(last) > ANALYSIS_INTERVAL,
            None => self.access_times.len() >= 10,
        }
    }

    /// Update access frequency
    fn update_frequency(&mut self) {
        if self.access_times.len() < 2 {
            return;
        }

        let first = self.access_times.front().unwrap();
        let last = self.access_times.back().unwrap();
        let duration = last.duration_since(*first).as_secs_f64();

        if duration > 0.0 {
            self.frequency = self.access_times.len() as f64 / duration;
        }
    }

    /// Analyze sequential vs random access patterns
    fn analyze_sequentiality(&mut self) {
        if self.access_sizes.len() < 3 {
            return;
        }

        let mut sequential_count = 0;
        let mut total_comparisons = 0;

        let access_sizes_vec: Vec<_> = self.access_sizes.iter().collect();
        for window in access_sizes_vec.windows(2) {
            let diff = if *window[1] > *window[0] {
                *window[1] - *window[0]
            } else {
                *window[0] - *window[1]
            };

            // Consider sequential if accesses are within reasonable stride
            if diff <= *window[0] {
                sequential_count += 1;
            }
            total_comparisons += 1;
        }

        if total_comparisons > 0 {
            self.sequential_score = sequential_count as f64 / total_comparisons as f64;
            self.random_score = 1.0 - self.sequential_score;
        }
    }

    /// Analyze temporal and spatial locality
    fn analyze_locality(&mut self) {
        if self.access_times.len() < 3 {
            return;
        }

        // Analyze temporal locality (recent accesses)
        let recent_window = Duration::from_secs(1);
        let now = Instant::now();
        let recent_accesses = self
            .access_times
            .iter()
            .filter(|&time| now.duration_since(*time) < recent_window)
            .count();

        self.temporal_locality = recent_accesses as f64 / self.access_times.len() as f64;

        // Analyze spatial locality (similar access sizes)
        let mut locality_score = 0.0;
        let mut comparisons = 0;

        let access_sizes_vec2: Vec<_> = self.access_sizes.iter().collect();
        for window in access_sizes_vec2.windows(3) {
            let size_var = ((*window[0] as f64 - *window[1] as f64).powi(2)
                + (*window[1] as f64 - *window[2] as f64).powi(2))
                / 2.0;

            // Lower variance indicates better spatial locality
            locality_score += 1.0 / (1.0 + size_var);
            comparisons += 1;
        }

        if comparisons > 0 {
            self.spatial_locality = locality_score / comparisons as f64;
        }
    }
}

impl AllocationTracker {
    /// Create a new allocation tracker
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            access_patterns: HashMap::new(),
            total_allocations: 0,
            total_bytes: 0,
            performance_hints: Vec::new(),
        }
    }

    /// Track a new allocation
    pub fn track_allocation(&mut self, allocation: MemoryAllocation) {
        self.total_allocations += 1;
        self.total_bytes += allocation.size as u64;

        // Initialize access pattern
        self.access_patterns
            .insert(allocation.ptr, AccessPattern::new());

        self.allocations.insert(allocation.ptr, allocation);
    }

    /// Record access to an allocation
    pub fn record_access(&mut self, ptr: usize, access_type: AccessType, bytes: usize) {
        // Update allocation stats
        if let Some(allocation) = self.allocations.get_mut(&ptr) {
            allocation.record_access(access_type, bytes);
        }

        // Update access pattern
        if let Some(pattern) = self.access_patterns.get_mut(&ptr) {
            pattern.record_access(access_type, bytes);

            if pattern.needs_analysis() {
                pattern.analyze_patterns();
                self.generate_performance_hints(ptr);
            }
        }
    }

    /// Remove an allocation (when deallocated)
    pub fn untrack_allocation(&mut self, ptr: usize) {
        if let Some(mut allocation) = self.allocations.remove(&ptr) {
            // Record deallocation event
            allocation.lifetime_events.push(LifetimeEvent {
                timestamp: Instant::now(),
                event_type: LifetimeEventType::Deallocated,
                details: "Memory deallocated".to_string(),
            });

            self.total_bytes = self.total_bytes.saturating_sub(allocation.size as u64);
        }

        self.access_patterns.remove(&ptr);
    }

    /// Get allocation by pointer
    pub fn get_allocation(&self, ptr: usize) -> Option<&MemoryAllocation> {
        self.allocations.get(&ptr)
    }

    /// Get access pattern by pointer
    pub fn get_access_pattern(&self, ptr: usize) -> Option<&AccessPattern> {
        self.access_patterns.get(&ptr)
    }

    /// Get all active allocations
    pub fn active_allocations(&self) -> impl Iterator<Item = &MemoryAllocation> {
        self.allocations.values().filter(|alloc| alloc.is_active())
    }

    /// Get total memory usage
    pub fn total_memory_usage(&self) -> usize {
        self.allocations
            .values()
            .filter(|alloc| alloc.is_active())
            .map(|alloc| alloc.size)
            .sum()
    }

    /// Generate performance hints for a specific allocation
    fn generate_performance_hints(&mut self, ptr: usize) {
        let (allocation, pattern) =
            match (self.allocations.get(&ptr), self.access_patterns.get(&ptr)) {
                (Some(alloc), Some(pat)) => (alloc, pat),
                _ => return,
            };

        let mut hints = Vec::new();

        // Check for poor cache locality
        if pattern.spatial_locality < 0.3 {
            hints.push(PerformanceHint {
                hint_type: PerformanceHintType::PoorCacheLocality,
                severity: HintSeverity::Warning,
                description: "Poor spatial locality detected in memory accesses".to_string(),
                suggested_action: "Consider reorganizing data layout or access patterns"
                    .to_string(),
                impact_estimate: 0.2,
            });
        }

        // Check for random access patterns
        if pattern.random_score > 0.7 {
            hints.push(PerformanceHint {
                hint_type: PerformanceHintType::SuboptimalAccessPattern,
                severity: HintSeverity::Info,
                description: "Random access pattern detected".to_string(),
                suggested_action: "Consider prefetching or data reorganization".to_string(),
                impact_estimate: 0.15,
            });
        }

        // Check for unused memory
        if allocation.usage_stats.access_count == 0 && allocation.age() > Duration::from_secs(60) {
            hints.push(PerformanceHint {
                hint_type: PerformanceHintType::UnusedMemory,
                severity: HintSeverity::Warning,
                description: "Memory allocated but never accessed".to_string(),
                suggested_action: "Consider deallocating unused memory".to_string(),
                impact_estimate: 0.1,
            });
        }

        // Add hints to the global list
        self.performance_hints.extend(hints);
    }

    /// Get all performance hints
    pub fn performance_hints(&self) -> &[PerformanceHint] {
        &self.performance_hints
    }

    /// Clear old performance hints
    pub fn clear_old_hints(&mut self) {
        // In a real implementation, we'd filter by timestamp
        self.performance_hints.clear();
    }
}

impl Default for AllocationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AccessPattern {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryType::Device => write!(f, "Device"),
            MemoryType::Host => write!(f, "Host"),
            MemoryType::Unified => write!(f, "Unified"),
            MemoryType::Pinned => write!(f, "Pinned"),
            MemoryType::Texture => write!(f, "Texture"),
            MemoryType::Constant => write!(f, "Constant"),
            MemoryType::Shared => write!(f, "Shared"),
            MemoryType::MemoryMapped => write!(f, "MemoryMapped"),
        }
    }
}

impl std::fmt::Display for AccessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccessType::Read => write!(f, "Read"),
            AccessType::Write => write!(f, "Write"),
            AccessType::ReadWrite => write!(f, "ReadWrite"),
        }
    }
}

impl std::fmt::Display for PressureLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PressureLevel::None => write!(f, "None"),
            PressureLevel::Low => write!(f, "Low"),
            PressureLevel::Medium => write!(f, "Medium"),
            PressureLevel::High => write!(f, "High"),
            PressureLevel::Critical => write!(f, "Critical"),
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
            123,
            AllocationContext::UserAllocation {
                request_id: "test".to_string(),
            },
        );

        let allocation = MemoryAllocation::new(0x1000, 1024, source, MemoryType::Host, None);

        assert_eq!(allocation.ptr, 0x1000);
        assert_eq!(allocation.size, 1024);
        assert_eq!(allocation.memory_type, MemoryType::Host);
        assert!(allocation.is_active());
    }

    #[test]
    fn test_allocation_access_tracking() {
        let source = AllocationSource::new(
            "test_function".to_string(),
            None,
            123,
            AllocationContext::UserAllocation {
                request_id: "test".to_string(),
            },
        );

        let mut allocation = MemoryAllocation::new(0x1000, 1024, source, MemoryType::Host, None);

        allocation.record_access(AccessType::Read, 512);
        allocation.record_access(AccessType::Write, 256);

        assert_eq!(allocation.usage_stats.access_count, 2);
        assert_eq!(allocation.usage_stats.bytes_read, 512);
        assert_eq!(allocation.usage_stats.bytes_written, 256);
        assert_eq!(allocation.total_bytes_accessed(), 768);
    }

    #[test]
    fn test_access_pattern_tracking() {
        let mut pattern = AccessPattern::new();

        pattern.record_access(AccessType::Read, 1024);
        pattern.record_access(AccessType::Read, 1024);
        pattern.record_access(AccessType::Write, 2048);

        assert_eq!(pattern.access_types.len(), 3);
        assert!(pattern.frequency > 0.0);
    }

    #[test]
    fn test_allocation_tracker() {
        let mut tracker = AllocationTracker::new();

        let source = AllocationSource::new(
            "test_function".to_string(),
            None,
            123,
            AllocationContext::UserAllocation {
                request_id: "test".to_string(),
            },
        );

        let allocation = MemoryAllocation::new(0x1000, 1024, source, MemoryType::Host, None);

        tracker.track_allocation(allocation);
        tracker.record_access(0x1000, AccessType::Read, 512);

        assert_eq!(tracker.total_memory_usage(), 1024);
        assert!(tracker.get_allocation(0x1000).is_some());
        assert!(tracker.get_access_pattern(0x1000).is_some());
    }

    #[test]
    fn test_memory_type_display() {
        assert_eq!(format!("{}", MemoryType::Device), "Device");
        assert_eq!(format!("{}", MemoryType::Host), "Host");
        assert_eq!(format!("{}", MemoryType::Unified), "Unified");
    }

    #[test]
    fn test_allocation_source_description() {
        let source = AllocationSource::new(
            "test_function".to_string(),
            Some(("test.rs".to_string(), 42)),
            123,
            AllocationContext::UserAllocation {
                request_id: "test".to_string(),
            },
        );

        let description = source.description();
        assert!(description.contains("test_function"));
        assert!(description.contains("test.rs:42"));
    }
}
