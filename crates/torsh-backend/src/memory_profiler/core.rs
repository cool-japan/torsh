//! # Memory Profiler Core Components
//!
//! This module contains the core memory profiler infrastructure including the main
//! MemoryProfiler struct, configuration types, and fundamental data structures for
//! memory tracking and profiling.
//!
//! ## Key Components
//!
//! - **MemoryProfiler** - Main profiler struct that orchestrates all memory profiling activities
//! - **MemoryProfilerConfig** - Configuration options for controlling profiler behavior
//! - **MemorySnapshot** - Point-in-time memory usage snapshots
//! - **GlobalMemoryStats** - Global memory statistics aggregation
//! - **DeviceMemoryUsage** - Per-device memory usage tracking
//! - **HostMemoryUsage** - Host memory usage tracking
//!
//! ## Usage Example
//!
//! ```rust
//! use torsh_backend::memory_profiler::core::{MemoryProfiler, MemoryProfilerConfig};
//! use torsh_backend::profiler::SimpleProfiler;
//! use std::time::Duration;
//!
//! // Create configuration
//! let config = MemoryProfilerConfig {
//!     enable_allocation_tracking: true,
//!     enable_pressure_monitoring: true,
//!     snapshot_interval: Duration::from_secs(10),
//!     ..Default::default()
//! };
//!
//! // Create profiler
//! let base_profiler = Box::new(SimpleProfiler::new());
//! let memory_profiler = MemoryProfiler::new(base_profiler, config);
//! ```

use crate::profiler::Profiler;
use crate::Device;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use torsh_core::error::Result;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

// Re-export types from other modules for convenience
use super::allocation::MemoryAllocation;
use super::allocation::PressureLevel;
use super::allocation::{AccessPattern, PerformanceHint};
use super::fragmentation::{FragmentationConfig, FragmentationTracker};
use super::pressure::{MemoryPressureEvent, MemoryPressureIndicators};
use super::scirs2::{ScirS2Integration, ScirS2IntegrationConfig};

/// Comprehensive memory profiler with scirs2 integration
///
/// The MemoryProfiler is the central component for memory profiling operations in ToRSh.
/// It orchestrates allocation tracking, memory pressure monitoring, access pattern analysis,
/// fragmentation tracking, and SciRS2 integration to provide comprehensive memory insights.
///
/// # Features
///
/// - **Allocation Tracking**: Detailed tracking of memory allocations and deallocations
/// - **Pressure Monitoring**: Real-time memory pressure monitoring and alerting
/// - **Pattern Analysis**: Access pattern analysis for optimization opportunities
/// - **Fragmentation Tracking**: Memory fragmentation detection and mitigation
/// - **SciRS2 Integration**: Deep integration with SciRS2 memory management
/// - **Performance Hints**: Automatic generation of performance optimization suggestions
///
/// # Thread Safety
///
/// The MemoryProfiler is designed to be thread-safe and can be safely shared across
/// multiple threads using Arc<MemoryProfiler>.
///
/// # Example
///
/// ```rust,ignore
/// use torsh_backend::memory_profiler::core::{MemoryProfiler, MemoryProfilerConfig};
/// use torsh_backend::profiler::SimpleProfiler;
///
/// let config = MemoryProfilerConfig::default();
/// let base_profiler = Box::new(SimpleProfiler::new());
/// let profiler = MemoryProfiler::new(base_profiler, config);
///
/// // Start profiling
/// profiler.start_profiling()?;
///
/// // Your memory operations here...
///
/// // Get memory statistics
/// let stats = profiler.get_memory_stats();
/// println!("Peak memory usage: {} bytes", stats.peak_memory_usage);
/// ```
pub struct MemoryProfiler {
    /// Base profiler functionality
    base_profiler: Box<dyn Profiler + Send + Sync>,

    /// Memory allocation tracking (using pointer addresses as keys)
    allocations: Arc<RwLock<HashMap<usize, MemoryAllocation>>>,

    /// Memory pool statistics
    _pool_stats: Arc<RwLock<HashMap<String, MemoryPoolStats>>>,

    /// Memory usage history
    usage_history: Arc<Mutex<VecDeque<MemorySnapshot>>>,

    /// Memory pressure events
    pressure_events: Arc<Mutex<Vec<MemoryPressureEvent>>>,

    /// Memory access patterns (using pointer addresses as keys)
    access_patterns: Arc<RwLock<HashMap<usize, AccessPattern>>>,

    /// Configuration
    config: MemoryProfilerConfig,

    /// Global statistics
    global_stats: Arc<Mutex<GlobalMemoryStats>>,

    /// Peak memory watermarks per device
    peak_watermarks: Arc<RwLock<HashMap<Device, usize>>>,

    /// Memory fragmentation tracker
    fragmentation_tracker: Arc<Mutex<FragmentationTracker>>,

    /// SciRS2 integration state
    scirs2_integration: Arc<Mutex<ScirS2Integration>>,
}

/// Memory profiler configuration
///
/// Controls the behavior and features of the memory profiler. Each feature can be
/// independently enabled or disabled based on performance requirements and use cases.
///
/// # Performance Considerations
///
/// - **Allocation tracking**: Low overhead for basic tracking, higher for detailed analysis
/// - **Access pattern analysis**: Moderate overhead, useful for optimization
/// - **Pressure monitoring**: Very low overhead, recommended for production
/// - **Fragmentation tracking**: Low to moderate overhead depending on frequency
/// - **Stack traces**: High overhead, use only for debugging
///
/// # Example
///
/// ```rust,ignore
/// use torsh_backend::memory_profiler::core::MemoryProfilerConfig;
/// use std::time::Duration;
///
/// // Production configuration
/// let production_config = MemoryProfilerConfig {
///     enable_allocation_tracking: true,
///     enable_access_pattern_analysis: false, // Disabled for performance
///     enable_pressure_monitoring: true,
///     enable_fragmentation_tracking: true,
///     enable_stack_traces: false, // Too expensive for production
///     snapshot_interval: Duration::from_secs(30),
///     ..Default::default()
/// };
///
/// // Debug configuration
/// let debug_config = MemoryProfilerConfig {
///     enable_stack_traces: true,
///     snapshot_interval: Duration::from_secs(1),
///     ..Default::default()
/// };
/// ```
#[derive(Debug)]
pub struct MemoryProfilerConfig {
    /// Enable detailed allocation tracking
    pub enable_allocation_tracking: bool,

    /// Enable access pattern analysis
    pub enable_access_pattern_analysis: bool,

    /// Enable memory pressure monitoring
    pub enable_pressure_monitoring: bool,

    /// Enable fragmentation tracking
    pub enable_fragmentation_tracking: bool,

    /// Enable SciRS2 integration
    pub enable_scirs2_integration: bool,

    /// Maximum number of tracked allocations
    pub max_tracked_allocations: usize,

    /// Memory snapshot interval
    pub snapshot_interval: Duration,

    /// Access pattern analysis window
    pub access_pattern_window: Duration,

    /// Performance hint generation threshold
    pub hint_threshold: f64,

    /// Enable stack trace collection
    pub enable_stack_traces: bool,

    /// Memory pressure threshold (percentage)
    pub memory_pressure_threshold: f64,

    /// Fragmentation threshold for alerts
    pub fragmentation_alert_threshold: f64,
}

impl Default for MemoryProfilerConfig {
    fn default() -> Self {
        Self {
            enable_allocation_tracking: true,
            enable_access_pattern_analysis: true,
            enable_pressure_monitoring: true,
            enable_fragmentation_tracking: true,
            enable_scirs2_integration: true,
            max_tracked_allocations: 100000,
            snapshot_interval: Duration::from_secs(10),
            access_pattern_window: Duration::from_secs(60),
            hint_threshold: 0.1,
            enable_stack_traces: false, // Expensive, disabled by default
            memory_pressure_threshold: 0.85, // 85%
            fragmentation_alert_threshold: 0.3, // 30%
        }
    }
}

/// Memory pool statistics
///
/// Tracks statistics for a specific memory pool, including allocation patterns,
/// usage efficiency, and performance characteristics.
#[derive(Debug)]
pub struct MemoryPoolStats {
    /// Pool identifier
    pub pool_id: String,

    /// Associated device
    pub device: Option<Device>,

    /// Total pool size
    pub total_size: usize,

    /// Currently allocated memory
    pub allocated_size: usize,

    /// Peak allocated memory
    pub peak_allocated_size: usize,

    /// Number of allocations
    pub allocation_count: AtomicUsize,

    /// Number of deallocations
    pub deallocation_count: AtomicUsize,

    /// Average allocation size
    pub average_allocation_size: f64,

    /// Pool utilization efficiency (0.0 to 1.0)
    pub utilization_efficiency: f64,

    /// Pool creation timestamp
    pub created_at: Instant,

    /// Last activity timestamp
    pub last_activity: Instant,

    /// Pool-specific performance hints
    pub performance_hints: Vec<PerformanceHint>,
}

/// Point-in-time memory usage snapshot
///
/// Captures a comprehensive view of memory usage across all devices and memory types
/// at a specific point in time. Used for historical analysis and trend detection.
///
/// # Usage
///
/// ```rust,ignore
/// use torsh_backend::memory_profiler::core::MemorySnapshot;
///
/// let snapshot = profiler.take_snapshot()?;
/// println!("Total memory usage: {} bytes", snapshot.total_memory_usage());
/// println!("Host memory: {} bytes", snapshot.host_memory.total_allocated);
///
/// for (device, usage) in &snapshot.device_memory {
///     println!("Device {:?}: {} bytes", device, usage.total_allocated);
/// }
/// ```
#[derive(Debug)]
pub struct MemorySnapshot {
    /// Snapshot timestamp
    pub timestamp: Instant,

    /// Total memory usage across all devices and memory types
    pub total_memory_bytes: usize,

    /// Host memory usage
    pub host_memory: HostMemoryUsage,

    /// Per-device memory usage
    pub device_memory: HashMap<Device, DeviceMemoryUsage>,

    /// Memory pressure indicators at snapshot time
    pub pressure_indicators: MemoryPressureIndicators,

    /// Number of active allocations
    pub active_allocations: usize,

    /// Memory fragmentation score (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fragmentation_score: f64,

    /// Cache hit rates
    pub cache_stats: HashMap<String, CacheStats>,
}

impl MemorySnapshot {
    /// Calculate total memory usage across all devices and memory types
    pub fn total_memory_usage(&self) -> usize {
        let device_total: usize = self
            .device_memory
            .values()
            .map(|usage| usage.total_allocated)
            .sum();

        self.host_memory.total_allocated + device_total
    }

    /// Get memory usage for a specific device
    pub fn device_usage(&self, device: &Device) -> Option<&DeviceMemoryUsage> {
        self.device_memory.get(device)
    }

    /// Calculate overall memory utilization efficiency
    pub fn utilization_efficiency(&self) -> f64 {
        if self.total_memory_bytes == 0 {
            return 0.0;
        }

        let total_used = self.total_memory_usage();
        total_used as f64 / self.total_memory_bytes as f64
    }

    /// Check if memory pressure is above threshold
    pub fn is_under_pressure(&self, threshold: f64) -> bool {
        let pressure_value = match self.pressure_indicators.system_pressure {
            PressureLevel::None => 0.0,
            PressureLevel::Low => 0.25,
            PressureLevel::Medium => 0.5,
            PressureLevel::High => 0.75,
            PressureLevel::Critical => 1.0,
        };
        pressure_value > threshold
    }
}

/// Per-device memory usage tracking
///
/// Detailed memory usage information for a specific compute device (GPU, TPU, etc.).
/// Includes breakdown by memory type and performance characteristics.
#[derive(Debug)]
pub struct DeviceMemoryUsage {
    /// Device identifier
    pub device: Device,

    /// Total allocated memory on device
    pub total_allocated: usize,

    /// Peak allocated memory
    pub peak_allocated: usize,

    /// Available memory on device
    pub available_memory: usize,

    /// Memory bandwidth utilization
    pub bandwidth_utilization: BandwidthUtilization,

    /// Cache statistics
    pub cache_stats: HashMap<String, CacheStats>,

    /// Memory access patterns
    pub access_patterns: Vec<String>, // Pattern identifiers

    /// Device-specific performance hints
    pub performance_hints: Vec<PerformanceHint>,
}

impl DeviceMemoryUsage {
    /// Calculate memory utilization percentage
    pub fn utilization_percentage(&self) -> f64 {
        let total_memory = self.total_allocated + self.available_memory;
        if total_memory == 0 {
            return 0.0;
        }
        (self.total_allocated as f64 / total_memory as f64) * 100.0
    }

    /// Check if device is under memory pressure
    pub fn is_under_pressure(&self, threshold: f64) -> bool {
        self.utilization_percentage() > threshold
    }
}

/// Host memory usage tracking
///
/// Detailed memory usage information for host (CPU) memory, including system memory,
/// process memory, and ToRSh-specific allocations.
#[derive(Debug)]
pub struct HostMemoryUsage {
    /// Total allocated host memory
    pub total_allocated: usize,

    /// Peak allocated host memory
    pub peak_allocated: usize,

    /// Available system memory
    pub available_system_memory: usize,

    /// Process memory usage (RSS)
    pub process_memory_rss: usize,

    /// Process virtual memory usage
    pub process_memory_virtual: usize,

    /// Memory mapped files
    pub memory_mapped_size: usize,

    /// Swap usage
    pub swap_usage: usize,

    /// Host cache statistics
    pub cache_stats: HashMap<String, CacheStats>,
}

impl HostMemoryUsage {
    /// Calculate host memory utilization percentage
    pub fn utilization_percentage(&self) -> f64 {
        let total_system = self.total_allocated + self.available_system_memory;
        if total_system == 0 {
            return 0.0;
        }
        (self.total_allocated as f64 / total_system as f64) * 100.0
    }

    /// Get total memory footprint including virtual memory
    pub fn total_footprint(&self) -> usize {
        self.process_memory_virtual + self.memory_mapped_size
    }
}

/// Global memory statistics aggregation
///
/// Aggregates memory statistics across all devices and memory types to provide
/// a system-wide view of memory usage and performance.
#[derive(Debug)]
pub struct GlobalMemoryStats {
    /// Total allocations performed
    pub total_allocations: AtomicU64,

    /// Total deallocations performed
    pub total_deallocations: AtomicU64,

    /// Peak memory usage across all devices
    pub peak_memory_usage: AtomicUsize,

    /// Current memory usage
    pub current_memory_usage: AtomicUsize,

    /// Total bytes allocated (cumulative)
    pub total_bytes_allocated: AtomicU64,

    /// Total bytes deallocated (cumulative)
    pub total_bytes_deallocated: AtomicU64,

    /// Average allocation size
    pub average_allocation_size: AtomicUsize,

    /// Number of out-of-memory events
    pub oom_events: AtomicUsize,

    /// Memory pressure events count
    pub pressure_events_count: AtomicUsize,

    /// Fragmentation events count
    pub fragmentation_events_count: AtomicUsize,

    /// Performance optimization suggestions count
    pub optimization_suggestions_count: AtomicUsize,
}

impl Default for GlobalMemoryStats {
    fn default() -> Self {
        Self {
            total_allocations: AtomicU64::new(0),
            total_deallocations: AtomicU64::new(0),
            peak_memory_usage: AtomicUsize::new(0),
            current_memory_usage: AtomicUsize::new(0),
            total_bytes_allocated: AtomicU64::new(0),
            total_bytes_deallocated: AtomicU64::new(0),
            average_allocation_size: AtomicUsize::new(0),
            oom_events: AtomicUsize::new(0),
            pressure_events_count: AtomicUsize::new(0),
            fragmentation_events_count: AtomicUsize::new(0),
            optimization_suggestions_count: AtomicUsize::new(0),
        }
    }
}

impl Clone for GlobalMemoryStats {
    fn clone(&self) -> Self {
        use std::sync::atomic::Ordering;
        Self {
            total_allocations: AtomicU64::new(self.total_allocations.load(Ordering::Relaxed)),
            total_deallocations: AtomicU64::new(self.total_deallocations.load(Ordering::Relaxed)),
            peak_memory_usage: AtomicUsize::new(self.peak_memory_usage.load(Ordering::Relaxed)),
            current_memory_usage: AtomicUsize::new(
                self.current_memory_usage.load(Ordering::Relaxed),
            ),
            total_bytes_allocated: AtomicU64::new(
                self.total_bytes_allocated.load(Ordering::Relaxed),
            ),
            total_bytes_deallocated: AtomicU64::new(
                self.total_bytes_deallocated.load(Ordering::Relaxed),
            ),
            average_allocation_size: AtomicUsize::new(
                self.average_allocation_size.load(Ordering::Relaxed),
            ),
            oom_events: AtomicUsize::new(self.oom_events.load(Ordering::Relaxed)),
            pressure_events_count: AtomicUsize::new(
                self.pressure_events_count.load(Ordering::Relaxed),
            ),
            fragmentation_events_count: AtomicUsize::new(
                self.fragmentation_events_count.load(Ordering::Relaxed),
            ),
            optimization_suggestions_count: AtomicUsize::new(
                self.optimization_suggestions_count.load(Ordering::Relaxed),
            ),
        }
    }
}

impl GlobalMemoryStats {
    /// Get current outstanding allocations count
    pub fn outstanding_allocations(&self) -> u64 {
        let allocs = self.total_allocations.load(Ordering::Relaxed);
        let deallocs = self.total_deallocations.load(Ordering::Relaxed);
        allocs.saturating_sub(deallocs)
    }

    /// Get current outstanding bytes
    pub fn outstanding_bytes(&self) -> u64 {
        let alloc_bytes = self.total_bytes_allocated.load(Ordering::Relaxed);
        let dealloc_bytes = self.total_bytes_deallocated.load(Ordering::Relaxed);
        alloc_bytes.saturating_sub(dealloc_bytes)
    }

    /// Update peak memory usage if current usage is higher
    pub fn update_peak_usage(&self, current_usage: usize) {
        self.current_memory_usage
            .store(current_usage, Ordering::Relaxed);

        // Use compare-and-swap to atomically update peak if current is higher
        let mut peak = self.peak_memory_usage.load(Ordering::Relaxed);
        while current_usage > peak {
            match self.peak_memory_usage.compare_exchange_weak(
                peak,
                current_usage,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current_peak) => peak = current_peak,
            }
        }
    }

    /// Record a new allocation
    pub fn record_allocation(&self, size: usize) {
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_allocated
            .fetch_add(size as u64, Ordering::Relaxed);

        // Update average allocation size
        let total_allocs = self.total_allocations.load(Ordering::Relaxed);
        let total_bytes = self.total_bytes_allocated.load(Ordering::Relaxed);
        if total_allocs > 0 {
            let avg_size = total_bytes / total_allocs;
            self.average_allocation_size
                .store(avg_size as usize, Ordering::Relaxed);
        }
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, size: usize) {
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_deallocated
            .fetch_add(size as u64, Ordering::Relaxed);
    }
}

/// Cache statistics tracking
///
/// Tracks cache performance metrics including hit rates, miss rates, and efficiency indicators.
#[derive(Debug)]
pub struct CacheStats {
    /// Cache identifier
    pub cache_name: String,

    /// Cache hits
    pub hits: AtomicU64,

    /// Cache misses
    pub misses: AtomicU64,

    /// Cache evictions
    pub evictions: AtomicU64,

    /// Total cache size
    pub total_size: usize,

    /// Used cache size
    pub used_size: AtomicUsize,

    /// Average access time
    pub average_access_time: Duration,
}

impl CacheStats {
    /// Calculate cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            return 0.0;
        }

        hits as f64 / total as f64
    }

    /// Calculate cache utilization percentage
    pub fn utilization(&self) -> f64 {
        if self.total_size == 0 {
            return 0.0;
        }

        let used = self.used_size.load(Ordering::Relaxed);
        (used as f64 / self.total_size as f64) * 100.0
    }
}

/// Bandwidth utilization tracking
///
/// Tracks memory bandwidth utilization patterns for performance analysis and optimization.
#[derive(Debug)]
pub struct BandwidthUtilization {
    /// Device identifier
    pub device: Device,

    /// Peak bandwidth achieved (bytes/sec)
    pub peak_bandwidth: u64,

    /// Average bandwidth over measurement window (bytes/sec)
    pub average_bandwidth: u64,

    /// Theoretical maximum bandwidth (bytes/sec)
    pub theoretical_max_bandwidth: u64,

    /// Read bandwidth utilization
    pub read_bandwidth: u64,

    /// Write bandwidth utilization
    pub write_bandwidth: u64,

    /// Bidirectional bandwidth utilization
    pub bidirectional_bandwidth: u64,

    /// Measurement window duration
    pub measurement_window: Duration,
}

impl BandwidthUtilization {
    /// Calculate bandwidth efficiency percentage
    pub fn efficiency_percentage(&self) -> f64 {
        if self.theoretical_max_bandwidth == 0 {
            return 0.0;
        }

        (self.average_bandwidth as f64 / self.theoretical_max_bandwidth as f64) * 100.0
    }

    /// Check if bandwidth utilization is optimal
    pub fn is_optimal(&self, threshold: f64) -> bool {
        self.efficiency_percentage() >= threshold
    }
}

// Implementation of the main MemoryProfiler struct
impl MemoryProfiler {
    /// Create a new memory profiler
    ///
    /// # Arguments
    /// * `base_profiler` - Base profiler implementation
    /// * `config` - Profiler configuration
    ///
    /// # Example
    /// ```rust,ignore
    /// use torsh_backend::memory_profiler::core::{MemoryProfiler, MemoryProfilerConfig};
    /// use torsh_backend::profiler::SimpleProfiler;
    ///
    /// let config = MemoryProfilerConfig::default();
    /// let base_profiler = Box::new(SimpleProfiler::new());
    /// let profiler = MemoryProfiler::new(base_profiler, config);
    /// ```
    pub fn new(
        base_profiler: Box<dyn Profiler + Send + Sync>,
        config: MemoryProfilerConfig,
    ) -> Self {
        Self {
            base_profiler,
            allocations: Arc::new(RwLock::new(HashMap::new())),
            _pool_stats: Arc::new(RwLock::new(HashMap::new())),
            usage_history: Arc::new(Mutex::new(VecDeque::new())),
            pressure_events: Arc::new(Mutex::new(Vec::new())),
            access_patterns: Arc::new(RwLock::new(HashMap::new())),
            config,
            global_stats: Arc::new(Mutex::new(GlobalMemoryStats::default())),
            peak_watermarks: Arc::new(RwLock::new(HashMap::new())),
            fragmentation_tracker: Arc::new(Mutex::new(FragmentationTracker::new(
                FragmentationConfig::default(),
            ))),
            scirs2_integration: Arc::new(Mutex::new(ScirS2Integration::new(
                ScirS2IntegrationConfig::default(),
            ))),
        }
    }

    /// Get profiler configuration
    pub fn config(&self) -> &MemoryProfilerConfig {
        &self.config
    }

    /// Check if allocation tracking is enabled
    pub fn is_allocation_tracking_enabled(&self) -> bool {
        self.config.enable_allocation_tracking
    }

    /// Check if access pattern analysis is enabled
    pub fn is_access_pattern_analysis_enabled(&self) -> bool {
        self.config.enable_access_pattern_analysis
    }

    /// Check if pressure monitoring is enabled
    pub fn is_pressure_monitoring_enabled(&self) -> bool {
        self.config.enable_pressure_monitoring
    }

    /// Check if fragmentation tracking is enabled
    pub fn is_fragmentation_tracking_enabled(&self) -> bool {
        self.config.enable_fragmentation_tracking
    }

    /// Check if SciRS2 integration is enabled
    pub fn is_scirs2_integration_enabled(&self) -> bool {
        self.config.enable_scirs2_integration
    }

    /// Get global memory statistics
    pub fn get_global_stats(&self) -> GlobalMemoryStats {
        (*self.global_stats.lock()).clone()
    }

    /// Get current memory usage snapshot
    pub fn take_snapshot(&self) -> Result<MemorySnapshot> {
        // Implementation would gather data from all tracking systems
        // This is a simplified version
        let now = Instant::now();
        let global_stats = self.get_global_stats();

        Ok(MemorySnapshot {
            timestamp: now,
            total_memory_bytes: global_stats.current_memory_usage.load(Ordering::Relaxed),
            host_memory: HostMemoryUsage {
                total_allocated: 0, // Would be populated from actual data
                peak_allocated: 0,
                available_system_memory: 0,
                process_memory_rss: 0,
                process_memory_virtual: 0,
                memory_mapped_size: 0,
                swap_usage: 0,
                cache_stats: HashMap::new(),
            },
            device_memory: HashMap::new(),
            pressure_indicators: MemoryPressureIndicators::default(),
            active_allocations: global_stats.outstanding_allocations() as usize,
            fragmentation_score: 0.0,
            cache_stats: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiler::SimpleProfiler;

    #[test]
    fn test_memory_profiler_creation() {
        let config = MemoryProfilerConfig::default();
        let base_profiler = Box::new(SimpleProfiler::new());
        let _profiler = MemoryProfiler::new(base_profiler, config);
    }

    #[test]
    fn test_memory_profiler_config_default() {
        let config = MemoryProfilerConfig::default();
        assert!(config.enable_allocation_tracking);
        assert!(config.enable_access_pattern_analysis);
        assert!(config.enable_pressure_monitoring);
        assert!(!config.enable_stack_traces);
        assert_eq!(config.max_tracked_allocations, 100000);
    }

    #[test]
    fn test_global_memory_stats() {
        let stats = GlobalMemoryStats::default();

        // Test initial state
        assert_eq!(stats.outstanding_allocations(), 0);
        assert_eq!(stats.outstanding_bytes(), 0);

        // Test allocation recording
        stats.record_allocation(1024);
        assert_eq!(stats.total_allocations.load(Ordering::Relaxed), 1);
        assert_eq!(stats.total_bytes_allocated.load(Ordering::Relaxed), 1024);
        assert_eq!(stats.outstanding_allocations(), 1);
        assert_eq!(stats.outstanding_bytes(), 1024);

        // Test deallocation recording
        stats.record_deallocation(512);
        assert_eq!(stats.total_deallocations.load(Ordering::Relaxed), 1);
        assert_eq!(stats.total_bytes_deallocated.load(Ordering::Relaxed), 512);
        assert_eq!(stats.outstanding_bytes(), 512);
    }

    #[test]
    fn test_cache_stats() {
        let cache_stats = CacheStats {
            cache_name: "test_cache".to_string(),
            hits: AtomicU64::new(80),
            misses: AtomicU64::new(20),
            evictions: AtomicU64::new(5),
            total_size: 1024,
            used_size: AtomicUsize::new(512),
            average_access_time: Duration::from_nanos(100),
        };

        assert_eq!(cache_stats.hit_rate(), 0.8);
        assert_eq!(cache_stats.utilization(), 50.0);
    }

    #[test]
    fn test_bandwidth_utilization() {
        let bandwidth = BandwidthUtilization {
            device: Device::cpu().unwrap(),
            peak_bandwidth: 800_000_000,
            average_bandwidth: 600_000_000,
            theoretical_max_bandwidth: 1_000_000_000,
            read_bandwidth: 300_000_000,
            write_bandwidth: 300_000_000,
            bidirectional_bandwidth: 600_000_000,
            measurement_window: Duration::from_secs(10),
        };

        assert_eq!(bandwidth.efficiency_percentage(), 60.0);
        assert!(bandwidth.is_optimal(50.0));
        assert!(!bandwidth.is_optimal(70.0));
    }

    #[test]
    fn test_memory_snapshot() {
        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            total_memory_bytes: 2048,
            host_memory: HostMemoryUsage {
                total_allocated: 1024,
                peak_allocated: 1536,
                available_system_memory: 4096,
                process_memory_rss: 1024,
                process_memory_virtual: 2048,
                memory_mapped_size: 512,
                swap_usage: 0,
                cache_stats: HashMap::new(),
            },
            device_memory: HashMap::new(),
            pressure_indicators: MemoryPressureIndicators::default(),
            active_allocations: 10,
            fragmentation_score: 0.1,
            cache_stats: HashMap::new(),
        };

        assert_eq!(snapshot.total_memory_usage(), 1024);
        assert_eq!(snapshot.utilization_efficiency(), 0.5);
    }
}
