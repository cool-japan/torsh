//! Memory Pressure Monitoring and Health Management
//!
//! This module provides comprehensive memory pressure monitoring, including
//! real-time health assessment, pressure event tracking, and automated
//! response mechanisms for memory management.

use crate::Device;
use super::core::{MemoryType, PressureLevel};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

// =============================================================================
// MEMORY USAGE SNAPSHOTS AND TRACKING
// =============================================================================

/// Comprehensive memory usage snapshot
///
/// Captures the complete memory state at a specific point in time,
/// including device usage, host usage, and system pressure indicators.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Snapshot timestamp
    pub timestamp: Instant,

    /// Per-device memory usage
    pub device_usage: HashMap<Device, DeviceMemoryUsage>,

    /// Host memory usage
    pub host_usage: HostMemoryUsage,

    /// System-wide memory pressure (0.0 to 1.0)
    pub memory_pressure: f64,

    /// Active allocations count
    pub active_allocations: usize,

    /// Total allocated bytes
    pub total_allocated: usize,

    /// Memory fragmentation level (0.0 to 1.0)
    pub fragmentation_level: f64,

    /// Bandwidth utilization statistics
    pub bandwidth_utilization: BandwidthUtilization,

    /// Memory efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
}

impl MemorySnapshot {
    /// Creates a new memory snapshot
    pub fn new() -> Self {
        Self {
            timestamp: Instant::now(),
            device_usage: HashMap::new(),
            host_usage: HostMemoryUsage::default(),
            memory_pressure: 0.0,
            active_allocations: 0,
            total_allocated: 0,
            fragmentation_level: 0.0,
            bandwidth_utilization: BandwidthUtilization::default(),
            efficiency_score: 1.0,
        }
    }

    /// Calculates overall system health score (0.0 to 1.0)
    pub fn health_score(&self) -> f64 {
        let pressure_health = 1.0 - self.memory_pressure;
        let fragmentation_health = 1.0 - self.fragmentation_level;
        let bandwidth_health = self.bandwidth_utilization.efficiency;

        // Weighted average of health factors
        (pressure_health * 0.4) + (fragmentation_health * 0.3) + (bandwidth_health * 0.3)
    }

    /// Checks if the system is under memory pressure
    pub fn is_under_pressure(&self) -> bool {
        self.memory_pressure > 0.7
    }

    /// Gets the most constrained device
    pub fn most_constrained_device(&self) -> Option<(&Device, &DeviceMemoryUsage)> {
        self.device_usage
            .iter()
            .max_by(|(_, a), (_, b)| a.utilization_percent.partial_cmp(&b.utilization_percent).unwrap())
    }

    /// Calculates total system memory utilization
    pub fn total_utilization(&self) -> f64 {
        let host_util = if self.host_usage.total_memory > 0 {
            self.host_usage.process_memory as f64 / self.host_usage.total_memory as f64
        } else {
            0.0
        };

        let device_util: f64 = self.device_usage
            .values()
            .map(|usage| usage.utilization_percent / 100.0)
            .sum::<f64>() / self.device_usage.len().max(1) as f64;

        // Weighted average of host and device utilization
        (host_util * 0.6) + (device_util * 0.4)
    }
}

impl Default for MemorySnapshot {
    fn default() -> Self {
        Self::new()
    }
}

/// Device-specific memory usage information
#[derive(Debug, Clone)]
pub struct DeviceMemoryUsage {
    /// Total device memory in bytes
    pub total_memory: usize,

    /// Used memory in bytes
    pub used_memory: usize,

    /// Free memory in bytes
    pub free_memory: usize,

    /// Reserved memory in bytes
    pub reserved_memory: usize,

    /// Memory utilization percentage (0.0 to 100.0)
    pub utilization_percent: f64,

    /// Memory bandwidth usage (0.0 to 1.0)
    pub bandwidth_usage: f64,

    /// Active memory transfers
    pub active_transfers: usize,

    /// Memory type breakdown
    pub memory_breakdown: HashMap<MemoryType, usize>,

    /// Pressure indicators specific to this device
    pub pressure_indicators: DevicePressureIndicators,
}

impl DeviceMemoryUsage {
    /// Creates a new device memory usage record
    pub fn new(total_memory: usize) -> Self {
        Self {
            total_memory,
            used_memory: 0,
            free_memory: total_memory,
            reserved_memory: 0,
            utilization_percent: 0.0,
            bandwidth_usage: 0.0,
            active_transfers: 0,
            memory_breakdown: HashMap::new(),
            pressure_indicators: DevicePressureIndicators::default(),
        }
    }

    /// Updates memory usage statistics
    pub fn update_usage(&mut self, used: usize, reserved: usize) {
        self.used_memory = used;
        self.reserved_memory = reserved;
        self.free_memory = self.total_memory.saturating_sub(used + reserved);
        self.utilization_percent = if self.total_memory > 0 {
            (used as f64 / self.total_memory as f64) * 100.0
        } else {
            0.0
        };
    }

    /// Calculates available memory considering fragmentation
    pub fn available_memory(&self) -> usize {
        // Account for fragmentation by reducing available memory
        let fragmentation_factor = 1.0 - (self.pressure_indicators.fragmentation_level * 0.3);
        (self.free_memory as f64 * fragmentation_factor) as usize
    }

    /// Checks if device is critically low on memory
    pub fn is_critically_low(&self) -> bool {
        self.utilization_percent > 95.0 || self.pressure_indicators.pressure_level >= PressureLevel::Critical
    }

    /// Gets the effective pressure level considering all factors
    pub fn effective_pressure(&self) -> PressureLevel {
        let util_pressure = PressureLevel::from_f64(self.utilization_percent / 100.0);
        let bandwidth_pressure = PressureLevel::from_f64(self.bandwidth_usage);
        let fragmentation_pressure = PressureLevel::from_f64(self.pressure_indicators.fragmentation_level);

        // Return the highest pressure level
        [util_pressure, bandwidth_pressure, fragmentation_pressure]
            .iter()
            .max()
            .copied()
            .unwrap_or(PressureLevel::None)
    }
}

/// Device-specific pressure indicators
#[derive(Debug, Clone, Default)]
pub struct DevicePressureIndicators {
    /// Device memory pressure level
    pub pressure_level: PressureLevel,

    /// Fragmentation level (0.0 to 1.0)
    pub fragmentation_level: f64,

    /// Memory allocation failure rate
    pub allocation_failure_rate: f64,

    /// Out-of-memory events in last hour
    pub oom_events: u32,

    /// Memory transfer congestion level (0.0 to 1.0)
    pub transfer_congestion: f64,
}

/// Host memory usage information
#[derive(Debug, Clone, Default)]
pub struct HostMemoryUsage {
    /// Total system memory in bytes
    pub total_memory: usize,

    /// Available memory in bytes
    pub available_memory: usize,

    /// Memory used by current process in bytes
    pub process_memory: usize,

    /// Pinned memory usage in bytes
    pub pinned_memory: usize,

    /// Virtual memory usage in bytes
    pub virtual_memory: usize,

    /// Cached memory in bytes
    pub cached_memory: usize,

    /// Buffered memory in bytes
    pub buffered_memory: usize,

    /// Memory pressure indicators
    pub pressure_indicators: MemoryPressureIndicators,
}

impl HostMemoryUsage {
    /// Calculates system memory utilization percentage
    pub fn utilization_percent(&self) -> f64 {
        if self.total_memory > 0 {
            let used = self.total_memory - self.available_memory;
            (used as f64 / self.total_memory as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Calculates process memory utilization percentage
    pub fn process_utilization_percent(&self) -> f64 {
        if self.total_memory > 0 {
            (self.process_memory as f64 / self.total_memory as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Checks if system is under memory pressure
    pub fn is_under_pressure(&self) -> bool {
        self.pressure_indicators.system_pressure >= PressureLevel::Medium ||
        self.utilization_percent() > 80.0
    }

    /// Gets effective memory available for new allocations
    pub fn effective_available(&self) -> usize {
        // Account for pressure and fragmentation
        let pressure_factor = match self.pressure_indicators.system_pressure {
            PressureLevel::None => 1.0,
            PressureLevel::Low => 0.9,
            PressureLevel::Medium => 0.7,
            PressureLevel::High => 0.5,
            PressureLevel::Critical => 0.2,
        };

        (self.available_memory as f64 * pressure_factor) as usize
    }
}

/// Comprehensive memory pressure indicators
#[derive(Debug, Clone, Default)]
pub struct MemoryPressureIndicators {
    /// System-wide memory pressure level
    pub system_pressure: PressureLevel,

    /// Process-specific memory pressure level
    pub process_pressure: PressureLevel,

    /// Swap usage in bytes
    pub swap_usage: usize,

    /// Page fault rate (faults per second)
    pub page_fault_rate: f64,

    /// Memory allocation failure rate (0.0 to 1.0)
    pub allocation_failure_rate: f64,

    /// Memory compaction events per hour
    pub compaction_events: u32,

    /// Out-of-memory killer activations in last hour
    pub oom_kills: u32,

    /// Memory reclaim pressure (0.0 to 1.0)
    pub reclaim_pressure: f64,
}

impl MemoryPressureIndicators {
    /// Calculates overall pressure score (0.0 to 1.0)
    pub fn pressure_score(&self) -> f64 {
        let system_score = self.system_pressure.as_f64();
        let process_score = self.process_pressure.as_f64();
        let failure_score = self.allocation_failure_rate;
        let reclaim_score = self.reclaim_pressure;

        // Weighted combination of pressure factors
        (system_score * 0.3) + (process_score * 0.3) + (failure_score * 0.2) + (reclaim_score * 0.2)
    }

    /// Checks if immediate action is required
    pub fn requires_immediate_action(&self) -> bool {
        self.system_pressure >= PressureLevel::Critical ||
        self.allocation_failure_rate > 0.1 ||
        self.oom_kills > 0
    }
}

// =============================================================================
// MEMORY PRESSURE EVENTS AND RESPONSES
// =============================================================================

/// Memory pressure event record
#[derive(Debug, Clone)]
pub struct MemoryPressureEvent {
    /// Event timestamp
    pub timestamp: Instant,

    /// Pressure level that triggered the event
    pub pressure_level: PressureLevel,

    /// Affected device (if device-specific)
    pub device: Option<Device>,

    /// Memory type primarily affected
    pub memory_type: MemoryType,

    /// Total memory at time of event
    pub total_memory: usize,

    /// Available memory at time of event
    pub available_memory: usize,

    /// Memory utilization at time of event
    pub utilization_percent: f64,

    /// Actions taken in response
    pub actions_taken: Vec<PressureAction>,

    /// Event resolution time
    pub resolution_time: Option<Duration>,

    /// Event severity score (0.0 to 1.0)
    pub severity_score: f64,

    /// Whether this event caused system instability
    pub caused_instability: bool,
}

impl MemoryPressureEvent {
    /// Creates a new memory pressure event
    pub fn new(
        pressure_level: PressureLevel,
        device: Option<Device>,
        memory_type: MemoryType,
        total_memory: usize,
        available_memory: usize,
    ) -> Self {
        let utilization_percent = if total_memory > 0 {
            ((total_memory - available_memory) as f64 / total_memory as f64) * 100.0
        } else {
            0.0
        };

        let severity_score = pressure_level.as_f64() * (utilization_percent / 100.0);

        Self {
            timestamp: Instant::now(),
            pressure_level,
            device,
            memory_type,
            total_memory,
            available_memory,
            utilization_percent,
            actions_taken: Vec::new(),
            resolution_time: None,
            severity_score,
            caused_instability: false,
        }
    }

    /// Adds an action taken in response to this event
    pub fn add_action(&mut self, action: PressureAction) {
        self.actions_taken.push(action);
    }

    /// Marks the event as resolved
    pub fn resolve(&mut self) {
        self.resolution_time = Some(self.timestamp.elapsed());
    }

    /// Calculates the effectiveness of the response
    pub fn response_effectiveness(&self) -> f64 {
        if self.actions_taken.is_empty() {
            return 0.0;
        }

        // Simple heuristic based on resolution time and actions taken
        let time_factor = if let Some(resolution_time) = self.resolution_time {
            let seconds = resolution_time.as_secs_f64();
            if seconds < 1.0 {
                1.0
            } else if seconds < 10.0 {
                0.8
            } else if seconds < 60.0 {
                0.6
            } else {
                0.3
            }
        } else {
            0.0 // Unresolved
        };

        let action_factor = (self.actions_taken.len() as f64).min(5.0) / 5.0;

        time_factor * action_factor
    }

    /// Checks if this was a critical event
    pub fn is_critical(&self) -> bool {
        self.pressure_level >= PressureLevel::Critical ||
        self.utilization_percent > 95.0 ||
        self.caused_instability
    }
}

/// Actions taken during memory pressure events
#[derive(Debug, Clone)]
pub enum PressureAction {
    /// Freed unused memory
    FreedUnusedMemory { amount: usize },

    /// Compacted memory pools
    CompactedPools { pools_affected: usize },

    /// Triggered garbage collection
    TriggeredGarbageCollection,

    /// Reduced cache sizes
    ReducedCaches { cache_reduction: usize },

    /// Swapped memory to disk
    SwappedToDisk { amount: usize },

    /// Killed low-priority allocations
    KilledAllocations { count: usize },

    /// Requested more memory from system
    RequestedMoreMemory { amount: usize },

    /// Defragmented memory
    DefragmentedMemory { recovered_space: usize },

    /// Flushed write buffers
    FlushedBuffers { buffer_count: usize },

    /// Paused non-critical operations
    PausedOperations { operation_count: usize },

    /// Moved data to slower storage
    MovedToSlowerStorage { amount: usize },

    /// Reduced allocation rate
    ReducedAllocationRate { factor: f64 },
}

impl PressureAction {
    /// Gets the estimated memory impact of this action
    pub fn memory_impact(&self) -> usize {
        match self {
            PressureAction::FreedUnusedMemory { amount } => *amount,
            PressureAction::CompactedPools { pools_affected } => pools_affected * 1024 * 1024, // Estimate
            PressureAction::TriggeredGarbageCollection => 0, // Variable
            PressureAction::ReducedCaches { cache_reduction } => *cache_reduction,
            PressureAction::SwappedToDisk { amount } => *amount,
            PressureAction::KilledAllocations { count } => count * 1024, // Estimate
            PressureAction::RequestedMoreMemory { amount } => *amount,
            PressureAction::DefragmentedMemory { recovered_space } => *recovered_space,
            PressureAction::FlushedBuffers { buffer_count } => buffer_count * 4096, // Estimate
            PressureAction::PausedOperations { .. } => 0,
            PressureAction::MovedToSlowerStorage { amount } => *amount,
            PressureAction::ReducedAllocationRate { .. } => 0, // Preventive
        }
    }

    /// Gets the urgency score of this action (0.0 to 1.0)
    pub fn urgency_score(&self) -> f64 {
        match self {
            PressureAction::FreedUnusedMemory { .. } => 0.6,
            PressureAction::CompactedPools { .. } => 0.7,
            PressureAction::TriggeredGarbageCollection => 0.5,
            PressureAction::ReducedCaches { .. } => 0.4,
            PressureAction::SwappedToDisk { .. } => 0.8,
            PressureAction::KilledAllocations { .. } => 0.9,
            PressureAction::RequestedMoreMemory { .. } => 0.3,
            PressureAction::DefragmentedMemory { .. } => 0.5,
            PressureAction::FlushedBuffers { .. } => 0.3,
            PressureAction::PausedOperations { .. } => 0.7,
            PressureAction::MovedToSlowerStorage { .. } => 0.6,
            PressureAction::ReducedAllocationRate { .. } => 0.2,
        }
    }
}

// =============================================================================
// BANDWIDTH UTILIZATION TRACKING
// =============================================================================

/// Memory bandwidth utilization statistics
#[derive(Debug, Clone, Default)]
pub struct BandwidthUtilization {
    /// Total memory bandwidth capacity (GB/s)
    pub total_bandwidth: f64,

    /// Current bandwidth usage (GB/s)
    pub current_usage: f64,

    /// Peak bandwidth usage recorded (GB/s)
    pub peak_usage: f64,

    /// Bandwidth efficiency (0.0 to 1.0)
    pub efficiency: f64,

    /// Per-device bandwidth breakdown
    pub device_breakdown: HashMap<Device, f64>,

    /// Historical bandwidth samples
    pub history: VecDeque<BandwidthSample>,

    /// Bandwidth saturation events
    pub saturation_events: u32,
}

impl BandwidthUtilization {
    /// Creates a new bandwidth utilization tracker
    pub fn new(total_bandwidth: f64) -> Self {
        Self {
            total_bandwidth,
            current_usage: 0.0,
            peak_usage: 0.0,
            efficiency: 0.0,
            device_breakdown: HashMap::new(),
            history: VecDeque::new(),
            saturation_events: 0,
        }
    }

    /// Updates bandwidth usage statistics
    pub fn update_usage(&mut self, current_usage: f64) {
        self.current_usage = current_usage;
        self.peak_usage = self.peak_usage.max(current_usage);
        self.efficiency = if self.total_bandwidth > 0.0 {
            (current_usage / self.total_bandwidth).min(1.0)
        } else {
            0.0
        };

        // Add to history
        self.history.push_back(BandwidthSample {
            timestamp: Instant::now(),
            usage: current_usage,
        });

        // Keep only recent history (last 1000 samples)
        while self.history.len() > 1000 {
            self.history.pop_front();
        }

        // Check for saturation
        if self.efficiency > 0.95 {
            self.saturation_events += 1;
        }
    }

    /// Calculates average bandwidth over the last duration
    pub fn average_usage(&self, duration: Duration) -> f64 {
        let cutoff = Instant::now() - duration;
        let recent_samples: Vec<_> = self.history
            .iter()
            .filter(|sample| sample.timestamp > cutoff)
            .collect();

        if recent_samples.is_empty() {
            self.current_usage
        } else {
            recent_samples.iter().map(|s| s.usage).sum::<f64>() / recent_samples.len() as f64
        }
    }

    /// Checks if bandwidth is currently saturated
    pub fn is_saturated(&self) -> bool {
        self.efficiency > 0.9
    }

    /// Predicts if bandwidth will become saturated soon
    pub fn saturation_risk(&self) -> f64 {
        if self.history.len() < 10 {
            return 0.0;
        }

        // Calculate trend over recent samples
        let recent: Vec<_> = self.history.iter().rev().take(10).collect();
        let trend = if recent.len() >= 2 {
            let first = recent.last().unwrap().usage;
            let last = recent.first().unwrap().usage;
            last - first
        } else {
            0.0
        };

        // Predict saturation risk based on current usage and trend
        let current_risk = self.efficiency;
        let trend_risk = if trend > 0.0 {
            (trend / self.total_bandwidth).min(0.5)
        } else {
            0.0
        };

        (current_risk + trend_risk).min(1.0)
    }
}

/// Bandwidth usage sample
#[derive(Debug, Clone)]
pub struct BandwidthSample {
    /// Sample timestamp
    pub timestamp: Instant,

    /// Bandwidth usage at this time (GB/s)
    pub usage: f64,
}

// =============================================================================
// MEMORY POOL STATISTICS
// =============================================================================

/// Memory pool performance statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    /// Pool name/identifier
    pub name: String,

    /// Total pool capacity in bytes
    pub total_capacity: usize,

    /// Currently allocated from pool
    pub allocated_from_pool: usize,

    /// Peak allocation from pool
    pub peak_allocated: usize,

    /// Number of allocations served
    pub allocations_served: u64,

    /// Number of cache hits
    pub cache_hits: u64,

    /// Number of cache misses
    pub cache_misses: u64,

    /// Average allocation size
    pub avg_allocation_size: f64,

    /// Pool efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,

    /// Fragmentation ratio (0.0 to 1.0)
    pub fragmentation_ratio: f64,

    /// Pool utilization over time
    pub utilization_history: Vec<(Instant, f64)>,

    /// Pool pressure level
    pub pressure_level: PressureLevel,
}

impl MemoryPoolStats {
    /// Creates new pool statistics
    pub fn new(name: String, total_capacity: usize) -> Self {
        Self {
            name,
            total_capacity,
            allocated_from_pool: 0,
            peak_allocated: 0,
            allocations_served: 0,
            cache_hits: 0,
            cache_misses: 0,
            avg_allocation_size: 0.0,
            efficiency_score: 1.0,
            fragmentation_ratio: 0.0,
            utilization_history: Vec::new(),
            pressure_level: PressureLevel::None,
        }
    }

    /// Calculates cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests > 0 {
            self.cache_hits as f64 / total_requests as f64
        } else {
            0.0
        }
    }

    /// Calculates pool utilization percentage
    pub fn utilization_percent(&self) -> f64 {
        if self.total_capacity > 0 {
            (self.allocated_from_pool as f64 / self.total_capacity as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Updates pool statistics
    pub fn update(&mut self, allocated: usize, allocations: u64, hits: u64, misses: u64) {
        self.allocated_from_pool = allocated;
        self.peak_allocated = self.peak_allocated.max(allocated);
        self.allocations_served = allocations;
        self.cache_hits = hits;
        self.cache_misses = misses;

        if allocations > 0 {
            self.avg_allocation_size = allocated as f64 / allocations as f64;
        }

        // Update utilization history
        self.utilization_history.push((Instant::now(), self.utilization_percent()));

        // Keep only recent history
        if self.utilization_history.len() > 1000 {
            self.utilization_history.remove(0);
        }

        // Update pressure level based on utilization
        self.pressure_level = PressureLevel::from_f64(self.utilization_percent() / 100.0);
    }

    /// Checks if pool needs attention
    pub fn needs_attention(&self) -> bool {
        self.utilization_percent() > 90.0 ||
        self.hit_rate() < 0.8 ||
        self.fragmentation_ratio > 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_snapshot_creation() {
        let snapshot = MemorySnapshot::new();
        assert!(snapshot.health_score() > 0.0);
        assert!(!snapshot.is_under_pressure());
    }

    #[test]
    fn test_device_memory_usage() {
        let mut usage = DeviceMemoryUsage::new(1024 * 1024 * 1024); // 1GB
        usage.update_usage(512 * 1024 * 1024, 0); // 512MB used
        assert_eq!(usage.utilization_percent, 50.0);
        assert!(!usage.is_critically_low());
    }

    #[test]
    fn test_pressure_level_conversion() {
        assert_eq!(PressureLevel::from_f64(0.95), PressureLevel::Critical);
        assert_eq!(PressureLevel::from_f64(0.5), PressureLevel::Medium);
        assert_eq!(PressureLevel::from_f64(0.1), PressureLevel::None);
    }

    #[test]
    fn test_pressure_event() {
        let mut event = MemoryPressureEvent::new(
            PressureLevel::High,
            None,
            MemoryType::Host,
            1024 * 1024 * 1024,
            100 * 1024 * 1024,
        );

        event.add_action(PressureAction::FreedUnusedMemory { amount: 50 * 1024 * 1024 });
        event.resolve();

        assert!(event.response_effectiveness() > 0.0);
        assert!(event.is_critical());
    }

    #[test]
    fn test_bandwidth_utilization() {
        let mut bandwidth = BandwidthUtilization::new(100.0); // 100 GB/s
        bandwidth.update_usage(75.0); // 75 GB/s

        assert_eq!(bandwidth.efficiency, 0.75);
        assert!(!bandwidth.is_saturated());

        bandwidth.update_usage(95.0);
        assert!(bandwidth.is_saturated());
    }

    #[test]
    fn test_memory_pool_stats() {
        let mut pool = MemoryPoolStats::new("test_pool".to_string(), 1024 * 1024);
        pool.update(512 * 1024, 100, 80, 20);

        assert_eq!(pool.utilization_percent(), 50.0);
        assert_eq!(pool.hit_rate(), 0.8);
        assert!(!pool.needs_attention());
    }

    #[test]
    fn test_pressure_indicators() {
        let mut indicators = MemoryPressureIndicators::default();
        indicators.system_pressure = PressureLevel::High;
        indicators.allocation_failure_rate = 0.15;

        assert!(indicators.pressure_score() > 0.5);
        assert!(indicators.requires_immediate_action());
    }
}