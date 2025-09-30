//! Memory pressure monitoring and management
//!
//! This module provides comprehensive memory pressure monitoring capabilities including:
//! - Real-time memory pressure detection and event tracking
//! - System and per-device memory usage monitoring
//! - Automated pressure response actions and mitigation strategies
//! - Bandwidth utilization tracking and optimization
//! - Memory pressure indicators and threshold management

use crate::memory_profiler::allocation::{MemoryType, PressureLevel};
use crate::Device;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Memory pressure event tracking
///
/// Records significant memory pressure events with detailed context
/// and tracking of mitigation actions taken.
#[derive(Debug, Clone)]
pub struct MemoryPressureEvent {
    /// Event timestamp
    pub timestamp: Instant,

    /// Pressure level
    pub pressure_level: PressureLevel,

    /// Affected device (if any)
    pub device: Option<Device>,

    /// Memory type affected
    pub memory_type: MemoryType,

    /// Total memory at time of event
    pub total_memory: usize,

    /// Available memory at time of event
    pub available_memory: usize,

    /// Actions taken
    pub actions_taken: Vec<PressureAction>,

    /// Event resolution time
    pub resolution_time: Option<Duration>,
}

/// Actions taken during memory pressure
///
/// Categorizes different mitigation strategies that can be employed
/// during memory pressure situations.
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
}

/// Memory usage snapshot
///
/// Captures a comprehensive view of memory usage across all devices
/// and system components at a specific point in time.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Snapshot timestamp
    pub timestamp: Instant,

    /// Per-device memory usage
    pub device_usage: HashMap<Device, DeviceMemoryUsage>,

    /// Host memory usage
    pub host_usage: HostMemoryUsage,

    /// System-wide memory pressure
    pub memory_pressure: f64,

    /// Active allocations count
    pub active_allocations: usize,

    /// Total allocated bytes
    pub total_allocated: usize,

    /// Memory fragmentation level
    pub fragmentation_level: f64,

    /// Bandwidth utilization
    pub bandwidth_utilization: BandwidthUtilization,
}

/// Device-specific memory usage
///
/// Tracks detailed memory usage statistics for individual compute devices.
#[derive(Debug, Clone)]
pub struct DeviceMemoryUsage {
    /// Total device memory
    pub total_memory: usize,

    /// Used memory
    pub used_memory: usize,

    /// Free memory
    pub free_memory: usize,

    /// Reserved memory
    pub reserved_memory: usize,

    /// Memory utilization percentage
    pub utilization_percent: f64,

    /// Memory bandwidth usage
    pub bandwidth_usage: f64,

    /// Active memory transfers
    pub active_transfers: usize,
}

/// Host memory usage information
///
/// Provides comprehensive host (CPU) memory usage statistics and pressure indicators.
#[derive(Debug, Clone)]
pub struct HostMemoryUsage {
    /// Total system memory
    pub total_memory: usize,

    /// Available memory
    pub available_memory: usize,

    /// Memory used by process
    pub process_memory: usize,

    /// Pinned memory usage
    pub pinned_memory: usize,

    /// Virtual memory usage
    pub virtual_memory: usize,

    /// Memory pressure indicators
    pub pressure_indicators: MemoryPressureIndicators,
}

/// Memory pressure indicators
///
/// Detailed indicators of memory pressure conditions at system and process levels.
#[derive(Debug, Clone)]
pub struct MemoryPressureIndicators {
    /// System memory pressure level
    pub system_pressure: PressureLevel,

    /// Process memory pressure level
    pub process_pressure: PressureLevel,

    /// Swap usage
    pub swap_usage: usize,

    /// Page fault rate
    pub page_fault_rate: f64,

    /// Memory allocation failure rate
    pub allocation_failure_rate: f64,
}

/// Bandwidth utilization statistics
///
/// Tracks memory bandwidth usage across devices and provides optimization insights.
#[derive(Debug, Clone)]
pub struct BandwidthUtilization {
    /// Memory bandwidth capacity (GB/s)
    pub total_bandwidth: f64,

    /// Current bandwidth usage (GB/s)
    pub current_usage: f64,

    /// Peak bandwidth usage (GB/s)
    pub peak_usage: f64,

    /// Bandwidth efficiency
    pub efficiency: f64,

    /// Per-device bandwidth breakdown
    pub device_breakdown: HashMap<Device, f64>,
}

/// Memory pressure monitor
///
/// Centralized monitoring and management of memory pressure across all devices
/// with automated response capabilities and detailed event tracking.
pub struct MemoryPressureMonitor {
    /// Current memory snapshots by device
    current_snapshots: Arc<RwLock<HashMap<Device, MemorySnapshot>>>,

    /// Historical pressure events
    pressure_events: Arc<Mutex<Vec<MemoryPressureEvent>>>,

    /// Global pressure statistics
    global_stats: Arc<Mutex<GlobalPressureStats>>,

    /// Pressure thresholds configuration
    thresholds: PressureThresholds,

    /// Event callbacks for pressure notifications
    event_callbacks: Vec<Box<dyn Fn(&MemoryPressureEvent) + Send + Sync>>,

    /// Last pressure check timestamp
    last_check: Arc<Mutex<Option<Instant>>>,

    /// Automatic mitigation enabled
    auto_mitigation: bool,
}

impl std::fmt::Debug for MemoryPressureMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryPressureMonitor")
            .field("current_snapshots", &self.current_snapshots)
            .field("pressure_events", &self.pressure_events)
            .field("global_stats", &self.global_stats)
            .field("thresholds", &self.thresholds)
            .field(
                "event_callbacks",
                &format!("{} callbacks", self.event_callbacks.len()),
            )
            .field("last_check", &self.last_check)
            .field("auto_mitigation", &self.auto_mitigation)
            .finish()
    }
}

/// Global pressure statistics
///
/// Aggregated statistics tracking memory pressure patterns and system health.
#[derive(Debug, Default)]
pub struct GlobalPressureStats {
    /// Total pressure events
    pub total_events: AtomicU64,

    /// Events by pressure level
    pub events_by_level: HashMap<PressureLevel, AtomicU64>,

    /// Total memory freed by mitigation
    pub total_memory_freed: AtomicUsize,

    /// Average event resolution time
    pub avg_resolution_time: AtomicU64, // In milliseconds

    /// Current system pressure level
    pub current_system_pressure: PressureLevel,

    /// Peak memory usage recorded
    pub peak_memory_usage: AtomicUsize,

    /// Memory pressure frequency (events per hour)
    pub pressure_frequency: f64,
}

/// Pressure threshold configuration
///
/// Configurable thresholds for triggering different levels of memory pressure responses.
#[derive(Debug, Clone)]
pub struct PressureThresholds {
    /// Low pressure threshold (percentage)
    pub low_pressure: f64,

    /// Medium pressure threshold (percentage)
    pub medium_pressure: f64,

    /// High pressure threshold (percentage)
    pub high_pressure: f64,

    /// Critical pressure threshold (percentage)
    pub critical_pressure: f64,

    /// Bandwidth utilization warning threshold
    pub bandwidth_warning: f64,

    /// Allocation failure rate threshold
    pub allocation_failure_threshold: f64,

    /// Page fault rate threshold
    pub page_fault_threshold: f64,
}

impl MemoryPressureEvent {
    /// Create a new memory pressure event
    pub fn new(
        pressure_level: PressureLevel,
        device: Option<Device>,
        memory_type: MemoryType,
        total_memory: usize,
        available_memory: usize,
    ) -> Self {
        Self {
            timestamp: Instant::now(),
            pressure_level,
            device,
            memory_type,
            total_memory,
            available_memory,
            actions_taken: Vec::new(),
            resolution_time: None,
        }
    }

    /// Add a pressure action to this event
    pub fn add_action(&mut self, action: PressureAction) {
        self.actions_taken.push(action);
    }

    /// Mark the event as resolved
    pub fn mark_resolved(&mut self) {
        self.resolution_time = Some(self.timestamp.elapsed());
    }

    /// Get memory usage percentage at time of event
    pub fn memory_usage_percent(&self) -> f64 {
        if self.total_memory == 0 {
            0.0
        } else {
            let used = self.total_memory.saturating_sub(self.available_memory);
            (used as f64 / self.total_memory as f64) * 100.0
        }
    }

    /// Check if event was resolved successfully
    pub fn is_resolved(&self) -> bool {
        self.resolution_time.is_some()
    }

    /// Get total memory freed by actions
    pub fn total_memory_freed(&self) -> usize {
        self.actions_taken
            .iter()
            .map(|action| match action {
                PressureAction::FreedUnusedMemory { amount } => *amount,
                PressureAction::SwappedToDisk { amount } => *amount,
                PressureAction::ReducedCaches { cache_reduction } => *cache_reduction,
                _ => 0,
            })
            .sum()
    }
}

impl MemorySnapshot {
    /// Create a new memory snapshot
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
        }
    }

    /// Calculate overall system memory pressure
    pub fn calculate_system_pressure(&mut self) {
        let mut total_pressure = 0.0;
        let mut device_count = 0;

        // Include host pressure
        total_pressure += self.host_usage.get_pressure_score();
        device_count += 1;

        // Include device pressure
        for usage in self.device_usage.values() {
            total_pressure += usage.utilization_percent / 100.0;
            device_count += 1;
        }

        self.memory_pressure = if device_count > 0 {
            total_pressure / device_count as f64
        } else {
            0.0
        };
    }

    /// Get the highest pressure device
    pub fn highest_pressure_device(&self) -> Option<(Device, f64)> {
        self.device_usage
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.utilization_percent
                    .partial_cmp(&b.utilization_percent)
                    .unwrap()
            })
            .map(|(device, usage)| (device.clone(), usage.utilization_percent))
    }

    /// Check if any device is experiencing critical pressure
    pub fn has_critical_pressure(&self, threshold: f64) -> bool {
        self.memory_pressure > threshold
            || self
                .device_usage
                .values()
                .any(|usage| usage.utilization_percent > threshold * 100.0)
    }
}

impl DeviceMemoryUsage {
    /// Create new device memory usage
    pub fn new(total_memory: usize) -> Self {
        Self {
            total_memory,
            used_memory: 0,
            free_memory: total_memory,
            reserved_memory: 0,
            utilization_percent: 0.0,
            bandwidth_usage: 0.0,
            active_transfers: 0,
        }
    }

    /// Update memory usage and recalculate percentages
    pub fn update_usage(&mut self, used: usize, reserved: usize) {
        self.used_memory = used;
        self.reserved_memory = reserved;
        self.free_memory = self.total_memory.saturating_sub(used + reserved);
        self.utilization_percent = if self.total_memory > 0 {
            ((used + reserved) as f64 / self.total_memory as f64) * 100.0
        } else {
            0.0
        };
    }

    /// Check if device is under memory pressure
    pub fn is_under_pressure(&self, threshold: f64) -> bool {
        self.utilization_percent > threshold
    }

    /// Get available memory for allocation
    pub fn available_memory(&self) -> usize {
        self.free_memory
    }

    /// Get fragmentation ratio (reserved vs used)
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.used_memory > 0 {
            self.reserved_memory as f64 / self.used_memory as f64
        } else {
            0.0
        }
    }
}

impl HostMemoryUsage {
    /// Create default host memory usage
    pub fn default() -> Self {
        Self {
            total_memory: 0,
            available_memory: 0,
            process_memory: 0,
            pinned_memory: 0,
            virtual_memory: 0,
            pressure_indicators: MemoryPressureIndicators::default(),
        }
    }

    /// Get overall pressure score for host memory
    pub fn get_pressure_score(&self) -> f64 {
        if self.total_memory == 0 {
            return 0.0;
        }

        let usage_ratio =
            (self.total_memory - self.available_memory) as f64 / self.total_memory as f64;
        let pressure_multiplier = match self.pressure_indicators.system_pressure {
            PressureLevel::None => 1.0,
            PressureLevel::Low => 1.2,
            PressureLevel::Medium => 1.5,
            PressureLevel::High => 2.0,
            PressureLevel::Critical => 3.0,
        };

        usage_ratio * pressure_multiplier
    }

    /// Check if host memory is critically low
    /// threshold is the minimum available memory in GB
    pub fn is_critically_low(&self, threshold: f64) -> bool {
        let available_gb = self.available_memory as f64 / (1024.0 * 1024.0 * 1024.0);
        available_gb < threshold
    }

    /// Update pressure indicators
    pub fn update_pressure_indicators(&mut self, page_faults: f64, alloc_failures: f64) {
        self.pressure_indicators.page_fault_rate = page_faults;
        self.pressure_indicators.allocation_failure_rate = alloc_failures;

        // Determine pressure level based on multiple factors
        let usage_ratio =
            (self.total_memory - self.available_memory) as f64 / self.total_memory as f64;

        self.pressure_indicators.system_pressure = if usage_ratio > 0.95 || alloc_failures > 0.1 {
            PressureLevel::Critical
        } else if usage_ratio > 0.85 || page_faults > 1000.0 {
            PressureLevel::High
        } else if usage_ratio > 0.75 || page_faults > 500.0 {
            PressureLevel::Medium
        } else if usage_ratio > 0.60 {
            PressureLevel::Low
        } else {
            PressureLevel::None
        };
    }
}

impl MemoryPressureIndicators {
    /// Create default pressure indicators
    pub fn default() -> Self {
        Self {
            system_pressure: PressureLevel::None,
            process_pressure: PressureLevel::None,
            swap_usage: 0,
            page_fault_rate: 0.0,
            allocation_failure_rate: 0.0,
        }
    }

    /// Get combined pressure score
    pub fn combined_pressure_score(&self) -> f64 {
        let system_score = match self.system_pressure {
            PressureLevel::None => 0.0,
            PressureLevel::Low => 0.2,
            PressureLevel::Medium => 0.4,
            PressureLevel::High => 0.7,
            PressureLevel::Critical => 1.0,
        };

        let process_score = match self.process_pressure {
            PressureLevel::None => 0.0,
            PressureLevel::Low => 0.2,
            PressureLevel::Medium => 0.4,
            PressureLevel::High => 0.7,
            PressureLevel::Critical => 1.0,
        };

        (system_score + process_score) / 2.0
    }

    /// Check if immediate action is required
    pub fn requires_immediate_action(&self) -> bool {
        matches!(self.system_pressure, PressureLevel::Critical)
            || matches!(self.process_pressure, PressureLevel::Critical)
            || self.allocation_failure_rate > 0.1
    }
}

impl BandwidthUtilization {
    /// Create default bandwidth utilization
    pub fn default() -> Self {
        Self {
            total_bandwidth: 0.0,
            current_usage: 0.0,
            peak_usage: 0.0,
            efficiency: 0.0,
            device_breakdown: HashMap::new(),
        }
    }

    /// Update bandwidth usage
    pub fn update_usage(&mut self, current: f64) {
        self.current_usage = current;
        if current > self.peak_usage {
            self.peak_usage = current;
        }

        self.efficiency = if self.total_bandwidth > 0.0 {
            (self.current_usage / self.total_bandwidth).min(1.0)
        } else {
            0.0
        };
    }

    /// Check if bandwidth is underutilized
    pub fn is_underutilized(&self, threshold: f64) -> bool {
        self.efficiency < threshold
    }

    /// Check if bandwidth is saturated
    pub fn is_saturated(&self, threshold: f64) -> bool {
        self.efficiency > threshold
    }

    /// Get bandwidth headroom
    pub fn headroom_gbps(&self) -> f64 {
        (self.total_bandwidth - self.current_usage).max(0.0)
    }
}

impl MemoryPressureMonitor {
    /// Create a new memory pressure monitor
    pub fn new(thresholds: PressureThresholds, auto_mitigation: bool) -> Self {
        let mut events_by_level = HashMap::new();
        events_by_level.insert(PressureLevel::Low, AtomicU64::new(0));
        events_by_level.insert(PressureLevel::Medium, AtomicU64::new(0));
        events_by_level.insert(PressureLevel::High, AtomicU64::new(0));
        events_by_level.insert(PressureLevel::Critical, AtomicU64::new(0));

        let global_stats = GlobalPressureStats {
            total_events: AtomicU64::new(0),
            events_by_level,
            total_memory_freed: AtomicUsize::new(0),
            avg_resolution_time: AtomicU64::new(0),
            current_system_pressure: PressureLevel::None,
            peak_memory_usage: AtomicUsize::new(0),
            pressure_frequency: 0.0,
        };

        Self {
            current_snapshots: Arc::new(RwLock::new(HashMap::new())),
            pressure_events: Arc::new(Mutex::new(Vec::new())),
            global_stats: Arc::new(Mutex::new(global_stats)),
            thresholds,
            event_callbacks: Vec::new(),
            last_check: Arc::new(Mutex::new(None)),
            auto_mitigation,
        }
    }

    /// Update memory snapshot for a device
    pub fn update_snapshot(&self, device: Device, snapshot: MemorySnapshot) {
        let mut snapshots = self.current_snapshots.write();
        snapshots.insert(device, snapshot);
    }

    /// Check for memory pressure across all devices
    pub fn check_memory_pressure(&self) -> Vec<MemoryPressureEvent> {
        let mut events = Vec::new();
        let snapshots = self.current_snapshots.read();

        for (device, snapshot) in snapshots.iter() {
            // Check device pressure
            if let Some(device_usage) = snapshot.device_usage.get(device) {
                if let Some(event) = self.check_device_pressure(device.clone(), device_usage) {
                    events.push(event);
                }
            }

            // Check host pressure
            if let Some(event) = self.check_host_pressure(&snapshot.host_usage) {
                events.push(event);
            }

            // Check bandwidth pressure
            if let Some(event) = self.check_bandwidth_pressure(&snapshot.bandwidth_utilization) {
                events.push(event);
            }
        }

        // Record events and trigger callbacks
        for event in &events {
            self.record_pressure_event(event.clone());
        }

        *self.last_check.lock() = Some(Instant::now());
        events
    }

    /// Record a pressure event
    pub fn record_pressure_event(&self, event: MemoryPressureEvent) {
        // Update global statistics
        {
            let mut stats = self.global_stats.lock();
            stats.total_events.fetch_add(1, Ordering::Relaxed);

            if let Some(counter) = stats.events_by_level.get(&event.pressure_level) {
                counter.fetch_add(1, Ordering::Relaxed);
            }

            if event.pressure_level > stats.current_system_pressure {
                stats.current_system_pressure = event.pressure_level;
            }
        }

        // Trigger callbacks
        for callback in &self.event_callbacks {
            callback(&event);
        }

        // Store event
        self.pressure_events.lock().push(event);
    }

    /// Get recent pressure events
    pub fn recent_events(&self, since: Duration) -> Vec<MemoryPressureEvent> {
        let cutoff = Instant::now() - since;
        self.pressure_events
            .lock()
            .iter()
            .filter(|event| event.timestamp > cutoff)
            .cloned()
            .collect()
    }

    /// Check device-specific pressure
    fn check_device_pressure(
        &self,
        device: Device,
        usage: &DeviceMemoryUsage,
    ) -> Option<MemoryPressureEvent> {
        let pressure_level = if usage.utilization_percent > self.thresholds.critical_pressure {
            PressureLevel::Critical
        } else if usage.utilization_percent > self.thresholds.high_pressure {
            PressureLevel::High
        } else if usage.utilization_percent > self.thresholds.medium_pressure {
            PressureLevel::Medium
        } else if usage.utilization_percent > self.thresholds.low_pressure {
            PressureLevel::Low
        } else {
            return None;
        };

        Some(MemoryPressureEvent::new(
            pressure_level,
            Some(device),
            MemoryType::Device,
            usage.total_memory,
            usage.available_memory(),
        ))
    }

    /// Check host memory pressure
    fn check_host_pressure(&self, usage: &HostMemoryUsage) -> Option<MemoryPressureEvent> {
        let pressure_score = usage.get_pressure_score();

        let pressure_level = if pressure_score > self.thresholds.critical_pressure {
            PressureLevel::Critical
        } else if pressure_score > self.thresholds.high_pressure {
            PressureLevel::High
        } else if pressure_score > self.thresholds.medium_pressure {
            PressureLevel::Medium
        } else if pressure_score > self.thresholds.low_pressure {
            PressureLevel::Low
        } else {
            return None;
        };

        Some(MemoryPressureEvent::new(
            pressure_level,
            None,
            MemoryType::Host,
            usage.total_memory,
            usage.available_memory,
        ))
    }

    /// Check bandwidth pressure
    fn check_bandwidth_pressure(
        &self,
        bandwidth: &BandwidthUtilization,
    ) -> Option<MemoryPressureEvent> {
        if bandwidth.efficiency > self.thresholds.bandwidth_warning {
            let pressure_level = if bandwidth.efficiency > 0.95 {
                PressureLevel::Critical
            } else if bandwidth.efficiency > 0.85 {
                PressureLevel::High
            } else {
                PressureLevel::Medium
            };

            // Create a synthetic pressure event for bandwidth saturation
            Some(MemoryPressureEvent::new(
                pressure_level,
                None,
                MemoryType::Host, // Bandwidth affects host-device transfers
                (bandwidth.total_bandwidth * 1024.0 * 1024.0 * 1024.0) as usize, // Convert GB/s to bytes
                (bandwidth.headroom_gbps() * 1024.0 * 1024.0 * 1024.0) as usize,
            ))
        } else {
            None
        }
    }
}

impl Default for PressureThresholds {
    fn default() -> Self {
        Self {
            low_pressure: 60.0,
            medium_pressure: 75.0,
            high_pressure: 85.0,
            critical_pressure: 95.0,
            bandwidth_warning: 80.0,
            allocation_failure_threshold: 0.05,
            page_fault_threshold: 1000.0,
        }
    }
}

impl std::fmt::Display for PressureAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PressureAction::FreedUnusedMemory { amount } => {
                write!(f, "Freed {} bytes of unused memory", amount)
            }
            PressureAction::CompactedPools { pools_affected } => {
                write!(f, "Compacted {} memory pools", pools_affected)
            }
            PressureAction::TriggeredGarbageCollection => {
                write!(f, "Triggered garbage collection")
            }
            PressureAction::ReducedCaches { cache_reduction } => {
                write!(f, "Reduced caches by {} bytes", cache_reduction)
            }
            PressureAction::SwappedToDisk { amount } => {
                write!(f, "Swapped {} bytes to disk", amount)
            }
            PressureAction::KilledAllocations { count } => {
                write!(f, "Killed {} low-priority allocations", count)
            }
            PressureAction::RequestedMoreMemory { amount } => {
                write!(f, "Requested {} additional bytes from system", amount)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pressure_event_creation() {
        let event = MemoryPressureEvent::new(
            PressureLevel::High,
            None,
            MemoryType::Host,
            1024 * 1024 * 1024, // 1GB
            128 * 1024 * 1024,  // 128MB available
        );

        assert_eq!(event.pressure_level, PressureLevel::High);
        assert!(event.memory_usage_percent() > 85.0);
        assert!(!event.is_resolved());
    }

    #[test]
    fn test_device_memory_usage() {
        let mut usage = DeviceMemoryUsage::new(1024 * 1024 * 1024); // 1GB
        usage.update_usage(512 * 1024 * 1024, 128 * 1024 * 1024); // 512MB used, 128MB reserved

        assert_eq!(usage.utilization_percent, 62.5); // (512 + 128) / 1024 * 100
        assert!(usage.is_under_pressure(60.0));
        assert!(!usage.is_under_pressure(70.0));
    }

    #[test]
    fn test_host_memory_pressure_calculation() {
        let mut host_usage = HostMemoryUsage::default();
        host_usage.total_memory = 8 * 1024 * 1024 * 1024; // 8GB
        host_usage.available_memory = 1 * 1024 * 1024 * 1024; // 1GB available

        host_usage.update_pressure_indicators(100.0, 0.01);

        let pressure_score = host_usage.get_pressure_score();
        assert!(pressure_score > 0.8); // Should indicate high pressure
        assert!(!host_usage.is_critically_low(0.5)); // Not critically low with 1GB available
    }

    #[test]
    fn test_bandwidth_utilization() {
        let mut bandwidth = BandwidthUtilization::default();
        bandwidth.total_bandwidth = 100.0; // 100 GB/s
        bandwidth.update_usage(85.0); // 85 GB/s current usage

        assert_eq!(bandwidth.efficiency, 0.85);
        assert!(bandwidth.is_saturated(0.80));
        assert!(!bandwidth.is_underutilized(0.50));
        assert_eq!(bandwidth.headroom_gbps(), 15.0);
    }

    #[test]
    fn test_memory_snapshot() {
        let mut snapshot = MemorySnapshot::new();

        // Add host usage
        snapshot.host_usage.total_memory = 8 * 1024 * 1024 * 1024;
        snapshot.host_usage.available_memory = 2 * 1024 * 1024 * 1024;

        // Add device usage
        let device = Device::cpu().unwrap(); // Assuming Device::cpu exists
        let mut device_usage = DeviceMemoryUsage::new(4 * 1024 * 1024 * 1024);
        device_usage.update_usage(3 * 1024 * 1024 * 1024, 0);
        snapshot.device_usage.insert(device, device_usage);

        snapshot.calculate_system_pressure();

        assert!(snapshot.memory_pressure > 0.5); // Should show moderate pressure
        assert!(snapshot.has_critical_pressure(0.7)); // Device is at 75% utilization, should be critical above 70%
    }

    #[test]
    fn test_pressure_monitor() {
        let thresholds = PressureThresholds::default();
        let monitor = MemoryPressureMonitor::new(thresholds, true);

        let mut snapshot = MemorySnapshot::new();
        let device = Device::cpu().unwrap();

        // Create high pressure scenario
        let mut device_usage = DeviceMemoryUsage::new(1024 * 1024 * 1024);
        device_usage.update_usage(900 * 1024 * 1024, 0); // 87.9% utilization
        snapshot.device_usage.insert(device.clone(), device_usage);

        monitor.update_snapshot(device, snapshot);
        let events = monitor.check_memory_pressure();

        assert!(!events.is_empty());
        assert!(events
            .iter()
            .any(|e| e.pressure_level >= PressureLevel::High));
    }

    #[test]
    fn test_pressure_thresholds() {
        let thresholds = PressureThresholds::default();

        assert_eq!(thresholds.low_pressure, 60.0);
        assert_eq!(thresholds.critical_pressure, 95.0);
        assert!(thresholds.medium_pressure > thresholds.low_pressure);
        assert!(thresholds.high_pressure > thresholds.medium_pressure);
    }
}
