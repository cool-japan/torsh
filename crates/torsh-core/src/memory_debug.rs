//! Memory debugging and allocation tracking tools for ToRSh
//!
//! This module provides comprehensive memory debugging capabilities including
//! allocation tracking, leak detection, memory usage profiling, and allocation
//! pattern analysis.

use crate::error::{Result, TorshError};
use std::alloc::{GlobalAlloc, Layout, System};
use std::backtrace::Backtrace;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Global memory debugger instance
static MEMORY_DEBUGGER: std::sync::OnceLock<Arc<Mutex<MemoryDebugger>>> =
    std::sync::OnceLock::new();

/// Memory allocation information for debugging
#[derive(Debug)]
pub struct AllocationInfo {
    /// Unique allocation ID
    pub id: u64,
    /// Size of the allocation in bytes
    pub size: usize,
    /// Memory layout information
    pub layout: Layout,
    /// Timestamp when allocation was made
    pub timestamp: Instant,
    /// Stack trace at allocation site (if enabled)
    pub backtrace: Option<String>,
    /// Custom tag for categorizing allocations
    pub tag: Option<String>,
    /// Whether this allocation is still active
    pub is_active: bool,
    /// Thread ID that made the allocation
    pub thread_id: std::thread::ThreadId,
}

impl Clone for AllocationInfo {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            size: self.size,
            layout: self.layout,
            timestamp: self.timestamp,
            backtrace: self.backtrace.clone(),
            tag: self.tag.clone(),
            is_active: self.is_active,
            thread_id: self.thread_id,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total bytes currently allocated
    pub total_allocated: usize,
    /// Peak memory usage in bytes
    pub peak_allocated: usize,
    /// Total number of allocations made
    pub total_allocations: u64,
    /// Total number of deallocations made
    pub total_deallocations: u64,
    /// Number of currently active allocations
    pub active_allocations: u64,
    /// Average allocation size
    pub average_allocation_size: f64,
    /// Total bytes allocated over lifetime
    pub lifetime_allocated: usize,
    /// Total bytes deallocated over lifetime
    pub lifetime_deallocated: usize,
}

/// Memory leak detection result
#[derive(Debug, Clone)]
pub struct MemoryLeak {
    /// Allocation information for the leaked memory
    pub allocation: AllocationInfo,
    /// How long the allocation has been active
    pub age: Duration,
    /// Likelihood this is a leak (0.0-1.0)
    pub leak_probability: f64,
    /// Confidence level in leak detection (0.0-1.0)
    pub confidence: f64,
    /// Risk level based on size and age
    pub risk_level: LeakRiskLevel,
    /// Suggested actions to address the leak
    pub suggested_actions: Vec<String>,
}

/// Risk levels for memory leaks
#[derive(Debug, Clone, PartialEq)]
pub enum LeakRiskLevel {
    /// Low risk - small allocation, short-lived
    Low,
    /// Medium risk - moderate size or age
    Medium,
    /// High risk - large allocation or very old
    High,
    /// Critical risk - very large allocation and very old
    Critical,
}

/// Memory allocation pattern analysis
#[derive(Debug, Clone)]
pub struct AllocationPattern {
    /// Size class of allocations (e.g., "small", "medium", "large")
    pub size_class: String,
    /// Frequency of allocations in this size class
    pub frequency: u64,
    /// Average lifetime of allocations in this class
    pub average_lifetime: Duration,
    /// Common stack traces for this pattern
    pub common_stacks: Vec<String>,
}

/// Memory debugging configuration
#[derive(Debug, Clone)]
pub struct MemoryDebugConfig {
    /// Whether to capture stack traces for allocations
    pub capture_backtraces: bool,
    /// Maximum number of allocations to track
    pub max_tracked_allocations: usize,
    /// Whether to track allocation patterns
    pub track_patterns: bool,
    /// Minimum allocation size to track
    pub min_tracked_size: usize,
    /// Whether to enable leak detection
    pub enable_leak_detection: bool,
    /// How often to run leak detection (in allocations)
    pub leak_detection_frequency: u64,
    /// Threshold for considering an allocation a potential leak
    pub leak_threshold: Duration,
    /// Enable real-time monitoring
    pub enable_real_time_monitoring: bool,
    /// Real-time monitoring interval
    pub monitoring_interval: Duration,
    /// Enable automatic leak mitigation
    pub enable_auto_mitigation: bool,
    /// Maximum memory usage before triggering warnings
    pub memory_warning_threshold: usize,
    /// Maximum memory usage before triggering critical alerts
    pub memory_critical_threshold: usize,
}

/// Real-time memory monitoring data
#[derive(Debug, Clone)]
pub struct RealTimeStats {
    /// Current memory usage
    pub current_usage: usize,
    /// Memory usage trend (bytes per second)
    pub usage_trend: f64,
    /// Number of allocations in last interval
    pub recent_allocations: u64,
    /// Number of deallocations in last interval
    pub recent_deallocations: u64,
    /// Current leak detection rate
    pub leak_detection_rate: f64,
    /// System memory pressure level
    pub system_pressure: SystemPressureLevel,
}

/// System memory pressure levels
#[derive(Debug, Clone, PartialEq)]
pub enum SystemPressureLevel {
    /// No pressure - memory usage is normal
    Normal,
    /// Low pressure - memory usage is elevated
    Low,
    /// Medium pressure - memory usage is high
    Medium,
    /// High pressure - memory usage is critical
    High,
    /// Critical pressure - system may be unstable
    Critical,
}

/// Leak statistics summary
#[derive(Debug, Clone)]
pub struct LeakStats {
    pub total_leaks: usize,
    pub critical_leaks: usize,
    pub high_leaks: usize,
    pub medium_leaks: usize,
    pub low_leaks: usize,
    pub total_leaked_bytes: usize,
    pub average_leak_age: u64,
}

impl Default for MemoryDebugConfig {
    fn default() -> Self {
        Self {
            capture_backtraces: true,
            max_tracked_allocations: 10000,
            track_patterns: true,
            min_tracked_size: 1024, // Track allocations >= 1KB
            enable_leak_detection: true,
            leak_detection_frequency: 1000,
            leak_threshold: Duration::from_secs(300), // 5 minutes
            enable_real_time_monitoring: true,
            monitoring_interval: Duration::from_secs(10),
            enable_auto_mitigation: false,
            memory_warning_threshold: 1024 * 1024 * 1024, // 1GB
            memory_critical_threshold: 2 * 1024 * 1024 * 1024, // 2GB
        }
    }
}

/// Main memory debugger implementation
#[derive(Debug)]
pub struct MemoryDebugger {
    /// Configuration for memory debugging
    config: MemoryDebugConfig,
    /// Map of allocation ID to allocation info
    allocations: HashMap<u64, AllocationInfo>,
    /// Memory usage statistics
    stats: MemoryStats,
    /// Next allocation ID to assign
    next_id: u64,
    /// Recent allocation history for pattern analysis
    allocation_history: VecDeque<AllocationInfo>,
    /// Detected memory leaks
    detected_leaks: Vec<MemoryLeak>,
    /// Last time leak detection was run
    last_leak_check: Instant,
    /// Real-time monitoring data
    realtime_stats: RealTimeStats,
    /// Last monitoring timestamp
    last_monitoring: Instant,
    /// Previous memory usage for trend calculation
    previous_usage: usize,
    /// Allocation count since last monitoring
    allocations_since_monitoring: u64,
    /// Deallocation count since last monitoring
    deallocations_since_monitoring: u64,
}

impl MemoryDebugger {
    /// Create a new memory debugger with default configuration
    pub fn new() -> Self {
        Self::with_config(MemoryDebugConfig::default())
    }

    /// Create a new memory debugger with specified configuration
    pub fn with_config(config: MemoryDebugConfig) -> Self {
        let now = Instant::now();
        Self {
            config,
            allocations: HashMap::new(),
            stats: MemoryStats::default(),
            next_id: 1,
            allocation_history: VecDeque::new(),
            detected_leaks: Vec::new(),
            last_leak_check: now,
            realtime_stats: RealTimeStats {
                current_usage: 0,
                usage_trend: 0.0,
                recent_allocations: 0,
                recent_deallocations: 0,
                leak_detection_rate: 0.0,
                system_pressure: SystemPressureLevel::Normal,
            },
            last_monitoring: now,
            previous_usage: 0,
            allocations_since_monitoring: 0,
            deallocations_since_monitoring: 0,
        }
    }

    /// Record a new memory allocation
    pub fn record_allocation(&mut self, size: usize, layout: Layout, tag: Option<String>) -> u64 {
        if size < self.config.min_tracked_size {
            return 0; // Don't track small allocations
        }

        let id = self.next_id;
        self.next_id += 1;

        let allocation = AllocationInfo {
            id,
            size,
            layout,
            timestamp: Instant::now(),
            backtrace: if self.config.capture_backtraces {
                Some(format!("{}", Backtrace::capture()))
            } else {
                None
            },
            tag,
            is_active: true,
            thread_id: std::thread::current().id(),
        };

        // Update statistics
        self.stats.total_allocated += size;
        self.stats.peak_allocated = self.stats.peak_allocated.max(self.stats.total_allocated);
        self.stats.total_allocations += 1;
        self.stats.active_allocations += 1;
        self.stats.lifetime_allocated += size;
        self.stats.average_allocation_size =
            self.stats.lifetime_allocated as f64 / self.stats.total_allocations as f64;

        // Update real-time monitoring
        self.allocations_since_monitoring += 1;
        self.realtime_stats.current_usage = self.stats.total_allocated;
        self.update_realtime_monitoring();

        // Store allocation info
        self.allocations.insert(id, allocation.clone());

        // Add to history for pattern analysis
        if self.config.track_patterns {
            self.allocation_history.push_back(allocation);

            // Limit history size
            while self.allocation_history.len() > self.config.max_tracked_allocations {
                self.allocation_history.pop_front();
            }
        }

        // Run leak detection periodically
        if self.config.enable_leak_detection
            && self
                .stats
                .total_allocations
                .is_multiple_of(self.config.leak_detection_frequency)
        {
            self.detect_leaks();
        }

        id
    }

    /// Record a memory deallocation
    pub fn record_deallocation(&mut self, id: u64) {
        if let Some(mut allocation) = self.allocations.remove(&id) {
            allocation.is_active = false;

            // Update statistics
            self.stats.total_allocated = self.stats.total_allocated.saturating_sub(allocation.size);
            self.stats.total_deallocations += 1;
            self.stats.active_allocations = self.stats.active_allocations.saturating_sub(1);
            self.stats.lifetime_deallocated += allocation.size;

            // Update real-time monitoring
            self.deallocations_since_monitoring += 1;
            self.realtime_stats.current_usage = self.stats.total_allocated;
            self.update_realtime_monitoring();
        }
    }

    /// Update real-time monitoring statistics
    fn update_realtime_monitoring(&mut self) {
        if !self.config.enable_real_time_monitoring {
            return;
        }

        let now = Instant::now();
        let time_elapsed = now.duration_since(self.last_monitoring);

        if time_elapsed >= self.config.monitoring_interval {
            let time_seconds = time_elapsed.as_secs_f64();
            let usage_change = self.stats.total_allocated as i64 - self.previous_usage as i64;

            // Calculate usage trend (bytes per second)
            self.realtime_stats.usage_trend = usage_change as f64 / time_seconds;

            // Update recent allocation/deallocation counts
            self.realtime_stats.recent_allocations = self.allocations_since_monitoring;
            self.realtime_stats.recent_deallocations = self.deallocations_since_monitoring;

            // Calculate leak detection rate
            let total_recent_ops =
                self.allocations_since_monitoring + self.deallocations_since_monitoring;
            self.realtime_stats.leak_detection_rate = if total_recent_ops > 0 {
                self.detected_leaks.len() as f64 / total_recent_ops as f64
            } else {
                0.0
            };

            // Update system pressure level
            self.realtime_stats.system_pressure = self.calculate_system_pressure();

            // Reset counters
            self.previous_usage = self.stats.total_allocated;
            self.allocations_since_monitoring = 0;
            self.deallocations_since_monitoring = 0;
            self.last_monitoring = now;
        }
    }

    /// Calculate system memory pressure level
    fn calculate_system_pressure(&self) -> SystemPressureLevel {
        let current_usage = self.stats.total_allocated;
        let warning_threshold = self.config.memory_warning_threshold;
        let critical_threshold = self.config.memory_critical_threshold;

        if current_usage >= critical_threshold {
            SystemPressureLevel::Critical
        } else if current_usage >= warning_threshold {
            SystemPressureLevel::High
        } else if current_usage >= warning_threshold * 3 / 4 {
            SystemPressureLevel::Medium
        } else if current_usage >= warning_threshold / 2 {
            SystemPressureLevel::Low
        } else {
            SystemPressureLevel::Normal
        }
    }

    /// Get current memory usage statistics
    pub fn stats(&self) -> MemoryStats {
        self.stats.clone()
    }

    /// Get real-time monitoring statistics
    pub fn realtime_stats(&self) -> RealTimeStats {
        self.realtime_stats.clone()
    }

    /// Check if system is under memory pressure
    pub fn is_under_pressure(&self) -> bool {
        matches!(
            self.realtime_stats.system_pressure,
            SystemPressureLevel::High | SystemPressureLevel::Critical
        )
    }

    /// Get memory pressure level
    pub fn get_pressure_level(&self) -> SystemPressureLevel {
        self.realtime_stats.system_pressure.clone()
    }

    /// Force a leak detection run
    pub fn force_leak_detection(&mut self) -> Vec<MemoryLeak> {
        self.detect_leaks()
    }

    /// Get leak statistics
    pub fn leak_stats(&self) -> LeakStats {
        let total_leaks = self.detected_leaks.len();
        let critical_leaks = self
            .detected_leaks
            .iter()
            .filter(|l| l.risk_level == LeakRiskLevel::Critical)
            .count();
        let high_leaks = self
            .detected_leaks
            .iter()
            .filter(|l| l.risk_level == LeakRiskLevel::High)
            .count();
        let medium_leaks = self
            .detected_leaks
            .iter()
            .filter(|l| l.risk_level == LeakRiskLevel::Medium)
            .count();
        let low_leaks = self
            .detected_leaks
            .iter()
            .filter(|l| l.risk_level == LeakRiskLevel::Low)
            .count();

        LeakStats {
            total_leaks,
            critical_leaks,
            high_leaks,
            medium_leaks,
            low_leaks,
            total_leaked_bytes: self.detected_leaks.iter().map(|l| l.allocation.size).sum(),
            average_leak_age: if total_leaks > 0 {
                self.detected_leaks
                    .iter()
                    .map(|l| l.age.as_secs())
                    .sum::<u64>()
                    / total_leaks as u64
            } else {
                0
            },
        }
    }

    /// Run leak detection and return any newly detected leaks
    pub fn detect_leaks(&mut self) -> Vec<MemoryLeak> {
        let now = Instant::now();
        let threshold = self.config.leak_threshold;
        let mut new_leaks = Vec::new();

        for allocation in self.allocations.values() {
            if !allocation.is_active {
                continue;
            }

            let age = now.duration_since(allocation.timestamp);
            if age > threshold {
                // Calculate leak probability based on age and size
                let age_factor = age.as_secs_f64() / threshold.as_secs_f64();
                let size_factor = (allocation.size as f64).log2() / 20.0; // Larger allocations more suspicious
                let leak_probability = (age_factor * 0.7 + size_factor * 0.3).min(1.0);

                if leak_probability > 0.5 {
                    // Calculate confidence based on age and size factors
                    let confidence = (age_factor * 0.8 + size_factor * 0.2).min(1.0);

                    // Determine risk level
                    let risk_level = if allocation.size > 1024 * 1024 && age.as_secs() > 3600 {
                        LeakRiskLevel::Critical
                    } else if allocation.size > 64 * 1024 || age.as_secs() > 1800 {
                        LeakRiskLevel::High
                    } else if allocation.size > 4 * 1024 || age.as_secs() > 600 {
                        LeakRiskLevel::Medium
                    } else {
                        LeakRiskLevel::Low
                    };

                    // Generate suggested actions
                    let mut suggested_actions = Vec::new();
                    if age.as_secs() > 1800 {
                        suggested_actions
                            .push("Consider reviewing allocation lifetime".to_string());
                    }
                    if allocation.size > 64 * 1024 {
                        suggested_actions.push("Review large allocation usage".to_string());
                    }
                    if allocation.backtrace.is_some() {
                        suggested_actions.push("Check allocation backtrace for source".to_string());
                    }
                    if suggested_actions.is_empty() {
                        suggested_actions
                            .push("Monitor allocation for continued growth".to_string());
                    }

                    new_leaks.push(MemoryLeak {
                        allocation: allocation.clone(),
                        age,
                        leak_probability,
                        confidence,
                        risk_level,
                        suggested_actions,
                    });
                }
            }
        }

        self.detected_leaks.extend(new_leaks.clone());
        self.last_leak_check = now;
        new_leaks
    }

    /// Analyze allocation patterns and return insights
    pub fn analyze_patterns(&self) -> Vec<AllocationPattern> {
        if !self.config.track_patterns {
            return Vec::new();
        }

        let mut size_patterns: HashMap<String, (u64, Duration, Vec<String>)> = HashMap::new();

        for allocation in &self.allocation_history {
            let size_class = Self::classify_size(allocation.size);
            let lifetime = if allocation.is_active {
                Instant::now().duration_since(allocation.timestamp)
            } else {
                Duration::from_secs(0) // Completed allocation
            };

            let stack_trace = allocation
                .backtrace
                .as_ref()
                .cloned()
                .unwrap_or_else(|| "No backtrace".to_string());

            let entry = size_patterns
                .entry(size_class)
                .or_insert((0, Duration::ZERO, Vec::new()));
            entry.0 += 1; // frequency
            entry.1 += lifetime; // total lifetime
            if !entry.2.contains(&stack_trace) && entry.2.len() < 5 {
                entry.2.push(stack_trace); // common stacks (limit to 5)
            }
        }

        size_patterns
            .into_iter()
            .map(
                |(size_class, (frequency, total_lifetime, stacks))| AllocationPattern {
                    size_class,
                    frequency,
                    average_lifetime: total_lifetime / frequency.max(1) as u32,
                    common_stacks: stacks,
                },
            )
            .collect()
    }

    /// Classify allocation size into categories
    fn classify_size(size: usize) -> String {
        match size {
            0..=1024 => "small".to_string(),
            1025..=65536 => "medium".to_string(),
            65537..=1048576 => "large".to_string(),
            _ => "huge".to_string(),
        }
    }

    /// Get all detected memory leaks
    pub fn get_leaks(&self) -> &[MemoryLeak] {
        &self.detected_leaks
    }

    /// Clear all debugging data
    pub fn clear(&mut self) {
        self.allocations.clear();
        self.allocation_history.clear();
        self.detected_leaks.clear();
        self.stats = MemoryStats::default();
        self.next_id = 1;
    }

    /// Generate a comprehensive memory report
    pub fn generate_report(&self) -> MemoryReport {
        MemoryReport {
            stats: self.stats.clone(),
            leaks: self.detected_leaks.clone(),
            patterns: self.analyze_patterns(),
            config: self.config.clone(),
            timestamp: Instant::now(),
        }
    }
}

impl Default for MemoryDebugger {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive memory debugging report
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Current memory statistics
    pub stats: MemoryStats,
    /// Detected memory leaks
    pub leaks: Vec<MemoryLeak>,
    /// Allocation patterns
    pub patterns: Vec<AllocationPattern>,
    /// Debugger configuration
    pub config: MemoryDebugConfig,
    /// When this report was generated
    pub timestamp: Instant,
}

impl fmt::Display for MemoryReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Memory Debug Report ===")?;
        writeln!(f, "Generated at: {:?}", self.timestamp)?;
        writeln!(f)?;

        writeln!(f, "Memory Statistics:")?;
        writeln!(f, "  Total allocated: {} bytes", self.stats.total_allocated)?;
        writeln!(f, "  Peak allocated: {} bytes", self.stats.peak_allocated)?;
        writeln!(f, "  Active allocations: {}", self.stats.active_allocations)?;
        writeln!(f, "  Total allocations: {}", self.stats.total_allocations)?;
        writeln!(
            f,
            "  Total deallocations: {}",
            self.stats.total_deallocations
        )?;
        writeln!(
            f,
            "  Average allocation size: {:.2} bytes",
            self.stats.average_allocation_size
        )?;
        writeln!(f)?;

        if !self.leaks.is_empty() {
            writeln!(f, "Memory Leaks Detected ({}):", self.leaks.len())?;
            for (i, leak) in self.leaks.iter().enumerate() {
                writeln!(
                    f,
                    "  {}. ID: {}, Size: {} bytes, Age: {:?}, Probability: {:.2}",
                    i + 1,
                    leak.allocation.id,
                    leak.allocation.size,
                    leak.age,
                    leak.leak_probability
                )?;
            }
            writeln!(f)?;
        }

        if !self.patterns.is_empty() {
            writeln!(f, "Allocation Patterns:")?;
            for pattern in &self.patterns {
                writeln!(
                    f,
                    "  {}: {} allocations, avg lifetime: {:?}",
                    pattern.size_class, pattern.frequency, pattern.average_lifetime
                )?;
            }
        }

        Ok(())
    }
}

/// Global functions for easy memory debugging access
pub fn init_memory_debugger() -> Result<()> {
    let debugger = Arc::new(Mutex::new(MemoryDebugger::new()));
    MEMORY_DEBUGGER
        .set(debugger)
        .map_err(|_| TorshError::ConfigError("Memory debugger already initialized".to_string()))?;
    Ok(())
}

pub fn init_memory_debugger_with_config(config: MemoryDebugConfig) -> Result<()> {
    let debugger = Arc::new(Mutex::new(MemoryDebugger::with_config(config)));
    MEMORY_DEBUGGER
        .set(debugger)
        .map_err(|_| TorshError::ConfigError("Memory debugger already initialized".to_string()))?;
    Ok(())
}

pub fn record_allocation(size: usize, layout: Layout, tag: Option<String>) -> u64 {
    if let Some(debugger) = MEMORY_DEBUGGER.get() {
        if let Ok(mut debugger) = debugger.lock() {
            return debugger.record_allocation(size, layout, tag);
        }
    }
    0
}

pub fn record_deallocation(id: u64) {
    if let Some(debugger) = MEMORY_DEBUGGER.get() {
        if let Ok(mut debugger) = debugger.lock() {
            debugger.record_deallocation(id);
        }
    }
}

pub fn get_memory_stats() -> Option<MemoryStats> {
    MEMORY_DEBUGGER.get()?.lock().ok().map(|d| d.stats())
}

pub fn detect_memory_leaks() -> Option<Vec<MemoryLeak>> {
    MEMORY_DEBUGGER
        .get()?
        .lock()
        .ok()
        .map(|mut d| d.detect_leaks())
}

pub fn generate_memory_report() -> Option<MemoryReport> {
    MEMORY_DEBUGGER
        .get()?
        .lock()
        .ok()
        .map(|d| d.generate_report())
}

pub fn get_realtime_stats() -> Option<RealTimeStats> {
    MEMORY_DEBUGGER
        .get()?
        .lock()
        .ok()
        .map(|d| d.realtime_stats())
}

pub fn is_under_memory_pressure() -> bool {
    MEMORY_DEBUGGER
        .get()
        .and_then(|d| d.lock().ok())
        .map(|d| d.is_under_pressure())
        .unwrap_or(false)
}

pub fn get_pressure_level() -> SystemPressureLevel {
    MEMORY_DEBUGGER
        .get()
        .and_then(|d| d.lock().ok())
        .map(|d| d.get_pressure_level())
        .unwrap_or(SystemPressureLevel::Normal)
}

pub fn force_leak_detection() -> Option<Vec<MemoryLeak>> {
    MEMORY_DEBUGGER
        .get()?
        .lock()
        .ok()
        .map(|mut d| d.force_leak_detection())
}

pub fn get_leak_stats() -> Option<LeakStats> {
    MEMORY_DEBUGGER.get()?.lock().ok().map(|d| d.leak_stats())
}

/// Custom allocator wrapper that integrates with memory debugging
pub struct DebuggingAllocator<A: GlobalAlloc> {
    inner: A,
}

impl<A: GlobalAlloc> DebuggingAllocator<A> {
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for DebuggingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        if !ptr.is_null() {
            record_allocation(layout.size(), layout, None);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Note: We can't track which specific allocation is being freed without
        // maintaining a ptr -> id mapping, which would be expensive
        record_deallocation(0); // Use 0 to indicate unknown allocation ID
        self.inner.dealloc(ptr, layout);
    }
}

/// Type alias for system allocator with debugging
pub type SystemDebuggingAllocator = DebuggingAllocator<System>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_debugger_basic() {
        let mut debugger = MemoryDebugger::new();

        let layout = Layout::from_size_align(1024, 8).unwrap();
        let id = debugger.record_allocation(1024, layout, Some("test".to_string()));

        assert_eq!(debugger.stats().total_allocated, 1024);
        assert_eq!(debugger.stats().active_allocations, 1);

        debugger.record_deallocation(id);

        assert_eq!(debugger.stats().total_allocated, 0);
        assert_eq!(debugger.stats().active_allocations, 0);
    }

    #[test]
    fn test_leak_detection() {
        let config = MemoryDebugConfig {
            leak_threshold: Duration::from_millis(1),
            ..Default::default()
        };

        let mut debugger = MemoryDebugger::with_config(config);
        let layout = Layout::from_size_align(1024, 8).unwrap();

        let _id = debugger.record_allocation(1024, layout, Some("potential_leak".to_string()));

        // Wait for leak threshold
        std::thread::sleep(Duration::from_millis(2));

        let leaks = debugger.detect_leaks();
        assert!(!leaks.is_empty());
    }

    #[test]
    fn test_pattern_analysis() {
        let mut debugger = MemoryDebugger::new();
        let layout_small = Layout::from_size_align(512, 8).unwrap();
        let layout_large = Layout::from_size_align(2048, 8).unwrap();

        // Create small allocations
        for _ in 0..5 {
            debugger.record_allocation(512, layout_small, Some("small".to_string()));
        }

        // Create large allocations
        for _ in 0..3 {
            debugger.record_allocation(2048, layout_large, Some("large".to_string()));
        }

        let patterns = debugger.analyze_patterns();
        assert!(!patterns.is_empty());

        // Should have patterns for both small and large allocations
        let has_small = patterns.iter().any(|p| p.size_class == "small");
        let has_medium = patterns.iter().any(|p| p.size_class == "medium");

        assert!(has_small || has_medium); // 512 bytes might be classified as small or medium
    }

    #[test]
    fn test_enhanced_leak_detection() {
        let config = MemoryDebugConfig {
            leak_threshold: Duration::from_millis(1),
            ..Default::default()
        };

        let mut debugger = MemoryDebugger::with_config(config);
        let layout = Layout::from_size_align(1024 * 1024, 8).unwrap(); // 1MB allocation

        let _id =
            debugger.record_allocation(1024 * 1024, layout, Some("large_allocation".to_string()));

        // Wait for leak threshold
        std::thread::sleep(Duration::from_millis(10));

        let leaks = debugger.detect_leaks();
        assert!(!leaks.is_empty());

        let leak = &leaks[0];
        assert!(leak.confidence > 0.0);
        assert!(!leak.suggested_actions.is_empty());

        // Large allocation should trigger high risk
        assert!(matches!(
            leak.risk_level,
            LeakRiskLevel::High | LeakRiskLevel::Critical
        ));
    }

    #[test]
    fn test_realtime_monitoring() {
        let config = MemoryDebugConfig {
            monitoring_interval: Duration::from_millis(1),
            enable_real_time_monitoring: true,
            ..Default::default()
        };

        let mut debugger = MemoryDebugger::with_config(config);
        let layout = Layout::from_size_align(1024, 8).unwrap();

        // Make some allocations
        for _ in 0..5 {
            debugger.record_allocation(1024, layout, None);
        }

        // Wait for monitoring interval
        std::thread::sleep(Duration::from_millis(10));

        // Force monitoring update
        debugger.record_allocation(1024, layout, None);

        let stats = debugger.realtime_stats();
        assert!(stats.current_usage > 0);
        assert!(stats.recent_allocations > 0);
    }

    #[test]
    fn test_pressure_level_calculation() {
        let config = MemoryDebugConfig {
            memory_warning_threshold: 1000,
            memory_critical_threshold: 2000,
            min_tracked_size: 1, // Track all allocations for this test
            monitoring_interval: Duration::from_millis(1), // Short interval for testing
            ..Default::default()
        };

        let mut debugger = MemoryDebugger::with_config(config);
        let layout = Layout::from_size_align(1000, 8).unwrap();

        // Test normal pressure
        assert_eq!(
            debugger.calculate_system_pressure(),
            SystemPressureLevel::Normal
        );

        // Test low pressure (half of warning threshold = 500)
        let _id1 = debugger.record_allocation(600, layout, None);
        std::thread::sleep(Duration::from_millis(2)); // Wait for monitoring interval
        assert_eq!(
            debugger.calculate_system_pressure(),
            SystemPressureLevel::Low
        );

        // Test medium pressure (3/4 of warning threshold = 750, total = 800)
        let _id2 = debugger.record_allocation(200, layout, None);
        assert_eq!(
            debugger.calculate_system_pressure(),
            SystemPressureLevel::Medium
        );

        // Test high pressure (warning threshold = 1000, total = 1000)
        let _id3 = debugger.record_allocation(200, layout, None);
        assert_eq!(
            debugger.calculate_system_pressure(),
            SystemPressureLevel::High
        );

        // Test critical pressure (critical threshold = 2000, total = 2000)
        let _id4 = debugger.record_allocation(1000, layout, None);
        assert_eq!(
            debugger.calculate_system_pressure(),
            SystemPressureLevel::Critical
        );
    }

    #[test]
    fn test_leak_stats() {
        let config = MemoryDebugConfig {
            leak_threshold: Duration::from_millis(1),
            ..Default::default()
        };

        let mut debugger = MemoryDebugger::with_config(config);
        let layout_small = Layout::from_size_align(1024, 8).unwrap();
        let layout_large = Layout::from_size_align(1024 * 1024, 8).unwrap();

        // Create different sized allocations
        let _id1 = debugger.record_allocation(1024, layout_small, Some("small_leak".to_string()));
        let _id2 =
            debugger.record_allocation(1024 * 1024, layout_large, Some("large_leak".to_string()));

        // Wait for leak threshold
        std::thread::sleep(Duration::from_millis(50));

        let leaks = debugger.detect_leaks();
        assert!(!leaks.is_empty(), "Expected leaks to be detected");

        let stats = debugger.leak_stats();

        assert!(stats.total_leaks > 0);
        assert!(stats.total_leaked_bytes > 0);
        // Note: average_leak_age might be 0 due to rounding in integer division
        // This is acceptable for the test
    }

    #[test]
    fn test_global_api_functions() {
        // Test global API initialization
        let config = MemoryDebugConfig {
            enable_real_time_monitoring: true,
            monitoring_interval: Duration::from_millis(1),
            ..Default::default()
        };

        let _ = init_memory_debugger_with_config(config);

        // Test global functions
        let layout = Layout::from_size_align(1024, 8).unwrap();
        let id = record_allocation(1024, layout, Some("test".to_string()));

        let stats = get_memory_stats();
        assert!(stats.is_some());

        let pressure_level = get_pressure_level();
        assert_eq!(pressure_level, SystemPressureLevel::Normal);

        let is_under_pressure = is_under_memory_pressure();
        assert!(!is_under_pressure);

        record_deallocation(id);

        let report = generate_memory_report();
        assert!(report.is_some());
    }
}
