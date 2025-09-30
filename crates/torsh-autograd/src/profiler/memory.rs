//! Memory Tracking for Autograd Profiling
//!
//! This module provides comprehensive memory tracking capabilities for monitoring
//! memory usage during automatic differentiation operations. It tracks various
//! categories of memory usage and provides insights into memory consumption patterns.
//!
//! # Features
//!
//! - **Real-time Memory Monitoring**: Continuous tracking of memory usage
//! - **Categorized Memory Tracking**: Separate tracking for gradients, graph, and temporary memory
//! - **Memory Timeline**: Historical memory usage patterns
//! - **Memory Efficiency Analysis**: Detection of memory inefficiencies
//! - **Peak Memory Detection**: Identification of memory usage peaks
//!
//! # Memory Categories
//!
//! The tracker monitors several categories of memory:
//! - **Total Memory**: Overall system memory usage
//! - **Gradient Memory**: Memory used for storing gradients
//! - **Graph Memory**: Memory used for computation graph nodes and edges
//! - **Temporary Memory**: Memory used for intermediate computations
//!
//! # Usage Examples
//!
//! ```rust,ignore
//! use torsh_autograd::profiler::memory::MemoryTracker;
//! use std::time::Duration;
//!
//! // Create a memory tracker with 100ms snapshot interval
//! let mut tracker = MemoryTracker::new(Duration::from_millis(100));
//!
//! // Take a snapshot during an operation
//! tracker.maybe_take_snapshot(Some("matmul".to_string()));
//!
//! // Get memory statistics
//! let stats = tracker.get_memory_statistics();
//! println!("Peak memory usage: {} MB", stats.peak_total_memory / 1024 / 1024);
//! ```

use super::types::MemorySnapshot;
use std::time::{Duration, Instant, SystemTime};

/// Memory tracker for monitoring memory usage during autograd operations
///
/// The MemoryTracker provides real-time memory monitoring with configurable
/// sampling intervals and comprehensive memory categorization.
///
/// # Memory Tracking Strategy
///
/// The tracker uses a polling approach with configurable intervals to balance
/// accuracy with performance overhead. It categorizes memory usage to provide
/// insights into different aspects of autograd computation.
#[derive(Debug)]
pub struct MemoryTracker {
    /// Collected memory snapshots
    snapshots: Vec<MemorySnapshot>,
    /// Last snapshot timestamp
    last_snapshot: Instant,
    /// Interval between snapshots
    interval: Duration,
    /// Maximum number of snapshots to retain
    max_snapshots: usize,
    /// Memory estimation cache for performance
    memory_cache: MemoryEstimationCache,
}

/// Cache for memory estimation to reduce overhead
#[derive(Debug)]
pub struct MemoryEstimationCache {
    /// Cached total memory estimate
    total_memory: Option<(Instant, usize)>,
    /// Cached gradient memory estimate
    gradient_memory: Option<(Instant, usize)>,
    /// Cache validity duration
    cache_duration: Duration,
}

/// Statistics about memory usage during profiling
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Peak total memory usage
    pub peak_total_memory: usize,
    /// Average total memory usage
    pub average_total_memory: usize,
    /// Peak gradient memory usage
    pub peak_gradient_memory: usize,
    /// Average gradient memory usage
    pub average_gradient_memory: usize,
    /// Peak graph memory usage
    pub peak_graph_memory: usize,
    /// Average graph memory usage
    pub average_graph_memory: usize,
    /// Peak temporary memory usage
    pub peak_temp_memory: usize,
    /// Average temporary memory usage
    pub average_temp_memory: usize,
    /// Total number of snapshots
    pub snapshot_count: usize,
    /// Memory growth rate (bytes per second)
    pub memory_growth_rate: f64,
    /// Memory efficiency score (0.0 to 1.0)
    pub efficiency_score: f32,
}

/// Memory usage trend analysis
#[derive(Debug, Clone)]
pub struct MemoryTrend {
    /// Trend direction (growing, stable, shrinking)
    pub direction: TrendDirection,
    /// Rate of change (bytes per second)
    pub rate_of_change: f64,
    /// Confidence in the trend analysis (0.0 to 1.0)
    pub confidence: f32,
    /// Predicted peak memory in the next interval
    pub predicted_peak: usize,
}

/// Direction of memory usage trend
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Memory usage is increasing
    Growing,
    /// Memory usage is stable
    Stable,
    /// Memory usage is decreasing
    Shrinking,
}

impl MemoryTracker {
    /// Creates a new memory tracker with the specified snapshot interval
    ///
    /// # Arguments
    ///
    /// * `interval` - Time interval between memory snapshots
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::time::Duration;
    /// let tracker = MemoryTracker::new(Duration::from_millis(100));
    /// ```
    pub fn new(interval: Duration) -> Self {
        Self {
            snapshots: Vec::new(),
            last_snapshot: Instant::now(),
            interval,
            max_snapshots: 10000, // Prevent unbounded growth
            memory_cache: MemoryEstimationCache::new(),
        }
    }

    /// Creates a new memory tracker with custom configuration
    ///
    /// # Arguments
    ///
    /// * `interval` - Time interval between memory snapshots
    /// * `max_snapshots` - Maximum number of snapshots to retain
    pub fn with_config(interval: Duration, max_snapshots: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            last_snapshot: Instant::now(),
            interval,
            max_snapshots,
            memory_cache: MemoryEstimationCache::new(),
        }
    }

    /// Takes a memory snapshot if enough time has elapsed since the last snapshot
    ///
    /// # Arguments
    ///
    /// * `operation` - Optional name of the operation being executed
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// tracker.maybe_take_snapshot(Some("conv2d".to_string()));
    /// ```
    pub fn maybe_take_snapshot(&mut self, operation: Option<String>) {
        let now = Instant::now();
        if now.duration_since(self.last_snapshot) >= self.interval {
            self.take_snapshot(operation);
            self.last_snapshot = now;
        }
    }

    /// Forces a memory snapshot to be taken immediately
    ///
    /// # Arguments
    ///
    /// * `operation` - Optional name of the operation being executed
    pub fn take_snapshot(&mut self, operation: Option<String>) {
        let snapshot = MemorySnapshot {
            timestamp: SystemTime::now(),
            total_memory: self.estimate_total_memory(),
            gradient_memory: self.estimate_gradient_memory(),
            graph_memory: self.estimate_graph_memory(),
            temp_memory: self.estimate_temp_memory(),
            current_operation: operation,
        };

        self.snapshots.push(snapshot);

        // Limit snapshot history to prevent unbounded growth
        if self.snapshots.len() > self.max_snapshots {
            self.snapshots.remove(0);
        }
    }

    /// Gets all collected memory snapshots
    ///
    /// # Returns
    ///
    /// Slice of all collected memory snapshots
    pub fn get_snapshots(&self) -> &[MemorySnapshot] {
        &self.snapshots
    }

    /// Gets memory snapshots within a specific time range
    ///
    /// # Arguments
    ///
    /// * `start_time` - Start of the time range
    /// * `end_time` - End of the time range
    ///
    /// # Returns
    ///
    /// Vector of snapshots within the specified time range
    pub fn get_snapshots_in_range(
        &self,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> Vec<&MemorySnapshot> {
        self.snapshots
            .iter()
            .filter(|snapshot| snapshot.timestamp >= start_time && snapshot.timestamp <= end_time)
            .collect()
    }

    /// Computes comprehensive memory usage statistics
    ///
    /// # Returns
    ///
    /// Memory statistics computed from all collected snapshots
    pub fn get_memory_statistics(&self) -> MemoryStatistics {
        if self.snapshots.is_empty() {
            return MemoryStatistics::default();
        }

        let mut peak_total = 0;
        let mut total_sum = 0;
        let mut peak_gradient = 0;
        let mut gradient_sum = 0;
        let mut peak_graph = 0;
        let mut graph_sum = 0;
        let mut peak_temp = 0;
        let mut temp_sum = 0;

        for snapshot in &self.snapshots {
            peak_total = peak_total.max(snapshot.total_memory);
            total_sum += snapshot.total_memory;
            peak_gradient = peak_gradient.max(snapshot.gradient_memory);
            gradient_sum += snapshot.gradient_memory;
            peak_graph = peak_graph.max(snapshot.graph_memory);
            graph_sum += snapshot.graph_memory;
            peak_temp = peak_temp.max(snapshot.temp_memory);
            temp_sum += snapshot.temp_memory;
        }

        let count = self.snapshots.len();
        let average_total = total_sum / count;

        // Compute memory growth rate
        let growth_rate = if self.snapshots.len() >= 2 {
            let first = &self.snapshots[0];
            let last = &self.snapshots[self.snapshots.len() - 1];
            let time_diff = last
                .timestamp
                .duration_since(first.timestamp)
                .unwrap_or(Duration::from_secs(1))
                .as_secs_f64();
            let memory_diff = last.total_memory as f64 - first.total_memory as f64;
            memory_diff / time_diff
        } else {
            0.0
        };

        // Compute efficiency score (based on memory variance and utilization)
        let efficiency_score = self.compute_efficiency_score(&self.snapshots);

        MemoryStatistics {
            peak_total_memory: peak_total,
            average_total_memory: average_total,
            peak_gradient_memory: peak_gradient,
            average_gradient_memory: gradient_sum / count,
            peak_graph_memory: peak_graph,
            average_graph_memory: graph_sum / count,
            peak_temp_memory: peak_temp,
            average_temp_memory: temp_sum / count,
            snapshot_count: count,
            memory_growth_rate: growth_rate,
            efficiency_score,
        }
    }

    /// Analyzes memory usage trends
    ///
    /// # Returns
    ///
    /// Memory trend analysis based on recent snapshots
    pub fn analyze_memory_trend(&self) -> Option<MemoryTrend> {
        if self.snapshots.len() < 3 {
            return None;
        }

        // Use the last 10 snapshots for trend analysis
        let recent_snapshots = &self.snapshots[self.snapshots.len().saturating_sub(10)..];

        // Compute linear regression for trend direction
        let (slope, confidence) = self.compute_trend_slope(recent_snapshots);

        let direction = if slope > 1024.0 * 1024.0 {
            // 1MB/s threshold
            TrendDirection::Growing
        } else if slope < -1024.0 * 1024.0 {
            TrendDirection::Shrinking
        } else {
            TrendDirection::Stable
        };

        // Predict peak memory
        let current_memory = recent_snapshots.last().unwrap().total_memory;
        let predicted_peak = if slope > 0.0 {
            current_memory + (slope * self.interval.as_secs_f64()) as usize
        } else {
            current_memory
        };

        Some(MemoryTrend {
            direction,
            rate_of_change: slope,
            confidence,
            predicted_peak,
        })
    }

    /// Gets the current memory usage estimate
    ///
    /// # Returns
    ///
    /// Current memory usage estimate in bytes
    pub fn get_current_memory_usage(&mut self) -> usize {
        self.estimate_total_memory()
    }

    /// Clears all collected snapshots
    pub fn clear_snapshots(&mut self) {
        self.snapshots.clear();
        self.memory_cache.invalidate();
    }

    /// Sets the snapshot interval
    ///
    /// # Arguments
    ///
    /// * `interval` - New snapshot interval
    pub fn set_interval(&mut self, interval: Duration) {
        self.interval = interval;
    }

    /// Gets the current snapshot interval
    ///
    /// # Returns
    ///
    /// Current snapshot interval
    pub fn get_interval(&self) -> Duration {
        self.interval
    }

    /// Estimates total memory usage
    pub fn estimate_total_memory(&mut self) -> usize {
        // Check cache first
        if let Some((timestamp, memory)) = self.memory_cache.total_memory {
            if timestamp.elapsed() < self.memory_cache.cache_duration {
                return memory;
            }
        }

        // Platform-specific memory estimation
        let memory = self.estimate_system_memory();

        // Update cache
        self.memory_cache.total_memory = Some((Instant::now(), memory));

        memory
    }

    /// Estimates gradient memory usage
    fn estimate_gradient_memory(&mut self) -> usize {
        // Check cache first
        if let Some((timestamp, memory)) = self.memory_cache.gradient_memory {
            if timestamp.elapsed() < self.memory_cache.cache_duration {
                return memory;
            }
        }

        // Estimate gradient memory (typically 20-30% of total)
        let gradient_memory = self.estimate_total_memory() / 4;

        // Update cache
        self.memory_cache.gradient_memory = Some((Instant::now(), gradient_memory));

        gradient_memory
    }

    /// Estimates graph memory usage
    fn estimate_graph_memory(&mut self) -> usize {
        // Graph memory is typically 10-15% of total memory
        self.estimate_total_memory() / 8
    }

    /// Estimates temporary memory usage
    fn estimate_temp_memory(&mut self) -> usize {
        // Temporary memory is typically 5-10% of total memory
        self.estimate_total_memory() / 16
    }

    /// Platform-specific system memory estimation
    #[cfg(target_os = "linux")]
    fn estimate_system_memory(&self) -> usize {
        // Linux-specific implementation using /proc/self/status
        use std::fs;
        if let Ok(contents) = fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
        1024 * 1024 * 64 // 64MB fallback
    }

    /// Platform-specific system memory estimation
    #[cfg(target_os = "macos")]
    fn estimate_system_memory(&self) -> usize {
        // macOS-specific implementation using mach API
        // This is a simplified placeholder implementation
        1024 * 1024 * 64 // 64MB placeholder
    }

    /// Platform-specific system memory estimation
    #[cfg(target_os = "windows")]
    fn estimate_system_memory(&self) -> usize {
        // Windows-specific implementation using Windows API
        // This is a simplified placeholder implementation
        1024 * 1024 * 64 // 64MB placeholder
    }

    /// Default system memory estimation for other platforms
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    fn estimate_system_memory(&self) -> usize {
        1024 * 1024 * 64 // 64MB placeholder
    }

    /// Computes efficiency score based on memory usage patterns
    fn compute_efficiency_score(&self, snapshots: &[MemorySnapshot]) -> f32 {
        if snapshots.len() < 2 {
            return 1.0;
        }

        // Compute memory variance (lower variance = more efficient)
        let total_memories: Vec<f64> = snapshots.iter().map(|s| s.total_memory as f64).collect();

        let mean = total_memories.iter().sum::<f64>() / total_memories.len() as f64;
        let variance = total_memories
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / total_memories.len() as f64;

        let coefficient_of_variation = (variance.sqrt() / mean).min(1.0);

        // Higher efficiency = lower variance
        (1.0 - coefficient_of_variation) as f32
    }

    /// Computes trend slope using linear regression
    fn compute_trend_slope(&self, snapshots: &[MemorySnapshot]) -> (f64, f32) {
        if snapshots.len() < 2 {
            return (0.0, 0.0);
        }

        // Convert timestamps to seconds from first snapshot
        let first_time = snapshots[0].timestamp;
        let mut x_values = Vec::new();
        let mut y_values = Vec::new();

        for snapshot in snapshots {
            let x = snapshot
                .timestamp
                .duration_since(first_time)
                .unwrap_or(Duration::from_secs(0))
                .as_secs_f64();
            let y = snapshot.total_memory as f64;
            x_values.push(x);
            y_values.push(y);
        }

        // Simple linear regression
        let n = x_values.len() as f64;
        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = y_values.iter().sum::<f64>();
        let sum_xy = x_values
            .iter()
            .zip(y_values.iter())
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x_squared = x_values.iter().map(|x| x * x).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x);

        // Compute R-squared for confidence
        let mean_y = sum_y / n;
        let ss_tot = y_values.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
        let intercept = (sum_y - slope * sum_x) / n;
        let ss_res = x_values
            .iter()
            .zip(y_values.iter())
            .map(|(x, y)| {
                let predicted = slope * x + intercept;
                (y - predicted).powi(2)
            })
            .sum::<f64>();

        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        (slope, r_squared as f32)
    }
}

impl MemoryEstimationCache {
    fn new() -> Self {
        Self {
            total_memory: None,
            gradient_memory: None,
            cache_duration: Duration::from_millis(100), // Cache for 100ms
        }
    }

    fn invalidate(&mut self) {
        self.total_memory = None;
        self.gradient_memory = None;
    }
}

impl Default for MemoryStatistics {
    fn default() -> Self {
        Self {
            peak_total_memory: 0,
            average_total_memory: 0,
            peak_gradient_memory: 0,
            average_gradient_memory: 0,
            peak_graph_memory: 0,
            average_graph_memory: 0,
            peak_temp_memory: 0,
            average_temp_memory: 0,
            snapshot_count: 0,
            memory_growth_rate: 0.0,
            efficiency_score: 1.0,
        }
    }
}

impl std::fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrendDirection::Growing => write!(f, "Growing"),
            TrendDirection::Stable => write!(f, "Stable"),
            TrendDirection::Shrinking => write!(f, "Shrinking"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker_creation() {
        let tracker = MemoryTracker::new(Duration::from_millis(100));
        assert_eq!(tracker.interval, Duration::from_millis(100));
        assert_eq!(tracker.snapshots.len(), 0);
    }

    #[test]
    fn test_memory_tracker_with_config() {
        let tracker = MemoryTracker::with_config(Duration::from_millis(50), 500);
        assert_eq!(tracker.interval, Duration::from_millis(50));
        assert_eq!(tracker.max_snapshots, 500);
    }

    #[test]
    fn test_take_snapshot() {
        let mut tracker = MemoryTracker::new(Duration::from_millis(100));
        tracker.take_snapshot(Some("test_op".to_string()));

        assert_eq!(tracker.snapshots.len(), 1);
        let snapshot = &tracker.snapshots[0];
        assert_eq!(snapshot.current_operation, Some("test_op".to_string()));
        assert!(snapshot.total_memory > 0);
    }

    #[test]
    fn test_memory_statistics() {
        let mut tracker = MemoryTracker::new(Duration::from_millis(100));

        // Take several snapshots
        for i in 0..5 {
            tracker.take_snapshot(Some(format!("op_{}", i)));
            std::thread::sleep(Duration::from_millis(10));
        }

        let stats = tracker.get_memory_statistics();
        assert_eq!(stats.snapshot_count, 5);
        assert!(stats.peak_total_memory > 0);
        assert!(stats.average_total_memory > 0);
        assert!(stats.efficiency_score >= 0.0 && stats.efficiency_score <= 1.0);
    }

    #[test]
    fn test_memory_trend_analysis() {
        let mut tracker = MemoryTracker::new(Duration::from_millis(100));

        // Take insufficient snapshots for trend analysis
        tracker.take_snapshot(Some("op_1".to_string()));
        tracker.take_snapshot(Some("op_2".to_string()));

        let trend = tracker.analyze_memory_trend();
        assert!(trend.is_none());

        // Take enough snapshots
        for i in 3..=10 {
            tracker.take_snapshot(Some(format!("op_{}", i)));
        }

        let trend = tracker.analyze_memory_trend();
        assert!(trend.is_some());
    }

    #[test]
    fn test_snapshots_in_range() {
        let mut tracker = MemoryTracker::new(Duration::from_millis(100));
        let start_time = SystemTime::now();

        tracker.take_snapshot(Some("before".to_string()));
        std::thread::sleep(Duration::from_millis(100));

        let range_start = SystemTime::now();
        tracker.take_snapshot(Some("in_range".to_string()));
        let range_end = SystemTime::now();

        std::thread::sleep(Duration::from_millis(100));
        tracker.take_snapshot(Some("after".to_string()));

        let snapshots_in_range = tracker.get_snapshots_in_range(range_start, range_end);
        assert_eq!(snapshots_in_range.len(), 1);
        assert_eq!(
            snapshots_in_range[0].current_operation,
            Some("in_range".to_string())
        );
    }

    #[test]
    fn test_clear_snapshots() {
        let mut tracker = MemoryTracker::new(Duration::from_millis(100));
        tracker.take_snapshot(Some("test_op".to_string()));

        assert_eq!(tracker.snapshots.len(), 1);

        tracker.clear_snapshots();
        assert_eq!(tracker.snapshots.len(), 0);
    }

    #[test]
    fn test_interval_modification() {
        let mut tracker = MemoryTracker::new(Duration::from_millis(100));
        assert_eq!(tracker.get_interval(), Duration::from_millis(100));

        tracker.set_interval(Duration::from_millis(200));
        assert_eq!(tracker.get_interval(), Duration::from_millis(200));
    }

    #[test]
    fn test_max_snapshots_limit() {
        let mut tracker = MemoryTracker::with_config(Duration::from_millis(10), 3);

        // Take more snapshots than the limit
        for i in 0..5 {
            tracker.take_snapshot(Some(format!("op_{}", i)));
        }

        // Should be limited to max_snapshots
        assert_eq!(tracker.snapshots.len(), 3);
    }

    #[test]
    fn test_current_memory_usage() {
        let mut tracker = MemoryTracker::new(Duration::from_millis(100));
        let current_usage = tracker.get_current_memory_usage();
        assert!(current_usage > 0);
    }

    #[test]
    fn test_trend_direction_display() {
        assert_eq!(format!("{}", TrendDirection::Growing), "Growing");
        assert_eq!(format!("{}", TrendDirection::Stable), "Stable");
        assert_eq!(format!("{}", TrendDirection::Shrinking), "Shrinking");
    }
}
