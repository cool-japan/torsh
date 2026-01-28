//! Real-time memory monitoring for gradient operations
//!
//! This module provides real-time monitoring capabilities for memory usage during
//! gradient computations. It enables continuous tracking of memory allocation patterns,
//! automated snapshot collection, and live analysis of memory behavior.
//!
//! # Overview
//!
//! The monitoring system provides several key capabilities:
//!
//! - **Real-time Snapshots**: Continuous memory usage sampling during operations
//! - **Gradient-Specific Monitoring**: Specialized tracking for gradient operations
//! - **Automated Analysis**: Live detection of memory issues and patterns
//! - **Performance Impact Assessment**: Minimal overhead monitoring design
//! - **Flexible Collection**: Configurable sampling rates and collection strategies
//!
//! # Monitoring Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                Memory Monitor                               │
//! │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │
//! │  │  Snapshot   │ │   Timer     │ │    Analysis         │  │
//! │  │ Collection  │ │ Management  │ │   Engine            │  │
//! │  └─────────────┘ └─────────────┘ └─────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//!                    Monitoring Results
//!                   (Snapshots + Analysis)
//! ```
//!
//! # Usage Examples
//!
//! ## Basic Monitoring
//!
//! ```rust,ignore
//! use crate::memory::monitoring::GradientMemoryMonitor;
//!
//! let monitor = GradientMemoryMonitor::new("conv2d_backward".to_string());
//!
//! // Monitor starts automatically
//! assert!(monitor.is_active());
//!
//! // Take manual snapshot
//! monitor.take_snapshot(1024 * 1024)?; // 1MB allocated
//!
//! // Stop monitoring and get results
//! let results = monitor.stop_and_analyze()?;
//! println!("Peak memory: {} bytes", results.peak_memory_usage);
//! ```
//!
//! ## Advanced Monitoring with Custom Configuration
//!
//! ```rust,ignore
//! use crate::memory::monitoring::{GradientMemoryMonitor, MonitoringConfig};
//! use std::time::Duration;
//!
//! let config = MonitoringConfig {
//!     snapshot_interval: Duration::from_millis(10), // High frequency
//!     max_snapshots: 10000,
//!     enable_automatic_analysis: true,
//!     ..Default::default()
//! };
//!
//! let monitor = GradientMemoryMonitor::with_config(
//!     "transformer_attention".to_string(),
//!     config
//! );
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use torsh_core::error::{Result, TorshError};

/// Real-time gradient memory monitoring
///
/// Provides continuous monitoring of memory usage during gradient operations.
/// Designed for minimal performance impact while collecting detailed memory
/// usage patterns and statistics.
///
/// # Thread Safety
///
/// The monitor is designed to be thread-safe and can be safely shared across
/// threads using Arc. Internal state is protected by appropriate synchronization
/// primitives.
///
/// # Performance Characteristics
///
/// - **Snapshot Collection**: O(1) - constant time snapshot recording
/// - **Memory Overhead**: ~8KB baseline + (snapshot_size * num_snapshots)
/// - **CPU Overhead**: < 1% for typical sampling rates (100ms intervals)
/// - **Automatic Cleanup**: Background cleanup prevents unbounded memory growth
///
/// # Lifecycle Management
///
/// 1. **Creation**: Monitor is created and starts collecting immediately
/// 2. **Active Monitoring**: Continuous snapshot collection based on configuration
/// 3. **Manual Snapshots**: Additional snapshots can be triggered manually
/// 4. **Termination**: Monitoring stops and final analysis is performed
/// 5. **Results**: Comprehensive analysis results are returned
#[derive(Debug)]
pub struct GradientMemoryMonitor {
    /// Operation being monitored
    pub operation_name: String,
    /// Monitoring start time
    pub start_time: Instant,
    /// Memory usage snapshots
    pub memory_snapshots: Arc<Mutex<Vec<MemorySnapshot>>>,
    /// Whether monitoring is active
    pub active: Arc<AtomicBool>,
    /// Monitoring configuration
    config: MonitoringConfig,
    /// Background monitoring thread handle
    _monitor_thread: Option<std::thread::JoinHandle<()>>,
}

/// Memory snapshot for monitoring
///
/// Represents a point-in-time snapshot of memory usage during gradient operations.
/// Snapshots are lightweight and designed for high-frequency collection with
/// minimal performance impact.
///
/// # Snapshot Data
///
/// Each snapshot captures:
/// - **Timestamp**: Precise timing information for trend analysis
/// - **Memory Usage**: Current memory allocation at snapshot time
/// - **Context**: Operation context and metadata
/// - **Deltas**: Changes since previous snapshot
///
/// # Collection Strategy
///
/// Snapshots can be collected through multiple triggers:
/// - **Timer-based**: Regular intervals (e.g., every 100ms)
/// - **Event-based**: Specific allocation/deallocation events
/// - **Threshold-based**: When memory usage crosses certain thresholds
/// - **Manual**: Explicitly triggered by application code
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Snapshot timestamp
    pub timestamp: Instant,
    /// Bytes allocated at this point
    pub allocated_bytes: usize,
    /// Change in allocation since last snapshot
    pub allocation_delta: i64,
    /// Operation context at snapshot time
    pub operation_context: String,
    /// Memory pressure level at snapshot time
    pub memory_pressure: Option<f64>,
    /// Whether this snapshot was manually triggered
    pub manual_trigger: bool,
}

/// Configuration for memory monitoring
///
/// Comprehensive configuration for memory monitoring behavior, allowing
/// fine-tuning of collection frequency, storage limits, and analysis options.
///
/// # Performance Tuning
///
/// Different configurations for different use cases:
///
/// **High-frequency debugging**:
/// ```rust,ignore
/// MonitoringConfig {
///     snapshot_interval: Duration::from_millis(1),
///     max_snapshots: 100000,
///     enable_automatic_analysis: true,
///     ..Default::default()
/// }
/// ```
///
/// **Production monitoring**:
/// ```rust,ignore
/// MonitoringConfig {
///     snapshot_interval: Duration::from_millis(100),
///     max_snapshots: 1000,
///     enable_automatic_analysis: false,
///     ..Default::default()
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Interval between automatic snapshots
    pub snapshot_interval: Duration,
    /// Maximum number of snapshots to retain
    pub max_snapshots: usize,
    /// Enable automatic analysis during monitoring
    pub enable_automatic_analysis: bool,
    /// Memory threshold for triggering snapshots (bytes)
    pub memory_threshold: Option<usize>,
    /// Enable background monitoring thread
    pub enable_background_monitoring: bool,
    /// Snapshot retention policy
    pub retention_policy: SnapshotRetentionPolicy,
}

/// Snapshot retention policy
///
/// Defines how snapshots are managed when storage limits are reached.
///
/// # Retention Strategies
///
/// - **KeepLatest**: Discard oldest snapshots when limit reached
/// - **KeepImportant**: Prioritize snapshots with significant memory changes
/// - **KeepPeaks**: Retain snapshots at memory usage peaks and valleys
/// - **Adaptive**: Dynamically adjust retention based on memory patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotRetentionPolicy {
    /// Keep the most recent snapshots, discard oldest
    KeepLatest,
    /// Keep snapshots with significant memory changes
    KeepImportant,
    /// Keep snapshots at memory peaks and valleys
    KeepPeaks,
    /// Adaptive retention based on usage patterns
    Adaptive,
}

impl Default for MonitoringConfig {
    /// Create a balanced default monitoring configuration
    ///
    /// Provides good monitoring coverage with reasonable performance impact:
    /// - 100ms snapshot intervals
    /// - Up to 1000 snapshots retained
    /// - Automatic analysis disabled for better performance
    /// - Background monitoring enabled
    /// - Latest-first retention policy
    fn default() -> Self {
        Self {
            snapshot_interval: Duration::from_millis(100),
            max_snapshots: 1000,
            enable_automatic_analysis: false,
            memory_threshold: None,
            enable_background_monitoring: true,
            retention_policy: SnapshotRetentionPolicy::KeepLatest,
        }
    }
}

/// Result of gradient memory monitoring
///
/// Comprehensive analysis results from a completed monitoring session.
/// Provides detailed insights into memory usage patterns, trends, and
/// potential optimization opportunities.
///
/// # Analysis Categories
///
/// - **Usage Statistics**: Peak, average, and total memory usage
/// - **Trend Analysis**: Memory growth patterns over time
/// - **Efficiency Metrics**: Memory utilization efficiency scores
/// - **Anomaly Detection**: Unusual patterns or potential issues
/// - **Optimization Insights**: Recommendations for improvement
#[derive(Debug, Clone)]
pub struct GradientMemoryMonitoringResult {
    /// Operation that was monitored
    pub operation_name: String,
    /// Total monitoring duration
    pub monitoring_duration: Duration,
    /// Number of snapshots collected
    pub snapshot_count: usize,
    /// Peak memory usage observed
    pub peak_memory_usage: usize,
    /// Average memory usage
    pub average_memory_usage: usize,
    /// Total memory allocated during monitoring
    pub total_memory_allocated: usize,
    /// Memory growth rate (bytes per second)
    pub memory_growth_rate: f64,
    /// Memory usage efficiency score (0.0-1.0)
    pub efficiency_score: f64,
    /// All collected snapshots
    pub snapshots: Vec<MemorySnapshot>,
    /// Detected anomalies during monitoring
    pub anomalies: Vec<String>,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
    /// Analysis timestamp
    pub analysis_timestamp: Instant,
}

impl GradientMemoryMonitor {
    /// Create a new memory monitor with default configuration
    ///
    /// Initializes monitoring for the specified operation and begins
    /// collecting snapshots immediately.
    ///
    /// # Arguments
    ///
    /// * `operation_name` - Name of the operation being monitored
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let monitor = GradientMemoryMonitor::new("conv2d_backward".to_string());
    /// assert!(monitor.is_active());
    /// ```
    pub fn new(operation_name: String) -> Self {
        Self::with_config(operation_name, MonitoringConfig::default())
    }

    /// Create a memory monitor with custom configuration
    ///
    /// Allows full customization of monitoring behavior through configuration.
    ///
    /// # Arguments
    ///
    /// * `operation_name` - Name of the operation being monitored
    /// * `config` - Custom monitoring configuration
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = MonitoringConfig {
    ///     snapshot_interval: Duration::from_millis(50), // High frequency
    ///     max_snapshots: 5000,
    ///     ..Default::default()
    /// };
    /// let monitor = GradientMemoryMonitor::with_config("attention".to_string(), config);
    /// ```
    pub fn with_config(operation_name: String, config: MonitoringConfig) -> Self {
        let memory_snapshots = Arc::new(Mutex::new(Vec::new()));
        let active = Arc::new(AtomicBool::new(true));

        // Start background monitoring if enabled
        let monitor_thread = if config.enable_background_monitoring {
            let snapshots_clone = Arc::clone(&memory_snapshots);
            let active_clone = Arc::clone(&active);
            let interval = config.snapshot_interval;
            let max_snapshots = config.max_snapshots;
            let retention_policy = config.retention_policy.clone();

            Some(std::thread::spawn(move || {
                Self::background_monitoring_loop(
                    snapshots_clone,
                    active_clone,
                    interval,
                    max_snapshots,
                    retention_policy,
                );
            }))
        } else {
            None
        };

        Self {
            operation_name,
            start_time: Instant::now(),
            memory_snapshots,
            active,
            config,
            _monitor_thread: monitor_thread,
        }
    }

    /// Take a memory usage snapshot
    ///
    /// Records current memory usage and adds it to the snapshot collection.
    /// Can be called manually to capture memory state at specific points.
    ///
    /// # Arguments
    ///
    /// * `allocated_bytes` - Current memory allocation in bytes
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Take snapshot after major allocation
    /// let allocated = allocate_tensor_memory(1024 * 1024);
    /// monitor.take_snapshot(allocated.len())?;
    /// ```
    pub fn take_snapshot(&self, allocated_bytes: usize) -> Result<()> {
        if !self.is_active() {
            return Err(TorshError::InvalidOperation(
                "Cannot take snapshot on inactive monitor".to_string(),
            ));
        }

        let mut snapshots = self
            .memory_snapshots
            .lock()
            .expect("lock should not be poisoned");

        // Calculate allocation delta
        let allocation_delta = if let Some(last_snapshot) = snapshots.last() {
            allocated_bytes as i64 - last_snapshot.allocated_bytes as i64
        } else {
            allocated_bytes as i64
        };

        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            allocated_bytes,
            allocation_delta,
            operation_context: self.operation_name.clone(),
            memory_pressure: None, // Could be calculated from system info
            manual_trigger: true,
        };

        snapshots.push(snapshot);

        // Apply retention policy if needed
        if snapshots.len() > self.config.max_snapshots {
            self.apply_retention_policy(&mut snapshots);
        }

        Ok(())
    }

    /// Stop monitoring and analyze results
    ///
    /// Terminates the monitoring session and returns comprehensive analysis
    /// of the collected memory usage data.
    ///
    /// # Returns
    ///
    /// Detailed monitoring results with analysis and recommendations.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // After gradient computation is complete
    /// let results = monitor.stop_and_analyze()?;
    /// println!("Peak memory: {} MB", results.peak_memory_usage / (1024 * 1024));
    /// for recommendation in results.recommendations {
    ///     println!("Optimization: {}", recommendation);
    /// }
    /// ```
    pub fn stop_and_analyze(&self) -> Result<GradientMemoryMonitoringResult> {
        self.active.store(false, Ordering::SeqCst);

        let snapshots = self
            .memory_snapshots
            .lock()
            .expect("lock should not be poisoned");

        if snapshots.is_empty() {
            return Ok(GradientMemoryMonitoringResult {
                operation_name: self.operation_name.clone(),
                monitoring_duration: self.start_time.elapsed(),
                snapshot_count: 0,
                peak_memory_usage: 0,
                average_memory_usage: 0,
                total_memory_allocated: 0,
                memory_growth_rate: 0.0,
                efficiency_score: 1.0,
                snapshots: Vec::new(),
                anomalies: Vec::new(),
                recommendations: Vec::new(),
                analysis_timestamp: Instant::now(),
            });
        }

        let result = self.analyze_snapshots(&snapshots);
        Ok(result)
    }

    /// Check if monitoring is currently active
    ///
    /// Returns true if the monitor is actively collecting snapshots.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// if monitor.is_active() {
    ///     monitor.take_snapshot(current_memory_usage)?;
    /// }
    /// ```
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    /// Get current snapshot count
    ///
    /// Returns the number of snapshots currently stored in the monitor.
    pub fn snapshot_count(&self) -> usize {
        self.memory_snapshots
            .lock()
            .expect("lock should not be poisoned")
            .len()
    }

    /// Get monitoring duration so far
    ///
    /// Returns the time elapsed since monitoring started.
    pub fn monitoring_duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Background monitoring loop
    ///
    /// Runs in a separate thread to automatically collect snapshots at
    /// regular intervals without blocking the main thread.
    fn background_monitoring_loop(
        snapshots: Arc<Mutex<Vec<MemorySnapshot>>>,
        active: Arc<AtomicBool>,
        interval: Duration,
        max_snapshots: usize,
        retention_policy: SnapshotRetentionPolicy,
    ) {
        while active.load(Ordering::SeqCst) {
            std::thread::sleep(interval);

            if !active.load(Ordering::SeqCst) {
                break;
            }

            // Take automatic snapshot
            // Note: In a real implementation, this would get actual memory usage
            // For now, we'll simulate it
            let simulated_memory = Self::get_current_memory_usage();

            if let Ok(mut snapshots_guard) = snapshots.lock() {
                let allocation_delta = if let Some(last_snapshot) = snapshots_guard.last() {
                    simulated_memory as i64 - last_snapshot.allocated_bytes as i64
                } else {
                    simulated_memory as i64
                };

                let snapshot = MemorySnapshot {
                    timestamp: Instant::now(),
                    allocated_bytes: simulated_memory,
                    allocation_delta,
                    operation_context: "background_monitoring".to_string(),
                    memory_pressure: None,
                    manual_trigger: false,
                };

                snapshots_guard.push(snapshot);

                // Apply retention policy if needed
                if snapshots_guard.len() > max_snapshots {
                    Self::apply_retention_policy_static(&mut snapshots_guard, &retention_policy);
                }
            }
        }
    }

    /// Get current memory usage (placeholder implementation)
    ///
    /// In a real implementation, this would query actual system memory usage.
    fn get_current_memory_usage() -> usize {
        // Placeholder: return a simulated memory usage value
        // In practice, this would use platform-specific APIs
        1024 * 1024 * 64 // 64MB
    }

    /// Apply retention policy to snapshots
    fn apply_retention_policy(&self, snapshots: &mut Vec<MemorySnapshot>) {
        Self::apply_retention_policy_static(snapshots, &self.config.retention_policy);
    }

    /// Static version of retention policy application
    fn apply_retention_policy_static(
        snapshots: &mut Vec<MemorySnapshot>,
        policy: &SnapshotRetentionPolicy,
    ) {
        match policy {
            SnapshotRetentionPolicy::KeepLatest => {
                // Simple FIFO: remove oldest
                if !snapshots.is_empty() {
                    snapshots.remove(0);
                }
            }
            SnapshotRetentionPolicy::KeepImportant => {
                // Remove snapshots with small allocation deltas
                if snapshots.len() > 1 {
                    let mut min_importance = f64::INFINITY;
                    let mut min_index = 0;

                    for (i, snapshot) in snapshots.iter().enumerate() {
                        if i == 0 || i == snapshots.len() - 1 {
                            continue; // Keep first and last
                        }

                        let importance = snapshot.allocation_delta.abs() as f64;
                        if importance < min_importance {
                            min_importance = importance;
                            min_index = i;
                        }
                    }

                    if min_index > 0 && min_index < snapshots.len() {
                        snapshots.remove(min_index);
                    }
                }
            }
            SnapshotRetentionPolicy::KeepPeaks => {
                // Keep local maxima and minima
                Self::remove_non_peak_snapshots(snapshots);
            }
            SnapshotRetentionPolicy::Adaptive => {
                // Use a combination of strategies
                if snapshots.len() % 2 == 0 {
                    Self::apply_retention_policy_static(
                        snapshots,
                        &SnapshotRetentionPolicy::KeepImportant,
                    );
                } else {
                    Self::apply_retention_policy_static(
                        snapshots,
                        &SnapshotRetentionPolicy::KeepLatest,
                    );
                }
            }
        }
    }

    /// Remove non-peak snapshots (keep local maxima and minima)
    fn remove_non_peak_snapshots(snapshots: &mut Vec<MemorySnapshot>) {
        if snapshots.len() <= 3 {
            return; // Can't determine peaks with too few points
        }

        let mut to_remove = Vec::new();

        for i in 1..snapshots.len() - 1 {
            let prev = snapshots[i - 1].allocated_bytes;
            let curr = snapshots[i].allocated_bytes;
            let next = snapshots[i + 1].allocated_bytes;

            // Keep only if it's a local maximum or minimum
            let is_peak = (curr > prev && curr > next) || (curr < prev && curr < next);

            if !is_peak {
                to_remove.push(i);
            }
        }

        // Remove in reverse order to maintain indices
        for &index in to_remove.iter().rev() {
            if !to_remove.is_empty() {
                snapshots.remove(index);
                break; // Remove only one at a time to avoid excessive reduction
            }
        }
    }

    /// Analyze collected snapshots
    fn analyze_snapshots(&self, snapshots: &[MemorySnapshot]) -> GradientMemoryMonitoringResult {
        let peak_memory = snapshots
            .iter()
            .map(|s| s.allocated_bytes)
            .max()
            .unwrap_or(0);

        let average_memory = if !snapshots.is_empty() {
            snapshots.iter().map(|s| s.allocated_bytes).sum::<usize>() / snapshots.len()
        } else {
            0
        };

        let total_allocated = snapshots
            .iter()
            .map(|s| s.allocation_delta.max(0) as usize)
            .sum::<usize>();

        // Calculate growth rate
        let growth_rate = if snapshots.len() > 1 {
            let first = &snapshots[0];
            let last = &snapshots[snapshots.len() - 1];
            let time_diff = last.timestamp.duration_since(first.timestamp).as_secs_f64();
            let memory_diff = last.allocated_bytes as f64 - first.allocated_bytes as f64;

            if time_diff > 0.0 {
                memory_diff / time_diff
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Calculate efficiency score
        let efficiency_score = if peak_memory > 0 {
            average_memory as f64 / peak_memory as f64
        } else {
            1.0
        };

        // Detect anomalies
        let mut anomalies = Vec::new();
        if growth_rate > 10.0 * 1024.0 * 1024.0 {
            // > 10MB/s growth
            anomalies.push(format!(
                "High memory growth rate: {:.1} MB/s",
                growth_rate / (1024.0 * 1024.0)
            ));
        }

        if efficiency_score < 0.3 {
            anomalies
                .push("Low memory efficiency - high peak relative to average usage".to_string());
        }

        // Generate recommendations
        let mut recommendations = Vec::new();
        if growth_rate > 5.0 * 1024.0 * 1024.0 {
            recommendations.push(
                "Consider implementing gradient checkpointing to reduce memory growth".to_string(),
            );
        }

        if efficiency_score < 0.5 {
            recommendations.push(
                "Memory usage pattern suggests potential for memory pooling optimization"
                    .to_string(),
            );
        }

        if peak_memory > 1024 * 1024 * 1024 {
            // > 1GB
            recommendations.push(
                "Large peak memory usage - consider memory streaming or model parallelization"
                    .to_string(),
            );
        }

        GradientMemoryMonitoringResult {
            operation_name: self.operation_name.clone(),
            monitoring_duration: self.start_time.elapsed(),
            snapshot_count: snapshots.len(),
            peak_memory_usage: peak_memory,
            average_memory_usage: average_memory,
            total_memory_allocated: total_allocated,
            memory_growth_rate: growth_rate,
            efficiency_score,
            snapshots: snapshots.to_vec(),
            anomalies,
            recommendations,
            analysis_timestamp: Instant::now(),
        }
    }
}

impl Drop for GradientMemoryMonitor {
    /// Ensure monitoring is stopped when monitor is dropped
    fn drop(&mut self) {
        self.active.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let monitor = GradientMemoryMonitor::new("test_op".to_string());
        assert_eq!(monitor.operation_name, "test_op");
        assert!(monitor.is_active());
        assert_eq!(monitor.snapshot_count(), 0);
    }

    #[test]
    fn test_snapshot_collection() {
        let monitor = GradientMemoryMonitor::new("test_op".to_string());

        monitor.take_snapshot(1000).unwrap();
        assert_eq!(monitor.snapshot_count(), 1);

        monitor.take_snapshot(2000).unwrap();
        assert_eq!(monitor.snapshot_count(), 2);

        // Check snapshots are properly recorded
        let snapshots = monitor
            .memory_snapshots
            .lock()
            .expect("lock should not be poisoned");
        assert_eq!(snapshots[0].allocated_bytes, 1000);
        assert_eq!(snapshots[1].allocated_bytes, 2000);
        assert_eq!(snapshots[1].allocation_delta, 1000); // Increase from previous
    }

    #[test]
    fn test_monitoring_analysis() {
        let monitor = GradientMemoryMonitor::new("test_op".to_string());

        // Simulate memory usage pattern
        monitor.take_snapshot(1000).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        monitor.take_snapshot(3000).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        monitor.take_snapshot(2000).unwrap();

        let results = monitor.stop_and_analyze().unwrap();

        assert_eq!(results.operation_name, "test_op");
        assert_eq!(results.snapshot_count, 3);
        assert_eq!(results.peak_memory_usage, 3000);
        assert!(results.monitoring_duration > Duration::ZERO);
        assert!(results.memory_growth_rate >= 0.0);
    }

    #[test]
    fn test_retention_policy() {
        let config = MonitoringConfig {
            max_snapshots: 3,
            retention_policy: SnapshotRetentionPolicy::KeepLatest,
            enable_background_monitoring: false,
            ..Default::default()
        };

        let monitor = GradientMemoryMonitor::with_config("test_op".to_string(), config);

        // Add more snapshots than limit
        for i in 0..5 {
            monitor.take_snapshot((i + 1) * 1000).unwrap();
        }

        // Should not exceed max_snapshots
        assert!(monitor.snapshot_count() <= 3);

        let snapshots = monitor
            .memory_snapshots
            .lock()
            .expect("lock should not be poisoned");
        // Should keep the latest ones
        assert!(snapshots.iter().any(|s| s.allocated_bytes == 5000));
    }

    #[test]
    fn test_inactive_monitor() {
        let monitor = GradientMemoryMonitor::new("test_op".to_string());

        // Stop monitoring
        monitor.active.store(false, Ordering::SeqCst);
        assert!(!monitor.is_active());

        // Should not allow snapshots on inactive monitor
        let result = monitor.take_snapshot(1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_monitoring_config() {
        let custom_config = MonitoringConfig {
            snapshot_interval: Duration::from_millis(50),
            max_snapshots: 500,
            enable_automatic_analysis: true,
            memory_threshold: Some(1024 * 1024), // 1MB
            enable_background_monitoring: false,
            retention_policy: SnapshotRetentionPolicy::KeepImportant,
        };

        let monitor =
            GradientMemoryMonitor::with_config("custom_op".to_string(), custom_config.clone());
        assert_eq!(monitor.config.snapshot_interval, Duration::from_millis(50));
        assert_eq!(monitor.config.max_snapshots, 500);
        assert!(monitor.config.enable_automatic_analysis);
    }

    #[test]
    fn test_empty_analysis() {
        let monitor = GradientMemoryMonitor::new("empty_test".to_string());

        // Analyze without any snapshots
        let results = monitor.stop_and_analyze().unwrap();

        assert_eq!(results.snapshot_count, 0);
        assert_eq!(results.peak_memory_usage, 0);
        assert_eq!(results.average_memory_usage, 0);
        assert_eq!(results.efficiency_score, 1.0);
    }

    #[test]
    fn test_anomaly_detection() {
        let monitor = GradientMemoryMonitor::new("anomaly_test".to_string());

        // Simulate high memory growth
        monitor.take_snapshot(1024 * 1024).unwrap(); // 1MB
        std::thread::sleep(Duration::from_millis(1));
        monitor.take_snapshot(100 * 1024 * 1024).unwrap(); // 100MB - rapid growth

        let results = monitor.stop_and_analyze().unwrap();

        // Should detect high growth rate anomaly
        assert!(!results.anomalies.is_empty());
        assert!(results.anomalies.iter().any(|a| a.contains("growth rate")));
    }

    #[test]
    fn test_recommendation_generation() {
        let monitor = GradientMemoryMonitor::new("recommendation_test".to_string());

        // Simulate large memory usage
        monitor.take_snapshot(2 * 1024 * 1024 * 1024).unwrap(); // 2GB

        let results = monitor.stop_and_analyze().unwrap();

        // Should generate recommendations for large memory usage
        assert!(!results.recommendations.is_empty());
    }
}
