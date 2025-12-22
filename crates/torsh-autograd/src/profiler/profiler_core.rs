//! Core autograd profiler implementation
//!
//! This module provides the main `AutogradProfiler` implementation that coordinates
//! all profiling functionality including timing, memory tracking, hardware monitoring,
//! performance analysis, and complexity analysis.
//!
//! # Overview
//!
//! The `AutogradProfiler` is the central component that orchestrates all profiling
//! activities for automatic differentiation operations. It integrates:
//!
//! - **Operation Timing**: Precise measurement of operation execution times
//! - **Memory Tracking**: Monitoring memory usage patterns and allocations
//! - **Hardware Monitoring**: Real-time hardware utilization tracking
//! - **Performance Analysis**: Bottleneck detection and analysis
//! - **Session Management**: Complete profiling session lifecycle management
//!
//! # Examples
//!
//! ```rust,ignore
//! use crate::profiler::{AutogradProfiler, ProfilerConfig};
//! use crate::context::AutogradContext;
//!
//! let config = ProfilerConfig::default();
//! let mut profiler = AutogradProfiler::new(config);
//!
//! // Start profiling session
//! profiler.start_session("training_run_1".to_string())?;
//!
//! // Profile operations
//! profiler.start_operation("forward_pass".to_string())?;
//! // ... perform operation ...
//! profiler.end_operation("forward_pass")?;
//!
//! // End session and get results
//! let profile = profiler.end_session()?;
//! let report = profiler.generate_report(&profile)?;
//! println!("{}", report);
//! ```

use crate::context::AutogradContext;
use crate::profiler::analysis::PerformanceAnalyzer;
use crate::profiler::hardware::HardwareMonitor;
use crate::profiler::memory::MemoryTracker;
use crate::profiler::types::{
    AutogradProfile, HardwareUtilization, OperationProfile, ProfileSummary,
};
use std::collections::HashMap;
use std::fmt::Write;
use std::time::{Duration, Instant, SystemTime};
use torsh_core::error::{Result, TorshError};
use tracing::{info, warn};

/// Autograd performance profiler
///
/// The main profiler that coordinates all profiling activities for automatic
/// differentiation operations. Provides comprehensive performance analysis
/// including timing, memory usage, hardware utilization, and bottleneck detection.
///
/// # Thread Safety
///
/// This profiler is not thread-safe and should be used from a single thread
/// or protected by appropriate synchronization primitives.
///
/// # Performance Impact
///
/// Profiling introduces minimal overhead when disabled, but can have measurable
/// impact when enabled. Use `ProfilerConfig` to selectively enable only needed
/// profiling features for production workloads.
pub struct AutogradProfiler {
    /// Configuration
    config: ProfilerConfig,
    /// Current session profile
    current_profile: Option<AutogradProfile>,
    /// Operation timing stack
    operation_stack: Vec<(String, Instant)>,
    /// Memory tracking
    memory_tracker: MemoryTracker,
    /// Hardware monitor
    hardware_monitor: HardwareMonitor,
    /// Performance analyzer
    analyzer: PerformanceAnalyzer,
}

/// Configuration for the profiler
///
/// Controls which profiling features are enabled and their behavior.
/// Allows fine-tuning of profiling overhead vs. detail level.
///
/// # Performance Considerations
///
/// - **Memory tracking**: Low overhead, recommended for most use cases
/// - **Hardware monitoring**: Medium overhead, useful for optimization
/// - **Bottleneck detection**: Low overhead, provides valuable insights
/// - **Detailed timing**: Very low overhead when using efficient timers
/// - **FLOPS counting**: High overhead, only enable when needed
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Enable memory tracking
    pub enable_memory_tracking: bool,
    /// Enable hardware monitoring
    pub enable_hardware_monitoring: bool,
    /// Enable bottleneck detection
    pub enable_bottleneck_detection: bool,
    /// Memory snapshot interval
    pub memory_snapshot_interval: Duration,
    /// Maximum number of operation profiles to keep
    pub max_operation_profiles: usize,
    /// Enable detailed timing
    pub enable_detailed_timing: bool,
    /// Enable FLOPS counting
    pub enable_flops_counting: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_memory_tracking: true,
            enable_hardware_monitoring: true,
            enable_bottleneck_detection: true,
            memory_snapshot_interval: Duration::from_millis(100),
            max_operation_profiles: 1000,
            enable_detailed_timing: true,
            enable_flops_counting: false, // Expensive to compute
        }
    }
}

impl AutogradProfiler {
    /// Create a new autograd profiler
    ///
    /// Initializes all profiling components according to the provided configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying which profiling features to enable
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = ProfilerConfig {
    ///     enable_memory_tracking: true,
    ///     enable_hardware_monitoring: false, // Disable for lower overhead
    ///     ..ProfilerConfig::default()
    /// };
    /// let profiler = AutogradProfiler::new(config);
    /// ```
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            memory_tracker: MemoryTracker::new(config.memory_snapshot_interval),
            hardware_monitor: HardwareMonitor::new(),
            analyzer: PerformanceAnalyzer::new(),
            config,
            current_profile: None,
            operation_stack: Vec::new(),
        }
    }

    /// Start a new profiling session
    ///
    /// Begins a new profiling session with the specified ID. Only one session
    /// can be active at a time.
    ///
    /// # Arguments
    ///
    /// * `session_id` - Unique identifier for the profiling session
    ///
    /// # Errors
    ///
    /// Returns an error if a session is already active.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// profiler.start_session("training_epoch_1".to_string())?;
    /// ```
    pub fn start_session(&mut self, session_id: String) -> Result<()> {
        info!("Starting autograd profiling session: {}", session_id);

        self.current_profile = Some(AutogradProfile {
            session_id,
            start_time: SystemTime::now(),
            duration: Duration::ZERO,
            total_operations: 0,
            operation_profiles: HashMap::new(),
            memory_timeline: Vec::new(),
            bottlenecks: Vec::new(),
            hardware_utilization: HardwareUtilization {
                cpu_utilization: 0.0,
                gpu_utilization: None,
                memory_utilization: 0.0,
                memory_bandwidth_utilization: 0.0,
                cache_hit_rate: 0.0,
            },
            summary: ProfileSummary {
                total_time: Duration::ZERO,
                forward_time: Duration::ZERO,
                backward_time: Duration::ZERO,
                memory_time: Duration::ZERO,
                peak_memory: 0,
                average_memory: 0,
                total_flops: 0.0,
                flops_per_second: 0.0,
                most_expensive_operation: None,
                memory_bound_percentage: 0.0,
                compute_bound_percentage: 0.0,
            },
        });

        Ok(())
    }

    /// End the current profiling session
    ///
    /// Finalizes the current profiling session, performs final analysis,
    /// and returns the complete profiling results.
    ///
    /// # Returns
    ///
    /// Returns the `AutogradProfile` containing all profiling data and analysis.
    ///
    /// # Errors
    ///
    /// Returns an error if no session is currently active or if finalization fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let profile = profiler.end_session()?;
    /// println!("Session duration: {:?}", profile.duration);
    /// ```
    pub fn end_session(&mut self) -> Result<AutogradProfile> {
        if let Some(mut profile) = self.current_profile.take() {
            let end_time = SystemTime::now();
            profile.duration = end_time
                .duration_since(profile.start_time)
                .unwrap_or(Duration::ZERO);

            // Finalize analysis
            self.finalize_profile(&mut profile)?;

            info!(
                "Ended profiling session: {} (duration: {:?})",
                profile.session_id, profile.duration
            );
            Ok(profile)
        } else {
            Err(TorshError::AutogradError(
                "No active profiling session".to_string(),
            ))
        }
    }

    /// Start timing an operation
    ///
    /// Begins timing an operation and optionally records memory and hardware state.
    /// Operations can be nested, and timing is tracked using a stack.
    ///
    /// # Arguments
    ///
    /// * `operation_name` - Name of the operation being started
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// profiler.start_operation("matrix_multiply".to_string())?;
    /// // ... perform matrix multiplication ...
    /// profiler.end_operation("matrix_multiply")?;
    /// ```
    pub fn start_operation(&mut self, operation_name: String) -> Result<()> {
        if self.config.enable_detailed_timing {
            self.operation_stack
                .push((operation_name.clone(), Instant::now()));
        }

        if self.config.enable_memory_tracking {
            self.memory_tracker
                .maybe_take_snapshot(Some(operation_name));
        }

        if self.config.enable_hardware_monitoring {
            self.hardware_monitor.maybe_update_utilization();
        }

        Ok(())
    }

    /// End timing an operation
    ///
    /// Completes timing for the specified operation and records the results.
    /// The operation name must match the most recent `start_operation` call.
    ///
    /// # Arguments
    ///
    /// * `operation_name` - Name of the operation being ended
    ///
    /// # Errors
    ///
    /// Logs a warning if the operation name doesn't match the expected operation
    /// on the timing stack, but continues execution to maintain robustness.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// profiler.start_operation("forward_pass".to_string())?;
    /// // ... forward pass computation ...
    /// profiler.end_operation("forward_pass")?;
    /// ```
    pub fn end_operation(&mut self, operation_name: &str) -> Result<()> {
        if !self.config.enable_detailed_timing {
            return Ok(());
        }

        if let Some((name, start_time)) = self.operation_stack.pop() {
            if name == operation_name {
                let duration = start_time.elapsed();
                self.record_operation_timing(operation_name, duration)?;
            } else {
                warn!(
                    "Operation stack mismatch: expected {}, got {}",
                    name, operation_name
                );
                // Put it back and continue
                self.operation_stack.push((name, start_time));
            }
        }

        Ok(())
    }

    /// Record gradient computation timing
    ///
    /// Records the time spent computing gradients for a specific operation.
    /// This information is used for backward pass analysis and optimization.
    ///
    /// # Arguments
    ///
    /// * `operation_name` - Name of the operation
    /// * `duration` - Time spent computing gradients
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let start = Instant::now();
    /// // ... gradient computation ...
    /// let grad_time = start.elapsed();
    /// profiler.record_gradient_timing("conv2d", grad_time)?;
    /// ```
    pub fn record_gradient_timing(
        &mut self,
        operation_name: &str,
        duration: Duration,
    ) -> Result<()> {
        if let Some(profile) = &mut self.current_profile {
            if let Some(op_profile) = profile.operation_profiles.get_mut(operation_name) {
                op_profile.gradient_time = Some(duration);
            }
        }
        Ok(())
    }

    /// Profile a computation graph execution
    ///
    /// Profiles the execution of a computation graph operation, automatically
    /// handling timing, memory tracking, and error propagation.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Autograd context for the operation
    /// * `operation_name` - Name of the operation being profiled
    /// * `operation` - Closure that performs the computation
    ///
    /// # Returns
    ///
    /// Returns the result of the operation closure.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let result = profiler.profile_graph_execution(
    ///     &mut ctx,
    ///     "neural_network_forward",
    ///     |ctx| {
    ///         // Neural network forward pass computation
    ///         Ok(computed_output)
    ///     }
    /// )?;
    /// ```
    pub fn profile_graph_execution<F, R>(
        &mut self,
        ctx: &mut AutogradContext,
        operation_name: &str,
        operation: F,
    ) -> Result<R>
    where
        F: FnOnce(&mut AutogradContext) -> Result<R>,
    {
        self.start_operation(operation_name.to_string())?;

        let start_memory = if self.config.enable_memory_tracking {
            Some(self.memory_tracker.estimate_total_memory())
        } else {
            None
        };

        let result = operation(ctx);

        let end_memory = if self.config.enable_memory_tracking {
            Some(self.memory_tracker.estimate_total_memory())
        } else {
            None
        };

        self.end_operation(operation_name)?;

        // Record memory allocation if tracking is enabled
        if let (Some(start_mem), Some(end_mem)) = (start_memory, end_memory) {
            if let Some(profile) = &mut self.current_profile {
                if let Some(op_profile) = profile.operation_profiles.get_mut(operation_name) {
                    op_profile.memory_allocated = end_mem.saturating_sub(start_mem);
                }
            }
        }

        result
    }

    /// Generate a comprehensive performance report
    ///
    /// Creates a detailed human-readable report containing all profiling results
    /// including timing analysis, memory usage, hardware utilization, and bottlenecks.
    ///
    /// # Arguments
    ///
    /// * `profile` - The profiling session results to report on
    ///
    /// # Returns
    ///
    /// Returns a formatted string containing the complete performance report.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let profile = profiler.end_session()?;
    /// let report = profiler.generate_report(&profile)?;
    /// println!("{}", report);
    /// // Write to file
    /// std::fs::write("profile_report.txt", report)?;
    /// ```
    pub fn generate_report(&self, profile: &AutogradProfile) -> Result<String> {
        let mut report = String::new();

        writeln!(report, "=== Autograd Performance Report ===").unwrap();
        writeln!(report, "Session: {}", profile.session_id).unwrap();
        writeln!(report, "Duration: {:?}", profile.duration).unwrap();
        writeln!(report, "Total Operations: {}", profile.total_operations).unwrap();
        writeln!(report).unwrap();

        // Summary statistics
        writeln!(report, "=== Summary Statistics ===").unwrap();
        writeln!(report, "Total Time: {:?}", profile.summary.total_time).unwrap();
        writeln!(report, "Forward Time: {:?}", profile.summary.forward_time).unwrap();
        writeln!(report, "Backward Time: {:?}", profile.summary.backward_time).unwrap();
        writeln!(
            report,
            "Peak Memory: {} MB",
            profile.summary.peak_memory / 1024 / 1024
        )
        .unwrap();
        writeln!(
            report,
            "Average Memory: {} MB",
            profile.summary.average_memory / 1024 / 1024
        )
        .unwrap();
        if profile.summary.total_flops > 0.0 {
            writeln!(report, "Total FLOPS: {:.2e}", profile.summary.total_flops).unwrap();
            writeln!(
                report,
                "FLOPS/sec: {:.2e}",
                profile.summary.flops_per_second
            )
            .unwrap();
        }
        if let Some(ref most_expensive) = profile.summary.most_expensive_operation {
            writeln!(report, "Most Expensive Operation: {most_expensive}").unwrap();
        }
        writeln!(report).unwrap();

        // Hardware utilization
        writeln!(report, "=== Hardware Utilization ===").unwrap();
        writeln!(
            report,
            "CPU Utilization: {:.1}%",
            profile.hardware_utilization.cpu_utilization
        )
        .unwrap();
        if let Some(gpu_util) = profile.hardware_utilization.gpu_utilization {
            writeln!(report, "GPU Utilization: {gpu_util:.1}%").unwrap();
        }
        writeln!(
            report,
            "Memory Utilization: {:.1}%",
            profile.hardware_utilization.memory_utilization
        )
        .unwrap();
        writeln!(
            report,
            "Memory Bandwidth: {:.1}%",
            profile.hardware_utilization.memory_bandwidth_utilization
        )
        .unwrap();
        writeln!(
            report,
            "Cache Hit Rate: {:.1}%",
            profile.hardware_utilization.cache_hit_rate
        )
        .unwrap();
        writeln!(report).unwrap();

        // Top operations by time
        writeln!(report, "=== Top Operations by Time ===").unwrap();
        let mut ops_by_time: Vec<_> = profile.operation_profiles.iter().collect();
        ops_by_time.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

        for (i, (op_name, op_profile)) in ops_by_time.iter().take(10).enumerate() {
            writeln!(
                report,
                "{}. {} - Total: {:?}, Avg: {:?}, Count: {}",
                i + 1,
                op_name,
                op_profile.total_time,
                op_profile.average_time,
                op_profile.execution_count
            )
            .unwrap();
        }
        writeln!(report).unwrap();

        // Performance bottlenecks
        if !profile.bottlenecks.is_empty() {
            writeln!(report, "=== Performance Bottlenecks ===").unwrap();
            for (i, bottleneck) in profile.bottlenecks.iter().enumerate() {
                writeln!(
                    report,
                    "{}. {:?} - {} (Severity: {:.2})",
                    i + 1,
                    bottleneck.bottleneck_type,
                    bottleneck.operation,
                    bottleneck.severity
                )
                .unwrap();
                writeln!(report, "   Description: {}", bottleneck.description).unwrap();
                writeln!(report, "   Suggestion: {}", bottleneck.suggestion).unwrap();
                writeln!(report).unwrap();
            }
        }

        // Memory timeline summary
        if !profile.memory_timeline.is_empty() {
            writeln!(report, "=== Memory Usage Timeline ===").unwrap();
            let peak_snapshot = profile
                .memory_timeline
                .iter()
                .max_by_key(|s| s.total_memory)
                .unwrap();
            let avg_memory = profile
                .memory_timeline
                .iter()
                .map(|s| s.total_memory)
                .sum::<usize>()
                / profile.memory_timeline.len();

            writeln!(
                report,
                "Peak Memory: {} MB",
                peak_snapshot.total_memory / 1024 / 1024
            )
            .unwrap();
            writeln!(report, "Average Memory: {} MB", avg_memory / 1024 / 1024).unwrap();
            writeln!(
                report,
                "Memory Snapshots: {}",
                profile.memory_timeline.len()
            )
            .unwrap();
        }

        Ok(report)
    }

    /// Export profile data in JSON format
    ///
    /// Exports profiling results in JSON format for integration with external
    /// analysis tools or dashboards.
    ///
    /// # Arguments
    ///
    /// * `profile` - The profiling session results to export
    ///
    /// # Returns
    ///
    /// Returns a JSON string representation of the profiling data.
    ///
    /// # Note
    ///
    /// This is a simplified JSON export. In production, consider using
    /// serde_json for more robust serialization.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let profile = profiler.end_session()?;
    /// let json_data = profiler.export_json(&profile)?;
    /// std::fs::write("profile_data.json", json_data)?;
    /// ```
    pub fn export_json(&self, profile: &AutogradProfile) -> Result<String> {
        // In a real implementation, you would use serde_json
        // For now, return a simplified JSON representation
        let mut json = String::new();

        writeln!(json, "{{").unwrap();
        writeln!(json, "  \"session_id\": \"{}\",", profile.session_id).unwrap();
        writeln!(json, "  \"duration_ms\": {},", profile.duration.as_millis()).unwrap();
        writeln!(
            json,
            "  \"total_operations\": {},",
            profile.total_operations
        )
        .unwrap();
        writeln!(
            json,
            "  \"peak_memory_mb\": {},",
            profile.summary.peak_memory / 1024 / 1024
        )
        .unwrap();
        writeln!(json, "  \"hardware_utilization\": {{").unwrap();
        writeln!(
            json,
            "    \"cpu\": {},",
            profile.hardware_utilization.cpu_utilization
        )
        .unwrap();
        if let Some(gpu_util) = profile.hardware_utilization.gpu_utilization {
            writeln!(json, "    \"gpu\": {gpu_util},").unwrap();
        }
        writeln!(
            json,
            "    \"memory\": {}",
            profile.hardware_utilization.memory_utilization
        )
        .unwrap();
        writeln!(json, "  }},").unwrap();
        writeln!(json, "  \"bottlenecks\": {},", profile.bottlenecks.len()).unwrap();
        writeln!(json, "}}").unwrap();

        Ok(json)
    }

    /// Helper: Record operation timing
    ///
    /// Internal method to record timing data for an operation.
    /// Updates or creates operation profile with timing statistics.
    fn record_operation_timing(&mut self, operation_name: &str, duration: Duration) -> Result<()> {
        if let Some(profile) = &mut self.current_profile {
            profile.total_operations += 1;

            let op_profile = profile
                .operation_profiles
                .entry(operation_name.to_string())
                .or_insert_with(|| OperationProfile {
                    operation_name: operation_name.to_string(),
                    execution_count: 0,
                    total_time: Duration::ZERO,
                    average_time: Duration::ZERO,
                    min_time: Duration::MAX,
                    max_time: Duration::ZERO,
                    memory_allocated: 0,
                    peak_memory: 0,
                    flops: 0.0,
                    input_sizes: Vec::new(),
                    output_sizes: Vec::new(),
                    gradient_time: None,
                });

            op_profile.execution_count += 1;
            op_profile.total_time += duration;
            op_profile.average_time = op_profile.total_time / op_profile.execution_count as u32;
            op_profile.min_time = op_profile.min_time.min(duration);
            op_profile.max_time = op_profile.max_time.max(duration);
        }

        Ok(())
    }

    /// Helper: Finalize profile analysis
    ///
    /// Performs final analysis including memory timeline copying,
    /// hardware utilization recording, bottleneck detection,
    /// and summary statistics calculation.
    fn finalize_profile(&mut self, profile: &mut AutogradProfile) -> Result<()> {
        // Copy memory snapshots
        profile.memory_timeline = self.memory_tracker.get_snapshots().to_vec();

        // Copy hardware utilization
        if let Some(utilization) = self.hardware_monitor.get_current_utilization() {
            profile.hardware_utilization = utilization.clone();
        }

        // Analyze bottlenecks
        profile.bottlenecks = self.analyzer.analyze_bottlenecks(profile);

        // Calculate summary statistics
        self.calculate_summary_statistics(profile)?;

        Ok(())
    }

    /// Helper: Calculate summary statistics
    ///
    /// Computes comprehensive summary statistics including timing breakdowns,
    /// memory usage patterns, and performance metrics.
    fn calculate_summary_statistics(&self, profile: &mut AutogradProfile) -> Result<()> {
        let mut total_time = Duration::ZERO;
        let mut forward_time = Duration::ZERO;
        let mut backward_time = Duration::ZERO;
        let mut most_expensive_op: Option<String> = None;
        let mut max_time = Duration::ZERO;

        for (op_name, op_profile) in &profile.operation_profiles {
            total_time += op_profile.total_time;

            // Classify operations
            if op_name.contains("forward") || op_name.contains("Forward") {
                forward_time += op_profile.total_time;
            } else if op_name.contains("backward") || op_name.contains("Backward") {
                backward_time += op_profile.total_time;
            }

            // Find most expensive operation
            if op_profile.total_time > max_time {
                max_time = op_profile.total_time;
                most_expensive_op = Some(op_name.clone());
            }
        }

        // Memory statistics
        let (peak_memory, average_memory) = if !profile.memory_timeline.is_empty() {
            let peak = profile
                .memory_timeline
                .iter()
                .map(|s| s.total_memory)
                .max()
                .unwrap_or(0);
            let avg = profile
                .memory_timeline
                .iter()
                .map(|s| s.total_memory)
                .sum::<usize>()
                / profile.memory_timeline.len();
            (peak, avg)
        } else {
            (0, 0)
        };

        profile.summary = ProfileSummary {
            total_time,
            forward_time,
            backward_time,
            memory_time: Duration::ZERO, // Could be calculated from memory operations
            peak_memory,
            average_memory,
            total_flops: 0.0, // Would need to be calculated based on operations
            flops_per_second: 0.0,
            most_expensive_operation: most_expensive_op,
            memory_bound_percentage: 50.0,  // Placeholder
            compute_bound_percentage: 50.0, // Placeholder
        };

        // Calculate FLOPS per second if we have FLOPS data
        if profile.summary.total_flops > 0.0 && total_time.as_secs_f64() > 0.0 {
            profile.summary.flops_per_second =
                profile.summary.total_flops / total_time.as_secs_f64();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let config = ProfilerConfig::default();
        let profiler = AutogradProfiler::new(config);
        assert!(profiler.current_profile.is_none());
    }

    #[test]
    fn test_profiling_session() {
        let config = ProfilerConfig::default();
        let mut profiler = AutogradProfiler::new(config);

        // Start session
        profiler.start_session("test_session".to_string()).unwrap();
        assert!(profiler.current_profile.is_some());

        // Simulate some operations
        profiler.start_operation("test_op".to_string()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
        profiler.end_operation("test_op").unwrap();

        // End session
        let profile = profiler.end_session().unwrap();
        assert_eq!(profile.session_id, "test_session");
        assert!(profile.total_operations > 0);
    }

    #[test]
    fn test_operation_timing() {
        let config = ProfilerConfig::default();
        let mut profiler = AutogradProfiler::new(config);

        profiler.start_session("timing_test".to_string()).unwrap();

        // Time an operation
        let start = Instant::now();
        profiler.start_operation("test_timing".to_string()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        profiler.end_operation("test_timing").unwrap();
        let _duration = start.elapsed();

        let profile = profiler.end_session().unwrap();
        assert!(profile.operation_profiles.contains_key("test_timing"));

        let op_profile = &profile.operation_profiles["test_timing"];
        assert_eq!(op_profile.execution_count, 1);
        assert!(op_profile.total_time >= std::time::Duration::from_millis(4)); // Allow some tolerance
    }

    #[test]
    fn test_config_defaults() {
        let config = ProfilerConfig::default();
        assert!(config.enable_memory_tracking);
        assert!(config.enable_hardware_monitoring);
        assert!(config.enable_bottleneck_detection);
        assert!(config.enable_detailed_timing);
        assert!(!config.enable_flops_counting); // Should be false by default due to overhead
        assert_eq!(config.memory_snapshot_interval, Duration::from_millis(100));
        assert_eq!(config.max_operation_profiles, 1000);
    }

    #[test]
    fn test_operation_stack_mismatch() {
        let config = ProfilerConfig::default();
        let mut profiler = AutogradProfiler::new(config);

        profiler.start_session("stack_test".to_string()).unwrap();

        // Start nested operations
        profiler.start_operation("outer_op".to_string()).unwrap();
        profiler.start_operation("inner_op".to_string()).unwrap();

        // End in wrong order - should handle gracefully
        profiler.end_operation("outer_op").unwrap(); // Should warn but continue

        let profile = profiler.end_session().unwrap();
        // Should still have recorded some timing data
        assert_eq!(profile.session_id, "stack_test");
    }

    #[test]
    fn test_gradient_timing() {
        let config = ProfilerConfig::default();
        let mut profiler = AutogradProfiler::new(config);

        profiler.start_session("gradient_test".to_string()).unwrap();

        // Record operation first
        profiler.start_operation("conv2d".to_string()).unwrap();
        profiler.end_operation("conv2d").unwrap();

        // Record gradient timing
        let grad_time = Duration::from_millis(15);
        profiler
            .record_gradient_timing("conv2d", grad_time)
            .unwrap();

        let profile = profiler.end_session().unwrap();
        let op_profile = &profile.operation_profiles["conv2d"];
        assert_eq!(op_profile.gradient_time, Some(grad_time));
    }

    #[test]
    fn test_report_generation() {
        let config = ProfilerConfig::default();
        let mut profiler = AutogradProfiler::new(config);

        profiler.start_session("report_test".to_string()).unwrap();

        // Add some operations
        profiler
            .start_operation("forward_pass".to_string())
            .unwrap();
        std::thread::sleep(Duration::from_millis(2));
        profiler.end_operation("forward_pass").unwrap();

        profiler
            .start_operation("backward_pass".to_string())
            .unwrap();
        std::thread::sleep(Duration::from_millis(3));
        profiler.end_operation("backward_pass").unwrap();

        let profile = profiler.end_session().unwrap();
        let report = profiler.generate_report(&profile).unwrap();

        // Check that report contains expected sections
        assert!(report.contains("=== Autograd Performance Report ==="));
        assert!(report.contains("=== Summary Statistics ==="));
        assert!(report.contains("=== Hardware Utilization ==="));
        assert!(report.contains("=== Top Operations by Time ==="));
        assert!(report.contains("forward_pass"));
        assert!(report.contains("backward_pass"));
    }

    #[test]
    fn test_json_export() {
        let config = ProfilerConfig::default();
        let mut profiler = AutogradProfiler::new(config);

        profiler.start_session("json_test".to_string()).unwrap();
        profiler.start_operation("test_op".to_string()).unwrap();
        profiler.end_operation("test_op").unwrap();

        let profile = profiler.end_session().unwrap();
        let json = profiler.export_json(&profile).unwrap();

        // Check that JSON contains expected fields
        assert!(json.contains("\"session_id\": \"json_test\""));
        assert!(json.contains("\"duration_ms\":"));
        assert!(json.contains("\"total_operations\":"));
        assert!(json.contains("\"hardware_utilization\":"));
    }
}
