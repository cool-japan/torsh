//! Comprehensive system metrics collection for benchmarks
//!
//! This module provides a complete metrics collection system organized into focused modules:
//! - Core metrics collection (MetricsCollector, SystemMetrics, etc.)
//! - Resource tracking (MemoryTracker, CpuTracker)
//! - Performance profiling (PerformanceProfiler, ScopedProfiler)
//! - Cross-framework comparison (CrossFrameworkComparison)
//! - Power consumption monitoring (PowerMonitor, PowerMetrics)
//!
//! The module maintains full backward compatibility while providing improved organization.

// Core modules
pub mod core;
pub mod cross_framework;
pub mod power;
pub mod profiling;
pub mod tracking;

// Re-export core types for backward compatibility
pub use core::{CpuStats, MemoryStats, MetricsCollector, SystemMetrics};

// Re-export tracking components
pub use tracking::{CpuTracker, MemoryTracker};

// Re-export profiling components
pub use profiling::{
    FunctionStats, PerformanceProfiler, PerformanceReport, ProfileEvent, ProfileEventType,
    ScopedProfiler,
};

// Re-export cross-framework types
pub use cross_framework::{
    CrossFrameworkComparison, Framework, FrameworkComparison, FrameworkPerformance, HardwareInfo,
    OperationType, ScalabilityAnalysis, ScalabilityPoint, UnifiedMetrics,
};

// Re-export power monitoring
pub use power::{PowerMetrics, PowerMonitor};

/// Macro for scoped profiling
#[macro_export]
macro_rules! profile {
    ($profiler:expr, $name:expr) => {
        let _scoped_profiler = $crate::metrics::ScopedProfiler::new($profiler, $name);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_metrics_collector_basic_usage() {
        let mut collector = MetricsCollector::new();
        collector.start();

        // Simulate some work
        std::thread::sleep(Duration::from_millis(10));

        let metrics = collector.stop();
        assert!(metrics.elapsed_time > Duration::ZERO);
    }

    #[test]
    fn test_memory_tracker_functionality() {
        let mut tracker = MemoryTracker::new();
        tracker.start();

        // Sample some data
        tracker.sample();

        let stats = tracker.stop();
        assert_eq!(stats.allocation_count, 0); // Placeholder implementation
    }

    #[test]
    fn test_cpu_tracker_functionality() {
        let mut tracker = CpuTracker::new();
        tracker.start();

        tracker.sample();

        let stats = tracker.stop();
        assert!(stats.cores_used > 0);
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new();

        profiler.begin_event("test_operation");
        std::thread::sleep(Duration::from_millis(1));
        profiler.end_event("test_operation");

        let report = profiler.generate_report();
        assert_eq!(report.function_stats.len(), 1);
        assert_eq!(report.function_stats[0].name, "test_operation");
        assert_eq!(report.function_stats[0].call_count, 1);
    }

    #[test]
    fn test_scoped_profiler() {
        let mut profiler = PerformanceProfiler::new();

        {
            let _scoped = ScopedProfiler::new(&mut profiler, "scoped_test");
            std::thread::sleep(Duration::from_millis(1));
        } // ScopedProfiler drops here and automatically ends the event

        let report = profiler.generate_report();
        assert_eq!(report.function_stats.len(), 1);
        assert_eq!(report.function_stats[0].name, "scoped_test");
    }

    #[test]
    fn test_cross_framework_comparison() {
        use cross_framework::converters;

        let mut comparison = CrossFrameworkComparison::new();

        let metrics = converters::create_default_unified_metrics(
            Framework::Torsh,
            OperationType::MatrixMultiplication,
            1000.0, // 1 microsecond
            vec![100, 100],
        );

        comparison.add_metrics(metrics);

        let frameworks = comparison.get_frameworks();
        assert_eq!(frameworks.len(), 1);
        assert_eq!(frameworks[0], Framework::Torsh);
    }

    #[test]
    fn test_power_monitor_basic() {
        let _monitor = PowerMonitor::new();

        // Test that we can create a monitor with custom interval
        let _custom_monitor = PowerMonitor::new().with_sampling_interval(Duration::from_millis(50));

        // Basic functionality test (without actually starting monitoring)
        let default_metrics = PowerMetrics::default();
        assert_eq!(default_metrics.total_energy_joules, 0.0);
        assert_eq!(default_metrics.average_power_watts, 0.0);
    }

    #[test]
    fn test_system_metrics_calculations() {
        let memory_stats = MemoryStats {
            initial_usage_mb: 100.0,
            peak_usage_mb: 200.0,
            final_usage_mb: 150.0,
            allocated_mb: 50.0,
            deallocated_mb: 0.0,
            allocation_count: 10,
            deallocation_count: 5,
        };

        let cpu_stats = CpuStats {
            average_usage_percent: 75.0,
            peak_usage_percent: 90.0,
            cores_used: 4,
            context_switches: 1000,
        };

        let system_metrics = SystemMetrics {
            elapsed_time: Duration::from_secs(1),
            memory_stats,
            cpu_stats,
        };

        assert_eq!(system_metrics.cpu_utilization(), 75.0);
        assert_eq!(system_metrics.memory_efficiency(1000), 5.0); // 1000 ops / 200 MB
    }

    #[test]
    fn test_framework_display() {
        assert_eq!(format!("{}", Framework::Torsh), "ToRSh");
        assert_eq!(format!("{}", Framework::TensorFlow), "TensorFlow");
        assert_eq!(format!("{}", Framework::PyTorch), "PyTorch");
    }

    #[test]
    fn test_operation_type_display() {
        assert_eq!(
            format!("{}", OperationType::MatrixMultiplication),
            "Matrix Multiplication"
        );
        assert_eq!(
            format!("{}", OperationType::Convolution2D),
            "2D Convolution"
        );
        assert_eq!(
            format!("{}", OperationType::ElementWiseAddition),
            "Element-wise Addition"
        );
    }
}
