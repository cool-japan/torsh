//! Performance profiling for autograd operations
//!
//! This module provides comprehensive performance profiling capabilities for
//! automatic differentiation operations, including timing, memory usage,
//! hardware monitoring, bottleneck detection, and computational complexity analysis.
//!
//! # Overview
//!
//! The profiler system consists of several specialized components:
//!
//! - **Core Profiler**: Main `AutogradProfiler` that orchestrates all profiling activities
//! - **Memory Tracking**: Real-time memory usage monitoring and leak detection
//! - **Hardware Monitoring**: CPU, GPU, and memory subsystem utilization tracking
//! - **Performance Analysis**: Automatic bottleneck detection and optimization suggestions
//! - **Complexity Analysis**: Computational complexity classification and scaling predictions
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    AutogradProfiler                             │
//! │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
//! │  │   Memory    │ │  Hardware   │ │ Performance │ │ Complexity  ││
//! │  │   Tracker   │ │   Monitor   │ │  Analyzer   │ │  Analyzer   ││
//! │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
//! └─────────────────────────────────────────────────────────────────┘
//!                                │
//!                                ▼
//!                        AutogradProfile
//!                     (Complete analysis results)
//! ```
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use torsh_autograd::profiler::{AutogradProfiler, ProfilerConfig};
//! use torsh_autograd::context::AutogradContext;
//!
//! // Create profiler with default configuration
//! let config = ProfilerConfig::default();
//! let mut profiler = AutogradProfiler::new(config);
//!
//! // Start profiling session
//! # fn example() -> torsh_core::error::Result<()> {
//! profiler.start_session("training_epoch_1".to_string())?;
//!
//! // Profile individual operations
//! profiler.start_operation("forward_pass".to_string())?;
//! // ... perform forward pass ...
//! profiler.end_operation("forward_pass")?;
//!
//! profiler.start_operation("backward_pass".to_string())?;
//! // ... perform backward pass ...
//! profiler.end_operation("backward_pass")?;
//!
//! // End session and analyze results
//! let profile = profiler.end_session()?;
//! let report = profiler.generate_report(&profile)?;
//! println!("{}", report);
//! # Ok(())
//! # }
//! ```
//!
//! # Advanced Usage
//!
//! ## Custom Configuration
//!
//! ```rust,ignore
//! use std::time::Duration;
//! use torsh_autograd::profiler::ProfilerConfig;
//!
//! let config = ProfilerConfig {
//!     enable_memory_tracking: true,
//!     enable_hardware_monitoring: false,  // Disable for lower overhead
//!     enable_bottleneck_detection: true,
//!     memory_snapshot_interval: Duration::from_millis(50),  // More frequent snapshots
//!     max_operation_profiles: 5000,  // Track more operations
//!     enable_detailed_timing: true,
//!     enable_flops_counting: false,  // Keep disabled due to overhead
//! };
//! ```
//!
//! ## Complexity Analysis
//!
//! ```rust,ignore
//! use torsh_autograd::profiler::complexity::{ComplexityAnalyzer, ComplexityClass};
//! use std::time::Duration;
//!
//! let mut analyzer = ComplexityAnalyzer::new();
//!
//! // Record performance data for different input sizes
//! # fn example() -> torsh_core::error::Result<()> {
//! for size in [100, 200, 400, 800, 1600] {
//!     let time = Duration::from_millis(size as u64 / 10); // Linear scaling
//!     let memory = size * 4; // Linear memory usage
//!     analyzer.record_performance("linear_op", size, time, memory);
//! }
//!
//! // Analyze computational complexity
//! let analysis = analyzer.analyze_complexity("linear_op")?;
//! assert_eq!(analysis.time_complexity, ComplexityClass::Linear);
//! println!("Predicted time for 10K elements: {:?}",
//!          analysis.performance_prediction.time_predictions[0].1);
//! # Ok(())
//! # }
//! ```
//!
//! ## Graph Execution Profiling
//!
//! ```rust,ignore
//! use torsh_autograd::context::AutogradContext;
//!
//! let mut ctx = AutogradContext::new();
//! let mut profiler = AutogradProfiler::new(ProfilerConfig::default());
//!
//! # fn example() -> torsh_core::error::Result<()> {
//! profiler.start_session("neural_network_training".to_string())?;
//!
//! // Profile entire computation graph
//! let result = profiler.profile_graph_execution(
//!     &mut ctx,
//!     "neural_network_forward",
//!     |ctx| {
//!         // Your neural network computation here
//!         // Memory allocation is automatically tracked
//!         Ok(computed_output)
//!     }
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! ## Overhead Levels
//!
//! | Feature | Overhead | Use Case |
//! |---------|----------|----------|
//! | Basic timing | Minimal (~1-2%) | Always recommended |
//! | Memory tracking | Low (~2-5%) | Development and optimization |
//! | Hardware monitoring | Medium (~5-10%) | Performance analysis |
//! | FLOPS counting | High (~10-20%) | Detailed algorithm analysis |
//! | Complexity analysis | Variable | Research and algorithm selection |
//!
//! ## Production Recommendations
//!
//! - Enable basic timing and memory tracking
//! - Disable FLOPS counting unless specifically needed
//! - Use longer snapshot intervals (100-500ms) for reduced overhead
//! - Consider sampling-based profiling for large-scale deployments
//!
//! # Platform Support
//!
//! The profiler provides platform-specific optimizations:
//!
//! - **Linux**: Full hardware monitoring via `/proc` and `sysfs`
//! - **macOS**: Hardware monitoring via system APIs
//! - **Windows**: Basic hardware monitoring via WMI
//! - **Cross-platform**: Memory tracking and timing work on all platforms

// Module declarations
pub mod analysis;
pub mod complexity;
pub mod hardware;
pub mod memory;
pub mod profiler_core;
pub mod types;

// Re-export main types for convenience
pub use profiler_core::{AutogradProfiler, ProfilerConfig};
pub use types::{
    AutogradProfile, BottleneckType, HardwareUtilization, MemorySnapshot, OperationProfile,
    PerformanceBottleneck, ProfileSummary,
};

// Re-export specialized analyzers
pub use analysis::{AnalysisConfig, BottleneckThresholds, PerformanceAnalyzer};
pub use complexity::{
    ComplexityAnalysis, ComplexityAnalyzer, ComplexityClass, PerformancePrediction,
};
pub use hardware::HardwareMonitor;
pub use memory::{MemoryEstimationCache, MemoryStatistics, MemoryTracker};

/// Profiler error types
pub mod error {
    pub use torsh_core::error::{Result, TorshError};

    /// Profiler-specific error extension
    pub trait ProfilerErrorExt {
        /// Create profiler-specific error
        fn profiler_error(msg: impl Into<String>) -> Self;
    }

    impl ProfilerErrorExt for TorshError {
        fn profiler_error(msg: impl Into<String>) -> Self {
            TorshError::AutogradError(format!("Profiler: {}", msg.into()))
        }
    }
}

/// Prelude module for common profiler imports
pub mod prelude {
    //! Common imports for profiler usage
    //!
    //! This module provides convenient access to the most commonly used
    //! profiler types and functions.
    //!
    //! # Example
    //!
    //! ```rust,ignore
    //! use torsh_autograd::profiler::prelude::*;
    //!
    //! let mut profiler = AutogradProfiler::new(ProfilerConfig::default());
    //! # fn example() -> torsh_core::error::Result<()> {
    //! profiler.start_session("my_session".to_string())?;
    //! // ... profiling operations ...
    //! let profile = profiler.end_session()?;
    //! # Ok(())
    //! # }
    //! ```

    pub use super::error::{Result, TorshError};
    pub use super::{
        AutogradProfile, AutogradProfiler, BottleneckType, ComplexityAnalyzer, ComplexityClass,
        HardwareUtilization, MemorySnapshot, OperationProfile, PerformanceAnalyzer,
        PerformanceBottleneck, ProfileSummary, ProfilerConfig,
    };
}

/// Utility functions for profiler configuration and setup
pub mod utils {
    use super::*;
    use std::time::Duration;

    /// Create a minimal overhead profiler configuration
    ///
    /// Suitable for production environments where profiling overhead
    /// must be minimized while still providing useful insights.
    ///
    /// # Features Enabled
    /// - Basic operation timing
    /// - Memory tracking with longer intervals
    /// - Bottleneck detection
    ///
    /// # Features Disabled
    /// - Hardware monitoring (medium overhead)
    /// - FLOPS counting (high overhead)
    /// - Detailed memory snapshots
    pub fn minimal_config() -> ProfilerConfig {
        ProfilerConfig {
            enable_memory_tracking: true,
            enable_hardware_monitoring: false,
            enable_bottleneck_detection: true,
            memory_snapshot_interval: Duration::from_millis(500), // Less frequent
            max_operation_profiles: 500,                          // Reduced capacity
            enable_detailed_timing: true,
            enable_flops_counting: false,
        }
    }

    /// Create a comprehensive profiler configuration
    ///
    /// Suitable for development and optimization phases where detailed
    /// profiling information is more important than overhead.
    ///
    /// # Features Enabled
    /// - All profiling features
    /// - High-frequency memory snapshots
    /// - Hardware monitoring
    /// - Large operation capacity
    ///
    /// # Note
    /// This configuration has significant overhead and should only be used
    /// for detailed performance analysis.
    pub fn comprehensive_config() -> ProfilerConfig {
        ProfilerConfig {
            enable_memory_tracking: true,
            enable_hardware_monitoring: true,
            enable_bottleneck_detection: true,
            memory_snapshot_interval: Duration::from_millis(25), // High frequency
            max_operation_profiles: 10000,                       // Large capacity
            enable_detailed_timing: true,
            enable_flops_counting: false, // Still disabled due to extreme overhead
        }
    }

    /// Create a memory-focused profiler configuration
    ///
    /// Optimized for memory usage analysis and leak detection.
    /// Disables CPU-intensive features while maximizing memory tracking detail.
    pub fn memory_focused_config() -> ProfilerConfig {
        ProfilerConfig {
            enable_memory_tracking: true,
            enable_hardware_monitoring: false,
            enable_bottleneck_detection: true,
            memory_snapshot_interval: Duration::from_millis(10), // Very high frequency
            max_operation_profiles: 2000,
            enable_detailed_timing: true,
            enable_flops_counting: false,
        }
    }

    /// Create a timing-focused profiler configuration
    ///
    /// Optimized for operation timing analysis with minimal memory overhead.
    /// Ideal for performance optimization and bottleneck identification.
    pub fn timing_focused_config() -> ProfilerConfig {
        ProfilerConfig {
            enable_memory_tracking: false,
            enable_hardware_monitoring: true,
            enable_bottleneck_detection: true,
            memory_snapshot_interval: Duration::from_secs(1), // Minimal memory tracking
            max_operation_profiles: 5000,
            enable_detailed_timing: true,
            enable_flops_counting: false,
        }
    }

    /// Validate profiler configuration for common issues
    ///
    /// Checks configuration parameters for potential problems and returns
    /// warnings or suggestions for optimization.
    ///
    /// # Returns
    ///
    /// Vector of warning messages for potentially problematic settings.
    pub fn validate_config(config: &ProfilerConfig) -> Vec<String> {
        let mut warnings = Vec::new();

        if config.enable_flops_counting {
            warnings.push(
                "FLOPS counting enabled - this adds significant overhead (10-20%)".to_string(),
            );
        }

        if config.memory_snapshot_interval < Duration::from_millis(10) {
            warnings
                .push("Memory snapshot interval < 10ms may cause excessive overhead".to_string());
        }

        if config.max_operation_profiles > 50000 {
            warnings
                .push("Large max_operation_profiles may consume significant memory".to_string());
        }

        if config.enable_hardware_monitoring
            && config.memory_snapshot_interval < Duration::from_millis(100)
        {
            warnings.push(
                "Hardware monitoring + frequent memory snapshots may cause high overhead"
                    .to_string(),
            );
        }

        if !config.enable_memory_tracking
            && !config.enable_hardware_monitoring
            && !config.enable_detailed_timing
        {
            warnings.push(
                "All major profiling features disabled - profiler will provide minimal data"
                    .to_string(),
            );
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_module_exports() {
        // Test that all main types are accessible
        let config = ProfilerConfig::default();
        let _profiler = AutogradProfiler::new(config);

        // Test complexity analyzer
        let _complexity_analyzer = ComplexityAnalyzer::new();

        // Test hardware monitor
        let _hardware_monitor = HardwareMonitor::new();

        // Test memory tracker
        let _memory_tracker = MemoryTracker::new(Duration::from_millis(100));

        // Test performance analyzer
        let _performance_analyzer = PerformanceAnalyzer::new();
    }

    #[test]
    fn test_prelude_imports() {
        use super::prelude::*;

        // Should be able to create profiler using prelude imports
        let config = ProfilerConfig::default();
        let _profiler = AutogradProfiler::new(config);

        // Should have access to all main types
        let _complexity_class = ComplexityClass::Linear;
        let _bottleneck_type = BottleneckType::CpuCompute;
    }

    #[test]
    fn test_utility_configs() {
        let minimal = utils::minimal_config();
        assert!(!minimal.enable_hardware_monitoring);
        assert!(!minimal.enable_flops_counting);
        assert_eq!(minimal.memory_snapshot_interval, Duration::from_millis(500));

        let comprehensive = utils::comprehensive_config();
        assert!(comprehensive.enable_hardware_monitoring);
        assert!(comprehensive.enable_memory_tracking);
        assert_eq!(
            comprehensive.memory_snapshot_interval,
            Duration::from_millis(25)
        );

        let memory_focused = utils::memory_focused_config();
        assert!(memory_focused.enable_memory_tracking);
        assert!(!memory_focused.enable_hardware_monitoring);
        assert_eq!(
            memory_focused.memory_snapshot_interval,
            Duration::from_millis(10)
        );

        let timing_focused = utils::timing_focused_config();
        assert!(!timing_focused.enable_memory_tracking);
        assert!(timing_focused.enable_hardware_monitoring);
        assert!(timing_focused.enable_detailed_timing);
    }

    #[test]
    fn test_config_validation() {
        // Test valid config
        let good_config = ProfilerConfig::default();
        let warnings = utils::validate_config(&good_config);
        assert!(warnings.is_empty());

        // Test problematic config
        let bad_config = ProfilerConfig {
            enable_flops_counting: true,
            memory_snapshot_interval: Duration::from_millis(1),
            max_operation_profiles: 100000,
            enable_hardware_monitoring: true,
            ..ProfilerConfig::default()
        };
        let warnings = utils::validate_config(&bad_config);
        assert!(!warnings.is_empty());
        assert!(warnings.iter().any(|w| w.contains("FLOPS counting")));
        assert!(warnings.iter().any(|w| w.contains("snapshot interval")));
        assert!(warnings
            .iter()
            .any(|w| w.contains("max_operation_profiles")));

        // Test minimal config
        let minimal_config = ProfilerConfig {
            enable_memory_tracking: false,
            enable_hardware_monitoring: false,
            enable_detailed_timing: false,
            ..ProfilerConfig::default()
        };
        let warnings = utils::validate_config(&minimal_config);
        assert!(warnings.iter().any(|w| w.contains("minimal data")));
    }

    #[test]
    fn test_error_extension() {
        use super::error::{ProfilerErrorExt, TorshError};

        let error = TorshError::profiler_error("test error message");
        match error {
            TorshError::AutogradError(msg) => {
                assert!(msg.contains("Profiler: test error message"));
            }
            _ => panic!("Expected AutogradError"),
        }
    }
}
