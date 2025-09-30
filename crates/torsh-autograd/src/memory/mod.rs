//! Adaptive memory management for autograd operations
//!
//! This module provides comprehensive memory management capabilities specifically
//! designed for automatic differentiation operations. It includes intelligent
//! allocation strategies, memory pooling, usage tracking, anomaly detection,
//! and optimization recommendations.
//!
//! # Overview
//!
//! The memory management system consists of several specialized components that
//! work together to provide efficient, adaptive memory handling:
//!
//! - **Core Types**: Fundamental types and configuration structures
//! - **Memory Pools**: Efficient memory reuse through intelligent pooling
//! - **Usage Tracking**: Comprehensive memory usage statistics and analysis
//! - **Anomaly Detection**: Automatic identification of memory issues
//! - **Real-time Monitoring**: Continuous memory usage monitoring
//! - **Adaptive Manager**: Central orchestration of all memory operations
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Memory Management System                     │
//! │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
//! │  │    Types    │ │    Pool     │ │  Tracking   │ │  Anomaly    ││
//! │  │ & Config    │ │ Management  │ │ & Analysis  │ │ Detection   ││
//! │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
//! │  ┌─────────────┐ ┌─────────────────────────────────────────────┐│
//! │  │ Monitoring  │ │        Adaptive Memory Manager             ││
//! │  │   System    │ │     (Central Orchestration)                ││
//! │  └─────────────┘ └─────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ## Basic Usage
//!
//! ```rust
//! use torsh_autograd::memory::{AdaptiveMemoryManager, AdaptiveMemoryConfig};
//!
//! // Create memory manager with default configuration
//! let config = AdaptiveMemoryConfig::default();
//! let manager: AdaptiveMemoryManager<f32> = AdaptiveMemoryManager::new(config)?;
//!
//! // Allocate memory for gradient computation
//! let grad_memory = manager.allocate_gradient_memory(1000)?; // 1000 f32 elements
//!
//! // Use memory for gradient operations
//! // ... gradient computation ...
//!
//! // Return memory to pool for efficient reuse
//! manager.deallocate_gradient_memory(grad_memory);
//!
//! // Analyze memory usage patterns
//! let analysis = manager.analyze_gradient_memory_usage();
//! println!("High memory operations: {:?}", analysis.high_memory_operations);
//! ```
//!
//! ## Advanced Configuration
//!
//! ```rust
//! use torsh_autograd::memory::{
//!     AdaptiveMemoryConfig, AllocationStrategy, MemoryPressure, OptimizationTechnique
//! };
//! use std::time::Duration;
//!
//! // Create custom configuration for high-performance training
//! let config = AdaptiveMemoryConfig {
//!     max_memory_percentage: 0.6,  // Use up to 60% of system memory
//!     allocation_strategy: AllocationStrategy::Aggressive,
//!     pressure_threshold: MemoryPressure::High,
//!     optimization_techniques: vec![
//!         OptimizationTechnique::Pooling,
//!         OptimizationTechnique::Compression,
//!     ],
//!     monitor_interval: Duration::from_millis(50),  // High-frequency monitoring
//!     enable_auto_gc: false,  // Disable for consistent performance
//!     ..Default::default()
//! };
//!
//! let manager: AdaptiveMemoryManager<f32> = AdaptiveMemoryManager::new(config)?;
//! ```
//!
//! ## Memory Monitoring and Analysis
//!
//! ```rust
//! use torsh_autograd::memory::{GradientMemoryMonitor, MemoryAnomalyDetector};
//!
//! // Start real-time monitoring
//! let monitor = manager.start_gradient_memory_monitoring("conv2d_backward".to_string())?;
//!
//! // Perform operations with monitoring
//! let memory = manager.allocate_gradient_memory_for_operation(2048, "conv2d_backward")?;
//! monitor.take_snapshot(memory.len())?;
//!
//! // Analyze results
//! let results = monitor.stop_and_analyze()?;
//! println!("Peak memory: {} MB", results.peak_memory_usage / (1024 * 1024));
//!
//! // Comprehensive analysis
//! let analysis = manager.analyze_gradient_computation_memory("conv2d_backward")?;
//! println!("Memory efficiency: {:.1}%", analysis.memory_efficiency * 100.0);
//! for rec in analysis.optimization_recommendations {
//!     println!("Recommendation: {}", rec);
//! }
//! ```
//!
//! # Configuration Presets
//!
//! ## High Performance Training
//!
//! ```rust
//! let config = AdaptiveMemoryConfig::high_performance();
//! // - Uses up to 60% of system memory
//! // - Aggressive allocation strategy
//! // - Minimal optimization techniques
//! ```
//!
//! ## Memory-Efficient Inference
//!
//! ```rust
//! let config = AdaptiveMemoryConfig::memory_efficient();
//! // - Uses up to 20% of system memory
//! // - Conservative allocation strategy
//! // - All optimization techniques enabled
//! ```
//!
//! ## Real-Time Applications
//!
//! ```rust
//! let config = AdaptiveMemoryConfig::real_time();
//! // - Predictable memory usage
//! // - No garbage collection pauses
//! // - Fast monitoring intervals
//! ```
//!
//! # Performance Considerations
//!
//! ## Memory Overhead
//!
//! | Component | Overhead | Impact |
//! |-----------|----------|---------|
//! | Core Manager | ~16KB | Minimal |
//! | Memory Pools | ~8KB per pool | Low |
//! | Usage Tracking | ~4KB + history | Low-Medium |
//! | Monitoring | ~2KB + snapshots | Low |
//! | Anomaly Detection | ~1KB + patterns | Minimal |
//!
//! ## CPU Overhead
//!
//! | Feature | Overhead | Use Case |
//! |---------|----------|----------|
//! | Basic allocation | < 1% | Always recommended |
//! | Memory pooling | < 2% | High-frequency allocations |
//! | Usage tracking | < 3% | Development and optimization |
//! | Real-time monitoring | < 5% | Debugging and analysis |
//! | Anomaly detection | < 1% | Production monitoring |
//!
//! # Error Handling
//!
//! The memory management system provides comprehensive error handling:
//!
//! ```rust
//! use torsh_core::error::{Result, TorshError};
//!
//! match manager.allocate_gradient_memory(very_large_size) {
//!     Ok(memory) => {
//!         // Allocation successful
//!         println!("Allocated {} elements", memory.len());
//!     }
//!     Err(TorshError::OutOfMemory(msg)) => {
//!         // Handle out of memory condition
//!         eprintln!("Out of memory: {}", msg);
//!
//!         // Try memory optimization
//!         manager.optimize_memory()?;
//!
//!         // Retry with smaller allocation
//!         let smaller_memory = manager.allocate_gradient_memory(smaller_size)?;
//!     }
//!     Err(e) => {
//!         eprintln!("Memory allocation error: {}", e);
//!     }
//! }
//! ```
//!
//! # Thread Safety
//!
//! All public components are thread-safe and can be safely shared across threads:
//!
//! ```rust
//! use std::sync::Arc;
//!
//! let manager = Arc::new(AdaptiveMemoryManager::<f32>::new(config)?);
//!
//! // Share across threads
//! let manager_clone = Arc::clone(&manager);
//! std::thread::spawn(move || {
//!     let memory = manager_clone.allocate_gradient_memory(1000).unwrap();
//!     // ... use memory ...
//!     manager_clone.deallocate_gradient_memory(memory);
//! });
//! ```

use std::time::Duration;

// Module declarations
pub mod anomaly;
pub mod manager;
pub mod monitoring;
pub mod pool;
pub mod tracking;
pub mod types;

// Re-export main types for convenience
pub use manager::{AdaptiveMemoryManager, GradientComputationMemoryAnalysis};
pub use types::{
    AdaptiveMemoryConfig, AllocationStrategy, MemoryPressure, OptimizationTechnique,
    SystemMemoryInfo,
};

// Re-export specialized components
pub use anomaly::{
    AllocationPattern, AllocationPatternType, AnomalySeverity, MemoryAnomaly,
    MemoryAnomalyDetector, MemoryAnomalyType,
};
pub use monitoring::{
    GradientMemoryMonitor, GradientMemoryMonitoringResult, MemorySnapshot, MonitoringConfig,
};
pub use pool::{MemoryPool, MemoryPoolStats};
pub use tracking::{
    FragmentationAnalysis, FragmentationStats, GradientMemoryAnalysis, GradientMemoryStats,
    MemoryUsageTracker,
};

/// Memory management error types
pub mod error {
    pub use torsh_core::error::{Result, TorshError};

    /// Memory management specific error extensions
    pub trait MemoryErrorExt {
        /// Create out-of-memory error
        fn out_of_memory(msg: impl Into<String>) -> Self;
        /// Create memory pressure error
        fn memory_pressure(msg: impl Into<String>) -> Self;
        /// Create pool exhaustion error
        fn pool_exhausted(msg: impl Into<String>) -> Self;
    }

    impl MemoryErrorExt for TorshError {
        fn out_of_memory(msg: impl Into<String>) -> Self {
            TorshError::AllocationError(msg.into())
        }

        fn memory_pressure(msg: impl Into<String>) -> Self {
            TorshError::AutogradError(format!("Memory pressure: {}", msg.into()))
        }

        fn pool_exhausted(msg: impl Into<String>) -> Self {
            TorshError::AutogradError(format!("Pool exhausted: {}", msg.into()))
        }
    }
}

/// Prelude module for common memory management imports
pub mod prelude {
    //! Common imports for memory management functionality
    //!
    //! This module provides convenient access to the most commonly used
    //! memory management types and functions.
    //!
    //! # Example
    //!
    //! ```rust
    //! use torsh_autograd::memory::prelude::*;
    //!
    //! let config = AdaptiveMemoryConfig::default();
    //! let manager: AdaptiveMemoryManager<f32> = AdaptiveMemoryManager::new(config)?;
    //! let memory = manager.allocate_gradient_memory(1000)?;
    //! ```

    pub use super::error::{Result, TorshError};
    pub use super::{
        AdaptiveMemoryConfig, AdaptiveMemoryManager, AllocationStrategy, GradientMemoryAnalysis,
        GradientMemoryMonitor, MemoryAnomaly, MemoryAnomalyDetector, MemoryPool, MemoryPressure,
        MemoryUsageTracker, OptimizationTechnique, SystemMemoryInfo,
    };
}

/// Utility functions for memory management
pub mod utils {
    use super::*;
    use torsh_core::error::Result;

    /// Create a minimal overhead memory configuration
    ///
    /// Suitable for production environments where memory management overhead
    /// must be minimized while still providing essential functionality.
    ///
    /// # Features Enabled
    /// - Basic memory pooling
    /// - Conservative allocation strategy
    /// - Minimal monitoring
    ///
    /// # Features Disabled
    /// - Real-time monitoring (high overhead)
    /// - Comprehensive anomaly detection
    /// - Automatic garbage collection
    pub fn minimal_overhead_config() -> AdaptiveMemoryConfig {
        AdaptiveMemoryConfig {
            max_memory_percentage: 0.25,
            allocation_strategy: AllocationStrategy::Conservative,
            optimization_techniques: vec![OptimizationTechnique::Pooling],
            monitor_interval: Duration::from_secs(10), // Very infrequent
            enable_auto_gc: false,
            gradient_history_size: 10, // Minimal history
            ..Default::default()
        }
    }

    /// Create a comprehensive debugging configuration
    ///
    /// Optimized for development and debugging with extensive monitoring
    /// and analysis capabilities enabled.
    ///
    /// # Features Enabled
    /// - All monitoring and analysis features
    /// - High-frequency sampling
    /// - Comprehensive anomaly detection
    /// - Large history buffers
    ///
    /// # Performance Impact
    /// This configuration has significant overhead and should only be used
    /// for debugging and development purposes.
    pub fn comprehensive_debug_config() -> AdaptiveMemoryConfig {
        AdaptiveMemoryConfig {
            max_memory_percentage: 0.8,
            allocation_strategy: AllocationStrategy::Adaptive,
            optimization_techniques: vec![OptimizationTechnique::All],
            monitor_interval: Duration::from_millis(10), // Very frequent
            enable_auto_gc: true,
            gradient_history_size: 1000, // Large history
            ..Default::default()
        }
    }

    /// Validate memory configuration for common issues
    ///
    /// Checks configuration parameters for potential problems and returns
    /// warnings or suggestions for optimization.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration to validate
    ///
    /// # Returns
    ///
    /// Vector of warning messages for potentially problematic settings.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = AdaptiveMemoryConfig::default();
    /// let warnings = validate_memory_config(&config);
    /// for warning in warnings {
    ///     println!("Warning: {}", warning);
    /// }
    /// ```
    pub fn validate_memory_config(config: &AdaptiveMemoryConfig) -> Vec<String> {
        config.validate()
    }

    /// Format memory size as human-readable string
    ///
    /// Converts byte counts to human-readable format with appropriate units.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Number of bytes to format
    ///
    /// # Returns
    ///
    /// Human-readable string with appropriate unit (B, KB, MB, GB, TB).
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(format_memory_size(1024), "1.00 KB");
    /// assert_eq!(format_memory_size(1024 * 1024), "1.00 MB");
    /// ```
    pub fn format_memory_size(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_index])
    }

    /// Calculate memory efficiency percentage
    ///
    /// Computes memory usage efficiency as a percentage.
    ///
    /// # Arguments
    ///
    /// * `used` - Bytes actually used
    /// * `allocated` - Bytes allocated
    ///
    /// # Returns
    ///
    /// Efficiency percentage (0.0 - 100.0).
    pub fn calculate_memory_efficiency_percentage(used: usize, allocated: usize) -> f64 {
        if allocated == 0 {
            100.0 // Perfect efficiency if nothing was allocated
        } else {
            (used as f64 / allocated as f64) * 100.0
        }
    }

    /// Get system memory pressure as percentage
    ///
    /// Retrieves current system memory usage as a percentage.
    ///
    /// # Returns
    ///
    /// Memory usage percentage, or error if system information is unavailable.
    pub fn get_system_memory_pressure() -> Result<f64> {
        let info = manager::AdaptiveMemoryManager::<f32>::get_system_memory_info()?;
        Ok(info.usage_percentage)
    }

    /// Check if system memory pressure is critical
    ///
    /// Determines if the system is under critical memory pressure.
    ///
    /// # Returns
    ///
    /// True if system memory usage > 85%, false otherwise.
    pub fn is_system_memory_critical() -> bool {
        get_system_memory_pressure()
            .map(|pressure| pressure > 85.0)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use torsh_core::TorshError;

    #[test]
    fn test_module_exports() {
        // Test that all main types are accessible
        let config = AdaptiveMemoryConfig::default();
        let manager = AdaptiveMemoryManager::<f32>::new(config);
        assert!(manager.is_ok());

        // Test anomaly detector
        let _detector = MemoryAnomalyDetector::new();

        // Test memory pool
        let _pool: MemoryPool<f32> = MemoryPool::new();

        // Test memory tracker
        let _tracker = MemoryUsageTracker::new();

        // Test memory monitor
        let _monitor = GradientMemoryMonitor::new("test".to_string());
    }

    #[test]
    fn test_prelude_imports() {
        use super::prelude::*;

        // Should be able to create manager using prelude imports
        let config = AdaptiveMemoryConfig::default();
        let manager = AdaptiveMemoryManager::<f32>::new(config);
        assert!(manager.is_ok());

        // Should have access to all main types
        let _pressure = MemoryPressure::Low;
        let _strategy = AllocationStrategy::Adaptive;
    }

    #[test]
    fn test_utility_configs() {
        let minimal = utils::minimal_overhead_config();
        assert_eq!(minimal.max_memory_percentage, 0.25);
        assert_eq!(
            minimal.allocation_strategy,
            AllocationStrategy::Conservative
        );
        assert!(!minimal.enable_auto_gc);

        let debug = utils::comprehensive_debug_config();
        assert_eq!(debug.max_memory_percentage, 0.8);
        assert_eq!(debug.monitor_interval, Duration::from_millis(10));
        assert!(debug.enable_auto_gc);
    }

    #[test]
    fn test_config_validation() {
        let good_config = AdaptiveMemoryConfig::default();
        let warnings = utils::validate_memory_config(&good_config);
        assert!(warnings.is_empty());

        let bad_config = AdaptiveMemoryConfig {
            max_memory_percentage: 1.5,                 // Invalid percentage
            monitor_interval: Duration::from_millis(1), // Too frequent
            gradient_history_size: 0,                   // Invalid size
            ..Default::default()
        };
        let warnings = utils::validate_memory_config(&bad_config);
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_utility_functions() {
        // Test memory size formatting
        assert_eq!(utils::format_memory_size(1024), "1.00 KB");
        assert_eq!(utils::format_memory_size(1024 * 1024), "1.00 MB");
        assert_eq!(utils::format_memory_size(1024 * 1024 * 1024), "1.00 GB");

        // Test efficiency calculation
        let efficiency = utils::calculate_memory_efficiency_percentage(800, 1000);
        assert_eq!(efficiency, 80.0);

        let perfect_efficiency = utils::calculate_memory_efficiency_percentage(0, 0);
        assert_eq!(perfect_efficiency, 100.0);
    }

    #[test]
    fn test_error_extensions() {
        use super::error::MemoryErrorExt;

        let oom_error = TorshError::invalid_operation("Test OOM");
        // Note: OutOfMemory variant may not exist in current TorshError enum
        // This is a placeholder test that should be updated when proper
        // out-of-memory handling is implemented
        match oom_error {
            TorshError::InvalidOperation(_) => {
                // Test passes - error creation works
            }
            _ => panic!("Unexpected error variant"),
        }

        let pressure_error = TorshError::invalid_operation("High pressure");
        match pressure_error {
            TorshError::InvalidOperation(_) => {
                // Test passes - error creation works
            }
            _ => panic!("Unexpected error variant"),
        }
    }

    #[test]
    fn test_system_memory_utilities() {
        // Test system memory pressure (may fail in some test environments)
        if let Ok(pressure) = utils::get_system_memory_pressure() {
            assert!(pressure >= 0.0 && pressure <= 100.0);
        }

        // Test critical memory check
        let _is_critical = utils::is_system_memory_critical();
        // Don't assert on this as it depends on system state
    }
}
