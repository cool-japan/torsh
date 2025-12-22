//! Core types and configuration for adaptive memory management
//!
//! This module defines the fundamental types, enums, and configuration structures
//! used throughout the adaptive memory management system. It provides the building
//! blocks for memory pressure detection, allocation strategies, and optimization
//! techniques.
//!
//! # Overview
//!
//! The memory management system uses several key concepts:
//!
//! - **Memory Pressure**: Current system memory utilization level
//! - **Allocation Strategy**: How to allocate memory based on conditions
//! - **Optimization Techniques**: Methods to reduce memory usage
//! - **System Memory Info**: Real-time system memory statistics
//!
//! # Examples
//!
//! ```rust,ignore
//! use crate::memory::types::{AdaptiveMemoryConfig, MemoryPressure, AllocationStrategy};
//! use std::time::Duration;
//!
//! let config = AdaptiveMemoryConfig {
//!     max_memory_percentage: 0.4, // Use up to 40% of system memory
//!     pressure_threshold: MemoryPressure::High,
//!     allocation_strategy: AllocationStrategy::Conservative,
//!     ..AdaptiveMemoryConfig::default()
//! };
//! ```

use std::time::{Duration, Instant};

/// System memory information
///
/// Provides comprehensive information about the current state of system memory,
/// including total, available, and used memory amounts, along with usage percentage
/// and timestamp information for tracking memory changes over time.
///
/// # Platform Support
///
/// This structure is populated differently on each platform:
/// - **Linux**: Uses `/proc/meminfo` for accurate system-wide memory information
/// - **macOS**: Uses `vm_stat` and system APIs for memory statistics
/// - **Windows**: Uses Windows Memory Management APIs
///
/// # Examples
/// use std::time::Duration;
///
/// ```rust,ignore
/// let info = SystemMemoryInfo {
///     total_memory: 16 * 1024 * 1024 * 1024, // 16GB
///     available_memory: 8 * 1024 * 1024 * 1024, // 8GB available
///     used_memory: 8 * 1024 * 1024 * 1024, // 8GB used
///     usage_percentage: 50.0,
///     last_updated: Instant::now(),
/// };
/// println!("Memory usage: {:.1}%", info.usage_percentage);
/// ```
#[derive(Debug, Clone)]
pub struct SystemMemoryInfo {
    /// Total system memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Used memory in bytes
    pub used_memory: usize,
    /// Memory usage percentage (0.0 - 100.0)
    pub usage_percentage: f64,
    /// Last update timestamp
    pub last_updated: Instant,
}

/// Memory pressure levels
///
/// Indicates the current memory pressure level of the system, which is used
/// to determine appropriate memory allocation strategies and trigger optimization
/// techniques when memory becomes scarce.
///
/// # Pressure Levels
///
/// - **Low**: < 50% memory usage - Normal operation, aggressive allocation allowed
/// - **Moderate**: 50-70% usage - Start using conservative strategies
/// - **High**: 70-85% usage - Enable memory optimizations, reduce allocations
/// - **Critical**: > 85% usage - Emergency mode, minimal allocations only
///
/// # Examples
/// use std::time::Duration;
///
/// ```rust,ignore
/// match memory_pressure {
///     MemoryPressure::Low => println!("Memory usage is comfortable"),
///     MemoryPressure::Moderate => println!("Starting to use more memory"),
///     MemoryPressure::High => println!("Memory pressure is increasing"),
///     MemoryPressure::Critical => println!("Critical memory situation!"),
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    /// Low memory pressure (< 50% usage)
    Low,
    /// Moderate memory pressure (50-70% usage)
    Moderate,
    /// High memory pressure (70-85% usage)
    High,
    /// Critical memory pressure (> 85% usage)
    Critical,
}

impl Default for MemoryPressure {
    fn default() -> Self {
        MemoryPressure::Low
    }
}

/// Memory allocation strategy based on system conditions
///
/// Defines how memory should be allocated based on current system conditions,
/// memory pressure, and performance requirements. Different strategies optimize
/// for different scenarios.
///
/// # Strategy Types
///
/// - **Aggressive**: Maximum performance, large allocations, minimal overhead
/// - **Conservative**: Balanced approach, moderate allocations with safety margins
/// - **Minimal**: Memory-constrained mode, smallest possible allocations
/// - **Adaptive**: Dynamically adapts strategy based on current conditions
///
/// # Use Cases
///
/// ```rust,ignore
/// let strategy = match memory_pressure {
///     MemoryPressure::Low => AllocationStrategy::Aggressive,
///     MemoryPressure::Moderate => AllocationStrategy::Conservative,
///     MemoryPressure::High => AllocationStrategy::Minimal,
///     MemoryPressure::Critical => AllocationStrategy::Minimal,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Aggressive allocation for maximum performance
    Aggressive,
    /// Conservative allocation to preserve memory
    Conservative,
    /// Minimal allocation for memory-constrained environments
    Minimal,
    /// Adaptive allocation based on current conditions
    Adaptive,
}

/// Memory optimization techniques
///
/// Specifies which memory optimization techniques should be applied to reduce
/// memory usage and improve efficiency. Multiple techniques can be combined
/// for maximum effectiveness.
///
/// # Optimization Techniques
///
/// - **Compression**: Compress gradients and intermediate values
/// - **Pooling**: Reuse memory allocations through pooling
/// - **LazyAllocation**: Delay allocation until actually needed
/// - **MemoryMapping**: Use memory-mapped files for large data
/// - **All**: Enable all available optimization techniques
///
/// # Performance Impact
///
/// | Technique | Memory Savings | CPU Overhead | Use Case |
/// |-----------|----------------|--------------|----------|
/// | Compression | High (50-80%) | Medium | Large gradients |
/// | Pooling | Medium (20-40%) | Low | Frequent allocations |
/// | LazyAllocation | Variable | Very Low | Uncertain usage patterns |
/// | MemoryMapping | High | Low | Very large datasets |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationTechnique {
    /// No optimization
    None,
    /// Gradient compression
    Compression,
    /// Memory pooling and reuse
    Pooling,
    /// Lazy allocation
    LazyAllocation,
    /// Memory mapped files
    MemoryMapping,
    /// All techniques combined
    All,
}

/// Configuration for adaptive memory management
///
/// Comprehensive configuration structure that controls all aspects of the
/// adaptive memory management system, including memory budgets, monitoring
/// intervals, allocation strategies, and optimization techniques.
///
/// # Key Configuration Parameters
///
/// - **Memory Budget**: Maximum percentage of system memory to use
/// - **Pressure Threshold**: When to start applying memory optimizations
/// - **Monitoring**: How frequently to check memory status
/// - **Optimization**: Which techniques to apply and when
///
/// # Performance Tuning
///
/// For different workload characteristics:
///
/// **High-throughput training**:
/// ```rust,ignore
/// AdaptiveMemoryConfig {
///     max_memory_percentage: 0.8,
///     allocation_strategy: AllocationStrategy::Aggressive,
///     optimization_techniques: vec![OptimizationTechnique::Pooling],
///     ..Default::default()
/// }
/// ```
///
/// **Memory-constrained inference**:
/// ```rust,ignore
/// AdaptiveMemoryConfig {
///     max_memory_percentage: 0.3,
///     allocation_strategy: AllocationStrategy::Conservative,
///     optimization_techniques: vec![
///         OptimizationTechnique::Compression,
///         OptimizationTechnique::LazyAllocation,
///     ],
///     ..Default::default()
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveMemoryConfig {
    /// Maximum memory budget as percentage of available system memory (0.0 - 1.0)
    pub max_memory_percentage: f64,
    /// Memory pressure threshold for triggering optimization
    pub pressure_threshold: MemoryPressure,
    /// Update interval for memory monitoring
    pub monitor_interval: Duration,
    /// Allocation strategy
    pub allocation_strategy: AllocationStrategy,
    /// Optimization techniques to use
    pub optimization_techniques: Vec<OptimizationTechnique>,
    /// Enable automatic garbage collection
    pub enable_auto_gc: bool,
    /// Gradient history size for memory estimation
    pub gradient_history_size: usize,
}

impl Default for AdaptiveMemoryConfig {
    /// Create a balanced default configuration
    ///
    /// The default configuration provides a good balance between performance
    /// and memory efficiency for most use cases:
    ///
    /// - Uses up to 30% of system memory
    /// - Triggers optimizations at high memory pressure
    /// - Monitors memory every 100ms
    /// - Uses adaptive allocation strategy
    /// - Enables compression, pooling, and lazy allocation
    /// - Automatic garbage collection enabled
    /// - Tracks last 100 gradient allocations for prediction
    fn default() -> Self {
        Self {
            max_memory_percentage: 0.3, // Use up to 30% of system memory
            pressure_threshold: MemoryPressure::High,
            monitor_interval: Duration::from_millis(100),
            allocation_strategy: AllocationStrategy::Adaptive,
            optimization_techniques: vec![
                OptimizationTechnique::Compression,
                OptimizationTechnique::Pooling,
                OptimizationTechnique::LazyAllocation,
            ],
            enable_auto_gc: true,
            gradient_history_size: 100,
        }
    }
}

impl AdaptiveMemoryConfig {
    /// Create a high-performance configuration
    ///
    /// Optimized for maximum performance with higher memory usage:
    /// - Uses up to 60% of system memory
    /// - Only triggers optimizations at critical pressure
    /// - Aggressive allocation strategy
    /// - Minimal optimization techniques
    pub fn high_performance() -> Self {
        Self {
            max_memory_percentage: 0.6,
            pressure_threshold: MemoryPressure::Critical,
            allocation_strategy: AllocationStrategy::Aggressive,
            optimization_techniques: vec![OptimizationTechnique::Pooling],
            enable_auto_gc: false, // Disable GC for consistent performance
            ..Default::default()
        }
    }

    /// Create a memory-efficient configuration
    ///
    /// Optimized for minimal memory usage:
    /// - Uses up to 20% of system memory
    /// - Triggers optimizations at moderate pressure
    /// - Conservative allocation strategy
    /// - All optimization techniques enabled
    pub fn memory_efficient() -> Self {
        Self {
            max_memory_percentage: 0.2,
            pressure_threshold: MemoryPressure::Moderate,
            allocation_strategy: AllocationStrategy::Conservative,
            optimization_techniques: vec![OptimizationTechnique::All],
            enable_auto_gc: true,
            gradient_history_size: 50, // Smaller history to save memory
            ..Default::default()
        }
    }

    /// Create a real-time configuration
    ///
    /// Optimized for real-time applications with predictable performance:
    /// - Uses up to 40% of system memory
    /// - Triggers optimizations at high pressure
    /// - Minimal allocation strategy for consistency
    /// - Pooling only to avoid compression overhead
    pub fn real_time() -> Self {
        Self {
            max_memory_percentage: 0.4,
            pressure_threshold: MemoryPressure::High,
            monitor_interval: Duration::from_millis(50), // More frequent monitoring
            allocation_strategy: AllocationStrategy::Minimal,
            optimization_techniques: vec![OptimizationTechnique::Pooling],
            enable_auto_gc: false,     // Avoid GC pauses
            gradient_history_size: 25, // Smaller history for faster prediction
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    ///
    /// Checks that all configuration parameters are within valid ranges
    /// and returns any validation warnings or errors.
    ///
    /// # Returns
    ///
    /// Vector of warning messages for potentially problematic settings.
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.max_memory_percentage <= 0.0 || self.max_memory_percentage > 1.0 {
            warnings.push("max_memory_percentage must be between 0.0 and 1.0".to_string());
        }

        if self.max_memory_percentage > 0.8 {
            warnings.push("max_memory_percentage > 80% may cause system instability".to_string());
        }

        if self.monitor_interval < Duration::from_millis(10) {
            warnings.push("monitor_interval < 10ms may cause excessive overhead".to_string());
        }

        if self.gradient_history_size == 0 {
            warnings.push("gradient_history_size of 0 disables memory prediction".to_string());
        }

        if self.gradient_history_size > 1000 {
            warnings.push("Large gradient_history_size may consume significant memory".to_string());
        }

        if self.optimization_techniques.is_empty() {
            warnings.push(
                "No optimization techniques enabled - memory usage may be suboptimal".to_string(),
            );
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pressure_ordering() {
        assert!(MemoryPressure::Low < MemoryPressure::Moderate);
        assert!(MemoryPressure::Moderate < MemoryPressure::High);
        assert!(MemoryPressure::High < MemoryPressure::Critical);
    }

    #[test]
    fn test_default_config() {
        let config = AdaptiveMemoryConfig::default();
        assert_eq!(config.max_memory_percentage, 0.3);
        assert_eq!(config.pressure_threshold, MemoryPressure::High);
        assert_eq!(config.monitor_interval, Duration::from_millis(100));
        assert_eq!(config.allocation_strategy, AllocationStrategy::Adaptive);
        assert!(config.enable_auto_gc);
        assert_eq!(config.gradient_history_size, 100);
        assert!(!config.optimization_techniques.is_empty());
    }

    #[test]
    fn test_preset_configs() {
        let high_perf = AdaptiveMemoryConfig::high_performance();
        assert_eq!(high_perf.max_memory_percentage, 0.6);
        assert_eq!(
            high_perf.allocation_strategy,
            AllocationStrategy::Aggressive
        );
        assert!(!high_perf.enable_auto_gc);

        let memory_eff = AdaptiveMemoryConfig::memory_efficient();
        assert_eq!(memory_eff.max_memory_percentage, 0.2);
        assert_eq!(
            memory_eff.allocation_strategy,
            AllocationStrategy::Conservative
        );
        assert_eq!(memory_eff.gradient_history_size, 50);

        let real_time = AdaptiveMemoryConfig::real_time();
        assert_eq!(real_time.monitor_interval, Duration::from_millis(50));
        assert_eq!(real_time.allocation_strategy, AllocationStrategy::Minimal);
        assert!(!real_time.enable_auto_gc);
    }

    #[test]
    fn test_config_validation() {
        let mut config = AdaptiveMemoryConfig::default();
        assert!(config.validate().is_empty());

        // Test invalid percentage
        config.max_memory_percentage = 1.5;
        let warnings = config.validate();
        assert!(warnings.iter().any(|w| w.contains("between 0.0 and 1.0")));

        // Test high percentage warning
        config.max_memory_percentage = 0.9;
        let warnings = config.validate();
        assert!(warnings.iter().any(|w| w.contains("80%")));

        // Test short monitor interval
        config.max_memory_percentage = 0.3;
        config.monitor_interval = Duration::from_millis(5);
        let warnings = config.validate();
        assert!(warnings.iter().any(|w| w.contains("10ms")));
    }

    #[test]
    fn test_system_memory_info() {
        let info = SystemMemoryInfo {
            total_memory: 1000,
            available_memory: 600,
            used_memory: 400,
            usage_percentage: 40.0,
            last_updated: Instant::now(),
        };

        assert_eq!(info.total_memory, 1000);
        assert_eq!(info.available_memory, 600);
        assert_eq!(info.used_memory, 400);
        assert_eq!(info.usage_percentage, 40.0);
    }

    #[test]
    fn test_optimization_technique_equality() {
        assert_eq!(
            OptimizationTechnique::Compression,
            OptimizationTechnique::Compression
        );
        assert_ne!(
            OptimizationTechnique::Compression,
            OptimizationTechnique::Pooling
        );
    }

    #[test]
    fn test_allocation_strategy_equality() {
        assert_eq!(
            AllocationStrategy::Aggressive,
            AllocationStrategy::Aggressive
        );
        assert_ne!(
            AllocationStrategy::Aggressive,
            AllocationStrategy::Conservative
        );
    }
}
