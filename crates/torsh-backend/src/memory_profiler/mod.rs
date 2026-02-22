//! Comprehensive memory profiling system with SciRS2 integration
//!
//! This module provides a complete memory profiling solution with the following capabilities:
//!
//! ## Core Features
//! - **Allocation Tracking**: Detailed tracking of memory allocations with source information
//! - **Pressure Monitoring**: Real-time memory pressure detection and mitigation
//! - **Pattern Analysis**: Advanced access pattern recognition and optimization
//! - **Fragmentation Management**: Comprehensive fragmentation tracking and mitigation
//! - **SciRS2 Integration**: Deep integration with the SciRS2 ecosystem
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use torsh_backend::memory_profiler::{MemoryProfiler, MemoryProfilerConfig};
//! use torsh_backend::profiler::Profiler;
//!
//! // Create a memory profiler with default configuration
//! let config = MemoryProfilerConfig::default();
//! let base_profiler: Box<dyn Profiler + Send + Sync> = todo!(); // Your profiler implementation
//! let profiler = MemoryProfiler::new(base_profiler, config);
//!
//! // The profiler automatically tracks allocations, pressure, and patterns
//! ```
//!
//! ## Module Organization
//!
//! - [`core`]: Core memory profiler infrastructure
//! - [`allocation`]: Memory allocation tracking and management
//! - [`pressure`]: Memory pressure monitoring and mitigation
//! - [`patterns`]: Access pattern analysis and optimization
//! - [`fragmentation`]: Fragmentation tracking and defragmentation
//! - [`scirs2`]: SciRS2 ecosystem integration
//!
//! ## Collections by Use Case
//!
//! For convenience, common types are grouped by use case:
//!
//! - [`collections::monitoring`]: Types for basic memory monitoring
//! - [`collections::profiling`]: Types for advanced memory profiling
//! - [`collections::optimization`]: Types for performance optimization
//! - [`collections::scirs2`]: Types for SciRS2 integration

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use std::time::{Duration, Instant};

// Module declarations
pub mod allocation;
pub mod core;
pub mod fragmentation;
pub mod patterns;
pub mod pressure;
pub mod scirs2;

// Re-export all public types for backward compatibility
pub use core::{GlobalMemoryStats, MemoryProfiler, MemoryProfilerConfig};

pub use allocation::{
    AccessPattern,

    AccessType,
    AllocationContext,
    AllocationSource,
    AllocationTracker,

    AllocationUsageStats,
    // Cache statistics
    CacheStats,

    HintSeverity,

    // Lifetime tracking
    LifetimeEvent,
    LifetimeEventType,

    // Core allocation types
    MemoryAllocation,
    // Memory and access types
    MemoryType,
    // Performance hints
    PerformanceHint,
    PerformanceHintType,
    // Pressure level (shared across modules)
    PressureLevel,
};

pub use pressure::{
    BandwidthUtilization,

    // Usage statistics
    DeviceMemoryUsage,
    GlobalPressureStats,
    HostMemoryUsage,
    // Pressure monitoring
    MemoryPressureEvent,
    MemoryPressureIndicators,
    MemoryPressureMonitor,
    MemorySnapshot,

    PressureAction,
    // Configuration and statistics
    PressureThresholds,
};

pub use patterns::{
    AccessDirection,
    AccessDistribution,

    // Pattern analysis
    AccessPatternAnalyzer,
    // Predictions
    AccessPrediction,
    CacheBehaviorPrediction,
    CacheLevel,

    CacheWarmingRecommendation,
    OptimizationComplexity,
    OptimizationTimeline,

    OptimizationType,
    // Configuration and statistics
    PatternAnalysisConfig,
    PatternClassification,
    // Optimization suggestions
    PatternOptimizationSuggestion,
    PatternStatistics,
    PatternType,
    PredictedAccess,
};

pub use fragmentation::{
    ActionTimeline,
    // Advanced metrics
    AdvancedFragmentationMetrics,
    // Compaction and defragmentation
    CompactionStats,
    FragmentationCause,
    FragmentationConfig,

    FragmentationEvent,
    // Analysis and prediction
    FragmentationImpact,
    FragmentationPredictionModel,
    FragmentationRecovery,

    FragmentationRisk,

    // Fragmentation tracking
    FragmentationTracker,
    FutureImpactPrediction,
};

// NOTE: scirs2 meta crate removed - these types may need to be imported from specific sub-crates if needed
// TODO: If these types are required, import from the appropriate scirs2-* sub-crate
/*
pub use scirs2::{
    // Advanced features
    AllocatorAdvancedMetrics,
    IntegrationStatus,

    PoolAdvancedAnalytics,
    PoolHealthAssessment,
    // Statistics and monitoring
    ScirS2AllocatorStats,
    ScirS2Event,

    // SciRS2 integration
    ScirS2Integration,
    ScirS2IntegrationConfig,
    ScirS2PoolInfo,
};
*/

/// Prelude module for convenient imports
///
/// Import this module to get access to the most commonly used types:
///
/// ```rust
/// use torsh_backend::memory_profiler::prelude::*;
/// ```
pub mod prelude {
    pub use super::{
        // SciRS2 integration - imported from scirs2 submodule
        scirs2::ScirS2Event,
        scirs2::ScirS2Integration,
        // Pattern analysis
        AccessPatternAnalyzer,
        AccessType,

        AllocationTracker,
        FragmentationRisk,

        // Fragmentation tracking
        FragmentationTracker,
        HintSeverity,
        // Essential allocation types
        MemoryAllocation,
        MemoryPressureEvent,
        // Pressure monitoring
        MemoryPressureMonitor,
        // Core profiler
        MemoryProfiler,
        MemoryProfilerConfig,

        MemoryType,
        PatternType,

        // Performance optimization
        PerformanceHint,
        PerformanceHintType,
        PressureLevel,
    };
}

/// Collections of types organized by use case
pub mod collections {
    /// Types for basic memory monitoring
    pub mod monitoring {
        pub use crate::memory_profiler::{
            BandwidthUtilization, DeviceMemoryUsage, GlobalMemoryStats, HostMemoryUsage,
            MemoryPressureEvent, MemoryProfiler, MemorySnapshot,
        };
    }

    /// Types for advanced memory profiling
    pub mod profiling {
        pub use crate::memory_profiler::{
            AccessPattern, AllocationContext, AllocationSource, AllocationTracker,
            AllocationUsageStats, CacheStats, LifetimeEvent, LifetimeEventType, MemoryAllocation,
        };
    }

    /// Types for performance optimization
    pub mod optimization {
        pub use crate::memory_profiler::{
            AccessPatternAnalyzer, CompactionStats, FragmentationRecovery, FragmentationTracker,
            HintSeverity, OptimizationType, PatternOptimizationSuggestion, PerformanceHint,
            PerformanceHintType,
        };
    }

    /// Types for SciRS2 integration
    pub mod scirs2 {
        pub use crate::memory_profiler::scirs2::{
            AllocatorAdvancedMetrics, IntegrationStatus, PoolAdvancedAnalytics,
            ScirS2AllocatorStats, ScirS2Event, ScirS2Integration, ScirS2IntegrationConfig,
            ScirS2PoolInfo,
        };
    }

    /// Types for fragmentation analysis
    pub mod fragmentation {
        pub use crate::memory_profiler::{
            AdvancedFragmentationMetrics, CompactionStats, FragmentationCause, FragmentationEvent,
            FragmentationImpact, FragmentationRecovery, FragmentationRisk, FragmentationTracker,
        };
    }

    /// Types for pattern analysis
    pub mod patterns {
        pub use crate::memory_profiler::{
            AccessPatternAnalyzer, AccessPrediction, CacheBehaviorPrediction,
            PatternClassification, PatternOptimizationSuggestion, PatternStatistics, PatternType,
            PredictedAccess,
        };
    }
}

/// Factory functions for creating common configurations
pub mod factory {
    use super::scirs2::ScirS2IntegrationConfig;
    use super::*;
    use std::time::Duration;

    /// Create a default memory profiler configuration
    pub fn default_config() -> MemoryProfilerConfig {
        MemoryProfilerConfig::default()
    }

    /// Create a high-performance memory profiler configuration
    pub fn high_performance_config() -> MemoryProfilerConfig {
        MemoryProfilerConfig {
            enable_allocation_tracking: true,
            enable_access_pattern_analysis: true,
            enable_pressure_monitoring: true,
            enable_fragmentation_tracking: true,
            enable_scirs2_integration: true,
            max_tracked_allocations: 1000000,
            snapshot_interval: Duration::from_secs(1),
            access_pattern_window: Duration::from_secs(30),
            hint_threshold: 0.05,
            enable_stack_traces: false,
            memory_pressure_threshold: 0.90,
            fragmentation_alert_threshold: 0.25,
        }
    }

    /// Create a low-overhead memory profiler configuration
    pub fn low_overhead_config() -> MemoryProfilerConfig {
        MemoryProfilerConfig {
            enable_allocation_tracking: true,
            enable_access_pattern_analysis: false,
            enable_pressure_monitoring: true,
            enable_fragmentation_tracking: false,
            enable_scirs2_integration: true,
            max_tracked_allocations: 10000,
            snapshot_interval: Duration::from_secs(60),
            access_pattern_window: Duration::from_secs(300),
            hint_threshold: 0.2,
            enable_stack_traces: false,
            memory_pressure_threshold: 0.80,
            fragmentation_alert_threshold: 0.4,
        }
    }

    /// Create a debug memory profiler configuration
    pub fn debug_config() -> MemoryProfilerConfig {
        MemoryProfilerConfig {
            enable_allocation_tracking: true,
            enable_access_pattern_analysis: true,
            enable_pressure_monitoring: true,
            enable_fragmentation_tracking: true,
            enable_scirs2_integration: true,
            max_tracked_allocations: 100000,
            snapshot_interval: Duration::from_secs(5),
            access_pattern_window: Duration::from_secs(60),
            hint_threshold: 0.01,
            enable_stack_traces: true,
            memory_pressure_threshold: 0.75,
            fragmentation_alert_threshold: 0.2,
        }
    }

    /// Create a default pattern analysis configuration
    pub fn default_pattern_config() -> PatternAnalysisConfig {
        PatternAnalysisConfig::default()
    }

    /// Create a high-sensitivity pattern analysis configuration
    pub fn sensitive_pattern_config() -> PatternAnalysisConfig {
        PatternAnalysisConfig {
            min_pattern_length: 3,
            analysis_window: Duration::from_secs(30),
            confidence_threshold: 0.3,
            enable_prediction: true,
            enable_optimization_suggestions: true,
            max_tracked_patterns: 50000,
            classification_sensitivity: 0.05,
            cache_analysis_depth: 5,
        }
    }

    /// Create a default fragmentation configuration
    pub fn default_fragmentation_config() -> FragmentationConfig {
        FragmentationConfig::default()
    }

    /// Create an aggressive fragmentation configuration
    pub fn aggressive_fragmentation_config() -> FragmentationConfig {
        FragmentationConfig {
            alert_threshold: 0.2,
            critical_threshold: 0.5,
            compaction_threshold: 0.3,
            auto_compaction: true,
            max_compaction_frequency: 20,
            enable_prediction: true,
            metrics_interval: Duration::from_secs(30),
            history_retention: Duration::from_secs(14 * 24 * 60 * 60),
        }
    }

    /// Create a default SciRS2 integration configuration
    pub fn default_scirs2_config() -> ScirS2IntegrationConfig {
        ScirS2IntegrationConfig::default()
    }

    /// Create a comprehensive SciRS2 integration configuration
    pub fn comprehensive_scirs2_config() -> ScirS2IntegrationConfig {
        use scirs2::{AdvancedIntegrationConfig, ProfilingDetailLevel};

        ScirS2IntegrationConfig {
            enable_realtime_sync: true,
            sync_interval: Duration::from_secs(1),
            enable_event_callbacks: true,
            track_allocation_patterns: true,
            enable_optimization_suggestions: true,
            advanced_config: AdvancedIntegrationConfig {
                enable_predictive_modeling: true,
                model_update_frequency: Duration::from_secs(30),
                enable_automated_optimization: true,
                optimization_aggressiveness: 0.7,
                enable_health_monitoring: true,
                health_check_interval: Duration::from_secs(15),
                enable_performance_profiling: true,
                profiling_detail_level: ProfilingDetailLevel::Comprehensive,
            },
        }
    }

    /// Create a default pressure monitoring configuration
    pub fn default_pressure_config() -> PressureThresholds {
        PressureThresholds::default()
    }

    /// Create a sensitive pressure monitoring configuration
    pub fn sensitive_pressure_config() -> PressureThresholds {
        PressureThresholds {
            low_pressure: 50.0,
            medium_pressure: 65.0,
            high_pressure: 75.0,
            critical_pressure: 85.0,
            bandwidth_warning: 70.0,
            allocation_failure_threshold: 0.02,
            page_fault_threshold: 500.0,
        }
    }
}

/// Utility functions for working with memory profiler data
pub mod utils {
    use super::*;
    use std::time::{Duration, Instant};

    /// Calculate memory efficiency from allocation statistics
    pub fn calculate_memory_efficiency(total_allocated: usize, actually_used: usize) -> f64 {
        if total_allocated == 0 {
            1.0
        } else {
            actually_used as f64 / total_allocated as f64
        }
    }

    /// Calculate fragmentation score from free block distribution
    pub fn calculate_fragmentation_score(
        largest_free_block: usize,
        total_free_memory: usize,
        free_block_count: usize,
    ) -> f64 {
        if total_free_memory == 0 || free_block_count == 0 {
            return 0.0;
        }

        let size_fragmentation = 1.0 - (largest_free_block as f64 / total_free_memory as f64);
        let count_fragmentation = (free_block_count as f64).log2() / 20.0;

        ((size_fragmentation * 0.7) + (count_fragmentation * 0.3)).min(1.0)
    }

    /// Calculate allocation rate from allocation tracker
    pub fn calculate_allocation_rate(tracker: &AllocationTracker, time_window: Duration) -> f64 {
        // This would require access to allocation timestamps
        // In a real implementation, this would analyze the allocation history
        tracker.total_memory_usage() as f64 / time_window.as_secs_f64()
    }

    /// Convert pressure level to numeric score
    pub fn pressure_level_to_score(level: PressureLevel) -> f64 {
        match level {
            PressureLevel::None => 0.0,
            PressureLevel::Low => 0.25,
            PressureLevel::Medium => 0.5,
            PressureLevel::High => 0.75,
            PressureLevel::Critical => 1.0,
        }
    }

    /// Convert numeric score to pressure level
    pub fn score_to_pressure_level(score: f64) -> PressureLevel {
        if score >= 0.9 {
            PressureLevel::Critical
        } else if score >= 0.7 {
            PressureLevel::High
        } else if score >= 0.5 {
            PressureLevel::Medium
        } else if score >= 0.2 {
            PressureLevel::Low
        } else {
            PressureLevel::None
        }
    }

    /// Format memory size for human-readable output
    pub fn format_memory_size(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        const THRESHOLD: f64 = 1024.0;

        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= THRESHOLD && unit_index < UNITS.len() - 1 {
            size /= THRESHOLD;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", size as usize, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }

    /// Format duration for human-readable output
    pub fn format_duration(duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        let days = total_seconds / 86400;
        let hours = (total_seconds % 86400) / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;

        if days > 0 {
            format!("{}d {}h {}m {}s", days, hours, minutes, seconds)
        } else if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}s", seconds)
        } else {
            format!("{}ms", duration.as_millis())
        }
    }

    /// Calculate age from timestamp
    pub fn calculate_age(timestamp: Instant) -> Duration {
        Instant::now().duration_since(timestamp)
    }

    /// Check if timestamp is recent
    pub fn is_recent(timestamp: Instant, threshold: Duration) -> bool {
        calculate_age(timestamp) <= threshold
    }

    /// Create a summary of memory profiler state
    pub fn create_profiler_summary(_profiler: &MemoryProfiler) -> ProfilerSummary {
        ProfilerSummary {
            total_allocations: 0, // Would need access to internal state
            current_memory_usage: format_memory_size(0),
            peak_memory_usage: format_memory_size(0),
            memory_efficiency: 0.0,
            fragmentation_level: 0.0,
            pressure_level: PressureLevel::None,
            optimization_opportunities: 0,
            last_update: Instant::now(),
        }
    }
}

/// Summary information about memory profiler state
#[derive(Debug, Clone)]
pub struct ProfilerSummary {
    /// Total number of allocations tracked
    pub total_allocations: u64,

    /// Current memory usage (formatted)
    pub current_memory_usage: String,

    /// Peak memory usage (formatted)
    pub peak_memory_usage: String,

    /// Overall memory efficiency
    pub memory_efficiency: f64,

    /// Current fragmentation level
    pub fragmentation_level: f64,

    /// Current pressure level
    pub pressure_level: PressureLevel,

    /// Number of optimization opportunities identified
    pub optimization_opportunities: usize,

    /// Last update timestamp
    pub last_update: Instant,
}

/// Memory profiler builder for convenient construction
pub struct MemoryProfilerBuilder {
    config: MemoryProfilerConfig,
}

impl MemoryProfilerBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: MemoryProfilerConfig::default(),
        }
    }

    /// Enable or disable allocation tracking
    pub fn allocation_tracking(mut self, enabled: bool) -> Self {
        self.config.enable_allocation_tracking = enabled;
        self
    }

    /// Enable or disable access pattern analysis
    pub fn pattern_analysis(mut self, enabled: bool) -> Self {
        self.config.enable_access_pattern_analysis = enabled;
        self
    }

    /// Enable or disable pressure monitoring
    pub fn pressure_monitoring(mut self, enabled: bool) -> Self {
        self.config.enable_pressure_monitoring = enabled;
        self
    }

    /// Enable or disable fragmentation tracking
    pub fn fragmentation_tracking(mut self, enabled: bool) -> Self {
        self.config.enable_fragmentation_tracking = enabled;
        self
    }

    /// Enable or disable SciRS2 integration
    pub fn scirs2_integration(mut self, enabled: bool) -> Self {
        self.config.enable_scirs2_integration = enabled;
        self
    }

    /// Set maximum tracked allocations
    pub fn max_allocations(mut self, max: usize) -> Self {
        self.config.max_tracked_allocations = max;
        self
    }

    /// Set snapshot interval
    pub fn snapshot_interval(mut self, interval: Duration) -> Self {
        self.config.snapshot_interval = interval;
        self
    }

    /// Set memory pressure threshold
    pub fn pressure_threshold(mut self, threshold: f64) -> Self {
        self.config.memory_pressure_threshold = threshold;
        self
    }

    /// Set fragmentation alert threshold
    pub fn fragmentation_threshold(mut self, threshold: f64) -> Self {
        self.config.fragmentation_alert_threshold = threshold;
        self
    }

    /// Enable or disable stack trace collection
    pub fn stack_traces(mut self, enabled: bool) -> Self {
        self.config.enable_stack_traces = enabled;
        self
    }

    /// Build the memory profiler with the specified configuration
    pub fn build(
        self,
        base_profiler: Box<dyn crate::profiler::Profiler + Send + Sync>,
    ) -> MemoryProfiler {
        MemoryProfiler::new(base_profiler, self.config)
    }

    /// Get the current configuration
    pub fn config(&self) -> &MemoryProfilerConfig {
        &self.config
    }
}

impl Default for MemoryProfilerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience macros for common operations
#[macro_export]
macro_rules! memory_profiler {
    // Create with default configuration
    () => {
        $crate::memory_profiler::MemoryProfilerBuilder::new()
    };

    // Create with custom configuration
    ($($field:ident = $value:expr),* $(,)?) => {
        $crate::memory_profiler::MemoryProfilerBuilder::new()
        $(
            .$field($value)
        )*
    };
}

/// Convenience macro for creating performance hints
#[macro_export]
macro_rules! performance_hint {
    ($hint_type:expr, $severity:expr, $description:expr, $action:expr, $impact:expr) => {
        $crate::memory_profiler::PerformanceHint {
            hint_type: $hint_type,
            severity: $severity,
            description: $description.to_string(),
            suggested_action: $action.to_string(),
            impact_estimate: $impact,
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_memory_profiler_builder() {
        let builder = MemoryProfilerBuilder::new()
            .allocation_tracking(true)
            .pattern_analysis(false)
            .max_allocations(50000)
            .snapshot_interval(Duration::from_secs(30));

        let config = builder.config();
        assert!(config.enable_allocation_tracking);
        assert!(!config.enable_access_pattern_analysis);
        assert_eq!(config.max_tracked_allocations, 50000);
        assert_eq!(config.snapshot_interval, Duration::from_secs(30));
    }

    #[test]
    fn test_factory_configs() {
        let default_config = factory::default_config();
        assert!(default_config.enable_allocation_tracking);

        let high_perf_config = factory::high_performance_config();
        assert_eq!(high_perf_config.max_tracked_allocations, 1000000);

        let low_overhead_config = factory::low_overhead_config();
        assert!(!low_overhead_config.enable_access_pattern_analysis);

        let debug_config = factory::debug_config();
        assert!(debug_config.enable_stack_traces);
    }

    #[test]
    fn test_utils_functions() {
        // Test memory efficiency calculation
        assert_eq!(utils::calculate_memory_efficiency(1000, 800), 0.8);
        assert_eq!(utils::calculate_memory_efficiency(0, 0), 1.0);

        // Test fragmentation score calculation
        let frag_score = utils::calculate_fragmentation_score(1000, 10000, 10);
        assert!(frag_score >= 0.0 && frag_score <= 1.0);

        // Test pressure level conversion
        assert_eq!(utils::pressure_level_to_score(PressureLevel::High), 0.75);
        assert_eq!(utils::score_to_pressure_level(0.9), PressureLevel::Critical);

        // Test memory size formatting
        assert_eq!(utils::format_memory_size(1024), "1.00 KB");
        assert_eq!(utils::format_memory_size(1048576), "1.00 MB");
        assert_eq!(utils::format_memory_size(500), "500 B");

        // Test duration formatting
        assert_eq!(
            utils::format_duration(Duration::from_secs(3661)),
            "1h 1m 1s"
        );
        assert_eq!(utils::format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(utils::format_duration(Duration::from_millis(500)), "500ms");
    }

    #[test]
    fn test_profiler_summary() {
        let summary = ProfilerSummary {
            total_allocations: 1000,
            current_memory_usage: "10.5 MB".to_string(),
            peak_memory_usage: "15.2 MB".to_string(),
            memory_efficiency: 0.85,
            fragmentation_level: 0.15,
            pressure_level: PressureLevel::Low,
            optimization_opportunities: 3,
            last_update: Instant::now(),
        };

        assert_eq!(summary.total_allocations, 1000);
        assert_eq!(summary.memory_efficiency, 0.85);
        assert_eq!(summary.optimization_opportunities, 3);
    }

    #[test]
    fn test_collections_exports() {
        // Test that collections re-export the correct types
        #![allow(unused_imports)]
        use collections::monitoring::*;
        use collections::optimization::*;
        use collections::profiling::*;
        use collections::scirs2::*;

        // Just verify that the types are accessible
        // The actual functionality is tested in individual module tests
    }

    #[test]
    fn test_prelude_exports() {
        use prelude::*;

        // Verify that essential types are available
        let _pressure_level = PressureLevel::None;
        let _memory_type = MemoryType::Host;
        let _access_type = AccessType::Read;
        let _hint_severity = HintSeverity::Info;
    }

    #[test]
    fn test_macro_usage() {
        // Test the convenience macros
        let _builder = memory_profiler!();
        let _builder2 = memory_profiler!(allocation_tracking = true, max_allocations = 10000,);

        let hint = performance_hint!(
            PerformanceHintType::SuboptimalAccessPattern,
            HintSeverity::Warning,
            "Test hint",
            "Test action",
            0.3
        );

        assert_eq!(hint.impact_estimate, 0.3);
        assert_eq!(hint.description, "Test hint");
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that all the original types are still accessible
        // This ensures backward compatibility after the module split
        use scirs2::ScirS2IntegrationConfig;

        // Core types
        let _config = MemoryProfilerConfig::default();
        let _stats = GlobalMemoryStats::default();

        // Allocation types
        let _memory_type = MemoryType::Host;
        let _access_type = AccessType::Read;
        let _pressure_level = PressureLevel::None;

        // Pattern types
        let _pattern_config = PatternAnalysisConfig::default();

        // Fragmentation types
        let _frag_config = FragmentationConfig::default();
        let _frag_risk = FragmentationRisk::Low;

        // SciRS2 types
        let _scirs2_config = ScirS2IntegrationConfig::default();

        // All types should be accessible without qualification
        // after importing the module root (except scirs2 types which require explicit import)
    }
}
