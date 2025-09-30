//! SciRS2 Integration Module
//!
//! Comprehensive SciRS2 ecosystem integration providing advanced memory management,
//! real-time monitoring, ML-based optimization, and performance analytics.
//!
//! This module provides a modular architecture for SciRS2 integration with:
//! - Configuration management with comprehensive validation
//! - Real-time allocator statistics and performance metrics
//! - Advanced memory pool management and health assessment
//! - Event-driven architecture with comprehensive event processing
//! - Real-time monitoring with alerting and dashboard capabilities
//! - ML-based optimization with predictive allocation and anomaly detection
//! - Main integration orchestration coordinating all components
//!
//! # Architecture
//!
//! The module is organized into focused components:
//! - `config`: Configuration types and validation
//! - `statistics`: Allocator performance metrics and analytics
//! - `pool_management`: Memory pool management and optimization
//! - `event_system`: Event handling and processing framework
//! - `monitoring`: Real-time monitoring and alerting system
//! - `optimization`: ML-based optimization and predictive features
//! - `integration`: Main orchestration layer coordinating all components
//!
//! # Usage
//!
//! ```rust
//! use torsh_backend::memory_profiler::scirs2::{ScirS2Integration, ScirS2IntegrationConfig};
//!
//! // Create configuration
//! let config = ScirS2IntegrationConfig {
//!     enable_realtime_sync: true,
//!     enable_optimization_suggestions: true,
//!     ..Default::default()
//! };
//!
//! // Initialize integration
//! let mut integration = ScirS2Integration::new(config);
//! let result = integration.activate();
//! assert!(result.is_ok());
//!
//! // Process events
//! let event = ScirS2Event::Allocation {
//!     ptr: std::ptr::null_mut(),
//!     size: 1024,
//!     allocator: "scirs2_default".to_string(),
//!     allocation_context: AllocationEventContext::new(),
//! };
//! integration.process_event(event);
//!
//! // Get optimization suggestions
//! let suggestions = integration.get_optimization_suggestions();
//! ```

// Missing enum definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    Conservative,
    Balanced,
    Aggressive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitoringLevel {
    Basic,
    Comprehensive,
    Detailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTrackingLevel {
    Basic,
    Comprehensive,
    Detailed,
}

pub mod config;
pub mod event_system;
pub mod integration;
pub mod monitoring;
pub mod optimization;
pub mod pool_management;
pub mod statistics;

// Re-export main types for convenient access
pub use config::{
    validate_config, AdvancedIntegrationConfig, AlertSeverity, ComparisonType, IntegrationStatus,
    OptimizationType, PoolOptimizationRecommendation, ProfilingDetailLevel,
    ScirS2IntegrationConfig,
};

pub use statistics::{
    AggregateMetrics, AllocationUsageStats, AllocatorAdvancedMetrics, AllocatorPerformanceProfile,
    AllocatorStatsAggregator, MemoryStateSnapshot, PerformanceSnapshot, ScirS2AllocatorStats,
};

pub use pool_management::{
    PoolAdvancedAnalytics, PoolHealthAssessment, PoolStatsAggregator, ScirS2PoolInfo,
    SystemPoolMetrics,
};

pub use event_system::{
    AllocationEventContext, DeallocationEventContext, EventFilter, EventSeverity, EventStatistics,
    MemoryPressureContext, OptimizationContext, PoolCreationContext, PoolDestructionContext,
    ScirS2Event, ScirS2EventProcessor,
};

pub use monitoring::{
    ActiveAlert, AlertCondition, DashboardData, MonitoringConfig, MonitoringDataPoint,
    ScirS2MonitoringSystem,
};

pub use optimization::{
    AdvancedScirS2Features, AnomalyDetector, AutoOptimizationSystem, DetectedAnomaly, MLModel,
    OptimizationMetrics, OptimizationResult, OptimizationTask, PredictiveAllocationEngine,
    ScirS2OptimizationEngine,
};

pub use integration::{AggregateStatistics, IntegrationMetrics, ScirS2Integration};

// Re-export commonly used enums
pub use config::{CleanupStatus, FragmentationType, PressureTrend, UtilizationChangeReason};

// Integration module only exports ScirS2Integration, IntegrationMetrics, and AggregateStatistics

/// Default SciRS2 integration instance
///
/// Provides a convenient way to access a default SciRS2 integration
/// with standard configuration for common use cases.
pub fn default_integration() -> ScirS2Integration {
    ScirS2Integration::new(ScirS2IntegrationConfig::default())
}

/// Create optimized SciRS2 integration for production use
///
/// Returns a SciRS2 integration configured with production-ready settings
/// including real-time monitoring, optimization suggestions, and advanced features.
pub fn production_integration() -> ScirS2Integration {
    let config = ScirS2IntegrationConfig {
        enable_realtime_sync: true,
        enable_event_callbacks: true,
        track_allocation_patterns: true,
        enable_optimization_suggestions: true,
        advanced_config: AdvancedIntegrationConfig {
            enable_predictive_modeling: true,
            enable_automated_optimization: true,
            enable_health_monitoring: true,
            enable_performance_profiling: true,
            profiling_detail_level: ProfilingDetailLevel::Comprehensive,
            optimization_aggressiveness: 0.8, // Aggressive optimization
            ..Default::default()
        },
        ..Default::default()
    };

    ScirS2Integration::new(config)
}

/// Create lightweight SciRS2 integration for development use
///
/// Returns a SciRS2 integration configured with minimal overhead
/// suitable for development and testing environments.
pub fn development_integration() -> ScirS2Integration {
    use std::time::Duration;
    let config = ScirS2IntegrationConfig {
        enable_realtime_sync: false,
        sync_interval: Duration::from_secs(30),
        enable_event_callbacks: false,
        track_allocation_patterns: false,
        enable_optimization_suggestions: false,
        advanced_config: AdvancedIntegrationConfig::conservative(),
    };

    ScirS2Integration::new(config)
}

/// Validate SciRS2 integration health
///
/// Comprehensive health check function that examines all aspects
/// of a SciRS2 integration instance and returns detailed health information.
pub fn validate_integration_health(integration: &ScirS2Integration) -> HealthReport {
    let status = integration.get_status();
    let metrics = integration.get_integration_metrics();
    let suggestions = integration.get_optimization_suggestions();

    let mut issues = Vec::new();
    let mut warnings = Vec::new();
    let mut recommendations = Vec::new();

    // Check basic health
    if !integration.is_healthy() {
        issues.push("Integration is not healthy".to_string());
    }

    // Check sync status
    if let Some(last_sync) = status.last_sync {
        let sync_age = std::time::Instant::now().duration_since(last_sync);
        if sync_age > integration.config.sync_interval * 2 {
            warnings.push("Last synchronization is old".to_string());
        }
    } else {
        warnings.push("No synchronization performed yet".to_string());
    }

    // Check success rate
    if metrics.success_rate < 0.9 {
        if metrics.success_rate < 0.7 {
            issues.push(format!(
                "Low success rate: {:.1}%",
                metrics.success_rate * 100.0
            ));
        } else {
            warnings.push(format!(
                "Moderate success rate: {:.1}%",
                metrics.success_rate * 100.0
            ));
        }
    }

    // Check optimization suggestions
    if suggestions.len() > 10 {
        recommendations
            .push("High number of optimization suggestions - consider applying them".to_string());
    }

    // Check allocator health
    for (name, stats) in integration.get_all_allocator_stats() {
        if !stats.is_healthy() {
            issues.push(format!("Allocator '{}' is not healthy", name));
        }
        if stats.memory_efficiency < 0.8 {
            warnings.push(format!(
                "Allocator '{}' has low efficiency: {:.1}%",
                name,
                stats.memory_efficiency * 100.0
            ));
        }
    }

    // Check pool health
    for (id, pool) in integration.get_all_pools() {
        if !pool.is_healthy() {
            issues.push(format!("Pool '{}' is not healthy", id));
        }
        if pool.utilization > 0.9 {
            warnings.push(format!(
                "Pool '{}' has high utilization: {:.1}%",
                id,
                pool.utilization * 100.0
            ));
        }
    }

    let overall_status = if !issues.is_empty() {
        OverallHealthStatus::Unhealthy
    } else if !warnings.is_empty() {
        OverallHealthStatus::Warning
    } else {
        OverallHealthStatus::Healthy
    };

    HealthReport {
        overall_status,
        issues,
        warnings,
        recommendations,
        health_score: status.health_score,
        uptime: metrics.uptime,
        total_events_processed: metrics.total_events_processed,
        success_rate: metrics.success_rate,
    }
}

/// Overall health status
#[derive(Debug, Clone, PartialEq)]
pub enum OverallHealthStatus {
    Healthy,
    Warning,
    Unhealthy,
}

/// Comprehensive health report
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Overall health status
    pub overall_status: OverallHealthStatus,

    /// Critical issues requiring immediate attention
    pub issues: Vec<String>,

    /// Warnings that should be monitored
    pub warnings: Vec<String>,

    /// Optimization recommendations
    pub recommendations: Vec<String>,

    /// Numerical health score (0.0 to 1.0)
    pub health_score: f64,

    /// Integration uptime
    pub uptime: std::time::Duration,

    /// Total events processed
    pub total_events_processed: u64,

    /// Overall success rate
    pub success_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_default_integration_creation() {
        let integration = default_integration();
        assert!(!integration.active);
        assert_eq!(integration.allocator_stats.read().len(), 0);
        assert_eq!(integration.pool_info.read().len(), 0);
    }

    #[test]
    fn test_production_integration_creation() {
        let integration = production_integration();
        assert!(!integration.active);
        assert!(integration.config.enable_realtime_sync);
        assert!(integration.config.enable_optimization_suggestions);
        // Note: optimization_level and monitoring_level fields are not part of ScirS2IntegrationConfig
        // Additional configuration could be added if needed
    }

    #[test]
    fn test_development_integration_creation() {
        let integration = development_integration();
        assert!(!integration.active);
        assert!(!integration.config.enable_realtime_sync);
        assert!(!integration.config.enable_optimization_suggestions);
        // Note: optimization_level and monitoring_level fields are not part of ScirS2IntegrationConfig
        // Conservative configuration would focus on basic fields only
    }

    #[test]
    fn test_integration_activation() {
        let mut integration = default_integration();
        let result = integration.activate();

        // Should succeed with default config
        assert!(result.is_ok());
        assert!(integration.active);
    }

    #[test]
    fn test_integration_configuration_validation() {
        let invalid_config = ScirS2IntegrationConfig {
            sync_interval: Duration::from_secs(0),
            ..Default::default()
        };

        let result = validate_config(&invalid_config);
        assert!(result.is_err());
    }

    #[test]
    fn test_event_processing() {
        let mut integration = default_integration();
        let _ = integration.activate();

        let event = ScirS2Event::Allocation {
            ptr: std::ptr::null_mut(),
            size: 1024,
            allocator: "test_allocator".to_string(),
            allocation_context: AllocationEventContext {
                thread_id: 1,
                reason: "test allocation".to_string(),
                performance_snapshot: PerformanceSnapshot::new(),
                memory_snapshot: MemoryStateSnapshot::new(),
            },
        };

        integration.process_event(event);

        // Should have processed the event
        let metrics = integration.get_integration_metrics();
        assert!(metrics.total_events_processed > 0);
    }

    #[test]
    fn test_allocator_statistics_tracking() {
        let mut integration = default_integration();
        let _ = integration.activate();

        // Process several allocation events
        for i in 0..5 {
            let event = ScirS2Event::Allocation {
                ptr: std::ptr::null_mut(),
                size: 1024 * (i + 1),
                allocator: "test_allocator".to_string(),
                allocation_context: AllocationEventContext {
                    thread_id: 1,
                    reason: format!("test allocation {}", i),
                    performance_snapshot: PerformanceSnapshot::new(),
                    memory_snapshot: MemoryStateSnapshot::new(),
                },
            };

            integration.process_event(event);
        }

        // Should have statistics for the allocator
        let stats = integration.get_allocator_stats("test_allocator");
        assert!(stats.is_some());
    }

    #[test]
    fn test_optimization_suggestions() {
        let mut integration = ScirS2Integration::new(ScirS2IntegrationConfig {
            enable_optimization_suggestions: true,
            ..Default::default()
        });

        let _ = integration.activate();

        // Create an allocator with poor efficiency
        let mut poor_stats = ScirS2AllocatorStats::new("inefficient_allocator".to_string());
        poor_stats.memory_efficiency = 0.5; // Poor efficiency
        poor_stats.allocation_failures = 5; // Some failures

        integration
            .allocator_stats
            .write()
            .insert("inefficient_allocator".to_string(), poor_stats);

        let suggestions = integration.get_optimization_suggestions();
        assert!(!suggestions.is_empty());

        // Should have suggestions for both low efficiency and failures
        let efficiency_suggestion = suggestions
            .iter()
            .any(|s| s.description.contains("efficiency"));
        let failure_suggestion = suggestions
            .iter()
            .any(|s| s.description.contains("failures"));

        assert!(efficiency_suggestion);
        assert!(failure_suggestion);
    }

    #[test]
    fn test_health_validation() {
        let integration = default_integration();
        let health_report = validate_integration_health(&integration);

        // Fresh integration should have warnings but not be unhealthy
        match health_report.overall_status {
            OverallHealthStatus::Healthy | OverallHealthStatus::Warning => {}
            OverallHealthStatus::Unhealthy => panic!("Fresh integration should not be unhealthy"),
        }

        assert!(health_report.health_score >= 0.0);
        assert!(health_report.health_score <= 1.0);
    }

    #[test]
    fn test_synchronization() {
        let mut integration = default_integration();
        let _ = integration.activate();

        let result = integration.sync_statistics();
        assert!(result.is_ok());

        // Should have updated last sync time
        assert!(integration.get_status().last_sync.is_some());
    }

    #[test]
    fn test_pool_management() {
        let mut integration = default_integration();
        let _ = integration.activate();

        // Sync should create sample pool
        let _ = integration.sync_statistics();

        let pools = integration.get_all_pools();
        assert!(!pools.is_empty());

        // Should have the sample tensor pool
        assert!(pools.contains_key("tensor_pool_1"));
    }

    #[test]
    fn test_integration_deactivation() {
        let mut integration = default_integration();
        let _ = integration.activate();

        assert!(integration.active);

        integration.deactivate();
        assert!(!integration.active);
        assert!(integration.get_status().last_sync.is_none());
    }

    #[test]
    fn test_configuration_updates() {
        let mut integration = default_integration();
        let _ = integration.activate();

        let new_config = ScirS2IntegrationConfig {
            enable_optimization_suggestions: true,
            enable_realtime_sync: true, // Enhanced monitoring setting
            ..integration.config.clone()
        };

        let result = integration.update_config(new_config.clone());
        assert!(result.is_ok());
        // Note: monitoring_level field is not part of ScirS2IntegrationConfig
        assert!(integration.config.enable_optimization_suggestions);
    }

    #[test]
    fn test_buffer_flushing() {
        let mut integration = default_integration();
        let _ = integration.activate();

        // Process some events
        for i in 0..10 {
            let event = ScirS2Event::Allocation {
                ptr: std::ptr::null_mut(),
                size: 1024,
                allocator: format!("allocator_{}", i % 3),
                allocation_context: AllocationEventContext {
                    thread_id: 1,
                    reason: "Test allocation".to_string(),
                    performance_snapshot: PerformanceSnapshot::new(),
                    memory_snapshot: MemoryStateSnapshot::new(),
                },
            };
            integration.process_event(event);
        }

        // Flush all buffers
        integration.flush_all();

        // Should not cause any issues
        assert!(integration.active);
    }

    #[test]
    fn test_aggregate_statistics() {
        let mut integration = default_integration();
        let _ = integration.activate();

        // Process some events and sync
        let event = ScirS2Event::Allocation {
            ptr: std::ptr::null_mut(),
            size: 1024,
            allocator: "test_allocator".to_string(),
            allocation_context: AllocationEventContext {
                thread_id: 1,
                reason: "Test allocation".to_string(),
                performance_snapshot: PerformanceSnapshot::new(),
                memory_snapshot: MemoryStateSnapshot::new(),
            },
        };

        integration.process_event(event);
        let _ = integration.sync_statistics();

        let aggregate_stats = integration.get_aggregate_statistics();
        assert!(aggregate_stats.monitoring_active);
        assert!(aggregate_stats.total_optimization_suggestions >= 0);
    }

    #[test]
    fn test_performance_metrics() {
        let mut integration = production_integration();
        let _ = integration.activate();

        // Process events to generate metrics
        for i in 0..20 {
            let event = ScirS2Event::Allocation {
                ptr: std::ptr::null_mut(),
                size: 1024 * (i + 1),
                allocator: "performance_test".to_string(),
                allocation_context: AllocationEventContext {
                    thread_id: 1,
                    reason: "Test allocation".to_string(),
                    performance_snapshot: PerformanceSnapshot::new(),
                    memory_snapshot: MemoryStateSnapshot::new(),
                },
            };
            integration.process_event(event);
        }

        // Sync to populate stats
        let _ = integration.sync_statistics();

        let metrics = integration.get_integration_metrics();
        assert!(metrics.total_events_processed >= 20);
        assert!(metrics.success_rate >= 0.0);
        assert!(metrics.success_rate <= 1.0);
    }

    #[test]
    fn test_comprehensive_integration_workflow() {
        let mut integration = production_integration();

        // 1. Activate integration
        let activate_result = integration.activate();
        assert!(activate_result.is_ok());
        assert!(integration.active);

        // 2. Process various events
        let events = vec![
            ScirS2Event::Allocation {
                ptr: std::ptr::null_mut(),
                size: 2048,
                allocator: "tensor_allocator".to_string(),
                allocation_context: AllocationEventContext {
                    thread_id: 1,
                    reason: "Test allocation".to_string(),
                    performance_snapshot: PerformanceSnapshot::new(),
                    memory_snapshot: MemoryStateSnapshot::new(),
                },
            },
            ScirS2Event::PoolCreated {
                pool_id: "custom_pool".to_string(),
                capacity: 1024 * 1024,
                pool_type: "custom".to_string(),
                creation_context: PoolCreationContext {
                    creation_reason: "test pool".to_string(),
                    initial_config: std::collections::HashMap::new(),
                    expected_usage: "testing".to_string(),
                },
            },
            ScirS2Event::MemoryPressure {
                level: crate::memory_profiler::allocation::PressureLevel::Medium,
                available_memory: 1024 * 1024 * 100,
                pressure_context: MemoryPressureContext {
                    system_memory_usage: 1024 * 1024 * 500,
                    scirs2_memory_usage: 1024 * 1024 * 50,
                    active_allocators: vec!["tensor_allocator".to_string()],
                    pressure_trend: PressureTrend::Increasing,
                },
            },
        ];

        for event in events {
            integration.process_event(event);
        }

        // 3. Synchronize statistics
        let sync_result = integration.sync_statistics();
        assert!(sync_result.is_ok());

        // 4. Get optimization suggestions
        let suggestions = integration.get_optimization_suggestions();
        // Should have suggestions based on the events

        // 5. Get health report
        let health_report = validate_integration_health(&integration);
        assert!(health_report.health_score >= 0.0);

        // 6. Get performance metrics
        let metrics = integration.get_integration_metrics();
        assert!(metrics.total_events_processed >= 3);

        // 7. Get dashboard data
        let dashboard = integration.get_dashboard_data();
        // Should have monitoring data

        // 8. Test configuration update
        let new_config = ScirS2IntegrationConfig {
            enable_realtime_sync: false, // Conservative setting
            ..integration.config.clone()
        };
        let config_result = integration.update_config(new_config);
        assert!(config_result.is_ok());

        // 9. Flush all buffers
        integration.flush_all();

        // 10. Deactivate integration
        integration.deactivate();
        assert!(!integration.active);
    }
}
