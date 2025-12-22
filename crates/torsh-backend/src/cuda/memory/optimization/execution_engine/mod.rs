//! CUDA Execution Engine Modules
//!
//! This module contains the comprehensive modular architecture for the CUDA
//! optimization execution engine. Each module provides specialized functionality
//! for managing different aspects of the execution pipeline.

/// Configuration management for execution engine components
pub mod config;

/// Task management and scheduling system
pub mod task_management;

/// Resource allocation and optimization management
pub mod resource_management;

/// Fault tolerance, recovery, and retry mechanisms
pub mod fault_tolerance;

/// Performance monitoring and metrics collection
pub mod performance_monitoring;

/// Security, authentication, and access control
pub mod security_management;

/// Load balancing and workload distribution
pub mod load_balancing;

/// Hardware management and device abstraction
pub mod hardware_management;

/// Minimal integration layer for basic functionality
pub mod minimal_integration;

// Re-export key types for easier access
pub use config::*;

// Export task management types
pub use task_management::{ResourceType, TaskId, TaskManager, TaskPriority};

// Export performance monitoring types (includes TaskStatus)
pub use performance_monitoring::TaskStatus;

// Export resource management types
pub use load_balancing::ResourceId;
pub use resource_management::GpuResourceManager;

// Export fault tolerance types
pub use fault_tolerance::{FailureHandlingResult, FaultToleranceManager, RetryDecision};

// Export performance monitoring types
pub use performance_monitoring::{BottleneckRecord, MetricDataPoint, PerformanceMonitoringManager};

// Export security management types
pub use security_management::{AuthenticationResult, SecurityManager, SecuritySession};

// Export load balancing types
pub use load_balancing::{LoadBalancingManager, LoadLevel, WorkloadDistribution};

// Export hardware management types
pub use hardware_management::{GpuDevice, HardwareManager, HealthStatus};

// Always available minimal integration
pub use minimal_integration::{
    MinimalEngineConfig, MinimalEngineError, MinimalExecutionEngine, MinimalTask, TaskResult,
    TaskStatus as MinimalTaskStatus,
};
