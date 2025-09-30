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

// Full module re-exports (may have compilation dependencies)
#[cfg(feature = "full-execution-engine")]
pub use task_management::{TaskId, TaskManager, TaskPriority, TaskStatus};

#[cfg(feature = "full-execution-engine")]
pub use resource_management::{ResourceId, ResourceManager, ResourceType};

#[cfg(feature = "full-execution-engine")]
pub use fault_tolerance::{FailureHandlingResult, FaultToleranceManager, RetryDecision};

#[cfg(feature = "full-execution-engine")]
pub use performance_monitoring::{BottleneckRecord, MetricDataPoint, PerformanceMonitoringManager};

#[cfg(feature = "full-execution-engine")]
pub use security_management::{AuthenticationResult, SecurityManager, SecuritySession};

#[cfg(feature = "full-execution-engine")]
pub use load_balancing::{LoadBalancingManager, LoadLevel, WorkloadDistribution};

#[cfg(feature = "full-execution-engine")]
pub use hardware_management::{GpuDevice, HardwareManager, HealthStatus};

// Always available minimal integration
pub use minimal_integration::{
    MinimalEngineConfig, MinimalEngineError, MinimalExecutionEngine, MinimalTask, TaskResult,
    TaskStatus as MinimalTaskStatus,
};
