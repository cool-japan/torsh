//! Distributed autograd operations for large-scale training
//!
//! This module provides efficient gradient accumulation, synchronization, and
//! communication patterns for distributed deep learning training across multiple
//! devices and nodes.

// Core modules
pub mod common;
pub mod metrics;

// Implementation modules (to be migrated)
// pub mod operations;
// pub mod communication;
// pub mod coordination;
// pub mod recovery;
// pub mod async_ops;
// pub mod consistency;

// Re-export commonly used types from submodules
pub use common::{
    AllReduceAlgorithm, CommunicationPattern, CompressionStrategy, DistributedBackend,
    DistributedConfig, DistributedConfigBuilder, OperationStatus, ReductionOp, SyncOperationType,
};

pub use metrics::{
    DistributedStats, GradientStats, MetricsCollector, MetricsReport, PerformanceSummary,
    SyncStatistics,
};

// Convenience type aliases
/// Type alias for distributed configuration
pub type Config = DistributedConfig;

/// Type alias for sync operation type
pub type OpType = SyncOperationType;

/// Type alias for operation status
pub type Status = OperationStatus;
