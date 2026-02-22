//! Common utilities and types for distributed autograd operations

pub mod types;

// Re-export commonly used types
pub use types::{
    AllReduceAlgorithm, CommunicationPattern, CompressionStrategy, DistributedBackend,
    DistributedConfig, DistributedConfigBuilder, OperationStatus, ReductionOp, SyncOperationType,
};
