//! Core configuration types for distributed gradient computation
//!
//! This module provides the foundational configuration types and enums
//! for distributed training operations in ToRSh.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Distributed training backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedBackend {
    /// No distributed training
    None,
    /// NCCL for NVIDIA GPUs
    Nccl,
    /// Gloo for CPU and mixed environments
    Gloo,
    /// MPI for HPC environments
    Mpi,
    /// Custom implementation
    Custom,
}

/// Gradient reduction operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Sum all gradients
    Sum,
    /// Average gradients across devices
    Mean,
    /// Maximum gradient values
    Max,
    /// Minimum gradient values
    Min,
}

/// Communication pattern for gradient synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationPattern {
    /// All-reduce operation (all devices get the reduced result)
    AllReduce,
    /// Reduce-scatter (each device gets a portion of the result)
    ReduceScatter,
    /// All-gather (all devices get concatenated data)
    AllGather,
    /// Parameter server pattern
    ParameterServer,
    /// Ring topology communication
    Ring,
    /// Tree topology communication
    Tree,
}

/// Gradient compression strategy for bandwidth optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// No compression
    None,
    /// Quantization to lower precision
    Quantization,
    /// Sparsification (top-k gradients)
    Sparsification,
    /// Error feedback compression
    ErrorFeedback,
    /// Gradient sketching
    Sketching,
}

/// Configuration for distributed gradient accumulation
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Backend type
    pub backend: DistributedBackend,
    /// World size (total number of processes)
    pub world_size: usize,
    /// Local rank of this process
    pub rank: usize,
    /// Reduction operation
    pub reduction_op: ReductionOp,
    /// Communication pattern
    pub communication_pattern: CommunicationPattern,
    /// Compression strategy
    pub compression: CompressionStrategy,
    /// Bucket size for gradient bucketing (bytes)
    pub bucket_size: usize,
    /// Timeout for communication operations
    pub timeout: Duration,
    /// Whether to overlap computation and communication
    pub overlap_comm_comp: bool,
    /// Number of gradient accumulation steps before synchronization
    pub gradient_accumulation_steps: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            backend: DistributedBackend::None,
            world_size: 1,
            rank: 0,
            reduction_op: ReductionOp::Mean,
            communication_pattern: CommunicationPattern::AllReduce,
            compression: CompressionStrategy::None,
            bucket_size: 25 * 1024 * 1024, // 25MB
            timeout: Duration::from_secs(30),
            overlap_comm_comp: true,
            gradient_accumulation_steps: 1,
        }
    }
}

/// Statistics for distributed gradient accumulation
#[derive(Debug, Clone, Default)]
pub struct DistributedStats {
    /// Total number of communication operations
    pub total_communications: usize,
    /// Total communication time
    pub total_comm_time: Duration,
    /// Total data communicated (bytes)
    pub total_data_communicated: usize,
    /// Average communication bandwidth (MB/s)
    pub avg_bandwidth_mbps: f64,
    /// Number of gradient accumulation steps
    pub accumulation_steps: usize,
    /// Compression ratio (if compression is used)
    pub compression_ratio: f64,
    /// Synchronization overhead
    pub sync_overhead: Duration,
}