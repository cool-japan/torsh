//! Common types and configuration for distributed autograd operations

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Distributed training backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

impl Default for DistributedBackend {
    fn default() -> Self {
        Self::None
    }
}

impl DistributedBackend {
    /// Check if backend supports GPU operations
    pub fn supports_gpu(&self) -> bool {
        matches!(self, DistributedBackend::Nccl | DistributedBackend::Custom)
    }

    /// Check if backend supports CPU operations
    pub fn supports_cpu(&self) -> bool {
        matches!(
            self,
            DistributedBackend::Gloo | DistributedBackend::Mpi | DistributedBackend::Custom
        )
    }

    /// Get backend name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            DistributedBackend::None => "none",
            DistributedBackend::Nccl => "nccl",
            DistributedBackend::Gloo => "gloo",
            DistributedBackend::Mpi => "mpi",
            DistributedBackend::Custom => "custom",
        }
    }
}

/// Gradient reduction operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

impl Default for ReductionOp {
    fn default() -> Self {
        Self::Mean
    }
}

impl ReductionOp {
    /// Get operation name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            ReductionOp::Sum => "sum",
            ReductionOp::Mean => "mean",
            ReductionOp::Max => "max",
            ReductionOp::Min => "min",
        }
    }

    /// Check if operation preserves gradient scale
    pub fn preserves_scale(&self) -> bool {
        matches!(self, ReductionOp::Sum | ReductionOp::Mean)
    }
}

/// Communication pattern for gradient synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

impl Default for CommunicationPattern {
    fn default() -> Self {
        Self::AllReduce
    }
}

impl CommunicationPattern {
    /// Get pattern name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            CommunicationPattern::AllReduce => "all_reduce",
            CommunicationPattern::ReduceScatter => "reduce_scatter",
            CommunicationPattern::AllGather => "all_gather",
            CommunicationPattern::ParameterServer => "parameter_server",
            CommunicationPattern::Ring => "ring",
            CommunicationPattern::Tree => "tree",
        }
    }

    /// Get bandwidth efficiency score (0.0 to 1.0)
    pub fn bandwidth_efficiency(&self) -> f32 {
        match self {
            CommunicationPattern::AllReduce => 0.9,
            CommunicationPattern::ReduceScatter => 0.85,
            CommunicationPattern::AllGather => 0.7,
            CommunicationPattern::ParameterServer => 0.6,
            CommunicationPattern::Ring => 0.95,
            CommunicationPattern::Tree => 0.8,
        }
    }

    /// Check if pattern requires a coordinator/master node
    pub fn requires_coordinator(&self) -> bool {
        matches!(self, CommunicationPattern::ParameterServer)
    }
}

/// Gradient compression strategy for bandwidth optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

impl Default for CompressionStrategy {
    fn default() -> Self {
        Self::None
    }
}

impl CompressionStrategy {
    /// Get compression ratio estimate (0.0 to 1.0)
    pub fn compression_ratio(&self) -> f32 {
        match self {
            CompressionStrategy::None => 1.0,
            CompressionStrategy::Quantization => 0.5,
            CompressionStrategy::Sparsification => 0.1,
            CompressionStrategy::ErrorFeedback => 0.3,
            CompressionStrategy::Sketching => 0.2,
        }
    }

    /// Get strategy name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            CompressionStrategy::None => "none",
            CompressionStrategy::Quantization => "quantization",
            CompressionStrategy::Sparsification => "sparsification",
            CompressionStrategy::ErrorFeedback => "error_feedback",
            CompressionStrategy::Sketching => "sketching",
        }
    }

    /// Check if compression is lossy
    pub fn is_lossy(&self) -> bool {
        !matches!(self, CompressionStrategy::None)
    }
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
    /// Maximum number of retries for failed operations
    pub max_retries: usize,
    /// Enable gradient clipping during synchronization
    pub enable_gradient_clipping: bool,
    /// Gradient clipping threshold
    pub gradient_clip_threshold: f32,
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
            max_retries: 3,
            enable_gradient_clipping: false,
            gradient_clip_threshold: 1.0,
        }
    }
}

impl DistributedConfig {
    /// Create a configuration for single-node training
    pub fn single_node() -> Self {
        Self {
            backend: DistributedBackend::None,
            world_size: 1,
            rank: 0,
            ..Default::default()
        }
    }

    /// Create a configuration for multi-node NCCL training
    pub fn nccl_multinode(world_size: usize, rank: usize) -> Self {
        Self {
            backend: DistributedBackend::Nccl,
            world_size,
            rank,
            communication_pattern: CommunicationPattern::AllReduce,
            ..Default::default()
        }
    }

    /// Create a configuration for CPU-based Gloo training
    pub fn gloo_cpu(world_size: usize, rank: usize) -> Self {
        Self {
            backend: DistributedBackend::Gloo,
            world_size,
            rank,
            communication_pattern: CommunicationPattern::AllReduce,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for bandwidth-constrained environments
    pub fn bandwidth_optimized(world_size: usize, rank: usize) -> Self {
        Self {
            backend: DistributedBackend::Gloo,
            world_size,
            rank,
            communication_pattern: CommunicationPattern::Ring,
            compression: CompressionStrategy::Sparsification,
            bucket_size: 10 * 1024 * 1024, // 10MB
            gradient_accumulation_steps: 4,
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.world_size == 0 {
            return Err("World size must be greater than 0".to_string());
        }

        if self.rank >= self.world_size {
            return Err("Rank must be less than world size".to_string());
        }

        if self.bucket_size == 0 {
            return Err("Bucket size must be greater than 0".to_string());
        }

        if self.gradient_accumulation_steps == 0 {
            return Err("Gradient accumulation steps must be greater than 0".to_string());
        }

        if self.gradient_clip_threshold <= 0.0 {
            return Err("Gradient clip threshold must be positive".to_string());
        }

        // Check backend compatibility
        match (self.backend, self.world_size) {
            (DistributedBackend::None, size) if size > 1 => {
                return Err("Cannot use None backend with world size > 1".to_string());
            }
            _ => {}
        }

        Ok(())
    }

    /// Check if this configuration represents the master/coordinator rank
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }

    /// Get estimated bandwidth usage multiplier
    pub fn bandwidth_multiplier(&self) -> f32 {
        let pattern_efficiency = self.communication_pattern.bandwidth_efficiency();
        let compression_ratio = self.compression.compression_ratio();
        let accumulation_factor = 1.0 / self.gradient_accumulation_steps as f32;

        pattern_efficiency * compression_ratio * accumulation_factor
    }

    /// Get human-readable configuration summary
    pub fn summary(&self) -> String {
        format!(
            "DistributedConfig {{ backend: {}, world_size: {}, rank: {}, pattern: {}, compression: {} }}",
            self.backend.as_str(),
            self.world_size,
            self.rank,
            self.communication_pattern.as_str(),
            self.compression.as_str()
        )
    }
}

/// Builder pattern for DistributedConfig
pub struct DistributedConfigBuilder {
    config: DistributedConfig,
}

impl DistributedConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: DistributedConfig::default(),
        }
    }

    /// Set the backend type
    pub fn backend(mut self, backend: DistributedBackend) -> Self {
        self.config.backend = backend;
        self
    }

    /// Set world size and rank
    pub fn world(mut self, world_size: usize, rank: usize) -> Self {
        self.config.world_size = world_size;
        self.config.rank = rank;
        self
    }

    /// Set reduction operation
    pub fn reduction_op(mut self, op: ReductionOp) -> Self {
        self.config.reduction_op = op;
        self
    }

    /// Set communication pattern
    pub fn communication_pattern(mut self, pattern: CommunicationPattern) -> Self {
        self.config.communication_pattern = pattern;
        self
    }

    /// Set compression strategy
    pub fn compression(mut self, compression: CompressionStrategy) -> Self {
        self.config.compression = compression;
        self
    }

    /// Set bucket size in bytes
    pub fn bucket_size(mut self, size: usize) -> Self {
        self.config.bucket_size = size;
        self
    }

    /// Set timeout duration
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Enable or disable computation/communication overlap
    pub fn overlap_comm_comp(mut self, enable: bool) -> Self {
        self.config.overlap_comm_comp = enable;
        self
    }

    /// Set gradient accumulation steps
    pub fn gradient_accumulation_steps(mut self, steps: usize) -> Self {
        self.config.gradient_accumulation_steps = steps;
        self
    }

    /// Set maximum retry attempts
    pub fn max_retries(mut self, retries: usize) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Enable gradient clipping with threshold
    pub fn gradient_clipping(mut self, threshold: f32) -> Self {
        self.config.enable_gradient_clipping = true;
        self.config.gradient_clip_threshold = threshold;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<DistributedConfig, String> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for DistributedConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Synchronization operation type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyncOperationType {
    /// All-reduce with specific algorithm
    AllReduce(AllReduceAlgorithm),
    /// Point-to-point gradient exchange
    P2PExchange(usize, usize), // source, destination
    /// Broadcast from root
    Broadcast(usize), // root rank
    /// Reduce to specific rank
    Reduce(usize), // root rank
    /// Custom synchronization pattern
    Custom(String),
}

impl Default for SyncOperationType {
    fn default() -> Self {
        Self::AllReduce(AllReduceAlgorithm::Ring)
    }
}

impl SyncOperationType {
    /// Get operation name as string
    pub fn as_str(&self) -> String {
        match self {
            SyncOperationType::AllReduce(alg) => format!("all_reduce_{}", alg.as_str()),
            SyncOperationType::P2PExchange(src, dst) => format!("p2p_exchange_{}_{}", src, dst),
            SyncOperationType::Broadcast(root) => format!("broadcast_{}", root),
            SyncOperationType::Reduce(root) => format!("reduce_{}", root),
            SyncOperationType::Custom(name) => format!("custom_{}", name),
        }
    }

    /// Check if operation involves all ranks
    pub fn involves_all_ranks(&self) -> bool {
        matches!(self, SyncOperationType::AllReduce(_))
    }

    /// Get estimated bandwidth efficiency (0.0 to 1.0)
    pub fn bandwidth_efficiency(&self) -> f32 {
        match self {
            SyncOperationType::AllReduce(alg) => alg.bandwidth_efficiency(),
            SyncOperationType::P2PExchange(_, _) => 1.0,
            SyncOperationType::Broadcast(_) => 0.8,
            SyncOperationType::Reduce(_) => 0.7,
            SyncOperationType::Custom(_) => 0.5,
        }
    }
}

/// All-reduce algorithm variants
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllReduceAlgorithm {
    /// Ring algorithm (bandwidth optimal)
    Ring,
    /// Tree algorithm (latency optimal)
    Tree,
    /// Butterfly algorithm (hybrid)
    Butterfly,
    /// Recursive halving/doubling
    RecursiveHalving,
    /// Hierarchical (multi-level)
    Hierarchical,
}

impl Default for AllReduceAlgorithm {
    fn default() -> Self {
        Self::Ring
    }
}

impl AllReduceAlgorithm {
    /// Get algorithm name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            AllReduceAlgorithm::Ring => "ring",
            AllReduceAlgorithm::Tree => "tree",
            AllReduceAlgorithm::Butterfly => "butterfly",
            AllReduceAlgorithm::RecursiveHalving => "recursive_halving",
            AllReduceAlgorithm::Hierarchical => "hierarchical",
        }
    }

    /// Get bandwidth efficiency (0.0 to 1.0)
    pub fn bandwidth_efficiency(&self) -> f32 {
        match self {
            AllReduceAlgorithm::Ring => 1.0,
            AllReduceAlgorithm::Tree => 0.7,
            AllReduceAlgorithm::Butterfly => 0.9,
            AllReduceAlgorithm::RecursiveHalving => 0.85,
            AllReduceAlgorithm::Hierarchical => 0.8,
        }
    }

    /// Get latency efficiency (0.0 to 1.0)
    pub fn latency_efficiency(&self) -> f32 {
        match self {
            AllReduceAlgorithm::Ring => 0.6,
            AllReduceAlgorithm::Tree => 1.0,
            AllReduceAlgorithm::Butterfly => 0.9,
            AllReduceAlgorithm::RecursiveHalving => 0.8,
            AllReduceAlgorithm::Hierarchical => 0.85,
        }
    }

    /// Check if algorithm is suitable for large data transfers
    pub fn suitable_for_large_data(&self) -> bool {
        matches!(
            self,
            AllReduceAlgorithm::Ring | AllReduceAlgorithm::Butterfly
        )
    }

    /// Check if algorithm is suitable for latency-sensitive operations
    pub fn suitable_for_low_latency(&self) -> bool {
        matches!(
            self,
            AllReduceAlgorithm::Tree | AllReduceAlgorithm::Butterfly
        )
    }
}

/// Operation status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperationStatus {
    /// Operation is pending
    Pending,
    /// Operation is in progress
    InProgress,
    /// Operation completed successfully
    Completed,
    /// Operation failed
    Failed(String),
    /// Operation timed out
    TimedOut,
}

impl Default for OperationStatus {
    fn default() -> Self {
        Self::Pending
    }
}

impl OperationStatus {
    /// Check if operation is complete (either success or failure)
    pub fn is_complete(&self) -> bool {
        matches!(
            self,
            OperationStatus::Completed | OperationStatus::Failed(_) | OperationStatus::TimedOut
        )
    }

    /// Check if operation was successful
    pub fn is_success(&self) -> bool {
        matches!(self, OperationStatus::Completed)
    }

    /// Check if operation failed
    pub fn is_failure(&self) -> bool {
        matches!(self, OperationStatus::Failed(_) | OperationStatus::TimedOut)
    }

    /// Get failure reason if operation failed
    pub fn failure_reason(&self) -> Option<&str> {
        match self {
            OperationStatus::Failed(reason) => Some(reason),
            OperationStatus::TimedOut => Some("timeout"),
            _ => None,
        }
    }

    /// Get status name as string
    pub fn as_str(&self) -> &str {
        match self {
            OperationStatus::Pending => "pending",
            OperationStatus::InProgress => "in_progress",
            OperationStatus::Completed => "completed",
            OperationStatus::Failed(_) => "failed",
            OperationStatus::TimedOut => "timed_out",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_backend() {
        assert_eq!(DistributedBackend::default(), DistributedBackend::None);
        assert!(DistributedBackend::Nccl.supports_gpu());
        assert!(!DistributedBackend::Nccl.supports_cpu());
        assert!(DistributedBackend::Gloo.supports_cpu());
        assert_eq!(DistributedBackend::Nccl.as_str(), "nccl");
    }

    #[test]
    fn test_reduction_op() {
        assert_eq!(ReductionOp::default(), ReductionOp::Mean);
        assert!(ReductionOp::Mean.preserves_scale());
        assert!(!ReductionOp::Max.preserves_scale());
        assert_eq!(ReductionOp::Sum.as_str(), "sum");
    }

    #[test]
    fn test_communication_pattern() {
        assert_eq!(
            CommunicationPattern::default(),
            CommunicationPattern::AllReduce
        );
        assert!(CommunicationPattern::ParameterServer.requires_coordinator());
        assert!(!CommunicationPattern::AllReduce.requires_coordinator());
        assert!(CommunicationPattern::Ring.bandwidth_efficiency() > 0.9);
    }

    #[test]
    fn test_compression_strategy() {
        assert_eq!(CompressionStrategy::default(), CompressionStrategy::None);
        assert!(!CompressionStrategy::None.is_lossy());
        assert!(CompressionStrategy::Quantization.is_lossy());
        assert!(CompressionStrategy::Sparsification.compression_ratio() < 0.2);
    }

    #[test]
    fn test_distributed_config_validation() {
        let config = DistributedConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = DistributedConfig {
            world_size: 0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());

        let invalid_rank = DistributedConfig {
            world_size: 2,
            rank: 2,
            ..Default::default()
        };
        assert!(invalid_rank.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = DistributedConfigBuilder::new()
            .backend(DistributedBackend::Nccl)
            .world(4, 1)
            .compression(CompressionStrategy::Quantization)
            .gradient_clipping(0.5)
            .build()
            .unwrap();

        assert_eq!(config.backend, DistributedBackend::Nccl);
        assert_eq!(config.world_size, 4);
        assert_eq!(config.rank, 1);
        assert_eq!(config.compression, CompressionStrategy::Quantization);
        assert!(config.enable_gradient_clipping);
        assert_eq!(config.gradient_clip_threshold, 0.5);
    }

    #[test]
    fn test_predefined_configs() {
        let single = DistributedConfig::single_node();
        assert_eq!(single.world_size, 1);
        assert_eq!(single.backend, DistributedBackend::None);

        let nccl = DistributedConfig::nccl_multinode(8, 2);
        assert_eq!(nccl.world_size, 8);
        assert_eq!(nccl.rank, 2);
        assert_eq!(nccl.backend, DistributedBackend::Nccl);

        let bandwidth_opt = DistributedConfig::bandwidth_optimized(4, 1);
        assert_eq!(
            bandwidth_opt.communication_pattern,
            CommunicationPattern::Ring
        );
        assert_eq!(
            bandwidth_opt.compression,
            CompressionStrategy::Sparsification
        );
    }

    #[test]
    fn test_config_utilities() {
        let config = DistributedConfig::nccl_multinode(4, 0);
        assert!(config.is_master());
        assert!(config.bandwidth_multiplier() > 0.0);
        assert!(config.summary().contains("nccl"));
    }
}
