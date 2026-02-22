//! Configuration types and validation for 3D parallelism
//!
//! This module contains all configuration types, validation logic,
//! and ranking systems for coordinating 3D parallelism operations.

use crate::{TorshDistributedError, TorshResult};
use std::collections::HashMap;

/// Configuration for 3D parallelism (Data, Tensor, Pipeline)
#[derive(Debug, Clone)]
pub struct ThreeDParallelismConfig {
    /// Data parallel dimension size
    pub dp_size: usize,
    /// Tensor parallel dimension size
    pub tp_size: usize,
    /// Pipeline parallel dimension size
    pub pp_size: usize,
    /// Total number of layers in the model
    pub num_layers: usize,
    /// Micro-batch size for pipeline parallelism
    pub micro_batch_size: usize,
    /// Memory optimization strategy
    pub memory_strategy: MemoryOptimizationStrategy,
    /// Communication optimization strategy
    pub comm_strategy: CommunicationStrategy,
    /// Whether to enable gradient checkpointing
    pub enable_gradient_checkpointing: bool,
    /// Whether to enable mixed precision training
    pub enable_mixed_precision: bool,
    /// Pipeline schedule type
    pub pipeline_schedule: PipelineSchedule,
    /// Maximum memory usage per device (in GB)
    pub max_memory_per_device: f32,
    /// Communication timeout in milliseconds
    pub communication_timeout_ms: u64,
}

impl Default for ThreeDParallelismConfig {
    fn default() -> Self {
        Self {
            dp_size: 1,
            tp_size: 1,
            pp_size: 1,
            num_layers: 24,
            micro_batch_size: 1,
            memory_strategy: MemoryOptimizationStrategy::Standard,
            comm_strategy: CommunicationStrategy::AllReduce,
            enable_gradient_checkpointing: false,
            enable_mixed_precision: false,
            pipeline_schedule: PipelineSchedule::Interleaved,
            max_memory_per_device: 8.0,
            communication_timeout_ms: 30000,
        }
    }
}

impl ThreeDParallelismConfig {
    /// Validate configuration against available world size
    pub fn validate(&self, world_size: usize) -> TorshResult<()> {
        let expected_world_size = self.dp_size * self.tp_size * self.pp_size;
        if expected_world_size != world_size {
            return Err(TorshDistributedError::InvalidArgument {
                arg: "world_size".to_string(),
                expected: format!("{} devices", expected_world_size),
                reason: format!(
                    "3D parallelism configuration mismatch: expected {} devices ({}*{}*{}), got {}",
                    expected_world_size, self.dp_size, self.tp_size, self.pp_size, world_size
                ),
            });
        }

        if self.num_layers % self.pp_size != 0 {
            return Err(TorshDistributedError::InvalidArgument {
                arg: "num_layers".to_string(),
                expected: format!("divisible by {}", self.pp_size),
                reason: format!(
                    "Number of layers ({}) must be divisible by pipeline parallel size ({})",
                    self.num_layers, self.pp_size
                ),
            });
        }

        if self.micro_batch_size == 0 {
            return Err(TorshDistributedError::invalid_argument(
                "micro_batch_size",
                "greater than 0",
                "Micro-batch size must be greater than 0",
            ));
        }

        Ok(())
    }

    /// Get number of layers per pipeline stage
    pub fn layers_per_stage(&self) -> usize {
        self.num_layers / self.pp_size
    }

    /// Calculate memory requirements per device
    pub fn memory_requirements(&self) -> MemoryRequirements {
        let layers_per_stage = self.layers_per_stage();
        let model_memory_per_layer = 1024.0; // MB per layer (rough estimate)

        let model_memory = layers_per_stage as f32 * model_memory_per_layer / self.tp_size as f32;
        let activation_memory = match self.memory_strategy {
            MemoryOptimizationStrategy::Basic => model_memory * 2.0,
            MemoryOptimizationStrategy::Standard => model_memory * 1.5,
            MemoryOptimizationStrategy::Aggressive => model_memory * 1.2,
            MemoryOptimizationStrategy::Extreme => model_memory * 1.0,
        };

        let optimizer_memory = model_memory * 2.0; // Adam optimizer
        let total_memory = model_memory + activation_memory + optimizer_memory;

        MemoryRequirements {
            model_memory_mb: model_memory,
            activation_memory_mb: activation_memory,
            optimizer_memory_mb: optimizer_memory,
            total_memory_mb: total_memory,
        }
    }
}

/// Memory optimization strategies for 3D parallelism
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryOptimizationStrategy {
    /// Basic memory management with minimal optimizations
    Basic,
    /// Standard memory management with gradient checkpointing
    Standard,
    /// Aggressive memory optimization with activation recomputation
    Aggressive,
    /// Extreme memory optimization with disk offloading
    Extreme,
}

/// Communication optimization strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CommunicationStrategy {
    /// Standard all-reduce communication
    AllReduce,
    /// Hierarchical all-reduce with local reduction first
    HierarchicalAllReduce,
    /// Ring-based all-reduce for better bandwidth utilization
    RingAllReduce,
    /// Tree-based all-reduce for latency optimization
    TreeAllReduce,
    /// Adaptive strategy that switches based on message size
    Adaptive,
}

/// Pipeline scheduling strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineSchedule {
    /// Simple round-robin scheduling
    RoundRobin,
    /// Interleaved scheduling for better pipeline utilization
    Interleaved,
    /// GPipe scheduling with micro-batching
    GPipe,
    /// 1F1B (One Forward One Backward) scheduling
    OneForwardOneBackward,
}

/// Memory requirement breakdown
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    pub model_memory_mb: f32,
    pub activation_memory_mb: f32,
    pub optimizer_memory_mb: f32,
    pub total_memory_mb: f32,
}

/// Rank mapping for 3D parallelism coordinates
#[derive(Debug, Clone)]
pub struct RankMapping {
    /// Global rank in the distributed system
    pub global_rank: usize,
    /// Data parallel rank (0 to dp_size-1)
    pub dp_rank: usize,
    /// Tensor parallel rank (0 to tp_size-1)
    pub tp_rank: usize,
    /// Pipeline parallel rank (0 to pp_size-1)
    pub pp_rank: usize,
    /// Local rank on the node
    pub local_rank: usize,
    /// World size
    pub world_size: usize,
    /// 3D parallelism configuration
    pub config: ThreeDParallelismConfig,
}

impl RankMapping {
    /// Create new rank mapping from global rank and configuration
    pub fn new(config: &ThreeDParallelismConfig, global_rank: usize) -> Self {
        let world_size = config.dp_size * config.tp_size * config.pp_size;

        // Calculate 3D coordinates from global rank
        // Layout: [dp][tp][pp] with pp as the fastest changing dimension
        let pp_rank = global_rank % config.pp_size;
        let tp_rank = (global_rank / config.pp_size) % config.tp_size;
        let dp_rank = global_rank / (config.pp_size * config.tp_size);

        let local_rank = global_rank % 8; // Assuming 8 GPUs per node

        Self {
            global_rank,
            dp_rank,
            tp_rank,
            pp_rank,
            local_rank,
            world_size,
            config: config.clone(),
        }
    }

    /// Get global rank from 3D coordinates
    pub fn from_3d_coords(
        config: &ThreeDParallelismConfig,
        dp_rank: usize,
        tp_rank: usize,
        pp_rank: usize,
    ) -> usize {
        dp_rank * (config.tp_size * config.pp_size) + tp_rank * config.pp_size + pp_rank
    }

    /// Check if this rank is the first in the data parallel group
    pub fn is_dp_head(&self) -> bool {
        self.dp_rank == 0
    }

    /// Check if this rank is the first in the tensor parallel group
    pub fn is_tp_head(&self) -> bool {
        self.tp_rank == 0
    }

    /// Check if this rank is the first in the pipeline parallel group
    pub fn is_pp_head(&self) -> bool {
        self.pp_rank == 0
    }

    /// Check if this rank is the last in the pipeline parallel group
    pub fn is_pp_tail(&self) -> bool {
        self.pp_rank == self.config.pp_size - 1
    }

    /// Get ranks in the same data parallel group
    pub fn dp_group_ranks(&self) -> Vec<usize> {
        (0..self.config.dp_size)
            .map(|dp| Self::from_3d_coords(&self.config, dp, self.tp_rank, self.pp_rank))
            .collect()
    }

    /// Get ranks in the same tensor parallel group
    pub fn tp_group_ranks(&self) -> Vec<usize> {
        (0..self.config.tp_size)
            .map(|tp| Self::from_3d_coords(&self.config, self.dp_rank, tp, self.pp_rank))
            .collect()
    }

    /// Get ranks in the same pipeline parallel group
    pub fn pp_group_ranks(&self) -> Vec<usize> {
        (0..self.config.pp_size)
            .map(|pp| Self::from_3d_coords(&self.config, self.dp_rank, self.tp_rank, pp))
            .collect()
    }

    /// Get the next rank in the pipeline
    pub fn next_pp_rank(&self) -> Option<usize> {
        if self.pp_rank < self.config.pp_size - 1 {
            Some(Self::from_3d_coords(
                &self.config,
                self.dp_rank,
                self.tp_rank,
                self.pp_rank + 1,
            ))
        } else {
            None
        }
    }

    /// Get the previous rank in the pipeline
    pub fn prev_pp_rank(&self) -> Option<usize> {
        if self.pp_rank > 0 {
            Some(Self::from_3d_coords(
                &self.config,
                self.dp_rank,
                self.tp_rank,
                self.pp_rank - 1,
            ))
        } else {
            None
        }
    }
}

/// Process group identifiers for different parallelism dimensions
#[derive(Debug, Clone)]
pub struct ProcessGroupIds {
    /// Data parallel process groups
    pub dp_groups: HashMap<(usize, usize), String>, // (tp_rank, pp_rank) -> group_id
    /// Tensor parallel process groups
    pub tp_groups: HashMap<(usize, usize), String>, // (dp_rank, pp_rank) -> group_id
    /// Pipeline parallel process groups
    pub pp_groups: HashMap<(usize, usize), String>, // (dp_rank, tp_rank) -> group_id
}

impl ProcessGroupIds {
    /// Create process group identifiers for a given configuration
    pub fn new(config: &ThreeDParallelismConfig) -> Self {
        let mut dp_groups = HashMap::new();
        let mut tp_groups = HashMap::new();
        let mut pp_groups = HashMap::new();

        // Create DP groups: one group for each (tp_rank, pp_rank) combination
        for tp_rank in 0..config.tp_size {
            for pp_rank in 0..config.pp_size {
                let group_id = format!("dp_group_tp{}_pp{}", tp_rank, pp_rank);
                dp_groups.insert((tp_rank, pp_rank), group_id);
            }
        }

        // Create TP groups: one group for each (dp_rank, pp_rank) combination
        for dp_rank in 0..config.dp_size {
            for pp_rank in 0..config.pp_size {
                let group_id = format!("tp_group_dp{}_pp{}", dp_rank, pp_rank);
                tp_groups.insert((dp_rank, pp_rank), group_id);
            }
        }

        // Create PP groups: one group for each (dp_rank, tp_rank) combination
        for dp_rank in 0..config.dp_size {
            for tp_rank in 0..config.tp_size {
                let group_id = format!("pp_group_dp{}_tp{}", dp_rank, tp_rank);
                pp_groups.insert((dp_rank, tp_rank), group_id);
            }
        }

        Self {
            dp_groups,
            tp_groups,
            pp_groups,
        }
    }

    /// Get data parallel group ID for given coordinates
    pub fn get_dp_group_id(&self, tp_rank: usize, pp_rank: usize) -> Option<&String> {
        self.dp_groups.get(&(tp_rank, pp_rank))
    }

    /// Get tensor parallel group ID for given coordinates
    pub fn get_tp_group_id(&self, dp_rank: usize, pp_rank: usize) -> Option<&String> {
        self.tp_groups.get(&(dp_rank, pp_rank))
    }

    /// Get pipeline parallel group ID for given coordinates
    pub fn get_pp_group_id(&self, dp_rank: usize, tp_rank: usize) -> Option<&String> {
        self.pp_groups.get(&(dp_rank, tp_rank))
    }
}
