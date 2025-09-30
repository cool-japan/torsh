//! Core Data Structures for Computation Graph Optimization
//!
//! This module defines the fundamental types, enums, and configurations
//! used throughout the computation graph optimization system.

use parking_lot::Mutex;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use torsh_core::error::Result;

/// Node identifier type
pub type NodeId = usize;

/// Graph node representing a computation operation
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique node identifier
    pub id: NodeId,
    /// Operation name
    pub op_name: String,
    /// Input tensors
    pub inputs: Vec<NodeId>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Whether this node requires gradients
    pub requires_grad: bool,
    /// Execution priority (higher = execute first)
    pub priority: i32,
    /// Memory usage estimate in bytes
    pub memory_usage: usize,
    /// Computation cost estimate (arbitrary units)
    pub compute_cost: f64,
    /// Whether this node can be executed in-place
    pub can_execute_in_place: bool,
    /// Whether this node's result can be recomputed if needed
    pub can_recompute: bool,
}

/// Computation graph with optimization capabilities
pub struct OptimizedGraph {
    /// Internal graph representation
    pub graph: DiGraph<GraphNode, ()>,
    /// Node lookup by ID
    pub node_lookup: HashMap<NodeId, NodeIndex>,
    /// Execution order cache
    pub execution_order: Arc<RwLock<Option<Vec<NodeId>>>>,
    /// Memory usage tracking
    pub memory_tracker: Arc<Mutex<MemoryTracker>>,
    /// Optimization configuration
    pub config: GraphOptConfig,
    /// Graph statistics
    pub stats: Arc<RwLock<GraphStats>>,
    /// Compressed checkpoints storage
    pub compressed_checkpoints: Arc<Mutex<HashMap<NodeId, CompressedCheckpoint>>>,
    /// Compression statistics
    pub compression_stats: Arc<RwLock<CompressionStats>>,
    /// Nested checkpoints storage
    pub nested_checkpoints: Arc<Mutex<HashMap<NodeId, NestedCheckpoint>>>,
    /// Nested checkpointing configuration
    pub nested_config: NestedCheckpointConfig,
}

/// Graph optimization configuration
#[derive(Debug, Clone)]
pub struct GraphOptConfig {
    /// Enable operator fusion
    pub enable_fusion: bool,
    /// Enable dead code elimination
    pub enable_dce: bool,
    /// Enable common subexpression elimination
    pub enable_cse: bool,
    /// Enable memory planning
    pub enable_memory_planning: bool,
    /// Enable parallel execution
    pub enable_parallel_execution: bool,
    /// Maximum memory budget in bytes
    pub memory_budget: usize,
    /// Enable gradient checkpointing
    pub enable_checkpointing: bool,
    /// Checkpointing strategy
    pub checkpoint_strategy: CheckpointStrategy,
    /// Enable checkpoint compression
    pub enable_checkpoint_compression: bool,
    /// Checkpoint compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    /// Compression quality (0.0 = maximum compression, 1.0 = maximum quality)
    pub compression_quality: f32,
}

impl Default for GraphOptConfig {
    fn default() -> Self {
        Self {
            enable_fusion: true,
            enable_dce: true,
            enable_cse: true,
            enable_memory_planning: true,
            enable_parallel_execution: true,
            memory_budget: 1024 * 1024 * 1024, // 1GB
            enable_checkpointing: true,
            checkpoint_strategy: CheckpointStrategy::Intelligent,
            enable_checkpoint_compression: true,
            compression_algorithm: CompressionAlgorithm::LZ4,
            compression_quality: 0.8,
        }
    }
}

/// Checkpointing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointStrategy {
    /// No checkpointing
    None,
    /// Checkpoint every N operations
    EveryN(usize),
    /// Adaptive checkpointing based on memory usage
    Adaptive,
    /// Checkpoint based on computation cost
    CostBased,
    /// Intelligent checkpointing using memory/compute analysis with ML-inspired heuristics
    Intelligent,
    /// Nested checkpointing with hierarchical structure
    Nested,
    /// Hybrid strategy combining multiple approaches
    Hybrid,
}

/// Checkpoint compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// LZ4 compression (fast, moderate compression)
    LZ4,
    /// Snappy compression (very fast, moderate compression)
    Snappy,
    /// ZSTD compression (configurable speed/compression trade-off)
    ZSTD,
    /// Brotli compression (slow, high compression)
    Brotli,
    /// Custom float quantization
    FloatQuantization,
    /// Adaptive compression (chooses best algorithm based on data)
    Adaptive,
}

/// Memory usage tracker
#[derive(Debug, Clone, Default)]
pub struct MemoryTracker {
    /// Current memory usage per node
    pub node_memory: HashMap<NodeId, usize>,
    /// Peak memory usage
    pub peak_memory: usize,
    /// Current total memory usage
    pub current_memory: usize,
    /// Memory allocation timeline
    pub memory_timeline: Vec<(NodeId, usize, bool)>, // (node_id, size, is_allocation)
}

/// Compressed checkpoint data
#[derive(Debug, Clone)]
pub struct CompressedCheckpoint {
    /// Node identifier this checkpoint belongs to
    pub node_id: NodeId,
    /// Original tensor shape
    pub original_shape: Vec<usize>,
    /// Original data type
    pub dtype: String,
    /// Compressed data
    pub compressed_data: Vec<u8>,
    /// Compression algorithm used
    pub algorithm: CompressionAlgorithm,
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio (compressed_size / original_size)
    pub compression_ratio: f32,
    /// Compression quality setting used
    pub quality: f32,
}

/// Checkpoint compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Total checkpoints compressed
    pub total_checkpoints: usize,
    /// Total original size in bytes
    pub total_original_size: usize,
    /// Total compressed size in bytes
    pub total_compressed_size: usize,
    /// Average compression ratio
    pub average_compression_ratio: f32,
    /// Total compression time in milliseconds
    pub total_compression_time_ms: u64,
    /// Total decompression time in milliseconds
    pub total_decompression_time_ms: u64,
}

/// Nested checkpoint level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CheckpointLevel {
    /// Root level checkpoint (highest level)
    Root = 0,
    /// Level 1 checkpoint
    Level1 = 1,
    /// Level 2 checkpoint
    Level2 = 2,
    /// Level 3 checkpoint
    Level3 = 3,
}

/// Nested checkpoint data with hierarchical structure
#[derive(Debug, Clone)]
pub struct NestedCheckpoint {
    /// Base checkpoint data
    pub base: CompressedCheckpoint,
    /// Checkpoint level in the hierarchy
    pub level: CheckpointLevel,
    /// Parent checkpoint ID (None for root level)
    pub parent_id: Option<NodeId>,
    /// Child checkpoint IDs
    pub child_ids: Vec<NodeId>,
    /// Memory region covered by this checkpoint
    pub memory_region: MemoryRegion,
    /// Sub-graph nodes covered by this checkpoint
    pub subgraph_nodes: Vec<NodeId>,
    /// Recomputation cost for this checkpoint level
    pub recomputation_cost: f64,
    /// Whether this checkpoint can be subdivided further
    pub can_subdivide: bool,
}

/// Memory region covered by a checkpoint
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Start position in execution order
    pub start_position: usize,
    /// End position in execution order
    pub end_position: usize,
    /// Total memory usage in this region
    pub total_memory: usize,
    /// Peak memory usage in this region
    pub peak_memory: usize,
    /// Number of operations in this region
    pub operation_count: usize,
}

/// Nested checkpointing configuration
#[derive(Debug, Clone)]
pub struct NestedCheckpointConfig {
    /// Maximum nesting depth
    pub max_depth: usize,
    /// Minimum region size for subdivision
    pub min_region_size: usize,
    /// Memory threshold for creating nested checkpoints
    pub memory_threshold: usize,
    /// Enable adaptive subdivision based on memory pressure
    pub enable_adaptive_subdivision: bool,
    /// Compression algorithms per level
    pub compression_per_level: Vec<CompressionAlgorithm>,
}

impl Default for NestedCheckpointConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            min_region_size: 10,
            memory_threshold: 100 * 1024 * 1024, // 100MB
            enable_adaptive_subdivision: true,
            compression_per_level: vec![
                CompressionAlgorithm::LZ4,               // Root level: fast compression
                CompressionAlgorithm::ZSTD,              // Level 1: balanced
                CompressionAlgorithm::Brotli,            // Level 2: high compression
                CompressionAlgorithm::FloatQuantization, // Level 3: maximum compression
            ],
        }
    }
}

/// Node information for checkpoint placement analysis
#[derive(Debug, Clone)]
pub struct CheckpointNodeInfo {
    /// Node identifier
    pub id: NodeId,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Computation cost (arbitrary units)
    pub compute_cost: f64,
    /// Whether this node can be recomputed if needed
    pub can_recompute: bool,
    /// Ratio of memory usage to computation cost (lower is better for checkpointing)
    pub memory_to_compute_ratio: f64,
}

/// Extended node analysis for intelligent checkpoint placement
#[derive(Debug, Clone)]
pub struct CheckpointNodeAnalysis {
    /// Node identifier
    pub node_id: NodeId,
    /// Position in execution order
    pub position: usize,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Computation cost (arbitrary units)
    pub compute_cost: f64,
    /// Cost to recompute this node
    pub recomputation_cost: f64,
    /// Memory efficiency (memory/compute ratio)
    pub memory_efficiency: f64,
    /// Position factor (0.0 at start, 1.0 at end)
    pub position_factor: f64,
    /// Graph centrality score
    pub centrality_score: f64,
    /// Number of input dependencies
    pub input_degree: usize,
    /// Number of output dependencies
    pub output_degree: usize,
    /// Whether this node can be recomputed
    pub can_recompute: bool,
}

/// Graph execution statistics
#[derive(Debug, Clone, Default)]
pub struct GraphStats {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of optimized nodes
    pub optimized_nodes: usize,
    /// Total execution time
    pub total_execution_time_ms: u64,
    /// Peak memory usage
    pub peak_memory_bytes: usize,
    /// Number of fused operations
    pub fused_operations: usize,
    /// Number of eliminated nodes
    pub eliminated_nodes: usize,
    /// Parallel execution efficiency
    pub parallel_efficiency: f64,
}

/// Memory information for external querying
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Current memory usage
    pub current_usage: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Number of nodes with memory tracking
    pub node_count: usize,
    /// Memory efficiency (current / budget)
    pub memory_efficiency: f64,
}

/// Optimization benchmark results
#[derive(Debug, Clone)]
pub struct OptimizationBenchmark {
    /// Time before optimization (ms)
    pub time_before_ms: u64,
    /// Time after optimization (ms)
    pub time_after_ms: u64,
    /// Memory before optimization (bytes)
    pub memory_before_bytes: usize,
    /// Memory after optimization (bytes)
    pub memory_after_bytes: usize,
    /// Number of operations before optimization
    pub ops_before: usize,
    /// Number of operations after optimization
    pub ops_after: usize,
    /// Speedup ratio (time_before / time_after)
    pub speedup_ratio: f64,
    /// Memory reduction ratio
    pub memory_reduction_ratio: f64,
    /// Operation reduction ratio
    pub operation_reduction_ratio: f64,
}

/// Type alias for optimization results
pub type OptResult<T> = Result<T>;
