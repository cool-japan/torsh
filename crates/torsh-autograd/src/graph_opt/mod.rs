//! Computation Graph Optimization Framework
//!
//! This module provides advanced optimizations for computation graph construction,
//! traversal, and memory management to improve autograd performance. The framework
//! is organized into specialized modules for maintainability and extensibility.
//!
//! # Architecture
//!
//! The optimization system is built around several specialized modules:
//!
//! - **`graph_types`**: Core data structures, enums, and configurations
//! - **`graph_builder`**: Graph construction and node/edge management
//! - **`optimization`**: Core optimization passes (DCE, CSE, fusion)
//! - **`memory_planning`**: Memory usage analysis and lifetime planning
//!
//! Additional modules for advanced features:
//! - **`checkpoint_planning`**: Gradient checkpointing strategies
//! - **`execution`**: Sequential and parallel execution engines
//! - **`compression`**: Checkpoint compression algorithms
//! - **`nested_checkpointing`**: Hierarchical checkpointing
//! - **`testing`**: Utilities for testing and benchmarking
//!
//! # Usage Examples
//!
//! ```rust,ignore
//! use torsh_autograd::graph_opt::{OptimizedGraph, GraphOptConfig};
//!
//! # fn example() -> torsh_core::error::Result<()> {
//! // Create optimized graph with default configuration
//! let config = GraphOptConfig::default();
//! let mut graph = OptimizedGraph::new(config);
//!
//! // Build computation graph
//! // ... add nodes and edges ...
//!
//! // Apply optimizations
//! graph.optimize()?;
//!
//! // Execute with memory planning
//! graph.execute()?;
//! # Ok(())
//! # }
//! ```

// Core modules (implemented)
pub mod graph_builder;
pub mod graph_types;
pub mod memory_planning;
pub mod optimization;

// Advanced modules (to be implemented based on need)
// pub mod checkpoint_planning;
// pub mod execution;
// pub mod compression;
// pub mod nested_checkpointing;
// pub mod testing;

// Re-export core types and functionality
pub use graph_types::{
    CheckpointLevel,
    CheckpointNodeAnalysis,
    // Analysis types
    CheckpointNodeInfo,
    CheckpointStrategy,
    // Checkpoint types
    CompressedCheckpoint,
    CompressionAlgorithm,
    CompressionStats,
    GraphNode,
    // Configuration and strategies
    GraphOptConfig,
    GraphStats,

    MemoryInfo,
    MemoryRegion,

    // Memory and tracking types
    MemoryTracker,
    NestedCheckpoint,
    NestedCheckpointConfig,

    // Core graph types
    NodeId,
    // Result type alias
    OptResult,
    OptimizationBenchmark,

    OptimizedGraph,
};

// Re-export memory planning types
pub use memory_planning::{MemoryAnalysis, MemoryRequirements};

// Convenience type aliases
pub type GraphResult<T> = torsh_core::error::Result<T>;
pub type NodeIndex = petgraph::graph::NodeIndex;

/// Create a new OptimizedGraph with default configuration
///
/// # Returns
/// * `OptimizedGraph` - New graph instance ready for optimization
pub fn new_graph() -> OptimizedGraph {
    OptimizedGraph::new(GraphOptConfig::default())
}

/// Create a new OptimizedGraph with custom configuration
///
/// # Arguments
/// * `config` - Custom optimization configuration
///
/// # Returns
/// * `OptimizedGraph` - New graph instance with custom configuration
pub fn new_graph_with_config(config: GraphOptConfig) -> OptimizedGraph {
    OptimizedGraph::new(config)
}

/// Create a configuration optimized for speed
///
/// # Returns
/// * `GraphOptConfig` - Configuration optimized for fastest execution
pub fn speed_optimized_config() -> GraphOptConfig {
    GraphOptConfig {
        enable_fusion: true,
        enable_dce: true,
        enable_cse: true,
        enable_memory_planning: false, // Skip for speed
        enable_parallel_execution: true,
        memory_budget: 2 * 1024 * 1024 * 1024, // 2GB
        enable_checkpointing: false,           // Skip for speed
        checkpoint_strategy: CheckpointStrategy::None,
        enable_checkpoint_compression: false,
        compression_algorithm: CompressionAlgorithm::None,
        compression_quality: 0.0,
    }
}

/// Create a configuration optimized for memory efficiency
///
/// # Returns
/// * `GraphOptConfig` - Configuration optimized for minimal memory usage
pub fn memory_optimized_config() -> GraphOptConfig {
    GraphOptConfig {
        enable_fusion: true,
        enable_dce: true,
        enable_cse: true,
        enable_memory_planning: true,
        enable_parallel_execution: false, // Reduce memory overhead
        memory_budget: 512 * 1024 * 1024, // 512MB
        enable_checkpointing: true,
        checkpoint_strategy: CheckpointStrategy::Adaptive,
        enable_checkpoint_compression: true,
        compression_algorithm: CompressionAlgorithm::ZSTD,
        compression_quality: 0.9,
    }
}

/// Create a configuration balanced for both speed and memory
///
/// # Returns
/// * `GraphOptConfig` - Balanced configuration
pub fn balanced_config() -> GraphOptConfig {
    GraphOptConfig::default()
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use super::{
        balanced_config, memory_optimized_config, new_graph, new_graph_with_config,
        speed_optimized_config, GraphNode, GraphOptConfig, GraphResult, GraphStats, MemoryInfo,
        NodeId, OptimizedGraph,
    };
}
