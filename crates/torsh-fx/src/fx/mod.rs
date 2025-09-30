//! FX Graph system - unified interface
//!
//! This module provides a comprehensive graph transformation framework for ToRSh.
//! The modular architecture separates concerns while maintaining a clean, unified interface.

// Module declarations
pub mod analysis;
pub mod constructors;
pub mod core;
pub mod operations;
pub mod serialization;
pub mod types;

// Re-export core types for convenience
pub use types::{Edge, GraphStats, MemoryEstimate, Node};

// Re-export serialization types
pub use serialization::{SerializableGraph, TorshResult};

// FX Graph is defined in core.rs but the implementation is spread across modules
use petgraph::graph::{Graph, NodeIndex};
use types::{Edge as FxEdge, Node as FxNode};

/// FX Graph representation
#[derive(Debug, Clone)]
pub struct FxGraph {
    pub graph: Graph<FxNode, FxEdge>,
    pub inputs: Vec<NodeIndex>,
    pub outputs: Vec<NodeIndex>,
}

// All functionality is implemented through the various modules
// via impl blocks for FxGraph in each module
