//! Core types and data structures for the FX graph system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    Input(String),
    Call(String, Vec<String>),
    Output,
    /// Conditional node: if condition, then_branch, else_branch
    Conditional {
        condition: String,
        then_branch: Vec<String>,
        else_branch: Vec<String>,
    },
    /// Loop node: condition, body, loop_vars
    Loop {
        condition: String,
        body: Vec<String>,
        loop_vars: Vec<String>,
    },
    /// Merge node for control flow convergence
    Merge {
        inputs: Vec<String>,
    },
    /// GetAttr node for attribute access
    GetAttr {
        target: String,
        attr: String,
    },
}

/// Graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub name: String,
}

/// Detailed statistics about graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub input_count: usize,
    pub output_count: usize,
    pub max_fanout: usize,
    pub max_fanin: usize,
    pub average_fanout: f64,
    pub depth: usize,
    pub is_linear: bool,
    pub is_pipeline: bool,
    pub has_cycles: bool,
    pub complexity_score: f64,
    pub node_type_distribution: HashMap<String, usize>,
    pub operation_distribution: HashMap<String, usize>,
    pub fanout_distribution: HashMap<usize, usize>,
    pub fanin_distribution: HashMap<usize, usize>,
}

/// Memory usage estimation for a graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub total_bytes: usize,
    pub nodes_bytes: usize,
    pub edges_bytes: usize,
    pub metadata_bytes: usize,
    pub overhead_bytes: usize,
    pub estimated_peak_multiplier: f64,
}
