//! Tensor Network Representations for Specialized Applications
//!
//! This module provides a comprehensive tensor network framework for quantum computing,
//! physics simulations, and efficient computation of certain neural network architectures.
//!
//! Tensor networks represent high-dimensional tensors as networks of lower-dimensional
//! tensors connected by indices, enabling efficient computation and storage.
//!
//! # Supported Structures
//!
//! - **Matrix Product States (MPS)**: 1D tensor networks for quantum states
//! - **Projected Entangled Pair States (PEPS)**: 2D tensor networks
//! - **Tree Tensor Networks (TTN)**: Hierarchical tensor networks
//! - **Tensor Train (TT)**: Decomposition for high-dimensional tensors
//!
//! # Examples
//!
//! ```
//! use torsh_core::tensor_network::*;
//!
//! // Create a tensor network node
//! let node = TensorNode::new(0, vec![2, 3, 4]);
//! assert_eq!(node.rank(), 3);
//!
//! // Create an MPS (Matrix Product State)
//! let mps = MatrixProductState::new(vec![2, 2, 2], 4);
//! assert_eq!(mps.num_sites(), 3);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

/// Unique identifier for a tensor network node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

impl NodeId {
    /// Create a new node ID
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    /// Get the underlying ID
    pub fn id(&self) -> usize {
        self.0
    }
}

/// Unique identifier for a tensor network edge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub usize);

impl EdgeId {
    /// Create a new edge ID
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    /// Get the underlying ID
    pub fn id(&self) -> usize {
        self.0
    }
}

/// Index dimension for tensor network edges
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexDim(pub usize);

impl IndexDim {
    /// Create a new index dimension
    pub fn new(dim: usize) -> Self {
        Self(dim)
    }

    /// Get the dimension size
    pub fn dim(&self) -> usize {
        self.0
    }
}

/// Tensor network node representing a tensor with named indices
#[derive(Debug, Clone)]
pub struct TensorNode {
    /// Unique node identifier
    pub id: NodeId,
    /// Dimensions of each index
    pub index_dims: Vec<usize>,
    /// Labels for each index (optional)
    pub index_labels: Vec<String>,
    /// Whether this is an open (external) or closed (internal) node
    pub is_open: bool,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl TensorNode {
    /// Create a new tensor node
    pub fn new(id: usize, index_dims: Vec<usize>) -> Self {
        let num_indices = index_dims.len();
        Self {
            id: NodeId::new(id),
            index_dims,
            index_labels: (0..num_indices).map(|i| format!("idx_{}", i)).collect(),
            is_open: true,
            metadata: HashMap::new(),
        }
    }

    /// Create a node with custom index labels
    pub fn with_labels(id: usize, index_dims: Vec<usize>, labels: Vec<String>) -> Self {
        assert_eq!(
            index_dims.len(),
            labels.len(),
            "Number of dimensions must match number of labels"
        );
        Self {
            id: NodeId::new(id),
            index_dims,
            index_labels: labels,
            is_open: true,
            metadata: HashMap::new(),
        }
    }

    /// Get the rank (number of indices) of this node
    pub fn rank(&self) -> usize {
        self.index_dims.len()
    }

    /// Get the total number of elements in this tensor
    pub fn numel(&self) -> usize {
        self.index_dims.iter().product()
    }

    /// Get the dimension of a specific index
    pub fn index_dim(&self, idx: usize) -> Option<usize> {
        self.index_dims.get(idx).copied()
    }

    /// Get the label of a specific index
    pub fn index_label(&self, idx: usize) -> Option<&str> {
        self.index_labels.get(idx).map(|s| s.as_str())
    }

    /// Mark this node as closed (internal)
    pub fn close(&mut self) {
        self.is_open = false;
    }

    /// Mark this node as open (external)
    pub fn open(&mut self) {
        self.is_open = true;
    }

    /// Add custom metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Edge connecting two tensor network nodes
#[derive(Debug, Clone)]
pub struct TensorEdge {
    /// Unique edge identifier
    pub id: EdgeId,
    /// Source node and index
    pub source: (NodeId, usize),
    /// Target node and index
    pub target: (NodeId, usize),
    /// Dimension of the shared index
    pub bond_dim: usize,
    /// Custom edge label
    pub label: String,
}

impl TensorEdge {
    /// Create a new tensor edge
    pub fn new(
        id: usize,
        source: (NodeId, usize),
        target: (NodeId, usize),
        bond_dim: usize,
    ) -> Self {
        Self {
            id: EdgeId::new(id),
            source,
            target,
            bond_dim,
            label: format!("edge_{}", id),
        }
    }

    /// Create an edge with a custom label
    pub fn with_label(
        id: usize,
        source: (NodeId, usize),
        target: (NodeId, usize),
        bond_dim: usize,
        label: impl Into<String>,
    ) -> Self {
        Self {
            id: EdgeId::new(id),
            source,
            target,
            bond_dim,
            label: label.into(),
        }
    }

    /// Check if this edge connects to a specific node
    pub fn connects_to(&self, node_id: NodeId) -> bool {
        self.source.0 == node_id || self.target.0 == node_id
    }

    /// Get the other node connected by this edge
    pub fn other_node(&self, node_id: NodeId) -> Option<NodeId> {
        if self.source.0 == node_id {
            Some(self.target.0)
        } else if self.target.0 == node_id {
            Some(self.source.0)
        } else {
            None
        }
    }
}

/// Tensor network graph structure
#[derive(Debug, Clone)]
pub struct TensorNetwork {
    /// All nodes in the network
    pub nodes: HashMap<NodeId, TensorNode>,
    /// All edges in the network
    pub edges: HashMap<EdgeId, TensorEdge>,
    /// Adjacency list for efficient traversal
    adjacency: HashMap<NodeId, Vec<EdgeId>>,
    /// Counter for generating unique node IDs
    next_node_id: usize,
    /// Counter for generating unique edge IDs
    next_edge_id: usize,
}

impl TensorNetwork {
    /// Create a new empty tensor network
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency: HashMap::new(),
            next_node_id: 0,
            next_edge_id: 0,
        }
    }

    /// Add a node to the network
    pub fn add_node(&mut self, index_dims: Vec<usize>) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;

        let node = TensorNode::new(id, index_dims);
        let node_id = node.id;
        self.nodes.insert(node_id, node);
        self.adjacency.insert(node_id, Vec::new());

        node_id
    }

    /// Add a node with custom labels
    pub fn add_node_with_labels(&mut self, index_dims: Vec<usize>, labels: Vec<String>) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;

        let node = TensorNode::with_labels(id, index_dims, labels);
        let node_id = node.id;
        self.nodes.insert(node_id, node);
        self.adjacency.insert(node_id, Vec::new());

        node_id
    }

    /// Add an edge between two nodes
    pub fn add_edge(
        &mut self,
        source: NodeId,
        source_idx: usize,
        target: NodeId,
        target_idx: usize,
        bond_dim: usize,
    ) -> Result<EdgeId, TensorNetworkError> {
        // Validate nodes exist
        if !self.nodes.contains_key(&source) {
            return Err(TensorNetworkError::NodeNotFound(source));
        }
        if !self.nodes.contains_key(&target) {
            return Err(TensorNetworkError::NodeNotFound(target));
        }

        // Validate index dimensions match
        let source_node = &self.nodes[&source];
        let target_node = &self.nodes[&target];

        if source_idx >= source_node.rank() {
            return Err(TensorNetworkError::InvalidIndex(source, source_idx));
        }
        if target_idx >= target_node.rank() {
            return Err(TensorNetworkError::InvalidIndex(target, target_idx));
        }

        if source_node.index_dims[source_idx] != bond_dim {
            return Err(TensorNetworkError::DimensionMismatch {
                expected: source_node.index_dims[source_idx],
                found: bond_dim,
            });
        }
        if target_node.index_dims[target_idx] != bond_dim {
            return Err(TensorNetworkError::DimensionMismatch {
                expected: target_node.index_dims[target_idx],
                found: bond_dim,
            });
        }

        // Create edge
        let id = self.next_edge_id;
        self.next_edge_id += 1;

        let edge = TensorEdge::new(id, (source, source_idx), (target, target_idx), bond_dim);
        let edge_id = edge.id;

        self.edges.insert(edge_id, edge);
        self.adjacency
            .get_mut(&source)
            .expect("source node should exist in adjacency map")
            .push(edge_id);
        self.adjacency
            .get_mut(&target)
            .expect("target node should exist in adjacency map")
            .push(edge_id);

        Ok(edge_id)
    }

    /// Get the number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get all neighbors of a node
    pub fn neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.adjacency
            .get(&node_id)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|edge_id| {
                        self.edges
                            .get(edge_id)
                            .and_then(|edge| edge.other_node(node_id))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all edges connected to a node
    pub fn node_edges(&self, node_id: NodeId) -> Vec<&TensorEdge> {
        self.adjacency
            .get(&node_id)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|edge_id| self.edges.get(edge_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if the network is connected
    pub fn is_connected(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }

        let mut visited = HashSet::new();
        let start = *self
            .nodes
            .keys()
            .next()
            .expect("nodes should have at least one key after is_empty check");
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(node_id) = queue.pop_front() {
            for neighbor in self.neighbors(node_id) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        visited.len() == self.nodes.len()
    }

    /// Calculate the total bond dimension (sum of all edge bond dimensions)
    pub fn total_bond_dimension(&self) -> usize {
        self.edges.values().map(|edge| edge.bond_dim).sum()
    }

    /// Get the maximum bond dimension in the network
    pub fn max_bond_dimension(&self) -> usize {
        self.edges
            .values()
            .map(|edge| edge.bond_dim)
            .max()
            .unwrap_or(0)
    }
}

impl Default for TensorNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Matrix Product State (MPS) - 1D tensor network
#[derive(Debug, Clone)]
pub struct MatrixProductState {
    /// Physical dimensions at each site
    pub physical_dims: Vec<usize>,
    /// Bond dimensions between sites
    pub bond_dims: Vec<usize>,
    /// Underlying tensor network
    pub network: TensorNetwork,
    /// Node IDs in left-to-right order
    pub site_nodes: Vec<NodeId>,
}

impl MatrixProductState {
    /// Create a new MPS with uniform bond dimension
    pub fn new(physical_dims: Vec<usize>, bond_dim: usize) -> Self {
        let num_sites = physical_dims.len();
        let mut network = TensorNetwork::new();
        let mut site_nodes = Vec::new();

        // Create nodes for each site
        for (i, &phys_dim) in physical_dims.iter().enumerate() {
            let dims = if i == 0 {
                // First site: [physical, right_bond]
                vec![phys_dim, bond_dim]
            } else if i == num_sites - 1 {
                // Last site: [left_bond, physical]
                vec![bond_dim, phys_dim]
            } else {
                // Middle sites: [left_bond, physical, right_bond]
                vec![bond_dim, phys_dim, bond_dim]
            };

            let node_id = network.add_node(dims);
            site_nodes.push(node_id);
        }

        // Connect adjacent sites
        let bond_dims = vec![bond_dim; num_sites.saturating_sub(1)];
        for i in 0..num_sites - 1 {
            let left = site_nodes[i];
            let right = site_nodes[i + 1];

            // Connect right index of left node to left index of right node
            let right_idx = if i == 0 { 1 } else { 2 };
            let left_idx = 0;

            network
                .add_edge(left, right_idx, right, left_idx, bond_dim)
                .expect("Failed to add edge in MPS construction");
        }

        Self {
            physical_dims,
            bond_dims,
            network,
            site_nodes,
        }
    }

    /// Create an MPS with varying bond dimensions
    pub fn with_bond_dims(physical_dims: Vec<usize>, bond_dims: Vec<usize>) -> Self {
        assert_eq!(
            bond_dims.len(),
            physical_dims.len() - 1,
            "Number of bond dimensions must be one less than number of sites"
        );

        let num_sites = physical_dims.len();
        let mut network = TensorNetwork::new();
        let mut site_nodes = Vec::new();

        // Create nodes for each site
        for (i, &phys_dim) in physical_dims.iter().enumerate() {
            let dims = if i == 0 {
                vec![phys_dim, bond_dims[0]]
            } else if i == num_sites - 1 {
                vec![bond_dims[i - 1], phys_dim]
            } else {
                vec![bond_dims[i - 1], phys_dim, bond_dims[i]]
            };

            let node_id = network.add_node(dims);
            site_nodes.push(node_id);
        }

        // Connect adjacent sites
        for i in 0..num_sites - 1 {
            let left = site_nodes[i];
            let right = site_nodes[i + 1];

            let right_idx = if i == 0 { 1 } else { 2 };
            let left_idx = 0;

            network
                .add_edge(left, right_idx, right, left_idx, bond_dims[i])
                .expect("Failed to add edge in MPS construction");
        }

        Self {
            physical_dims,
            bond_dims,
            network,
            site_nodes,
        }
    }

    /// Get the number of sites in the MPS
    pub fn num_sites(&self) -> usize {
        self.physical_dims.len()
    }

    /// Get the bond dimension at a specific position
    pub fn bond_dim(&self, pos: usize) -> Option<usize> {
        self.bond_dims.get(pos).copied()
    }

    /// Get the maximum bond dimension
    pub fn max_bond_dim(&self) -> usize {
        self.bond_dims.iter().copied().max().unwrap_or(0)
    }
}

/// Projected Entangled Pair State (PEPS) - 2D tensor network
#[derive(Debug, Clone)]
pub struct ProjectedEntangledPairState {
    /// Grid dimensions (rows, cols)
    pub grid_shape: (usize, usize),
    /// Physical dimension at each site
    pub physical_dim: usize,
    /// Bond dimension for horizontal connections
    pub horizontal_bond_dim: usize,
    /// Bond dimension for vertical connections
    pub vertical_bond_dim: usize,
    /// Underlying tensor network
    pub network: TensorNetwork,
    /// Node IDs organized in grid layout
    pub grid_nodes: Vec<Vec<NodeId>>,
}

impl ProjectedEntangledPairState {
    /// Create a new PEPS with uniform bond dimensions
    pub fn new(rows: usize, cols: usize, physical_dim: usize, bond_dim: usize) -> Self {
        let mut network = TensorNetwork::new();
        let mut grid_nodes = Vec::new();

        // Create nodes in grid layout
        for i in 0..rows {
            let mut row_nodes = Vec::new();
            for j in 0..cols {
                // Determine node dimensions based on position
                let mut dims = vec![physical_dim]; // Physical index

                // Add bond indices based on position
                if i > 0 {
                    dims.push(bond_dim); // Up bond
                }
                if i < rows - 1 {
                    dims.push(bond_dim); // Down bond
                }
                if j > 0 {
                    dims.push(bond_dim); // Left bond
                }
                if j < cols - 1 {
                    dims.push(bond_dim); // Right bond
                }

                let node_id = network.add_node(dims);
                row_nodes.push(node_id);
            }
            grid_nodes.push(row_nodes);
        }

        // Connect horizontal neighbors
        for i in 0..rows {
            for j in 0..cols - 1 {
                let left = grid_nodes[i][j];
                let right = grid_nodes[i][j + 1];

                // Find right index of left node and left index of right node
                let left_right_idx = Self::get_bond_index(i, j, rows, cols, bond_dim, "right");
                let right_left_idx = Self::get_bond_index(i, j + 1, rows, cols, bond_dim, "left");

                network
                    .add_edge(left, left_right_idx, right, right_left_idx, bond_dim)
                    .expect("Failed to add horizontal edge");
            }
        }

        // Connect vertical neighbors
        for i in 0..rows - 1 {
            for j in 0..cols {
                let top = grid_nodes[i][j];
                let bottom = grid_nodes[i + 1][j];

                let top_down_idx = Self::get_bond_index(i, j, rows, cols, bond_dim, "down");
                let bottom_up_idx = Self::get_bond_index(i + 1, j, rows, cols, bond_dim, "up");

                network
                    .add_edge(top, top_down_idx, bottom, bottom_up_idx, bond_dim)
                    .expect("Failed to add vertical edge");
            }
        }

        Self {
            grid_shape: (rows, cols),
            physical_dim,
            horizontal_bond_dim: bond_dim,
            vertical_bond_dim: bond_dim,
            network,
            grid_nodes,
        }
    }

    /// Helper to get bond index based on position and direction
    fn get_bond_index(
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
        _bond_dim: usize,
        direction: &str,
    ) -> usize {
        let mut idx = 1; // Physical index is always 0

        match direction {
            "up" => {
                if i > 0 {
                    idx
                } else {
                    panic!("No up bond at row 0")
                }
            }
            "down" => {
                if i > 0 {
                    idx += 1;
                }
                if i < rows - 1 {
                    idx
                } else {
                    panic!("No down bond at last row")
                }
            }
            "left" => {
                if i > 0 {
                    idx += 1;
                }
                if i < rows - 1 {
                    idx += 1;
                }
                if j > 0 {
                    idx
                } else {
                    panic!("No left bond at col 0")
                }
            }
            "right" => {
                if i > 0 {
                    idx += 1;
                }
                if i < rows - 1 {
                    idx += 1;
                }
                if j > 0 {
                    idx += 1;
                }
                if j < cols - 1 {
                    idx
                } else {
                    panic!("No right bond at last col")
                }
            }
            _ => panic!("Invalid direction: {}", direction),
        }
    }

    /// Get the node at a specific grid position
    pub fn node_at(&self, row: usize, col: usize) -> Option<NodeId> {
        self.grid_nodes.get(row).and_then(|r| r.get(col).copied())
    }

    /// Get the total number of sites
    pub fn num_sites(&self) -> usize {
        self.grid_shape.0 * self.grid_shape.1
    }
}

/// Errors that can occur in tensor network operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorNetworkError {
    /// Node not found in network
    NodeNotFound(NodeId),
    /// Edge not found in network
    EdgeNotFound(EdgeId),
    /// Invalid index for node
    InvalidIndex(NodeId, usize),
    /// Dimension mismatch between nodes
    DimensionMismatch { expected: usize, found: usize },
    /// Network is disconnected
    DisconnectedNetwork,
    /// Invalid contraction operation
    InvalidContraction(String),
}

impl std::fmt::Display for TensorNetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "Node {} not found in network", id.0),
            Self::EdgeNotFound(id) => write!(f, "Edge {} not found in network", id.0),
            Self::InvalidIndex(node, idx) => {
                write!(f, "Invalid index {} for node {}", idx, node.0)
            }
            Self::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, found {}",
                    expected, found
                )
            }
            Self::DisconnectedNetwork => write!(f, "Tensor network is disconnected"),
            Self::InvalidContraction(msg) => write!(f, "Invalid contraction: {}", msg),
        }
    }
}

impl std::error::Error for TensorNetworkError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_node_creation() {
        let node = TensorNode::new(0, vec![2, 3, 4]);
        assert_eq!(node.id.id(), 0);
        assert_eq!(node.rank(), 3);
        assert_eq!(node.numel(), 24);
        assert_eq!(node.index_dim(0), Some(2));
        assert_eq!(node.index_dim(1), Some(3));
        assert_eq!(node.index_dim(2), Some(4));
    }

    #[test]
    fn test_tensor_node_with_labels() {
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let node = TensorNode::with_labels(0, vec![2, 3, 4], labels);
        assert_eq!(node.index_label(0), Some("a"));
        assert_eq!(node.index_label(1), Some("b"));
        assert_eq!(node.index_label(2), Some("c"));
    }

    #[test]
    fn test_tensor_node_open_close() {
        let mut node = TensorNode::new(0, vec![2, 3]);
        assert!(node.is_open);
        node.close();
        assert!(!node.is_open);
        node.open();
        assert!(node.is_open);
    }

    #[test]
    fn test_tensor_edge_creation() {
        let edge = TensorEdge::new(0, (NodeId::new(0), 0), (NodeId::new(1), 1), 5);
        assert_eq!(edge.id.id(), 0);
        assert_eq!(edge.source, (NodeId::new(0), 0));
        assert_eq!(edge.target, (NodeId::new(1), 1));
        assert_eq!(edge.bond_dim, 5);
    }

    #[test]
    fn test_tensor_edge_connects_to() {
        let edge = TensorEdge::new(0, (NodeId::new(0), 0), (NodeId::new(1), 1), 5);
        assert!(edge.connects_to(NodeId::new(0)));
        assert!(edge.connects_to(NodeId::new(1)));
        assert!(!edge.connects_to(NodeId::new(2)));
    }

    #[test]
    fn test_tensor_edge_other_node() {
        let edge = TensorEdge::new(0, (NodeId::new(0), 0), (NodeId::new(1), 1), 5);
        assert_eq!(edge.other_node(NodeId::new(0)), Some(NodeId::new(1)));
        assert_eq!(edge.other_node(NodeId::new(1)), Some(NodeId::new(0)));
        assert_eq!(edge.other_node(NodeId::new(2)), None);
    }

    #[test]
    fn test_tensor_network_add_node() {
        let mut network = TensorNetwork::new();
        let node_id = network.add_node(vec![2, 3]);
        assert_eq!(network.num_nodes(), 1);
        assert!(network.nodes.contains_key(&node_id));
    }

    #[test]
    fn test_tensor_network_add_edge() {
        let mut network = TensorNetwork::new();
        let node1 = network.add_node(vec![2, 3]);
        let node2 = network.add_node(vec![3, 4]);

        let edge_id = network.add_edge(node1, 1, node2, 0, 3).unwrap();
        assert_eq!(network.num_edges(), 1);
        assert!(network.edges.contains_key(&edge_id));
    }

    #[test]
    fn test_tensor_network_dimension_mismatch() {
        let mut network = TensorNetwork::new();
        let node1 = network.add_node(vec![2, 3]);
        let node2 = network.add_node(vec![4, 5]);

        // Try to connect indices with different dimensions
        let result = network.add_edge(node1, 1, node2, 0, 3);
        assert!(matches!(
            result,
            Err(TensorNetworkError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_tensor_network_neighbors() {
        let mut network = TensorNetwork::new();
        let node1 = network.add_node(vec![2, 2]);
        let node2 = network.add_node(vec![2, 2]);
        let node3 = network.add_node(vec![2, 2]);

        network.add_edge(node1, 0, node2, 0, 2).unwrap();
        network.add_edge(node1, 1, node3, 0, 2).unwrap();

        let neighbors = network.neighbors(node1);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&node2));
        assert!(neighbors.contains(&node3));
    }

    #[test]
    fn test_tensor_network_is_connected() {
        let mut network = TensorNetwork::new();
        let node1 = network.add_node(vec![2, 2]);
        let node2 = network.add_node(vec![2, 2]);
        let node3 = network.add_node(vec![2, 2]);

        // Initially disconnected
        assert!(!network.is_connected());

        // Connect all nodes
        network.add_edge(node1, 0, node2, 0, 2).unwrap();
        network.add_edge(node2, 1, node3, 0, 2).unwrap();

        // Now connected
        assert!(network.is_connected());
    }

    #[test]
    fn test_mps_creation() {
        let mps = MatrixProductState::new(vec![2, 2, 2], 4);
        assert_eq!(mps.num_sites(), 3);
        assert_eq!(mps.max_bond_dim(), 4);
        assert_eq!(mps.network.num_nodes(), 3);
        assert_eq!(mps.network.num_edges(), 2);
    }

    #[test]
    fn test_mps_with_bond_dims() {
        let mps = MatrixProductState::with_bond_dims(vec![2, 2, 2, 2], vec![4, 8, 4]);
        assert_eq!(mps.num_sites(), 4);
        assert_eq!(mps.max_bond_dim(), 8);
        assert_eq!(mps.bond_dim(0), Some(4));
        assert_eq!(mps.bond_dim(1), Some(8));
        assert_eq!(mps.bond_dim(2), Some(4));
    }

    #[test]
    fn test_peps_creation() {
        let peps = ProjectedEntangledPairState::new(2, 3, 2, 4);
        assert_eq!(peps.grid_shape, (2, 3));
        assert_eq!(peps.physical_dim, 2);
        assert_eq!(peps.num_sites(), 6);
        assert_eq!(peps.network.num_nodes(), 6);
    }

    #[test]
    fn test_peps_node_at() {
        let peps = ProjectedEntangledPairState::new(2, 2, 2, 4);
        assert!(peps.node_at(0, 0).is_some());
        assert!(peps.node_at(1, 1).is_some());
        assert!(peps.node_at(2, 0).is_none());
    }

    #[test]
    fn test_node_id_equality() {
        let id1 = NodeId::new(5);
        let id2 = NodeId::new(5);
        let id3 = NodeId::new(6);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_tensor_network_max_bond_dim() {
        let mut network = TensorNetwork::new();
        let node1 = network.add_node(vec![2, 3, 4]);
        let node2 = network.add_node(vec![3, 5]);
        let node3 = network.add_node(vec![4, 6]);

        network.add_edge(node1, 1, node2, 0, 3).unwrap();
        network.add_edge(node1, 2, node3, 0, 4).unwrap();

        assert_eq!(network.max_bond_dimension(), 4);
    }

    #[test]
    fn test_tensor_network_total_bond_dim() {
        let mut network = TensorNetwork::new();
        let node1 = network.add_node(vec![2, 3, 4]);
        let node2 = network.add_node(vec![3, 5]);
        let node3 = network.add_node(vec![4, 6]);

        network.add_edge(node1, 1, node2, 0, 3).unwrap();
        network.add_edge(node1, 2, node3, 0, 4).unwrap();

        assert_eq!(network.total_bond_dimension(), 7); // 3 + 4
    }
}
