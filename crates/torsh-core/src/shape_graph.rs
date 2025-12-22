//! Graph-Based Shape Inference for Optimization
//!
//! This module provides graph-based shape inference capabilities that enable
//! compile-time shape analysis and optimization. By building a computation graph
//! of shape operations, we can:
//!
//! - Infer output shapes from input shapes
//! - Detect shape errors at graph construction time
//! - Optimize shape calculations by caching and reuse
//! - Enable shape fusion and elimination of redundant operations
//! - Provide better error messages with full operation context
//!
//! # Architecture
//!
//! The shape graph consists of nodes representing shape values and edges
//! representing operations that transform shapes. Each node stores:
//! - The shape value (if known)
//! - Dependencies on other nodes
//! - Metadata about the operation that produced it
//!
//! # Example
//!
//! ```ignore
//! use torsh_core::shape_graph::*;
//!
//! let mut graph = ShapeGraph::new();
//! let input = graph.add_input(vec![2, 3, 4]);
//! let reshaped = graph.reshape(input, vec![2, 12]);
//! let transposed = graph.transpose(reshaped, vec![1, 0]);
//!
//! // Infer the final shape
//! let output_shape = graph.infer_shape(transposed).unwrap();
//! assert_eq!(output_shape, vec![12, 2]);
//! ```

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};
use core::fmt;
#[cfg(feature = "std")]
use std::collections::BTreeMap;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Node ID in the shape graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(usize);

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

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Shape operation type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeOp {
    /// Input shape (leaf node)
    Input,
    /// Reshape operation
    Reshape { target_shape: Vec<usize> },
    /// Transpose operation
    Transpose { axes: Vec<usize> },
    /// Broadcast operation
    Broadcast { target_shape: Vec<usize> },
    /// Concatenate operation along axis
    Concatenate { axis: usize, other: NodeId },
    /// Stack operation along new axis
    Stack { axis: usize, other: NodeId },
    /// Squeeze operation (remove dimensions of size 1)
    Squeeze { axes: Option<Vec<usize>> },
    /// Unsqueeze operation (add dimensions of size 1)
    Unsqueeze { axes: Vec<usize> },
    /// Slice operation
    Slice { ranges: Vec<(usize, usize)> },
    /// Expand operation (broadcast without materialization)
    Expand { target_shape: Vec<usize> },
    /// Flatten operation
    Flatten { start_dim: usize, end_dim: usize },
    /// Permute operation (generalized transpose)
    Permute { dims: Vec<usize> },
}

impl fmt::Display for ShapeOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeOp::Input => write!(f, "Input"),
            ShapeOp::Reshape { target_shape } => write!(f, "Reshape({:?})", target_shape),
            ShapeOp::Transpose { axes } => write!(f, "Transpose({:?})", axes),
            ShapeOp::Broadcast { target_shape } => write!(f, "Broadcast({:?})", target_shape),
            ShapeOp::Concatenate { axis, other } => {
                write!(f, "Concatenate(axis={}, {})", axis, other)
            }
            ShapeOp::Stack { axis, other } => write!(f, "Stack(axis={}, {})", axis, other),
            ShapeOp::Squeeze { axes } => write!(f, "Squeeze({:?})", axes),
            ShapeOp::Unsqueeze { axes } => write!(f, "Unsqueeze({:?})", axes),
            ShapeOp::Slice { ranges } => write!(f, "Slice({:?})", ranges),
            ShapeOp::Expand { target_shape } => write!(f, "Expand({:?})", target_shape),
            ShapeOp::Flatten { start_dim, end_dim } => {
                write!(f, "Flatten({}..{})", start_dim, end_dim)
            }
            ShapeOp::Permute { dims } => write!(f, "Permute({:?})", dims),
        }
    }
}

/// Shape graph node
#[derive(Debug, Clone)]
pub struct ShapeNode {
    /// Node ID
    id: NodeId,
    /// Known shape (if computed)
    shape: Option<Vec<usize>>,
    /// Operation that produces this node
    op: ShapeOp,
    /// Dependencies (input nodes)
    dependencies: Vec<NodeId>,
    /// Metadata for debugging
    name: Option<String>,
}

impl ShapeNode {
    /// Create a new shape node
    pub fn new(id: NodeId, op: ShapeOp, dependencies: Vec<NodeId>) -> Self {
        Self {
            id,
            shape: None,
            op,
            dependencies,
            name: None,
        }
    }

    /// Get the node ID
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Set the shape for this node
    pub fn set_shape(&mut self, shape: Vec<usize>) {
        self.shape = Some(shape);
    }

    /// Get the shape if known
    pub fn shape(&self) -> Option<&[usize]> {
        self.shape.as_deref()
    }

    /// Get the operation
    pub fn op(&self) -> &ShapeOp {
        &self.op
    }

    /// Get the dependencies
    pub fn dependencies(&self) -> &[NodeId] {
        &self.dependencies
    }

    /// Set a name for debugging
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    /// Get the name if set
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// Shape inference error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeInferenceError {
    /// Node not found
    NodeNotFound(NodeId),
    /// Invalid reshape target
    InvalidReshape {
        source_shape: Vec<usize>,
        target_shape: Vec<usize>,
        reason: String,
    },
    /// Invalid transpose axes
    InvalidTranspose {
        shape: Vec<usize>,
        axes: Vec<usize>,
        reason: String,
    },
    /// Invalid broadcast
    InvalidBroadcast {
        source_shape: Vec<usize>,
        target_shape: Vec<usize>,
        reason: String,
    },
    /// Invalid concatenation
    InvalidConcatenate {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
        axis: usize,
        reason: String,
    },
    /// Invalid slice
    InvalidSlice {
        shape: Vec<usize>,
        ranges: Vec<(usize, usize)>,
        reason: String,
    },
    /// Cyclic dependency detected
    CyclicDependency(NodeId),
    /// Unknown shape (cannot infer)
    UnknownShape(NodeId),
}

impl fmt::Display for ShapeInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeInferenceError::NodeNotFound(id) => write!(f, "Node not found: {}", id),
            ShapeInferenceError::InvalidReshape {
                source_shape,
                target_shape,
                reason,
            } => write!(
                f,
                "Invalid reshape from {:?} to {:?}: {}",
                source_shape, target_shape, reason
            ),
            ShapeInferenceError::InvalidTranspose {
                shape,
                axes,
                reason,
            } => {
                write!(
                    f,
                    "Invalid transpose of {:?} with axes {:?}: {}",
                    shape, axes, reason
                )
            }
            ShapeInferenceError::InvalidBroadcast {
                source_shape,
                target_shape,
                reason,
            } => write!(
                f,
                "Invalid broadcast from {:?} to {:?}: {}",
                source_shape, target_shape, reason
            ),
            ShapeInferenceError::InvalidConcatenate {
                shape1,
                shape2,
                axis,
                reason,
            } => write!(
                f,
                "Invalid concatenate of {:?} and {:?} along axis {}: {}",
                shape1, shape2, axis, reason
            ),
            ShapeInferenceError::InvalidSlice {
                shape,
                ranges,
                reason,
            } => {
                write!(
                    f,
                    "Invalid slice of {:?} with ranges {:?}: {}",
                    shape, ranges, reason
                )
            }
            ShapeInferenceError::CyclicDependency(id) => {
                write!(f, "Cyclic dependency detected at {}", id)
            }
            ShapeInferenceError::UnknownShape(id) => write!(f, "Unknown shape for {}", id),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ShapeInferenceError {}

/// Result type for shape inference
pub type InferenceResult<T> = Result<T, ShapeInferenceError>;

/// Shape graph for inference and optimization
#[derive(Debug, Clone)]
pub struct ShapeGraph {
    /// All nodes in the graph
    nodes: BTreeMap<NodeId, ShapeNode>,
    /// Next node ID
    next_id: usize,
    /// Cached inference results
    cache: BTreeMap<NodeId, Vec<usize>>,
}

impl ShapeGraph {
    /// Create a new empty shape graph
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            next_id: 0,
            cache: BTreeMap::new(),
        }
    }

    /// Allocate a new node ID
    fn alloc_id(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add an input node with a known shape
    pub fn add_input(&mut self, shape: Vec<usize>) -> NodeId {
        let id = self.alloc_id();
        let mut node = ShapeNode::new(id, ShapeOp::Input, Vec::new());
        node.set_shape(shape.clone());
        self.nodes.insert(id, node);
        self.cache.insert(id, shape);
        id
    }

    /// Add a reshape operation
    pub fn reshape(&mut self, input: NodeId, target_shape: Vec<usize>) -> NodeId {
        let id = self.alloc_id();
        let node = ShapeNode::new(
            id,
            ShapeOp::Reshape {
                target_shape: target_shape.clone(),
            },
            vec![input],
        );
        self.nodes.insert(id, node);
        id
    }

    /// Add a transpose operation
    pub fn transpose(&mut self, input: NodeId, axes: Vec<usize>) -> NodeId {
        let id = self.alloc_id();
        let node = ShapeNode::new(id, ShapeOp::Transpose { axes }, vec![input]);
        self.nodes.insert(id, node);
        id
    }

    /// Add a broadcast operation
    pub fn broadcast(&mut self, input: NodeId, target_shape: Vec<usize>) -> NodeId {
        let id = self.alloc_id();
        let node = ShapeNode::new(
            id,
            ShapeOp::Broadcast {
                target_shape: target_shape.clone(),
            },
            vec![input],
        );
        self.nodes.insert(id, node);
        id
    }

    /// Add a concatenate operation
    pub fn concatenate(&mut self, input1: NodeId, input2: NodeId, axis: usize) -> NodeId {
        let id = self.alloc_id();
        let node = ShapeNode::new(
            id,
            ShapeOp::Concatenate {
                axis,
                other: input2,
            },
            vec![input1, input2],
        );
        self.nodes.insert(id, node);
        id
    }

    /// Add a stack operation
    pub fn stack(&mut self, input1: NodeId, input2: NodeId, axis: usize) -> NodeId {
        let id = self.alloc_id();
        let node = ShapeNode::new(
            id,
            ShapeOp::Stack {
                axis,
                other: input2,
            },
            vec![input1, input2],
        );
        self.nodes.insert(id, node);
        id
    }

    /// Add a squeeze operation
    pub fn squeeze(&mut self, input: NodeId, axes: Option<Vec<usize>>) -> NodeId {
        let id = self.alloc_id();
        let node = ShapeNode::new(id, ShapeOp::Squeeze { axes }, vec![input]);
        self.nodes.insert(id, node);
        id
    }

    /// Add an unsqueeze operation
    pub fn unsqueeze(&mut self, input: NodeId, axes: Vec<usize>) -> NodeId {
        let id = self.alloc_id();
        let node = ShapeNode::new(id, ShapeOp::Unsqueeze { axes }, vec![input]);
        self.nodes.insert(id, node);
        id
    }

    /// Add a flatten operation
    pub fn flatten(&mut self, input: NodeId, start_dim: usize, end_dim: usize) -> NodeId {
        let id = self.alloc_id();
        let node = ShapeNode::new(id, ShapeOp::Flatten { start_dim, end_dim }, vec![input]);
        self.nodes.insert(id, node);
        id
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&ShapeNode> {
        self.nodes.get(&id)
    }

    /// Infer the shape for a node
    pub fn infer_shape(&mut self, node_id: NodeId) -> InferenceResult<Vec<usize>> {
        // Check cache first
        if let Some(cached) = self.cache.get(&node_id) {
            return Ok(cached.clone());
        }

        // Get the node
        let node = self
            .nodes
            .get(&node_id)
            .ok_or(ShapeInferenceError::NodeNotFound(node_id))?
            .clone();

        // If shape is already known, return it
        if let Some(shape) = node.shape() {
            let result = shape.to_vec();
            self.cache.insert(node_id, result.clone());
            return Ok(result);
        }

        // Infer based on operation
        let inferred_shape = match &node.op {
            ShapeOp::Input => {
                return Err(ShapeInferenceError::UnknownShape(node_id));
            }
            ShapeOp::Reshape { target_shape } => {
                let input_id = node.dependencies[0];
                let input_shape = self.infer_shape(input_id)?;
                Self::infer_reshape(&input_shape, target_shape)?
            }
            ShapeOp::Transpose { axes } => {
                let input_id = node.dependencies[0];
                let input_shape = self.infer_shape(input_id)?;
                Self::infer_transpose(&input_shape, axes)?
            }
            ShapeOp::Broadcast { target_shape } => {
                let input_id = node.dependencies[0];
                let input_shape = self.infer_shape(input_id)?;
                Self::infer_broadcast(&input_shape, target_shape)?
            }
            ShapeOp::Concatenate { axis, .. } => {
                let input1_id = node.dependencies[0];
                let input2_id = node.dependencies[1];
                let shape1 = self.infer_shape(input1_id)?;
                let shape2 = self.infer_shape(input2_id)?;
                Self::infer_concatenate(&shape1, &shape2, *axis)?
            }
            ShapeOp::Stack { axis, .. } => {
                let input1_id = node.dependencies[0];
                let input2_id = node.dependencies[1];
                let shape1 = self.infer_shape(input1_id)?;
                let shape2 = self.infer_shape(input2_id)?;
                Self::infer_stack(&shape1, &shape2, *axis)?
            }
            ShapeOp::Squeeze { axes } => {
                let input_id = node.dependencies[0];
                let input_shape = self.infer_shape(input_id)?;
                Self::infer_squeeze(&input_shape, axes.as_ref())?
            }
            ShapeOp::Unsqueeze { axes } => {
                let input_id = node.dependencies[0];
                let input_shape = self.infer_shape(input_id)?;
                Self::infer_unsqueeze(&input_shape, axes)?
            }
            ShapeOp::Flatten { start_dim, end_dim } => {
                let input_id = node.dependencies[0];
                let input_shape = self.infer_shape(input_id)?;
                Self::infer_flatten(&input_shape, *start_dim, *end_dim)?
            }
            _ => {
                return Err(ShapeInferenceError::UnknownShape(node_id));
            }
        };

        // Cache the result
        self.cache.insert(node_id, inferred_shape.clone());

        // Update the node
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.set_shape(inferred_shape.clone());
        }

        Ok(inferred_shape)
    }

    /// Infer reshape output shape
    fn infer_reshape(input_shape: &[usize], target_shape: &[usize]) -> InferenceResult<Vec<usize>> {
        let input_numel: usize = input_shape.iter().product();
        let mut output_shape = target_shape.to_vec();

        // Handle -1 in target shape (infer dimension)
        let neg_count = target_shape.iter().filter(|&&x| x == usize::MAX).count();
        if neg_count > 1 {
            return Err(ShapeInferenceError::InvalidReshape {
                source_shape: input_shape.to_vec(),
                target_shape: target_shape.to_vec(),
                reason: "At most one dimension can be inferred".to_string(),
            });
        }

        if neg_count == 1 {
            let known_product: usize = target_shape.iter().filter(|&&x| x != usize::MAX).product();
            if known_product == 0 || input_numel % known_product != 0 {
                return Err(ShapeInferenceError::InvalidReshape {
                    source_shape: input_shape.to_vec(),
                    target_shape: target_shape.to_vec(),
                    reason: "Cannot infer dimension size".to_string(),
                });
            }
            let inferred = input_numel / known_product;
            for dim in &mut output_shape {
                if *dim == usize::MAX {
                    *dim = inferred;
                }
            }
        }

        let output_numel: usize = output_shape.iter().product();
        if input_numel != output_numel {
            return Err(ShapeInferenceError::InvalidReshape {
                source_shape: input_shape.to_vec(),
                target_shape: target_shape.to_vec(),
                reason: format!(
                    "Element count mismatch: {} vs {}",
                    input_numel, output_numel
                ),
            });
        }

        Ok(output_shape)
    }

    /// Infer transpose output shape
    fn infer_transpose(input_shape: &[usize], axes: &[usize]) -> InferenceResult<Vec<usize>> {
        if axes.len() != input_shape.len() {
            return Err(ShapeInferenceError::InvalidTranspose {
                shape: input_shape.to_vec(),
                axes: axes.to_vec(),
                reason: "Axes count must match shape rank".to_string(),
            });
        }

        let mut output_shape = vec![0; input_shape.len()];
        for (i, &axis) in axes.iter().enumerate() {
            if axis >= input_shape.len() {
                return Err(ShapeInferenceError::InvalidTranspose {
                    shape: input_shape.to_vec(),
                    axes: axes.to_vec(),
                    reason: format!("Axis {} out of bounds", axis),
                });
            }
            output_shape[i] = input_shape[axis];
        }

        Ok(output_shape)
    }

    /// Infer broadcast output shape
    fn infer_broadcast(
        input_shape: &[usize],
        target_shape: &[usize],
    ) -> InferenceResult<Vec<usize>> {
        if input_shape.len() > target_shape.len() {
            return Err(ShapeInferenceError::InvalidBroadcast {
                source_shape: input_shape.to_vec(),
                target_shape: target_shape.to_vec(),
                reason: "Source rank exceeds target rank".to_string(),
            });
        }

        let offset = target_shape.len() - input_shape.len();
        for (i, &dim) in input_shape.iter().enumerate() {
            let target_dim = target_shape[offset + i];
            if dim != 1 && dim != target_dim {
                return Err(ShapeInferenceError::InvalidBroadcast {
                    source_shape: input_shape.to_vec(),
                    target_shape: target_shape.to_vec(),
                    reason: format!(
                        "Dimension {} cannot be broadcast: {} to {}",
                        i, dim, target_dim
                    ),
                });
            }
        }

        Ok(target_shape.to_vec())
    }

    /// Infer concatenate output shape
    fn infer_concatenate(
        shape1: &[usize],
        shape2: &[usize],
        axis: usize,
    ) -> InferenceResult<Vec<usize>> {
        if shape1.len() != shape2.len() {
            return Err(ShapeInferenceError::InvalidConcatenate {
                shape1: shape1.to_vec(),
                shape2: shape2.to_vec(),
                axis,
                reason: "Shapes must have same rank".to_string(),
            });
        }

        if axis >= shape1.len() {
            return Err(ShapeInferenceError::InvalidConcatenate {
                shape1: shape1.to_vec(),
                shape2: shape2.to_vec(),
                axis,
                reason: format!("Axis {} out of bounds", axis),
            });
        }

        for (i, (&dim1, &dim2)) in shape1.iter().zip(shape2.iter()).enumerate() {
            if i != axis && dim1 != dim2 {
                return Err(ShapeInferenceError::InvalidConcatenate {
                    shape1: shape1.to_vec(),
                    shape2: shape2.to_vec(),
                    axis,
                    reason: format!("Dimension {} mismatch: {} vs {}", i, dim1, dim2),
                });
            }
        }

        let mut output = shape1.to_vec();
        output[axis] += shape2[axis];
        Ok(output)
    }

    /// Infer stack output shape
    fn infer_stack(shape1: &[usize], shape2: &[usize], axis: usize) -> InferenceResult<Vec<usize>> {
        if shape1 != shape2 {
            return Err(ShapeInferenceError::InvalidConcatenate {
                shape1: shape1.to_vec(),
                shape2: shape2.to_vec(),
                axis,
                reason: "Shapes must be identical for stack".to_string(),
            });
        }

        if axis > shape1.len() {
            return Err(ShapeInferenceError::InvalidConcatenate {
                shape1: shape1.to_vec(),
                shape2: shape2.to_vec(),
                axis,
                reason: format!("Axis {} out of bounds", axis),
            });
        }

        let mut output = Vec::with_capacity(shape1.len() + 1);
        output.extend_from_slice(&shape1[..axis]);
        output.push(2);
        output.extend_from_slice(&shape1[axis..]);
        Ok(output)
    }

    /// Infer squeeze output shape
    fn infer_squeeze(
        input_shape: &[usize],
        axes: Option<&Vec<usize>>,
    ) -> InferenceResult<Vec<usize>> {
        let output = if let Some(axes) = axes {
            input_shape
                .iter()
                .enumerate()
                .filter(|(i, &dim)| !axes.contains(i) || dim != 1)
                .map(|(_, &dim)| dim)
                .collect()
        } else {
            input_shape
                .iter()
                .filter(|&&dim| dim != 1)
                .copied()
                .collect()
        };
        Ok(output)
    }

    /// Infer unsqueeze output shape
    fn infer_unsqueeze(input_shape: &[usize], axes: &[usize]) -> InferenceResult<Vec<usize>> {
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();

        // Build output shape by inserting 1s at the specified axes
        let final_rank = input_shape.len() + axes.len();
        let mut output = Vec::with_capacity(final_rank);
        let mut input_idx = 0;
        let mut axes_idx = 0;

        for i in 0..final_rank {
            if axes_idx < sorted_axes.len() && sorted_axes[axes_idx] == i {
                // This position is an unsqueezed axis
                output.push(1);
                axes_idx += 1;
            } else {
                // This position comes from the input
                if input_idx >= input_shape.len() {
                    return Err(ShapeInferenceError::UnknownShape(NodeId(0)));
                }
                output.push(input_shape[input_idx]);
                input_idx += 1;
            }
        }

        Ok(output)
    }

    /// Infer flatten output shape
    fn infer_flatten(
        input_shape: &[usize],
        start_dim: usize,
        end_dim: usize,
    ) -> InferenceResult<Vec<usize>> {
        if start_dim >= input_shape.len() || end_dim >= input_shape.len() || start_dim > end_dim {
            return Err(ShapeInferenceError::UnknownShape(NodeId(0)));
        }

        let flattened_size: usize = input_shape[start_dim..=end_dim].iter().product();
        let mut output = Vec::with_capacity(input_shape.len() - (end_dim - start_dim));
        output.extend_from_slice(&input_shape[..start_dim]);
        output.push(flattened_size);
        if end_dim + 1 < input_shape.len() {
            output.extend_from_slice(&input_shape[end_dim + 1..]);
        }

        Ok(output)
    }

    /// Clear the inference cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get all nodes in topological order
    pub fn topological_sort(&self) -> InferenceResult<Vec<NodeId>> {
        let mut result = Vec::new();
        let mut visited = BTreeMap::new();
        let mut temp_mark = BTreeMap::new();

        for &node_id in self.nodes.keys() {
            if !visited.contains_key(&node_id) {
                self.visit_node(node_id, &mut visited, &mut temp_mark, &mut result)?;
            }
        }

        Ok(result)
    }

    fn visit_node(
        &self,
        node_id: NodeId,
        visited: &mut BTreeMap<NodeId, bool>,
        temp_mark: &mut BTreeMap<NodeId, bool>,
        result: &mut Vec<NodeId>,
    ) -> InferenceResult<()> {
        if visited.get(&node_id) == Some(&true) {
            return Ok(());
        }

        if temp_mark.get(&node_id) == Some(&true) {
            return Err(ShapeInferenceError::CyclicDependency(node_id));
        }

        temp_mark.insert(node_id, true);

        if let Some(node) = self.nodes.get(&node_id) {
            for &dep_id in &node.dependencies {
                self.visit_node(dep_id, visited, temp_mark, result)?;
            }
        }

        temp_mark.insert(node_id, false);
        visited.insert(node_id, true);
        result.push(node_id);

        Ok(())
    }

    /// Count the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for ShapeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;
    use std::vec;

    #[test]
    fn test_input_node() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 3, 4]);

        let shape = graph.infer_shape(input).unwrap();
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_reshape() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 3, 4]);
        let reshaped = graph.reshape(input, vec![2, 12]);

        let shape = graph.infer_shape(reshaped).unwrap();
        assert_eq!(shape, vec![2, 12]);
    }

    #[test]
    fn test_transpose() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 3, 4]);
        let transposed = graph.transpose(input, vec![2, 0, 1]);

        let shape = graph.infer_shape(transposed).unwrap();
        assert_eq!(shape, vec![4, 2, 3]);
    }

    #[test]
    fn test_broadcast() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![1, 3, 1]);
        let broadcasted = graph.broadcast(input, vec![2, 3, 4]);

        let shape = graph.infer_shape(broadcasted).unwrap();
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_concatenate() {
        let mut graph = ShapeGraph::new();
        let input1 = graph.add_input(vec![2, 3, 4]);
        let input2 = graph.add_input(vec![2, 5, 4]);
        let concatenated = graph.concatenate(input1, input2, 1);

        let shape = graph.infer_shape(concatenated).unwrap();
        assert_eq!(shape, vec![2, 8, 4]);
    }

    #[test]
    fn test_stack() {
        let mut graph = ShapeGraph::new();
        let input1 = graph.add_input(vec![2, 3, 4]);
        let input2 = graph.add_input(vec![2, 3, 4]);
        let stacked = graph.stack(input1, input2, 1);

        let shape = graph.infer_shape(stacked).unwrap();
        assert_eq!(shape, vec![2, 2, 3, 4]);
    }

    #[test]
    fn test_squeeze() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 1, 3, 1, 4]);
        let squeezed = graph.squeeze(input, None);

        let shape = graph.infer_shape(squeezed).unwrap();
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_unsqueeze() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 3, 4]);
        let unsqueezed = graph.unsqueeze(input, vec![1, 3]);

        let shape = graph.infer_shape(unsqueezed).unwrap();
        assert_eq!(shape, vec![2, 1, 3, 1, 4]);
    }

    #[test]
    fn test_flatten() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 3, 4, 5]);
        let flattened = graph.flatten(input, 1, 2);

        let shape = graph.infer_shape(flattened).unwrap();
        assert_eq!(shape, vec![2, 12, 5]);
    }

    #[test]
    fn test_complex_graph() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 3, 4]);
        let reshaped = graph.reshape(input, vec![2, 12]);
        let transposed = graph.transpose(reshaped, vec![1, 0]);
        let unsqueezed = graph.unsqueeze(transposed, vec![1]);

        let shape = graph.infer_shape(unsqueezed).unwrap();
        assert_eq!(shape, vec![12, 1, 2]);
    }

    #[test]
    fn test_invalid_reshape() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 3, 4]);
        let reshaped = graph.reshape(input, vec![2, 13]); // 24 != 26

        assert!(graph.infer_shape(reshaped).is_err());
    }

    #[test]
    fn test_invalid_transpose() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 3, 4]);
        let transposed = graph.transpose(input, vec![0, 1]); // Wrong number of axes

        assert!(graph.infer_shape(transposed).is_err());
    }

    #[test]
    fn test_invalid_broadcast() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 3, 4]);
        let broadcasted = graph.broadcast(input, vec![2, 5, 4]); // 3 != 5

        assert!(graph.infer_shape(broadcasted).is_err());
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 3, 4]);
        let reshaped = graph.reshape(input, vec![2, 12]);
        let transposed = graph.transpose(reshaped, vec![1, 0]);

        let sorted = graph.topological_sort().unwrap();
        assert_eq!(sorted.len(), 3);

        // Input should come before reshape, reshape before transpose
        let input_pos = sorted.iter().position(|&id| id == input).unwrap();
        let reshape_pos = sorted.iter().position(|&id| id == reshaped).unwrap();
        let transpose_pos = sorted.iter().position(|&id| id == transposed).unwrap();

        assert!(input_pos < reshape_pos);
        assert!(reshape_pos < transpose_pos);
    }

    #[test]
    fn test_cache() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 3, 4]);
        let reshaped = graph.reshape(input, vec![2, 12]);

        // First inference
        let shape1 = graph.infer_shape(reshaped).unwrap();

        // Second inference should use cache
        let shape2 = graph.infer_shape(reshaped).unwrap();

        assert_eq!(shape1, shape2);
        assert_eq!(shape1, vec![2, 12]);
    }

    #[test]
    fn test_clear_cache() {
        let mut graph = ShapeGraph::new();
        let input = graph.add_input(vec![2, 3, 4]);
        let reshaped = graph.reshape(input, vec![2, 12]);

        // Infer and cache
        let _ = graph.infer_shape(reshaped).unwrap();

        // Clear cache
        graph.clear_cache();

        // Should still work after clearing cache
        let shape = graph.infer_shape(reshaped).unwrap();
        assert_eq!(shape, vec![2, 12]);
    }

    #[test]
    fn test_node_count() {
        let mut graph = ShapeGraph::new();
        assert_eq!(graph.node_count(), 0);

        let input = graph.add_input(vec![2, 3, 4]);
        assert_eq!(graph.node_count(), 1);

        let _reshaped = graph.reshape(input, vec![2, 12]);
        assert_eq!(graph.node_count(), 2);
    }
}
