//! Computation graph representation and analysis
//!
//! This module provides a comprehensive framework for representing, building, and analyzing
//! computation graphs used in JIT compilation. The module is organized into several
//! sub-modules for better maintainability:
//!
//! - [`core`]: Core graph structures and fundamental operations
//! - [`operations`]: Operation definitions and related structures
//! - [`builder`]: Graph construction utilities and builder pattern
//! - [`metadata`]: Graph metadata and configuration structures
//! - [`control_flow`]: Control flow analysis and loop detection
//!
//! # Examples
//!
//! ## Building a Simple Graph
//!
//! ```rust
//! use torsh_jit::graph::{GraphBuilder, Operation};
//! use torsh_core::{Shape, DType};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut builder = GraphBuilder::new();
//!
//! // Add input node
//! let input = builder.add_input(
//!     "input".to_string(),
//!     Shape::new(vec![1, 3, 224, 224]),
//!     DType::F32
//! );
//!
//! // Add ReLU operation
//! let relu = builder.add_unary_op(
//!     "relu".to_string(),
//!     Operation::Relu,
//!     input
//! )?;
//!
//! // Mark as output
//! builder.mark_output(relu)?;
//!
//! // Build the graph
//! let graph = builder.build()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Analyzing Control Flow
//!
//! ```rust
//! use torsh_jit::graph::{ComputationGraph, ControlFlowAnalysis};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let graph = ComputationGraph::new();
//! let analysis = ControlFlowAnalysis::analyze(&graph)?;
//!
//! println!("Found {} loops and {} conditionals",
//!     analysis.stats.loop_count,
//!     analysis.stats.conditional_count
//! );
//! # Ok(())
//! # }
//! ```

pub mod builder;
pub mod control_flow;
pub mod core;
pub mod metadata;
pub mod operations;

// Re-export commonly used types for convenience
pub use core::{
    shape_from_slice, ComputationGraph, Edge, Node, NodeId, OperationCategory,
    SerializableNodeIndex,
};

pub use operations::{
    Attribute, BatchNormInfo, BlockInfo, BlockType, ConstantInfo, ConstantValue, Conv2dInfo,
    ForInfo, IfInfo, LinearInfo, MergeInfo, MergeStrategy, Operation, ParameterInfo, PoolInfo,
    SliceRange, WhileInfo,
};

pub use builder::{GraphBuilder, GraphStatistics};
pub use control_flow::{
    ConditionalInfo, ControlFlowAnalysis, ControlFlowStats, LoopInfo, LoopType,
};
pub use metadata::{GraphMetadata, OptimizationLevel};

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::{DType, Shape};

    #[test]
    fn test_graph_creation() {
        let graph = ComputationGraph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_graph_builder() {
        let mut builder = GraphBuilder::new();

        let input = builder.add_input("input".to_string(), Shape::new(vec![1, 784]), DType::F32);

        let relu = builder
            .add_unary_op("relu".to_string(), Operation::Relu, input)
            .unwrap();

        builder.mark_output(relu).unwrap();

        let graph = builder.build().unwrap();
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_operation_properties() {
        assert!(Operation::Add.is_elementwise());
        assert!(Operation::Sum {
            dims: vec![0],
            keepdim: false
        }
        .is_reduction());
        assert!(Operation::Reshape { shape: vec![1, -1] }.modifies_shape());
        assert!(Operation::Relu.is_fusible());

        assert_eq!(Operation::Add.expected_inputs(), 2);
        assert_eq!(Operation::Relu.expected_inputs(), 1);
        assert_eq!(Operation::Input.expected_inputs(), 0);
    }

    #[test]
    fn test_graph_validation() {
        let mut builder = GraphBuilder::new();

        let input = builder.add_input("input".to_string(), Shape::new(vec![1, 784]), DType::F32);

        builder.mark_output(input).unwrap();

        let graph = builder.build().unwrap();
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_replace_node_with_input() {
        let mut graph = ComputationGraph::new();

        // Create a simple chain: input -> relu -> output
        let input = graph.add_node(
            Node::new(Operation::Input, "input".to_string())
                .with_output_shapes(vec![Some(Shape::new(vec![1, 10]))])
                .with_dtypes(vec![DType::F32]),
        );
        let relu = graph.add_node(
            Node::new(Operation::Relu, "relu".to_string())
                .with_output_shapes(vec![Some(Shape::new(vec![1, 10]))])
                .with_dtypes(vec![DType::F32]),
        );
        let output = graph.add_node(
            Node::new(Operation::Input, "output".to_string())
                .with_output_shapes(vec![Some(Shape::new(vec![1, 10]))])
                .with_dtypes(vec![DType::F32]),
        );

        graph.add_edge(
            input,
            relu,
            Edge {
                src_output: 0,
                dst_input: 0,
            },
        );
        graph.add_edge(
            relu,
            output,
            Edge {
                src_output: 0,
                dst_input: 0,
            },
        );
        graph.add_output(output);

        let initial_node_count = graph.node_count();

        // Replace relu with input (bypass the relu)
        let result = graph.replace_node_with_input(relu, input);
        assert!(result.is_ok());

        // One node should have been removed (relu)
        assert_eq!(graph.node_count(), initial_node_count - 1);

        // Since NodeIndex can be reused by petgraph after node removal,
        // we need to find nodes by their properties (name) rather than old indices
        let actual_output_id = graph
            .nodes()
            .find(|(_, n)| n.name == "output")
            .map(|(id, _)| id)
            .expect("Output node should exist");

        let actual_input_id = graph
            .nodes()
            .find(|(_, n)| n.name == "input")
            .map(|(id, _)| id)
            .expect("Input node should exist");

        // Verify that input now connects directly to output
        let output_predecessors: Vec<_> = graph.predecessors(actual_output_id).collect();
        assert_eq!(
            output_predecessors.len(),
            1,
            "Output should have exactly 1 predecessor"
        );
        assert_eq!(
            output_predecessors[0], actual_input_id,
            "Output's predecessor should be input"
        );
    }

    #[test]
    fn test_replace_node_with_sequence() {
        let mut graph = ComputationGraph::new();

        // Create: input -> placeholder -> output
        let input = graph.add_node(
            Node::new(Operation::Input, "input".to_string())
                .with_output_shapes(vec![Some(Shape::new(vec![1, 10]))])
                .with_dtypes(vec![DType::F32]),
        );
        let placeholder = graph.add_node(
            Node::new(Operation::Input, "placeholder".to_string())
                .with_output_shapes(vec![Some(Shape::new(vec![1, 10]))])
                .with_dtypes(vec![DType::F32]),
        );
        let output = graph.add_node(
            Node::new(Operation::Input, "output".to_string())
                .with_output_shapes(vec![Some(Shape::new(vec![1, 10]))])
                .with_dtypes(vec![DType::F32]),
        );

        graph.add_edge(
            input,
            placeholder,
            Edge {
                src_output: 0,
                dst_input: 0,
            },
        );
        graph.add_edge(
            placeholder,
            output,
            Edge {
                src_output: 0,
                dst_input: 0,
            },
        );
        graph.add_output(output);

        // Create a sequence of relu -> tanh to replace the placeholder
        let sequence = vec![
            Node::new(Operation::Relu, "relu".to_string())
                .with_output_shapes(vec![Some(Shape::new(vec![1, 10]))])
                .with_dtypes(vec![DType::F32]),
            Node::new(Operation::Tanh, "tanh".to_string())
                .with_output_shapes(vec![Some(Shape::new(vec![1, 10]))])
                .with_dtypes(vec![DType::F32]),
        ];

        let initial_node_count = graph.node_count();
        let result = graph.replace_node_with_sequence(placeholder, &sequence);
        assert!(result.is_ok());

        // Placeholder should be removed, but 2 nodes added, so net +1
        assert_eq!(graph.node_count(), initial_node_count + 1);

        // Verify graph structure: input should connect to a relu-like node,
        // and output should have a tanh-like node as predecessor
        let output_predecessors: Vec<_> = graph.predecessors(output).collect();
        assert_eq!(
            output_predecessors.len(),
            1,
            "Output should have exactly one predecessor"
        );

        // The predecessor of output should be one of the newly added nodes
        let output_pred_id = output_predecessors[0];
        let output_pred_node = graph
            .node(output_pred_id)
            .expect("Output predecessor should exist");
        // The last node in the sequence was Tanh
        assert_eq!(
            output_pred_node.operation,
            Operation::Tanh,
            "Output predecessor should be the Tanh node"
        );
    }

    #[test]
    fn test_replace_node_with_input_error_on_non_predecessor() {
        let mut graph = ComputationGraph::new();

        let node1 = graph.add_node(
            Node::new(Operation::Input, "node1".to_string())
                .with_output_shapes(vec![Some(Shape::new(vec![1, 10]))])
                .with_dtypes(vec![DType::F32]),
        );
        let node2 = graph.add_node(
            Node::new(Operation::Relu, "node2".to_string())
                .with_output_shapes(vec![Some(Shape::new(vec![1, 10]))])
                .with_dtypes(vec![DType::F32]),
        );

        // node1 is not a predecessor of node2, so this should fail
        let result = graph.replace_node_with_input(node2, node1);
        assert!(result.is_err());
    }

    #[test]
    fn test_replace_node_with_empty_sequence_error() {
        let mut graph = ComputationGraph::new();

        let node = graph.add_node(
            Node::new(Operation::Input, "node".to_string())
                .with_output_shapes(vec![Some(Shape::new(vec![1, 10]))])
                .with_dtypes(vec![DType::F32]),
        );

        // Empty sequence should fail
        let result = graph.replace_node_with_sequence(node, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_node_categories() {
        use crate::graph::core::OperationCategory;

        let node = Node::new(Operation::Add, "add".to_string());
        assert_eq!(node.operation_category(), OperationCategory::ElementWise);

        let node = Node::new(Operation::MatMul, "matmul".to_string());
        assert_eq!(node.operation_category(), OperationCategory::LinearAlgebra);

        let node = Node::new(
            Operation::Conv2d(Conv2dInfo {
                in_channels: 3,
                out_channels: 64,
                kernel_size: (3, 3),
                stride: (1, 1),
                padding: (1, 1),
                dilation: (1, 1),
                groups: 1,
            }),
            "conv".to_string(),
        );
        assert_eq!(node.operation_category(), OperationCategory::NeuralNetwork);
    }

    #[test]
    fn test_graph_statistics() {
        let mut builder = GraphBuilder::new();

        let input = builder.add_input("input".to_string(), Shape::new(vec![1, 784]), DType::F32);

        let relu1 = builder
            .add_unary_op("relu1".to_string(), Operation::Relu, input)
            .unwrap();

        let relu2 = builder
            .add_unary_op("relu2".to_string(), Operation::Relu, relu1)
            .unwrap();

        builder.mark_output(relu2).unwrap();

        let stats = builder.statistics();
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.edge_count, 2);
        assert_eq!(stats.input_count, 1);
        assert_eq!(stats.output_count, 1);

        // Check operation counts
        assert_eq!(stats.operation_counts.get("input"), Some(&1));
        assert_eq!(stats.operation_counts.get("relu"), Some(&2));
    }

    #[test]
    fn test_control_flow_analysis() {
        let graph = ComputationGraph::new();
        let analysis = ControlFlowAnalysis::analyze(&graph).unwrap();

        assert_eq!(analysis.stats.total_nodes, 0);
        assert_eq!(analysis.stats.loop_count, 0);
        assert_eq!(analysis.stats.conditional_count, 0);
    }

    #[test]
    fn test_metadata() {
        let metadata = GraphMetadata::new("test_graph".to_string())
            .with_version("1.0.0".to_string())
            .with_creator("test".to_string())
            .with_custom("key".to_string(), "value".to_string());

        assert_eq!(metadata.name, "test_graph");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.creator, "test");
        assert_eq!(metadata.get_custom("key"), Some(&"value".to_string()));
    }
}
