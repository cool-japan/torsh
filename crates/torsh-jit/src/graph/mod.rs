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
