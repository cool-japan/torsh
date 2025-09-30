//! Simple compilation test to verify basic functionality

use crate::{graph::Operation, ComputationGraph, Edge, JitCompiler, JitConfig, Node};
use torsh_core::{DType, DeviceType, Shape};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_compilation() {
        // Test that we can create basic structures
        let config = JitConfig::default();
        let _compiler = JitCompiler::new(config);

        // Test graph creation
        let mut graph = ComputationGraph::new();
        let node = Node::new(Operation::Input, "test".to_string())
            .with_output_shapes(vec![Some(Shape::new(vec![1, 2, 3]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu);

        let _node_id = graph.add_node(node);
        assert!(true); // If we reach here, basic compilation works
    }
}
