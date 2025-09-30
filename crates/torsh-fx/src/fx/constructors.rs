//! Graph construction utilities and convenience methods

use crate::fx::types::{Edge, Node};
use crate::FxGraph;
use petgraph::graph::NodeIndex;

impl FxGraph {
    /// Convenience method to create a graph with a single operation
    pub fn single_op(op_name: &str, input_names: Vec<String>) -> Self {
        let mut graph = Self::new();

        // Add input nodes
        let mut input_indices = Vec::new();
        for input_name in &input_names {
            let input_idx = graph.add_node(Node::Input(input_name.clone()));
            input_indices.push(input_idx);
            graph.add_input(input_idx);
        }

        // Add operation node
        let op_idx = graph.add_node(Node::Call(op_name.to_string(), input_names.clone()));

        // Add output node
        let output_idx = graph.add_node(Node::Output);
        graph.add_output(output_idx);

        // Connect inputs to operation
        for (i, input_idx) in input_indices.iter().enumerate() {
            graph.add_edge(
                *input_idx,
                op_idx,
                Edge {
                    name: format!("input_{i}"),
                },
            );
        }

        // Connect operation to output
        graph.add_edge(
            op_idx,
            output_idx,
            Edge {
                name: "output".to_string(),
            },
        );

        graph
    }

    /// Convenience method to chain multiple operations sequentially
    pub fn sequential_ops(ops: &[&str]) -> Self {
        let mut graph = Self::new();

        if ops.is_empty() {
            return graph;
        }

        // Add input node
        let input_idx = graph.add_node(Node::Input("input".to_string()));
        graph.add_input(input_idx);

        let mut prev_idx = input_idx;

        // Add operations sequentially
        for (i, op_name) in ops.iter().enumerate() {
            let op_idx = graph.add_node(Node::Call(
                op_name.to_string(),
                vec![format!("intermediate_{i}")],
            ));

            graph.add_edge(
                prev_idx,
                op_idx,
                Edge {
                    name: format!("connection_{i}"),
                },
            );

            prev_idx = op_idx;
        }

        // Add output node
        let output_idx = graph.add_node(Node::Output);
        graph.add_output(output_idx);

        graph.add_edge(
            prev_idx,
            output_idx,
            Edge {
                name: "final_output".to_string(),
            },
        );

        graph
    }

    /// Create a minimal test graph for debugging
    pub fn debug_minimal() -> Self {
        Self::sequential_ops(&["debug_op"])
    }

    /// Create a test graph with branching for debugging
    pub fn debug_branching() -> Self {
        let mut graph = Self::new();

        // Create branching structure: input -> op1 -> op2
        //                                   -> op3 -> output
        let input = graph.add_node(Node::Input("x".to_string()));
        let op1 = graph.add_node(Node::Call("branch_op1".to_string(), vec!["x".to_string()]));
        let op2 = graph.add_node(Node::Call("branch_op2".to_string(), vec![]));
        let op3 = graph.add_node(Node::Call("branch_op3".to_string(), vec![]));
        let output = graph.add_node(Node::Output);

        graph.add_input(input);
        graph.add_output(output);

        // Connect the branching structure
        graph.add_edge(
            input,
            op1,
            Edge {
                name: "input_to_op1".to_string(),
            },
        );
        graph.add_edge(
            op1,
            op2,
            Edge {
                name: "op1_to_op2".to_string(),
            },
        );
        graph.add_edge(
            op1,
            op3,
            Edge {
                name: "op1_to_op3".to_string(),
            },
        );
        graph.add_edge(
            op2,
            output,
            Edge {
                name: "op2_to_output".to_string(),
            },
        );
        graph.add_edge(
            op3,
            output,
            Edge {
                name: "op3_to_output".to_string(),
            },
        );

        graph
    }
}
