use torsh_jit::fusion::{FusionStrategy, KernelFusion};
use torsh_jit::graph::{ComputationGraph, Node, Operation, Conv2dInfo, Edge, shape_from_slice};
use torsh_core::{DType, DeviceType};
use std::collections::HashMap;

fn main() {
    let mut graph = ComputationGraph::new();

    // Create conv -> relu pattern
    let input = graph.add_node(Node {
        name: "input".to_string(),
        op: Operation::Input,
        output_shape: shape_from_slice(&[1, 3, 224, 224]),
        dtype: DType::F32,
        device: DeviceType::Cpu,
        attrs: HashMap::new(),
        inputs: Vec::new(),
        is_output: false,
    });

    let conv = graph.add_node(Node {
        name: "conv".to_string(),
        op: Operation::Conv2d(Conv2dInfo {
            in_channels: 3,
            out_channels: 64,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            groups: 1,
            bias: false,
        }),
        output_shape: shape_from_slice(&[1, 64, 224, 224]),
        dtype: DType::F32,
        device: DeviceType::Cpu,
        attrs: HashMap::new(),
        inputs: Vec::new(),
        is_output: false,
    });

    let relu = graph.add_node(Node {
        name: "relu".to_string(),
        op: Operation::Relu,
        output_shape: shape_from_slice(&[1, 64, 224, 224]),
        dtype: DType::F32,
        device: DeviceType::Cpu,
        attrs: HashMap::new(),
        inputs: Vec::new(),
        is_output: false,
    });

    graph.add_edge(input, conv, Edge::default());
    graph.add_edge(conv, relu, Edge::default());
    graph.add_input(input);
    graph.add_output(relu);

    println!("Original graph validation: {:?}", graph.validate());

    // Test fusion
    let fusion = KernelFusion::new(FusionStrategy::Default);
    match fusion.apply(graph) {
        Ok(fused_graph) => {
            match fused_graph.validate() {
                Ok(_) => println!("Fused graph is valid!"),
                Err(e) => println!("Fused graph validation error: {}", e),
            }
        }
        Err(e) => println!("Fusion failed: {:?}", e),
    }
}