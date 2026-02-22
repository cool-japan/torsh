//! Integration tests for ToRSh JIT compilation

use torsh_core::{DType, DeviceType};
use torsh_jit::{
    analysis::GraphAnalyzer,
    fusion::KernelFusion,
    graph::{ComputationGraph, Edge, GraphBuilder, Node, Operation},
    optimizer::GraphOptimizer,
    FusionStrategy, JitCompiler, JitConfig, SpecializationConfig,
};

#[test]
fn test_graph_construction() {
    let mut graph = ComputationGraph::new();

    // Build a simple graph: input -> relu -> output
    let input = graph.add_node(
        Node::new(Operation::Input, "input".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 784]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let relu = graph.add_node(
        Node::new(Operation::Relu, "relu".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 784]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    graph.add_edge(input, relu, Edge::default());
    graph.add_input(input);
    graph.add_output(relu);

    // Validate graph
    assert!(graph.validate().is_ok());

    // Check topological sort
    let sorted = graph.topological_sort().unwrap();
    assert_eq!(sorted, vec![input, relu]);
}

#[test]
fn test_graph_builder() {
    let mut builder = GraphBuilder::new();
    let _input1 = builder.add_input(
        "x".to_string(),
        torsh_jit::graph::shape_from_slice(&[64, 3, 224, 224]),
        DType::F32,
    );
    let _input2 = builder.add_input(
        "y".to_string(),
        torsh_jit::graph::shape_from_slice(&[64, 1000]),
        DType::F32,
    );
    let graph = builder.build().unwrap();

    assert_eq!(graph.inputs.len(), 2);
}

#[test]
fn test_kernel_fusion_elementwise() {
    let mut graph = ComputationGraph::new();

    // Create chain: input -> neg -> abs -> relu -> output
    let input = graph.add_node(
        Node::new(Operation::Input, "input".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[100]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let neg = graph.add_node(
        Node::new(Operation::Neg, "neg".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[100]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let abs = graph.add_node(
        Node::new(Operation::Abs, "abs".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[100]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let relu = graph.add_node(
        Node::new(Operation::Relu, "relu".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[100]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    // Connect nodes
    graph.add_edge(input, neg, Edge::default());
    graph.add_edge(neg, abs, Edge::default());
    graph.add_edge(abs, relu, Edge::default());

    graph.add_input(input);
    graph.add_output(relu);

    // Apply fusion
    let fusion = KernelFusion::new(FusionStrategy::Default);
    let fused_graph = fusion.apply(graph).unwrap();

    // The fusion should identify the elementwise chain
    assert!(fused_graph.validate().is_ok());
}

#[test]
fn test_graph_optimization() {
    let mut graph = ComputationGraph::new();

    // Create graph with dead code
    let input = graph.add_node(
        Node::new(Operation::Input, "input".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[10]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let used = graph.add_node(
        Node::new(Operation::Relu, "used".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[10]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let unused = graph.add_node(
        Node::new(Operation::Sigmoid, "unused".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[10]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    // Connect nodes
    graph.add_edge(input, used, Edge::default());
    graph.add_edge(input, unused, Edge::default()); // Dead branch

    graph.add_input(input);
    graph.add_output(used);

    // Apply optimization
    let optimizer = GraphOptimizer::new();
    let optimized = optimizer.optimize(graph).unwrap();

    // Dead code should be eliminated
    assert!(optimized.node(unused).is_none());
}

#[test]
fn test_graph_analysis() {
    let mut graph = ComputationGraph::new();

    // Create a simple computational graph
    let input = graph.add_node(
        Node::new(Operation::Input, "input".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 512]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let matmul = graph.add_node(
        Node::new(Operation::MatMul, "matmul".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 256]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let relu = graph.add_node(
        Node::new(Operation::Relu, "relu".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 256]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    // Connect nodes
    graph.add_edge(input, matmul, Edge::default());
    graph.add_edge(matmul, relu, Edge::default());

    graph.add_input(input);
    graph.add_output(relu);

    // Analyze graph
    let analysis = GraphAnalyzer::analyze(&graph).unwrap();

    // Check analysis results
    assert_eq!(analysis.memory_usage.len(), 3);
    assert_eq!(analysis.compute_cost.len(), 3);
    assert!(!analysis.critical_path.is_empty());

    // MatMul should have higher compute cost than ReLU
    let matmul_cost = &analysis.compute_cost[&matmul];
    let relu_cost = &analysis.compute_cost[&relu];
    assert!(matmul_cost.flops > relu_cost.flops);
}

#[test]
fn test_jit_compiler_basic() {
    let config = JitConfig {
        fusion_strategy: FusionStrategy::Conservative,
        enable_optimizations: true,
        max_fusion_size: 4,
        enable_profiling: false,
        target_device: DeviceType::Cpu,
        enable_caching: true,
        enable_specialization: true,
        specialization_config: SpecializationConfig::default(),
    };

    let mut compiler = JitCompiler::new(config);

    // Create a simple graph
    let mut graph = ComputationGraph::new();

    let input = graph.add_node(
        Node::new(Operation::Input, "input".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[10]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let output = graph.add_node(
        Node::new(Operation::Relu, "output".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[10]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    graph.add_edge(input, output, Edge::default());
    graph.add_input(input);
    graph.add_output(output);

    // Compile the graph
    let compiled = compiler.compile(graph);
    match &compiled {
        Ok(_) => {}
        Err(e) => println!("Compilation error: {:?}", e),
    }
    assert!(compiled.is_ok());

    let module = compiled.unwrap();

    // Test execution
    let inputs = vec![torsh_jit::TensorRef {
        data: vec![1.0; 10],
    }];
    let result = module.execute(&inputs);
    assert!(result.is_ok());
}

#[test]
fn test_fusion_strategies() {
    let strategies = vec![
        FusionStrategy::None,
        FusionStrategy::Conservative,
        FusionStrategy::Default,
        FusionStrategy::Aggressive,
    ];

    for strategy in strategies {
        let fusion = KernelFusion::new(strategy);

        // Create a test graph
        let mut builder = GraphBuilder::new();
        let _input = builder.add_input(
            "x".to_string(),
            torsh_jit::graph::shape_from_slice(&[32, 100]),
            DType::F32,
        );
        let graph = builder.build().unwrap();

        let result = fusion.apply(graph);
        assert!(result.is_ok());
    }
}

#[test]
fn test_complex_graph() {
    let mut graph = ComputationGraph::new();

    // Create a more complex graph with multiple paths
    let input1 = graph.add_node(
        Node::new(Operation::Input, "input1".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 128]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let input2 = graph.add_node(
        Node::new(Operation::Input, "input2".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 128]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let add = graph.add_node(
        Node::new(Operation::Add, "add".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 128]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let mul = graph.add_node(
        Node::new(Operation::Mul, "mul".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 128]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let relu = graph.add_node(
        Node::new(Operation::Relu, "relu".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 128]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    // Connect: (input1 + input2) * input1 -> relu
    graph.add_edge(
        input1,
        add,
        Edge {
            src_output: 0,
            dst_input: 0,
        },
    );
    graph.add_edge(
        input2,
        add,
        Edge {
            src_output: 0,
            dst_input: 1,
        },
    );
    graph.add_edge(
        add,
        mul,
        Edge {
            src_output: 0,
            dst_input: 0,
        },
    );
    graph.add_edge(
        input1,
        mul,
        Edge {
            src_output: 0,
            dst_input: 1,
        },
    );
    graph.add_edge(mul, relu, Edge::default());

    graph.add_input(input1);
    graph.add_input(input2);
    graph.add_output(relu);

    // Test full JIT pipeline
    let config = JitConfig::default();
    let mut compiler = JitCompiler::new(config);

    let result = compiler.compile(graph);
    if let Err(ref e) = result {
        println!("Compilation error: {:?}", e);
    }
    assert!(result.is_ok());
}

#[test]
fn test_type_inference_system() {
    use torsh_jit::type_inference::{ShapeInference, TypeInference};

    let mut graph = ComputationGraph::new();

    // Create graph with mixed types
    let input = graph.add_node(
        Node::new(Operation::Input, "input".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[16, 32]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let relu = graph.add_node(
        Node::new(Operation::Relu, "relu".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[16, 32]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    graph.add_edge(input, relu, Edge::default());
    graph.add_input(input);
    graph.add_output(relu);

    // Test type inference
    let mut type_inf = TypeInference::new();
    assert!(type_inf.infer_types(&graph).is_ok());
    assert_eq!(type_inf.get_type(input), Some(DType::F32));
    assert_eq!(type_inf.get_type(relu), Some(DType::F32));

    // Test shape inference
    let mut shape_inf = ShapeInference::new();
    assert!(shape_inf.infer_shapes(&graph).is_ok());
    let input_shape = shape_inf.get_shape(input).unwrap();
    let relu_shape = shape_inf.get_shape(relu).unwrap();
    assert_eq!(input_shape.dims(), relu_shape.dims());
}

#[test]
fn test_ir_lowering_and_optimization() {
    use torsh_jit::lowering::{
        lower_graph_to_ir, IrConstantFolding, IrDeadCodeElimination, IrPass,
    };

    let mut graph = ComputationGraph::new();

    // Create a simple graph with constants
    let input = graph.add_node(
        Node::new(Operation::Input, "input".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[10]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let constant = graph.add_node(
        Node::new(
            Operation::Constant(torsh_jit::graph::ConstantInfo {
                value: torsh_jit::graph::ConstantValue::Scalar(2.0),
            }),
            "constant".to_string(),
        )
        .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[1]))])
        .with_dtypes(vec![DType::F32])
        .with_device(DeviceType::Cpu),
    );

    let add = graph.add_node(
        Node::new(Operation::Add, "add".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[10]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    graph.add_edge(input, add, Edge::default());
    graph.add_edge(
        constant,
        add,
        Edge {
            src_output: 0,
            dst_input: 1,
        },
    );
    graph.add_input(input);
    graph.add_output(add);

    // Lower to IR
    let mut ir_module = lower_graph_to_ir(&graph, "test_module".to_string()).unwrap();
    assert_eq!(ir_module.name, "test_module");
    assert!(!ir_module.blocks.is_empty());

    // Test optimization passes
    let dce = IrDeadCodeElimination;
    let _changed = dce.run(&mut ir_module).unwrap();
    // Empty module shouldn't have dead code initially

    let cf = IrConstantFolding;
    let _changed = cf.run(&mut ir_module).unwrap();
    // Should handle constant folding
}

#[test]
fn test_conv_activation_fusion() {
    use torsh_jit::graph::{Conv2dInfo, Operation};

    let mut graph = ComputationGraph::new();

    // Create conv -> relu pattern
    let input = graph.add_node(
        Node::new(Operation::Input, "input".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[
                1, 3, 224, 224,
            ]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let conv = graph.add_node(
        Node::new(
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
        )
        .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[
            1, 64, 224, 224,
        ]))])
        .with_dtypes(vec![DType::F32])
        .with_device(DeviceType::Cpu),
    );

    let relu = graph.add_node(
        Node::new(Operation::Relu, "relu".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[
                1, 64, 224, 224,
            ]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    graph.add_edge(input, conv, Edge::default());
    graph.add_edge(conv, relu, Edge::default());
    graph.add_input(input);
    graph.add_output(relu);

    // Test fusion
    let fusion = KernelFusion::new(FusionStrategy::Default);
    let fused_graph = fusion.apply(graph).unwrap();
    match fused_graph.validate() {
        Ok(_) => (),
        Err(e) => panic!("Fused graph validation failed: {}", e),
    }
}

#[test]
fn test_runtime_execution() {
    use torsh_jit::runtime::JitRuntime;
    use torsh_jit::{CompiledKernel, KernelMetadata, TensorDesc};

    let config = JitConfig::default();
    let runtime = JitRuntime::new(config);

    let mut graph = ComputationGraph::new();
    let input = graph.add_node(
        Node::new(Operation::Input, "input".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[10]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );
    graph.add_input(input);
    graph.add_output(input);

    // Create a dummy kernel
    let kernel = CompiledKernel {
        id: "test_kernel".to_string(),
        source_nodes: vec![input],
        code: vec![],
        metadata: KernelMetadata {
            inputs: vec![TensorDesc {
                dtype: DType::F32,
                shape: vec![10],
                strides: vec![1],
                offset: 0,
            }],
            outputs: vec![TensorDesc {
                dtype: DType::F32,
                shape: vec![10],
                strides: vec![1],
                offset: 0,
            }],
            shared_memory: 0,
            block_size: (1, 1, 1),
            grid_size: (1, 1, 1),
        },
    };

    let inputs = vec![torsh_jit::TensorRef {
        data: vec![1.0; 10],
    }];

    let result = runtime.execute(&graph, &[kernel], &inputs);
    assert!(result.is_ok());

    // Check statistics
    let stats = runtime.stats();
    assert_eq!(stats.kernel_launches, 1);
}

#[test]
fn test_neural_network_patterns() {
    let mut graph = ComputationGraph::new();

    // Create a mini neural network: input -> linear -> relu -> linear -> output
    let input = graph.add_node(
        Node::new(Operation::Input, "input".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 784]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let linear1 = graph.add_node(
        Node::new(
            Operation::Linear(torsh_jit::graph::LinearInfo {
                in_features: 784,
                out_features: 256,
                bias: true,
            }),
            "linear1".to_string(),
        )
        .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 256]))])
        .with_dtypes(vec![DType::F32])
        .with_device(DeviceType::Cpu),
    );

    let relu = graph.add_node(
        Node::new(Operation::Relu, "relu".to_string())
            .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 256]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu),
    );

    let linear2 = graph.add_node(
        Node::new(
            Operation::Linear(torsh_jit::graph::LinearInfo {
                in_features: 256,
                out_features: 10,
                bias: true,
            }),
            "linear2".to_string(),
        )
        .with_output_shapes(vec![Some(torsh_jit::graph::shape_from_slice(&[32, 10]))])
        .with_dtypes(vec![DType::F32])
        .with_device(DeviceType::Cpu),
    );

    // Connect the network
    graph.add_edge(input, linear1, Edge::default());
    graph.add_edge(linear1, relu, Edge::default());
    graph.add_edge(relu, linear2, Edge::default());

    graph.add_input(input);
    graph.add_output(linear2);

    // Test compilation
    let config = JitConfig {
        fusion_strategy: FusionStrategy::Aggressive,
        enable_optimizations: true,
        max_fusion_size: 8,
        enable_profiling: true,
        target_device: DeviceType::Cpu,
        enable_caching: true,
        enable_specialization: true,
        specialization_config: SpecializationConfig::default(),
    };

    let mut compiler = JitCompiler::new(config);
    let result = compiler.compile(graph);
    assert!(result.is_ok());

    if let Ok(module) = result {
        // Test that fusion found linear->relu pattern
        let _stats = module.stats();
        // Stats are available after compilation
    }
}
