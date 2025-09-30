# ToRSh FX Tutorial

ToRSh FX is a functional transformation framework that provides graph-based symbolic execution, optimization passes, and code generation capabilities for the ToRSh deep learning framework.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Graph Construction](#basic-graph-construction)
3. [Symbolic Tracing](#symbolic-tracing)
4. [Optimization Passes](#optimization-passes)
5. [Code Generation](#code-generation)
6. [Dynamic Shapes](#dynamic-shapes)
7. [Quantization](#quantization)
8. [Distributed Execution](#distributed-execution)
9. [Advanced Features](#advanced-features)

## Getting Started

Add torsh-fx to your `Cargo.toml`:

```toml
[dependencies]
torsh-fx = "0.1.0-alpha.1"
```

Import the basic components:

```rust
use torsh_fx::{
    FxGraph, ModuleTracer, Node, Edge,
    PassManager, OperationFusionPass, DeadCodeEliminationPass,
    CodeGenerator, OnnxExporter
};
```

## Basic Graph Construction

### Creating a Simple Graph

```rust
use torsh_fx::{FxGraph, ModuleTracer};

fn create_simple_graph() -> FxGraph {
    let mut tracer = ModuleTracer::new();
    
    // Add input node
    let input = tracer.add_input("x");
    
    // Add operations
    let linear = tracer.add_call("linear", vec!["x".to_string()]);
    let relu = tracer.add_call("relu", vec!["node_0".to_string()]);
    
    // Add output
    let output = tracer.add_output("node_1");
    
    tracer.finalize()
}

fn main() {
    let graph = create_simple_graph();
    
    // Print graph information
    graph.print();
    println!("Graph has {} nodes and {} edges", 
             graph.node_count(), graph.edge_count());
}
```

### Working with Graph Serialization

```rust
use torsh_fx::FxGraph;

fn serialization_example() -> Result<(), Box<dyn std::error::Error>> {
    let graph = create_simple_graph();
    
    // Serialize to JSON
    let json = graph.to_json()?;
    println!("JSON representation:\n{}", json);
    
    // Serialize to binary
    let binary = graph.to_binary()?;
    println!("Binary size: {} bytes", binary.len());
    
    // Deserialize from JSON
    let restored_graph = FxGraph::from_json(&json)?;
    assert_eq!(graph.node_count(), restored_graph.node_count());
    
    Ok(())
}
```

## Symbolic Tracing

### Basic Module Tracing

```rust
use torsh_fx::{ModuleTracer, SymbolicTensor, TracingProxy};

// Example module implementation
struct LinearModule {
    input_size: usize,
    output_size: usize,
}

impl torsh_fx::Module for LinearModule {
    fn forward(&self, inputs: &[String]) -> torsh_fx::TorshResult<Vec<String>> {
        // Basic implementation
        Ok(vec!["linear_output".to_string()])
    }
}

// Trace a module
fn trace_module() -> torsh_fx::TorshResult<FxGraph> {
    let module = LinearModule {
        input_size: 784,
        output_size: 10,
    };
    
    // Create tracer and trace the module
    torsh_fx::trace(&module)
}
```

### Control Flow Tracing

```rust
fn trace_with_control_flow() -> FxGraph {
    let mut tracer = ModuleTracer::new();
    
    // Input
    let input = tracer.add_input("x");
    
    // Condition: x > 0
    let condition = tracer.add_call("gt", vec!["x".to_string()]);
    
    // Then branch: relu(x)
    let then_result = tracer.add_call("relu", vec!["x".to_string()]);
    
    // Else branch: sigmoid(x)  
    let else_result = tracer.add_call("sigmoid", vec!["x".to_string()]);
    
    // Conditional node
    let conditional = tracer.add_conditional(
        "node_0",
        vec!["node_1".to_string()],
        vec!["node_2".to_string()]
    );
    
    // Output
    let output = tracer.add_output("node_3");
    
    tracer.finalize()
}
```

### Loop Tracing

```rust
fn trace_with_loop() -> FxGraph {
    let mut tracer = ModuleTracer::new();
    
    // Input and initial state
    let input = tracer.add_input("x");
    let counter = tracer.add_input("counter");
    
    // Loop condition: counter < 10
    let condition = tracer.add_call("lt", vec!["counter".to_string()]);
    
    // Loop body: x = x + 1, counter = counter + 1
    let add_x = tracer.add_call("add", vec!["x".to_string()]);
    let add_counter = tracer.add_call("add", vec!["counter".to_string()]);
    
    // Loop node
    let loop_node = tracer.add_loop(
        "node_0",
        vec!["node_1".to_string(), "node_2".to_string()],
        vec!["x".to_string(), "counter".to_string()]
    );
    
    // Output
    let output = tracer.add_output("node_3");
    
    tracer.finalize()
}
```

## Optimization Passes

### Applying Individual Passes

```rust
use torsh_fx::{
    PassManager, OperationFusionPass, DeadCodeEliminationPass, 
    ConstantFoldingPass, CommonSubexpressionEliminationPass
};

fn apply_optimization_passes() -> torsh_fx::TorshResult<()> {
    let mut graph = create_simple_graph();
    
    // Apply operation fusion
    let fusion_pass = OperationFusionPass;
    fusion_pass.apply(&mut graph)?;
    
    // Apply dead code elimination
    let dce_pass = DeadCodeEliminationPass;
    dce_pass.apply(&mut graph)?;
    
    // Apply constant folding
    let cf_pass = ConstantFoldingPass;
    cf_pass.apply(&mut graph)?;
    
    println!("Optimized graph has {} nodes", graph.node_count());
    Ok(())
}
```

### Using Pass Manager

```rust
fn optimize_with_pass_manager() -> torsh_fx::TorshResult<()> {
    let mut graph = create_simple_graph();
    
    // Create default optimization pipeline
    let pass_manager = PassManager::default_optimization_passes();
    
    // Run all passes
    pass_manager.run(&mut graph)?;
    
    println!("Optimization complete!");
    Ok(())
}

fn aggressive_optimization() -> torsh_fx::TorshResult<()> {
    let mut graph = create_simple_graph();
    
    // Use aggressive optimization pipeline
    let pass_manager = PassManager::aggressive_optimization_passes();
    pass_manager.run(&mut graph)?;
    
    println!("Aggressive optimization complete!");
    Ok(())
}
```

### Custom Pass Implementation

```rust
use torsh_fx::{Pass, FxGraph, Node, TorshResult};

struct CustomOptimizationPass;

impl Pass for CustomOptimizationPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Custom optimization logic
        let mut optimizations = 0;
        
        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, _) = node {
                if op_name == "relu" {
                    // Example: replace relu with leaky_relu for better gradients
                    if let Some(Node::Call(ref mut current_op, ref _args)) = 
                        graph.graph.node_weight_mut(idx) {
                        *current_op = "leaky_relu".to_string();
                        optimizations += 1;
                    }
                }
            }
        }
        
        println!("Applied {} custom optimizations", optimizations);
        Ok(())
    }
    
    fn name(&self) -> &str {
        "custom_optimization"
    }
}

fn use_custom_pass() -> TorshResult<()> {
    let mut graph = create_simple_graph();
    let custom_pass = CustomOptimizationPass;
    custom_pass.apply(&mut graph)?;
    Ok(())
}
```

## Code Generation

### Python Code Generation

```rust
use torsh_fx::CodeGenerator;

fn generate_python_code() -> torsh_fx::TorshResult<()> {
    let graph = create_simple_graph();
    
    // Generate PyTorch code
    let pytorch_code = graph.to_python()?;
    println!("PyTorch code:\n{}", pytorch_code);
    
    // Generate NumPy code
    let generator = CodeGenerator::new();
    let numpy_code = generator.generate_code(&graph, "numpy")?;
    println!("NumPy code:\n{}", numpy_code);
    
    Ok(())
}
```

### C++ Code Generation

```rust
fn generate_cpp_code() -> torsh_fx::TorshResult<()> {
    let graph = create_simple_graph();
    
    // Generate LibTorch C++ code
    let libtorch_code = graph.to_cpp()?;
    println!("LibTorch code:\n{}", libtorch_code);
    
    // Generate standard C++ code
    let generator = CodeGenerator::new();
    let std_cpp_code = generator.generate_code(&graph, "std_cpp")?;
    println!("Standard C++ code:\n{}", std_cpp_code);
    
    Ok(())
}
```

### Hardware-Specific Code Generation

```rust
fn generate_hardware_specific_code() -> torsh_fx::TorshResult<()> {
    let graph = create_simple_graph();
    let generator = CodeGenerator::new();
    
    // Generate CUDA kernels
    let cuda_code = generator.generate_code(&graph, "cuda")?;
    println!("CUDA code:\n{}", cuda_code);
    
    // Generate TensorRT code
    let tensorrt_code = generator.generate_code(&graph, "tensorrt")?;
    println!("TensorRT code:\n{}", tensorrt_code);
    
    // Generate XLA HLO
    let xla_code = generator.generate_code(&graph, "xla")?;
    println!("XLA HLO code:\n{}", xla_code);
    
    Ok(())
}
```

## Dynamic Shapes

### Working with Dynamic Dimensions

```rust
use torsh_fx::{DynamicDim, DynamicShape, DynamicShapeInfo};

fn dynamic_shapes_example() -> torsh_fx::TorshResult<()> {
    // Create dynamic dimensions
    let batch_dim = DynamicDim::new("batch", Some(1), Some(1024));
    let seq_len = DynamicDim::new("seq_len", Some(1), Some(512));
    
    // Create dynamic shape
    let dynamic_shape = DynamicShape::new(vec![
        batch_dim,
        seq_len,
        DynamicDim::new("hidden", Some(768), Some(768)), // Fixed dimension
    ]);
    
    // Create shape info with constraints
    let shape_info = DynamicShapeInfo::new(dynamic_shape);
    
    println!("Dynamic shape: {:?}", shape_info);
    Ok(())
}
```

### Shape Constraints

```rust
use torsh_fx::{ShapeConstraint, DynamicShapeInferenceContext};

fn shape_constraints_example() -> torsh_fx::TorshResult<()> {
    let mut context = DynamicShapeInferenceContext::new();
    
    // Add constraints
    context.add_constraint(ShapeConstraint::Equal("batch".to_string(), "batch2".to_string()))?;
    context.add_constraint(ShapeConstraint::Divisible("seq_len".to_string(), 8))?;
    
    // Validate constraints
    let is_valid = context.validate_constraints();
    println!("Constraints valid: {}", is_valid);
    
    Ok(())
}
```

## Quantization

### Post-Training Quantization

```rust
use torsh_fx::{QuantizationAnnotation, QuantizationParams, CalibrationData};

fn post_training_quantization() -> torsh_fx::TorshResult<()> {
    let mut graph = create_simple_graph();
    
    // Create quantization parameters
    let quant_params = QuantizationParams {
        bits: 8,
        scale: 0.1,
        zero_point: 128,
        scheme: "symmetric".to_string(),
    };
    
    // Add quantization annotations
    for (idx, node) in graph.nodes() {
        if let Node::Call(op_name, _) = node {
            if op_name == "linear" || op_name == "conv2d" {
                // Annotate quantizable operations
                let annotation = QuantizationAnnotation::new(
                    idx,
                    quant_params.clone(),
                    true // quantize weights
                );
            }
        }
    }
    
    println!("Added quantization annotations");
    Ok(())
}
```

### Quantization-Aware Training (QAT)

```rust
use torsh_fx::QATUtils;

fn quantization_aware_training() -> torsh_fx::TorshResult<()> {
    let mut graph = create_simple_graph();
    
    // Prepare graph for QAT
    QATUtils::prepare_qat(&mut graph)?;
    
    // The graph now has fake quantization nodes inserted
    println!("Graph prepared for QAT with {} nodes", graph.node_count());
    
    // After training, convert to quantized model
    let quantized_graph = QATUtils::convert_to_quantized(&graph)?;
    println!("Converted to quantized model with {} nodes", 
             quantized_graph.node_count());
    
    Ok(())
}
```

## Distributed Execution

### Creating Distributed Execution Plans

```rust
use torsh_fx::{
    DistributedConfig, DistributedExecutor, DistributionStrategy,
    CommunicationBackendType, create_execution_plan
};

fn distributed_execution_example() -> torsh_fx::TorshResult<()> {
    let graph = create_simple_graph();
    
    // Configure distributed execution
    let config = DistributedConfig {
        world_size: 4,
        rank: 0,
        backend: CommunicationBackendType::NCCL,
        strategy: DistributionStrategy::DataParallel,
    };
    
    // Create execution plan
    let plan = create_execution_plan(&graph, &config)?;
    
    // Execute distributed
    let executor = DistributedExecutor::new(config);
    let results = executor.execute(&plan)?;
    
    println!("Distributed execution completed");
    Ok(())
}
```

### Model Parallelism

```rust
fn model_parallel_example() -> torsh_fx::TorshResult<()> {
    let graph = create_simple_graph();
    
    let config = DistributedConfig {
        world_size: 2,
        rank: 0,
        backend: CommunicationBackendType::NCCL,
        strategy: DistributionStrategy::ModelParallel,
    };
    
    let plan = create_execution_plan(&graph, &config)?;
    println!("Model parallel plan created with {} partitions", 
             plan.partitions.len());
    
    Ok(())
}
```

## Advanced Features

### ONNX Export

```rust
use torsh_fx::{OnnxExporter, export_to_onnx};

fn onnx_export_example() -> torsh_fx::TorshResult<()> {
    let graph = create_simple_graph();
    
    // Export to ONNX
    let onnx_model = export_to_onnx(&graph, Some("my_model".to_string()))?;
    
    // Save to file
    onnx_model.save_to_file("model.onnx")?;
    
    // Export to JSON format
    let json_model = graph.to_onnx_json()?;
    println!("ONNX JSON: {}", json_model);
    
    Ok(())
}
```

### Graph Visualization

```rust
use torsh_fx::GraphDebugger;

fn visualization_example() -> torsh_fx::TorshResult<()> {
    let graph = create_simple_graph();
    
    // Create debugger
    let debugger = GraphDebugger::new(&graph);
    
    // Generate text visualization
    let text_viz = debugger.generate_text_visualization()?;
    println!("Text visualization:\n{}", text_viz);
    
    // Generate DOT format for Graphviz
    let dot_viz = debugger.generate_dot_visualization()?;
    println!("DOT visualization:\n{}", dot_viz);
    
    // Generate HTML visualization
    let html_viz = debugger.generate_html_visualization()?;
    std::fs::write("graph_viz.html", html_viz)?;
    
    Ok(())
}
```

### Custom Backend Integration

```rust
use torsh_fx::{
    CustomBackend, BackendRegistry, BackendCapability, BackendInfo,
    register_backend_factory, execute_with_backend
};

// Implement custom backend
struct MyCustomBackend;

impl CustomBackend for MyCustomBackend {
    fn execute(&self, graph: &FxGraph) -> torsh_fx::BackendResult<Vec<String>> {
        // Custom execution logic
        println!("Executing on custom backend");
        Ok(vec!["result".to_string()])
    }
    
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "my_backend".to_string(),
            version: "1.0.0".to_string(),
            capabilities: vec![
                BackendCapability::GraphExecution,
                BackendCapability::Optimization,
            ],
            device_types: vec!["custom_device".to_string()],
        }
    }
}

fn custom_backend_example() -> torsh_fx::TorshResult<()> {
    let graph = create_simple_graph();
    
    // Register custom backend
    register_backend_factory("my_backend", Box::new(|| {
        Box::new(MyCustomBackend)
    }));
    
    // Execute on custom backend
    let results = execute_with_backend(&graph, "my_backend")?;
    println!("Custom backend results: {:?}", results);
    
    Ok(())
}
```

## Best Practices

1. **Graph Construction**: Use `ModuleTracer` for systematic graph building
2. **Optimization**: Apply optimization passes in the correct order
3. **Memory Management**: Use in-place operations when possible
4. **Dynamic Shapes**: Add appropriate constraints for shape validation
5. **Quantization**: Use calibration data for better quantization accuracy
6. **Error Handling**: Always handle `TorshResult` properly
7. **Testing**: Write tests for custom passes and backends

## Conclusion

ToRSh FX provides a comprehensive framework for graph transformations, optimizations, and code generation. This tutorial covered the basic usage patterns, but the framework supports many more advanced features for production deployment.

For more detailed examples, see the `examples/` directory in the ToRSh repository.