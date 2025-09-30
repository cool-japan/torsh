# torsh-fx

Graph transformation and optimization framework for ToRSh, providing TorchFX-compatible functionality.

## Overview

TorshFX is a toolkit for capturing, analyzing, and transforming PyTorch-style programs. It provides:

- **Graph Capture**: Convert eager mode code to graph representation
- **Graph Transformation**: Modify and optimize computational graphs
- **Symbolic Tracing**: Trace through Python code to build graphs
- **Graph Optimization**: Apply passes for performance improvements
- **Code Generation**: Convert graphs back to executable code

## Usage

### Basic Symbolic Tracing

```rust
use torsh_fx::prelude::*;
use torsh_nn::prelude::*;

// Define a model
struct MyModel {
    linear1: Linear,
    linear2: Linear,
}

impl Module for MyModel {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = F::relu(&x)?;
        self.linear2.forward(&x)
    }
}

// Trace the model
let model = MyModel::new();
let tracer = Tracer::new();
let graph_module = tracer.trace(model)?;

// Print the graph
println!("{}", graph_module.graph);
```

### Graph Transformation

```rust
use torsh_fx::passes::*;

// Apply optimization passes
let optimized = graph_module
    .transform(FuseOperations::new())?
    .transform(EliminateDeadCode::new())?
    .transform(ConstantFolding::new())?;

// Custom transformation
struct ReplaceReluWithGelu;

impl GraphTransform for ReplaceReluWithGelu {
    fn transform(&self, graph: &mut Graph) -> Result<()> {
        for node in graph.nodes_mut() {
            if node.op == "call_function" && node.target == "relu" {
                node.target = "gelu";
            }
        }
        Ok(())
    }
}

let transformed = graph_module.transform(ReplaceReluWithGelu)?;
```

### Subgraph Matching and Rewriting

```rust
use torsh_fx::subgraph::*;

// Define a pattern to match
let pattern = pattern! {
    conv = call_function("conv2d", [x, weight], {});
    bn = call_function("batch_norm", [conv], {});
    relu = call_function("relu", [bn], {});
    output = relu;
};

// Define replacement
let replacement = |matched: &MatchedSubgraph| -> Result<Node> {
    // Create fused conv-bn-relu node
    let fused = Node::call_function(
        "fused_conv_bn_relu",
        vec![matched["x"], matched["weight"]],
        matched.get_kwargs("conv"),
    );
    Ok(fused)
};

// Apply rewriter
let rewriter = SubgraphRewriter::new(pattern, replacement);
let optimized = rewriter.rewrite(&graph_module)?;
```

### Quantization with FX

```rust
use torsh_fx::quantization::*;

// Prepare model for quantization
let prepared = prepare_fx(
    graph_module,
    QuantConfig::default()
        .with_backend("fbgemm")
        .with_activation_observer(MinMaxObserver::new())
        .with_weight_observer(PerChannelMinMaxObserver::new()),
)?;

// Run calibration
for batch in calibration_data {
    prepared.forward(&batch)?;
}

// Convert to quantized model
let quantized = convert_fx(prepared)?;
```

### Graph Analysis

```rust
use torsh_fx::analysis::*;

// Analyze graph properties
let analyzer = GraphAnalyzer::new(&graph_module.graph);

// Get operation count
let op_count = analyzer.count_operations();
println!("Total operations: {}", op_count.total());
println!("Convolutions: {}", op_count.get("conv2d"));

// Analyze shapes
let shape_prop = ShapePropagator::new();
let shapes = shape_prop.propagate(&graph_module, &example_input)?;

// Find bottlenecks
let profiler = GraphProfiler::new();
let profile = profiler.profile(&graph_module, &example_input)?;
let bottlenecks = profile.find_bottlenecks(5);
```

### Custom Graph Passes

```rust
use torsh_fx::passes::{GraphPass, PassManager};

// Define custom pass
struct MyCustomPass {
    // Pass configuration
}

impl GraphPass for MyCustomPass {
    fn run(&self, graph: &mut Graph) -> Result<bool> {
        let mut modified = false;
        
        // Your transformation logic
        for node in graph.nodes_mut() {
            // Modify nodes
            modified = true;
        }
        
        Ok(modified)
    }
}

// Use pass manager
let pass_manager = PassManager::new()
    .add_pass(MyCustomPass::new())
    .add_pass(CommonSubexpressionElimination::new())
    .add_pass(DeadCodeElimination::new());

let optimized = pass_manager.run(graph_module)?;
```

### Interpreter Mode

```rust
use torsh_fx::interpreter::*;

// Create custom interpreter
struct DebugInterpreter {
    base: Interpreter,
}

impl DebugInterpreter {
    fn run_node(&mut self, node: &Node) -> Result<Value> {
        println!("Executing: {:?}", node);
        let result = self.base.run_node(node)?;
        println!("Result shape: {:?}", result.shape());
        Ok(result)
    }
}

// Run with custom interpreter
let interpreter = DebugInterpreter::new(graph_module);
let output = interpreter.run(&input)?;
```

### Serialization

```rust
// Save graph module
graph_module.save("model.fx")?;

// Load graph module
let loaded = GraphModule::load("model.fx")?;

// Export to ONNX-like format
let exported = graph_module.export()?;
```

## Graph IR

The FX intermediate representation (IR) consists of:

- **Nodes**: Individual operations (placeholder, call_function, call_method, call_module, output)
- **Graph**: DAG of nodes representing computation
- **GraphModule**: Combination of graph and module state

## Integration with JIT

```rust
// Convert FX graph to JIT
let jit_module = torsh_jit::compile_fx(graph_module)?;

// Optimize with JIT
let optimized = jit_module.optimize()?;
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.