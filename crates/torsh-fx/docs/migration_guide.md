# ToRSh FX Migration Guide

This guide helps you migrate from other graph-based frameworks to ToRSh FX, or upgrade between different versions of ToRSh FX.

## Table of Contents

1. [Migrating from PyTorch FX](#migrating-from-pytorch-fx)
2. [Migrating from TensorFlow Graph](#migrating-from-tensorflow-graph)
3. [Migrating from ONNX](#migrating-from-onnx)
4. [Migrating from TorchScript](#migrating-from-torchscript)
5. [Upgrading ToRSh FX Versions](#upgrading-torsh-fx-versions)
6. [Common Migration Patterns](#common-migration-patterns)
7. [Troubleshooting](#troubleshooting)

## Migrating from PyTorch FX

### Overview

PyTorch FX and ToRSh FX share similar design principles but differ in implementation language and specific APIs.

### Key Differences

| Aspect | PyTorch FX | ToRSh FX |
|--------|------------|----------|
| Language | Python | Rust |
| Type System | Dynamic | Static |
| Memory Management | Garbage Collection | RAII/Ownership |
| Concurrency | GIL limitations | Native parallelism |
| Serialization | Pickle | JSON/Binary |

### Migration Steps

#### 1. Graph Construction

**PyTorch FX:**
```python
import torch
import torch.fx as fx

class SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        return torch.relu(self.linear(x))

module = SimpleModule()
traced = fx.symbolic_trace(module)
```

**ToRSh FX:**
```rust
use torsh_fx::{FxGraph, ModuleTracer};

fn create_simple_module() -> FxGraph {
    let mut tracer = ModuleTracer::new();
    
    let input = tracer.add_input("x");
    let linear = tracer.add_call("linear", vec!["x".to_string()]);
    let relu = tracer.add_call("relu", vec!["node_0".to_string()]);
    let output = tracer.add_output("node_1");
    
    tracer.finalize()
}
```

#### 2. Graph Transformation

**PyTorch FX:**
```python
def replace_relu_with_gelu(gm: fx.GraphModule):
    for node in gm.graph.nodes:
        if node.op == 'call_function' and node.target == torch.relu:
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(torch.nn.functional.gelu, node.args)
                node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
    gm.recompile()
```

**ToRSh FX:**
```rust
use torsh_fx::{Pass, FxGraph, Node};

struct ReluToGeluPass;

impl Pass for ReluToGeluPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        let mut replacements = Vec::new();
        
        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, args) = node {
                if op_name == "relu" {
                    replacements.push((idx, "gelu".to_string(), args.clone()));
                }
            }
        }
        
        for (idx, new_op, args) in replacements {
            if let Some(node) = graph.graph.node_weight_mut(idx) {
                *node = Node::Call(new_op, args);
            }
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "relu_to_gelu"
    }
}
```

#### 3. Optimization Passes

**PyTorch FX:**
```python
from torch.fx.passes.split_module import split_module
from torch.fx.passes.operator_support import OperatorSupport

# Apply optimization passes
optimized = some_optimization_pass(traced)
```

**ToRSh FX:**
```rust
use torsh_fx::{PassManager, OperationFusionPass, DeadCodeEliminationPass};

fn optimize_graph(graph: &mut FxGraph) -> TorshResult<()> {
    let mut pass_manager = PassManager::new();
    pass_manager.add_pass(Box::new(OperationFusionPass));
    pass_manager.add_pass(Box::new(DeadCodeEliminationPass));
    pass_manager.run(graph)
}
```

### Migration Helper Functions

```rust
/// Helper function to convert PyTorch FX node types to ToRSh FX
fn convert_pytorch_node_type(pytorch_op: &str) -> Option<String> {
    match pytorch_op {
        "call_function" => Some("call".to_string()),
        "call_method" => Some("call".to_string()),
        "get_attr" => Some("get_attr".to_string()),
        "placeholder" => Some("input".to_string()),
        "output" => Some("output".to_string()),
        _ => None,
    }
}

/// Convert PyTorch FX graph structure to ToRSh FX
fn convert_from_pytorch_fx(pytorch_graph_json: &str) -> TorshResult<FxGraph> {
    // Parse PyTorch FX JSON representation
    let pytorch_data: serde_json::Value = serde_json::from_str(pytorch_graph_json)?;
    
    let mut tracer = ModuleTracer::new();
    let mut node_mapping = HashMap::new();
    
    // Convert nodes
    if let Some(nodes) = pytorch_data["nodes"].as_array() {
        for node in nodes {
            let node_name = node["name"].as_str().unwrap();
            let op_type = node["op"].as_str().unwrap();
            
            match op_type {
                "placeholder" => {
                    let idx = tracer.add_input(node_name);
                    node_mapping.insert(node_name.to_string(), idx);
                }
                "call_function" | "call_method" => {
                    let target = node["target"].as_str().unwrap();
                    let args = node["args"].as_array()
                        .map(|arr| arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect())
                        .unwrap_or_default();
                    
                    let idx = tracer.add_call(target, args);
                    node_mapping.insert(node_name.to_string(), idx);
                }
                "output" => {
                    let args = node["args"].as_array().unwrap();
                    if let Some(input_name) = args[0].as_str() {
                        let idx = tracer.add_output(input_name);
                        node_mapping.insert(node_name.to_string(), idx);
                    }
                }
                _ => {
                    eprintln!("Warning: Unsupported node type: {}", op_type);
                }
            }
        }
    }
    
    Ok(tracer.finalize())
}
```

## Migrating from TensorFlow Graph

### Key Differences

| Aspect | TensorFlow | ToRSh FX |
|--------|------------|----------|
| Execution | Session-based | Direct execution |
| Variables | Mutable state | Immutable values |
| Control Flow | tf.cond, tf.while_loop | Conditional, Loop nodes |
| Scoping | Variable scopes | Explicit naming |

### Migration Steps

#### 1. Session-based to Direct Execution

**TensorFlow:**
```python
import tensorflow as tf

# Define graph
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Execute with session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(y, feed_dict={x: input_data})
```

**ToRSh FX:**
```rust
fn create_tensorflow_equivalent() -> FxGraph {
    let mut tracer = ModuleTracer::new();
    
    // Inputs
    let x = tracer.add_input("x");
    let W = tracer.add_input("W");
    let b = tracer.add_input("b");
    
    // Operations
    let matmul = tracer.add_call("matmul", vec!["x".to_string(), "W".to_string()]);
    let add = tracer.add_call("add", vec!["node_0".to_string(), "b".to_string()]);
    let softmax = tracer.add_call("softmax", vec!["node_1".to_string()]);
    
    // Output
    let output = tracer.add_output("node_2");
    
    tracer.finalize()
}
```

#### 2. Control Flow Migration

**TensorFlow:**
```python
def cond_example(x):
    return tf.cond(
        tf.greater(x, 0),
        lambda: tf.square(x),
        lambda: tf.abs(x)
    )
```

**ToRSh FX:**
```rust
fn create_conditional_example() -> FxGraph {
    let mut tracer = ModuleTracer::new();
    
    let x = tracer.add_input("x");
    let condition = tracer.add_call("greater", vec!["x".to_string(), "zero".to_string()]);
    
    // True branch
    let square = tracer.add_call("square", vec!["x".to_string()]);
    
    // False branch
    let abs = tracer.add_call("abs", vec!["x".to_string()]);
    
    // Conditional
    let conditional = tracer.add_conditional(
        "node_0",
        vec!["node_1".to_string()],
        vec!["node_2".to_string()]
    );
    
    let output = tracer.add_output("node_3");
    tracer.finalize()
}
```

### TensorFlow to ToRSh FX Converter

```rust
use std::collections::HashMap;

struct TensorFlowConverter {
    node_mapping: HashMap<String, String>,
    operation_mapping: HashMap<String, String>,
}

impl TensorFlowConverter {
    fn new() -> Self {
        let mut operation_mapping = HashMap::new();
        
        // Common operation mappings
        operation_mapping.insert("MatMul".to_string(), "matmul".to_string());
        operation_mapping.insert("Add".to_string(), "add".to_string());
        operation_mapping.insert("Relu".to_string(), "relu".to_string());
        operation_mapping.insert("Softmax".to_string(), "softmax".to_string());
        operation_mapping.insert("Conv2D".to_string(), "conv2d".to_string());
        operation_mapping.insert("MaxPool".to_string(), "max_pool2d".to_string());
        
        Self {
            node_mapping: HashMap::new(),
            operation_mapping,
        }
    }
    
    fn convert_tensorflow_pb(pb_path: &str) -> TorshResult<FxGraph> {
        // This would require parsing TensorFlow's protobuf format
        // For demonstration, we'll show the structure
        
        let mut tracer = ModuleTracer::new();
        
        // Parse protobuf and extract graph definition
        // let graph_def = parse_tensorflow_pb(pb_path)?;
        
        // Convert each node
        // for node in graph_def.nodes {
        //     let torsh_op = self.map_tensorflow_op(&node.op);
        //     let args = self.map_tensorflow_inputs(&node.inputs);
        //     tracer.add_call(&torsh_op, args);
        // }
        
        Ok(tracer.finalize())
    }
    
    fn map_tensorflow_op(&self, tf_op: &str) -> String {
        self.operation_mapping
            .get(tf_op)
            .cloned()
            .unwrap_or_else(|| {
                eprintln!("Warning: Unknown TensorFlow operation: {}", tf_op);
                tf_op.to_lowercase()
            })
    }
}
```

## Migrating from ONNX

### Overview

ONNX provides a standardized format that can be directly imported into ToRSh FX.

### Direct ONNX Import

```rust
use torsh_fx::{OnnxImporter, FxGraph};

fn import_onnx_model(onnx_path: &str) -> TorshResult<FxGraph> {
    let importer = OnnxImporter::new();
    importer.import_from_file(onnx_path)
}

// For ONNX models with custom operations
fn import_onnx_with_custom_ops(onnx_path: &str) -> TorshResult<FxGraph> {
    let mut importer = OnnxImporter::new();
    
    // Register custom operation mappings
    importer.register_custom_op("CustomOp", |inputs, attrs| {
        // Convert custom ONNX operation to ToRSh FX representation
        Ok("custom_torsh_op".to_string())
    });
    
    importer.import_from_file(onnx_path)
}
```

### ONNX Operation Mapping

```rust
struct OnnxOperationMapper {
    mappings: HashMap<String, String>,
}

impl OnnxOperationMapper {
    fn new() -> Self {
        let mut mappings = HashMap::new();
        
        // Core ONNX operations
        mappings.insert("Add".to_string(), "add".to_string());
        mappings.insert("Sub".to_string(), "sub".to_string());
        mappings.insert("Mul".to_string(), "mul".to_string());
        mappings.insert("Div".to_string(), "div".to_string());
        mappings.insert("MatMul".to_string(), "matmul".to_string());
        mappings.insert("Relu".to_string(), "relu".to_string());
        mappings.insert("Sigmoid".to_string(), "sigmoid".to_string());
        mappings.insert("Tanh".to_string(), "tanh".to_string());
        mappings.insert("Softmax".to_string(), "softmax".to_string());
        mappings.insert("Conv".to_string(), "conv2d".to_string());
        mappings.insert("MaxPool".to_string(), "max_pool2d".to_string());
        mappings.insert("GlobalAveragePool".to_string(), "global_avg_pool2d".to_string());
        mappings.insert("BatchNormalization".to_string(), "batch_norm".to_string());
        mappings.insert("Reshape".to_string(), "reshape".to_string());
        mappings.insert("Transpose".to_string(), "transpose".to_string());
        mappings.insert("Concat".to_string(), "concat".to_string());
        mappings.insert("Split".to_string(), "split".to_string());
        
        Self { mappings }
    }
    
    fn map_operation(&self, onnx_op: &str) -> Option<&str> {
        self.mappings.get(onnx_op).map(|s| s.as_str())
    }
}
```

## Migrating from TorchScript

### Overview

TorchScript graphs can be converted to ToRSh FX with some manual adaptation.

### TorchScript to ToRSh FX

```rust
use torsh_fx::{TorchScriptImporter, FxGraph};

fn import_torchscript_model(model_path: &str) -> TorshResult<FxGraph> {
    let importer = TorchScriptImporter::new();
    importer.import_from_file(model_path)
}

// Manual conversion for complex TorchScript models
fn convert_torchscript_manually() -> FxGraph {
    let mut tracer = ModuleTracer::new();
    
    // TorchScript often has more complex control flow
    // Convert to structured control flow in ToRSh FX
    
    let input = tracer.add_input("input");
    
    // Convert TorchScript prim::If to Conditional
    let condition = tracer.add_call("condition_check", vec!["input".to_string()]);
    let then_branch = tracer.add_call("then_operation", vec!["input".to_string()]);
    let else_branch = tracer.add_call("else_operation", vec!["input".to_string()]);
    
    let conditional = tracer.add_conditional(
        "node_0",
        vec!["node_1".to_string()],
        vec!["node_2".to_string()]
    );
    
    let output = tracer.add_output("node_3");
    tracer.finalize()
}
```

## Upgrading ToRSh FX Versions

### Version 0.1 to 1.0

#### Breaking Changes

1. **Node API Changes**
```rust
// Old (0.1)
Node::Call(String, Vec<String>)

// New (1.0)
Node::Call {
    operation: String,
    arguments: Vec<String>,
    attributes: HashMap<String, AttributeValue>,
}
```

2. **Pass Interface Changes**
```rust
// Old (0.1)
trait Pass {
    fn apply(&self, graph: &mut FxGraph) -> Result<(), PassError>;
}

// New (1.0)
trait Pass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()>;
    fn name(&self) -> &str;
    fn dependencies(&self) -> Vec<&str> { vec![] }
}
```

#### Migration Script

```rust
/// Migrate graphs from version 0.1 to 1.0
fn migrate_v01_to_v10(old_graph: &V01Graph) -> TorshResult<FxGraph> {
    let mut tracer = ModuleTracer::new();
    
    // Convert nodes with new structure
    for (idx, old_node) in old_graph.nodes() {
        match old_node {
            V01Node::Call(op, args) => {
                // Convert to new format with attributes
                let mut attributes = HashMap::new();
                
                // Migrate common attributes based on operation type
                match op.as_str() {
                    "conv2d" => {
                        attributes.insert("stride".to_string(), AttributeValue::IntList(vec![1, 1]));
                        attributes.insert("padding".to_string(), AttributeValue::IntList(vec![0, 0]));
                    }
                    "linear" => {
                        // Linear operations might have bias attribute
                        attributes.insert("bias".to_string(), AttributeValue::Bool(true));
                    }
                    _ => {}
                }
                
                tracer.add_call_with_attributes(op, args.clone(), attributes);
            }
            V01Node::Input(name) => {
                tracer.add_input(name);
            }
            V01Node::Output => {
                tracer.add_output("final_output");
            }
        }
    }
    
    Ok(tracer.finalize())
}
```

### Version 1.0 to 2.0

#### New Features

1. **First-class Function Support**
2. **Advanced Type System**
3. **Hardware-specific Annotations**

#### Backward Compatibility

```rust
/// Compatibility layer for v1.0 graphs in v2.0
struct V10CompatibilityLayer;

impl V10CompatibilityLayer {
    fn wrap_v10_graph(v10_graph: V10Graph) -> V20Graph {
        // Automatically upgrade v1.0 graphs to v2.0 format
        let mut v20_graph = V20Graph::new();
        
        // Add type annotations based on inference
        for (idx, node) in v10_graph.nodes() {
            let inferred_type = infer_node_type(&node);
            let annotated_node = add_type_annotation(node, inferred_type);
            v20_graph.add_node(annotated_node);
        }
        
        v20_graph
    }
    
    fn infer_node_type(node: &V10Node) -> TensorType {
        // Implement type inference for v1.0 nodes
        match node {
            V10Node::Call { operation, .. } => {
                match operation.as_str() {
                    "linear" => TensorType::float32_2d(),
                    "conv2d" => TensorType::float32_4d(),
                    _ => TensorType::float32_nd(),
                }
            }
            _ => TensorType::float32_nd(),
        }
    }
}
```

## Common Migration Patterns

### 1. Handling Dynamic Shapes

```rust
// Convert dynamic shapes from other frameworks
fn migrate_dynamic_shapes(framework_shape: &FrameworkShape) -> DynamicShape {
    let mut dims = Vec::new();
    
    for dim in &framework_shape.dimensions {
        match dim {
            FrameworkDim::Fixed(size) => {
                dims.push(DynamicDim::fixed(*size));
            }
            FrameworkDim::Variable(name) => {
                dims.push(DynamicDim::new(name, None, None));
            }
            FrameworkDim::Batch => {
                dims.push(DynamicDim::new("batch", Some(1), Some(1024)));
            }
        }
    }
    
    DynamicShape::new(dims)
}
```

### 2. Operation Mapping

```rust
/// Generic operation mapper for different frameworks
struct OperationMapper {
    mappings: HashMap<(String, String), String>, // (framework, op) -> torsh_op
}

impl OperationMapper {
    fn new() -> Self {
        let mut mappings = HashMap::new();
        
        // PyTorch mappings
        mappings.insert(("pytorch".to_string(), "torch.nn.functional.relu".to_string()), "relu".to_string());
        mappings.insert(("pytorch".to_string(), "torch.matmul".to_string()), "matmul".to_string());
        
        // TensorFlow mappings
        mappings.insert(("tensorflow".to_string(), "tf.nn.relu".to_string()), "relu".to_string());
        mappings.insert(("tensorflow".to_string(), "tf.linalg.matmul".to_string()), "matmul".to_string());
        
        // ONNX mappings
        mappings.insert(("onnx".to_string(), "Relu".to_string()), "relu".to_string());
        mappings.insert(("onnx".to_string(), "MatMul".to_string()), "matmul".to_string());
        
        Self { mappings }
    }
    
    fn map_operation(&self, framework: &str, op: &str) -> Option<String> {
        self.mappings.get(&(framework.to_string(), op.to_string())).cloned()
    }
}
```

### 3. Attribute Conversion

```rust
/// Convert framework-specific attributes to ToRSh FX format
fn convert_attributes(framework: &str, attrs: &FrameworkAttributes) -> HashMap<String, AttributeValue> {
    let mut torsh_attrs = HashMap::new();
    
    match framework {
        "pytorch" => {
            for (key, value) in &attrs.pytorch_attrs {
                let torsh_value = match value {
                    PyTorchValue::Int(i) => AttributeValue::Int(*i),
                    PyTorchValue::Float(f) => AttributeValue::Float(*f),
                    PyTorchValue::IntList(list) => AttributeValue::IntList(list.clone()),
                    PyTorchValue::String(s) => AttributeValue::String(s.clone()),
                    PyTorchValue::Bool(b) => AttributeValue::Bool(*b),
                };
                torsh_attrs.insert(key.clone(), torsh_value);
            }
        }
        "tensorflow" => {
            // Convert TensorFlow attributes
            for (key, value) in &attrs.tf_attrs {
                let torsh_value = convert_tf_attribute(value);
                torsh_attrs.insert(key.clone(), torsh_value);
            }
        }
        "onnx" => {
            // Convert ONNX attributes
            for (key, value) in &attrs.onnx_attrs {
                let torsh_value = convert_onnx_attribute(value);
                torsh_attrs.insert(key.clone(), torsh_value);
            }
        }
        _ => {
            eprintln!("Warning: Unknown framework for attribute conversion: {}", framework);
        }
    }
    
    torsh_attrs
}
```

## Troubleshooting

### Common Issues

#### 1. Type Mismatches

**Problem:** Operations expecting different tensor types.

**Solution:**
```rust
fn fix_type_mismatches(graph: &mut FxGraph) -> TorshResult<()> {
    let type_checker = TypeChecker::new();
    let type_errors = type_checker.check_graph(graph)?;
    
    for error in type_errors {
        match error {
            TypeError::IncompatibleTypes { node_idx, expected, actual } => {
                // Insert type conversion node
                insert_type_conversion(graph, node_idx, &expected, &actual)?;
            }
            TypeError::MissingTypeInfo { node_idx } => {
                // Infer and add type information
                let inferred_type = infer_node_type(graph, node_idx)?;
                add_type_annotation(graph, node_idx, inferred_type)?;
            }
        }
    }
    
    Ok(())
}
```

#### 2. Control Flow Issues

**Problem:** Complex control flow not mapping directly.

**Solution:**
```rust
fn restructure_control_flow(graph: &mut FxGraph) -> TorshResult<()> {
    // Identify unstructured control flow
    let cfg_analyzer = ControlFlowAnalyzer::new();
    let unstructured_regions = cfg_analyzer.find_unstructured_regions(graph)?;
    
    for region in unstructured_regions {
        // Convert to structured control flow
        let structured = convert_to_structured_cf(graph, &region)?;
        replace_region(graph, &region, &structured)?;
    }
    
    Ok(())
}
```

#### 3. Missing Operations

**Problem:** Framework-specific operations not available in ToRSh FX.

**Solution:**
```rust
fn handle_missing_operations(graph: &mut FxGraph) -> TorshResult<()> {
    let missing_ops = find_missing_operations(graph)?;
    
    for (node_idx, op_name) in missing_ops {
        match op_name.as_str() {
            "framework_specific_op" => {
                // Decompose into primitive operations
                let primitive_subgraph = decompose_operation(&op_name)?;
                replace_node_with_subgraph(graph, node_idx, primitive_subgraph)?;
            }
            _ => {
                // Register as custom operation
                register_custom_operation(&op_name)?;
            }
        }
    }
    
    Ok(())
}
```

### Migration Validation

```rust
/// Validate that migration was successful
fn validate_migration(original: &OriginalGraph, migrated: &FxGraph) -> TorshResult<()> {
    // Check structural properties
    assert_eq!(count_operations(original), count_operations_fx(migrated));
    
    // Check semantic equivalence with test inputs
    let test_inputs = generate_test_inputs(original)?;
    let original_outputs = execute_original(original, &test_inputs)?;
    let migrated_outputs = execute_fx(migrated, &test_inputs)?;
    
    compare_outputs(&original_outputs, &migrated_outputs)?;
    
    Ok(())
}

fn compare_outputs(original: &[Tensor], migrated: &[Tensor]) -> TorshResult<()> {
    assert_eq!(original.len(), migrated.len());
    
    for (orig, migr) in original.iter().zip(migrated.iter()) {
        let diff = compute_tensor_difference(orig, migr)?;
        if diff > 1e-5 {
            return Err(TorshError::ValidationError {
                message: format!("Output difference too large: {}", diff),
            });
        }
    }
    
    Ok(())
}
```

### Performance Comparison

```rust
/// Compare performance before and after migration
fn benchmark_migration(original: &OriginalGraph, migrated: &FxGraph) -> TorshResult<()> {
    let test_inputs = generate_performance_test_inputs()?;
    
    // Benchmark original
    let original_time = benchmark_execution(|| {
        execute_original(original, &test_inputs)
    })?;
    
    // Benchmark migrated
    let migrated_time = benchmark_execution(|| {
        execute_fx(migrated, &test_inputs)
    })?;
    
    println!("Performance comparison:");
    println!("  Original: {:?}", original_time);
    println!("  Migrated: {:?}", migrated_time);
    println!("  Speedup: {:.2}x", 
             original_time.as_secs_f64() / migrated_time.as_secs_f64());
    
    Ok(())
}
```

## Migration Tools

### Automated Migration Script

```rust
/// Automated migration tool
struct MigrationTool {
    source_framework: String,
    target_version: String,
    validation_enabled: bool,
}

impl MigrationTool {
    fn new(source_framework: &str) -> Self {
        Self {
            source_framework: source_framework.to_string(),
            target_version: "1.0".to_string(),
            validation_enabled: true,
        }
    }
    
    fn migrate(&self, input_path: &str, output_path: &str) -> TorshResult<()> {
        println!("Starting migration from {} to ToRSh FX {}", 
                 self.source_framework, self.target_version);
        
        // Step 1: Parse source format
        let source_graph = self.parse_source_format(input_path)?;
        
        // Step 2: Convert to ToRSh FX
        let mut fx_graph = self.convert_to_fx(&source_graph)?;
        
        // Step 3: Apply optimizations
        self.apply_migration_optimizations(&mut fx_graph)?;
        
        // Step 4: Validate if enabled
        if self.validation_enabled {
            self.validate_conversion(&source_graph, &fx_graph)?;
        }
        
        // Step 5: Save result
        self.save_fx_graph(&fx_graph, output_path)?;
        
        println!("Migration completed successfully!");
        Ok(())
    }
    
    fn parse_source_format(&self, path: &str) -> TorshResult<SourceGraph> {
        match self.source_framework.as_str() {
            "pytorch" => self.parse_pytorch_fx(path),
            "tensorflow" => self.parse_tensorflow_pb(path),
            "onnx" => self.parse_onnx_model(path),
            _ => Err(TorshError::UnsupportedFormat {
                format: self.source_framework.clone(),
            }),
        }
    }
}

// Command-line interface
fn main() -> TorshResult<()> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() != 4 {
        eprintln!("Usage: migrate_to_torsh_fx <framework> <input> <output>");
        eprintln!("Supported frameworks: pytorch, tensorflow, onnx");
        std::process::exit(1);
    }
    
    let framework = &args[1];
    let input_path = &args[2];
    let output_path = &args[3];
    
    let migration_tool = MigrationTool::new(framework);
    migration_tool.migrate(input_path, output_path)?;
    
    Ok(())
}
```

This migration guide provides comprehensive coverage for moving to ToRSh FX from other frameworks and upgrading between versions. The key is to understand the conceptual mappings and use the provided tools and patterns to automate as much of the process as possible.