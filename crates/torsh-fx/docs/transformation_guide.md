# ToRSh FX Transformation Guide

This guide provides comprehensive documentation on graph transformations in ToRSh FX, including optimization passes, pattern matching, and custom transformations.

## Table of Contents

1. [Transformation Overview](#transformation-overview)
2. [Built-in Optimization Passes](#built-in-optimization-passes)
3. [Pass Manager](#pass-manager)
4. [Pattern Matching](#pattern-matching)
5. [Custom Pass Development](#custom-pass-development)
6. [Subgraph Rewriting](#subgraph-rewriting)
7. [Advanced Transformations](#advanced-transformations)
8. [Performance Considerations](#performance-considerations)

## Transformation Overview

ToRSh FX provides a powerful transformation framework that allows you to optimize and modify computational graphs through various passes. Transformations can:

- Optimize performance through operation fusion
- Reduce memory usage through in-place operations
- Eliminate dead code and common subexpressions
- Simplify mathematical expressions
- Prepare graphs for specific hardware targets

### Core Concepts

- **Pass**: A transformation that modifies a graph
- **Pattern**: A subgraph structure to match and replace
- **Rewriter**: A component that applies pattern-based transformations
- **Pass Manager**: Orchestrates the execution of multiple passes

## Built-in Optimization Passes

### Operation Fusion Pass

Combines adjacent operations to reduce kernel launch overhead and improve memory locality.

```rust
use torsh_fx::{OperationFusionPass, FxGraph};

fn apply_operation_fusion(graph: &mut FxGraph) -> torsh_fx::TorshResult<()> {
    let fusion_pass = OperationFusionPass;
    fusion_pass.apply(graph)?;
    
    println!("Applied operation fusion");
    Ok(())
}
```

**Common Fusion Patterns:**
- `linear` + `relu` → `linear_relu`
- `conv2d` + `batch_norm` → `conv2d_bn`
- `matmul` + `bias_add` → `linear`

### Dead Code Elimination Pass

Removes nodes that don't contribute to the final output.

```rust
use torsh_fx::{DeadCodeEliminationPass, FxGraph};

fn eliminate_dead_code(graph: &mut FxGraph) -> torsh_fx::TorshResult<()> {
    let dce_pass = DeadCodeEliminationPass;
    dce_pass.apply(graph)?;
    
    println!("Eliminated dead code");
    Ok(())
}
```

**What Gets Eliminated:**
- Unreachable nodes
- Unused intermediate results
- Disconnected subgraphs

### Constant Folding Pass

Evaluates constant expressions at compile time.

```rust
use torsh_fx::{ConstantFoldingPass, FxGraph};

fn fold_constants(graph: &mut FxGraph) -> torsh_fx::TorshResult<()> {
    let cf_pass = ConstantFoldingPass;
    cf_pass.apply(graph)?;
    
    println!("Folded constants");
    Ok(())
}
```

**Examples:**
- `add(2, 3)` → `5`
- `mul(x, 1)` → `x`
- `add(x, 0)` → `x`

### Common Subexpression Elimination (CSE)

Identifies and eliminates redundant computations.

```rust
use torsh_fx::{CommonSubexpressionEliminationPass, FxGraph};

fn eliminate_common_subexpressions(graph: &mut FxGraph) -> torsh_fx::TorshResult<()> {
    let cse_pass = CommonSubexpressionEliminationPass;
    cse_pass.apply(graph)?;
    
    println!("Eliminated common subexpressions");
    Ok(())
}
```

**Example:**
```
x = a + b
y = a + b  # Redundant
z = x * 2
```
After CSE:
```
x = a + b
y = x      # Reuse x
z = x * 2
```

### Memory Optimization Pass

Optimizes memory usage through in-place operations.

```rust
use torsh_fx::{MemoryOptimizationPass, FxGraph};

fn optimize_memory(graph: &mut FxGraph) -> torsh_fx::TorshResult<()> {
    let mem_pass = MemoryOptimizationPass;
    mem_pass.apply(graph)?;
    
    println!("Optimized memory usage");
    Ok(())
}
```

**Optimizations:**
- In-place operations for elementwise functions
- Tensor lifetime analysis
- Memory layout optimization

### Graph Simplification Pass

Simplifies mathematical expressions using algebraic identities.

```rust
use torsh_fx::{GraphSimplificationPass, FxGraph};

fn simplify_graph(graph: &mut FxGraph) -> torsh_fx::TorshResult<()> {
    let simp_pass = GraphSimplificationPass;
    simp_pass.apply(graph)?;
    
    println!("Simplified graph");
    Ok(())
}
```

**Simplification Rules:**
- `x + 0` → `x`
- `x * 1` → `x`
- `x * 0` → `0`
- `x - x` → `0`

### Loop Optimization Pass

Optimizes loop constructs through unrolling and vectorization.

```rust
use torsh_fx::{LoopOptimizationPass, FxGraph};

fn optimize_loops(graph: &mut FxGraph) -> torsh_fx::TorshResult<()> {
    let loop_pass = LoopOptimizationPass;
    loop_pass.apply(graph)?;
    
    println!("Optimized loops");
    Ok(())
}
```

**Optimizations:**
- Loop unrolling for small iterations
- Vectorization of element-wise operations
- Loop fusion when possible

## Pass Manager

The Pass Manager orchestrates the execution of multiple optimization passes.

### Default Optimization Pipeline

```rust
use torsh_fx::{PassManager, FxGraph};

fn run_default_optimizations(graph: &mut FxGraph) -> torsh_fx::TorshResult<()> {
    let pass_manager = PassManager::default_optimization_passes();
    pass_manager.run(graph)?;
    
    println!("Applied default optimizations");
    Ok(())
}
```

**Default Pipeline:**
1. Graph Simplification
2. Constant Folding
3. Common Subexpression Elimination
4. Dead Code Elimination
5. Operation Fusion
6. Memory Optimization
7. Loop Optimization

### Aggressive Optimization Pipeline

```rust
fn run_aggressive_optimizations(graph: &mut FxGraph) -> torsh_fx::TorshResult<()> {
    let pass_manager = PassManager::aggressive_optimization_passes();
    pass_manager.run(graph)?;
    
    println!("Applied aggressive optimizations");
    Ok(())
}
```

**Aggressive Pipeline:**
- Multiple rounds of optimization
- More aggressive fusion patterns
- Advanced loop transformations

### Custom Pipeline

```rust
use torsh_fx::{PassManager, OperationFusionPass, DeadCodeEliminationPass};

fn create_custom_pipeline() -> PassManager {
    let mut manager = PassManager::new();
    
    // Add passes in specific order
    manager.add_pass(Box::new(OperationFusionPass));
    manager.add_pass(Box::new(DeadCodeEliminationPass));
    // Add more passes as needed
    
    manager
}

fn run_custom_pipeline(graph: &mut FxGraph) -> torsh_fx::TorshResult<()> {
    let pass_manager = create_custom_pipeline();
    pass_manager.run(graph)?;
    
    Ok(())
}
```

## Pattern Matching

ToRSh FX provides sophisticated pattern matching for subgraph transformations.

### Basic Pattern Matching

```rust
use torsh_fx::{PatternMatcher, SubgraphPattern};

fn create_fusion_pattern() -> SubgraphPattern {
    // Pattern: linear -> relu
    SubgraphPattern::new()
        .add_node("linear", "linear")
        .add_node("relu", "relu")
        .add_edge("linear", "relu")
        .set_root("relu")
}

fn apply_pattern_matching(graph: &mut FxGraph) -> torsh_fx::TorshResult<()> {
    let pattern = create_fusion_pattern();
    let matcher = PatternMatcher::new();
    
    let matches = matcher.find_matches(graph, &pattern)?;
    println!("Found {} pattern matches", matches.len());
    
    Ok(())
}
```

### Complex Patterns

```rust
fn create_attention_pattern() -> SubgraphPattern {
    // Pattern: Multi-head attention
    SubgraphPattern::new()
        .add_node("q_proj", "linear")
        .add_node("k_proj", "linear")
        .add_node("v_proj", "linear")
        .add_node("attention", "scaled_dot_product_attention")
        .add_node("output_proj", "linear")
        .add_edges(&[
            ("q_proj", "attention"),
            ("k_proj", "attention"),
            ("v_proj", "attention"),
            ("attention", "output_proj"),
        ])
        .set_root("output_proj")
}
```

### Pattern Replacement

```rust
use torsh_fx::{SubgraphRewriter, ReplacementPattern};

fn create_replacement_pattern() -> ReplacementPattern {
    ReplacementPattern::new()
        .replace_subgraph("linear_relu_pattern")
        .with_single_node("fused_linear_relu")
        .preserve_inputs()
        .preserve_outputs()
}

fn apply_pattern_replacement(graph: &mut FxGraph) -> torsh_fx::TorshResult<()> {
    let pattern = create_fusion_pattern();
    let replacement = create_replacement_pattern();
    let rewriter = SubgraphRewriter::new();
    
    rewriter.apply_replacement(graph, &pattern, &replacement)?;
    
    Ok(())
}
```

## Custom Pass Development

### Implementing the Pass Trait

```rust
use torsh_fx::{Pass, FxGraph, Node, TorshResult};

struct BatchNormFusionPass;

impl Pass for BatchNormFusionPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        let mut fusions = Vec::new();
        
        // Find conv2d -> batch_norm patterns
        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, _) = node {
                if op_name == "batch_norm" {
                    // Check for preceding conv2d
                    let predecessors: Vec<_> = graph.graph
                        .neighbors_directed(idx, petgraph::Direction::Incoming)
                        .collect();
                    
                    for pred_idx in predecessors {
                        if let Some(Node::Call(pred_op, _)) = graph.get_node(pred_idx) {
                            if pred_op == "conv2d" {
                                fusions.push((pred_idx, idx));
                            }
                        }
                    }
                }
            }
        }
        
        // Apply fusions
        for (conv_idx, bn_idx) in fusions {
            self.fuse_conv_bn(graph, conv_idx, bn_idx)?;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "batch_norm_fusion"
    }
}

impl BatchNormFusionPass {
    fn fuse_conv_bn(
        &self, 
        graph: &mut FxGraph, 
        conv_idx: petgraph::graph::NodeIndex,
        bn_idx: petgraph::graph::NodeIndex
    ) -> TorshResult<()> {
        // Implementation of conv-bn fusion
        if let Some(Node::Call(ref mut op_name, ref args)) = graph.graph.node_weight_mut(conv_idx) {
            *op_name = "conv2d_bn".to_string();
            
            // Redirect edges from bn to conv
            let successors: Vec<_> = graph.graph
                .neighbors_directed(bn_idx, petgraph::Direction::Outgoing)
                .collect();
            
            for succ in successors {
                if let Some(edge_idx) = graph.graph.find_edge(bn_idx, succ) {
                    let edge = graph.graph[edge_idx].clone();
                    graph.graph.remove_edge(edge_idx);
                    graph.graph.add_edge(conv_idx, succ, edge);
                }
            }
            
            // Remove batch norm node
            graph.graph.remove_node(bn_idx);
        }
        
        Ok(())
    }
}
```

### Stateful Passes

```rust
struct QuantizationPass {
    calibration_data: Vec<f32>,
    quantization_scheme: String,
}

impl QuantizationPass {
    fn new(calibration_data: Vec<f32>) -> Self {
        Self {
            calibration_data,
            quantization_scheme: "symmetric".to_string(),
        }
    }
    
    fn calculate_scale(&self, tensor_name: &str) -> f32 {
        // Calculate quantization scale from calibration data
        let max_val = self.calibration_data.iter()
            .fold(0.0f32, |acc, &x| acc.max(x.abs()));
        max_val / 127.0 // 8-bit symmetric quantization
    }
}

impl Pass for QuantizationPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Add quantization nodes before each operation
        let mut quantization_nodes = Vec::new();
        
        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, args) = node {
                if self.should_quantize(op_name) {
                    // Insert quantization node
                    let scale = self.calculate_scale(&format!("{}_{}", op_name, idx.index()));
                    quantization_nodes.push((idx, scale));
                }
            }
        }
        
        // Insert quantization nodes
        for (node_idx, scale) in quantization_nodes {
            self.insert_quantization_node(graph, node_idx, scale)?;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "quantization"
    }
}

impl QuantizationPass {
    fn should_quantize(&self, op_name: &str) -> bool {
        matches!(op_name, "linear" | "conv2d" | "conv1d")
    }
    
    fn insert_quantization_node(
        &self,
        graph: &mut FxGraph,
        node_idx: petgraph::graph::NodeIndex,
        scale: f32,
    ) -> TorshResult<()> {
        // Implementation details for inserting quantization nodes
        let quant_node = graph.graph.add_node(Node::Call(
            "quantize".to_string(),
            vec![format!("scale_{}", scale)]
        ));
        
        // Update graph connections
        // ... implementation details ...
        
        Ok(())
    }
}
```

## Subgraph Rewriting

### Advanced Rewriting Patterns

```rust
use torsh_fx::{SubgraphRewriter, FusionPattern};

fn create_attention_fusion() -> FusionPattern {
    FusionPattern::new("multi_head_attention")
        .match_sequence(&[
            "linear", // Q projection
            "linear", // K projection  
            "linear", // V projection
            "scaled_dot_product_attention",
            "linear", // Output projection
        ])
        .with_constraints(&[
            "input_dim_match",
            "head_count_consistent",
        ])
        .replace_with("fused_multi_head_attention")
}

fn apply_attention_fusion(graph: &mut FxGraph) -> TorshResult<()> {
    let fusion_pattern = create_attention_fusion();
    let rewriter = SubgraphRewriter::new();
    
    rewriter.apply_fusion_pattern(graph, &fusion_pattern)?;
    
    Ok(())
}
```

### Conditional Rewriting

```rust
struct ConditionalRewriter {
    conditions: Vec<Box<dyn Fn(&FxGraph, &SubgraphMatch) -> bool>>,
}

impl ConditionalRewriter {
    fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }
    
    fn add_condition<F>(mut self, condition: F) -> Self 
    where 
        F: Fn(&FxGraph, &SubgraphMatch) -> bool + 'static 
    {
        self.conditions.push(Box::new(condition));
        self
    }
    
    fn should_apply_rewrite(&self, graph: &FxGraph, match_: &SubgraphMatch) -> bool {
        self.conditions.iter().all(|condition| condition(graph, match_))
    }
}

// Usage example
fn create_conditional_fusion() -> ConditionalRewriter {
    ConditionalRewriter::new()
        .add_condition(|graph, match_| {
            // Only fuse if input tensors are large enough
            let input_size = estimate_tensor_size(graph, &match_.inputs[0]);
            input_size > 1000
        })
        .add_condition(|graph, match_| {
            // Only fuse if not in training mode
            !is_training_mode(graph)
        })
}
```

## Advanced Transformations

### Dynamic Shape Handling

```rust
use torsh_fx::{DynamicShapeInferenceContext, ShapeConstraint};

struct DynamicShapePass {
    context: DynamicShapeInferenceContext,
}

impl DynamicShapePass {
    fn new() -> Self {
        Self {
            context: DynamicShapeInferenceContext::new(),
        }
    }
}

impl Pass for DynamicShapePass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Add shape constraints
        self.context.add_constraint(
            ShapeConstraint::Divisible("batch_size".to_string(), 8)
        )?;
        
        // Infer shapes for all nodes
        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, args) = node {
                let inferred_shape = self.context.infer_shape(op_name, args)?;
                // Update node with shape information
            }
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "dynamic_shape_inference"
    }
}
```

### Hardware-Specific Optimizations

```rust
#[derive(Clone, Debug)]
enum TargetDevice {
    CPU,
    CUDA,
    TPU,
    Metal,
}

struct DeviceSpecificPass {
    target_device: TargetDevice,
}

impl DeviceSpecificPass {
    fn new(target_device: TargetDevice) -> Self {
        Self { target_device }
    }
}

impl Pass for DeviceSpecificPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        match self.target_device {
            TargetDevice::CUDA => self.apply_cuda_optimizations(graph),
            TargetDevice::TPU => self.apply_tpu_optimizations(graph),
            TargetDevice::Metal => self.apply_metal_optimizations(graph),
            TargetDevice::CPU => self.apply_cpu_optimizations(graph),
        }
    }
    
    fn name(&self) -> &str {
        "device_specific_optimization"
    }
}

impl DeviceSpecificPass {
    fn apply_cuda_optimizations(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // CUDA-specific optimizations
        // - Use Tensor Core operations when possible
        // - Optimize memory access patterns
        // - Fuse operations for better occupancy
        Ok(())
    }
    
    fn apply_tpu_optimizations(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // TPU-specific optimizations
        // - Use XLA-compatible operations
        // - Optimize for matrix multiplication units
        // - Handle dynamic shapes efficiently
        Ok(())
    }
    
    fn apply_metal_optimizations(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Metal Performance Shaders optimizations
        // - Use MPS neural network operations
        // - Optimize for Apple Silicon
        Ok(())
    }
    
    fn apply_cpu_optimizations(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // CPU-specific optimizations
        // - Vectorize operations
        // - Optimize cache usage
        // - Use SIMD instructions
        Ok(())
    }
}
```

### Multi-Pass Optimization

```rust
struct MultiPassOptimizer {
    max_iterations: usize,
    convergence_threshold: f32,
}

impl MultiPassOptimizer {
    fn new() -> Self {
        Self {
            max_iterations: 10,
            convergence_threshold: 0.01,
        }
    }
    
    fn optimize_until_convergence(&self, graph: &mut FxGraph) -> TorshResult<()> {
        let mut previous_node_count = graph.node_count();
        
        for iteration in 0..self.max_iterations {
            // Apply optimization passes
            let pass_manager = PassManager::default_optimization_passes();
            pass_manager.run(graph)?;
            
            let current_node_count = graph.node_count();
            let change_ratio = (previous_node_count as f32 - current_node_count as f32) 
                              / previous_node_count as f32;
            
            println!("Iteration {}: {} nodes (change: {:.2}%)", 
                     iteration, current_node_count, change_ratio * 100.0);
            
            // Check for convergence
            if change_ratio < self.convergence_threshold {
                println!("Converged after {} iterations", iteration + 1);
                break;
            }
            
            previous_node_count = current_node_count;
        }
        
        Ok(())
    }
}
```

## Performance Considerations

### Pass Ordering

The order of optimization passes can significantly impact both performance and correctness:

1. **Early Passes**: Graph simplification, constant folding
2. **Middle Passes**: CSE, operation fusion
3. **Late Passes**: Dead code elimination, memory optimization

### Benchmarking Transformations

```rust
use std::time::Instant;

fn benchmark_transformation<F>(name: &str, graph: &mut FxGraph, transform: F) -> TorshResult<()>
where
    F: FnOnce(&mut FxGraph) -> TorshResult<()>
{
    let start_nodes = graph.node_count();
    let start_time = Instant::now();
    
    transform(graph)?;
    
    let duration = start_time.elapsed();
    let end_nodes = graph.node_count();
    
    println!("{}: {} -> {} nodes in {:?}", 
             name, start_nodes, end_nodes, duration);
    
    Ok(())
}

// Usage
fn benchmark_optimizations(graph: &mut FxGraph) -> TorshResult<()> {
    benchmark_transformation("Fusion", graph, |g| {
        OperationFusionPass.apply(g)
    })?;
    
    benchmark_transformation("DCE", graph, |g| {
        DeadCodeEliminationPass.apply(g)
    })?;
    
    Ok(())
}
```

### Memory-Efficient Transformations

```rust
struct MemoryEfficientPass;

impl Pass for MemoryEfficientPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Process graph in chunks to avoid memory spikes
        let chunk_size = 100;
        let node_indices: Vec<_> = graph.graph.node_indices().collect();
        
        for chunk in node_indices.chunks(chunk_size) {
            self.process_chunk(graph, chunk)?;
            
            // Optional: trigger garbage collection between chunks
            // This helps with memory management for very large graphs
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "memory_efficient"
    }
}

impl MemoryEfficientPass {
    fn process_chunk(
        &self, 
        graph: &mut FxGraph, 
        chunk: &[petgraph::graph::NodeIndex]
    ) -> TorshResult<()> {
        // Process only the nodes in this chunk
        for &node_idx in chunk {
            if let Some(node) = graph.get_node(node_idx) {
                // Apply transformations to this specific node
                self.transform_node(graph, node_idx, node)?;
            }
        }
        Ok(())
    }
    
    fn transform_node(
        &self,
        graph: &mut FxGraph,
        idx: petgraph::graph::NodeIndex,
        node: &Node,
    ) -> TorshResult<()> {
        // Node-specific transformation logic
        Ok(())
    }
}
```

## Best Practices

1. **Pass Development**:
   - Keep passes focused on a single optimization
   - Ensure passes are idempotent when possible
   - Add comprehensive tests for edge cases

2. **Pattern Matching**:
   - Start with simple patterns and gradually increase complexity
   - Use constraints to avoid incorrect matches
   - Test patterns on various graph structures

3. **Performance**:
   - Profile pass execution times
   - Use appropriate data structures for graph traversal
   - Consider memory usage for large graphs

4. **Debugging**:
   - Add logging to track transformation progress
   - Use graph visualization to verify correctness
   - Implement rollback mechanisms for failed transformations

This transformation guide provides the foundation for understanding and extending ToRSh FX's optimization capabilities. For specific use cases, refer to the examples in the codebase and the FX tutorial.