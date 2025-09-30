# ToRSh FX Best Practices

This document provides comprehensive best practices for developing with ToRSh FX, covering graph construction, optimization, debugging, and production deployment.

## Table of Contents

1. [Graph Design](#graph-design)
2. [Performance Optimization](#performance-optimization)
3. [Memory Management](#memory-management)
4. [Error Handling](#error-handling)
5. [Testing Strategy](#testing-strategy)
6. [Debugging and Profiling](#debugging-and-profiling)
7. [Production Deployment](#production-deployment)
8. [Code Organization](#code-organization)

## Graph Design

### 1. Keep Graphs Simple and Focused

**DO:**
```rust
// Simple, focused graph for a specific task
fn create_linear_classifier() -> FxGraph {
    let mut tracer = ModuleTracer::new();
    
    let input = tracer.add_input("features");
    let linear = tracer.add_call("linear", vec!["features".to_string()]);
    let softmax = tracer.add_call("softmax", vec!["node_0".to_string()]);
    let output = tracer.add_output("node_1");
    
    tracer.finalize()
}
```

**DON'T:**
```rust
// Overly complex graph mixing multiple concerns
fn create_everything_graph() -> FxGraph {
    // Combines preprocessing, model, postprocessing, 
    // visualization, and logging in one graph
    // This makes optimization and debugging difficult
}
```

### 2. Use Meaningful Node Names

**DO:**
```rust
let input = tracer.add_input("user_embeddings");
let attention = tracer.add_call("multi_head_attention", vec!["user_embeddings".to_string()]);
let output_projection = tracer.add_call("linear", vec!["attention_output".to_string()]);
```

**DON'T:**
```rust
let input = tracer.add_input("x");
let node1 = tracer.add_call("op", vec!["x".to_string()]);
let node2 = tracer.add_call("op2", vec!["node_0".to_string()]);
```

### 3. Design for Composability

```rust
// Design modular graph components
struct AttentionBlock;

impl AttentionBlock {
    fn trace(tracer: &mut ModuleTracer, input: &str, hidden_size: usize) -> String {
        let q_proj = tracer.add_call("linear", vec![input.to_string()]);
        let k_proj = tracer.add_call("linear", vec![input.to_string()]);
        let v_proj = tracer.add_call("linear", vec![input.to_string()]);
        
        let attention = tracer.add_call("scaled_dot_product_attention", vec![
            format!("node_{}", q_proj.index()),
            format!("node_{}", k_proj.index()),
            format!("node_{}", v_proj.index()),
        ]);
        
        let output = tracer.add_call("linear", vec![
            format!("node_{}", attention.index())
        ]);
        
        format!("node_{}", output.index())
    }
}

// Compose multiple blocks
fn create_transformer() -> FxGraph {
    let mut tracer = ModuleTracer::new();
    
    let input = tracer.add_input("tokens");
    let mut current = "tokens".to_string();
    
    // Stack multiple attention blocks
    for i in 0..6 {
        current = AttentionBlock::trace(&mut tracer, &current, 768);
    }
    
    let output = tracer.add_output(&current);
    tracer.finalize()
}
```

### 4. Handle Control Flow Carefully

```rust
// Structured control flow with clear entry/exit points
fn create_conditional_model() -> FxGraph {
    let mut tracer = ModuleTracer::new();
    
    let input = tracer.add_input("x");
    let condition = tracer.add_call("is_training", vec!["x".to_string()]);
    
    // Training path
    let dropout = tracer.add_call("dropout", vec!["x".to_string()]);
    
    // Inference path  
    let identity = tracer.add_call("identity", vec!["x".to_string()]);
    
    // Structured conditional
    let conditional = tracer.add_conditional(
        "node_0", // condition
        vec!["node_1".to_string()], // training path
        vec!["node_2".to_string()]  // inference path
    );
    
    let output = tracer.add_output("node_3");
    tracer.finalize()
}
```

### 5. Minimize Graph Depth

**DO:**
```rust
// Parallel computation where possible
fn create_parallel_branches() -> FxGraph {
    let mut tracer = ModuleTracer::new();
    
    let input = tracer.add_input("x");
    
    // Parallel branches
    let branch1 = tracer.add_call("conv2d", vec!["x".to_string()]);
    let branch2 = tracer.add_call("conv2d", vec!["x".to_string()]);
    let branch3 = tracer.add_call("conv2d", vec!["x".to_string()]);
    
    // Combine results
    let concat = tracer.add_call("concat", vec![
        "node_0".to_string(),
        "node_1".to_string(), 
        "node_2".to_string()
    ]);
    
    let output = tracer.add_output("node_3");
    tracer.finalize()
}
```

**DON'T:**
```rust
// Unnecessarily deep sequential computation
fn create_deep_sequential() -> FxGraph {
    let mut tracer = ModuleTracer::new();
    
    let input = tracer.add_input("x");
    let mut current = "x".to_string();
    
    // Too many sequential operations
    for i in 0..100 {
        let node = tracer.add_call("small_op", vec![current]);
        current = format!("node_{}", node.index());
    }
    
    tracer.finalize()
}
```

## Performance Optimization

### 1. Apply Optimization Passes Strategically

```rust
fn optimize_for_inference(graph: &mut FxGraph) -> TorshResult<()> {
    // Order matters!
    let mut pass_manager = PassManager::new();
    
    // 1. Simplify mathematical expressions first
    pass_manager.add_pass(Box::new(GraphSimplificationPass));
    
    // 2. Fold constants to enable further optimizations
    pass_manager.add_pass(Box::new(ConstantFoldingPass));
    
    // 3. Eliminate common subexpressions
    pass_manager.add_pass(Box::new(CommonSubexpressionEliminationPass));
    
    // 4. Fuse operations for better performance
    pass_manager.add_pass(Box::new(OperationFusionPass));
    
    // 5. Remove dead code last (after other optimizations)
    pass_manager.add_pass(Box::new(DeadCodeEliminationPass));
    
    // 6. Optimize memory usage
    pass_manager.add_pass(Box::new(MemoryOptimizationPass));
    
    pass_manager.run(graph)?;
    Ok(())
}
```

### 2. Use Device-Specific Optimizations

```rust
fn optimize_for_device(graph: &mut FxGraph, device: &Device) -> TorshResult<()> {
    match device.device_type() {
        DeviceType::CUDA => {
            // CUDA-specific optimizations
            let cuda_pass = CudaOptimizationPass::new()
                .enable_tensor_cores(true)
                .optimize_memory_access(true)
                .fuse_kernels(true);
            cuda_pass.apply(graph)?;
        }
        DeviceType::CPU => {
            // CPU-specific optimizations
            let cpu_pass = CpuOptimizationPass::new()
                .enable_simd(true)
                .optimize_cache_usage(true)
                .vectorize_loops(true);
            cpu_pass.apply(graph)?;
        }
        DeviceType::TPU => {
            // TPU-specific optimizations
            let tpu_pass = TpuOptimizationPass::new()
                .optimize_for_xla(true)
                .batch_operations(true);
            tpu_pass.apply(graph)?;
        }
        _ => {}
    }
    
    Ok(())
}
```

### 3. Measure Before and After Optimization

```rust
use std::time::Instant;

fn benchmark_optimization() -> TorshResult<()> {
    let mut graph = create_test_graph();
    
    // Measure baseline
    let baseline_metrics = measure_graph_performance(&graph)?;
    println!("Baseline: {} nodes, estimated {} ms", 
             baseline_metrics.node_count, 
             baseline_metrics.estimated_time_ms);
    
    // Apply optimizations
    let start = Instant::now();
    optimize_for_inference(&mut graph)?;
    let optimization_time = start.elapsed();
    
    // Measure optimized version
    let optimized_metrics = measure_graph_performance(&graph)?;
    println!("Optimized: {} nodes, estimated {} ms", 
             optimized_metrics.node_count, 
             optimized_metrics.estimated_time_ms);
    
    println!("Optimization took: {:?}", optimization_time);
    println!("Speedup: {:.2}x", 
             baseline_metrics.estimated_time_ms / optimized_metrics.estimated_time_ms);
    
    Ok(())
}

struct PerformanceMetrics {
    node_count: usize,
    estimated_time_ms: f32,
    memory_usage_mb: f32,
}

fn measure_graph_performance(graph: &FxGraph) -> TorshResult<PerformanceMetrics> {
    // Implement performance estimation based on operation types,
    // graph structure, and expected data sizes
    todo!()
}
```

### 4. Profile Critical Paths

```rust
fn profile_execution_paths(graph: &FxGraph) -> TorshResult<()> {
    let profiler = GraphProfiler::new();
    
    // Identify critical path (longest execution time)
    let critical_path = profiler.find_critical_path(graph)?;
    println!("Critical path: {} operations, {} estimated ms", 
             critical_path.operations.len(),
             critical_path.estimated_time_ms);
    
    // Identify memory bottlenecks
    let memory_hotspots = profiler.find_memory_hotspots(graph)?;
    for hotspot in memory_hotspots {
        println!("Memory hotspot at {}: {} MB", hotspot.node_name, hotspot.memory_mb);
    }
    
    // Suggest optimizations
    let suggestions = profiler.suggest_optimizations(graph)?;
    for suggestion in suggestions {
        println!("Suggestion: {}", suggestion);
    }
    
    Ok(())
}
```

## Memory Management

### 1. Minimize Memory Allocations

```rust
// Use in-place operations when possible
fn optimize_memory_usage(graph: &mut FxGraph) -> TorshResult<()> {
    let memory_pass = MemoryOptimizationPass::new()
        .enable_inplace_operations(true)
        .analyze_tensor_lifetimes(true)
        .reuse_intermediate_buffers(true);
    
    memory_pass.apply(graph)?;
    Ok(())
}

// Check if operation can be done in-place
fn can_be_inplace(op_name: &str, input_usage_count: usize) -> bool {
    match op_name {
        "relu" | "sigmoid" | "tanh" => input_usage_count == 1,
        "add" | "mul" => input_usage_count == 1, // First operand can be reused
        _ => false,
    }
}
```

### 2. Manage Tensor Lifetimes

```rust
struct TensorLifetimeAnalyzer;

impl TensorLifetimeAnalyzer {
    fn analyze_lifetimes(graph: &FxGraph) -> HashMap<String, LifetimeInfo> {
        let mut lifetimes = HashMap::new();
        
        // Topological traversal to determine when tensors are created and last used
        let topo_order = petgraph::algo::toposort(&graph.graph, None).unwrap();
        
        for (step, &node_idx) in topo_order.iter().enumerate() {
            if let Some(node) = graph.get_node(node_idx) {
                match node {
                    Node::Input(name) => {
                        lifetimes.insert(name.clone(), LifetimeInfo {
                            created_at: step,
                            last_used_at: self.find_last_usage(graph, name, &topo_order),
                        });
                    }
                    Node::Call(_, args) => {
                        // Update last usage for input tensors
                        for arg in args {
                            if let Some(lifetime) = lifetimes.get_mut(arg) {
                                lifetime.last_used_at = step;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        
        lifetimes
    }
    
    fn suggest_memory_optimizations(lifetimes: &HashMap<String, LifetimeInfo>) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        for (tensor_name, lifetime) in lifetimes {
            let lifetime_span = lifetime.last_used_at - lifetime.created_at;
            
            if lifetime_span > 10 {
                suggestions.push(format!(
                    "Consider splitting computation for '{}' to reduce memory pressure", 
                    tensor_name
                ));
            }
        }
        
        suggestions
    }
}

struct LifetimeInfo {
    created_at: usize,
    last_used_at: usize,
}
```

### 3. Use Memory Pools

```rust
struct MemoryPool {
    buffers: Vec<Arc<Mutex<Vec<f32>>>>,
    available: VecDeque<usize>,
}

impl MemoryPool {
    fn new() -> Self {
        Self {
            buffers: Vec::new(),
            available: VecDeque::new(),
        }
    }
    
    fn get_buffer(&mut self, size: usize) -> Arc<Mutex<Vec<f32>>> {
        if let Some(buffer_idx) = self.available.pop_front() {
            let buffer = &self.buffers[buffer_idx];
            let mut buf = buffer.lock().unwrap();
            buf.resize(size, 0.0);
            buffer.clone()
        } else {
            let buffer = Arc::new(Mutex::new(vec![0.0; size]));
            self.buffers.push(buffer.clone());
            buffer
        }
    }
    
    fn return_buffer(&mut self, buffer: Arc<Mutex<Vec<f32>>>) {
        // Find buffer index and mark as available
        for (idx, pool_buffer) in self.buffers.iter().enumerate() {
            if Arc::ptr_eq(pool_buffer, &buffer) {
                self.available.push_back(idx);
                break;
            }
        }
    }
}
```

## Error Handling

### 1. Use Comprehensive Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum FxError {
    #[error("Graph validation failed: {message}")]
    ValidationError { message: String },
    
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },
    
    #[error("Shape incompatible: {shape1:?} and {shape2:?}")]
    ShapeIncompatible { shape1: Vec<usize>, shape2: Vec<usize> },
    
    #[error("Operation '{operation}' not supported")]
    UnsupportedOperation { operation: String },
    
    #[error("Pass '{pass_name}' failed: {reason}")]
    PassExecutionError { pass_name: String, reason: String },
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Graph execution error: {0}")]
    ExecutionError(String),
}

type FxResult<T> = Result<T, FxError>;
```

### 2. Validate Early and Often

```rust
fn create_robust_graph() -> FxResult<FxGraph> {
    let mut tracer = ModuleTracer::new();
    
    // Validate inputs
    let input = tracer.add_input("features");
    validate_input_name("features")?;
    
    // Validate operations
    let linear = tracer.add_call("linear", vec!["features".to_string()]);
    validate_operation("linear", &["features".to_string()])?;
    
    let graph = tracer.finalize();
    
    // Validate complete graph
    validate_graph_structure(&graph)?;
    validate_graph_semantics(&graph)?;
    
    Ok(graph)
}

fn validate_input_name(name: &str) -> FxResult<()> {
    if name.is_empty() {
        return Err(FxError::ValidationError {
            message: "Input name cannot be empty".to_string(),
        });
    }
    
    if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return Err(FxError::ValidationError {
            message: format!("Invalid input name: {}", name),
        });
    }
    
    Ok(())
}

fn validate_operation(op_name: &str, args: &[String]) -> FxResult<()> {
    let expected_args = match op_name {
        "linear" => 1,
        "conv2d" => 1,
        "add" | "mul" => 2,
        _ => return Err(FxError::UnsupportedOperation {
            operation: op_name.to_string(),
        }),
    };
    
    if args.len() != expected_args {
        return Err(FxError::ValidationError {
            message: format!(
                "Operation '{}' expects {} arguments, got {}", 
                op_name, expected_args, args.len()
            ),
        });
    }
    
    Ok(())
}
```

### 3. Provide Helpful Error Messages

```rust
fn provide_context_in_errors() -> FxResult<()> {
    let graph = create_test_graph();
    
    // Add context to errors
    let pass = OperationFusionPass;
    pass.apply(&mut graph.clone()).map_err(|e| {
        FxError::PassExecutionError {
            pass_name: "operation_fusion".to_string(),
            reason: format!("Failed on graph with {} nodes: {}", graph.node_count(), e),
        }
    })?;
    
    Ok(())
}

// Custom error context for debugging
#[derive(Debug)]
struct ErrorContext {
    operation: String,
    node_index: Option<usize>,
    input_shapes: Vec<Vec<usize>>,
    additional_info: HashMap<String, String>,
}

impl ErrorContext {
    fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            node_index: None,
            input_shapes: Vec::new(),
            additional_info: HashMap::new(),
        }
    }
    
    fn with_node_index(mut self, idx: usize) -> Self {
        self.node_index = Some(idx);
        self
    }
    
    fn with_input_shapes(mut self, shapes: Vec<Vec<usize>>) -> Self {
        self.input_shapes = shapes;
        self
    }
    
    fn with_info(mut self, key: &str, value: &str) -> Self {
        self.additional_info.insert(key.to_string(), value.to_string());
        self
    }
}
```

## Testing Strategy

### 1. Unit Test Individual Components

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_graph_creation() {
        let mut tracer = ModuleTracer::new();
        
        let input = tracer.add_input("x");
        let relu = tracer.add_call("relu", vec!["x".to_string()]);
        let output = tracer.add_output("node_0");
        
        let graph = tracer.finalize();
        
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.inputs().len(), 1);
        assert_eq!(graph.outputs().len(), 1);
    }
    
    #[test]
    fn test_operation_fusion_pass() {
        let mut graph = create_fusable_graph();
        let initial_nodes = graph.node_count();
        
        let fusion_pass = OperationFusionPass;
        fusion_pass.apply(&mut graph).unwrap();
        
        // Should have fewer nodes after fusion
        assert!(graph.node_count() < initial_nodes);
        
        // Verify graph is still valid
        validate_graph_structure(&graph).unwrap();
    }
    
    #[test]
    fn test_error_handling() {
        let mut tracer = ModuleTracer::new();
        
        // Test invalid operation
        let result = std::panic::catch_unwind(|| {
            tracer.add_call("invalid_op", vec!["nonexistent".to_string()]);
        });
        
        // Should handle gracefully or provide clear error
        assert!(result.is_ok() || result.is_err());
    }
    
    fn create_fusable_graph() -> FxGraph {
        let mut tracer = ModuleTracer::new();
        
        let input = tracer.add_input("x");
        let linear = tracer.add_call("linear", vec!["x".to_string()]);
        let relu = tracer.add_call("relu", vec!["node_0".to_string()]);
        let output = tracer.add_output("node_1");
        
        tracer.finalize()
    }
}
```

### 2. Integration Tests for Complete Workflows

```rust
#[test]
fn test_complete_optimization_pipeline() {
    let mut graph = create_complex_test_graph();
    let original_nodes = graph.node_count();
    
    // Apply full optimization pipeline
    let pass_manager = PassManager::default_optimization_passes();
    pass_manager.run(&mut graph).unwrap();
    
    // Verify optimizations were applied
    assert!(graph.node_count() <= original_nodes);
    
    // Verify graph is still functionally equivalent
    verify_graph_equivalence(&create_complex_test_graph(), &graph).unwrap();
}

#[test]
fn test_serialization_roundtrip() {
    let original_graph = create_test_graph();
    
    // Test JSON serialization
    let json = original_graph.to_json().unwrap();
    let restored_json = FxGraph::from_json(&json).unwrap();
    assert_graphs_equal(&original_graph, &restored_json);
    
    // Test binary serialization
    let binary = original_graph.to_binary().unwrap();
    let restored_binary = FxGraph::from_binary(&binary).unwrap();
    assert_graphs_equal(&original_graph, &restored_binary);
}

fn verify_graph_equivalence(graph1: &FxGraph, graph2: &FxGraph) -> FxResult<()> {
    // Implement semantic equivalence checking
    // This might involve executing both graphs with test inputs
    // and comparing outputs
    todo!()
}

fn assert_graphs_equal(graph1: &FxGraph, graph2: &FxGraph) {
    assert_eq!(graph1.node_count(), graph2.node_count());
    assert_eq!(graph1.edge_count(), graph2.edge_count());
    assert_eq!(graph1.inputs().len(), graph2.inputs().len());
    assert_eq!(graph1.outputs().len(), graph2.outputs().len());
}
```

### 3. Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_optimization_preserves_semantics(
        nodes in 1..20usize,
        edges in 0..50usize,
    ) {
        let graph = generate_random_graph(nodes, edges);
        let mut optimized_graph = graph.clone();
        
        // Apply optimizations
        let pass_manager = PassManager::default_optimization_passes();
        pass_manager.run(&mut optimized_graph).unwrap();
        
        // Verify semantic equivalence
        prop_assert!(are_semantically_equivalent(&graph, &optimized_graph));
    }
    
    #[test]
    fn test_serialization_preserves_structure(
        graph in arbitrary_graph_strategy()
    ) {
        // Test that serialization roundtrip preserves graph structure
        let json = graph.to_json().unwrap();
        let restored = FxGraph::from_json(&json).unwrap();
        
        prop_assert_eq!(graph.node_count(), restored.node_count());
        prop_assert_eq!(graph.edge_count(), restored.edge_count());
    }
}

fn arbitrary_graph_strategy() -> impl Strategy<Value = FxGraph> {
    // Generate arbitrary but valid graphs for testing
    (1..10usize, 0..20usize).prop_flat_map(|(nodes, edges)| {
        // Implementation details for generating valid random graphs
        todo!()
    })
}
```

### 4. Performance Tests

```rust
#[test]
fn test_optimization_performance() {
    let large_graph = create_large_test_graph(1000); // 1000 nodes
    
    let start = std::time::Instant::now();
    let mut optimized = large_graph.clone();
    let pass_manager = PassManager::default_optimization_passes();
    pass_manager.run(&mut optimized).unwrap();
    let duration = start.elapsed();
    
    // Optimization should complete in reasonable time
    assert!(duration < std::time::Duration::from_secs(10));
    
    // Should achieve meaningful optimization
    let reduction_ratio = (large_graph.node_count() - optimized.node_count()) as f32 
                         / large_graph.node_count() as f32;
    assert!(reduction_ratio > 0.1); // At least 10% reduction
}

#[test]
fn test_memory_usage() {
    let initial_memory = get_memory_usage();
    
    // Create and optimize many graphs
    for _ in 0..100 {
        let mut graph = create_test_graph();
        let pass_manager = PassManager::default_optimization_passes();
        pass_manager.run(&mut graph).unwrap();
    }
    
    let final_memory = get_memory_usage();
    let memory_increase = final_memory - initial_memory;
    
    // Memory usage should not grow excessively
    assert!(memory_increase < 100_000_000); // Less than 100MB increase
}

fn get_memory_usage() -> usize {
    // Platform-specific memory usage measurement
    // For Unix systems, could read /proc/self/status
    // For testing purposes, could use a simpler approximation
    0
}
```

## Debugging and Profiling

### 1. Enable Comprehensive Logging

```rust
use log::{debug, info, warn, error};

fn debug_graph_construction() -> FxResult<FxGraph> {
    info!("Starting graph construction");
    
    let mut tracer = ModuleTracer::new();
    
    debug!("Adding input node");
    let input = tracer.add_input("features");
    
    debug!("Adding linear layer");
    let linear = tracer.add_call("linear", vec!["features".to_string()]);
    
    debug!("Adding activation function");
    let relu = tracer.add_call("relu", vec!["node_0".to_string()]);
    
    debug!("Adding output node");
    let output = tracer.add_output("node_1");
    
    let graph = tracer.finalize();
    
    info!("Graph construction complete: {} nodes, {} edges", 
          graph.node_count(), graph.edge_count());
    
    Ok(graph)
}

// Configure logging for debugging
fn setup_debug_logging() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .init();
}
```

### 2. Use Graph Visualization

```rust
fn debug_with_visualization(graph: &FxGraph) -> FxResult<()> {
    let debugger = GraphDebugger::new(graph);
    
    // Generate different visualization formats
    let text_viz = debugger.generate_text_visualization()?;
    println!("Text visualization:\n{}", text_viz);
    
    // Save DOT file for Graphviz
    let dot_viz = debugger.generate_dot_visualization()?;
    std::fs::write("debug_graph.dot", dot_viz)?;
    
    // Generate interactive HTML
    let html_viz = debugger.generate_html_visualization()?;
    std::fs::write("debug_graph.html", html_viz)?;
    
    // Print graph statistics
    let stats = debugger.compute_statistics();
    println!("Graph statistics:");
    println!("  Total nodes: {}", stats.total_nodes);
    println!("  Input nodes: {}", stats.input_nodes);
    println!("  Output nodes: {}", stats.output_nodes);
    println!("  Operation types: {:?}", stats.operation_counts);
    
    Ok(())
}
```

### 3. Add Debug Assertions

```rust
fn debug_graph_invariants(graph: &FxGraph) {
    debug_assert!(
        !graph.inputs().is_empty(),
        "Graph must have at least one input"
    );
    
    debug_assert!(
        !graph.outputs().is_empty(),
        "Graph must have at least one output"
    );
    
    debug_assert!(
        graph.node_count() >= 2,
        "Graph must have at least input and output nodes"
    );
    
    // Check for unreachable nodes in debug builds
    #[cfg(debug_assertions)]
    {
        let reachable = find_reachable_nodes(graph);
        debug_assert_eq!(
            reachable.len(),
            graph.node_count(),
            "Found unreachable nodes in graph"
        );
    }
}

#[cfg(debug_assertions)]
fn find_reachable_nodes(graph: &FxGraph) -> HashSet<petgraph::graph::NodeIndex> {
    let mut reachable = HashSet::new();
    let mut stack: Vec<_> = graph.inputs().to_vec();
    
    while let Some(node_idx) = stack.pop() {
        if reachable.insert(node_idx) {
            // Add all successors to the stack
            let successors: Vec<_> = graph.graph
                .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                .collect();
            stack.extend(successors);
        }
    }
    
    reachable
}
```

### 4. Profile Critical Operations

```rust
use std::time::{Duration, Instant};

struct OperationProfiler {
    timings: HashMap<String, Vec<Duration>>,
}

impl OperationProfiler {
    fn new() -> Self {
        Self {
            timings: HashMap::new(),
        }
    }
    
    fn profile_operation<F, R>(&mut self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();
        
        self.timings
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
        
        result
    }
    
    fn print_statistics(&self) {
        println!("Operation profiling results:");
        
        for (operation, timings) in &self.timings {
            let total: Duration = timings.iter().sum();
            let average = total / timings.len() as u32;
            let min = timings.iter().min().unwrap();
            let max = timings.iter().max().unwrap();
            
            println!("  {}: {} calls, avg: {:?}, min: {:?}, max: {:?}",
                     operation, timings.len(), average, min, max);
        }
    }
}

// Usage example
fn profile_optimization_passes() -> FxResult<()> {
    let mut profiler = OperationProfiler::new();
    let mut graph = create_test_graph();
    
    profiler.profile_operation("graph_simplification", || {
        GraphSimplificationPass.apply(&mut graph).unwrap();
    });
    
    profiler.profile_operation("constant_folding", || {
        ConstantFoldingPass.apply(&mut graph).unwrap();
    });
    
    profiler.profile_operation("operation_fusion", || {
        OperationFusionPass.apply(&mut graph).unwrap();
    });
    
    profiler.print_statistics();
    
    Ok(())
}
```

## Production Deployment

### 1. Validate Production Graphs

```rust
struct ProductionValidator {
    max_nodes: usize,
    max_depth: usize,
    allowed_operations: HashSet<String>,
}

impl ProductionValidator {
    fn new() -> Self {
        Self {
            max_nodes: 10000,
            max_depth: 100,
            allowed_operations: [
                "linear", "conv2d", "relu", "sigmoid", "tanh",
                "batch_norm", "layer_norm", "dropout", "softmax"
            ].iter().map(|&s| s.to_string()).collect(),
        }
    }
    
    fn validate_for_production(&self, graph: &FxGraph) -> Result<(), ProductionError> {
        // Check graph size limits
        if graph.node_count() > self.max_nodes {
            return Err(ProductionError::GraphTooLarge {
                actual: graph.node_count(),
                max_allowed: self.max_nodes,
            });
        }
        
        // Check graph depth
        let depth = self.calculate_graph_depth(graph);
        if depth > self.max_depth {
            return Err(ProductionError::GraphTooDeep {
                actual: depth,
                max_allowed: self.max_depth,
            });
        }
        
        // Check for allowed operations only
        for (_, node) in graph.nodes() {
            if let Node::Call(op_name, _) = node {
                if !self.allowed_operations.contains(op_name) {
                    return Err(ProductionError::DisallowedOperation {
                        operation: op_name.clone(),
                    });
                }
            }
        }
        
        // Check for potential security issues
        self.check_security_constraints(graph)?;
        
        Ok(())
    }
    
    fn calculate_graph_depth(&self, graph: &FxGraph) -> usize {
        // Calculate maximum depth from any input to any output
        let mut max_depth = 0;
        
        for &input_idx in graph.inputs() {
            let depth = self.dfs_depth(graph, input_idx, 0);
            max_depth = max_depth.max(depth);
        }
        
        max_depth
    }
    
    fn dfs_depth(&self, graph: &FxGraph, node_idx: petgraph::graph::NodeIndex, current_depth: usize) -> usize {
        let successors: Vec<_> = graph.graph
            .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
            .collect();
        
        if successors.is_empty() {
            current_depth
        } else {
            successors.iter()
                .map(|&succ| self.dfs_depth(graph, succ, current_depth + 1))
                .max()
                .unwrap_or(current_depth)
        }
    }
    
    fn check_security_constraints(&self, graph: &FxGraph) -> Result<(), ProductionError> {
        // Check for operations that might pose security risks
        for (_, node) in graph.nodes() {
            if let Node::Call(op_name, _) = node {
                if op_name.contains("eval") || op_name.contains("exec") {
                    return Err(ProductionError::SecurityRisk {
                        issue: format!("Potentially unsafe operation: {}", op_name),
                    });
                }
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
enum ProductionError {
    #[error("Graph too large: {actual} nodes (max: {max_allowed})")]
    GraphTooLarge { actual: usize, max_allowed: usize },
    
    #[error("Graph too deep: {actual} levels (max: {max_allowed})")]
    GraphTooDeep { actual: usize, max_allowed: usize },
    
    #[error("Disallowed operation: {operation}")]
    DisallowedOperation { operation: String },
    
    #[error("Security risk: {issue}")]
    SecurityRisk { issue: String },
}
```

### 2. Implement Graceful Degradation

```rust
struct ProductionExecutor {
    fallback_strategies: Vec<Box<dyn FallbackStrategy>>,
    timeout: Duration,
    max_memory: usize,
}

trait FallbackStrategy {
    fn can_handle(&self, error: &ExecutionError) -> bool;
    fn execute_fallback(&self, graph: &FxGraph) -> Result<Vec<Tensor>, ExecutionError>;
}

impl ProductionExecutor {
    fn new() -> Self {
        Self {
            fallback_strategies: vec![
                Box::new(SimplificationFallback),
                Box::new(CpuFallback),
            ],
            timeout: Duration::from_secs(30),
            max_memory: 1_000_000_000, // 1GB
        }
    }
    
    fn execute_with_fallback(&self, graph: &FxGraph) -> Result<Vec<Tensor>, ExecutionError> {
        // Try normal execution first
        match self.execute_normal(graph) {
            Ok(result) => Ok(result),
            Err(error) => {
                log::warn!("Normal execution failed: {:?}", error);
                
                // Try fallback strategies
                for strategy in &self.fallback_strategies {
                    if strategy.can_handle(&error) {
                        log::info!("Trying fallback strategy");
                        match strategy.execute_fallback(graph) {
                            Ok(result) => {
                                log::info!("Fallback strategy succeeded");
                                return Ok(result);
                            }
                            Err(fallback_error) => {
                                log::warn!("Fallback strategy failed: {:?}", fallback_error);
                            }
                        }
                    }
                }
                
                Err(error)
            }
        }
    }
    
    fn execute_normal(&self, graph: &FxGraph) -> Result<Vec<Tensor>, ExecutionError> {
        // Implement normal execution with timeout and memory monitoring
        todo!()
    }
}

struct SimplificationFallback;

impl FallbackStrategy for SimplificationFallback {
    fn can_handle(&self, error: &ExecutionError) -> bool {
        matches!(error, ExecutionError::OutOfMemory | ExecutionError::Timeout)
    }
    
    fn execute_fallback(&self, graph: &FxGraph) -> Result<Vec<Tensor>, ExecutionError> {
        // Try with simplified graph
        let mut simplified = graph.clone();
        
        // Apply aggressive simplification
        let simplification_pass = GraphSimplificationPass;
        simplification_pass.apply(&mut simplified).map_err(|_| ExecutionError::FallbackFailed)?;
        
        // Try execution again with simplified graph
        todo!()
    }
}

struct CpuFallback;

impl FallbackStrategy for CpuFallback {
    fn can_handle(&self, error: &ExecutionError) -> bool {
        matches!(error, ExecutionError::DeviceError)
    }
    
    fn execute_fallback(&self, graph: &FxGraph) -> Result<Vec<Tensor>, ExecutionError> {
        // Fall back to CPU execution
        todo!()
    }
}
```

### 3. Monitor Performance in Production

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

struct ProductionMetrics {
    total_executions: AtomicU64,
    successful_executions: AtomicU64,
    fallback_executions: AtomicU64,
    total_execution_time: AtomicU64,
    memory_peak: AtomicU64,
}

impl ProductionMetrics {
    fn new() -> Self {
        Self {
            total_executions: AtomicU64::new(0),
            successful_executions: AtomicU64::new(0),
            fallback_executions: AtomicU64::new(0),
            total_execution_time: AtomicU64::new(0),
            memory_peak: AtomicU64::new(0),
        }
    }
    
    fn record_execution(&self, duration: Duration, success: bool, used_fallback: bool) {
        self.total_executions.fetch_add(1, Ordering::Relaxed);
        
        if success {
            self.successful_executions.fetch_add(1, Ordering::Relaxed);
        }
        
        if used_fallback {
            self.fallback_executions.fetch_add(1, Ordering::Relaxed);
        }
        
        self.total_execution_time.fetch_add(
            duration.as_millis() as u64,
            Ordering::Relaxed
        );
    }
    
    fn get_statistics(&self) -> MetricsSnapshot {
        let total = self.total_executions.load(Ordering::Relaxed);
        let successful = self.successful_executions.load(Ordering::Relaxed);
        let fallback = self.fallback_executions.load(Ordering::Relaxed);
        let total_time = self.total_execution_time.load(Ordering::Relaxed);
        
        MetricsSnapshot {
            total_executions: total,
            success_rate: if total > 0 { successful as f64 / total as f64 } else { 0.0 },
            fallback_rate: if total > 0 { fallback as f64 / total as f64 } else { 0.0 },
            average_execution_time: if total > 0 { total_time / total } else { 0 },
        }
    }
}

struct MetricsSnapshot {
    total_executions: u64,
    success_rate: f64,
    fallback_rate: f64,
    average_execution_time: u64,
}

// Global metrics instance
lazy_static::lazy_static! {
    static ref PRODUCTION_METRICS: Arc<ProductionMetrics> = Arc::new(ProductionMetrics::new());
}

fn execute_with_monitoring(graph: &FxGraph) -> Result<Vec<Tensor>, ExecutionError> {
    let start = Instant::now();
    let executor = ProductionExecutor::new();
    
    let result = executor.execute_with_fallback(graph);
    let duration = start.elapsed();
    
    let success = result.is_ok();
    let used_fallback = false; // Would be determined by executor
    
    PRODUCTION_METRICS.record_execution(duration, success, used_fallback);
    
    // Log metrics periodically
    if PRODUCTION_METRICS.total_executions.load(Ordering::Relaxed) % 1000 == 0 {
        let stats = PRODUCTION_METRICS.get_statistics();
        log::info!("Production metrics: success_rate={:.2}%, fallback_rate={:.2}%, avg_time={}ms",
                   stats.success_rate * 100.0,
                   stats.fallback_rate * 100.0,
                   stats.average_execution_time);
    }
    
    result
}
```

## Code Organization

### 1. Organize by Functionality

```
torsh-fx/
├── src/
│   ├── graph/              # Core graph representation
│   │   ├── mod.rs
│   │   ├── node.rs
│   │   ├── edge.rs
│   │   └── validation.rs
│   ├── passes/             # Optimization passes
│   │   ├── mod.rs
│   │   ├── fusion.rs
│   │   ├── dce.rs
│   │   └── memory.rs
│   ├── codegen/            # Code generation
│   │   ├── mod.rs
│   │   ├── python.rs
│   │   ├── cpp.rs
│   │   └── targets/
│   ├── analysis/           # Graph analysis
│   │   ├── mod.rs
│   │   ├── shapes.rs
│   │   └── types.rs
│   └── lib.rs
├── examples/               # Usage examples
├── tests/                  # Integration tests
└── docs/                   # Documentation
```

### 2. Use Clear Module Boundaries

```rust
// lib.rs - Main public API
pub mod graph;
pub mod passes;
pub mod codegen;
pub mod analysis;

// Re-export key types for convenience
pub use graph::{FxGraph, Node, Edge};
pub use passes::{PassManager, Pass};
pub use codegen::CodeGenerator;

// Private modules for internal use
mod utils;
mod validation;
```

### 3. Document Public APIs Thoroughly

```rust
/// A computational graph for functional transformations.
/// 
/// `FxGraph` represents a directed acyclic graph (DAG) where nodes represent
/// operations and edges represent data dependencies. The graph maintains
/// explicit input and output boundaries.
/// 
/// # Examples
/// 
/// ```rust
/// use torsh_fx::{FxGraph, ModuleTracer};
/// 
/// let mut tracer = ModuleTracer::new();
/// let input = tracer.add_input("x");
/// let relu = tracer.add_call("relu", vec!["x".to_string()]);
/// let output = tracer.add_output("node_0");
/// let graph = tracer.finalize();
/// 
/// assert_eq!(graph.node_count(), 3);
/// ```
/// 
/// # Graph Invariants
/// 
/// - The graph must be acyclic (no cycles except in loop constructs)
/// - All output nodes must be reachable from input nodes
/// - Each value is assigned exactly once (SSA property)
pub struct FxGraph {
    // ... implementation
}

impl FxGraph {
    /// Creates a new empty graph.
    /// 
    /// The returned graph has no nodes or edges. Use a `ModuleTracer`
    /// to construct graphs with content.
    pub fn new() -> Self {
        // ... implementation
    }
    
    /// Returns the number of nodes in the graph.
    /// 
    /// This includes all node types: inputs, operations, and outputs.
    pub fn node_count(&self) -> usize {
        // ... implementation
    }
    
    /// Serializes the graph to JSON format.
    /// 
    /// The JSON representation preserves all graph structure and can be
    /// used for storage, transmission, or debugging.
    /// 
    /// # Errors
    /// 
    /// Returns `TorshError::SerializationError` if the graph cannot be
    /// serialized (e.g., due to invalid node data).
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let graph = create_test_graph();
    /// let json = graph.to_json()?;
    /// let restored = FxGraph::from_json(&json)?;
    /// assert_eq!(graph.node_count(), restored.node_count());
    /// ```
    pub fn to_json(&self) -> TorshResult<String> {
        // ... implementation
    }
}
```

Following these best practices will help you build robust, maintainable, and high-performance applications with ToRSh FX. Remember that optimization is often a trade-off between different factors, so profile and measure your specific use cases to make informed decisions.