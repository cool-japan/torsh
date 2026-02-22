//! Graph transformation passes

use crate::{FxGraph, Node, TorshResult};
use std::collections::HashMap;

/// Pass trait for graph transformations
pub trait Pass {
    /// Apply the pass to the graph
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()>;

    /// Get the name of this pass
    fn name(&self) -> &str;
}

/// Operation fusion pass
pub struct OperationFusionPass;

impl Pass for OperationFusionPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Simple fusion: look for relu following linear/conv operations
        let mut fusions = Vec::new();

        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, _) = node {
                if op_name == "relu" {
                    // Check if there's a linear/conv operation feeding into this relu
                    let predecessors: Vec<_> = graph
                        .graph
                        .neighbors_directed(idx, petgraph::Direction::Incoming)
                        .collect();

                    for pred_idx in predecessors {
                        if let Some(Node::Call(pred_op, _)) = graph.get_node(pred_idx) {
                            if pred_op == "linear" || pred_op == "conv2d" {
                                fusions.push((pred_idx, idx, format!("{pred_op}_relu")));
                            }
                        }
                    }
                }
            }
        }

        // Apply fusions (simplified - in practice would need to update edges properly)
        for (linear_idx, _relu_idx, fused_op) in fusions {
            if let Some(Node::Call(_, ref args)) = graph.get_node(linear_idx).cloned() {
                // Replace the linear node with a fused operation
                graph.graph[linear_idx] = Node::Call(fused_op, args.clone());
                // Would need to remove relu node and update edges in a full implementation
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "operation_fusion"
    }
}

/// Dead code elimination pass
pub struct DeadCodeEliminationPass;

impl Pass for DeadCodeEliminationPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Mark all nodes reachable from outputs
        let mut reachable = std::collections::HashSet::new();
        let mut stack = graph.outputs().to_vec();

        while let Some(node_idx) = stack.pop() {
            if reachable.insert(node_idx) {
                // Add all predecessors to the stack
                let predecessors: Vec<_> = graph
                    .graph
                    .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                    .collect();
                stack.extend(predecessors);
            }
        }

        // Collect nodes to remove (those not reachable)
        let all_nodes: Vec<_> = graph.graph.node_indices().collect();
        let to_remove: Vec<_> = all_nodes
            .into_iter()
            .filter(|&idx| !reachable.contains(&idx))
            .collect();

        // Remove unreachable nodes
        for node_idx in to_remove {
            graph.graph.remove_node(node_idx);
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "dead_code_elimination"
    }
}

/// Constant folding pass
pub struct ConstantFoldingPass;

impl Pass for ConstantFoldingPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Look for operations that can be evaluated at compile time
        // This is a simplified version - would need actual constant propagation

        let constants: HashMap<String, f32> = HashMap::new();

        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, args) = node {
                // Example: if we see "add(const1, const2)" where both are constants,
                // we could replace it with the computed result
                if op_name == "add" && args.len() == 2 {
                    // Check if both arguments are known constants
                    if constants.contains_key(&args[0]) && constants.contains_key(&args[1]) {
                        // In a full implementation, we would compute the result
                        // and replace this node with a constant
                        println!("Could fold constant addition at node {:?}", idx);
                    }
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "constant_folding"
    }
}

/// Pass manager for organizing and running passes
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    /// Create a new pass manager
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a pass to the manager
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Run all passes on the graph
    pub fn run(&self, graph: &mut FxGraph) -> TorshResult<()> {
        for pass in &self.passes {
            println!("Running pass: {}", pass.name());
            pass.apply(graph)?;
        }
        Ok(())
    }

    /// Create a default pass manager with common optimization passes
    pub fn default_optimization_passes() -> Self {
        let mut manager = Self::new();
        manager.add_pass(Box::new(GraphSimplificationPass));
        manager.add_pass(Box::new(ConstantFoldingPass));
        manager.add_pass(Box::new(CommonSubexpressionEliminationPass));
        manager.add_pass(Box::new(DeadCodeEliminationPass));
        manager.add_pass(Box::new(OperationFusionPass));
        manager.add_pass(Box::new(MemoryOptimizationPass));
        manager.add_pass(Box::new(LoopOptimizationPass));
        manager
    }

    /// Create an aggressive optimization pass manager
    pub fn aggressive_optimization_passes() -> Self {
        let mut manager = Self::new();
        // Run multiple rounds of optimization
        manager.add_pass(Box::new(GraphSimplificationPass));
        manager.add_pass(Box::new(ConstantFoldingPass));
        manager.add_pass(Box::new(CommonSubexpressionEliminationPass));
        manager.add_pass(Box::new(DeadCodeEliminationPass));
        manager.add_pass(Box::new(OperationFusionPass));
        // Second round
        manager.add_pass(Box::new(GraphSimplificationPass));
        manager.add_pass(Box::new(CommonSubexpressionEliminationPass));
        manager.add_pass(Box::new(DeadCodeEliminationPass));
        manager.add_pass(Box::new(MemoryOptimizationPass));
        manager.add_pass(Box::new(LoopOptimizationPass));
        manager
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for operation fusion
pub fn fuse_operations(graph: &mut FxGraph) -> TorshResult<()> {
    let pass = OperationFusionPass;
    pass.apply(graph)
}

/// Convenience function for dead code elimination
pub fn eliminate_dead_code(graph: &mut FxGraph) -> TorshResult<()> {
    let pass = DeadCodeEliminationPass;
    pass.apply(graph)
}

/// Convenience function for constant folding
pub fn fold_constants(graph: &mut FxGraph) -> TorshResult<()> {
    let pass = ConstantFoldingPass;
    pass.apply(graph)
}

/// Common Subexpression Elimination (CSE) pass
pub struct CommonSubexpressionEliminationPass;

impl Pass for CommonSubexpressionEliminationPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        use std::collections::HashMap;

        // Map from operation signature to node index
        let mut expression_map: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();
        let mut nodes_to_replace: Vec<(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex)> =
            Vec::new();

        // Find common subexpressions
        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, args) = node {
                // Create a signature for this operation
                let args_str = args.join(",");
                let signature = format!("{op_name}({args_str})");

                if let Some(&existing_idx) = expression_map.get(&signature) {
                    // Found a duplicate expression
                    nodes_to_replace.push((idx, existing_idx));
                } else {
                    // First occurrence of this expression
                    expression_map.insert(signature, idx);
                }
            }
        }

        // Replace duplicate nodes
        for (duplicate_idx, original_idx) in nodes_to_replace {
            // Redirect all outputs of duplicate to point to original
            let successors: Vec<_> = graph
                .graph
                .neighbors_directed(duplicate_idx, petgraph::Direction::Outgoing)
                .collect();

            for successor_idx in successors {
                // Find and remove edge from duplicate to successor
                if let Some(edge_idx) = graph.graph.find_edge(duplicate_idx, successor_idx) {
                    let edge = graph.graph[edge_idx].clone();
                    graph.graph.remove_edge(edge_idx);

                    // Add edge from original to successor
                    graph.graph.add_edge(original_idx, successor_idx, edge);
                }
            }

            // Remove the duplicate node
            graph.graph.remove_node(duplicate_idx);
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "common_subexpression_elimination"
    }
}

/// Memory optimization pass
pub struct MemoryOptimizationPass;

impl Pass for MemoryOptimizationPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Analyze tensor lifetimes and identify opportunities for in-place operations
        let mut in_place_candidates = Vec::new();

        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, args) = node {
                // Check if this operation can be done in-place
                if self.can_be_inplace(op_name) && args.len() == 1 {
                    // Find the input node
                    let predecessors: Vec<_> = graph
                        .graph
                        .neighbors_directed(idx, petgraph::Direction::Incoming)
                        .collect();

                    if predecessors.len() == 1 {
                        let input_idx = predecessors[0];

                        // Check if input has only one use (this operation)
                        let input_uses: Vec<_> = graph
                            .graph
                            .neighbors_directed(input_idx, petgraph::Direction::Outgoing)
                            .collect();

                        if input_uses.len() == 1 {
                            in_place_candidates.push((idx, op_name.clone()));
                        }
                    }
                }
            }
        }

        // Mark operations as in-place (in practice, this would modify the operation metadata)
        for (idx, op_name) in in_place_candidates {
            // Replace operation with in-place version
            if let Some(Node::Call(ref mut current_op, ref _args)) =
                graph.graph.node_weight_mut(idx)
            {
                *current_op = format!("{op_name}_inplace");
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "memory_optimization"
    }
}

impl MemoryOptimizationPass {
    /// Check if an operation can be performed in-place
    fn can_be_inplace(&self, op_name: &str) -> bool {
        matches!(op_name, "relu" | "sigmoid" | "tanh" | "add" | "mul")
    }
}

/// Loop optimization pass
pub struct LoopOptimizationPass;

impl Pass for LoopOptimizationPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Find loop nodes and optimize them
        let mut loop_optimizations = Vec::new();

        for (idx, node) in graph.nodes() {
            if let Node::Loop {
                condition: _,
                body,
                loop_vars: _,
            } = node
            {
                // Analyze loop for optimization opportunities
                if self.can_unroll_loop(body) {
                    loop_optimizations.push((idx, "unroll"));
                } else if self.can_vectorize_loop(body) {
                    loop_optimizations.push((idx, "vectorize"));
                }
            }
        }

        // Apply optimizations
        for (idx, optimization) in loop_optimizations {
            if let Some(Node::Loop { ref mut body, .. }) = graph.graph.node_weight_mut(idx) {
                match optimization {
                    "unroll" => {
                        // Mark loop for unrolling
                        body.push("unrolled".to_string());
                    }
                    "vectorize" => {
                        // Mark loop for vectorization
                        body.push("vectorized".to_string());
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "loop_optimization"
    }
}

impl LoopOptimizationPass {
    fn can_unroll_loop(&self, _body: &[String]) -> bool {
        // Simplified heuristic: small loops can be unrolled
        true // For demonstration
    }

    fn can_vectorize_loop(&self, _body: &[String]) -> bool {
        // Simplified heuristic: element-wise operations can be vectorized
        true // For demonstration
    }
}

/// Graph simplification pass
pub struct GraphSimplificationPass;

impl Pass for GraphSimplificationPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        let mut simplifications = Vec::new();

        // Find patterns that can be simplified
        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, args) = node {
                match op_name.as_str() {
                    "add" => {
                        // Check for add(x, 0) or add(0, x) patterns
                        if args.len() == 2 && (args[0] == "zero" || args[1] == "zero") {
                            simplifications.push((idx, "identity"));
                        }
                    }
                    "mul" => {
                        // Check for mul(x, 1) or mul(1, x) patterns
                        if args.len() == 2 && (args[0] == "one" || args[1] == "one") {
                            simplifications.push((idx, "identity"));
                        }
                        // Check for mul(x, 0) or mul(0, x) patterns
                        if args.len() == 2 && (args[0] == "zero" || args[1] == "zero") {
                            simplifications.push((idx, "zero"));
                        }
                    }
                    _ => {}
                }
            }
        }

        // Apply simplifications
        for (idx, simplification) in simplifications {
            match simplification {
                "identity" => {
                    // Replace with identity operation (just pass through the non-constant input)
                    if let Some(Node::Call(ref mut op_name, ref mut args)) =
                        graph.graph.node_weight_mut(idx)
                    {
                        *op_name = "identity".to_string();
                        args.retain(|arg| arg != "zero" && arg != "one");
                    }
                }
                "zero" => {
                    // Replace with constant zero
                    if let Some(node) = graph.graph.node_weight_mut(idx) {
                        *node = Node::Call("constant_zero".to_string(), vec![]);
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "graph_simplification"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::ModuleTracer;

    #[test]
    fn test_pass_manager() {
        let mut manager = PassManager::new();
        manager.add_pass(Box::new(DeadCodeEliminationPass));

        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("relu", vec!["x".to_string()]);
        tracer.add_output("node_0");
        let mut graph = tracer.finalize();

        // Should run without error
        manager.run(&mut graph).unwrap();
    }

    #[test]
    fn test_operation_fusion_pass() {
        let pass = OperationFusionPass;

        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("linear", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let mut graph = tracer.finalize();

        // Should run without error
        pass.apply(&mut graph).unwrap();
    }
}
