//! Kernel fusion optimization for JIT compilation

use crate::graph::{ComputationGraph, NodeId, Operation};
use crate::JitResult;
use indexmap::IndexSet;
use std::collections::HashSet;

/// Fusion strategy configuration
#[derive(Debug, Clone, PartialEq)]
pub enum FusionStrategy {
    /// No fusion
    None,

    /// Conservative fusion - only fuse simple patterns
    Conservative,

    /// Default fusion - balance between performance and compilation time
    Default,

    /// Aggressive fusion - maximize fusion opportunities
    Aggressive,

    /// Custom fusion with specific rules
    Custom(FusionRules),
}

/// Custom fusion rules
#[derive(Debug, Clone, PartialEq)]
pub struct FusionRules {
    /// Maximum nodes in a fusion group
    pub max_group_size: usize,

    /// Allow fusion across different devices
    pub cross_device_fusion: bool,

    /// Fusion patterns to look for
    pub patterns: Vec<FusionPattern>,
}

/// Fusion pattern definition
#[derive(Debug, Clone, PartialEq)]
pub struct FusionPattern {
    /// Pattern name
    pub name: String,

    /// Operations that can be fused
    pub ops: Vec<Operation>,

    /// Priority (higher = preferred)
    pub priority: i32,
}

/// Kernel fusion optimizer
pub struct KernelFusion {
    strategy: FusionStrategy,
}

#[allow(dead_code)]
impl KernelFusion {
    /// Create a new kernel fusion optimizer
    pub fn new(strategy: FusionStrategy) -> Self {
        Self { strategy }
    }

    /// Apply kernel fusion to the graph
    pub fn apply(&self, graph: ComputationGraph) -> JitResult<ComputationGraph> {
        match &self.strategy {
            FusionStrategy::None => Ok(graph),
            FusionStrategy::Conservative => self.apply_conservative_fusion(graph),
            FusionStrategy::Default => self.apply_default_fusion(graph),
            FusionStrategy::Aggressive => self.apply_aggressive_fusion(graph),
            FusionStrategy::Custom(rules) => self.apply_custom_fusion(graph, rules),
        }
    }

    /// Conservative fusion - only fuse element-wise operations
    fn apply_conservative_fusion(&self, graph: ComputationGraph) -> JitResult<ComputationGraph> {
        let fusion_groups = self.find_elementwise_chains(&graph)?;
        self.create_fused_graph(graph, fusion_groups)
    }

    /// Default fusion - fuse common patterns
    fn apply_default_fusion(&self, graph: ComputationGraph) -> JitResult<ComputationGraph> {
        let mut fusion_groups = Vec::new();

        // Find element-wise chains
        fusion_groups.extend(self.find_elementwise_chains(&graph)?);

        // Find conv+activation patterns
        fusion_groups.extend(self.find_conv_activation_patterns(&graph)?);

        // Find linear+activation patterns
        fusion_groups.extend(self.find_linear_activation_patterns(&graph)?);

        // Merge overlapping groups
        let merged_groups = self.merge_fusion_groups(fusion_groups);

        self.create_fused_graph(graph, merged_groups)
    }

    /// Aggressive fusion - maximize fusion opportunities
    fn apply_aggressive_fusion(&self, graph: ComputationGraph) -> JitResult<ComputationGraph> {
        let mut fusion_groups = Vec::new();

        // All patterns from default
        fusion_groups.extend(self.find_elementwise_chains(&graph)?);
        fusion_groups.extend(self.find_conv_activation_patterns(&graph)?);
        fusion_groups.extend(self.find_linear_activation_patterns(&graph)?);

        // Additional aggressive patterns
        fusion_groups.extend(self.find_reduction_chains(&graph)?);
        fusion_groups.extend(self.find_matmul_chains(&graph)?);

        // Try to grow fusion groups
        let grown_groups = self.grow_fusion_groups(&graph, fusion_groups)?;

        self.create_fused_graph(graph, grown_groups)
    }

    /// Apply custom fusion rules
    fn apply_custom_fusion(
        &self,
        graph: ComputationGraph,
        rules: &FusionRules,
    ) -> JitResult<ComputationGraph> {
        let mut fusion_groups = Vec::new();

        // Apply each pattern
        for pattern in &rules.patterns {
            fusion_groups.extend(self.find_pattern_matches(&graph, pattern)?);
        }

        // Apply size limits
        let limited_groups: Vec<_> = fusion_groups
            .into_iter()
            .map(|group| {
                if group.len() > rules.max_group_size {
                    group.into_iter().take(rules.max_group_size).collect()
                } else {
                    group
                }
            })
            .collect();

        self.create_fused_graph(graph, limited_groups)
    }

    /// Find chains of element-wise operations
    fn find_elementwise_chains(&self, graph: &ComputationGraph) -> JitResult<Vec<Vec<NodeId>>> {
        let mut chains = Vec::new();
        let mut visited = HashSet::new();

        for (node_id, node) in graph.nodes() {
            if visited.contains(&node_id) {
                continue;
            }

            if self.is_elementwise(&node.op) {
                let chain = self.build_elementwise_chain(graph, node_id, &mut visited)?;
                if chain.len() > 1 {
                    chains.push(chain);
                }
            }
        }

        Ok(chains)
    }

    /// Find conv+activation patterns
    fn find_conv_activation_patterns(
        &self,
        graph: &ComputationGraph,
    ) -> JitResult<Vec<Vec<NodeId>>> {
        let mut patterns = Vec::new();

        for (conv_id, conv_node) in graph.nodes() {
            if let Operation::Conv2d(_) = &conv_node.op {
                // Check successors for activations
                for succ_id in graph.successors(conv_id) {
                    if let Some(succ_node) = graph.node(succ_id) {
                        if self.is_activation(&succ_node.op) {
                            patterns.push(vec![conv_id, succ_id]);
                        }
                    }
                }
            }
        }

        Ok(patterns)
    }

    /// Find linear+activation patterns
    fn find_linear_activation_patterns(
        &self,
        graph: &ComputationGraph,
    ) -> JitResult<Vec<Vec<NodeId>>> {
        let mut patterns = Vec::new();

        for (linear_id, linear_node) in graph.nodes() {
            if let Operation::Linear(_) = &linear_node.op {
                // Check successors for activations
                for succ_id in graph.successors(linear_id) {
                    if let Some(succ_node) = graph.node(succ_id) {
                        if self.is_activation(&succ_node.op) {
                            patterns.push(vec![linear_id, succ_id]);
                        }
                    }
                }
            }
        }

        Ok(patterns)
    }

    /// Find reduction operation chains
    fn find_reduction_chains(&self, graph: &ComputationGraph) -> JitResult<Vec<Vec<NodeId>>> {
        let mut chains = Vec::new();
        let mut visited = HashSet::new();

        for (node_id, node) in graph.nodes() {
            if visited.contains(&node_id) {
                continue;
            }

            if self.is_reduction(&node.op) {
                let chain = self.build_reduction_chain(graph, node_id, &mut visited)?;
                if chain.len() > 1 {
                    chains.push(chain);
                }
            }
        }

        Ok(chains)
    }

    /// Find matmul chains (e.g., matmul + bias + activation)
    fn find_matmul_chains(&self, graph: &ComputationGraph) -> JitResult<Vec<Vec<NodeId>>> {
        let mut chains = Vec::new();

        for (mm_id, mm_node) in graph.nodes() {
            if matches!(&mm_node.op, Operation::MatMul | Operation::BatchMatMul) {
                let mut chain = vec![mm_id];
                let mut current = mm_id;

                // Look for bias add
                for succ_id in graph.successors(current) {
                    if let Some(succ_node) = graph.node(succ_id) {
                        if matches!(&succ_node.op, Operation::Add) {
                            chain.push(succ_id);
                            current = succ_id;
                            break;
                        }
                    }
                }

                // Look for activation
                for succ_id in graph.successors(current) {
                    if let Some(succ_node) = graph.node(succ_id) {
                        if self.is_activation(&succ_node.op) {
                            chain.push(succ_id);
                            break;
                        }
                    }
                }

                if chain.len() > 1 {
                    chains.push(chain);
                }
            }
        }

        Ok(chains)
    }

    /// Find matches for a specific pattern
    fn find_pattern_matches(
        &self,
        graph: &ComputationGraph,
        pattern: &FusionPattern,
    ) -> JitResult<Vec<Vec<NodeId>>> {
        // Simple pattern matching - could be made more sophisticated
        let mut matches = Vec::new();

        for (node_id, node) in graph.nodes() {
            if pattern.ops.contains(&node.op) {
                let mut group = vec![node_id];

                // Try to extend the group with compatible operations
                for succ_id in graph.successors(node_id) {
                    if let Some(succ_node) = graph.node(succ_id) {
                        if pattern.ops.contains(&succ_node.op) {
                            group.push(succ_id);
                        }
                    }
                }

                if group.len() > 1 {
                    matches.push(group);
                }
            }
        }

        Ok(matches)
    }

    /// Build a chain of element-wise operations
    fn build_elementwise_chain(
        &self,
        graph: &ComputationGraph,
        start: NodeId,
        visited: &mut HashSet<NodeId>,
    ) -> JitResult<Vec<NodeId>> {
        let mut chain = vec![start];
        visited.insert(start);

        let mut current = start;
        while let Some(next) = self.find_single_elementwise_successor(graph, current) {
            if visited.contains(&next) {
                break;
            }

            chain.push(next);
            visited.insert(next);
            current = next;
        }

        Ok(chain)
    }

    /// Build a chain of reduction operations
    fn build_reduction_chain(
        &self,
        graph: &ComputationGraph,
        start: NodeId,
        visited: &mut HashSet<NodeId>,
    ) -> JitResult<Vec<NodeId>> {
        let mut chain = vec![start];
        visited.insert(start);

        // Look for compatible reductions that can be fused
        for succ_id in graph.successors(start) {
            if let Some(succ_node) = graph.node(succ_id) {
                if self.is_reduction(&succ_node.op)
                    && self.can_fuse_reductions(&graph.node(start).unwrap().op, &succ_node.op)
                {
                    chain.push(succ_id);
                    visited.insert(succ_id);
                }
            }
        }

        Ok(chain)
    }

    /// Find single element-wise successor
    fn find_single_elementwise_successor(
        &self,
        graph: &ComputationGraph,
        node: NodeId,
    ) -> Option<NodeId> {
        let successors: Vec<_> = graph.successors(node).collect();

        if successors.len() == 1 {
            let succ_id = successors[0];
            if let Some(succ_node) = graph.node(succ_id) {
                if self.is_elementwise(&succ_node.op) {
                    // Check if this is the only predecessor
                    let predecessors: Vec<_> = graph.predecessors(succ_id).collect();
                    if predecessors.len() == 1 {
                        return Some(succ_id);
                    }
                }
            }
        }

        None
    }

    /// Merge overlapping fusion groups
    fn merge_fusion_groups(&self, groups: Vec<Vec<NodeId>>) -> Vec<Vec<NodeId>> {
        // Simple merge strategy - could be improved
        let mut merged = Vec::new();
        let mut used = HashSet::new();

        for group in groups {
            let new_group: IndexSet<NodeId> = group.into_iter().collect();

            // Check if any node is already used
            let overlap = new_group.iter().any(|n| used.contains(n));

            if !overlap {
                for &node in &new_group {
                    used.insert(node);
                }
                merged.push(new_group.into_iter().collect());
            }
        }

        merged
    }

    /// Grow fusion groups by including compatible neighbors
    fn grow_fusion_groups(
        &self,
        graph: &ComputationGraph,
        groups: Vec<Vec<NodeId>>,
    ) -> JitResult<Vec<Vec<NodeId>>> {
        let mut grown = Vec::new();

        for group in groups {
            let mut grown_group: IndexSet<NodeId> = group.into_iter().collect();
            let mut changed = true;

            while changed && grown_group.len() < 16 {
                // Limit growth
                changed = false;

                // Try to add compatible predecessors
                let current_nodes: Vec<_> = grown_group.iter().copied().collect();
                for node in current_nodes {
                    for pred in graph.predecessors(node) {
                        if !grown_group.contains(&pred) {
                            if let Some(pred_node) = graph.node(pred) {
                                if self.can_add_to_group(graph, &grown_group, pred, &pred_node.op) {
                                    grown_group.insert(pred);
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }

            grown.push(grown_group.into_iter().collect());
        }

        Ok(grown)
    }

    /// Check if an operation is element-wise
    fn is_elementwise(&self, op: &Operation) -> bool {
        matches!(
            op,
            Operation::Neg
                | Operation::Abs
                | Operation::Exp
                | Operation::Log
                | Operation::Sqrt
                | Operation::Sin
                | Operation::Cos
                | Operation::Tanh
                | Operation::Sigmoid
                | Operation::Relu
                | Operation::Gelu
                | Operation::Add
                | Operation::Sub
                | Operation::Mul
                | Operation::Div
                | Operation::Pow
                | Operation::Maximum
                | Operation::Minimum
        )
    }

    /// Check if an operation is an activation
    fn is_activation(&self, op: &Operation) -> bool {
        matches!(
            op,
            Operation::Relu | Operation::Sigmoid | Operation::Tanh | Operation::Gelu
        )
    }

    /// Check if an operation is a reduction
    fn is_reduction(&self, op: &Operation) -> bool {
        matches!(
            op,
            Operation::Sum { .. }
                | Operation::Mean { .. }
                | Operation::Max { .. }
                | Operation::Min { .. }
        )
    }

    /// Check if two reductions can be fused
    fn can_fuse_reductions(&self, op1: &Operation, op2: &Operation) -> bool {
        // For now, only fuse reductions along different dimensions
        match (op1, op2) {
            (Operation::Sum { dims: dims1, .. }, Operation::Sum { dims: dims2, .. })
            | (Operation::Mean { dims: dims1, .. }, Operation::Mean { dims: dims2, .. }) => {
                // Check if dimensions don't overlap
                dims1.iter().all(|d| !dims2.contains(d))
            }
            _ => false,
        }
    }

    /// Check if a node can be added to a fusion group
    fn can_add_to_group(
        &self,
        graph: &ComputationGraph,
        group: &IndexSet<NodeId>,
        node: NodeId,
        op: &Operation,
    ) -> bool {
        // Check if all dependencies are in the group or are external inputs
        for pred in graph.predecessors(node) {
            if !group.contains(&pred) && !graph.inputs.contains(&pred) {
                return false;
            }
        }

        // Check if operation is compatible
        self.is_fusable_operation(op)
    }

    /// Check if an operation can be fused
    fn is_fusable_operation(&self, op: &Operation) -> bool {
        self.is_elementwise(op) || self.is_activation(op) || self.is_simple_reshape(op)
    }

    /// Check if operation is a simple reshape that can be fused
    fn is_simple_reshape(&self, op: &Operation) -> bool {
        matches!(
            op,
            Operation::Reshape { .. } | Operation::Squeeze { .. } | Operation::Unsqueeze { .. }
        )
    }

    /// Advanced pattern matching for fusion opportunities
    fn find_advanced_patterns(&self, graph: &ComputationGraph) -> JitResult<Vec<Vec<NodeId>>> {
        let mut patterns = Vec::new();

        // Find LayerNorm-like patterns
        patterns.extend(self.find_layernorm_patterns(graph)?);

        // Find GELU approximation patterns
        patterns.extend(self.find_gelu_patterns(graph)?);

        // Find attention patterns
        patterns.extend(self.find_attention_patterns(graph)?);

        Ok(patterns)
    }

    /// Find LayerNorm patterns: mean -> variance -> normalize
    fn find_layernorm_patterns(&self, graph: &ComputationGraph) -> JitResult<Vec<Vec<NodeId>>> {
        let mut patterns = Vec::new();

        for (node_id, node) in graph.nodes() {
            // Look for variance computation pattern
            if let Operation::Mean { .. } = &node.op {
                // Check if this could be part of a LayerNorm
                if let Some(pattern) = self.try_match_layernorm_from_mean(graph, node_id) {
                    patterns.push(pattern);
                }
            }
        }

        Ok(patterns)
    }

    /// Try to match LayerNorm pattern starting from a mean operation
    fn try_match_layernorm_from_mean(
        &self,
        graph: &ComputationGraph,
        mean_node: NodeId,
    ) -> Option<Vec<NodeId>> {
        let mut pattern = vec![mean_node];

        // Look for subsequent operations that might be part of LayerNorm
        for succ in graph.successors(mean_node) {
            if let Some(succ_node) = graph.node(succ) {
                match &succ_node.op {
                    Operation::Sub => {
                        pattern.push(succ);
                        // Continue pattern matching...
                    }
                    _ => continue,
                }
            }
        }

        if pattern.len() > 3 {
            Some(pattern)
        } else {
            None
        }
    }

    /// Find GELU approximation patterns
    fn find_gelu_patterns(&self, graph: &ComputationGraph) -> JitResult<Vec<Vec<NodeId>>> {
        let mut patterns = Vec::new();

        // GELU can be approximated as: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        for (node_id, node) in graph.nodes() {
            if let Operation::Tanh = &node.op {
                if let Some(pattern) = self.try_match_gelu_from_tanh(graph, node_id) {
                    patterns.push(pattern);
                }
            }
        }

        Ok(patterns)
    }

    /// Try to match GELU pattern starting from tanh
    fn try_match_gelu_from_tanh(
        &self,
        _graph: &ComputationGraph,
        tanh_node: NodeId,
    ) -> Option<Vec<NodeId>> {
        // This would implement the full GELU pattern matching
        // For now, return a simple pattern
        Some(vec![tanh_node])
    }

    /// Find attention computation patterns
    fn find_attention_patterns(&self, graph: &ComputationGraph) -> JitResult<Vec<Vec<NodeId>>> {
        let mut patterns = Vec::new();

        // Look for QKV computation patterns
        for (node_id, node) in graph.nodes() {
            if let Operation::MatMul = &node.op {
                if let Some(pattern) = self.try_match_attention_from_matmul(graph, node_id) {
                    patterns.push(pattern);
                }
            }
        }

        Ok(patterns)
    }

    /// Try to match attention pattern starting from matrix multiplication
    fn try_match_attention_from_matmul(
        &self,
        graph: &ComputationGraph,
        matmul_node: NodeId,
    ) -> Option<Vec<NodeId>> {
        let mut pattern = vec![matmul_node];

        // Look for softmax after matmul (attention scores)
        for succ in graph.successors(matmul_node) {
            if let Some(succ_node) = graph.node(succ) {
                // This would be more sophisticated pattern matching
                if self.is_attention_related(&succ_node.op) {
                    pattern.push(succ);
                }
            }
        }

        if pattern.len() > 1 {
            Some(pattern)
        } else {
            None
        }
    }

    /// Check if operation is related to attention computation
    fn is_attention_related(&self, op: &Operation) -> bool {
        matches!(
            op,
            Operation::Div | // Scale by sqrt(d_k)
            Operation::Add | // Add bias or mask
            // Note: Softmax would be a custom operation
            Operation::MatMul // Value computation
        )
    }

    /// Create a new graph with fused nodes
    fn create_fused_graph(
        &self,
        graph: ComputationGraph,
        fusion_groups: Vec<Vec<NodeId>>,
    ) -> JitResult<ComputationGraph> {
        use crate::graph::{ComputationGraph, Edge, Node, Operation};
        use std::collections::{HashMap, HashSet};

        if fusion_groups.is_empty() {
            return Ok(graph);
        }

        // Create a new graph to avoid node index invalidation issues
        let mut new_graph = ComputationGraph::new();
        let mut node_mapping: HashMap<NodeId, NodeId> = HashMap::new();

        // Get all nodes to be fused
        let fused_nodes: HashSet<NodeId> = fusion_groups.iter().flatten().copied().collect();

        // First pass: Add non-fused nodes to the new graph
        for (node_id, node) in graph.nodes() {
            if !fused_nodes.contains(&node_id) {
                let new_node_id = new_graph.add_node(node.clone());
                node_mapping.insert(node_id, new_node_id);
            }
        }

        // Second pass: Create fused nodes
        for group in &fusion_groups {
            if group.is_empty() {
                continue;
            }

            // Collect information about the fusion group
            let mut group_inputs = IndexSet::new();
            let mut group_outputs = IndexSet::new();
            let mut operations = Vec::new();

            // Find all inputs and outputs of the fusion group
            for &node_id in group {
                if let Some(node) = graph.node(node_id) {
                    operations.push(node.op.clone());

                    // Collect inputs (predecessors not in the group)
                    for pred in graph.predecessors(node_id) {
                        if !group.contains(&pred) {
                            group_inputs.insert(pred);
                        }
                    }

                    // Collect outputs (successors not in the group)
                    for succ in graph.successors(node_id) {
                        if !group.contains(&succ) {
                            group_outputs.insert(succ);
                        }
                    }
                }
            }

            // Create a fused node
            let fused_node_name = format!(
                "fused_{}",
                group.iter().map(|id| id.index()).min().unwrap_or(0)
            );

            // Determine output shape and dtype from the last node in the group
            let last_node_id = group.last().unwrap();
            let (output_shape, dtype, device) = if let Some(last_node) = graph.node(*last_node_id) {
                (
                    last_node.output_shape.clone(),
                    last_node.dtype,
                    last_node.device,
                )
            } else {
                continue; // Skip if node not found
            };

            // Create the fused operation
            let fused_op = Operation::FusedKernel {
                name: fused_node_name.clone(),
                ops: operations,
                input_count: group_inputs.len(),
                output_count: 1, // For now, assume single output
            };

            // Add the fused node to the new graph
            let fused_node = Node::new(fused_op, fused_node_name)
                .with_output_shapes(vec![Some(output_shape)])
                .with_dtypes(vec![dtype])
                .with_device(device);

            let fused_node_id = new_graph.add_node(fused_node);

            // Map all nodes in the group to the fused node
            for &node_id in group {
                node_mapping.insert(node_id, fused_node_id);
            }
        }

        // Third pass: Add edges to the new graph
        for (src, dst, edge) in graph.edges() {
            if let (Some(&new_src), Some(&new_dst)) =
                (node_mapping.get(&src), node_mapping.get(&dst))
            {
                // Skip self-edges that might occur from fusion
                if new_src != new_dst {
                    new_graph.add_edge(new_src, new_dst, edge.clone());
                }
            }
        }

        // Update inputs and outputs lists
        let mut new_inputs = Vec::new();
        for input in &graph.inputs {
            if let Some(&new_id) = node_mapping.get(input) {
                if !new_inputs.contains(&new_id) {
                    new_inputs.push(new_id);
                }
            }
        }

        let mut new_outputs = Vec::new();
        for output in &graph.outputs {
            if let Some(&new_id) = node_mapping.get(output) {
                if !new_outputs.contains(&new_id) {
                    new_outputs.push(new_id);
                }
            }
        }

        // Set the new graph's inputs and outputs
        new_graph.inputs = new_inputs;
        new_graph.outputs = new_outputs;
        new_graph.metadata = graph.metadata.clone();

        Ok(new_graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_strategy() {
        let strategy = FusionStrategy::Default;
        let _fusion = KernelFusion::new(strategy);

        // Basic test
        assert!(true);
    }

    #[test]
    fn test_elementwise_detection() {
        let fusion = KernelFusion::new(FusionStrategy::Default);

        assert!(fusion.is_elementwise(&Operation::Add));
        assert!(fusion.is_elementwise(&Operation::Relu));
        assert!(
            !fusion.is_elementwise(&Operation::Conv2d(crate::graph::Conv2dInfo {
                in_channels: 3,
                out_channels: 64,
                kernel_size: (3, 3),
                stride: (1, 1),
                padding: (1, 1),
                dilation: (1, 1),
                groups: 1,
            }))
        );
    }

    #[test]
    fn test_activation_detection() {
        let fusion = KernelFusion::new(FusionStrategy::Default);

        assert!(fusion.is_activation(&Operation::Relu));
        assert!(fusion.is_activation(&Operation::Sigmoid));
        assert!(!fusion.is_activation(&Operation::Add));
    }
}
