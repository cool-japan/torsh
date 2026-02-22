//! Analysis utilities for JIT compilation

use crate::graph::{ComputationGraph, Node, NodeId, Operation};
use crate::{JitError, JitResult};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

/// Analysis results for a computation graph
#[derive(Debug, Clone)]
pub struct GraphAnalysis {
    /// Memory usage per node
    pub memory_usage: HashMap<NodeId, MemoryInfo>,

    /// Computational complexity per node
    pub compute_cost: HashMap<NodeId, ComputeCost>,

    /// Data dependencies
    pub dependencies: DependencyInfo,

    /// Critical path through the graph
    pub critical_path: Vec<NodeId>,

    /// Parallelization opportunities
    pub parallel_groups: Vec<Vec<NodeId>>,
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Output tensor size in bytes
    pub output_size: usize,

    /// Temporary memory required
    pub temp_size: usize,

    /// Total memory footprint
    pub total_size: usize,

    /// Memory access pattern
    pub access_pattern: AccessPattern,
}

/// Memory access pattern
#[derive(Debug, Clone, PartialEq)]
pub enum AccessPattern {
    Sequential,
    Strided { stride: usize },
    Random,
    Broadcast,
}

/// Computational cost estimation
#[derive(Debug, Clone)]
pub struct ComputeCost {
    /// Floating point operations
    pub flops: u64,

    /// Memory operations (loads + stores)
    pub memory_ops: u64,

    /// Estimated cycles (device-specific)
    pub cycles: u64,

    /// Operation intensity (flops / memory_ops)
    pub intensity: f32,
}

/// Dependency information
#[derive(Debug, Clone)]
pub struct DependencyInfo {
    /// Direct dependencies (node -> predecessors)
    pub direct: HashMap<NodeId, Vec<NodeId>>,

    /// Transitive dependencies (node -> all ancestors)
    pub transitive: HashMap<NodeId, HashSet<NodeId>>,

    /// Dependency depth for each node
    pub depth: HashMap<NodeId, usize>,
}

/// Graph analyzer
pub struct GraphAnalyzer;

impl GraphAnalyzer {
    /// Analyze a computation graph
    pub fn analyze(graph: &ComputationGraph) -> JitResult<GraphAnalysis> {
        let memory_usage = Self::analyze_memory(graph)?;
        let compute_cost = Self::analyze_compute(graph)?;
        let dependencies = Self::analyze_dependencies(graph)?;
        let critical_path = Self::find_critical_path(graph, &compute_cost)?;
        let parallel_groups = Self::find_parallel_groups(graph, &dependencies)?;

        Ok(GraphAnalysis {
            memory_usage,
            compute_cost,
            dependencies,
            critical_path,
            parallel_groups,
        })
    }

    /// Analyze memory usage
    fn analyze_memory(graph: &ComputationGraph) -> JitResult<HashMap<NodeId, MemoryInfo>> {
        let mut memory_info = HashMap::new();

        for (node_id, node) in graph.nodes() {
            let info = Self::compute_memory_info(node)?;
            memory_info.insert(node_id, info);
        }

        Ok(memory_info)
    }

    /// Compute memory info for a node
    fn compute_memory_info(node: &Node) -> JitResult<MemoryInfo> {
        let element_size = match node.dtype {
            torsh_core::DType::F32 => 4,
            torsh_core::DType::F64 => 8,
            torsh_core::DType::I32 => 4,
            torsh_core::DType::I64 => 8,
            torsh_core::DType::I8 => 1,
            torsh_core::DType::U8 => 1,
            torsh_core::DType::U32 => 4,
            torsh_core::DType::U64 => 8,
            torsh_core::DType::Bool => 1,
            torsh_core::DType::F16 | torsh_core::DType::BF16 | torsh_core::DType::I16 => 2,
            torsh_core::DType::C64 => 8,
            torsh_core::DType::C128 => 16,
            torsh_core::DType::QInt8 | torsh_core::DType::QUInt8 => 1,
            torsh_core::DType::QInt32 => 4, // Quantized 32-bit type
        };

        let num_elements = node.output_shape.numel();
        let output_size = num_elements * element_size;

        // Estimate temporary memory based on operation
        let (temp_size, access_pattern) = match &node.op {
            Operation::MatMul | Operation::BatchMatMul => {
                // Matrix multiplication may need temporary storage
                (output_size, AccessPattern::Sequential)
            }
            Operation::Conv2d(_) => {
                // Convolution needs im2col buffer
                (output_size * 2, AccessPattern::Strided { stride: 1 })
            }
            Operation::Transpose { .. } => (0, AccessPattern::Strided { stride: 1 }),
            Operation::Sum { .. } | Operation::Mean { .. } => {
                (element_size * 1024, AccessPattern::Sequential) // Small temp buffer
            }
            _ => (0, AccessPattern::Sequential),
        };

        Ok(MemoryInfo {
            output_size,
            temp_size,
            total_size: output_size + temp_size,
            access_pattern,
        })
    }

    /// Analyze computational cost
    fn analyze_compute(graph: &ComputationGraph) -> JitResult<HashMap<NodeId, ComputeCost>> {
        let mut compute_costs = HashMap::new();

        for (node_id, node) in graph.nodes() {
            let cost = Self::estimate_compute_cost(node)?;
            compute_costs.insert(node_id, cost);
        }

        Ok(compute_costs)
    }

    /// Estimate computational cost for a node
    fn estimate_compute_cost(node: &Node) -> JitResult<ComputeCost> {
        let num_elements = node.output_shape.numel();

        let (flops, memory_ops) = match &node.op {
            // Element-wise operations
            Operation::Add | Operation::Sub => (num_elements as u64, num_elements as u64 * 3),
            Operation::Mul | Operation::Div => (num_elements as u64, num_elements as u64 * 3),
            Operation::Exp | Operation::Log | Operation::Sqrt => {
                (num_elements as u64 * 10, num_elements as u64 * 2)
            }
            Operation::Sin | Operation::Cos => (num_elements as u64 * 20, num_elements as u64 * 2),

            // Activations
            Operation::Relu => (num_elements as u64, num_elements as u64 * 2),
            Operation::Sigmoid | Operation::Tanh => {
                (num_elements as u64 * 5, num_elements as u64 * 2)
            }
            Operation::Gelu => (num_elements as u64 * 10, num_elements as u64 * 2),

            // Matrix operations
            Operation::MatMul => {
                // Assuming shape is [M, K] x [K, N] -> [M, N]
                if node.output_shape.ndim() >= 2 {
                    let dims = node.output_shape.dims();
                    let m = dims[dims.len() - 2];
                    let n = dims[dims.len() - 1];
                    let k = m; // Estimate, would need input shapes
                    ((2 * m * n * k) as u64, (m * k + k * n + m * n) as u64)
                } else {
                    (num_elements as u64, num_elements as u64 * 2)
                }
            }

            // Reductions
            Operation::Sum { .. } | Operation::Mean { .. } => {
                (num_elements as u64, num_elements as u64 + 1)
            }

            // Convolution
            Operation::Conv2d(info) => {
                // FLOPs = 2 * output_size * kernel_size * in_channels
                let kernel_ops = info.kernel_size.0 * info.kernel_size.1 * info.in_channels;
                (
                    num_elements as u64 * kernel_ops as u64 * 2,
                    num_elements as u64 * 3,
                )
            }

            _ => (num_elements as u64, num_elements as u64 * 2),
        };

        let intensity = if memory_ops > 0 {
            flops as f32 / memory_ops as f32
        } else {
            0.0
        };

        // Simple cycle estimation (would be device-specific in practice)
        let cycles = flops.max(memory_ops * 4);

        Ok(ComputeCost {
            flops,
            memory_ops,
            cycles,
            intensity,
        })
    }

    /// Analyze dependencies
    fn analyze_dependencies(graph: &ComputationGraph) -> JitResult<DependencyInfo> {
        let mut direct = HashMap::new();
        let mut transitive = HashMap::new();
        let mut depth = HashMap::new();

        // Get topological order
        let order = graph
            .topological_sort()
            .map_err(|e| JitError::GraphError(format!("{:?}", e)))?;

        // Build dependency information
        for &node_id in &order {
            // Direct dependencies
            let preds: Vec<_> = graph.predecessors(node_id).collect();
            direct.insert(node_id, preds.clone());

            // Transitive dependencies
            let mut trans_deps = HashSet::new();
            for &pred in &preds {
                trans_deps.insert(pred);
                if let Some(pred_trans) = transitive.get(&pred) {
                    trans_deps.extend(pred_trans);
                }
            }
            transitive.insert(node_id, trans_deps);

            // Depth
            let node_depth = preds
                .iter()
                .map(|&p| depth.get(&p).copied().unwrap_or(0))
                .max()
                .unwrap_or(0)
                + 1;
            depth.insert(node_id, node_depth);
        }

        Ok(DependencyInfo {
            direct,
            transitive,
            depth,
        })
    }

    /// Find the critical path through the graph
    fn find_critical_path(
        graph: &ComputationGraph,
        compute_costs: &HashMap<NodeId, ComputeCost>,
    ) -> JitResult<Vec<NodeId>> {
        let mut distances = HashMap::new();
        let mut predecessors = HashMap::new();

        // Initialize distances
        for (node_id, _) in graph.nodes() {
            distances.insert(node_id, 0u64);
        }

        // Compute longest path using topological order
        let order = graph
            .topological_sort()
            .map_err(|e| JitError::GraphError(format!("{:?}", e)))?;

        for &node_id in &order {
            let node_cost = compute_costs.get(&node_id).map(|c| c.cycles).unwrap_or(0);

            let current_dist = distances[&node_id] + node_cost;

            // Update distances to successors
            for succ_id in graph.successors(node_id) {
                let succ_dist = distances.get(&succ_id).copied().unwrap_or(0);

                if current_dist > succ_dist {
                    distances.insert(succ_id, current_dist);
                    predecessors.insert(succ_id, node_id);
                }
            }
        }

        // Find the output with maximum distance
        let mut end_node = None;
        let mut max_dist = 0;

        for &output in &graph.outputs {
            if let Some(&dist) = distances.get(&output) {
                if dist > max_dist {
                    max_dist = dist;
                    end_node = Some(output);
                }
            }
        }

        // Reconstruct path
        let mut path = Vec::new();
        let mut current = end_node;

        while let Some(node) = current {
            path.push(node);
            current = predecessors.get(&node).copied();
        }

        path.reverse();
        Ok(path)
    }

    /// Find groups of nodes that can be executed in parallel
    fn find_parallel_groups(
        _graph: &ComputationGraph,
        dependencies: &DependencyInfo,
    ) -> JitResult<Vec<Vec<NodeId>>> {
        let mut groups = Vec::new();
        let mut assigned = HashSet::new();

        // Group nodes by depth
        let mut depth_groups: HashMap<usize, Vec<NodeId>> = HashMap::new();
        for (&node_id, &depth) in &dependencies.depth {
            depth_groups.entry(depth).or_default().push(node_id);
        }

        // Create parallel groups from each depth level
        let mut depths: Vec<_> = depth_groups.keys().copied().collect();
        depths.sort();

        for depth in depths {
            if let Some(nodes) = depth_groups.get(&depth) {
                let mut current_group = Vec::new();

                for &node in nodes {
                    if !assigned.contains(&node) {
                        // Check if node can be added to current group
                        let can_add = current_group.iter().all(|&other| {
                            !Self::has_dependency(dependencies, node, other)
                                && !Self::has_dependency(dependencies, other, node)
                        });

                        if can_add {
                            current_group.push(node);
                            assigned.insert(node);
                        }
                    }
                }

                if !current_group.is_empty() {
                    groups.push(current_group);
                }
            }
        }

        Ok(groups)
    }

    /// Check if node1 depends on node2
    fn has_dependency(dependencies: &DependencyInfo, node1: NodeId, node2: NodeId) -> bool {
        dependencies
            .transitive
            .get(&node1)
            .map(|deps| deps.contains(&node2))
            .unwrap_or(false)
    }
}

/// Data flow analysis for optimization opportunities
#[derive(Debug, Clone)]
pub struct DataFlowAnalysis {
    /// Variable definitions: which node defines each variable
    pub definitions: HashMap<String, NodeId>,

    /// Variable uses: which nodes use each variable
    pub uses: HashMap<String, Vec<NodeId>>,

    /// Live variables at each node
    pub live_variables: HashMap<NodeId, HashSet<String>>,

    /// Reaching definitions for each node
    pub reaching_definitions: HashMap<NodeId, HashMap<String, NodeId>>,

    /// Available expressions at each node
    pub available_expressions: HashMap<NodeId, HashSet<Expression>>,

    /// Dead code nodes
    pub dead_code: Vec<NodeId>,

    /// Common subexpressions
    pub common_subexpressions: Vec<CommonSubexpression>,
}

/// Expression representation for CSE analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expression {
    /// Operation type
    pub operation: String,
    /// Input variables
    pub inputs: Vec<String>,
    /// Additional attributes for matching
    pub attributes: HashMap<String, String>,
}

impl Hash for Expression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.operation.hash(state);
        self.inputs.hash(state);
        // Hash attributes by sorting keys to ensure consistent order
        let mut attr_pairs: Vec<_> = self.attributes.iter().collect();
        attr_pairs.sort_by_key(|(k, _)| *k);
        attr_pairs.hash(state);
    }
}

/// Common subexpression that can be eliminated
#[derive(Debug, Clone)]
pub struct CommonSubexpression {
    /// The expression
    pub expression: Expression,
    /// Nodes that compute this expression
    pub instances: Vec<NodeId>,
    /// Potential savings (memory, compute)
    pub savings: OptimizationSavings,
}

/// Savings from applying an optimization
#[derive(Debug, Clone)]
pub struct OptimizationSavings {
    /// Memory savings in bytes
    pub memory_bytes: usize,
    /// Compute savings in FLOPs
    pub compute_flops: u64,
    /// Estimated speedup factor
    pub speedup_factor: f32,
}

/// Data flow analyzer
pub struct DataFlowAnalyzer;

impl DataFlowAnalyzer {
    /// Perform complete data flow analysis
    pub fn analyze(graph: &ComputationGraph) -> JitResult<DataFlowAnalysis> {
        let mut analysis = DataFlowAnalysis {
            definitions: HashMap::new(),
            uses: HashMap::new(),
            live_variables: HashMap::new(),
            reaching_definitions: HashMap::new(),
            available_expressions: HashMap::new(),
            dead_code: Vec::new(),
            common_subexpressions: Vec::new(),
        };

        // Build def-use chains
        Self::build_def_use_chains(graph, &mut analysis)?;

        // Compute live variables
        Self::compute_live_variables(graph, &mut analysis)?;

        // Compute reaching definitions
        Self::compute_reaching_definitions(graph, &mut analysis)?;

        // Find available expressions
        Self::compute_available_expressions(graph, &mut analysis)?;

        // Identify dead code
        Self::identify_dead_code(graph, &mut analysis)?;

        // Find common subexpressions
        Self::find_common_subexpressions(graph, &mut analysis)?;

        Ok(analysis)
    }

    /// Build definition-use chains
    fn build_def_use_chains(
        graph: &ComputationGraph,
        analysis: &mut DataFlowAnalysis,
    ) -> JitResult<()> {
        for (node_id, node) in graph.nodes() {
            let var_name = node.name.clone();

            // This node defines the variable
            analysis.definitions.insert(var_name.clone(), node_id);

            // Find which variables this node uses
            let used_vars = Self::get_input_variables(graph, node_id);
            for var in used_vars {
                analysis.uses.entry(var).or_default().push(node_id);
            }
        }
        Ok(())
    }

    /// Get input variables for a node
    fn get_input_variables(graph: &ComputationGraph, node_id: NodeId) -> Vec<String> {
        let mut vars = Vec::new();

        // Get predecessors and their variable names
        for edge in graph.edges_directed(node_id, petgraph::Direction::Incoming) {
            if let Some(pred_node) = graph.node(edge.source()) {
                vars.push(pred_node.name.clone());
            }
        }

        vars
    }

    /// Compute live variables using backward data flow analysis
    fn compute_live_variables(
        graph: &ComputationGraph,
        analysis: &mut DataFlowAnalysis,
    ) -> JitResult<()> {
        let topo_order = graph
            .topological_sort()
            .map_err(|e| JitError::AnalysisError(format!("Topological sort failed: {}", e)))?;

        // Initialize with empty sets
        for &node_id in &topo_order {
            analysis.live_variables.insert(node_id, HashSet::new());
        }

        // Backward pass
        let mut changed = true;
        while changed {
            changed = false;

            for &node_id in topo_order.iter().rev() {
                let mut new_live = HashSet::new();

                // Add variables used by successors
                for edge in graph.edges_directed(node_id, petgraph::Direction::Outgoing) {
                    let succ_id = edge.target();
                    let used_vars = Self::get_input_variables(graph, succ_id);
                    for var in used_vars {
                        new_live.insert(var);
                    }

                    if let Some(succ_live) = analysis.live_variables.get(&succ_id) {
                        new_live.extend(succ_live.clone());
                    }
                }

                // Remove variable defined by this node
                if let Some(node) = graph.node(node_id) {
                    new_live.remove(&node.name);
                }

                // Add variables used by this node
                let used_vars = Self::get_input_variables(graph, node_id);
                for var in used_vars {
                    new_live.insert(var);
                }

                if analysis.live_variables.get(&node_id) != Some(&new_live) {
                    analysis.live_variables.insert(node_id, new_live);
                    changed = true;
                }
            }
        }

        Ok(())
    }

    /// Compute reaching definitions
    fn compute_reaching_definitions(
        graph: &ComputationGraph,
        analysis: &mut DataFlowAnalysis,
    ) -> JitResult<()> {
        let topo_order = graph
            .topological_sort()
            .map_err(|e| JitError::AnalysisError(format!("Topological sort failed: {}", e)))?;

        // Initialize
        for &node_id in &topo_order {
            analysis
                .reaching_definitions
                .insert(node_id, HashMap::new());
        }

        // Forward pass
        let mut changed = true;
        while changed {
            changed = false;

            for &node_id in &topo_order {
                let mut new_defs = HashMap::new();

                // Union of reaching definitions from predecessors
                for edge in graph.edges_directed(node_id, petgraph::Direction::Incoming) {
                    let pred_id = edge.source();
                    if let Some(pred_defs) = analysis.reaching_definitions.get(&pred_id) {
                        for (var, &def_node) in pred_defs {
                            new_defs.insert(var.clone(), def_node);
                        }
                    }
                }

                // This node defines a variable
                if let Some(node) = graph.node(node_id) {
                    new_defs.insert(node.name.clone(), node_id);
                }

                if analysis.reaching_definitions.get(&node_id) != Some(&new_defs) {
                    analysis.reaching_definitions.insert(node_id, new_defs);
                    changed = true;
                }
            }
        }

        Ok(())
    }

    /// Compute available expressions
    fn compute_available_expressions(
        graph: &ComputationGraph,
        analysis: &mut DataFlowAnalysis,
    ) -> JitResult<()> {
        let topo_order = graph
            .topological_sort()
            .map_err(|e| JitError::AnalysisError(format!("Topological sort failed: {}", e)))?;

        // Initialize
        for &node_id in &topo_order {
            analysis
                .available_expressions
                .insert(node_id, HashSet::new());
        }

        // Forward pass
        let mut changed = true;
        while changed {
            changed = false;

            for &node_id in &topo_order {
                let mut new_exprs = HashSet::new();

                // Intersection of available expressions from predecessors
                let mut pred_exprs = None;
                for edge in graph.edges_directed(node_id, petgraph::Direction::Incoming) {
                    let pred_id = edge.source();
                    if let Some(exprs) = analysis.available_expressions.get(&pred_id) {
                        match pred_exprs {
                            None => pred_exprs = Some(exprs.clone()),
                            Some(ref mut current) => {
                                *current = current.intersection(exprs).cloned().collect();
                            }
                        }
                    }
                }

                if let Some(exprs) = pred_exprs {
                    new_exprs = exprs;
                }

                // Add expression computed by this node
                if let Some(node) = graph.node(node_id) {
                    let expr = Self::node_to_expression(graph, node_id, node);
                    new_exprs.insert(expr);
                }

                if analysis.available_expressions.get(&node_id) != Some(&new_exprs) {
                    analysis.available_expressions.insert(node_id, new_exprs);
                    changed = true;
                }
            }
        }

        Ok(())
    }

    /// Convert a node to an expression
    fn node_to_expression(graph: &ComputationGraph, node_id: NodeId, node: &Node) -> Expression {
        let operation = format!("{:?}", node.op);
        let inputs = Self::get_input_variables(graph, node_id);
        let mut attributes = HashMap::new();

        // Add relevant attributes
        for (key, attr) in &node.attrs {
            let value = match attr {
                crate::graph::Attribute::String(s) => s.clone(),
                crate::graph::Attribute::Int(i) => i.to_string(),
                crate::graph::Attribute::Float(f) => f.to_string(),
                crate::graph::Attribute::Bool(b) => b.to_string(),
                _ => "complex".to_string(),
            };
            attributes.insert(key.clone(), value);
        }

        Expression {
            operation,
            inputs,
            attributes,
        }
    }

    /// Identify dead code
    fn identify_dead_code(
        graph: &ComputationGraph,
        analysis: &mut DataFlowAnalysis,
    ) -> JitResult<()> {
        let outputs: HashSet<_> = graph.outputs.iter().copied().collect();

        for (node_id, _) in graph.nodes() {
            // A node is dead if:
            // 1. It's not an output node
            // 2. No live variable depends on it
            // 3. It has no side effects

            if outputs.contains(&node_id) {
                continue; // Output nodes are always live
            }

            let is_used = analysis.uses.values().any(|users| users.contains(&node_id));

            if !is_used {
                if let Some(node) = graph.node(node_id) {
                    // Check if the operation has side effects
                    if !Self::has_side_effects(&node.op) {
                        analysis.dead_code.push(node_id);
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if an operation has side effects
    fn has_side_effects(op: &Operation) -> bool {
        match op {
            Operation::Custom(_) => true, // Conservative assumption
            _ => false,                   // Pure operations
        }
    }

    /// Find common subexpressions
    fn find_common_subexpressions(
        graph: &ComputationGraph,
        analysis: &mut DataFlowAnalysis,
    ) -> JitResult<()> {
        let mut expr_to_nodes: HashMap<Expression, Vec<NodeId>> = HashMap::new();

        // Group nodes by their expressions
        for (node_id, node) in graph.nodes() {
            let expr = Self::node_to_expression(graph, node_id, node);
            expr_to_nodes.entry(expr).or_default().push(node_id);
        }

        // Find expressions computed by multiple nodes
        for (expr, nodes) in expr_to_nodes {
            if nodes.len() > 1 {
                let savings = Self::estimate_cse_savings(graph, &nodes);
                analysis.common_subexpressions.push(CommonSubexpression {
                    expression: expr,
                    instances: nodes,
                    savings,
                });
            }
        }

        Ok(())
    }

    /// Estimate savings from eliminating a common subexpression
    fn estimate_cse_savings(graph: &ComputationGraph, nodes: &[NodeId]) -> OptimizationSavings {
        let mut total_memory = 0;
        let mut total_flops = 0;

        for &node_id in nodes {
            if let Some(node) = graph.node(node_id) {
                // Estimate memory savings (all but one instance)
                let element_size = match node.dtype {
                    torsh_core::DType::F32 => 4,
                    torsh_core::DType::F64 => 8,
                    _ => 4, // Default
                };
                total_memory += node.output_shape.numel() * element_size;

                // Estimate compute savings
                total_flops += match &node.op {
                    Operation::Add | Operation::Sub | Operation::Mul => {
                        node.output_shape.numel() as u64
                    }
                    Operation::MatMul => {
                        // Simplified: assume square matrices
                        let n = (node.output_shape.numel() as f64).sqrt() as u64;
                        n * n * n // O(n^3) for matrix multiplication
                    }
                    _ => node.output_shape.numel() as u64,
                };
            }
        }

        // Savings from eliminating all but one instance
        let instances = nodes.len();
        if instances > 1 {
            let memory_savings = total_memory * (instances - 1) / instances;
            let compute_savings = total_flops * (instances - 1) as u64 / instances as u64;
            let speedup = 1.0 + (instances - 1) as f32 * 0.1; // Conservative estimate

            OptimizationSavings {
                memory_bytes: memory_savings,
                compute_flops: compute_savings,
                speedup_factor: speedup,
            }
        } else {
            OptimizationSavings {
                memory_bytes: 0,
                compute_flops: 0,
                speedup_factor: 1.0,
            }
        }
    }
}

impl DataFlowAnalysis {
    /// Get optimization recommendations
    pub fn get_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Dead code elimination
        if !self.dead_code.is_empty() {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::DeadCodeElimination,
                description: format!("Remove {} dead code nodes", self.dead_code.len()),
                nodes: self.dead_code.clone(),
                estimated_savings: OptimizationSavings {
                    memory_bytes: self.dead_code.len() * 1024, // Rough estimate
                    compute_flops: self.dead_code.len() as u64 * 100,
                    speedup_factor: 1.0 + self.dead_code.len() as f32 * 0.01,
                },
            });
        }

        // Common subexpression elimination
        for cse in &self.common_subexpressions {
            if cse.instances.len() > 1 {
                recommendations.push(OptimizationRecommendation {
                    optimization_type: OptimizationType::CommonSubexpressionElimination,
                    description: format!(
                        "Eliminate common subexpression computed by {} nodes",
                        cse.instances.len()
                    ),
                    nodes: cse.instances.clone(),
                    estimated_savings: cse.savings.clone(),
                });
            }
        }

        recommendations
    }
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Human-readable description
    pub description: String,
    /// Nodes involved in the optimization
    pub nodes: Vec<NodeId>,
    /// Estimated savings
    pub estimated_savings: OptimizationSavings,
}

/// Types of optimizations
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    DeadCodeElimination,
    CommonSubexpressionElimination,
    LoopInvariantCodeMotion,
    ConstantFolding,
    StrengthReduction,
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::{DType, DeviceType, Shape};

    #[test]
    fn test_memory_info_computation() {
        let node = Node::new(Operation::Relu, "test".to_string())
            .with_output_shapes(vec![Some(Shape::new(vec![32, 64]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu);

        let info = GraphAnalyzer::compute_memory_info(&node).unwrap();
        assert_eq!(info.output_size, 32 * 64 * 4); // 4 bytes per f32
        assert_eq!(info.temp_size, 0); // ReLU needs no temp storage
    }

    #[test]
    fn test_compute_cost_estimation() {
        let node = Node::new(Operation::Add, "add".to_string())
            .with_output_shapes(vec![Some(Shape::new(vec![1000]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu);

        let cost = GraphAnalyzer::estimate_compute_cost(&node).unwrap();
        assert_eq!(cost.flops, 1000);
        assert_eq!(cost.memory_ops, 3000); // 2 reads + 1 write
        assert!(cost.intensity < 1.0); // Memory bound
    }
}
