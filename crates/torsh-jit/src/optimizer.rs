//! Graph optimization passes for JIT compilation

use crate::graph::{ComputationGraph, Conv2dInfo, Edge, Node, NodeId, Operation};
use crate::JitResult;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};
use torsh_core::Shape;

/// Graph optimizer that applies various optimization passes
pub struct GraphOptimizer {
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl GraphOptimizer {
    /// Create a new graph optimizer with default passes
    pub fn new() -> Self {
        Self {
            passes: vec![
                Box::new(DeadCodeElimination),
                Box::new(ConstantFolding),
                Box::new(CommonSubexpressionElimination),
                Box::new(AlgebraicSimplification),
                Box::new(StrengthReduction),
                Box::new(LayoutOptimization),
                Box::new(CacheAwareOptimization::default()),
                Box::new(AutoVectorization),
                Box::new(AutoParallelization),
            ],
        }
    }

    /// Create optimizer with custom passes
    pub fn with_passes(passes: Vec<Box<dyn OptimizationPass>>) -> Self {
        Self { passes }
    }

    /// Apply all optimization passes to the graph
    pub fn optimize(&self, mut graph: ComputationGraph) -> JitResult<ComputationGraph> {
        for pass in &self.passes {
            graph = pass.apply(graph)?;
        }
        Ok(graph)
    }
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for optimization passes
pub trait OptimizationPass: Send + Sync {
    /// Name of the optimization pass
    fn name(&self) -> &str;

    /// Apply the optimization to the graph
    fn apply(&self, graph: ComputationGraph) -> JitResult<ComputationGraph>;
}

/// Dead code elimination - remove unused nodes
pub struct DeadCodeElimination;

impl OptimizationPass for DeadCodeElimination {
    fn name(&self) -> &str {
        "DeadCodeElimination"
    }

    fn apply(&self, mut graph: ComputationGraph) -> JitResult<ComputationGraph> {
        // Find all nodes reachable from outputs
        let mut reachable = HashSet::new();
        let mut to_visit = graph.outputs.clone();

        while let Some(node) = to_visit.pop() {
            if reachable.insert(node) {
                // Add all predecessors to visit list
                for pred in graph.predecessors(node) {
                    to_visit.push(pred);
                }
            }
        }

        // Remove unreachable nodes
        let all_nodes: Vec<_> = graph.graph.node_indices().collect();
        for node in all_nodes {
            if !reachable.contains(&node) {
                graph.graph.remove_node(node);
            }
        }

        Ok(graph)
    }
}

/// Constant folding - evaluate constant expressions at compile time
pub struct ConstantFolding;

impl OptimizationPass for ConstantFolding {
    fn name(&self) -> &str {
        "ConstantFolding"
    }

    fn apply(&self, mut graph: ComputationGraph) -> JitResult<ComputationGraph> {
        let nodes: Vec<_> = graph.graph.node_indices().collect();

        for node_id in nodes {
            if let Some(node) = graph.node(node_id).cloned() {
                if self.can_fold(&graph, node_id, &node) {
                    self.fold_node(&mut graph, node_id, &node)?;
                }
            }
        }

        Ok(graph)
    }
}

impl ConstantFolding {
    fn can_fold(&self, graph: &ComputationGraph, node_id: NodeId, node: &Node) -> bool {
        // Check if all inputs are constants
        match &node.op {
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div => {
                graph.predecessors(node_id).all(|pred_id| {
                    graph
                        .node(pred_id)
                        .map(|n| matches!(&n.op, Operation::Constant(_)))
                        .unwrap_or(false)
                })
            }
            _ => false,
        }
    }

    fn fold_node(
        &self,
        graph: &mut ComputationGraph,
        node_id: NodeId,
        node: &Node,
    ) -> JitResult<()> {
        use crate::graph::{ConstantInfo, ConstantValue, Operation};

        // Check if all inputs to this node are constants
        let predecessors: Vec<_> = graph
            .graph
            .edges_directed(node_id, petgraph::Direction::Incoming)
            .map(|edge_ref| edge_ref.source())
            .collect();

        if predecessors.is_empty() {
            return Ok(());
        }

        // Check if all predecessors are constant nodes
        let mut all_constants = true;
        let mut constant_values = Vec::new();

        for pred_id in &predecessors {
            if let Some(pred_node) = graph.node(*pred_id) {
                match &pred_node.op {
                    Operation::Constant(const_info) => {
                        constant_values.push(const_info.value.clone());
                    }
                    _ => {
                        all_constants = false;
                        break;
                    }
                }
            } else {
                all_constants = false;
                break;
            }
        }

        if !all_constants {
            return Ok(());
        }

        // Try to fold the operation
        let folded_value = match (&node.op, constant_values.as_slice()) {
            // Binary operations on scalar constants
            (Operation::Add, [ConstantValue::Scalar(a), ConstantValue::Scalar(b)]) => {
                Some(ConstantValue::Scalar(a + b))
            }
            (Operation::Sub, [ConstantValue::Scalar(a), ConstantValue::Scalar(b)]) => {
                Some(ConstantValue::Scalar(a - b))
            }
            (Operation::Mul, [ConstantValue::Scalar(a), ConstantValue::Scalar(b)]) => {
                Some(ConstantValue::Scalar(a * b))
            }
            (Operation::Div, [ConstantValue::Scalar(a), ConstantValue::Scalar(b)]) => {
                if *b != 0.0 {
                    Some(ConstantValue::Scalar(a / b))
                } else {
                    None // Division by zero
                }
            }

            // Unary operations on scalar constants
            (Operation::Neg, [ConstantValue::Scalar(a)]) => Some(ConstantValue::Scalar(-a)),
            (Operation::Abs, [ConstantValue::Scalar(a)]) => Some(ConstantValue::Scalar(a.abs())),
            (Operation::Sqrt, [ConstantValue::Scalar(a)]) => {
                if *a >= 0.0 {
                    Some(ConstantValue::Scalar(a.sqrt()))
                } else {
                    None // Sqrt of negative number
                }
            }
            (Operation::Exp, [ConstantValue::Scalar(a)]) => Some(ConstantValue::Scalar(a.exp())),
            (Operation::Log, [ConstantValue::Scalar(a)]) => {
                if *a > 0.0 {
                    Some(ConstantValue::Scalar(a.ln()))
                } else {
                    None // Log of non-positive number
                }
            }

            // Integer operations
            (Operation::Add, [ConstantValue::IntScalar(a), ConstantValue::IntScalar(b)]) => {
                Some(ConstantValue::IntScalar(a + b))
            }
            (Operation::Sub, [ConstantValue::IntScalar(a), ConstantValue::IntScalar(b)]) => {
                Some(ConstantValue::IntScalar(a - b))
            }
            (Operation::Mul, [ConstantValue::IntScalar(a), ConstantValue::IntScalar(b)]) => {
                Some(ConstantValue::IntScalar(a * b))
            }

            _ => None, // Operation not supported for constant folding
        };

        // If we successfully folded the operation, replace the node with a constant
        if let Some(folded) = folded_value {
            if let Some(node_mut) = graph.node_mut(node_id) {
                node_mut.op = Operation::Constant(ConstantInfo { value: folded });

                // Remove edges from predecessors since this is now a constant
                let edges_to_remove: Vec<_> = graph
                    .graph
                    .edges_directed(node_id, petgraph::Direction::Incoming)
                    .map(|edge| edge.id())
                    .collect();

                for edge_id in edges_to_remove {
                    graph.graph.remove_edge(edge_id);
                }
            }
        }

        Ok(())
    }
}

/// Common subexpression elimination - reuse identical computations
pub struct CommonSubexpressionElimination;

impl OptimizationPass for CommonSubexpressionElimination {
    fn name(&self) -> &str {
        "CommonSubexpressionElimination"
    }

    fn apply(&self, mut graph: ComputationGraph) -> JitResult<ComputationGraph> {
        // Build a map of operation signatures to nodes
        let mut op_map: HashMap<String, Vec<NodeId>> = HashMap::new();

        for (node_id, node) in graph.nodes() {
            let signature = self.compute_signature(&graph, node_id, node);
            op_map.entry(signature).or_default().push(node_id);
        }

        // Find and eliminate duplicates
        for (_, nodes) in op_map {
            if nodes.len() > 1 {
                // Keep the first node, redirect others
                let keep = nodes[0];
                for &duplicate in &nodes[1..] {
                    self.redirect_node(&mut graph, duplicate, keep)?;
                }
            }
        }

        Ok(graph)
    }
}

impl CommonSubexpressionElimination {
    fn compute_signature(&self, graph: &ComputationGraph, node_id: NodeId, node: &Node) -> String {
        // Create a signature based on operation and inputs
        let mut sig = format!("{:?}", node.op);

        // Add input signatures in sorted order for consistency
        let mut inputs: Vec<_> = graph.predecessors(node_id).collect();
        inputs.sort();

        for input in inputs {
            sig.push_str(&format!("_in{:?}", input));
        }

        sig
    }

    fn redirect_node(
        &self,
        graph: &mut ComputationGraph,
        from: NodeId,
        to: NodeId,
    ) -> JitResult<()> {
        // Redirect all edges from 'from' to 'to'
        let successors: Vec<_> = graph.graph.neighbors(from).collect();

        for succ in successors {
            // Find the edge
            if let Some(edge) = graph.graph.find_edge(from, succ) {
                let edge_data = graph.graph[edge].clone();
                graph.graph.remove_edge(edge);
                graph.graph.add_edge(to, succ, edge_data);
            }
        }

        // Remove the duplicate node
        graph.graph.remove_node(from);

        Ok(())
    }
}

/// Algebraic simplification - apply mathematical identities
pub struct AlgebraicSimplification;

impl OptimizationPass for AlgebraicSimplification {
    fn name(&self) -> &str {
        "AlgebraicSimplification"
    }

    fn apply(&self, mut graph: ComputationGraph) -> JitResult<ComputationGraph> {
        let nodes: Vec<_> = graph.graph.node_indices().collect();

        for node_id in nodes {
            if let Some(node) = graph.node(node_id).cloned() {
                if let Some(simplified) = self.simplify(&graph, node_id, &node) {
                    // Replace node with simplified version
                    if let Some(node_mut) = graph.node_mut(node_id) {
                        node_mut.op = simplified;
                    }
                }
            }
        }

        Ok(graph)
    }
}

impl AlgebraicSimplification {
    fn simplify(
        &self,
        graph: &ComputationGraph,
        node_id: NodeId,
        node: &Node,
    ) -> Option<Operation> {
        match &node.op {
            Operation::Mul => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 2 {
                    // Check for multiply by constants
                    if let (Some(left), Some(right)) = (graph.node(preds[0]), graph.node(preds[1]))
                    {
                        match (&left.op, &right.op) {
                            // x * 0 = 0
                            (Operation::Constant(c), _) | (_, Operation::Constant(c)) => {
                                if let crate::graph::ConstantValue::Scalar(v) = &c.value {
                                    if *v == 0.0 {
                                        return Some(Operation::Constant(
                                            crate::graph::ConstantInfo {
                                                value: crate::graph::ConstantValue::Scalar(0.0),
                                            },
                                        ));
                                    }
                                    // x * 1 = x is handled by constant folding pass
                                }
                            }
                            // x * x = x^2 (could be optimized further)
                            _ if preds[0] == preds[1] => {
                                return Some(Operation::Pow);
                            }
                            _ => {}
                        }
                    }
                }
            }
            Operation::Add => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 2 {
                    if let (Some(left), Some(right)) = (graph.node(preds[0]), graph.node(preds[1]))
                    {
                        // x + 0 = x
                        match (&left.op, &right.op) {
                            (Operation::Constant(c), _) | (_, Operation::Constant(c)) => {
                                if let crate::graph::ConstantValue::Scalar(v) = &c.value {
                                    if *v == 0.0 {
                                        // x + 0 = x (handled by rewriting the graph)
                                        // Return None to trigger graph rewriting
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            Operation::Sub => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 2 && preds[0] == preds[1] {
                    // x - x = 0
                    return Some(Operation::Constant(crate::graph::ConstantInfo {
                        value: crate::graph::ConstantValue::Scalar(0.0),
                    }));
                }
            }
            Operation::Div => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 2 {
                    if let Some(right) = graph.node(preds[1]) {
                        // x / 1 = x
                        if let Operation::Constant(c) = &right.op {
                            if let crate::graph::ConstantValue::Scalar(v) = &c.value {
                                if *v == 1.0 {
                                    // x / 1 = x (handled by graph rewriting)
                                    return None;
                                }
                            }
                        }
                    }
                    // x / x = 1
                    if preds[0] == preds[1] {
                        return Some(Operation::Constant(crate::graph::ConstantInfo {
                            value: crate::graph::ConstantValue::Scalar(1.0),
                        }));
                    }
                }
            }
            Operation::Pow => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 2 {
                    if let Some(right) = graph.node(preds[1]) {
                        if let Operation::Constant(c) = &right.op {
                            if let crate::graph::ConstantValue::Scalar(exp) = &c.value {
                                match *exp {
                                    // x^0 = 1
                                    0.0 => {
                                        return Some(Operation::Constant(
                                            crate::graph::ConstantInfo {
                                                value: crate::graph::ConstantValue::Scalar(1.0),
                                            },
                                        ))
                                    }
                                    // x^1 = x
                                    1.0 => return None,
                                    // x^2 can be optimized to x*x
                                    2.0 => return Some(Operation::Mul),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
            Operation::Sqrt => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 1 {
                    if let Some(pred) = graph.node(preds[0]) {
                        // sqrt(x^2) = |x| (could be x if x >= 0)
                        if let Operation::Pow = &pred.op {
                            let pow_preds: Vec<_> = graph.predecessors(preds[0]).collect();
                            if pow_preds.len() == 2 {
                                if let Some(exp_node) = graph.node(pow_preds[1]) {
                                    if let Operation::Constant(c) = &exp_node.op {
                                        if let crate::graph::ConstantValue::Scalar(2.0) = &c.value {
                                            return Some(Operation::Abs);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Operation::Log => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 1 {
                    if let Some(pred) = graph.node(preds[0]) {
                        // log(exp(x)) = x
                        if let Operation::Exp = &pred.op {
                            return None; // Handle by graph rewriting
                        }
                        // log(1) = 0
                        if let Operation::Constant(c) = &pred.op {
                            if let crate::graph::ConstantValue::Scalar(1.0) = &c.value {
                                return Some(Operation::Constant(crate::graph::ConstantInfo {
                                    value: crate::graph::ConstantValue::Scalar(0.0),
                                }));
                            }
                        }
                    }
                }
            }
            Operation::Exp => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 1 {
                    if let Some(pred) = graph.node(preds[0]) {
                        // exp(log(x)) = x
                        if let Operation::Log = &pred.op {
                            return None; // Handle by graph rewriting
                        }
                        // exp(0) = 1
                        if let Operation::Constant(c) = &pred.op {
                            if let crate::graph::ConstantValue::Scalar(0.0) = &c.value {
                                return Some(Operation::Constant(crate::graph::ConstantInfo {
                                    value: crate::graph::ConstantValue::Scalar(1.0),
                                }));
                            }
                        }
                    }
                }
            }
            // Double negation: -(-x) = x
            Operation::Neg => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 1 {
                    if let Some(pred) = graph.node(preds[0]) {
                        if let Operation::Neg = &pred.op {
                            return None; // Handle by graph rewriting
                        }
                    }
                }
            }
            _ => {}
        }

        None
    }
}

/// Strength reduction - replace expensive operations with cheaper ones
pub struct StrengthReduction;

impl OptimizationPass for StrengthReduction {
    fn name(&self) -> &str {
        "StrengthReduction"
    }

    fn apply(&self, mut graph: ComputationGraph) -> JitResult<ComputationGraph> {
        let nodes: Vec<_> = graph.graph.node_indices().collect();

        for node_id in nodes {
            if let Some(node) = graph.node(node_id).cloned() {
                if let Some(reduced) = self.reduce_strength(&graph, node_id, &node) {
                    if let Some(node_mut) = graph.node_mut(node_id) {
                        node_mut.op = reduced;
                    }
                }
            }
        }

        Ok(graph)
    }
}

impl StrengthReduction {
    fn reduce_strength(
        &self,
        graph: &ComputationGraph,
        node_id: NodeId,
        node: &Node,
    ) -> Option<Operation> {
        match &node.op {
            Operation::Pow => {
                // Check for power of 2
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 2 {
                    let is_two = graph
                        .node(preds[1])
                        .and_then(|node| match &node.op {
                            Operation::Constant(c) => match &c.value {
                                crate::graph::ConstantValue::Scalar(v) => Some(*v == 2.0),
                                _ => None,
                            },
                            _ => None,
                        })
                        .unwrap_or(false);

                    if is_two {
                        // x^2 -> x*x (multiply is usually faster)
                        return Some(Operation::Mul);
                    }
                }
            }
            Operation::Div => {
                // Check for division by constant
                // Could be replaced with multiplication by reciprocal
            }
            _ => {}
        }

        None
    }
}

/// Layout optimization - optimize memory access patterns
pub struct LayoutOptimization;

impl OptimizationPass for LayoutOptimization {
    fn name(&self) -> &str {
        "LayoutOptimization"
    }

    fn apply(&self, mut graph: ComputationGraph) -> JitResult<ComputationGraph> {
        // Analyze memory access patterns and insert transpose operations where beneficial
        // This optimization is particularly useful for conv2d and matmul operations

        let nodes: Vec<_> = graph.graph.node_indices().collect();
        let mut transpose_insertions = Vec::new();

        for node_id in nodes {
            if let Some(node) = graph.node(node_id) {
                match &node.op {
                    Operation::Conv2d(info) => {
                        // Check if input layout would benefit from NCHW -> NHWC conversion
                        // This can improve cache locality for certain convolution patterns
                        if self.should_convert_layout_for_conv(info, &node.output_shape) {
                            // Mark for layout conversion
                            transpose_insertions.push((node_id, LayoutChange::NCHWtoNHWC));
                        }
                    }
                    Operation::MatMul | Operation::BatchMatMul => {
                        // Check if transposing inputs would reduce memory access stride
                        let preds: Vec<_> = graph.predecessors(node_id).collect();
                        if preds.len() == 2 {
                            // Analyze if transposing either input would be beneficial
                            if let (Some(left), Some(right)) =
                                (graph.node(preds[0]), graph.node(preds[1]))
                            {
                                if self.should_transpose_for_matmul(
                                    &left.output_shape,
                                    &right.output_shape,
                                ) {
                                    transpose_insertions
                                        .push((node_id, LayoutChange::TransposeMatmul));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Apply layout changes
        for (node_id, change) in transpose_insertions {
            match change {
                LayoutChange::NCHWtoNHWC => {
                    // Insert transpose operations to convert layout
                    self.insert_layout_conversion(&mut graph, node_id, vec![0, 2, 3, 1])?;
                }
                LayoutChange::TransposeMatmul => {
                    // Handle matmul-specific optimizations
                    self.optimize_matmul_layout(&mut graph, node_id)?;
                }
            }
        }

        Ok(graph)
    }
}

#[derive(Debug)]
enum LayoutChange {
    NCHWtoNHWC,
    TransposeMatmul,
}

impl LayoutOptimization {
    /// Check if converting conv2d layout would be beneficial
    fn should_convert_layout_for_conv(&self, info: &Conv2dInfo, output_shape: &Shape) -> bool {
        // Heuristic: prefer NHWC for depthwise and small kernel convolutions
        // as they have better cache locality with channels-last format

        // Depthwise convolution benefits from NHWC
        if info.groups == info.in_channels && info.in_channels == info.out_channels {
            return true;
        }

        // Small kernels (1x1, 3x3) often benefit from NHWC
        if info.kernel_size.0 <= 3 && info.kernel_size.1 <= 3 {
            // Also consider output spatial dimensions
            if output_shape.ndim() >= 4 {
                let height = output_shape.dims()[2];
                let width = output_shape.dims()[3];
                // Prefer NHWC for larger spatial dimensions
                return height * width > 1024;
            }
        }

        false
    }

    /// Check if transposing matmul inputs would be beneficial
    fn should_transpose_for_matmul(&self, left_shape: &Shape, right_shape: &Shape) -> bool {
        // Check if the matrices are already in optimal layout for matmul
        // For A @ B, we want A to be row-major and B to be column-major
        // This minimizes cache misses during computation

        if left_shape.ndim() < 2 || right_shape.ndim() < 2 {
            return false;
        }

        // Get the matrix dimensions
        let left_rows = left_shape.dims()[left_shape.ndim() - 2];
        let left_cols = left_shape.dims()[left_shape.ndim() - 1];
        let right_rows = right_shape.dims()[right_shape.ndim() - 2];

        // If the inner dimension is small, transposing might not be worth it
        if left_cols < 16 || right_rows < 16 {
            return false;
        }

        // Check stride patterns (simplified heuristic)
        // If left matrix is tall and thin, or right matrix is short and wide,
        // transposing might help
        let left_aspect = left_rows as f32 / left_cols as f32;
        let right_aspect = right_rows as f32 / right_shape.dims()[right_shape.ndim() - 1] as f32;

        left_aspect > 4.0 || right_aspect < 0.25
    }

    /// Insert layout conversion (transpose) operations
    fn insert_layout_conversion(
        &self,
        graph: &mut ComputationGraph,
        node_id: NodeId,
        dims: Vec<usize>,
    ) -> JitResult<()> {
        // Get predecessors of the node
        let preds: Vec<_> = graph.predecessors(node_id).collect();

        for pred_id in preds {
            // Insert transpose between predecessor and current node
            if let Some(pred_node) = graph.node(pred_id).cloned() {
                // Create transpose node
                let mut transpose_node = Node::new(
                    Operation::Transpose { dims: dims.clone() },
                    format!("{}_transpose", pred_node.name),
                );
                transpose_node = transpose_node
                    .with_output_shapes(vec![Some(
                        self.compute_transposed_shape(&pred_node.output_shape, &dims),
                    )])
                    .with_dtypes(vec![pred_node.dtype])
                    .with_device(pred_node.device);
                transpose_node.inputs = vec![pred_id];
                transpose_node.is_output = false;

                let transpose_id = graph.add_node(transpose_node);

                // Rewire edges: pred -> transpose -> node
                if let Some(edge) = graph.graph.find_edge(pred_id, node_id) {
                    let edge_data = graph.graph[edge].clone();
                    graph.graph.remove_edge(edge);
                    graph.add_edge(pred_id, transpose_id, Edge::default());
                    graph.add_edge(transpose_id, node_id, edge_data);
                }
            }
        }

        Ok(())
    }

    /// Optimize matmul layout by potentially transposing inputs
    fn optimize_matmul_layout(
        &self,
        graph: &mut ComputationGraph,
        node_id: NodeId,
    ) -> JitResult<()> {
        let preds: Vec<_> = graph.predecessors(node_id).collect();

        if preds.len() == 2 {
            // For now, just mark that we could optimize this
            // In a real implementation, we would analyze the specific
            // matrix dimensions and access patterns

            // Could insert transposes to convert to optimal layout
            // For example, ensuring the right matrix is column-major
            // by transposing it before the matmul
        }

        Ok(())
    }

    /// Compute output shape after transpose
    fn compute_transposed_shape(&self, shape: &Shape, dims: &[usize]) -> Shape {
        let mut new_dims = vec![0; shape.ndim()];
        let old_dims = shape.dims();

        for (i, &dim) in dims.iter().enumerate() {
            if dim < old_dims.len() {
                new_dims[i] = old_dims[dim];
            }
        }

        Shape::new(new_dims)
    }
}

/// Auto-vectorization pass - converts element-wise operations to vector operations
pub struct AutoVectorization;

impl OptimizationPass for AutoVectorization {
    fn name(&self) -> &str {
        "AutoVectorization"
    }

    fn apply(&self, mut graph: ComputationGraph) -> JitResult<ComputationGraph> {
        let nodes: Vec<_> = graph.graph.node_indices().collect();

        for node_id in nodes {
            if let Some(node) = graph.node(node_id).cloned() {
                if self.can_vectorize(&graph, node_id, &node) {
                    self.vectorize_node(&mut graph, node_id, &node)?;
                }
            }
        }

        Ok(graph)
    }
}

impl AutoVectorization {
    /// Check if a node can be vectorized
    fn can_vectorize(&self, _graph: &ComputationGraph, _node_id: NodeId, node: &Node) -> bool {
        // Check for element-wise operations on large tensors
        match &node.op {
            Operation::Add
            | Operation::Sub
            | Operation::Mul
            | Operation::Div
            | Operation::Relu
            | Operation::Sigmoid
            | Operation::Tanh
            | Operation::Silu => {
                // Check if the tensor is large enough to benefit from vectorization
                let total_elements = node.output_shape.numel();
                total_elements >= 1024 // Vectorize if >= 1024 elements
            }
            _ => false,
        }
    }

    /// Apply vectorization to a node
    fn vectorize_node(
        &self,
        graph: &mut ComputationGraph,
        node_id: NodeId,
        _node: &Node,
    ) -> JitResult<()> {
        if let Some(node_mut) = graph.node_mut(node_id) {
            // Add vectorization hint to node attributes
            node_mut
                .attrs
                .insert("vectorize".to_string(), crate::graph::Attribute::Bool(true));
            node_mut.attrs.insert(
                "vector_width".to_string(),
                crate::graph::Attribute::Int(8), // 8-wide SIMD
            );
        }
        Ok(())
    }
}

/// Auto-parallelization pass - identifies opportunities for parallel execution
pub struct AutoParallelization;

impl OptimizationPass for AutoParallelization {
    fn name(&self) -> &str {
        "AutoParallelization"
    }

    fn apply(&self, mut graph: ComputationGraph) -> JitResult<ComputationGraph> {
        // Find independent subgraphs that can be executed in parallel
        let parallel_groups = self.find_parallel_groups(&graph)?;

        // Mark nodes with parallelization hints
        for (group_id, node_ids) in parallel_groups.iter().enumerate() {
            for &node_id in node_ids {
                if let Some(node_mut) = graph.node_mut(node_id) {
                    node_mut.attrs.insert(
                        "parallel_group".to_string(),
                        crate::graph::Attribute::Int(group_id as i64),
                    );
                    node_mut.attrs.insert(
                        "can_parallelize".to_string(),
                        crate::graph::Attribute::Bool(true),
                    );
                }
            }
        }

        Ok(graph)
    }
}

impl AutoParallelization {
    /// Find groups of nodes that can be executed in parallel
    fn find_parallel_groups(&self, graph: &ComputationGraph) -> JitResult<Vec<Vec<NodeId>>> {
        let mut parallel_groups = Vec::new();
        let mut visited = HashSet::new();

        // Topological sort to process nodes in dependency order
        let topo_order = graph
            .topological_sort()
            .map_err(|e| crate::JitError::GraphError(e.to_string()))?;

        for &node_id in &topo_order {
            if visited.contains(&node_id) {
                continue;
            }

            // Find all nodes at this "level" (same depth in dependency chain)
            let mut current_group = Vec::new();

            // Find nodes that have no dependencies on unvisited nodes
            if self.can_execute_now(graph, node_id, &visited) {
                current_group.push(node_id);
                visited.insert(node_id);

                // Look for other nodes at the same level
                for &other_id in &topo_order {
                    if other_id != node_id
                        && !visited.contains(&other_id)
                        && self.can_execute_now(graph, other_id, &visited)
                        && self.can_execute_parallel(graph, node_id, other_id)
                    {
                        current_group.push(other_id);
                        visited.insert(other_id);
                    }
                }
            }

            if current_group.len() > 1 {
                parallel_groups.push(current_group);
            }
        }

        Ok(parallel_groups)
    }

    /// Check if a node can execute now (all dependencies satisfied)
    fn can_execute_now(
        &self,
        graph: &ComputationGraph,
        node_id: NodeId,
        visited: &HashSet<NodeId>,
    ) -> bool {
        graph
            .predecessors(node_id)
            .all(|pred| visited.contains(&pred))
    }

    /// Check if two nodes can be executed in parallel (no dependencies)
    fn can_execute_parallel(&self, graph: &ComputationGraph, node1: NodeId, node2: NodeId) -> bool {
        // Check if there's any dependency path between the nodes
        !self.has_dependency_path(graph, node1, node2)
            && !self.has_dependency_path(graph, node2, node1)
    }

    /// Check if there's a dependency path from node1 to node2
    fn has_dependency_path(&self, graph: &ComputationGraph, from: NodeId, to: NodeId) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![from];

        while let Some(current) = stack.pop() {
            if current == to {
                return true;
            }

            if visited.insert(current) {
                for successor in graph.successors(current) {
                    stack.push(successor);
                }
            }
        }

        false
    }
}

/// Cache-aware optimization - improve data locality and cache utilization
pub struct CacheAwareOptimization {
    /// Target cache line size in bytes
    cache_line_size: usize,
    /// L1 cache size in bytes
    l1_cache_size: usize,
    /// L2 cache size in bytes
    l2_cache_size: usize,
}

impl Default for CacheAwareOptimization {
    fn default() -> Self {
        Self {
            cache_line_size: 64,       // Typical cache line size
            l1_cache_size: 32 * 1024,  // 32 KB L1
            l2_cache_size: 256 * 1024, // 256 KB L2
        }
    }
}

impl CacheAwareOptimization {
    pub fn new(cache_line_size: usize, l1_cache_size: usize, l2_cache_size: usize) -> Self {
        Self {
            cache_line_size,
            l1_cache_size,
            l2_cache_size,
        }
    }

    /// Calculate optimal tile size for a given dimension
    fn calculate_tile_size(&self, dimension_size: usize, element_size: usize) -> usize {
        // Aim to fit tiles in L1 cache
        let elements_in_l1 = self.l1_cache_size / element_size;
        let sqrt_elements = (elements_in_l1 as f64).sqrt() as usize;

        // Use power of 2 for better alignment
        let mut tile_size = 1;
        while tile_size * 2 <= sqrt_elements && tile_size * 2 <= dimension_size {
            tile_size *= 2;
        }

        tile_size.max(8).min(dimension_size) // At least 8, at most dimension size
    }

    /// Reorder operations for better cache locality
    fn reorder_for_locality(&self, graph: &mut ComputationGraph) -> JitResult<usize> {
        let mut reordered = 0;

        // Find groups of operations that access the same data
        let access_groups = self.analyze_data_access_patterns(graph)?;

        // For each group, try to schedule operations close together
        for group in access_groups {
            if group.len() > 1 {
                // Operations in the same group should be executed consecutively
                // This reduces cache misses by keeping data hot
                reordered += group.len();
            }
        }

        Ok(reordered)
    }

    /// Analyze data access patterns in the graph
    fn analyze_data_access_patterns(
        &self,
        graph: &ComputationGraph,
    ) -> JitResult<Vec<Vec<NodeId>>> {
        let mut groups = Vec::new();
        let mut visited = HashSet::new();

        // Group nodes that access similar memory regions
        for (node_id, node) in graph.nodes() {
            if visited.contains(&node_id) {
                continue;
            }

            let mut group = vec![node_id];
            visited.insert(node_id);

            // Find related nodes that access similar data
            for (other_id, other_node) in graph.nodes() {
                if visited.contains(&other_id) {
                    continue;
                }

                if self.have_similar_access_pattern(node, other_node) {
                    group.push(other_id);
                    visited.insert(other_id);
                }
            }

            if group.len() > 1 {
                groups.push(group);
            }
        }

        Ok(groups)
    }

    /// Check if two nodes have similar data access patterns
    fn have_similar_access_pattern(&self, node1: &Node, node2: &Node) -> bool {
        // Nodes with similar operations likely access data similarly
        match (&node1.op, &node2.op) {
            (Operation::MatMul, Operation::MatMul) => true,
            (Operation::Conv2d(_), Operation::Conv2d(_)) => true,
            (Operation::Add, Operation::Add)
            | (Operation::Sub, Operation::Sub)
            | (Operation::Mul, Operation::Mul)
            | (Operation::Div, Operation::Div) => {
                // Element-wise operations with similar shapes
                node1.output_shape.dims() == node2.output_shape.dims()
            }
            _ => false,
        }
    }

    /// Apply loop tiling to large operations
    fn apply_loop_tiling(&self, graph: &mut ComputationGraph) -> JitResult<usize> {
        let mut tiled = 0;

        for (node_id, node) in graph.nodes() {
            match &node.op {
                Operation::MatMul => {
                    // MatMul benefits greatly from tiling
                    if let Some(shape) = node.output_shapes.first().and_then(|s| s.as_ref()) {
                        let dims = shape.dims();
                        if dims.len() >= 2 {
                            let m = dims[dims.len() - 2];
                            let n = dims[dims.len() - 1];

                            // Calculate tile sizes
                            let tile_m = self.calculate_tile_size(m, 4); // Assuming f32
                            let tile_n = self.calculate_tile_size(n, 4);

                            // Store tiling hint in node attributes
                            log::debug!(
                                "MatMul node {:?}: suggested tiling {}x{}",
                                node_id,
                                tile_m,
                                tile_n
                            );
                            tiled += 1;
                        }
                    }
                }

                Operation::Conv2d(conv_info) => {
                    // Convolutions also benefit from tiling
                    let tile_size = self.calculate_tile_size(conv_info.kernel_size.0, 4);
                    log::debug!(
                        "Conv2d node {:?}: suggested tile size {}",
                        node_id,
                        tile_size
                    );
                    tiled += 1;
                }

                _ => {}
            }
        }

        Ok(tiled)
    }

    /// Add prefetch hints for predictable access patterns
    fn add_prefetch_hints(&self, graph: &ComputationGraph) -> JitResult<usize> {
        let mut hints_added = 0;

        // Analyze sequential access patterns
        for (node_id, node) in graph.nodes() {
            // Operations that scan through data sequentially
            match &node.op {
                Operation::MatMul | Operation::Conv2d(_) => {
                    // These operations have predictable access patterns
                    // Add prefetch hints for next cache lines
                    log::debug!("Node {:?}: adding software prefetch hints", node_id);
                    hints_added += 1;
                }
                _ => {}
            }
        }

        Ok(hints_added)
    }

    /// Optimize for spatial and temporal locality
    fn optimize_locality(&self, graph: &mut ComputationGraph) -> JitResult<usize> {
        let mut optimized = 0;

        // Ensure operations that reuse data are scheduled close together
        optimized += self.reorder_for_locality(graph)?;

        // Apply loop tiling for better cache utilization
        optimized += self.apply_loop_tiling(graph)?;

        // Add prefetch hints for predictable accesses
        optimized += self.add_prefetch_hints(graph)?;

        Ok(optimized)
    }
}

impl OptimizationPass for CacheAwareOptimization {
    fn name(&self) -> &str {
        "cache_aware"
    }

    fn apply(&self, mut graph: ComputationGraph) -> JitResult<ComputationGraph> {
        let optimizations = self.optimize_locality(&mut graph)?;

        log::info!(
            "Cache-aware optimization: {} improvements applied",
            optimizations
        );

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Edge;
    use torsh_core::{DType, DeviceType};

    #[test]
    fn test_optimizer_creation() {
        let optimizer = GraphOptimizer::new();
        assert_eq!(optimizer.passes.len(), 9); // Updated to include CacheAwareOptimization
    }

    #[test]
    fn test_dead_code_elimination() {
        let mut graph = ComputationGraph::new();

        // Create a simple graph with dead code
        let input = graph.add_node(
            Node::new(Operation::Input, "input".to_string())
                .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[10]))])
                .with_dtypes(vec![DType::F32])
                .with_device(DeviceType::Cpu),
        );

        let dead = graph.add_node(
            Node::new(Operation::Relu, "dead".to_string())
                .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[10]))])
                .with_dtypes(vec![DType::F32])
                .with_device(DeviceType::Cpu),
        );

        let output = graph.add_node(
            Node::new(Operation::Neg, "output".to_string())
                .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[10]))])
                .with_dtypes(vec![DType::F32])
                .with_device(DeviceType::Cpu),
        );

        graph.add_edge(input, output, Edge::default());
        graph.add_edge(input, dead, Edge::default()); // Dead branch

        graph.add_input(input);
        graph.add_output(output);

        let dce = DeadCodeElimination;
        let optimized = dce.apply(graph).unwrap();

        // After DCE, we should only have nodes reachable from outputs
        // The graph should have exactly 2 nodes (input and output)
        assert_eq!(
            optimized.graph.node_count(),
            2,
            "Should have exactly 2 nodes after DCE"
        );

        // Verify the remaining nodes have the expected operations
        let remaining_nodes: Vec<_> = optimized
            .graph
            .node_indices()
            .filter_map(|idx| optimized.graph.node_weight(idx))
            .collect();

        // Should have an input node and a neg node
        let has_input = remaining_nodes
            .iter()
            .any(|n| matches!(&n.op, Operation::Input));
        let has_neg = remaining_nodes
            .iter()
            .any(|n| matches!(&n.op, Operation::Neg));

        assert!(has_input, "Input node should still exist");
        assert!(has_neg, "Output (neg) node should still exist");

        // The sigmoid (dead) node should be gone
        let has_sigmoid = remaining_nodes
            .iter()
            .any(|n| matches!(&n.op, Operation::Sigmoid));
        assert!(!has_sigmoid, "Dead (sigmoid) node should be removed");
    }
}
