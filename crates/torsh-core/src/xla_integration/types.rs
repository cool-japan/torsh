//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::device::DeviceType;
use crate::dtype::DType;
use crate::error::{Result, TorshError};
use crate::shape::Shape;

use super::functions::XlaPass;

/// Dead code elimination pass
///
/// Removes operations that do not contribute to the final result.
pub struct DeadCodeEliminationPass;
impl DeadCodeEliminationPass {
    /// Mark all nodes reachable from root
    pub(super) fn mark_reachable(
        computation: &XlaComputation,
        root: XlaNodeId,
        reachable: &mut Vec<bool>,
    ) {
        if reachable[root.0] {
            return;
        }
        reachable[root.0] = true;
        if let Some(node) = computation.nodes.get(root.0) {
            for &operand in &node.operands {
                Self::mark_reachable(computation, operand, reachable);
            }
        }
    }
}
/// Metadata for XLA operations
#[derive(Debug, Clone, Default)]
pub struct XlaMetadata {
    /// Dimensions to reduce over (for Reduce operation)
    pub reduce_dims: Vec<usize>,
    /// Permutation for transpose
    pub transpose_perm: Vec<usize>,
    /// Broadcast dimensions
    pub broadcast_dims: Vec<usize>,
    /// Concatenation dimension
    pub concat_dim: Option<usize>,
    /// Slice start indices
    pub slice_start: Vec<usize>,
    /// Slice end indices
    pub slice_end: Vec<usize>,
    /// Slice strides
    pub slice_strides: Vec<usize>,
    /// Custom call target name
    pub custom_call_target: Option<String>,
}
/// Algebraic simplification pass
///
/// Applies algebraic identities to simplify expressions.
/// Examples: x + 0 = x, x * 1 = x, x * 0 = 0
pub struct AlgebraicSimplificationPass;
impl AlgebraicSimplificationPass {
    /// Check if a node is a constant zero
    fn is_constant_zero(_node: &XlaNode) -> bool {
        false
    }
    /// Check if a node is a constant one
    fn is_constant_one(_node: &XlaNode) -> bool {
        false
    }
    /// Find simplification opportunities
    pub(super) fn find_simplifications(computation: &XlaComputation) -> usize {
        let mut count = 0;
        for node in &computation.nodes {
            match node.opcode {
                HloOpcode::Add => {
                    if node.operands.len() == 2 {
                        let op1 = computation.nodes.get(node.operands[0].0);
                        let op2 = computation.nodes.get(node.operands[1].0);
                        if let (Some(n1), Some(n2)) = (op1, op2) {
                            if Self::is_constant_zero(n1) || Self::is_constant_zero(n2) {
                                count += 1;
                            }
                        }
                    }
                }
                HloOpcode::Multiply => {
                    if node.operands.len() == 2 {
                        let op1 = computation.nodes.get(node.operands[0].0);
                        let op2 = computation.nodes.get(node.operands[1].0);
                        if let (Some(n1), Some(n2)) = (op1, op2) {
                            if Self::is_constant_one(n1) || Self::is_constant_one(n2) {
                                count += 1;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        count
    }
}
/// XLA compiler configuration
#[derive(Debug, Clone)]
pub struct XlaConfig {
    /// Target device for compilation
    pub target: XlaTarget,
    /// Enable aggressive fusion
    pub enable_fusion: bool,
    /// Enable layout optimization
    pub enable_layout_optimization: bool,
    /// Enable algebraic simplification
    pub enable_algebraic_simplification: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
}
impl XlaConfig {
    /// Create a new configuration
    pub fn new(target: XlaTarget) -> Self {
        Self {
            target,
            ..Default::default()
        }
    }
    /// Set fusion enabled
    pub fn with_fusion(mut self, enable: bool) -> Self {
        self.enable_fusion = enable;
        self
    }
    /// Set layout optimization enabled
    pub fn with_layout_optimization(mut self, enable: bool) -> Self {
        self.enable_layout_optimization = enable;
        self
    }
    /// Set algebraic simplification enabled
    pub fn with_algebraic_simplification(mut self, enable: bool) -> Self {
        self.enable_algebraic_simplification = enable;
        self
    }
    /// Set optimization level
    pub fn with_optimization_level(mut self, level: u8) -> Self {
        self.optimization_level = level.min(3);
        self
    }
    /// Create aggressive optimization configuration
    pub fn aggressive() -> Self {
        Self {
            target: XlaTarget::Cpu,
            enable_fusion: true,
            enable_layout_optimization: true,
            enable_algebraic_simplification: true,
            optimization_level: 3,
        }
    }
    /// Create conservative optimization configuration
    pub fn conservative() -> Self {
        Self {
            target: XlaTarget::Cpu,
            enable_fusion: false,
            enable_layout_optimization: false,
            enable_algebraic_simplification: false,
            optimization_level: 0,
        }
    }
}
/// Compiled XLA computation
///
/// Represents a complete XLA computation graph ready for execution.
#[derive(Debug, Clone)]
pub struct XlaComputation {
    /// Computation name
    pub name: String,
    /// All nodes in topological order
    pub nodes: Vec<XlaNode>,
    /// Root node (output)
    pub root: XlaNodeId,
    /// Configuration for compilation and optimization
    pub config: XlaConfig,
}
impl XlaComputation {
    /// Get computation name
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Get all nodes
    pub fn nodes(&self) -> &[XlaNode] {
        &self.nodes
    }
    /// Get root node ID
    pub fn root_id(&self) -> XlaNodeId {
        self.root
    }
    /// Get root node
    pub fn root_node(&self) -> Option<&XlaNode> {
        self.nodes.iter().find(|n| n.id == self.root)
    }
    /// Get output shape
    pub fn output_shape(&self) -> Option<&Shape> {
        self.root_node().map(|n| &n.shape)
    }
    /// Get output data type
    pub fn output_dtype(&self) -> Option<DType> {
        self.root_node().map(|n| n.dtype)
    }
    /// Get total number of nodes in computation
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
    /// Count operations by type
    pub fn operation_counts(&self) -> Vec<(HloOpcode, usize)> {
        let mut counts: Vec<(HloOpcode, usize)> = Vec::new();
        let mut opcodes = Vec::new();
        for node in &self.nodes {
            if let Some(pos) = opcodes.iter().position(|&op| op == node.opcode) {
                counts[pos].1 += 1;
            } else {
                opcodes.push(node.opcode);
                counts.push((node.opcode, 1));
            }
        }
        counts
    }
    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.opcode == HloOpcode::Parameter)
            .count()
    }
    /// Validate computation graph
    pub fn validate(&self) -> Result<()> {
        if !self.nodes.iter().any(|n| n.id == self.root) {
            return Err(TorshError::dimension_error(
                "Root node not found in computation",
                "validate",
            ));
        }
        for node in &self.nodes {
            for &operand_id in &node.operands {
                if !self.nodes.iter().any(|n| n.id == operand_id) {
                    return Err(TorshError::dimension_error(
                        &format!("Operand {:?} not found for node {:?}", operand_id, node.id),
                        "validate",
                    ));
                }
            }
        }
        Ok(())
    }
    /// Generate HLO text representation (simplified)
    pub fn to_hlo_text(&self) -> String {
        let mut text = format!("HloModule {}\n\n", self.name);
        text.push_str("ENTRY main {\n");
        for node in &self.nodes {
            let operand_refs: Vec<String> = node
                .operands
                .iter()
                .map(|id| format!("%{}", id.0))
                .collect();
            let line = format!(
                "  %{} = {} {}({})\n",
                node.id.0,
                node.dtype.name(),
                node.opcode.name(),
                operand_refs.join(", ")
            );
            text.push_str(&line);
        }
        text.push_str(&format!("  ROOT %{}\n", self.root.0));
        text.push_str("}\n");
        text
    }
}
impl XlaComputation {
    /// Run optimization passes on this computation
    pub fn optimize(&mut self) -> Result<PassStatistics> {
        let manager = XlaPassManager::new();
        manager.run(self)
    }
    /// Run optimization passes with custom pass manager
    pub fn optimize_with(&mut self, manager: &XlaPassManager) -> Result<PassStatistics> {
        manager.run(self)
    }
    /// Run a single optimization pass
    pub fn run_pass(&mut self, pass: &dyn XlaPass) -> Result<PassStatistics> {
        pass.run(self)
    }
}
/// Operation fusion pass
///
/// Combines multiple operations into fused kernels for better performance.
pub struct OperationFusionPass;
impl OperationFusionPass {
    /// Check if two operations can be fused
    fn can_fuse(op1: &XlaNode, op2: &XlaNode) -> bool {
        op1.opcode.is_elementwise()
            && op2.opcode.is_elementwise()
            && op1.shape == op2.shape
            && op1.dtype == op2.dtype
    }
    /// Find fusion opportunities
    pub(super) fn find_fusion_candidates(
        computation: &XlaComputation,
    ) -> Vec<(XlaNodeId, XlaNodeId)> {
        let mut candidates = Vec::new();
        for i in 0..computation.nodes.len() {
            for j in (i + 1)..computation.nodes.len() {
                let node1 = &computation.nodes[i];
                let node2 = &computation.nodes[j];
                if node2.operands.contains(&XlaNodeId(i)) && Self::can_fuse(node1, node2) {
                    candidates.push((XlaNodeId(i), XlaNodeId(j)));
                }
            }
        }
        candidates
    }
}
/// Copy Elimination Pass
///
/// Eliminates unnecessary copy operations in the computation graph.
/// This pass identifies and removes redundant data copies that don't serve
/// any semantic purpose, reducing memory bandwidth usage and improving performance.
///
/// # Optimizations
/// - Remove identity copies (copy that doesn't change data)
/// - Eliminate copies to/from same memory location
/// - Merge multiple copies into a single operation
/// - Remove copies that are immediately overwritten
///
/// # Example
/// ```rust,ignore
/// // Before: Unnecessary copy operations
/// let temp = Copy(A)
/// let result = Add(temp, B)
///
/// // After: Direct use without copy
/// let result = Add(A, B)
/// ```
pub struct CopyEliminationPass;
impl CopyEliminationPass {
    /// Count copy operations that could potentially be eliminated
    pub(super) fn count_eliminable_copies(computation: &XlaComputation) -> usize {
        computation
            .nodes
            .iter()
            .filter(|node| node.opcode == HloOpcode::Copy)
            .count()
    }
}
/// Node identifier in XLA computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct XlaNodeId(pub usize);
/// HLO (High-Level Optimizer) Operation Codes
///
/// These represent the fundamental operations in XLA's intermediate representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HloOpcode {
    /// Parameter input to computation
    Parameter,
    /// Constant value
    Constant,
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Subtract,
    /// Element-wise multiplication
    Multiply,
    /// Element-wise division
    Divide,
    /// Matrix multiplication (dot product)
    Dot,
    /// Convolution operation
    Convolution,
    /// Reduce operation (sum, max, min, etc.)
    Reduce,
    /// Reshape tensor
    Reshape,
    /// Transpose dimensions
    Transpose,
    /// Broadcast to larger shape
    Broadcast,
    /// Concatenate tensors
    Concatenate,
    /// Slice tensor
    Slice,
    /// Gather elements
    Gather,
    /// Scatter elements
    Scatter,
    /// Select elements (ternary conditional)
    Select,
    /// Tuple of multiple values
    Tuple,
    /// Get element from tuple
    GetTupleElement,
    /// Map function over elements
    Map,
    /// Element-wise maximum
    Maximum,
    /// Element-wise minimum
    Minimum,
    /// Element-wise power
    Power,
    /// Element-wise absolute value
    Abs,
    /// Element-wise exponential
    Exp,
    /// Element-wise natural logarithm
    Log,
    /// Element-wise square root
    Sqrt,
    /// Element-wise trigonometric sine
    Sin,
    /// Element-wise trigonometric cosine
    Cos,
    /// Element-wise hyperbolic tangent
    Tanh,
    /// Pad tensor with values
    Pad,
    /// Reverse dimensions
    Reverse,
    /// Sort elements
    Sort,
    /// Convert between data types
    Convert,
    /// Bitcast reinterpret
    BitcastConvert,
    /// Copy operation
    Copy,
    /// While loop
    While,
    /// Conditional operation
    Conditional,
    /// Custom call to external function
    CustomCall,
}
impl HloOpcode {
    /// Get human-readable name for opcode
    pub fn name(&self) -> &'static str {
        match self {
            HloOpcode::Parameter => "parameter",
            HloOpcode::Constant => "constant",
            HloOpcode::Add => "add",
            HloOpcode::Subtract => "subtract",
            HloOpcode::Multiply => "multiply",
            HloOpcode::Divide => "divide",
            HloOpcode::Dot => "dot",
            HloOpcode::Convolution => "convolution",
            HloOpcode::Reduce => "reduce",
            HloOpcode::Reshape => "reshape",
            HloOpcode::Transpose => "transpose",
            HloOpcode::Broadcast => "broadcast",
            HloOpcode::Concatenate => "concatenate",
            HloOpcode::Slice => "slice",
            HloOpcode::Gather => "gather",
            HloOpcode::Scatter => "scatter",
            HloOpcode::Select => "select",
            HloOpcode::Tuple => "tuple",
            HloOpcode::GetTupleElement => "get-tuple-element",
            HloOpcode::Map => "map",
            HloOpcode::Maximum => "maximum",
            HloOpcode::Minimum => "minimum",
            HloOpcode::Power => "power",
            HloOpcode::Abs => "abs",
            HloOpcode::Exp => "exp",
            HloOpcode::Log => "log",
            HloOpcode::Sqrt => "sqrt",
            HloOpcode::Sin => "sin",
            HloOpcode::Cos => "cos",
            HloOpcode::Tanh => "tanh",
            HloOpcode::Pad => "pad",
            HloOpcode::Reverse => "reverse",
            HloOpcode::Sort => "sort",
            HloOpcode::Convert => "convert",
            HloOpcode::BitcastConvert => "bitcast-convert",
            HloOpcode::Copy => "copy",
            HloOpcode::While => "while",
            HloOpcode::Conditional => "conditional",
            HloOpcode::CustomCall => "custom-call",
        }
    }
    /// Check if operation is element-wise
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            HloOpcode::Add
                | HloOpcode::Subtract
                | HloOpcode::Multiply
                | HloOpcode::Divide
                | HloOpcode::Maximum
                | HloOpcode::Minimum
                | HloOpcode::Power
                | HloOpcode::Abs
                | HloOpcode::Exp
                | HloOpcode::Log
                | HloOpcode::Sqrt
                | HloOpcode::Sin
                | HloOpcode::Cos
                | HloOpcode::Tanh
        )
    }
    /// Check if operation is a reduction
    pub fn is_reduction(&self) -> bool {
        matches!(self, HloOpcode::Reduce)
    }
    /// Check if operation is shape-changing
    pub fn is_shape_changing(&self) -> bool {
        matches!(
            self,
            HloOpcode::Reshape
                | HloOpcode::Transpose
                | HloOpcode::Broadcast
                | HloOpcode::Concatenate
                | HloOpcode::Slice
                | HloOpcode::Pad
        )
    }
}
/// Parallelization Analysis Pass
///
/// Analyzes the computation graph to identify opportunities for parallel execution.
/// This pass identifies independent operations that can be executed concurrently,
/// enabling better utilization of multi-core CPUs and multi-GPU systems.
///
/// # Parallelization Strategies
/// - **Data Parallelism**: Split operations across batch dimension
/// - **Model Parallelism**: Split large operations across multiple devices
/// - **Pipeline Parallelism**: Overlap execution of independent operations
/// - **Instruction-Level Parallelism**: Identify independent operations in the graph
///
/// # Analysis
/// 1. Build dependency graph showing data flow between operations
/// 2. Identify independent operation groups (no data dependencies)
/// 3. Estimate execution cost and determine parallelization benefit
/// 4. Mark operations that should execute in parallel
///
/// # Example
/// ```rust,ignore
/// // Sequential execution
/// let a = Op1(input)
/// let b = Op2(input)  // Independent from Op1
/// let c = Op3(a, b)
///
/// // After parallelization analysis
/// parallel {
///     let a = Op1(input)  // Execute in parallel
///     let b = Op2(input)  // Execute in parallel
/// }
/// let c = Op3(a, b)  // Wait for both to complete
/// ```
pub struct ParallelizationAnalysisPass;
impl ParallelizationAnalysisPass {
    /// Count groups of independent operations that can execute in parallel
    pub(super) fn count_independent_operation_groups(computation: &XlaComputation) -> usize {
        let mut groups = 0;
        for i in 0..computation.nodes.len() {
            for j in (i + 1)..computation.nodes.len() {
                let node_i = &computation.nodes[i];
                let node_j = &computation.nodes[j];
                let has_shared_deps = node_i
                    .operands
                    .iter()
                    .any(|op_i| node_j.operands.contains(op_i));
                let depends_on_each_other =
                    node_i.operands.contains(&node_j.id) || node_j.operands.contains(&node_i.id);
                if !has_shared_deps && !depends_on_each_other {
                    groups += 1;
                }
            }
        }
        groups
    }
    /// Count operations that can be parallelized across batch dimension
    pub(super) fn count_batch_parallelizable_ops(computation: &XlaComputation) -> usize {
        computation
            .nodes
            .iter()
            .filter(|node| {
                !node.shape.is_scalar()
                    && node.shape.ndim() >= 2
                    && node.shape.dims().first().map_or(false, |&d| d > 1)
                    && matches!(
                        node.opcode,
                        HloOpcode::Dot
                            | HloOpcode::Convolution
                            | HloOpcode::Add
                            | HloOpcode::Multiply
                    )
            })
            .count()
    }
}
/// XLA computation node
///
/// Represents a single operation in the XLA computation graph.
#[derive(Debug, Clone)]
pub struct XlaNode {
    /// Unique identifier
    pub id: XlaNodeId,
    /// Operation code
    pub opcode: HloOpcode,
    /// Human-readable name
    pub name: String,
    /// Output shape
    pub shape: Shape,
    /// Output data type
    pub dtype: DType,
    /// Input node IDs
    pub operands: Vec<XlaNodeId>,
    /// Additional metadata (dimensions, etc.)
    pub metadata: XlaMetadata,
}
/// Layout Optimization Pass
///
/// Optimizes tensor memory layout for better cache performance and memory access patterns.
/// This pass analyzes the computation graph and determines optimal memory layouts (row-major,
/// column-major, tiled) for each tensor based on how it's accessed.
///
/// # Optimizations
/// - Convert layouts to maximize cache line utilization
/// - Minimize memory padding and alignment issues
/// - Optimize for SIMD and vector operations
/// - Consider hardware cache line sizes
///
/// # Example
/// ```rust,ignore
/// // Before: Transpose followed by matmul may have poor cache performance
/// let transposed = Transpose(A)
/// let result = Dot(transposed, B)
///
/// // After: Layout optimization changes memory layout of A instead of explicit transpose
/// let result = Dot(A_rowmajor, B)  // A stored in optimal layout
/// ```
pub struct LayoutOptimizationPass;
impl LayoutOptimizationPass {
    /// Count operations that could benefit from layout optimization
    pub(super) fn count_layout_opportunities(computation: &XlaComputation) -> usize {
        computation
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    node.opcode,
                    HloOpcode::Transpose | HloOpcode::Dot | HloOpcode::Convolution
                )
            })
            .count()
    }
}
/// Constant folding optimization pass
///
/// Evaluates constant expressions at compile time.
pub struct ConstantFoldingPass;
/// Pass manager to orchestrate multiple optimization passes
pub struct XlaPassManager {
    /// Registered passes
    pub(super) passes: Vec<Box<dyn XlaPass>>,
    /// Whether to run passes until convergence
    pub(super) run_until_fixed_point: bool,
    /// Maximum number of iterations
    pub(super) max_iterations: usize,
}
impl XlaPassManager {
    /// Create a new pass manager
    pub fn new() -> Self {
        Self::default()
    }
    /// Add a pass to the manager
    pub fn add_pass(&mut self, pass: Box<dyn XlaPass>) {
        self.passes.push(pass);
    }
    /// Set whether to run until fixed point
    pub fn with_fixed_point(mut self, enabled: bool) -> Self {
        self.run_until_fixed_point = enabled;
        self
    }
    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }
    /// Run all passes on a computation
    pub fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut total_stats = PassStatistics::new();
        let config = computation.config.clone();
        if self.run_until_fixed_point {
            for iteration in 0..self.max_iterations {
                let mut iteration_stats = PassStatistics::new();
                for pass in &self.passes {
                    if pass.should_run(&config) {
                        let stats = pass.run(computation)?;
                        iteration_stats.merge(&stats);
                    }
                }
                total_stats.merge(&iteration_stats);
                if !iteration_stats.has_changes() {
                    break;
                }
                if iteration == self.max_iterations - 1 {
                    return Err(TorshError::invalid_operation(
                        "XLA optimization did not converge within iteration limit",
                    ));
                }
            }
        } else {
            for pass in &self.passes {
                if pass.should_run(&config) {
                    let stats = pass.run(computation)?;
                    total_stats.merge(&stats);
                }
            }
        }
        Ok(total_stats)
    }
    /// Get the list of registered passes
    pub fn passes(&self) -> Vec<&str> {
        self.passes.iter().map(|p| p.name()).collect()
    }
}
/// XLA compilation target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XlaTarget {
    /// CPU compilation
    Cpu,
    /// CUDA GPU compilation
    Gpu,
    /// TPU compilation
    Tpu,
}
impl XlaTarget {
    /// Convert from DeviceType
    pub fn from_device_type(device: DeviceType) -> Self {
        match device {
            DeviceType::Cpu => XlaTarget::Cpu,
            DeviceType::Cuda(_) | DeviceType::Metal(_) | DeviceType::Wgpu(_) => XlaTarget::Gpu,
        }
    }
    /// Get target name
    pub fn name(&self) -> &'static str {
        match self {
            XlaTarget::Cpu => "cpu",
            XlaTarget::Gpu => "gpu",
            XlaTarget::Tpu => "tpu",
        }
    }
}
/// XLA computation builder
///
/// Incrementally constructs an XLA computation graph.
pub struct XlaBuilder {
    /// Computation name
    name: String,
    /// Nodes in the computation
    nodes: Vec<XlaNode>,
    /// Next node ID
    next_id: usize,
}
impl XlaBuilder {
    /// Create a new XLA computation builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            next_id: 0,
        }
    }
    /// Get the next node ID
    fn allocate_id(&mut self) -> XlaNodeId {
        let id = XlaNodeId(self.next_id);
        self.next_id += 1;
        id
    }
    /// Add a parameter to the computation
    pub fn add_parameter(
        &mut self,
        param_num: usize,
        shape: &[usize],
        dtype: DType,
    ) -> Result<XlaNodeId> {
        let id = self.allocate_id();
        let shape = Shape::from_dims(shape.to_vec())?;
        self.nodes.push(XlaNode {
            id,
            opcode: HloOpcode::Parameter,
            name: format!("param_{}", param_num),
            shape,
            dtype,
            operands: Vec::new(),
            metadata: XlaMetadata::default(),
        });
        Ok(id)
    }
    /// Add a constant to the computation
    pub fn add_constant(&mut self, shape: &[usize], dtype: DType) -> Result<XlaNodeId> {
        let id = self.allocate_id();
        let shape = Shape::from_dims(shape.to_vec())?;
        self.nodes.push(XlaNode {
            id,
            opcode: HloOpcode::Constant,
            name: format!("constant_{}", id.0),
            shape,
            dtype,
            operands: Vec::new(),
            metadata: XlaMetadata::default(),
        });
        Ok(id)
    }
    /// Add element-wise binary operation
    fn add_binary_op(
        &mut self,
        opcode: HloOpcode,
        lhs: XlaNodeId,
        rhs: XlaNodeId,
    ) -> Result<XlaNodeId> {
        let (shape, dtype) = {
            let lhs_node = self.get_node(lhs)?;
            let rhs_node = self.get_node(rhs)?;
            if lhs_node.shape.dims() != rhs_node.shape.dims() {
                return Err(TorshError::dimension_error(
                    &format!(
                        "Binary operation requires compatible shapes, got {:?} and {:?}",
                        lhs_node.shape.dims(),
                        rhs_node.shape.dims()
                    ),
                    opcode.name(),
                ));
            }
            (lhs_node.shape.clone(), lhs_node.dtype)
        };
        let id = self.allocate_id();
        self.nodes.push(XlaNode {
            id,
            opcode,
            name: format!("{}_{}", opcode.name(), id.0),
            shape,
            dtype,
            operands: vec![lhs, rhs],
            metadata: XlaMetadata::default(),
        });
        Ok(id)
    }
    /// Add element-wise addition
    pub fn add_add(&mut self, lhs: XlaNodeId, rhs: XlaNodeId) -> Result<XlaNodeId> {
        self.add_binary_op(HloOpcode::Add, lhs, rhs)
    }
    /// Add element-wise subtraction
    pub fn add_subtract(&mut self, lhs: XlaNodeId, rhs: XlaNodeId) -> Result<XlaNodeId> {
        self.add_binary_op(HloOpcode::Subtract, lhs, rhs)
    }
    /// Add element-wise multiplication
    pub fn add_multiply(&mut self, lhs: XlaNodeId, rhs: XlaNodeId) -> Result<XlaNodeId> {
        self.add_binary_op(HloOpcode::Multiply, lhs, rhs)
    }
    /// Add element-wise division
    pub fn add_divide(&mut self, lhs: XlaNodeId, rhs: XlaNodeId) -> Result<XlaNodeId> {
        self.add_binary_op(HloOpcode::Divide, lhs, rhs)
    }
    /// Add matrix multiplication (dot product)
    pub fn add_dot(&mut self, lhs: XlaNodeId, rhs: XlaNodeId) -> Result<XlaNodeId> {
        let (shape, dtype) = {
            let lhs_node = self.get_node(lhs)?;
            let rhs_node = self.get_node(rhs)?;
            if lhs_node.shape.ndim() != 2 || rhs_node.shape.ndim() != 2 {
                return Err(TorshError::dimension_error(
                    "Dot operation requires 2D matrices",
                    "dot",
                ));
            }
            let lhs_dims = lhs_node.shape.dims();
            let rhs_dims = rhs_node.shape.dims();
            if lhs_dims[1] != rhs_dims[0] {
                return Err(TorshError::dimension_error(
                    &format!(
                        "Dot operation dimension mismatch: {}x{} and {}x{}",
                        lhs_dims[0], lhs_dims[1], rhs_dims[0], rhs_dims[1]
                    ),
                    "dot",
                ));
            }
            (Shape::from_2d(lhs_dims[0], rhs_dims[1])?, lhs_node.dtype)
        };
        let id = self.allocate_id();
        self.nodes.push(XlaNode {
            id,
            opcode: HloOpcode::Dot,
            name: format!("dot_{}", id.0),
            shape,
            dtype,
            operands: vec![lhs, rhs],
            metadata: XlaMetadata::default(),
        });
        Ok(id)
    }
    /// Add reshape operation
    pub fn add_reshape(&mut self, operand: XlaNodeId, new_shape: &[usize]) -> Result<XlaNodeId> {
        let (shape, dtype) = {
            let operand_node = self.get_node(operand)?;
            let old_numel = operand_node.shape.numel();
            let new_numel: usize = new_shape.iter().product();
            if old_numel != new_numel {
                return Err(TorshError::dimension_error(
                    &format!(
                        "Reshape requires same number of elements: {} != {}",
                        old_numel, new_numel
                    ),
                    "reshape",
                ));
            }
            (Shape::from_dims(new_shape.to_vec())?, operand_node.dtype)
        };
        let id = self.allocate_id();
        self.nodes.push(XlaNode {
            id,
            opcode: HloOpcode::Reshape,
            name: format!("reshape_{}", id.0),
            shape,
            dtype,
            operands: vec![operand],
            metadata: XlaMetadata::default(),
        });
        Ok(id)
    }
    /// Add transpose operation
    pub fn add_transpose(
        &mut self,
        operand: XlaNodeId,
        permutation: &[usize],
    ) -> Result<XlaNodeId> {
        let (shape, dtype) = {
            let operand_node = self.get_node(operand)?;
            if permutation.len() != operand_node.shape.ndim() {
                return Err(TorshError::dimension_error(
                    "Transpose permutation must match number of dimensions",
                    "transpose",
                ));
            }
            let old_dims = operand_node.shape.dims();
            let new_dims: Vec<usize> = permutation.iter().map(|&i| old_dims[i]).collect();
            (Shape::from_dims(new_dims)?, operand_node.dtype)
        };
        let id = self.allocate_id();
        self.nodes.push(XlaNode {
            id,
            opcode: HloOpcode::Transpose,
            name: format!("transpose_{}", id.0),
            shape,
            dtype,
            operands: vec![operand],
            metadata: XlaMetadata {
                transpose_perm: permutation.to_vec(),
                ..Default::default()
            },
        });
        Ok(id)
    }
    /// Add copy operation
    pub fn add_copy(&mut self, operand: XlaNodeId) -> Result<XlaNodeId> {
        let (shape, dtype) = {
            let operand_node = self.get_node(operand)?;
            (operand_node.shape.clone(), operand_node.dtype)
        };
        let id = self.allocate_id();
        self.nodes.push(XlaNode {
            id,
            opcode: HloOpcode::Copy,
            name: format!("copy_{}", id.0),
            shape,
            dtype,
            operands: vec![operand],
            metadata: XlaMetadata::default(),
        });
        Ok(id)
    }
    /// Get node by ID
    fn get_node(&self, id: XlaNodeId) -> Result<&XlaNode> {
        self.nodes
            .iter()
            .find(|n| n.id == id)
            .ok_or_else(|| TorshError::dimension_error(&format!("Node {:?} not found", id), "xla"))
    }
    /// Build the final computation
    pub fn build(self, root: XlaNodeId) -> Result<XlaComputation> {
        let _ = self.nodes.iter().find(|n| n.id == root).ok_or_else(|| {
            TorshError::dimension_error(&format!("Root node {:?} not found", root), "xla")
        })?;
        Ok(XlaComputation {
            name: self.name,
            nodes: self.nodes,
            root,
            config: XlaConfig::default(),
        })
    }
    /// Get number of nodes in the computation
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}
/// Statistics about optimization pass execution
#[derive(Debug, Clone, Default)]
pub struct PassStatistics {
    /// Number of nodes removed
    pub nodes_removed: usize,
    /// Number of nodes added
    pub nodes_added: usize,
    /// Number of nodes modified
    pub nodes_modified: usize,
    /// Whether the pass changed the graph
    pub changed: bool,
}
impl PassStatistics {
    /// Create empty statistics
    pub fn new() -> Self {
        Self::default()
    }
    /// Check if the pass made any changes
    pub fn has_changes(&self) -> bool {
        self.changed
    }
    /// Merge with another statistics
    pub fn merge(&mut self, other: &PassStatistics) {
        self.nodes_removed += other.nodes_removed;
        self.nodes_added += other.nodes_added;
        self.nodes_modified += other.nodes_modified;
        self.changed |= other.changed;
    }
}
/// Common subexpression elimination pass
///
/// Deduplicates identical operations to reduce redundant computation.
pub struct CommonSubexpressionEliminationPass;
impl CommonSubexpressionEliminationPass {
    /// Check if two nodes are equivalent (same operation, same operands)
    pub(super) fn nodes_equivalent(node1: &XlaNode, node2: &XlaNode) -> bool {
        node1.opcode == node2.opcode
            && node1.dtype == node2.dtype
            && node1.operands == node2.operands
            && Self::metadata_equivalent(&node1.metadata, &node2.metadata)
    }
    /// Check if metadata is equivalent
    fn metadata_equivalent(meta1: &XlaMetadata, meta2: &XlaMetadata) -> bool {
        meta1.reduce_dims == meta2.reduce_dims
            && meta1.transpose_perm == meta2.transpose_perm
            && meta1.broadcast_dims == meta2.broadcast_dims
            && meta1.concat_dim == meta2.concat_dim
    }
}
/// Memory Allocation Optimization Pass
///
/// Optimizes memory allocations by analyzing buffer lifetimes and enabling buffer reuse.
/// This pass identifies opportunities to reuse memory buffers instead of allocating new ones,
/// which significantly reduces memory pressure and improves performance.
///
/// # Optimizations
/// - Analyze tensor lifetimes to determine when buffers can be reused
/// - Identify in-place operation opportunities (where output can reuse input buffer)
/// - Minimize peak memory usage through optimal buffer allocation scheduling
/// - Reduce memory fragmentation by grouping similar-sized allocations
///
/// # Memory Reuse Patterns
/// 1. **In-place Operations**: Output overwrites input (e.g., ReLU, addition with same dest)
/// 2. **Sequential Reuse**: Output of Op A reused for output of Op B after A is consumed
/// 3. **Temporary Buffer Pooling**: Common temporary buffers shared across operations
///
/// # Example
/// ```rust,ignore
/// // Before: Each operation allocates new buffer
/// let t1 = Add(A, B)      // Allocates buffer_1
/// let t2 = Mul(t1, C)     // Allocates buffer_2
/// let t3 = ReLU(t2)       // Allocates buffer_3
///
/// // After: Memory allocation optimization
/// let t1 = Add(A, B)      // Allocates buffer_1
/// let t2 = Mul(t1, C)     // Reuses buffer_1 (t1 no longer needed)
/// let t3 = ReLU(t2)       // In-place in buffer_1 (ReLU doesn't need separate output)
/// ```
pub struct MemoryAllocationOptimizationPass;
impl MemoryAllocationOptimizationPass {
    /// Count opportunities for buffer reuse across operations
    pub(super) fn count_buffer_reuse_opportunities(computation: &XlaComputation) -> usize {
        let mut count = 0;
        for node in &computation.nodes {
            if !node.operands.is_empty() {
                for &operand_id in &node.operands {
                    let uses = computation
                        .nodes
                        .iter()
                        .filter(|n| n.operands.contains(&operand_id))
                        .count();
                    if uses == 1 {
                        count += 1;
                        break;
                    }
                }
            }
        }
        count
    }
    /// Count opportunities for in-place operations
    pub(super) fn count_inplace_opportunities(computation: &XlaComputation) -> usize {
        computation
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    node.opcode,
                    HloOpcode::Add
                        | HloOpcode::Subtract
                        | HloOpcode::Multiply
                        | HloOpcode::Divide
                        | HloOpcode::Abs
                        | HloOpcode::Exp
                        | HloOpcode::Log
                        | HloOpcode::Sqrt
                        | HloOpcode::Sin
                        | HloOpcode::Cos
                        | HloOpcode::Tanh
                )
            })
            .count()
    }
}
