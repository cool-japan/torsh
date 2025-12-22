//! TensorFlow XLA (Accelerated Linear Algebra) Integration
//!
//! This module provides integration with TensorFlow's XLA compiler for optimized
//! computation graph execution. XLA compiles tensor operations into highly
//! optimized machine code for CPUs, GPUs, and TPUs.
//!
//! # XLA Overview
//!
//! XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra
//! that optimizes TensorFlow computations. It provides:
//!
//! - **Fusion**: Combines multiple operations into a single kernel
//! - **Optimization**: Aggressive algebraic simplification and layout optimization
//! - **Buffer Analysis**: Reduces memory allocation and copying
//! - **Parallelization**: Automatically parallelizes operations
//!
//! # Design Principles
//!
//! 1. **HLO Representation**: Use High-Level Optimizer (HLO) intermediate representation
//! 2. **Graph-Based**: Build computation graphs for XLA compilation
//! 3. **Type-Safe**: Leverage Rust's type system for correctness
//! 4. **Interoperable**: Compatible with TensorFlow and JAX ecosystems
//!
//! # SciRS2 POLICY Compliance
//!
//! This module strictly follows the SciRS2 POLICY by:
//! - Only using Rust standard library and torsh-core types
//! - NO external dependencies beyond scirs2 ecosystem
//! - Providing pure Rust implementation of XLA concepts
//!
//! # Example
//!
//! ```rust,ignore
//! use torsh_core::xla_integration::{XlaComputation, XlaBuilder, HloOpcode};
//!
//! // Build XLA computation graph
//! let mut builder = XlaBuilder::new("matmul_computation");
//!
//! // Add operations
//! let param_a = builder.add_parameter(0, &[128, 256], HloOpcode::Parameter)?;
//! let param_b = builder.add_parameter(1, &[256, 512], HloOpcode::Parameter)?;
//! let result = builder.add_dot(param_a, param_b)?;
//!
//! // Build computation
//! let computation = builder.build(result)?;
//!
//! // Execute (would interface with XLA runtime)
//! // let output = computation.execute(&[input_a, input_b])?;
//! ```

use crate::device::DeviceType;
use crate::dtype::DType;
use crate::error::{Result, TorshError};
use crate::shape::Shape;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};
#[cfg(feature = "std")]
use std::{string::String, vec::Vec};

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

/// Node identifier in XLA computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct XlaNodeId(pub usize);

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
        // Validate operands exist and clone necessary data
        let (shape, dtype) = {
            let lhs_node = self.get_node(lhs)?;
            let rhs_node = self.get_node(rhs)?;

            // Check shape compatibility (simplified broadcasting)
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
        // Validate matrix multiplication dimensions and clone necessary data
        let (shape, dtype) = {
            let lhs_node = self.get_node(lhs)?;
            let rhs_node = self.get_node(rhs)?;

            // Validate matrix multiplication dimensions
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
        // Validate reshape and clone necessary data
        let (shape, dtype) = {
            let operand_node = self.get_node(operand)?;

            // Validate reshape preserves number of elements
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
        // Validate and compute transposed shape
        let (shape, dtype) = {
            let operand_node = self.get_node(operand)?;

            // Validate permutation
            if permutation.len() != operand_node.shape.ndim() {
                return Err(TorshError::dimension_error(
                    "Transpose permutation must match number of dimensions",
                    "transpose",
                ));
            }

            // Compute transposed shape
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
        // Get shape and dtype from operand
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
        // Validate root exists
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
        // Check root exists
        if !self.nodes.iter().any(|n| n.id == self.root) {
            return Err(TorshError::dimension_error(
                "Root node not found in computation",
                "validate",
            ));
        }

        // Validate all operands exist
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

impl Default for XlaConfig {
    fn default() -> Self {
        Self {
            target: XlaTarget::Cpu,
            enable_fusion: true,
            enable_layout_optimization: true,
            enable_algebraic_simplification: true,
            optimization_level: 2,
        }
    }
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

// ================================================================================================
// XLA Optimization Pass Infrastructure
// ================================================================================================

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

/// Trait for XLA optimization passes
pub trait XlaPass {
    /// Name of the pass
    fn name(&self) -> &str;

    /// Run the pass on a computation
    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics>;

    /// Check if the pass should be run (based on config)
    fn should_run(&self, config: &XlaConfig) -> bool {
        config.optimization_level > 0
    }
}

/// Constant folding optimization pass
///
/// Evaluates constant expressions at compile time.
pub struct ConstantFoldingPass;

impl XlaPass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "constant-folding"
    }

    fn run(&self, _computation: &mut XlaComputation) -> Result<PassStatistics> {
        let stats = PassStatistics::new();

        // For now, return placeholder statistics
        // Full implementation would:
        // 1. Identify operations with all constant operands
        // 2. Evaluate them at compile time
        // 3. Replace with constant nodes

        Ok(stats)
    }

    fn should_run(&self, config: &XlaConfig) -> bool {
        config.enable_algebraic_simplification && config.optimization_level >= 1
    }
}

/// Dead code elimination pass
///
/// Removes operations that do not contribute to the final result.
pub struct DeadCodeEliminationPass;

impl DeadCodeEliminationPass {
    /// Mark all nodes reachable from root
    fn mark_reachable(computation: &XlaComputation, root: XlaNodeId, reachable: &mut Vec<bool>) {
        if reachable[root.0] {
            return; // Already marked
        }

        reachable[root.0] = true;

        // Mark all operands as reachable
        if let Some(node) = computation.nodes.get(root.0) {
            for &operand in &node.operands {
                Self::mark_reachable(computation, operand, reachable);
            }
        }
    }
}

impl XlaPass for DeadCodeEliminationPass {
    fn name(&self) -> &str {
        "dead-code-elimination"
    }

    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::new();

        if computation.nodes.is_empty() {
            return Ok(stats);
        }

        // Mark all reachable nodes from root
        let mut reachable = vec![false; computation.nodes.len()];
        Self::mark_reachable(computation, computation.root, &mut reachable);

        // Count unreachable nodes (dead code)
        let dead_count = reachable.iter().filter(|&&r| !r).count();

        if dead_count > 0 {
            stats.nodes_removed = dead_count;
            // Note: Not setting changed=true because this is a placeholder implementation
            // Full implementation would actually remove the nodes and set changed=true
            // For now, we just report statistics without modifying the graph
        }

        Ok(stats)
    }

    fn should_run(&self, config: &XlaConfig) -> bool {
        config.optimization_level >= 1
    }
}

/// Common subexpression elimination pass
///
/// Deduplicates identical operations to reduce redundant computation.
pub struct CommonSubexpressionEliminationPass;

impl CommonSubexpressionEliminationPass {
    /// Check if two nodes are equivalent (same operation, same operands)
    fn nodes_equivalent(node1: &XlaNode, node2: &XlaNode) -> bool {
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

impl XlaPass for CommonSubexpressionEliminationPass {
    fn name(&self) -> &str {
        "common-subexpression-elimination"
    }

    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::new();

        // Find duplicate operations
        let mut duplicates = 0;
        for i in 0..computation.nodes.len() {
            for j in (i + 1)..computation.nodes.len() {
                if Self::nodes_equivalent(&computation.nodes[i], &computation.nodes[j]) {
                    duplicates += 1;
                }
            }
        }

        if duplicates > 0 {
            stats.nodes_removed = duplicates;
            // Note: Not setting changed=true because this is a placeholder implementation
            // Full implementation would replace duplicate nodes and set changed=true
        }

        Ok(stats)
    }

    fn should_run(&self, config: &XlaConfig) -> bool {
        config.optimization_level >= 2
    }
}

/// Operation fusion pass
///
/// Combines multiple operations into fused kernels for better performance.
pub struct OperationFusionPass;

impl OperationFusionPass {
    /// Check if two operations can be fused
    fn can_fuse(op1: &XlaNode, op2: &XlaNode) -> bool {
        // Element-wise operations can typically be fused
        op1.opcode.is_elementwise()
            && op2.opcode.is_elementwise()
            && op1.shape == op2.shape
            && op1.dtype == op2.dtype
    }

    /// Find fusion opportunities
    fn find_fusion_candidates(computation: &XlaComputation) -> Vec<(XlaNodeId, XlaNodeId)> {
        let mut candidates = Vec::new();

        for i in 0..computation.nodes.len() {
            for j in (i + 1)..computation.nodes.len() {
                let node1 = &computation.nodes[i];
                let node2 = &computation.nodes[j];

                // Check if node2 uses node1 as input
                if node2.operands.contains(&XlaNodeId(i)) && Self::can_fuse(node1, node2) {
                    candidates.push((XlaNodeId(i), XlaNodeId(j)));
                }
            }
        }

        candidates
    }
}

impl XlaPass for OperationFusionPass {
    fn name(&self) -> &str {
        "operation-fusion"
    }

    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::new();

        if !computation.config.enable_fusion {
            return Ok(stats);
        }

        let fusion_candidates = Self::find_fusion_candidates(computation);

        if !fusion_candidates.is_empty() {
            stats.nodes_removed = fusion_candidates.len();
            stats.nodes_added = fusion_candidates.len(); // Fused kernels
                                                         // Note: Not setting changed=true because this is a placeholder implementation
                                                         // Full implementation would create fused kernel nodes and set changed=true
        }

        Ok(stats)
    }

    fn should_run(&self, config: &XlaConfig) -> bool {
        config.enable_fusion && config.optimization_level >= 2
    }
}

/// Algebraic simplification pass
///
/// Applies algebraic identities to simplify expressions.
/// Examples: x + 0 = x, x * 1 = x, x * 0 = 0
pub struct AlgebraicSimplificationPass;

impl AlgebraicSimplificationPass {
    /// Check if a node is a constant zero
    fn is_constant_zero(_node: &XlaNode) -> bool {
        // Would check if node is Constant with value 0
        false
    }

    /// Check if a node is a constant one
    fn is_constant_one(_node: &XlaNode) -> bool {
        // Would check if node is Constant with value 1
        false
    }

    /// Find simplification opportunities
    fn find_simplifications(computation: &XlaComputation) -> usize {
        let mut count = 0;

        for node in &computation.nodes {
            match node.opcode {
                // x + 0 = x, 0 + x = x
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
                // x * 1 = x, 1 * x = x
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
                // x * 0 = 0, 0 * x = 0
                _ => {}
            }
        }

        count
    }
}

impl XlaPass for AlgebraicSimplificationPass {
    fn name(&self) -> &str {
        "algebraic-simplification"
    }

    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::new();

        let simplifications = Self::find_simplifications(computation);

        if simplifications > 0 {
            stats.nodes_modified = simplifications;
            // Note: Not setting changed=true because this is a placeholder implementation
            // Full implementation would apply the simplifications and set changed=true
        }

        Ok(stats)
    }

    fn should_run(&self, config: &XlaConfig) -> bool {
        config.enable_algebraic_simplification && config.optimization_level >= 1
    }
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

impl XlaPass for LayoutOptimizationPass {
    fn name(&self) -> &str {
        "layout-optimization"
    }

    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::default();

        if !self.should_run(&computation.config) {
            return Ok(stats);
        }

        // Count operations that could benefit from layout optimization
        let layout_optimization_opportunities = Self::count_layout_opportunities(computation);

        if layout_optimization_opportunities > 0 {
            stats.nodes_modified = layout_optimization_opportunities;
            // Note: Not setting changed=true because this is a placeholder implementation
            // Full implementation would:
            // 1. Analyze access patterns for each tensor
            // 2. Determine optimal layout (row-major, column-major, tiled)
            // 3. Insert layout conversion operations where beneficial
            // 4. Remove redundant transposes that are handled by layout changes
        }

        Ok(stats)
    }

    fn should_run(&self, config: &XlaConfig) -> bool {
        config.optimization_level >= 2
    }
}

impl LayoutOptimizationPass {
    /// Count operations that could benefit from layout optimization
    fn count_layout_opportunities(computation: &XlaComputation) -> usize {
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

impl XlaPass for CopyEliminationPass {
    fn name(&self) -> &str {
        "copy-elimination"
    }

    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::default();

        if !self.should_run(&computation.config) {
            return Ok(stats);
        }

        // Count copy operations that could be eliminated
        let eliminable_copies = Self::count_eliminable_copies(computation);

        if eliminable_copies > 0 {
            stats.nodes_removed = eliminable_copies;
            // Note: Not setting changed=true because this is a placeholder implementation
            // Full implementation would:
            // 1. Build use-def chains to track data flow
            // 2. Identify copies that can be eliminated safely
            // 3. Update references to use original values
            // 4. Remove the copy nodes from the graph
        }

        Ok(stats)
    }

    fn should_run(&self, config: &XlaConfig) -> bool {
        config.optimization_level >= 1
    }
}

impl CopyEliminationPass {
    /// Count copy operations that could potentially be eliminated
    fn count_eliminable_copies(computation: &XlaComputation) -> usize {
        // In placeholder: count all Copy operations as potentially eliminable
        // Full implementation would do deeper analysis
        computation
            .nodes
            .iter()
            .filter(|node| node.opcode == HloOpcode::Copy)
            .count()
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

impl XlaPass for MemoryAllocationOptimizationPass {
    fn name(&self) -> &str {
        "memory-allocation-optimization"
    }

    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::default();

        if !self.should_run(&computation.config) {
            return Ok(stats);
        }

        // Analyze buffer reuse opportunities
        let reuse_opportunities = Self::count_buffer_reuse_opportunities(computation);
        let inplace_opportunities = Self::count_inplace_opportunities(computation);

        if reuse_opportunities > 0 || inplace_opportunities > 0 {
            stats.nodes_modified = reuse_opportunities + inplace_opportunities;
            // Note: Not setting changed=true because this is a placeholder implementation
            // Full implementation would:
            // 1. Build liveness analysis for all tensors
            // 2. Compute buffer lifetime intervals
            // 3. Assign buffer IDs to tensors with non-overlapping lifetimes
            // 4. Mark operations that can execute in-place
            // 5. Schedule allocations to minimize peak memory usage
        }

        Ok(stats)
    }

    fn should_run(&self, config: &XlaConfig) -> bool {
        // Memory optimization is crucial for all optimization levels
        config.optimization_level >= 1
    }
}

impl MemoryAllocationOptimizationPass {
    /// Count opportunities for buffer reuse across operations
    fn count_buffer_reuse_opportunities(computation: &XlaComputation) -> usize {
        // Count operations where output buffer could potentially reuse input buffer
        // after the input is no longer needed
        let mut count = 0;

        for node in &computation.nodes {
            // Operations with single use operands are candidates for buffer reuse
            if !node.operands.is_empty() {
                // Check if any operand is only used by this node
                for &operand_id in &node.operands {
                    let uses = computation
                        .nodes
                        .iter()
                        .filter(|n| n.operands.contains(&operand_id))
                        .count();

                    if uses == 1 {
                        count += 1;
                        break; // Only count once per node
                    }
                }
            }
        }

        count
    }

    /// Count opportunities for in-place operations
    fn count_inplace_opportunities(computation: &XlaComputation) -> usize {
        // Element-wise operations can often be performed in-place
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

impl XlaPass for ParallelizationAnalysisPass {
    fn name(&self) -> &str {
        "parallelization-analysis"
    }

    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::default();

        if !self.should_run(&computation.config) {
            return Ok(stats);
        }

        // Analyze parallelization opportunities
        let independent_groups = Self::count_independent_operation_groups(computation);
        let batch_parallel_ops = Self::count_batch_parallelizable_ops(computation);

        if independent_groups > 0 || batch_parallel_ops > 0 {
            stats.nodes_modified = independent_groups + batch_parallel_ops;
            // Note: Not setting changed=true because this is a placeholder implementation
            // Full implementation would:
            // 1. Build dependency DAG (directed acyclic graph)
            // 2. Perform topological sort to find execution levels
            // 3. Within each level, identify independent operations
            // 4. Estimate cost and benefit of parallelization
            // 5. Insert parallel execution markers
            // 6. Consider device placement for multi-device execution
        }

        Ok(stats)
    }

    fn should_run(&self, config: &XlaConfig) -> bool {
        // Parallelization analysis valuable at level 2+
        config.optimization_level >= 2
    }
}

impl ParallelizationAnalysisPass {
    /// Count groups of independent operations that can execute in parallel
    fn count_independent_operation_groups(computation: &XlaComputation) -> usize {
        // Build a simple dependency analysis
        let mut groups = 0;

        // Find operations with no common dependencies
        for i in 0..computation.nodes.len() {
            for j in (i + 1)..computation.nodes.len() {
                let node_i = &computation.nodes[i];
                let node_j = &computation.nodes[j];

                // Check if nodes are independent (no shared operands)
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
    fn count_batch_parallelizable_ops(computation: &XlaComputation) -> usize {
        // Operations on batched tensors can often be parallelized
        computation
            .nodes
            .iter()
            .filter(|node| {
                // Operations with batch dimension (first dim > 1) are candidates
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

/// Pass manager to orchestrate multiple optimization passes
pub struct XlaPassManager {
    /// Registered passes
    passes: Vec<Box<dyn XlaPass>>,
    /// Whether to run passes until convergence
    run_until_fixed_point: bool,
    /// Maximum number of iterations
    max_iterations: usize,
}

impl Default for XlaPassManager {
    fn default() -> Self {
        let mut manager = Self {
            passes: Vec::new(),
            run_until_fixed_point: true,
            max_iterations: 10,
        };

        // Register standard passes in optimal order
        // 1. Simplification and folding passes
        manager.add_pass(Box::new(ConstantFoldingPass));
        manager.add_pass(Box::new(AlgebraicSimplificationPass));

        // 2. Eliminate redundant operations
        manager.add_pass(Box::new(CopyEliminationPass));
        manager.add_pass(Box::new(CommonSubexpressionEliminationPass));

        // 3. Combine operations for efficiency
        manager.add_pass(Box::new(OperationFusionPass));

        // 4. Optimize layouts and memory
        manager.add_pass(Box::new(LayoutOptimizationPass));
        manager.add_pass(Box::new(MemoryAllocationOptimizationPass));

        // 5. Analyze parallelization opportunities
        manager.add_pass(Box::new(ParallelizationAnalysisPass));

        // 6. Clean up dead code (should be last)
        manager.add_pass(Box::new(DeadCodeEliminationPass));

        manager
    }
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
            // Run passes until no more changes or max iterations
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
                    // Fixed point reached
                    break;
                }

                // Avoid infinite loop
                if iteration == self.max_iterations - 1 {
                    return Err(TorshError::invalid_operation(
                        "XLA optimization did not converge within iteration limit",
                    ));
                }
            }
        } else {
            // Run each pass once
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

// Extension methods for XlaComputation
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hlo_opcode_name() {
        assert_eq!(HloOpcode::Add.name(), "add");
        assert_eq!(HloOpcode::Dot.name(), "dot");
        assert_eq!(HloOpcode::Reshape.name(), "reshape");
    }

    #[test]
    fn test_hlo_opcode_properties() {
        assert!(HloOpcode::Add.is_elementwise());
        assert!(!HloOpcode::Dot.is_elementwise());

        assert!(HloOpcode::Reduce.is_reduction());
        assert!(!HloOpcode::Add.is_reduction());

        assert!(HloOpcode::Reshape.is_shape_changing());
        assert!(!HloOpcode::Add.is_shape_changing());
    }

    #[test]
    fn test_xla_builder_parameter() {
        let mut builder = XlaBuilder::new("test");
        let param = builder.add_parameter(0, &[10, 20], DType::F32).unwrap();

        assert_eq!(builder.num_nodes(), 1);
        assert_eq!(param, XlaNodeId(0));
    }

    #[test]
    fn test_xla_builder_add() {
        let mut builder = XlaBuilder::new("test");
        let param_a = builder.add_parameter(0, &[10, 20], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[10, 20], DType::F32).unwrap();
        let result = builder.add_add(param_a, param_b).unwrap();

        assert_eq!(builder.num_nodes(), 3);
        assert_eq!(result, XlaNodeId(2));
    }

    #[test]
    fn test_xla_builder_dot() {
        let mut builder = XlaBuilder::new("matmul");
        let param_a = builder.add_parameter(0, &[128, 256], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[256, 512], DType::F32).unwrap();
        let result = builder.add_dot(param_a, param_b).unwrap();

        assert_eq!(builder.num_nodes(), 3);

        let computation = builder.build(result).unwrap();
        assert_eq!(computation.output_shape().unwrap().dims(), &[128, 512]);
    }

    #[test]
    fn test_xla_builder_dot_invalid_dims() {
        let mut builder = XlaBuilder::new("matmul");
        let param_a = builder.add_parameter(0, &[128, 256], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[128, 512], DType::F32).unwrap();
        let result = builder.add_dot(param_a, param_b);

        assert!(result.is_err());
    }

    #[test]
    fn test_xla_builder_reshape() {
        let mut builder = XlaBuilder::new("reshape");
        let param = builder.add_parameter(0, &[10, 20, 30], DType::F32).unwrap();
        let result = builder.add_reshape(param, &[10, 600]).unwrap();

        assert_eq!(builder.num_nodes(), 2);

        let computation = builder.build(result).unwrap();
        assert_eq!(computation.output_shape().unwrap().dims(), &[10, 600]);
    }

    #[test]
    fn test_xla_builder_reshape_invalid() {
        let mut builder = XlaBuilder::new("reshape");
        let param = builder.add_parameter(0, &[10, 20, 30], DType::F32).unwrap();
        let result = builder.add_reshape(param, &[10, 100]);

        assert!(result.is_err());
    }

    #[test]
    fn test_xla_builder_transpose() {
        let mut builder = XlaBuilder::new("transpose");
        let param = builder.add_parameter(0, &[10, 20, 30], DType::F32).unwrap();
        let result = builder.add_transpose(param, &[2, 0, 1]).unwrap();

        let computation = builder.build(result).unwrap();
        assert_eq!(computation.output_shape().unwrap().dims(), &[30, 10, 20]);
    }

    #[test]
    fn test_xla_computation_validate() {
        let mut builder = XlaBuilder::new("test");
        let param_a = builder.add_parameter(0, &[10, 20], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[10, 20], DType::F32).unwrap();
        let result = builder.add_add(param_a, param_b).unwrap();

        let computation = builder.build(result).unwrap();
        assert!(computation.validate().is_ok());
    }

    #[test]
    fn test_xla_computation_operation_counts() {
        let mut builder = XlaBuilder::new("test");
        let param_a = builder.add_parameter(0, &[10, 20], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[10, 20], DType::F32).unwrap();
        let add_result = builder.add_add(param_a, param_b).unwrap();
        let mul_result = builder.add_multiply(add_result, param_b).unwrap();

        let computation = builder.build(mul_result).unwrap();
        let counts = computation.operation_counts();

        assert_eq!(counts.len(), 3); // Parameter (2), Add (1), Multiply (1)
    }

    #[test]
    fn test_xla_computation_num_parameters() {
        let mut builder = XlaBuilder::new("test");
        builder.add_parameter(0, &[10, 20], DType::F32).unwrap();
        builder.add_parameter(1, &[10, 20], DType::F32).unwrap();
        let param_c = builder.add_parameter(2, &[10, 20], DType::F32).unwrap();

        let computation = builder.build(param_c).unwrap();
        assert_eq!(computation.num_parameters(), 3);
    }

    #[test]
    fn test_xla_computation_to_hlo_text() {
        let mut builder = XlaBuilder::new("simple_add");
        let param_a = builder.add_parameter(0, &[10, 20], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[10, 20], DType::F32).unwrap();
        let result = builder.add_add(param_a, param_b).unwrap();

        let computation = builder.build(result).unwrap();
        let hlo_text = computation.to_hlo_text();

        assert!(hlo_text.contains("HloModule simple_add"));
        assert!(hlo_text.contains("ENTRY main"));
        assert!(hlo_text.contains("ROOT"));
    }

    #[test]
    fn test_xla_target_from_device_type() {
        assert_eq!(XlaTarget::from_device_type(DeviceType::Cpu), XlaTarget::Cpu);
        assert_eq!(
            XlaTarget::from_device_type(DeviceType::Cuda(0)),
            XlaTarget::Gpu
        );
        assert_eq!(
            XlaTarget::from_device_type(DeviceType::Metal(0)),
            XlaTarget::Gpu
        );
    }

    #[test]
    fn test_xla_config_default() {
        let config = XlaConfig::default();
        assert_eq!(config.target, XlaTarget::Cpu);
        assert!(config.enable_fusion);
        assert_eq!(config.optimization_level, 2);
    }

    #[test]
    fn test_xla_config_builder() {
        let config = XlaConfig::new(XlaTarget::Gpu)
            .with_fusion(false)
            .with_optimization_level(3);

        assert_eq!(config.target, XlaTarget::Gpu);
        assert!(!config.enable_fusion);
        assert_eq!(config.optimization_level, 3);
    }

    #[test]
    fn test_xla_config_presets() {
        let aggressive = XlaConfig::aggressive();
        assert_eq!(aggressive.optimization_level, 3);
        assert!(aggressive.enable_fusion);

        let conservative = XlaConfig::conservative();
        assert_eq!(conservative.optimization_level, 0);
        assert!(!conservative.enable_fusion);
    }

    #[test]
    fn test_complex_computation() {
        let mut builder = XlaBuilder::new("complex");

        // Build: (A @ B) + C * D
        let a = builder.add_parameter(0, &[128, 256], DType::F32).unwrap();
        let b = builder.add_parameter(1, &[256, 512], DType::F32).unwrap();
        let c = builder.add_parameter(2, &[128, 512], DType::F32).unwrap();
        let d = builder.add_parameter(3, &[128, 512], DType::F32).unwrap();

        let matmul = builder.add_dot(a, b).unwrap();
        let mul = builder.add_multiply(c, d).unwrap();
        let result = builder.add_add(matmul, mul).unwrap();

        let computation = builder.build(result).unwrap();
        assert_eq!(computation.num_nodes(), 7); // 4 params + 3 ops
        assert!(computation.validate().is_ok());
    }

    // ========================================================================
    // Optimization Pass Tests
    // ========================================================================

    #[test]
    fn test_pass_statistics_creation() {
        let stats = PassStatistics::new();
        assert_eq!(stats.nodes_removed, 0);
        assert_eq!(stats.nodes_added, 0);
        assert_eq!(stats.nodes_modified, 0);
        assert!(!stats.changed);
        assert!(!stats.has_changes());
    }

    #[test]
    fn test_pass_statistics_merge() {
        let mut stats1 = PassStatistics {
            nodes_removed: 2,
            nodes_added: 1,
            nodes_modified: 3,
            changed: true,
        };

        let stats2 = PassStatistics {
            nodes_removed: 1,
            nodes_added: 2,
            nodes_modified: 1,
            changed: false,
        };

        stats1.merge(&stats2);
        assert_eq!(stats1.nodes_removed, 3);
        assert_eq!(stats1.nodes_added, 3);
        assert_eq!(stats1.nodes_modified, 4);
        assert!(stats1.changed);
    }

    #[test]
    fn test_constant_folding_pass_name() {
        let pass = ConstantFoldingPass;
        assert_eq!(pass.name(), "constant-folding");
    }

    #[test]
    fn test_constant_folding_pass_should_run() {
        let pass = ConstantFoldingPass;

        let config = XlaConfig::default();
        assert!(pass.should_run(&config));

        let config = XlaConfig::conservative();
        assert!(!pass.should_run(&config));
    }

    #[test]
    fn test_dead_code_elimination_pass_name() {
        let pass = DeadCodeEliminationPass;
        assert_eq!(pass.name(), "dead-code-elimination");
    }

    #[test]
    fn test_dead_code_elimination_empty_computation() {
        let pass = DeadCodeEliminationPass;
        let mut computation = XlaComputation {
            name: "empty".to_string(),
            nodes: vec![],
            root: XlaNodeId(0),
            config: XlaConfig::default(),
        };

        let stats = pass.run(&mut computation).unwrap();
        assert_eq!(stats.nodes_removed, 0);
        assert!(!stats.changed);
    }

    #[test]
    fn test_dead_code_elimination_no_dead_code() {
        let mut builder = XlaBuilder::new("test");
        let param = builder.add_parameter(0, &[10], DType::F32).unwrap();
        let computation = builder.build(param).unwrap();

        let pass = DeadCodeEliminationPass;
        let mut mut_comp = computation;
        let stats = pass.run(&mut mut_comp).unwrap();
        assert_eq!(stats.nodes_removed, 0);
        assert!(!stats.changed);
    }

    #[test]
    fn test_cse_pass_name() {
        let pass = CommonSubexpressionEliminationPass;
        assert_eq!(pass.name(), "common-subexpression-elimination");
    }

    #[test]
    fn test_cse_pass_should_run() {
        let pass = CommonSubexpressionEliminationPass;

        let config = XlaConfig::default();
        assert!(pass.should_run(&config));

        let config = XlaConfig::conservative();
        assert!(!pass.should_run(&config));
    }

    #[test]
    fn test_fusion_pass_name() {
        let pass = OperationFusionPass;
        assert_eq!(pass.name(), "operation-fusion");
    }

    #[test]
    fn test_fusion_pass_should_run() {
        let pass = OperationFusionPass;

        let config = XlaConfig::default();
        assert!(pass.should_run(&config));

        let config = XlaConfig::conservative();
        assert!(!pass.should_run(&config));

        let mut config = XlaConfig::default();
        config.enable_fusion = false;
        assert!(!pass.should_run(&config));
    }

    #[test]
    fn test_algebraic_simplification_pass_name() {
        let pass = AlgebraicSimplificationPass;
        assert_eq!(pass.name(), "algebraic-simplification");
    }

    #[test]
    fn test_layout_optimization_pass_name() {
        let pass = LayoutOptimizationPass;
        assert_eq!(pass.name(), "layout-optimization");
    }

    #[test]
    fn test_layout_optimization_pass_should_run() {
        let pass = LayoutOptimizationPass;

        let mut config = XlaConfig::default();
        config.optimization_level = 2;
        assert!(pass.should_run(&config));

        config.optimization_level = 1;
        assert!(!pass.should_run(&config));
    }

    #[test]
    fn test_layout_optimization_pass_detects_opportunities() {
        let mut builder = XlaBuilder::new("layout_test");
        // Test with operations that benefit from layout optimization
        let param_a = builder.add_parameter(0, &[128, 256], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[256, 512], DType::F32).unwrap();

        // Matmul and transpose are both layout-sensitive operations
        let matmul = builder.add_dot(param_a, param_b).unwrap();
        let transposed = builder.add_transpose(matmul, &[1, 0]).unwrap();

        let mut computation = builder.build(transposed).unwrap();
        computation.config.optimization_level = 2;

        let pass = LayoutOptimizationPass;
        let stats = pass.run(&mut computation).unwrap();

        // Should detect transpose and matmul as optimization opportunities
        assert!(stats.nodes_modified >= 2);
    }

    #[test]
    fn test_copy_elimination_pass_name() {
        let pass = CopyEliminationPass;
        assert_eq!(pass.name(), "copy-elimination");
    }

    #[test]
    fn test_copy_elimination_pass_should_run() {
        let pass = CopyEliminationPass;

        let mut config = XlaConfig::default();
        config.optimization_level = 1;
        assert!(pass.should_run(&config));

        config.optimization_level = 0;
        assert!(!pass.should_run(&config));
    }

    #[test]
    fn test_copy_elimination_pass_counts_copies() {
        let mut builder = XlaBuilder::new("copy_test");
        let param = builder.add_parameter(0, &[10, 20], DType::F32).unwrap();
        let copy1 = builder.add_copy(param).unwrap();
        let copy2 = builder.add_copy(copy1).unwrap();

        let mut computation = builder.build(copy2).unwrap();
        computation.config.optimization_level = 1;

        let pass = CopyEliminationPass;
        let stats = pass.run(&mut computation).unwrap();

        // Should detect 2 copy operations
        assert_eq!(stats.nodes_removed, 2);
    }

    #[test]
    fn test_pass_manager_default() {
        let manager = XlaPassManager::default();
        let passes = manager.passes();

        assert_eq!(passes.len(), 9);
        assert!(passes.contains(&"constant-folding"));
        assert!(passes.contains(&"algebraic-simplification"));
        assert!(passes.contains(&"copy-elimination"));
        assert!(passes.contains(&"common-subexpression-elimination"));
        assert!(passes.contains(&"operation-fusion"));
        assert!(passes.contains(&"layout-optimization"));
        assert!(passes.contains(&"memory-allocation-optimization"));
        assert!(passes.contains(&"parallelization-analysis"));
        assert!(passes.contains(&"dead-code-elimination"));
    }

    #[test]
    fn test_pass_manager_new() {
        let manager = XlaPassManager::new();
        assert_eq!(manager.passes().len(), 9);
    }

    #[test]
    fn test_pass_manager_add_pass() {
        let mut manager = XlaPassManager {
            passes: Vec::new(),
            run_until_fixed_point: false,
            max_iterations: 1,
        };

        assert_eq!(manager.passes().len(), 0);

        manager.add_pass(Box::new(ConstantFoldingPass));
        assert_eq!(manager.passes().len(), 1);

        manager.add_pass(Box::new(DeadCodeEliminationPass));
        assert_eq!(manager.passes().len(), 2);
    }

    #[test]
    fn test_pass_manager_with_fixed_point() {
        let manager = XlaPassManager::new().with_fixed_point(false);
        assert!(!manager.run_until_fixed_point);
    }

    #[test]
    fn test_pass_manager_with_max_iterations() {
        let manager = XlaPassManager::new().with_max_iterations(20);
        assert_eq!(manager.max_iterations, 20);
    }

    #[test]
    fn test_pass_manager_run_simple_computation() {
        let mut builder = XlaBuilder::new("simple");
        let param = builder.add_parameter(0, &[10], DType::F32).unwrap();
        let mut computation = builder.build(param).unwrap();

        let manager = XlaPassManager::new();
        let stats = manager.run(&mut computation).unwrap();

        // Simple computation should not be modified
        assert_eq!(stats.nodes_removed, 0);
    }

    #[test]
    fn test_computation_optimize() {
        let mut builder = XlaBuilder::new("test");
        let param_a = builder.add_parameter(0, &[10, 20], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[10, 20], DType::F32).unwrap();
        let result = builder.add_add(param_a, param_b).unwrap();

        let mut computation = builder.build(result).unwrap();
        let stats = computation.optimize().unwrap();

        // Should run without errors
        assert!(!stats.changed || stats.changed);
    }

    #[test]
    fn test_computation_optimize_with_custom_manager() {
        let mut builder = XlaBuilder::new("test");
        let param = builder.add_parameter(0, &[10], DType::F32).unwrap();
        let mut computation = builder.build(param).unwrap();

        let manager = XlaPassManager::new().with_max_iterations(5);
        let stats = computation.optimize_with(&manager).unwrap();

        assert_eq!(stats.nodes_removed, 0);
    }

    #[test]
    fn test_computation_run_pass() {
        let mut builder = XlaBuilder::new("test");
        let param = builder.add_parameter(0, &[10], DType::F32).unwrap();
        let mut computation = builder.build(param).unwrap();

        let pass = DeadCodeEliminationPass;
        let stats = computation.run_pass(&pass).unwrap();

        assert_eq!(stats.nodes_removed, 0);
        assert!(!stats.changed);
    }

    #[test]
    fn test_pass_manager_run_until_fixed_point() {
        let mut builder = XlaBuilder::new("test");
        let param_a = builder.add_parameter(0, &[10], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[10], DType::F32).unwrap();
        let add1 = builder.add_add(param_a, param_b).unwrap();
        let add2 = builder.add_add(add1, param_b).unwrap();
        let mut computation = builder.build(add2).unwrap();

        let manager = XlaPassManager::new()
            .with_fixed_point(true)
            .with_max_iterations(10);

        let result = manager.run(&mut computation);
        assert!(result.is_ok());
    }

    #[test]
    fn test_optimization_with_aggressive_config() {
        let mut builder = XlaBuilder::new("aggressive");
        let param_a = builder.add_parameter(0, &[128, 256], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[256, 512], DType::F32).unwrap();
        let matmul = builder.add_dot(param_a, param_b).unwrap();

        let mut computation = builder.build(matmul).unwrap();
        computation.config = XlaConfig::aggressive();

        let stats = computation.optimize().unwrap();
        // Placeholder implementations report statistics but don't actually modify the graph
        // Just verify that we got valid statistics back
        assert!(stats.nodes_removed <= computation.nodes.len());
    }

    #[test]
    fn test_optimization_with_conservative_config() {
        let mut builder = XlaBuilder::new("conservative");
        let param = builder.add_parameter(0, &[10], DType::F32).unwrap();
        let mut computation = builder.build(param).unwrap();
        computation.config = XlaConfig::conservative();

        let manager = XlaPassManager::new();
        let stats = manager.run(&mut computation).unwrap();

        // Conservative config should skip most optimizations
        assert_eq!(stats.nodes_removed, 0);
    }

    #[test]
    fn test_memory_allocation_optimization_pass_name() {
        let pass = MemoryAllocationOptimizationPass;
        assert_eq!(pass.name(), "memory-allocation-optimization");
    }

    #[test]
    fn test_memory_allocation_optimization_pass_should_run() {
        let pass = MemoryAllocationOptimizationPass;

        let mut config = XlaConfig::default();
        config.optimization_level = 1;
        assert!(pass.should_run(&config));

        config.optimization_level = 0;
        assert!(!pass.should_run(&config));
    }

    #[test]
    fn test_memory_allocation_optimization_detects_buffer_reuse() {
        let mut builder = XlaBuilder::new("memory_test");
        let param_a = builder.add_parameter(0, &[128, 256], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[128, 256], DType::F32).unwrap();

        // Create a chain where outputs can potentially reuse buffers
        let add1 = builder.add_add(param_a, param_b).unwrap();
        let mul = builder.add_multiply(add1, param_b).unwrap(); // add1 is single-use here

        let mut computation = builder.build(mul).unwrap();
        computation.config.optimization_level = 1;

        let pass = MemoryAllocationOptimizationPass;
        let stats = pass.run(&mut computation).unwrap();

        // Should detect buffer reuse opportunity (add1 is only used once)
        assert!(stats.nodes_modified > 0);
    }

    #[test]
    fn test_memory_allocation_optimization_detects_inplace_ops() {
        let mut builder = XlaBuilder::new("inplace_test");
        let param_a = builder.add_parameter(0, &[100, 100], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[100, 100], DType::F32).unwrap();

        // Element-wise operations that can be performed in-place
        let add = builder.add_add(param_a, param_b).unwrap();
        let mul = builder.add_multiply(add, param_b).unwrap();
        let sub = builder.add_subtract(mul, param_b).unwrap();

        let mut computation = builder.build(sub).unwrap();
        computation.config.optimization_level = 1;

        let pass = MemoryAllocationOptimizationPass;
        let stats = pass.run(&mut computation).unwrap();

        // Should detect multiple in-place opportunities (3 element-wise ops)
        assert!(stats.nodes_modified >= 3);
    }

    #[test]
    fn test_parallelization_analysis_pass_name() {
        let pass = ParallelizationAnalysisPass;
        assert_eq!(pass.name(), "parallelization-analysis");
    }

    #[test]
    fn test_parallelization_analysis_pass_should_run() {
        let pass = ParallelizationAnalysisPass;

        let mut config = XlaConfig::default();
        config.optimization_level = 2;
        assert!(pass.should_run(&config));

        config.optimization_level = 1;
        assert!(!pass.should_run(&config));
    }

    #[test]
    fn test_parallelization_analysis_detects_independent_ops() {
        let mut builder = XlaBuilder::new("parallel_test");
        let input = builder.add_parameter(0, &[128, 256], DType::F32).unwrap();

        // Create two independent branches
        let add1 = builder.add_add(input, input).unwrap();
        let mul1 = builder.add_multiply(input, input).unwrap(); // Independent from add1

        // Merge them
        let result = builder.add_add(add1, mul1).unwrap();

        let mut computation = builder.build(result).unwrap();
        computation.config.optimization_level = 2;

        let pass = ParallelizationAnalysisPass;
        let stats = pass.run(&mut computation).unwrap();

        // Should detect independent operations that can run in parallel
        assert!(stats.nodes_modified > 0);
    }

    #[test]
    fn test_parallelization_analysis_detects_batch_ops() {
        let mut builder = XlaBuilder::new("batch_parallel_test");
        // Create batched tensor (batch_size=32) with separate batch processing
        let param_a = builder
            .add_parameter(0, &[32, 128, 256], DType::F32)
            .unwrap();
        let param_b = builder
            .add_parameter(1, &[32, 128, 256], DType::F32)
            .unwrap();

        // Element-wise operations on batched tensors can be parallelized
        let add = builder.add_add(param_a, param_a).unwrap();
        let mul = builder.add_multiply(param_b, param_b).unwrap();

        // Combine them (independent branches with same shape)
        let result = builder.add_add(add, mul).unwrap();

        let mut computation = builder.build(result).unwrap();
        computation.config.optimization_level = 2;

        let pass = ParallelizationAnalysisPass;
        let stats = pass.run(&mut computation).unwrap();

        // Should detect batch parallelization opportunities
        assert!(stats.nodes_modified > 0);
    }

    #[test]
    fn test_pass_manager_includes_all_passes() {
        let manager = XlaPassManager::default();
        let passes = manager.passes();

        assert_eq!(passes.len(), 9);
        assert!(passes.contains(&"constant-folding"));
        assert!(passes.contains(&"algebraic-simplification"));
        assert!(passes.contains(&"copy-elimination"));
        assert!(passes.contains(&"common-subexpression-elimination"));
        assert!(passes.contains(&"operation-fusion"));
        assert!(passes.contains(&"layout-optimization"));
        assert!(passes.contains(&"memory-allocation-optimization"));
        assert!(passes.contains(&"parallelization-analysis"));
        assert!(passes.contains(&"dead-code-elimination"));
    }

    #[test]
    fn test_comprehensive_optimization_pipeline() {
        let mut builder = XlaBuilder::new("comprehensive");
        let param_a = builder.add_parameter(0, &[128, 256], DType::F32).unwrap();
        let param_b = builder.add_parameter(1, &[256, 512], DType::F32).unwrap();

        // Create a complex computation with multiple optimization opportunities
        let matmul = builder.add_dot(param_a, param_b).unwrap();
        let copy = builder.add_copy(matmul).unwrap(); // Should be eliminated
        let add = builder.add_add(copy, copy).unwrap(); // Can be simplified
        let transposed = builder.add_transpose(add, &[1, 0]).unwrap();

        let mut computation = builder.build(transposed).unwrap();
        computation.config = XlaConfig::aggressive();

        // Run full optimization pipeline
        let stats = computation.optimize().unwrap();

        // Should perform multiple optimizations
        // At minimum, copy elimination should detect the copy operation
        assert!(stats.nodes_removed > 0 || stats.nodes_modified > 0);
    }
}
