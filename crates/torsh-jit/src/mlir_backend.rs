//! MLIR Backend for ToRSh JIT Compilation
//!
//! This module provides MLIR (Multi-Level Intermediate Representation) code generation
//! from ToRSh's internal IR. MLIR provides better optimization opportunities and
//! more advanced compilation techniques compared to simpler backends.
//!
//! The backend generates MLIR code in various dialects:
//! - `arith` dialect for arithmetic operations
//! - `tensor` dialect for tensor operations
//! - `linalg` dialect for linear algebra operations
//! - `func` dialect for function definitions

use crate::{
    graph::{Attribute, ComputationGraph, ConstantValue, Operation},
    JitError, JitResult, NodeId,
};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use std::fmt::Write;
use torsh_core::{DType, Shape};

/// MLIR backend for code generation
#[derive(Debug, Clone)]
pub struct MlirBackend {
    /// Generated MLIR module string
    module: String,
    /// Symbol table for variable naming
    symbol_table: HashMap<NodeId, String>,
    /// Next unique identifier
    next_id: usize,
    /// Module name
    module_name: String,
}

impl Default for MlirBackend {
    fn default() -> Self {
        Self::new("torsh_module")
    }
}

impl MlirBackend {
    /// Create a new MLIR backend
    pub fn new(module_name: &str) -> Self {
        Self {
            module: String::new(),
            symbol_table: HashMap::new(),
            next_id: 0,
            module_name: module_name.to_string(),
        }
    }

    /// Generate MLIR code from computation graph
    pub fn generate(&mut self, graph: &ComputationGraph) -> JitResult<String> {
        self.reset();

        // Generate module header
        self.generate_module_header()?;

        // Generate function signature
        self.generate_function_header(graph)?;

        // Generate constants
        self.generate_constants(graph)?;

        // Generate operations in topological order
        self.generate_operations(graph)?;

        // Generate function footer
        self.generate_function_footer(graph)?;

        // Generate module footer
        self.generate_module_footer()?;

        Ok(self.module.clone())
    }

    /// Reset the backend state
    fn reset(&mut self) {
        self.module.clear();
        self.symbol_table.clear();
        self.next_id = 0;
    }

    /// Generate MLIR module header
    fn generate_module_header(&mut self) -> JitResult<()> {
        writeln!(self.module, "// Generated MLIR module from ToRSh JIT")
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(self.module, "module @{} {{", self.module_name)
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        Ok(())
    }

    /// Generate function header
    fn generate_function_header(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        // Count input and output nodes
        let input_count = self.count_input_nodes(graph);
        let output_count = self.count_output_nodes(graph);

        write!(self.module, "  func.func @main(")
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Generate input parameters
        for i in 0..input_count {
            if i > 0 {
                write!(self.module, ", ").map_err(|e| JitError::CodeGenError(e.to_string()))?;
            }
            write!(self.module, "%arg{}: tensor<*xf32>", i)
                .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        }

        write!(self.module, ") -> ").map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Generate return type
        if output_count == 1 {
            write!(self.module, "tensor<*xf32>")
                .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        } else {
            write!(self.module, "(").map_err(|e| JitError::CodeGenError(e.to_string()))?;
            for i in 0..output_count {
                if i > 0 {
                    write!(self.module, ", ").map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
                write!(self.module, "tensor<*xf32>")
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
            }
            write!(self.module, ")").map_err(|e| JitError::CodeGenError(e.to_string()))?;
        }

        writeln!(self.module, " {{").map_err(|e| JitError::CodeGenError(e.to_string()))?;

        Ok(())
    }

    /// Generate constants
    fn generate_constants(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        for (node_id, node) in graph.nodes() {
            if let Operation::Constant(ref const_info) = node.op {
                let var_name = self.get_or_create_symbol(node_id);

                match &const_info.value {
                    ConstantValue::Scalar(val) => {
                        writeln!(
                            self.module,
                            "    %{} = arith.constant {} : f32",
                            var_name, val
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::IntScalar(val) => {
                        writeln!(
                            self.module,
                            "    %{} = arith.constant {} : i64",
                            var_name, val
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::Tensor {
                        shape: _,
                        data,
                        dtype: _,
                    } => {
                        // Generate tensor constant
                        let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                        writeln!(
                            self.module,
                            "    %{} = arith.constant dense<{}> : tensor<{}xf32>",
                            var_name,
                            self.format_tensor_data(&f32_data),
                            data.len()
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::Bool(val) => {
                        writeln!(
                            self.module,
                            "    %{} = arith.constant {} : i1",
                            var_name,
                            if *val { "true" } else { "false" }
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::Int(val) => {
                        writeln!(
                            self.module,
                            "    %{} = arith.constant {} : i64",
                            var_name, val
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::UInt(val) => {
                        writeln!(
                            self.module,
                            "    %{} = arith.constant {} : ui64",
                            var_name, val
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::Float(val) => {
                        writeln!(
                            self.module,
                            "    %{} = arith.constant {} : f64",
                            var_name, val
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::String(val) => {
                        writeln!(
                            self.module,
                            "    %{} = arith.constant \"{}\" : !llvm.ptr<i8>",
                            var_name,
                            val.replace('"', "\\\"")
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::FloatArray(arr) => {
                        writeln!(
                            self.module,
                            "    %{} = arith.constant dense<{}> : tensor<{}xf32>",
                            var_name,
                            self.format_tensor_data(arr),
                            arr.len()
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::IntArray(arr) => {
                        let arr_str = arr
                            .iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        writeln!(
                            self.module,
                            "    %{} = arith.constant dense<[{}]> : tensor<{}xi64>",
                            var_name,
                            arr_str,
                            arr.len()
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::Array(_) => {
                        // Complex arrays are represented as generic tensors
                        writeln!(
                            self.module,
                            "    %{} = arith.constant dense<0.0> : tensor<1xf32>",
                            var_name
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::Complex { real, imag } => {
                        writeln!(
                            self.module,
                            "    %{} = complex.constant [{}, {}] : complex<f64>",
                            var_name, real, imag
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::None => {
                        writeln!(self.module, "    %{} = llvm.null : !llvm.ptr<i8>", var_name)
                            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::Undefined => {
                        writeln!(
                            self.module,
                            "    %{} = llvm.undef : !llvm.ptr<i8>",
                            var_name
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Generate operations
    fn generate_operations(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        // Get nodes in topological order
        let topo_order = self.topological_sort(graph)?;

        for node_id in topo_order {
            if let Some(node) = graph.node(node_id) {
                if !matches!(node.op, Operation::Constant(_)) {
                    self.generate_operation(graph, node_id, node)?;
                }
            }
        }

        Ok(())
    }

    /// Generate a single operation
    fn generate_operation(
        &mut self,
        graph: &ComputationGraph,
        node_id: NodeId,
        node: &crate::Node,
    ) -> JitResult<()> {
        let output_var = self.get_or_create_symbol(node_id);
        let inputs = self.get_input_symbols(graph, node_id);

        match &node.op {
            Operation::Add => {
                if inputs.len() >= 2 {
                    writeln!(
                        self.module,
                        "    %{} = arith.addf %{}, %{} : tensor<*xf32>",
                        output_var, inputs[0], inputs[1]
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
            Operation::Sub => {
                if inputs.len() >= 2 {
                    writeln!(
                        self.module,
                        "    %{} = arith.subf %{}, %{} : tensor<*xf32>",
                        output_var, inputs[0], inputs[1]
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
            Operation::Mul => {
                if inputs.len() >= 2 {
                    writeln!(
                        self.module,
                        "    %{} = arith.mulf %{}, %{} : tensor<*xf32>",
                        output_var, inputs[0], inputs[1]
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
            Operation::Div => {
                if inputs.len() >= 2 {
                    writeln!(
                        self.module,
                        "    %{} = arith.divf %{}, %{} : tensor<*xf32>",
                        output_var, inputs[0], inputs[1]
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
            Operation::MatMul => {
                if inputs.len() >= 2 {
                    writeln!(
                        self.module,
                        "    %{} = linalg.matmul ins(%{}, %{} : tensor<*xf32>, tensor<*xf32>) outs(%{} : tensor<*xf32>) -> tensor<*xf32>",
                        output_var, inputs[0], inputs[1], output_var
                    ).map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
            Operation::Relu => {
                if !inputs.is_empty() {
                    // Generate zero constant for comparison
                    let zero_var = format!("zero_{}", self.next_id);
                    self.next_id += 1;
                    writeln!(self.module, "    %{} = arith.constant 0.0 : f32", zero_var)
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                    writeln!(
                        self.module,
                        "    %{} = arith.maximumf %{}, %{} : tensor<*xf32>",
                        output_var, inputs[0], zero_var
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
            Operation::Sigmoid => {
                if !inputs.is_empty() {
                    // sigmoid(x) = 1 / (1 + exp(-x))
                    let neg_var = format!("neg_{}", self.next_id);
                    let exp_var = format!("exp_{}", self.next_id);
                    let one_var = format!("one_{}", self.next_id);
                    let add_var = format!("add_{}", self.next_id);
                    self.next_id += 4;

                    writeln!(
                        self.module,
                        "    %{} = arith.negf %{} : tensor<*xf32>",
                        neg_var, inputs[0]
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                    writeln!(
                        self.module,
                        "    %{} = math.exp %{} : tensor<*xf32>",
                        exp_var, neg_var
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                    writeln!(self.module, "    %{} = arith.constant 1.0 : f32", one_var)
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                    writeln!(
                        self.module,
                        "    %{} = arith.addf %{}, %{} : tensor<*xf32>",
                        add_var, one_var, exp_var
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                    writeln!(
                        self.module,
                        "    %{} = arith.divf %{}, %{} : tensor<*xf32>",
                        output_var, one_var, add_var
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
            Operation::Tanh => {
                if !inputs.is_empty() {
                    writeln!(
                        self.module,
                        "    %{} = math.tanh %{} : tensor<*xf32>",
                        output_var, inputs[0]
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
            Operation::Exp => {
                if !inputs.is_empty() {
                    writeln!(
                        self.module,
                        "    %{} = math.exp %{} : tensor<*xf32>",
                        output_var, inputs[0]
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
            Operation::Log => {
                if !inputs.is_empty() {
                    writeln!(
                        self.module,
                        "    %{} = math.log %{} : tensor<*xf32>",
                        output_var, inputs[0]
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
            Operation::Reshape { .. } => {
                if !inputs.is_empty() {
                    let shape_str = self.format_shape(&node.output_shape);
                    writeln!(
                        self.module,
                        "    %{} = tensor.reshape %{} : (tensor<*xf32>) -> tensor<{}xf32>",
                        output_var, inputs[0], shape_str
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
            Operation::Transpose { .. } => {
                if !inputs.is_empty() {
                    // Get permutation from attributes if available
                    let perm = if let Some(Attribute::IntList(perm)) = node.attrs.get("permutation")
                    {
                        perm.iter()
                            .map(|&x| x.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    } else {
                        "1, 0".to_string() // Default 2D transpose
                    };

                    writeln!(
                        self.module,
                        "    %{} = linalg.transpose ins(%{} : tensor<*xf32>) outs(%{} : tensor<*xf32>) permutation = [{}]",
                        output_var, inputs[0], output_var, perm
                    ).map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
            Operation::Custom(op_name) => {
                let input_str = inputs.join(", ");
                writeln!(
                    self.module,
                    "    %{} = call @custom_{}({}) : ({}) -> tensor<*xf32>",
                    output_var,
                    op_name,
                    input_str,
                    inputs
                        .iter()
                        .map(|_| "tensor<*xf32>")
                        .collect::<Vec<_>>()
                        .join(", ")
                )
                .map_err(|e| JitError::CodeGenError(e.to_string()))?;
            }
            _ => {
                // Generic operation - generate a placeholder
                let input_str = inputs.join(", ");
                writeln!(self.module, "    // Unsupported operation: {:?}", node.op)
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                if !inputs.is_empty() {
                    writeln!(
                        self.module,
                        "    %{} = %{} // placeholder",
                        output_var, inputs[0]
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
        }

        Ok(())
    }

    /// Generate function footer with return statement
    fn generate_function_footer(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        // Find output nodes (nodes with no outgoing edges)
        let mut output_vars = Vec::new();

        for (node_id, _) in graph.nodes() {
            let has_outgoing = graph
                .edges_directed(node_id, petgraph::Direction::Outgoing)
                .next()
                .is_some();
            if !has_outgoing {
                if let Some(var_name) = self.symbol_table.get(&node_id) {
                    output_vars.push(var_name.clone());
                }
            }
        }

        // Generate return statement
        if output_vars.is_empty() {
            writeln!(self.module, "    return")
                .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        } else if output_vars.len() == 1 {
            writeln!(
                self.module,
                "    return %{} : tensor<*xf32>",
                output_vars[0]
            )
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        } else {
            let return_vars = output_vars
                .iter()
                .map(|v| format!("%{}", v))
                .collect::<Vec<_>>()
                .join(", ");
            let return_types = output_vars
                .iter()
                .map(|_| "tensor<*xf32>")
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(self.module, "    return {} : {}", return_vars, return_types)
                .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        }

        writeln!(self.module, "  }}").map_err(|e| JitError::CodeGenError(e.to_string()))?;
        Ok(())
    }

    /// Generate module footer
    fn generate_module_footer(&mut self) -> JitResult<()> {
        writeln!(self.module, "}}").map_err(|e| JitError::CodeGenError(e.to_string()))?;
        Ok(())
    }

    /// Get or create a symbol for a node
    fn get_or_create_symbol(&mut self, node_id: NodeId) -> String {
        if let Some(symbol) = self.symbol_table.get(&node_id) {
            symbol.clone()
        } else {
            let symbol = format!("v{}", self.next_id);
            self.next_id += 1;
            self.symbol_table.insert(node_id, symbol.clone());
            symbol
        }
    }

    /// Get input symbols for a node
    fn get_input_symbols(&mut self, graph: &ComputationGraph, node_id: NodeId) -> Vec<String> {
        let mut inputs = Vec::new();

        for edge_ref in graph
            .graph
            .edges_directed(node_id, petgraph::Direction::Incoming)
        {
            let src_id = edge_ref.source();
            let symbol = self.get_or_create_symbol(src_id);
            inputs.push(symbol);
        }

        inputs
    }

    /// Perform topological sort of the graph
    fn topological_sort(&self, graph: &ComputationGraph) -> JitResult<Vec<NodeId>> {
        use petgraph::algo::toposort;

        toposort(&graph.graph, None)
            .map_err(|_| JitError::GraphError("Cyclic graph detected".to_string()))
    }

    /// Count input nodes in the graph
    fn count_input_nodes(&self, graph: &ComputationGraph) -> usize {
        graph
            .nodes()
            .filter(|(node_id, _)| {
                graph
                    .edges_directed(*node_id, petgraph::Direction::Incoming)
                    .next()
                    .is_none()
            })
            .count()
    }

    /// Count output nodes in the graph
    fn count_output_nodes(&self, graph: &ComputationGraph) -> usize {
        graph
            .nodes()
            .filter(|(node_id, _)| {
                graph
                    .edges_directed(*node_id, petgraph::Direction::Outgoing)
                    .next()
                    .is_none()
            })
            .count()
    }

    /// Format tensor data for MLIR
    fn format_tensor_data(&self, data: &[f32]) -> String {
        if data.len() == 1 {
            data[0].to_string()
        } else {
            format!(
                "[{}]",
                data.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }

    /// Format shape for MLIR type
    fn format_shape(&self, shape: &Shape) -> String {
        if shape.dims().is_empty() {
            "*".to_string()
        } else {
            shape
                .dims()
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x")
        }
    }
}

/// MLIR optimization passes
#[derive(Debug, Clone)]
pub struct MlirOptimizer {
    /// Optimization level (0-3)
    opt_level: u8,
    /// Enable specific passes
    passes: Vec<MlirPass>,
}

/// MLIR optimization pass
#[derive(Debug, Clone)]
pub enum MlirPass {
    /// Canonicalization pass
    Canonicalize,
    /// Common subexpression elimination
    Cse,
    /// Dead code elimination
    Dce,
    /// Constant folding
    ConstantFold,
    /// Loop optimization
    LoopOptimize,
    /// Buffer optimization
    BufferOptimize,
    /// Tensor optimization
    TensorOptimize,
    /// Arithmetic simplification
    ArithSimplify,
}

impl Default for MlirOptimizer {
    fn default() -> Self {
        Self::new(2)
    }
}

impl MlirOptimizer {
    /// Create a new MLIR optimizer
    pub fn new(opt_level: u8) -> Self {
        let passes = match opt_level {
            0 => vec![],
            1 => vec![MlirPass::Canonicalize, MlirPass::Dce],
            2 => vec![
                MlirPass::Canonicalize,
                MlirPass::Cse,
                MlirPass::Dce,
                MlirPass::ConstantFold,
                MlirPass::ArithSimplify,
            ],
            _ => vec![
                MlirPass::Canonicalize,
                MlirPass::Cse,
                MlirPass::Dce,
                MlirPass::ConstantFold,
                MlirPass::ArithSimplify,
                MlirPass::LoopOptimize,
                MlirPass::BufferOptimize,
                MlirPass::TensorOptimize,
            ],
        };

        Self { opt_level, passes }
    }

    /// Add a specific optimization pass
    pub fn add_pass(&mut self, pass: MlirPass) {
        self.passes.push(pass);
    }

    /// Apply optimization passes to MLIR code
    pub fn optimize(&self, mlir_code: &str) -> JitResult<String> {
        let mut optimized = mlir_code.to_string();

        for pass in &self.passes {
            optimized = self.apply_pass(&optimized, pass)?;
        }

        Ok(optimized)
    }

    /// Apply a single optimization pass
    fn apply_pass(&self, mlir_code: &str, pass: &MlirPass) -> JitResult<String> {
        match pass {
            MlirPass::Canonicalize => self.canonicalize(mlir_code),
            MlirPass::Cse => self.common_subexpression_elimination(mlir_code),
            MlirPass::Dce => self.dead_code_elimination(mlir_code),
            MlirPass::ConstantFold => self.constant_folding(mlir_code),
            MlirPass::ArithSimplify => self.arithmetic_simplification(mlir_code),
            _ => Ok(mlir_code.to_string()), // Other passes not implemented yet
        }
    }

    /// Canonicalization pass
    fn canonicalize(&self, mlir_code: &str) -> JitResult<String> {
        // Simple canonicalization - remove redundant operations
        let mut result = mlir_code.to_string();

        // Remove identity operations like x + 0, x * 1
        result = result.replace("arith.addf %v, %zero", "%v");
        result = result.replace("arith.mulf %v, %one", "%v");

        Ok(result)
    }

    /// Common subexpression elimination
    fn common_subexpression_elimination(&self, mlir_code: &str) -> JitResult<String> {
        // Basic CSE implementation
        let mut seen_expressions = HashMap::new();
        let mut result = String::new();

        for line in mlir_code.lines() {
            if line.trim().starts_with('%') && (line.contains("arith.") || line.contains("math.")) {
                // Extract the operation part
                if let Some(eq_pos) = line.find('=') {
                    let var_part = line[..eq_pos].trim();
                    let op_part = line[eq_pos + 1..].trim();

                    if let Some(existing_var) = seen_expressions.get(op_part) {
                        // Replace with existing variable
                        result.push_str(&format!("    {} = {} // CSE\n", var_part, existing_var));
                    } else {
                        seen_expressions.insert(op_part.to_string(), var_part.to_string());
                        result.push_str(line);
                        result.push('\n');
                    }
                } else {
                    result.push_str(line);
                    result.push('\n');
                }
            } else {
                result.push_str(line);
                result.push('\n');
            }
        }

        Ok(result)
    }

    /// Dead code elimination
    fn dead_code_elimination(&self, mlir_code: &str) -> JitResult<String> {
        // Simple DCE - remove unused variables
        let mut used_vars = std::collections::HashSet::new();
        let lines: Vec<&str> = mlir_code.lines().collect();

        // First pass: find all used variables
        for line in &lines {
            // Look for variable usage (not definition)
            let parts: Vec<&str> = line.split_whitespace().collect();
            for part in parts {
                if part.starts_with('%') && !line.trim_start().starts_with(&part[1..]) {
                    used_vars.insert(part.to_string());
                }
            }
        }

        // Second pass: keep only used definitions
        let mut result = String::new();
        for line in lines {
            if line.trim().starts_with('%') && line.contains('=') {
                if let Some(var_end) = line.find('=') {
                    let var_part = line[..var_end].trim();
                    if used_vars.contains(var_part) {
                        result.push_str(line);
                        result.push('\n');
                    }
                    // Else skip unused definition
                } else {
                    result.push_str(line);
                    result.push('\n');
                }
            } else {
                result.push_str(line);
                result.push('\n');
            }
        }

        Ok(result)
    }

    /// Constant folding pass
    fn constant_folding(&self, mlir_code: &str) -> JitResult<String> {
        let result = mlir_code.to_string();

        // Simple constant folding patterns
        // This is a basic implementation - a real one would need proper parsing

        // Fold addition of constants
        if result.contains("arith.constant") && result.contains("arith.addf") {
            // This would need proper MLIR parsing to implement correctly
            // For now, just return the original code
        }

        Ok(result)
    }

    /// Arithmetic simplification pass
    fn arithmetic_simplification(&self, mlir_code: &str) -> JitResult<String> {
        let mut result = mlir_code.to_string();

        // Simplify arithmetic operations
        // x + 0 = x
        result = result.replace(
            "arith.addf %v, %zero : tensor<*xf32>",
            "// simplified: x + 0 = x\n    // %result = %v",
        );

        // x * 1 = x
        result = result.replace(
            "arith.mulf %v, %one : tensor<*xf32>",
            "// simplified: x * 1 = x\n    // %result = %v",
        );

        // x * 0 = 0
        result = result.replace(
            "arith.mulf %v, %zero : tensor<*xf32>",
            "// simplified: x * 0 = 0\n    // %result = %zero",
        );

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ComputationGraph, ConstantInfo, Operation};
    use torsh_core::DeviceType;

    #[test]
    fn test_mlir_backend_creation() {
        let backend = MlirBackend::new("test_module");
        assert_eq!(backend.module_name, "test_module");
        assert!(backend.symbol_table.is_empty());
    }

    #[test]
    fn test_mlir_optimizer() {
        let optimizer = MlirOptimizer::new(2);
        assert_eq!(optimizer.opt_level, 2);
        assert!(!optimizer.passes.is_empty());
    }

    #[test]
    fn test_simple_mlir_generation() {
        let mut backend = MlirBackend::new("test");
        let mut graph = ComputationGraph::new();

        // Add a simple constant node
        let node = crate::Node::new(
            Operation::Constant(ConstantInfo {
                value: ConstantValue::Scalar(42.0),
            }),
            "const1".to_string(),
        )
        .with_output_shapes(vec![Some(Shape::new(vec![1]))])
        .with_dtypes(vec![DType::F32])
        .with_device(DeviceType::Cpu);

        graph.add_node(node);

        let result = backend.generate(&graph);
        assert!(result.is_ok());

        let mlir_code = result.unwrap();
        assert!(mlir_code.contains("module @test"));
        assert!(mlir_code.contains("arith.constant 42"));
    }
}
