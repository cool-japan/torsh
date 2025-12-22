//! LLVM Backend for ToRSh JIT Compilation
//!
//! This module provides LLVM IR code generation from ToRSh's internal IR.
//! LLVM provides excellent optimization capabilities and target-specific
//! code generation for maximum performance.
//!
//! The backend generates LLVM IR with:
//! - Optimized tensor operations
//! - Target-specific optimizations
//! - Auto-vectorization support
//! - Advanced optimization passes

use crate::{
    graph::{ComputationGraph, ConstantValue, Operation},
    JitError, JitResult, NodeId,
};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use std::fmt::Write;

/// LLVM backend for code generation
#[derive(Debug, Clone)]
pub struct LlvmBackend {
    /// Generated LLVM IR module string
    module: String,
    /// Symbol table for variable naming
    symbol_table: HashMap<NodeId, String>,
    /// Next unique identifier
    next_id: usize,
    /// Module name
    module_name: String,
    /// Target triple
    target_triple: String,
    /// Data layout
    data_layout: String,
}

impl Default for LlvmBackend {
    fn default() -> Self {
        Self::new("torsh_module")
    }
}

impl LlvmBackend {
    /// Create a new LLVM backend
    pub fn new(module_name: &str) -> Self {
        Self {
            module: String::new(),
            symbol_table: HashMap::new(),
            next_id: 0,
            module_name: module_name.to_string(),
            target_triple: "x86_64-unknown-linux-gnu".to_string(),
            data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
                .to_string(),
        }
    }

    /// Set target triple
    pub fn set_target_triple(&mut self, triple: String) {
        self.target_triple = triple;
    }

    /// Set data layout
    pub fn set_data_layout(&mut self, layout: String) {
        self.data_layout = layout;
    }

    /// Generate LLVM IR code from computation graph
    pub fn generate(&mut self, graph: &ComputationGraph) -> JitResult<String> {
        self.reset();

        // Generate module header
        self.generate_module_header()?;

        // Generate global declarations
        self.generate_global_declarations()?;

        // Generate main function
        self.generate_main_function(graph)?;

        // Generate helper functions
        self.generate_helper_functions()?;

        Ok(self.module.clone())
    }

    /// Reset the backend state
    fn reset(&mut self) {
        self.module.clear();
        self.symbol_table.clear();
        self.next_id = 0;
    }

    /// Generate LLVM module header
    fn generate_module_header(&mut self) -> JitResult<()> {
        writeln!(self.module, "; Generated LLVM IR module from ToRSh JIT")
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(self.module, "; Module: {}", self.module_name)
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(self.module, "").map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(self.module, "target triple = \"{}\"", self.target_triple)
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(self.module, "target datalayout = \"{}\"", self.data_layout)
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(self.module, "").map_err(|e| JitError::CodeGenError(e.to_string()))?;

        Ok(())
    }

    /// Generate global declarations
    fn generate_global_declarations(&mut self) -> JitResult<()> {
        // Declare external functions for tensor operations
        writeln!(self.module, "; External function declarations")
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Math functions
        writeln!(self.module, "declare float @expf(float)")
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(self.module, "declare float @logf(float)")
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(self.module, "declare float @tanhf(float)")
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(self.module, "declare float @sqrtf(float)")
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Vector math functions
        writeln!(
            self.module,
            "declare <4 x float> @llvm.exp.v4f32(<4 x float>)"
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(
            self.module,
            "declare <4 x float> @llvm.log.v4f32(<4 x float>)"
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(
            self.module,
            "declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)"
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Memory functions
        writeln!(self.module, "declare i8* @malloc(i64)")
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(self.module, "declare void @free(i8*)")
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(
            self.module,
            "declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)"
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(self.module, "").map_err(|e| JitError::CodeGenError(e.to_string()))?;

        Ok(())
    }

    /// Generate main function
    fn generate_main_function(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        // Function signature
        let input_count = self.count_input_nodes(graph);
        let output_count = self.count_output_nodes(graph);

        write!(self.module, "define ").map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Return type
        if output_count == 1 {
            write!(self.module, "float* ").map_err(|e| JitError::CodeGenError(e.to_string()))?;
        } else {
            write!(self.module, "{{").map_err(|e| JitError::CodeGenError(e.to_string()))?;
            for i in 0..output_count {
                if i > 0 {
                    write!(self.module, ", ").map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
                write!(self.module, "float*").map_err(|e| JitError::CodeGenError(e.to_string()))?;
            }
            write!(self.module, "}} ").map_err(|e| JitError::CodeGenError(e.to_string()))?;
        }

        write!(self.module, "@main(").map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Input parameters
        for i in 0..input_count {
            if i > 0 {
                write!(self.module, ", ").map_err(|e| JitError::CodeGenError(e.to_string()))?;
            }
            write!(self.module, "float* %input{}, i64 %size{}", i, i)
                .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        }

        writeln!(self.module, ") {{").map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Function body
        self.generate_function_body(graph)?;

        writeln!(self.module, "}}").map_err(|e| JitError::CodeGenError(e.to_string()))?;

        Ok(())
    }

    /// Generate function body
    fn generate_function_body(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        writeln!(self.module, "entry:").map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Generate constants
        self.generate_llvm_constants(graph)?;

        // Generate operations in topological order
        let topo_order = self.topological_sort(graph)?;

        for node_id in topo_order {
            if let Some(node) = graph.node(node_id) {
                if !matches!(node.op, Operation::Constant(_)) {
                    self.generate_llvm_operation(graph, node_id, node)?;
                }
            }
        }

        // Generate return statement
        self.generate_return_statement(graph)?;

        Ok(())
    }

    /// Generate LLVM constants
    fn generate_llvm_constants(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        for (node_id, node) in graph.nodes() {
            if let Operation::Constant(ref const_info) = node.op {
                let var_name = self.get_or_create_symbol(node_id);

                match &const_info.value {
                    ConstantValue::Scalar(val) => {
                        // Allocate space for scalar
                        writeln!(self.module, "  %{}_ptr = alloca float, align 4", var_name)
                            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                        writeln!(
                            self.module,
                            "  store float {}, float* %{}_ptr, align 4",
                            val, var_name
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::IntScalar(val) => {
                        writeln!(self.module, "  %{}_ptr = alloca i64, align 8", var_name)
                            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                        writeln!(
                            self.module,
                            "  store i64 {}, i64* %{}_ptr, align 8",
                            val, var_name
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::Tensor {
                        shape: _,
                        data,
                        dtype: _,
                    } => {
                        // Allocate space for tensor data
                        let size = data.len();
                        writeln!(self.module, "  %{}_size = add i64 0, {}", var_name, size)
                            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                        writeln!(
                            self.module,
                            "  %{}_bytes = mul i64 %{}_size, 4",
                            var_name, var_name
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                        writeln!(
                            self.module,
                            "  %{}_raw = call i8* @malloc(i64 %{}_bytes)",
                            var_name, var_name
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                        writeln!(
                            self.module,
                            "  %{}_ptr = bitcast i8* %{}_raw to float*",
                            var_name, var_name
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                        // Initialize tensor data
                        for (i, &value) in data.iter().enumerate() {
                            writeln!(
                                self.module,
                                "  %{}_elem{}_ptr = getelementptr inbounds float, float* %{}_ptr, i64 {}",
                                var_name, i, var_name, i
                            ).map_err(|e| JitError::CodeGenError(e.to_string()))?;

                            writeln!(
                                self.module,
                                "  store float {}, float* %{}_elem{}_ptr, align 4",
                                value, var_name, i
                            )
                            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                        }
                    }
                    ConstantValue::Bool(val) => {
                        writeln!(self.module, "  %{}_ptr = alloca i1, align 1", var_name)
                            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                        writeln!(
                            self.module,
                            "  store i1 {}, i1* %{}_ptr, align 1",
                            if *val { "true" } else { "false" },
                            var_name
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::Int(val) => {
                        writeln!(self.module, "  %{}_ptr = alloca i64, align 8", var_name)
                            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                        writeln!(
                            self.module,
                            "  store i64 {}, i64* %{}_ptr, align 8",
                            val, var_name
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    ConstantValue::Float(val) => {
                        writeln!(self.module, "  %{}_ptr = alloca double, align 8", var_name)
                            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                        writeln!(
                            self.module,
                            "  store double {}, double* %{}_ptr, align 8",
                            val, var_name
                        )
                        .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                    // Handle remaining variants with default behavior
                    _ => {
                        writeln!(self.module, "  ; Unhandled constant type for {}", var_name)
                            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Generate LLVM operation
    fn generate_llvm_operation(
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
                    self.generate_elementwise_binary_op(&output_var, &inputs, "fadd")?;
                }
            }
            Operation::Sub => {
                if inputs.len() >= 2 {
                    self.generate_elementwise_binary_op(&output_var, &inputs, "fsub")?;
                }
            }
            Operation::Mul => {
                if inputs.len() >= 2 {
                    self.generate_elementwise_binary_op(&output_var, &inputs, "fmul")?;
                }
            }
            Operation::Div => {
                if inputs.len() >= 2 {
                    self.generate_elementwise_binary_op(&output_var, &inputs, "fdiv")?;
                }
            }
            Operation::MatMul => {
                if inputs.len() >= 2 {
                    self.generate_matmul(&output_var, &inputs)?;
                }
            }
            Operation::Relu => {
                if !inputs.is_empty() {
                    self.generate_relu(&output_var, &inputs[0])?;
                }
            }
            Operation::Sigmoid => {
                if !inputs.is_empty() {
                    self.generate_sigmoid(&output_var, &inputs[0])?;
                }
            }
            Operation::Tanh => {
                if !inputs.is_empty() {
                    self.generate_tanh(&output_var, &inputs[0])?;
                }
            }
            Operation::Exp => {
                if !inputs.is_empty() {
                    self.generate_unary_math_op(&output_var, &inputs[0], "exp")?;
                }
            }
            Operation::Log => {
                if !inputs.is_empty() {
                    self.generate_unary_math_op(&output_var, &inputs[0], "log")?;
                }
            }
            Operation::Neg => {
                if !inputs.is_empty() {
                    self.generate_neg(&output_var, &inputs[0])?;
                }
            }
            _ => {
                // Generic operation - generate a placeholder
                writeln!(self.module, "  ; Unsupported operation: {:?}", node.op)
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                if !inputs.is_empty() {
                    writeln!(
                        self.module,
                        "  %{}_ptr = alloca float*, align 8",
                        output_var
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;

                    writeln!(
                        self.module,
                        "  store float* %{}_ptr, float** %{}_ptr",
                        inputs[0], output_var
                    )
                    .map_err(|e| JitError::CodeGenError(e.to_string()))?;
                }
            }
        }

        Ok(())
    }

    /// Generate elementwise binary operation
    fn generate_elementwise_binary_op(
        &mut self,
        output_var: &str,
        inputs: &[String],
        op: &str,
    ) -> JitResult<()> {
        // Assume inputs have the same size for simplicity
        let size_var = format!("{}_size", output_var);
        let loop_var = format!("{}_loop", output_var);
        let cond_var = format!("{}_cond", output_var);
        let next_var = format!("{}_next", output_var);

        // Allocate output tensor
        writeln!(
            self.module,
            "  %{} = add i64 0, 1024 ; placeholder size",
            size_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{}_bytes = mul i64 %{}, 4",
            output_var, size_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{}_raw = call i8* @malloc(i64 %{}_bytes)",
            output_var, output_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{}_ptr = bitcast i8* %{}_raw to float*",
            output_var, output_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Initialize loop
        writeln!(self.module, "  br label %{}_head", loop_var)
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(self.module, "{}:", format!("{}_head", loop_var))
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{}_i = phi i64 [ 0, %entry ], [ %{}, %{}_body ]",
            loop_var, next_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{} = icmp ult i64 %{}_i, %{}",
            cond_var, loop_var, size_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  br i1 %{}, label %{}_body, label %{}_end",
            cond_var, loop_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Loop body
        writeln!(self.module, "{}:", format!("{}_body", loop_var))
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Load operands
        writeln!(
            self.module,
            "  %{}_a_ptr = getelementptr inbounds float, float* %{}_ptr, i64 %{}_i",
            loop_var, inputs[0], loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{}_a = load float, float* %{}_a_ptr, align 4",
            loop_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{}_b_ptr = getelementptr inbounds float, float* %{}_ptr, i64 %{}_i",
            loop_var, inputs[1], loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{}_b = load float, float* %{}_b_ptr, align 4",
            loop_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Perform operation
        writeln!(
            self.module,
            "  %{}_result = {} float %{}_a, %{}_b",
            loop_var, op, loop_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Store result
        writeln!(
            self.module,
            "  %{}_out_ptr = getelementptr inbounds float, float* %{}_ptr, i64 %{}_i",
            loop_var, output_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  store float %{}_result, float* %{}_out_ptr, align 4",
            loop_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Increment and branch
        writeln!(self.module, "  %{} = add i64 %{}_i, 1", next_var, loop_var)
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(self.module, "  br label %{}_head", loop_var)
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Loop end
        writeln!(self.module, "{}:", format!("{}_end", loop_var))
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        Ok(())
    }

    /// Generate matrix multiplication
    fn generate_matmul(&mut self, output_var: &str, inputs: &[String]) -> JitResult<()> {
        // Simplified matrix multiplication - in practice would need proper size handling
        writeln!(
            self.module,
            "  ; Matrix multiplication: {} = {} * {}",
            output_var, inputs[0], inputs[1]
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Placeholder implementation
        writeln!(
            self.module,
            "  %{}_ptr = call float* @matmul_impl(float* %{}_ptr, float* %{}_ptr)",
            output_var, inputs[0], inputs[1]
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        Ok(())
    }

    /// Generate ReLU operation
    fn generate_relu(&mut self, output_var: &str, input_var: &str) -> JitResult<()> {
        self.generate_elementwise_unary_op(output_var, input_var, |module, i, input, output| {
            writeln!(module, "  %{}_zero = add float 0.0, 0.0", i)?;
            writeln!(
                module,
                "  %{}_cmp = fcmp ogt float %{}, %{}_zero",
                i, input, i
            )?;
            writeln!(
                module,
                "  %{} = select i1 %{}_cmp, float %{}, float %{}_zero",
                output, i, input, i
            )?;
            Ok(())
        })
    }

    /// Generate sigmoid operation
    fn generate_sigmoid(&mut self, output_var: &str, input_var: &str) -> JitResult<()> {
        self.generate_elementwise_unary_op(output_var, input_var, |module, i, input, output| {
            writeln!(module, "  %{}_neg = fsub float 0.0, %{}", i, input)?;
            writeln!(module, "  %{}_exp = call float @expf(float %{}_neg)", i, i)?;
            writeln!(module, "  %{}_one = add float 1.0, 0.0", i)?;
            writeln!(module, "  %{}_sum = fadd float %{}_one, %{}_exp", i, i, i)?;
            writeln!(module, "  %{} = fdiv float %{}_one, %{}_sum", output, i, i)?;
            Ok(())
        })
    }

    /// Generate tanh operation
    fn generate_tanh(&mut self, output_var: &str, input_var: &str) -> JitResult<()> {
        self.generate_elementwise_unary_op(output_var, input_var, |module, i, input, output| {
            writeln!(
                module,
                "  %{} = call float @tanhf(float %{})",
                output, input
            )?;
            Ok(())
        })
    }

    /// Generate unary math operation
    fn generate_unary_math_op(
        &mut self,
        output_var: &str,
        input_var: &str,
        op: &str,
    ) -> JitResult<()> {
        self.generate_elementwise_unary_op(output_var, input_var, |module, i, input, output| {
            writeln!(
                module,
                "  %{} = call float @{}f(float %{})",
                output, op, input
            )?;
            Ok(())
        })
    }

    /// Generate negation operation
    fn generate_neg(&mut self, output_var: &str, input_var: &str) -> JitResult<()> {
        self.generate_elementwise_unary_op(output_var, input_var, |module, i, input, output| {
            writeln!(module, "  %{} = fsub float 0.0, %{}", output, input)?;
            Ok(())
        })
    }

    /// Generate elementwise unary operation with custom body
    fn generate_elementwise_unary_op<F>(
        &mut self,
        output_var: &str,
        input_var: &str,
        body_gen: F,
    ) -> JitResult<()>
    where
        F: Fn(&mut String, &str, &str, &str) -> Result<(), std::fmt::Error>,
    {
        let size_var = format!("{}_size", output_var);
        let loop_var = format!("{}_loop", output_var);
        let cond_var = format!("{}_cond", output_var);
        let next_var = format!("{}_next", output_var);

        // Allocate output tensor
        writeln!(
            self.module,
            "  %{} = add i64 0, 1024 ; placeholder size",
            size_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{}_bytes = mul i64 %{}, 4",
            output_var, size_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{}_raw = call i8* @malloc(i64 %{}_bytes)",
            output_var, output_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{}_ptr = bitcast i8* %{}_raw to float*",
            output_var, output_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Loop structure similar to binary ops
        writeln!(self.module, "  br label %{}_head", loop_var)
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(self.module, "{}:", format!("{}_head", loop_var))
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{}_i = phi i64 [ 0, %entry ], [ %{}, %{}_body ]",
            loop_var, next_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{} = icmp ult i64 %{}_i, %{}",
            cond_var, loop_var, size_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  br i1 %{}, label %{}_body, label %{}_end",
            cond_var, loop_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Loop body
        writeln!(self.module, "{}:", format!("{}_body", loop_var))
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Load input
        writeln!(
            self.module,
            "  %{}_in_ptr = getelementptr inbounds float, float* %{}_ptr, i64 %{}_i",
            loop_var, input_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  %{}_in = load float, float* %{}_in_ptr, align 4",
            loop_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Generate custom operation body
        body_gen(
            &mut self.module,
            &loop_var,
            &format!("{}_in", loop_var),
            &format!("{}_result", loop_var),
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Store result
        writeln!(
            self.module,
            "  %{}_out_ptr = getelementptr inbounds float, float* %{}_ptr, i64 %{}_i",
            loop_var, output_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(
            self.module,
            "  store float %{}_result, float* %{}_out_ptr, align 4",
            loop_var, loop_var
        )
        .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Increment and branch
        writeln!(self.module, "  %{} = add i64 %{}_i, 1", next_var, loop_var)
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        writeln!(self.module, "  br label %{}_head", loop_var)
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Loop end
        writeln!(self.module, "{}:", format!("{}_end", loop_var))
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        Ok(())
    }

    /// Generate return statement
    fn generate_return_statement(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        // Find output nodes
        let mut output_vars = Vec::new();

        for (node_id, _) in graph.nodes() {
            let has_outgoing = graph
                .edges_directed(node_id, petgraph::Direction::Outgoing)
                .next()
                .is_some();
            if !has_outgoing {
                if let Some(var_name) = self.symbol_table.get(&node_id) {
                    output_vars.push(format!("{}_ptr", var_name));
                }
            }
        }

        if output_vars.is_empty() {
            writeln!(self.module, "  ret void")
                .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        } else if output_vars.len() == 1 {
            writeln!(self.module, "  ret float* %{}", output_vars[0])
                .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        } else {
            // Multiple outputs - create a struct
            writeln!(
                self.module,
                "  %result = alloca {{{}}}*, align 8",
                output_vars
                    .iter()
                    .map(|_| "float*")
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

            for (i, var) in output_vars.iter().enumerate() {
                writeln!(
                    self.module,
                    "  %result_ptr{} = getelementptr inbounds {{{}}}, {{{}}}* %result, i32 0, i32 {}",
                    i,
                    output_vars.iter().map(|_| "float*").collect::<Vec<_>>().join(", "),
                    output_vars.iter().map(|_| "float*").collect::<Vec<_>>().join(", "),
                    i
                ).map_err(|e| JitError::CodeGenError(e.to_string()))?;

                writeln!(
                    self.module,
                    "  store float* %{}, float** %result_ptr{}, align 8",
                    var, i
                )
                .map_err(|e| JitError::CodeGenError(e.to_string()))?;
            }

            writeln!(
                self.module,
                "  %result_val = load {{{}}}, {{{}}}* %result, align 8",
                output_vars
                    .iter()
                    .map(|_| "float*")
                    .collect::<Vec<_>>()
                    .join(", "),
                output_vars
                    .iter()
                    .map(|_| "float*")
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

            writeln!(
                self.module,
                "  ret {{{}}} %result_val",
                output_vars
                    .iter()
                    .map(|_| "float*")
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;
        }

        Ok(())
    }

    /// Generate helper functions
    fn generate_helper_functions(&mut self) -> JitResult<()> {
        writeln!(self.module, "").map_err(|e| JitError::CodeGenError(e.to_string()))?;
        writeln!(self.module, "; Helper function declarations")
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

        // Matrix multiplication helper
        writeln!(self.module, "declare float* @matmul_impl(float*, float*)")
            .map_err(|e| JitError::CodeGenError(e.to_string()))?;

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

        for edge in graph.edges_directed(node_id, petgraph::Direction::Incoming) {
            let src_id = edge.source();
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
}

/// LLVM optimization configuration
#[derive(Debug, Clone)]
pub struct LlvmOptimizer {
    /// Optimization level (0-3)
    opt_level: u8,
    /// Target-specific optimizations
    target_specific: bool,
    /// Vectorization enabled
    vectorize: bool,
    /// Loop optimizations enabled
    loop_opt: bool,
}

impl Default for LlvmOptimizer {
    fn default() -> Self {
        Self::new(2)
    }
}

impl LlvmOptimizer {
    /// Create a new LLVM optimizer
    pub fn new(opt_level: u8) -> Self {
        Self {
            opt_level,
            target_specific: opt_level >= 2,
            vectorize: opt_level >= 2,
            loop_opt: opt_level >= 1,
        }
    }

    /// Enable target-specific optimizations
    pub fn enable_target_specific(&mut self, enabled: bool) {
        self.target_specific = enabled;
    }

    /// Enable vectorization
    pub fn enable_vectorization(&mut self, enabled: bool) {
        self.vectorize = enabled;
    }

    /// Enable loop optimizations
    pub fn enable_loop_optimization(&mut self, enabled: bool) {
        self.loop_opt = enabled;
    }

    /// Apply optimizations to LLVM IR
    pub fn optimize(&self, llvm_ir: &str) -> JitResult<String> {
        // In a real implementation, this would use LLVM's optimization passes
        // For now, return a simple optimized version
        let mut optimized = llvm_ir.to_string();

        if self.opt_level >= 1 {
            optimized = self.apply_basic_optimizations(optimized)?;
        }

        if self.opt_level >= 2 {
            optimized = self.apply_advanced_optimizations(optimized)?;
        }

        if self.opt_level >= 3 {
            optimized = self.apply_aggressive_optimizations(optimized)?;
        }

        Ok(optimized)
    }

    /// Apply basic optimizations
    fn apply_basic_optimizations(&self, ir: String) -> JitResult<String> {
        let mut result = ir;

        // Remove dead stores
        result = result.replace("store float %unused,", "; removed dead store:");

        // Simplify arithmetic
        result = result.replace("fadd float %x, 0.0", "; simplified: %x");
        result = result.replace("fmul float %x, 1.0", "; simplified: %x");
        result = result.replace("fmul float %x, 0.0", "; simplified: 0.0");

        Ok(result)
    }

    /// Apply advanced optimizations
    fn apply_advanced_optimizations(&self, ir: String) -> JitResult<String> {
        let mut result = ir;

        if self.vectorize {
            // Add vectorization hints
            result = result.replace(
                "for.body:",
                "for.body: ; vectorizable loop\n  !llvm.loop !{!llvm.loop.vectorize.enable, i1 true}"
            );
        }

        if self.loop_opt {
            // Add loop unrolling hints
            result = result.replace(
                "for.inc:",
                "for.inc: ; unroll loop\n  !llvm.loop !{!llvm.loop.unroll.count, i32 4}",
            );
        }

        Ok(result)
    }

    /// Apply aggressive optimizations
    fn apply_aggressive_optimizations(&self, ir: String) -> JitResult<String> {
        let mut result = ir;

        // Add aggressive inlining attributes
        result = result.replace("define ", "define alwaysinline ");

        // Add fast math flags
        result = result.replace("fadd float", "fadd fast float");
        result = result.replace("fsub float", "fsub fast float");
        result = result.replace("fmul float", "fmul fast float");
        result = result.replace("fdiv float", "fdiv fast float");

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ComputationGraph, ConstantInfo, Operation};
    use torsh_core::{DType, DeviceType, Shape};

    #[test]
    fn test_llvm_backend_creation() {
        let backend = LlvmBackend::new("test_module");
        assert_eq!(backend.module_name, "test_module");
        assert!(backend.symbol_table.is_empty());
    }

    #[test]
    fn test_llvm_optimizer() {
        let optimizer = LlvmOptimizer::new(2);
        assert_eq!(optimizer.opt_level, 2);
        assert!(optimizer.target_specific);
        assert!(optimizer.vectorize);
    }

    #[test]
    fn test_simple_llvm_generation() {
        let mut backend = LlvmBackend::new("test");
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

        let llvm_ir = result.unwrap();
        assert!(llvm_ir.contains("target triple"));
        assert!(llvm_ir.contains("define"));
        assert!(llvm_ir.contains("42"));
    }
}
