//! Constant Folding Optimization Pass
//!
//! This module provides constant folding capabilities for computational graphs.
//! It identifies operations with constant inputs and pre-computes their results,
//! replacing the operations with constant nodes to reduce runtime computation.

use crate::pattern_matching::graph::{ComputationGraph, GraphNode};
use crate::TorshResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::TorshError;

// =============================================================================
// Constant Folding Configuration
// =============================================================================

/// Configuration for constant folding behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldingConfig {
    /// Whether to apply aggressive folding (includes activations and transformations)
    pub aggressive: bool,
    /// Maximum number of folding iterations
    pub max_iterations: usize,
    /// Whether to preserve debugging information in folded nodes
    pub preserve_debug_info: bool,
    /// Custom foldable operation types
    pub custom_foldable_ops: Vec<String>,
    /// Precision threshold for numerical computations
    pub precision_threshold: f64,
    /// Whether to enable caching of computed values
    pub enable_caching: bool,
}

impl Default for FoldingConfig {
    fn default() -> Self {
        Self {
            aggressive: false,
            max_iterations: 5,
            preserve_debug_info: false,
            custom_foldable_ops: Vec::new(),
            precision_threshold: 1e-10,
            enable_caching: true,
        }
    }
}

// =============================================================================
// Constant Value Types
// =============================================================================

/// Represents different types of constant values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstantValue {
    /// 32-bit floating point value
    Float32(f32),
    /// 64-bit floating point value
    Float64(f64),
    /// 32-bit signed integer
    Int32(i32),
    /// 64-bit signed integer
    Int64(i64),
    /// Boolean value
    Bool(bool),
    /// String value
    String(String),
    /// Array of float values
    FloatArray(Vec<f32>),
    /// Tensor shape
    Shape(Vec<usize>),
}

impl ConstantValue {
    /// Convert to f32 if possible
    pub fn to_f32(&self) -> TorshResult<f32> {
        match self {
            ConstantValue::Float32(v) => Ok(*v),
            ConstantValue::Float64(v) => Ok(*v as f32),
            ConstantValue::Int32(v) => Ok(*v as f32),
            ConstantValue::Int64(v) => Ok(*v as f32),
            ConstantValue::Bool(v) => Ok(if *v { 1.0 } else { 0.0 }),
            _ => Err(TorshError::InvalidArgument(
                "Cannot convert to f32".to_string(),
            )),
        }
    }

    /// Convert to f64 if possible
    pub fn to_f64(&self) -> TorshResult<f64> {
        match self {
            ConstantValue::Float32(v) => Ok(*v as f64),
            ConstantValue::Float64(v) => Ok(*v),
            ConstantValue::Int32(v) => Ok(*v as f64),
            ConstantValue::Int64(v) => Ok(*v as f64),
            ConstantValue::Bool(v) => Ok(if *v { 1.0 } else { 0.0 }),
            _ => Err(TorshError::InvalidArgument(
                "Cannot convert to f64".to_string(),
            )),
        }
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match self {
            ConstantValue::Float32(v) => v.to_string(),
            ConstantValue::Float64(v) => v.to_string(),
            ConstantValue::Int32(v) => v.to_string(),
            ConstantValue::Int64(v) => v.to_string(),
            ConstantValue::Bool(v) => v.to_string(),
            ConstantValue::String(v) => v.clone(),
            ConstantValue::FloatArray(v) => format!("{:?}", v),
            ConstantValue::Shape(v) => format!("{:?}", v),
        }
    }

    /// Parse from string
    pub fn from_string(s: &str, value_type: &str) -> TorshResult<Self> {
        match value_type {
            "f32" | "float32" => {
                let v = s
                    .parse::<f32>()
                    .map_err(|_| TorshError::InvalidArgument("Cannot parse f32".to_string()))?;
                Ok(ConstantValue::Float32(v))
            }
            "f64" | "float64" => {
                let v = s
                    .parse::<f64>()
                    .map_err(|_| TorshError::InvalidArgument("Cannot parse f64".to_string()))?;
                Ok(ConstantValue::Float64(v))
            }
            "i32" | "int32" => {
                let v = s
                    .parse::<i32>()
                    .map_err(|_| TorshError::InvalidArgument("Cannot parse i32".to_string()))?;
                Ok(ConstantValue::Int32(v))
            }
            "i64" | "int64" => {
                let v = s
                    .parse::<i64>()
                    .map_err(|_| TorshError::InvalidArgument("Cannot parse i64".to_string()))?;
                Ok(ConstantValue::Int64(v))
            }
            "bool" => {
                let v = s
                    .parse::<bool>()
                    .map_err(|_| TorshError::InvalidArgument("Cannot parse bool".to_string()))?;
                Ok(ConstantValue::Bool(v))
            }
            _ => Ok(ConstantValue::String(s.to_string())),
        }
    }
}

// =============================================================================
// Constant Folding Pass
// =============================================================================

/// Advanced constant folding pass with comprehensive operation support
#[derive(Debug)]
pub struct ConstantFoldingPass {
    /// Configuration for folding behavior
    config: FoldingConfig,
    /// Cache for computed constant values
    constant_cache: HashMap<String, ConstantValue>,
    /// Statistics tracking
    stats: FoldingStatistics,
}

impl ConstantFoldingPass {
    /// Create a new constant folding pass
    pub fn new() -> Self {
        Self {
            config: FoldingConfig::default(),
            constant_cache: HashMap::new(),
            stats: FoldingStatistics::default(),
        }
    }

    /// Create a constant folding pass with custom configuration
    pub fn with_config(config: FoldingConfig) -> Self {
        Self {
            config,
            constant_cache: HashMap::new(),
            stats: FoldingStatistics::default(),
        }
    }

    /// Enable or disable aggressive folding
    pub fn set_aggressive(&mut self, aggressive: bool) {
        self.config.aggressive = aggressive;
    }

    /// Add a custom foldable operation type
    pub fn add_foldable_op(&mut self, op_type: String) {
        self.config.custom_foldable_ops.push(op_type);
    }

    /// Get folding statistics
    pub fn get_statistics(&self) -> &FoldingStatistics {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.stats = FoldingStatistics::default();
    }

    /// Clear the constant cache
    pub fn clear_cache(&mut self) {
        self.constant_cache.clear();
    }

    /// Apply constant folding to a graph
    pub fn fold(&mut self, graph: &mut ComputationGraph) -> TorshResult<FoldingResult> {
        let initial_node_count = graph.nodes.len();
        let mut total_folded = 0;
        let mut iteration = 0;

        self.stats.folding_runs += 1;

        // Clear cache for fresh computation if not enabled
        if !self.config.enable_caching {
            self.constant_cache.clear();
        }

        // Repeat until no more constants can be folded
        while iteration < self.config.max_iterations {
            let foldable_nodes = self.find_foldable_nodes(graph)?;

            if foldable_nodes.is_empty() {
                break; // No more foldable nodes
            }

            let folded_this_iteration = foldable_nodes.len();

            for node_id in foldable_nodes {
                if self.fold_constant_node(graph, &node_id)? {
                    self.stats.nodes_folded += 1;
                }
            }

            total_folded += folded_this_iteration;
            iteration += 1;

            // For non-aggressive mode, one pass is usually enough
            if !self.config.aggressive && iteration >= 1 {
                break;
            }
        }

        let final_node_count = graph.nodes.len();

        Ok(FoldingResult {
            nodes_folded: total_folded,
            constants_created: total_folded,
            iterations: iteration,
            initial_node_count,
            final_node_count,
            computation_savings: self.estimate_computation_savings(total_folded),
            success: true,
        })
    }

    /// Find nodes that can be constant folded
    fn find_foldable_nodes(&mut self, graph: &ComputationGraph) -> TorshResult<Vec<String>> {
        let mut foldable_nodes = Vec::new();

        // Process nodes in execution order to ensure dependencies are computed first
        for node_id in graph.get_execution_order() {
            if let Some(node) = graph.get_node(node_id) {
                if self.can_fold_node(node, graph) {
                    foldable_nodes.push(node_id.clone());
                }
            }
        }

        Ok(foldable_nodes)
    }

    /// Check if a node can be constant folded
    fn can_fold_node(&self, node: &GraphNode, graph: &ComputationGraph) -> bool {
        // A node can be folded if:
        // 1. It's not already a constant
        // 2. All its inputs are constants or already folded
        // 3. It's a foldable operation type

        if self.is_constant_node(node) {
            return false; // Already a constant
        }

        if !self.is_foldable_operation(&node.op_type) {
            return false; // Operation cannot be folded
        }

        // Check if all inputs are constants
        for input_id in &node.inputs {
            if let Some(input_node) = graph.get_node(input_id) {
                if !self.is_constant_node(input_node) && !self.constant_cache.contains_key(input_id)
                {
                    return false; // Input is not constant
                }
            } else {
                return false; // Input node doesn't exist
            }
        }

        true
    }

    /// Check if a node represents a constant value
    fn is_constant_node(&self, node: &GraphNode) -> bool {
        matches!(
            node.op_type.as_str(),
            "constant" | "const" | "parameter" | "scalar" | "literal"
        )
    }

    /// Check if an operation type can be constant folded
    fn is_foldable_operation(&self, op_type: &str) -> bool {
        // Check custom foldable operations first
        if self
            .config
            .custom_foldable_ops
            .contains(&op_type.to_string())
        {
            return true;
        }

        match op_type {
            // Basic arithmetic operations
            "add" | "sub" | "mul" | "div" | "mod" | "pow" | "neg" => true,
            // Math functions
            "abs" | "sqrt" | "exp" | "log" | "log10" | "log2" => true,
            "sin" | "cos" | "tan" | "asin" | "acos" | "atan" => true,
            "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" => true,
            "floor" | "ceil" | "round" | "trunc" => true,
            // Comparison operations
            "eq" | "ne" | "lt" | "le" | "gt" | "ge" => true,
            // Logical operations
            "and" | "or" | "not" | "xor" => true,
            // Min/max operations
            "min" | "max" | "clamp" => true,
            // Type conversion (if aggressive)
            "cast" | "convert" => self.config.aggressive,
            // Quantization operations (if all inputs are constant)
            "quantize" | "dequantize" => true,
            // Shape operations (if aggressive and shape is constant)
            "reshape" | "transpose" | "squeeze" | "unsqueeze" => self.config.aggressive,
            // Activations (if aggressive mode)
            "relu" | "leaky_relu" | "elu" | "selu" => self.config.aggressive,
            "sigmoid" | "tanh" | "softmax" | "gelu" => self.config.aggressive,
            // Reduction operations on constant tensors (if aggressive)
            "sum" | "mean" | "max" | "min" | "prod" => self.config.aggressive,
            _ => false,
        }
    }

    /// Fold a constant node and replace it with a constant
    fn fold_constant_node(
        &mut self,
        graph: &mut ComputationGraph,
        node_id: &str,
    ) -> TorshResult<bool> {
        let node = graph
            .get_node(node_id)
            .ok_or_else(|| {
                TorshError::InvalidArgument("Node not found for constant folding".to_string())
            })?
            .clone();

        // Compute the constant value
        let constant_value = self.compute_constant_value(&node, graph)?;

        // Create a new constant node
        let folded_id = format!("{}_folded", node_id);
        let mut constant_node = GraphNode::new(folded_id.clone(), "constant".to_string());

        // Set the computed value
        constant_node.set_attribute("value".to_string(), constant_value.to_string());
        constant_node.set_attribute("dtype".to_string(), self.get_value_type(&constant_value));

        // Preserve debugging information if configured
        if self.config.preserve_debug_info {
            constant_node.set_attribute("original_op".to_string(), node.op_type.clone());
            constant_node.set_attribute("original_id".to_string(), node.id.clone());
            constant_node
                .set_attribute("folded_by".to_string(), "constant_folding_pass".to_string());
        }

        // Preserve the original node's outputs
        constant_node.outputs = node.outputs.clone();

        // Update cache
        if self.config.enable_caching {
            self.constant_cache
                .insert(folded_id.clone(), constant_value);
        }

        // Update statistics
        *self
            .stats
            .folded_by_op_type
            .entry(node.op_type.clone())
            .or_insert(0) += 1;

        // Replace the original node
        graph.remove_node(node_id);
        graph.add_node(constant_node);

        Ok(true)
    }

    /// Compute the constant value for a node
    fn compute_constant_value(
        &self,
        node: &GraphNode,
        graph: &ComputationGraph,
    ) -> TorshResult<ConstantValue> {
        let input_values = self.get_input_values(node, graph)?;

        match node.op_type.as_str() {
            // Binary arithmetic operations
            "add" => self.compute_add(&input_values),
            "sub" => self.compute_sub(&input_values),
            "mul" => self.compute_mul(&input_values),
            "div" => self.compute_div(&input_values),
            "mod" => self.compute_mod(&input_values),
            "pow" => self.compute_pow(&input_values),

            // Unary operations
            "neg" => self.compute_neg(&input_values),
            "abs" => self.compute_abs(&input_values),
            "sqrt" => self.compute_sqrt(&input_values),
            "exp" => self.compute_exp(&input_values),
            "log" => self.compute_log(&input_values),
            "log10" => self.compute_log10(&input_values),
            "log2" => self.compute_log2(&input_values),

            // Trigonometric functions
            "sin" => self.compute_sin(&input_values),
            "cos" => self.compute_cos(&input_values),
            "tan" => self.compute_tan(&input_values),
            "asin" => self.compute_asin(&input_values),
            "acos" => self.compute_acos(&input_values),
            "atan" => self.compute_atan(&input_values),

            // Hyperbolic functions
            "sinh" => self.compute_sinh(&input_values),
            "cosh" => self.compute_cosh(&input_values),
            "tanh" => self.compute_tanh(&input_values),

            // Rounding functions
            "floor" => self.compute_floor(&input_values),
            "ceil" => self.compute_ceil(&input_values),
            "round" => self.compute_round(&input_values),
            "trunc" => self.compute_trunc(&input_values),

            // Comparison operations
            "eq" => self.compute_eq(&input_values),
            "ne" => self.compute_ne(&input_values),
            "lt" => self.compute_lt(&input_values),
            "le" => self.compute_le(&input_values),
            "gt" => self.compute_gt(&input_values),
            "ge" => self.compute_ge(&input_values),

            // Logical operations
            "and" => self.compute_and(&input_values),
            "or" => self.compute_or(&input_values),
            "not" => self.compute_not(&input_values),
            "xor" => self.compute_xor(&input_values),

            // Min/max operations
            "min" => self.compute_min(&input_values),
            "max" => self.compute_max(&input_values),
            "clamp" => self.compute_clamp(&input_values),

            // Activation functions
            "relu" => self.compute_relu(&input_values),
            "leaky_relu" => self.compute_leaky_relu(&input_values, node),
            "sigmoid" => self.compute_sigmoid(&input_values),

            _ => Err(TorshError::InvalidArgument(format!(
                "Unsupported operation for constant folding: {}",
                node.op_type
            ))),
        }
    }

    /// Get constant values for all inputs of a node
    fn get_input_values(
        &self,
        node: &GraphNode,
        graph: &ComputationGraph,
    ) -> TorshResult<Vec<ConstantValue>> {
        let mut values = Vec::new();

        for input_id in &node.inputs {
            if let Some(value) = self.constant_cache.get(input_id) {
                values.push(value.clone());
            } else if let Some(input_node) = graph.get_node(input_id) {
                if self.is_constant_node(input_node) {
                    let value = self.extract_constant_value(input_node)?;
                    values.push(value);
                } else {
                    return Err(TorshError::InvalidArgument(format!(
                        "Input {input_id} is not constant"
                    )));
                }
            } else {
                return Err(TorshError::InvalidArgument(format!(
                    "Input node {input_id} not found"
                )));
            }
        }

        Ok(values)
    }

    /// Extract constant value from a constant node
    fn extract_constant_value(&self, node: &GraphNode) -> TorshResult<ConstantValue> {
        if let Some(value_str) = node.get_attribute("value") {
            let dtype = node
                .get_attribute("dtype")
                .unwrap_or(&"f32".to_string())
                .clone();
            ConstantValue::from_string(value_str, &dtype)
        } else {
            Err(TorshError::InvalidArgument(
                "Constant node missing value attribute".to_string(),
            ))
        }
    }

    /// Get the type string for a constant value
    fn get_value_type(&self, value: &ConstantValue) -> String {
        match value {
            ConstantValue::Float32(_) => "f32".to_string(),
            ConstantValue::Float64(_) => "f64".to_string(),
            ConstantValue::Int32(_) => "i32".to_string(),
            ConstantValue::Int64(_) => "i64".to_string(),
            ConstantValue::Bool(_) => "bool".to_string(),
            ConstantValue::String(_) => "string".to_string(),
            ConstantValue::FloatArray(_) => "float_array".to_string(),
            ConstantValue::Shape(_) => "shape".to_string(),
        }
    }

    /// Estimate computation savings from folding
    fn estimate_computation_savings(&self, folded_count: usize) -> f64 {
        // Rough estimate: each folded operation saves 1 unit of computation
        folded_count as f64
    }

    // =============================================================================
    // Arithmetic Operations
    // =============================================================================

    fn compute_add(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        let sum = inputs
            .iter()
            .try_fold(0.0f64, |acc, val| val.to_f64().map(|v| acc + v))?;
        Ok(ConstantValue::Float64(sum))
    }

    fn compute_sub(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Sub requires exactly 2 inputs".to_string(),
            ));
        }
        let a = inputs[0].to_f64()?;
        let b = inputs[1].to_f64()?;
        Ok(ConstantValue::Float64(a - b))
    }

    fn compute_mul(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        let product = inputs
            .iter()
            .try_fold(1.0f64, |acc, val| val.to_f64().map(|v| acc * v))?;
        Ok(ConstantValue::Float64(product))
    }

    fn compute_div(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Div requires exactly 2 inputs".to_string(),
            ));
        }
        let a = inputs[0].to_f64()?;
        let b = inputs[1].to_f64()?;
        if b.abs() < self.config.precision_threshold {
            return Err(TorshError::InvalidArgument("Division by zero".to_string()));
        }
        Ok(ConstantValue::Float64(a / b))
    }

    fn compute_mod(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Mod requires exactly 2 inputs".to_string(),
            ));
        }
        let a = inputs[0].to_f64()?;
        let b = inputs[1].to_f64()?;
        if b.abs() < self.config.precision_threshold {
            return Err(TorshError::InvalidArgument("Modulo by zero".to_string()));
        }
        Ok(ConstantValue::Float64(a % b))
    }

    fn compute_pow(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Pow requires exactly 2 inputs".to_string(),
            ));
        }
        let base = inputs[0].to_f64()?;
        let exp = inputs[1].to_f64()?;
        Ok(ConstantValue::Float64(base.powf(exp)))
    }

    // =============================================================================
    // Unary Operations
    // =============================================================================

    fn compute_neg(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Neg requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(-val))
    }

    fn compute_abs(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Abs requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.abs()))
    }

    fn compute_sqrt(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Sqrt requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        if val < 0.0 {
            return Err(TorshError::InvalidArgument(
                "Sqrt of negative number".to_string(),
            ));
        }
        Ok(ConstantValue::Float64(val.sqrt()))
    }

    fn compute_exp(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Exp requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.exp()))
    }

    fn compute_log(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Log requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        if val <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Log of non-positive number".to_string(),
            ));
        }
        Ok(ConstantValue::Float64(val.ln()))
    }

    fn compute_log10(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Log10 requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        if val <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Log10 of non-positive number".to_string(),
            ));
        }
        Ok(ConstantValue::Float64(val.log10()))
    }

    fn compute_log2(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Log2 requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        if val <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Log2 of non-positive number".to_string(),
            ));
        }
        Ok(ConstantValue::Float64(val.log2()))
    }

    // =============================================================================
    // Trigonometric Functions
    // =============================================================================

    fn compute_sin(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Sin requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.sin()))
    }

    fn compute_cos(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Cos requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.cos()))
    }

    fn compute_tan(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Tan requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.tan()))
    }

    fn compute_asin(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Asin requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        if val < -1.0 || val > 1.0 {
            return Err(TorshError::InvalidArgument(
                "Asin input out of range [-1, 1]".to_string(),
            ));
        }
        Ok(ConstantValue::Float64(val.asin()))
    }

    fn compute_acos(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Acos requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        if val < -1.0 || val > 1.0 {
            return Err(TorshError::InvalidArgument(
                "Acos input out of range [-1, 1]".to_string(),
            ));
        }
        Ok(ConstantValue::Float64(val.acos()))
    }

    fn compute_atan(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Atan requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.atan()))
    }

    // =============================================================================
    // Hyperbolic Functions
    // =============================================================================

    fn compute_sinh(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Sinh requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.sinh()))
    }

    fn compute_cosh(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Cosh requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.cosh()))
    }

    fn compute_tanh(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Tanh requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.tanh()))
    }

    // =============================================================================
    // Rounding Functions
    // =============================================================================

    fn compute_floor(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Floor requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.floor()))
    }

    fn compute_ceil(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Ceil requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.ceil()))
    }

    fn compute_round(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Round requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.round()))
    }

    fn compute_trunc(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Trunc requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.trunc()))
    }

    // =============================================================================
    // Comparison Operations
    // =============================================================================

    fn compute_eq(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Eq requires exactly 2 inputs".to_string(),
            ));
        }
        let a = inputs[0].to_f64()?;
        let b = inputs[1].to_f64()?;
        Ok(ConstantValue::Bool(
            (a - b).abs() < self.config.precision_threshold,
        ))
    }

    fn compute_ne(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Ne requires exactly 2 inputs".to_string(),
            ));
        }
        let a = inputs[0].to_f64()?;
        let b = inputs[1].to_f64()?;
        Ok(ConstantValue::Bool(
            (a - b).abs() >= self.config.precision_threshold,
        ))
    }

    fn compute_lt(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Lt requires exactly 2 inputs".to_string(),
            ));
        }
        let a = inputs[0].to_f64()?;
        let b = inputs[1].to_f64()?;
        Ok(ConstantValue::Bool(a < b))
    }

    fn compute_le(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Le requires exactly 2 inputs".to_string(),
            ));
        }
        let a = inputs[0].to_f64()?;
        let b = inputs[1].to_f64()?;
        Ok(ConstantValue::Bool(a <= b))
    }

    fn compute_gt(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Gt requires exactly 2 inputs".to_string(),
            ));
        }
        let a = inputs[0].to_f64()?;
        let b = inputs[1].to_f64()?;
        Ok(ConstantValue::Bool(a > b))
    }

    fn compute_ge(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Ge requires exactly 2 inputs".to_string(),
            ));
        }
        let a = inputs[0].to_f64()?;
        let b = inputs[1].to_f64()?;
        Ok(ConstantValue::Bool(a >= b))
    }

    // =============================================================================
    // Logical Operations
    // =============================================================================

    fn compute_and(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "And requires exactly 2 inputs".to_string(),
            ));
        }
        let a = inputs[0].to_f64()? != 0.0;
        let b = inputs[1].to_f64()? != 0.0;
        Ok(ConstantValue::Bool(a && b))
    }

    fn compute_or(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Or requires exactly 2 inputs".to_string(),
            ));
        }
        let a = inputs[0].to_f64()? != 0.0;
        let b = inputs[1].to_f64()? != 0.0;
        Ok(ConstantValue::Bool(a || b))
    }

    fn compute_not(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Not requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()? != 0.0;
        Ok(ConstantValue::Bool(!val))
    }

    fn compute_xor(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Xor requires exactly 2 inputs".to_string(),
            ));
        }
        let a = inputs[0].to_f64()? != 0.0;
        let b = inputs[1].to_f64()? != 0.0;
        Ok(ConstantValue::Bool(a ^ b))
    }

    // =============================================================================
    // Min/Max Operations
    // =============================================================================

    fn compute_min(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Min requires at least 1 input".to_string(),
            ));
        }
        let values: Result<Vec<f64>, _> = inputs.iter().map(|v| v.to_f64()).collect();
        let values = values?;
        let min_val = values.into_iter().fold(f64::INFINITY, f64::min);
        Ok(ConstantValue::Float64(min_val))
    }

    fn compute_max(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Max requires at least 1 input".to_string(),
            ));
        }
        let values: Result<Vec<f64>, _> = inputs.iter().map(|v| v.to_f64()).collect();
        let values = values?;
        let max_val = values.into_iter().fold(f64::NEG_INFINITY, f64::max);
        Ok(ConstantValue::Float64(max_val))
    }

    fn compute_clamp(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 3 {
            return Err(TorshError::InvalidArgument(
                "Clamp requires exactly 3 inputs (value, min, max)".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        let min_val = inputs[1].to_f64()?;
        let max_val = inputs[2].to_f64()?;
        Ok(ConstantValue::Float64(val.clamp(min_val, max_val)))
    }

    // =============================================================================
    // Activation Functions
    // =============================================================================

    fn compute_relu(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "ReLU requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(val.max(0.0)))
    }

    fn compute_leaky_relu(
        &self,
        inputs: &[ConstantValue],
        node: &GraphNode,
    ) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "LeakyReLU requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        let alpha = node
            .get_attribute("alpha")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.01);
        Ok(ConstantValue::Float64(if val > 0.0 {
            val
        } else {
            alpha * val
        }))
    }

    fn compute_sigmoid(&self, inputs: &[ConstantValue]) -> TorshResult<ConstantValue> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Sigmoid requires exactly 1 input".to_string(),
            ));
        }
        let val = inputs[0].to_f64()?;
        Ok(ConstantValue::Float64(1.0 / (1.0 + (-val).exp())))
    }
}

impl Default for ConstantFoldingPass {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Statistics and Results
// =============================================================================

/// Statistics for constant folding
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FoldingStatistics {
    /// Number of folding runs
    pub folding_runs: usize,
    /// Total nodes folded
    pub nodes_folded: usize,
    /// Nodes folded by operation type
    pub folded_by_op_type: HashMap<String, usize>,
}

/// Result of constant folding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldingResult {
    /// Number of nodes folded
    pub nodes_folded: usize,
    /// Number of constant nodes created
    pub constants_created: usize,
    /// Number of iterations performed
    pub iterations: usize,
    /// Initial node count
    pub initial_node_count: usize,
    /// Final node count
    pub final_node_count: usize,
    /// Estimated computation savings
    pub computation_savings: f64,
    /// Whether folding was successful
    pub success: bool,
}

// =============================================================================
// Specialized Folding Passes
// =============================================================================

/// Create a conservative folding pass (basic operations only)
pub fn create_conservative_pass() -> ConstantFoldingPass {
    let config = FoldingConfig {
        aggressive: false,
        max_iterations: 3,
        preserve_debug_info: true,
        ..Default::default()
    };

    ConstantFoldingPass::with_config(config)
}

/// Create an aggressive folding pass (includes activations and transformations)
pub fn create_aggressive_pass() -> ConstantFoldingPass {
    let config = FoldingConfig {
        aggressive: true,
        max_iterations: 10,
        preserve_debug_info: false,
        ..Default::default()
    };

    ConstantFoldingPass::with_config(config)
}

/// Create a high-precision folding pass
pub fn create_high_precision_pass() -> ConstantFoldingPass {
    let config = FoldingConfig {
        aggressive: true,
        precision_threshold: 1e-15,
        enable_caching: true,
        ..Default::default()
    };

    ConstantFoldingPass::with_config(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern_matching::graph::{ComputationGraph, GraphNode};

    #[test]
    fn test_constant_value_conversions() {
        let val = ConstantValue::Float32(3.14);
        assert!((val.to_f32().unwrap() - 3.14).abs() < 1e-6);
        assert!((val.to_f64().unwrap() - 3.14).abs() < 1e-6);

        let bool_val = ConstantValue::Bool(true);
        assert_eq!(bool_val.to_f32().unwrap(), 1.0);

        let int_val = ConstantValue::Int32(42);
        assert_eq!(int_val.to_f32().unwrap(), 42.0);
    }

    #[test]
    fn test_folding_pass_creation() {
        let pass = ConstantFoldingPass::new();
        assert!(!pass.config.aggressive);
        assert_eq!(pass.config.max_iterations, 5);

        let aggressive_pass = create_aggressive_pass();
        assert!(aggressive_pass.config.aggressive);
        assert_eq!(aggressive_pass.config.max_iterations, 10);
    }

    #[test]
    fn test_constant_node_detection() {
        let pass = ConstantFoldingPass::new();

        let const_node = GraphNode::new("const1".to_string(), "constant".to_string());
        assert!(pass.is_constant_node(&const_node));

        let relu_node = GraphNode::new("relu1".to_string(), "relu".to_string());
        assert!(!pass.is_constant_node(&relu_node));
    }

    #[test]
    fn test_foldable_operation_detection() {
        let pass = ConstantFoldingPass::new();

        // Basic arithmetic should be foldable
        assert!(pass.is_foldable_operation("add"));
        assert!(pass.is_foldable_operation("mul"));
        assert!(pass.is_foldable_operation("div"));

        // Math functions should be foldable
        assert!(pass.is_foldable_operation("sin"));
        assert!(pass.is_foldable_operation("exp"));
        assert!(pass.is_foldable_operation("sqrt"));

        // Non-aggressive mode shouldn't fold activations
        assert!(!pass.is_foldable_operation("relu"));

        let mut aggressive_pass = ConstantFoldingPass::new();
        aggressive_pass.set_aggressive(true);
        assert!(aggressive_pass.is_foldable_operation("relu"));
    }

    #[test]
    fn test_arithmetic_operations() {
        let pass = ConstantFoldingPass::new();

        // Test addition
        let inputs = vec![ConstantValue::Float32(2.0), ConstantValue::Float32(3.0)];
        let result = pass.compute_add(&inputs).unwrap();
        assert_eq!(result.to_f64().unwrap(), 5.0);

        // Test multiplication
        let result = pass.compute_mul(&inputs).unwrap();
        assert_eq!(result.to_f64().unwrap(), 6.0);

        // Test division
        let result = pass.compute_div(&inputs).unwrap();
        assert!((result.to_f64().unwrap() - (2.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_math_functions() {
        let pass = ConstantFoldingPass::new();

        let inputs = vec![ConstantValue::Float32(4.0)];
        let result = pass.compute_sqrt(&inputs).unwrap();
        assert_eq!(result.to_f64().unwrap(), 2.0);

        let inputs = vec![ConstantValue::Float32(0.0)];
        let result = pass.compute_exp(&inputs).unwrap();
        assert_eq!(result.to_f64().unwrap(), 1.0);

        let inputs = vec![ConstantValue::Float32(std::f32::consts::PI / 2.0)];
        let result = pass.compute_sin(&inputs).unwrap();
        assert!((result.to_f64().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_comparison_operations() {
        let pass = ConstantFoldingPass::new();

        let inputs = vec![ConstantValue::Float32(2.0), ConstantValue::Float32(3.0)];
        let result = pass.compute_lt(&inputs).unwrap();
        if let ConstantValue::Bool(val) = result {
            assert!(val);
        } else {
            panic!("Expected bool result");
        }

        let result = pass.compute_gt(&inputs).unwrap();
        if let ConstantValue::Bool(val) = result {
            assert!(!val);
        } else {
            panic!("Expected bool result");
        }
    }

    #[test]
    fn test_error_handling() {
        let pass = ConstantFoldingPass::new();

        // Division by zero
        let inputs = vec![ConstantValue::Float32(1.0), ConstantValue::Float32(0.0)];
        assert!(pass.compute_div(&inputs).is_err());

        // Sqrt of negative number
        let inputs = vec![ConstantValue::Float32(-1.0)];
        assert!(pass.compute_sqrt(&inputs).is_err());

        // Wrong number of inputs
        let inputs = vec![ConstantValue::Float32(1.0)];
        assert!(pass.compute_add(&inputs).is_ok()); // Should work with 1 input
        assert!(pass.compute_div(&inputs).is_err()); // Should fail with 1 input
    }

    #[test]
    fn test_constant_folding_integration() {
        let mut graph = ComputationGraph::new();

        // Create constant nodes
        let mut const1 = GraphNode::new("const1".to_string(), "constant".to_string());
        const1.set_attribute("value".to_string(), "2.0".to_string());
        const1.set_attribute("dtype".to_string(), "f32".to_string());

        let mut const2 = GraphNode::new("const2".to_string(), "constant".to_string());
        const2.set_attribute("value".to_string(), "3.0".to_string());
        const2.set_attribute("dtype".to_string(), "f32".to_string());

        // Create add operation
        let add_node = GraphNode::new("add1".to_string(), "add".to_string());

        graph.add_node(const1);
        graph.add_node(const2);
        graph.add_node(add_node);

        graph.connect_nodes("const1", "add1").unwrap();
        graph.connect_nodes("const2", "add1").unwrap();

        let mut pass = ConstantFoldingPass::new();
        let result = pass.fold(&mut graph).unwrap();

        assert!(result.success);
        assert!(result.nodes_folded > 0);
    }

    #[test]
    fn test_specialized_passes() {
        let conservative = create_conservative_pass();
        assert!(!conservative.config.aggressive);
        assert!(conservative.config.preserve_debug_info);

        let aggressive = create_aggressive_pass();
        assert!(aggressive.config.aggressive);
        assert!(!aggressive.config.preserve_debug_info);

        let high_precision = create_high_precision_pass();
        assert_eq!(high_precision.config.precision_threshold, 1e-15);
    }
}
