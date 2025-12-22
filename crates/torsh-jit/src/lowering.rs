//! Graph to IR lowering pass
//!
//! This module converts high-level computation graphs to lower-level IR
//! suitable for optimization and code generation.

use crate::graph::{ComputationGraph, Node, NodeId, Operation};
use crate::ir::{
    shape_dtype_to_tensor_type, ConstantData, IrBuilder, IrModule, IrOpcode, IrType, IrValue,
    Terminator, ValueDef, ValueKind,
};
use crate::{JitError, JitResult};
use std::collections::HashMap;

/// Lowering context for graph to IR conversion
pub struct LoweringContext {
    /// IR builder
    builder: IrBuilder,

    /// Mapping from graph nodes to IR values
    node_to_value: HashMap<NodeId, IrValue>,

    /// Mapping from graph nodes to IR types
    node_to_type: HashMap<NodeId, IrType>,
}

impl LoweringContext {
    /// Create a new lowering context
    pub fn new(module_name: String) -> Self {
        Self {
            builder: IrBuilder::new(module_name),
            node_to_value: HashMap::new(),
            node_to_type: HashMap::new(),
        }
    }

    /// Lower a computation graph to IR
    pub fn lower(mut self, graph: &ComputationGraph) -> JitResult<IrModule> {
        // Create entry block
        let entry_block = self.builder.add_block();

        // Process types first
        self.lower_types(graph)?;

        // Process input nodes
        self.lower_inputs(graph)?;

        // Process all nodes in topological order
        let order = graph
            .topological_sort()
            .map_err(|e| JitError::GraphError(format!("{:?}", e)))?;

        for &node_id in &order {
            if let Some(node) = graph.node(node_id) {
                self.lower_node(graph, node_id, node)?;
            }
        }

        // Process outputs
        self.lower_outputs(graph)?;

        // Set return terminator
        let output_values: Vec<_> = graph
            .outputs
            .iter()
            .filter_map(|&id| self.node_to_value.get(&id).copied())
            .collect();

        if output_values.len() == 1 {
            self.builder.set_terminator(Terminator::Return {
                value: Some(output_values[0]),
            });
        } else if output_values.is_empty() {
            self.builder
                .set_terminator(Terminator::Return { value: None });
        } else {
            // Multiple outputs - need to pack into struct
            // For now, just return the first one
            self.builder.set_terminator(Terminator::Return {
                value: output_values.first().copied(),
            });
        }

        let mut module = self.builder.build();
        module.entry_block = entry_block;

        // Set module inputs and outputs
        module.inputs = graph
            .inputs
            .iter()
            .filter_map(|&id| self.node_to_value.get(&id).copied())
            .collect();

        module.outputs = graph
            .outputs
            .iter()
            .filter_map(|&id| self.node_to_value.get(&id).copied())
            .collect();

        Ok(module)
    }

    /// Lower type information for all nodes
    fn lower_types(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        for (node_id, node) in graph.nodes() {
            let ir_type = self.lower_node_type(node)?;
            self.node_to_type.insert(node_id, ir_type);
        }
        Ok(())
    }

    /// Lower a single node's type
    fn lower_node_type(&mut self, node: &Node) -> JitResult<IrType> {
        let type_kind = shape_dtype_to_tensor_type(&node.output_shape, node.dtype);
        Ok(self.builder.get_type(type_kind))
    }

    /// Lower input nodes
    fn lower_inputs(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        for (i, &input_id) in graph.inputs.iter().enumerate() {
            if let Some(_node) = graph.node(input_id) {
                let ir_type = self.node_to_type[&input_id];

                let val_def = ValueDef {
                    ty: ir_type,
                    source_node: Some(input_id),
                    kind: ValueKind::Parameter { index: i },
                };

                let ir_value = self.builder.module.add_value(val_def);
                self.node_to_value.insert(input_id, ir_value);
            }
        }
        Ok(())
    }

    /// Lower output nodes
    fn lower_outputs(&mut self, _graph: &ComputationGraph) -> JitResult<()> {
        // Outputs are handled by setting the return terminator
        Ok(())
    }

    /// Lower a single computation node
    fn lower_node(
        &mut self,
        graph: &ComputationGraph,
        node_id: NodeId,
        node: &Node,
    ) -> JitResult<()> {
        // Skip input nodes (already processed)
        if matches!(node.op, Operation::Input) {
            return Ok(());
        }

        let result_type = self.node_to_type[&node_id];

        // Get operands from predecessors
        let operands: Vec<IrValue> = graph
            .predecessors(node_id)
            .filter_map(|pred_id| self.node_to_value.get(&pred_id).copied())
            .collect();

        // Convert operation to IR opcode
        let opcode = self.operation_to_opcode(&node.op)?;

        // Handle special cases
        match &node.op {
            Operation::Constant(const_info) => {
                self.lower_constant(node_id, const_info, result_type)?;
            }
            Operation::Parameter(_) => {
                // Parameters are handled separately
            }
            _ => {
                // Regular instruction
                let result = self
                    .builder
                    .add_instruction(opcode, operands, Some(result_type));
                if let Some(ir_value) = result {
                    self.node_to_value.insert(node_id, ir_value);
                }
            }
        }

        Ok(())
    }

    /// Lower a constant node
    fn lower_constant(
        &mut self,
        node_id: NodeId,
        const_info: &crate::graph::ConstantInfo,
        result_type: IrType,
    ) -> JitResult<()> {
        let data = match &const_info.value {
            crate::graph::ConstantValue::Scalar(val) => ConstantData::Float(*val),
            crate::graph::ConstantValue::IntScalar(val) => ConstantData::Int(*val),
            crate::graph::ConstantValue::Tensor {
                shape: _,
                data,
                dtype: _,
            } => ConstantData::Array(data.iter().map(|&x| ConstantData::Float(x)).collect()),
            crate::graph::ConstantValue::Bool(val) => ConstantData::Int(if *val { 1 } else { 0 }),
            crate::graph::ConstantValue::Int(val) => ConstantData::Int(*val),
            crate::graph::ConstantValue::Float(val) => ConstantData::Float(*val),
            _ => ConstantData::Float(0.0), // Default for unhandled types
        };

        let val_def = ValueDef {
            ty: result_type,
            source_node: Some(node_id),
            kind: ValueKind::Constant { data },
        };

        let ir_value = self.builder.module.add_value(val_def);
        self.node_to_value.insert(node_id, ir_value);
        Ok(())
    }

    /// Convert graph operation to IR opcode
    fn operation_to_opcode(&self, op: &Operation) -> JitResult<IrOpcode> {
        let opcode = match op {
            // Element-wise unary operations
            Operation::Neg => IrOpcode::Neg,
            Operation::Abs => IrOpcode::Abs,
            Operation::Exp => IrOpcode::Exp,
            Operation::Log => IrOpcode::Log,
            Operation::Sqrt => IrOpcode::Sqrt,
            Operation::Sin => IrOpcode::Sin,
            Operation::Cos => IrOpcode::Cos,
            Operation::Tanh => IrOpcode::Tanh,
            Operation::Sigmoid => IrOpcode::Sigmoid,
            Operation::Relu => IrOpcode::Relu,
            Operation::Gelu => IrOpcode::Gelu,
            Operation::Silu => IrOpcode::Intrinsic("silu".to_string()),
            Operation::Softmax { .. } => IrOpcode::Softmax,
            Operation::LogSoftmax { .. } => IrOpcode::Intrinsic("logsoftmax".to_string()),

            // Element-wise binary operations
            Operation::Add => IrOpcode::Add,
            Operation::Sub => IrOpcode::Sub,
            Operation::Mul => IrOpcode::Mul,
            Operation::Div => IrOpcode::Div,
            Operation::Pow => IrOpcode::Intrinsic("pow".to_string()),
            Operation::Maximum => IrOpcode::Intrinsic("maximum".to_string()),
            Operation::Minimum => IrOpcode::Intrinsic("minimum".to_string()),

            // Reduction operations
            Operation::Sum { .. } => IrOpcode::Sum,
            Operation::Mean { .. } => IrOpcode::Mean,
            Operation::Max { .. } => IrOpcode::Max,
            Operation::Min { .. } => IrOpcode::Min,

            // Matrix operations
            Operation::MatMul => IrOpcode::MatMul,
            Operation::BatchMatMul => IrOpcode::Intrinsic("batch_matmul".to_string()),

            // Shape operations
            Operation::Reshape { .. } => IrOpcode::Reshape,
            Operation::Transpose { .. } => IrOpcode::Transpose,
            Operation::Squeeze { .. } => IrOpcode::Intrinsic("squeeze".to_string()),
            Operation::Unsqueeze { .. } => IrOpcode::Intrinsic("unsqueeze".to_string()),
            Operation::Slice { .. } => IrOpcode::Slice,
            Operation::Concat { .. } => IrOpcode::Concat,

            // Neural network operations
            Operation::Conv2d(_) => IrOpcode::Conv2d,
            Operation::Linear(_) => IrOpcode::Intrinsic("linear".to_string()),
            Operation::BatchNorm2d(_) => IrOpcode::Intrinsic("batch_norm".to_string()),
            Operation::Dropout { .. } => IrOpcode::Intrinsic("dropout".to_string()),
            Operation::MaxPool2d(_) => IrOpcode::Pool2d,
            Operation::AvgPool2d(_) => IrOpcode::Pool2d,

            // Special operations
            Operation::Input => {
                return Err(JitError::GraphError(
                    "Input should not be lowered".to_string(),
                ))
            }
            Operation::Parameter(_) => {
                return Err(JitError::GraphError(
                    "Parameter should not be lowered".to_string(),
                ))
            }
            Operation::Constant(_) => IrOpcode::Const,

            // Control flow (future)
            Operation::If(_) => IrOpcode::CondBr,
            Operation::While(_) => IrOpcode::Intrinsic("while".to_string()),

            // Custom operations
            Operation::Custom(name) => IrOpcode::Intrinsic(name.clone()),

            // Fused operations
            Operation::FusedKernel { name, .. } => IrOpcode::Intrinsic(name.clone()),

            // Additional control flow operations
            Operation::For(_) => IrOpcode::Intrinsic("for".to_string()),
            Operation::Break => IrOpcode::Intrinsic("break".to_string()),
            Operation::Continue => IrOpcode::Intrinsic("continue".to_string()),
            Operation::Return(_) => IrOpcode::Intrinsic("return".to_string()),
            Operation::Block(_) => IrOpcode::Intrinsic("block".to_string()),
            Operation::Merge(_) => IrOpcode::Intrinsic("merge".to_string()),

            // Additional indexing and data manipulation operations
            Operation::Split { .. } => IrOpcode::Split,
            Operation::Gather { .. } => IrOpcode::Intrinsic("gather".to_string()),
            Operation::Scatter { .. } => IrOpcode::Intrinsic("scatter".to_string()),

            // Additional normalization operations
            Operation::BatchNorm => IrOpcode::Intrinsic("batch_norm".to_string()),
            Operation::LayerNorm => IrOpcode::Intrinsic("layer_norm".to_string()),

            // Loss functions
            Operation::CrossEntropy { .. } => IrOpcode::Intrinsic("cross_entropy".to_string()),
            Operation::MSELoss => IrOpcode::Intrinsic("mse_loss".to_string()),
            Operation::BCELoss => IrOpcode::Intrinsic("bce_loss".to_string()),
            Operation::Nop => IrOpcode::Nop,
        };

        Ok(opcode)
    }
}

/// Lower a computation graph to IR
pub fn lower_graph_to_ir(graph: &ComputationGraph, module_name: String) -> JitResult<IrModule> {
    let context = LoweringContext::new(module_name);
    context.lower(graph)
}

/// IR optimization pass trait
pub trait IrPass {
    /// Name of the pass
    fn name(&self) -> &str;

    /// Apply the pass to the IR module
    fn run(&self, module: &mut IrModule) -> JitResult<bool>;
}

/// Dead code elimination pass for IR
pub struct IrDeadCodeElimination;

impl IrPass for IrDeadCodeElimination {
    fn name(&self) -> &str {
        "IrDeadCodeElimination"
    }

    fn run(&self, module: &mut IrModule) -> JitResult<bool> {
        let mut changed = false;

        // Find all live values (reachable from outputs)
        let mut live_values = std::collections::HashSet::new();

        // Mark outputs as live
        for &output in &module.outputs {
            self.mark_live(module, output, &mut live_values);
        }

        // Remove dead instructions
        for (_, block) in &mut module.blocks {
            let mut new_instructions = Vec::new();

            for instr in &block.instructions {
                let is_live = if let Some(result) = instr.result {
                    live_values.contains(&result)
                } else {
                    // Instructions without results might have side effects
                    self.has_side_effects(&instr.opcode)
                };

                if is_live {
                    new_instructions.push(instr.clone());
                } else {
                    changed = true;
                }
            }

            block.instructions = new_instructions;
        }

        Ok(changed)
    }
}

impl IrDeadCodeElimination {
    fn mark_live(
        &self,
        module: &IrModule,
        value: IrValue,
        live_values: &mut std::collections::HashSet<IrValue>,
    ) {
        Self::mark_live_recursive(module, value, live_values);
    }

    fn mark_live_recursive(
        module: &IrModule,
        value: IrValue,
        live_values: &mut std::collections::HashSet<IrValue>,
    ) {
        if live_values.contains(&value) {
            return;
        }

        live_values.insert(value);

        // Find the instruction that defines this value
        for (_, block) in &module.blocks {
            for instr in &block.instructions {
                if instr.result == Some(value) {
                    // Mark all operands as live
                    for &operand in &instr.operands {
                        Self::mark_live_recursive(module, operand, live_values);
                    }
                    break;
                }
            }
        }
    }

    fn has_side_effects(&self, opcode: &IrOpcode) -> bool {
        matches!(opcode, IrOpcode::Store | IrOpcode::Call | IrOpcode::Return)
    }
}

/// Constant folding pass for IR
pub struct IrConstantFolding;

impl IrPass for IrConstantFolding {
    fn name(&self) -> &str {
        "IrConstantFolding"
    }

    fn run(&self, module: &mut IrModule) -> JitResult<bool> {
        let mut changed = false;
        let mut folded_constants = Vec::new();

        // First pass: collect foldable instructions
        for (_, block) in &module.blocks {
            for instr in &block.instructions {
                if let Some(folded_value) = self.try_fold_instruction(module, instr) {
                    if let Some(result) = instr.result {
                        folded_constants.push((result, folded_value));
                    }
                }
            }
        }

        // Second pass: update value definitions
        for (result, folded_value) in folded_constants {
            if let Some(value_def) = module.values.get_mut(&result) {
                value_def.kind = ValueKind::Constant { data: folded_value };
                changed = true;
            }
        }

        Ok(changed)
    }
}

impl IrConstantFolding {
    fn try_fold_instruction(&self, module: &IrModule, instr: &Instruction) -> Option<ConstantData> {
        // Only fold if all operands are constants
        let operand_values: Vec<_> = instr
            .operands
            .iter()
            .filter_map(|&op| {
                module.get_value(op).and_then(|def| {
                    if let ValueKind::Constant { ref data } = def.kind {
                        Some(data.clone())
                    } else {
                        None
                    }
                })
            })
            .collect();

        if operand_values.len() != instr.operands.len() {
            return None; // Not all operands are constants
        }

        // Perform constant folding based on operation
        match (&instr.opcode, operand_values.as_slice()) {
            (IrOpcode::Add, [ConstantData::Float(a), ConstantData::Float(b)]) => {
                Some(ConstantData::Float(a + b))
            }
            (IrOpcode::Sub, [ConstantData::Float(a), ConstantData::Float(b)]) => {
                Some(ConstantData::Float(a - b))
            }
            (IrOpcode::Mul, [ConstantData::Float(a), ConstantData::Float(b)]) => {
                Some(ConstantData::Float(a * b))
            }
            (IrOpcode::Div, [ConstantData::Float(a), ConstantData::Float(b)]) => {
                if *b != 0.0 {
                    Some(ConstantData::Float(a / b))
                } else {
                    None
                }
            }
            (IrOpcode::Neg, [ConstantData::Float(a)]) => Some(ConstantData::Float(-a)),
            _ => None,
        }
    }
}

use crate::ir::Instruction;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Edge;
    use torsh_core::{DType, DeviceType};

    #[test]
    fn test_simple_graph_lowering() {
        let mut graph = ComputationGraph::new();

        // Create input -> relu -> output
        let input = graph.add_node(
            Node::new(Operation::Input, "input".to_string())
                .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[10]))])
                .with_dtypes(vec![DType::F32])
                .with_device(DeviceType::Cpu),
        );

        let relu = graph.add_node(
            Node::new(Operation::Relu, "relu".to_string())
                .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[10]))])
                .with_dtypes(vec![DType::F32])
                .with_device(DeviceType::Cpu),
        );

        graph.add_edge(input, relu, Edge::default());
        graph.add_input(input);
        graph.add_output(relu);

        // Lower to IR
        let ir_module = lower_graph_to_ir(&graph, "test_module".to_string()).unwrap();

        assert_eq!(ir_module.name, "test_module");
        assert_eq!(ir_module.inputs.len(), 1);
        assert_eq!(ir_module.outputs.len(), 1);
        assert!(!ir_module.blocks.is_empty());
    }

    #[test]
    fn test_ir_optimization() {
        let mut module = IrModule::new("test".to_string());

        // Test dead code elimination
        let dce = IrDeadCodeElimination;
        let changed = dce.run(&mut module).unwrap();

        // Empty module shouldn't change
        assert!(!changed);
    }
}
