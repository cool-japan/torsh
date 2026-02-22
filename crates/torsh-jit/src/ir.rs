//! Intermediate Representation (IR) for JIT compilation
//!
//! This module provides a lower-level IR that sits between the high-level
//! computation graph and the target-specific code generation.

use crate::graph::NodeId;
use indexmap::IndexMap;
use std::collections::HashMap;
use torsh_core::{DType, Shape};

/// Intermediate Representation of a computation
#[derive(Debug, Clone)]
pub struct IrModule {
    /// Module name
    pub name: String,

    /// Input values
    pub inputs: Vec<IrValue>,

    /// Output values
    pub outputs: Vec<IrValue>,

    /// Basic blocks containing instructions
    pub blocks: IndexMap<BlockId, BasicBlock>,

    /// Entry block
    pub entry_block: BlockId,

    /// Value definitions
    pub values: IndexMap<IrValue, ValueDef>,

    /// Type definitions
    pub types: IndexMap<IrType, TypeDef>,
}

/// Basic block identifier
pub type BlockId = u32;

/// IR value identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IrValue(pub u32);

/// IR type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IrType(pub u32);

/// Basic block containing a sequence of instructions
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Block identifier
    pub id: BlockId,

    /// Block parameters (phi nodes)
    pub params: Vec<IrValue>,

    /// Instructions in execution order
    pub instructions: Vec<Instruction>,

    /// Block terminator
    pub terminator: Option<Terminator>,
}

/// Single instruction in IR
#[derive(Debug, Clone)]
pub struct Instruction {
    /// Result value (if any)
    pub result: Option<IrValue>,

    /// Operation to perform
    pub opcode: IrOpcode,

    /// Input operands
    pub operands: Vec<IrValue>,

    /// Additional attributes
    pub attrs: HashMap<String, IrAttribute>,
}

/// IR operation codes
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum IrOpcode {
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    Abs,

    // Mathematical functions
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tanh,
    Sigmoid,

    // Logical operations
    And,
    Or,
    Not,
    Xor,

    // Comparison operations
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Memory operations
    Load,
    Store,
    Alloca,

    // Tensor operations
    Reshape,
    Transpose,
    Slice,
    Concat,
    Split,

    // Matrix operations
    MatMul,
    Conv2d,
    Pool2d,

    // Reduction operations
    Sum,
    Mean,
    Max,
    Min,
    Argmax,
    Argmin,

    // Activation functions
    Relu,
    Gelu,
    Softmax,

    // Control flow
    Br,
    CondBr,
    Call,
    Return,

    // Constants
    Const,

    // Type conversions
    Cast,
    Bitcast,

    // Intrinsics
    Intrinsic(String),

    // No-operation
    Nop,
}

/// Block terminator (control flow)
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Unconditional branch
    Branch { target: BlockId },

    /// Conditional branch
    CondBranch {
        condition: IrValue,
        then_block: BlockId,
        else_block: BlockId,
    },

    /// Return from function
    Return { value: Option<IrValue> },

    /// Unreachable code
    Unreachable,
}

/// Value definition
#[derive(Debug, Clone)]
pub struct ValueDef {
    /// Value type
    pub ty: IrType,

    /// Source location (for debugging)
    pub source_node: Option<NodeId>,

    /// Value kind
    pub kind: ValueKind,
}

/// Kind of value
#[derive(Debug, Clone)]
pub enum ValueKind {
    /// Function parameter
    Parameter { index: usize },

    /// Instruction result
    Instruction { block: BlockId, index: usize },

    /// Constant value
    Constant { data: ConstantData },

    /// Undefined value
    Undef,
}

/// Constant data
#[derive(Debug, Clone)]
pub enum ConstantData {
    /// Scalar integer
    Int(i64),

    /// Scalar float
    Float(f64),

    /// Boolean
    Bool(bool),

    /// String
    String(String),

    /// Array of constants
    Array(Vec<ConstantData>),

    /// Tensor data
    Tensor { shape: Vec<usize>, data: Vec<f32> },
}

/// Type definition
#[derive(Debug, Clone)]
pub struct TypeDef {
    /// Type kind
    pub kind: TypeKind,

    /// Size in bytes (if known)
    pub size: Option<usize>,

    /// Alignment requirements
    pub align: Option<usize>,
}

/// Kind of type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeKind {
    /// Void type
    Void,

    /// Boolean type
    Bool,

    /// Integer types
    I8,
    I16,
    I32,
    I64,

    /// Unsigned integer types
    U8,
    U16,
    U32,
    U64,

    /// Floating point types
    F16,
    F32,
    F64,

    /// Complex types
    C64,
    C128,

    /// Pointer type
    Pointer {
        pointee: IrType,
    },

    /// Array type
    Array {
        element: IrType,
        length: usize,
    },

    /// Tensor type
    Tensor {
        element: IrType,
        shape: Vec<usize>,
    },

    /// Function type
    Function {
        params: Vec<IrType>,
        return_type: Option<IrType>,
    },

    /// Struct type
    Struct {
        fields: Vec<IrType>,
    },
}

/// IR attribute value
#[derive(Debug, Clone)]
pub enum IrAttribute {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Array(Vec<IrAttribute>),
}

impl IrModule {
    /// Create a new empty IR module
    pub fn new(name: String) -> Self {
        Self {
            name,
            inputs: Vec::new(),
            outputs: Vec::new(),
            blocks: IndexMap::new(),
            entry_block: 0,
            values: IndexMap::new(),
            types: IndexMap::new(),
        }
    }

    /// Add a new basic block
    pub fn add_block(&mut self) -> BlockId {
        let id = self.blocks.len() as BlockId;
        let block = BasicBlock {
            id,
            params: Vec::new(),
            instructions: Vec::new(),
            terminator: None,
        };
        self.blocks.insert(id, block);
        id
    }

    /// Add a new value
    pub fn add_value(&mut self, def: ValueDef) -> IrValue {
        let id = IrValue(self.values.len() as u32);
        self.values.insert(id, def);
        id
    }

    /// Add a value (for external access)
    pub fn add_value_external(&mut self, def: ValueDef) -> IrValue {
        self.add_value(def)
    }

    /// Add a new type
    pub fn add_type(&mut self, def: TypeDef) -> IrType {
        // Check if type already exists
        for (&existing_id, existing_def) in &self.types {
            if existing_def.kind == def.kind {
                return existing_id;
            }
        }

        let id = IrType(self.types.len() as u32);
        self.types.insert(id, def);
        id
    }

    /// Get a block by ID
    pub fn get_block(&self, id: BlockId) -> Option<&BasicBlock> {
        self.blocks.get(&id)
    }

    /// Get a mutable block by ID
    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut BasicBlock> {
        self.blocks.get_mut(&id)
    }

    /// Get value definition
    pub fn get_value(&self, value: IrValue) -> Option<&ValueDef> {
        self.values.get(&value)
    }

    /// Get type definition
    pub fn get_type(&self, ty: IrType) -> Option<&TypeDef> {
        self.types.get(&ty)
    }

    /// Validate the IR module
    pub fn validate(&self) -> Result<(), String> {
        // Check entry block exists
        if !self.blocks.contains_key(&self.entry_block) {
            return Err("Entry block does not exist".to_string());
        }

        // Validate each block
        for (id, block) in &self.blocks {
            if *id != block.id {
                return Err(format!("Block ID mismatch: {} != {}", id, block.id));
            }

            // Validate instructions
            for (i, instr) in block.instructions.iter().enumerate() {
                self.validate_instruction(instr, *id, i)?;
            }

            // Validate terminator
            if let Some(ref term) = block.terminator {
                self.validate_terminator(term)?;
            }
        }

        // Validate values
        for (value, def) in &self.values {
            self.validate_value(*value, def)?;
        }

        Ok(())
    }

    fn validate_instruction(
        &self,
        instr: &Instruction,
        block_id: BlockId,
        _index: usize,
    ) -> Result<(), String> {
        // Check operands exist
        for &operand in &instr.operands {
            if !self.values.contains_key(&operand) {
                return Err(format!(
                    "Operand {:?} not found in block {}",
                    operand, block_id
                ));
            }
        }

        // Check result type consistency
        if let Some(result) = instr.result {
            if !self.values.contains_key(&result) {
                return Err(format!(
                    "Result {:?} not found in block {}",
                    result, block_id
                ));
            }
        }

        Ok(())
    }

    fn validate_terminator(&self, term: &Terminator) -> Result<(), String> {
        match term {
            Terminator::Branch { target } => {
                if !self.blocks.contains_key(target) {
                    return Err(format!("Branch target block {} does not exist", target));
                }
            }
            Terminator::CondBranch {
                condition,
                then_block,
                else_block,
            } => {
                // Validate condition value exists
                if !self.values.contains_key(condition) {
                    return Err(format!("Condition value {:?} does not exist", condition));
                }
                // Validate target blocks exist
                if !self.blocks.contains_key(then_block) {
                    return Err(format!("Then block {} does not exist", then_block));
                }
                if !self.blocks.contains_key(else_block) {
                    return Err(format!("Else block {} does not exist", else_block));
                }
            }
            Terminator::Return { value } => {
                if let Some(val) = value {
                    if !self.values.contains_key(val) {
                        return Err(format!("Return value {:?} does not exist", val));
                    }
                }
            }
            Terminator::Unreachable => {
                // Nothing to validate for unreachable
            }
        }
        Ok(())
    }

    fn validate_value(&self, _value: IrValue, def: &ValueDef) -> Result<(), String> {
        // Check type exists
        if !self.types.contains_key(&def.ty) {
            return Err(format!("Type {:?} not found", def.ty));
        }

        Ok(())
    }

    /// Get a function-like interface for debugging compatibility
    /// For now, treat the entry block as the main "function"
    pub fn get_function(&self, _name: &str) -> Option<&BasicBlock> {
        self.blocks.get(&self.entry_block)
    }

    /// Inline small functions (placeholder implementation)
    pub fn inline_small_functions(&mut self) -> crate::JitResult<()> {
        // Placeholder implementation for function inlining
        // This would analyze the module and inline small functions
        Ok(())
    }

    /// Get all functions in the module (returns an iterator over blocks as functions)
    pub fn functions_mut(&mut self) -> impl Iterator<Item = &mut BasicBlock> {
        self.blocks.values_mut()
    }

    /// Get all instructions in the module
    pub fn instructions(&self) -> impl Iterator<Item = &Instruction> {
        self.blocks
            .values()
            .flat_map(|block| block.instructions.iter())
    }

    /// Get all instructions in the module (mutable)
    pub fn instructions_mut(&mut self) -> impl Iterator<Item = &mut Instruction> {
        self.blocks
            .values_mut()
            .flat_map(|block| block.instructions.iter_mut())
    }

    /// Retain instructions that satisfy the predicate
    pub fn retain_instructions<F>(&mut self, mut predicate: F)
    where
        F: FnMut(usize, &Instruction) -> bool,
    {
        let mut global_idx = 0;
        for block in self.blocks.values_mut() {
            block.instructions.retain(|instruction| {
                let keep = predicate(global_idx, instruction);
                global_idx += 1;
                keep
            });
        }
    }

    /// Remove unused functions (placeholder implementation)
    pub fn remove_unused_functions(&mut self) -> crate::JitResult<()> {
        // Placeholder - in a real implementation, this would analyze function usage
        // and remove unused function blocks
        Ok(())
    }
}

impl BasicBlock {
    /// Get the instructions in this block
    pub fn instructions(&self) -> &Vec<Instruction> {
        &self.instructions
    }
}

impl Instruction {
    /// Check if this instruction produces a value
    pub fn produces_value(&self) -> bool {
        self.result.is_some()
    }

    /// Get the operands of this instruction
    pub fn operands(&self) -> &Vec<IrValue> {
        &self.operands
    }
}

/// IR builder for constructing IR modules
pub struct IrBuilder {
    pub module: IrModule,
    current_block: Option<BlockId>,
    #[allow(dead_code)]
    value_counter: u32,
    type_cache: HashMap<TypeKind, IrType>,
}

impl IrBuilder {
    /// Create a new IR builder
    pub fn new(module_name: String) -> Self {
        Self {
            module: IrModule::new(module_name),
            current_block: None,
            value_counter: 0,
            type_cache: HashMap::new(),
        }
    }

    /// Set current block for instruction insertion
    pub fn set_current_block(&mut self, block: BlockId) {
        self.current_block = Some(block);
    }

    /// Add a new basic block and set it as current
    pub fn add_block(&mut self) -> BlockId {
        let id = self.module.add_block();
        self.current_block = Some(id);
        id
    }

    /// Get or create a type
    pub fn get_type(&mut self, kind: TypeKind) -> IrType {
        if let Some(&existing) = self.type_cache.get(&kind) {
            return existing;
        }

        let size = match &kind {
            TypeKind::Void => Some(0),
            TypeKind::Bool | TypeKind::I8 | TypeKind::U8 => Some(1),
            TypeKind::I16 | TypeKind::U16 | TypeKind::F16 => Some(2),
            TypeKind::I32 | TypeKind::U32 | TypeKind::F32 => Some(4),
            TypeKind::I64 | TypeKind::U64 | TypeKind::F64 | TypeKind::C64 => Some(8),
            TypeKind::C128 => Some(16),
            _ => None,
        };

        let ty = self.module.add_type(TypeDef {
            kind: kind.clone(),
            size,
            align: size,
        });

        self.type_cache.insert(kind, ty);
        ty
    }

    /// Create a constant value
    pub fn const_int(&mut self, value: i64, ty: IrType) -> IrValue {
        let val_def = ValueDef {
            ty,
            source_node: None,
            kind: ValueKind::Constant {
                data: ConstantData::Int(value),
            },
        };
        self.module.add_value(val_def)
    }

    /// Create a constant float
    pub fn const_float(&mut self, value: f64, ty: IrType) -> IrValue {
        let val_def = ValueDef {
            ty,
            source_node: None,
            kind: ValueKind::Constant {
                data: ConstantData::Float(value),
            },
        };
        self.module.add_value(val_def)
    }

    /// Add an instruction to the current block
    pub fn add_instruction(
        &mut self,
        opcode: IrOpcode,
        operands: Vec<IrValue>,
        result_type: Option<IrType>,
    ) -> Option<IrValue> {
        let current_block = self.current_block.expect("No current block set");

        let result = if let Some(ty) = result_type {
            let val_def = ValueDef {
                ty,
                source_node: None,
                kind: ValueKind::Instruction {
                    block: current_block,
                    index: 0, // Will be updated
                },
            };
            Some(self.module.add_value(val_def))
        } else {
            None
        };

        let instr = Instruction {
            result,
            opcode,
            operands,
            attrs: HashMap::new(),
        };

        if let Some(block) = self.module.get_block_mut(current_block) {
            block.instructions.push(instr);
        }

        result
    }

    /// Set block terminator
    pub fn set_terminator(&mut self, terminator: Terminator) {
        if let Some(current_block) = self.current_block {
            if let Some(block) = self.module.get_block_mut(current_block) {
                block.terminator = Some(terminator);
            }
        }
    }

    /// Build the final IR module
    pub fn build(self) -> IrModule {
        self.module
    }
}

/// Convert a torsh DType to IR type
pub fn dtype_to_ir_type(dtype: DType) -> TypeKind {
    match dtype {
        DType::Bool => TypeKind::Bool,
        DType::I8 => TypeKind::I8,
        DType::I16 => TypeKind::I16,
        DType::I32 => TypeKind::I32,
        DType::I64 => TypeKind::I64,
        DType::U8 => TypeKind::U8,
        DType::U32 => TypeKind::U32,
        DType::U64 => TypeKind::U64,
        DType::F16 => TypeKind::F16,
        DType::BF16 => TypeKind::F16, // Approximate
        DType::F32 => TypeKind::F32,
        DType::F64 => TypeKind::F64,
        DType::C64 => TypeKind::C64,
        DType::C128 => TypeKind::C128,
        DType::QInt8 => TypeKind::I8,   // Quantized as regular int8
        DType::QUInt8 => TypeKind::U8,  // Quantized as regular uint8
        DType::QInt32 => TypeKind::I32, // Quantized as regular int32
    }
}

/// Convert a shape and dtype to tensor type
pub fn shape_dtype_to_tensor_type(shape: &Shape, dtype: DType) -> TypeKind {
    let _element_type_kind = dtype_to_ir_type(dtype);
    TypeKind::Tensor {
        element: IrType(0), // Placeholder, will be resolved by builder
        shape: shape.dims().to_vec(),
    }
}

// Compatibility type aliases for advanced features
pub type IrFunction = IrModule;
pub type IrInstruction = Instruction;

// Additional compatibility types that might be needed
pub type InterproceduralResult<T> = Result<T, String>;
pub type AnalysisResult<T> = Result<T, String>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_module_creation() {
        let module = IrModule::new("test".to_string());
        assert_eq!(module.name, "test");
        assert!(module.blocks.is_empty());
        assert!(module.values.is_empty());
    }

    #[test]
    fn test_ir_builder() {
        let mut builder = IrBuilder::new("test_module".to_string());

        // Create types
        let i32_type = builder.get_type(TypeKind::I32);
        let f32_type = builder.get_type(TypeKind::F32);

        // Create a block
        let block = builder.add_block();
        assert_eq!(block, 0);

        // Create constants
        let const1 = builder.const_int(42, i32_type);
        let const2 = builder.const_float(3.14, f32_type);

        // Add instruction
        let result = builder.add_instruction(IrOpcode::Add, vec![const1, const2], Some(f32_type));
        assert!(result.is_some());

        // Build module
        let module = builder.build();
        assert_eq!(module.name, "test_module");
        assert!(!module.blocks.is_empty());
        assert!(!module.values.is_empty());
    }

    #[test]
    fn test_module_validation() {
        let mut builder = IrBuilder::new("valid_module".to_string());
        let _block = builder.add_block();
        builder.set_terminator(Terminator::Return { value: None });

        let module = builder.build();
        assert!(module.validate().is_ok());
    }
}
