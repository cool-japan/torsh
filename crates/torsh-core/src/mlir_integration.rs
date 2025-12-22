//! MLIR (Multi-Level Intermediate Representation) Integration
//!
//! This module provides integration with MLIR, a compiler infrastructure framework
//! that enables flexible multi-level optimization of tensor operations.
//!
//! # MLIR Overview
//!
//! MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure
//! that provides a flexible framework for building domain-specific compilers.
//! Key features include:
//!
//! - **Dialects**: Extensible operation sets for different domains
//! - **Progressive Lowering**: Transform high-level operations to low-level code
//! - **Type System**: Rich type system with attributes and constraints
//! - **Pass Infrastructure**: Composable optimization passes
//!
//! # Design Principles
//!
//! 1. **Dialect-Based**: Support multiple MLIR dialects (Tensor, Linalg, Affine, SCF)
//! 2. **Type-Safe**: Leverage Rust's type system for IR correctness
//! 3. **Composable**: Enable pass composition and optimization pipelines
//! 4. **Interoperable**: Compatible with LLVM and other compiler infrastructure
//!
//! # SciRS2 POLICY Compliance
//!
//! This module strictly follows the SciRS2 POLICY by:
//! - Only using Rust standard library and torsh-core types
//! - NO external dependencies beyond scirs2 ecosystem
//! - Providing pure Rust implementation of MLIR concepts
//!
//! # Example
//!
//! ```rust,ignore
//! use torsh_core::mlir_integration::{MlirModule, MlirBuilder, TensorDialect};
//!
//! // Build MLIR module
//! let mut builder = MlirBuilder::new();
//!
//! // Add tensor operations
//! let input = builder.add_tensor_input(&[128, 256], F32)?;
//! let weight = builder.add_tensor_constant(&[256, 512], F32)?;
//! let result = builder.add_matmul(input, weight)?;
//!
//! // Build module
//! let module = builder.build()?;
//!
//! // Apply optimizations
//! let optimized = module.optimize()?;
//! ```

use crate::dtype::DType;
use crate::error::Result;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};
#[cfg(feature = "std")]
use std::{string::String, vec::Vec};

/// MLIR Dialect identifiers
///
/// MLIR organizes operations into dialects for different abstraction levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MlirDialect {
    /// Tensor dialect for high-level tensor operations
    Tensor,

    /// Linalg (Linear Algebra) dialect for structured operations
    Linalg,

    /// Affine dialect for polyhedral optimization
    Affine,

    /// SCF (Structured Control Flow) dialect
    Scf,

    /// Arithmetic dialect for basic operations
    Arith,

    /// MemRef dialect for buffer operations
    MemRef,

    /// GPU dialect for GPU operations
    Gpu,

    /// LLVM dialect for low-level operations
    Llvm,

    /// Builtin dialect (types and operations)
    Builtin,
}

impl MlirDialect {
    /// Get dialect name
    pub fn name(&self) -> &'static str {
        match self {
            MlirDialect::Tensor => "tensor",
            MlirDialect::Linalg => "linalg",
            MlirDialect::Affine => "affine",
            MlirDialect::Scf => "scf",
            MlirDialect::Arith => "arith",
            MlirDialect::MemRef => "memref",
            MlirDialect::Gpu => "gpu",
            MlirDialect::Llvm => "llvm",
            MlirDialect::Builtin => "builtin",
        }
    }

    /// Check if dialect is high-level
    pub fn is_high_level(&self) -> bool {
        matches!(self, MlirDialect::Tensor | MlirDialect::Linalg)
    }

    /// Check if dialect is low-level
    pub fn is_low_level(&self) -> bool {
        matches!(self, MlirDialect::Llvm | MlirDialect::MemRef)
    }

    /// Get typical lowering target
    pub fn lowering_target(&self) -> Option<MlirDialect> {
        match self {
            MlirDialect::Tensor => Some(MlirDialect::Linalg),
            MlirDialect::Linalg => Some(MlirDialect::Affine),
            MlirDialect::Affine => Some(MlirDialect::Scf),
            MlirDialect::Scf => Some(MlirDialect::Llvm),
            _ => None,
        }
    }
}

/// MLIR Operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MlirOpcode {
    // Tensor dialect operations
    TensorEmpty,
    TensorExtract,
    TensorInsert,
    TensorFromElements,

    // Linalg dialect operations
    LinalgMatmul,
    LinalgDot,
    LinalgConv,
    LinalgPooling,
    LinalgBroadcast,
    LinalgTranspose,
    LinalgReduce,

    // Arithmetic operations
    ArithAddf,
    ArithSubf,
    ArithMulf,
    ArithDivf,
    ArithAddi,
    ArithSubi,
    ArithMuli,
    ArithDivi,

    // Memory operations
    MemRefAlloc,
    MemRefDealloc,
    MemRefLoad,
    MemRefStore,

    // Control flow
    ScfFor,
    ScfIf,
    ScfWhile,
    ScfParallel,

    // Function operations
    FuncFunc,
    FuncReturn,
    FuncCall,

    // Builtin operations
    ModuleOp,
    UnrealizedConversionCast,
}

impl MlirOpcode {
    /// Get operation name
    pub fn name(&self) -> &'static str {
        match self {
            MlirOpcode::TensorEmpty => "tensor.empty",
            MlirOpcode::TensorExtract => "tensor.extract",
            MlirOpcode::TensorInsert => "tensor.insert",
            MlirOpcode::TensorFromElements => "tensor.from_elements",
            MlirOpcode::LinalgMatmul => "linalg.matmul",
            MlirOpcode::LinalgDot => "linalg.dot",
            MlirOpcode::LinalgConv => "linalg.conv_2d",
            MlirOpcode::LinalgPooling => "linalg.pooling",
            MlirOpcode::LinalgBroadcast => "linalg.broadcast",
            MlirOpcode::LinalgTranspose => "linalg.transpose",
            MlirOpcode::LinalgReduce => "linalg.reduce",
            MlirOpcode::ArithAddf => "arith.addf",
            MlirOpcode::ArithSubf => "arith.subf",
            MlirOpcode::ArithMulf => "arith.mulf",
            MlirOpcode::ArithDivf => "arith.divf",
            MlirOpcode::ArithAddi => "arith.addi",
            MlirOpcode::ArithSubi => "arith.subi",
            MlirOpcode::ArithMuli => "arith.muli",
            MlirOpcode::ArithDivi => "arith.divi",
            MlirOpcode::MemRefAlloc => "memref.alloc",
            MlirOpcode::MemRefDealloc => "memref.dealloc",
            MlirOpcode::MemRefLoad => "memref.load",
            MlirOpcode::MemRefStore => "memref.store",
            MlirOpcode::ScfFor => "scf.for",
            MlirOpcode::ScfIf => "scf.if",
            MlirOpcode::ScfWhile => "scf.while",
            MlirOpcode::ScfParallel => "scf.parallel",
            MlirOpcode::FuncFunc => "func.func",
            MlirOpcode::FuncReturn => "func.return",
            MlirOpcode::FuncCall => "func.call",
            MlirOpcode::ModuleOp => "builtin.module",
            MlirOpcode::UnrealizedConversionCast => "builtin.unrealized_conversion_cast",
        }
    }

    /// Get dialect this operation belongs to
    pub fn dialect(&self) -> MlirDialect {
        match self {
            MlirOpcode::TensorEmpty
            | MlirOpcode::TensorExtract
            | MlirOpcode::TensorInsert
            | MlirOpcode::TensorFromElements => MlirDialect::Tensor,

            MlirOpcode::LinalgMatmul
            | MlirOpcode::LinalgDot
            | MlirOpcode::LinalgConv
            | MlirOpcode::LinalgPooling
            | MlirOpcode::LinalgBroadcast
            | MlirOpcode::LinalgTranspose
            | MlirOpcode::LinalgReduce => MlirDialect::Linalg,

            MlirOpcode::ArithAddf
            | MlirOpcode::ArithSubf
            | MlirOpcode::ArithMulf
            | MlirOpcode::ArithDivf
            | MlirOpcode::ArithAddi
            | MlirOpcode::ArithSubi
            | MlirOpcode::ArithMuli
            | MlirOpcode::ArithDivi => MlirDialect::Arith,

            MlirOpcode::MemRefAlloc
            | MlirOpcode::MemRefDealloc
            | MlirOpcode::MemRefLoad
            | MlirOpcode::MemRefStore => MlirDialect::MemRef,

            MlirOpcode::ScfFor
            | MlirOpcode::ScfIf
            | MlirOpcode::ScfWhile
            | MlirOpcode::ScfParallel => MlirDialect::Scf,

            MlirOpcode::FuncFunc | MlirOpcode::FuncReturn | MlirOpcode::FuncCall => {
                MlirDialect::Builtin
            }

            MlirOpcode::ModuleOp | MlirOpcode::UnrealizedConversionCast => MlirDialect::Builtin,
        }
    }
}

/// MLIR Value identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MlirValue(pub usize);

/// MLIR Operation
#[derive(Debug, Clone)]
pub struct MlirOp {
    /// Operation code
    pub opcode: MlirOpcode,

    /// Input values
    pub operands: Vec<MlirValue>,

    /// Output values
    pub results: Vec<MlirValue>,

    /// Operation attributes
    pub attributes: MlirAttributes,

    /// Result types
    pub result_types: Vec<MlirType>,
}

/// MLIR Type system
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlirType {
    /// Tensor type with shape and element type
    Tensor {
        shape: Vec<i64>,
        element_type: DType,
    },

    /// MemRef type (memory buffer reference)
    MemRef {
        shape: Vec<i64>,
        element_type: DType,
        memory_space: Option<usize>,
    },

    /// Scalar type
    Scalar(DType),

    /// Integer type
    Integer { width: u32, signed: bool },

    /// Float type
    Float { width: u32 },

    /// Index type (platform-dependent integer)
    Index,

    /// Function type
    Function {
        inputs: Vec<Box<MlirType>>,
        outputs: Vec<Box<MlirType>>,
    },

    /// None type (void)
    None,
}

impl MlirType {
    /// Create tensor type from shape and dtype
    pub fn tensor(shape: &[usize], dtype: DType) -> Self {
        MlirType::Tensor {
            shape: shape.iter().map(|&s| s as i64).collect(),
            element_type: dtype,
        }
    }

    /// Create memref type from shape and dtype
    pub fn memref(shape: &[usize], dtype: DType) -> Self {
        MlirType::MemRef {
            shape: shape.iter().map(|&s| s as i64).collect(),
            element_type: dtype,
            memory_space: None,
        }
    }

    /// Create scalar type
    pub fn scalar(dtype: DType) -> Self {
        MlirType::Scalar(dtype)
    }

    /// Create function type
    pub fn function(inputs: Vec<MlirType>, outputs: Vec<MlirType>) -> Self {
        MlirType::Function {
            inputs: inputs.into_iter().map(Box::new).collect(),
            outputs: outputs.into_iter().map(Box::new).collect(),
        }
    }

    /// Check if type is a tensor
    pub fn is_tensor(&self) -> bool {
        matches!(self, MlirType::Tensor { .. })
    }

    /// Check if type is a memref
    pub fn is_memref(&self) -> bool {
        matches!(self, MlirType::MemRef { .. })
    }

    /// Get type name for MLIR textual format
    pub fn to_mlir_string(&self) -> String {
        match self {
            MlirType::Tensor {
                shape,
                element_type,
            } => {
                let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
                format!("tensor<{}x{}>", shape_str.join("x"), element_type.name())
            }
            MlirType::MemRef {
                shape,
                element_type,
                memory_space,
            } => {
                let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
                let base = format!("memref<{}x{}>", shape_str.join("x"), element_type.name());
                if let Some(space) = memory_space {
                    format!("{}, {}>", &base[..base.len() - 1], space)
                } else {
                    base
                }
            }
            MlirType::Scalar(dtype) => dtype.name().to_string(),
            MlirType::Integer { width, signed } => {
                if *signed {
                    format!("i{}", width)
                } else {
                    format!("ui{}", width)
                }
            }
            MlirType::Float { width } => format!("f{}", width),
            MlirType::Index => "index".to_string(),
            MlirType::Function { inputs, outputs } => {
                let inputs_str: Vec<String> = inputs.iter().map(|t| t.to_mlir_string()).collect();
                let outputs_str: Vec<String> = outputs.iter().map(|t| t.to_mlir_string()).collect();
                format!(
                    "({}) -> ({})",
                    inputs_str.join(", "),
                    outputs_str.join(", ")
                )
            }
            MlirType::None => "none".to_string(),
        }
    }
}

/// MLIR Operation attributes
#[derive(Debug, Clone, Default)]
pub struct MlirAttributes {
    /// String attributes
    pub strings: Vec<(String, String)>,

    /// Integer attributes
    pub integers: Vec<(String, i64)>,

    /// Boolean attributes
    pub bools: Vec<(String, bool)>,

    /// Type attributes
    pub types: Vec<(String, MlirType)>,
}

impl MlirAttributes {
    /// Create new empty attributes
    pub fn new() -> Self {
        Self::default()
    }

    /// Add string attribute
    pub fn add_string(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.strings.push((name.into(), value.into()));
    }

    /// Add integer attribute
    pub fn add_integer(&mut self, name: impl Into<String>, value: i64) {
        self.integers.push((name.into(), value));
    }

    /// Add boolean attribute
    pub fn add_bool(&mut self, name: impl Into<String>, value: bool) {
        self.bools.push((name.into(), value));
    }

    /// Add type attribute
    pub fn add_type(&mut self, name: impl Into<String>, ty: MlirType) {
        self.types.push((name.into(), ty));
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
            && self.integers.is_empty()
            && self.bools.is_empty()
            && self.types.is_empty()
    }
}

/// MLIR Module builder
pub struct MlirBuilder {
    /// Operations in the module
    operations: Vec<MlirOp>,

    /// Next value ID
    next_value: usize,

    /// Module name
    name: String,

    /// Type registry for values
    value_types: Vec<MlirType>,
}

impl MlirBuilder {
    /// Create a new MLIR module builder
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            next_value: 0,
            name: "main".to_string(),
            value_types: Vec::new(),
        }
    }

    /// Create builder with custom name
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            operations: Vec::new(),
            next_value: 0,
            name: name.into(),
            value_types: Vec::new(),
        }
    }

    /// Allocate a new value ID and register its type
    fn allocate_value(&mut self, ty: MlirType) -> MlirValue {
        let id = MlirValue(self.next_value);
        self.value_types.push(ty);
        self.next_value += 1;
        id
    }

    /// Get the type of a value
    fn get_value_type(&self, value: MlirValue) -> Option<&MlirType> {
        self.value_types.get(value.0)
    }

    /// Add a tensor input parameter
    pub fn add_tensor_input(&mut self, shape: &[usize], dtype: DType) -> Result<MlirValue> {
        let ty = MlirType::tensor(shape, dtype);
        let result = self.allocate_value(ty.clone());

        self.operations.push(MlirOp {
            opcode: MlirOpcode::TensorEmpty,
            operands: Vec::new(),
            results: vec![result],
            attributes: MlirAttributes::new(),
            result_types: vec![ty],
        });

        Ok(result)
    }

    /// Add a tensor constant
    pub fn add_tensor_constant(&mut self, shape: &[usize], dtype: DType) -> Result<MlirValue> {
        let ty = MlirType::tensor(shape, dtype);
        let result = self.allocate_value(ty.clone());

        let mut attrs = MlirAttributes::new();
        attrs.add_string("value", "constant");

        self.operations.push(MlirOp {
            opcode: MlirOpcode::TensorFromElements,
            operands: Vec::new(),
            results: vec![result],
            attributes: attrs,
            result_types: vec![ty],
        });

        Ok(result)
    }

    /// Add matrix multiplication operation
    pub fn add_matmul(&mut self, lhs: MlirValue, rhs: MlirValue) -> Result<MlirValue> {
        // Validate and infer result type from operand types
        let result_type = match (self.get_value_type(lhs), self.get_value_type(rhs)) {
            (
                Some(MlirType::Tensor {
                    shape: lhs_shape,
                    element_type: lhs_dtype,
                }),
                Some(MlirType::Tensor {
                    shape: rhs_shape,
                    element_type: _,
                }),
            ) => {
                // Validate matmul dimensions
                if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
                    return Err(crate::error::TorshError::dimension_error(
                        "Matrix multiplication requires 2D tensors",
                        "matmul",
                    ));
                }
                if lhs_shape[1] != rhs_shape[0] {
                    return Err(crate::error::TorshError::dimension_error(
                        &format!(
                            "Incompatible dimensions for matmul: {}x{} and {}x{}",
                            lhs_shape[0], lhs_shape[1], rhs_shape[0], rhs_shape[1]
                        ),
                        "matmul",
                    ));
                }
                // Infer result type: [M, K] @ [K, N] = [M, N]
                MlirType::Tensor {
                    shape: vec![lhs_shape[0], rhs_shape[1]],
                    element_type: *lhs_dtype,
                }
            }
            _ => {
                return Err(crate::error::TorshError::dimension_error(
                    "Matmul operands must be tensors",
                    "matmul",
                ));
            }
        };

        let result = self.allocate_value(result_type.clone());

        self.operations.push(MlirOp {
            opcode: MlirOpcode::LinalgMatmul,
            operands: vec![lhs, rhs],
            results: vec![result],
            attributes: MlirAttributes::new(),
            result_types: vec![result_type],
        });

        Ok(result)
    }

    /// Add element-wise addition
    pub fn add_add(&mut self, lhs: MlirValue, rhs: MlirValue, dtype: DType) -> Result<MlirValue> {
        let result_type = MlirType::scalar(dtype);
        let result = self.allocate_value(result_type.clone());

        let opcode = match dtype {
            DType::F16 | DType::F32 | DType::F64 | DType::BF16 => MlirOpcode::ArithAddf,
            _ => MlirOpcode::ArithAddi,
        };

        self.operations.push(MlirOp {
            opcode,
            operands: vec![lhs, rhs],
            results: vec![result],
            attributes: MlirAttributes::new(),
            result_types: vec![result_type],
        });

        Ok(result)
    }

    /// Add transpose operation
    pub fn add_transpose(
        &mut self,
        operand: MlirValue,
        permutation: &[usize],
    ) -> Result<MlirValue> {
        // Infer result type from operand type and permutation
        let result_type = match self.get_value_type(operand) {
            Some(MlirType::Tensor {
                shape,
                element_type,
            }) => {
                // Validate permutation
                if permutation.len() != shape.len() {
                    return Err(crate::error::TorshError::dimension_error(
                        "Transpose permutation must match tensor rank",
                        "transpose",
                    ));
                }
                // Compute transposed shape
                let new_shape: Vec<i64> = permutation.iter().map(|&i| shape[i]).collect();
                MlirType::Tensor {
                    shape: new_shape,
                    element_type: *element_type,
                }
            }
            _ => {
                return Err(crate::error::TorshError::dimension_error(
                    "Transpose operand must be a tensor",
                    "transpose",
                ));
            }
        };

        let result = self.allocate_value(result_type.clone());

        let mut attrs = MlirAttributes::new();
        let perm_str: Vec<String> = permutation.iter().map(|p| p.to_string()).collect();
        attrs.add_string("permutation", perm_str.join(","));

        self.operations.push(MlirOp {
            opcode: MlirOpcode::LinalgTranspose,
            operands: vec![operand],
            results: vec![result],
            attributes: attrs,
            result_types: vec![result_type],
        });

        Ok(result)
    }

    /// Build the final MLIR module
    pub fn build(self) -> Result<MlirModule> {
        Ok(MlirModule {
            name: self.name,
            operations: self.operations,
        })
    }

    /// Get number of operations
    pub fn num_operations(&self) -> usize {
        self.operations.len()
    }
}

impl Default for MlirBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// MLIR Module
#[derive(Debug, Clone)]
pub struct MlirModule {
    /// Module name
    pub name: String,

    /// Operations in the module
    pub operations: Vec<MlirOp>,
}

impl MlirModule {
    /// Get module name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get all operations
    pub fn operations(&self) -> &[MlirOp] {
        &self.operations
    }

    /// Count operations by dialect
    pub fn operations_by_dialect(&self) -> Vec<(MlirDialect, usize)> {
        let mut counts: Vec<(MlirDialect, usize)> = Vec::new();
        let mut dialects = Vec::new();

        for op in &self.operations {
            let dialect = op.opcode.dialect();
            if let Some(pos) = dialects.iter().position(|&d| d == dialect) {
                counts[pos].1 += 1;
            } else {
                dialects.push(dialect);
                counts.push((dialect, 1));
            }
        }

        counts
    }

    /// Generate MLIR textual format (simplified)
    pub fn to_mlir_text(&self) -> String {
        let mut text = format!("module @{} {{\n", self.name);

        for op in self.operations.iter() {
            let result_refs: Vec<String> = op.results.iter().map(|v| format!("%{}", v.0)).collect();
            let operand_refs: Vec<String> =
                op.operands.iter().map(|v| format!("%{}", v.0)).collect();

            let line = if !result_refs.is_empty() {
                format!(
                    "  {} = {}({})\n",
                    result_refs.join(", "),
                    op.opcode.name(),
                    operand_refs.join(", ")
                )
            } else {
                format!("  {}({})\n", op.opcode.name(), operand_refs.join(", "))
            };

            text.push_str(&line);
        }

        text.push_str("}\n");
        text
    }

    /// Apply optimization passes (placeholder)
    pub fn optimize(&self) -> Result<MlirModule> {
        // Placeholder: In a real implementation, this would apply MLIR passes
        Ok(self.clone())
    }

    /// Lower to target dialect
    pub fn lower_to(&self, _target: MlirDialect) -> Result<MlirModule> {
        // Placeholder: In a real implementation, this would lower operations
        Ok(self.clone())
    }
}

/// MLIR Pass type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlirPass {
    /// Canonicalization pass
    Canonicalize,

    /// Common subexpression elimination
    CSE,

    /// Dead code elimination
    DCE,

    /// Loop fusion
    LoopFusion,

    /// Loop tiling
    LoopTiling,

    /// Buffer allocation
    BufferAllocation,

    /// Lower to LLVM
    LowerToLLVM,
}

impl MlirPass {
    /// Get pass name
    pub fn name(&self) -> &'static str {
        match self {
            MlirPass::Canonicalize => "canonicalize",
            MlirPass::CSE => "cse",
            MlirPass::DCE => "dce",
            MlirPass::LoopFusion => "loop-fusion",
            MlirPass::LoopTiling => "loop-tiling",
            MlirPass::BufferAllocation => "buffer-allocation",
            MlirPass::LowerToLLVM => "lower-to-llvm",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlir_dialect_name() {
        assert_eq!(MlirDialect::Tensor.name(), "tensor");
        assert_eq!(MlirDialect::Linalg.name(), "linalg");
    }

    #[test]
    fn test_mlir_dialect_levels() {
        assert!(MlirDialect::Tensor.is_high_level());
        assert!(!MlirDialect::Llvm.is_high_level());

        assert!(MlirDialect::Llvm.is_low_level());
        assert!(!MlirDialect::Tensor.is_low_level());
    }

    #[test]
    fn test_mlir_dialect_lowering() {
        assert_eq!(
            MlirDialect::Tensor.lowering_target(),
            Some(MlirDialect::Linalg)
        );
        assert_eq!(
            MlirDialect::Linalg.lowering_target(),
            Some(MlirDialect::Affine)
        );
    }

    #[test]
    fn test_mlir_opcode_name() {
        assert_eq!(MlirOpcode::LinalgMatmul.name(), "linalg.matmul");
        assert_eq!(MlirOpcode::ArithAddf.name(), "arith.addf");
    }

    #[test]
    fn test_mlir_opcode_dialect() {
        assert_eq!(MlirOpcode::LinalgMatmul.dialect(), MlirDialect::Linalg);
        assert_eq!(MlirOpcode::ArithAddf.dialect(), MlirDialect::Arith);
    }

    #[test]
    fn test_mlir_type_tensor() {
        let ty = MlirType::tensor(&[128, 256], DType::F32);
        assert!(ty.is_tensor());
        assert!(!ty.is_memref());
    }

    #[test]
    fn test_mlir_type_to_string() {
        let ty = MlirType::tensor(&[128, 256], DType::F32);
        let str_repr = ty.to_mlir_string();
        assert!(str_repr.contains("tensor"));
        assert!(str_repr.contains("128"));
        assert!(str_repr.contains("256"));
    }

    #[test]
    fn test_mlir_builder_input() {
        let mut builder = MlirBuilder::new();
        let input = builder.add_tensor_input(&[10, 20], DType::F32).unwrap();
        assert_eq!(input, MlirValue(0));
        assert_eq!(builder.num_operations(), 1);
    }

    #[test]
    fn test_mlir_builder_matmul() {
        let mut builder = MlirBuilder::new();
        let lhs = builder.add_tensor_input(&[128, 256], DType::F32).unwrap();
        let rhs = builder.add_tensor_input(&[256, 512], DType::F32).unwrap();
        let result = builder.add_matmul(lhs, rhs).unwrap();

        assert_eq!(result, MlirValue(2));
        assert_eq!(builder.num_operations(), 3);
    }

    #[test]
    fn test_mlir_builder_add() {
        let mut builder = MlirBuilder::new();
        let lhs = builder.add_tensor_input(&[10], DType::F32).unwrap();
        let rhs = builder.add_tensor_input(&[10], DType::F32).unwrap();
        let result = builder.add_add(lhs, rhs, DType::F32).unwrap();

        assert_eq!(result, MlirValue(2));
    }

    #[test]
    fn test_mlir_module_build() {
        let mut builder = MlirBuilder::new();
        let lhs = builder.add_tensor_input(&[128, 256], DType::F32).unwrap();
        let rhs = builder.add_tensor_input(&[256, 512], DType::F32).unwrap();
        let _result = builder.add_matmul(lhs, rhs).unwrap();

        let module = builder.build().unwrap();
        assert_eq!(module.name(), "main");
        assert_eq!(module.operations().len(), 3);
    }

    #[test]
    fn test_mlir_module_operations_by_dialect() {
        let mut builder = MlirBuilder::new();
        let lhs = builder.add_tensor_input(&[128, 256], DType::F32).unwrap();
        let rhs = builder.add_tensor_input(&[256, 512], DType::F32).unwrap();
        let _result = builder.add_matmul(lhs, rhs).unwrap();

        let module = builder.build().unwrap();
        let counts = module.operations_by_dialect();

        assert_eq!(counts.len(), 2); // Tensor and Linalg
    }

    #[test]
    fn test_mlir_module_to_text() {
        let mut builder = MlirBuilder::with_name("test");
        let lhs = builder.add_tensor_input(&[10, 20], DType::F32).unwrap();
        let rhs = builder.add_tensor_input(&[10, 20], DType::F32).unwrap();
        let _result = builder.add_add(lhs, rhs, DType::F32).unwrap();

        let module = builder.build().unwrap();
        let text = module.to_mlir_text();

        assert!(text.contains("module @test"));
        assert!(text.contains("tensor.empty"));
        assert!(text.contains("arith.addf"));
    }

    #[test]
    fn test_mlir_attributes() {
        let mut attrs = MlirAttributes::new();
        assert!(attrs.is_empty());

        attrs.add_string("name", "test");
        attrs.add_integer("value", 42);
        attrs.add_bool("flag", true);

        assert!(!attrs.is_empty());
        assert_eq!(attrs.strings.len(), 1);
        assert_eq!(attrs.integers.len(), 1);
        assert_eq!(attrs.bools.len(), 1);
    }

    #[test]
    fn test_mlir_pass_name() {
        assert_eq!(MlirPass::Canonicalize.name(), "canonicalize");
        assert_eq!(MlirPass::CSE.name(), "cse");
    }

    #[test]
    fn test_mlir_transpose() {
        let mut builder = MlirBuilder::new();
        let input = builder.add_tensor_input(&[128, 256], DType::F32).unwrap();
        let transposed = builder.add_transpose(input, &[1, 0]).unwrap();

        assert_eq!(transposed, MlirValue(1));
        assert_eq!(builder.num_operations(), 2);
    }

    #[test]
    fn test_mlir_type_function() {
        let input_types = vec![
            MlirType::tensor(&[128, 256], DType::F32),
            MlirType::tensor(&[256, 512], DType::F32),
        ];
        let output_types = vec![MlirType::tensor(&[128, 512], DType::F32)];

        let func_type = MlirType::function(input_types, output_types);
        let str_repr = func_type.to_mlir_string();

        assert!(str_repr.contains("->"));
    }

    #[test]
    fn test_complex_module() {
        let mut builder = MlirBuilder::with_name("complex");

        // Build: C = (A @ B) + D
        let a = builder.add_tensor_input(&[128, 256], DType::F32).unwrap();
        let b = builder.add_tensor_input(&[256, 512], DType::F32).unwrap();
        let d = builder.add_tensor_input(&[128, 512], DType::F32).unwrap();

        let matmul = builder.add_matmul(a, b).unwrap();
        let _result = builder.add_add(matmul, d, DType::F32).unwrap();

        let module = builder.build().unwrap();
        assert_eq!(module.operations().len(), 5);

        let dialect_counts = module.operations_by_dialect();
        assert!(dialect_counts.len() >= 2); // At least Tensor and Linalg
    }

    #[test]
    fn test_mlir_type_validation_invalid_matmul_dims() {
        let mut builder = MlirBuilder::new();
        let a = builder.add_tensor_input(&[128, 256], DType::F32).unwrap();
        let b = builder.add_tensor_input(&[128, 512], DType::F32).unwrap(); // Wrong dimension

        // Should fail: 256 != 128
        let result = builder.add_matmul(a, b);
        assert!(result.is_err());
    }

    #[test]
    fn test_mlir_type_validation_invalid_matmul_rank() {
        let mut builder = MlirBuilder::new();
        let a = builder.add_tensor_input(&[128], DType::F32).unwrap(); // 1D tensor
        let b = builder.add_tensor_input(&[128, 512], DType::F32).unwrap();

        // Should fail: 1D tensor not allowed for matmul
        let result = builder.add_matmul(a, b);
        assert!(result.is_err());
    }

    #[test]
    fn test_mlir_type_validation_invalid_transpose() {
        let mut builder = MlirBuilder::new();
        let input = builder.add_tensor_input(&[10, 20, 30], DType::F32).unwrap();

        // Should fail: permutation length doesn't match tensor rank
        let result = builder.add_transpose(input, &[1, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mlir_type_inference_matmul() {
        let mut builder = MlirBuilder::new();
        let a = builder.add_tensor_input(&[128, 256], DType::F32).unwrap();
        let b = builder.add_tensor_input(&[256, 512], DType::F64).unwrap();
        let result = builder.add_matmul(a, b).unwrap();

        // Verify result type was inferred correctly
        let result_type = builder.get_value_type(result).unwrap();
        match result_type {
            MlirType::Tensor {
                shape,
                element_type,
            } => {
                assert_eq!(shape, &[128, 512]);
                assert_eq!(*element_type, DType::F32); // Should use lhs dtype
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_mlir_type_inference_transpose() {
        let mut builder = MlirBuilder::new();
        let input = builder.add_tensor_input(&[10, 20, 30], DType::F32).unwrap();
        let result = builder.add_transpose(input, &[2, 0, 1]).unwrap();

        // Verify result shape was inferred correctly
        let result_type = builder.get_value_type(result).unwrap();
        match result_type {
            MlirType::Tensor {
                shape,
                element_type,
            } => {
                assert_eq!(shape, &[30, 10, 20]);
                assert_eq!(*element_type, DType::F32);
            }
            _ => panic!("Expected tensor type"),
        }
    }
}
