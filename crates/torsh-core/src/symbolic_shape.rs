//! Symbolic shape support for dynamic graphs
//!
//! This module provides symbolic shape representation and inference for dynamic computation graphs.
//! Symbolic shapes allow tensors to have dimensions that are unknown at graph construction time
//! but can be inferred or constrained through operations.
//!
//! # Features
//! - Symbolic dimension representation (unknown, constrained, expression-based)
//! - Shape inference through operation constraints
//! - Broadcasting with symbolic dimensions
//! - Runtime shape validation and materialization
//! - Dimension constraint solving

use crate::error::{Result, TorshError};
use crate::shape::Shape;

#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as HashMap;

/// Symbolic dimension identifier
pub type SymbolId = u64;

/// Symbolic dimension representing a runtime-determined size
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicDim {
    /// Concrete dimension with known size
    Concrete(usize),

    /// Unknown dimension (will be determined at runtime)
    Unknown { id: SymbolId, name: Option<String> },

    /// Dimension with minimum/maximum constraints
    Constrained {
        id: SymbolId,
        name: Option<String>,
        min: Option<usize>,
        max: Option<usize>,
    },

    /// Expression-based dimension (e.g., 2*N, N+1)
    Expression {
        id: SymbolId,
        expr: Box<DimExpression>,
    },

    /// Dimension that equals another dimension
    Aliased { id: SymbolId, alias_of: SymbolId },
}

impl SymbolicDim {
    /// Create a concrete dimension
    pub fn concrete(size: usize) -> Self {
        Self::Concrete(size)
    }

    /// Create an unknown dimension
    pub fn unknown(id: SymbolId, name: Option<String>) -> Self {
        Self::Unknown { id, name }
    }

    /// Create a constrained dimension
    pub fn constrained(
        id: SymbolId,
        name: Option<String>,
        min: Option<usize>,
        max: Option<usize>,
    ) -> Self {
        Self::Constrained { id, name, min, max }
    }

    /// Create an expression-based dimension
    pub fn expression(id: SymbolId, expr: DimExpression) -> Self {
        Self::Expression {
            id,
            expr: Box::new(expr),
        }
    }

    /// Create an aliased dimension
    pub fn aliased(id: SymbolId, alias_of: SymbolId) -> Self {
        Self::Aliased { id, alias_of }
    }

    /// Check if this is a concrete dimension
    pub fn is_concrete(&self) -> bool {
        matches!(self, Self::Concrete(_))
    }

    /// Get the concrete value if available
    pub fn as_concrete(&self) -> Option<usize> {
        match self {
            Self::Concrete(size) => Some(*size),
            _ => None,
        }
    }

    /// Get the symbol ID
    pub fn symbol_id(&self) -> Option<SymbolId> {
        match self {
            Self::Concrete(_) => None,
            Self::Unknown { id, .. }
            | Self::Constrained { id, .. }
            | Self::Expression { id, .. }
            | Self::Aliased { id, .. } => Some(*id),
        }
    }

    /// Get the dimension name
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Unknown { name, .. } | Self::Constrained { name, .. } => {
                name.as_ref().map(|s| s.as_str())
            }
            _ => None,
        }
    }

    /// Check if dimension satisfies constraints
    pub fn satisfies_constraints(&self, value: usize) -> bool {
        match self {
            Self::Concrete(size) => value == *size,
            Self::Unknown { .. } => true,
            Self::Constrained { min, max, .. } => {
                let min_ok = min.map(|m| value >= m).unwrap_or(true);
                let max_ok = max.map(|m| value <= m).unwrap_or(true);
                min_ok && max_ok
            }
            _ => true, // Expression and aliased checked elsewhere
        }
    }

    /// Try to unify two symbolic dimensions
    pub fn unify(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            // Concrete dimensions must match
            (Self::Concrete(a), Self::Concrete(b)) => {
                if a == b {
                    Ok(Self::Concrete(*a))
                } else {
                    Err(TorshError::ShapeMismatch {
                        expected: vec![*a],
                        got: vec![*b],
                    })
                }
            }

            // Concrete with symbolic -> constrain to concrete
            (Self::Concrete(size), sym) | (sym, Self::Concrete(size)) => {
                if sym.satisfies_constraints(*size) {
                    Ok(Self::Concrete(*size))
                } else {
                    Err(TorshError::InvalidShape(format!(
                        "Dimension {} violates constraints",
                        size
                    )))
                }
            }

            // Unknown dimensions -> keep one with constraints if available
            (Self::Unknown { id, name }, other) | (other, Self::Unknown { id, name }) => {
                match other {
                    Self::Unknown { .. } => Ok(Self::Unknown {
                        id: *id,
                        name: name.clone(),
                    }),
                    Self::Constrained {
                        min, max, name, id, ..
                    } => Ok(Self::Constrained {
                        id: *id,
                        name: name.clone(),
                        min: *min,
                        max: *max,
                    }),
                    _ => Ok(self.clone()),
                }
            }

            // Constrained dimensions -> merge constraints
            (
                Self::Constrained {
                    id,
                    name,
                    min: min1,
                    max: max1,
                },
                Self::Constrained {
                    min: min2,
                    max: max2,
                    ..
                },
            ) => {
                let min = match (min1, min2) {
                    (Some(a), Some(b)) => Some((*a).max(*b)),
                    (Some(a), None) | (None, Some(a)) => Some(*a),
                    (None, None) => None,
                };

                let max = match (max1, max2) {
                    (Some(a), Some(b)) => Some((*a).min(*b)),
                    (Some(a), None) | (None, Some(a)) => Some(*a),
                    (None, None) => None,
                };

                // Check validity
                if let (Some(min_val), Some(max_val)) = (min, max) {
                    if min_val > max_val {
                        return Err(TorshError::InvalidShape(
                            "Incompatible constraints".to_string(),
                        ));
                    }
                }

                Ok(Self::Constrained {
                    id: *id,
                    name: name.clone(),
                    min,
                    max,
                })
            }

            _ => Ok(self.clone()), // Default: keep first
        }
    }
}

/// Dimension expression for computed dimensions
#[derive(Debug, Clone, PartialEq)]
pub enum DimExpression {
    /// Reference to a symbol
    Symbol(SymbolId),

    /// Constant value
    Constant(i64),

    /// Addition
    Add(Box<DimExpression>, Box<DimExpression>),

    /// Subtraction
    Sub(Box<DimExpression>, Box<DimExpression>),

    /// Multiplication
    Mul(Box<DimExpression>, Box<DimExpression>),

    /// Division (integer division)
    Div(Box<DimExpression>, Box<DimExpression>),

    /// Modulo
    Mod(Box<DimExpression>, Box<DimExpression>),

    /// Maximum of two expressions
    Max(Box<DimExpression>, Box<DimExpression>),

    /// Minimum of two expressions
    Min(Box<DimExpression>, Box<DimExpression>),
}

impl DimExpression {
    /// Evaluate expression given symbol values
    pub fn eval(&self, symbols: &HashMap<SymbolId, usize>) -> Result<usize> {
        let result = match self {
            Self::Symbol(id) => symbols
                .get(id)
                .copied()
                .ok_or_else(|| TorshError::InvalidShape(format!("Unknown symbol: {}", id)))?,

            Self::Constant(val) => {
                if *val < 0 {
                    return Err(TorshError::InvalidShape(format!(
                        "Negative dimension: {}",
                        val
                    )));
                }
                *val as usize
            }

            Self::Add(a, b) => a.eval(symbols)? + b.eval(symbols)?,
            Self::Sub(a, b) => {
                let a_val = a.eval(symbols)?;
                let b_val = b.eval(symbols)?;
                a_val.checked_sub(b_val).ok_or_else(|| {
                    TorshError::InvalidShape("Subtraction resulted in negative value".to_string())
                })?
            }
            Self::Mul(a, b) => a.eval(symbols)? * b.eval(symbols)?,
            Self::Div(a, b) => {
                let b_val = b.eval(symbols)?;
                if b_val == 0 {
                    return Err(TorshError::InvalidShape("Division by zero".to_string()));
                }
                a.eval(symbols)? / b_val
            }
            Self::Mod(a, b) => {
                let b_val = b.eval(symbols)?;
                if b_val == 0 {
                    return Err(TorshError::InvalidShape("Modulo by zero".to_string()));
                }
                a.eval(symbols)? % b_val
            }
            Self::Max(a, b) => a.eval(symbols)?.max(b.eval(symbols)?),
            Self::Min(a, b) => a.eval(symbols)?.min(b.eval(symbols)?),
        };

        Ok(result)
    }

    /// Get all symbol IDs referenced in this expression
    pub fn referenced_symbols(&self) -> Vec<SymbolId> {
        let mut symbols = Vec::new();
        self.collect_symbols(&mut symbols);
        symbols.sort();
        symbols.dedup();
        symbols
    }

    fn collect_symbols(&self, symbols: &mut Vec<SymbolId>) {
        match self {
            Self::Symbol(id) => symbols.push(*id),
            Self::Constant(_) => {}
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Mod(a, b)
            | Self::Max(a, b)
            | Self::Min(a, b) => {
                a.collect_symbols(symbols);
                b.collect_symbols(symbols);
            }
        }
    }
}

/// Symbolic shape with symbolic dimensions
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolicShape {
    /// Dimensions (may be symbolic)
    dims: Vec<SymbolicDim>,

    /// Shape name for debugging
    name: Option<String>,
}

impl SymbolicShape {
    /// Create a symbolic shape
    pub fn new(dims: Vec<SymbolicDim>) -> Self {
        Self { dims, name: None }
    }

    /// Create from concrete shape
    pub fn from_concrete(shape: &Shape) -> Self {
        let dims = shape
            .dims()
            .iter()
            .map(|&d| SymbolicDim::Concrete(d))
            .collect();
        Self { dims, name: None }
    }

    /// Set shape name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Get dimensions
    pub fn dims(&self) -> &[SymbolicDim] {
        &self.dims
    }

    /// Check if all dimensions are concrete
    pub fn is_concrete(&self) -> bool {
        self.dims.iter().all(|d| d.is_concrete())
    }

    /// Try to materialize to concrete shape
    pub fn materialize(&self, symbols: &HashMap<SymbolId, usize>) -> Result<Shape> {
        let mut concrete_dims = Vec::with_capacity(self.dims.len());

        for dim in &self.dims {
            let size = match dim {
                SymbolicDim::Concrete(size) => *size,
                SymbolicDim::Unknown { id, .. }
                | SymbolicDim::Constrained { id, .. }
                | SymbolicDim::Aliased { id, .. } => symbols.get(id).copied().ok_or_else(|| {
                    TorshError::InvalidShape(format!("Unresolved symbol: {}", id))
                })?,
                SymbolicDim::Expression { expr, .. } => expr.eval(symbols)?,
            };

            // Validate constraints
            if !dim.satisfies_constraints(size) {
                return Err(TorshError::InvalidShape(format!(
                    "Value {} violates constraints for dimension",
                    size
                )));
            }

            concrete_dims.push(size);
        }

        Ok(Shape::new(concrete_dims))
    }

    /// Unify with another symbolic shape
    pub fn unify(&self, other: &Self) -> Result<Self> {
        if self.ndim() != other.ndim() {
            return Err(TorshError::InvalidShape(format!(
                "Cannot unify shapes with different ranks: {} vs {}",
                self.ndim(),
                other.ndim()
            )));
        }

        let mut unified_dims = Vec::with_capacity(self.ndim());
        for (d1, d2) in self.dims.iter().zip(other.dims.iter()) {
            unified_dims.push(d1.unify(d2)?);
        }

        Ok(Self::new(unified_dims))
    }

    /// Get all symbol IDs used in this shape
    pub fn symbol_ids(&self) -> Vec<SymbolId> {
        let mut ids = Vec::new();
        for dim in &self.dims {
            if let Some(id) = dim.symbol_id() {
                ids.push(id);
            }
            if let SymbolicDim::Expression { expr, .. } = dim {
                ids.extend(expr.referenced_symbols());
            }
        }
        ids.sort();
        ids.dedup();
        ids
    }

    /// Broadcast with another symbolic shape
    pub fn broadcast_with(&self, other: &Self) -> Result<Self> {
        let max_ndim = self.ndim().max(other.ndim());
        let mut result_dims = Vec::with_capacity(max_ndim);

        // Pad shorter shape with 1s from the left
        let self_padded = self.pad_left(max_ndim);
        let other_padded = other.pad_left(max_ndim);

        for (d1, d2) in self_padded.iter().zip(other_padded.iter()) {
            // Broadcasting rules:
            // - If one dimension is 1, result is the other dimension
            // - If both are equal, result is that dimension
            // - Otherwise, try to unify

            match (d1.as_concrete(), d2.as_concrete()) {
                (Some(1), _) => result_dims.push(d2.clone()),
                (_, Some(1)) => result_dims.push(d1.clone()),
                (Some(a), Some(b)) => {
                    if a == b {
                        result_dims.push(d1.clone());
                    } else {
                        return Err(TorshError::BroadcastError {
                            shape1: vec![a],
                            shape2: vec![b],
                        });
                    }
                }
                _ => {
                    // Symbolic dimensions - try to unify
                    result_dims.push(d1.unify(d2)?);
                }
            }
        }

        Ok(Self::new(result_dims))
    }

    fn pad_left(&self, target_ndim: usize) -> Vec<SymbolicDim> {
        if self.ndim() >= target_ndim {
            return self.dims.clone();
        }

        let mut padded = vec![SymbolicDim::Concrete(1); target_ndim - self.ndim()];
        padded.extend(self.dims.iter().cloned());
        padded
    }
}

/// Symbol registry for managing symbolic dimensions
#[derive(Debug, Clone)]
pub struct SymbolRegistry {
    next_id: SymbolId,
    symbols: HashMap<SymbolId, SymbolInfo>,
    named_symbols: HashMap<String, SymbolId>,
}

#[derive(Debug, Clone)]
struct SymbolInfo {
    name: Option<String>,
    value: Option<usize>,
    constraints: Option<(Option<usize>, Option<usize>)>,
}

impl SymbolRegistry {
    /// Create a new symbol registry
    pub fn new() -> Self {
        Self {
            next_id: 0,
            symbols: HashMap::new(),
            named_symbols: HashMap::new(),
        }
    }

    /// Create a new symbol
    pub fn create_symbol(&mut self, name: Option<String>) -> SymbolId {
        let id = self.next_id;
        self.next_id += 1;

        let info = SymbolInfo {
            name: name.clone(),
            value: None,
            constraints: None,
        };

        self.symbols.insert(id, info);

        if let Some(ref n) = name {
            self.named_symbols.insert(n.clone(), id);
        }

        id
    }

    /// Get symbol by name
    pub fn get_symbol(&self, name: &str) -> Option<SymbolId> {
        self.named_symbols.get(name).copied()
    }

    /// Set symbol value
    pub fn set_value(&mut self, id: SymbolId, value: usize) -> Result<()> {
        if let Some(info) = self.symbols.get_mut(&id) {
            // Check constraints
            if let Some((min, max)) = info.constraints {
                if let Some(min_val) = min {
                    if value < min_val {
                        return Err(TorshError::InvalidShape(format!(
                            "Value {} below minimum {}",
                            value, min_val
                        )));
                    }
                }
                if let Some(max_val) = max {
                    if value > max_val {
                        return Err(TorshError::InvalidShape(format!(
                            "Value {} above maximum {}",
                            value, max_val
                        )));
                    }
                }
            }

            info.value = Some(value);
            Ok(())
        } else {
            Err(TorshError::InvalidShape(format!("Unknown symbol: {}", id)))
        }
    }

    /// Get symbol value
    pub fn get_value(&self, id: SymbolId) -> Option<usize> {
        self.symbols.get(&id).and_then(|info| info.value)
    }

    /// Set symbol constraints
    pub fn set_constraints(
        &mut self,
        id: SymbolId,
        min: Option<usize>,
        max: Option<usize>,
    ) -> Result<()> {
        if let Some(info) = self.symbols.get_mut(&id) {
            info.constraints = Some((min, max));
            Ok(())
        } else {
            Err(TorshError::InvalidShape(format!("Unknown symbol: {}", id)))
        }
    }

    /// Get all symbol values
    pub fn values(&self) -> HashMap<SymbolId, usize> {
        self.symbols
            .iter()
            .filter_map(|(id, info)| info.value.map(|v| (*id, v)))
            .collect()
    }

    /// Get symbol name
    pub fn get_name(&self, id: SymbolId) -> Option<&str> {
        self.symbols.get(&id).and_then(|info| info.name.as_deref())
    }
}

impl Default for SymbolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Shape inference engine for symbolic shapes
#[derive(Debug)]
pub struct ShapeInference {
    registry: SymbolRegistry,
}

impl ShapeInference {
    /// Create a new shape inference engine
    pub fn new() -> Self {
        Self {
            registry: SymbolRegistry::new(),
        }
    }

    /// Create a symbolic dimension
    pub fn create_dim(&mut self, name: Option<String>) -> SymbolicDim {
        let id = self.registry.create_symbol(name.clone());
        SymbolicDim::unknown(id, name)
    }

    /// Infer shape for binary operation
    pub fn infer_binary_op(
        &mut self,
        left: &SymbolicShape,
        right: &SymbolicShape,
    ) -> Result<SymbolicShape> {
        left.broadcast_with(right)
    }

    /// Infer shape for matmul operation
    pub fn infer_matmul(
        &mut self,
        left: &SymbolicShape,
        right: &SymbolicShape,
    ) -> Result<SymbolicShape> {
        if left.ndim() < 2 || right.ndim() < 2 {
            return Err(TorshError::InvalidShape(
                "Matmul requires at least 2D tensors".to_string(),
            ));
        }

        // Check inner dimensions match
        let left_inner = &left.dims()[left.ndim() - 1];
        let right_inner = &right.dims()[right.ndim() - 2];
        left_inner.unify(right_inner)?;

        // Result shape: [...batch dims, M, N]
        let mut result_dims = Vec::new();

        // Broadcast batch dimensions
        if left.ndim() > 2 || right.ndim() > 2 {
            let left_batch = &left.dims()[..left.ndim() - 2];
            let right_batch = &right.dims()[..right.ndim() - 2];

            let batch_shape = SymbolicShape::new(left_batch.to_vec());
            let right_batch_shape = SymbolicShape::new(right_batch.to_vec());
            let broadcasted_batch = batch_shape.broadcast_with(&right_batch_shape)?;

            result_dims.extend(broadcasted_batch.dims().iter().cloned());
        }

        // Add output dimensions
        result_dims.push(left.dims()[left.ndim() - 2].clone());
        result_dims.push(right.dims()[right.ndim() - 1].clone());

        Ok(SymbolicShape::new(result_dims))
    }

    /// Materialize symbolic shape to concrete shape
    pub fn materialize(&self, shape: &SymbolicShape) -> Result<Shape> {
        shape.materialize(&self.registry.values())
    }

    /// Set symbol value
    pub fn set_symbol_value(&mut self, id: SymbolId, value: usize) -> Result<()> {
        self.registry.set_value(id, value)
    }
}

impl Default for ShapeInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_dim_creation() {
        let concrete = SymbolicDim::concrete(10);
        assert!(concrete.is_concrete());
        assert_eq!(concrete.as_concrete(), Some(10));

        let unknown = SymbolicDim::unknown(0, Some("N".to_string()));
        assert!(!unknown.is_concrete());
        assert_eq!(unknown.name(), Some("N"));
    }

    #[test]
    fn test_dim_expression_eval() {
        let mut symbols = HashMap::new();
        symbols.insert(0, 10);
        symbols.insert(1, 5);

        let expr = DimExpression::Add(
            Box::new(DimExpression::Symbol(0)),
            Box::new(DimExpression::Constant(2)),
        );
        assert_eq!(expr.eval(&symbols).unwrap(), 12);

        let expr = DimExpression::Mul(
            Box::new(DimExpression::Symbol(1)),
            Box::new(DimExpression::Constant(2)),
        );
        assert_eq!(expr.eval(&symbols).unwrap(), 10);
    }

    #[test]
    fn test_symbolic_shape_materialization() {
        let shape = SymbolicShape::new(vec![
            SymbolicDim::Concrete(2),
            SymbolicDim::unknown(0, Some("N".to_string())),
            SymbolicDim::Concrete(3),
        ]);

        let mut symbols = HashMap::new();
        symbols.insert(0, 10);

        let concrete = shape.materialize(&symbols).unwrap();
        assert_eq!(concrete.dims(), &[2, 10, 3]);
    }

    #[test]
    fn test_symbolic_broadcasting() {
        let shape1 = SymbolicShape::new(vec![
            SymbolicDim::Concrete(1),
            SymbolicDim::unknown(0, None),
        ]);

        let shape2 = SymbolicShape::new(vec![SymbolicDim::Concrete(5), SymbolicDim::Concrete(3)]);

        let result = shape1.broadcast_with(&shape2).unwrap();
        assert_eq!(result.ndim(), 2);
        assert_eq!(result.dims()[0].as_concrete(), Some(5));
    }

    #[test]
    fn test_shape_inference_matmul() {
        let mut inference = ShapeInference::new();

        let left = SymbolicShape::new(vec![
            SymbolicDim::Concrete(2),
            SymbolicDim::Concrete(3),
            SymbolicDim::Concrete(4),
        ]);

        let right = SymbolicShape::new(vec![SymbolicDim::Concrete(4), SymbolicDim::Concrete(5)]);

        let result = inference.infer_matmul(&left, &right).unwrap();
        assert_eq!(result.ndim(), 3);
        assert_eq!(result.dims()[0].as_concrete(), Some(2));
        assert_eq!(result.dims()[1].as_concrete(), Some(3));
        assert_eq!(result.dims()[2].as_concrete(), Some(5));
    }

    #[test]
    fn test_symbol_registry() {
        let mut registry = SymbolRegistry::new();

        let id = registry.create_symbol(Some("batch_size".to_string()));
        assert_eq!(registry.get_symbol("batch_size"), Some(id));

        registry.set_value(id, 32).unwrap();
        assert_eq!(registry.get_value(id), Some(32));
    }

    #[test]
    fn test_constrained_dimension() {
        let dim = SymbolicDim::constrained(0, Some("N".to_string()), Some(1), Some(100));

        assert!(dim.satisfies_constraints(50));
        assert!(dim.satisfies_constraints(1));
        assert!(dim.satisfies_constraints(100));
        assert!(!dim.satisfies_constraints(0));
        assert!(!dim.satisfies_constraints(101));
    }

    #[test]
    fn test_dimension_unification() {
        let concrete = SymbolicDim::Concrete(10);
        let unknown = SymbolicDim::unknown(0, None);

        let unified = concrete.unify(&unknown).unwrap();
        assert_eq!(unified.as_concrete(), Some(10));

        let constrained = SymbolicDim::constrained(1, None, Some(5), Some(15));
        let unified = concrete.unify(&constrained).unwrap();
        assert_eq!(unified.as_concrete(), Some(10));
    }
}
