//! Dynamic shape support for FX graphs
//!
//! This module provides support for dynamic shapes where tensor dimensions can be unknown
//! at compile time and inferred at runtime.

use crate::TorshResult;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use torsh_core::{dtype::DType, error::TorshError, shape::Shape};

/// Represents a dynamic dimension that can be static or symbolic
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DynamicDim {
    /// Static dimension with known size
    Static(usize),
    /// Symbolic dimension with a name (e.g., "batch_size", "seq_len")
    Symbolic(String),
    /// Unknown dimension size
    Unknown,
}

impl fmt::Display for DynamicDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynamicDim::Static(size) => write!(f, "{}", size),
            DynamicDim::Symbolic(name) => write!(f, "{}", name),
            DynamicDim::Unknown => write!(f, "?"),
        }
    }
}

impl DynamicDim {
    /// Create a static dimension
    pub fn static_dim(size: usize) -> Self {
        DynamicDim::Static(size)
    }

    /// Create a symbolic dimension
    pub fn symbolic(name: impl Into<String>) -> Self {
        DynamicDim::Symbolic(name.into())
    }

    /// Create an unknown dimension
    pub fn unknown() -> Self {
        DynamicDim::Unknown
    }

    /// Check if this dimension is static
    pub fn is_static(&self) -> bool {
        matches!(self, DynamicDim::Static(_))
    }

    /// Check if this dimension is symbolic
    pub fn is_symbolic(&self) -> bool {
        matches!(self, DynamicDim::Symbolic(_))
    }

    /// Check if this dimension is unknown
    pub fn is_unknown(&self) -> bool {
        matches!(self, DynamicDim::Unknown)
    }

    /// Get the static size if available
    pub fn static_size(&self) -> Option<usize> {
        match self {
            DynamicDim::Static(size) => Some(*size),
            _ => None,
        }
    }

    /// Get the symbolic name if available
    pub fn symbolic_name(&self) -> Option<&str> {
        match self {
            DynamicDim::Symbolic(name) => Some(name),
            _ => None,
        }
    }
}

/// Dynamic shape that can contain static, symbolic, or unknown dimensions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DynamicShape {
    dims: Vec<DynamicDim>,
}

impl fmt::Display for DynamicShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl DynamicShape {
    /// Create a new dynamic shape
    pub fn new(dims: Vec<DynamicDim>) -> Self {
        Self { dims }
    }

    /// Create a dynamic shape from static dimensions
    pub fn from_static(dims: Vec<usize>) -> Self {
        Self {
            dims: dims.into_iter().map(DynamicDim::Static).collect(),
        }
    }

    /// Create a dynamic shape from a static Shape
    pub fn from_shape(shape: &Shape) -> Self {
        Self::from_static(shape.dims().to_vec())
    }

    /// Get the dimensions
    pub fn dims(&self) -> &[DynamicDim] {
        &self.dims
    }

    /// Get the rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Check if all dimensions are static
    pub fn is_static(&self) -> bool {
        self.dims.iter().all(|d| d.is_static())
    }

    /// Check if any dimension is symbolic
    pub fn has_symbolic_dims(&self) -> bool {
        self.dims.iter().any(|d| d.is_symbolic())
    }

    /// Check if any dimension is unknown
    pub fn has_unknown_dims(&self) -> bool {
        self.dims.iter().any(|d| d.is_unknown())
    }

    /// Convert to static shape if all dimensions are static
    pub fn to_static_shape(&self) -> Option<Shape> {
        if self.is_static() {
            let static_dims: Vec<usize> =
                self.dims.iter().map(|d| d.static_size().unwrap()).collect();
            Some(Shape::new(static_dims))
        } else {
            None
        }
    }

    /// Get all symbolic dimension names
    pub fn symbolic_dims(&self) -> HashSet<String> {
        self.dims
            .iter()
            .filter_map(|d| d.symbolic_name().map(|s| s.to_string()))
            .collect()
    }

    /// Substitute symbolic dimensions with concrete values
    pub fn substitute(&self, substitutions: &HashMap<String, usize>) -> Self {
        let new_dims = self
            .dims
            .iter()
            .map(|dim| match dim {
                DynamicDim::Symbolic(name) => {
                    if let Some(&size) = substitutions.get(name) {
                        DynamicDim::Static(size)
                    } else {
                        dim.clone()
                    }
                }
                _ => dim.clone(),
            })
            .collect();

        DynamicShape::new(new_dims)
    }

    /// Create a shape with all dimensions unknown
    pub fn unknown_shape(rank: usize) -> Self {
        Self {
            dims: vec![DynamicDim::Unknown; rank],
        }
    }
}

/// Shape constraint for dynamic shapes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeConstraint {
    /// Two dimensions must be equal
    Equal(DynamicDim, DynamicDim),
    /// Dimension must be greater than or equal to a value
    GreaterEqual(DynamicDim, usize),
    /// Dimension must be less than or equal to a value
    LessEqual(DynamicDim, usize),
    /// Dimension must be divisible by a value
    Divisible(DynamicDim, usize),
    /// Two shapes must be broadcastable
    Broadcastable(DynamicShape, DynamicShape),
}

/// Extended shape information with dynamic shape support
#[derive(Debug, Clone)]
pub struct DynamicShapeInfo {
    pub shape: DynamicShape,
    pub dtype: DType,
    pub constraints: Vec<ShapeConstraint>,
}

/// Serializable version of DynamicShapeInfo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableDynamicShapeInfo {
    pub shape: DynamicShape,
    pub dtype: String,
    pub constraints: Vec<ShapeConstraint>,
}

impl From<&DynamicShapeInfo> for SerializableDynamicShapeInfo {
    fn from(info: &DynamicShapeInfo) -> Self {
        Self {
            shape: info.shape.clone(),
            dtype: format!("{:?}", info.dtype),
            constraints: info.constraints.clone(),
        }
    }
}

impl DynamicShapeInfo {
    pub fn new(shape: DynamicShape, dtype: DType) -> Self {
        Self {
            shape,
            dtype,
            constraints: Vec::new(),
        }
    }

    pub fn with_constraints(
        shape: DynamicShape,
        dtype: DType,
        constraints: Vec<ShapeConstraint>,
    ) -> Self {
        Self {
            shape,
            dtype,
            constraints,
        }
    }

    /// Create from static shape info
    pub fn from_static(shape: &Shape, dtype: DType) -> Self {
        Self::new(DynamicShape::from_shape(shape), dtype)
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: ShapeConstraint) {
        self.constraints.push(constraint);
    }

    /// Check if shape is fully resolved (all static)
    pub fn is_resolved(&self) -> bool {
        self.shape.is_static()
    }

    /// Get static shape if available
    pub fn static_shape(&self) -> Option<Shape> {
        self.shape.to_static_shape()
    }
}

/// Shape inference context with dynamic shape support
pub struct DynamicShapeInferenceContext {
    /// Shape information for each node
    shapes: HashMap<petgraph::graph::NodeIndex, DynamicShapeInfo>,
    /// Global constraints
    constraints: Vec<ShapeConstraint>,
    /// Symbolic dimension mappings
    symbol_mappings: HashMap<String, DynamicDim>,
}

impl DynamicShapeInferenceContext {
    pub fn new() -> Self {
        Self {
            shapes: HashMap::new(),
            constraints: Vec::new(),
            symbol_mappings: HashMap::new(),
        }
    }

    /// Set shape information for a node
    pub fn set_shape(&mut self, node: petgraph::graph::NodeIndex, shape_info: DynamicShapeInfo) {
        self.shapes.insert(node, shape_info);
    }

    /// Get shape information for a node
    pub fn get_shape(&self, node: petgraph::graph::NodeIndex) -> Option<&DynamicShapeInfo> {
        self.shapes.get(&node)
    }

    /// Add a global constraint
    pub fn add_constraint(&mut self, constraint: ShapeConstraint) {
        self.constraints.push(constraint);
    }

    /// Register a symbolic dimension mapping
    pub fn register_symbol(&mut self, name: String, dim: DynamicDim) {
        self.symbol_mappings.insert(name, dim);
    }

    /// Resolve symbolic dimensions using known mappings
    pub fn resolve_symbols(&mut self) -> TorshResult<()> {
        for (_, shape_info) in self.shapes.iter_mut() {
            let mut resolved_dims = Vec::new();

            for dim in shape_info.shape.dims() {
                match dim {
                    DynamicDim::Symbolic(name) => {
                        if let Some(resolved_dim) = self.symbol_mappings.get(name) {
                            resolved_dims.push(resolved_dim.clone());
                        } else {
                            resolved_dims.push(dim.clone());
                        }
                    }
                    _ => resolved_dims.push(dim.clone()),
                }
            }

            shape_info.shape = DynamicShape::new(resolved_dims);
        }

        Ok(())
    }

    /// Infer dynamic shapes for broadcasting
    pub fn broadcast_dynamic_shapes(
        &self,
        shape1: &DynamicShape,
        shape2: &DynamicShape,
    ) -> TorshResult<DynamicShape> {
        let dims1 = shape1.dims();
        let dims2 = shape2.dims();

        let max_len = dims1.len().max(dims2.len());
        let mut result_dims = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let dim1 = if i < dims1.len() {
                &dims1[dims1.len() - 1 - i]
            } else {
                &DynamicDim::Static(1)
            };
            let dim2 = if i < dims2.len() {
                &dims2[dims2.len() - 1 - i]
            } else {
                &DynamicDim::Static(1)
            };

            let result_dim = self.broadcast_dimensions(dim1, dim2)?;
            result_dims.push(result_dim);
        }

        result_dims.reverse();
        Ok(DynamicShape::new(result_dims))
    }

    /// Broadcast two dimensions
    fn broadcast_dimensions(
        &self,
        dim1: &DynamicDim,
        dim2: &DynamicDim,
    ) -> TorshResult<DynamicDim> {
        match (dim1, dim2) {
            (DynamicDim::Static(1), dim) | (dim, DynamicDim::Static(1)) => Ok(dim.clone()),
            (DynamicDim::Static(a), DynamicDim::Static(b)) => {
                if a == b {
                    Ok(DynamicDim::Static(*a))
                } else {
                    Err(TorshError::InvalidArgument(format!(
                        "Cannot broadcast dimensions {} and {}",
                        a, b
                    )))
                }
            }
            (DynamicDim::Symbolic(name1), DynamicDim::Symbolic(name2)) => {
                if name1 == name2 {
                    Ok(dim1.clone())
                } else {
                    // Create a constraint that these must be equal or one must be 1
                    Ok(DynamicDim::Symbolic(format!("max({name1}, {name2})")))
                }
            }
            (DynamicDim::Static(size), DynamicDim::Symbolic(_))
            | (DynamicDim::Symbolic(_), DynamicDim::Static(size)) => {
                if *size == 1 {
                    Ok(if dim1.is_symbolic() {
                        dim1.clone()
                    } else {
                        dim2.clone()
                    })
                } else {
                    // The symbolic dimension should equal the static one or be 1
                    Ok(if dim1.is_symbolic() {
                        dim1.clone()
                    } else {
                        dim2.clone()
                    })
                }
            }
            (DynamicDim::Unknown, dim) | (dim, DynamicDim::Unknown) => {
                // Unknown dimensions can broadcast with anything
                Ok(dim.clone())
            }
        }
    }

    /// Infer shape for matrix multiplication with dynamic shapes
    pub fn matmul_dynamic_shape(
        &self,
        shape1: &DynamicShape,
        shape2: &DynamicShape,
    ) -> TorshResult<DynamicShape> {
        let dims1 = shape1.dims();
        let dims2 = shape2.dims();

        if dims1.len() < 2 || dims2.len() < 2 {
            return Err(TorshError::InvalidArgument(
                "Matmul requires at least 2D tensors".to_string(),
            ));
        }

        let m = &dims1[dims1.len() - 2];
        let k1 = &dims1[dims1.len() - 1];
        let k2 = &dims2[dims2.len() - 2];
        let n = &dims2[dims2.len() - 1];

        // Validate that inner dimensions are compatible
        self.validate_matmul_inner_dims(k1, k2)?;

        // Handle batch dimensions with broadcasting
        let batch_dims1 = &dims1[..dims1.len() - 2];
        let batch_dims2 = &dims2[..dims2.len() - 2];

        let broadcasted_batch = if !batch_dims1.is_empty() || !batch_dims2.is_empty() {
            let batch_shape1 = DynamicShape::new(batch_dims1.to_vec());
            let batch_shape2 = DynamicShape::new(batch_dims2.to_vec());
            self.broadcast_dynamic_shapes(&batch_shape1, &batch_shape2)?
        } else {
            DynamicShape::new(vec![])
        };

        // Combine batch dimensions with matrix dimensions
        let mut result_dims = broadcasted_batch.dims().to_vec();
        result_dims.push(m.clone());
        result_dims.push(n.clone());

        Ok(DynamicShape::new(result_dims))
    }

    /// Validate matrix multiplication inner dimensions
    fn validate_matmul_inner_dims(&self, k1: &DynamicDim, k2: &DynamicDim) -> TorshResult<()> {
        match (k1, k2) {
            (DynamicDim::Static(a), DynamicDim::Static(b)) => {
                if a != b {
                    return Err(TorshError::InvalidArgument(format!(
                        "Incompatible dimensions for matmul: {} vs {}",
                        a, b
                    )));
                }
            }
            (DynamicDim::Symbolic(name1), DynamicDim::Symbolic(name2)) => {
                if name1 != name2 {
                    // Add constraint that these dimensions must be equal
                    // In a full implementation, we'd track this constraint
                }
            }
            // For mixed static/symbolic or unknown dimensions, we can't validate at this point
            _ => {}
        }

        Ok(())
    }

    /// Get all unresolved symbolic dimensions
    pub fn get_unresolved_symbols(&self) -> HashSet<String> {
        let mut symbols = HashSet::new();

        for (_, shape_info) in &self.shapes {
            symbols.extend(shape_info.shape.symbolic_dims());
        }

        // Remove already mapped symbols
        for mapped_symbol in self.symbol_mappings.keys() {
            symbols.remove(mapped_symbol);
        }

        symbols
    }

    /// Validate all constraints
    pub fn validate_constraints(&self, substitutions: &HashMap<String, usize>) -> TorshResult<()> {
        for constraint in &self.constraints {
            self.validate_constraint(constraint, substitutions)?;
        }

        for (_, shape_info) in &self.shapes {
            for constraint in &shape_info.constraints {
                self.validate_constraint(constraint, substitutions)?;
            }
        }

        Ok(())
    }

    /// Validate a single constraint
    fn validate_constraint(
        &self,
        constraint: &ShapeConstraint,
        substitutions: &HashMap<String, usize>,
    ) -> TorshResult<()> {
        match constraint {
            ShapeConstraint::Equal(dim1, dim2) => {
                let resolved1 = self.resolve_dim_value(dim1, substitutions);
                let resolved2 = self.resolve_dim_value(dim2, substitutions);

                if let (Some(val1), Some(val2)) = (resolved1, resolved2) {
                    if val1 != val2 {
                        return Err(TorshError::InvalidArgument(format!(
                            "Constraint violation: {} != {}",
                            val1, val2
                        )));
                    }
                }
            }
            ShapeConstraint::GreaterEqual(dim, min_val) => {
                if let Some(val) = self.resolve_dim_value(dim, substitutions) {
                    if val < *min_val {
                        return Err(TorshError::InvalidArgument(format!(
                            "Constraint violation: {} < {}",
                            val, min_val
                        )));
                    }
                }
            }
            ShapeConstraint::LessEqual(dim, max_val) => {
                if let Some(val) = self.resolve_dim_value(dim, substitutions) {
                    if val > *max_val {
                        return Err(TorshError::InvalidArgument(format!(
                            "Constraint violation: {} > {}",
                            val, max_val
                        )));
                    }
                }
            }
            ShapeConstraint::Divisible(dim, divisor) => {
                if let Some(val) = self.resolve_dim_value(dim, substitutions) {
                    if val % divisor != 0 {
                        return Err(TorshError::InvalidArgument(format!(
                            "Constraint violation: {} not divisible by {}",
                            val, divisor
                        )));
                    }
                }
            }
            ShapeConstraint::Broadcastable(_, _) => {
                // This would be validated during shape inference
            }
        }

        Ok(())
    }

    /// Resolve a dimension to a concrete value if possible
    fn resolve_dim_value(
        &self,
        dim: &DynamicDim,
        substitutions: &HashMap<String, usize>,
    ) -> Option<usize> {
        match dim {
            DynamicDim::Static(val) => Some(*val),
            DynamicDim::Symbolic(name) => substitutions.get(name).copied(),
            DynamicDim::Unknown => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_dim_creation() {
        let static_dim = DynamicDim::static_dim(32);
        assert!(static_dim.is_static());
        assert_eq!(static_dim.static_size(), Some(32));

        let symbolic_dim = DynamicDim::symbolic("batch_size");
        assert!(symbolic_dim.is_symbolic());
        assert_eq!(symbolic_dim.symbolic_name(), Some("batch_size"));

        let unknown_dim = DynamicDim::unknown();
        assert!(unknown_dim.is_unknown());
    }

    #[test]
    fn test_dynamic_shape() {
        let shape = DynamicShape::new(vec![
            DynamicDim::symbolic("batch_size"),
            DynamicDim::static_dim(128),
            DynamicDim::unknown(),
        ]);

        assert_eq!(shape.rank(), 3);
        assert!(!shape.is_static());
        assert!(shape.has_symbolic_dims());
        assert!(shape.has_unknown_dims());

        let symbols = shape.symbolic_dims();
        assert!(symbols.contains("batch_size"));
    }

    #[test]
    fn test_shape_substitution() {
        let shape = DynamicShape::new(vec![
            DynamicDim::symbolic("batch_size"),
            DynamicDim::static_dim(128),
            DynamicDim::symbolic("seq_len"),
        ]);

        let mut substitutions = HashMap::new();
        substitutions.insert("batch_size".to_string(), 32);
        substitutions.insert("seq_len".to_string(), 256);

        let resolved = shape.substitute(&substitutions);
        assert!(resolved.is_static());
        assert_eq!(resolved.to_static_shape().unwrap().dims(), &[32, 128, 256]);
    }

    #[test]
    fn test_broadcasting() {
        let ctx = DynamicShapeInferenceContext::new();

        let shape1 = DynamicShape::new(vec![
            DynamicDim::symbolic("batch_size"),
            DynamicDim::static_dim(1),
            DynamicDim::static_dim(128),
        ]);

        let shape2 = DynamicShape::new(vec![
            DynamicDim::static_dim(64),
            DynamicDim::static_dim(128),
        ]);

        let result = ctx.broadcast_dynamic_shapes(&shape1, &shape2).unwrap();
        assert_eq!(result.rank(), 3);

        // The result should have batch_size, 64, 128
        match &result.dims()[0] {
            DynamicDim::Symbolic(name) => assert_eq!(name, "batch_size"),
            _ => panic!("Expected symbolic dimension"),
        }
        assert_eq!(result.dims()[1], DynamicDim::static_dim(64));
        assert_eq!(result.dims()[2], DynamicDim::static_dim(128));
    }

    #[test]
    fn test_matmul_dynamic_shapes() {
        let ctx = DynamicShapeInferenceContext::new();

        let shape1 = DynamicShape::new(vec![
            DynamicDim::symbolic("batch_size"),
            DynamicDim::static_dim(128),
            DynamicDim::static_dim(256),
        ]);

        let shape2 = DynamicShape::new(vec![
            DynamicDim::static_dim(256),
            DynamicDim::static_dim(512),
        ]);

        let result = ctx.matmul_dynamic_shape(&shape1, &shape2).unwrap();
        assert_eq!(result.rank(), 3);

        match &result.dims()[0] {
            DynamicDim::Symbolic(name) => assert_eq!(name, "batch_size"),
            _ => panic!("Expected symbolic dimension"),
        }
        assert_eq!(result.dims()[1], DynamicDim::static_dim(128));
        assert_eq!(result.dims()[2], DynamicDim::static_dim(512));
    }
}
