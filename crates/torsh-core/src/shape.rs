//! Shape and stride utilities for tensors

use crate::error::{Result, TorshError};
use std::fmt;

/// Shape of a tensor
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape from dimensions
    pub fn new(dims: Vec<usize>) -> Self {
        Shape { dims }
    }

    /// Create a scalar shape (0-dimensional)
    pub fn scalar() -> Self {
        Shape { dims: vec![] }
    }

    /// Get the number of dimensions (rank)
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Get the dimensions
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Check if the shape is empty (contains zero)
    pub fn is_empty(&self) -> bool {
        self.dims.iter().any(|&d| d == 0)
    }

    /// Check if the shape is scalar
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Get the size of a specific dimension
    pub fn size(&self, dim: i32) -> Result<usize> {
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim };

        if dim < 0 || dim >= ndim {
            return Err(TorshError::IndexOutOfBounds {
                index: dim as usize,
                size: ndim as usize,
            });
        }

        Ok(self.dims[dim as usize])
    }

    /// Create default strides for this shape (row-major/C-contiguous)
    pub fn default_strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.ndim()];
        for i in (0..self.ndim() - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Check if two shapes are compatible for broadcasting
    pub fn broadcast_compatible(&self, other: &Shape) -> bool {
        let ndim = self.ndim().max(other.ndim());

        for i in 0..ndim {
            let dim1 = if i < self.ndim() {
                self.dims[self.ndim() - 1 - i]
            } else {
                1
            };

            let dim2 = if i < other.ndim() {
                other.dims[other.ndim() - 1 - i]
            } else {
                1
            };

            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false;
            }
        }

        true
    }

    /// Compute the broadcasted shape
    pub fn broadcast_shape(&self, other: &Shape) -> Result<Shape> {
        if !self.broadcast_compatible(other) {
            return Err(TorshError::ShapeMismatch {
                expected: self.dims.clone(),
                got: other.dims.clone(),
            });
        }

        let ndim = self.ndim().max(other.ndim());
        let mut result_dims = vec![1; ndim];

        for i in 0..ndim {
            let dim1 = if i < self.ndim() {
                self.dims[self.ndim() - 1 - i]
            } else {
                1
            };

            let dim2 = if i < other.ndim() {
                other.dims[other.ndim() - 1 - i]
            } else {
                1
            };

            result_dims[ndim - 1 - i] = dim1.max(dim2);
        }

        Ok(Shape::new(result_dims))
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims.to_vec())
    }
}

/// Stride information for tensor storage
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Stride {
    strides: Vec<usize>,
}

impl Stride {
    /// Create new strides
    pub fn new(strides: Vec<usize>) -> Self {
        Stride { strides }
    }

    /// Get the strides
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Check if the strides represent a contiguous tensor
    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        if self.strides.len() != shape.ndim() {
            return false;
        }

        let expected = shape.default_strides();
        self.strides == expected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_broadcasting() {
        let shape1 = Shape::new(vec![1, 3, 1]);
        let shape2 = Shape::new(vec![2, 1, 4]);

        assert!(shape1.broadcast_compatible(&shape2));

        let result = shape1.broadcast_shape(&shape2).unwrap();
        assert_eq!(result.dims(), &[2, 3, 4]);
    }
}
