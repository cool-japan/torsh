//! Core shape types and fundamental operations

use crate::error::{Result, TorshError};

// Constants for commonly used error messages to reduce heap allocations
const ZERO_DIMENSION_ERROR: &str = "Shape cannot contain zero dimensions";

/// Core Shape type representing tensor dimensions
///
/// A Shape represents the dimensions of a tensor and provides fundamental
/// operations for querying and manipulating tensor shapes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape from dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.dims(), &[2, 3, 4]);
    /// ```
    pub fn new(dims: Vec<usize>) -> Self {
        Shape { dims }
    }

    /// Create a shape from dimensions with validation (no zero dimensions)
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// // Valid shape
    /// let shape = Shape::from_dims(vec![2, 3, 4]).unwrap();
    /// assert_eq!(shape.dims(), &[2, 3, 4]);
    ///
    /// // Invalid shape with zero dimension
    /// let result = Shape::from_dims(vec![2, 0, 4]);
    /// assert!(result.is_err());
    /// ```
    pub fn from_dims<T: Into<Vec<usize>>>(dims: T) -> Result<Self> {
        let dims = dims.into();
        if dims.contains(&0) {
            return Err(TorshError::InvalidShape(ZERO_DIMENSION_ERROR.to_string()));
        }
        Ok(Shape::new(dims))
    }

    /// Create a shape from a slice with validation
    pub fn from_slice(dims: &[usize]) -> Result<Self> {
        if dims.contains(&0) {
            return Err(TorshError::InvalidShape(ZERO_DIMENSION_ERROR.to_string()));
        }
        Ok(Shape::new(dims.to_vec()))
    }

    /// Create a shape from an array with validation
    pub fn from_array<const N: usize>(dims: [usize; N]) -> Result<Self> {
        if dims.contains(&0) {
            return Err(TorshError::InvalidShape(ZERO_DIMENSION_ERROR.to_string()));
        }
        Ok(Shape::new(dims.to_vec()))
    }

    /// Create a 1D shape with validation
    pub fn from_1d(d1: usize) -> Result<Self> {
        if d1 == 0 {
            return Err(TorshError::InvalidShape(ZERO_DIMENSION_ERROR.to_string()));
        }
        Ok(Shape::new(vec![d1]))
    }

    /// Create a 2D shape with validation
    pub fn from_2d(d1: usize, d2: usize) -> Result<Self> {
        if d1 == 0 || d2 == 0 {
            return Err(TorshError::InvalidShape(ZERO_DIMENSION_ERROR.to_string()));
        }
        Ok(Shape::new(vec![d1, d2]))
    }

    /// Create a 3D shape with validation
    pub fn from_3d(d1: usize, d2: usize, d3: usize) -> Result<Self> {
        if d1 == 0 || d2 == 0 || d3 == 0 {
            return Err(TorshError::InvalidShape(ZERO_DIMENSION_ERROR.to_string()));
        }
        Ok(Shape::new(vec![d1, d2, d3]))
    }

    /// Create a 4D shape with validation
    pub fn from_4d(d1: usize, d2: usize, d3: usize, d4: usize) -> Result<Self> {
        if d1 == 0 || d2 == 0 || d3 == 0 || d4 == 0 {
            return Err(TorshError::InvalidShape(ZERO_DIMENSION_ERROR.to_string()));
        }
        Ok(Shape::new(vec![d1, d2, d3, d4]))
    }

    /// Create a scalar shape (0-dimensional)
    pub const fn scalar() -> Self {
        Shape { dims: Vec::new() }
    }

    /// Get the number of dimensions (rank)
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// let scalar = Shape::new(vec![]);
    /// assert_eq!(scalar.ndim(), 0);
    ///
    /// let vector = Shape::new(vec![10]);
    /// assert_eq!(vector.ndim(), 1);
    ///
    /// let matrix = Shape::new(vec![3, 4]);
    /// assert_eq!(matrix.ndim(), 2);
    /// ```
    pub const fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Get contiguous strides for this shape
    pub fn strides(&self) -> Vec<usize> {
        if self.dims.is_empty() {
            return vec![];
        }

        let mut strides = vec![1; self.dims.len()];
        for i in (0..self.dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Get default (contiguous) strides for this shape
    /// This is an alias for `strides()` for backward compatibility
    pub fn default_strides(&self) -> Vec<usize> {
        self.strides()
    }

    /// Check if the shape represents a contiguous tensor
    /// (always true for shapes, as they define the default contiguous layout)
    pub fn is_contiguous(&self) -> bool {
        true
    }

    /// Get the dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.dims(), &[2, 3, 4]);
    ///
    /// let scalar = Shape::new(vec![]);
    /// assert_eq!(scalar.dims(), &[] as &[usize]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get the total number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.numel(), 24);
    ///
    /// let scalar = Shape::new(vec![]);
    /// assert_eq!(scalar.numel(), 1);
    ///
    /// let vector = Shape::new(vec![5]);
    /// assert_eq!(vector.numel(), 5);
    /// ```
    pub fn numel(&self) -> usize {
        self.dims
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .unwrap_or(usize::MAX)
    }

    /// Check if the shape is empty (contains zero)
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert!(!shape.is_empty());
    ///
    /// let empty_shape = Shape::new(vec![2, 0, 4]);
    /// assert!(empty_shape.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.dims.contains(&0)
    }

    /// Check if the shape is scalar
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// let scalar = Shape::new(vec![]);
    /// assert!(scalar.is_scalar());
    ///
    /// let vector = Shape::new(vec![5]);
    /// assert!(!vector.is_scalar());
    ///
    /// let matrix = Shape::new(vec![2, 3]);
    /// assert!(!matrix.is_scalar());
    /// ```
    pub const fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Get the size of a specific dimension
    ///
    /// Supports negative indexing (e.g., -1 for last dimension)
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.size(0).unwrap(), 2);
    /// assert_eq!(shape.size(-1).unwrap(), 4);
    /// ```
    pub fn size(&self, dim: i32) -> Result<usize> {
        let ndim = self.dims.len() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim };

        if dim < 0 || dim >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid dimension {} for shape with {} dimensions",
                dim, ndim
            )));
        }

        Ok(self.dims[dim as usize])
    }

    /// Check if two shapes are compatible for broadcasting
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// let shape1 = Shape::new(vec![3, 1, 4]);
    /// let shape2 = Shape::new(vec![2, 4]);
    /// assert!(shape1.is_broadcastable_with(&shape2));
    ///
    /// let shape3 = Shape::new(vec![3, 2]);
    /// let shape4 = Shape::new(vec![3, 4]);
    /// assert!(!shape3.is_broadcastable_with(&shape4));
    /// ```
    pub fn is_broadcastable_with(&self, other: &Shape) -> bool {
        let mut dims1 = self.dims.iter().rev();
        let mut dims2 = other.dims.iter().rev();

        loop {
            match (dims1.next(), dims2.next()) {
                (Some(&d1), Some(&d2)) => {
                    if d1 != d2 && d1 != 1 && d2 != 1 {
                        return false;
                    }
                }
                (None, None) => break,
                _ => {} // One is shorter than the other, which is fine
            }
        }

        true
    }

    /// Compute the broadcast shape with another shape
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// let shape1 = Shape::new(vec![3, 1, 4]);
    /// let shape2 = Shape::new(vec![2, 4]);
    /// let broadcast_shape = shape1.broadcast_with(&shape2).unwrap();
    /// assert_eq!(broadcast_shape.dims(), &[3, 2, 4]);
    /// ```
    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape> {
        if !self.is_broadcastable_with(other) {
            return Err(TorshError::BroadcastError {
                shape1: self.dims.clone(),
                shape2: other.dims.clone(),
            });
        }

        let max_ndim = self.ndim().max(other.ndim());
        let mut result_dims = vec![1; max_ndim];

        let self_dims_padded = self.dims_padded_to(max_ndim);
        let other_dims_padded = other.dims_padded_to(max_ndim);

        for i in 0..max_ndim {
            let d1 = self_dims_padded[i];
            let d2 = other_dims_padded[i];
            result_dims[i] = d1.max(d2);
        }

        Ok(Shape::new(result_dims))
    }

    /// Get dimensions padded to a specific length with 1s at the beginning
    fn dims_padded_to(&self, target_len: usize) -> Vec<usize> {
        if self.ndim() >= target_len {
            self.dims.clone()
        } else {
            let mut padded = vec![1; target_len - self.ndim()];
            padded.extend_from_slice(&self.dims);
            padded
        }
    }

    /// Check if the shape represents a matrix (2D)
    pub fn is_matrix(&self) -> bool {
        self.ndim() == 2
    }

    /// Check if the shape represents a vector (1D)
    pub fn is_vector(&self) -> bool {
        self.ndim() == 1
    }

    /// Get the shape as a reference to the inner vector
    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }

    /// Convert the shape to a vector
    pub fn to_vec(&self) -> Vec<usize> {
        self.dims.clone()
    }

    /// Create a new shape with an additional dimension of size 1 at the specified position
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// let shape = Shape::new(vec![3, 4]);
    /// let unsqueezed = shape.unsqueeze(1).unwrap();
    /// assert_eq!(unsqueezed.dims(), &[3, 1, 4]);
    /// ```
    pub fn unsqueeze(&self, dim: i32) -> Result<Shape> {
        let ndim = self.dims.len() as i32;
        let new_ndim = ndim + 1;
        let dim = if dim < 0 { new_ndim + dim } else { dim };

        if dim < 0 || dim > ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid dimension {} for shape with {} dimensions",
                dim, ndim
            )));
        }

        let mut new_dims = self.dims.clone();
        new_dims.insert(dim as usize, 1);
        Ok(Shape::new(new_dims))
    }

    /// Create a new shape with dimensions of size 1 removed
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// let shape = Shape::new(vec![1, 3, 1, 4]);
    /// let squeezed = shape.squeeze();
    /// assert_eq!(squeezed.dims(), &[3, 4]);
    /// ```
    pub fn squeeze(&self) -> Shape {
        let new_dims: Vec<usize> = self.dims.iter().filter(|&&d| d != 1).copied().collect();
        Shape::new(new_dims)
    }

    /// Create a new shape with a specific dimension of size 1 removed
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::shape::Shape;
    ///
    /// let shape = Shape::new(vec![1, 3, 1, 4]);
    /// let squeezed = shape.squeeze_dim(0).unwrap();
    /// assert_eq!(squeezed.dims(), &[3, 1, 4]);
    /// ```
    pub fn squeeze_dim(&self, dim: i32) -> Result<Shape> {
        let ndim = self.dims.len() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim };

        if dim < 0 || dim >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid dimension {} for shape with {} dimensions",
                dim, ndim
            )));
        }

        let dim_size = self.dims[dim as usize];
        if dim_size != 1 {
            return Err(TorshError::InvalidOperation(format!(
                "Cannot squeeze dimension {} with size {}",
                dim, dim_size
            )));
        }

        let mut new_dims = self.dims.clone();
        new_dims.remove(dim as usize);
        Ok(Shape::new(new_dims))
    }
}

impl Default for Shape {
    fn default() -> Self {
        Shape::scalar()
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

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Shape::new(dims.to_vec())
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_scalar() {
            write!(f, "[]")
        } else {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.numel(), 24);
        assert!(!shape.is_scalar());
        assert!(!shape.is_empty());
    }

    #[test]
    fn test_shape_validation() {
        let valid_shape = Shape::from_dims(vec![2, 3, 4]);
        assert!(valid_shape.is_ok());

        let invalid_shape = Shape::from_dims(vec![2, 0, 4]);
        assert!(invalid_shape.is_err());
    }

    #[test]
    fn test_scalar_shape() {
        let scalar = Shape::scalar();
        assert!(scalar.is_scalar());
        assert_eq!(scalar.ndim(), 0);
        assert_eq!(scalar.numel(), 1);
        assert_eq!(scalar.dims(), &[] as &[usize]);
    }

    #[test]
    fn test_broadcasting() {
        let shape1 = Shape::new(vec![3, 1, 4]);
        let shape2 = Shape::new(vec![2, 4]);
        assert!(shape1.is_broadcastable_with(&shape2));

        let broadcast_shape = shape1.broadcast_with(&shape2).unwrap();
        assert_eq!(broadcast_shape.dims(), &[3, 2, 4]);

        let shape3 = Shape::new(vec![3, 2]);
        let shape4 = Shape::new(vec![3, 4]);
        assert!(!shape3.is_broadcastable_with(&shape4));
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let shape = Shape::new(vec![1, 3, 1, 4]);
        let squeezed = shape.squeeze();
        assert_eq!(squeezed.dims(), &[3, 4]);

        let shape2 = Shape::new(vec![3, 4]);
        let unsqueezed = shape2.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.dims(), &[3, 1, 4]);

        let squeeze_dim = shape.squeeze_dim(0).unwrap();
        assert_eq!(squeeze_dim.dims(), &[3, 1, 4]);
    }

    #[test]
    fn test_shape_properties() {
        let matrix = Shape::new(vec![3, 4]);
        assert!(matrix.is_matrix());
        assert!(!matrix.is_vector());

        let vector = Shape::new(vec![5]);
        assert!(!vector.is_matrix());
        assert!(vector.is_vector());

        assert_eq!(matrix.size(0).unwrap(), 3);
        assert_eq!(matrix.size(-1).unwrap(), 4);
    }

    #[test]
    fn test_shape_conversions() {
        let dims = vec![2, 3, 4];
        let shape: Shape = dims.clone().into();
        assert_eq!(shape.dims(), &dims);

        let arr = [2, 3, 4];
        let shape: Shape = arr.into();
        assert_eq!(shape.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_shape_display() {
        let scalar = Shape::scalar();
        assert_eq!(format!("{}", scalar), "[]");

        let matrix = Shape::new(vec![3, 4]);
        assert_eq!(format!("{}", matrix), "[3, 4]");
    }
}
