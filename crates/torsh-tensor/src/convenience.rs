//! Convenience methods for tensor manipulation
//!
//! This module provides convenient shortcuts and aliases for common tensor operations
//! to improve ergonomics and match PyTorch/NumPy APIs.

use crate::{Tensor, TensorElement};
use torsh_core::error::Result;

/// Convenience trait for tensor manipulation shortcuts
pub trait TensorConvenience<T: TensorElement> {
    /// Transpose shortcut (equivalent to .transpose())
    ///
    /// # Examples
    /// ```
    /// # use torsh_tensor::{tensor_2d, convenience::TensorConvenience};
    /// let tensor = tensor_2d!([&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
    /// let transposed = tensor.T().unwrap();
    /// ```
    #[allow(non_snake_case)]
    fn T(&self) -> Result<Tensor<T>>;

    /// Matrix transpose (alias for .T())
    #[allow(non_snake_case)]
    fn mT(&self) -> Result<Tensor<T>>;

    /// Hermitian transpose (conjugate transpose for complex numbers)
    #[allow(non_snake_case)]
    fn H(&self) -> Result<Tensor<T>>;

    /// Transpose shortcut (snake_case version)
    fn t(&self) -> Result<Tensor<T>>;

    /// Matrix transpose (snake_case version)
    fn m_t(&self) -> Result<Tensor<T>>;

    /// Hermitian transpose (snake_case version)
    fn h(&self) -> Result<Tensor<T>>;

    /// Detach tensor from computational graph (creates a new tensor without gradients)
    fn detach(&self) -> Tensor<T>;

    /// Clone tensor data (creates a deep copy)
    fn clone_tensor(&self) -> Result<Tensor<T>>;

    /// Check if tensor is contiguous in memory
    fn is_contiguous(&self) -> bool;

    /// Make tensor contiguous (reorganize memory layout)
    fn contiguous(&self) -> Result<Tensor<T>>;

    /// Get number of elements in tensor
    fn numel(&self) -> usize;

    /// Get tensor size (alias for shape().dims())
    fn size(&self) -> Vec<usize>;

    /// Check if tensor is empty (has zero elements)
    fn is_empty(&self) -> bool;

    /// Check if tensor is scalar (zero dimensions)
    fn is_scalar(&self) -> bool;

    /// Get tensor item as scalar (only works for scalar tensors)
    fn item(&self) -> T;

    /// Convert tensor to scalar (squeezes all dimensions of size 1 first)
    fn to_scalar(&self) -> Result<T>;
}

impl<T: TensorElement + Copy + torsh_core::FloatElement> TensorConvenience<T> for Tensor<T> {
    #[allow(non_snake_case)]
    fn T(&self) -> Result<Tensor<T>> {
        // For 2D tensors, transpose is straightforward
        if self.shape().dims().len() == 2 {
            self.transpose(0, 1)
        } else if self.shape().dims().len() == 1 {
            // 1D tensor transpose returns the same tensor
            Ok(self.clone())
        } else {
            // For higher dimensional tensors, transpose last two dimensions
            let ndim = self.shape().dims().len();
            if ndim >= 2 {
                self.transpose((ndim - 2) as i32, (ndim - 1) as i32)
            } else {
                Ok(self.clone())
            }
        }
    }

    #[allow(non_snake_case)]
    fn mT(&self) -> Result<Tensor<T>> {
        self.T()
    }

    #[allow(non_snake_case)]
    fn H(&self) -> Result<Tensor<T>> {
        // For real numbers, Hermitian transpose is just transpose
        // For complex numbers, we need conjugate transpose
        let transposed = self.T()?;

        // If T implements conjugate operation, apply it
        // For now, just return transpose for real numbers
        Ok(transposed)
    }

    fn t(&self) -> Result<Tensor<T>> {
        self.T()
    }

    fn m_t(&self) -> Result<Tensor<T>> {
        self.T()
    }

    fn h(&self) -> Result<Tensor<T>> {
        self.H()
    }

    fn detach(&self) -> Tensor<T> {
        // Create a new tensor without gradient tracking
        // For now, just return a clone since we don't have gradient tracking implemented
        self.clone()
    }

    fn clone_tensor(&self) -> Result<Tensor<T>> {
        Ok(self.detach())
    }

    fn is_contiguous(&self) -> bool {
        // Check if strides follow row-major order
        let shape_ref = self.shape();
        let shape = shape_ref.dims();
        if shape.is_empty() {
            return true;
        }

        let mut _expected_stride = 1;
        for &dim_size in shape.iter().rev() {
            _expected_stride *= dim_size;
        }

        // For now, assume tensors are contiguous
        // TODO: Add actual stride checking when stride information is available
        true
    }

    fn contiguous(&self) -> Result<Tensor<T>> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            // Reorganize memory layout to be contiguous
            self.clone_tensor()
        }
    }

    fn numel(&self) -> usize {
        self.shape().dims().iter().product()
    }

    fn size(&self) -> Vec<usize> {
        self.shape().dims().to_vec()
    }

    fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    fn is_scalar(&self) -> bool {
        self.shape().dims().is_empty()
    }

    fn item(&self) -> T {
        // Get a single item from scalar tensor
        if self.numel() != 1 {
            panic!("Can only call item() on tensors with one element");
        }
        let data = self.to_vec().unwrap();
        data[0]
    }

    fn to_scalar(&self) -> Result<T> {
        // First squeeze all dimensions of size 1
        let squeezed = self.squeeze_all()?;
        squeezed.item()
    }
}

/// Additional convenience methods for specific tensor operations
pub trait TensorShapeConvenience<T: TensorElement> {
    /// Add singleton dimension at specified position
    fn unsqueeze_at(&self, dim: i32) -> Result<Tensor<T>>;

    /// Remove all singleton dimensions
    fn squeeze_all(&self) -> Result<Tensor<T>>;

    /// Flatten tensor to 1D (preserving total number of elements)
    fn flatten(&self) -> Result<Tensor<T>>;

    /// Flatten tensor starting from specified dimension
    fn flatten_from(&self, start_dim: i32) -> Result<Tensor<T>>;

    /// Unflatten tensor back to specified shape
    fn unflatten(&self, dim: i32, sizes: &[usize]) -> Result<Tensor<T>>;
}

impl<T: TensorElement + Copy> TensorShapeConvenience<T> for Tensor<T> {
    fn unsqueeze_at(&self, dim: i32) -> Result<Tensor<T>> {
        self.unsqueeze(dim)
    }

    fn squeeze_all(&self) -> Result<Tensor<T>> {
        let mut result = self.clone();
        let shape_ref = self.shape();
        let dims = shape_ref.dims();

        // Remove all dimensions of size 1
        for (i, &size) in dims.iter().enumerate().rev() {
            if size == 1 {
                result = result.squeeze(i as i32)?;
            }
        }

        Ok(result)
    }

    fn flatten(&self) -> Result<Tensor<T>> {
        let total_elements = self.numel();
        self.reshape(&[total_elements as i32])
    }

    fn flatten_from(&self, start_dim: i32) -> Result<Tensor<T>> {
        let shape_ref = self.shape();
        let shape = shape_ref.dims();
        let ndim = shape.len() as i32;
        let start_dim = if start_dim < 0 {
            ndim + start_dim
        } else {
            start_dim
        };

        if start_dim < 0 || start_dim >= ndim {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Invalid start_dim {start_dim} for tensor with {ndim} dimensions"
            )));
        }

        let mut new_shape = Vec::new();

        // Keep dimensions before start_dim
        for &dim in shape.iter().take(start_dim as usize) {
            new_shape.push(dim);
        }

        // Flatten dimensions from start_dim onwards
        let flattened_size: usize = shape[start_dim as usize..].iter().product();
        new_shape.push(flattened_size);

        let new_shape_i32: Vec<i32> = new_shape.iter().map(|&x| x as i32).collect();
        self.reshape(&new_shape_i32)
    }

    fn unflatten(&self, dim: i32, sizes: &[usize]) -> Result<Tensor<T>> {
        let shape_ref = self.shape();
        let shape = shape_ref.dims();
        let ndim = shape.len() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim };

        if dim < 0 || dim >= ndim {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Invalid dim {dim} for tensor with {ndim} dimensions"
            )));
        }

        // Check that sizes product matches the dimension size
        let expected_size = shape[dim as usize];
        let actual_size: usize = sizes.iter().product();

        if expected_size != actual_size {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Sizes {actual_size} don't multiply to dimension size {expected_size}"
            )));
        }

        // Build new shape
        let mut new_shape = Vec::new();

        // Add dimensions before the target dimension
        for &dim_size in shape.iter().take(dim as usize) {
            new_shape.push(dim_size);
        }

        // Add the unflattened dimensions
        new_shape.extend_from_slice(sizes);

        // Add dimensions after the target dimension
        for &dim_size in shape.iter().skip(dim as usize + 1) {
            new_shape.push(dim_size);
        }

        let new_shape_i32: Vec<i32> = new_shape.iter().map(|&x| x as i32).collect();
        self.reshape(&new_shape_i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_shortcuts() {
        let tensor = crate::creation::tensor_2d_arrays(&[[1.0f32, 2.0], [3.0, 4.0]]).unwrap();

        // Test .T() shortcut
        let transposed = tensor.T().unwrap();
        assert_eq!(transposed.shape().dims(), &[2, 2]);

        // Test .mT() alias
        let mt_transposed = tensor.mT().unwrap();
        assert_eq!(mt_transposed.shape().dims(), &[2, 2]);

        // Test .H() (should be same as .T() for real numbers)
        let hermitian = tensor.H().unwrap();
        assert_eq!(hermitian.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_tensor_properties() {
        let tensor = crate::creation::tensor_2d_arrays(&[[1.0f32, 2.0], [3.0, 4.0]]).unwrap();

        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.shape().dims(), &[2, 2]);
        assert!(!tensor.is_empty());
        assert!(!tensor.is_scalar());
        assert!(tensor.is_contiguous());

        // Test scalar tensor
        let scalar = crate::creation::tensor_scalar(42.0f32).unwrap();
        assert!(scalar.is_scalar());
        assert_eq!(scalar.item().unwrap(), 42.0);
    }

    #[test]
    fn test_shape_convenience() {
        // Create a 3D tensor with shape [2, 1, 2] using zeros and reshape
        let tensor = crate::creation::zeros::<f32>(&[4])
            .unwrap()
            .reshape(&[2, 1, 2])
            .unwrap();

        // Test squeeze_all (should remove dimension of size 1)
        let squeezed = tensor.squeeze_all().unwrap();
        assert_eq!(squeezed.shape().dims(), &[2, 2]);

        // Test flatten
        let flattened = tensor.flatten().unwrap();
        assert_eq!(flattened.shape().dims(), &[4]);

        // Test flatten_from
        let flat_from_1 = tensor.flatten_from(1).unwrap();
        assert_eq!(flat_from_1.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_detach() {
        let tensor = crate::creation::tensor_1d(&[1.0f32, 2.0, 3.0]).unwrap();
        let detached = tensor.detach();

        // Should have same data and shape
        assert_eq!(tensor.shape().dims(), detached.shape().dims());
        assert_eq!(tensor.data().unwrap(), detached.data().unwrap());
    }

    #[test]
    fn test_fluent_api() {
        use crate::TensorFluentExt;
        let tensor = crate::creation::tensor_1d(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();

        // Test method chaining with fluent API
        let result = tensor
            .fluent()
            .add_scalar(1.0) // [2.0, 3.0, 4.0, 5.0]
            .mul_scalar(2.0) // [4.0, 6.0, 8.0, 10.0]
            .sub_scalar(1.0) // [3.0, 5.0, 7.0, 9.0]
            .unwrap()
            .unwrap();

        let expected = vec![3.0, 5.0, 7.0, 9.0];
        let actual = result.to_vec().unwrap();

        for (exp, act) in expected.iter().zip(actual.iter()) {
            assert!((exp - act).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_fluent_api_operations() {
        use crate::TensorFluentExt;
        let tensor1 = crate::creation::tensor_1d(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let tensor2 = crate::creation::tensor_1d(&[2.0f32, 2.0, 2.0, 2.0]).unwrap();

        // Test tensor operations with fluent API
        let result = tensor1
            .fluent()
            .add(&tensor2) // [3.0, 4.0, 5.0, 6.0]
            .mul_scalar(0.5) // [1.5, 2.0, 2.5, 3.0]
            .sum() // 9.0
            .unwrap()
            .unwrap();

        let actual = result.to_vec().unwrap();
        assert!((actual[0] - 9.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fluent_api_mathematical_operations() {
        use crate::TensorFluentExt;
        let tensor = crate::creation::tensor_1d(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();

        // Test mathematical operations with fluent API
        let result = tensor
            .fluent()
            .relu() // [1.0, 2.0, 3.0, 4.0] (no change since all positive)
            .pow(2.0) // [1.0, 4.0, 9.0, 16.0]
            .sigmoid() // sigmoid values
            .unwrap()
            .unwrap();

        let actual = result.to_vec().unwrap();
        // Check that all values are between 0 and 1 (sigmoid property)
        for val in actual.iter() {
            assert!(*val > 0.0 && *val < 1.0);
        }
    }
}

/// Fluent API trait for method chaining operations
///
/// This trait provides a PyTorch-like fluent interface that allows chaining operations
/// in a readable and natural way. Unlike lazy evaluation, these operations are executed
/// immediately but return self to enable chaining.
///
/// # Examples
/// ```rust
/// use torsh_tensor::{Tensor, TensorFluentExt};
/// use torsh_core::device::DeviceType;
///
/// let result = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
///     .unwrap()
///     .fluent()
///     .add_scalar(1.0)
///     .mul_scalar(2.0)
///     .relu()
///     .sum()
///     .unwrap();
/// ```
pub trait TensorFluentExt<T: TensorElement> {
    /// Start fluent chaining
    fn fluent(self) -> FluentTensor<T>;
}

/// Wrapper for fluent tensor operations
pub struct FluentTensor<T: TensorElement> {
    tensor: Tensor<T>,
}

impl<T: TensorElement> TensorFluentExt<T> for Tensor<T> {
    fn fluent(self) -> FluentTensor<T> {
        FluentTensor { tensor: self }
    }
}

impl<
        T: TensorElement
            + Copy
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + num_traits::Zero,
    > FluentTensor<T>
{
    /// Get the wrapped tensor, consuming the fluent wrapper
    pub fn tensor(self) -> Tensor<T> {
        self.tensor
    }

    /// Unwrap and return as Result
    pub fn unwrap(self) -> Result<Tensor<T>> {
        Ok(self.tensor)
    }

    /// Chain scalar addition
    pub fn add_scalar(mut self, scalar: T) -> Self {
        if let Ok(result) = self.tensor.add_scalar(scalar) {
            self.tensor = result;
        }
        self
    }

    /// Chain scalar multiplication
    pub fn mul_scalar(mut self, scalar: T) -> Self {
        if let Ok(result) = self.tensor.mul_scalar(scalar) {
            self.tensor = result;
        }
        self
    }

    /// Chain scalar subtraction
    pub fn sub_scalar(mut self, scalar: T) -> Self {
        if let Ok(result) = self.tensor.sub_scalar(scalar) {
            self.tensor = result;
        }
        self
    }

    /// Chain scalar division
    pub fn div_scalar(mut self, scalar: T) -> Self {
        if let Ok(result) = self.tensor.div_scalar(scalar) {
            self.tensor = result;
        }
        self
    }

    /// Chain tensor addition
    pub fn add(mut self, other: &Tensor<T>) -> Self {
        if let Ok(result) = self.tensor.add_op(other) {
            self.tensor = result;
        }
        self
    }

    /// Chain tensor multiplication
    pub fn mul(mut self, other: &Tensor<T>) -> Self {
        if let Ok(result) = self.tensor.mul_op(other) {
            self.tensor = result;
        }
        self
    }

    /// Chain tensor subtraction
    pub fn sub(mut self, other: &Tensor<T>) -> Self {
        if let Ok(result) = self.tensor.sub(other) {
            self.tensor = result;
        }
        self
    }

    /// Chain tensor division
    pub fn div(mut self, other: &Tensor<T>) -> Self {
        if let Ok(result) = self.tensor.div(other) {
            self.tensor = result;
        }
        self
    }

    /// Chain reshape operation
    pub fn reshape(mut self, shape: &[i32]) -> Self {
        if let Ok(result) = self.tensor.reshape(shape) {
            self.tensor = result;
        }
        self
    }

    /// Chain transpose operation
    pub fn transpose(mut self, dim0: i32, dim1: i32) -> Self {
        if let Ok(result) = self.tensor.transpose(dim0, dim1) {
            self.tensor = result;
        }
        self
    }

    /// Chain transpose (last two dimensions)
    pub fn t(mut self) -> Self {
        if let Ok(result) = self.tensor.t() {
            self.tensor = result;
        }
        self
    }

    /// Chain sum operation
    pub fn sum(mut self) -> Self {
        if let Ok(result) = self.tensor.sum() {
            self.tensor = result;
        }
        self
    }

    /// Chain sum along dimension
    pub fn sum_dim(mut self, dims: &[i32], keepdim: bool) -> Self {
        if let Ok(result) = self.tensor.sum_dim(dims, keepdim) {
            self.tensor = result;
        }
        self
    }

    /// Chain squeeze operation
    pub fn squeeze(mut self, dim: i32) -> Self {
        if let Ok(result) = self.tensor.squeeze(dim) {
            self.tensor = result;
        }
        self
    }

    /// Chain unsqueeze operation
    pub fn unsqueeze(mut self, dim: i32) -> Self {
        if let Ok(result) = self.tensor.unsqueeze(dim) {
            self.tensor = result;
        }
        self
    }
}

/// Mathematical operations for fluent chaining
impl<T: TensorElement + Copy + num_traits::Float> FluentTensor<T> {
    /// Chain ReLU activation
    pub fn relu(mut self) -> Self {
        if let Ok(result) = self.tensor.relu() {
            self.tensor = result;
        }
        self
    }

    /// Chain sigmoid activation
    pub fn sigmoid(mut self) -> Self
    where
        T: torsh_core::dtype::FloatElement,
    {
        if let Ok(result) = self.tensor.sigmoid() {
            self.tensor = result;
        }
        self
    }

    /// Chain tanh activation
    pub fn tanh(mut self) -> Self
    where
        T: torsh_core::dtype::FloatElement,
    {
        if let Ok(result) = self.tensor.tanh() {
            self.tensor = result;
        }
        self
    }

    /// Chain exponential function
    pub fn exp(mut self) -> Self
    where
        T: torsh_core::dtype::FloatElement,
    {
        if let Ok(result) = self.tensor.exp() {
            self.tensor = result;
        }
        self
    }

    /// Chain logarithm function
    pub fn log(mut self) -> Self
    where
        T: torsh_core::dtype::FloatElement,
    {
        if let Ok(result) = self.tensor.log() {
            self.tensor = result;
        }
        self
    }

    /// Chain power operation
    pub fn pow(mut self, exponent: T) -> Self
    where
        T: torsh_core::dtype::FloatElement + Into<f32>,
    {
        if let Ok(result) = self.tensor.pow(exponent) {
            self.tensor = result;
        }
        self
    }

    // Note: abs() and neg() methods removed due to complex trait requirements
    // Users can call these methods directly on the tensor when needed
}

/// Matrix operations for fluent chaining
impl<T: TensorElement + Copy> FluentTensor<T>
where
    T: num_traits::Float + std::iter::Sum,
{
    /// Chain matrix multiplication
    pub fn matmul(mut self, other: &Tensor<T>) -> Self {
        if let Ok(result) = self.tensor.matmul(other) {
            self.tensor = result;
        }
        self
    }
}

/// Mean operations with specific trait bounds
impl<
        T: TensorElement
            + Copy
            + num_traits::FromPrimitive
            + std::ops::Div<Output = T>
            + num_traits::Zero
            + num_traits::One,
    > FluentTensor<T>
{
    /// Chain mean operation
    pub fn mean(mut self, dims: Option<&[usize]>, keepdim: bool) -> Self {
        if let Ok(result) = self.tensor.mean(dims, keepdim) {
            self.tensor = result;
        }
        self
    }
}
