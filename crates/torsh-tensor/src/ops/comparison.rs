//! Comparison and logical operations for tensors
//!
//! This module provides comprehensive comparison and logical operations including:
//! - Element-wise comparisons (eq, ne, gt, lt, ge, le)
//! - Scalar comparisons
//! - Logical operations (and, or, xor, not)
//! - Maximum and minimum operations
//! - Broadcasting support for all operations

use crate::{Tensor, TensorElement};
use crate::broadcast::BroadcastShape;
use torsh_core::error::{Result, TorshError};

/// Comparison operations
impl<T: TensorElement + PartialOrd> Tensor<T> {
    /// Element-wise equality
    pub fn eq(&self, other: &Self) -> Result<Tensor<bool>> {
        self.comparison_op(other, |a, b| a == b)
    }

    /// Element-wise inequality
    pub fn ne(&self, other: &Self) -> Result<Tensor<bool>> {
        self.comparison_op(other, |a, b| a != b)
    }

    /// Element-wise greater than
    pub fn gt(&self, other: &Self) -> Result<Tensor<bool>> {
        self.comparison_op(other, |a, b| a > b)
    }

    /// Element-wise less than
    pub fn lt(&self, other: &Self) -> Result<Tensor<bool>> {
        self.comparison_op(other, |a, b| a < b)
    }

    /// Element-wise greater than or equal
    pub fn ge(&self, other: &Self) -> Result<Tensor<bool>> {
        self.comparison_op(other, |a, b| a >= b)
    }

    /// Element-wise less than or equal
    pub fn le(&self, other: &Self) -> Result<Tensor<bool>> {
        self.comparison_op(other, |a, b| a <= b)
    }

    /// Maximum with another tensor
    pub fn maximum(&self, other: &Self) -> Result<Self>
    where
        T: Copy + Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
    {
        self.broadcast_binary_op(other, |a, b| if a > b { a } else { b })
    }

    /// Minimum with another tensor
    pub fn minimum(&self, other: &Self) -> Result<Self>
    where
        T: Copy + Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
    {
        self.broadcast_binary_op(other, |a, b| if a < b { a } else { b })
    }

    /// Generic comparison operation with broadcasting
    fn comparison_op<F>(&self, other: &Self, op: F) -> Result<Tensor<bool>>
    where
        F: Fn(&T, &T) -> bool,
    {
        // Check if tensors are broadcast compatible
        if !self.can_broadcast_with(other) {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: other.shape().dims().to_vec(),
            });
        }

        // If shapes are identical, use optimized path
        if self.shape() == other.shape() {
            let self_data = self.data()?;
            let other_data = other.data()?;

            let result_data: Vec<bool> = self_data
                .iter()
                .zip(other_data.iter())
                .map(|(a, b)| op(a, b))
                .collect();

            return Tensor::from_data(
                result_data,
                self.shape().dims().to_vec(),
                self.device,
            );
        }

        // Compute broadcasted shape
        let broadcast_shape = self.shape().broadcast_shape(&other.shape())?;
        let broadcast_dims = broadcast_shape.dims();
        let broadcast_size = broadcast_shape.numel();

        let self_data = self.data()?;
        let other_data = other.data()?;

        let mut result_data = Vec::with_capacity(broadcast_size);

        // Compute broadcasting for each element
        for flat_idx in 0..broadcast_size {
            let broadcast_indices = self.flat_to_multi_index_ops(flat_idx, broadcast_dims);

            let self_idx = self.broadcast_index(&broadcast_indices, broadcast_dims);
            let other_idx = other.broadcast_index(&broadcast_indices, broadcast_dims);

            let self_val = &self_data[self_idx];
            let other_val = &other_data[other_idx];

            result_data.push(op(self_val, other_val));
        }

        Tensor::from_data(
            result_data,
            broadcast_dims.to_vec(),
            self.device,
        )
    }

    /// Element-wise equality with scalar
    pub fn eq_scalar(&self, scalar: T) -> Result<Tensor<bool>> {
        self.comparison_scalar_op(scalar, |a, b| a == &b)
    }

    /// Element-wise inequality with scalar
    pub fn ne_scalar(&self, scalar: T) -> Result<Tensor<bool>> {
        self.comparison_scalar_op(scalar, |a, b| a != &b)
    }

    /// Element-wise greater than scalar
    pub fn gt_scalar(&self, scalar: T) -> Result<Tensor<bool>> {
        self.comparison_scalar_op(scalar, |a, b| a > &b)
    }

    /// Element-wise less than scalar
    pub fn lt_scalar(&self, scalar: T) -> Result<Tensor<bool>> {
        self.comparison_scalar_op(scalar, |a, b| a < &b)
    }

    /// Element-wise greater than or equal to scalar
    pub fn ge_scalar(&self, scalar: T) -> Result<Tensor<bool>> {
        self.comparison_scalar_op(scalar, |a, b| a >= &b)
    }

    /// Element-wise less than or equal to scalar
    pub fn le_scalar(&self, scalar: T) -> Result<Tensor<bool>> {
        self.comparison_scalar_op(scalar, |a, b| a <= &b)
    }

    /// Generic comparison operation with scalar
    fn comparison_scalar_op<F>(&self, scalar: T, op: F) -> Result<Tensor<bool>>
    where
        F: Fn(&T, T) -> bool,
    {
        let self_data = self.data()?;

        let result_data: Vec<bool> = self_data.iter().map(|a| op(a, scalar)).collect();

        Tensor::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Helper method: Convert flat index to multi-dimensional indices
    pub(crate) fn flat_to_multi_index_ops(&self, flat_idx: usize, dims: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; dims.len()];
        let mut remaining = flat_idx;

        // Calculate indices in row-major order
        for i in 0..dims.len() {
            let stride = dims[i + 1..].iter().product::<usize>().max(1);
            indices[i] = remaining / stride;
            remaining %= stride;
        }

        indices
    }

    /// Helper method: Get the actual index in the tensor data for broadcasting
    pub(crate) fn broadcast_index(
        &self,
        broadcast_indices: &[usize],
        broadcast_dims: &[usize],
    ) -> usize {
        let self_shape = self.shape();
        let self_dims = self_shape.dims();
        let self_ndim = self_dims.len();
        let broadcast_ndim = broadcast_dims.len();

        let mut actual_indices = vec![0; self_ndim];

        // Map from broadcast indices to actual indices (right-aligned)
        // Broadcasting aligns dimensions from the right
        let offset = broadcast_ndim.saturating_sub(self_ndim);

        for i in 0..self_ndim {
            let broadcast_dim_idx = offset + i;
            if broadcast_dim_idx < broadcast_ndim {
                let broadcast_idx = broadcast_indices[broadcast_dim_idx];
                let self_dim = self_dims[i];

                // If this dimension has size 1, always use index 0 (broadcasting)
                actual_indices[i] = if self_dim == 1 { 0 } else { broadcast_idx };
            }
        }

        // Convert multi-dimensional indices to flat index
        let mut flat_idx = 0;
        for i in 0..self_ndim {
            let stride = self_dims[i + 1..].iter().product::<usize>().max(1);
            flat_idx += actual_indices[i] * stride;
        }

        flat_idx
    }
}

/// Logical operations for boolean tensors
impl Tensor<bool> {
    /// Element-wise logical AND
    pub fn logical_and(&self, other: &Self) -> Result<Self> {
        self.logical_op(other, |a, b| a && b)
    }

    /// Element-wise logical OR
    pub fn logical_or(&self, other: &Self) -> Result<Self> {
        self.logical_op(other, |a, b| a || b)
    }

    /// Element-wise logical XOR
    pub fn logical_xor(&self, other: &Self) -> Result<Self> {
        self.logical_op(other, |a, b| a ^ b)
    }

    /// Element-wise logical NOT
    pub fn logical_not(&self) -> Result<Self> {
        let data = self.data()?;

        let result_data: Vec<bool> = data.iter().map(|&x| !x).collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Element-wise logical AND with scalar
    pub fn logical_and_scalar(&self, scalar: bool) -> Result<Self> {
        if !scalar {
            // AND with false always gives false
            Self::from_data(
                vec![false; self.shape().numel()],
                self.shape().dims().to_vec(),
                self.device,
            )
        } else {
            // AND with true preserves the original values
            Ok(self.clone())
        }
    }

    /// Element-wise logical OR with scalar
    pub fn logical_or_scalar(&self, scalar: bool) -> Result<Self> {
        if scalar {
            // OR with true always gives true
            Self::from_data(
                vec![true; self.shape().numel()],
                self.shape().dims().to_vec(),
                self.device,
            )
        } else {
            // OR with false preserves the original values
            Ok(self.clone())
        }
    }

    /// Element-wise logical XOR with scalar
    pub fn logical_xor_scalar(&self, scalar: bool) -> Result<Self> {
        if scalar {
            // XOR with true flips all values
            self.logical_not()
        } else {
            // XOR with false preserves the original values
            Ok(self.clone())
        }
    }

    /// Generic logical operation with broadcasting
    fn logical_op<F>(&self, other: &Self, op: F) -> Result<Self>
    where
        F: Fn(bool, bool) -> bool,
    {
        // Check if tensors are broadcast compatible
        if !self.can_broadcast_with(other) {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: other.shape().dims().to_vec(),
            });
        }

        // If shapes are identical, use optimized path
        if self.shape() == other.shape() {
            let self_data = self.data()?;
            let other_data = other.data()?;

            let result_data: Vec<bool> = self_data
                .iter()
                .zip(other_data.iter())
                .map(|(&a, &b)| op(a, b))
                .collect();

            return Self::from_data(
                result_data,
                self.shape().dims().to_vec(),
                self.device,
            );
        }

        // Compute broadcasted shape
        let broadcast_shape = self.shape().broadcast_shape(&other.shape())?;
        let broadcast_dims = broadcast_shape.dims();
        let broadcast_size = broadcast_shape.numel();

        let self_data = self.data()?;
        let other_data = other.data()?;

        let mut result_data = Vec::with_capacity(broadcast_size);

        // Compute broadcasting for each element
        for flat_idx in 0..broadcast_size {
            let broadcast_indices = self.flat_to_multi_index_ops(flat_idx, broadcast_dims);

            let self_idx = self.broadcast_index(&broadcast_indices, broadcast_dims);
            let other_idx = other.broadcast_index(&broadcast_indices, broadcast_dims);

            let self_val = self_data[self_idx];
            let other_val = other_data[other_idx];

            result_data.push(op(self_val, other_val));
        }

        Self::from_data(
            result_data,
            broadcast_dims.to_vec(),
            self.device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_comparison_operations() {
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![1.0f32, 3.0, 2.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();

        // Test element-wise equality
        let eq_result = a.eq(&b).unwrap();
        let eq_data = eq_result.data().unwrap();
        assert_eq!(eq_data.as_slice(), &[true, false, false, true]);

        // Test element-wise greater than
        let gt_result = a.gt(&b).unwrap();
        let gt_data = gt_result.data().unwrap();
        assert_eq!(gt_data.as_slice(), &[false, false, true, false]);

        // Test element-wise less than
        let lt_result = a.lt(&b).unwrap();
        let lt_data = lt_result.data().unwrap();
        assert_eq!(lt_data.as_slice(), &[false, true, false, false]);
    }

    #[test]
    fn test_scalar_comparisons() {
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();

        // Test scalar equality
        let eq_scalar_result = a.eq_scalar(2.0).unwrap();
        let eq_scalar_data = eq_scalar_result.data().unwrap();
        assert_eq!(eq_scalar_data.as_slice(), &[false, true, false, false]);

        // Test scalar greater than
        let gt_scalar_result = a.gt_scalar(2.0).unwrap();
        let gt_scalar_data = gt_scalar_result.data().unwrap();
        assert_eq!(gt_scalar_data.as_slice(), &[false, false, true, true]);
    }

    #[test]
    fn test_logical_operations() {
        let a = Tensor::from_data(vec![true, false, true, false], vec![4], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![true, true, false, false], vec![4], DeviceType::Cpu).unwrap();

        // Test logical AND
        let and_result = a.logical_and(&b).unwrap();
        let and_data = and_result.data().unwrap();
        assert_eq!(and_data.as_slice(), &[true, false, false, false]);

        // Test logical OR
        let or_result = a.logical_or(&b).unwrap();
        let or_data = or_result.data().unwrap();
        assert_eq!(or_data.as_slice(), &[true, true, true, false]);

        // Test logical NOT
        let not_result = a.logical_not().unwrap();
        let not_data = not_result.data().unwrap();
        assert_eq!(not_data.as_slice(), &[false, true, false, true]);
    }

    #[test]
    fn test_maximum_minimum() {
        let a = Tensor::from_data(vec![1.0f32, 4.0, 2.0, 3.0], vec![4], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![2.0f32, 3.0, 4.0, 1.0], vec![4], DeviceType::Cpu).unwrap();

        // Test maximum
        let max_result = a.maximum(&b).unwrap();
        let max_data = max_result.data().unwrap();
        assert_eq!(max_data.as_slice(), &[2.0, 4.0, 4.0, 3.0]);

        // Test minimum
        let min_result = a.minimum(&b).unwrap();
        let min_data = min_result.data().unwrap();
        assert_eq!(min_data.as_slice(), &[1.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_masked_fill() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();
        let mask = Tensor::from_data(vec![true, false, true, false], vec![4], DeviceType::Cpu).unwrap();

        let result = tensor.masked_fill(&mask, 0.0).unwrap();
        let data = result.data().unwrap();

        assert_eq!(data.as_slice(), &[0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_masked_fill_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).unwrap();
        let mask = Tensor::from_data(
            vec![true, false, true, false, true, false],
            vec![2, 3],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.masked_fill(&mask, -1.0).unwrap();
        let data = result.data().unwrap();

        assert_eq!(data.as_slice(), &[-1.0, 2.0, -1.0, 4.0, -1.0, 6.0]);
    }

    #[test]
    fn test_masked_fill_inplace() {
        let mut tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();
        let mask = Tensor::from_data(vec![false, true, false, true], vec![4], DeviceType::Cpu).unwrap();

        tensor.masked_fill_(&mask, 99.0).unwrap();
        let data = tensor.data().unwrap();

        assert_eq!(data.as_slice(), &[1.0, 99.0, 3.0, 99.0]);
    }

    #[test]
    fn test_masked_fill_requires_grad() {
        let mut tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();
        tensor.requires_grad = true;
        let mask = Tensor::from_data(vec![true, false, true, false], vec![4], DeviceType::Cpu).unwrap();

        let result = tensor.masked_fill_(&mask, 0.0);

        assert!(result.is_err()); // Should fail for tensors with requires_grad
    }

    #[test]
    fn test_nonzero_1d() {
        let tensor = Tensor::from_data(vec![0.0f32, 1.0, 0.0, 2.0, 0.0, 3.0], vec![6], DeviceType::Cpu).unwrap();

        let result = tensor.nonzero().unwrap();
        let data = result.data().unwrap();
        let shape = result.shape().dims();

        assert_eq!(shape, &[3, 1]); // 3 non-zero elements, 1D tensor
        assert_eq!(data.as_slice(), &[1i64, 3, 5]); // Indices of non-zero values
    }

    #[test]
    fn test_nonzero_2d() {
        let tensor = Tensor::from_data(
            vec![0.0f32, 1.0, 0.0, 2.0, 0.0, 3.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.nonzero().unwrap();
        let data = result.data().unwrap();
        let shape = result.shape().dims();

        assert_eq!(shape, &[3, 2]); // 3 non-zero elements, 2D tensor
        // Each row contains [row_idx, col_idx]
        // Value 1.0 at [0, 1], 2.0 at [1, 0], 3.0 at [1, 2]
        assert_eq!(data.as_slice(), &[0i64, 1, 1, 0, 1, 2]);
    }

    #[test]
    fn test_nonzero_all_zeros() {
        let tensor = Tensor::from_data(vec![0.0f32, 0.0, 0.0], vec![3], DeviceType::Cpu).unwrap();

        let result = tensor.nonzero().unwrap();
        let shape = result.shape().dims();

        assert_eq!(shape, &[0, 1]); // No non-zero elements
    }

    #[test]
    fn test_nonzero_all_nonzero() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();

        let result = tensor.nonzero().unwrap();
        let data = result.data().unwrap();
        let shape = result.shape().dims();

        assert_eq!(shape, &[3, 1]); // All 3 elements are non-zero
        assert_eq!(data.as_slice(), &[0i64, 1, 2]);
    }
}

// ✅ Masked operations for PyTorch compatibility
impl<T: TensorElement + Copy> Tensor<T> {
    /// Fill elements where mask is true with value
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.masked_fill(tensor, mask, value)`
    pub fn masked_fill(&self, mask: &Tensor<bool>, value: T) -> Result<Self> {
        if self.shape() != mask.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().to_vec(),
                got: mask.shape().to_vec(),
            });
        }

        let data = self.data()?;
        let mask_data = mask.data()?;

        let result: Vec<T> = data
            .iter()
            .zip(mask_data.iter())
            .map(|(&val, &mask_val)| if mask_val { value } else { val })
            .collect();

        Self::from_data(result, self.shape().to_vec(), self.device())
    }

    /// In-place fill elements where mask is true with value
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.Tensor.masked_fill_(mask, value)`
    pub fn masked_fill_(&mut self, mask: &Tensor<bool>, value: T) -> Result<&mut Self> {
        if self.requires_grad {
            return Err(TorshError::InvalidArgument(
                "In-place operation on tensor that requires grad is not allowed".to_string(),
            ));
        }

        if self.shape() != mask.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().to_vec(),
                got: mask.shape().to_vec(),
            });
        }

        let mask_data = mask.data()?;

        for (i, &mask_val) in mask_data.iter().enumerate() {
            if mask_val {
                self.storage.set(i, value)?;
            }
        }

        Ok(self)
    }

    /// Return indices of non-zero elements
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.nonzero(tensor, as_tuple=False)`
    pub fn nonzero(&self) -> Result<Tensor<i64>>
    where
        T: PartialEq + num_traits::Zero,
    {
        let data = self.data()?;
        let shape = self.shape().dims();
        let ndim = shape.len();

        // Find all non-zero indices
        let mut nonzero_indices = Vec::new();

        for (flat_idx, &val) in data.iter().enumerate() {
            if val != <T as num_traits::Zero>::zero() {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; ndim];
                let mut remaining = flat_idx;

                for d in (0..ndim).rev() {
                    let mut stride = 1;
                    for dim in d + 1..ndim {
                        stride *= shape[dim];
                    }
                    indices[d] = remaining / stride;
                    remaining %= stride;
                }

                // Store as [num_nonzero, ndim] format
                for &idx in &indices {
                    nonzero_indices.push(idx as i64);
                }
            }
        }

        let num_nonzero = nonzero_indices.len() / ndim;

        if num_nonzero == 0 {
            // Return empty tensor with shape [0, ndim]
            return Tensor::<i64>::from_data(vec![], vec![0, ndim], self.device());
        }

        // Return as [num_nonzero, ndim] tensor
        Tensor::<i64>::from_data(nonzero_indices, vec![num_nonzero, ndim], self.device())
    }
}