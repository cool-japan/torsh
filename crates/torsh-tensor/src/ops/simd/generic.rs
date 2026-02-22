//! Generic SIMD utilities and operations

use crate::{Tensor, TensorElement};
use torsh_core::error::Result;

impl<T: TensorElement> Tensor<T> {
    /// Element-wise operation for tensors with identical shapes
    pub fn element_wise_op<F>(&self, other: &Self, op: F) -> Result<Self>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: Send + Sync,
    {
        if self.shape() != other.shape() {
            return Err(torsh_core::error::TorshError::ShapeMismatch {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            });
        }

        let self_data = self.data();
        let other_data = other.data();
        let result_data: Vec<T> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        Self::from_vec_and_shape(result_data, self.shape().to_vec())
    }

    /// In-place element-wise operation for tensors with identical shapes
    pub fn element_wise_op_inplace<F>(&mut self, other: &Self, op: F) -> Result<()>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: Send + Sync,
    {
        if self.shape() != other.shape() {
            return Err(torsh_core::error::TorshError::ShapeMismatch {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            });
        }

        let other_data = other.data();
        let self_data = self.data_mut();

        for (a, &b) in self_data.iter_mut().zip(other_data.iter()) {
            *a = op(*a, b);
        }

        Ok(())
    }

    /// In-place broadcasting binary operation
    pub fn broadcast_binary_op_inplace<F>(&mut self, other: &Self, op: F) -> Result<()>
    where
        F: Fn(T, T) -> T + Send + Sync + Clone,
        T: Send + Sync,
    {
        // Check if broadcasting is possible
        let self_shape = self.shape();
        let other_shape = other.shape();

        if !self.can_broadcast_with(other) {
            return Err(torsh_core::error::TorshError::BroadcastError {
                shape1: self_shape.to_vec(),
                shape2: other_shape.to_vec(),
            });
        }

        // If shapes are identical, use faster element-wise operation
        if self_shape == other_shape {
            return self.element_wise_op_inplace(other, op);
        }

        // Perform broadcasting operation
        let self_strides = self.strides();
        let other_strides = other.strides();
        let other_data = other.data();
        let self_data = self.data_mut();

        for i in 0..self.numel() {
            let self_idx = self.linear_to_multi_index(i);
            let other_idx = self.broadcast_indices(&self_idx, other_shape);
            let other_linear_idx = Self::multi_to_linear_index(&other_idx, other_strides);

            self_data[i] = op(self_data[i], other_data[other_linear_idx]);
        }

        Ok(())
    }

    /// Helper function to check if two tensors can be broadcast together
    pub fn can_broadcast_with(&self, other: &Self) -> bool {
        let self_shape = self.shape();
        let other_shape = other.shape();

        let max_dims = self_shape.len().max(other_shape.len());

        for i in 0..max_dims {
            let self_dim = if i < self_shape.len() {
                self_shape[self_shape.len() - 1 - i]
            } else {
                1
            };

            let other_dim = if i < other_shape.len() {
                other_shape[other_shape.len() - 1 - i]
            } else {
                1
            };

            if self_dim != 1 && other_dim != 1 && self_dim != other_dim {
                return false;
            }
        }

        true
    }

    /// Helper function to broadcast an index for a given shape
    fn broadcast_indices(&self, index: &[usize], target_shape: &[usize]) -> Vec<usize> {
        let mut result = vec![0; target_shape.len()];
        let self_shape = self.shape();

        for i in 0..target_shape.len() {
            let self_dim_idx = if i < self_shape.len() {
                self_shape.len() - 1 - (target_shape.len() - 1 - i)
            } else {
                continue;
            };

            if self_dim_idx < index.len() {
                result[i] = if self_shape[self_dim_idx] == 1 {
                    0
                } else {
                    index[self_dim_idx]
                };
            }
        }

        result
    }

    /// Convert linear index to multi-dimensional index
    fn linear_to_multi_index(&self, linear_idx: usize) -> Vec<usize> {
        let shape = self.shape();
        let mut index = vec![0; shape.len()];
        let mut remaining = linear_idx;

        for i in (0..shape.len()).rev() {
            let stride = shape[i+1..].iter().product::<usize>();
            index[i] = remaining / stride;
            remaining %= stride;
        }

        index
    }

    /// Convert multi-dimensional index to linear index
    fn multi_to_linear_index(index: &[usize], strides: &[usize]) -> usize {
        index.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum()
    }
}