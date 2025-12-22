//! Sparse activation functions
//!
//! This module provides activation functions optimized for sparse tensors,
//! including ReLU, Sigmoid, Tanh, GELU, and Leaky ReLU while preserving or enhancing sparsity patterns.

use crate::{CooTensor, CscTensor, CsrTensor, SparseTensor, TorshResult};

/// Helper function to unzip triplets
fn unzip_triplets(triplets: Vec<(usize, usize, f32)>) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let mut rows = Vec::with_capacity(triplets.len());
    let mut cols = Vec::with_capacity(triplets.len());
    let mut values = Vec::with_capacity(triplets.len());

    for (row, col, value) in triplets {
        rows.push(row);
        cols.push(col);
        values.push(value);
    }

    (rows, cols, values)
}

/// Sparse ReLU activation function
///
/// Applies ReLU (Rectified Linear Unit) to sparse tensors. Since ReLU zeros out negative values,
/// this can actually increase sparsity, making it very efficient for sparse tensors.
pub struct SparseReLU {
    /// Whether to apply ReLU in-place (modifies original sparsity pattern)
    #[allow(dead_code)]
    inplace: bool,
}

impl SparseReLU {
    /// Create a new sparse ReLU activation
    pub fn new(inplace: bool) -> Self {
        Self { inplace }
    }

    /// Forward pass for sparse tensors
    pub fn forward(&self, input: &dyn SparseTensor) -> TorshResult<Box<dyn SparseTensor>> {
        // Convert to COO for processing
        let coo = input.to_coo()?;
        let triplets = coo.triplets();
        let shape = input.shape().clone();

        // Apply ReLU: max(0, x) - only keep positive values
        let activated_triplets: Vec<(usize, usize, f32)> = triplets
            .into_iter()
            .filter_map(|(row, col, val)| {
                if val > 0.0 {
                    Some((row, col, val))
                } else {
                    None // Remove negative values, increasing sparsity
                }
            })
            .collect();

        // Create new COO tensor with activated values
        let (rows, cols, values) = unzip_triplets(activated_triplets);
        let activated_coo = CooTensor::new(rows, cols, values, shape)?;

        // Convert back to original format
        match input.format() {
            crate::SparseFormat::Coo => Ok(Box::new(activated_coo)),
            crate::SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&activated_coo)?)),
            crate::SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&activated_coo)?)),
            _ => Ok(Box::new(activated_coo)), // Default to COO for other formats
        }
    }
}

/// Sparse Sigmoid activation function
///
/// Applies sigmoid activation to sparse tensors. Note that sigmoid maps inputs to (0, 1),
/// so this may reduce sparsity as it doesn't produce exact zeros.
pub struct SparseSigmoid;

impl Default for SparseSigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseSigmoid {
    /// Create a new sparse sigmoid activation
    pub fn new() -> Self {
        Self
    }

    /// Forward pass for sparse tensors
    pub fn forward(&self, input: &dyn SparseTensor) -> TorshResult<Box<dyn SparseTensor>> {
        // Convert to COO for processing
        let coo = input.to_coo()?;
        let triplets = coo.triplets();
        let shape = input.shape().clone();

        // Apply sigmoid: 1 / (1 + exp(-x))
        let activated_triplets: Vec<(usize, usize, f32)> = triplets
            .into_iter()
            .map(|(row, col, val)| {
                let sigmoid_val = 1.0 / (1.0 + (-val).exp());
                (row, col, sigmoid_val)
            })
            .collect();

        // Create new COO tensor with activated values
        let (rows, cols, values) = unzip_triplets(activated_triplets);
        let activated_coo = CooTensor::new(rows, cols, values, shape)?;

        // Convert back to original format
        match input.format() {
            crate::SparseFormat::Coo => Ok(Box::new(activated_coo)),
            crate::SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&activated_coo)?)),
            crate::SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&activated_coo)?)),
            _ => Ok(Box::new(activated_coo)), // Default to COO for other formats
        }
    }
}

/// Sparse Tanh activation function
///
/// Applies hyperbolic tangent activation to sparse tensors. Maps inputs to (-1, 1).
pub struct SparseTanh;

impl Default for SparseTanh {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseTanh {
    /// Create a new sparse tanh activation
    pub fn new() -> Self {
        Self
    }

    /// Forward pass for sparse tensors
    pub fn forward(&self, input: &dyn SparseTensor) -> TorshResult<Box<dyn SparseTensor>> {
        // Convert to COO for processing
        let coo = input.to_coo()?;
        let triplets = coo.triplets();
        let shape = input.shape().clone();

        // Apply tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        let activated_triplets: Vec<(usize, usize, f32)> = triplets
            .into_iter()
            .map(|(row, col, val)| {
                let tanh_val = val.tanh();
                (row, col, tanh_val)
            })
            .collect();

        // Create new COO tensor with activated values
        let (rows, cols, values) = unzip_triplets(activated_triplets);
        let activated_coo = CooTensor::new(rows, cols, values, shape)?;

        // Convert back to original format
        match input.format() {
            crate::SparseFormat::Coo => Ok(Box::new(activated_coo)),
            crate::SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&activated_coo)?)),
            crate::SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&activated_coo)?)),
            _ => Ok(Box::new(activated_coo)), // Default to COO for other formats
        }
    }
}

/// Sparse GELU activation function
///
/// Applies Gaussian Error Linear Unit to sparse tensors. GELU is defined as:
/// GELU(x) = x * Φ(x), where Φ(x) is the cumulative distribution function of the standard Gaussian distribution.
/// We use the approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
pub struct SparseGELU;

impl Default for SparseGELU {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseGELU {
    /// Create a new sparse GELU activation
    pub fn new() -> Self {
        Self
    }

    /// Forward pass for sparse tensors
    pub fn forward(&self, input: &dyn SparseTensor) -> TorshResult<Box<dyn SparseTensor>> {
        // Convert to COO for processing
        let coo = input.to_coo()?;
        let triplets = coo.triplets();
        let shape = input.shape().clone();

        // Apply GELU approximation: x * sigmoid(1.702 * x)
        let activated_triplets: Vec<(usize, usize, f32)> = triplets
            .into_iter()
            .map(|(row, col, val)| {
                let sigmoid_val = 1.0 / (1.0 + (-1.702 * val).exp());
                let gelu_val = val * sigmoid_val;
                (row, col, gelu_val)
            })
            .collect();

        // Create new COO tensor with activated values
        let (rows, cols, values) = unzip_triplets(activated_triplets);
        let activated_coo = CooTensor::new(rows, cols, values, shape)?;

        // Convert back to original format
        match input.format() {
            crate::SparseFormat::Coo => Ok(Box::new(activated_coo)),
            crate::SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&activated_coo)?)),
            crate::SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&activated_coo)?)),
            _ => Ok(Box::new(activated_coo)), // Default to COO for other formats
        }
    }
}

/// Sparse Leaky ReLU activation function
///
/// Applies Leaky ReLU to sparse tensors: LeakyReLU(x) = max(αx, x) where α is a small slope for negative values.
/// This preserves more information than regular ReLU and may maintain better gradient flow.
pub struct SparseLeakyReLU {
    /// Slope for negative values (typically 0.01)
    negative_slope: f32,
}

impl SparseLeakyReLU {
    /// Create a new sparse Leaky ReLU activation
    pub fn new(negative_slope: f32) -> Self {
        Self { negative_slope }
    }

    /// Forward pass for sparse tensors
    pub fn forward(&self, input: &dyn SparseTensor) -> TorshResult<Box<dyn SparseTensor>> {
        // Convert to COO for processing
        let coo = input.to_coo()?;
        let triplets = coo.triplets();
        let shape = input.shape().clone();

        // Apply Leaky ReLU: max(negative_slope * x, x)
        let activated_triplets: Vec<(usize, usize, f32)> = triplets
            .into_iter()
            .map(|(row, col, val)| {
                let leaky_relu_val = if val > 0.0 {
                    val
                } else {
                    self.negative_slope * val
                };
                (row, col, leaky_relu_val)
            })
            .collect();

        // Create new COO tensor with activated values
        let (rows, cols, values) = unzip_triplets(activated_triplets);
        let activated_coo = CooTensor::new(rows, cols, values, shape)?;

        // Convert back to original format
        match input.format() {
            crate::SparseFormat::Coo => Ok(Box::new(activated_coo)),
            crate::SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&activated_coo)?)),
            crate::SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&activated_coo)?)),
            _ => Ok(Box::new(activated_coo)), // Default to COO for other formats
        }
    }
}
