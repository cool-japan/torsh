//! Sparse activation functions
//!
//! This module provides activation functions optimized for sparse tensors. Activation functions
//! are crucial for introducing non-linearity in neural networks. These implementations are
//! designed to work efficiently with sparse data structures.
//!
//! # Key Features
//! - Preserve or enhance sparsity where possible (e.g., ReLU increases sparsity)
//! - Efficient processing of only non-zero elements
//! - Support for all sparse tensor formats (COO, CSR, CSC)
//!
//! # Available Activations
//! - ReLU: Rectified Linear Unit - excellent for sparsity
//! - Leaky ReLU: ReLU with small negative slope
//! - GELU: Gaussian Error Linear Unit - smooth alternative to ReLU
//! - Sigmoid: S-shaped curve mapping to (0, 1)
//! - Tanh: Hyperbolic tangent mapping to (-1, 1)

use crate::{CooTensor, CsrTensor, CscTensor, SparseTensor, SparseFormat, TorshResult};
use scirs2_core::random::{Random, rng};
use std::collections::HashMap;
use torsh_core::{Shape, TorshError};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Helper function to unzip triplets for COO tensor creation
fn unzip_triplets(triplets: Vec<(usize, usize, f32)>) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    triplets.into_iter().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut rows, mut cols, mut vals), (r, c, v)| {
            rows.push(r);
            cols.push(c);
            vals.push(v);
            (rows, cols, vals)
        },
    )
}

/// Sparse ReLU activation function
///
/// Applies ReLU (Rectified Linear Unit) to sparse tensors: ReLU(x) = max(0, x).
/// This is particularly efficient for sparse tensors because negative values become zero,
/// which can actually increase sparsity, making it very efficient for sparse tensors.
///
/// # Mathematical Formulation
/// ReLU(x) = { x if x > 0, 0 if x ≤ 0 }
///
/// # Sparsity Impact
/// ReLU is ideal for sparse tensors because:
/// - It zeros out negative values, potentially increasing sparsity
/// - Only positive values need to be stored
/// - Very efficient computation (just filtering)
#[derive(Debug, Clone)]
pub struct SparseReLU {
    /// Whether to apply ReLU in-place (modifies original sparsity pattern)
    inplace: bool,
}

impl SparseReLU {
    /// Create a new sparse ReLU activation
    ///
    /// # Arguments
    /// * `inplace` - Whether to modify the input tensor in-place (memory efficient)
    ///
    /// # Example
    /// ```rust
    /// use torsh_sparse::nn::activations::SparseReLU;
    ///
    /// let relu = SparseReLU::new(false);
    /// ```
    pub fn new(inplace: bool) -> Self {
        Self { inplace }
    }

    /// Create ReLU with default settings (not in-place)
    pub fn default() -> Self {
        Self::new(false)
    }

    /// Forward pass for sparse tensors
    ///
    /// # Arguments
    /// * `input` - Input sparse tensor
    ///
    /// # Returns
    /// * Output tensor with ReLU applied, potentially sparser than input
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
            SparseFormat::Coo => Ok(Box::new(activated_coo)),
            SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&activated_coo)?)),
            SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&activated_coo)?)),
        }
    }

    /// Check if activation is in-place
    pub fn inplace(&self) -> bool {
        self.inplace
    }
}

/// Sparse Leaky ReLU activation function
///
/// Applies Leaky ReLU to sparse tensors: LeakyReLU(x) = max(αx, x) where α is a small slope
/// for negative values. This preserves more information than regular ReLU and may maintain
/// better gradient flow during backpropagation.
///
/// # Mathematical Formulation
/// LeakyReLU(x) = { x if x > 0, α*x if x ≤ 0 }
///
/// Where α (negative_slope) is typically a small value like 0.01.
#[derive(Debug, Clone)]
pub struct SparseLeakyReLU {
    /// Slope for negative values (typically 0.01)
    negative_slope: f32,
}

impl SparseLeakyReLU {
    /// Create a new sparse Leaky ReLU activation
    ///
    /// # Arguments
    /// * `negative_slope` - Slope for negative values (typically 0.01)
    ///
    /// # Example
    /// ```rust
    /// use torsh_sparse::nn::activations::SparseLeakyReLU;
    ///
    /// let leaky_relu = SparseLeakyReLU::new(0.01);
    /// ```
    pub fn new(negative_slope: f32) -> Self {
        Self { negative_slope }
    }

    /// Create Leaky ReLU with default slope (0.01)
    pub fn default() -> Self {
        Self::new(0.01)
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
            .filter(|(_, _, val)| val.abs() > 1e-10) // Remove near-zero values to maintain sparsity
            .collect();

        // Create new COO tensor with activated values
        let (rows, cols, values) = unzip_triplets(activated_triplets);
        let activated_coo = CooTensor::new(rows, cols, values, shape)?;

        // Convert back to original format
        match input.format() {
            SparseFormat::Coo => Ok(Box::new(activated_coo)),
            SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&activated_coo)?)),
            SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&activated_coo)?)),
        }
    }

    /// Get the negative slope parameter
    pub fn negative_slope(&self) -> f32 {
        self.negative_slope
    }
}

/// Sparse Sigmoid activation function
///
/// Applies sigmoid activation to sparse tensors: sigmoid(x) = 1 / (1 + exp(-x)).
/// Note that sigmoid maps inputs to (0, 1), so this may reduce sparsity as it
/// doesn't produce exact zeros for any finite input.
///
/// # Mathematical Formulation
/// sigmoid(x) = 1 / (1 + e^(-x))
///
/// # Sparsity Impact
/// Sigmoid reduces sparsity because it maps all real numbers to (0, 1),
/// never producing exact zeros.
#[derive(Debug, Clone)]
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
            SparseFormat::Coo => Ok(Box::new(activated_coo)),
            SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&activated_coo)?)),
            SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&activated_coo)?)),
        }
    }
}

/// Sparse Tanh activation function
///
/// Applies hyperbolic tangent activation to sparse tensors: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)).
/// Maps inputs to (-1, 1). Like sigmoid, this may reduce sparsity as it doesn't produce exact zeros.
///
/// # Mathematical Formulation
/// tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
#[derive(Debug, Clone)]
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
            SparseFormat::Coo => Ok(Box::new(activated_coo)),
            SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&activated_coo)?)),
            SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&activated_coo)?)),
        }
    }
}

/// Sparse GELU activation function
///
/// Applies Gaussian Error Linear Unit to sparse tensors. GELU is defined as:
/// GELU(x) = x * Φ(x), where Φ(x) is the cumulative distribution function of
/// the standard Gaussian distribution.
///
/// We use the approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
///
/// # Mathematical Formulation
/// GELU(x) = x * Φ(x) ≈ x * sigmoid(1.702 * x)
///
/// Where Φ(x) is the standard Gaussian CDF.
#[derive(Debug, Clone)]
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
            .filter(|(_, _, val)| val.abs() > 1e-10) // Remove near-zero values
            .collect();

        // Create new COO tensor with activated values
        let (rows, cols, values) = unzip_triplets(activated_triplets);
        let activated_coo = CooTensor::new(rows, cols, values, shape)?;

        // Convert back to original format
        match input.format() {
            SparseFormat::Coo => Ok(Box::new(activated_coo)),
            SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&activated_coo)?)),
            SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&activated_coo)?)),
        }
    }
}

/// Sparse Swish activation function
///
/// Applies Swish activation: Swish(x) = x * sigmoid(x).
/// This is similar to GELU but uses sigmoid(x) instead of the Gaussian CDF approximation.
#[derive(Debug, Clone)]
pub struct SparseSwish;

impl Default for SparseSwish {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseSwish {
    /// Create a new sparse Swish activation
    pub fn new() -> Self {
        Self
    }

    /// Forward pass for sparse tensors
    pub fn forward(&self, input: &dyn SparseTensor) -> TorshResult<Box<dyn SparseTensor>> {
        // Convert to COO for processing
        let coo = input.to_coo()?;
        let triplets = coo.triplets();
        let shape = input.shape().clone();

        // Apply Swish: x * sigmoid(x)
        let activated_triplets: Vec<(usize, usize, f32)> = triplets
            .into_iter()
            .map(|(row, col, val)| {
                let sigmoid_val = 1.0 / (1.0 + (-val).exp());
                let swish_val = val * sigmoid_val;
                (row, col, swish_val)
            })
            .filter(|(_, _, val)| val.abs() > 1e-10) // Remove near-zero values
            .collect();

        // Create new COO tensor with activated values
        let (rows, cols, values) = unzip_triplets(activated_triplets);
        let activated_coo = CooTensor::new(rows, cols, values, shape)?;

        // Convert back to original format
        match input.format() {
            SparseFormat::Coo => Ok(Box::new(activated_coo)),
            SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&activated_coo)?)),
            SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&activated_coo)?)),
        }
    }
}

/// Sparse ELU (Exponential Linear Unit) activation function
///
/// Applies ELU activation: ELU(x) = { x if x > 0, α(e^x - 1) if x ≤ 0 }
#[derive(Debug, Clone)]
pub struct SparseELU {
    /// Alpha parameter for negative values
    alpha: f32,
}

impl SparseELU {
    /// Create a new sparse ELU activation
    ///
    /// # Arguments
    /// * `alpha` - Alpha parameter for negative values (typically 1.0)
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Create ELU with default alpha (1.0)
    pub fn default() -> Self {
        Self::new(1.0)
    }

    /// Forward pass for sparse tensors
    pub fn forward(&self, input: &dyn SparseTensor) -> TorshResult<Box<dyn SparseTensor>> {
        // Convert to COO for processing
        let coo = input.to_coo()?;
        let triplets = coo.triplets();
        let shape = input.shape().clone();

        // Apply ELU: { x if x > 0, α(e^x - 1) if x ≤ 0 }
        let activated_triplets: Vec<(usize, usize, f32)> = triplets
            .into_iter()
            .map(|(row, col, val)| {
                let elu_val = if val > 0.0 {
                    val
                } else {
                    self.alpha * (val.exp() - 1.0)
                };
                (row, col, elu_val)
            })
            .filter(|(_, _, val)| val.abs() > 1e-10) // Remove near-zero values
            .collect();

        // Create new COO tensor with activated values
        let (rows, cols, values) = unzip_triplets(activated_triplets);
        let activated_coo = CooTensor::new(rows, cols, values, shape)?;

        // Convert back to original format
        match input.format() {
            SparseFormat::Coo => Ok(Box::new(activated_coo)),
            SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&activated_coo)?)),
            SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&activated_coo)?)),
        }
    }

    /// Get the alpha parameter
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_tensor::SparseFormat;

    fn create_test_coo() -> CooTensor {
        let row_indices = vec![0, 0, 1, 1, 2];
        let col_indices = vec![0, 1, 0, 2, 1];
        let values = vec![2.0, -1.0, 3.0, -2.0, 1.0];
        let shape = Shape::new(vec![3, 3]);
        CooTensor::new(row_indices, col_indices, values, shape).unwrap()
    }

    #[test]
    fn test_sparse_relu() {
        let relu = SparseReLU::new(false);
        let input = create_test_coo();
        let output = relu.forward(&input).unwrap();

        // ReLU should remove negative values, so output should have fewer non-zeros
        let output_coo = output.to_coo().unwrap();
        let triplets = output_coo.triplets();

        // Should only have positive values: [2.0, 3.0, 1.0]
        assert_eq!(triplets.len(), 3);
        for (_, _, val) in triplets {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_sparse_leaky_relu() {
        let leaky_relu = SparseLeakyReLU::new(0.1);
        let input = create_test_coo();
        let output = leaky_relu.forward(&input).unwrap();

        let output_coo = output.to_coo().unwrap();
        let triplets = output_coo.triplets();

        // Should preserve all values but scale negative ones
        assert_eq!(triplets.len(), 5);
        assert_eq!(leaky_relu.negative_slope(), 0.1);
    }

    #[test]
    fn test_sparse_sigmoid() {
        let sigmoid = SparseSigmoid::new();
        let input = create_test_coo();
        let output = sigmoid.forward(&input).unwrap();

        let output_coo = output.to_coo().unwrap();
        let triplets = output_coo.triplets();

        // Sigmoid preserves all values, maps to (0, 1)
        assert_eq!(triplets.len(), 5);
        for (_, _, val) in triplets {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_sparse_tanh() {
        let tanh = SparseTanh::new();
        let input = create_test_coo();
        let output = tanh.forward(&input).unwrap();

        let output_coo = output.to_coo().unwrap();
        let triplets = output_coo.triplets();

        // Tanh preserves all values, maps to (-1, 1)
        assert_eq!(triplets.len(), 5);
        for (_, _, val) in triplets {
            assert!(val > -1.0 && val < 1.0);
        }
    }

    #[test]
    fn test_sparse_gelu() {
        let gelu = SparseGELU::new();
        let input = create_test_coo();
        let output = gelu.forward(&input).unwrap();

        let output_coo = output.to_coo().unwrap();
        let triplets = output_coo.triplets();

        // GELU should process all values
        assert!(triplets.len() > 0);
    }

    #[test]
    fn test_sparse_swish() {
        let swish = SparseSwish::new();
        let input = create_test_coo();
        let output = swish.forward(&input).unwrap();

        let output_coo = output.to_coo().unwrap();
        let triplets = output_coo.triplets();

        // Swish should process all values
        assert!(triplets.len() > 0);
    }

    #[test]
    fn test_sparse_elu() {
        let elu = SparseELU::new(1.0);
        let input = create_test_coo();
        let output = elu.forward(&input).unwrap();

        let output_coo = output.to_coo().unwrap();
        let triplets = output_coo.triplets();

        // ELU should process all values
        assert!(triplets.len() > 0);
        assert_eq!(elu.alpha(), 1.0);
    }

    #[test]
    fn test_activation_defaults() {
        let _relu = SparseReLU::default();
        let _leaky_relu = SparseLeakyReLU::default();
        let _sigmoid = SparseSigmoid::default();
        let _tanh = SparseTanh::default();
        let _gelu = SparseGELU::default();
        let _swish = SparseSwish::default();
        let _elu = SparseELU::default();
    }

    #[test]
    fn test_format_preservation() {
        let relu = SparseReLU::new(false);
        let coo_input = create_test_coo();
        let csr_input = CsrTensor::from_coo(&coo_input).unwrap();

        let coo_output = relu.forward(&coo_input).unwrap();
        let csr_output = relu.forward(&csr_input).unwrap();

        assert_eq!(coo_output.format(), SparseFormat::Coo);
        assert_eq!(csr_output.format(), SparseFormat::Csr);
    }

    #[test]
    fn test_sparsity_increase_with_relu() {
        let relu = SparseReLU::new(false);
        let input = create_test_coo();
        let input_nnz = input.nnz();

        let output = relu.forward(&input).unwrap();
        let output_nnz = output.nnz();

        // ReLU should reduce the number of non-zeros (remove negatives)
        assert!(output_nnz <= input_nnz);
    }
}