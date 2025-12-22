//! # Gated Activation Functions
//!
//! This module contains gated activation functions that use gating mechanisms to control
//! information flow. These functions split their input into two parts and use one part
//! to gate the other, creating more sophisticated activation patterns commonly used
//! in transformer architectures and advanced neural networks.
//!
//! ## Included Activation Functions
//!
//! - **GLU** - Gated Linear Unit: `GLU(X) = (XW + b) ⊗ σ(XV + c)`
//! - **GEGLU** - Gaussian Error Gated Linear Unit: uses GELU for gating
//! - **ReGLU** - ReLU Gated Linear Unit: uses ReLU for gating
//! - **SwiGLU** - Swish Gated Linear Unit: uses SiLU/Swish for gating
//!
//! ## Gating Mechanism
//!
//! All gated functions follow a similar pattern:
//! 1. Split input tensor into two halves along a specified dimension
//! 2. Apply an activation function to one half (the "gate")
//! 3. Element-wise multiply the gate with the other half
//!
//! ## Usage Example
//!
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # fn main() -> Result<()> {
//! use torsh_nn::layers::activation::gated::{GLU, GEGLU, ReGLU, SwiGLU};
//! use torsh_nn::Module;
//! use torsh_tensor::Tensor;
//!
//! // Create gated activation functions
//! let glu = GLU::new(-1); // Split along last dimension
//! let geglu = GEGLU::new(-1);
//! let reglu = ReGLU::new(-1);
//! let swiglu = SwiGLU::new(-1);
//!
//! // Input must have even size in the split dimension
//! let input = randn(&[2, 8])?; // 8 is even, will be split into 2x4
//! let glu_output = glu.forward(&input)?; // Output shape: [2, 4]
//! # Ok(())
//! # }
//! ```

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation::*, Tensor};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{collections::HashMap, string::String};

#[cfg(not(feature = "std"))]
use alloc::string::String;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// GLU (Gated Linear Unit) activation function
///
/// Applies the gated linear unit function by splitting the input tensor into two halves
/// along a specified dimension and applying element-wise multiplication with a sigmoid gate.
///
/// # Mathematical Definition
/// ```text
/// GLU(X) = Linear(X) ⊗ σ(Gate(X))
/// ```
/// where X is split into two parts, and σ is the sigmoid function.
///
/// # Implementation
/// The function splits the input along the specified dimension and computes:
/// ```text
/// GLU(input) = first_half ⊗ sigmoid(second_half)
/// ```
///
/// # Properties
/// - **Gating mechanism**: Uses sigmoid to gate information flow
/// - **Dimension reduction**: Output size is half the input size in the split dimension
/// - **Learnable gates**: Gates are computed from the input itself
/// - **Transformer usage**: Commonly used in transformer feed-forward layers
///
/// # Requirements
/// - Input tensor must have even size in the split dimension
/// - Split dimension must be valid for the input tensor
///
/// # Example
/// ```rust
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_nn::layers::activation::gated::GLU;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let glu = GLU::new(-1); // Split along last dimension
/// let input = randn(&[2, 8])?; // Last dim (8)? must be even
/// let output = glu.forward(&input)?; // Output shape: [2, 4]
/// # Ok(())
/// # }
/// ```
pub struct GLU {
    base: ModuleBase,
    dim: isize,
}

impl GLU {
    /// Create a new GLU activation
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to split the input. Negative values count from the end.
    ///           Default: -1 (last dimension)
    pub fn new(dim: isize) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }

    /// Create GLU with default dimension (-1, last dimension)
    pub fn default_dim() -> Self {
        Self::new(-1)
    }

    /// Gets the split dimension
    pub fn dim(&self) -> isize {
        self.dim
    }
}

impl Default for GLU {
    fn default() -> Self {
        Self::default_dim()
    }
}

impl Module for GLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape_binding = input.shape();
        let input_shape = shape_binding.dims();
        let split_dim = if self.dim < 0 {
            (input_shape.len() as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        if split_dim >= input_shape.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                self.dim,
                input_shape.len()
            )));
        }

        let split_size = input_shape[split_dim];
        if split_size % 2 != 0 {
            return Err(TorshError::InvalidArgument(format!(
                "Input dimension {} must be even for GLU, got {}",
                split_dim, split_size
            )));
        }

        let half_size = split_size / 2;

        // Split input into two halves
        let first_half = input.narrow(split_dim as i32, 0, half_size)?;
        let second_half = input.narrow(split_dim as i32, half_size as i64, half_size)?;

        // Apply sigmoid to second half (gate)
        let neg_second = second_half.neg()?;
        let exp_neg = neg_second.exp()?;
        let one_plus_exp = exp_neg.add_scalar(1.0)?;
        let ones_tensor = ones(second_half.shape().dims())?;
        let sigmoid_gate = ones_tensor.div(&one_plus_exp)?;

        // Element-wise multiply first_half with gated second_half
        first_half.mul(&sigmoid_gate)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// GEGLU (Gaussian Error Gated Linear Unit) activation function
///
/// Applies GELU-gated linear unit by splitting the input and using GELU as the gate function.
/// This combines the benefits of GLU's gating mechanism with GELU's smooth activation properties.
///
/// # Mathematical Definition
/// ```text
/// GEGLU(X) = Linear(X) ⊗ GELU(Gate(X))
/// ```
/// where GELU is the Gaussian Error Linear Unit function.
///
/// # Implementation
/// ```text
/// GEGLU(input) = first_half ⊗ GELU(second_half)
/// ```
///
/// # Properties
/// - **GELU gating**: Uses GELU's smooth properties for gating
/// - **Transformer-friendly**: Popular in modern transformer architectures
/// - **Better gradients**: GELU provides better gradient flow than sigmoid
/// - **Non-monotonic gating**: GELU's non-monotonic nature can be beneficial
///
/// # Example
/// ```rust
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_nn::layers::activation::gated::GEGLU;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let geglu = GEGLU::new(-1);
/// let input = randn(&[2, 8])?;
/// let output = geglu.forward(&input)?; // Output shape: [2, 4]
/// # Ok(())
/// # }
/// ```
pub struct GEGLU {
    base: ModuleBase,
    dim: isize,
}

impl GEGLU {
    /// Create a new GEGLU activation
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to split the input (default: -1, last dimension)
    pub fn new(dim: isize) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }

    /// Gets the split dimension
    pub fn dim(&self) -> isize {
        self.dim
    }
}

impl Default for GEGLU {
    fn default() -> Self {
        Self::new(-1) // Split along last dimension by default
    }
}

impl Module for GEGLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();

        // Get the actual dimension index
        let actual_dim = if self.dim < 0 {
            (input_shape.len() as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        if actual_dim >= input_shape.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                self.dim,
                input_shape.len()
            )));
        }

        let dim_size = input_shape[actual_dim];
        if dim_size % 2 != 0 {
            return Err(TorshError::InvalidArgument(format!(
                "Input dimension {} must be even for GEGLU, got {}",
                actual_dim, dim_size
            )));
        }

        let half_size = dim_size / 2;

        // Split the input into two halves
        let first_half = input.narrow(actual_dim as i32, 0, half_size)?;
        let second_half = input.narrow(actual_dim as i32, half_size as i64, half_size)?;

        // Apply GELU to the second half (approximate version for efficiency)
        let gelu_gate = self.apply_gelu(&second_half)?;

        // Element-wise multiplication
        first_half.mul(&gelu_gate)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl GEGLU {
    /// Apply GELU activation (approximate version)
    fn apply_gelu(&self, input: &Tensor) -> Result<Tensor> {
        // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let x_cubed = input.pow(3.0)?;
        let coeff_tensor = full(input.shape().dims(), 0.044715)?;
        let term = x_cubed.mul(&coeff_tensor)?;
        let inner = input.add(&term)?;
        let scale_tensor = full(input.shape().dims(), (2.0 / std::f32::consts::PI).sqrt())?;
        let scaled = inner.mul(&scale_tensor)?;
        let tanh_result = scaled.tanh()?;
        let ones_tensor = ones(input.shape().dims())?;
        let one_plus_tanh = tanh_result.add(&ones_tensor)?;
        let half_tensor = full(input.shape().dims(), 0.5)?;
        let half_x = input.mul(&half_tensor)?;
        half_x.mul(&one_plus_tanh)
    }
}

/// ReGLU (ReLU Gated Linear Unit) activation function
///
/// Applies ReLU-gated linear unit by splitting the input and using ReLU as the gate function.
/// This provides a simpler, more computationally efficient gating mechanism.
///
/// # Mathematical Definition
/// ```text
/// ReGLU(X) = Linear(X) ⊗ ReLU(Gate(X))
/// ```
///
/// # Implementation
/// ```text
/// ReGLU(input) = first_half ⊗ ReLU(second_half)
/// ```
///
/// # Properties
/// - **Simple gating**: Uses ReLU's simple, efficient computation
/// - **Sparse gating**: ReLU creates sparse gate patterns
/// - **Fast computation**: No expensive operations like exp or tanh
/// - **Memory efficient**: ReLU gating is very memory efficient
///
/// # Example
/// ```rust
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_nn::layers::activation::gated::ReGLU;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let reglu = ReGLU::new(-1);
/// let input = randn(&[2, 8])?;
/// let output = reglu.forward(&input)?; // Output shape: [2, 4]
/// # Ok(())
/// # }
/// ```
pub struct ReGLU {
    base: ModuleBase,
    dim: isize,
}

impl ReGLU {
    /// Create a new ReGLU activation
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to split the input (default: -1, last dimension)
    pub fn new(dim: isize) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }

    /// Gets the split dimension
    pub fn dim(&self) -> isize {
        self.dim
    }
}

impl Default for ReGLU {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl Module for ReGLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();

        let actual_dim = if self.dim < 0 {
            (input_shape.len() as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        if actual_dim >= input_shape.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                self.dim,
                input_shape.len()
            )));
        }

        let dim_size = input_shape[actual_dim];
        if dim_size % 2 != 0 {
            return Err(TorshError::InvalidArgument(format!(
                "Input dimension {} must be even for ReGLU, got {}",
                actual_dim, dim_size
            )));
        }

        let half_size = dim_size / 2;

        // Split the input
        let first_half = input.narrow(actual_dim as i32, 0, half_size)?;
        let second_half = input.narrow(actual_dim as i32, half_size as i64, half_size)?;

        // Apply ReLU to second half: max(0, x)
        let zero = zeros(second_half.shape().dims())?;
        let relu_gate = second_half.maximum(&zero)?;

        // Element-wise multiplication
        first_half.mul(&relu_gate)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// SwiGLU (Swish Gated Linear Unit) activation function
///
/// Applies SiLU/Swish-gated linear unit by splitting the input and using SiLU as the gate function.
/// SiLU (also known as Swish) provides smooth, self-gated properties that work well in practice.
///
/// # Mathematical Definition
/// ```text
/// SwiGLU(X) = Linear(X) ⊗ SiLU(Gate(X))
/// ```
/// where SiLU(x) = x * sigmoid(x).
///
/// # Implementation
/// ```text
/// SwiGLU(input) = first_half ⊗ SiLU(second_half)
/// ```
///
/// # Properties
/// - **Self-gated**: SiLU uses input to gate itself
/// - **Smooth**: Continuously differentiable gating function
/// - **Popular**: Widely used in modern transformer architectures
/// - **Better than ReLU**: Often outperforms ReLU-based gating
///
/// # Example
/// ```rust
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_nn::layers::activation::gated::SwiGLU;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let swiglu = SwiGLU::new(-1);
/// let input = randn(&[2, 8])?;
/// let output = swiglu.forward(&input)?; // Output shape: [2, 4]
/// # Ok(())
/// # }
/// ```
pub struct SwiGLU {
    base: ModuleBase,
    dim: isize,
}

impl SwiGLU {
    /// Create a new SwiGLU activation
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to split the input (default: -1, last dimension)
    pub fn new(dim: isize) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }

    /// Gets the split dimension
    pub fn dim(&self) -> isize {
        self.dim
    }
}

impl Default for SwiGLU {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl Module for SwiGLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();

        let actual_dim = if self.dim < 0 {
            (input_shape.len() as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        if actual_dim >= input_shape.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                self.dim,
                input_shape.len()
            )));
        }

        let dim_size = input_shape[actual_dim];
        if dim_size % 2 != 0 {
            return Err(TorshError::InvalidArgument(format!(
                "Input dimension {} must be even for SwiGLU, got {}",
                actual_dim, dim_size
            )));
        }

        let half_size = dim_size / 2;

        // Split the input
        let first_half = input.narrow(actual_dim as i32, 0, half_size)?;
        let second_half = input.narrow(actual_dim as i32, half_size as i64, half_size)?;

        // Apply SiLU to second half: x * sigmoid(x)
        let neg_second = second_half.neg()?;
        let exp_neg = neg_second.exp()?;
        let one_plus_exp = exp_neg.add_scalar(1.0)?;
        let ones_tensor = ones(second_half.shape().dims())?;
        let sigmoid = ones_tensor.div(&one_plus_exp)?;
        let silu_gate = second_half.mul(&sigmoid)?;

        // Element-wise multiplication
        first_half.mul(&silu_gate)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glu_dimension_parameter() {
        let glu = GLU::new(-1);
        assert_eq!(glu.dim(), -1);

        let glu_dim0 = GLU::new(0);
        assert_eq!(glu_dim0.dim(), 0);
    }

    #[test]
    fn test_glu_forward_shape() -> Result<()> {
        let glu = GLU::new(-1);
        let input = randn(&[2, 8])?; // Last dimension is even
        let output = glu.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 4]); // Last dim halved
        Ok(())
    }

    #[test]
    fn test_glu_invalid_dimension() -> Result<()> {
        let glu = GLU::new(-1);
        let input = randn(&[2, 7])?; // Odd last dimension
        let result = glu.forward(&input);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_geglu_forward_shape() -> Result<()> {
        let geglu = GEGLU::new(-1);
        let input = randn(&[2, 8])?;
        let output = geglu.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 4]);
        Ok(())
    }

    #[test]
    fn test_reglu_forward_shape() -> Result<()> {
        let reglu = ReGLU::new(-1);
        let input = randn(&[2, 8])?;
        let output = reglu.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 4]);
        Ok(())
    }

    #[test]
    fn test_swiglu_forward_shape() -> Result<()> {
        let swiglu = SwiGLU::new(-1);
        let input = randn(&[2, 8])?;
        let output = swiglu.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 4]);
        Ok(())
    }

    #[test]
    fn test_dimension_handling() -> Result<()> {
        // Test positive dimension
        let glu = GLU::new(1);
        let input = randn(&[2, 8, 3])?; // Dimension 1 has size 8 (even)
        let output = glu.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 4, 3]); // Dimension 1 halved

        // Test negative dimension
        let glu_neg = GLU::new(-2);
        let input = randn(&[2, 8, 3])?; // Dimension -2 is dimension 1
        let output = glu_neg.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 4, 3]);

        Ok(())
    }

    #[test]
    fn test_training_mode_toggle() -> Result<()> {
        let mut glu = GLU::new(-1);
        assert!(glu.training());

        glu.eval();
        assert!(!glu.training());

        glu.train();
        assert!(glu.training());

        Ok(())
    }

    #[test]
    fn test_default_implementations() {
        let _glu = GLU::default();
        let _geglu = GEGLU::default();
        let _reglu = ReGLU::default();
        let _swiglu = SwiGLU::default();
    }

    #[test]
    fn test_convenience_constructors() {
        let default_glu = GLU::default_dim();
        assert_eq!(default_glu.dim(), -1);
    }

    #[test]
    fn test_error_handling() -> Result<()> {
        let glu = GLU::new(10); // Invalid dimension
        let input = randn(&[2, 8])?; // Only 2 dimensions
        let result = glu.forward(&input);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_all_gated_functions_consistency() -> Result<()> {
        let input = randn(&[4, 16])?; // Even last dimension

        let glu = GLU::new(-1);
        let geglu = GEGLU::new(-1);
        let reglu = ReGLU::new(-1);
        let swiglu = SwiGLU::new(-1);

        let glu_out = glu.forward(&input)?;
        let geglu_out = geglu.forward(&input)?;
        let reglu_out = reglu.forward(&input)?;
        let swiglu_out = swiglu.forward(&input)?;

        // All should have the same output shape
        assert_eq!(glu_out.shape(), geglu_out.shape());
        assert_eq!(glu_out.shape(), reglu_out.shape());
        assert_eq!(glu_out.shape(), swiglu_out.shape());
        assert_eq!(glu_out.shape().dims(), &[4, 8]); // Halved last dimension

        Ok(())
    }
}
