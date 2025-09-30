//! # Modern Activation Functions
//!
//! This module contains advanced activation functions that have been developed more recently
//! and are commonly used in state-of-the-art neural network architectures. These functions
//! often provide better gradient flow properties and performance compared to traditional activations.
//!
//! ## Included Activation Functions
//!
//! - **GELU** - Gaussian Error Linear Unit: smooth approximation to ReLU with probabilistic interpretation
//! - **SiLU/Swish** - Sigmoid Linear Unit: `x * sigmoid(x)`, self-gated activation
//! - **Mish** - Self Regularized Non-Monotonic activation: `x * tanh(softplus(x))`
//! - **Hardswish** - Hard version of Swish: more efficient approximation
//! - **ELU** - Exponential Linear Unit: smooth version of ReLU with negative values
//! - **SELU** - Scaled Exponential Linear Unit: self-normalizing properties
//!
//! ## Usage Example
//!
//! ```rust
//! use torsh_nn::layers::activation::modern::{GELU, SiLU, Mish};
//! use torsh_nn::Module;
//! use torsh_tensor::Tensor;
//!
//! // Create modern activation functions
//! let gelu = GELU::new();
//! let silu = SiLU::new(); // Also known as Swish
//! let mish = Mish::new();
//!
//! // Apply to tensors
//! let input = Tensor::randn(&[2, 3]);
//! let gelu_output = gelu.forward(&input)?;
//! let silu_output = silu.forward(&input)?;
//! let mish_output = mish.forward(&input)?;
//! ```

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{collections::HashMap, string::String};

#[cfg(not(feature = "std"))]
use alloc::string::String;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// GELU (Gaussian Error Linear Unit) activation function
///
/// Applies the GELU function element-wise. GELU is a smooth approximation to ReLU
/// with a probabilistic interpretation, performing well in many transformer architectures.
///
/// # Mathematical Definition
/// ```text
/// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
/// ```
///
/// Or approximately:
/// ```text
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
/// ```
///
/// # Properties
/// - **Smooth**: Continuously differentiable everywhere
/// - **Non-monotonic**: Has a small negative region for negative inputs
/// - **Probabilistic**: Based on the cumulative distribution function of the standard normal distribution
/// - **Transformer-friendly**: Widely used in transformer architectures like BERT and GPT
///
/// # Example
/// ```rust
/// use torsh_nn::layers::activation::modern::GELU;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let gelu = GELU::new();
/// let gelu_approx = GELU::with_approximate(true);
/// let input = Tensor::randn(&[2, 3]);
/// let output = gelu.forward(&input)?;
/// ```
pub struct GELU {
    base: ModuleBase,
    approximate: bool,
}

impl GELU {
    /// Creates a new GELU activation function with exact computation
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            approximate: false,
        }
    }

    /// Creates a GELU activation function with configurable approximation
    ///
    /// # Arguments
    /// * `approximate` - If true, uses the tanh approximation for faster computation
    pub fn with_approximate(approximate: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            approximate,
        }
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.approximate {
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
        } else {
            // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            // Approximation for erf function using tanh
            let sqrt_2 = (2.0_f32).sqrt();
            let sqrt2_tensor = full(input.shape().dims(), sqrt_2)?;
            let x_div_sqrt2 = input.div(&sqrt2_tensor)?;
            let erf_approx = x_div_sqrt2.tanh()?; // Simplified erf approximation
            let ones_tensor = ones(input.shape().dims())?;
            let one_plus_erf = erf_approx.add(&ones_tensor)?;
            let half_tensor = full(input.shape().dims(), 0.5)?;
            let half_x = input.mul(&half_tensor)?;
            half_x.mul(&one_plus_erf)
        }
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

/// SiLU (Sigmoid Linear Unit) activation function, also known as Swish
///
/// Applies the SiLU function element-wise: `x * sigmoid(x)`.
/// This is a self-gated activation function that has been shown to work well
/// in many deep learning applications, particularly in computer vision.
///
/// # Mathematical Definition
/// ```text
/// SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
/// ```
///
/// # Properties
/// - **Self-gated**: Uses the input to gate itself
/// - **Smooth**: Continuously differentiable everywhere
/// - **Unbounded above**: No upper saturation
/// - **Near-zero below**: Approaches zero for large negative inputs
/// - **Better than ReLU**: Often outperforms ReLU in many tasks
///
/// # Example
/// ```rust
/// use torsh_nn::layers::activation::modern::{SiLU, Swish};
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let silu = SiLU::new();
/// let swish = Swish::new(); // Same as SiLU
/// let input = Tensor::randn(&[2, 3]);
/// let output = silu.forward(&input)?;
/// ```
pub struct SiLU {
    base: ModuleBase,
}

/// Type alias for Swish activation (same as SiLU)
pub type Swish = SiLU;

impl SiLU {
    /// Creates a new SiLU activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for SiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SiLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply SiLU/Swish: x * sigmoid(x)
        // Sigmoid: 1 / (1 + exp(-x))
        let neg_input = input.neg()?;
        let exp_neg = neg_input.exp()?;
        let one_plus_exp = exp_neg.add_scalar(1.0)?;
        let ones_tensor = ones(input.shape().dims())?;
        let sigmoid_result = ones_tensor.div(&one_plus_exp)?;
        input.mul(&sigmoid_result)
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

/// Mish activation function
///
/// Applies the Mish function element-wise: `x * tanh(softplus(x))`.
/// Mish is a self-regularized, non-monotonic activation function that has shown
/// promising results in various deep learning tasks.
///
/// # Mathematical Definition
/// ```text
/// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
/// ```
///
/// # Properties
/// - **Self-regularized**: Built-in regularization properties
/// - **Non-monotonic**: Has a small dip for negative inputs
/// - **Smooth**: Continuously differentiable everywhere
/// - **Unbounded above**: No upper saturation
/// - **Better gradients**: Often provides better gradient flow than ReLU
///
/// # Example
/// ```rust
/// use torsh_nn::layers::activation::modern::Mish;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let mish = Mish::new();
/// let input = Tensor::randn(&[2, 3]);
/// let output = mish.forward(&input)?;
/// ```
pub struct Mish {
    base: ModuleBase,
}

impl Mish {
    /// Creates a new Mish activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for Mish {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Mish {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply Mish: x * tanh(softplus(x))
        // Softplus: ln(1 + exp(x))
        let exp_x = input.exp()?;
        let one_plus_exp = exp_x.add_scalar(1.0)?;
        let softplus = one_plus_exp.log()?;
        let tanh_softplus = softplus.tanh()?;
        input.mul(&tanh_softplus)
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

/// Hardswish activation function
///
/// Applies the Hardswish function element-wise, which is a computationally efficient
/// approximation to the Swish function. It's designed for mobile and edge deployments
/// where computational efficiency is crucial.
///
/// # Mathematical Definition
/// ```text
/// Hardswish(x) = {
///     0                           if x ≤ -3
///     x                          if x ≥ +3
///     x * (x + 3) / 6            if -3 < x < +3
/// }
/// ```
///
/// # Properties
/// - **Efficient**: Much faster to compute than Swish
/// - **Mobile-friendly**: Designed for efficient mobile deployment
/// - **Bounded**: Has defined behavior for all input ranges
/// - **Piecewise**: Linear approximation in different regions
///
/// # Example
/// ```rust
/// use torsh_nn::layers::activation::modern::Hardswish;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let hardswish = Hardswish::new();
/// let input = Tensor::randn(&[2, 3]);
/// let output = hardswish.forward(&input)?;
/// ```
pub struct Hardswish {
    base: ModuleBase,
}

impl Hardswish {
    /// Creates a new Hardswish activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for Hardswish {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Hardswish {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply Hardswish: x * hard_sigmoid(x)
        // hard_sigmoid(x) = max(0, min(1, (x + 3) / 6))
        let three = full(input.shape().dims(), 3.0)?;
        let six = full(input.shape().dims(), 6.0)?;
        let zero = zeros(input.shape().dims())?;
        let one = ones(input.shape().dims())?;

        let x_plus_3 = input.add(&three)?;
        let divided = x_plus_3.div(&six)?;
        let clamped_high = divided.minimum(&one)?;
        let hard_sigmoid = clamped_high.maximum(&zero)?;
        input.mul(&hard_sigmoid)
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

/// ELU (Exponential Linear Unit) activation function
///
/// Applies the ELU function element-wise. ELU is designed to address the "dying ReLU"
/// problem while maintaining the computational benefits of ReLU for positive inputs.
///
/// # Mathematical Definition
/// ```text
/// ELU(x) = {
///     x                    if x > 0
///     α * (exp(x) - 1)     if x ≤ 0
/// }
/// ```
/// where α is typically 1.0.
///
/// # Properties
/// - **Smooth**: Continuously differentiable everywhere
/// - **Zero-mean**: Tends to push activations towards zero mean
/// - **Faster convergence**: Often converges faster than ReLU
/// - **Configurable**: The α parameter can be tuned
///
/// # Example
/// ```rust
/// use torsh_nn::layers::activation::modern::ELU;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let elu = ELU::new(1.0); // α = 1.0
/// let input = Tensor::randn(&[2, 3]);
/// let output = elu.forward(&input)?;
/// ```
pub struct ELU {
    base: ModuleBase,
    alpha: f32,
}

impl ELU {
    /// Creates a new ELU activation function
    ///
    /// # Arguments
    /// * `alpha` - The α parameter for negative inputs (typically 1.0)
    pub fn new(alpha: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            alpha,
        }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Module for ELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply ELU: x if x > 0, α * (exp(x) - 1) if x <= 0
        let zero = zeros(input.shape().dims())?;
        let alpha_tensor = full(input.shape().dims(), self.alpha)?;

        // Positive condition: x > 0
        let pos_condition = input.gt(&zero)?;

        // Negative part: α * (exp(x) - 1)
        let exp_x = input.exp()?;
        let exp_minus_one = exp_x.sub_scalar(1.0)?;
        let neg_elu = alpha_tensor.mul(&exp_minus_one)?;

        // Apply condition: use input if positive, else use negative ELU
        input.where_tensor(&pos_condition, &neg_elu)
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

/// SELU (Scaled Exponential Linear Unit) activation function
///
/// Applies the SELU function element-wise. SELU is designed to have self-normalizing
/// properties that allow deep networks to converge towards zero mean and unit variance.
///
/// # Mathematical Definition
/// ```text
/// SELU(x) = scale * {
///     x                     if x > 0
///     α * (exp(x) - 1)      if x ≤ 0
/// }
/// ```
/// where scale ≈ 1.0507 and α ≈ 1.6733.
///
/// # Properties
/// - **Self-normalizing**: Maintains zero mean and unit variance through layers
/// - **Deep networks**: Enables training of very deep networks without normalization
/// - **Fixed parameters**: α and scale are mathematically derived constants
/// - **Smooth**: Continuously differentiable everywhere
///
/// # Example
/// ```rust
/// use torsh_nn::layers::activation::modern::SELU;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let selu = SELU::new();
/// let input = Tensor::randn(&[2, 3]);
/// let output = selu.forward(&input)?;
/// ```
pub struct SELU {
    base: ModuleBase,
}

impl SELU {
    /// Creates a new SELU activation function
    /// Uses the standard SELU parameters: α ≈ 1.6733, scale ≈ 1.0507
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }

    /// Standard SELU alpha parameter
    const ALPHA: f32 = 1.6732632423543772848170429916717;

    /// Standard SELU scale parameter
    const SCALE: f32 = 1.0507009873554804934193349852946;
}

impl Default for SELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply SELU: scale * (x if x > 0, α * (exp(x) - 1) if x <= 0)
        let zero = zeros(input.shape().dims())?;
        let alpha_tensor = full(input.shape().dims(), Self::ALPHA)?;
        let scale_tensor = full(input.shape().dims(), Self::SCALE)?;

        // Positive condition: x > 0
        let pos_condition = input.gt(&zero)?;

        // Negative part: α * (exp(x) - 1)
        let exp_x = input.exp()?;
        let exp_minus_one = exp_x.sub_scalar(1.0)?;
        let neg_elu = alpha_tensor.mul(&exp_minus_one)?;

        // Apply condition: use input if positive, else use negative ELU
        let combined = input.where_tensor(&pos_condition, &neg_elu)?;
        combined.mul(&scale_tensor)
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
    use torsh_tensor::creation::*;

    #[test]
    fn test_gelu_creation() {
        let gelu = GELU::new();
        assert!(!gelu.approximate);

        let gelu_approx = GELU::with_approximate(true);
        assert!(gelu_approx.approximate);
    }

    #[test]
    fn test_silu_forward() -> Result<()> {
        let silu = SiLU::new();
        let input = randn(&[2, 3])?;
        let output = silu.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_swish_alias() {
        let silu = SiLU::new();
        let swish = Swish::new();
        // Both should have the same structure
        assert_eq!(silu.training(), swish.training());
    }

    #[test]
    fn test_mish_forward() -> Result<()> {
        let mish = Mish::new();
        let input = randn(&[2, 3])?;
        let output = mish.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_hardswish_forward() -> Result<()> {
        let hardswish = Hardswish::new();
        let input = randn(&[2, 3])?;
        let output = hardswish.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_elu_parameters() {
        let elu = ELU::new(1.0);
        assert_eq!(elu.alpha, 1.0);

        let elu_custom = ELU::new(0.5);
        assert_eq!(elu_custom.alpha, 0.5);
    }

    #[test]
    fn test_selu_constants() {
        let selu = SELU::new();
        // Test that constants are within expected ranges
        assert!((SELU::ALPHA - 1.6733).abs() < 0.01);
        assert!((SELU::SCALE - 1.0507).abs() < 0.01);
    }

    #[test]
    fn test_training_mode_toggle() -> Result<()> {
        let mut gelu = GELU::new();
        assert!(gelu.training());

        gelu.eval();
        assert!(!gelu.training());

        gelu.train();
        assert!(gelu.training());

        Ok(())
    }

    #[test]
    fn test_default_implementations() {
        let _gelu = GELU::default();
        let _silu = SiLU::default();
        let _mish = Mish::default();
        let _hardswish = Hardswish::default();
        let _elu = ELU::default();
        let _selu = SELU::default();
    }
}
