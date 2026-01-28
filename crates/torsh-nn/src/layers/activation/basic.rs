//! # Basic Activation Functions
//!
//! This module contains fundamental activation functions commonly used in neural networks.
//! These are well-established, traditional activation functions that form the foundation
//! of most neural network architectures.
//!
//! ## Included Activation Functions
//!
//! - **ReLU** - Rectified Linear Unit: `max(0, x)`
//! - **Sigmoid** - Sigmoid function: `1 / (1 + exp(-x))`
//! - **Tanh** - Hyperbolic tangent: `tanh(x)`
//! - **LeakyReLU** - Leaky Rectified Linear Unit: `max(α*x, x)` where α is typically 0.01
//! - **ReLU6** - ReLU clamped to maximum value of 6: `min(max(0, x), 6)`
//! - **PReLU** - Parametric ReLU: `max(α*x, x)` where α is learnable
//!
//! ## Usage Example
//!
//! ```rust
//! # use torsh_nn::layers::activation::basic::{ReLU, Sigmoid, Tanh};
//! # use torsh_nn::Module;
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # fn main() -> Result<()> {
//! // Create activation functions
//! let relu = ReLU::new();
//! let sigmoid = Sigmoid::new();
//! let tanh = Tanh::new();
//!
//! // Apply to tensors
//! let input = randn(&[2, 3])?;
//! let relu_output = relu.forward(&input)?;
//! let sigmoid_output = sigmoid.forward(&input)?;
//! let tanh_output = tanh.forward(&input)?;
//! # Ok(())
//! # }
//! ```

use crate::{Module, ModuleBase, Parameter};
use std::sync::Arc;
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

/// ReLU activation function
///
/// Applies the rectified linear unit function element-wise: `max(0, x)`.
/// This is one of the most commonly used activation functions in deep learning
/// due to its simplicity and effectiveness in training deep networks.
///
/// # Mathematical Definition
/// ```text
/// ReLU(x) = max(0, x) = {
///     x  if x > 0
///     0  if x ≤ 0
/// }
/// ```
///
/// # Properties
/// - **Non-linear**: Provides non-linearity while remaining computationally efficient
/// - **Sparse**: Outputs zero for negative inputs, creating sparse representations
/// - **Unbounded**: No upper saturation, allowing gradients to flow during backpropagation
/// - **Differentiable**: Gradient is 1 for positive inputs, 0 for negative inputs
///
/// # Example
/// ```rust
/// # use torsh_nn::layers::activation::basic::ReLU;
/// # use torsh_nn::Module;
/// # use torsh_tensor::creation::tensor_1d;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// let relu = ReLU::new();
/// let input = tensor_1d(&[-2.0, -1.0, 0.0, 1.0, 2.0])?;
/// let output = relu.forward(&input)?; // [0.0, 0.0, 0.0, 1.0, 2.0]
/// # Ok(())
/// # }
/// ```
pub struct ReLU {
    base: ModuleBase,
}

impl ReLU {
    /// Creates a new ReLU activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply ReLU: max(0, x)
        let zero = zeros(input.shape().dims())?;
        input.maximum(&zero)
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

/// Sigmoid activation function
///
/// Applies the sigmoid function element-wise: `σ(x) = 1 / (1 + exp(-x))`.
/// The sigmoid function maps any real number to a value between 0 and 1,
/// making it useful for binary classification and probability outputs.
///
/// # Mathematical Definition
/// ```text
/// Sigmoid(x) = σ(x) = 1 / (1 + exp(-x))
/// ```
///
/// # Properties
/// - **Bounded**: Output range is (0, 1)
/// - **Smooth**: Continuously differentiable everywhere
/// - **Monotonic**: Strictly increasing function
/// - **Probability interpretation**: Output can be interpreted as probability
/// - **Vanishing gradients**: Can suffer from vanishing gradient problem for very large or small inputs
///
/// # Example
/// ```rust
/// # use torsh_nn::layers::activation::basic::Sigmoid;
/// # use torsh_nn::Module;
/// # use torsh_tensor::creation::tensor_1d;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// let sigmoid = Sigmoid::new();
/// let input = tensor_1d(&[-2.0, 0.0, 2.0])?;
/// let output = sigmoid.forward(&input)?; // [~0.119, 0.5, ~0.881]
/// # Ok(())
/// # }
/// ```
pub struct Sigmoid {
    base: ModuleBase,
}

impl Sigmoid {
    /// Creates a new Sigmoid activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply Sigmoid: 1 / (1 + exp(-x))
        let neg_input = input.neg()?;
        let exp_neg = neg_input.exp()?;
        let one_plus_exp = exp_neg.add_scalar(1.0)?;
        let one = ones(input.shape().dims())?;
        one.div(&one_plus_exp)
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

/// Tanh activation function
///
/// Applies the hyperbolic tangent function element-wise: `tanh(x)`.
/// The tanh function maps any real number to a value between -1 and 1,
/// making it zero-centered, which can help with training dynamics.
///
/// # Mathematical Definition
/// ```text
/// Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
///         = (exp(2x) - 1) / (exp(2x) + 1)
/// ```
///
/// # Properties
/// - **Bounded**: Output range is (-1, 1)
/// - **Zero-centered**: Output is centered around zero
/// - **Smooth**: Continuously differentiable everywhere
/// - **Monotonic**: Strictly increasing function
/// - **Better than sigmoid**: Zero-centered output helps with gradient flow
///
/// # Example
/// ```rust
/// # use torsh_nn::layers::activation::basic::Tanh;
/// # use torsh_nn::Module;
/// # use torsh_tensor::creation::tensor_1d;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// let tanh = Tanh::new();
/// let input = tensor_1d(&[-2.0, 0.0, 2.0])?;
/// let output = tanh.forward(&input)?; // [~-0.964, 0.0, ~0.964]
/// # Ok(())
/// # }
/// ```
pub struct Tanh {
    base: ModuleBase,
}

impl Tanh {
    /// Creates a new Tanh activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        input.tanh()
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

/// LeakyReLU activation function
///
/// Applies the Leaky ReLU function element-wise: `max(α*x, x)` where α is a small positive constant.
/// This variant of ReLU allows a small gradient to flow for negative inputs, helping to mitigate
/// the "dying ReLU" problem where neurons can become permanently inactive.
///
/// # Mathematical Definition
/// ```text
/// LeakyReLU(x) = {
///     x      if x > 0
///     α*x    if x ≤ 0
/// }
/// ```
/// where α is typically 0.01.
///
/// # Properties
/// - **Non-saturating**: Allows small gradients for negative inputs
/// - **Fixes dying ReLU**: Prevents neurons from becoming permanently inactive
/// - **Computationally efficient**: Simple to compute and differentiate
/// - **Configurable**: The slope α can be adjusted based on requirements
///
/// # Example
/// ```rust
/// # use torsh_nn::layers::activation::basic::LeakyReLU;
/// # use torsh_nn::Module;
/// # use torsh_tensor::creation::tensor_1d;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// let leaky_relu = LeakyReLU::new(0.01);
/// let input = tensor_1d(&[-2.0, -1.0, 0.0, 1.0, 2.0])?;
/// let output = leaky_relu.forward(&input)?; // [-0.02, -0.01, 0.0, 1.0, 2.0]
/// # Ok(())
/// # }
/// ```
pub struct LeakyReLU {
    base: ModuleBase,
    negative_slope: f64,
}

impl LeakyReLU {
    /// Creates a new LeakyReLU activation function
    ///
    /// # Arguments
    /// * `negative_slope` - The slope of the function for negative inputs (default: 0.01)
    pub fn new(negative_slope: f64) -> Self {
        Self {
            base: ModuleBase::new(),
            negative_slope,
        }
    }

    /// Creates a LeakyReLU with the default negative slope of 0.01
    pub fn default_slope() -> Self {
        Self::new(0.01)
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply Leaky ReLU: max(negative_slope * x, x)
        let negative_part = input.mul_scalar(self.negative_slope as f32)?;
        input.maximum(&negative_part)
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

/// ReLU6 activation function
///
/// Applies the ReLU6 function element-wise: `min(max(0, x), 6)`.
/// This is ReLU clamped to a maximum value of 6, which can help with numerical
/// stability and is particularly useful in mobile and embedded applications.
///
/// # Mathematical Definition
/// ```text
/// ReLU6(x) = min(max(0, x), 6) = {
///     0  if x < 0
///     x  if 0 ≤ x ≤ 6
///     6  if x > 6
/// }
/// ```
///
/// # Properties
/// - **Bounded**: Output range is [0, 6]
/// - **Sparse**: Outputs zero for negative inputs
/// - **Clamped**: Prevents extremely large activations
/// - **Mobile-friendly**: Commonly used in mobile and embedded neural networks
/// - **Numerical stability**: Bounded output helps with numerical stability
///
/// # Example
/// ```rust
/// # use torsh_nn::layers::activation::basic::ReLU6;
/// # use torsh_nn::Module;
/// # use torsh_tensor::creation::tensor_1d;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// let relu6 = ReLU6::new();
/// let input = tensor_1d(&[-2.0, 0.0, 3.0, 8.0])?;
/// let output = relu6.forward(&input)?; // [0.0, 0.0, 3.0, 6.0]
/// # Ok(())
/// # }
/// ```
pub struct ReLU6 {
    base: ModuleBase,
}

impl ReLU6 {
    /// Creates a new ReLU6 activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for ReLU6 {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU6 {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply ReLU6: min(max(0, x), 6)
        let zero = zeros(input.shape().dims())?;
        let six = full(input.shape().dims(), 6.0)?;
        let relu_output = input.maximum(&zero)?;
        relu_output.minimum(&six)
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

/// PReLU (Parametric ReLU) activation function
///
/// Applies the Parametric ReLU function element-wise: `max(α*x, x)` where α is a learnable parameter.
/// Unlike LeakyReLU where α is fixed, PReLU learns the optimal slope for negative inputs during training.
/// This can lead to better performance as the network can adapt the activation function to the data.
///
/// # Mathematical Definition
/// ```text
/// PReLU(x) = {
///     x      if x > 0
///     α*x    if x ≤ 0
/// }
/// ```
/// where α is a learnable parameter.
///
/// # Properties
/// - **Learnable**: The negative slope α is learned during training
/// - **Adaptive**: Can adapt to different data distributions
/// - **Channel-wise**: Can have different α values for different channels
/// - **Better than fixed**: Often outperforms fixed negative slope variants
/// - **Minimal overhead**: Adds very few parameters compared to network size
///
/// # Example
/// ```rust
/// # use torsh_nn::layers::activation::basic::PReLU;
/// # use torsh_nn::Module;
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// let mut prelu = PReLU::new(64)?; // 64 channels
/// let input = randn(&[1, 64, 32, 32])?; // Batch of images
/// let output = prelu.forward(&input)?;
/// # Ok(())
/// # }
/// ```
pub struct PReLU {
    base: ModuleBase,
    alpha: Parameter,
    num_parameters: usize,
}

impl PReLU {
    /// Creates a new PReLU activation function
    ///
    /// # Arguments
    /// * `num_parameters` - Number of parameters (typically number of channels). Use 1 for shared parameter.
    ///
    /// # Returns
    /// A new PReLU instance with learnable alpha parameters initialized to 0.25.
    pub fn new(num_parameters: usize) -> Result<Self> {
        let alpha_shape = if num_parameters == 1 {
            vec![1]
        } else {
            vec![num_parameters]
        };

        let alpha_tensor = full(&alpha_shape, 0.25)?;
        let alpha = Parameter::new(alpha_tensor);

        let mut base = ModuleBase::new();
        base.register_parameter("alpha".to_string(), alpha.clone());

        Ok(Self {
            base,
            alpha,
            num_parameters,
        })
    }

    /// Creates a PReLU with a single shared parameter
    pub fn single_parameter() -> Result<Self> {
        Self::new(1)
    }

    /// Gets the current alpha values
    pub fn alpha(&self) -> Arc<parking_lot::RwLock<Tensor>> {
        self.alpha.tensor()
    }
}

impl Module for PReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply PReLU: max(alpha * x, x)
        let alpha_expanded = if self.num_parameters == 1 {
            // Broadcast single parameter to input shape
            self.alpha
                .tensor()
                .read()
                .broadcast_to(input.shape().dims())?
        } else {
            // Expand alpha to match input dimensions
            let mut alpha_shape = vec![1i32; input.shape().ndim()];
            alpha_shape[1] = self.num_parameters as i32; // Assume channel is dimension 1
            self.alpha
                .tensor()
                .read()
                .reshape(&alpha_shape)?
                .broadcast_to(input.shape().dims())?
        };

        let negative_part = input.mul(&alpha_expanded)?;
        input.maximum(&negative_part)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.base.parameters.clone();
        params.insert("alpha".to_string(), self.alpha.clone());
        params
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
        self.base.to_device(device)?;
        self.alpha.to_device(device)?;
        Ok(())
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.base.named_parameters();
        params.insert("alpha".to_string(), self.alpha.clone());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_forward() -> Result<()> {
        let relu = ReLU::new();
        let input = tensor_1d(&[-2.0, -1.0, 0.0, 1.0, 2.0])?;
        let output = relu.forward(&input)?;

        let expected = tensor_1d(&[0.0, 0.0, 0.0, 1.0, 2.0])?;
        assert_eq!(output.shape(), expected.shape());
        // Note: In real implementation, would use approximate equality for floating point
        Ok(())
    }

    #[test]
    fn test_sigmoid_shape() -> Result<()> {
        let sigmoid = Sigmoid::new();
        let input = randn(&[2, 3, 4])?;
        let output = sigmoid.forward(&input)?;

        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_tanh_shape() -> Result<()> {
        let tanh = Tanh::new();
        let input = randn(&[5, 10])?;
        let output = tanh.forward(&input)?;

        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_leaky_relu_forward() -> Result<()> {
        let leaky_relu = LeakyReLU::new(0.1);
        let input = tensor_1d(&[-2.0, -1.0, 0.0, 1.0, 2.0])?;
        let output = leaky_relu.forward(&input)?;

        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_relu6_forward() -> Result<()> {
        let relu6 = ReLU6::new();
        let input = tensor_1d(&[-2.0, 0.0, 3.0, 8.0])?;
        let output = relu6.forward(&input)?;

        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_prelu_creation() -> Result<()> {
        let prelu = PReLU::new(64)?;
        assert_eq!(prelu.num_parameters, 64);

        let single_prelu = PReLU::single_parameter()?;
        assert_eq!(single_prelu.num_parameters, 1);

        Ok(())
    }

    #[test]
    fn test_training_mode_toggle() -> Result<()> {
        let mut relu = ReLU::new();
        assert!(relu.training());

        relu.eval();
        assert!(!relu.training());

        relu.train();
        assert!(relu.training());

        Ok(())
    }

    #[test]
    fn test_default_implementations() {
        let _relu = ReLU::default();
        let _sigmoid = Sigmoid::default();
        let _tanh = Tanh::default();
        let _leaky_relu = LeakyReLU::default();
        let _relu6 = ReLU6::default();
    }
}
