//! # Smooth Activation Functions
//!
//! This module contains activation functions that provide smooth, continuously differentiable
//! transformations. These functions are characterized by their smooth gradients and lack of
//! sharp discontinuities, making them well-suited for gradient-based optimization.
//!
//! ## Included Activation Functions
//!
//! - **Softplus** - Smooth approximation to ReLU: `log(1 + exp(x))`
//! - **Softsign** - Smooth alternative to tanh: `x / (1 + |x|)`
//! - **Hardsigmoid** - Piecewise linear approximation to sigmoid
//!
//! ## Usage Example
//!
//! ```rust
//! # use torsh_tensor::creation::{randn, tensor_1d};
//! # use torsh_core::error::Result;
//! # fn main() -> Result<()> {
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # fn main() -> Result<()> {
//! use torsh_nn::layers::activation::smooth::{Softplus, Softsign, Hardsigmoid};
//! use torsh_nn::Module;
//! use torsh_tensor::Tensor;
//!
//! // Create smooth activation functions
//! let softplus = Softplus::new(1.0, 20.0); // β=1.0, threshold=20.0
//! let softsign = Softsign::new();
//! let hardsigmoid = Hardsigmoid::new();
//!
//! // Apply to tensors
//! let input = randn(&[2, 3])?;
//! let softplus_output = softplus.forward(&input)?;
//! let softsign_output = softsign.forward(&input)?;
//! let hardsigmoid_output = hardsigmoid.forward(&input)?;
//! # Ok(())
//! # }
//! # Ok(())
//! # }
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

/// Softplus activation function
///
/// Applies the softplus function element-wise: `(1/β) * log(1 + exp(β * x))`.
/// Softplus is a smooth approximation to the ReLU function that has the advantage
/// of being differentiable everywhere and having a strictly positive output.
///
/// # Mathematical Definition
/// ```text
/// Softplus(x) = (1/β) * log(1 + exp(β * x))
/// ```
/// where β controls the sharpness of the approximation.
///
/// # Properties
/// - **Smooth**: Continuously differentiable everywhere
/// - **Positive**: Always outputs positive values
/// - **ReLU approximation**: Approaches ReLU as β → ∞
/// - **Configurable sharpness**: β parameter controls steepness
/// - **Numerically stable**: Uses threshold for large inputs to prevent overflow
///
/// # Parameters
/// - **β (beta)**: Controls the sharpness of the approximation (default: 1.0)
/// - **threshold**: Values above β*threshold use linear approximation (default: 20.0)
///
/// # Example
/// ```rust
/// # use torsh_tensor::creation::{randn, tensor_1d};
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_nn::layers::activation::smooth::Softplus;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let softplus = Softplus::new(1.0, 20.0); // Standard softplus
/// let sharp_softplus = Softplus::new(5.0, 20.0); // Sharper approximation to ReLU
/// let input = tensor_1d(&[-2.0, -1.0, 0.0, 1.0, 2.0])?;
/// let output = softplus.forward(&input)?;
/// # Ok(())
/// # }
/// # Ok(())
/// # }
/// ```
pub struct Softplus {
    base: ModuleBase,
    beta: f32,
    threshold: f32,
}

impl Softplus {
    /// Creates a new Softplus activation function
    ///
    /// # Arguments
    /// * `beta` - The β parameter controlling sharpness (typically 1.0)
    /// * `threshold` - Threshold for numerical stability (typically 20.0)
    pub fn new(beta: f32, threshold: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            beta,
            threshold,
        }
    }

    /// Creates a standard Softplus with β=1.0 and threshold=20.0
    pub fn standard() -> Self {
        Self::new(1.0, 20.0)
    }

    /// Gets the beta parameter
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Gets the threshold parameter
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

impl Default for Softplus {
    fn default() -> Self {
        Self::new(1.0, 20.0)
    }
}

impl Module for Softplus {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply softplus: (1/β) * log(1 + exp(β * x))
        // Use threshold for numerical stability
        let beta_tensor = full(input.shape().dims(), self.beta)?;
        let beta_x = input.mul(&beta_tensor)?;
        let threshold_tensor = full(input.shape().dims(), self.threshold)?;

        // For values above threshold, use linear approximation to avoid overflow
        let above_threshold = beta_x.gt(&threshold_tensor)?;

        // Softplus part for regular values
        let exp_beta_x = beta_x.exp()?;
        let ones_tensor = ones(input.shape().dims())?;
        let one_plus_exp = exp_beta_x.add(&ones_tensor)?;
        let log_part = one_plus_exp.log()?;
        let inv_beta_tensor = full(input.shape().dims(), 1.0 / self.beta)?;
        let softplus_part = log_part.mul(&inv_beta_tensor)?;

        // Use linear approximation (x) if above threshold, else use softplus
        input.where_tensor(&above_threshold, &softplus_part)
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

/// Softsign activation function
///
/// Applies the softsign function element-wise: `x / (1 + |x|)`.
/// Softsign is a smooth alternative to the tanh function that is computationally
/// simpler and has different saturation properties.
///
/// # Mathematical Definition
/// ```text
/// Softsign(x) = x / (1 + |x|)
/// ```
///
/// # Properties
/// - **Bounded**: Output range is (-1, 1)
/// - **Smooth**: Continuously differentiable everywhere
/// - **Zero-centered**: f(0) = 0
/// - **Symmetric**: f(-x) = -f(x)
/// - **Slow saturation**: Saturates more slowly than tanh
/// - **Computationally simple**: No exponentials required
///
/// # Comparison with Tanh
/// - Softsign saturates more slowly than tanh
/// - Computationally simpler (no exp operations)
/// - Different gradient properties near saturation
///
/// # Example
/// ```rust
/// # use torsh_tensor::creation::{randn, tensor_1d};
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_nn::layers::activation::smooth::Softsign;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let softsign = Softsign::new();
/// let input = tensor_1d(&[-5.0, -1.0, 0.0, 1.0, 5.0])?;
/// let output = softsign.forward(&input)?; // [-0.833, -0.5, 0.0, 0.5, 0.833]
/// # Ok(())
/// # }
/// # Ok(())
/// # }
/// ```
pub struct Softsign {
    base: ModuleBase,
}

impl Softsign {
    /// Creates a new Softsign activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for Softsign {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Softsign {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply softsign: x / (1 + |x|)
        let abs_input = input.abs()?;
        let ones_tensor = ones(input.shape().dims())?;
        let denominator = abs_input.add(&ones_tensor)?;
        input.div(&denominator)
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

/// Hardsigmoid activation function
///
/// Applies a piecewise linear approximation to the sigmoid function.
/// This function is computationally efficient and is commonly used in
/// mobile and embedded applications where computational resources are limited.
///
/// # Mathematical Definition
/// ```text
/// Hardsigmoid(x) = max(0, min(1, α * x + β))
/// ```
/// where α and β are parameters (typically α = 0.2, β = 0.5).
///
/// # Standard Form
/// ```text
/// Hardsigmoid(x) = max(0, min(1, 0.2 * x + 0.5))
/// ```
///
/// # Properties
/// - **Piecewise linear**: Three linear segments
/// - **Bounded**: Output range is [0, 1]
/// - **Efficient**: No exponential computations required
/// - **Mobile-friendly**: Ideal for resource-constrained environments
/// - **Sigmoid approximation**: Approximates sigmoid in the linear region
///
/// # Example
/// ```rust
/// # use torsh_tensor::creation::{randn, tensor_1d};
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_nn::layers::activation::smooth::Hardsigmoid;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let hardsigmoid = Hardsigmoid::new();
/// let input = tensor_1d(&[-3.0, -1.0, 0.0, 1.0, 3.0])?;
/// let output = hardsigmoid.forward(&input)?; // [0.0, 0.3, 0.5, 0.7, 1.0]
/// # Ok(())
/// # }
/// # Ok(())
/// # }
/// ```
pub struct Hardsigmoid {
    base: ModuleBase,
    alpha: f32,
    beta: f32,
}

impl Hardsigmoid {
    /// Creates a new Hardsigmoid activation function with custom parameters
    ///
    /// # Arguments
    /// * `alpha` - The slope parameter (typically 0.2)
    /// * `beta` - The offset parameter (typically 0.5)
    pub fn new_with_params(alpha: f32, beta: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            alpha,
            beta,
        }
    }

    /// Creates a standard Hardsigmoid with α=0.2 and β=0.5
    pub fn new() -> Self {
        Self::new_with_params(0.2, 0.5)
    }

    /// Gets the alpha parameter
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Gets the beta parameter
    pub fn beta(&self) -> f32 {
        self.beta
    }
}

impl Default for Hardsigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Hardsigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply hardsigmoid: max(0, min(1, α * x + β))
        let alpha_tensor = full(input.shape().dims(), self.alpha)?;
        let beta_tensor = full(input.shape().dims(), self.beta)?;
        let zero_tensor = zeros(input.shape().dims())?;
        let one_tensor = ones(input.shape().dims())?;

        // Calculate α * x + β
        let linear = input.mul(&alpha_tensor)?.add(&beta_tensor)?;

        // Apply min(1, α * x + β)
        let clamped_high = linear.minimum(&one_tensor)?;

        // Apply max(0, min(1, α * x + β))
        clamped_high.maximum(&zero_tensor)
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
    fn test_softplus_parameters() {
        let softplus = Softplus::new(2.0, 30.0);
        assert_eq!(softplus.beta(), 2.0);
        assert_eq!(softplus.threshold(), 30.0);

        let standard_softplus = Softplus::standard();
        assert_eq!(standard_softplus.beta(), 1.0);
        assert_eq!(standard_softplus.threshold(), 20.0);
    }

    #[test]
    fn test_softplus_forward() -> Result<()> {
        let softplus = Softplus::new(1.0, 20.0);
        let input = randn(&[2, 3])?;
        let output = softplus.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_softsign_forward() -> Result<()> {
        let softsign = Softsign::new();
        let input = randn(&[2, 3])?;
        let output = softsign.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_hardsigmoid_parameters() {
        let hardsigmoid = Hardsigmoid::new_with_params(0.1, 0.6);
        assert_eq!(hardsigmoid.alpha(), 0.1);
        assert_eq!(hardsigmoid.beta(), 0.6);

        let standard_hardsigmoid = Hardsigmoid::new();
        assert_eq!(standard_hardsigmoid.alpha(), 0.2);
        assert_eq!(standard_hardsigmoid.beta(), 0.5);
    }

    #[test]
    fn test_hardsigmoid_forward() -> Result<()> {
        let hardsigmoid = Hardsigmoid::new();
        let input = randn(&[2, 3])?;
        let output = hardsigmoid.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_softplus_numerical_behavior() -> Result<()> {
        let softplus = Softplus::new(1.0, 20.0);

        // Test with small values (should use softplus formula)
        let small_input = full(&[2, 2], 1.0)?;
        let result = softplus.forward(&small_input)?;
        assert_eq!(result.shape(), small_input.shape());

        // Test with large values (should use linear approximation)
        let large_input = full(&[2, 2], 25.0)?;
        let result = softplus.forward(&large_input)?;
        assert_eq!(result.shape(), large_input.shape());

        Ok(())
    }

    #[test]
    fn test_softsign_range() -> Result<()> {
        let softsign = Softsign::new();

        // Test that output is bounded in (-1, 1)
        let large_positive = full(&[2, 2], 1000.0)?;
        let result = softsign.forward(&large_positive)?;
        assert_eq!(result.shape(), large_positive.shape());

        let large_negative = full(&[2, 2], -1000.0)?;
        let result = softsign.forward(&large_negative)?;
        assert_eq!(result.shape(), large_negative.shape());

        Ok(())
    }

    #[test]
    fn test_hardsigmoid_range() -> Result<()> {
        let hardsigmoid = Hardsigmoid::new();

        // Test that output is bounded in [0, 1]
        let large_positive = full(&[2, 2], 1000.0)?;
        let result = hardsigmoid.forward(&large_positive)?;
        assert_eq!(result.shape(), large_positive.shape());

        let large_negative = full(&[2, 2], -1000.0)?;
        let result = hardsigmoid.forward(&large_negative)?;
        assert_eq!(result.shape(), large_negative.shape());

        Ok(())
    }

    #[test]
    fn test_training_mode_toggle() -> Result<()> {
        let mut softplus = Softplus::new(1.0, 20.0);
        assert!(softplus.training());

        softplus.eval();
        assert!(!softplus.training());

        softplus.train();
        assert!(softplus.training());

        Ok(())
    }

    #[test]
    fn test_default_implementations() {
        let _softplus = Softplus::default();
        let _softsign = Softsign::default();
        let _hardsigmoid = Hardsigmoid::default();
    }

    #[test]
    fn test_convenience_constructors() {
        let standard_softplus = Softplus::standard();
        assert_eq!(standard_softplus.beta(), 1.0);
        assert_eq!(standard_softplus.threshold(), 20.0);

        let custom_hardsigmoid = Hardsigmoid::new_with_params(0.1, 0.3);
        assert_eq!(custom_hardsigmoid.alpha(), 0.1);
        assert_eq!(custom_hardsigmoid.beta(), 0.3);
    }
}
