//! # Threshold and Shrinking Activation Functions
//!
//! This module contains activation functions that perform thresholding and shrinking operations.
//! These functions are useful for creating sparse representations and implementing various
//! non-linear transformations with specific threshold behaviors.
//!
//! ## Included Activation Functions
//!
//! - **Hardshrink** - Hard shrinkage function: zeros out values within a threshold range
//! - **Softshrink** - Soft shrinkage function: soft thresholding with linear shrinkage
//! - **Hardtanh** - Hard hyperbolic tangent: clamped linear function
//! - **Threshold** - Basic threshold function: replaces values below threshold with a constant
//! - **Tanhshrink** - Tanh shrinkage function: `x - tanh(x)`
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
//! use torsh_nn::layers::activation::threshold::{Hardshrink, Softshrink, Hardtanh, Threshold};
//! use torsh_nn::Module;
//! use torsh_tensor::Tensor;
//!
//! // Create threshold and shrinking functions
//! let hardshrink = Hardshrink::new(0.5);
//! let softshrink = Softshrink::new(0.5);
//! let hardtanh = Hardtanh::new(-1.0, 1.0);
//! let threshold = Threshold::new(0.1, 0.0);
//!
//! // Apply to tensors
//! let input = randn(&[2, 3])?;
//! let hardshrink_output = hardshrink.forward(&input)?;
//! let softshrink_output = softshrink.forward(&input)?;
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

/// Hardshrink activation function
///
/// Applies the hard shrinkage function element-wise. This function zeros out values
/// whose absolute value is less than or equal to a threshold λ (lambda), and keeps
/// other values unchanged. It creates sparse representations by eliminating small values.
///
/// # Mathematical Definition
/// ```text
/// Hardshrink(x) = {
///     x    if |x| > λ
///     0    if |x| ≤ λ
/// }
/// ```
///
/// # Properties
/// - **Sparsity-inducing**: Creates sparse representations by zeroing small values
/// - **Piecewise**: Discontinuous function with sharp transitions
/// - **Configurable**: Threshold λ can be adjusted based on requirements
/// - **Symmetric**: Treats positive and negative values symmetrically
///
/// # Example
/// ```rust
/// # use torsh_tensor::creation::{randn, tensor_1d};
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_nn::layers::activation::threshold::Hardshrink;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let hardshrink = Hardshrink::new(0.5); // λ = 0.5
/// let input = tensor_1d(&[-1.0, -0.3, 0.0, 0.3, 1.0])?;
/// let output = hardshrink.forward(&input)?; // [-1.0, 0.0, 0.0, 0.0, 1.0]
/// # Ok(())
/// # }
/// # Ok(())
/// # }
/// ```
pub struct Hardshrink {
    base: ModuleBase,
    lambd: f32,
}

impl Hardshrink {
    /// Creates a new Hardshrink activation function
    ///
    /// # Arguments
    /// * `lambd` - The threshold parameter λ. Values with |x| ≤ λ are set to zero.
    pub fn new(lambd: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            lambd,
        }
    }

    /// Gets the current lambda (threshold) value
    pub fn lambda(&self) -> f32 {
        self.lambd
    }
}

impl Default for Hardshrink {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Module for Hardshrink {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply hardshrink: x if |x| > λ, else 0
        let abs_input = input.abs()?;
        let lambda_tensor = full(input.shape().dims(), self.lambd)?;
        let zeros_tensor = zeros(input.shape().dims())?;

        // Use where to create the mask directly
        let condition = abs_input.gt(&lambda_tensor)?;
        input.where_tensor(&condition, &zeros_tensor)
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

/// Softshrink activation function
///
/// Applies the soft shrinkage function element-wise. This function performs soft thresholding,
/// which shrinks values towards zero by a constant amount λ (lambda) when their absolute
/// value exceeds the threshold, and sets them to zero otherwise.
///
/// # Mathematical Definition
/// ```text
/// Softshrink(x) = {
///     x - λ    if x > λ
///     x + λ    if x < -λ
///     0        if |x| ≤ λ
/// }
/// ```
///
/// # Properties
/// - **Soft thresholding**: Smoothly shrinks values rather than hard cutoff
/// - **Sparsity-inducing**: Creates sparse representations by zeroing small values
/// - **Continuous**: Piecewise linear but continuous function
/// - **Symmetric**: Treats positive and negative values symmetrically
/// - **Regularization**: Often used for L1-like regularization effects
///
/// # Example
/// ```rust
/// # use torsh_tensor::creation::{randn, tensor_1d};
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_nn::layers::activation::threshold::Softshrink;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let softshrink = Softshrink::new(0.5); // λ = 0.5
/// let input = tensor_1d(&[-2.0, -0.3, 0.0, 0.3, 2.0])?;
/// let output = softshrink.forward(&input)?; // [-1.5, 0.0, 0.0, 0.0, 1.5]
/// # Ok(())
/// # }
/// # Ok(())
/// # }
/// ```
pub struct Softshrink {
    base: ModuleBase,
    lambd: f32,
}

impl Softshrink {
    /// Creates a new Softshrink activation function
    ///
    /// # Arguments
    /// * `lambd` - The shrinkage parameter λ. Values with |x| ≤ λ are set to zero,
    ///             others are shrunk by λ towards zero.
    pub fn new(lambd: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            lambd,
        }
    }

    /// Gets the current lambda (shrinkage) value
    pub fn lambda(&self) -> f32 {
        self.lambd
    }
}

impl Default for Softshrink {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Module for Softshrink {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply softshrink: sign(x) * max(|x| - λ, 0)
        let abs_input = input.abs()?;
        let lambda_tensor = full(input.shape().dims(), self.lambd)?;
        let zero_tensor = zeros(input.shape().dims())?;

        // Calculate |x| - λ
        let abs_minus_lambda = abs_input.sub(&lambda_tensor)?;

        // Apply max(|x| - λ, 0)
        let thresholded = abs_minus_lambda.maximum(&zero_tensor)?;

        // Apply sign(x) by using input.sign() or manual computation
        let sign_input = input.sign()?;
        thresholded.mul(&sign_input)
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

/// Hardtanh activation function
///
/// Applies the hard hyperbolic tangent function element-wise. This function clamps
/// input values to a specified range [min_val, max_val], creating a piecewise linear
/// function that is computationally efficient compared to the standard tanh function.
///
/// # Mathematical Definition
/// ```text
/// Hardtanh(x) = {
///     max_val    if x > max_val
///     x          if min_val ≤ x ≤ max_val
///     min_val    if x < min_val
/// }
/// ```
///
/// # Properties
/// - **Bounded**: Output is always within [min_val, max_val]
/// - **Piecewise linear**: Computationally efficient approximation to tanh
/// - **Configurable range**: Min and max values can be customized
/// - **Gradient preservation**: Has gradient 1 in the linear region
///
/// # Example
/// ```rust
/// # use torsh_tensor::creation::{randn, tensor_1d};
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_nn::layers::activation::threshold::Hardtanh;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let hardtanh = Hardtanh::new(-1.0, 1.0); // Clamp to [-1, 1]
/// let input = tensor_1d(&[-2.0, -0.5, 0.0, 0.5, 2.0])?;
/// let output = hardtanh.forward(&input)?; // [-1.0, -0.5, 0.0, 0.5, 1.0]
/// # Ok(())
/// # }
/// # Ok(())
/// # }
/// ```
pub struct Hardtanh {
    base: ModuleBase,
    min_val: f32,
    max_val: f32,
}

impl Hardtanh {
    /// Creates a new Hardtanh activation function
    ///
    /// # Arguments
    /// * `min_val` - The minimum value for clamping
    /// * `max_val` - The maximum value for clamping
    ///
    /// # Panics
    /// Panics if min_val >= max_val
    pub fn new(min_val: f32, max_val: f32) -> Self {
        assert!(min_val < max_val, "min_val must be less than max_val");
        Self {
            base: ModuleBase::new(),
            min_val,
            max_val,
        }
    }

    /// Creates a standard Hardtanh with range [-1, 1]
    pub fn standard() -> Self {
        Self::new(-1.0, 1.0)
    }

    /// Gets the minimum value
    pub fn min_val(&self) -> f32 {
        self.min_val
    }

    /// Gets the maximum value
    pub fn max_val(&self) -> f32 {
        self.max_val
    }
}

impl Default for Hardtanh {
    fn default() -> Self {
        Self::new(-1.0, 1.0)
    }
}

impl Module for Hardtanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply hardtanh: clamp(x, min_val, max_val)
        let min_tensor = full(input.shape().dims(), self.min_val)?;
        let max_tensor = full(input.shape().dims(), self.max_val)?;

        // Clamp to [min_val, max_val]
        let clamped_high = input.minimum(&max_tensor)?;
        clamped_high.maximum(&min_tensor)
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

/// Threshold activation function
///
/// Applies a basic threshold function element-wise. Values above the threshold
/// are kept unchanged, while values at or below the threshold are replaced
/// with a specified value (typically 0).
///
/// # Mathematical Definition
/// ```text
/// Threshold(x) = {
///     x        if x > threshold
///     value    if x ≤ threshold
/// }
/// ```
///
/// # Properties
/// - **Binary decision**: Simple threshold-based activation
/// - **Configurable**: Both threshold and replacement value can be set
/// - **Discontinuous**: Sharp transition at the threshold
/// - **Asymmetric**: Different behavior above and below threshold
///
/// # Example
/// ```rust
/// # use torsh_tensor::creation::{randn, tensor_1d};
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_nn::layers::activation::threshold::Threshold;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let threshold = Threshold::new(0.1, 0.0); // Threshold=0.1, replace with 0.0
/// let input = tensor_1d(&[-0.5, 0.05, 0.1, 0.2, 0.5])?;
/// let output = threshold.forward(&input)?; // [0.0, 0.0, 0.0, 0.2, 0.5]
/// # Ok(())
/// # }
/// # Ok(())
/// # }
/// ```
pub struct Threshold {
    base: ModuleBase,
    threshold: f32,
    value: f32,
}

impl Threshold {
    /// Creates a new Threshold activation function
    ///
    /// # Arguments
    /// * `threshold` - The threshold value for comparison
    /// * `value` - The value to replace inputs that are ≤ threshold
    pub fn new(threshold: f32, value: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            threshold,
            value,
        }
    }

    /// Creates a Threshold that zeros out values ≤ threshold
    pub fn zeroing(threshold: f32) -> Self {
        Self::new(threshold, 0.0)
    }

    /// Gets the threshold value
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Gets the replacement value
    pub fn value(&self) -> f32 {
        self.value
    }
}

impl Default for Threshold {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl Module for Threshold {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply threshold: x if x > threshold, else value
        let threshold_tensor = full(input.shape().dims(), self.threshold)?;
        let value_tensor = full(input.shape().dims(), self.value)?;

        // Apply: input if input > threshold, else value
        let condition = input.gt(&threshold_tensor)?;
        input.where_tensor(&condition, &value_tensor)
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

/// Tanhshrink activation function
///
/// Applies the tanh shrinkage function element-wise: `x - tanh(x)`.
/// This function shrinks the input by subtracting its hyperbolic tangent,
/// creating an activation that has interesting properties for certain applications.
///
/// # Mathematical Definition
/// ```text
/// Tanhshrink(x) = x - tanh(x)
/// ```
///
/// # Properties
/// - **Shrinkage**: Always reduces the magnitude of the input
/// - **Bounded shrinkage**: The shrinkage amount is bounded by tanh(x)
/// - **Zero-preserving**: f(0) = 0
/// - **Odd function**: f(-x) = -f(x)
/// - **Smooth**: Continuously differentiable everywhere
///
/// # Example
/// ```rust
/// # use torsh_tensor::creation::{randn, tensor_1d};
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_nn::layers::activation::threshold::Tanhshrink;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// let tanhshrink = Tanhshrink::new();
/// let input = tensor_1d(&[-2.0, -1.0, 0.0, 1.0, 2.0])?;
/// let output = tanhshrink.forward(&input)?;
/// // Output will be x - tanh(x) for each element
/// # Ok(())
/// # }
/// # Ok(())
/// # }
/// ```
pub struct Tanhshrink {
    base: ModuleBase,
}

impl Tanhshrink {
    /// Creates a new Tanhshrink activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for Tanhshrink {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Tanhshrink {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply tanhshrink: x - tanh(x)
        let tanh_result = input.tanh()?;
        input.sub(&tanh_result)
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
    fn test_hardshrink_parameters() {
        let hardshrink = Hardshrink::new(0.3);
        assert_eq!(hardshrink.lambda(), 0.3);

        let default_hardshrink = Hardshrink::default();
        assert_eq!(default_hardshrink.lambda(), 0.5);
    }

    #[test]
    fn test_hardshrink_forward() -> Result<()> {
        let hardshrink = Hardshrink::new(0.5);
        let input = randn(&[2, 3])?;
        let output = hardshrink.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_softshrink_parameters() {
        let softshrink = Softshrink::new(0.3);
        assert_eq!(softshrink.lambda(), 0.3);

        let default_softshrink = Softshrink::default();
        assert_eq!(default_softshrink.lambda(), 0.5);
    }

    #[test]
    fn test_softshrink_forward() -> Result<()> {
        let softshrink = Softshrink::new(0.5);
        let input = randn(&[2, 3])?;
        let output = softshrink.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_hardtanh_parameters() {
        let hardtanh = Hardtanh::new(-2.0, 3.0);
        assert_eq!(hardtanh.min_val(), -2.0);
        assert_eq!(hardtanh.max_val(), 3.0);

        let standard_hardtanh = Hardtanh::standard();
        assert_eq!(standard_hardtanh.min_val(), -1.0);
        assert_eq!(standard_hardtanh.max_val(), 1.0);
    }

    #[test]
    #[should_panic(expected = "min_val must be less than max_val")]
    fn test_hardtanh_invalid_range() {
        let _hardtanh = Hardtanh::new(1.0, -1.0); // Should panic
    }

    #[test]
    fn test_hardtanh_forward() -> Result<()> {
        let hardtanh = Hardtanh::new(-1.0, 1.0);
        let input = randn(&[2, 3])?;
        let output = hardtanh.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_threshold_parameters() {
        let threshold = Threshold::new(0.1, -1.0);
        assert_eq!(threshold.threshold(), 0.1);
        assert_eq!(threshold.value(), -1.0);

        let zeroing_threshold = Threshold::zeroing(0.5);
        assert_eq!(zeroing_threshold.threshold(), 0.5);
        assert_eq!(zeroing_threshold.value(), 0.0);
    }

    #[test]
    fn test_threshold_forward() -> Result<()> {
        let threshold = Threshold::new(0.0, -1.0);
        let input = randn(&[2, 3])?;
        let output = threshold.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_tanhshrink_forward() -> Result<()> {
        let tanhshrink = Tanhshrink::new();
        let input = randn(&[2, 3])?;
        let output = tanhshrink.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_training_mode_toggle() -> Result<()> {
        let mut hardshrink = Hardshrink::new(0.5);
        assert!(hardshrink.training());

        hardshrink.eval();
        assert!(!hardshrink.training());

        hardshrink.train();
        assert!(hardshrink.training());

        Ok(())
    }

    #[test]
    fn test_default_implementations() {
        let _hardshrink = Hardshrink::default();
        let _softshrink = Softshrink::default();
        let _hardtanh = Hardtanh::default();
        let _threshold = Threshold::default();
        let _tanhshrink = Tanhshrink::default();
    }

    #[test]
    fn test_convenience_constructors() {
        let standard_hardtanh = Hardtanh::standard();
        assert_eq!(standard_hardtanh.min_val(), -1.0);
        assert_eq!(standard_hardtanh.max_val(), 1.0);

        let zeroing_threshold = Threshold::zeroing(0.1);
        assert_eq!(zeroing_threshold.threshold(), 0.1);
        assert_eq!(zeroing_threshold.value(), 0.0);
    }
}
