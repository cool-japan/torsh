//! Advanced and Modern Activation Functions
//!
//! This module contains state-of-the-art activation functions that have been
//! developed in recent years for improved neural network performance:
//! - Modern activations (GELU, SiLU/Swish, Mish, Hardswish)
//! - Gated Linear Units (GLU, GEGLU, ReGLU, SwiGLU)
//!
//! These activations are commonly used in modern architectures like Transformers,
//! Vision Transformers, and efficient mobile networks.

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

// Import basic activations for use in gated units
use super::basic::{ReLU, Sigmoid};
use super::normalization::Softplus;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{collections::HashMap, string::String};

#[cfg(not(feature = "std"))]
use alloc::string::String;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

// =============================================================================
// MODERN ACTIVATION FUNCTIONS
// =============================================================================

/// Gaussian Error Linear Unit (GELU) activation function
///
/// GELU is defined as: GELU(x) = 0.5 * x * (1 + erf(x / √2))
///
/// An approximation is also available: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// GELU is widely used in transformer architectures and has shown superior
/// performance compared to ReLU in many scenarios.
///
/// # Parameters
/// - `approximate`: Whether to use the tanh approximation (default: false)
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::advanced::GELU;
/// use torsh_nn::Module;
///
/// // Exact GELU
/// let gelu = GELU::new();
/// let output = gelu.forward(&input_tensor)?;
///
/// // Approximate GELU (faster computation)
/// let gelu_approx = GELU::with_approximate(true);
/// let output_approx = gelu_approx.forward(&input_tensor)?;
/// ```
pub struct GELU {
    base: ModuleBase,
    approximate: bool,
}

impl GELU {
    /// Creates a new GELU activation with exact computation
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            approximate: false,
        }
    }

    /// Creates a new GELU activation with specified approximation mode
    pub fn with_approximate(approximate: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            approximate,
        }
    }

    /// Creates a new GELU activation with tanh approximation (faster)
    pub fn approximate() -> Self {
        Self::with_approximate(true)
    }

    /// Creates a new GELU activation with exact computation (more accurate)
    pub fn exact() -> Self {
        Self::with_approximate(false)
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
            // Approximate GELU: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            let x_cubed = input.pow(3.0)?;
            let x_cubed_scaled = x_cubed.scalar_mul(0.044715)?;
            let inner_term = input.add(&x_cubed_scaled)?;

            let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
            let scaled_term = inner_term.scalar_mul(sqrt_2_over_pi)?;
            let tanh_term = scaled_term.tanh()?;

            let one_plus_tanh = tanh_term.add(&ones(input.shape().dims())?)?;
            let half_x = input.scalar_mul(0.5)?;

            half_x.mul(&one_plus_tanh)
        } else {
            // Exact GELU: 0.5 * x * (1 + erf(x / √2))
            let sqrt_2 = (2.0_f32).sqrt();
            let x_div_sqrt2 = input.scalar_mul(1.0 / sqrt_2)?;
            let erf_result = x_div_sqrt2.erf()?;
            let one_plus_erf = erf_result.add(&ones(input.shape().dims())?)?;
            let half_x = input.scalar_mul(0.5)?;

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

impl std::fmt::Debug for GELU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GELU")
            .field("approximate", &self.approximate)
            .finish()
    }
}

/// Sigmoid Linear Unit (SiLU) activation function
///
/// Also known as Swish activation function.
/// SiLU(x) = x * sigmoid(x)
///
/// SiLU has been shown to outperform ReLU in many deep networks and is
/// particularly effective in neural architecture search applications.
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::advanced::{SiLU, Swish};
/// use torsh_nn::Module;
///
/// let silu = SiLU::new();
/// let output = silu.forward(&input_tensor)?;
///
/// // Swish is an alias for SiLU
/// let swish = Swish::new();
/// let output_swish = swish.forward(&input_tensor)?;
/// ```
pub struct SiLU {
    base: ModuleBase,
}

/// Type alias for Swish activation (same as SiLU)
pub type Swish = SiLU;

impl SiLU {
    /// Creates a new SiLU/Swish activation function
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
        let sigmoid_result = input.sigmoid()?;
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

impl std::fmt::Debug for SiLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SiLU").finish()
    }
}

/// Mish activation function
///
/// Mish(x) = x * tanh(softplus(x))
///
/// Mish is a smooth, non-monotonic activation function that has shown
/// competitive performance with Swish while being computationally efficient.
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::advanced::Mish;
/// use torsh_nn::Module;
///
/// let mish = Mish::new();
/// let output = mish.forward(&input_tensor)?;
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
        // softplus(x) = ln(1 + exp(x))
        let exp_x = input.exp()?;
        let one_plus_exp = ones(input.shape().dims())?.add(&exp_x)?;
        let softplus_result = one_plus_exp.log()?;
        let tanh_result = softplus_result.tanh()?;

        input.mul(&tanh_result)
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

impl std::fmt::Debug for Mish {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mish").finish()
    }
}

/// Hard Swish activation function
///
/// Hardswish(x) = x * hardsigmoid(x) = x * max(0, min(1, (x + 3) / 6))
///
/// Hard Swish is a computationally efficient approximation to Swish that
/// uses only add, multiply, and ReLU operations. It's widely used in
/// mobile-optimized neural networks like MobileNetV3.
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::advanced::Hardswish;
/// use torsh_nn::Module;
///
/// let hardswish = Hardswish::new();
/// let output = hardswish.forward(&input_tensor)?;
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
        // Apply hardswish: x * hardsigmoid(x)
        // hardsigmoid(x) = max(0, min(1, (x + 3) / 6))
        let three = full(input.shape().dims(), 3.0)?;
        let six = full(input.shape().dims(), 6.0)?;
        let zero = zeros(input.shape().dims())?;
        let one = ones(input.shape().dims())?;

        let x_plus_3 = input.add(&three)?;
        let divided = x_plus_3.scalar_mul(1.0 / 6.0)?;
        let clipped_upper = divided.minimum(&one)?;
        let hardsigmoid = clipped_upper.maximum(&zero)?;

        input.mul(&hardsigmoid)
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

impl std::fmt::Debug for Hardswish {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hardswish").finish()
    }
}

// =============================================================================
// GATED LINEAR UNITS
// =============================================================================

/// Gated Linear Unit (GLU) activation function
///
/// GLU splits the input into two halves along the specified dimension,
/// applies sigmoid to one half (gate), and element-wise multiplies with the other half.
///
/// Input shape: [..., 2*d] -> Output shape: [..., d]
/// Formula: GLU(x) = x₁ ⊙ σ(x₂), where x is split into x₁ and x₂
///
/// # Parameters
/// - `dim`: Dimension along which to split the input (default: -1, last dimension)
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::advanced::GLU;
/// use torsh_nn::Module;
///
/// let glu = GLU::new(-1); // Split along last dimension
/// let output = glu.forward(&input_tensor)?; // Input: [batch, 2*hidden] -> [batch, hidden]
/// ```
pub struct GLU {
    base: ModuleBase,
    dim: isize,
}

impl GLU {
    /// Creates a new GLU activation function
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to split the input (negative indexing supported)
    pub fn new(dim: isize) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }

    /// Creates GLU that splits along the last dimension
    pub fn last_dim() -> Self {
        Self::new(-1)
    }

    /// Creates GLU that splits along the channel dimension (index 1)
    pub fn channel_dim() -> Self {
        Self::new(1)
    }
}

impl Default for GLU {
    fn default() -> Self {
        Self::last_dim()
    }
}

impl Module for GLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape().dims();
        let split_dim = if self.dim < 0 {
            (input_shape.len() as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        if split_dim >= input_shape.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                self.dim,
                input_shape.len()
            )));
        }

        let split_size = input_shape[split_dim];
        if split_size % 2 != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Input dimension {} must be even for GLU, got {}",
                split_size, split_size
            )));
        }

        // Split input into two halves along the specified dimension
        let chunks = input.chunk(2, split_dim as i32)?;
        if chunks.len() != 2 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Failed to split input into two chunks".to_string(),
            ));
        }

        let value = &chunks[0];
        let gate = &chunks[1];

        // Apply sigmoid to the gate and multiply with the value
        let sigmoid_gate = gate.sigmoid()?;
        value.mul(&sigmoid_gate)
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

impl std::fmt::Debug for GLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLU").field("dim", &self.dim).finish()
    }
}

/// GELU-Gated Linear Unit (GEGLU) activation function
///
/// GEGLU is a variant of GLU that uses GELU activation instead of sigmoid for gating.
/// It has shown superior performance in transformer feed-forward networks.
///
/// Input shape: [..., 2*d] -> Output shape: [..., d]
/// Formula: GEGLU(x) = x₁ ⊙ GELU(x₂), where x is split into x₁ and x₂
///
/// # Parameters
/// - `dim`: Dimension along which to split the input (default: -1, last dimension)
/// - `approximate_gelu`: Whether to use approximate GELU (default: false)
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::advanced::GEGLU;
/// use torsh_nn::Module;
///
/// let geglu = GEGLU::new(-1, false); // Exact GELU
/// let output = geglu.forward(&input_tensor)?;
/// ```
pub struct GEGLU {
    base: ModuleBase,
    dim: isize,
    approximate_gelu: bool,
}

impl GEGLU {
    /// Creates a new GEGLU activation function
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to split the input
    /// * `approximate_gelu` - Whether to use approximate GELU computation
    pub fn new(dim: isize, approximate_gelu: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
            approximate_gelu,
        }
    }

    /// Creates GEGLU with exact GELU on the last dimension
    pub fn exact() -> Self {
        Self::new(-1, false)
    }

    /// Creates GEGLU with approximate GELU on the last dimension
    pub fn approximate() -> Self {
        Self::new(-1, true)
    }
}

impl Default for GEGLU {
    fn default() -> Self {
        Self::exact()
    }
}

impl Module for GEGLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape().dims();
        let split_dim = if self.dim < 0 {
            (input_shape.len() as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        if split_dim >= input_shape.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                self.dim,
                input_shape.len()
            )));
        }

        let split_size = input_shape[split_dim];
        if split_size % 2 != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Input dimension {} must be even for GEGLU, got {}",
                split_size, split_size
            )));
        }

        // Split input into two halves along the specified dimension
        let chunks = input.chunk(2, split_dim as i32)?;
        if chunks.len() != 2 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Failed to split input into two chunks".to_string(),
            ));
        }

        let value = &chunks[0];
        let gate = &chunks[1];

        // Apply GELU to the gate
        let gelu = GELU::with_approximate(self.approximate_gelu);
        let gelu_gate = gelu.forward(gate)?;

        // Element-wise multiplication of value and gated activation
        value.mul(&gelu_gate)
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

impl std::fmt::Debug for GEGLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GEGLU")
            .field("dim", &self.dim)
            .field("approximate_gelu", &self.approximate_gelu)
            .finish()
    }
}

/// ReLU-Gated Linear Unit (ReGLU) activation function
///
/// ReGLU is a variant of GLU that uses ReLU activation for gating.
///
/// Input shape: [..., 2*d] -> Output shape: [..., d]
/// Formula: ReGLU(x) = x₁ ⊙ ReLU(x₂), where x is split into x₁ and x₂
///
/// # Parameters
/// - `dim`: Dimension along which to split the input (default: -1, last dimension)
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::advanced::ReGLU;
/// use torsh_nn::Module;
///
/// let reglu = ReGLU::new(-1);
/// let output = reglu.forward(&input_tensor)?;
/// ```
pub struct ReGLU {
    base: ModuleBase,
    dim: isize,
}

impl ReGLU {
    /// Creates a new ReGLU activation function
    pub fn new(dim: isize) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }

    /// Creates ReGLU that splits along the last dimension
    pub fn last_dim() -> Self {
        Self::new(-1)
    }
}

impl Default for ReGLU {
    fn default() -> Self {
        Self::last_dim()
    }
}

impl Module for ReGLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape().dims();
        let split_dim = if self.dim < 0 {
            (input_shape.len() as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        if split_dim >= input_shape.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                self.dim,
                input_shape.len()
            )));
        }

        let split_size = input_shape[split_dim];
        if split_size % 2 != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Input dimension {} must be even for ReGLU, got {}",
                split_size, split_size
            )));
        }

        // Split input into two halves along the specified dimension
        let chunks = input.chunk(2, split_dim as i32)?;
        if chunks.len() != 2 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Failed to split input into two chunks".to_string(),
            ));
        }

        let value = &chunks[0];
        let gate = &chunks[1];

        // Apply ReLU to the gate
        let relu = ReLU::new();
        let relu_gate = relu.forward(gate)?;

        // Element-wise multiplication of value and gated activation
        value.mul(&relu_gate)
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

impl std::fmt::Debug for ReGLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReGLU").field("dim", &self.dim).finish()
    }
}

/// Swish-Gated Linear Unit (SwiGLU) activation function
///
/// SwiGLU is a variant of GLU that uses SiLU/Swish activation for gating.
/// It has shown excellent performance in large language models.
///
/// Input shape: [..., 2*d] -> Output shape: [..., d]
/// Formula: SwiGLU(x) = x₁ ⊙ SiLU(x₂), where x is split into x₁ and x₂
///
/// # Parameters
/// - `dim`: Dimension along which to split the input (default: -1, last dimension)
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::advanced::SwiGLU;
/// use torsh_nn::Module;
///
/// let swiglu = SwiGLU::new(-1);
/// let output = swiglu.forward(&input_tensor)?;
/// ```
pub struct SwiGLU {
    base: ModuleBase,
    dim: isize,
}

impl SwiGLU {
    /// Creates a new SwiGLU activation function
    pub fn new(dim: isize) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }

    /// Creates SwiGLU that splits along the last dimension
    pub fn last_dim() -> Self {
        Self::new(-1)
    }
}

impl Default for SwiGLU {
    fn default() -> Self {
        Self::last_dim()
    }
}

impl Module for SwiGLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape().dims();
        let split_dim = if self.dim < 0 {
            (input_shape.len() as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        if split_dim >= input_shape.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                self.dim,
                input_shape.len()
            )));
        }

        let split_size = input_shape[split_dim];
        if split_size % 2 != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Input dimension {} must be even for SwiGLU, got {}",
                split_size, split_size
            )));
        }

        // Split input into two halves along the specified dimension
        let chunks = input.chunk(2, split_dim as i32)?;
        if chunks.len() != 2 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Failed to split input into two chunks".to_string(),
            ));
        }

        let value = &chunks[0];
        let gate = &chunks[1];

        // Apply SiLU (Swish) to the gate
        let silu = SiLU::new();
        let silu_gate = silu.forward(gate)?;

        // Element-wise multiplication of value and gated activation
        value.mul(&silu_gate)
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

impl std::fmt::Debug for SwiGLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwiGLU").field("dim", &self.dim).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::*;

    #[test]
    fn test_gelu_forward() {
        let gelu = GELU::new();
        let input = Tensor::from_data(vec![0.0], vec![1], DeviceType::Cpu).unwrap();
        let output = gelu.forward(&input).unwrap();

        // GELU(0) should be 0
        assert_relative_eq!(output.to_vec().unwrap()[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_gelu_approximate_vs_exact() {
        let input = Tensor::from_data(vec![1.0], vec![1], DeviceType::Cpu).unwrap();

        let gelu_exact = GELU::exact();
        let gelu_approx = GELU::approximate();

        let output_exact = gelu_exact.forward(&input).unwrap();
        let output_approx = gelu_approx.forward(&input).unwrap();

        // They should be close but not identical
        let diff = (output_exact.to_vec().unwrap()[0] - output_approx.to_vec().unwrap()[0]).abs();
        assert!(diff < 0.1); // Should be reasonably close
    }

    #[test]
    fn test_silu_forward() {
        let silu = SiLU::new();
        let input = Tensor::from_data(vec![0.0, 1.0], vec![2], DeviceType::Cpu).unwrap();
        let output = silu.forward(&input).unwrap();
        let output_vec = output.to_vec().unwrap();

        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert_relative_eq!(output_vec[0], 0.0, epsilon = 1e-5);

        // SiLU(1) = 1 * sigmoid(1) ≈ 1 * 0.7311 ≈ 0.7311
        assert!(output_vec[1] > 0.5 && output_vec[1] < 1.0);
    }

    #[test]
    fn test_mish_forward() {
        let mish = Mish::new();
        let input = Tensor::from_data(vec![0.0, 1.0], vec![2], DeviceType::Cpu).unwrap();
        let output = mish.forward(&input).unwrap();
        let output_vec = output.to_vec().unwrap();

        // Mish(0) should be approximately 0
        assert_relative_eq!(output_vec[0], 0.0, epsilon = 1e-2);

        // Mish(1) should be positive and close to 1
        assert!(output_vec[1] > 0.5 && output_vec[1] < 1.5);
    }

    #[test]
    fn test_hardswish_forward() {
        let hardswish = Hardswish::new();
        let input = Tensor::from_data(vec![-3.0, 0.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let output = hardswish.forward(&input).unwrap();
        let output_vec = output.to_vec().unwrap();

        // Hardswish(-3) should be 0 (since hardsigmoid(-3) = 0)
        assert_relative_eq!(output_vec[0], 0.0, epsilon = 1e-5);

        // Hardswish(0) = 0 * hardsigmoid(0) = 0
        assert_relative_eq!(output_vec[1], 0.0, epsilon = 1e-5);

        // Hardswish(3) = 3 * hardsigmoid(3) = 3 * 1 = 3
        assert_relative_eq!(output_vec[2], 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_glu_forward() {
        // Input shape: [2, 4] -> Output shape: [2, 2]
        let input = Tensor::from_data(
            vec![
                1.0, 2.0, 0.0, 1.0, // First row
                3.0, 4.0, -1.0, 2.0,
            ], // Second row
            vec![2, 4],
            DeviceType::Cpu,
        )
        .unwrap();

        let glu = GLU::new(-1); // Split along last dimension
        let output = glu.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[2, 2]);

        // GLU splits [1,2,0,1] into [1,2] and [0,1], then [1,2] * sigmoid([0,1])
        // GLU splits [3,4,-1,2] into [3,4] and [-1,2], then [3,4] * sigmoid([-1,2])
        let output_vec = output.to_vec().unwrap();
        assert_eq!(output_vec.len(), 4);
    }

    #[test]
    fn test_glu_invalid_dimension() {
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let glu = GLU::new(-1);

        // Should fail because dimension size is odd (3)
        assert!(glu.forward(&input).is_err());
    }

    #[test]
    fn test_geglu_forward() {
        let input = Tensor::from_data(vec![1.0, 2.0, 0.0, 1.0], vec![4], DeviceType::Cpu).unwrap();

        let geglu = GEGLU::exact();
        let output = geglu.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[2]);
    }

    #[test]
    fn test_reglu_forward() {
        let input = Tensor::from_data(vec![1.0, 2.0, -1.0, 1.0], vec![4], DeviceType::Cpu).unwrap();

        let reglu = ReGLU::new(-1);
        let output = reglu.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[2]);

        // ReGLU([1,2,-1,1]) -> [1,2] * ReLU([-1,1]) = [1,2] * [0,1] = [0,2]
        let output_vec = output.to_vec().unwrap();
        assert_relative_eq!(output_vec[0], 0.0, epsilon = 1e-5); // 1 * ReLU(-1) = 1 * 0 = 0
        assert_relative_eq!(output_vec[1], 2.0, epsilon = 1e-5); // 2 * ReLU(1) = 2 * 1 = 2
    }

    #[test]
    fn test_swiglu_forward() {
        let input = Tensor::from_data(vec![1.0, 2.0, 0.0, 1.0], vec![4], DeviceType::Cpu).unwrap();

        let swiglu = SwiGLU::new(-1);
        let output = swiglu.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[2]);
    }

    #[test]
    fn test_module_interface() {
        let mut gelu = GELU::new();

        // Test training mode
        assert!(gelu.training()); // Should be true by default
        gelu.eval();
        assert!(!gelu.training());
        gelu.train();
        assert!(gelu.training());

        // Test parameters (should be empty for activation functions)
        assert!(gelu.parameters().is_empty());
        assert!(gelu.named_parameters().is_empty());
    }

    #[test]
    fn test_swish_alias() {
        let silu = SiLU::new();
        let swish = Swish::new();

        let input = Tensor::from_data(vec![1.0], vec![1], DeviceType::Cpu).unwrap();

        let silu_output = silu.forward(&input).unwrap();
        let swish_output = swish.forward(&input).unwrap();

        // They should be identical
        assert_eq!(
            silu_output.to_vec().unwrap(),
            swish_output.to_vec().unwrap()
        );
    }
}
