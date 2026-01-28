//! Basic Activation Functions
//!
//! This module contains traditional activation function families that form
//! the foundation of neural network training:
//! - ReLU family (ReLU, LeakyReLU, ReLU6, PReLU, ELU, SELU)
//! - Sigmoid family (Sigmoid, Hardsigmoid, LogSigmoid)
//! - Tanh family (Tanh, Hardtanh, Tanhshrink)
//! - Threshold-based (Threshold, Hardshrink, Softshrink)

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

// =============================================================================
// ReLU FAMILY ACTIVATIONS
// =============================================================================

/// Rectified Linear Unit (ReLU) activation function
///
/// Applies the element-wise function: ReLU(x) = max(0, x)
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::basic::ReLU;
/// use torsh_nn::Module;
///
/// let relu = ReLU::new();
/// let output = relu.forward(&input_tensor)?;
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

impl std::fmt::Debug for ReLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReLU").finish()
    }
}

/// Leaky Rectified Linear Unit (LeakyReLU) activation function
///
/// Applies the element-wise function:
/// LeakyReLU(x) = max(negative_slope * x, x)
///
/// # Parameters
/// - `negative_slope`: Controls the angle of the negative slope (default: 0.01)
pub struct LeakyReLU {
    base: ModuleBase,
    negative_slope: f32,
}

impl LeakyReLU {
    /// Creates a new LeakyReLU with the specified negative slope
    pub fn new(negative_slope: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            negative_slope,
        }
    }

    /// Creates a new LeakyReLU with default negative slope (0.01)
    pub fn default_slope() -> Self {
        Self::new(0.01)
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::default_slope()
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply LeakyReLU: max(negative_slope * x, x)
        let scaled = input.scalar_mul(self.negative_slope)?;
        input.maximum(&scaled)
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

impl std::fmt::Debug for LeakyReLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeakyReLU")
            .field("negative_slope", &self.negative_slope)
            .finish()
    }
}

/// ReLU6 activation function
///
/// Applies the element-wise function: ReLU6(x) = min(max(0, x), 6)
///
/// Commonly used in mobile networks for quantization-friendly activations.
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

impl std::fmt::Debug for ReLU6 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReLU6").finish()
    }
}

/// Parametric Rectified Linear Unit (PReLU) activation function
///
/// Applies the element-wise function:
/// PReLU(x) = max(a * x, x) where 'a' is a learnable parameter
///
/// # Parameters
/// - `num_parameters`: Number of learnable parameters (default: 1)
/// - `init`: Initial value for the parameters (default: 0.25)
pub struct PReLU {
    base: ModuleBase,
    num_parameters: usize,
    weight: Parameter,
}

impl PReLU {
    /// Creates a new PReLU with specified number of parameters
    pub fn new(num_parameters: usize, init: f32) -> Result<Self> {
        let weight_data = vec![init; num_parameters];
        let weight_tensor = Tensor::from_data(weight_data, vec![num_parameters], DeviceType::Cpu)?;

        Ok(Self {
            base: ModuleBase::new(),
            num_parameters,
            weight: Parameter::new(weight_tensor),
        })
    }

    /// Creates a new PReLU with default parameters (1 parameter, init=0.25)
    pub fn default_params() -> Result<Self> {
        Self::new(1, 0.25)
    }
}

impl Module for PReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply PReLU: max(weight * x, x)
        let weight_expanded = self.weight.tensor().broadcast_to(input.shape().dims())?;
        let scaled = input.mul(&weight_expanded)?;
        input.maximum(&scaled)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.base.parameters.clone();
        params.insert("weight".to_string(), self.weight.clone());
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
        self.weight.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.base.named_parameters();
        params.insert("weight".to_string(), self.weight.clone());
        params
    }
}

impl std::fmt::Debug for PReLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PReLU")
            .field("num_parameters", &self.num_parameters)
            .finish()
    }
}

/// Exponential Linear Unit (ELU) activation function
///
/// Applies the element-wise function:
/// ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
///
/// # Parameters
/// - `alpha`: The alpha value for the ELU formulation (default: 1.0)
pub struct ELU {
    base: ModuleBase,
    alpha: f32,
}

impl ELU {
    /// Creates a new ELU with the specified alpha
    pub fn new(alpha: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            alpha,
        }
    }

    /// Creates a new ELU with default alpha (1.0)
    pub fn default_alpha() -> Self {
        Self::new(1.0)
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::default_alpha()
    }
}

impl Module for ELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply ELU: max(0, x) + min(0, alpha * (exp(x) - 1))
        let zero = zeros(input.shape().dims())?;
        let one = ones(input.shape().dims())?;

        let exp_x = input.exp()?;
        let exp_minus_one = exp_x.sub(&one)?;
        let alpha_term = exp_minus_one.scalar_mul(self.alpha)?;

        let positive_part = input.maximum(&zero)?;
        let negative_part = alpha_term.minimum(&zero)?;

        positive_part.add(&negative_part)
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

impl std::fmt::Debug for ELU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ELU").field("alpha", &self.alpha).finish()
    }
}

/// Scaled Exponential Linear Unit (SELU) activation function
///
/// Applies the element-wise function:
/// SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
///
/// Self-normalizing neural networks use these specific constants.
pub struct SELU {
    base: ModuleBase,
}

impl SELU {
    /// Creates a new SELU activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }

    /// SELU constants from the paper
    const ALPHA: f32 = 1.6732632423543772848170429916717;
    const SCALE: f32 = 1.0507009873554804934193349852946;
}

impl Default for SELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
        let zero = zeros(input.shape().dims())?;
        let one = ones(input.shape().dims())?;

        let exp_x = input.exp()?;
        let exp_minus_one = exp_x.sub(&one)?;
        let alpha_term = exp_minus_one.scalar_mul(Self::ALPHA)?;

        let positive_part = input.maximum(&zero)?;
        let negative_part = alpha_term.minimum(&zero)?;

        let elu_result = positive_part.add(&negative_part)?;
        elu_result.scalar_mul(Self::SCALE)
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

impl std::fmt::Debug for SELU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SELU").finish()
    }
}

// =============================================================================
// SIGMOID FAMILY ACTIVATIONS
// =============================================================================

/// Sigmoid activation function
///
/// Applies the element-wise function: Sigmoid(x) = 1 / (1 + exp(-x))
///
/// Outputs values in the range (0, 1).
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
        // Apply sigmoid: 1 / (1 + exp(-x))
        let neg_input = input.neg()?;
        let exp_neg = neg_input.exp()?;
        let ones_tensor = ones(input.shape().dims())?;
        let one_plus_exp = exp_neg.add_op(&ones_tensor)?;
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

impl std::fmt::Debug for Sigmoid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sigmoid").finish()
    }
}

/// Hard Sigmoid activation function
///
/// Applies the element-wise function:
/// HardSigmoid(x) = max(0, min(1, (x + 3) / 6))
///
/// A piecewise linear approximation to the sigmoid function.
pub struct Hardsigmoid {
    base: ModuleBase,
}

impl Hardsigmoid {
    /// Creates a new HardSigmoid activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for Hardsigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Hardsigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply hard sigmoid: max(0, min(1, (x + 3) / 6))
        let three = full(input.shape().dims(), 3.0)?;
        let six = full(input.shape().dims(), 6.0)?;
        let zero = zeros(input.shape().dims())?;
        let one = ones(input.shape().dims())?;

        let x_plus_3 = input.add(&three)?;
        let divided = x_plus_3.div(&six)?;
        let clipped_upper = divided.minimum(&one)?;
        clipped_upper.maximum(&zero)
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

impl std::fmt::Debug for Hardsigmoid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hardsigmoid").finish()
    }
}

/// Log Sigmoid activation function
///
/// Applies the element-wise function: LogSigmoid(x) = log(sigmoid(x))
///
/// Numerically stable implementation using log-sum-exp trick.
pub struct LogSigmoid {
    base: ModuleBase,
}

impl LogSigmoid {
    /// Creates a new LogSigmoid activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for LogSigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LogSigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply log sigmoid: log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
        // For numerical stability: -max(0, -x) - log(1 + exp(-abs(x)))
        let abs_x = input.abs()?;
        let neg_abs_x = abs_x.neg()?;
        let exp_neg_abs = neg_abs_x.exp()?;
        let one = ones(input.shape().dims())?;
        let one_plus_exp = one.add(&exp_neg_abs)?;
        let log_term = one_plus_exp.log()?;

        let zero = zeros(input.shape().dims())?;
        let neg_input = input.neg()?;
        let max_term = zero.maximum(&neg_input)?;

        let result = max_term.add(&log_term)?;
        result.neg()
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

impl std::fmt::Debug for LogSigmoid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogSigmoid").finish()
    }
}

// =============================================================================
// TANH FAMILY ACTIVATIONS
// =============================================================================

/// Hyperbolic Tangent (Tanh) activation function
///
/// Applies the element-wise function: Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
///
/// Outputs values in the range (-1, 1).
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
        // Apply tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
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

impl std::fmt::Debug for Tanh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tanh").finish()
    }
}

/// Hard Tanh activation function
///
/// Applies the element-wise function:
/// HardTanh(x) = max(min_val, min(max_val, x))
///
/// # Parameters
/// - `min_val`: Minimum value (default: -1.0)
/// - `max_val`: Maximum value (default: 1.0)
pub struct Hardtanh {
    base: ModuleBase,
    min_val: f32,
    max_val: f32,
}

impl Hardtanh {
    /// Creates a new HardTanh with specified min and max values
    pub fn new(min_val: f32, max_val: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            min_val,
            max_val,
        }
    }

    /// Creates a new HardTanh with default values (-1.0, 1.0)
    pub fn default_range() -> Self {
        Self::new(-1.0, 1.0)
    }
}

impl Default for Hardtanh {
    fn default() -> Self {
        Self::default_range()
    }
}

impl Module for Hardtanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply hard tanh: max(min_val, min(max_val, x))
        let min_tensor = full(input.shape().dims(), self.min_val)?;
        let max_tensor = full(input.shape().dims(), self.max_val)?;

        let clipped_upper = input.minimum(&max_tensor)?;
        clipped_upper.maximum(&min_tensor)
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

impl std::fmt::Debug for Hardtanh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hardtanh")
            .field("min_val", &self.min_val)
            .field("max_val", &self.max_val)
            .finish()
    }
}

/// Tanh Shrink activation function
///
/// Applies the element-wise function: Tanhshrink(x) = x - tanh(x)
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
        let tanh_x = input.tanh()?;
        input.sub(&tanh_x)
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

impl std::fmt::Debug for Tanhshrink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tanhshrink").finish()
    }
}

// =============================================================================
// THRESHOLD-BASED ACTIVATIONS
// =============================================================================

/// Threshold activation function
///
/// Applies the element-wise function:
/// Threshold(x) = value if x > threshold else 0
///
/// # Parameters
/// - `threshold`: The threshold value (default: 1.0)
/// - `value`: The value to output when above threshold (default: 1.0)
pub struct Threshold {
    base: ModuleBase,
    threshold: f32,
    value: f32,
}

impl Threshold {
    /// Creates a new Threshold with specified threshold and value
    pub fn new(threshold: f32, value: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            threshold,
            value,
        }
    }

    /// Creates a new Threshold with default values (threshold=1.0, value=1.0)
    pub fn default_params() -> Self {
        Self::new(1.0, 1.0)
    }
}

impl Default for Threshold {
    fn default() -> Self {
        Self::default_params()
    }
}

impl Module for Threshold {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply threshold: value if x > threshold else 0
        let threshold_tensor = full(input.shape().dims(), self.threshold)?;
        let value_tensor = full(input.shape().dims(), self.value)?;
        let zero_tensor = zeros(input.shape().dims())?;

        // Create mask where input > threshold
        let mask = input.gt(&threshold_tensor)?;

        // Apply mask: where(mask, value, 0)
        mask.where_tensor(&value_tensor, &zero_tensor)
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

impl std::fmt::Debug for Threshold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Threshold")
            .field("threshold", &self.threshold)
            .field("value", &self.value)
            .finish()
    }
}

/// Hard Shrink activation function
///
/// Applies the element-wise function:
/// HardShrink(x) = x if |x| > lambda else 0
///
/// # Parameters
/// - `lambd`: The lambda value for the Hardshrink formulation (default: 0.5)
pub struct Hardshrink {
    base: ModuleBase,
    lambd: f32,
}

impl Hardshrink {
    /// Creates a new Hardshrink with specified lambda
    pub fn new(lambd: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            lambd,
        }
    }

    /// Creates a new Hardshrink with default lambda (0.5)
    pub fn default_lambda() -> Self {
        Self::new(0.5)
    }
}

impl Default for Hardshrink {
    fn default() -> Self {
        Self::default_lambda()
    }
}

impl Module for Hardshrink {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply hard shrink: x if |x| > lambda else 0
        let abs_input = input.abs()?;
        let lambda_tensor = full(input.shape().dims(), self.lambd)?;
        let zero_tensor = zeros(input.shape().dims())?;

        // Create mask where |x| > lambda
        let mask = abs_input.gt(&lambda_tensor)?;

        // Apply mask: where(mask, x, 0)
        mask.where_tensor(input, &zero_tensor)
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

impl std::fmt::Debug for Hardshrink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hardshrink")
            .field("lambd", &self.lambd)
            .finish()
    }
}

/// Soft Shrink activation function
///
/// Applies the element-wise function:
/// SoftShrink(x) = sign(x) * max(0, |x| - lambda)
///
/// # Parameters
/// - `lambd`: The lambda value for the Softshrink formulation (default: 0.5)
pub struct Softshrink {
    base: ModuleBase,
    lambd: f32,
}

impl Softshrink {
    /// Creates a new Softshrink with specified lambda
    pub fn new(lambd: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            lambd,
        }
    }

    /// Creates a new Softshrink with default lambda (0.5)
    pub fn default_lambda() -> Self {
        Self::new(0.5)
    }
}

impl Default for Softshrink {
    fn default() -> Self {
        Self::default_lambda()
    }
}

impl Module for Softshrink {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply soft shrink: sign(x) * max(0, |x| - lambda)
        let abs_input = input.abs()?;
        let lambda_tensor = full(input.shape().dims(), self.lambd)?;
        let zero_tensor = zeros(input.shape().dims())?;

        let abs_minus_lambda = abs_input.sub(&lambda_tensor)?;
        let shrunk_abs = abs_minus_lambda.maximum(&zero_tensor)?;

        // Get sign of input
        let sign_input = input.sign()?;

        sign_input.mul(&shrunk_abs)
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

impl std::fmt::Debug for Softshrink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Softshrink")
            .field("lambd", &self.lambd)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::*;

    #[test]
    fn test_relu_forward() {
        let relu = ReLU::new();
        let input = Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], vec![4], DeviceType::Cpu).unwrap();
        let output = relu.forward(&input).unwrap();
        let expected = vec![0.0, 0.0, 1.0, 2.0];

        assert_eq!(output.to_vec().expect("tensor to vec conversion should succeed"), expected);
    }

    #[test]
    fn test_leaky_relu_forward() {
        let leaky_relu = LeakyReLU::new(0.1);
        let input = Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], vec![4], DeviceType::Cpu).unwrap();
        let output = leaky_relu.forward(&input).unwrap();
        let expected = vec![-0.1, 0.0, 1.0, 2.0];

        for (actual, expected) in output.to_vec().expect("tensor to vec conversion should succeed").iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_relu6_forward() {
        let relu6 = ReLU6::new();
        let input = Tensor::from_data(vec![-1.0, 0.0, 3.0, 7.0], vec![4], DeviceType::Cpu).unwrap();
        let output = relu6.forward(&input).unwrap();
        let expected = vec![0.0, 0.0, 3.0, 6.0];

        assert_eq!(output.to_vec().expect("tensor to vec conversion should succeed"), expected);
    }

    #[test]
    fn test_sigmoid_forward() {
        let sigmoid = Sigmoid::new();
        let input = Tensor::from_data(vec![0.0], vec![1], DeviceType::Cpu).unwrap();
        let output = sigmoid.forward(&input).unwrap();

        // sigmoid(0) should be 0.5
        assert_relative_eq!(output.to_vec().expect("tensor to vec conversion should succeed")[0], 0.5, epsilon = 1e-5);
    }

    #[test]
    fn test_tanh_forward() {
        let tanh = Tanh::new();
        let input = Tensor::from_data(vec![0.0], vec![1], DeviceType::Cpu).unwrap();
        let output = tanh.forward(&input).unwrap();

        // tanh(0) should be 0.0
        assert_relative_eq!(output.to_vec().expect("tensor to vec conversion should succeed")[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hardtanh_forward() {
        let hardtanh = Hardtanh::new(-1.0, 1.0);
        let input = Tensor::from_data(vec![-2.0, 0.0, 2.0], vec![3], DeviceType::Cpu).unwrap();
        let output = hardtanh.forward(&input).unwrap();
        let expected = vec![-1.0, 0.0, 1.0];

        assert_eq!(output.to_vec().expect("tensor to vec conversion should succeed"), expected);
    }

    #[test]
    fn test_threshold_forward() {
        let threshold = Threshold::new(1.0, 10.0);
        let input = Tensor::from_data(vec![0.5, 1.0, 1.5], vec![3], DeviceType::Cpu).unwrap();
        let output = threshold.forward(&input).unwrap();
        let expected = vec![0.0, 0.0, 10.0];

        assert_eq!(output.to_vec().expect("tensor to vec conversion should succeed"), expected);
    }

    #[test]
    fn test_hardshrink_forward() {
        let hardshrink = Hardshrink::new(1.0);
        let input =
            Tensor::from_data(vec![-2.0, -0.5, 0.5, 2.0], vec![4], DeviceType::Cpu).unwrap();
        let output = hardshrink.forward(&input).unwrap();
        let expected = vec![-2.0, 0.0, 0.0, 2.0];

        assert_eq!(output.to_vec().expect("tensor to vec conversion should succeed"), expected);
    }

    #[test]
    fn test_module_interface() {
        let mut relu = ReLU::new();

        // Test training mode
        assert!(relu.training()); // Should be true by default
        relu.eval();
        assert!(!relu.training());
        relu.train();
        assert!(relu.training());

        // Test parameters (should be empty for ReLU)
        assert!(relu.parameters().is_empty());
        assert!(relu.named_parameters().is_empty());
    }

    #[test]
    fn test_prelu_parameters() {
        let prelu = PReLU::new(1, 0.25).unwrap();
        let params = prelu.parameters();

        assert_eq!(params.len(), 1);
        assert!(params.contains_key("weight"));
    }
}
