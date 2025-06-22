//! Activation function layers

use crate::{Module, ModuleBase, Parameter};
use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// ReLU activation function
pub struct ReLU {
    base: ModuleBase,
}

impl ReLU {
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
        let zero = zeros(input.shape().dims());
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Sigmoid activation function
pub struct Sigmoid {
    base: ModuleBase,
}

impl Sigmoid {
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
        let one_plus_exp = exp_neg.add(&ones(input.shape().dims()))?;
        let one = ones(input.shape().dims());
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Tanh activation function
pub struct Tanh {
    base: ModuleBase,
}

impl Tanh {
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
        let exp_pos = input.exp()?;
        let neg_input = input.neg()?;
        let exp_neg = neg_input.exp()?;
        let numerator = exp_pos.sub(&exp_neg)?;
        let denominator = exp_pos.add(&exp_neg)?;
        numerator.div(&denominator)
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// GELU activation function
pub struct GELU {
    base: ModuleBase,
    approximate: bool,
}

impl GELU {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            approximate: false,
        }
    }

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
            let term = x_cubed.mul(&full(input.shape().dims(), 0.044715))?;
            let inner = input.add(&term)?;
            let scaled = inner.mul(&full(
                input.shape().dims(),
                (2.0 / std::f32::consts::PI).sqrt(),
            ))?;
            let tanh_result = scaled.tanh()?;
            let one_plus_tanh = tanh_result.add(&ones(input.shape().dims()))?;
            let half_x = input.mul(&full(input.shape().dims(), 0.5))?;
            half_x.mul(&one_plus_tanh)
        } else {
            // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            let sqrt_2 = (2.0_f32).sqrt();
            let x_div_sqrt2 = input.div(&full(input.shape().dims(), sqrt_2))?;
            let erf_result = x_div_sqrt2.erf()?;
            let one_plus_erf = erf_result.add(&ones(input.shape().dims()))?;
            let half_x = input.mul(&full(input.shape().dims(), 0.5))?;
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// LeakyReLU activation function
pub struct LeakyReLU {
    base: ModuleBase,
    negative_slope: f32,
}

impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            negative_slope,
        }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply LeakyReLU: max(0, x) + negative_slope * min(0, x)
        let zero = zeros(input.shape().dims());
        let positive_part = input.maximum(&zero)?;
        let negative_part = input.minimum(&zero)?;
        let scaled_negative =
            negative_part.mul(&full(input.shape().dims(), self.negative_slope))?;
        positive_part.add(&scaled_negative)
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Softmax activation function
pub struct Softmax {
    base: ModuleBase,
    dim: Option<usize>,
}

impl Softmax {
    pub fn new(dim: Option<usize>) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply softmax: exp(x) / sum(exp(x))
        let exp_input = input.exp()?;
        let sum_exp = if let Some(dim) = self.dim {
            exp_input.sum_dim(&[dim as i32], true)?
        } else {
            exp_input.sum()?
        };
        exp_input.div(&sum_exp)
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// LogSoftmax activation function
pub struct LogSoftmax {
    base: ModuleBase,
    dim: Option<usize>,
}

impl LogSoftmax {
    pub fn new(dim: Option<usize>) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }
}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Module for LogSoftmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply log_softmax: log(softmax(x)) = x - log(sum(exp(x)))
        let exp_input = input.exp()?;
        let sum_exp = if let Some(dim) = self.dim {
            exp_input.sum_dim(&[dim as i32], true)?
        } else {
            exp_input.sum()?
        };
        let log_sum_exp = sum_exp.log()?;
        input.sub(&log_sum_exp)
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}
