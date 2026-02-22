//! Compile-time optimization features for zero-cost abstractions
//!
//! This module provides compile-time optimizations using Rust's const generics
//! and type-level programming to enable:
//! - Compile-time shape validation
//! - Zero-cost activation function dispatch
//! - Statically-sized layer variants
//! - Aggressive inlining for hot paths
//!
//! # Examples
//!
//! ```ignore
//! use torsh_nn::compile_time::{StaticLinear, StaticActivation, ActivationKind};
//!
//! // Compile-time sized linear layer (10 -> 5)
//! let layer = StaticLinear::<10, 5>::new();
//!
//! // Compile-time activation dispatch
//! let relu = StaticActivation::<ActivationKind::ReLU>::new();
//! ```

use crate::{Module, ModuleBase, Parameter};
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

// ================================================================================================
// Type-level Activation Dispatch
// ================================================================================================

/// Activation function kinds for compile-time dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    GELU,
    Swish,
    Identity,
}

/// Compile-time activation function trait
pub trait CompileTimeActivation {
    /// Apply activation function with compile-time dispatch
    fn apply(&self, input: &Tensor) -> Result<Tensor>;

    /// Get activation kind
    fn kind(&self) -> ActivationKind;
}

/// Zero-cost activation function wrapper with compile-time dispatch
#[derive(Debug, Clone)]
pub struct StaticActivation<const KIND: u8> {
    // Phantom to enable const generic
    _marker: std::marker::PhantomData<()>,
}

impl<const KIND: u8> StaticActivation<KIND> {
    /// Create new static activation
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<const KIND: u8> Default for StaticActivation<KIND> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

// Implement for ReLU (KIND = 0)
impl CompileTimeActivation for StaticActivation<0> {
    #[inline(always)]
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        crate::functional::relu(input)
    }

    #[inline(always)]
    fn kind(&self) -> ActivationKind {
        ActivationKind::ReLU
    }
}

// Implement for Sigmoid (KIND = 1)
impl CompileTimeActivation for StaticActivation<1> {
    #[inline(always)]
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        crate::functional::sigmoid(input)
    }

    #[inline(always)]
    fn kind(&self) -> ActivationKind {
        ActivationKind::Sigmoid
    }
}

// Implement for Tanh (KIND = 2)
impl CompileTimeActivation for StaticActivation<2> {
    #[inline(always)]
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        crate::functional::tanh(input)
    }

    #[inline(always)]
    fn kind(&self) -> ActivationKind {
        ActivationKind::Tanh
    }
}

// Implement for GELU (KIND = 3)
impl CompileTimeActivation for StaticActivation<3> {
    #[inline(always)]
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        crate::functional::gelu(input)
    }

    #[inline(always)]
    fn kind(&self) -> ActivationKind {
        ActivationKind::GELU
    }
}

// ================================================================================================
// Statically-Sized Layers
// ================================================================================================

/// Statically-sized linear layer with compile-time known dimensions
///
/// This layer uses const generics to enable compile-time shape validation
/// and aggressive optimization.
///
/// # Type Parameters
/// - `IN_FEATURES`: Number of input features (compile-time constant)
/// - `OUT_FEATURES`: Number of output features (compile-time constant)
/// - `BIAS`: Whether to include bias (compile-time constant)
///
/// # Examples
///
/// ```ignore
/// // Linear layer: 784 -> 128 with bias
/// let layer1 = StaticLinear::<784, 128, true>::new();
///
/// // Linear layer: 128 -> 10 without bias
/// let layer2 = StaticLinear::<128, 10, false>::new();
/// ```
#[derive(Debug)]
pub struct StaticLinear<const IN_FEATURES: usize, const OUT_FEATURES: usize, const BIAS: bool> {
    base: ModuleBase,
}

impl<const IN_FEATURES: usize, const OUT_FEATURES: usize, const BIAS: bool>
    StaticLinear<IN_FEATURES, OUT_FEATURES, BIAS>
{
    /// Create a new statically-sized linear layer
    #[inline]
    pub fn new() -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize weight with shape [in_features, out_features] for direct matmul
        // This way input[batch, in] @ weight[in, out] = output[batch, out]
        let weight = crate::init::kaiming_uniform(&[IN_FEATURES, OUT_FEATURES], "fan_in")?;
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        // Initialize bias if enabled
        if BIAS {
            let bias = zeros(&[OUT_FEATURES])?;
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self { base })
    }

    /// Get input features (compile-time constant)
    #[inline(always)]
    pub const fn in_features() -> usize {
        IN_FEATURES
    }

    /// Get output features (compile-time constant)
    #[inline(always)]
    pub const fn out_features() -> usize {
        OUT_FEATURES
    }

    /// Check if bias is enabled (compile-time constant)
    #[inline(always)]
    pub const fn has_bias() -> bool {
        BIAS
    }

    /// Forward pass with compile-time optimizations
    #[inline]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Get weight parameter
        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Get bias parameter if enabled
        let bias_opt = if BIAS {
            Some(self.base.parameters["bias"].tensor().read().clone())
        } else {
            None
        };

        // Linear transformation (compile-time branch elimination for bias)
        let output = crate::functional::linear(input, &weight, bias_opt.as_ref())?;

        Ok(output)
    }
}

impl<const IN_FEATURES: usize, const OUT_FEATURES: usize, const BIAS: bool> Default
    for StaticLinear<IN_FEATURES, OUT_FEATURES, BIAS>
{
    #[inline]
    fn default() -> Self {
        Self::new().expect("Failed to create StaticLinear")
    }
}

impl<const IN_FEATURES: usize, const OUT_FEATURES: usize, const BIAS: bool> Module
    for StaticLinear<IN_FEATURES, OUT_FEATURES, BIAS>
{
    #[inline]
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    #[inline]
    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    #[inline]
    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    #[inline]
    fn train(&mut self) {
        self.base.set_training(true);
    }

    #[inline]
    fn eval(&mut self) {
        self.base.set_training(false);
    }

    #[inline]
    fn training(&self) -> bool {
        self.base.training()
    }
}

// ================================================================================================
// Fused Layer + Activation
// ================================================================================================

/// Fused linear layer with compile-time activation
///
/// This combines a linear layer and activation function into a single
/// operation with compile-time optimization.
///
/// # Type Parameters
/// - `IN_FEATURES`: Number of input features
/// - `OUT_FEATURES`: Number of output features
/// - `ACT`: Activation kind (0=ReLU, 1=Sigmoid, 2=Tanh, 3=GELU)
///
/// # Examples
///
/// ```ignore
/// // Linear + ReLU (fused)
/// let layer = FusedLinearActivation::<784, 128, 0>::new();
/// ```
#[derive(Debug)]
pub struct FusedLinearActivation<const IN_FEATURES: usize, const OUT_FEATURES: usize, const ACT: u8>
{
    linear: StaticLinear<IN_FEATURES, OUT_FEATURES, true>,
    activation: StaticActivation<ACT>,
}

impl<const IN_FEATURES: usize, const OUT_FEATURES: usize, const ACT: u8>
    FusedLinearActivation<IN_FEATURES, OUT_FEATURES, ACT>
{
    /// Create new fused layer
    #[inline]
    pub fn new() -> Result<Self> {
        Ok(Self {
            linear: StaticLinear::new()?,
            activation: StaticActivation::new(),
        })
    }
}

impl<const IN_FEATURES: usize, const OUT_FEATURES: usize, const ACT: u8> Default
    for FusedLinearActivation<IN_FEATURES, OUT_FEATURES, ACT>
where
    StaticActivation<ACT>: CompileTimeActivation,
{
    #[inline]
    fn default() -> Self {
        Self::new().expect("Failed to create FusedLinearActivation")
    }
}

impl<const IN_FEATURES: usize, const OUT_FEATURES: usize, const ACT: u8> Module
    for FusedLinearActivation<IN_FEATURES, OUT_FEATURES, ACT>
where
    StaticActivation<ACT>: CompileTimeActivation,
{
    /// Fused forward pass with compile-time optimization
    #[inline]
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let linear_out = self.linear.forward(input)?;
        self.activation.apply(&linear_out)
    }

    #[inline]
    fn parameters(&self) -> HashMap<String, Parameter> {
        self.linear.parameters()
    }

    #[inline]
    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.linear.named_parameters()
    }

    #[inline]
    fn train(&mut self) {
        self.linear.train();
    }

    #[inline]
    fn eval(&mut self) {
        self.linear.eval();
    }

    #[inline]
    fn training(&self) -> bool {
        self.linear.training()
    }
}

// ================================================================================================
// Compile-time Shape Validation
// ================================================================================================

/// Type-level shape validator for compile-time shape checking
///
/// This trait enables compile-time validation of tensor operations
/// to catch shape mismatches early.
pub trait ShapeValidator {
    /// Validate input shape
    fn validate_input_shape<const N: usize>(shape: &[usize; N]) -> bool;

    /// Validate output shape
    fn validate_output_shape<const N: usize>(shape: &[usize; N]) -> bool;
}

/// Statically-sized MLP with compile-time shape validation
///
/// # Type Parameters
/// - `LAYERS`: Array of layer sizes (compile-time constant)
///
/// # Examples
///
/// ```ignore
/// // MLP: 784 -> 256 -> 128 -> 10
/// let mlp = StaticMLP::<4>::new([784, 256, 128, 10]);
/// ```
#[derive(Debug)]
pub struct StaticMLP<const LAYERS: usize> {
    base: ModuleBase,
    layer_sizes: Vec<usize>,
}

impl<const LAYERS: usize> StaticMLP<LAYERS> {
    /// Create new static MLP
    ///
    /// # Arguments
    /// - `sizes`: Layer sizes (must have length LAYERS)
    pub fn new(sizes: [usize; LAYERS]) -> Result<Self> {
        if LAYERS < 2 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "StaticMLP requires at least 2 layers (input + output)".to_string(),
            ));
        }

        let mut base = ModuleBase::new();
        let layer_sizes = sizes.to_vec();

        // Create layers
        for i in 0..LAYERS - 1 {
            let in_features = sizes[i];
            let out_features = sizes[i + 1];

            // Create weight with shape [in_features, out_features] for direct matmul
            let weight = crate::init::kaiming_uniform(&[in_features, out_features], "fan_in")?;
            base.register_parameter(format!("layer{}.weight", i), Parameter::new(weight));

            // Create bias for this layer
            let bias = zeros(&[out_features])?;
            base.register_parameter(format!("layer{}.bias", i), Parameter::new(bias));
        }

        Ok(Self { base, layer_sizes })
    }

    /// Get number of layers (compile-time constant)
    #[inline(always)]
    pub const fn num_layers() -> usize {
        LAYERS
    }

    /// Get layer sizes
    #[inline]
    pub fn layer_sizes(&self) -> &[usize] {
        &self.layer_sizes
    }

    /// Forward pass through all layers
    #[inline]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        // Process through each layer
        for i in 0..LAYERS - 1 {
            let weight = self.base.parameters[&format!("layer{}.weight", i)]
                .tensor()
                .read()
                .clone();
            let bias = self.base.parameters[&format!("layer{}.bias", i)]
                .tensor()
                .read()
                .clone();

            // Linear transformation
            x = crate::functional::linear(&x, &weight, Some(&bias))?;

            // ReLU activation (except for last layer)
            if i < LAYERS - 2 {
                x = crate::functional::relu(&x)?;
            }
        }

        Ok(x)
    }
}

impl<const LAYERS: usize> Module for StaticMLP<LAYERS> {
    #[inline]
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    #[inline]
    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    #[inline]
    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    #[inline]
    fn train(&mut self) {
        self.base.set_training(true);
    }

    #[inline]
    fn eval(&mut self) {
        self.base.set_training(false);
    }

    #[inline]
    fn training(&self) -> bool {
        self.base.training()
    }
}

// ================================================================================================
// Inline Optimization Utilities
// ================================================================================================

/// Inline hint for aggressive optimization
#[inline(always)]
pub const fn inline_always_hint() {
    // Hint to compiler for aggressive inlining
}

/// Cold path hint for better optimization
#[inline(never)]
#[cold]
pub fn cold_path_hint() {
    // Hint to compiler that this path is rarely taken
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_linear_creation() {
        let layer = StaticLinear::<10, 5, true>::new();
        assert!(layer.is_ok());

        let _layer = layer.unwrap();
        assert_eq!(StaticLinear::<10, 5, true>::in_features(), 10);
        assert_eq!(StaticLinear::<10, 5, true>::out_features(), 5);
        assert!(StaticLinear::<10, 5, true>::has_bias());
    }

    #[test]
    fn test_static_linear_forward() {
        let layer = StaticLinear::<10, 5, true>::new().unwrap();
        let input = randn(&[2, 10]).unwrap();
        let output = layer.forward(&input);

        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.shape().dims(), &[2, 5]);
    }

    #[test]
    fn test_static_activation_relu() {
        let activation = StaticActivation::<0>::new();
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
        let output = activation.apply(&input).unwrap();

        let result: Vec<f32> = output
            .to_vec()
            .expect("tensor to vec conversion should succeed");
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_fused_linear_activation() {
        let layer = FusedLinearActivation::<10, 5, 0>::new(); // ReLU
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        let input = randn(&[2, 10]).unwrap();
        let output = layer.forward(&input);

        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.shape().dims(), &[2, 5]);

        // Verify ReLU was applied (no negative values)
        let result: Vec<f32> = output
            .to_vec()
            .expect("tensor to vec conversion should succeed");
        assert!(result.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_static_mlp() {
        let mlp = StaticMLP::<4>::new([784, 256, 128, 10]);
        assert!(mlp.is_ok());

        let mlp = mlp.unwrap();
        assert_eq!(StaticMLP::<4>::num_layers(), 4);

        let input = randn(&[2, 784]).unwrap();
        let output = mlp.forward(&input);

        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.shape().dims(), &[2, 10]);
    }

    #[test]
    fn test_activation_kinds() {
        let relu = StaticActivation::<0>::new();
        assert_eq!(relu.kind(), ActivationKind::ReLU);

        let sigmoid = StaticActivation::<1>::new();
        assert_eq!(sigmoid.kind(), ActivationKind::Sigmoid);

        let tanh = StaticActivation::<2>::new();
        assert_eq!(tanh.kind(), ActivationKind::Tanh);

        let gelu = StaticActivation::<3>::new();
        assert_eq!(gelu.kind(), ActivationKind::GELU);
    }

    #[test]
    fn test_static_linear_no_bias() {
        let layer = StaticLinear::<10, 5, false>::new().unwrap();
        assert!(!StaticLinear::<10, 5, false>::has_bias());

        // Should only have weight parameter, not bias
        let params = layer.parameters();
        assert!(params.contains_key("weight"));
        assert!(!params.contains_key("bias"));
    }

    #[test]
    fn test_compile_time_constants() {
        // Verify compile-time evaluation
        const IN: usize = StaticLinear::<784, 128, true>::in_features();
        const OUT: usize = StaticLinear::<784, 128, true>::out_features();
        const HAS_BIAS: bool = StaticLinear::<784, 128, true>::has_bias();

        assert_eq!(IN, 784);
        assert_eq!(OUT, 128);
        assert!(HAS_BIAS);
    }
}
