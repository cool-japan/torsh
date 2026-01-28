//! Neural Network Activation Functions
//!
//! This module provides a comprehensive suite of activation functions organized
//! into logical categories for improved maintainability and discoverability.
//!
//! ## Architecture
//!
//! The activation functions are organized into three specialized modules:
//! - `basic` - Traditional activation function families (ReLU, Sigmoid, Tanh, threshold-based)
//! - `normalization` - Normalization and smooth activations (Softmax, LogSoftmax, Softplus, Softsign)
//! - `advanced` - Modern and gated activations (GELU, SiLU, Mish, GLU variants)
//!
//! ## Quick Start
//!
//! ```rust
//! use torsh_nn::layers::activations::{ReLU, GELU, Softmax, SwiGLU};
//! use torsh_nn::Module;
//!
//! // Basic activation
//! let relu = ReLU::new();
//! let output = relu.forward(&input)?;
//!
//! // Modern activation
//! let gelu = GELU::exact();
//! let output = gelu.forward(&input)?;
//!
//! // Normalization
//! let softmax = Softmax::new(Some(1));
//! let output = softmax.forward(&input)?;
//!
//! // Gated Linear Unit
//! let swiglu = SwiGLU::new(-1);
//! let output = swiglu.forward(&input)?; // Input: [..., 2*d] -> Output: [..., d]
//! ```
//!
//! ## Backward Compatibility
//!
//! All activation functions are re-exported at the module level to maintain
//! 100% backward compatibility with existing code.

// Specialized activation modules
pub mod advanced;
pub mod basic;
pub mod normalization;

// =============================================================================
// RE-EXPORTS FOR BACKWARD COMPATIBILITY
// =============================================================================

// Basic activation functions
pub use basic::{
    Hardshrink,
    Hardsigmoid,
    Hardtanh,
    LeakyReLU,
    LogSigmoid,

    PReLU,
    // ReLU family
    ReLU,
    ReLU6,
    // Sigmoid family
    Sigmoid,
    Softshrink,
    // Tanh family
    Tanh,
    Tanhshrink,

    // Threshold-based
    Threshold,
    ELU,
    SELU,
};

// Normalization and smooth activation functions
pub use normalization::{
    // Utility functions
    log_sum_exp,
    stable_log_softmax,
    stable_softmax,
    LogSoftmax,

    // Probability distributions
    Softmax,
    // Smooth functions
    Softplus,
    Softsign,
};

// Advanced and modern activation functions
pub use advanced::{
    Hardswish,

    Mish,
    ReGLU,
    SiLU,
    SwiGLU,
    Swish, // Alias for SiLU
    GEGLU,
    // Modern activations
    GELU,
    // Gated Linear Units
    GLU,
};

// =============================================================================
// ACTIVATION FACTORY AND UTILITIES
// =============================================================================

/// Activation function factory for creating activations by name
///
/// This provides a convenient way to create activation functions dynamically
/// based on string identifiers, useful for configuration-driven model building.
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::ActivationFactory;
///
/// let relu = ActivationFactory::create("relu").unwrap();
/// let gelu = ActivationFactory::create("gelu").unwrap();
/// let swiglu = ActivationFactory::create("swiglu").unwrap();
/// ```
pub struct ActivationFactory;

impl ActivationFactory {
    /// Creates an activation function by name
    ///
    /// # Arguments
    /// * `name` - The name of the activation function (case-insensitive)
    ///
    /// # Supported Names
    /// - Basic: "relu", "leaky_relu", "relu6", "prelu", "elu", "selu"
    /// - Sigmoid: "sigmoid", "hardsigmoid", "logsigmoid"
    /// - Tanh: "tanh", "hardtanh", "tanhshrink"
    /// - Threshold: "threshold", "hardshrink", "softshrink"
    /// - Normalization: "softmax", "log_softmax", "softplus", "softsign"
    /// - Modern: "gelu", "silu", "swish", "mish", "hardswish"
    /// - Gated: "glu", "geglu", "reglu", "swiglu"
    ///
    /// # Returns
    /// `Some(Box<dyn Module>)` if the activation exists, `None` otherwise
    pub fn create(name: &str) -> Option<Box<dyn crate::Module>> {
        match name.to_lowercase().as_str() {
            // Basic activations
            "relu" => Some(Box::new(ReLU::new())),
            "leaky_relu" | "leakyrelu" => Some(Box::new(LeakyReLU::default())),
            "relu6" => Some(Box::new(ReLU6::new())),
            "prelu" => Some(Box::new(PReLU::default_params().expect("PReLU default params should succeed"))),
            "elu" => Some(Box::new(ELU::default())),
            "selu" => Some(Box::new(SELU::new())),

            // Sigmoid family
            "sigmoid" => Some(Box::new(Sigmoid::new())),
            "hardsigmoid" => Some(Box::new(Hardsigmoid::new())),
            "logsigmoid" | "log_sigmoid" => Some(Box::new(LogSigmoid::new())),

            // Tanh family
            "tanh" => Some(Box::new(Tanh::new())),
            "hardtanh" => Some(Box::new(Hardtanh::default())),
            "tanhshrink" => Some(Box::new(Tanhshrink::new())),

            // Threshold-based
            "threshold" => Some(Box::new(Threshold::default_params())),
            "hardshrink" => Some(Box::new(Hardshrink::default())),
            "softshrink" => Some(Box::new(Softshrink::default())),

            // Normalization
            "softmax" => Some(Box::new(Softmax::new(None))),
            "log_softmax" | "logsoftmax" => Some(Box::new(LogSoftmax::new(None))),
            "softplus" => Some(Box::new(Softplus::default())),
            "softsign" => Some(Box::new(Softsign::new())),

            // Modern activations
            "gelu" => Some(Box::new(GELU::new())),
            "gelu_exact" => Some(Box::new(GELU::exact())),
            "gelu_approx" | "gelu_approximate" => Some(Box::new(GELU::approximate())),
            "silu" | "swish" => Some(Box::new(SiLU::new())),
            "mish" => Some(Box::new(Mish::new())),
            "hardswish" => Some(Box::new(Hardswish::new())),

            // Gated Linear Units
            "glu" => Some(Box::new(GLU::default())),
            "geglu" => Some(Box::new(GEGLU::default())),
            "reglu" => Some(Box::new(ReGLU::default())),
            "swiglu" => Some(Box::new(SwiGLU::default())),

            _ => None,
        }
    }

    /// Creates an activation function with custom parameters
    ///
    /// # Arguments
    /// * `name` - The name of the activation function
    /// * `params` - A map of parameter names to values
    ///
    /// # Examples
    /// ```rust
    /// use std::collections::HashMap;
    /// use torsh_nn::layers::activations::ActivationFactory;
    ///
    /// let mut params = HashMap::new();
    /// params.insert("negative_slope".to_string(), "0.2".to_string());
    /// let leaky_relu = ActivationFactory::create_with_params("leaky_relu", &params).unwrap();
    /// ```
    pub fn create_with_params(
        name: &str,
        params: &std::collections::HashMap<String, String>,
    ) -> Option<Box<dyn crate::Module>> {
        match name.to_lowercase().as_str() {
            "leaky_relu" | "leakyrelu" => {
                let slope = params
                    .get("negative_slope")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.01);
                Some(Box::new(LeakyReLU::new(slope)))
            }
            "elu" => {
                let alpha = params
                    .get("alpha")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                Some(Box::new(ELU::new(alpha)))
            }
            "softmax" => {
                let dim = params.get("dim").and_then(|s| s.parse::<usize>().ok());
                Some(Box::new(Softmax::new(dim)))
            }
            "log_softmax" | "logsoftmax" => {
                let dim = params.get("dim").and_then(|s| s.parse::<usize>().ok());
                Some(Box::new(LogSoftmax::new(dim)))
            }
            "softplus" => {
                let beta = params
                    .get("beta")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                let threshold = params
                    .get("threshold")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(20.0);
                Some(Box::new(Softplus::new(beta, threshold)))
            }
            "hardtanh" => {
                let min_val = params
                    .get("min_val")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(-1.0);
                let max_val = params
                    .get("max_val")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                Some(Box::new(Hardtanh::new(min_val, max_val)))
            }
            "threshold" => {
                let threshold = params
                    .get("threshold")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                let value = params
                    .get("value")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                Some(Box::new(Threshold::new(threshold, value)))
            }
            "hardshrink" => {
                let lambd = params
                    .get("lambd")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                Some(Box::new(Hardshrink::new(lambd)))
            }
            "softshrink" => {
                let lambd = params
                    .get("lambd")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                Some(Box::new(Softshrink::new(lambd)))
            }
            "gelu" => {
                let approximate = params
                    .get("approximate")
                    .and_then(|s| s.parse::<bool>().ok())
                    .unwrap_or(false);
                Some(Box::new(GELU::with_approximate(approximate)))
            }
            "glu" => {
                let dim = params
                    .get("dim")
                    .and_then(|s| s.parse::<isize>().ok())
                    .unwrap_or(-1);
                Some(Box::new(GLU::new(dim)))
            }
            "geglu" => {
                let dim = params
                    .get("dim")
                    .and_then(|s| s.parse::<isize>().ok())
                    .unwrap_or(-1);
                let approximate = params
                    .get("approximate_gelu")
                    .and_then(|s| s.parse::<bool>().ok())
                    .unwrap_or(false);
                Some(Box::new(GEGLU::new(dim, approximate)))
            }
            "reglu" => {
                let dim = params
                    .get("dim")
                    .and_then(|s| s.parse::<isize>().ok())
                    .unwrap_or(-1);
                Some(Box::new(ReGLU::new(dim)))
            }
            "swiglu" => {
                let dim = params
                    .get("dim")
                    .and_then(|s| s.parse::<isize>().ok())
                    .unwrap_or(-1);
                Some(Box::new(SwiGLU::new(dim)))
            }
            _ => Self::create(name),
        }
    }

    /// Lists all available activation function names
    pub fn available_activations() -> Vec<&'static str> {
        vec![
            // Basic
            "relu",
            "leaky_relu",
            "relu6",
            "prelu",
            "elu",
            "selu",
            // Sigmoid
            "sigmoid",
            "hardsigmoid",
            "logsigmoid",
            // Tanh
            "tanh",
            "hardtanh",
            "tanhshrink",
            // Threshold
            "threshold",
            "hardshrink",
            "softshrink",
            // Normalization
            "softmax",
            "log_softmax",
            "softplus",
            "softsign",
            // Modern
            "gelu",
            "gelu_exact",
            "gelu_approx",
            "silu",
            "swish",
            "mish",
            "hardswish",
            // Gated
            "glu",
            "geglu",
            "reglu",
            "swiglu",
        ]
    }
}

/// Activation configuration builder for complex setups
///
/// Provides a fluent interface for building activation configurations
/// with validation and type safety.
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::ActivationBuilder;
///
/// let activation = ActivationBuilder::new("gelu")
///     .approximate(true)
///     .build()
///     .unwrap();
///
/// let glu = ActivationBuilder::new("swiglu")
///     .dim(-1)
///     .build()
///     .unwrap();
/// ```
pub struct ActivationBuilder {
    name: String,
    params: std::collections::HashMap<String, String>,
}

impl ActivationBuilder {
    /// Creates a new activation builder
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            params: std::collections::HashMap::new(),
        }
    }

    /// Sets the dimension parameter (for multi-dimensional activations)
    pub fn dim(mut self, dim: isize) -> Self {
        self.params.insert("dim".to_string(), dim.to_string());
        self
    }

    /// Sets the dimension parameter as usize (for multi-dimensional activations)
    pub fn dim_usize(mut self, dim: usize) -> Self {
        self.params.insert("dim".to_string(), dim.to_string());
        self
    }

    /// Sets the alpha parameter
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.params.insert("alpha".to_string(), alpha.to_string());
        self
    }

    /// Sets the beta parameter
    pub fn beta(mut self, beta: f32) -> Self {
        self.params.insert("beta".to_string(), beta.to_string());
        self
    }

    /// Sets the threshold parameter
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.params
            .insert("threshold".to_string(), threshold.to_string());
        self
    }

    /// Sets the lambda parameter
    pub fn lambda(mut self, lambd: f32) -> Self {
        self.params.insert("lambd".to_string(), lambd.to_string());
        self
    }

    /// Sets the negative slope parameter (for LeakyReLU)
    pub fn negative_slope(mut self, slope: f32) -> Self {
        self.params
            .insert("negative_slope".to_string(), slope.to_string());
        self
    }

    /// Sets the approximate parameter (for GELU)
    pub fn approximate(mut self, approx: bool) -> Self {
        self.params
            .insert("approximate".to_string(), approx.to_string());
        self
    }

    /// Sets the approximate GELU parameter (for GEGLU)
    pub fn approximate_gelu(mut self, approx: bool) -> Self {
        self.params
            .insert("approximate_gelu".to_string(), approx.to_string());
        self
    }

    /// Sets the min_val parameter (for Hardtanh)
    pub fn min_val(mut self, min_val: f32) -> Self {
        self.params
            .insert("min_val".to_string(), min_val.to_string());
        self
    }

    /// Sets the max_val parameter (for Hardtanh)
    pub fn max_val(mut self, max_val: f32) -> Self {
        self.params
            .insert("max_val".to_string(), max_val.to_string());
        self
    }

    /// Sets the value parameter (for Threshold)
    pub fn value(mut self, value: f32) -> Self {
        self.params.insert("value".to_string(), value.to_string());
        self
    }

    /// Builds the activation function
    pub fn build(self) -> Option<Box<dyn crate::Module>> {
        ActivationFactory::create_with_params(&self.name, &self.params)
    }
}

// =============================================================================
// COMMON PRESETS AND UTILITIES
// =============================================================================

/// Common activation presets for different model architectures
pub mod presets {
    use super::*;

    /// Get recommended activations for different neural network types
    pub fn for_architecture(arch: &str) -> Box<dyn crate::Module> {
        match arch.to_lowercase().as_str() {
            "transformer" | "attention" => Box::new(GELU::new()),
            "cnn" | "convnet" | "resnet" => Box::new(ReLU::new()),
            "mobile" | "mobilenet" => Box::new(Hardswish::new()),
            "efficientnet" => Box::new(SiLU::new()),
            "vision_transformer" | "vit" => Box::new(GELU::new()),
            "bert" | "gpt" => Box::new(GELU::new()),
            "lstm" | "gru" | "rnn" => Box::new(Tanh::new()),
            "autoencoder" => Box::new(ReLU::new()),
            "gan" => Box::new(LeakyReLU::new(0.2)),
            _ => Box::new(ReLU::new()), // Default fallback
        }
    }

    /// Get modern activation replacements for legacy functions
    pub fn modern_replacement(legacy: &str) -> Box<dyn crate::Module> {
        match legacy.to_lowercase().as_str() {
            "relu" => Box::new(GELU::new()),
            "sigmoid" => Box::new(SiLU::new()),
            "tanh" => Box::new(Mish::new()),
            "swish" => Box::new(SiLU::new()),
            _ => ActivationFactory::create(legacy).unwrap_or_else(|| Box::new(ReLU::new())),
        }
    }

    /// Get efficient mobile-optimized activations
    pub fn mobile_optimized(standard: &str) -> Box<dyn crate::Module> {
        match standard.to_lowercase().as_str() {
            "relu" => Box::new(ReLU6::new()),
            "swish" | "silu" => Box::new(Hardswish::new()),
            "sigmoid" => Box::new(Hardsigmoid::new()),
            "gelu" => Box::new(GELU::approximate()),
            _ => ActivationFactory::create(standard).unwrap_or_else(|| Box::new(ReLU6::new())),
        }
    }
}

/// Performance benchmarking utilities for activation functions
pub mod benchmarks {
    use super::*;
    use std::time::Instant;

    /// Benchmark activation function performance
    pub fn benchmark_activation(
        activation: &dyn crate::Module,
        input: &torsh_tensor::Tensor,
        iterations: usize,
    ) -> Result<(f64, f64), torsh_core::error::TorshError> {
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();
            let _output = activation.forward(input)?;
            let elapsed = start.elapsed().as_secs_f64();
            times.push(elapsed);
        }

        let mean_time = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times
            .iter()
            .map(|&time| (time - mean_time).powi(2))
            .sum::<f64>()
            / times.len() as f64;
        let std_dev = variance.sqrt();

        Ok((mean_time, std_dev))
    }

    /// Compare multiple activation functions
    pub fn compare_activations(
        activations: &[(&str, Box<dyn crate::Module>)],
        input: &torsh_tensor::Tensor,
        iterations: usize,
    ) -> Vec<(String, f64, f64)> {
        activations
            .iter()
            .filter_map(|(name, activation)| {
                benchmark_activation(activation.as_ref(), input, iterations)
                    .ok()
                    .map(|(mean, std)| (name.to_string(), mean, std))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_activation_factory() {
        // Test basic creation
        assert!(ActivationFactory::create("relu").is_some());
        assert!(ActivationFactory::create("gelu").is_some());
        assert!(ActivationFactory::create("swiglu").is_some());
        assert!(ActivationFactory::create("nonexistent").is_none());
    }

    #[test]
    fn test_activation_factory_with_params() {
        let mut params = std::collections::HashMap::new();
        params.insert("negative_slope".to_string(), "0.2".to_string());

        let activation = ActivationFactory::create_with_params("leaky_relu", &params);
        assert!(activation.is_some());
    }

    #[test]
    fn test_activation_builder() {
        let activation = ActivationBuilder::new("gelu").approximate(true).build();
        assert!(activation.is_some());

        let glu = ActivationBuilder::new("swiglu").dim(-1).build();
        assert!(glu.is_some());
    }

    #[test]
    fn test_available_activations() {
        let activations = ActivationFactory::available_activations();
        assert!(!activations.is_empty());
        assert!(activations.contains(&"relu"));
        assert!(activations.contains(&"gelu"));
        assert!(activations.contains(&"swiglu"));
    }

    #[test]
    fn test_presets() {
        let transformer_activation = presets::for_architecture("transformer");
        let mobile_activation = presets::for_architecture("mobile");
        let cnn_activation = presets::for_architecture("cnn");

        // Just verify they can be created
        assert_eq!(
            std::mem::discriminant(&*transformer_activation),
            std::mem::discriminant(&*Box::new(GELU::new()) as Box<dyn crate::Module>)
        );
    }

    #[test]
    fn test_module_integration() {
        // Test that re-exported activations work correctly
        let relu = ReLU::new();
        let gelu = GELU::new();
        let softmax = Softmax::new(None);
        let swiglu = SwiGLU::new(-1);

        // Test basic properties
        assert_eq!(relu.training(), true); // Default training mode
        assert_eq!(gelu.training(), true);
        assert_eq!(softmax.training(), true);
        assert_eq!(swiglu.training(), true);
    }

    #[test]
    fn test_backward_compatibility() {
        // Ensure all major activation types are accessible
        let _relu = ReLU::new();
        let _leaky_relu = LeakyReLU::new(0.01);
        let _sigmoid = Sigmoid::new();
        let _tanh = Tanh::new();
        let _gelu = GELU::new();
        let _silu = SiLU::new();
        let _swish = Swish::new(); // Alias test
        let _softmax = Softmax::new(None);
        let _glu = GLU::new(-1);
        let _geglu = GEGLU::new(-1, false);
        let _reglu = ReGLU::new(-1);
        let _swiglu = SwiGLU::new(-1);

        // If this compiles, backward compatibility is maintained
    }

    #[test]
    fn test_performance_benchmarking() {
        let input = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();
        let relu = ReLU::new();

        let result = benchmarks::benchmark_activation(&relu, &input, 5);
        assert!(result.is_ok());

        let (mean_time, std_dev) = result.unwrap();
        assert!(mean_time >= 0.0);
        assert!(std_dev >= 0.0);
    }
}
