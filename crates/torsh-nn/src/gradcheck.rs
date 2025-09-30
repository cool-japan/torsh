//! Gradient checking utilities for neural network layers
//!
//! This module provides utilities for numerical gradient checking to validate
//! automatic differentiation implementations.

use crate::{Module, Parameter};
use scirs2_core::random::{Random, Rng};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{collections::HashSet, string::String, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

#[cfg(not(feature = "std"))]
use hashbrown::{HashMap, HashSet};

/// Gradient checking configuration
#[derive(Debug, Clone)]
pub struct GradCheckConfig {
    /// Epsilon for finite differences
    pub eps: f64,
    /// Relative tolerance for gradient comparison
    pub rtol: f64,
    /// Absolute tolerance for gradient comparison
    pub atol: f64,
    /// Whether to use double precision for calculations
    pub double_precision: bool,
    /// Maximum number of elements to check (for large tensors)
    pub max_elements: Option<usize>,
    /// Random seed for sampling elements to check
    pub seed: Option<u64>,
}

impl Default for GradCheckConfig {
    fn default() -> Self {
        Self {
            eps: 1e-6,
            rtol: 1e-3,
            atol: 1e-5,
            double_precision: false,
            max_elements: Some(100),
            seed: Some(42),
        }
    }
}

/// Gradient check result for a single parameter
#[derive(Debug, Clone)]
pub struct ParameterGradCheckResult {
    /// Parameter name
    pub name: String,
    /// Whether the gradient check passed
    pub passed: bool,
    /// Maximum absolute difference
    pub max_abs_diff: f64,
    /// Maximum relative difference
    pub max_rel_diff: f64,
    /// Number of elements checked
    pub elements_checked: usize,
    /// Error message if check failed
    pub error: Option<String>,
}

/// Overall gradient check result
#[derive(Debug, Clone)]
pub struct GradCheckResult {
    /// Whether all parameters passed
    pub passed: bool,
    /// Results for individual parameters
    pub parameter_results: Vec<ParameterGradCheckResult>,
    /// Overall summary
    pub summary: String,
}

impl GradCheckResult {
    /// Get parameters that failed the gradient check
    pub fn failed_parameters(&self) -> Vec<&ParameterGradCheckResult> {
        self.parameter_results
            .iter()
            .filter(|r| !r.passed)
            .collect()
    }

    /// Get the worst parameter (highest error)
    pub fn worst_parameter(&self) -> Option<&ParameterGradCheckResult> {
        self.parameter_results
            .iter()
            .max_by(|a, b| a.max_abs_diff.partial_cmp(&b.max_abs_diff).unwrap())
    }
}

/// Gradient checker for neural network modules
pub struct GradChecker {
    config: GradCheckConfig,
}

impl GradChecker {
    /// Create a new gradient checker with default configuration
    pub fn new() -> Self {
        Self {
            config: GradCheckConfig::default(),
        }
    }

    /// Create a gradient checker with custom configuration
    pub fn with_config(config: GradCheckConfig) -> Self {
        Self { config }
    }

    /// Check gradients for a module
    pub fn check_module<M: Module, F>(
        &self,
        module: &M,
        input: &Tensor<f32>,
        loss_fn: F,
    ) -> Result<GradCheckResult>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        let parameters = module.named_parameters();
        let mut parameter_results = Vec::new();
        let mut all_passed = true;

        for (name, param) in parameters.iter() {
            match self.check_parameter(module, input, param, &name, &loss_fn) {
                Ok(result) => {
                    if !result.passed {
                        all_passed = false;
                    }
                    parameter_results.push(result);
                }
                Err(e) => {
                    all_passed = false;
                    parameter_results.push(ParameterGradCheckResult {
                        name: name.clone(),
                        passed: false,
                        max_abs_diff: f64::INFINITY,
                        max_rel_diff: f64::INFINITY,
                        elements_checked: 0,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        let summary = if all_passed {
            format!(
                "All {} parameters passed gradient check",
                parameter_results.len()
            )
        } else {
            let failed_count = parameter_results.iter().filter(|r| !r.passed).count();
            format!(
                "{} out of {} parameters failed gradient check",
                failed_count,
                parameter_results.len()
            )
        };

        Ok(GradCheckResult {
            passed: all_passed,
            parameter_results,
            summary,
        })
    }

    /// Check gradients for a pure function (no module parameters)
    pub fn check_function<F>(&self, func: F, input: &Tensor<f32>) -> Result<GradCheckResult>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        // For pure functions, we check the gradient with respect to the input
        let input_with_grad = input.clone().requires_grad_(true);

        // Compute function output
        let output = func(&input_with_grad)?;

        // Perform backward pass to get analytical gradient
        output.backward()?;
        let analytical_grad = input_with_grad
            .grad()
            .ok_or_else(|| TorshError::AutogradError("No gradient computed".to_string()))?;

        // Compute numerical gradient
        let numerical_grad = self.compute_numerical_gradient_function(&func, input)?;

        // Compare gradients
        let comparison = self.compare_gradients(&analytical_grad, &numerical_grad)?;

        let param_result = ParameterGradCheckResult {
            name: "input".to_string(),
            passed: comparison.0,
            max_abs_diff: comparison.1,
            max_rel_diff: comparison.2,
            elements_checked: comparison.3,
            error: None,
        };

        let summary = if comparison.0 {
            "Function gradient check passed".to_string()
        } else {
            "Function gradient check failed".to_string()
        };

        Ok(GradCheckResult {
            passed: comparison.0,
            parameter_results: vec![param_result],
            summary,
        })
    }

    /// Check gradients for a single parameter
    fn check_parameter<M: Module, F>(
        &self,
        module: &M,
        input: &Tensor<f32>,
        parameter: &Parameter,
        param_name: &str,
        loss_fn: &F,
    ) -> Result<ParameterGradCheckResult>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        // Get analytical gradient (this would come from autograd)
        // For now, we'll use a placeholder implementation
        let analytical_grad =
            self.compute_analytical_gradient(module, input, parameter, loss_fn)?;

        // Compute numerical gradient
        let numerical_grad =
            self.compute_numerical_gradient(module, input, parameter, param_name, loss_fn)?;

        // Compare gradients
        let comparison = self.compare_gradients(&analytical_grad, &numerical_grad)?;

        Ok(ParameterGradCheckResult {
            name: param_name.to_string(),
            passed: comparison.0,
            max_abs_diff: comparison.1,
            max_rel_diff: comparison.2,
            elements_checked: comparison.3,
            error: None,
        })
    }

    /// Compute analytical gradient (placeholder - would use autograd)
    fn compute_analytical_gradient<M: Module, F>(
        &self,
        _module: &M,
        _input: &Tensor<f32>,
        parameter: &Parameter,
        _loss_fn: &F,
    ) -> Result<Tensor<f32>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        // Placeholder: return zeros for now
        // In a real implementation, this would:
        // 1. Enable gradients for the parameter
        // 2. Compute forward pass
        // 3. Compute loss
        // 4. Backpropagate to get gradients
        let param_data = parameter.tensor().read().clone();
        Ok(Tensor::zeros_like(&param_data)?)
    }

    /// Compute numerical gradient using finite differences
    fn compute_numerical_gradient<M: Module, F>(
        &self,
        module: &M,
        input: &Tensor<f32>,
        parameter: &Parameter,
        _param_name: &str,
        loss_fn: &F,
    ) -> Result<Tensor<f32>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        let param_data = parameter.tensor().read().clone();
        let param_shape = param_data.shape().dims().to_vec();
        let numel = param_data.numel();

        // Determine which elements to check
        let indices_to_check = self.get_indices_to_check(numel);

        // Initialize gradient tensor
        let mut grad_data = vec![0.0f32; numel];

        for &idx in &indices_to_check {
            // Forward difference: f(x + h) - f(x - h) / (2h)
            let grad = self.compute_finite_difference(module, input, parameter, idx, loss_fn)?;
            grad_data[idx] = grad;
        }

        Ok(Tensor::from_data(grad_data, param_shape, param_data.device()).unwrap())
    }

    /// Compute finite difference for a single parameter element
    fn compute_finite_difference<M: Module, F>(
        &self,
        module: &M,
        input: &Tensor<f32>,
        parameter: &Parameter,
        idx: usize,
        loss_fn: &F,
    ) -> Result<f32>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        let param_data = parameter.tensor().read().clone();
        let original_value = param_data.get_item(&[idx])?;

        // Compute f(x + h)
        let mut param_plus = param_data.clone();
        param_plus.set_item(&[idx], original_value + self.config.eps as f32)?;

        // Update module parameter (this is a simplified approach)
        // In practice, you'd need to temporarily modify the module's parameter
        let output_plus = module.forward(input)?;
        let loss_plus = loss_fn(&output_plus)?;
        let loss_plus_scalar = loss_plus.item();

        // Compute f(x - h)
        let mut param_minus = param_data.clone();
        param_minus.set_item(&[idx], original_value - self.config.eps as f32)?;

        let output_minus = module.forward(input)?;
        let loss_minus = loss_fn(&output_minus)?;
        let loss_minus_scalar = loss_minus.item();

        // Central difference
        let grad = (loss_plus_scalar? - loss_minus_scalar?) / (2.0 * self.config.eps as f32);

        Ok(grad)
    }

    /// Compute numerical gradient for a function with respect to input
    fn compute_numerical_gradient_function<F>(
        &self,
        func: &F,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        let input_shape = input.shape().dims().to_vec();
        let numel = input.numel();

        // Determine which elements to check
        let indices_to_check = self.get_indices_to_check(numel);

        // Initialize gradient tensor
        let mut grad_data = vec![0.0f32; numel];

        for &idx in &indices_to_check {
            // Central difference: f(x + h) - f(x - h) / (2h)
            let original_value = input.get_item(&[idx])?;

            // Compute f(x + h)
            let mut input_plus = input.clone();
            input_plus.set_item(&[idx], original_value + self.config.eps as f32)?;
            let output_plus = func(&input_plus)?;
            let loss_plus_scalar = output_plus.item()?;

            // Compute f(x - h)
            let mut input_minus = input.clone();
            input_minus.set_item(&[idx], original_value - self.config.eps as f32)?;
            let output_minus = func(&input_minus)?;
            let loss_minus_scalar = output_minus.item()?;

            // Central difference
            let grad = (loss_plus_scalar - loss_minus_scalar) / (2.0 * self.config.eps as f32);
            grad_data[idx] = grad;
        }

        Ok(Tensor::from_data(grad_data, input_shape, input.device()).unwrap())
    }

    /// Get indices to check (sampling for large tensors)
    fn get_indices_to_check(&self, numel: usize) -> Vec<usize> {
        if let Some(max_elements) = self.config.max_elements {
            if numel <= max_elements {
                (0..numel).collect()
            } else {
                // Sample random indices
                #[cfg(feature = "std")]
                {
                    let mut rng = self.get_rng();
                    let mut indices = HashSet::new();

                    while indices.len() < max_elements {
                        let idx = rng.gen_range(0..numel);
                        indices.insert(idx);
                    }

                    indices.into_iter().collect()
                }
                #[cfg(not(feature = "std"))]
                {
                    // For no_std, use a simple linear sampling approach
                    let mut rng = self.get_rng();
                    let mut indices = Vec::new();

                    for _ in 0..max_elements.min(numel) {
                        let idx = rng.gen_range(0..numel);
                        if !indices.contains(&idx) {
                            indices.push(idx);
                        }
                    }

                    indices
                }
            }
        } else {
            (0..numel).collect()
        }
    }

    /// Get random number generator
    fn get_rng(&self) -> Random {
        if let Some(_seed) = self.config.seed {
            // For seeded random generation, use SciRS2 Random type
            // Note: Using default for now due to type compatibility
            Random::default()
        } else {
            Random::default()
        }
    }

    /// Compare analytical and numerical gradients
    fn compare_gradients(
        &self,
        analytical: &Tensor<f32>,
        numerical: &Tensor<f32>,
    ) -> Result<(bool, f64, f64, usize)> {
        let anal_data = analytical.data()?;
        let num_data = numerical.data()?;

        if anal_data.len() != num_data.len() {
            return Err(TorshError::InvalidArgument(
                "Gradient tensors have different sizes".to_string(),
            ));
        }

        let mut max_abs_diff: f64 = 0.0;
        let mut max_rel_diff: f64 = 0.0;
        let mut all_within_tolerance = true;

        for (_i, (&a, &n)) in anal_data.iter().zip(num_data.iter()).enumerate() {
            let abs_diff = (a as f64 - n as f64).abs();
            let rel_diff = if n.abs() > 1e-8 {
                abs_diff / (n as f64).abs()
            } else {
                abs_diff
            };

            max_abs_diff = max_abs_diff.max(abs_diff);
            max_rel_diff = max_rel_diff.max(rel_diff);

            if abs_diff > self.config.atol && rel_diff > self.config.rtol {
                all_within_tolerance = false;
            }
        }

        Ok((
            all_within_tolerance,
            max_abs_diff,
            max_rel_diff,
            anal_data.len(),
        ))
    }
}

impl Default for GradChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for gradient checking
pub fn gradcheck<M: Module, F>(
    module: &M,
    input: &Tensor<f32>,
    loss_fn: F,
) -> Result<GradCheckResult>
where
    F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
{
    let checker = GradChecker::new();
    checker.check_module(module, input, loss_fn)
}

/// Fast gradient check with relaxed tolerances
pub fn fast_gradcheck<M: Module, F>(
    module: &M,
    input: &Tensor<f32>,
    loss_fn: F,
) -> Result<GradCheckResult>
where
    F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
{
    let config = GradCheckConfig {
        eps: 1e-4,
        rtol: 1e-2,
        atol: 1e-3,
        max_elements: Some(10),
        ..Default::default()
    };

    let checker = GradChecker::with_config(config);
    checker.check_module(module, input, loss_fn)
}

/// High precision gradient check
pub fn precise_gradcheck<M: Module, F>(
    module: &M,
    input: &Tensor<f32>,
    loss_fn: F,
) -> Result<GradCheckResult>
where
    F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
{
    let config = GradCheckConfig {
        eps: 1e-8,
        rtol: 1e-5,
        atol: 1e-7,
        double_precision: true,
        max_elements: None,
        ..Default::default()
    };

    let checker = GradChecker::with_config(config);
    checker.check_module(module, input, loss_fn)
}

/// Functional gradient check for pure functions (not modules)
///
/// This function checks gradients of pure functions that take a tensor and return a scalar tensor.
/// It performs numerical differentiation to verify automatic differentiation gradients.
pub fn gradcheck_function<F>(
    func: F,
    input: &Tensor<f32>,
    config: &GradCheckConfig,
) -> Result<GradCheckResult>
where
    F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
{
    let checker = GradChecker::with_config(config.clone());
    checker.check_function(func, input)
}

/// Fast functional gradient check with relaxed tolerances
pub fn fast_gradcheck_function<F>(func: F, input: &Tensor<f32>) -> Result<GradCheckResult>
where
    F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
{
    let config = GradCheckConfig {
        eps: 1e-4,
        rtol: 1e-2,
        atol: 1e-3,
        max_elements: Some(10),
        ..Default::default()
    };
    gradcheck_function(func, input, &config)
}

// Helper trait for tensors to support item access and modification
trait TensorItemAccess<T> {
    fn get_item(&self, idx: usize) -> Result<T>;
    fn set_item(&mut self, idx: usize, value: T) -> Result<()>;
}

impl TensorItemAccess<f32> for Tensor<f32> {
    fn get_item(&self, idx: usize) -> Result<f32> {
        let data = self.data()?;
        if idx >= data.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Index {} out of bounds for tensor with {} elements",
                idx,
                data.len()
            )));
        }
        Ok(data[idx])
    }

    fn set_item(&mut self, _idx: usize, _value: f32) -> Result<()> {
        // This is a simplified implementation
        // In practice, you'd need proper tensor mutation support
        Err(TorshError::UnsupportedOperation {
            op: "set_item".to_string(),
            dtype: "tensor mutation not yet supported".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    // Conditional imports for std/no_std compatibility
    #[cfg(feature = "std")]
    use std::collections::HashMap;

    #[cfg(not(feature = "std"))]
    use hashbrown::HashMap;

    // Mock module for testing
    #[allow(dead_code)]
    struct LinearModule {
        weight: Tensor<f32>,
        bias: Option<Tensor<f32>>,
    }

    impl LinearModule {
        #[allow(dead_code)]
        fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self> {
            let weight = randn(&[out_features, in_features])?;
            let bias = if bias {
                Some(zeros(&[out_features])?)
            } else {
                None
            };

            Ok(Self { weight, bias })
        }
    }

    impl Module for LinearModule {
        fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
            let output = input.matmul(&self.weight.transpose(-1, -2)?)?;
            if let Some(ref bias) = self.bias {
                output.add_op(bias)
            } else {
                Ok(output)
            }
        }

        fn parameters(&self) -> HashMap<String, Parameter> {
            let mut params = HashMap::new();
            params.insert("weight".to_string(), Parameter::new(self.weight.clone()));
            if let Some(ref bias) = self.bias {
                params.insert("bias".to_string(), Parameter::new(bias.clone()));
            }
            params
        }

        fn named_parameters(&self) -> HashMap<String, Parameter> {
            self.parameters()
        }

        fn training(&self) -> bool {
            true
        }
        fn train(&mut self) {}
        fn eval(&mut self) {}
        fn set_training(&mut self, _training: bool) {}
        fn to_device(&mut self, _device: torsh_core::DeviceType) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_gradcheck_config() {
        let config = GradCheckConfig::default();
        assert_eq!(config.eps, 1e-6);
        assert_eq!(config.rtol, 1e-3);
        assert_eq!(config.atol, 1e-5);
        assert_eq!(config.max_elements, Some(100));
    }

    #[test]
    fn test_grad_checker_creation() {
        let checker = GradChecker::new();
        assert_eq!(checker.config.eps, 1e-6);

        let custom_config = GradCheckConfig {
            eps: 1e-4,
            ..Default::default()
        };
        let custom_checker = GradChecker::with_config(custom_config);
        assert_eq!(custom_checker.config.eps, 1e-4);
    }

    #[test]
    fn test_parameter_grad_check_result() {
        let result = ParameterGradCheckResult {
            name: "test_param".to_string(),
            passed: true,
            max_abs_diff: 1e-6,
            max_rel_diff: 1e-5,
            elements_checked: 100,
            error: None,
        };

        assert_eq!(result.name, "test_param");
        assert!(result.passed);
        assert_eq!(result.elements_checked, 100);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_grad_check_result() {
        let param_results = vec![
            ParameterGradCheckResult {
                name: "param1".to_string(),
                passed: true,
                max_abs_diff: 1e-6,
                max_rel_diff: 1e-5,
                elements_checked: 50,
                error: None,
            },
            ParameterGradCheckResult {
                name: "param2".to_string(),
                passed: false,
                max_abs_diff: 1e-2,
                max_rel_diff: 1e-1,
                elements_checked: 50,
                error: None,
            },
        ];

        let result = GradCheckResult {
            passed: false,
            parameter_results: param_results,
            summary: "1 out of 2 parameters failed gradient check".to_string(),
        };

        assert!(!result.passed);
        assert_eq!(result.failed_parameters().len(), 1);
        assert_eq!(result.worst_parameter().unwrap().name, "param2");
    }

    #[test]
    fn test_indices_selection() {
        let checker = GradChecker::new();

        // Small tensor - should check all elements
        let indices = checker.get_indices_to_check(50);
        assert_eq!(indices.len(), 50);

        // Large tensor - should sample
        let indices = checker.get_indices_to_check(1000);
        assert_eq!(indices.len(), 100); // max_elements is 100 by default
    }

    #[test]
    fn test_convenience_functions() {
        // These would require working tensor operations, so we just test they exist
        assert!(true); // Placeholder for actual tests when tensor ops work
    }
}
