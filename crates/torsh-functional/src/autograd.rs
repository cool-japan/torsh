//! Custom autograd function creation utilities
//!
//! This module provides utilities for creating custom differentiable operations
//! that can be used with the automatic differentiation system.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Trait for custom autograd functions
pub trait CustomAutogradFunction {
    /// Forward pass computation
    fn forward(&self, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>>;

    /// Backward pass computation
    fn backward(
        &self,
        grad_outputs: &[Tensor],
        inputs: &[Tensor],
    ) -> TorshResult<Vec<Option<Tensor>>>;

    /// Number of inputs expected by this function
    fn num_inputs(&self) -> usize;

    /// Number of outputs produced by this function
    fn num_outputs(&self) -> usize;

    /// Get the name of this function (for debugging)
    fn name(&self) -> &str;
}

/// Context for storing intermediate values during forward pass
#[derive(Debug, Clone)]
pub struct AutogradContext {
    /// Saved tensors for backward pass
    pub saved_tensors: Vec<Tensor>,
    /// Saved values for backward pass
    pub saved_values: HashMap<String, f32>,
    /// Saved shapes for backward pass
    pub saved_shapes: HashMap<String, Vec<usize>>,
    /// Whether to save tensors for backward pass
    pub needs_input_grad: Vec<bool>,
}

impl AutogradContext {
    /// Create a new autograd context
    pub fn new(num_inputs: usize) -> Self {
        Self {
            saved_tensors: Vec::new(),
            saved_values: HashMap::new(),
            saved_shapes: HashMap::new(),
            needs_input_grad: vec![true; num_inputs],
        }
    }

    /// Save a tensor for backward pass
    pub fn save_tensor(&mut self, tensor: Tensor) {
        self.saved_tensors.push(tensor);
    }

    /// Save a scalar value for backward pass
    pub fn save_value(&mut self, key: &str, value: f32) {
        self.saved_values.insert(key.to_string(), value);
    }

    /// Save a shape for backward pass
    pub fn save_shape(&mut self, key: &str, shape: Vec<usize>) {
        self.saved_shapes.insert(key.to_string(), shape);
    }

    /// Get saved tensor by index
    pub fn get_saved_tensor(&self, index: usize) -> Option<&Tensor> {
        self.saved_tensors.get(index)
    }

    /// Get saved value by key
    pub fn get_saved_value(&self, key: &str) -> Option<f32> {
        self.saved_values.get(key).copied()
    }

    /// Get saved shape by key
    pub fn get_saved_shape(&self, key: &str) -> Option<&Vec<usize>> {
        self.saved_shapes.get(key)
    }

    /// Set whether input gradients are needed
    pub fn set_needs_input_grad(&mut self, needs_grad: Vec<bool>) {
        self.needs_input_grad = needs_grad;
    }

    /// Check if input gradient is needed
    pub fn needs_input_grad(&self, index: usize) -> bool {
        self.needs_input_grad.get(index).copied().unwrap_or(false)
    }
}

/// Base trait for creating custom autograd functions with context
pub trait CustomAutogradFunctionWithContext {
    /// Forward pass with context
    fn forward(&self, ctx: &mut AutogradContext, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>>;

    /// Backward pass with context
    fn backward(
        &self,
        ctx: &AutogradContext,
        grad_outputs: &[Tensor],
    ) -> TorshResult<Vec<Option<Tensor>>>;

    /// Number of inputs expected by this function
    fn num_inputs(&self) -> usize;

    /// Number of outputs produced by this function
    fn num_outputs(&self) -> usize;

    /// Get the name of this function (for debugging)
    fn name(&self) -> &str;
}

/// Registry for custom autograd functions
pub struct AutogradRegistry {
    functions: HashMap<String, Arc<dyn CustomAutogradFunction + Send + Sync>>,
}

impl AutogradRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    /// Register a custom function
    pub fn register<F>(&mut self, name: String, function: F)
    where
        F: CustomAutogradFunction + Send + Sync + 'static,
    {
        self.functions.insert(name, Arc::new(function));
    }

    /// Get a registered function by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn CustomAutogradFunction + Send + Sync>> {
        self.functions.get(name).cloned()
    }

    /// List all registered functions
    pub fn list_functions(&self) -> Vec<&String> {
        self.functions.keys().collect()
    }
}

impl Default for AutogradRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply a custom autograd function
pub fn apply_custom_function<F>(function: F, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>>
where
    F: CustomAutogradFunction,
{
    // Validate inputs
    if inputs.len() != function.num_inputs() {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Expected {} inputs, got {}",
                function.num_inputs(),
                inputs.len()
            ),
            "apply_custom_function",
        ));
    }

    // Apply forward pass
    let outputs = function.forward(inputs)?;

    // Validate outputs
    if outputs.len() != function.num_outputs() {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Expected {} outputs, got {}",
                function.num_outputs(),
                outputs.len()
            ),
            "apply_custom_function",
        ));
    }

    Ok(outputs)
}

/// Apply a custom autograd function with context
pub fn apply_custom_function_with_context<F>(
    function: F,
    inputs: &[Tensor],
) -> TorshResult<Vec<Tensor>>
where
    F: CustomAutogradFunctionWithContext,
{
    // Validate inputs
    if inputs.len() != function.num_inputs() {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Expected {} inputs, got {}",
                function.num_inputs(),
                inputs.len()
            ),
            "apply_custom_function_with_context",
        ));
    }

    // Create context
    let mut ctx = AutogradContext::new(inputs.len());

    // Apply forward pass
    let outputs = function.forward(&mut ctx, inputs)?;

    // Validate outputs
    if outputs.len() != function.num_outputs() {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Expected {} outputs, got {}",
                function.num_outputs(),
                outputs.len()
            ),
            "apply_custom_function_with_context",
        ));
    }

    Ok(outputs)
}

/// Example custom function: Element-wise square
pub struct SquareFunction;

impl CustomAutogradFunction for SquareFunction {
    fn forward(&self, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>> {
        let input = &inputs[0];
        let output = input.mul_op(input)?;
        Ok(vec![output])
    }

    fn backward(
        &self,
        grad_outputs: &[Tensor],
        inputs: &[Tensor],
    ) -> TorshResult<Vec<Option<Tensor>>> {
        let grad_output = &grad_outputs[0];
        let input = &inputs[0];

        // d/dx(x^2) = 2x
        let two = Tensor::from_data(vec![2.0f32], vec![1], input.device())?;
        let grad_input = grad_output.mul_op(&input.mul_op(&two)?)?;

        Ok(vec![Some(grad_input)])
    }

    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "square"
    }
}

/// Example custom function: Element-wise exponential
pub struct ExpFunction;

impl CustomAutogradFunction for ExpFunction {
    fn forward(&self, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>> {
        let input = &inputs[0];
        let output = input.exp()?;
        Ok(vec![output])
    }

    fn backward(
        &self,
        grad_outputs: &[Tensor],
        inputs: &[Tensor],
    ) -> TorshResult<Vec<Option<Tensor>>> {
        let grad_output = &grad_outputs[0];
        let input = &inputs[0];

        // d/dx(exp(x)) = exp(x)
        let exp_input = input.exp()?;
        let grad_input = grad_output.mul_op(&exp_input)?;

        Ok(vec![Some(grad_input)])
    }

    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "exp"
    }
}

/// Example custom function with context: Scaled addition
pub struct ScaledAddFunction {
    scale: f32,
}

impl ScaledAddFunction {
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }
}

impl CustomAutogradFunctionWithContext for ScaledAddFunction {
    fn forward(&self, ctx: &mut AutogradContext, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>> {
        let a = &inputs[0];
        let b = &inputs[1];

        // Save scale for backward pass
        ctx.save_value("scale", self.scale);

        // Compute scale * a + b
        let scaled_a = a.mul_scalar(self.scale)?;
        let output = scaled_a.add_op(b)?;

        Ok(vec![output])
    }

    fn backward(
        &self,
        ctx: &AutogradContext,
        grad_outputs: &[Tensor],
    ) -> TorshResult<Vec<Option<Tensor>>> {
        let grad_output = &grad_outputs[0];
        let scale = ctx.get_saved_value("scale").unwrap_or(1.0);

        // Gradients: d/da = scale, d/db = 1
        let grad_a = if ctx.needs_input_grad(0) {
            Some(grad_output.mul_scalar(scale)?)
        } else {
            None
        };

        let grad_b = if ctx.needs_input_grad(1) {
            Some(grad_output.clone())
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "scaled_add"
    }
}

/// Macro for creating simple custom autograd functions
#[macro_export]
macro_rules! create_custom_autograd_function {
    (
        name: $name:ident,
        inputs: $num_inputs:expr,
        outputs: $num_outputs:expr,
        forward: |$inputs:ident| $forward_body:expr,
        backward: |$grad_outputs:ident, $backward_inputs:ident| $backward_body:expr
    ) => {
        pub struct $name;

        impl CustomAutogradFunction for $name {
            fn forward(&self, $inputs: &[Tensor]) -> TorshResult<Vec<Tensor>> {
                $forward_body
            }

            fn backward(
                &self,
                $grad_outputs: &[Tensor],
                $backward_inputs: &[Tensor],
            ) -> TorshResult<Vec<Option<Tensor>>> {
                $backward_body
            }

            fn num_inputs(&self) -> usize {
                $num_inputs
            }
            fn num_outputs(&self) -> usize {
                $num_outputs
            }
            fn name(&self) -> &str {
                stringify!($name)
            }
        }
    };
}

/// Create a global registry for custom functions
static GLOBAL_REGISTRY: OnceLock<Mutex<AutogradRegistry>> = OnceLock::new();

/// Get the global autograd registry
pub fn get_global_registry() -> &'static Mutex<AutogradRegistry> {
    GLOBAL_REGISTRY.get_or_init(|| Mutex::new(AutogradRegistry::new()))
}

/// Register a custom function globally
pub fn register_custom_function<F>(name: String, function: F)
where
    F: CustomAutogradFunction + Send + Sync + 'static,
{
    get_global_registry()
        .lock()
        .expect("autograd registry lock should not be poisoned")
        .register(name, function);
}

/// Apply a globally registered function
pub fn apply_registered_function(name: &str, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>> {
    let registry = get_global_registry()
        .lock()
        .expect("lock should not be poisoned");
    let function = registry.get(name).ok_or_else(|| {
        TorshError::invalid_argument_with_context(
            &format!("Function '{}' not found in registry", name),
            "apply_registered_function",
        )
    })?;

    // Validate inputs
    if inputs.len() != function.num_inputs() {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Expected {} inputs, got {}",
                function.num_inputs(),
                inputs.len()
            ),
            "apply_registered_function",
        ));
    }

    // Apply forward pass
    let outputs = function.forward(inputs)?;

    // Validate outputs
    if outputs.len() != function.num_outputs() {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Expected {} outputs, got {}",
                function.num_outputs(),
                outputs.len()
            ),
            "apply_registered_function",
        ));
    }

    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_function() -> TorshResult<()> {
        let input = Tensor::from_data(vec![2.0, 3.0, 4.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let square_fn = SquareFunction;

        let outputs = apply_custom_function(square_fn, &[input.clone()])?;
        let output_data = outputs[0].to_vec()?;

        assert!((output_data[0] - 4.0).abs() < 1e-6);
        assert!((output_data[1] - 9.0).abs() < 1e-6);
        assert!((output_data[2] - 16.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_exp_function() -> TorshResult<()> {
        let input = Tensor::from_data(vec![0.0, 1.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let exp_fn = ExpFunction;

        let outputs = apply_custom_function(exp_fn, &[input.clone()])?;
        let output_data = outputs[0].to_vec()?;

        assert!((output_data[0] - 1.0).abs() < 1e-6);
        assert!((output_data[1] - std::f32::consts::E).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_scaled_add_function() -> TorshResult<()> {
        let a = Tensor::from_data(vec![1.0, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let b = Tensor::from_data(vec![3.0, 4.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let scaled_add_fn = ScaledAddFunction::new(2.0);

        let outputs = apply_custom_function_with_context(scaled_add_fn, &[a, b])?;
        let output_data = outputs[0].to_vec()?;

        // 2 * 1 + 3 = 5, 2 * 2 + 4 = 8
        assert!((output_data[0] - 5.0).abs() < 1e-6);
        assert!((output_data[1] - 8.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_registry() -> TorshResult<()> {
        let mut registry = AutogradRegistry::new();
        registry.register("square".to_string(), SquareFunction);

        let function = registry.get("square").unwrap();
        assert_eq!(function.name(), "square");
        assert_eq!(function.num_inputs(), 1);
        assert_eq!(function.num_outputs(), 1);

        Ok(())
    }

    #[test]
    fn test_global_registry() -> TorshResult<()> {
        register_custom_function("test_square".to_string(), SquareFunction);

        let input = Tensor::from_data(vec![3.0], vec![1], torsh_core::DeviceType::Cpu)?;
        let outputs = apply_registered_function("test_square", &[input])?;
        let output_data = outputs[0].to_vec()?;

        assert!((output_data[0] - 9.0).abs() < 1e-6);

        Ok(())
    }
}
