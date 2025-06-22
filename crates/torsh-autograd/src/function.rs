//! Autograd function definitions leveraging scirs2

use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;
// use scirs2::autograd::{self as ag};
use std::any::Any;

/// Trait for custom autograd functions
pub trait Function {
    /// Forward pass
    fn forward<T>(
        &self,
        ctx: &mut FunctionContext,
        inputs: &[&Tensor<T>],
    ) -> Result<Vec<Tensor<T>>>
    where
        T: torsh_core::dtype::TensorElement;

    /// Backward pass
    fn backward<T>(
        &self,
        ctx: &mut FunctionContext,
        grad_outputs: &[&Tensor<T>],
    ) -> Result<Vec<Option<Tensor<T>>>>
    where
        T: torsh_core::dtype::TensorElement;

    /// Name of the function for debugging
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// Context for storing values between forward and backward passes
pub struct FunctionContext {
    /// Saved tensors
    saved_tensors: Vec<Box<dyn Any>>,
    /// Saved values
    saved_values: Vec<Box<dyn Any>>,
    /// Whether to materialize gradients for non-differentiable tensors
    #[allow(dead_code)]
    materialize_grads: bool,
}

impl Default for FunctionContext {
    fn default() -> Self {
        Self {
            saved_tensors: Vec::new(),
            saved_values: Vec::new(),
            materialize_grads: true,
        }
    }
}

impl FunctionContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Save tensors for backward pass
    pub fn save_for_backward<T>(&mut self, tensors: &[&Tensor<T>])
    where
        T: torsh_core::dtype::TensorElement + 'static,
    {
        for tensor in tensors {
            self.saved_tensors.push(Box::new((*tensor).clone()));
        }
    }

    /// Get saved tensors
    pub fn saved_tensors<T>(&self) -> Result<Vec<Tensor<T>>>
    where
        T: torsh_core::dtype::TensorElement + 'static,
    {
        self.saved_tensors
            .iter()
            .map(|t| {
                t.downcast_ref::<Tensor<T>>()
                    .ok_or_else(|| {
                        TorshError::AutogradError("Type mismatch in saved tensors".to_string())
                    })
                    .cloned()
            })
            .collect()
    }

    /// Save arbitrary values
    pub fn save_value<V: Any + 'static>(&mut self, value: V) {
        self.saved_values.push(Box::new(value));
    }

    /// Get saved value
    pub fn get_saved_value<V: Any + 'static>(&self, index: usize) -> Result<&V> {
        self.saved_values
            .get(index)
            .and_then(|v| v.downcast_ref::<V>())
            .ok_or_else(|| {
                TorshError::AutogradError("Saved value not found or type mismatch".to_string())
            })
    }
}

/// Note: scirs2 doesn't have a high-level Function trait interface
/// This would need to be implemented differently to work with scirs2's API
/// For now, we'll provide placeholder implementations
/// Apply a custom function
pub fn apply_function<F, T>(function: F, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>>
where
    F: Function + 'static,
    T: torsh_core::dtype::TensorElement,
{
    // Note: scirs2 doesn't have a high-level custom function API
    // This would need to be implemented using lower-level operations
    // For now, return a placeholder error
    let _ = (function, inputs); // Suppress unused warnings
    Err(TorshError::AutogradError(
        "Custom function application not yet implemented with current scirs2 API".to_string(),
    ))
}

/// Common autograd functions implemented using scirs2
pub mod functions {
    use super::*;
    // Note: scirs2 doesn't have a high-level functions module
    // We'll implement basic functions using tensor_ops

    /// ReLU function
    pub struct ReLU;

    impl Function for ReLU {
        fn forward<T>(
            &self,
            ctx: &mut FunctionContext,
            inputs: &[&Tensor<T>],
        ) -> Result<Vec<Tensor<T>>>
        where
            T: torsh_core::dtype::TensorElement,
        {
            if inputs.len() != 1 {
                return Err(TorshError::AutogradError(
                    "ReLU expects exactly one input".to_string(),
                ));
            }

            // Save input for backward
            ctx.save_for_backward(inputs);

            // Note: scirs2 doesn't have high-level relu function
            // This would need to be implemented using tensor_ops
            // For now, return a placeholder
            Ok(vec![inputs[0].clone()])
        }

        fn backward<T>(
            &self,
            ctx: &mut FunctionContext,
            grad_outputs: &[&Tensor<T>],
        ) -> Result<Vec<Option<Tensor<T>>>>
        where
            T: torsh_core::dtype::TensorElement,
        {
            let saved = ctx.saved_tensors::<T>()?;
            let _input = &saved[0];

            // Note: scirs2 doesn't have high-level comparison and arithmetic functions
            // This would need to be implemented using tensor_ops
            // For now, return a placeholder
            Ok(vec![Some(grad_outputs[0].clone())])
        }

        fn name(&self) -> &str {
            "ReLU"
        }
    }

    /// Dropout function
    pub struct Dropout {
        p: f32,
        training: bool,
    }

    impl Dropout {
        pub fn new(p: f32, training: bool) -> Self {
            Self { p, training }
        }
    }

    impl Function for Dropout {
        fn forward<T>(
            &self,
            _ctx: &mut FunctionContext,
            inputs: &[&Tensor<T>],
        ) -> Result<Vec<Tensor<T>>>
        where
            T: torsh_core::dtype::TensorElement,
        {
            if !self.training || self.p == 0.0 {
                return Ok(vec![inputs[0].clone()]);
            }

            // Note: scirs2 doesn't have high-level bernoulli, mul, div_scalar functions
            // This would need to be implemented using tensor_ops
            // For now, return input unchanged
            Ok(vec![inputs[0].clone()])
        }

        fn backward<T>(
            &self,
            _ctx: &mut FunctionContext,
            grad_outputs: &[&Tensor<T>],
        ) -> Result<Vec<Option<Tensor<T>>>>
        where
            T: torsh_core::dtype::TensorElement,
        {
            if !self.training || self.p == 0.0 {
                return Ok(vec![Some(grad_outputs[0].clone())]);
            }

            // Note: scirs2 doesn't have high-level mul, div_scalar functions
            // This would need to be implemented using tensor_ops
            // For now, return gradient unchanged
            Ok(vec![Some(grad_outputs[0].clone())])
        }

        fn name(&self) -> &str {
            "Dropout"
        }
    }
}
