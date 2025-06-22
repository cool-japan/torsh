//! Automatic differentiation engine for ToRSh
//!
//! This crate provides a PyTorch-compatible autograd API that fully leverages
//! scirs2-autograd's powerful automatic differentiation capabilities.

pub mod context;
pub mod function;
pub mod grad_mode;

use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;
// Temporarily disable scirs2 integration until API compatibility is resolved
// use scirs2::autograd::{self as ag, VariableEnvironment, Context, Tensor as AgTensor};
// use scirs2::autograd::tensor_ops as T;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

// Global variable environment for managing tensor gradients
// Note: Temporarily disabled due to scirs2 API incompatibility
// static VARIABLE_ENV: once_cell::sync::Lazy<Arc<RwLock<VariableEnvironment<f32>>>> =
//     once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(VariableEnvironment::new())));

/// Global gradient mode state
static GRAD_MODE: once_cell::sync::Lazy<Arc<RwLock<GradMode>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(GradMode::new())));

/// Gradient mode manager
struct GradMode {
    enabled: bool,
    /// Stack of gradient mode states for nested contexts
    mode_stack: Vec<bool>,
}

impl GradMode {
    fn new() -> Self {
        Self {
            enabled: true,
            mode_stack: Vec::new(),
        }
    }

    fn push(&mut self, enabled: bool) {
        self.mode_stack.push(self.enabled);
        self.enabled = enabled;
    }

    fn pop(&mut self) {
        if let Some(prev) = self.mode_stack.pop() {
            self.enabled = prev;
        }
    }
}

/// Check if gradients are enabled
pub fn is_grad_enabled() -> bool {
    GRAD_MODE.read().enabled
}

/// Set gradient computation mode
pub fn set_grad_enabled(enabled: bool) {
    GRAD_MODE.write().enabled = enabled;
}

/// Perform backward pass on a tensor using scirs2's autograd
pub fn backward<T>(
    tensor: &Tensor<T>,
    gradient: Option<&Tensor<T>>,
    retain_graph: bool,
) -> Result<()>
where
    T: torsh_core::dtype::TensorElement,
{
    if !tensor.requires_grad() {
        return Err(TorshError::AutogradError(
            "Called backward on tensor that doesn't require grad".to_string(),
        ));
    }

    // For non-scalar outputs, gradient must be provided
    if tensor.shape().numel() != 1 && gradient.is_none() {
        return Err(TorshError::AutogradError(
            "Gradient must be provided for non-scalar outputs".to_string(),
        ));
    }

    // Temporarily disabled - would use scirs2's autograd when API is compatible
    let _ = (gradient, retain_graph); // Suppress unused warnings

    // TODO: Implement proper scirs2 integration when API compatibility is resolved
    Err(TorshError::AutogradError(
        "Backward pass not yet implemented - scirs2 integration pending".to_string(),
    ))
}

/// Compute gradient of outputs with respect to inputs using scirs2
pub fn grad<T>(
    outputs: &[&Tensor<T>],
    inputs: &[&Tensor<T>],
    grad_outputs: Option<&[&Tensor<T>]>,
    retain_graph: bool,
    create_graph: bool,
) -> Result<Vec<Option<Tensor<T>>>>
where
    T: torsh_core::dtype::TensorElement,
{
    // Validate inputs
    if outputs.is_empty() {
        return Err(TorshError::AutogradError(
            "grad requires at least one output".to_string(),
        ));
    }

    if inputs.is_empty() {
        return Err(TorshError::AutogradError(
            "grad requires at least one input".to_string(),
        ));
    }

    if let Some(grad_outs) = grad_outputs {
        if grad_outs.len() != outputs.len() {
            return Err(TorshError::AutogradError(
                "Number of grad_outputs must match outputs".to_string(),
            ));
        }
    }

    // Temporarily disabled - would use scirs2's autograd when API is compatible
    let _ = (outputs, inputs, grad_outputs, retain_graph, create_graph); // Suppress unused warnings

    // TODO: Implement proper scirs2 integration when API compatibility is resolved
    Err(TorshError::AutogradError(
        "Gradient computation not yet implemented - scirs2 integration pending".to_string(),
    ))
}

/// Context manager for disabling gradient computation
pub struct NoGradGuard;

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl NoGradGuard {
    pub fn new() -> Self {
        GRAD_MODE.write().push(false);

        // Note: scirs2 doesn't have global set_grad_enabled function
        // Gradient mode is controlled through context

        Self
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_MODE.write().pop();

        // Note: scirs2 gradient mode is context-based
        // Global state is managed by ToRSh
    }
}

/// Disable gradient computation
pub fn no_grad() -> NoGradGuard {
    NoGradGuard::new()
}

/// Context manager for enabling gradient computation
pub struct EnableGradGuard;

impl Default for EnableGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl EnableGradGuard {
    pub fn new() -> Self {
        GRAD_MODE.write().push(true);

        // Note: scirs2 doesn't have global set_grad_enabled function
        // Gradient mode is controlled through context

        Self
    }
}

impl Drop for EnableGradGuard {
    fn drop(&mut self) {
        GRAD_MODE.write().pop();

        // Note: scirs2 gradient mode is context-based
        // Global state is managed by ToRSh
    }
}

/// Enable gradient computation
pub fn enable_grad() -> EnableGradGuard {
    EnableGradGuard::new()
}

/// Advanced autograd functions leveraging scirs2
pub mod functional {
    use super::*;

    /// Compute Jacobian matrix
    pub fn jacobian<T>(
        func: impl Fn(&Tensor<T>) -> Result<Tensor<T>>,
        inputs: &Tensor<T>,
        create_graph: bool,
    ) -> Result<Tensor<T>>
    where
        T: torsh_core::dtype::TensorElement,
    {
        // Note: scirs2 doesn't have high-level jacobian function
        // This would need to be implemented using T::grad in a loop
        // For now, return a placeholder error
        let _ = (func, inputs, create_graph); // Suppress unused warnings
        Err(TorshError::AutogradError(
            "Jacobian computation not yet implemented with current scirs2 API".to_string(),
        ))
    }

    /// Compute Hessian matrix
    pub fn hessian<T>(
        func: impl Fn(&Tensor<T>) -> Result<Tensor<T>>,
        inputs: &Tensor<T>,
        create_graph: bool,
    ) -> Result<Tensor<T>>
    where
        T: torsh_core::dtype::TensorElement,
    {
        // Note: scirs2 doesn't have high-level hessian function
        // This would need to be implemented using repeated T::grad calls
        // For now, return a placeholder error
        let _ = (func, inputs, create_graph); // Suppress unused warnings
        Err(TorshError::AutogradError(
            "Hessian computation not yet implemented with current scirs2 API".to_string(),
        ))
    }

    /// Vector-Jacobian product (vjp)
    pub fn vjp<T>(
        func: impl Fn(&Tensor<T>) -> Result<Tensor<T>>,
        inputs: &Tensor<T>,
        v: &Tensor<T>,
        create_graph: bool,
    ) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: torsh_core::dtype::TensorElement,
    {
        // Note: scirs2 doesn't have high-level vjp function
        // This would need to be implemented using T::grad
        // For now, return a placeholder error
        let _ = (v, create_graph); // Suppress unused warnings
        let _output = func(inputs)?;
        Err(TorshError::AutogradError(
            "VJP computation not yet implemented with current scirs2 API".to_string(),
        ))
    }

    /// Jacobian-vector product (jvp)
    pub fn jvp<T>(
        func: impl Fn(&Tensor<T>) -> Result<Tensor<T>>,
        inputs: &Tensor<T>,
        v: &Tensor<T>,
        create_graph: bool,
    ) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: torsh_core::dtype::TensorElement,
    {
        // Note: scirs2 doesn't have high-level jvp function
        // This would need to be implemented using forward-mode differentiation
        // For now, return a placeholder error
        let _ = (v, create_graph); // Suppress unused warnings
        let _output = func(inputs)?;
        Err(TorshError::AutogradError(
            "JVP computation not yet implemented with current scirs2 API".to_string(),
        ))
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::functional::{hessian, jacobian, jvp, vjp};
    pub use crate::{
        backward, enable_grad, grad, is_grad_enabled, no_grad, set_grad_enabled, EnableGradGuard,
        NoGradGuard,
    };

    // Re-export key scirs2 autograd functionality
    // Temporarily disabled due to API compatibility issues
    // pub use scirs2::autograd::{
    //     VariableEnvironment,
    //     tensor_ops as autograd_ops,
    // };
}

/// Gradient accumulation utilities
pub mod accumulate {
    use super::*;

    /// Accumulate gradients across multiple backward passes
    pub struct GradientAccumulator {
        // Temporarily use f32 placeholders instead of AgTensor
        accumulated_grads: HashMap<String, f32>,
        num_accumulations: usize,
    }

    impl Default for GradientAccumulator {
        fn default() -> Self {
            Self {
                accumulated_grads: HashMap::new(),
                num_accumulations: 0,
            }
        }
    }

    impl GradientAccumulator {
        pub fn new() -> Self {
            Self {
                accumulated_grads: HashMap::new(),
                num_accumulations: 0,
            }
        }

        /// Accumulate gradients from current backward pass
        pub fn accumulate(&mut self) {
            // Temporarily disabled - would use scirs2's variable environment
            // TODO: Implement proper gradient accumulation when scirs2 integration is ready

            self.num_accumulations += 1;
        }

        /// Get averaged gradients
        pub fn average(&self) -> HashMap<String, f32> {
            if self.num_accumulations == 0 {
                return HashMap::new();
            }

            let divisor = self.num_accumulations as f32;

            self.accumulated_grads
                .iter()
                .map(|(name, grad)| (name.clone(), grad / divisor))
                .collect()
        }

        /// Reset accumulator
        pub fn reset(&mut self) {
            self.accumulated_grads.clear();
            self.num_accumulations = 0;
        }
    }
}

/// Checkpointing utilities for memory-efficient training
pub mod checkpoint {
    use super::*;

    /// Checkpoint a function to save memory during backward pass
    pub fn checkpoint<T, F>(func: F, inputs: &[Tensor<T>]) -> Result<Vec<Tensor<T>>>
    where
        T: torsh_core::dtype::TensorElement,
        F: Fn(&[Tensor<T>]) -> Result<Vec<Tensor<T>>>,
    {
        // Temporarily disabled - would use scirs2's checkpointing when available
        // For now, just call the function directly
        func(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_mode() {
        assert!(is_grad_enabled());

        {
            let _guard = no_grad();
            assert!(!is_grad_enabled());

            {
                let _inner_guard = enable_grad();
                assert!(is_grad_enabled());
            }

            assert!(!is_grad_enabled());
        }

        assert!(is_grad_enabled());
    }

    #[test]
    fn test_gradient_accumulator() {
        let acc = accumulate::GradientAccumulator::new();

        // Test that accumulator starts empty
        let initial_grads = acc.average();
        assert!(initial_grads.is_empty());

        // Would test accumulation with actual gradients in integration tests
    }
}
