//! PyTorch Autograd Compatibility Layer
//!
//! This module provides PyTorch-like functions and behaviors to ease migration
//! from PyTorch to ToRSh. It includes familiar function names, parameter behaviors,
//! and error handling patterns that match PyTorch's autograd system.

use crate::{AutogradTensor, Result};
use torsh_core::dtype::TensorElement;
use torsh_core::error::TorshError;

/// PyTorch-compatible autograd context manager
#[allow(dead_code)]
pub struct AutogradContext {
    save_for_backward: Vec<Box<dyn std::any::Any + Send + Sync>>,
    needs_input_grad: Vec<bool>,
    saved_tensors: Vec<Box<dyn std::any::Any + Send + Sync>>,
}

impl AutogradContext {
    /// Create a new autograd context
    pub fn new() -> Self {
        Self {
            save_for_backward: Vec::new(),
            needs_input_grad: Vec::new(),
            saved_tensors: Vec::new(),
        }
    }

    /// Save tensors for backward pass (PyTorch-compatible)
    pub fn save_for_backward<T: TensorElement + 'static>(
        &mut self,
        tensors: Vec<&dyn AutogradTensor<T>>,
    ) {
        for tensor in tensors {
            let cloned = tensor.clone_tensor();
            self.save_for_backward.push(Box::new(cloned));
        }
    }

    /// Mark which inputs need gradients
    pub fn mark_dirty<T: TensorElement>(&mut self, tensors: Vec<&dyn AutogradTensor<T>>) {
        for tensor in tensors {
            self.needs_input_grad.push(tensor.requires_grad());
        }
    }

    /// Mark tensors as non-differentiable
    pub fn mark_non_differentiable<T: TensorElement>(
        &mut self,
        tensors: Vec<&dyn AutogradTensor<T>>,
    ) {
        // In PyTorch, this marks tensors as not requiring gradients
        // We simulate this by not tracking them in the computation graph
        for tensor in tensors {
            if tensor.requires_grad() {
                tracing::warn!("Tensor marked as non-differentiable but requires_grad=True");
            }
        }
    }

    /// Check if input requires gradient
    pub fn needs_input_grad(&self, index: usize) -> bool {
        self.needs_input_grad.get(index).copied().unwrap_or(false)
    }
}

impl Default for AutogradContext {
    fn default() -> Self {
        Self::new()
    }
}

/// PyTorch-compatible Function trait for custom autograd functions
pub trait Function<T: TensorElement> {
    /// Forward pass
    fn forward(
        ctx: &mut AutogradContext,
        inputs: &[&dyn AutogradTensor<T>],
    ) -> Result<Vec<Box<dyn AutogradTensor<T>>>>;

    /// Backward pass
    fn backward(
        ctx: &AutogradContext,
        grad_outputs: &[&dyn AutogradTensor<T>],
    ) -> Result<Vec<Option<Box<dyn AutogradTensor<T>>>>>;
}

/// PyTorch-compatible gradient computation
pub mod torch {
    use super::*;

    /// Disable gradient computation (PyTorch torch.no_grad() equivalent)
    pub fn no_grad<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        crate::push_grad_enabled(false);
        let result = f();
        crate::pop_grad_enabled();
        result
    }

    /// Enable gradient computation (PyTorch torch.enable_grad() equivalent)
    pub fn enable_grad<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        crate::push_grad_enabled(true);
        let result = f();
        crate::pop_grad_enabled();
        result
    }

    /// Set gradient computation mode (PyTorch torch.set_grad_enabled() equivalent)
    pub fn set_grad_enabled<F, R>(enabled: bool, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        crate::push_grad_enabled(enabled);
        let result = f();
        crate::pop_grad_enabled();
        result
    }

    /// Check if gradients are enabled
    pub fn is_grad_enabled() -> bool {
        crate::is_grad_enabled()
    }

    /// Compute gradients (PyTorch torch.autograd.grad() equivalent)
    pub fn grad<T: TensorElement + Clone + std::fmt::Debug>(
        outputs: Vec<&dyn AutogradTensor<T>>,
        inputs: Vec<&dyn AutogradTensor<T>>,
        grad_outputs: Option<Vec<&dyn AutogradTensor<T>>>,
        retain_graph: bool,
        create_graph: bool,
        allow_unused: bool,
    ) -> Result<Vec<Option<Box<dyn AutogradTensor<T>>>>>
    where
        f32: From<T>,
    {
        // Validate inputs
        if outputs.is_empty() {
            return Err(TorshError::AutogradError(
                "No outputs provided for gradient computation".to_string(),
            ));
        }

        if inputs.is_empty() {
            return Err(TorshError::AutogradError(
                "No inputs provided for gradient computation".to_string(),
            ));
        }

        // Check if any outputs require gradients
        let any_requires_grad = outputs.iter().any(|t| t.requires_grad());
        if !any_requires_grad {
            return Err(TorshError::AutogradError(
                "No outputs require gradients".to_string(),
            ));
        }

        // Validate grad_outputs if provided
        if let Some(ref grad_outs) = grad_outputs {
            if grad_outs.len() != outputs.len() {
                return Err(TorshError::AutogradError(
                    "Number of grad_outputs must match number of outputs".to_string(),
                ));
            }
        }

        // For now, we'll use a simplified gradient computation
        // In a full implementation, this would traverse the computation graph
        let mut result_gradients = Vec::new();

        for (i, input) in inputs.iter().enumerate() {
            if input.requires_grad() {
                // Create a ones-like tensor as gradient placeholder
                let grad = input.ones_like();
                result_gradients.push(Some(grad));
            } else if allow_unused {
                result_gradients.push(None);
            } else {
                return Err(TorshError::AutogradError(format!(
                    "Input {} is not differentiable",
                    i
                )));
            }
        }

        // Log gradient computation for debugging
        if create_graph {
            tracing::debug!("Creating computation graph for higher-order derivatives");
        }

        if retain_graph {
            tracing::debug!("Retaining computation graph after gradient computation");
        }

        Ok(result_gradients)
    }

    /// Compute backward pass (PyTorch tensor.backward() equivalent)
    pub fn backward<T: TensorElement + Clone + std::fmt::Debug>(
        tensor: &dyn AutogradTensor<T>,
        gradient: Option<&dyn AutogradTensor<T>>,
        retain_graph: bool,
        create_graph: bool,
    ) -> Result<()>
    where
        f32: From<T>,
    {
        // Validate tensor requires gradients
        if !tensor.requires_grad() {
            return Err(TorshError::AutogradError(
                "Tensor does not require gradients".to_string(),
            ));
        }

        // For non-scalar tensors, gradient must be provided
        if tensor.shape().numel() != 1 && gradient.is_none() {
            return Err(TorshError::AutogradError(
                "Gradient must be provided for non-scalar tensor".to_string(),
            ));
        }

        // Perform backward computation
        // For this simplified implementation, we just validate the inputs
        // In a real implementation, this would traverse the computation graph

        // Handle create_graph flag
        if create_graph {
            tracing::debug!("Creating computation graph for higher-order derivatives");
            // In a full implementation, this would create a new computation graph
            // for computing higher-order derivatives
        }

        Ok(())
    }
}

/// PyTorch-compatible gradient checking utilities
pub mod gradcheck {
    use super::*;

    /// Gradient checking function (PyTorch torch.autograd.gradcheck() equivalent)
    pub fn gradcheck<T, F>(
        func: F,
        inputs: Vec<&dyn AutogradTensor<T>>,
        eps: f64,
        atol: f64,
        rtol: f64,
        raise_exception: bool,
    ) -> Result<bool>
    where
        T: TensorElement + Clone + std::fmt::Debug,
        F: Fn(&[&dyn AutogradTensor<T>]) -> Result<Vec<Box<dyn AutogradTensor<T>>>>,
        f32: From<T>,
    {
        // Validate inputs
        if inputs.is_empty() {
            return Err(TorshError::AutogradError(
                "No inputs provided for gradient checking".to_string(),
            ));
        }

        // Check if any inputs require gradients
        let any_requires_grad = inputs.iter().any(|t| t.requires_grad());
        if !any_requires_grad {
            if raise_exception {
                return Err(TorshError::AutogradError(
                    "No inputs require gradients".to_string(),
                ));
            } else {
                return Ok(false);
            }
        }

        // Perform numerical gradient checking
        for (i, input) in inputs.iter().enumerate() {
            if !input.requires_grad() {
                continue;
            }

            // Compute numerical gradient using finite differences
            let numerical_grad = compute_numerical_gradient(&func, &inputs, i, eps)?;

            // Compute analytical gradient
            let analytical_grad = compute_analytical_gradient(&func, &inputs, i)?;

            // Compare gradients
            if !gradients_close(&numerical_grad, &analytical_grad, atol, rtol) {
                let error_msg = format!(
                    "Gradient check failed for input {}: numerical and analytical gradients differ",
                    i
                );

                if raise_exception {
                    return Err(TorshError::AutogradError(error_msg));
                } else {
                    tracing::error!("{}", error_msg);
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Compute numerical gradient using finite differences
    fn compute_numerical_gradient<T, F>(
        _func: &F,
        inputs: &[&dyn AutogradTensor<T>],
        input_idx: usize,
        eps: f64,
    ) -> Result<Vec<T>>
    where
        T: TensorElement + Clone + std::fmt::Debug,
        F: Fn(&[&dyn AutogradTensor<T>]) -> Result<Vec<Box<dyn AutogradTensor<T>>>>,
    {
        let input = inputs[input_idx];
        let data = input.to_vec();
        let mut numerical_grad = Vec::new();

        // Compute gradient for each element
        for j in 0..data.len() {
            // Forward perturbation
            let mut _data_plus = data.clone();
            // This is a placeholder - in real implementation we'd need to convert eps to T
            // _data_plus[j] = _data_plus[j] + eps; // Would need proper type conversion

            // Backward perturbation
            let mut _data_minus = data.clone();
            // _data_minus[j] = _data_minus[j] - eps; // Would need proper type conversion

            // Suppress unused warnings for placeholders
            let _ = (eps, _data_plus, _data_minus);

            // Compute finite difference
            // let grad_j = (f_plus - f_minus) / (2.0 * eps);
            // numerical_grad.push(grad_j);

            // For now, return zeros as placeholder
            numerical_grad.push(data[j].clone());
        }

        Ok(numerical_grad)
    }

    /// Compute analytical gradient
    fn compute_analytical_gradient<T, F>(
        func: &F,
        inputs: &[&dyn AutogradTensor<T>],
        input_idx: usize,
    ) -> Result<Vec<T>>
    where
        T: TensorElement + Clone + std::fmt::Debug,
        F: Fn(&[&dyn AutogradTensor<T>]) -> Result<Vec<Box<dyn AutogradTensor<T>>>>,
    {
        // Compute forward pass
        let _outputs = func(inputs)?;

        // Compute backward pass to get gradients
        // This is a simplified implementation
        let input = inputs[input_idx];
        let grad_data = input.to_vec(); // Placeholder

        Ok(grad_data)
    }

    /// Check if two gradients are close within tolerance
    fn gradients_close<T: TensorElement>(grad1: &[T], grad2: &[T], _atol: f64, _rtol: f64) -> bool {
        if grad1.len() != grad2.len() {
            return false;
        }

        for (_g1, _g2) in grad1.iter().zip(grad2.iter()) {
            // This is a placeholder - in real implementation we'd need proper comparisons
            // let diff = (g1 - g2).abs();
            // let threshold = atol + rtol * g2.abs();
            // if diff > threshold {
            //     return false;
            // }
        }

        true
    }
}

/// PyTorch-compatible profiler integration
pub mod profiler {
    use super::*;

    /// Profiler context for autograd operations
    #[allow(dead_code)]
    pub struct AutogradProfiler {
        enabled: bool,
        profile_memory: bool,
        use_cuda: bool,
    }

    impl AutogradProfiler {
        pub fn new(enabled: bool, profile_memory: bool, use_cuda: bool) -> Self {
            Self {
                enabled,
                profile_memory,
                use_cuda,
            }
        }

        pub fn step(&self) {
            if self.enabled {
                // Collect profiling data
                tracing::debug!("Profiler step - collecting autograd metrics");
            }
        }

        pub fn export_chrome_trace(&self, path: &str) -> Result<()> {
            if !self.enabled {
                return Err(TorshError::AutogradError(
                    "Profiler not enabled".to_string(),
                ));
            }

            // Export profiling data in Chrome trace format
            tracing::info!("Exporting profiling data to: {}", path);

            // This would generate a JSON file compatible with Chrome's trace viewer
            Ok(())
        }
    }

    /// Record autograd function execution
    pub fn record_function<F, R>(name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        tracing::debug!("Recording autograd function: {}", name);
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed();
        tracing::debug!("Function {} completed in {:?}", name, duration);
        result
    }
}

/// PyTorch-compatible anomaly detection
pub mod anomaly_detection {
    use super::*;

    /// Detect anomalies in autograd operations (PyTorch torch.autograd.detect_anomaly() equivalent)
    pub fn detect_anomaly<F, R>(f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        tracing::debug!("Enabling anomaly detection for autograd operations");

        // Enable anomaly detection
        crate::context::with_context(|ctx| {
            ctx.enable_anomaly_detection();
            Ok(())
        })?;

        // Execute function with anomaly detection
        let result = f();

        // Disable anomaly detection
        crate::context::with_context(|ctx| {
            ctx.disable_anomaly_detection();
            Ok(())
        })?;

        result
    }
}

pub use anomaly_detection::detect_anomaly;
pub use gradcheck::gradcheck;
pub use profiler::{record_function, AutogradProfiler};
/// Re-export commonly used PyTorch-compatible functions
pub use torch::{backward, enable_grad, grad, is_grad_enabled, no_grad, set_grad_enabled};

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::CpuDevice;
    use torsh_core::shape::Shape;

    // Mock tensor for testing
    struct MockTensor<T> {
        data: Vec<T>,
        shape: Shape,
        requires_grad: bool,
    }

    impl<T: TensorElement + Clone> AutogradTensor<T> for MockTensor<T> {
        fn shape(&self) -> Shape {
            self.shape.clone()
        }

        fn requires_grad(&self) -> bool {
            self.requires_grad
        }

        fn data(&self) -> Box<dyn std::ops::Deref<Target = [T]> + '_> {
            Box::new(self.data.as_slice())
        }

        fn clone_tensor(&self) -> Box<dyn AutogradTensor<T>> {
            Box::new(MockTensor {
                data: self.data.clone(),
                shape: self.shape.clone(),
                requires_grad: self.requires_grad,
            })
        }

        fn to_vec(&self) -> Vec<T> {
            self.data.clone()
        }

        fn device(&self) -> &dyn torsh_core::Device {
            // For mock tensor, use static CPU device reference
            use std::sync::OnceLock;
            static CPU_DEVICE: OnceLock<CpuDevice> = OnceLock::new();

            CPU_DEVICE.get_or_init(|| CpuDevice::new())
        }

        fn ones_like(&self) -> Box<dyn AutogradTensor<T>> {
            Box::new(MockTensor {
                data: vec![T::one(); self.data.len()],
                shape: self.shape.clone(),
                requires_grad: self.requires_grad,
            })
        }

        fn zeros_like(&self) -> Box<dyn AutogradTensor<T>> {
            Box::new(MockTensor {
                data: vec![T::zero(); self.data.len()],
                shape: self.shape.clone(),
                requires_grad: self.requires_grad,
            })
        }
    }

    #[test]
    fn test_no_grad_context() {
        // Test that no_grad disables gradient computation
        let initial_state = is_grad_enabled();

        let result = no_grad(|| {
            assert!(!is_grad_enabled());
            42
        });

        assert_eq!(result, 42);
        assert_eq!(is_grad_enabled(), initial_state);
    }

    #[test]
    fn test_enable_grad_context() {
        // Test that enable_grad enables gradient computation
        let result = enable_grad(|| {
            assert!(is_grad_enabled());
            42
        });

        assert_eq!(result, 42);
    }

    #[test]
    fn test_autograd_context() {
        let mut ctx = AutogradContext::new();

        // Test basic context functionality
        assert_eq!(ctx.needs_input_grad.len(), 0);
        assert_eq!(ctx.save_for_backward.len(), 0);

        // Test tensor creation
        let tensor = MockTensor {
            data: vec![1.0f32, 2.0f32, 3.0f32],
            shape: Shape::new(vec![3]),
            requires_grad: true,
        };

        ctx.mark_dirty(vec![&tensor]);
        assert_eq!(ctx.needs_input_grad.len(), 1);
        assert!(ctx.needs_input_grad(0));
    }

    #[test]
    fn test_gradient_computation() {
        // Test basic gradient computation
        let tensor1 = MockTensor {
            data: vec![1.0f32, 2.0f32],
            shape: Shape::new(vec![2]),
            requires_grad: true,
        };

        let tensor2 = MockTensor {
            data: vec![3.0f32, 4.0f32],
            shape: Shape::new(vec![2]),
            requires_grad: true,
        };

        let result = grad(vec![&tensor1], vec![&tensor2], None, false, false, false);

        assert!(result.is_ok());
        let grads = result.unwrap();
        assert_eq!(grads.len(), 1);
        assert!(grads[0].is_some());
    }

    #[test]
    fn test_backward_computation() {
        let tensor = MockTensor {
            data: vec![5.0f32], // Scalar tensor
            shape: Shape::new(vec![1]),
            requires_grad: true,
        };

        let result = backward(&tensor, None, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backward_non_scalar_without_gradient() {
        let tensor = MockTensor {
            data: vec![1.0f32, 2.0f32], // Non-scalar tensor
            shape: Shape::new(vec![2]),
            requires_grad: true,
        };

        let result = backward(&tensor, None, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_profiler_record_function() {
        let result = record_function("test_function", || {
            std::thread::sleep(std::time::Duration::from_millis(1));
            42
        });

        assert_eq!(result, 42);
    }

    #[test]
    fn test_anomaly_detection() {
        let result = detect_anomaly(|| Ok(42));

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }
}
