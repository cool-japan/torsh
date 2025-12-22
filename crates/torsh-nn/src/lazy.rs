//! Lazy initialization utilities for neural network modules
//!
//! This module provides infrastructure for lazy initialization of neural network
//! layers, where parameters are only created when the first forward pass is made
//! and the input shape is known.

use crate::{Module, Parameter};
use parking_lot::Mutex;
use torsh_core::{
    device::DeviceType,
    error::{Result, TorshError},
    shape::Shape,
};
use torsh_tensor::Tensor;

// SciRS2 policy compliance for random number generation
use scirs2_core::random::{Random, Rng};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{boxed::Box, collections::HashMap, string::String, sync::Arc, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Trait for modules that support lazy initialization
pub trait LazyModule: Module {
    /// Check if the module has been initialized
    fn is_initialized(&self) -> bool;

    /// Initialize the module with the given input shape
    fn initialize(&mut self, input_shape: &Shape) -> Result<()>;

    /// Get the expected input shape after initialization
    fn input_shape(&self) -> Option<Shape>;

    /// Get the expected output shape (if known)
    fn output_shape(&self, input_shape: &Shape) -> Option<Shape>;
}

/// Lazy initialization state
#[derive(Debug, Clone)]
pub enum LazyState {
    /// Module is not initialized yet
    Uninitialized,
    /// Module is currently being initialized (to prevent recursion)
    Initializing,
    /// Module is fully initialized with the given input shape
    Initialized { input_shape: Shape },
}

/// Lazy module wrapper that defers initialization until first forward pass
pub struct LazyWrapper<M> {
    /// The wrapped module
    module: Option<M>,
    /// Initialization state
    state: Arc<Mutex<LazyState>>,
    /// Factory function to create the module
    #[allow(dead_code)]
    factory: Option<Box<dyn Fn(&Shape) -> Result<M> + Send + Sync>>,
    /// Training mode
    training: bool,
    /// Device
    device: DeviceType,
}

impl<M: Module + Send + Sync> LazyWrapper<M> {
    /// Create a new lazy wrapper with a factory function
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn(&Shape) -> Result<M> + Send + Sync + 'static,
    {
        Self {
            module: None,
            state: Arc::new(Mutex::new(LazyState::Uninitialized)),
            factory: Some(Box::new(factory)),
            training: true,
            device: DeviceType::Cpu,
        }
    }

    /// Create a new lazy wrapper from an existing module that implements LazyModule
    pub fn from_lazy_module(module: M) -> Self
    where
        M: LazyModule,
    {
        let state = if module.is_initialized() {
            // If already initialized, we need the input shape
            if let Some(input_shape) = module.input_shape() {
                LazyState::Initialized { input_shape }
            } else {
                LazyState::Uninitialized
            }
        } else {
            LazyState::Uninitialized
        };

        Self {
            module: Some(module),
            state: Arc::new(Mutex::new(state)),
            factory: None,
            training: true,
            device: DeviceType::Cpu,
        }
    }

    /// Initialize the wrapped module if not already initialized
    #[allow(dead_code)]
    fn ensure_initialized(&mut self, input_shape: &Shape) -> Result<()> {
        let mut state = self.state.lock();

        match &*state {
            LazyState::Initialized { .. } => {
                // Already initialized
                return Ok(());
            }
            LazyState::Initializing => {
                return Err(TorshError::RuntimeError(
                    "Circular dependency in lazy module initialization".to_string(),
                ));
            }
            LazyState::Uninitialized => {
                // Need to initialize
            }
        }

        // Mark as initializing
        *state = LazyState::Initializing;
        drop(state);

        // Initialize the module
        let result = if let Some(ref factory) = self.factory {
            // Create module using factory
            match factory(input_shape) {
                Ok(mut module) => {
                    // Set training mode and device to match wrapper
                    if self.training {
                        module.train();
                    } else {
                        module.eval();
                    }
                    module.to_device(self.device)?;

                    self.module = Some(module);
                    Ok(())
                }
                Err(e) => Err(e),
            }
        } else if let Some(ref mut _module) = self.module {
            // Initialize existing LazyModule
            // Since we can't downcast to trait objects, we'll just return Ok for now
            // In a real implementation, this would need to handle specific types
            Ok(())
        } else {
            Err(TorshError::RuntimeError(
                "No module or factory available for initialization".to_string(),
            ))
        };

        // Update state based on result
        let mut state = self.state.lock();
        match result {
            Ok(()) => {
                *state = LazyState::Initialized {
                    input_shape: input_shape.clone(),
                };
            }
            Err(_) => {
                *state = LazyState::Uninitialized;
            }
        }

        result
    }

    /// Get a reference to the wrapped module (if initialized)
    pub fn module(&self) -> Option<&M> {
        self.module.as_ref()
    }

    /// Get a mutable reference to the wrapped module (if initialized)
    pub fn module_mut(&mut self) -> Option<&mut M> {
        self.module.as_mut()
    }

    /// Check if the module is initialized
    pub fn is_initialized(&self) -> bool {
        matches!(&*self.state.lock(), LazyState::Initialized { .. })
    }

    /// Get the input shape used for initialization (if initialized)
    pub fn input_shape(&self) -> Option<Shape> {
        if let LazyState::Initialized { input_shape } = &*self.state.lock() {
            Some(input_shape.clone())
        } else {
            None
        }
    }
}

impl<M: Module + Send + Sync> Module for LazyWrapper<M> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let _input_shape = input.shape();

        // Ensure we're initialized (need mutable access for this)
        // This is a limitation of the current Module trait design
        // In practice, this would need to be handled differently
        if !self.is_initialized() {
            return Err(TorshError::RuntimeError(
                "LazyWrapper requires mutable access for initialization during forward pass. Consider initializing explicitly.".to_string()
            ));
        }

        if let Some(ref module) = self.module {
            module.forward(input)
        } else {
            Err(TorshError::RuntimeError(
                "Module not initialized".to_string(),
            ))
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        if let Some(ref module) = self.module {
            module.parameters()
        } else {
            HashMap::new()
        }
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        if let Some(ref module) = self.module {
            module.named_parameters()
        } else {
            HashMap::new()
        }
    }

    fn training(&self) -> bool {
        self.training
    }

    fn train(&mut self) {
        self.training = true;
        if let Some(ref mut module) = self.module {
            module.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        if let Some(ref mut module) = self.module {
            module.eval();
        }
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        if let Some(ref mut module) = self.module {
            module.set_training(training);
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.device = device;
        if let Some(ref mut module) = self.module {
            module.to_device(device)
        } else {
            Ok(())
        }
    }

    fn name(&self) -> Option<&str> {
        if let Some(ref module) = self.module {
            module.name()
        } else {
            Some("LazyWrapper")
        }
    }

    fn children(&self) -> Vec<&dyn Module> {
        if let Some(ref module) = self.module {
            vec![module as &dyn Module]
        } else {
            Vec::new()
        }
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut module) = self.module {
            module.zero_grad();
        }
    }

    fn extra_repr(&self) -> String {
        let state_str = match &*self.state.lock() {
            LazyState::Uninitialized => "uninitialized".to_string(),
            LazyState::Initializing => "initializing...".to_string(),
            LazyState::Initialized { input_shape } => {
                format!("initialized(input_shape={:?})", input_shape.dims())
            }
        };

        if let Some(ref module) = self.module {
            format!("LazyWrapper({}): {}", state_str, module.extra_repr())
        } else {
            format!("LazyWrapper({})", state_str)
        }
    }
}

/// Lazy Linear layer that determines its input size from the first forward pass
#[derive(Debug)]
pub struct LazyLinear {
    /// Output features
    out_features: usize,
    /// Whether to use bias
    bias: bool,
    /// Weight parameter (initialized lazily)
    weight: Option<Parameter>,
    /// Bias parameter (initialized lazily)
    bias_param: Option<Parameter>,
    /// Initialization state
    state: Arc<Mutex<LazyState>>,
    /// Training mode
    training: bool,
    /// Device
    device: DeviceType,
}

impl LazyLinear {
    /// Create a new lazy linear layer
    pub fn new(out_features: usize, bias: bool) -> Self {
        Self {
            out_features,
            bias,
            weight: None,
            bias_param: None,
            state: Arc::new(Mutex::new(LazyState::Uninitialized)),
            training: true,
            device: DeviceType::Cpu,
        }
    }

    /// Create with default bias=true
    pub fn with_features(out_features: usize) -> Self {
        Self::new(out_features, true)
    }
}

impl LazyModule for LazyLinear {
    fn is_initialized(&self) -> bool {
        matches!(&*self.state.lock(), LazyState::Initialized { .. })
    }

    fn initialize(&mut self, input_shape: &Shape) -> Result<()> {
        let mut state = self.state.lock();

        if matches!(&*state, LazyState::Initialized { .. }) {
            return Ok(());
        }

        if matches!(&*state, LazyState::Initializing) {
            return Err(TorshError::RuntimeError(
                "Circular dependency in LazyLinear initialization".to_string(),
            ));
        }

        *state = LazyState::Initializing;
        drop(state);

        // Determine input features from the last dimension
        let dims = input_shape.dims();
        if dims.is_empty() {
            return Err(TorshError::InvalidShape(
                "Input tensor must have at least 1 dimension for LazyLinear".to_string(),
            ));
        }

        let in_features = dims[dims.len() - 1];

        // Initialize weight: (out_features, in_features)
        // Apply Xavier/Glorot uniform initialization
        // Xavier: uniform(-sqrt(6/(in_features + out_features)), sqrt(6/(in_features + out_features)))
        let bound = (6.0 / (in_features + self.out_features) as f32).sqrt();

        // Create random data using SciRS2
        let mut rng = Random::seed(0);
        let weight_data: Vec<f32> = (0..self.out_features * in_features)
            .map(|_| rng.random::<f32>() * 2.0 * bound - bound)
            .collect();

        let weight_tensor = Tensor::from_data(
            weight_data,
            vec![self.out_features, in_features],
            self.device,
        )
        .unwrap();

        self.weight = Some(Parameter::new(weight_tensor));

        // Initialize bias if needed
        if self.bias {
            let bias_data = vec![0.0f32; self.out_features];
            let bias_tensor =
                Tensor::from_data(bias_data, vec![self.out_features], self.device).unwrap();

            // Initialize bias to zero (common practice)
            self.bias_param = Some(Parameter::new(bias_tensor));
        }

        // Update state
        let mut state = self.state.lock();
        *state = LazyState::Initialized {
            input_shape: input_shape.clone(),
        };

        Ok(())
    }

    fn input_shape(&self) -> Option<Shape> {
        if let LazyState::Initialized { input_shape } = &*self.state.lock() {
            Some(input_shape.clone())
        } else {
            None
        }
    }

    fn output_shape(&self, input_shape: &Shape) -> Option<Shape> {
        let dims = input_shape.dims();
        if dims.is_empty() {
            return None;
        }

        // Output shape: (..., out_features)
        let mut output_dims = dims.to_vec();
        let last_idx = output_dims.len() - 1;
        output_dims[last_idx] = self.out_features;

        Some(Shape::new(output_dims))
    }
}

impl Module for LazyLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.is_initialized() {
            return Err(TorshError::RuntimeError(
                "LazyLinear not initialized. Call initialize() or use LazyWrapper.".to_string(),
            ));
        }

        let weight = self.weight.as_ref().unwrap();
        let weight_tensor = weight.tensor().read().clone();

        // Perform linear transformation: input @ weight.T + bias
        let mut output = input.matmul(&weight_tensor.transpose(0, 1)?)?;

        if let Some(ref bias_param) = self.bias_param {
            let bias_tensor = bias_param.tensor().read().clone();
            output = output.add_op(&bias_tensor)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if let Some(ref weight) = self.weight {
            params.insert("weight".to_string(), weight.clone());
        }

        if let Some(ref bias) = self.bias_param {
            params.insert("bias".to_string(), bias.clone());
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.device = device;

        if let Some(ref mut weight) = self.weight {
            let weight_tensor = weight.tensor().read().clone().to(device)?;
            *weight.tensor().write() = weight_tensor;
        }

        if let Some(ref mut bias) = self.bias_param {
            let bias_tensor = bias.tensor().read().clone().to(device)?;
            *bias.tensor().write() = bias_tensor;
        }

        Ok(())
    }

    fn name(&self) -> Option<&str> {
        Some("LazyLinear")
    }

    fn extra_repr(&self) -> String {
        let initialized = if self.is_initialized() {
            if let Some(input_shape) = self.input_shape() {
                let dims = input_shape.dims();
                let in_features = dims[dims.len() - 1];
                format!("in_features={}, ", in_features)
            } else {
                "".to_string()
            }
        } else {
            "uninitialized, ".to_string()
        };

        format!(
            "{}out_features={}, bias={}",
            initialized, self.out_features, self.bias
        )
    }
}

/// Convenience function to create a lazy linear layer
pub fn lazy_linear(out_features: usize) -> LazyLinear {
    LazyLinear::new(out_features, true)
}

/// Convenience function to create a lazy linear layer without bias
pub fn lazy_linear_no_bias(out_features: usize) -> LazyLinear {
    LazyLinear::new(out_features, false)
}

/// Macro for creating lazy modules with factory functions
#[macro_export]
macro_rules! lazy_module {
    ($factory:expr) => {
        $crate::lazy::LazyWrapper::new($factory)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Linear;
    use torsh_tensor::Tensor;

    #[test]
    fn test_lazy_linear_initialization() {
        let mut lazy_linear = LazyLinear::new(10, true);

        // Should not be initialized initially
        assert!(!lazy_linear.is_initialized());
        assert!(lazy_linear.input_shape().is_none());

        // Initialize with input shape
        let input_shape = Shape::new(vec![32, 20]); // batch_size=32, in_features=20
        lazy_linear.initialize(&input_shape).unwrap();

        // Should be initialized now
        assert!(lazy_linear.is_initialized());
        assert_eq!(lazy_linear.input_shape().unwrap().dims(), &[32, 20]);

        // Should have parameters now
        let params = lazy_linear.parameters();
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));

        // Weight should have correct shape: (out_features, in_features)
        let weight = params.get("weight").unwrap();
        let weight_binding = weight.tensor();
        let weight_tensor = weight_binding.read();
        assert_eq!(weight_tensor.shape().dims(), &[10, 20]);

        // Bias should have correct shape: (out_features,)
        let bias = params.get("bias").unwrap();
        let bias_binding = bias.tensor();
        let bias_tensor = bias_binding.read();
        assert_eq!(bias_tensor.shape().dims(), &[10]);
    }

    #[test]
    fn test_lazy_linear_forward() {
        let mut lazy_linear = LazyLinear::new(5, true);

        // Create input tensor
        let input = Tensor::ones(&[2, 10], DeviceType::Cpu).unwrap();

        // Initialize
        lazy_linear.initialize(&input.shape()).unwrap();

        // Forward pass
        let output = lazy_linear.forward(&input).unwrap();

        // Output should have correct shape: (batch_size, out_features)
        assert_eq!(output.shape().dims(), &[2, 5]);
    }

    #[test]
    fn test_lazy_wrapper_with_factory() {
        let lazy_wrapper = LazyWrapper::new(|input_shape: &Shape| {
            let dims = input_shape.dims();
            let in_features = dims[dims.len() - 1];
            Ok(Linear::new(in_features, 8, true))
        });

        assert!(!lazy_wrapper.is_initialized());

        // Note: In practice, you'd need mutable access to initialize during forward
        // This test shows the structure but can't test the full forward pass
        // due to the immutable forward() method in the Module trait
    }

    #[test]
    fn test_output_shape_prediction() {
        let lazy_linear = LazyLinear::new(7, false);

        let input_shape = Shape::new(vec![16, 32, 15]);
        let output_shape = lazy_linear.output_shape(&input_shape).unwrap();

        // Should change last dimension to out_features
        assert_eq!(output_shape.dims(), &[16, 32, 7]);
    }

    #[test]
    fn test_extra_repr() {
        let mut lazy_linear = LazyLinear::new(12, true);

        // Before initialization
        let repr_before = lazy_linear.extra_repr();
        assert!(repr_before.contains("uninitialized"));
        assert!(repr_before.contains("out_features=12"));
        assert!(repr_before.contains("bias=true"));

        // After initialization
        let input_shape = Shape::new(vec![8, 24]);
        lazy_linear.initialize(&input_shape).unwrap();

        let repr_after = lazy_linear.extra_repr();
        assert!(repr_after.contains("in_features=24"));
        assert!(repr_after.contains("out_features=12"));
        assert!(!repr_after.contains("uninitialized"));
    }
}
