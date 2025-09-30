//! Lazy initialization layers that infer input dimensions from first forward pass

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{collections::HashMap, sync::Mutex};

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
#[cfg(not(feature = "std"))]
use parking_lot::Mutex;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// LazyLinear layer that infers input features from first forward pass
pub struct LazyLinear {
    base: ModuleBase,
    out_features: usize,
    use_bias: bool,
    /// Tracks whether the layer has been initialized
    initialized: Mutex<bool>,
    /// Stores the inferred input features after first forward pass
    in_features: Mutex<Option<usize>>,
}

impl LazyLinear {
    /// Create a new lazy linear layer
    pub fn new(out_features: usize, bias: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            out_features,
            use_bias: bias,
            initialized: Mutex::new(false),
            in_features: Mutex::new(None),
        }
    }

    /// Initialize the layer with the given input features
    pub fn initialize_with_features(&mut self, in_features: usize) -> Result<()> {
        // Check if already initialized
        if self.is_initialized() {
            return Ok(());
        }

        // Initialize weight with shape [in_features, out_features]
        let weight = crate::init::xavier_uniform(&[in_features, self.out_features])?;
        self.base
            .register_parameter("weight".to_string(), Parameter::new(weight));

        if self.use_bias {
            let bias_tensor = zeros(&[self.out_features]).unwrap();
            self.base
                .register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }

        // Store the inferred in_features
        *self.in_features.lock().unwrap() = Some(in_features);
        *self.initialized.lock().unwrap() = true;

        Ok(())
    }

    /// Check if the layer is initialized
    pub fn is_initialized(&self) -> bool {
        *self.initialized.lock().unwrap()
    }

    /// Get the inferred input features (None if not initialized)
    pub fn in_features(&self) -> Option<usize> {
        *self.in_features.lock().unwrap()
    }
}

impl Module for LazyLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Check if we need to initialize
        if !self.is_initialized() {
            // Infer in_features from input shape
            let shape = input.shape();
            let input_shape = shape.dims();
            if input_shape.is_empty() {
                return Err(torsh_core::error::TorshError::InvalidShape(
                    "LazyLinear requires at least 1D input".to_string(),
                ));
            }

            let in_features = input_shape[input_shape.len() - 1];

            // Return error with helpful message about initialization
            return Err(torsh_core::error::TorshError::Other(
                format!("LazyLinear not initialized. Detected in_features={}. Call initialize_lazy({}) first.", in_features, in_features)
            ));
        }

        // Perform standard linear transformation
        let weight = self.base.parameters["weight"].tensor().read().clone();
        let output = input.matmul(&weight)?;

        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            Ok(output.add_op(&bias)?)
        } else {
            Ok(output)
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

/// LazyConv1d layer that infers input channels from first forward pass
pub struct LazyConv1d {
    base: ModuleBase,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    use_bias: bool,
    initialized: Mutex<bool>,
    in_channels: Mutex<Option<usize>>,
}

impl LazyConv1d {
    pub fn new(
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias: bias,
            initialized: Mutex::new(false),
            in_channels: Mutex::new(None),
        }
    }

    fn initialize(&mut self, in_channels: usize) -> Result<()> {
        // Validate groups
        if in_channels % self.groups != 0 || self.out_channels % self.groups != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "in_channels ({}) and out_channels ({}) must be divisible by groups ({})",
                in_channels, self.out_channels, self.groups
            )));
        }

        // Initialize weight
        let weight_shape = vec![
            self.out_channels,
            in_channels / self.groups,
            self.kernel_size,
        ];
        let weight = crate::init::xavier_uniform(&weight_shape)?;
        self.base
            .register_parameter("weight".to_string(), Parameter::new(weight));

        if self.use_bias {
            let bias = zeros(&[self.out_channels])?;
            self.base
                .register_parameter("bias".to_string(), Parameter::new(bias));
        }

        *self.in_channels.lock().unwrap() = Some(in_channels);
        *self.initialized.lock().unwrap() = true;

        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        *self.initialized.lock().unwrap()
    }

    pub fn in_channels(&self) -> Option<usize> {
        *self.in_channels.lock().unwrap()
    }
}

impl Module for LazyConv1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.is_initialized() {
            let shape = input.shape();
            let input_shape = shape.dims();
            if input_shape.len() < 2 {
                return Err(torsh_core::error::TorshError::InvalidShape(
                    "LazyConv1d requires at least 2D input [batch, channels, ...]".to_string(),
                ));
            }

            let in_channels = input_shape[1];

            return Err(torsh_core::error::TorshError::Other(
                format!("LazyConv1d not initialized. Detected in_channels={}. Call initialize_lazy() first.", in_channels)
            ));
        }

        // Use the conv1d implementation from torsh_tensor
        let weight = self.base.parameters["weight"].tensor().read().clone();
        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            input.conv1d(
                &weight,
                Some(&bias),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        } else {
            input.conv1d(
                &weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
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

/// LazyConv2d layer that infers input channels from first forward pass
pub struct LazyConv2d {
    base: ModuleBase,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    use_bias: bool,
    initialized: Mutex<bool>,
    in_channels: Mutex<Option<usize>>,
}

impl LazyConv2d {
    pub fn new(
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        bias: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias: bias,
            initialized: Mutex::new(false),
            in_channels: Mutex::new(None),
        }
    }

    fn initialize(&mut self, in_channels: usize) -> Result<()> {
        // Validate groups
        if in_channels % self.groups != 0 || self.out_channels % self.groups != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "in_channels ({}) and out_channels ({}) must be divisible by groups ({})",
                in_channels, self.out_channels, self.groups
            )));
        }

        // Initialize weight
        let weight_shape = vec![
            self.out_channels,
            in_channels / self.groups,
            self.kernel_size.0,
            self.kernel_size.1,
        ];
        let weight = crate::init::xavier_uniform(&weight_shape)?;
        self.base
            .register_parameter("weight".to_string(), Parameter::new(weight));

        if self.use_bias {
            let bias = zeros(&[self.out_channels])?;
            self.base
                .register_parameter("bias".to_string(), Parameter::new(bias));
        }

        *self.in_channels.lock().unwrap() = Some(in_channels);
        *self.initialized.lock().unwrap() = true;

        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        *self.initialized.lock().unwrap()
    }

    pub fn in_channels(&self) -> Option<usize> {
        *self.in_channels.lock().unwrap()
    }
}

impl Module for LazyConv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.is_initialized() {
            let shape = input.shape();
            let input_shape = shape.dims();
            if input_shape.len() < 3 {
                return Err(torsh_core::error::TorshError::InvalidShape(
                    "LazyConv2d requires at least 3D input [batch, channels, height, ...]"
                        .to_string(),
                ));
            }

            let in_channels = input_shape[1];

            return Err(torsh_core::error::TorshError::Other(
                format!("LazyConv2d not initialized. Detected in_channels={}. Call initialize_lazy() first.", in_channels)
            ));
        }

        // Use the conv2d implementation from torsh_tensor
        let weight = self.base.parameters["weight"].tensor().read().clone();
        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            input.conv2d(
                &weight,
                Some(&bias),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        } else {
            input.conv2d(
                &weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
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

/// Extension trait for lazy initialization
pub trait LazyModule: Module {
    /// Initialize the lazy module with inferred dimensions from input
    fn initialize_lazy(&mut self, input: &Tensor) -> Result<()>;
}

impl LazyModule for LazyLinear {
    fn initialize_lazy(&mut self, input: &Tensor) -> Result<()> {
        if self.is_initialized() {
            return Ok(());
        }

        let shape = input.shape();
        let input_shape = shape.dims();
        if input_shape.is_empty() {
            return Err(torsh_core::error::TorshError::InvalidShape(
                "LazyLinear requires at least 1D input".to_string(),
            ));
        }

        let in_features = input_shape[input_shape.len() - 1];
        self.initialize_with_features(in_features)
    }
}

impl LazyModule for LazyConv1d {
    fn initialize_lazy(&mut self, input: &Tensor) -> Result<()> {
        if self.is_initialized() {
            return Ok(());
        }

        let shape = input.shape();
        let input_shape = shape.dims();
        if input_shape.len() < 2 {
            return Err(torsh_core::error::TorshError::InvalidShape(
                "LazyConv1d requires at least 2D input [batch, channels, ...]".to_string(),
            ));
        }

        let in_channels = input_shape[1];
        self.initialize(in_channels)
    }
}

impl LazyModule for LazyConv2d {
    fn initialize_lazy(&mut self, input: &Tensor) -> Result<()> {
        if self.is_initialized() {
            return Ok(());
        }

        let shape = input.shape();
        let input_shape = shape.dims();
        if input_shape.len() < 3 {
            return Err(torsh_core::error::TorshError::InvalidShape(
                "LazyConv2d requires at least 3D input [batch, channels, height, ...]".to_string(),
            ));
        }

        let in_channels = input_shape[1];
        self.initialize(in_channels)
    }
}

// Convenience constructors for common cases
impl LazyConv1d {
    /// Create LazyConv1d with default stride=1, padding=0, dilation=1, groups=1
    pub fn simple(out_channels: usize, kernel_size: usize, bias: bool) -> Self {
        Self::new(out_channels, kernel_size, 1, 0, 1, 1, bias)
    }
}

impl LazyConv2d {
    /// Create LazyConv2d with default stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
    pub fn simple(out_channels: usize, kernel_size: (usize, usize), bias: bool) -> Self {
        Self::new(out_channels, kernel_size, (1, 1), (0, 0), (1, 1), 1, bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_lazy_linear_initialization() -> Result<()> {
        let mut lazy_linear = LazyLinear::new(10, true);
        assert!(!lazy_linear.is_initialized());
        assert_eq!(lazy_linear.in_features(), None);

        // Create input tensor
        let input = randn::<f32>(&[32, 20])?;

        // Initialize the layer
        lazy_linear.initialize_lazy(&input)?;

        assert!(lazy_linear.is_initialized());
        assert_eq!(lazy_linear.in_features(), Some(20));
        assert_eq!(lazy_linear.parameters().len(), 2); // weight + bias

        Ok(())
    }

    #[test]
    fn test_lazy_conv1d_initialization() -> Result<()> {
        let mut lazy_conv = LazyConv1d::simple(16, 3, true);
        assert!(!lazy_conv.is_initialized());

        // Create input tensor [batch, channels, length]
        let input = randn::<f32>(&[8, 32, 100])?;

        // Initialize the layer
        lazy_conv.initialize_lazy(&input)?;

        assert!(lazy_conv.is_initialized());
        assert_eq!(lazy_conv.in_channels(), Some(32));

        Ok(())
    }

    #[test]
    fn test_lazy_conv2d_initialization() -> Result<()> {
        let mut lazy_conv = LazyConv2d::simple(64, (3, 3), false);
        assert!(!lazy_conv.is_initialized());

        // Create input tensor [batch, channels, height, width]
        let input = randn::<f32>(&[4, 3, 224, 224])?;

        // Initialize the layer
        lazy_conv.initialize_lazy(&input)?;

        assert!(lazy_conv.is_initialized());
        assert_eq!(lazy_conv.in_channels(), Some(3));
        assert_eq!(lazy_conv.parameters().len(), 1); // weight only, no bias

        Ok(())
    }
}
