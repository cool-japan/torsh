//! Pooling layers

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// 2D max pooling layer
pub struct MaxPool2d {
    base: ModuleBase,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    dilation: (usize, usize),
    ceil_mode: bool,
}

impl MaxPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
        dilation: (usize, usize),
        ceil_mode: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        }
    }

    pub fn with_kernel_size(kernel_size: usize) -> Self {
        Self::new((kernel_size, kernel_size), None, (0, 0), (1, 1), false)
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Max pooling implementation
        // Input shape: [batch_size, channels, height, width]
        let binding = input.shape();
        let input_shape = binding.dims();
        let stride = self.stride.unwrap_or(self.kernel_size);

        let output_height = if self.ceil_mode {
            ((input_shape[2] + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                as f32
                / stride.0 as f32)
                .ceil() as usize
                + 1
        } else {
            (input_shape[2] + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                / stride.0
                + 1
        };

        let output_width = if self.ceil_mode {
            ((input_shape[3] + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                as f32
                / stride.1 as f32)
                .ceil() as usize
                + 1
        } else {
            (input_shape[3] + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                / stride.1
                + 1
        };

        let output_shape = [input_shape[0], input_shape[1], output_height, output_width];

        // Placeholder implementation - real max pooling would be implemented in backend
        let output = zeros(&output_shape)?;
        Ok(output)
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

/// 2D average pooling layer
pub struct AvgPool2d {
    base: ModuleBase,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    ceil_mode: bool,
    #[allow(dead_code)]
    count_include_pad: bool,
}

impl AvgPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
        }
    }

    pub fn with_kernel_size(kernel_size: usize) -> Self {
        Self::new((kernel_size, kernel_size), None, (0, 0), false, true)
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Average pooling implementation
        let binding = input.shape();
        let input_shape = binding.dims();
        let stride = self.stride.unwrap_or(self.kernel_size);

        let output_height = if self.ceil_mode {
            ((input_shape[2] + 2 * self.padding.0 - self.kernel_size.0) as f32 / stride.0 as f32)
                .ceil() as usize
                + 1
        } else {
            (input_shape[2] + 2 * self.padding.0 - self.kernel_size.0) / stride.0 + 1
        };

        let output_width = if self.ceil_mode {
            ((input_shape[3] + 2 * self.padding.1 - self.kernel_size.1) as f32 / stride.1 as f32)
                .ceil() as usize
                + 1
        } else {
            (input_shape[3] + 2 * self.padding.1 - self.kernel_size.1) / stride.1 + 1
        };

        let output_shape = [input_shape[0], input_shape[1], output_height, output_width];

        // Placeholder implementation
        let output = zeros(&output_shape)?;
        Ok(output)
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

/// Adaptive 2D average pooling layer
pub struct AdaptiveAvgPool2d {
    base: ModuleBase,
    output_size: (Option<usize>, Option<usize>),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (Option<usize>, Option<usize>)) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
        }
    }

    pub fn with_output_size(output_size: usize) -> Self {
        Self::new((Some(output_size), Some(output_size)))
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Adaptive average pooling implementation
        let binding = input.shape();
        let input_shape = binding.dims();

        // Defensive bounds checking - ensure we have at least 4 dimensions
        if input_shape.len() < 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                format!(
                    "AdaptiveAvgPool2d expects 4D input (batch_size, channels, height, width), got {}D: {:?}",
                    input_shape.len(),
                    input_shape
                )
            ));
        }

        let output_height = self.output_size.0.unwrap_or(input_shape[2]);
        let output_width = self.output_size.1.unwrap_or(input_shape[3]);

        let output_shape = [input_shape[0], input_shape[1], output_height, output_width];

        // Placeholder implementation
        let output = zeros(&output_shape)?;
        Ok(output)
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

impl std::fmt::Debug for MaxPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxPool2d")
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .finish()
    }
}

impl std::fmt::Debug for AvgPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AvgPool2d")
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .finish()
    }
}

impl std::fmt::Debug for AdaptiveAvgPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveAvgPool2d")
            .field("output_size", &self.output_size)
            .finish()
    }
}

/// LP pooling 1D layer
pub struct LPPool1d {
    base: ModuleBase,
    norm_type: f32,
    kernel_size: usize,
    stride: Option<usize>,
    ceil_mode: bool,
}

impl LPPool1d {
    pub fn new(norm_type: f32, kernel_size: usize, stride: Option<usize>, ceil_mode: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            norm_type,
            kernel_size,
            stride,
            ceil_mode,
        }
    }
}

impl Module for LPPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let stride = self.stride.unwrap_or(self.kernel_size);

        let output_length = if self.ceil_mode {
            ((input_shape[2] - self.kernel_size) as f32 / stride as f32).ceil() as usize + 1
        } else {
            (input_shape[2] - self.kernel_size) / stride + 1
        };

        let output_shape = [input_shape[0], input_shape[1], output_length];

        // Placeholder implementation - would need proper LP pooling
        let output = zeros(&output_shape)?;
        Ok(output)
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

/// LP pooling 2D layer
pub struct LPPool2d {
    base: ModuleBase,
    norm_type: f32,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    ceil_mode: bool,
}

impl LPPool2d {
    pub fn new(
        norm_type: f32,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        ceil_mode: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            norm_type,
            kernel_size,
            stride,
            ceil_mode,
        }
    }
}

impl Module for LPPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let stride = self.stride.unwrap_or(self.kernel_size);

        let output_height = if self.ceil_mode {
            ((input_shape[2] - self.kernel_size.0) as f32 / stride.0 as f32).ceil() as usize + 1
        } else {
            (input_shape[2] - self.kernel_size.0) / stride.0 + 1
        };

        let output_width = if self.ceil_mode {
            ((input_shape[3] - self.kernel_size.1) as f32 / stride.1 as f32).ceil() as usize + 1
        } else {
            (input_shape[3] - self.kernel_size.1) / stride.1 + 1
        };

        let output_shape = [input_shape[0], input_shape[1], output_height, output_width];

        // Placeholder implementation - would need proper LP pooling
        let output = zeros(&output_shape)?;
        Ok(output)
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

/// Fractional max pooling 1D layer
pub struct FractionalMaxPool1d {
    base: ModuleBase,
    kernel_size: usize,
    output_size: Option<usize>,
    output_ratio: Option<f32>,
    return_indices: bool,
}

impl FractionalMaxPool1d {
    pub fn new(
        kernel_size: usize,
        output_size: Option<usize>,
        output_ratio: Option<f32>,
        return_indices: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            output_size,
            output_ratio,
            return_indices,
        }
    }

    pub fn with_kernel_size(kernel_size: usize) -> Self {
        Self::new(kernel_size, None, Some(0.5), false)
    }

    pub fn with_output_ratio(kernel_size: usize, output_ratio: f32) -> Self {
        Self::new(kernel_size, None, Some(output_ratio), false)
    }

    /// Forward pass returning both output and optionally indices
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();

        let output_length = if let Some(ol) = self.output_size {
            ol
        } else if let Some(r) = self.output_ratio {
            (input_shape[2] as f32 * r) as usize
        } else {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Either output_size or output_ratio must be specified".to_string(),
            ));
        };

        let output_shape = [input_shape[0], input_shape[1], output_length];

        // Placeholder implementation - would need proper fractional max pooling
        let output = zeros(&output_shape)?;

        let indices = if self.return_indices {
            Some(zeros(&output_shape)?) // Placeholder for indices
        } else {
            None
        };

        Ok((output, indices))
    }
}

impl Module for FractionalMaxPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_with_indices(input)?;
        Ok(output)
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

/// Fractional max pooling 2D layer  
pub struct FractionalMaxPool2d {
    base: ModuleBase,
    kernel_size: (usize, usize),
    output_size: Option<(usize, usize)>,
    output_ratio: Option<(f32, f32)>,
    return_indices: bool,
}

impl FractionalMaxPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        output_size: Option<(usize, usize)>,
        output_ratio: Option<(f32, f32)>,
        return_indices: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            output_size,
            output_ratio,
            return_indices,
        }
    }

    pub fn with_kernel_size(kernel_size: (usize, usize)) -> Self {
        Self::new(kernel_size, None, Some((0.5, 0.5)), false)
    }

    pub fn with_output_ratio(kernel_size: (usize, usize), output_ratio: (f32, f32)) -> Self {
        Self::new(kernel_size, None, Some(output_ratio), false)
    }

    /// Forward pass returning both output and optionally indices
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();

        let (output_height, output_width) = if let Some((oh, ow)) = self.output_size {
            (oh, ow)
        } else if let Some((rh, rw)) = self.output_ratio {
            (
                (input_shape[2] as f32 * rh) as usize,
                (input_shape[3] as f32 * rw) as usize,
            )
        } else {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Either output_size or output_ratio must be specified".to_string(),
            ));
        };

        let output_shape = [input_shape[0], input_shape[1], output_height, output_width];

        // Placeholder implementation - would need proper fractional max pooling
        let output = zeros(&output_shape)?;

        let indices = if self.return_indices {
            Some(zeros(&output_shape)?) // Placeholder for indices
        } else {
            None
        };

        Ok((output, indices))
    }
}

impl Module for FractionalMaxPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_with_indices(input)?;
        Ok(output)
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

/// Fractional max pooling 3D layer
pub struct FractionalMaxPool3d {
    base: ModuleBase,
    kernel_size: (usize, usize, usize),
    output_size: Option<(usize, usize, usize)>,
    output_ratio: Option<(f32, f32, f32)>,
    return_indices: bool,
}

impl FractionalMaxPool3d {
    pub fn new(
        kernel_size: (usize, usize, usize),
        output_size: Option<(usize, usize, usize)>,
        output_ratio: Option<(f32, f32, f32)>,
        return_indices: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            output_size,
            output_ratio,
            return_indices,
        }
    }

    pub fn with_kernel_size(kernel_size: (usize, usize, usize)) -> Self {
        Self::new(kernel_size, None, Some((0.5, 0.5, 0.5)), false)
    }

    pub fn with_output_ratio(
        kernel_size: (usize, usize, usize),
        output_ratio: (f32, f32, f32),
    ) -> Self {
        Self::new(kernel_size, None, Some(output_ratio), false)
    }

    /// Forward pass returning both output and optionally indices
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();

        let (output_depth, output_height, output_width) =
            if let Some((od, oh, ow)) = self.output_size {
                (od, oh, ow)
            } else if let Some((rd, rh, rw)) = self.output_ratio {
                (
                    (input_shape[2] as f32 * rd) as usize,
                    (input_shape[3] as f32 * rh) as usize,
                    (input_shape[4] as f32 * rw) as usize,
                )
            } else {
                return Err(torsh_core::error::TorshError::InvalidArgument(
                    "Either output_size or output_ratio must be specified".to_string(),
                ));
            };

        let output_shape = [
            input_shape[0],
            input_shape[1],
            output_depth,
            output_height,
            output_width,
        ];

        // Placeholder implementation - would need proper fractional max pooling
        let output = zeros(&output_shape)?;

        let indices = if self.return_indices {
            Some(zeros(&output_shape)?) // Placeholder for indices
        } else {
            None
        };

        Ok((output, indices))
    }
}

impl Module for FractionalMaxPool3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_with_indices(input)?;
        Ok(output)
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

impl std::fmt::Debug for LPPool1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LPPool1d")
            .field("norm_type", &self.norm_type)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .finish()
    }
}

impl std::fmt::Debug for LPPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LPPool2d")
            .field("norm_type", &self.norm_type)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .finish()
    }
}

impl std::fmt::Debug for FractionalMaxPool1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FractionalMaxPool1d")
            .field("kernel_size", &self.kernel_size)
            .field("output_size", &self.output_size)
            .field("output_ratio", &self.output_ratio)
            .field("return_indices", &self.return_indices)
            .finish()
    }
}

impl std::fmt::Debug for FractionalMaxPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FractionalMaxPool2d")
            .field("kernel_size", &self.kernel_size)
            .field("output_size", &self.output_size)
            .field("output_ratio", &self.output_ratio)
            .field("return_indices", &self.return_indices)
            .finish()
    }
}

impl std::fmt::Debug for FractionalMaxPool3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FractionalMaxPool3d")
            .field("kernel_size", &self.kernel_size)
            .field("output_size", &self.output_size)
            .field("output_ratio", &self.output_ratio)
            .field("return_indices", &self.return_indices)
            .finish()
    }
}

/// Adaptive 2D max pooling layer
pub struct AdaptiveMaxPool2d {
    base: ModuleBase,
    output_size: (Option<usize>, Option<usize>),
    return_indices: bool,
}

impl AdaptiveMaxPool2d {
    pub fn new(output_size: (Option<usize>, Option<usize>), return_indices: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
            return_indices,
        }
    }

    pub fn with_output_size(output_size: usize) -> Self {
        Self::new((Some(output_size), Some(output_size)), false)
    }

    /// Forward pass returning both output and optionally indices
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();

        let output_height = self.output_size.0.unwrap_or(input_shape[2]);
        let output_width = self.output_size.1.unwrap_or(input_shape[3]);

        let output_shape = [input_shape[0], input_shape[1], output_height, output_width];

        // Placeholder implementation - real adaptive max pooling would compute optimal kernel sizes and strides
        let output = zeros(&output_shape)?;

        let indices = if self.return_indices {
            Some(zeros(&output_shape)?) // Placeholder for indices
        } else {
            None
        };

        Ok((output, indices))
    }
}

impl Module for AdaptiveMaxPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_with_indices(input)?;
        Ok(output)
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

/// Adaptive 1D average pooling layer
pub struct AdaptiveAvgPool1d {
    base: ModuleBase,
    output_size: usize,
}

impl AdaptiveAvgPool1d {
    pub fn new(output_size: usize) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
        }
    }
}

impl Module for AdaptiveAvgPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();

        let output_shape = [input_shape[0], input_shape[1], self.output_size];

        // Placeholder implementation
        let output = zeros(&output_shape)?;
        Ok(output)
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

/// Adaptive 1D max pooling layer
pub struct AdaptiveMaxPool1d {
    base: ModuleBase,
    output_size: usize,
    return_indices: bool,
}

impl AdaptiveMaxPool1d {
    pub fn new(output_size: usize, return_indices: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
            return_indices,
        }
    }

    /// Forward pass returning both output and optionally indices
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();

        let output_shape = [input_shape[0], input_shape[1], self.output_size];

        // Placeholder implementation
        let output = zeros(&output_shape)?;

        let indices = if self.return_indices {
            Some(zeros(&output_shape)?) // Placeholder for indices
        } else {
            None
        };

        Ok((output, indices))
    }
}

impl Module for AdaptiveMaxPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_with_indices(input)?;
        Ok(output)
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

/// Adaptive 3D average pooling layer
pub struct AdaptiveAvgPool3d {
    base: ModuleBase,
    output_size: (Option<usize>, Option<usize>, Option<usize>),
}

impl AdaptiveAvgPool3d {
    pub fn new(output_size: (Option<usize>, Option<usize>, Option<usize>)) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
        }
    }

    pub fn with_output_size(output_size: usize) -> Self {
        Self::new((Some(output_size), Some(output_size), Some(output_size)))
    }
}

impl Module for AdaptiveAvgPool3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();

        let output_depth = self.output_size.0.unwrap_or(input_shape[2]);
        let output_height = self.output_size.1.unwrap_or(input_shape[3]);
        let output_width = self.output_size.2.unwrap_or(input_shape[4]);

        let output_shape = [
            input_shape[0],
            input_shape[1],
            output_depth,
            output_height,
            output_width,
        ];

        // Placeholder implementation
        let output = zeros(&output_shape)?;
        Ok(output)
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

/// Adaptive 3D max pooling layer
pub struct AdaptiveMaxPool3d {
    base: ModuleBase,
    output_size: (Option<usize>, Option<usize>, Option<usize>),
    return_indices: bool,
}

impl AdaptiveMaxPool3d {
    pub fn new(
        output_size: (Option<usize>, Option<usize>, Option<usize>),
        return_indices: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
            return_indices,
        }
    }

    pub fn with_output_size(output_size: usize) -> Self {
        Self::new(
            (Some(output_size), Some(output_size), Some(output_size)),
            false,
        )
    }

    /// Forward pass returning both output and optionally indices
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();

        let output_depth = self.output_size.0.unwrap_or(input_shape[2]);
        let output_height = self.output_size.1.unwrap_or(input_shape[3]);
        let output_width = self.output_size.2.unwrap_or(input_shape[4]);

        let output_shape = [
            input_shape[0],
            input_shape[1],
            output_depth,
            output_height,
            output_width,
        ];

        // Placeholder implementation
        let output = zeros(&output_shape)?;

        let indices = if self.return_indices {
            Some(zeros(&output_shape)?) // Placeholder for indices
        } else {
            None
        };

        Ok((output, indices))
    }
}

impl Module for AdaptiveMaxPool3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_with_indices(input)?;
        Ok(output)
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

/// Global pooling utilities
pub struct GlobalPool;

impl GlobalPool {
    /// Global average pooling - reduces spatial dimensions to 1x1
    pub fn global_avg_pool2d(input: &Tensor) -> Result<Tensor> {
        let adaptive_pool = AdaptiveAvgPool2d::new((Some(1), Some(1)));
        adaptive_pool.forward(input)
    }

    /// Global max pooling - reduces spatial dimensions to 1x1  
    pub fn global_max_pool2d(input: &Tensor) -> Result<Tensor> {
        let adaptive_pool = AdaptiveMaxPool2d::new((Some(1), Some(1)), false);
        adaptive_pool.forward(input)
    }

    /// Global average pooling for 1D inputs
    pub fn global_avg_pool1d(input: &Tensor) -> Result<Tensor> {
        let adaptive_pool = AdaptiveAvgPool1d::new(1);
        adaptive_pool.forward(input)
    }

    /// Global max pooling for 1D inputs
    pub fn global_max_pool1d(input: &Tensor) -> Result<Tensor> {
        let adaptive_pool = AdaptiveMaxPool1d::new(1, false);
        adaptive_pool.forward(input)
    }

    /// Global average pooling for 3D inputs
    pub fn global_avg_pool3d(input: &Tensor) -> Result<Tensor> {
        let adaptive_pool = AdaptiveAvgPool3d::new((Some(1), Some(1), Some(1)));
        adaptive_pool.forward(input)
    }

    /// Global max pooling for 3D inputs
    pub fn global_max_pool3d(input: &Tensor) -> Result<Tensor> {
        let adaptive_pool = AdaptiveMaxPool3d::new((Some(1), Some(1), Some(1)), false);
        adaptive_pool.forward(input)
    }
}

impl std::fmt::Debug for AdaptiveMaxPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveMaxPool2d")
            .field("output_size", &self.output_size)
            .field("return_indices", &self.return_indices)
            .finish()
    }
}

impl std::fmt::Debug for AdaptiveAvgPool1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveAvgPool1d")
            .field("output_size", &self.output_size)
            .finish()
    }
}

impl std::fmt::Debug for AdaptiveMaxPool1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveMaxPool1d")
            .field("output_size", &self.output_size)
            .field("return_indices", &self.return_indices)
            .finish()
    }
}

impl std::fmt::Debug for AdaptiveAvgPool3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveAvgPool3d")
            .field("output_size", &self.output_size)
            .finish()
    }
}

impl std::fmt::Debug for AdaptiveMaxPool3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveMaxPool3d")
            .field("output_size", &self.output_size)
            .field("return_indices", &self.return_indices)
            .finish()
    }
}

/// 1D max pooling layer
pub struct MaxPool1d {
    base: ModuleBase,
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
}

impl MaxPool1d {
    pub fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        }
    }

    pub fn with_kernel_size(kernel_size: usize) -> Self {
        Self::new(kernel_size, None, 0, 1, false)
    }
}

impl Module for MaxPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input shape: [batch_size, channels, length] or [batch_size, length]
        let binding = input.shape();
        let input_shape = binding.dims();
        let stride = self.stride.unwrap_or(self.kernel_size);

        if input_shape.len() != 2 && input_shape.len() != 3 {
            return Err(torsh_core::TorshError::InvalidArgument(
                "MaxPool1d expects 2D [N, L] or 3D [N, C, L] input".to_string(),
            ));
        }

        let length_dim = input_shape.len() - 1;
        let input_length = input_shape[length_dim];

        let output_length = if self.ceil_mode {
            ((input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) as f32
                / stride as f32)
                .ceil() as usize
                + 1
        } else {
            (input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / stride
                + 1
        };

        let mut output_shape = input_shape.to_vec();
        output_shape[length_dim] = output_length;

        // Placeholder implementation
        let output = zeros(&output_shape)?;
        Ok(output)
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

impl std::fmt::Debug for MaxPool1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxPool1d")
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("dilation", &self.dilation)
            .field("ceil_mode", &self.ceil_mode)
            .finish()
    }
}

/// 3D max pooling layer  
pub struct MaxPool3d {
    base: ModuleBase,
    kernel_size: (usize, usize, usize),
    stride: Option<(usize, usize, usize)>,
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    ceil_mode: bool,
}

impl MaxPool3d {
    pub fn new(
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        ceil_mode: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        }
    }

    pub fn with_kernel_size(kernel_size: usize) -> Self {
        Self::new(
            (kernel_size, kernel_size, kernel_size),
            None,
            (0, 0, 0),
            (1, 1, 1),
            false,
        )
    }
}

impl Module for MaxPool3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input shape: [batch_size, channels, depth, height, width]
        let binding = input.shape();
        let input_shape = binding.dims();
        let stride = self.stride.unwrap_or(self.kernel_size);

        if input_shape.len() != 5 {
            return Err(torsh_core::TorshError::InvalidArgument(
                "MaxPool3d expects 5D input [N, C, D, H, W]".to_string(),
            ));
        }

        let output_depth = if self.ceil_mode {
            ((input_shape[2] + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                as f32
                / stride.0 as f32)
                .ceil() as usize
                + 1
        } else {
            (input_shape[2] + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                / stride.0
                + 1
        };

        let output_height = if self.ceil_mode {
            ((input_shape[3] + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                as f32
                / stride.1 as f32)
                .ceil() as usize
                + 1
        } else {
            (input_shape[3] + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                / stride.1
                + 1
        };

        let output_width = if self.ceil_mode {
            ((input_shape[4] + 2 * self.padding.2 - self.dilation.2 * (self.kernel_size.2 - 1) - 1)
                as f32
                / stride.2 as f32)
                .ceil() as usize
                + 1
        } else {
            (input_shape[4] + 2 * self.padding.2 - self.dilation.2 * (self.kernel_size.2 - 1) - 1)
                / stride.2
                + 1
        };

        let output_shape = [
            input_shape[0],
            input_shape[1],
            output_depth,
            output_height,
            output_width,
        ];

        // Placeholder implementation
        let output = zeros(&output_shape)?;
        Ok(output)
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

impl std::fmt::Debug for MaxPool3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxPool3d")
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("dilation", &self.dilation)
            .field("ceil_mode", &self.ceil_mode)
            .finish()
    }
}
