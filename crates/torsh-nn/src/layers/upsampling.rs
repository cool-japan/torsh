//! Upsampling and downsampling layers

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// Pixel shuffle layer for upsampling
///
/// Rearranges elements in a tensor from depth to spatial dimensions.
/// This is useful for implementing sub-pixel convolution layers for super-resolution.
///
/// The input tensor is expected to have shape [N, C*r^2, H, W] for 2D pixel shuffle,
/// where r is the upscale factor, and the output will have shape [N, C, H*r, W*r].
pub struct PixelShuffle {
    base: ModuleBase,
    upscale_factor: usize,
}

impl PixelShuffle {
    /// Create a new PixelShuffle layer
    ///
    /// # Arguments
    /// * `upscale_factor` - Factor by which to upscale spatial dimensions
    pub fn new(upscale_factor: usize) -> Self {
        Self {
            base: ModuleBase::new(),
            upscale_factor,
        }
    }

    /// Get the upscale factor
    pub fn upscale_factor(&self) -> usize {
        self.upscale_factor
    }
}

impl Module for PixelShuffle {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();

        // Expect input shape: [N, C*r^2, H, W]
        if input_shape.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "PixelShuffle expects 4D input (N, C, H, W)".to_string(),
            ));
        }

        let batch_size = input_shape[0];
        let channels_in = input_shape[1];
        let height_in = input_shape[2];
        let width_in = input_shape[3];

        let r = self.upscale_factor;
        let r_squared = r * r;

        // Check if channel dimension is divisible by r^2
        if channels_in % r_squared != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Input channels {} must be divisible by upscale_factor^2 = {}",
                channels_in, r_squared
            )));
        }

        let channels_out = channels_in / r_squared;
        let height_out = height_in * r;
        let width_out = width_in * r;

        let output_shape = [batch_size, channels_out, height_out, width_out];

        // Placeholder implementation - real pixel shuffle would rearrange tensor elements
        // The actual implementation would involve:
        // 1. Reshape input from [N, C*r^2, H, W] to [N, C, r, r, H, W]
        // 2. Permute dimensions to [N, C, H, r, W, r]
        // 3. Reshape to [N, C, H*r, W*r]
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

/// Pixel unshuffle layer for downsampling
///
/// Rearranges elements in a tensor from spatial dimensions to depth.
/// This is the inverse operation of pixel shuffle.
///
/// The input tensor is expected to have shape [N, C, H, W] for 2D pixel unshuffle,
/// where H and W must be divisible by the downscale factor r,
/// and the output will have shape [N, C*r^2, H/r, W/r].
pub struct PixelUnshuffle {
    base: ModuleBase,
    downscale_factor: usize,
}

impl PixelUnshuffle {
    /// Create a new PixelUnshuffle layer
    ///
    /// # Arguments
    /// * `downscale_factor` - Factor by which to downscale spatial dimensions
    pub fn new(downscale_factor: usize) -> Self {
        Self {
            base: ModuleBase::new(),
            downscale_factor,
        }
    }

    /// Get the downscale factor
    pub fn downscale_factor(&self) -> usize {
        self.downscale_factor
    }
}

impl Module for PixelUnshuffle {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();

        // Expect input shape: [N, C, H, W]
        if input_shape.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "PixelUnshuffle expects 4D input (N, C, H, W)".to_string(),
            ));
        }

        let batch_size = input_shape[0];
        let channels_in = input_shape[1];
        let height_in = input_shape[2];
        let width_in = input_shape[3];

        let r = self.downscale_factor;

        // Check if spatial dimensions are divisible by downscale factor
        if height_in % r != 0 || width_in % r != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Input spatial dimensions ({}, {}) must be divisible by downscale_factor {}",
                height_in, width_in, r
            )));
        }

        let r_squared = r * r;
        let channels_out = channels_in * r_squared;
        let height_out = height_in / r;
        let width_out = width_in / r;

        let output_shape = [batch_size, channels_out, height_out, width_out];

        // Placeholder implementation - real pixel unshuffle would rearrange tensor elements
        // The actual implementation would involve:
        // 1. Reshape input from [N, C, H, W] to [N, C, H/r, r, W/r, r]
        // 2. Permute dimensions to [N, C, r, r, H/r, W/r]
        // 3. Reshape to [N, C*r^2, H/r, W/r]
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

/// 1D Pixel shuffle layer for 1D upsampling
///
/// Similar to PixelShuffle but for 1D tensors.
/// Input shape: [N, C*r, L] -> Output shape: [N, C, L*r]
pub struct PixelShuffle1d {
    base: ModuleBase,
    upscale_factor: usize,
}

impl PixelShuffle1d {
    pub fn new(upscale_factor: usize) -> Self {
        Self {
            base: ModuleBase::new(),
            upscale_factor,
        }
    }

    pub fn upscale_factor(&self) -> usize {
        self.upscale_factor
    }
}

impl Module for PixelShuffle1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();

        if input_shape.len() != 3 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "PixelShuffle1d expects 3D input (N, C, L)".to_string(),
            ));
        }

        let batch_size = input_shape[0];
        let channels_in = input_shape[1];
        let length_in = input_shape[2];

        let r = self.upscale_factor;

        if channels_in % r != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Input channels {} must be divisible by upscale_factor {}",
                channels_in, r
            )));
        }

        let channels_out = channels_in / r;
        let length_out = length_in * r;

        let output_shape = [batch_size, channels_out, length_out];

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

/// 1D Pixel unshuffle layer for 1D downsampling
///
/// Similar to PixelUnshuffle but for 1D tensors.
/// Input shape: [N, C, L] -> Output shape: [N, C*r, L/r]
pub struct PixelUnshuffle1d {
    base: ModuleBase,
    downscale_factor: usize,
}

impl PixelUnshuffle1d {
    pub fn new(downscale_factor: usize) -> Self {
        Self {
            base: ModuleBase::new(),
            downscale_factor,
        }
    }

    pub fn downscale_factor(&self) -> usize {
        self.downscale_factor
    }
}

impl Module for PixelUnshuffle1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();

        if input_shape.len() != 3 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "PixelUnshuffle1d expects 3D input (N, C, L)".to_string(),
            ));
        }

        let batch_size = input_shape[0];
        let channels_in = input_shape[1];
        let length_in = input_shape[2];

        let r = self.downscale_factor;

        if length_in % r != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Input length {} must be divisible by downscale_factor {}",
                length_in, r
            )));
        }

        let channels_out = channels_in * r;
        let length_out = length_in / r;

        let output_shape = [batch_size, channels_out, length_out];

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

// Debug implementations
impl std::fmt::Debug for PixelShuffle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixelShuffle")
            .field("upscale_factor", &self.upscale_factor)
            .finish()
    }
}

impl std::fmt::Debug for PixelUnshuffle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixelUnshuffle")
            .field("downscale_factor", &self.downscale_factor)
            .finish()
    }
}

impl std::fmt::Debug for PixelShuffle1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixelShuffle1d")
            .field("upscale_factor", &self.upscale_factor)
            .finish()
    }
}

impl std::fmt::Debug for PixelUnshuffle1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixelUnshuffle1d")
            .field("downscale_factor", &self.downscale_factor)
            .finish()
    }
}

/// Utility functions for pixel shuffling operations
pub mod utils {
    /// Calculate output spatial dimensions for pixel shuffle
    pub fn pixel_shuffle_output_size(
        input_size: (usize, usize),
        upscale_factor: usize,
    ) -> (usize, usize) {
        (input_size.0 * upscale_factor, input_size.1 * upscale_factor)
    }

    /// Calculate output spatial dimensions for pixel unshuffle
    pub fn pixel_unshuffle_output_size(
        input_size: (usize, usize),
        downscale_factor: usize,
    ) -> Result<(usize, usize), String> {
        if input_size.0 % downscale_factor != 0 || input_size.1 % downscale_factor != 0 {
            return Err(format!(
                "Input size {:?} must be divisible by downscale factor {}",
                input_size, downscale_factor
            ));
        }
        Ok((
            input_size.0 / downscale_factor,
            input_size.1 / downscale_factor,
        ))
    }

    /// Calculate required input channels for pixel shuffle
    pub fn pixel_shuffle_input_channels(output_channels: usize, upscale_factor: usize) -> usize {
        output_channels * upscale_factor * upscale_factor
    }

    /// Calculate output channels for pixel unshuffle
    pub fn pixel_unshuffle_output_channels(
        input_channels: usize,
        downscale_factor: usize,
    ) -> usize {
        input_channels * downscale_factor * downscale_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_pixel_shuffle_creation() {
        let layer = PixelShuffle::new(2);
        assert_eq!(layer.upscale_factor(), 2);
    }

    #[test]
    fn test_pixel_unshuffle_creation() {
        let layer = PixelUnshuffle::new(2);
        assert_eq!(layer.downscale_factor(), 2);
    }

    #[test]
    fn test_pixel_shuffle_output_shape() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let layer = PixelShuffle::new(2);
        let input = zeros(&[1, 12, 16, 16])?; // 1 batch, 12 channels, 16x16
        let result = layer.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        let binding = output.shape();
        let output_shape = binding.dims();
        assert_eq!(output_shape, &[1, 3, 32, 32]); // 3 channels, 32x32
        Ok(())
    }

    #[test]
    fn test_pixel_unshuffle_output_shape() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let layer = PixelUnshuffle::new(2);
        let input = zeros(&[1, 3, 32, 32])?; // 1 batch, 3 channels, 32x32
        let result = layer.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        let binding = output.shape();
        let output_shape = binding.dims();
        assert_eq!(output_shape, &[1, 12, 16, 16]); // 12 channels, 16x16
        Ok(())
    }

    #[test]
    fn test_pixel_shuffle_invalid_channels() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let layer = PixelShuffle::new(2);
        let input = zeros(&[1, 11, 16, 16])?; // 11 channels, not divisible by 4
        let result = layer.forward(&input);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_pixel_unshuffle_invalid_dimensions(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let layer = PixelUnshuffle::new(3);
        let input = zeros(&[1, 3, 16, 17])?; // 17 is not divisible by 3
        let result = layer.forward(&input);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_utils_functions() {
        use super::utils::*;

        assert_eq!(pixel_shuffle_output_size((16, 16), 2), (32, 32));
        assert_eq!(pixel_unshuffle_output_size((32, 32), 2).unwrap(), (16, 16));
        assert!(pixel_unshuffle_output_size((17, 16), 2).is_err());

        assert_eq!(pixel_shuffle_input_channels(3, 2), 12);
        assert_eq!(pixel_unshuffle_output_channels(3, 2), 12);
    }
}
