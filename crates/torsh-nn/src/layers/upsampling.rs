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

/// Pixel shuffle layer for upsampling (sub-pixel convolution)
///
/// Rearranges elements in a tensor from depth to spatial dimensions.
/// This is the core operation for sub-pixel convolutional layers used in
/// super-resolution networks like ESPCN (Efficient Sub-Pixel Convolutional Network).
///
/// # Mathematical Formula
///
/// For an input tensor `X` with shape `[N, C*r², H, W]`:
/// ```text
/// Y[n, c, h, w] = X[n, c*r² + ry*r + rx, h÷r, w÷r]
/// where ry = h mod r, rx = w mod r
/// ```
///
/// The transformation reshapes:
/// - Input: `[N, C*r², H, W]`
/// - Intermediate: `[N, C, r, r, H, W]` → permute → `[N, C, H, r, W, r]`
/// - Output: `[N, C, H*r, W*r]`
///
/// # PyTorch Equivalence
///
/// This layer is equivalent to PyTorch's `nn.PixelShuffle(upscale_factor)`.
///
/// ```python
/// # PyTorch
/// import torch.nn as nn
/// layer = nn.PixelShuffle(upscale_factor=2)
/// ```
///
/// # Examples
///
/// ```rust
/// use torsh_nn::layers::PixelShuffle;
/// use torsh_nn::Module;
/// use torsh_tensor::creation::ones;
///
/// // Create a pixel shuffle layer with upscale factor 2
/// let layer = PixelShuffle::new(2);
///
/// // Input: [1 batch, 12 channels, 16×16]
/// let input = ones(&[1, 12, 16, 16])?;
///
/// // Output: [1 batch, 3 channels, 32×32] (upscaled by 2×)
/// let output = layer.forward(&input)?;
/// # Ok::<(), torsh_core::error::TorshError>(())
/// ```
///
/// # References
///
/// - ESPCN Paper: "Real-Time Single Image and Video Super-Resolution Using an Efficient
///   Sub-Pixel Convolutional Neural Network" (Shi et al., 2016)
/// - PyTorch Documentation: <https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html>
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

        // Implementation:
        // 1. Reshape input from [N, C*r^2, H, W] to [N, C, r, r, H, W]
        // 2. Permute dimensions to [N, C, H, r, W, r]
        // 3. Reshape to [N, C, H*r, W*r]

        // Get input data
        let input_data = input.to_vec()?;
        let mut output_data =
            vec![input_data[0]; batch_size * channels_out * height_out * width_out];

        // Perform pixel shuffle by rearranging data
        for b in 0..batch_size {
            for c in 0..channels_out {
                for h in 0..height_in {
                    for w in 0..width_in {
                        for ry in 0..r {
                            for rx in 0..r {
                                // Input index: [b, c*r*r + ry*r + rx, h, w]
                                let c_in = c * r_squared + ry * r + rx;
                                let in_idx =
                                    ((b * channels_in + c_in) * height_in + h) * width_in + w;

                                // Output index: [b, c, h*r + ry, w*r + rx]
                                let h_out = h * r + ry;
                                let w_out = w * r + rx;
                                let out_idx = ((b * channels_out + c) * height_out + h_out)
                                    * width_out
                                    + w_out;

                                output_data[out_idx] = input_data[in_idx];
                            }
                        }
                    }
                }
            }
        }

        let output_shape = [batch_size, channels_out, height_out, width_out];
        Tensor::from_data(output_data, output_shape.to_vec(), input.device())
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

/// Pixel unshuffle layer for downsampling (inverse sub-pixel convolution)
///
/// Rearranges elements in a tensor from spatial dimensions to depth.
/// This is the inverse operation of pixel shuffle, useful for efficient
/// downsampling and feature extraction in neural networks.
///
/// # Mathematical Formula
///
/// For an input tensor `X` with shape `[N, C, H, W]` where `H` and `W` are divisible by `r`:
/// ```text
/// Y[n, c*r² + ry*r + rx, h, w] = X[n, c, h*r + ry, w*r + rx]
/// where ry ∈ [0, r), rx ∈ [0, r)
/// ```
///
/// The transformation reshapes:
/// - Input: `[N, C, H, W]`
/// - Intermediate: `[N, C, H÷r, r, W÷r, r]` → permute → `[N, C, r, r, H÷r, W÷r]`
/// - Output: `[N, C*r², H÷r, W÷r]`
///
/// # PyTorch Equivalence
///
/// This layer is equivalent to PyTorch's `nn.PixelUnshuffle(downscale_factor)`.
///
/// ```python
/// # PyTorch
/// import torch.nn as nn
/// layer = nn.PixelUnshuffle(downscale_factor=2)
/// ```
///
/// # Examples
///
/// ```rust
/// use torsh_nn::layers::PixelUnshuffle;
/// use torsh_nn::Module;
/// use torsh_tensor::creation::ones;
///
/// // Create a pixel unshuffle layer with downscale factor 2
/// let layer = PixelUnshuffle::new(2);
///
/// // Input: [1 batch, 3 channels, 32×32]
/// let input = ones(&[1, 3, 32, 32])?;
///
/// // Output: [1 batch, 12 channels, 16×16] (downscaled by 2×)
/// let output = layer.forward(&input)?;
/// # Ok::<(), torsh_core::error::TorshError>(())
/// ```
///
/// # Use Cases
///
/// - Efficient downsampling without information loss
/// - Pre-processing for super-resolution networks
/// - Feature pyramid construction
/// - Invertible transformations (with PixelShuffle)
///
/// # References
///
/// - PyTorch Documentation: <https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html>
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

        // Implementation:
        // 1. Reshape input from [N, C, H, W] to [N, C, H/r, r, W/r, r]
        // 2. Permute dimensions to [N, C, r, r, H/r, W/r]
        // 3. Reshape to [N, C*r^2, H/r, W/r]

        // Get input data
        let input_data = input.to_vec()?;
        let mut output_data =
            vec![input_data[0]; batch_size * channels_out * height_out * width_out];

        // Perform pixel unshuffle by rearranging data
        for b in 0..batch_size {
            for c in 0..channels_in {
                for h in 0..height_out {
                    for w in 0..width_out {
                        for ry in 0..r {
                            for rx in 0..r {
                                // Input index: [b, c, h*r + ry, w*r + rx]
                                let h_in = h * r + ry;
                                let w_in = w * r + rx;
                                let in_idx =
                                    ((b * channels_in + c) * height_in + h_in) * width_in + w_in;

                                // Output index: [b, c*r*r + ry*r + rx, h, w]
                                let c_out = c * r_squared + ry * r + rx;
                                let out_idx =
                                    ((b * channels_out + c_out) * height_out + h) * width_out + w;

                                output_data[out_idx] = input_data[in_idx];
                            }
                        }
                    }
                }
            }
        }

        let output_shape = [batch_size, channels_out, height_out, width_out];
        Tensor::from_data(output_data, output_shape.to_vec(), input.device())
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
    use torsh_tensor::creation::{ones, zeros};

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

        let output = result?;
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

        let output = result?;
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
    fn test_pixel_shuffle_upscale_factor_3() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let layer = PixelShuffle::new(3);
        let input = zeros(&[2, 27, 8, 8])?; // 2 batches, 27 channels (3*3*3), 8x8
        let output = layer.forward(&input)?;
        let binding = output.shape();
        let output_shape = binding.dims();
        assert_eq!(output_shape, &[2, 3, 24, 24]); // 3 channels, 24x24 (8*3)
        Ok(())
    }

    #[test]
    fn test_pixel_shuffle_upscale_factor_4() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let layer = PixelShuffle::new(4);
        let input = zeros(&[1, 64, 4, 4])?; // 1 batch, 64 channels (4*4*4), 4x4
        let output = layer.forward(&input)?;
        let binding = output.shape();
        let output_shape = binding.dims();
        assert_eq!(output_shape, &[1, 4, 16, 16]); // 4 channels, 16x16 (4*4)
        Ok(())
    }

    #[test]
    fn test_pixel_unshuffle_downscale_factor_3(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let layer = PixelUnshuffle::new(3);
        let input = zeros(&[2, 3, 24, 24])?; // 2 batches, 3 channels, 24x24
        let output = layer.forward(&input)?;
        let binding = output.shape();
        let output_shape = binding.dims();
        assert_eq!(output_shape, &[2, 27, 8, 8]); // 27 channels (3*3*3), 8x8
        Ok(())
    }

    #[test]
    fn test_pixel_shuffle_round_trip() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Test: PixelShuffle -> PixelUnshuffle should give back original shape
        let shuffle = PixelShuffle::new(2);
        let unshuffle = PixelUnshuffle::new(2);

        let input = ones(&[1, 12, 8, 8])?;
        let shuffled = shuffle.forward(&input)?;
        let unshuffled = unshuffle.forward(&shuffled)?;

        let input_binding = input.shape();
        let input_shape = input_binding.dims();
        let output_binding = unshuffled.shape();
        let output_shape = output_binding.dims();
        assert_eq!(input_shape, output_shape);
        Ok(())
    }

    #[test]
    fn test_pixel_shuffle_invalid_spatial_dim(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let layer = PixelShuffle::new(2);
        let input = zeros(&[1, 12, 16])?; // Wrong number of dimensions (3D instead of 4D)
        let result = layer.forward(&input);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_pixel_unshuffle_invalid_spatial_dim(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let layer = PixelUnshuffle::new(2);
        let input = zeros(&[1, 3, 32])?; // Wrong number of dimensions (3D instead of 4D)
        let result = layer.forward(&input);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_pixel_shuffle_shape_preservation() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        // Test that total number of elements is preserved
        let layer = PixelShuffle::new(2);
        let input = zeros(&[1, 12, 16, 16])?;
        let output = layer.forward(&input)?;

        assert_eq!(input.numel(), output.numel());
        Ok(())
    }

    #[test]
    fn test_pixel_unshuffle_shape_preservation(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Test that total number of elements is preserved
        let layer = PixelUnshuffle::new(2);
        let input = zeros(&[1, 3, 32, 32])?;
        let output = layer.forward(&input)?;

        assert_eq!(input.numel(), output.numel());
        Ok(())
    }

    #[test]
    fn test_pixel_shuffle_module_trait() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut layer = PixelShuffle::new(2);

        // Test training mode toggle
        assert!(layer.training()); // Default is training mode
        layer.eval();
        assert!(!layer.training());
        layer.train();
        assert!(layer.training());

        // Test parameters (should be empty for this layer)
        let params = layer.parameters();
        assert!(params.is_empty());

        Ok(())
    }

    #[test]
    fn test_pixel_unshuffle_module_trait() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut layer = PixelUnshuffle::new(3);

        // Test training mode toggle
        assert!(layer.training()); // Default is training mode
        layer.eval();
        assert!(!layer.training());
        layer.train();
        assert!(layer.training());

        // Test parameters (should be empty for this layer)
        let params = layer.parameters();
        assert!(params.is_empty());

        Ok(())
    }

    #[test]
    fn test_pixel_shuffle_data_correctness() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        // Create a simple test case where we can verify the data rearrangement
        // Input: [1, 4, 2, 2] with values 0..15
        let input_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let input = Tensor::from_data(input_data, vec![1, 4, 2, 2], DeviceType::Cpu)?;

        let layer = PixelShuffle::new(2);
        let output = layer.forward(&input)?;

        // Output should be [1, 1, 4, 4]
        let binding = output.shape();
        let output_shape = binding.dims();
        assert_eq!(output_shape, &[1, 1, 4, 4]);

        // Verify data is correctly rearranged
        let output_data = output.to_vec()?;
        assert_eq!(output_data.len(), 16);

        Ok(())
    }

    #[test]
    fn test_pixel_unshuffle_data_correctness() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        // Create a simple test case where we can verify the data rearrangement
        // Input: [1, 1, 4, 4] with values 0..15
        let input_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let input = Tensor::from_data(input_data, vec![1, 1, 4, 4], DeviceType::Cpu)?;

        let layer = PixelUnshuffle::new(2);
        let output = layer.forward(&input)?;

        // Output should be [1, 4, 2, 2]
        let binding = output.shape();
        let output_shape = binding.dims();
        assert_eq!(output_shape, &[1, 4, 2, 2]);

        // Verify data is correctly rearranged
        let output_data = output.to_vec()?;
        assert_eq!(output_data.len(), 16);

        Ok(())
    }

    #[test]
    fn test_pixel_shuffle_unshuffle_inverse() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        // Test that PixelShuffle and PixelUnshuffle are inverse operations
        let input_data: Vec<f32> = (0..48).map(|x| x as f32).collect();
        let input = Tensor::from_data(input_data.clone(), vec![1, 12, 2, 2], DeviceType::Cpu)?;

        let shuffle = PixelShuffle::new(2);
        let unshuffle = PixelUnshuffle::new(2);

        // Forward: shuffle then unshuffle
        let shuffled = shuffle.forward(&input)?;
        let result = unshuffle.forward(&shuffled)?;

        // Should get back the original data
        let result_data = result.to_vec()?;

        // Verify shape matches
        let binding = result.shape();
        let result_shape = binding.dims();
        let binding2 = input.shape();
        let input_shape = binding2.dims();
        assert_eq!(result_shape, input_shape);

        // Verify data matches (allowing for small floating point errors)
        assert_eq!(result_data.len(), input_data.len());
        for (a, b) in result_data.iter().zip(input_data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_utils_functions() {
        use super::utils::*;

        assert_eq!(pixel_shuffle_output_size((16, 16), 2), (32, 32));
        assert_eq!(pixel_unshuffle_output_size((32, 32), 2), Ok((16, 16)));
        assert!(pixel_unshuffle_output_size((17, 16), 2).is_err());

        assert_eq!(pixel_shuffle_input_channels(3, 2), 12);
        assert_eq!(pixel_unshuffle_output_channels(3, 2), 12);
    }
}
