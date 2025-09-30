//! Super-Resolution model implementations
//!
//! This module provides implementations of various super-resolution techniques including:
//! - SRCNN (Super-Resolution Convolutional Neural Network)
//! - ESPCN (Efficient Sub-Pixel Convolutional Neural Network)
//! - Enhanced Deep Residual Networks (EDSR-style)
//! - Sub-pixel convolution layers

use crate::{Result, VisionError};
use torsh_nn::functional::pooling::max_pool2d;
use torsh_nn::prelude::*;
use torsh_tensor::prelude::*;
use torsh_tensor::Tensor;

/// SRCNN (Super-Resolution Convolutional Neural Network)
/// A simple 3-layer CNN for image super-resolution
pub struct SRCNN {
    conv1: Conv2d, // Feature extraction
    conv2: Conv2d, // Non-linear mapping
    conv3: Conv2d, // Reconstruction
    scale_factor: usize,
}

impl SRCNN {
    /// Create a new SRCNN model
    /// - scale_factor: Super-resolution scale (2x, 3x, 4x, etc.)
    pub fn new(scale_factor: usize) -> Result<Self> {
        Ok(Self {
            conv1: Conv2d::new(1, 64, (9, 9), (1, 1), (4, 4), (1, 1), true, 1), // 9x9 kernel, 64 filters
            conv2: Conv2d::new(64, 32, (1, 1), (1, 1), (0, 0), (1, 1), true, 1), // 1x1 kernel, 32 filters
            conv3: Conv2d::new(32, 1, (5, 5), (1, 1), (2, 2), (1, 1), true, 1), // 5x5 kernel, 1 filter
            scale_factor,
        })
    }

    /// Forward pass through SRCNN
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Input should be bicubic upsampled low-resolution image
        let x = self.conv1.forward(x)?;
        let x = x.relu()?;

        let x = self.conv2.forward(&x)?;
        let x = x.relu()?;

        let x = self.conv3.forward(&x)?;
        Ok(x)
    }

    /// Preprocess low-resolution image (bicubic upsampling)
    pub fn preprocess(&self, lr_image: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = lr_image.shape();
        let (batch, channels, height, width) = (
            shape.dims()[0],
            shape.dims()[1],
            shape.dims()[2],
            shape.dims()[3],
        );

        let new_height = height * self.scale_factor;
        let new_width = width * self.scale_factor;

        // Simple bilinear upsampling (bicubic would be more accurate)
        crate::ops::resize(lr_image, (new_width, new_height))
    }
}

/// ESPCN (Efficient Sub-Pixel Convolutional Neural Network)
/// Uses sub-pixel convolution for efficient upsampling
pub struct ESPCN {
    conv1: Conv2d,
    conv2: Conv2d,
    subpixel_conv: SubPixelConv2d,
    scale_factor: usize,
}

impl ESPCN {
    /// Create a new ESPCN model
    pub fn new(scale_factor: usize) -> Result<Self> {
        let r = scale_factor;
        Ok(Self {
            conv1: Conv2d::new(1, 64, (5, 5), (1, 1), (2, 2), (1, 1), true, 1),
            conv2: Conv2d::new(64, 32, (3, 3), (1, 1), (1, 1), (1, 1), true, 1),
            subpixel_conv: SubPixelConv2d::new(32, 1, 3, r)?,
            scale_factor,
        })
    }

    /// Forward pass through ESPCN
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let x = self.conv1.forward(x)?;
        let x = x.relu()?;

        let x = self.conv2.forward(&x)?;
        let x = x.relu()?;

        let x = self.subpixel_conv.forward(&x)?;
        Ok(x)
    }
}

/// Sub-pixel convolution layer for efficient upsampling
pub struct SubPixelConv2d {
    conv: Conv2d,
    scale_factor: usize,
}

impl SubPixelConv2d {
    /// Create a new sub-pixel convolution layer
    /// - in_channels: Number of input channels
    /// - out_channels: Number of output channels
    /// - kernel_size: Convolution kernel size
    /// - scale_factor: Upsampling factor
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        scale_factor: usize,
    ) -> Result<Self> {
        let padding = kernel_size / 2;
        let conv_out_channels = out_channels * scale_factor * scale_factor;

        Ok(Self {
            conv: Conv2d::new(
                in_channels,
                conv_out_channels,
                (kernel_size, kernel_size),
                (1, 1),
                (padding, padding),
                (1, 1),
                true,
                1,
            ),
            scale_factor,
        })
    }

    /// Forward pass with sub-pixel convolution
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let conv_out = self.conv.forward(x)?;
        self.pixel_shuffle(&conv_out)
    }

    /// Pixel shuffle operation to rearrange sub-pixels
    fn pixel_shuffle(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = x.shape();
        let (batch, channels, height, width) = (
            shape.dims()[0],
            shape.dims()[1],
            shape.dims()[2],
            shape.dims()[3],
        );

        let r = self.scale_factor;
        let out_channels = channels / (r * r);

        if channels % (r * r) != 0 {
            return Err(VisionError::InvalidShape(format!(
                "Channels {} not divisible by scale_factor^2 {}",
                channels,
                r * r
            )));
        }

        // Reshape and permute to perform pixel shuffle
        let x_reshaped = x.view(&[
            batch as i32,
            out_channels as i32,
            r as i32,
            r as i32,
            height as i32,
            width as i32,
        ])?;

        // Permute dimensions to group sub-pixels
        let x_permuted = x_reshaped.permute(&[0, 1, 4, 2, 5, 3])?;

        // Reshape to final output size
        let output = x_permuted.view(&[
            batch as i32,
            out_channels as i32,
            (height * r) as i32,
            (width * r) as i32,
        ])?;

        Ok(output)
    }
}

/// Enhanced Deep Super-Resolution (EDSR-style) network
pub struct EDSR {
    conv_first: Conv2d,
    res_blocks: Vec<ResBlock>,
    conv_last: Conv2d,
    upsampler: Upsampler,
    scale_factor: usize,
    num_features: usize,
}

impl EDSR {
    /// Create a new EDSR model
    /// - scale_factor: Super-resolution scale factor
    /// - num_features: Number of feature maps
    /// - num_blocks: Number of residual blocks
    pub fn new(scale_factor: usize, num_features: usize, num_blocks: usize) -> Result<Self> {
        let mut res_blocks = Vec::new();
        for _ in 0..num_blocks {
            res_blocks.push(ResBlock::new(num_features)?);
        }

        Ok(Self {
            conv_first: Conv2d::new(3, num_features, (3, 3), (1, 1), (1, 1), (1, 1), true, 1),
            res_blocks,
            conv_last: Conv2d::new(
                num_features,
                num_features,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                true,
                1,
            ),
            upsampler: Upsampler::new(num_features, 3, scale_factor)?,
            scale_factor,
            num_features,
        })
    }

    /// Forward pass through EDSR
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let x = self.conv_first.forward(x)?;
        let res = x.clone();

        let mut x = x;
        for block in &self.res_blocks {
            x = block.forward(&x)?;
        }

        let x = self.conv_last.forward(&x)?;
        let x = x.add(&res)?; // Global residual connection

        let x = self.upsampler.forward(&x)?;
        Ok(x)
    }
}

/// Residual block for EDSR
pub struct ResBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    res_scale: f32,
}

impl ResBlock {
    pub fn new(num_features: usize) -> Result<Self> {
        Ok(Self {
            conv1: Conv2d::new(
                num_features,
                num_features,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                true,
                1,
            ),
            conv2: Conv2d::new(
                num_features,
                num_features,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                true,
                1,
            ),
            res_scale: 0.1, // Residual scaling for training stability
        })
    }

    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let res = x;

        let x = self.conv1.forward(x)?;
        let x = x.relu()?;
        let x = self.conv2.forward(&x)?;
        let x = x.mul_scalar(self.res_scale)?;

        let x = x.add(res)?;
        Ok(x)
    }
}

/// Upsampler module with sub-pixel convolution
pub struct Upsampler {
    layers: Vec<UpsampleBlock>,
}

impl Upsampler {
    /// Create upsampler for given scale factor
    pub fn new(in_channels: usize, out_channels: usize, scale_factor: usize) -> Result<Self> {
        let mut layers = Vec::new();
        let mut current_channels = in_channels;
        let mut remaining_scale = scale_factor;

        // Decompose scale factor into powers of 2 and 3
        while remaining_scale > 1 {
            if remaining_scale % 4 == 0 {
                layers.push(UpsampleBlock::new(current_channels, current_channels, 2)?);
                layers.push(UpsampleBlock::new(current_channels, current_channels, 2)?);
                remaining_scale /= 4;
            } else if remaining_scale % 2 == 0 {
                layers.push(UpsampleBlock::new(current_channels, current_channels, 2)?);
                remaining_scale /= 2;
            } else if remaining_scale % 3 == 0 {
                layers.push(UpsampleBlock::new(current_channels, current_channels, 3)?);
                remaining_scale /= 3;
            } else {
                return Err(VisionError::InvalidInput(format!(
                    "Unsupported scale factor: {}",
                    scale_factor
                )));
            }
        }

        // Final convolution to output channels
        if layers.is_empty() {
            layers.push(UpsampleBlock::new(in_channels, out_channels, 1)?);
        } else {
            // Add final 1x1 conv to convert to output channels without changing spatial dimensions
            layers.push(UpsampleBlock::new(current_channels, out_channels, 1)?);
        }

        Ok(Self { layers })
    }

    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

/// Individual upsampling block
pub struct UpsampleBlock {
    subpixel_conv: Option<SubPixelConv2d>,
    conv: Option<Conv2d>,
    scale: usize,
}

impl UpsampleBlock {
    pub fn new(in_channels: usize, out_channels: usize, scale: usize) -> Result<Self> {
        if scale > 1 {
            Ok(Self {
                subpixel_conv: Some(SubPixelConv2d::new(in_channels, out_channels, 3, scale)?),
                conv: None,
                scale,
            })
        } else {
            Ok(Self {
                subpixel_conv: None,
                conv: Some(Conv2d::new(
                    in_channels,
                    out_channels,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    true,
                    1,
                )),
                scale,
            })
        }
    }

    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        if let Some(ref subpixel) = self.subpixel_conv {
            Ok(subpixel.forward(x)?)
        } else if let Some(ref conv) = self.conv {
            Ok(conv.forward(x)?)
        } else {
            Ok(x.clone())
        }
    }
}

/// Loss functions for super-resolution training
pub struct SuperResolutionLoss {
    l1_weight: f32,
    perceptual_weight: f32,
    perceptual_network: Option<VGGFeatureExtractor>,
}

impl SuperResolutionLoss {
    pub fn new(l1_weight: f32, perceptual_weight: f32, use_perceptual: bool) -> Result<Self> {
        let perceptual_network = if use_perceptual {
            Some(VGGFeatureExtractor::new()?)
        } else {
            None
        };

        Ok(Self {
            l1_weight,
            perceptual_weight,
            perceptual_network,
        })
    }

    /// Compute total loss for super-resolution
    pub fn compute_loss(&self, pred: &Tensor<f32>, target: &Tensor<f32>) -> Result<Tensor<f32>> {
        // L1 loss
        let l1_loss = pred.sub(target)?.abs()?.mean(None, false)?;
        let weighted_l1 = l1_loss.mul_scalar(self.l1_weight)?;

        if let Some(ref vgg) = self.perceptual_network {
            // Perceptual loss
            let pred_features = vgg.extract_features(pred)?;
            let target_features = vgg.extract_features(target)?;

            let mut perceptual_loss = zeros(&[1])?;
            for (pred_feat, target_feat) in pred_features.iter().zip(target_features.iter()) {
                let feat_loss = pred_feat.sub(target_feat)?.pow(2.0)?.mean(None, false)?;
                perceptual_loss = perceptual_loss.add(&feat_loss)?;
            }

            let weighted_perceptual = perceptual_loss.mul_scalar(self.perceptual_weight)?;
            let total_loss = weighted_l1.add(&weighted_perceptual)?;
            Ok(total_loss)
        } else {
            Ok(weighted_l1)
        }
    }
}

/// Simplified VGG feature extractor for perceptual loss
pub struct VGGFeatureExtractor {
    conv1_1: Conv2d,
    conv1_2: Conv2d,
    conv2_1: Conv2d,
    conv2_2: Conv2d,
    conv3_1: Conv2d,
}

impl VGGFeatureExtractor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            conv1_1: Conv2d::new(3, 64, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv1_2: Conv2d::new(64, 64, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv2_1: Conv2d::new(64, 128, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv2_2: Conv2d::new(128, 128, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv3_1: Conv2d::new(128, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
        })
    }

    pub fn extract_features(&self, x: &Tensor<f32>) -> Result<Vec<Tensor<f32>>> {
        let mut features = Vec::new();

        let x = self.conv1_1.forward(x)?.relu()?;
        let x = self.conv1_2.forward(&x)?.relu()?;
        features.push(x.clone());

        let x = max_pool2d(&x, (2, 2), Some((2, 2)), Some((0, 0)), None)?;
        let x = self.conv2_1.forward(&x)?.relu()?;
        let x = self.conv2_2.forward(&x)?.relu()?;
        features.push(x.clone());

        let x = max_pool2d(&x, (2, 2), Some((2, 2)), Some((0, 0)), None)?;
        let x = self.conv3_1.forward(&x)?.relu()?;
        features.push(x);

        Ok(features)
    }
}

/// Super-resolution utilities
pub mod super_resolution_utils {
    use super::*;

    /// Create SRCNN model with default settings
    pub fn create_srcnn(scale_factor: usize) -> Result<SRCNN> {
        SRCNN::new(scale_factor)
    }

    /// Create ESPCN model with default settings
    pub fn create_espcn(scale_factor: usize) -> Result<ESPCN> {
        ESPCN::new(scale_factor)
    }

    /// Create EDSR model with default settings
    pub fn create_edsr(scale_factor: usize) -> Result<EDSR> {
        EDSR::new(scale_factor, 64, 16) // 64 features, 16 blocks
    }

    /// Compute PSNR between two images
    pub fn compute_psnr(pred: &Tensor<f32>, target: &Tensor<f32>, max_val: f32) -> Result<f32> {
        let mse = pred.sub(target)?.pow(2.0)?.mean(None, false)?;
        let mse_val = mse.item()?;

        if mse_val == 0.0 {
            Ok(f32::INFINITY)
        } else {
            let psnr = 20.0 * (max_val / mse_val.sqrt()).log10();
            Ok(psnr)
        }
    }

    /// Compute SSIM between two images (simplified version)
    pub fn compute_ssim(pred: &Tensor<f32>, target: &Tensor<f32>) -> Result<f32> {
        let mu1 = pred.mean(None, false)?;
        let mu2 = target.mean(None, false)?;

        let mu1_sq = mu1.pow(2.0)?;
        let mu2_sq = mu2.pow(2.0)?;
        let mu1_mu2 = mu1.mul(&mu2)?;

        let sigma1_sq = pred.pow(2.0)?.mean(None, false)?.sub(&mu1_sq)?;
        let sigma2_sq = target.pow(2.0)?.mean(None, false)?.sub(&mu2_sq)?;
        let sigma12 = pred.mul(target)?.mean(None, false)?.sub(&mu1_mu2)?;

        let c1 = 0.01_f32.powi(2);
        let c2 = 0.03_f32.powi(2);

        let numerator = mu1_mu2
            .mul_scalar(2.0)?
            .add_scalar(c1)?
            .mul(&sigma12.mul_scalar(2.0)?.add_scalar(c2)?)?;
        let denominator = mu1_sq
            .add(&mu2_sq)?
            .add_scalar(c1)?
            .mul(&sigma1_sq.add(&sigma2_sq)?.add_scalar(c2)?)?;

        let ssim = numerator.div(&denominator)?;
        Ok(ssim.item()?)
    }

    /// Preprocess image for super-resolution training
    pub fn preprocess_for_training(
        hr_image: &Tensor<f32>,
        scale_factor: usize,
    ) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let shape = hr_image.shape();
        let (_, _, height, width) = (
            shape.dims()[0],
            shape.dims()[1],
            shape.dims()[2],
            shape.dims()[3],
        );

        // Create low-resolution version
        let lr_height = height / scale_factor;
        let lr_width = width / scale_factor;
        let lr_image = crate::ops::resize(hr_image, (lr_width, lr_height))?;

        Ok((lr_image, hr_image.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::rand;

    #[test]
    fn test_srcnn() {
        let model = SRCNN::new(2).unwrap();
        let lr_input = rand(&[1, 1, 64, 64]).unwrap();
        let hr_input = model.preprocess(&lr_input).unwrap();
        let output = model.forward(&hr_input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 128, 128]);
    }

    #[test]
    fn test_espcn() {
        let model = ESPCN::new(3).unwrap();
        let input = rand(&[1, 1, 64, 64]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 192, 192]);
    }

    #[test]
    fn test_subpixel_conv() {
        let layer = SubPixelConv2d::new(32, 1, 3, 2).unwrap();
        let input = rand(&[1, 32, 64, 64]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 128, 128]);
    }

    #[test]
    fn test_edsr() {
        let model = EDSR::new(2, 32, 4).unwrap();
        let input = rand(&[1, 3, 64, 64]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 3, 128, 128]);
    }

    #[test]
    fn test_pixel_shuffle() {
        let layer = SubPixelConv2d::new(16, 4, 3, 2).unwrap();
        let input = rand(&[1, 16, 32, 32]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 4, 64, 64]);
    }
}
