//! Neural Style Transfer implementations
//!
//! This module provides implementations of neural style transfer techniques including:
//! - Gatys et al. neural style transfer
//! - Fast neural style transfer networks
//! - Perceptual loss functions

use crate::{Result, VisionError};
use torsh_nn::functional::pooling::max_pool2d;
use torsh_nn::prelude::*;
use torsh_tensor::prelude::*;
use torsh_tensor::{stats::StatMode, Tensor};

/// VGG19-based perceptual loss network for style transfer
pub struct VGGPerceptualLoss {
    conv1_1: Conv2d,
    conv1_2: Conv2d,
    conv2_1: Conv2d,
    conv2_2: Conv2d,
    conv3_1: Conv2d,
    conv3_2: Conv2d,
    conv3_3: Conv2d,
    conv3_4: Conv2d,
    conv4_1: Conv2d,
    conv4_2: Conv2d,
    conv4_3: Conv2d,
    conv4_4: Conv2d,
    conv5_1: Conv2d,
}

impl VGGPerceptualLoss {
    pub fn new() -> Result<Self> {
        Ok(Self {
            conv1_1: Conv2d::new(3, 64, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv1_2: Conv2d::new(64, 64, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv2_1: Conv2d::new(64, 128, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv2_2: Conv2d::new(128, 128, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv3_1: Conv2d::new(128, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv3_2: Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv3_3: Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv3_4: Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv4_1: Conv2d::new(256, 512, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv4_2: Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv4_3: Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv4_4: Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv5_1: Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
        })
    }

    /// Extract features at different layers for perceptual loss
    pub fn extract_features(&self, x: &Tensor<f32>) -> Result<Vec<Tensor<f32>>> {
        let mut features = Vec::new();

        let mut x = self.conv1_1.forward(x)?.relu()?;
        let mut x = self.conv1_2.forward(&x)?.relu()?;
        features.push(x.clone()); // relu1_2

        let mut x = max_pool2d(&x, (2, 2), Some((2, 2)), Some((0, 0)), None)?;
        let mut x = self.conv2_1.forward(&x)?.relu()?;
        let mut x = self.conv2_2.forward(&x)?.relu()?;
        features.push(x.clone()); // relu2_2

        let mut x = max_pool2d(&x, (2, 2), Some((2, 2)), Some((0, 0)), None)?;
        let mut x = self.conv3_1.forward(&x)?.relu()?;
        let mut x = self.conv3_2.forward(&x)?.relu()?;
        let mut x = self.conv3_3.forward(&x)?.relu()?;
        let mut x = self.conv3_4.forward(&x)?.relu()?;
        features.push(x.clone()); // relu3_4

        let mut x = max_pool2d(&x, (2, 2), Some((2, 2)), Some((0, 0)), None)?;
        let mut x = self.conv4_1.forward(&x)?.relu()?;
        let mut x = self.conv4_2.forward(&x)?.relu()?;
        let mut x = self.conv4_3.forward(&x)?.relu()?;
        let mut x = self.conv4_4.forward(&x)?.relu()?;
        features.push(x.clone()); // relu4_4

        let mut x = max_pool2d(&x, (2, 2), Some((2, 2)), Some((0, 0)), None)?;
        let x = self.conv5_1.forward(&x)?.relu()?;
        features.push(x); // relu5_1

        Ok(features)
    }

    /// Compute content loss between two feature maps
    pub fn content_loss(
        &self,
        generated: &Tensor<f32>,
        target: &Tensor<f32>,
    ) -> Result<Tensor<f32>> {
        let diff = generated.sub(target)?;
        let squared_diff = diff.pow(2.0)?;
        let loss = squared_diff.mean(None, false);
        Ok(loss?)
    }

    /// Compute style loss using Gram matrices
    pub fn style_loss(&self, generated: &Tensor<f32>, target: &Tensor<f32>) -> Result<Tensor<f32>> {
        let gram_gen = self.gram_matrix(generated)?;
        let gram_target = self.gram_matrix(target)?;

        let diff = gram_gen.sub(&gram_target)?;
        let squared_diff = diff.pow(2.0)?;
        let loss = squared_diff.mean(None, false);
        Ok(loss?)
    }

    /// Compute Gram matrix for style representation
    fn gram_matrix(&self, features: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = features.shape();
        let (batch_size, channels, height, width) = (
            shape.dims()[0],
            shape.dims()[1],
            shape.dims()[2],
            shape.dims()[3],
        );

        // Reshape to (batch_size, channels, height * width)
        let features_flat =
            features.view(&[batch_size as i32, channels as i32, (height * width) as i32])?;

        // Manual batched matmul for Gram matrix computation
        let mut batch_grams = Vec::new();

        for b in 0..batch_size {
            // Extract features for this batch: [channels, height*width]
            let batch_features = features_flat.narrow(0, b as i64, 1)?.squeeze(0)?; // [channels, height*width]
            let batch_features_t = batch_features.transpose(0, 1)?; // [height*width, channels]

            // Compute Gram matrix: [channels, height*width] @ [height*width, channels] = [channels, channels]
            let gram = batch_features.matmul(&batch_features_t)?;
            batch_grams.push(gram);
        }

        // Stack results back into batch: [batch_size, channels, channels]
        let mut gram_data = Vec::new();
        let gram_size = channels * channels;
        for gram_tensor in &batch_grams {
            let data = gram_tensor.to_vec()?;
            gram_data.extend(data);
        }
        let gram = Tensor::from_vec(gram_data, &[batch_size, channels, channels])?;

        // Normalize by number of elements
        let normalization_factor = (height * width) as f32;
        let normalized_gram = gram.div_scalar(normalization_factor)?;

        Ok(normalized_gram)
    }
}

/// Fast neural style transfer network
pub struct FastStyleTransferNet {
    encoder: StyleEncoder,
    residual_blocks: Vec<ResidualBlock>,
    decoder: StyleDecoder,
}

impl FastStyleTransferNet {
    pub fn new() -> Result<Self> {
        let mut residual_blocks = Vec::new();
        for _ in 0..5 {
            residual_blocks.push(ResidualBlock::new(128)?);
        }

        Ok(Self {
            encoder: StyleEncoder::new()?,
            residual_blocks,
            decoder: StyleDecoder::new()?,
        })
    }

    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut x = self.encoder.forward(x)?;

        for block in &self.residual_blocks {
            x = block.forward(&x)?;
        }

        let x = self.decoder.forward(&x)?;
        Ok(x)
    }
}

/// Encoder network for style transfer
pub struct StyleEncoder {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    instance_norm1: InstanceNorm2d,
    instance_norm2: InstanceNorm2d,
    instance_norm3: InstanceNorm2d,
}

impl StyleEncoder {
    pub fn new() -> Result<Self> {
        Ok(Self {
            conv1: Conv2d::new(3, 32, (9, 9), (1, 1), (4, 4), (1, 1), false, 1),
            conv2: Conv2d::new(32, 64, (3, 3), (2, 2), (1, 1), (1, 1), false, 1),
            conv3: Conv2d::new(64, 128, (3, 3), (2, 2), (1, 1), (1, 1), false, 1),
            instance_norm1: InstanceNorm2d::new(32)?,
            instance_norm2: InstanceNorm2d::new(64)?,
            instance_norm3: InstanceNorm2d::new(128)?,
        })
    }

    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let x = self.conv1.forward(x)?;
        let x = self.instance_norm1.forward(&x)?;
        let x = x.relu()?;

        let x = self.conv2.forward(&x)?;
        let x = self.instance_norm2.forward(&x)?;
        let x = x.relu()?;

        let x = self.conv3.forward(&x)?;
        let x = self.instance_norm3.forward(&x)?;
        let x = x.relu()?;

        Ok(x)
    }
}

/// Decoder network for style transfer
pub struct StyleDecoder {
    conv1: ConvTranspose2d,
    conv2: ConvTranspose2d,
    conv3: Conv2d,
    instance_norm1: InstanceNorm2d,
    instance_norm2: InstanceNorm2d,
}

impl StyleDecoder {
    pub fn new() -> Result<Self> {
        Ok(Self {
            conv1: ConvTranspose2d::new(128, 64, (3, 3), (2, 2), (1, 1), (1, 1), (1, 1), false, 1),
            conv2: ConvTranspose2d::new(64, 32, (3, 3), (2, 2), (1, 1), (1, 1), (1, 1), false, 1),
            conv3: Conv2d::new(32, 3, (9, 9), (1, 1), (4, 4), (1, 1), false, 1),
            instance_norm1: InstanceNorm2d::new(64)?,
            instance_norm2: InstanceNorm2d::new(32)?,
        })
    }

    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let x = self.conv1.forward(x)?;
        let x = self.instance_norm1.forward(&x)?;
        let x = x.relu()?;

        let x = self.conv2.forward(&x)?;
        let x = self.instance_norm2.forward(&x)?;
        let x = x.relu()?;

        let x = self.conv3.forward(&x)?;
        let x = x.tanh()?; // Output in [-1, 1] range

        Ok(x)
    }
}

/// Residual block for style transfer network
pub struct ResidualBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    instance_norm1: InstanceNorm2d,
    instance_norm2: InstanceNorm2d,
}

impl ResidualBlock {
    pub fn new(channels: usize) -> Result<Self> {
        Ok(Self {
            conv1: Conv2d::new(channels, channels, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            conv2: Conv2d::new(channels, channels, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            instance_norm1: InstanceNorm2d::new(channels)?,
            instance_norm2: InstanceNorm2d::new(channels)?,
        })
    }

    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let residual = x;

        let out = self.conv1.forward(x)?;
        let out = self.instance_norm1.forward(&out)?;
        let out = out.relu()?;

        let out = self.conv2.forward(&out)?;
        let out = self.instance_norm2.forward(&out)?;

        let out = out.add(residual)?;
        Ok(out)
    }
}

/// Instance normalization layer
pub struct InstanceNorm2d {
    eps: f32,
    affine: bool,
    weight: Option<Parameter>,
    bias: Option<Parameter>,
}

impl InstanceNorm2d {
    pub fn new(num_features: usize) -> Result<Self> {
        let weight = Some(Parameter::new(ones(&[num_features]).unwrap()));
        let bias = Some(Parameter::new(zeros(&[num_features]).unwrap()));

        Ok(Self {
            eps: 1e-5,
            affine: true,
            weight,
            bias,
        })
    }

    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();
        let (batch_size, channels, height, width) = (
            shape.dims()[0],
            shape.dims()[1],
            shape.dims()[2],
            shape.dims()[3],
        );

        // Manual computation to avoid .item() issues in variance calculation
        let input_reshaped =
            input.view(&[batch_size as i32, channels as i32, (height * width) as i32])?;

        let mut batch_means = Vec::new();
        let mut batch_variances = Vec::new();

        for b in 0..batch_size {
            for c in 0..channels {
                // Get channel data for this batch and channel: [H*W]
                let channel_data = input_reshaped
                    .narrow(0, b as i64, 1)?
                    .narrow(1, c as i64, 1)?
                    .squeeze(0)?
                    .squeeze(0)?;
                let channel_vec = channel_data.to_vec()?;

                // Compute mean
                let sum: f32 = channel_vec.iter().sum();
                let mean_val = sum / (height * width) as f32;
                batch_means.push(mean_val);

                // Compute variance
                let var_sum: f32 = channel_vec.iter().map(|&x| (x - mean_val).powi(2)).sum();
                let var_val = var_sum / (height * width) as f32;
                batch_variances.push(var_val);
            }
        }

        // Create mean and variance tensors
        let mean_tensor = Tensor::from_vec(batch_means, &[batch_size, channels])?;
        let variance_tensor = Tensor::from_vec(batch_variances, &[batch_size, channels])?;

        // Normalize
        let mean_expanded = mean_tensor.view(&[batch_size as i32, channels as i32, 1, 1])?;
        let variance_expanded =
            variance_tensor.view(&[batch_size as i32, channels as i32, 1, 1])?;

        let normalized = input.sub(&mean_expanded)?;
        let std = variance_expanded.add_scalar(self.eps)?.sqrt()?;
        let normalized = normalized.div(&std)?;

        if self.affine {
            if let (Some(weight), Some(bias)) = (&self.weight, &self.bias) {
                let weight_expanded = weight.tensor().read().view(&[1, channels as i32, 1, 1])?;
                let bias_expanded = bias.tensor().read().view(&[1, channels as i32, 1, 1])?;

                let scaled = normalized.mul(&weight_expanded)?;
                let output = scaled.add(&bias_expanded)?;
                Ok(output)
            } else {
                Ok(normalized)
            }
        } else {
            Ok(normalized)
        }
    }
}

/// Neural style transfer training loss
pub struct StyleTransferLoss {
    perceptual_net: VGGPerceptualLoss,
    content_weight: f32,
    style_weight: f32,
    content_layers: Vec<usize>,
    style_layers: Vec<usize>,
}

impl StyleTransferLoss {
    pub fn new(content_weight: f32, style_weight: f32) -> Result<Self> {
        Ok(Self {
            perceptual_net: VGGPerceptualLoss::new()?,
            content_weight,
            style_weight,
            content_layers: vec![3],           // relu4_4
            style_layers: vec![0, 1, 2, 3, 4], // All layers
        })
    }

    pub fn compute_loss(
        &self,
        generated: &Tensor<f32>,
        content_target: &Tensor<f32>,
        style_target: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
        let gen_features = self.perceptual_net.extract_features(generated)?;
        let content_features = self.perceptual_net.extract_features(content_target)?;
        let style_features = self.perceptual_net.extract_features(style_target)?;

        // Content loss
        let mut content_loss = zeros(&[1])?;
        for &layer_idx in &self.content_layers {
            let layer_loss = self
                .perceptual_net
                .content_loss(&gen_features[layer_idx], &content_features[layer_idx])?;
            content_loss = content_loss.add(&layer_loss)?;
        }
        content_loss = content_loss.mul_scalar(self.content_weight)?;

        // Style loss
        let mut style_loss = zeros(&[1])?;
        for &layer_idx in &self.style_layers {
            let layer_loss = self
                .perceptual_net
                .style_loss(&gen_features[layer_idx], &style_features[layer_idx])?;
            style_loss = style_loss.add(&layer_loss)?;
        }
        style_loss = style_loss.mul_scalar(self.style_weight)?;

        // Total loss
        let total_loss = content_loss.add(&style_loss)?;

        Ok((total_loss, content_loss, style_loss))
    }
}

/// Style transfer utilities
pub mod style_transfer_utils {
    use super::*;

    /// Preprocess image for style transfer (normalize to [-1, 1])
    pub fn preprocess_image(image: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Assume input is in [0, 1] range, convert to [-1, 1]
        let preprocessed = image.mul_scalar(2.0)?.sub_scalar(1.0)?;
        Ok(preprocessed)
    }

    /// Postprocess image from style transfer (normalize to [0, 1])
    pub fn postprocess_image(image: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Convert from [-1, 1] to [0, 1]
        let postprocessed = image.add_scalar(1.0)?.div_scalar(2.0)?;
        let clamped = postprocessed.clamp(0.0, 1.0)?;
        Ok(clamped)
    }

    /// Create default style transfer network
    pub fn create_style_transfer_network() -> Result<FastStyleTransferNet> {
        FastStyleTransferNet::new()
    }

    /// Create default perceptual loss
    pub fn create_perceptual_loss(
        content_weight: f32,
        style_weight: f32,
    ) -> Result<StyleTransferLoss> {
        StyleTransferLoss::new(content_weight, style_weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::rand;

    #[test]
    #[ignore = "Slow test (>60s) - VGG model initialization is heavy"]
    fn test_vgg_perceptual_loss() {
        let vgg = VGGPerceptualLoss::new().unwrap();
        let input = rand(&[1, 3, 224, 224]).unwrap();
        let features = vgg.extract_features(&input).unwrap();
        assert_eq!(features.len(), 5);
    }

    #[test]
    #[ignore = "Slow test (>60s) - Style transfer network initialization is heavy"]
    fn test_fast_style_transfer_net() {
        let net = FastStyleTransferNet::new().unwrap();
        let input = rand(&[1, 3, 256, 256]).unwrap();
        let output = net.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 3, 256, 256]);
    }

    #[test]
    fn test_instance_norm() {
        let norm = InstanceNorm2d::new(64).unwrap();
        let input = rand(&[2, 64, 32, 32]).unwrap();
        let output = norm.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), input.shape().dims());
    }

    #[test]
    fn test_gram_matrix() {
        let vgg = VGGPerceptualLoss::new().unwrap();
        let features = rand(&[1, 64, 32, 32]).unwrap();
        let gram = vgg.gram_matrix(&features).unwrap();
        assert_eq!(gram.shape().dims(), &[1, 64, 64]);
    }
}
