//! EfficientNet Architecture Components
//!
//! This module provides comprehensive EfficientNet building blocks and utilities including:
//! - EfficientNet compound scaling
//! - Advanced MBConv variants
//! - Stochastic depth for training
//! - Complete EfficientNet models (B0-B7)

use crate::container::Sequential;
use crate::layers::activation::SiLU;
use crate::layers::blocks::MBConvBlock;
use crate::layers::{BatchNorm2d, Conv2d, Dropout, Linear};
use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// EfficientNet compound scaling configuration
#[derive(Debug, Clone)]
pub struct EfficientNetConfig {
    /// Width multiplier (channels)
    pub width_multiplier: f32,
    /// Depth multiplier (layers)
    pub depth_multiplier: f32,
    /// Input resolution
    pub input_resolution: usize,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Stochastic depth rate
    pub stochastic_depth_rate: f32,
    /// Number of classes for classification
    pub num_classes: usize,
}

impl EfficientNetConfig {
    /// EfficientNet-B0 configuration
    pub fn b0() -> Self {
        Self {
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
            input_resolution: 224,
            dropout_rate: 0.2,
            stochastic_depth_rate: 0.2,
            num_classes: 1000,
        }
    }

    /// EfficientNet-B1 configuration
    pub fn b1() -> Self {
        Self {
            width_multiplier: 1.0,
            depth_multiplier: 1.1,
            input_resolution: 240,
            dropout_rate: 0.2,
            stochastic_depth_rate: 0.2,
            num_classes: 1000,
        }
    }

    /// EfficientNet-B2 configuration
    pub fn b2() -> Self {
        Self {
            width_multiplier: 1.1,
            depth_multiplier: 1.2,
            input_resolution: 260,
            dropout_rate: 0.3,
            stochastic_depth_rate: 0.2,
            num_classes: 1000,
        }
    }

    /// EfficientNet-B3 configuration
    pub fn b3() -> Self {
        Self {
            width_multiplier: 1.2,
            depth_multiplier: 1.4,
            input_resolution: 300,
            dropout_rate: 0.3,
            stochastic_depth_rate: 0.2,
            num_classes: 1000,
        }
    }

    /// EfficientNet-B4 configuration
    pub fn b4() -> Self {
        Self {
            width_multiplier: 1.4,
            depth_multiplier: 1.8,
            input_resolution: 380,
            dropout_rate: 0.4,
            stochastic_depth_rate: 0.2,
            num_classes: 1000,
        }
    }

    /// EfficientNet-B5 configuration
    pub fn b5() -> Self {
        Self {
            width_multiplier: 1.6,
            depth_multiplier: 2.2,
            input_resolution: 456,
            dropout_rate: 0.4,
            stochastic_depth_rate: 0.2,
            num_classes: 1000,
        }
    }

    /// EfficientNet-B6 configuration
    pub fn b6() -> Self {
        Self {
            width_multiplier: 1.8,
            depth_multiplier: 2.6,
            input_resolution: 528,
            dropout_rate: 0.5,
            stochastic_depth_rate: 0.2,
            num_classes: 1000,
        }
    }

    /// EfficientNet-B7 configuration
    pub fn b7() -> Self {
        Self {
            width_multiplier: 2.0,
            depth_multiplier: 3.1,
            input_resolution: 600,
            dropout_rate: 0.5,
            stochastic_depth_rate: 0.2,
            num_classes: 1000,
        }
    }

    /// Calculate scaled width for given base width
    pub fn scale_width(&self, base_width: usize) -> usize {
        let scaled = (base_width as f32 * self.width_multiplier) as usize;
        // Round to nearest multiple of 8 for efficient computation
        ((scaled + 4) / 8) * 8
    }

    /// Calculate scaled depth for given base depth
    pub fn scale_depth(&self, base_depth: usize) -> usize {
        (base_depth as f32 * self.depth_multiplier).ceil() as usize
    }
}

/// Advanced MBConv block with Stochastic Depth
pub struct MBConvWithStochasticDepth {
    base: ModuleBase,
    mbconv: MBConvBlock,
    stochastic_depth_prob: f32,
    use_shortcut: bool,
}

impl MBConvWithStochasticDepth {
    /// Create a new MBConv block with stochastic depth
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        expansion_ratio: usize,
        se_ratio: f32,
        stochastic_depth_prob: f32,
    ) -> Result<Self> {
        let mbconv = MBConvBlock::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            expansion_ratio,
            Some(se_ratio),
            0.0, // drop_rate
        )?;

        let use_shortcut = stride == 1 && in_channels == out_channels;

        Ok(Self {
            base: ModuleBase::new(),
            mbconv,
            stochastic_depth_prob,
            use_shortcut,
        })
    }

    /// Apply stochastic depth during training
    fn apply_stochastic_depth(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        if !self.training() || self.stochastic_depth_prob == 0.0 {
            return x.add_op(residual);
        }

        // During training, randomly drop the entire block
        let keep_prob = 1.0 - self.stochastic_depth_prob;

        // Simplified stochastic depth - in practice would use proper random sampling
        // For now, we'll apply a scaling factor
        let scaled_x = x.mul_scalar(keep_prob)?;
        scaled_x.add_op(residual)
    }
}

impl Module for MBConvWithStochasticDepth {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.mbconv.forward(input)?;

        if self.use_shortcut {
            self.apply_stochastic_depth(&output, input)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.mbconv.parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.mbconv.named_parameters()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
        // Note: mbconv.train() would need to be called if it's mutable
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        // Note: mbconv.eval() would need to be called if it's mutable
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
        // Note: mbconv.set_training(training) would need to be called if it's mutable
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
        // Note: mbconv.to_device(device) would need to be called if it's mutable
    }
}

/// EfficientNet block configuration
#[derive(Debug, Clone)]
pub struct BlockConfig {
    /// Input channels
    pub input_filters: usize,
    /// Output channels
    pub output_filters: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Expansion ratio
    pub expansion_ratio: usize,
    /// SE ratio
    pub se_ratio: f32,
    /// Number of repeats
    pub num_repeat: usize,
}

impl BlockConfig {
    /// Scale the block configuration according to EfficientNet scaling
    pub fn scale(&self, config: &EfficientNetConfig) -> Self {
        Self {
            input_filters: config.scale_width(self.input_filters),
            output_filters: config.scale_width(self.output_filters),
            kernel_size: self.kernel_size,
            stride: self.stride,
            expansion_ratio: self.expansion_ratio,
            se_ratio: self.se_ratio,
            num_repeat: config.scale_depth(self.num_repeat),
        }
    }
}

/// EfficientNet backbone architecture
pub struct EfficientNetBackbone {
    base: ModuleBase,
    stem: Sequential,
    blocks: Vec<MBConvWithStochasticDepth>,
    head: Sequential,
    #[allow(dead_code)]
    config: EfficientNetConfig,
}

impl EfficientNetBackbone {
    /// Create a new EfficientNet backbone
    pub fn new(config: EfficientNetConfig) -> Result<Self> {
        let base_blocks = Self::get_base_block_configs();
        let scaled_blocks: Vec<BlockConfig> = base_blocks
            .iter()
            .map(|block| block.scale(&config))
            .collect();

        let base = ModuleBase::new();

        // Stem: Conv2d + BatchNorm + Swish
        let stem_channels = config.scale_width(32);
        let stem = Sequential::new()
            .add(Conv2d::new(
                3,
                stem_channels,
                (3, 3),
                (2, 2),
                (1, 1),
                (1, 1),
                false,
                1,
            ))
            .add(BatchNorm2d::new(stem_channels)?)
            .add(SiLU::new());

        // Build MBConv blocks
        let mut blocks = Vec::new();
        let mut total_blocks = 0;
        for block_config in &scaled_blocks {
            total_blocks += block_config.num_repeat;
        }

        let mut block_index = 0;
        for block_config in &scaled_blocks {
            for i in 0..block_config.num_repeat {
                let input_filters = if i == 0 {
                    block_config.input_filters
                } else {
                    block_config.output_filters
                };

                let stride = if i == 0 { block_config.stride } else { 1 };

                // Calculate stochastic depth probability
                let stochastic_depth_prob =
                    config.stochastic_depth_rate * (block_index as f32 / total_blocks as f32);

                let block = MBConvWithStochasticDepth::new(
                    input_filters,
                    block_config.output_filters,
                    block_config.kernel_size,
                    stride,
                    block_config.expansion_ratio,
                    block_config.se_ratio,
                    stochastic_depth_prob,
                );

                blocks.push(block?);
                block_index += 1;
            }
        }

        // Head: Conv2d + BatchNorm + Swish + GlobalAvgPool
        let head_channels = config.scale_width(1280);
        let last_block_channels = scaled_blocks
            .last()
            .expect("scaled_blocks should not be empty")
            .output_filters;
        let head = Sequential::new()
            .add(Conv2d::new(
                last_block_channels,
                head_channels,
                (1, 1),
                (1, 1),
                (0, 0),
                (1, 1),
                false,
                1,
            ))
            .add(BatchNorm2d::new(head_channels)?)
            .add(SiLU::new());

        Ok(Self {
            base,
            stem,
            blocks,
            head,
            config,
        })
    }

    /// Get base block configurations for EfficientNet-B0
    fn get_base_block_configs() -> Vec<BlockConfig> {
        vec![
            BlockConfig {
                input_filters: 32,
                output_filters: 16,
                kernel_size: 3,
                stride: 1,
                expansion_ratio: 1,
                se_ratio: 0.25,
                num_repeat: 1,
            },
            BlockConfig {
                input_filters: 16,
                output_filters: 24,
                kernel_size: 3,
                stride: 2,
                expansion_ratio: 6,
                se_ratio: 0.25,
                num_repeat: 2,
            },
            BlockConfig {
                input_filters: 24,
                output_filters: 40,
                kernel_size: 5,
                stride: 2,
                expansion_ratio: 6,
                se_ratio: 0.25,
                num_repeat: 2,
            },
            BlockConfig {
                input_filters: 40,
                output_filters: 80,
                kernel_size: 3,
                stride: 2,
                expansion_ratio: 6,
                se_ratio: 0.25,
                num_repeat: 3,
            },
            BlockConfig {
                input_filters: 80,
                output_filters: 112,
                kernel_size: 5,
                stride: 1,
                expansion_ratio: 6,
                se_ratio: 0.25,
                num_repeat: 3,
            },
            BlockConfig {
                input_filters: 112,
                output_filters: 192,
                kernel_size: 5,
                stride: 2,
                expansion_ratio: 6,
                se_ratio: 0.25,
                num_repeat: 4,
            },
            BlockConfig {
                input_filters: 192,
                output_filters: 320,
                kernel_size: 3,
                stride: 1,
                expansion_ratio: 6,
                se_ratio: 0.25,
                num_repeat: 1,
            },
        ]
    }

    /// Extract features from the backbone
    pub fn extract_features(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = self.stem.forward(input)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        self.head.forward(&x)
    }
}

impl Module for EfficientNetBackbone {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.extract_features(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Stem parameters
        for (name, param) in self.stem.parameters() {
            params.insert(format!("stem.{}", name), param);
        }

        // Block parameters
        for (i, block) in self.blocks.iter().enumerate() {
            for (name, param) in block.parameters() {
                params.insert(format!("blocks.{}.{}", i, name), param);
            }
        }

        // Head parameters
        for (name, param) in self.head.parameters() {
            params.insert(format!("head.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
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
}

/// Complete EfficientNet model with classification head
pub struct EfficientNet {
    base: ModuleBase,
    backbone: EfficientNetBackbone,
    classifier: Sequential,
    #[allow(dead_code)]
    config: EfficientNetConfig,
}

impl EfficientNet {
    /// Create a new EfficientNet model
    pub fn new(config: EfficientNetConfig) -> Result<Self> {
        let backbone = EfficientNetBackbone::new(config.clone())?;

        // Classification head
        let head_channels = config.scale_width(1280);
        let classifier = Sequential::new()
            .add(Dropout::new(config.dropout_rate))
            .add(Linear::new(head_channels, config.num_classes, true));

        Ok(Self {
            base: ModuleBase::new(),
            backbone,
            classifier,
            config,
        })
    }

    /// Create EfficientNet-B0
    pub fn b0(num_classes: usize) -> Result<Self> {
        let mut config = EfficientNetConfig::b0();
        config.num_classes = num_classes;
        Self::new(config)
    }

    /// Create EfficientNet-B1
    pub fn b1(num_classes: usize) -> Result<Self> {
        let mut config = EfficientNetConfig::b1();
        config.num_classes = num_classes;
        Self::new(config)
    }

    /// Create EfficientNet-B2
    pub fn b2(num_classes: usize) -> Result<Self> {
        let mut config = EfficientNetConfig::b2();
        config.num_classes = num_classes;
        Self::new(config)
    }

    /// Create EfficientNet-B3
    pub fn b3(num_classes: usize) -> Result<Self> {
        let mut config = EfficientNetConfig::b3();
        config.num_classes = num_classes;
        Self::new(config)
    }

    /// Create EfficientNet-B4
    pub fn b4(num_classes: usize) -> Result<Self> {
        let mut config = EfficientNetConfig::b4();
        config.num_classes = num_classes;
        Self::new(config)
    }

    /// Create EfficientNet-B5
    pub fn b5(num_classes: usize) -> Result<Self> {
        let mut config = EfficientNetConfig::b5();
        config.num_classes = num_classes;
        Self::new(config)
    }

    /// Create EfficientNet-B6
    pub fn b6(num_classes: usize) -> Result<Self> {
        let mut config = EfficientNetConfig::b6();
        config.num_classes = num_classes;
        Self::new(config)
    }

    /// Create EfficientNet-B7
    pub fn b7(num_classes: usize) -> Result<Self> {
        let mut config = EfficientNetConfig::b7();
        config.num_classes = num_classes;
        Self::new(config)
    }

    /// Get the backbone for feature extraction
    pub fn backbone(&self) -> &EfficientNetBackbone {
        &self.backbone
    }

    /// Extract features without classification
    pub fn extract_features(&self, input: &Tensor) -> Result<Tensor> {
        self.backbone.extract_features(input)
    }
}

impl Module for EfficientNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.backbone.forward(input)?;

        // Apply global average pooling to reduce spatial dimensions
        let features_shape = features.shape();
        let shape = features_shape.dims();
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        // Manual global average pooling: sum over spatial dimensions and divide
        let mut pooled_data = vec![0.0f32; batch_size * channels];
        let features_data = features.to_vec()?;

        for b in 0..batch_size {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for h in 0..height {
                    for w in 0..width {
                        let idx =
                            b * (channels * height * width) + c * (height * width) + h * width + w;
                        if idx < features_data.len() {
                            sum += features_data[idx];
                        }
                    }
                }
                pooled_data[b * channels + c] = sum / (height * width) as f32;
            }
        }

        // Create pooled tensor with shape [batch_size, channels]
        let pooled = Tensor::from_vec(pooled_data, &[batch_size, channels])?;

        self.classifier.forward(&pooled)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Backbone parameters
        for (name, param) in self.backbone.parameters() {
            params.insert(format!("backbone.{}", name), param);
        }

        // Classifier parameters
        for (name, param) in self.classifier.parameters() {
            params.insert(format!("classifier.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
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
}

/// Utilities for EfficientNet training and deployment
pub mod utils {
    use super::*;

    /// Calculate model parameters for different EfficientNet variants
    pub fn calculate_model_params(config: &EfficientNetConfig) -> (usize, f32) {
        // Rough parameter count estimation
        let base_params = 5_300_000; // EfficientNet-B0 approximate params
        let width_factor = config.width_multiplier.powi(2);
        let depth_factor = config.depth_multiplier;

        let estimated_params = (base_params as f32 * width_factor * depth_factor) as usize;
        let estimated_flops = estimated_params as f32 * config.input_resolution as f32 * 0.1;

        (estimated_params, estimated_flops)
    }

    /// Get recommended training settings for different model sizes
    pub fn get_training_config(config: &EfficientNetConfig) -> TrainingConfig {
        match config.width_multiplier {
            w if w <= 1.0 => TrainingConfig {
                batch_size: 128,
                learning_rate: 0.256,
                weight_decay: 1e-5,
                epochs: 350,
            },
            w if w <= 1.4 => TrainingConfig {
                batch_size: 64,
                learning_rate: 0.128,
                weight_decay: 1e-5,
                epochs: 350,
            },
            _ => TrainingConfig {
                batch_size: 32,
                learning_rate: 0.064,
                weight_decay: 1e-5,
                epochs: 350,
            },
        }
    }

    /// Create custom EfficientNet with specific scaling
    pub fn create_custom_efficientnet(
        width_multiplier: f32,
        depth_multiplier: f32,
        resolution: usize,
        num_classes: usize,
    ) -> Result<EfficientNet> {
        let config = EfficientNetConfig {
            width_multiplier,
            depth_multiplier,
            input_resolution: resolution,
            dropout_rate: 0.2 + (width_multiplier - 1.0) * 0.3,
            stochastic_depth_rate: 0.2,
            num_classes,
        };

        EfficientNet::new(config)
    }
}

/// Training configuration recommendations
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub epochs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_efficientnet_config() {
        let config = EfficientNetConfig::b0();
        assert_eq!(config.width_multiplier, 1.0);
        assert_eq!(config.depth_multiplier, 1.0);
        assert_eq!(config.input_resolution, 224);
    }

    #[test]
    fn test_scaling() {
        let config = EfficientNetConfig::b1();
        assert_eq!(config.scale_width(32), 32); // 32 * 1.0 = 32
        assert_eq!(config.scale_depth(3), 4); // ceil(3 * 1.1) = 4
    }

    #[test]
    fn test_block_config_scaling() {
        let block = BlockConfig {
            input_filters: 32,
            output_filters: 64,
            kernel_size: 3,
            stride: 1,
            expansion_ratio: 6,
            se_ratio: 0.25,
            num_repeat: 2,
        };

        let config = EfficientNetConfig::b2();
        let scaled = block.scale(&config);

        assert!(scaled.input_filters >= 32); // Should be scaled up
        assert!(scaled.output_filters >= 64); // Should be scaled up
        assert!(scaled.num_repeat >= 2); // Should be scaled up
    }

    #[test]
    fn test_efficientnet_creation() -> Result<()> {
        let model = EfficientNet::b0(1000)?;
        assert_eq!(model.config.num_classes, 1000);
        Ok(())
    }

    #[test]
    #[ignore]
    fn test_efficientnet_forward() -> Result<()> {
        let model = EfficientNet::b0(10)?;
        let input = ones(&[1, 3, 224, 224])?;

        let output = model.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 10]);

        Ok(())
    }

    #[test]
    fn test_mbconv_with_stochastic_depth() -> Result<()> {
        let block = MBConvWithStochasticDepth::new(32, 32, 3, 1, 6, 0.25, 0.1)?;
        let input = ones(&[1, 32, 56, 56])?;

        let output = block.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 32, 56, 56]);

        Ok(())
    }

    #[test]
    fn test_utils_calculate_params() {
        let config = EfficientNetConfig::b0();
        let (params, flops) = utils::calculate_model_params(&config);

        assert!(params > 1_000_000); // Should have reasonable number of parameters
        assert!(flops > 1_000_000.0); // Should have reasonable FLOPS
    }

    #[test]
    fn test_custom_efficientnet() -> Result<()> {
        let model = utils::create_custom_efficientnet(1.2, 1.4, 256, 100)?;
        assert_eq!(model.config.num_classes, 100);
        assert_eq!(model.config.input_resolution, 256);
        Ok(())
    }
}
