//! MobileNet Architecture Components
//!
//! This module provides comprehensive MobileNet building blocks and architectures including:
//! - Depthwise Separable Convolutions
//! - MobileNetV1 and MobileNetV2 components
//! - Inverted Residual blocks
//! - Width and resolution multiplier scaling
//! - Complete MobileNet models

use crate::container::Sequential;
use crate::layers::activation::ReLU6;
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

/// MobileNet configuration
#[derive(Debug, Clone)]
pub struct MobileNetConfig {
    /// Width multiplier for scaling channels
    pub width_multiplier: f32,
    /// Resolution multiplier for scaling input size
    pub resolution_multiplier: f32,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Number of classes
    pub num_classes: usize,
    /// Minimum channel depth
    pub min_depth: usize,
}

impl Default for MobileNetConfig {
    fn default() -> Self {
        Self {
            width_multiplier: 1.0,
            resolution_multiplier: 1.0,
            dropout_rate: 0.2,
            num_classes: 1000,
            min_depth: 8,
        }
    }
}

impl MobileNetConfig {
    /// Calculate scaled channels with minimum depth constraint
    pub fn scale_channels(&self, channels: usize) -> usize {
        let scaled = (channels as f32 * self.width_multiplier) as usize;
        let divisor = 8;
        let scaled_rounded = ((scaled + divisor / 2) / divisor) * divisor;
        scaled_rounded.max(self.min_depth)
    }

    /// Calculate scaled resolution
    pub fn scale_resolution(&self, resolution: usize) -> usize {
        (resolution as f32 * self.resolution_multiplier) as usize
    }
}

/// Depthwise Separable Convolution
///
/// Splits standard convolution into depthwise and pointwise operations
/// for significant parameter and computation reduction.
pub struct DepthwiseSeparableConv {
    base: ModuleBase,
    depthwise: Conv2d,
    depthwise_bn: BatchNorm2d,
    pointwise: Conv2d,
    pointwise_bn: BatchNorm2d,
    activation: ReLU6,
}

impl DepthwiseSeparableConv {
    /// Create a new depthwise separable convolution
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        Ok(Self {
            base: ModuleBase::new(),
            depthwise: Conv2d::new(
                in_channels,
                in_channels,
                (kernel_size, kernel_size),
                (stride, stride),
                (padding, padding),
                (1, 1),
                false,
                in_channels,
            ),
            depthwise_bn: BatchNorm2d::new(in_channels)?,
            pointwise: Conv2d::new(
                in_channels,
                out_channels,
                (1, 1),
                (1, 1),
                (0, 0),
                (1, 1),
                false,
                1,
            ),
            pointwise_bn: BatchNorm2d::new(out_channels)?,
            activation: ReLU6::new(),
        })
    }

    /// Create with default 3x3 kernel
    pub fn new_3x3(in_channels: usize, out_channels: usize, stride: usize) -> Result<Self> {
        Self::new(in_channels, out_channels, 3, stride, 1)
    }
}

impl Module for DepthwiseSeparableConv {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Depthwise convolution
        let mut x = self.depthwise.forward(input)?;
        x = self.depthwise_bn.forward(&x)?;
        x = self.activation.forward(&x)?;

        // Pointwise convolution
        x = self.pointwise.forward(&x)?;
        x = self.pointwise_bn.forward(&x)?;
        x = self.activation.forward(&x)?;

        Ok(x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.depthwise.parameters() {
            params.insert(format!("depthwise.{}", name), param);
        }
        for (name, param) in self.depthwise_bn.parameters() {
            params.insert(format!("depthwise_bn.{}", name), param);
        }
        for (name, param) in self.pointwise.parameters() {
            params.insert(format!("pointwise.{}", name), param);
        }
        for (name, param) in self.pointwise_bn.parameters() {
            params.insert(format!("pointwise_bn.{}", name), param);
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

/// Inverted Residual Block (MobileNetV2)
///
/// Uses inverted bottleneck design: expand -> depthwise -> project
/// with linear bottlenecks to preserve information.
pub struct InvertedResidual {
    base: ModuleBase,
    expand_conv: Option<Sequential>,
    depthwise_conv: Sequential,
    project_conv: Sequential,
    use_shortcut: bool,
    #[allow(dead_code)]
    stride: usize,
}

impl InvertedResidual {
    /// Create a new inverted residual block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        expansion_ratio: usize,
    ) -> Result<Self> {
        let expanded_channels = in_channels * expansion_ratio;
        let use_shortcut = stride == 1 && in_channels == out_channels;

        // Expansion phase (if expansion ratio > 1)
        let expand_conv = if expansion_ratio > 1 {
            Some(
                Sequential::new()
                    .add(Conv2d::new(
                        in_channels,
                        expanded_channels,
                        (1, 1),
                        (1, 1),
                        (0, 0),
                        (1, 1),
                        false,
                        1,
                    ))
                    .add(BatchNorm2d::new(expanded_channels)?)
                    .add(ReLU6::new()),
            )
        } else {
            None
        };

        // Depthwise convolution
        let depthwise_conv = Sequential::new()
            .add(Conv2d::new(
                expanded_channels,
                expanded_channels,
                (3, 3),
                (stride, stride),
                (1, 1),
                (1, 1),
                false,
                expanded_channels,
            ))
            .add(BatchNorm2d::new(expanded_channels)?)
            .add(ReLU6::new());

        // Projection phase (linear bottleneck - no activation)
        let project_conv = Sequential::new()
            .add(Conv2d::new(
                expanded_channels,
                out_channels,
                (1, 1),
                (1, 1),
                (0, 0),
                (1, 1),
                false,
                1,
            ))
            .add(BatchNorm2d::new(out_channels)?);

        Ok(Self {
            base: ModuleBase::new(),
            expand_conv,
            depthwise_conv,
            project_conv,
            use_shortcut,
            stride,
        })
    }

    /// Create with typical expansion ratio of 6
    pub fn new_standard(in_channels: usize, out_channels: usize, stride: usize) -> Result<Self> {
        Self::new(in_channels, out_channels, stride, 6)
    }
}

impl Module for InvertedResidual {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        // Expansion
        if let Some(ref expand) = self.expand_conv {
            x = expand.forward(&x)?;
        }

        // Depthwise
        x = self.depthwise_conv.forward(&x)?;

        // Projection
        x = self.project_conv.forward(&x)?;

        // Residual connection
        if self.use_shortcut {
            x = x.add_op(input)?;
        }

        Ok(x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if let Some(ref expand) = self.expand_conv {
            for (name, param) in expand.parameters() {
                params.insert(format!("expand.{}", name), param);
            }
        }

        for (name, param) in self.depthwise_conv.parameters() {
            params.insert(format!("depthwise.{}", name), param);
        }

        for (name, param) in self.project_conv.parameters() {
            params.insert(format!("project.{}", name), param);
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

/// MobileNetV1 block configuration
#[derive(Debug, Clone)]
pub struct MobileNetV1Block {
    pub in_channels: usize,
    pub out_channels: usize,
    pub stride: usize,
}

/// MobileNetV2 block configuration
#[derive(Debug, Clone)]
pub struct MobileNetV2Block {
    pub in_channels: usize,
    pub out_channels: usize,
    pub stride: usize,
    pub expansion_ratio: usize,
    pub num_repeat: usize,
}

/// MobileNetV1 Architecture
pub struct MobileNetV1 {
    base: ModuleBase,
    features: Sequential,
    classifier: Sequential,
    #[allow(dead_code)]
    config: MobileNetConfig,
}

impl MobileNetV1 {
    /// Create a new MobileNetV1
    pub fn new(config: MobileNetConfig) -> Result<Self> {
        let mut features = Sequential::new();

        // First standard convolution
        let first_channels = config.scale_channels(32);
        features = features
            .add(Conv2d::new(
                3,
                first_channels,
                (3, 3),
                (2, 2),
                (1, 1),
                (1, 1),
                false,
                1,
            ))
            .add(BatchNorm2d::new(first_channels)?)
            .add(ReLU6::new());

        // Depthwise separable convolution blocks
        let block_configs = Self::get_block_configs();
        let mut in_channels = first_channels;

        for block in block_configs {
            let out_channels = config.scale_channels(block.out_channels);
            features = features.add(DepthwiseSeparableConv::new_3x3(
                in_channels,
                out_channels,
                block.stride,
            )?);
            in_channels = out_channels;
        }

        // Note: Global average pooling would be applied during forward pass

        // Classifier
        let classifier = Sequential::new()
            .add(Dropout::new(config.dropout_rate))
            .add(Linear::new(in_channels, config.num_classes, true));

        Ok(Self {
            base: ModuleBase::new(),
            features,
            classifier,
            config,
        })
    }

    /// Get MobileNetV1 block configurations
    fn get_block_configs() -> Vec<MobileNetV1Block> {
        vec![
            MobileNetV1Block {
                in_channels: 32,
                out_channels: 64,
                stride: 1,
            },
            MobileNetV1Block {
                in_channels: 64,
                out_channels: 128,
                stride: 2,
            },
            MobileNetV1Block {
                in_channels: 128,
                out_channels: 128,
                stride: 1,
            },
            MobileNetV1Block {
                in_channels: 128,
                out_channels: 256,
                stride: 2,
            },
            MobileNetV1Block {
                in_channels: 256,
                out_channels: 256,
                stride: 1,
            },
            MobileNetV1Block {
                in_channels: 256,
                out_channels: 512,
                stride: 2,
            },
            MobileNetV1Block {
                in_channels: 512,
                out_channels: 512,
                stride: 1,
            },
            MobileNetV1Block {
                in_channels: 512,
                out_channels: 512,
                stride: 1,
            },
            MobileNetV1Block {
                in_channels: 512,
                out_channels: 512,
                stride: 1,
            },
            MobileNetV1Block {
                in_channels: 512,
                out_channels: 512,
                stride: 1,
            },
            MobileNetV1Block {
                in_channels: 512,
                out_channels: 512,
                stride: 1,
            },
            MobileNetV1Block {
                in_channels: 512,
                out_channels: 1024,
                stride: 2,
            },
            MobileNetV1Block {
                in_channels: 1024,
                out_channels: 1024,
                stride: 1,
            },
        ]
    }

    /// Create MobileNetV1 with standard width multipliers
    pub fn mobilenet_v1_0_25(num_classes: usize) -> Result<Self> {
        let config = MobileNetConfig {
            width_multiplier: 0.25,
            num_classes,
            ..Default::default()
        };
        Self::new(config)
    }

    pub fn mobilenet_v1_0_5(num_classes: usize) -> Result<Self> {
        let config = MobileNetConfig {
            width_multiplier: 0.5,
            num_classes,
            ..Default::default()
        };
        Self::new(config)
    }

    pub fn mobilenet_v1_0_75(num_classes: usize) -> Result<Self> {
        let config = MobileNetConfig {
            width_multiplier: 0.75,
            num_classes,
            ..Default::default()
        };
        Self::new(config)
    }

    pub fn mobilenet_v1_1_0(num_classes: usize) -> Result<Self> {
        let config = MobileNetConfig {
            width_multiplier: 1.0,
            num_classes,
            ..Default::default()
        };
        Self::new(config)
    }
}

impl Module for MobileNetV1 {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.features.forward(input)?;

        // Flatten for classifier
        let batch_size = features.shape().dims()[0];
        let feature_size = features.shape().dims().iter().skip(1).product::<usize>();
        let flattened = features.reshape(&[batch_size as i32, feature_size as i32])?;

        self.classifier.forward(&flattened)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.features.parameters() {
            params.insert(format!("features.{}", name), param);
        }

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

/// MobileNetV2 Architecture
pub struct MobileNetV2 {
    base: ModuleBase,
    features: Sequential,
    classifier: Sequential,
    #[allow(dead_code)]
    config: MobileNetConfig,
}

impl MobileNetV2 {
    /// Create a new MobileNetV2
    pub fn new(config: MobileNetConfig) -> Result<Self> {
        let mut features = Sequential::new();

        // First standard convolution
        let first_channels = config.scale_channels(32);
        features = features
            .add(Conv2d::new(
                3,
                first_channels,
                (3, 3),
                (2, 2),
                (1, 1),
                (1, 1),
                false,
                1,
            ))
            .add(BatchNorm2d::new(first_channels)?)
            .add(ReLU6::new());

        // Inverted residual blocks
        let block_configs = Self::get_block_configs();
        let mut in_channels = first_channels;

        for block_config in block_configs {
            let out_channels = config.scale_channels(block_config.out_channels);

            for i in 0..block_config.num_repeat {
                let stride = if i == 0 { block_config.stride } else { 1 };
                let input_channels = if i == 0 { in_channels } else { out_channels };

                features = features.add(InvertedResidual::new(
                    input_channels,
                    out_channels,
                    stride,
                    block_config.expansion_ratio,
                )?);
            }
            in_channels = out_channels;
        }

        // Final convolution
        let final_channels = config.scale_channels(1280);
        features = features
            .add(Conv2d::new(
                in_channels,
                final_channels,
                (1, 1),
                (1, 1),
                (0, 0),
                (1, 1),
                false,
                1,
            ))
            .add(BatchNorm2d::new(final_channels)?)
            .add(ReLU6::new());

        // Classifier
        let classifier = Sequential::new()
            .add(Dropout::new(config.dropout_rate))
            .add(Linear::new(final_channels, config.num_classes, true));

        Ok(Self {
            base: ModuleBase::new(),
            features,
            classifier,
            config,
        })
    }

    /// Get MobileNetV2 block configurations
    fn get_block_configs() -> Vec<MobileNetV2Block> {
        vec![
            MobileNetV2Block {
                in_channels: 32,
                out_channels: 16,
                stride: 1,
                expansion_ratio: 1,
                num_repeat: 1,
            },
            MobileNetV2Block {
                in_channels: 16,
                out_channels: 24,
                stride: 2,
                expansion_ratio: 6,
                num_repeat: 2,
            },
            MobileNetV2Block {
                in_channels: 24,
                out_channels: 32,
                stride: 2,
                expansion_ratio: 6,
                num_repeat: 3,
            },
            MobileNetV2Block {
                in_channels: 32,
                out_channels: 64,
                stride: 2,
                expansion_ratio: 6,
                num_repeat: 4,
            },
            MobileNetV2Block {
                in_channels: 64,
                out_channels: 96,
                stride: 1,
                expansion_ratio: 6,
                num_repeat: 3,
            },
            MobileNetV2Block {
                in_channels: 96,
                out_channels: 160,
                stride: 2,
                expansion_ratio: 6,
                num_repeat: 3,
            },
            MobileNetV2Block {
                in_channels: 160,
                out_channels: 320,
                stride: 1,
                expansion_ratio: 6,
                num_repeat: 1,
            },
        ]
    }

    /// Create MobileNetV2 with standard width multipliers
    pub fn mobilenet_v2_0_35(num_classes: usize) -> Result<Self> {
        let config = MobileNetConfig {
            width_multiplier: 0.35,
            num_classes,
            ..Default::default()
        };
        Self::new(config)
    }

    pub fn mobilenet_v2_0_5(num_classes: usize) -> Result<Self> {
        let config = MobileNetConfig {
            width_multiplier: 0.5,
            num_classes,
            ..Default::default()
        };
        Self::new(config)
    }

    pub fn mobilenet_v2_0_75(num_classes: usize) -> Result<Self> {
        let config = MobileNetConfig {
            width_multiplier: 0.75,
            num_classes,
            ..Default::default()
        };
        Self::new(config)
    }

    pub fn mobilenet_v2_1_0(num_classes: usize) -> Result<Self> {
        let config = MobileNetConfig {
            width_multiplier: 1.0,
            num_classes,
            ..Default::default()
        };
        Self::new(config)
    }

    pub fn mobilenet_v2_1_4(num_classes: usize) -> Result<Self> {
        let config = MobileNetConfig {
            width_multiplier: 1.4,
            num_classes,
            ..Default::default()
        };
        Self::new(config)
    }
}

impl Module for MobileNetV2 {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.features.forward(input)?;

        // Apply global average pooling to reduce spatial dimensions
        // Instead of mean, use adaptive average pooling to ensure consistent output size
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
                        sum += features_data[idx];
                    }
                }
                pooled_data[b * channels + c] = sum / (height * width) as f32;
            }
        }

        let pooled = Tensor::from_vec(pooled_data, &[batch_size, channels])?;

        self.classifier.forward(&pooled)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.features.parameters() {
            params.insert(format!("features.{}", name), param);
        }

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

/// Utilities for MobileNet models
pub mod utils {
    use super::*;

    /// Calculate model efficiency metrics
    pub fn calculate_efficiency_metrics(config: &MobileNetConfig) -> EfficiencyMetrics {
        let base_params = 3_500_000; // Approximate MobileNetV2 1.0x parameters
        let base_flops = 300_000_000; // Approximate MobileNetV2 1.0x FLOPs

        let width_factor = config.width_multiplier.powi(2);
        let resolution_factor = config.resolution_multiplier.powi(2);

        EfficiencyMetrics {
            estimated_params: (base_params as f32 * width_factor) as usize,
            estimated_flops: (base_flops as f32 * width_factor * resolution_factor) as usize,
            efficiency_score: 1.0 / (width_factor * resolution_factor),
        }
    }

    /// Get recommended deployment settings
    pub fn get_deployment_config(config: &MobileNetConfig) -> DeploymentConfig {
        match config.width_multiplier {
            w if w <= 0.35 => DeploymentConfig {
                target_device: "mobile".to_string(),
                quantization_recommended: true,
                batch_size: 1,
                memory_mb: 50,
            },
            w if w <= 0.75 => DeploymentConfig {
                target_device: "edge".to_string(),
                quantization_recommended: true,
                batch_size: 4,
                memory_mb: 100,
            },
            _ => DeploymentConfig {
                target_device: "server".to_string(),
                quantization_recommended: false,
                batch_size: 32,
                memory_mb: 200,
            },
        }
    }

    /// Create custom MobileNet with specific multipliers
    pub fn create_custom_mobilenet_v2(
        width_multiplier: f32,
        resolution_multiplier: f32,
        num_classes: usize,
    ) -> Result<MobileNetV2> {
        let config = MobileNetConfig {
            width_multiplier,
            resolution_multiplier,
            num_classes,
            ..Default::default()
        };
        MobileNetV2::new(config)
    }

    /// Compare different MobileNet configurations
    pub fn compare_configurations(configs: &[MobileNetConfig]) -> Vec<ComparisonResult> {
        configs
            .iter()
            .map(|config| {
                let metrics = calculate_efficiency_metrics(config);
                ComparisonResult {
                    config: config.clone(),
                    metrics,
                    relative_speed: 1.0 / (config.width_multiplier * config.resolution_multiplier),
                    relative_accuracy: config.width_multiplier.powf(0.5), // Rough estimate
                }
            })
            .collect()
    }
}

/// Model efficiency metrics
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    pub estimated_params: usize,
    pub estimated_flops: usize,
    pub efficiency_score: f32,
}

/// Deployment configuration
#[derive(Debug, Clone)]
pub struct DeploymentConfig {
    pub target_device: String,
    pub quantization_recommended: bool,
    pub batch_size: usize,
    pub memory_mb: usize,
}

/// Configuration comparison result
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub config: MobileNetConfig,
    pub metrics: EfficiencyMetrics,
    pub relative_speed: f32,
    pub relative_accuracy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_mobilenet_config() {
        let config = MobileNetConfig::default();
        assert_eq!(config.width_multiplier, 1.0);
        assert_eq!(config.scale_channels(32), 32);
    }

    #[test]
    fn test_depthwise_separable_conv() -> Result<()> {
        let conv = DepthwiseSeparableConv::new_3x3(32, 64, 1)?;
        let input = ones(&[1, 32, 56, 56])?;

        let output = conv.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 64, 56, 56]);

        Ok(())
    }

    #[test]
    fn test_inverted_residual() -> Result<()> {
        let block = InvertedResidual::new_standard(32, 64, 2)?;
        let input = ones(&[1, 32, 56, 56])?;

        let output = block.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 64, 28, 28]);

        Ok(())
    }

    #[test]
    fn test_inverted_residual_shortcut() -> Result<()> {
        let block = InvertedResidual::new_standard(32, 32, 1)?;
        let input = ones(&[1, 32, 56, 56])?;

        let output = block.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 32, 56, 56]);

        Ok(())
    }

    #[test]
    fn test_mobilenet_v1_creation() {
        let model = MobileNetV1::mobilenet_v1_1_0(1000).unwrap();
        assert_eq!(model.config.num_classes, 1000);
        assert_eq!(model.config.width_multiplier, 1.0);
    }

    #[test]
    fn test_mobilenet_v2_creation() {
        let model = MobileNetV2::mobilenet_v2_1_0(1000).unwrap();
        assert_eq!(model.config.num_classes, 1000);
        assert_eq!(model.config.width_multiplier, 1.0);
    }

    #[test]
    fn test_mobilenet_v2_forward() -> Result<()> {
        let model = MobileNetV2::mobilenet_v2_0_35(10)?;
        let input = ones(&[1, 3, 224, 224])?;

        let output = model.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 10]);

        Ok(())
    }

    #[test]
    fn test_width_multiplier_scaling() {
        let config = MobileNetConfig {
            width_multiplier: 0.5,
            ..Default::default()
        };

        assert_eq!(config.scale_channels(32), 16);
        assert_eq!(config.scale_channels(64), 32);
    }

    #[test]
    fn test_efficiency_metrics() {
        let config = MobileNetConfig {
            width_multiplier: 0.5,
            resolution_multiplier: 0.75,
            ..Default::default()
        };

        let metrics = utils::calculate_efficiency_metrics(&config);
        assert!(metrics.estimated_params < 3_500_000); // Should be smaller
        assert!(metrics.efficiency_score > 1.0); // Should be more efficient
    }

    #[test]
    fn test_deployment_config() {
        let config = MobileNetConfig {
            width_multiplier: 0.35,
            ..Default::default()
        };

        let deploy_config = utils::get_deployment_config(&config);
        assert_eq!(deploy_config.target_device, "mobile");
        assert!(deploy_config.quantization_recommended);
    }

    #[test]
    fn test_custom_mobilenet() -> Result<()> {
        let model = utils::create_custom_mobilenet_v2(0.75, 0.875, 100)?;
        assert_eq!(model.config.num_classes, 100);
        assert_eq!(model.config.width_multiplier, 0.75);
        assert_eq!(model.config.resolution_multiplier, 0.875);
        Ok(())
    }
}
