//! MobileNet Architecture Implementations for ToRSh Deep Learning Framework
//!
//! This module provides comprehensive implementations of the MobileNet family of efficient
//! neural network architectures designed for mobile and edge devices.
//!
//! ## Supported Architectures
//!
//! - **MobileNet V2**: Inverted residuals and linear bottlenecks
//! - Support for various width multipliers (0.35x, 0.5x, 0.75x, 1.0x)
//! - Efficient depthwise separable convolutions
//!
//! ## Key Features
//!
//! - **Inverted Residual Blocks**: Core building blocks with expansion-depthwise-projection pattern
//! - **Depthwise Separable Convolutions**: Factorized convolutions for efficiency
//! - **Width Multiplier**: Adjustable model capacity for different deployment constraints
//! - **ReLU6 Activation**: Optimized for quantization and mobile inference
//!
//! ## Example Usage
//!
//! ```rust
//! use torsh_models::vision::mobilenet::*;
//!
//! // Create MobileNet V2 with default parameters
//! let model = MobileNetV2::mobilenet_v2(1000);
//!
//! // Create smaller variant for edge deployment
//! let small_model = MobileNetV2::mobilenet_v2_050(1000); // 0.5x width
//!
//! // Forward pass
//! let input = torsh_tensor::creation::randn(&[1, 3, 224, 224])?;
//! let output = model.forward(&input)?;
//! ```

use torsh_core::error::{Result, TorshError};
use torsh_core::DeviceType;
use std::collections::HashMap;
use torsh_tensor::Tensor;
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};

/// Inverted Residual Block (IRB) for MobileNet V2
///
/// The core building block of MobileNet V2, implementing the inverted residual structure:
/// 1. **Expansion**: 1x1 pointwise convolution to expand channels
/// 2. **Depthwise**: 3x3 depthwise convolution for spatial filtering
/// 3. **Projection**: 1x1 pointwise convolution to project back to output channels
///
/// Key features:
/// - Residual connections when stride=1 and input_channels=output_channels
/// - ReLU6 activation for better quantization
/// - Linear bottleneck (no activation after final projection)
#[derive(Debug)]
pub struct InvertedResidualBlock {
    conv1: Conv2d,      // 1x1 expansion
    bn1: BatchNorm2d,
    conv2: Conv2d,      // 3x3 depthwise
    bn2: BatchNorm2d,
    conv3: Conv2d,      // 1x1 projection
    bn3: BatchNorm2d,
    relu6: ReLU6,
    use_residual: bool,
    stride: usize,
    expand_ratio: usize,
}

impl InvertedResidualBlock {
    /// Creates a new inverted residual block
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `stride` - Stride for the depthwise convolution
    /// * `expand_ratio` - Channel expansion ratio (typically 6)
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        expand_ratio: usize,
    ) -> Self {
        let expanded_channels = in_channels * expand_ratio;
        let use_residual = stride == 1 && in_channels == out_channels;

        // 1x1 Pointwise (expansion)
        let conv1 = Conv2d::new(
            in_channels,
            expanded_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            false,
            1,
        );
        let bn1 = BatchNorm2d::new(expanded_channels);

        // 3x3 Depthwise
        let conv2 = Conv2d::new(
            expanded_channels,
            expanded_channels,
            (3, 3),
            (stride, stride),
            (1, 1),
            (1, 1),
            false,
            expanded_channels, // groups = in_channels for depthwise
        );
        let bn2 = BatchNorm2d::new(expanded_channels);

        // 1x1 Pointwise (linear projection)
        let conv3 = Conv2d::new(
            expanded_channels,
            out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            false,
            1,
        );
        let bn3 = BatchNorm2d::new(out_channels);

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            relu6: ReLU6::new(),
            use_residual,
            stride,
            expand_ratio,
        }
    }

    /// Get expansion ratio
    pub fn expansion_ratio(&self) -> usize {
        self.expand_ratio
    }

    /// Get stride
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Check if block uses residual connection
    pub fn has_residual(&self) -> bool {
        self.use_residual
    }
}

impl Module for InvertedResidualBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();

        // Expansion (1x1 conv) - only if expand_ratio > 1
        let mut out = if self.expand_ratio > 1 {
            let expanded = self.conv1.forward(x)?;
            let expanded = self.bn1.forward(&expanded)?;
            self.relu6.forward(&expanded)?
        } else {
            x.clone()
        };

        // Depthwise (3x3 conv)
        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;
        out = self.relu6.forward(&out)?;

        // Linear projection (1x1 conv without activation)
        out = self.conv3.forward(&out)?;
        out = self.bn3.forward(&out)?;

        // Residual connection
        if self.use_residual {
            out = out.add(&identity)?;
        }

        Ok(out)
    }

    fn train(&mut self) {
        if self.expand_ratio > 1 {
            self.conv1.train();
            self.bn1.train();
        }
        self.conv2.train();
        self.bn2.train();
        self.conv3.train();
        self.bn3.train();
    }

    fn eval(&mut self) {
        if self.expand_ratio > 1 {
            self.conv1.eval();
            self.bn1.eval();
        }
        self.conv2.eval();
        self.bn2.eval();
        self.conv3.eval();
        self.bn3.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if self.expand_ratio > 1 {
            for (name, param) in self.conv1.parameters() {
                params.insert(format!("expansion.{}", name), param);
            }
            for (name, param) in self.bn1.parameters() {
                params.insert(format!("expansion_bn.{}", name), param);
            }
        }

        for (name, param) in self.conv2.parameters() {
            params.insert(format!("depthwise.{}", name), param);
        }
        for (name, param) in self.bn2.parameters() {
            params.insert(format!("depthwise_bn.{}", name), param);
        }
        for (name, param) in self.conv3.parameters() {
            params.insert(format!("projection.{}", name), param);
        }
        for (name, param) in self.bn3.parameters() {
            params.insert(format!("projection_bn.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.conv2.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        if self.expand_ratio > 1 {
            self.conv1.to_device(device)?;
            self.bn1.to_device(device)?;
        }
        self.conv2.to_device(device)?;
        self.bn2.to_device(device)?;
        self.conv3.to_device(device)?;
        self.bn3.to_device(device)?;
        Ok(())
    }
}

/// MobileNet V2 Configuration
#[derive(Debug, Clone)]
pub struct MobileNetV2Config {
    pub num_classes: usize,
    pub width_mult: f32,
    pub dropout_rate: f32,
    pub input_channels: usize,
    /// Configuration: (expand_ratio, channels, num_blocks, stride)
    pub block_configs: Vec<(usize, usize, usize, usize)>,
}

impl MobileNetV2Config {
    /// Create default configuration for ImageNet
    pub fn default_imagenet(num_classes: usize) -> Self {
        Self {
            num_classes,
            width_mult: 1.0,
            dropout_rate: 0.2,
            input_channels: 3,
            block_configs: vec![
                (1, 16, 1, 1),   // Stage 1
                (6, 24, 2, 2),   // Stage 2
                (6, 32, 3, 2),   // Stage 3
                (6, 64, 4, 2),   // Stage 4
                (6, 96, 3, 1),   // Stage 5
                (6, 160, 3, 2),  // Stage 6
                (6, 320, 1, 1),  // Stage 7
            ],
        }
    }

    /// Create configuration with custom width multiplier
    pub fn with_width_mult(num_classes: usize, width_mult: f32) -> Self {
        let mut config = Self::default_imagenet(num_classes);
        config.width_mult = width_mult;
        config
    }
}

/// MobileNet V2 model
///
/// Implements MobileNetV2: "Inverted Residuals and Linear Bottlenecks"
///
/// Key improvements over V1:
/// - Inverted residual structure (expand-depthwise-project)
/// - Linear bottlenecks (no activation after final projection)
/// - More efficient parameter usage
/// - Better accuracy-efficiency trade-off
#[derive(Debug)]
pub struct MobileNetV2 {
    config: MobileNetV2Config,

    // Initial convolution
    conv1: Conv2d,
    bn1: BatchNorm2d,

    // Inverted residual blocks
    layers: Vec<Vec<InvertedResidualBlock>>,

    // Final convolution
    conv2: Conv2d,
    bn2: BatchNorm2d,

    // Classifier
    avgpool: AdaptiveAvgPool2d,
    dropout: Dropout,
    classifier: Linear,
    relu6: ReLU6,
}

impl MobileNetV2 {
    /// Creates a new MobileNet V2 model
    ///
    /// # Arguments
    /// * `num_classes` - Number of output classes
    /// * `width_mult` - Width multiplier for channels (0.35, 0.5, 0.75, 1.0, 1.4)
    /// * `dropout_rate` - Dropout rate before classifier
    pub fn new(num_classes: usize, width_mult: f32, dropout_rate: f32) -> Self {
        let config = MobileNetV2Config::with_width_mult(num_classes, width_mult);
        Self::from_config(config.clone())
    }

    /// Create model from configuration
    pub fn from_config(config: MobileNetV2Config) -> Self {
        let relu6 = ReLU6::new();

        // First conv layer
        let first_channels = Self::make_divisible(32 * config.width_mult);
        let conv1 = Conv2d::new(
            config.input_channels,
            first_channels,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            false,
            1
        );
        let bn1 = BatchNorm2d::new(first_channels);

        // Build inverted residual blocks
        let mut layers = Vec::new();
        let mut input_channels = first_channels;

        for (expand_ratio, out_channels, num_blocks, stride) in &config.block_configs {
            let out_channels = Self::make_divisible(*out_channels as f32 * config.width_mult);
            let mut stage_blocks = Vec::new();

            // First block with given stride
            stage_blocks.push(InvertedResidualBlock::new(
                input_channels,
                out_channels,
                *stride,
                *expand_ratio,
            ));

            // Remaining blocks with stride 1
            for _ in 1..*num_blocks {
                stage_blocks.push(InvertedResidualBlock::new(
                    out_channels,
                    out_channels,
                    1,
                    *expand_ratio,
                ));
            }

            layers.push(stage_blocks);
            input_channels = out_channels;
        }

        // Last conv layer
        let last_channels = Self::make_divisible(1280.0 * config.width_mult).max(input_channels);
        let conv2 = Conv2d::new(
            input_channels,
            last_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            false,
            1,
        );
        let bn2 = BatchNorm2d::new(last_channels);

        // Classifier
        let avgpool = AdaptiveAvgPool2d::new((Some(1), Some(1)));
        let dropout = Dropout::new(config.dropout_rate);
        let classifier = Linear::new(last_channels, config.num_classes, true);

        Self {
            config,
            conv1,
            bn1,
            layers,
            conv2,
            bn2,
            avgpool,
            dropout,
            classifier,
            relu6,
        }
    }

    /// Make number divisible by 8 for efficient computation
    fn make_divisible(v: f32) -> usize {
        let new_v = ((v + 4.0) / 8.0).floor() * 8.0;
        std::cmp::max(new_v as usize, 8)
    }

    /// Create MobileNet V2 with default parameters (width_mult=1.0)
    pub fn mobilenet_v2(num_classes: usize) -> Self {
        Self::new(num_classes, 1.0, 0.2)
    }

    /// Create MobileNet V2 with 1.4x width (highest accuracy)
    pub fn mobilenet_v2_140(num_classes: usize) -> Self {
        Self::new(num_classes, 1.4, 0.2)
    }

    /// Create MobileNet V2 with 0.75x width
    pub fn mobilenet_v2_075(num_classes: usize) -> Self {
        Self::new(num_classes, 0.75, 0.2)
    }

    /// Create MobileNet V2 with 0.5x width
    pub fn mobilenet_v2_050(num_classes: usize) -> Self {
        Self::new(num_classes, 0.5, 0.2)
    }

    /// Create MobileNet V2 with 0.35x width (most efficient)
    pub fn mobilenet_v2_035(num_classes: usize) -> Self {
        Self::new(num_classes, 0.35, 0.2)
    }

    /// Get model configuration
    pub fn config(&self) -> &MobileNetV2Config {
        &self.config
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters().values().map(|p| {
            let data = p.data.read().expect("lock should not be poisoned");
            data.numel()
        }).sum()
    }

    /// Get model complexity metrics
    pub fn complexity_info(&self) -> (usize, f32) {
        let params = self.num_parameters();
        let width = self.config.width_mult;
        (params, width)
    }
}

impl Module for MobileNetV2 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // First conv
        let mut x = self.conv1.forward(x)?;
        x = self.bn1.forward(&x)?;
        x = self.relu6.forward(&x)?;

        // Inverted residual blocks
        for stage in &self.layers {
            for block in stage {
                x = block.forward(&x)?;
            }
        }

        // Last conv
        x = self.conv2.forward(&x)?;
        x = self.bn2.forward(&x)?;
        x = self.relu6.forward(&x)?;

        // Classifier
        x = self.avgpool.forward(&x)?;
        let batch_size = x.size(0)?;
        let features = x.size(1)?;
        x = x.view(&[batch_size, features])?;
        x = self.dropout.forward(&x)?;
        x = self.classifier.forward(&x)?;

        Ok(x)
    }

    fn train(&mut self) {
        self.conv1.train();
        self.bn1.train();
        for stage in &mut self.layers {
            for block in stage {
                block.train();
            }
        }
        self.conv2.train();
        self.bn2.train();
        self.dropout.train();
        self.classifier.train();
    }

    fn eval(&mut self) {
        self.conv1.eval();
        self.bn1.eval();
        for stage in &mut self.layers {
            for block in stage {
                block.eval();
            }
        }
        self.conv2.eval();
        self.bn2.eval();
        self.dropout.eval();
        self.classifier.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Initial convolution
        for (name, param) in self.conv1.parameters() {
            params.insert(format!("features.0.{}", name), param);
        }
        for (name, param) in self.bn1.parameters() {
            params.insert(format!("features.1.{}", name), param);
        }

        // Inverted residual blocks
        let mut layer_idx = 2; // Start after initial conv and bn
        for (stage_idx, stage) in self.layers.iter().enumerate() {
            for (block_idx, block) in stage.iter().enumerate() {
                for (name, param) in block.parameters() {
                    let prefixed_name = format!("features.{}.{}", layer_idx, name);
                    params.insert(prefixed_name, param);
                }
                layer_idx += 1;
            }
        }

        // Final convolution
        for (name, param) in self.conv2.parameters() {
            params.insert(format!("features.{}.{}", layer_idx, name), param);
        }
        for (name, param) in self.bn2.parameters() {
            params.insert(format!("features.{}.{}", layer_idx + 1, name), param);
        }

        // Classifier
        for (name, param) in self.classifier.parameters() {
            params.insert(format!("classifier.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.conv1.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.conv1.to_device(device)?;
        self.bn1.to_device(device)?;

        for stage in &mut self.layers {
            for block in stage {
                block.to_device(device)?;
            }
        }

        self.conv2.to_device(device)?;
        self.bn2.to_device(device)?;
        self.classifier.to_device(device)?;

        Ok(())
    }
}

/// Factory for creating MobileNet variants
pub struct MobileNetFactory;

impl MobileNetFactory {
    /// Create any MobileNet variant by name
    pub fn create(variant: &str, num_classes: usize) -> Result<MobileNetV2> {
        match variant.to_lowercase().as_str() {
            "v2" | "mobilenet-v2" | "mobilenet_v2" => {
                Ok(MobileNetV2::mobilenet_v2(num_classes))
            }
            "v2-1.4" | "mobilenet-v2-1.4" | "mobilenet_v2_140" => {
                Ok(MobileNetV2::mobilenet_v2_140(num_classes))
            }
            "v2-0.75" | "mobilenet-v2-0.75" | "mobilenet_v2_075" => {
                Ok(MobileNetV2::mobilenet_v2_075(num_classes))
            }
            "v2-0.5" | "mobilenet-v2-0.5" | "mobilenet_v2_050" => {
                Ok(MobileNetV2::mobilenet_v2_050(num_classes))
            }
            "v2-0.35" | "mobilenet-v2-0.35" | "mobilenet_v2_035" => {
                Ok(MobileNetV2::mobilenet_v2_035(num_classes))
            }
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown MobileNet variant: {}. Available: v2, v2-1.4, v2-0.75, v2-0.5, v2-0.35",
                variant
            ))),
        }
    }

    /// Get model information
    pub fn model_info(variant: &str) -> Result<String> {
        let info = match variant.to_lowercase().as_str() {
            "v2" | "mobilenet-v2" => {
                "MobileNet V2 (1.0x): Inverted residuals with linear bottlenecks (~3.4M parameters)"
            }
            "v2-1.4" | "mobilenet-v2-1.4" => {
                "MobileNet V2 (1.4x): Highest accuracy variant (~6.9M parameters)"
            }
            "v2-0.75" | "mobilenet-v2-0.75" => {
                "MobileNet V2 (0.75x): Balanced efficiency and accuracy (~2.6M parameters)"
            }
            "v2-0.5" | "mobilenet-v2-0.5" => {
                "MobileNet V2 (0.5x): Efficient variant for mobile devices (~1.95M parameters)"
            }
            "v2-0.35" | "mobilenet-v2-0.35" => {
                "MobileNet V2 (0.35x): Most efficient variant for edge devices (~1.66M parameters)"
            }
            _ => return Err(TorshError::InvalidArgument(format!("Unknown variant: {}", variant))),
        };
        Ok(info.to_string())
    }

    /// List all available variants
    pub fn available_variants() -> Vec<&'static str> {
        vec!["v2", "v2-1.4", "v2-0.75", "v2-0.5", "v2-0.35"]
    }

    /// Get model complexity comparison
    pub fn complexity_comparison() -> String {
        format!(
            "MobileNet V2 Complexity Comparison:\n\
            ┌─────────────┬─────────────┬──────────────┬─────────────┐\n\
            │ Variant     │ Width Mult  │ Parameters   │ Use Case    │\n\
            ├─────────────┼─────────────┼──────────────┼─────────────┤\n\
            │ V2-0.35     │ 0.35x       │ ~1.66M       │ Edge/IoT    │\n\
            │ V2-0.5      │ 0.5x        │ ~1.95M       │ Mobile      │\n\
            │ V2-0.75     │ 0.75x       │ ~2.6M        │ Balanced    │\n\
            │ V2-1.0      │ 1.0x        │ ~3.4M        │ Standard    │\n\
            │ V2-1.4      │ 1.4x        │ ~6.9M        │ High Acc    │\n\
            └─────────────┴─────────────┴──────────────┴─────────────┘"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::Tensor;

    #[test]
    fn test_inverted_residual_block() {
        let mut block = InvertedResidualBlock::new(32, 64, 2, 6);

        // Test basic properties
        assert_eq!(block.expansion_ratio(), 6);
        assert_eq!(block.stride(), 2);
        assert!(!block.has_residual()); // Different channels and stride != 1

        let input = torsh_tensor::creation::randn(&[1, 32, 56, 56]).unwrap();
        let output = block.forward(&input).unwrap();

        // Output should have stride applied
        assert_eq!(output.shape(), &[1, 64, 28, 28]);

        // Test residual connection
        let mut residual_block = InvertedResidualBlock::new(64, 64, 1, 6);
        assert!(residual_block.has_residual());

        // Test train/eval modes
        block.train();
        assert!(block.training());

        block.eval();
        assert!(!block.training());
    }

    #[test]
    fn test_mobilenet_v2_variants() {
        let variants = [
            ("1.0x", MobileNetV2::mobilenet_v2(1000)),
            ("1.4x", MobileNetV2::mobilenet_v2_140(1000)),
            ("0.75x", MobileNetV2::mobilenet_v2_075(1000)),
            ("0.5x", MobileNetV2::mobilenet_v2_050(1000)),
            ("0.35x", MobileNetV2::mobilenet_v2_035(1000)),
        ];

        for (name, model) in variants {
            let input = torsh_tensor::creation::randn(&[1, 3, 224, 224]).unwrap();
            let output = model.forward(&input).unwrap();

            // All should output 1000 classes
            assert_eq!(output.shape(), &[1, 1000], "Failed for {}", name);

            // Check parameter count increases with width multiplier
            let params = model.num_parameters();
            assert!(params > 1_000_000, "Model {} should have >1M parameters", name);
        }
    }

    #[test]
    fn test_mobilenet_factory() {
        // Test factory creation
        let model = MobileNetFactory::create("v2", 1000).unwrap();
        assert_eq!(model.config().num_classes, 1000);
        assert_eq!(model.config().width_mult, 1.0);

        // Test width multiplier variants
        let small_model = MobileNetFactory::create("v2-0.5", 100).unwrap();
        assert_eq!(small_model.config().width_mult, 0.5);

        // Test invalid variant
        assert!(MobileNetFactory::create("invalid", 1000).is_err());

        // Test model info
        let info = MobileNetFactory::model_info("v2").unwrap();
        assert!(info.contains("MobileNet V2"));
        assert!(info.contains("3.4M"));

        // Test available variants
        let variants = MobileNetFactory::available_variants();
        assert!(variants.contains(&"v2"));
        assert!(variants.contains(&"v2-0.5"));
    }

    #[test]
    fn test_mobilenet_config() {
        let config = MobileNetV2Config::default_imagenet(1000);
        assert_eq!(config.num_classes, 1000);
        assert_eq!(config.width_mult, 1.0);
        assert_eq!(config.input_channels, 3);

        let custom_config = MobileNetV2Config::with_width_mult(100, 0.75);
        assert_eq!(custom_config.width_mult, 0.75);
        assert_eq!(custom_config.num_classes, 100);
    }

    #[test]
    fn test_make_divisible() {
        assert_eq!(MobileNetV2::make_divisible(1.0), 8);
        assert_eq!(MobileNetV2::make_divisible(24.0), 24);
        assert_eq!(MobileNetV2::make_divisible(25.0), 24);
        assert_eq!(MobileNetV2::make_divisible(32.0), 32);
    }

    #[test]
    fn test_model_complexity() {
        let models = [
            MobileNetV2::mobilenet_v2_035(10),
            MobileNetV2::mobilenet_v2_050(10),
            MobileNetV2::mobilenet_v2_075(10),
            MobileNetV2::mobilenet_v2(10),
            MobileNetV2::mobilenet_v2_140(10),
        ];

        let mut prev_params = 0;
        for (i, model) in models.iter().enumerate() {
            let (params, width) = model.complexity_info();

            // Parameters should generally increase with width multiplier
            if i > 0 {
                assert!(params >= prev_params, "Parameters should increase with width");
            }

            prev_params = params;

            // Check width multiplier is as expected
            let expected_widths = [0.35, 0.5, 0.75, 1.0, 1.4];
            assert!((width - expected_widths[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_mobilenet_parameters() {
        let model = MobileNetV2::mobilenet_v2_050(10);
        let params = model.parameters();

        // Should have features and classifier parameters
        assert!(params.keys().any(|k| k.starts_with("features")));
        assert!(params.keys().any(|k| k.starts_with("classifier")));

        // Check that we have expansion, depthwise, and projection parameters
        assert!(params.keys().any(|k| k.contains("expansion")));
        assert!(params.keys().any(|k| k.contains("depthwise")));
        assert!(params.keys().any(|k| k.contains("projection")));
    }

    #[test]
    fn test_forward_pass_shapes() {
        let model = MobileNetV2::mobilenet_v2(10);

        // Test different batch sizes
        for batch_size in [1, 4, 8] {
            let input = torsh_tensor::creation::randn(&[batch_size, 3, 224, 224]).unwrap();
            let output = model.forward(&input).unwrap();
            assert_eq!(output.shape(), &[batch_size, 10]);
        }

        // Test different input sizes (should work with global average pooling)
        for size in [192, 224, 256] {
            let input = torsh_tensor::creation::randn(&[1, 3, size, size]).unwrap();
            let output = model.forward(&input).unwrap();
            assert_eq!(output.shape(), &[1, 10]);
        }
    }

    #[test]
    fn test_complexity_comparison() {
        let comparison = MobileNetFactory::complexity_comparison();
        assert!(comparison.contains("Width Mult"));
        assert!(comparison.contains("Parameters"));
        assert!(comparison.contains("V2-0.35"));
        assert!(comparison.contains("V2-1.4"));
    }
}

// Types are already public, no need for re-export