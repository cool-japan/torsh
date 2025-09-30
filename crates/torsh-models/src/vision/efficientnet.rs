//! EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
//!
//! EfficientNet is a family of convolutional neural networks that achieves
//! state-of-the-art accuracy with significantly fewer parameters and FLOPS
//! than previous models. The key innovation is compound scaling that uniformly
//! scales depth, width, and resolution using a simple compound coefficient.
//!
//! Key Features:
//! - Mobile Inverted Bottleneck Convolution (MBConv) blocks
//! - Squeeze-and-Excitation (SE) attention mechanism
//! - Compound scaling method for depth, width, and resolution
//! - Variants from EfficientNet-B0 to EfficientNet-B7
//! - Swish activation function for improved performance
//! - Depthwise separable convolutions for efficiency
//!
//! References:
//! - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (Tan & Le, 2019)
//! - MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (Howard et al., 2017)
//! - Squeeze-and-Excitation Networks (Hu et al., 2018)

use std::collections::HashMap;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// EfficientNet model variants with scaling parameters
#[derive(Debug, Clone, PartialEq)]
pub enum EfficientNetVariant {
    /// EfficientNet-B0: baseline model
    B0,
    /// EfficientNet-B1: width=1.0, depth=1.1
    B1,
    /// EfficientNet-B2: width=1.1, depth=1.2
    B2,
    /// EfficientNet-B3: width=1.2, depth=1.4
    B3,
    /// EfficientNet-B4: width=1.4, depth=1.8
    B4,
    /// EfficientNet-B5: width=1.6, depth=2.2
    B5,
    /// EfficientNet-B6: width=1.8, depth=2.6
    B6,
    /// EfficientNet-B7: width=2.0, depth=3.1
    B7,
}

impl EfficientNetVariant {
    /// Get scaling parameters (width_mult, depth_mult, dropout_rate)
    pub fn scaling_params(&self) -> (f32, f32, f32) {
        match self {
            EfficientNetVariant::B0 => (1.0, 1.0, 0.2),
            EfficientNetVariant::B1 => (1.0, 1.1, 0.2),
            EfficientNetVariant::B2 => (1.1, 1.2, 0.3),
            EfficientNetVariant::B3 => (1.2, 1.4, 0.3),
            EfficientNetVariant::B4 => (1.4, 1.8, 0.4),
            EfficientNetVariant::B5 => (1.6, 2.2, 0.4),
            EfficientNetVariant::B6 => (1.8, 2.6, 0.5),
            EfficientNetVariant::B7 => (2.0, 3.1, 0.5),
        }
    }

    /// Get expected parameter count
    pub fn parameter_count(&self) -> u64 {
        match self {
            EfficientNetVariant::B0 => 5_288_548,
            EfficientNetVariant::B1 => 7_794_184,
            EfficientNetVariant::B2 => 9_109_994,
            EfficientNetVariant::B3 => 12_233_232,
            EfficientNetVariant::B4 => 19_341_616,
            EfficientNetVariant::B5 => 30_389_784,
            EfficientNetVariant::B6 => 43_040_704,
            EfficientNetVariant::B7 => 66_347_960,
        }
    }

    /// Get typical input resolution
    pub fn input_resolution(&self) -> usize {
        match self {
            EfficientNetVariant::B0 => 224,
            EfficientNetVariant::B1 => 240,
            EfficientNetVariant::B2 => 260,
            EfficientNetVariant::B3 => 300,
            EfficientNetVariant::B4 => 380,
            EfficientNetVariant::B5 => 456,
            EfficientNetVariant::B6 => 528,
            EfficientNetVariant::B7 => 600,
        }
    }
}

/// Configuration for EfficientNet block (MBConv)
#[derive(Debug, Clone)]
pub struct EfficientNetBlockConfig {
    /// Expansion ratio for inverted bottleneck
    pub expand_ratio: usize,
    /// Kernel size for depthwise convolution
    pub kernel_size: usize,
    /// Stride for the block
    pub stride: usize,
    /// Input channels for the block
    pub in_channels: usize,
    /// Output channels for the block
    pub out_channels: usize,
    /// Number of times to repeat this block
    pub num_repeat: usize,
    /// Squeeze-and-Excitation ratio (None to disable)
    pub se_ratio: Option<f32>,
}

/// EfficientNet configuration
#[derive(Debug, Clone)]
pub struct EfficientNetConfig {
    /// EfficientNet variant
    pub variant: EfficientNetVariant,
    /// Number of output classes
    pub num_classes: usize,
    /// Input channels (typically 3 for RGB)
    pub input_channels: usize,
    /// Custom block configurations (None to use default)
    pub block_configs: Option<Vec<EfficientNetBlockConfig>>,
    /// Custom scaling parameters (None to use variant defaults)
    pub scaling_params: Option<(f32, f32, f32)>, // (width, depth, dropout)
}

impl Default for EfficientNetConfig {
    fn default() -> Self {
        Self {
            variant: EfficientNetVariant::B0,
            num_classes: 1000,
            input_channels: 3,
            block_configs: None,
            scaling_params: None,
        }
    }
}

impl EfficientNetConfig {
    /// Create EfficientNet-B0 configuration
    pub fn efficientnet_b0(num_classes: usize) -> Self {
        Self {
            variant: EfficientNetVariant::B0,
            num_classes,
            ..Default::default()
        }
    }

    /// Create EfficientNet-B3 configuration
    pub fn efficientnet_b3(num_classes: usize) -> Self {
        Self {
            variant: EfficientNetVariant::B3,
            num_classes,
            ..Default::default()
        }
    }

    /// Create EfficientNet-B7 configuration
    pub fn efficientnet_b7(num_classes: usize) -> Self {
        Self {
            variant: EfficientNetVariant::B7,
            num_classes,
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.num_classes == 0 {
            return Err(TorshError::InvalidArgument(
                "num_classes must be greater than 0".to_string(),
            ));
        }
        if self.input_channels == 0 {
            return Err(TorshError::InvalidArgument(
                "input_channels must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Squeeze-and-Excitation block for channel attention
#[derive(Debug)]
pub struct SEBlock {
    avg_pool: AdaptiveAvgPool2d,
    fc1: Linear,
    fc2: Linear,
    sigmoid: Sigmoid,
    reduction_ratio: usize,
}

impl SEBlock {
    /// Create a new SE block
    pub fn new(in_channels: usize, reduction_ratio: usize) -> Self {
        let reduced_channels = in_channels / reduction_ratio;

        Self {
            avg_pool: AdaptiveAvgPool2d::new((Some(1), Some(1))),
            fc1: Linear::new(in_channels, reduced_channels, true),
            fc2: Linear::new(reduced_channels, in_channels, true),
            sigmoid: Sigmoid::new(),
            reduction_ratio,
        }
    }

    /// Create SE block with ratio
    pub fn with_ratio(in_channels: usize, se_ratio: f32) -> Self {
        let reduction_ratio = ((1.0 / se_ratio) as usize).max(1);
        Self::new(in_channels, reduction_ratio)
    }
}

impl Module for SEBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.size(0)?;
        let channels = x.size(1)?;

        // Global average pooling
        let se = self.avg_pool.forward(x)?;
        let se = se.view(&[batch_size, channels])?;

        // Squeeze and excitation
        let se = self.fc1.forward(&se)?;
        let se = se.relu()?;
        let se = self.fc2.forward(&se)?;
        let se = self.sigmoid.forward(&se)?;

        // Reshape and apply
        let se = se.view(&[batch_size, channels, 1, 1])?;
        x.mul(&se)
    }

    fn train(&mut self) {
        self.fc1.train();
        self.fc2.train();
    }

    fn eval(&mut self) {
        self.fc1.eval();
        self.fc2.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.fc1.parameters() {
            params.insert(format!("fc1.{}", name), param);
        }
        for (name, param) in self.fc2.parameters() {
            params.insert(format!("fc2.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.fc1.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.fc1.to_device(device)?;
        self.fc2.to_device(device)?;
        Ok(())
    }
}

/// Depthwise Separable Convolution
#[derive(Debug)]
pub struct DepthwiseSeparableConv {
    depthwise: Conv2d,
    pointwise: Conv2d,
}

impl DepthwiseSeparableConv {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let depthwise = Conv2d::new(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            (1, 1),
            false,
            in_channels, // groups = in_channels for depthwise
        );

        let pointwise = Conv2d::new(
            in_channels,
            out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            false,
            1,
        );

        Self {
            depthwise,
            pointwise,
        }
    }
}

impl Module for DepthwiseSeparableConv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.depthwise.forward(x)?;
        self.pointwise.forward(&x)
    }

    fn train(&mut self) {
        self.depthwise.train();
        self.pointwise.train();
    }

    fn eval(&mut self) {
        self.depthwise.eval();
        self.pointwise.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.depthwise.parameters() {
            params.insert(format!("depthwise.{}", name), param);
        }
        for (name, param) in self.pointwise.parameters() {
            params.insert(format!("pointwise.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.depthwise.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.depthwise.to_device(device)?;
        self.pointwise.to_device(device)?;
        Ok(())
    }
}

/// Mobile Inverted Bottleneck Convolution (MBConv) block
///
/// The core building block of EfficientNet, featuring:
/// 1. Expansion phase (optional): 1x1 conv to expand channels
/// 2. Depthwise convolution: efficient spatial filtering
/// 3. Squeeze-and-Excitation: channel attention
/// 4. Projection phase: 1x1 conv to desired output channels
/// 5. Skip connection: residual connection when input/output match
#[derive(Debug)]
pub struct MBConvBlock {
    expand_conv: Option<Conv2d>,
    expand_bn: Option<BatchNorm2d>,
    dwconv: Conv2d,
    dwconv_bn: BatchNorm2d,
    se_block: Option<SEBlock>,
    project_conv: Conv2d,
    project_bn: BatchNorm2d,
    swish: SiLU,
    dropout: Option<Dropout>,
    use_skip_connection: bool,
    expand_ratio: usize,
    stride: usize,
}

impl MBConvBlock {
    /// Create a new MBConv block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        expand_ratio: usize,
        se_ratio: Option<f32>,
        dropout_rate: Option<f32>,
    ) -> Self {
        let expanded_channels = in_channels * expand_ratio;
        let use_skip_connection = stride == 1 && in_channels == out_channels;

        // Expansion phase (only if expand_ratio > 1)
        let (expand_conv, expand_bn) = if expand_ratio > 1 {
            let conv = Conv2d::new(
                in_channels,
                expanded_channels,
                (1, 1),
                (1, 1),
                (0, 0),
                (1, 1),
                false,
                1,
            );
            let bn = BatchNorm2d::new(expanded_channels);
            (Some(conv), Some(bn))
        } else {
            (None, None)
        };

        // Depthwise convolution
        let padding = kernel_size / 2;
        let dwconv = Conv2d::new(
            expanded_channels,
            expanded_channels,
            (kernel_size, kernel_size),
            (stride, stride),
            (padding, padding),
            (1, 1),
            false,
            expanded_channels, // groups = channels for depthwise
        );
        let dwconv_bn = BatchNorm2d::new(expanded_channels);

        // Squeeze-and-Excitation
        let se_block = se_ratio.map(|ratio| SEBlock::with_ratio(expanded_channels, ratio));

        // Projection phase
        let project_conv = Conv2d::new(
            expanded_channels,
            out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            false,
            1,
        );
        let project_bn = BatchNorm2d::new(out_channels);

        let swish = SiLU::new();
        let dropout = dropout_rate.map(Dropout::new);

        Self {
            expand_conv,
            expand_bn,
            dwconv,
            dwconv_bn,
            se_block,
            project_conv,
            project_bn,
            swish,
            dropout,
            use_skip_connection,
            expand_ratio,
            stride,
        }
    }
}

impl Module for MBConvBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();

        // Expansion phase
        if let (Some(expand_conv), Some(expand_bn)) = (&self.expand_conv, &self.expand_bn) {
            out = expand_conv.forward(&out)?;
            out = expand_bn.forward(&out)?;
            out = self.swish.forward(&out)?;
        }

        // Depthwise convolution
        out = self.dwconv.forward(&out)?;
        out = self.dwconv_bn.forward(&out)?;
        out = self.swish.forward(&out)?;

        // Squeeze-and-Excitation
        if let Some(ref se) = self.se_block {
            out = se.forward(&out)?;
        }

        // Projection phase
        out = self.project_conv.forward(&out)?;
        out = self.project_bn.forward(&out)?;

        // Skip connection with dropout
        if self.use_skip_connection {
            if let Some(ref dropout) = self.dropout {
                out = dropout.forward(&out)?;
            }
            out = x.add(&out)?;
        }

        Ok(out)
    }

    fn train(&mut self) {
        if let (Some(ref mut conv), Some(ref mut bn)) = (&mut self.expand_conv, &mut self.expand_bn)
        {
            conv.train();
            bn.train();
        }
        self.dwconv.train();
        self.dwconv_bn.train();
        if let Some(ref mut se) = self.se_block {
            se.train();
        }
        self.project_conv.train();
        self.project_bn.train();
        if let Some(ref mut dropout) = self.dropout {
            dropout.train();
        }
    }

    fn eval(&mut self) {
        if let (Some(ref mut conv), Some(ref mut bn)) = (&mut self.expand_conv, &mut self.expand_bn)
        {
            conv.eval();
            bn.eval();
        }
        self.dwconv.eval();
        self.dwconv_bn.eval();
        if let Some(ref mut se) = self.se_block {
            se.eval();
        }
        self.project_conv.eval();
        self.project_bn.eval();
        if let Some(ref mut dropout) = self.dropout {
            dropout.eval();
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if let (Some(ref conv), Some(ref bn)) = (&self.expand_conv, &self.expand_bn) {
            for (name, param) in conv.parameters() {
                params.insert(format!("expand_conv.{}", name), param);
            }
            for (name, param) in bn.parameters() {
                params.insert(format!("expand_bn.{}", name), param);
            }
        }

        for (name, param) in self.dwconv.parameters() {
            params.insert(format!("dwconv.{}", name), param);
        }
        for (name, param) in self.dwconv_bn.parameters() {
            params.insert(format!("dwconv_bn.{}", name), param);
        }

        if let Some(ref se) = self.se_block {
            for (name, param) in se.parameters() {
                params.insert(format!("se.{}", name), param);
            }
        }

        for (name, param) in self.project_conv.parameters() {
            params.insert(format!("project_conv.{}", name), param);
        }
        for (name, param) in self.project_bn.parameters() {
            params.insert(format!("project_bn.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dwconv.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        if let (Some(ref mut conv), Some(ref mut bn)) = (&mut self.expand_conv, &mut self.expand_bn)
        {
            conv.to_device(device)?;
            bn.to_device(device)?;
        }

        self.dwconv.to_device(device)?;
        self.dwconv_bn.to_device(device)?;

        if let Some(ref mut se) = self.se_block {
            se.to_device(device)?;
        }

        self.project_conv.to_device(device)?;
        self.project_bn.to_device(device)?;

        Ok(())
    }
}

/// EfficientNet model implementation
#[derive(Debug)]
pub struct EfficientNet {
    stem_conv: Conv2d,
    stem_bn: BatchNorm2d,
    blocks: Vec<Vec<MBConvBlock>>,
    head_conv: Conv2d,
    head_bn: BatchNorm2d,
    avgpool: AdaptiveAvgPool2d,
    dropout: Dropout,
    classifier: Linear,
    swish: SiLU,
    config: EfficientNetConfig,
}

impl EfficientNet {
    /// Create EfficientNet from configuration
    pub fn from_config(config: EfficientNetConfig) -> Self {
        config.validate().expect("Invalid EfficientNet configuration");

        let (width_mult, depth_mult, dropout_rate) = config
            .scaling_params
            .unwrap_or_else(|| config.variant.scaling_params());

        let block_configs = config
            .block_configs
            .clone()
            .unwrap_or_else(Self::get_b0_config);

        Self::new(block_configs, config.num_classes, width_mult, depth_mult, dropout_rate, config)
    }

    /// Create EfficientNet with given configuration
    pub fn new(
        block_configs: Vec<EfficientNetBlockConfig>,
        num_classes: usize,
        width_mult: f32,
        depth_mult: f32,
        dropout_rate: f32,
        config: EfficientNetConfig,
    ) -> Self {
        let swish = SiLU::new();

        // Stem
        let stem_channels = Self::round_channels(32, width_mult);
        let stem_conv = Conv2d::new(
            config.input_channels,
            stem_channels,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            false,
            1,
        );
        let stem_bn = BatchNorm2d::new(stem_channels);

        // Build blocks
        let mut blocks = Vec::new();
        let mut in_channels = stem_channels;

        for block_config in block_configs {
            let out_channels = Self::round_channels(block_config.out_channels, width_mult);
            let num_repeat = Self::round_repeats(block_config.num_repeat, depth_mult);

            let mut stage_blocks = Vec::new();

            // First block in stage (may have stride > 1)
            let first_block = MBConvBlock::new(
                in_channels,
                out_channels,
                block_config.kernel_size,
                block_config.stride,
                block_config.expand_ratio,
                block_config.se_ratio,
                Some(dropout_rate),
            );
            stage_blocks.push(first_block);

            // Remaining blocks (stride = 1)
            for _ in 1..num_repeat {
                let block = MBConvBlock::new(
                    out_channels,
                    out_channels,
                    block_config.kernel_size,
                    1, // stride = 1 for repeated blocks
                    block_config.expand_ratio,
                    block_config.se_ratio,
                    Some(dropout_rate),
                );
                stage_blocks.push(block);
            }

            blocks.push(stage_blocks);
            in_channels = out_channels;
        }

        // Head
        let head_channels = Self::round_channels(1280, width_mult);
        let head_conv = Conv2d::new(
            in_channels,
            head_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            false,
            1,
        );
        let head_bn = BatchNorm2d::new(head_channels);

        // Classifier
        let avgpool = AdaptiveAvgPool2d::new((Some(1), Some(1)));
        let dropout = Dropout::new(dropout_rate);
        let classifier = Linear::new(head_channels, num_classes, true);

        Self {
            stem_conv,
            stem_bn,
            blocks,
            head_conv,
            head_bn,
            avgpool,
            dropout,
            classifier,
            swish,
            config,
        }
    }

    /// Round channels to nearest multiple of 8
    fn round_channels(channels: usize, width_mult: f32) -> usize {
        let channels = (channels as f32 * width_mult) as usize;
        let new_channels = std::cmp::max(8, (channels + 4) / 8 * 8);
        if (new_channels as f32) < 0.9 * (channels as f32) {
            new_channels + 8
        } else {
            new_channels
        }
    }

    /// Round number of repeats
    fn round_repeats(repeats: usize, depth_mult: f32) -> usize {
        std::cmp::max(1, (repeats as f32 * depth_mult).ceil() as usize)
    }

    /// Get default block configurations for EfficientNet-B0
    pub fn get_b0_config() -> Vec<EfficientNetBlockConfig> {
        vec![
            EfficientNetBlockConfig {
                expand_ratio: 1,
                kernel_size: 3,
                stride: 1,
                in_channels: 32,
                out_channels: 16,
                num_repeat: 1,
                se_ratio: Some(0.25),
            },
            EfficientNetBlockConfig {
                expand_ratio: 6,
                kernel_size: 3,
                stride: 2,
                in_channels: 16,
                out_channels: 24,
                num_repeat: 2,
                se_ratio: Some(0.25),
            },
            EfficientNetBlockConfig {
                expand_ratio: 6,
                kernel_size: 5,
                stride: 2,
                in_channels: 24,
                out_channels: 40,
                num_repeat: 2,
                se_ratio: Some(0.25),
            },
            EfficientNetBlockConfig {
                expand_ratio: 6,
                kernel_size: 3,
                stride: 2,
                in_channels: 40,
                out_channels: 80,
                num_repeat: 3,
                se_ratio: Some(0.25),
            },
            EfficientNetBlockConfig {
                expand_ratio: 6,
                kernel_size: 5,
                stride: 1,
                in_channels: 80,
                out_channels: 112,
                num_repeat: 3,
                se_ratio: Some(0.25),
            },
            EfficientNetBlockConfig {
                expand_ratio: 6,
                kernel_size: 5,
                stride: 2,
                in_channels: 112,
                out_channels: 192,
                num_repeat: 4,
                se_ratio: Some(0.25),
            },
            EfficientNetBlockConfig {
                expand_ratio: 6,
                kernel_size: 3,
                stride: 1,
                in_channels: 192,
                out_channels: 320,
                num_repeat: 1,
                se_ratio: Some(0.25),
            },
        ]
    }

    /// Create EfficientNet-B0
    pub fn efficientnet_b0(num_classes: usize) -> Self {
        Self::from_config(EfficientNetConfig::efficientnet_b0(num_classes))
    }

    /// Create EfficientNet-B1
    pub fn efficientnet_b1(num_classes: usize) -> Self {
        let config = EfficientNetConfig {
            variant: EfficientNetVariant::B1,
            num_classes,
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Create EfficientNet-B2
    pub fn efficientnet_b2(num_classes: usize) -> Self {
        let config = EfficientNetConfig {
            variant: EfficientNetVariant::B2,
            num_classes,
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Create EfficientNet-B3
    pub fn efficientnet_b3(num_classes: usize) -> Self {
        Self::from_config(EfficientNetConfig::efficientnet_b3(num_classes))
    }

    /// Create EfficientNet-B4
    pub fn efficientnet_b4(num_classes: usize) -> Self {
        let config = EfficientNetConfig {
            variant: EfficientNetVariant::B4,
            num_classes,
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Create EfficientNet-B5
    pub fn efficientnet_b5(num_classes: usize) -> Self {
        let config = EfficientNetConfig {
            variant: EfficientNetVariant::B5,
            num_classes,
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Create EfficientNet-B6
    pub fn efficientnet_b6(num_classes: usize) -> Self {
        let config = EfficientNetConfig {
            variant: EfficientNetVariant::B6,
            num_classes,
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Create EfficientNet-B7
    pub fn efficientnet_b7(num_classes: usize) -> Self {
        let config = EfficientNetConfig {
            variant: EfficientNetVariant::B7,
            num_classes,
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Get model configuration
    pub fn config(&self) -> &EfficientNetConfig {
        &self.config
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters().len()
    }

    /// Get model information
    pub fn model_info(&self) -> String {
        format!(
            "EfficientNet-{:?}: {} parameters, {} classes, {}x{} input",
            self.config.variant,
            self.num_parameters(),
            self.config.num_classes,
            self.config.variant.input_resolution(),
            self.config.variant.input_resolution()
        )
    }
}

impl Module for EfficientNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Stem
        let mut x = self.stem_conv.forward(x)?;
        x = self.stem_bn.forward(&x)?;
        x = self.swish.forward(&x)?;

        // Blocks
        for stage in &self.blocks {
            for block in stage {
                x = block.forward(&x)?;
            }
        }

        // Head
        x = self.head_conv.forward(&x)?;
        x = self.head_bn.forward(&x)?;
        x = self.swish.forward(&x)?;

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
        self.stem_conv.train();
        self.stem_bn.train();
        for stage in &mut self.blocks {
            for block in stage {
                block.train();
            }
        }
        self.head_conv.train();
        self.head_bn.train();
        self.dropout.train();
        self.classifier.train();
    }

    fn eval(&mut self) {
        self.stem_conv.eval();
        self.stem_bn.eval();
        for stage in &mut self.blocks {
            for block in stage {
                block.eval();
            }
        }
        self.head_conv.eval();
        self.head_bn.eval();
        self.dropout.eval();
        self.classifier.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.stem_conv.parameters() {
            params.insert(format!("stem_conv.{}", name), param);
        }
        for (name, param) in self.stem_bn.parameters() {
            params.insert(format!("stem_bn.{}", name), param);
        }

        for (stage_idx, stage) in self.blocks.iter().enumerate() {
            for (block_idx, block) in stage.iter().enumerate() {
                for (name, param) in block.parameters() {
                    let prefixed_name = format!("stage{}_block{}.{}", stage_idx, block_idx, name);
                    params.insert(prefixed_name, param);
                }
            }
        }

        for (name, param) in self.head_conv.parameters() {
            params.insert(format!("head_conv.{}", name), param);
        }
        for (name, param) in self.head_bn.parameters() {
            params.insert(format!("head_bn.{}", name), param);
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
        self.stem_conv.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.stem_conv.to_device(device)?;
        self.stem_bn.to_device(device)?;

        for stage in &mut self.blocks {
            for block in stage {
                block.to_device(device)?;
            }
        }

        self.head_conv.to_device(device)?;
        self.head_bn.to_device(device)?;
        self.classifier.to_device(device)?;

        Ok(())
    }
}

/// EfficientNet Factory for creating model variants
pub struct EfficientNetFactory;

impl EfficientNetFactory {
    /// Create EfficientNet model by name
    pub fn create_by_name(model_name: &str, num_classes: Option<usize>) -> Result<EfficientNet> {
        let num_classes = num_classes.unwrap_or(1000);

        match model_name.to_lowercase().as_str() {
            "efficientnet-b0" | "efficientnet_b0" | "b0" => Ok(EfficientNet::efficientnet_b0(num_classes)),
            "efficientnet-b1" | "efficientnet_b1" | "b1" => Ok(EfficientNet::efficientnet_b1(num_classes)),
            "efficientnet-b2" | "efficientnet_b2" | "b2" => Ok(EfficientNet::efficientnet_b2(num_classes)),
            "efficientnet-b3" | "efficientnet_b3" | "b3" => Ok(EfficientNet::efficientnet_b3(num_classes)),
            "efficientnet-b4" | "efficientnet_b4" | "b4" => Ok(EfficientNet::efficientnet_b4(num_classes)),
            "efficientnet-b5" | "efficientnet_b5" | "b5" => Ok(EfficientNet::efficientnet_b5(num_classes)),
            "efficientnet-b6" | "efficientnet_b6" | "b6" => Ok(EfficientNet::efficientnet_b6(num_classes)),
            "efficientnet-b7" | "efficientnet_b7" | "b7" => Ok(EfficientNet::efficientnet_b7(num_classes)),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown EfficientNet variant: {}. Available: b0, b1, b2, b3, b4, b5, b6, b7",
                model_name
            ))),
        }
    }

    /// Get model information
    pub fn model_info(model_name: &str) -> Result<String> {
        let variant = match model_name.to_lowercase().as_str() {
            "efficientnet-b0" | "efficientnet_b0" | "b0" => EfficientNetVariant::B0,
            "efficientnet-b1" | "efficientnet_b1" | "b1" => EfficientNetVariant::B1,
            "efficientnet-b2" | "efficientnet_b2" | "b2" => EfficientNetVariant::B2,
            "efficientnet-b3" | "efficientnet_b3" | "b3" => EfficientNetVariant::B3,
            "efficientnet-b4" | "efficientnet_b4" | "b4" => EfficientNetVariant::B4,
            "efficientnet-b5" | "efficientnet_b5" | "b5" => EfficientNetVariant::B5,
            "efficientnet-b6" | "efficientnet_b6" | "b6" => EfficientNetVariant::B6,
            "efficientnet-b7" | "efficientnet_b7" | "b7" => EfficientNetVariant::B7,
            _ => return Err(TorshError::InvalidArgument(format!(
                "Unknown EfficientNet variant: {}",
                model_name
            ))),
        };

        let (width, depth, dropout) = variant.scaling_params();
        Ok(format!(
            "EfficientNet-{:?}: {} parameters, {}x{} input, width={:.1}, depth={:.1}, dropout={:.1}",
            variant,
            variant.parameter_count(),
            variant.input_resolution(),
            variant.input_resolution(),
            width,
            depth,
            dropout
        ))
    }

    /// Compare EfficientNet variants
    pub fn compare_variants() -> String {
        let variants = [
            EfficientNetVariant::B0, EfficientNetVariant::B1, EfficientNetVariant::B2, EfficientNetVariant::B3,
            EfficientNetVariant::B4, EfficientNetVariant::B5, EfficientNetVariant::B6, EfficientNetVariant::B7,
        ];

        let mut comparison = String::from("EfficientNet Variant Comparison:\n");
        for variant in &variants {
            let (width, depth, dropout) = variant.scaling_params();
            comparison.push_str(&format!(
                "EfficientNet-{:?}: {} params, {}x{} input, W={:.1}/D={:.1}/Drop={:.1}\n",
                variant,
                variant.parameter_count(),
                variant.input_resolution(),
                variant.input_resolution(),
                width,
                depth,
                dropout
            ));
        }
        comparison
    }

    /// Get all available variants
    pub fn available_variants() -> Vec<&'static str> {
        vec!["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
    }
}

// Comprehensive test suite for EfficientNet models
#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_efficientnet_config_creation() {
        let config = EfficientNetConfig::efficientnet_b0(10);
        assert_eq!(config.variant, EfficientNetVariant::B0);
        assert_eq!(config.num_classes, 10);

        let config = EfficientNetConfig::efficientnet_b3(1000);
        assert_eq!(config.variant, EfficientNetVariant::B3);
        assert_eq!(config.num_classes, 1000);
    }

    #[test]
    fn test_efficientnet_config_validation() {
        let mut config = EfficientNetConfig::default();
        assert!(config.validate().is_ok());

        config.num_classes = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_efficientnet_variant_properties() {
        let variant = EfficientNetVariant::B0;
        let (width, depth, dropout) = variant.scaling_params();
        assert_eq!(width, 1.0);
        assert_eq!(depth, 1.0);
        assert_eq!(dropout, 0.2);
        assert_eq!(variant.input_resolution(), 224);

        let variant = EfficientNetVariant::B7;
        let (width, depth, dropout) = variant.scaling_params();
        assert_eq!(width, 2.0);
        assert_eq!(depth, 3.1);
        assert_eq!(dropout, 0.5);
        assert_eq!(variant.input_resolution(), 600);
    }

    #[test]
    fn test_se_block_creation() {
        let se = SEBlock::new(64, 16);
        assert_eq!(se.reduction_ratio, 16);

        let se = SEBlock::with_ratio(64, 0.25);
        assert_eq!(se.reduction_ratio, 4);
    }

    #[test]
    fn test_mbconv_block_creation() {
        let block = MBConvBlock::new(32, 64, 3, 1, 6, Some(0.25), Some(0.2));
        assert_eq!(block.expand_ratio, 6);
        assert_eq!(block.stride, 1);
        assert!(block.use_skip_connection == false); // in_channels != out_channels
    }

    #[test]
    fn test_efficientnet_model_creation() {
        let model = EfficientNet::efficientnet_b0(10);
        assert_eq!(model.config().variant, EfficientNetVariant::B0);
        assert_eq!(model.config().num_classes, 10);
    }

    #[test]
    fn test_efficientnet_factory() {
        let model = EfficientNetFactory::create_by_name("b0", Some(10));
        assert!(model.is_ok());

        let model = EfficientNetFactory::create_by_name("efficientnet-b3", None);
        assert!(model.is_ok());

        let invalid_model = EfficientNetFactory::create_by_name("b999", None);
        assert!(invalid_model.is_err());
    }

    #[test]
    fn test_model_info() {
        let info = EfficientNetFactory::model_info("b0");
        assert!(info.is_ok());
        assert!(info.unwrap().contains("EfficientNet-B0"));

        let info = EfficientNetFactory::model_info("b7");
        assert!(info.is_ok());
        assert!(info.unwrap().contains("EfficientNet-B7"));
    }

    #[test]
    fn test_variant_comparison() {
        let comparison = EfficientNetFactory::compare_variants();
        assert!(comparison.contains("EfficientNet-B0"));
        assert!(comparison.contains("EfficientNet-B7"));
        assert!(comparison.contains("params"));
    }

    #[test]
    fn test_available_variants() {
        let variants = EfficientNetFactory::available_variants();
        assert_eq!(variants.len(), 8);
        assert!(variants.contains(&"b0"));
        assert!(variants.contains(&"b7"));
    }

    #[test]
    fn test_parameter_counts() {
        assert!(EfficientNetVariant::B0.parameter_count() < EfficientNetVariant::B3.parameter_count());
        assert!(EfficientNetVariant::B3.parameter_count() < EfficientNetVariant::B7.parameter_count());
    }

    #[test]
    fn test_scaling_functions() {
        // Test channel rounding
        assert_eq!(EfficientNet::round_channels(32, 1.0), 32);
        assert_eq!(EfficientNet::round_channels(32, 1.1), 40); // Rounded up to multiple of 8

        // Test repeat rounding
        assert_eq!(EfficientNet::round_repeats(2, 1.0), 2);
        assert_eq!(EfficientNet::round_repeats(2, 1.1), 3); // Ceiling of 2.2
    }

    #[test]
    fn test_block_configs() {
        let configs = EfficientNet::get_b0_config();
        assert_eq!(configs.len(), 7); // EfficientNet-B0 has 7 stages
        assert_eq!(configs[0].expand_ratio, 1); // First stage doesn't expand
        assert_eq!(configs[1].expand_ratio, 6); // Others use 6x expansion
    }
}