use std::collections::HashMap;
use torsh_core::{device::DeviceType, error::Result};
use torsh_nn::functional::{adaptive_avg_pool2d, relu, sigmoid};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

use crate::{ModelConfig, VisionModel};

/// EfficientNet model configuration
#[derive(Clone)]
pub struct EfficientNetConfig {
    pub width_coefficient: f32,
    pub depth_coefficient: f32,
    pub resolution: usize,
    pub dropout_rate: f32,
}

impl EfficientNetConfig {
    pub fn efficientnet_b0() -> Self {
        Self {
            width_coefficient: 1.0,
            depth_coefficient: 1.0,
            resolution: 224,
            dropout_rate: 0.2,
        }
    }

    pub fn efficientnet_b1() -> Self {
        Self {
            width_coefficient: 1.0,
            depth_coefficient: 1.1,
            resolution: 240,
            dropout_rate: 0.2,
        }
    }

    pub fn efficientnet_b2() -> Self {
        Self {
            width_coefficient: 1.1,
            depth_coefficient: 1.2,
            resolution: 260,
            dropout_rate: 0.3,
        }
    }

    pub fn efficientnet_b3() -> Self {
        Self {
            width_coefficient: 1.2,
            depth_coefficient: 1.4,
            resolution: 300,
            dropout_rate: 0.3,
        }
    }

    pub fn efficientnet_b4() -> Self {
        Self {
            width_coefficient: 1.4,
            depth_coefficient: 1.8,
            resolution: 380,
            dropout_rate: 0.4,
        }
    }

    pub fn efficientnet_b5() -> Self {
        Self {
            width_coefficient: 1.6,
            depth_coefficient: 2.2,
            resolution: 456,
            dropout_rate: 0.4,
        }
    }

    pub fn efficientnet_b6() -> Self {
        Self {
            width_coefficient: 1.8,
            depth_coefficient: 2.6,
            resolution: 528,
            dropout_rate: 0.5,
        }
    }

    pub fn efficientnet_b7() -> Self {
        Self {
            width_coefficient: 2.0,
            depth_coefficient: 3.1,
            resolution: 600,
            dropout_rate: 0.5,
        }
    }
}

/// Mobile Inverted Bottleneck Conv block (MBConv)
pub struct MBConvBlock {
    expand_conv: Option<Conv2d>,
    expand_bn: Option<BatchNorm2d>,
    depthwise_conv: Conv2d,
    depthwise_bn: BatchNorm2d,
    project_conv: Conv2d,
    project_bn: BatchNorm2d,
    se_conv1: Option<Conv2d>,
    se_conv2: Option<Conv2d>,
    drop_connect_rate: f32,
    expand_ratio: usize,
    input_channels: usize,
    output_channels: usize,
    stride: usize,
}

impl MBConvBlock {
    pub fn new(
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
        stride: usize,
        expand_ratio: usize,
        se_ratio: f32,
        drop_connect_rate: f32,
    ) -> Result<Self> {
        let expanded_channels = input_channels * expand_ratio;
        let se_channels = (input_channels as f32 * se_ratio).max(1.0) as usize;

        // Expansion phase (only if expand_ratio != 1)
        let (expand_conv, expand_bn) = if expand_ratio != 1 {
            (
                Some(Conv2d::new(
                    input_channels,
                    expanded_channels,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    false,
                    1,
                )),
                Some(BatchNorm2d::new(expanded_channels)?),
            )
        } else {
            (None, None)
        };

        // Depthwise convolution
        let padding = kernel_size / 2;
        let depthwise_conv = Conv2d::new(
            expanded_channels,
            expanded_channels,
            (kernel_size, kernel_size),
            (stride, stride),
            (padding, padding),
            (1, 1),
            false,
            expanded_channels, // groups = in_channels for depthwise
        );
        let depthwise_bn = BatchNorm2d::new(expanded_channels)?;

        // Squeeze-and-Excitation (if se_ratio > 0)
        let (se_conv1, se_conv2) = if se_ratio > 0.0 {
            (
                Some(Conv2d::new(
                    expanded_channels,
                    se_channels,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    false,
                    1,
                )),
                Some(Conv2d::new(
                    se_channels,
                    expanded_channels,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    false,
                    1,
                )),
            )
        } else {
            (None, None)
        };

        // Projection phase
        let project_conv = Conv2d::new(
            expanded_channels,
            output_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            false,
            1,
        );
        let project_bn = BatchNorm2d::new(output_channels)?;

        Ok(Self {
            expand_conv,
            expand_bn,
            depthwise_conv,
            depthwise_bn,
            project_conv,
            project_bn,
            se_conv1,
            se_conv2,
            drop_connect_rate,
            expand_ratio,
            input_channels,
            output_channels,
            stride,
        })
    }
}

impl Module for MBConvBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        // Expansion phase
        if let (Some(expand_conv), Some(expand_bn)) = (&self.expand_conv, &self.expand_bn) {
            x = expand_conv.forward(&x)?;
            x = expand_bn.forward(&x)?;
            x = relu(&x)?;
        }

        // Depthwise convolution
        x = self.depthwise_conv.forward(&x)?;
        x = self.depthwise_bn.forward(&x)?;
        x = relu(&x)?;

        // Squeeze-and-Excitation
        if let (Some(se_conv1), Some(se_conv2)) = (&self.se_conv1, &self.se_conv2) {
            let se_x = adaptive_avg_pool2d(&x, (Some(1), Some(1)))?;
            let se_x = se_conv1.forward(&se_x)?;
            let se_x = relu(&se_x)?;
            let se_x = se_conv2.forward(&se_x)?;
            let se_x = sigmoid(&se_x)?;
            x = x.mul(&se_x)?;
        }

        // Projection phase
        x = self.project_conv.forward(&x)?;
        x = self.project_bn.forward(&x)?;

        // Skip connection
        if self.stride == 1 && self.input_channels == self.output_channels {
            // NOTE: Drop connect (stochastic depth) omitted in v0.1.0-rc.1
            // This regularization technique randomly drops residual connections during training
            // Model works without it; adds ~1-2% accuracy improvement in original paper
            // Deferred to v0.2.0 - See ROADMAP.md
            x = x.add(input)?;
        }

        Ok(x)
    }

    fn train(&mut self) {
        if let Some(expand_conv) = &mut self.expand_conv {
            expand_conv.train();
        }
        if let Some(expand_bn) = &mut self.expand_bn {
            expand_bn.train();
        }
        self.depthwise_conv.train();
        self.depthwise_bn.train();
        self.project_conv.train();
        self.project_bn.train();
        if let Some(se_conv1) = &mut self.se_conv1 {
            se_conv1.train();
        }
        if let Some(se_conv2) = &mut self.se_conv2 {
            se_conv2.train();
        }
    }

    fn eval(&mut self) {
        if let Some(expand_conv) = &mut self.expand_conv {
            expand_conv.eval();
        }
        if let Some(expand_bn) = &mut self.expand_bn {
            expand_bn.eval();
        }
        self.depthwise_conv.eval();
        self.depthwise_bn.eval();
        self.project_conv.eval();
        self.project_bn.eval();
        if let Some(se_conv1) = &mut self.se_conv1 {
            se_conv1.eval();
        }
        if let Some(se_conv2) = &mut self.se_conv2 {
            se_conv2.eval();
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        if let Some(expand_conv) = &self.expand_conv {
            params.extend(expand_conv.parameters());
        }
        if let Some(expand_bn) = &self.expand_bn {
            params.extend(expand_bn.parameters());
        }
        params.extend(self.depthwise_conv.parameters());
        params.extend(self.depthwise_bn.parameters());
        params.extend(self.project_conv.parameters());
        params.extend(self.project_bn.parameters());
        if let Some(se_conv1) = &self.se_conv1 {
            params.extend(se_conv1.parameters());
        }
        if let Some(se_conv2) = &self.se_conv2 {
            params.extend(se_conv2.parameters());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        // Return training status based on first submodule
        if let Some(expand_conv) = &self.expand_conv {
            expand_conv.training()
        } else {
            self.depthwise_conv.training()
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        if let Some(expand_conv) = &mut self.expand_conv {
            expand_conv.to_device(device)?;
        }
        if let Some(expand_bn) = &mut self.expand_bn {
            expand_bn.to_device(device)?;
        }
        self.depthwise_conv.to_device(device)?;
        self.depthwise_bn.to_device(device)?;
        self.project_conv.to_device(device)?;
        self.project_bn.to_device(device)?;
        if let Some(se_conv1) = &mut self.se_conv1 {
            se_conv1.to_device(device)?;
        }
        if let Some(se_conv2) = &mut self.se_conv2 {
            se_conv2.to_device(device)?;
        }
        Ok(())
    }
}

/// EfficientNet model implementation
pub struct EfficientNet {
    stem_conv: Conv2d,
    stem_bn: BatchNorm2d,
    blocks: Vec<MBConvBlock>,
    head_conv: Conv2d,
    head_bn: BatchNorm2d,
    dropout: Dropout,
    classifier: Linear,
    num_classes: usize,
    head_channels: usize,
}

impl EfficientNet {
    pub fn new(config: ModelConfig, variant_config: EfficientNetConfig) -> Result<Self> {
        let num_classes = config.num_classes;

        // Calculate block settings based on compound scaling
        let base_blocks = [
            // (input_ch, output_ch, kernel_size, stride, expand_ratio, se_ratio, repeats)
            (32, 16, 3, 1, 1, 0.25, 1),
            (16, 24, 3, 2, 6, 0.25, 2),
            (24, 40, 5, 2, 6, 0.25, 2),
            (40, 80, 3, 2, 6, 0.25, 3),
            (80, 112, 5, 1, 6, 0.25, 3),
            (112, 192, 5, 2, 6, 0.25, 4),
            (192, 320, 3, 1, 6, 0.25, 1),
        ];

        // Stem
        let stem_channels = Self::round_filters(32, variant_config.width_coefficient);
        let stem_conv = Conv2d::new(3, stem_channels, (3, 3), (2, 2), (1, 1), (1, 1), false, 1);
        let stem_bn = BatchNorm2d::new(stem_channels)?;

        // Build blocks
        let mut blocks = Vec::new();
        let mut in_channels = stem_channels;

        for (_base_in_ch, base_out_ch, kernel_size, stride, expand_ratio, se_ratio, base_repeats) in
            base_blocks
        {
            let out_channels = Self::round_filters(base_out_ch, variant_config.width_coefficient);
            let repeats = Self::round_repeats(base_repeats, variant_config.depth_coefficient);

            for i in 0..repeats {
                let block_stride = if i == 0 { stride } else { 1 };
                blocks.push(MBConvBlock::new(
                    in_channels,
                    out_channels,
                    kernel_size,
                    block_stride,
                    expand_ratio,
                    se_ratio,
                    variant_config.dropout_rate,
                )?);
                in_channels = out_channels;
            }
        }

        // Head
        let head_channels = Self::round_filters(1280, variant_config.width_coefficient);
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
        let head_bn = BatchNorm2d::new(head_channels)?;

        // Classifier
        let dropout = Dropout::new(variant_config.dropout_rate);
        let classifier = Linear::new(head_channels, num_classes, true);

        Ok(Self {
            stem_conv,
            stem_bn,
            blocks,
            head_conv,
            head_bn,
            dropout,
            classifier,
            num_classes,
            head_channels,
        })
    }

    fn round_filters(filters: usize, width_coefficient: f32) -> usize {
        let filters = (filters as f32 * width_coefficient) as usize;
        let divisor = 8;
        let mut new_filters = ((filters + divisor / 2) / divisor) * divisor;
        if new_filters < (0.9 * filters as f32) as usize {
            new_filters += divisor;
        }
        new_filters.max(divisor)
    }

    fn round_repeats(repeats: usize, depth_coefficient: f32) -> usize {
        ((repeats as f32 * depth_coefficient) as usize).max(1)
    }

    pub fn efficientnet_b0(config: ModelConfig) -> Result<Self> {
        Self::new(config, EfficientNetConfig::efficientnet_b0())
    }

    pub fn efficientnet_b1(config: ModelConfig) -> Result<Self> {
        Self::new(config, EfficientNetConfig::efficientnet_b1())
    }

    pub fn efficientnet_b2(config: ModelConfig) -> Result<Self> {
        Self::new(config, EfficientNetConfig::efficientnet_b2())
    }

    pub fn efficientnet_b3(config: ModelConfig) -> Result<Self> {
        Self::new(config, EfficientNetConfig::efficientnet_b3())
    }

    pub fn efficientnet_b4(config: ModelConfig) -> Result<Self> {
        Self::new(config, EfficientNetConfig::efficientnet_b4())
    }

    pub fn efficientnet_b5(config: ModelConfig) -> Result<Self> {
        Self::new(config, EfficientNetConfig::efficientnet_b5())
    }

    pub fn efficientnet_b6(config: ModelConfig) -> Result<Self> {
        Self::new(config, EfficientNetConfig::efficientnet_b6())
    }

    pub fn efficientnet_b7(config: ModelConfig) -> Result<Self> {
        Self::new(config, EfficientNetConfig::efficientnet_b7())
    }
}

impl Module for EfficientNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Stem
        let mut x = self.stem_conv.forward(input)?;
        x = self.stem_bn.forward(&x)?;
        x = relu(&x)?;

        // MBConv blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Head
        x = self.head_conv.forward(&x)?;
        x = self.head_bn.forward(&x)?;
        x = relu(&x)?;

        // Global average pooling
        x = adaptive_avg_pool2d(&x, (Some(1), Some(1)))?;

        // Flatten
        let batch_size = x.shape().dims()[0];
        x = x.view(&[batch_size as i32, self.head_channels as i32])?;

        // Dropout + classifier
        x = self.dropout.forward(&x)?;
        self.classifier.forward(&x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.stem_conv.parameters());
        params.extend(self.stem_bn.parameters());
        for (i, block) in self.blocks.iter().enumerate() {
            let block_params = block.parameters();
            for (key, param) in block_params {
                params.insert(format!("block{}.{}", i, key), param);
            }
        }
        params.extend(self.head_conv.parameters());
        params.extend(self.head_bn.parameters());
        params.extend(self.classifier.parameters());
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
        for block in &mut self.blocks {
            block.to_device(device)?;
        }
        self.head_conv.to_device(device)?;
        self.head_bn.to_device(device)?;
        self.dropout.to_device(device)?;
        self.classifier.to_device(device)?;
        Ok(())
    }

    fn train(&mut self) {
        self.stem_conv.train();
        self.stem_bn.train();
        for block in &mut self.blocks {
            block.train();
        }
        self.head_conv.train();
        self.head_bn.train();
        self.dropout.train();
        self.classifier.train();
    }

    fn eval(&mut self) {
        self.stem_conv.eval();
        self.stem_bn.eval();
        for block in &mut self.blocks {
            block.eval();
        }
        self.head_conv.eval();
        self.head_bn.eval();
        self.dropout.eval();
        self.classifier.eval();
    }
}

impl VisionModel for EfficientNet {
    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn input_size(&self) -> (usize, usize) {
        (224, 224) // Default, but varies by variant
    }

    fn name(&self) -> &str {
        "EfficientNet"
    }
}
