//! ALIGN Vision Encoder
//!
//! EfficientNet-based vision encoder for ALIGN model with MBConv blocks
//! and advanced mobile-optimized convolution architecture.

use std::collections::HashMap;
use torsh_core::{error::Result, DeviceType};
use torsh_nn::prelude::{Conv2d, Dropout};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

use super::mbconv::{GlobalAveragePooling2d, MBConvBlock};
use crate::multimodal::align::config::ALIGNVisionConfig;

/// ALIGN Vision Encoder (EfficientNet-based architecture)
#[derive(Debug)]
pub struct ALIGNVisionEncoder {
    stem: Conv2d,
    blocks: Vec<MBConvBlock>,
    head: Conv2d,
    pooling: GlobalAveragePooling2d,
    dropout: Dropout,
    config: ALIGNVisionConfig,
}

impl ALIGNVisionEncoder {
    pub fn new(config: ALIGNVisionConfig) -> Self {
        let stem = Conv2d::new(3, config.stem_size, (3, 3), (2, 2), (1, 1), (1, 1), true, 1);

        let mut blocks = Vec::new();
        let mut in_channels = config.stem_size;

        for block_args in &config.blocks_args {
            for i in 0..block_args.num_repeat {
                let stride = if i == 0 { block_args.stride } else { 1 };
                let input_filters = if i == 0 {
                    in_channels
                } else {
                    block_args.output_filters
                };

                blocks.push(
                    MBConvBlock::new(
                        input_filters,
                        block_args.output_filters,
                        block_args.kernel_size,
                        stride,
                        block_args.expand_ratio,
                        block_args.se_ratio,
                        config.drop_connect_rate,
                    )
                    .unwrap(),
                );

                in_channels = block_args.output_filters;
            }
        }

        let head = Conv2d::new(
            in_channels,
            config.head_size,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            true,
            1,
        );
        let pooling = GlobalAveragePooling2d::new();
        let dropout = Dropout::new(config.dropout_rate);

        Self {
            stem,
            blocks,
            head,
            pooling,
            dropout,
            config,
        }
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters().len()
    }
}

impl Module for ALIGNVisionEncoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = self.stem.forward(input)?;
        x = x.relu()?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        x = self.head.forward(&x)?;
        x = x.relu()?;
        x = self.pooling.forward(&x)?;
        x = self.dropout.forward(&x)?;

        Ok(x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.stem.parameters() {
            params.insert(format!("stem.{}", name), param);
        }

        for (i, block) in self.blocks.iter().enumerate() {
            for (name, param) in block.parameters() {
                params.insert(format!("blocks.{}.{}", i, name), param);
            }
        }

        for (name, param) in self.head.parameters() {
            params.insert(format!("head.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropout.training() && self.blocks.iter().all(|b| b.training())
    }

    fn train(&mut self) {
        self.dropout.train();
        for block in &mut self.blocks {
            block.train();
        }
    }

    fn eval(&mut self) {
        self.dropout.eval();
        for block in &mut self.blocks {
            block.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.stem.to_device(device)?;
        for block in &mut self.blocks {
            block.to_device(device)?;
        }
        self.head.to_device(device)?;
        Ok(())
    }
}
