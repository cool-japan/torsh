//! Mobile Inverted Bottleneck Convolution (MBConv) Block
//!
//! Implementation of MBConv blocks with Squeeze-and-Excitation attention
//! for EfficientNet-style vision architectures.

use std::collections::HashMap;
use torsh_core::{error::Result, DeviceType};
use torsh_nn::prelude::{BatchNorm2d, Conv2d, Dropout, Linear};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// MBConv (Mobile Inverted Bottleneck Convolution) Block for EfficientNet-style architecture
#[derive(Debug)]
pub struct MBConvBlock {
    expand_conv: Option<Conv2d>,
    expand_bn: Option<BatchNorm2d>,
    depthwise_conv: Conv2d,
    depthwise_bn: BatchNorm2d,
    se: Option<SqueezeExcitation>,
    project_conv: Conv2d,
    project_bn: BatchNorm2d,
    dropout: Dropout,
    stride: usize,
    use_residual: bool,
}

impl MBConvBlock {
    pub fn new(
        input_filters: usize,
        output_filters: usize,
        kernel_size: usize,
        stride: usize,
        expand_ratio: usize,
        se_ratio: f32,
        drop_connect_rate: f32,
    ) -> Result<Self> {
        let expanded_filters = input_filters * expand_ratio;
        let use_residual = stride == 1 && input_filters == output_filters;

        // Expand convolution (if expand ratio > 1)
        let (expand_conv, expand_bn) = if expand_ratio > 1 {
            (
                Some(Conv2d::new(
                    input_filters,
                    expanded_filters,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    true,
                    1,
                )),
                Some(BatchNorm2d::new(expanded_filters)?),
            )
        } else {
            (None, None)
        };

        // Depthwise convolution
        let padding = kernel_size / 2;
        let depthwise_conv = Conv2d::new(
            expanded_filters,
            expanded_filters,
            (kernel_size, kernel_size),
            (stride, stride),
            (padding, padding),
            (1, 1),
            true,
            expanded_filters,
        );
        let depthwise_bn = BatchNorm2d::new(expanded_filters)?;

        // Squeeze and Excitation
        let se = if se_ratio > 0.0 {
            Some(SqueezeExcitation::new(expanded_filters, se_ratio))
        } else {
            None
        };

        // Project convolution
        let project_conv = Conv2d::new(
            expanded_filters,
            output_filters,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            true,
            1,
        );
        let project_bn = BatchNorm2d::new(output_filters)?;

        let dropout = Dropout::new(drop_connect_rate);

        Ok(Self {
            expand_conv,
            expand_bn,
            depthwise_conv,
            depthwise_bn,
            se,
            project_conv,
            project_bn,
            dropout,
            stride,
            use_residual,
        })
    }
}

impl Module for MBConvBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        // Expand phase
        if let (Some(expand_conv), Some(expand_bn)) = (&self.expand_conv, &self.expand_bn) {
            x = expand_conv.forward(&x)?;
            x = expand_bn.forward(&x)?;
            x = x.relu()?; // ReLU activation (placeholder for Swish)
        }

        // Depthwise convolution
        x = self.depthwise_conv.forward(&x)?;
        x = self.depthwise_bn.forward(&x)?;
        x = x.relu()?; // ReLU activation (placeholder for Swish)

        // Squeeze and Excitation
        if let Some(se) = &self.se {
            x = se.forward(&x)?;
        }

        // Project phase
        x = self.project_conv.forward(&x)?;
        x = self.project_bn.forward(&x)?;

        // Residual connection
        if self.use_residual {
            x = self.dropout.forward(&x)?;
            x = input.add(&x)?;
        }

        Ok(x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if let Some(expand_conv) = &self.expand_conv {
            for (name, param) in expand_conv.parameters() {
                params.insert(format!("expand_conv.{}", name), param);
            }
        }

        if let Some(expand_bn) = &self.expand_bn {
            for (name, param) in expand_bn.parameters() {
                params.insert(format!("expand_bn.{}", name), param);
            }
        }

        for (name, param) in self.depthwise_conv.parameters() {
            params.insert(format!("depthwise_conv.{}", name), param);
        }

        for (name, param) in self.depthwise_bn.parameters() {
            params.insert(format!("depthwise_bn.{}", name), param);
        }

        if let Some(se) = &self.se {
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
        self.dropout.training()
    }

    fn train(&mut self) {
        self.dropout.train();
        if let Some(expand_bn) = &mut self.expand_bn {
            expand_bn.train();
        }
        self.depthwise_bn.train();
        self.project_bn.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
        if let Some(expand_bn) = &mut self.expand_bn {
            expand_bn.eval();
        }
        self.depthwise_bn.eval();
        self.project_bn.eval();
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
        if let Some(se) = &mut self.se {
            se.to_device(device)?;
        }
        self.project_conv.to_device(device)?;
        self.project_bn.to_device(device)?;
        Ok(())
    }
}

/// Squeeze and Excitation module for channel attention
#[derive(Debug)]
pub struct SqueezeExcitation {
    reduce: Linear,
    expand: Linear,
    se_ratio: f32,
}

impl SqueezeExcitation {
    pub fn new(filters: usize, se_ratio: f32) -> Self {
        let reduced_filters = ((filters as f32 * se_ratio) as usize).max(1);
        let reduce = Linear::new(filters, reduced_filters, true);
        let expand = Linear::new(reduced_filters, filters, true);

        Self {
            reduce,
            expand,
            se_ratio,
        }
    }
}

impl Module for SqueezeExcitation {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.size(0)?;
        let channels = input.size(1)?;

        // Global average pooling
        let se = input.mean(Some(&[2, 3]), true)?;
        let se = se.view(&[batch_size as i32, channels as i32])?;

        // Squeeze
        let se = self.reduce.forward(&se)?;
        let se = se.relu()?;

        // Excitation
        let se = self.expand.forward(&se)?;
        let se = se.sigmoid()?;

        // Reshape and apply
        let se = se.view(&[batch_size as i32, channels as i32, 1, 1])?;
        input.mul(&se)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.reduce.parameters() {
            params.insert(format!("reduce.{}", name), param);
        }
        for (name, param) in self.expand.parameters() {
            params.insert(format!("expand.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        true // Linear layers don't have training state
    }

    fn train(&mut self) {
        // Linear layers don't have training state
    }

    fn eval(&mut self) {
        // Linear layers don't have training state
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.reduce.to_device(device)?;
        self.expand.to_device(device)?;
        Ok(())
    }
}

/// Global Average Pooling 2D layer
#[derive(Debug)]
pub struct GlobalAveragePooling2d;

impl GlobalAveragePooling2d {
    pub fn new() -> Self {
        Self
    }
}

impl Module for GlobalAveragePooling2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.mean(Some(&[2, 3]), false)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn training(&self) -> bool {
        true
    }

    fn train(&mut self) {
        // No parameters to train
    }

    fn eval(&mut self) {
        // No parameters to train
    }

    fn to_device(&mut self, _device: DeviceType) -> Result<()> {
        Ok(())
    }
}
