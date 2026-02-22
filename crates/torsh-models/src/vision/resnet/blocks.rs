//! ResNet building blocks

use std::collections::HashMap;
use torsh_core::{error::Result, DeviceType};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// Squeeze-and-Excitation block
pub struct SEBlock {
    avg_pool: AdaptiveAvgPool2d,
    fc1: Linear,
    fc2: Linear,
    sigmoid: Sigmoid,
    reduction_ratio: usize,
}

impl SEBlock {
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
}

impl Module for SEBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.size(0)?;
        let channels = x.size(1)?;

        // Global average pooling
        let se = self.avg_pool.forward(x)?;
        let se = se.view(&[batch_size as i32, channels as i32])?;

        // Squeeze
        let se = self.fc1.forward(&se)?;
        let se = se.relu()?;

        // Excitation
        let se = self.fc2.forward(&se)?;
        let se = self.sigmoid.forward(&se)?;

        // Reshape and apply
        let se = se.view(&[batch_size as i32, channels as i32, 1, 1])?;
        x.mul(&se)
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
        self.fc1.training() && self.fc2.training()
    }

    fn train(&mut self) {
        self.fc1.train();
        self.fc2.train();
        self.avg_pool.train();
        self.sigmoid.train();
    }

    fn eval(&mut self) {
        self.fc1.eval();
        self.fc2.eval();
        self.avg_pool.eval();
        self.sigmoid.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.fc1.to_device(device)?;
        self.fc2.to_device(device)?;
        Ok(())
    }
}

/// Basic residual block for ResNet-18/34
pub struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<Sequential>,
    stride: usize,
    se_block: Option<SEBlock>,
}

impl BasicBlock {
    pub fn new(
        inplanes: usize,
        planes: usize,
        stride: usize,
        downsample: Option<Sequential>,
        use_se: bool,
        se_reduction_ratio: usize,
    ) -> Result<Self> {
        let conv1 = Conv2d::new(
            inplanes,
            planes,
            (3, 3),
            (stride, stride),
            (1, 1),
            (1, 1),
            false,
            1,
        );
        let bn1 = BatchNorm2d::new(planes)?;
        let relu = ReLU::new();
        let conv2 = Conv2d::new(planes, planes, (3, 3), (1, 1), (1, 1), (1, 1), false, 1);
        let bn2 = BatchNorm2d::new(planes)?;

        let se_block = if use_se {
            Some(SEBlock::new(planes, se_reduction_ratio))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            relu,
            conv2,
            bn2,
            downsample,
            stride,
            se_block,
        })
    }

    /// Create block without SE
    pub fn simple(
        inplanes: usize,
        planes: usize,
        stride: usize,
        downsample: Option<Sequential>,
    ) -> Result<Self> {
        Self::new(inplanes, planes, stride, downsample, false, 16)
    }

    /// Get the expansion factor for basic blocks
    pub const fn expansion() -> usize {
        1
    }
}

impl Module for BasicBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();

        let mut out = self.conv1.forward(x)?;
        out = self.bn1.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;

        // Apply SE block if present
        if let Some(ref se) = self.se_block {
            out = se.forward(&out)?;
        }

        // Apply downsample to identity if needed
        let identity = if let Some(ref downsample) = self.downsample {
            downsample.forward(&identity)?
        } else {
            identity
        };

        out = out.add(&identity)?;
        out = self.relu.forward(&out)?;

        Ok(out)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.conv1.parameters() {
            params.insert(format!("conv1.{}", name), param);
        }
        for (name, param) in self.bn1.parameters() {
            params.insert(format!("bn1.{}", name), param);
        }
        for (name, param) in self.conv2.parameters() {
            params.insert(format!("conv2.{}", name), param);
        }
        for (name, param) in self.bn2.parameters() {
            params.insert(format!("bn2.{}", name), param);
        }

        if let Some(ref downsample) = self.downsample {
            for (name, param) in downsample.parameters() {
                params.insert(format!("downsample.{}", name), param);
            }
        }

        if let Some(ref se) = self.se_block {
            for (name, param) in se.parameters() {
                params.insert(format!("se.{}", name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.conv1.training() && self.conv2.training() && self.bn1.training() && self.bn2.training()
    }

    fn train(&mut self) {
        self.conv1.train();
        self.bn1.train();
        self.relu.train();
        self.conv2.train();
        self.bn2.train();
        if let Some(ref mut downsample) = self.downsample {
            downsample.train();
        }
        if let Some(ref mut se) = self.se_block {
            se.train();
        }
    }

    fn eval(&mut self) {
        self.conv1.eval();
        self.bn1.eval();
        self.relu.eval();
        self.conv2.eval();
        self.bn2.eval();
        if let Some(ref mut downsample) = self.downsample {
            downsample.eval();
        }
        if let Some(ref mut se) = self.se_block {
            se.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.conv1.to_device(device)?;
        self.bn1.to_device(device)?;
        self.conv2.to_device(device)?;
        self.bn2.to_device(device)?;
        if let Some(ref mut downsample) = self.downsample {
            downsample.to_device(device)?;
        }
        if let Some(ref mut se) = self.se_block {
            se.to_device(device)?;
        }
        Ok(())
    }
}

/// Bottleneck block for ResNet-50/101/152
pub struct BottleneckBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    relu: ReLU,
    downsample: Option<Sequential>,
    stride: usize,
    se_block: Option<SEBlock>,
    groups: usize,
    base_width: usize,
}

impl BottleneckBlock {
    pub fn new(
        inplanes: usize,
        planes: usize,
        stride: usize,
        downsample: Option<Sequential>,
        groups: usize,
        base_width: usize,
        use_se: bool,
        se_reduction_ratio: usize,
    ) -> Result<Self> {
        let width = (planes as f32 * (base_width as f32 / 64.0) * groups as f32) as usize;

        let conv1 = Conv2d::new(inplanes, width, (1, 1), (1, 1), (0, 0), (1, 1), false, 1);
        let bn1 = BatchNorm2d::new(width)?;

        let conv2 = Conv2d::new(
            width,
            width,
            (3, 3),
            (stride, stride),
            (1, 1),
            (1, 1),
            false,
            groups,
        );
        let bn2 = BatchNorm2d::new(width)?;

        let conv3 = Conv2d::new(width, planes * 4, (1, 1), (1, 1), (0, 0), (1, 1), false, 1);
        let bn3 = BatchNorm2d::new(planes * 4)?;
        let relu = ReLU::new();

        let se_block = if use_se {
            Some(SEBlock::new(planes * 4, se_reduction_ratio))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            relu,
            downsample,
            stride,
            se_block,
            groups,
            base_width,
        })
    }

    /// Create block without SE and default grouping
    pub fn simple(
        inplanes: usize,
        planes: usize,
        stride: usize,
        downsample: Option<Sequential>,
    ) -> Result<Self> {
        Self::new(inplanes, planes, stride, downsample, 1, 64, false, 16)
    }

    /// Get the expansion factor for bottleneck blocks
    pub const fn expansion() -> usize {
        4
    }
}

impl Module for BottleneckBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();

        let mut out = self.conv1.forward(x)?;
        out = self.bn1.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv3.forward(&out)?;
        out = self.bn3.forward(&out)?;

        // Apply SE block if present
        if let Some(ref se) = self.se_block {
            out = se.forward(&out)?;
        }

        // Apply downsample to identity if needed
        let identity = if let Some(ref downsample) = self.downsample {
            downsample.forward(&identity)?
        } else {
            identity
        };

        out = out.add(&identity)?;
        out = self.relu.forward(&out)?;

        Ok(out)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.conv1.parameters() {
            params.insert(format!("conv1.{}", name), param);
        }
        for (name, param) in self.bn1.parameters() {
            params.insert(format!("bn1.{}", name), param);
        }
        for (name, param) in self.conv2.parameters() {
            params.insert(format!("conv2.{}", name), param);
        }
        for (name, param) in self.bn2.parameters() {
            params.insert(format!("bn2.{}", name), param);
        }
        for (name, param) in self.conv3.parameters() {
            params.insert(format!("conv3.{}", name), param);
        }
        for (name, param) in self.bn3.parameters() {
            params.insert(format!("bn3.{}", name), param);
        }

        if let Some(ref downsample) = self.downsample {
            for (name, param) in downsample.parameters() {
                params.insert(format!("downsample.{}", name), param);
            }
        }

        if let Some(ref se) = self.se_block {
            for (name, param) in se.parameters() {
                params.insert(format!("se.{}", name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.conv1.training()
            && self.conv2.training()
            && self.conv3.training()
            && self.bn1.training()
            && self.bn2.training()
            && self.bn3.training()
    }

    fn train(&mut self) {
        self.conv1.train();
        self.bn1.train();
        self.conv2.train();
        self.bn2.train();
        self.conv3.train();
        self.bn3.train();
        self.relu.train();
        if let Some(ref mut downsample) = self.downsample {
            downsample.train();
        }
        if let Some(ref mut se) = self.se_block {
            se.train();
        }
    }

    fn eval(&mut self) {
        self.conv1.eval();
        self.bn1.eval();
        self.conv2.eval();
        self.bn2.eval();
        self.conv3.eval();
        self.bn3.eval();
        self.relu.eval();
        if let Some(ref mut downsample) = self.downsample {
            downsample.eval();
        }
        if let Some(ref mut se) = self.se_block {
            se.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.conv1.to_device(device)?;
        self.bn1.to_device(device)?;
        self.conv2.to_device(device)?;
        self.bn2.to_device(device)?;
        self.conv3.to_device(device)?;
        self.bn3.to_device(device)?;
        if let Some(ref mut downsample) = self.downsample {
            downsample.to_device(device)?;
        }
        if let Some(ref mut se) = self.se_block {
            se.to_device(device)?;
        }
        Ok(())
    }
}
