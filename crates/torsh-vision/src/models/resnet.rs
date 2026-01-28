use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

use crate::{ModelConfig, VisionModel};

/// Basic ResNet block
pub struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<Sequential>,
    relu: ReLU,
    stride: usize,
    is_training: bool,
}

impl BasicBlock {
    pub fn new(
        in_planes: usize,
        planes: usize,
        stride: usize,
        downsample: Option<Sequential>,
    ) -> Result<Self> {
        Ok(Self {
            conv1: Conv2d::new(
                in_planes,
                planes,
                (3, 3),
                (stride, stride),
                (1, 1),
                (1, 1),
                false,
                1,
            ),
            bn1: BatchNorm2d::new(planes)?,
            conv2: Conv2d::new(planes, planes, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            bn2: BatchNorm2d::new(planes)?,
            downsample,
            relu: ReLU::new(),
            stride,
            is_training: true,
        })
    }

    pub fn expansion() -> usize {
        1
    }
}

impl Module for BasicBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut out = self.conv1.forward(input)?;
        out = self.bn1.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;

        let residual = if let Some(ref downsample) = self.downsample {
            downsample.forward(input)?
        } else {
            input.clone()
        };

        out = out.add(&residual)?;
        self.relu.forward(&out)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some(ref downsample) = self.downsample {
            params.extend(downsample.parameters());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn train(&mut self) {
        self.is_training = true;
        self.conv1.train();
        self.bn1.train();
        self.conv2.train();
        self.bn2.train();
        if let Some(ref mut downsample) = self.downsample {
            downsample.train();
        }
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.conv1.eval();
        self.bn1.eval();
        self.conv2.eval();
        self.bn2.eval();
        if let Some(ref mut downsample) = self.downsample {
            downsample.eval();
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
        Ok(())
    }

    fn set_training(&mut self, training: bool) {
        if training {
            self.train();
        } else {
            self.eval();
        }
    }
}

/// Bottleneck ResNet block
pub struct Bottleneck {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    downsample: Option<Sequential>,
    relu: ReLU,
    stride: usize,
    is_training: bool,
}

impl Bottleneck {
    pub fn new(
        in_planes: usize,
        planes: usize,
        stride: usize,
        downsample: Option<Sequential>,
    ) -> Result<Self> {
        let expansion = Self::expansion();
        Ok(Self {
            conv1: Conv2d::new(in_planes, planes, (1, 1), (1, 1), (0, 0), (1, 1), false, 1),
            bn1: BatchNorm2d::new(planes)?,
            conv2: Conv2d::new(
                planes,
                planes,
                (3, 3),
                (stride, stride),
                (1, 1),
                (1, 1),
                false,
                1,
            ),
            bn2: BatchNorm2d::new(planes)?,
            conv3: Conv2d::new(
                planes,
                planes * expansion,
                (1, 1),
                (1, 1),
                (0, 0),
                (1, 1),
                false,
                1,
            ),
            bn3: BatchNorm2d::new(planes * expansion)?,
            downsample,
            relu: ReLU::new(),
            stride,
            is_training: true,
        })
    }

    pub fn expansion() -> usize {
        4
    }
}

impl Module for Bottleneck {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut out = self.conv1.forward(input)?;
        out = self.bn1.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv3.forward(&out)?;
        out = self.bn3.forward(&out)?;

        let residual = if let Some(ref downsample) = self.downsample {
            downsample.forward(input)?
        } else {
            input.clone()
        };

        out = out.add(&residual)?;
        self.relu.forward(&out)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.conv3.parameters());
        params.extend(self.bn3.parameters());
        if let Some(ref downsample) = self.downsample {
            params.extend(downsample.parameters());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn train(&mut self) {
        self.is_training = true;
        self.conv1.train();
        self.bn1.train();
        self.conv2.train();
        self.bn2.train();
        self.conv3.train();
        self.bn3.train();
        if let Some(ref mut downsample) = self.downsample {
            downsample.train();
        }
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.conv1.eval();
        self.bn1.eval();
        self.conv2.eval();
        self.bn2.eval();
        self.conv3.eval();
        self.bn3.eval();
        if let Some(ref mut downsample) = self.downsample {
            downsample.eval();
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
        Ok(())
    }

    fn set_training(&mut self, training: bool) {
        if training {
            self.train();
        } else {
            self.eval();
        }
    }
}

/// ResNet model
pub struct ResNet {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    maxpool: MaxPool2d,
    layer1: Sequential,
    layer2: Sequential,
    layer3: Sequential,
    layer4: Sequential,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear,
    num_classes: usize,
    is_training: bool,
}

impl ResNet {
    fn make_layer<B: Module + 'static>(
        planes: usize,
        blocks: usize,
        stride: usize,
        in_planes: usize,
        expansion: usize,
        block_fn: impl Fn(usize, usize, usize, Option<Sequential>) -> B,
    ) -> Result<(Sequential, usize)> {
        let mut layers = Sequential::new();
        let mut downsample = None;

        if stride != 1 || in_planes != planes * expansion {
            downsample = Some(
                Sequential::new()
                    .add(Conv2d::new(
                        in_planes,
                        planes * expansion,
                        (1, 1),
                        (stride, stride),
                        (0, 0),
                        (1, 1),
                        false,
                        1,
                    ))
                    .add(BatchNorm2d::new(planes * expansion)?),
            );
        }

        layers = layers.add(block_fn(in_planes, planes, stride, downsample));
        let current_in_planes = planes * expansion;

        for _ in 1..blocks {
            layers = layers.add(block_fn(current_in_planes, planes, 1, None));
        }

        Ok((layers, current_in_planes))
    }

    pub fn resnet18(config: ModelConfig) -> Result<Self> {
        Self::new::<BasicBlock>(
            &[2, 2, 2, 2],
            config,
            |in_p, p, s, d| BasicBlock::new(in_p, p, s, d).expect("block creation should succeed"),
            BasicBlock::expansion(),
        )
    }

    pub fn resnet34(config: ModelConfig) -> Result<Self> {
        Self::new::<BasicBlock>(
            &[3, 4, 6, 3],
            config,
            |in_p, p, s, d| BasicBlock::new(in_p, p, s, d).expect("block creation should succeed"),
            BasicBlock::expansion(),
        )
    }

    pub fn resnet50(config: ModelConfig) -> Result<Self> {
        Self::new::<Bottleneck>(
            &[3, 4, 6, 3],
            config,
            |in_p, p, s, d| Bottleneck::new(in_p, p, s, d).expect("block creation should succeed"),
            Bottleneck::expansion(),
        )
    }

    pub fn resnet101(config: ModelConfig) -> Result<Self> {
        Self::new::<Bottleneck>(
            &[3, 4, 23, 3],
            config,
            |in_p, p, s, d| Bottleneck::new(in_p, p, s, d).expect("block creation should succeed"),
            Bottleneck::expansion(),
        )
    }

    pub fn resnet152(config: ModelConfig) -> Result<Self> {
        Self::new::<Bottleneck>(
            &[3, 8, 36, 3],
            config,
            |in_p, p, s, d| Bottleneck::new(in_p, p, s, d).expect("block creation should succeed"),
            Bottleneck::expansion(),
        )
    }

    fn new<B: Module + 'static>(
        layers: &[usize],
        config: ModelConfig,
        block_fn: impl Fn(usize, usize, usize, Option<Sequential>) -> B + Clone,
        expansion: usize,
    ) -> Result<Self> {
        let mut in_planes = 64;

        // Create layers
        let (layer1, new_in_planes) =
            Self::make_layer(64, layers[0], 1, in_planes, expansion, block_fn.clone())?;
        in_planes = new_in_planes;

        let (layer2, new_in_planes) =
            Self::make_layer(128, layers[1], 2, in_planes, expansion, block_fn.clone())?;
        in_planes = new_in_planes;

        let (layer3, new_in_planes) =
            Self::make_layer(256, layers[2], 2, in_planes, expansion, block_fn.clone())?;
        in_planes = new_in_planes;

        let (layer4, _) = Self::make_layer(512, layers[3], 2, in_planes, expansion, block_fn)?;

        Ok(Self {
            conv1: Conv2d::new(3, 64, (7, 7), (2, 2), (3, 3), (1, 1), false, 1),
            bn1: BatchNorm2d::new(64)?,
            relu: ReLU::new(),
            maxpool: MaxPool2d::new((3, 3), Some((2, 2)), (1, 1), (1, 1), false),
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool: AdaptiveAvgPool2d::new((Some(1), Some(1))),
            fc: Linear::new(512 * expansion, config.num_classes, true),
            num_classes: config.num_classes,
            is_training: true,
        })
    }
}

impl Module for ResNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = self.conv1.forward(input)?;
        x = self.bn1.forward(&x)?;
        x = self.relu.forward(&x)?;
        x = self.maxpool.forward(&x)?;

        x = self.layer1.forward(&x)?;
        x = self.layer2.forward(&x)?;
        x = self.layer3.forward(&x)?;
        x = self.layer4.forward(&x)?;

        x = self.avgpool.forward(&x)?;
        let batch_size = x.size(0)?;
        let features = x.size(1)?;
        x = x.view(&[batch_size as i32, features as i32])?;
        self.fc.forward(&x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.layer1.parameters());
        params.extend(self.layer2.parameters());
        params.extend(self.layer3.parameters());
        params.extend(self.layer4.parameters());
        params.extend(self.fc.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn train(&mut self) {
        self.is_training = true;
        self.conv1.train();
        self.bn1.train();
        self.layer1.train();
        self.layer2.train();
        self.layer3.train();
        self.layer4.train();
        self.fc.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.conv1.eval();
        self.bn1.eval();
        self.layer1.eval();
        self.layer2.eval();
        self.layer3.eval();
        self.layer4.eval();
        self.fc.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.conv1.to_device(device)?;
        self.bn1.to_device(device)?;
        self.layer1.to_device(device)?;
        self.layer2.to_device(device)?;
        self.layer3.to_device(device)?;
        self.layer4.to_device(device)?;
        self.fc.to_device(device)?;
        Ok(())
    }

    fn set_training(&mut self, training: bool) {
        if training {
            self.train();
        } else {
            self.eval();
        }
    }
}

impl VisionModel for ResNet {
    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn input_size(&self) -> (usize, usize) {
        (224, 224)
    }

    fn name(&self) -> &str {
        "ResNet"
    }
}
