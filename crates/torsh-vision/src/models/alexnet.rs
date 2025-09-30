use std::collections::HashMap;
use torsh_core::{device::DeviceType, error::Result};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

use crate::{ModelConfig, VisionModel};

/// AlexNet model - placeholder implementation
pub struct AlexNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc: Linear,
    num_classes: usize,
    is_training: bool,
}

impl AlexNet {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            conv1: Conv2d::new(3, 64, (11, 11), (4, 4), (2, 2), (1, 1), false, 1),
            conv2: Conv2d::new(64, 128, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            fc: Linear::new(128, config.num_classes, true),
            num_classes: config.num_classes,
            is_training: true,
        }
    }
}

impl Module for AlexNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = self.conv1.forward(input)?;
        x = self.conv2.forward(&x)?;
        self.fc.forward(&x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
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
        self.conv2.train();
        self.fc.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.conv1.eval();
        self.conv2.eval();
        self.fc.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.conv1.to_device(device)?;
        self.conv2.to_device(device)?;
        self.fc.to_device(device)?;
        Ok(())
    }
}

impl VisionModel for AlexNet {
    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn input_size(&self) -> (usize, usize) {
        (224, 224)
    }

    fn name(&self) -> &str {
        "AlexNet"
    }
}
