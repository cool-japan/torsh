use std::collections::HashMap;
use torsh_core::{device::DeviceType, error::Result};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

use crate::{ModelConfig, VisionModel};

/// VGG model configurations
/// Each tuple represents (out_channels, num_layers) where 'M' means MaxPool
type VGGConfig = &'static [VGGLayer];

#[derive(Debug, Clone)]
enum VGGLayer {
    Conv(usize), // output channels
    MaxPool,
}

/// VGG model implementation
pub struct VGG {
    features: Sequential,
    avgpool: AdaptiveAvgPool2d,
    classifier: Sequential,
    num_classes: usize,
    is_training: bool,
}

impl VGG {
    fn get_config(variant: &str) -> VGGConfig {
        match variant {
            "vgg11" => &[
                VGGLayer::Conv(64),
                VGGLayer::MaxPool,
                VGGLayer::Conv(128),
                VGGLayer::MaxPool,
                VGGLayer::Conv(256),
                VGGLayer::Conv(256),
                VGGLayer::MaxPool,
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::MaxPool,
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::MaxPool,
            ],
            "vgg13" => &[
                VGGLayer::Conv(64),
                VGGLayer::Conv(64),
                VGGLayer::MaxPool,
                VGGLayer::Conv(128),
                VGGLayer::Conv(128),
                VGGLayer::MaxPool,
                VGGLayer::Conv(256),
                VGGLayer::Conv(256),
                VGGLayer::MaxPool,
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::MaxPool,
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::MaxPool,
            ],
            "vgg16" => &[
                VGGLayer::Conv(64),
                VGGLayer::Conv(64),
                VGGLayer::MaxPool,
                VGGLayer::Conv(128),
                VGGLayer::Conv(128),
                VGGLayer::MaxPool,
                VGGLayer::Conv(256),
                VGGLayer::Conv(256),
                VGGLayer::Conv(256),
                VGGLayer::MaxPool,
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::MaxPool,
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::MaxPool,
            ],
            "vgg19" => &[
                VGGLayer::Conv(64),
                VGGLayer::Conv(64),
                VGGLayer::MaxPool,
                VGGLayer::Conv(128),
                VGGLayer::Conv(128),
                VGGLayer::MaxPool,
                VGGLayer::Conv(256),
                VGGLayer::Conv(256),
                VGGLayer::Conv(256),
                VGGLayer::Conv(256),
                VGGLayer::MaxPool,
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::MaxPool,
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::Conv(512),
                VGGLayer::MaxPool,
            ],
            _ => panic!("Unknown VGG variant: {}", variant),
        }
    }

    fn make_features(config: VGGConfig, batch_norm: bool) -> Result<Sequential> {
        let mut layers = Sequential::new();
        let mut in_channels = 3;

        for layer in config {
            match layer {
                VGGLayer::Conv(out_channels) => {
                    layers = layers.add(Conv2d::new(
                        in_channels,
                        *out_channels,
                        (3, 3),
                        (1, 1),
                        (1, 1),
                        (1, 1),
                        false,
                        1,
                    ));

                    if batch_norm {
                        layers = layers.add(BatchNorm2d::new(*out_channels)?);
                    }

                    layers = layers.add(ReLU::new());
                    in_channels = *out_channels;
                }
                VGGLayer::MaxPool => {
                    layers =
                        layers.add(MaxPool2d::new((2, 2), Some((2, 2)), (0, 0), (1, 1), false));
                }
            }
        }

        Ok(layers)
    }

    fn make_classifier(num_classes: usize, dropout: f32) -> Sequential {
        Sequential::new()
            .add(Linear::new(512 * 7 * 7, 4096, true))
            .add(ReLU::new())
            .add(Dropout::new(dropout))
            .add(Linear::new(4096, 4096, true))
            .add(ReLU::new())
            .add(Dropout::new(dropout))
            .add(Linear::new(4096, num_classes, true))
    }

    fn new(variant: &str, config: ModelConfig, batch_norm: bool) -> Result<Self> {
        let vgg_config = Self::get_config(variant);
        let features = Self::make_features(vgg_config, batch_norm)?;
        let avgpool = AdaptiveAvgPool2d::new((Some(7), Some(7)));
        let classifier = Self::make_classifier(config.num_classes, config.dropout);

        Ok(Self {
            features,
            avgpool,
            classifier,
            num_classes: config.num_classes,
            is_training: true,
        })
    }

    pub fn vgg11(config: ModelConfig) -> Result<Self> {
        Self::new("vgg11", config, false)
    }

    pub fn vgg11_bn(config: ModelConfig) -> Result<Self> {
        Self::new("vgg11", config, true)
    }

    pub fn vgg13(config: ModelConfig) -> Result<Self> {
        Self::new("vgg13", config, false)
    }

    pub fn vgg13_bn(config: ModelConfig) -> Result<Self> {
        Self::new("vgg13", config, true)
    }

    pub fn vgg16(config: ModelConfig) -> Result<Self> {
        Self::new("vgg16", config, false)
    }

    pub fn vgg16_bn(config: ModelConfig) -> Result<Self> {
        Self::new("vgg16", config, true)
    }

    pub fn vgg19(config: ModelConfig) -> Result<Self> {
        Self::new("vgg19", config, false)
    }

    pub fn vgg19_bn(config: ModelConfig) -> Result<Self> {
        Self::new("vgg19", config, true)
    }
}

impl Module for VGG {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = self.features.forward(input)?;
        x = self.avgpool.forward(&x)?;

        // Flatten for classifier
        let batch_size = x.shape().dims()[0];
        x = x.view(&[batch_size as i32, -1])?;

        self.classifier.forward(&x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.features.parameters());
        params.extend(self.classifier.parameters());
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
        self.features.train();
        self.classifier.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.features.eval();
        self.classifier.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.features.to_device(device)?;
        self.avgpool.to_device(device)?;
        self.classifier.to_device(device)?;
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

impl VisionModel for VGG {
    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn input_size(&self) -> (usize, usize) {
        (224, 224)
    }

    fn name(&self) -> &str {
        "VGG"
    }
}
