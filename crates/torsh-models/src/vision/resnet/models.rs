//! ResNet model implementations

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use super::blocks::{BasicBlock, BottleneckBlock};
use super::config::{ResNetConfig, ResNetVariant};
use std::collections::HashMap;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// ResNet model implementation
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
    dropout: Option<Dropout>,
    config: ResNetConfig,
    inplanes: usize,
}

impl ResNet {
    /// Create a new ResNet from configuration
    pub fn new(config: ResNetConfig) -> Result<Self> {
        config
            .validate()
            .map_err(|e| TorshError::InvalidOperation(e))?;

        let mut model = Self {
            conv1: Conv2d::new(
                config.in_channels,
                config.stem_channels,
                (7, 7),
                (2, 2),
                (3, 3),
                (1, 1),
                false,
                1,
            ),
            bn1: BatchNorm2d::new(config.stem_channels)?,
            relu: ReLU::new(),
            maxpool: MaxPool2d::new((3, 3), Some((2, 2)), (1, 1), (1, 1), false),
            layer1: Sequential::new(),
            layer2: Sequential::new(),
            layer3: Sequential::new(),
            layer4: Sequential::new(),
            avgpool: AdaptiveAvgPool2d::new((Some(1), Some(1))),
            fc: Linear::new(
                config.stage_channels().last().unwrap() * config.expansion(),
                config.num_classes,
                true,
            ),
            dropout: if config.dropout > 0.0 {
                Some(Dropout::new(config.dropout))
            } else {
                None
            },
            config: config.clone(),
            inplanes: config.stem_channels,
        };

        // Build the residual layers
        let layers = config.variant.layer_config();
        let stage_channels = config.stage_channels();

        model.layer1 = model.make_layer(stage_channels[0], layers[0], 1)?;
        model.layer2 = model.make_layer(stage_channels[1], layers[1], 2)?;
        model.layer3 = model.make_layer(stage_channels[2], layers[2], 2)?;
        model.layer4 = model.make_layer(stage_channels[3], layers[3], 2)?;

        Ok(model)
    }

    /// Create ResNet-18
    pub fn resnet18(num_classes: usize) -> Result<Self> {
        let config = ResNetConfig::resnet18(num_classes);
        Self::new(config)
    }

    /// Create ResNet-34
    pub fn resnet34(num_classes: usize) -> Result<Self> {
        let config = ResNetConfig::resnet34(num_classes);
        Self::new(config)
    }

    /// Create ResNet-50
    pub fn resnet50(num_classes: usize) -> Result<Self> {
        let config = ResNetConfig::resnet50(num_classes);
        Self::new(config)
    }

    /// Create ResNet-101
    pub fn resnet101(num_classes: usize) -> Result<Self> {
        let config = ResNetConfig::resnet101(num_classes);
        Self::new(config)
    }

    /// Create ResNet-152
    pub fn resnet152(num_classes: usize) -> Result<Self> {
        let config = ResNetConfig::resnet152(num_classes);
        Self::new(config)
    }

    /// Create ResNet from configuration
    pub fn from_config(config: ResNetConfig) -> Result<Self> {
        Self::new(config)
    }

    /// Make a residual layer
    fn make_layer(&mut self, planes: usize, blocks: usize, stride: usize) -> Result<Sequential> {
        let mut downsample = None;

        // Create downsample if needed
        if stride != 1 || self.inplanes != planes * self.config.expansion() {
            let downsample_conv = Conv2d::new(
                self.inplanes,
                planes * self.config.expansion(),
                (1, 1),
                (stride, stride),
                (0, 0),
                (1, 1),
                false,
                1,
            );
            let downsample_bn = BatchNorm2d::new(planes * self.config.expansion())?;

            let seq = Sequential::new().add(downsample_conv).add(downsample_bn);
            downsample = Some(seq);
        }

        let mut layers = Sequential::new();

        // First block with potential stride and downsample
        if self.config.variant.uses_bottleneck() {
            let block = BottleneckBlock::new(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.config.groups,
                self.config.width_per_group,
                self.config.use_se,
                self.config.se_reduction_ratio,
            )?;
            layers = layers.add(block);
        } else {
            let block = BasicBlock::new(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.config.use_se,
                self.config.se_reduction_ratio,
            )?;
            layers = layers.add(block);
        }

        self.inplanes = planes * self.config.expansion();

        // Remaining blocks
        for _i in 1..blocks {
            if self.config.variant.uses_bottleneck() {
                let block = BottleneckBlock::new(
                    self.inplanes,
                    planes,
                    1,
                    None,
                    self.config.groups,
                    self.config.width_per_group,
                    self.config.use_se,
                    self.config.se_reduction_ratio,
                )?;
                layers = layers.add(block);
            } else {
                let block = BasicBlock::new(
                    self.inplanes,
                    planes,
                    1,
                    None,
                    self.config.use_se,
                    self.config.se_reduction_ratio,
                )?;
                layers = layers.add(block);
            }
        }

        Ok(layers)
    }

    /// Extract features (without classification head)
    pub fn extract_features(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv1.forward(x)?;
        x = self.bn1.forward(&x)?;
        x = self.relu.forward(&x)?;
        x = self.maxpool.forward(&x)?;

        x = self.layer1.forward(&x)?;
        x = self.layer2.forward(&x)?;
        x = self.layer3.forward(&x)?;
        x = self.layer4.forward(&x)?;

        x = self.avgpool.forward(&x)?;
        x = x.flatten()?;

        Ok(x)
    }

    /// Get the feature dimensions
    pub fn feature_dim(&self) -> usize {
        // stage_channels() already includes expansion for bottleneck architectures
        *self.config.stage_channels().last().unwrap()
    }

    /// Get the configuration
    pub fn config(&self) -> &ResNetConfig {
        &self.config
    }
}

impl Module for ResNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.extract_features(x)?;

        // Apply dropout if configured
        if let Some(ref dropout) = self.dropout {
            x = dropout.forward(&x)?;
        }

        x = self.fc.forward(&x)?;

        Ok(x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.conv1.parameters() {
            params.insert(format!("conv1.{}", name), param);
        }
        for (name, param) in self.bn1.parameters() {
            params.insert(format!("bn1.{}", name), param);
        }
        for (name, param) in self.layer1.parameters() {
            params.insert(format!("layer1.{}", name), param);
        }
        for (name, param) in self.layer2.parameters() {
            params.insert(format!("layer2.{}", name), param);
        }
        for (name, param) in self.layer3.parameters() {
            params.insert(format!("layer3.{}", name), param);
        }
        for (name, param) in self.layer4.parameters() {
            params.insert(format!("layer4.{}", name), param);
        }
        for (name, param) in self.fc.parameters() {
            params.insert(format!("fc.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.conv1.training()
            && self.bn1.training()
            && self.layer1.training()
            && self.layer2.training()
            && self.layer3.training()
            && self.layer4.training()
            && self.fc.training()
    }

    fn train(&mut self) {
        self.conv1.train();
        self.bn1.train();
        self.relu.train();
        self.maxpool.train();
        self.layer1.train();
        self.layer2.train();
        self.layer3.train();
        self.layer4.train();
        self.avgpool.train();
        self.fc.train();
        if let Some(ref mut dropout) = self.dropout {
            dropout.train();
        }
    }

    fn eval(&mut self) {
        self.conv1.eval();
        self.bn1.eval();
        self.relu.eval();
        self.maxpool.eval();
        self.layer1.eval();
        self.layer2.eval();
        self.layer3.eval();
        self.layer4.eval();
        self.avgpool.eval();
        self.fc.eval();
        if let Some(ref mut dropout) = self.dropout {
            dropout.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.conv1.to_device(device)?;
        self.bn1.to_device(device)?;
        self.layer1.to_device(device)?;
        self.layer2.to_device(device)?;
        self.layer3.to_device(device)?;
        self.layer4.to_device(device)?;
        self.fc.to_device(device)?;
        if let Some(ref mut dropout) = self.dropout {
            dropout.to_device(device)?;
        }
        Ok(())
    }
}

/// ResNet model builder for easy construction
pub struct ResNetBuilder {
    config: ResNetConfig,
}

impl ResNetBuilder {
    /// Create a new builder
    pub fn new(variant: ResNetVariant) -> Self {
        Self {
            config: ResNetConfig {
                variant,
                ..Default::default()
            },
        }
    }

    /// Set number of classes
    pub fn num_classes(mut self, num_classes: usize) -> Self {
        self.config.num_classes = num_classes;
        self
    }

    /// Set input channels
    pub fn input_channels(mut self, channels: usize) -> Self {
        self.config.in_channels = channels;
        self
    }

    /// Enable Squeeze-and-Excitation
    pub fn with_se(mut self, reduction_ratio: usize) -> Self {
        self.config.use_se = true;
        self.config.se_reduction_ratio = reduction_ratio;
        self
    }

    /// Set dropout probability
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.config.dropout = dropout;
        self
    }

    /// Configure for ResNeXt
    pub fn resnext(mut self, groups: usize, width_per_group: usize) -> Self {
        self.config.groups = groups;
        self.config.width_per_group = width_per_group;
        self
    }

    /// Build the ResNet model
    pub fn build(self) -> Result<ResNet> {
        ResNet::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resnet18_creation() -> Result<()> {
        let model = ResNet::resnet18(10)?;
        assert_eq!(model.config().variant, ResNetVariant::ResNet18);
        assert_eq!(model.config().num_classes, 10);
        assert_eq!(model.feature_dim(), 512); // ResNet-18 final feature dimension
        Ok(())
    }

    #[test]
    fn test_resnet50_creation() -> Result<()> {
        let model = ResNet::resnet50(1000)?;
        assert_eq!(model.config().variant, ResNetVariant::ResNet50);
        assert_eq!(model.config().num_classes, 1000);
        assert_eq!(model.feature_dim(), 2048); // ResNet-50 final feature dimension
        Ok(())
    }

    #[test]
    fn test_resnet_builder() -> Result<()> {
        let model = ResNetBuilder::new(ResNetVariant::ResNet34)
            .num_classes(100)
            .input_channels(1)
            .with_se(16)
            .dropout(0.1)
            .build()?;

        assert_eq!(model.config().variant, ResNetVariant::ResNet34);
        assert_eq!(model.config().num_classes, 100);
        assert_eq!(model.config().in_channels, 1);
        assert!(model.config().use_se);
        assert_eq!(model.config().dropout, 0.1);

        Ok(())
    }

    #[test]
    fn test_config_validation() {
        let mut config = ResNetConfig::resnet18(0); // Invalid: 0 classes
        assert!(config.validate().is_err());

        config.num_classes = 10;
        assert!(config.validate().is_ok());
    }

    /*
    // These tests would require actual tensor operations
    #[test]
    fn test_resnet_forward_pass() -> Result<()> {
        let model = ResNet::resnet18(10)?;
        let input = creation::randn(&[2, 3, 224, 224])?; // Batch of 2 RGB images
        let output = model.forward(&input)?;
        assert_eq!(output.shape(), &[2, 10]); // Batch size x num_classes
        Ok(())
    }

    #[test]
    fn test_feature_extraction() -> Result<()> {
        let model = ResNet::resnet50(1000)?;
        let input = creation::randn(&[1, 3, 224, 224])?;
        let features = model.extract_features(&input)?;
        assert_eq!(features.shape(), &[1, 2048]); // Batch size x feature_dim
        Ok(())
    }
    */
}
