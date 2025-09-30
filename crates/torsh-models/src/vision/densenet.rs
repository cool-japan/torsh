//! DenseNet Architecture Implementation for ToRSh Deep Learning Framework
//!
//! This module provides a comprehensive implementation of the DenseNet family of neural networks,
//! which introduced dense connectivity patterns and efficient parameter usage.
//!
//! ## Key Features
//!
//! - **Dense Connectivity**: Each layer receives feature maps from all preceding layers
//! - **Growth Rate**: Controls the number of feature maps each layer contributes
//! - **Bottleneck Design**: 1x1 convolutions reduce computational cost
//! - **Transition Layers**: Reduce feature map dimensions between dense blocks
//! - **Compression**: Reduces the number of feature maps in transition layers
//!
//! ## Supported Variants
//!
//! - **DenseNet-121**: 4 dense blocks with [6, 12, 24, 16] layers (~8M parameters)
//! - **DenseNet-169**: 4 dense blocks with [6, 12, 32, 32] layers (~14M parameters)
//! - **DenseNet-201**: 4 dense blocks with [6, 12, 48, 32] layers (~20M parameters)
//! - **DenseNet-161**: 4 dense blocks with [6, 12, 36, 24] layers and growth_rate=48 (~29M parameters)
//!
//! ## Architecture Overview
//!
//! ```text
//! Input -> Initial Conv -> MaxPool -> Dense Block 1 -> Transition 1 ->
//!          Dense Block 2 -> Transition 2 -> Dense Block 3 -> Transition 3 ->
//!          Dense Block 4 -> Global AvgPool -> Classifier -> Output
//! ```
//!
//! ## Example Usage
//!
//! ```rust
//! use torsh_models::vision::densenet::*;
//!
//! // Create DenseNet-121 for ImageNet classification
//! let model = DenseNet::densenet121(1000);
//!
//! // Create DenseNet-169 with custom growth rate
//! let custom_model = DenseNet::new(
//!     32,               // growth_rate
//!     &[6, 12, 32, 32], // block_config
//!     64,               // num_init_features
//!     4,                // bn_size
//!     Some(0.0),        // dropout_rate
//!     1000,             // num_classes
//!     0.5,              // compression_factor
//! );
//!
//! // Forward pass
//! let input = torsh_tensor::creation::randn(&[1, 3, 224, 224])?;
//! let output = model.forward(&input)?;
//! ```

use crate::error::{Result, TorshError};
use std::collections::HashMap;
use torsh_core::{DeviceType, Tensor};
use torsh_nn::{
    activations::ReLU,
    dropout::Dropout,
    layers::{Conv2d, Linear},
    normalization::BatchNorm2d,
    pooling::{AdaptiveAvgPool2d, AvgPool2d, MaxPool2d},
    module::{Module, Parameter},
};

/// Dense Layer for DenseNet
///
/// The fundamental building block of DenseNet that implements:
/// 1. **Bottleneck**: BN-ReLU-Conv1x1 to reduce channels (bn_size * growth_rate)
/// 2. **Growth**: BN-ReLU-Conv3x3 to produce growth_rate new feature maps
/// 3. **Concatenation**: New features are concatenated with input
///
/// The bottleneck design significantly reduces computational cost by first
/// reducing the number of input feature maps before the expensive 3x3 convolution.
#[derive(Debug)]
pub struct DenseLayer {
    bn1: BatchNorm2d,
    relu1: ReLU,
    conv1: Conv2d,      // 1x1 bottleneck
    bn2: BatchNorm2d,
    relu2: ReLU,
    conv2: Conv2d,      // 3x3 growth
    dropout: Option<Dropout>,
    growth_rate: usize,
}

impl DenseLayer {
    /// Creates a new dense layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels from previous layers
    /// * `growth_rate` - Number of feature maps this layer will produce (typically 32)
    /// * `bn_size` - Bottleneck size multiplier (typically 4)
    /// * `dropout_rate` - Optional dropout rate for regularization
    pub fn new(
        in_channels: usize,
        growth_rate: usize,
        bn_size: usize,
        dropout_rate: Option<f32>,
    ) -> Self {
        let bottleneck_channels = bn_size * growth_rate;

        // 1x1 conv for bottleneck (reduces computational cost)
        let bn1 = BatchNorm2d::new(in_channels);
        let relu1 = ReLU::new();
        let conv1 = Conv2d::new(
            in_channels,
            bottleneck_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            false,
            1,
        );

        // 3x3 conv for growth (produces growth_rate feature maps)
        let bn2 = BatchNorm2d::new(bottleneck_channels);
        let relu2 = ReLU::new();
        let conv2 = Conv2d::new(
            bottleneck_channels,
            growth_rate,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            false,
            1,
        );

        let dropout = dropout_rate.map(Dropout::new);

        Self {
            bn1,
            relu1,
            conv1,
            bn2,
            relu2,
            conv2,
            dropout,
            growth_rate,
        }
    }

    /// Get the growth rate (number of output feature maps)
    pub fn growth_rate(&self) -> usize {
        self.growth_rate
    }
}

impl Module for DenseLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Bottleneck path: BN -> ReLU -> Conv1x1
        let mut bottleneck = self.bn1.forward(x)?;
        bottleneck = self.relu1.forward(&bottleneck)?;
        bottleneck = self.conv1.forward(&bottleneck)?;

        if let Some(ref dropout) = self.dropout {
            bottleneck = dropout.forward(&bottleneck)?;
        }

        // Growth path: BN -> ReLU -> Conv3x3
        let mut new_features = self.bn2.forward(&bottleneck)?;
        new_features = self.relu2.forward(&new_features)?;
        new_features = self.conv2.forward(&new_features)?;

        if let Some(ref dropout) = self.dropout {
            new_features = dropout.forward(&new_features)?;
        }

        // Dense connectivity: concatenate new features with input
        Tensor::cat(&[x, &new_features], 1)
    }

    fn train(&mut self) {
        self.bn1.train();
        self.conv1.train();
        self.bn2.train();
        self.conv2.train();
        if let Some(ref mut dropout) = self.dropout {
            dropout.train();
        }
    }

    fn eval(&mut self) {
        self.bn1.eval();
        self.conv1.eval();
        self.bn2.eval();
        self.conv2.eval();
        if let Some(ref mut dropout) = self.dropout {
            dropout.eval();
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Bottleneck parameters
        for (name, param) in self.bn1.parameters() {
            params.insert(format!("norm1.{}", name), param);
        }
        for (name, param) in self.conv1.parameters() {
            params.insert(format!("conv1.{}", name), param);
        }

        // Growth parameters
        for (name, param) in self.bn2.parameters() {
            params.insert(format!("norm2.{}", name), param);
        }
        for (name, param) in self.conv2.parameters() {
            params.insert(format!("conv2.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.bn1.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.bn1.to_device(device)?;
        self.conv1.to_device(device)?;
        self.bn2.to_device(device)?;
        self.conv2.to_device(device)?;
        Ok(())
    }
}

/// Dense Block containing multiple Dense Layers
///
/// A dense block consists of multiple dense layers with dense connectivity.
/// Each layer receives feature maps from all previous layers in the block,
/// leading to feature reuse and improved gradient flow.
#[derive(Debug)]
pub struct DenseBlock {
    layers: Vec<DenseLayer>,
    growth_rate: usize,
}

impl DenseBlock {
    /// Creates a new dense block
    ///
    /// # Arguments
    /// * `num_layers` - Number of dense layers in this block
    /// * `in_channels` - Number of input channels
    /// * `growth_rate` - Growth rate for each layer
    /// * `bn_size` - Bottleneck size multiplier
    /// * `dropout_rate` - Optional dropout rate
    pub fn new(
        num_layers: usize,
        in_channels: usize,
        growth_rate: usize,
        bn_size: usize,
        dropout_rate: Option<f32>,
    ) -> Self {
        let mut layers = Vec::new();

        for i in 0..num_layers {
            // Each layer receives features from all previous layers
            let layer_in_channels = in_channels + i * growth_rate;
            layers.push(DenseLayer::new(
                layer_in_channels,
                growth_rate,
                bn_size,
                dropout_rate,
            ));
        }

        Self { layers, growth_rate }
    }

    /// Calculate output channels after this dense block
    pub fn out_channels(&self, in_channels: usize) -> usize {
        in_channels + self.layers.len() * self.growth_rate
    }

    /// Get number of layers in this block
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get growth rate
    pub fn growth_rate(&self) -> usize {
        self.growth_rate
    }
}

impl Module for DenseBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut features = x.clone();

        // Apply each dense layer, concatenating features
        for layer in &self.layers {
            features = layer.forward(&features)?;
        }

        Ok(features)
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                let prefixed_name = format!("denselayer{}.{}", i + 1, name);
                params.insert(prefixed_name, param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.layers.first().map(|l| l.training()).unwrap_or(true)
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

/// Transition Layer between Dense Blocks
///
/// Transition layers are placed between dense blocks to:
/// 1. **Reduce feature map dimensions**: Use 1x1 conv to reduce channels
/// 2. **Downsample spatial dimensions**: Use average pooling with stride 2
/// 3. **Control model complexity**: Apply compression to limit parameter growth
#[derive(Debug)]
pub struct TransitionLayer {
    bn: BatchNorm2d,
    relu: ReLU,
    conv: Conv2d,       // 1x1 compression
    pool: AvgPool2d,    // 2x2 downsampling
}

impl TransitionLayer {
    /// Creates a new transition layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels (typically in_channels * compression_factor)
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        let bn = BatchNorm2d::new(in_channels);
        let relu = ReLU::new();
        let conv = Conv2d::new(
            in_channels,
            out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            false,
            1,
        );
        let pool = AvgPool2d::new((2, 2)).stride((2, 2));

        Self {
            bn,
            relu,
            conv,
            pool,
        }
    }

    /// Get compression ratio
    pub fn compression_ratio(&self, in_channels: usize, out_channels: usize) -> f32 {
        out_channels as f32 / in_channels as f32
    }
}

impl Module for TransitionLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.bn.forward(x)?;
        x = self.relu.forward(&x)?;
        x = self.conv.forward(&x)?;
        x = self.pool.forward(&x)?;
        Ok(x)
    }

    fn train(&mut self) {
        self.bn.train();
        self.conv.train();
    }

    fn eval(&mut self) {
        self.bn.eval();
        self.conv.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.bn.parameters() {
            params.insert(format!("norm.{}", name), param);
        }
        for (name, param) in self.conv.parameters() {
            params.insert(format!("conv.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.bn.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.bn.to_device(device)?;
        self.conv.to_device(device)?;
        Ok(())
    }
}

/// DenseNet Configuration
#[derive(Debug, Clone)]
pub struct DenseNetConfig {
    pub growth_rate: usize,
    pub block_config: Vec<usize>,
    pub num_init_features: usize,
    pub bn_size: usize,
    pub dropout_rate: Option<f32>,
    pub num_classes: usize,
    pub compression_factor: f32,
}

impl DenseNetConfig {
    /// DenseNet-121 configuration
    pub fn densenet121(num_classes: usize) -> Self {
        Self {
            growth_rate: 32,
            block_config: vec![6, 12, 24, 16],
            num_init_features: 64,
            bn_size: 4,
            dropout_rate: Some(0.0),
            num_classes,
            compression_factor: 0.5,
        }
    }

    /// DenseNet-169 configuration
    pub fn densenet169(num_classes: usize) -> Self {
        Self {
            growth_rate: 32,
            block_config: vec![6, 12, 32, 32],
            num_init_features: 64,
            bn_size: 4,
            dropout_rate: Some(0.0),
            num_classes,
            compression_factor: 0.5,
        }
    }

    /// DenseNet-201 configuration
    pub fn densenet201(num_classes: usize) -> Self {
        Self {
            growth_rate: 32,
            block_config: vec![6, 12, 48, 32],
            num_init_features: 64,
            bn_size: 4,
            dropout_rate: Some(0.0),
            num_classes,
            compression_factor: 0.5,
        }
    }

    /// DenseNet-161 configuration (wider growth rate)
    pub fn densenet161(num_classes: usize) -> Self {
        Self {
            growth_rate: 48,
            block_config: vec![6, 12, 36, 24],
            num_init_features: 96,
            bn_size: 4,
            dropout_rate: Some(0.0),
            num_classes,
            compression_factor: 0.5,
        }
    }
}

/// DenseNet model
///
/// Implements "Densely Connected Convolutional Networks" architecture.
/// Key innovations:
/// - Dense connectivity pattern (each layer connects to all subsequent layers)
/// - Feature reuse and improved gradient flow
/// - Efficient parameter usage
/// - Bottleneck design to reduce computational cost
#[derive(Debug)]
pub struct DenseNet {
    config: DenseNetConfig,

    // Initial feature extraction
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    maxpool: MaxPool2d,

    // Dense blocks and transitions
    dense_blocks: Vec<DenseBlock>,
    transitions: Vec<TransitionLayer>,

    // Final classification
    final_bn: BatchNorm2d,
    final_relu: ReLU,
    avgpool: AdaptiveAvgPool2d,
    classifier: Linear,

    growth_rate: usize,
}

impl DenseNet {
    /// Creates a new DenseNet model with custom configuration
    ///
    /// # Arguments
    /// * `growth_rate` - Number of feature maps each layer produces (typically 32)
    /// * `block_config` - Number of layers in each dense block (e.g., [6, 12, 24, 16])
    /// * `num_init_features` - Number of feature maps after initial convolution (typically 64)
    /// * `bn_size` - Bottleneck size multiplier (typically 4)
    /// * `dropout_rate` - Optional dropout rate
    /// * `num_classes` - Number of output classes
    /// * `compression_factor` - Compression factor for transition layers (typically 0.5)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        growth_rate: usize,
        block_config: &[usize],
        num_init_features: usize,
        bn_size: usize,
        dropout_rate: Option<f32>,
        num_classes: usize,
        compression_factor: f32,
    ) -> Self {
        let config = DenseNetConfig {
            growth_rate,
            block_config: block_config.to_vec(),
            num_init_features,
            bn_size,
            dropout_rate,
            num_classes,
            compression_factor,
        };

        let relu = ReLU::new();

        // Initial feature extraction (following ImageNet pre-processing)
        let conv1 = Conv2d::new(
            3,                    // RGB input
            num_init_features,
            (7, 7),
            (2, 2),               // stride 2 for downsampling
            (3, 3),               // padding to maintain reasonable size
            (1, 1),
            false,
            1,
        );
        let bn1 = BatchNorm2d::new(num_init_features);
        let maxpool = MaxPool2d::new((3, 3)).stride((2, 2)).padding((1, 1));

        let mut dense_blocks = Vec::new();
        let mut transitions = Vec::new();
        let mut num_features = num_init_features;

        // Create dense blocks and transitions
        for (i, &num_layers) in block_config.iter().enumerate() {
            // Dense block
            let block = DenseBlock::new(
                num_layers,
                num_features,
                growth_rate,
                bn_size,
                dropout_rate,
            );
            dense_blocks.push(block);

            // Update feature count after dense block
            num_features = num_features + num_layers * growth_rate;

            // Transition layer (except after the last block)
            if i != block_config.len() - 1 {
                let out_features = (num_features as f32 * compression_factor) as usize;
                let transition = TransitionLayer::new(num_features, out_features);
                transitions.push(transition);
                num_features = out_features;
            }
        }

        // Final classification layers
        let final_bn = BatchNorm2d::new(num_features);
        let final_relu = ReLU::new();
        let avgpool = AdaptiveAvgPool2d::new((Some(1), Some(1)));
        let classifier = Linear::new(num_features, num_classes, true);

        Self {
            config,
            conv1,
            bn1,
            relu,
            maxpool,
            dense_blocks,
            transitions,
            final_bn,
            final_relu,
            avgpool,
            classifier,
            growth_rate,
        }
    }

    /// Create DenseNet-121 (8M parameters)
    pub fn densenet121(num_classes: usize) -> Self {
        let config = DenseNetConfig::densenet121(num_classes);
        Self::new(
            config.growth_rate,
            &config.block_config,
            config.num_init_features,
            config.bn_size,
            config.dropout_rate,
            config.num_classes,
            config.compression_factor,
        )
    }

    /// Create DenseNet-169 (14M parameters)
    pub fn densenet169(num_classes: usize) -> Self {
        let config = DenseNetConfig::densenet169(num_classes);
        Self::new(
            config.growth_rate,
            &config.block_config,
            config.num_init_features,
            config.bn_size,
            config.dropout_rate,
            config.num_classes,
            config.compression_factor,
        )
    }

    /// Create DenseNet-201 (20M parameters)
    pub fn densenet201(num_classes: usize) -> Self {
        let config = DenseNetConfig::densenet201(num_classes);
        Self::new(
            config.growth_rate,
            &config.block_config,
            config.num_init_features,
            config.bn_size,
            config.dropout_rate,
            config.num_classes,
            config.compression_factor,
        )
    }

    /// Create DenseNet-161 (29M parameters, wider growth rate)
    pub fn densenet161(num_classes: usize) -> Self {
        let config = DenseNetConfig::densenet161(num_classes);
        Self::new(
            config.growth_rate,
            &config.block_config,
            config.num_init_features,
            config.bn_size,
            config.dropout_rate,
            config.num_classes,
            config.compression_factor,
        )
    }

    /// Get model configuration
    pub fn config(&self) -> &DenseNetConfig {
        &self.config
    }

    /// Get growth rate
    pub fn growth_rate(&self) -> usize {
        self.growth_rate
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters().values().map(|p| {
            let data = p.data.read().unwrap();
            data.numel()
        }).sum()
    }

    /// Get model complexity information
    pub fn complexity_info(&self) -> (usize, usize, f32) {
        let params = self.num_parameters();
        let total_layers: usize = self.config.block_config.iter().sum();
        let compression = self.config.compression_factor;
        (params, total_layers, compression)
    }
}

impl Module for DenseNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Initial feature extraction
        let mut x = self.conv1.forward(x)?;
        x = self.bn1.forward(&x)?;
        x = self.relu.forward(&x)?;
        x = self.maxpool.forward(&x)?;

        // Dense blocks with transitions
        for (i, block) in self.dense_blocks.iter().enumerate() {
            x = block.forward(&x)?;

            // Apply transition if not the last block
            if i < self.transitions.len() {
                x = self.transitions[i].forward(&x)?;
            }
        }

        // Final layers
        x = self.final_bn.forward(&x)?;
        x = self.final_relu.forward(&x)?;
        x = self.avgpool.forward(&x)?;

        // Classifier
        let batch_size = x.size(0)?;
        let features = x.size(1)?;
        x = x.view(&[batch_size, features])?;
        x = self.classifier.forward(&x)?;

        Ok(x)
    }

    fn train(&mut self) {
        self.conv1.train();
        self.bn1.train();
        for block in &mut self.dense_blocks {
            block.train();
        }
        for transition in &mut self.transitions {
            transition.train();
        }
        self.final_bn.train();
        self.classifier.train();
    }

    fn eval(&mut self) {
        self.conv1.eval();
        self.bn1.eval();
        for block in &mut self.dense_blocks {
            block.eval();
        }
        for transition in &mut self.transitions {
            transition.eval();
        }
        self.final_bn.eval();
        self.classifier.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Initial layers
        for (name, param) in self.conv1.parameters() {
            params.insert(format!("features.conv0.{}", name), param);
        }
        for (name, param) in self.bn1.parameters() {
            params.insert(format!("features.norm0.{}", name), param);
        }

        // Dense blocks and transitions
        let mut layer_idx = 0;
        for (block_idx, block) in self.dense_blocks.iter().enumerate() {
            for (name, param) in block.parameters() {
                let prefixed_name = format!("features.denseblock{}.{}", block_idx + 1, name);
                params.insert(prefixed_name, param);
            }

            // Transition layer (if exists)
            if block_idx < self.transitions.len() {
                for (name, param) in self.transitions[block_idx].parameters() {
                    let prefixed_name = format!("features.transition{}.{}", block_idx + 1, name);
                    params.insert(prefixed_name, param);
                }
            }
        }

        // Final layers
        for (name, param) in self.final_bn.parameters() {
            params.insert(format!("features.norm5.{}", name), param);
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
        self.conv1.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.conv1.to_device(device)?;
        self.bn1.to_device(device)?;

        for block in &mut self.dense_blocks {
            block.to_device(device)?;
        }

        for transition in &mut self.transitions {
            transition.to_device(device)?;
        }

        self.final_bn.to_device(device)?;
        self.classifier.to_device(device)?;

        Ok(())
    }
}

/// Factory for creating DenseNet variants
pub struct DenseNetFactory;

impl DenseNetFactory {
    /// Create any DenseNet variant by name
    pub fn create(variant: &str, num_classes: usize) -> Result<DenseNet> {
        match variant.to_lowercase().as_str() {
            "121" | "densenet-121" | "densenet121" => {
                Ok(DenseNet::densenet121(num_classes))
            }
            "169" | "densenet-169" | "densenet169" => {
                Ok(DenseNet::densenet169(num_classes))
            }
            "201" | "densenet-201" | "densenet201" => {
                Ok(DenseNet::densenet201(num_classes))
            }
            "161" | "densenet-161" | "densenet161" => {
                Ok(DenseNet::densenet161(num_classes))
            }
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown DenseNet variant: {}. Available: 121, 169, 201, 161",
                variant
            ))),
        }
    }

    /// Get model information
    pub fn model_info(variant: &str) -> Result<String> {
        let info = match variant.to_lowercase().as_str() {
            "121" | "densenet-121" => {
                "DenseNet-121: 4 dense blocks [6,12,24,16], growth_rate=32 (~8M parameters)"
            }
            "169" | "densenet-169" => {
                "DenseNet-169: 4 dense blocks [6,12,32,32], growth_rate=32 (~14M parameters)"
            }
            "201" | "densenet-201" => {
                "DenseNet-201: 4 dense blocks [6,12,48,32], growth_rate=32 (~20M parameters)"
            }
            "161" | "densenet-161" => {
                "DenseNet-161: 4 dense blocks [6,12,36,24], growth_rate=48 (~29M parameters)"
            }
            _ => return Err(TorshError::InvalidArgument(format!("Unknown variant: {}", variant))),
        };
        Ok(info.to_string())
    }

    /// List all available variants
    pub fn available_variants() -> Vec<&'static str> {
        vec!["121", "169", "201", "161"]
    }

    /// Get model complexity comparison
    pub fn complexity_comparison() -> String {
        format!(
            "DenseNet Complexity Comparison:\n\
            ┌─────────────┬─────────────┬──────────────┬──────────────┬─────────────┐\n\
            │ Variant     │ Blocks      │ Growth Rate  │ Parameters   │ Accuracy    │\n\
            ├─────────────┼─────────────┼──────────────┼──────────────┼─────────────┤\n\
            │ DenseNet-121│ [6,12,24,16]│ 32           │ ~8M          │ Baseline    │\n\
            │ DenseNet-169│ [6,12,32,32]│ 32           │ ~14M         │ Higher      │\n\
            │ DenseNet-201│ [6,12,48,32]│ 32           │ ~20M         │ Highest-32  │\n\
            │ DenseNet-161│ [6,12,36,24]│ 48           │ ~29M         │ Highest-48  │\n\
            └─────────────┴─────────────┴──────────────┴──────────────┴─────────────┘"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::Tensor;

    #[test]
    fn test_dense_layer() {
        let mut layer = DenseLayer::new(64, 32, 4, Some(0.1));

        assert_eq!(layer.growth_rate(), 32);

        let input = torsh_tensor::creation::randn(&[2, 64, 56, 56]).unwrap();
        let output = layer.forward(&input).unwrap();

        // Output should have input channels + growth_rate channels
        assert_eq!(output.shape(), &[2, 96, 56, 56]); // 64 + 32 = 96

        // Test train/eval modes
        layer.train();
        assert!(layer.training());

        layer.eval();
        assert!(!layer.training());
    }

    #[test]
    fn test_dense_block() {
        let mut block = DenseBlock::new(6, 64, 32, 4, None);

        assert_eq!(block.num_layers(), 6);
        assert_eq!(block.growth_rate(), 32);
        assert_eq!(block.out_channels(64), 256); // 64 + 6*32 = 256

        let input = torsh_tensor::creation::randn(&[1, 64, 56, 56]).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 256, 56, 56]);
    }

    #[test]
    fn test_transition_layer() {
        let mut transition = TransitionLayer::new(256, 128);

        assert_eq!(transition.compression_ratio(256, 128), 0.5);

        let input = torsh_tensor::creation::randn(&[1, 256, 56, 56]).unwrap();
        let output = transition.forward(&input).unwrap();

        // Should reduce both channels and spatial dimensions
        assert_eq!(output.shape(), &[1, 128, 28, 28]); // /2 spatial, /2 channels
    }

    #[test]
    fn test_densenet_variants() {
        let variants = [
            ("121", DenseNet::densenet121(1000)),
            ("169", DenseNet::densenet169(1000)),
            ("201", DenseNet::densenet201(1000)),
            ("161", DenseNet::densenet161(1000)),
        ];

        for (name, model) in variants {
            let input = torsh_tensor::creation::randn(&[1, 3, 224, 224]).unwrap();
            let output = model.forward(&input).unwrap();

            // All should output 1000 classes
            assert_eq!(output.shape(), &[1, 1000], "Failed for DenseNet-{}", name);

            // Check that 161 has different growth rate
            if name == "161" {
                assert_eq!(model.growth_rate(), 48);
            } else {
                assert_eq!(model.growth_rate(), 32);
            }
        }
    }

    #[test]
    fn test_densenet_factory() {
        // Test factory creation
        let model = DenseNetFactory::create("121", 1000).unwrap();
        assert_eq!(model.config().num_classes, 1000);
        assert_eq!(model.config().growth_rate, 32);

        // Test larger variant
        let large_model = DenseNetFactory::create("201", 100).unwrap();
        assert_eq!(large_model.config().block_config, vec![6, 12, 48, 32]);

        // Test invalid variant
        assert!(DenseNetFactory::create("invalid", 1000).is_err());

        // Test model info
        let info = DenseNetFactory::model_info("169").unwrap();
        assert!(info.contains("DenseNet-169"));
        assert!(info.contains("14M"));

        // Test available variants
        let variants = DenseNetFactory::available_variants();
        assert!(variants.contains(&"121"));
        assert!(variants.contains(&"201"));
    }

    #[test]
    fn test_densenet_config() {
        let config = DenseNetConfig::densenet121(1000);
        assert_eq!(config.num_classes, 1000);
        assert_eq!(config.growth_rate, 32);
        assert_eq!(config.block_config, vec![6, 12, 24, 16]);
        assert_eq!(config.compression_factor, 0.5);

        let config_161 = DenseNetConfig::densenet161(100);
        assert_eq!(config_161.growth_rate, 48);
        assert_eq!(config_161.num_init_features, 96);
    }

    #[test]
    fn test_model_complexity() {
        let models = [
            ("121", DenseNet::densenet121(10)),
            ("169", DenseNet::densenet169(10)),
            ("201", DenseNet::densenet201(10)),
            ("161", DenseNet::densenet161(10)),
        ];

        let mut prev_params = 0;
        for (name, model) in &models {
            let (params, total_layers, compression) = model.complexity_info();

            // Parameters should generally increase with model size
            if name != &"121" {
                assert!(params >= prev_params, "Parameters should increase for {}", name);
            }

            assert_eq!(compression, 0.5);
            assert!(total_layers > 30, "Should have >30 total layers");

            prev_params = params;
        }
    }

    #[test]
    fn test_densenet_parameters() {
        let model = DenseNet::densenet121(10);
        let params = model.parameters();

        // Should have features and classifier parameters
        assert!(params.keys().any(|k| k.starts_with("features")));
        assert!(params.keys().any(|k| k.starts_with("classifier")));

        // Should have dense block parameters
        assert!(params.keys().any(|k| k.contains("denseblock")));
        assert!(params.keys().any(|k| k.contains("transition")));

        // Should have initial conv and norm
        assert!(params.keys().any(|k| k.contains("conv0")));
        assert!(params.keys().any(|k| k.contains("norm0")));
    }

    #[test]
    fn test_forward_pass_shapes() {
        let model = DenseNet::densenet121(10);

        // Test different batch sizes
        for batch_size in [1, 4, 8] {
            let input = torsh_tensor::creation::randn(&[batch_size, 3, 224, 224]).unwrap();
            let output = model.forward(&input).unwrap();
            assert_eq!(output.shape(), &[batch_size, 10]);
        }
    }

    #[test]
    fn test_complexity_comparison() {
        let comparison = DenseNetFactory::complexity_comparison();
        assert!(comparison.contains("Growth Rate"));
        assert!(comparison.contains("Parameters"));
        assert!(comparison.contains("DenseNet-121"));
        assert!(comparison.contains("DenseNet-161"));
    }
}

/// Re-export commonly used types
pub use self::{
    DenseLayer,
    DenseBlock,
    TransitionLayer,
    DenseNet,
    DenseNetConfig,
    DenseNetFactory,
};