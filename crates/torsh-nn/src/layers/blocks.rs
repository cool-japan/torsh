//! Pre-built neural network blocks
//!
//! This module contains common building blocks used in neural network architectures,
//! such as ResNet blocks, DenseNet blocks, and other reusable components.

use crate::container::Sequential;
use crate::layers::{BatchNorm2d, Conv2d, Dropout, Linear, ReLU};
use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// ResNet Basic Block
///
/// Implements the basic residual block from "Deep Residual Learning for Image Recognition"
/// with two 3x3 convolutions and a skip connection.
pub struct BasicBlock {
    base: ModuleBase,
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<Sequential>,
    #[allow(dead_code)]
    stride: usize,
}

impl BasicBlock {
    /// Create a new basic residual block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        #[allow(dead_code)] stride: usize,
        downsample: Option<Sequential>,
    ) -> Result<Self> {
        Ok(Self {
            base: ModuleBase::new(),
            conv1: Conv2d::new(
                in_channels,
                out_channels,
                (3, 3),
                (stride, stride),
                (1, 1),
                (1, 1),
                false,
                1,
            ),
            bn1: BatchNorm2d::new(out_channels)?,
            relu: ReLU::new(),
            conv2: Conv2d::new(
                out_channels,
                out_channels,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                false,
                1,
            ),
            bn2: BatchNorm2d::new(out_channels)?,
            downsample,
            stride,
        })
    }

    /// Create a basic block with downsampling for dimension matching
    pub fn with_downsample(
        in_channels: usize,
        out_channels: usize,
        #[allow(dead_code)] stride: usize,
    ) -> Result<Self> {
        let downsample = if stride != 1 || in_channels != out_channels {
            Some(
                Sequential::new()
                    .add(Conv2d::new(
                        in_channels,
                        out_channels,
                        (1, 1),
                        (stride, stride),
                        (0, 0),
                        (1, 1),
                        false,
                        1,
                    ))
                    .add(BatchNorm2d::new(out_channels)?),
            )
        } else {
            None
        };

        Ok(Self::new(in_channels, out_channels, stride, downsample)?)
    }
}

impl Module for BasicBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let identity = input.clone();

        let mut out = self.conv1.forward(input)?;
        out = self.bn1.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;

        let residual = if let Some(ref downsample) = self.downsample {
            downsample.forward(&identity)?
        } else {
            identity
        };

        out = out.add_op(&residual)?;
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

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.base.set_training(true);
        // Note: Individual layer training mode would be set here in a full implementation
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        // Note: Individual layer evaluation mode would be set here in a full implementation
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        // Note: Individual layer device transfer would be done here in a full implementation
        Ok(())
    }
}

/// ResNet Bottleneck Block
///
/// Implements the bottleneck residual block with 1x1, 3x3, 1x1 convolutions
/// for deeper ResNet architectures (ResNet-50, ResNet-101, ResNet-152).
pub struct BottleneckBlock {
    base: ModuleBase,
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    relu: ReLU,
    downsample: Option<Sequential>,
    #[allow(dead_code)]
    stride: usize,
}

impl BottleneckBlock {
    /// Create a new bottleneck residual block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        #[allow(dead_code)] stride: usize,
        downsample: Option<Sequential>,
    ) -> Result<Self> {
        let mid_channels = out_channels / 4;

        Ok(Self {
            base: ModuleBase::new(),
            conv1: Conv2d::new(
                in_channels,
                mid_channels,
                (1, 1),
                (1, 1),
                (0, 0),
                (1, 1),
                false,
                1,
            ),
            bn1: BatchNorm2d::new(mid_channels)?,
            conv2: Conv2d::new(
                mid_channels,
                mid_channels,
                (3, 3),
                (stride, stride),
                (1, 1),
                (1, 1),
                false,
                1,
            ),
            bn2: BatchNorm2d::new(mid_channels)?,
            conv3: Conv2d::new(
                mid_channels,
                out_channels,
                (1, 1),
                (1, 1),
                (0, 0),
                (1, 1),
                false,
                1,
            ),
            bn3: BatchNorm2d::new(out_channels)?,
            relu: ReLU::new(),
            downsample,
            stride,
        })
    }

    /// Create a bottleneck block with downsampling
    pub fn with_downsample(
        in_channels: usize,
        out_channels: usize,
        #[allow(dead_code)] stride: usize,
    ) -> Result<Self> {
        let downsample = if stride != 1 || in_channels != out_channels {
            Some(
                Sequential::new()
                    .add(Conv2d::new(
                        in_channels,
                        out_channels,
                        (1, 1),
                        (stride, stride),
                        (0, 0),
                        (1, 1),
                        false,
                        1,
                    ))
                    .add(BatchNorm2d::new(out_channels)?),
            )
        } else {
            None
        };

        Ok(Self::new(in_channels, out_channels, stride, downsample)?)
    }
}

impl Module for BottleneckBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let identity = input.clone();

        let mut out = self.conv1.forward(input)?;
        out = self.bn1.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv3.forward(&out)?;
        out = self.bn3.forward(&out)?;

        let residual = if let Some(ref downsample) = self.downsample {
            downsample.forward(&identity)?
        } else {
            identity
        };

        out = out.add_op(&residual)?;
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

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

/// DenseNet Dense Block
///
/// Implements the dense block from "Densely Connected Convolutional Networks"
/// where each layer receives feature maps from all preceding layers.
pub struct DenseBlock {
    base: ModuleBase,
    layers: Vec<DenseLayer>,
    #[allow(dead_code)]
    num_layers: usize,
}

impl DenseBlock {
    /// Create a new dense block
    pub fn new(
        num_layers: usize,
        num_input_features: usize,
        growth_rate: usize,
        bn_size: usize,
        drop_rate: f32,
    ) -> Result<Self> {
        let mut layers = Vec::new();

        for i in 0..num_layers {
            let layer = DenseLayer::new(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
            )?;
            layers.push(layer);
        }

        Ok(Self {
            base: ModuleBase::new(),
            layers,
            num_layers,
        })
    }

    /// Manual concatenation along channel dimension for NCHW tensors
    fn manual_concat_features(&self, features: &[Tensor]) -> Result<Tensor> {
        if features.is_empty() {
            return Err(TorshError::InvalidArgument("No features to concatenate".to_string()));
        }

        if features.len() == 1 {
            return Ok(features[0].clone());
        }

        // Calculate total channels
        let mut total_channels = 0;
        for feature in features {
            total_channels += feature.shape().dims()[1];
        }

        let batch_size = features[0].shape().dims()[0];
        let height = features[0].shape().dims()[2];
        let width = features[0].shape().dims()[3];

        // Create output data by concatenating along channel dimension
        let mut output_data = Vec::new();

        for b in 0..batch_size {
            for feature in features {
                let feature_data = feature.to_vec()?;
                let channels = feature.shape().dims()[1];

                for c in 0..channels {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = b * (channels * height * width) + c * (height * width) + h * width + w;
                            if idx < feature_data.len() {
                                output_data.push(feature_data[idx]);
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_vec(output_data, &[batch_size, total_channels, height, width])
    }
}

impl Module for DenseBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut features = vec![input.clone()];

        for layer in &self.layers {
            // Concatenate all previous feature maps
            let concat_features = self.manual_concat_features(&features)?;

            let new_features = layer.forward(&concat_features)?;
            features.push(new_features);
        }

        // Return concatenation of all features
        self.manual_concat_features(&features)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer{}.{}", i, name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

/// Dense Layer for DenseNet
///
/// A single dense layer consisting of BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3) with dropout.
pub struct DenseLayer {
    base: ModuleBase,
    norm1: BatchNorm2d,
    relu1: ReLU,
    conv1: Conv2d,
    norm2: BatchNorm2d,
    relu2: ReLU,
    conv2: Conv2d,
    dropout: Option<Dropout>,
}

impl DenseLayer {
    /// Create a new dense layer
    pub fn new(
        num_input_features: usize,
        growth_rate: usize,
        bn_size: usize,
        drop_rate: f32,
    ) -> Result<Self> {
        let inter_features = bn_size * growth_rate;

        Ok(Self {
            base: ModuleBase::new(),
            norm1: BatchNorm2d::new(num_input_features)?,
            relu1: ReLU::new(),
            conv1: Conv2d::new(
                num_input_features,
                inter_features,
                (1, 1),
                (1, 1),
                (0, 0),
                (1, 1),
                false,
                1,
            ),
            norm2: BatchNorm2d::new(inter_features)?,
            relu2: ReLU::new(),
            conv2: Conv2d::new(
                inter_features,
                growth_rate,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                false,
                1,
            ),
            dropout: if drop_rate > 0.0 {
                Some(Dropout::new(drop_rate))
            } else {
                None
            },
        })
    }
}

impl Module for DenseLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut out = self.norm1.forward(input)?;
        out = self.relu1.forward(&out)?;
        out = self.conv1.forward(&out)?;
        out = self.norm2.forward(&out)?;
        out = self.relu2.forward(&out)?;
        out = self.conv2.forward(&out)?;

        if let Some(ref dropout) = self.dropout {
            out = dropout.forward(&out)?;
        }

        Ok(out)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.norm1.parameters() {
            params.insert(format!("norm1.{}", name), param);
        }
        for (name, param) in self.conv1.parameters() {
            params.insert(format!("conv1.{}", name), param);
        }
        for (name, param) in self.norm2.parameters() {
            params.insert(format!("norm2.{}", name), param);
        }
        for (name, param) in self.conv2.parameters() {
            params.insert(format!("conv2.{}", name), param);
        }

        if let Some(ref dropout) = self.dropout {
            for (name, param) in dropout.parameters() {
                params.insert(format!("dropout.{}", name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

/// Transition Layer for DenseNet
///
/// Reduces the number of feature maps between dense blocks using 1x1 conv and pooling.
pub struct TransitionLayer {
    base: ModuleBase,
    norm: BatchNorm2d,
    relu: ReLU,
    conv: Conv2d,
    pool: crate::layers::pooling::AvgPool2d,
}

impl TransitionLayer {
    /// Create a new transition layer
    pub fn new(num_input_features: usize, num_output_features: usize) -> Result<Self> {
        Ok(Self {
            base: ModuleBase::new(),
            norm: BatchNorm2d::new(num_input_features)?,
            relu: ReLU::new(),
            conv: Conv2d::new(
                num_input_features,
                num_output_features,
                (1, 1),
                (1, 1),
                (0, 0),
                (1, 1),
                false,
                1,
            ),
            pool: crate::layers::pooling::AvgPool2d::new(
                (2, 2),
                Some((2, 2)),
                (0, 0),
                false,
                false,
            ),
        })
    }
}

impl Module for TransitionLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut out = self.norm.forward(input)?;
        out = self.relu.forward(&out)?;
        out = self.conv.forward(&out)?;
        out = self.pool.forward(&out)?;
        Ok(out)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.norm.parameters() {
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

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

/// Squeeze-and-Excitation Block
///
/// Implements the SE block from "Squeeze-and-Excitation Networks"
/// for adaptive channel-wise feature recalibration.
pub struct SEBlock {
    base: ModuleBase,
    global_pool: crate::layers::pooling::AdaptiveAvgPool2d,
    fc1: Linear,
    relu: ReLU,
    fc2: Linear,
    sigmoid: crate::layers::activation::Sigmoid,
    #[allow(dead_code)]
    reduction: usize,
}

impl SEBlock {
    /// Create a new SE block
    pub fn new(channels: usize, reduction: usize) -> Result<Self> {
        let reduced_channels = channels / reduction;

        Ok(Self {
            base: ModuleBase::new(),
            global_pool: crate::layers::pooling::AdaptiveAvgPool2d::new((Some(1), Some(1))),
            fc1: Linear::new(channels, reduced_channels, true),
            relu: ReLU::new(),
            fc2: Linear::new(reduced_channels, channels, true),
            sigmoid: crate::layers::activation::Sigmoid::new(),
            reduction,
        })
    }
}

impl Module for SEBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let [batch_size, channels, _height, _width] = shape.dims() else {
            return Err(TorshError::InvalidArgument(
                "Expected 4D input tensor".to_string(),
            ));
        };

        // Global average pooling
        let mut se = self.global_pool.forward(input)?;

        // Reshape to (batch_size, channels)
        se = se.reshape(&[*batch_size as i32, *channels as i32])?;

        // Excitation
        se = self.fc1.forward(&se)?;
        se = self.relu.forward(&se)?;
        se = self.fc2.forward(&se)?;
        se = self.sigmoid.forward(&se)?;

        // Debug: Check actual shape after sigmoid
        let se_shape = se.shape();

        // If tensor is flattened to 1D, we need to add batch dimension back
        let se = if se_shape.dims().len() == 1 && se_shape.dims()[0] == *channels {
            // Tensor lost batch dimension, add it back
            se.reshape(&[1i32, *channels as i32])?
        } else if se_shape.dims() == &[*batch_size, *channels] {
            // Shape is correct
            se
        } else {
            // Try to reshape to expected 2D shape
            se.reshape(&[*batch_size as i32, *channels as i32])?
        };

        // Reshape back to (batch_size, channels, 1, 1)
        let se = se.reshape(&[*batch_size as i32, *channels as i32, 1, 1])?;

        // Scale the input
        let output = input.mul_op(&se)?;
        Ok(output)
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

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

/// Mobile Inverted Bottleneck Block (MBConv)
///
/// Implements the inverted residual block from MobileNetV2 and EfficientNet
/// with depthwise separable convolutions.
pub struct MBConvBlock {
    base: ModuleBase,
    expand_conv: Option<Conv2d>,
    expand_bn: Option<BatchNorm2d>,
    depthwise_conv: Conv2d,
    depthwise_bn: BatchNorm2d,
    se_block: Option<SEBlock>,
    project_conv: Conv2d,
    project_bn: BatchNorm2d,
    relu: ReLU,
    use_residual: bool,
    drop_rate: f32,
}

impl MBConvBlock {
    /// Create a new MBConv block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        #[allow(dead_code)] stride: usize,
        expand_ratio: usize,
        se_ratio: Option<f32>,
        drop_rate: f32,
    ) -> Result<Self> {
        let expanded_channels = in_channels * expand_ratio;
        let use_residual = stride == 1 && in_channels == out_channels;

        // Expansion phase (1x1 conv)
        let (expand_conv, expand_bn) = if expand_ratio != 1 {
            let expand_bn = BatchNorm2d::new(expanded_channels)?;
            (
                Some(Conv2d::new(
                    in_channels,
                    expanded_channels,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    false,
                    1,
                )),
                Some(expand_bn),
            )
        } else {
            (None, None)
        };

        // Depthwise conv
        let depthwise_conv = Conv2d::new(
            expanded_channels,
            expanded_channels,
            (kernel_size, kernel_size),
            (stride, stride),
            (kernel_size / 2, kernel_size / 2),
            (1, 1),
            false,
            expanded_channels, // groups = in_channels for depthwise
        );

        // SE block
        let se_block = if let Some(ratio) = se_ratio {
            let se_channels = (expanded_channels as f32 * ratio).max(1.0) as usize;
            let reduction = expanded_channels / se_channels.max(1);
            Some(SEBlock::new(expanded_channels, reduction)?)
        } else {
            None
        };

        // Projection phase (1x1 conv)
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

        Ok(Self {
            base: ModuleBase::new(),
            expand_conv,
            expand_bn,
            depthwise_conv,
            depthwise_bn: BatchNorm2d::new(expanded_channels)?,
            se_block,
            project_conv,
            project_bn: BatchNorm2d::new(out_channels)?,
            relu: ReLU::new(),
            use_residual,
            drop_rate,
        })
    }
}

impl Module for MBConvBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        // Expansion phase
        if let (Some(ref expand_conv), Some(ref expand_bn)) = (&self.expand_conv, &self.expand_bn) {
            x = expand_conv.forward(&x)?;
            x = expand_bn.forward(&x)?;
            x = self.relu.forward(&x)?;
        }

        // Depthwise conv
        x = self.depthwise_conv.forward(&x)?;
        x = self.depthwise_bn.forward(&x)?;
        x = self.relu.forward(&x)?;

        // SE block
        if let Some(ref se_block) = self.se_block {
            x = se_block.forward(&x)?;
        }

        // Projection phase
        x = self.project_conv.forward(&x)?;
        x = self.project_bn.forward(&x)?;

        // Residual connection
        if self.use_residual {
            // Apply dropout if specified
            if self.drop_rate > 0.0 && self.training() {
                // Simple dropout implementation - in practice would use proper dropout layer
                // For now, just add the residual connection
                x = x.add_op(input)?;
            } else {
                x = x.add_op(input)?;
            }
        }

        Ok(x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if let Some(ref expand_conv) = self.expand_conv {
            for (name, param) in expand_conv.parameters() {
                params.insert(format!("expand_conv.{}", name), param);
            }
        }
        if let Some(ref expand_bn) = self.expand_bn {
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

        if let Some(ref se_block) = self.se_block {
            for (name, param) in se_block.parameters() {
                params.insert(format!("se_block.{}", name), param);
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

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_basic_block() -> Result<()> {
        let block = BasicBlock::with_downsample(64, 128, 2)?;
        let input = randn::<f32>(&[2, 64, 32, 32])?;
        let output = block.forward(&input)?;

        // Should downsample spatial dimensions and change channels
        assert_eq!(output.shape().dims(), &[2, 128, 16, 16]);

        Ok(())
    }

    #[test]
    fn test_bottleneck_block() -> Result<()> {
        let block = BottleneckBlock::with_downsample(256, 1024, 2)?;
        let input = randn::<f32>(&[1, 256, 56, 56])?;
        let output = block.forward(&input)?;

        // Should downsample and change channels
        assert_eq!(output.shape().dims(), &[1, 1024, 28, 28]);

        Ok(())
    }

    #[test]
    fn test_dense_block() -> Result<()> {
        let block = DenseBlock::new(4, 64, 32, 4, 0.0)?;
        let input = randn::<f32>(&[1, 64, 16, 16])?;
        let output = block.forward(&input)?;

        // Should concatenate features: 64 + 4*32 = 192 channels
        assert_eq!(output.shape().dims(), &[1, 192, 16, 16]);

        Ok(())
    }

    #[test]
    fn test_se_block() -> Result<()> {
        let block = SEBlock::new(256, 16)?;
        let input = randn::<f32>(&[2, 256, 14, 14])?;
        let output = block.forward(&input)?;

        // Should maintain same shape
        assert_eq!(output.shape().dims(), &[2, 256, 14, 14]);

        Ok(())
    }

    #[test]
    fn test_mbconv_block() -> Result<()> {
        let block = MBConvBlock::new(32, 64, 3, 2, 6, Some(0.25), 0.1)?;
        let input = randn::<f32>(&[1, 32, 112, 112])?;
        let output = block.forward(&input)?;

        // Should downsample and change channels
        assert_eq!(output.shape().dims(), &[1, 64, 56, 56]);

        Ok(())
    }

    #[test]
    fn test_transition_layer() -> Result<()> {
        let layer = TransitionLayer::new(128, 64)?;
        let input = randn::<f32>(&[1, 128, 32, 32])?;
        let output = layer.forward(&input)?;

        // Should reduce channels and spatial dimensions
        assert_eq!(output.shape().dims(), &[1, 64, 16, 16]);

        Ok(())
    }
}
