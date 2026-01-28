//! Video model implementations for ToRSh
//!
//! This module provides various video understanding architectures including
//! 3D ResNet, SlowFast Networks, and Video Transformers for processing
//! temporal video data.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::DeviceType;
use torsh_nn::prelude::{
    BatchNorm3d, Conv3d, Dropout, LayerNorm, Linear, MaxPool3d, MultiheadAttention,
};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// Configuration for 3D ResNet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNet3DConfig {
    pub input_channels: usize,
    pub num_classes: usize,
    pub layers: Vec<usize>, // Number of blocks in each layer
    pub initial_channels: usize,
    pub temporal_kernel_size: usize,
    pub temporal_stride: usize,
    pub spatial_stride: usize,
    pub dropout_rate: f64,
}

impl Default for ResNet3DConfig {
    fn default() -> Self {
        Self {
            input_channels: 3,
            num_classes: 400,         // Kinetics dataset
            layers: vec![2, 2, 2, 2], // ResNet-18 structure
            initial_channels: 64,
            temporal_kernel_size: 3,
            temporal_stride: 1,
            spatial_stride: 2,
            dropout_rate: 0.5,
        }
    }
}

/// 3D Convolutional Block for ResNet3D
pub struct BasicBlock3D {
    conv1: Conv3d,
    bn1: BatchNorm3d,
    conv2: Conv3d,
    bn2: BatchNorm3d,
    shortcut: Option<Conv3d>,
    shortcut_bn: Option<BatchNorm3d>,
    stride: usize,
}

impl BasicBlock3D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        _temporal_stride: usize,
    ) -> torsh_core::error::Result<Self> {
        let conv1 = Conv3d::new(
            in_channels,
            out_channels,
            (3, 3, 3),
            (stride, stride, stride),
            (1, 1, 1),
            (1, 1, 1),
            false,
            1,
        );
        let bn1 = BatchNorm3d::new(out_channels)?;
        let conv2 = Conv3d::new(
            out_channels,
            out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            false,
            1,
        );
        let bn2 = BatchNorm3d::new(out_channels)?;

        let (shortcut, shortcut_bn) = if stride != 1 || in_channels != out_channels {
            (
                Some(Conv3d::new(
                    in_channels,
                    out_channels,
                    (1, 1, 1),
                    (stride, stride, stride),
                    (0, 0, 0),
                    (1, 1, 1),
                    false,
                    1,
                )),
                Some(BatchNorm3d::new(out_channels)?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut,
            shortcut_bn,
            stride,
        })
    }

    pub fn forward(&self, x: &Tensor) -> torsh_core::error::Result<Tensor> {
        let identity = x.clone();

        let mut out = self.conv1.forward(x)?;
        out = self.bn1.forward(&out)?;
        out = out.relu()?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;

        let identity =
            if let (Some(shortcut), Some(shortcut_bn)) = (&self.shortcut, &self.shortcut_bn) {
                let identity = shortcut.forward(&identity)?;
                shortcut_bn.forward(&identity)?
            } else {
                identity
            };

        out = out.add(&identity)?;
        out.relu()
    }
}

impl Module for BasicBlock3D {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());

        if let Some(shortcut) = &self.shortcut {
            params.extend(shortcut.parameters());
        }
        if let Some(shortcut_bn) = &self.shortcut_bn {
            params.extend(shortcut_bn.parameters());
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.conv1.named_parameters() {
            params.insert(format!("conv1.{}", name), param);
        }
        for (name, param) in self.bn1.named_parameters() {
            params.insert(format!("bn1.{}", name), param);
        }
        for (name, param) in self.conv2.named_parameters() {
            params.insert(format!("conv2.{}", name), param);
        }
        for (name, param) in self.bn2.named_parameters() {
            params.insert(format!("bn2.{}", name), param);
        }

        if let Some(shortcut) = &self.shortcut {
            for (name, param) in shortcut.named_parameters() {
                params.insert(format!("shortcut.{}", name), param);
            }
        }
        if let Some(shortcut_bn) = &self.shortcut_bn {
            for (name, param) in shortcut_bn.named_parameters() {
                params.insert(format!("shortcut_bn.{}", name), param);
            }
        }

        params
    }

    fn training(&self) -> bool {
        self.bn1.training()
    }

    fn train(&mut self) {
        self.conv1.train();
        self.bn1.train();
        self.conv2.train();
        self.bn2.train();
        if let Some(shortcut) = &mut self.shortcut {
            shortcut.train();
        }
        if let Some(shortcut_bn) = &mut self.shortcut_bn {
            shortcut_bn.train();
        }
    }

    fn eval(&mut self) {
        self.conv1.eval();
        self.bn1.eval();
        self.conv2.eval();
        self.bn2.eval();
        if let Some(shortcut) = &mut self.shortcut {
            shortcut.eval();
        }
        if let Some(shortcut_bn) = &mut self.shortcut_bn {
            shortcut_bn.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        self.conv1.to_device(device)?;
        self.bn1.to_device(device)?;
        self.conv2.to_device(device)?;
        self.bn2.to_device(device)?;
        if let Some(shortcut) = &mut self.shortcut {
            shortcut.to_device(device)?;
        }
        if let Some(shortcut_bn) = &mut self.shortcut_bn {
            shortcut_bn.to_device(device)?;
        }
        Ok(())
    }
}

/// 3D ResNet for video classification
pub struct ResNet3D {
    conv1: Conv3d,
    bn1: BatchNorm3d,
    maxpool: MaxPool3d,
    layers: Vec<Vec<BasicBlock3D>>,
    avgpool: Parameter, // Placeholder for adaptive average pooling
    fc: Linear,
    dropout: Dropout,
    config: ResNet3DConfig,
}

impl ResNet3D {
    pub fn new(config: ResNet3DConfig) -> torsh_core::error::Result<Self> {
        // Initial convolution
        let conv1 = Conv3d::new(
            config.input_channels,
            config.initial_channels,
            (7, 7, 7),
            (
                config.spatial_stride,
                config.spatial_stride,
                config.spatial_stride,
            ),
            (3, 3, 3),
            (1, 1, 1),
            false,
            1,
        );
        let bn1 = BatchNorm3d::new(config.initial_channels)?;
        let maxpool = MaxPool3d::new((3, 3, 3), Some((2, 2, 2)), (1, 1, 1), (1, 1, 1), false);

        // Build residual layers
        let mut layers = Vec::new();
        let mut in_channels = config.initial_channels;
        let mut out_channels = config.initial_channels;

        for (layer_idx, &num_blocks) in config.layers.iter().enumerate() {
            let mut layer_blocks = Vec::new();
            let stride = if layer_idx > 0 { 2 } else { 1 };

            // First block (may have stride)
            layer_blocks.push(BasicBlock3D::new(
                in_channels,
                out_channels,
                stride,
                config.temporal_stride,
            )?);

            // Remaining blocks
            for _ in 1..num_blocks {
                layer_blocks.push(BasicBlock3D::new(out_channels, out_channels, 1, 1)?);
            }

            layers.push(layer_blocks);
            in_channels = out_channels;
            out_channels *= 2;
        }

        // Final fully connected layer
        let fc = Linear::new(in_channels, config.num_classes, true);

        Ok(Self {
            conv1,
            bn1,
            maxpool,
            layers,
            avgpool: Parameter::new(torsh_tensor::creation::zeros(&[1])?), // Placeholder
            fc,
            dropout: Dropout::new(config.dropout_rate as f32),
            config,
        })
    }
}

impl Module for ResNet3D {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let mut x = input.clone();

        // Initial convolution and pooling
        x = self.conv1.forward(&x)?;
        x = self.bn1.forward(&x)?;
        x = x.relu()?;
        x = self.maxpool.forward(&x)?;

        // Residual layers
        for layer in &self.layers {
            for block in layer {
                x = block.forward(&x)?;
            }
        }

        // Global average pooling
        x = x.mean(Some(&[2, 3, 4]), false)?; // Average over spatial and temporal dimensions

        // Dropout and classification
        x = self.dropout.forward(&x)?;
        x = self.fc.forward(&x)?;

        Ok(x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for (block_idx, block) in layer.iter().enumerate() {
                for (name, param) in block.parameters() {
                    params.insert(
                        format!("layer_{}.block_{}.{}", layer_idx, block_idx, name),
                        param,
                    );
                }
            }
        }

        params.extend(self.fc.parameters());
        params.extend(self.dropout.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        self.conv1.train();
        self.bn1.train();
        for layer in &mut self.layers {
            for block in layer {
                block.train();
            }
        }
        self.fc.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.conv1.eval();
        self.bn1.eval();
        for layer in &mut self.layers {
            for block in layer {
                block.eval();
            }
        }
        self.fc.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        self.conv1.to_device(device)?;
        self.bn1.to_device(device)?;
        for layer in &mut self.layers {
            for block in layer {
                block.to_device(device)?;
            }
        }
        self.fc.to_device(device)?;
        self.dropout.to_device(device)
    }
}

/// Configuration for SlowFast Networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowFastConfig {
    pub num_classes: usize,
    pub slow_channels: usize,
    pub fast_channels: usize,
    pub slow_temporal_stride: usize,
    pub fast_temporal_stride: usize,
    pub alpha: usize, // Temporal sampling rate ratio
    pub beta: usize,  // Channel capacity ratio
    pub fusion_kernel_size: usize,
    pub dropout_rate: f64,
}

impl Default for SlowFastConfig {
    fn default() -> Self {
        Self {
            num_classes: 400,
            slow_channels: 64,
            fast_channels: 8,
            slow_temporal_stride: 4,
            fast_temporal_stride: 1,
            alpha: 8,
            beta: 8,
            fusion_kernel_size: 7,
            dropout_rate: 0.5,
        }
    }
}

/// SlowFast Network for video understanding
pub struct SlowFast {
    slow_pathway: ResNet3D,
    fast_pathway: ResNet3D,
    lateral_connections: Vec<Conv3d>,
    fusion_layer: Linear,
    dropout: Dropout,
    config: SlowFastConfig,
}

impl SlowFast {
    pub fn new(config: SlowFastConfig) -> torsh_core::error::Result<Self> {
        // Slow pathway configuration
        let slow_config = ResNet3DConfig {
            input_channels: 3,
            num_classes: config.num_classes,
            layers: vec![2, 2, 2, 2],
            initial_channels: config.slow_channels,
            temporal_kernel_size: 1,
            temporal_stride: config.slow_temporal_stride,
            spatial_stride: 2,
            dropout_rate: config.dropout_rate,
        };

        // Fast pathway configuration (fewer channels)
        let fast_config = ResNet3DConfig {
            input_channels: 3,
            num_classes: config.num_classes,
            layers: vec![2, 2, 2, 2],
            initial_channels: config.fast_channels,
            temporal_kernel_size: 3,
            temporal_stride: config.fast_temporal_stride,
            spatial_stride: 2,
            dropout_rate: config.dropout_rate,
        };

        let slow_pathway = ResNet3D::new(slow_config)?;
        let fast_pathway = ResNet3D::new(fast_config)?;

        // Lateral connections for fusion
        let lateral_connections = vec![
            Conv3d::new(
                config.fast_channels,
                config.slow_channels / config.beta,
                (
                    config.fusion_kernel_size,
                    config.fusion_kernel_size,
                    config.fusion_kernel_size,
                ),
                (1, 1, 1),
                (
                    config.fusion_kernel_size / 2,
                    config.fusion_kernel_size / 2,
                    config.fusion_kernel_size / 2,
                ),
                (1, 1, 1),
                false,
                1,
            ),
            Conv3d::new(
                config.fast_channels * 2,
                config.slow_channels * 2 / config.beta,
                (
                    config.fusion_kernel_size,
                    config.fusion_kernel_size,
                    config.fusion_kernel_size,
                ),
                (1, 1, 1),
                (
                    config.fusion_kernel_size / 2,
                    config.fusion_kernel_size / 2,
                    config.fusion_kernel_size / 2,
                ),
                (1, 1, 1),
                false,
                1,
            ),
        ];

        // Fusion layer
        let fusion_input_dim = config.slow_channels + config.fast_channels;
        let fusion_layer = Linear::new(fusion_input_dim, config.num_classes, true);

        Ok(Self {
            slow_pathway,
            fast_pathway,
            lateral_connections,
            fusion_layer,
            dropout: Dropout::new(config.dropout_rate as f32),
            config,
        })
    }

    pub fn forward_dual_pathway(
        &self,
        slow_input: &Tensor,
        fast_input: &Tensor,
    ) -> torsh_core::error::Result<Tensor> {
        // Process through slow pathway
        let slow_features = self.slow_pathway.forward(slow_input)?;

        // Process through fast pathway
        let fast_features = self.fast_pathway.forward(fast_input)?;

        // Fusion (simplified - in practice would use lateral connections at multiple layers)
        let fused_features = Tensor::cat(&[&slow_features, &fast_features], 1)?;

        // Final classification
        let output = self.dropout.forward(&fused_features)?;
        let output = self.fusion_layer.forward(&output)?;

        Ok(output)
    }
}

impl Module for SlowFast {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        // For simplicity, split input into slow and fast pathways
        let slow_input = input.clone();
        let fast_input = input.clone();
        self.forward_dual_pathway(&slow_input, &fast_input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.slow_pathway.parameters() {
            params.insert(format!("slow_pathway.{}", name), param);
        }
        for (name, param) in self.fast_pathway.parameters() {
            params.insert(format!("fast_pathway.{}", name), param);
        }

        for (i, lateral) in self.lateral_connections.iter().enumerate() {
            for (name, param) in lateral.parameters() {
                params.insert(format!("lateral_{}.{}", i, name), param);
            }
        }

        params.extend(self.fusion_layer.parameters());
        params.extend(self.dropout.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        self.slow_pathway.train();
        self.fast_pathway.train();
        for lateral in &mut self.lateral_connections {
            lateral.train();
        }
        self.fusion_layer.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.slow_pathway.eval();
        self.fast_pathway.eval();
        for lateral in &mut self.lateral_connections {
            lateral.eval();
        }
        self.fusion_layer.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        self.slow_pathway.to_device(device)?;
        self.fast_pathway.to_device(device)?;
        for lateral in &mut self.lateral_connections {
            lateral.to_device(device)?;
        }
        self.fusion_layer.to_device(device)?;
        self.dropout.to_device(device)
    }
}

/// Configuration for Video Transformer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoTransformerConfig {
    pub input_dim: usize,
    pub num_classes: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_ratio: usize,
    pub num_frames: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub dropout_rate: f64,
    pub attention_dropout_rate: f64,
}

impl Default for VideoTransformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 3 * 16 * 16, // 3 channels, 16x16 patches
            num_classes: 400,
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_ratio: 4,
            num_frames: 16,
            patch_size: 16,
            temporal_patch_size: 2,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.1,
        }
    }
}

/// Video Transformer Block
pub struct VideoTransformerBlock {
    attention: MultiheadAttention,
    norm1: LayerNorm,
    mlp: Vec<Linear>,
    norm2: LayerNorm,
    dropout: Dropout,
}

impl VideoTransformerBlock {
    pub fn new(hidden_dim: usize, num_heads: usize, mlp_ratio: usize, dropout_rate: f64) -> Self {
        let attention = MultiheadAttention::new(hidden_dim, num_heads);
        let norm1 = LayerNorm::new(vec![hidden_dim], 1e-6, true, DeviceType::Cpu)
            .expect("failed to create VideoTransformerBlock norm1");
        let norm2 = LayerNorm::new(vec![hidden_dim], 1e-6, true, DeviceType::Cpu)
            .expect("failed to create VideoTransformerBlock norm2");

        let mlp_hidden_dim = hidden_dim * mlp_ratio;
        let mlp = vec![
            Linear::new(hidden_dim, mlp_hidden_dim, true),
            Linear::new(mlp_hidden_dim, hidden_dim, true),
        ];

        Self {
            attention,
            norm1,
            mlp,
            norm2,
            dropout: Dropout::new(dropout_rate as f32),
        }
    }

    pub fn forward(&self, x: &Tensor) -> torsh_core::error::Result<Tensor> {
        // Self-attention with residual connection
        let normed_x = self.norm1.forward(x)?;
        let attention_out = self.attention.forward(&normed_x)?;
        let x = x.add(&self.dropout.forward(&attention_out)?)?;

        // MLP with residual connection
        let normed_x = self.norm2.forward(&x)?;
        let mut mlp_out = normed_x;

        for (i, linear) in self.mlp.iter().enumerate() {
            mlp_out = linear.forward(&mlp_out)?;
            if i == 0 {
                mlp_out = mlp_out.gelu()?;
            }
        }

        mlp_out = self.dropout.forward(&mlp_out)?;
        x.add(&mlp_out)
    }
}

impl Module for VideoTransformerBlock {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.attention.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.dropout.parameters());

        for (i, linear) in self.mlp.iter().enumerate() {
            for (name, param) in linear.parameters() {
                params.insert(format!("mlp_{}.{}", i, name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.attention.named_parameters() {
            params.insert(format!("attention.{}", name), param);
        }
        for (name, param) in self.norm1.named_parameters() {
            params.insert(format!("norm1.{}", name), param);
        }
        for (name, param) in self.norm2.named_parameters() {
            params.insert(format!("norm2.{}", name), param);
        }
        for (name, param) in self.dropout.named_parameters() {
            params.insert(format!("dropout.{}", name), param);
        }

        for (i, linear) in self.mlp.iter().enumerate() {
            for (name, param) in linear.named_parameters() {
                params.insert(format!("mlp_{}.{}", i, name), param);
            }
        }

        params
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        self.attention.train();
        self.norm1.train();
        self.norm2.train();
        self.dropout.train();
        for linear in &mut self.mlp {
            linear.train();
        }
    }

    fn eval(&mut self) {
        self.attention.eval();
        self.norm1.eval();
        self.norm2.eval();
        self.dropout.eval();
        for linear in &mut self.mlp {
            linear.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        self.attention.to_device(device)?;
        self.norm1.to_device(device)?;
        self.norm2.to_device(device)?;
        self.dropout.to_device(device)?;
        for linear in &mut self.mlp {
            linear.to_device(device)?;
        }
        Ok(())
    }
}

/// Video Transformer for spatiotemporal understanding
pub struct VideoTransformer {
    patch_embedding: Linear,
    cls_token: Parameter,
    positional_embeddings: Parameter,
    temporal_embeddings: Parameter,
    blocks: Vec<VideoTransformerBlock>,
    norm: LayerNorm,
    head: Linear,
    dropout: Dropout,
    config: VideoTransformerConfig,
}

impl VideoTransformer {
    pub fn new(config: VideoTransformerConfig) -> Self {
        let patch_embedding = Linear::new(config.input_dim, config.hidden_dim, true);

        // Learnable parameters
        let cls_token = Parameter::new(
            torsh_tensor::creation::randn(&[1, 1, config.hidden_dim])
                .expect("failed to create VideoTransformer cls_token"),
        );

        let num_patches = (config.num_frames / config.temporal_patch_size)
            * ((224 / config.patch_size) * (224 / config.patch_size)); // Assuming 224x224 input
        let positional_embeddings = Parameter::new(
            torsh_tensor::creation::randn(&[1, num_patches + 1, config.hidden_dim])
                .expect("failed to create VideoTransformer positional embeddings"),
        );

        let temporal_embeddings = Parameter::new(
            torsh_tensor::creation::randn(&[
                1,
                config.num_frames / config.temporal_patch_size,
                config.hidden_dim,
            ])
            .expect("failed to create VideoTransformer temporal embeddings"),
        );

        // Transformer blocks
        let mut blocks = Vec::new();
        for _ in 0..config.num_layers {
            blocks.push(VideoTransformerBlock::new(
                config.hidden_dim,
                config.num_heads,
                config.mlp_ratio,
                config.dropout_rate,
            ));
        }

        let norm = LayerNorm::new(vec![config.hidden_dim], 1e-6, true, DeviceType::Cpu)
            .expect("failed to create VideoTransformer layer norm");
        let head = Linear::new(config.hidden_dim, config.num_classes, true);

        Self {
            patch_embedding,
            cls_token,
            positional_embeddings,
            temporal_embeddings,
            blocks,
            norm,
            head,
            dropout: Dropout::new(config.dropout_rate as f32),
            config,
        }
    }
}

impl Module for VideoTransformer {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let batch_size = input.size(0)?;

        // Patch embedding (simplified - would need proper 3D patch extraction)
        let mut x = self.patch_embedding.forward(input)?;

        // Add CLS token
        let cls_tokens = {
            let cls_token_tensor = self.cls_token.tensor();
            let cls_token_data = cls_token_tensor.read();
            cls_token_data.repeat(&[batch_size, 1, 1])?
        };
        x = Tensor::cat(&[&cls_tokens, &x], 1)?;

        // Add positional embeddings
        x = x.add(&self.positional_embeddings.tensor().read())?;

        // Apply dropout
        x = self.dropout.forward(&x)?;

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Final normalization
        x = self.norm.forward(&x)?;

        // Classification head (use CLS token)
        let cls_output = x.narrow(1, 0, 1)?.squeeze(1)?;
        let output = self.head.forward(&cls_output)?;

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.patch_embedding.parameters());
        params.insert("cls_token".to_string(), self.cls_token.clone());
        params.insert(
            "positional_embeddings".to_string(),
            self.positional_embeddings.clone(),
        );
        params.insert(
            "temporal_embeddings".to_string(),
            self.temporal_embeddings.clone(),
        );

        for (i, block) in self.blocks.iter().enumerate() {
            for (name, param) in block.parameters() {
                params.insert(format!("block_{}.{}", i, name), param);
            }
        }

        params.extend(self.norm.parameters());
        params.extend(self.head.parameters());
        params.extend(self.dropout.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        self.patch_embedding.train();
        for block in &mut self.blocks {
            block.train();
        }
        self.norm.train();
        self.head.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.patch_embedding.eval();
        for block in &mut self.blocks {
            block.eval();
        }
        self.norm.eval();
        self.head.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        self.patch_embedding.to_device(device)?;
        self.cls_token.to_device(device)?;
        self.positional_embeddings.to_device(device)?;
        self.temporal_embeddings.to_device(device)?;

        for block in &mut self.blocks {
            block.to_device(device)?;
        }

        self.norm.to_device(device)?;
        self.head.to_device(device)?;
        self.dropout.to_device(device)
    }
}

/// Video Architecture enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VideoArchitecture {
    ResNet3D,
    SlowFast,
    VideoTransformer,
}

/// Unified Video model type
pub enum VideoModel {
    ResNet3D(ResNet3D),
    SlowFast(SlowFast),
    VideoTransformer(VideoTransformer),
}

impl Module for VideoModel {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        match self {
            VideoModel::ResNet3D(model) => model.forward(input),
            VideoModel::SlowFast(model) => model.forward(input),
            VideoModel::VideoTransformer(model) => model.forward(input),
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        match self {
            VideoModel::ResNet3D(model) => model.parameters(),
            VideoModel::SlowFast(model) => model.parameters(),
            VideoModel::VideoTransformer(model) => model.parameters(),
        }
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        match self {
            VideoModel::ResNet3D(model) => model.named_parameters(),
            VideoModel::SlowFast(model) => model.named_parameters(),
            VideoModel::VideoTransformer(model) => model.named_parameters(),
        }
    }

    fn training(&self) -> bool {
        match self {
            VideoModel::ResNet3D(model) => model.training(),
            VideoModel::SlowFast(model) => model.training(),
            VideoModel::VideoTransformer(model) => model.training(),
        }
    }

    fn train(&mut self) {
        match self {
            VideoModel::ResNet3D(model) => model.train(),
            VideoModel::SlowFast(model) => model.train(),
            VideoModel::VideoTransformer(model) => model.train(),
        }
    }

    fn eval(&mut self) {
        match self {
            VideoModel::ResNet3D(model) => model.eval(),
            VideoModel::SlowFast(model) => model.eval(),
            VideoModel::VideoTransformer(model) => model.eval(),
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        match self {
            VideoModel::ResNet3D(model) => model.to_device(device),
            VideoModel::SlowFast(model) => model.to_device(device),
            VideoModel::VideoTransformer(model) => model.to_device(device),
        }
    }
}
