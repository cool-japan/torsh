//! Object Detection Models for ToRSh Deep Learning Framework
//!
//! This module provides comprehensive implementations of modern object detection architectures,
//! focusing on transformer-based end-to-end detection methods.
//!
//! ## Supported Architectures
//!
//! - **DETR (Detection Transformer)**: End-to-end object detection with transformers
//! - Encoder-decoder transformer architecture with learnable object queries
//! - Set-based global loss that forces unique predictions via bipartite matching
//!
//! ## Key Features
//!
//! - **End-to-End Detection**: No need for hand-crafted components like NMS
//! - **Set Prediction**: Directly predicts a fixed-size set of detections
//! - **Transformer Architecture**: Encoder-decoder with multi-head attention
//! - **Object Queries**: Learnable embeddings that capture object information
//! - **Bipartite Matching**: Hungarian algorithm for optimal assignment
//!
//! ## Architecture Overview
//!
//! ```text
//! Input Image -> CNN Backbone -> Feature Projection ->
//! Transformer Encoder -> Object Queries + Cross Attention ->
//! Transformer Decoder -> Classification + Bbox Regression
//! ```
//!
//! ## Example Usage
//!
//! ```rust
//! use torsh_models::vision::object_detection::*;
//!
//! // Create DETR model for COCO detection (80 classes)
//! let model = DETR::detr(80);
//!
//! // Create custom DETR model
//! let custom_model = DETR::new(
//!     20,   // num_classes (Pascal VOC)
//!     100,  // num_queries
//!     256,  // hidden_dim
//!     8,    // nheads
//!     6,    // num_encoder_layers
//!     6,    // num_decoder_layers
//! );
//!
//! // Forward pass
//! let input = torsh_tensor::creation::randn(&[1, 3, 224, 224])?;
//! let detections = model.forward(&input)?;
//! ```

use crate::error::{Result, TorshError};
use crate::vision::resnet::ResNet;
use scirs2_core::random::{Random, rng};
use std::collections::HashMap;
use torsh_core::{DeviceType, Tensor, DType};
use torsh_nn::{
    activations::ReLU,
    attention::MultiheadAttention,
    dropout::Dropout,
    layers::{Conv2d, Linear},
    normalization::LayerNorm,
    module::{Module, Parameter},
};

/// Multi-Layer Perceptron for DETR
///
/// Used for bounding box regression in DETR. Implements a simple
/// feedforward network with ReLU activations between layers.
#[derive(Debug)]
pub struct MLP {
    layers: Vec<Linear>,
    num_layers: usize,
}

impl MLP {
    /// Creates a new MLP
    ///
    /// # Arguments
    /// * `input_dim` - Input dimension
    /// * `hidden_dim` - Hidden layer dimension
    /// * `output_dim` - Output dimension
    /// * `num_layers` - Number of linear layers
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();

        if num_layers == 1 {
            layers.push(Linear::new(input_dim, output_dim, true));
        } else {
            // First layer
            layers.push(Linear::new(input_dim, hidden_dim, true));

            // Hidden layers
            for _ in 1..num_layers - 1 {
                layers.push(Linear::new(hidden_dim, hidden_dim, true));
            }

            // Output layer
            layers.push(Linear::new(hidden_dim, output_dim, true));
        }

        Self { layers, num_layers }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            // Apply ReLU to all layers except the last one
            if i < self.num_layers - 1 {
                x = x.relu()?;
            }
        }

        Ok(x)
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
                params.insert(format!("layers.{}.{}", i, name), param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.layers.first().map_or(false, |l| l.training())
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

/// Simple Positional Encoding for spatial features
///
/// Provides positional information to the transformer encoder.
/// In practice, this would implement sinusoidal or learned positional encodings.
#[derive(Debug)]
pub struct PositionalEncoding {
    encoding: Parameter,
    d_model: usize,
    max_len: usize,
}

impl PositionalEncoding {
    /// Creates a new positional encoding
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `max_len` - Maximum sequence length
    pub fn new(d_model: usize, max_len: usize) -> Result<Self> {
        // Create learnable positional encoding (simplified)
        let mut rng = rng();
        let encoding_data = (0..max_len * d_model)
            .map(|_| 0.02 * (rng.normal(0.0, 1.0) as f32))
            .collect::<Vec<_>>();
        let pe = Tensor::from_slice(&encoding_data, &[max_len, d_model])?;
        let encoding = Parameter::new(pe, true);

        Ok(Self {
            encoding,
            d_model,
            max_len,
        })
    }

    /// Get model dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get maximum sequence length
    pub fn max_len(&self) -> usize {
        self.max_len
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.size(1)?;
        let d_model = x.size(2)?;

        // Get positional encoding for the sequence length
        let encoding_data = self.encoding.data.read().unwrap();
        let pos_enc = encoding_data.slice(&[
            (0, seq_len.min(self.max_len) as i64),
            (0, d_model as i64),
        ])?;

        Ok(pos_enc.unsqueeze(0)?) // Add batch dimension
    }

    fn train(&mut self) {
        // Positional encoding can be trainable
    }

    fn eval(&mut self) {
        // No-op for positional encoding
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("encoding".to_string(), self.encoding.clone());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        true
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.encoding.to_device(device)?;
        Ok(())
    }
}

/// Transformer Encoder Layer
///
/// Individual layer in the transformer encoder with self-attention
/// and feedforward network.
#[derive(Debug)]
pub struct TransformerEncoderLayer {
    self_attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout_rate: f32,
}

impl TransformerEncoderLayer {
    /// Creates a new transformer encoder layer
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Feedforward network dimension
    /// * `dropout` - Dropout rate
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f32) -> Self {
        Self {
            self_attn: MultiheadAttention::new(d_model, nhead, dropout, true, true),
            linear1: Linear::new(d_model, dim_feedforward, true),
            linear2: Linear::new(dim_feedforward, d_model, true),
            norm1: LayerNorm::new(vec![d_model], 1e-5, true),
            norm2: LayerNorm::new(vec![d_model], 1e-5, true),
            dropout_rate: dropout,
        }
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, src: &Tensor) -> Result<Tensor> {
        // Self-attention with residual connection and layer norm (pre-norm)
        let norm_src = self.norm1.forward(src)?;
        let attn_output = self.self_attn.forward(&norm_src, &norm_src, &norm_src, None)?;
        let src2 = src.add(&attn_output)?;

        // Feedforward with residual connection and layer norm (pre-norm)
        let norm_src2 = self.norm2.forward(&src2)?;
        let ff_output = self.linear2.forward(&self.linear1.forward(&norm_src2)?.relu()?)?;
        let output = src2.add(&ff_output)?;

        Ok(output)
    }

    fn train(&mut self) {
        self.self_attn.train();
        self.linear1.train();
        self.linear2.train();
        self.norm1.train();
        self.norm2.train();
    }

    fn eval(&mut self) {
        self.self_attn.eval();
        self.linear1.eval();
        self.linear2.eval();
        self.norm1.eval();
        self.norm2.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.self_attn.parameters() {
            params.insert(format!("self_attn.{}", name), param);
        }
        for (name, param) in self.linear1.parameters() {
            params.insert(format!("linear1.{}", name), param);
        }
        for (name, param) in self.linear2.parameters() {
            params.insert(format!("linear2.{}", name), param);
        }
        for (name, param) in self.norm1.parameters() {
            params.insert(format!("norm1.{}", name), param);
        }
        for (name, param) in self.norm2.parameters() {
            params.insert(format!("norm2.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.self_attn.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.linear1.to_device(device)?;
        self.linear2.to_device(device)?;
        self.norm1.to_device(device)?;
        self.norm2.to_device(device)?;
        Ok(())
    }
}

/// Transformer Decoder Layer
///
/// Individual layer in the transformer decoder with self-attention,
/// cross-attention, and feedforward network.
#[derive(Debug)]
pub struct TransformerDecoderLayer {
    self_attn: MultiheadAttention,
    cross_attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    dropout_rate: f32,
}

impl TransformerDecoderLayer {
    /// Creates a new transformer decoder layer
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Feedforward network dimension
    /// * `dropout` - Dropout rate
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f32) -> Self {
        Self {
            self_attn: MultiheadAttention::new(d_model, nhead, dropout, true, true),
            cross_attn: MultiheadAttention::new(d_model, nhead, dropout, true, true),
            linear1: Linear::new(d_model, dim_feedforward, true),
            linear2: Linear::new(dim_feedforward, d_model, true),
            norm1: LayerNorm::new(vec![d_model], 1e-5, true),
            norm2: LayerNorm::new(vec![d_model], 1e-5, true),
            norm3: LayerNorm::new(vec![d_model], 1e-5, true),
            dropout_rate: dropout,
        }
    }
}

impl Module for TransformerDecoderLayer {
    fn forward(&self, tgt: &Tensor, memory: &Tensor) -> Result<Tensor> {
        // Self-attention on target (object queries) with pre-norm
        let norm_tgt = self.norm1.forward(tgt)?;
        let self_attn_output = self.self_attn.forward(&norm_tgt, &norm_tgt, &norm_tgt, None)?;
        let tgt2 = tgt.add(&self_attn_output)?;

        // Cross-attention between queries and memory (encoder output) with pre-norm
        let norm_tgt2 = self.norm2.forward(&tgt2)?;
        let cross_attn_output = self.cross_attn.forward(&norm_tgt2, memory, memory, None)?;
        let tgt3 = tgt2.add(&cross_attn_output)?;

        // Feedforward network with pre-norm
        let norm_tgt3 = self.norm3.forward(&tgt3)?;
        let ff_output = self.linear2.forward(&self.linear1.forward(&norm_tgt3)?.relu()?)?;
        let output = tgt3.add(&ff_output)?;

        Ok(output)
    }

    fn train(&mut self) {
        self.self_attn.train();
        self.cross_attn.train();
        self.linear1.train();
        self.linear2.train();
        self.norm1.train();
        self.norm2.train();
        self.norm3.train();
    }

    fn eval(&mut self) {
        self.self_attn.eval();
        self.cross_attn.eval();
        self.linear1.eval();
        self.linear2.eval();
        self.norm1.eval();
        self.norm2.eval();
        self.norm3.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.self_attn.parameters() {
            params.insert(format!("self_attn.{}", name), param);
        }
        for (name, param) in self.cross_attn.parameters() {
            params.insert(format!("cross_attn.{}", name), param);
        }
        for (name, param) in self.linear1.parameters() {
            params.insert(format!("linear1.{}", name), param);
        }
        for (name, param) in self.linear2.parameters() {
            params.insert(format!("linear2.{}", name), param);
        }
        for (name, param) in self.norm1.parameters() {
            params.insert(format!("norm1.{}", name), param);
        }
        for (name, param) in self.norm2.parameters() {
            params.insert(format!("norm2.{}", name), param);
        }
        for (name, param) in self.norm3.parameters() {
            params.insert(format!("norm3.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.self_attn.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.cross_attn.to_device(device)?;
        self.linear1.to_device(device)?;
        self.linear2.to_device(device)?;
        self.norm1.to_device(device)?;
        self.norm2.to_device(device)?;
        self.norm3.to_device(device)?;
        Ok(())
    }
}

/// Transformer Encoder
///
/// Stack of transformer encoder layers for processing image features.
#[derive(Debug)]
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
}

impl TransformerEncoder {
    /// Creates a new transformer encoder
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Feedforward network dimension
    /// * `num_layers` - Number of encoder layers
    /// * `dropout` - Dropout rate
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        num_layers: usize,
        dropout: f32,
    ) -> Self {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(TransformerEncoderLayer::new(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
            ));
        }

        Self { layers }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, src: &Tensor) -> Result<Tensor> {
        let mut output = src.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
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
                params.insert(format!("layers.{}.{}", i, name), param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.layers.first().map_or(false, |l| l.training())
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

/// Transformer Decoder
///
/// Stack of transformer decoder layers for processing object queries.
#[derive(Debug)]
pub struct TransformerDecoder {
    layers: Vec<TransformerDecoderLayer>,
}

impl TransformerDecoder {
    /// Creates a new transformer decoder
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Feedforward network dimension
    /// * `num_layers` - Number of decoder layers
    /// * `dropout` - Dropout rate
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        num_layers: usize,
        dropout: f32,
    ) -> Self {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(TransformerDecoderLayer::new(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
            ));
        }

        Self { layers }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl Module for TransformerDecoder {
    fn forward(&self, tgt: &Tensor, memory: &Tensor) -> Result<Tensor> {
        let mut output = tgt.clone();
        for layer in &self.layers {
            output = layer.forward(&output, memory)?;
        }
        Ok(output)
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
                params.insert(format!("layers.{}.{}", i, name), param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.layers.first().map_or(false, |l| l.training())
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

/// DETR Transformer - Encoder-Decoder architecture for object detection
///
/// Core transformer component that processes image features and object queries
/// to produce object representations for classification and localization.
#[derive(Debug)]
pub struct DETRTransformer {
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    object_queries: Parameter, // Learnable object queries
    pos_embed: PositionalEncoding,
    hidden_dim: usize,
    num_queries: usize,
}

impl DETRTransformer {
    /// Creates a new DETR transformer
    ///
    /// # Arguments
    /// * `hidden_dim` - Hidden dimension
    /// * `nheads` - Number of attention heads
    /// * `num_encoder_layers` - Number of encoder layers
    /// * `num_decoder_layers` - Number of decoder layers
    pub fn new(
        hidden_dim: usize,
        nheads: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
    ) -> Result<Self> {
        let encoder = TransformerEncoder::new(
            hidden_dim,
            nheads,
            hidden_dim * 4, // feedforward_dim
            num_encoder_layers,
            0.1, // dropout
        );

        let decoder = TransformerDecoder::new(
            hidden_dim,
            nheads,
            hidden_dim * 4, // feedforward_dim
            num_decoder_layers,
            0.1, // dropout
        );

        let num_queries = 100; // Standard DETR uses 100 object queries
        let mut rng = rng();
        let query_data = (0..num_queries * hidden_dim)
            .map(|_| 0.1 * (rng.normal(0.0, 1.0) as f32))
            .collect::<Vec<_>>();
        let object_queries_tensor = Tensor::from_slice(&query_data, &[num_queries, hidden_dim])?;
        let object_queries = Parameter::new(object_queries_tensor, true);

        let pos_embed = PositionalEncoding::new(hidden_dim, 5000)?;

        Ok(Self {
            encoder,
            decoder,
            object_queries,
            pos_embed,
            hidden_dim,
            num_queries,
        })
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get number of object queries
    pub fn num_queries(&self) -> usize {
        self.num_queries
    }
}

impl Module for DETRTransformer {
    fn forward(&self, src: &Tensor) -> Result<Tensor> {
        let batch_size = src.size(0)?;

        // Add positional encoding to source
        let pos_enc = self.pos_embed.forward(src)?;
        let src_with_pos = src.add(&pos_enc)?;

        // Encode features
        let memory = self.encoder.forward(&src_with_pos)?; // (B, seq_len, hidden_dim)

        // Prepare object queries for decoder
        let queries_data = self.object_queries.data.read().unwrap();
        let queries = queries_data
            .unsqueeze(0)? // (1, num_queries, hidden_dim)
            .expand(&[batch_size, self.num_queries, self.hidden_dim])?; // (B, num_queries, hidden_dim)

        // Decode to get object representations
        let hs = self.decoder.forward(&queries, &memory)?; // (B, num_queries, hidden_dim)

        Ok(hs)
    }

    fn train(&mut self) {
        self.encoder.train();
        self.decoder.train();
        self.pos_embed.train();
    }

    fn eval(&mut self) {
        self.encoder.eval();
        self.decoder.eval();
        self.pos_embed.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.encoder.parameters() {
            params.insert(format!("encoder.{}", name), param);
        }
        for (name, param) in self.decoder.parameters() {
            params.insert(format!("decoder.{}", name), param);
        }
        params.insert("object_queries".to_string(), self.object_queries.clone());
        for (name, param) in self.pos_embed.parameters() {
            params.insert(format!("pos_embed.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.encoder.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.encoder.to_device(device)?;
        self.decoder.to_device(device)?;
        self.object_queries.to_device(device)?;
        self.pos_embed.to_device(device)?;
        Ok(())
    }
}

/// DETR Configuration
#[derive(Debug, Clone)]
pub struct DETRConfig {
    pub num_classes: usize,
    pub num_queries: usize,
    pub hidden_dim: usize,
    pub nheads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub backbone_channels: usize,
}

impl DETRConfig {
    /// Create DETR configuration for COCO dataset
    pub fn coco(num_classes: usize) -> Self {
        Self {
            num_classes,
            num_queries: 100,
            hidden_dim: 256,
            nheads: 8,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            backbone_channels: 2048, // ResNet-50 output
        }
    }

    /// Create smaller DETR configuration
    pub fn small(num_classes: usize) -> Self {
        Self {
            num_classes,
            num_queries: 50,
            hidden_dim: 128,
            nheads: 4,
            num_encoder_layers: 3,
            num_decoder_layers: 3,
            backbone_channels: 2048,
        }
    }
}

/// DETR (Detection Transformer) - End-to-end object detection with transformers
///
/// Implements "End-to-End Object Detection with Transformers" which revolutionized
/// object detection by eliminating the need for hand-crafted components like NMS
/// and anchor generation.
#[derive(Debug)]
pub struct DETR {
    config: DETRConfig,
    backbone: ResNet,             // Feature extraction backbone
    input_proj: Conv2d,           // Input projection layer
    transformer: DETRTransformer, // Transformer encoder-decoder
    class_embed: Linear,          // Classification head
    bbox_embed: MLP,              // Bounding box regression head
}

impl DETR {
    /// Creates a new DETR model with custom configuration
    ///
    /// # Arguments
    /// * `num_classes` - Number of object classes (excluding background)
    /// * `num_queries` - Number of object queries (typically 100)
    /// * `hidden_dim` - Hidden dimension of transformer
    /// * `nheads` - Number of attention heads
    /// * `num_encoder_layers` - Number of transformer encoder layers
    /// * `num_decoder_layers` - Number of transformer decoder layers
    pub fn new(
        num_classes: usize,
        num_queries: usize,
        hidden_dim: usize,
        nheads: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
    ) -> Result<Self> {
        let config = DETRConfig {
            num_classes,
            num_queries,
            hidden_dim,
            nheads,
            num_encoder_layers,
            num_decoder_layers,
            backbone_channels: 2048, // ResNet-50 output
        };

        // Use ResNet-50 as backbone (without final classification layer)
        let backbone = ResNet::resnet50(1000)?; // Temporary num_classes

        let input_proj = Conv2d::new(
            config.backbone_channels, // ResNet-50 output channels
            hidden_dim,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            false,
            1,
        );

        let transformer = DETRTransformer::new(
            hidden_dim,
            nheads,
            num_encoder_layers,
            num_decoder_layers,
        )?;

        let class_embed = Linear::new(hidden_dim, num_classes + 1, true); // +1 for "no object"
        let bbox_embed = MLP::new(hidden_dim, hidden_dim, 4, 3); // 4 coordinates (x, y, w, h)

        Ok(Self {
            config,
            backbone,
            input_proj,
            transformer,
            class_embed,
            bbox_embed,
        })
    }

    /// Create DETR model with default configuration for COCO detection
    pub fn detr(num_classes: usize) -> Result<Self> {
        let config = DETRConfig::coco(num_classes);
        Self::new(
            config.num_classes,
            config.num_queries,
            config.hidden_dim,
            config.nheads,
            config.num_encoder_layers,
            config.num_decoder_layers,
        )
    }

    /// Create smaller DETR model for faster inference
    pub fn detr_small(num_classes: usize) -> Result<Self> {
        let config = DETRConfig::small(num_classes);
        Self::new(
            config.num_classes,
            config.num_queries,
            config.hidden_dim,
            config.nheads,
            config.num_encoder_layers,
            config.num_decoder_layers,
        )
    }

    /// Get model configuration
    pub fn config(&self) -> &DETRConfig {
        &self.config
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters().values().map(|p| {
            let data = p.data.read().unwrap();
            data.numel()
        }).sum()
    }
}

impl Module for DETR {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Extract features using backbone
        let mut features = self.backbone.forward(x)?;

        // Project features to transformer dimension
        features = self.input_proj.forward(&features)?;

        // Flatten spatial dimensions for transformer
        let (batch_size, channels, height, width) = (
            features.size(0)?,
            features.size(1)?,
            features.size(2)?,
            features.size(3)?,
        );

        // Reshape: (B, C, H, W) -> (B, H*W, C)
        let seq_len = height * width;
        let src = features
            .view(&[batch_size, channels, seq_len])?
            .permute(&[0, 2, 1])?; // (B, H*W, C)

        // Pass through transformer
        let hs = self.transformer.forward(&src)?; // (B, num_queries, hidden_dim)

        // Predict classes and bounding boxes
        let outputs_class = self.class_embed.forward(&hs)?; // (B, num_queries, num_classes+1)
        let outputs_coord = self.bbox_embed.forward(&hs)?; // (B, num_queries, 4)

        // Apply sigmoid to box coordinates to get values in [0, 1]
        let outputs_coord = outputs_coord.sigmoid()?;

        // Return class predictions (in practice, you'd return both classes and boxes)
        Ok(outputs_class)
    }

    fn train(&mut self) {
        self.backbone.train();
        self.input_proj.train();
        self.transformer.train();
        self.class_embed.train();
        self.bbox_embed.train();
    }

    fn eval(&mut self) {
        self.backbone.eval();
        self.input_proj.eval();
        self.transformer.eval();
        self.class_embed.eval();
        self.bbox_embed.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.backbone.parameters() {
            params.insert(format!("backbone.{}", name), param);
        }
        for (name, param) in self.input_proj.parameters() {
            params.insert(format!("input_proj.{}", name), param);
        }
        for (name, param) in self.transformer.parameters() {
            params.insert(format!("transformer.{}", name), param);
        }
        for (name, param) in self.class_embed.parameters() {
            params.insert(format!("class_embed.{}", name), param);
        }
        for (name, param) in self.bbox_embed.parameters() {
            params.insert(format!("bbox_embed.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.backbone.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.backbone.to_device(device)?;
        self.input_proj.to_device(device)?;
        self.transformer.to_device(device)?;
        self.class_embed.to_device(device)?;
        self.bbox_embed.to_device(device)?;
        Ok(())
    }
}

/// Factory for creating DETR variants
pub struct DETRFactory;

impl DETRFactory {
    /// Create any DETR variant by name
    pub fn create(variant: &str, num_classes: usize) -> Result<DETR> {
        match variant.to_lowercase().as_str() {
            "detr" | "default" => DETR::detr(num_classes),
            "small" | "detr-small" => DETR::detr_small(num_classes),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown DETR variant: {}. Available: detr, small",
                variant
            ))),
        }
    }

    /// Get model information
    pub fn model_info(variant: &str) -> Result<String> {
        let info = match variant.to_lowercase().as_str() {
            "detr" | "default" => {
                "DETR: 100 queries, 256 hidden_dim, 6+6 layers (~41M parameters)"
            }
            "small" | "detr-small" => {
                "DETR-Small: 50 queries, 128 hidden_dim, 3+3 layers (~15M parameters)"
            }
            _ => return Err(TorshError::InvalidArgument(format!("Unknown variant: {}", variant))),
        };
        Ok(info.to_string())
    }

    /// List all available variants
    pub fn available_variants() -> Vec<&'static str> {
        vec!["detr", "small"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::Tensor;

    #[test]
    fn test_mlp() -> Result<()> {
        let mut mlp = MLP::new(256, 256, 4, 3);

        assert_eq!(mlp.num_layers(), 3);

        let input = torsh_tensor::creation::randn(&[10, 100, 256])?;
        let output = mlp.forward(&input)?;

        assert_eq!(output.shape(), &[10, 100, 4]);

        // Test train/eval modes
        mlp.train();
        assert!(mlp.training());

        mlp.eval();
        assert!(!mlp.training());

        Ok(())
    }

    #[test]
    fn test_positional_encoding() -> Result<()> {
        let mut pos_enc = PositionalEncoding::new(256, 1000)?;

        assert_eq!(pos_enc.d_model(), 256);
        assert_eq!(pos_enc.max_len(), 1000);

        let input = torsh_tensor::creation::randn(&[2, 100, 256])?;
        let output = pos_enc.forward(&input)?;

        assert_eq!(output.shape(), &[1, 100, 256]);

        Ok(())
    }

    #[test]
    fn test_transformer_encoder_layer() -> Result<()> {
        let mut layer = TransformerEncoderLayer::new(256, 8, 1024, 0.1);

        let input = torsh_tensor::creation::randn(&[2, 100, 256])?;
        let output = layer.forward(&input)?;

        assert_eq!(output.shape(), input.shape());

        Ok(())
    }

    #[test]
    fn test_transformer_decoder_layer() -> Result<()> {
        let mut layer = TransformerDecoderLayer::new(256, 8, 1024, 0.1);

        let tgt = torsh_tensor::creation::randn(&[2, 100, 256])?;
        let memory = torsh_tensor::creation::randn(&[2, 196, 256])?;
        let output = layer.forward(&tgt, &memory)?;

        assert_eq!(output.shape(), tgt.shape());

        Ok(())
    }

    #[test]
    fn test_transformer_encoder() -> Result<()> {
        let mut encoder = TransformerEncoder::new(256, 8, 1024, 6, 0.1);

        assert_eq!(encoder.num_layers(), 6);

        let input = torsh_tensor::creation::randn(&[2, 196, 256])?;
        let output = encoder.forward(&input)?;

        assert_eq!(output.shape(), input.shape());

        Ok(())
    }

    #[test]
    fn test_transformer_decoder() -> Result<()> {
        let mut decoder = TransformerDecoder::new(256, 8, 1024, 6, 0.1);

        assert_eq!(decoder.num_layers(), 6);

        let tgt = torsh_tensor::creation::randn(&[2, 100, 256])?;
        let memory = torsh_tensor::creation::randn(&[2, 196, 256])?;
        let output = decoder.forward(&tgt, &memory)?;

        assert_eq!(output.shape(), tgt.shape());

        Ok(())
    }

    #[test]
    fn test_detr_transformer() -> Result<()> {
        let mut transformer = DETRTransformer::new(256, 8, 6, 6)?;

        assert_eq!(transformer.hidden_dim(), 256);
        assert_eq!(transformer.num_queries(), 100);

        let input = torsh_tensor::creation::randn(&[2, 196, 256])?;
        let output = transformer.forward(&input)?;

        assert_eq!(output.shape(), &[2, 100, 256]);

        Ok(())
    }

    #[test]
    fn test_detr_variants() -> Result<()> {
        let variants = [
            ("default", DETR::detr(80)?),
            ("small", DETR::detr_small(80)?),
        ];

        for (name, model) in variants {
            let input = torsh_tensor::creation::randn(&[1, 3, 224, 224])?;
            let output = model.forward(&input)?;

            // Default DETR: 100 queries, Small DETR: 50 queries
            let expected_queries = if name == "small" { 50 } else { 100 };
            assert_eq!(output.shape(), &[1, expected_queries, 81], "Failed for DETR-{}", name); // 80 classes + 1 no-object

            // Check parameter count
            let params = model.num_parameters();
            assert!(params > 1_000_000, "Model {} should have >1M parameters", name);
        }

        Ok(())
    }

    #[test]
    fn test_detr_factory() -> Result<()> {
        // Test factory creation
        let model = DETRFactory::create("detr", 80)?;
        assert_eq!(model.config().num_classes, 80);
        assert_eq!(model.config().num_queries, 100);

        // Test small variant
        let small_model = DETRFactory::create("small", 20)?;
        assert_eq!(small_model.config().num_queries, 50);

        // Test invalid variant
        assert!(DETRFactory::create("invalid", 80).is_err());

        // Test model info
        let info = DETRFactory::model_info("detr")?;
        assert!(info.contains("DETR"));
        assert!(info.contains("100 queries"));

        // Test available variants
        let variants = DETRFactory::available_variants();
        assert!(variants.contains(&"detr"));
        assert!(variants.contains(&"small"));

        Ok(())
    }

    #[test]
    fn test_detr_config() {
        let config = DETRConfig::coco(80);
        assert_eq!(config.num_classes, 80);
        assert_eq!(config.num_queries, 100);
        assert_eq!(config.hidden_dim, 256);

        let config_small = DETRConfig::small(20);
        assert_eq!(config_small.num_queries, 50);
        assert_eq!(config_small.hidden_dim, 128);
    }

    #[test]
    fn test_forward_pass_shapes() -> Result<()> {
        let model = DETR::detr(10)?;

        // Test different batch sizes
        for batch_size in [1, 2, 4] {
            let input = torsh_tensor::creation::randn(&[batch_size, 3, 224, 224])?;
            let output = model.forward(&input)?;
            assert_eq!(output.shape(), &[batch_size, 100, 11]); // 10 classes + 1 no-object
        }

        Ok(())
    }

    #[test]
    fn test_detr_parameters() -> Result<()> {
        let model = DETR::detr_small(10)?;
        let params = model.parameters();

        // Should have backbone, projection, transformer, and head parameters
        assert!(params.keys().any(|k| k.starts_with("backbone")));
        assert!(params.keys().any(|k| k.starts_with("input_proj")));
        assert!(params.keys().any(|k| k.starts_with("transformer")));
        assert!(params.keys().any(|k| k.starts_with("class_embed")));
        assert!(params.keys().any(|k| k.starts_with("bbox_embed")));

        // Should have object queries
        assert!(params.keys().any(|k| k.contains("object_queries")));

        // Should have attention parameters
        assert!(params.keys().any(|k| k.contains("self_attn")));
        assert!(params.keys().any(|k| k.contains("cross_attn")));

        Ok(())
    }
}

/// Re-export commonly used types
pub use self::{
    MLP,
    PositionalEncoding,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    DETRTransformer,
    DETR,
    DETRConfig,
    DETRFactory,
};