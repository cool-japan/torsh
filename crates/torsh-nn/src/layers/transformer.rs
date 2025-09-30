//! Transformer architecture components
//!
//! This module implements the standard Transformer architecture including:
//! - TransformerEncoder and TransformerEncoderLayer
//! - TransformerDecoder and TransformerDecoderLayer  
//! - Positional encoding utilities

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// Transformer Encoder Layer
///
/// A single layer of the Transformer encoder, consisting of:
/// - Multi-head self-attention
/// - Position-wise feed-forward network
/// - Residual connections and layer normalization
pub struct TransformerEncoderLayer {
    base: ModuleBase,
    d_model: usize,
    nhead: usize,
    dim_feedforward: usize,
    dropout: f32,
    activation: String,
    layer_norm_eps: f32,
    batch_first: bool,
    norm_first: bool,
}

impl TransformerEncoderLayer {
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: Option<usize>,
        dropout: Option<f32>,
        activation: Option<String>,
        layer_norm_eps: Option<f32>,
        batch_first: Option<bool>,
        norm_first: Option<bool>,
    ) -> Result<Self> {
        let mut base = ModuleBase::new();

        let dim_feedforward = dim_feedforward.unwrap_or(2048);
        let dropout = dropout.unwrap_or(0.1);
        let activation = activation.unwrap_or_else(|| "relu".to_string());
        let layer_norm_eps = layer_norm_eps.unwrap_or(1e-5);
        let batch_first = batch_first.unwrap_or(false);
        let norm_first = norm_first.unwrap_or(false);

        // Self-attention weights
        let self_attn_weight = crate::init::xavier_uniform(&[3 * d_model, d_model])?;
        let self_attn_out_weight = crate::init::xavier_uniform(&[d_model, d_model])?;
        base.register_parameter(
            "self_attn.in_proj_weight".to_string(),
            Parameter::new(self_attn_weight),
        );
        base.register_parameter(
            "self_attn.out_proj.weight".to_string(),
            Parameter::new(self_attn_out_weight),
        );

        // Self-attention biases
        let self_attn_bias = zeros(&[3 * d_model])?;
        let self_attn_out_bias = zeros(&[d_model])?;
        base.register_parameter(
            "self_attn.in_proj_bias".to_string(),
            Parameter::new(self_attn_bias),
        );
        base.register_parameter(
            "self_attn.out_proj.bias".to_string(),
            Parameter::new(self_attn_out_bias),
        );

        // Feed-forward network weights
        let linear1_weight = crate::init::xavier_uniform(&[dim_feedforward, d_model])?;
        let linear2_weight = crate::init::xavier_uniform(&[d_model, dim_feedforward])?;
        base.register_parameter("linear1.weight".to_string(), Parameter::new(linear1_weight));
        base.register_parameter("linear2.weight".to_string(), Parameter::new(linear2_weight));

        // Feed-forward network biases
        let linear1_bias = zeros(&[dim_feedforward])?;
        let linear2_bias = zeros(&[d_model])?;
        base.register_parameter("linear1.bias".to_string(), Parameter::new(linear1_bias));
        base.register_parameter("linear2.bias".to_string(), Parameter::new(linear2_bias));

        // Layer normalization weights and biases
        let norm1_weight = ones(&[d_model])?;
        let norm1_bias = zeros(&[d_model])?;
        let norm2_weight = ones(&[d_model])?;
        let norm2_bias = zeros(&[d_model])?;
        base.register_parameter("norm1.weight".to_string(), Parameter::new(norm1_weight));
        base.register_parameter("norm1.bias".to_string(), Parameter::new(norm1_bias));
        base.register_parameter("norm2.weight".to_string(), Parameter::new(norm2_weight));
        base.register_parameter("norm2.bias".to_string(), Parameter::new(norm2_bias));

        Ok(Self {
            base,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
        })
    }

    /// Apply multi-head self-attention
    fn self_attention(&self, src: &Tensor, src_mask: Option<&Tensor>) -> Result<Tensor> {
        let in_proj_weight = self.base.parameters["self_attn.in_proj_weight"]
            .tensor()
            .read()
            .clone();
        let out_proj_weight = self.base.parameters["self_attn.out_proj.weight"]
            .tensor()
            .read()
            .clone();
        let in_proj_bias = self.base.parameters["self_attn.in_proj_bias"]
            .tensor()
            .read()
            .clone();
        let out_proj_bias = self.base.parameters["self_attn.out_proj.bias"]
            .tensor()
            .read()
            .clone();

        // Project input to Q, K, V
        let qkv = src
            .matmul(&in_proj_weight.transpose(0, 1)?)?
            .add_op(&in_proj_bias)?;

        let src_shape_binding = src.shape();
        let src_shape = src_shape_binding.dims();
        let batch_size = if self.batch_first {
            src_shape[0]
        } else {
            src_shape[1]
        };
        let seq_len = if self.batch_first {
            src_shape[1]
        } else {
            src_shape[0]
        };

        // Split into Q, K, V
        let chunk_size = self.d_model;
        let q = qkv.narrow((qkv.shape().ndim() - 1) as i32, 0, chunk_size)?;
        let k = qkv.narrow(
            (qkv.shape().ndim() - 1) as i32,
            chunk_size as i64,
            chunk_size,
        )?;
        let v = qkv.narrow(
            (qkv.shape().ndim() - 1) as i32,
            (2 * chunk_size) as i64,
            chunk_size,
        )?;

        // Reshape for multi-head attention
        let head_dim = self.d_model / self.nhead;

        let q = if self.batch_first {
            q.reshape(&[
                batch_size as i32,
                seq_len as i32,
                self.nhead as i32,
                head_dim as i32,
            ])?
            .transpose(1, 2)?
        } else {
            q.reshape(&[
                seq_len as i32,
                batch_size as i32,
                self.nhead as i32,
                head_dim as i32,
            ])?
            .transpose(0, 2)?
            .transpose(1, 3)?
        };

        let k = if self.batch_first {
            k.reshape(&[
                batch_size as i32,
                seq_len as i32,
                self.nhead as i32,
                head_dim as i32,
            ])?
            .transpose(1, 2)?
        } else {
            k.reshape(&[
                seq_len as i32,
                batch_size as i32,
                self.nhead as i32,
                head_dim as i32,
            ])?
            .transpose(0, 2)?
            .transpose(1, 3)?
        };

        let v = if self.batch_first {
            v.reshape(&[
                batch_size as i32,
                seq_len as i32,
                self.nhead as i32,
                head_dim as i32,
            ])?
            .transpose(1, 2)?
        } else {
            v.reshape(&[
                seq_len as i32,
                batch_size as i32,
                self.nhead as i32,
                head_dim as i32,
            ])?
            .transpose(0, 2)?
            .transpose(1, 3)?
        };

        // Scaled dot-product attention
        let d_k = head_dim as f32;
        let scale = 1.0 / d_k.sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)?)?.mul_scalar(scale)?;

        // Apply mask if provided
        let scores = if let Some(mask) = src_mask {
            let large_neg = mask.mul_scalar(-1e9)?;
            scores.add_op(&large_neg)?
        } else {
            scores
        };

        // Apply softmax and dropout
        let attn_weights = scores.softmax(-1)?;
        let attn_weights = if self.dropout > 0.0 && self.training() {
            crate::functional::dropout(&attn_weights, self.dropout, self.training())?
        } else {
            attn_weights
        };

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to original format
        let attn_output = if self.batch_first {
            attn_output.transpose(1, 2)?.contiguous()?.reshape(&[
                batch_size as i32,
                seq_len as i32,
                self.d_model as i32,
            ])?
        } else {
            attn_output
                .transpose(0, 2)?
                .transpose(1, 3)?
                .contiguous()?
                .reshape(&[seq_len as i32, batch_size as i32, self.d_model as i32])?
        };

        // Output projection
        attn_output
            .matmul(&out_proj_weight.transpose(0, 1)?)?
            .add_op(&out_proj_bias)
    }

    /// Apply position-wise feed-forward network
    fn feed_forward(&self, src: &Tensor) -> Result<Tensor> {
        let linear1_weight = self.base.parameters["linear1.weight"]
            .tensor()
            .read()
            .clone();
        let linear1_bias = self.base.parameters["linear1.bias"].tensor().read().clone();
        let linear2_weight = self.base.parameters["linear2.weight"]
            .tensor()
            .read()
            .clone();
        let linear2_bias = self.base.parameters["linear2.bias"].tensor().read().clone();

        // First linear transformation
        let x = src
            .matmul(&linear1_weight.transpose(0, 1)?)?
            .add_op(&linear1_bias)?;

        // Apply activation function
        let x = match self.activation.as_str() {
            "relu" => x.relu()?,
            "gelu" => x.gelu()?,
            "silu" | "swish" => {
                // SiLU activation: x * sigmoid(x)
                let sigmoid_x = x.sigmoid()?;
                x.mul_op(&sigmoid_x)?
            }
            _ => x.relu()?, // Default to ReLU
        };

        // Apply dropout
        let x = if self.dropout > 0.0 && self.training() {
            crate::functional::dropout(&x, self.dropout, self.training())?
        } else {
            x
        };

        // Second linear transformation
        x.matmul(&linear2_weight.transpose(0, 1)?)?
            .add_op(&linear2_bias)
    }

    /// Apply layer normalization
    fn layer_norm(&self, x: &Tensor, norm_type: usize) -> Result<Tensor> {
        let weight_key = format!("norm{}.weight", norm_type);
        let bias_key = format!("norm{}.bias", norm_type);

        let weight = self.base.parameters[&weight_key].tensor().read().clone();
        let bias = self.base.parameters[&bias_key].tensor().read().clone();

        // Compute mean and variance along the last dimension
        let last_dim = x.shape().ndim() - 1;
        let mean = x.mean(Some(&[last_dim]), true)?;
        let var = x.var(
            Some(&[last_dim]),
            true,
            torsh_tensor::stats::StatMode::Sample,
        )?;

        // Normalize
        let normalized = x
            .sub(&mean)?
            .div(&(var.add_scalar(self.layer_norm_eps)?.sqrt()?))?;

        // Scale and shift
        normalized.mul_op(&weight)?.add_op(&bias)
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl TransformerEncoderLayer {
    /// Forward pass with optional attention mask
    pub fn forward_with_mask(&self, src: &Tensor, src_mask: Option<&Tensor>) -> Result<Tensor> {
        let x = src;

        if self.norm_first {
            // Pre-normalization variant
            let norm_x = self.layer_norm(x, 1)?;
            let attn_output = self.self_attention(&norm_x, src_mask)?;
            let x = x.add_op(&attn_output)?; // Residual connection

            let norm_x = self.layer_norm(&x, 2)?;
            let ff_output = self.feed_forward(&norm_x)?;
            x.add_op(&ff_output) // Residual connection
        } else {
            // Post-normalization variant (original Transformer)
            let attn_output = self.self_attention(x, src_mask)?;
            let x = x.add_op(&attn_output)?; // Residual connection
            let x = self.layer_norm(&x, 1)?;

            let ff_output = self.feed_forward(&x)?;
            let x = x.add_op(&ff_output)?; // Residual connection
            self.layer_norm(&x, 2)
        }
    }
}

/// Transformer Encoder
///
/// Stack of N TransformerEncoderLayer modules
pub struct TransformerEncoder {
    base: ModuleBase,
    layers: Vec<TransformerEncoderLayer>,
    num_layers: usize,
    #[allow(dead_code)]
    norm: Option<()>, // LayerNorm would go here
}

impl TransformerEncoder {
    pub fn new(encoder_layer: TransformerEncoderLayer, num_layers: usize) -> Result<Self> {
        let mut base = ModuleBase::new();
        let mut layers = Vec::with_capacity(num_layers);

        // Create copies of the encoder layer
        // Note: In a complete implementation, we'd need proper parameter sharing/copying
        for i in 0..num_layers {
            let layer = TransformerEncoderLayer::new(
                encoder_layer.d_model,
                encoder_layer.nhead,
                Some(encoder_layer.dim_feedforward),
                Some(encoder_layer.dropout),
                Some(encoder_layer.activation.clone()),
                Some(encoder_layer.layer_norm_eps),
                Some(encoder_layer.batch_first),
                Some(encoder_layer.norm_first),
            )?;

            // Register layer parameters with unique names
            for (name, param) in layer.parameters() {
                let layer_name = format!("layers.{}.{}", i, name);
                base.register_parameter(layer_name, param);
            }

            layers.push(layer);
        }

        Ok(Self {
            base,
            layers,
            num_layers,
            norm: None,
        })
    }

    /// Forward pass through all encoder layers
    pub fn forward_with_mask(&self, src: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let mut output = src.clone();

        for layer in &self.layers {
            output = layer.forward_with_mask(&output, mask)?;
        }

        Ok(output)
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Complete Transformer model
///
/// Combines embedding, positional encoding, and transformer encoder
pub struct Transformer {
    base: ModuleBase,
    d_model: usize,
    nhead: usize,
    num_encoder_layers: usize,
    dim_feedforward: usize,
    dropout: f32,
    #[allow(dead_code)]
    max_seq_length: usize,
    vocab_size: Option<usize>,
}

impl Transformer {
    pub fn new(
        d_model: usize,
        nhead: usize,
        num_encoder_layers: usize,
        num_decoder_layers: Option<usize>,
        dim_feedforward: Option<usize>,
        dropout: Option<f32>,
        max_seq_length: Option<usize>,
        vocab_size: Option<usize>,
    ) -> Result<Self> {
        let mut base = ModuleBase::new();
        let dim_feedforward = dim_feedforward.unwrap_or(2048);
        let dropout = dropout.unwrap_or(0.1);
        let max_seq_length = max_seq_length.unwrap_or(1024);
        let _num_decoder_layers = num_decoder_layers.unwrap_or(0);

        // Create positional encoding
        let pos_encoding = create_positional_encoding(max_seq_length, d_model)?;
        base.register_parameter("pos_encoding".to_string(), Parameter::new(pos_encoding));

        // Create input embedding if vocab_size is provided
        if let Some(vocab_size) = vocab_size {
            let embedding_weight = crate::init::xavier_uniform(&[vocab_size, d_model])?;
            base.register_parameter(
                "embedding.weight".to_string(),
                Parameter::new(embedding_weight),
            );
        }

        Ok(Self {
            base,
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
            max_seq_length,
            vocab_size,
        })
    }

    /// Create the encoder layers
    pub fn create_encoder(&self) -> Result<TransformerEncoder> {
        let encoder_layer = TransformerEncoderLayer::new(
            self.d_model,
            self.nhead,
            Some(self.dim_feedforward),
            Some(self.dropout),
            None,        // Use default activation (ReLU)
            None,        // Use default layer norm eps
            Some(true),  // batch_first = true
            Some(false), // norm_first = false (post-norm)
        )?;

        TransformerEncoder::new(encoder_layer, self.num_encoder_layers)
    }

    /// Apply positional encoding to input
    pub fn add_positional_encoding(&self, x: &Tensor) -> Result<Tensor> {
        let pos_encoding = self.base.parameters["pos_encoding"].tensor().read().clone();
        let seq_len = x.shape().dims()[1]; // Assuming batch_first format

        // Slice positional encoding to match sequence length
        let pos_slice = pos_encoding.narrow(0, 0, seq_len)?;

        // Add positional encoding (broadcasting)
        x.add_op(&pos_slice.unsqueeze(0)?)
    }

    /// Embedding lookup (if vocab_size was provided)
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        if self.vocab_size.is_none() {
            return Err(torsh_core::TorshError::InvalidArgument(
                "Transformer was not initialized with vocab_size".to_string(),
            ));
        }

        let embedding_weight = self.base.parameters["embedding.weight"]
            .tensor()
            .read()
            .clone();

        // Simple embedding lookup - in practice this would be more sophisticated
        // For now, assume input_ids are already in the right format
        input_ids.matmul(&embedding_weight)
    }
}

impl Module for Transformer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // This is a simplified forward pass
        // In practice, you'd handle embedding lookup, positional encoding, etc.
        let encoder = self.create_encoder()?;
        encoder.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

// Helper function to create positional encoding
fn create_positional_encoding(max_len: usize, d_model: usize) -> Result<Tensor> {
    let mut pos_encoding = vec![0.0f32; max_len * d_model];

    for pos in 0..max_len {
        for i in (0..d_model).step_by(2) {
            let angle = pos as f32 / 10000.0_f32.powf(i as f32 / d_model as f32);

            pos_encoding[pos * d_model + i] = angle.sin();
            if i + 1 < d_model {
                pos_encoding[pos * d_model + i + 1] = angle.cos();
            }
        }
    }

    Tensor::from_vec(pos_encoding, &[max_len, d_model])
}

impl std::fmt::Debug for TransformerEncoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerEncoderLayer")
            .field("d_model", &self.d_model)
            .field("nhead", &self.nhead)
            .field("dim_feedforward", &self.dim_feedforward)
            .field("dropout", &self.dropout)
            .field("activation", &self.activation)
            .finish()
    }
}

impl std::fmt::Debug for TransformerEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerEncoder")
            .field("num_layers", &self.num_layers)
            .finish()
    }
}

impl std::fmt::Debug for Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transformer")
            .field("d_model", &self.d_model)
            .field("nhead", &self.nhead)
            .field("num_encoder_layers", &self.num_encoder_layers)
            .field("dim_feedforward", &self.dim_feedforward)
            .field("vocab_size", &self.vocab_size)
            .finish()
    }
}
