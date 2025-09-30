//! T5 (Text-to-Text Transfer Transformer) implementation
//!
//! This module implements the T5 encoder-decoder architecture with relative position encoding,
//! layer normalization variants, and support for text-to-text generation tasks.

use super::transformer::{FeedForward, MultiHeadAttention};
use crate::{TextModel, TextModelConfig};
use std::collections::HashMap;
use torsh_core::{device::DeviceType, Result};
use torsh_nn::{prelude::*, Module, Parameter};
use torsh_tensor::creation::*;
use torsh_tensor::Tensor;

/// T5 layer normalization (RMS norm variant)
pub struct T5LayerNorm {
    weight: Parameter,
    variance_epsilon: f32,
    hidden_dim: usize,
    is_training: bool,
}

impl T5LayerNorm {
    pub fn new(hidden_dim: usize, eps: f32) -> Self {
        Self {
            weight: Parameter::new(ones(&[hidden_dim])),
            variance_epsilon: eps,
            hidden_dim,
            is_training: true,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // T5 uses RMS norm: x / sqrt(mean(x^2) + eps) * weight
        let input_dtype = input.clone();
        let input_squared = input.mul(input)?;

        // Calculate mean across the last dimension
        let mean = input_squared.mean_keepdim(&[-1])?;
        let variance = mean.add_scalar(self.variance_epsilon)?;
        let inv_std = variance.rsqrt()?;

        let normalized = input.mul(&inv_std)?;
        normalized.mul(&self.weight.data())
    }
}

impl Module for T5LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
    }

    fn eval(&mut self) {
        self.is_training = false;
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        // Move weight parameter to device
        self.weight = Parameter::new(self.weight.data().to_device(device)?);
        Ok(())
    }
}

/// T5 relative position bias for attention
pub struct T5RelativePositionBias {
    relative_attention_bias: Embedding,
    num_buckets: usize,
    max_distance: usize,
    num_heads: usize,
    is_training: bool,
}

impl T5RelativePositionBias {
    pub fn new(num_heads: usize, num_buckets: usize, max_distance: usize) -> Self {
        Self {
            relative_attention_bias: Embedding::new(num_buckets, num_heads),
            num_buckets,
            max_distance,
            num_heads,
            is_training: true,
        }
    }

    fn _relative_position_bucket(&self, relative_position: i32, bidirectional: bool) -> usize {
        let mut relative_buckets = 0;
        let mut n = relative_position;

        if bidirectional {
            self.num_buckets /= 2;
            if n > 0 {
                relative_buckets += self.num_buckets;
            } else {
                n = -n;
            }
        } else {
            n = (-n).max(0);
        }

        let max_exact = self.num_buckets / 2;
        let is_small = n < max_exact as i32;

        if is_small {
            n as usize + relative_buckets
        } else {
            let val = max_exact as f32
                + ((n as f32).ln() / (self.max_distance as f32 / max_exact as f32).ln()
                    * (self.num_buckets / 2 - max_exact) as f32);
            (val as usize).min(self.num_buckets - 1) + relative_buckets
        }
    }

    pub fn compute_bias(
        &self,
        query_length: usize,
        key_length: usize,
        bidirectional: bool,
    ) -> Result<Tensor> {
        // Create position bias matrix - simplified implementation
        let bias_shape = [self.num_heads, query_length, key_length];
        let bias: Tensor<f32> = zeros(&bias_shape);
        Ok(bias)
    }
}

impl Module for T5RelativePositionBias {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Default forward pass - bias computation is done separately
        Ok(input.clone())
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.relative_attention_bias.parameters() {
            params.insert(format!("relative_attention_bias.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        self.relative_attention_bias.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.relative_attention_bias.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.relative_attention_bias.to_device(device)
    }
}

/// T5 Multi-head attention with relative position bias
pub struct T5Attention {
    base_attention: MultiHeadAttention,
    relative_position_bias: Option<T5RelativePositionBias>,
    has_relative_attention_bias: bool,
    is_training: bool,
}

impl T5Attention {
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        dropout: f32,
        has_relative_attention_bias: bool,
        device: DeviceType,
    ) -> Result<Self> {
        let relative_position_bias = if has_relative_attention_bias {
            Some(T5RelativePositionBias::new(num_heads, 32, 128)) // T5 defaults
        } else {
            None
        };

        Ok(Self {
            base_attention: MultiHeadAttention::new(hidden_dim, num_heads, dropout, device)?,
            relative_position_bias,
            has_relative_attention_bias,
            is_training: true,
        })
    }

    pub fn forward_with_bias(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        position_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Compute relative position bias if available
        let computed_bias = if let Some(key_value) = key.shape().dims().get(1) {
            if let Some(query_len) = query.shape().dims().get(1) {
                if self.has_relative_attention_bias {
                    if let Some(ref pos_bias) = self.relative_position_bias {
                        Some(pos_bias.compute_bias(*query_len, *key_value, true)?)
                    } else {
                        None
                    }
                } else {
                    position_bias.cloned()
                }
            } else {
                position_bias.cloned()
            }
        } else {
            position_bias.cloned()
        };

        // Combine attention mask and position bias
        let combined_mask = match (attention_mask, computed_bias.as_ref()) {
            (Some(mask), Some(bias)) => Some(mask.add(bias)?),
            (Some(mask), None) => Some(mask.clone()),
            (None, Some(bias)) => Some(bias.clone()),
            (None, None) => None,
        };

        self.base_attention
            .forward_with_mask(query, combined_mask.as_ref())
    }
}

impl Module for T5Attention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_bias(input, input, input, None, None)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.base_attention.parameters() {
            params.insert(format!("base_attention.{}", name), param);
        }

        if let Some(ref pos_bias) = self.relative_position_bias {
            for (name, param) in pos_bias.parameters() {
                params.insert(format!("relative_position_bias.{}", name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        self.base_attention.train();
        if let Some(ref mut pos_bias) = self.relative_position_bias {
            pos_bias.train();
        }
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.base_attention.eval();
        if let Some(ref mut pos_bias) = self.relative_position_bias {
            pos_bias.eval();
        }
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base_attention.to_device(device)?;
        if let Some(ref mut pos_bias) = self.relative_position_bias {
            pos_bias.to_device(device)?;
        }
        Ok(())
    }
}

/// T5 encoder layer
pub struct T5EncoderLayer {
    self_attention: T5Attention,
    feed_forward: FeedForward,
    layer_norm: T5LayerNorm,
    dropout: Dropout,
    is_training: bool,
}

impl T5EncoderLayer {
    pub fn new(config: &TextModelConfig, device: DeviceType) -> Result<Self> {
        Ok(Self {
            self_attention: T5Attention::new(
                config.hidden_dim,
                config.num_heads,
                config.attention_dropout,
                true, // First layer has relative attention bias
                device,
            )?,
            feed_forward: FeedForward::new(
                config.hidden_dim,
                config.intermediate_dim,
                config.dropout,
                device,
            )?,
            layer_norm: T5LayerNorm::new(config.hidden_dim, config.layer_norm_eps),
            dropout: Dropout::new(config.dropout),
            is_training: true,
        })
    }

    pub fn forward_with_mask(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // T5 uses different normalization order compared to standard transformer
        let normed_input = self.layer_norm.forward(input)?;
        let attention_output = self.self_attention.forward_with_bias(
            &normed_input,
            &normed_input,
            &normed_input,
            attention_mask,
            None,
        )?;
        let attention_output = self.dropout.forward(&attention_output)?;
        let hidden_states = input.add(&attention_output)?;

        // Feed forward
        let normed_hidden = self.layer_norm.forward(&hidden_states)?;
        let feed_forward_output = self.feed_forward.forward(&normed_hidden)?;
        let feed_forward_output = self.dropout.forward(&feed_forward_output)?;
        hidden_states.add(&feed_forward_output)
    }
}

impl Module for T5EncoderLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // T5 uses different normalization order compared to standard transformer
        let normed_input = self.layer_norm.forward(input)?;
        let attention_output = self.self_attention.forward(&normed_input)?;
        let attention_output = self.dropout.forward(&attention_output)?;
        let hidden_states = input.add(&attention_output)?;

        // Feed forward
        let normed_hidden = self.layer_norm.forward(&hidden_states)?;
        let feed_forward_output = self.feed_forward.forward(&normed_hidden)?;
        let feed_forward_output = self.dropout.forward(&feed_forward_output)?;
        hidden_states.add(&feed_forward_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.self_attention.parameters() {
            params.insert(format!("self_attention.{}", name), param);
        }
        for (name, param) in self.feed_forward.parameters() {
            params.insert(format!("feed_forward.{}", name), param);
        }
        for (name, param) in self.layer_norm.parameters() {
            params.insert(format!("layer_norm.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        self.self_attention.train();
        self.feed_forward.train();
        self.layer_norm.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.self_attention.eval();
        self.feed_forward.eval();
        self.layer_norm.eval();
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.self_attention.to_device(device)?;
        self.feed_forward.to_device(device)?;
        self.layer_norm.to_device(device)?;
        Ok(())
    }
}

/// T5 decoder layer with cross-attention
pub struct T5DecoderLayer {
    self_attention: T5Attention,
    cross_attention: T5Attention,
    feed_forward: FeedForward,
    layer_norm_self_attn: T5LayerNorm,
    layer_norm_cross_attn: T5LayerNorm,
    layer_norm_ff: T5LayerNorm,
    dropout: Dropout,
    is_training: bool,
}

impl T5DecoderLayer {
    pub fn new(config: &TextModelConfig, device: DeviceType) -> Result<Self> {
        Ok(Self {
            self_attention: T5Attention::new(
                config.hidden_dim,
                config.num_heads,
                config.attention_dropout,
                true, // Self attention has relative bias
                device,
            )?,
            cross_attention: T5Attention::new(
                config.hidden_dim,
                config.num_heads,
                config.attention_dropout,
                false, // Cross attention doesn't use relative bias
                device,
            )?,
            feed_forward: FeedForward::new(
                config.hidden_dim,
                config.intermediate_dim,
                config.dropout,
                device,
            )?,
            layer_norm_self_attn: T5LayerNorm::new(config.hidden_dim, config.layer_norm_eps),
            layer_norm_cross_attn: T5LayerNorm::new(config.hidden_dim, config.layer_norm_eps),
            layer_norm_ff: T5LayerNorm::new(config.hidden_dim, config.layer_norm_eps),
            dropout: Dropout::new(config.dropout),
            is_training: true,
        })
    }

    pub fn forward_with_encoder_hidden(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        self_attention_mask: Option<&Tensor>,
        cross_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self attention
        let normed_hidden = self.layer_norm_self_attn.forward(hidden_states)?;
        let self_attention_output = self.self_attention.forward_with_bias(
            &normed_hidden,
            &normed_hidden,
            &normed_hidden,
            self_attention_mask,
            None,
        )?;
        let self_attention_output = self.dropout.forward(&self_attention_output)?;
        let hidden_states = hidden_states.add(&self_attention_output)?;

        // Cross attention (if encoder hidden states provided)
        let hidden_states = if let Some(encoder_hidden) = encoder_hidden_states {
            let normed_hidden = self.layer_norm_cross_attn.forward(&hidden_states)?;
            let cross_attention_output = self.cross_attention.forward_with_bias(
                &normed_hidden,
                encoder_hidden,
                encoder_hidden,
                cross_attention_mask,
                None,
            )?;
            let cross_attention_output = self.dropout.forward(&cross_attention_output)?;
            hidden_states.add(&cross_attention_output)?
        } else {
            hidden_states
        };

        // Feed forward
        let normed_hidden = self.layer_norm_ff.forward(&hidden_states)?;
        let feed_forward_output = self.feed_forward.forward(&normed_hidden)?;
        let feed_forward_output = self.dropout.forward(&feed_forward_output)?;
        hidden_states.add(&feed_forward_output)
    }
}

impl Module for T5DecoderLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_encoder_hidden(input, None, None, None)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.self_attention.parameters() {
            params.insert(format!("self_attention.{}", name), param);
        }
        for (name, param) in self.cross_attention.parameters() {
            params.insert(format!("cross_attention.{}", name), param);
        }
        for (name, param) in self.feed_forward.parameters() {
            params.insert(format!("feed_forward.{}", name), param);
        }
        for (name, param) in self.layer_norm_self_attn.parameters() {
            params.insert(format!("layer_norm_self_attn.{}", name), param);
        }
        for (name, param) in self.layer_norm_cross_attn.parameters() {
            params.insert(format!("layer_norm_cross_attn.{}", name), param);
        }
        for (name, param) in self.layer_norm_ff.parameters() {
            params.insert(format!("layer_norm_ff.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        self.self_attention.train();
        self.cross_attention.train();
        self.feed_forward.train();
        self.layer_norm_self_attn.train();
        self.layer_norm_cross_attn.train();
        self.layer_norm_ff.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.self_attention.eval();
        self.cross_attention.eval();
        self.feed_forward.eval();
        self.layer_norm_self_attn.eval();
        self.layer_norm_cross_attn.eval();
        self.layer_norm_ff.eval();
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.self_attention.to_device(device)?;
        self.cross_attention.to_device(device)?;
        self.feed_forward.to_device(device)?;
        self.layer_norm_self_attn.to_device(device)?;
        self.layer_norm_cross_attn.to_device(device)?;
        self.layer_norm_ff.to_device(device)?;
        Ok(())
    }
}

/// T5 encoder stack
pub struct T5Encoder {
    layers: Vec<T5EncoderLayer>,
    final_layer_norm: T5LayerNorm,
    dropout: Dropout,
    is_training: bool,
}

impl T5Encoder {
    pub fn new(config: &TextModelConfig, device: DeviceType) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            layers.push(T5EncoderLayer::new(config, device)?);
        }

        Ok(Self {
            layers,
            final_layer_norm: T5LayerNorm::new(config.hidden_dim, config.layer_norm_eps),
            dropout: Dropout::new(config.dropout),
            is_training: true,
        })
    }

    pub fn forward_with_mask(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = input.clone();

        for layer in &self.layers {
            hidden_states = layer.forward_with_mask(&hidden_states, attention_mask)?;
        }

        // Final layer norm
        hidden_states = self.final_layer_norm.forward(&hidden_states)?;
        self.dropout.forward(&hidden_states)
    }
}

impl Module for T5Encoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden_states = input.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Final layer norm
        hidden_states = self.final_layer_norm.forward(&hidden_states)?;
        self.dropout.forward(&hidden_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }

        for (name, param) in self.final_layer_norm.parameters() {
            params.insert(format!("final_layer_norm.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        for layer in &mut self.layers {
            layer.train();
        }
        self.final_layer_norm.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
        self.final_layer_norm.eval();
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        self.final_layer_norm.to_device(device)?;
        Ok(())
    }
}

/// T5 decoder stack
pub struct T5Decoder {
    layers: Vec<T5DecoderLayer>,
    final_layer_norm: T5LayerNorm,
    dropout: Dropout,
    is_training: bool,
}

impl T5Decoder {
    pub fn new(config: &TextModelConfig, device: DeviceType) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            layers.push(T5DecoderLayer::new(config, device)?);
        }

        Ok(Self {
            layers,
            final_layer_norm: T5LayerNorm::new(config.hidden_dim, config.layer_norm_eps),
            dropout: Dropout::new(config.dropout),
            is_training: true,
        })
    }

    pub fn forward_with_encoder_hidden(
        &self,
        input: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        self_attention_mask: Option<&Tensor>,
        cross_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = input.clone();

        for layer in &self.layers {
            hidden_states = layer.forward_with_encoder_hidden(
                &hidden_states,
                encoder_hidden_states,
                self_attention_mask,
                cross_attention_mask,
            )?;
        }

        // Final layer norm
        hidden_states = self.final_layer_norm.forward(&hidden_states)?;
        self.dropout.forward(&hidden_states)
    }
}

impl Module for T5Decoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_encoder_hidden(input, None, None, None)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }

        for (name, param) in self.final_layer_norm.parameters() {
            params.insert(format!("final_layer_norm.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        for layer in &mut self.layers {
            layer.train();
        }
        self.final_layer_norm.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
        self.final_layer_norm.eval();
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        self.final_layer_norm.to_device(device)?;
        Ok(())
    }
}

/// T5 main model (encoder-decoder)
pub struct T5Model {
    shared_embeddings: Embedding,
    encoder: T5Encoder,
    decoder: T5Decoder,
    config: TextModelConfig,
    is_training: bool,
}

impl T5Model {
    pub fn new(config: TextModelConfig, device: DeviceType) -> Result<Self> {
        Ok(Self {
            shared_embeddings: Embedding::new(config.vocab_size, config.hidden_dim),
            encoder: T5Encoder::new(&config, device)?,
            decoder: T5Decoder::new(&config, device)?,
            config,
            is_training: true,
        })
    }

    pub fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let input_embeddings = self.shared_embeddings.forward(input_ids)?;

        // Convert attention mask to attention bias if provided
        let attention_mask = if let Some(mask) = attention_mask {
            // Convert binary mask (1 for attend, 0 for not attend) to attention bias
            // (0 for attend, -inf for not attend)
            let inverted_mask = mask.sub_scalar(1.0)?.mul_scalar(-1.0)?;
            let attention_bias = inverted_mask.mul_scalar(f32::NEG_INFINITY)?;
            Some(attention_bias)
        } else {
            None
        };

        self.encoder
            .forward_with_mask(&input_embeddings, attention_mask.as_ref())
    }

    pub fn decode(
        &self,
        decoder_input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        decoder_attention_mask: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let decoder_embeddings = self.shared_embeddings.forward(decoder_input_ids)?;
        self.decoder.forward_with_encoder_hidden(
            &decoder_embeddings,
            Some(encoder_hidden_states),
            decoder_attention_mask,
            encoder_attention_mask,
        )
    }

    pub fn forward_encoder_decoder(
        &self,
        input_ids: &Tensor,
        decoder_input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let encoder_outputs = self.encode(input_ids, attention_mask)?;
        let decoder_outputs = self.decode(
            decoder_input_ids,
            &encoder_outputs,
            decoder_attention_mask,
            attention_mask,
        )?;
        Ok((encoder_outputs, decoder_outputs))
    }
}

impl Module for T5Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For basic forward pass, just use encoder
        self.encode(input, None)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.shared_embeddings.parameters() {
            params.insert(format!("shared.{}", name), param);
        }
        for (name, param) in self.encoder.parameters() {
            params.insert(format!("encoder.{}", name), param);
        }
        for (name, param) in self.decoder.parameters() {
            params.insert(format!("decoder.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        self.shared_embeddings.train();
        self.encoder.train();
        self.decoder.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.shared_embeddings.eval();
        self.encoder.eval();
        self.decoder.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.shared_embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        self.decoder.to_device(device)?;
        Ok(())
    }
}

impl TextModel for T5Model {
    fn name(&self) -> &str {
        "T5"
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn max_seq_length(&self) -> usize {
        self.config.max_position_embeddings
    }
}

/// T5 for conditional generation (with language modeling head)
pub struct T5ForConditionalGeneration {
    transformer: T5Model,
    lm_head: Linear,
    is_training: bool,
}

impl T5ForConditionalGeneration {
    pub fn new(config: TextModelConfig, device: DeviceType) -> Result<Self> {
        Ok(Self {
            transformer: T5Model::new(config.clone(), device)?,
            lm_head: Linear::new(config.hidden_dim, config.vocab_size, false), // T5 doesn't use bias in lm_head
            is_training: true,
        })
    }

    pub fn generate(
        &self,
        input_ids: &Tensor,
        decoder_start_token_id: i32,
        max_length: usize,
        _num_beams: usize,
        _temperature: f32,
    ) -> Result<Tensor> {
        // Encode input
        let encoder_outputs = self.transformer.encode(input_ids, None)?;

        // Initialize decoder input with start token
        let batch_size = input_ids.size(0)?;
        let mut decoder_input_ids = full(&[batch_size, 1], decoder_start_token_id as f32);

        // Generate tokens autoregressively
        for _ in 1..max_length {
            let decoder_outputs =
                self.transformer
                    .decode(&decoder_input_ids, &encoder_outputs, None, None)?;

            // Get logits for next token prediction
            let logits = self.lm_head.forward(&decoder_outputs)?;
            let seq_len = logits.size(1)? as i64;
            let next_token_logits = logits.narrow(1, seq_len - 1, 1).unwrap();

            // For now, just take argmax (greedy decoding)
            let next_token = next_token_logits.argmax(Some(-1)).unwrap();

            // Concatenate next token to decoder input
            // Simplified implementation - proper tensor concatenation needed
            break; // Placeholder until proper concatenation is implemented
        }

        Ok(decoder_input_ids)
    }
}

impl Module for T5ForConditionalGeneration {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let outputs = self.transformer.forward(input)?;
        self.lm_head.forward(&outputs)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.transformer.parameters() {
            params.insert(format!("transformer.{}", name), param);
        }
        for (name, param) in self.lm_head.parameters() {
            params.insert(format!("lm_head.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        self.transformer.train();
        self.lm_head.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.transformer.eval();
        self.lm_head.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.transformer.to_device(device)?;
        self.lm_head.to_device(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_t5_layer_norm() {
        let mut layer_norm = T5LayerNorm::new(768, 1e-6);
        let input: Tensor<f32> = randn(&[2, 10, 768]);
        let output = layer_norm.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 10, 768]);
    }

    #[test]
    fn test_t5_model_creation() {
        let config = TextModelConfig::t5_small();
        let model = T5Model::new(config, DeviceType::Cpu).unwrap();
        assert_eq!(model.name(), "T5");
        assert_eq!(model.vocab_size(), 32128);
        assert_eq!(model.hidden_dim(), 512);
    }

    #[test]
    fn test_t5_encoder_forward() {
        let config = TextModelConfig::t5_small();
        let mut encoder = T5Encoder::new(&config, DeviceType::Cpu).unwrap();
        let input: Tensor<f32> = randn(&[2, 10, 512]);
        let output = encoder.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 10, 512]);
    }

    #[test]
    fn test_t5_decoder_forward() {
        let config = TextModelConfig::t5_small();
        let mut decoder = T5Decoder::new(&config, DeviceType::Cpu).unwrap();
        let input: Tensor<f32> = randn(&[2, 8, 512]);
        let encoder_hidden: Tensor<f32> = randn(&[2, 10, 512]);
        let output = decoder
            .forward_with_encoder_hidden(&input, Some(&encoder_hidden), None, None)
            .unwrap();
        assert_eq!(output.shape().dims(), &[2, 8, 512]);
    }

    #[test]
    fn test_t5_conditional_generation_creation() {
        let config = TextModelConfig::t5_small();
        let model = T5ForConditionalGeneration::new(config, DeviceType::Cpu).unwrap();
        // Test that model can be created without panicking
        assert!(model.training());
    }
}
