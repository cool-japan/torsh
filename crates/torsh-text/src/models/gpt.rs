use super::transformer::{FeedForward, MultiHeadAttention};
use crate::{TextModel, TextModelConfig};
use std::collections::HashMap;
use torsh_core::{device::DeviceType, Result};
use torsh_nn::{prelude::*, Module, Parameter};
use torsh_tensor::creation::*;
use torsh_tensor::Tensor;

/// Create a causal mask for GPT-style autoregressive attention
fn create_causal_mask(seq_len: usize, device: DeviceType) -> Result<Tensor> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];

    // Fill upper triangle with negative infinity
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }

    Tensor::from_vec(mask_data, [seq_len, seq_len], device)
}

/// Apply top-k filtering to logits
fn apply_top_k_filtering(logits: &Tensor, k: usize) -> Result<Tensor> {
    let vocab_size = logits.size(-1)?;
    if k >= vocab_size {
        return Ok(logits.clone());
    }

    // Get top-k values and indices
    let (values, indices) = logits.topk(k as i64, -1, true, true)?;

    // Create mask for top-k positions
    let mut filtered_logits = Tensor::full_like(logits, f32::NEG_INFINITY)?;

    // Set top-k values
    for i in 0..k {
        let idx = indices.select(-1, i as i64);
        let val = values.select(-1, i as i64);
        filtered_logits = filtered_logits.scatter_(-1, &idx.unsqueeze(-1), &val.unsqueeze(-1))?;
    }

    Ok(filtered_logits)
}

/// Apply top-p (nucleus) filtering to logits
fn apply_top_p_filtering(logits: &Tensor, p: f32) -> Result<Tensor> {
    if p >= 1.0 {
        return Ok(logits.clone());
    }

    // Sort logits in descending order
    let (sorted_logits, sorted_indices) = logits.sort(-1, true)?;

    // Convert to probabilities
    let sorted_probs = sorted_logits.softmax(-1)?;

    // Calculate cumulative probabilities
    let cumulative_probs = sorted_probs.cumsum(-1)?;

    // Create mask for values above the threshold
    let mask = cumulative_probs.le_scalar(p)?;

    // Apply mask to logits
    let mut filtered_logits = Tensor::full_like(logits, f32::NEG_INFINITY)?;

    // This is a simplified version - in practice, you'd need to handle the indexing more carefully
    // For now, just return the sorted logits where cumulative prob <= p
    let masked_sorted_logits = sorted_logits.where_self(
        &mask,
        &Tensor::full_like(&sorted_logits, f32::NEG_INFINITY)?,
    )?;

    // Scatter back to original positions
    filtered_logits = filtered_logits.scatter_(-1, &sorted_indices, &masked_sorted_logits)?;

    Ok(filtered_logits)
}

/// GPT decoder layer
pub struct GPTDecoderLayer {
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
    is_training: bool,
}

impl GPTDecoderLayer {
    pub fn new(config: &TextModelConfig) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(
                config.hidden_dim,
                config.num_heads,
                config.attention_dropout,
                DeviceType::Cpu,
            )
            .unwrap(),
            feed_forward: FeedForward::new(
                config.hidden_dim,
                config.intermediate_dim,
                config.dropout,
                DeviceType::Cpu,
            )
            .unwrap(),
            norm1: LayerNorm::new(vec![config.hidden_dim]),
            norm2: LayerNorm::new(vec![config.hidden_dim]),
            dropout: Dropout::new(config.dropout),
            is_training: true,
        }
    }

    pub fn forward_with_mask(
        &self,
        input: &Tensor,
        causal_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Pre-norm architecture with causal mask
        let residual = input.clone();
        let hidden = self.norm1.forward(input)?;
        let attn_output = self.self_attn.forward_with_mask(&hidden, causal_mask)?;
        let attn_output = self.dropout.forward(&attn_output)?;
        let hidden = residual.add(&attn_output)?;

        let residual = hidden.clone();
        let normed = self.norm2.forward(&hidden)?;
        let ff_output = self.feed_forward.forward(&normed)?;
        let ff_output = self.dropout.forward(&ff_output)?;
        residual.add(&ff_output)
    }
}

impl Module for GPTDecoderLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Pre-norm architecture
        let residual = input.clone();
        let hidden = self.norm1.forward(input)?;
        let attn_output = self.self_attn.forward(&hidden)?;
        let attn_output = self.dropout.forward(&attn_output)?;
        let hidden = residual.add(&attn_output)?;

        let residual = hidden.clone();
        let normed = self.norm2.forward(&hidden)?;
        let ff_output = self.feed_forward.forward(&normed)?;
        let ff_output = self.dropout.forward(&ff_output)?;
        residual.add(&ff_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.self_attn.parameters() {
            params.insert(format!("self_attn.{}", name), param);
        }
        for (name, param) in self.feed_forward.parameters() {
            params.insert(format!("feed_forward.{}", name), param);
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

    fn train(&mut self) {
        self.is_training = true;
        self.self_attn.train();
        self.feed_forward.train();
        self.norm1.train();
        self.norm2.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.self_attn.eval();
        self.feed_forward.eval();
        self.norm1.eval();
        self.norm2.eval();
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.feed_forward.to_device(device)?;
        self.norm1.to_device(device)?;
        self.norm2.to_device(device)?;
        Ok(())
    }
}

/// GPT model
pub struct GPTModel {
    embeddings: Embedding,
    position_embeddings: Embedding,
    layers: Vec<GPTDecoderLayer>,
    ln_f: LayerNorm,
    dropout: Dropout,
    config: TextModelConfig,
    is_training: bool,
}

impl GPTModel {
    pub fn new(config: TextModelConfig) -> Self {
        let layers = (0..config.num_layers)
            .map(|_| GPTDecoderLayer::new(&config))
            .collect();

        Self {
            embeddings: Embedding::new(config.vocab_size, config.hidden_dim),
            position_embeddings: Embedding::new(config.max_position_embeddings, config.hidden_dim),
            layers,
            ln_f: LayerNorm::new(vec![config.hidden_dim]),
            dropout: Dropout::new(config.dropout),
            config,
            is_training: true,
        }
    }

    fn create_causal_mask(
        seq_len: usize,
        _device: torsh_core::device::DeviceType,
    ) -> Result<Tensor> {
        // Create lower triangular mask for causal attention
        let mask: Tensor<f32> = ones(&[seq_len, seq_len]);
        // For now, skip mask processing - proper implementation needed
        let mask = mask.mul_scalar(-1e9).unwrap();
        Ok(mask)
    }
}

impl Module for GPTModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let seq_len = input.size(1)?;

        // Get token embeddings
        let token_embeddings = self.embeddings.forward(input)?;

        // Get position embeddings
        // For now, use dummy position encoding - proper implementation needed
        let position_ids = input.clone(); // Workaround
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;

        // Combine embeddings
        let mut hidden_states = token_embeddings.add(&position_embeddings)?;
        hidden_states = self.dropout.forward(&hidden_states)?;

        // Create and apply causal mask
        let causal_mask = create_causal_mask(seq_len, DeviceType::Cpu)?;

        // Pass through decoder layers with causal mask
        for layer in &self.layers {
            hidden_states = layer.forward_with_mask(&hidden_states, Some(&causal_mask))?;
        }

        // Final layer norm
        self.ln_f.forward(&hidden_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.embeddings.parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }
        for (name, param) in self.position_embeddings.parameters() {
            params.insert(format!("position_embeddings.{}", name), param);
        }

        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }

        for (name, param) in self.ln_f.parameters() {
            params.insert(format!("ln_f.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        self.embeddings.train();
        self.position_embeddings.train();
        for layer in &mut self.layers {
            layer.train();
        }
        self.ln_f.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.embeddings.eval();
        self.position_embeddings.eval();
        for layer in &mut self.layers {
            layer.eval();
        }
        self.ln_f.eval();
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.embeddings.to_device(device)?;
        self.position_embeddings.to_device(device)?;
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        self.ln_f.to_device(device)?;
        Ok(())
    }
}

impl TextModel for GPTModel {
    fn name(&self) -> &str {
        "GPT"
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

/// GPT for language modeling (with LM head)
pub struct GPTForCausalLM {
    transformer: GPTModel,
    lm_head: Linear,
    is_training: bool,
}

impl GPTForCausalLM {
    pub fn new(config: TextModelConfig) -> Self {
        Self {
            transformer: GPTModel::new(config.clone()),
            lm_head: Linear::new(config.hidden_dim, config.vocab_size, true),
            is_training: true,
        }
    }

    /// Generate text autoregressively
    pub fn generate(
        &self,
        input_ids: &Tensor,
        max_length: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> Result<Tensor> {
        let current_ids = input_ids.clone();

        for _ in 0..max_length {
            // Get logits for next token
            let outputs = self.forward(&current_ids)?;
            let seq_len = outputs.size(1)? as i64;
            let mut next_token_logits = outputs.narrow(1, seq_len - 1, 1).unwrap();

            // Apply temperature
            next_token_logits = if temperature != 1.0 {
                next_token_logits.div_scalar(temperature)?
            } else {
                next_token_logits
            };

            // Apply top-k filtering if specified
            if let Some(k) = top_k {
                next_token_logits = apply_top_k_filtering(&next_token_logits, k)?;
            }

            // Apply top-p filtering if specified
            if let Some(p) = top_p {
                next_token_logits = apply_top_p_filtering(&next_token_logits, p)?;
            }

            // Sample next token (simplified - just take argmax for now)
            let probs = next_token_logits.softmax(-1)?;
            let _next_token = probs.argmax(Some(-1)).unwrap();

            // Append to sequence
            // For now, just break - proper concatenation needs implementation
            break;
        }

        Ok(current_ids)
    }
}

impl Module for GPTForCausalLM {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden_states = self.transformer.forward(input)?;
        self.lm_head.forward(&hidden_states)
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
        self.lm_head.to_device(device)?;
        Ok(())
    }
}
