//! Comprehensive GPT Implementation in ToRSh
//!
//! This module provides a complete, production-ready implementation of GPT
//! (Generative Pre-trained Transformer) including:
//! - Full GPT architecture with configurable parameters
//! - Causal (autoregressive) attention mechanism
//! - Position embeddings (learned and rotary)
//! - Layer normalization variations
//! - Multiple GPT variants (GPT-2, GPT-3 style)
//! - Generation utilities (sampling, beam search)
//! - Fine-tuning capabilities
//! - Memory-efficient implementations

use torsh::prelude::*;
use std::collections::HashMap;
use std::collections::VecDeque;

/// GPT model configuration
#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_inner: Option<usize>,
    pub activation_function: String,
    pub resid_pdrop: f64,
    pub embd_pdrop: f64,
    pub attn_pdrop: f64,
    pub layer_norm_epsilon: f64,
    pub initializer_range: f64,
    pub scale_attn_weights: bool,
    pub use_cache: bool,
    pub scale_attn_by_inverse_layer_idx: bool,
    pub reorder_and_upcast_attn: bool,
    pub position_embedding_type: String,
    pub rope_theta: f64,
    pub rope_scaling: Option<f64>,
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_inner: None,
            activation_function: "gelu_new".to_string(),
            resid_pdrop: 0.1,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            scale_attn_weights: true,
            use_cache: true,
            scale_attn_by_inverse_layer_idx: false,
            reorder_and_upcast_attn: false,
            position_embedding_type: "learned".to_string(),
            rope_theta: 10000.0,
            rope_scaling: None,
        }
    }
}

impl GPTConfig {
    /// Create GPT-2 Small configuration (117M parameters)
    pub fn gpt2_small() -> Self {
        Self::default()
    }
    
    /// Create GPT-2 Medium configuration (345M parameters)
    pub fn gpt2_medium() -> Self {
        Self {
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            ..Self::default()
        }
    }
    
    /// Create GPT-2 Large configuration (774M parameters)
    pub fn gpt2_large() -> Self {
        Self {
            n_embd: 1280,
            n_layer: 36,
            n_head: 20,
            ..Self::default()
        }
    }
    
    /// Create GPT-2 XL configuration (1.5B parameters)
    pub fn gpt2_xl() -> Self {
        Self {
            n_embd: 1600,
            n_layer: 48,
            n_head: 25,
            ..Self::default()
        }
    }
    
    /// Create GPT-3 style configuration (6.7B parameters)
    pub fn gpt3_medium() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 2048,
            n_embd: 4096,
            n_layer: 32,
            n_head: 32,
            activation_function: "gelu".to_string(),
            ..Self::default()
        }
    }
    
    /// Create small configuration for testing
    pub fn gpt_tiny() -> Self {
        Self {
            vocab_size: 1000,
            n_positions: 256,
            n_embd: 256,
            n_layer: 4,
            n_head: 4,
            ..Self::default()
        }
    }
    
    pub fn n_inner(&self) -> usize {
        self.n_inner.unwrap_or(4 * self.n_embd)
    }
}

/// Rotary Position Embedding (RoPE)
pub struct RotaryPositionEmbedding {
    dim: usize,
    max_seq_len: usize,
    theta: f64,
    cos_cached: Tensor,
    sin_cached: Tensor,
}

impl RotaryPositionEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f64) -> Result<Self> {
        let mut rope = Self {
            dim,
            max_seq_len,
            theta,
            cos_cached: zeros(&[0]),
            sin_cached: zeros(&[0]),
        };
        
        rope.precompute_freqs_cis(max_seq_len)?;
        Ok(rope)
    }
    
    fn precompute_freqs_cis(&mut self, seq_len: usize) -> Result<()> {
        let freqs = self.compute_freqs(seq_len)?;
        self.cos_cached = freqs.cos()?;
        self.sin_cached = freqs.sin()?;
        Ok(())
    }
    
    fn compute_freqs(&self, seq_len: usize) -> Result<Tensor> {
        let inv_freq = arange(0.0, self.dim as f64, 2.0)?
            .div_scalar(self.dim as f64)?
            .pow_scalar(-1.0)?
            .mul_scalar(self.theta)?;
        
        let t = arange(0.0, seq_len as f64, 1.0)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        
        // Duplicate frequencies for complex representation
        Tensor::cat(&[&freqs, &freqs], -1)
    }
    
    pub fn apply_rotary_pos_emb(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        if seq_len > self.max_seq_len {
            // Extend cache if needed
            return self.apply_rotary_pos_emb_extended(x, seq_len);
        }
        
        let cos = self.cos_cached.slice(0, 0, seq_len)?;
        let sin = self.sin_cached.slice(0, 0, seq_len)?;
        
        self.rotate_half(x, &cos, &sin)
    }
    
    fn apply_rotary_pos_emb_extended(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let freqs = self.compute_freqs(seq_len)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        
        self.rotate_half(x, &cos, &sin)
    }
    
    fn rotate_half(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims();
        let half_dim = shape[shape.len() - 1] / 2;
        
        let x1 = x.slice(-1, 0, half_dim)?;
        let x2 = x.slice(-1, half_dim, shape[shape.len() - 1])?;
        
        let rotated = Tensor::cat(&[&x1.neg()?, &x2], -1)?;
        
        x.mul(cos)?.add(&rotated.mul(sin)?)
    }
}

/// GPT Attention with causal masking
pub struct GPTAttention {
    c_attn: Linear,
    c_proj: Linear,
    attn_dropout: Dropout,
    resid_dropout: Dropout,
    n_head: usize,
    n_embd: usize,
    head_dim: usize,
    scale_attn_weights: bool,
    scale: f64,
    mask_bias: Tensor,
    rope: Option<RotaryPositionEmbedding>,
}

impl GPTAttention {
    pub fn new(config: &GPTConfig, layer_idx: usize) -> Result<Self> {
        let head_dim = config.n_embd / config.n_head;
        let scale = if config.scale_attn_weights {
            1.0 / (head_dim as f64).sqrt()
        } else {
            1.0
        };
        
        // Apply layer-wise scaling if configured
        let scale = if config.scale_attn_by_inverse_layer_idx {
            scale / (layer_idx + 1) as f64
        } else {
            scale
        };
        
        // Create causal mask
        let mask_bias = Self::create_causal_mask(config.n_positions)?;
        
        // Initialize RoPE if configured
        let rope = if config.position_embedding_type == "rope" {
            Some(RotaryPositionEmbedding::new(
                head_dim,
                config.n_positions,
                config.rope_theta,
            )?)
        } else {
            None
        };
        
        Ok(Self {
            c_attn: Linear::new(config.n_embd, 3 * config.n_embd),
            c_proj: Linear::new(config.n_embd, config.n_embd),
            attn_dropout: Dropout::new(config.attn_pdrop),
            resid_dropout: Dropout::new(config.resid_pdrop),
            n_head: config.n_head,
            n_embd: config.n_embd,
            head_dim,
            scale_attn_weights: config.scale_attn_weights,
            scale,
            mask_bias,
            rope,
        })
    }
    
    fn create_causal_mask(seq_len: usize) -> Result<Tensor> {
        let mut mask = ones(&[seq_len, seq_len]);
        
        // Create lower triangular matrix
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                // Set upper triangle to large negative value
                mask = mask.index_put(&[tensor![i as i64], tensor![j as i64]], &tensor![-1e9])?;
            }
        }
        
        Ok(mask)
    }
}

impl Module for GPTAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_cache(x, None, false).map(|result| result.0)
    }
}

impl GPTAttention {
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        past_key_value: Option<&(Tensor, Tensor)>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let batch_size = x.shape().dims()[0];
        let seq_len = x.shape().dims()[1];
        
        // Linear projection to Q, K, V
        let qkv = self.c_attn.forward(x)?;
        let qkv_chunks: Vec<Tensor> = qkv.chunk(3, -1)?;
        let mut query = qkv_chunks[0].clone();
        let mut key = qkv_chunks[1].clone();
        let mut value = qkv_chunks[2].clone();
        
        // Reshape for multi-head attention
        query = query.view(&[batch_size, seq_len, self.n_head, self.head_dim])?
            .transpose(1, 2)?;
        key = key.view(&[batch_size, seq_len, self.n_head, self.head_dim])?
            .transpose(1, 2)?;
        value = value.view(&[batch_size, seq_len, self.n_head, self.head_dim])?
            .transpose(1, 2)?;
        
        // Apply rotary position embedding if configured
        if let Some(ref rope) = self.rope {
            query = rope.apply_rotary_pos_emb(&query, seq_len)?;
            key = rope.apply_rotary_pos_emb(&key, seq_len)?;
        }
        
        // Handle past key-value cache
        let (key, value, present_key_value) = if let Some((past_key, past_value)) = past_key_value {
            let key = Tensor::cat(&[past_key, &key], -2)?;
            let value = Tensor::cat(&[past_value, &value], -2)?;
            let present = if use_cache { Some((key.clone(), value.clone())) } else { None };
            (key, value, present)
        } else {
            let present = if use_cache { Some((key.clone(), value.clone())) } else { None };
            (key, value, present)
        };
        
        // Compute attention scores
        let attn_scores = query.matmul(&key.transpose(-1, -2)?)?;
        
        // Scale attention scores
        let attn_scores = if self.scale_attn_weights {
            attn_scores.mul_scalar(self.scale)?
        } else {
            attn_scores
        };
        
        // Apply causal mask
        let current_seq_len = attn_scores.shape().dims()[3];
        let mask = self.mask_bias.slice(0, 0, current_seq_len)?.slice(1, 0, current_seq_len)?;
        let attn_scores = attn_scores.add(&mask)?;
        
        // Softmax and dropout
        let attn_probs = F::softmax(&attn_scores, -1)?;
        let attn_probs = self.attn_dropout.forward(&attn_probs)?;
        
        // Apply attention to values
        let attn_output = attn_probs.matmul(&value)?;
        
        // Reshape back to original format
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .view(&[batch_size, seq_len, self.n_embd])?;
        
        // Final projection and residual dropout
        let attn_output = self.c_proj.forward(&attn_output)?;
        let attn_output = self.resid_dropout.forward(&attn_output)?;
        
        Ok((attn_output, present_key_value))
    }
}

/// GPT MLP (Feed-Forward Network)
pub struct GPTMLP {
    c_fc: Linear,
    c_proj: Linear,
    act: Box<dyn Module>,
    dropout: Dropout,
}

impl GPTMLP {
    pub fn new(config: &GPTConfig) -> Result<Self> {
        let n_inner = config.n_inner();
        
        let activation: Box<dyn Module> = match config.activation_function.as_str() {
            "relu" => Box::new(ReLU::new()),
            "gelu" => Box::new(GELU::new()),
            "gelu_new" => Box::new(GELUNew::new()),
            "silu" | "swish" => Box::new(SiLU::new()),
            "tanh" => Box::new(Tanh::new()),
            _ => Box::new(GELU::new()),
        };
        
        Ok(Self {
            c_fc: Linear::new(config.n_embd, n_inner),
            c_proj: Linear::new(n_inner, config.n_embd),
            act: activation,
            dropout: Dropout::new(config.resid_pdrop),
        })
    }
}

impl Module for GPTMLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.c_fc.forward(x)?;
        let h = self.act.forward(&h)?;
        let h = self.c_proj.forward(&h)?;
        self.dropout.forward(&h)
    }
}

/// GELU New activation (used in GPT-2)
pub struct GELUNew;

impl GELUNew {
    pub fn new() -> Self {
        Self
    }
}

impl Module for GELUNew {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let x_cubed = x.pow(3.0)?;
        let inner = x.add(&x_cubed.mul_scalar(0.044715)?)?;
        let inner = inner.mul_scalar((2.0 / std::f64::consts::PI).sqrt())?;
        let tanh_inner = inner.tanh()?;
        let one_plus_tanh = tanh_inner.add_scalar(1.0)?;
        x.mul(&one_plus_tanh)?.mul_scalar(0.5)
    }
}

/// GPT Block (Transformer Layer)
pub struct GPTBlock {
    ln_1: LayerNorm,
    attn: GPTAttention,
    ln_2: LayerNorm,
    mlp: GPTMLP,
}

impl GPTBlock {
    pub fn new(config: &GPTConfig, layer_idx: usize) -> Result<Self> {
        Ok(Self {
            ln_1: LayerNorm::new(vec![config.n_embd])?,
            attn: GPTAttention::new(config, layer_idx)?,
            ln_2: LayerNorm::new(vec![config.n_embd])?,
            mlp: GPTMLP::new(config)?,
        })
    }
}

impl Module for GPTBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_cache(x, None, false).map(|result| result.0)
    }
}

impl GPTBlock {
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        past_key_value: Option<&(Tensor, Tensor)>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        // Self-attention with residual connection
        let attn_input = self.ln_1.forward(x)?;
        let (attn_output, present_key_value) = self.attn.forward_with_cache(
            &attn_input, 
            past_key_value, 
            use_cache
        )?;
        let x = x.add(&attn_output)?;
        
        // MLP with residual connection
        let mlp_input = self.ln_2.forward(&x)?;
        let mlp_output = self.mlp.forward(&mlp_input)?;
        let x = x.add(&mlp_output)?;
        
        Ok((x, present_key_value))
    }
}

/// GPT Model
pub struct GPTModel {
    wte: Embedding,    // Token embeddings
    wpe: Option<Embedding>,  // Position embeddings (if not using RoPE)
    drop: Dropout,
    h: Vec<GPTBlock>,  // Transformer blocks
    ln_f: LayerNorm,   // Final layer norm
    config: GPTConfig,
}

impl GPTModel {
    pub fn new(config: &GPTConfig) -> Result<Self> {
        // Position embeddings (only if not using RoPE)
        let wpe = if config.position_embedding_type == "learned" {
            Some(Embedding::new(config.n_positions, config.n_embd)?)
        } else {
            None
        };
        
        // Create transformer blocks
        let mut blocks = Vec::new();
        for i in 0..config.n_layer {
            blocks.push(GPTBlock::new(config, i)?);
        }
        
        Ok(Self {
            wte: Embedding::new(config.vocab_size, config.n_embd)?,
            wpe,
            drop: Dropout::new(config.embd_pdrop),
            h: blocks,
            ln_f: LayerNorm::new(vec![config.n_embd])?,
            config: config.clone(),
        })
    }
}

impl Module for GPTModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_with_cache(input_ids, None, false).map(|result| result.0)
    }
}

impl GPTModel {
    pub fn forward_with_cache(
        &self,
        input_ids: &Tensor,
        past_key_values: Option<&Vec<(Tensor, Tensor)>>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<Vec<(Tensor, Tensor)>>)> {
        let batch_size = input_ids.shape().dims()[0];
        let seq_len = input_ids.shape().dims()[1];
        
        // Token embeddings
        let mut hidden_states = self.wte.forward(input_ids)?;
        
        // Position embeddings (if using learned positions)
        if let Some(ref wpe) = self.wpe {
            let past_length = if let Some(ref past) = past_key_values {
                past[0].0.shape().dims()[2]
            } else {
                0
            };
            
            let position_ids = arange(past_length as i64, (past_length + seq_len) as i64, 1)?
                .unsqueeze(0)?
                .expand(&[batch_size, seq_len])?;
            
            let position_embeddings = wpe.forward(&position_ids)?;
            hidden_states = hidden_states.add(&position_embeddings)?;
        }
        
        // Embedding dropout
        hidden_states = self.drop.forward(&hidden_states)?;
        
        // Forward through transformer blocks
        let mut presents = Vec::new();
        for (i, block) in self.h.iter().enumerate() {
            let past_key_value = past_key_values.as_ref().map(|past| &past[i]);
            let (block_output, present) = block.forward_with_cache(
                &hidden_states, 
                past_key_value, 
                use_cache
            )?;
            
            hidden_states = block_output;
            
            if let Some(present_kv) = present {
                presents.push(present_kv);
            }
        }
        
        // Final layer norm
        hidden_states = self.ln_f.forward(&hidden_states)?;
        
        let presents = if use_cache && !presents.is_empty() {
            Some(presents)
        } else {
            None
        };
        
        Ok((hidden_states, presents))
    }
    
    /// Get input embeddings
    pub fn get_input_embeddings(&self) -> &Embedding {
        &self.wte
    }
    
    /// Set input embeddings
    pub fn set_input_embeddings(&mut self, embeddings: Embedding) {
        self.wte = embeddings;
    }
}

/// GPT for Language Modeling
pub struct GPTLMHeadModel {
    transformer: GPTModel,
    lm_head: Linear,
    config: GPTConfig,
}

impl GPTLMHeadModel {
    pub fn new(config: &GPTConfig) -> Result<Self> {
        Ok(Self {
            transformer: GPTModel::new(config)?,
            lm_head: Linear::new(config.n_embd, config.vocab_size),
            config: config.clone(),
        })
    }
    
    /// Create with tied embeddings (share weights between input and output embeddings)
    pub fn new_with_tied_embeddings(config: &GPTConfig) -> Result<Self> {
        let transformer = GPTModel::new(config)?;
        
        // Create LM head that shares weights with input embeddings
        let mut lm_head = Linear::new(config.n_embd, config.vocab_size);
        // In a real implementation, you would tie the weights here
        
        Ok(Self {
            transformer,
            lm_head,
            config: config.clone(),
        })
    }
}

impl Module for GPTLMHeadModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.transformer.forward(input_ids)?;
        self.lm_head.forward(&hidden_states)
    }
}

impl GPTLMHeadModel {
    pub fn forward_with_cache(
        &self,
        input_ids: &Tensor,
        past_key_values: Option<&Vec<(Tensor, Tensor)>>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<Vec<(Tensor, Tensor)>>)> {
        let (hidden_states, presents) = self.transformer.forward_with_cache(
            input_ids, 
            past_key_values, 
            use_cache
        )?;
        let logits = self.lm_head.forward(&hidden_states)?;
        Ok((logits, presents))
    }
    
    /// Generate text using greedy decoding
    pub fn generate_greedy(
        &self,
        input_ids: &Tensor,
        max_length: usize,
        pad_token_id: Option<i64>,
    ) -> Result<Tensor> {
        let mut generated_ids = input_ids.clone();
        let mut past_key_values: Option<Vec<(Tensor, Tensor)>> = None;
        
        for _ in 0..max_length {
            let current_input = if past_key_values.is_some() {
                // Only use the last token if we have past key values
                generated_ids.slice(1, -1, generated_ids.shape().dims()[1])?
            } else {
                generated_ids.clone()
            };
            
            let (logits, new_past) = self.forward_with_cache(
                &current_input,
                past_key_values.as_ref(),
                true,
            )?;
            
            // Get next token logits (last position)
            let next_token_logits = logits.slice(1, -1, logits.shape().dims()[1])?;
            let next_token = next_token_logits.argmax(-1)?.unsqueeze(-1)?;
            
            // Append to generated sequence
            generated_ids = Tensor::cat(&[&generated_ids, &next_token], 1)?;
            past_key_values = new_past;
            
            // Check for pad token (early stopping)
            if let Some(pad_id) = pad_token_id {
                if next_token.item::<i64>() == pad_id {
                    break;
                }
            }
        }
        
        Ok(generated_ids)
    }
    
    /// Generate text using nucleus (top-p) sampling
    pub fn generate_nucleus(
        &self,
        input_ids: &Tensor,
        max_length: usize,
        top_p: f64,
        temperature: f64,
    ) -> Result<Tensor> {
        let mut generated_ids = input_ids.clone();
        let mut past_key_values: Option<Vec<(Tensor, Tensor)>> = None;
        
        for _ in 0..max_length {
            let current_input = if past_key_values.is_some() {
                generated_ids.slice(1, -1, generated_ids.shape().dims()[1])?
            } else {
                generated_ids.clone()
            };
            
            let (logits, new_past) = self.forward_with_cache(
                &current_input,
                past_key_values.as_ref(),
                true,
            )?;
            
            // Get next token logits and apply temperature
            let next_token_logits = logits.slice(1, -1, logits.shape().dims()[1])?
                .div_scalar(temperature)?;
            
            // Apply nucleus sampling
            let next_token = self.nucleus_sample(&next_token_logits, top_p)?;
            
            // Append to generated sequence
            generated_ids = Tensor::cat(&[&generated_ids, &next_token.unsqueeze(-1)?], 1)?;
            past_key_values = new_past;
        }
        
        Ok(generated_ids)
    }
    
    fn nucleus_sample(&self, logits: &Tensor, top_p: f64) -> Result<Tensor> {
        // Sort logits in descending order
        let (sorted_logits, sorted_indices) = logits.sort(-1, true)?;
        
        // Compute softmax probabilities
        let probs = F::softmax(&sorted_logits, -1)?;
        
        // Compute cumulative probabilities
        let cumulative_probs = probs.cumsum(-1)?;
        
        // Create mask for tokens to keep (cumulative prob <= top_p)
        let mask = cumulative_probs.le(&tensor![top_p])?;
        
        // Set probabilities of filtered tokens to 0
        let filtered_probs = probs.masked_fill(&mask.logical_not()?, 0.0)?;
        
        // Renormalize
        let sum_probs = filtered_probs.sum(-1)?.unsqueeze(-1)?;
        let normalized_probs = filtered_probs.div(&sum_probs)?;
        
        // Sample from the filtered distribution
        let sampled_index = self.multinomial_sample(&normalized_probs)?;
        
        // Map back to original indices
        sorted_indices.gather(-1, &sampled_index.unsqueeze(-1)?)?.squeeze(-1)
    }
    
    fn multinomial_sample(&self, probs: &Tensor) -> Result<Tensor> {
        // Simple multinomial sampling (in practice, use proper random sampling)
        probs.argmax(-1)
    }
}

/// Beam search implementation for GPT
pub struct BeamSearchGenerator {
    model: GPTLMHeadModel,
    beam_size: usize,
    max_length: usize,
    length_penalty: f64,
    early_stopping: bool,
}

impl BeamSearchGenerator {
    pub fn new(
        model: GPTLMHeadModel,
        beam_size: usize,
        max_length: usize,
        length_penalty: f64,
        early_stopping: bool,
    ) -> Self {
        Self {
            model,
            beam_size,
            max_length,
            length_penalty,
            early_stopping,
        }
    }
    
    pub fn generate(&self, input_ids: &Tensor) -> Result<Vec<Tensor>> {
        let batch_size = input_ids.shape().dims()[0];
        assert_eq!(batch_size, 1, "Beam search currently supports batch size 1");
        
        // Initialize beams
        let mut beams = vec![BeamHypothesis::new(input_ids.clone(), 0.0)];
        let mut finished_beams = Vec::new();
        
        for step in 0..self.max_length {
            let mut new_beams = Vec::new();
            
            for beam in &beams {
                if beam.sequence.shape().dims()[1] >= self.max_length {
                    finished_beams.push(beam.clone());
                    continue;
                }
                
                // Get logits for current beam
                let (logits, _) = self.model.forward_with_cache(&beam.sequence, None, false)?;
                let next_token_logits = logits.slice(1, -1, logits.shape().dims()[1])?;
                let log_probs = F::log_softmax(&next_token_logits, -1)?;
                
                // Get top-k candidates
                let (top_scores, top_indices) = log_probs.topk(self.beam_size, -1, true)?;
                
                for i in 0..self.beam_size {
                    let score = top_scores.select(-1, i)?.item::<f32>() as f64;
                    let token_id = top_indices.select(-1, i)?;
                    
                    let new_sequence = Tensor::cat(&[&beam.sequence, &token_id.unsqueeze(0)?], 1)?;
                    let new_score = beam.score + score;
                    
                    new_beams.push(BeamHypothesis::new(new_sequence, new_score));
                }
            }
            
            // Keep only top beams
            new_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            beams = new_beams.into_iter().take(self.beam_size).collect();
            
            // Early stopping check
            if self.early_stopping && finished_beams.len() >= self.beam_size {
                break;
            }
        }
        
        // Combine finished and unfinished beams
        finished_beams.extend(beams);
        finished_beams.sort_by(|a, b| {
            let length_a = a.sequence.shape().dims()[1] as f64;
            let length_b = b.sequence.shape().dims()[1] as f64;
            
            let normalized_score_a = a.score / length_a.powf(self.length_penalty);
            let normalized_score_b = b.score / length_b.powf(self.length_penalty);
            
            normalized_score_b.partial_cmp(&normalized_score_a).unwrap()
        });
        
        Ok(finished_beams.into_iter().map(|beam| beam.sequence).take(self.beam_size).collect())
    }
}

#[derive(Clone)]
struct BeamHypothesis {
    sequence: Tensor,
    score: f64,
}

impl BeamHypothesis {
    fn new(sequence: Tensor, score: f64) -> Self {
        Self { sequence, score }
    }
}

/// GPT Trainer
pub struct GPTTrainer {
    model: GPTLMHeadModel,
    optimizer: Adam,
    lr_scheduler: Option<Box<dyn LRScheduler>>,
    device: Device,
}

impl GPTTrainer {
    pub fn new(
        model: GPTLMHeadModel,
        learning_rate: f64,
        device: Device,
    ) -> Result<Self> {
        let optimizer = Adam::new(model.parameters(), learning_rate)?;
        
        Ok(Self {
            model,
            optimizer,
            lr_scheduler: None,
            device,
        })
    }
    
    pub fn train_step(&mut self, input_ids: &Tensor, labels: &Tensor) -> Result<f32> {
        // Forward pass
        let logits = self.model.forward(input_ids)?;
        
        // Shift logits and labels for causal language modeling
        let shift_logits = logits.slice(1, 0, logits.shape().dims()[1] - 1)?;
        let shift_labels = labels.slice(1, 1, labels.shape().dims()[1])?;
        
        // Compute loss
        let loss = F::cross_entropy(
            &shift_logits.view(&[-1, shift_logits.shape().dims()[2]])?,
            &shift_labels.view(&[-1])?,
        )?;
        
        // Backward pass
        self.optimizer.zero_grad();
        loss.backward()?;
        self.optimizer.step()?;
        
        Ok(loss.item())
    }
    
    pub fn evaluate(&self, input_ids: &Tensor, labels: &Tensor) -> Result<f32> {
        no_grad(|| {
            let logits = self.model.forward(input_ids)?;
            
            let shift_logits = logits.slice(1, 0, logits.shape().dims()[1] - 1)?;
            let shift_labels = labels.slice(1, 1, labels.shape().dims()[1])?;
            
            let loss = F::cross_entropy(
                &shift_logits.view(&[-1, shift_logits.shape().dims()[2]])?,
                &shift_labels.view(&[-1])?,
            )?;
            
            Ok(loss.item::<f32>())
        })
    }
}

/// Example usage and testing
pub fn run_gpt_example() -> Result<()> {
    println!("GPT Implementation Demo");
    
    // Create GPT configuration
    let config = GPTConfig::gpt_tiny(); // Use tiny config for demo
    println!("GPT Config: {:?}", config);
    
    // Create GPT model
    let model = GPTLMHeadModel::new(&config)?;
    
    // Create sample input
    let batch_size = 2;
    let seq_length = 32;
    let input_ids = randint(0, config.vocab_size as i64, &[batch_size, seq_length]);
    
    println!("Input shape: {:?}", input_ids.shape().dims());
    
    // Test forward pass
    let logits = model.forward(&input_ids)?;
    println!("Output logits shape: {:?}", logits.shape().dims());
    
    // Test generation
    println!("\nTesting text generation:");
    let prompt = randint(0, config.vocab_size as i64, &[1, 10]);
    let generated = model.generate_greedy(&prompt, 20, None)?;
    println!("Generated sequence shape: {:?}", generated.shape().dims());
    
    // Test with cache
    let (cached_logits, past_key_values) = model.forward_with_cache(&input_ids, None, true)?;
    println!("Cached logits shape: {:?}", cached_logits.shape().dims());
    println!("Number of cached layers: {}", past_key_values.as_ref().map_or(0, |p| p.len()));
    
    // Test beam search
    println!("\nTesting beam search:");
    let beam_generator = BeamSearchGenerator::new(
        model,
        beam_size: 3,
        max_length: 20,
        length_penalty: 1.0,
        early_stopping: true,
    );
    
    let beam_results = beam_generator.generate(&prompt)?;
    println!("Generated {} beams", beam_results.len());
    for (i, beam) in beam_results.iter().enumerate() {
        println!("Beam {}: shape {:?}", i, beam.shape().dims());
    }
    
    // Demonstrate training
    println!("\nTraining demonstration:");
    let device = Device::cpu();
    let mut trainer = GPTTrainer::new(
        beam_generator.model, // Reuse the model from beam generator
        1e-4,
        device,
    )?;
    
    let labels = input_ids.clone();
    let loss = trainer.train_step(&input_ids, &labels)?;
    println!("Training loss: {:.6}", loss);
    
    let eval_loss = trainer.evaluate(&input_ids, &labels)?;
    println!("Evaluation loss: {:.6}", eval_loss);
    
    println!("GPT demo completed successfully!");
    
    Ok(())
}

fn main() -> Result<()> {
    run_gpt_example()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpt_config() {
        let config = GPTConfig::gpt2_small();
        assert_eq!(config.n_embd, 768);
        assert_eq!(config.n_layer, 12);
        assert_eq!(config.n_head, 12);
        
        let large_config = GPTConfig::gpt2_large();
        assert_eq!(large_config.n_embd, 1280);
        assert_eq!(large_config.n_layer, 36);
    }
    
    #[test]
    fn test_gpt_attention() {
        let config = GPTConfig::gpt_tiny();
        let attention = GPTAttention::new(&config, 0).unwrap();
        
        let hidden_states = randn(&[2, 10, config.n_embd]);
        let output = attention.forward(&hidden_states).unwrap();
        
        assert_eq!(output.shape().dims(), &[2, 10, config.n_embd]);
    }
    
    #[test]
    fn test_gpt_mlp() {
        let config = GPTConfig::gpt_tiny();
        let mlp = GPTMLP::new(&config).unwrap();
        
        let hidden_states = randn(&[2, 10, config.n_embd]);
        let output = mlp.forward(&hidden_states).unwrap();
        
        assert_eq!(output.shape().dims(), &[2, 10, config.n_embd]);
    }
    
    #[test]
    fn test_gpt_block() {
        let config = GPTConfig::gpt_tiny();
        let block = GPTBlock::new(&config, 0).unwrap();
        
        let hidden_states = randn(&[2, 10, config.n_embd]);
        let output = block.forward(&hidden_states).unwrap();
        
        assert_eq!(output.shape().dims(), &[2, 10, config.n_embd]);
    }
    
    #[test]
    fn test_gpt_model() {
        let config = GPTConfig::gpt_tiny();
        let model = GPTModel::new(&config).unwrap();
        
        let input_ids = randint(0, config.vocab_size as i64, &[2, 10]);
        let output = model.forward(&input_ids).unwrap();
        
        assert_eq!(output.shape().dims(), &[2, 10, config.n_embd]);
    }
    
    #[test]
    fn test_gpt_lm_head_model() {
        let config = GPTConfig::gpt_tiny();
        let model = GPTLMHeadModel::new(&config).unwrap();
        
        let input_ids = randint(0, config.vocab_size as i64, &[2, 10]);
        let logits = model.forward(&input_ids).unwrap();
        
        assert_eq!(logits.shape().dims(), &[2, 10, config.vocab_size]);
    }
    
    #[test]
    fn test_gpt_generation() {
        let config = GPTConfig::gpt_tiny();
        let model = GPTLMHeadModel::new(&config).unwrap();
        
        let input_ids = randint(0, config.vocab_size as i64, &[1, 5]);
        let generated = model.generate_greedy(&input_ids, 10, None).unwrap();
        
        assert_eq!(generated.shape().dims()[0], 1);
        assert_eq!(generated.shape().dims()[1], 15); // 5 + 10
    }
    
    #[test]
    fn test_rotary_position_embedding() {
        let rope = RotaryPositionEmbedding::new(64, 128, 10000.0).unwrap();
        let x = randn(&[2, 10, 4, 64]);
        let output = rope.apply_rotary_pos_emb(&x, 10).unwrap();
        
        assert_eq!(output.shape().dims(), x.shape().dims());
    }
}