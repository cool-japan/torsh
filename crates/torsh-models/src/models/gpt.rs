//! GPT (Generative Pre-trained Transformer) Models
//!
//! This module contains comprehensive implementations of GPT transformer models,
//! extracted from the massive monolithic nlp.rs file as part of Phase 14 systematic refactoring.
//!
//! # GPT Architecture Components
//!
//! - `Gpt2Config` - Model configuration with small/medium/large/xl variants
//! - `Gpt2Embeddings` - Token and position embeddings for GPT-2
//! - `Gpt2Attention` - Causal multi-head self-attention mechanism
//! - `Gpt2MLP` - Feed-forward network with GELU activation
//! - `Gpt2Block` - Complete GPT transformer block (attention + MLP)
//! - `Gpt2Model` - Complete GPT-2 base model for representation learning
//! - `Gpt2LMHeadModel` - GPT-2 with language modeling head for text generation
//!
//! # Key Features of GPT Architecture
//!
//! - **Causal attention**: Future tokens are masked during attention computation
//! - **Autoregressive generation**: Models generate text left-to-right
//! - **Position embeddings**: Learnable position embeddings for sequence modeling
//! - **Layer normalization**: Applied before attention and MLP (pre-norm architecture)
//! - **Residual connections**: Skip connections around attention and MLP blocks
//!
//! # Model Variants
//!
//! - **GPT-2 Small**: 117M parameters (768 hidden, 12 layers, 12 heads)
//! - **GPT-2 Medium**: 345M parameters (1024 hidden, 24 layers, 16 heads)
//! - **GPT-2 Large**: 762M parameters (1280 hidden, 36 layers, 20 heads)
//! - **GPT-2 XL**: 1.5B parameters (1600 hidden, 48 layers, 25 heads)
//!
//! # Usage Examples
//!
//! ```rust
//! use crate::models::gpt::*;
//!
//! // Create GPT-2 small model for generation
//! let config = Gpt2Config::gpt2_small();
//! let model = Gpt2LMHeadModel::new(config);
//!
//! // Generate text
//! let input_ids = Tensor::new(&[batch_size, seq_len])?;
//! let logits = model.forward(&input_ids)?;
//!
//! // Create different model sizes
//! let medium_model = Gpt2LMHeadModel::gpt2_medium();
//! let large_model = Gpt2LMHeadModel::gpt2_large();
//! ```

use crate::ModelResult;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};
use std::collections::HashMap;

// ========================================
// GPT-2 CONFIGURATION
// ========================================

/// Configuration for GPT-2 model
#[derive(Debug, Clone)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_embd: usize,        // Hidden size / embedding dimension
    pub n_layer: usize,       // Number of transformer layers
    pub n_head: usize,        // Number of attention heads
    pub n_ctx: usize,         // Context length / max sequence length
    pub resid_pdrop: f32,     // Residual dropout probability
    pub embd_pdrop: f32,      // Embedding dropout probability
    pub attn_pdrop: f32,      // Attention dropout probability
    pub layer_norm_epsilon: f32,
    pub initializer_range: f32,
    pub use_bias: bool,       // Whether to use bias in linear layers
    pub eos_token_id: usize,
    pub bos_token_id: usize,
    pub pad_token_id: usize,
}

impl Default for Gpt2Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_ctx: 1024,
            resid_pdrop: 0.1,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_bias: true,
            eos_token_id: 50256,
            bos_token_id: 50256,
            pad_token_id: 50256,
        }
    }
}

impl Gpt2Config {
    /// Create GPT-2 small configuration (117M parameters)
    pub fn gpt2_small() -> Self {
        Self::default()
    }

    /// Create GPT-2 medium configuration (345M parameters)
    pub fn gpt2_medium() -> Self {
        Self {
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            ..Self::default()
        }
    }

    /// Create GPT-2 large configuration (762M parameters)
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

    /// Create a custom GPT-2 configuration for experimentation
    pub fn gpt2_custom(n_embd: usize, n_layer: usize, n_head: usize, n_ctx: usize) -> Self {
        Self {
            n_embd,
            n_layer,
            n_head,
            n_ctx,
            ..Self::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.n_embd % self.n_head != 0 {
            return Err(TorshError::InvalidArgument(
                "n_embd must be divisible by n_head".to_string()
            ));
        }

        if self.n_layer == 0 {
            return Err(TorshError::InvalidArgument(
                "n_layer must be greater than 0".to_string()
            ));
        }

        if self.n_ctx == 0 {
            return Err(TorshError::InvalidArgument(
                "n_ctx must be greater than 0".to_string()
            ));
        }

        Ok(())
    }

    /// Get attention head size
    pub fn head_size(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Get model size name
    pub fn model_size_name(&self) -> &'static str {
        match (self.n_embd, self.n_layer, self.n_head) {
            (768, 12, 12) => "gpt2-small",
            (1024, 24, 16) => "gpt2-medium",
            (1280, 36, 20) => "gpt2-large",
            (1600, 48, 25) => "gpt2-xl",
            _ => "gpt2-custom",
        }
    }

    /// Estimate parameter count (approximate)
    pub fn estimated_parameters(&self) -> usize {
        // Rough parameter count estimation
        let vocab_params = self.vocab_size * self.n_embd;
        let pos_params = self.n_ctx * self.n_embd;
        let layer_params = self.n_layer * (
            4 * self.n_embd * self.n_embd +  // attention weights
            2 * self.n_embd +                // layer norm
            8 * self.n_embd * self.n_embd    // MLP weights
        );
        vocab_params + pos_params + layer_params
    }
}

// ========================================
// GPT-2 EMBEDDINGS
// ========================================

/// GPT-2 Embeddings (token + position)
pub struct Gpt2Embeddings {
    wte: Embedding,           // Word token embeddings
    wpe: Embedding,           // Position embeddings
    dropout: Dropout,
    config: Gpt2Config,
}

impl Gpt2Embeddings {
    pub fn new(config: Gpt2Config) -> Self {
        let wte = Embedding::new(config.vocab_size, config.n_embd);
        let wpe = Embedding::new(config.n_ctx, config.n_embd);
        let dropout = Dropout::new(config.embd_pdrop);

        Self {
            wte,
            wpe,
            dropout,
            config,
        }
    }

    /// Create position IDs for the input
    fn create_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.size(1)?;
        let batch_size = input_ids.size(0)?;

        if seq_length > self.config.n_ctx {
            return Err(TorshError::InvalidArgument(
                format!("Sequence length {} exceeds context length {}", seq_length, self.config.n_ctx)
            ));
        }

        let position_ids = Tensor::arange(0.0, seq_length as f32, 1.0, input_ids.device())?;
        Ok(position_ids.unsqueeze(0)?.expand(&[batch_size, seq_length])?)
    }
}

impl Module for Gpt2Embeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let position_ids = self.create_position_ids(input_ids)?;

        // Get token and position embeddings
        let token_embeddings = self.wte.forward(input_ids)?;
        let position_embeddings = self.wpe.forward(&position_ids)?;

        // Combine embeddings
        let embeddings = token_embeddings.add(&position_embeddings)?;

        // Apply dropout
        self.dropout.forward(&embeddings)
    }

    fn train(&mut self) {
        self.wte.train();
        self.wpe.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.wte.eval();
        self.wpe.eval();
        self.dropout.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.wte.parameters() {
            params.insert(format!("wte.{}", name), param);
        }

        for (name, param) in self.wpe.parameters() {
            params.insert(format!("wpe.{}", name), param);
        }

        params
    }
}

// ========================================
// GPT-2 CAUSAL ATTENTION
// ========================================

/// GPT-2 Causal Multi-Head Self-Attention
pub struct Gpt2Attention {
    c_attn: Linear,           // Combined QKV projection
    c_proj: Linear,           // Output projection
    attn_dropout: Dropout,
    resid_dropout: Dropout,
    n_head: usize,
    head_size: usize,
    scale: f32,
    config: Gpt2Config,
}

impl Gpt2Attention {
    pub fn new(config: Gpt2Config) -> Self {
        let head_size = config.n_embd / config.n_head;
        let scale = 1.0 / (head_size as f32).sqrt();

        // Single linear layer for Q, K, V projections (GPT-2 style)
        let c_attn = Linear::new(config.n_embd, 3 * config.n_embd, config.use_bias);
        let c_proj = Linear::new(config.n_embd, config.n_embd, config.use_bias);

        let attn_dropout = Dropout::new(config.attn_pdrop);
        let resid_dropout = Dropout::new(config.resid_pdrop);

        Self {
            c_attn,
            c_proj,
            attn_dropout,
            resid_dropout,
            n_head: config.n_head,
            head_size,
            scale,
            config,
        }
    }

    /// Create causal attention mask
    fn create_causal_mask(&self, seq_length: usize, device: &DeviceType) -> Result<Tensor> {
        let mut mask_data = vec![f32::NEG_INFINITY; seq_length * seq_length];

        // Set upper triangle (future positions) to -inf
        for i in 0..seq_length {
            for j in 0..=i {
                mask_data[i * seq_length + j] = 0.0;
            }
        }

        Tensor::from_data(mask_data, vec![seq_length, seq_length], *device)
    }

    /// Split tensor for multi-head attention
    fn split_heads(&self, tensor: &Tensor) -> Result<Tensor> {
        let batch_size = tensor.size(0)?;
        let seq_length = tensor.size(1)?;

        // Reshape from [batch, seq, n_embd] to [batch, seq, n_head, head_size]
        let tensor = tensor.view(&[batch_size, seq_length, self.n_head, self.head_size])?;

        // Transpose to [batch, n_head, seq, head_size]
        tensor.permute(&[0, 2, 1, 3])
    }

    /// Merge tensor from multi-head attention
    fn merge_heads(&self, tensor: &Tensor) -> Result<Tensor> {
        let batch_size = tensor.size(0)?;
        let seq_length = tensor.size(2)?;

        // Transpose from [batch, n_head, seq, head_size] to [batch, seq, n_head, head_size]
        let tensor = tensor.permute(&[0, 2, 1, 3])?;

        // Reshape to [batch, seq, n_embd]
        tensor.contiguous()?.view(&[batch_size, seq_length, self.config.n_embd])
    }
}

impl Module for Gpt2Attention {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let batch_size = hidden_states.size(0)?;
        let seq_length = hidden_states.size(1)?;

        // Compute Q, K, V in one go
        let qkv = self.c_attn.forward(hidden_states)?;

        // Split into Q, K, V
        let q = qkv.narrow(2, 0, self.config.n_embd)?;
        let k = qkv.narrow(2, self.config.n_embd, self.config.n_embd)?;
        let v = qkv.narrow(2, 2 * self.config.n_embd, self.config.n_embd)?;

        // Split heads
        let q = self.split_heads(&q)?;
        let k = self.split_heads(&k)?;
        let v = self.split_heads(&v)?;

        // Compute attention scores
        let attn_scores = q.matmul(&k.transpose(-1, -2)?)?;
        let attn_scores = attn_scores.mul_scalar(self.scale)?;

        // Apply causal mask
        let causal_mask = self.create_causal_mask(seq_length, hidden_states.device())?;
        let attn_scores = attn_scores.add(&causal_mask)?;

        // Apply softmax and dropout
        let attn_probs = attn_scores.softmax(-1)?;
        let attn_probs = self.attn_dropout.forward(&attn_probs)?;

        // Apply attention to values
        let attn_output = attn_probs.matmul(&v)?;

        // Merge heads and project
        let attn_output = self.merge_heads(&attn_output)?;
        let attn_output = self.c_proj.forward(&attn_output)?;
        let attn_output = self.resid_dropout.forward(&attn_output)?;

        Ok(attn_output)
    }

    fn train(&mut self) {
        self.c_attn.train();
        self.c_proj.train();
        self.attn_dropout.train();
        self.resid_dropout.train();
    }

    fn eval(&mut self) {
        self.c_attn.eval();
        self.c_proj.eval();
        self.attn_dropout.eval();
        self.resid_dropout.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.c_attn.parameters() {
            params.insert(format!("c_attn.{}", name), param);
        }

        for (name, param) in self.c_proj.parameters() {
            params.insert(format!("c_proj.{}", name), param);
        }

        params
    }
}

// ========================================
// GPT-2 MLP (FEED-FORWARD)
// ========================================

/// GPT-2 MLP (Multi-Layer Perceptron) / Feed-Forward Network
pub struct Gpt2MLP {
    c_fc: Linear,             // First linear layer (expand)
    c_proj: Linear,           // Second linear layer (project back)
    dropout: Dropout,
    config: Gpt2Config,
}

impl Gpt2MLP {
    pub fn new(config: Gpt2Config) -> Self {
        // GPT-2 uses 4x expansion in MLP
        let intermediate_size = 4 * config.n_embd;

        let c_fc = Linear::new(config.n_embd, intermediate_size, config.use_bias);
        let c_proj = Linear::new(intermediate_size, config.n_embd, config.use_bias);
        let dropout = Dropout::new(config.resid_pdrop);

        Self {
            c_fc,
            c_proj,
            dropout,
            config,
        }
    }
}

impl Module for Gpt2MLP {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // First linear layer + GELU activation
        let hidden_states = self.c_fc.forward(hidden_states)?;
        let hidden_states = hidden_states.gelu()?;

        // Second linear layer + dropout
        let hidden_states = self.c_proj.forward(&hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;

        Ok(hidden_states)
    }

    fn train(&mut self) {
        self.c_fc.train();
        self.c_proj.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.c_fc.eval();
        self.c_proj.eval();
        self.dropout.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.c_fc.parameters() {
            params.insert(format!("c_fc.{}", name), param);
        }

        for (name, param) in self.c_proj.parameters() {
            params.insert(format!("c_proj.{}", name), param);
        }

        params
    }
}

// ========================================
// GPT-2 TRANSFORMER BLOCK
// ========================================

/// GPT-2 Transformer Block
pub struct Gpt2Block {
    ln_1: LayerNorm,          // Layer norm before attention
    attn: Gpt2Attention,      // Multi-head attention
    ln_2: LayerNorm,          // Layer norm before MLP
    mlp: Gpt2MLP,             // Feed-forward network
}

impl Gpt2Block {
    pub fn new(config: Gpt2Config) -> Self {
        let ln_1 = LayerNorm::new(vec![config.n_embd], config.layer_norm_epsilon, true);
        let attn = Gpt2Attention::new(config.clone());
        let ln_2 = LayerNorm::new(vec![config.n_embd], config.layer_norm_epsilon, true);
        let mlp = Gpt2MLP::new(config);

        Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        }
    }
}

impl Module for Gpt2Block {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Pre-norm architecture: LayerNorm -> Attention -> Residual
        let normed_states = self.ln_1.forward(hidden_states)?;
        let attn_output = self.attn.forward(&normed_states)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        // Pre-norm architecture: LayerNorm -> MLP -> Residual
        let normed_states = self.ln_2.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed_states)?;
        hidden_states.add(&mlp_output)
    }

    fn train(&mut self) {
        self.ln_1.train();
        self.attn.train();
        self.ln_2.train();
        self.mlp.train();
    }

    fn eval(&mut self) {
        self.ln_1.eval();
        self.attn.eval();
        self.ln_2.eval();
        self.mlp.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.ln_1.parameters() {
            params.insert(format!("ln_1.{}", name), param);
        }

        for (name, param) in self.attn.parameters() {
            params.insert(format!("attn.{}", name), param);
        }

        for (name, param) in self.ln_2.parameters() {
            params.insert(format!("ln_2.{}", name), param);
        }

        for (name, param) in self.mlp.parameters() {
            params.insert(format!("mlp.{}", name), param);
        }

        params
    }
}

// ========================================
// GPT-2 BASE MODEL
// ========================================

/// GPT-2 Base Model (for representation learning)
pub struct Gpt2Model {
    config: Gpt2Config,
    wte: Embedding,           // Word token embeddings
    wpe: Embedding,           // Position embeddings
    drop: Dropout,            // Embedding dropout
    h: Vec<Gpt2Block>,        // Transformer blocks
    ln_f: LayerNorm,          // Final layer norm
}

impl Gpt2Model {
    pub fn new(config: Gpt2Config) -> Self {
        config.validate().expect("Invalid GPT-2 configuration");

        let wte = Embedding::new(config.vocab_size, config.n_embd);
        let wpe = Embedding::new(config.n_ctx, config.n_embd);
        let drop = Dropout::new(config.embd_pdrop);

        let mut h = Vec::new();
        for _ in 0..config.n_layer {
            h.push(Gpt2Block::new(config.clone()));
        }

        let ln_f = LayerNorm::new(vec![config.n_embd], config.layer_norm_epsilon, true);

        Self {
            config,
            wte,
            wpe,
            drop,
            h,
            ln_f,
        }
    }

    /// Create GPT-2 small model
    pub fn gpt2_small() -> Self {
        Self::new(Gpt2Config::gpt2_small())
    }

    /// Create GPT-2 medium model
    pub fn gpt2_medium() -> Self {
        Self::new(Gpt2Config::gpt2_medium())
    }

    /// Create GPT-2 large model
    pub fn gpt2_large() -> Self {
        Self::new(Gpt2Config::gpt2_large())
    }

    /// Create GPT-2 XL model
    pub fn gpt2_xl() -> Self {
        Self::new(Gpt2Config::gpt2_xl())
    }

    /// Get model configuration
    pub fn config(&self) -> &Gpt2Config {
        &self.config
    }

    /// Create position IDs for the input
    fn create_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.size(1)?;
        let batch_size = input_ids.size(0)?;

        if seq_length > self.config.n_ctx {
            return Err(TorshError::InvalidArgument(
                format!("Sequence length {} exceeds context length {}", seq_length, self.config.n_ctx)
            ));
        }

        let position_ids = Tensor::arange(0.0, seq_length as f32, 1.0, input_ids.device())?;
        Ok(position_ids.unsqueeze(0)?.expand(&[batch_size, seq_length])?)
    }
}

impl Module for Gpt2Model {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let position_ids = self.create_position_ids(input_ids)?;

        // Get embeddings
        let token_embeddings = self.wte.forward(input_ids)?;
        let position_embeddings = self.wpe.forward(&position_ids)?;

        // Combine embeddings and apply dropout
        let mut hidden_states = token_embeddings.add(&position_embeddings)?;
        hidden_states = self.drop.forward(&hidden_states)?;

        // Pass through transformer blocks
        for block in &self.h {
            hidden_states = block.forward(&hidden_states)?;
        }

        // Final layer norm
        self.ln_f.forward(&hidden_states)
    }

    fn train(&mut self) {
        self.wte.train();
        self.wpe.train();
        self.drop.train();
        for block in &mut self.h {
            block.train();
        }
        self.ln_f.train();
    }

    fn eval(&mut self) {
        self.wte.eval();
        self.wpe.eval();
        self.drop.eval();
        for block in &mut self.h {
            block.eval();
        }
        self.ln_f.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.wte.parameters() {
            params.insert(format!("wte.{}", name), param);
        }

        for (name, param) in self.wpe.parameters() {
            params.insert(format!("wpe.{}", name), param);
        }

        for (i, block) in self.h.iter().enumerate() {
            for (name, param) in block.parameters() {
                params.insert(format!("h.{}.{}", i, name), param);
            }
        }

        for (name, param) in self.ln_f.parameters() {
            params.insert(format!("ln_f.{}", name), param);
        }

        params
    }
}

// ========================================
// GPT-2 LANGUAGE MODEL HEAD
// ========================================

/// GPT-2 with Language Modeling Head (for text generation)
pub struct Gpt2LMHeadModel {
    transformer: Gpt2Model,
    lm_head: Linear,          // Language modeling head (tied with embeddings)
}

impl Gpt2LMHeadModel {
    pub fn new(config: Gpt2Config) -> Self {
        let transformer = Gpt2Model::new(config.clone());

        // Language modeling head - often tied with token embeddings
        let lm_head = Linear::new(config.n_embd, config.vocab_size, false);

        Self {
            transformer,
            lm_head,
        }
    }

    /// Create GPT-2 small for language modeling
    pub fn gpt2_small() -> Self {
        Self::new(Gpt2Config::gpt2_small())
    }

    /// Create GPT-2 medium for language modeling
    pub fn gpt2_medium() -> Self {
        Self::new(Gpt2Config::gpt2_medium())
    }

    /// Create GPT-2 large for language modeling
    pub fn gpt2_large() -> Self {
        Self::new(Gpt2Config::gpt2_large())
    }

    /// Create GPT-2 XL for language modeling
    pub fn gpt2_xl() -> Self {
        Self::new(Gpt2Config::gpt2_xl())
    }

    /// Get model configuration
    pub fn config(&self) -> &Gpt2Config {
        self.transformer.config()
    }

    /// Generate text (greedy decoding)
    pub fn generate_greedy(&self, input_ids: &Tensor, max_length: usize) -> Result<Tensor> {
        let mut current_ids = input_ids.clone();
        let batch_size = input_ids.size(0)?;
        let start_length = input_ids.size(1)?;

        for _ in start_length..max_length {
            // Forward pass
            let logits = self.forward(&current_ids)?;

            // Get last token logits
            let last_logits = logits.select(1, -1)?;

            // Greedy selection (argmax)
            let next_token = last_logits.argmax(-1, false)?;

            // Append to sequence
            current_ids = Tensor::cat(&[current_ids, next_token.unsqueeze(1)?], 1)?;
        }

        Ok(current_ids)
    }
}

impl Module for Gpt2LMHeadModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.transformer.forward(input_ids)?;
        let logits = self.lm_head.forward(&hidden_states)?;
        Ok(logits)
    }

    fn train(&mut self) {
        self.transformer.train();
        self.lm_head.train();
    }

    fn eval(&mut self) {
        self.transformer.eval();
        self.lm_head.eval();
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
}

// ========================================
// GPT UTILITIES AND FACTORY FUNCTIONS
// ========================================

/// GPT model factory and utilities
pub struct GptFactory;

impl GptFactory {
    /// Create a GPT-2 model from configuration
    pub fn create_model(config: Gpt2Config) -> Gpt2Model {
        Gpt2Model::new(config)
    }

    /// Create a GPT-2 language model
    pub fn create_lm_model(config: Gpt2Config) -> Gpt2LMHeadModel {
        Gpt2LMHeadModel::new(config)
    }

    /// Get available GPT model variants
    pub fn available_models() -> Vec<&'static str> {
        vec!["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    }

    /// Get model parameter counts
    pub fn model_parameters() -> Vec<(&'static str, usize)> {
        vec![
            ("gpt2-small", 117_000_000),
            ("gpt2-medium", 345_000_000),
            ("gpt2-large", 762_000_000),
            ("gpt2-xl", 1_500_000_000),
        ]
    }

    /// Create model by name
    pub fn create_by_name(model_name: &str, with_lm_head: bool) -> Result<Box<dyn Module>> {
        match model_name {
            "gpt2-small" => {
                if with_lm_head {
                    Ok(Box::new(Gpt2LMHeadModel::gpt2_small()))
                } else {
                    Ok(Box::new(Gpt2Model::gpt2_small()))
                }
            },
            "gpt2-medium" => {
                if with_lm_head {
                    Ok(Box::new(Gpt2LMHeadModel::gpt2_medium()))
                } else {
                    Ok(Box::new(Gpt2Model::gpt2_medium()))
                }
            },
            "gpt2-large" => {
                if with_lm_head {
                    Ok(Box::new(Gpt2LMHeadModel::gpt2_large()))
                } else {
                    Ok(Box::new(Gpt2Model::gpt2_large()))
                }
            },
            "gpt2-xl" => {
                if with_lm_head {
                    Ok(Box::new(Gpt2LMHeadModel::gpt2_xl()))
                } else {
                    Ok(Box::new(Gpt2Model::gpt2_xl()))
                }
            },
            _ => Err(TorshError::InvalidArgument(
                format!("Unknown GPT model: {}", model_name)
            ))
        }
    }

    /// Get configuration by model name
    pub fn get_config(model_name: &str) -> Result<Gpt2Config> {
        match model_name {
            "gpt2-small" => Ok(Gpt2Config::gpt2_small()),
            "gpt2-medium" => Ok(Gpt2Config::gpt2_medium()),
            "gpt2-large" => Ok(Gpt2Config::gpt2_large()),
            "gpt2-xl" => Ok(Gpt2Config::gpt2_xl()),
            _ => Err(TorshError::InvalidArgument(
                format!("Unknown GPT model: {}", model_name)
            ))
        }
    }
}

// ========================================
// COMPREHENSIVE TESTING FRAMEWORK
// ========================================

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::Device;

    #[test]
    fn test_gpt2_config() {
        let config = Gpt2Config::gpt2_small();
        assert_eq!(config.n_embd, 768);
        assert_eq!(config.n_layer, 12);
        assert_eq!(config.n_head, 12);
        assert_eq!(config.model_size_name(), "gpt2-small");
        assert!(config.validate().is_ok());

        let medium_config = Gpt2Config::gpt2_medium();
        assert_eq!(medium_config.n_embd, 1024);
        assert_eq!(medium_config.n_layer, 24);
        assert_eq!(medium_config.model_size_name(), "gpt2-medium");

        let large_config = Gpt2Config::gpt2_large();
        assert_eq!(large_config.n_embd, 1280);
        assert_eq!(large_config.model_size_name(), "gpt2-large");

        let xl_config = Gpt2Config::gpt2_xl();
        assert_eq!(xl_config.n_embd, 1600);
        assert_eq!(xl_config.model_size_name(), "gpt2-xl");
    }

    #[test]
    fn test_gpt2_model_creation() {
        let config = Gpt2Config::gpt2_small();
        let model = Gpt2Model::new(config);
        assert_eq!(model.config().n_embd, 768);

        let small_model = Gpt2Model::gpt2_small();
        assert_eq!(small_model.config().n_embd, 768);

        let medium_model = Gpt2Model::gpt2_medium();
        assert_eq!(medium_model.config().n_embd, 1024);
    }

    #[test]
    fn test_gpt2_lm_head_model() {
        let model = Gpt2LMHeadModel::gpt2_small();
        assert_eq!(model.config().n_embd, 768);

        let medium_model = Gpt2LMHeadModel::gpt2_medium();
        assert_eq!(medium_model.config().n_embd, 1024);
    }

    #[test]
    fn test_gpt_factory() {
        let available_models = GptFactory::available_models();
        assert!(available_models.contains(&"gpt2-small"));
        assert!(available_models.contains(&"gpt2-medium"));
        assert!(available_models.contains(&"gpt2-large"));
        assert!(available_models.contains(&"gpt2-xl"));

        // Test parameter counts
        let param_counts = GptFactory::model_parameters();
        assert_eq!(param_counts.len(), 4);

        // Test model creation by name
        let base_model = GptFactory::create_by_name("gpt2-small", false);
        assert!(base_model.is_ok());

        let lm_model = GptFactory::create_by_name("gpt2-medium", true);
        assert!(lm_model.is_ok());

        let invalid_model = GptFactory::create_by_name("invalid-model", false);
        assert!(invalid_model.is_err());
    }

    #[test]
    fn test_gpt2_head_size() {
        let config = Gpt2Config::gpt2_small();
        assert_eq!(config.head_size(), 64); // 768 / 12

        let medium_config = Gpt2Config::gpt2_medium();
        assert_eq!(medium_config.head_size(), 64); // 1024 / 16

        let large_config = Gpt2Config::gpt2_large();
        assert_eq!(large_config.head_size(), 64); // 1280 / 20
    }

    #[test]
    fn test_gpt2_parameter_estimation() {
        let config = Gpt2Config::gpt2_small();
        let estimated = config.estimated_parameters();
        assert!(estimated > 100_000_000); // Should be > 100M
        assert!(estimated < 200_000_000); // Should be < 200M

        let medium_config = Gpt2Config::gpt2_medium();
        let medium_estimated = medium_config.estimated_parameters();
        assert!(medium_estimated > estimated); // Medium should be larger
    }

    #[test]
    fn test_gpt2_forward_pass() {
        let model = Gpt2Model::gpt2_small();

        // Create dummy input
        let batch_size = 2;
        let seq_length = 10;
        let input_ids = Tensor::zeros(&[batch_size, seq_length]).unwrap();

        // Test forward pass shape
        let output = model.forward(&input_ids);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.size(0).unwrap(), batch_size);
        assert_eq!(output.size(1).unwrap(), seq_length);
        assert_eq!(output.size(2).unwrap(), 768); // n_embd
    }

    #[test]
    fn test_gpt2_lm_head_forward_pass() {
        let model = Gpt2LMHeadModel::gpt2_small();

        // Create dummy input
        let batch_size = 2;
        let seq_length = 10;
        let input_ids = Tensor::zeros(&[batch_size, seq_length]).unwrap();

        // Test forward pass shape
        let output = model.forward(&input_ids);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.size(0).unwrap(), batch_size);
        assert_eq!(output.size(1).unwrap(), seq_length);
        assert_eq!(output.size(2).unwrap(), 50257); // vocab_size
    }

    #[test]
    fn test_gpt2_config_validation() {
        let mut config = Gpt2Config::gpt2_small();

        // Valid config
        assert!(config.validate().is_ok());

        // Invalid: n_embd not divisible by n_head
        config.n_embd = 777;  // Not divisible by 12
        assert!(config.validate().is_err());

        // Invalid: zero layers
        config.n_embd = 768;
        config.n_layer = 0;
        assert!(config.validate().is_err());

        // Invalid: zero context length
        config.n_layer = 12;
        config.n_ctx = 0;
        assert!(config.validate().is_err());
    }
}

/// Re-export commonly used GPT components
pub use {
    Gpt2Config, Gpt2Model, Gpt2LMHeadModel, GptFactory,
    Gpt2Embeddings, Gpt2Attention, Gpt2MLP, Gpt2Block
};