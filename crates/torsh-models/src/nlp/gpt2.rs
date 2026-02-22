//! # GPT-2 (Generative Pre-trained Transformer 2) Implementation
//!
//! This module provides a comprehensive implementation of GPT-2 models in the ToRSh framework.
//! GPT-2 is an autoregressive language model that uses transformer decoder architecture
//! with causal self-attention for text generation and language modeling tasks.
//!
//! ## Models Included
//!
//! - **Gpt2Config**: Configuration for GPT-2 models
//! - **Gpt2Embeddings**: Token and position embeddings
//! - **Gpt2Attention**: Causal self-attention mechanism with masking
//! - **Gpt2MLP**: Feed-forward network with GELU activation
//! - **Gpt2Block**: Complete transformer decoder block
//! - **Gpt2Model**: Complete GPT-2 transformer model
//! - **Gpt2LMHeadModel**: GPT-2 with language modeling head
//!
//! ## Usage Example
//!
//! ```rust
//! use torsh_models::nlp::gpt2::{Gpt2Config, Gpt2Model, Gpt2LMHeadModel};
//! use torsh_tensor::Tensor;
//!
//! // Create GPT-2 small model
//! let mut model = Gpt2Model::gpt2_small();
//!
//! // Or create for language modeling
//! let mut lm_model = Gpt2LMHeadModel::gpt2_small_lm();
//!
//! // Forward pass
//! let input_ids = Tensor::zeros(&[1, 10]).unwrap();
//! let output = model.forward(&input_ids).unwrap();
//! ```
//!
//! ## Model Variants
//!
//! - **GPT-2 Small**: 12 layers, 768 hidden size, 12 attention heads (117M parameters)
//! - **GPT-2 Medium**: 24 layers, 1024 hidden size, 16 attention heads (345M parameters)
//! - **GPT-2 Large**: 36 layers, 1280 hidden size, 20 attention heads (762M parameters)
//! - **GPT-2 XL**: 48 layers, 1600 hidden size, 25 attention heads (1.5B parameters)
//!
//! All models follow the GPT-2 paper specifications and are compatible with
//! HuggingFace transformers library weights.

use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;
use torsh_nn::{
    activations::GELU,
    dropout::Dropout,
    embedding::Embedding,
    layers::Linear,
    layer_norm::LayerNorm,
    module::{Module, Parameter},
};

/// Configuration for GPT-2 model
///
/// This struct contains all the hyperparameters needed to define a GPT-2 model architecture.
/// It provides factory methods for standard model sizes (small, medium, large, xl).
#[derive(Debug, Clone)]
pub struct Gpt2Config {
    /// Size of the vocabulary
    pub vocab_size: usize,
    /// Size of the hidden/embedding dimension
    pub n_embd: usize,
    /// Number of transformer layers
    pub n_layer: usize,
    /// Number of attention heads
    pub n_head: usize,
    /// Maximum context length
    pub n_ctx: usize,
    /// Residual dropout probability
    pub resid_pdrop: f32,
    /// Embedding dropout probability
    pub embd_pdrop: f32,
    /// Attention dropout probability
    pub attn_pdrop: f32,
    /// Layer normalization epsilon
    pub layer_norm_epsilon: f32,
    /// Weight initialization range
    pub initializer_range: f32,
    /// Whether to use bias in linear layers
    pub use_bias: bool,
    /// End-of-sequence token ID
    pub eos_token_id: usize,
    /// Beginning-of-sequence token ID
    pub bos_token_id: usize,
    /// Padding token ID
    pub pad_token_id: usize,
}

impl Default for Gpt2Config {
    /// Returns the default GPT-2 small configuration
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
    ///
    /// # Returns
    ///
    /// A `Gpt2Config` with parameters for the small model
    pub fn gpt2_small() -> Self {
        Self::default()
    }

    /// Create GPT-2 medium configuration (345M parameters)
    ///
    /// # Returns
    ///
    /// A `Gpt2Config` with parameters for the medium model
    pub fn gpt2_medium() -> Self {
        Self {
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            ..Self::default()
        }
    }

    /// Create GPT-2 large configuration (762M parameters)
    ///
    /// # Returns
    ///
    /// A `Gpt2Config` with parameters for the large model
    pub fn gpt2_large() -> Self {
        Self {
            n_embd: 1280,
            n_layer: 36,
            n_head: 20,
            ..Self::default()
        }
    }

    /// Create GPT-2 XL configuration (1.5B parameters)
    ///
    /// # Returns
    ///
    /// A `Gpt2Config` with parameters for the XL model
    pub fn gpt2_xl() -> Self {
        Self {
            n_embd: 1600,
            n_layer: 48,
            n_head: 25,
            ..Self::default()
        }
    }

    /// Validates the configuration parameters
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err` with description if invalid
    pub fn validate(&self) -> Result<()> {
        if self.n_embd % self.n_head != 0 {
            return Err(TorshError::ModelError(
                "n_embd must be divisible by n_head".to_string()
            ));
        }
        if self.vocab_size == 0 {
            return Err(TorshError::ModelError("vocab_size must be > 0".to_string()));
        }
        if self.n_layer == 0 {
            return Err(TorshError::ModelError("n_layer must be > 0".to_string()));
        }
        if self.n_ctx == 0 {
            return Err(TorshError::ModelError("n_ctx must be > 0".to_string()));
        }
        Ok(())
    }

    /// Gets the head dimension
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Gets the intermediate size for MLP (4x the embedding dimension)
    pub fn intermediate_size(&self) -> usize {
        4 * self.n_embd
    }
}

/// GPT-2 Embeddings
///
/// Combines word embeddings and position embeddings for input representation.
/// GPT-2 uses learned positional embeddings unlike some other transformers.
pub struct Gpt2Embeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    dropout: Dropout,
    config: Gpt2Config,
}

impl Gpt2Embeddings {
    /// Creates new GPT-2 embeddings
    ///
    /// # Arguments
    ///
    /// * `config` - GPT-2 configuration
    ///
    /// # Returns
    ///
    /// New `Gpt2Embeddings` instance
    pub fn new(config: Gpt2Config) -> Self {
        let word_embeddings = Embedding::new(config.vocab_size, config.n_embd);
        let position_embeddings = Embedding::new(config.n_ctx, config.n_embd);
        let dropout = Dropout::new(config.embd_pdrop);

        Self {
            word_embeddings,
            position_embeddings,
            dropout,
            config,
        }
    }

    /// Gets the configuration
    pub fn config(&self) -> &Gpt2Config {
        &self.config
    }
}

impl Module for Gpt2Embeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.size(1)?;

        // Create position ids (0, 1, 2, ..., seq_length-1)
        let position_ids: Vec<f32> = (0..seq_length).map(|i| i as f32).collect();
        let position_tensor = Tensor::from_slice(&position_ids, &[1, seq_length])?;

        // Get embeddings
        let inputs_embeds = self.word_embeddings.forward(input_ids)?;
        let position_embeds = self.position_embeddings.forward(&position_tensor)?;

        // Sum embeddings
        let mut embeddings = inputs_embeds.add(&position_embeds)?;

        // Apply dropout
        embeddings = self.dropout.forward(&embeddings)?;

        Ok(embeddings)
    }

    fn train(&mut self) {
        self.word_embeddings.train();
        self.position_embeddings.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.word_embeddings.eval();
        self.position_embeddings.eval();
        self.dropout.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.word_embeddings.parameters());
        params.extend(self.position_embeddings.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.word_embeddings.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.word_embeddings.to_device(device)?;
        self.position_embeddings.to_device(device)?;
        Ok(())
    }
}

/// GPT-2 Self-Attention with causal masking
///
/// Implements causal (autoregressive) self-attention mechanism.
/// Uses a single linear layer for combined Q, K, V projections as in the original implementation.
pub struct Gpt2Attention {
    c_attn: Linear,        // Combined query, key, value projection
    c_proj: Linear,        // Output projection
    attn_dropout: Dropout, // Attention dropout
    resid_dropout: Dropout, // Residual dropout
    config: Gpt2Config,
}

impl Gpt2Attention {
    /// Creates new GPT-2 attention layer
    ///
    /// # Arguments
    ///
    /// * `config` - GPT-2 configuration
    ///
    /// # Returns
    ///
    /// New `Gpt2Attention` instance
    pub fn new(config: Gpt2Config) -> Self {
        let c_attn = Linear::new(config.n_embd, 3 * config.n_embd, config.use_bias);
        let c_proj = Linear::new(config.n_embd, config.n_embd, config.use_bias);
        let attn_dropout = Dropout::new(config.attn_pdrop);
        let resid_dropout = Dropout::new(config.resid_pdrop);

        Self {
            c_attn,
            c_proj,
            attn_dropout,
            resid_dropout,
            config,
        }
    }

    /// Creates causal attention mask for autoregressive generation
    ///
    /// # Arguments
    ///
    /// * `seq_length` - Sequence length for the mask
    ///
    /// # Returns
    ///
    /// Causal mask tensor with shape [seq_length, seq_length]
    fn create_causal_mask(&self, seq_length: usize) -> Result<Tensor> {
        let mut mask_data = vec![f32::NEG_INFINITY; seq_length * seq_length];

        // Set upper triangle to 0 (allowed positions), lower triangle stays -inf (masked)
        for i in 0..seq_length {
            for j in 0..=i {
                mask_data[i * seq_length + j] = 0.0;
            }
        }

        Tensor::from_slice(&mask_data, &[seq_length, seq_length])
    }

    /// Gets the configuration
    pub fn config(&self) -> &Gpt2Config {
        &self.config
    }
}

impl Module for Gpt2Attention {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let batch_size = hidden_states.size(0)?;
        let seq_length = hidden_states.size(1)?;
        let embed_dim = hidden_states.size(2)?;

        let head_dim = embed_dim / self.config.n_head;

        // Compute Q, K, V all at once
        let qkv = self.c_attn.forward(hidden_states)?;
        let qkv_reshaped = qkv.view(&[batch_size, seq_length, 3, self.config.n_head, head_dim])?;

        // Split into Q, K, V and transpose to [batch, head, seq, head_dim]
        let query = qkv_reshaped
            .slice(&[
                (0, batch_size as i64),
                (0, seq_length as i64),
                (0, 1),
                (0, self.config.n_head as i64),
                (0, head_dim as i64),
            ])?
            .squeeze_dim(2)?
            .transpose(1, 2)?;
        let key = qkv_reshaped
            .slice(&[
                (0, batch_size as i64),
                (0, seq_length as i64),
                (1, 2),
                (0, self.config.n_head as i64),
                (0, head_dim as i64),
            ])?
            .squeeze_dim(2)?
            .transpose(1, 2)?;
        let value = qkv_reshaped
            .slice(&[
                (0, batch_size as i64),
                (0, seq_length as i64),
                (2, 3),
                (0, self.config.n_head as i64),
                (0, head_dim as i64),
            ])?
            .squeeze_dim(2)?
            .transpose(1, 2)?;

        // Scaled dot-product attention with causal mask
        let scale = (head_dim as f32).sqrt();
        let scores = query.matmul(&key.transpose(-2, -1)?)?.div_scalar(scale)?;

        // Apply causal mask
        let causal_mask = self.create_causal_mask(seq_length)?;
        let masked_scores = scores.add(&causal_mask)?;

        // Apply softmax
        let attention_probs = masked_scores.softmax(-1)?;
        let attention_probs = self.attn_dropout.forward(&attention_probs)?;

        // Apply attention to values
        let context = attention_probs.matmul(&value)?;

        // Reshape back to [batch, seq, embed_dim]
        let context = context
            .transpose(1, 2)?
            .contiguous()?
            .view(&[batch_size, seq_length, embed_dim])?;

        // Output projection
        let output = self.c_proj.forward(&context)?;
        let output = self.resid_dropout.forward(&output)?;

        Ok(output)
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
        params.extend(self.c_attn.parameters());
        params.extend(self.c_proj.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.c_attn.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.c_attn.to_device(device)?;
        self.c_proj.to_device(device)?;
        Ok(())
    }
}

/// GPT-2 Feed Forward Network (MLP)
///
/// Implements the position-wise feed-forward network used in each transformer block.
/// Uses GELU activation and 4x expansion ratio as specified in the GPT-2 paper.
pub struct Gpt2MLP {
    c_fc: Linear,       // First linear layer (expansion)
    c_proj: Linear,     // Second linear layer (contraction)
    dropout: Dropout,   // Residual dropout
    activation: GELU,   // GELU activation
    config: Gpt2Config,
}

impl Gpt2MLP {
    /// Creates new GPT-2 MLP layer
    ///
    /// # Arguments
    ///
    /// * `config` - GPT-2 configuration
    ///
    /// # Returns
    ///
    /// New `Gpt2MLP` instance
    pub fn new(config: Gpt2Config) -> Self {
        let intermediate_size = 4 * config.n_embd; // GPT-2 uses 4x expansion
        let c_fc = Linear::new(config.n_embd, intermediate_size, config.use_bias);
        let c_proj = Linear::new(intermediate_size, config.n_embd, config.use_bias);
        let dropout = Dropout::new(config.resid_pdrop);
        let activation = GELU::new();

        Self {
            c_fc,
            c_proj,
            dropout,
            activation,
            config,
        }
    }

    /// Gets the configuration
    pub fn config(&self) -> &Gpt2Config {
        &self.config
    }
}

impl Module for Gpt2MLP {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.c_fc.forward(hidden_states)?;
        let hidden_states = self.activation.forward(&hidden_states)?;
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
        params.extend(self.c_fc.parameters());
        params.extend(self.c_proj.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.c_fc.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.c_fc.to_device(device)?;
        self.c_proj.to_device(device)?;
        Ok(())
    }
}

/// GPT-2 Transformer Block
///
/// A complete transformer decoder block combining layer normalization,
/// self-attention, and feed-forward networks with residual connections.
/// Uses pre-normalization architecture as in the original GPT-2.
pub struct Gpt2Block {
    ln_1: LayerNorm,  // Layer norm before attention
    attn: Gpt2Attention, // Self-attention
    ln_2: LayerNorm,  // Layer norm before MLP
    mlp: Gpt2MLP,     // Feed-forward network
}

impl Gpt2Block {
    /// Creates new GPT-2 transformer block
    ///
    /// # Arguments
    ///
    /// * `config` - GPT-2 configuration
    ///
    /// # Returns
    ///
    /// New `Gpt2Block` instance
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

    /// Gets the attention component
    pub fn attention(&self) -> &Gpt2Attention {
        &self.attn
    }

    /// Gets the MLP component
    pub fn mlp(&self) -> &Gpt2MLP {
        &self.mlp
    }
}

impl Module for Gpt2Block {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Pre-norm architecture: LayerNorm before attention
        let residual = hidden_states.clone();
        let hidden_states = self.ln_1.forward(hidden_states)?;
        let attn_output = self.attn.forward(&hidden_states)?;
        let hidden_states = residual.add(&attn_output)?; // Residual connection

        // Pre-norm architecture: LayerNorm before MLP
        let residual = hidden_states.clone();
        let hidden_states = self.ln_2.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&hidden_states)?;
        let hidden_states = residual.add(&mlp_output)?; // Residual connection

        Ok(hidden_states)
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
        params.extend(self.ln_1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ln_2.parameters());
        params.extend(self.mlp.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.ln_1.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.ln_1.to_device(device)?;
        self.attn.to_device(device)?;
        self.ln_2.to_device(device)?;
        self.mlp.to_device(device)?;
        Ok(())
    }
}

/// GPT-2 Model
///
/// The complete GPT-2 transformer model consisting of embeddings,
/// multiple transformer blocks, and final layer normalization.
pub struct Gpt2Model {
    config: Gpt2Config,
    embeddings: Gpt2Embeddings,
    blocks: Vec<Gpt2Block>,
    ln_f: LayerNorm, // Final layer norm
}

impl Gpt2Model {
    /// Creates new GPT-2 model
    ///
    /// # Arguments
    ///
    /// * `config` - GPT-2 configuration
    ///
    /// # Returns
    ///
    /// New `Gpt2Model` instance
    pub fn new(config: Gpt2Config) -> Self {
        let embeddings = Gpt2Embeddings::new(config.clone());

        let mut blocks = Vec::new();
        for _ in 0..config.n_layer {
            blocks.push(Gpt2Block::new(config.clone()));
        }

        let ln_f = LayerNorm::new(vec![config.n_embd], config.layer_norm_epsilon, true);

        Self {
            config,
            embeddings,
            blocks,
            ln_f,
        }
    }

    /// Create GPT-2 small model (117M parameters)
    ///
    /// # Returns
    ///
    /// New GPT-2 small model
    pub fn gpt2_small() -> Self {
        Self::new(Gpt2Config::gpt2_small())
    }

    /// Create GPT-2 medium model (345M parameters)
    ///
    /// # Returns
    ///
    /// New GPT-2 medium model
    pub fn gpt2_medium() -> Self {
        Self::new(Gpt2Config::gpt2_medium())
    }

    /// Create GPT-2 large model (762M parameters)
    ///
    /// # Returns
    ///
    /// New GPT-2 large model
    pub fn gpt2_large() -> Self {
        Self::new(Gpt2Config::gpt2_large())
    }

    /// Create GPT-2 XL model (1.5B parameters)
    ///
    /// # Returns
    ///
    /// New GPT-2 XL model
    pub fn gpt2_xl() -> Self {
        Self::new(Gpt2Config::gpt2_xl())
    }

    /// Gets the model configuration
    pub fn config(&self) -> &Gpt2Config {
        &self.config
    }

    /// Gets the embeddings component
    pub fn embeddings(&self) -> &Gpt2Embeddings {
        &self.embeddings
    }

    /// Gets the number of transformer blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Gets a reference to a specific transformer block
    pub fn block(&self, index: usize) -> Option<&Gpt2Block> {
        self.blocks.get(index)
    }

    /// Gets the final layer normalization
    pub fn final_ln(&self) -> &LayerNorm {
        &self.ln_f
    }
}

impl Module for Gpt2Model {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        // Pass through all transformer blocks
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states)?;
        }

        // Final layer norm
        hidden_states = self.ln_f.forward(&hidden_states)?;

        Ok(hidden_states)
    }

    fn train(&mut self) {
        self.embeddings.train();
        for block in &mut self.blocks {
            block.train();
        }
        self.ln_f.train();
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        for block in &mut self.blocks {
            block.eval();
        }
        self.ln_f.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.embeddings.parameters() {
            params.insert(format!("embeddings_{}", name), param);
        }

        for (i, block) in self.blocks.iter().enumerate() {
            for (name, param) in block.parameters() {
                params.insert(format!("blocks_{}_{}", i, name), param);
            }
        }

        for (name, param) in self.ln_f.parameters() {
            params.insert(format!("ln_f_{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.embeddings.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.embeddings.to_device(device)?;
        for block in &mut self.blocks {
            block.to_device(device)?;
        }
        self.ln_f.to_device(device)?;
        Ok(())
    }
}

/// GPT-2 for Language Modeling
///
/// GPT-2 model with a language modeling head for next-token prediction.
/// This is the standard configuration for text generation and language modeling.
pub struct Gpt2LMHeadModel {
    transformer: Gpt2Model,
    lm_head: Linear, // Language modeling head
}

impl Gpt2LMHeadModel {
    /// Creates new GPT-2 language modeling model
    ///
    /// # Arguments
    ///
    /// * `config` - GPT-2 configuration
    ///
    /// # Returns
    ///
    /// New `Gpt2LMHeadModel` instance
    pub fn new(config: Gpt2Config) -> Self {
        let transformer = Gpt2Model::new(config.clone());
        let lm_head = Linear::new(config.n_embd, config.vocab_size, false);

        Self {
            transformer,
            lm_head,
        }
    }

    /// Create GPT-2 small for language modeling
    ///
    /// # Returns
    ///
    /// New GPT-2 small language modeling model
    pub fn gpt2_small_lm() -> Self {
        Self::new(Gpt2Config::gpt2_small())
    }

    /// Create GPT-2 medium for language modeling
    ///
    /// # Returns
    ///
    /// New GPT-2 medium language modeling model
    pub fn gpt2_medium_lm() -> Self {
        Self::new(Gpt2Config::gpt2_medium())
    }

    /// Create GPT-2 large for language modeling
    ///
    /// # Returns
    ///
    /// New GPT-2 large language modeling model
    pub fn gpt2_large_lm() -> Self {
        Self::new(Gpt2Config::gpt2_large())
    }

    /// Create GPT-2 XL for language modeling
    ///
    /// # Returns
    ///
    /// New GPT-2 XL language modeling model
    pub fn gpt2_xl_lm() -> Self {
        Self::new(Gpt2Config::gpt2_xl())
    }

    /// Gets the transformer component
    pub fn transformer(&self) -> &Gpt2Model {
        &self.transformer
    }

    /// Gets the language modeling head
    pub fn lm_head(&self) -> &Linear {
        &self.lm_head
    }

    /// Generate next token logits for the given input
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits for next token prediction
    pub fn generate_logits(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward(input_ids)
    }
}

impl Module for Gpt2LMHeadModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let transformer_outputs = self.transformer.forward(input_ids)?;
        let logits = self.lm_head.forward(&transformer_outputs)?;
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
        params.extend(self.transformer.parameters());
        params.extend(self.lm_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.transformer.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.transformer.to_device(device)?;
        self.lm_head.to_device(device)?;
        Ok(())
    }
}

/// Factory functions for creating GPT-2 models
pub mod factory {
    use super::*;

    /// Creates GPT-2 small model
    pub fn gpt2_small() -> Gpt2Model {
        Gpt2Model::gpt2_small()
    }

    /// Creates GPT-2 medium model
    pub fn gpt2_medium() -> Gpt2Model {
        Gpt2Model::gpt2_medium()
    }

    /// Creates GPT-2 large model
    pub fn gpt2_large() -> Gpt2Model {
        Gpt2Model::gpt2_large()
    }

    /// Creates GPT-2 XL model
    pub fn gpt2_xl() -> Gpt2Model {
        Gpt2Model::gpt2_xl()
    }

    /// Creates GPT-2 small for language modeling
    pub fn gpt2_small_lm() -> Gpt2LMHeadModel {
        Gpt2LMHeadModel::gpt2_small_lm()
    }

    /// Creates GPT-2 medium for language modeling
    pub fn gpt2_medium_lm() -> Gpt2LMHeadModel {
        Gpt2LMHeadModel::gpt2_medium_lm()
    }

    /// Creates GPT-2 large for language modeling
    pub fn gpt2_large_lm() -> Gpt2LMHeadModel {
        Gpt2LMHeadModel::gpt2_large_lm()
    }

    /// Creates GPT-2 XL for language modeling
    pub fn gpt2_xl_lm() -> Gpt2LMHeadModel {
        Gpt2LMHeadModel::gpt2_xl_lm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    #[test]
    fn test_gpt2_config_creation() {
        let config = Gpt2Config::gpt2_small();
        assert_eq!(config.n_embd, 768);
        assert_eq!(config.n_layer, 12);
        assert_eq!(config.n_head, 12);

        let medium_config = Gpt2Config::gpt2_medium();
        assert_eq!(medium_config.n_embd, 1024);
        assert_eq!(medium_config.n_layer, 24);
        assert_eq!(medium_config.n_head, 16);

        let large_config = Gpt2Config::gpt2_large();
        assert_eq!(large_config.n_embd, 1280);
        assert_eq!(large_config.n_layer, 36);
        assert_eq!(large_config.n_head, 20);

        let xl_config = Gpt2Config::gpt2_xl();
        assert_eq!(xl_config.n_embd, 1600);
        assert_eq!(xl_config.n_layer, 48);
        assert_eq!(xl_config.n_head, 25);
    }

    #[test]
    fn test_gpt2_config_validation() {
        let mut config = Gpt2Config::gpt2_small();
        assert!(config.validate().is_ok());

        // Test invalid configuration
        config.n_head = 7; // Should not divide 768
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gpt2_config_derived_values() {
        let config = Gpt2Config::gpt2_small();
        assert_eq!(config.head_dim(), 768 / 12);
        assert_eq!(config.intermediate_size(), 4 * 768);
    }

    #[test]
    fn test_gpt2_embeddings_creation() {
        let config = Gpt2Config::gpt2_small();
        let embeddings = Gpt2Embeddings::new(config.clone());
        assert_eq!(embeddings.config().n_embd, 768);
    }

    #[test]
    fn test_gpt2_attention_causal_mask() {
        let config = Gpt2Config::gpt2_small();
        let attention = Gpt2Attention::new(config);
        let mask = attention.create_causal_mask(3).unwrap();

        // Check mask shape
        assert_eq!(mask.dims(), &[3, 3]);
    }

    #[test]
    fn test_gpt2_model_creation() {
        let model = Gpt2Model::gpt2_small();
        assert_eq!(model.num_blocks(), 12);
        assert!(model.block(0).is_some());
        assert!(model.block(12).is_none());
    }

    #[test]
    fn test_gpt2_lm_head_model_creation() {
        let lm_model = Gpt2LMHeadModel::gpt2_small_lm();
        assert_eq!(lm_model.transformer().config().n_embd, 768);
    }

    #[test]
    fn test_gpt2_forward_pass() {
        let mut model = Gpt2Model::gpt2_small();
        let input_ids = Tensor::zeros(&[1, 10]).unwrap();

        let result = model.forward(&input_ids);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dims().len(), 3); // [batch, seq_len, hidden_size]
        assert_eq!(output.size(2).unwrap(), 768); // hidden_size
    }

    #[test]
    fn test_gpt2_lm_head_forward() {
        let mut lm_model = Gpt2LMHeadModel::gpt2_small_lm();
        let input_ids = Tensor::zeros(&[1, 10]).unwrap();

        let result = lm_model.forward(&input_ids);
        assert!(result.is_ok());

        let logits = result.unwrap();
        assert_eq!(logits.size(logits.dims().len() - 1).unwrap(), 50257); // vocab_size
    }

    #[test]
    fn test_gpt2_training_mode() {
        let mut model = Gpt2Model::gpt2_small();

        model.train();
        assert!(model.training());

        model.eval();
        assert!(!model.training());
    }

    #[test]
    fn test_gpt2_parameter_count() {
        let model = Gpt2Model::gpt2_small();
        let params = model.parameters();
        assert!(!params.is_empty());

        // Check that we have parameters from all components
        let param_names: Vec<&String> = params.keys().collect();
        assert!(param_names.iter().any(|name| name.contains("embeddings")));
        assert!(param_names.iter().any(|name| name.contains("blocks")));
        assert!(param_names.iter().any(|name| name.contains("ln_f")));
    }

    #[test]
    fn test_gpt2_block_components() {
        let config = Gpt2Config::gpt2_small();
        let block = Gpt2Block::new(config.clone());

        // Verify components exist
        assert_eq!(block.attention().config().n_embd, 768);
        assert_eq!(block.mlp().config().n_embd, 768);
    }

    #[test]
    fn test_factory_functions() {
        let model = factory::gpt2_small();
        assert_eq!(model.config().n_embd, 768);

        let medium_model = factory::gpt2_medium();
        assert_eq!(medium_model.config().n_embd, 1024);

        let lm_model = factory::gpt2_small_lm();
        assert_eq!(lm_model.transformer().config().n_embd, 768);
    }

    #[test]
    fn test_generate_logits() {
        let lm_model = Gpt2LMHeadModel::gpt2_small_lm();
        let input_ids = Tensor::zeros(&[1, 5]).unwrap();

        let result = lm_model.generate_logits(&input_ids);
        assert!(result.is_ok());

        let logits = result.unwrap();
        assert_eq!(logits.dims(), &[1, 5, 50257]); // [batch, seq, vocab]
    }
}