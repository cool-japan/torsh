//! GPT-2 Implementation from Scratch using ToRSh
//! 
//! This example demonstrates:
//! - Building the complete GPT-2 architecture
//! - Multi-head self-attention implementation
//! - Positional embeddings and layer normalization
//! - Text generation with different sampling strategies
//! - Loading pretrained weights (conceptual)

use torsh::prelude::*;
use torsh::nn::{Module, Linear, Embedding, LayerNorm, Dropout};
use torsh::optim::{AdamW, Optimizer};
use torsh::tensor::Tensor;
use std::error::Error;
use std::collections::HashMap;

/// GPT-2 configuration
#[derive(Debug, Clone)]
struct GPT2Config {
    vocab_size: usize,
    n_positions: usize,
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    n_inner: Option<usize>, // None means 4 * n_embd
    activation_function: String,
    resid_pdrop: f32,
    embd_pdrop: f32,
    attn_pdrop: f32,
    layer_norm_epsilon: f32,
    initializer_range: f32,
    summary_type: String,
    summary_use_proj: bool,
    summary_activation: Option<String>,
    summary_proj_to_labels: bool,
    summary_first_dropout: f32,
    scale_attn_weights: bool,
    gradient_checkpointing: bool,
}

impl Default for GPT2Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_inner: None,
            activation_function: "gelu".to_string(),
            resid_pdrop: 0.1,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            summary_type: "cls_index".to_string(),
            summary_use_proj: true,
            summary_activation: None,
            summary_proj_to_labels: true,
            summary_first_dropout: 0.1,
            scale_attn_weights: true,
            gradient_checkpointing: false,
        }
    }
}

impl GPT2Config {
    /// GPT-2 Small (124M parameters)
    fn gpt2_small() -> Self {
        Self::default()
    }
    
    /// GPT-2 Medium (355M parameters)
    fn gpt2_medium() -> Self {
        Self {
            n_embd: 1024,
            n_head: 16,
            n_layer: 24,
            ..Self::default()
        }
    }
    
    /// GPT-2 Large (774M parameters)
    fn gpt2_large() -> Self {
        Self {
            n_embd: 1280,
            n_head: 20,
            n_layer: 36,
            ..Self::default()
        }
    }
    
    /// GPT-2 XL (1.5B parameters)
    fn gpt2_xl() -> Self {
        Self {
            n_embd: 1600,
            n_head: 25,
            n_layer: 48,
            ..Self::default()
        }
    }
}

/// GELU activation function
fn gelu(x: &Tensor) -> Result<Tensor, torsh::TorshError> {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let x_cubed = x.pow(3.0)?;
    let inner = x.add(&x_cubed.mul_scalar(0.044715)?)?;
    let inner = inner.mul_scalar((2.0 / std::f32::consts::PI).sqrt())?;
    let tanh_out = inner.tanh()?;
    let gelu_out = x.mul(&tanh_out.add_scalar(1.0)?)?;
    gelu_out.mul_scalar(0.5)
}

/// Multi-head self-attention
#[derive(Debug)]
struct GPT2Attention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    
    c_attn: Linear, // Combined Q, K, V projection
    c_proj: Linear, // Output projection
    attn_dropout: Dropout,
    resid_dropout: Dropout,
    
    scale_attn_weights: bool,
}

impl GPT2Attention {
    fn new(config: &GPT2Config) -> Self {
        let embed_dim = config.n_embd;
        let num_heads = config.n_head;
        let head_dim = embed_dim / num_heads;
        
        Self {
            embed_dim,
            num_heads,
            head_dim,
            scale: if config.scale_attn_weights {
                1.0 / (head_dim as f32).sqrt()
            } else {
                1.0
            },
            
            c_attn: Linear::new(embed_dim, 3 * embed_dim, true),
            c_proj: Linear::new(embed_dim, embed_dim, true),
            attn_dropout: Dropout::new(config.attn_pdrop),
            resid_dropout: Dropout::new(config.resid_pdrop),
            
            scale_attn_weights: config.scale_attn_weights,
        }
    }
    
    fn split_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor, torsh::TorshError> {
        // Reshape from (batch, seq_len, embed_dim) to (batch, num_heads, seq_len, head_dim)
        let x = x.view(&[batch_size as i32, seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        x.transpose(1, 2)
    }
    
    fn merge_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor, torsh::TorshError> {
        // Reshape from (batch, num_heads, seq_len, head_dim) to (batch, seq_len, embed_dim)
        let x = x.transpose(1, 2)?;
        x.contiguous()?.view(&[batch_size as i32, seq_len as i32, self.embed_dim as i32])
    }
}

impl Module for GPT2Attention {
    type Error = torsh::TorshError;
    
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, Self::Error> {
        let shape = hidden_states.shape();
        let batch_size = shape.dims()[0];
        let seq_len = shape.dims()[1];
        
        // Project to Q, K, V
        let qkv = self.c_attn.forward(hidden_states)?;
        
        // Split into Q, K, V
        let qkv_splits = qkv.chunk(3, -1)?;
        let query = &qkv_splits[0];
        let key = &qkv_splits[1];
        let value = &qkv_splits[2];
        
        // Split heads
        let query = self.split_heads(query, batch_size, seq_len)?;
        let key = self.split_heads(key, batch_size, seq_len)?;
        let value = self.split_heads(value, batch_size, seq_len)?;
        
        // Compute attention scores
        let scores = query.matmul(&key.transpose(-2, -1)?)?;
        let scores = scores.mul_scalar(self.scale)?;
        
        // Apply causal mask
        let mask = self.create_causal_mask(seq_len)?;
        let scores = scores.masked_fill(&mask, f32::NEG_INFINITY)?;
        
        // Softmax
        let attn_weights = scores.softmax(-1)?;
        let attn_weights = self.attn_dropout.forward(&attn_weights)?;
        
        // Apply attention to values
        let attn_output = attn_weights.matmul(&value)?;
        
        // Merge heads
        let attn_output = self.merge_heads(&attn_output, batch_size, seq_len)?;
        
        // Output projection
        let attn_output = self.c_proj.forward(&attn_output)?;
        self.resid_dropout.forward(&attn_output)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![];
        params.extend(self.c_attn.parameters());
        params.extend(self.c_proj.parameters());
        params
    }
    
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = vec![];
        
        for (name, param) in self.c_attn.named_parameters() {
            params.push((format!("c_attn.{}", name), param));
        }
        for (name, param) in self.c_proj.named_parameters() {
            params.push((format!("c_proj.{}", name), param));
        }
        
        params
    }
}

impl GPT2Attention {
    fn create_causal_mask(&self, seq_len: usize) -> Result<Tensor, torsh::TorshError> {
        // Create lower triangular mask
        let mask = Tensor::ones(&[seq_len, seq_len])?;
        let mask = mask.tril(0)?;
        let mask = mask.eq(&Tensor::zeros(&[seq_len, seq_len])?)?;
        
        // Add batch and head dimensions
        mask.unsqueeze(0)?.unsqueeze(0)
    }
}

/// Feed-forward network
#[derive(Debug)]
struct GPT2MLP {
    c_fc: Linear,
    c_proj: Linear,
    dropout: Dropout,
}

impl GPT2MLP {
    fn new(config: &GPT2Config) -> Self {
        let inner_dim = config.n_inner.unwrap_or(4 * config.n_embd);
        
        Self {
            c_fc: Linear::new(config.n_embd, inner_dim, true),
            c_proj: Linear::new(inner_dim, config.n_embd, true),
            dropout: Dropout::new(config.resid_pdrop),
        }
    }
}

impl Module for GPT2MLP {
    type Error = torsh::TorshError;
    
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, Self::Error> {
        let hidden_states = self.c_fc.forward(hidden_states)?;
        let hidden_states = gelu(&hidden_states)?;
        let hidden_states = self.c_proj.forward(&hidden_states)?;
        self.dropout.forward(&hidden_states)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![];
        params.extend(self.c_fc.parameters());
        params.extend(self.c_proj.parameters());
        params
    }
    
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = vec![];
        
        for (name, param) in self.c_fc.named_parameters() {
            params.push((format!("c_fc.{}", name), param));
        }
        for (name, param) in self.c_proj.named_parameters() {
            params.push((format!("c_proj.{}", name), param));
        }
        
        params
    }
}

/// Transformer block
#[derive(Debug)]
struct GPT2Block {
    ln_1: LayerNorm,
    attn: GPT2Attention,
    ln_2: LayerNorm,
    mlp: GPT2MLP,
}

impl GPT2Block {
    fn new(config: &GPT2Config) -> Self {
        Self {
            ln_1: LayerNorm::new(vec![config.n_embd], config.layer_norm_epsilon, true),
            attn: GPT2Attention::new(config),
            ln_2: LayerNorm::new(vec![config.n_embd], config.layer_norm_epsilon, true),
            mlp: GPT2MLP::new(config),
        }
    }
}

impl Module for GPT2Block {
    type Error = torsh::TorshError;
    
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, Self::Error> {
        // Self-attention with residual
        let residual = hidden_states.clone();
        let hidden_states = self.ln_1.forward(hidden_states)?;
        let attn_output = self.attn.forward(&hidden_states)?;
        let hidden_states = residual.add(&attn_output)?;
        
        // MLP with residual
        let residual = hidden_states.clone();
        let hidden_states_norm = self.ln_2.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&hidden_states_norm)?;
        residual.add(&mlp_output)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![];
        params.extend(self.ln_1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ln_2.parameters());
        params.extend(self.mlp.parameters());
        params
    }
    
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = vec![];
        
        for (name, param) in self.ln_1.named_parameters() {
            params.push((format!("ln_1.{}", name), param));
        }
        for (name, param) in self.attn.named_parameters() {
            params.push((format!("attn.{}", name), param));
        }
        for (name, param) in self.ln_2.named_parameters() {
            params.push((format!("ln_2.{}", name), param));
        }
        for (name, param) in self.mlp.named_parameters() {
            params.push((format!("mlp.{}", name), param));
        }
        
        params
    }
}

/// Complete GPT-2 model
#[derive(Debug)]
struct GPT2Model {
    config: GPT2Config,
    
    wte: Embedding, // Token embeddings
    wpe: Embedding, // Position embeddings
    drop: Dropout,
    
    h: Vec<GPT2Block>, // Transformer blocks
    ln_f: LayerNorm,   // Final layer norm
}

impl GPT2Model {
    fn new(config: GPT2Config) -> Self {
        let mut blocks = Vec::new();
        for _ in 0..config.n_layer {
            blocks.push(GPT2Block::new(&config));
        }
        
        Self {
            wte: Embedding::new(config.vocab_size, config.n_embd),
            wpe: Embedding::new(config.n_positions, config.n_embd),
            drop: Dropout::new(config.embd_pdrop),
            
            h: blocks,
            ln_f: LayerNorm::new(vec![config.n_embd], config.layer_norm_epsilon, true),
            
            config,
        }
    }
}

impl Module for GPT2Model {
    type Error = torsh::TorshError;
    
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor, Self::Error> {
        let shape = input_ids.shape();
        let batch_size = shape.dims()[0];
        let seq_len = shape.dims()[1];
        
        // Token embeddings
        let inputs_embeds = self.wte.forward(input_ids)?;
        
        // Position embeddings
        let position_ids = Tensor::arange(0, seq_len as i64, 1)?;
        let position_embeds = self.wpe.forward(&position_ids)?;
        
        // Combine embeddings
        let hidden_states = inputs_embeds.add(&position_embeds)?;
        let mut hidden_states = self.drop.forward(&hidden_states)?;
        
        // Pass through transformer blocks
        for block in &self.h {
            hidden_states = block.forward(&hidden_states)?;
        }
        
        // Final layer norm
        self.ln_f.forward(&hidden_states)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![];
        
        params.extend(self.wte.parameters());
        params.extend(self.wpe.parameters());
        
        for block in &self.h {
            params.extend(block.parameters());
        }
        
        params.extend(self.ln_f.parameters());
        
        params
    }
    
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = vec![];
        
        for (name, param) in self.wte.named_parameters() {
            params.push((format!("wte.{}", name), param));
        }
        for (name, param) in self.wpe.named_parameters() {
            params.push((format!("wpe.{}", name), param));
        }
        
        for (i, block) in self.h.iter().enumerate() {
            for (name, param) in block.named_parameters() {
                params.push((format!("h.{}.{}", i, name), param));
            }
        }
        
        for (name, param) in self.ln_f.named_parameters() {
            params.push((format!("ln_f.{}", name), param));
        }
        
        params
    }
}

/// GPT-2 Language Model Head
#[derive(Debug)]
struct GPT2LMHeadModel {
    transformer: GPT2Model,
    lm_head: Linear,
}

impl GPT2LMHeadModel {
    fn new(config: GPT2Config) -> Self {
        let transformer = GPT2Model::new(config.clone());
        
        // The language modeling head is tied with the input embeddings
        // In a real implementation, we would share weights
        let lm_head = Linear::new(config.n_embd, config.vocab_size, false);
        
        Self {
            transformer,
            lm_head,
        }
    }
    
    /// Generate text using different sampling strategies
    fn generate(
        &self,
        input_ids: &Tensor,
        max_length: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> Result<Tensor, torsh::TorshError> {
        let mut current_ids = input_ids.clone();
        
        for _ in 0..max_length {
            // Get logits for next token
            let outputs = self.forward(&current_ids)?;
            let next_token_logits = outputs.select(-2, -1)?; // Get last position
            
            // Apply temperature
            let next_token_logits = next_token_logits.div_scalar(temperature)?;
            
            // Apply top-k filtering
            let next_token_logits = if let Some(k) = top_k {
                self.top_k_filtering(&next_token_logits, k)?
            } else {
                next_token_logits
            };
            
            // Apply top-p (nucleus) filtering
            let next_token_logits = if let Some(p) = top_p {
                self.top_p_filtering(&next_token_logits, p)?
            } else {
                next_token_logits
            };
            
            // Sample from the distribution
            let probs = next_token_logits.softmax(-1)?;
            let next_token = probs.multinomial(1, true)?;
            
            // Append to sequence
            current_ids = current_ids.cat(&next_token.unsqueeze(0)?, -1)?;
            
            // Stop if we hit the end token
            if next_token.item::<i64>() == 50256 { // GPT-2 EOS token
                break;
            }
        }
        
        Ok(current_ids)
    }
    
    fn top_k_filtering(&self, logits: &Tensor, k: usize) -> Result<Tensor, torsh::TorshError> {
        // Keep only top k values
        let (top_k_values, top_k_indices) = logits.topk(k as i64, -1, true, true)?;
        let min_value = top_k_values.select(-1, -1)?;
        
        // Set all values below threshold to -inf
        logits.masked_fill(&logits.lt(&min_value)?, f32::NEG_INFINITY)
    }
    
    fn top_p_filtering(&self, logits: &Tensor, p: f32) -> Result<Tensor, torsh::TorshError> {
        // Sort by descending probability
        let (sorted_logits, sorted_indices) = logits.sort(-1, true)?;
        let cumulative_probs = sorted_logits.softmax(-1)?.cumsum(-1)?;
        
        // Find where cumulative probability exceeds p
        let sorted_indices_to_remove = cumulative_probs.gt_scalar(p)?;
        
        // Shift right to keep first token above threshold
        let sorted_indices_to_remove = sorted_indices_to_remove.roll(1, -1)?;
        
        // Set values to -inf
        let mut filtered_logits = sorted_logits.clone();
        filtered_logits = filtered_logits.masked_fill(&sorted_indices_to_remove, f32::NEG_INFINITY)?;
        
        // Restore original order
        filtered_logits.scatter(-1, &sorted_indices, &filtered_logits)
    }
}

impl Module for GPT2LMHeadModel {
    type Error = torsh::TorshError;
    
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor, Self::Error> {
        let hidden_states = self.transformer.forward(input_ids)?;
        self.lm_head.forward(&hidden_states)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.transformer.parameters();
        params.extend(self.lm_head.parameters());
        params
    }
    
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = vec![];
        
        for (name, param) in self.transformer.named_parameters() {
            params.push((format!("transformer.{}", name), param));
        }
        for (name, param) in self.lm_head.named_parameters() {
            params.push((format!("lm_head.{}", name), param));
        }
        
        params
    }
}

/// Simple tokenizer for demo
struct SimpleTokenizer {
    vocab: HashMap<String, i64>,
    inverse_vocab: HashMap<i64, String>,
}

impl SimpleTokenizer {
    fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut inverse_vocab = HashMap::new();
        
        // Add special tokens
        vocab.insert("<|endoftext|>".to_string(), 50256);
        inverse_vocab.insert(50256, "<|endoftext|>".to_string());
        
        // Add some example tokens
        let tokens = vec![
            "Hello", "world", "!", "The", "quick", "brown", "fox", "jumps",
            "over", "the", "lazy", "dog", ".", "GPT", "-", "2", "is", "a",
            "large", "language", "model", "trained", "by", "OpenAI", ",",
            "with", "175", "billion", "parameters", "and", "can", "generate",
            "human", "like", "text", "based", "on", "input", "prompts"
        ];
        
        for (i, token) in tokens.iter().enumerate() {
            vocab.insert(token.to_string(), i as i64);
            inverse_vocab.insert(i as i64, token.to_string());
        }
        
        Self { vocab, inverse_vocab }
    }
    
    fn encode(&self, text: &str) -> Vec<i64> {
        text.split_whitespace()
            .filter_map(|word| self.vocab.get(word).copied())
            .collect()
    }
    
    fn decode(&self, ids: &[i64]) -> String {
        ids.iter()
            .filter_map(|id| self.inverse_vocab.get(id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Display model statistics
fn show_model_stats(config: &GPT2Config) -> Result<(), Box<dyn Error>> {
    println!("\n📊 Model Statistics");
    println!("===================");
    
    // Calculate parameter counts
    let embedding_params = 2 * config.vocab_size * config.n_embd; // wte + wpe
    let attention_params_per_layer = 4 * config.n_embd * config.n_embd; // Q,K,V,O projections
    let mlp_params_per_layer = 8 * config.n_embd * config.n_embd; // 2 projections with 4x hidden
    let ln_params_per_layer = 4 * config.n_embd; // 2 layer norms with weight + bias
    
    let total_transformer_params = config.n_layer * (
        attention_params_per_layer + mlp_params_per_layer + ln_params_per_layer
    );
    
    let lm_head_params = config.vocab_size * config.n_embd;
    let total_params = embedding_params + total_transformer_params + lm_head_params;
    
    println!("Configuration:");
    println!("  - Vocabulary size: {:,}", config.vocab_size);
    println!("  - Context length: {:,}", config.n_positions);
    println!("  - Embedding dimension: {:,}", config.n_embd);
    println!("  - Number of layers: {}", config.n_layer);
    println!("  - Number of heads: {}", config.n_head);
    println!("  - Head dimension: {}", config.n_embd / config.n_head);
    
    println!("\nParameter breakdown:");
    println!("  - Embeddings: {:.1}M", embedding_params as f32 / 1_000_000.0);
    println!("  - Transformer blocks: {:.1}M", total_transformer_params as f32 / 1_000_000.0);
    println!("  - Language model head: {:.1}M", lm_head_params as f32 / 1_000_000.0);
    println!("  - Total parameters: {:.1}M", total_params as f32 / 1_000_000.0);
    
    // Memory estimation
    let param_memory_mb = (total_params * 4) as f32 / 1_000_000.0; // 4 bytes per float32
    let activation_memory_mb = (config.n_layer * 34 * config.n_positions * config.n_embd * 4) as f32 / 1_000_000.0;
    
    println!("\nMemory requirements (approximate):");
    println!("  - Parameters: {:.1} MB", param_memory_mb);
    println!("  - Activations (seq_len={}): {:.1} MB", config.n_positions, activation_memory_mb);
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("🤖 ToRSh GPT-2 Implementation from Scratch");
    println!("==========================================");
    
    // Set random seed
    torsh::manual_seed(42);
    
    // Create different GPT-2 configurations
    let configs = vec![
        ("GPT-2 Small (124M)", GPT2Config::gpt2_small()),
        ("GPT-2 Medium (355M)", GPT2Config::gpt2_medium()),
        ("GPT-2 Large (774M)", GPT2Config::gpt2_large()),
        ("GPT-2 XL (1.5B)", GPT2Config::gpt2_xl()),
    ];
    
    // Display all model variants
    println!("\n📋 Available GPT-2 Variants:");
    for (name, config) in &configs {
        println!("\n{}", name);
        show_model_stats(&config)?;
    }
    
    // Use GPT-2 Small for demonstration
    println!("\n🎯 Using GPT-2 Small for demonstration...");
    let config = GPT2Config::gpt2_small();
    let model = GPT2LMHeadModel::new(config.clone());
    
    // Create tokenizer
    let tokenizer = SimpleTokenizer::new();
    
    // Example: Forward pass
    println!("\n🔄 Testing forward pass...");
    let input_text = "Hello world";
    let input_ids = tokenizer.encode(input_text);
    let input_tensor = Tensor::from_vec(input_ids.clone(), &[1, input_ids.len()])?;
    
    println!("Input: \"{}\"", input_text);
    println!("Token IDs: {:?}", input_ids);
    
    let output = model.forward(&input_tensor)?;
    println!("Output shape: {:?} (batch × seq_len × vocab_size)", output.shape());
    
    // Example: Text generation
    println!("\n✍️  Text Generation Demo");
    println!("========================");
    
    let prompts = vec![
        "The quick brown fox",
        "GPT-2 is a",
        "Hello world",
    ];
    
    for prompt in prompts {
        println!("\nPrompt: \"{}\"", prompt);
        
        let input_ids = tokenizer.encode(prompt);
        let input_tensor = Tensor::from_vec(input_ids, &[1, input_ids.len()])?;
        
        // Generate with different strategies
        println!("\n1. Greedy decoding (temperature=1.0):");
        let generated = model.generate(&input_tensor, 20, 1.0, None, None)?;
        let generated_ids = generated.to_vec1::<i64>()?;
        let generated_text = tokenizer.decode(&generated_ids);
        println!("   {}", generated_text);
        
        println!("\n2. Temperature sampling (temperature=0.8):");
        let generated = model.generate(&input_tensor, 20, 0.8, None, None)?;
        let generated_ids = generated.to_vec1::<i64>()?;
        let generated_text = tokenizer.decode(&generated_ids);
        println!("   {}", generated_text);
        
        println!("\n3. Top-k sampling (k=50, temperature=0.9):");
        let generated = model.generate(&input_tensor, 20, 0.9, Some(50), None)?;
        let generated_ids = generated.to_vec1::<i64>()?;
        let generated_text = tokenizer.decode(&generated_ids);
        println!("   {}", generated_text);
        
        println!("\n4. Nucleus sampling (p=0.95, temperature=1.0):");
        let generated = model.generate(&input_tensor, 20, 1.0, None, Some(0.95))?;
        let generated_ids = generated.to_vec1::<i64>()?;
        let generated_text = tokenizer.decode(&generated_ids);
        println!("   {}", generated_text);
    }
    
    // Training example
    println!("\n🎓 Training Example");
    println!("===================");
    
    // Create optimizer
    let mut optimizer = AdamW::builder()
        .learning_rate(5e-5)
        .beta1(0.9)
        .beta2(0.999)
        .epsilon(1e-8)
        .weight_decay(0.01)
        .build();
    
    // Add parameters
    for param in model.parameters() {
        optimizer.add_param_group(param.clone());
    }
    
    // Training data
    let training_texts = vec![
        "The quick brown fox jumps over the lazy dog",
        "GPT-2 is a large language model",
        "Hello world ! GPT-2 can generate text",
    ];
    
    println!("Training on {} examples...", training_texts.len());
    
    // Simulate training loop
    for epoch in 1..=3 {
        let mut total_loss = 0.0;
        
        for text in &training_texts {
            // Tokenize
            let tokens = tokenizer.encode(text);
            if tokens.len() < 2 {
                continue;
            }
            
            // Create input and target
            let input_ids = &tokens[..tokens.len()-1];
            let target_ids = &tokens[1..];
            
            let input_tensor = Tensor::from_vec(input_ids.to_vec(), &[1, input_ids.len()])?;
            let target_tensor = Tensor::from_vec(target_ids.to_vec(), &[1, target_ids.len()])?;
            
            // Forward pass
            let logits = model.forward(&input_tensor)?;
            
            // Compute loss (cross-entropy)
            let loss = compute_language_modeling_loss(&logits, &target_tensor)?;
            
            // Backward pass
            optimizer.zero_grad()?;
            loss.backward()?;
            optimizer.step()?;
            
            total_loss += loss.item::<f32>();
        }
        
        let avg_loss = total_loss / training_texts.len() as f32;
        let perplexity = avg_loss.exp();
        
        println!("Epoch {}: Loss = {:.4}, Perplexity = {:.2}", 
                epoch, avg_loss, perplexity);
    }
    
    // Model analysis
    println!("\n📊 Model Analysis");
    println!("=================");
    
    // Attention pattern visualization (conceptual)
    println!("\nAttention patterns (conceptual):");
    println!("Layer 1: Focusing on local context (adjacent tokens)");
    println!("Layer 6: Capturing syntactic dependencies");
    println!("Layer 12: Understanding semantic relationships");
    
    // Parameter distribution
    let total_params: usize = model.parameters().iter()
        .map(|p| p.numel())
        .sum();
    
    println!("\nParameter utilization:");
    println!("  Total trainable parameters: {:,}", total_params);
    println!("  Average parameters per layer: {:,}", total_params / config.n_layer);
    
    println!("\n✅ GPT-2 example completed successfully!");
    
    Ok(())
}

/// Compute language modeling loss
fn compute_language_modeling_loss(
    logits: &Tensor,
    targets: &Tensor,
) -> Result<Tensor, torsh::TorshError> {
    // Reshape logits and targets for cross-entropy
    let batch_size = logits.shape().dims()[0];
    let seq_len = logits.shape().dims()[1];
    let vocab_size = logits.shape().dims()[2];
    
    let logits = logits.view(&[(batch_size * seq_len) as i32, vocab_size as i32])?;
    let targets = targets.view(&[-1])?;
    
    // Cross-entropy loss
    let log_probs = logits.log_softmax(-1)?;
    let gathered = log_probs.gather(1, &targets.unsqueeze(1)?)?;
    let loss = -gathered.mean()?;
    
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpt2_config() {
        let config = GPT2Config::default();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.n_positions, 1024);
        assert_eq!(config.n_embd, 768);
        assert_eq!(config.n_layer, 12);
        assert_eq!(config.n_head, 12);
    }
    
    #[test]
    fn test_attention() -> Result<(), Box<dyn Error>> {
        let config = GPT2Config::default();
        let attention = GPT2Attention::new(&config);
        
        let hidden_states = Tensor::randn(&[2, 10, 768])?; // batch=2, seq_len=10
        let output = attention.forward(&hidden_states)?;
        
        assert_eq!(output.shape(), &[2, 10, 768]);
        
        Ok(())
    }
    
    #[test]
    fn test_gpt2_block() -> Result<(), Box<dyn Error>> {
        let config = GPT2Config::default();
        let block = GPT2Block::new(&config);
        
        let hidden_states = Tensor::randn(&[2, 10, 768])?;
        let output = block.forward(&hidden_states)?;
        
        assert_eq!(output.shape(), &[2, 10, 768]);
        
        Ok(())
    }
    
    #[test]
    fn test_full_model() -> Result<(), Box<dyn Error>> {
        let config = GPT2Config {
            n_layer: 2, // Use fewer layers for testing
            ..GPT2Config::default()
        };
        
        let model = GPT2LMHeadModel::new(config);
        let input_ids = Tensor::randint(0, 50257, &[1, 5])?; // batch=1, seq_len=5
        
        let output = model.forward(&input_ids)?;
        assert_eq!(output.shape(), &[1, 5, 50257]);
        
        Ok(())
    }
}