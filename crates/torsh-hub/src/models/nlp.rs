//! NLP models for the ToRSh Hub Model Zoo
//!
//! This module contains implementations of popular natural language processing models
//! including BERT, GPT, and Transformer architectures.

use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_tensor::Tensor;

/// Multi-Head Attention mechanism
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f32) -> Self {
        assert_eq!(
            embed_dim % num_heads,
            0,
            "embed_dim must be divisible by num_heads"
        );
        let head_dim = embed_dim / num_heads;

        Self {
            num_heads,
            head_dim,
            embed_dim,
            q_proj: Linear::new(embed_dim, embed_dim, true),
            k_proj: Linear::new(embed_dim, embed_dim, true),
            v_proj: Linear::new(embed_dim, embed_dim, true),
            out_proj: Linear::new(embed_dim, embed_dim, true),
            dropout: Dropout::new(dropout),
        }
    }

    /// Manual softmax implementation to avoid .item() issues in torsh-nn Softmax layer
    fn manual_softmax(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // For a 2D tensor [seq_len, seq_len], apply softmax along the last dimension
        let data = input.data()?;
        let shape_binding = input.shape();
        let shape = shape_binding.dims();

        if shape.len() != 2 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "manual_softmax only supports 2D tensors".to_string(),
            ));
        }

        let rows = shape[0];
        let cols = shape[1];
        let mut result_data = vec![0.0f32; data.len()];

        for row in 0..rows {
            let row_start = row * cols;
            let row_end = row_start + cols;
            let row_data = &data[row_start..row_end];

            // Find max for numerical stability
            let max_val = row_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Compute exp(x - max) and sum
            let mut exp_sum = 0.0f32;
            let mut exp_values = vec![0.0f32; cols];

            for (i, &val) in row_data.iter().enumerate() {
                let exp_val = (val - max_val).exp();
                exp_values[i] = exp_val;
                exp_sum += exp_val;
            }

            // Compute softmax values
            for (i, exp_val) in exp_values.iter().enumerate() {
                result_data[row_start + i] = exp_val / exp_sum;
            }
        }

        Tensor::from_data(result_data, shape.to_vec(), input.device())
    }

    /// Get query projection layer
    pub fn q_proj(&self) -> &Linear {
        &self.q_proj
    }

    /// Get key projection layer
    pub fn k_proj(&self) -> &Linear {
        &self.k_proj
    }

    /// Get value projection layer
    pub fn v_proj(&self) -> &Linear {
        &self.v_proj
    }

    /// Get output projection layer
    pub fn out_proj(&self) -> &Linear {
        &self.out_proj
    }

    /// Get number of attention heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get dimension per head
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get embedding dimension
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get dropout layer
    pub fn dropout(&self) -> &Dropout {
        &self.dropout
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let (batch_size, seq_len, _) = (
            x.shape().dims()[0],
            x.shape().dims()[1],
            x.shape().dims()[2],
        );

        // Project to Q, K, V - reshape to 2D for Linear layers

        // Reshape input from [batch_size, seq_len, embed_dim] to [batch_size * seq_len, embed_dim]
        let x_2d = x.view(&[(batch_size * seq_len) as i32, self.embed_dim as i32])?;

        let q_2d = self.q_proj.forward(&x_2d)?;
        let k_2d = self.k_proj.forward(&x_2d)?;
        let v_2d = self.v_proj.forward(&x_2d)?;

        // Reshape back to 3D: [batch_size, seq_len, embed_dim]
        let q = q_2d.view(&[batch_size as i32, seq_len as i32, self.embed_dim as i32])?;
        let k = k_2d.view(&[batch_size as i32, seq_len as i32, self.embed_dim as i32])?;
        let v = v_2d.view(&[batch_size as i32, seq_len as i32, self.embed_dim as i32])?;

        // Reshape for multi-head attention
        let q = q
            .view(&[
                batch_size as i32,
                seq_len as i32,
                self.num_heads as i32,
                self.head_dim as i32,
            ])?
            .transpose(1, 2)?; // [batch, num_heads, seq_len, head_dim]
        let k = k
            .view(&[
                batch_size as i32,
                seq_len as i32,
                self.num_heads as i32,
                self.head_dim as i32,
            ])?
            .transpose(1, 2)?;
        let v = v
            .view(&[
                batch_size as i32,
                seq_len as i32,
                self.num_heads as i32,
                self.head_dim as i32,
            ])?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();

        // Simple approach: process each sample in the batch individually
        let mut batch_outputs = Vec::new();

        for batch_idx in 0..batch_size {
            // Extract tensors for this batch sample: [num_heads, seq_len, head_dim]
            let q_batch = q.select(0, batch_idx as i64)?; // [num_heads, seq_len, head_dim]
            let k_batch = k.select(0, batch_idx as i64)?; // [num_heads, seq_len, head_dim]
            let v_batch = v.select(0, batch_idx as i64)?; // [num_heads, seq_len, head_dim]

            let mut head_outputs = Vec::new();

            for head_idx in 0..self.num_heads {
                // Extract tensors for this head: [seq_len, head_dim]
                let q_head = q_batch.select(0, head_idx as i64)?; // [seq_len, head_dim]
                let k_head = k_batch.select(0, head_idx as i64)?; // [seq_len, head_dim]
                let v_head = v_batch.select(0, head_idx as i64)?; // [seq_len, head_dim]

                // Transpose k for attention: [head_dim, seq_len]
                let k_head_t = k_head.transpose(0, 1)?;

                // Compute attention scores: [seq_len, head_dim] x [head_dim, seq_len] = [seq_len, seq_len]
                let scores = q_head.matmul(&k_head_t)?.div_scalar(scale)?;

                // Apply softmax to get attention weights - manual implementation to avoid .item() issues
                let attn_weights = self.manual_softmax(&scores)?;

                // Apply attention to values: [seq_len, seq_len] x [seq_len, head_dim] = [seq_len, head_dim]
                let head_output = attn_weights.matmul(&v_head)?;
                head_outputs.push(head_output);
            }

            // Stack head outputs: [num_heads, seq_len, head_dim]
            let mut batch_data = Vec::new();
            for head_output in &head_outputs {
                let head_data = head_output.data()?;
                batch_data.extend_from_slice(&head_data);
            }

            let batch_output = Tensor::from_data(
                batch_data,
                vec![self.num_heads, seq_len, self.head_dim],
                q.device(),
            )?;
            batch_outputs.push(batch_output);
        }

        // Stack batch outputs: [batch_size, num_heads, seq_len, head_dim]
        let mut all_data = Vec::new();
        for batch_output in &batch_outputs {
            let batch_data = batch_output.data()?;
            all_data.extend_from_slice(&batch_data);
        }

        let attn_output = Tensor::from_data(
            all_data,
            vec![batch_size, self.num_heads, seq_len, self.head_dim],
            q.device(),
        )?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.view(&[
            batch_size as i32,
            seq_len as i32,
            self.embed_dim as i32,
        ])?;

        // Final projection - reshape to 2D for Linear layer
        let attn_output_2d =
            attn_output.view(&[(batch_size * seq_len) as i32, self.embed_dim as i32])?;
        let projected_2d = self.out_proj.forward(&attn_output_2d)?;
        let projected =
            projected_2d.view(&[batch_size as i32, seq_len as i32, self.embed_dim as i32])?;

        Ok(projected)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.q_proj.named_parameters() {
            params.insert(format!("q_proj.{}", name), param);
        }
        for (name, param) in self.k_proj.named_parameters() {
            params.insert(format!("k_proj.{}", name), param);
        }
        for (name, param) in self.v_proj.named_parameters() {
            params.insert(format!("v_proj.{}", name), param);
        }
        for (name, param) in self.out_proj.named_parameters() {
            params.insert(format!("out_proj.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.q_proj.named_parameters() {
            params.insert(format!("q_proj.{}", name), param);
        }
        for (name, param) in self.k_proj.named_parameters() {
            params.insert(format!("k_proj.{}", name), param);
        }
        for (name, param) in self.v_proj.named_parameters() {
            params.insert(format!("v_proj.{}", name), param);
        }
        for (name, param) in self.out_proj.named_parameters() {
            params.insert(format!("out_proj.{}", name), param);
        }

        params
    }

    fn train(&mut self) {
        self.q_proj.train();
        self.k_proj.train();
        self.v_proj.train();
        self.out_proj.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.q_proj.eval();
        self.k_proj.eval();
        self.v_proj.eval();
        self.out_proj.eval();
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.q_proj.training()
    }

    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor<f32>>,
        strict: bool,
    ) -> Result<()> {
        // Load projections
        let q_proj_dict = self.extract_sub_dict(state_dict, "q_proj.");
        self.q_proj.load_state_dict(&q_proj_dict, strict)?;

        let k_proj_dict = self.extract_sub_dict(state_dict, "k_proj.");
        self.k_proj.load_state_dict(&k_proj_dict, strict)?;

        let v_proj_dict = self.extract_sub_dict(state_dict, "v_proj.");
        self.v_proj.load_state_dict(&v_proj_dict, strict)?;

        let out_proj_dict = self.extract_sub_dict(state_dict, "out_proj.");
        self.out_proj.load_state_dict(&out_proj_dict, strict)?;

        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, Tensor<f32>> {
        let mut state_dict = HashMap::new();

        for (name, param) in self.q_proj.named_parameters() {
            state_dict.insert(format!("q_proj.{}", name), param.tensor().read().clone());
        }
        for (name, param) in self.k_proj.named_parameters() {
            state_dict.insert(format!("k_proj.{}", name), param.tensor().read().clone());
        }
        for (name, param) in self.v_proj.named_parameters() {
            state_dict.insert(format!("v_proj.{}", name), param.tensor().read().clone());
        }
        for (name, param) in self.out_proj.named_parameters() {
            state_dict.insert(format!("out_proj.{}", name), param.tensor().read().clone());
        }

        state_dict
    }
}

impl MultiHeadAttention {
    fn extract_sub_dict(
        &self,
        state_dict: &HashMap<String, Tensor<f32>>,
        prefix: &str,
    ) -> HashMap<String, Tensor<f32>> {
        state_dict
            .iter()
            .filter_map(|(k, v)| {
                if k.starts_with(prefix) {
                    Some((k.strip_prefix(prefix).unwrap().to_string(), v.clone()))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Transformer Block
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    mlp: Sequential,
    norm2: LayerNorm,
    dropout: Dropout,
}

impl TransformerBlock {
    pub fn new(embed_dim: usize, num_heads: usize, mlp_dim: usize, dropout: f32) -> Self {
        let mlp = Sequential::new()
            .add(Linear::new(embed_dim, mlp_dim, true))
            .add(ReLU::new())
            .add(Dropout::new(dropout))
            .add(Linear::new(mlp_dim, embed_dim, true));

        Self {
            attention: MultiHeadAttention::new(embed_dim, num_heads, dropout),
            norm1: LayerNorm::new(vec![embed_dim], 1e-5, true, DeviceType::Cpu)
                .expect("Failed to create LayerNorm"),
            mlp,
            norm2: LayerNorm::new(vec![embed_dim], 1e-5, true, DeviceType::Cpu)
                .expect("Failed to create LayerNorm"),
            dropout: Dropout::new(dropout),
        }
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Self-attention with residual connection
        let attn_out = self.attention.forward(x)?;
        let x = &self.norm1.forward(&(&attn_out + x))?;

        // MLP with residual connection
        let mlp_out = self.mlp.forward(x)?;
        let output = self.norm2.forward(&(&mlp_out + x))?;

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.attention.parameters() {
            params.insert(format!("attention.{}", name), param);
        }
        for (name, param) in self.norm1.named_parameters() {
            params.insert(format!("norm1.{}", name), param);
        }
        for (name, param) in self.mlp.named_parameters() {
            params.insert(format!("mlp.{}", name), param);
        }
        for (name, param) in self.norm2.named_parameters() {
            params.insert(format!("norm2.{}", name), param);
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
        for (name, param) in self.mlp.named_parameters() {
            params.insert(format!("mlp.{}", name), param);
        }
        for (name, param) in self.norm2.named_parameters() {
            params.insert(format!("norm2.{}", name), param);
        }

        params
    }

    fn train(&mut self) {
        self.attention.train();
        self.norm1.train();
        self.mlp.train();
        self.norm2.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.attention.eval();
        self.norm1.eval();
        self.mlp.eval();
        self.norm2.eval();
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.attention.training()
    }

    fn load_state_dict(
        &mut self,
        _state_dict: &HashMap<String, Tensor<f32>>,
        _strict: bool,
    ) -> Result<()> {
        // Implementation would load all parameters
        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, Tensor<f32>> {
        // Implementation would save all parameters
        HashMap::new()
    }
}

/// BERT-like encoder model
pub struct BertEncoder {
    embeddings: BertEmbeddings,
    layers: Vec<TransformerBlock>,
    pooler: Linear,
    num_layers: usize,
}

impl BertEncoder {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        mlp_dim: usize,
        max_position_embeddings: usize,
        dropout: f32,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(TransformerBlock::new(
                embed_dim, num_heads, mlp_dim, dropout,
            ));
        }

        Ok(Self {
            embeddings: BertEmbeddings::new(
                vocab_size,
                embed_dim,
                max_position_embeddings,
                dropout,
            )?,
            layers,
            pooler: Linear::new(embed_dim, embed_dim, true),
            num_layers,
        })
    }

    /// Create BERT-Base configuration
    pub fn bert_base(vocab_size: usize) -> Result<Self> {
        Self::new(vocab_size, 768, 12, 12, 3072, 512, 0.1)
    }

    /// Create BERT-Large configuration
    pub fn bert_large(vocab_size: usize) -> Result<Self> {
        Self::new(vocab_size, 1024, 24, 16, 4096, 512, 0.1)
    }
}

impl Module for BertEncoder {
    fn forward(&self, input_ids: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        // Apply transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Pool first token ([CLS] token) for classification
        use torsh_tensor::creation::from_vec;
        let index_tensor = from_vec(vec![0i64], &[1], torsh_core::DeviceType::Cpu)?;
        let pooled_output = hidden_states.index_select(1, &index_tensor)?;
        let pooled_output = pooled_output.squeeze(1)?;
        let pooled_output = self.pooler.forward(&pooled_output)?;
        let pooled_output = pooled_output.tanh()?;

        Ok(pooled_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.embeddings.parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layers.{}.{}", i, name), param);
            }
        }
        for (name, param) in self.pooler.named_parameters() {
            params.insert(format!("pooler.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.embeddings.named_parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.named_parameters() {
                params.insert(format!("layers.{}.{}", i, name), param);
            }
        }
        for (name, param) in self.pooler.named_parameters() {
            params.insert(format!("pooler.{}", name), param);
        }

        params
    }

    fn train(&mut self) {
        self.embeddings.train();
        for layer in &mut self.layers {
            layer.train();
        }
        self.pooler.train();
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        for layer in &mut self.layers {
            layer.eval();
        }
        self.pooler.eval();
    }

    fn training(&self) -> bool {
        self.embeddings.training()
    }

    fn load_state_dict(
        &mut self,
        _state_dict: &HashMap<String, Tensor<f32>>,
        _strict: bool,
    ) -> Result<()> {
        // Implementation would load all parameters
        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, Tensor<f32>> {
        // Implementation would save all parameters
        HashMap::new()
    }
}

/// BERT Embeddings (token + position + token type)
pub struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    max_position_embeddings: usize,
}

impl BertEmbeddings {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        max_position_embeddings: usize,
        dropout: f32,
    ) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(vocab_size, embed_dim),
            position_embeddings: Embedding::new(max_position_embeddings, embed_dim),
            token_type_embeddings: Embedding::new(2, embed_dim), // For sentence A/B
            layer_norm: LayerNorm::new(vec![embed_dim], 1e-5, true, DeviceType::Cpu)
                .expect("Failed to create LayerNorm"),
            dropout: Dropout::new(dropout),
            max_position_embeddings,
        })
    }
}

impl Module for BertEmbeddings {
    fn forward(&self, input_ids: &Tensor<f32>) -> Result<Tensor<f32>> {
        let seq_len = input_ids.shape().dims()[1];

        // Word embeddings
        let inputs_embeds = self.word_embeddings.forward(input_ids)?;

        // Position embeddings
        let position_ids = torsh_tensor::creation::arange(0i64, seq_len as i64, 1i64)?
            .unsqueeze(0)?
            .expand(&[input_ids.shape().dims()[0], seq_len])?;
        let position_embeds = self.position_embeddings.forward({
            let pos_data = position_ids.to_vec()?;
            let pos_f32_data: Vec<f32> = pos_data.iter().map(|&x| x as f32).collect();
            &Tensor::from_data(
                pos_f32_data,
                position_ids.shape().dims().to_vec(),
                DeviceType::Cpu,
            )?
        })?;

        // Token type embeddings (default to 0)
        let token_type_ids =
            Tensor::zeros(&[input_ids.shape().dims()[0], seq_len], DeviceType::Cpu)?;
        let token_type_embeds = self.token_type_embeddings.forward(&token_type_ids)?;

        // Sum embeddings
        let embeddings = &(&inputs_embeds + &position_embeds) + &token_type_embeds;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;

        Ok(embeddings)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.word_embeddings.named_parameters() {
            params.insert(format!("word_embeddings.{}", name), param);
        }
        for (name, param) in self.position_embeddings.named_parameters() {
            params.insert(format!("position_embeddings.{}", name), param);
        }
        for (name, param) in self.token_type_embeddings.named_parameters() {
            params.insert(format!("token_type_embeddings.{}", name), param);
        }
        for (name, param) in self.layer_norm.named_parameters() {
            params.insert(format!("layer_norm.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.word_embeddings.named_parameters() {
            params.insert(format!("word_embeddings.{}", name), param);
        }
        for (name, param) in self.position_embeddings.named_parameters() {
            params.insert(format!("position_embeddings.{}", name), param);
        }
        for (name, param) in self.token_type_embeddings.named_parameters() {
            params.insert(format!("token_type_embeddings.{}", name), param);
        }
        for (name, param) in self.layer_norm.named_parameters() {
            params.insert(format!("layer_norm.{}", name), param);
        }

        params
    }

    fn train(&mut self) {
        self.word_embeddings.train();
        self.position_embeddings.train();
        self.token_type_embeddings.train();
        self.layer_norm.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.word_embeddings.eval();
        self.position_embeddings.eval();
        self.token_type_embeddings.eval();
        self.layer_norm.eval();
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.word_embeddings.training()
    }

    fn load_state_dict(
        &mut self,
        _state_dict: &HashMap<String, Tensor<f32>>,
        _strict: bool,
    ) -> Result<()> {
        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, Tensor<f32>> {
        HashMap::new()
    }
}

/// GPT-like decoder model
pub struct GPTDecoder {
    embeddings: GPTEmbeddings,
    layers: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    num_layers: usize,
}

impl GPTDecoder {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        mlp_dim: usize,
        max_position_embeddings: usize,
        dropout: f32,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(TransformerBlock::new(
                embed_dim, num_heads, mlp_dim, dropout,
            ));
        }

        Ok(Self {
            embeddings: GPTEmbeddings::new(
                vocab_size,
                embed_dim,
                max_position_embeddings,
                dropout,
            )?,
            layers,
            final_norm: LayerNorm::new(vec![embed_dim], 1e-5, true, DeviceType::Cpu)
                .expect("Failed to create LayerNorm"),
            num_layers,
        })
    }

    /// Create GPT-2 Small configuration
    pub fn gpt2_small(vocab_size: usize) -> Result<Self> {
        Self::new(vocab_size, 768, 12, 12, 3072, 1024, 0.1)
    }

    /// Create GPT-2 Medium configuration
    pub fn gpt2_medium(vocab_size: usize) -> Result<Self> {
        Self::new(vocab_size, 1024, 24, 16, 4096, 1024, 0.1)
    }
}

impl Module for GPTDecoder {
    fn forward(&self, input_ids: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        // Apply transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Final layer norm
        hidden_states = self.final_norm.forward(&hidden_states)?;

        Ok(hidden_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.embeddings.parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layers.{}.{}", i, name), param);
            }
        }
        for (name, param) in self.final_norm.named_parameters() {
            params.insert(format!("final_norm.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.embeddings.named_parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.named_parameters() {
                params.insert(format!("layers.{}.{}", i, name), param);
            }
        }
        for (name, param) in self.final_norm.named_parameters() {
            params.insert(format!("final_norm.{}", name), param);
        }

        params
    }

    fn train(&mut self) {
        self.embeddings.train();
        for layer in &mut self.layers {
            layer.train();
        }
        self.final_norm.train();
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        for layer in &mut self.layers {
            layer.eval();
        }
        self.final_norm.eval();
    }

    fn training(&self) -> bool {
        self.embeddings.training()
    }

    fn load_state_dict(
        &mut self,
        _state_dict: &HashMap<String, Tensor<f32>>,
        _strict: bool,
    ) -> Result<()> {
        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, Tensor<f32>> {
        HashMap::new()
    }
}

/// GPT Embeddings (token + position)
pub struct GPTEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    dropout: Dropout,
    max_position_embeddings: usize,
}

impl GPTEmbeddings {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        max_position_embeddings: usize,
        dropout: f32,
    ) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(vocab_size, embed_dim),
            position_embeddings: Embedding::new(max_position_embeddings, embed_dim),
            dropout: Dropout::new(dropout),
            max_position_embeddings,
        })
    }
}

impl Module for GPTEmbeddings {
    fn forward(&self, input_ids: &Tensor<f32>) -> Result<Tensor<f32>> {
        let seq_len = input_ids.shape().dims()[1];

        // Word embeddings
        let inputs_embeds = self.word_embeddings.forward(input_ids)?;

        // Position embeddings
        let position_ids = torsh_tensor::creation::arange(0i64, seq_len as i64, 1i64)?
            .unsqueeze(0)?
            .expand(&[input_ids.shape().dims()[0], seq_len])?;
        let position_embeds = self.position_embeddings.forward({
            let pos_data = position_ids.to_vec()?;
            let pos_f32_data: Vec<f32> = pos_data.iter().map(|&x| x as f32).collect();
            &Tensor::from_data(
                pos_f32_data,
                position_ids.shape().dims().to_vec(),
                DeviceType::Cpu,
            )?
        })?;

        // Sum embeddings
        let embeddings = &inputs_embeds + &position_embeds;
        let embeddings = self.dropout.forward(&embeddings)?;

        Ok(embeddings)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.word_embeddings.named_parameters() {
            params.insert(format!("word_embeddings.{}", name), param);
        }
        for (name, param) in self.position_embeddings.named_parameters() {
            params.insert(format!("position_embeddings.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.word_embeddings.named_parameters() {
            params.insert(format!("word_embeddings.{}", name), param);
        }
        for (name, param) in self.position_embeddings.named_parameters() {
            params.insert(format!("position_embeddings.{}", name), param);
        }

        params
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

    fn training(&self) -> bool {
        self.word_embeddings.training()
    }

    fn load_state_dict(
        &mut self,
        _state_dict: &HashMap<String, Tensor<f32>>,
        _strict: bool,
    ) -> Result<()> {
        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, Tensor<f32>> {
        HashMap::new()
    }
}

/// Convenience functions for creating pre-built NLP models
pub mod pretrained {
    use super::*;

    /// Load BERT-Base with pretrained weights
    pub fn bert_base_uncased(pretrained: bool) -> Result<Box<dyn Module>> {
        let vocab_size = 30522; // BERT vocabulary size
        let mut model = BertEncoder::bert_base(vocab_size)?;

        if pretrained {
            println!(
                "Loading BERT-Base with random weights (pretrained weights not implemented yet)"
            );
        }

        Ok(Box::new(model))
    }

    /// Load BERT-Large with pretrained weights
    pub fn bert_large_uncased(pretrained: bool) -> Result<Box<dyn Module>> {
        let vocab_size = 30522;
        let mut model = BertEncoder::bert_large(vocab_size)?;

        if pretrained {
            println!(
                "Loading BERT-Large with random weights (pretrained weights not implemented yet)"
            );
        }

        Ok(Box::new(model))
    }

    /// Load GPT-2 Small with pretrained weights
    pub fn gpt2_small(pretrained: bool) -> Result<Box<dyn Module>> {
        let vocab_size = 50257; // GPT-2 vocabulary size
        let mut model = GPTDecoder::gpt2_small(vocab_size)?;

        if pretrained {
            println!(
                "Loading GPT-2 Small with random weights (pretrained weights not implemented yet)"
            );
        }

        Ok(Box::new(model))
    }

    /// Load GPT-2 Medium with pretrained weights
    pub fn gpt2_medium(pretrained: bool) -> Result<Box<dyn Module>> {
        let vocab_size = 50257;
        let mut model = GPTDecoder::gpt2_medium(vocab_size)?;

        if pretrained {
            println!(
                "Loading GPT-2 Medium with random weights (pretrained weights not implemented yet)"
            );
        }

        Ok(Box::new(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randint;

    #[test]
    fn test_multi_head_attention_creation() {
        let attention = MultiHeadAttention::new(768, 12, 0.1);
        assert_eq!(attention.num_heads, 12);
        assert_eq!(attention.head_dim, 64);
    }

    #[test]
    fn test_transformer_block_creation() {
        let _block = TransformerBlock::new(768, 12, 3072, 0.1);
        // Test passes if no panic occurs
    }

    #[test]
    fn test_bert_encoder_creation() -> Result<()> {
        let model = BertEncoder::bert_base(30522)?;
        assert_eq!(model.num_layers, 12);
        Ok(())
    }

    #[test]
    fn test_gpt_decoder_creation() -> Result<()> {
        let model = GPTDecoder::gpt2_small(50257)?;
        assert_eq!(model.num_layers, 12);
        Ok(())
    }

    #[test]
    fn test_bert_embeddings_creation() -> Result<()> {
        let embeddings = BertEmbeddings::new(30522, 768, 512, 0.1)?;
        assert_eq!(embeddings.max_position_embeddings, 512);
        Ok(())
    }
}
