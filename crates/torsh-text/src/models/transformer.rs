use crate::TextModelConfig;
use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_core::Result;
use torsh_nn::{prelude::*, Module, Parameter};
use torsh_tensor::Tensor;

/// Multi-head attention module
pub struct MultiHeadAttention {
    num_heads: usize,
    hidden_dim: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    dropout: Dropout,
    is_training: bool,
}

impl MultiHeadAttention {
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        dropout: f32,
        _device: DeviceType,
    ) -> Result<Self> {
        assert!(
            hidden_dim % num_heads == 0,
            "hidden_dim must be divisible by num_heads"
        );
        let head_dim = hidden_dim / num_heads;

        Ok(Self {
            num_heads,
            hidden_dim,
            head_dim,
            q_proj: Linear::new(hidden_dim, hidden_dim, true),
            k_proj: Linear::new(hidden_dim, hidden_dim, true),
            v_proj: Linear::new(hidden_dim, hidden_dim, true),
            out_proj: Linear::new(hidden_dim, hidden_dim, true),
            dropout: Dropout::new(dropout),
            is_training: true,
        })
    }

    fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let scale = (self.head_dim as f32).sqrt();

        // Compute attention scores
        let scores = query.matmul(&key.transpose(-2, -1)?)?.div_scalar(scale)?;

        // Apply attention mask if provided
        let scores = if let Some(mask) = attention_mask {
            scores.add(mask)?
        } else {
            scores
        };

        // Apply softmax
        let attn_weights = scores.softmax(-1)?;
        let attn_weights = self.dropout.forward(&attn_weights)?;

        // Apply attention to values
        attn_weights.matmul(value)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.q_proj.parameters() {
            params.insert(format!("q_proj.{}", name), param);
        }
        for (name, param) in self.k_proj.parameters() {
            params.insert(format!("k_proj.{}", name), param);
        }
        for (name, param) in self.v_proj.parameters() {
            params.insert(format!("v_proj.{}", name), param);
        }
        for (name, param) in self.out_proj.parameters() {
            params.insert(format!("out_proj.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        self.q_proj.train();
        self.k_proj.train();
        self.v_proj.train();
        self.out_proj.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.q_proj.eval();
        self.k_proj.eval();
        self.v_proj.eval();
        self.out_proj.eval();
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: torsh_core::device::DeviceType) -> Result<()> {
        self.q_proj.to_device(device)?;
        self.k_proj.to_device(device)?;
        self.v_proj.to_device(device)?;
        self.out_proj.to_device(device)?;
        Ok(())
    }
}

impl MultiHeadAttention {
    pub fn forward_with_mask(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let batch_size = input.size(0)?;
        let seq_len = input.size(1)?;

        // Project to Q, K, V
        let query = self.q_proj.forward(input)?;
        let key = self.k_proj.forward(input)?;
        let value = self.v_proj.forward(input)?;

        // Reshape for multi-head attention
        let query = query
            .view(&[
                batch_size as i32,
                seq_len as i32,
                self.num_heads as i32,
                self.head_dim as i32,
            ])?
            .transpose(1, 2)?;
        let key = key
            .view(&[
                batch_size as i32,
                seq_len as i32,
                self.num_heads as i32,
                self.head_dim as i32,
            ])?
            .transpose(1, 2)?;
        let value = value
            .view(&[
                batch_size as i32,
                seq_len as i32,
                self.num_heads as i32,
                self.head_dim as i32,
            ])?
            .transpose(1, 2)?;

        // Apply attention
        let attn_output =
            self.scaled_dot_product_attention(&query, &key, &value, attention_mask)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.view(&[
            batch_size as i32,
            seq_len as i32,
            self.hidden_dim as i32,
        ])?;

        // Final projection
        self.out_proj.forward(&attn_output)
    }
}

/// Feed-forward network
pub struct FeedForward {
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
    is_training: bool,
}

impl FeedForward {
    pub fn new(
        hidden_dim: usize,
        intermediate_dim: usize,
        dropout: f32,
        _device: DeviceType,
    ) -> Result<Self> {
        Ok(Self {
            fc1: Linear::new(hidden_dim, intermediate_dim, true),
            fc2: Linear::new(intermediate_dim, hidden_dim, true),
            dropout: Dropout::new(dropout),
            is_training: true,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.fc1.forward(input)?;
        let hidden = hidden.gelu()?; // Use GELU activation for modern transformer architectures
        let hidden = self.dropout.forward(&hidden)?;
        self.fc2.forward(&hidden)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.fc1.parameters() {
            params.insert(format!("fc1.{}", name), param);
        }
        for (name, param) in self.fc2.parameters() {
            params.insert(format!("fc2.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        self.fc1.train();
        self.fc2.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.fc1.eval();
        self.fc2.eval();
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: torsh_core::device::DeviceType) -> Result<()> {
        self.fc1.to_device(device)?;
        self.fc2.to_device(device)?;
        Ok(())
    }
}

/// Transformer encoder layer
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
    is_training: bool,
}

impl TransformerEncoderLayer {
    pub fn new(config: &TextModelConfig, device: DeviceType) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(
                config.hidden_dim,
                config.num_heads,
                config.attention_dropout,
                device,
            )
            .unwrap(),
            feed_forward: FeedForward::new(
                config.hidden_dim,
                config.intermediate_dim,
                config.dropout,
                device,
            )
            .unwrap(),
            norm1: LayerNorm::new(vec![config.hidden_dim]),
            norm2: LayerNorm::new(vec![config.hidden_dim]),
            dropout: Dropout::new(config.dropout),
            is_training: true,
        })
    }

    pub fn forward_with_mask(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self attention with residual and attention mask
        let attn_output = self.self_attn.forward_with_mask(input, attention_mask)?;
        let attn_output = self.dropout.forward(&attn_output)?;
        let hidden = input.add(&attn_output)?;
        let hidden = self.norm1.forward(&hidden)?;

        // Feed forward with residual
        let ff_output = self.feed_forward.forward(&hidden)?;
        let ff_output = self.dropout.forward(&ff_output)?;
        let output = hidden.add(&ff_output)?;
        self.norm2.forward(&output)
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Self attention with residual
        let attn_output = self.self_attn.forward(input)?;
        let attn_output = self.dropout.forward(&attn_output)?;
        let hidden = input.add(&attn_output)?;
        let hidden = self.norm1.forward(&hidden)?;

        // Feed forward with residual
        let ff_output = self.feed_forward.forward(&hidden)?;
        let ff_output = self.dropout.forward(&ff_output)?;
        let output = hidden.add(&ff_output)?;
        self.norm2.forward(&output)
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

    fn to_device(&mut self, device: torsh_core::device::DeviceType) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.feed_forward.to_device(device)?;
        self.norm1.to_device(device)?;
        self.norm2.to_device(device)?;
        Ok(())
    }
}

/// Transformer encoder
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
    is_training: bool,
}

impl TransformerEncoder {
    pub fn new(config: &TextModelConfig, device: DeviceType) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            layers.push(TransformerEncoderLayer::new(config, device)?);
        }

        Ok(Self {
            layers,
            is_training: true,
        })
    }

    pub fn forward_with_mask(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden = input.clone();

        for layer in &self.layers {
            hidden = layer.forward_with_mask(&hidden, attention_mask)?;
        }

        Ok(hidden)
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden = input.clone();

        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }

        Ok(hidden)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
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
    }

    fn eval(&mut self) {
        self.is_training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: torsh_core::device::DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}
