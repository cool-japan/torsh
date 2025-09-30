//! ALIGN BERT Components
//!
//! BERT encoder components including transformer layers, attention mechanisms,
//! and feed-forward networks for ALIGN text processing.

use std::collections::HashMap;
use torsh_core::{error::Result, DeviceType};
use torsh_nn::prelude::{Dropout, LayerNorm, Linear, MultiheadAttention};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

use crate::multimodal::align::config::ALIGNTextConfig;

/// ALIGN BERT Encoder (stack of transformer layers)
pub struct ALIGNBertEncoder {
    layers: Vec<ALIGNBertLayer>,
}

impl ALIGNBertEncoder {
    pub fn new(config: ALIGNTextConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|_| ALIGNBertLayer::new(config.clone()))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { layers })
    }
}

impl Module for ALIGNBertEncoder {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        Ok(hidden_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer.{}.{}", i, name), param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.layers.iter().all(|layer| layer.training())
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

/// ALIGN BERT Layer (transformer block with attention and feed-forward)
pub struct ALIGNBertLayer {
    attention: ALIGNBertAttention,
    intermediate: ALIGNBertIntermediate,
    output: ALIGNBertOutput,
}

impl ALIGNBertLayer {
    pub fn new(config: ALIGNTextConfig) -> Result<Self> {
        let attention = ALIGNBertAttention::new(config.clone())?;
        let intermediate = ALIGNBertIntermediate::new(config.clone());
        let output = ALIGNBertOutput::new(config.clone())?;

        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }
}

impl Module for ALIGNBertLayer {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self.output.forward(&intermediate_output)?;

        Ok(layer_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.attention.parameters() {
            params.insert(format!("attention.{}", name), param);
        }

        for (name, param) in self.intermediate.parameters() {
            params.insert(format!("intermediate.{}", name), param);
        }

        for (name, param) in self.output.parameters() {
            params.insert(format!("output.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.attention.training() && self.intermediate.training() && self.output.training()
    }

    fn train(&mut self) {
        self.attention.train();
        self.intermediate.train();
        self.output.train();
    }

    fn eval(&mut self) {
        self.attention.eval();
        self.intermediate.eval();
        self.output.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.attention.to_device(device)?;
        self.intermediate.to_device(device)?;
        self.output.to_device(device)?;
        Ok(())
    }
}

/// ALIGN BERT Attention (multi-head self-attention)
pub struct ALIGNBertAttention {
    self_attention: MultiheadAttention,
    output: ALIGNBertSelfOutput,
}

impl ALIGNBertAttention {
    pub fn new(config: ALIGNTextConfig) -> Result<Self> {
        let self_attention =
            MultiheadAttention::new(config.hidden_size, config.num_attention_heads);
        let output = ALIGNBertSelfOutput::new(config)?;

        Ok(Self {
            self_attention,
            output,
        })
    }
}

impl Module for ALIGNBertAttention {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward(hidden_states)?;
        let attention_output = self.output.forward(&self_outputs)?;

        Ok(attention_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.self_attention.parameters() {
            params.insert(format!("self.{}", name), param);
        }

        for (name, param) in self.output.parameters() {
            params.insert(format!("output.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.self_attention.training() && self.output.training()
    }

    fn train(&mut self) {
        self.self_attention.train();
        self.output.train();
    }

    fn eval(&mut self) {
        self.self_attention.eval();
        self.output.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.self_attention.to_device(device)?;
        self.output.to_device(device)?;
        Ok(())
    }
}

/// ALIGN BERT Self Output (attention output projection with residual connection)
pub struct ALIGNBertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl ALIGNBertSelfOutput {
    pub fn new(config: ALIGNTextConfig) -> Result<Self> {
        let dense = Linear::new(config.hidden_size, config.hidden_size, true);
        let layer_norm = LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps as f64,
            true,
            torsh_core::DeviceType::Cpu,
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }
}

impl ALIGNBertSelfOutput {
    /// Forward pass with explicit hidden_states and input_tensor
    pub fn forward_with_residual(
        &self,
        hidden_states: &Tensor,
        input_tensor: &Tensor,
    ) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        let hidden_states = self.layer_norm.forward(&hidden_states.add(input_tensor)?)?;

        Ok(hidden_states)
    }
}

impl Module for ALIGNBertSelfOutput {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For Module trait, just apply dense + dropout + layer_norm without residual
        let output = self.dense.forward(input)?;
        let output = self.dropout.forward(&output)?;
        let output = self.layer_norm.forward(&output)?;
        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.dense.parameters() {
            params.insert(format!("dense.{}", name), param);
        }

        for (name, param) in self.layer_norm.parameters() {
            params.insert(format!("LayerNorm.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.dense.to_device(device)?;
        self.layer_norm.to_device(device)?;
        Ok(())
    }
}

/// ALIGN BERT Intermediate layer (feed-forward expansion)
#[derive(Debug)]
pub struct ALIGNBertIntermediate {
    dense: Linear,
}

impl ALIGNBertIntermediate {
    pub fn new(config: ALIGNTextConfig) -> Self {
        let dense = Linear::new(config.hidden_size, config.intermediate_size, true);

        Self { dense }
    }
}

impl Module for ALIGNBertIntermediate {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = hidden_states.gelu()?;

        Ok(hidden_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.dense.parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.dense.named_parameters()
    }

    fn training(&self) -> bool {
        true // Linear layer doesn't have training state
    }

    fn train(&mut self) {
        // Linear layer doesn't have training state
    }

    fn eval(&mut self) {
        // Linear layer doesn't have training state
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.dense.to_device(device)
    }
}

/// ALIGN BERT Output layer (feed-forward projection with residual connection)
pub struct ALIGNBertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl ALIGNBertOutput {
    pub fn new(config: ALIGNTextConfig) -> Result<Self> {
        let dense = Linear::new(config.intermediate_size, config.hidden_size, true);
        let layer_norm = LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps as f64,
            true,
            torsh_core::DeviceType::Cpu,
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }
}

impl ALIGNBertOutput {
    /// Forward pass with explicit hidden_states and input_tensor
    pub fn forward_with_residual(
        &self,
        hidden_states: &Tensor,
        input_tensor: &Tensor,
    ) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        let hidden_states = self.layer_norm.forward(&hidden_states.add(input_tensor)?)?;

        Ok(hidden_states)
    }
}

impl Module for ALIGNBertOutput {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For Module trait, just apply dense + dropout + layer_norm without residual
        let output = self.dense.forward(input)?;
        let output = self.dropout.forward(&output)?;
        let output = self.layer_norm.forward(&output)?;
        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.dense.parameters() {
            params.insert(format!("dense.{}", name), param);
        }

        for (name, param) in self.layer_norm.parameters() {
            params.insert(format!("LayerNorm.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.dense.to_device(device)?;
        self.layer_norm.to_device(device)?;
        Ok(())
    }
}
