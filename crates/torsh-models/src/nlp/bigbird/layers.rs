//! BigBird Transformer Layers

use crate::nlp::bigbird::attention::BigBirdSparseAttention;
use crate::nlp::bigbird::config::BigBirdConfig;
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_tensor::Tensor;

pub struct BigBirdLayer {
    attention: BigBirdSparseAttention,
    intermediate: Linear,
    output: Linear,
    attention_layer_norm: LayerNorm,
    output_layer_norm: LayerNorm,
    dropout: Dropout,
    config: BigBirdConfig,
}

impl BigBirdLayer {
    pub fn new(config: BigBirdConfig) -> Result<Self> {
        Ok(Self {
            attention: BigBirdSparseAttention::new(config.clone())?,
            intermediate: Linear::new(config.hidden_size, config.intermediate_size, true),
            output: Linear::new(config.intermediate_size, config.hidden_size, true),
            attention_layer_norm: LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps as f64,
                true,
                torsh_core::DeviceType::Cpu,
            )?,
            output_layer_norm: LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps as f64,
                true,
                torsh_core::DeviceType::Cpu,
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
            config,
        })
    }

    pub fn config(&self) -> &BigBirdConfig {
        &self.config
    }
}

impl Module for BigBirdLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(input)?;
        let attention_output = self.dropout.forward(&attention_output)?;
        let hidden_states = input.add(&attention_output)?;
        let hidden_states = self.attention_layer_norm.forward(&hidden_states)?;

        let intermediate_output = self.intermediate.forward(&hidden_states)?;
        let intermediate_output = intermediate_output.gelu()?;
        let layer_output = self.output.forward(&intermediate_output)?;
        let layer_output = self.dropout.forward(&layer_output)?;
        let layer_output = hidden_states.add(&layer_output)?;
        self.output_layer_norm.forward(&layer_output)
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
        for (name, param) in self.attention_layer_norm.parameters() {
            params.insert(format!("attention_layer_norm.{}", name), param);
        }
        for (name, param) in self.output_layer_norm.parameters() {
            params.insert(format!("output_layer_norm.{}", name), param);
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
        self.attention.train();
        self.intermediate.train();
        self.output.train();
        self.attention_layer_norm.train();
        self.output_layer_norm.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.attention.eval();
        self.intermediate.eval();
        self.output.eval();
        self.attention_layer_norm.eval();
        self.output_layer_norm.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.attention.to_device(device)?;
        self.intermediate.to_device(device)?;
        self.output.to_device(device)?;
        self.attention_layer_norm.to_device(device)?;
        self.output_layer_norm.to_device(device)?;
        self.dropout.to_device(device)?;
        Ok(())
    }
}

pub struct BigBirdEncoder {
    layers: Vec<BigBirdLayer>,
    config: BigBirdConfig,
}

impl BigBirdEncoder {
    pub fn new(config: BigBirdConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(BigBirdLayer::new(config.clone())?);
        }
        Ok(Self { layers, config })
    }

    pub fn config(&self) -> &BigBirdConfig {
        &self.config
    }
}

impl Module for BigBirdEncoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden_states = input.clone();
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
        self.layers.first().map_or(false, |l| l.training())
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

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}
