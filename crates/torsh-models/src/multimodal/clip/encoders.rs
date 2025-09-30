//! CLIP vision and text encoders

use super::config::{CLIPTextConfig, CLIPVisionConfig};
use super::embeddings::{CLIPTextEmbeddings, CLIPVisionEmbeddings};
use super::layers::CLIPEncoderLayer;
use std::collections::HashMap;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// CLIP Vision Transformer
pub struct CLIPVisionTransformer {
    embeddings: CLIPVisionEmbeddings,
    encoder: CLIPVisionEncoder,
    layernorm: LayerNorm,
    config: CLIPVisionConfig,
}

impl CLIPVisionTransformer {
    pub fn new(config: CLIPVisionConfig) -> Result<Self> {
        let embeddings = CLIPVisionEmbeddings::new(config.clone())?;
        let encoder = CLIPVisionEncoder::new(config.clone())?;
        let layernorm = LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps as f64,
            true,
            torsh_core::DeviceType::Cpu,
        )?;

        Ok(Self {
            embeddings,
            encoder,
            layernorm,
            config,
        })
    }
}

impl Module for CLIPVisionTransformer {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(pixel_values)?;
        let encoder_outputs = self.encoder.forward(&hidden_states)?;
        let sequence_output = self.layernorm.forward(&encoder_outputs)?;

        // Extract [CLS] token representation (first token)
        let pooled_output = sequence_output.select(1, 0)?;

        Ok(pooled_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.embeddings.parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }
        for (name, param) in self.encoder.parameters() {
            params.insert(format!("encoder.{}", name), param);
        }
        for (name, param) in self.layernorm.parameters() {
            params.insert(format!("layernorm.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.embeddings.training() && self.encoder.training() && self.layernorm.training()
    }

    fn train(&mut self) {
        self.embeddings.train();
        self.encoder.train();
        self.layernorm.train();
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        self.encoder.eval();
        self.layernorm.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        self.layernorm.to_device(device)?;
        Ok(())
    }
}

/// CLIP Vision Encoder (stack of transformer layers)
pub struct CLIPVisionEncoder {
    layers: Vec<CLIPEncoderLayer>,
}

impl CLIPVisionEncoder {
    pub fn new(config: CLIPVisionConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|_| CLIPEncoderLayer::new(config.clone()))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { layers })
    }
}

impl Module for CLIPVisionEncoder {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut current_states = hidden_states.clone();

        for layer in &self.layers {
            current_states = layer.forward(&current_states)?;
        }

        Ok(current_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layers.{}.{}", layer_idx, name), param);
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

/// CLIP Text Transformer
pub struct CLIPTextTransformer {
    embeddings: CLIPTextEmbeddings,
    encoder: CLIPTextEncoder,
    final_layer_norm: LayerNorm,
    config: CLIPTextConfig,
}

impl CLIPTextTransformer {
    pub fn new(config: CLIPTextConfig) -> Result<Self> {
        let embeddings = CLIPTextEmbeddings::new(config.clone())?;
        let encoder = CLIPTextEncoder::new(config.clone())?;
        let final_layer_norm = LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps as f64,
            true,
            torsh_core::DeviceType::Cpu,
        )?;

        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            config,
        })
    }
}

impl Module for CLIPTextTransformer {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(input_ids)?;
        let encoder_outputs = self.encoder.forward(&hidden_states)?;
        let sequence_output = self.final_layer_norm.forward(&encoder_outputs)?;

        // Find the last non-padding token for each sequence in the batch
        // For now, use the last token as a simple pooling strategy
        let seq_length = sequence_output.size(1)?;
        let pooled_output = sequence_output.select(1, (seq_length - 1) as i64)?;

        Ok(pooled_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.embeddings.parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }
        for (name, param) in self.encoder.parameters() {
            params.insert(format!("encoder.{}", name), param);
        }
        for (name, param) in self.final_layer_norm.parameters() {
            params.insert(format!("final_layer_norm.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.embeddings.training() && self.encoder.training() && self.final_layer_norm.training()
    }

    fn train(&mut self) {
        self.embeddings.train();
        self.encoder.train();
        self.final_layer_norm.train();
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        self.encoder.eval();
        self.final_layer_norm.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        self.final_layer_norm.to_device(device)?;
        Ok(())
    }
}

/// CLIP Text Encoder (stack of transformer layers)
pub struct CLIPTextEncoder {
    layers: Vec<CLIPEncoderLayer>,
}

impl CLIPTextEncoder {
    pub fn new(config: CLIPTextConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|_| CLIPEncoderLayer::new_for_text(config.clone()))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { layers })
    }
}

impl Module for CLIPTextEncoder {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut current_states = hidden_states.clone();

        for layer in &self.layers {
            current_states = layer.forward(&current_states)?;
        }

        Ok(current_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layers.{}.{}", layer_idx, name), param);
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
