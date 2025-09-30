//! CLIP embedding layers

use super::config::{CLIPTextConfig, CLIPVisionConfig};
use std::collections::HashMap;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::type_conversions::*;
use torsh_tensor::{creation, Tensor};

/// CLIP Vision Embeddings - converts image patches to embeddings
pub struct CLIPVisionEmbeddings {
    patch_embedding: Conv2d,
    class_embedding: Parameter,
    position_embedding: Embedding,
    position_ids: Tensor<i64>,
    config: CLIPVisionConfig,
}

impl CLIPVisionEmbeddings {
    pub fn new(config: CLIPVisionConfig) -> Result<Self> {
        let patch_embedding = Conv2d::new(
            config.num_channels,
            config.hidden_size,
            (config.patch_size, config.patch_size),
            (config.patch_size, config.patch_size),
            (0, 0),
            (1, 1),
            false,
            1,
        );

        let class_embedding = Parameter::new(creation::randn(&[config.hidden_size])?);

        let num_patches = config.num_patches();
        let num_positions = config.sequence_length();
        let position_embedding = Embedding::new(num_positions, config.hidden_size);

        // Create position IDs tensor
        let position_ids = creation::arange(0i64, num_positions as i64, 1i64)?;

        Ok(Self {
            patch_embedding,
            class_embedding,
            position_embedding,
            position_ids,
            config,
        })
    }

    /// Get the number of patches
    pub fn num_patches(&self) -> usize {
        self.config.num_patches()
    }
}

impl Module for CLIPVisionEmbeddings {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.size(0)?;

        // Convert image patches to embeddings
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;

        // Flatten spatial dimensions: [batch, embed_dim, height, width] -> [batch, embed_dim, num_patches]
        let patch_embeds = patch_embeds.flatten()?;

        // Transpose to [batch, num_patches, embed_dim]
        let patch_embeds = patch_embeds.transpose(1, 2)?;

        // Add class token to each sample in the batch
        let class_embedding_tensor = self.class_embedding.tensor();
        let class_embedding_guard = class_embedding_tensor.read();
        let class_embeds = class_embedding_guard.unsqueeze(0)?.unsqueeze(0)?;
        let class_embeds = class_embeds.expand(&[batch_size, 1, self.config.hidden_size])?;

        // Concatenate class token with patch embeddings
        let embeddings = Tensor::cat(&[&class_embeds, &patch_embeds], 1)?;

        // Add positional embeddings
        let position_ids_f32 = self.position_ids.to_f32_simd()?;
        let position_embeddings = self.position_embedding.forward(&position_ids_f32)?;
        let position_embeddings = position_embeddings.unsqueeze(0)?.expand(&[
            batch_size,
            position_embeddings.size(0)?,
            self.config.hidden_size,
        ])?;

        let embeddings = embeddings.add(&position_embeddings)?;

        Ok(embeddings)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.patch_embedding.parameters() {
            params.insert(format!("patch_embedding.{}", name), param);
        }
        params.insert("class_embedding".to_string(), self.class_embedding.clone());
        for (name, param) in self.position_embedding.parameters() {
            params.insert(format!("position_embedding.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.patch_embedding.training()
    }

    fn train(&mut self) {
        self.patch_embedding.train();
    }

    fn eval(&mut self) {
        self.patch_embedding.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.patch_embedding.to_device(device)?;
        // Note: class_embedding and position_ids would also need device transfer
        self.position_embedding.to_device(device)?;
        Ok(())
    }
}

/// CLIP Text Embeddings
pub struct CLIPTextEmbeddings {
    token_embedding: Embedding,
    position_embedding: Embedding,
    position_ids: Tensor<i64>,
    config: CLIPTextConfig,
}

impl CLIPTextEmbeddings {
    pub fn new(config: CLIPTextConfig) -> Result<Self> {
        let token_embedding = Embedding::new(config.vocab_size, config.hidden_size);
        let position_embedding = Embedding::new(config.max_position_embeddings, config.hidden_size);

        // Create position IDs tensor
        let position_ids = creation::arange(0i64, config.max_position_embeddings as i64, 1i64)?;

        Ok(Self {
            token_embedding,
            position_embedding,
            position_ids,
            config,
        })
    }

    /// Create position IDs for a given sequence length
    pub fn create_position_ids(&self, seq_length: usize) -> Result<Tensor<i64>> {
        creation::arange(0i64, seq_length as i64, 1i64)
    }
}

impl Module for CLIPTextEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.size(1)?;
        let batch_size = input_ids.size(0)?;

        // Token embeddings
        let token_embeddings = self.token_embedding.forward(input_ids)?;

        // Position embeddings
        let position_ids = if seq_length <= self.config.max_position_embeddings {
            self.position_ids.narrow(0, 0, seq_length)?
        } else {
            self.create_position_ids(seq_length)?
        };

        let position_ids_f32 = position_ids.to_f32_simd()?;
        let position_embeddings = self.position_embedding.forward(&position_ids_f32)?;

        // Expand position embeddings to match batch size
        let position_embeddings = position_embeddings.unsqueeze(0)?.expand(&[
            batch_size,
            seq_length,
            self.config.hidden_size,
        ])?;

        // Add token and position embeddings
        let embeddings = token_embeddings.add(&position_embeddings)?;

        Ok(embeddings)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.token_embedding.parameters() {
            params.insert(format!("token_embedding.{}", name), param);
        }
        for (name, param) in self.position_embedding.parameters() {
            params.insert(format!("position_embedding.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.token_embedding.training() && self.position_embedding.training()
    }

    fn train(&mut self) {
        self.token_embedding.train();
        self.position_embedding.train();
    }

    fn eval(&mut self) {
        self.token_embedding.eval();
        self.position_embedding.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.token_embedding.to_device(device)?;
        self.position_embedding.to_device(device)?;
        Ok(())
    }
}
