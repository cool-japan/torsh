//! Vision Transformer embedding layers

use super::config::{PatchEmbedStrategy, ViTConfig};
use std::collections::HashMap;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};

/// Patch embedding layer for Vision Transformer
pub struct PatchEmbedding {
    img_size: usize,
    patch_size: usize,
    num_patches: usize,
    embed_dim: usize,
    proj: Conv2d,
    strategy: PatchEmbedStrategy,
}

impl PatchEmbedding {
    pub fn new(config: &ViTConfig) -> Self {
        let num_patches = config.num_patches();

        let proj = Conv2d::new(
            config.in_channels,
            config.embed_dim,
            (config.patch_size, config.patch_size),
            (config.patch_size, config.patch_size), // stride = patch_size for non-overlapping patches
            (0, 0),
            (1, 1),
            false,
            1,
        );

        Self {
            img_size: config.img_size,
            patch_size: config.patch_size,
            num_patches,
            embed_dim: config.embed_dim,
            proj,
            strategy: config.patch_embed_strategy.clone(),
        }
    }

    /// Get the number of patches
    pub fn num_patches(&self) -> usize {
        self.num_patches
    }
}

impl Module for PatchEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Input shape: [batch_size, channels, height, width]
        let batch_size = x.size(0)?;

        // Apply patch embedding projection
        let x = self.proj.forward(x)?;

        // Reshape from [batch_size, embed_dim, H_patches, W_patches] to [batch_size, num_patches, embed_dim]
        let patches_h = self.img_size / self.patch_size;
        let patches_w = self.img_size / self.patch_size;

        let x = x.view(&[
            batch_size as i32,
            self.embed_dim as i32,
            (patches_h * patches_w) as i32,
        ])?;
        let x = x.transpose(1, 2)?; // [batch_size, num_patches, embed_dim]

        Ok(x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.proj.parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.proj.parameters() {
            params.insert(format!("proj.{}", name), param);
        }
        params
    }

    fn training(&self) -> bool {
        self.proj.training()
    }

    fn train(&mut self) {
        self.proj.train();
    }

    fn eval(&mut self) {
        self.proj.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.proj.to_device(device)
    }
}

/// Position embedding for Vision Transformer
pub struct PositionalEmbedding {
    pos_embed: Parameter,
    seq_length: usize,
    embed_dim: usize,
    learnable: bool,
}

impl PositionalEmbedding {
    /// Create learnable positional embeddings
    pub fn learnable(seq_length: usize, embed_dim: usize) -> Result<Self> {
        let pos_embed = Parameter::new(creation::randn(&[1, seq_length, embed_dim])?);

        Ok(Self {
            pos_embed,
            seq_length,
            embed_dim,
            learnable: true,
        })
    }

    /// Create sinusoidal positional embeddings (fixed)
    pub fn sinusoidal(seq_length: usize, embed_dim: usize) -> Result<Self> {
        let pos_embed = Self::create_sinusoidal_embeddings(seq_length, embed_dim)?;
        let pos_embed = Parameter::new(pos_embed);

        Ok(Self {
            pos_embed,
            seq_length,
            embed_dim,
            learnable: false,
        })
    }

    /// Create sinusoidal position embeddings
    fn create_sinusoidal_embeddings(seq_length: usize, embed_dim: usize) -> Result<Tensor> {
        let mut embeddings = Vec::with_capacity(seq_length * embed_dim);

        for pos in 0..seq_length {
            for i in 0..embed_dim {
                let angle = pos as f32 / 10000_f32.powf(2.0 * (i / 2) as f32 / embed_dim as f32);
                if i % 2 == 0 {
                    embeddings.push(angle.sin());
                } else {
                    embeddings.push(angle.cos());
                }
            }
        }

        Tensor::from_data(embeddings, vec![1, seq_length, embed_dim], DeviceType::Cpu)
    }

    /// Get position embeddings for given sequence length
    pub fn get_embeddings(&self, seq_length: usize) -> Result<Tensor> {
        if seq_length <= self.seq_length {
            // Use subset of embeddings
            self.pos_embed.tensor().read().narrow(1, 0, seq_length)
        } else {
            // Interpolate for longer sequences
            self.interpolate_embeddings(seq_length)
        }
    }

    /// Interpolate position embeddings for different sequence lengths
    fn interpolate_embeddings(&self, new_seq_length: usize) -> Result<Tensor> {
        // Simple interpolation - in practice, would use more sophisticated methods
        if new_seq_length == self.seq_length {
            return Ok(self.pos_embed.tensor().read().clone());
        }

        // For simplicity, just repeat the last embedding if extending
        if new_seq_length > self.seq_length {
            let pos_embed_tensor = self.pos_embed.tensor();
            let current = pos_embed_tensor.read();
            let last_embed = current.narrow(1, (self.seq_length - 1) as i64, 1)?;
            let mut extended = current.clone();

            for _ in self.seq_length..new_seq_length {
                extended = Tensor::cat(&[&extended, &last_embed], 1)?;
            }

            Ok(extended)
        } else {
            // Truncate if shorter
            self.pos_embed.tensor().read().narrow(1, 0, new_seq_length)
        }
    }
}

impl Module for PositionalEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_length = x.size(1)?;
        let pos_embeddings = self.get_embeddings(seq_length)?;

        // Expand to match batch size
        let batch_size = x.size(0)?;
        let pos_embeddings = pos_embeddings.expand(&[batch_size, seq_length, self.embed_dim])?;

        x.add(&pos_embeddings)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        if self.learnable {
            let mut params = HashMap::new();
            params.insert("pos_embed".to_string(), self.pos_embed.clone());
            params
        } else {
            HashMap::new() // Fixed embeddings don't have learnable parameters
        }
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        true // Stateless
    }

    fn train(&mut self) {
        // No-op for positional embeddings
    }

    fn eval(&mut self) {
        // No-op for positional embeddings
    }

    fn to_device(&mut self, _device: DeviceType) -> Result<()> {
        // Parameters are handled separately
        Ok(())
    }
}

/// Class token for Vision Transformer
pub struct ClassToken {
    cls_token: Parameter,
    embed_dim: usize,
}

impl ClassToken {
    pub fn new(embed_dim: usize) -> Result<Self> {
        let cls_token = Parameter::new(creation::randn(&[1, 1, embed_dim])?);

        Ok(Self {
            cls_token,
            embed_dim,
        })
    }
}

impl Module for ClassToken {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.size(0)?;

        // Expand class token to match batch size
        let cls_tokens = self
            .cls_token
            .tensor()
            .read()
            .expand(&[batch_size, 1, self.embed_dim])?;

        // Concatenate class token at the beginning
        Tensor::cat(&[&cls_tokens, x], 1)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("cls_token".to_string(), self.cls_token.clone());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        true
    }

    fn train(&mut self) {
        // No-op
    }

    fn eval(&mut self) {
        // No-op
    }

    fn to_device(&mut self, _device: DeviceType) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_embedding_creation() {
        let config = ViTConfig::vit_base_patch16_224();
        let patch_embed = PatchEmbedding::new(&config);

        assert_eq!(patch_embed.num_patches(), 196); // (224/16)^2
        assert_eq!(patch_embed.embed_dim, 768);
    }

    #[test]
    fn test_positional_embedding_creation() -> Result<()> {
        let pos_embed = PositionalEmbedding::learnable(197, 768)?; // 196 patches + 1 class token
        assert_eq!(pos_embed.seq_length, 197);
        assert_eq!(pos_embed.embed_dim, 768);
        assert!(pos_embed.learnable);

        let sin_embed = PositionalEmbedding::sinusoidal(197, 768)?;
        assert!(!sin_embed.learnable);

        Ok(())
    }

    #[test]
    fn test_class_token_creation() -> Result<()> {
        let cls_token = ClassToken::new(768)?;
        assert_eq!(cls_token.embed_dim, 768);
        Ok(())
    }

    /*
    // These tests would require actual tensor operations
    #[test]
    fn test_patch_embedding_forward() -> Result<()> {
        let config = ViTConfig::vit_base_patch16_224();
        let mut patch_embed = PatchEmbedding::new(&config);

        let input = creation::randn(&[2, 3, 224, 224])?;
        let output = patch_embed.forward(&input)?;

        assert_eq!(output.shape(), &[2, 196, 768]); // [batch, num_patches, embed_dim]
        Ok(())
    }
    */
}
