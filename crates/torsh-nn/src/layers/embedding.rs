//! Embedding layers

use crate::{Module, ModuleBase, Parameter};
use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation::*, Tensor};

/// Embedding layer that maps discrete tokens to continuous vectors
pub struct Embedding {
    base: ModuleBase,
    num_embeddings: usize,
    embedding_dim: usize,
    padding_idx: Option<usize>,
    max_norm: Option<f32>,
    norm_type: f32,
    scale_grad_by_freq: bool,
    sparse: bool,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let mut base = ModuleBase::new();
        
        // Initialize embedding weight matrix
        let weight = crate::init::xavier_uniform(&[num_embeddings, embedding_dim]);
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        Self {
            base,
            num_embeddings,
            embedding_dim,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
        }
    }

    pub fn with_padding_idx(num_embeddings: usize, embedding_dim: usize, padding_idx: usize) -> Self {
        let mut embedding = Self::new(num_embeddings, embedding_dim);
        embedding.padding_idx = Some(padding_idx);
        embedding
    }

    pub fn with_config(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        max_norm: Option<f32>,
        norm_type: f32,
        scale_grad_by_freq: bool,
        sparse: bool,
    ) -> Self {
        let mut embedding = Self::new(num_embeddings, embedding_dim);
        embedding.padding_idx = padding_idx;
        embedding.max_norm = max_norm;
        embedding.norm_type = norm_type;
        embedding.scale_grad_by_freq = scale_grad_by_freq;
        embedding.sparse = sparse;
        embedding
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Embedding lookup
        // Input shape: any shape containing indices
        // Output shape: input_shape + [embedding_dim]
        
        let weight = self.base.parameters["weight"].tensor().read().clone();
        
        // This is a simplified implementation
        // Real implementation would perform proper embedding lookup
        let input_shape = input.shape();
        let mut output_shape = input_shape.clone();
        output_shape.push(self.embedding_dim);
        
        // Placeholder - actual implementation would index into weight matrix
        let output = zeros(&output_shape);
        
        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl std::fmt::Debug for Embedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedding")
            .field("num_embeddings", &self.num_embeddings)
            .field("embedding_dim", &self.embedding_dim)
            .field("padding_idx", &self.padding_idx)
            .finish()
    }
}