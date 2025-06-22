//! Attention mechanism layers

use crate::{Module, ModuleBase, Parameter};
use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// Multi-head attention mechanism
pub struct MultiheadAttention {
    base: ModuleBase,
    embed_dim: usize,
    num_heads: usize,
    dropout: f32,
    bias: bool,
    add_bias_kv: bool,
    add_zero_attn: bool,
    kdim: Option<usize>,
    vdim: Option<usize>,
    batch_first: bool,
}

impl MultiheadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let mut base = ModuleBase::new();

        assert!(
            embed_dim % num_heads == 0,
            "embed_dim must be divisible by num_heads"
        );

        // Initialize projection weights
        let in_proj_weight = crate::init::xavier_uniform(&[3 * embed_dim, embed_dim]);
        let out_proj_weight = crate::init::xavier_uniform(&[embed_dim, embed_dim]);

        base.register_parameter("in_proj_weight".to_string(), Parameter::new(in_proj_weight));
        base.register_parameter(
            "out_proj.weight".to_string(),
            Parameter::new(out_proj_weight),
        );

        // Initialize biases
        let in_proj_bias = zeros(&[3 * embed_dim]);
        let out_proj_bias = zeros(&[embed_dim]);

        base.register_parameter("in_proj_bias".to_string(), Parameter::new(in_proj_bias));
        base.register_parameter("out_proj.bias".to_string(), Parameter::new(out_proj_bias));

        Self {
            base,
            embed_dim,
            num_heads,
            dropout: 0.0,
            bias: true,
            add_bias_kv: false,
            add_zero_attn: false,
            kdim: None,
            vdim: None,
            batch_first: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        embed_dim: usize,
        num_heads: usize,
        dropout: f32,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        kdim: Option<usize>,
        vdim: Option<usize>,
        batch_first: bool,
    ) -> Self {
        let mut attention = Self::new(embed_dim, num_heads);
        attention.dropout = dropout;
        attention.bias = bias;
        attention.add_bias_kv = add_bias_kv;
        attention.add_zero_attn = add_zero_attn;
        attention.kdim = kdim;
        attention.vdim = vdim;
        attention.batch_first = batch_first;
        attention
    }
}

impl Module for MultiheadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Multi-head attention forward pass
        // This is a simplified implementation
        // Real implementation would handle query, key, value projections and attention computation

        let in_proj_weight = self.base.parameters["in_proj_weight"]
            .tensor()
            .read()
            .clone();
        let out_proj_weight = self.base.parameters["out_proj.weight"]
            .tensor()
            .read()
            .clone();

        // Simplified projection - actual implementation would split into Q, K, V
        let projected = input.matmul(&in_proj_weight.transpose(0, 1)?)?;

        if self.bias {
            let in_proj_bias = self.base.parameters["in_proj_bias"].tensor().read().clone();
            let _projected = projected.add(&in_proj_bias)?;
        }

        // Simplified output projection
        let output = projected.matmul(&out_proj_weight.transpose(0, 1)?)?;

        if self.bias {
            let out_proj_bias = self.base.parameters["out_proj.bias"]
                .tensor()
                .read()
                .clone();
            return output.add(&out_proj_bias);
        }

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

impl std::fmt::Debug for MultiheadAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiheadAttention")
            .field("embed_dim", &self.embed_dim)
            .field("num_heads", &self.num_heads)
            .field("dropout", &self.dropout)
            .finish()
    }
}
