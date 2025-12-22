//! Common neural network modules

use crate::{Module, ModuleBase, Parameter};
use parking_lot::RwLock;
use scirs2_neural::{activations as sci_act, layers as sci_layers};
use std::collections::HashMap;
use std::sync::Arc;
use torsh_autograd::prelude::*;
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation::*, Tensor};

/// Linear (fully connected) layer
pub struct Linear {
    base: ModuleBase,
    in_features: usize,
    out_features: usize,
    use_bias: bool,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let mut base = ModuleBase::new();

        // Initialize weight with shape [in_features, out_features] for direct matmul
        // This way input[batch, in_features] @ weight[in_features, out_features] = output[batch, out_features]
        let weight = crate::init::xavier_uniform(&[in_features, out_features]);
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        if bias {
            let bias_tensor = zeros(&[out_features])?;
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }

        Self {
            base,
            in_features,
            out_features,
            use_bias: bias,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified linear transformation using basic tensor operations
        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Compute input @ weight
        // Weight is now [in_features, out_features], so direct matmul works
        let output = input.matmul(&weight)?;

        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            output.add_op(&bias)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base.all_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        self.base.all_named_parameters()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        // Move all parameters to device
        for param in self.parameters() {
            let mut tensor = param.write();
            *tensor = tensor.clone().to(device)?;
        }
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!(
            "in_features={}, out_features={}, bias={}",
            self.in_features, self.out_features, self.use_bias
        )
    }
}

/// 1D Convolutional layer
pub struct Conv1d {
    base: ModuleBase,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    use_bias: bool,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        dilation: Option<usize>,
        groups: Option<usize>,
        bias: bool,
    ) -> Self {
        let mut base = ModuleBase::new();

        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);
        let groups = groups.unwrap_or(1);

        let weight_shape = [out_channels, in_channels / groups, kernel_size];
        let weight = crate::init::kaiming_uniform(&weight_shape, "fan_out");
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        if bias {
            let bias_tensor = zeros(&[out_channels])?;
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }

        Self {
            base,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias: bias,
        }
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.base.parameters["weight"].tensor().read().clone();

        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            crate::functional::conv1d(
                input,
                &weight,
                Some(&bias),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        } else {
            crate::functional::conv1d(
                input,
                &weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        }
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base.all_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        self.base.all_named_parameters()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        for param in self.parameters() {
            let mut tensor = param.write();
            *tensor = tensor.clone().to(device)?;
        }
        Ok(())
    }
}

/// Multi-Head Attention module
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
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        dropout: Option<f32>,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        kdim: Option<usize>,
        vdim: Option<usize>,
        batch_first: bool,
    ) -> Self {
        let mut base = ModuleBase::new();
        let dropout = dropout.unwrap_or(0.0);

        let kdim = kdim.unwrap_or(embed_dim);
        let vdim = vdim.unwrap_or(embed_dim);

        // Create input projection weights
        let in_proj_weight = crate::init::xavier_uniform(&[3 * embed_dim, embed_dim]);
        base.register_parameter("in_proj_weight".to_string(), Parameter::new(in_proj_weight));

        if bias {
            let in_proj_bias = zeros(&[3 * embed_dim])?;
            base.register_parameter("in_proj_bias".to_string(), Parameter::new(in_proj_bias));
        }

        // Output projection
        let out_proj_weight = crate::init::xavier_uniform(&[embed_dim, embed_dim]);
        base.register_parameter(
            "out_proj.weight".to_string(),
            Parameter::new(out_proj_weight),
        );

        if bias {
            let out_proj_bias = zeros(&[embed_dim])?;
            base.register_parameter("out_proj.bias".to_string(), Parameter::new(out_proj_bias));
        }

        Self {
            base,
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim: Some(kdim),
            vdim: Some(vdim),
            batch_first,
        }
    }
}

impl Module for MultiheadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Implement scaled dot-product multi-head attention
        // Simplified version: self-attention with query=key=value=input
        // Input: [batch, seq_len, embed_dim]
        // Output: [batch, seq_len, embed_dim]

        let input_shape = input.shape().dims();
        if input_shape.len() != 3 {
            return Err(torsh_core::error::TorshError::InvalidShape(
                format!("MultiheadAttention expects 3D input [batch, seq, embed], got {}D", input_shape.len())
            ));
        }

        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let embed_dim = input_shape[2];

        // Get projection weights
        let q_proj = self.base.parameters.get("q_proj_weight")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing q_proj_weight".to_string()))?
            .tensor().read();
        let k_proj = self.base.parameters.get("k_proj_weight")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing k_proj_weight".to_string()))?
            .tensor().read();
        let v_proj = self.base.parameters.get("v_proj_weight")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing v_proj_weight".to_string()))?
            .tensor().read();
        let out_proj = self.base.parameters.get("out_proj_weight")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing out_proj_weight".to_string()))?
            .tensor().read();

        let q_w = q_proj.to_vec()?;
        let k_w = k_proj.to_vec()?;
        let v_w = v_proj.to_vec()?;
        let o_w = out_proj.to_vec()?;
        let input_data = input.to_vec()?;

        let head_dim = embed_dim / self.num_heads;
        let scale = (head_dim as f32).sqrt();

        // Storage for output
        let mut output_data = vec![0.0f32; batch_size * seq_len * embed_dim];

        // Process each batch
        for b in 0..batch_size {
            // Project Q, K, V for all positions
            let mut queries = vec![vec![0.0f32; embed_dim]; seq_len];
            let mut keys = vec![vec![0.0f32; embed_dim]; seq_len];
            let mut values = vec![vec![0.0f32; embed_dim]; seq_len];

            for pos in 0..seq_len {
                let input_base = b * seq_len * embed_dim + pos * embed_dim;

                // Q = input @ W_q
                for i in 0..embed_dim {
                    for j in 0..embed_dim {
                        let inp = input_data[input_base + j];
                        queries[pos][i] += inp * q_w[i * embed_dim + j];
                        keys[pos][i] += inp * k_w[i * embed_dim + j];
                        values[pos][i] += inp * v_w[i * embed_dim + j];
                    }
                }
            }

            // Process each head
            let mut head_outputs = vec![vec![vec![0.0f32; head_dim]; seq_len]; self.num_heads];

            for h in 0..self.num_heads {
                let head_start = h * head_dim;
                let head_end = head_start + head_dim;

                // Compute attention for this head
                for q_pos in 0..seq_len {
                    let mut attention_weights = vec![0.0f32; seq_len];
                    let mut max_score = f32::NEG_INFINITY;

                    // Compute attention scores
                    for k_pos in 0..seq_len {
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            let q_val = queries[q_pos][head_start + d];
                            let k_val = keys[k_pos][head_start + d];
                            score += q_val * k_val;
                        }
                        score /= scale;
                        attention_weights[k_pos] = score;
                        max_score = max_score.max(score);
                    }

                    // Softmax
                    let mut sum_exp = 0.0f32;
                    for weight in attention_weights.iter_mut() {
                        *weight = (*weight - max_score).exp();
                        sum_exp += *weight;
                    }
                    for weight in attention_weights.iter_mut() {
                        *weight /= sum_exp;
                    }

                    // Weighted sum of values
                    for d in 0..head_dim {
                        let mut sum = 0.0f32;
                        for k_pos in 0..seq_len {
                            sum += attention_weights[k_pos] * values[k_pos][head_start + d];
                        }
                        head_outputs[h][q_pos][d] = sum;
                    }
                }
            }

            // Concatenate heads and project
            for pos in 0..seq_len {
                // Concatenate all heads
                let mut concat = vec![0.0f32; embed_dim];
                for h in 0..self.num_heads {
                    for d in 0..head_dim {
                        concat[h * head_dim + d] = head_outputs[h][pos][d];
                    }
                }

                // Output projection
                for i in 0..embed_dim {
                    let mut sum = 0.0f32;
                    for j in 0..embed_dim {
                        sum += concat[j] * o_w[i * embed_dim + j];
                    }
                    let output_idx = b * seq_len * embed_dim + pos * embed_dim + i;
                    output_data[output_idx] = sum;
                }
            }
        }

        Tensor::from_vec(output_data, &[batch_size, seq_len, embed_dim])
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base.all_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        self.base.all_named_parameters()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        for param in self.parameters() {
            let mut tensor = param.write();
            *tensor = tensor.clone().to(device)?;
        }
        Ok(())
    }
}

/// Embedding layer
pub struct Embedding {
    base: ModuleBase,
    num_embeddings: usize,
    embedding_dim: usize,
    padding_idx: Option<i64>,
    max_norm: Option<f32>,
    norm_type: f32,
    scale_grad_by_freq: bool,
    sparse: bool,
}

impl Embedding {
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<i64>,
        max_norm: Option<f32>,
        norm_type: Option<f32>,
        scale_grad_by_freq: bool,
        sparse: bool,
    ) -> Self {
        let mut base = ModuleBase::new();
        let norm_type = norm_type.unwrap_or(2.0);

        // Initialize embedding weights
        let weight = crate::init::xavier_uniform(&[num_embeddings, embedding_dim]);
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        Self {
            base,
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
        }
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Implement proper embedding lookup
        // Input: [batch_size, seq_len] or [seq_len] containing indices
        // Output: [batch_size, seq_len, embedding_dim] or [seq_len, embedding_dim]

        let weight = self.base.parameters["weight"].tensor().read();
        let indices = input.to_vec()?;
        let input_shape = input.shape().dims();

        // Get embedding dimensions
        let weight_shape = weight.shape().dims();
        let embedding_dim = weight_shape[1];

        // Get weight data
        let weight_data = weight.to_vec()?;

        // Build output
        let total_lookups = indices.len();
        let mut output_data = Vec::with_capacity(total_lookups * embedding_dim);

        for &idx_f32 in indices.iter() {
            let idx = idx_f32 as usize;

            // Bounds check
            if idx >= self.num_embeddings {
                return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                    "Index {} out of bounds for embedding with {} entries", idx, self.num_embeddings
                )));
            }

            // Copy embedding vector
            let start = idx * embedding_dim;
            let end = start + embedding_dim;
            output_data.extend_from_slice(&weight_data[start..end]);
        }

        // Create output shape
        let mut output_shape = input_shape.to_vec();
        output_shape.push(embedding_dim);

        Tensor::from_vec(output_data, &output_shape)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base.all_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        self.base.all_named_parameters()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        for param in self.parameters() {
            let mut tensor = param.write();
            *tensor = tensor.clone().to(device)?;
        }
        Ok(())
    }
}

/// ReLU activation layer
pub struct ReLU {
    base: ModuleBase,
    inplace: bool,
}

impl ReLU {
    pub fn new(inplace: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            inplace,
        }
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::functional::relu(input)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!("inplace={}", self.inplace)
    }
}

/// Sigmoid activation layer
pub struct Sigmoid {
    base: ModuleBase,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(crate::functional::sigmoid(input))
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        Ok(())
    }
}

/// Tanh activation layer
pub struct Tanh {
    base: ModuleBase,
}

impl Tanh {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(crate::functional::tanh(input))
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        Ok(())
    }
}

/// MaxPool2d layer
pub struct MaxPool2d {
    base: ModuleBase,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    dilation: (usize, usize),
    return_indices: bool,
    ceil_mode: bool,
}

impl MaxPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        return_indices: bool,
        ceil_mode: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            stride,
            padding: padding.unwrap_or((0, 0)),
            dilation: dilation.unwrap_or((1, 1)),
            return_indices,
            ceil_mode,
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::functional::max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            Some(self.padding),
            Some(self.dilation),
        )
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!(
            "kernel_size={:?}, stride={:?}, padding={:?}",
            self.kernel_size, self.stride, self.padding
        )
    }
}

/// AvgPool2d layer
pub struct AvgPool2d {
    base: ModuleBase,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Option<i32>,
}

impl AvgPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Option<i32>,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            stride,
            padding: padding.unwrap_or((0, 0)),
            ceil_mode,
            count_include_pad,
            divisor_override,
        }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::functional::avg_pool2d(input, self.kernel_size, self.stride, Some(self.padding))
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!(
            "kernel_size={:?}, stride={:?}, padding={:?}",
            self.kernel_size, self.stride, self.padding
        )
    }
}

/// AdaptiveAvgPool2d layer
pub struct AdaptiveAvgPool2d {
    base: ModuleBase,
    output_size: (Option<usize>, Option<usize>),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (Option<usize>, Option<usize>)) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
        }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::functional::adaptive_avg_pool2d(input, self.output_size)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!("output_size={:?}", self.output_size)
    }
}

/// 2D Convolutional layer
pub struct Conv2d {
    base: ModuleBase,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    use_bias: bool,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        groups: Option<usize>,
        bias: bool,
    ) -> Self {
        let mut base = ModuleBase::new();

        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        let dilation = dilation.unwrap_or((1, 1));
        let groups = groups.unwrap_or(1);

        // Initialize weights
        let weight_shape = [
            out_channels,
            in_channels / groups,
            kernel_size.0,
            kernel_size.1,
        ];
        let weight = crate::init::kaiming_uniform(&weight_shape, "fan_out");
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        if bias {
            let bias_tensor = zeros(&[out_channels])?;
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }

        Self {
            base,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias: bias,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.base.parameters["weight"].tensor().read().clone();

        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            crate::functional::conv2d(
                input,
                &weight,
                Some(&bias),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        } else {
            crate::functional::conv2d(
                input,
                &weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        }
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base.all_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        self.base.all_named_parameters()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        for param in self.parameters() {
            let mut tensor = param.write();
            *tensor = tensor.clone().to(device)?;
        }
        Ok(())
    }
}

/// BatchNorm2d layer
pub struct BatchNorm2d {
    base: ModuleBase,
    num_features: usize,
    eps: f32,
    momentum: f32,
    affine: bool,
    track_running_stats: bool,
}

impl BatchNorm2d {
    pub fn new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: bool,
        track_running_stats: bool,
    ) -> Self {
        let mut base = ModuleBase::new();
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);

        if affine {
            let weight = ones(&[num_features])?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));

            let bias = zeros(&[num_features])?;
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        if track_running_stats {
            let running_mean = zeros(&[num_features])?;
            base.register_buffer("running_mean".to_string(), running_mean);

            let running_var = ones(&[num_features])?;
            base.register_buffer("running_var".to_string(), running_var);

            let num_batches_tracked = zeros(&[])?;
            base.register_buffer("num_batches_tracked".to_string(), num_batches_tracked);
        }

        Self {
            base,
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplify by cloning tensors for now
        let running_mean = if self.track_running_stats {
            Some(self.base.buffers["running_mean"].read().clone())
        } else {
            None
        };

        let running_var = if self.track_running_stats {
            Some(self.base.buffers["running_var"].read().clone())
        } else {
            None
        };

        let weight = if self.affine {
            Some(self.base.parameters["weight"].tensor().read().clone())
        } else {
            None
        };

        let bias = if self.affine {
            Some(self.base.parameters["bias"].tensor().read().clone())
        } else {
            None
        };

        crate::functional::batch_norm(
            input,
            running_mean.as_ref(),
            running_var.as_ref(),
            weight.as_ref(),
            bias.as_ref(),
            self.base.training,
            self.momentum,
            self.eps,
        )
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base.all_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        self.base.all_named_parameters()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        for param in self.parameters() {
            let mut tensor = param.write();
            *tensor = tensor.clone().to(device)?;
        }
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!(
            "num_features={}, eps={}, momentum={}, affine={}, track_running_stats={}",
            self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats
        )
    }
}

/// LayerNorm layer
pub struct LayerNorm {
    base: ModuleBase,
    normalized_shape: Vec<usize>,
    eps: f32,
    elementwise_affine: bool,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: Option<f32>, elementwise_affine: bool) -> Self {
        let mut base = ModuleBase::new();
        let eps = eps.unwrap_or(1e-5);

        if elementwise_affine {
            let weight = ones(&normalized_shape)?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));

            let bias = zeros(&normalized_shape)?;
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Self {
            base,
            normalized_shape,
            eps,
            elementwise_affine,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = if self.elementwise_affine {
            Some(self.base.parameters["weight"].tensor().read().clone())
        } else {
            None
        };

        let bias = if self.elementwise_affine {
            Some(self.base.parameters["bias"].tensor().read().clone())
        } else {
            None
        };

        crate::functional::layer_norm(
            input,
            &self.normalized_shape,
            weight.as_ref(),
            bias.as_ref(),
            self.eps,
        )
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base.all_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        self.base.all_named_parameters()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        for param in self.parameters() {
            let mut tensor = param.write();
            *tensor = tensor.clone().to(device)?;
        }
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!(
            "normalized_shape={:?}, eps={}, elementwise_affine={}",
            self.normalized_shape, self.eps, self.elementwise_affine
        )
    }
}

/// Dropout layer
pub struct Dropout {
    base: ModuleBase,
    p: f32,
    inplace: bool,
}

impl Dropout {
    pub fn new(p: f32, inplace: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            p,
            inplace,
        }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(crate::functional::dropout(
            input,
            self.p,
            self.base.training,
        ))
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!("p={}, inplace={}", self.p, self.inplace)
    }
}

/// GELU activation layer
pub struct GELU {
    base: ModuleBase,
    approximate: String,
}

impl GELU {
    pub fn new(approximate: Option<String>) -> Self {
        Self {
            base: ModuleBase::new(),
            approximate: approximate.unwrap_or_else(|| "none".to_string()),
        }
    }
}

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(crate::functional::gelu(input))
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!("approximate={}", self.approximate)
    }
}

/// LeakyReLU activation layer
pub struct LeakyReLU {
    base: ModuleBase,
    negative_slope: f32,
    inplace: bool,
}

impl LeakyReLU {
    pub fn new(negative_slope: f32, inplace: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            negative_slope,
            inplace,
        }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(crate::functional::leaky_relu(input, self.negative_slope))
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!(
            "negative_slope={}, inplace={}",
            self.negative_slope, self.inplace
        )
    }
}

/// Softmax activation layer
pub struct Softmax {
    base: ModuleBase,
    dim: Option<i32>,
}

impl Softmax {
    pub fn new(dim: Option<i32>) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::functional::softmax(input, self.dim.unwrap_or(-1))
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!("dim={:?}", self.dim)
    }
}

/// LogSoftmax activation layer
pub struct LogSoftmax {
    base: ModuleBase,
    dim: Option<i32>,
}

impl LogSoftmax {
    pub fn new(dim: Option<i32>) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }
}

impl Module for LogSoftmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::functional::log_softmax(input, self.dim.unwrap_or(-1))
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!("dim={:?}", self.dim)
    }
}

/// Basic RNN layer
pub struct RNN {
    base: ModuleBase,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bias: bool,
    batch_first: bool,
    dropout: f32,
    bidirectional: bool,
}

impl RNN {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        bias: bool,
        batch_first: bool,
        dropout: Option<f32>,
        bidirectional: bool,
    ) -> Self {
        let mut base = ModuleBase::new();
        let num_layers = num_layers.unwrap_or(1);
        let dropout = dropout.unwrap_or(0.0);

        // Initialize weights for each layer
        for layer in 0..num_layers {
            let input_dim = if layer == 0 { input_size } else { hidden_size };

            // Input-to-hidden weights
            let w_ih = crate::init::xavier_uniform(&[hidden_size, input_dim]);
            base.register_parameter(format!("weight_ih_l{}", layer), Parameter::new(w_ih));

            // Hidden-to-hidden weights
            let w_hh = crate::init::xavier_uniform(&[hidden_size, hidden_size]);
            base.register_parameter(format!("weight_hh_l{}", layer), Parameter::new(w_hh));

            if bias {
                let b_ih = zeros(&[hidden_size])?;
                base.register_parameter(format!("bias_ih_l{}", layer), Parameter::new(b_ih));

                let b_hh = zeros(&[hidden_size])?;
                base.register_parameter(format!("bias_hh_l{}", layer), Parameter::new(b_hh));
            }

            if bidirectional {
                // Reverse direction weights
                let w_ih_reverse = crate::init::xavier_uniform(&[hidden_size, input_dim]);
                base.register_parameter(
                    format!("weight_ih_l{}_reverse", layer),
                    Parameter::new(w_ih_reverse),
                );

                let w_hh_reverse = crate::init::xavier_uniform(&[hidden_size, hidden_size]);
                base.register_parameter(
                    format!("weight_hh_l{}_reverse", layer),
                    Parameter::new(w_hh_reverse),
                );

                if bias {
                    let b_ih_reverse = zeros(&[hidden_size])?;
                    base.register_parameter(
                        format!("bias_ih_l{}_reverse", layer),
                        Parameter::new(b_ih_reverse),
                    );

                    let b_hh_reverse = zeros(&[hidden_size])?;
                    base.register_parameter(
                        format!("bias_hh_l{}_reverse", layer),
                        Parameter::new(b_hh_reverse),
                    );
                }
            }
        }

        Self {
            base,
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        }
    }

    // Getter methods
    pub fn input_size(&self) -> usize {
        self.input_size
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    pub fn bias(&self) -> bool {
        self.bias
    }
    pub fn batch_first(&self) -> bool {
        self.batch_first
    }
    pub fn dropout(&self) -> f32 {
        self.dropout
    }
    pub fn bidirectional(&self) -> bool {
        self.bidirectional
    }
}

impl Module for RNN {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Implement basic RNN forward pass
        // Input: [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
        // Output: [seq_len, batch, hidden_size] or [batch, seq_len, hidden_size]

        let input_shape = input.shape().dims();
        let (seq_len, batch_size, _input_size) = if self.batch_first {
            (input_shape[1], input_shape[0], input_shape[2])
        } else {
            (input_shape[0], input_shape[1], input_shape[2])
        };

        // Initialize hidden state with zeros
        let mut hidden = vec![vec![0.0f32; self.hidden_size]; batch_size];

        // Get parameters for first layer (simplified - only use first layer)
        let weight_ih = self.base.parameters.get("weight_ih_l0")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing weight_ih_l0".to_string()))?
            .tensor().read();
        let weight_hh = self.base.parameters.get("weight_hh_l0")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing weight_hh_l0".to_string()))?
            .tensor().read();
        let bias_ih = self.base.parameters.get("bias_ih_l0")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing bias_ih_l0".to_string()))?
            .tensor().read();
        let bias_hh = self.base.parameters.get("bias_hh_l0")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing bias_hh_l0".to_string()))?
            .tensor().read();

        let w_ih_data = weight_ih.to_vec()?;
        let w_hh_data = weight_hh.to_vec()?;
        let b_ih_data = bias_ih.to_vec()?;
        let b_hh_data = bias_hh.to_vec()?;
        let input_data = input.to_vec()?;

        // Compute input size from input shape
        let input_size = if self.batch_first { input_shape[2] } else { input_shape[2] };

        // Storage for all timestep outputs
        let mut all_outputs = Vec::with_capacity(seq_len * batch_size * self.hidden_size);

        // Process each timestep
        for t in 0..seq_len {
            for b in 0..batch_size {
                // Get input at timestep t for batch b
                let input_idx_base = if self.batch_first {
                    b * seq_len * input_size + t * input_size
                } else {
                    t * batch_size * input_size + b * input_size
                };

                // Compute: h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
                for h in 0..self.hidden_size {
                    let mut sum = b_ih_data[h] + b_hh_data[h];

                    // W_ih @ x_t
                    for i in 0..input_size {
                        let x_val = input_data[input_idx_base + i];
                        let w_val = w_ih_data[h * input_size + i];
                        sum += w_val * x_val;
                    }

                    // W_hh @ h_{t-1}
                    for h_prev in 0..self.hidden_size {
                        let h_val = hidden[b][h_prev];
                        let w_val = w_hh_data[h * self.hidden_size + h_prev];
                        sum += w_val * h_val;
                    }

                    // Apply tanh activation
                    hidden[b][h] = sum.tanh();
                }
            }

            // Store outputs for this timestep
            for b in 0..batch_size {
                all_outputs.extend_from_slice(&hidden[b]);
            }
        }

        // Reshape output
        let output_shape = if self.batch_first {
            vec![batch_size, seq_len, self.hidden_size]
        } else {
            vec![seq_len, batch_size, self.hidden_size]
        };

        // Reorder if batch_first
        let final_output = if self.batch_first {
            // all_outputs is in [seq, batch, hidden] order, need [batch, seq, hidden]
            let mut reordered = Vec::with_capacity(all_outputs.len());
            for b in 0..batch_size {
                for t in 0..seq_len {
                    let src_idx = (t * batch_size + b) * self.hidden_size;
                    reordered.extend_from_slice(&all_outputs[src_idx..src_idx + self.hidden_size]);
                }
            }
            reordered
        } else {
            all_outputs
        };

        Tensor::from_vec(final_output, &output_shape)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base.all_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        self.base.all_named_parameters()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        for param in self.parameters() {
            let mut tensor = param.write();
            *tensor = tensor.clone().to(device)?;
        }
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!("input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, dropout={}, bidirectional={}", 
                self.input_size, self.hidden_size, self.num_layers, self.bias, self.batch_first, self.dropout, self.bidirectional)
    }
}

/// LSTM layer
pub struct LSTM {
    base: ModuleBase,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bias: bool,
    batch_first: bool,
    dropout: f32,
    bidirectional: bool,
}

impl LSTM {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        bias: bool,
        batch_first: bool,
        dropout: Option<f32>,
        bidirectional: bool,
    ) -> Self {
        let mut base = ModuleBase::new();
        let num_layers = num_layers.unwrap_or(1);
        let dropout = dropout.unwrap_or(0.0);

        // Initialize weights for each layer
        for layer in 0..num_layers {
            let input_dim = if layer == 0 { input_size } else { hidden_size };

            // Input-to-hidden weights (4 gates: i, f, g, o)
            let w_ih = crate::init::xavier_uniform(&[4 * hidden_size, input_dim]);
            base.register_parameter(format!("weight_ih_l{}", layer), Parameter::new(w_ih));

            // Hidden-to-hidden weights (4 gates: i, f, g, o)
            let w_hh = crate::init::xavier_uniform(&[4 * hidden_size, hidden_size]);
            base.register_parameter(format!("weight_hh_l{}", layer), Parameter::new(w_hh));

            if bias {
                let b_ih = zeros(&[4 * hidden_size])?;
                base.register_parameter(format!("bias_ih_l{}", layer), Parameter::new(b_ih));

                let b_hh = zeros(&[4 * hidden_size])?;
                base.register_parameter(format!("bias_hh_l{}", layer), Parameter::new(b_hh));
            }

            if bidirectional {
                // Reverse direction weights
                let w_ih_reverse = crate::init::xavier_uniform(&[4 * hidden_size, input_dim]);
                base.register_parameter(
                    format!("weight_ih_l{}_reverse", layer),
                    Parameter::new(w_ih_reverse),
                );

                let w_hh_reverse = crate::init::xavier_uniform(&[4 * hidden_size, hidden_size]);
                base.register_parameter(
                    format!("weight_hh_l{}_reverse", layer),
                    Parameter::new(w_hh_reverse),
                );

                if bias {
                    let b_ih_reverse = zeros(&[4 * hidden_size])?;
                    base.register_parameter(
                        format!("bias_ih_l{}_reverse", layer),
                        Parameter::new(b_ih_reverse),
                    );

                    let b_hh_reverse = zeros(&[4 * hidden_size])?;
                    base.register_parameter(
                        format!("bias_hh_l{}_reverse", layer),
                        Parameter::new(b_hh_reverse),
                    );
                }
            }
        }

        Self {
            base,
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        }
    }

    // Getter methods
    pub fn input_size(&self) -> usize {
        self.input_size
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    pub fn bias(&self) -> bool {
        self.bias
    }
    pub fn batch_first(&self) -> bool {
        self.batch_first
    }
    pub fn dropout(&self) -> f32 {
        self.dropout
    }
    pub fn bidirectional(&self) -> bool {
        self.bidirectional
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Implement LSTM forward pass
        // Input: [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
        // Output: [seq_len, batch, hidden_size] or [batch, seq_len, hidden_size]
        // LSTM: i_t = σ(W_ii x_t + b_ii + W_hi h_{t-1} + b_hi)
        //       f_t = σ(W_if x_t + b_if + W_hf h_{t-1} + b_hf)
        //       g_t = tanh(W_ig x_t + b_ig + W_hg h_{t-1} + b_hg)
        //       o_t = σ(W_io x_t + b_io + W_ho h_{t-1} + b_ho)
        //       c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        //       h_t = o_t ⊙ tanh(c_t)

        let input_shape = input.shape().dims();
        let (seq_len, batch_size, _input_size) = if self.batch_first {
            (input_shape[1], input_shape[0], input_shape[2])
        } else {
            (input_shape[0], input_shape[1], input_shape[2])
        };

        // Initialize hidden and cell states with zeros
        let mut hidden = vec![vec![0.0f32; self.hidden_size]; batch_size];
        let mut cell = vec![vec![0.0f32; self.hidden_size]; batch_size];

        // Get parameters for first layer
        let weight_ih = self.base.parameters.get("weight_ih_l0")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing weight_ih_l0".to_string()))?
            .tensor().read();
        let weight_hh = self.base.parameters.get("weight_hh_l0")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing weight_hh_l0".to_string()))?
            .tensor().read();
        let bias_ih = self.base.parameters.get("bias_ih_l0")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing bias_ih_l0".to_string()))?
            .tensor().read();
        let bias_hh = self.base.parameters.get("bias_hh_l0")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing bias_hh_l0".to_string()))?
            .tensor().read();

        let w_ih_data = weight_ih.to_vec()?;
        let w_hh_data = weight_hh.to_vec()?;
        let b_ih_data = bias_ih.to_vec()?;
        let b_hh_data = bias_hh.to_vec()?;
        let input_data = input.to_vec()?;

        let input_size = input_shape[if self.batch_first { 2 } else { 2 }];
        let mut all_outputs = Vec::with_capacity(seq_len * batch_size * self.hidden_size);

        // Process each timestep
        for t in 0..seq_len {
            for b in 0..batch_size {
                // Get input at timestep t for batch b
                let input_idx_base = if self.batch_first {
                    b * seq_len * input_size + t * input_size
                } else {
                    t * batch_size * input_size + b * input_size
                };

                // Compute gates: LSTM has 4 gates (input, forget, cell, output)
                // Weights are stacked: [i, f, g, o]
                for gate in 0..4 {
                    for h in 0..self.hidden_size {
                        let gate_h = gate * self.hidden_size + h;
                        let mut sum = b_ih_data[gate_h] + b_hh_data[gate_h];

                        // W_ih @ x_t
                        for i in 0..input_size {
                            let x_val = input_data[input_idx_base + i];
                            let w_val = w_ih_data[gate_h * input_size + i];
                            sum += w_val * x_val;
                        }

                        // W_hh @ h_{t-1}
                        for h_prev in 0..self.hidden_size {
                            let h_val = hidden[b][h_prev];
                            let w_val = w_hh_data[gate_h * self.hidden_size + h_prev];
                            sum += w_val * h_val;
                        }

                        // Apply activation and store in temporary gate values
                        let gate_val = match gate {
                            0 | 1 | 3 => 1.0 / (1.0 + (-sum).exp()), // sigmoid for i, f, o
                            2 => sum.tanh(), // tanh for g
                            _ => unreachable!(),
                        };

                        // Update cell and hidden based on gate
                        match gate {
                            0 => { /* input gate - will use in gate 2 */ },
                            1 => cell[b][h] *= gate_val, // forget gate
                            2 => {
                                // cell gate - combine with input gate
                                let i_val = {
                                    let ig_h = h;
                                    let mut i_sum = b_ih_data[ig_h] + b_hh_data[ig_h];
                                    for i in 0..input_size {
                                        i_sum += w_ih_data[ig_h * input_size + i] * input_data[input_idx_base + i];
                                    }
                                    for h_prev in 0..self.hidden_size {
                                        i_sum += w_hh_data[ig_h * self.hidden_size + h_prev] * hidden[b][h_prev];
                                    }
                                    1.0 / (1.0 + (-i_sum).exp())
                                };
                                cell[b][h] += i_val * gate_val;
                            },
                            3 => hidden[b][h] = gate_val * cell[b][h].tanh(), // output gate
                            _ => unreachable!(),
                        }
                    }
                }
            }

            // Store outputs for this timestep
            for b in 0..batch_size {
                all_outputs.extend_from_slice(&hidden[b]);
            }
        }

        // Reshape output
        let output_shape = if self.batch_first {
            vec![batch_size, seq_len, self.hidden_size]
        } else {
            vec![seq_len, batch_size, self.hidden_size]
        };

        // Reorder if batch_first
        let final_output = if self.batch_first {
            let mut reordered = Vec::with_capacity(all_outputs.len());
            for b in 0..batch_size {
                for t in 0..seq_len {
                    let src_idx = (t * batch_size + b) * self.hidden_size;
                    reordered.extend_from_slice(&all_outputs[src_idx..src_idx + self.hidden_size]);
                }
            }
            reordered
        } else {
            all_outputs
        };

        Tensor::from_vec(final_output, &output_shape)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base.all_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        self.base.all_named_parameters()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        for param in self.parameters() {
            let mut tensor = param.write();
            *tensor = tensor.clone().to(device)?;
        }
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!("input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, dropout={}, bidirectional={}", 
                self.input_size, self.hidden_size, self.num_layers, self.bias, self.batch_first, self.dropout, self.bidirectional)
    }
}

/// GRU layer
pub struct GRU {
    base: ModuleBase,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bias: bool,
    batch_first: bool,
    dropout: f32,
    bidirectional: bool,
}

impl GRU {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        bias: bool,
        batch_first: bool,
        dropout: Option<f32>,
        bidirectional: bool,
    ) -> Self {
        let mut base = ModuleBase::new();
        let num_layers = num_layers.unwrap_or(1);
        let dropout = dropout.unwrap_or(0.0);

        // Initialize weights for each layer
        for layer in 0..num_layers {
            let input_dim = if layer == 0 { input_size } else { hidden_size };

            // Input-to-hidden weights (3 gates: r, z, n)
            let w_ih = crate::init::xavier_uniform(&[3 * hidden_size, input_dim]);
            base.register_parameter(format!("weight_ih_l{}", layer), Parameter::new(w_ih));

            // Hidden-to-hidden weights (3 gates: r, z, n)
            let w_hh = crate::init::xavier_uniform(&[3 * hidden_size, hidden_size]);
            base.register_parameter(format!("weight_hh_l{}", layer), Parameter::new(w_hh));

            if bias {
                let b_ih = zeros(&[3 * hidden_size])?;
                base.register_parameter(format!("bias_ih_l{}", layer), Parameter::new(b_ih));

                let b_hh = zeros(&[3 * hidden_size])?;
                base.register_parameter(format!("bias_hh_l{}", layer), Parameter::new(b_hh));
            }

            if bidirectional {
                // Reverse direction weights
                let w_ih_reverse = crate::init::xavier_uniform(&[3 * hidden_size, input_dim]);
                base.register_parameter(
                    format!("weight_ih_l{}_reverse", layer),
                    Parameter::new(w_ih_reverse),
                );

                let w_hh_reverse = crate::init::xavier_uniform(&[3 * hidden_size, hidden_size]);
                base.register_parameter(
                    format!("weight_hh_l{}_reverse", layer),
                    Parameter::new(w_hh_reverse),
                );

                if bias {
                    let b_ih_reverse = zeros(&[3 * hidden_size])?;
                    base.register_parameter(
                        format!("bias_ih_l{}_reverse", layer),
                        Parameter::new(b_ih_reverse),
                    );

                    let b_hh_reverse = zeros(&[3 * hidden_size])?;
                    base.register_parameter(
                        format!("bias_hh_l{}_reverse", layer),
                        Parameter::new(b_hh_reverse),
                    );
                }
            }
        }

        Self {
            base,
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        }
    }

    // Getter methods
    pub fn input_size(&self) -> usize {
        self.input_size
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    pub fn bias(&self) -> bool {
        self.bias
    }
    pub fn batch_first(&self) -> bool {
        self.batch_first
    }
    pub fn dropout(&self) -> f32 {
        self.dropout
    }
    pub fn bidirectional(&self) -> bool {
        self.bidirectional
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Implement GRU forward pass
        // Input: [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
        // Output: [seq_len, batch, hidden_size] or [batch, seq_len, hidden_size]
        // GRU: r_t = σ(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)
        //      z_t = σ(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)
        //      n_t = tanh(W_in x_t + b_in + r_t ⊙ (W_hn h_{t-1} + b_hn))
        //      h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}

        let input_shape = input.shape().dims();
        let (seq_len, batch_size, _input_size) = if self.batch_first {
            (input_shape[1], input_shape[0], input_shape[2])
        } else {
            (input_shape[0], input_shape[1], input_shape[2])
        };

        // Initialize hidden state with zeros
        let mut hidden = vec![vec![0.0f32; self.hidden_size]; batch_size];

        // Get parameters for first layer
        let weight_ih = self.base.parameters.get("weight_ih_l0")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing weight_ih_l0".to_string()))?
            .tensor().read();
        let weight_hh = self.base.parameters.get("weight_hh_l0")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing weight_hh_l0".to_string()))?
            .tensor().read();
        let bias_ih = self.base.parameters.get("bias_ih_l0")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing bias_ih_l0".to_string()))?
            .tensor().read();
        let bias_hh = self.base.parameters.get("bias_hh_l0")
            .ok_or_else(|| torsh_core::error::TorshError::InvalidArgument("Missing bias_hh_l0".to_string()))?
            .tensor().read();

        let w_ih_data = weight_ih.to_vec()?;
        let w_hh_data = weight_hh.to_vec()?;
        let b_ih_data = bias_ih.to_vec()?;
        let b_hh_data = bias_hh.to_vec()?;
        let input_data = input.to_vec()?;

        let input_size = input_shape[if self.batch_first { 2 } else { 2 }];
        let mut all_outputs = Vec::with_capacity(seq_len * batch_size * self.hidden_size);

        // Process each timestep
        for t in 0..seq_len {
            for b in 0..batch_size {
                // Get input at timestep t for batch b
                let input_idx_base = if self.batch_first {
                    b * seq_len * input_size + t * input_size
                } else {
                    t * batch_size * input_size + b * input_size
                };

                // Temporary storage for gates
                let mut r_gate = vec![0.0f32; self.hidden_size];
                let mut z_gate = vec![0.0f32; self.hidden_size];
                let mut n_gate = vec![0.0f32; self.hidden_size];

                // Compute reset gate (r), update gate (z), and new gate (n)
                // Weights are stacked: [r, z, n]
                for gate in 0..3 {
                    for h in 0..self.hidden_size {
                        let gate_h = gate * self.hidden_size + h;
                        let mut sum = b_ih_data[gate_h] + b_hh_data[gate_h];

                        // W_ih @ x_t
                        for i in 0..input_size {
                            let x_val = input_data[input_idx_base + i];
                            let w_val = w_ih_data[gate_h * input_size + i];
                            sum += w_val * x_val;
                        }

                        // W_hh @ h_{t-1} (for r and z gates)
                        // For n gate, we apply reset gate first
                        if gate < 2 {
                            for h_prev in 0..self.hidden_size {
                                let h_val = hidden[b][h_prev];
                                let w_val = w_hh_data[gate_h * self.hidden_size + h_prev];
                                sum += w_val * h_val;
                            }
                        }

                        // Store gate value
                        match gate {
                            0 => r_gate[h] = 1.0 / (1.0 + (-sum).exp()), // sigmoid for reset
                            1 => z_gate[h] = 1.0 / (1.0 + (-sum).exp()), // sigmoid for update
                            2 => {
                                // For new gate, apply reset to hidden state
                                let mut n_sum = sum;
                                for h_prev in 0..self.hidden_size {
                                    let h_val = hidden[b][h_prev];
                                    let w_val = w_hh_data[gate_h * self.hidden_size + h_prev];
                                    n_sum += r_gate[h_prev] * w_val * h_val;
                                }
                                n_gate[h] = n_sum.tanh(); // tanh for new
                            },
                            _ => unreachable!(),
                        }
                    }
                }

                // Update hidden state: h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
                for h in 0..self.hidden_size {
                    hidden[b][h] = (1.0 - z_gate[h]) * n_gate[h] + z_gate[h] * hidden[b][h];
                }
            }

            // Store outputs for this timestep
            for b in 0..batch_size {
                all_outputs.extend_from_slice(&hidden[b]);
            }
        }

        // Reshape output
        let output_shape = if self.batch_first {
            vec![batch_size, seq_len, self.hidden_size]
        } else {
            vec![seq_len, batch_size, self.hidden_size]
        };

        // Reorder if batch_first
        let final_output = if self.batch_first {
            let mut reordered = Vec::with_capacity(all_outputs.len());
            for b in 0..batch_size {
                for t in 0..seq_len {
                    let src_idx = (t * batch_size + b) * self.hidden_size;
                    reordered.extend_from_slice(&all_outputs[src_idx..src_idx + self.hidden_size]);
                }
            }
            reordered
        } else {
            all_outputs
        };

        Tensor::from_vec(final_output, &output_shape)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base.all_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        self.base.all_named_parameters()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        for param in self.parameters() {
            let mut tensor = param.write();
            *tensor = tensor.clone().to(device)?;
        }
        Ok(())
    }

    fn extra_repr(&self) -> String {
        format!("input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, dropout={}, bidirectional={}", 
                self.input_size, self.hidden_size, self.num_layers, self.bias, self.batch_first, self.dropout, self.bidirectional)
    }
}
