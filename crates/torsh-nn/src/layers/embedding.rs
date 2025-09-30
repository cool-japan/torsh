//! Embedding layers

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::Result;
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
        let weight = crate::init::xavier_uniform(&[num_embeddings, embedding_dim])
            .expect("Failed to initialize embedding weight");
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

    pub fn with_padding_idx(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: usize,
    ) -> Self {
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

        let _weight = self.base.parameters["weight"].tensor().read().clone();

        // This is a simplified implementation
        // Real implementation would perform proper embedding lookup
        let binding = input.shape();
        let input_shape = binding.dims();
        let mut output_shape = input_shape.to_vec();
        output_shape.push(self.embedding_dim);

        // Placeholder - actual implementation would index into weight matrix
        let output = zeros(&output_shape)?;

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

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
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

/// Positional Encoding Utilities
///
/// Comprehensive collection of positional encoding methods used in modern transformer architectures

/// Types of positional encoding supported
#[derive(Debug, Clone)]
pub enum PositionalEncodingType {
    /// Sinusoidal positional encoding (original Transformer)
    Sinusoidal,
    /// Learnable/trainable positional embeddings
    Learnable,
    /// Relative positional encoding
    Relative,
    /// Rotary Positional Embedding (RoPE)
    Rotary { base: f32 },
    /// ALiBi (Attention with Linear Biases)
    Alibi,
}

/// Sinusoidal Positional Encoding
///
/// The original positional encoding from "Attention Is All You Need"
/// Uses sine and cosine functions of different frequencies
pub struct SinusoidalPositionalEncoding {
    base: ModuleBase,
    d_model: usize,
    max_len: usize,
    dropout: f32,
}

impl SinusoidalPositionalEncoding {
    pub fn new(d_model: usize, max_len: usize, dropout: f32) -> Self {
        let mut base = ModuleBase::new();

        // Create fixed sinusoidal positional encoding
        let pe = create_sinusoidal_encoding(max_len, d_model);
        base.register_parameter("pe".to_string(), Parameter::new(pe));

        Self {
            base,
            d_model,
            max_len,
            dropout,
        }
    }
}

impl Module for SinusoidalPositionalEncoding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let pe = self.base.parameters["pe"].tensor().read().clone();
        let seq_len = input.shape().dims()[1]; // Assuming [batch, seq, dim]

        // Slice PE to match sequence length
        let pe_slice = pe.narrow(0, 0, seq_len.min(self.max_len))?;

        // Add positional encoding
        let output = input.add_op(&pe_slice.unsqueeze(0)?)?;

        // Apply dropout if specified
        if self.dropout > 0.0 && self.training() {
            crate::functional::dropout(&output, self.dropout, self.training())
        } else {
            Ok(output)
        }
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

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Learnable Positional Encoding
///
/// Trainable positional embeddings that are learned during training
pub struct LearnablePositionalEncoding {
    base: ModuleBase,
    d_model: usize,
    max_len: usize,
    dropout: f32,
}

impl LearnablePositionalEncoding {
    pub fn new(d_model: usize, max_len: usize, dropout: f32) -> Self {
        let mut base = ModuleBase::new();

        // Create learnable positional embeddings
        let pe = crate::init::xavier_uniform(&[max_len, d_model])
            .expect("Failed to initialize positional encoding");
        base.register_parameter("pe".to_string(), Parameter::new(pe));

        Self {
            base,
            d_model,
            max_len,
            dropout,
        }
    }
}

impl Module for LearnablePositionalEncoding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let pe = self.base.parameters["pe"].tensor().read().clone();
        let seq_len = input.shape().dims()[1]; // Assuming [batch, seq, dim]

        // Slice PE to match sequence length
        let pe_slice = pe.narrow(0, 0, seq_len.min(self.max_len))?;

        // Add positional encoding
        let output = input.add_op(&pe_slice.unsqueeze(0)?)?;

        // Apply dropout if specified
        if self.dropout > 0.0 && self.training() {
            crate::functional::dropout(&output, self.dropout, self.training())
        } else {
            Ok(output)
        }
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

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Rotary Positional Embedding (RoPE)
///
/// Implements Rotary Position Embedding from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
/// Applies rotation matrices to queries and keys based on their positions
pub struct RotaryPositionalEmbedding {
    d_model: usize,
    base: f32,
    max_seq_len: usize,
}

impl RotaryPositionalEmbedding {
    pub fn new(d_model: usize, base: f32, max_seq_len: usize) -> Self {
        Self {
            d_model,
            base,
            max_seq_len,
        }
    }

    /// Apply rotary position embedding to query and key tensors
    ///
    /// Args:
    /// - q: Query tensor [batch, heads, seq_len, head_dim]
    /// - k: Key tensor [batch, heads, seq_len, head_dim]
    ///
    /// Returns: (rotated_q, rotated_k)
    pub fn apply_rope(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = q.shape().dims()[2];
        let head_dim = q.shape().dims()[3];

        // Create frequency matrix
        let freqs = self.create_frequencies(seq_len, head_dim)?;

        // Apply rotation to q and k
        let q_rot = self.rotate_tensor(q, &freqs)?;
        let k_rot = self.rotate_tensor(k, &freqs)?;

        Ok((q_rot, k_rot))
    }

    fn create_frequencies(&self, seq_len: usize, head_dim: usize) -> Result<Tensor> {
        let mut freqs = Vec::new();

        for pos in 0..seq_len {
            for i in (0..head_dim).step_by(2) {
                let freq = 1.0 / self.base.powf(i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;

                freqs.push(angle.cos());
                freqs.push(-angle.sin());
                freqs.push(angle.sin());
                freqs.push(angle.cos());
            }
        }

        Tensor::from_vec(freqs, &[seq_len, head_dim, 2, 2])
    }

    fn rotate_tensor(&self, x: &Tensor, _freqs: &Tensor) -> Result<Tensor> {
        // This is a simplified implementation
        // Real RoPE requires complex tensor operations for rotation
        // For now, return the input tensor as-is
        Ok(x.clone())
    }
}

/// ALiBi (Attention with Linear Biases) Positional Encoding
///
/// Implements ALiBi from "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
/// Uses linear biases instead of positional encodings
pub struct AlibiPositionalBias {
    num_heads: usize,
    max_seq_len: usize,
}

impl AlibiPositionalBias {
    pub fn new(num_heads: usize, max_seq_len: usize) -> Self {
        Self {
            num_heads,
            max_seq_len,
        }
    }

    /// Create ALiBi bias matrix for attention scores
    ///
    /// Args:
    /// - seq_len: Current sequence length
    ///
    /// Returns: Bias tensor [num_heads, seq_len, seq_len]
    pub fn create_bias(&self, seq_len: usize) -> Result<Tensor> {
        let mut bias_data = Vec::new();

        // Create slopes for each head (geometric progression)
        let slopes = self.get_slopes();

        for head in 0..self.num_heads {
            let slope = slopes[head];

            for i in 0..seq_len {
                for j in 0..seq_len {
                    // ALiBi bias: -slope * |i - j|
                    let distance = (i as i32 - j as i32).abs() as f32;
                    let bias = -slope * distance;
                    bias_data.push(bias);
                }
            }
        }

        Tensor::from_vec(bias_data, &[self.num_heads, seq_len, seq_len])
    }

    fn get_slopes(&self) -> Vec<f32> {
        let ratio = 2.0_f32.powf(-8.0 / self.num_heads as f32);
        let mut slopes = Vec::new();

        for i in 0..self.num_heads {
            let slope = ratio.powf(i as f32 + 1.0);
            slopes.push(slope);
        }

        slopes
    }
}

/// Relative Positional Encoding
///
/// Implements relative position representations that focus on relative distances
/// between tokens rather than absolute positions
pub struct RelativePositionalEncoding {
    base: ModuleBase,
    d_model: usize,
    max_relative_distance: usize,
}

impl RelativePositionalEncoding {
    pub fn new(d_model: usize, max_relative_distance: usize) -> Self {
        let mut base = ModuleBase::new();

        // Create relative position embeddings
        // We need embeddings for positions from -max_relative_distance to +max_relative_distance
        let num_positions = 2 * max_relative_distance + 1;
        let relative_pe = crate::init::xavier_uniform(&[num_positions, d_model])
            .expect("Failed to initialize relative positional encoding");
        base.register_parameter("relative_pe".to_string(), Parameter::new(relative_pe));

        Self {
            base,
            d_model,
            max_relative_distance,
        }
    }

    /// Get relative position embeddings for a given sequence length
    ///
    /// Args:
    /// - seq_len: Length of the sequence
    ///
    /// Returns: Relative position matrix [seq_len, seq_len, d_model]
    pub fn get_relative_embeddings(&self, seq_len: usize) -> Result<Tensor> {
        let relative_pe = self.base.parameters["relative_pe"].tensor().read().clone();
        let mut relative_data = Vec::new();

        for i in 0..seq_len {
            for j in 0..seq_len {
                // Calculate relative distance, clamped to max_relative_distance
                let relative_distance = (i as i32 - j as i32).clamp(
                    -(self.max_relative_distance as i32),
                    self.max_relative_distance as i32,
                );

                // Convert to index (add offset for negative distances)
                let idx = (relative_distance + self.max_relative_distance as i32) as usize;

                // Get the embedding for this relative distance
                let embedding = relative_pe.narrow(0, idx as i64, 1)?.squeeze(0)?;
                let embedding_data = embedding.to_vec()?;
                relative_data.extend(embedding_data);
            }
        }

        Tensor::from_vec(relative_data, &[seq_len, seq_len, self.d_model])
    }
}

impl Module for RelativePositionalEncoding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For relative encoding, the input is typically the attention scores
        // This is a placeholder implementation
        Ok(input.clone())
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

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Position Interpolation utilities for extending sequence lengths
pub struct PositionInterpolation;

impl PositionInterpolation {
    /// Interpolate positional encodings to support longer sequences
    ///
    /// Args:
    /// - pe: Original positional encoding [max_len, d_model]
    /// - new_max_len: New maximum sequence length
    ///
    /// Returns: Interpolated positional encoding [new_max_len, d_model]
    pub fn interpolate_positions(pe: &Tensor, new_max_len: usize) -> Result<Tensor> {
        let shape = pe.shape();
        let pe_shape = shape.dims();
        let old_max_len = pe_shape[0];
        let d_model = pe_shape[1];

        if new_max_len <= old_max_len {
            // Just slice if new length is smaller
            return pe.narrow(0, 0, new_max_len);
        }

        // Simple linear interpolation (in practice, this would be more sophisticated)
        let scale_factor = old_max_len as f32 / new_max_len as f32;
        let mut interpolated_data = Vec::new();

        for new_pos in 0..new_max_len {
            let old_pos_f = new_pos as f32 * scale_factor;
            let old_pos_low = old_pos_f.floor() as usize;
            let old_pos_high = (old_pos_low + 1).min(old_max_len - 1);
            let alpha = old_pos_f - old_pos_low as f32;

            // Get embeddings at adjacent positions
            let pe_low = pe.narrow(0, old_pos_low as i64, 1)?.squeeze(0)?;
            let pe_high = pe.narrow(0, old_pos_high as i64, 1)?.squeeze(0)?;

            // Linear interpolation
            let pe_low_data = pe_low.to_vec()?;
            let pe_high_data = pe_high.to_vec()?;

            for i in 0..d_model {
                let interpolated = pe_low_data[i] * (1.0 - alpha) + pe_high_data[i] * alpha;
                interpolated_data.push(interpolated);
            }
        }

        Tensor::from_vec(interpolated_data, &[new_max_len, d_model])
    }

    /// Create frequency-based position interpolation for RoPE
    pub fn interpolate_rope_frequencies(base: f32, scale_factor: f32, d_model: usize) -> Vec<f32> {
        let mut freqs = Vec::new();

        for i in (0..d_model).step_by(2) {
            let freq = 1.0 / (base * scale_factor).powf(i as f32 / d_model as f32);
            freqs.push(freq);
        }

        freqs
    }
}

// Helper function to create sinusoidal positional encoding
fn create_sinusoidal_encoding(max_len: usize, d_model: usize) -> Tensor {
    let mut pos_encoding = vec![0.0f32; max_len * d_model];

    for pos in 0..max_len {
        for i in (0..d_model).step_by(2) {
            let angle = pos as f32 / 10000.0_f32.powf(i as f32 / d_model as f32);

            pos_encoding[pos * d_model + i] = angle.sin();
            if i + 1 < d_model {
                pos_encoding[pos * d_model + i + 1] = angle.cos();
            }
        }
    }

    Tensor::from_vec(pos_encoding, &[max_len, d_model]).unwrap()
}

impl std::fmt::Debug for SinusoidalPositionalEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SinusoidalPositionalEncoding")
            .field("d_model", &self.d_model)
            .field("max_len", &self.max_len)
            .field("dropout", &self.dropout)
            .finish()
    }
}

impl std::fmt::Debug for LearnablePositionalEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LearnablePositionalEncoding")
            .field("d_model", &self.d_model)
            .field("max_len", &self.max_len)
            .field("dropout", &self.dropout)
            .finish()
    }
}

impl std::fmt::Debug for RotaryPositionalEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RotaryPositionalEmbedding")
            .field("d_model", &self.d_model)
            .field("base", &self.base)
            .field("max_seq_len", &self.max_seq_len)
            .finish()
    }
}

impl std::fmt::Debug for AlibiPositionalBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlibiPositionalBias")
            .field("num_heads", &self.num_heads)
            .field("max_seq_len", &self.max_seq_len)
            .finish()
    }
}

impl std::fmt::Debug for RelativePositionalEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RelativePositionalEncoding")
            .field("d_model", &self.d_model)
            .field("max_relative_distance", &self.max_relative_distance)
            .finish()
    }
}
