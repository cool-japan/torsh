//! Embedding layers

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

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

        let weight = self.base.parameters["weight"].tensor().read().clone();
        let weight_data = weight.to_vec()?;

        // Get input indices
        let input_data = input.to_vec()?;
        let binding = input.shape();
        let input_shape = binding.dims();

        // Calculate output shape
        let mut output_shape = input_shape.to_vec();
        output_shape.push(self.embedding_dim);

        // Calculate total number of lookups
        let num_indices: usize = input_shape.iter().product();
        let total_output_size = num_indices * self.embedding_dim;

        let mut output_data = Vec::with_capacity(total_output_size);

        // Perform embedding lookup for each index
        for &idx_f32 in input_data.iter() {
            // Convert f32 index to usize
            let idx = idx_f32 as usize;

            // Handle padding_idx if set
            if let Some(padding_idx) = self.padding_idx {
                if idx == padding_idx {
                    // Return zeros for padding index
                    output_data.extend(vec![0.0; self.embedding_dim]);
                    continue;
                }
            }

            // Bounds check
            if idx >= self.num_embeddings {
                return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                    "Index {} out of bounds for embedding with {} embeddings",
                    idx, self.num_embeddings
                )));
            }

            // Lookup embedding vector for this index
            let start_idx = idx * self.embedding_dim;
            let end_idx = start_idx + self.embedding_dim;

            // Get the embedding vector
            let mut embedding_vec = weight_data[start_idx..end_idx].to_vec();

            // Apply max_norm if specified
            if let Some(max_norm) = self.max_norm {
                let norm = if self.norm_type == 2.0 {
                    // L2 norm
                    embedding_vec.iter().map(|x| x * x).sum::<f32>().sqrt()
                } else {
                    // L_p norm
                    embedding_vec
                        .iter()
                        .map(|x| x.abs().powf(self.norm_type))
                        .sum::<f32>()
                        .powf(1.0 / self.norm_type)
                };

                if norm > max_norm {
                    // Renormalize to max_norm
                    let scale = max_norm / norm;
                    for val in &mut embedding_vec {
                        *val *= scale;
                    }
                }
            }

            output_data.extend(embedding_vec);
        }

        Tensor::from_vec(output_data, &output_shape)
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
        // Implement relative positional encoding as used in Transformer-XL and similar models
        // The input is typically attention scores or query/key representations
        // Shape: [batch_size, seq_len, d_model] or [batch_size, num_heads, seq_len, seq_len]

        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();

        // Determine if input is attention scores (4D) or representations (3D)
        match input_shape.len() {
            3 => {
                // Input is [batch_size, seq_len, d_model]
                // Add relative positional bias
                let batch_size = input_shape[0];
                let seq_len = input_shape[1];
                let d_model = input_shape[2];

                // Get positional embeddings for relative positions
                // Relative positions range from -(seq_len-1) to +(seq_len-1)
                let max_relative_position = (seq_len - 1) * 2 + 1;

                // Get embedding weights
                let embeddings = self.base.parameters.get("weight").ok_or_else(|| {
                    TorshError::InvalidArgument(
                        "RelativePositionalEncoding missing weight parameter".to_string(),
                    )
                })?;

                let embedding_data = embeddings.tensor().read().to_vec()?;
                let input_data = input.to_vec()?;

                let mut output_data = vec![0.0f32; batch_size * seq_len * d_model];

                // For each position, add relative positional embeddings
                for b in 0..batch_size {
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            // Calculate relative position: j - i
                            let relative_pos = (j as i32 - i as i32) + (seq_len - 1) as i32;
                            let relative_pos_clamped =
                                relative_pos.max(0).min(max_relative_position as i32 - 1) as usize;

                            // Get positional embedding for this relative position
                            let emb_start = relative_pos_clamped * d_model;

                            // Add to output (only once per position i, average over j)
                            for d in 0..d_model {
                                let input_idx = b * seq_len * d_model + i * d_model + d;
                                let output_idx = b * seq_len * d_model + i * d_model + d;

                                if j == 0 {
                                    // Initialize with input value
                                    output_data[output_idx] = input_data[input_idx];
                                }

                                // Add fractional contribution from relative position embedding
                                output_data[output_idx] +=
                                    embedding_data[emb_start + d] / seq_len as f32;
                            }
                        }
                    }
                }

                Tensor::from_vec(output_data, input_shape)
            }
            4 => {
                // Input is attention scores [batch_size, num_heads, seq_len, seq_len]
                // Add relative position bias to attention scores
                let batch_size = input_shape[0];
                let num_heads = input_shape[1];
                let seq_len_q = input_shape[2];
                let seq_len_k = input_shape[3];

                // For simplicity, assume seq_len_q == seq_len_k
                if seq_len_q != seq_len_k {
                    return Err(TorshError::InvalidArgument(
                        "RelativePositionalEncoding requires square attention matrices".to_string(),
                    ));
                }

                let seq_len = seq_len_q;
                let max_relative_position = (seq_len - 1) * 2 + 1;

                // Get embedding weights
                let embeddings = self.base.parameters.get("weight").ok_or_else(|| {
                    TorshError::InvalidArgument(
                        "RelativePositionalEncoding missing weight parameter".to_string(),
                    )
                })?;

                let embedding_data = embeddings.tensor().read().to_vec()?;
                let input_data = input.to_vec()?;

                let mut output_data = vec![0.0f32; batch_size * num_heads * seq_len * seq_len];

                // Add relative position bias to each attention score
                for b in 0..batch_size {
                    for h in 0..num_heads {
                        for i in 0..seq_len {
                            for j in 0..seq_len {
                                // Calculate relative position: j - i
                                let relative_pos = (j as i32 - i as i32) + (seq_len - 1) as i32;
                                let relative_pos_clamped =
                                    relative_pos.max(0).min(max_relative_position as i32 - 1)
                                        as usize;

                                // Input and output index
                                let idx = b * num_heads * seq_len * seq_len
                                    + h * seq_len * seq_len
                                    + i * seq_len
                                    + j;

                                // Add relative position bias (use first dimension of embedding as bias)
                                output_data[idx] =
                                    input_data[idx] + embedding_data[relative_pos_clamped];
                            }
                        }
                    }
                }

                Tensor::from_vec(output_data, input_shape)
            }
            _ => {
                // Unsupported shape, return input unchanged
                Ok(input.clone())
            }
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

    Tensor::from_vec(pos_encoding, &[max_len, d_model]).expect("tensor creation should succeed")
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

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_embedding_basic_lookup() -> Result<()> {
        // Create a small embedding: 5 tokens, 3-dimensional embeddings
        let mut embedding = Embedding::new(5, 3);

        // Set deterministic weights for testing
        let weight_data = vec![
            1.0, 2.0, 3.0, // Token 0
            4.0, 5.0, 6.0, // Token 1
            7.0, 8.0, 9.0, // Token 2
            10.0, 11.0, 12.0, // Token 3
            13.0, 14.0, 15.0, // Token 4
        ];
        let weight = Tensor::from_vec(weight_data, &[5, 3])?;
        *embedding
            .base
            .parameters
            .get_mut("weight")
            .unwrap()
            .tensor()
            .write() = weight;

        // Test single index lookup
        let input = Tensor::from_vec(vec![2.0], &[1])?;
        let output = embedding.forward(&input)?;

        let output_data = output.to_vec()?;
        assert_eq!(output.shape().dims(), &[1, 3]);
        assert_relative_eq!(output_data[0], 7.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[1], 8.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[2], 9.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_embedding_multiple_indices() -> Result<()> {
        // Create embedding: 4 tokens, 2-dimensional
        let mut embedding = Embedding::new(4, 2);

        let weight_data = vec![
            1.0, 2.0, // Token 0
            3.0, 4.0, // Token 1
            5.0, 6.0, // Token 2
            7.0, 8.0, // Token 3
        ];
        let weight = Tensor::from_vec(weight_data, &[4, 2])?;
        *embedding
            .base
            .parameters
            .get_mut("weight")
            .unwrap()
            .tensor()
            .write() = weight;

        // Lookup multiple indices: [0, 2, 1]
        let input = Tensor::from_vec(vec![0.0, 2.0, 1.0], &[3])?;
        let output = embedding.forward(&input)?;

        let output_data = output.to_vec()?;
        assert_eq!(output.shape().dims(), &[3, 2]);

        // Token 0: [1.0, 2.0]
        assert_relative_eq!(output_data[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[1], 2.0, epsilon = 1e-6);

        // Token 2: [5.0, 6.0]
        assert_relative_eq!(output_data[2], 5.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[3], 6.0, epsilon = 1e-6);

        // Token 1: [3.0, 4.0]
        assert_relative_eq!(output_data[4], 3.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[5], 4.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_embedding_2d_indices() -> Result<()> {
        // Test with 2D input (batch of sequences)
        let mut embedding = Embedding::new(3, 2);

        let weight_data = vec![
            1.0, 2.0, // Token 0
            3.0, 4.0, // Token 1
            5.0, 6.0, // Token 2
        ];
        let weight = Tensor::from_vec(weight_data, &[3, 2])?;
        *embedding
            .base
            .parameters
            .get_mut("weight")
            .unwrap()
            .tensor()
            .write() = weight;

        // Input: 2 sequences of length 2
        // [[0, 1],
        //  [2, 0]]
        let input = Tensor::from_vec(vec![0.0, 1.0, 2.0, 0.0], &[2, 2])?;
        let output = embedding.forward(&input)?;

        assert_eq!(output.shape().dims(), &[2, 2, 2]); // [batch, seq_len, embedding_dim]

        let output_data = output.to_vec()?;

        // Sequence 0, Token 0 (index 0): [1.0, 2.0]
        assert_relative_eq!(output_data[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[1], 2.0, epsilon = 1e-6);

        // Sequence 0, Token 1 (index 1): [3.0, 4.0]
        assert_relative_eq!(output_data[2], 3.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[3], 4.0, epsilon = 1e-6);

        // Sequence 1, Token 0 (index 2): [5.0, 6.0]
        assert_relative_eq!(output_data[4], 5.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[5], 6.0, epsilon = 1e-6);

        // Sequence 1, Token 1 (index 0): [1.0, 2.0]
        assert_relative_eq!(output_data[6], 1.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[7], 2.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_embedding_with_padding_idx() -> Result<()> {
        // Test padding index functionality
        let mut embedding = Embedding::with_padding_idx(4, 3, 0);

        let weight_data = vec![
            1.0, 2.0, 3.0, // Token 0 (padding - should be ignored)
            4.0, 5.0, 6.0, // Token 1
            7.0, 8.0, 9.0, // Token 2
            10.0, 11.0, 12.0, // Token 3
        ];
        let weight = Tensor::from_vec(weight_data, &[4, 3])?;
        *embedding
            .base
            .parameters
            .get_mut("weight")
            .unwrap()
            .tensor()
            .write() = weight;

        // Lookup including padding index 0
        let input = Tensor::from_vec(vec![1.0, 0.0, 2.0], &[3])?;
        let output = embedding.forward(&input)?;

        let output_data = output.to_vec()?;

        // Token 1: [4.0, 5.0, 6.0]
        assert_relative_eq!(output_data[0], 4.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[1], 5.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[2], 6.0, epsilon = 1e-6);

        // Token 0 (padding): [0.0, 0.0, 0.0]
        assert_relative_eq!(output_data[3], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[4], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[5], 0.0, epsilon = 1e-6);

        // Token 2: [7.0, 8.0, 9.0]
        assert_relative_eq!(output_data[6], 7.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[7], 8.0, epsilon = 1e-6);
        assert_relative_eq!(output_data[8], 9.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_embedding_out_of_bounds() {
        // Test that out-of-bounds index is rejected
        let mut embedding = Embedding::new(3, 2);

        let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = Tensor::from_vec(weight_data, &[3, 2]).unwrap();
        *embedding
            .base
            .parameters
            .get_mut("weight")
            .unwrap()
            .tensor()
            .write() = weight;

        // Try to lookup index 5 (out of bounds for num_embeddings=3)
        let input = Tensor::from_vec(vec![5.0], &[1]).unwrap();
        let result = embedding.forward(&input);

        assert!(result.is_err());
        if let Err(torsh_core::error::TorshError::InvalidArgument(msg)) = result {
            assert!(msg.contains("out of bounds"));
        } else {
            panic!("Expected InvalidArgument error for out-of-bounds index");
        }
    }

    #[test]
    fn test_embedding_with_max_norm() -> Result<()> {
        // Test max_norm renormalization
        let mut embedding = Embedding::with_config(
            3,
            2,
            None,      // padding_idx
            Some(1.0), // max_norm
            2.0,       // norm_type (L2)
            false,     // scale_grad_by_freq
            false,     // sparse
        );

        // Create embeddings with large norms
        let weight_data = vec![
            3.0, 4.0, // Token 0: L2 norm = 5.0 (will be scaled down)
            1.0, 0.0, // Token 1: L2 norm = 1.0 (already within max_norm)
            0.6, 0.8, // Token 2: L2 norm = 1.0 (already within max_norm)
        ];
        let weight = Tensor::from_vec(weight_data, &[3, 2])?;
        *embedding
            .base
            .parameters
            .get_mut("weight")
            .unwrap()
            .tensor()
            .write() = weight;

        // Lookup token 0 (should be renormalized)
        let input = Tensor::from_vec(vec![0.0], &[1])?;
        let output = embedding.forward(&input)?;

        let output_data = output.to_vec()?;

        // Original: [3.0, 4.0], norm = 5.0
        // After renormalization to max_norm=1.0: [0.6, 0.8]
        assert_relative_eq!(output_data[0], 0.6, epsilon = 1e-6);
        assert_relative_eq!(output_data[1], 0.8, epsilon = 1e-6);

        // Verify the renormalized vector has norm <= max_norm
        let norm = (output_data[0] * output_data[0] + output_data[1] * output_data[1]).sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_embedding_shape_preservation() -> Result<()> {
        // Test that output shape is correct for various input shapes
        let embedding = Embedding::new(10, 5);

        // 1D input: [3] -> [3, 5]
        let input1d = Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3])?;
        let output1d = embedding.forward(&input1d)?;
        assert_eq!(output1d.shape().dims(), &[3, 5]);

        // 2D input: [2, 4] -> [2, 4, 5]
        let input2d = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 4])?;
        let output2d = embedding.forward(&input2d)?;
        assert_eq!(output2d.shape().dims(), &[2, 4, 5]);

        // 3D input: [2, 3, 2] -> [2, 3, 2, 5]
        let input3d = Tensor::from_vec(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 1.0],
            &[2, 3, 2],
        )?;
        let output3d = embedding.forward(&input3d)?;
        assert_eq!(output3d.shape().dims(), &[2, 3, 2, 5]);

        Ok(())
    }
}
