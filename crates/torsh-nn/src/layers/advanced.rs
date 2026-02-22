//! Advanced neural network layers using SciRS2 algorithms

use crate::{Module, Parameter};
use torsh_core::error::TorshError;
use torsh_tensor::{
    creation::{ones, randn, zeros},
    Tensor,
};

/// Multi-Head Attention layer
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    pub d_v: usize,

    // Weight matrices
    pub w_q: Parameter,
    pub w_k: Parameter,
    pub w_v: Parameter,
    pub w_o: Parameter,

    // Optional bias terms
    pub bias_q: Option<Parameter>,
    pub bias_k: Option<Parameter>,
    pub bias_v: Option<Parameter>,
    pub bias_o: Option<Parameter>,

    pub dropout: f64,
    pub scale: f64,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    pub fn new(
        d_model: usize,
        num_heads: usize,
        dropout: f64,
        bias: bool,
    ) -> Result<Self, TorshError> {
        let d_k = d_model / num_heads;
        let d_v = d_model / num_heads;

        let scale = 1.0 / (d_k as f64).sqrt();

        // Initialize weight matrices with Xavier/Glorot initialization
        let fan_in = d_model as f64;
        let fan_out = d_model as f64;
        let std = (2.0 / (fan_in + fan_out)).sqrt();

        let w_q = Parameter::new(randn(&[d_model, d_model])?.mul_scalar(std as f32)?);
        let w_k = Parameter::new(randn(&[d_model, d_model])?.mul_scalar(std as f32)?);
        let w_v = Parameter::new(randn(&[d_model, d_model])?.mul_scalar(std as f32)?);
        let w_o = Parameter::new(randn(&[d_model, d_model])?.mul_scalar(std as f32)?);

        let bias_q = if bias {
            Some(Parameter::new(zeros(&[d_model])?))
        } else {
            None
        };
        let bias_k = if bias {
            Some(Parameter::new(zeros(&[d_model])?))
        } else {
            None
        };
        let bias_v = if bias {
            Some(Parameter::new(zeros(&[d_model])?))
        } else {
            None
        };
        let bias_o = if bias {
            Some(Parameter::new(zeros(&[d_model])?))
        } else {
            None
        };

        Ok(Self {
            num_heads,
            d_model,
            d_k,
            d_v,
            w_q,
            w_k,
            w_v,
            w_o,
            bias_q,
            bias_k,
            bias_v,
            bias_o,
            dropout,
            scale,
        })
    }

    /// Apply attention mechanism
    ///
    /// Implements scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch_size, num_heads, seq_len, d_k]
    /// * `k` - Key tensor [batch_size, num_heads, seq_len, d_k]
    /// * `v` - Value tensor [batch_size, num_heads, seq_len, d_v]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Attention output [batch_size, num_heads, seq_len, d_v]
    pub fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, TorshError> {
        // Input shape: [batch_size, num_heads, seq_len, d_k]
        let q_shape_binding = q.shape();
        let q_shape = q_shape_binding.dims();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_len_q = q_shape[2];
        let d_k = q_shape[3];

        let k_shape_binding = k.shape();
        let k_shape = k_shape_binding.dims();
        let seq_len_k = k_shape[2];

        // Step 1: Compute attention scores: Q @ K^T
        // We need to transpose the last two dimensions of K
        // K: [batch, heads, seq_k, d_k] -> K^T: [batch, heads, d_k, seq_k]

        // Flatten batch and heads dimensions for easier processing
        // q: [batch*heads, seq_q, d_k]
        // k: [batch*heads, seq_k, d_k]
        // v: [batch*heads, seq_k, d_v]
        let batch_heads = batch_size * num_heads;

        let q_flat = q.view(&[batch_heads as i32, seq_len_q as i32, d_k as i32])?;
        let k_flat = k.view(&[batch_heads as i32, seq_len_k as i32, d_k as i32])?;
        let v_flat = v.view(&[batch_heads as i32, seq_len_k as i32, d_k as i32])?;

        // Compute Q @ K^T for each batch*head
        // We need to do this manually since we need per-batch matmul
        let q_data = q_flat.to_vec()?;
        let k_data = k_flat.to_vec()?;
        let v_data = v_flat.to_vec()?;

        let mut scores_data = vec![0.0f32; batch_heads * seq_len_q * seq_len_k];

        // For each batch*head
        for bh in 0..batch_heads {
            let q_offset = bh * seq_len_q * d_k;
            let k_offset = bh * seq_len_k * d_k;
            let scores_offset = bh * seq_len_q * seq_len_k;

            // Compute Q @ K^T for this batch*head
            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    let mut dot_product = 0.0f32;
                    for d in 0..d_k {
                        let q_val = q_data[q_offset + i * d_k + d];
                        let k_val = k_data[k_offset + j * d_k + d];
                        dot_product += q_val * k_val;
                    }
                    // Scale by sqrt(d_k) for numerical stability
                    scores_data[scores_offset + i * seq_len_k + j] =
                        dot_product / (d_k as f32).sqrt();
                }
            }
        }

        // Step 2: Apply mask if provided (add large negative value to masked positions)
        if let Some(mask_tensor) = mask {
            let mask_data = mask_tensor.to_vec()?;
            for i in 0..scores_data.len() {
                if mask_data[i] == 0.0 {
                    scores_data[i] = -1e9; // Large negative value for masked positions
                }
            }
        }

        // Step 3: Apply softmax over the last dimension (seq_len_k)
        // Softmax is applied row-wise (for each query position)
        for bh in 0..batch_heads {
            for i in 0..seq_len_q {
                let row_offset = bh * seq_len_q * seq_len_k + i * seq_len_k;

                // Find max for numerical stability
                let max_val = scores_data[row_offset..row_offset + seq_len_k]
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Compute exp(x - max) and sum
                let mut exp_sum = 0.0f32;
                for j in 0..seq_len_k {
                    let idx = row_offset + j;
                    scores_data[idx] = (scores_data[idx] - max_val).exp();
                    exp_sum += scores_data[idx];
                }

                // Normalize
                for j in 0..seq_len_k {
                    let idx = row_offset + j;
                    scores_data[idx] /= exp_sum + 1e-9; // Add epsilon for numerical stability
                }
            }
        }

        // Step 4: Apply attention to values: attention_weights @ V
        // scores: [batch*heads, seq_q, seq_k]
        // v: [batch*heads, seq_k, d_k]
        // output: [batch*heads, seq_q, d_k]
        let mut output_data = vec![0.0f32; batch_heads * seq_len_q * d_k];

        for bh in 0..batch_heads {
            let scores_offset = bh * seq_len_q * seq_len_k;
            let v_offset = bh * seq_len_k * d_k;
            let output_offset = bh * seq_len_q * d_k;

            for i in 0..seq_len_q {
                for d in 0..d_k {
                    let mut weighted_sum = 0.0f32;
                    for j in 0..seq_len_k {
                        let attention_weight = scores_data[scores_offset + i * seq_len_k + j];
                        let v_val = v_data[v_offset + j * d_k + d];
                        weighted_sum += attention_weight * v_val;
                    }
                    output_data[output_offset + i * d_k + d] = weighted_sum;
                }
            }
        }

        // Reshape back to [batch_size, num_heads, seq_len_q, d_k]
        let output_flat = Tensor::from_vec(output_data, &[batch_heads, seq_len_q, d_k])?;
        output_flat.view(&[
            batch_size as i32,
            num_heads as i32,
            seq_len_q as i32,
            d_k as i32,
        ])
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor, TorshError> {
        let batch_size = input.shape().dims()[0];
        let seq_len = input.shape().dims()[1];
        let d_model = input.shape().dims()[2];

        // Reshape input to 2D for linear transformations: [batch_size * seq_len, d_model]
        let input_2d = input.view(&[(batch_size * seq_len) as i32, d_model as i32])?;

        // Linear transformations for Q, K, V
        let q_2d = input_2d.matmul(&self.w_q.clone_data())?;
        let k_2d = input_2d.matmul(&self.w_k.clone_data())?;
        let v_2d = input_2d.matmul(&self.w_v.clone_data())?;

        // Reshape back to 3D: [batch_size, seq_len, d_model]
        let q = q_2d.view(&[batch_size as i32, seq_len as i32, d_model as i32])?;
        let k = k_2d.view(&[batch_size as i32, seq_len as i32, d_model as i32])?;
        let v = v_2d.view(&[batch_size as i32, seq_len as i32, d_model as i32])?;

        // Add bias if present
        let q = if let Some(ref bias) = self.bias_q {
            q.add(&bias.clone_data())?
        } else {
            q
        };

        let k = if let Some(ref bias) = self.bias_k {
            k.add(&bias.clone_data())?
        } else {
            k
        };

        let v = if let Some(ref bias) = self.bias_v {
            v.add(&bias.clone_data())?
        } else {
            v
        };

        // Reshape for multi-head attention
        // [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        let q = q
            .view(&[
                batch_size as i32,
                seq_len as i32,
                self.num_heads as i32,
                self.d_k as i32,
            ])?
            .transpose(1, 2)?;
        let k = k
            .view(&[
                batch_size as i32,
                seq_len as i32,
                self.num_heads as i32,
                self.d_k as i32,
            ])?
            .transpose(1, 2)?;
        let v = v
            .view(&[
                batch_size as i32,
                seq_len as i32,
                self.num_heads as i32,
                self.d_v as i32,
            ])?
            .transpose(1, 2)?;

        // Apply attention
        let attention_output = self.attention(&q, &k, &v, None)?;

        // Reshape back to original dimensions
        let attention_output = attention_output.transpose(1, 2)?.contiguous()?.view(&[
            batch_size as i32,
            seq_len as i32,
            self.d_model as i32,
        ])?;

        // Final linear transformation - reshape to 2D for matmul
        let output_2d =
            attention_output.view(&[(batch_size * seq_len) as i32, self.d_model as i32])?;
        let output_transformed = output_2d.matmul(&self.w_o.clone_data())?;
        let output =
            output_transformed.view(&[batch_size as i32, seq_len as i32, self.d_model as i32])?;

        if let Some(ref bias) = self.bias_o {
            Ok(output.add(&bias.clone_data())?)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> std::collections::HashMap<String, Parameter> {
        let mut params = std::collections::HashMap::new();
        params.insert("w_q".to_string(), self.w_q.clone());
        params.insert("w_k".to_string(), self.w_k.clone());
        params.insert("w_v".to_string(), self.w_v.clone());
        params.insert("w_o".to_string(), self.w_o.clone());

        if let Some(ref bias) = self.bias_q {
            params.insert("bias_q".to_string(), bias.clone());
        }
        if let Some(ref bias) = self.bias_k {
            params.insert("bias_k".to_string(), bias.clone());
        }
        if let Some(ref bias) = self.bias_v {
            params.insert("bias_v".to_string(), bias.clone());
        }
        if let Some(ref bias) = self.bias_o {
            params.insert("bias_o".to_string(), bias.clone());
        }

        params
    }

    fn train(&mut self) {
        // Set to training mode
    }

    fn eval(&mut self) {
        // Set to evaluation mode
    }
}

/// Advanced Layer Normalization with learnable parameters for transformers
pub struct AdvancedLayerNorm {
    pub normalized_shape: Vec<usize>,
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub eps: f64,
}

impl AdvancedLayerNorm {
    /// Create a new layer normalization layer
    pub fn new(normalized_shape: Vec<usize>, bias: bool, eps: f64) -> Result<Self, TorshError> {
        let num_features = normalized_shape.iter().product();

        let weight = Parameter::new(ones(&[num_features])?);
        let bias = if bias {
            Some(Parameter::new(zeros(&[num_features])?))
        } else {
            None
        };

        Ok(Self {
            normalized_shape,
            weight,
            bias,
            eps,
        })
    }
}

impl Module for AdvancedLayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor, TorshError> {
        // Implement proper layer normalization
        // Layer norm: (x - mean) / sqrt(var + eps) * weight + bias

        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let num_features = self.normalized_shape.iter().product::<usize>();

        // Verify that the normalized shape matches the last dimensions of input
        let input_suffix = &input_shape[input_shape.len() - self.normalized_shape.len()..];
        if input_suffix != self.normalized_shape.as_slice() {
            return Err(TorshError::InvalidArgument(format!(
                "Normalized shape {:?} doesn't match input shape suffix {:?}",
                self.normalized_shape, input_suffix
            )));
        }

        // Calculate batch dimensions
        let batch_size: usize = input_shape[..input_shape.len() - self.normalized_shape.len()]
            .iter()
            .product();

        // Get input data
        let input_data = input.to_vec()?;
        let weight_data = self.weight.clone_data().to_vec()?;
        let bias_data = if let Some(ref bias) = self.bias {
            Some(bias.clone_data().to_vec()?)
        } else {
            None
        };

        let mut output_data = vec![0.0f32; input_data.len()];

        // Process each instance (normalize over the last dimensions)
        for b in 0..batch_size {
            let instance_offset = b * num_features;
            let instance = &input_data[instance_offset..instance_offset + num_features];

            // Compute mean
            let mean: f32 = instance.iter().sum::<f32>() / num_features as f32;

            // Compute variance
            let variance: f32 =
                instance.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / num_features as f32;

            // Normalize and apply affine transformation
            let inv_std = 1.0 / (variance + self.eps as f32).sqrt();

            for i in 0..num_features {
                let normalized = (instance[i] - mean) * inv_std;
                let scaled = normalized * weight_data[i];
                let shifted = if let Some(ref bias) = bias_data {
                    scaled + bias[i]
                } else {
                    scaled
                };
                output_data[instance_offset + i] = shifted;
            }
        }

        Tensor::from_vec(output_data, input_shape)
    }

    fn parameters(&self) -> std::collections::HashMap<String, Parameter> {
        let mut params = std::collections::HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        if let Some(ref bias) = self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

/// Positional Encoding for Transformer models
pub struct PositionalEncoding {
    pub encoding: Tensor,
    pub dropout: f64,
}

impl PositionalEncoding {
    /// Create positional encoding
    ///
    /// Creates sinusoidal positional encodings as described in "Attention Is All You Need"
    /// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    pub fn new(d_model: usize, max_len: usize, dropout: f64) -> Result<Self, TorshError> {
        // Create sinusoidal positional encoding
        let mut encoding_data = vec![0.0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in (0..d_model).step_by(2) {
                let angle = pos as f32 / 10000.0_f32.powf(i as f32 / d_model as f32);

                // Apply sin to even indices
                encoding_data[pos * d_model + i] = angle.sin();

                // Apply cos to odd indices
                if i + 1 < d_model {
                    encoding_data[pos * d_model + i + 1] = angle.cos();
                }
            }
        }

        let encoding = Tensor::from_vec(encoding_data, &[max_len, d_model])?;

        Ok(Self { encoding, dropout })
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, input: &Tensor) -> Result<Tensor, TorshError> {
        // Input shape: [batch_size, seq_len, d_model]
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let seq_len = input_shape[1];
        let d_model = input_shape[2];

        // Get the positional encoding for the sequence length
        // encoding shape: [max_len, d_model]
        // We need: [seq_len, d_model]
        let encoding_shape_binding = self.encoding.shape();
        let encoding_shape = encoding_shape_binding.dims();
        let max_len = encoding_shape[0];

        if seq_len > max_len {
            return Err(TorshError::InvalidArgument(format!(
                "Sequence length {} exceeds maximum positional encoding length {}",
                seq_len, max_len
            )));
        }

        // Slice the encoding to match sequence length
        let encoding_data = self.encoding.to_vec()?;
        let seq_encoding_data: Vec<f32> = encoding_data[..seq_len * d_model].to_vec();

        // Create tensor with shape [seq_len, d_model]
        let seq_encoding = Tensor::from_vec(seq_encoding_data, &[seq_len, d_model])?;

        // Add positional encoding to input
        // input: [batch_size, seq_len, d_model]
        // seq_encoding: [seq_len, d_model]
        // Broadcasting: seq_encoding will be added to each batch

        let input_data = input.to_vec()?;
        let encoding_slice = seq_encoding.to_vec()?;

        let batch_size = input_shape[0];
        let mut output_data = vec![0.0f32; input_data.len()];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..d_model {
                    let input_idx = b * seq_len * d_model + s * d_model + d;
                    let encoding_idx = s * d_model + d;
                    output_data[input_idx] = input_data[input_idx] + encoding_slice[encoding_idx];
                }
            }
        }

        let output = Tensor::from_vec(output_data, input_shape)?;

        // Note: Dropout would be applied here in training mode
        // For now, dropout is not implemented as it requires training mode tracking
        Ok(output)
    }

    fn parameters(&self) -> std::collections::HashMap<String, Parameter> {
        // Positional encoding is not learnable
        std::collections::HashMap::new()
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_head_attention() {
        let mha = MultiHeadAttention::new(512, 8, 0.1, true).unwrap();
        let input = randn(&[2, 10, 512]).unwrap(); // batch_size=2, seq_len=10, d_model=512

        let output = mha.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 10, 512]);
    }

    #[test]
    fn test_advanced_layer_norm() {
        let ln = AdvancedLayerNorm::new(vec![512], true, 1e-5).unwrap();
        let input = randn(&[2, 10, 512]).unwrap();

        let output = ln.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 10, 512]);
    }
}
