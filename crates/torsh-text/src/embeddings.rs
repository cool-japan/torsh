use crate::{Result, TextError};
use std::collections::HashMap;
use std::path::Path;
use torsh_core::{device::DeviceType as Device, dtype::DType};
use torsh_linalg::matrix_functions::matrix_norm;
use torsh_tensor::creation::{arange, rand, randn, tensor_1d};
use torsh_tensor::Tensor;

// Temporary placeholder types until torsh-nn is available
#[derive(Debug, Clone)]
pub struct Parameter {
    tensor: Tensor,
    requires_grad: bool,
}

impl Parameter {
    pub fn new(tensor: Tensor, requires_grad: bool) -> Self {
        Self {
            tensor,
            requires_grad,
        }
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }
}

pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn parameters(&self) -> Vec<&Parameter>;
    fn train(&mut self, mode: bool);
    fn eval(&mut self);
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: Parameter,
    bias: Parameter,
    eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: usize, eps: f32, device: &Device, _dtype: DType) -> Result<Self> {
        let weight = Parameter::new(Tensor::<f32>::ones(&[normalized_shape], *device)?, true);
        let bias = Parameter::new(Tensor::<f32>::zeros(&[normalized_shape], *device)?, true);

        Ok(Self { weight, bias, eps })
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified layer norm implementation
        let last_dim = input.shape().dims().len() - 1;
        let mean = input.mean(Some(&[last_dim]), true)?;
        let diff = input - &mean;
        let variance = diff.pow(2.0)?.mean(Some(&[last_dim]), true)?;
        let eps_tensor = Tensor::from_vec(vec![self.eps], &[1])?;
        let var_plus_eps = &variance + &eps_tensor;
        let std_dev = var_plus_eps.sqrt()?;
        let normalized = &diff / &std_dev;
        let scaled = &normalized * self.weight.tensor();
        let shifted = &scaled + self.bias.tensor();
        Ok(shifted)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight, &self.bias]
    }

    fn train(&mut self, _mode: bool) {
        // No-op for this simple implementation
    }

    fn eval(&mut self) {
        // No-op for this simple implementation
    }
}

// ============================================================================
// Word Embeddings
// ============================================================================

#[derive(Debug, Clone)]
pub struct WordEmbedding {
    pub weight: Parameter,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
    pub max_norm: Option<f32>,
    pub norm_type: f32,
    pub scale_grad_by_freq: bool,
    pub sparse: bool,
}

impl WordEmbedding {
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        max_norm: Option<f32>,
        norm_type: f32,
        scale_grad_by_freq: bool,
        sparse: bool,
        _device: &Device,
        _dtype: DType,
    ) -> Result<Self> {
        let weight = Parameter::new(randn::<f32>(&[vocab_size, embedding_dim])?, true);

        // Zero out padding embeddings if specified
        if let Some(pad_idx) = padding_idx {
            if pad_idx < vocab_size {
                let weight_data = weight.tensor().clone();
                let mut padding_row =
                    weight_data.index_select(0, &tensor_1d(&[pad_idx as i64])?)?;
                let _ = padding_row.fill_(0.0)?;
            }
        }

        Ok(Self {
            weight,
            vocab_size,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
        })
    }

    pub fn from_pretrained(
        embeddings: Tensor,
        freeze: bool,
        padding_idx: Option<usize>,
        max_norm: Option<f32>,
        norm_type: f32,
        scale_grad_by_freq: bool,
        sparse: bool,
    ) -> Result<Self> {
        let shape = embeddings.shape();
        if shape.ndim() != 2 {
            return Err(TextError::ModelError(
                "Embedding tensor must be 2-dimensional".to_string(),
            ));
        }

        let dims = shape.dims();
        let vocab_size = dims[0];
        let embedding_dim = dims[1];

        let weight = Parameter::new(embeddings, !freeze);

        Ok(Self {
            weight,
            vocab_size,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn simple(
        vocab_size: usize,
        embedding_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::new(
            vocab_size,
            embedding_dim,
            None,
            None,
            2.0,
            false,
            false,
            device,
            dtype,
        )
    }
}

impl Module for WordEmbedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Convert float indices to i64 for indexing
        let float_data = input.to_vec()?;
        let int_data: Vec<i64> = float_data.iter().map(|&x| x.round() as i64).collect();
        let indices = Tensor::from_vec(int_data, input.shape().dims())?;
        let embeddings = self.weight.tensor().index_select(0, &indices)?;

        // Apply max norm if specified
        if let Some(max_norm) = self.max_norm {
            // Compute L2 norm along the last dimension for each embedding
            let squared = embeddings.pow(2.0)?;
            let last_dim = (embeddings.shape().ndim() - 1) as i32;
            let norms = squared.sum_dim(&[last_dim], true)?;
            let l2_norms = norms.sqrt()?;

            // Create scale factor tensor: min(max_norm / norm, 1.0)
            let max_norm_tensor = Tensor::from_vec(vec![max_norm], &[1])?;
            let scale_factors = l2_norms.div(&max_norm_tensor)?;
            let ones = Tensor::ones_like(&scale_factors)?;
            let scale_factors = scale_factors.minimum(&ones)?;

            // Apply scaling to embeddings
            let scaled_embeddings = embeddings.mul_op(&scale_factors)?;
            return Ok(scaled_embeddings);
        }

        Ok(embeddings)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight]
    }

    fn train(&mut self, _mode: bool) {
        // No-op for embeddings
    }

    fn eval(&mut self) {
        // No-op for embeddings
    }
}

// ============================================================================
// Positional Encoding
// ============================================================================

#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    pub encoding: Tensor,
    pub dropout: f32,
    pub max_len: usize,
    pub d_model: usize,
}

impl PositionalEncoding {
    pub fn new(
        d_model: usize,
        dropout: f32,
        max_len: usize,
        _device: &Device,
        _dtype: DType,
    ) -> Result<Self> {
        // Create proper sinusoidal positional encoding using CPU computation
        // This is more efficient than attempting tensor operations for sine/cosine

        // Create proper sinusoidal positional encoding
        // For even indices (0, 2, 4, ...), use sine
        // For odd indices (1, 3, 5, ...), use cosine
        let mut pe_data = vec![0.0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle =
                    pos as f32 * (-10000.0_f32.ln() / d_model as f32 * (i / 2) as f32).exp();
                if i % 2 == 0 {
                    pe_data[pos * d_model + i] = angle.sin();
                } else {
                    pe_data[pos * d_model + i] = angle.cos();
                }
            }
        }

        let pe = Tensor::from_vec(pe_data, &[max_len, d_model])?;

        Ok(Self {
            encoding: pe,
            dropout,
            max_len,
            d_model,
        })
    }

    pub fn sinusoidal(
        d_model: usize,
        max_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::new(d_model, 0.0, max_len, device, dtype)
    }

    pub fn learnable(
        d_model: usize,
        max_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<LearnablePositionalEncoding> {
        LearnablePositionalEncoding::new(d_model, max_len, device, dtype)
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let seq_len = input.shape().dims()[1];
        if seq_len > self.max_len {
            return Err(TextError::ModelError(format!(
                "Sequence length {} exceeds maximum length {}",
                seq_len, self.max_len
            )));
        }

        let pos_encoding = self.encoding.narrow(0, 0, seq_len)?.unsqueeze(0)?;

        let output = input + &pos_encoding;

        if self.dropout > 0.0 {
            // Apply dropout (simplified - would need proper dropout implementation)
            Ok(output)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> Vec<&Parameter> {
        vec![] // Sinusoidal encoding has no learnable parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op for fixed positional encoding
    }

    fn eval(&mut self) {
        // No-op for fixed positional encoding
    }
}

// ============================================================================
// Learnable Positional Encoding
// ============================================================================

#[derive(Debug, Clone)]
pub struct LearnablePositionalEncoding {
    pub weight: Parameter,
    pub max_len: usize,
    pub d_model: usize,
}

impl LearnablePositionalEncoding {
    pub fn new(d_model: usize, max_len: usize, _device: &Device, _dtype: DType) -> Result<Self> {
        let weight = Parameter::new(randn::<f32>(&[max_len, d_model])?, true);

        Ok(Self {
            weight,
            max_len,
            d_model,
        })
    }
}

impl Module for LearnablePositionalEncoding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let seq_len = input.shape().dims()[1];
        if seq_len > self.max_len {
            return Err(TextError::ModelError(format!(
                "Sequence length {} exceeds maximum length {}",
                seq_len, self.max_len
            )));
        }

        let pos_encoding = self.weight.tensor().narrow(0, 0, seq_len)?.unsqueeze(0)?;

        Ok(input + &pos_encoding)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight]
    }

    fn train(&mut self, _mode: bool) {
        // No-op
    }

    fn eval(&mut self) {
        // No-op
    }
}

// ============================================================================
// Token Type Embeddings
// ============================================================================

#[derive(Debug, Clone)]
pub struct TokenTypeEmbedding {
    pub embedding: WordEmbedding,
    pub num_types: usize,
}

impl TokenTypeEmbedding {
    pub fn new(
        num_types: usize,
        embedding_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let embedding = WordEmbedding::simple(num_types, embedding_dim, device, dtype)?;

        Ok(Self {
            embedding,
            num_types,
        })
    }
}

impl Module for TokenTypeEmbedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.embedding.forward(input)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        self.embedding.parameters()
    }

    fn train(&mut self, mode: bool) {
        self.embedding.train(mode);
    }

    fn eval(&mut self) {
        self.embedding.eval();
    }
}

// ============================================================================
// Combined Embeddings (Token + Position + Type)
// ============================================================================

#[derive(Debug, Clone)]
pub struct CombinedEmbeddings {
    pub token_embeddings: WordEmbedding,
    pub position_embeddings: Option<LearnablePositionalEncoding>,
    pub token_type_embeddings: Option<TokenTypeEmbedding>,
    pub layer_norm: Option<LayerNorm>,
    pub dropout: f32,
}

impl CombinedEmbeddings {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        max_position_embeddings: usize,
        num_token_types: Option<usize>,
        layer_norm_eps: Option<f32>,
        dropout: f32,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let token_embeddings = WordEmbedding::simple(vocab_size, embedding_dim, device, dtype)?;

        let position_embeddings = Some(LearnablePositionalEncoding::new(
            embedding_dim,
            max_position_embeddings,
            device,
            dtype,
        )?);

        let token_type_embeddings = if let Some(num_types) = num_token_types {
            Some(TokenTypeEmbedding::new(
                num_types,
                embedding_dim,
                device,
                dtype,
            )?)
        } else {
            None
        };

        let layer_norm = if let Some(eps) = layer_norm_eps {
            Some(LayerNorm::new(embedding_dim, eps, device, dtype)?)
        } else {
            None
        };

        Ok(Self {
            token_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        })
    }

    pub fn forward_with_positions_and_types(
        &self,
        input_ids: &Tensor,
        position_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut embeddings = self.token_embeddings.forward(input_ids)?;

        // Add position embeddings
        if let Some(pos_emb) = &self.position_embeddings {
            if let Some(pos_ids) = position_ids {
                // Convert position_ids to i64 for indexing
                let float_data: Vec<f32> = pos_ids.to_vec()?;
                let mut int_data = Vec::new();
                for val in float_data.iter() {
                    int_data.push(val.round() as i64);
                }
                let pos_indices = Tensor::from_vec(int_data, pos_ids.shape().dims())?;
                let pos_emb_selected = pos_emb.weight.tensor().index_select(0, &pos_indices)?;
                embeddings = &embeddings + &pos_emb_selected;
            } else {
                // Use default position indices
                let seq_len = input_ids.shape().dims()[1];
                let pos_ids = arange(0i64, seq_len as i64, 1i64)?;
                let pos_emb_selected = pos_emb.weight.tensor().index_select(0, &pos_ids)?;
                embeddings = &embeddings + &pos_emb_selected;
            }
        }

        // Add token type embeddings
        if let Some(type_emb) = &self.token_type_embeddings {
            if let Some(type_ids) = token_type_ids {
                let type_emb_out = type_emb.forward(type_ids)?;
                embeddings = &embeddings + &type_emb_out;
            } else {
                // Use default token type (all zeros)
                let zeros = Tensor::zeros_like(input_ids)?;
                let type_emb_out = type_emb.forward(&zeros)?;
                embeddings = &embeddings + &type_emb_out;
            }
        }

        // Apply layer normalization
        if let Some(ln) = &self.layer_norm {
            embeddings = ln.forward(&embeddings)?;
        }

        // Apply dropout (simplified - would need proper dropout implementation)
        if self.dropout > 0.0 {
            // Would apply dropout here
        }

        Ok(embeddings)
    }
}

impl Module for CombinedEmbeddings {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_positions_and_types(input, None, None)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = self.token_embeddings.parameters();

        if let Some(pos_emb) = &self.position_embeddings {
            params.extend(pos_emb.parameters());
        }

        if let Some(type_emb) = &self.token_type_embeddings {
            params.extend(type_emb.parameters());
        }

        if let Some(ln) = &self.layer_norm {
            params.extend(ln.parameters());
        }

        params
    }

    fn train(&mut self, mode: bool) {
        self.token_embeddings.train(mode);
        if let Some(pos_emb) = &mut self.position_embeddings {
            pos_emb.train(mode);
        }
        if let Some(type_emb) = &mut self.token_type_embeddings {
            type_emb.train(mode);
        }
        if let Some(ln) = &mut self.layer_norm {
            ln.train(mode);
        }
    }

    fn eval(&mut self) {
        self.token_embeddings.eval();
        if let Some(pos_emb) = &mut self.position_embeddings {
            pos_emb.eval();
        }
        if let Some(type_emb) = &mut self.token_type_embeddings {
            type_emb.eval();
        }
        if let Some(ln) = &mut self.layer_norm {
            ln.eval();
        }
    }
}

// ============================================================================
// Embedding Utilities
// ============================================================================

pub struct EmbeddingUtils;

impl EmbeddingUtils {
    /// Load pre-trained word embeddings from a text file (word2vec/GloVe format)
    pub fn load_pretrained_embeddings(
        path: &Path,
        vocab: &HashMap<String, u32>,
        embedding_dim: usize,
        _device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path).map_err(|e| TextError::IoError(e))?;
        let reader = BufReader::new(file);

        let vocab_size = vocab.len();
        let mut embedding_matrix = vec![vec![0.0f32; embedding_dim]; vocab_size];

        for line in reader.lines() {
            let line = line.map_err(|e| TextError::IoError(e))?;
            let parts: Vec<&str> = line.trim().split_whitespace().collect();

            if parts.len() != embedding_dim + 1 {
                continue;
            }

            let word = parts[0];
            if let Some(&idx) = vocab.get(word) {
                for (i, &value_str) in parts[1..].iter().enumerate() {
                    if let Ok(value) = value_str.parse::<f32>() {
                        embedding_matrix[idx as usize][i] = value;
                    }
                }
            }
        }

        // Convert to tensor
        let flat_embeddings: Vec<f32> = embedding_matrix.into_iter().flatten().collect();
        let tensor =
            Tensor::from_vec(flat_embeddings, &[vocab_size, embedding_dim])?.to_dtype(dtype)?;

        Ok(tensor)
    }

    /// Initialize embeddings with Xavier/Glorot uniform distribution
    pub fn xavier_uniform_embeddings(
        vocab_size: usize,
        embedding_dim: usize,
        _device: &Device,
        _dtype: DType,
    ) -> Result<Tensor> {
        let fan_in = vocab_size as f32;
        let fan_out = embedding_dim as f32;
        let bound = (6.0 / (fan_in + fan_out)).sqrt();

        let uniform = rand(&[vocab_size, embedding_dim])?;
        Ok(uniform.mul_scalar(2.0 * bound)?.add_scalar(-bound)?)
    }

    /// Initialize embeddings with Xavier/Glorot normal distribution
    pub fn xavier_normal_embeddings(
        vocab_size: usize,
        embedding_dim: usize,
        _device: &Device,
        _dtype: DType,
    ) -> Result<Tensor> {
        let fan_in = vocab_size as f32;
        let fan_out = embedding_dim as f32;
        let std = (2.0 / (fan_in + fan_out)).sqrt();

        Ok(randn(&[vocab_size, embedding_dim])?.mul_scalar(std)?)
    }

    /// Compute cosine similarity between embeddings
    pub fn cosine_similarity(embeddings: &Tensor, idx1: usize, idx2: usize) -> Result<f32> {
        let emb1 = embeddings.select(0, idx1 as i64)?;
        let emb2 = embeddings.select(0, idx2 as i64)?;

        let dot_product = emb1.mul_op(&emb2)?.sum()?;
        let norm1 = emb1.mul_op(&emb1)?.sum()?.sqrt()?;
        let norm2 = emb2.mul_op(&emb2)?.sum()?.sqrt()?;

        let similarity = dot_product.div(&norm1.mul_op(&norm2)?)?;
        Ok(similarity.item()?)
    }

    /// Find most similar embeddings to a given embedding
    pub fn most_similar(
        embeddings: &Tensor,
        target_idx: usize,
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let _target_emb = embeddings.select(0, target_idx as i64)?;
        let vocab_size = embeddings.shape().dims()[0];

        let mut similarities = Vec::new();

        for i in 0..vocab_size {
            if i == target_idx {
                continue;
            }

            let similarity = Self::cosine_similarity(embeddings, target_idx, i)?;
            similarities.push((i, similarity));
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(top_k);

        Ok(similarities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::{device::DeviceType as Device, dtype::DType};

    #[test]
    fn test_word_embedding_creation() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vocab_size = 1000;
        let embedding_dim = 300;

        let embedding = WordEmbedding::simple(vocab_size, embedding_dim, &device, dtype);
        assert!(embedding.is_ok());

        let embedding = embedding.unwrap();
        assert_eq!(embedding.vocab_size, vocab_size);
        assert_eq!(embedding.embedding_dim, embedding_dim);
    }

    #[test]
    fn test_positional_encoding_creation() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let d_model = 512;
        let max_len = 1000;

        let pos_encoding = PositionalEncoding::sinusoidal(d_model, max_len, &device, dtype);
        assert!(pos_encoding.is_ok());

        let pos_encoding = pos_encoding.unwrap();
        assert_eq!(pos_encoding.d_model, d_model);
        assert_eq!(pos_encoding.max_len, max_len);
    }

    #[test]
    fn test_embedding_forward() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vocab_size = 100;
        let embedding_dim = 64;

        let embedding = WordEmbedding::simple(vocab_size, embedding_dim, &device, dtype).unwrap();
        let input = Tensor::from_vec(vec![1.0_f32, 5.0, 10.0, 25.0], &[4]).unwrap();

        let result = embedding.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims(), &[4, embedding_dim]);
    }
}
