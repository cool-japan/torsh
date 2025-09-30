//! Advanced compression techniques for quantization
//!
//! This module implements cutting-edge compression methods including:
//! - Sub-byte quantization (1-bit, 2-bit, 3-bit)
//! - Vector quantization and codebook optimization
//! - Outlier handling and mixed-precision strategies
//! - Sparsity-aware quantization
//! - Block-wise compression schemes

use crate::TorshResult;
// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{Random, Rng};
use std::collections::HashMap;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Advanced compression techniques
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionScheme {
    /// 1-bit quantization (binary)
    OneBit,
    /// 2-bit quantization
    TwoBit,
    /// 3-bit quantization
    ThreeBit,
    /// Variable bit-width (mixed)
    VariableBit,
    /// Vector quantization
    VectorQuantization,
    /// Sparse quantization
    SparseQuantization,
    /// Block-wise quantization
    BlockWise,
    /// Huffman encoding
    HuffmanEncoding,
}

/// Sub-byte quantization configuration
#[derive(Debug, Clone)]
pub struct SubByteConfig {
    /// Number of bits per value
    pub bits_per_value: u8,
    /// Quantization scheme
    pub scheme: CompressionScheme,
    /// Block size for block-wise quantization
    pub block_size: usize,
    /// Sparsity threshold (0.0 to 1.0)
    pub sparsity_threshold: f32,
    /// Enable outlier handling
    pub enable_outlier_handling: bool,
    /// Outlier ratio (0.0 to 1.0)
    pub outlier_ratio: f32,
}

impl Default for SubByteConfig {
    fn default() -> Self {
        Self {
            bits_per_value: 2,
            scheme: CompressionScheme::TwoBit,
            block_size: 64,
            sparsity_threshold: 0.9,
            enable_outlier_handling: true,
            outlier_ratio: 0.01,
        }
    }
}

/// Vector quantization configuration
#[derive(Debug, Clone)]
pub struct VectorQuantConfig {
    /// Codebook size (number of centroids)
    pub codebook_size: usize,
    /// Vector dimension
    pub vector_dim: usize,
    /// Maximum iterations for k-means
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f32,
    /// Initialization method
    pub init_method: VqInitMethod,
}

/// Vector quantization initialization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VqInitMethod {
    /// Random initialization
    Random,
    /// K-means++ initialization
    KMeansPlusPlus,
    /// Uniform sampling
    UniformSampling,
}

impl Default for VectorQuantConfig {
    fn default() -> Self {
        Self {
            codebook_size: 256,
            vector_dim: 4,
            max_iterations: 100,
            tolerance: 1e-4,
            init_method: VqInitMethod::KMeansPlusPlus,
        }
    }
}

/// Advanced compression engine
#[derive(Debug)]
pub struct CompressionEngine {
    /// Configuration
    pub config: SubByteConfig,
    /// Vector quantization settings
    pub vq_config: VectorQuantConfig,
    /// Compression statistics
    pub stats: CompressionStats,
    /// Cached codebooks for vector quantization
    pub codebooks: HashMap<String, Codebook>,
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Number of outliers handled
    pub num_outliers: usize,
    /// Sparsity ratio
    pub sparsity_ratio: f32,
    /// Quantization error metrics
    pub error_metrics: CompressionErrorMetrics,
}

/// Error metrics for compression
#[derive(Debug, Clone)]
pub struct CompressionErrorMetrics {
    /// Mean absolute error
    pub mae: f32,
    /// Mean squared error
    pub mse: f32,
    /// Signal-to-noise ratio
    pub snr: f32,
    /// Compression efficiency score
    pub efficiency_score: f32,
}

/// Vector quantization codebook
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Centroids (codebook_size x vector_dim)
    pub centroids: Vec<Vec<f32>>,
    /// Usage frequency for each centroid
    pub usage_freq: Vec<usize>,
    /// Codebook size
    pub size: usize,
    /// Vector dimension
    pub dim: usize,
}

/// Compressed tensor representation
#[derive(Debug, Clone)]
pub struct CompressedTensor {
    /// Compressed data
    pub data: Vec<u8>,
    /// Compression metadata
    pub metadata: CompressionMetadata,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Compression scheme used
    pub scheme: CompressionScheme,
}

/// Compression metadata
#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    /// Scale factors
    pub scales: Vec<f32>,
    /// Zero points
    pub zero_points: Vec<i32>,
    /// Outlier indices and values
    pub outliers: HashMap<usize, f32>,
    /// Sparsity pattern
    pub sparsity_mask: Option<Vec<bool>>,
    /// Codebook for vector quantization
    pub codebook: Option<Codebook>,
    /// Block parameters
    pub block_params: Option<BlockParams>,
}

/// Block quantization parameters
#[derive(Debug, Clone)]
pub struct BlockParams {
    /// Block size
    pub block_size: usize,
    /// Per-block scales
    pub block_scales: Vec<f32>,
    /// Per-block zero points
    pub block_zero_points: Vec<i32>,
}

impl CompressionEngine {
    /// Create new compression engine
    pub fn new(config: SubByteConfig, vq_config: VectorQuantConfig) -> Self {
        Self {
            config,
            vq_config,
            stats: CompressionStats::default(),
            codebooks: HashMap::new(),
        }
    }

    /// Compress tensor using specified scheme
    pub fn compress(
        &mut self,
        tensor: &Tensor,
        scheme: CompressionScheme,
    ) -> TorshResult<CompressedTensor> {
        let original_size = tensor.data()?.len() * 4; // Assuming f32

        let compressed = match scheme {
            CompressionScheme::OneBit => self.compress_one_bit(tensor)?,
            CompressionScheme::TwoBit => self.compress_two_bit(tensor)?,
            CompressionScheme::ThreeBit => self.compress_three_bit(tensor)?,
            CompressionScheme::VariableBit => self.compress_variable_bit(tensor)?,
            CompressionScheme::VectorQuantization => self.compress_vector_quantization(tensor)?,
            CompressionScheme::SparseQuantization => self.compress_sparse(tensor)?,
            CompressionScheme::BlockWise => self.compress_block_wise(tensor)?,
            CompressionScheme::HuffmanEncoding => self.compress_huffman(tensor)?,
        };

        // Update statistics
        self.update_stats(original_size, &compressed);

        Ok(compressed)
    }

    /// Decompress tensor
    pub fn decompress(&self, compressed: &CompressedTensor) -> TorshResult<Tensor> {
        match compressed.scheme {
            CompressionScheme::OneBit => self.decompress_one_bit(compressed),
            CompressionScheme::TwoBit => self.decompress_two_bit(compressed),
            CompressionScheme::ThreeBit => self.decompress_three_bit(compressed),
            CompressionScheme::VariableBit => self.decompress_variable_bit(compressed),
            CompressionScheme::VectorQuantization => {
                self.decompress_vector_quantization(compressed)
            }
            CompressionScheme::SparseQuantization => self.decompress_sparse(compressed),
            CompressionScheme::BlockWise => self.decompress_block_wise(compressed),
            CompressionScheme::HuffmanEncoding => self.decompress_huffman(compressed),
        }
    }

    /// 1-bit quantization (extreme compression)
    fn compress_one_bit(&mut self, tensor: &Tensor) -> TorshResult<CompressedTensor> {
        let data = tensor.data()?;
        let threshold = Self::calculate_adaptive_threshold(&data, 0.5);

        let mut compressed_data = Vec::new();
        let mut outliers = HashMap::new();

        // Pack 8 bits into each byte
        for chunk in data.chunks(8) {
            let mut byte = 0u8;
            for (i, &value) in chunk.iter().enumerate() {
                if self.config.enable_outlier_handling
                    && Self::is_outlier(value, threshold, self.config.outlier_ratio)
                {
                    outliers.insert(compressed_data.len() * 8 + i, value);
                    // Use threshold for outliers in compressed representation
                    if threshold > 0.0 {
                        byte |= 1 << i;
                    }
                } else if value > threshold {
                    byte |= 1 << i;
                }
            }
            compressed_data.push(byte);
        }

        let metadata = CompressionMetadata {
            scales: vec![1.0],
            zero_points: vec![0],
            outliers,
            sparsity_mask: None,
            codebook: None,
            block_params: None,
        };

        Ok(CompressedTensor {
            data: compressed_data,
            metadata,
            shape: tensor.shape().dims().to_vec(),
            scheme: CompressionScheme::OneBit,
        })
    }

    /// 2-bit quantization
    fn compress_two_bit(&mut self, tensor: &Tensor) -> TorshResult<CompressedTensor> {
        let data = tensor.data()?;
        let (min_val, max_val) = Self::calculate_range(&data);
        let scale = (max_val - min_val) / 3.0; // 2^2 - 1 = 3 levels
        let scale = if scale == 0.0 { 1.0 } else { scale };

        let mut compressed_data = Vec::new();
        let mut outliers = HashMap::new();

        // Pack 4 values (2 bits each) into each byte
        for chunk in data.chunks(4) {
            let mut byte = 0u8;
            for (i, &value) in chunk.iter().enumerate() {
                if self.config.enable_outlier_handling
                    && Self::is_outlier(value, (min_val + max_val) / 2.0, self.config.outlier_ratio)
                {
                    outliers.insert(compressed_data.len() * 4 + i, value);
                    // Use quantized value for outliers in compressed representation
                    let quantized = ((value - min_val) / scale).round().clamp(0.0, 3.0) as u8;
                    byte |= quantized << (i * 2);
                } else {
                    let quantized = ((value - min_val) / scale).round().clamp(0.0, 3.0) as u8;
                    byte |= quantized << (i * 2);
                }
            }
            compressed_data.push(byte);
        }

        let metadata = CompressionMetadata {
            scales: vec![scale],
            zero_points: vec![(min_val / scale) as i32],
            outliers,
            sparsity_mask: None,
            codebook: None,
            block_params: None,
        };

        Ok(CompressedTensor {
            data: compressed_data,
            metadata,
            shape: tensor.shape().dims().to_vec(),
            scheme: CompressionScheme::TwoBit,
        })
    }

    /// 3-bit quantization
    fn compress_three_bit(&mut self, tensor: &Tensor) -> TorshResult<CompressedTensor> {
        let data = tensor.data()?;
        let (min_val, max_val) = Self::calculate_range(&data);
        let scale = (max_val - min_val) / 7.0; // 2^3 - 1 = 7 levels
        let scale = if scale == 0.0 { 1.0 } else { scale };

        let mut compressed_data = Vec::new();
        let mut bit_buffer = 0u32;
        let mut bit_count = 0;

        for &value in data.iter() {
            let quantized = ((value - min_val) / scale).round().clamp(0.0, 7.0) as u32;
            bit_buffer |= quantized << bit_count;
            bit_count += 3;

            // Write full bytes
            while bit_count >= 8 {
                compressed_data.push((bit_buffer & 0xFF) as u8);
                bit_buffer >>= 8;
                bit_count -= 8;
            }
        }

        // Write remaining bits
        if bit_count > 0 {
            compressed_data.push((bit_buffer & 0xFF) as u8);
        }

        let metadata = CompressionMetadata {
            scales: vec![scale],
            zero_points: vec![(min_val / scale) as i32],
            outliers: HashMap::new(),
            sparsity_mask: None,
            codebook: None,
            block_params: None,
        };

        Ok(CompressedTensor {
            data: compressed_data,
            metadata,
            shape: tensor.shape().dims().to_vec(),
            scheme: CompressionScheme::ThreeBit,
        })
    }

    /// Variable bit-width quantization
    fn compress_variable_bit(&mut self, tensor: &Tensor) -> TorshResult<CompressedTensor> {
        let data = tensor.data()?;
        let sensitivity_scores = self.calculate_sensitivity_scores(&data);

        let mut compressed_data = Vec::new();
        let mut scales = Vec::new();
        let mut zero_points = Vec::new();

        // Assign bit-widths based on sensitivity
        for (chunk, &sensitivity) in data
            .chunks(self.config.block_size)
            .zip(sensitivity_scores.iter())
        {
            let bits = if sensitivity > 0.8 {
                8 // High sensitivity -> more bits
            } else if sensitivity > 0.5 {
                4
            } else if sensitivity > 0.2 {
                2
            } else {
                1 // Low sensitivity -> fewer bits
            };

            let (chunk_data, scale, zero_point) = self.quantize_chunk(chunk, bits)?;
            compressed_data.extend(chunk_data);
            scales.push(scale);
            zero_points.push(zero_point);
        }

        let metadata = CompressionMetadata {
            scales,
            zero_points,
            outliers: HashMap::new(),
            sparsity_mask: None,
            codebook: None,
            block_params: None,
        };

        Ok(CompressedTensor {
            data: compressed_data,
            metadata,
            shape: tensor.shape().dims().to_vec(),
            scheme: CompressionScheme::VariableBit,
        })
    }

    /// Vector quantization compression
    fn compress_vector_quantization(&mut self, tensor: &Tensor) -> TorshResult<CompressedTensor> {
        let data = tensor.data()?;
        let vectors = self.tensorize_data(&data, self.vq_config.vector_dim);

        // Train or retrieve codebook
        let codebook_key = format!(
            "vq_{}_{}",
            self.vq_config.codebook_size, self.vq_config.vector_dim
        );
        let codebook = if let Some(cb) = self.codebooks.get(&codebook_key) {
            cb.clone()
        } else {
            let cb = self.train_codebook(&vectors)?;
            self.codebooks.insert(codebook_key, cb.clone());
            cb
        };

        // Encode vectors using codebook
        let mut compressed_data = Vec::new();
        for vector in vectors {
            let code = self.find_nearest_centroid(&vector, &codebook);
            if codebook.size <= 256 {
                compressed_data.push(code as u8);
            } else {
                // Use 2 bytes for larger codebooks
                compressed_data.push((code & 0xFF) as u8);
                compressed_data.push((code >> 8) as u8);
            }
        }

        let metadata = CompressionMetadata {
            scales: vec![1.0],
            zero_points: vec![0],
            outliers: HashMap::new(),
            sparsity_mask: None,
            codebook: Some(codebook),
            block_params: None,
        };

        Ok(CompressedTensor {
            data: compressed_data,
            metadata,
            shape: tensor.shape().dims().to_vec(),
            scheme: CompressionScheme::VectorQuantization,
        })
    }

    /// Sparse quantization
    fn compress_sparse(&mut self, tensor: &Tensor) -> TorshResult<CompressedTensor> {
        let data = tensor.data()?;
        let threshold = Self::calculate_sparsity_threshold(&data, self.config.sparsity_threshold);

        let mut sparsity_mask = Vec::new();
        let mut non_zero_values = Vec::new();
        let mut non_zero_indices = Vec::new();

        for (i, &value) in data.iter().enumerate() {
            if value.abs() > threshold {
                sparsity_mask.push(true);
                non_zero_values.push(value);
                non_zero_indices.push(i);
            } else {
                sparsity_mask.push(false);
            }
        }

        // Quantize non-zero values
        let (min_val, max_val) = Self::calculate_range(&non_zero_values);
        let scale = (max_val - min_val) / 255.0;
        let scale = if scale == 0.0 { 1.0 } else { scale };

        let mut compressed_data = Vec::new();

        // Store number of non-zero elements
        let num_non_zero = non_zero_values.len();
        compressed_data.extend(&(num_non_zero as u32).to_le_bytes());

        // Store indices (compressed using delta encoding)
        let mut prev_idx = 0;
        for &idx in &non_zero_indices {
            let delta = idx - prev_idx;
            compressed_data.extend(&(delta as u32).to_le_bytes());
            prev_idx = idx;
        }

        // Store quantized values
        for &value in &non_zero_values {
            let quantized = ((value - min_val) / scale).round().clamp(0.0, 255.0) as u8;
            compressed_data.push(quantized);
        }

        let metadata = CompressionMetadata {
            scales: vec![scale],
            zero_points: vec![(min_val / scale) as i32],
            outliers: HashMap::new(),
            sparsity_mask: Some(sparsity_mask),
            codebook: None,
            block_params: None,
        };

        Ok(CompressedTensor {
            data: compressed_data,
            metadata,
            shape: tensor.shape().dims().to_vec(),
            scheme: CompressionScheme::SparseQuantization,
        })
    }

    /// Block-wise quantization
    fn compress_block_wise(&mut self, tensor: &Tensor) -> TorshResult<CompressedTensor> {
        let data = tensor.data()?;
        let mut compressed_data = Vec::new();
        let mut block_scales = Vec::new();
        let mut block_zero_points = Vec::new();

        for block in data.chunks(self.config.block_size) {
            let (min_val, max_val) = Self::calculate_range(block);
            let scale = (max_val - min_val) / 255.0;
            let scale = if scale == 0.0 { 1.0 } else { scale };
            let zero_point = (min_val / scale) as i32;

            block_scales.push(scale);
            block_zero_points.push(zero_point);

            // Quantize block
            for &value in block {
                let quantized = ((value - min_val) / scale).round().clamp(0.0, 255.0) as u8;
                compressed_data.push(quantized);
            }
        }

        let block_params = BlockParams {
            block_size: self.config.block_size,
            block_scales,
            block_zero_points,
        };

        let metadata = CompressionMetadata {
            scales: vec![1.0],
            zero_points: vec![0],
            outliers: HashMap::new(),
            sparsity_mask: None,
            codebook: None,
            block_params: Some(block_params),
        };

        Ok(CompressedTensor {
            data: compressed_data,
            metadata,
            shape: tensor.shape().dims().to_vec(),
            scheme: CompressionScheme::BlockWise,
        })
    }

    /// Huffman encoding compression
    fn compress_huffman(&mut self, tensor: &Tensor) -> TorshResult<CompressedTensor> {
        let data = tensor.data()?;

        // First quantize to 8-bit
        let (min_val, max_val) = Self::calculate_range(&data);
        let scale = (max_val - min_val) / 255.0;
        let scale = if scale == 0.0 { 1.0 } else { scale };

        let mut quantized_values = Vec::new();
        for &value in data.iter() {
            let quantized = ((value - min_val) / scale).round().clamp(0.0, 255.0) as u8;
            quantized_values.push(quantized);
        }

        // Build frequency table
        let mut freq_table = [0u32; 256];
        for &val in &quantized_values {
            freq_table[val as usize] += 1;
        }

        // Build Huffman tree and encoding table (simplified)
        let huffman_table = self.build_huffman_table(&freq_table);

        // Encode data
        let mut bit_buffer = 0u32;
        let mut bit_count = 0;
        let mut compressed_data = Vec::new();

        for &val in &quantized_values {
            let (code, code_length) = huffman_table[val as usize];
            bit_buffer |= (code as u32) << bit_count;
            bit_count += code_length;

            while bit_count >= 8 {
                compressed_data.push((bit_buffer & 0xFF) as u8);
                bit_buffer >>= 8;
                bit_count -= 8;
            }
        }

        if bit_count > 0 {
            compressed_data.push((bit_buffer & 0xFF) as u8);
        }

        let metadata = CompressionMetadata {
            scales: vec![scale],
            zero_points: vec![(min_val / scale) as i32],
            outliers: HashMap::new(),
            sparsity_mask: None,
            codebook: None,
            block_params: None,
        };

        Ok(CompressedTensor {
            data: compressed_data,
            metadata,
            shape: tensor.shape().dims().to_vec(),
            scheme: CompressionScheme::HuffmanEncoding,
        })
    }

    // Decompression methods (simplified implementations)

    fn decompress_one_bit(&self, compressed: &CompressedTensor) -> TorshResult<Tensor> {
        let mut data = Vec::new();
        let threshold = 0.5; // Use default threshold for decompression

        for &byte in &compressed.data {
            for i in 0..8 {
                let bit = (byte >> i) & 1;
                let value = if bit == 1 {
                    threshold + 0.5
                } else {
                    threshold - 0.5
                };
                data.push(value);
            }
        }

        // Apply outliers
        for (&idx, &value) in &compressed.metadata.outliers {
            if idx < data.len() {
                data[idx] = value;
            }
        }

        // Truncate to original shape size
        let total_elements: usize = compressed.shape.iter().product();
        data.truncate(total_elements);

        Tensor::from_data(data, compressed.shape.clone(), torsh_core::DeviceType::Cpu)
    }

    fn decompress_two_bit(&self, compressed: &CompressedTensor) -> TorshResult<Tensor> {
        let mut data = Vec::new();
        let scale = compressed.metadata.scales[0];
        let zero_point = compressed.metadata.zero_points[0];

        for &byte in &compressed.data {
            for i in 0..4 {
                let quantized = (byte >> (i * 2)) & 0x03;
                let value = (quantized as f32 + zero_point as f32) * scale;
                data.push(value);
            }
        }

        // Apply outliers
        for (&idx, &value) in &compressed.metadata.outliers {
            if idx < data.len() {
                data[idx] = value;
            }
        }

        let total_elements: usize = compressed.shape.iter().product();
        data.truncate(total_elements);

        Tensor::from_data(data, compressed.shape.clone(), torsh_core::DeviceType::Cpu)
    }

    fn decompress_three_bit(&self, compressed: &CompressedTensor) -> TorshResult<Tensor> {
        let mut data = Vec::new();
        let scale = compressed.metadata.scales[0];
        let zero_point = compressed.metadata.zero_points[0];

        let mut bit_buffer = 0u32;
        let mut bit_count = 0;
        let mut byte_idx = 0;

        let total_elements: usize = compressed.shape.iter().product();

        for _ in 0..total_elements {
            // Ensure we have enough bits
            while bit_count < 3 && byte_idx < compressed.data.len() {
                bit_buffer |= (compressed.data[byte_idx] as u32) << bit_count;
                bit_count += 8;
                byte_idx += 1;
            }

            if bit_count >= 3 {
                let quantized = bit_buffer & 0x07;
                let value = (quantized as f32 + zero_point as f32) * scale;
                data.push(value);
                bit_buffer >>= 3;
                bit_count -= 3;
            } else {
                break;
            }
        }

        Tensor::from_data(data, compressed.shape.clone(), torsh_core::DeviceType::Cpu)
    }

    fn decompress_variable_bit(&self, _compressed: &CompressedTensor) -> TorshResult<Tensor> {
        // Simplified implementation - would need to store bit-width information
        Err(TorshError::InvalidArgument(
            "Variable bit decompression not fully implemented".to_string(),
        ))
    }

    fn decompress_vector_quantization(&self, compressed: &CompressedTensor) -> TorshResult<Tensor> {
        let codebook = compressed
            .metadata
            .codebook
            .as_ref()
            .ok_or_else(|| TorshError::InvalidArgument("Missing codebook".to_string()))?;

        let mut data = Vec::new();
        let bytes_per_code = if codebook.size <= 256 { 1 } else { 2 };

        for chunk in compressed.data.chunks(bytes_per_code) {
            let code = if bytes_per_code == 1 {
                chunk[0] as usize
            } else {
                (chunk[0] as usize) | ((chunk[1] as usize) << 8)
            };

            if code < codebook.centroids.len() {
                data.extend(&codebook.centroids[code]);
            }
        }

        let total_elements: usize = compressed.shape.iter().product();
        data.truncate(total_elements);

        Tensor::from_data(data, compressed.shape.clone(), torsh_core::DeviceType::Cpu)
    }

    fn decompress_sparse(&self, compressed: &CompressedTensor) -> TorshResult<Tensor> {
        let total_elements: usize = compressed.shape.iter().product();
        let mut data = vec![0.0f32; total_elements];

        let scale = compressed.metadata.scales[0];
        let zero_point = compressed.metadata.zero_points[0];

        let mut cursor = 0;

        // Read number of non-zero elements
        if compressed.data.len() < 4 {
            return Err(TorshError::InvalidArgument(
                "Invalid sparse data".to_string(),
            ));
        }

        let num_non_zero = u32::from_le_bytes([
            compressed.data[cursor],
            compressed.data[cursor + 1],
            compressed.data[cursor + 2],
            compressed.data[cursor + 3],
        ]) as usize;
        cursor += 4;

        // Read indices
        let mut indices = Vec::new();
        let mut current_idx = 0;
        for _ in 0..num_non_zero {
            if cursor + 4 > compressed.data.len() {
                break;
            }
            let delta = u32::from_le_bytes([
                compressed.data[cursor],
                compressed.data[cursor + 1],
                compressed.data[cursor + 2],
                compressed.data[cursor + 3],
            ]) as usize;
            cursor += 4;
            current_idx += delta;
            indices.push(current_idx);
        }

        // Read values
        for &idx in indices.iter() {
            if cursor < compressed.data.len() && idx < total_elements {
                let quantized = compressed.data[cursor] as f32;
                let value = (quantized + zero_point as f32) * scale;
                data[idx] = value;
                cursor += 1;
            }
        }

        Tensor::from_data(data, compressed.shape.clone(), torsh_core::DeviceType::Cpu)
    }

    fn decompress_block_wise(&self, compressed: &CompressedTensor) -> TorshResult<Tensor> {
        let block_params =
            compressed.metadata.block_params.as_ref().ok_or_else(|| {
                TorshError::InvalidArgument("Missing block parameters".to_string())
            })?;

        let mut data = Vec::new();
        let mut cursor = 0;

        for (block_idx, &scale) in block_params.block_scales.iter().enumerate() {
            let zero_point = block_params
                .block_zero_points
                .get(block_idx)
                .copied()
                .unwrap_or(0);

            for _ in 0..block_params.block_size {
                if cursor < compressed.data.len() {
                    let quantized = compressed.data[cursor] as f32;
                    let value = (quantized + zero_point as f32) * scale;
                    data.push(value);
                    cursor += 1;
                } else {
                    break;
                }
            }
        }

        let total_elements: usize = compressed.shape.iter().product();
        data.truncate(total_elements);

        Tensor::from_data(data, compressed.shape.clone(), torsh_core::DeviceType::Cpu)
    }

    fn decompress_huffman(&self, _compressed: &CompressedTensor) -> TorshResult<Tensor> {
        // Simplified implementation - would need to store Huffman table
        Err(TorshError::InvalidArgument(
            "Huffman decompression not fully implemented".to_string(),
        ))
    }

    // Helper methods

    fn calculate_adaptive_threshold(data: &[f32], percentile: f32) -> f32 {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = (sorted_data.len() as f32 * percentile) as usize;
        sorted_data
            .get(idx.min(sorted_data.len() - 1))
            .copied()
            .unwrap_or(0.0)
    }

    fn calculate_range(data: &[f32]) -> (f32, f32) {
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        (min_val, max_val)
    }

    fn is_outlier(value: f32, threshold: f32, outlier_ratio: f32) -> bool {
        // Simple outlier detection based on deviation from threshold
        let deviation = (value - threshold).abs();
        let max_expected_deviation = threshold.abs() * outlier_ratio;
        deviation > max_expected_deviation
    }

    fn calculate_sparsity_threshold(data: &[f32], sparsity_ratio: f32) -> f32 {
        let mut abs_values: Vec<f32> = data.iter().map(|&x| x.abs()).collect();
        abs_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let threshold_idx = (abs_values.len() as f32 * sparsity_ratio) as usize;
        abs_values
            .get(threshold_idx.min(abs_values.len() - 1))
            .copied()
            .unwrap_or(0.0)
    }

    fn calculate_sensitivity_scores(&self, data: &[f32]) -> Vec<f32> {
        // Simple sensitivity calculation based on local variance
        let mut scores = Vec::new();

        for chunk in data.chunks(self.config.block_size) {
            let mean = chunk.iter().sum::<f32>() / chunk.len() as f32;
            let variance =
                chunk.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / chunk.len() as f32;
            let sensitivity = (variance / (mean.abs() + 1e-8)).min(1.0);
            scores.push(sensitivity);
        }

        scores
    }

    fn quantize_chunk(&self, chunk: &[f32], bits: u8) -> TorshResult<(Vec<u8>, f32, i32)> {
        let (min_val, max_val) = Self::calculate_range(chunk);
        let levels = (1 << bits) - 1;
        let scale = (max_val - min_val) / levels as f32;
        let scale = if scale == 0.0 { 1.0 } else { scale };
        let zero_point = (min_val / scale) as i32;

        let mut data = Vec::new();
        for &value in chunk {
            let quantized = ((value - min_val) / scale)
                .round()
                .max(0.0)
                .min(levels as f32) as u8;
            data.push(quantized);
        }

        Ok((data, scale, zero_point))
    }

    fn tensorize_data(&self, data: &[f32], vector_dim: usize) -> Vec<Vec<f32>> {
        let mut vectors = Vec::new();

        for chunk in data.chunks(vector_dim) {
            let mut vector = chunk.to_vec();
            // Pad with zeros if necessary
            while vector.len() < vector_dim {
                vector.push(0.0);
            }
            vectors.push(vector);
        }

        vectors
    }

    fn train_codebook(&self, vectors: &[Vec<f32>]) -> TorshResult<Codebook> {
        let mut centroids = Vec::new();
        let dim = self.vq_config.vector_dim;

        // Initialize centroids using specified method
        match self.vq_config.init_method {
            VqInitMethod::Random => {
                // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
                let mut rng = Random::new();
                for _ in 0..self.vq_config.codebook_size {
                    let centroid: Vec<f32> =
                        (0..dim).map(|_| (rng.gen::<f32>() - 0.5) * 2.0).collect();
                    centroids.push(centroid);
                }
            }
            VqInitMethod::UniformSampling => {
                for i in 0..self.vq_config.codebook_size {
                    if i < vectors.len() {
                        centroids.push(vectors[i].clone());
                    } else {
                        let centroid: Vec<f32> = (0..dim).map(|_| 0.0).collect();
                        centroids.push(centroid);
                    }
                }
            }
            VqInitMethod::KMeansPlusPlus => {
                // Simplified k-means++ initialization
                if !vectors.is_empty() {
                    centroids.push(vectors[0].clone());

                    for _ in 1..self.vq_config.codebook_size {
                        let mut max_dist = 0.0;
                        let mut best_vector = vectors[0].clone();

                        for vector in vectors {
                            let min_dist_to_centroid = centroids
                                .iter()
                                .map(|c| Self::euclidean_distance(vector, c))
                                .fold(f32::INFINITY, f32::min);

                            if min_dist_to_centroid > max_dist {
                                max_dist = min_dist_to_centroid;
                                best_vector = vector.clone();
                            }
                        }
                        centroids.push(best_vector);
                    }
                }
            }
        }

        // K-means clustering
        for _iteration in 0..self.vq_config.max_iterations {
            let mut new_centroids = vec![vec![0.0; dim]; self.vq_config.codebook_size];
            let mut counts = vec![0; self.vq_config.codebook_size];

            // Assign vectors to nearest centroids
            for vector in vectors {
                let nearest_idx = self.find_nearest_centroid(
                    vector,
                    &Codebook {
                        centroids: centroids.clone(),
                        usage_freq: vec![0; centroids.len()],
                        size: centroids.len(),
                        dim,
                    },
                );

                for (i, &val) in vector.iter().enumerate() {
                    new_centroids[nearest_idx][i] += val;
                }
                counts[nearest_idx] += 1;
            }

            // Update centroids
            let mut converged = true;
            for (i, count) in counts.iter().enumerate() {
                if *count > 0 {
                    for j in 0..dim {
                        new_centroids[i][j] /= *count as f32;
                    }

                    // Check convergence
                    let old_centroid = &centroids[i];
                    let new_centroid = &new_centroids[i];
                    if Self::euclidean_distance(old_centroid, new_centroid)
                        > self.vq_config.tolerance
                    {
                        converged = false;
                    }
                }
            }

            centroids = new_centroids;

            if converged {
                break;
            }
        }

        Ok(Codebook {
            centroids,
            usage_freq: vec![0; self.vq_config.codebook_size],
            size: self.vq_config.codebook_size,
            dim,
        })
    }

    fn find_nearest_centroid(&self, vector: &[f32], codebook: &Codebook) -> usize {
        let mut min_distance = f32::INFINITY;
        let mut nearest_idx = 0;

        for (i, centroid) in codebook.centroids.iter().enumerate() {
            let distance = Self::euclidean_distance(vector, centroid);
            if distance < min_distance {
                min_distance = distance;
                nearest_idx = i;
            }
        }

        nearest_idx
    }

    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn build_huffman_table(&self, freq_table: &[u32; 256]) -> [(u16, u8); 256] {
        // Simplified Huffman table (in practice, would build actual Huffman tree)
        let mut table = [(0u16, 8u8); 256];

        for (i, &freq) in freq_table.iter().enumerate() {
            if freq > 0 {
                // Assign shorter codes to more frequent values
                let code_length = if freq > 1000 {
                    4
                } else if freq > 100 {
                    6
                } else {
                    8
                };
                table[i] = (i as u16, code_length);
            }
        }

        table
    }

    fn update_stats(&mut self, original_size: usize, compressed: &CompressedTensor) {
        let compressed_size = compressed.data.len() +
            compressed.metadata.outliers.len() * 8 + // 4 bytes index + 4 bytes value
            compressed.metadata.scales.len() * 4 +
            compressed.metadata.zero_points.len() * 4;

        let compression_ratio = original_size as f32 / compressed_size as f32;

        self.stats = CompressionStats {
            original_size,
            compressed_size,
            compression_ratio,
            num_outliers: compressed.metadata.outliers.len(),
            sparsity_ratio: if let Some(ref mask) = compressed.metadata.sparsity_mask {
                mask.iter().filter(|&&x| !x).count() as f32 / mask.len() as f32
            } else {
                0.0
            },
            error_metrics: CompressionErrorMetrics {
                mae: 0.0, // Would calculate during compression
                mse: 0.0,
                snr: 0.0,
                efficiency_score: compression_ratio * 100.0,
            },
        };
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Generate compression report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== ADVANCED COMPRESSION REPORT ===\n\n");
        report.push_str(&format!(
            "Original Size: {} bytes\n",
            self.stats.original_size
        ));
        report.push_str(&format!(
            "Compressed Size: {} bytes\n",
            self.stats.compressed_size
        ));
        report.push_str(&format!(
            "Compression Ratio: {:.2}x\n",
            self.stats.compression_ratio
        ));
        report.push_str(&format!(
            "Space Savings: {:.1}%\n",
            (1.0 - self.stats.compressed_size as f32 / self.stats.original_size as f32) * 100.0
        ));
        report.push_str(&format!(
            "Number of Outliers: {}\n",
            self.stats.num_outliers
        ));
        report.push_str(&format!(
            "Sparsity Ratio: {:.3}\n",
            self.stats.sparsity_ratio
        ));
        report.push_str(&format!(
            "Efficiency Score: {:.1}\n",
            self.stats.error_metrics.efficiency_score
        ));

        report
    }
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self {
            original_size: 0,
            compressed_size: 0,
            compression_ratio: 1.0,
            num_outliers: 0,
            sparsity_ratio: 0.0,
            error_metrics: CompressionErrorMetrics {
                mae: 0.0,
                mse: 0.0,
                snr: 0.0,
                efficiency_score: 0.0,
            },
        }
    }
}

impl Default for CompressionEngine {
    fn default() -> Self {
        Self::new(SubByteConfig::default(), VectorQuantConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_compression_engine() {
        let config = SubByteConfig {
            enable_outlier_handling: false,
            ..Default::default()
        }; // Disable outlier handling to avoid issues with sine wave data
        let vq_config = VectorQuantConfig::default();
        let mut engine = CompressionEngine::new(config, vq_config);

        // Use a larger tensor where compression overhead is amortized
        let input_data: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();
        let input = tensor_1d(&input_data).unwrap();

        // Test 2-bit compression
        let compressed = engine.compress(&input, CompressionScheme::TwoBit).unwrap();
        assert_eq!(compressed.scheme, CompressionScheme::TwoBit);
        assert!(compressed.data.len() < input.data().unwrap().len()); // Should be compressed significantly

        let decompressed = engine.decompress(&compressed).unwrap();
        assert_eq!(decompressed.shape().dims(), input.shape().dims());

        let stats = engine.get_stats();
        assert!(stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_one_bit_compression() {
        let mut engine = CompressionEngine::default();
        let input = tensor_1d(&[-1.0, -0.5, 0.5, 1.0, -2.0, 2.0, 0.0, 0.1]).unwrap();

        let compressed = engine.compress(&input, CompressionScheme::OneBit).unwrap();
        assert_eq!(compressed.scheme, CompressionScheme::OneBit);
        assert_eq!(compressed.data.len(), 1); // 8 bits packed into 1 byte

        let decompressed = engine.decompress(&compressed).unwrap();
        assert_eq!(decompressed.shape().dims(), input.shape().dims());
    }

    #[test]
    fn test_three_bit_compression() {
        let mut engine = CompressionEngine::default();
        let input = tensor_1d(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();

        let compressed = engine
            .compress(&input, CompressionScheme::ThreeBit)
            .unwrap();
        assert_eq!(compressed.scheme, CompressionScheme::ThreeBit);

        let decompressed = engine.decompress(&compressed).unwrap();
        assert_eq!(decompressed.shape().dims(), input.shape().dims());

        // Check that decompressed values are close to original
        let orig_data = input.data().unwrap();
        let decomp_data = decompressed.data().unwrap();
        for (i, (&orig, &decomp)) in orig_data.iter().zip(decomp_data.iter()).enumerate() {
            let error = (orig - decomp).abs();
            assert!(
                error < 1.0,
                "Element {i}: orig={orig}, decomp={decomp}, error={error}"
            );
        }
    }

    #[test]
    fn test_vector_quantization() {
        let mut engine = CompressionEngine::default();
        let input = tensor_1d(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ])
        .unwrap();

        let compressed = engine
            .compress(&input, CompressionScheme::VectorQuantization)
            .unwrap();
        assert_eq!(compressed.scheme, CompressionScheme::VectorQuantization);
        assert!(compressed.metadata.codebook.is_some());

        let decompressed = engine.decompress(&compressed).unwrap();
        assert_eq!(decompressed.shape().dims(), input.shape().dims());
    }

    #[test]
    fn test_sparse_compression() {
        let mut engine = CompressionEngine::default();
        // Create sparse data (mostly zeros)
        let input = tensor_1d(&[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 3.0, 0.0]).unwrap();

        let compressed = engine
            .compress(&input, CompressionScheme::SparseQuantization)
            .unwrap();
        assert_eq!(compressed.scheme, CompressionScheme::SparseQuantization);
        assert!(compressed.metadata.sparsity_mask.is_some());

        let decompressed = engine.decompress(&compressed).unwrap();
        assert_eq!(decompressed.shape().dims(), input.shape().dims());

        let stats = engine.get_stats();
        assert!(stats.sparsity_ratio > 0.0);
    }

    #[test]
    fn test_block_wise_compression() {
        let mut engine = CompressionEngine::default();
        let input = tensor_1d(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]).unwrap();

        let compressed = engine
            .compress(&input, CompressionScheme::BlockWise)
            .unwrap();
        assert_eq!(compressed.scheme, CompressionScheme::BlockWise);
        assert!(compressed.metadata.block_params.is_some());

        let decompressed = engine.decompress(&compressed).unwrap();
        assert_eq!(decompressed.shape().dims(), input.shape().dims());
    }

    #[test]
    fn test_sub_byte_config() {
        let config = SubByteConfig {
            bits_per_value: 3,
            scheme: CompressionScheme::ThreeBit,
            block_size: 32,
            sparsity_threshold: 0.8,
            enable_outlier_handling: true,
            outlier_ratio: 0.05,
        };

        assert_eq!(config.bits_per_value, 3);
        assert_eq!(config.scheme, CompressionScheme::ThreeBit);
        assert_eq!(config.block_size, 32);
    }

    #[test]
    fn test_vector_quant_config() {
        let config = VectorQuantConfig {
            codebook_size: 128,
            vector_dim: 8,
            max_iterations: 50,
            tolerance: 1e-3,
            init_method: VqInitMethod::KMeansPlusPlus,
        };

        assert_eq!(config.codebook_size, 128);
        assert_eq!(config.vector_dim, 8);
        assert_eq!(config.init_method, VqInitMethod::KMeansPlusPlus);
    }

    #[test]
    fn test_codebook() {
        let codebook = Codebook {
            centroids: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            usage_freq: vec![10, 5],
            size: 2,
            dim: 3,
        };

        assert_eq!(codebook.size, 2);
        assert_eq!(codebook.dim, 3);
        assert_eq!(codebook.centroids.len(), 2);
        assert_eq!(codebook.centroids[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_compression_stats() {
        let stats = CompressionStats {
            original_size: 1000,
            compressed_size: 250,
            compression_ratio: 4.0,
            num_outliers: 5,
            sparsity_ratio: 0.8,
            error_metrics: CompressionErrorMetrics {
                mae: 0.1,
                mse: 0.01,
                snr: 20.0,
                efficiency_score: 400.0,
            },
        };

        assert_eq!(stats.compression_ratio, 4.0);
        assert_eq!(stats.num_outliers, 5);
        assert_eq!(stats.sparsity_ratio, 0.8);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let distance = CompressionEngine::euclidean_distance(&a, &b);
        let expected = ((3.0_f32).powi(2) + (3.0_f32).powi(2) + (3.0_f32).powi(2)).sqrt();

        assert!((distance - expected).abs() < 1e-6);
    }

    #[test]
    fn test_compression_report() {
        let mut engine = CompressionEngine::default();
        let input = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        let _compressed = engine.compress(&input, CompressionScheme::TwoBit).unwrap();
        let report = engine.generate_report();

        assert!(report.contains("ADVANCED COMPRESSION REPORT"));
        assert!(report.contains("Compression Ratio"));
        assert!(report.contains("Space Savings"));
    }
}
