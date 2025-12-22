//! Gradient Compression for Distributed Training
//!
//! This module implements various gradient compression techniques to reduce
//! communication overhead in distributed training. Includes quantization,
//! sparsification, and error feedback mechanisms.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_tensor::Tensor;
use tracing::{debug, info};

/// Gradient compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression method to use
    pub method: CompressionMethod,
    /// Compression ratio (0.0 to 1.0)
    pub compression_ratio: f32,
    /// Whether to use error feedback
    pub error_feedback: bool,
    /// Momentum for error feedback
    pub error_feedback_momentum: f32,
    /// Whether to use memory-efficient compression
    pub memory_efficient: bool,
    /// Warmup steps before applying compression
    pub warmup_steps: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            method: CompressionMethod::TopK { k: 0.1 },
            compression_ratio: 0.1,
            error_feedback: true,
            error_feedback_momentum: 0.9,
            memory_efficient: true,
            warmup_steps: 100,
        }
    }
}

/// Supported compression methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompressionMethod {
    /// Top-K sparsification (keep top k% of gradients)
    TopK { k: f32 },
    /// Random sparsification
    RandomK { k: f32 },
    /// Threshold-based sparsification
    Threshold { threshold: f32 },
    /// Quantization to specific number of bits
    Quantization { bits: u8 },
    /// Sign-based compression (SignSGD)
    SignSGD,
    /// Gradient sketching using Count-Sketch
    Sketching { sketch_size: usize },
    /// PowerSGD low-rank approximation
    PowerSGD { rank: usize },
    /// Ternary quantization (-1, 0, +1)
    TernaryQuant { threshold: f32 },
    /// Bimodal quantization (adaptive binning)
    BimodalQuant { num_bins: usize },
    /// Natural compression (based on gradient distribution)
    NaturalCompression { compression_factor: f32 },
    /// Layerwise adaptive compression
    LayerwiseAdaptive { base_ratio: f32, sensitivity: f32 },
    /// EF21 compression with momentum
    EF21 {
        compression_ratio: f32,
        momentum: f32,
    },
    /// No compression (baseline)
    None,
}

/// Compressed gradient representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedGradient {
    /// Compression method used
    pub method: CompressionMethod,
    /// Compressed data
    pub data: CompressedData,
    /// Original gradient shape
    pub original_shape: Vec<usize>,
    /// Metadata for decompression
    pub metadata: CompressionMetadata,
}

/// Compressed data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressedData {
    /// Sparse representation with indices and values
    Sparse {
        indices: Vec<usize>,
        values: Vec<f32>,
    },
    /// Quantized values
    Quantized {
        values: Vec<u8>,
        scale: f32,
        zero_point: u8,
    },
    /// Sign representation
    Signs { signs: Vec<bool>, norm: f32 },
    /// Low-rank factors
    LowRank {
        left_factor: Vec<f32>,
        right_factor: Vec<f32>,
        rank: usize,
    },
    /// Sketch representation
    Sketch {
        sketch: Vec<f32>,
        hash_a: Vec<u32>,
        hash_b: Vec<u32>,
    },
    /// Ternary representation (-1, 0, +1)
    Ternary { values: Vec<i8>, scale: f32 },
    /// Bimodal quantization bins
    Bimodal {
        bin_indices: Vec<u8>,
        bin_centers: Vec<f32>,
    },
    /// Natural compression (frequency-based)
    Natural {
        values: Vec<f32>,
        frequencies: Vec<u32>,
        codebook: Vec<f32>,
    },
    /// EF21 compressed representation
    EF21 {
        compressed_values: Vec<f32>,
        error_feedback: Vec<f32>,
    },
}

/// Metadata for compression/decompression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetadata {
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Error norm introduced by compression
    pub error_norm: f32,
    /// Original gradient norm
    pub original_norm: f32,
    /// Timestamp
    pub timestamp: u64,
}

/// Gradient compressor
pub struct GradientCompressor {
    /// Configuration
    config: CompressionConfig,
    /// Error feedback buffers
    error_buffers: HashMap<String, Tensor>,
    /// Step counter
    step_count: usize,
    /// Compression statistics
    stats: CompressionStats,
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Total number of compressions performed
    pub total_compressions: u64,
    /// Average compression ratio
    pub avg_compression_ratio: f64,
    /// Total communication reduction (bytes)
    pub total_communication_reduction: u64,
    /// Average error norm introduced
    pub avg_error_norm: f64,
    /// Time spent on compression (ms)
    pub compression_time_ms: f64,
}

impl GradientCompressor {
    /// Create a new gradient compressor
    pub fn new(config: CompressionConfig) -> Self {
        info!(
            "Initializing gradient compressor with method: {:?}",
            config.method
        );

        Self {
            config,
            error_buffers: HashMap::new(),
            step_count: 0,
            stats: CompressionStats::default(),
        }
    }

    /// Compress a gradient tensor
    pub fn compress(
        &mut self,
        gradient: &Tensor,
        param_name: &str,
    ) -> TorshResult<CompressedGradient> {
        let start_time = std::time::Instant::now();

        // Skip compression during warmup
        if self.step_count < self.config.warmup_steps {
            return self.no_compression(gradient, param_name);
        }

        // Apply error feedback if enabled
        let adjusted_gradient = if self.config.error_feedback {
            self.apply_error_feedback(gradient, param_name)?
        } else {
            gradient.clone()
        };

        let compressed = match &self.config.method {
            CompressionMethod::TopK { k } => self.compress_top_k(&adjusted_gradient, *k)?,
            CompressionMethod::RandomK { k } => self.compress_random_k(&adjusted_gradient, *k)?,
            CompressionMethod::Threshold { threshold } => {
                self.compress_threshold(&adjusted_gradient, *threshold)?
            }
            CompressionMethod::Quantization { bits } => {
                self.compress_quantization(&adjusted_gradient, *bits)?
            }
            CompressionMethod::SignSGD => self.compress_sign_sgd(&adjusted_gradient)?,
            CompressionMethod::Sketching { sketch_size } => {
                self.compress_sketching(&adjusted_gradient, *sketch_size)?
            }
            CompressionMethod::PowerSGD { rank } => {
                self.compress_power_sgd(&adjusted_gradient, *rank)?
            }
            CompressionMethod::TernaryQuant { threshold } => {
                self.compress_ternary(&adjusted_gradient, *threshold)?
            }
            CompressionMethod::BimodalQuant { num_bins } => {
                self.compress_bimodal(&adjusted_gradient, *num_bins)?
            }
            CompressionMethod::NaturalCompression { compression_factor } => {
                self.compress_natural(&adjusted_gradient, *compression_factor)?
            }
            CompressionMethod::LayerwiseAdaptive {
                base_ratio,
                sensitivity,
            } => self.compress_layerwise_adaptive(
                &adjusted_gradient,
                *base_ratio,
                *sensitivity,
                param_name,
            )?,
            CompressionMethod::EF21 {
                compression_ratio,
                momentum,
            } => self.compress_ef21(
                &adjusted_gradient,
                *compression_ratio,
                *momentum,
                param_name,
            )?,
            CompressionMethod::None => return self.no_compression(gradient, param_name),
        };

        // Store compression error for error feedback
        if self.config.error_feedback {
            self.update_error_feedback(&compressed, gradient, param_name)?;
        }

        // Update statistics
        let compression_time = start_time.elapsed().as_millis() as f64;
        self.update_stats(&compressed, compression_time);

        self.step_count += 1;
        Ok(compressed)
    }

    /// Decompress a gradient
    pub fn decompress(&self, compressed: &CompressedGradient) -> TorshResult<Tensor> {
        match &compressed.data {
            CompressedData::Sparse { indices, values } => {
                self.decompress_sparse(indices, values, &compressed.original_shape)
            }
            CompressedData::Quantized {
                values,
                scale,
                zero_point,
            } => self.decompress_quantized(values, *scale, *zero_point, &compressed.original_shape),
            CompressedData::Signs { signs, norm } => {
                self.decompress_sign_sgd(signs, *norm, &compressed.original_shape)
            }
            CompressedData::LowRank {
                left_factor,
                right_factor,
                rank,
            } => self.decompress_power_sgd(
                left_factor,
                right_factor,
                *rank,
                &compressed.original_shape,
            ),
            CompressedData::Sketch {
                sketch,
                hash_a,
                hash_b,
            } => self.decompress_sketching(sketch, hash_a, hash_b, &compressed.original_shape),
            CompressedData::Ternary { values, scale } => {
                self.decompress_ternary(values, *scale, &compressed.original_shape)
            }
            CompressedData::Bimodal {
                bin_indices,
                bin_centers,
            } => self.decompress_bimodal(bin_indices, bin_centers, &compressed.original_shape),
            CompressedData::Natural {
                values,
                frequencies: _,
                codebook,
            } => self.decompress_natural(values, codebook, &compressed.original_shape),
            CompressedData::EF21 {
                compressed_values,
                error_feedback: _,
            } => self.decompress_ef21(compressed_values, &compressed.original_shape),
        }
    }

    /// Apply error feedback
    fn apply_error_feedback(&mut self, gradient: &Tensor, param_name: &str) -> TorshResult<Tensor> {
        if let Some(error_buffer) = self.error_buffers.get(param_name) {
            // adjusted_gradient = gradient + momentum * error_buffer
            let scaled_error = error_buffer.mul_scalar(self.config.error_feedback_momentum)?;
            Ok(gradient.add(&scaled_error)?)
        } else {
            Ok(gradient.clone())
        }
    }

    /// Update error feedback buffer
    fn update_error_feedback(
        &mut self,
        compressed: &CompressedGradient,
        original: &Tensor,
        param_name: &str,
    ) -> TorshResult<()> {
        let decompressed = self.decompress(compressed)?;
        let error = original.sub(&decompressed)?;
        self.error_buffers.insert(param_name.to_string(), error);
        Ok(())
    }

    /// Top-K sparsification
    fn compress_top_k(&self, gradient: &Tensor, k: f32) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let numel = flat_grad.numel();
        let k_elements = ((numel as f32) * k).ceil() as usize;

        // Get absolute values for sorting
        let abs_grad = flat_grad.abs()?;
        let grad_data = flat_grad.to_vec()?;
        let abs_data = abs_grad.to_vec()?;

        // Find top-k indices
        let mut indexed_values: Vec<(usize, f32)> =
            abs_data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut indices = Vec::new();
        let mut values = Vec::new();

        for &(idx, _) in indexed_values.iter().take(k_elements) {
            indices.push(idx);
            values.push(grad_data[idx]);
        }

        debug!("Top-K compression: kept {}/{} elements", k_elements, numel);

        let original_norm = gradient.norm()?.item()?;
        let compression_ratio = k;

        Ok(CompressedGradient {
            method: CompressionMethod::TopK { k },
            data: CompressedData::Sparse { indices, values },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: CompressionMetadata {
                compression_ratio,
                error_norm: 0.0, // Would calculate actual error
                original_norm,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Random-K sparsification
    fn compress_random_k(&self, gradient: &Tensor, k: f32) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let numel = flat_grad.numel();
        let k_elements = ((numel as f32) * k).ceil() as usize;

        let grad_data = flat_grad.to_vec()?;

        // Random sampling of indices
        let mut indices = Vec::new();
        let mut values = Vec::new();

        // Simple deterministic "random" selection for reproducibility
        let step = numel / k_elements.max(1);
        for i in (0..numel).step_by(step).take(k_elements) {
            indices.push(i);
            values.push(grad_data[i]);
        }

        debug!(
            "Random-K compression: kept {}/{} elements",
            k_elements, numel
        );

        let original_norm = gradient.norm()?.item()?;

        Ok(CompressedGradient {
            method: CompressionMethod::RandomK { k },
            data: CompressedData::Sparse { indices, values },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: CompressionMetadata {
                compression_ratio: k,
                error_norm: 0.0,
                original_norm,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Threshold-based sparsification
    fn compress_threshold(
        &self,
        gradient: &Tensor,
        threshold: f32,
    ) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;

        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &value) in grad_data.iter().enumerate() {
            if value.abs() >= threshold {
                indices.push(i);
                values.push(value);
            }
        }

        let compression_ratio = indices.len() as f32 / grad_data.len() as f32;
        debug!(
            "Threshold compression: kept {}/{} elements",
            indices.len(),
            grad_data.len()
        );

        let original_norm = gradient.norm()?.item()?;

        Ok(CompressedGradient {
            method: CompressionMethod::Threshold { threshold },
            data: CompressedData::Sparse { indices, values },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: CompressionMetadata {
                compression_ratio,
                error_norm: 0.0,
                original_norm,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Quantization compression
    fn compress_quantization(
        &self,
        gradient: &Tensor,
        bits: u8,
    ) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;

        // Simple uniform quantization
        let min_val = grad_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = grad_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let levels = (1 << bits) - 1;
        let scale = (max_val - min_val) / levels as f32;
        let zero_point = (-min_val / scale).round() as u8;

        let mut quantized_values = Vec::new();
        for &value in &grad_data {
            let quantized = ((value / scale) + zero_point as f32)
                .round()
                .clamp(0.0, levels as f32) as u8;
            quantized_values.push(quantized);
        }

        debug!("Quantization: {} bits, {} levels", bits, levels);

        let original_norm = gradient.norm()?.item()?;
        let compression_ratio = (bits as f32) / 32.0; // Assuming original is fp32

        Ok(CompressedGradient {
            method: CompressionMethod::Quantization { bits },
            data: CompressedData::Quantized {
                values: quantized_values,
                scale,
                zero_point,
            },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: CompressionMetadata {
                compression_ratio,
                error_norm: 0.0,
                original_norm,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Sign SGD compression
    fn compress_sign_sgd(&self, gradient: &Tensor) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;
        let norm = gradient.norm()?.item()?;

        let signs: Vec<bool> = grad_data.iter().map(|&x| x >= 0.0).collect();

        debug!(
            "SignSGD compression: {} elements -> {} bits",
            grad_data.len(),
            signs.len()
        );

        Ok(CompressedGradient {
            method: CompressionMethod::SignSGD,
            data: CompressedData::Signs { signs, norm },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: CompressionMetadata {
                compression_ratio: 1.0 / 32.0, // 1 bit vs 32 bits
                error_norm: 0.0,
                original_norm: norm,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Sketching compression (simplified)
    fn compress_sketching(
        &self,
        gradient: &Tensor,
        sketch_size: usize,
    ) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;

        // Simple sketch: just take first sketch_size elements
        let sketch: Vec<f32> = grad_data.iter().take(sketch_size).copied().collect();

        // Mock hash functions
        let hash_a: Vec<u32> = (0..grad_data.len()).map(|i| (i * 17 + 23) as u32).collect();
        let hash_b: Vec<u32> = (0..grad_data.len()).map(|i| (i * 37 + 41) as u32).collect();

        let compression_ratio = sketch_size as f32 / grad_data.len() as f32;
        let original_norm = gradient.norm()?.item()?;

        debug!(
            "Sketching compression: {} -> {} elements",
            grad_data.len(),
            sketch_size
        );

        Ok(CompressedGradient {
            method: CompressionMethod::Sketching { sketch_size },
            data: CompressedData::Sketch {
                sketch,
                hash_a,
                hash_b,
            },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: CompressionMetadata {
                compression_ratio,
                error_norm: 0.0,
                original_norm,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// PowerSGD compression (simplified)
    fn compress_power_sgd(
        &self,
        gradient: &Tensor,
        rank: usize,
    ) -> TorshResult<CompressedGradient> {
        let shape_obj = gradient.shape();
        let shape = shape_obj.dims();
        if shape.len() != 2 {
            return Err(TorshDistributedError::invalid_argument(
                "gradient",
                format!("PowerSGD requires 2D tensors, got {}D tensor", shape.len()),
                "2D tensor with shape [rows, cols]",
            ));
        }

        let rows = shape[0];
        let cols = shape[1];

        // Mock low-rank factorization: A ≈ P @ Q^T
        let left_factor_size = rows * rank;
        let right_factor_size = cols * rank;

        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;

        // Simplified: just take portions of the gradient as factors
        let left_factor: Vec<f32> = grad_data.iter().take(left_factor_size).copied().collect();
        let right_factor: Vec<f32> = grad_data
            .iter()
            .skip(left_factor_size)
            .take(right_factor_size)
            .copied()
            .collect();

        let compression_ratio =
            (left_factor_size + right_factor_size) as f32 / grad_data.len() as f32;
        let original_norm = gradient.norm()?.item()?;

        debug!(
            "PowerSGD compression: rank {}, ratio {:.3}",
            rank, compression_ratio
        );

        Ok(CompressedGradient {
            method: CompressionMethod::PowerSGD { rank },
            data: CompressedData::LowRank {
                left_factor,
                right_factor,
                rank,
            },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: CompressionMetadata {
                compression_ratio,
                error_norm: 0.0,
                original_norm,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Ternary quantization compression
    fn compress_ternary(
        &self,
        gradient: &Tensor,
        threshold: f32,
    ) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;
        let original_norm = gradient.norm()?.item()?;

        // Compute scaling factor based on gradient magnitude
        let scale = original_norm / (grad_data.len() as f32).sqrt();

        let mut ternary_values = Vec::new();
        for &value in &grad_data {
            let normalized = value / scale;
            let ternary = if normalized > threshold {
                1i8
            } else if normalized < -threshold {
                -1i8
            } else {
                0i8
            };
            ternary_values.push(ternary);
        }

        let compression_ratio = 2.0 / 32.0; // ~2 bits per value vs 32 bits
        debug!(
            "Ternary compression: threshold {}, scale {:.6}",
            threshold, scale
        );

        Ok(CompressedGradient {
            method: CompressionMethod::TernaryQuant { threshold },
            data: CompressedData::Ternary {
                values: ternary_values,
                scale,
            },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: CompressionMetadata {
                compression_ratio,
                error_norm: 0.0,
                original_norm,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Bimodal quantization compression
    fn compress_bimodal(
        &self,
        gradient: &Tensor,
        num_bins: usize,
    ) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;
        let original_norm = gradient.norm()?.item()?;

        // Find min and max values for binning
        let min_val = grad_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = grad_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Create bin centers
        let mut bin_centers = Vec::new();
        for i in 0..num_bins {
            let center = min_val + (max_val - min_val) * (i as f32 + 0.5) / (num_bins as f32);
            bin_centers.push(center);
        }

        // Assign each value to nearest bin
        let mut bin_indices = Vec::new();
        for &value in &grad_data {
            let mut best_bin = 0;
            let mut best_distance = f32::INFINITY;

            for (bin_idx, &center) in bin_centers.iter().enumerate() {
                let distance = (value - center).abs();
                if distance < best_distance {
                    best_distance = distance;
                    best_bin = bin_idx;
                }
            }
            bin_indices.push(best_bin as u8);
        }

        let bits_per_bin = (num_bins as f32).log2().ceil();
        let compression_ratio = bits_per_bin / 32.0;
        debug!(
            "Bimodal compression: {} bins, {:.1} bits/value",
            num_bins, bits_per_bin
        );

        Ok(CompressedGradient {
            method: CompressionMethod::BimodalQuant { num_bins },
            data: CompressedData::Bimodal {
                bin_indices,
                bin_centers,
            },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: CompressionMetadata {
                compression_ratio,
                error_norm: 0.0,
                original_norm,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Natural compression based on gradient distribution
    fn compress_natural(
        &self,
        gradient: &Tensor,
        compression_factor: f32,
    ) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;
        let original_norm = gradient.norm()?.item()?;

        // Create frequency histogram for natural encoding
        let num_unique = (grad_data.len() as f32 * compression_factor).ceil() as usize;
        let mut value_counts: std::collections::HashMap<i32, u32> =
            std::collections::HashMap::new();

        // Quantize values for frequency counting
        let scale = 10000.0; // Fixed point scale
        for &value in &grad_data {
            let quantized = (value * scale).round() as i32;
            *value_counts.entry(quantized).or_insert(0) += 1;
        }

        // Get most frequent values
        let mut sorted_values: Vec<_> = value_counts.into_iter().collect();
        sorted_values.sort_by(|a, b| b.1.cmp(&a.1));
        sorted_values.truncate(num_unique);

        // Create codebook and compressed representation
        let codebook: Vec<f32> = sorted_values
            .iter()
            .map(|(v, _)| *v as f32 / scale)
            .collect();
        let frequencies: Vec<u32> = sorted_values.iter().map(|(_, f)| *f).collect();

        // Encode values using codebook
        let mut compressed_values = Vec::new();
        for &value in &grad_data {
            // Find closest codebook entry
            let mut best_idx = 0;
            let mut best_distance = f32::INFINITY;
            for (idx, &codebook_val) in codebook.iter().enumerate() {
                let distance = (value - codebook_val).abs();
                if distance < best_distance {
                    best_distance = distance;
                    best_idx = idx;
                }
            }
            compressed_values.push(best_idx as f32);
        }

        debug!(
            "Natural compression: {} unique values from {} total",
            num_unique,
            grad_data.len()
        );

        Ok(CompressedGradient {
            method: CompressionMethod::NaturalCompression { compression_factor },
            data: CompressedData::Natural {
                values: compressed_values,
                frequencies,
                codebook,
            },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: CompressionMetadata {
                compression_ratio: compression_factor,
                error_norm: 0.0,
                original_norm,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Layerwise adaptive compression
    fn compress_layerwise_adaptive(
        &self,
        gradient: &Tensor,
        base_ratio: f32,
        sensitivity: f32,
        param_name: &str,
    ) -> TorshResult<CompressedGradient> {
        let _original_norm = gradient.norm()?.item();

        // Adapt compression ratio based on layer sensitivity
        let layer_sensitivity = if param_name.contains("weight") {
            1.0
        } else {
            sensitivity
        };
        let adapted_ratio = base_ratio * layer_sensitivity;

        // Use TopK with adapted ratio
        self.compress_top_k(gradient, adapted_ratio)
    }

    /// EF21 compression with error feedback and momentum
    fn compress_ef21(
        &mut self,
        gradient: &Tensor,
        compression_ratio: f32,
        momentum: f32,
        param_name: &str,
    ) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;
        let original_norm = gradient.norm()?.item()?;

        // Get or create error feedback buffer
        let error_key = format!("ef21_{}", param_name);
        let error_feedback = if let Some(prev_error) = self.error_buffers.get(&error_key) {
            prev_error.flatten()?.to_vec()?
        } else {
            vec![0.0; grad_data.len()]
        };

        // Apply momentum to error feedback
        let mut adjusted_grad = Vec::new();
        for (&grad_val, &error_val) in grad_data.iter().zip(error_feedback.iter()) {
            adjusted_grad.push(grad_val + momentum * error_val);
        }

        // Compress using TopK
        let k_elements = (grad_data.len() as f32 * compression_ratio).ceil() as usize;
        let mut indexed_values: Vec<(usize, f32)> = adjusted_grad
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut compressed_values = vec![0.0; grad_data.len()];
        let mut new_error_feedback = adjusted_grad.clone();

        // Keep top-k values
        for &(idx, _) in indexed_values.iter().take(k_elements) {
            compressed_values[idx] = adjusted_grad[idx];
            new_error_feedback[idx] = 0.0; // Reset error for transmitted values
        }

        // Update error feedback buffer
        let error_tensor = Tensor::from_vec(new_error_feedback.clone(), gradient.shape().dims())?;
        self.error_buffers.insert(error_key, error_tensor);

        debug!(
            "EF21 compression: kept {}/{} elements with momentum {}",
            k_elements,
            grad_data.len(),
            momentum
        );

        Ok(CompressedGradient {
            method: CompressionMethod::EF21 {
                compression_ratio,
                momentum,
            },
            data: CompressedData::EF21 {
                compressed_values,
                error_feedback: new_error_feedback,
            },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: CompressionMetadata {
                compression_ratio,
                error_norm: 0.0,
                original_norm,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// No compression (passthrough)
    fn no_compression(
        &self,
        gradient: &Tensor,
        _param_name: &str,
    ) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;
        let indices: Vec<usize> = (0..grad_data.len()).collect();

        let original_norm = gradient.norm()?.item()?;

        Ok(CompressedGradient {
            method: CompressionMethod::None,
            data: CompressedData::Sparse {
                indices,
                values: grad_data,
            },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: CompressionMetadata {
                compression_ratio: 1.0,
                error_norm: 0.0,
                original_norm,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Decompress sparse representation
    fn decompress_sparse(
        &self,
        indices: &[usize],
        values: &[f32],
        shape: &[usize],
    ) -> TorshResult<Tensor> {
        let total_elements: usize = shape.iter().product();
        let mut data = vec![0.0; total_elements];

        for (&idx, &val) in indices.iter().zip(values.iter()) {
            if idx < total_elements {
                data[idx] = val;
            }
        }

        Ok(Tensor::from_vec(data, shape)?)
    }

    /// Decompress quantized representation
    fn decompress_quantized(
        &self,
        values: &[u8],
        scale: f32,
        zero_point: u8,
        shape: &[usize],
    ) -> TorshResult<Tensor> {
        let data: Vec<f32> = values
            .iter()
            .map(|&q| (q as f32 - zero_point as f32) * scale)
            .collect();

        Ok(Tensor::from_vec(data, shape)?)
    }

    /// Decompress SignSGD representation
    fn decompress_sign_sgd(
        &self,
        signs: &[bool],
        norm: f32,
        shape: &[usize],
    ) -> TorshResult<Tensor> {
        let total_elements: usize = shape.iter().product();
        let magnitude = norm / (total_elements as f32).sqrt();

        let data: Vec<f32> = signs
            .iter()
            .map(|&sign| if sign { magnitude } else { -magnitude })
            .collect();

        Ok(Tensor::from_vec(data, shape)?)
    }

    /// Decompress PowerSGD representation (simplified)
    fn decompress_power_sgd(
        &self,
        left_factor: &[f32],
        right_factor: &[f32],
        _rank: usize,
        shape: &[usize],
    ) -> TorshResult<Tensor> {
        // Simplified: just combine the factors somehow
        let total_elements: usize = shape.iter().product();
        let mut data = vec![0.0; total_elements];

        let left_len = left_factor.len();
        let right_len = right_factor.len();

        for i in 0..total_elements.min(left_len + right_len) {
            if i < left_len {
                data[i] = left_factor[i];
            } else {
                data[i] = right_factor[i - left_len];
            }
        }

        Ok(Tensor::from_vec(data, shape)?)
    }

    /// Decompress sketching representation (simplified)
    fn decompress_sketching(
        &self,
        sketch: &[f32],
        _hash_a: &[u32],
        _hash_b: &[u32],
        shape: &[usize],
    ) -> TorshResult<Tensor> {
        let total_elements: usize = shape.iter().product();
        let mut data = vec![0.0; total_elements];

        // Simplified: just spread sketch values
        for (i, &val) in sketch.iter().enumerate() {
            if i < total_elements {
                data[i] = val;
            }
        }

        Ok(Tensor::from_vec(data, shape)?)
    }

    /// Decompress ternary representation
    fn decompress_ternary(
        &self,
        values: &[i8],
        scale: f32,
        shape: &[usize],
    ) -> TorshResult<Tensor> {
        let data: Vec<f32> = values
            .iter()
            .map(|&ternary| (ternary as f32) * scale)
            .collect();

        Ok(Tensor::from_vec(data, shape)?)
    }

    /// Decompress bimodal representation
    fn decompress_bimodal(
        &self,
        bin_indices: &[u8],
        bin_centers: &[f32],
        shape: &[usize],
    ) -> TorshResult<Tensor> {
        let data: Vec<f32> = bin_indices
            .iter()
            .map(|&bin_idx| bin_centers.get(bin_idx as usize).copied().unwrap_or(0.0))
            .collect();

        Ok(Tensor::from_vec(data, shape)?)
    }

    /// Decompress natural representation
    fn decompress_natural(
        &self,
        values: &[f32],
        codebook: &[f32],
        shape: &[usize],
    ) -> TorshResult<Tensor> {
        let data: Vec<f32> = values
            .iter()
            .map(|&idx| {
                let idx_usize = idx as usize;
                codebook.get(idx_usize).copied().unwrap_or(0.0)
            })
            .collect();

        Ok(Tensor::from_vec(data, shape)?)
    }

    /// Decompress EF21 representation
    fn decompress_ef21(&self, compressed_values: &[f32], shape: &[usize]) -> TorshResult<Tensor> {
        // For EF21, the compressed values are already in the correct format
        Ok(Tensor::from_vec(compressed_values.to_vec(), shape)?)
    }

    /// Update compression statistics
    fn update_stats(&mut self, compressed: &CompressedGradient, compression_time: f64) {
        self.stats.total_compressions += 1;
        self.stats.avg_compression_ratio = (self.stats.avg_compression_ratio
            * (self.stats.total_compressions - 1) as f64
            + compressed.metadata.compression_ratio as f64)
            / self.stats.total_compressions as f64;
        self.stats.compression_time_ms += compression_time;
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Reset error feedback buffers
    pub fn reset_error_feedback(&mut self) {
        self.error_buffers.clear();
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_config() {
        let config = CompressionConfig::default();
        assert_eq!(config.compression_ratio, 0.1);
        assert!(config.error_feedback);
        assert_eq!(config.warmup_steps, 100);
    }

    #[test]
    fn test_compression_methods() {
        assert_ne!(
            CompressionMethod::TopK { k: 0.1 },
            CompressionMethod::SignSGD
        );
        assert_ne!(
            CompressionMethod::Quantization { bits: 8 },
            CompressionMethod::None
        );
    }

    #[tokio::test]
    async fn test_gradient_compressor_creation() {
        let config = CompressionConfig::default();
        let compressor = GradientCompressor::new(config);

        assert_eq!(compressor.step_count(), 0);
        assert_eq!(compressor.get_stats().total_compressions, 0);
    }

    #[tokio::test]
    async fn test_top_k_compression() -> TorshResult<()> {
        let config = CompressionConfig {
            method: CompressionMethod::TopK { k: 0.5 },
            warmup_steps: 0,
            ..Default::default()
        };
        let mut compressor = GradientCompressor::new(config);

        let gradient = torsh_tensor::creation::randn(&[10, 10])?;
        let compressed = compressor.compress(&gradient, "test_param")?;

        match &compressed.data {
            CompressedData::Sparse { indices, values } => {
                assert_eq!(indices.len(), values.len());
                assert!(indices.len() <= 50); // Top 50% of 100 elements
            }
            _ => panic!("Expected sparse compression for TopK"),
        }

        let decompressed = compressor.decompress(&compressed)?;
        assert_eq!(decompressed.shape().dims(), gradient.shape().dims());

        Ok(())
    }

    #[tokio::test]
    async fn test_sign_sgd_compression() -> TorshResult<()> {
        let config = CompressionConfig {
            method: CompressionMethod::SignSGD,
            warmup_steps: 0,
            ..Default::default()
        };
        let mut compressor = GradientCompressor::new(config);

        let gradient = torsh_tensor::creation::randn(&[5, 5])?;
        let compressed = compressor.compress(&gradient, "test_param")?;

        match &compressed.data {
            CompressedData::Signs { signs, norm } => {
                assert_eq!(signs.len(), 25); // 5x5 = 25 elements
                assert!(*norm > 0.0);
            }
            _ => panic!("Expected sign compression for SignSGD"),
        }

        let decompressed = compressor.decompress(&compressed)?;
        assert_eq!(decompressed.shape().dims(), gradient.shape().dims());

        Ok(())
    }

    #[tokio::test]
    async fn test_quantization_compression() -> TorshResult<()> {
        let config = CompressionConfig {
            method: CompressionMethod::Quantization { bits: 8 },
            warmup_steps: 0,
            ..Default::default()
        };
        let mut compressor = GradientCompressor::new(config);

        let gradient = torsh_tensor::creation::randn(&[4, 4])?;
        let compressed = compressor.compress(&gradient, "test_param")?;

        match &compressed.data {
            CompressedData::Quantized {
                values,
                scale,
                zero_point: _,
            } => {
                assert_eq!(values.len(), 16); // 4x4 = 16 elements
                assert!(*scale > 0.0);
                // zero_point is u8, always <= 255
            }
            _ => panic!("Expected quantized compression"),
        }

        let decompressed = compressor.decompress(&compressed)?;
        assert_eq!(decompressed.shape().dims(), gradient.shape().dims());

        Ok(())
    }

    #[tokio::test]
    async fn test_no_compression() -> TorshResult<()> {
        let config = CompressionConfig {
            method: CompressionMethod::None,
            warmup_steps: 0,
            ..Default::default()
        };
        let mut compressor = GradientCompressor::new(config);

        let gradient = torsh_tensor::creation::randn(&[3, 3])?;
        let compressed = compressor.compress(&gradient, "test_param")?;

        assert_eq!(compressed.metadata.compression_ratio, 1.0);

        let decompressed = compressor.decompress(&compressed)?;
        assert_eq!(decompressed.shape().dims(), gradient.shape().dims());

        Ok(())
    }

    #[test]
    fn test_compression_stats() {
        let stats = CompressionStats {
            total_compressions: 100,
            avg_compression_ratio: 0.25,
            total_communication_reduction: 1024 * 1024, // 1MB
            avg_error_norm: 0.01,
            compression_time_ms: 250.5,
        };

        assert_eq!(stats.total_compressions, 100);
        assert_eq!(stats.avg_compression_ratio, 0.25);
        assert_eq!(stats.total_communication_reduction, 1024 * 1024);
    }
}
