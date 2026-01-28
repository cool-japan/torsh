//! Advanced gradient compression techniques for memory-limited environments
//!
//! This module provides sophisticated compression algorithms for gradients that can
//! significantly reduce memory usage and communication overhead in distributed training.

use parking_lot::Mutex;
use scirs2_core::numeric::{FromPrimitive, ToPrimitive};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use torsh_core::dtype::FloatElement;
use torsh_core::error::{Result, TorshError};

/// Gradient compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression algorithm to use
    pub algorithm: CompressionAlgorithm,
    /// Target compression ratio (0.0 to 1.0)
    pub target_ratio: f64,
    /// Error tolerance for lossy compression
    pub error_tolerance: f64,
    /// Memory budget in bytes
    pub memory_budget: usize,
    /// Whether to use adaptive compression
    pub adaptive: bool,
    /// Minimum sparsity threshold for sparse compression
    pub sparsity_threshold: f64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Quantization8Bit,
            target_ratio: 0.25, // 4x compression
            error_tolerance: 1e-4,
            memory_budget: 100 * 1024 * 1024, // 100MB
            adaptive: true,
            sparsity_threshold: 0.01, // 1% sparsity threshold
        }
    }
}

/// Available compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// 8-bit quantization
    Quantization8Bit,
    /// 4-bit quantization
    Quantization4Bit,
    /// 2-bit quantization
    Quantization2Bit,
    /// 1-bit quantization (sign only)
    Quantization1Bit,
    /// Top-K sparsification
    TopKSparsification,
    /// Random sparsification
    RandomSparsification,
    /// Gradient sketching with random projections
    GradientSketching,
    /// PowerSGD low-rank compression
    PowerSGD,
    /// Error feedback compression
    ErrorFeedback,
    /// Adaptive compression (chooses best algorithm)
    Adaptive,
}

/// Compressed gradient representation
#[derive(Debug, Clone)]
pub struct CompressedGradient {
    /// Original shape of the gradient
    pub original_shape: Vec<usize>,
    /// Compressed data
    pub data: Vec<u8>,
    /// Compression metadata
    pub metadata: CompressionMetadata,
    /// Algorithm used for compression
    pub algorithm: CompressionAlgorithm,
}

/// Compression metadata
#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    /// Scale factor for quantization
    pub scale: f64,
    /// Zero point for quantization
    pub zero_point: i32,
    /// Indices for sparse compression
    pub indices: Vec<usize>,
    /// Random seed for reproducible compression
    pub seed: u64,
    /// Rank for low-rank compression
    pub rank: usize,
    /// Error compensation
    pub error: Vec<f64>,
}

impl Default for CompressionMetadata {
    fn default() -> Self {
        Self {
            scale: 1.0,
            zero_point: 0,
            indices: Vec::new(),
            seed: 0,
            rank: 0,
            error: Vec::new(),
        }
    }
}

/// Gradient compressor with advanced algorithms
#[derive(Clone)]
pub struct GradientCompressor<T: FloatElement> {
    /// Compression configuration
    config: CompressionConfig,
    /// Error feedback buffer for each parameter
    error_feedback: Arc<RwLock<HashMap<String, Vec<T>>>>,
    /// Compression statistics
    stats: Arc<RwLock<CompressionStats>>,
    /// Random number generator state
    rng_state: Arc<Mutex<u64>>,
}

impl<T: FloatElement> std::fmt::Debug for GradientCompressor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientCompressor")
            .field("config", &self.config)
            .field(
                "error_feedback_size",
                &self
                    .error_feedback
                    .read()
                    .expect("lock should not be poisoned")
                    .len(),
            )
            .field(
                "stats",
                &self.stats.read().expect("lock should not be poisoned"),
            )
            .finish()
    }
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Total number of compressions
    pub total_compressions: usize,
    /// Total bytes before compression
    pub total_bytes_original: usize,
    /// Total bytes after compression
    pub total_bytes_compressed: usize,
    /// Average compression ratio
    pub avg_compression_ratio: f64,
    /// Total compression time
    pub total_compression_time_ms: u64,
    /// Total decompression time
    pub total_decompression_time_ms: u64,
    /// Compression error (if applicable)
    pub compression_error: f64,
}

impl<T: FloatElement + FromPrimitive + ToPrimitive> GradientCompressor<T> {
    /// Create a new gradient compressor
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            error_feedback: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CompressionStats::default())),
            rng_state: Arc::new(Mutex::new(42)), // Fixed seed for reproducibility
        }
    }

    /// Compress gradients according to the configured algorithm
    pub fn compress(
        &mut self,
        gradients: &[T],
        parameter_name: &str,
    ) -> Result<CompressedGradient> {
        let start_time = std::time::Instant::now();

        let algorithm = if self.config.adaptive {
            self.choose_best_algorithm(gradients)?
        } else {
            self.config.algorithm
        };

        let compressed = match algorithm {
            CompressionAlgorithm::None => self.compress_none(gradients)?,
            CompressionAlgorithm::Quantization8Bit => self.compress_quantization_8bit(gradients)?,
            CompressionAlgorithm::Quantization4Bit => self.compress_quantization_4bit(gradients)?,
            CompressionAlgorithm::Quantization2Bit => self.compress_quantization_2bit(gradients)?,
            CompressionAlgorithm::Quantization1Bit => self.compress_quantization_1bit(gradients)?,
            CompressionAlgorithm::TopKSparsification => {
                self.compress_top_k_sparsification(gradients)?
            }
            CompressionAlgorithm::RandomSparsification => {
                self.compress_random_sparsification(gradients)?
            }
            CompressionAlgorithm::GradientSketching => {
                self.compress_gradient_sketching(gradients)?
            }
            CompressionAlgorithm::PowerSGD => self.compress_power_sgd(gradients)?,
            CompressionAlgorithm::ErrorFeedback => {
                self.compress_error_feedback(gradients, parameter_name)?
            }
            CompressionAlgorithm::Adaptive => {
                return Err(TorshError::AutogradError(
                    "Adaptive algorithm should have been resolved".to_string(),
                ));
            }
        };

        // Update statistics
        let compression_time = start_time.elapsed().as_millis() as u64;
        let mut stats = self.stats.write().expect("lock should not be poisoned");
        stats.total_compressions += 1;
        stats.total_bytes_original += std::mem::size_of_val(gradients);
        stats.total_bytes_compressed += compressed.data.len();
        stats.total_compression_time_ms += compression_time;

        let compression_ratio =
            compressed.data.len() as f64 / std::mem::size_of_val(gradients) as f64;
        stats.avg_compression_ratio = (stats.avg_compression_ratio
            * (stats.total_compressions - 1) as f64
            + compression_ratio)
            / stats.total_compressions as f64;

        Ok(compressed)
    }

    /// Decompress gradients
    pub fn decompress(&mut self, compressed: &CompressedGradient) -> Result<Vec<T>> {
        let start_time = std::time::Instant::now();

        let decompressed = match compressed.algorithm {
            CompressionAlgorithm::None => self.decompress_none(compressed)?,
            CompressionAlgorithm::Quantization8Bit => {
                self.decompress_quantization_8bit(compressed)?
            }
            CompressionAlgorithm::Quantization4Bit => {
                self.decompress_quantization_4bit(compressed)?
            }
            CompressionAlgorithm::Quantization2Bit => {
                self.decompress_quantization_2bit(compressed)?
            }
            CompressionAlgorithm::Quantization1Bit => {
                self.decompress_quantization_1bit(compressed)?
            }
            CompressionAlgorithm::TopKSparsification => {
                self.decompress_top_k_sparsification(compressed)?
            }
            CompressionAlgorithm::RandomSparsification => {
                self.decompress_random_sparsification(compressed)?
            }
            CompressionAlgorithm::GradientSketching => {
                self.decompress_gradient_sketching(compressed)?
            }
            CompressionAlgorithm::PowerSGD => self.decompress_power_sgd(compressed)?,
            CompressionAlgorithm::ErrorFeedback => self.decompress_error_feedback(compressed)?,
            CompressionAlgorithm::Adaptive => {
                return Err(TorshError::AutogradError(
                    "Cannot decompress adaptive algorithm directly".to_string(),
                ));
            }
        };

        // Update decompression statistics
        let decompression_time = start_time.elapsed().as_millis() as u64;
        self.stats
            .write()
            .expect("lock should not be poisoned")
            .total_decompression_time_ms += decompression_time;

        Ok(decompressed)
    }

    /// Choose the best compression algorithm based on gradient characteristics
    fn choose_best_algorithm(&self, gradients: &[T]) -> Result<CompressionAlgorithm> {
        // Analyze gradient characteristics
        let sparsity = self.calculate_sparsity(gradients);
        let variance = self.calculate_variance(gradients);
        let magnitude = self.calculate_magnitude(gradients);

        // Choose algorithm based on characteristics
        if sparsity > self.config.sparsity_threshold {
            Ok(CompressionAlgorithm::TopKSparsification)
        } else if variance < 0.01 && magnitude < 1.0 {
            Ok(CompressionAlgorithm::Quantization2Bit)
        } else if variance < 0.1 {
            Ok(CompressionAlgorithm::Quantization4Bit)
        } else if gradients.len() > 10000 {
            Ok(CompressionAlgorithm::PowerSGD)
        } else {
            Ok(CompressionAlgorithm::Quantization8Bit)
        }
    }

    /// Calculate sparsity (fraction of near-zero elements)
    fn calculate_sparsity(&self, gradients: &[T]) -> f64 {
        let threshold = <T as torsh_core::dtype::TensorElement>::from_f64(1e-8)
            .expect("f64 conversion should succeed");
        let near_zero_count = gradients.iter().filter(|&&x| x.abs() < threshold).count();
        near_zero_count as f64 / gradients.len() as f64
    }

    /// Calculate variance of gradients
    fn calculate_variance(&self, gradients: &[T]) -> f64 {
        if gradients.is_empty() {
            return 0.0;
        }

        let mean = gradients
            .iter()
            .map(|x| ToPrimitive::to_f64(x).expect("f64 conversion should succeed"))
            .sum::<f64>()
            / gradients.len() as f64;

        let variance = gradients
            .iter()
            .map(|x| {
                let val = ToPrimitive::to_f64(x).expect("f64 conversion should succeed");
                (val - mean).powi(2)
            })
            .sum::<f64>()
            / gradients.len() as f64;

        variance
    }

    /// Calculate magnitude (RMS) of gradients
    fn calculate_magnitude(&self, gradients: &[T]) -> f64 {
        if gradients.is_empty() {
            return 0.0;
        }

        let sum_squares = gradients
            .iter()
            .map(|x| {
                ToPrimitive::to_f64(x)
                    .expect("f64 conversion should succeed")
                    .powi(2)
            })
            .sum::<f64>();

        (sum_squares / gradients.len() as f64).sqrt()
    }

    /// No compression (passthrough)
    fn compress_none(&self, gradients: &[T]) -> Result<CompressedGradient> {
        let data = unsafe {
            std::slice::from_raw_parts(
                gradients.as_ptr() as *const u8,
                std::mem::size_of_val(gradients),
            )
            .to_vec()
        };

        Ok(CompressedGradient {
            original_shape: vec![gradients.len()],
            data,
            metadata: CompressionMetadata::default(),
            algorithm: CompressionAlgorithm::None,
        })
    }

    /// 8-bit quantization compression
    fn compress_quantization_8bit(&self, gradients: &[T]) -> Result<CompressedGradient> {
        // Find min and max values
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &grad in gradients {
            let val = ToPrimitive::to_f64(&grad).expect("f64 conversion should succeed");
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        // Calculate scale and zero point
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round() as i32;

        // Quantize gradients
        let mut quantized = Vec::with_capacity(gradients.len());
        for &grad in gradients {
            let val = ToPrimitive::to_f64(&grad).expect("f64 conversion should succeed");
            let quantized_val = ((val / scale) + zero_point as f64).round() as u8;
            quantized.push(quantized_val);
        }

        let metadata = CompressionMetadata {
            scale,
            zero_point,
            ..Default::default()
        };

        Ok(CompressedGradient {
            original_shape: vec![gradients.len()],
            data: quantized,
            metadata,
            algorithm: CompressionAlgorithm::Quantization8Bit,
        })
    }

    /// 4-bit quantization compression
    fn compress_quantization_4bit(&self, gradients: &[T]) -> Result<CompressedGradient> {
        // Similar to 8-bit but with 4-bit precision
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &grad in gradients {
            let val = ToPrimitive::to_f64(&grad).expect("f64 conversion should succeed");
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        let scale = (max_val - min_val) / 15.0; // 4-bit has 16 levels (0-15)
        let zero_point = (-min_val / scale).round() as i32;

        // Pack two 4-bit values into each byte
        let mut quantized = Vec::with_capacity(gradients.len().div_ceil(2));
        for chunk in gradients.chunks(2) {
            let first = if !chunk.is_empty() {
                let val = ToPrimitive::to_f64(&chunk[0]).expect("f64 conversion should succeed");
                ((val / scale) + zero_point as f64).round().clamp(0.0, 15.0) as u8
            } else {
                0
            };

            let second = if chunk.len() > 1 {
                let val = ToPrimitive::to_f64(&chunk[1]).expect("f64 conversion should succeed");
                ((val / scale) + zero_point as f64).round().clamp(0.0, 15.0) as u8
            } else {
                0
            };

            quantized.push((first << 4) | second);
        }

        let metadata = CompressionMetadata {
            scale,
            zero_point,
            ..Default::default()
        };

        Ok(CompressedGradient {
            original_shape: vec![gradients.len()],
            data: quantized,
            metadata,
            algorithm: CompressionAlgorithm::Quantization4Bit,
        })
    }

    /// 2-bit quantization compression
    fn compress_quantization_2bit(&self, gradients: &[T]) -> Result<CompressedGradient> {
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &grad in gradients {
            let val = ToPrimitive::to_f64(&grad).expect("f64 conversion should succeed");
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        let scale = (max_val - min_val) / 3.0; // 2-bit has 4 levels (0-3)
        let zero_point = (-min_val / scale).round() as i32;

        // Pack four 2-bit values into each byte
        let mut quantized = Vec::with_capacity(gradients.len().div_ceil(4));
        for chunk in gradients.chunks(4) {
            let mut byte_val = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                let val_f64 = ToPrimitive::to_f64(&val).expect("f64 conversion should succeed");
                let quantized_val = ((val_f64 / scale) + zero_point as f64)
                    .round()
                    .clamp(0.0, 3.0) as u8;
                byte_val |= quantized_val << (i * 2);
            }
            quantized.push(byte_val);
        }

        let metadata = CompressionMetadata {
            scale,
            zero_point,
            ..Default::default()
        };

        Ok(CompressedGradient {
            original_shape: vec![gradients.len()],
            data: quantized,
            metadata,
            algorithm: CompressionAlgorithm::Quantization2Bit,
        })
    }

    /// 1-bit quantization (sign only)
    fn compress_quantization_1bit(&self, gradients: &[T]) -> Result<CompressedGradient> {
        // Calculate magnitude for scaling
        let magnitude = self.calculate_magnitude(gradients);

        // Pack eight 1-bit values into each byte
        let mut quantized = Vec::with_capacity(gradients.len().div_ceil(8));
        for chunk in gradients.chunks(8) {
            let mut byte_val = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                let val_f64 = ToPrimitive::to_f64(&val).expect("f64 conversion should succeed");
                if val_f64 >= 0.0 {
                    byte_val |= 1 << i;
                }
            }
            quantized.push(byte_val);
        }

        let metadata = CompressionMetadata {
            scale: magnitude,
            ..Default::default()
        };

        Ok(CompressedGradient {
            original_shape: vec![gradients.len()],
            data: quantized,
            metadata,
            algorithm: CompressionAlgorithm::Quantization1Bit,
        })
    }

    /// Top-K sparsification
    fn compress_top_k_sparsification(&self, gradients: &[T]) -> Result<CompressedGradient> {
        let k = (gradients.len() as f64 * (1.0 - self.config.sparsity_threshold)).round() as usize;

        // Create (value, index) pairs and sort by absolute value
        let mut value_index_pairs: Vec<(f64, usize)> = gradients
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                (
                    ToPrimitive::to_f64(&val)
                        .expect("f64 conversion should succeed")
                        .abs(),
                    i,
                )
            })
            .collect();

        value_index_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Keep only top-k values
        let top_k_indices: Vec<usize> = value_index_pairs
            .iter()
            .take(k)
            .map(|(_, idx)| *idx)
            .collect();

        // Store compressed values and indices
        let mut compressed_data = Vec::new();
        let mut indices = Vec::new();

        for &idx in &top_k_indices {
            let val = ToPrimitive::to_f64(&gradients[idx]).expect("f64 conversion should succeed");
            let bytes = val.to_le_bytes();
            compressed_data.extend_from_slice(&bytes);
            indices.push(idx);
        }

        let metadata = CompressionMetadata {
            indices,
            ..Default::default()
        };

        Ok(CompressedGradient {
            original_shape: vec![gradients.len()],
            data: compressed_data,
            metadata,
            algorithm: CompressionAlgorithm::TopKSparsification,
        })
    }

    /// Random sparsification
    fn compress_random_sparsification(&self, gradients: &[T]) -> Result<CompressedGradient> {
        let keep_prob = 1.0 - self.config.sparsity_threshold;
        let mut rng_state = self.rng_state.lock();

        let mut compressed_data = Vec::new();
        let mut indices = Vec::new();

        for (i, &val) in gradients.iter().enumerate() {
            // Simple linear congruential generator
            *rng_state = (1103515245_u64.wrapping_mul(*rng_state).wrapping_add(12345)) % (1 << 31);
            let random_val = *rng_state as f64 / (1u64 << 31) as f64;

            if random_val < keep_prob {
                let val_f64 = ToPrimitive::to_f64(&val).expect("f64 conversion should succeed");
                let bytes = val_f64.to_le_bytes();
                compressed_data.extend_from_slice(&bytes);
                indices.push(i);
            }
        }

        let metadata = CompressionMetadata {
            indices,
            seed: *rng_state,
            ..Default::default()
        };

        Ok(CompressedGradient {
            original_shape: vec![gradients.len()],
            data: compressed_data,
            metadata,
            algorithm: CompressionAlgorithm::RandomSparsification,
        })
    }

    /// Gradient sketching compression
    fn compress_gradient_sketching(&self, gradients: &[T]) -> Result<CompressedGradient> {
        // Reduce dimensionality using random projections
        let sketch_size = (gradients.len() as f64 * self.config.target_ratio).round() as usize;
        let sketch_size = sketch_size.max(1);

        let mut rng_state = self.rng_state.lock();
        let mut sketch = vec![0.0; sketch_size];

        for &grad in gradients {
            let val = ToPrimitive::to_f64(&grad).expect("f64 conversion should succeed");

            // Apply random projections
            for j in 0..sketch_size {
                *rng_state =
                    (1103515245_u64.wrapping_mul(*rng_state).wrapping_add(12345)) % (1 << 31);
                let random_sign = if (*rng_state % 2) == 0 { 1.0 } else { -1.0 };
                sketch[j] += val * random_sign;
            }
        }

        // Convert sketch to bytes
        let mut compressed_data = Vec::new();
        for &val in &sketch {
            let bytes = val.to_le_bytes();
            compressed_data.extend_from_slice(&bytes);
        }

        let metadata = CompressionMetadata {
            seed: *rng_state,
            rank: sketch_size,
            ..Default::default()
        };

        Ok(CompressedGradient {
            original_shape: vec![gradients.len()],
            data: compressed_data,
            metadata,
            algorithm: CompressionAlgorithm::GradientSketching,
        })
    }

    /// PowerSGD low-rank compression
    fn compress_power_sgd(&self, gradients: &[T]) -> Result<CompressedGradient> {
        // For simplicity, we'll use a basic low-rank approximation
        let rank = (gradients.len() as f64 * self.config.target_ratio)
            .sqrt()
            .round() as usize;
        let rank = rank.max(1).min(gradients.len());

        // Create a simplified low-rank representation
        // In a full implementation, this would use SVD or power iteration
        let mut compressed_data = Vec::new();

        // Store first 'rank' values as the low-rank approximation
        for i in 0..rank.min(gradients.len()) {
            let val = ToPrimitive::to_f64(&gradients[i]).expect("f64 conversion should succeed");
            let bytes = val.to_le_bytes();
            compressed_data.extend_from_slice(&bytes);
        }

        let metadata = CompressionMetadata {
            rank,
            ..Default::default()
        };

        Ok(CompressedGradient {
            original_shape: vec![gradients.len()],
            data: compressed_data,
            metadata,
            algorithm: CompressionAlgorithm::PowerSGD,
        })
    }

    /// Error feedback compression
    fn compress_error_feedback(
        &mut self,
        gradients: &[T],
        parameter_name: &str,
    ) -> Result<CompressedGradient> {
        // Get or create error feedback buffer
        let mut error_feedback = self
            .error_feedback
            .write()
            .expect("lock should not be poisoned");
        let error_buffer = error_feedback
            .entry(parameter_name.to_string())
            .or_insert_with(|| {
                vec![<T as torsh_core::dtype::TensorElement>::zero(); gradients.len()]
            });

        // Ensure buffer has correct size
        if error_buffer.len() != gradients.len() {
            error_buffer.resize(
                gradients.len(),
                <T as torsh_core::dtype::TensorElement>::zero(),
            );
        }

        // Add error feedback to gradients
        let mut compensated_gradients = Vec::with_capacity(gradients.len());
        for (&grad, &error) in gradients.iter().zip(error_buffer.iter()) {
            compensated_gradients.push(grad + error);
        }

        // Compress using quantization
        let compressed = self.compress_quantization_8bit(&compensated_gradients)?;

        // Calculate new error
        let decompressed = self.decompress_quantization_8bit(&compressed)?;
        for (i, (&original, &decompressed_val)) in compensated_gradients
            .iter()
            .zip(decompressed.iter())
            .enumerate()
        {
            let error = original - decompressed_val;
            error_buffer[i] = error;
        }

        Ok(CompressedGradient {
            algorithm: CompressionAlgorithm::ErrorFeedback,
            ..compressed
        })
    }

    // Decompression methods (implementations for each algorithm)
    fn decompress_none(&self, compressed: &CompressedGradient) -> Result<Vec<T>> {
        let gradients = unsafe {
            std::slice::from_raw_parts(
                compressed.data.as_ptr() as *const T,
                compressed.data.len() / std::mem::size_of::<T>(),
            )
            .to_vec()
        };
        Ok(gradients)
    }

    fn decompress_quantization_8bit(&self, compressed: &CompressedGradient) -> Result<Vec<T>> {
        let scale = compressed.metadata.scale;
        let zero_point = compressed.metadata.zero_point;

        let mut gradients = Vec::with_capacity(compressed.data.len());
        for &quantized_val in &compressed.data {
            let dequantized = (quantized_val as f64 - zero_point as f64) * scale;
            gradients.push(
                <T as torsh_core::dtype::TensorElement>::from_f64(dequantized)
                    .expect("f64 conversion should succeed"),
            );
        }

        Ok(gradients)
    }

    fn decompress_quantization_4bit(&self, compressed: &CompressedGradient) -> Result<Vec<T>> {
        let scale = compressed.metadata.scale;
        let zero_point = compressed.metadata.zero_point;
        let original_size = compressed.original_shape[0];

        let mut gradients = Vec::with_capacity(original_size);
        for &byte_val in &compressed.data {
            // Extract first 4-bit value
            let first = (byte_val >> 4) & 0x0F;
            let dequantized_first = (first as f64 - zero_point as f64) * scale;
            gradients.push(
                <T as torsh_core::dtype::TensorElement>::from_f64(dequantized_first)
                    .expect("f64 conversion should succeed"),
            );

            if gradients.len() < original_size {
                // Extract second 4-bit value
                let second = byte_val & 0x0F;
                let dequantized_second = (second as f64 - zero_point as f64) * scale;
                gradients.push(
                    <T as torsh_core::dtype::TensorElement>::from_f64(dequantized_second)
                        .expect("f64 conversion should succeed"),
                );
            }
        }

        gradients.truncate(original_size);
        Ok(gradients)
    }

    fn decompress_quantization_2bit(&self, compressed: &CompressedGradient) -> Result<Vec<T>> {
        let scale = compressed.metadata.scale;
        let zero_point = compressed.metadata.zero_point;
        let original_size = compressed.original_shape[0];

        let mut gradients = Vec::with_capacity(original_size);
        for &byte_val in &compressed.data {
            for i in 0..4 {
                if gradients.len() >= original_size {
                    break;
                }
                let quantized_val = (byte_val >> (i * 2)) & 0x03;
                let dequantized = (quantized_val as f64 - zero_point as f64) * scale;
                gradients.push(
                    <T as torsh_core::dtype::TensorElement>::from_f64(dequantized)
                        .expect("f64 conversion should succeed"),
                );
            }
        }

        gradients.truncate(original_size);
        Ok(gradients)
    }

    fn decompress_quantization_1bit(&self, compressed: &CompressedGradient) -> Result<Vec<T>> {
        let magnitude = compressed.metadata.scale;
        let original_size = compressed.original_shape[0];

        let mut gradients = Vec::with_capacity(original_size);
        for &byte_val in &compressed.data {
            for i in 0..8 {
                if gradients.len() >= original_size {
                    break;
                }
                let sign_bit = (byte_val >> i) & 1;
                let value = if sign_bit == 1 { magnitude } else { -magnitude };
                gradients.push(
                    <T as torsh_core::dtype::TensorElement>::from_f64(value)
                        .expect("f64 conversion should succeed"),
                );
            }
        }

        gradients.truncate(original_size);
        Ok(gradients)
    }

    fn decompress_top_k_sparsification(&self, compressed: &CompressedGradient) -> Result<Vec<T>> {
        let original_size = compressed.original_shape[0];
        let mut gradients = vec![<T as torsh_core::dtype::TensorElement>::zero(); original_size];

        let values_per_element = std::mem::size_of::<f64>();
        let num_values = compressed.data.len() / values_per_element;

        for (i, &idx) in compressed
            .metadata
            .indices
            .iter()
            .take(num_values)
            .enumerate()
        {
            let start = i * values_per_element;
            let end = start + values_per_element;
            if end <= compressed.data.len() && idx < original_size {
                let bytes = &compressed.data[start..end];
                let value = f64::from_le_bytes(bytes.try_into().unwrap());
                gradients[idx] = <T as torsh_core::dtype::TensorElement>::from_f64(value)
                    .expect("f64 conversion should succeed");
            }
        }

        Ok(gradients)
    }

    fn decompress_random_sparsification(&self, compressed: &CompressedGradient) -> Result<Vec<T>> {
        let original_size = compressed.original_shape[0];
        let mut gradients = vec![<T as torsh_core::dtype::TensorElement>::zero(); original_size];

        let values_per_element = std::mem::size_of::<f64>();
        let num_values = compressed.data.len() / values_per_element;

        for (i, &idx) in compressed
            .metadata
            .indices
            .iter()
            .take(num_values)
            .enumerate()
        {
            let start = i * values_per_element;
            let end = start + values_per_element;
            if end <= compressed.data.len() && idx < original_size {
                let bytes = &compressed.data[start..end];
                let value = f64::from_le_bytes(bytes.try_into().unwrap());
                gradients[idx] = <T as torsh_core::dtype::TensorElement>::from_f64(value)
                    .expect("f64 conversion should succeed");
            }
        }

        Ok(gradients)
    }

    fn decompress_gradient_sketching(&self, _compressed: &CompressedGradient) -> Result<Vec<T>> {
        // Gradient sketching is lossy and cannot be perfectly reconstructed
        // This would require additional information or approximation
        Err(TorshError::AutogradError(
            "Gradient sketching decompression not implemented - lossy compression".to_string(),
        ))
    }

    fn decompress_power_sgd(&self, compressed: &CompressedGradient) -> Result<Vec<T>> {
        let original_size = compressed.original_shape[0];
        let rank = compressed.metadata.rank;

        let mut gradients = vec![<T as torsh_core::dtype::TensorElement>::zero(); original_size];

        // Simple reconstruction: broadcast first 'rank' values
        let values_per_element = std::mem::size_of::<f64>();
        let num_values = (compressed.data.len() / values_per_element).min(rank);

        for i in 0..num_values.min(original_size) {
            let start = i * values_per_element;
            let end = start + values_per_element;
            let bytes = &compressed.data[start..end];
            let value = f64::from_le_bytes(bytes.try_into().unwrap());
            gradients[i] = <T as torsh_core::dtype::TensorElement>::from_f64(value)
                .expect("f64 conversion should succeed");
        }

        Ok(gradients)
    }

    fn decompress_error_feedback(&self, compressed: &CompressedGradient) -> Result<Vec<T>> {
        // Error feedback uses the same decompression as the underlying algorithm
        self.decompress_quantization_8bit(compressed)
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> CompressionStats {
        self.stats
            .read()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Reset compression statistics
    pub fn reset_stats(&mut self) {
        *self.stats.write().expect("lock should not be poisoned") = CompressionStats::default();
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: CompressionConfig) {
        self.config = new_config;
    }
}

/// Utilities for compression analysis
pub mod utils {
    use super::*;

    /// Analyze gradient characteristics to suggest optimal compression
    pub fn analyze_gradients<T: FloatElement + ToPrimitive>(gradients: &[T]) -> GradientAnalysis {
        if gradients.is_empty() {
            return GradientAnalysis::default();
        }

        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut sum_squares = 0.0;
        let mut zero_count = 0;

        for &val in gradients {
            let val_f64 = ToPrimitive::to_f64(&val).expect("f64 conversion should succeed");
            min_val = min_val.min(val_f64);
            max_val = max_val.max(val_f64);
            sum += val_f64;
            sum_squares += val_f64 * val_f64;

            if val_f64.abs() < 1e-8 {
                zero_count += 1;
            }
        }

        let n = gradients.len() as f64;
        let mean = sum / n;
        let variance = (sum_squares / n) - (mean * mean);
        let std_dev = variance.sqrt();
        let sparsity = zero_count as f64 / n;

        GradientAnalysis {
            min_value: min_val,
            max_value: max_val,
            mean,
            std_dev,
            sparsity,
            dynamic_range: max_val - min_val,
            recommended_algorithm: if sparsity > 0.1 {
                CompressionAlgorithm::TopKSparsification
            } else if std_dev < 0.01 {
                CompressionAlgorithm::Quantization2Bit
            } else if std_dev < 0.1 {
                CompressionAlgorithm::Quantization4Bit
            } else {
                CompressionAlgorithm::Quantization8Bit
            },
        }
    }

    /// Benchmark different compression algorithms
    pub fn benchmark_compression<T: FloatElement + FromPrimitive + ToPrimitive>(
        gradients: &[T],
        algorithms: &[CompressionAlgorithm],
    ) -> Vec<CompressionBenchmark> {
        let mut results = Vec::new();

        for &algorithm in algorithms {
            let config = CompressionConfig {
                algorithm,
                ..Default::default()
            };

            let mut compressor = GradientCompressor::new(config);

            let start_time = std::time::Instant::now();
            if let Ok(compressed) = compressor.compress(gradients, "benchmark") {
                let compression_time = start_time.elapsed();

                let start_decomp = std::time::Instant::now();
                if let Ok(decompressed) = compressor.decompress(&compressed) {
                    let decompression_time = start_decomp.elapsed();

                    // Calculate error
                    let error = calculate_compression_error(gradients, &decompressed);

                    let compression_ratio =
                        compressed.data.len() as f64 / std::mem::size_of_val(gradients) as f64;

                    results.push(CompressionBenchmark {
                        algorithm,
                        compression_ratio,
                        compression_time,
                        decompression_time,
                        error,
                        compressed_size: compressed.data.len(),
                        original_size: std::mem::size_of_val(gradients),
                    });
                }
            }
        }

        results
    }

    /// Calculate compression error (MSE)
    fn calculate_compression_error<T: FloatElement + ToPrimitive>(
        original: &[T],
        reconstructed: &[T],
    ) -> f64 {
        if original.len() != reconstructed.len() {
            return f64::INFINITY;
        }

        let mse = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| {
                let diff = ToPrimitive::to_f64(&a).expect("f64 conversion should succeed")
                    - ToPrimitive::to_f64(&b).expect("f64 conversion should succeed");
                diff * diff
            })
            .sum::<f64>()
            / original.len() as f64;

        mse
    }
}

/// Gradient analysis results
#[derive(Debug, Clone)]
pub struct GradientAnalysis {
    pub min_value: f64,
    pub max_value: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub sparsity: f64,
    pub dynamic_range: f64,
    pub recommended_algorithm: CompressionAlgorithm,
}

impl Default for GradientAnalysis {
    fn default() -> Self {
        Self {
            min_value: 0.0,
            max_value: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            sparsity: 0.0,
            dynamic_range: 0.0,
            recommended_algorithm: CompressionAlgorithm::Quantization8Bit,
        }
    }
}

/// Compression benchmark results
#[derive(Debug, Clone)]
pub struct CompressionBenchmark {
    pub algorithm: CompressionAlgorithm,
    pub compression_ratio: f64,
    pub compression_time: std::time::Duration,
    pub decompression_time: std::time::Duration,
    pub error: f64,
    pub compressed_size: usize,
    pub original_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quantization_8bit() {
        let gradients: Vec<f32> = vec![0.1, 0.2, -0.3, 0.4, -0.5];
        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::Quantization8Bit,
            ..Default::default()
        };

        let mut compressor = GradientCompressor::new(config);
        let compressed = compressor.compress(&gradients, "test").unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed.len(), gradients.len());

        // Check that decompressed values are close to original (within quantization error)
        for (i, (&original, &reconstructed)) in
            gradients.iter().zip(decompressed.iter()).enumerate()
        {
            let error = (original - reconstructed).abs();
            assert!(
                error < 0.01,
                "Value {} decompression error too large: {}",
                i,
                error
            );
        }
    }

    #[test]
    fn test_top_k_sparsification() {
        let gradients: Vec<f32> = vec![0.1, 0.01, -0.3, 0.002, -0.5, 0.001];
        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::TopKSparsification,
            sparsity_threshold: 0.5, // Keep 50%
            ..Default::default()
        };

        let mut compressor = GradientCompressor::new(config);
        let compressed = compressor.compress(&gradients, "test").unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed.len(), gradients.len());

        // Check that large values are preserved (with some tolerance for compression)
        assert_relative_eq!(decompressed[2], -0.3, epsilon = 0.1);
        assert_relative_eq!(decompressed[4], -0.5, epsilon = 0.1);
    }

    #[test]
    fn test_adaptive_compression() {
        let gradients: Vec<f32> = vec![0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3]; // Sparse
        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::Adaptive,
            sparsity_threshold: 0.4, // 40% sparsity threshold
            ..Default::default()
        };

        let mut compressor = GradientCompressor::new(config);
        let compressed = compressor.compress(&gradients, "test").unwrap();

        // Should choose sparsification for sparse data
        assert_eq!(
            compressed.algorithm,
            CompressionAlgorithm::TopKSparsification
        );
    }

    #[test]
    fn test_compression_stats() {
        let gradients: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let config = CompressionConfig::default();

        let mut compressor = GradientCompressor::new(config);
        compressor.compress(&gradients, "test").unwrap();

        let stats = compressor.get_stats();
        assert_eq!(stats.total_compressions, 1);
        assert!(stats.total_bytes_original > 0);
        assert!(stats.total_bytes_compressed > 0);
    }

    #[test]
    fn test_gradient_analysis() {
        let gradients: Vec<f32> = vec![0.1, 0.0, 0.2, 0.0, 0.3];
        let analysis = utils::analyze_gradients(&gradients);

        assert_eq!(analysis.sparsity, 0.4); // 2 out of 5 are zero
        assert_relative_eq!(analysis.mean, 0.12, epsilon = 1e-6);
        assert!(analysis.std_dev > 0.0);
    }
}
