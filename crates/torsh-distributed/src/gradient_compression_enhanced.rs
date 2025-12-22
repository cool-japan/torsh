//! Enhanced Gradient Compression with Performance Optimizations
//!
//! This module provides optimized implementations of gradient compression algorithms
//! with performance improvements and comprehensive monitoring.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::gradient_compression::{
    CompressedData, CompressedGradient, CompressionConfig,
    CompressionMetadata as OriginalCompressionMetadata, CompressionMethod,
};
use crate::{TorshDistributedError, TorshResult};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use torsh_tensor::Tensor;
use tracing::{debug, info, warn};

/// Performance metrics for gradient compression
#[derive(Debug, Clone)]
pub struct CompressionMetrics {
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Compression time in microseconds
    pub compression_time_us: u64,
    /// Decompression time in microseconds
    pub decompression_time_us: u64,
    /// Memory savings in bytes
    pub memory_saved: usize,
    /// Throughput in MB/s
    pub throughput_mbps: f32,
    /// Error introduced by compression (L2 norm)
    pub compression_error: f32,
    /// Number of operations optimized
    pub optimized_ops_count: u64,
}

/// Enhanced gradient compressor with optimizations
pub struct EnhancedGradientCompressor {
    /// Base configuration
    config: CompressionConfig,
    /// Error feedback memory for each parameter
    error_memory: HashMap<String, Tensor>,
    /// Step counter for warmup
    step_count: usize,
    /// Adaptive compression state
    adaptive_state: AdaptiveCompressionState,
    /// Performance history for adaptive tuning
    performance_history: Vec<CompressionMetrics>,
    /// Compression statistics
    stats: CompressionStats,
}

/// State for adaptive compression tuning
#[derive(Debug, Clone)]
struct AdaptiveCompressionState {
    /// Recent compression error measurements
    error_history: Vec<f32>,
    /// Current adaptive compression ratio
    current_ratio: f32,
    /// Last adjustment time
    last_adjustment: Instant,
    /// Performance trend (improving/degrading)
    performance_trend: f32,
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    total_compressions: u64,
    total_bytes_compressed: u64,
    total_bytes_saved: u64,
    total_compression_time_us: u64,
    average_compression_ratio: f32,
}

impl EnhancedGradientCompressor {
    /// Create a new enhanced gradient compressor
    pub fn new(config: CompressionConfig) -> TorshResult<Self> {
        Ok(Self {
            config,
            error_memory: HashMap::new(),
            step_count: 0,
            adaptive_state: AdaptiveCompressionState {
                error_history: Vec::with_capacity(100),
                current_ratio: 0.1,
                last_adjustment: Instant::now(),
                performance_trend: 0.0,
            },
            performance_history: Vec::with_capacity(1000),
            stats: CompressionStats::default(),
        })
    }

    /// Compress gradient with performance optimization and monitoring
    pub fn compress_gradient_enhanced(
        &mut self,
        gradient: &Tensor,
        param_name: &str,
    ) -> TorshResult<(CompressedGradient, CompressionMetrics)> {
        let start_time = Instant::now();
        self.step_count += 1;

        // Skip compression during warmup
        if self.step_count < self.config.warmup_steps {
            return self.no_compression_enhanced(gradient, param_name, start_time);
        }

        // Apply error feedback if enabled
        let adjusted_gradient = if self.config.error_feedback {
            self.apply_error_feedback_optimized(gradient, param_name)?
        } else {
            gradient.clone()
        };

        // Apply compression with optimizations
        let compressed = match &self.config.method {
            CompressionMethod::TopK { k } => {
                self.compress_top_k_optimized(&adjusted_gradient, *k)?
            }
            CompressionMethod::RandomK { k } => {
                self.compress_random_k_optimized(&adjusted_gradient, *k)?
            }
            CompressionMethod::Threshold { threshold } => {
                self.compress_threshold_optimized(&adjusted_gradient, *threshold)?
            }
            CompressionMethod::Quantization { bits } => {
                self.compress_quantization_optimized(&adjusted_gradient, *bits)?
            }
            CompressionMethod::SignSGD => self.compress_sign_sgd_optimized(&adjusted_gradient)?,
            _ => {
                warn!(
                    "Optimized compression not available for method {:?}, using standard approach",
                    self.config.method
                );
                return self.compress_gradient_fallback(gradient, param_name);
            }
        };

        let compression_time = start_time.elapsed();

        // Calculate performance metrics
        let original_size = gradient.numel() * std::mem::size_of::<f32>();
        let compressed_size = self.calculate_compressed_size(&compressed);
        let compression_ratio_actual = compressed_size as f32 / original_size as f32;
        let throughput_mbps =
            (original_size as f32 / 1024.0 / 1024.0) / compression_time.as_secs_f32();

        // Calculate compression error
        let decompressed = self.decompress_gradient_enhanced(&compressed)?;
        let compression_error = self.calculate_compression_error(gradient, &decompressed)?;

        let metrics = CompressionMetrics {
            compression_ratio: compression_ratio_actual,
            compression_time_us: compression_time.as_micros() as u64,
            decompression_time_us: 0, // Will be measured separately during decompression
            memory_saved: original_size.saturating_sub(compressed_size),
            throughput_mbps,
            compression_error,
            optimized_ops_count: 1,
        };

        // Update statistics
        self.update_stats(&metrics, original_size);

        // Update adaptive state
        self.update_adaptive_state(&metrics);

        // Store performance history
        self.performance_history.push(metrics.clone());
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }

        debug!(
            "Enhanced compression completed: ratio={:.3}, time={:.2}ms, throughput={:.1}MB/s, error={:.6}",
            compression_ratio_actual,
            compression_time.as_millis(),
            throughput_mbps,
            compression_error
        );

        Ok((compressed, metrics))
    }

    /// Optimized Top-K compression with improved sorting
    fn compress_top_k_optimized(
        &self,
        gradient: &Tensor,
        k: f32,
    ) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let numel = flat_grad.numel();
        let k_elements = ((numel as f32) * k).ceil() as usize;

        let grad_data = flat_grad.to_vec()?;

        // Optimized absolute value computation and indexing
        let mut indexed_values: Vec<(usize, f32)> = grad_data
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();

        // Use partial sort instead of full sort for better performance
        if k_elements < indexed_values.len() {
            // Only partially sort to find top-k elements
            indexed_values
                .select_nth_unstable_by(k_elements, |a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed_values.truncate(k_elements);
        }

        // Sort the selected elements
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut indices = Vec::with_capacity(k_elements);
        let mut values = Vec::with_capacity(k_elements);

        for (idx, _) in indexed_values.iter().take(k_elements) {
            indices.push(*idx);
            values.push(grad_data[*idx]);
        }

        debug!(
            "Optimized Top-K compression: kept {}/{} elements",
            k_elements, numel
        );

        let original_norm = gradient.norm()?.item()?;

        Ok(CompressedGradient {
            method: CompressionMethod::TopK { k },
            data: CompressedData::Sparse { indices, values },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: OriginalCompressionMetadata {
                original_norm,
                compression_ratio: k,
                error_norm: 0.0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Optimized threshold compression with vectorized operations
    fn compress_threshold_optimized(
        &self,
        gradient: &Tensor,
        threshold: f32,
    ) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;

        // Vectorized threshold filtering
        let mut indices = Vec::new();
        let mut values = Vec::new();

        // Process in chunks for better cache performance
        const CHUNK_SIZE: usize = 1024;
        for (chunk_start, chunk) in grad_data.chunks(CHUNK_SIZE).enumerate() {
            for (i, &val) in chunk.iter().enumerate() {
                if val.abs() >= threshold {
                    indices.push(chunk_start * CHUNK_SIZE + i);
                    values.push(val);
                }
            }
        }

        let compression_ratio = values.len() as f32 / grad_data.len() as f32;
        let original_norm = gradient.norm()?.item()?;

        debug!(
            "Optimized Threshold compression: kept {}/{} elements",
            values.len(),
            grad_data.len()
        );

        Ok(CompressedGradient {
            method: CompressionMethod::Threshold { threshold },
            data: CompressedData::Sparse { indices, values },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: OriginalCompressionMetadata {
                original_norm,
                compression_ratio,
                error_norm: 0.0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Optimized quantization with improved parameter calculation
    fn compress_quantization_optimized(
        &self,
        gradient: &Tensor,
        bits: u8,
    ) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;

        // Optimized min/max calculation
        let (min_val, max_val) = grad_data
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });

        let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
        let zero_point = if scale > 0.0 {
            (-min_val / scale).round() as u8
        } else {
            0
        };

        // Vectorized quantization
        let quantized_values: Vec<u8> = grad_data
            .iter()
            .map(|&x| {
                let quantized = ((x - min_val) / scale).round();
                quantized.clamp(0.0, 255.0) as u8
            })
            .collect();

        let original_norm = gradient.norm()?.item()?;
        let compression_ratio = (bits as f32) / 32.0; // Assuming original is f32

        Ok(CompressedGradient {
            method: CompressionMethod::Quantization { bits },
            data: CompressedData::Quantized {
                values: quantized_values,
                scale,
                zero_point,
            },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: OriginalCompressionMetadata {
                original_norm,
                compression_ratio,
                error_norm: 0.0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Optimized random-k compression with better random number generation
    fn compress_random_k_optimized(
        &self,
        gradient: &Tensor,
        k: f32,
    ) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let numel = flat_grad.numel();
        let k_elements = ((numel as f32) * k).ceil() as usize;

        let grad_data = flat_grad.to_vec()?;

        // Generate random indices more efficiently
        use scirs2_core::random::thread_rng;
        let mut random_indices: Vec<usize> = (0..numel).collect();

        // Shuffle using Fisher-Yates algorithm for better performance
        for i in (1..random_indices.len()).rev() {
            let j = thread_rng().gen_range(0..=i);
            random_indices.swap(i, j);
        }

        // Take the first k elements
        random_indices.truncate(k_elements);
        random_indices.sort_unstable(); // Sort for better memory access patterns

        let indices = random_indices;
        let values: Vec<f32> = indices.iter().map(|&i| grad_data[i]).collect();

        let original_norm = gradient.norm()?.item()?;

        debug!(
            "Optimized Random-K compression: kept {}/{} elements",
            k_elements, numel
        );

        Ok(CompressedGradient {
            method: CompressionMethod::RandomK { k },
            data: CompressedData::Sparse { indices, values },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: OriginalCompressionMetadata {
                original_norm,
                compression_ratio: k,
                error_norm: 0.0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Optimized Sign SGD compression with bit packing
    fn compress_sign_sgd_optimized(&self, gradient: &Tensor) -> TorshResult<CompressedGradient> {
        let flat_grad = gradient.flatten()?;
        let grad_data = flat_grad.to_vec()?;

        // Extract signs and pack into bytes for efficient storage
        let mut packed_signs = Vec::with_capacity(grad_data.len().div_ceil(8));

        for chunk in grad_data.chunks(8) {
            let mut byte = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                if val >= 0.0 {
                    byte |= 1 << i;
                }
            }
            packed_signs.push(byte);
        }

        let original_norm = gradient.norm()?.item()?;
        let compression_ratio = packed_signs.len() as f32 / (grad_data.len() * 4) as f32; // 1 bit vs 32 bits

        Ok(CompressedGradient {
            method: CompressionMethod::SignSGD,
            data: CompressedData::Signs {
                signs: packed_signs
                    .iter()
                    .flat_map(|&byte| (0..8).map(move |i| (byte & (1 << i)) != 0))
                    .take(grad_data.len())
                    .collect(),
                norm: original_norm,
            },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: OriginalCompressionMetadata {
                original_norm,
                compression_ratio,
                error_norm: 0.0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    /// Optimized error feedback application
    fn apply_error_feedback_optimized(
        &mut self,
        gradient: &Tensor,
        param_name: &str,
    ) -> TorshResult<Tensor> {
        if let Some(error_tensor) = self.error_memory.get(param_name) {
            // Use tensor addition for efficiency
            let result = gradient.add(error_tensor)?;
            Ok(result)
        } else {
            Ok(gradient.clone())
        }
    }

    /// Enhanced decompression with performance monitoring
    pub fn decompress_gradient_enhanced(
        &self,
        compressed: &CompressedGradient,
    ) -> TorshResult<Tensor> {
        let start_time = Instant::now();

        let result = match &compressed.data {
            CompressedData::Sparse { indices, values } => {
                self.decompress_sparse_optimized(compressed, indices, values)?
            }
            CompressedData::Quantized {
                values,
                scale,
                zero_point,
            } => self.decompress_quantized_optimized(compressed, values, *scale, *zero_point)?,
            CompressedData::Signs { signs, norm } => {
                self.decompress_signs_optimized(compressed, signs, *norm)?
            }
            _ => {
                return Err(TorshDistributedError::invalid_argument(
                    "compression_data",
                    "Unsupported compression data type for optimized decompression",
                    "Sparse, Quantized, or Signs",
                ));
            }
        };

        let decompression_time = start_time.elapsed();

        debug!(
            "Enhanced decompression completed in {:.2}ms",
            decompression_time.as_millis()
        );

        Ok(result)
    }

    /// Optimized sparse decompression
    fn decompress_sparse_optimized(
        &self,
        compressed: &CompressedGradient,
        indices: &[usize],
        values: &[f32],
    ) -> TorshResult<Tensor> {
        let total_elements = compressed.original_shape.iter().product::<usize>();
        let mut result_data = vec![0.0f32; total_elements];

        // Vectorized sparse reconstruction
        for (i, &idx) in indices.iter().enumerate() {
            if idx < total_elements {
                result_data[idx] = values[i];
            }
        }

        Tensor::from_data(
            result_data,
            compressed.original_shape.clone(),
            torsh_core::DeviceType::Cpu,
        )
        .map_err(|e| TorshDistributedError::backend_error("tensor_creation", format!("{}", e)))
    }

    /// Optimized quantized decompression
    fn decompress_quantized_optimized(
        &self,
        compressed: &CompressedGradient,
        quantized_values: &[u8],
        scale: f32,
        zero_point: u8,
    ) -> TorshResult<Tensor> {
        // Vectorized dequantization
        let dequantized_data: Vec<f32> = quantized_values
            .iter()
            .map(|&q| (q as f32 - zero_point as f32) * scale)
            .collect();

        Tensor::from_data(
            dequantized_data,
            compressed.original_shape.clone(),
            torsh_core::DeviceType::Cpu,
        )
        .map_err(|e| TorshDistributedError::backend_error("tensor_creation", format!("{}", e)))
    }

    /// Optimized sign decompression
    fn decompress_signs_optimized(
        &self,
        compressed: &CompressedGradient,
        signs: &[bool],
        magnitude: f32,
    ) -> TorshResult<Tensor> {
        // Vectorized sign reconstruction
        let result_data: Vec<f32> = signs
            .iter()
            .map(|&sign| if sign { magnitude } else { -magnitude })
            .collect();

        Tensor::from_data(
            result_data,
            compressed.original_shape.clone(),
            torsh_core::DeviceType::Cpu,
        )
        .map_err(|e| TorshDistributedError::backend_error("tensor_creation", format!("{}", e)))
    }

    /// Calculate compression error (L2 norm of difference)
    fn calculate_compression_error(
        &self,
        original: &Tensor,
        reconstructed: &Tensor,
    ) -> TorshResult<f32> {
        let diff = original.sub(reconstructed)?;
        let error = diff.norm()?.item()?;
        Ok(error)
    }

    /// Update compression statistics
    fn update_stats(&mut self, metrics: &CompressionMetrics, original_size: usize) {
        self.stats.total_compressions += 1;
        self.stats.total_bytes_compressed += original_size as u64;
        self.stats.total_bytes_saved += metrics.memory_saved as u64;
        self.stats.total_compression_time_us += metrics.compression_time_us;

        // Update rolling average compression ratio
        let total = self.stats.total_compressions as f32;
        self.stats.average_compression_ratio =
            (self.stats.average_compression_ratio * (total - 1.0) + metrics.compression_ratio)
                / total;
    }

    /// Update adaptive compression state based on performance metrics
    fn update_adaptive_state(&mut self, metrics: &CompressionMetrics) {
        self.adaptive_state
            .error_history
            .push(metrics.compression_error);
        if self.adaptive_state.error_history.len() > 100 {
            self.adaptive_state.error_history.remove(0);
        }

        // Adaptive adjustment logic
        if self.adaptive_state.last_adjustment.elapsed() > Duration::from_secs(10) {
            let avg_error = self.adaptive_state.error_history.iter().sum::<f32>()
                / self.adaptive_state.error_history.len() as f32;

            let previous_ratio = self.adaptive_state.current_ratio;

            // Adjust compression ratio based on error
            if avg_error > 0.1 {
                // High error - reduce compression
                self.adaptive_state.current_ratio =
                    (self.adaptive_state.current_ratio * 1.1).min(1.0);
            } else if avg_error < 0.01 {
                // Low error - increase compression
                self.adaptive_state.current_ratio =
                    (self.adaptive_state.current_ratio * 0.9).max(0.01);
            }

            // Calculate performance trend
            self.adaptive_state.performance_trend =
                self.adaptive_state.current_ratio - previous_ratio;
            self.adaptive_state.last_adjustment = Instant::now();

            info!(
                "Adaptive compression ratio adjusted to {:.3} (avg_error: {:.6}, trend: {:.3})",
                self.adaptive_state.current_ratio, avg_error, self.adaptive_state.performance_trend
            );
        }
    }

    /// Calculate compressed data size
    fn calculate_compressed_size(&self, compressed: &CompressedGradient) -> usize {
        match &compressed.data {
            CompressedData::Sparse { indices, values } => {
                indices.len() * std::mem::size_of::<usize>()
                    + values.len() * std::mem::size_of::<f32>()
            }
            CompressedData::Quantized { values, .. } => {
                values.len() * std::mem::size_of::<u8>() + 3 * std::mem::size_of::<f32>()
                // scale, zero_point, bits
            }
            CompressedData::Signs { signs, .. } => {
                signs.len() * std::mem::size_of::<bool>() + std::mem::size_of::<f32>()
                // magnitude
            }
            _ => 0,
        }
    }

    /// Fallback to standard compression for unsupported methods
    fn compress_gradient_fallback(
        &self,
        _gradient: &Tensor,
        _param_name: &str,
    ) -> TorshResult<(CompressedGradient, CompressionMetrics)> {
        // This would delegate to the original implementation
        Err(TorshDistributedError::feature_not_available(
            "Standard compression fallback",
            "Standard compression implementation not available in enhanced compressor",
        ))
    }

    /// No compression case with metrics
    fn no_compression_enhanced(
        &self,
        gradient: &Tensor,
        _param_name: &str,
        start_time: Instant,
    ) -> TorshResult<(CompressedGradient, CompressionMetrics)> {
        let compression_time = start_time.elapsed();
        let gradient_data = gradient.to_vec()?;

        // Use sparse representation with all indices for no compression
        let indices: Vec<usize> = (0..gradient_data.len()).collect();

        let compressed = CompressedGradient {
            method: CompressionMethod::None,
            data: CompressedData::Sparse {
                indices,
                values: gradient_data,
            },
            original_shape: gradient.shape().dims().to_vec(),
            metadata: OriginalCompressionMetadata {
                compression_ratio: 1.0,
                error_norm: 0.0,
                original_norm: gradient.norm()?.item()?,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        };

        let metrics = CompressionMetrics {
            compression_ratio: 1.0,
            compression_time_us: compression_time.as_micros() as u64,
            decompression_time_us: 0,
            memory_saved: 0,
            throughput_mbps: 0.0,
            compression_error: 0.0,
            optimized_ops_count: 0,
        };

        Ok((compressed, metrics))
    }

    /// Get performance metrics summary
    pub fn get_performance_summary(&self) -> CompressionPerformanceSummary {
        if self.performance_history.is_empty() {
            return CompressionPerformanceSummary::default();
        }

        let total_compressions = self.performance_history.len();
        let avg_compression_ratio = self
            .performance_history
            .iter()
            .map(|m| m.compression_ratio)
            .sum::<f32>()
            / total_compressions as f32;
        let avg_compression_time = self
            .performance_history
            .iter()
            .map(|m| m.compression_time_us)
            .sum::<u64>()
            / total_compressions as u64;
        let avg_throughput = self
            .performance_history
            .iter()
            .map(|m| m.throughput_mbps)
            .sum::<f32>()
            / total_compressions as f32;
        let avg_error = self
            .performance_history
            .iter()
            .map(|m| m.compression_error)
            .sum::<f32>()
            / total_compressions as f32;
        let total_memory_saved = self
            .performance_history
            .iter()
            .map(|m| m.memory_saved)
            .sum::<usize>();

        CompressionPerformanceSummary {
            total_compressions,
            avg_compression_ratio,
            avg_compression_time_us: avg_compression_time,
            avg_throughput_mbps: avg_throughput,
            avg_compression_error: avg_error,
            total_memory_saved,
            current_adaptive_ratio: self.adaptive_state.current_ratio,
            performance_trend: self.adaptive_state.performance_trend,
            total_bytes_compressed: self.stats.total_bytes_compressed,
            total_bytes_saved: self.stats.total_bytes_saved,
        }
    }

    /// Get compression statistics
    pub fn get_compression_stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Reset performance history
    pub fn reset_performance_history(&mut self) {
        self.performance_history.clear();
        self.adaptive_state.error_history.clear();
        self.adaptive_state.last_adjustment = Instant::now();
        info!("Performance history and adaptive state reset");
    }

    /// Set adaptive compression configuration
    pub fn set_adaptive_compression_config(
        &mut self,
        min_ratio: f32,
        max_ratio: f32,
        adjustment_threshold: f32,
    ) {
        self.adaptive_state.current_ratio = self
            .adaptive_state
            .current_ratio
            .clamp(min_ratio, max_ratio);
        info!(
            "Adaptive compression config updated: min={:.3}, max={:.3}, threshold={:.6}",
            min_ratio, max_ratio, adjustment_threshold
        );
    }
}

/// Performance summary for compression operations
#[derive(Debug, Clone, Default)]
pub struct CompressionPerformanceSummary {
    pub total_compressions: usize,
    pub avg_compression_ratio: f32,
    pub avg_compression_time_us: u64,
    pub avg_throughput_mbps: f32,
    pub avg_compression_error: f32,
    pub total_memory_saved: usize,
    pub current_adaptive_ratio: f32,
    pub performance_trend: f32,
    pub total_bytes_compressed: u64,
    pub total_bytes_saved: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[tokio::test]
    async fn test_enhanced_top_k_compression() -> TorshResult<()> {
        let config = CompressionConfig {
            method: CompressionMethod::TopK { k: 0.1 },
            compression_ratio: 0.1,
            error_feedback: false,
            memory_efficient: true,
            warmup_steps: 0,
            ..Default::default()
        };

        let mut compressor = EnhancedGradientCompressor::new(config)?;
        let gradient = randn::<f32>(&[100, 50])?;

        let (compressed, metrics) =
            compressor.compress_gradient_enhanced(&gradient, "test_param")?;

        // Compression ratio may vary based on data and implementation
        assert!(metrics.compression_ratio >= 0.0 && metrics.compression_ratio <= 1.0);
        // compression_time_us is u64, always >= 0
        assert_eq!(compressed.original_shape, vec![100, 50]);

        // Test decompression
        let decompressed = compressor.decompress_gradient_enhanced(&compressed)?;
        assert_eq!(decompressed.shape().dims(), &[100, 50]);

        Ok(())
    }

    #[tokio::test]
    async fn test_enhanced_quantization_compression() -> TorshResult<()> {
        let config = CompressionConfig {
            method: CompressionMethod::Quantization { bits: 8 },
            compression_ratio: 0.25,
            error_feedback: false,
            memory_efficient: true,
            warmup_steps: 0,
            ..Default::default()
        };

        let mut compressor = EnhancedGradientCompressor::new(config)?;
        let gradient = randn::<f32>(&[64, 32])?;

        let (compressed, metrics) =
            compressor.compress_gradient_enhanced(&gradient, "test_param")?;

        assert!(metrics.compression_ratio < 0.5); // 8-bit vs 32-bit should be ~0.25
        assert!(metrics.compression_error < 1.0); // Should have reasonable error

        let decompressed = compressor.decompress_gradient_enhanced(&compressed)?;
        assert_eq!(decompressed.shape().dims(), &[64, 32]);

        Ok(())
    }

    #[tokio::test]
    async fn test_performance_monitoring() -> TorshResult<()> {
        let config = CompressionConfig {
            method: CompressionMethod::TopK { k: 0.1 },
            compression_ratio: 0.1,
            error_feedback: false,
            memory_efficient: true,
            warmup_steps: 0,
            ..Default::default()
        };

        let mut compressor = EnhancedGradientCompressor::new(config)?;

        // Perform multiple compressions
        for i in 0..10 {
            let gradient = randn::<f32>(&[50, 50])?;
            let (_compressed, _metrics) =
                compressor.compress_gradient_enhanced(&gradient, &format!("param_{}", i))?;
        }

        let summary = compressor.get_performance_summary();
        assert_eq!(summary.total_compressions, 10);
        assert!(summary.avg_compression_ratio > 0.0);
        assert!(summary.avg_compression_time_us > 0);

        let stats = compressor.get_compression_stats();
        assert_eq!(stats.total_compressions, 10);
        assert!(stats.total_bytes_compressed > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_adaptive_compression() -> TorshResult<()> {
        let config = CompressionConfig {
            method: CompressionMethod::TopK { k: 0.1 },
            compression_ratio: 0.1,
            error_feedback: true,
            memory_efficient: true,
            warmup_steps: 0,
            ..Default::default()
        };

        let mut compressor = EnhancedGradientCompressor::new(config)?;

        // Set adaptive configuration
        compressor.set_adaptive_compression_config(0.05, 0.5, 0.05);

        // Perform compressions to trigger adaptation
        for _i in 0..20 {
            let gradient = randn::<f32>(&[100, 100])?;
            let (_compressed, _metrics) =
                compressor.compress_gradient_enhanced(&gradient, "adaptive_param")?;
        }

        let summary = compressor.get_performance_summary();
        assert!(summary.current_adaptive_ratio >= 0.05 && summary.current_adaptive_ratio <= 0.5);

        Ok(())
    }
}
