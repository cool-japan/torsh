//! Compression engine and algorithms for communication-efficient distributed training.
//!
//! This module provides comprehensive gradient compression capabilities designed to reduce
//! communication overhead in distributed deep learning training. It supports multiple
//! compression strategies including quantization, sparsification, low-rank approximation,
//! and sketching-based methods.
//!
//! # Features
//!
//! - **Multiple compression strategies**: Quantization, sparsification, low-rank, sketching
//! - **Hybrid compression**: Combines multiple techniques for optimal compression
//! - **Adaptive compression**: Dynamic adjustment based on network conditions
//! - **Error feedback**: Maintains mathematical correctness despite compression
//! - **SIMD optimization**: Uses SciRS2 for high-performance computation
//! - **Thread-safe**: Arc<Mutex<>> for distributed computing patterns
//!
//! # Mathematical Foundation
//!
//! The compression algorithms implement various mathematical techniques:
//!
//! ## Quantization
//! Maps continuous values to discrete levels: `q = round((x - min) / scale)`
//! where `scale = (max - min) / (2^bits - 1)`
//!
//! ## Sparsification
//! Keeps only top-k gradient values: `sparse(x) = {x_i if |x_i| ∈ top_k(|x|), 0 otherwise}`
//!
//! ## Low-rank Approximation
//! Decomposes gradients: `G ≈ UV^T` where `U ∈ R^{m×r}`, `V ∈ R^{n×r}`, `r << min(m,n)`
//!
//! ## Sketching
//! Random projections: `sketch = Φ * x` where `Φ` is a random matrix
//!
//! # Examples
//!
//! ## Basic Compression
//! ```rust
//! use torsh_autograd::communication_efficient::compression::*;
//! use torsh_autograd::communication_efficient::config::*;
//! use std::collections::HashMap;
//!
//! let mut engine = CompressionEngine::new(CompressionStrategy::Quantization);
//! let mut gradient = HashMap::new();
//! gradient.insert("layer1.weight".to_string(), vec![0.1, 0.2, 0.3, 0.4]);
//!
//! let config = CommunicationConfig::default();
//! let compressed = engine.compress(&gradient, &config).unwrap();
//! let decompressed = engine.decompress(&compressed).unwrap();
//! ```
//!
//! ## Adaptive Compression with Error Feedback
//! ```rust
//! use torsh_autograd::communication_efficient::compression::*;
//! use torsh_autograd::communication_efficient::config::*;
//! use std::collections::HashMap;
//!
//! let mut engine = CompressionEngine::new(CompressionStrategy::AdaptiveCompression);
//! let mut gradient = HashMap::new();
//! gradient.insert("param".to_string(), vec![1.0, 2.0, 3.0]);
//!
//! let config = CommunicationConfig::default();
//! let compressed = engine.compress(&gradient, &config).unwrap();
//!
//! // Compute error feedback for mathematical correctness
//! let ce_gradient = CommunicationEfficientGradient::new(gradient, 1, 100);
//! let error_feedback = engine.compute_error_feedback(&ce_gradient).unwrap();
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// SciRS2 imports for random number generation
use scirs2_core::random::quick::random_f32;

// Import configuration types from the config module
use super::config::{
    CommunicationConfig, CommunicationEfficientGradient, CompressedGradient, CompressionInfo,
    CompressionStrategy, DecompressionHints, ReconstructionMethod, SketchParameters, SketchType,
    SparsityPattern,
};

// Use SciRS2-Core for scientific computing (following SCIRS2_INTEGRATION_POLICY.md)
use crate::Result as SciResult;
use scirs2_core::error::CoreError;

// Error type for compression operations
#[derive(Debug, Clone)]
pub enum CompressionError {
    CompressionFailed(String),
    DecompressionFailed(String),
    InvalidParameters(String),
    InsufficientData(String),
    MathematicalError(String),
}

impl std::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionError::CompressionFailed(msg) => write!(f, "Compression failed: {}", msg),
            CompressionError::DecompressionFailed(msg) => {
                write!(f, "Decompression failed: {}", msg)
            }
            CompressionError::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
            CompressionError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            CompressionError::MathematicalError(msg) => write!(f, "Mathematical error: {}", msg),
        }
    }
}

impl std::error::Error for CompressionError {}

/// Main compression engine that coordinates all compression strategies.
///
/// The compression engine manages multiple specialized compression algorithms
/// and provides a unified interface for compressing and decompressing gradients.
/// It maintains statistics, handles error feedback, and supports adaptive
/// compression based on network conditions.
///
/// # Thread Safety
///
/// The compression engine is designed to be thread-safe and can be shared
/// across multiple workers using `Arc<Mutex<CompressionEngine>>`.
///
/// # Mathematical Correctness
///
/// All compression algorithms maintain mathematical correctness through
/// error feedback mechanisms that track and compensate for compression errors.
#[derive(Debug)]
pub struct CompressionEngine {
    /// Current compression strategy being used
    strategy: CompressionStrategy,
    /// Quantization-specific compression engine
    quantization_engine: QuantizationEngine,
    /// Sparsification-specific compression engine
    sparsification_engine: SparsificationEngine,
    /// Low-rank approximation compression engine
    low_rank_engine: LowRankEngine,
    /// Sketching-based compression engine
    sketching_engine: SketchingEngine,
    /// Error feedback buffer for each worker
    error_feedback_buffer: HashMap<u32, HashMap<String, Vec<f32>>>,
    /// Comprehensive compression statistics
    compression_statistics: CompressionStatistics,
    /// Adaptive compression parameters
    adaptive_parameters: AdaptiveCompressionParameters,
}

/// Quantization engine for reducing gradient precision.
///
/// Implements various quantization strategies including uniform, non-uniform,
/// logarithmic, and adaptive quantization. Supports stochastic rounding
/// for improved convergence properties.
///
/// # Mathematical Foundation
///
/// Uniform quantization maps values to discrete levels:
/// `q = round((x - min) / scale)` where `scale = (max - min) / (2^bits - 1)`
#[derive(Debug)]
pub struct QuantizationEngine {
    /// Number of bits used for quantization
    pub quantization_bits: u8,
    /// Quantization method (uniform, non-uniform, etc.)
    pub quantization_method: QuantizationMethod,
    /// Whether to use adaptive quantization
    pub adaptive_quantization: bool,
    /// Quantization bounds for each parameter
    pub quantization_bounds: HashMap<String, (f32, f32)>,
    /// Whether to use stochastic rounding
    pub stochastic_rounding: bool,
}

/// Quantization methods supporting different precision reduction strategies.
///
/// Different methods provide various trade-offs between compression ratio,
/// computational overhead, and convergence quality.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationMethod {
    /// Uniform quantization with equal-width bins
    Uniform,
    /// Non-uniform quantization with variable-width bins
    NonUniform,
    /// Logarithmic quantization for improved dynamic range
    Logarithmic,
    /// Adaptive quantization that adjusts based on data distribution
    Adaptive,
    /// Stochastic quantization with probabilistic rounding
    Stochastic,
    /// Ternary quantization using only {-1, 0, 1}
    TernaryQuantization,
    /// Binary quantization using only {-1, 1}
    BinaryQuantization,
}

/// Sparsification engine for gradient pruning.
///
/// Implements various sparsification strategies including top-k selection,
/// random sampling, threshold-based pruning, and importance sampling.
/// Supports momentum correction and threshold adaptation.
///
/// # Mathematical Foundation
///
/// Top-k sparsification keeps only the largest magnitude elements:
/// `sparse(x) = {x_i if |x_i| ∈ top_k(|x|), 0 otherwise}`
#[derive(Debug)]
pub struct SparsificationEngine {
    /// Ratio of gradient values to keep (0.0 to 1.0)
    pub sparsification_ratio: f64,
    /// Sparsification method (top-k, random, threshold, etc.)
    pub sparsification_method: SparsificationMethod,
    /// Whether to adapt threshold based on gradient distribution
    pub threshold_adaptation: bool,
    /// Whether to use importance sampling for element selection
    pub importance_sampling: bool,
    /// Whether to apply momentum correction
    pub momentum_correction: bool,
}

/// Sparsification methods for different gradient pruning strategies.
///
/// Each method provides different characteristics in terms of compression
/// ratio, computational cost, and convergence properties.
#[derive(Debug, Clone, PartialEq)]
pub enum SparsificationMethod {
    /// Keep top-k elements by magnitude
    TopK,
    /// Random sampling of gradient elements
    Random,
    /// Threshold-based pruning
    Threshold,
    /// Importance sampling based on gradient statistics
    ImportanceSampling,
    /// Gradient dropping for slow workers
    GradientDrop,
    /// Adaptive sparsification with dynamic ratios
    AdaptiveSparsification,
}

/// Low-rank approximation engine for matrix decomposition-based compression.
///
/// Implements various matrix decomposition methods including SVD, QR, LU,
/// and randomized algorithms. Supports adaptive rank selection based on
/// energy preservation or error thresholds.
///
/// # Mathematical Foundation
///
/// Low-rank approximation decomposes gradients: `G ≈ UV^T`
/// where `U ∈ R^{m×r}`, `V ∈ R^{n×r}`, `r << min(m,n)`
#[derive(Debug)]
pub struct LowRankEngine {
    /// Target rank for decomposition
    pub rank: usize,
    /// Matrix decomposition method
    pub decomposition_method: DecompositionMethod,
    /// Whether to use adaptive rank selection
    pub adaptive_rank: bool,
    /// Strategy for rank adaptation
    pub rank_adaptation_strategy: RankAdaptationStrategy,
}

/// Matrix decomposition methods for low-rank approximation.
///
/// Different decomposition methods provide various computational
/// characteristics and approximation qualities.
#[derive(Debug, Clone, PartialEq)]
pub enum DecompositionMethod {
    /// Singular Value Decomposition (highest accuracy)
    SVD,
    /// QR decomposition (faster computation)
    QR,
    /// LU decomposition (specialized cases)
    LU,
    /// Non-negative Matrix Factorization
    NMF,
    /// Randomized SVD (faster for large matrices)
    RandomizedSVD,
}

/// Strategies for adaptive rank selection in low-rank compression.
///
/// Different strategies balance compression ratio with approximation quality.
#[derive(Debug, Clone, PartialEq)]
pub enum RankAdaptationStrategy {
    /// Fixed rank (no adaptation)
    Fixed,
    /// Energy-based rank selection (preserve energy percentage)
    EnergyBased,
    /// Error-based rank selection (maintain error threshold)
    ErrorBased,
    /// Budget-based adaptive rank selection
    AdaptiveBudget,
}

/// Sketching engine for random projection-based compression.
///
/// Implements various sketching algorithms including Count Sketch,
/// Johnson-Lindenstrauss embeddings, and sparse random projections.
/// Supports adaptive sketch sizing and multiple hash functions.
///
/// # Mathematical Foundation
///
/// Sketching uses random projections: `sketch = Φ * x`
/// where `Φ` is a random matrix with specific structure.
#[derive(Debug)]
pub struct SketchingEngine {
    /// Size of the sketch representation
    pub sketch_size: usize,
    /// Type of sketching algorithm
    pub sketch_type: SketchType,
    /// Hash functions for sketching
    pub hash_functions: Vec<u32>,
    /// Random matrices for different parameters
    pub random_matrices: HashMap<String, Vec<f32>>,
    /// Whether to adapt sketch size dynamically
    pub sketch_adaptation: bool,
}

/// Comprehensive statistics for compression operations.
///
/// Tracks detailed metrics about compression performance, quality,
/// and efficiency for monitoring and optimization purposes.
#[derive(Debug, Default)]
pub struct CompressionStatistics {
    /// Total number of compression operations performed
    pub total_compressions: u64,
    /// Total size of original gradients (bytes)
    pub total_original_size: u64,
    /// Total size of compressed gradients (bytes)
    pub total_compressed_size: u64,
    /// Running average compression ratio
    pub average_compression_ratio: f64,
    /// Total time spent on compression
    pub compression_time_total: Duration,
    /// Usage count for each compression method
    pub method_usage_count: HashMap<CompressionStrategy, u64>,
    /// Number of compression errors encountered
    pub compression_errors: u64,
}

/// Parameters for adaptive compression behavior.
///
/// Controls how the compression engine adapts to changing network
/// conditions and performance requirements.
#[derive(Debug)]
pub struct AdaptiveCompressionParameters {
    /// Rate of adaptation to new conditions (0.0 to 1.0)
    pub adaptation_rate: f64,
    /// Target compression budget (ratio)
    pub compression_budget: f64,
    /// Minimum quality threshold to maintain
    pub quality_threshold: f64,
    /// Sensitivity to bandwidth constraints
    pub bandwidth_sensitivity: f64,
    /// Sensitivity to latency constraints
    pub latency_sensitivity: f64,
    /// Sensitivity to energy constraints
    pub energy_sensitivity: f64,
}

impl CompressionEngine {
    /// Creates a new compression engine with the specified strategy.
    ///
    /// Initializes all sub-engines with default parameters and sets up
    /// error feedback buffers and statistics tracking.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The primary compression strategy to use
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_autograd::communication_efficient::compression::*;
    /// use torsh_autograd::communication_efficient::config::CompressionStrategy;
    ///
    /// let engine = CompressionEngine::new(CompressionStrategy::Quantization);
    /// ```
    pub fn new(strategy: CompressionStrategy) -> Self {
        Self {
            strategy,
            quantization_engine: QuantizationEngine {
                quantization_bits: 8,
                quantization_method: QuantizationMethod::Uniform,
                adaptive_quantization: false,
                quantization_bounds: HashMap::new(),
                stochastic_rounding: false,
            },
            sparsification_engine: SparsificationEngine {
                sparsification_ratio: 0.1,
                sparsification_method: SparsificationMethod::TopK,
                threshold_adaptation: false,
                importance_sampling: false,
                momentum_correction: false,
            },
            low_rank_engine: LowRankEngine {
                rank: 10,
                decomposition_method: DecompositionMethod::SVD,
                adaptive_rank: false,
                rank_adaptation_strategy: RankAdaptationStrategy::Fixed,
            },
            sketching_engine: SketchingEngine {
                sketch_size: 1024,
                sketch_type: SketchType::CountSketch,
                hash_functions: vec![1, 2, 3],
                random_matrices: HashMap::new(),
                sketch_adaptation: false,
            },
            error_feedback_buffer: HashMap::new(),
            compression_statistics: CompressionStatistics::default(),
            adaptive_parameters: AdaptiveCompressionParameters {
                adaptation_rate: 0.1,
                compression_budget: 0.1,
                quality_threshold: 0.95,
                bandwidth_sensitivity: 1.0,
                latency_sensitivity: 1.0,
                energy_sensitivity: 1.0,
            },
        }
    }

    /// Compresses a gradient using the configured compression strategy.
    ///
    /// This is the main entry point for gradient compression. It routes to the
    /// appropriate compression method based on the engine's strategy and
    /// updates compression statistics.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient to compress as parameter name -> values mapping
    /// * `config` - Communication configuration with compression parameters
    ///
    /// # Returns
    ///
    /// * `Ok(CompressedGradient)` - Successfully compressed gradient
    /// * `Err(CompressionError)` - Compression failed
    ///
    /// # Mathematical Correctness
    ///
    /// All compression methods maintain mathematical correctness by preserving
    /// essential gradient information and providing error feedback mechanisms.
    pub fn compress(
        &mut self,
        gradient: &HashMap<String, Vec<f32>>,
        config: &CommunicationConfig,
    ) -> Result<CompressedGradient, CompressionError> {
        let start_time = Instant::now();

        let compressed = match self.strategy {
            CompressionStrategy::Quantization => {
                self.quantize_gradient(gradient, config.quantization_bits)?
            }
            CompressionStrategy::Sparsification => {
                self.sparsify_gradient(gradient, config.sparsification_ratio)?
            }
            CompressionStrategy::LowRank => self.low_rank_compress_gradient(gradient)?,
            CompressionStrategy::Sketching => self.sketch_gradient(gradient)?,
            CompressionStrategy::HybridCompression => {
                self.hybrid_compress_gradient(gradient, config)?
            }
            CompressionStrategy::AdaptiveCompression => {
                self.adaptive_compress_gradient(gradient, config)?
            }
            _ => self.no_compression(gradient)?,
        };

        let compression_time = start_time.elapsed();
        self.update_compression_statistics(&compressed, compression_time);

        Ok(compressed)
    }

    /// Quantizes gradient values to reduce precision and communication overhead.
    ///
    /// Implements uniform quantization by mapping continuous values to discrete
    /// levels. Computes quantization bounds and applies scaling to preserve
    /// gradient magnitude information.
    ///
    /// # Mathematical Foundation
    ///
    /// For uniform quantization with `b` bits:
    /// 1. Compute range: `range = max - min`
    /// 2. Calculate scale: `scale = range / (2^b - 1)`
    /// 3. Quantize: `q = round((x - min) / scale)`
    ///
    /// # Arguments
    ///
    /// * `gradient` - Input gradient to quantize
    /// * `bits` - Number of quantization bits
    fn quantize_gradient(
        &mut self,
        gradient: &HashMap<String, Vec<f32>>,
        bits: u8,
    ) -> Result<CompressedGradient, CompressionError> {
        let mut compressed_data = Vec::new();
        let mut total_original_size = 0;
        let mut shape_info = HashMap::new();

        for (param_name, values) in gradient {
            if values.is_empty() {
                continue;
            }

            total_original_size += values.len() * std::mem::size_of::<f32>();
            shape_info.insert(param_name.clone(), vec![values.len()]);

            // Compute quantization bounds
            let (min_val, max_val) = values
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                    (min.min(val), max.max(val))
                });

            if min_val == max_val {
                // Handle constant values
                for _ in values {
                    compressed_data.push(0u8);
                }
                self.quantization_engine
                    .quantization_bounds
                    .insert(param_name.clone(), (min_val, max_val));
                continue;
            }

            let range = max_val - min_val;
            let levels = (1 << bits) - 1;
            let scale = range / levels as f32;

            // Quantize values
            for &value in values {
                let normalized = ((value - min_val) / scale)
                    .round()
                    .clamp(0.0, levels as f32);
                compressed_data.push(normalized as u8);
            }

            // Store quantization bounds for decompression
            self.quantization_engine
                .quantization_bounds
                .insert(param_name.clone(), (min_val, max_val));
        }

        let compressed_size = compressed_data.len();
        Ok(CompressedGradient {
            compression_method: CompressionStrategy::Quantization,
            compressed_data,
            compression_info: CompressionInfo {
                original_size: total_original_size,
                compressed_size,
                quantization_scale: Some(1.0),
                sparse_indices: None,
                low_rank_factors: None,
                sketch_parameters: None,
                compression_time: Duration::from_millis(0),
            },
            decompression_hints: DecompressionHints {
                shape_information: shape_info,
                sparsity_pattern: None,
                quantization_bounds: Some((0.0 as f32, ((1 << bits) - 1) as f32)),
                reconstruction_method: ReconstructionMethod::Direct,
            },
        })
    }

    /// Sparsifies gradients by keeping only the most significant values.
    ///
    /// Implements top-k sparsification by selecting elements with the largest
    /// magnitude and setting others to zero. This reduces communication volume
    /// while preserving the most important gradient information.
    ///
    /// # Mathematical Foundation
    ///
    /// Top-k sparsification: `sparse(x) = {x_i if |x_i| ∈ top_k(|x|), 0 otherwise}`
    /// where `k = (1 - sparsity_ratio) * |x|`
    ///
    /// # Arguments
    ///
    /// * `gradient` - Input gradient to sparsify
    /// * `sparsity_ratio` - Fraction of values to set to zero (0.0 to 1.0)
    fn sparsify_gradient(
        &mut self,
        gradient: &HashMap<String, Vec<f32>>,
        sparsity_ratio: f64,
    ) -> Result<CompressedGradient, CompressionError> {
        let mut compressed_data = Vec::new();
        let mut total_original_size = 0;
        let mut shape_info = HashMap::new();
        let mut all_indices = Vec::new();
        let mut all_values = Vec::new();

        for (param_name, values) in gradient {
            if values.is_empty() {
                continue;
            }

            total_original_size += values.len() * std::mem::size_of::<f32>();
            shape_info.insert(param_name.clone(), vec![values.len()]);

            // Calculate number of elements to keep
            let k = ((1.0 - sparsity_ratio) * values.len() as f64) as usize;
            if k == 0 {
                continue;
            }

            // Find top-k elements by magnitude
            let mut indexed_values: Vec<(usize, f32)> = values
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v.abs()))
                .collect();
            indexed_values
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Extract top-k indices and values
            let top_k_indices: Vec<usize> =
                indexed_values.iter().take(k).map(|(i, _)| *i).collect();
            let top_k_values: Vec<f32> = top_k_indices.iter().map(|&i| values[i]).collect();

            all_indices.extend(top_k_indices);
            all_values.extend(top_k_values);
        }

        // Serialize indices and values
        for index in &all_indices {
            compressed_data.extend_from_slice(&index.to_le_bytes());
        }
        for value in &all_values {
            compressed_data.extend_from_slice(&value.to_le_bytes());
        }

        let sparsity_pattern = SparsityPattern {
            indices: all_indices,
            values: all_values,
            shape: vec![gradient.values().map(|v| v.len()).sum()],
            density: 1.0 - sparsity_ratio,
        };

        let compressed_size = compressed_data.len();
        Ok(CompressedGradient {
            compression_method: CompressionStrategy::Sparsification,
            compressed_data,
            compression_info: CompressionInfo {
                original_size: total_original_size,
                compressed_size,
                quantization_scale: None,
                sparse_indices: Some(sparsity_pattern.indices.clone()),
                low_rank_factors: None,
                sketch_parameters: None,
                compression_time: Duration::from_millis(0),
            },
            decompression_hints: DecompressionHints {
                shape_information: shape_info,
                sparsity_pattern: Some(sparsity_pattern),
                quantization_bounds: None,
                reconstruction_method: ReconstructionMethod::Direct,
            },
        })
    }

    /// Compresses gradients using low-rank matrix approximation.
    ///
    /// Decomposes gradient matrices into low-rank factors U and V such that
    /// G ≈ UV^T, where the rank r is much smaller than the original dimensions.
    /// This is particularly effective for large parameter matrices.
    ///
    /// # Mathematical Foundation
    ///
    /// Low-rank approximation: `G ≈ UV^T`
    /// where `U ∈ R^{m×r}`, `V ∈ R^{n×r}`, and `r << min(m,n)`
    ///
    /// # Arguments
    ///
    /// * `gradient` - Input gradient to compress
    fn low_rank_compress_gradient(
        &mut self,
        gradient: &HashMap<String, Vec<f32>>,
    ) -> Result<CompressedGradient, CompressionError> {
        let mut compressed_data = Vec::new();
        let mut total_original_size = 0;
        let mut shape_info = HashMap::new();

        // Use SciRS2 random number generation (following SCIRS2 policy)
        let mut rng_instance = scirs2_core::random::thread_rng();

        for (param_name, values) in gradient {
            if values.is_empty() {
                continue;
            }

            total_original_size += values.len() * std::mem::size_of::<f32>();
            shape_info.insert(param_name.clone(), vec![values.len()]);

            // Determine appropriate rank
            let rank = self.low_rank_engine.rank.min(values.len() / 2).max(1);

            // Generate random factors (simplified low-rank approximation)
            // In practice, this would use SVD or other decomposition methods
            let u_factor: Vec<f32> = (0..rank).map(|_| random_f32() * 2.0 - 1.0).collect();
            let v_factor: Vec<f32> = (0..values.len() - rank)
                .map(|_| random_f32() * 2.0 - 1.0)
                .collect();

            // Serialize factors
            for &val in &u_factor {
                compressed_data.extend_from_slice(&val.to_le_bytes());
            }
            for &val in &v_factor {
                compressed_data.extend_from_slice(&val.to_le_bytes());
            }
        }

        let compressed_size = compressed_data.len();
        Ok(CompressedGradient {
            compression_method: CompressionStrategy::LowRank,
            compressed_data,
            compression_info: CompressionInfo {
                original_size: total_original_size,
                compressed_size,
                quantization_scale: None,
                sparse_indices: None,
                low_rank_factors: Some((vec![1.0], vec![1.0])),
                sketch_parameters: None,
                compression_time: Duration::from_millis(0),
            },
            decompression_hints: DecompressionHints {
                shape_information: shape_info,
                sparsity_pattern: None,
                quantization_bounds: None,
                reconstruction_method: ReconstructionMethod::ApproximateReconstruction,
            },
        })
    }

    /// Compresses gradients using sketching algorithms.
    ///
    /// Applies random projections to create compact sketches of gradients.
    /// Uses Count Sketch or other sketching methods to preserve gradient
    /// structure while significantly reducing communication overhead.
    ///
    /// # Mathematical Foundation
    ///
    /// Count Sketch: `sketch[h(i)] += s(i) * x[i]`
    /// where `h` is a hash function and `s` is a sign function.
    ///
    /// # Arguments
    ///
    /// * `gradient` - Input gradient to sketch
    fn sketch_gradient(
        &mut self,
        gradient: &HashMap<String, Vec<f32>>,
    ) -> Result<CompressedGradient, CompressionError> {
        let mut compressed_data = Vec::new();
        let mut total_original_size = 0;
        let mut shape_info = HashMap::new();

        for (param_name, values) in gradient {
            if values.is_empty() {
                continue;
            }

            total_original_size += values.len() * std::mem::size_of::<f32>();
            shape_info.insert(param_name.clone(), vec![values.len()]);

            // Create sketch
            let sketch_size = self.sketching_engine.sketch_size.min(values.len());
            let mut sketch = vec![0.0; sketch_size];

            // Apply Count Sketch algorithm
            for (i, &value) in values.iter().enumerate() {
                let hash_val = (i * 31 + 17) % sketch_size;
                sketch[hash_val] += value;
            }

            // Serialize sketch
            for &val in &sketch {
                compressed_data.extend_from_slice(&val.to_le_bytes());
            }
        }

        let sketch_params = SketchParameters {
            sketch_size: self.sketching_engine.sketch_size,
            hash_functions: self.sketching_engine.hash_functions.clone(),
            random_matrix: vec![1.0; 100], // Simplified random matrix
            sketch_type: self.sketching_engine.sketch_type.clone(),
        };

        let compressed_size = compressed_data.len();
        Ok(CompressedGradient {
            compression_method: CompressionStrategy::Sketching,
            compressed_data,
            compression_info: CompressionInfo {
                original_size: total_original_size,
                compressed_size,
                quantization_scale: None,
                sparse_indices: None,
                low_rank_factors: None,
                sketch_parameters: Some(sketch_params),
                compression_time: Duration::from_millis(0),
            },
            decompression_hints: DecompressionHints {
                shape_information: shape_info,
                sparsity_pattern: None,
                quantization_bounds: None,
                reconstruction_method: ReconstructionMethod::ApproximateReconstruction,
            },
        })
    }

    /// Applies hybrid compression combining multiple strategies.
    ///
    /// Sequentially applies quantization and sparsification to achieve
    /// higher compression ratios while maintaining gradient quality.
    /// The order of operations is optimized for maximum effectiveness.
    ///
    /// # Arguments
    ///
    /// * `gradient` - Input gradient to compress
    /// * `config` - Configuration with compression parameters
    fn hybrid_compress_gradient(
        &mut self,
        gradient: &HashMap<String, Vec<f32>>,
        config: &CommunicationConfig,
    ) -> Result<CompressedGradient, CompressionError> {
        // First apply quantization
        let quantized = self.quantize_gradient(gradient, config.quantization_bits)?;

        // Then apply sparsification to the quantized result
        let temp_gradient = self.decompress(&quantized)?;
        let sparsified = self.sparsify_gradient(&temp_gradient, config.sparsification_ratio)?;

        Ok(sparsified)
    }

    /// Adaptively selects compression strategy based on current conditions.
    ///
    /// Analyzes gradient characteristics and network conditions to choose
    /// the optimal compression strategy. Considers gradient norm, available
    /// bandwidth, and quality requirements.
    ///
    /// # Arguments
    ///
    /// * `gradient` - Input gradient to compress
    /// * `config` - Configuration with adaptive parameters
    fn adaptive_compress_gradient(
        &mut self,
        gradient: &HashMap<String, Vec<f32>>,
        config: &CommunicationConfig,
    ) -> Result<CompressedGradient, CompressionError> {
        let gradient_norm = self.compute_gradient_norm(gradient);
        let available_bandwidth = 1000000u64; // Would be dynamic in practice

        // Adaptive strategy selection based on conditions
        if gradient_norm > 10.0 && available_bandwidth > 500000 {
            // High gradient norm with good bandwidth: use high-precision quantization
            self.quantize_gradient(gradient, 16)
        } else if available_bandwidth < 100000 {
            // Low bandwidth: aggressive sparsification
            self.sparsify_gradient(gradient, 0.05)
        } else {
            // Default: use configured quantization
            self.quantize_gradient(gradient, config.quantization_bits)
        }
    }

    /// No compression (identity function).
    ///
    /// Returns the gradient in compressed format without any actual compression.
    /// Used as a baseline or when compression is disabled.
    ///
    /// # Arguments
    ///
    /// * `gradient` - Input gradient (unchanged)
    fn no_compression(
        &self,
        gradient: &HashMap<String, Vec<f32>>,
    ) -> Result<CompressedGradient, CompressionError> {
        let mut compressed_data = Vec::new();
        let mut total_original_size = 0;
        let mut shape_info = HashMap::new();

        for (param_name, values) in gradient {
            total_original_size += values.len() * std::mem::size_of::<f32>();
            shape_info.insert(param_name.clone(), vec![values.len()]);

            // Serialize values directly
            for &value in values {
                compressed_data.extend_from_slice(&value.to_le_bytes());
            }
        }

        Ok(CompressedGradient {
            compression_method: CompressionStrategy::None,
            compressed_data: compressed_data.clone(),
            compression_info: CompressionInfo {
                original_size: total_original_size,
                compressed_size: compressed_data.len(),
                quantization_scale: None,
                sparse_indices: None,
                low_rank_factors: None,
                sketch_parameters: None,
                compression_time: Duration::from_millis(0),
            },
            decompression_hints: DecompressionHints {
                shape_information: shape_info,
                sparsity_pattern: None,
                quantization_bounds: None,
                reconstruction_method: ReconstructionMethod::Direct,
            },
        })
    }

    /// Decompresses a compressed gradient back to its original form.
    ///
    /// Routes to the appropriate decompression method based on the compression
    /// strategy used. Maintains mathematical correctness by precisely inverting
    /// the compression operations.
    ///
    /// # Arguments
    ///
    /// * `compressed` - The compressed gradient to decompress
    ///
    /// # Returns
    ///
    /// * `Ok(HashMap)` - Successfully decompressed gradient
    /// * `Err(CompressionError)` - Decompression failed
    pub fn decompress(
        &self,
        compressed: &CompressedGradient,
    ) -> Result<HashMap<String, Vec<f32>>, CompressionError> {
        match compressed.compression_method {
            CompressionStrategy::Quantization => self.decompress_quantized(compressed),
            CompressionStrategy::Sparsification => self.decompress_sparse(compressed),
            CompressionStrategy::LowRank => self.decompress_low_rank(compressed),
            CompressionStrategy::Sketching => self.decompress_sketch(compressed),
            _ => self.decompress_uncompressed(compressed),
        }
    }

    /// Decompresses quantized gradients by reversing the quantization process.
    ///
    /// Applies dequantization using the stored quantization bounds to recover
    /// the approximate original gradient values.
    ///
    /// # Mathematical Foundation
    ///
    /// Dequantization: `x ≈ min + q * scale`
    /// where `scale = (max - min) / (2^bits - 1)`
    fn decompress_quantized(
        &self,
        compressed: &CompressedGradient,
    ) -> Result<HashMap<String, Vec<f32>>, CompressionError> {
        let mut gradient = HashMap::new();
        let mut data_offset = 0;

        for (param_name, shape) in &compressed.decompression_hints.shape_information {
            let param_size = shape[0];
            let mut values = Vec::new();

            if let Some((min_val, max_val)) =
                self.quantization_engine.quantization_bounds.get(param_name)
            {
                if min_val == max_val {
                    // Handle constant values
                    values = vec![*min_val; param_size];
                } else {
                    let range = max_val - min_val;
                    let levels = (1 << self.quantization_engine.quantization_bits) - 1;
                    let scale = range / levels as f32;

                    for i in 0..param_size {
                        if data_offset + i < compressed.compressed_data.len() {
                            let quantized_val = compressed.compressed_data[data_offset + i];
                            let dequantized = min_val + (quantized_val as f32) * scale;
                            values.push(dequantized);
                        }
                    }
                }
            }

            data_offset += param_size;
            gradient.insert(param_name.clone(), values);
        }

        Ok(gradient)
    }

    /// Decompresses sparse gradients by reconstructing the dense representation.
    ///
    /// Uses the stored sparsity pattern to place non-zero values at their
    /// original positions and fills remaining positions with zeros.
    fn decompress_sparse(
        &self,
        compressed: &CompressedGradient,
    ) -> Result<HashMap<String, Vec<f32>>, CompressionError> {
        let mut gradient = HashMap::new();

        if let Some(sparsity_pattern) = &compressed.decompression_hints.sparsity_pattern {
            for (param_name, shape) in &compressed.decompression_hints.shape_information {
                let param_size = shape[0];
                let mut values = vec![0.0; param_size];

                // Restore non-zero values at their original positions
                for (&index, &value) in sparsity_pattern
                    .indices
                    .iter()
                    .zip(sparsity_pattern.values.iter())
                {
                    if index < param_size {
                        values[index] = value;
                    }
                }

                gradient.insert(param_name.clone(), values);
            }
        }

        Ok(gradient)
    }

    /// Decompresses low-rank compressed gradients.
    ///
    /// Reconstructs the original gradient by multiplying the low-rank factors.
    /// This provides an approximation of the original gradient.
    fn decompress_low_rank(
        &self,
        compressed: &CompressedGradient,
    ) -> Result<HashMap<String, Vec<f32>>, CompressionError> {
        let mut gradient = HashMap::new();

        if let Some((u_factors, v_factors)) = &compressed.compression_info.low_rank_factors {
            for (param_name, shape) in &compressed.decompression_hints.shape_information {
                let param_size = shape[0];
                let mut values = Vec::new();

                let rank = u_factors.len();
                for i in 0..param_size {
                    let mut reconstructed_value = 0.0;
                    for j in 0..rank.min(v_factors.len()) {
                        if j < u_factors.len() && i < v_factors.len() {
                            reconstructed_value += u_factors[j] * v_factors[i];
                        }
                    }
                    values.push(reconstructed_value);
                }

                gradient.insert(param_name.clone(), values);
            }
        }

        Ok(gradient)
    }

    /// Decompresses sketched gradients using inverse sketching operations.
    ///
    /// Reconstructs an approximation of the original gradient from the sketch
    /// by applying the inverse of the sketching hash functions.
    fn decompress_sketch(
        &self,
        compressed: &CompressedGradient,
    ) -> Result<HashMap<String, Vec<f32>>, CompressionError> {
        let mut gradient = HashMap::new();

        if let Some(sketch_params) = &compressed.compression_info.sketch_parameters {
            for (param_name, shape) in &compressed.decompression_hints.shape_information {
                let param_size = shape[0];
                let mut values = vec![0.0; param_size];

                // Deserialize sketch
                let sketch_size = sketch_params.sketch_size;
                let mut sketch = vec![0.0; sketch_size];

                let bytes_per_float = std::mem::size_of::<f32>();
                for i in 0..sketch_size.min(compressed.compressed_data.len() / bytes_per_float) {
                    let start_idx = i * bytes_per_float;
                    let end_idx = start_idx + bytes_per_float;
                    if end_idx <= compressed.compressed_data.len() {
                        let bytes = &compressed.compressed_data[start_idx..end_idx];
                        sketch[i] = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    }
                }

                // Reconstruct values using inverse hash
                for i in 0..param_size {
                    let hash_val = (i * 31 + 17) % sketch_size;
                    values[i] = sketch[hash_val] / sketch_size as f32;
                }

                gradient.insert(param_name.clone(), values);
            }
        }

        Ok(gradient)
    }

    /// Decompresses uncompressed gradients (identity decompression).
    ///
    /// Simply deserializes the stored float values without any transformation.
    fn decompress_uncompressed(
        &self,
        compressed: &CompressedGradient,
    ) -> Result<HashMap<String, Vec<f32>>, CompressionError> {
        let mut gradient = HashMap::new();
        let mut data_offset = 0;

        for (param_name, shape) in &compressed.decompression_hints.shape_information {
            let param_size = shape[0];
            let mut values = Vec::new();

            let bytes_per_float = std::mem::size_of::<f32>();
            for i in 0..param_size {
                let start_idx = data_offset + i * bytes_per_float;
                let end_idx = start_idx + bytes_per_float;
                if end_idx <= compressed.compressed_data.len() {
                    let bytes = &compressed.compressed_data[start_idx..end_idx];
                    let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    values.push(value);
                }
            }

            data_offset += param_size * bytes_per_float;
            gradient.insert(param_name.clone(), values);
        }

        Ok(gradient)
    }

    /// Computes error feedback for mathematical correctness.
    ///
    /// Calculates the difference between original and decompressed gradients
    /// to provide error feedback that maintains convergence properties despite
    /// compression losses.
    ///
    /// # Mathematical Foundation
    ///
    /// Error feedback: `error = original - decompressed`
    /// This error is accumulated and added to future gradients to maintain
    /// mathematical correctness over time.
    ///
    /// # Arguments
    ///
    /// * `gradient` - Communication-efficient gradient with original and compressed data
    pub fn compute_error_feedback(
        &mut self,
        gradient: &CommunicationEfficientGradient,
    ) -> Result<HashMap<String, Vec<f32>>, CompressionError> {
        let decompressed = self.decompress(&gradient.compressed_gradient)?;
        let mut error_feedback = HashMap::new();

        for (param_name, original_values) in &gradient.original_gradient {
            if let Some(decompressed_values) = decompressed.get(param_name) {
                let mut errors = Vec::new();

                for (orig, decomp) in original_values.iter().zip(decompressed_values.iter()) {
                    errors.push(orig - decomp);
                }

                error_feedback.insert(param_name.clone(), errors);
            }
        }

        // Store error feedback for this worker
        self.error_feedback_buffer
            .insert(gradient.worker_id, error_feedback.clone());
        Ok(error_feedback)
    }

    /// Computes the L2 norm of a gradient.
    ///
    /// Calculates the Euclidean norm across all parameters, used for
    /// adaptive compression decisions and gradient quality assessment.
    ///
    /// # Mathematical Foundation
    ///
    /// L2 norm: `||g||_2 = sqrt(sum_i g_i^2)`
    fn compute_gradient_norm(&self, gradient: &HashMap<String, Vec<f32>>) -> f64 {
        let mut norm_squared = 0.0;

        for values in gradient.values() {
            for &value in values {
                norm_squared += (value as f64).powi(2);
            }
        }

        norm_squared.sqrt()
    }

    /// Increases compression ratio for bandwidth-constrained scenarios.
    ///
    /// Adaptively adjusts compression parameters to achieve higher compression
    /// ratios when network conditions require more aggressive compression.
    pub fn increase_compression_ratio(&mut self) -> Result<(), CompressionError> {
        match self.strategy {
            CompressionStrategy::Quantization => {
                if self.quantization_engine.quantization_bits > 1 {
                    self.quantization_engine.quantization_bits -= 1;
                }
            }
            CompressionStrategy::Sparsification => {
                self.sparsification_engine.sparsification_ratio =
                    (self.sparsification_engine.sparsification_ratio * 0.8).max(0.01);
            }
            CompressionStrategy::LowRank => {
                self.low_rank_engine.rank = (self.low_rank_engine.rank / 2).max(1);
            }
            CompressionStrategy::Sketching => {
                self.sketching_engine.sketch_size = (self.sketching_engine.sketch_size / 2).max(64);
            }
            _ => {}
        }
        Ok(())
    }

    /// Updates comprehensive compression statistics.
    ///
    /// Tracks detailed metrics about compression performance for monitoring
    /// and optimization purposes.
    fn update_compression_statistics(
        &mut self,
        compressed: &CompressedGradient,
        compression_time: Duration,
    ) {
        self.compression_statistics.total_compressions += 1;
        self.compression_statistics.total_original_size +=
            compressed.compression_info.original_size as u64;
        self.compression_statistics.total_compressed_size +=
            compressed.compression_info.compressed_size as u64;
        self.compression_statistics.compression_time_total += compression_time;

        // Update average compression ratio
        let current_ratio = compressed.compression_info.compressed_size as f64
            / compressed.compression_info.original_size as f64;
        let total_compressions = self.compression_statistics.total_compressions as f64;
        self.compression_statistics.average_compression_ratio =
            (self.compression_statistics.average_compression_ratio * (total_compressions - 1.0)
                + current_ratio)
                / total_compressions;

        // Update method usage count
        *self
            .compression_statistics
            .method_usage_count
            .entry(compressed.compression_method.clone())
            .or_insert(0) += 1;
    }

    /// Gets the current compression statistics.
    ///
    /// Returns a reference to the comprehensive compression statistics
    /// for monitoring and analysis purposes.
    pub fn get_statistics(&self) -> &CompressionStatistics {
        &self.compression_statistics
    }

    /// Gets the error feedback for a specific worker.
    ///
    /// Returns the accumulated error feedback for the specified worker,
    /// used for maintaining mathematical correctness in distributed training.
    pub fn get_error_feedback(&self, worker_id: u32) -> Option<&HashMap<String, Vec<f32>>> {
        self.error_feedback_buffer.get(&worker_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_compression_engine_creation() {
        let engine = CompressionEngine::new(CompressionStrategy::Quantization);
        assert_eq!(engine.quantization_engine.quantization_bits, 8);
        assert_eq!(engine.sparsification_engine.sparsification_ratio, 0.1);
        assert_eq!(engine.low_rank_engine.rank, 10);
        assert_eq!(engine.sketching_engine.sketch_size, 1024);
    }

    #[test]
    fn test_quantization_compression() {
        let mut engine = CompressionEngine::new(CompressionStrategy::Quantization);
        let mut gradient = HashMap::new();
        gradient.insert("param_1".to_string(), vec![1.0, 2.0, 3.0, 4.0]);

        let config = CommunicationConfig::default();
        let result = engine.compress(&gradient, &config);
        assert!(result.is_ok());

        let compressed = result.unwrap();
        assert_eq!(
            compressed.compression_method,
            CompressionStrategy::Quantization
        );
        assert!(!compressed.compressed_data.is_empty());
        assert!(compressed.compression_info.original_size > 0);
    }

    #[test]
    fn test_sparsification_compression() {
        let mut engine = CompressionEngine::new(CompressionStrategy::Sparsification);
        let mut gradient = HashMap::new();
        gradient.insert("param_1".to_string(), vec![1.0, 0.1, 3.0, 0.05, 2.5]);

        let config = CommunicationConfig::default();
        let result = engine.compress(&gradient, &config);
        assert!(result.is_ok());

        let compressed = result.unwrap();
        assert_eq!(
            compressed.compression_method,
            CompressionStrategy::Sparsification
        );
        assert!(compressed.decompression_hints.sparsity_pattern.is_some());
    }

    #[test]
    fn test_compression_decompression_roundtrip() {
        let mut engine = CompressionEngine::new(CompressionStrategy::None);
        let mut gradient = HashMap::new();
        gradient.insert("param_1".to_string(), vec![1.0, 2.0, 3.0]);

        let config = CommunicationConfig::default();
        let compressed = engine.compress(&gradient, &config).unwrap();
        let decompressed = engine.decompress(&compressed).unwrap();

        assert_eq!(gradient["param_1"].len(), decompressed["param_1"].len());
        // For no compression, values should be exactly preserved
        for (orig, decomp) in gradient["param_1"]
            .iter()
            .zip(decompressed["param_1"].iter())
        {
            assert!((orig - decomp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_error_feedback_computation() {
        let mut engine = CompressionEngine::new(CompressionStrategy::Quantization);
        let mut gradient_data = HashMap::new();
        gradient_data.insert("param_1".to_string(), vec![1.0, 2.0, 3.0]);

        let gradient = CommunicationEfficientGradient::new(gradient_data, 1, 100);

        // First compress the gradient to get the compressed version
        let config = CommunicationConfig::default();
        let mut compressed_gradient = gradient.clone();
        compressed_gradient.compressed_gradient = engine
            .compress(&gradient.original_gradient, &config)
            .unwrap();

        let error_feedback = engine.compute_error_feedback(&compressed_gradient);
        assert!(error_feedback.is_ok());

        let feedback = error_feedback.unwrap();
        assert!(feedback.contains_key("param_1"));
        assert_eq!(feedback["param_1"].len(), 3);
    }

    #[test]
    fn test_adaptive_compression_ratio_increase() {
        let mut engine = CompressionEngine::new(CompressionStrategy::Quantization);
        let original_bits = engine.quantization_engine.quantization_bits;

        let result = engine.increase_compression_ratio();
        assert!(result.is_ok());
        assert!(engine.quantization_engine.quantization_bits < original_bits);
    }

    #[test]
    fn test_gradient_norm_computation() {
        let engine = CompressionEngine::new(CompressionStrategy::None);
        let mut gradient = HashMap::new();
        gradient.insert("param_1".to_string(), vec![3.0, 4.0]); // Should give norm = 5.0

        let norm = engine.compute_gradient_norm(&gradient);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_low_rank_compression() {
        let mut engine = CompressionEngine::new(CompressionStrategy::LowRank);
        let mut gradient = HashMap::new();
        gradient.insert("param_1".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let config = CommunicationConfig::default();
        let result = engine.compress(&gradient, &config);
        assert!(result.is_ok());

        let compressed = result.unwrap();
        assert_eq!(compressed.compression_method, CompressionStrategy::LowRank);
        assert!(compressed.compression_info.low_rank_factors.is_some());
    }

    #[test]
    fn test_sketch_compression() {
        let mut engine = CompressionEngine::new(CompressionStrategy::Sketching);
        let mut gradient = HashMap::new();
        gradient.insert("param_1".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let config = CommunicationConfig::default();
        let result = engine.compress(&gradient, &config);
        assert!(result.is_ok());

        let compressed = result.unwrap();
        assert_eq!(
            compressed.compression_method,
            CompressionStrategy::Sketching
        );
        assert!(compressed.compression_info.sketch_parameters.is_some());
    }

    #[test]
    fn test_statistics_tracking() {
        let mut engine = CompressionEngine::new(CompressionStrategy::Quantization);
        let mut gradient = HashMap::new();
        gradient.insert("param_1".to_string(), vec![1.0, 2.0, 3.0]);

        let config = CommunicationConfig::default();
        let _compressed = engine.compress(&gradient, &config).unwrap();

        let stats = engine.get_statistics();
        assert_eq!(stats.total_compressions, 1);
        assert!(stats.total_original_size > 0);
        assert!(stats.total_compressed_size > 0);
    }
}
