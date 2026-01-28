//! Sparse tensor operations for ToRSh
//!
//! This crate provides comprehensive sparse tensor representations and operations,
//! supporting multiple sparse formats including COO, CSR, CSC, BSR, DIA, ELL, DSR, and RLE.
//!
//! ## Key Features
//!
//! - **Multiple Sparse Formats**: Support for COO, CSR, CSC, BSR, DIA, ELL, DSR, and RLE formats
//! - **Automatic Format Selection**: Intelligent format selection based on sparsity patterns
//! - **Neural Network Integration**: Sparse layers, optimizers, and activation functions
//! - **GPU Acceleration**: CUDA support for sparse operations
//! - **Memory Management**: Advanced memory pooling and optimization
//! - **Interoperability**: Integration with SciPy, MATLAB, and HDF5
//! - **Performance Tools**: Profiling, benchmarking, and autotuning capabilities
//!
//! ## Usage Examples
//!
//! ```rust,no_run
//! use torsh_sparse::{CooTensor, CsrTensor, SparseFormat};
//! use torsh_core::Shape;
//!
//! // Create a COO tensor from triplets
//! let triplets = vec![(0, 0, 1.0f32), (1, 1, 2.0f32), (2, 2, 3.0f32)];
//! let coo = CooTensor::from_triplets(triplets, (3, 3)).unwrap();
//!
//! // Convert to CSR format for efficient row operations
//! let csr = coo.to_csr().unwrap();
//!
//! // Perform sparse operations
//! let result = csr.transpose().unwrap();
//! ```
//!
//! ## Performance Considerations
//!
//! - **COO**: Best for construction and format conversion
//! - **CSR**: Optimized for row-based operations and matrix-vector multiplication
//! - **CSC**: Optimized for column-based operations
//! - **BSR**: Efficient for block-structured sparse matrices
//! - **DIA**: Memory-efficient for diagonal-dominant matrices
//! - **ELL**: SIMD-friendly format for GPU operations

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

use std::collections::HashMap;
use torsh_core::{DType, DeviceType, Result, Shape, TorshError};
use torsh_tensor::Tensor;

/// Convenience type alias for Results in this crate
pub type TorshResult<T> = Result<T>;

pub mod autograd;
pub mod bsr;
pub mod conversions;
pub mod coo;
pub mod csc;
pub mod csr;
pub mod custom_kernels;
pub mod dia;
pub mod dsr;
pub mod ell;
pub mod gpu;
pub mod hdf5_support;
pub mod hybrid;
pub mod layers;
pub mod linalg;
pub mod matlab_compat;
pub mod matrix_market;
pub mod memory_management;
pub mod nn;
pub mod ops;
pub mod optimizers;
pub mod pattern_analysis;
pub mod performance_tools;
pub mod rle;
pub mod scipy_sparse;
// pub mod scirs2_integration; // temporarily disabled due to import issues

// Enhanced SciRS2 integration
#[cfg(feature = "scirs2-integration")]
pub mod scirs2_sparse_integration;
pub mod symmetric;
pub mod unified_interface;

// Re-exports
pub use bsr::BsrTensor;
pub use coo::CooTensor;
pub use csc::CscTensor;
pub use csr::CsrTensor;
pub use dia::DiaTensor;
pub use dsr::DsrTensor;
pub use ell::EllTensor;
pub use rle::RleTensor;
pub use symmetric::{SymmetricMode, SymmetricTensor};

// GPU support
pub use gpu::{CudaSparseOps, CudaSparseTensor, CudaSparseTensorFactory};

// Autograd support
pub use autograd::{SparseAutogradTensor, SparseData, SparseGradFn, SparseGradientAccumulator};

// SciRS2 integration (temporarily disabled due to import issues)
// pub use scirs2_integration::{scirs2_add, scirs2_enhanced_ops};

// Enhanced SciRS2 sparse integration
#[cfg(feature = "scirs2-integration")]
pub use scirs2_sparse_integration::{
    create_gpu_sparse_processor, create_nn_sparse_processor, create_sparse_processor,
    SciRS2SparseProcessor, SparseConfig as ScirsSparseConfig,
};

// Neural network layers and optimizers
pub use nn::{
    // Type aliases for convenience
    Format,
    GraphConvolution,
    InitConfig,
    LayerConfig,
    SparseAdam,
    SparseAttention,
    // Advanced layer implementations
    SparseConv2d,
    SparseConverter,
    SparseEmbedding,
    SparseEmbeddingStats,
    // Configuration types
    // SparseFormat, // Defined locally to avoid conflict
    SparseInitConfig,
    SparseLayer,
    SparseLayerConfig,
    // Layer implementations
    SparseLinear,
    SparseMemoryStats,
    // Core traits
    SparseOptimizer,
    SparsePatternAnalysis,
    // Optimizers
    SparseSGD,
    SparseStats,
    // Utilities
    SparseWeightGenerator,
};

// Hybrid formats and utilities
pub use hybrid::{auto_select_format, HybridTensor, PartitionStrategy, SparsityPattern};

// Pattern analysis utilities
pub use pattern_analysis::{
    AdvancedSparsityPattern, ClusteringAlgorithm, MatrixReorderer, PatternAnalyzer,
    PatternStatistics, PatternVisualizer, ReorderingAlgorithm,
};

// Performance tools
pub use performance_tools::{
    AutoTuner, BenchmarkConfig, CachePerformanceResult, HardwareBenchmark, MemoryAnalysis,
    OperationStatistics, PerformanceExporter, PerformanceMeasurement, PerformanceReport, PlotData,
    SparseProfiler, SystemInfo, TensorBoardExporter, TrendAnalysis, TrendAnalyzer, TrendDirection,
};

// Matrix Market I/O
pub use matrix_market::{
    MatrixMarketField, MatrixMarketFormat, MatrixMarketHeader, MatrixMarketIO, MatrixMarketObject,
    MatrixMarketSize, MatrixMarketSymmetry, MatrixMarketUtils,
};

// Custom optimized kernels
pub use custom_kernels::{
    ElementWiseKernels, FormatConversionKernels, KernelDispatcher, ReductionKernels,
    SparseMatMulKernels,
};

// SciPy sparse interoperability
pub use scipy_sparse::{ScipyFormat, ScipySparseData, ScipySparseIntegration};

// MATLAB sparse interoperability
pub use matlab_compat::{
    export_to_matlab_script, matlab_sparse_from_triplets, MatlabSparseCompat, MatlabSparseMatrix,
};

// HDF5 sparse interoperability
pub use hdf5_support::{load_sparse_matrix, save_sparse_matrix, Hdf5SparseIO, Hdf5SparseMetadata};

// Unified sparse tensor interface
pub use unified_interface::{
    AccessPatterns, MemoryStats, OptimizationConfig, OptimizationFlags, OptimizationReport,
    PerformanceHints, PerformanceSummary, TensorMetadata, UnifiedSparseTensor,
    UnifiedSparseTensorFactory,
};

// Memory management
pub use memory_management::{
    create_sparse_with_memory_management, MemoryAwareSparseBuilder, MemoryPoolConfig, MemoryReport,
    MemoryStatistics, SparseMemoryHandle, SparseMemoryManager, SparseMemoryPool,
};

// Conversion utilities
pub use conversions::{direct_conversions, optimization, patterns, validation, ConversionHints};

/// Layout format for sparse tensors
///
/// Different sparse formats are optimized for different use cases and access patterns.
/// Choose the appropriate format based on your matrix characteristics and operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseFormat {
    /// Coordinate format (COO) - stores (row, col, value) triplets
    ///
    /// **Best for**: Matrix construction, format conversion, random insertion
    /// **Memory**: 3 * nnz storage (row indices, col indices, values)
    /// **Operations**: Efficient addition, inefficient matrix-vector multiplication
    Coo,

    /// Compressed Sparse Row (CSR) - row-oriented compressed format
    ///
    /// **Best for**: Matrix-vector multiplication, row slicing, iterating by rows
    /// **Memory**: (nnz + n + 1) storage (values, col_indices, row_ptr)
    /// **Operations**: Fast row access, efficient SpMV, slow column access
    Csr,

    /// Compressed Sparse Column (CSC) - column-oriented compressed format
    ///
    /// **Best for**: Matrix-vector multiplication (A^T * x), column slicing
    /// **Memory**: (nnz + m + 1) storage (values, row_indices, col_ptr)
    /// **Operations**: Fast column access, efficient transpose operations
    Csc,

    /// Block Sparse Row (BSR) - stores dense blocks in sparse locations
    ///
    /// **Best for**: Matrices with dense block structure, finite element methods
    /// **Memory**: Efficient for matrices with natural block structure
    /// **Operations**: BLAS-optimized operations on dense blocks
    Bsr,

    /// Diagonal format (DIA) - stores diagonals efficiently
    ///
    /// **Best for**: Matrices with few non-zero diagonals, finite difference schemes
    /// **Memory**: Very compact for diagonal-dominant matrices
    /// **Operations**: Fast diagonal operations, limited to diagonal patterns
    Dia,

    /// Dynamic Sparse Row (DSR) - dynamic insertion/deletion support
    ///
    /// **Best for**: Matrices that change structure frequently during computation
    /// **Memory**: Tree-based storage allows dynamic modifications
    /// **Operations**: Efficient insertion/deletion, slower than static formats
    Dsr,

    /// ELLPACK format (ELL) - fixed-width row storage
    ///
    /// **Best for**: GPU operations, SIMD vectorization, matrices with uniform row density
    /// **Memory**: Can have significant overhead for irregular matrices
    /// **Operations**: SIMD-friendly, efficient on parallel architectures
    Ell,

    /// Run-Length Encoded format (RLE) - compresses consecutive zeros
    ///
    /// **Best for**: Matrices with long runs of consecutive non-zeros
    /// **Memory**: Excellent compression for specific patterns
    /// **Operations**: Specialized for pattern-specific matrices
    Rle,

    /// Symmetric sparse format (SYM) - stores only lower/upper triangle
    ///
    /// **Best for**: Symmetric matrices from finite element analysis, optimization
    /// **Memory**: Roughly half the storage of equivalent full format
    /// **Operations**: Specialized symmetric operations, automatic symmetry enforcement
    Symmetric,
}

/// Trait for sparse tensor operations
///
/// This trait provides a unified interface for all sparse tensor formats,
/// enabling polymorphic operations and seamless format conversions.
///
/// ## Implementation Notes
///
/// All sparse tensor types implement this trait, allowing for:
/// - Format-agnostic operations through trait objects
/// - Automatic format selection based on operation requirements
/// - Efficient conversion between different sparse representations
///
/// ## Example
///
/// ```rust,no_run
/// use torsh_sparse::{SparseTensor, CooTensor};
///
/// fn analyze_sparsity(tensor: &dyn SparseTensor) -> f32 {
///     tensor.sparsity()
/// }
///
/// let coo = CooTensor::from_triplets(vec![(0, 0, 1.0)], (10, 10)).unwrap();
/// println!("Sparsity: {:.2}%", analyze_sparsity(&coo) * 100.0);
/// ```
pub trait SparseTensor {
    /// Get the sparse format used by this tensor
    ///
    /// Returns the specific sparse format (COO, CSR, CSC, etc.) that this tensor uses internally.
    /// This can be used for format-specific optimizations or debugging.
    fn format(&self) -> SparseFormat;

    /// Get the shape of the tensor
    ///
    /// Returns a reference to the tensor's shape, which describes its dimensions.
    /// For sparse matrices, this is typically a 2D shape [rows, cols].
    fn shape(&self) -> &Shape;

    /// Get the data type of the tensor elements
    ///
    /// Returns the DType (typically F32 for f32 values) used to store the non-zero elements.
    fn dtype(&self) -> DType;

    /// Get the device where the tensor is stored
    ///
    /// Returns the device type (CPU, CUDA, etc.) where the tensor data resides.
    fn device(&self) -> DeviceType;

    /// Get the number of non-zero elements
    ///
    /// Returns the count of explicitly stored non-zero values. Note that this may include
    /// some actual zeros that are explicitly stored in the sparse representation.
    fn nnz(&self) -> usize;

    /// Convert to dense tensor representation
    ///
    /// Creates a full dense tensor with all zeros filled in. This can be memory-intensive
    /// for large sparse matrices. Use with caution for matrices with high dimensions.
    ///
    /// # Performance Note
    /// This operation has O(m*n) memory complexity and should be avoided for large matrices.
    fn to_dense(&self) -> TorshResult<Tensor>;

    /// Convert to COO (Coordinate) format
    ///
    /// Converts to COO format, which stores explicit (row, col, value) triplets.
    /// This is useful for format conversion and matrix construction operations.
    fn to_coo(&self) -> TorshResult<CooTensor>;

    /// Convert to CSR (Compressed Sparse Row) format
    ///
    /// Converts to CSR format, which is optimized for row-wise operations and
    /// matrix-vector multiplication (Ax).
    fn to_csr(&self) -> TorshResult<CsrTensor>;

    /// Convert to CSC (Compressed Sparse Column) format
    ///
    /// Converts to CSC format, which is optimized for column-wise operations and
    /// matrix-vector multiplication with transposed matrices (A^T x).
    fn to_csc(&self) -> TorshResult<CscTensor>;

    /// Calculate sparsity ratio (fraction of zero elements)
    ///
    /// Returns a value between 0.0 and 1.0, where:
    /// - 0.0 = completely dense (no zeros)
    /// - 1.0 = completely sparse (all zeros)
    ///
    /// Formula: sparsity = 1.0 - (nnz / total_elements)
    ///
    /// # Example
    /// ```rust,no_run
    /// # use torsh_sparse::{SparseTensor, CooTensor};
    /// let tensor = CooTensor::from_triplets(vec![(0, 0, 1.0)], (10, 10)).unwrap();
    /// assert_eq!(tensor.sparsity(), 0.99); // 99% sparse (1 non-zero out of 100 elements)
    /// ```
    fn sparsity(&self) -> f32 {
        let total_elements = self.shape().numel();
        if total_elements == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f32 / total_elements as f32)
        }
    }

    /// Cast as Any for downcasting to concrete types
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Create a sparse tensor from a dense tensor
pub fn sparse_from_dense(
    dense: &Tensor,
    format: SparseFormat,
    threshold: Option<f32>,
) -> TorshResult<Box<dyn SparseTensor + Send + Sync>> {
    let threshold = threshold.unwrap_or(0.0);

    match format {
        SparseFormat::Coo => {
            let coo = CooTensor::from_dense(dense, threshold)?;
            Ok(Box::new(coo))
        }
        SparseFormat::Csr => {
            let csr = CsrTensor::from_dense(dense, threshold)?;
            Ok(Box::new(csr))
        }
        SparseFormat::Csc => {
            let csc = CscTensor::from_dense(dense, threshold)?;
            Ok(Box::new(csc))
        }
        SparseFormat::Bsr => {
            // For BSR, use a default block size of 2x2
            let coo = CooTensor::from_dense(dense, threshold)?;
            let bsr = BsrTensor::from_coo(&coo, (2, 2))?;
            Ok(Box::new(bsr))
        }
        SparseFormat::Dia => {
            let dia = DiaTensor::from_dense(dense, threshold)?;
            Ok(Box::new(dia))
        }
        SparseFormat::Dsr => {
            let dsr = DsrTensor::from_dense(dense, threshold)?;
            Ok(Box::new(dsr))
        }
        SparseFormat::Ell => {
            let ell = EllTensor::from_dense(dense, threshold)?;
            Ok(Box::new(ell))
        }
        SparseFormat::Rle => {
            let rle = RleTensor::from_dense(dense, threshold)?;
            Ok(Box::new(rle))
        }
        SparseFormat::Symmetric => {
            let sym = SymmetricTensor::from_dense(dense, SymmetricMode::Upper, threshold)?;
            Ok(Box::new(sym))
        }
    }
}

/// Automatically create the optimal sparse tensor format from a dense tensor
///
/// This function analyzes the sparsity pattern and selects the most efficient
/// sparse format for the given tensor.
pub fn sparse_auto_from_dense(
    dense: &Tensor,
    threshold: Option<f32>,
) -> TorshResult<Box<dyn SparseTensor + Send + Sync>> {
    let threshold = threshold.unwrap_or(0.0);
    let optimal_format = hybrid::auto_select_format(dense, threshold)?;
    sparse_from_dense(dense, optimal_format, Some(threshold))
}

/// Create a hybrid sparse tensor with automatic partitioning
///
/// This creates a hybrid tensor that can use different sparse formats
/// for different regions of the matrix, optimizing for both storage and computation.
pub fn sparse_hybrid_from_dense(
    dense: &Tensor,
    strategy: PartitionStrategy,
    threshold: Option<f32>,
) -> TorshResult<HybridTensor> {
    let threshold = threshold.unwrap_or(0.0);
    let coo = CooTensor::from_dense(dense, threshold)?;
    HybridTensor::from_sparse(coo, strategy)
}

/// Format selection configuration for advanced users
#[derive(Debug, Clone)]
pub struct FormatConfig {
    /// Threshold for considering elements as zero
    pub threshold: f32,
    /// Minimum density to consider a region as dense
    pub dense_threshold: f32,
    /// Block size for block-based analysis
    pub block_size: (usize, usize),
    /// Whether to enable hybrid format selection
    pub enable_hybrid: bool,
    /// Whether to analyze sparsity patterns
    pub analyze_patterns: bool,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            dense_threshold: 0.1,
            block_size: (32, 32),
            enable_hybrid: false,
            analyze_patterns: true,
        }
    }
}

impl FormatConfig {
    /// Create a configuration optimized for memory efficiency
    pub fn memory_optimized() -> Self {
        Self {
            threshold: 1e-12,
            dense_threshold: 0.05,
            block_size: (16, 16),
            enable_hybrid: true,
            analyze_patterns: true,
        }
    }

    /// Create a configuration optimized for computational performance
    pub fn performance_optimized() -> Self {
        Self {
            threshold: 1e-8,
            dense_threshold: 0.2,
            block_size: (64, 64),
            enable_hybrid: false,
            analyze_patterns: false,
        }
    }

    /// Validate the configuration parameters
    pub fn validate(&self) -> TorshResult<()> {
        if self.threshold < 0.0 {
            return Err(TorshError::InvalidArgument(
                "Threshold must be non-negative".to_string(),
            ));
        }

        if self.dense_threshold < 0.0 || self.dense_threshold > 1.0 {
            return Err(TorshError::InvalidArgument(
                "Dense threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.block_size.0 == 0 || self.block_size.1 == 0 {
            return Err(TorshError::InvalidArgument(
                "Block size dimensions must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Create a configuration with custom threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Create a configuration with custom block size
    pub fn with_block_size(mut self, block_size: (usize, usize)) -> Self {
        self.block_size = block_size;
        self
    }

    /// Enable or disable hybrid format selection
    pub fn with_hybrid(mut self, enable: bool) -> Self {
        self.enable_hybrid = enable;
        self
    }
}

/// Advanced sparse tensor creation with detailed configuration
pub fn sparse_from_dense_with_config(
    dense: &Tensor,
    config: FormatConfig,
) -> TorshResult<Box<dyn SparseTensor + Send + Sync>> {
    // Validate configuration first
    config.validate()?;

    if config.enable_hybrid {
        let strategy = if config.analyze_patterns {
            PartitionStrategy::PatternBased
        } else {
            PartitionStrategy::BlockBased {
                block_size: config.block_size,
            }
        };

        let hybrid = sparse_hybrid_from_dense(dense, strategy, Some(config.threshold))?;
        Ok(Box::new(hybrid))
    } else {
        sparse_auto_from_dense(dense, Some(config.threshold))
    }
}

/// Utility to convert between sparse formats
pub fn convert_sparse_format(
    sparse: &dyn SparseTensor,
    target_format: SparseFormat,
) -> TorshResult<Box<dyn SparseTensor + Send + Sync>> {
    match target_format {
        SparseFormat::Coo => Ok(Box::new(sparse.to_coo()?)),
        SparseFormat::Csr => Ok(Box::new(sparse.to_csr()?)),
        SparseFormat::Csc => Ok(Box::new(sparse.to_csc()?)),
        SparseFormat::Bsr => {
            let coo = sparse.to_coo()?;
            let bsr = BsrTensor::from_coo(&coo, (2, 2))?;
            Ok(Box::new(bsr))
        }
        SparseFormat::Dia => {
            let coo = sparse.to_coo()?;
            let dia = DiaTensor::from_coo(&coo)?;
            Ok(Box::new(dia))
        }
        SparseFormat::Dsr => {
            let coo = sparse.to_coo()?;
            let dsr = DsrTensor::from_coo(&coo)?;
            Ok(Box::new(dsr))
        }
        SparseFormat::Ell => {
            let coo = sparse.to_coo()?;
            let ell = EllTensor::from_coo(&coo)?;
            Ok(Box::new(ell))
        }
        SparseFormat::Rle => {
            let coo = sparse.to_coo()?;
            let rle = RleTensor::from_coo(&coo)?;
            Ok(Box::new(rle))
        }
        SparseFormat::Symmetric => {
            let coo = sparse.to_coo()?;
            let sym = SymmetricTensor::from_coo(&coo, SymmetricMode::Upper, 1e-6)?;
            Ok(Box::new(sym))
        }
    }
}

// Performance comparison utilities are already defined above in this module

/// Analyze sparse tensor characteristics for format optimization
#[derive(Debug, Clone)]
pub struct SparseAnalysis {
    /// Current sparse format
    pub format: SparseFormat,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Sparsity ratio (0.0 = dense, 1.0 = empty)
    pub sparsity: f32,
    /// Recommended optimal format
    pub recommended_format: SparseFormat,
    /// Detected sparsity pattern
    pub pattern: SparsityPattern,
    /// Storage efficiency (bytes per non-zero element)
    pub storage_efficiency: f32,
}

/// Format performance comparison result
#[derive(Debug, Clone)]
pub struct FormatPerformanceComparison {
    /// Test tensor characteristics
    pub tensor_info: SparseAnalysis,
    /// Performance results for each format
    pub format_results: HashMap<SparseFormat, FormatPerformanceResult>,
    /// Recommended format based on overall performance
    pub recommended_format: SparseFormat,
    /// Performance improvement factor over worst format
    pub improvement_factor: f32,
}

/// Performance result for a specific format
#[derive(Debug, Clone)]
pub struct FormatPerformanceResult {
    /// Format tested
    pub format: SparseFormat,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Creation time in nanoseconds
    pub creation_time_ns: u64,
    /// Matrix-vector multiplication time in nanoseconds (if applicable)
    pub spmv_time_ns: Option<u64>,
    /// Format conversion time from COO in nanoseconds
    pub conversion_time_ns: u64,
    /// Overall performance score (lower is better)
    pub performance_score: f32,
}

/// Compare performance across different sparse formats for a given tensor
///
/// This function converts a sparse tensor to all supported formats and measures
/// performance characteristics including memory usage, conversion time, and
/// operation performance. Useful for determining the optimal format for
/// specific use cases and access patterns.
///
/// # Arguments
/// * `sparse` - The input sparse tensor to analyze
/// * `include_operations` - Whether to benchmark actual operations (slower but more accurate)
///
/// # Returns
/// A comprehensive performance comparison across all supported formats
///
/// # Example
/// ```rust,no_run
/// use torsh_sparse::{CooTensor, compare_format_performance};
///
/// let triplets = vec![(0, 0, 1.0f32), (1, 1, 2.0f32), (100, 100, 3.0f32)];
/// let coo = CooTensor::from_triplets(triplets, (1000, 1000)).unwrap();
///
/// let comparison = compare_format_performance(&coo, true).unwrap();
/// println!("Recommended format: {:?}", comparison.recommended_format);
/// println!("Performance improvement: {:.2}x", comparison.improvement_factor);
/// ```
pub fn compare_format_performance(
    sparse: &dyn SparseTensor,
    include_operations: bool,
) -> TorshResult<FormatPerformanceComparison> {
    // Get basic tensor analysis
    let tensor_info = analyze_sparse_tensor(sparse)?;
    let mut format_results = HashMap::new();

    // Convert to COO as baseline
    let coo = sparse.to_coo()?;

    // Test each format
    let formats_to_test = vec![
        SparseFormat::Coo,
        SparseFormat::Csr,
        SparseFormat::Csc,
        SparseFormat::Bsr,
        SparseFormat::Dia,
        SparseFormat::Dsr,
        SparseFormat::Ell,
        SparseFormat::Rle,
        SparseFormat::Symmetric,
    ];

    for format in formats_to_test {
        let result = benchmark_format_performance(&coo, format, include_operations)?;
        format_results.insert(format, result);
    }

    // Determine best format based on overall score
    let recommended_format = format_results
        .iter()
        .min_by(|a, b| {
            a.1.performance_score
                .partial_cmp(&b.1.performance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(format, _)| *format)
        .unwrap_or(SparseFormat::Csr);

    // Calculate improvement factor
    let best_score = format_results[&recommended_format].performance_score;
    let worst_score = format_results
        .values()
        .map(|r| r.performance_score)
        .fold(0.0f32, |a, b| a.max(b));

    let improvement_factor = if best_score > 0.0 {
        worst_score / best_score
    } else {
        1.0
    };

    Ok(FormatPerformanceComparison {
        tensor_info,
        format_results,
        recommended_format,
        improvement_factor,
    })
}

/// Benchmark performance characteristics for a specific format
fn benchmark_format_performance(
    coo: &CooTensor,
    format: SparseFormat,
    include_operations: bool,
) -> TorshResult<FormatPerformanceResult> {
    use std::time::Instant;

    // Measure conversion time
    let conversion_start = Instant::now();
    let converted = match format {
        SparseFormat::Coo => Box::new(coo.clone()) as Box<dyn SparseTensor + Send + Sync>,
        SparseFormat::Csr => Box::new(coo.to_csr()?) as Box<dyn SparseTensor + Send + Sync>,
        SparseFormat::Csc => Box::new(coo.to_csc()?) as Box<dyn SparseTensor + Send + Sync>,
        SparseFormat::Bsr => {
            let bsr = BsrTensor::from_coo(coo, (2, 2))?;
            Box::new(bsr) as Box<dyn SparseTensor + Send + Sync>
        }
        SparseFormat::Dia => {
            let dia = DiaTensor::from_coo(coo)?;
            Box::new(dia) as Box<dyn SparseTensor + Send + Sync>
        }
        SparseFormat::Dsr => {
            let dsr = DsrTensor::from_coo(coo)?;
            Box::new(dsr) as Box<dyn SparseTensor + Send + Sync>
        }
        SparseFormat::Ell => {
            let ell = EllTensor::from_coo(coo)?;
            Box::new(ell) as Box<dyn SparseTensor + Send + Sync>
        }
        SparseFormat::Rle => {
            let rle = RleTensor::from_coo(coo)?;
            Box::new(rle) as Box<dyn SparseTensor + Send + Sync>
        }
        SparseFormat::Symmetric => {
            let sym = SymmetricTensor::from_coo(coo, SymmetricMode::Upper, 1e-6)?;
            Box::new(sym) as Box<dyn SparseTensor + Send + Sync>
        }
    };
    let conversion_time_ns = conversion_start.elapsed().as_nanos() as u64;

    // Estimate memory usage (simplified)
    let memory_usage = estimate_memory_usage(&*converted);

    // Measure creation time (conversion time serves as proxy)
    let creation_time_ns = conversion_time_ns;

    // Optionally measure operation performance
    let spmv_time_ns = if include_operations && coo.shape().dims()[0] <= 1000 {
        // Only benchmark on reasonably sized matrices
        measure_spmv_performance(&*converted).ok()
    } else {
        None
    };

    // Calculate overall performance score (weighted combination of factors)
    let mut performance_score = 0.0f32;

    // Memory efficiency (normalized by nnz)
    let memory_per_nnz = if converted.nnz() > 0 {
        memory_usage as f32 / converted.nnz() as f32
    } else {
        0.0
    };
    performance_score += memory_per_nnz * 0.3; // 30% weight

    // Conversion time (normalized)
    performance_score += (conversion_time_ns as f32 / 1_000_000.0) * 0.2; // 20% weight in ms

    // Operation performance (if available)
    if let Some(spmv_ns) = spmv_time_ns {
        performance_score += (spmv_ns as f32 / 1_000_000.0) * 0.5; // 50% weight in ms
    } else {
        // If no operation benchmark, increase weight of other factors
        performance_score += memory_per_nnz * 0.25; // Additional 25% to memory
        performance_score += (conversion_time_ns as f32 / 1_000_000.0) * 0.25; // Additional 25% to conversion
    }

    Ok(FormatPerformanceResult {
        format,
        memory_usage,
        creation_time_ns,
        spmv_time_ns,
        conversion_time_ns,
        performance_score,
    })
}

/// Estimate memory usage for a sparse tensor (simplified calculation)
fn estimate_memory_usage(tensor: &dyn SparseTensor) -> usize {
    let nnz = tensor.nnz();
    match tensor.format() {
        SparseFormat::Coo => nnz * 12, // 3 arrays (row, col, val) * 4 bytes each
        SparseFormat::Csr => nnz * 8 + tensor.shape().dims()[0] * 4, // vals + indices + row_ptr
        SparseFormat::Csc => nnz * 8 + tensor.shape().dims()[1] * 4, // vals + indices + col_ptr
        SparseFormat::Bsr => nnz * 8,  // Approximate for block storage
        SparseFormat::Dia => nnz * 8,  // Diagonal storage
        SparseFormat::Dsr => nnz * 16, // Dynamic storage with overhead
        SparseFormat::Ell => nnz * 8,  // ELLPACK storage
        SparseFormat::Rle => nnz * 6,  // Run-length encoded
        SparseFormat::Symmetric => nnz * 6, // Roughly half storage
    }
}

/// Measure sparse matrix-vector multiplication performance
fn measure_spmv_performance(tensor: &dyn SparseTensor) -> TorshResult<u64> {
    use std::time::Instant;
    use torsh_tensor::creation::ones;

    // Create a dense vector for multiplication
    let vector = ones::<f32>(&[tensor.shape().dims()[1]])?;

    // Warm-up run
    let _ = crate::ops::spmm(tensor, &vector)?;

    // Measured run
    let start = Instant::now();
    let _ = crate::ops::spmm(tensor, &vector)?;
    let duration = start.elapsed();

    Ok(duration.as_nanos() as u64)
}

/// Analyze a sparse tensor and provide optimization recommendations
pub fn analyze_sparse_tensor(sparse: &dyn SparseTensor) -> TorshResult<SparseAnalysis> {
    let format = sparse.format();
    let nnz = sparse.nnz();
    let sparsity = sparse.sparsity();
    let shape = sparse.shape();

    // Convert to COO for pattern analysis
    let coo = sparse.to_coo()?;
    let triplets = coo.triplets();
    let pattern = hybrid::HybridTensor::analyze_sparsity_pattern(&triplets, shape)?;

    // Recommend optimal format based on analysis
    let recommended_format = match pattern {
        SparsityPattern::Diagonal => SparseFormat::Dia,
        SparsityPattern::Banded { .. } => {
            // Check if matrix is symmetric for symmetric format recommendation
            if is_matrix_symmetric(&coo) {
                SparseFormat::Symmetric
            } else {
                SparseFormat::Ell
            }
        }
        SparsityPattern::BlockDiagonal { .. } => SparseFormat::Bsr,
        SparsityPattern::Random => {
            // Check for run-length encoding opportunities
            if has_consecutive_patterns(&coo) {
                SparseFormat::Rle
            } else if sparsity > 0.9 {
                SparseFormat::Coo
            } else {
                SparseFormat::Csr // Default for sparsity <= 0.9
            }
        }
    };

    // Estimate storage efficiency (rough approximation)
    let storage_efficiency = match format {
        SparseFormat::Coo => 12.0, // 3 values per element (row, col, val)
        SparseFormat::Csr => 8.0 + (4.0 * shape.dims()[0] as f32 / nnz as f32), // values + indices + row pointers
        SparseFormat::Csc => 8.0 + (4.0 * shape.dims()[1] as f32 / nnz as f32), // values + indices + col pointers
        SparseFormat::Bsr => 8.0,       // Approximate for block storage
        SparseFormat::Dia => 8.0,       // Diagonal storage
        SparseFormat::Dsr => 16.0,      // Dynamic sparse row (higher overhead for BTreeMap)
        SparseFormat::Ell => 8.0,       // ELLPACK storage
        SparseFormat::Rle => 6.0,       // Run-length encoding (row, col, length, values)
        SparseFormat::Symmetric => 6.0, // Half storage for symmetric matrices
    };

    Ok(SparseAnalysis {
        format,
        nnz,
        sparsity,
        recommended_format,
        pattern,
        storage_efficiency,
    })
}

/// Check if a COO matrix is symmetric
fn is_matrix_symmetric(coo: &CooTensor) -> bool {
    use std::collections::HashMap;

    let triplets = coo.triplets();
    let mut element_map: HashMap<(usize, usize), f32> = HashMap::new();

    // Build element map
    for (row, col, value) in &triplets {
        element_map.insert((*row, *col), *value);
    }

    // Check symmetry
    for (row, col, value) in &triplets {
        if *row != *col {
            if let Some(&sym_value) = element_map.get(&(*col, *row)) {
                if (value - sym_value).abs() > 1e-6 {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    true
}

/// Check if a COO matrix has consecutive patterns that would benefit from RLE
fn has_consecutive_patterns(coo: &CooTensor) -> bool {
    use std::collections::HashMap;

    let triplets = coo.triplets();
    let mut row_elements: HashMap<usize, Vec<usize>> = HashMap::new();

    // Group elements by row
    for (row, col, _) in &triplets {
        row_elements.entry(*row).or_default().push(*col);
    }

    let mut consecutive_count = 0;
    let mut total_elements = 0;

    // Check for consecutive patterns in each row
    for (_, mut cols_in_row) in row_elements {
        cols_in_row.sort_unstable();
        total_elements += cols_in_row.len();

        for window in cols_in_row.windows(2) {
            if window[1] == window[0] + 1 {
                consecutive_count += 1;
            }
        }
    }

    // Return true if more than 30% of elements are part of consecutive sequences
    if total_elements == 0 {
        false
    } else {
        (consecutive_count as f32 / total_elements as f32) > 0.3
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::autograd::{
        SparseAutogradTensor, SparseData, SparseGradFn, SparseGradientAccumulator,
    };
    pub use crate::bsr::BsrTensor;
    pub use crate::coo::CooTensor;
    pub use crate::csc::CscTensor;
    pub use crate::csr::CsrTensor;
    pub use crate::dia::DiaTensor;
    pub use crate::dsr::DsrTensor;
    pub use crate::ell::EllTensor;
    pub use crate::gpu::{CudaSparseOps, CudaSparseTensor, CudaSparseTensorFactory};
    pub use crate::rle::RleTensor;
    pub use crate::symmetric::{SymmetricMode, SymmetricTensor};
    // Re-export from lib.rs instead of unified_interface
    pub use crate::{
        analyze_sparse_tensor, compare_format_performance, sparse_from_dense, SparseFormat,
        SparseTensor,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_sparse_format() {
        // Test sparse format creation
        let dense = zeros::<f32>(&[3, 4]).unwrap();

        // Set some non-zero values
        // dense[[0, 1]] = 1.0
        // dense[[1, 2]] = 2.0
        // dense[[2, 0]] = 3.0

        let sparse = sparse_from_dense(&dense, SparseFormat::Coo, None).unwrap();
        assert_eq!(sparse.format(), SparseFormat::Coo);
        assert_eq!(sparse.shape(), &dense.shape());
    }

    #[test]
    fn test_format_performance_comparison() {
        // Create a simple sparse tensor for testing
        let triplets = vec![(0, 0, 1.0f32), (1, 1, 2.0f32), (2, 2, 3.0f32)];
        let coo = CooTensor::from_triplets(triplets, (10, 10)).unwrap();

        // Test performance comparison without operations (faster)
        let comparison = compare_format_performance(&coo, false).unwrap();

        // Verify basic properties
        assert!(!comparison.format_results.is_empty());
        assert!(comparison.improvement_factor >= 1.0);

        // Check that COO format is present in results
        assert!(comparison.format_results.contains_key(&SparseFormat::Coo));

        // Verify recommended format is valid
        assert!(comparison
            .format_results
            .contains_key(&comparison.recommended_format));

        // Check that performance scores are reasonable
        for result in comparison.format_results.values() {
            assert!(result.performance_score >= 0.0);
            assert!(result.memory_usage > 0);
            // conversion_time_ns is u64, so it's always >= 0
        }
    }

    #[test]
    fn test_sparse_analysis() {
        // Create a diagonal sparse tensor
        let triplets = vec![(0, 0, 1.0f32), (1, 1, 2.0f32), (2, 2, 3.0f32)];
        let coo = CooTensor::from_triplets(triplets, (3, 3)).unwrap();

        let analysis = analyze_sparse_tensor(&coo).unwrap();

        // Verify analysis properties
        assert_eq!(analysis.format, SparseFormat::Coo);
        assert_eq!(analysis.nnz, 3);
        assert!(analysis.sparsity > 0.0 && analysis.sparsity <= 1.0);
        assert!(analysis.storage_efficiency > 0.0);

        // For a diagonal matrix, DIA format should be recommended
        assert_eq!(analysis.recommended_format, SparseFormat::Dia);
    }
}
