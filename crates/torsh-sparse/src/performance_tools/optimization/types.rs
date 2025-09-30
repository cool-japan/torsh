//! Shared types and enums for optimization module

use crate::SparseFormat;

/// Result of auto-tuning analysis
#[derive(Debug, Clone)]
pub struct TuningResult {
    /// Input characteristics that led to this result
    pub input_characteristics: InputCharacteristics,
    /// Recommended sparse format
    pub recommended_format: SparseFormat,
    /// Expected performance improvement
    pub performance_score: f64,
    /// Confidence in recommendation (0-1)
    pub confidence: f64,
    /// Reasoning for the recommendation
    pub reasoning: Vec<String>,
}

/// Characteristics of input data for tuning decisions
#[derive(Debug, Clone)]
pub struct InputCharacteristics {
    /// Matrix dimensions
    pub dimensions: (usize, usize),
    /// Sparsity ratio (0-1)
    pub sparsity: f64,
    /// Distribution pattern of non-zeros
    pub distribution_pattern: DistributionPattern,
    /// Expected operation types
    pub operation_types: Vec<OperationType>,
    /// Memory constraints
    pub memory_budget: Option<usize>,
}

/// Pattern of non-zero distribution in sparse matrices
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionPattern {
    /// Randomly distributed non-zeros
    Random,
    /// Block-structured non-zeros
    Block { block_size: (usize, usize) },
    /// Banded structure
    Banded { bandwidth: usize },
    /// Diagonal or near-diagonal
    Diagonal,
    /// Row-wise clustering
    RowClustered,
    /// Column-wise clustering
    ColumnClustered,
    /// Unknown or mixed pattern
    Mixed,
}

/// Types of operations to optimize for
#[derive(Debug, Clone, PartialEq)]
pub enum OperationType {
    /// Matrix-vector multiplication
    MatrixVector,
    /// Matrix-matrix multiplication
    MatrixMatrix,
    /// Transposition
    Transpose,
    /// Element-wise operations
    ElementWise,
    /// Factorization (LU, Cholesky, etc.)
    Factorization,
    /// Iterative solvers
    IterativeSolver,
}

/// Optimization strategies for auto-tuning
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Minimize execution time
    Speed,
    /// Minimize memory usage
    Memory,
    /// Balance speed and memory
    Balanced,
    /// Maximize cache efficiency
    CacheEfficient,
    /// Custom weighted strategy
    Custom {
        speed_weight: f64,
        memory_weight: f64,
        cache_weight: f64,
    },
}
