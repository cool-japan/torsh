//! Error types for clustering operations

use thiserror::Error;
use torsh_core::error::TorshError;

/// Result type for clustering operations
pub type ClusterResult<T> = Result<T, ClusterError>;

/// Errors that can occur during clustering operations
#[derive(Error, Debug)]
pub enum ClusterError {
    /// Invalid number of clusters
    #[error("Invalid number of clusters: {0}. Must be positive and less than number of samples")]
    InvalidClusters(usize),

    /// Invalid input data
    #[error("Invalid input data: {0}")]
    InvalidInput(String),

    /// Convergence failure
    #[error("Algorithm failed to converge after {max_iters} iterations")]
    ConvergenceFailure { max_iters: usize },

    /// Empty dataset
    #[error("Dataset is empty")]
    EmptyDataset,

    /// Insufficient data points
    #[error("Insufficient data points: need at least {required}, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// Invalid distance metric
    #[error("Invalid distance metric: {0}")]
    InvalidDistanceMetric(String),

    /// Invalid linkage criterion
    #[error("Invalid linkage criterion: {0}")]
    InvalidLinkage(String),

    /// Invalid epsilon parameter for DBSCAN
    #[error("Invalid epsilon parameter: {0}. Must be positive")]
    InvalidEpsilon(f64),

    /// Invalid minimum samples parameter
    #[error("Invalid minimum samples: {0}. Must be positive")]
    InvalidMinSamples(usize),

    /// Invalid covariance type for Gaussian Mixture
    #[error("Invalid covariance type: {0}")]
    InvalidCovarianceType(String),

    /// Singular matrix error
    #[error("Singular matrix encountered during computation")]
    SingularMatrix,

    /// Tensor operation error
    #[error("Tensor operation failed: {0}")]
    TensorError(#[from] TorshError),

    /// SciRS2 core error
    #[error("SciRS2 core error: {0}")]
    SciRS2Error(String),

    /// Invalid initialization method
    #[error("Invalid initialization method: {0}")]
    InvalidInitialization(String),

    /// Invalid affinity matrix
    #[error("Invalid affinity matrix: {0}")]
    InvalidAffinityMatrix(String),

    /// Memory allocation error
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),

    /// Invalid feature dimension
    #[error("Invalid feature dimension: expected {expected}, got {actual}")]
    InvalidFeatureDimension { expected: usize, actual: usize },

    /// Invalid cluster assignment
    #[error("Invalid cluster assignment: {0}")]
    InvalidAssignment(String),

    /// Numerical instability
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Algorithm not implemented
    #[error("Algorithm not implemented: {0}")]
    NotImplemented(String),
}

impl ClusterError {
    /// Create a new SciRS2 error
    pub fn scirs2_error(msg: impl Into<String>) -> Self {
        Self::SciRS2Error(msg.into())
    }

    /// Create a new invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a new configuration error
    pub fn config_error(msg: impl Into<String>) -> Self {
        Self::ConfigError(msg.into())
    }

    /// Create a new numerical instability error
    pub fn numerical_instability(msg: impl Into<String>) -> Self {
        Self::NumericalInstability(msg.into())
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            ClusterError::ConvergenceFailure { .. }
                | ClusterError::NumericalInstability(_)
                | ClusterError::ConfigError(_)
        )
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ClusterError::EmptyDataset
            | ClusterError::SingularMatrix
            | ClusterError::MemoryError(_) => ErrorSeverity::Critical,

            ClusterError::InvalidClusters(_)
            | ClusterError::InvalidInput(_)
            | ClusterError::InvalidDistanceMetric(_)
            | ClusterError::InvalidLinkage(_)
            | ClusterError::InvalidEpsilon(_)
            | ClusterError::InvalidMinSamples(_)
            | ClusterError::InvalidCovarianceType(_)
            | ClusterError::InvalidInitialization(_)
            | ClusterError::InvalidAffinityMatrix(_)
            | ClusterError::InvalidFeatureDimension { .. }
            | ClusterError::InvalidAssignment(_)
            | ClusterError::ConfigError(_) => ErrorSeverity::High,

            ClusterError::ConvergenceFailure { .. } | ClusterError::NumericalInstability(_) => {
                ErrorSeverity::Medium
            }

            ClusterError::InsufficientData { .. }
            | ClusterError::TensorError(_)
            | ClusterError::SciRS2Error(_)
            | ClusterError::NotImplemented(_) => ErrorSeverity::Low,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Context information for clustering errors
#[derive(Debug, Clone)]
pub struct ClusterErrorContext {
    pub algorithm: String,
    pub data_shape: Option<Vec<usize>>,
    pub n_clusters: Option<usize>,
    pub iteration: Option<usize>,
    pub additional_info: Option<String>,
}

impl ClusterErrorContext {
    /// Create a new error context
    pub fn new(algorithm: impl Into<String>) -> Self {
        Self {
            algorithm: algorithm.into(),
            data_shape: None,
            n_clusters: None,
            iteration: None,
            additional_info: None,
        }
    }

    /// Set data shape
    pub fn with_data_shape(mut self, shape: Vec<usize>) -> Self {
        self.data_shape = Some(shape);
        self
    }

    /// Set number of clusters
    pub fn with_n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = Some(n_clusters);
        self
    }

    /// Set iteration number
    pub fn with_iteration(mut self, iteration: usize) -> Self {
        self.iteration = Some(iteration);
        self
    }

    /// Set additional information
    pub fn with_info(mut self, info: impl Into<String>) -> Self {
        self.additional_info = Some(info.into());
        self
    }
}
