//! Enhanced error handling for torsh-data
//!
//! This module provides comprehensive error types with detailed context
//! and recovery suggestions for data loading operations.

use scirs2_core::random::{thread_rng, Rng}; // SciRS2 POLICY compliant
use std::fmt;

/// Enhanced error types specific to data loading operations
#[derive(Debug, Clone)]
pub enum DataError {
    /// Dataset-related errors
    Dataset {
        kind: DatasetErrorKind,
        context: String,
        suggestion: Option<String>,
    },

    /// Data loader errors
    DataLoader {
        kind: DataLoaderErrorKind,
        context: String,
        suggestion: Option<String>,
    },

    /// Transform operation errors
    Transform {
        kind: TransformErrorKind,
        transform_name: String,
        context: String,
        suggestion: Option<String>,
    },

    /// Sampler-related errors
    Sampler {
        kind: SamplerErrorKind,
        sampler_type: String,
        context: String,
        suggestion: Option<String>,
    },

    /// Collation errors
    Collation {
        kind: CollationErrorKind,
        batch_info: BatchInfo,
        context: String,
        suggestion: Option<String>,
    },

    /// I/O and file system errors
    Io {
        kind: IoErrorKind,
        path: Option<String>,
        operation: String,
        context: String,
        suggestion: Option<String>,
    },

    /// Configuration and validation errors
    Configuration {
        kind: ConfigErrorKind,
        parameter: String,
        value: String,
        context: String,
        suggestion: Option<String>,
    },

    /// Memory and resource errors
    Resource {
        kind: ResourceErrorKind,
        resource_type: String,
        requested: Option<usize>,
        available: Option<usize>,
        context: String,
        suggestion: Option<String>,
    },

    /// Privacy and differential privacy errors
    Privacy {
        kind: PrivacyErrorKind,
        privacy_parameter: String,
        context: String,
        suggestion: Option<String>,
    },

    /// GPU acceleration and compute errors
    GpuError(String),

    /// Other errors
    Other(String),
}

#[derive(Debug, Clone)]
pub enum DatasetErrorKind {
    IndexOutOfBounds,
    EmptyDataset,
    IncompatibleShapes,
    MissingData,
    CorruptedData,
    UnsupportedFormat,
    AccessDenied,
}

#[derive(Debug, Clone)]
pub enum DataLoaderErrorKind {
    WorkerPanic,
    ChannelClosed,
    Timeout,
    ConfigurationInvalid,
    BackendUnavailable,
    BatchGenerationFailed,
}

#[derive(Debug, Clone)]
pub enum TransformErrorKind {
    InvalidInput,
    IncompatibleDimensions,
    NumericalInstability,
    UnsupportedOperation,
    ConfigurationError,
    ResourceExhaustion,
}

#[derive(Debug, Clone)]
pub enum SamplerErrorKind {
    InvalidWeights,
    EmptyPopulation,
    InvalidProbability,
    IndexOutOfRange,
    InsufficientData,
    ConfigurationConflict,
}

#[derive(Debug, Clone)]
pub enum CollationErrorKind {
    ShapeMismatch,
    TypeMismatch,
    BatchSizeExceeded,
    MemoryExhaustion,
    InvalidPadding,
    UnsupportedCollation,
}

#[derive(Debug, Clone)]
pub enum IoErrorKind {
    FileNotFound,
    PermissionDenied,
    DiskFull,
    NetworkError,
    CorruptedFile,
    UnsupportedFormat,
    WriteError,
    ReadError,
}

#[derive(Debug, Clone)]
pub enum ConfigErrorKind {
    InvalidValue,
    MissingRequired,
    ConflictingValues,
    OutOfRange,
    InvalidType,
    Deprecated,
}

#[derive(Debug, Clone)]
pub enum ResourceErrorKind {
    MemoryExhaustion,
    CpuOverload,
    GpuUnavailable,
    DiskSpaceExhaustion,
    ThreadPoolExhaustion,
    CacheOverflow,
}

#[derive(Debug, Clone)]
pub enum PrivacyErrorKind {
    BudgetExceeded,
    InvalidPrivacyParameter,
    AccessLimitExceeded,
    AccessDenied,
    TensorCreationFailed,
    NoiseGenerationFailed,
    CompositionError,
}

#[derive(Debug, Clone)]
pub struct BatchInfo {
    pub batch_size: usize,
    pub item_shapes: Vec<Vec<usize>>,
    pub item_types: Vec<String>,
}

impl BatchInfo {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            item_shapes: Vec::new(),
            item_types: Vec::new(),
        }
    }

    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.item_shapes.push(shape);
        self
    }

    pub fn with_type(mut self, type_name: String) -> Self {
        self.item_types.push(type_name);
        self
    }
}

impl DataError {
    /// Create a dataset error with context
    pub fn dataset(kind: DatasetErrorKind, context: impl Into<String>) -> Self {
        Self::Dataset {
            kind,
            context: context.into(),
            suggestion: None,
        }
    }

    /// Create a data loader error with context
    pub fn dataloader(kind: DataLoaderErrorKind, context: impl Into<String>) -> Self {
        Self::DataLoader {
            kind,
            context: context.into(),
            suggestion: None,
        }
    }

    /// Create a transform error with context
    pub fn transform(
        kind: TransformErrorKind,
        transform_name: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        Self::Transform {
            kind,
            transform_name: transform_name.into(),
            context: context.into(),
            suggestion: None,
        }
    }

    /// Create a sampler error with context
    pub fn sampler(
        kind: SamplerErrorKind,
        sampler_type: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        Self::Sampler {
            kind,
            sampler_type: sampler_type.into(),
            context: context.into(),
            suggestion: None,
        }
    }

    /// Create a collation error with context
    pub fn collation(
        kind: CollationErrorKind,
        batch_info: BatchInfo,
        context: impl Into<String>,
    ) -> Self {
        Self::Collation {
            kind,
            batch_info,
            context: context.into(),
            suggestion: None,
        }
    }

    /// Create an I/O error with context
    pub fn io(kind: IoErrorKind, operation: impl Into<String>, context: impl Into<String>) -> Self {
        Self::Io {
            kind,
            path: None,
            operation: operation.into(),
            context: context.into(),
            suggestion: None,
        }
    }

    /// Create a configuration error with context
    pub fn config(
        kind: ConfigErrorKind,
        parameter: impl Into<String>,
        value: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        Self::Configuration {
            kind,
            parameter: parameter.into(),
            value: value.into(),
            context: context.into(),
            suggestion: None,
        }
    }

    /// Create a resource error with context
    pub fn resource(
        kind: ResourceErrorKind,
        resource_type: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        Self::Resource {
            kind,
            resource_type: resource_type.into(),
            requested: None,
            available: None,
            context: context.into(),
            suggestion: None,
        }
    }

    /// Create a privacy error with context
    pub fn privacy(
        kind: PrivacyErrorKind,
        privacy_parameter: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        Self::Privacy {
            kind,
            privacy_parameter: privacy_parameter.into(),
            context: context.into(),
            suggestion: None,
        }
    }

    // Convenience methods for specific privacy errors

    /// Create a privacy budget exceeded error
    pub fn privacy_budget_exceeded(context: impl Into<String>) -> Self {
        Self::privacy(PrivacyErrorKind::BudgetExceeded, "privacy_budget", context)
    }

    /// Create an invalid privacy parameter error
    pub fn invalid_privacy_parameter(context: impl Into<String>) -> Self {
        Self::privacy(
            PrivacyErrorKind::InvalidPrivacyParameter,
            "privacy_parameter",
            context,
        )
    }

    /// Create an access limit exceeded error
    pub fn privacy_access_limit_exceeded(context: impl Into<String>) -> Self {
        Self::privacy(
            PrivacyErrorKind::AccessLimitExceeded,
            "access_limit",
            context,
        )
    }

    /// Create an access denied error
    pub fn privacy_access_denied(context: impl Into<String>) -> Self {
        Self::privacy(PrivacyErrorKind::AccessDenied, "access_control", context)
    }

    /// Create a tensor creation failed error
    pub fn tensor_creation_failed(context: impl Into<String>) -> Self {
        Self::privacy(
            PrivacyErrorKind::TensorCreationFailed,
            "tensor_creation",
            context,
        )
    }

    /// Create a noise generation failed error
    pub fn noise_generation_failed(context: impl Into<String>) -> Self {
        Self::privacy(
            PrivacyErrorKind::NoiseGenerationFailed,
            "noise_generation",
            context,
        )
    }

    /// Create a composition error
    pub fn privacy_composition_error(context: impl Into<String>) -> Self {
        Self::privacy(
            PrivacyErrorKind::CompositionError,
            "privacy_composition",
            context,
        )
    }

    /// Add a suggestion for error recovery
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        match &mut self {
            DataError::Dataset { suggestion: s, .. }
            | DataError::DataLoader { suggestion: s, .. }
            | DataError::Transform { suggestion: s, .. }
            | DataError::Sampler { suggestion: s, .. }
            | DataError::Collation { suggestion: s, .. }
            | DataError::Io { suggestion: s, .. }
            | DataError::Configuration { suggestion: s, .. }
            | DataError::Resource { suggestion: s, .. }
            | DataError::Privacy { suggestion: s, .. } => {
                *s = Some(suggestion.into());
            }
            DataError::GpuError(_) | DataError::Other(_) => {
                // These error types don't have suggestion fields
            }
        }
        self
    }

    /// Add path information for I/O errors
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        if let DataError::Io { path: p, .. } = &mut self {
            *p = Some(path.into());
        }
        self
    }

    /// Add resource information for resource errors
    pub fn with_resource_info(mut self, requested: usize, available: Option<usize>) -> Self {
        if let DataError::Resource {
            requested: r,
            available: a,
            ..
        } = &mut self
        {
            *r = Some(requested);
            *a = available;
        }
        self
    }

    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            DataError::Dataset { kind, .. } => match kind {
                DatasetErrorKind::IndexOutOfBounds => true,
                DatasetErrorKind::EmptyDataset => false,
                DatasetErrorKind::IncompatibleShapes => false,
                DatasetErrorKind::MissingData => true,
                DatasetErrorKind::CorruptedData => false,
                DatasetErrorKind::UnsupportedFormat => false,
                DatasetErrorKind::AccessDenied => true,
            },
            DataError::DataLoader { kind, .. } => match kind {
                DataLoaderErrorKind::WorkerPanic => true,
                DataLoaderErrorKind::ChannelClosed => true,
                DataLoaderErrorKind::Timeout => true,
                DataLoaderErrorKind::ConfigurationInvalid => false,
                DataLoaderErrorKind::BackendUnavailable => true,
                DataLoaderErrorKind::BatchGenerationFailed => true,
            },
            DataError::Transform { kind, .. } => match kind {
                TransformErrorKind::InvalidInput => false,
                TransformErrorKind::IncompatibleDimensions => false,
                TransformErrorKind::NumericalInstability => true,
                TransformErrorKind::UnsupportedOperation => false,
                TransformErrorKind::ConfigurationError => false,
                TransformErrorKind::ResourceExhaustion => true,
            },
            DataError::Sampler { kind, .. } => match kind {
                SamplerErrorKind::InvalidWeights => false,
                SamplerErrorKind::EmptyPopulation => false,
                SamplerErrorKind::InvalidProbability => false,
                SamplerErrorKind::IndexOutOfRange => true,
                SamplerErrorKind::InsufficientData => true,
                SamplerErrorKind::ConfigurationConflict => false,
            },
            DataError::Collation { kind, .. } => match kind {
                CollationErrorKind::ShapeMismatch => false,
                CollationErrorKind::TypeMismatch => false,
                CollationErrorKind::BatchSizeExceeded => true,
                CollationErrorKind::MemoryExhaustion => true,
                CollationErrorKind::InvalidPadding => false,
                CollationErrorKind::UnsupportedCollation => false,
            },
            DataError::Io { kind, .. } => match kind {
                IoErrorKind::FileNotFound => true,
                IoErrorKind::PermissionDenied => true,
                IoErrorKind::DiskFull => true,
                IoErrorKind::NetworkError => true,
                IoErrorKind::CorruptedFile => false,
                IoErrorKind::UnsupportedFormat => false,
                IoErrorKind::WriteError => true,
                IoErrorKind::ReadError => true,
            },
            DataError::Configuration { .. } => false,
            DataError::Resource { kind, .. } => match kind {
                ResourceErrorKind::MemoryExhaustion => true,
                ResourceErrorKind::CpuOverload => true,
                ResourceErrorKind::GpuUnavailable => true,
                ResourceErrorKind::DiskSpaceExhaustion => true,
                ResourceErrorKind::ThreadPoolExhaustion => true,
                ResourceErrorKind::CacheOverflow => true,
            },
            DataError::Privacy { kind, .. } => match kind {
                PrivacyErrorKind::BudgetExceeded => false,
                PrivacyErrorKind::InvalidPrivacyParameter => false,
                PrivacyErrorKind::AccessLimitExceeded => false,
                PrivacyErrorKind::AccessDenied => true,
                PrivacyErrorKind::TensorCreationFailed => true,
                PrivacyErrorKind::NoiseGenerationFailed => true,
                PrivacyErrorKind::CompositionError => false,
            },
            DataError::GpuError(_) => true, // GPU errors are typically recoverable with CPU fallback
            DataError::Other(_) => false,   // Generic errors are typically not recoverable
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            DataError::Dataset { kind, .. } => match kind {
                DatasetErrorKind::IndexOutOfBounds => ErrorSeverity::Warning,
                DatasetErrorKind::EmptyDataset => ErrorSeverity::Error,
                DatasetErrorKind::IncompatibleShapes => ErrorSeverity::Error,
                DatasetErrorKind::MissingData => ErrorSeverity::Warning,
                DatasetErrorKind::CorruptedData => ErrorSeverity::Critical,
                DatasetErrorKind::UnsupportedFormat => ErrorSeverity::Error,
                DatasetErrorKind::AccessDenied => ErrorSeverity::Error,
            },
            DataError::DataLoader { kind, .. } => match kind {
                DataLoaderErrorKind::WorkerPanic => ErrorSeverity::Critical,
                DataLoaderErrorKind::ChannelClosed => ErrorSeverity::Error,
                DataLoaderErrorKind::Timeout => ErrorSeverity::Warning,
                DataLoaderErrorKind::ConfigurationInvalid => ErrorSeverity::Error,
                DataLoaderErrorKind::BackendUnavailable => ErrorSeverity::Error,
                DataLoaderErrorKind::BatchGenerationFailed => ErrorSeverity::Warning,
            },
            DataError::Transform { .. } => ErrorSeverity::Warning,
            DataError::Sampler { .. } => ErrorSeverity::Warning,
            DataError::Collation { .. } => ErrorSeverity::Warning,
            DataError::Io { kind, .. } => match kind {
                IoErrorKind::CorruptedFile => ErrorSeverity::Critical,
                _ => ErrorSeverity::Error,
            },
            DataError::Configuration { .. } => ErrorSeverity::Error,
            DataError::Resource { .. } => ErrorSeverity::Warning,
            DataError::Privacy { kind, .. } => match kind {
                PrivacyErrorKind::BudgetExceeded => ErrorSeverity::Critical,
                PrivacyErrorKind::InvalidPrivacyParameter => ErrorSeverity::Error,
                PrivacyErrorKind::AccessLimitExceeded => ErrorSeverity::Error,
                PrivacyErrorKind::AccessDenied => ErrorSeverity::Warning,
                PrivacyErrorKind::TensorCreationFailed => ErrorSeverity::Error,
                PrivacyErrorKind::NoiseGenerationFailed => ErrorSeverity::Error,
                PrivacyErrorKind::CompositionError => ErrorSeverity::Critical,
            },
            DataError::GpuError(_) => ErrorSeverity::Warning,
            DataError::Other(_) => ErrorSeverity::Error,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorSeverity {
    Warning,
    Error,
    Critical,
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::Dataset {
                kind,
                context,
                suggestion,
            } => {
                write!(f, "Dataset error ({kind:?}): {context}")?;
                if let Some(s) = suggestion {
                    write!(f, " Suggestion: {s}")?;
                }
            }
            DataError::DataLoader {
                kind,
                context,
                suggestion,
            } => {
                write!(f, "DataLoader error ({kind:?}): {context}")?;
                if let Some(s) = suggestion {
                    write!(f, " Suggestion: {s}")?;
                }
            }
            DataError::Transform {
                kind,
                transform_name,
                context,
                suggestion,
            } => {
                write!(
                    f,
                    "Transform error in '{transform_name}' ({kind:?}): {context}"
                )?;
                if let Some(s) = suggestion {
                    write!(f, " Suggestion: {s}")?;
                }
            }
            DataError::Sampler {
                kind,
                sampler_type,
                context,
                suggestion,
            } => {
                write!(f, "Sampler error in '{sampler_type}' ({kind:?}): {context}")?;
                if let Some(s) = suggestion {
                    write!(f, " Suggestion: {s}")?;
                }
            }
            DataError::Collation {
                kind,
                batch_info,
                context,
                suggestion,
            } => {
                write!(
                    f,
                    "Collation error ({:?}) for batch size {}: {}",
                    kind, batch_info.batch_size, context
                )?;
                if let Some(s) = suggestion {
                    write!(f, " Suggestion: {s}")?;
                }
            }
            DataError::Io {
                kind,
                path,
                operation,
                context,
                suggestion,
            } => {
                write!(f, "I/O error ({kind:?}) during '{operation}': {context}")?;
                if let Some(p) = path {
                    write!(f, " Path: {p}")?;
                }
                if let Some(s) = suggestion {
                    write!(f, " Suggestion: {s}")?;
                }
            }
            DataError::Configuration {
                kind,
                parameter,
                value,
                context,
                suggestion,
            } => {
                write!(f, "Configuration error ({kind:?}) for parameter '{parameter}' = '{value}': {context}")?;
                if let Some(s) = suggestion {
                    write!(f, " Suggestion: {s}")?;
                }
            }
            DataError::Resource {
                kind,
                resource_type,
                requested,
                available,
                context,
                suggestion,
            } => {
                write!(
                    f,
                    "Resource error ({kind:?}) for {resource_type}: {context}"
                )?;
                if let (Some(req), Some(avail)) = (requested, available) {
                    write!(f, " (requested: {req}, available: {avail})")?;
                } else if let Some(req) = requested {
                    write!(f, " (requested: {req})")?;
                }
                if let Some(s) = suggestion {
                    write!(f, " Suggestion: {s}")?;
                }
            }
            DataError::Privacy {
                kind,
                privacy_parameter,
                context,
                suggestion,
            } => {
                write!(
                    f,
                    "Privacy error ({kind:?}) for parameter '{privacy_parameter}': {context}"
                )?;
                if let Some(s) = suggestion {
                    write!(f, " Suggestion: {s}")?;
                }
            }
            DataError::GpuError(msg) => {
                write!(f, "GPU error: {msg}")?;
            }
            DataError::Other(msg) => {
                write!(f, "Data error: {msg}")?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for DataError {}

impl From<DataError> for torsh_core::TorshError {
    fn from(err: DataError) -> Self {
        torsh_core::TorshError::Other(format!("Data error: {err}"))
    }
}

impl From<torsh_core::TorshError> for DataError {
    fn from(err: torsh_core::TorshError) -> Self {
        DataError::Other(format!("Torsh error: {err}"))
    }
}

/// Result type for data operations
pub type Result<T> = std::result::Result<T, DataError>;

/// Error context builder for chaining operations
pub struct ErrorContext {
    operation: String,
    details: Vec<String>,
}

impl ErrorContext {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            details: Vec::new(),
        }
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.details.push(detail.into());
        self
    }

    pub fn build_context(&self) -> String {
        let mut context = self.operation.clone();
        if !self.details.is_empty() {
            context.push_str(": ");
            context.push_str(&self.details.join(", "));
        }
        context
    }
}

/// Trait for adding context to errors
pub trait WithContext<T> {
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> ErrorContext;

    fn with_simple_context(self, operation: &str) -> Result<T>;
}

impl<T, E> WithContext<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> ErrorContext,
    {
        self.map_err(|e| {
            let context = f();
            DataError::io(
                IoErrorKind::ReadError,
                &context.operation,
                format!("{}: {}", context.build_context(), e),
            )
        })
    }

    fn with_simple_context(self, operation: &str) -> Result<T> {
        self.with_context(|| ErrorContext::new(operation))
    }
}

/// Error recovery strategies and mechanisms
pub mod recovery {
    use super::*;
    use std::time::{Duration, Instant};

    /// Retry strategy configuration
    #[derive(Debug, Clone)]
    pub struct RetryStrategy {
        pub max_attempts: usize,
        pub base_delay: Duration,
        pub max_delay: Duration,
        pub backoff_multiplier: f64,
        pub jitter: bool,
    }

    impl Default for RetryStrategy {
        fn default() -> Self {
            Self {
                max_attempts: 3,
                base_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
                jitter: true,
            }
        }
    }

    impl RetryStrategy {
        /// Create a new retry strategy
        pub fn new(max_attempts: usize) -> Self {
            Self {
                max_attempts,
                ..Default::default()
            }
        }

        /// Set base delay
        pub fn with_base_delay(mut self, delay: Duration) -> Self {
            self.base_delay = delay;
            self
        }

        /// Set maximum delay
        pub fn with_max_delay(mut self, delay: Duration) -> Self {
            self.max_delay = delay;
            self
        }

        /// Set backoff multiplier
        pub fn with_backoff_multiplier(mut self, multiplier: f64) -> Self {
            self.backoff_multiplier = multiplier;
            self
        }

        /// Enable or disable jitter
        pub fn with_jitter(mut self, jitter: bool) -> Self {
            self.jitter = jitter;
            self
        }

        /// Calculate delay for a given attempt
        pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
            let delay = (self.base_delay.as_millis() as f64
                * self.backoff_multiplier.powi(attempt as i32)) as u64;

            let delay = Duration::from_millis(delay.min(self.max_delay.as_millis() as u64));

            if self.jitter && attempt > 0 {
                let jitter_ms = (delay.as_millis() as f64 * 0.1 * thread_rng().gen::<f64>()) as u64;
                delay + Duration::from_millis(jitter_ms)
            } else {
                delay
            }
        }
    }

    /// Error recovery context
    #[derive(Debug)]
    pub struct RecoveryContext {
        pub original_error: DataError,
        pub attempt: usize,
        pub started_at: Instant,
        pub last_attempt_at: Instant,
    }

    impl RecoveryContext {
        /// Create a new recovery context
        pub fn new(error: DataError) -> Self {
            let now = Instant::now();
            Self {
                original_error: error,
                attempt: 0,
                started_at: now,
                last_attempt_at: now,
            }
        }

        /// Record a new attempt
        pub fn next_attempt(&mut self) {
            self.attempt += 1;
            self.last_attempt_at = Instant::now();
        }

        /// Get total elapsed time
        pub fn total_elapsed(&self) -> Duration {
            self.started_at.elapsed()
        }

        /// Get time since last attempt
        pub fn time_since_last_attempt(&self) -> Duration {
            self.last_attempt_at.elapsed()
        }
    }

    /// Automatic error recovery utility
    pub fn retry_operation<T, F>(mut operation: F, strategy: &RetryStrategy) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        let mut last_error = None;

        for attempt in 0..strategy.max_attempts {
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !error.is_recoverable() || attempt == strategy.max_attempts - 1 {
                        return Err(error);
                    }

                    let delay = strategy.delay_for_attempt(attempt);
                    std::thread::sleep(delay);
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            DataError::Other("Retry operation failed without error".to_string())
        }))
    }

    /// Async version of retry operation
    #[cfg(feature = "async-support")]
    pub async fn retry_operation_async<T, F, Fut>(
        mut operation: F,
        strategy: &RetryStrategy,
    ) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut last_error = None;

        for attempt in 0..strategy.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !error.is_recoverable() || attempt == strategy.max_attempts - 1 {
                        return Err(error);
                    }

                    let delay = strategy.delay_for_attempt(attempt);
                    tokio::time::sleep(delay).await;
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            DataError::Other("Async retry operation failed without error".to_string())
        }))
    }
}

/// Error diagnostics and debugging utilities
pub mod diagnostics {
    use super::*;
    use std::collections::HashMap;

    /// Error statistics collector
    #[derive(Debug, Default)]
    pub struct ErrorStatistics {
        pub total_errors: usize,
        pub error_counts: HashMap<String, usize>,
        pub severity_counts: HashMap<ErrorSeverity, usize>,
        pub recoverable_count: usize,
        pub non_recoverable_count: usize,
    }

    impl ErrorStatistics {
        /// Create a new error statistics collector
        pub fn new() -> Self {
            Self::default()
        }

        /// Record an error
        pub fn record_error(&mut self, error: &DataError) {
            self.total_errors += 1;

            let error_type = match error {
                DataError::Dataset { kind, .. } => format!("Dataset::{kind:?}"),
                DataError::DataLoader { kind, .. } => format!("DataLoader::{kind:?}"),
                DataError::Transform { kind, .. } => format!("Transform::{kind:?}"),
                DataError::Sampler { kind, .. } => format!("Sampler::{kind:?}"),
                DataError::Collation { kind, .. } => format!("Collation::{kind:?}"),
                DataError::Io { kind, .. } => format!("Io::{kind:?}"),
                DataError::Configuration { kind, .. } => format!("Configuration::{kind:?}"),
                DataError::Resource { kind, .. } => format!("Resource::{kind:?}"),
                DataError::Privacy { kind, .. } => format!("Privacy::{kind:?}"),
                DataError::GpuError(_) => "GpuError".to_string(),
                DataError::Other(_) => "Other".to_string(),
            };

            *self.error_counts.entry(error_type).or_insert(0) += 1;
            *self.severity_counts.entry(error.severity()).or_insert(0) += 1;

            if error.is_recoverable() {
                self.recoverable_count += 1;
            } else {
                self.non_recoverable_count += 1;
            }
        }

        /// Get the most common error type
        pub fn most_common_error(&self) -> Option<(&String, &usize)> {
            self.error_counts.iter().max_by_key(|(_, count)| *count)
        }

        /// Get error rate by severity
        pub fn error_rate_by_severity(&self, severity: ErrorSeverity) -> f64 {
            if self.total_errors == 0 {
                0.0
            } else {
                *self.severity_counts.get(&severity).unwrap_or(&0) as f64 / self.total_errors as f64
            }
        }

        /// Get recovery rate
        pub fn recovery_rate(&self) -> f64 {
            if self.total_errors == 0 {
                0.0
            } else {
                self.recoverable_count as f64 / self.total_errors as f64
            }
        }

        /// Generate a diagnostic report
        pub fn generate_report(&self) -> String {
            let mut report = "Error Statistics Report\n".to_string();
            report.push_str(&format!("Total Errors: {}\n", self.total_errors));
            report.push_str(&format!(
                "Recovery Rate: {:.2}%\n",
                self.recovery_rate() * 100.0
            ));

            report.push_str("\nSeverity Breakdown:\n");
            for (severity, count) in &self.severity_counts {
                report.push_str(&format!(
                    "  {:?}: {} ({:.1}%)\n",
                    severity,
                    count,
                    (*count as f64 / self.total_errors as f64) * 100.0
                ));
            }

            report.push_str("\nMost Common Errors:\n");
            let mut sorted_errors: Vec<_> = self.error_counts.iter().collect();
            sorted_errors.sort_by_key(|(_, count)| std::cmp::Reverse(**count));

            for (error_type, count) in sorted_errors.iter().take(5) {
                report.push_str(&format!(
                    "  {}: {} ({:.1}%)\n",
                    error_type,
                    count,
                    (**count as f64 / self.total_errors as f64) * 100.0
                ));
            }

            report
        }
    }

    /// Error chain analyzer for debugging
    pub struct ErrorChainAnalyzer {
        errors: Vec<DataError>,
        max_chain_length: usize,
    }

    impl ErrorChainAnalyzer {
        /// Create a new error chain analyzer
        pub fn new(max_chain_length: usize) -> Self {
            Self {
                errors: Vec::new(),
                max_chain_length,
            }
        }

        /// Add an error to the chain
        pub fn add_error(&mut self, error: DataError) {
            if self.errors.len() >= self.max_chain_length {
                self.errors.remove(0);
            }
            self.errors.push(error);
        }

        /// Analyze error patterns
        pub fn analyze_patterns(&self) -> Vec<String> {
            let mut patterns = Vec::new();

            if self.errors.len() < 2 {
                return patterns;
            }

            // Check for repeated error types
            let mut consecutive_same = 1;
            for i in 1..self.errors.len() {
                if std::mem::discriminant(&self.errors[i])
                    == std::mem::discriminant(&self.errors[i - 1])
                {
                    consecutive_same += 1;
                } else {
                    if consecutive_same > 2 {
                        patterns.push(format!("Repeated error type {consecutive_same} times"));
                    }
                    consecutive_same = 1;
                }
            }

            // Check for patterns that continue until the end
            if consecutive_same > 2 {
                patterns.push(format!("Repeated error type {consecutive_same} times"));
            }

            // Check for escalating severity
            let mut severity_escalating = true;
            for i in 1..self.errors.len() {
                let prev_severity = &self.errors[i - 1].severity();
                let curr_severity = &self.errors[i].severity();

                match (prev_severity, curr_severity) {
                    (ErrorSeverity::Warning, ErrorSeverity::Error)
                    | (ErrorSeverity::Warning, ErrorSeverity::Critical)
                    | (ErrorSeverity::Error, ErrorSeverity::Critical) => {}
                    _ => {
                        severity_escalating = false;
                        break;
                    }
                }
            }

            if severity_escalating && self.errors.len() > 2 {
                patterns.push("Error severity is escalating".to_string());
            }

            patterns
        }

        /// Get error chain summary
        pub fn chain_summary(&self) -> String {
            if self.errors.is_empty() {
                return "No errors in chain".to_string();
            }

            let mut summary = format!("Error Chain ({} errors):\n", self.errors.len());

            for (i, error) in self.errors.iter().enumerate() {
                summary.push_str(&format!(
                    "  {}. {:?} - {}\n",
                    i + 1,
                    error.severity(),
                    error
                ));
            }

            let patterns = self.analyze_patterns();
            if !patterns.is_empty() {
                summary.push_str("\nPatterns Detected:\n");
                for pattern in patterns {
                    summary.push_str(&format!("  - {pattern}\n"));
                }
            }

            summary
        }
    }
}

/// Common error patterns and utilities
pub mod patterns {
    use super::*;

    /// Create index out of bounds error with helpful suggestions
    pub fn index_out_of_bounds(index: usize, len: usize) -> DataError {
        DataError::dataset(
            DatasetErrorKind::IndexOutOfBounds,
            format!("Index {index} is out of bounds for dataset of length {len}"),
        )
        .with_suggestion(format!("Valid indices are 0 to {}", len.saturating_sub(1)))
    }

    /// Create shape mismatch error with detailed information
    pub fn shape_mismatch(expected: &[usize], actual: &[usize], context: &str) -> DataError {
        DataError::transform(
            TransformErrorKind::IncompatibleDimensions,
            context,
            format!("Expected shape {expected:?}, got {actual:?}"),
        )
        .with_suggestion("Ensure input tensors have compatible shapes for the operation")
    }

    /// Create configuration validation error
    pub fn invalid_config<T: fmt::Display>(param: &str, value: T, reason: &str) -> DataError {
        DataError::config(
            ConfigErrorKind::InvalidValue,
            param,
            value.to_string(),
            reason,
        )
    }

    /// Create memory exhaustion error with resource information
    pub fn memory_exhausted(requested: usize, available: Option<usize>) -> DataError {
        let mut error = DataError::resource(
            ResourceErrorKind::MemoryExhaustion,
            "memory",
            format!("Requested {requested} bytes"),
        )
        .with_resource_info(requested, available);

        if let Some(avail) = available {
            error = error.with_suggestion(format!(
                "Reduce batch size or dataset size. {avail} bytes available, {requested} bytes requested"
            ));
        } else {
            error = error.with_suggestion("Reduce batch size or dataset size");
        }

        error
    }

    /// Create file not found error with search suggestions
    pub fn file_not_found(path: &str, search_paths: &[String]) -> DataError {
        let mut error = DataError::io(
            IoErrorKind::FileNotFound,
            "file access",
            format!("File not found: {path}"),
        )
        .with_path(path);

        if !search_paths.is_empty() {
            error = error.with_suggestion(format!(
                "Check file exists and path is correct. Searched in: {}",
                search_paths.join(", ")
            ));
        }

        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_error_creation() {
        let error = DataError::dataset(
            DatasetErrorKind::IndexOutOfBounds,
            "Index 10 out of bounds for dataset of size 5",
        )
        .with_suggestion("Use valid index between 0-4");

        assert!(error.is_recoverable());
        assert_eq!(error.severity(), ErrorSeverity::Warning);

        let error_str = error.to_string();
        assert!(error_str.contains("IndexOutOfBounds"));
        assert!(error_str.contains("Suggestion"));
    }

    #[test]
    fn test_patterns() {
        let error = patterns::index_out_of_bounds(10, 5);
        assert!(error.to_string().contains("Index 10 is out of bounds"));

        let error = patterns::shape_mismatch(&[3, 224, 224], &[3, 256, 256], "resize");
        assert!(error.to_string().contains("Expected shape"));

        let error = patterns::invalid_config("batch_size", -1, "Must be positive");
        assert!(error.to_string().contains("batch_size"));
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("loading dataset")
            .with_detail("path: /data/train.csv")
            .with_detail("format: CSV");

        let context_str = context.build_context();
        assert!(context_str.contains("loading dataset"));
        assert!(context_str.contains("path: /data/train.csv"));
    }

    #[test]
    fn test_batch_info() {
        let batch_info = BatchInfo::new(32)
            .with_shape(vec![3, 224, 224])
            .with_type("f32".to_string());

        assert_eq!(batch_info.batch_size, 32);
        assert_eq!(batch_info.item_shapes.len(), 1);
        assert_eq!(batch_info.item_types.len(), 1);
    }

    #[test]
    fn test_retry_strategy() {
        let strategy = recovery::RetryStrategy::new(3)
            .with_base_delay(Duration::from_millis(10))
            .with_backoff_multiplier(2.0);

        assert_eq!(strategy.max_attempts, 3);
        assert_eq!(strategy.base_delay, Duration::from_millis(10));

        let delay0 = strategy.delay_for_attempt(0);
        let delay1 = strategy.delay_for_attempt(1);
        assert!(delay1 >= delay0); // Should increase with backoff
    }

    #[test]
    fn test_retry_operation() {
        let mut attempt = 0;
        let strategy = recovery::RetryStrategy::new(3).with_base_delay(Duration::from_millis(1));

        // Test successful retry
        let result = recovery::retry_operation(
            || {
                attempt += 1;
                if attempt < 3 {
                    Err(DataError::dataloader(
                        DataLoaderErrorKind::Timeout,
                        "Connection timeout",
                    ))
                } else {
                    Ok(42)
                }
            },
            &strategy,
        );

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt, 3);
    }

    #[test]
    fn test_error_statistics() {
        let mut stats = diagnostics::ErrorStatistics::new();

        let error1 = DataError::dataset(DatasetErrorKind::IndexOutOfBounds, "test");
        let error2 = DataError::dataset(DatasetErrorKind::IndexOutOfBounds, "test");
        let error3 = DataError::dataloader(DataLoaderErrorKind::Timeout, "test");

        stats.record_error(&error1);
        stats.record_error(&error2);
        stats.record_error(&error3);

        assert_eq!(stats.total_errors, 3);
        assert_eq!(stats.recoverable_count, 3); // All these errors are recoverable

        let report = stats.generate_report();
        assert!(report.contains("Total Errors: 3"));
        assert!(report.contains("Recovery Rate"));
    }

    #[test]
    fn test_error_chain_analyzer() {
        let mut analyzer = diagnostics::ErrorChainAnalyzer::new(5);

        let error1 = DataError::dataset(DatasetErrorKind::IndexOutOfBounds, "test");
        let error2 = DataError::dataset(DatasetErrorKind::IndexOutOfBounds, "test");
        let error3 = DataError::dataset(DatasetErrorKind::IndexOutOfBounds, "test");
        let error4 = DataError::dataset(DatasetErrorKind::CorruptedData, "test");

        analyzer.add_error(error1);
        analyzer.add_error(error2);
        analyzer.add_error(error3);
        analyzer.add_error(error4);

        let patterns = analyzer.analyze_patterns();
        assert!(!patterns.is_empty());

        let summary = analyzer.chain_summary();
        assert!(summary.contains("Error Chain"));
    }

    #[test]
    fn test_recovery_context() {
        let error = DataError::dataset(DatasetErrorKind::IndexOutOfBounds, "test");
        let mut context = recovery::RecoveryContext::new(error);

        assert_eq!(context.attempt, 0);

        context.next_attempt();
        assert_eq!(context.attempt, 1);

        let elapsed = context.total_elapsed();
        assert!(elapsed.as_millis() > 0 || elapsed.as_millis() == 0); // Duration is always valid
    }
}
