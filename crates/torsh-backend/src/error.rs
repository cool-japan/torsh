//! Unified backend error handling using TorshError
//!
//! This module provides consistent error handling across all backends
//! by standardizing on TorshError from torsh-core.

// Re-export TorshError types for unified error handling
pub use torsh_core::error::{Result as BackendResult, TorshError as BackendError};

/// Error severity levels for better error classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrorSeverity {
    /// Information level - not an error but important info
    Info,
    /// Warning level - potential issue but recoverable
    Warning,
    /// Error level - definite error but system can continue
    Error,
    /// Critical level - critical error requiring immediate attention
    Critical,
    /// Fatal level - unrecoverable error, system must stop
    Fatal,
}

/// Error category for better error classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Memory related errors (allocation, deallocation, etc.)
    Memory,
    /// Computation errors (kernel execution, numerical errors)
    Computation,
    /// Hardware/device related errors
    Hardware,
    /// Configuration and initialization errors
    Configuration,
    /// Network and communication errors
    Communication,
    /// Input/output errors
    Io,
    /// Security related errors
    Security,
    /// Resource exhaustion errors
    ResourceExhaustion,
    /// Timeout related errors
    Timeout,
    /// Validation and parameter errors
    Validation,
    /// Internal system errors
    Internal,
    /// External dependency errors
    External,
}

/// Enhanced error context information for better debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The operation being performed when error occurred
    pub operation: String,
    /// Device where the error occurred
    pub device: Option<String>,
    /// Backend type where the error occurred
    pub backend: Option<String>,
    /// Additional context information
    pub details: Option<String>,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Error category
    pub category: ErrorCategory,
    /// Timestamp when error occurred
    pub timestamp: std::time::SystemTime,
    /// Source location (file:line) where error originated
    pub source_location: Option<String>,
    /// Thread ID where error occurred
    pub thread_id: Option<String>,
    /// Error chain - previous error that led to this one
    pub cause: Option<Box<ErrorContext>>,
    /// Suggested recovery actions
    pub recovery_suggestions: Vec<String>,
    /// Error code for programmatic handling
    pub error_code: Option<String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            device: None,
            backend: None,
            details: None,
            severity: ErrorSeverity::Error,
            category: ErrorCategory::Internal,
            timestamp: std::time::SystemTime::now(),
            source_location: None,
            thread_id: Some(format!("{:?}", std::thread::current().id())),
            cause: None,
            recovery_suggestions: Vec::new(),
            error_code: None,
        }
    }

    /// Create a new error context with category and severity
    pub fn new_with_category(
        operation: impl Into<String>,
        category: ErrorCategory,
        severity: ErrorSeverity,
    ) -> Self {
        Self {
            operation: operation.into(),
            device: None,
            backend: None,
            details: None,
            severity,
            category,
            timestamp: std::time::SystemTime::now(),
            source_location: None,
            thread_id: Some(format!("{:?}", std::thread::current().id())),
            cause: None,
            recovery_suggestions: Vec::new(),
            error_code: None,
        }
    }

    /// Add device information
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = Some(device.into());
        self
    }

    /// Add backend information
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backend = Some(backend.into());
        self
    }

    /// Add additional details
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Set error severity
    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set error category
    pub fn with_category(mut self, category: ErrorCategory) -> Self {
        self.category = category;
        self
    }

    /// Add source location information
    pub fn with_source_location(mut self, location: impl Into<String>) -> Self {
        self.source_location = Some(location.into());
        self
    }

    /// Add error code
    pub fn with_error_code(mut self, code: impl Into<String>) -> Self {
        self.error_code = Some(code.into());
        self
    }

    /// Add a recovery suggestion
    pub fn add_recovery_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.recovery_suggestions.push(suggestion.into());
        self
    }

    /// Add multiple recovery suggestions
    pub fn with_recovery_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.recovery_suggestions = suggestions;
        self
    }

    /// Chain this error with a previous error
    pub fn with_cause(mut self, cause: ErrorContext) -> Self {
        self.cause = Some(Box::new(cause));
        self
    }

    /// Convert error context to a formatted string
    pub fn format(&self) -> String {
        let mut parts = vec![
            format!("[{}]", self.severity_string()),
            format!("[{}]", self.category_string()),
            self.operation.clone(),
        ];

        if let Some(ref backend) = self.backend {
            parts.push(format!("backend: {}", backend));
        }

        if let Some(ref device) = self.device {
            parts.push(format!("device: {}", device));
        }

        if let Some(ref error_code) = self.error_code {
            parts.push(format!("code: {}", error_code));
        }

        if let Some(ref details) = self.details {
            parts.push(format!("details: {}", details));
        }

        if let Some(ref location) = self.source_location {
            parts.push(format!("at: {}", location));
        }

        let mut result = parts.join(", ");

        // Add recovery suggestions if available
        if !self.recovery_suggestions.is_empty() {
            result.push_str(&format!(
                "\nSuggested recovery actions: {}",
                self.recovery_suggestions.join("; ")
            ));
        }

        // Add cause chain if available
        if let Some(ref cause) = self.cause {
            result.push_str(&format!("\nCaused by: {}", cause.format()));
        }

        result
    }

    /// Get severity as string
    pub fn severity_string(&self) -> &'static str {
        match self.severity {
            ErrorSeverity::Info => "INFO",
            ErrorSeverity::Warning => "WARN",
            ErrorSeverity::Error => "ERROR",
            ErrorSeverity::Critical => "CRITICAL",
            ErrorSeverity::Fatal => "FATAL",
        }
    }

    /// Get category as string
    pub fn category_string(&self) -> &'static str {
        match self.category {
            ErrorCategory::Memory => "MEMORY",
            ErrorCategory::Computation => "COMPUTE",
            ErrorCategory::Hardware => "HARDWARE",
            ErrorCategory::Configuration => "CONFIG",
            ErrorCategory::Communication => "COMM",
            ErrorCategory::Io => "IO",
            ErrorCategory::Security => "SECURITY",
            ErrorCategory::ResourceExhaustion => "RESOURCE",
            ErrorCategory::Timeout => "TIMEOUT",
            ErrorCategory::Validation => "VALIDATION",
            ErrorCategory::Internal => "INTERNAL",
            ErrorCategory::External => "EXTERNAL",
        }
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self.severity,
            ErrorSeverity::Info | ErrorSeverity::Warning | ErrorSeverity::Error
        ) && !self.recovery_suggestions.is_empty()
    }

    /// Get the depth of the error chain
    pub fn chain_depth(&self) -> usize {
        match &self.cause {
            Some(cause) => 1 + cause.chain_depth(),
            None => 0,
        }
    }
}

/// Helper trait for adding context to errors
pub trait ErrorContextExt<T> {
    /// Add operation context to the error
    fn with_context(self, context: ErrorContext) -> BackendResult<T>;

    /// Add simple operation context
    fn with_operation(self, operation: &str) -> BackendResult<T>;
}

impl<T, E> ErrorContextExt<T> for Result<T, E>
where
    E: Into<BackendError>,
{
    fn with_context(self, context: ErrorContext) -> BackendResult<T> {
        self.map_err(|e| {
            let base_error = e.into();
            match base_error {
                BackendError::BackendError(msg) => {
                    BackendError::BackendError(format!("{}: {}", context.format(), msg))
                }
                BackendError::ComputeError(msg) => {
                    BackendError::ComputeError(format!("{}: {}", context.format(), msg))
                }
                BackendError::AllocationError(msg) => {
                    BackendError::AllocationError(format!("{}: {}", context.format(), msg))
                }
                other => other, // Keep other error types as-is
            }
        })
    }

    fn with_operation(self, operation: &str) -> BackendResult<T> {
        self.with_context(ErrorContext::new(operation))
    }
}

/// Macros for simplified error creation
#[macro_export]
macro_rules! backend_error {
    ($operation:expr, $category:expr, $severity:expr, $message:expr) => {
        $crate::error::BackendError::BackendError(
            $crate::error::ErrorContext::new_with_category($operation, $category, $severity)
                .format()
                + ": "
                + &$message.to_string(),
        )
    };
    ($operation:expr, $message:expr) => {
        backend_error!(
            $operation,
            $crate::error::ErrorCategory::Internal,
            $crate::error::ErrorSeverity::Error,
            $message
        )
    };
}

#[macro_export]
macro_rules! compute_error {
    ($operation:expr, $backend:expr, $device:expr, $message:expr) => {
        $crate::error::BackendError::BackendError(
            $crate::error::ErrorContext::new_with_category(
                $operation,
                $crate::error::ErrorCategory::Computation,
                $crate::error::ErrorSeverity::Error,
            )
            .with_backend($backend)
            .with_device($device)
            .format()
                + ": "
                + &$message.to_string(),
        )
    };
}

#[macro_export]
macro_rules! memory_error {
    ($size:expr, $backend:expr, $device:expr, $message:expr) => {
        $crate::error::conversion::memory_error_with_context(
            $message,
            $size,
            $backend,
            Some($device),
        )
    };
    ($size:expr, $backend:expr, $message:expr) => {
        $crate::error::conversion::memory_error_with_context($message, $size, $backend, None)
    };
}

/// Standardized error reporting and logging system
pub struct ErrorReporter {
    backend_name: String,
    enable_logging: bool,
    error_callback: Option<Box<dyn Fn(&ErrorContext) + Send + Sync>>,
}

impl ErrorReporter {
    pub fn new(backend_name: String) -> Self {
        Self {
            backend_name,
            enable_logging: true,
            error_callback: None,
        }
    }

    pub fn with_logging(mut self, enable: bool) -> Self {
        self.enable_logging = enable;
        self
    }

    pub fn with_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&ErrorContext) + Send + Sync + 'static,
    {
        self.error_callback = Some(Box::new(callback));
        self
    }

    pub fn report_error(&self, error: &BackendError) {
        if self.enable_logging {
            match error {
                BackendError::BackendError(msg) => {
                    eprintln!("[{}] Backend Error: {}", self.backend_name, msg)
                }
                BackendError::ComputeError(msg) => {
                    eprintln!("[{}] Compute Error: {}", self.backend_name, msg)
                }
                BackendError::AllocationError(msg) => {
                    eprintln!("[{}] Memory Error: {}", self.backend_name, msg)
                }
                _ => eprintln!("[{}] Error: {:?}", self.backend_name, error),
            }
        }
    }

    pub fn report_context(&self, context: &ErrorContext) {
        if self.enable_logging {
            eprintln!("[{}] {}", self.backend_name, context.format());
        }

        if let Some(ref callback) = self.error_callback {
            callback(context);
        }
    }
}

/// Error recovery system for automatic error handling
pub struct ErrorRecoverySystem {
    recovery_strategies: std::collections::HashMap<ErrorCategory, Vec<RecoveryStrategy>>,
    max_retry_attempts: u32,
}

#[derive(Debug, Clone)]
pub struct RecoveryStrategy {
    pub name: String,
    pub description: String,
    pub auto_retry: bool,
    pub max_attempts: u32,
    pub backoff_ms: u64,
}

impl ErrorRecoverySystem {
    pub fn new() -> Self {
        let mut system = Self {
            recovery_strategies: std::collections::HashMap::new(),
            max_retry_attempts: 3,
        };

        // Add default recovery strategies
        system.add_default_strategies();
        system
    }

    fn add_default_strategies(&mut self) {
        // Memory errors
        self.add_strategy(
            ErrorCategory::Memory,
            RecoveryStrategy {
                name: "garbage_collection".to_string(),
                description: "Force garbage collection and retry".to_string(),
                auto_retry: true,
                max_attempts: 2,
                backoff_ms: 100,
            },
        );

        // Timeout errors
        self.add_strategy(
            ErrorCategory::Timeout,
            RecoveryStrategy {
                name: "exponential_backoff".to_string(),
                description: "Retry with exponential backoff".to_string(),
                auto_retry: true,
                max_attempts: 3,
                backoff_ms: 1000,
            },
        );

        // Hardware errors
        self.add_strategy(
            ErrorCategory::Hardware,
            RecoveryStrategy {
                name: "device_reset".to_string(),
                description: "Reset device context and retry".to_string(),
                auto_retry: false,
                max_attempts: 1,
                backoff_ms: 5000,
            },
        );
    }

    pub fn add_strategy(&mut self, category: ErrorCategory, strategy: RecoveryStrategy) {
        self.recovery_strategies
            .entry(category)
            .or_insert_with(Vec::new)
            .push(strategy);
    }

    pub fn get_recovery_strategies(
        &self,
        category: &ErrorCategory,
    ) -> Option<&Vec<RecoveryStrategy>> {
        self.recovery_strategies.get(category)
    }

    pub fn should_auto_retry(&self, context: &ErrorContext) -> bool {
        if let Some(strategies) = self.get_recovery_strategies(&context.category) {
            strategies
                .iter()
                .any(|s| s.auto_retry && context.severity <= ErrorSeverity::Error)
        } else {
            false
        }
    }
}

impl Default for ErrorRecoverySystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified error tracking and analytics
#[derive(Debug, Clone, Default)]
pub struct ErrorStatistics {
    pub total_errors: u64,
    pub errors_by_category: std::collections::HashMap<ErrorCategory, u64>,
    pub errors_by_severity: std::collections::HashMap<ErrorSeverity, u64>,
    pub errors_by_backend: std::collections::HashMap<String, u64>,
    pub recovery_attempts: u64,
    pub successful_recoveries: u64,
}

impl ErrorStatistics {
    pub fn record_error(&mut self, context: &ErrorContext) {
        self.total_errors += 1;
        *self
            .errors_by_category
            .entry(context.category.clone())
            .or_insert(0) += 1;
        *self.errors_by_severity.entry(context.severity).or_insert(0) += 1;

        if let Some(ref backend) = context.backend {
            *self.errors_by_backend.entry(backend.clone()).or_insert(0) += 1;
        }
    }

    pub fn record_recovery_attempt(&mut self) {
        self.recovery_attempts += 1;
    }

    pub fn record_successful_recovery(&mut self) {
        self.successful_recoveries += 1;
    }

    pub fn get_recovery_success_rate(&self) -> f64 {
        if self.recovery_attempts > 0 {
            self.successful_recoveries as f64 / self.recovery_attempts as f64
        } else {
            0.0
        }
    }

    pub fn get_most_frequent_error_category(&self) -> Option<(&ErrorCategory, &u64)> {
        self.errors_by_category
            .iter()
            .max_by_key(|(_, count)| *count)
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Backend-specific error conversion utilities
pub mod conversion {
    use super::*;

    /// Convert CUDA errors to TorshError with proper context
    pub fn cuda_error_with_context(
        error: impl std::fmt::Display,
        operation: &str,
        device_id: Option<usize>,
    ) -> BackendError {
        let mut context = ErrorContext::new_with_category(
            operation,
            ErrorCategory::Hardware,
            ErrorSeverity::Error,
        )
        .with_backend("CUDA")
        .with_error_code("CUDA_ERROR")
        .add_recovery_suggestion("Check CUDA installation and driver version")
        .add_recovery_suggestion("Verify GPU memory availability")
        .add_recovery_suggestion("Try reducing batch size or model size");

        if let Some(device_id) = device_id {
            context = context.with_device(format!("cuda:{}", device_id));
        }

        BackendError::BackendError(format!("{}: {}", context.format(), error))
    }

    /// Convert CPU errors to TorshError with proper context
    pub fn cpu_error_with_context(error: impl std::fmt::Display, operation: &str) -> BackendError {
        let context = ErrorContext::new_with_category(
            operation,
            ErrorCategory::Computation,
            ErrorSeverity::Error,
        )
        .with_backend("CPU")
        .with_error_code("CPU_ERROR")
        .add_recovery_suggestion("Check system memory availability")
        .add_recovery_suggestion("Reduce number of parallel threads")
        .add_recovery_suggestion("Try smaller input sizes");

        BackendError::BackendError(format!("{}: {}", context.format(), error))
    }

    /// Convert Metal errors to TorshError with proper context
    pub fn metal_error_with_context(
        error: impl std::fmt::Display,
        operation: &str,
        device_id: Option<usize>,
    ) -> BackendError {
        let mut context = ErrorContext::new_with_category(
            operation,
            ErrorCategory::Hardware,
            ErrorSeverity::Error,
        )
        .with_backend("Metal")
        .with_error_code("METAL_ERROR")
        .add_recovery_suggestion("Check macOS version compatibility")
        .add_recovery_suggestion("Verify Metal Performance Shaders framework")
        .add_recovery_suggestion("Try reducing GPU memory usage");

        if let Some(device_id) = device_id {
            context = context.with_device(format!("metal:{}", device_id));
        }

        BackendError::BackendError(format!("{}: {}", context.format(), error))
    }

    /// Convert WebGPU errors to TorshError with proper context
    pub fn webgpu_error_with_context(
        error: impl std::fmt::Display,
        operation: &str,
        adapter_name: Option<&str>,
    ) -> BackendError {
        let mut context = ErrorContext::new_with_category(
            operation,
            ErrorCategory::Hardware,
            ErrorSeverity::Error,
        )
        .with_backend("WebGPU")
        .with_error_code("WEBGPU_ERROR")
        .add_recovery_suggestion("Check WebGPU browser support")
        .add_recovery_suggestion("Verify GPU driver compatibility")
        .add_recovery_suggestion("Try different adapter if available");

        if let Some(adapter_name) = adapter_name {
            context = context.with_device(adapter_name.to_string());
        }

        BackendError::BackendError(format!("{}: {}", context.format(), error))
    }

    /// Convert memory allocation errors with context
    pub fn memory_error_with_context(
        error: impl std::fmt::Display,
        size: usize,
        backend: &str,
        device: Option<&str>,
    ) -> BackendError {
        let size_mb = size as f64 / (1024.0 * 1024.0);
        let mut context = ErrorContext::new_with_category(
            "memory_allocation",
            ErrorCategory::Memory,
            if size > 1024 * 1024 * 1024 {
                ErrorSeverity::Critical
            } else {
                ErrorSeverity::Error
            },
        )
        .with_backend(backend)
        .with_details(format!("size: {} bytes ({:.2} MB)", size, size_mb))
        .with_error_code("MEMORY_ALLOCATION_FAILED")
        .add_recovery_suggestion("Free unused memory")
        .add_recovery_suggestion("Reduce batch size or model parameters")
        .add_recovery_suggestion("Enable memory optimization features");

        if size > 512 * 1024 * 1024 {
            // > 512MB
            context = context
                .add_recovery_suggestion("Consider using memory-mapped files for large datasets");
        }

        if let Some(device) = device {
            context = context.with_device(device.to_string());
        }

        BackendError::AllocationError(format!("{}: {}", context.format(), error))
    }

    /// Convert kernel execution errors with context
    pub fn kernel_error_with_context(
        error: impl std::fmt::Display,
        kernel_name: &str,
        backend: &str,
        device: Option<&str>,
    ) -> BackendError {
        let mut context = ErrorContext::new_with_category(
            format!("kernel_execution:{}", kernel_name),
            ErrorCategory::Computation,
            ErrorSeverity::Error,
        )
        .with_backend(backend)
        .with_error_code("KERNEL_EXECUTION_FAILED")
        .add_recovery_suggestion("Check kernel parameters and input dimensions")
        .add_recovery_suggestion("Verify workgroup/block sizes are valid")
        .add_recovery_suggestion("Enable debug mode for detailed error information");

        if let Some(device) = device {
            context = context.with_device(device.to_string());
        }

        BackendError::BackendError(format!("{}: {}", context.format(), error))
    }

    /// Convert timeout errors with context
    pub fn timeout_error_with_context(
        operation: &str,
        timeout_seconds: f64,
        backend: &str,
        device: Option<&str>,
    ) -> BackendError {
        let mut context = ErrorContext::new_with_category(
            operation,
            ErrorCategory::Timeout,
            ErrorSeverity::Warning,
        )
        .with_backend(backend)
        .with_details(format!("timeout: {:.2}s", timeout_seconds))
        .with_error_code("OPERATION_TIMEOUT")
        .add_recovery_suggestion("Increase timeout value")
        .add_recovery_suggestion("Optimize operation parameters")
        .add_recovery_suggestion("Split operation into smaller chunks");

        if let Some(device) = device {
            context = context.with_device(device.to_string());
        }

        BackendError::BackendError(format!("{}: Operation timed out", context.format()))
    }

    /// Convert validation errors with context
    pub fn validation_error_with_context(
        parameter_name: &str,
        expected: &str,
        actual: &str,
        operation: &str,
    ) -> BackendError {
        let context = ErrorContext::new_with_category(
            operation,
            ErrorCategory::Validation,
            ErrorSeverity::Error,
        )
        .with_details(format!(
            "parameter: {}, expected: {}, actual: {}",
            parameter_name, expected, actual
        ))
        .with_error_code("PARAMETER_VALIDATION_FAILED")
        .add_recovery_suggestion("Check parameter documentation")
        .add_recovery_suggestion("Verify input data types and shapes")
        .add_recovery_suggestion("Use parameter validation utilities");

        BackendError::BackendError(format!(
            "{}: Invalid parameter {}",
            context.format(),
            parameter_name
        ))
    }

    /// Convert resource exhaustion errors with context
    pub fn resource_exhaustion_error_with_context(
        resource_type: &str,
        current_usage: f64,
        max_limit: f64,
        backend: &str,
    ) -> BackendError {
        let usage_percent = (current_usage / max_limit) * 100.0;
        let context = ErrorContext::new_with_category(
            "resource_check",
            ErrorCategory::ResourceExhaustion,
            ErrorSeverity::Critical,
        )
        .with_backend(backend)
        .with_details(format!(
            "resource: {}, usage: {:.1}% ({:.2}/{:.2})",
            resource_type, usage_percent, current_usage, max_limit
        ))
        .with_error_code("RESOURCE_EXHAUSTED")
        .add_recovery_suggestion("Free unused resources")
        .add_recovery_suggestion("Reduce concurrent operations")
        .add_recovery_suggestion("Increase resource limits if possible");

        BackendError::BackendError(format!("{}: Resource exhausted", context.format()))
    }

    /// Convert configuration errors with context
    pub fn config_error_with_context(
        parameter: &str,
        value: &str,
        operation: &str,
        backend: &str,
    ) -> BackendError {
        let context = ErrorContext::new_with_category(
            operation,
            ErrorCategory::Configuration,
            ErrorSeverity::Error,
        )
        .with_backend(backend)
        .with_details(format!("parameter: {}, value: {}", parameter, value))
        .with_error_code("CONFIG_ERROR")
        .add_recovery_suggestion("Check configuration documentation")
        .add_recovery_suggestion("Verify parameter values and types")
        .add_recovery_suggestion("Use default configuration if available");

        BackendError::BackendError(format!("{}: Invalid configuration", context.format()))
    }

    /// Convert IO errors with context
    pub fn io_error_with_context(
        error: impl std::fmt::Display,
        file_path: Option<&str>,
        operation: &str,
    ) -> BackendError {
        let mut context =
            ErrorContext::new_with_category(operation, ErrorCategory::Io, ErrorSeverity::Error)
                .with_error_code("IO_ERROR")
                .add_recovery_suggestion("Check file permissions")
                .add_recovery_suggestion("Verify disk space availability")
                .add_recovery_suggestion("Ensure file path is accessible");

        if let Some(path) = file_path {
            context = context.with_details(format!("file: {}", path));
        }

        BackendError::BackendError(format!("{}: {}", context.format(), error))
    }

    /// Convert security errors with context
    pub fn security_error_with_context(
        error: impl std::fmt::Display,
        operation: &str,
        backend: &str,
    ) -> BackendError {
        let context = ErrorContext::new_with_category(
            operation,
            ErrorCategory::Security,
            ErrorSeverity::Critical,
        )
        .with_backend(backend)
        .with_error_code("SECURITY_ERROR")
        .add_recovery_suggestion("Check security policies and permissions")
        .add_recovery_suggestion("Verify authentication credentials")
        .add_recovery_suggestion("Review access control settings");

        BackendError::BackendError(format!("{}: {}", context.format(), error))
    }
}
