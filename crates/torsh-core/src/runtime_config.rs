//! Runtime Configuration System for ToRSh Core
//!
//! Provides centralized runtime configuration for debugging, validation,
//! and performance monitoring features. This module allows dynamic control
//! of framework behavior without recompilation.
//!
//! # Features
//!
//! - **Debug Modes**: Control assertion levels and validation strictness
//! - **Performance Monitoring**: Enable/disable metrics collection per operation type
//! - **Memory Tracking**: Configure memory debugging and leak detection
//! - **Validation Levels**: Adjust shape/dtype validation strictness
//! - **Logging Control**: Fine-grained logging configuration
//!
//! # Examples
//!
//! ```rust
//! use torsh_core::runtime_config::{RuntimeConfig, DebugLevel, ValidationLevel};
//!
//! // Enable maximum debugging for development
//! RuntimeConfig::global().set_debug_level(DebugLevel::Verbose);
//! RuntimeConfig::global().set_validation_level(ValidationLevel::Strict);
//!
//! // Production mode with minimal overhead
//! RuntimeConfig::global().set_debug_level(DebugLevel::None);
//! RuntimeConfig::global().set_validation_level(ValidationLevel::Essential);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::telemetry::LogLevel;

/// Global runtime configuration instance
static RUNTIME_CONFIG: OnceLock<Arc<Mutex<RuntimeConfigInternal>>> = OnceLock::new();

/// Debug level for assertions and validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DebugLevel {
    /// No debug checks (production mode)
    None = 0,
    /// Only critical assertions that prevent undefined behavior
    Essential = 1,
    /// Standard debug assertions
    Standard = 2,
    /// Verbose debugging with extensive checks
    Verbose = 3,
    /// Paranoid mode with maximum validation (slowest)
    Paranoid = 4,
}

impl Default for DebugLevel {
    fn default() -> Self {
        #[cfg(debug_assertions)]
        {
            Self::Standard
        }
        #[cfg(not(debug_assertions))]
        {
            Self::Essential
        }
    }
}

/// Validation level for shape and dtype operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValidationLevel {
    /// Only validate critical constraints that prevent crashes
    Essential = 0,
    /// Standard validation for common errors
    Standard = 1,
    /// Strict validation with comprehensive checks
    Strict = 2,
    /// Maximum validation including performance-impacting checks
    Maximum = 3,
}

impl Default for ValidationLevel {
    fn default() -> Self {
        #[cfg(debug_assertions)]
        {
            Self::Strict
        }
        #[cfg(not(debug_assertions))]
        {
            Self::Standard
        }
    }
}

/// Performance monitoring scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MonitoringScope {
    /// No performance monitoring
    None,
    /// Monitor only critical operations (< 1% overhead)
    Minimal,
    /// Monitor common operations (< 5% overhead)
    Standard,
    /// Monitor all operations (may add significant overhead)
    Comprehensive,
}

impl Default for MonitoringScope {
    fn default() -> Self {
        Self::Standard
    }
}

/// Memory tracking configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryTrackingConfig {
    /// Enable allocation tracking
    pub track_allocations: bool,
    /// Enable deallocation tracking
    pub track_deallocations: bool,
    /// Enable leak detection
    pub detect_leaks: bool,
    /// Enable memory pattern analysis
    pub analyze_patterns: bool,
    /// Maximum tracked allocations before sampling
    pub max_tracked_allocations: usize,
}

impl Default for MemoryTrackingConfig {
    fn default() -> Self {
        Self {
            track_allocations: cfg!(debug_assertions),
            track_deallocations: cfg!(debug_assertions),
            detect_leaks: cfg!(debug_assertions),
            analyze_patterns: false, // Expensive, opt-in
            max_tracked_allocations: 10_000,
        }
    }
}

/// Operation-specific configuration
#[derive(Debug, Clone)]
pub struct OperationConfig {
    /// Enable performance metrics for this operation
    pub enable_metrics: bool,
    /// Enable validation for this operation
    pub enable_validation: bool,
    /// Custom timeout for this operation (None = no timeout)
    pub timeout_ms: Option<u64>,
    /// Custom memory limit for this operation (None = no limit)
    pub memory_limit_bytes: Option<usize>,
}

impl Default for OperationConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_validation: true,
            timeout_ms: None,
            memory_limit_bytes: None,
        }
    }
}

/// Internal runtime configuration state
#[derive(Debug, Clone)]
struct RuntimeConfigInternal {
    /// Current debug level
    debug_level: DebugLevel,
    /// Current validation level
    validation_level: ValidationLevel,
    /// Current monitoring scope
    monitoring_scope: MonitoringScope,
    /// Memory tracking configuration
    memory_tracking: MemoryTrackingConfig,
    /// Log level for telemetry
    log_level: LogLevel,
    /// Per-operation custom configuration
    operation_configs: HashMap<String, OperationConfig>,
    /// Environment detection
    is_testing: bool,
    /// Whether to panic on warnings in debug mode
    panic_on_warnings: bool,
}

impl Default for RuntimeConfigInternal {
    fn default() -> Self {
        Self {
            debug_level: DebugLevel::default(),
            validation_level: ValidationLevel::default(),
            monitoring_scope: MonitoringScope::default(),
            memory_tracking: MemoryTrackingConfig::default(),
            log_level: LogLevel::Info,
            operation_configs: HashMap::new(),
            is_testing: std::env::var("RUST_TEST").is_ok(),
            panic_on_warnings: false,
        }
    }
}

/// Runtime configuration manager
///
/// Provides thread-safe access to global runtime configuration.
/// Configuration changes are immediately visible to all threads.
#[derive(Clone)]
pub struct RuntimeConfig {
    inner: Arc<Mutex<RuntimeConfigInternal>>,
}

impl RuntimeConfig {
    /// Get the global runtime configuration instance
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_core::runtime_config::RuntimeConfig;
    ///
    /// let config = RuntimeConfig::global();
    /// println!("Debug level: {:?}", config.debug_level());
    /// ```
    pub fn global() -> Self {
        let inner = RUNTIME_CONFIG
            .get_or_init(|| Arc::new(Mutex::new(RuntimeConfigInternal::default())))
            .clone();
        Self { inner }
    }

    /// Create a new isolated runtime configuration (for testing)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_core::runtime_config::RuntimeConfig;
    ///
    /// let config = RuntimeConfig::new();
    /// // This config is independent of the global config
    /// ```
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(RuntimeConfigInternal::default())),
        }
    }

    /// Get the current debug level
    pub fn debug_level(&self) -> DebugLevel {
        self.inner.lock().unwrap().debug_level
    }

    /// Set the debug level
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_core::runtime_config::{RuntimeConfig, DebugLevel};
    ///
    /// RuntimeConfig::global().set_debug_level(DebugLevel::Verbose);
    /// ```
    pub fn set_debug_level(&self, level: DebugLevel) {
        self.inner.lock().unwrap().debug_level = level;
    }

    /// Get the current validation level
    pub fn validation_level(&self) -> ValidationLevel {
        self.inner.lock().unwrap().validation_level
    }

    /// Set the validation level
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_core::runtime_config::{RuntimeConfig, ValidationLevel};
    ///
    /// RuntimeConfig::global().set_validation_level(ValidationLevel::Maximum);
    /// ```
    pub fn set_validation_level(&self, level: ValidationLevel) {
        self.inner.lock().unwrap().validation_level = level;
    }

    /// Get the current monitoring scope
    pub fn monitoring_scope(&self) -> MonitoringScope {
        self.inner.lock().unwrap().monitoring_scope
    }

    /// Set the monitoring scope
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_core::runtime_config::{RuntimeConfig, MonitoringScope};
    ///
    /// RuntimeConfig::global().set_monitoring_scope(MonitoringScope::Comprehensive);
    /// ```
    pub fn set_monitoring_scope(&self, scope: MonitoringScope) {
        self.inner.lock().unwrap().monitoring_scope = scope;
    }

    /// Get memory tracking configuration
    pub fn memory_tracking(&self) -> MemoryTrackingConfig {
        self.inner.lock().unwrap().memory_tracking
    }

    /// Set memory tracking configuration
    pub fn set_memory_tracking(&self, config: MemoryTrackingConfig) {
        self.inner.lock().unwrap().memory_tracking = config;
    }

    /// Get the current log level
    pub fn log_level(&self) -> LogLevel {
        self.inner.lock().unwrap().log_level
    }

    /// Set the log level
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_core::runtime_config::RuntimeConfig;
    /// use torsh_core::telemetry::LogLevel;
    ///
    /// RuntimeConfig::global().set_log_level(LogLevel::Debug);
    /// ```
    pub fn set_log_level(&self, level: LogLevel) {
        self.inner.lock().unwrap().log_level = level;
    }

    /// Check if currently running in test mode
    pub fn is_testing(&self) -> bool {
        self.inner.lock().unwrap().is_testing
    }

    /// Set testing mode
    pub fn set_testing(&self, testing: bool) {
        self.inner.lock().unwrap().is_testing = testing;
    }

    /// Check if warnings should panic in debug mode
    pub fn panic_on_warnings(&self) -> bool {
        self.inner.lock().unwrap().panic_on_warnings
    }

    /// Set whether to panic on warnings in debug mode
    pub fn set_panic_on_warnings(&self, panic: bool) {
        self.inner.lock().unwrap().panic_on_warnings = panic;
    }

    /// Get configuration for a specific operation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_core::runtime_config::RuntimeConfig;
    ///
    /// let config = RuntimeConfig::global();
    /// if let Some(op_config) = config.get_operation_config("matmul") {
    ///     println!("Matmul metrics enabled: {}", op_config.enable_metrics);
    /// }
    /// ```
    pub fn get_operation_config(&self, operation: &str) -> Option<OperationConfig> {
        self.inner
            .lock()
            .unwrap()
            .operation_configs
            .get(operation)
            .cloned()
    }

    /// Set configuration for a specific operation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_core::runtime_config::{RuntimeConfig, OperationConfig};
    ///
    /// let mut config = OperationConfig::default();
    /// config.timeout_ms = Some(1000); // 1 second timeout
    ///
    /// RuntimeConfig::global().set_operation_config("slow_operation", config);
    /// ```
    pub fn set_operation_config(&self, operation: impl Into<String>, config: OperationConfig) {
        self.inner
            .lock()
            .unwrap()
            .operation_configs
            .insert(operation.into(), config);
    }

    /// Remove configuration for a specific operation
    pub fn remove_operation_config(&self, operation: &str) -> Option<OperationConfig> {
        self.inner
            .lock()
            .unwrap()
            .operation_configs
            .remove(operation)
    }

    /// Clear all operation-specific configurations
    pub fn clear_operation_configs(&self) {
        self.inner.lock().unwrap().operation_configs.clear();
    }

    /// Check if an operation should collect metrics
    pub fn should_collect_metrics(&self, operation: &str) -> bool {
        let guard = self.inner.lock().unwrap();

        // Check operation-specific config first
        if let Some(op_config) = guard.operation_configs.get(operation) {
            return op_config.enable_metrics;
        }

        // Fall back to monitoring scope
        match guard.monitoring_scope {
            MonitoringScope::None => false,
            MonitoringScope::Minimal => {
                // Only critical operations
                matches!(operation, "matmul" | "conv2d" | "backward")
            }
            MonitoringScope::Standard => {
                // Common operations
                !matches!(operation, "add" | "mul" | "sub" | "div")
            }
            MonitoringScope::Comprehensive => true,
        }
    }

    /// Check if an operation should perform validation
    pub fn should_validate(&self, operation: &str) -> bool {
        let guard = self.inner.lock().unwrap();

        // Check operation-specific config first
        if let Some(op_config) = guard.operation_configs.get(operation) {
            return op_config.enable_validation;
        }

        // Fall back to validation level
        guard.validation_level >= ValidationLevel::Standard
    }

    /// Check if essential validation should be performed
    pub fn should_validate_essential(&self) -> bool {
        let guard = self.inner.lock().unwrap();
        guard.validation_level >= ValidationLevel::Essential
    }

    /// Check if standard validation should be performed
    pub fn should_validate_standard(&self) -> bool {
        let guard = self.inner.lock().unwrap();
        guard.validation_level >= ValidationLevel::Standard
    }

    /// Check if strict validation should be performed
    pub fn should_validate_strict(&self) -> bool {
        let guard = self.inner.lock().unwrap();
        guard.validation_level >= ValidationLevel::Strict
    }

    /// Check if maximum validation should be performed
    pub fn should_validate_maximum(&self) -> bool {
        let guard = self.inner.lock().unwrap();
        guard.validation_level >= ValidationLevel::Maximum
    }

    /// Apply a preset configuration for specific environments
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_core::runtime_config::{RuntimeConfig, ConfigPreset};
    ///
    /// // Development mode
    /// RuntimeConfig::global().apply_preset(ConfigPreset::Development);
    ///
    /// // Production mode
    /// RuntimeConfig::global().apply_preset(ConfigPreset::Production);
    /// ```
    pub fn apply_preset(&self, preset: ConfigPreset) {
        let mut guard = self.inner.lock().unwrap();
        match preset {
            ConfigPreset::Development => {
                guard.debug_level = DebugLevel::Verbose;
                guard.validation_level = ValidationLevel::Maximum;
                guard.monitoring_scope = MonitoringScope::Comprehensive;
                guard.memory_tracking = MemoryTrackingConfig {
                    track_allocations: true,
                    track_deallocations: true,
                    detect_leaks: true,
                    analyze_patterns: true,
                    max_tracked_allocations: 50_000,
                };
                guard.log_level = LogLevel::Debug;
                guard.panic_on_warnings = false;
            }
            ConfigPreset::Testing => {
                guard.debug_level = DebugLevel::Standard;
                guard.validation_level = ValidationLevel::Strict;
                guard.monitoring_scope = MonitoringScope::Standard;
                guard.memory_tracking = MemoryTrackingConfig {
                    track_allocations: true,
                    track_deallocations: true,
                    detect_leaks: true,
                    analyze_patterns: false,
                    max_tracked_allocations: 10_000,
                };
                guard.log_level = LogLevel::Info;
                guard.panic_on_warnings = true;
                guard.is_testing = true;
            }
            ConfigPreset::Production => {
                guard.debug_level = DebugLevel::None;
                guard.validation_level = ValidationLevel::Essential;
                guard.monitoring_scope = MonitoringScope::Minimal;
                guard.memory_tracking = MemoryTrackingConfig {
                    track_allocations: false,
                    track_deallocations: false,
                    detect_leaks: false,
                    analyze_patterns: false,
                    max_tracked_allocations: 0,
                };
                guard.log_level = LogLevel::Warn;
                guard.panic_on_warnings = false;
            }
            ConfigPreset::Profiling => {
                guard.debug_level = DebugLevel::Essential;
                guard.validation_level = ValidationLevel::Standard;
                guard.monitoring_scope = MonitoringScope::Comprehensive;
                guard.memory_tracking = MemoryTrackingConfig {
                    track_allocations: true,
                    track_deallocations: true,
                    detect_leaks: false,
                    analyze_patterns: true,
                    max_tracked_allocations: 100_000,
                };
                guard.log_level = LogLevel::Info;
                guard.panic_on_warnings = false;
            }
        }
    }

    /// Reset to default configuration
    pub fn reset(&self) {
        *self.inner.lock().unwrap() = RuntimeConfigInternal::default();
    }

    /// Get a snapshot of the current configuration (for debugging)
    pub fn snapshot(&self) -> RuntimeConfigSnapshot {
        let guard = self.inner.lock().unwrap();
        RuntimeConfigSnapshot {
            debug_level: guard.debug_level,
            validation_level: guard.validation_level,
            monitoring_scope: guard.monitoring_scope,
            memory_tracking: guard.memory_tracking,
            log_level: guard.log_level,
            is_testing: guard.is_testing,
            panic_on_warnings: guard.panic_on_warnings,
            operation_count: guard.operation_configs.len(),
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::global()
    }
}

/// Configuration preset for common environments
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigPreset {
    /// Development mode: maximum debugging, comprehensive validation
    Development,
    /// Testing mode: standard debugging, strict validation, panics on warnings
    Testing,
    /// Production mode: minimal overhead, essential checks only
    Production,
    /// Profiling mode: comprehensive metrics, minimal validation overhead
    Profiling,
}

/// Snapshot of runtime configuration (for debugging/reporting)
#[derive(Debug, Clone)]
pub struct RuntimeConfigSnapshot {
    pub debug_level: DebugLevel,
    pub validation_level: ValidationLevel,
    pub monitoring_scope: MonitoringScope,
    pub memory_tracking: MemoryTrackingConfig,
    pub log_level: LogLevel,
    pub is_testing: bool,
    pub panic_on_warnings: bool,
    pub operation_count: usize,
}

/// Convenience macros for debug assertions with configurable levels
#[macro_export]
macro_rules! torsh_debug_assert {
    ($cond:expr) => {
        if $crate::runtime_config::RuntimeConfig::global().debug_level() >= $crate::runtime_config::DebugLevel::Standard {
            assert!($cond);
        }
    };
    ($cond:expr, $($arg:tt)+) => {
        if $crate::runtime_config::RuntimeConfig::global().debug_level() >= $crate::runtime_config::DebugLevel::Standard {
            assert!($cond, $($arg)+);
        }
    };
}

/// Verbose debug assertion (only in Verbose or Paranoid mode)
#[macro_export]
macro_rules! torsh_debug_assert_verbose {
    ($cond:expr) => {
        if $crate::runtime_config::RuntimeConfig::global().debug_level() >= $crate::runtime_config::DebugLevel::Verbose {
            assert!($cond);
        }
    };
    ($cond:expr, $($arg:tt)+) => {
        if $crate::runtime_config::RuntimeConfig::global().debug_level() >= $crate::runtime_config::DebugLevel::Verbose {
            assert!($cond, $($arg)+);
        }
    };
}

/// Essential assertion (always runs unless debug level is None)
#[macro_export]
macro_rules! torsh_assert_essential {
    ($cond:expr) => {
        if $crate::runtime_config::RuntimeConfig::global().debug_level() >= $crate::runtime_config::DebugLevel::Essential {
            assert!($cond);
        }
    };
    ($cond:expr, $($arg:tt)+) => {
        if $crate::runtime_config::RuntimeConfig::global().debug_level() >= $crate::runtime_config::DebugLevel::Essential {
            assert!($cond, $($arg)+);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_creation() {
        let config = RuntimeConfig::new();
        assert_eq!(config.debug_level(), DebugLevel::default());
        assert_eq!(config.validation_level(), ValidationLevel::default());
    }

    #[test]
    fn test_debug_level_setting() {
        let config = RuntimeConfig::new();
        config.set_debug_level(DebugLevel::Paranoid);
        assert_eq!(config.debug_level(), DebugLevel::Paranoid);
    }

    #[test]
    fn test_validation_level_setting() {
        let config = RuntimeConfig::new();
        config.set_validation_level(ValidationLevel::Maximum);
        assert_eq!(config.validation_level(), ValidationLevel::Maximum);
    }

    #[test]
    fn test_monitoring_scope_setting() {
        let config = RuntimeConfig::new();
        config.set_monitoring_scope(MonitoringScope::Comprehensive);
        assert_eq!(config.monitoring_scope(), MonitoringScope::Comprehensive);
    }

    #[test]
    fn test_operation_config() {
        let config = RuntimeConfig::new();

        let op_config = OperationConfig {
            enable_metrics: false,
            enable_validation: true,
            timeout_ms: Some(5000),
            memory_limit_bytes: Some(1024 * 1024), // 1MB
        };

        config.set_operation_config("test_op", op_config.clone());

        let retrieved = config.get_operation_config("test_op").unwrap();
        assert!(!retrieved.enable_metrics);
        assert!(retrieved.enable_validation);
        assert_eq!(retrieved.timeout_ms, Some(5000));
        assert_eq!(retrieved.memory_limit_bytes, Some(1024 * 1024));
    }

    #[test]
    fn test_should_collect_metrics() {
        let config = RuntimeConfig::new();

        // Test monitoring scope
        config.set_monitoring_scope(MonitoringScope::None);
        assert!(!config.should_collect_metrics("matmul"));

        config.set_monitoring_scope(MonitoringScope::Minimal);
        assert!(config.should_collect_metrics("matmul"));
        assert!(!config.should_collect_metrics("add"));

        config.set_monitoring_scope(MonitoringScope::Comprehensive);
        assert!(config.should_collect_metrics("add"));
    }

    #[test]
    fn test_should_validate() {
        let config = RuntimeConfig::new();

        config.set_validation_level(ValidationLevel::Essential);
        assert!(!config.should_validate("any_operation"));

        config.set_validation_level(ValidationLevel::Standard);
        assert!(config.should_validate("any_operation"));
    }

    #[test]
    fn test_development_preset() {
        let config = RuntimeConfig::new();
        config.apply_preset(ConfigPreset::Development);

        assert_eq!(config.debug_level(), DebugLevel::Verbose);
        assert_eq!(config.validation_level(), ValidationLevel::Maximum);
        assert_eq!(config.monitoring_scope(), MonitoringScope::Comprehensive);
        assert!(config.memory_tracking().track_allocations);
    }

    #[test]
    fn test_production_preset() {
        let config = RuntimeConfig::new();
        config.apply_preset(ConfigPreset::Production);

        assert_eq!(config.debug_level(), DebugLevel::None);
        assert_eq!(config.validation_level(), ValidationLevel::Essential);
        assert_eq!(config.monitoring_scope(), MonitoringScope::Minimal);
        assert!(!config.memory_tracking().track_allocations);
    }

    #[test]
    fn test_testing_preset() {
        let config = RuntimeConfig::new();
        config.apply_preset(ConfigPreset::Testing);

        assert_eq!(config.debug_level(), DebugLevel::Standard);
        assert_eq!(config.validation_level(), ValidationLevel::Strict);
        assert!(config.panic_on_warnings());
        assert!(config.is_testing());
    }

    #[test]
    fn test_snapshot() {
        let config = RuntimeConfig::new();
        config.set_debug_level(DebugLevel::Verbose);

        let snapshot = config.snapshot();
        assert_eq!(snapshot.debug_level, DebugLevel::Verbose);
    }

    #[test]
    fn test_reset() {
        let config = RuntimeConfig::new();
        config.set_debug_level(DebugLevel::Paranoid);
        config.reset();
        assert_eq!(config.debug_level(), DebugLevel::default());
    }

    #[test]
    fn test_clear_operation_configs() {
        let config = RuntimeConfig::new();
        config.set_operation_config("op1", OperationConfig::default());
        config.set_operation_config("op2", OperationConfig::default());

        config.clear_operation_configs();
        assert!(config.get_operation_config("op1").is_none());
        assert!(config.get_operation_config("op2").is_none());
    }
}
