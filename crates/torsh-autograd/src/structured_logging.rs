//! Structured logging for autograd operations
//!
//! This module provides comprehensive structured logging capabilities for automatic
//! differentiation operations, including performance metrics, operation traces,
//! gradient statistics, and debugging information.

use parking_lot::RwLock;
use scirs2_core::random::{thread_rng, Rng}; // SciRS2 POLICY compliant
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use torsh_core::dtype::TensorElement;
use torsh_core::error::Result;

/// Log levels for autograd operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Categories of autograd operations for logging
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationCategory {
    Forward,
    Backward,
    GradientComputation,
    TensorCreation,
    MemoryManagement,
    ComputationGraph,
    Optimization,
    Validation,
    Communication,
    Checkpointing,
}

/// Structured log entry for autograd operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutogradLogEntry {
    /// Timestamp when the log entry was created
    pub timestamp: u64,
    /// Unique identifier for this log entry
    pub id: u64,
    /// Log level
    pub level: LogLevel,
    /// Operation category
    pub category: OperationCategory,
    /// Human-readable message
    pub message: String,
    /// Operation name
    pub operation: String,
    /// Duration of the operation (if completed)
    pub duration_ms: Option<u64>,
    /// Tensor information
    pub tensors: Vec<TensorInfo>,
    /// Memory usage information
    pub memory_info: Option<MemoryInfo>,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Thread ID
    pub thread_id: String,
    /// Process ID
    pub process_id: u32,
    /// Error information (if applicable)
    pub error: Option<ErrorInfo>,
}

/// Information about tensors involved in an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Tensor identifier
    pub id: Option<usize>,
    /// Tensor name (if available)
    pub name: Option<String>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
    /// Device location
    pub device: String,
    /// Whether gradients are required
    pub requires_grad: bool,
    /// Memory size in bytes
    pub memory_size: usize,
    /// Gradient statistics (if available)
    pub grad_stats: Option<GradientStats>,
}

/// Statistics about gradients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStats {
    /// Mean gradient value
    pub mean: f64,
    /// Standard deviation of gradients
    pub std: f64,
    /// Minimum gradient value
    pub min: f64,
    /// Maximum gradient value
    pub max: f64,
    /// L2 norm of gradients
    pub l2_norm: f64,
    /// Percentage of zero gradients
    pub sparsity: f64,
    /// Number of NaN values
    pub nan_count: usize,
    /// Number of infinite values
    pub inf_count: usize,
}

/// Memory usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total memory allocated in bytes
    pub total_allocated: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Memory fragmentation percentage
    pub fragmentation: f64,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Memory pool statistics
    pub pool_stats: HashMap<String, usize>,
}

/// Error information for logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
    /// Recovery attempts made
    pub recovery_attempts: usize,
    /// Whether the error was recoverable
    pub recoverable: bool,
}

/// Configuration for structured logging
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Minimum log level to record
    pub min_level: LogLevel,
    /// Categories to log
    pub enabled_categories: Vec<OperationCategory>,
    /// Whether to include tensor data in logs
    pub include_tensor_data: bool,
    /// Whether to include memory information
    pub include_memory_info: bool,
    /// Whether to include performance metrics
    pub include_metrics: bool,
    /// Maximum number of log entries to keep in memory
    pub max_entries: usize,
    /// Whether to write logs to file
    pub file_output: Option<String>,
    /// Whether to enable real-time monitoring
    pub real_time_monitoring: bool,
    /// Sampling rate for high-frequency operations (0.0 to 1.0)
    pub sampling_rate: f64,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            min_level: LogLevel::Info,
            enabled_categories: vec![
                OperationCategory::Forward,
                OperationCategory::Backward,
                OperationCategory::GradientComputation,
            ],
            include_tensor_data: true,
            include_memory_info: true,
            include_metrics: true,
            max_entries: 10000,
            file_output: None,
            real_time_monitoring: false,
            sampling_rate: 1.0,
        }
    }
}

/// Structured logger for autograd operations
pub struct AutogradLogger {
    config: LoggingConfig,
    entries: Arc<RwLock<Vec<AutogradLogEntry>>>,
    entry_counter: AtomicU64,
    active_operations: Arc<RwLock<HashMap<String, OperationTracker>>>,
    statistics: Arc<RwLock<LoggingStatistics>>,
}

/// Tracks active operations for timing
#[derive(Debug, Clone)]
struct OperationTracker {
    start_time: Instant,
    category: OperationCategory,
    #[allow(dead_code)]
    metadata: HashMap<String, String>,
}

/// Statistics about logging activity
#[derive(Debug, Clone, Default)]
pub struct LoggingStatistics {
    /// Total number of log entries created
    pub total_entries: usize,
    /// Entries by category
    pub entries_by_category: HashMap<OperationCategory, usize>,
    /// Entries by level
    pub entries_by_level: HashMap<LogLevel, usize>,
    /// Average operation duration by category
    pub avg_duration_by_category: HashMap<OperationCategory, f64>,
    /// Total logging overhead time
    pub logging_overhead_ms: u64,
}

impl AutogradLogger {
    /// Create a new autograd logger with default configuration
    pub fn new() -> Self {
        Self::with_config(LoggingConfig::default())
    }

    /// Create a new autograd logger with custom configuration
    pub fn with_config(config: LoggingConfig) -> Self {
        Self {
            config,
            entries: Arc::new(RwLock::new(Vec::new())),
            entry_counter: AtomicU64::new(0),
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(LoggingStatistics::default())),
        }
    }

    /// Start logging an operation
    pub fn start_operation(
        &self,
        operation: impl Into<String>,
        category: OperationCategory,
    ) -> String {
        let operation_name = operation.into();
        let operation_id = format!(
            "{}_{}",
            operation_name,
            self.entry_counter.fetch_add(1, Ordering::SeqCst)
        );

        let tracker = OperationTracker {
            start_time: Instant::now(),
            category,
            metadata: HashMap::new(),
        };

        self.active_operations
            .write()
            .insert(operation_id.clone(), tracker);

        // Log operation start
        self.log_entry(
            LogLevel::Debug,
            category,
            format!("Started operation: {}", operation_name),
            operation_name,
            None,
            Vec::new(),
            HashMap::new(),
            HashMap::new(),
        );

        operation_id
    }

    /// End logging an operation
    pub fn end_operation(
        &self,
        operation_id: &str,
        success: bool,
        error: Option<ErrorInfo>,
    ) -> Option<Duration> {
        let tracker = self.active_operations.write().remove(operation_id);

        if let Some(tracker) = tracker {
            let duration = tracker.start_time.elapsed();
            let level = if success {
                LogLevel::Debug
            } else {
                LogLevel::Error
            };
            let message = if success {
                format!("Completed operation: {} in {:?}", operation_id, duration)
            } else {
                format!("Failed operation: {} after {:?}", operation_id, duration)
            };

            // Update statistics
            {
                let mut stats = self.statistics.write();
                let durations = stats
                    .avg_duration_by_category
                    .entry(tracker.category)
                    .or_insert(0.0);
                *durations = (*durations + duration.as_millis() as f64) / 2.0; // Running average
            }

            // Log operation completion
            self.log_entry(
                level,
                tracker.category,
                message,
                operation_id.to_string(),
                Some(duration.as_millis() as u64),
                Vec::new(),
                HashMap::new(),
                HashMap::new(),
            );

            if let Some(err) = error {
                self.log_error(tracker.category, operation_id, err);
            }

            Some(duration)
        } else {
            tracing::warn!("Operation {} not found in active operations", operation_id);
            None
        }
    }

    /// Log a tensor operation
    pub fn log_tensor_operation<T: TensorElement>(
        &self,
        level: LogLevel,
        category: OperationCategory,
        operation: impl Into<String>,
        message: impl Into<String>,
        tensors: Vec<TensorInfo>,
        metrics: HashMap<String, f64>,
    ) {
        // Check if we should sample this operation
        if !self.should_log_operation(level, category) {
            return;
        }

        let operation_name = operation.into();
        let log_message = message.into();

        self.log_entry(
            level,
            category,
            log_message,
            operation_name,
            None,
            tensors,
            metrics,
            HashMap::new(),
        );
    }

    /// Log gradient statistics
    pub fn log_gradient_stats(
        &self,
        operation: impl Into<String>,
        tensor_name: impl Into<String>,
        stats: GradientStats,
    ) {
        let metrics = {
            let mut m = HashMap::new();
            m.insert("grad_mean".to_string(), stats.mean);
            m.insert("grad_std".to_string(), stats.std);
            m.insert("grad_l2_norm".to_string(), stats.l2_norm);
            m.insert("grad_sparsity".to_string(), stats.sparsity);
            m.insert("grad_nan_count".to_string(), stats.nan_count as f64);
            m.insert("grad_inf_count".to_string(), stats.inf_count as f64);
            m
        };

        let metadata = {
            let mut m = HashMap::new();
            m.insert("tensor_name".to_string(), tensor_name.into());
            m
        };

        self.log_entry(
            LogLevel::Info,
            OperationCategory::GradientComputation,
            format!("Gradient statistics for tensor"),
            operation.into(),
            None,
            Vec::new(),
            metrics,
            metadata,
        );
    }

    /// Log memory usage
    pub fn log_memory_usage(&self, operation: impl Into<String>, memory_info: MemoryInfo) {
        let metrics = {
            let mut m = HashMap::new();
            m.insert(
                "memory_total".to_string(),
                memory_info.total_allocated as f64,
            );
            m.insert("memory_peak".to_string(), memory_info.peak_usage as f64);
            m.insert(
                "memory_current".to_string(),
                memory_info.current_usage as f64,
            );
            m.insert(
                "memory_fragmentation".to_string(),
                memory_info.fragmentation,
            );
            m.insert(
                "active_allocations".to_string(),
                memory_info.active_allocations as f64,
            );
            m
        };

        self.log_entry(
            LogLevel::Info,
            OperationCategory::MemoryManagement,
            format!(
                "Memory usage: {} bytes allocated",
                memory_info.total_allocated
            ),
            operation.into(),
            None,
            Vec::new(),
            metrics,
            HashMap::new(),
        );
    }

    /// Log an error
    pub fn log_error(
        &self,
        category: OperationCategory,
        operation: impl Into<String>,
        error: ErrorInfo,
    ) {
        let metadata = {
            let mut m = HashMap::new();
            m.insert("error_type".to_string(), error.error_type.clone());
            m.insert("recoverable".to_string(), error.recoverable.to_string());
            m.insert(
                "recovery_attempts".to_string(),
                error.recovery_attempts.to_string(),
            );
            m
        };

        self.log_entry(
            LogLevel::Error,
            category,
            format!("Error: {}", error.message),
            operation.into(),
            None,
            Vec::new(),
            HashMap::new(),
            metadata,
        );
    }

    /// Check if an operation should be logged based on sampling rate
    fn should_log_operation(&self, level: LogLevel, category: OperationCategory) -> bool {
        // Check minimum level
        if !self.level_enabled(level) {
            return false;
        }

        // Check enabled categories
        if !self.config.enabled_categories.contains(&category) {
            return false;
        }

        // Apply sampling rate - SciRS2 POLICY compliant
        if self.config.sampling_rate < 1.0 {
            thread_rng().random::<f64>() < self.config.sampling_rate
        } else {
            true
        }
    }

    /// Check if a log level is enabled
    fn level_enabled(&self, level: LogLevel) -> bool {
        match (level, self.config.min_level) {
            (LogLevel::Error, _) => true,
            (LogLevel::Warn, LogLevel::Error) => false,
            (LogLevel::Warn, _) => true,
            (LogLevel::Info, LogLevel::Error | LogLevel::Warn) => false,
            (LogLevel::Info, _) => true,
            (LogLevel::Debug, LogLevel::Error | LogLevel::Warn | LogLevel::Info) => false,
            (LogLevel::Debug, _) => true,
            (LogLevel::Trace, LogLevel::Trace) => true,
            (LogLevel::Trace, _) => false,
        }
    }

    /// Internal method to create and store a log entry
    fn log_entry(
        &self,
        level: LogLevel,
        category: OperationCategory,
        message: String,
        operation: String,
        duration_ms: Option<u64>,
        tensors: Vec<TensorInfo>,
        metrics: HashMap<String, f64>,
        metadata: HashMap<String, String>,
    ) {
        let entry = AutogradLogEntry {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            id: self.entry_counter.fetch_add(1, Ordering::SeqCst),
            level,
            category,
            message: message.clone(),
            operation,
            duration_ms,
            tensors,
            memory_info: None, // Could be populated if available
            metrics,
            metadata,
            thread_id: format!("{:?}", std::thread::current().id()),
            process_id: std::process::id(),
            error: None,
        };

        // Store entry
        {
            let mut entries = self.entries.write();
            entries.push(entry);

            // Trim if exceeding max entries
            if entries.len() > self.config.max_entries {
                entries.remove(0);
            }
        }

        // Update statistics
        {
            let mut stats = self.statistics.write();
            stats.total_entries += 1;
            *stats.entries_by_category.entry(category).or_insert(0) += 1;
            *stats.entries_by_level.entry(level).or_insert(0) += 1;
        }

        // Write to tracing framework
        match level {
            LogLevel::Error => tracing::error!(category = ?category, "{}", message),
            LogLevel::Warn => tracing::warn!(category = ?category, "{}", message),
            LogLevel::Info => tracing::info!(category = ?category, "{}", message),
            LogLevel::Debug => tracing::debug!(category = ?category, "{}", message),
            LogLevel::Trace => tracing::trace!(category = ?category, "{}", message),
        }
    }

    /// Get all log entries
    pub fn get_entries(&self) -> Vec<AutogradLogEntry> {
        self.entries.read().clone()
    }

    /// Get entries filtered by criteria
    pub fn get_filtered_entries(
        &self,
        min_level: Option<LogLevel>,
        category: Option<OperationCategory>,
        operation: Option<&str>,
        since: Option<u64>,
    ) -> Vec<AutogradLogEntry> {
        let entries = self.entries.read();

        entries
            .iter()
            .filter(|entry| {
                if let Some(level) = min_level {
                    if !self.level_passes_filter(entry.level, level) {
                        return false;
                    }
                }

                if let Some(cat) = category {
                    if entry.category != cat {
                        return false;
                    }
                }

                if let Some(op) = operation {
                    if entry.operation != op {
                        return false;
                    }
                }

                if let Some(timestamp) = since {
                    if entry.timestamp < timestamp {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect()
    }

    /// Check if a log level passes a minimum level filter
    fn level_passes_filter(&self, entry_level: LogLevel, min_level: LogLevel) -> bool {
        match (entry_level, min_level) {
            (LogLevel::Error, _) => true,
            (LogLevel::Warn, LogLevel::Error) => false,
            (LogLevel::Warn, _) => true,
            (LogLevel::Info, LogLevel::Error | LogLevel::Warn) => false,
            (LogLevel::Info, _) => true,
            (LogLevel::Debug, LogLevel::Error | LogLevel::Warn | LogLevel::Info) => false,
            (LogLevel::Debug, _) => true,
            (LogLevel::Trace, LogLevel::Trace) => true,
            (LogLevel::Trace, _) => false,
        }
    }

    /// Get logging statistics
    pub fn get_statistics(&self) -> LoggingStatistics {
        self.statistics.read().clone()
    }

    /// Clear all log entries
    pub fn clear_entries(&self) {
        self.entries.write().clear();
    }

    /// Export logs to JSON format
    pub fn export_to_json(&self) -> Result<String> {
        let entries = self.get_entries();
        serde_json::to_string_pretty(&entries).map_err(|e| {
            torsh_core::error::TorshError::AutogradError(format!(
                "Failed to serialize logs to JSON: {}",
                e
            ))
        })
    }

    /// Export logs to CSV format
    pub fn export_to_csv(&self) -> Result<String> {
        let entries = self.get_entries();
        let mut csv = String::new();

        // Header
        csv.push_str(
            "timestamp,id,level,category,operation,duration_ms,message,thread_id,process_id\n",
        );

        // Data rows
        for entry in entries {
            csv.push_str(&format!(
                "{},{},{:?},{:?},{},{},{},{},{}\n",
                entry.timestamp,
                entry.id,
                entry.level,
                entry.category,
                entry.operation,
                entry.duration_ms.unwrap_or(0),
                entry.message.replace(',', ";"), // Escape commas
                entry.thread_id,
                entry.process_id
            ));
        }

        Ok(csv)
    }
}

impl Default for AutogradLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Global autograd logger instance
static GLOBAL_LOGGER: once_cell::sync::Lazy<Arc<RwLock<Option<Arc<AutogradLogger>>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(None)));

/// Initialize the global autograd logger
pub fn init_logger(config: LoggingConfig) {
    let logger = Arc::new(AutogradLogger::with_config(config));
    *GLOBAL_LOGGER.write() = Some(logger);
}

/// Get the global autograd logger
pub fn get_logger() -> Option<Arc<AutogradLogger>> {
    GLOBAL_LOGGER.read().clone()
}

/// Convenience macros for logging
#[macro_export]
macro_rules! log_autograd {
    ($level:expr, $category:expr, $operation:expr, $message:expr) => {
        if let Some(logger) = $crate::structured_logging::get_logger() {
            logger.log_tensor_operation(
                $level,
                $category,
                $operation,
                $message,
                Vec::new(),
                std::collections::HashMap::new(),
            );
        }
    };

    ($level:expr, $category:expr, $operation:expr, $message:expr, $tensors:expr) => {
        if let Some(logger) = $crate::structured_logging::get_logger() {
            logger.log_tensor_operation(
                $level,
                $category,
                $operation,
                $message,
                $tensors,
                std::collections::HashMap::new(),
            );
        }
    };

    ($level:expr, $category:expr, $operation:expr, $message:expr, $tensors:expr, $metrics:expr) => {
        if let Some(logger) = $crate::structured_logging::get_logger() {
            logger
                .log_tensor_operation($level, $category, $operation, $message, $tensors, $metrics);
        }
    };
}

/// Helper function to compute gradient statistics
pub fn compute_gradient_stats<T: TensorElement + num_traits::Float + num_traits::ToPrimitive>(
    gradients: &[T],
) -> GradientStats {
    if gradients.is_empty() {
        return GradientStats {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            l2_norm: 0.0,
            sparsity: 0.0,
            nan_count: 0,
            inf_count: 0,
        };
    }

    let mut sum = 0.0;
    let mut sum_squared = 0.0;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let mut zero_count = 0;
    let mut nan_count = 0;
    let mut inf_count = 0;

    for &grad in gradients {
        let val = <T as torsh_core::TensorElement>::to_f64(&grad).unwrap_or(0.0);

        if val.is_nan() {
            nan_count += 1;
            continue;
        }

        if val.is_infinite() {
            inf_count += 1;
            continue;
        }

        sum += val;
        sum_squared += val * val;
        min_val = min_val.min(val);
        max_val = max_val.max(val);

        if val.abs() < 1e-10 {
            zero_count += 1;
        }
    }

    let valid_count = gradients.len() - nan_count - inf_count;
    let mean = if valid_count > 0 {
        sum / valid_count as f64
    } else {
        0.0
    };
    let variance = if valid_count > 1 {
        (sum_squared / valid_count as f64) - (mean * mean)
    } else {
        0.0
    };
    let std = variance.sqrt();
    let l2_norm = sum_squared.sqrt();
    let sparsity = zero_count as f64 / gradients.len() as f64;

    GradientStats {
        mean,
        std,
        min: if min_val == f64::INFINITY {
            0.0
        } else {
            min_val
        },
        max: if max_val == f64::NEG_INFINITY {
            0.0
        } else {
            max_val
        },
        l2_norm,
        sparsity,
        nan_count,
        inf_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logger_creation() {
        let logger = AutogradLogger::new();
        assert_eq!(logger.get_entries().len(), 0);
    }

    #[test]
    fn test_operation_tracking() {
        let logger = AutogradLogger::new();

        let op_id = logger.start_operation("test_op", OperationCategory::Forward);
        assert!(logger.active_operations.read().contains_key(&op_id));

        std::thread::sleep(Duration::from_millis(10));

        let duration = logger.end_operation(&op_id, true, None);
        assert!(duration.is_some());
        assert!(!logger.active_operations.read().contains_key(&op_id));
    }

    #[test]
    fn test_tensor_logging() {
        let logger = AutogradLogger::new();

        let tensor_info = TensorInfo {
            id: Some(1),
            name: Some("test_tensor".to_string()),
            shape: vec![2, 3],
            dtype: "f32".to_string(),
            device: "cpu".to_string(),
            requires_grad: true,
            memory_size: 24,
            grad_stats: None,
        };

        logger.log_tensor_operation::<f32>(
            LogLevel::Info,
            OperationCategory::Forward,
            "test_operation",
            "Test message",
            vec![tensor_info],
            HashMap::new(),
        );

        let entries = logger.get_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].tensors.len(), 1);
        assert_eq!(entries[0].tensors[0].shape, vec![2, 3]);
    }

    #[test]
    fn test_gradient_stats_computation() {
        let gradients = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_gradient_stats(&gradients);

        assert_eq!(stats.mean, 3.0);
        assert!((stats.std - (2.0f64).sqrt()).abs() < 1e-6);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
    }

    #[test]
    fn test_gradient_stats_with_nan_inf() {
        let gradients = vec![1.0f32, f32::NAN, f32::INFINITY, 2.0, f32::NEG_INFINITY];
        let stats = compute_gradient_stats(&gradients);

        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.inf_count, 2);
        assert_eq!(stats.mean, 1.5); // Mean of valid values: (1.0 + 2.0) / 2
    }

    #[test]
    fn test_log_filtering() {
        let logger = AutogradLogger::new();

        // Log entries with different levels and categories
        logger.log_tensor_operation::<f32>(
            LogLevel::Info,
            OperationCategory::Forward,
            "forward_op",
            "Forward message",
            Vec::new(),
            HashMap::new(),
        );

        logger.log_tensor_operation::<f32>(
            LogLevel::Error,
            OperationCategory::Backward,
            "backward_op",
            "Backward message",
            Vec::new(),
            HashMap::new(),
        );

        // Test filtering by level
        let error_entries = logger.get_filtered_entries(Some(LogLevel::Error), None, None, None);
        assert_eq!(error_entries.len(), 1);
        assert_eq!(error_entries[0].level, LogLevel::Error);

        // Test filtering by category
        let forward_entries =
            logger.get_filtered_entries(None, Some(OperationCategory::Forward), None, None);
        assert_eq!(forward_entries.len(), 1);
        assert_eq!(forward_entries[0].category, OperationCategory::Forward);
    }

    #[test]
    fn test_export_to_json() {
        let logger = AutogradLogger::new();

        logger.log_tensor_operation::<f32>(
            LogLevel::Info,
            OperationCategory::Forward,
            "test_op",
            "Test message",
            Vec::new(),
            HashMap::new(),
        );

        let json = logger.export_to_json().unwrap();
        assert!(json.contains("test_op"));
        assert!(json.contains("Test message"));
    }

    #[test]
    fn test_export_to_csv() {
        let logger = AutogradLogger::new();

        logger.log_tensor_operation::<f32>(
            LogLevel::Info,
            OperationCategory::Forward,
            "test_op",
            "Test message",
            Vec::new(),
            HashMap::new(),
        );

        let csv = logger.export_to_csv().unwrap();
        assert!(csv.contains("timestamp,id,level"));
        assert!(csv.contains("test_op"));
        assert!(csv.contains("Test message"));
    }

    #[test]
    fn test_global_logger() {
        let config = LoggingConfig::default();
        init_logger(config);

        let logger = get_logger();
        assert!(logger.is_some());
    }

    #[test]
    fn test_memory_logging() {
        let logger = AutogradLogger::new();

        let memory_info = MemoryInfo {
            total_allocated: 1024,
            peak_usage: 2048,
            current_usage: 512,
            fragmentation: 10.5,
            active_allocations: 5,
            pool_stats: HashMap::new(),
        };

        logger.log_memory_usage("memory_test", memory_info);

        let entries = logger.get_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].category, OperationCategory::MemoryManagement);
        assert!(entries[0].metrics.contains_key("memory_total"));
    }
}
