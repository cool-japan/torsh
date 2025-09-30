//! Enhanced error handling and propagation for autograd operations
//!
//! This module provides comprehensive error handling utilities, context-aware
//! error messages, error recovery strategies, and debugging information for
//! automatic differentiation operations.

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;
use torsh_core::error::{Result, TorshError};

/// Conversion from AutogradError to TorshError
impl From<AutogradError> for TorshError {
    fn from(error: AutogradError) -> Self {
        TorshError::AutogradError(error.to_string())
    }
}

/// Conversion from std::io::Error to AutogradError
impl From<std::io::Error> for AutogradError {
    fn from(error: std::io::Error) -> Self {
        AutogradError::gradient_computation("io_error", format!("I/O error: {}", error))
    }
}

/// Enhanced autograd-specific error types
#[derive(Debug, Clone)]
pub enum AutogradError {
    /// Gradient computation failed
    GradientComputation {
        operation: String,
        tensor_id: Option<usize>,
        context: String,
        source: Option<Box<AutogradError>>,
    },
    /// Computation graph operation failed
    ComputationGraph {
        node_id: Option<usize>,
        operation: String,
        reason: String,
        graph_state: GraphState,
    },
    /// Shape mismatch in gradient operations
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        operation: String,
        tensor_names: Vec<String>,
    },
    /// Memory allocation failed
    MemoryAllocation {
        requested_size: usize,
        available_size: Option<usize>,
        operation: String,
    },
    /// Numerical instability detected
    NumericalInstability {
        value: f64,
        threshold: f64,
        operation: String,
        suggestion: String,
    },
    /// Backward pass failed
    BackwardPass {
        step: usize,
        total_steps: usize,
        node_name: String,
        underlying_error: String,
    },
    /// Tensor state inconsistency
    TensorState {
        tensor_id: usize,
        expected_state: String,
        actual_state: String,
        operation: String,
    },
    /// Configuration error
    Configuration {
        parameter: String,
        value: String,
        reason: String,
        valid_range: Option<String>,
    },
    /// Resource exhaustion
    ResourceExhaustion {
        resource_type: ResourceType,
        limit: usize,
        requested: usize,
        suggestions: Vec<String>,
    },
}

/// Types of resources that can be exhausted
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    Memory,
    ComputationNodes,
    TensorCount,
    GraphDepth,
    ThreadPoolSize,
}

/// Current state of the computation graph
#[derive(Debug, Clone)]
pub struct GraphState {
    pub node_count: usize,
    pub edge_count: usize,
    pub depth: usize,
    pub has_cycles: bool,
    pub memory_usage: usize,
}

impl fmt::Display for AutogradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutogradError::GradientComputation {
                operation,
                tensor_id,
                context,
                source,
            } => {
                write!(
                    f,
                    "Gradient computation failed in operation '{}': {}",
                    operation, context
                )?;
                if let Some(id) = tensor_id {
                    write!(f, " (tensor ID: {})", id)?;
                }
                if let Some(src) = source {
                    write!(f, "\nCaused by: {}", src)?;
                }
                Ok(())
            }
            AutogradError::ComputationGraph {
                node_id,
                operation,
                reason,
                graph_state,
            } => {
                write!(f, "Computation graph error in '{}': {}", operation, reason)?;
                if let Some(id) = node_id {
                    write!(f, " (node ID: {})", id)?;
                }
                write!(
                    f,
                    "\nGraph state: {} nodes, {} edges, depth {}",
                    graph_state.node_count, graph_state.edge_count, graph_state.depth
                )?;
                Ok(())
            }
            AutogradError::ShapeMismatch {
                expected,
                actual,
                operation,
                tensor_names,
            } => {
                write!(
                    f,
                    "Shape mismatch in operation '{}': expected {:?}, got {:?}",
                    operation, expected, actual
                )?;
                if !tensor_names.is_empty() {
                    write!(f, "\nTensors involved: {}", tensor_names.join(", "))?;
                }
                Ok(())
            }
            AutogradError::MemoryAllocation {
                requested_size,
                available_size,
                operation,
            } => {
                write!(
                    f,
                    "Memory allocation failed in '{}': requested {} bytes",
                    operation, requested_size
                )?;
                if let Some(available) = available_size {
                    write!(f, ", available {} bytes", available)?;
                }
                Ok(())
            }
            AutogradError::NumericalInstability {
                value,
                threshold,
                operation,
                suggestion,
            } => {
                write!(f, "Numerical instability detected in '{}': value {} exceeds threshold {}\nSuggestion: {}", 
                       operation, value, threshold, suggestion)
            }
            AutogradError::BackwardPass {
                step,
                total_steps,
                node_name,
                underlying_error,
            } => {
                write!(
                    f,
                    "Backward pass failed at step {}/{} in node '{}': {}",
                    step, total_steps, node_name, underlying_error
                )
            }
            AutogradError::TensorState {
                tensor_id,
                expected_state,
                actual_state,
                operation,
            } => {
                write!(f, "Tensor state inconsistency in '{}': tensor {} expected to be '{}', but was '{}'", 
                       operation, tensor_id, expected_state, actual_state)
            }
            AutogradError::Configuration {
                parameter,
                value,
                reason,
                valid_range,
            } => {
                write!(
                    f,
                    "Configuration error: parameter '{}' = '{}' is invalid: {}",
                    parameter, value, reason
                )?;
                if let Some(range) = valid_range {
                    write!(f, "\nValid range: {}", range)?;
                }
                Ok(())
            }
            AutogradError::ResourceExhaustion {
                resource_type,
                limit,
                requested,
                suggestions,
            } => {
                write!(
                    f,
                    "Resource exhaustion: {:?} limit {} exceeded (requested {})",
                    resource_type, limit, requested
                )?;
                if !suggestions.is_empty() {
                    write!(f, "\nSuggestions: {}", suggestions.join(", "))?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for AutogradError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AutogradError::GradientComputation {
                source: Some(src), ..
            } => Some(src as &dyn std::error::Error),
            _ => None,
        }
    }
}

/// Error context manager for tracking operation stack
#[derive(Debug, Clone)]
pub struct ErrorContext {
    operation_stack: Vec<OperationContext>,
    tensor_metadata: HashMap<usize, TensorMetadata>,
    error_count: usize,
    warning_count: usize,
}

/// Context information for an operation
#[derive(Debug, Clone)]
pub struct OperationContext {
    pub name: String,
    pub start_time: Instant,
    pub tensor_ids: Vec<usize>,
    pub metadata: HashMap<String, String>,
}

/// Metadata about a tensor for error reporting
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub name: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub requires_grad: bool,
    pub creation_location: Option<String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new() -> Self {
        Self {
            operation_stack: Vec::new(),
            tensor_metadata: HashMap::new(),
            error_count: 0,
            warning_count: 0,
        }
    }

    /// Push an operation onto the context stack
    pub fn push_operation(&mut self, name: impl Into<String>, tensor_ids: Vec<usize>) {
        let operation = OperationContext {
            name: name.into(),
            start_time: Instant::now(),
            tensor_ids,
            metadata: HashMap::new(),
        };
        self.operation_stack.push(operation);
    }

    /// Pop an operation from the context stack
    pub fn pop_operation(&mut self) -> Option<OperationContext> {
        self.operation_stack.pop()
    }

    /// Register tensor metadata
    pub fn register_tensor(&mut self, id: usize, metadata: TensorMetadata) {
        self.tensor_metadata.insert(id, metadata);
    }

    /// Get the current operation stack as a string
    pub fn operation_stack_string(&self) -> String {
        if self.operation_stack.is_empty() {
            "No active operations".to_string()
        } else {
            let operations: Vec<String> = self
                .operation_stack
                .iter()
                .map(|op| {
                    let elapsed = op.start_time.elapsed();
                    format!("{} ({:?})", op.name, elapsed)
                })
                .collect();
            format!("Operation stack: {}", operations.join(" -> "))
        }
    }

    /// Get tensor names involved in current operation
    pub fn current_tensor_names(&self) -> Vec<String> {
        if let Some(current_op) = self.operation_stack.last() {
            current_op
                .tensor_ids
                .iter()
                .filter_map(|&id| {
                    self.tensor_metadata
                        .get(&id)
                        .and_then(|meta| meta.name.clone())
                        .or_else(|| Some(format!("tensor_{}", id)))
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Increment error count
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Increment warning count
    pub fn record_warning(&mut self) {
        self.warning_count += 1;
    }

    /// Get error statistics
    pub fn error_stats(&self) -> (usize, usize) {
        (self.error_count, self.warning_count)
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    static ERROR_CONTEXT: std::cell::RefCell<ErrorContext> = std::cell::RefCell::new(ErrorContext::new());
}

/// Execute a function with error context tracking
pub fn with_error_context<F, R>(
    operation_name: impl Into<String>,
    tensor_ids: Vec<usize>,
    f: F,
) -> Result<R>
where
    F: FnOnce() -> Result<R>,
{
    let op_name = operation_name.into();

    // Push operation context
    ERROR_CONTEXT.with(|ctx| {
        ctx.borrow_mut().push_operation(op_name.clone(), tensor_ids);
    });

    // Execute function and handle errors
    let result = match f() {
        Ok(value) => Ok(value),
        Err(e) => {
            // Record error and enhance with context
            ERROR_CONTEXT.with(|ctx| {
                let mut ctx_ref = ctx.borrow_mut();
                ctx_ref.record_error();

                // Create enhanced error with context
                let context_string = ctx_ref.operation_stack_string();
                let tensor_names = ctx_ref.current_tensor_names();

                match e {
                    TorshError::AutogradError(msg) => Err(TorshError::AutogradError(format!(
                        "{}\nContext: {}\nTensors: [{}]",
                        msg,
                        context_string,
                        tensor_names.join(", ")
                    ))),
                    other => Err(other),
                }
            })
        }
    };

    // Pop operation context
    ERROR_CONTEXT.with(|ctx| {
        ctx.borrow_mut().pop_operation();
    });

    result
}

/// Register tensor metadata for error reporting
pub fn register_tensor_metadata(id: usize, metadata: TensorMetadata) {
    ERROR_CONTEXT.with(|ctx| {
        ctx.borrow_mut().register_tensor(id, metadata);
    });
}

/// Get current error context information
pub fn get_error_context_info() -> String {
    ERROR_CONTEXT.with(|ctx| ctx.borrow().operation_stack_string())
}

/// Error recovery strategies
pub mod recovery {
    use super::*;

    /// Recovery strategy for autograd operations
    #[derive(Debug, Clone)]
    pub enum RecoveryStrategy {
        /// Retry the operation with different parameters
        Retry {
            max_attempts: usize,
            backoff_ms: u64,
            parameter_adjustments: HashMap<String, String>,
        },
        /// Fallback to a simpler algorithm
        Fallback {
            fallback_operation: String,
            performance_impact: String,
        },
        /// Skip the operation and continue
        Skip {
            default_value: Option<String>,
            impact_assessment: String,
        },
        /// Abort the entire computation
        Abort {
            cleanup_required: bool,
            error_message: String,
        },
    }

    /// Recovery manager for autograd operations
    pub struct RecoveryManager {
        strategies: HashMap<String, RecoveryStrategy>,
        recovery_attempts: HashMap<String, usize>,
        max_recovery_attempts: usize,
    }

    impl RecoveryManager {
        /// Create a new recovery manager
        pub fn new() -> Self {
            let mut manager = Self {
                strategies: HashMap::new(),
                recovery_attempts: HashMap::new(),
                max_recovery_attempts: 3,
            };
            manager.setup_default_strategies();
            manager
        }

        /// Set up default recovery strategies
        fn setup_default_strategies(&mut self) {
            // Gradient computation recovery
            self.strategies.insert(
                "gradient_computation".to_string(),
                RecoveryStrategy::Retry {
                    max_attempts: 3,
                    backoff_ms: 100,
                    parameter_adjustments: {
                        let mut adj = HashMap::new();
                        adj.insert("eps".to_string(), "increase_by_10x".to_string());
                        adj.insert("tolerance".to_string(), "decrease_by_10x".to_string());
                        adj
                    },
                },
            );

            // Memory allocation recovery
            self.strategies.insert(
                "memory_allocation".to_string(),
                RecoveryStrategy::Fallback {
                    fallback_operation: "use_cpu_memory".to_string(),
                    performance_impact: "May be 2-5x slower".to_string(),
                },
            );

            // Numerical instability recovery
            self.strategies.insert(
                "numerical_instability".to_string(),
                RecoveryStrategy::Retry {
                    max_attempts: 2,
                    backoff_ms: 0,
                    parameter_adjustments: {
                        let mut adj = HashMap::new();
                        adj.insert("precision".to_string(), "increase".to_string());
                        adj.insert(
                            "regularization".to_string(),
                            "add_small_epsilon".to_string(),
                        );
                        adj
                    },
                },
            );
        }

        /// Attempt recovery for a failed operation
        pub fn attempt_recovery(
            &mut self,
            operation: &str,
            error: &AutogradError,
        ) -> Option<RecoveryStrategy> {
            let current_attempts = self.recovery_attempts.get(operation).unwrap_or(&0);

            if *current_attempts >= self.max_recovery_attempts {
                tracing::warn!(
                    "Maximum recovery attempts exceeded for operation: {}",
                    operation
                );
                return None;
            }

            // Find appropriate strategy based on error type
            let strategy_key = match error {
                AutogradError::GradientComputation { .. } => "gradient_computation",
                AutogradError::MemoryAllocation { .. } => "memory_allocation",
                AutogradError::NumericalInstability { .. } => "numerical_instability",
                _ => "default",
            };

            if let Some(strategy) = self.strategies.get(strategy_key).cloned() {
                // Increment attempt counter
                self.recovery_attempts
                    .insert(operation.to_string(), current_attempts + 1);

                tracing::info!(
                    "Attempting recovery for operation '{}' using strategy: {:?}",
                    operation,
                    strategy
                );

                Some(strategy)
            } else {
                tracing::warn!("No recovery strategy found for operation: {}", operation);
                None
            }
        }

        /// Reset recovery attempts for an operation
        pub fn reset_attempts(&mut self, operation: &str) {
            self.recovery_attempts.remove(operation);
        }

        /// Get recovery statistics
        pub fn get_stats(&self) -> HashMap<String, usize> {
            self.recovery_attempts.clone()
        }
    }

    impl Default for RecoveryManager {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Automatic error recovery executor
    pub struct AutoRecoveryExecutor {
        recovery_manager: RecoveryManager,
        enable_auto_recovery: bool,
        recovery_statistics: RecoveryStatistics,
    }

    /// Statistics about recovery attempts
    #[derive(Debug, Clone, Default)]
    pub struct RecoveryStatistics {
        pub total_errors: usize,
        pub successful_recoveries: usize,
        pub failed_recoveries: usize,
        pub recovery_by_strategy: HashMap<String, usize>,
        pub most_common_errors: HashMap<String, usize>,
    }

    impl AutoRecoveryExecutor {
        /// Create a new auto recovery executor
        pub fn new(enable_auto_recovery: bool) -> Self {
            Self {
                recovery_manager: RecoveryManager::new(),
                enable_auto_recovery,
                recovery_statistics: RecoveryStatistics::default(),
            }
        }

        /// Execute an operation with automatic error recovery
        pub fn execute_with_recovery<F, R>(
            &mut self,
            operation_name: &str,
            operation: F,
        ) -> Result<R>
        where
            F: Fn() -> Result<R> + Clone,
        {
            if !self.enable_auto_recovery {
                return operation();
            }

            let mut _last_error = None;
            let mut attempts = 0;
            const MAX_ATTEMPTS: usize = 3;

            loop {
                attempts += 1;

                match operation() {
                    Ok(result) => {
                        // Success - reset recovery attempts for this operation
                        self.recovery_manager.reset_attempts(operation_name);
                        if attempts > 1 {
                            self.recovery_statistics.successful_recoveries += 1;
                            tracing::info!("Successfully recovered from error in operation '{}' after {} attempts", 
                                         operation_name, attempts);
                        }
                        return Ok(result);
                    }
                    Err(e) => {
                        self.recovery_statistics.total_errors += 1;
                        _last_error = Some(e.clone());

                        // Convert TorshError to AutogradError for recovery analysis
                        let autograd_error = self.convert_to_autograd_error(&e, operation_name);

                        // Record error type for statistics
                        let error_type = self.classify_error(&autograd_error);
                        *self
                            .recovery_statistics
                            .most_common_errors
                            .entry(error_type.clone())
                            .or_insert(0) += 1;

                        if attempts >= MAX_ATTEMPTS {
                            self.recovery_statistics.failed_recoveries += 1;
                            tracing::error!(
                                "Failed to recover from error in operation '{}' after {} attempts",
                                operation_name,
                                attempts
                            );
                            return Err(e);
                        }

                        // Attempt recovery
                        if let Some(strategy) = self
                            .recovery_manager
                            .attempt_recovery(operation_name, &autograd_error)
                        {
                            *self
                                .recovery_statistics
                                .recovery_by_strategy
                                .entry(format!("{:?}", strategy))
                                .or_insert(0) += 1;

                            // Apply recovery strategy
                            if let Err(recovery_error) =
                                self.apply_recovery_strategy(&strategy, operation_name)
                            {
                                tracing::warn!("Recovery strategy failed: {:?}", recovery_error);
                                continue;
                            }

                            // Wait if backoff is specified
                            if let RecoveryStrategy::Retry { backoff_ms, .. } = strategy {
                                if backoff_ms > 0 {
                                    std::thread::sleep(std::time::Duration::from_millis(
                                        backoff_ms,
                                    ));
                                }
                            }
                        } else {
                            // No recovery strategy available
                            return Err(e);
                        }
                    }
                }
            }
        }

        /// Convert TorshError to AutogradError for recovery analysis
        fn convert_to_autograd_error(&self, error: &TorshError, operation: &str) -> AutogradError {
            match error {
                TorshError::AutogradError(msg) => {
                    // Try to parse the error message to determine specific error type
                    if msg.contains("shape") || msg.contains("dimension") {
                        AutogradError::ShapeMismatch {
                            expected: vec![],
                            actual: vec![],
                            operation: operation.to_string(),
                            tensor_names: vec![],
                        }
                    } else if msg.contains("memory") || msg.contains("allocation") {
                        AutogradError::MemoryAllocation {
                            requested_size: 0,
                            available_size: None,
                            operation: operation.to_string(),
                        }
                    } else if msg.contains("NaN")
                        || msg.contains("infinite")
                        || msg.contains("numerical")
                    {
                        AutogradError::NumericalInstability {
                            value: 0.0,
                            threshold: 0.0,
                            operation: operation.to_string(),
                            suggestion: "Consider using more stable numerical methods".to_string(),
                        }
                    } else {
                        AutogradError::GradientComputation {
                            operation: operation.to_string(),
                            tensor_id: None,
                            context: msg.clone(),
                            source: None,
                        }
                    }
                }
                _ => AutogradError::GradientComputation {
                    operation: operation.to_string(),
                    tensor_id: None,
                    context: format!("{:?}", error),
                    source: None,
                },
            }
        }

        /// Classify error for statistics
        fn classify_error(&self, error: &AutogradError) -> String {
            match error {
                AutogradError::GradientComputation { .. } => "gradient_computation".to_string(),
                AutogradError::ComputationGraph { .. } => "computation_graph".to_string(),
                AutogradError::ShapeMismatch { .. } => "shape_mismatch".to_string(),
                AutogradError::MemoryAllocation { .. } => "memory_allocation".to_string(),
                AutogradError::NumericalInstability { .. } => "numerical_instability".to_string(),
                AutogradError::BackwardPass { .. } => "backward_pass".to_string(),
                AutogradError::TensorState { .. } => "tensor_state".to_string(),
                AutogradError::Configuration { .. } => "configuration".to_string(),
                AutogradError::ResourceExhaustion { .. } => "resource_exhaustion".to_string(),
            }
        }

        /// Apply a recovery strategy
        fn apply_recovery_strategy(
            &mut self,
            strategy: &RecoveryStrategy,
            operation: &str,
        ) -> Result<()> {
            match strategy {
                RecoveryStrategy::Retry {
                    parameter_adjustments,
                    ..
                } => {
                    // Apply parameter adjustments for retry
                    self.apply_parameter_adjustments(parameter_adjustments, operation)?;
                    tracing::debug!(
                        "Applied parameter adjustments for retry in operation '{}'",
                        operation
                    );
                }
                RecoveryStrategy::Fallback {
                    fallback_operation,
                    performance_impact,
                } => {
                    tracing::info!(
                        "Falling back to '{}' for operation '{}'. {}",
                        fallback_operation,
                        operation,
                        performance_impact
                    );
                    // In a real implementation, this would switch to the fallback algorithm
                }
                RecoveryStrategy::Skip {
                    default_value,
                    impact_assessment,
                } => {
                    tracing::warn!("Skipping operation '{}'. {}", operation, impact_assessment);
                    if let Some(default) = default_value {
                        tracing::info!("Using default value: {}", default);
                    }
                }
                RecoveryStrategy::Abort {
                    cleanup_required,
                    error_message,
                } => {
                    if *cleanup_required {
                        tracing::info!("Performing cleanup before abort");
                        // Perform cleanup operations
                    }
                    return Err(TorshError::AutogradError(error_message.clone()));
                }
            }
            Ok(())
        }

        /// Apply parameter adjustments for recovery
        fn apply_parameter_adjustments(
            &self,
            adjustments: &HashMap<String, String>,
            operation: &str,
        ) -> Result<()> {
            for (param, adjustment) in adjustments {
                tracing::debug!(
                    "Adjusting parameter '{}' with '{}' for operation '{}'",
                    param,
                    adjustment,
                    operation
                );

                // In a real implementation, this would modify the actual parameters
                // For now, we just log the adjustments
                match adjustment.as_str() {
                    "increase_by_10x" => tracing::debug!("Increasing {} by 10x", param),
                    "decrease_by_10x" => tracing::debug!("Decreasing {} by 10x", param),
                    "increase" => tracing::debug!("Increasing {}", param),
                    "add_small_epsilon" => tracing::debug!("Adding small epsilon to {}", param),
                    _ => tracing::debug!("Applying adjustment '{}' to {}", adjustment, param),
                }
            }
            Ok(())
        }

        /// Get recovery statistics
        pub fn get_statistics(&self) -> &RecoveryStatistics {
            &self.recovery_statistics
        }

        /// Reset recovery statistics
        pub fn reset_statistics(&mut self) {
            self.recovery_statistics = RecoveryStatistics::default();
        }

        /// Enable or disable automatic recovery
        pub fn set_auto_recovery(&mut self, enabled: bool) {
            self.enable_auto_recovery = enabled;
            tracing::info!(
                "Automatic error recovery {}",
                if enabled { "enabled" } else { "disabled" }
            );
        }

        /// Add a custom recovery strategy
        pub fn add_custom_strategy(&mut self, operation: String, strategy: RecoveryStrategy) {
            self.recovery_manager
                .strategies
                .insert(operation.clone(), strategy);
            tracing::info!(
                "Added custom recovery strategy for operation '{}'",
                operation
            );
        }

        /// Generate recovery report
        pub fn generate_recovery_report(&self) -> String {
            let stats = &self.recovery_statistics;
            let mut report = String::new();

            report.push_str("=== Automatic Error Recovery Report ===\n");
            report.push_str(&format!(
                "Total errors encountered: {}\n",
                stats.total_errors
            ));
            report.push_str(&format!(
                "Successful recoveries: {}\n",
                stats.successful_recoveries
            ));
            report.push_str(&format!("Failed recoveries: {}\n", stats.failed_recoveries));

            if stats.total_errors > 0 {
                let recovery_rate =
                    (stats.successful_recoveries as f64 / stats.total_errors as f64) * 100.0;
                report.push_str(&format!("Recovery success rate: {:.1}%\n", recovery_rate));
            }

            if !stats.most_common_errors.is_empty() {
                report.push_str("\nMost common error types:\n");
                let mut errors: Vec<_> = stats.most_common_errors.iter().collect();
                errors.sort_by(|a, b| b.1.cmp(a.1));

                for (error_type, count) in errors.iter().take(5) {
                    report.push_str(&format!("  - {}: {} occurrences\n", error_type, count));
                }
            }

            if !stats.recovery_by_strategy.is_empty() {
                report.push_str("\nRecovery strategies used:\n");
                for (strategy, count) in &stats.recovery_by_strategy {
                    report.push_str(&format!("  - {}: {} times\n", strategy, count));
                }
            }

            report
        }
    }
}

/// Validation utilities for autograd operations
pub mod validation {
    use super::*;

    /// Validate tensor shapes for operations
    pub fn validate_tensor_shapes(
        expected: &[usize],
        actual: &[usize],
        operation: &str,
        tensor_name: Option<&str>,
    ) -> Result<()> {
        if expected != actual {
            let error = AutogradError::ShapeMismatch {
                expected: expected.to_vec(),
                actual: actual.to_vec(),
                operation: operation.to_string(),
                tensor_names: tensor_name.map(|n| vec![n.to_string()]).unwrap_or_default(),
            };
            return Err(TorshError::AutogradError(error.to_string()));
        }
        Ok(())
    }

    /// Validate numerical stability
    pub fn validate_numerical_stability(value: f64, operation: &str, threshold: f64) -> Result<()> {
        if value.is_nan() {
            return Err(TorshError::AutogradError(format!(
                "NaN detected in operation '{}': consider using gradient clipping or regularization",
                operation
            )));
        }

        if value.is_infinite() {
            return Err(TorshError::AutogradError(format!(
                "Infinity detected in operation '{}': gradient may be exploding",
                operation
            )));
        }

        if value.abs() > threshold {
            let suggestion = if value.abs() > 1e10 {
                "Consider gradient clipping or reducing learning rate"
            } else {
                "Consider adding regularization"
            };

            let error = AutogradError::NumericalInstability {
                value,
                threshold,
                operation: operation.to_string(),
                suggestion: suggestion.to_string(),
            };
            return Err(TorshError::AutogradError(error.to_string()));
        }

        Ok(())
    }

    /// Validate tensor state consistency
    pub fn validate_tensor_state(
        tensor_id: usize,
        expected_requires_grad: bool,
        actual_requires_grad: bool,
        operation: &str,
    ) -> Result<()> {
        if expected_requires_grad != actual_requires_grad {
            let error = AutogradError::TensorState {
                tensor_id,
                expected_state: if expected_requires_grad {
                    "requires_grad=True"
                } else {
                    "requires_grad=False"
                }
                .to_string(),
                actual_state: if actual_requires_grad {
                    "requires_grad=True"
                } else {
                    "requires_grad=False"
                }
                .to_string(),
                operation: operation.to_string(),
            };
            return Err(TorshError::AutogradError(error.to_string()));
        }
        Ok(())
    }

    /// Validate memory requirements
    pub fn validate_memory_requirements(
        required_bytes: usize,
        available_bytes: Option<usize>,
        operation: &str,
    ) -> Result<()> {
        if let Some(available) = available_bytes {
            if required_bytes > available {
                let error = AutogradError::MemoryAllocation {
                    requested_size: required_bytes,
                    available_size: Some(available),
                    operation: operation.to_string(),
                };
                return Err(TorshError::AutogradError(error.to_string()));
            }
        }

        // Check for unreasonably large allocations
        const MAX_REASONABLE_ALLOCATION: usize = 10 * 1024 * 1024 * 1024; // 10GB
        if required_bytes > MAX_REASONABLE_ALLOCATION {
            tracing::warn!(
                "Large memory allocation requested in '{}': {} bytes. Consider using gradient checkpointing.",
                operation, required_bytes
            );
        }

        Ok(())
    }
}

/// Convenience functions for creating common autograd errors
impl AutogradError {
    /// Create a gradient computation error
    pub fn gradient_computation(operation: impl Into<String>, context: impl Into<String>) -> Self {
        AutogradError::GradientComputation {
            operation: operation.into(),
            tensor_id: None,
            context: context.into(),
            source: None,
        }
    }

    /// Create a gradient computation error with tensor ID
    pub fn gradient_computation_with_id(
        operation: impl Into<String>,
        tensor_id: usize,
        context: impl Into<String>,
    ) -> Self {
        AutogradError::GradientComputation {
            operation: operation.into(),
            tensor_id: Some(tensor_id),
            context: context.into(),
            source: None,
        }
    }

    /// Create a shape mismatch error
    pub fn shape_mismatch(
        operation: impl Into<String>,
        expected: Vec<usize>,
        actual: Vec<usize>,
    ) -> Self {
        AutogradError::ShapeMismatch {
            expected,
            actual,
            operation: operation.into(),
            tensor_names: vec![],
        }
    }

    /// Create a memory allocation error
    pub fn memory_allocation(operation: impl Into<String>, requested_size: usize) -> Self {
        AutogradError::MemoryAllocation {
            requested_size,
            available_size: None,
            operation: operation.into(),
        }
    }

    /// Create a numerical instability error
    pub fn numerical_instability(
        operation: impl Into<String>,
        value: f64,
        suggestion: impl Into<String>,
    ) -> Self {
        AutogradError::NumericalInstability {
            value,
            threshold: 1e10,
            operation: operation.into(),
            suggestion: suggestion.into(),
        }
    }

    /// Create a backward pass error
    pub fn backward_pass(
        step: usize,
        total_steps: usize,
        node_name: impl Into<String>,
        underlying_error: impl Into<String>,
    ) -> Self {
        AutogradError::BackwardPass {
            step,
            total_steps,
            node_name: node_name.into(),
            underlying_error: underlying_error.into(),
        }
    }
}

/// Macros for convenient error creation and propagation
#[macro_export]
macro_rules! autograd_error {
    (gradient_computation, $op:expr, $ctx:expr) => {
        AutogradError::gradient_computation($op, $ctx)
    };
    (gradient_computation, $op:expr, $tensor_id:expr, $ctx:expr) => {
        AutogradError::gradient_computation_with_id($op, $tensor_id, $ctx)
    };
    (shape_mismatch, $op:expr, $expected:expr, $actual:expr) => {
        AutogradError::shape_mismatch($op, $expected, $actual)
    };
    (memory_allocation, $op:expr, $size:expr) => {
        AutogradError::memory_allocation($op, $size)
    };
    (numerical_instability, $op:expr, $value:expr, $suggestion:expr) => {
        AutogradError::numerical_instability($op, $value, $suggestion)
    };
    (backward_pass, $step:expr, $total:expr, $node:expr, $error:expr) => {
        AutogradError::backward_pass($step, $total, $node, $error)
    };
}

/// Macro for propagating autograd errors with additional context
#[macro_export]
macro_rules! autograd_propagate {
    ($result:expr, $operation:expr) => {
        $result.map_err(|e| match e {
            TorshError::AutogradError(msg) => AutogradError::gradient_computation($operation, msg),
            _ => AutogradError::gradient_computation($operation, e.to_string()),
        })?
    };
}

/// Result type alias for autograd operations
pub type AutogradResult<T> = std::result::Result<T, AutogradError>;

/// Convert AutogradResult to torsh-core Result
pub fn to_torsh_result<T>(result: AutogradResult<T>) -> Result<T> {
    result.map_err(|e| e.into())
}

/// Context builder for enhanced error messages
pub struct ErrorContextBuilder {
    operation: String,
    tensor_info: HashMap<String, String>,
    additional_context: Vec<String>,
}

impl ErrorContextBuilder {
    /// Create a new error context
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            tensor_info: HashMap::new(),
            additional_context: Vec::new(),
        }
    }

    /// Add tensor information
    pub fn with_tensor(mut self, name: impl Into<String>, info: impl Into<String>) -> Self {
        self.tensor_info.insert(name.into(), info.into());
        self
    }

    /// Add additional context
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.additional_context.push(context.into());
        self
    }

    /// Build a gradient computation error with this context
    pub fn gradient_error(self, message: impl Into<String>) -> AutogradError {
        let mut context = message.into();

        if !self.tensor_info.is_empty() {
            context.push_str(&format!("\nTensor info: {:?}", self.tensor_info));
        }

        if !self.additional_context.is_empty() {
            context.push_str(&format!(
                "\nAdditional context: {}",
                self.additional_context.join(", ")
            ));
        }

        AutogradError::gradient_computation(self.operation, context)
    }

    /// Build a shape mismatch error with this context
    pub fn shape_error(self, expected: Vec<usize>, actual: Vec<usize>) -> AutogradError {
        let mut error = AutogradError::shape_mismatch(self.operation, expected, actual);

        if let AutogradError::ShapeMismatch { tensor_names, .. } = &mut error {
            *tensor_names = self.tensor_info.keys().cloned().collect();
        }

        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_operations() {
        let mut ctx = ErrorContext::new();

        ctx.push_operation("test_op", vec![1, 2]);
        assert_eq!(ctx.operation_stack.len(), 1);
        assert_eq!(ctx.operation_stack[0].name, "test_op");
        assert_eq!(ctx.operation_stack[0].tensor_ids, vec![1, 2]);

        let popped = ctx.pop_operation();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().name, "test_op");
        assert_eq!(ctx.operation_stack.len(), 0);
    }

    #[test]
    fn test_tensor_metadata_registration() {
        let mut ctx = ErrorContext::new();
        let metadata = TensorMetadata {
            name: Some("test_tensor".to_string()),
            shape: vec![2, 3],
            dtype: "f32".to_string(),
            device: "cpu".to_string(),
            requires_grad: true,
            creation_location: Some("test.rs:42".to_string()),
        };

        ctx.register_tensor(123, metadata.clone());
        assert!(ctx.tensor_metadata.contains_key(&123));
        assert_eq!(ctx.tensor_metadata[&123].name, metadata.name);
    }

    #[test]
    fn test_autograd_error_display() {
        let error = AutogradError::GradientComputation {
            operation: "backward".to_string(),
            tensor_id: Some(42),
            context: "invalid gradient shape".to_string(),
            source: None,
        };

        let error_str = error.to_string();
        assert!(error_str.contains("Gradient computation failed"));
        assert!(error_str.contains("backward"));
        assert!(error_str.contains("tensor ID: 42"));
    }

    #[test]
    fn test_shape_mismatch_error() {
        let error = AutogradError::ShapeMismatch {
            expected: vec![2, 3],
            actual: vec![3, 2],
            operation: "matmul".to_string(),
            tensor_names: vec!["A".to_string(), "B".to_string()],
        };

        let error_str = error.to_string();
        assert!(error_str.contains("Shape mismatch"));
        assert!(error_str.contains("matmul"));
        assert!(error_str.contains("[2, 3]"));
        assert!(error_str.contains("[3, 2]"));
        assert!(error_str.contains("A, B"));
    }

    #[test]
    fn test_with_error_context() {
        let result = with_error_context("test_operation", vec![1, 2], || Ok(42));

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_error_context_with_failure() {
        let result: Result<i32> = with_error_context("failing_operation", vec![1], || {
            Err(TorshError::AutogradError("original error".to_string()))
        });

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("original error"));
        assert!(error_msg.contains("Context:"));
    }

    #[test]
    fn test_recovery_manager() {
        let mut manager = recovery::RecoveryManager::new();

        let error = AutogradError::NumericalInstability {
            value: 1e20,
            threshold: 1e10,
            operation: "backward".to_string(),
            suggestion: "test".to_string(),
        };

        let strategy = manager.attempt_recovery("test_op", &error);
        assert!(strategy.is_some());

        // Test max attempts
        for _ in 0..5 {
            manager.attempt_recovery("test_op", &error);
        }

        let no_strategy = manager.attempt_recovery("test_op", &error);
        assert!(no_strategy.is_none());
    }

    #[test]
    fn test_validation_functions() {
        // Test shape validation
        assert!(validation::validate_tensor_shapes(&[2, 3], &[2, 3], "test", None).is_ok());

        assert!(validation::validate_tensor_shapes(&[2, 3], &[3, 2], "test", None).is_err());

        // Test numerical stability
        assert!(validation::validate_numerical_stability(1.0, "test", 1e10).is_ok());

        assert!(validation::validate_numerical_stability(f64::NAN, "test", 1e10).is_err());

        assert!(validation::validate_numerical_stability(f64::INFINITY, "test", 1e10).is_err());

        // Test tensor state
        assert!(validation::validate_tensor_state(1, true, true, "test").is_ok());

        assert!(validation::validate_tensor_state(1, true, false, "test").is_err());
    }

    #[test]
    fn test_autograd_error_convenience_methods() {
        // Test gradient computation error
        let error = AutogradError::gradient_computation("test_op", "test failed");
        match error {
            AutogradError::GradientComputation {
                operation, context, ..
            } => {
                assert_eq!(operation, "test_op");
                assert_eq!(context, "test failed");
            }
            _ => panic!("Wrong error type"),
        }

        // Test shape mismatch error
        let error = AutogradError::shape_mismatch("matmul", vec![2, 3], vec![3, 4]);
        match error {
            AutogradError::ShapeMismatch {
                operation,
                expected,
                actual,
                ..
            } => {
                assert_eq!(operation, "matmul");
                assert_eq!(expected, vec![2, 3]);
                assert_eq!(actual, vec![3, 4]);
            }
            _ => panic!("Wrong error type"),
        }

        // Test memory allocation error
        let error = AutogradError::memory_allocation("alloc", 1024);
        match error {
            AutogradError::MemoryAllocation {
                operation,
                requested_size,
                ..
            } => {
                assert_eq!(operation, "alloc");
                assert_eq!(requested_size, 1024);
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_error_context_builder() {
        let context = ErrorContextBuilder::new("test_operation")
            .with_tensor("input", "shape: [2, 3], dtype: f32")
            .with_tensor("weight", "shape: [3, 4], dtype: f32")
            .with_context("batch_size=32")
            .with_context("training=true");

        let error = context.gradient_error("Forward pass failed");

        match error {
            AutogradError::GradientComputation {
                operation, context, ..
            } => {
                assert_eq!(operation, "test_operation");
                assert!(context.contains("Forward pass failed"));
                assert!(context.contains("Tensor info:"));
                assert!(context.contains("batch_size=32"));
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_torsh_error_conversion() {
        let autograd_error = AutogradError::gradient_computation("test", "test failure");
        let torsh_error: TorshError = autograd_error.into();

        match torsh_error {
            TorshError::AutogradError(msg) => {
                assert!(msg.contains("test"));
                assert!(msg.contains("test failure"));
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_autograd_result_conversion() {
        let success: AutogradResult<i32> = Ok(42);
        let converted = to_torsh_result(success).unwrap();
        assert_eq!(converted, 42);

        let error: AutogradResult<i32> = Err(AutogradError::gradient_computation("test", "failed"));
        let converted = to_torsh_result(error);
        assert!(converted.is_err());
    }
}
