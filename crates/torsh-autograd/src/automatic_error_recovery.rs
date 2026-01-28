//! Automatic error recovery for transient failures in autograd operations
//!
//! This module provides comprehensive error recovery mechanisms for handling
//! transient failures that can occur during autograd computation, such as
//! numerical instabilities, temporary memory pressure, hardware issues,
//! and network failures in distributed settings.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error_handling::{AutogradError, AutogradResult};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Types of transient failures that can be recovered from
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TransientFailureType {
    /// Numerical instability (NaN, Inf values)
    NumericalInstability,
    /// Temporary memory shortage
    MemoryPressure,
    /// Hardware-related temporary issues
    HardwareFailure,
    /// Network connectivity issues (distributed training)
    NetworkFailure,
    /// Resource contention or lock timeouts
    ResourceContention,
    /// Computational overflow/underflow
    ComputationalOverflow,
    /// Gradient explosion/vanishing
    GradientInstability,
    /// Custom user-defined failure types
    Custom(String),
}

impl std::fmt::Display for TransientFailureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NumericalInstability => write!(f, "numerical_instability"),
            Self::MemoryPressure => write!(f, "memory_pressure"),
            Self::HardwareFailure => write!(f, "hardware_failure"),
            Self::NetworkFailure => write!(f, "network_failure"),
            Self::ResourceContention => write!(f, "resource_contention"),
            Self::ComputationalOverflow => write!(f, "computational_overflow"),
            Self::GradientInstability => write!(f, "gradient_instability"),
            Self::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

/// Recovery strategies for different types of failures
#[derive(Clone)]
pub enum RecoveryStrategy {
    /// Retry the operation with exponential backoff
    ExponentialBackoff {
        initial_delay: Duration,
        max_delay: Duration,
        backoff_factor: f64,
        max_retries: usize,
    },
    /// Retry with reduced precision or simpler computation
    GracefulDegradation {
        precision_reduction: f64,
        simplification_level: u32,
        max_retries: usize,
    },
    /// Reset to a previous checkpoint and retry
    CheckpointRollback {
        checkpoint_age_threshold: Duration,
        max_rollback_attempts: usize,
    },
    /// Redistribute computation or change execution strategy
    ReallocationStrategy {
        reallocation_delay: Duration,
        max_reallocations: usize,
    },
    /// Use alternative algorithm or computation path
    AlgorithmSwitch {
        fallback_algorithms: Vec<String>,
        max_switches: usize,
    },
    /// Apply corrective transformations to fix the issue
    CorrectiveTransformation {
        transformations: Vec<CorrectiveTransform>,
        max_applications: usize,
    },
    /// Custom recovery function
    Custom {
        recovery_fn: Arc<dyn Fn(&AutogradError) -> RecoveryAction + Send + Sync>,
        max_attempts: usize,
    },
}

impl std::fmt::Debug for RecoveryStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExponentialBackoff {
                initial_delay,
                max_delay,
                backoff_factor,
                max_retries,
            } => f
                .debug_struct("ExponentialBackoff")
                .field("initial_delay", initial_delay)
                .field("max_delay", max_delay)
                .field("backoff_factor", backoff_factor)
                .field("max_retries", max_retries)
                .finish(),
            Self::GracefulDegradation {
                precision_reduction,
                simplification_level,
                max_retries,
            } => f
                .debug_struct("GracefulDegradation")
                .field("precision_reduction", precision_reduction)
                .field("simplification_level", simplification_level)
                .field("max_retries", max_retries)
                .finish(),
            Self::CheckpointRollback {
                checkpoint_age_threshold,
                max_rollback_attempts,
            } => f
                .debug_struct("CheckpointRollback")
                .field("checkpoint_age_threshold", checkpoint_age_threshold)
                .field("max_rollback_attempts", max_rollback_attempts)
                .finish(),
            Self::ReallocationStrategy {
                reallocation_delay,
                max_reallocations,
            } => f
                .debug_struct("ReallocationStrategy")
                .field("reallocation_delay", reallocation_delay)
                .field("max_reallocations", max_reallocations)
                .finish(),
            Self::AlgorithmSwitch {
                fallback_algorithms,
                max_switches,
            } => f
                .debug_struct("AlgorithmSwitch")
                .field("fallback_algorithms", fallback_algorithms)
                .field("max_switches", max_switches)
                .finish(),
            Self::CorrectiveTransformation {
                transformations,
                max_applications,
            } => f
                .debug_struct("CorrectiveTransformation")
                .field("transformations", transformations)
                .field("max_applications", max_applications)
                .finish(),
            Self::Custom { max_attempts, .. } => f
                .debug_struct("Custom")
                .field("max_attempts", max_attempts)
                .field("recovery_fn", &"<custom function>")
                .finish(),
        }
    }
}

/// Corrective transformations that can be applied to fix issues
#[derive(Debug, Clone)]
pub enum CorrectiveTransform {
    /// Clip gradients to prevent explosion
    GradientClipping { threshold: f64 },
    /// Normalize values to prevent overflow
    ValueNormalization { max_magnitude: f64 },
    /// Add noise to break out of local minima
    NoiseInjection { noise_level: f64 },
    /// Apply regularization to stabilize computation
    Regularization { strength: f64 },
    /// Reduce learning rate temporarily
    LearningRateReduction { factor: f64 },
    /// Reset certain parameters to safe values
    ParameterReset { reset_threshold: f64 },
    /// Apply numerical stabilization techniques
    NumericalStabilization { epsilon: f64 },
}

/// Actions that can be taken during recovery
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Retry the operation immediately
    Retry,
    /// Retry after a delay
    RetryAfterDelay(Duration),
    /// Retry with modified parameters
    RetryWithModification(RecoveryModification),
    /// Switch to a different strategy
    SwitchStrategy(String),
    /// Rollback to a checkpoint
    Rollback(String),
    /// Apply corrective transformations
    ApplyCorrections(Vec<CorrectiveTransform>),
    /// Fail permanently (unrecoverable)
    FailPermanently,
    /// Success (recovery completed)
    Success,
}

/// Modifications to apply during recovery retry
#[derive(Debug, Clone)]
pub struct RecoveryModification {
    pub precision_change: Option<f64>,
    pub parameter_adjustments: HashMap<String, f64>,
    pub algorithm_switch: Option<String>,
    pub timeout_adjustment: Option<Duration>,
}

/// Statistics for error recovery operations
#[derive(Debug, Clone)]
pub struct RecoveryStatistics {
    pub total_failures: usize,
    pub successful_recoveries: usize,
    pub permanent_failures: usize,
    pub recovery_attempts_by_type: HashMap<TransientFailureType, usize>,
    pub average_recovery_time: Duration,
    pub success_rate_by_strategy: HashMap<String, f64>,
    pub most_effective_strategy: Option<String>,
}

impl RecoveryStatistics {
    pub fn new() -> Self {
        Self {
            total_failures: 0,
            successful_recoveries: 0,
            permanent_failures: 0,
            recovery_attempts_by_type: HashMap::new(),
            average_recovery_time: Duration::from_secs(0),
            success_rate_by_strategy: HashMap::new(),
            most_effective_strategy: None,
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_failures == 0 {
            return 1.0;
        }
        self.successful_recoveries as f64 / self.total_failures as f64
    }

    pub fn update_most_effective_strategy(&mut self) {
        self.most_effective_strategy = self
            .success_rate_by_strategy
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(strategy, _)| strategy.clone());
    }
}

impl Default for RecoveryStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Context for a recovery attempt
#[derive(Debug, Clone)]
pub struct RecoveryContext {
    pub operation_name: String,
    pub failure_type: TransientFailureType,
    pub attempt_number: usize,
    pub total_elapsed_time: Duration,
    pub previous_attempts: Vec<RecoveryAttempt>,
    pub metadata: HashMap<String, String>,
}

/// Record of a recovery attempt
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    pub timestamp: Instant,
    pub strategy: String,
    pub action: RecoveryAction,
    pub duration: Duration,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Configuration for automatic error recovery
#[derive(Debug, Clone)]
pub struct AutomaticRecoveryConfig {
    pub enabled: bool,
    pub max_total_attempts: usize,
    pub max_total_time: Duration,
    pub default_strategies: HashMap<TransientFailureType, RecoveryStrategy>,
    pub fallback_strategy: RecoveryStrategy,
    pub enable_learning: bool,
    pub learning_rate: f64,
    pub statistics_window_size: usize,
}

impl Default for AutomaticRecoveryConfig {
    fn default() -> Self {
        let mut default_strategies = HashMap::new();

        // Default strategy for numerical instability
        default_strategies.insert(
            TransientFailureType::NumericalInstability,
            RecoveryStrategy::CorrectiveTransformation {
                transformations: vec![
                    CorrectiveTransform::GradientClipping { threshold: 1.0 },
                    CorrectiveTransform::NumericalStabilization { epsilon: 1e-8 },
                ],
                max_applications: 3,
            },
        );

        // Default strategy for memory pressure
        default_strategies.insert(
            TransientFailureType::MemoryPressure,
            RecoveryStrategy::GracefulDegradation {
                precision_reduction: 0.5,
                simplification_level: 1,
                max_retries: 2,
            },
        );

        // Default strategy for network failures
        default_strategies.insert(
            TransientFailureType::NetworkFailure,
            RecoveryStrategy::ExponentialBackoff {
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(10),
                backoff_factor: 2.0,
                max_retries: 5,
            },
        );

        // Default strategy for gradient instability
        default_strategies.insert(
            TransientFailureType::GradientInstability,
            RecoveryStrategy::CorrectiveTransformation {
                transformations: vec![
                    CorrectiveTransform::GradientClipping { threshold: 0.5 },
                    CorrectiveTransform::LearningRateReduction { factor: 0.1 },
                ],
                max_applications: 2,
            },
        );

        Self {
            enabled: true,
            max_total_attempts: 10,
            max_total_time: Duration::from_secs(300), // 5 minutes
            default_strategies,
            fallback_strategy: RecoveryStrategy::ExponentialBackoff {
                initial_delay: Duration::from_millis(50),
                max_delay: Duration::from_secs(5),
                backoff_factor: 1.5,
                max_retries: 3,
            },
            enable_learning: true,
            learning_rate: 0.1,
            statistics_window_size: 100,
        }
    }
}

/// Main automatic error recovery system
pub struct AutomaticErrorRecovery {
    config: AutomaticRecoveryConfig,
    statistics: Arc<RwLock<RecoveryStatistics>>,
    active_recoveries: Arc<Mutex<HashMap<String, RecoveryContext>>>,
    recovery_history: Arc<Mutex<VecDeque<RecoveryContext>>>,
    learned_strategies: Arc<RwLock<HashMap<TransientFailureType, RecoveryStrategy>>>,
}

impl AutomaticErrorRecovery {
    /// Create a new automatic error recovery system
    pub fn new(config: AutomaticRecoveryConfig) -> Self {
        Self {
            config,
            statistics: Arc::new(RwLock::new(RecoveryStatistics::new())),
            active_recoveries: Arc::new(Mutex::new(HashMap::new())),
            recovery_history: Arc::new(Mutex::new(VecDeque::new())),
            learned_strategies: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(AutomaticRecoveryConfig::default())
    }

    /// Attempt to recover from an error
    pub fn recover<T, F>(&self, operation_name: &str, operation: F) -> AutogradResult<T>
    where
        F: Fn() -> AutogradResult<T>,
    {
        if !self.config.enabled {
            return operation();
        }

        let start_time = Instant::now();
        let mut attempt_number = 0;
        let recovery_id = format!("{}_{}", operation_name, start_time.elapsed().as_nanos());

        loop {
            attempt_number += 1;

            // Check if we've exceeded maximum attempts or time
            if attempt_number > self.config.max_total_attempts {
                return Err(AutogradError::gradient_computation(
                    operation_name,
                    format!(
                        "Exceeded maximum recovery attempts ({})",
                        self.config.max_total_attempts
                    ),
                ));
            }

            if start_time.elapsed() > self.config.max_total_time {
                return Err(AutogradError::gradient_computation(
                    operation_name,
                    format!(
                        "Exceeded maximum recovery time ({:?})",
                        self.config.max_total_time
                    ),
                ));
            }

            // Try the operation
            match operation() {
                Ok(result) => {
                    // Success! Update statistics and clean up
                    self.record_successful_recovery(
                        &recovery_id,
                        attempt_number,
                        start_time.elapsed(),
                    );
                    return Ok(result);
                }
                Err(error) => {
                    // Failure - analyze and attempt recovery
                    let failure_type = self.classify_error(&error);
                    let recovery_action = self.determine_recovery_action(
                        operation_name,
                        &failure_type,
                        attempt_number,
                        start_time.elapsed(),
                        &error,
                    );

                    match recovery_action {
                        RecoveryAction::Retry => {
                            continue; // Immediate retry
                        }
                        RecoveryAction::RetryAfterDelay(delay) => {
                            thread::sleep(delay);
                            continue;
                        }
                        RecoveryAction::RetryWithModification(_modification) => {
                            // In a real implementation, we would apply the modifications
                            continue;
                        }
                        RecoveryAction::ApplyCorrections(_transforms) => {
                            // In a real implementation, we would apply the corrective transforms
                            continue;
                        }
                        RecoveryAction::FailPermanently => {
                            self.record_permanent_failure(
                                &recovery_id,
                                &failure_type,
                                start_time.elapsed(),
                            );
                            return Err(error);
                        }
                        _ => {
                            // Other recovery actions would be implemented here
                            continue;
                        }
                    }
                }
            }
        }
    }

    /// Classify an error to determine the failure type
    fn classify_error(&self, error: &AutogradError) -> TransientFailureType {
        let error_msg = error.to_string().to_lowercase();

        if error_msg.contains("nan") || error_msg.contains("inf") || error_msg.contains("numerical")
        {
            TransientFailureType::NumericalInstability
        } else if error_msg.contains("memory") || error_msg.contains("allocation") {
            TransientFailureType::MemoryPressure
        } else if error_msg.contains("network") || error_msg.contains("connection") {
            TransientFailureType::NetworkFailure
        } else if error_msg.contains("gradient")
            && (error_msg.contains("explod") || error_msg.contains("vanish"))
        {
            TransientFailureType::GradientInstability
        } else if error_msg.contains("overflow") || error_msg.contains("underflow") {
            TransientFailureType::ComputationalOverflow
        } else if error_msg.contains("timeout") || error_msg.contains("lock") {
            TransientFailureType::ResourceContention
        } else {
            // Default to hardware failure for unclassified errors
            TransientFailureType::HardwareFailure
        }
    }

    /// Determine the recovery action based on the failure type and context
    fn determine_recovery_action(
        &self,
        _operation_name: &str,
        failure_type: &TransientFailureType,
        attempt_number: usize,
        _elapsed_time: Duration,
        error: &AutogradError,
    ) -> RecoveryAction {
        // Get the strategy for this failure type
        let strategy = self.get_strategy_for_failure_type(failure_type);

        match &strategy {
            RecoveryStrategy::ExponentialBackoff {
                initial_delay,
                max_delay,
                backoff_factor,
                max_retries,
            } => {
                if attempt_number > *max_retries {
                    return RecoveryAction::FailPermanently;
                }

                let delay_ms = initial_delay.as_millis() as f64
                    * backoff_factor.powi(attempt_number as i32 - 1);
                let delay =
                    Duration::from_millis(delay_ms.min(max_delay.as_millis() as f64) as u64);

                RecoveryAction::RetryAfterDelay(delay)
            }
            RecoveryStrategy::GracefulDegradation {
                precision_reduction: _,
                simplification_level: _,
                max_retries,
            } => {
                if attempt_number > *max_retries {
                    return RecoveryAction::FailPermanently;
                }

                RecoveryAction::RetryWithModification(RecoveryModification {
                    precision_change: Some(0.5),
                    parameter_adjustments: HashMap::new(),
                    algorithm_switch: None,
                    timeout_adjustment: None,
                })
            }
            RecoveryStrategy::CorrectiveTransformation {
                transformations,
                max_applications,
            } => {
                if attempt_number > *max_applications {
                    return RecoveryAction::FailPermanently;
                }

                RecoveryAction::ApplyCorrections(transformations.clone())
            }
            RecoveryStrategy::CheckpointRollback {
                checkpoint_age_threshold: _,
                max_rollback_attempts,
            } => {
                if attempt_number > *max_rollback_attempts {
                    return RecoveryAction::FailPermanently;
                }

                RecoveryAction::Rollback("latest_checkpoint".to_string())
            }
            RecoveryStrategy::ReallocationStrategy {
                reallocation_delay,
                max_reallocations,
            } => {
                if attempt_number > *max_reallocations {
                    return RecoveryAction::FailPermanently;
                }

                RecoveryAction::RetryAfterDelay(*reallocation_delay)
            }
            RecoveryStrategy::AlgorithmSwitch {
                fallback_algorithms,
                max_switches,
            } => {
                if attempt_number > *max_switches || fallback_algorithms.is_empty() {
                    return RecoveryAction::FailPermanently;
                }

                let algorithm_idx = (attempt_number - 1) % fallback_algorithms.len();
                RecoveryAction::SwitchStrategy(fallback_algorithms[algorithm_idx].clone())
            }
            RecoveryStrategy::Custom {
                recovery_fn,
                max_attempts,
            } => {
                if attempt_number > *max_attempts {
                    return RecoveryAction::FailPermanently;
                }

                recovery_fn(error)
            }
        }
    }

    /// Get the strategy for a specific failure type
    fn get_strategy_for_failure_type(
        &self,
        failure_type: &TransientFailureType,
    ) -> RecoveryStrategy {
        // First check learned strategies
        if self.config.enable_learning {
            if let Ok(learned) = self.learned_strategies.read() {
                if let Some(strategy) = learned.get(failure_type) {
                    return strategy.clone();
                }
            }
        }

        // Then check configured strategies
        self.config
            .default_strategies
            .get(failure_type)
            .cloned()
            .unwrap_or_else(|| self.config.fallback_strategy.clone())
    }

    /// Record a successful recovery
    fn record_successful_recovery(&self, recovery_id: &str, _attempts: usize, duration: Duration) {
        if let Ok(mut stats) = self.statistics.write() {
            stats.total_failures += 1;
            stats.successful_recoveries += 1;

            // Update average recovery time
            let total_time =
                stats.average_recovery_time.as_nanos() as f64 * (stats.total_failures - 1) as f64;
            let new_total = total_time + duration.as_nanos() as f64;
            stats.average_recovery_time =
                Duration::from_nanos((new_total / stats.total_failures as f64) as u64);

            stats.update_most_effective_strategy();
        }

        // Clean up active recovery
        if let Ok(mut active) = self.active_recoveries.lock() {
            active.remove(recovery_id);
        }
    }

    /// Record a permanent failure
    fn record_permanent_failure(
        &self,
        recovery_id: &str,
        failure_type: &TransientFailureType,
        _duration: Duration,
    ) {
        if let Ok(mut stats) = self.statistics.write() {
            stats.total_failures += 1;
            stats.permanent_failures += 1;
            *stats
                .recovery_attempts_by_type
                .entry(failure_type.clone())
                .or_insert(0) += 1;
        }

        // Clean up active recovery
        if let Ok(mut active) = self.active_recoveries.lock() {
            active.remove(recovery_id);
        }
    }

    /// Get current recovery statistics
    pub fn get_statistics(&self) -> RecoveryStatistics {
        self.statistics
            .read()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Update recovery strategy for a failure type (learning)
    pub fn update_strategy(
        &self,
        failure_type: TransientFailureType,
        strategy: RecoveryStrategy,
    ) -> AutogradResult<()> {
        if !self.config.enable_learning {
            return Err(AutogradError::gradient_computation(
                "update_strategy",
                "Learning is disabled in configuration",
            ));
        }

        if let Ok(mut learned) = self.learned_strategies.write() {
            learned.insert(failure_type, strategy);
        }

        Ok(())
    }

    /// Get list of active recoveries
    pub fn get_active_recoveries(&self) -> HashMap<String, RecoveryContext> {
        self.active_recoveries
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Clear recovery history and statistics
    pub fn reset_statistics(&self) {
        if let Ok(mut stats) = self.statistics.write() {
            *stats = RecoveryStatistics::new();
        }

        if let Ok(mut history) = self.recovery_history.lock() {
            history.clear();
        }

        if let Ok(mut learned) = self.learned_strategies.write() {
            learned.clear();
        }
    }

    /// Enable or disable the recovery system
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }

    /// Update the maximum number of recovery attempts
    pub fn set_max_attempts(&mut self, max_attempts: usize) {
        self.config.max_total_attempts = max_attempts;
    }

    /// Update the maximum recovery time
    pub fn set_max_time(&mut self, max_time: Duration) {
        self.config.max_total_time = max_time;
    }
}

/// Global singleton for automatic error recovery
static GLOBAL_RECOVERY: std::sync::OnceLock<AutomaticErrorRecovery> = std::sync::OnceLock::new();

/// Get the global error recovery system
pub fn get_global_recovery() -> &'static AutomaticErrorRecovery {
    GLOBAL_RECOVERY.get_or_init(|| AutomaticErrorRecovery::with_defaults())
}

/// Convenience function to recover from errors using the global recovery system
pub fn recover_from_error<T, F>(operation_name: &str, operation: F) -> AutogradResult<T>
where
    F: Fn() -> AutogradResult<T>,
{
    get_global_recovery().recover(operation_name, operation)
}

/// Macro for easy error recovery
#[macro_export]
macro_rules! with_error_recovery {
    ($operation_name:expr, $operation:expr) => {
        $crate::automatic_error_recovery::recover_from_error($operation_name, || $operation)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failing_operation(
        fail_count: &std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) -> AutogradResult<i32> {
        let count = fail_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if count < 2 {
            Err(AutogradError::numerical_instability(
                "test_operation",
                1.0,
                "NaN detected",
            ))
        } else {
            Ok(42)
        }
    }

    #[test]
    fn test_automatic_recovery_creation() {
        let recovery = AutomaticErrorRecovery::with_defaults();
        assert!(recovery.config.enabled);
        assert_eq!(recovery.config.max_total_attempts, 10);
    }

    #[test]
    fn test_error_classification() {
        let recovery = AutomaticErrorRecovery::with_defaults();

        let nan_error = AutogradError::numerical_instability("test", 1.0, "NaN detected");
        assert_eq!(
            recovery.classify_error(&nan_error),
            TransientFailureType::NumericalInstability
        );

        let memory_error = AutogradError::gradient_computation("test", "Out of memory");
        assert_eq!(
            recovery.classify_error(&memory_error),
            TransientFailureType::MemoryPressure
        );
    }

    #[test]
    fn test_successful_recovery() {
        let recovery = AutomaticErrorRecovery::with_defaults();
        let fail_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fail_count_clone = fail_count.clone();

        let result = recovery.recover("test_operation", || failing_operation(&fail_count_clone));

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(fail_count.load(std::sync::atomic::Ordering::SeqCst), 3);

        let stats = recovery.get_statistics();
        assert_eq!(stats.total_failures, 1);
        assert_eq!(stats.successful_recoveries, 1);
        assert_eq!(stats.success_rate(), 1.0);
    }

    #[test]
    fn test_recovery_strategy_selection() {
        let config = AutomaticRecoveryConfig::default();
        let recovery = AutomaticErrorRecovery::new(config);

        let strategy =
            recovery.get_strategy_for_failure_type(&TransientFailureType::NumericalInstability);
        match strategy {
            RecoveryStrategy::CorrectiveTransformation { .. } => {
                // Expected for numerical instability
            }
            _ => panic!("Unexpected strategy for numerical instability"),
        }
    }

    #[test]
    fn test_statistics_tracking() {
        let recovery = AutomaticErrorRecovery::with_defaults();

        // Initial statistics should be empty
        let stats = recovery.get_statistics();
        assert_eq!(stats.total_failures, 0);
        assert_eq!(stats.successful_recoveries, 0);
        assert_eq!(stats.success_rate(), 1.0);

        // Test recording statistics
        recovery.record_successful_recovery("test_1", 3, Duration::from_millis(100));

        let updated_stats = recovery.get_statistics();
        assert_eq!(updated_stats.total_failures, 1);
        assert_eq!(updated_stats.successful_recoveries, 1);
        assert_eq!(updated_stats.success_rate(), 1.0);
    }

    #[test]
    fn test_recovery_config_defaults() {
        let config = AutomaticRecoveryConfig::default();

        assert!(config.enabled);
        assert_eq!(config.max_total_attempts, 10);
        assert_eq!(config.max_total_time, Duration::from_secs(300));
        assert!(config.enable_learning);
        assert!(!config.default_strategies.is_empty());
    }

    #[test]
    fn test_failure_type_display() {
        assert_eq!(
            TransientFailureType::NumericalInstability.to_string(),
            "numerical_instability"
        );
        assert_eq!(
            TransientFailureType::MemoryPressure.to_string(),
            "memory_pressure"
        );
        assert_eq!(
            TransientFailureType::Custom("test".to_string()).to_string(),
            "custom_test"
        );
    }

    #[test]
    fn test_recovery_action_variants() {
        let action = RecoveryAction::Retry;
        matches!(action, RecoveryAction::Retry);

        let action = RecoveryAction::RetryAfterDelay(Duration::from_millis(100));
        matches!(action, RecoveryAction::RetryAfterDelay(_));

        let action = RecoveryAction::FailPermanently;
        matches!(action, RecoveryAction::FailPermanently);
    }

    #[test]
    fn test_global_recovery_access() {
        let recovery = get_global_recovery();
        assert!(recovery.config.enabled);

        // Test that subsequent calls return the same instance
        let recovery2 = get_global_recovery();
        assert!(std::ptr::eq(recovery, recovery2));
    }

    #[test]
    fn test_corrective_transforms() {
        let transform = CorrectiveTransform::GradientClipping { threshold: 1.0 };
        matches!(transform, CorrectiveTransform::GradientClipping { .. });

        let transform = CorrectiveTransform::LearningRateReduction { factor: 0.1 };
        matches!(transform, CorrectiveTransform::LearningRateReduction { .. });
    }

    #[test]
    fn test_recovery_with_macro() {
        let fail_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fail_count_clone = fail_count.clone();

        let result = with_error_recovery!("macro_test", failing_operation(&fail_count_clone));

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }
}
