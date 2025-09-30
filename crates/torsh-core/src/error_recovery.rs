//! Error recovery mechanisms for graceful degradation

use crate::error::{Result, TorshError};
use std::fmt;

/// Strategy for recovering from errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Fail immediately (no recovery)
    FailFast,
    /// Retry the operation with the same parameters
    Retry {
        max_attempts: usize,
        delay_ms: Option<u64>,
    },
    /// Use a fallback implementation
    Fallback,
    /// Skip the operation and continue
    Skip,
    /// Use default/safe values
    UseDefault,
    /// Custom recovery strategy
    Custom(String),
}

/// Recovery context containing information about the failed operation
#[derive(Debug, Clone)]
pub struct RecoveryContext {
    /// The original error that triggered recovery
    pub error: TorshError,
    /// Operation that failed
    pub operation: String,
    /// Number of recovery attempts made so far
    pub attempt_count: usize,
    /// Maximum number of attempts allowed
    pub max_attempts: usize,
    /// Additional context information
    pub context: std::collections::HashMap<String, String>,
}

/// Result of a recovery attempt
#[derive(Debug)]
pub enum RecoveryResult<T> {
    /// Recovery succeeded with a result
    Success(T),
    /// Recovery failed, but operation can continue with degraded functionality
    Degraded(T, String),
    /// Recovery failed, operation should be retried
    Retry,
    /// Recovery failed completely
    Failed(TorshError),
}

/// Trait for implementing error recovery mechanisms
pub trait ErrorRecovery<T> {
    /// Attempt to recover from an error
    fn recover(&self, context: &RecoveryContext) -> RecoveryResult<T>;

    /// Check if recovery is possible for the given error
    fn can_recover(&self, error: &TorshError) -> bool;

    /// Get the recovery strategy used by this implementation
    fn strategy(&self) -> &RecoveryStrategy;
}

/// Basic recovery implementation that uses fallback strategies
#[derive(Debug)]
pub struct BasicRecovery<T> {
    strategy: RecoveryStrategy,
    fallback_value: Option<T>,
}

impl<T: Clone> BasicRecovery<T> {
    /// Create a new basic recovery with the given strategy
    pub fn new(strategy: RecoveryStrategy) -> Self {
        Self {
            strategy,
            fallback_value: None,
        }
    }

    /// Create a recovery that uses a default value
    pub fn with_default(default_value: T) -> Self {
        Self {
            strategy: RecoveryStrategy::UseDefault,
            fallback_value: Some(default_value),
        }
    }

    /// Create a retry recovery with specified parameters
    pub fn retry(max_attempts: usize, delay_ms: Option<u64>) -> Self {
        Self {
            strategy: RecoveryStrategy::Retry {
                max_attempts,
                delay_ms,
            },
            fallback_value: None,
        }
    }
}

impl<T: Clone + fmt::Debug> ErrorRecovery<T> for BasicRecovery<T> {
    fn recover(&self, context: &RecoveryContext) -> RecoveryResult<T> {
        match &self.strategy {
            RecoveryStrategy::FailFast => RecoveryResult::Failed(context.error.clone()),
            RecoveryStrategy::Retry {
                max_attempts,
                delay_ms,
            } => {
                if context.attempt_count < *max_attempts {
                    if let Some(delay) = delay_ms {
                        std::thread::sleep(std::time::Duration::from_millis(*delay));
                    }
                    RecoveryResult::Retry
                } else {
                    RecoveryResult::Failed(TorshError::RuntimeError(format!(
                        "Failed after {max_attempts} retry attempts"
                    )))
                }
            }
            RecoveryStrategy::UseDefault => {
                if let Some(ref default_val) = self.fallback_value {
                    RecoveryResult::Degraded(
                        default_val.clone(),
                        "Using default value due to error".to_string(),
                    )
                } else {
                    RecoveryResult::Failed(TorshError::RuntimeError(
                        "No default value available for recovery".to_string(),
                    ))
                }
            }
            RecoveryStrategy::Skip => RecoveryResult::Failed(TorshError::RuntimeError(
                "Skip recovery not supported for this type".to_string(),
            )),
            RecoveryStrategy::Fallback => RecoveryResult::Failed(TorshError::RuntimeError(
                "Fallback recovery not implemented".to_string(),
            )),
            RecoveryStrategy::Custom(name) => RecoveryResult::Failed(TorshError::RuntimeError(
                format!("Custom recovery '{name}' not implemented"),
            )),
        }
    }

    fn can_recover(&self, _error: &TorshError) -> bool {
        match self.strategy {
            RecoveryStrategy::FailFast => false,
            RecoveryStrategy::UseDefault => self.fallback_value.is_some(),
            _ => true,
        }
    }

    fn strategy(&self) -> &RecoveryStrategy {
        &self.strategy
    }
}

/// Recovery manager that coordinates multiple recovery strategies
pub struct RecoveryManager<T> {
    strategies: Vec<Box<dyn ErrorRecovery<T>>>,
    default_strategy: RecoveryStrategy,
}

impl<T> fmt::Debug for RecoveryManager<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecoveryManager")
            .field("strategy_count", &self.strategies.len())
            .field("default_strategy", &self.default_strategy)
            .finish()
    }
}

impl<T: Clone + fmt::Debug + 'static> RecoveryManager<T> {
    /// Create a new recovery manager
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            default_strategy: RecoveryStrategy::FailFast,
        }
    }

    /// Add a recovery strategy
    pub fn add_strategy(mut self, strategy: Box<dyn ErrorRecovery<T>>) -> Self {
        self.strategies.push(strategy);
        self
    }

    /// Set the default recovery strategy
    pub fn with_default_strategy(mut self, strategy: RecoveryStrategy) -> Self {
        self.default_strategy = strategy;
        self
    }

    /// Attempt recovery using available strategies
    pub fn attempt_recovery(&self, mut context: RecoveryContext) -> RecoveryResult<T> {
        // Try each strategy in order
        for strategy in &self.strategies {
            if strategy.can_recover(&context.error) {
                match strategy.recover(&context) {
                    RecoveryResult::Success(result) => return RecoveryResult::Success(result),
                    RecoveryResult::Degraded(result, msg) => {
                        return RecoveryResult::Degraded(result, msg)
                    }
                    RecoveryResult::Retry => {
                        context.attempt_count += 1;
                        if context.attempt_count < context.max_attempts {
                            return RecoveryResult::Retry;
                        }
                    }
                    RecoveryResult::Failed(_) => continue,
                }
            }
        }

        // If no strategy worked, use default strategy
        let default_recovery = BasicRecovery::new(self.default_strategy.clone());
        default_recovery.recover(&context)
    }
}

impl<T: Clone + fmt::Debug + 'static> Default for RecoveryManager<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper for operations that can recover from errors
#[derive(Debug)]
pub struct RecoverableOperation<T> {
    recovery_manager: RecoveryManager<T>,
    operation_name: String,
    max_attempts: usize,
}

impl<T: Clone + fmt::Debug + 'static> RecoverableOperation<T> {
    /// Create a new recoverable operation
    pub fn new(operation_name: impl Into<String>) -> Self {
        Self {
            recovery_manager: RecoveryManager::new(),
            operation_name: operation_name.into(),
            max_attempts: 3,
        }
    }

    /// Set the recovery manager
    pub fn with_recovery_manager(mut self, manager: RecoveryManager<T>) -> Self {
        self.recovery_manager = manager;
        self
    }

    /// Set maximum retry attempts
    pub fn with_max_attempts(mut self, attempts: usize) -> Self {
        self.max_attempts = attempts;
        self
    }

    /// Execute an operation with recovery
    pub fn execute<F>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        let mut attempt_count = 0;

        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let context = RecoveryContext {
                        error: error.clone(),
                        operation: self.operation_name.clone(),
                        attempt_count,
                        max_attempts: self.max_attempts,
                        context: std::collections::HashMap::new(),
                    };

                    match self.recovery_manager.attempt_recovery(context) {
                        RecoveryResult::Success(result) => return Ok(result),
                        RecoveryResult::Degraded(result, _msg) => {
                            // Could log the degradation message here
                            return Ok(result);
                        }
                        RecoveryResult::Retry => {
                            attempt_count += 1;
                            if attempt_count >= self.max_attempts {
                                return Err(TorshError::RuntimeError(format!(
                                    "Operation '{}' failed after {} attempts",
                                    self.operation_name, self.max_attempts
                                )));
                            }
                            continue;
                        }
                        RecoveryResult::Failed(recovery_error) => {
                            return Err(TorshError::wrap_with_location(
                                recovery_error,
                                format!("Recovery failed for operation '{}'", self.operation_name),
                            ));
                        }
                    }
                }
            }
        }
    }

    /// Execute an operation with recovery and context
    pub fn execute_with_context<F>(
        &self,
        mut operation: F,
        context: std::collections::HashMap<String, String>,
    ) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        let mut attempt_count = 0;

        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let recovery_context = RecoveryContext {
                        error: error.clone(),
                        operation: self.operation_name.clone(),
                        attempt_count,
                        max_attempts: self.max_attempts,
                        context: context.clone(),
                    };

                    match self.recovery_manager.attempt_recovery(recovery_context) {
                        RecoveryResult::Success(result) => return Ok(result),
                        RecoveryResult::Degraded(result, _msg) => return Ok(result),
                        RecoveryResult::Retry => {
                            attempt_count += 1;
                            if attempt_count >= self.max_attempts {
                                return Err(error);
                            }
                            continue;
                        }
                        RecoveryResult::Failed(recovery_error) => return Err(recovery_error),
                    }
                }
            }
        }
    }
}

/// Convenience function to create a simple retry operation
pub fn with_retry<T, F>(operation: F, max_attempts: usize) -> Result<T>
where
    F: FnMut() -> Result<T>,
    T: Clone + fmt::Debug + 'static,
{
    let recovery_manager = RecoveryManager::new()
        .add_strategy(Box::new(BasicRecovery::retry(max_attempts, Some(100))));

    let recoverable = RecoverableOperation::new("retry_operation")
        .with_recovery_manager(recovery_manager)
        .with_max_attempts(max_attempts);

    recoverable.execute(operation)
}

/// Convenience function to create an operation with a fallback value
pub fn with_fallback<T, F>(operation: F, fallback: T) -> Result<T>
where
    F: FnMut() -> Result<T>,
    T: Clone + fmt::Debug + 'static,
{
    let recovery_manager =
        RecoveryManager::new().add_strategy(Box::new(BasicRecovery::with_default(fallback)));

    let recoverable =
        RecoverableOperation::new("fallback_operation").with_recovery_manager(recovery_manager);

    recoverable.execute(operation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_recovery_retry() {
        let recovery = BasicRecovery::<i32>::retry(3, Some(1));
        let context = RecoveryContext {
            error: TorshError::RuntimeError("Test error".to_string()),
            operation: "test_op".to_string(),
            attempt_count: 1,
            max_attempts: 3,
            context: std::collections::HashMap::new(),
        };

        match recovery.recover(&context) {
            RecoveryResult::Retry => {} // Expected retry result
            _ => panic!("Expected retry result"),
        }
    }

    #[test]
    fn test_basic_recovery_default() {
        let recovery = BasicRecovery::with_default(42);
        let context = RecoveryContext {
            error: TorshError::RuntimeError("Test error".to_string()),
            operation: "test_op".to_string(),
            attempt_count: 1,
            max_attempts: 3,
            context: std::collections::HashMap::new(),
        };

        match recovery.recover(&context) {
            RecoveryResult::Degraded(value, _) => assert_eq!(value, 42),
            _ => panic!("Expected degraded result with default value"),
        }
    }

    #[test]
    fn test_recoverable_operation_success() {
        let recoverable = RecoverableOperation::new("test_operation");
        let result = recoverable.execute(|| Ok(42));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_with_retry_convenience() {
        let mut counter = 0;
        let result = with_retry(
            || {
                counter += 1;
                if counter < 3 {
                    Err(TorshError::RuntimeError("Not ready".to_string()))
                } else {
                    Ok(42)
                }
            },
            5,
        );

        assert_eq!(result.unwrap(), 42);
        assert_eq!(counter, 3);
    }

    #[test]
    fn test_with_fallback_convenience() {
        let result = with_fallback(
            || Err(TorshError::RuntimeError("Operation failed".to_string())),
            100,
        );

        assert_eq!(result.unwrap(), 100);
    }

    #[test]
    fn test_recovery_manager() {
        let manager = RecoveryManager::new()
            .add_strategy(Box::new(BasicRecovery::retry(2, Some(1))))
            .add_strategy(Box::new(BasicRecovery::with_default(999)));

        let context = RecoveryContext {
            error: TorshError::RuntimeError("Test error".to_string()),
            operation: "test_op".to_string(),
            attempt_count: 3, // Exceed retry limit
            max_attempts: 3,
            context: std::collections::HashMap::new(),
        };

        match manager.attempt_recovery(context) {
            RecoveryResult::Degraded(value, _) => assert_eq!(value, 999),
            _ => panic!("Expected degraded result with default value"),
        }
    }
}
