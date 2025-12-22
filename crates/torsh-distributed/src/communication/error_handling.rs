//! Error handling utilities for communication operations
//!
//! This module consolidates error handling patterns used across
//! distributed communication modules.

use crate::{TorshDistributedError, TorshResult};
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Wrap a generic error into a communication error with operation context
pub fn wrap_communication_error<T>(
    operation: &str,
    result: std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>,
) -> TorshResult<T> {
    result.map_err(|e| TorshDistributedError::CommunicationError {
        operation: operation.to_string(),
        cause: e.to_string(),
    })
}

/// Retry configuration for communication operations
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial backoff delay
    pub initial_delay: Duration,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Total timeout for all retry attempts
    pub total_timeout: Option<Duration>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            max_delay: Duration::from_secs(5),
            total_timeout: Some(Duration::from_secs(30)),
        }
    }
}

/// Retry an operation with exponential backoff
pub async fn retry_with_backoff<T, F, Fut>(operation: F, config: RetryConfig) -> TorshResult<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = TorshResult<T>>,
{
    let start_time = Instant::now();
    let mut current_delay = config.initial_delay;
    let mut last_error = None;

    for attempt in 0..=config.max_retries {
        // Check total timeout
        if let Some(total_timeout) = config.total_timeout {
            if start_time.elapsed() > total_timeout {
                return Err(TorshDistributedError::OperationTimeout {
                    operation: "retry_with_backoff".to_string(),
                    timeout_secs: total_timeout.as_secs(),
                });
            }
        }

        // Try the operation
        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                last_error = Some(error);

                // If this was the last attempt, don't sleep
                if attempt == config.max_retries {
                    break;
                }

                // Check if error is retryable
                if !is_retryable_error(last_error.as_ref().unwrap()) {
                    break;
                }

                // Sleep with exponential backoff
                sleep(current_delay).await;
                current_delay = std::cmp::min(
                    Duration::from_millis(
                        (current_delay.as_millis() as f64 * config.backoff_multiplier) as u64,
                    ),
                    config.max_delay,
                );
            }
        }
    }

    // Return the last error if all retries failed
    Err(
        last_error.unwrap_or_else(|| TorshDistributedError::CommunicationError {
            operation: "retry_with_backoff".to_string(),
            cause: "Unknown error during retry".to_string(),
        }),
    )
}

/// Synchronous version of retry with backoff
pub fn retry_with_backoff_sync<T, F>(operation: F, config: RetryConfig) -> TorshResult<T>
where
    F: Fn() -> TorshResult<T>,
{
    let start_time = Instant::now();
    let mut current_delay = config.initial_delay;
    let mut last_error = None;

    for attempt in 0..=config.max_retries {
        // Check total timeout
        if let Some(total_timeout) = config.total_timeout {
            if start_time.elapsed() > total_timeout {
                return Err(TorshDistributedError::OperationTimeout {
                    operation: "retry_with_backoff_sync".to_string(),
                    timeout_secs: total_timeout.as_secs(),
                });
            }
        }

        // Try the operation
        match operation() {
            Ok(result) => return Ok(result),
            Err(error) => {
                last_error = Some(error);

                // If this was the last attempt, don't sleep
                if attempt == config.max_retries {
                    break;
                }

                // Check if error is retryable
                if !is_retryable_error(last_error.as_ref().unwrap()) {
                    break;
                }

                // Sleep with exponential backoff
                std::thread::sleep(current_delay);
                current_delay = std::cmp::min(
                    Duration::from_millis(
                        (current_delay.as_millis() as f64 * config.backoff_multiplier) as u64,
                    ),
                    config.max_delay,
                );
            }
        }
    }

    // Return the last error if all retries failed
    Err(
        last_error.unwrap_or_else(|| TorshDistributedError::CommunicationError {
            operation: "retry_with_backoff_sync".to_string(),
            cause: "Unknown error during retry".to_string(),
        }),
    )
}

/// Check if an error is retryable
pub fn is_retryable_error(error: &TorshDistributedError) -> bool {
    match error {
        TorshDistributedError::BackendError { .. } => true,
        TorshDistributedError::CommunicationError { .. } => true,
        TorshDistributedError::InvalidArgument { .. } => false,
        TorshDistributedError::SerializationError(_) => false,
        TorshDistributedError::OperationTimeout { .. } => true,
        TorshDistributedError::IoError(_) => true,
        TorshDistributedError::InternalError(_) => false,
        TorshDistributedError::ProcessFailure { .. } => true,
        TorshDistributedError::CheckpointError { .. } => true,
        TorshDistributedError::BackendNotInitialized => false,
        TorshDistributedError::RankOutOfBounds { .. } => false,
        TorshDistributedError::FeatureNotAvailable { .. } => false,
        TorshDistributedError::ProcessGroupNotFound { .. } => false,
        TorshDistributedError::TensorShapeMismatch { .. } => false,
        TorshDistributedError::MemoryAllocationFailed { .. } => false,
        TorshDistributedError::ConfigurationError { .. } => false,
    }
}

/// Timeout wrapper for async operations
pub async fn with_timeout<T, F, Fut>(
    operation: F,
    timeout: Duration,
    operation_name: &str,
) -> TorshResult<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = TorshResult<T>>,
{
    match tokio::time::timeout(timeout, operation()).await {
        Ok(result) => result,
        Err(_) => Err(TorshDistributedError::OperationTimeout {
            operation: operation_name.to_string(),
            timeout_secs: timeout.as_secs(),
        }),
    }
}

/// Create a standard communication error
pub fn communication_error(
    operation: &str,
    cause: impl std::fmt::Display,
) -> TorshDistributedError {
    TorshDistributedError::CommunicationError {
        operation: operation.to_string(),
        cause: cause.to_string(),
    }
}

/// Create a backend error
pub fn backend_error(
    backend: &str,
    message: impl std::fmt::Display,
) -> torsh_core::error::TorshError {
    TorshDistributedError::BackendError {
        backend: backend.to_string(),
        message: message.to_string(),
    }
    .into()
}

/// Create a timeout error
pub fn timeout_error(operation: &str, timeout_secs: u64) -> torsh_core::error::TorshError {
    TorshDistributedError::OperationTimeout {
        operation: operation.to_string(),
        timeout_secs,
    }
    .into()
}

/// Create a process failure error
pub fn process_failure_error(
    rank: u32,
    operation: &str,
    cause: impl std::fmt::Display,
) -> torsh_core::error::TorshError {
    TorshDistributedError::ProcessFailure {
        rank,
        operation: operation.to_string(),
        cause: cause.to_string(),
    }
    .into()
}

/// Error aggregation utilities for collecting multiple errors
pub struct ErrorCollector {
    errors: Vec<torsh_core::error::TorshError>,
    operation: String,
}

impl ErrorCollector {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            errors: Vec::new(),
            operation: operation.into(),
        }
    }

    pub fn add_error(&mut self, error: torsh_core::error::TorshError) {
        self.errors.push(error);
    }

    pub fn add_result<T>(&mut self, result: TorshResult<T>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                self.errors.push(error.into());
                None
            }
        }
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn into_result<T>(self, success_value: T) -> TorshResult<T> {
        if self.errors.is_empty() {
            Ok(success_value)
        } else {
            // Combine all errors into a single error message
            let combined_message = self
                .errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");

            Err(TorshDistributedError::CommunicationError {
                operation: self.operation,
                cause: combined_message,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_with_backoff_success() {
        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            ..Default::default()
        };

        let result = retry_with_backoff(
            move || {
                let count = attempt_count_clone.fetch_add(1, Ordering::SeqCst);
                async move {
                    if count < 2 {
                        Err(communication_error("test", "simulated failure"))
                    } else {
                        Ok("success")
                    }
                }
            },
            config,
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_with_backoff_failure() {
        let config = RetryConfig {
            max_retries: 2,
            initial_delay: Duration::from_millis(1),
            ..Default::default()
        };

        let result: TorshResult<()> = retry_with_backoff(
            || async {
                Err(TorshDistributedError::communication_error(
                    "test",
                    "persistent failure",
                ))
            },
            config,
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_with_timeout_success() {
        let result = with_timeout(
            || async { Ok("success") },
            Duration::from_millis(100),
            "test_operation",
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[tokio::test]
    async fn test_with_timeout_failure() {
        let result = with_timeout(
            || async {
                sleep(Duration::from_millis(200)).await;
                Ok("should_timeout")
            },
            Duration::from_millis(50),
            "test_operation",
        )
        .await;

        assert!(result.is_err());
    }

    #[test]
    fn test_error_collector() {
        let mut collector = ErrorCollector::new("test_operation");

        // Add some errors
        collector.add_error(communication_error("op1", "error 1").into());
        collector.add_error(communication_error("op2", "error 2").into());

        assert!(collector.has_errors());

        let result: TorshResult<String> = collector.into_result("success".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_error_collector_no_errors() {
        let collector = ErrorCollector::new("test_operation");

        assert!(!collector.has_errors());

        let result: TorshResult<String> = collector.into_result("success".to_string());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }
}
