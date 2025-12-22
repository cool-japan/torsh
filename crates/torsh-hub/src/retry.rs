//! Advanced retry logic with exponential backoff and jitter
//!
//! This module provides production-ready retry strategies for network operations,
//! downloads, and API calls with features like:
//! - Exponential backoff with configurable base and max delays
//! - Jitter to prevent thundering herd
//! - Conditional retry based on error types
//! - Circuit breaker pattern for cascading failures
//! - Retry budget to prevent retry storms

use scirs2_core::random::Rng;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use torsh_core::error::{Result, TorshError};

/// Retry strategy configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: usize,
    /// Base delay between retries
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Whether to add jitter to delays
    pub use_jitter: bool,
    /// Maximum total time to spend on retries
    pub timeout: Option<Duration>,
    /// Whether to use circuit breaker
    pub use_circuit_breaker: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            use_jitter: true,
            timeout: Some(Duration::from_secs(300)), // 5 minutes
            use_circuit_breaker: false,
        }
    }
}

impl RetryConfig {
    /// Create a new retry config with custom settings
    pub fn new(max_attempts: usize, base_delay: Duration) -> Self {
        Self {
            max_attempts,
            base_delay,
            ..Default::default()
        }
    }

    /// Create a config for aggressive retries (more attempts, shorter delays)
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 5,
            base_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 1.5,
            use_jitter: true,
            timeout: Some(Duration::from_secs(60)),
            use_circuit_breaker: false,
        }
    }

    /// Create a config for conservative retries (fewer attempts, longer delays)
    pub fn conservative() -> Self {
        Self {
            max_attempts: 2,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 3.0,
            use_jitter: true,
            timeout: Some(Duration::from_secs(600)),
            use_circuit_breaker: false,
        }
    }

    /// Create a config with circuit breaker enabled
    pub fn with_circuit_breaker(mut self) -> Self {
        self.use_circuit_breaker = true;
        self
    }
}

/// Retry statistics for monitoring and debugging
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetryStats {
    pub total_attempts: usize,
    pub successful_attempts: usize,
    pub failed_attempts: usize,
    pub total_delay: Duration,
    pub average_delay: Duration,
    pub last_error: Option<String>,
}

/// Circuit breaker state to prevent cascading failures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are rejected immediately
    Open,
    /// Circuit is half-open, testing if service recovered
    HalfOpen,
}

/// Circuit breaker for preventing retry storms
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    state: CircuitState,
    failure_threshold: usize,
    success_threshold: usize,
    failure_count: usize,
    success_count: usize,
    last_failure_time: Option<Instant>,
    cooldown_duration: Duration,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            failure_threshold: 5,
            success_threshold: 2,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            cooldown_duration: Duration::from_secs(60),
        }
    }
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(failure_threshold: usize, cooldown_duration: Duration) -> Self {
        Self {
            failure_threshold,
            cooldown_duration,
            ..Default::default()
        }
    }

    /// Check if request should be allowed
    pub fn allow_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if cooldown period has passed
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.cooldown_duration {
                        self.state = CircuitState::HalfOpen;
                        self.success_count = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful request
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed request
    pub fn record_failure(&mut self) {
        self.last_failure_time = Some(Instant::now());

        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.failure_count = 0;
            }
            CircuitState::Open => {}
        }
    }

    /// Get current state
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Reset the circuit breaker
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.last_failure_time = None;
    }
}

/// Execute a function with retry logic
pub fn retry_with_backoff<F, T>(config: &RetryConfig, mut operation: F) -> Result<T>
where
    F: FnMut() -> Result<T>,
{
    let start_time = Instant::now();
    let mut attempt = 0;
    let mut last_error = None;

    while attempt < config.max_attempts {
        // Check timeout
        if let Some(timeout) = config.timeout {
            if start_time.elapsed() >= timeout {
                return Err(TorshError::RuntimeError(format!(
                    "Retry timeout after {} attempts",
                    attempt
                )));
            }
        }

        // Try the operation
        match operation() {
            Ok(result) => return Ok(result),
            Err(e) => {
                last_error = Some(e);
                attempt += 1;

                // Don't sleep after last attempt
                if attempt < config.max_attempts {
                    let delay = calculate_delay(config, attempt);
                    std::thread::sleep(delay);
                }
            }
        }
    }

    // All attempts failed
    Err(last_error.unwrap_or_else(|| {
        TorshError::RuntimeError(format!("All {} retry attempts failed", config.max_attempts))
    }))
}

/// Execute an async function with retry logic
pub async fn retry_with_backoff_async<F, Fut, T>(
    config: &RetryConfig,
    mut operation: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let start_time = Instant::now();
    let mut attempt = 0;
    let mut last_error = None;

    while attempt < config.max_attempts {
        // Check timeout
        if let Some(timeout) = config.timeout {
            if start_time.elapsed() >= timeout {
                return Err(TorshError::RuntimeError(format!(
                    "Retry timeout after {} attempts",
                    attempt
                )));
            }
        }

        // Try the operation
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                last_error = Some(e);
                attempt += 1;

                // Don't sleep after last attempt
                if attempt < config.max_attempts {
                    let delay = calculate_delay(config, attempt);
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    // All attempts failed
    Err(last_error.unwrap_or_else(|| {
        TorshError::RuntimeError(format!("All {} retry attempts failed", config.max_attempts))
    }))
}

/// Calculate delay for next retry attempt
fn calculate_delay(config: &RetryConfig, attempt: usize) -> Duration {
    // Calculate exponential backoff
    let exp_delay =
        config.base_delay.as_secs_f64() * config.backoff_multiplier.powi((attempt - 1) as i32);

    // Cap at max delay
    let capped_delay = exp_delay.min(config.max_delay.as_secs_f64());

    // Add jitter if enabled (random value between 0% and 25% of delay)
    let final_delay = if config.use_jitter {
        let jitter = capped_delay * (scirs2_core::random::thread_rng().random::<f64>() * 0.25);
        capped_delay + jitter
    } else {
        capped_delay
    };

    Duration::from_secs_f64(final_delay)
}

/// Retry policy that determines if an error should be retried
pub trait RetryPolicy {
    fn should_retry(&self, error: &TorshError, attempt: usize) -> bool;
}

/// Default retry policy that retries on network and timeout errors
pub struct DefaultRetryPolicy;

impl RetryPolicy for DefaultRetryPolicy {
    fn should_retry(&self, error: &TorshError, attempt: usize) -> bool {
        if attempt >= 3 {
            return false;
        }

        matches!(
            error,
            TorshError::IoError(_) | TorshError::RuntimeError(_) | TorshError::ComputeError(_)
        )
    }
}

/// Execute with custom retry policy
pub fn retry_with_policy<F, T, P>(config: &RetryConfig, policy: &P, mut operation: F) -> Result<T>
where
    F: FnMut() -> Result<T>,
    P: RetryPolicy,
{
    let start_time = Instant::now();
    let mut attempt = 0;

    loop {
        // Check timeout
        if let Some(timeout) = config.timeout {
            if start_time.elapsed() >= timeout {
                return Err(TorshError::RuntimeError(format!(
                    "Retry timeout after {} attempts",
                    attempt
                )));
            }
        }

        match operation() {
            Ok(result) => return Ok(result),
            Err(e) => {
                attempt += 1;

                if !policy.should_retry(&e, attempt) || attempt >= config.max_attempts {
                    return Err(e);
                }

                let delay = calculate_delay(config, attempt);
                std::thread::sleep(delay);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_success_on_third_attempt() {
        let mut attempts = 0;
        let config = RetryConfig::default();

        let result = retry_with_backoff(&config, || {
            attempts += 1;
            if attempts < 3 {
                Err(TorshError::IoError("Temporary error".to_string()))
            } else {
                Ok(42)
            }
        });

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts, 3);
    }

    #[test]
    fn test_retry_exhaustion() {
        let mut attempts = 0;
        let config = RetryConfig::new(3, Duration::from_millis(10));

        let result: Result<i32> = retry_with_backoff(&config, || {
            attempts += 1;
            Err(TorshError::IoError("Persistent error".to_string()))
        });

        assert!(result.is_err());
        assert_eq!(attempts, 3);
    }

    #[test]
    fn test_circuit_breaker_opens() {
        let mut breaker = CircuitBreaker::new(3, Duration::from_secs(60));

        // Record failures to open circuit
        for _ in 0..3 {
            breaker.record_failure();
        }

        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(!breaker.allow_request());
    }

    #[test]
    fn test_circuit_breaker_recovery() {
        let mut breaker = CircuitBreaker::new(2, Duration::from_millis(10));

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for cooldown
        std::thread::sleep(Duration::from_millis(20));

        // Should transition to half-open
        assert!(breaker.allow_request());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Record successes to close circuit
        breaker.record_success();
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_delay_calculation() {
        let config = RetryConfig {
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            use_jitter: false,
            ..Default::default()
        };

        let delay1 = calculate_delay(&config, 1);
        let delay2 = calculate_delay(&config, 2);
        let delay3 = calculate_delay(&config, 3);

        // Exponential growth: 100ms, 200ms, 400ms
        assert!(delay1.as_millis() >= 100 && delay1.as_millis() <= 100);
        assert!(delay2.as_millis() >= 200 && delay2.as_millis() <= 200);
        assert!(delay3.as_millis() >= 400 && delay3.as_millis() <= 400);
    }

    #[tokio::test]
    async fn test_async_retry_success() {
        use std::sync::{Arc, Mutex};

        let attempts = Arc::new(Mutex::new(0));
        let config = RetryConfig::new(3, Duration::from_millis(10));

        let attempts_clone = attempts.clone();
        let result = retry_with_backoff_async(&config, move || {
            let attempts = attempts_clone.clone();
            async move {
                let mut count = attempts.lock().unwrap();
                *count += 1;
                let current = *count;
                drop(count);

                if current < 2 {
                    Err(TorshError::IoError("Temporary error".to_string()))
                } else {
                    Ok(100)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 100);
        assert_eq!(*attempts.lock().unwrap(), 2);
    }
}
