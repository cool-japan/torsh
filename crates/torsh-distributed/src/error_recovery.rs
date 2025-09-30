//! Error handling and recovery mechanisms for distributed training
//!
//! This module provides robust error handling and recovery capabilities
//! for distributed training operations, including retry logic, circuit
//! breakers, and failure detection.

use crate::{TorshDistributedError, TorshResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

/// Configuration for retry policies
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter factor to avoid thundering herd
    pub jitter_factor: f64,
    /// Whether to use exponential backoff
    pub exponential_backoff: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
            exponential_backoff: true,
        }
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to trip the circuit breaker
    pub failure_threshold: u32,
    /// Success threshold to close the circuit breaker
    pub success_threshold: u32,
    /// Timeout before attempting to half-open the circuit
    pub timeout: Duration,
    /// Time window for failure counting
    pub failure_window: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            failure_window: Duration::from_secs(60),
        }
    }
}

/// Circuit breaker for protecting against cascading failures
#[derive(Debug)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<Mutex<CircuitBreakerState>>,
    failure_count: Arc<AtomicU64>,
    success_count: Arc<AtomicU64>,
    last_failure_time: Arc<Mutex<Option<Instant>>>,
    last_success_time: Arc<Mutex<Option<Instant>>>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(CircuitBreakerState::Closed)),
            failure_count: Arc::new(AtomicU64::new(0)),
            success_count: Arc::new(AtomicU64::new(0)),
            last_failure_time: Arc::new(Mutex::new(None)),
            last_success_time: Arc::new(Mutex::new(None)),
        }
    }

    /// Check if a request should be allowed through
    pub fn allow_request(&self) -> bool {
        let state = self.state.lock().unwrap();
        match *state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                // Check if timeout has passed
                if let Some(last_failure) = *self.last_failure_time.lock().unwrap() {
                    last_failure.elapsed() >= self.config.timeout
                } else {
                    true
                }
            }
            CircuitBreakerState::HalfOpen => true,
        }
    }

    /// Record a successful operation
    pub fn record_success(&self) {
        let mut state = self.state.lock().unwrap();
        self.success_count.fetch_add(1, Ordering::Relaxed);
        *self.last_success_time.lock().unwrap() = Some(Instant::now());

        match *state {
            CircuitBreakerState::HalfOpen => {
                if self.success_count.load(Ordering::Relaxed)
                    >= self.config.success_threshold as u64
                {
                    info!(
                        "Circuit breaker transitioning to CLOSED state after {} successes",
                        self.success_count.load(Ordering::Relaxed)
                    );
                    *state = CircuitBreakerState::Closed;
                    self.failure_count.store(0, Ordering::Relaxed);
                    self.success_count.store(0, Ordering::Relaxed);
                }
            }
            CircuitBreakerState::Open => {
                // Transition to half-open if timeout has passed
                if let Some(last_failure) = *self.last_failure_time.lock().unwrap() {
                    if last_failure.elapsed() >= self.config.timeout {
                        info!("Circuit breaker transitioning to HALF_OPEN state after timeout");
                        *state = CircuitBreakerState::HalfOpen;
                        self.success_count.store(1, Ordering::Relaxed);
                    }
                }
            }
            CircuitBreakerState::Closed => {
                // Reset failure count on success
                if self.failure_count.load(Ordering::Relaxed) > 0 {
                    self.failure_count.store(0, Ordering::Relaxed);
                }
            }
        }
    }

    /// Record a failed operation
    pub fn record_failure(&self) {
        let mut state = self.state.lock().unwrap();
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        *self.last_failure_time.lock().unwrap() = Some(Instant::now());

        match *state {
            CircuitBreakerState::Closed => {
                if self.failure_count.load(Ordering::Relaxed)
                    >= self.config.failure_threshold as u64
                {
                    warn!(
                        "Circuit breaker transitioning to OPEN state after {} failures",
                        self.failure_count.load(Ordering::Relaxed)
                    );
                    *state = CircuitBreakerState::Open;
                    self.success_count.store(0, Ordering::Relaxed);
                }
            }
            CircuitBreakerState::HalfOpen => {
                warn!("Circuit breaker transitioning back to OPEN state due to failure in half-open state");
                *state = CircuitBreakerState::Open;
                self.success_count.store(0, Ordering::Relaxed);
            }
            CircuitBreakerState::Open => {
                // Already open, just update counters
            }
        }
    }

    /// Get current state
    pub fn state(&self) -> CircuitBreakerState {
        self.state.lock().unwrap().clone()
    }

    /// Get failure count
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::Relaxed)
    }

    /// Get success count
    pub fn success_count(&self) -> u64 {
        self.success_count.load(Ordering::Relaxed)
    }
}

/// Retry executor with circuit breaker support
#[derive(Debug)]
pub struct RetryExecutor {
    retry_config: RetryConfig,
    circuit_breaker: Option<CircuitBreaker>,
}

impl RetryExecutor {
    pub fn new(retry_config: RetryConfig) -> Self {
        Self {
            retry_config,
            circuit_breaker: None,
        }
    }

    pub fn with_circuit_breaker(
        retry_config: RetryConfig,
        circuit_breaker_config: CircuitBreakerConfig,
    ) -> Self {
        Self {
            retry_config,
            circuit_breaker: Some(CircuitBreaker::new(circuit_breaker_config)),
        }
    }

    /// Execute an operation with retry logic
    pub async fn execute<F, Fut, T>(&self, operation: F) -> TorshResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = TorshResult<T>>,
    {
        let mut attempt = 0;
        let mut delay = self.retry_config.initial_delay;

        loop {
            // Check circuit breaker
            if let Some(cb) = &self.circuit_breaker {
                if !cb.allow_request() {
                    error!("Circuit breaker is OPEN, rejecting request");
                    return Err(TorshDistributedError::communication_error(
                        "error_recovery",
                        "Circuit breaker is open",
                    )
                    .into());
                }
            }

            attempt += 1;
            debug!(
                "Executing operation, attempt {}/{}",
                attempt, self.retry_config.max_attempts
            );

            match operation().await {
                Ok(result) => {
                    if let Some(cb) = &self.circuit_breaker {
                        cb.record_success();
                    }
                    if attempt > 1 {
                        info!("Operation succeeded after {} attempts", attempt);
                    }
                    return Ok(result);
                }
                Err(err) => {
                    if let Some(cb) = &self.circuit_breaker {
                        cb.record_failure();
                    }

                    if attempt >= self.retry_config.max_attempts {
                        error!("Operation failed after {} attempts: {}", attempt, err);
                        return Err(err);
                    }

                    warn!(
                        "Operation failed on attempt {}, retrying in {:?}: {}",
                        attempt, delay, err
                    );

                    // Check if error is retryable
                    if !Self::is_retryable_error(&err) {
                        error!("Non-retryable error encountered: {}", err);
                        return Err(err);
                    }

                    // Sleep before retry
                    sleep(delay).await;

                    // Update delay for next attempt
                    if self.retry_config.exponential_backoff {
                        delay = Duration::from_millis(std::cmp::min(
                            (delay.as_millis() as f64 * self.retry_config.backoff_multiplier)
                                as u64,
                            self.retry_config.max_delay.as_millis() as u64,
                        ));

                        // Add jitter
                        if self.retry_config.jitter_factor > 0.0 {
                            // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
                            use scirs2_core::random::{Random, Rng};
                            let mut rng = Random::seed(42);
                            let jitter = rng.gen_range(0.0..=self.retry_config.jitter_factor);
                            let jitter_ms = (delay.as_millis() as f64 * jitter) as u64;
                            delay = Duration::from_millis(delay.as_millis() as u64 + jitter_ms);
                        }
                    }
                }
            }
        }
    }

    /// Check if an error is retryable
    fn is_retryable_error(error: &TorshDistributedError) -> bool {
        // Check if it's a distributed error by converting to string and checking pattern
        let error_str = error.to_string();

        if error_str.contains("Communication error") || error_str.contains("Backend error") {
            true
        } else if error_str.contains("Backend not initialized")
            || error_str.contains("Invalid argument")
            || error_str.contains("Rank out of bounds")
            || error_str.contains("Feature not available")
        {
            false
        } else {
            // Default to retryable for unknown errors
            true
        }
    }

    /// Get retry statistics
    pub fn get_stats(&self) -> RetryStats {
        if let Some(cb) = &self.circuit_breaker {
            RetryStats {
                circuit_breaker_state: Some(cb.state()),
                failure_count: cb.failure_count(),
                success_count: cb.success_count(),
            }
        } else {
            RetryStats {
                circuit_breaker_state: None,
                failure_count: 0,
                success_count: 0,
            }
        }
    }
}

/// Statistics for retry operations
#[derive(Debug, Clone)]
pub struct RetryStats {
    pub circuit_breaker_state: Option<CircuitBreakerState>,
    pub failure_count: u64,
    pub success_count: u64,
}

/// Health checker for distributed processes
#[derive(Debug)]
pub struct HealthChecker {
    healthy_ranks: Arc<Mutex<HashMap<u32, Instant>>>,
    unhealthy_ranks: Arc<Mutex<HashMap<u32, Instant>>>,
    health_check_interval: Duration,
    health_timeout: Duration,
}

impl HealthChecker {
    pub fn new(health_check_interval: Duration, health_timeout: Duration) -> Self {
        Self {
            healthy_ranks: Arc::new(Mutex::new(HashMap::new())),
            unhealthy_ranks: Arc::new(Mutex::new(HashMap::new())),
            health_check_interval,
            health_timeout,
        }
    }

    /// Mark a rank as healthy
    pub fn mark_healthy(&self, rank: u32) {
        let now = Instant::now();
        {
            let mut healthy = self.healthy_ranks.lock().unwrap();
            healthy.insert(rank, now);
        }
        {
            let mut unhealthy = self.unhealthy_ranks.lock().unwrap();
            unhealthy.remove(&rank);
        }
        debug!("Rank {} marked as healthy", rank);
    }

    /// Mark a rank as unhealthy
    pub fn mark_unhealthy(&self, rank: u32) {
        let now = Instant::now();
        {
            let mut unhealthy = self.unhealthy_ranks.lock().unwrap();
            unhealthy.insert(rank, now);
        }
        {
            let mut healthy = self.healthy_ranks.lock().unwrap();
            healthy.remove(&rank);
        }
        warn!("Rank {} marked as unhealthy", rank);
    }

    /// Get the health check interval
    pub fn health_check_interval(&self) -> Duration {
        self.health_check_interval
    }

    /// Check if a rank is healthy
    pub fn is_healthy(&self, rank: u32) -> bool {
        let healthy = self.healthy_ranks.lock().unwrap();
        if let Some(last_seen) = healthy.get(&rank) {
            last_seen.elapsed() < self.health_timeout
        } else {
            false
        }
    }

    /// Get all healthy ranks
    pub fn get_healthy_ranks(&self) -> Vec<u32> {
        let healthy = self.healthy_ranks.lock().unwrap();
        let now = Instant::now();
        healthy
            .iter()
            .filter(|(_, &last_seen)| now.duration_since(last_seen) < self.health_timeout)
            .map(|(&rank, _)| rank)
            .collect()
    }

    /// Get all unhealthy ranks
    pub fn get_unhealthy_ranks(&self) -> Vec<u32> {
        let unhealthy = self.unhealthy_ranks.lock().unwrap();
        unhealthy.keys().copied().collect()
    }

    /// Clean up stale entries
    pub fn cleanup_stale_entries(&self) {
        let now = Instant::now();
        {
            let mut healthy = self.healthy_ranks.lock().unwrap();
            healthy.retain(|_, &mut last_seen| {
                now.duration_since(last_seen) < self.health_timeout * 2
            });
        }
        {
            let mut unhealthy = self.unhealthy_ranks.lock().unwrap();
            unhealthy.retain(|_, &mut last_seen| {
                now.duration_since(last_seen) < self.health_timeout * 2
            });
        }
    }
}

/// Failure detector for distributed systems
#[derive(Debug)]
pub struct FailureDetector {
    health_checker: HealthChecker,
    retry_executor: RetryExecutor,
}

impl FailureDetector {
    pub fn new(
        health_check_interval: Duration,
        health_timeout: Duration,
        retry_config: RetryConfig,
        circuit_breaker_config: Option<CircuitBreakerConfig>,
    ) -> Self {
        let retry_executor = if let Some(cb_config) = circuit_breaker_config {
            RetryExecutor::with_circuit_breaker(retry_config, cb_config)
        } else {
            RetryExecutor::new(retry_config)
        };

        Self {
            health_checker: HealthChecker::new(health_check_interval, health_timeout),
            retry_executor,
        }
    }

    /// Execute an operation with failure detection and recovery
    pub async fn execute_with_recovery<F, Fut, T>(
        &self,
        operation: F,
        target_rank: Option<u32>,
    ) -> TorshResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = TorshResult<T>>,
    {
        // Check target rank health if specified
        if let Some(rank) = target_rank {
            if !self.health_checker.is_healthy(rank) {
                warn!("Target rank {} is not healthy, operation may fail", rank);
            }
        }

        // Execute with retry logic
        let result = self.retry_executor.execute(operation).await;

        // Update health status based on result
        if let Some(rank) = target_rank {
            match &result {
                Ok(_) => self.health_checker.mark_healthy(rank),
                Err(_) => self.health_checker.mark_unhealthy(rank),
            }
        }

        result
    }

    /// Get current health status
    pub fn get_health_status(&self) -> HealthStatus {
        HealthStatus {
            healthy_ranks: self.health_checker.get_healthy_ranks(),
            unhealthy_ranks: self.health_checker.get_unhealthy_ranks(),
            retry_stats: self.retry_executor.get_stats(),
        }
    }

    /// Perform periodic cleanup
    pub fn cleanup(&self) {
        self.health_checker.cleanup_stale_entries();
    }
}

/// Health status information
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub healthy_ranks: Vec<u32>,
    pub unhealthy_ranks: Vec<u32>,
    pub retry_stats: RetryStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_retry_executor() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(10),
            ..Default::default()
        };
        let executor = RetryExecutor::new(config);

        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let result = executor
            .execute(|| {
                let count = call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                async move {
                    if count < 3 {
                        Err(TorshDistributedError::communication_error(
                            "test_operation",
                            "Temporary failure",
                        )
                        .into())
                    } else {
                        Ok("success")
                    }
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Initially closed
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert!(cb.allow_request());

        // Record failures to trip the breaker
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);

        // Should not allow requests when open
        assert!(!cb.allow_request());

        // Wait for timeout
        sleep(Duration::from_millis(110)).await;
        assert!(cb.allow_request()); // Should transition to half-open

        // Record success to close
        cb.record_success();
        cb.record_success();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
    }

    #[test]
    fn test_health_checker() {
        let checker = HealthChecker::new(Duration::from_millis(100), Duration::from_millis(200));

        // Mark rank as healthy
        checker.mark_healthy(0);
        assert!(checker.is_healthy(0));
        assert_eq!(checker.get_healthy_ranks(), vec![0]);

        // Mark rank as unhealthy
        checker.mark_unhealthy(1);
        assert!(!checker.is_healthy(1));
        assert_eq!(checker.get_unhealthy_ranks(), vec![1]);
    }
}
