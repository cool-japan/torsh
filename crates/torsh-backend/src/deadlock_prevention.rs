//! Deadlock prevention utilities for backend operations
//!
//! This module requires std for proper synchronization primitives.

#[cfg(feature = "std")]
use std::sync::{Mutex, MutexGuard, TryLockError};
#[cfg(feature = "std")]
use std::time::{Duration, Instant};
use torsh_core::error::TorshError;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Result type for lock operations
pub type LockResult<T> = Result<T, TorshError>;

#[cfg(feature = "std")]
/// Lock timeout configuration
#[derive(Debug, Clone)]
pub struct LockTimeout {
    /// Maximum time to wait for a lock
    pub max_wait: Duration,

    /// Time between retry attempts
    pub retry_interval: Duration,

    /// Maximum number of retry attempts
    pub max_retries: usize,
}

impl Default for LockTimeout {
    fn default() -> Self {
        Self {
            max_wait: Duration::from_millis(5000),     // 5 seconds
            retry_interval: Duration::from_millis(10), // 10ms
            max_retries: 500,                          // 5 seconds / 10ms = 500 attempts
        }
    }
}

/// Lock utilities with deadlock prevention
pub struct SafeLock;

impl SafeLock {
    /// Acquire a mutex lock with timeout and retry mechanism
    pub fn acquire_with_timeout<'a, T>(
        mutex: &'a Mutex<T>,
        timeout: Option<LockTimeout>,
        operation_name: &str,
    ) -> LockResult<MutexGuard<'a, T>> {
        let timeout = timeout.unwrap_or_default();
        let start_time = Instant::now();
        let mut attempts = 0;

        loop {
            match mutex.try_lock() {
                Ok(guard) => return Ok(guard),
                Err(TryLockError::WouldBlock) => {
                    attempts += 1;

                    // Check if we've exceeded time or attempt limits
                    let elapsed = start_time.elapsed();
                    if elapsed >= timeout.max_wait || attempts >= timeout.max_retries {
                        return Err(TorshError::BackendError(format!(
                            "Failed to acquire lock for '{}' after {} attempts ({:?}). Possible deadlock detected.",
                            operation_name, attempts, elapsed
                        )));
                    }

                    // Exponential backoff with jitter to reduce lock contention
                    let backoff = timeout.retry_interval * (1 << (attempts.min(6))); // Cap exponential growth
                    let jitter = Duration::from_millis((attempts % 5) as u64); // Add small jitter
                    let sleep_duration = (backoff + jitter).min(Duration::from_millis(100)); // Cap at 100ms

                    // Use standard thread sleep for blocking operation
                    std::thread::sleep(sleep_duration);
                }
                Err(TryLockError::Poisoned(err)) => {
                    return Err(TorshError::BackendError(format!(
                        "Lock for '{}' is poisoned: {}",
                        operation_name, err
                    )));
                }
            }
        }
    }

    /// Acquire a mutex lock with a simple timeout (blocking version)
    pub fn acquire_blocking<'a, T>(
        mutex: &'a Mutex<T>,
        timeout: Option<LockTimeout>,
        operation_name: &str,
    ) -> LockResult<MutexGuard<'a, T>> {
        let timeout = timeout.unwrap_or_default();
        let start_time = Instant::now();
        let mut attempts = 0;

        loop {
            match mutex.try_lock() {
                Ok(guard) => return Ok(guard),
                Err(TryLockError::WouldBlock) => {
                    attempts += 1;

                    let elapsed = start_time.elapsed();
                    if elapsed >= timeout.max_wait || attempts >= timeout.max_retries {
                        return Err(TorshError::BackendError(format!(
                            "Failed to acquire lock for '{}' after {} attempts ({:?}). Possible deadlock detected.",
                            operation_name, attempts, elapsed
                        )));
                    }

                    // Simple linear backoff for blocking version
                    let sleep_duration = timeout.retry_interval.min(Duration::from_millis(50));
                    std::thread::sleep(sleep_duration);
                }
                Err(TryLockError::Poisoned(err)) => {
                    return Err(TorshError::BackendError(format!(
                        "Lock for '{}' is poisoned: {}",
                        operation_name, err
                    )));
                }
            }
        }
    }

    /// Try to acquire a lock without blocking
    pub fn try_acquire<'a, T>(
        mutex: &'a Mutex<T>,
        operation_name: &str,
    ) -> LockResult<Option<MutexGuard<'a, T>>> {
        match mutex.try_lock() {
            Ok(guard) => Ok(Some(guard)),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Poisoned(err)) => Err(TorshError::BackendError(format!(
                "Lock for '{}' is poisoned: {}",
                operation_name, err
            ))),
        }
    }
}

/// Lock ordering mechanism to prevent deadlocks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LockOrder {
    /// Global configuration locks (highest priority)
    GlobalConfig = 0,

    /// Memory managers and allocators
    MemoryManager = 1,

    /// Transfer caches and optimization data
    TransferCache = 2,

    /// Statistics and monitoring data
    Statistics = 3,

    /// Temporary buffers and staging areas
    TempBuffers = 4,

    /// Debug and tracking data (lowest priority)
    DebugTracking = 5,
}

/// Lock ordering validator to ensure consistent lock acquisition order
pub struct LockOrderValidator {
    current_locks: Vec<LockOrder>,
}

impl LockOrderValidator {
    /// Create a new lock order validator
    pub fn new() -> Self {
        Self {
            current_locks: Vec::new(),
        }
    }

    /// Check if acquiring a lock with the given order would create a deadlock
    pub fn can_acquire(&self, order: LockOrder) -> bool {
        self.current_locks.iter().all(|&existing| existing <= order)
    }

    /// Record that a lock with the given order has been acquired
    pub fn acquired(&mut self, order: LockOrder) -> Result<(), TorshError> {
        if !self.can_acquire(order) {
            return Err(TorshError::BackendError(format!(
                "Lock order violation: trying to acquire {:?} while holding {:?}",
                order, self.current_locks
            )));
        }

        self.current_locks.push(order);
        self.current_locks.sort();
        Ok(())
    }

    /// Record that a lock with the given order has been released
    pub fn released(&mut self, order: LockOrder) {
        if let Some(pos) = self.current_locks.iter().position(|&x| x == order) {
            self.current_locks.remove(pos);
        }
    }

    /// Get currently held locks
    pub fn current_locks(&self) -> &[LockOrder] {
        &self.current_locks
    }
}

impl Default for LockOrderValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Scoped lock guard that automatically updates lock order tracking
pub struct ScopedLockGuard<'a, T> {
    _guard: MutexGuard<'a, T>,
    order: LockOrder,
    validator: Option<&'a mut LockOrderValidator>,
}

impl<'a, T> ScopedLockGuard<'a, T> {
    /// Create a new scoped lock guard
    pub fn new(
        guard: MutexGuard<'a, T>,
        order: LockOrder,
        validator: Option<&'a mut LockOrderValidator>,
    ) -> Self {
        Self {
            _guard: guard,
            order,
            validator,
        }
    }
}

impl<'a, T> Drop for ScopedLockGuard<'a, T> {
    fn drop(&mut self) {
        if let Some(ref mut validator) = self.validator {
            validator.released(self.order);
        }
    }
}

impl<'a, T> std::ops::Deref for ScopedLockGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self._guard
    }
}

impl<'a, T> std::ops::DerefMut for ScopedLockGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self._guard
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_lock_timeout_configuration() {
        let timeout = LockTimeout::default();
        assert_eq!(timeout.max_wait, Duration::from_millis(5000));
        assert_eq!(timeout.retry_interval, Duration::from_millis(10));
        assert_eq!(timeout.max_retries, 500);
    }

    #[test]
    fn test_safe_lock_try_acquire() {
        let mutex = Mutex::new(42);

        // Should succeed when lock is available
        let result = SafeLock::try_acquire(&mutex, "test");
        assert!(result.is_ok());
        let guard = result.unwrap();
        assert!(guard.is_some());
        assert_eq!(*guard.unwrap(), 42);
    }

    #[test]
    fn test_safe_lock_contention() {
        let mutex = Arc::new(Mutex::new(0));
        let mutex_clone = Arc::clone(&mutex);

        // Hold lock in another thread
        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock().unwrap();
            thread::sleep(Duration::from_millis(100));
        });

        // Try to acquire with short timeout
        thread::sleep(Duration::from_millis(10)); // Let other thread acquire lock first

        let timeout = LockTimeout {
            max_wait: Duration::from_millis(50),
            retry_interval: Duration::from_millis(5),
            max_retries: 10,
        };

        let result = SafeLock::acquire_blocking(&mutex, Some(timeout), "test_contention");
        assert!(result.is_err()); // Should timeout

        handle.join().unwrap();
    }

    #[test]
    fn test_lock_order_validator() {
        let mut validator = LockOrderValidator::new();

        // Should be able to acquire locks in correct order
        assert!(validator.can_acquire(LockOrder::GlobalConfig));
        validator.acquired(LockOrder::GlobalConfig).unwrap();

        assert!(validator.can_acquire(LockOrder::MemoryManager));
        validator.acquired(LockOrder::MemoryManager).unwrap();

        assert!(validator.can_acquire(LockOrder::Statistics));
        validator.acquired(LockOrder::Statistics).unwrap();

        // Should not be able to acquire higher-priority lock
        assert!(!validator.can_acquire(LockOrder::TransferCache));
        assert!(validator.acquired(LockOrder::TransferCache).is_err());

        // Release locks
        validator.released(LockOrder::Statistics);
        validator.released(LockOrder::MemoryManager);
        validator.released(LockOrder::GlobalConfig);

        assert_eq!(validator.current_locks().len(), 0);
    }

    #[test]
    fn test_lock_order_enum() {
        // Verify lock order priorities
        assert!(LockOrder::GlobalConfig < LockOrder::MemoryManager);
        assert!(LockOrder::MemoryManager < LockOrder::TransferCache);
        assert!(LockOrder::TransferCache < LockOrder::Statistics);
        assert!(LockOrder::Statistics < LockOrder::TempBuffers);
        assert!(LockOrder::TempBuffers < LockOrder::DebugTracking);
    }
}
