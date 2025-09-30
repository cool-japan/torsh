//! Asynchronous operations for distributed gradient computation
//!
//! This module provides types and utilities for handling asynchronous gradient
//! synchronization operations in distributed training scenarios.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};
use torsh_core::dtype::FloatElement;
use torsh_core::error::Result;

/// Handle for asynchronous gradient synchronization
pub struct AsyncSyncHandle<T: FloatElement> {
    /// Unique identifier for this sync operation
    pub id: usize,
    /// Start time of the synchronization
    pub start_time: Instant,
    /// Whether the synchronization has completed
    pub completed: Arc<AtomicBool>,
    /// Size of data being synchronized
    pub data_size: usize,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FloatElement> AsyncSyncHandle<T> {
    /// Create a new async sync handle
    #[allow(dead_code)]
    pub(crate) fn new(id: usize, data_size: usize) -> Self {
        Self {
            id,
            start_time: Instant::now(),
            completed: Arc::new(AtomicBool::new(false)),
            data_size,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Check if the synchronization is complete
    pub fn is_complete(&self) -> bool {
        self.completed.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get elapsed time since sync started
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Mark the synchronization as completed
    pub(crate) fn mark_completed(&self) {
        self.completed
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get the data size being synchronized
    pub fn data_size(&self) -> usize {
        self.data_size
    }

    /// Get the operation ID
    pub fn id(&self) -> usize {
        self.id
    }
}

/// Network health information for distributed operations
#[derive(Debug, Clone)]
pub struct NetworkHealth {
    /// Whether the network is healthy
    pub is_healthy: bool,
    /// Average bandwidth in Mbps
    pub bandwidth_mbps: f64,
    /// Average latency in milliseconds
    pub latency_ms: f64,
    /// Packet loss rate (0.0 to 1.0)
    pub packet_loss_rate: f64,
    /// List of ranks that failed health checks
    pub failed_ranks: Vec<usize>,
}

impl Default for NetworkHealth {
    fn default() -> Self {
        Self {
            is_healthy: true,
            bandwidth_mbps: 0.0,
            latency_ms: 0.0,
            packet_loss_rate: 0.0,
            failed_ranks: Vec::new(),
        }
    }
}

impl NetworkHealth {
    /// Create a new network health instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if the network is considered healthy
    pub fn is_healthy(&self) -> bool {
        self.is_healthy && self.packet_loss_rate < 0.01 && self.latency_ms < 100.0
    }

    /// Get the effective bandwidth accounting for health issues
    pub fn effective_bandwidth(&self) -> f64 {
        if self.is_healthy() {
            self.bandwidth_mbps
        } else {
            self.bandwidth_mbps * (1.0 - self.packet_loss_rate).max(0.1)
        }
    }

    /// Add a failed rank to the health report
    pub fn add_failed_rank(&mut self, rank: usize) {
        if !self.failed_ranks.contains(&rank) {
            self.failed_ranks.push(rank);
            self.is_healthy = false;
        }
    }

    /// Update bandwidth metrics
    pub fn update_bandwidth(&mut self, bandwidth_mbps: f64) {
        self.bandwidth_mbps = bandwidth_mbps;
    }

    /// Update latency metrics
    pub fn update_latency(&mut self, latency_ms: f64) {
        self.latency_ms = latency_ms;
        if latency_ms > 100.0 {
            self.is_healthy = false;
        }
    }

    /// Update packet loss rate
    pub fn update_packet_loss(&mut self, loss_rate: f64) {
        self.packet_loss_rate = loss_rate.clamp(0.0, 1.0);
        if loss_rate > 0.01 {
            self.is_healthy = false;
        }
    }
}

/// Async operation result for distributed operations
#[derive(Debug, Clone)]
pub enum AsyncOperationResult {
    /// Operation completed successfully
    Success {
        /// Time taken to complete
        duration: Duration,
        /// Amount of data processed
        data_size: usize,
    },
    /// Operation failed
    Failure {
        /// Error message
        error: String,
        /// Time until failure
        duration: Duration,
    },
    /// Operation is still pending
    Pending {
        /// How long it has been running
        elapsed: Duration,
        /// Estimated remaining time
        estimated_remaining: Option<Duration>,
    },
}

/// Async operation status tracker
pub struct AsyncOperationTracker {
    /// Operations currently being tracked
    operations: std::collections::HashMap<usize, AsyncOperationInfo>,
    /// Next operation ID
    next_id: usize,
}

/// Information about a tracked async operation
#[derive(Debug, Clone)]
struct AsyncOperationInfo {
    /// Operation start time
    start_time: Instant,
    /// Operation type description
    operation_type: String,
    /// Data size involved
    data_size: usize,
    /// Estimated completion time
    estimated_duration: Option<Duration>,
}

impl AsyncOperationTracker {
    /// Create a new async operation tracker
    pub fn new() -> Self {
        Self {
            operations: std::collections::HashMap::new(),
            next_id: 0,
        }
    }

    /// Start tracking a new async operation
    pub fn start_operation(
        &mut self,
        operation_type: String,
        data_size: usize,
        estimated_duration: Option<Duration>,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let info = AsyncOperationInfo {
            start_time: Instant::now(),
            operation_type,
            data_size,
            estimated_duration,
        };

        self.operations.insert(id, info);
        id
    }

    /// Mark an operation as completed
    pub fn complete_operation(&mut self, id: usize) -> Option<AsyncOperationResult> {
        if let Some(info) = self.operations.remove(&id) {
            let duration = info.start_time.elapsed();
            Some(AsyncOperationResult::Success {
                duration,
                data_size: info.data_size,
            })
        } else {
            None
        }
    }

    /// Mark an operation as failed
    pub fn fail_operation(&mut self, id: usize, error: String) -> Option<AsyncOperationResult> {
        if let Some(info) = self.operations.remove(&id) {
            let duration = info.start_time.elapsed();
            Some(AsyncOperationResult::Failure { error, duration })
        } else {
            None
        }
    }

    /// Get the status of an operation
    pub fn get_operation_status(&self, id: usize) -> Option<AsyncOperationResult> {
        if let Some(info) = self.operations.get(&id) {
            let elapsed = info.start_time.elapsed();
            let estimated_remaining = info.estimated_duration.map(|est| {
                if elapsed < est {
                    est - elapsed
                } else {
                    Duration::from_millis(0)
                }
            });

            Some(AsyncOperationResult::Pending {
                elapsed,
                estimated_remaining,
            })
        } else {
            None
        }
    }

    /// Get all active operations
    pub fn get_active_operations(&self) -> Vec<usize> {
        self.operations.keys().cloned().collect()
    }

    /// Get the number of active operations
    pub fn active_count(&self) -> usize {
        self.operations.len()
    }

    /// Clear all operations (useful for cleanup)
    pub fn clear(&mut self) {
        self.operations.clear();
    }
}

impl Default for AsyncOperationTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for async operation management
pub mod utils {
    use super::*;

    /// Wait for multiple async handles to complete
    pub fn wait_for_all<T: FloatElement>(
        handles: &[AsyncSyncHandle<T>],
        timeout: Option<Duration>,
    ) -> Result<Vec<Duration>> {
        let start_time = Instant::now();
        let mut results = Vec::with_capacity(handles.len());

        for handle in handles {
            // Check timeout if specified
            if let Some(timeout_duration) = timeout {
                if start_time.elapsed() > timeout_duration {
                    return Err(torsh_core::error::TorshError::AutogradError(
                        "Timeout waiting for async operations".to_string(),
                    ));
                }
            }

            // Wait for this handle to complete
            while !handle.is_complete() {
                std::thread::sleep(Duration::from_micros(100));

                // Check timeout again
                if let Some(timeout_duration) = timeout {
                    if start_time.elapsed() > timeout_duration {
                        return Err(torsh_core::error::TorshError::AutogradError(
                            "Timeout waiting for async operations".to_string(),
                        ));
                    }
                }
            }

            results.push(handle.elapsed());
        }

        Ok(results)
    }

    /// Wait for any of the async handles to complete
    pub fn wait_for_any<T: FloatElement>(
        handles: &[AsyncSyncHandle<T>],
        timeout: Option<Duration>,
    ) -> Result<Option<usize>> {
        let start_time = Instant::now();

        loop {
            // Check each handle
            for (idx, handle) in handles.iter().enumerate() {
                if handle.is_complete() {
                    return Ok(Some(idx));
                }
            }

            // Check timeout
            if let Some(timeout_duration) = timeout {
                if start_time.elapsed() > timeout_duration {
                    return Ok(None); // Timeout, no handle completed
                }
            }

            // Brief sleep to avoid busy waiting
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    /// Create a simple async handle for testing
    #[cfg(test)]
    pub fn create_test_handle<T: FloatElement>(
        id: usize,
        data_size: usize,
        completed: bool,
    ) -> AsyncSyncHandle<T> {
        let handle = AsyncSyncHandle::new(id, data_size);
        if completed {
            handle.mark_completed();
        }
        handle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_sync_handle() {
        let handle = AsyncSyncHandle::<f32>::new(1, 1024);
        assert_eq!(handle.id(), 1);
        assert_eq!(handle.data_size(), 1024);
        assert!(!handle.is_complete());

        handle.mark_completed();
        assert!(handle.is_complete());
    }

    #[test]
    fn test_network_health() {
        let mut health = NetworkHealth::new();
        assert!(health.is_healthy());

        health.update_latency(150.0);
        assert!(!health.is_healthy());

        health.update_latency(50.0);
        health.update_packet_loss(0.02);
        assert!(!health.is_healthy());
    }

    #[test]
    fn test_async_operation_tracker() {
        let mut tracker = AsyncOperationTracker::new();
        assert_eq!(tracker.active_count(), 0);

        let id = tracker.start_operation("test".to_string(), 1024, None);
        assert_eq!(tracker.active_count(), 1);

        let result = tracker.complete_operation(id);
        assert!(matches!(result, Some(AsyncOperationResult::Success { .. })));
        assert_eq!(tracker.active_count(), 0);
    }
}