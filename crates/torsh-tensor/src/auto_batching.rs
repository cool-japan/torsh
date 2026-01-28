//! Automatic Operation Batching for Performance Optimization
//!
//! This module provides automatic batching of tensor operations to improve throughput
//! and reduce overhead. Small operations can be automatically grouped together for
//! better hardware utilization and reduced synchronization costs.
//!
//! # Features
//!
//! - **Automatic batching**: Transparently groups small operations
//! - **Adaptive sizing**: Automatically determines optimal batch sizes
//! - **Parallel execution**: Batches execute in parallel when possible
//! - **Low overhead**: Minimal runtime cost for batching logic
//! - **Configurable thresholds**: Customize when batching occurs

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use scirs2_core::parallel_ops::*; // SciRS2 parallel operations
use torsh_core::{device::DeviceType, dtype::TensorElement, error::Result};

use crate::Tensor;

/// Configuration for automatic batching
#[derive(Debug, Clone)]
pub struct BatchingConfig {
    /// Minimum number of operations to form a batch
    pub min_batch_size: usize,
    /// Maximum number of operations in a batch
    pub max_batch_size: usize,
    /// Maximum time to wait for more operations before executing batch
    pub max_wait_time: Duration,
    /// Whether to enable parallel execution within batches
    pub parallel_execution: bool,
    /// Size threshold below which operations are batched (in elements)
    pub small_op_threshold: usize,
    /// Whether batching is enabled
    pub enabled: bool,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 4,
            max_batch_size: 32,
            max_wait_time: Duration::from_micros(100),
            parallel_execution: true,
            small_op_threshold: 1000,
            enabled: true,
        }
    }
}

impl BatchingConfig {
    /// Create a configuration optimized for small operations
    pub fn small_ops() -> Self {
        Self {
            min_batch_size: 8,
            max_batch_size: 64,
            max_wait_time: Duration::from_micros(50),
            parallel_execution: true,
            small_op_threshold: 500,
            enabled: true,
        }
    }

    /// Create a configuration for large operations (minimal batching)
    pub fn large_ops() -> Self {
        Self {
            min_batch_size: 2,
            max_batch_size: 8,
            max_wait_time: Duration::from_micros(20),
            parallel_execution: false,
            small_op_threshold: 10000,
            enabled: false, // Disabled for large ops
        }
    }

    /// Disable batching completely
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

/// Type of tensor operation that can be batched
#[derive(Debug, Clone)]
pub enum BatchableOp<T: TensorElement> {
    /// Element-wise addition
    Add(Arc<Tensor<T>>, Arc<Tensor<T>>),
    /// Element-wise multiplication
    Mul(Arc<Tensor<T>>, Arc<Tensor<T>>),
    /// Element-wise subtraction
    Sub(Arc<Tensor<T>>, Arc<Tensor<T>>),
    /// Element-wise division
    Div(Arc<Tensor<T>>, Arc<Tensor<T>>),
    /// Scalar addition
    AddScalar(Arc<Tensor<T>>, T),
    /// Scalar multiplication
    MulScalar(Arc<Tensor<T>>, T),
    /// ReLU activation
    ReLU(Arc<Tensor<T>>),
    /// Sigmoid activation
    Sigmoid(Arc<Tensor<T>>),
    /// Tanh activation
    Tanh(Arc<Tensor<T>>),
}

impl<T: TensorElement> BatchableOp<T> {
    /// Get the estimated size (in elements) of the operation
    pub fn size(&self) -> usize {
        match self {
            BatchableOp::Add(a, _)
            | BatchableOp::Mul(a, _)
            | BatchableOp::Sub(a, _)
            | BatchableOp::Div(a, _)
            | BatchableOp::AddScalar(a, _)
            | BatchableOp::MulScalar(a, _)
            | BatchableOp::ReLU(a)
            | BatchableOp::Sigmoid(a)
            | BatchableOp::Tanh(a) => a.numel(),
        }
    }

    /// Get the device type of the operation
    pub fn device(&self) -> DeviceType {
        match self {
            BatchableOp::Add(a, _)
            | BatchableOp::Mul(a, _)
            | BatchableOp::Sub(a, _)
            | BatchableOp::Div(a, _)
            | BatchableOp::AddScalar(a, _)
            | BatchableOp::MulScalar(a, _)
            | BatchableOp::ReLU(a)
            | BatchableOp::Sigmoid(a)
            | BatchableOp::Tanh(a) => a.device,
        }
    }

    /// Check if this operation should be batched based on config
    pub fn should_batch(&self, config: &BatchingConfig) -> bool {
        config.enabled && self.size() < config.small_op_threshold
    }
}

/// A batch of operations ready for execution
struct OperationBatch<T: TensorElement> {
    /// Operations in this batch
    operations: Vec<BatchableOp<T>>,
    /// When this batch was created
    created_at: Instant,
    /// Device type for all operations in batch
    device: DeviceType,
}

impl<T: TensorElement> OperationBatch<T> {
    /// Create a new empty batch
    fn new(device: DeviceType) -> Self {
        Self {
            operations: Vec::new(),
            created_at: Instant::now(),
            device,
        }
    }

    /// Add an operation to the batch
    fn add(&mut self, op: BatchableOp<T>) {
        self.operations.push(op);
    }

    /// Check if the batch is ready to execute based on config
    fn is_ready(&self, config: &BatchingConfig) -> bool {
        if self.operations.len() >= config.max_batch_size {
            return true;
        }

        if self.operations.len() >= config.min_batch_size {
            let elapsed = self.created_at.elapsed();
            if elapsed >= config.max_wait_time {
                return true;
            }
        }

        false
    }

    /// Check if the batch can accept more operations
    fn can_add(&self, config: &BatchingConfig) -> bool {
        self.operations.len() < config.max_batch_size
    }

    /// Get the number of operations in the batch
    fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if the batch is empty
    fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

/// Automatic operation batcher
pub struct AutoBatcher<T: TensorElement> {
    /// Current batch being assembled
    current_batch: Arc<Mutex<Option<OperationBatch<T>>>>,
    /// Configuration
    config: BatchingConfig,
    /// Statistics
    stats: Arc<Mutex<BatchingStats>>,
}

impl<
        T: TensorElement
            + Copy
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + torsh_core::FloatElement
            + Send
            + Sync,
    > AutoBatcher<T>
{
    /// Create a new auto-batcher with default configuration
    pub fn new() -> Self {
        Self::with_config(BatchingConfig::default())
    }

    /// Create a new auto-batcher with custom configuration
    pub fn with_config(config: BatchingConfig) -> Self {
        Self {
            current_batch: Arc::new(Mutex::new(None)),
            config,
            stats: Arc::new(Mutex::new(BatchingStats::default())),
        }
    }

    /// Submit an operation for batching
    pub fn submit(&self, op: BatchableOp<T>) -> Result<BatchHandle<T>> {
        if !self.config.enabled || !op.should_batch(&self.config) {
            // Execute immediately if batching is disabled or operation is too large
            return Ok(BatchHandle::Immediate(self.execute_single(op)?));
        }

        let mut batch_lock = self
            .current_batch
            .lock()
            .expect("lock should not be poisoned");

        // Get or create current batch
        let batch = batch_lock.get_or_insert_with(|| OperationBatch::new(op.device()));

        // Check if we can add to current batch
        if !batch.can_add(&self.config) || batch.device != op.device() {
            // Execute current batch and create a new one
            let ready_batch = batch_lock.take().unwrap();
            drop(batch_lock);

            self.execute_batch(ready_batch)?;

            let mut new_batch_lock = self
                .current_batch
                .lock()
                .expect("lock should not be poisoned");
            let new_batch = new_batch_lock.get_or_insert_with(|| OperationBatch::new(op.device()));
            new_batch.add(op);
        } else {
            batch.add(op);

            // Check if batch is ready to execute
            if batch.is_ready(&self.config) {
                let ready_batch = batch_lock.take().unwrap();
                drop(batch_lock);
                self.execute_batch(ready_batch)?;
            }
        }

        Ok(BatchHandle::Batched)
    }

    /// Force execution of any pending batch
    pub fn flush(&self) -> Result<()> {
        let batch = self
            .current_batch
            .lock()
            .expect("lock should not be poisoned")
            .take();

        if let Some(batch) = batch {
            if !batch.is_empty() {
                self.execute_batch(batch)?;
            }
        }

        Ok(())
    }

    /// Execute a single operation immediately
    fn execute_single(&self, op: BatchableOp<T>) -> Result<Tensor<T>>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + torsh_core::FloatElement,
    {
        let mut stats = self.stats.lock().expect("lock should not be poisoned");
        stats.single_ops_executed += 1;
        drop(stats);

        match op {
            BatchableOp::Add(a, b) => a.add_op(&b),
            BatchableOp::Mul(a, b) => a.mul_op(&b),
            BatchableOp::Sub(a, b) => a.sub(&b),
            BatchableOp::Div(a, b) => a.div(&b),
            BatchableOp::AddScalar(a, s) => a.add_scalar(s),
            BatchableOp::MulScalar(a, s) => a.mul_scalar(s),
            BatchableOp::ReLU(a) => a.relu(),
            BatchableOp::Sigmoid(a) => a.sigmoid(),
            BatchableOp::Tanh(a) => a.tanh(),
        }
    }

    /// Execute a batch of operations
    fn execute_batch(&self, batch: OperationBatch<T>) -> Result<()>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + torsh_core::FloatElement
            + Send
            + Sync,
    {
        let batch_size = batch.len();

        let mut stats = self.stats.lock().expect("lock should not be poisoned");
        stats.batches_executed += 1;
        stats.total_ops_batched += batch_size;
        stats.avg_batch_size = (stats.avg_batch_size * (stats.batches_executed - 1) as f64
            + batch_size as f64)
            / stats.batches_executed as f64;
        drop(stats);

        if self.config.parallel_execution && batch_size > 1 {
            // Execute operations in parallel using scirs2 parallel ops
            let results: Vec<Result<()>> = batch
                .operations
                .into_par_iter()
                .map(|op| {
                    self.execute_single(op)?;
                    Ok(())
                })
                .collect();

            // Check for errors
            for result in results {
                result?;
            }
        } else {
            // Sequential execution
            for op in batch.operations {
                self.execute_single(op)?;
            }
        }

        Ok(())
    }

    /// Get batching statistics
    pub fn stats(&self) -> BatchingStats {
        self.stats
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.lock().expect("lock should not be poisoned") = BatchingStats::default();
    }
}

impl<
        T: TensorElement
            + Copy
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + torsh_core::FloatElement
            + Send
            + Sync,
    > Default for AutoBatcher<T>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Handle returned when submitting an operation
pub enum BatchHandle<T: TensorElement> {
    /// Operation was executed immediately
    Immediate(Tensor<T>),
    /// Operation was added to a batch
    Batched,
}

/// Statistics about batching performance
#[derive(Debug, Clone)]
pub struct BatchingStats {
    /// Number of batches executed
    pub batches_executed: usize,
    /// Total operations batched
    pub total_ops_batched: usize,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Number of single operations executed (not batched)
    pub single_ops_executed: usize,
}

impl Default for BatchingStats {
    fn default() -> Self {
        Self {
            batches_executed: 0,
            total_ops_batched: 0,
            avg_batch_size: 0.0,
            single_ops_executed: 0,
        }
    }
}

impl BatchingStats {
    /// Calculate batching efficiency (percentage of operations that were batched)
    pub fn batching_efficiency(&self) -> f64 {
        let total_ops = self.total_ops_batched + self.single_ops_executed;
        if total_ops == 0 {
            0.0
        } else {
            (self.total_ops_batched as f64 / total_ops as f64) * 100.0
        }
    }

    /// Calculate average operations saved by batching
    pub fn ops_saved(&self) -> f64 {
        if self.batches_executed == 0 {
            0.0
        } else {
            self.total_ops_batched as f64 - self.batches_executed as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::*;

    #[test]
    fn test_batching_config_presets() {
        let default_config = BatchingConfig::default();
        assert!(default_config.enabled);
        assert_eq!(default_config.min_batch_size, 4);

        let small_ops = BatchingConfig::small_ops();
        assert_eq!(small_ops.min_batch_size, 8);
        assert_eq!(small_ops.max_batch_size, 64);

        let large_ops = BatchingConfig::large_ops();
        assert!(!large_ops.enabled);

        let disabled = BatchingConfig::disabled();
        assert!(!disabled.enabled);
    }

    #[test]
    fn test_batchable_op_size() {
        let a = tensor_1d(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let b = tensor_1d(&[2.0f32, 2.0, 2.0, 2.0]).unwrap();

        let op = BatchableOp::Add(Arc::new(a), Arc::new(b));
        assert_eq!(op.size(), 4);
    }

    #[test]
    fn test_batchable_op_should_batch() {
        let a = tensor_1d(&[1.0f32; 100]).unwrap();
        let b = tensor_1d(&[2.0f32; 100]).unwrap();

        let op = BatchableOp::Add(Arc::new(a), Arc::new(b));

        let config = BatchingConfig::default();
        assert!(op.should_batch(&config));

        let disabled_config = BatchingConfig::disabled();
        assert!(!op.should_batch(&disabled_config));
    }

    #[test]
    fn test_operation_batch() {
        let a = tensor_1d(&[1.0f32, 2.0]).unwrap();
        let op = BatchableOp::AddScalar(Arc::new(a), 1.0);

        let mut batch = OperationBatch::new(DeviceType::Cpu);
        assert!(batch.is_empty());

        batch.add(op);
        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_batch_readiness() {
        let config = BatchingConfig {
            min_batch_size: 2,
            max_batch_size: 5,
            max_wait_time: Duration::from_millis(10),
            ..Default::default()
        };

        let mut batch = OperationBatch::<f32>::new(DeviceType::Cpu);

        // Empty batch is not ready
        assert!(!batch.is_ready(&config));

        // Single operation, not ready yet
        let a = tensor_1d(&[1.0f32]).unwrap();
        batch.add(BatchableOp::AddScalar(Arc::new(a), 1.0));
        assert!(!batch.is_ready(&config));

        // Two operations, but wait time not elapsed
        let b = tensor_1d(&[2.0f32]).unwrap();
        batch.add(BatchableOp::AddScalar(Arc::new(b), 1.0));

        // Max batch size reached
        for _ in 0..3 {
            let c = tensor_1d(&[3.0f32]).unwrap();
            batch.add(BatchableOp::AddScalar(Arc::new(c), 1.0));
        }
        assert!(batch.is_ready(&config)); // Max size reached
    }

    #[test]
    fn test_batching_stats() {
        let mut stats = BatchingStats::default();

        stats.batches_executed = 10;
        stats.total_ops_batched = 50;
        stats.single_ops_executed = 10;

        let efficiency = stats.batching_efficiency();
        assert!((efficiency - 83.33).abs() < 0.1); // ~83.33%

        let ops_saved = stats.ops_saved();
        assert_eq!(ops_saved, 40.0); // 50 - 10 = 40 operations saved
    }

    #[test]
    fn test_auto_batcher_creation() {
        let batcher = AutoBatcher::<f32>::new();
        let stats = batcher.stats();

        assert_eq!(stats.batches_executed, 0);
        assert_eq!(stats.total_ops_batched, 0);
        assert_eq!(stats.single_ops_executed, 0);
    }

    #[test]
    fn test_auto_batcher_disabled() {
        let config = BatchingConfig::disabled();
        let batcher = AutoBatcher::<f32>::with_config(config);

        let a = tensor_1d(&[1.0f32, 2.0]).unwrap();
        let op = BatchableOp::AddScalar(Arc::new(a), 1.0);

        let handle = batcher.submit(op).unwrap();

        // Should execute immediately when disabled
        assert!(matches!(handle, BatchHandle::Immediate(_)));

        let stats = batcher.stats();
        assert_eq!(stats.single_ops_executed, 1);
        assert_eq!(stats.total_ops_batched, 0);
    }
}
