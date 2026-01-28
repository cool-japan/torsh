//! Asynchronous Tensor Operations
//!
//! This module provides asynchronous tensor operations using futures-based API
//! for non-blocking computation, allowing better resource utilization and
//! parallelization of tensor operations.

use crate::{Tensor, TensorElement};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use torsh_core::error::Result;

/// Async operation handle that can be awaited
pub struct AsyncTensorOp<T: TensorElement> {
    inner: Pin<Box<dyn Future<Output = Result<Tensor<T>>> + Send + 'static>>,
}

impl<T: TensorElement> AsyncTensorOp<T> {
    /// Create a new async tensor operation
    pub fn new<F>(future: F) -> Self
    where
        F: Future<Output = Result<Tensor<T>>> + Send + 'static,
    {
        Self {
            inner: Box::pin(future),
        }
    }
}

impl<T: TensorElement> Future for AsyncTensorOp<T> {
    type Output = Result<Tensor<T>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

/// Async operation scheduler for managing concurrent tensor operations
pub struct AsyncOperationScheduler {
    /// Thread pool for executing operations
    thread_pool: Arc<rayon::ThreadPool>,
    /// Maximum concurrent operations (reserved for future rate limiting)
    _max_concurrent_ops: usize,
    /// Currently running operations count
    active_operations: Arc<Mutex<usize>>,
}

impl AsyncOperationScheduler {
    /// Create a new async operation scheduler
    pub fn new() -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(rayon::current_num_threads())
            .build()
            .expect("Failed to create thread pool");

        Self {
            thread_pool: Arc::new(thread_pool),
            _max_concurrent_ops: rayon::current_num_threads() * 2,
            active_operations: Arc::new(Mutex::new(0)),
        }
    }

    /// Create a scheduler with custom configuration
    pub fn with_config(num_threads: usize, _max_concurrent_ops: usize) -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Failed to create thread pool");

        Self {
            thread_pool: Arc::new(thread_pool),
            _max_concurrent_ops,
            active_operations: Arc::new(Mutex::new(0)),
        }
    }

    /// Schedule an async operation
    pub fn schedule<F, T>(&self, operation: F) -> AsyncTensorOp<T>
    where
        F: FnOnce() -> Result<Tensor<T>> + Send + 'static,
        T: TensorElement + Send + 'static,
    {
        let thread_pool = Arc::clone(&self.thread_pool);
        let active_ops = Arc::clone(&self.active_operations);

        let future = async move {
            // Wait for available slot
            loop {
                {
                    let mut active = active_ops.lock().expect("lock should not be poisoned");
                    if *active < 8 {
                        // Simple rate limiting
                        *active += 1;
                        break;
                    }
                }
                // Yield to allow other tasks to run
                tokio::task::yield_now().await;
            }

            // Execute operation on thread pool
            let result = thread_pool.install(|| operation());

            // Decrement active operations count
            {
                let mut active = active_ops.lock().expect("lock should not be poisoned");
                *active -= 1;
            }

            result
        };

        AsyncTensorOp::new(future)
    }

    /// Get current active operations count
    pub fn active_operations(&self) -> usize {
        *self
            .active_operations
            .lock()
            .expect("lock should not be poisoned")
    }
}

/// Global async operation scheduler
static ASYNC_SCHEDULER: std::sync::LazyLock<AsyncOperationScheduler> =
    std::sync::LazyLock::new(|| AsyncOperationScheduler::new());

/// Get the global async operation scheduler
pub fn get_async_scheduler() -> &'static AsyncOperationScheduler {
    &ASYNC_SCHEDULER
}

/// Async tensor operations implementation
impl<T: TensorElement + Copy + Send + 'static + num_traits::FromPrimitive + std::iter::Sum>
    Tensor<T>
{
    /// Asynchronous element-wise addition
    pub fn add_async(&self, other: &Self) -> AsyncTensorOp<T>
    where
        T: std::ops::Add<Output = T> + num_traits::Float,
    {
        let lhs = self.clone();
        let rhs = other.clone();

        get_async_scheduler().schedule(move || lhs.add_scirs2(&rhs))
    }

    /// Asynchronous element-wise multiplication
    pub fn mul_async(&self, other: &Self) -> AsyncTensorOp<T>
    where
        T: std::ops::Mul<Output = T> + num_traits::Float,
    {
        let lhs = self.clone();
        let rhs = other.clone();

        get_async_scheduler().schedule(move || lhs.mul_scirs2(&rhs))
    }

    /// Asynchronous matrix multiplication
    pub fn matmul_async(&self, other: &Self) -> AsyncTensorOp<T>
    where
        T: num_traits::Float + num_traits::Zero + num_traits::One,
    {
        let lhs = self.clone();
        let rhs = other.clone();

        get_async_scheduler().schedule(move || lhs.matmul_scirs2(&rhs))
    }

    /// Asynchronous sum reduction
    pub fn sum_async(&self) -> AsyncTensorOp<T>
    where
        T: std::ops::Add<Output = T> + num_traits::Zero,
    {
        let tensor = self.clone();

        get_async_scheduler().schedule(move || tensor.sum_scirs2())
    }

    /// Asynchronous mean reduction
    pub fn mean_async(&self) -> AsyncTensorOp<T>
    where
        T: std::ops::Add<Output = T> + std::ops::Div<Output = T> + num_traits::Zero + From<usize>,
    {
        let tensor = self.clone();

        get_async_scheduler().schedule(move || tensor.mean_scirs2())
    }

    /// Asynchronous ReLU activation
    pub fn relu_async(&self) -> AsyncTensorOp<T>
    where
        T: PartialOrd + num_traits::Zero,
    {
        let tensor = self.clone();

        get_async_scheduler().schedule(move || tensor.relu_scirs2())
    }

    /// Asynchronous sigmoid activation
    pub fn sigmoid_async(&self) -> AsyncTensorOp<T>
    where
        T: num_traits::Float,
    {
        let tensor = self.clone();

        get_async_scheduler().schedule(move || tensor.sigmoid_scirs2())
    }

    /// Asynchronous tanh activation
    pub fn tanh_async(&self) -> AsyncTensorOp<T>
    where
        T: num_traits::Float,
    {
        let tensor = self.clone();

        get_async_scheduler().schedule(move || tensor.tanh_scirs2())
    }
}

/// Async operation batching for multiple operations
pub struct AsyncBatch<T: TensorElement> {
    operations: Vec<AsyncTensorOp<T>>,
}

impl<T: TensorElement> AsyncBatch<T> {
    /// Create a new async batch
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Add an operation to the batch
    pub fn add_operation(mut self, op: AsyncTensorOp<T>) -> Self {
        self.operations.push(op);
        self
    }

    /// Execute all operations in the batch concurrently
    pub async fn execute_all(self) -> Result<Vec<Tensor<T>>> {
        let futures: Vec<_> = self.operations.into_iter().collect();

        // Use futures::future::join_all when available, for now use simple approach
        let mut results = Vec::new();
        for future in futures {
            results.push(future.await?);
        }

        Ok(results)
    }

    /// Execute operations with specified concurrency limit
    pub async fn execute_with_limit(self, limit: usize) -> Result<Vec<Tensor<T>>> {
        let mut results = Vec::with_capacity(self.operations.len());
        let mut futures = self.operations.into_iter();

        // Simple chunked execution
        loop {
            let chunk: Vec<_> = futures.by_ref().take(limit).collect();
            if chunk.is_empty() {
                break;
            }

            for future in chunk {
                results.push(future.await?);
            }
        }

        Ok(results)
    }
}

impl<T: TensorElement> Default for AsyncBatch<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for async operations
pub mod convenience {
    use super::*;

    /// Perform multiple tensor additions asynchronously
    pub async fn async_add_multiple<T>(tensors: &[Tensor<T>]) -> Result<Tensor<T>>
    where
        T: TensorElement
            + Copy
            + Send
            + 'static
            + std::ops::Add<Output = T>
            + num_traits::Float
            + num_traits::FromPrimitive
            + std::iter::Sum,
    {
        if tensors.is_empty() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Empty tensor list".to_string(),
            ));
        }

        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }

        let result = tensors[0].clone();
        let mut batch = AsyncBatch::new();

        for tensor in &tensors[1..] {
            let op = result.add_async(tensor);
            batch = batch.add_operation(op);
        }

        let results = batch.execute_all().await?;
        Ok(results
            .into_iter()
            .last()
            .expect("results should not be empty after batch execution"))
    }

    /// Perform matrix chain multiplication asynchronously
    pub async fn async_chain_matmul<T>(tensors: &[Tensor<T>]) -> Result<Tensor<T>>
    where
        T: TensorElement
            + Copy
            + Send
            + 'static
            + num_traits::Float
            + num_traits::Zero
            + num_traits::One
            + num_traits::FromPrimitive
            + std::iter::Sum,
    {
        if tensors.is_empty() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Empty tensor list".to_string(),
            ));
        }

        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }

        let mut result = tensors[0].clone();

        for tensor in &tensors[1..] {
            result = result.matmul_async(tensor).await?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[tokio::test]
    async fn test_async_add() {
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![4.0f32, 5.0, 6.0], vec![3], DeviceType::Cpu).unwrap();

        let result = a.add_async(&b).await.unwrap();
        let expected = vec![5.0f32, 7.0, 9.0];

        assert_eq!(result.to_vec().unwrap(), expected);
    }

    #[tokio::test]
    async fn test_async_matmul() {
        let a =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        let b =
            Tensor::from_data(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2], DeviceType::Cpu).unwrap();

        let result = a.matmul_async(&b).await.unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);

        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        let expected = vec![19.0f32, 22.0, 43.0, 50.0];
        assert_eq!(result.to_vec().unwrap(), expected);
    }

    #[tokio::test]
    async fn test_async_batch() {
        let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu).unwrap();
        let c = Tensor::from_data(vec![5.0f32, 6.0], vec![2], DeviceType::Cpu).unwrap();

        let batch = AsyncBatch::new()
            .add_operation(a.add_async(&b))
            .add_operation(a.mul_async(&c));

        let results = batch.execute_all().await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].to_vec().unwrap(), vec![4.0f32, 6.0]); // a + b
        assert_eq!(results[1].to_vec().unwrap(), vec![5.0f32, 12.0]); // a * c
    }

    #[test]
    fn test_async_scheduler() {
        let scheduler = AsyncOperationScheduler::new();
        assert_eq!(scheduler.active_operations(), 0);

        let custom_scheduler = AsyncOperationScheduler::with_config(4, 8);
        assert_eq!(custom_scheduler.active_operations(), 0);
    }
}
