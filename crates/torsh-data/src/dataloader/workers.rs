//! Worker pool functionality for multi-process data loading
//!
//! This module provides worker pool implementations for parallel data loading,
//! including both standard and persistent worker pools.

use crate::{collate::Collate, dataset::Dataset, sampler::BatchSampler};
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use torsh_core::error::Result;

/// Worker task containing indices to process
#[derive(Debug, Clone)]
struct WorkerTask {
    task_id: usize,
    indices: Vec<usize>,
}

/// Worker result containing processed batch
#[derive(Debug)]
pub struct WorkerResult<T> {
    pub task_id: usize,
    pub result: Result<T>,
}

/// Handle to a worker thread
struct WorkerHandle<T> {
    _thread: thread::JoinHandle<()>,
    _phantom: std::marker::PhantomData<T>,
}

/// Worker process manager for multi-process data loading
///
/// This struct manages a pool of worker threads that process batches in parallel,
/// allowing for efficient utilization of multiple CPU cores during data loading.
///
/// # Type Parameters
///
/// * `D` - Dataset type implementing the Dataset trait
/// * `C` - Collate function type implementing the Collate trait
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
/// use torsh_data::dataloader::workers::WorkerPool;
/// use torsh_data::dataset::TensorDataset;
/// use torsh_data::collate::DefaultCollate;
///
/// let dataset = Arc::new(TensorDataset::new(vec![1, 2, 3, 4, 5]));
/// let collate_fn = Arc::new(DefaultCollate);
/// let worker_pool = WorkerPool::new(dataset, collate_fn, 4);
///
/// // Submit tasks and collect results
/// worker_pool.submit_task(0, vec![0, 1]).unwrap();
/// let result = worker_pool.get_result().unwrap();
/// ```
pub struct WorkerPool<D, C>
where
    D: Dataset + Clone + Send + Sync + 'static,
    C: Collate<D::Item> + Clone + Send + Sync + 'static,
    D::Item: Send + 'static,
    C::Output: Send + 'static,
{
    #[allow(dead_code)]
    dataset: Arc<D>,
    #[allow(dead_code)]
    collate_fn: Arc<C>,
    num_workers: usize,
    #[allow(dead_code)]
    workers: Vec<WorkerHandle<C::Output>>,
    task_sender: mpsc::Sender<WorkerTask>,
    result_receiver: mpsc::Receiver<WorkerResult<C::Output>>,
}

impl<D, C> WorkerPool<D, C>
where
    D: Dataset + Clone + Send + Sync + 'static,
    C: Collate<D::Item> + Clone + Send + Sync + 'static,
    D::Item: Send + 'static,
    C::Output: Send + 'static,
{
    /// Create a new worker pool
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset to process
    /// * `collate_fn` - The collation function to apply to batches
    /// * `num_workers` - Number of worker threads to spawn
    ///
    /// # Returns
    ///
    /// A new WorkerPool ready to process tasks
    pub fn new(dataset: Arc<D>, collate_fn: Arc<C>, num_workers: usize) -> Self {
        let (task_sender, task_receiver) = mpsc::channel::<WorkerTask>();
        let (result_sender, result_receiver) = mpsc::channel::<WorkerResult<C::Output>>();

        let task_receiver = Arc::new(Mutex::new(task_receiver));
        let mut workers = Vec::with_capacity(num_workers);

        // Spawn worker threads
        for worker_id in 0..num_workers {
            let dataset_clone = Arc::clone(&dataset);
            let collate_fn_clone = Arc::clone(&collate_fn);
            let task_receiver_clone = Arc::clone(&task_receiver);
            let result_sender_clone = result_sender.clone();

            let worker_thread = thread::spawn(move || {
                Self::worker_loop(
                    worker_id,
                    dataset_clone,
                    collate_fn_clone,
                    task_receiver_clone,
                    result_sender_clone,
                );
            });

            workers.push(WorkerHandle {
                _thread: worker_thread,
                _phantom: std::marker::PhantomData,
            });
        }

        Self {
            dataset,
            collate_fn,
            num_workers,
            workers,
            task_sender,
            result_receiver,
        }
    }

    /// Worker thread loop
    ///
    /// This is the main loop that each worker thread runs to process tasks.
    #[allow(clippy::too_many_arguments)]
    fn worker_loop(
        _worker_id: usize,
        dataset: Arc<D>,
        collate_fn: Arc<C>,
        task_receiver: Arc<Mutex<mpsc::Receiver<WorkerTask>>>,
        result_sender: mpsc::Sender<WorkerResult<C::Output>>,
    ) {
        loop {
            // Try to receive a task
            let task = {
                let receiver = match task_receiver.lock() {
                    Ok(receiver) => receiver,
                    Err(_) => {
                        // Mutex is poisoned, worker should exit
                        break;
                    }
                };
                receiver.recv()
            };

            match task {
                Ok(WorkerTask { task_id, indices }) => {
                    // Process the batch
                    let batch_result = Self::process_batch(&*dataset, &*collate_fn, indices);

                    let result = WorkerResult {
                        task_id,
                        result: batch_result,
                    };

                    // Send result back
                    if result_sender.send(result).is_err() {
                        // Main thread is no longer listening, exit
                        break;
                    }
                }
                Err(_) => {
                    // Main thread closed the channel, exit
                    break;
                }
            }
        }
    }

    /// Process a batch of indices
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset to load from
    /// * `collate_fn` - The collation function to apply
    /// * `indices` - The indices of samples to load
    ///
    /// # Returns
    ///
    /// The collated batch result
    fn process_batch(dataset: &D, collate_fn: &C, indices: Vec<usize>) -> Result<C::Output> {
        let mut samples = Vec::with_capacity(indices.len());

        for idx in indices {
            match dataset.get(idx) {
                Ok(sample) => samples.push(sample),
                Err(e) => return Err(e),
            }
        }

        collate_fn.collate(samples)
    }

    /// Submit a task to the worker pool
    ///
    /// # Arguments
    ///
    /// * `task_id` - Unique identifier for the task
    /// * `indices` - Indices of samples to process
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of task submission
    pub fn submit_task(&self, task_id: usize, indices: Vec<usize>) -> Result<()> {
        let task = WorkerTask { task_id, indices };

        self.task_sender.send(task).map_err(|_| {
            torsh_core::error::TorshError::RuntimeError(
                "Failed to send task to worker pool".to_string(),
            )
        })?;

        Ok(())
    }

    /// Get a result from the worker pool
    ///
    /// This method blocks until a result is available from any worker.
    ///
    /// # Returns
    ///
    /// The next available worker result
    pub fn get_result(&self) -> Result<WorkerResult<C::Output>> {
        self.result_receiver.recv().map_err(|_| {
            torsh_core::error::TorshError::RuntimeError(
                "Failed to receive result from worker pool".to_string(),
            )
        })
    }

    /// Try to get a result without blocking
    ///
    /// # Returns
    ///
    /// Some(result) if a result is available, None otherwise
    pub fn try_get_result(&self) -> Option<WorkerResult<C::Output>> {
        self.result_receiver.try_recv().ok()
    }

    /// Get the number of workers
    ///
    /// # Returns
    ///
    /// The number of worker threads in the pool
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

/// Multi-process DataLoader iterator
///
/// This iterator coordinates with a WorkerPool to process batches in parallel,
/// maintaining a pipeline of pending tasks to maximize throughput.
///
/// # Type Parameters
///
/// * `D` - Dataset type
/// * `S` - Sampler type
/// * `C` - Collate function type
pub struct MultiProcessIterator<'a, D, S, C>
where
    D: Dataset + Clone + Send + Sync + 'static,
    S: BatchSampler,
    C: Collate<D::Item> + Clone + Send + Sync + 'static,
    D::Item: Send + 'static,
    C::Output: Send + 'static,
{
    sampler_iter: S::Iter,
    worker_pool: &'a WorkerPool<D, C>,
    pending_tasks: HashMap<usize, Vec<usize>>,
    next_task_id: usize,
    max_pending: usize,
}

impl<'a, D, S, C> MultiProcessIterator<'a, D, S, C>
where
    D: Dataset + Clone + Send + Sync + 'static,
    S: BatchSampler,
    C: Collate<D::Item> + Clone + Send + Sync + 'static,
    D::Item: Send + 'static,
    C::Output: Send + 'static,
{
    /// Create a new multi-process iterator
    ///
    /// # Arguments
    ///
    /// * `sampler_iter` - Iterator over batch indices
    /// * `worker_pool` - Worker pool to use for processing
    ///
    /// # Returns
    ///
    /// A new MultiProcessIterator
    pub fn new(sampler_iter: S::Iter, worker_pool: &'a WorkerPool<D, C>) -> Self {
        let max_pending = worker_pool.num_workers() * 2; // Buffer 2x the number of workers

        Self {
            sampler_iter,
            worker_pool,
            pending_tasks: HashMap::new(),
            next_task_id: 0,
            max_pending,
        }
    }

    /// Create a new multi-process iterator with custom buffer size
    ///
    /// # Arguments
    ///
    /// * `sampler_iter` - Iterator over batch indices
    /// * `worker_pool` - Worker pool to use for processing
    /// * `max_pending` - Maximum number of pending tasks
    ///
    /// # Returns
    ///
    /// A new MultiProcessIterator with custom buffering
    pub fn with_buffer_size(
        sampler_iter: S::Iter,
        worker_pool: &'a WorkerPool<D, C>,
        max_pending: usize,
    ) -> Self {
        Self {
            sampler_iter,
            worker_pool,
            pending_tasks: HashMap::new(),
            next_task_id: 0,
            max_pending,
        }
    }

    /// Submit tasks to keep the pipeline full
    fn submit_tasks(&mut self) {
        while self.pending_tasks.len() < self.max_pending {
            if let Some(indices) = self.sampler_iter.next() {
                let task_id = self.next_task_id;
                self.next_task_id += 1;

                if self
                    .worker_pool
                    .submit_task(task_id, indices.clone())
                    .is_ok()
                {
                    self.pending_tasks.insert(task_id, indices);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Get the number of pending tasks
    pub fn pending_count(&self) -> usize {
        self.pending_tasks.len()
    }

    /// Check if there are any pending tasks
    pub fn has_pending_tasks(&self) -> bool {
        !self.pending_tasks.is_empty()
    }
}

impl<D, S, C> Iterator for MultiProcessIterator<'_, D, S, C>
where
    D: Dataset + Clone + Send + Sync + 'static,
    S: BatchSampler,
    S::Iter: Iterator<Item = Vec<usize>>,
    C: Collate<D::Item> + Clone + Send + Sync + 'static,
    D::Item: Send + 'static,
    C::Output: Send + 'static,
{
    type Item = Result<C::Output>;

    fn next(&mut self) -> Option<Self::Item> {
        // Submit new tasks to keep pipeline full
        self.submit_tasks();

        // If no pending tasks, we're done
        if self.pending_tasks.is_empty() {
            return None;
        }

        // Wait for a result
        match self.worker_pool.get_result() {
            Ok(WorkerResult { task_id, result }) => {
                // Remove the completed task
                self.pending_tasks.remove(&task_id);
                Some(result)
            }
            Err(e) => Some(Err(e)),
        }
    }
}

/// Messages for persistent workers
#[derive(Debug)]
enum PersistentWorkerMessage {
    Task(WorkerTask),
    Shutdown,
}

/// Handle to a persistent worker
struct PersistentWorkerHandle {
    _thread: thread::JoinHandle<()>,
}

/// Persistent worker pool that keeps workers alive across epochs
///
/// Unlike the regular WorkerPool, this implementation keeps worker threads alive
/// between epochs, reducing the overhead of thread creation and destruction.
/// This is particularly useful for training scenarios with multiple epochs.
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
/// use torsh_data::dataloader::workers::PersistentWorkerPool;
/// use torsh_data::dataset::TensorDataset;
/// use torsh_data::collate::DefaultCollate;
///
/// let dataset = Arc::new(TensorDataset::new(vec![1, 2, 3, 4, 5]));
/// let collate_fn = Arc::new(DefaultCollate);
/// let worker_pool = PersistentWorkerPool::new(dataset, collate_fn, 4);
///
/// // Use across multiple epochs
/// for epoch in 0..10 {
///     worker_pool.reset_for_epoch().unwrap();
///     // Submit tasks for this epoch
/// }
///
/// // Shutdown when done
/// worker_pool.shutdown().unwrap();
/// ```
pub struct PersistentWorkerPool<D, C>
where
    D: Dataset + Clone + Send + Sync + 'static,
    C: Collate<D::Item> + Clone + Send + Sync + 'static,
    D::Item: Send + 'static,
    C::Output: Send + 'static,
{
    #[allow(dead_code)]
    dataset: Arc<D>,
    #[allow(dead_code)]
    collate_fn: Arc<C>,
    num_workers: usize,
    #[allow(dead_code)]
    workers: Vec<PersistentWorkerHandle>,
    task_sender: mpsc::Sender<PersistentWorkerMessage>,
    result_receiver: mpsc::Receiver<WorkerResult<C::Output>>,
    is_shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl<D, C> PersistentWorkerPool<D, C>
where
    D: Dataset + Clone + Send + Sync + 'static,
    C: Collate<D::Item> + Clone + Send + Sync + 'static,
    D::Item: Send + 'static,
    C::Output: Send + 'static,
{
    /// Create a new persistent worker pool
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset to process
    /// * `collate_fn` - The collation function to apply to batches
    /// * `num_workers` - Number of worker threads to spawn
    ///
    /// # Returns
    ///
    /// A new PersistentWorkerPool ready to process tasks
    pub fn new(dataset: Arc<D>, collate_fn: Arc<C>, num_workers: usize) -> Self {
        let (task_sender, task_receiver) = mpsc::channel::<PersistentWorkerMessage>();
        let (result_sender, result_receiver) = mpsc::channel::<WorkerResult<C::Output>>();

        let task_receiver = Arc::new(Mutex::new(task_receiver));
        let is_shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut workers = Vec::with_capacity(num_workers);

        // Spawn persistent worker threads
        for worker_id in 0..num_workers {
            let dataset_clone = Arc::clone(&dataset);
            let collate_fn_clone = Arc::clone(&collate_fn);
            let task_receiver_clone = Arc::clone(&task_receiver);
            let result_sender_clone = result_sender.clone();
            let is_shutdown_clone = Arc::clone(&is_shutdown);

            let worker_thread = thread::spawn(move || {
                Self::persistent_worker_loop(
                    worker_id,
                    dataset_clone,
                    collate_fn_clone,
                    task_receiver_clone,
                    result_sender_clone,
                    is_shutdown_clone,
                );
            });

            workers.push(PersistentWorkerHandle {
                _thread: worker_thread,
            });
        }

        Self {
            dataset,
            collate_fn,
            num_workers,
            workers,
            task_sender,
            result_receiver,
            is_shutdown,
        }
    }

    /// Persistent worker thread loop that stays alive
    ///
    /// This loop includes timeout-based shutdown checking to ensure workers
    /// can be cleanly terminated even if no tasks are pending.
    #[allow(clippy::too_many_arguments)]
    fn persistent_worker_loop(
        _worker_id: usize,
        dataset: Arc<D>,
        collate_fn: Arc<C>,
        task_receiver: Arc<Mutex<mpsc::Receiver<PersistentWorkerMessage>>>,
        result_sender: mpsc::Sender<WorkerResult<C::Output>>,
        is_shutdown: Arc<std::sync::atomic::AtomicBool>,
    ) {
        loop {
            // Check for shutdown signal
            if is_shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }

            // Try to receive a message with timeout to allow periodic shutdown checks
            let message = {
                let receiver = match task_receiver.lock() {
                    Ok(receiver) => receiver,
                    Err(_) => {
                        // Mutex is poisoned, worker should exit
                        break;
                    }
                };
                receiver.recv_timeout(std::time::Duration::from_millis(100))
            };

            match message {
                Ok(PersistentWorkerMessage::Task(WorkerTask { task_id, indices })) => {
                    // Process the batch
                    let batch_result = Self::process_batch(&*dataset, &*collate_fn, indices);

                    let result = WorkerResult {
                        task_id,
                        result: batch_result,
                    };

                    // Send result back
                    if result_sender.send(result).is_err() {
                        // Main thread is no longer listening, exit
                        break;
                    }
                }
                Ok(PersistentWorkerMessage::Shutdown) => {
                    // Explicit shutdown request
                    break;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Timeout occurred, continue the loop to check for shutdown
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Main thread closed the channel, exit
                    break;
                }
            }
        }
    }

    /// Process a batch of indices (same as WorkerPool)
    fn process_batch(dataset: &D, collate_fn: &C, indices: Vec<usize>) -> Result<C::Output> {
        let mut samples = Vec::with_capacity(indices.len());

        for idx in indices {
            match dataset.get(idx) {
                Ok(sample) => samples.push(sample),
                Err(e) => return Err(e),
            }
        }

        collate_fn.collate(samples)
    }

    /// Submit a task to the persistent worker pool
    ///
    /// # Arguments
    ///
    /// * `task_id` - Unique identifier for the task
    /// * `indices` - Indices of samples to process
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of task submission
    pub fn submit_task(&self, task_id: usize, indices: Vec<usize>) -> Result<()> {
        let message = PersistentWorkerMessage::Task(WorkerTask { task_id, indices });

        self.task_sender.send(message).map_err(|_| {
            torsh_core::error::TorshError::RuntimeError(
                "Failed to send task to persistent worker pool".to_string(),
            )
        })?;

        Ok(())
    }

    /// Get a result from the persistent worker pool
    ///
    /// This method blocks until a result is available from any worker.
    ///
    /// # Returns
    ///
    /// The next available worker result
    pub fn get_result(&self) -> Result<WorkerResult<C::Output>> {
        self.result_receiver.recv().map_err(|_| {
            torsh_core::error::TorshError::RuntimeError(
                "Failed to receive result from persistent worker pool".to_string(),
            )
        })
    }

    /// Get a result with timeout
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for a result
    ///
    /// # Returns
    ///
    /// The next available worker result or a timeout error
    pub fn get_result_timeout(
        &self,
        timeout: std::time::Duration,
    ) -> Result<WorkerResult<C::Output>> {
        self.result_receiver
            .recv_timeout(timeout)
            .map_err(|e| match e {
                mpsc::RecvTimeoutError::Timeout => torsh_core::error::TorshError::RuntimeError(
                    "Timeout waiting for result from persistent worker pool".to_string(),
                ),
                mpsc::RecvTimeoutError::Disconnected => {
                    torsh_core::error::TorshError::RuntimeError(
                        "Persistent worker pool disconnected".to_string(),
                    )
                }
            })
    }

    /// Try to get a result without blocking
    ///
    /// # Returns
    ///
    /// Some(result) if a result is available, None otherwise
    pub fn try_get_result(&self) -> Option<WorkerResult<C::Output>> {
        self.result_receiver.try_recv().ok()
    }

    /// Get the number of workers
    ///
    /// # Returns
    ///
    /// The number of worker threads in the pool
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Check if the pool is shutdown
    ///
    /// # Returns
    ///
    /// True if the pool has been shutdown, false otherwise
    pub fn is_shutdown(&self) -> bool {
        self.is_shutdown.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Shutdown the persistent worker pool
    ///
    /// This method gracefully shuts down all worker threads and should be called
    /// when the pool is no longer needed.
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of shutdown operation
    pub fn shutdown(&self) -> Result<()> {
        // Set shutdown flag
        self.is_shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);

        // Send shutdown messages to all workers
        for _ in 0..self.num_workers {
            if self
                .task_sender
                .send(PersistentWorkerMessage::Shutdown)
                .is_err()
            {
                // Channel is closed, workers are already shutdown
                break;
            }
        }

        Ok(())
    }

    /// Reset the pool for a new epoch (clears any pending tasks)
    ///
    /// This method can be called between epochs to reset the pool state.
    /// In the current implementation, this is a placeholder for future
    /// epoch-specific optimizations.
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of reset operation
    pub fn reset_for_epoch(&self) -> Result<()> {
        // For a full implementation, you might want to:
        // 1. Clear any pending tasks
        // 2. Reset worker state
        // 3. Optionally restart workers if needed

        // For now, this is a placeholder
        Ok(())
    }
}

impl<D, C> Drop for PersistentWorkerPool<D, C>
where
    D: Dataset + Clone + Send + Sync + 'static,
    C: Collate<D::Item> + Clone + Send + Sync + 'static,
    D::Item: Send + 'static,
    C::Output: Send + 'static,
{
    fn drop(&mut self) {
        // Ensure workers are shutdown when pool is dropped
        let _ = self.shutdown();
    }
}

/// Configuration for worker pools
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Maximum number of pending tasks for multi-process iterator
    pub max_pending_tasks: Option<usize>,
    /// Whether to use persistent workers
    pub persistent: bool,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            num_workers: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            max_pending_tasks: None,
            persistent: false,
        }
    }
}

impl WorkerConfig {
    /// Create a new worker configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of workers
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Set the maximum number of pending tasks
    pub fn max_pending_tasks(mut self, max_pending: usize) -> Self {
        self.max_pending_tasks = Some(max_pending);
        self
    }

    /// Enable persistent workers
    pub fn persistent(mut self, persistent: bool) -> Self {
        self.persistent = persistent;
        self
    }
}

/// Utility functions for worker management
pub mod utils {
    use super::*;

    /// Determine optimal number of workers based on system capabilities
    ///
    /// # Arguments
    ///
    /// * `cpu_intensive` - Whether the workload is CPU-intensive
    ///
    /// # Returns
    ///
    /// Recommended number of worker threads
    pub fn optimal_worker_count(cpu_intensive: bool) -> usize {
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        if cpu_intensive {
            // For CPU-intensive tasks, use fewer workers to avoid over-subscription
            (cpu_count * 3 / 4).max(1)
        } else {
            // For I/O-bound tasks, can use more workers
            cpu_count * 2
        }
    }

    /// Create a worker configuration optimized for training
    ///
    /// # Returns
    ///
    /// A WorkerConfig optimized for training workloads
    pub fn training_config() -> WorkerConfig {
        WorkerConfig::new()
            .num_workers(optimal_worker_count(false))
            .persistent(true)
            .max_pending_tasks(optimal_worker_count(false) * 3)
    }

    /// Create a worker configuration optimized for inference
    ///
    /// # Returns
    ///
    /// A WorkerConfig optimized for inference workloads
    pub fn inference_config() -> WorkerConfig {
        WorkerConfig::new()
            .num_workers(optimal_worker_count(true))
            .persistent(false)
            .max_pending_tasks(optimal_worker_count(true) * 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{collate::DefaultCollate, dataset::TensorDataset};

    #[test]
    fn test_worker_pool_creation() {
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();
        let dataset = Arc::new(TensorDataset::from_tensor(tensor));
        let collate_fn = Arc::new(DefaultCollate);
        let worker_pool = WorkerPool::new(dataset, collate_fn, 2);

        assert_eq!(worker_pool.num_workers(), 2);
    }

    #[test]
    fn test_worker_pool_task_submission() {
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();
        let dataset = Arc::new(TensorDataset::from_tensor(tensor));
        let collate_fn = Arc::new(DefaultCollate);
        let worker_pool = WorkerPool::new(dataset, collate_fn, 2);

        assert!(worker_pool.submit_task(0, vec![0, 1]).is_ok());
    }

    #[test]
    fn test_persistent_worker_pool_creation() {
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();
        let dataset = Arc::new(TensorDataset::from_tensor(tensor));
        let collate_fn = Arc::new(DefaultCollate);
        let worker_pool = PersistentWorkerPool::new(dataset, collate_fn, 2);

        assert_eq!(worker_pool.num_workers(), 2);
        assert!(!worker_pool.is_shutdown());
    }

    #[test]
    fn test_persistent_worker_pool_shutdown() {
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();
        let dataset = Arc::new(TensorDataset::from_tensor(tensor));
        let collate_fn = Arc::new(DefaultCollate);
        let worker_pool = PersistentWorkerPool::new(dataset, collate_fn, 2);

        assert!(worker_pool.shutdown().is_ok());
        assert!(worker_pool.is_shutdown());
    }

    #[test]
    fn test_worker_config() {
        let config = WorkerConfig::new()
            .num_workers(4)
            .max_pending_tasks(8)
            .persistent(true);

        assert_eq!(config.num_workers, 4);
        assert_eq!(config.max_pending_tasks, Some(8));
        assert!(config.persistent);
    }

    #[test]
    fn test_optimal_worker_count() {
        let cpu_intensive_count = utils::optimal_worker_count(true);
        let io_bound_count = utils::optimal_worker_count(false);

        assert!(cpu_intensive_count > 0);
        assert!(io_bound_count > 0);
        assert!(io_bound_count >= cpu_intensive_count);
    }

    #[test]
    fn test_training_config() {
        let config = utils::training_config();

        assert!(config.num_workers > 0);
        assert!(config.persistent);
        assert!(config.max_pending_tasks.is_some());
    }

    #[test]
    fn test_inference_config() {
        let config = utils::inference_config();

        assert!(config.num_workers > 0);
        assert!(!config.persistent);
        assert!(config.max_pending_tasks.is_some());
    }
}
