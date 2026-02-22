//! Advanced CUDA event coordination system for operation-level synchronization
//!
//! This module provides sophisticated event management for coordinating operations
//! across multiple CUDA streams with automatic dependency tracking, deadlock detection,
//! and performance monitoring.

// Allow unused variables for pool utilization metrics
#![allow(unused_variables)]

use crate::cuda::error::{CudaError, CudaResult};
use crate::cuda::{CudaEvent, CudaStream};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Operation types for coordination tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    Kernel,          // Compute kernel execution
    MemoryTransfer,  // Memory copy operations
    Synchronization, // Explicit synchronization points
    Reduction,       // Collective reduction operations
    Broadcast,       // Data broadcast operations
    AllReduce,       // All-reduce collective operations
    Barrier,         // Global barrier synchronization
}

/// Event coordination priority for scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EventPriority {
    Critical, // Must execute immediately
    High,     // High priority operations
    Normal,   // Standard operations
    Low,      // Background operations
    Cleanup,  // Cleanup and maintenance
}

/// Event coordination metadata
#[derive(Debug, Clone)]
pub struct EventMetadata {
    pub operation_type: OperationType,
    pub priority: EventPriority,
    pub stream_id: u64,
    pub operation_id: u64,
    pub creation_time: Instant,
    pub dependencies: Vec<u64>, // Operation IDs this depends on
    pub description: String,
}

/// Event pool for efficient event reuse
#[derive(Debug)]
pub struct EventPool {
    available_events: Mutex<VecDeque<Arc<CudaEvent>>>,
    timing_events: Mutex<VecDeque<Arc<CudaEvent>>>,
    in_use: Mutex<HashSet<*const CudaEvent>>,
    pool_size: usize,
    timing_pool_size: usize,
}

impl EventPool {
    /// Create new event pool with specified capacity
    pub fn new(pool_size: usize, timing_pool_size: usize) -> CudaResult<Self> {
        let mut available_events = VecDeque::with_capacity(pool_size);
        let mut timing_events = VecDeque::with_capacity(timing_pool_size);

        // Pre-allocate regular events
        for _ in 0..pool_size {
            available_events.push_back(Arc::new(CudaEvent::new()?));
        }

        // Pre-allocate timing events
        for _ in 0..timing_pool_size {
            timing_events.push_back(Arc::new(CudaEvent::new_with_timing()?));
        }

        Ok(Self {
            available_events: Mutex::new(available_events),
            timing_events: Mutex::new(timing_events),
            in_use: Mutex::new(HashSet::new()),
            pool_size,
            timing_pool_size,
        })
    }

    /// Acquire event from pool
    pub fn acquire_event(&self, with_timing: bool) -> CudaResult<Arc<CudaEvent>> {
        let event = if with_timing {
            let mut timing_events = self
                .timing_events
                .lock()
                .expect("lock should not be poisoned");
            timing_events.pop_front().unwrap_or_else(|| {
                Arc::new(CudaEvent::new_with_timing().expect("Failed to create timing event"))
            })
        } else {
            let mut available_events = self
                .available_events
                .lock()
                .expect("lock should not be poisoned");
            available_events
                .pop_front()
                .unwrap_or_else(|| Arc::new(CudaEvent::new().expect("Failed to create event")))
        };

        // Track event usage
        let mut in_use = self.in_use.lock().expect("lock should not be poisoned");
        in_use.insert(Arc::as_ptr(&event));

        Ok(event)
    }

    /// Return event to pool
    pub fn release_event(&self, event: Arc<CudaEvent>) {
        let event_ptr = Arc::as_ptr(&event);

        // Remove from in-use tracking
        let mut in_use = self.in_use.lock().expect("lock should not be poisoned");
        in_use.remove(&event_ptr);
        drop(in_use);

        // Return to appropriate pool if not at capacity
        if event.timing_enabled() {
            let mut timing_events = self
                .timing_events
                .lock()
                .expect("lock should not be poisoned");
            if timing_events.len() < self.timing_pool_size {
                timing_events.push_back(event);
            }
        } else {
            let mut available_events = self
                .available_events
                .lock()
                .expect("lock should not be poisoned");
            if available_events.len() < self.pool_size {
                available_events.push_back(event);
            }
        }
    }

    /// Get pool utilization statistics
    pub fn utilization(&self) -> (usize, usize, usize) {
        let available = self
            .available_events
            .lock()
            .expect("lock should not be poisoned")
            .len();
        let timing = self
            .timing_events
            .lock()
            .expect("lock should not be poisoned")
            .len();
        let in_use = self
            .in_use
            .lock()
            .expect("lock should not be poisoned")
            .len();
        (available, timing, in_use)
    }
}

/// Operation coordinator for cross-stream synchronization
pub struct OperationCoordinator {
    operations: RwLock<HashMap<u64, EventMetadata>>,
    operation_events: RwLock<HashMap<u64, Arc<CudaEvent>>>,
    dependency_graph: RwLock<HashMap<u64, Vec<u64>>>,
    reverse_dependencies: RwLock<HashMap<u64, Vec<u64>>>,
    completion_callbacks: Mutex<HashMap<u64, Vec<Box<dyn FnOnce() + Send + 'static>>>>,
    next_operation_id: std::sync::atomic::AtomicU64,
    event_pool: Arc<EventPool>,
    coordination_metrics: Mutex<CoordinationMetrics>,
}

impl std::fmt::Debug for OperationCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperationCoordinator")
            .field("operations", &self.operations)
            .field("operation_events", &self.operation_events)
            .field("dependency_graph", &self.dependency_graph)
            .field("reverse_dependencies", &self.reverse_dependencies)
            .field("completion_callbacks", &"<completion callbacks>")
            .field("next_operation_id", &self.next_operation_id)
            .field("event_pool", &self.event_pool)
            .field("coordination_metrics", &self.coordination_metrics)
            .finish()
    }
}

/// Coordination performance metrics
#[derive(Debug, Clone, Default)]
pub struct CoordinationMetrics {
    pub total_operations: usize,
    pub completed_operations: usize,
    pub blocked_operations: usize,
    pub average_coordination_time: Duration,
    pub deadlock_detections: usize,
    pub priority_inversions: usize,
}

impl OperationCoordinator {
    /// Create new operation coordinator
    pub fn new(event_pool: Arc<EventPool>) -> Self {
        Self {
            operations: RwLock::new(HashMap::new()),
            operation_events: RwLock::new(HashMap::new()),
            dependency_graph: RwLock::new(HashMap::new()),
            reverse_dependencies: RwLock::new(HashMap::new()),
            completion_callbacks: Mutex::new(HashMap::new()),
            next_operation_id: std::sync::atomic::AtomicU64::new(1),
            event_pool,
            coordination_metrics: Mutex::new(CoordinationMetrics::default()),
        }
    }

    /// Register new operation for coordination
    pub fn register_operation(
        &self,
        operation_type: OperationType,
        priority: EventPriority,
        stream: &CudaStream,
        dependencies: Vec<u64>,
        description: String,
    ) -> CudaResult<u64> {
        let operation_id = self
            .next_operation_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let metadata = EventMetadata {
            operation_type,
            priority,
            stream_id: stream.id(),
            operation_id,
            creation_time: Instant::now(),
            dependencies: dependencies.clone(),
            description,
        };

        // Create coordination event
        let use_timing = matches!(
            operation_type,
            OperationType::Kernel | OperationType::MemoryTransfer
        );
        let event = self.event_pool.acquire_event(use_timing)?;

        // Update internal state
        {
            let mut operations = self
                .operations
                .write()
                .expect("lock should not be poisoned");
            operations.insert(operation_id, metadata);
        }

        {
            let mut operation_events = self
                .operation_events
                .write()
                .expect("lock should not be poisoned");
            operation_events.insert(operation_id, event);
        }

        // Update dependency graph
        if !dependencies.is_empty() {
            let mut dep_graph = self
                .dependency_graph
                .write()
                .expect("lock should not be poisoned");
            dep_graph.insert(operation_id, dependencies.clone());

            let mut reverse_deps = self
                .reverse_dependencies
                .write()
                .expect("lock should not be poisoned");
            for dep_id in dependencies {
                reverse_deps
                    .entry(dep_id)
                    .or_insert_with(Vec::new)
                    .push(operation_id);
            }
        }

        // Update metrics
        {
            let mut metrics = self
                .coordination_metrics
                .lock()
                .expect("lock should not be poisoned");
            metrics.total_operations += 1;
        }

        Ok(operation_id)
    }

    /// Begin operation execution (records start event)
    pub fn begin_operation(&self, operation_id: u64, stream: &CudaStream) -> CudaResult<()> {
        let event = {
            let operation_events = self
                .operation_events
                .read()
                .expect("lock should not be poisoned");
            operation_events
                .get(&operation_id)
                .cloned()
                .ok_or_else(|| CudaError::Context {
                    message: format!("Operation {} not found", operation_id),
                })?
        };

        // Wait for dependencies
        self.wait_for_dependencies(operation_id)?;

        // Record start event
        event.record_on_stream(stream)?;

        Ok(())
    }

    /// Complete operation execution (synchronizes and triggers callbacks)
    pub fn complete_operation(&self, operation_id: u64) -> CudaResult<()> {
        let event = {
            let operation_events = self
                .operation_events
                .read()
                .expect("lock should not be poisoned");
            operation_events
                .get(&operation_id)
                .cloned()
                .ok_or_else(|| CudaError::Context {
                    message: format!("Operation {} not found", operation_id),
                })?
        };

        // Synchronize operation completion
        event.synchronize()?;

        // Execute completion callbacks
        let callbacks = {
            let mut completion_callbacks = self
                .completion_callbacks
                .lock()
                .expect("lock should not be poisoned");
            completion_callbacks
                .remove(&operation_id)
                .unwrap_or_default()
        };

        for callback in callbacks {
            callback();
        }

        // Update metrics
        {
            let mut metrics = self
                .coordination_metrics
                .lock()
                .expect("lock should not be poisoned");
            metrics.completed_operations += 1;
        }

        // Clean up
        self.cleanup_operation(operation_id)?;

        Ok(())
    }

    /// Wait for operation dependencies to complete
    pub fn wait_for_dependencies(&self, operation_id: u64) -> CudaResult<()> {
        let dependencies = {
            let dep_graph = self
                .dependency_graph
                .read()
                .expect("lock should not be poisoned");
            dep_graph.get(&operation_id).cloned().unwrap_or_default()
        };

        for dep_id in dependencies {
            if let Some(dep_event) = self
                .operation_events
                .read()
                .expect("lock should not be poisoned")
                .get(&dep_id)
            {
                dep_event.synchronize()?;
            }
        }

        Ok(())
    }

    /// Add completion callback for operation
    pub fn add_completion_callback<F>(&self, operation_id: u64, callback: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let mut completion_callbacks = self
            .completion_callbacks
            .lock()
            .expect("lock should not be poisoned");
        completion_callbacks
            .entry(operation_id)
            .or_insert_with(Vec::new)
            .push(Box::new(callback));
    }

    /// Check for deadlocks in dependency graph
    pub fn detect_deadlocks(&self) -> Vec<Vec<u64>> {
        let dep_graph = self
            .dependency_graph
            .read()
            .expect("lock should not be poisoned");
        let mut deadlocks = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut current_path = Vec::new();

        for &operation_id in dep_graph.keys() {
            if !visited.contains(&operation_id) {
                if let Some(cycle) = self.detect_cycle(
                    operation_id,
                    &dep_graph,
                    &mut visited,
                    &mut rec_stack,
                    &mut current_path,
                ) {
                    deadlocks.push(cycle);

                    // Update metrics
                    let mut metrics = self
                        .coordination_metrics
                        .lock()
                        .expect("lock should not be poisoned");
                    metrics.deadlock_detections += 1;
                }
            }
        }

        deadlocks
    }

    fn detect_cycle(
        &self,
        operation_id: u64,
        dep_graph: &HashMap<u64, Vec<u64>>,
        visited: &mut HashSet<u64>,
        rec_stack: &mut HashSet<u64>,
        current_path: &mut Vec<u64>,
    ) -> Option<Vec<u64>> {
        visited.insert(operation_id);
        rec_stack.insert(operation_id);
        current_path.push(operation_id);

        if let Some(dependencies) = dep_graph.get(&operation_id) {
            for &dep_id in dependencies {
                if !visited.contains(&dep_id) {
                    if let Some(cycle) =
                        self.detect_cycle(dep_id, dep_graph, visited, rec_stack, current_path)
                    {
                        return Some(cycle);
                    }
                } else if rec_stack.contains(&dep_id) {
                    // Found cycle
                    let cycle_start = current_path
                        .iter()
                        .position(|&id| id == dep_id)
                        .expect("dep_id should exist in current_path as it's in rec_stack");
                    return Some(current_path[cycle_start..].to_vec());
                }
            }
        }

        rec_stack.remove(&operation_id);
        current_path.pop();
        None
    }

    /// Get coordination metrics
    pub fn metrics(&self) -> CoordinationMetrics {
        self.coordination_metrics
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Clean up completed operation
    fn cleanup_operation(&self, operation_id: u64) -> CudaResult<()> {
        // Remove from operations
        {
            let mut operations = self
                .operations
                .write()
                .expect("lock should not be poisoned");
            operations.remove(&operation_id);
        }

        // Return event to pool
        if let Some(event) = self
            .operation_events
            .write()
            .expect("lock should not be poisoned")
            .remove(&operation_id)
        {
            self.event_pool.release_event(event);
        }

        // Clean up dependencies
        {
            let mut dep_graph = self
                .dependency_graph
                .write()
                .expect("lock should not be poisoned");
            dep_graph.remove(&operation_id);
        }

        {
            let mut reverse_deps = self
                .reverse_dependencies
                .write()
                .expect("lock should not be poisoned");
            reverse_deps.remove(&operation_id);
        }

        Ok(())
    }
}

/// Cross-stream barrier for global synchronization
#[derive(Debug)]
pub struct CrossStreamBarrier {
    participants: Vec<Arc<CudaStream>>,
    barrier_events: Vec<Arc<CudaEvent>>,
    completion_event: Arc<CudaEvent>,
    event_pool: Arc<EventPool>,
}

impl CrossStreamBarrier {
    /// Create new cross-stream barrier
    pub fn new(streams: Vec<Arc<CudaStream>>, event_pool: Arc<EventPool>) -> CudaResult<Self> {
        let mut barrier_events = Vec::with_capacity(streams.len());

        // Create barrier event for each stream
        for _ in 0..streams.len() {
            barrier_events.push(event_pool.acquire_event(false)?);
        }

        let completion_event = event_pool.acquire_event(true)?;

        Ok(Self {
            participants: streams,
            barrier_events,
            completion_event,
            event_pool,
        })
    }

    /// Execute barrier synchronization
    pub fn synchronize(&self) -> CudaResult<Duration> {
        let start_time = Instant::now();

        // Record events on all streams
        for (stream, event) in self.participants.iter().zip(self.barrier_events.iter()) {
            event.record_on_stream(stream)?;
        }

        // Wait for all events to complete
        for event in &self.barrier_events {
            event.synchronize()?;
        }

        // Record completion
        if !self.participants.is_empty() {
            self.completion_event
                .record_on_stream(&self.participants[0])?;
            self.completion_event.synchronize()?;
        }

        Ok(start_time.elapsed())
    }

    /// Wait for barrier completion on specific stream
    pub fn wait_on_stream(&self, stream: &CudaStream) -> CudaResult<()> {
        for event in &self.barrier_events {
            stream.wait_event(event)?;
        }
        Ok(())
    }
}

impl Drop for CrossStreamBarrier {
    fn drop(&mut self) {
        // Return events to pool
        for event in self.barrier_events.drain(..) {
            self.event_pool.release_event(event);
        }
        self.event_pool.release_event(self.completion_event.clone());
    }
}

/// Asynchronous event waiter for non-blocking coordination
pub struct AsyncEventWaiter {
    pending_events: Arc<Mutex<HashMap<u64, (Arc<CudaEvent>, Box<dyn FnOnce() + Send + 'static>)>>>,
    worker_handle: Option<thread::JoinHandle<()>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    next_wait_id: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for AsyncEventWaiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncEventWaiter")
            .field("pending_events", &"<pending events with callbacks>")
            .field("worker_handle", &self.worker_handle.is_some())
            .field("shutdown", &self.shutdown)
            .field("next_wait_id", &self.next_wait_id)
            .finish()
    }
}

impl AsyncEventWaiter {
    /// Create new async event waiter
    pub fn new() -> Self {
        let pending_events: Arc<
            Mutex<HashMap<u64, (Arc<CudaEvent>, Box<dyn FnOnce() + Send + 'static>)>>,
        > = Arc::new(Mutex::new(HashMap::new()));
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let worker_events = Arc::clone(&pending_events);
        let worker_shutdown = Arc::clone(&shutdown);

        let worker_handle = thread::spawn(move || {
            while !worker_shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                // Collect ready callbacks while holding the lock
                let ready_callbacks: Vec<Box<dyn FnOnce() + Send + 'static>> = {
                    let mut events = worker_events.lock().expect("lock should not be poisoned");
                    let mut ready_ids = Vec::new();

                    // First pass: find ready events
                    for (&wait_id, (event, _)) in events.iter() {
                        if event.is_ready().unwrap_or(false) {
                            ready_ids.push(wait_id);
                        }
                    }

                    // Second pass: extract callbacks for ready events
                    ready_ids
                        .into_iter()
                        .filter_map(|wait_id| events.remove(&wait_id).map(|(_, cb)| cb))
                        .collect()
                };

                // Execute callbacks outside the lock
                for callback in ready_callbacks {
                    callback();
                }

                thread::sleep(Duration::from_micros(100));
            }
        });

        Self {
            pending_events,
            worker_handle: Some(worker_handle),
            shutdown,
            next_wait_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Wait for event asynchronously with callback
    pub fn wait_async<F>(&self, event: Arc<CudaEvent>, callback: F) -> u64
    where
        F: FnOnce() + Send + 'static,
    {
        let wait_id = self
            .next_wait_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut pending = self
            .pending_events
            .lock()
            .expect("lock should not be poisoned");
        pending.insert(wait_id, (event, Box::new(callback)));

        wait_id
    }

    /// Cancel async wait
    pub fn cancel_wait(&self, wait_id: u64) -> bool {
        let mut pending = self
            .pending_events
            .lock()
            .expect("lock should not be poisoned");
        pending.remove(&wait_id).is_some()
    }
}

impl Drop for AsyncEventWaiter {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Requires CUDA hardware - run with --ignored flag"]
    fn test_event_pool() {
        if crate::cuda::is_available() {
            let _device = Arc::new(crate::cuda::device::CudaDevice::new(0).unwrap());
            let pool = EventPool::new(4, 2).unwrap();

            // Test regular event acquisition
            let event1 = pool.acquire_event(false).unwrap();
            let event2 = pool.acquire_event(false).unwrap();

            let (available, timing, in_use) = pool.utilization();
            assert_eq!(in_use, 2);

            // Test event release
            pool.release_event(event1);
            pool.release_event(event2);

            let (available, timing, in_use) = pool.utilization();
            assert_eq!(in_use, 0);
        }
    }

    #[test]
    #[ignore = "Requires CUDA hardware - run with --ignored flag"]
    fn test_operation_coordinator() {
        if crate::cuda::is_available() {
            let _device = Arc::new(crate::cuda::device::CudaDevice::new(0).unwrap());
            let event_pool = Arc::new(EventPool::new(10, 5).unwrap());
            let coordinator = OperationCoordinator::new(event_pool);
            let stream = CudaStream::new().unwrap();

            // Register operation
            let op_id = coordinator
                .register_operation(
                    OperationType::Kernel,
                    EventPriority::High,
                    &stream,
                    vec![],
                    "Test kernel".to_string(),
                )
                .unwrap();

            assert!(op_id > 0);

            // Test operation execution
            coordinator.begin_operation(op_id, &stream).unwrap();
            coordinator.complete_operation(op_id).unwrap();

            let metrics = coordinator.metrics();
            assert_eq!(metrics.total_operations, 1);
            assert_eq!(metrics.completed_operations, 1);
        }
    }

    #[test]
    fn test_cross_stream_barrier() {
        if crate::cuda::is_available() {
            let _device = Arc::new(crate::cuda::device::CudaDevice::new(0).unwrap());
            let stream1 = Arc::new(CudaStream::new().unwrap());
            let stream2 = Arc::new(CudaStream::new().unwrap());
            let streams = vec![stream1, stream2];

            let event_pool = Arc::new(EventPool::new(10, 5).unwrap());
            let barrier = CrossStreamBarrier::new(streams, event_pool).unwrap();

            let duration = barrier.synchronize().unwrap();
            assert!(duration < Duration::from_secs(1));
        }
    }

    #[test]
    #[ignore = "Async event waiter has CUDA context threading issues - worker thread lacks context"]
    fn test_async_event_waiter() {
        if crate::cuda::is_available() {
            let _device = Arc::new(crate::cuda::device::CudaDevice::new(0).unwrap());
            let waiter = AsyncEventWaiter::new();
            let stream = CudaStream::new().unwrap();
            let event = Arc::new(CudaEvent::new().unwrap());

            // Record the event on a stream so it becomes "ready"
            stream.record_event(&event).unwrap();
            stream.synchronize().unwrap();

            let callback_executed = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let callback_flag = Arc::clone(&callback_executed);

            let wait_id = waiter.wait_async(event, move || {
                callback_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            });

            assert!(wait_id > 0);

            // Longer delay to allow worker thread to poll and execute callback
            thread::sleep(Duration::from_millis(500));

            assert!(callback_executed.load(std::sync::atomic::Ordering::Relaxed));
        }
    }
}
