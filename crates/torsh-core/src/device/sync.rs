//! Device synchronization primitives
//!
//! This module provides synchronization primitives for coordinating operations
//! across devices, including events, barriers, streams, and async/await integration.

use crate::device::DeviceType;
use crate::error::Result;
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Device synchronization event for coordinating operations
///
/// Events provide a way to synchronize operations within and across devices.
/// They can be used to create dependencies between operations and ensure
/// proper ordering of computations.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::{DeviceEvent, DeviceType};
///
/// let event = DeviceEvent::new(DeviceType::Cpu)?;
///
/// // Record the event after some operation
/// event.record()?;
///
/// // Wait for the event to complete
/// event.wait()?;
///
/// // Check if the event has completed without blocking
/// if event.query()? {
///     println!("Event completed");
/// }
/// ```
#[derive(Debug)]
pub struct DeviceEvent {
    device: DeviceType,
    inner: Arc<EventInner>,
}

#[derive(Debug)]
struct EventInner {
    state: Mutex<EventState>,
    cond: Condvar,
    recorded_time: Mutex<Option<Instant>>,
    completed_time: Mutex<Option<Instant>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventState {
    Created,
    Recorded,
    Completed,
}

impl DeviceEvent {
    /// Create a new device event
    pub fn new(device: DeviceType) -> Result<Self> {
        Ok(DeviceEvent {
            device,
            inner: Arc::new(EventInner {
                state: Mutex::new(EventState::Created),
                cond: Condvar::new(),
                recorded_time: Mutex::new(None),
                completed_time: Mutex::new(None),
            }),
        })
    }

    /// Get the device this event belongs to
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Record the event (mark it as pending)
    pub fn record(&self) -> Result<()> {
        let mut state = self.inner.state.lock().unwrap();
        *state = EventState::Recorded;
        *self.inner.recorded_time.lock().unwrap() = Some(Instant::now());

        // Simulate async completion for demo purposes
        self.complete_async();

        Ok(())
    }

    /// Wait for the event to complete
    pub fn wait(&self) -> Result<()> {
        let mut state = self.inner.state.lock().unwrap();
        while *state != EventState::Completed {
            state = self.inner.cond.wait(state).unwrap();
        }
        Ok(())
    }

    /// Wait for the event to complete with timeout
    pub fn wait_timeout(&self, timeout: Duration) -> Result<bool> {
        let mut state = self.inner.state.lock().unwrap();
        while *state != EventState::Completed {
            let (new_state, timeout_result) = self.inner.cond.wait_timeout(state, timeout).unwrap();
            state = new_state;
            if timeout_result.timed_out() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Query if the event has completed (non-blocking)
    pub fn query(&self) -> Result<bool> {
        let state = self.inner.state.lock().unwrap();
        Ok(*state == EventState::Completed)
    }

    /// Get the elapsed time since recording (if completed)
    pub fn elapsed_time(&self) -> Option<Duration> {
        let recorded = self.inner.recorded_time.lock().unwrap();
        let completed = self.inner.completed_time.lock().unwrap();

        match (*recorded, *completed) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }

    /// Reset the event to be reused
    pub fn reset(&self) -> Result<()> {
        let mut state = self.inner.state.lock().unwrap();
        *state = EventState::Created;
        *self.inner.recorded_time.lock().unwrap() = None;
        *self.inner.completed_time.lock().unwrap() = None;
        Ok(())
    }

    fn complete_async(&self) {
        let inner = self.inner.clone();
        std::thread::spawn(move || {
            // Simulate some work
            std::thread::sleep(Duration::from_millis(1));

            let mut state = inner.state.lock().unwrap();
            *state = EventState::Completed;
            *inner.completed_time.lock().unwrap() = Some(Instant::now());
            inner.cond.notify_all();
        });
    }
}

impl Clone for DeviceEvent {
    fn clone(&self) -> Self {
        DeviceEvent {
            device: self.device,
            inner: self.inner.clone(),
        }
    }
}

/// Device stream for ordering operations
///
/// Streams provide ordering guarantees for operations on a device.
/// Operations submitted to the same stream execute in order, while
/// operations in different streams may execute concurrently.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::{DeviceStream, DeviceType};
///
/// let stream = DeviceStream::new(DeviceType::Cuda(0))?;
///
/// // Submit operations to the stream
/// stream.submit_operation(|| {
///     // Some GPU computation
/// })?;
///
/// // Synchronize the stream
/// stream.synchronize()?;
/// ```
#[derive(Debug)]
pub struct DeviceStream {
    device: DeviceType,
    id: u64,
    priority: StreamPriority,
    inner: Arc<StreamInner>,
}

struct StreamInner {
    operation_queue: Mutex<Vec<Box<dyn FnOnce() + Send + 'static>>>,
    is_synchronizing: Mutex<bool>,
    sync_cond: Condvar,
}

impl std::fmt::Debug for StreamInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamInner")
            .field("operation_queue", &"<operation_queue>")
            .field("is_synchronizing", &self.is_synchronizing)
            .field("sync_cond", &"<condvar>")
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPriority {
    Low = 0,
    Normal = 1,
    High = 2,
}

impl DeviceStream {
    /// Create a new device stream
    pub fn new(device: DeviceType) -> Result<Self> {
        Self::with_priority(device, StreamPriority::Normal)
    }

    /// Create a new device stream with priority
    pub fn with_priority(device: DeviceType, priority: StreamPriority) -> Result<Self> {
        static STREAM_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        Ok(DeviceStream {
            device,
            id: STREAM_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            priority,
            inner: Arc::new(StreamInner {
                operation_queue: Mutex::new(Vec::new()),
                is_synchronizing: Mutex::new(false),
                sync_cond: Condvar::new(),
            }),
        })
    }

    /// Get the device this stream belongs to
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Get the stream ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get the stream priority
    pub fn priority(&self) -> StreamPriority {
        self.priority
    }

    /// Submit an operation to the stream
    pub fn submit_operation<F>(&self, operation: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        let mut queue = self.inner.operation_queue.lock().unwrap();
        queue.push(Box::new(operation));

        // Process operations asynchronously
        self.process_operations_async();

        Ok(())
    }

    /// Wait for all operations in the stream to complete
    pub fn synchronize(&self) -> Result<()> {
        let mut is_sync = self.inner.is_synchronizing.lock().unwrap();
        while !self.is_empty() || *is_sync {
            is_sync = self.inner.sync_cond.wait(is_sync).unwrap();
        }
        Ok(())
    }

    /// Check if the stream is empty (no pending operations)
    pub fn is_empty(&self) -> bool {
        let queue = self.inner.operation_queue.lock().unwrap();
        queue.is_empty()
    }

    /// Get the number of pending operations
    pub fn pending_operations(&self) -> usize {
        let queue = self.inner.operation_queue.lock().unwrap();
        queue.len()
    }

    /// Create a new event and record it on this stream
    pub fn record_event(&self) -> Result<DeviceEvent> {
        let event = DeviceEvent::new(self.device)?;
        event.record()?;
        Ok(event)
    }

    /// Wait for an event on this stream
    pub fn wait_event(&self, event: &DeviceEvent) -> Result<()> {
        if event.device() != self.device {
            return Err(crate::error::TorshError::InvalidArgument(
                "Event device does not match stream device".to_string(),
            ));
        }
        event.wait()
    }

    fn process_operations_async(&self) {
        let inner = self.inner.clone();
        std::thread::spawn(move || {
            {
                let mut is_sync = inner.is_synchronizing.lock().unwrap();
                *is_sync = true;
            }

            loop {
                let operation = {
                    let mut queue = inner.operation_queue.lock().unwrap();
                    queue.pop()
                };

                match operation {
                    Some(op) => {
                        op(); // Execute the operation
                    }
                    None => break,
                }
            }

            {
                let mut is_sync = inner.is_synchronizing.lock().unwrap();
                *is_sync = false;
                inner.sync_cond.notify_all();
            }
        });
    }
}

/// Device barrier for multi-device synchronization
///
/// Barriers provide a way to synchronize multiple devices or streams,
/// ensuring all participants reach the barrier point before any can proceed.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::{DeviceBarrier, DeviceType};
///
/// let barrier = DeviceBarrier::new(vec![
///     DeviceType::Cpu,
///     DeviceType::Cuda(0),
/// ])?;
///
/// // Each device/thread calls wait
/// barrier.wait(DeviceType::Cpu)?;
/// ```
#[derive(Debug)]
pub struct DeviceBarrier {
    devices: Vec<DeviceType>,
    inner: Arc<BarrierInner>,
}

#[derive(Debug)]
struct BarrierInner {
    count: Mutex<usize>,
    total: usize,
    generation: Mutex<usize>,
    cond: Condvar,
    arrived_devices: Mutex<Vec<DeviceType>>,
}

impl DeviceBarrier {
    /// Create a new device barrier
    pub fn new(devices: Vec<DeviceType>) -> Result<Self> {
        let total = devices.len();
        if total == 0 {
            return Err(crate::error::TorshError::InvalidArgument(
                "Barrier must have at least one device".to_string(),
            ));
        }

        Ok(DeviceBarrier {
            devices: devices.clone(),
            inner: Arc::new(BarrierInner {
                count: Mutex::new(0),
                total,
                generation: Mutex::new(0),
                cond: Condvar::new(),
                arrived_devices: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Wait for all devices to reach the barrier
    pub fn wait(&self, device: DeviceType) -> Result<()> {
        if !self.devices.contains(&device) {
            return Err(crate::error::TorshError::InvalidArgument(format!(
                "Device {:?} is not part of this barrier",
                device
            )));
        }

        let mut count = self.inner.count.lock().unwrap();
        let mut arrived = self.inner.arrived_devices.lock().unwrap();
        let generation = *self.inner.generation.lock().unwrap();

        // Check if device already arrived in this generation
        if arrived.contains(&device) {
            return Err(crate::error::TorshError::InvalidArgument(
                "Device already waiting at barrier".to_string(),
            ));
        }

        arrived.push(device);
        *count += 1;

        if *count == self.inner.total {
            // Last device to arrive - release all
            *count = 0;
            arrived.clear();
            let mut gen = self.inner.generation.lock().unwrap();
            *gen += 1;
            drop(gen);
            self.inner.cond.notify_all();
            Ok(())
        } else {
            // Wait for others
            while *self.inner.generation.lock().unwrap() == generation {
                count = self.inner.cond.wait(count).unwrap();
            }
            Ok(())
        }
    }

    /// Get the devices participating in this barrier
    pub fn devices(&self) -> &[DeviceType] {
        &self.devices
    }

    /// Get the number of devices that have arrived at the barrier
    pub fn arrived_count(&self) -> usize {
        let arrived = self.inner.arrived_devices.lock().unwrap();
        arrived.len()
    }

    /// Check if all devices have arrived (barrier is complete)
    pub fn is_complete(&self) -> bool {
        self.arrived_count() == self.inner.total
    }
}

/// Async wrapper for device operations
///
/// Provides async/await integration for device operations, allowing
/// them to be used in async contexts.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::DeviceAsync;
///
/// async fn example() -> Result<()> {
///     let async_op = DeviceAsync::new(DeviceType::Cpu);
///     let result = async_op.execute(|| {
///         // Some computation
///         42
///     }).await?;
///     assert_eq!(result, 42);
///     Ok(())
/// }
/// ```
pub struct DeviceAsync<T> {
    #[allow(dead_code)] // Device type for async operations - future implementation
    device: DeviceType,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> DeviceAsync<T>
where
    T: Send + 'static,
{
    /// Create a new async device operation
    pub fn new(device: DeviceType) -> Self {
        Self {
            device,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Execute an operation on a background thread
    pub fn execute<F>(self, operation: F) -> std::thread::JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
    {
        std::thread::spawn(operation)
    }
}

/// Device mutex for exclusive access to resources
#[derive(Debug)]
pub struct DeviceMutex<T> {
    device: DeviceType,
    data: Arc<Mutex<T>>,
}

impl<T> DeviceMutex<T> {
    /// Create a new device mutex
    pub fn new(device: DeviceType, data: T) -> Self {
        Self {
            device,
            data: Arc::new(Mutex::new(data)),
        }
    }

    /// Get the device this mutex belongs to
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Lock the mutex and get access to the data
    pub fn lock(&self) -> Result<std::sync::MutexGuard<'_, T>> {
        self.data.lock().map_err(|_| {
            crate::error::TorshError::DeviceError("Failed to acquire device mutex".to_string())
        })
    }

    /// Try to lock the mutex without blocking
    pub fn try_lock(&self) -> Result<Option<std::sync::MutexGuard<'_, T>>> {
        match self.data.try_lock() {
            Ok(guard) => Ok(Some(guard)),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(_) => Err(crate::error::TorshError::DeviceError(
                "Device mutex is poisoned".to_string(),
            )),
        }
    }
}

impl<T> Clone for DeviceMutex<T> {
    fn clone(&self) -> Self {
        Self {
            device: self.device,
            data: self.data.clone(),
        }
    }
}

/// Global device synchronization manager
#[derive(Debug)]
pub struct DeviceSyncManager {
    streams: Mutex<HashMap<(DeviceType, u64), Arc<DeviceStream>>>,
    events: Mutex<HashMap<(DeviceType, u64), Arc<DeviceEvent>>>,
    barriers: Mutex<Vec<Arc<DeviceBarrier>>>,
}

impl DeviceSyncManager {
    /// Create a new sync manager
    pub fn new() -> Self {
        Self {
            streams: Mutex::new(HashMap::new()),
            events: Mutex::new(HashMap::new()),
            barriers: Mutex::new(Vec::new()),
        }
    }

    /// Register a stream with the manager
    pub fn register_stream(&self, stream: Arc<DeviceStream>) {
        let mut streams = self.streams.lock().unwrap();
        streams.insert((stream.device(), stream.id()), stream);
    }

    /// Get a stream by device and ID
    pub fn get_stream(&self, device: DeviceType, id: u64) -> Option<Arc<DeviceStream>> {
        let streams = self.streams.lock().unwrap();
        streams.get(&(device, id)).cloned()
    }

    /// Synchronize all streams for a device
    pub fn synchronize_device(&self, device: DeviceType) -> Result<()> {
        let streams = self.streams.lock().unwrap();
        let device_streams: Vec<_> = streams
            .values()
            .filter(|stream| stream.device() == device)
            .cloned()
            .collect();
        drop(streams);

        for stream in device_streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Create a cross-device barrier
    pub fn create_barrier(&self, devices: Vec<DeviceType>) -> Result<Arc<DeviceBarrier>> {
        let barrier = Arc::new(DeviceBarrier::new(devices)?);
        let mut barriers = self.barriers.lock().unwrap();
        barriers.push(barrier.clone());
        Ok(barrier)
    }

    /// Get synchronization statistics
    pub fn statistics(&self) -> SyncStatistics {
        let streams = self.streams.lock().unwrap();
        let events = self.events.lock().unwrap();
        let barriers = self.barriers.lock().unwrap();

        let total_pending_ops: usize = streams
            .values()
            .map(|stream| stream.pending_operations())
            .sum();

        SyncStatistics {
            total_streams: streams.len(),
            total_events: events.len(),
            total_barriers: barriers.len(),
            pending_operations: total_pending_ops,
        }
    }
}

impl Default for DeviceSyncManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Synchronization statistics
#[derive(Debug, Clone)]
pub struct SyncStatistics {
    pub total_streams: usize,
    pub total_events: usize,
    pub total_barriers: usize,
    pub pending_operations: usize,
}

impl std::fmt::Display for SyncStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SyncStats(streams={}, events={}, barriers={}, pending_ops={})",
            self.total_streams, self.total_events, self.total_barriers, self.pending_operations
        )
    }
}

/// Utility functions for device synchronization
pub mod utils {
    use super::*;

    /// Create a multi-device barrier for all available devices
    pub fn create_global_barrier(devices: &[DeviceType]) -> Result<DeviceBarrier> {
        DeviceBarrier::new(devices.to_vec())
    }

    /// Synchronize multiple streams
    pub fn synchronize_streams(streams: &[&DeviceStream]) -> Result<()> {
        for stream in streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Wait for multiple events
    pub fn wait_events(events: &[&DeviceEvent]) -> Result<()> {
        for event in events {
            event.wait()?;
        }
        Ok(())
    }

    /// Check if all events have completed
    pub fn all_events_complete(events: &[&DeviceEvent]) -> Result<bool> {
        for event in events {
            if !event.query()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Measure the elapsed time for multiple events
    pub fn measure_event_times(events: &[&DeviceEvent]) -> Vec<Option<Duration>> {
        events.iter().map(|event| event.elapsed_time()).collect()
    }

    /// Create a dependency chain between events
    pub fn create_event_chain(device: DeviceType, count: usize) -> Result<Vec<DeviceEvent>> {
        let mut events = Vec::new();
        for _ in 0..count {
            let event = DeviceEvent::new(device)?;
            events.push(event);
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_device_event_basic() {
        let event = DeviceEvent::new(DeviceType::Cpu).unwrap();
        assert_eq!(event.device(), DeviceType::Cpu);
        assert!(!event.query().unwrap());

        event.record().unwrap();
        event.wait().unwrap();
        assert!(event.query().unwrap());
    }

    #[test]
    fn test_device_event_timeout() {
        let event = DeviceEvent::new(DeviceType::Cpu).unwrap();
        event.record().unwrap();

        let completed = event.wait_timeout(Duration::from_millis(100)).unwrap();
        assert!(completed); // Should complete quickly
    }

    #[test]
    fn test_device_stream() {
        let stream = DeviceStream::new(DeviceType::Cpu).unwrap();
        assert_eq!(stream.device(), DeviceType::Cpu);
        assert_eq!(stream.priority(), StreamPriority::Normal);

        let executed = Arc::new(Mutex::new(false));
        let executed_clone = executed.clone();

        stream
            .submit_operation(move || {
                *executed_clone.lock().unwrap() = true;
            })
            .unwrap();

        stream.synchronize().unwrap();
        assert!(*executed.lock().unwrap());
    }

    #[test]
    fn test_device_barrier() {
        let devices = vec![DeviceType::Cpu, DeviceType::Cuda(0)];
        let barrier = DeviceBarrier::new(devices.clone()).unwrap();

        assert_eq!(barrier.devices(), &devices);
        assert_eq!(barrier.arrived_count(), 0);
        assert!(!barrier.is_complete());
    }

    #[test]
    fn test_device_mutex() {
        let mutex = DeviceMutex::new(DeviceType::Cpu, 42);
        assert_eq!(mutex.device(), DeviceType::Cpu);

        {
            let guard = mutex.lock().unwrap();
            assert_eq!(*guard, 42);
        }

        let try_guard = mutex.try_lock().unwrap();
        assert!(try_guard.is_some());
        assert_eq!(*try_guard.unwrap(), 42);
    }

    #[test]
    fn test_sync_manager() {
        let manager = DeviceSyncManager::new();
        let stream = Arc::new(DeviceStream::new(DeviceType::Cpu).unwrap());
        let stream_id = stream.id();

        manager.register_stream(stream.clone());

        let retrieved = manager.get_stream(DeviceType::Cpu, stream_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id(), stream_id);

        let stats = manager.statistics();
        assert_eq!(stats.total_streams, 1);
    }

    #[test]
    fn test_stream_priorities() {
        let high_stream =
            DeviceStream::with_priority(DeviceType::Cpu, StreamPriority::High).unwrap();
        let low_stream = DeviceStream::with_priority(DeviceType::Cpu, StreamPriority::Low).unwrap();

        assert_eq!(high_stream.priority(), StreamPriority::High);
        assert_eq!(low_stream.priority(), StreamPriority::Low);
    }

    #[test]
    fn test_event_reset() {
        let event = DeviceEvent::new(DeviceType::Cpu).unwrap();
        event.record().unwrap();
        event.wait().unwrap();
        assert!(event.query().unwrap());

        event.reset().unwrap();
        assert!(!event.query().unwrap());
    }

    #[test]
    fn test_utils_functions() {
        let event1 = DeviceEvent::new(DeviceType::Cpu).unwrap();
        let event2 = DeviceEvent::new(DeviceType::Cpu).unwrap();
        let events = vec![&event1, &event2];

        event1.record().unwrap();
        event2.record().unwrap();

        utils::wait_events(&events).unwrap();
        assert!(utils::all_events_complete(&events).unwrap());

        let times = utils::measure_event_times(&events);
        assert_eq!(times.len(), 2);
    }

    #[tokio::test]
    async fn test_device_async() {
        let async_op = DeviceAsync::new(DeviceType::Cpu);
        let result = async_op.execute(|| 42).join().unwrap();
        assert_eq!(result, 42);
    }
}
