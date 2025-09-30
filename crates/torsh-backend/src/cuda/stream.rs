//! Advanced CUDA stream management with async operations and performance optimization

use crate::cuda::memory::{CudaAllocation, MemoryAdvice, UnifiedAllocation};
use crate::error::{CudaError, CudaResult};
use cuda_sys::cudart::cudaStream_t;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Stream priority levels for scheduling optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamPriority {
    Low = 0,
    Normal = 1,
    High = 2,
}

/// Stream callback function type
type StreamCallback = Box<dyn FnOnce() + Send + 'static>;

/// Stream performance metrics
#[derive(Debug, Clone, Default)]
pub struct StreamMetrics {
    pub operations_count: usize,
    pub total_execution_time: Duration,
    pub memory_transfers: usize,
    pub kernel_launches: usize,
    pub average_latency: Duration,
    pub peak_memory_usage: usize,
}

/// Advanced CUDA stream wrapper with async operations and performance optimization
#[derive(Debug)]
pub struct CudaStream {
    stream: cust::Stream,
    id: u64,
    priority: StreamPriority,
    metrics: Arc<Mutex<StreamMetrics>>,
    callbacks: Arc<Mutex<Vec<StreamCallback>>>,
    dependency_events: Arc<Mutex<Vec<Arc<CudaEvent>>>>,
}

impl CudaStream {
    /// Create new CUDA stream with priority
    pub fn new() -> CudaResult<Self> {
        Self::new_with_priority(StreamPriority::Normal)
    }

    /// Create new CUDA stream with specified priority
    pub fn new_with_priority(priority: StreamPriority) -> CudaResult<Self> {
        let stream_flags = match priority {
            StreamPriority::High => cust::StreamFlags::NON_BLOCKING,
            StreamPriority::Normal => cust::StreamFlags::NON_BLOCKING,
            StreamPriority::Low => cust::StreamFlags::NON_BLOCKING,
        };

        let stream = cust::Stream::new(stream_flags, None)?;
        let id = stream.as_inner() as u64;

        Ok(Self {
            stream,
            id,
            priority,
            metrics: Arc::new(Mutex::new(StreamMetrics::default())),
            callbacks: Arc::new(Mutex::new(Vec::new())),
            dependency_events: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Create default stream (stream 0)
    pub fn default() -> CudaResult<Self> {
        let stream = cust::Stream::default();
        Ok(Self {
            stream,
            id: 0,
            priority: StreamPriority::Normal,
            metrics: Arc::new(Mutex::new(StreamMetrics::default())),
            callbacks: Arc::new(Mutex::new(Vec::new())),
            dependency_events: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Get stream ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get stream priority
    pub fn priority(&self) -> StreamPriority {
        self.priority
    }

    /// Get raw CUDA stream
    pub fn raw(&self) -> &cust::Stream {
        &self.stream
    }

    /// Get raw CUDA stream handle for FFI
    pub fn stream(&self) -> cudaStream_t {
        self.stream.as_inner() as *mut c_void as cudaStream_t
    }

    /// Get stream performance metrics
    pub fn metrics(&self) -> StreamMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Synchronize stream and execute callbacks
    pub fn synchronize(&self) -> CudaResult<()> {
        let start_time = Instant::now();
        self.stream.synchronize()?;

        // Execute any pending callbacks
        let callbacks = {
            let mut cb_vec = self.callbacks.lock().unwrap();
            std::mem::take(&mut *cb_vec)
        };

        for callback in callbacks {
            callback();
        }

        // Update metrics
        let elapsed = start_time.elapsed();
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_execution_time += elapsed;
        metrics.operations_count += 1;
        if metrics.operations_count > 0 {
            metrics.average_latency =
                metrics.total_execution_time / metrics.operations_count as u32;
        }

        Ok(())
    }

    /// Check if stream is ready
    pub fn is_ready(&self) -> CudaResult<bool> {
        match self.stream.query() {
            Ok(_) => {
                // Execute callbacks if stream is ready
                self.execute_callbacks_if_ready();
                Ok(true)
            }
            Err(cust::CudaError::NotReady) => Ok(false),
            Err(e) => Err(CudaError::Runtime(e)),
        }
    }

    /// Execute callbacks if stream is ready (non-blocking)
    fn execute_callbacks_if_ready(&self) {
        if let Ok(true) = self.stream.query() {
            let callbacks = {
                let mut cb_vec = self.callbacks.lock().unwrap();
                std::mem::take(&mut *cb_vec)
            };

            for callback in callbacks {
                callback();
            }
        }
    }

    /// Wait for event on this stream
    pub fn wait_event(&self, event: &CudaEvent) -> CudaResult<()> {
        self.stream
            .wait_event(event.raw(), cust::EventWaitFlags::empty())?;

        // Track dependency
        let mut deps = self.dependency_events.lock().unwrap();
        deps.push(event.clone());

        Ok(())
    }

    /// Record event on this stream
    pub fn record_event(&self, event: &CudaEvent) -> CudaResult<()> {
        event.raw().record(&self.stream)?;
        Ok(())
    }

    /// Add callback to execute when stream operations complete
    pub fn add_callback<F>(&self, callback: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let mut callbacks = self.callbacks.lock().unwrap();
        callbacks.push(Box::new(callback));
    }

    /// Async memory copy from host to device
    pub fn copy_from_host_async<T: Copy>(&self, dst: *mut T, src: &[T]) -> CudaResult<()> {
        let start_time = Instant::now();

        unsafe {
            let result = cuda_sys::cudaMemcpyAsync(
                dst as *mut c_void,
                src.as_ptr() as *const c_void,
                src.len() * std::mem::size_of::<T>(),
                cuda_sys::cudaMemcpyKind_cudaMemcpyHostToDevice,
                self.stream(),
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Async host-to-device copy failed: {:?}", result),
                });
            }
        }

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.memory_transfers += 1;

        Ok(())
    }

    /// Async memory copy from device to host
    pub fn copy_to_host_async<T: Copy>(&self, dst: &mut [T], src: *const T) -> CudaResult<()> {
        unsafe {
            let result = cuda_sys::cudaMemcpyAsync(
                dst.as_mut_ptr() as *mut c_void,
                src as *const c_void,
                dst.len() * std::mem::size_of::<T>(),
                cuda_sys::cudaMemcpyKind_cudaMemcpyDeviceToHost,
                self.stream(),
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Async device-to-host copy failed: {:?}", result),
                });
            }
        }

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.memory_transfers += 1;

        Ok(())
    }

    /// Async device-to-device memory copy
    pub fn copy_device_to_device_async<T: Copy>(
        &self,
        dst: *mut T,
        src: *const T,
        count: usize,
    ) -> CudaResult<()> {
        unsafe {
            let result = cuda_sys::cudaMemcpyAsync(
                dst as *mut c_void,
                src as *const c_void,
                count * std::mem::size_of::<T>(),
                cuda_sys::cudaMemcpyKind_cudaMemcpyDeviceToDevice,
                self.stream(),
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Async device-to-device copy failed: {:?}", result),
                });
            }
        }

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.memory_transfers += 1;

        Ok(())
    }

    /// Prefetch unified memory to device with this stream
    pub fn prefetch_to_device_async(
        &self,
        ptr: *mut u8,
        size: usize,
        device_id: Option<usize>,
    ) -> CudaResult<()> {
        let target_device = device_id.unwrap_or(0) as i32;

        unsafe {
            let result = cuda_sys::cudaMemPrefetchAsync(
                ptr as *const c_void,
                size,
                target_device,
                self.stream(),
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to prefetch memory: {:?}", result),
                });
            }
        }

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.memory_transfers += 1;

        Ok(())
    }

    /// Prefetch unified memory to host with this stream
    pub fn prefetch_to_host_async(&self, ptr: *mut u8, size: usize) -> CudaResult<()> {
        unsafe {
            let result = cuda_sys::cudaMemPrefetchAsync(
                ptr as *const c_void,
                size,
                cuda_sys::cudaCpuDeviceId as i32,
                self.stream(),
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to prefetch memory to host: {:?}", result),
                });
            }
        }

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.memory_transfers += 1;

        Ok(())
    }

    /// Set memory set to zero asynchronously
    pub fn memset_async<T>(&self, ptr: *mut T, value: u8, count: usize) -> CudaResult<()> {
        unsafe {
            let result = cuda_sys::cudaMemsetAsync(
                ptr as *mut c_void,
                value as i32,
                count * std::mem::size_of::<T>(),
                self.stream(),
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Async memset failed: {:?}", result),
                });
            }
        }

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.memory_transfers += 1;

        Ok(())
    }

    /// Wait for all dependencies to complete
    pub fn wait_for_dependencies(&self) -> CudaResult<()> {
        let deps = self.dependency_events.lock().unwrap();
        for event in deps.iter() {
            self.wait_event(event)?;
        }
        Ok(())
    }

    /// Clear all dependencies
    pub fn clear_dependencies(&self) {
        let mut deps = self.dependency_events.lock().unwrap();
        deps.clear();
    }

    /// Record kernel launch for metrics
    pub fn record_kernel_launch(&self) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.kernel_launches += 1;
    }

    /// Update peak memory usage for metrics
    pub fn update_peak_memory(&self, memory_usage: usize) {
        let mut metrics = self.metrics.lock().unwrap();
        if memory_usage > metrics.peak_memory_usage {
            metrics.peak_memory_usage = memory_usage;
        }
    }
}

impl Clone for CudaStream {
    fn clone(&self) -> Self {
        // Note: CUDA streams are not cloneable, so we create a new one with same priority
        Self::new_with_priority(self.priority).expect("Failed to create new CUDA stream")
    }
}

// Add Arc wrapper for easier sharing
type SharedCudaStream = Arc<CudaStream>;

/// Implement shared functionality for Arc<CudaStream>
impl From<CudaStream> for SharedCudaStream {
    fn from(stream: CudaStream) -> Self {
        Arc::new(stream)
    }
}

/// CUDA event for synchronization with enhanced timing capabilities
#[derive(Debug, Clone)]
pub struct CudaEvent {
    event: Arc<cust::Event>,
    creation_time: Instant,
    timing_enabled: bool,
}

impl CudaEvent {
    /// Create new CUDA event
    pub fn new() -> CudaResult<Self> {
        let event = cust::Event::new(cust::EventFlags::DISABLE_TIMING)?;
        Ok(Self {
            event: Arc::new(event),
            creation_time: Instant::now(),
            timing_enabled: false,
        })
    }

    /// Create event with timing capability
    pub fn new_with_timing() -> CudaResult<Self> {
        let event = cust::Event::new(cust::EventFlags::DEFAULT)?;
        Ok(Self {
            event: Arc::new(event),
            creation_time: Instant::now(),
            timing_enabled: true,
        })
    }

    /// Create event with blocking synchronization
    pub fn new_blocking() -> CudaResult<Self> {
        let event =
            cust::Event::new(cust::EventFlags::BLOCKING_SYNC | cust::EventFlags::DISABLE_TIMING)?;
        Ok(Self {
            event: Arc::new(event),
            creation_time: Instant::now(),
            timing_enabled: false,
        })
    }

    /// Get raw CUDA event
    pub fn raw(&self) -> Arc<cust::Event> {
        Arc::clone(&self.event)
    }

    /// Check if timing is enabled
    pub fn timing_enabled(&self) -> bool {
        self.timing_enabled
    }

    /// Get creation time
    pub fn creation_time(&self) -> Instant {
        self.creation_time
    }

    /// Synchronize on event
    pub fn synchronize(&self) -> CudaResult<()> {
        self.event.synchronize()?;
        Ok(())
    }

    /// Check if event is ready
    pub fn is_ready(&self) -> CudaResult<bool> {
        match self.event.query() {
            Ok(_) => Ok(true),
            Err(cust::CudaError::NotReady) => Ok(false),
            Err(e) => Err(CudaError::Runtime(e)),
        }
    }

    /// Get elapsed time between two events (in milliseconds)
    pub fn elapsed_time(&self, start: &CudaEvent) -> CudaResult<f32> {
        if !self.timing_enabled || !start.timing_enabled {
            return Err(CudaError::Context {
                message: "Timing not enabled for one or both events".to_string(),
            });
        }
        let time = self.event.elapsed_time_f32(&start.event)?;
        Ok(time)
    }

    /// Get elapsed wall clock time since creation
    pub fn wall_clock_elapsed(&self) -> Duration {
        self.creation_time.elapsed()
    }

    /// Record this event on a stream
    pub fn record_on_stream(&self, stream: &CudaStream) -> CudaResult<()> {
        self.event.record(stream.raw())?;
        Ok(())
    }
}

/// Stream pool for efficient stream management
#[derive(Debug)]
pub struct StreamPool {
    streams: Vec<Arc<CudaStream>>,
    current: std::sync::atomic::AtomicUsize,
}

impl StreamPool {
    /// Create new stream pool
    pub fn new(size: usize) -> CudaResult<Self> {
        let mut streams = Vec::with_capacity(size);
        for _ in 0..size {
            streams.push(Arc::new(CudaStream::new()?));
        }

        Ok(Self {
            streams,
            current: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Get next available stream
    pub fn get_stream(&self) -> Arc<CudaStream> {
        let idx = self
            .current
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.streams.len();
        Arc::clone(&self.streams[idx])
    }

    /// Synchronize all streams
    pub fn synchronize_all(&self) -> CudaResult<()> {
        for stream in &self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_creation() {
        if crate::is_available() {
            let stream = CudaStream::new();
            assert!(stream.is_ok());

            let default_stream = CudaStream::default();
            assert!(default_stream.is_ok());
            assert_eq!(default_stream.unwrap().id(), 0);
        }
    }

    #[test]
    fn test_event_creation() {
        if crate::is_available() {
            let event = CudaEvent::new();
            assert!(event.is_ok());

            let timing_event = CudaEvent::new_with_timing();
            assert!(timing_event.is_ok());
        }
    }

    #[test]
    fn test_stream_pool() {
        if crate::is_available() {
            let pool = StreamPool::new(4);
            assert!(pool.is_ok());

            let pool = pool.unwrap();
            let stream1 = pool.get_stream();
            let stream2 = pool.get_stream();

            // Different streams should have different IDs
            assert_ne!(stream1.id(), stream2.id());
        }
    }
}
