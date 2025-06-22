//! CUDA stream management

use std::sync::Arc;
use crate::error::{CudaError, CudaResult};

/// CUDA stream wrapper
#[derive(Debug)]
pub struct CudaStream {
    stream: cust::Stream,
    id: u64,
}

impl CudaStream {
    /// Create new CUDA stream
    pub fn new() -> CudaResult<Self> {
        let stream = cust::Stream::new(cust::StreamFlags::NON_BLOCKING, None)?;
        let id = stream.as_inner() as u64;
        Ok(Self { stream, id })
    }
    
    /// Create default stream (stream 0)
    pub fn default() -> CudaResult<Self> {
        let stream = cust::Stream::default();
        Ok(Self { stream, id: 0 })
    }
    
    /// Get stream ID
    pub fn id(&self) -> u64 {
        self.id
    }
    
    /// Get raw CUDA stream
    pub fn raw(&self) -> &cust::Stream {
        &self.stream
    }
    
    /// Synchronize stream
    pub fn synchronize(&self) -> CudaResult<()> {
        self.stream.synchronize()?;
        Ok(())
    }
    
    /// Check if stream is ready
    pub fn is_ready(&self) -> CudaResult<bool> {
        match self.stream.query() {
            Ok(_) => Ok(true),
            Err(cust::CudaError::NotReady) => Ok(false),
            Err(e) => Err(CudaError::Runtime(e)),
        }
    }
    
    /// Wait for event on this stream
    pub fn wait_event(&self, event: &CudaEvent) -> CudaResult<()> {
        self.stream.wait_event(event.raw(), cust::EventWaitFlags::empty())?;
        Ok(())
    }
    
    /// Record event on this stream
    pub fn record_event(&self, event: &CudaEvent) -> CudaResult<()> {
        event.raw().record(&self.stream)?;
        Ok(())
    }
}

impl Clone for CudaStream {
    fn clone(&self) -> Self {
        // Note: CUDA streams are not cloneable, so we create a new one
        Self::new().expect("Failed to create new CUDA stream")
    }
}

/// CUDA event for synchronization
#[derive(Debug)]
pub struct CudaEvent {
    event: cust::Event,
}

impl CudaEvent {
    /// Create new CUDA event
    pub fn new() -> CudaResult<Self> {
        let event = cust::Event::new(cust::EventFlags::DEFAULT)?;
        Ok(Self { event })
    }
    
    /// Create event with timing capability
    pub fn new_with_timing() -> CudaResult<Self> {
        let event = cust::Event::new(cust::EventFlags::DEFAULT)?;
        Ok(Self { event })
    }
    
    /// Get raw CUDA event
    pub fn raw(&self) -> &cust::Event {
        &self.event
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
        let time = self.event.elapsed_time_f32(start.raw())?;
        Ok(time)
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
        let idx = self.current.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % self.streams.len();
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