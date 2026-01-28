//! Stub implementations for CUDA graph features
//!
//! CUDA graph API is not available in current cuda-sys version.
//! These are placeholder types to allow compilation.

use super::error::{CudaError, CudaResult};
use super::stream::CudaStream;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use torsh_core::TorshError;

/// Placeholder for CUDA graph (not available in current cuda-sys)
#[derive(Debug, Clone)]
pub struct CudaGraph {
    _placeholder: (),
}

impl CudaGraph {
    pub fn new() -> CudaResult<Self> {
        Err(CudaError::from(TorshError::Unimplemented(
            "CUDA graph API not available in current cuda-sys version".to_string(),
        )))
    }

    /// Launch the graph on a stream (placeholder - returns error)
    pub fn launch(&self, _stream: &CudaStream) -> CudaResult<()> {
        Err(CudaError::from(TorshError::Unimplemented(
            "CUDA graph API not available in current cuda-sys version".to_string(),
        )))
    }
}

impl Default for CudaGraph {
    fn default() -> Self {
        Self { _placeholder: () }
    }
}

/// Placeholder for graph cache
#[derive(Debug, Default)]
pub struct GraphCache {
    graphs: HashMap<String, Arc<Mutex<CudaGraph>>>,
}

impl GraphCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, _key: &str) -> Option<Arc<Mutex<CudaGraph>>> {
        None
    }

    pub fn insert(&mut self, _key: String, _graph: Arc<Mutex<CudaGraph>>) {}

    pub fn clear(&mut self) {
        self.graphs.clear();
    }

    /// Get or create a graph with the given key (placeholder)
    pub fn get_or_create<F>(&self, key: &str, _create_fn: F) -> CudaResult<Arc<Mutex<CudaGraph>>>
    where
        F: FnOnce() -> CudaResult<CudaGraph>,
    {
        if let Some(graph) = self.graphs.get(key) {
            return Ok(Arc::clone(graph));
        }
        Err(CudaError::from(TorshError::Unimplemented(
            "CUDA graph API not available in current cuda-sys version".to_string(),
        )))
    }
}

/// Placeholder for graph capture context
#[derive(Debug)]
pub struct GraphCaptureContext {
    _stream: Arc<CudaStream>,
}

impl GraphCaptureContext {
    pub fn new(_stream: Arc<CudaStream>) -> CudaResult<Self> {
        Err(CudaError::from(TorshError::Unimplemented(
            "CUDA graph API not available in current cuda-sys version".to_string(),
        )))
    }

    /// End capture and return the graph (placeholder)
    pub fn end_capture(self) -> CudaResult<CudaGraph> {
        Err(CudaError::from(TorshError::Unimplemented(
            "CUDA graph API not available in current cuda-sys version".to_string(),
        )))
    }

    /// End capture (alias for end_capture)
    pub fn end(self) -> CudaResult<CudaGraph> {
        self.end_capture()
    }

    /// Abort the capture without creating a graph
    pub fn abort(self) -> CudaResult<()> {
        // Placeholder - just succeed since there's nothing to clean up
        Ok(())
    }

    /// Start capture (placeholder - returns error)
    pub fn start(&self) -> CudaResult<()> {
        Err(CudaError::from(TorshError::Unimplemented(
            "CUDA graph API not available in current cuda-sys version".to_string(),
        )))
    }
}
