//! Stub implementations for CUDA graph features
//!
//! CUDA graph API is not available in current cuda-sys version.
//! These are placeholder types to allow compilation.

use super::error::{CudaError, CudaResult};
use super::stream::CudaStream;
use std::collections::HashMap;
use std::sync::Arc;

/// Placeholder for CUDA graph (not available in current cuda-sys)
#[derive(Debug, Clone)]
pub struct CudaGraph {
    _placeholder: (),
}

impl CudaGraph {
    pub fn new() -> CudaResult<Self> {
        Err(CudaError::from(crate::error::TorshError::Unimplemented(
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
    graphs: HashMap<String, Arc<CudaGraph>>,
}

impl GraphCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, _key: &str) -> Option<Arc<CudaGraph>> {
        None
    }

    pub fn insert(&mut self, _key: String, _graph: Arc<CudaGraph>) {}

    pub fn clear(&mut self) {
        self.graphs.clear();
    }
}

/// Placeholder for graph capture context
#[derive(Debug)]
pub struct GraphCaptureContext {
    _stream: Arc<CudaStream>,
}

impl GraphCaptureContext {
    pub fn new(_stream: Arc<CudaStream>) -> CudaResult<Self> {
        Err(CudaError::from(crate::error::TorshError::Unimplemented(
            "CUDA graph API not available in current cuda-sys version".to_string(),
        )))
    }

    pub fn end_capture(self) -> CudaResult<CudaGraph> {
        Err(CudaError::from(crate::error::TorshError::Unimplemented(
            "CUDA graph API not available in current cuda-sys version".to_string(),
        )))
    }
}
