//! CUDA Graph capture and replay for high-performance execution
//!
//! This module provides CUDA Graph support for optimizing repeated workloads:
//! - Graph capture of operation sequences
//! - Graph instantiation and replay
//! - Memory pool integration
//! - Cross-stream graph coordination
//! - Performance monitoring and optimization
//!
//! Note: CUDA Graph API types are not available in cuda-sys 0.2.0.
//! This module defines placeholder types for forward compatibility.

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use crate::cuda::error::{CudaError, CudaResult};
use crate::cuda::CudaStream;

// ============================================================================
// CUDA Graph API type stubs (not available in cuda-sys 0.2.0)
// These will be replaced with actual cuda-sys types when a newer version
// with CUDA 10+ Graph API support is available.
// ============================================================================

/// Opaque handle to a CUDA graph (placeholder for cuda-sys cudaGraph_t)
pub type cudaGraph_t = *mut std::ffi::c_void;

/// Opaque handle to a CUDA graph node (placeholder for cuda-sys cudaGraphNode_t)
pub type cudaGraphNode_t = *mut std::ffi::c_void;

/// Opaque handle to an instantiated CUDA graph (placeholder for cuda-sys cudaGraphExec_t)
pub type cudaGraphExec_t = *mut std::ffi::c_void;

/// CUDA kernel node parameters (placeholder for cuda-sys cudaKernelNodeParams)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct cudaKernelNodeParams {
    pub func: *mut std::ffi::c_void,
    pub gridDimX: u32,
    pub gridDimY: u32,
    pub gridDimZ: u32,
    pub blockDimX: u32,
    pub blockDimY: u32,
    pub blockDimZ: u32,
    pub sharedMemBytes: u32,
    pub kernelParams: *mut *mut std::ffi::c_void,
    pub extra: *mut *mut std::ffi::c_void,
}

/// CUDA memcpy node parameters (placeholder for cuda-sys cudaMemcpyNodeParams)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct cudaMemcpyNodeParams {
    pub dst: *mut std::ffi::c_void,
    pub src: *const std::ffi::c_void,
    pub count: usize,
    pub kind: i32,
}

/// CUDA memset node parameters (placeholder for cuda-sys cudaMemsetNodeParams)
/// Note: This is a simplified version for graph node creation
#[repr(C)]
#[derive(Debug, Clone)]
pub struct cudaMemsetNodeParams {
    pub dst: *mut std::ffi::c_void,
    pub pitch: usize,
    pub value: u32,
    pub elementSize: u32,
    pub width: usize,
    pub height: usize,
}

impl cudaMemsetNodeParams {
    /// Create from high-level CudaMemsetNodeParams
    pub fn from_params(params: &CudaMemsetNodeParams) -> Self {
        Self {
            dst: params.dst,
            pitch: 0,
            value: params.value as u32,
            elementSize: 1,
            width: params.count,
            height: 1,
        }
    }
}

/// CUDA graph exec update result (placeholder for cuda-sys)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess = 0,
    cudaGraphExecUpdateError = 1,
    cudaGraphExecUpdateErrorTopologyChanged = 2,
    cudaGraphExecUpdateErrorNodeTypeChanged = 3,
    cudaGraphExecUpdateErrorFunctionChanged = 4,
    cudaGraphExecUpdateErrorParametersChanged = 5,
    cudaGraphExecUpdateErrorNotSupported = 6,
}

/// CUDA stream capture mode (placeholder for cuda-sys)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
}

// Placeholder functions that return errors since CUDA Graph API is not available
mod cuda_graph_stubs {
    use super::*;

    pub unsafe fn cudaGraphCreate(_graph: *mut cudaGraph_t, _flags: u32) -> i32 {
        // cudaErrorNotSupported = 801
        801
    }

    pub unsafe fn cudaGraphAddKernelNode(
        _node: *mut cudaGraphNode_t,
        _graph: cudaGraph_t,
        _deps: *const cudaGraphNode_t,
        _num_deps: usize,
        _params: *const cudaKernelNodeParams,
    ) -> i32 {
        801
    }

    pub unsafe fn cudaGraphAddMemcpyNode(
        _node: *mut cudaGraphNode_t,
        _graph: cudaGraph_t,
        _deps: *const cudaGraphNode_t,
        _num_deps: usize,
        _params: *const cudaMemcpyNodeParams,
    ) -> i32 {
        801
    }

    pub unsafe fn cudaGraphAddMemsetNode(
        _node: *mut cudaGraphNode_t,
        _graph: cudaGraph_t,
        _deps: *const cudaGraphNode_t,
        _num_deps: usize,
        _params: *const cudaMemsetNodeParams,
    ) -> i32 {
        801
    }

    pub unsafe fn cudaGraphInstantiate(
        _exec: *mut cudaGraphExec_t,
        _graph: cudaGraph_t,
        _error_node: *mut cudaGraphNode_t,
        _log_buffer: *mut i8,
        _buffer_size: usize,
    ) -> i32 {
        801
    }

    pub unsafe fn cudaGraphLaunch(_exec: cudaGraphExec_t, _stream: *mut std::ffi::c_void) -> i32 {
        801
    }

    pub unsafe fn cudaGraphExecDestroy(_exec: cudaGraphExec_t) -> i32 {
        801
    }

    pub unsafe fn cudaGraphDestroy(_graph: cudaGraph_t) -> i32 {
        801
    }

    pub unsafe fn cudaStreamBeginCapture(_stream: *mut std::ffi::c_void, _mode: i32) -> i32 {
        801
    }

    pub unsafe fn cudaStreamEndCapture(
        _stream: *mut std::ffi::c_void,
        _graph: *mut cudaGraph_t,
    ) -> i32 {
        801
    }

    pub unsafe fn cudaGraphExecUpdate(
        _exec: cudaGraphExec_t,
        _graph: cudaGraph_t,
        _error_node: *mut cudaGraphNode_t,
        _update_result: *mut cudaGraphExecUpdateResult,
    ) -> i32 {
        // Set result to error
        if !_update_result.is_null() {
            *_update_result = cudaGraphExecUpdateResult::cudaGraphExecUpdateError;
        }
        801
    }

    pub unsafe fn cudaGraphAddChildGraphNode(
        _node: *mut cudaGraphNode_t,
        _graph: cudaGraph_t,
        _deps: *const cudaGraphNode_t,
        _num_deps: usize,
        _child_graph: cudaGraph_t,
    ) -> i32 {
        801
    }

    pub unsafe fn cudaGraphClone(_cloned: *mut cudaGraph_t, _original: cudaGraph_t) -> i32 {
        801
    }

    pub unsafe fn cudaMalloc(_ptr: *mut *mut c_void, _size: usize) -> i32 {
        801
    }

    pub unsafe fn cudaFree(_ptr: *mut c_void) -> i32 {
        801
    }
}

/// CUDA memory copy kind (placeholder)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}

// Use stub functions since cuda-sys doesn't have Graph API
use cuda_graph_stubs::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// CUDA Graph wrapper for high-performance execution
#[derive(Debug)]
pub struct CudaGraph {
    graph: cudaGraph_t,
    nodes: Vec<cudaGraphNode_t>,
    dependencies: HashMap<usize, Vec<usize>>, // node_id -> dependency_node_ids
    memory_pools: Vec<Arc<GraphMemoryPool>>,
}

/// Success code constant
const CUDA_SUCCESS: i32 = 0;

impl CudaGraph {
    /// Create new empty CUDA graph
    pub fn new() -> CudaResult<Self> {
        let mut graph: cudaGraph_t = ptr::null_mut();

        unsafe {
            let result = cudaGraphCreate(&mut graph, 0);
            if result != CUDA_SUCCESS {
                return Err(CudaError::Context {
                    message: format!(
                        "Failed to create CUDA graph: error code {} (CUDA Graph API not available in cuda-sys 0.2.0)",
                        result
                    ),
                });
            }
        }

        Ok(Self {
            graph,
            nodes: Vec::new(),
            dependencies: HashMap::new(),
            memory_pools: Vec::new(),
        })
    }

    /// Add kernel node to graph
    pub fn add_kernel_node(
        &mut self,
        kernel_params: CudaKernelNodeParams,
        dependencies: &[usize],
    ) -> CudaResult<usize> {
        let mut node: cudaGraphNode_t = ptr::null_mut();
        let cuda_params = cudaKernelNodeParams {
            func: kernel_params.function,
            gridDimX: kernel_params.grid_dim.0,
            gridDimY: kernel_params.grid_dim.1,
            gridDimZ: kernel_params.grid_dim.2,
            blockDimX: kernel_params.block_dim.0,
            blockDimY: kernel_params.block_dim.1,
            blockDimZ: kernel_params.block_dim.2,
            sharedMemBytes: kernel_params.shared_memory_bytes,
            kernelParams: kernel_params.parameters.as_ptr() as *mut *mut c_void,
            extra: ptr::null_mut(),
        };

        // Convert dependencies to node pointers
        let dep_nodes: Vec<cudaGraphNode_t> =
            dependencies.iter().map(|&idx| self.nodes[idx]).collect();

        unsafe {
            let result = cudaGraphAddKernelNode(
                &mut node,
                self.graph,
                dep_nodes.as_ptr(),
                dep_nodes.len(),
                &cuda_params,
            );

            if result != CUDA_SUCCESS {
                return Err(CudaError::Context {
                    message: format!("Failed to add kernel node: error code {}", result),
                });
            }
        }

        let node_id = self.nodes.len();
        self.nodes.push(node);
        self.dependencies.insert(node_id, dependencies.to_vec());

        Ok(node_id)
    }

    /// Add memory copy node to graph
    pub fn add_memcpy_node(
        &mut self,
        copy_params: CudaMemcpyNodeParams,
        dependencies: &[usize],
    ) -> CudaResult<usize> {
        let mut node: cudaGraphNode_t = ptr::null_mut();
        let mut cuda_params = cudaMemcpyNodeParams {
            dst: copy_params.dst,
            src: copy_params.src,
            count: copy_params.count,
            kind: copy_params.kind as i32,
        };

        let dep_nodes: Vec<cudaGraphNode_t> =
            dependencies.iter().map(|&idx| self.nodes[idx]).collect();

        unsafe {
            let result = cudaGraphAddMemcpyNode(
                &mut node,
                self.graph,
                dep_nodes.as_ptr(),
                dep_nodes.len(),
                &mut cuda_params,
            );

            if result != CUDA_SUCCESS {
                return Err(CudaError::Context {
                    message: format!("Failed to add memcpy node: {:?}", result),
                });
            }
        }

        let node_id = self.nodes.len();
        self.nodes.push(node);
        self.dependencies.insert(node_id, dependencies.to_vec());

        Ok(node_id)
    }

    /// Add memory set node to graph
    pub fn add_memset_node(
        &mut self,
        memset_params: CudaMemsetNodeParams,
        dependencies: &[usize],
    ) -> CudaResult<usize> {
        let mut node: cudaGraphNode_t = ptr::null_mut();
        let mut cuda_params = cudaMemsetNodeParams::from_params(&memset_params);

        let dep_nodes: Vec<cudaGraphNode_t> =
            dependencies.iter().map(|&idx| self.nodes[idx]).collect();

        unsafe {
            let result = cudaGraphAddMemsetNode(
                &mut node,
                self.graph,
                dep_nodes.as_ptr(),
                dep_nodes.len(),
                &mut cuda_params,
            );

            if result != CUDA_SUCCESS {
                return Err(CudaError::Context {
                    message: format!("Failed to add memset node: {:?}", result),
                });
            }
        }

        let node_id = self.nodes.len();
        self.nodes.push(node);
        self.dependencies.insert(node_id, dependencies.to_vec());

        Ok(node_id)
    }

    /// Add child graph node for hierarchical composition
    pub fn add_child_graph_node(
        &mut self,
        child_graph: &CudaGraph,
        dependencies: &[usize],
    ) -> CudaResult<usize> {
        let mut node: cudaGraphNode_t = ptr::null_mut();

        let dep_nodes: Vec<cudaGraphNode_t> =
            dependencies.iter().map(|&idx| self.nodes[idx]).collect();

        unsafe {
            let result = cudaGraphAddChildGraphNode(
                &mut node,
                self.graph,
                dep_nodes.as_ptr(),
                dep_nodes.len(),
                child_graph.graph,
            );

            if result != CUDA_SUCCESS {
                return Err(CudaError::Context {
                    message: format!("Failed to add child graph node: {:?}", result),
                });
            }
        }

        let node_id = self.nodes.len();
        self.nodes.push(node);
        self.dependencies.insert(node_id, dependencies.to_vec());

        Ok(node_id)
    }

    /// Instantiate graph for execution
    pub fn instantiate(&self) -> CudaResult<CudaGraphExec> {
        CudaGraphExec::from_graph(self)
    }

    /// Clone graph for template-based instantiation
    pub fn clone_graph(&self) -> CudaResult<CudaGraph> {
        let mut cloned_graph: cudaGraph_t = ptr::null_mut();

        unsafe {
            let result = cudaGraphClone(&mut cloned_graph, self.graph);
            if result != CUDA_SUCCESS {
                return Err(CudaError::Context {
                    message: format!("Failed to clone graph: {:?}", result),
                });
            }
        }

        Ok(CudaGraph {
            graph: cloned_graph,
            nodes: self.nodes.clone(),
            dependencies: self.dependencies.clone(),
            memory_pools: self.memory_pools.clone(),
        })
    }

    /// Get raw graph handle
    pub fn raw_graph(&self) -> cudaGraph_t {
        self.graph
    }

    /// Get number of nodes in graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Validate graph structure
    pub fn validate(&self) -> CudaResult<()> {
        // Check for cycles in dependency graph
        if self.has_cycles() {
            return Err(CudaError::Context {
                message: "Graph contains cycles".to_string(),
            });
        }

        // Additional validation could be added here
        Ok(())
    }

    fn has_cycles(&self) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();

        for node_id in 0..self.nodes.len() {
            if self.has_cycle_util(node_id, &mut visited, &mut rec_stack) {
                return true;
            }
        }
        false
    }

    fn has_cycle_util(
        &self,
        node_id: usize,
        visited: &mut std::collections::HashSet<usize>,
        rec_stack: &mut std::collections::HashSet<usize>,
    ) -> bool {
        visited.insert(node_id);
        rec_stack.insert(node_id);

        if let Some(dependencies) = self.dependencies.get(&node_id) {
            for &dep_id in dependencies {
                if !visited.contains(&dep_id) && self.has_cycle_util(dep_id, visited, rec_stack) {
                    return true;
                }
                if rec_stack.contains(&dep_id) {
                    return true;
                }
            }
        }

        rec_stack.remove(&node_id);
        false
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe {
            cudaGraphDestroy(self.graph);
        }
    }
}

/// Executable CUDA graph instance
#[derive(Debug)]
pub struct CudaGraphExec {
    graph_exec: cudaGraphExec_t,
    execution_count: u64,
    total_execution_time: Duration,
    last_execution_time: Option<Duration>,
}

impl CudaGraphExec {
    /// Create graph executable from graph
    pub fn from_graph(graph: &CudaGraph) -> CudaResult<Self> {
        let mut graph_exec: cudaGraphExec_t = ptr::null_mut();

        unsafe {
            let result = cudaGraphInstantiate(
                &mut graph_exec,
                graph.graph,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            );

            if result != CUDA_SUCCESS {
                return Err(CudaError::Context {
                    message: format!("Failed to instantiate graph: {:?}", result),
                });
            }
        }

        Ok(Self {
            graph_exec,
            execution_count: 0,
            total_execution_time: Duration::from_secs(0),
            last_execution_time: None,
        })
    }

    /// Launch graph execution on stream
    pub fn launch(&mut self, stream: &CudaStream) -> CudaResult<()> {
        let start_time = Instant::now();

        unsafe {
            let result = cudaGraphLaunch(self.graph_exec, stream.stream() as *mut c_void);
            if result != CUDA_SUCCESS {
                return Err(CudaError::Context {
                    message: format!("Failed to launch graph: {:?}", result),
                });
            }
        }

        // Update execution metrics
        let execution_time = start_time.elapsed();
        self.execution_count += 1;
        self.total_execution_time += execution_time;
        self.last_execution_time = Some(execution_time);

        Ok(())
    }

    /// Update graph executable with new parameters
    pub fn update(&mut self, graph: &CudaGraph) -> CudaResult<bool> {
        let mut update_result: cudaGraphExecUpdateResult =
            cudaGraphExecUpdateResult::cudaGraphExecUpdateError;

        unsafe {
            let result = cudaGraphExecUpdate(
                self.graph_exec,
                graph.graph,
                ptr::null_mut(),
                &mut update_result,
            );

            if result != CUDA_SUCCESS {
                return Err(CudaError::Context {
                    message: format!("Failed to update graph exec: error code {}", result),
                });
            }
        }

        Ok(update_result == cudaGraphExecUpdateResult::cudaGraphExecUpdateSuccess)
    }

    /// Get execution statistics
    pub fn get_execution_stats(&self) -> GraphExecutionStats {
        let average_time = if self.execution_count > 0 {
            self.total_execution_time / self.execution_count as u32
        } else {
            Duration::from_secs(0)
        };

        GraphExecutionStats {
            execution_count: self.execution_count,
            total_execution_time: self.total_execution_time,
            average_execution_time: average_time,
            last_execution_time: self.last_execution_time,
        }
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        unsafe {
            cudaGraphExecDestroy(self.graph_exec);
        }
    }
}

/// Graph execution statistics
#[derive(Debug, Clone)]
pub struct GraphExecutionStats {
    pub execution_count: u64,
    pub total_execution_time: Duration,
    pub average_execution_time: Duration,
    pub last_execution_time: Option<Duration>,
}

/// Parameters for kernel node creation
#[derive(Debug, Clone)]
pub struct CudaKernelNodeParams {
    pub function: *mut c_void,
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_memory_bytes: u32,
    pub parameters: Vec<*mut c_void>,
}

/// Parameters for memory copy node creation
#[derive(Debug, Clone)]
pub struct CudaMemcpyNodeParams {
    pub dst: *mut c_void,
    pub src: *const c_void,
    pub count: usize,
    pub kind: cudaMemcpyKind,
}

/// Parameters for memory set node creation
#[derive(Debug, Clone)]
pub struct CudaMemsetNodeParams {
    pub dst: *mut c_void,
    pub value: i32,
    pub count: usize,
}

/// Graph capture session for recording operations
pub struct GraphCaptureSession {
    stream: Arc<CudaStream>,
    capturing: bool,
    capture_start_time: Option<Instant>,
}

impl GraphCaptureSession {
    /// Start graph capture on stream
    pub fn begin_capture(stream: Arc<CudaStream>) -> CudaResult<Self> {
        unsafe {
            let result = cudaStreamBeginCapture(
                stream.stream() as *mut c_void,
                cudaStreamCaptureMode::cudaStreamCaptureModeGlobal as i32,
            );

            if result != CUDA_SUCCESS {
                return Err(CudaError::Context {
                    message: format!("Failed to begin graph capture: {:?}", result),
                });
            }
        }

        Ok(Self {
            stream,
            capturing: true,
            capture_start_time: Some(Instant::now()),
        })
    }

    /// End graph capture and return captured graph
    pub fn end_capture(mut self) -> CudaResult<CudaGraph> {
        if !self.capturing {
            return Err(CudaError::Context {
                message: "Graph capture not active".to_string(),
            });
        }

        let mut graph: cudaGraph_t = ptr::null_mut();

        unsafe {
            let result = cudaStreamEndCapture(self.stream.stream() as *mut c_void, &mut graph);
            if result != CUDA_SUCCESS {
                return Err(CudaError::Context {
                    message: format!("Failed to end graph capture: {:?}", result),
                });
            }
        }

        self.capturing = false;

        Ok(CudaGraph {
            graph,
            nodes: Vec::new(), // Nodes are populated during capture
            dependencies: HashMap::new(),
            memory_pools: Vec::new(),
        })
    }

    /// Check if capture is active
    pub fn is_capturing(&self) -> bool {
        self.capturing
    }

    /// Get capture duration
    pub fn capture_duration(&self) -> Option<Duration> {
        self.capture_start_time.map(|start| start.elapsed())
    }
}

/// Memory pool for graph-based allocations
#[derive(Debug)]
pub struct GraphMemoryPool {
    pool: Arc<Mutex<Vec<(*mut c_void, usize)>>>, // (ptr, size) pairs
    total_allocated: usize,
    peak_usage: usize,
}

impl GraphMemoryPool {
    /// Create new graph memory pool
    pub fn new() -> Self {
        Self {
            pool: Arc::new(Mutex::new(Vec::new())),
            total_allocated: 0,
            peak_usage: 0,
        }
    }

    /// Allocate memory from pool
    pub fn allocate(&mut self, size: usize) -> CudaResult<*mut c_void> {
        let mut pool = self.pool.lock().expect("lock should not be poisoned");

        // Try to find existing allocation of suitable size
        for (i, &(_ptr, alloc_size)) in pool.iter().enumerate() {
            if alloc_size >= size {
                let allocated_ptr = pool.remove(i).0;
                self.total_allocated += size;
                if self.total_allocated > self.peak_usage {
                    self.peak_usage = self.total_allocated;
                }
                return Ok(allocated_ptr);
            }
        }

        // Allocate new memory
        let ptr = unsafe {
            let mut raw_ptr: *mut c_void = ptr::null_mut();
            let result = cudaMalloc(&mut raw_ptr, size);
            if result != CUDA_SUCCESS {
                return Err(CudaError::Context {
                    message: format!("Failed to allocate memory: {:?}", result),
                });
            }
            raw_ptr
        };

        self.total_allocated += size;
        if self.total_allocated > self.peak_usage {
            self.peak_usage = self.total_allocated;
        }

        Ok(ptr)
    }

    /// Return memory to pool
    pub fn deallocate(&mut self, ptr: *mut c_void, size: usize) {
        let mut pool = self.pool.lock().expect("lock should not be poisoned");
        pool.push((ptr, size));
        self.total_allocated = self.total_allocated.saturating_sub(size);
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryPoolStats {
        let pool = self.pool.lock().expect("lock should not be poisoned");
        MemoryPoolStats {
            total_allocated: self.total_allocated,
            peak_usage: self.peak_usage,
            free_blocks: pool.len(),
            total_free_memory: pool.iter().map(|(_, size)| size).sum(),
        }
    }

    /// Clear all allocations
    pub fn clear(&mut self) -> CudaResult<()> {
        let mut pool = self.pool.lock().expect("lock should not be poisoned");

        for &(ptr, _) in pool.iter() {
            unsafe {
                cudaFree(ptr);
            }
        }

        pool.clear();
        self.total_allocated = 0;
        Ok(())
    }
}

impl Drop for GraphMemoryPool {
    fn drop(&mut self) {
        let _ = self.clear();
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub free_blocks: usize,
    pub total_free_memory: usize,
}

/// High-level graph execution manager
pub struct GraphExecutionManager {
    graphs: HashMap<String, (CudaGraph, CudaGraphExec)>,
    capture_sessions: HashMap<String, GraphCaptureSession>,
    memory_pools: HashMap<String, Arc<Mutex<GraphMemoryPool>>>,
    execution_history: HashMap<String, Vec<Duration>>,
}

impl GraphExecutionManager {
    /// Create new graph execution manager
    pub fn new() -> Self {
        Self {
            graphs: HashMap::new(),
            capture_sessions: HashMap::new(),
            memory_pools: HashMap::new(),
            execution_history: HashMap::new(),
        }
    }

    /// Start capturing operations for a named graph
    pub fn begin_capture(&mut self, graph_name: String, stream: Arc<CudaStream>) -> CudaResult<()> {
        let session = GraphCaptureSession::begin_capture(stream)?;
        self.capture_sessions.insert(graph_name, session);
        Ok(())
    }

    /// End capture and store the graph
    pub fn end_capture(&mut self, graph_name: String) -> CudaResult<()> {
        let session =
            self.capture_sessions
                .remove(&graph_name)
                .ok_or_else(|| CudaError::Context {
                    message: format!("No active capture session for graph: {}", graph_name),
                })?;

        let graph = session.end_capture()?;
        let graph_exec = graph.instantiate()?;

        self.graphs.insert(graph_name, (graph, graph_exec));
        Ok(())
    }

    /// Execute a stored graph
    pub fn execute_graph(&mut self, graph_name: &str, stream: &CudaStream) -> CudaResult<Duration> {
        let start_time = Instant::now();

        if let Some((_, graph_exec)) = self.graphs.get_mut(graph_name) {
            graph_exec.launch(stream)?;

            let execution_time = start_time.elapsed();
            self.execution_history
                .entry(graph_name.to_string())
                .or_insert_with(Vec::new)
                .push(execution_time);

            Ok(execution_time)
        } else {
            Err(CudaError::Context {
                message: format!("Graph not found: {}", graph_name),
            })
        }
    }

    /// Get execution statistics for a graph
    pub fn get_graph_stats(&self, graph_name: &str) -> Option<GraphExecutionStats> {
        self.graphs
            .get(graph_name)
            .map(|(_, exec)| exec.get_execution_stats())
    }

    /// Update an existing graph with new version
    pub fn update_graph(&mut self, graph_name: &str, new_graph: CudaGraph) -> CudaResult<bool> {
        if let Some((old_graph, graph_exec)) = self.graphs.get_mut(graph_name) {
            let update_success = graph_exec.update(&new_graph)?;
            if update_success {
                *old_graph = new_graph;
            }
            Ok(update_success)
        } else {
            Err(CudaError::Context {
                message: format!("Graph not found: {}", graph_name),
            })
        }
    }

    /// Get memory pool for graph allocations
    pub fn get_memory_pool(&mut self, pool_name: String) -> Arc<Mutex<GraphMemoryPool>> {
        self.memory_pools
            .entry(pool_name)
            .or_insert_with(|| Arc::new(Mutex::new(GraphMemoryPool::new())))
            .clone()
    }

    /// Remove a graph and free resources
    pub fn remove_graph(&mut self, graph_name: &str) -> CudaResult<()> {
        self.graphs.remove(graph_name);
        self.execution_history.remove(graph_name);
        Ok(())
    }

    /// List all available graphs
    pub fn list_graphs(&self) -> Vec<String> {
        self.graphs.keys().cloned().collect()
    }

    /// Get performance summary for all graphs
    pub fn get_performance_summary(&self) -> HashMap<String, GraphPerformanceSummary> {
        let mut summary = HashMap::new();

        for (name, history) in &self.execution_history {
            if let Some((_, exec)) = self.graphs.get(name) {
                let stats = exec.get_execution_stats();
                let recent_times: Vec<_> = history.iter().rev().take(10).copied().collect();

                let trend = if recent_times.len() >= 2 {
                    let first_half: Duration = recent_times[recent_times.len() / 2..].iter().sum();
                    let second_half: Duration = recent_times[..recent_times.len() / 2].iter().sum();

                    let first_avg = first_half / (recent_times.len() / 2) as u32;
                    let second_avg =
                        second_half / (recent_times.len() - recent_times.len() / 2) as u32;

                    if second_avg < first_avg {
                        PerformanceTrend::Improving
                    } else if second_avg > first_avg {
                        PerformanceTrend::Degrading
                    } else {
                        PerformanceTrend::Stable
                    }
                } else {
                    PerformanceTrend::Stable
                };

                summary.insert(
                    name.clone(),
                    GraphPerformanceSummary {
                        execution_stats: stats,
                        recent_executions: recent_times.len(),
                        performance_trend: trend,
                    },
                );
            }
        }

        summary
    }
}

/// Performance trend analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
}

/// Performance summary for a graph
#[derive(Debug, Clone)]
pub struct GraphPerformanceSummary {
    pub execution_stats: GraphExecutionStats,
    pub recent_executions: usize,
    pub performance_trend: PerformanceTrend,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // CUDA Graph API not available in cuda-sys 0.2.0
    fn test_graph_creation() {
        if crate::cuda::is_available() {
            let _device = Arc::new(crate::cuda::device::CudaDevice::new(0).unwrap());
            let graph = CudaGraph::new();
            assert!(graph.is_ok());

            let graph = graph.unwrap();
            assert_eq!(graph.node_count(), 0);
        }
    }

    #[test]
    #[ignore] // CUDA Graph API not available in cuda-sys 0.2.0
    fn test_memory_pool() {
        let mut pool = GraphMemoryPool::new();

        // Test allocation
        if crate::cuda::is_available() {
            let _device = Arc::new(crate::cuda::device::CudaDevice::new(0).unwrap());
            let ptr_result = pool.allocate(1024);
            assert!(ptr_result.is_ok());

            let stats = pool.get_stats();
            assert_eq!(stats.total_allocated, 1024);
        }
    }

    #[test]
    fn test_execution_manager() {
        let manager = GraphExecutionManager::new();
        let graphs = manager.list_graphs();
        assert!(graphs.is_empty());
    }

    #[test]
    #[ignore] // CUDA Graph API not available in cuda-sys 0.2.0
    fn test_graph_validation() {
        if crate::cuda::is_available() {
            let _device = Arc::new(crate::cuda::device::CudaDevice::new(0).unwrap());
            let graph = CudaGraph::new().unwrap();
            assert!(graph.validate().is_ok());
        }
    }
}
