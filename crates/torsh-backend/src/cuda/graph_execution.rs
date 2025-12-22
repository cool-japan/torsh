//! CUDA Graph capture and replay for high-performance execution
//!
//! This module provides CUDA Graph support for optimizing repeated workloads:
//! - Graph capture of operation sequences
//! - Graph instantiation and replay
//! - Memory pool integration
//! - Cross-stream graph coordination
//! - Performance monitoring and optimization

use crate::cuda::error::{CudaError, CudaResult};
use crate::cuda::CudaStream;
use cuda_sys::cudart::{
    cudaGraphExec_t, cudaGraphNode_t, cudaGraph_t, cudaKernelNodeParams, cudaMemcpyNodeParams,
    cudaMemsetNodeParams,
};
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

impl CudaGraph {
    /// Create new empty CUDA graph
    pub fn new() -> CudaResult<Self> {
        let mut graph: cudaGraph_t = ptr::null_mut();

        unsafe {
            let result = cuda_sys::cudaGraphCreate(&mut graph, 0);
            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to create CUDA graph: {:?}", result),
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
        let mut cuda_params = cudaKernelNodeParams {
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
            let result = cuda_sys::cudaGraphAddKernelNode(
                &mut node,
                self.graph,
                dep_nodes.as_ptr(),
                dep_nodes.len(),
                &mut cuda_params,
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to add kernel node: {:?}", result),
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
            kind: copy_params.kind,
        };

        let dep_nodes: Vec<cudaGraphNode_t> =
            dependencies.iter().map(|&idx| self.nodes[idx]).collect();

        unsafe {
            let result = cuda_sys::cudaGraphAddMemcpyNode(
                &mut node,
                self.graph,
                dep_nodes.as_ptr(),
                dep_nodes.len(),
                &mut cuda_params,
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
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
        let mut cuda_params = cudaMemsetNodeParams {
            dst: memset_params.dst,
            value: memset_params.value,
            count: memset_params.count,
        };

        let dep_nodes: Vec<cudaGraphNode_t> =
            dependencies.iter().map(|&idx| self.nodes[idx]).collect();

        unsafe {
            let result = cuda_sys::cudaGraphAddMemsetNode(
                &mut node,
                self.graph,
                dep_nodes.as_ptr(),
                dep_nodes.len(),
                &mut cuda_params,
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
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
            let result = cuda_sys::cudaGraphAddChildGraphNode(
                &mut node,
                self.graph,
                dep_nodes.as_ptr(),
                dep_nodes.len(),
                child_graph.graph,
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
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
            let result = cuda_sys::cudaGraphClone(&mut cloned_graph, self.graph);
            if result != cuda_sys::cudaError_t::cudaSuccess {
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
            cuda_sys::cudaGraphDestroy(self.graph);
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
            let result = cuda_sys::cudaGraphInstantiate(
                &mut graph_exec,
                graph.graph,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
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
            let result = cuda_sys::cudaGraphLaunch(self.graph_exec, stream.stream());
            if result != cuda_sys::cudaError_t::cudaSuccess {
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
        let mut update_result: cuda_sys::cudaGraphExecUpdateResult =
            cuda_sys::cudaGraphExecUpdateResult::cudaGraphExecUpdateError;

        unsafe {
            let result = cuda_sys::cudaGraphExecUpdate(
                self.graph_exec,
                graph.graph,
                ptr::null_mut(),
                &mut update_result,
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to update graph exec: {:?}", result),
                });
            }
        }

        Ok(update_result == cuda_sys::cudaGraphExecUpdateResult::cudaGraphExecUpdateSuccess)
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
            cuda_sys::cudaGraphExecDestroy(self.graph_exec);
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
    pub kind: cuda_sys::cudaMemcpyKind,
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
            let result = cuda_sys::cudaStreamBeginCapture(
                stream.stream(),
                cuda_sys::cudaStreamCaptureMode::cudaStreamCaptureModeGlobal,
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
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
            let result = cuda_sys::cudaStreamEndCapture(self.stream.stream(), &mut graph);
            if result != cuda_sys::cudaError_t::cudaSuccess {
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
        let mut pool = self.pool.lock().unwrap();

        // Try to find existing allocation of suitable size
        for (i, &(ptr, alloc_size)) in pool.iter().enumerate() {
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
            let result = cuda_sys::cudaMalloc(&mut raw_ptr, size);
            if result != cuda_sys::cudaError_t::cudaSuccess {
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
        let mut pool = self.pool.lock().unwrap();
        pool.push((ptr, size));
        self.total_allocated = self.total_allocated.saturating_sub(size);
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryPoolStats {
        let pool = self.pool.lock().unwrap();
        MemoryPoolStats {
            total_allocated: self.total_allocated,
            peak_usage: self.peak_usage,
            free_blocks: pool.len(),
            total_free_memory: pool.iter().map(|(_, size)| size).sum(),
        }
    }

    /// Clear all allocations
    pub fn clear(&mut self) -> CudaResult<()> {
        let mut pool = self.pool.lock().unwrap();

        for &(ptr, _) in pool.iter() {
            unsafe {
                cuda_sys::cudaFree(ptr);
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
    fn test_graph_creation() {
        if crate::cuda::is_available() {
            let graph = CudaGraph::new();
            assert!(graph.is_ok());

            let graph = graph.unwrap();
            assert_eq!(graph.node_count(), 0);
        }
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = GraphMemoryPool::new();

        // Test allocation
        if crate::cuda::is_available() {
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
    fn test_graph_validation() {
        if crate::cuda::is_available() {
            let graph = CudaGraph::new().unwrap();
            assert!(graph.validate().is_ok());
        }
    }
}
