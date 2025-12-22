//! Advanced CUDA stream management extensions
//!
//! This module provides advanced stream management capabilities including:
//! - Stream-ordered memory allocation
//! - Multi-stream coordination and synchronization
//! - Performance profiling and metrics
//! - Smart stream allocation strategies
//! - Workload-aware optimization

use crate::cuda::error::{CudaError, CudaResult};
use crate::cuda::memory::CudaAllocation;
use crate::cuda::{CudaEvent, CudaStream, StreamPriority};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Workload characteristics for smart stream selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    Compute,      // Computation-heavy kernels
    Memory,       // Memory-bound operations
    Mixed,        // Balanced compute and memory
    Coordination, // Synchronization operations
}

/// Stream allocation strategy
#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    RoundRobin,   // Simple round-robin allocation
    LoadBalanced, // Based on current load
    Priority,     // Priority-based allocation
    Workload,     // Based on workload characteristics
}

/// Pool-wide performance metrics
#[derive(Debug, Clone, Default)]
pub struct PoolMetrics {
    pub total_allocations: usize,
    pub allocation_failures: usize,
    pub average_utilization: f32,
    pub peak_concurrent_streams: usize,
    pub strategy_effectiveness: HashMap<WorkloadType, f32>,
}

/// Advanced stream pool for efficient stream management with smart allocation
#[derive(Debug)]
pub struct AdvancedStreamPool {
    streams: Vec<Arc<CudaStream>>,
    priority_streams: HashMap<StreamPriority, Vec<Arc<CudaStream>>>,
    current: std::sync::atomic::AtomicUsize,
    allocation_strategy: AllocationStrategy,
    pool_metrics: Arc<Mutex<PoolMetrics>>,
    workload_history: Arc<Mutex<HashMap<WorkloadType, Vec<Duration>>>>,
}

impl AdvancedStreamPool {
    /// Create new stream pool with default configuration
    pub fn new(size: usize) -> CudaResult<Self> {
        Self::new_with_strategy(size, AllocationStrategy::RoundRobin)
    }

    /// Create new stream pool with specified allocation strategy
    pub fn new_with_strategy(size: usize, strategy: AllocationStrategy) -> CudaResult<Self> {
        let mut streams = Vec::with_capacity(size);
        let mut priority_streams = HashMap::new();

        // Create streams with different priorities
        let high_priority_count = size / 4; // 25% high priority
        let normal_priority_count = size / 2; // 50% normal priority
        let low_priority_count = size - high_priority_count - normal_priority_count; // 25% low priority

        // High priority streams
        let mut high_streams = Vec::new();
        for _ in 0..high_priority_count {
            let stream = Arc::new(CudaStream::new_with_priority(StreamPriority::High)?);
            streams.push(Arc::clone(&stream));
            high_streams.push(stream);
        }
        priority_streams.insert(StreamPriority::High, high_streams);

        // Normal priority streams
        let mut normal_streams = Vec::new();
        for _ in 0..normal_priority_count {
            let stream = Arc::new(CudaStream::new_with_priority(StreamPriority::Normal)?);
            streams.push(Arc::clone(&stream));
            normal_streams.push(stream);
        }
        priority_streams.insert(StreamPriority::Normal, normal_streams);

        // Low priority streams
        let mut low_streams = Vec::new();
        for _ in 0..low_priority_count {
            let stream = Arc::new(CudaStream::new_with_priority(StreamPriority::Low)?);
            streams.push(Arc::clone(&stream));
            low_streams.push(stream);
        }
        priority_streams.insert(StreamPriority::Low, low_streams);

        Ok(Self {
            streams,
            priority_streams,
            current: std::sync::atomic::AtomicUsize::new(0),
            allocation_strategy: strategy,
            pool_metrics: Arc::new(Mutex::new(PoolMetrics::default())),
            workload_history: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get next available stream using default strategy
    pub fn get_stream(&self) -> Arc<CudaStream> {
        self.get_stream_for_workload(WorkloadType::Mixed)
    }

    /// Get stream optimized for specific workload type
    pub fn get_stream_for_workload(&self, workload: WorkloadType) -> Arc<CudaStream> {
        let stream = match self.allocation_strategy {
            AllocationStrategy::RoundRobin => self.get_round_robin_stream(),
            AllocationStrategy::LoadBalanced => self.get_load_balanced_stream(),
            AllocationStrategy::Priority => self.get_priority_stream(StreamPriority::Normal),
            AllocationStrategy::Workload => self.get_workload_optimized_stream(workload),
        };

        // Update metrics
        let mut metrics = self.pool_metrics.lock().unwrap();
        metrics.total_allocations += 1;

        stream
    }

    /// Get stream with specific priority
    pub fn get_priority_stream(&self, priority: StreamPriority) -> Arc<CudaStream> {
        if let Some(priority_streams) = self.priority_streams.get(&priority) {
            if !priority_streams.is_empty() {
                let idx = self
                    .current
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    % priority_streams.len();
                return Arc::clone(&priority_streams[idx]);
            }
        }

        // Fallback to any available stream
        self.get_round_robin_stream()
    }

    fn get_round_robin_stream(&self) -> Arc<CudaStream> {
        let idx = self
            .current
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.streams.len();
        Arc::clone(&self.streams[idx])
    }

    fn get_load_balanced_stream(&self) -> Arc<CudaStream> {
        // Find stream with lowest utilization
        let mut best_stream = &self.streams[0];
        let mut best_score = f32::MAX;

        for stream in &self.streams {
            if let Ok(true) = stream.is_ready() {
                // Stream is ready, prefer this one
                return Arc::clone(stream);
            }

            let metrics = stream.metrics();
            let score = metrics.operations_count as f32 + metrics.memory_transfers as f32 * 0.5;
            if score < best_score {
                best_score = score;
                best_stream = stream;
            }
        }

        Arc::clone(best_stream)
    }

    fn get_workload_optimized_stream(&self, workload: WorkloadType) -> Arc<CudaStream> {
        match workload {
            WorkloadType::Compute => {
                // Prefer high-priority streams for compute workloads
                self.get_priority_stream(StreamPriority::High)
            }
            WorkloadType::Memory => {
                // Use normal priority for memory operations
                self.get_priority_stream(StreamPriority::Normal)
            }
            WorkloadType::Mixed => {
                // Use load balancing for mixed workloads
                self.get_load_balanced_stream()
            }
            WorkloadType::Coordination => {
                // Use low priority for coordination operations
                self.get_priority_stream(StreamPriority::Low)
            }
        }
    }

    /// Synchronize all streams
    pub fn synchronize_all(&self) -> CudaResult<()> {
        for stream in &self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Synchronize streams with specific priority
    pub fn synchronize_priority(&self, priority: StreamPriority) -> CudaResult<()> {
        if let Some(priority_streams) = self.priority_streams.get(&priority) {
            for stream in priority_streams {
                stream.synchronize()?;
            }
        }
        Ok(())
    }

    /// Get pool-wide metrics
    pub fn metrics(&self) -> PoolMetrics {
        self.pool_metrics.lock().unwrap().clone()
    }

    /// Record workload completion time for optimization
    pub fn record_workload_completion(&self, workload: WorkloadType, duration: Duration) {
        let mut history = self.workload_history.lock().unwrap();
        let workload_times = history.entry(workload).or_insert_with(Vec::new);
        workload_times.push(duration);

        // Keep only recent history (last 100 entries)
        if workload_times.len() > 100 {
            workload_times.remove(0);
        }
    }

    /// Get average completion time for workload type
    pub fn average_workload_time(&self, workload: WorkloadType) -> Option<Duration> {
        let history = self.workload_history.lock().unwrap();
        if let Some(times) = history.get(&workload) {
            if !times.is_empty() {
                let total = times.iter().sum::<Duration>();
                return Some(total / times.len() as u32);
            }
        }
        None
    }

    /// Optimize pool configuration based on usage patterns
    pub fn optimize_configuration(&mut self) -> CudaResult<()> {
        let history = self.workload_history.lock().unwrap();
        let mut metrics = self.pool_metrics.lock().unwrap();

        // Calculate effectiveness for each workload type
        for (workload, times) in history.iter() {
            if !times.is_empty() {
                let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
                let effectiveness = 1.0 / (avg_time.as_secs_f32() + 1.0);
                metrics
                    .strategy_effectiveness
                    .insert(*workload, effectiveness);
            }
        }

        // Update allocation strategy based on effectiveness
        let mut best_strategy = AllocationStrategy::RoundRobin;
        let mut best_score = 0.0;

        for &effectiveness in metrics.strategy_effectiveness.values() {
            if effectiveness > best_score {
                best_score = effectiveness;
                // Choose strategy based on patterns
                if best_score > 0.8 {
                    best_strategy = AllocationStrategy::Workload;
                } else if best_score > 0.6 {
                    best_strategy = AllocationStrategy::LoadBalanced;
                } else {
                    best_strategy = AllocationStrategy::Priority;
                }
            }
        }

        self.allocation_strategy = best_strategy;
        Ok(())
    }

    /// Check if any streams are ready (non-blocking)
    pub fn has_ready_streams(&self) -> bool {
        self.streams
            .iter()
            .any(|stream| stream.is_ready().unwrap_or(false))
    }

    /// Wait for any stream to become ready
    pub fn wait_for_any_ready(
        &self,
        timeout: Option<Duration>,
    ) -> CudaResult<Option<Arc<CudaStream>>> {
        let start_time = Instant::now();

        loop {
            // Check if any stream is ready
            for stream in &self.streams {
                if stream.is_ready()? {
                    return Ok(Some(Arc::clone(stream)));
                }
            }

            // Check timeout
            if let Some(timeout) = timeout {
                if start_time.elapsed() >= timeout {
                    return Ok(None);
                }
            }

            // Small sleep to avoid busy waiting
            thread::sleep(Duration::from_micros(100));
        }
    }
}

/// Stream-ordered memory allocator for efficient memory management
#[derive(Debug)]
pub struct StreamOrderedAllocator {
    pools: HashMap<u64, Vec<CudaAllocation>>, // Stream ID -> allocations
    allocation_sizes: HashMap<usize, Vec<CudaAllocation>>, // Size -> allocations
    total_allocated: usize,
    stream_dependencies: HashMap<u64, Vec<Arc<CudaEvent>>>, // Stream dependencies
}

impl StreamOrderedAllocator {
    /// Create new stream-ordered allocator
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            allocation_sizes: HashMap::new(),
            total_allocated: 0,
            stream_dependencies: HashMap::new(),
        }
    }

    /// Allocate memory for specific stream
    pub fn allocate_for_stream(
        &mut self,
        stream: &CudaStream,
        size: usize,
    ) -> CudaResult<CudaAllocation> {
        // Try to find existing allocation of suitable size
        if let Some(size_pool) = self.allocation_sizes.get_mut(&size) {
            if let Some(allocation) = size_pool.pop() {
                // Add to stream pool
                self.pools
                    .entry(stream.id())
                    .or_insert_with(Vec::new)
                    .push(allocation.clone());
                return Ok(allocation);
            }
        }

        // Allocate new memory
        let ptr = unsafe { cust::memory::cuda_malloc(size)? };
        let allocation = CudaAllocation::new(ptr, size, Self::size_class(size));

        // Track allocation
        self.pools
            .entry(stream.id())
            .or_insert_with(Vec::new)
            .push(allocation.clone());
        self.total_allocated += size;

        Ok(allocation)
    }

    /// Free memory when stream operations complete
    pub fn free_for_stream(&mut self, stream: &CudaStream) -> CudaResult<()> {
        if let Some(allocations) = self.pools.remove(&stream.id()) {
            for allocation in allocations {
                let size = allocation.size();
                self.allocation_sizes
                    .entry(size)
                    .or_insert_with(Vec::new)
                    .push(allocation);
            }
        }
        Ok(())
    }

    /// Add dependency between streams
    pub fn add_stream_dependency(
        &mut self,
        dependent_stream: &CudaStream,
        dependency_event: Arc<CudaEvent>,
    ) {
        self.stream_dependencies
            .entry(dependent_stream.id())
            .or_insert_with(Vec::new)
            .push(dependency_event);
    }

    /// Check if stream dependencies are satisfied
    pub fn dependencies_satisfied(&self, stream: &CudaStream) -> CudaResult<bool> {
        if let Some(deps) = self.stream_dependencies.get(&stream.id()) {
            for event in deps {
                if !event.is_ready()? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Get total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Clear dependencies for stream
    pub fn clear_dependencies(&mut self, stream: &CudaStream) {
        self.stream_dependencies.remove(&stream.id());
    }

    fn size_class(size: usize) -> usize {
        // Round up to nearest power of 2, minimum 256 bytes
        let min_size = 256;
        if size <= min_size {
            min_size
        } else {
            (size - 1).next_power_of_two().max(min_size)
        }
    }
}

/// Stream callback function type
type StreamCallback = Box<dyn FnOnce() + Send + 'static>;

/// Multi-stream coordinator for complex synchronization patterns
#[derive(Debug)]
pub struct MultiStreamCoordinator {
    streams: Vec<Arc<CudaStream>>,
    barrier_events: Vec<Arc<CudaEvent>>,
    execution_graph: HashMap<u64, Vec<u64>>, // Stream ID -> dependent stream IDs
    completion_callbacks: HashMap<u64, Vec<StreamCallback>>,
}

impl MultiStreamCoordinator {
    /// Create new multi-stream coordinator
    pub fn new(streams: Vec<Arc<CudaStream>>) -> Self {
        Self {
            streams,
            barrier_events: Vec::new(),
            execution_graph: HashMap::new(),
            completion_callbacks: HashMap::new(),
        }
    }

    /// Add dependency between streams
    pub fn add_dependency(
        &mut self,
        dependent: &CudaStream,
        dependency: &CudaStream,
    ) -> CudaResult<()> {
        self.execution_graph
            .entry(dependent.id())
            .or_insert_with(Vec::new)
            .push(dependency.id());

        // Create synchronization event
        let event = Arc::new(CudaEvent::new()?);
        dependency.record_event(&event)?;
        dependent.wait_event(&event)?;

        Ok(())
    }

    /// Create barrier across all streams
    pub fn create_barrier(&mut self) -> CudaResult<()> {
        let barrier_event = Arc::new(CudaEvent::new()?);

        // Record event on all streams
        for stream in &self.streams {
            stream.record_event(&barrier_event)?;
        }

        // Wait for barrier on all streams
        for stream in &self.streams {
            stream.wait_event(&barrier_event)?;
        }

        self.barrier_events.push(barrier_event);
        Ok(())
    }

    /// Execute parallel operations across streams
    pub fn execute_parallel<F>(&self, operation: F) -> CudaResult<()>
    where
        F: Fn(&CudaStream) -> CudaResult<()> + Send + Sync + Clone + 'static,
    {
        let handles: Vec<_> = self
            .streams
            .iter()
            .map(|stream| {
                let stream = Arc::clone(stream);
                let op = operation.clone();
                thread::spawn(move || op(&stream))
            })
            .collect();

        for handle in handles {
            handle.join().map_err(|_| CudaError::Context {
                message: "Thread execution failed".to_string(),
            })??;
        }

        Ok(())
    }

    /// Synchronize all streams in coordinator
    pub fn synchronize_all(&self) -> CudaResult<()> {
        for stream in &self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Add completion callback for stream
    pub fn add_completion_callback<F>(&mut self, stream: &CudaStream, callback: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.completion_callbacks
            .entry(stream.id())
            .or_insert_with(Vec::new)
            .push(Box::new(callback));
    }

    /// Execute completion callbacks for stream
    pub fn execute_callbacks(&mut self, stream: &CudaStream) {
        if let Some(callbacks) = self.completion_callbacks.remove(&stream.id()) {
            for callback in callbacks {
                callback();
            }
        }
    }

    /// Check if execution graph has cycles (deadlock detection)
    pub fn has_cycles(&self) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();

        for stream_id in self.execution_graph.keys() {
            if self.has_cycle_util(*stream_id, &mut visited, &mut rec_stack) {
                return true;
            }
        }
        false
    }

    fn has_cycle_util(
        &self,
        stream_id: u64,
        visited: &mut std::collections::HashSet<u64>,
        rec_stack: &mut std::collections::HashSet<u64>,
    ) -> bool {
        visited.insert(stream_id);
        rec_stack.insert(stream_id);

        if let Some(dependencies) = self.execution_graph.get(&stream_id) {
            for &dep_id in dependencies {
                if !visited.contains(&dep_id) && self.has_cycle_util(dep_id, visited, rec_stack) {
                    return true;
                }
                if rec_stack.contains(&dep_id) {
                    return true;
                }
            }
        }

        rec_stack.remove(&stream_id);
        false
    }
}

/// Stream performance profiler for detailed analysis
#[derive(Debug)]
pub struct StreamProfiler {
    stream_timings: HashMap<u64, Vec<(String, Duration)>>, // Stream ID -> (operation, duration)
    memory_transfers: HashMap<u64, usize>,                 // Stream ID -> transfer count
    kernel_launches: HashMap<u64, usize>,                  // Stream ID -> kernel count
    profiling_enabled: bool,
}

impl StreamProfiler {
    /// Create new stream profiler
    pub fn new() -> Self {
        Self {
            stream_timings: HashMap::new(),
            memory_transfers: HashMap::new(),
            kernel_launches: HashMap::new(),
            profiling_enabled: false,
        }
    }

    /// Enable profiling
    pub fn enable(&mut self) {
        self.profiling_enabled = true;
    }

    /// Disable profiling
    pub fn disable(&mut self) {
        self.profiling_enabled = false;
    }

    /// Record operation timing
    pub fn record_operation(&mut self, stream: &CudaStream, operation: &str, duration: Duration) {
        if self.profiling_enabled {
            self.stream_timings
                .entry(stream.id())
                .or_insert_with(Vec::new)
                .push((operation.to_string(), duration));
        }
    }

    /// Record memory transfer
    pub fn record_memory_transfer(&mut self, stream: &CudaStream) {
        if self.profiling_enabled {
            *self.memory_transfers.entry(stream.id()).or_insert(0) += 1;
        }
    }

    /// Record kernel launch
    pub fn record_kernel_launch(&mut self, stream: &CudaStream) {
        if self.profiling_enabled {
            *self.kernel_launches.entry(stream.id()).or_insert(0) += 1;
        }
    }

    /// Get profiling report for stream
    pub fn get_stream_report(&self, stream: &CudaStream) -> Option<StreamReport> {
        if !self.profiling_enabled {
            return None;
        }

        let timings = self.stream_timings.get(&stream.id())?.clone();
        let memory_transfers = self
            .memory_transfers
            .get(&stream.id())
            .copied()
            .unwrap_or(0);
        let kernel_launches = self.kernel_launches.get(&stream.id()).copied().unwrap_or(0);

        let total_time = timings.iter().map(|(_, d)| *d).sum();
        let operation_count = timings.len();

        Some(StreamReport {
            stream_id: stream.id(),
            total_time,
            operation_count,
            memory_transfers,
            kernel_launches,
            operations: timings,
        })
    }

    /// Get comprehensive profiling report
    pub fn get_comprehensive_report(&self) -> ProfilingReport {
        let mut stream_reports = Vec::new();

        for (&stream_id, timings) in &self.stream_timings {
            let memory_transfers = self.memory_transfers.get(&stream_id).copied().unwrap_or(0);
            let kernel_launches = self.kernel_launches.get(&stream_id).copied().unwrap_or(0);

            let total_time = timings.iter().map(|(_, d)| *d).sum();
            let operation_count = timings.len();

            stream_reports.push(StreamReport {
                stream_id,
                total_time,
                operation_count,
                memory_transfers,
                kernel_launches,
                operations: timings.clone(),
            });
        }

        ProfilingReport {
            streams: stream_reports,
            total_streams: self.stream_timings.len(),
            profiling_enabled: self.profiling_enabled,
        }
    }

    /// Clear all profiling data
    pub fn clear(&mut self) {
        self.stream_timings.clear();
        self.memory_transfers.clear();
        self.kernel_launches.clear();
    }
}

/// Stream profiling report
#[derive(Debug, Clone)]
pub struct StreamReport {
    pub stream_id: u64,
    pub total_time: Duration,
    pub operation_count: usize,
    pub memory_transfers: usize,
    pub kernel_launches: usize,
    pub operations: Vec<(String, Duration)>,
}

/// Comprehensive profiling report
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    pub streams: Vec<StreamReport>,
    pub total_streams: usize,
    pub profiling_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_stream_pool() {
        if crate::cuda::is_available() {
            let pool = AdvancedStreamPool::new(8);
            assert!(pool.is_ok());

            let pool = pool.unwrap();

            // Test different workload types
            let compute_stream = pool.get_stream_for_workload(WorkloadType::Compute);
            let memory_stream = pool.get_stream_for_workload(WorkloadType::Memory);
            let mixed_stream = pool.get_stream_for_workload(WorkloadType::Mixed);

            assert_ne!(compute_stream.id(), memory_stream.id());
            assert_ne!(memory_stream.id(), mixed_stream.id());

            // Test priority streams
            let high_priority = pool.get_priority_stream(StreamPriority::High);
            let normal_priority = pool.get_priority_stream(StreamPriority::Normal);
            let low_priority = pool.get_priority_stream(StreamPriority::Low);

            assert_eq!(high_priority.priority(), StreamPriority::High);
            assert_eq!(normal_priority.priority(), StreamPriority::Normal);
            assert_eq!(low_priority.priority(), StreamPriority::Low);
        }
    }

    #[test]
    fn test_stream_ordered_allocator() {
        if crate::cuda::is_available() {
            let mut allocator = StreamOrderedAllocator::new();
            let stream1 = CudaStream::new().unwrap();
            let stream2 = CudaStream::new().unwrap();

            // Allocate memory for different streams
            let alloc1 = allocator.allocate_for_stream(&stream1, 1024);
            let alloc2 = allocator.allocate_for_stream(&stream2, 2048);

            assert!(alloc1.is_ok());
            assert!(alloc2.is_ok());

            assert_eq!(allocator.total_allocated(), 1024 + 2048);

            // Free memory
            let _ = allocator.free_for_stream(&stream1);
            let _ = allocator.free_for_stream(&stream2);
        }
    }

    #[test]
    fn test_multi_stream_coordinator() {
        if crate::cuda::is_available() {
            let stream1 = Arc::new(CudaStream::new().unwrap());
            let stream2 = Arc::new(CudaStream::new().unwrap());
            let streams = vec![stream1.clone(), stream2.clone()];

            let mut coordinator = MultiStreamCoordinator::new(streams);

            // Test dependency addition
            let result = coordinator.add_dependency(&stream2, &stream1);
            assert!(result.is_ok());

            // Test cycle detection
            let has_cycles = coordinator.has_cycles();
            assert!(!has_cycles); // Should not have cycles with simple dependency

            // Test barrier creation
            let barrier_result = coordinator.create_barrier();
            assert!(barrier_result.is_ok());
        }
    }

    #[test]
    fn test_stream_profiler() {
        if crate::cuda::is_available() {
            let mut profiler = StreamProfiler::new();
            let stream = CudaStream::new().unwrap();

            // Enable profiling
            profiler.enable();
            assert!(profiler.profiling_enabled);

            // Record some operations
            profiler.record_operation(&stream, "test_kernel", Duration::from_millis(10));
            profiler.record_memory_transfer(&stream);
            profiler.record_kernel_launch(&stream);

            // Get report
            let report = profiler.get_stream_report(&stream);
            assert!(report.is_some());

            let report = report.unwrap();
            assert_eq!(report.operation_count, 1);
            assert_eq!(report.memory_transfers, 1);
            assert_eq!(report.kernel_launches, 1);

            // Test comprehensive report
            let comprehensive = profiler.get_comprehensive_report();
            assert_eq!(comprehensive.total_streams, 1);
            assert!(comprehensive.profiling_enabled);
        }
    }
}
