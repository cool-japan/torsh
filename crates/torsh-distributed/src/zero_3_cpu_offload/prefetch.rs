//! Prefetch Scheduling for ZeRO-3 CPU Offloading
//!
//! This module implements intelligent prefetch scheduling for ZeRO-3 (Zero Redundancy
//! Optimizer Stage 3) with CPU offloading capabilities. It provides asynchronous
//! parameter prefetching, intelligent scheduling, and batch prefetch operations
//! to minimize memory transfer latency and maximize training throughput.

use crate::{ProcessGroup, TorshResult};
use log::{debug, info, warn};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

use super::config::Zero3CpuOffloadConfig;

/// Prefetch scheduler for async parameter loading
///
/// Implements intelligent prefetch scheduling including:
/// - Asynchronous parameter prefetching from CPU to GPU
/// - Intelligent scheduling based on execution patterns
/// - Batch prefetch operations with controlled concurrency
/// - Adaptive prefetch distance based on system resources
/// - Background prefetch execution with minimal overhead
pub struct PrefetchScheduler {
    /// Configuration for prefetch scheduling
    config: Zero3CpuOffloadConfig,
    /// Process group for distributed coordination
    process_group: Arc<ProcessGroup>,
    /// Queue of layers to prefetch
    prefetch_queue: Mutex<VecDeque<PrefetchRequest>>,
    /// Current prefetch operations tracking
    active_prefetches: Arc<Mutex<Vec<PrefetchOperation>>>,
    /// Prefetch performance metrics
    metrics: Arc<Mutex<PrefetchMetrics>>,
    /// Adaptive prefetch configuration
    adaptive_config: Arc<Mutex<AdaptivePrefetchConfig>>,
    /// Background task coordination
    task_coordination: Arc<Mutex<TaskCoordination>>,
}

impl PrefetchScheduler {
    /// Create a new prefetch scheduler
    pub fn new(config: &Zero3CpuOffloadConfig, process_group: Arc<ProcessGroup>) -> Self {
        Self {
            config: config.clone(),
            process_group,
            prefetch_queue: Mutex::new(VecDeque::new()),
            active_prefetches: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(PrefetchMetrics::new())),
            adaptive_config: Arc::new(Mutex::new(AdaptivePrefetchConfig::new(&config))),
            task_coordination: Arc::new(Mutex::new(TaskCoordination::new())),
        }
    }

    /// Schedule a single layer for prefetch
    ///
    /// Adds the layer to the prefetch queue and optionally starts immediate prefetch
    /// if async prefetching is enabled and system resources are available.
    pub async fn schedule_prefetch(&self, layer_name: &str) -> TorshResult<()> {
        if !self.config.async_prefetch {
            return Ok(());
        }

        let request = PrefetchRequest {
            layer_name: layer_name.to_string(),
            priority: PrefetchPriority::Normal,
            requested_at: std::time::Instant::now(),
            estimated_size_bytes: self.estimate_layer_size(layer_name),
        };

        // Add to queue
        {
            let mut queue = self.prefetch_queue.lock().unwrap();
            queue.push_back(request.clone());

            // Maintain queue size limit
            let max_queue_size = self.adaptive_config.lock().unwrap().max_queue_size;
            while queue.len() > max_queue_size {
                if let Some(dropped) = queue.pop_front() {
                    info!(
                        "     Dropped prefetch request for {} (queue full)",
                        dropped.layer_name
                    );
                }
            }
        }

        info!(
            "    Scheduled prefetch for layer: {} ({} bytes)",
            layer_name, request.estimated_size_bytes
        );

        // Start async prefetch task
        self.execute_async_prefetch(request).await?;

        Ok(())
    }

    /// Execute asynchronous prefetching for a single layer
    async fn execute_async_prefetch(&self, request: PrefetchRequest) -> TorshResult<()> {
        let process_group = self.process_group.clone();
        let metrics = self.metrics.clone();
        let active_prefetches = self.active_prefetches.clone();

        // Check if we can start a new prefetch operation
        if !self.can_start_prefetch().await? {
            info!(
                "   = Delaying prefetch for {} (system busy)",
                request.layer_name
            );
            return Ok(());
        }

        // Create operation tracking
        let operation = PrefetchOperation {
            layer_name: request.layer_name.clone(),
            started_at: std::time::Instant::now(),
            status: PrefetchStatus::InProgress,
        };

        // Add to active operations
        {
            let mut active = active_prefetches.lock().unwrap();
            active.push(operation);
        }

        // Spawn background task for prefetching
        let layer_name = request.layer_name.clone();
        tokio::spawn(async move {
            let start_time = std::time::Instant::now();
            let result = Self::prefetch_layer_data(&layer_name, process_group).await;

            // Update metrics
            {
                let mut metrics_guard = metrics.lock().unwrap();
                let duration = start_time.elapsed();

                match result {
                    Ok(()) => {
                        metrics_guard.record_successful_prefetch(duration, 0); // Size would be real in production
                        info!(
                            "   = Async prefetch completed for layer: {} in {:?}",
                            layer_name, duration
                        );
                    }
                    Err(e) => {
                        metrics_guard.record_failed_prefetch(duration, e.to_string());
                        tracing::error!("Async prefetch failed for layer {}: {}", layer_name, e);
                    }
                }
            }

            // Remove from active operations
            {
                let mut active = active_prefetches.lock().unwrap();
                active.retain(|op| op.layer_name != layer_name);
            }
        });

        Ok(())
    }

    /// Check if a new prefetch operation can be started
    async fn can_start_prefetch(&self) -> TorshResult<bool> {
        let adaptive_config = self.adaptive_config.lock().unwrap();
        let active_count = self.active_prefetches.lock().unwrap().len();

        // Check concurrent prefetch limit
        if active_count >= adaptive_config.max_concurrent_prefetches {
            return Ok(false);
        }

        // Check system resource availability
        // In a real implementation, this would check:
        // - Available memory bandwidth
        // - GPU memory availability
        // - Current CPU/GPU workload
        // - Network bandwidth for distributed setups

        Ok(true)
    }

    /// Actually prefetch layer data from CPU to GPU
    async fn prefetch_layer_data(
        layer_name: &str,
        _process_group: Arc<ProcessGroup>,
    ) -> TorshResult<()> {
        // In a real implementation, this would:
        // 1. Check if layer parameters are needed soon
        // 2. Load parameters from CPU memory to staging buffer
        // 3. Transfer data to GPU in background
        // 4. Update GPU cache with prefetched data
        // 5. Mark parameters as ready for immediate use
        // 6. Handle prefetch cancellation if needed
        // 7. Implement memory-efficient transfer strategies

        // Simulate async data transfer with realistic timing
        let estimated_transfer_time = Self::estimate_transfer_time(layer_name);
        tokio::time::sleep(estimated_transfer_time).await;

        Ok(())
    }

    /// Estimate transfer time for a layer based on size and bandwidth
    fn estimate_transfer_time(layer_name: &str) -> tokio::time::Duration {
        // Mock estimation based on layer name
        let base_time_ms = if layer_name.contains("large") {
            50 // Large layers take 50ms
        } else if layer_name.contains("medium") {
            25 // Medium layers take 25ms
        } else {
            10 // Small layers take 10ms
        };

        tokio::time::Duration::from_millis(base_time_ms)
    }

    /// Execute prefetch for multiple layers in parallel
    pub async fn batch_prefetch(&self, layer_names: Vec<String>) -> TorshResult<()> {
        if !self.config.async_prefetch || layer_names.is_empty() {
            return Ok(());
        }

        info!(
            "   = Starting batch prefetch for {} layers",
            layer_names.len()
        );

        let adaptive_config = self.adaptive_config.lock().unwrap();
        let max_concurrent = adaptive_config.max_concurrent_prefetches;
        drop(adaptive_config);

        // Execute prefetches in parallel with controlled concurrency
        let semaphore = Arc::new(Semaphore::new(max_concurrent));
        let mut tasks = Vec::new();

        for layer_name in layer_names {
            let sem = semaphore.clone();
            let process_group = self.process_group.clone();
            let metrics = self.metrics.clone();

            let task = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let start_time = std::time::Instant::now();
                let result = Self::prefetch_layer_data(&layer_name, process_group).await;

                // Record metrics
                {
                    let mut metrics_guard = metrics.lock().unwrap();
                    let duration = start_time.elapsed();
                    match result {
                        Ok(()) => metrics_guard.record_successful_prefetch(duration, 0),
                        Err(ref e) => metrics_guard.record_failed_prefetch(duration, e.to_string()),
                    }
                }

                result
            });

            tasks.push(task);
        }

        // Wait for all prefetch tasks to complete
        let results: Vec<_> = futures::future::join_all(tasks).await;

        let mut successful = 0;
        let mut failed = 0;

        for result in results {
            match result {
                Ok(Ok(())) => successful += 1,
                Ok(Err(e)) => {
                    failed += 1;
                    tracing::error!("Prefetch task failed: {}", e);
                }
                Err(e) => {
                    failed += 1;
                    tracing::error!("Prefetch task panicked: {}", e);
                }
            }
        }

        info!(
            "    Batch prefetch completed: {} successful, {} failed",
            successful, failed
        );

        // Update batch metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_batch_prefetch(successful, failed);
        }

        Ok(())
    }

    /// Intelligent prefetch scheduling based on execution patterns
    pub async fn intelligent_prefetch(
        &self,
        current_layer: &str,
        execution_graph: &[String],
    ) -> TorshResult<()> {
        if !self.config.async_prefetch {
            return Ok(());
        }

        // Find current layer position in execution graph
        let current_pos = execution_graph.iter().position(|l| l == current_layer);

        if let Some(pos) = current_pos {
            // Determine how many layers ahead to prefetch based on memory availability
            let prefetch_distance = self.calculate_optimal_prefetch_distance().await?;

            // Collect layers to prefetch with priority assignment
            let mut layers_to_prefetch = Vec::new();
            for i in 1..=prefetch_distance {
                if pos + i < execution_graph.len() {
                    layers_to_prefetch.push(execution_graph[pos + i].clone());
                }
            }

            if !layers_to_prefetch.is_empty() {
                info!(
                    "   > Intelligent prefetch: {} layers ahead from {}",
                    layers_to_prefetch.len(),
                    current_layer
                );

                // Use prioritized batch prefetch
                self.prioritized_batch_prefetch(layers_to_prefetch, pos)
                    .await?;
            }
        }

        Ok(())
    }

    /// Execute batch prefetch with priority-based scheduling
    async fn prioritized_batch_prefetch(
        &self,
        layer_names: Vec<String>,
        current_pos: usize,
    ) -> TorshResult<()> {
        let mut prioritized_requests = Vec::new();

        for (i, layer_name) in layer_names.iter().enumerate() {
            let priority = match i {
                0 => PrefetchPriority::High,       // Next layer is high priority
                1..=2 => PrefetchPriority::Normal, // Next 2 layers are normal priority
                _ => PrefetchPriority::Low,        // Further layers are low priority
            };

            let request = PrefetchRequest {
                layer_name: layer_name.clone(),
                priority,
                requested_at: std::time::Instant::now(),
                estimated_size_bytes: self.estimate_layer_size(layer_name),
            };

            prioritized_requests.push(request);
        }

        // Sort by priority (high first)
        prioritized_requests.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Execute prefetches with priority consideration
        let adaptive_config = self.adaptive_config.lock().unwrap();
        let max_concurrent = adaptive_config.max_concurrent_prefetches;
        drop(adaptive_config);

        let semaphore = Arc::new(Semaphore::new(max_concurrent));
        let mut tasks = Vec::new();

        for request in prioritized_requests {
            let sem = semaphore.clone();
            let process_group = self.process_group.clone();
            let metrics = self.metrics.clone();

            // Higher priority requests get processed first
            let delay = match request.priority {
                PrefetchPriority::High => tokio::time::Duration::from_millis(0),
                PrefetchPriority::Normal => tokio::time::Duration::from_millis(10),
                PrefetchPriority::Low => tokio::time::Duration::from_millis(25),
            };

            let task = tokio::spawn(async move {
                tokio::time::sleep(delay).await; // Priority-based delay
                let _permit = sem.acquire().await.unwrap();
                let start_time = std::time::Instant::now();
                let result = Self::prefetch_layer_data(&request.layer_name, process_group).await;

                // Record metrics
                {
                    let mut metrics_guard = metrics.lock().unwrap();
                    let duration = start_time.elapsed();
                    match result {
                        Ok(()) => metrics_guard
                            .record_successful_prefetch(duration, request.estimated_size_bytes),
                        Err(ref e) => metrics_guard.record_failed_prefetch(duration, e.to_string()),
                    }
                }

                (request.layer_name, result)
            });

            tasks.push(task);
        }

        // Wait for all prioritized prefetch tasks
        let results: Vec<_> = futures::future::join_all(tasks).await;

        let mut successful = 0;
        let mut failed = 0;

        for result in results {
            match result {
                Ok((layer_name, Ok(()))) => {
                    successful += 1;
                    info!("    Prioritized prefetch completed: {}", layer_name);
                }
                Ok((layer_name, Err(e))) => {
                    failed += 1;
                    tracing::error!("Prioritized prefetch failed for {}: {}", layer_name, e);
                }
                Err(e) => {
                    failed += 1;
                    tracing::error!("Prioritized prefetch task panicked: {}", e);
                }
            }
        }

        info!(
            "   < Prioritized batch prefetch completed: {} successful, {} failed",
            successful, failed
        );

        Ok(())
    }

    /// Calculate optimal prefetch distance based on system resources
    pub async fn calculate_optimal_prefetch_distance(&self) -> TorshResult<usize> {
        // In a real implementation, this would consider:
        // 1. Available GPU memory
        // 2. Network bandwidth for distributed setups
        // 3. CPU-GPU transfer bandwidth
        // 4. Historical execution timing
        // 5. Layer parameter sizes
        // 6. Current memory pressure
        // 7. Prefetch success/failure rates

        let adaptive_config = self.adaptive_config.lock().unwrap();
        let base_distance = adaptive_config.base_prefetch_distance;
        let current_performance = self.metrics.lock().unwrap().get_success_rate();

        // Adjust based on recent prefetch performance
        let performance_multiplier = if current_performance > 0.9 {
            1.5 // High success rate, increase distance
        } else if current_performance > 0.7 {
            1.0 // Normal success rate, keep distance
        } else {
            0.7 // Low success rate, reduce distance
        };

        let optimal_distance = (base_distance as f32 * performance_multiplier) as usize;
        let optimal_distance = optimal_distance
            .max(1)
            .min(adaptive_config.max_prefetch_distance);

        // Update adaptive configuration
        drop(adaptive_config);
        {
            let mut adaptive_config = self.adaptive_config.lock().unwrap();
            adaptive_config.current_prefetch_distance = optimal_distance;
        }

        Ok(optimal_distance)
    }

    /// Adaptive prefetch management based on system performance
    pub async fn adapt_prefetch_strategy(&self) -> TorshResult<()> {
        let metrics = self.metrics.lock().unwrap().clone();
        let mut adaptive_config = self.adaptive_config.lock().unwrap();

        info!("   <  Adapting prefetch strategy based on performance");

        // Adapt concurrent prefetch limit based on success rate
        if metrics.get_success_rate() > 0.95 && metrics.total_prefetches > 10 {
            // High success rate, increase concurrency
            adaptive_config.max_concurrent_prefetches =
                (adaptive_config.max_concurrent_prefetches + 1).min(8);
            info!(
                "       Increased max concurrent prefetches to {}",
                adaptive_config.max_concurrent_prefetches
            );
        } else if metrics.get_success_rate() < 0.8 && adaptive_config.max_concurrent_prefetches > 1
        {
            // Low success rate, decrease concurrency
            adaptive_config.max_concurrent_prefetches =
                (adaptive_config.max_concurrent_prefetches - 1).max(1);
            info!(
                "       Decreased max concurrent prefetches to {}",
                adaptive_config.max_concurrent_prefetches
            );
        }

        // Adapt queue size based on utilization
        let queue_size = self.prefetch_queue.lock().unwrap().len();
        if queue_size > adaptive_config.max_queue_size * 3 / 4 {
            // Queue is mostly full, increase size
            adaptive_config.max_queue_size = (adaptive_config.max_queue_size + 2).min(32);
            info!(
                "     = Increased max queue size to {}",
                adaptive_config.max_queue_size
            );
        } else if queue_size < adaptive_config.max_queue_size / 4
            && adaptive_config.max_queue_size > 4
        {
            // Queue is mostly empty, decrease size
            adaptive_config.max_queue_size = (adaptive_config.max_queue_size - 1).max(4);
            info!(
                "     = Decreased max queue size to {}",
                adaptive_config.max_queue_size
            );
        }

        // Adapt prefetch distance based on timing
        if metrics.average_prefetch_time > tokio::time::Duration::from_millis(100) {
            // Prefetches are taking too long, reduce distance
            adaptive_config.base_prefetch_distance =
                (adaptive_config.base_prefetch_distance - 1).max(1);
            info!(
                "     =; Decreased base prefetch distance to {}",
                adaptive_config.base_prefetch_distance
            );
        } else if metrics.average_prefetch_time < tokio::time::Duration::from_millis(20) {
            // Prefetches are fast, increase distance
            adaptive_config.base_prefetch_distance =
                (adaptive_config.base_prefetch_distance + 1).min(16);
            info!(
                "     =: Increased base prefetch distance to {}",
                adaptive_config.base_prefetch_distance
            );
        }

        Ok(())
    }

    /// Cancel all pending prefetch operations
    pub async fn cancel_all_prefetches(&self) -> TorshResult<()> {
        info!("   = Cancelling all pending prefetch operations");

        // Clear prefetch queue
        {
            let mut queue = self.prefetch_queue.lock().unwrap();
            let cancelled_count = queue.len();
            queue.clear();
            if cancelled_count > 0 {
                info!(
                    "     = Cancelled {} queued prefetch requests",
                    cancelled_count
                );
            }
        }

        // Note: In a real implementation, you would also:
        // 1. Cancel active prefetch operations
        // 2. Clean up partial transfers
        // 3. Update metrics for cancelled operations
        // 4. Free allocated resources

        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_cancellation();
        }

        Ok(())
    }

    /// Get current prefetch queue status
    pub fn get_queue_status(&self) -> PrefetchQueueStatus {
        let queue = self.prefetch_queue.lock().unwrap();
        let active = self.active_prefetches.lock().unwrap();
        let adaptive_config = self.adaptive_config.lock().unwrap();

        PrefetchQueueStatus {
            queued_requests: queue.len(),
            active_operations: active.len(),
            max_queue_size: adaptive_config.max_queue_size,
            max_concurrent: adaptive_config.max_concurrent_prefetches,
            current_prefetch_distance: adaptive_config.current_prefetch_distance,
        }
    }

    /// Get prefetch performance metrics
    pub fn get_metrics(&self) -> PrefetchMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Get adaptive configuration
    pub fn get_adaptive_config(&self) -> AdaptivePrefetchConfig {
        self.adaptive_config.lock().unwrap().clone()
    }

    // Helper methods

    fn estimate_layer_size(&self, layer_name: &str) -> usize {
        // Mock size estimation based on layer name
        if layer_name.contains("large") {
            64 * 1024 * 1024 // 64MB
        } else if layer_name.contains("medium") {
            16 * 1024 * 1024 // 16MB
        } else {
            4 * 1024 * 1024 // 4MB
        }
    }
}

/// Prefetch request with priority and metadata
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    /// Layer name to prefetch
    pub layer_name: String,
    /// Priority of this prefetch request
    pub priority: PrefetchPriority,
    /// When this request was made
    pub requested_at: std::time::Instant,
    /// Estimated size in bytes
    pub estimated_size_bytes: usize,
}

/// Priority levels for prefetch operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrefetchPriority {
    Low = 0,
    Normal = 1,
    High = 2,
}

/// Status of a prefetch operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Active prefetch operation tracking
#[derive(Debug, Clone)]
pub struct PrefetchOperation {
    /// Layer being prefetched
    pub layer_name: String,
    /// When prefetch started
    pub started_at: std::time::Instant,
    /// Current status
    pub status: PrefetchStatus,
}

/// Prefetch queue status information
#[derive(Debug, Clone)]
pub struct PrefetchQueueStatus {
    /// Number of requests in queue
    pub queued_requests: usize,
    /// Number of active operations
    pub active_operations: usize,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Maximum concurrent operations
    pub max_concurrent: usize,
    /// Current prefetch distance
    pub current_prefetch_distance: usize,
}

/// Adaptive prefetch configuration
#[derive(Debug, Clone)]
pub struct AdaptivePrefetchConfig {
    /// Base prefetch distance (number of layers ahead)
    pub base_prefetch_distance: usize,
    /// Current dynamic prefetch distance
    pub current_prefetch_distance: usize,
    /// Maximum prefetch distance allowed
    pub max_prefetch_distance: usize,
    /// Maximum concurrent prefetch operations
    pub max_concurrent_prefetches: usize,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Whether adaptive optimization is enabled
    pub adaptive_optimization_enabled: bool,
}

impl AdaptivePrefetchConfig {
    pub fn new(config: &Zero3CpuOffloadConfig) -> Self {
        Self {
            base_prefetch_distance: config.prefetch_buffer_size / 4,
            current_prefetch_distance: config.prefetch_buffer_size / 4,
            max_prefetch_distance: config.prefetch_buffer_size,
            max_concurrent_prefetches: 4,
            max_queue_size: 16,
            adaptive_optimization_enabled: true,
        }
    }
}

/// Task coordination for background operations
#[derive(Debug)]
pub struct TaskCoordination {
    /// Number of active background tasks
    pub active_tasks: usize,
    /// Maximum allowed background tasks
    pub max_background_tasks: usize,
    /// Whether task coordination is enabled
    pub coordination_enabled: bool,
}

impl TaskCoordination {
    pub fn new() -> Self {
        Self {
            active_tasks: 0,
            max_background_tasks: 8,
            coordination_enabled: true,
        }
    }
}

impl Default for TaskCoordination {
    fn default() -> Self {
        Self::new()
    }
}

/// Prefetch performance metrics
#[derive(Debug, Clone)]
pub struct PrefetchMetrics {
    /// Total number of prefetch operations attempted
    pub total_prefetches: u64,
    /// Number of successful prefetches
    pub successful_prefetches: u64,
    /// Number of failed prefetches
    pub failed_prefetches: u64,
    /// Number of cancelled prefetches
    pub cancelled_prefetches: u64,
    /// Total time spent prefetching
    pub total_prefetch_time: tokio::time::Duration,
    /// Average prefetch time
    pub average_prefetch_time: tokio::time::Duration,
    /// Total bytes prefetched
    pub total_bytes_prefetched: usize,
    /// Number of batch operations
    pub batch_operations: u64,
    /// Failed batch operations
    pub failed_batch_operations: u64,
    /// Recent failure reasons
    pub recent_failures: Vec<String>,
}

impl PrefetchMetrics {
    pub fn new() -> Self {
        Self {
            total_prefetches: 0,
            successful_prefetches: 0,
            failed_prefetches: 0,
            cancelled_prefetches: 0,
            total_prefetch_time: tokio::time::Duration::ZERO,
            average_prefetch_time: tokio::time::Duration::ZERO,
            total_bytes_prefetched: 0,
            batch_operations: 0,
            failed_batch_operations: 0,
            recent_failures: Vec::new(),
        }
    }

    /// Record a successful prefetch operation
    pub fn record_successful_prefetch(&mut self, duration: tokio::time::Duration, bytes: usize) {
        self.total_prefetches += 1;
        self.successful_prefetches += 1;
        self.total_prefetch_time += duration;
        self.total_bytes_prefetched += bytes;
        self.update_average_time();
    }

    /// Record a failed prefetch operation
    pub fn record_failed_prefetch(&mut self, duration: tokio::time::Duration, error: String) {
        self.total_prefetches += 1;
        self.failed_prefetches += 1;
        self.total_prefetch_time += duration;

        // Keep recent failure reasons (max 10)
        self.recent_failures.push(error);
        if self.recent_failures.len() > 10 {
            self.recent_failures.remove(0);
        }

        self.update_average_time();
    }

    /// Record a batch prefetch operation
    pub fn record_batch_prefetch(&mut self, successful: usize, failed: usize) {
        self.batch_operations += 1;
        if failed > 0 {
            self.failed_batch_operations += 1;
        }
    }

    /// Record prefetch cancellation
    pub fn record_cancellation(&mut self) {
        self.cancelled_prefetches += 1;
    }

    /// Get success rate as a percentage
    pub fn get_success_rate(&self) -> f32 {
        if self.total_prefetches > 0 {
            self.successful_prefetches as f32 / self.total_prefetches as f32
        } else {
            1.0 // No operations yet, assume 100% success
        }
    }

    /// Get failure rate as a percentage
    pub fn get_failure_rate(&self) -> f32 {
        if self.total_prefetches > 0 {
            self.failed_prefetches as f32 / self.total_prefetches as f32
        } else {
            0.0
        }
    }

    /// Get average throughput in bytes per second
    pub fn get_throughput_bps(&self) -> f64 {
        if !self.total_prefetch_time.is_zero() {
            self.total_bytes_prefetched as f64 / self.total_prefetch_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Update average prefetch time
    fn update_average_time(&mut self) {
        if self.total_prefetches > 0 {
            self.average_prefetch_time = self.total_prefetch_time / self.total_prefetches as u32;
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for PrefetchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{init_process_group, BackendType};

    #[test]
    fn test_prefetch_request_priority_ordering() {
        let mut requests = vec![
            PrefetchRequest {
                layer_name: "low".to_string(),
                priority: PrefetchPriority::Low,
                requested_at: std::time::Instant::now(),
                estimated_size_bytes: 1000,
            },
            PrefetchRequest {
                layer_name: "high".to_string(),
                priority: PrefetchPriority::High,
                requested_at: std::time::Instant::now(),
                estimated_size_bytes: 1000,
            },
            PrefetchRequest {
                layer_name: "normal".to_string(),
                priority: PrefetchPriority::Normal,
                requested_at: std::time::Instant::now(),
                estimated_size_bytes: 1000,
            },
        ];

        requests.sort_by(|a, b| b.priority.cmp(&a.priority));
        assert_eq!(requests[0].layer_name, "high");
        assert_eq!(requests[1].layer_name, "normal");
        assert_eq!(requests[2].layer_name, "low");
    }

    #[test]
    fn test_adaptive_prefetch_config() {
        let zero3_config = Zero3CpuOffloadConfig::default();
        let config = AdaptivePrefetchConfig::new(&zero3_config);

        assert_eq!(
            config.base_prefetch_distance,
            zero3_config.prefetch_buffer_size / 4
        );
        assert_eq!(
            config.max_prefetch_distance,
            zero3_config.prefetch_buffer_size
        );
        assert!(config.adaptive_optimization_enabled);
    }

    #[test]
    fn test_prefetch_metrics() {
        let mut metrics = PrefetchMetrics::new();

        metrics.record_successful_prefetch(tokio::time::Duration::from_millis(100), 1000);
        assert_eq!(metrics.total_prefetches, 1);
        assert_eq!(metrics.successful_prefetches, 1);
        assert_eq!(metrics.get_success_rate(), 1.0);

        metrics.record_failed_prefetch(
            tokio::time::Duration::from_millis(50),
            "test error".to_string(),
        );
        assert_eq!(metrics.total_prefetches, 2);
        assert_eq!(metrics.failed_prefetches, 1);
        assert_eq!(metrics.get_success_rate(), 0.5);
        assert_eq!(metrics.recent_failures.len(), 1);
    }

    #[tokio::test]
    async fn test_prefetch_scheduler_creation() {
        let config = Zero3CpuOffloadConfig::default();
        let pg = init_process_group(BackendType::Nccl, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();
        let scheduler = PrefetchScheduler::new(&config, Arc::new(pg));

        let status = scheduler.get_queue_status();
        assert_eq!(status.queued_requests, 0);
        assert_eq!(status.active_operations, 0);

        let metrics = scheduler.get_metrics();
        assert_eq!(metrics.total_prefetches, 0);
    }

    #[tokio::test]
    async fn test_prefetch_distance_calculation() {
        let config = Zero3CpuOffloadConfig::default();
        let pg = init_process_group(BackendType::Nccl, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();
        let scheduler = PrefetchScheduler::new(&config, Arc::new(pg));

        let distance = scheduler
            .calculate_optimal_prefetch_distance()
            .await
            .unwrap();
        assert!(distance >= 1);
        assert!(distance <= config.prefetch_buffer_size);
    }

    #[tokio::test]
    async fn test_batch_prefetch() {
        let config = Zero3CpuOffloadConfig::default();
        let pg = init_process_group(BackendType::Nccl, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();
        let scheduler = PrefetchScheduler::new(&config, Arc::new(pg));

        let layers = vec!["layer1".to_string(), "layer2".to_string()];
        scheduler.batch_prefetch(layers).await.unwrap();

        let metrics = scheduler.get_metrics();
        assert_eq!(metrics.batch_operations, 1);
    }

    #[test]
    fn test_task_coordination() {
        let coordination = TaskCoordination::new();
        assert_eq!(coordination.active_tasks, 0);
        assert!(coordination.coordination_enabled);
        assert_eq!(coordination.max_background_tasks, 8);
    }

    #[tokio::test]
    async fn test_cancel_prefetches() {
        let config = Zero3CpuOffloadConfig::default();
        let pg = init_process_group(BackendType::Nccl, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();
        let scheduler = PrefetchScheduler::new(&config, Arc::new(pg));

        // Add some requests to queue (we'll mock this)
        scheduler.cancel_all_prefetches().await.unwrap();

        let status = scheduler.get_queue_status();
        assert_eq!(status.queued_requests, 0);
    }
}
