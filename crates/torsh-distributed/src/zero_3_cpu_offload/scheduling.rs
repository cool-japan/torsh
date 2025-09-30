//! Scheduling and Prefetch Coordination for ZeRO-3 CPU Offloading
//!
//! This module provides intelligent prefetch scheduling and asynchronous coordination
//! for ZeRO-3 distributed training. It implements sophisticated strategies for predicting
//! parameter needs and proactively loading them from CPU to GPU memory to minimize
//! training latency.

use crate::{ProcessGroup, TorshDistributedError, TorshResult};
use log::{debug, info, warn};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

use super::config::Zero3CpuOffloadConfig;

/// Prefetch scheduler for asynchronous parameter loading
///
/// Manages intelligent prefetching of parameters from CPU to GPU memory
/// based on execution patterns, memory availability, and performance metrics.
pub struct PrefetchScheduler {
    config: Zero3CpuOffloadConfig,
    process_group: Arc<ProcessGroup>,
    prefetch_queue: Mutex<VecDeque<PrefetchRequest>>,
    execution_history: Mutex<HashMap<String, ExecutionHistory>>,
    performance_metrics: Mutex<PrefetchPerformanceMetrics>,
    active_prefetch_tasks: Mutex<HashMap<String, tokio::task::JoinHandle<TorshResult<()>>>>,
}

impl PrefetchScheduler {
    /// Create a new prefetch scheduler
    pub fn new(config: &Zero3CpuOffloadConfig, process_group: Arc<ProcessGroup>) -> Self {
        info!(
            "⏰ Prefetch Scheduler initialized: async={}, buffer_size={}, overlap={}",
            config.async_prefetch, config.prefetch_buffer_size, config.overlap_computation
        );

        Self {
            config: config.clone(),
            process_group,
            prefetch_queue: Mutex::new(VecDeque::new()),
            execution_history: Mutex::new(HashMap::new()),
            performance_metrics: Mutex::new(PrefetchPerformanceMetrics::new()),
            active_prefetch_tasks: Mutex::new(HashMap::new()),
        }
    }

    /// Schedule a single layer for prefetching
    pub async fn schedule_prefetch(&self, layer_name: &str, priority: PrefetchPriority) -> TorshResult<()> {
        if !self.config.async_prefetch {
            return Ok(());
        }

        let request = PrefetchRequest {
            layer_name: layer_name.to_string(),
            priority,
            requested_at: Instant::now(),
            estimated_completion_time: None,
        };

        {
            let mut queue = self.prefetch_queue.lock().unwrap();

            // Insert based on priority (higher priority first)
            let insert_pos = queue.iter().position(|req| req.priority < priority)
                .unwrap_or(queue.len());

            queue.insert(insert_pos, request);
        }

        info!("   ⏰ Scheduled prefetch for layer: {} (priority: {:?})", layer_name, priority);

        // Start processing the queue
        self.process_prefetch_queue().await?;

        Ok(())
    }

    /// Process the prefetch queue
    async fn process_prefetch_queue(&self) -> TorshResult<()> {
        let mut requests_to_process = Vec::new();

        // Collect requests that need processing
        {
            let mut queue = self.prefetch_queue.lock().unwrap();
            let max_concurrent = self.config.prefetch_buffer_size.min(4); // Limit concurrent tasks

            let active_count = self.active_prefetch_tasks.lock().unwrap().len();
            let can_process = max_concurrent.saturating_sub(active_count);

            for _ in 0..can_process {
                if let Some(request) = queue.pop_front() {
                    requests_to_process.push(request);
                } else {
                    break;
                }
            }
        }

        // Process each request
        for request in requests_to_process {
            self.execute_async_prefetch(request).await?;
        }

        Ok(())
    }

    /// Execute asynchronous prefetching for a layer
    async fn execute_async_prefetch(&self, request: PrefetchRequest) -> TorshResult<()> {
        let layer_name = request.layer_name.clone();
        let process_group = self.process_group.clone();
        let start_time = Instant::now();

        // Record start time in performance metrics
        {
            let mut metrics = self.performance_metrics.lock().unwrap();
            metrics.prefetch_started(&layer_name, start_time);
        }

        // Spawn background task for prefetching
        let task = tokio::spawn(async move {
            let result = Self::prefetch_layer_data(&layer_name, process_group).await;
            if let Err(ref e) = result {
                tracing::error!("Async prefetch failed for layer {}: {}", layer_name, e);
            }
            result
        });

        // Store the task handle
        {
            let mut active_tasks = self.active_prefetch_tasks.lock().unwrap();
            active_tasks.insert(layer_name.clone(), task);
        }

        // Clean up completed tasks in the background
        self.cleanup_completed_tasks().await;

        Ok(())
    }

    /// Clean up completed prefetch tasks
    async fn cleanup_completed_tasks(&self) {
        let mut completed_tasks = Vec::new();

        // Check which tasks are completed
        {
            let mut active_tasks = self.active_prefetch_tasks.lock().unwrap();
            let mut to_remove = Vec::new();

            for (layer_name, task) in active_tasks.iter() {
                if task.is_finished() {
                    to_remove.push(layer_name.clone());
                }
            }

            for layer_name in to_remove {
                if let Some(task) = active_tasks.remove(&layer_name) {
                    completed_tasks.push((layer_name, task));
                }
            }
        }

        // Await completed tasks and record metrics
        for (layer_name, task) in completed_tasks {
            let end_time = Instant::now();
            match task.await {
                Ok(Ok(())) => {
                    let mut metrics = self.performance_metrics.lock().unwrap();
                    metrics.prefetch_completed(&layer_name, end_time, true);
                }
                Ok(Err(_)) | Err(_) => {
                    let mut metrics = self.performance_metrics.lock().unwrap();
                    metrics.prefetch_completed(&layer_name, end_time, false);
                }
            }
        }
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

        // Simulate async data transfer with realistic timing
        let transfer_time = Duration::from_millis(5 + (layer_name.len() % 20) as u64);
        tokio::time::sleep(transfer_time).await;

        info!("   📤 Async prefetch completed for layer: {}", layer_name);
        Ok(())
    }

    /// Execute prefetch for multiple layers in parallel
    pub async fn batch_prefetch(&self, layer_names: Vec<String>, priority: PrefetchPriority) -> TorshResult<BatchPrefetchResult> {
        if !self.config.async_prefetch || layer_names.is_empty() {
            return Ok(BatchPrefetchResult::default());
        }

        let start_time = Instant::now();

        info!(
            "    Starting batch prefetch for {} layers",
            layer_names.len()
        );

        // Execute prefetches in parallel with controlled concurrency
        let max_concurrent = self.config.prefetch_buffer_size.min(4);
        let semaphore = Arc::new(Semaphore::new(max_concurrent));
        let mut tasks = Vec::new();

        for layer_name in layer_names.iter() {
            let sem = semaphore.clone();
            let process_group = self.process_group.clone();
            let layer_name_clone = layer_name.clone();

            let task = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                Self::prefetch_layer_data(&layer_name_clone, process_group).await
            });

            tasks.push((layer_name.clone(), task));
        }

        // Wait for all prefetch tasks to complete
        let results = futures::future::join_all(tasks.into_iter().map(|(_, task)| task)).await;

        let mut successful = 0;
        let mut failed = 0;
        let mut failed_layers = Vec::new();

        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(Ok(())) => successful += 1,
                Ok(Err(e)) => {
                    failed += 1;
                    failed_layers.push(layer_names[i].clone());
                    tracing::error!("Prefetch task failed for {}: {}", layer_names[i], e);
                }
                Err(e) => {
                    failed += 1;
                    failed_layers.push(layer_names[i].clone());
                    tracing::error!("Prefetch task panicked for {}: {}", layer_names[i], e);
                }
            }
        }

        let duration = start_time.elapsed();

        info!(
            "    Batch prefetch completed: {} successful, {} failed in {:?}",
            successful, failed, duration
        );

        Ok(BatchPrefetchResult {
            total_layers: layer_names.len(),
            successful,
            failed,
            failed_layers,
            duration,
        })
    }

    /// Intelligent prefetch scheduling based on execution patterns
    pub async fn intelligent_prefetch(
        &self,
        current_layer: &str,
        execution_graph: &[String],
    ) -> TorshResult<IntelligentPrefetchResult> {
        if !self.config.async_prefetch {
            return Ok(IntelligentPrefetchResult::default());
        }

        let start_time = Instant::now();

        // Update execution history
        self.update_execution_history(current_layer).await;

        // Find current layer position in execution graph
        let current_pos = execution_graph.iter().position(|l| l == current_layer);

        if let Some(pos) = current_pos {
            // Determine how many layers ahead to prefetch based on memory availability
            let prefetch_distance = self.calculate_optimal_prefetch_distance().await?;

            // Collect layers to prefetch with intelligent prioritization
            let mut layers_to_prefetch = Vec::new();
            for i in 1..=prefetch_distance {
                if pos + i < execution_graph.len() {
                    let layer_name = &execution_graph[pos + i];
                    let priority = self.calculate_layer_priority(layer_name, i).await;
                    layers_to_prefetch.push((layer_name.clone(), priority));
                }
            }

            if !layers_to_prefetch.is_empty() {
                info!(
                    "   🧠 Intelligent prefetch: {} layers ahead from {}",
                    layers_to_prefetch.len(),
                    current_layer
                );

                // Sort by priority and prefetch
                layers_to_prefetch.sort_by(|a, b| b.1.cmp(&a.1));
                let layer_names: Vec<String> = layers_to_prefetch.iter().map(|(name, _)| name.clone()).collect();
                let highest_priority = layers_to_prefetch.first().map(|(_, p)| *p).unwrap_or(PrefetchPriority::Medium);

                let batch_result = self.batch_prefetch(layer_names.clone(), highest_priority).await?;

                return Ok(IntelligentPrefetchResult {
                    layers_scheduled: layer_names,
                    prefetch_distance,
                    duration: start_time.elapsed(),
                    batch_result: Some(batch_result),
                });
            }
        }

        Ok(IntelligentPrefetchResult {
            layers_scheduled: Vec::new(),
            prefetch_distance: 0,
            duration: start_time.elapsed(),
            batch_result: None,
        })
    }

    /// Update execution history for a layer
    async fn update_execution_history(&self, layer_name: &str) {
        let mut history = self.execution_history.lock().unwrap();
        let entry = history.entry(layer_name.to_string()).or_insert_with(ExecutionHistory::new);
        entry.record_execution(Instant::now());
    }

    /// Calculate priority for a layer based on historical patterns
    async fn calculate_layer_priority(&self, layer_name: &str, distance: usize) -> PrefetchPriority {
        let history = self.execution_history.lock().unwrap();

        if let Some(layer_history) = history.get(layer_name) {
            // Higher priority for frequently accessed layers
            let frequency_score = layer_history.access_frequency();

            // Lower priority for layers further away
            let distance_penalty = 1.0 / (distance as f32 + 1.0);

            let combined_score = frequency_score * distance_penalty;

            if combined_score > 0.7 {
                PrefetchPriority::High
            } else if combined_score > 0.3 {
                PrefetchPriority::Medium
            } else {
                PrefetchPriority::Low
            }
        } else {
            // Default priority for unknown layers
            match distance {
                1 => PrefetchPriority::High,
                2..=3 => PrefetchPriority::Medium,
                _ => PrefetchPriority::Low,
            }
        }
    }

    /// Calculate optimal prefetch distance based on system resources
    async fn calculate_optimal_prefetch_distance(&self) -> TorshResult<usize> {
        // In a real implementation, this would consider:
        // 1. Available GPU memory
        // 2. Network bandwidth for distributed setups
        // 3. CPU-GPU transfer bandwidth
        // 4. Historical execution timing
        // 5. Layer parameter sizes

        let base_distance = self.config.prefetch_buffer_size / 4; // Conservative estimate

        // Adjust based on performance history
        let metrics = self.performance_metrics.lock().unwrap();
        let success_rate = metrics.overall_success_rate();

        let adjusted_distance = if success_rate > 0.9 {
            base_distance + 2 // Increase if prefetching is working well
        } else if success_rate < 0.7 {
            base_distance.saturating_sub(1) // Decrease if having issues
        } else {
            base_distance
        };

        let optimal_distance = adjusted_distance.max(1).min(8); // Between 1 and 8 layers

        Ok(optimal_distance)
    }

    /// Cancel prefetch for a specific layer
    pub async fn cancel_prefetch(&self, layer_name: &str) -> TorshResult<bool> {
        // Remove from queue if not yet started
        {
            let mut queue = self.prefetch_queue.lock().unwrap();
            if let Some(pos) = queue.iter().position(|req| req.layer_name == layer_name) {
                queue.remove(pos);
                info!("   ❌ Cancelled queued prefetch for layer: {}", layer_name);
                return Ok(true);
            }
        }

        // Cancel active task if running
        {
            let mut active_tasks = self.active_prefetch_tasks.lock().unwrap();
            if let Some(task) = active_tasks.remove(layer_name) {
                task.abort();
                info!("   ❌ Cancelled active prefetch for layer: {}", layer_name);
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get prefetch scheduler statistics
    pub fn get_statistics(&self) -> PrefetchSchedulerStats {
        let queue_length = self.prefetch_queue.lock().unwrap().len();
        let active_tasks = self.active_prefetch_tasks.lock().unwrap().len();
        let metrics = self.performance_metrics.lock().unwrap().clone();
        let history_entries = self.execution_history.lock().unwrap().len();

        PrefetchSchedulerStats {
            queue_length,
            active_tasks,
            performance_metrics: metrics,
            tracked_layers: history_entries,
            async_prefetch_enabled: self.config.async_prefetch,
            prefetch_buffer_size: self.config.prefetch_buffer_size,
            overlap_computation: self.config.overlap_computation,
        }
    }

    /// Clear all prefetch queues and cancel active tasks
    pub async fn clear_all(&self) -> TorshResult<()> {
        // Clear queue
        {
            let mut queue = self.prefetch_queue.lock().unwrap();
            queue.clear();
        }

        // Cancel all active tasks
        let active_tasks: Vec<_> = {
            let mut tasks = self.active_prefetch_tasks.lock().unwrap();
            tasks.drain().collect()
        };

        for (layer_name, task) in active_tasks {
            task.abort();
            info!("   ❌ Cancelled prefetch task for: {}", layer_name);
        }

        info!("   🧹 Cleared all prefetch queues and tasks");
        Ok(())
    }
}

/// Priority levels for prefetch requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrefetchPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// A prefetch request in the scheduler queue
#[derive(Debug, Clone)]
struct PrefetchRequest {
    layer_name: String,
    priority: PrefetchPriority,
    requested_at: Instant,
    estimated_completion_time: Option<Instant>,
}

/// Execution history for a layer
#[derive(Debug, Clone)]
struct ExecutionHistory {
    access_times: VecDeque<Instant>,
    total_accesses: usize,
    last_access: Option<Instant>,
}

impl ExecutionHistory {
    fn new() -> Self {
        Self {
            access_times: VecDeque::new(),
            total_accesses: 0,
            last_access: None,
        }
    }

    fn record_execution(&mut self, time: Instant) {
        self.access_times.push_back(time);
        self.total_accesses += 1;
        self.last_access = Some(time);

        // Keep only recent access times (last 50)
        if self.access_times.len() > 50 {
            self.access_times.pop_front();
        }
    }

    fn access_frequency(&self) -> f32 {
        if self.access_times.len() < 2 {
            return 0.0;
        }

        let time_span = self.access_times.back().unwrap()
            .duration_since(*self.access_times.front().unwrap());

        if time_span.is_zero() {
            return 1.0;
        }

        (self.access_times.len() as f32) / time_span.as_secs_f32()
    }
}

/// Performance metrics for prefetch operations
#[derive(Debug, Clone)]
pub struct PrefetchPerformanceMetrics {
    total_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    total_duration: Duration,
    layer_metrics: HashMap<String, LayerPrefetchMetrics>,
}

impl PrefetchPerformanceMetrics {
    fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_duration: Duration::ZERO,
            layer_metrics: HashMap::new(),
        }
    }

    fn prefetch_started(&mut self, layer_name: &str, start_time: Instant) {
        self.total_requests += 1;
        let entry = self.layer_metrics.entry(layer_name.to_string())
            .or_insert_with(LayerPrefetchMetrics::new);
        entry.record_start(start_time);
    }

    fn prefetch_completed(&mut self, layer_name: &str, end_time: Instant, success: bool) {
        if success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }

        if let Some(entry) = self.layer_metrics.get_mut(layer_name) {
            entry.record_completion(end_time, success);
            if let Some(duration) = entry.last_duration {
                self.total_duration += duration;
            }
        }
    }

    pub fn overall_success_rate(&self) -> f32 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.successful_requests as f32 / self.total_requests as f32
        }
    }

    pub fn average_duration(&self) -> Duration {
        if self.successful_requests == 0 {
            Duration::ZERO
        } else {
            self.total_duration / self.successful_requests as u32
        }
    }
}

/// Performance metrics for a specific layer
#[derive(Debug, Clone)]
struct LayerPrefetchMetrics {
    requests: usize,
    successes: usize,
    failures: usize,
    total_duration: Duration,
    last_start: Option<Instant>,
    last_duration: Option<Duration>,
}

impl LayerPrefetchMetrics {
    fn new() -> Self {
        Self {
            requests: 0,
            successes: 0,
            failures: 0,
            total_duration: Duration::ZERO,
            last_start: None,
            last_duration: None,
        }
    }

    fn record_start(&mut self, start_time: Instant) {
        self.requests += 1;
        self.last_start = Some(start_time);
    }

    fn record_completion(&mut self, end_time: Instant, success: bool) {
        if success {
            self.successes += 1;
        } else {
            self.failures += 1;
        }

        if let Some(start_time) = self.last_start {
            let duration = end_time.duration_since(start_time);
            self.total_duration += duration;
            self.last_duration = Some(duration);
        }
    }
}

/// Result of a batch prefetch operation
#[derive(Debug, Clone)]
pub struct BatchPrefetchResult {
    pub total_layers: usize,
    pub successful: usize,
    pub failed: usize,
    pub failed_layers: Vec<String>,
    pub duration: Duration,
}

impl Default for BatchPrefetchResult {
    fn default() -> Self {
        Self {
            total_layers: 0,
            successful: 0,
            failed: 0,
            failed_layers: Vec::new(),
            duration: Duration::ZERO,
        }
    }
}

/// Result of intelligent prefetch operation
#[derive(Debug, Clone)]
pub struct IntelligentPrefetchResult {
    pub layers_scheduled: Vec<String>,
    pub prefetch_distance: usize,
    pub duration: Duration,
    pub batch_result: Option<BatchPrefetchResult>,
}

impl Default for IntelligentPrefetchResult {
    fn default() -> Self {
        Self {
            layers_scheduled: Vec::new(),
            prefetch_distance: 0,
            duration: Duration::ZERO,
            batch_result: None,
        }
    }
}

/// Statistics about prefetch scheduler performance
#[derive(Debug, Clone)]
pub struct PrefetchSchedulerStats {
    pub queue_length: usize,
    pub active_tasks: usize,
    pub performance_metrics: PrefetchPerformanceMetrics,
    pub tracked_layers: usize,
    pub async_prefetch_enabled: bool,
    pub prefetch_buffer_size: usize,
    pub overlap_computation: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{init_process_group, BackendType};

    #[tokio::test]
    async fn test_prefetch_scheduler_creation() {
        let config = Zero3CpuOffloadConfig::default();
        let process_group = Arc::new(init_process_group(BackendType::Gloo, 0, 1).await.unwrap());

        let scheduler = PrefetchScheduler::new(&config, process_group);
        let stats = scheduler.get_statistics();

        assert_eq!(stats.queue_length, 0);
        assert_eq!(stats.active_tasks, 0);
        assert!(stats.async_prefetch_enabled);
    }

    #[tokio::test]
    async fn test_schedule_prefetch() {
        let config = Zero3CpuOffloadConfig::default();
        let process_group = Arc::new(init_process_group(BackendType::Gloo, 0, 1).await.unwrap());

        let scheduler = PrefetchScheduler::new(&config, process_group);

        scheduler.schedule_prefetch("layer1", PrefetchPriority::High).await.unwrap();

        // Wait a bit for async processing
        tokio::time::sleep(Duration::from_millis(50)).await;

        let stats = scheduler.get_statistics();
        assert!(stats.performance_metrics.total_requests > 0);
    }

    #[tokio::test]
    async fn test_batch_prefetch() {
        let config = Zero3CpuOffloadConfig::default();
        let process_group = Arc::new(init_process_group(BackendType::Gloo, 0, 1).await.unwrap());

        let scheduler = PrefetchScheduler::new(&config, process_group);

        let layers = vec!["layer1".to_string(), "layer2".to_string(), "layer3".to_string()];
        let result = scheduler.batch_prefetch(layers.clone(), PrefetchPriority::Medium).await.unwrap();

        assert_eq!(result.total_layers, 3);
        assert_eq!(result.successful + result.failed, 3);
    }

    #[tokio::test]
    async fn test_intelligent_prefetch() {
        let config = Zero3CpuOffloadConfig::default();
        let process_group = Arc::new(init_process_group(BackendType::Gloo, 0, 1).await.unwrap());

        let scheduler = PrefetchScheduler::new(&config, process_group);

        let execution_graph = vec![
            "layer1".to_string(),
            "layer2".to_string(),
            "layer3".to_string(),
            "layer4".to_string(),
            "layer5".to_string(),
        ];

        let result = scheduler.intelligent_prefetch("layer1", &execution_graph).await.unwrap();

        assert!(result.prefetch_distance > 0);
        assert!(!result.layers_scheduled.is_empty() || result.prefetch_distance == 0);
    }

    #[tokio::test]
    async fn test_cancel_prefetch() {
        let mut config = Zero3CpuOffloadConfig::default();
        config.async_prefetch = true;
        let process_group = Arc::new(init_process_group(BackendType::Gloo, 0, 1).await.unwrap());

        let scheduler = PrefetchScheduler::new(&config, process_group);

        // Schedule a prefetch
        scheduler.schedule_prefetch("layer1", PrefetchPriority::Low).await.unwrap();

        // Try to cancel it
        let cancelled = scheduler.cancel_prefetch("layer1").await.unwrap();
        assert!(cancelled);

        // Try to cancel non-existent prefetch
        let not_cancelled = scheduler.cancel_prefetch("nonexistent").await.unwrap();
        assert!(!not_cancelled);
    }

    #[test]
    fn test_execution_history() {
        let mut history = ExecutionHistory::new();
        let now = Instant::now();

        history.record_execution(now);
        assert_eq!(history.total_accesses, 1);
        assert!(history.last_access.is_some());

        // Test frequency calculation
        let freq = history.access_frequency();
        assert!(freq >= 0.0);
    }

    #[test]
    fn test_prefetch_priority_ordering() {
        assert!(PrefetchPriority::Critical > PrefetchPriority::High);
        assert!(PrefetchPriority::High > PrefetchPriority::Medium);
        assert!(PrefetchPriority::Medium > PrefetchPriority::Low);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PrefetchPerformanceMetrics::new();
        let start_time = Instant::now();

        metrics.prefetch_started("layer1", start_time);
        assert_eq!(metrics.total_requests, 1);

        let end_time = start_time + Duration::from_millis(10);
        metrics.prefetch_completed("layer1", end_time, true);

        assert_eq!(metrics.successful_requests, 1);
        assert_eq!(metrics.overall_success_rate(), 1.0);
        assert!(metrics.average_duration() > Duration::ZERO);
    }
}