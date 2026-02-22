//! Multi-stream execution orchestrator for CUDA backend
//!
//! This module provides the highest-level interface for intelligent multi-stream execution:
//! - Unified scheduling across multiple streams
//! - Graph capture and replay optimization
//! - Dynamic resource management
//! - Performance monitoring and adaptation
//! - Integration with memory management and tensor operations

use crate::cuda::error::CudaResult;
use crate::cuda::{
    graph_execution::{GraphExecutionManager, GraphPerformanceSummary},
    intelligent_scheduler::{
        IntelligentStreamScheduler, MultiOperationCoordinator, SchedulerMetrics,
        SchedulingStrategy, WorkloadCharacteristics,
    },
    stream_advanced::{ProfilingReport, StreamProfiler, WorkloadType},
    CudaStream,
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Multi-stream execution orchestrator
pub struct MultiStreamOrchestrator {
    /// Core scheduling and coordination
    scheduler: IntelligentStreamScheduler,
    coordinator: MultiOperationCoordinator,

    /// Graph execution management
    graph_manager: GraphExecutionManager,

    /// Performance monitoring
    profiler: StreamProfiler,

    /// Adaptive configuration
    config: OrchestratorConfig,

    /// Execution history for learning
    execution_history: Arc<RwLock<ExecutionHistory>>,

    /// Active workload tracking
    active_workloads: Arc<Mutex<HashMap<String, ActiveWorkload>>>,

    /// Performance metrics
    metrics: Arc<Mutex<OrchestratorMetrics>>,
}

/// Configuration for the orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub max_concurrent_operations: usize,
    pub graph_capture_threshold: Duration, // Auto-capture graphs for operations taking longer than this
    pub adaptation_interval: Duration,     // How often to adapt scheduling strategy
    pub memory_pressure_threshold: f32,    // 0.0 - 1.0
    pub enable_auto_optimization: bool,
    pub performance_history_size: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_operations: 32,
            graph_capture_threshold: Duration::from_millis(10),
            adaptation_interval: Duration::from_secs(30),
            memory_pressure_threshold: 0.85,
            enable_auto_optimization: true,
            performance_history_size: 1000,
        }
    }
}

/// Active workload information
#[derive(Debug, Clone)]
struct ActiveWorkload {
    workload_id: String,
    characteristics: WorkloadCharacteristics,
    start_time: Instant,
    assigned_streams: Vec<Arc<CudaStream>>,
    dependencies: Vec<String>,
    graph_captured: bool,
}

/// Execution history for learning and adaptation
#[derive(Debug)]
struct ExecutionHistory {
    operation_history: VecDeque<OperationRecord>,
    performance_trends: HashMap<String, PerformanceTrend>,
    optimal_configurations: HashMap<WorkloadType, OptimalConfig>,
}

#[derive(Debug, Clone)]
struct OperationRecord {
    operation_id: String,
    workload_type: WorkloadType,
    execution_time: Duration,
    memory_usage: usize,
    stream_count: usize,
    scheduling_strategy: SchedulingStrategy,
    success: bool,
}

#[derive(Debug, Clone)]
struct PerformanceTrend {
    recent_times: VecDeque<Duration>,
    trend_direction: f32, // -1.0 (degrading) to 1.0 (improving)
    confidence: f32,      // 0.0 to 1.0
}

#[derive(Debug, Clone)]
struct OptimalConfig {
    preferred_strategy: SchedulingStrategy,
    optimal_stream_count: usize,
    use_graph_capture: bool,
    confidence: f32,
}

/// Orchestrator performance metrics
#[derive(Debug, Clone)]
pub struct OrchestratorMetrics {
    pub total_operations_executed: u64,
    pub successful_operations: u64,
    pub total_execution_time: Duration,
    pub average_execution_time: Duration,
    pub graph_capture_count: u64,
    pub graph_replay_count: u64,
    pub adaptive_optimizations: u64,
    pub current_active_operations: usize,
    pub peak_concurrent_operations: usize,
    pub scheduler_metrics: SchedulerMetrics,
}

impl Default for OrchestratorMetrics {
    fn default() -> Self {
        Self {
            total_operations_executed: 0,
            successful_operations: 0,
            total_execution_time: Duration::from_secs(0),
            average_execution_time: Duration::from_secs(0),
            graph_capture_count: 0,
            graph_replay_count: 0,
            adaptive_optimizations: 0,
            current_active_operations: 0,
            peak_concurrent_operations: 0,
            scheduler_metrics: SchedulerMetrics {
                total_operations_scheduled: 0,
                active_operations: 0,
                prediction_accuracy: 0.0,
                average_execution_time: Duration::from_secs(0),
                current_strategy: SchedulingStrategy::Balanced,
                stream_utilization: 0.0,
            },
        }
    }
}

impl MultiStreamOrchestrator {
    /// Create new multi-stream orchestrator
    pub fn new(config: OrchestratorConfig) -> CudaResult<Self> {
        let scheduler = IntelligentStreamScheduler::new(8, SchedulingStrategy::Balanced)?;
        let coordinator = MultiOperationCoordinator::new(SchedulingStrategy::Balanced)?;
        let graph_manager = GraphExecutionManager::new();
        let mut profiler = StreamProfiler::new();
        profiler.enable();

        let execution_history = Arc::new(RwLock::new(ExecutionHistory {
            operation_history: VecDeque::new(),
            performance_trends: HashMap::new(),
            optimal_configurations: HashMap::new(),
        }));

        Ok(Self {
            scheduler,
            coordinator,
            graph_manager,
            profiler,
            config,
            execution_history,
            active_workloads: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(OrchestratorMetrics::default())),
        })
    }

    /// Execute a single operation with optimal scheduling
    pub fn execute_operation<F>(
        &mut self,
        operation_id: String,
        characteristics: WorkloadCharacteristics,
        executor: F,
    ) -> CudaResult<ExecutionResult>
    where
        F: FnOnce(&CudaStream) -> CudaResult<()> + Send + 'static,
    {
        let start_time = Instant::now();

        // Check if we should use graph capture/replay
        let should_use_graph = self.should_use_graph_execution(&operation_id, &characteristics);

        if should_use_graph && self.has_cached_graph(&operation_id) {
            // Use cached graph
            return self.execute_cached_graph(&operation_id);
        }

        // Schedule operation
        let decision = self
            .scheduler
            .schedule_operation(operation_id.clone(), characteristics.clone())?;

        // Register active workload
        self.register_active_workload(&operation_id, &characteristics, &decision.stream)?;

        let execution_result = if should_use_graph && !self.has_cached_graph(&operation_id) {
            // Capture new graph
            self.execute_with_graph_capture(
                operation_id.clone(),
                decision.stream.clone(),
                executor,
            )?
        } else {
            // Direct execution
            self.execute_direct(decision.stream.clone(), executor)?
        };

        // Complete operation and update metrics
        let execution_time = start_time.elapsed();
        self.complete_operation(
            &operation_id,
            execution_time,
            execution_result.memory_bandwidth,
        )?;

        Ok(ExecutionResult {
            execution_time,
            memory_bandwidth: execution_result.memory_bandwidth,
            stream_id: decision.stream.id(),
            used_graph_capture: should_use_graph,
            success: true,
        })
    }

    /// Execute multiple operations with dependency management
    pub fn execute_batch<F>(
        &mut self,
        operations: Vec<(String, WorkloadCharacteristics)>,
        executor_factory: F,
    ) -> CudaResult<Vec<ExecutionResult>>
    where
        F: Fn(&str) -> Box<dyn FnOnce(&CudaStream) -> CudaResult<()> + Send>
            + Send
            + Sync
            + Clone
            + 'static,
    {
        let start_time = Instant::now();
        let mut results = Vec::new();

        // Use coordinator for batch execution with dependencies
        let _batch_results = self.coordinator.execute_parallel_workflow(
            operations.clone(),
            move |op_id, stream| {
                let executor = executor_factory(op_id);
                let exec_start = Instant::now();
                executor(stream)?;
                let exec_time = exec_start.elapsed();
                Ok((exec_time, 1_000_000_000)) // Placeholder bandwidth
            },
        )?;

        // Process results
        for (_op_id, _characteristics) in operations {
            results.push(ExecutionResult {
                execution_time: Duration::from_millis(10), // Placeholder
                memory_bandwidth: 1_000_000_000,
                stream_id: 0,
                used_graph_capture: false,
                success: true,
            });
        }

        // Update batch execution metrics
        let _total_time = start_time.elapsed();
        let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
        metrics.total_operations_executed += results.len() as u64;
        metrics.successful_operations += results.iter().filter(|r| r.success).count() as u64;

        Ok(results)
    }

    /// Execute a repeating workload with automatic optimization
    pub fn execute_repeating_workload<F>(
        &mut self,
        workload_id: String,
        characteristics: WorkloadCharacteristics,
        iterations: usize,
        executor: F,
    ) -> CudaResult<RepeatingWorkloadResult>
    where
        F: Fn(&CudaStream) -> CudaResult<()> + Send + Sync + Clone + 'static,
    {
        let mut execution_times = Vec::new();
        let mut total_bandwidth = 0u64;
        let start_time = Instant::now();

        // Capture graph on first iteration if beneficial
        let use_graph = self.should_use_graph_execution(&workload_id, &characteristics);
        let mut graph_captured = false;

        for iteration in 0..iterations {
            let iter_start = Instant::now();

            if use_graph && iteration == 0 {
                // Capture graph on first iteration
                let decision = self.scheduler.schedule_operation(
                    format!("{}_{}", workload_id, iteration),
                    characteristics.clone(),
                )?;

                self.graph_manager
                    .begin_capture(format!("{}_graph", workload_id), decision.stream.clone())?;

                let exec_result = executor(&decision.stream);

                self.graph_manager
                    .end_capture(format!("{}_graph", workload_id))?;

                if exec_result.is_ok() {
                    graph_captured = true;
                }
            } else if use_graph && graph_captured {
                // Use captured graph
                let result = self.execute_cached_graph(&format!("{}_graph", workload_id))?;
                execution_times.push(result.execution_time);
                total_bandwidth += result.memory_bandwidth;
                continue;
            } else {
                // Direct execution
                let decision = self.scheduler.schedule_operation(
                    format!("{}_{}", workload_id, iteration),
                    characteristics.clone(),
                )?;

                executor(&decision.stream)?;
            }

            let iter_time = iter_start.elapsed();
            execution_times.push(iter_time);
            total_bandwidth += 1_000_000_000; // Placeholder
        }

        let total_time = start_time.elapsed();
        let average_time = total_time / iterations as u32;

        // Update performance history
        self.update_repeating_workload_history(&workload_id, &execution_times);

        Ok(RepeatingWorkloadResult {
            total_iterations: iterations,
            total_execution_time: total_time,
            average_execution_time: average_time,
            execution_times,
            total_memory_bandwidth: total_bandwidth,
            graph_capture_used: graph_captured,
            performance_improvement: self.calculate_performance_improvement(&workload_id),
        })
    }

    /// Synchronize all active operations
    pub fn synchronize_all(&mut self) -> CudaResult<()> {
        // Create barrier across all active operations
        self.coordinator.get_metrics(); // This will trigger internal synchronization

        // Clear active workloads
        self.active_workloads
            .lock()
            .expect("lock should not be poisoned")
            .clear();

        Ok(())
    }

    /// Get comprehensive performance metrics
    pub fn get_metrics(&self) -> OrchestratorMetrics {
        let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
        metrics.scheduler_metrics = self.coordinator.get_metrics();
        metrics.clone()
    }

    /// Optimize orchestrator configuration based on performance history
    pub fn optimize_configuration(&mut self) -> CudaResult<OptimizationResult> {
        if !self.config.enable_auto_optimization {
            return Ok(OptimizationResult {
                optimizations_applied: 0,
                performance_improvement: 0.0,
                new_strategy: None,
            });
        }

        let mut optimizations_applied = 0;
        #[allow(unused_assignments)]
        let mut performance_improvement = 0.0;
        let mut new_strategy = None;

        // Analyze execution history
        let history = self
            .execution_history
            .read()
            .expect("lock should not be poisoned");

        // Optimize scheduling strategy based on workload patterns
        if let Some(optimal_strategy) = self.analyze_optimal_strategy(&history) {
            if optimal_strategy != self.coordinator.get_metrics().current_strategy {
                new_strategy = Some(optimal_strategy);
                optimizations_applied += 1;
            }
        }

        // Optimize graph capture thresholds
        let new_threshold = self.analyze_optimal_graph_threshold(&history);
        if new_threshold != self.config.graph_capture_threshold {
            self.config.graph_capture_threshold = new_threshold;
            optimizations_applied += 1;
        }

        // Calculate performance improvement
        performance_improvement = self.calculate_overall_performance_improvement(&history);

        // Update metrics
        let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
        metrics.adaptive_optimizations += optimizations_applied as u64;

        Ok(OptimizationResult {
            optimizations_applied,
            performance_improvement,
            new_strategy,
        })
    }

    /// Get profiling report for all streams
    pub fn get_profiling_report(&self) -> ProfilingReport {
        self.profiler.get_comprehensive_report()
    }

    /// Get graph performance summary
    pub fn get_graph_performance(&self) -> HashMap<String, GraphPerformanceSummary> {
        self.graph_manager.get_performance_summary()
    }

    // Private helper methods

    fn should_use_graph_execution(
        &self,
        operation_id: &str,
        characteristics: &WorkloadCharacteristics,
    ) -> bool {
        // Use graph capture for operations expected to take longer than threshold
        characteristics.estimated_compute_time > self.config.graph_capture_threshold
            || self.has_repeated_execution_pattern(operation_id)
    }

    fn has_cached_graph(&self, operation_id: &str) -> bool {
        self.graph_manager
            .list_graphs()
            .contains(&operation_id.to_string())
    }

    fn has_repeated_execution_pattern(&self, operation_id: &str) -> bool {
        let history = self
            .execution_history
            .read()
            .expect("lock should not be poisoned");
        let count = history
            .operation_history
            .iter()
            .filter(|record| record.operation_id.starts_with(operation_id))
            .count();
        count >= 3 // Consider it repeated if executed 3+ times
    }

    fn execute_cached_graph(&mut self, graph_id: &str) -> CudaResult<ExecutionResult> {
        // Create a dummy stream for graph execution
        let stream = Arc::new(CudaStream::new()?);
        let execution_time = self.graph_manager.execute_graph(graph_id, &stream)?;

        let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
        metrics.graph_replay_count += 1;

        Ok(ExecutionResult {
            execution_time,
            memory_bandwidth: 1_000_000_000, // Placeholder
            stream_id: stream.id(),
            used_graph_capture: true,
            success: true,
        })
    }

    fn execute_with_graph_capture<F>(
        &mut self,
        operation_id: String,
        stream: Arc<CudaStream>,
        executor: F,
    ) -> CudaResult<ExecutionResult>
    where
        F: FnOnce(&CudaStream) -> CudaResult<()> + Send + 'static,
    {
        let start_time = Instant::now();

        // Begin graph capture
        self.graph_manager
            .begin_capture(operation_id.clone(), stream.clone())?;

        // Execute operation
        executor(&stream)?;

        // End capture
        self.graph_manager.end_capture(operation_id)?;

        let execution_time = start_time.elapsed();

        let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
        metrics.graph_capture_count += 1;

        Ok(ExecutionResult {
            execution_time,
            memory_bandwidth: 1_000_000_000, // Placeholder
            stream_id: stream.id(),
            used_graph_capture: true,
            success: true,
        })
    }

    fn execute_direct<F>(
        &mut self,
        stream: Arc<CudaStream>,
        executor: F,
    ) -> CudaResult<ExecutionResult>
    where
        F: FnOnce(&CudaStream) -> CudaResult<()> + Send + 'static,
    {
        let start_time = Instant::now();

        executor(&stream)?;

        let execution_time = start_time.elapsed();

        Ok(ExecutionResult {
            execution_time,
            memory_bandwidth: 1_000_000_000, // Placeholder
            stream_id: stream.id(),
            used_graph_capture: false,
            success: true,
        })
    }

    fn register_active_workload(
        &mut self,
        operation_id: &str,
        characteristics: &WorkloadCharacteristics,
        stream: &Arc<CudaStream>,
    ) -> CudaResult<()> {
        let workload = ActiveWorkload {
            workload_id: operation_id.to_string(),
            characteristics: characteristics.clone(),
            start_time: Instant::now(),
            assigned_streams: vec![stream.clone()],
            dependencies: characteristics
                .synchronization_requirements
                .dependencies
                .clone(),
            graph_captured: false,
        };

        let mut active = self
            .active_workloads
            .lock()
            .expect("lock should not be poisoned");
        active.insert(operation_id.to_string(), workload);

        // Update peak concurrent operations
        let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
        metrics.current_active_operations = active.len();
        if active.len() > metrics.peak_concurrent_operations {
            metrics.peak_concurrent_operations = active.len();
        }

        Ok(())
    }

    fn complete_operation(
        &mut self,
        operation_id: &str,
        execution_time: Duration,
        memory_bandwidth: u64,
    ) -> CudaResult<()> {
        // Remove from active workloads
        let workload = self
            .active_workloads
            .lock()
            .expect("lock should not be poisoned")
            .remove(operation_id);

        if let Some(workload) = workload {
            // Update execution history
            let mut history = self
                .execution_history
                .write()
                .expect("lock should not be poisoned");
            let record = OperationRecord {
                operation_id: operation_id.to_string(),
                workload_type: workload.characteristics.workload_type,
                execution_time,
                memory_usage: 0, // Placeholder
                stream_count: workload.assigned_streams.len(),
                scheduling_strategy: SchedulingStrategy::Balanced, // Placeholder
                success: true,
            };

            history.operation_history.push_back(record);

            // Keep bounded history
            if history.operation_history.len() > self.config.performance_history_size {
                history.operation_history.pop_front();
            }

            // Complete scheduler operation
            self.scheduler.complete_operation(
                operation_id.to_string(),
                execution_time,
                memory_bandwidth,
            )?;
        }

        // Update metrics
        let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
        metrics.total_operations_executed += 1;
        metrics.successful_operations += 1;
        metrics.total_execution_time += execution_time;
        metrics.average_execution_time =
            metrics.total_execution_time / metrics.total_operations_executed as u32;
        metrics.current_active_operations = self
            .active_workloads
            .lock()
            .expect("lock should not be poisoned")
            .len();

        Ok(())
    }

    fn update_repeating_workload_history(
        &mut self,
        workload_id: &str,
        execution_times: &[Duration],
    ) {
        let mut history = self
            .execution_history
            .write()
            .expect("lock should not be poisoned");

        // Update performance trend
        let trend = history
            .performance_trends
            .entry(workload_id.to_string())
            .or_insert_with(|| PerformanceTrend {
                recent_times: VecDeque::new(),
                trend_direction: 0.0,
                confidence: 0.0,
            });

        for &time in execution_times {
            trend.recent_times.push_back(time);
            if trend.recent_times.len() > 20 {
                trend.recent_times.pop_front();
            }
        }

        // Calculate trend direction
        if trend.recent_times.len() >= 4 {
            let mid = trend.recent_times.len() / 2;
            let first_half: Duration = trend.recent_times.iter().take(mid).sum();
            let second_half: Duration = trend.recent_times.iter().skip(mid).sum();

            let first_avg = first_half.as_secs_f32() / mid as f32;
            let second_avg = second_half.as_secs_f32() / (trend.recent_times.len() - mid) as f32;

            trend.trend_direction = (first_avg - second_avg) / first_avg.max(second_avg);
            trend.confidence = (trend.recent_times.len() as f32 / 20.0).min(1.0);
        }
    }

    fn calculate_performance_improvement(&self, workload_id: &str) -> f32 {
        let history = self
            .execution_history
            .read()
            .expect("lock should not be poisoned");
        if let Some(trend) = history.performance_trends.get(workload_id) {
            trend.trend_direction * trend.confidence
        } else {
            0.0
        }
    }

    fn analyze_optimal_strategy(&self, history: &ExecutionHistory) -> Option<SchedulingStrategy> {
        // Simple analysis - in practice this would be more sophisticated
        let strategy_performance: HashMap<SchedulingStrategy, f32> = history
            .operation_history
            .iter()
            .fold(HashMap::new(), |mut acc, record| {
                let score = 1.0 / record.execution_time.as_secs_f32();
                *acc.entry(record.scheduling_strategy).or_insert(0.0) += score;
                acc
            });

        strategy_performance
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(strategy, _)| strategy)
    }

    fn analyze_optimal_graph_threshold(&self, history: &ExecutionHistory) -> Duration {
        // Analyze which operations benefited from graph capture
        let graph_beneficial_times: Vec<Duration> = history
            .operation_history
            .iter()
            .filter(|record| record.success)
            .map(|record| record.execution_time)
            .collect();

        if let Some(&median_time) = graph_beneficial_times.get(graph_beneficial_times.len() / 2) {
            median_time
        } else {
            self.config.graph_capture_threshold
        }
    }

    fn calculate_overall_performance_improvement(&self, history: &ExecutionHistory) -> f32 {
        if history.operation_history.len() < 10 {
            return 0.0;
        }

        let recent_count = history.operation_history.len().min(100);
        let start_idx = history.operation_history.len() - recent_count;

        let recent_avg: Duration = history
            .operation_history
            .iter()
            .skip(start_idx)
            .map(|op| op.execution_time)
            .sum::<Duration>()
            / recent_count as u32;

        let historical_count = start_idx.max(1);
        let historical_avg: Duration = history
            .operation_history
            .iter()
            .take(historical_count)
            .map(|op| op.execution_time)
            .sum::<Duration>()
            / historical_count as u32;

        (historical_avg.as_secs_f32() - recent_avg.as_secs_f32()) / historical_avg.as_secs_f32()
    }
}

/// Result of executing an operation
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub execution_time: Duration,
    pub memory_bandwidth: u64,
    pub stream_id: u64,
    pub used_graph_capture: bool,
    pub success: bool,
}

/// Result of executing a repeating workload
#[derive(Debug, Clone)]
pub struct RepeatingWorkloadResult {
    pub total_iterations: usize,
    pub total_execution_time: Duration,
    pub average_execution_time: Duration,
    pub execution_times: Vec<Duration>,
    pub total_memory_bandwidth: u64,
    pub graph_capture_used: bool,
    pub performance_improvement: f32, // Percentage improvement over baseline
}

/// Result of configuration optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub optimizations_applied: usize,
    pub performance_improvement: f32,
    pub new_strategy: Option<SchedulingStrategy>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Requires CUDA hardware - run with --ignored flag"]
    fn test_orchestrator_creation() {
        if crate::cuda::is_available() {
            let _device = Arc::new(crate::cuda::device::CudaDevice::new(0).unwrap());
            let config = OrchestratorConfig::default();
            let orchestrator = MultiStreamOrchestrator::new(config);
            assert!(orchestrator.is_ok());
        }
    }

    #[test]
    fn test_orchestrator_config() {
        let config = OrchestratorConfig::default();
        assert_eq!(config.max_concurrent_operations, 32);
        assert!(config.enable_auto_optimization);
    }

    #[test]
    fn test_execution_result() {
        let result = ExecutionResult {
            execution_time: Duration::from_millis(100),
            memory_bandwidth: 1_000_000_000,
            stream_id: 1,
            used_graph_capture: true,
            success: true,
        };

        assert_eq!(result.execution_time, Duration::from_millis(100));
        assert!(result.used_graph_capture);
        assert!(result.success);
    }
}
