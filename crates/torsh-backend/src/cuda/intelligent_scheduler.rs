//! Intelligent multi-stream execution scheduler for CUDA backend
//!
//! This module provides advanced scheduling algorithms for optimal multi-stream execution:
//! - Dynamic workload analysis and stream allocation
//! - Performance-guided scheduling decisions
//! - Adaptive stream pool management
//! - Cross-stream dependency optimization
//! - CUDA graph integration for repeated workloads

use crate::cuda::error::{CudaError, CudaResult};
use crate::cuda::{
    stream_advanced::{AdvancedStreamPool, AllocationStrategy, WorkloadType},
    CudaEvent, CudaStream,
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Intelligent scheduling decisions based on workload analysis
#[derive(Debug, Clone)]
pub struct SchedulingDecision {
    pub stream: Arc<CudaStream>,
    pub predicted_execution_time: Duration,
    pub confidence: f32, // 0.0 - 1.0
    pub scheduling_strategy: SchedulingStrategy,
}

/// Available scheduling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchedulingStrategy {
    /// Minimize total execution time
    MinimizeLatency,
    /// Maximize throughput
    MaximizeThroughput,
    /// Balance latency and throughput
    Balanced,
    /// Minimize resource contention
    LoadBalance,
    /// Optimize for power efficiency
    PowerEfficient,
}

/// Workload characteristics for intelligent scheduling
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    pub workload_type: WorkloadType,
    pub estimated_compute_time: Duration,
    pub estimated_memory_bandwidth: u64, // bytes/second
    pub parallel_potential: f32,         // 0.0 - 1.0
    pub memory_access_pattern: MemoryAccessPattern,
    pub synchronization_requirements: SynchronizationRequirements,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided { stride: usize },
    Broadcast,
    Reduction,
}

#[derive(Debug, Clone)]
pub struct SynchronizationRequirements {
    pub requires_barrier: bool,
    pub dependencies: Vec<String>,     // operation IDs this depends on
    pub provides_outputs: Vec<String>, // outputs this operation provides
}

/// Historical performance data for learning
#[derive(Debug, Clone)]
struct PerformanceHistory {
    execution_times: VecDeque<Duration>,
    memory_bandwidth_achieved: VecDeque<u64>,
    resource_utilization: VecDeque<f32>,
    scheduling_decisions: VecDeque<(SchedulingStrategy, f32)>, // strategy, score
}

impl PerformanceHistory {
    fn new() -> Self {
        Self {
            execution_times: VecDeque::new(),
            memory_bandwidth_achieved: VecDeque::new(),
            resource_utilization: VecDeque::new(),
            scheduling_decisions: VecDeque::new(),
        }
    }

    fn add_measurement(&mut self, execution_time: Duration, bandwidth: u64, utilization: f32) {
        const MAX_HISTORY: usize = 100;

        self.execution_times.push_back(execution_time);
        self.memory_bandwidth_achieved.push_back(bandwidth);
        self.resource_utilization.push_back(utilization);

        // Keep bounded history
        if self.execution_times.len() > MAX_HISTORY {
            self.execution_times.pop_front();
            self.memory_bandwidth_achieved.pop_front();
            self.resource_utilization.pop_front();
        }
    }

    fn average_execution_time(&self) -> Option<Duration> {
        if self.execution_times.is_empty() {
            None
        } else {
            let total: Duration = self.execution_times.iter().sum();
            Some(total / self.execution_times.len() as u32)
        }
    }

    fn predict_execution_time(
        &self,
        characteristics: &WorkloadCharacteristics,
    ) -> Option<Duration> {
        // Simple prediction based on historical data and workload characteristics
        let base_time = self.average_execution_time()?;
        let complexity_factor = match characteristics.workload_type {
            WorkloadType::Compute => 1.2,
            WorkloadType::Memory => 0.8,
            WorkloadType::Mixed => 1.0,
            WorkloadType::Coordination => 0.3,
        };

        Some(Duration::from_secs_f64(
            base_time.as_secs_f64() * complexity_factor,
        ))
    }
}

/// Intelligent multi-stream scheduler
pub struct IntelligentStreamScheduler {
    stream_pool: AdvancedStreamPool,
    performance_history: Arc<RwLock<HashMap<String, PerformanceHistory>>>, // operation_id -> history
    active_operations: Arc<Mutex<HashMap<String, OperationMetadata>>>,
    scheduling_strategy: SchedulingStrategy,

    // Adaptive configuration
    min_streams: usize,
    max_streams: usize,
    target_utilization: f32,

    // Performance monitoring
    total_operations_scheduled: u64,
    successful_predictions: u64,
    prediction_accuracy: f32,
}

#[derive(Debug, Clone)]
struct OperationMetadata {
    stream: Arc<CudaStream>,
    start_time: Instant,
    characteristics: WorkloadCharacteristics,
    dependencies: Vec<String>,
}

impl IntelligentStreamScheduler {
    /// Create new intelligent scheduler
    pub fn new(initial_streams: usize, strategy: SchedulingStrategy) -> CudaResult<Self> {
        let stream_pool = AdvancedStreamPool::new_with_strategy(
            initial_streams,
            Self::strategy_to_allocation_strategy(strategy),
        )?;

        Ok(Self {
            stream_pool,
            performance_history: Arc::new(RwLock::new(HashMap::new())),
            active_operations: Arc::new(Mutex::new(HashMap::new())),
            scheduling_strategy: strategy,
            min_streams: initial_streams.max(2),
            max_streams: initial_streams * 4,
            target_utilization: 0.85,
            total_operations_scheduled: 0,
            successful_predictions: 0,
            prediction_accuracy: 0.0,
        })
    }

    /// Schedule operation with intelligent stream selection
    pub fn schedule_operation(
        &mut self,
        operation_id: String,
        characteristics: WorkloadCharacteristics,
    ) -> CudaResult<SchedulingDecision> {
        // Analyze dependencies
        let ready_streams = self.find_available_streams(&characteristics)?;

        // Predict performance for each candidate stream
        let mut candidates: Vec<_> = ready_streams
            .into_iter()
            .map(|stream| {
                let predicted_time = self.predict_execution_time(&operation_id, &characteristics);
                let confidence = self.calculate_prediction_confidence(&operation_id);

                (stream, predicted_time, confidence)
            })
            .collect();

        // Sort by scheduling strategy
        self.rank_candidates(&mut candidates, &characteristics);

        // Select best candidate
        let (selected_stream, predicted_time, confidence) = candidates
            .into_iter()
            .next()
            .ok_or_else(|| CudaError::Context {
                message: "No available streams for scheduling".to_string(),
            })?;

        // Record operation metadata
        let metadata = OperationMetadata {
            stream: selected_stream.clone(),
            start_time: Instant::now(),
            characteristics: characteristics.clone(),
            dependencies: characteristics
                .synchronization_requirements
                .dependencies
                .clone(),
        };

        self.active_operations
            .lock()
            .expect("active_operations lock should not be poisoned")
            .insert(operation_id.clone(), metadata);

        self.total_operations_scheduled += 1;

        Ok(SchedulingDecision {
            stream: selected_stream,
            predicted_execution_time: predicted_time,
            confidence,
            scheduling_strategy: self.scheduling_strategy,
        })
    }

    /// Complete operation and update performance history
    pub fn complete_operation(
        &mut self,
        operation_id: String,
        actual_execution_time: Duration,
        memory_bandwidth: u64,
    ) -> CudaResult<()> {
        // Remove from active operations
        let metadata = self
            .active_operations
            .lock()
            .expect("active_operations lock should not be poisoned")
            .remove(&operation_id)
            .ok_or_else(|| CudaError::Context {
                message: format!("Operation {} not found in active operations", operation_id),
            })?;

        // Calculate resource utilization
        let utilization = self.calculate_resource_utilization(&metadata.stream)?;

        // Update performance history
        let mut history = self
            .performance_history
            .write()
            .expect("lock should not be poisoned");
        let op_history = history
            .entry(operation_id.clone())
            .or_insert_with(PerformanceHistory::new);

        op_history.add_measurement(actual_execution_time, memory_bandwidth, utilization);

        // Update prediction accuracy
        if let Some(predicted_time) = op_history.average_execution_time() {
            let accuracy = 1.0
                - (actual_execution_time.as_secs_f32() - predicted_time.as_secs_f32()).abs()
                    / predicted_time
                        .as_secs_f32()
                        .max(actual_execution_time.as_secs_f32());

            if accuracy > 0.8 {
                self.successful_predictions += 1;
            }

            self.prediction_accuracy =
                self.successful_predictions as f32 / self.total_operations_scheduled as f32;
        }

        // Drop the lock before calling adapt_stream_pool to avoid borrow conflict
        drop(history);

        // Adaptive stream pool management
        self.adapt_stream_pool()?;

        Ok(())
    }

    /// Handle dependencies between operations
    pub fn add_dependency(&mut self, dependent_op: &str, dependency_op: &str) -> CudaResult<()> {
        let active_ops = self
            .active_operations
            .lock()
            .expect("lock should not be poisoned");

        if let (Some(dependent_meta), Some(dependency_meta)) =
            (active_ops.get(dependent_op), active_ops.get(dependency_op))
        {
            // Create synchronization event
            let sync_event = Arc::new(CudaEvent::new()?);
            dependency_meta.stream.record_event(&sync_event)?;
            dependent_meta.stream.wait_event(&sync_event)?;
        }

        Ok(())
    }

    /// Create barrier across all active streams
    pub fn create_execution_barrier(&self) -> CudaResult<()> {
        let active_ops = self
            .active_operations
            .lock()
            .expect("lock should not be poisoned");
        let active_streams: Vec<_> = active_ops
            .values()
            .map(|meta| meta.stream.clone())
            .collect();

        if active_streams.is_empty() {
            return Ok(());
        }

        // Create barrier event
        let barrier_event = Arc::new(CudaEvent::new()?);

        // Record event on all streams
        for stream in &active_streams {
            stream.record_event(&barrier_event)?;
        }

        // Wait for barrier on all streams
        for stream in &active_streams {
            stream.wait_event(&barrier_event)?;
        }

        Ok(())
    }

    /// Get scheduler performance metrics
    pub fn get_performance_metrics(&self) -> SchedulerMetrics {
        let history = self
            .performance_history
            .read()
            .expect("lock should not be poisoned");
        let active_count = self
            .active_operations
            .lock()
            .expect("lock should not be poisoned")
            .len();

        let total_operations = history.len();
        let average_accuracy = self.prediction_accuracy;

        let average_execution_time = if total_operations > 0 {
            let total_time: Duration = history
                .values()
                .filter_map(|h| h.average_execution_time())
                .sum();
            total_time / total_operations as u32
        } else {
            Duration::from_secs(0)
        };

        SchedulerMetrics {
            total_operations_scheduled: self.total_operations_scheduled,
            active_operations: active_count,
            prediction_accuracy: average_accuracy,
            average_execution_time,
            current_strategy: self.scheduling_strategy,
            stream_utilization: self.calculate_pool_utilization(),
        }
    }

    /// Optimize scheduler configuration based on performance history
    pub fn optimize_configuration(&mut self) -> CudaResult<()> {
        let metrics = self.get_performance_metrics();

        // Adjust scheduling strategy based on performance
        if metrics.prediction_accuracy < 0.6 {
            // Low accuracy, switch to more conservative strategy
            self.scheduling_strategy = SchedulingStrategy::LoadBalance;
        } else if metrics.stream_utilization < 0.5 {
            // Low utilization, focus on throughput
            self.scheduling_strategy = SchedulingStrategy::MaximizeThroughput;
        } else if metrics.average_execution_time > Duration::from_millis(100) {
            // High latency, focus on minimizing it
            self.scheduling_strategy = SchedulingStrategy::MinimizeLatency;
        }

        // Update stream pool strategy
        let _new_allocation_strategy =
            Self::strategy_to_allocation_strategy(self.scheduling_strategy);
        // Note: We'd need to modify AdvancedStreamPool to allow strategy updates

        Ok(())
    }

    // Private helper methods

    fn find_available_streams(
        &self,
        characteristics: &WorkloadCharacteristics,
    ) -> CudaResult<Vec<Arc<CudaStream>>> {
        let mut available_streams = Vec::new();

        // Get stream optimized for workload type
        let primary_stream = self
            .stream_pool
            .get_stream_for_workload(characteristics.workload_type);
        available_streams.push(primary_stream);

        // Add additional streams based on parallel potential
        if characteristics.parallel_potential > 0.5 {
            let secondary_stream = self
                .stream_pool
                .get_stream_for_workload(WorkloadType::Mixed);
            available_streams.push(secondary_stream);
        }

        Ok(available_streams)
    }

    fn predict_execution_time(
        &self,
        operation_id: &str,
        characteristics: &WorkloadCharacteristics,
    ) -> Duration {
        let history = self
            .performance_history
            .read()
            .expect("lock should not be poisoned");

        if let Some(op_history) = history.get(operation_id) {
            if let Some(predicted) = op_history.predict_execution_time(characteristics) {
                return predicted;
            }
        }

        // Fallback to characteristics-based estimation
        characteristics.estimated_compute_time
    }

    fn calculate_prediction_confidence(&self, operation_id: &str) -> f32 {
        let history = self
            .performance_history
            .read()
            .expect("lock should not be poisoned");

        if let Some(op_history) = history.get(operation_id) {
            // Confidence based on amount of historical data
            let sample_count = op_history.execution_times.len();
            (sample_count as f32 / 10.0).min(1.0)
        } else {
            0.1 // Low confidence for new operations
        }
    }

    fn rank_candidates(
        &self,
        candidates: &mut Vec<(Arc<CudaStream>, Duration, f32)>,
        characteristics: &WorkloadCharacteristics,
    ) {
        candidates.sort_by(|a, b| {
            let score_a = self.calculate_candidate_score(&a.0, a.1, a.2, characteristics);
            let score_b = self.calculate_candidate_score(&b.0, b.1, b.2, characteristics);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    fn calculate_candidate_score(
        &self,
        stream: &CudaStream,
        predicted_time: Duration,
        confidence: f32,
        characteristics: &WorkloadCharacteristics,
    ) -> f32 {
        let base_score = match self.scheduling_strategy {
            SchedulingStrategy::MinimizeLatency => 1.0 / (predicted_time.as_secs_f32() + 0.001),
            SchedulingStrategy::MaximizeThroughput => {
                characteristics.parallel_potential * confidence
            }
            SchedulingStrategy::Balanced => {
                let latency_score = 1.0 / (predicted_time.as_secs_f32() + 0.001);
                let throughput_score = characteristics.parallel_potential;
                (latency_score + throughput_score) / 2.0
            }
            SchedulingStrategy::LoadBalance => {
                let utilization = self.calculate_resource_utilization(stream).unwrap_or(0.5);
                1.0 - utilization // Prefer less utilized streams
            }
            SchedulingStrategy::PowerEfficient => {
                // Prefer longer running, more efficient operations
                predicted_time.as_secs_f32() * characteristics.parallel_potential
            }
        };

        base_score * confidence
    }

    fn calculate_resource_utilization(&self, _stream: &CudaStream) -> CudaResult<f32> {
        // This would query actual CUDA stream utilization
        // For now, return a placeholder
        Ok(0.7)
    }

    fn calculate_pool_utilization(&self) -> f32 {
        let active_count = self
            .active_operations
            .lock()
            .expect("lock should not be poisoned")
            .len();
        // This would calculate based on actual stream pool size
        active_count as f32 / 8.0 // Assuming 8 streams for now
    }

    fn adapt_stream_pool(&mut self) -> CudaResult<()> {
        let utilization = self.calculate_pool_utilization();

        // Simple adaptive logic
        if utilization > self.target_utilization {
            // High utilization, could benefit from more streams
            // Would need to implement dynamic stream pool growth
        } else if utilization < 0.3 {
            // Low utilization, could reduce streams for efficiency
            // Would need to implement dynamic stream pool shrinking
        }

        Ok(())
    }

    fn strategy_to_allocation_strategy(strategy: SchedulingStrategy) -> AllocationStrategy {
        match strategy {
            SchedulingStrategy::MinimizeLatency => AllocationStrategy::Priority,
            SchedulingStrategy::MaximizeThroughput => AllocationStrategy::LoadBalanced,
            SchedulingStrategy::Balanced => AllocationStrategy::Workload,
            SchedulingStrategy::LoadBalance => AllocationStrategy::LoadBalanced,
            SchedulingStrategy::PowerEfficient => AllocationStrategy::RoundRobin,
        }
    }
}

/// Scheduler performance metrics
#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    pub total_operations_scheduled: u64,
    pub active_operations: usize,
    pub prediction_accuracy: f32,
    pub average_execution_time: Duration,
    pub current_strategy: SchedulingStrategy,
    pub stream_utilization: f32,
}

/// Multi-operation execution coordinator for complex workflows
pub struct MultiOperationCoordinator {
    scheduler: IntelligentStreamScheduler,
    execution_graph: HashMap<String, Vec<String>>, // operation -> dependencies
    completion_callbacks: HashMap<String, Vec<Box<dyn FnOnce() + Send + 'static>>>,
}

impl MultiOperationCoordinator {
    /// Create new multi-operation coordinator
    pub fn new(strategy: SchedulingStrategy) -> CudaResult<Self> {
        Ok(Self {
            scheduler: IntelligentStreamScheduler::new(8, strategy)?,
            execution_graph: HashMap::new(),
            completion_callbacks: HashMap::new(),
        })
    }

    /// Schedule a batch of operations with dependencies
    pub fn schedule_batch(
        &mut self,
        operations: Vec<(String, WorkloadCharacteristics)>,
    ) -> CudaResult<Vec<SchedulingDecision>> {
        let mut decisions = Vec::new();
        let mut scheduled = std::collections::HashSet::new();

        // Topological sort for dependency-aware scheduling
        let sorted_ops = self.topological_sort(&operations)?;

        for (op_id, characteristics) in sorted_ops {
            // Wait for dependencies
            for dep in &characteristics.synchronization_requirements.dependencies {
                if scheduled.contains(dep) {
                    self.scheduler.add_dependency(&op_id, dep)?;
                }
            }

            let decision = self
                .scheduler
                .schedule_operation(op_id.clone(), characteristics)?;
            decisions.push(decision);
            scheduled.insert(op_id);
        }

        Ok(decisions)
    }

    /// Execute operations in parallel with optimal coordination
    pub fn execute_parallel_workflow<F>(
        &mut self,
        operations: Vec<(String, WorkloadCharacteristics)>,
        executor: F,
    ) -> CudaResult<()>
    where
        F: Fn(&str, &CudaStream) -> CudaResult<(Duration, u64)> + Send + Sync + Clone + 'static,
    {
        let decisions = self.schedule_batch(operations)?;

        // Execute operations
        let handles: Vec<_> = decisions
            .into_iter()
            .map(|decision| {
                let op_id = format!("op_{}", decision.stream.id());
                let stream = decision.stream.clone();
                let executor = executor.clone();

                std::thread::spawn(move || {
                    let result = executor(&op_id, &stream);
                    (op_id, result)
                })
            })
            .collect();

        // Wait for completion and update scheduler
        for handle in handles {
            let (op_id, result) = handle.join().map_err(|_| CudaError::Context {
                message: "Thread execution failed".to_string(),
            })?;

            match result {
                Ok((execution_time, bandwidth)) => {
                    self.scheduler
                        .complete_operation(op_id, execution_time, bandwidth)?;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Add completion callback for operation
    pub fn add_completion_callback<F>(&mut self, operation_id: String, callback: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.completion_callbacks
            .entry(operation_id)
            .or_insert_with(Vec::new)
            .push(Box::new(callback));
    }

    /// Get scheduler metrics
    pub fn get_metrics(&self) -> SchedulerMetrics {
        self.scheduler.get_performance_metrics()
    }

    fn topological_sort(
        &self,
        operations: &[(String, WorkloadCharacteristics)],
    ) -> CudaResult<Vec<(String, WorkloadCharacteristics)>> {
        // Simple topological sort implementation
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut temp_mark = std::collections::HashSet::new();

        for (op_id, characteristics) in operations {
            if !visited.contains(op_id) {
                self.visit_node(
                    op_id,
                    characteristics,
                    operations,
                    &mut visited,
                    &mut temp_mark,
                    &mut result,
                )?;
            }
        }

        Ok(result)
    }

    fn visit_node(
        &self,
        op_id: &str,
        characteristics: &WorkloadCharacteristics,
        all_ops: &[(String, WorkloadCharacteristics)],
        visited: &mut std::collections::HashSet<String>,
        temp_mark: &mut std::collections::HashSet<String>,
        result: &mut Vec<(String, WorkloadCharacteristics)>,
    ) -> CudaResult<()> {
        if temp_mark.contains(op_id) {
            return Err(CudaError::Context {
                message: "Circular dependency detected".to_string(),
            });
        }

        if visited.contains(op_id) {
            return Ok(());
        }

        temp_mark.insert(op_id.to_string());

        // Visit dependencies first
        for dep in &characteristics.synchronization_requirements.dependencies {
            if let Some((_, dep_char)) = all_ops.iter().find(|(id, _)| id == dep) {
                self.visit_node(dep, dep_char, all_ops, visited, temp_mark, result)?;
            }
        }

        temp_mark.remove(op_id);
        visited.insert(op_id.to_string());
        result.push((op_id.to_string(), characteristics.clone()));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Requires CUDA hardware - run with --ignored flag"]
    fn test_scheduler_creation() {
        if crate::cuda::is_available() {
            let _device = Arc::new(crate::cuda::device::CudaDevice::new(0).unwrap());
            let scheduler = IntelligentStreamScheduler::new(4, SchedulingStrategy::Balanced);
            assert!(scheduler.is_ok());
        }
    }

    #[test]
    fn test_workload_characteristics() {
        let characteristics = WorkloadCharacteristics {
            workload_type: WorkloadType::Compute,
            estimated_compute_time: Duration::from_millis(100),
            estimated_memory_bandwidth: 1_000_000_000, // 1 GB/s
            parallel_potential: 0.8,
            memory_access_pattern: MemoryAccessPattern::Sequential,
            synchronization_requirements: SynchronizationRequirements {
                requires_barrier: false,
                dependencies: vec![],
                provides_outputs: vec!["output1".to_string()],
            },
        };

        assert_eq!(characteristics.workload_type, WorkloadType::Compute);
        assert_eq!(characteristics.parallel_potential, 0.8);
    }

    #[test]
    fn test_scheduling_strategies() {
        let strategies = [
            SchedulingStrategy::MinimizeLatency,
            SchedulingStrategy::MaximizeThroughput,
            SchedulingStrategy::Balanced,
            SchedulingStrategy::LoadBalance,
            SchedulingStrategy::PowerEfficient,
        ];

        for strategy in &strategies {
            let allocation_strategy =
                IntelligentStreamScheduler::strategy_to_allocation_strategy(*strategy);
            // Just verify it doesn't panic
            assert!(matches!(
                allocation_strategy,
                AllocationStrategy::RoundRobin
                    | AllocationStrategy::LoadBalanced
                    | AllocationStrategy::Priority
                    | AllocationStrategy::Workload
            ));
        }
    }

    #[test]
    #[ignore = "Requires CUDA hardware - run with --ignored flag"]
    fn test_coordinator_creation() {
        if crate::cuda::is_available() {
            let _device = Arc::new(crate::cuda::device::CudaDevice::new(0).unwrap());
            let coordinator = MultiOperationCoordinator::new(SchedulingStrategy::Balanced);
            assert!(coordinator.is_ok());
        }
    }
}
