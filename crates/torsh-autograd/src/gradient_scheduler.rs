//! Gradient computation scheduling optimization
//!
//! This module provides intelligent scheduling of gradient computations to optimize
//! performance, memory usage, and parallelization in autograd pipelines. It includes
//! dependency analysis, critical path optimization, memory-aware scheduling, and
//! adaptive scheduling strategies.

use crate::error_handling::{AutogradError, AutogradResult};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

/// Priority levels for gradient computation tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskPriority {
    Critical = 4,
    High = 3,
    Normal = 2,
    Low = 1,
    Background = 0,
}

/// Types of gradient computation tasks
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TaskType {
    Forward,
    Backward,
    ParameterUpdate,
    GradientAccumulation,
    MemoryCleanup,
    Checkpoint,
}

/// Gradient computation task
#[derive(Debug, Clone)]
pub struct GradientTask {
    pub id: usize,
    pub task_type: TaskType,
    pub priority: TaskPriority,
    pub estimated_duration: Duration,
    pub estimated_memory: usize,
    pub dependencies: Vec<usize>,
    pub tensor_ids: Vec<usize>,
    pub created_at: Instant,
    pub deadline: Option<Instant>,
    pub parallelizable: bool,
    pub memory_intensive: bool,
}

impl GradientTask {
    /// Create a new gradient task
    pub fn new(
        id: usize,
        task_type: TaskType,
        priority: TaskPriority,
        estimated_duration: Duration,
        estimated_memory: usize,
    ) -> Self {
        Self {
            id,
            task_type,
            priority,
            estimated_duration,
            estimated_memory,
            dependencies: Vec::new(),
            tensor_ids: Vec::new(),
            created_at: Instant::now(),
            deadline: None,
            parallelizable: false,
            memory_intensive: false,
        }
    }

    /// Add a dependency to this task
    pub fn add_dependency(&mut self, dependency_id: usize) {
        if !self.dependencies.contains(&dependency_id) {
            self.dependencies.push(dependency_id);
        }
    }

    /// Add multiple dependencies
    pub fn add_dependencies(&mut self, dependency_ids: &[usize]) {
        for &dep_id in dependency_ids {
            self.add_dependency(dep_id);
        }
    }

    /// Set deadline for task completion
    pub fn set_deadline(&mut self, deadline: Instant) {
        self.deadline = Some(deadline);
    }

    /// Mark task as parallelizable
    pub fn set_parallelizable(&mut self, parallelizable: bool) {
        self.parallelizable = parallelizable;
    }

    /// Mark task as memory intensive
    pub fn set_memory_intensive(&mut self, memory_intensive: bool) {
        self.memory_intensive = memory_intensive;
    }

    /// Get time until deadline
    pub fn time_until_deadline(&self) -> Option<Duration> {
        self.deadline.map(|deadline| {
            let now = Instant::now();
            if deadline > now {
                deadline.duration_since(now)
            } else {
                Duration::from_secs(0)
            }
        })
    }

    /// Check if task is overdue
    pub fn is_overdue(&self) -> bool {
        self.deadline
            .map_or(false, |deadline| Instant::now() > deadline)
    }
}

impl PartialEq for GradientTask {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for GradientTask {}

impl PartialOrd for GradientTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GradientTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: Priority (higher priority first)
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {
                // Secondary: Deadline urgency (closer deadline first)
                match (self.time_until_deadline(), other.time_until_deadline()) {
                    (Some(a), Some(b)) => a.cmp(&b),
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => {
                        // Tertiary: Task ID (for deterministic ordering)
                        self.id.cmp(&other.id)
                    }
                }
            }
            other => other, // Reverse for max-heap behavior
        }
    }
}

/// Scheduling strategy for gradient computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingStrategy {
    /// First-Come-First-Served
    FIFO,
    /// Priority-based scheduling
    Priority,
    /// Shortest-Job-First
    SJF,
    /// Critical path optimization
    CriticalPath,
    /// Memory-aware scheduling
    MemoryAware,
    /// Adaptive scheduling based on runtime performance
    Adaptive,
}

/// Resource constraints for scheduling
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub max_memory: usize,
    pub max_concurrent_tasks: usize,
    pub available_threads: usize,
    pub memory_threshold: f64, // 0.0 to 1.0
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory: 8 * 1024 * 1024 * 1024, // 8GB
            max_concurrent_tasks: num_cpus::get(),
            available_threads: num_cpus::get(),
            memory_threshold: 0.8, // 80% memory threshold
        }
    }
}

/// Statistics about scheduling performance
#[derive(Debug, Clone, Default)]
pub struct SchedulingStats {
    pub tasks_scheduled: usize,
    pub tasks_completed: usize,
    pub tasks_failed: usize,
    pub total_scheduling_time: Duration,
    pub average_wait_time: Duration,
    pub average_execution_time: Duration,
    pub memory_utilization: f64,
    pub thread_utilization: f64,
    pub deadline_misses: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Gradient computation scheduler
#[derive(Debug)]
pub struct GradientScheduler {
    /// Current scheduling strategy
    strategy: SchedulingStrategy,
    /// Task queue organized as a priority queue
    task_queue: BinaryHeap<GradientTask>,
    /// Currently executing tasks
    executing_tasks: HashMap<usize, GradientTask>,
    /// Completed tasks for dependency tracking
    completed_tasks: HashMap<usize, GradientTask>,
    /// Failed tasks
    failed_tasks: HashMap<usize, (GradientTask, String)>,
    /// Resource constraints
    constraints: ResourceConstraints,
    /// Current resource usage
    current_memory_usage: usize,
    current_task_count: usize,
    /// Scheduling statistics
    stats: SchedulingStats,
    /// Task dependency graph
    dependency_graph: HashMap<usize, Vec<usize>>,
    /// Reverse dependency graph (dependents)
    reverse_dependency_graph: HashMap<usize, Vec<usize>>,
    /// Task execution history for adaptive scheduling
    execution_history: HashMap<TaskType, Vec<Duration>>,
    /// Memory usage history
    memory_history: VecDeque<(Instant, usize)>,
    /// Next task ID
    next_task_id: usize,
}

impl GradientScheduler {
    /// Create a new gradient scheduler
    pub fn new(strategy: SchedulingStrategy, constraints: ResourceConstraints) -> Self {
        Self {
            strategy,
            task_queue: BinaryHeap::new(),
            executing_tasks: HashMap::new(),
            completed_tasks: HashMap::new(),
            failed_tasks: HashMap::new(),
            constraints,
            current_memory_usage: 0,
            current_task_count: 0,
            stats: SchedulingStats::default(),
            dependency_graph: HashMap::new(),
            reverse_dependency_graph: HashMap::new(),
            execution_history: HashMap::new(),
            memory_history: VecDeque::new(),
            next_task_id: 1,
        }
    }

    /// Create a scheduler with default constraints
    pub fn with_strategy(strategy: SchedulingStrategy) -> Self {
        Self::new(strategy, ResourceConstraints::default())
    }

    /// Schedule a new gradient computation task
    pub fn schedule_task(&mut self, mut task: GradientTask) -> AutogradResult<usize> {
        let task_id = task.id;

        // Update dependency graphs
        for &dep_id in &task.dependencies {
            self.dependency_graph
                .entry(dep_id)
                .or_insert_with(Vec::new)
                .push(task_id);
            self.reverse_dependency_graph
                .entry(task_id)
                .or_insert_with(Vec::new)
                .push(dep_id);
        }

        // Apply scheduling strategy optimizations
        self.apply_scheduling_optimizations(&mut task)?;

        // Add to task queue
        self.task_queue.push(task);
        self.stats.tasks_scheduled += 1;

        Ok(task_id)
    }

    /// Create and schedule a new task
    pub fn create_and_schedule_task(
        &mut self,
        task_type: TaskType,
        priority: TaskPriority,
        estimated_duration: Duration,
        estimated_memory: usize,
    ) -> AutogradResult<usize> {
        let task_id = self.next_task_id;
        self.next_task_id += 1;

        let task = GradientTask::new(
            task_id,
            task_type,
            priority,
            estimated_duration,
            estimated_memory,
        );
        self.schedule_task(task)
    }

    /// Get the next task to execute based on scheduling strategy
    pub fn get_next_task(&mut self) -> Option<GradientTask> {
        let mut checked_tasks = Vec::new();

        while let Some(task) = self.task_queue.pop() {
            // Check if all dependencies are satisfied
            if self.are_dependencies_satisfied(&task) {
                // Check resource constraints
                if self.can_execute_task(&task) {
                    // Put back all checked tasks before returning
                    for checked_task in checked_tasks {
                        self.task_queue.push(checked_task);
                    }

                    self.executing_tasks.insert(task.id, task.clone());
                    self.current_memory_usage += task.estimated_memory;
                    self.current_task_count += 1;
                    return Some(task);
                } else {
                    // Put task back for later
                    checked_tasks.push(task);
                }
            } else {
                // Put task back for later
                checked_tasks.push(task);
            }
        }

        // Put back all checked tasks
        for task in checked_tasks {
            self.task_queue.push(task);
        }

        None
    }

    /// Mark a task as completed
    pub fn complete_task(&mut self, task_id: usize) -> AutogradResult<()> {
        if let Some(task) = self.executing_tasks.remove(&task_id) {
            self.current_memory_usage = self
                .current_memory_usage
                .saturating_sub(task.estimated_memory);
            self.current_task_count = self.current_task_count.saturating_sub(1);

            // Record execution time for adaptive scheduling
            let execution_time = task.created_at.elapsed();
            self.execution_history
                .entry(task.task_type.clone())
                .or_insert_with(Vec::new)
                .push(execution_time);

            self.completed_tasks.insert(task_id, task);
            self.stats.tasks_completed += 1;

            // Update memory history
            self.update_memory_history();

            Ok(())
        } else {
            Err(AutogradError::gradient_computation(
                "complete_task",
                format!("Task {} not found in executing tasks", task_id),
            ))
        }
    }

    /// Mark a task as failed
    pub fn fail_task(&mut self, task_id: usize, error_message: String) -> AutogradResult<()> {
        if let Some(task) = self.executing_tasks.remove(&task_id) {
            self.current_memory_usage = self
                .current_memory_usage
                .saturating_sub(task.estimated_memory);
            self.current_task_count = self.current_task_count.saturating_sub(1);

            self.failed_tasks.insert(task_id, (task, error_message));
            self.stats.tasks_failed += 1;

            Ok(())
        } else {
            Err(AutogradError::gradient_computation(
                "fail_task",
                format!("Task {} not found in executing tasks", task_id),
            ))
        }
    }

    /// Get scheduling statistics
    pub fn get_stats(&self) -> SchedulingStats {
        self.stats.clone()
    }

    /// Get current resource utilization
    pub fn get_resource_utilization(&self) -> (f64, f64) {
        let memory_util = self.current_memory_usage as f64 / self.constraints.max_memory as f64;
        let thread_util =
            self.current_task_count as f64 / self.constraints.max_concurrent_tasks as f64;
        (memory_util, thread_util)
    }

    /// Change scheduling strategy
    pub fn set_strategy(&mut self, strategy: SchedulingStrategy) {
        self.strategy = strategy;
        // Rebuild queue with new strategy
        let tasks: Vec<_> = self.task_queue.drain().collect();
        for task in tasks {
            self.task_queue.push(task);
        }
    }

    /// Update resource constraints
    pub fn update_constraints(&mut self, constraints: ResourceConstraints) {
        self.constraints = constraints;
    }

    /// Get pending tasks count
    pub fn pending_tasks_count(&self) -> usize {
        self.task_queue.len()
    }

    /// Get executing tasks count
    pub fn executing_tasks_count(&self) -> usize {
        self.executing_tasks.len()
    }

    /// Check if scheduler is idle (no pending or executing tasks)
    pub fn is_idle(&self) -> bool {
        self.task_queue.is_empty() && self.executing_tasks.is_empty()
    }

    /// Apply scheduling strategy specific optimizations
    fn apply_scheduling_optimizations(&mut self, task: &mut GradientTask) -> AutogradResult<()> {
        match self.strategy {
            SchedulingStrategy::CriticalPath => {
                self.optimize_critical_path(task)?;
            }
            SchedulingStrategy::MemoryAware => {
                self.optimize_memory_usage(task)?;
            }
            SchedulingStrategy::Adaptive => {
                self.apply_adaptive_optimizations(task)?;
            }
            _ => {
                // No special optimizations for FIFO, Priority, SJF
            }
        }
        Ok(())
    }

    /// Optimize task for critical path scheduling
    fn optimize_critical_path(&mut self, task: &mut GradientTask) -> AutogradResult<()> {
        // Calculate critical path length through dependencies
        let critical_path_length = self.calculate_critical_path_length(task);

        // Adjust priority based on critical path position
        if critical_path_length > 10 {
            task.priority = TaskPriority::Critical;
        } else if critical_path_length > 5 {
            task.priority = TaskPriority::High;
        }

        Ok(())
    }

    /// Optimize task for memory-aware scheduling
    fn optimize_memory_usage(&mut self, task: &mut GradientTask) -> AutogradResult<()> {
        // Adjust scheduling based on memory pressure
        let memory_pressure = self.current_memory_usage as f64 / self.constraints.max_memory as f64;

        if memory_pressure > 0.8 {
            // High memory pressure - prioritize memory cleanup tasks
            if matches!(task.task_type, TaskType::MemoryCleanup) {
                task.priority = TaskPriority::Critical;
            } else if task.memory_intensive {
                // Delay memory-intensive tasks
                task.priority = TaskPriority::Low;
            }
        }

        Ok(())
    }

    /// Apply adaptive optimizations based on execution history
    fn apply_adaptive_optimizations(&mut self, task: &mut GradientTask) -> AutogradResult<()> {
        // Use execution history to estimate task duration more accurately
        if let Some(history) = self.execution_history.get(&task.task_type) {
            if !history.is_empty() {
                let avg_duration = history.iter().sum::<Duration>() / history.len() as u32;
                task.estimated_duration = avg_duration;
            }
        }

        // Adjust parallelization based on historical performance
        let (memory_util, thread_util) = self.get_resource_utilization();
        if thread_util < 0.7 && !task.memory_intensive {
            task.set_parallelizable(true);
        }

        Ok(())
    }

    /// Calculate critical path length for a task
    fn calculate_critical_path_length(&self, task: &GradientTask) -> usize {
        let mut visited = std::collections::HashSet::new();
        self.calculate_critical_path_recursive(task.id, &mut visited)
    }

    /// Recursive helper for critical path calculation
    fn calculate_critical_path_recursive(
        &self,
        task_id: usize,
        visited: &mut std::collections::HashSet<usize>,
    ) -> usize {
        if visited.contains(&task_id) {
            return 0; // Avoid cycles
        }
        visited.insert(task_id);

        if let Some(dependents) = self.dependency_graph.get(&task_id) {
            1 + dependents
                .iter()
                .map(|&dep_id| self.calculate_critical_path_recursive(dep_id, visited))
                .max()
                .unwrap_or(0)
        } else {
            1
        }
    }

    /// Check if all dependencies for a task are satisfied
    fn are_dependencies_satisfied(&self, task: &GradientTask) -> bool {
        task.dependencies
            .iter()
            .all(|&dep_id| self.completed_tasks.contains_key(&dep_id))
    }

    /// Check if a task can be executed given current resource constraints
    fn can_execute_task(&self, task: &GradientTask) -> bool {
        let memory_ok =
            self.current_memory_usage + task.estimated_memory <= self.constraints.max_memory;
        let concurrency_ok = self.current_task_count < self.constraints.max_concurrent_tasks;

        memory_ok && concurrency_ok
    }

    /// Update memory usage history
    fn update_memory_history(&mut self) {
        let now = Instant::now();
        self.memory_history
            .push_back((now, self.current_memory_usage));

        // Keep only recent history (last 1000 entries)
        while self.memory_history.len() > 1000 {
            self.memory_history.pop_front();
        }
    }
}

impl Default for GradientScheduler {
    fn default() -> Self {
        Self::with_strategy(SchedulingStrategy::Adaptive)
    }
}

/// Thread-safe scheduler wrapper for concurrent access
#[derive(Debug)]
pub struct ThreadSafeGradientScheduler {
    scheduler: Arc<RwLock<GradientScheduler>>,
}

impl ThreadSafeGradientScheduler {
    /// Create a new thread-safe scheduler
    pub fn new(strategy: SchedulingStrategy, constraints: ResourceConstraints) -> Self {
        Self {
            scheduler: Arc::new(RwLock::new(GradientScheduler::new(strategy, constraints))),
        }
    }

    /// Schedule a task
    pub fn schedule_task(&self, task: GradientTask) -> AutogradResult<usize> {
        self.scheduler.write().unwrap().schedule_task(task)
    }

    /// Get next task
    pub fn get_next_task(&self) -> Option<GradientTask> {
        self.scheduler.write().unwrap().get_next_task()
    }

    /// Complete a task
    pub fn complete_task(&self, task_id: usize) -> AutogradResult<()> {
        self.scheduler.write().unwrap().complete_task(task_id)
    }

    /// Fail a task
    pub fn fail_task(&self, task_id: usize, error_message: String) -> AutogradResult<()> {
        self.scheduler
            .write()
            .unwrap()
            .fail_task(task_id, error_message)
    }

    /// Get statistics
    pub fn get_stats(&self) -> SchedulingStats {
        self.scheduler.read().unwrap().get_stats()
    }

    /// Get resource utilization
    pub fn get_resource_utilization(&self) -> (f64, f64) {
        self.scheduler.read().unwrap().get_resource_utilization()
    }

    /// Check if idle
    pub fn is_idle(&self) -> bool {
        self.scheduler.read().unwrap().is_idle()
    }
}

/// Global scheduler instance
static GLOBAL_SCHEDULER: once_cell::sync::Lazy<ThreadSafeGradientScheduler> =
    once_cell::sync::Lazy::new(|| {
        ThreadSafeGradientScheduler::new(
            SchedulingStrategy::Adaptive,
            ResourceConstraints::default(),
        )
    });

/// Get the global gradient scheduler
pub fn get_global_scheduler() -> &'static ThreadSafeGradientScheduler {
    &GLOBAL_SCHEDULER
}

/// Convenience functions for global scheduler
pub fn schedule_gradient_task(
    task_type: TaskType,
    priority: TaskPriority,
    estimated_duration: Duration,
    estimated_memory: usize,
) -> AutogradResult<usize> {
    let mut scheduler = GLOBAL_SCHEDULER.scheduler.write().unwrap();
    scheduler.create_and_schedule_task(task_type, priority, estimated_duration, estimated_memory)
}

pub fn get_next_gradient_task() -> Option<GradientTask> {
    GLOBAL_SCHEDULER.get_next_task()
}

pub fn complete_gradient_task(task_id: usize) -> AutogradResult<()> {
    GLOBAL_SCHEDULER.complete_task(task_id)
}

pub fn fail_gradient_task(task_id: usize, error_message: String) -> AutogradResult<()> {
    GLOBAL_SCHEDULER.fail_task(task_id, error_message)
}

pub fn get_scheduler_stats() -> SchedulingStats {
    GLOBAL_SCHEDULER.get_stats()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_task_creation() {
        let task = GradientTask::new(
            1,
            TaskType::Backward,
            TaskPriority::High,
            Duration::from_millis(100),
            1024,
        );

        assert_eq!(task.id, 1);
        assert_eq!(task.task_type, TaskType::Backward);
        assert_eq!(task.priority, TaskPriority::High);
        assert_eq!(task.estimated_memory, 1024);
        assert!(!task.is_overdue());
    }

    #[test]
    fn test_task_dependencies() {
        let mut task = GradientTask::new(
            2,
            TaskType::ParameterUpdate,
            TaskPriority::Normal,
            Duration::from_millis(50),
            512,
        );

        task.add_dependency(1);
        task.add_dependencies(&[3, 4]);

        assert_eq!(task.dependencies, vec![1, 3, 4]);
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = GradientScheduler::with_strategy(SchedulingStrategy::Priority);
        assert!(scheduler.is_idle());
        assert_eq!(scheduler.pending_tasks_count(), 0);
        assert_eq!(scheduler.executing_tasks_count(), 0);
    }

    #[test]
    fn test_task_scheduling_and_execution() {
        let mut scheduler = GradientScheduler::with_strategy(SchedulingStrategy::Priority);

        // Create and schedule a task
        let task_id = scheduler
            .create_and_schedule_task(
                TaskType::Forward,
                TaskPriority::High,
                Duration::from_millis(100),
                1024,
            )
            .unwrap();

        assert_eq!(scheduler.pending_tasks_count(), 1);

        // Get the next task
        let task = scheduler.get_next_task().unwrap();
        assert_eq!(task.id, task_id);
        assert_eq!(scheduler.executing_tasks_count(), 1);

        // Complete the task
        scheduler.complete_task(task_id).unwrap();
        assert_eq!(scheduler.executing_tasks_count(), 0);
        assert_eq!(scheduler.stats.tasks_completed, 1);
    }

    #[test]
    fn test_dependency_satisfaction() {
        let mut scheduler = GradientScheduler::with_strategy(SchedulingStrategy::Priority);

        // Create task 1
        let task1_id = scheduler
            .create_and_schedule_task(
                TaskType::Forward,
                TaskPriority::High,
                Duration::from_millis(100),
                1024,
            )
            .unwrap();

        // Create task 2 that depends on task 1
        let task2_id = scheduler.next_task_id;
        scheduler.next_task_id += 1;
        let mut task2 = GradientTask::new(
            task2_id,
            TaskType::Backward,
            TaskPriority::High,
            Duration::from_millis(100),
            1024,
        );
        task2.add_dependency(task1_id);
        scheduler.schedule_task(task2).unwrap();

        // Task 1 should be available, task 2 should not
        let next_task = scheduler.get_next_task().unwrap();
        assert_eq!(next_task.id, task1_id);

        // No more tasks should be available until task 1 completes
        assert!(scheduler.get_next_task().is_none());

        // Complete task 1
        scheduler.complete_task(task1_id).unwrap();

        // Now task 2 should be available
        let next_task = scheduler.get_next_task().unwrap();
        assert_eq!(next_task.id, task2_id);
    }

    #[test]
    fn test_resource_constraints() {
        let constraints = ResourceConstraints {
            max_memory: 1024,
            max_concurrent_tasks: 1,
            available_threads: 1,
            memory_threshold: 0.8,
        };

        let mut scheduler = GradientScheduler::new(SchedulingStrategy::MemoryAware, constraints);

        // Schedule a task that uses all available memory
        let task_id = scheduler
            .create_and_schedule_task(
                TaskType::Forward,
                TaskPriority::High,
                Duration::from_millis(100),
                1024,
            )
            .unwrap();

        // Get the task
        let _task = scheduler.get_next_task().unwrap();

        // Try to schedule another task - should not be able to execute due to memory constraint
        let task2_id = scheduler
            .create_and_schedule_task(
                TaskType::Backward,
                TaskPriority::High,
                Duration::from_millis(100),
                512,
            )
            .unwrap();

        // No more tasks should be available due to resource constraints
        assert!(scheduler.get_next_task().is_none());

        // Complete first task
        scheduler.complete_task(task_id).unwrap();

        // Now second task should be available
        let next_task = scheduler.get_next_task().unwrap();
        assert_eq!(next_task.id, task2_id);
    }

    #[test]
    fn test_thread_safe_scheduler() {
        let scheduler = ThreadSafeGradientScheduler::new(
            SchedulingStrategy::Adaptive,
            ResourceConstraints::default(),
        );

        // Create and schedule a task
        let task = GradientTask::new(
            1,
            TaskType::Forward,
            TaskPriority::High,
            Duration::from_millis(100),
            1024,
        );

        let task_id = scheduler.schedule_task(task).unwrap();
        assert_eq!(task_id, 1);

        // Get and complete the task
        let task = scheduler.get_next_task().unwrap();
        scheduler.complete_task(task.id).unwrap();

        assert!(scheduler.is_idle());
    }

    #[test]
    fn test_global_scheduler_functions() {
        // Schedule a task using global functions
        let task_id = schedule_gradient_task(
            TaskType::GradientAccumulation,
            TaskPriority::Normal,
            Duration::from_millis(50),
            512,
        )
        .unwrap();

        // Get and complete the task
        if let Some(task) = get_next_gradient_task() {
            assert_eq!(task.id, task_id);
            complete_gradient_task(task.id).unwrap();
        }

        let stats = get_scheduler_stats();
        assert!(stats.tasks_completed > 0);
    }

    #[test]
    fn test_task_priority_ordering() {
        let task1 = GradientTask::new(
            1,
            TaskType::Forward,
            TaskPriority::Low,
            Duration::from_millis(100),
            1024,
        );
        let task2 = GradientTask::new(
            2,
            TaskType::Backward,
            TaskPriority::High,
            Duration::from_millis(100),
            1024,
        );
        let task3 = GradientTask::new(
            3,
            TaskType::ParameterUpdate,
            TaskPriority::Critical,
            Duration::from_millis(100),
            1024,
        );

        let mut scheduler = GradientScheduler::with_strategy(SchedulingStrategy::Priority);
        scheduler.schedule_task(task1).unwrap();
        scheduler.schedule_task(task2).unwrap();
        scheduler.schedule_task(task3).unwrap();

        // Tasks should be executed in priority order: Critical, High, Low
        let task = scheduler.get_next_task().unwrap();
        assert_eq!(task.priority, TaskPriority::Critical);

        let task = scheduler.get_next_task().unwrap();
        assert_eq!(task.priority, TaskPriority::High);

        let task = scheduler.get_next_task().unwrap();
        assert_eq!(task.priority, TaskPriority::Low);
    }
}
