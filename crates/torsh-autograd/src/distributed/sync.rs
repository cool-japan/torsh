//! Gradient synchronization scheduling and coordination
//!
//! This module provides tools for scheduling and coordinating gradient synchronization
//! operations in distributed training to optimize communication patterns and throughput.

use std::collections::VecDeque;
use std::time::Duration;

/// Gradient synchronization scheduler for optimizing communication patterns
pub struct GradientSyncScheduler {
    /// Pending synchronization operations
    pending_syncs: VecDeque<PendingSyncOp>,
    /// Maximum number of concurrent syncs
    max_concurrent_syncs: usize,
    /// Current active syncs
    active_syncs: usize,
}

/// Pending synchronization operation
#[derive(Debug, Clone)]
pub struct PendingSyncOp {
    /// Priority of this operation (higher is more urgent)
    pub priority: i32,
    /// Parameter groups to synchronize
    pub parameter_groups: Vec<String>,
    /// Estimated completion time
    pub estimated_time: Duration,
}

impl GradientSyncScheduler {
    /// Create a new synchronization scheduler
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            pending_syncs: VecDeque::new(),
            max_concurrent_syncs: max_concurrent,
            active_syncs: 0,
        }
    }

    /// Schedule a gradient synchronization operation
    pub fn schedule_sync(&mut self, parameter_groups: Vec<String>, priority: i32) {
        let op = PendingSyncOp {
            priority,
            parameter_groups,
            estimated_time: Duration::from_millis(100), // Placeholder estimate
        };

        // Insert in priority order
        let mut inserted = false;
        for (i, pending) in self.pending_syncs.iter().enumerate() {
            if op.priority > pending.priority {
                self.pending_syncs.insert(i, op.clone());
                inserted = true;
                break;
            }
        }

        if !inserted {
            self.pending_syncs.push_back(op);
        }
    }

    /// Schedule a synchronization operation with custom estimated time
    pub fn schedule_sync_with_estimate(
        &mut self,
        parameter_groups: Vec<String>,
        priority: i32,
        estimated_time: Duration,
    ) {
        let op = PendingSyncOp {
            priority,
            parameter_groups,
            estimated_time,
        };

        // Insert in priority order
        let mut inserted = false;
        for (i, pending) in self.pending_syncs.iter().enumerate() {
            if op.priority > pending.priority {
                self.pending_syncs.insert(i, op);
                inserted = true;
                break;
            }
        }

        if !inserted {
            self.pending_syncs.push_back(op);
        }
    }

    /// Get the next sync operation to execute
    pub fn next_sync(&mut self) -> Option<PendingSyncOp> {
        if self.active_syncs < self.max_concurrent_syncs {
            if let Some(op) = self.pending_syncs.pop_front() {
                self.active_syncs += 1;
                return Some(op);
            }
        }
        None
    }

    /// Mark a sync operation as completed
    pub fn complete_sync(&mut self) {
        if self.active_syncs > 0 {
            self.active_syncs -= 1;
        }
    }

    /// Get number of pending operations
    pub fn pending_count(&self) -> usize {
        self.pending_syncs.len()
    }

    /// Get number of active operations
    pub fn active_count(&self) -> usize {
        self.active_syncs
    }

    /// Get maximum concurrent operations
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent_syncs
    }

    /// Set maximum concurrent operations
    pub fn set_max_concurrent(&mut self, max_concurrent: usize) {
        self.max_concurrent_syncs = max_concurrent;
    }

    /// Clear all pending operations
    pub fn clear_pending(&mut self) {
        self.pending_syncs.clear();
    }

    /// Get estimated total time for all pending operations
    pub fn estimated_total_time(&self) -> Duration {
        let total_time_parallel = if self.pending_syncs.is_empty() {
            Duration::from_millis(0)
        } else {
            // Calculate time assuming operations can run in parallel
            let mut time_slots = vec![Duration::from_millis(0); self.max_concurrent_syncs];

            for op in &self.pending_syncs {
                // Find the earliest available slot
                let earliest_slot = time_slots.iter_mut().min().expect("reduction should succeed");
                *earliest_slot += op.estimated_time;
            }

            *time_slots.iter().max().expect("reduction should succeed")
        };

        total_time_parallel
    }

    /// Get the highest priority in pending operations
    pub fn highest_pending_priority(&self) -> Option<i32> {
        self.pending_syncs.front().map(|op| op.priority)
    }

    /// Get all pending operations sorted by priority
    pub fn get_pending_operations(&self) -> Vec<&PendingSyncOp> {
        self.pending_syncs.iter().collect()
    }

    /// Force execute next operation even if at concurrent limit
    pub fn force_next_sync(&mut self) -> Option<PendingSyncOp> {
        self.pending_syncs.pop_front().map(|op| {
            self.active_syncs += 1;
            op
        })
    }

    /// Update estimated time for operations affecting specific parameter groups
    pub fn update_estimates_for_groups(&mut self, parameter_groups: &[String], new_estimate: Duration) {
        for op in &mut self.pending_syncs {
            // Check if this operation involves any of the specified parameter groups
            if op.parameter_groups.iter().any(|group| parameter_groups.contains(group)) {
                op.estimated_time = new_estimate;
            }
        }
    }
}

/// Synchronization coordinator for managing multiple schedulers
pub struct SyncCoordinator {
    /// Schedulers for different types of operations
    schedulers: std::collections::HashMap<String, GradientSyncScheduler>,
    /// Global operation counter
    operation_counter: usize,
    /// Coordination strategy
    strategy: CoordinationStrategy,
}

/// Strategy for coordinating multiple synchronization schedulers
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinationStrategy {
    /// Round-robin between schedulers
    RoundRobin,
    /// Priority-based selection across all schedulers
    GlobalPriority,
    /// Fair share based on estimated completion times
    FairShare,
    /// Adaptive based on recent performance
    Adaptive,
}

impl SyncCoordinator {
    /// Create a new synchronization coordinator
    pub fn new(strategy: CoordinationStrategy) -> Self {
        Self {
            schedulers: std::collections::HashMap::new(),
            operation_counter: 0,
            strategy,
        }
    }

    /// Add a scheduler for a specific operation type
    pub fn add_scheduler(&mut self, name: String, scheduler: GradientSyncScheduler) {
        self.schedulers.insert(name, scheduler);
    }

    /// Get a mutable reference to a scheduler
    pub fn get_scheduler_mut(&mut self, name: &str) -> Option<&mut GradientSyncScheduler> {
        self.schedulers.get_mut(name)
    }

    /// Get a reference to a scheduler
    pub fn get_scheduler(&self, name: &str) -> Option<&GradientSyncScheduler> {
        self.schedulers.get(name)
    }

    /// Schedule an operation on a specific scheduler
    pub fn schedule_on(&mut self, scheduler_name: &str, parameter_groups: Vec<String>, priority: i32) -> bool {
        if let Some(scheduler) = self.schedulers.get_mut(scheduler_name) {
            scheduler.schedule_sync(parameter_groups, priority);
            true
        } else {
            false
        }
    }

    /// Get the next operation to execute using the coordination strategy
    pub fn next_operation(&mut self) -> Option<(String, PendingSyncOp)> {
        match self.strategy {
            CoordinationStrategy::RoundRobin => self.next_round_robin(),
            CoordinationStrategy::GlobalPriority => self.next_global_priority(),
            CoordinationStrategy::FairShare => self.next_fair_share(),
            CoordinationStrategy::Adaptive => self.next_adaptive(),
        }
    }

    /// Round-robin selection
    fn next_round_robin(&mut self) -> Option<(String, PendingSyncOp)> {
        let scheduler_names: Vec<String> = self.schedulers.keys().cloned().collect();
        if scheduler_names.is_empty() {
            return None;
        }

        let start_idx = self.operation_counter % scheduler_names.len();

        for i in 0..scheduler_names.len() {
            let idx = (start_idx + i) % scheduler_names.len();
            let name = &scheduler_names[idx];

            if let Some(scheduler) = self.schedulers.get_mut(name) {
                if let Some(op) = scheduler.next_sync() {
                    self.operation_counter += 1;
                    return Some((name.clone(), op));
                }
            }
        }

        None
    }

    /// Global priority selection
    fn next_global_priority(&mut self) -> Option<(String, PendingSyncOp)> {
        let mut best_priority = i32::MIN;
        let mut best_scheduler = None;

        for (name, scheduler) in &self.schedulers {
            if let Some(priority) = scheduler.highest_pending_priority() {
                if priority > best_priority {
                    best_priority = priority;
                    best_scheduler = Some(name.clone());
                }
            }
        }

        if let Some(scheduler_name) = best_scheduler {
            if let Some(scheduler) = self.schedulers.get_mut(&scheduler_name) {
                if let Some(op) = scheduler.next_sync() {
                    self.operation_counter += 1;
                    return Some((scheduler_name, op));
                }
            }
        }

        None
    }

    /// Fair share selection based on estimated completion times
    fn next_fair_share(&mut self) -> Option<(String, PendingSyncOp)> {
        let mut min_estimated_total = Duration::from_secs(u64::MAX);
        let mut best_scheduler = None;

        for (name, scheduler) in &self.schedulers {
            let estimated_total = scheduler.estimated_total_time();
            if estimated_total < min_estimated_total && scheduler.pending_count() > 0 {
                min_estimated_total = estimated_total;
                best_scheduler = Some(name.clone());
            }
        }

        if let Some(scheduler_name) = best_scheduler {
            if let Some(scheduler) = self.schedulers.get_mut(&scheduler_name) {
                if let Some(op) = scheduler.next_sync() {
                    self.operation_counter += 1;
                    return Some((scheduler_name, op));
                }
            }
        }

        None
    }

    /// Adaptive selection (placeholder implementation)
    fn next_adaptive(&mut self) -> Option<(String, PendingSyncOp)> {
        // For now, fall back to global priority
        // In a real implementation, this would track performance metrics
        // and adapt the strategy based on recent execution times
        self.next_global_priority()
    }

    /// Complete an operation on a specific scheduler
    pub fn complete_operation(&mut self, scheduler_name: &str) {
        if let Some(scheduler) = self.schedulers.get_mut(scheduler_name) {
            scheduler.complete_sync();
        }
    }

    /// Get total pending operations across all schedulers
    pub fn total_pending(&self) -> usize {
        self.schedulers.values().map(|s| s.pending_count()).sum()
    }

    /// Get total active operations across all schedulers
    pub fn total_active(&self) -> usize {
        self.schedulers.values().map(|s| s.active_count()).sum()
    }

    /// Get coordination strategy
    pub fn strategy(&self) -> &CoordinationStrategy {
        &self.strategy
    }

    /// Set coordination strategy
    pub fn set_strategy(&mut self, strategy: CoordinationStrategy) {
        self.strategy = strategy;
    }
}

impl Default for SyncCoordinator {
    fn default() -> Self {
        Self::new(CoordinationStrategy::GlobalPriority)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_sync_scheduler() {
        let mut scheduler = GradientSyncScheduler::new(2);
        assert_eq!(scheduler.pending_count(), 0);
        assert_eq!(scheduler.active_count(), 0);

        scheduler.schedule_sync(vec!["param1".to_string()], 1);
        assert_eq!(scheduler.pending_count(), 1);

        let op = scheduler.next_sync();
        assert!(op.is_some());
        assert_eq!(scheduler.active_count(), 1);
        assert_eq!(scheduler.pending_count(), 0);

        scheduler.complete_sync();
        assert_eq!(scheduler.active_count(), 0);
    }

    #[test]
    fn test_priority_ordering() {
        let mut scheduler = GradientSyncScheduler::new(1);

        scheduler.schedule_sync(vec!["low".to_string()], 1);
        scheduler.schedule_sync(vec!["high".to_string()], 10);
        scheduler.schedule_sync(vec!["medium".to_string()], 5);

        let op1 = scheduler.next_sync().unwrap();
        assert_eq!(op1.priority, 10);

        scheduler.complete_sync();
        let op2 = scheduler.next_sync().unwrap();
        assert_eq!(op2.priority, 5);
    }

    #[test]
    fn test_sync_coordinator() {
        let mut coordinator = SyncCoordinator::new(CoordinationStrategy::GlobalPriority);

        let scheduler1 = GradientSyncScheduler::new(2);
        let scheduler2 = GradientSyncScheduler::new(2);

        coordinator.add_scheduler("grad".to_string(), scheduler1);
        coordinator.add_scheduler("param".to_string(), scheduler2);

        coordinator.schedule_on("grad", vec!["layer1".to_string()], 5);
        coordinator.schedule_on("param", vec!["layer2".to_string()], 10);

        let (scheduler_name, op) = coordinator.next_operation().unwrap();
        assert_eq!(scheduler_name, "param");
        assert_eq!(op.priority, 10);
    }
}