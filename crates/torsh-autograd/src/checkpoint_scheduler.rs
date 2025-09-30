//! Checkpoint scheduling for optimal memory-compute trade-offs
//!
//! This module provides intelligent checkpoint scheduling algorithms that analyze
//! the computation graph and determine optimal checkpoint placement to minimize
//! memory usage while balancing computational overhead.

use crate::context::{AutogradContext, GraphStats};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use torsh_core::error::Result;
use tracing::{debug, info, warn};

/// Checkpoint scheduling strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckpointStrategy {
    /// Memory-first strategy: minimize memory usage
    MemoryFirst,
    /// Compute-first strategy: minimize computation overhead
    ComputeFirst,
    /// Balanced strategy: balance memory and computation
    Balanced,
    /// Adaptive strategy: choose based on system conditions
    Adaptive,
    /// Custom strategy with user-defined parameters
    Custom {
        memory_weight: f32,
        compute_weight: f32,
    },
}

/// Checkpoint scheduling configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Scheduling strategy
    pub strategy: CheckpointStrategy,
    /// Maximum memory usage (bytes)
    pub max_memory: Option<usize>,
    /// Target memory usage (bytes)
    pub target_memory: Option<usize>,
    /// Maximum computation overhead ratio
    pub max_compute_overhead: f32,
    /// Minimum interval between checkpoints (operations)
    pub min_checkpoint_interval: usize,
    /// Maximum interval between checkpoints (operations)
    pub max_checkpoint_interval: usize,
    /// Enable dynamic adjustment based on system conditions
    pub enable_adaptive: bool,
    /// Memory pressure threshold for triggering aggressive checkpointing
    pub memory_pressure_threshold: f32,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            strategy: CheckpointStrategy::Balanced,
            max_memory: None,
            target_memory: None,
            max_compute_overhead: 2.0, // 2x computation overhead max
            min_checkpoint_interval: 10,
            max_checkpoint_interval: 1000,
            enable_adaptive: true,
            memory_pressure_threshold: 0.8, // 80% memory usage
        }
    }
}

/// Checkpoint scheduler for managing automatic checkpointing
pub struct CheckpointScheduler {
    config: CheckpointConfig,
    operation_count: usize,
    last_checkpoint: usize,
    memory_history: VecDeque<(Instant, usize)>,
    compute_history: VecDeque<(Instant, Duration)>,
    checkpoint_history: Vec<CheckpointEvent>,
    system_memory: Option<usize>,
    memory_pressure: f32,
}

/// Event recorded when a checkpoint is created
#[derive(Debug, Clone)]
pub struct CheckpointEvent {
    pub timestamp: Instant,
    pub operation_count: usize,
    pub memory_usage: usize,
    pub strategy_used: CheckpointStrategy,
    pub reason: CheckpointReason,
}

/// Reason for creating a checkpoint
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckpointReason {
    /// Regular interval-based checkpoint
    Interval,
    /// Memory pressure triggered checkpoint
    MemoryPressure,
    /// Computation complexity triggered checkpoint
    ComputeComplexity,
    /// User-requested checkpoint
    Manual,
    /// Adaptive algorithm decision
    Adaptive,
}

/// Checkpoint placement decision
#[derive(Debug, Clone)]
pub struct CheckpointDecision {
    pub should_checkpoint: bool,
    pub strategy: CheckpointStrategy,
    pub reason: CheckpointReason,
    pub estimated_memory_savings: usize,
    pub estimated_compute_overhead: f32,
}

impl CheckpointScheduler {
    /// Create a new checkpoint scheduler
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            operation_count: 0,
            last_checkpoint: 0,
            memory_history: VecDeque::with_capacity(1000),
            compute_history: VecDeque::with_capacity(1000),
            checkpoint_history: Vec::new(),
            system_memory: Self::detect_system_memory(),
            memory_pressure: 0.0,
        }
    }

    /// Create a scheduler with default configuration
    pub fn default() -> Self {
        Self::new(CheckpointConfig::default())
    }

    /// Update configuration
    pub fn update_config(&mut self, config: CheckpointConfig) {
        self.config = config;
        info!(
            "Checkpoint scheduler configuration updated: {:?}",
            self.config
        );
    }

    /// Record an operation and check if checkpointing is needed
    pub fn record_operation(&mut self, ctx: &AutogradContext) -> Result<CheckpointDecision> {
        self.operation_count += 1;

        // Update memory and compute history
        self.update_history(ctx)?;

        // Check if we should create a checkpoint
        let decision = self.should_checkpoint(ctx)?;

        if decision.should_checkpoint {
            self.record_checkpoint(decision.clone())?;
        }

        Ok(decision)
    }

    /// Check if a checkpoint should be created
    fn should_checkpoint(&mut self, ctx: &AutogradContext) -> Result<CheckpointDecision> {
        let stats = ctx.graph_stats();
        let ops_since_last = self.operation_count - self.last_checkpoint;

        // Check interval-based checkpointing
        if ops_since_last >= self.config.max_checkpoint_interval {
            return Ok(CheckpointDecision {
                should_checkpoint: true,
                strategy: self.config.strategy,
                reason: CheckpointReason::Interval,
                estimated_memory_savings: self.estimate_memory_savings(&stats),
                estimated_compute_overhead: self.estimate_compute_overhead(&stats),
            });
        }

        // Check memory pressure
        if let Some(decision) = self.check_memory_pressure(&stats)? {
            return Ok(decision);
        }

        // Check compute complexity
        if let Some(decision) = self.check_compute_complexity(&stats)? {
            return Ok(decision);
        }

        // Adaptive decision making
        if self.config.enable_adaptive {
            if let Some(decision) = self.adaptive_decision(&stats)? {
                return Ok(decision);
            }
        }

        // No checkpoint needed
        Ok(CheckpointDecision {
            should_checkpoint: false,
            strategy: self.config.strategy,
            reason: CheckpointReason::Interval,
            estimated_memory_savings: 0,
            estimated_compute_overhead: 0.0,
        })
    }

    /// Check for memory pressure conditions
    fn check_memory_pressure(&self, stats: &GraphStats) -> Result<Option<CheckpointDecision>> {
        // Check against configured memory limits
        if let Some(max_memory) = self.config.max_memory {
            if stats.memory_usage > max_memory {
                return Ok(Some(CheckpointDecision {
                    should_checkpoint: true,
                    strategy: CheckpointStrategy::MemoryFirst,
                    reason: CheckpointReason::MemoryPressure,
                    estimated_memory_savings: stats.memory_usage / 2, // Rough estimate
                    estimated_compute_overhead: 1.5,
                }));
            }
        }

        // Check system memory pressure
        if let Some(system_memory) = self.system_memory {
            let memory_usage_ratio = stats.memory_usage as f32 / system_memory as f32;
            if memory_usage_ratio > self.config.memory_pressure_threshold {
                return Ok(Some(CheckpointDecision {
                    should_checkpoint: true,
                    strategy: CheckpointStrategy::MemoryFirst,
                    reason: CheckpointReason::MemoryPressure,
                    estimated_memory_savings: (stats.memory_usage as f32 * 0.3) as usize,
                    estimated_compute_overhead: 1.2,
                }));
            }
        }

        Ok(None)
    }

    /// Check for compute complexity conditions
    fn check_compute_complexity(&self, stats: &GraphStats) -> Result<Option<CheckpointDecision>> {
        // Simple heuristic: if graph is getting very large, checkpoint to break it up
        if stats.node_count > 10000 {
            return Ok(Some(CheckpointDecision {
                should_checkpoint: true,
                strategy: CheckpointStrategy::ComputeFirst,
                reason: CheckpointReason::ComputeComplexity,
                estimated_memory_savings: stats.memory_usage / 4,
                estimated_compute_overhead: 1.1,
            }));
        }

        Ok(None)
    }

    /// Adaptive decision making based on historical data
    fn adaptive_decision(&self, _stats: &GraphStats) -> Result<Option<CheckpointDecision>> {
        if self.memory_history.len() < 10 {
            return Ok(None); // Need more history for adaptive decisions
        }

        // Analyze memory growth trend
        let recent_memory: Vec<usize> = self
            .memory_history
            .iter()
            .rev()
            .take(5)
            .map(|(_, mem)| *mem)
            .collect();

        if recent_memory.len() >= 2 {
            let memory_growth_rate = (recent_memory[0] as f32
                - recent_memory[recent_memory.len() - 1] as f32)
                / recent_memory.len() as f32;

            // If memory is growing rapidly, be more aggressive with checkpointing
            if memory_growth_rate > 1024.0 * 1024.0 {
                // 1MB per operation
                return Ok(Some(CheckpointDecision {
                    should_checkpoint: true,
                    strategy: CheckpointStrategy::Adaptive,
                    reason: CheckpointReason::Adaptive,
                    estimated_memory_savings: (memory_growth_rate * 5.0) as usize,
                    estimated_compute_overhead: 1.3,
                }));
            }
        }

        Ok(None)
    }

    /// Estimate memory savings from checkpointing
    fn estimate_memory_savings(&self, stats: &GraphStats) -> usize {
        // Rough estimate based on graph size and cache
        let graph_overhead = stats.node_count * 1024; // Assume 1KB per node
        let cache_overhead = stats.cache_size * 512; // Assume 512B per cache entry
        (graph_overhead + cache_overhead) / 2
    }

    /// Estimate compute overhead from checkpointing
    fn estimate_compute_overhead(&self, stats: &GraphStats) -> f32 {
        // Estimate based on graph complexity
        let base_overhead = 1.1; // 10% base overhead
        let complexity_factor = (stats.node_count as f32).log10() / 100.0;
        base_overhead + complexity_factor
    }

    /// Update memory and compute history
    fn update_history(&mut self, ctx: &AutogradContext) -> Result<()> {
        let now = Instant::now();
        let stats = ctx.graph_stats();

        // Update memory history
        self.memory_history.push_back((now, stats.memory_usage));
        if self.memory_history.len() > 1000 {
            self.memory_history.pop_front();
        }

        // Update memory pressure
        if let Some(system_memory) = self.system_memory {
            self.memory_pressure = stats.memory_usage as f32 / system_memory as f32;
        }

        Ok(())
    }

    /// Record a checkpoint event
    fn record_checkpoint(&mut self, decision: CheckpointDecision) -> Result<()> {
        let event = CheckpointEvent {
            timestamp: Instant::now(),
            operation_count: self.operation_count,
            memory_usage: self.memory_history.back().map(|(_, mem)| *mem).unwrap_or(0),
            strategy_used: decision.strategy,
            reason: decision.reason,
        };

        self.checkpoint_history.push(event);
        self.last_checkpoint = self.operation_count;

        debug!(
            "Checkpoint created: operation={}, memory={}, strategy={:?}, reason={:?}",
            self.operation_count,
            self.memory_history.back().map(|(_, mem)| *mem).unwrap_or(0),
            decision.strategy,
            decision.reason
        );

        Ok(())
    }

    /// Get checkpoint statistics
    pub fn get_stats(&self) -> CheckpointStats {
        let total_checkpoints = self.checkpoint_history.len();
        let avg_interval = if total_checkpoints > 1 {
            self.operation_count / total_checkpoints
        } else {
            0
        };

        let memory_savings: usize = self
            .checkpoint_history
            .iter()
            .map(|event| event.memory_usage / 2) // Rough estimate
            .sum();

        CheckpointStats {
            total_operations: self.operation_count,
            total_checkpoints,
            avg_checkpoint_interval: avg_interval,
            current_memory_pressure: self.memory_pressure,
            estimated_memory_savings: memory_savings,
            checkpoint_efficiency: if total_checkpoints > 0 {
                memory_savings as f32 / total_checkpoints as f32
            } else {
                0.0
            },
        }
    }

    /// Detect system memory
    fn detect_system_memory() -> Option<usize> {
        // Try to detect system memory on Linux
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(content) = fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return Some(kb * 1024); // Convert to bytes
                            }
                        }
                    }
                }
            }
        }

        // Fallback for other systems or if detection fails
        None
    }

    /// Force a checkpoint with manual reason
    pub fn force_checkpoint(&mut self) -> CheckpointDecision {
        let decision = CheckpointDecision {
            should_checkpoint: true,
            strategy: self.config.strategy,
            reason: CheckpointReason::Manual,
            estimated_memory_savings: 0,
            estimated_compute_overhead: 0.0,
        };

        if let Err(e) = self.record_checkpoint(decision.clone()) {
            warn!("Failed to record manual checkpoint: {}", e);
        }

        decision
    }

    /// Reset scheduler state
    pub fn reset(&mut self) {
        self.operation_count = 0;
        self.last_checkpoint = 0;
        self.memory_history.clear();
        self.compute_history.clear();
        self.checkpoint_history.clear();
        self.memory_pressure = 0.0;
    }
}

/// Statistics about checkpoint scheduling
#[derive(Debug, Clone)]
pub struct CheckpointStats {
    pub total_operations: usize,
    pub total_checkpoints: usize,
    pub avg_checkpoint_interval: usize,
    pub current_memory_pressure: f32,
    pub estimated_memory_savings: usize,
    pub checkpoint_efficiency: f32,
}

/// Checkpoint scheduler that integrates with AutogradContext
pub struct IntegratedCheckpointScheduler {
    scheduler: CheckpointScheduler,
    enabled: bool,
}

impl IntegratedCheckpointScheduler {
    /// Create new integrated scheduler
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            scheduler: CheckpointScheduler::new(config),
            enabled: true,
        }
    }

    /// Enable or disable the scheduler
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if enabled {
            info!("Checkpoint scheduler enabled");
        } else {
            info!("Checkpoint scheduler disabled");
        }
    }

    /// Process operation and handle checkpointing
    pub fn process_operation(&mut self, ctx: &mut AutogradContext) -> Result<bool> {
        if !self.enabled {
            return Ok(false);
        }

        let decision = self.scheduler.record_operation(ctx)?;

        if decision.should_checkpoint {
            // Trigger checkpoint creation in the context
            ctx.check_memory_pressure()?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> CheckpointStats {
        self.scheduler.get_stats()
    }

    /// Update scheduler configuration
    pub fn update_config(&mut self, config: CheckpointConfig) {
        self.scheduler.update_config(config);
    }

    /// Force checkpoint
    pub fn force_checkpoint(&mut self) -> CheckpointDecision {
        self.scheduler.force_checkpoint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::AutogradContext;

    #[test]
    fn test_checkpoint_scheduler_creation() {
        let config = CheckpointConfig::default();
        let scheduler = CheckpointScheduler::new(config);

        assert_eq!(scheduler.operation_count, 0);
        assert_eq!(scheduler.last_checkpoint, 0);
    }

    #[test]
    fn test_checkpoint_decision_interval() {
        let mut config = CheckpointConfig::default();
        config.max_checkpoint_interval = 5;

        let mut scheduler = CheckpointScheduler::new(config);
        let ctx = AutogradContext::new();

        // First few operations should not trigger checkpoint
        for _ in 0..4 {
            let decision = scheduler.record_operation(&ctx).unwrap();
            assert!(!decision.should_checkpoint);
        }

        // 5th operation should trigger checkpoint
        let decision = scheduler.record_operation(&ctx).unwrap();
        assert!(decision.should_checkpoint);
        assert_eq!(decision.reason, CheckpointReason::Interval);
    }

    #[test]
    fn test_checkpoint_stats() {
        let config = CheckpointConfig::default();
        let scheduler = CheckpointScheduler::new(config);

        let stats = scheduler.get_stats();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.total_checkpoints, 0);
    }

    #[test]
    fn test_integrated_scheduler() {
        let config = CheckpointConfig::default();
        let mut integrated = IntegratedCheckpointScheduler::new(config);
        let mut ctx = AutogradContext::new();

        // Should not checkpoint initially
        let checkpointed = integrated.process_operation(&mut ctx).unwrap();
        assert!(!checkpointed);

        // Disable and verify it doesn't checkpoint
        integrated.set_enabled(false);
        let checkpointed = integrated.process_operation(&mut ctx).unwrap();
        assert!(!checkpointed);
    }
}
