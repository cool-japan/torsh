//! Load balancing functionality for expert utilization
//!
//! This module implements dynamic load balancing strategies for Mixture of Experts (MoE) models,
//! including routing adjustments, expert migration, and capacity reallocation.

use super::config::ExpertParallelismConfig;
use super::router::RoutingDecision;
use crate::TorshResult;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Expert rebalancing strategies
///
/// Defines the different approaches available for rebalancing expert load
/// when imbalances are detected in the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RebalancingStrategy {
    /// Adjust routing probabilities to favor underloaded experts
    ///
    /// This is the least disruptive strategy that works by biasing the router
    /// to send more tokens to underutilized experts.
    RoutingAdjustment,

    /// Migrate experts between devices/nodes
    ///
    /// More aggressive strategy that physically moves expert parameters
    /// between devices to achieve better load distribution.
    ExpertMigration,

    /// Reallocate expert capacities based on load
    ///
    /// Dynamically adjusts the maximum capacity of experts based on
    /// observed load patterns and predictions.
    CapacityReallocation,

    /// Combine multiple rebalancing strategies
    ///
    /// Uses a combination of the above strategies for maximum effectiveness
    /// in severe load imbalance situations.
    HybridApproach,
}

impl RebalancingStrategy {
    /// Get a description of the strategy
    pub fn description(&self) -> &'static str {
        match self {
            Self::RoutingAdjustment => "Bias routing probabilities to favor underloaded experts",
            Self::ExpertMigration => "Migrate expert parameters between devices",
            Self::CapacityReallocation => "Dynamically adjust expert capacities",
            Self::HybridApproach => "Combine multiple rebalancing strategies",
        }
    }

    /// Get the disruption level of this strategy (0.0 = no disruption, 1.0 = high disruption)
    pub fn disruption_level(&self) -> f32 {
        match self {
            Self::RoutingAdjustment => 0.1,
            Self::CapacityReallocation => 0.3,
            Self::ExpertMigration => 0.7,
            Self::HybridApproach => 0.5,
        }
    }

    /// Check if this strategy supports gradual rollback
    pub fn supports_rollback(&self) -> bool {
        matches!(self, Self::RoutingAdjustment | Self::CapacityReallocation)
    }
}

/// Expert migration plan
///
/// Represents a planned migration of expert resources between devices or nodes
/// to achieve better load distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertMigration {
    /// Source expert ID
    pub source_expert_id: usize,
    /// Target expert ID or location
    pub target_expert_id: usize,
    /// Type of migration to perform
    pub migration_type: MigrationType,
    /// Priority of this migration (higher = more urgent)
    pub priority: f32,
    /// Estimated duration for completion
    pub estimated_duration: f32,
    /// Expected load reduction on source
    pub expected_load_reduction: f32,
    /// Expected load increase on target
    pub expected_load_increase: f32,
}

impl ExpertMigration {
    /// Create a new expert migration plan
    pub fn new(
        source_expert_id: usize,
        target_expert_id: usize,
        migration_type: MigrationType,
        priority: f32,
        estimated_duration: f32,
    ) -> Self {
        Self {
            source_expert_id,
            target_expert_id,
            migration_type,
            priority,
            estimated_duration,
            expected_load_reduction: 0.3, // Default estimate
            expected_load_increase: 0.3,
        }
    }

    /// Check if this migration is high priority
    pub fn is_high_priority(&self) -> bool {
        self.priority > 5.0
    }

    /// Get the expected net load balance improvement
    pub fn expected_improvement(&self) -> f32 {
        (self.expected_load_reduction + self.expected_load_increase) / 2.0
    }
}

/// Types of expert migration
///
/// Defines the specific mechanisms available for migrating expert resources.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationType {
    /// Redistribute load between experts
    ///
    /// Adjusts routing to shift load without moving parameters.
    LoadRedistribution,

    /// Move expert to different device
    ///
    /// Physically transfers expert parameters to another device.
    DeviceMigration,

    /// Replicate expert for load balancing
    ///
    /// Creates copies of high-demand experts on multiple devices.
    ExpertReplication,

    /// Merge underutilized experts
    ///
    /// Combines multiple underutilized experts to improve efficiency.
    ExpertConsolidation,
}

impl MigrationType {
    /// Get the complexity of this migration type
    pub fn complexity(&self) -> u32 {
        match self {
            Self::LoadRedistribution => 1,
            Self::ExpertReplication => 2,
            Self::DeviceMigration => 3,
            Self::ExpertConsolidation => 4,
        }
    }

    /// Check if this migration type requires parameter copying
    pub fn requires_parameter_copy(&self) -> bool {
        matches!(self, Self::DeviceMigration | Self::ExpertReplication)
    }
}

/// Load balancer for expert utilization
///
/// Monitors expert load patterns and implements various strategies to maintain
/// balanced utilization across all experts in the system.
#[derive(Debug)]
pub struct LoadBalancer {
    config: ExpertParallelismConfig,
    expert_loads: Vec<f32>,
    load_history: VecDeque<Vec<f32>>,
    rebalancing_threshold: f32,
    routing_adjustments: Vec<f32>,
    routing_adjustment_decay: f32,
    expert_capacities: Vec<f32>,
    pending_migrations: VecDeque<ExpertMigration>,
    rebalancing_count: u64,
    last_rebalancing_step: u64,
    load_variance_history: VecDeque<f32>,
    strategy_effectiveness: std::collections::HashMap<RebalancingStrategy, f32>,
}

impl LoadBalancer {
    /// Create a new load balancer
    ///
    /// # Arguments
    ///
    /// * `config` - Expert parallelism configuration
    ///
    /// # Returns
    ///
    /// A new LoadBalancer instance
    pub fn new(config: &ExpertParallelismConfig) -> Self {
        Self {
            config: config.clone(),
            expert_loads: vec![0.0; config.num_experts],
            load_history: VecDeque::with_capacity(100),
            rebalancing_threshold: 0.2, // 20% load imbalance triggers rebalancing
            routing_adjustments: vec![1.0; config.num_experts],
            routing_adjustment_decay: 0.95,
            expert_capacities: vec![100.0; config.num_experts],
            pending_migrations: VecDeque::new(),
            rebalancing_count: 0,
            last_rebalancing_step: 0,
            load_variance_history: VecDeque::with_capacity(50),
            strategy_effectiveness: std::collections::HashMap::new(),
        }
    }

    /// Update expert load statistics based on routing decisions
    ///
    /// # Arguments
    ///
    /// * `routing_decision` - Latest routing decision with expert assignments
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn update_expert_load(&mut self, routing_decision: &RoutingDecision) -> TorshResult<()> {
        let alpha = 0.1; // Smoothing factor for exponential moving average
        let total_tokens = routing_decision.total_tokens as f32;

        if total_tokens == 0.0 {
            return Ok(());
        }

        // Update exponential moving average of expert loads
        for (expert_id, &current_capacity) in routing_decision.expert_capacities.iter().enumerate()
        {
            let current_load = current_capacity as f32 / total_tokens;
            if expert_id < self.expert_loads.len() {
                self.expert_loads[expert_id] =
                    alpha * current_load + (1.0 - alpha) * self.expert_loads[expert_id];
            }
        }

        // Store load history for trend analysis
        self.load_history.push_back(self.expert_loads.clone());
        if self.load_history.len() > 100 {
            self.load_history.pop_front();
        }

        // Track load variance over time
        let load_variance = self.calculate_load_variance();
        self.load_variance_history.push_back(load_variance);
        if self.load_variance_history.len() > 50 {
            self.load_variance_history.pop_front();
        }

        // Check if rebalancing is needed
        if self.should_rebalance() {
            self.trigger_rebalancing()?;
        }

        self.last_rebalancing_step += 1;

        Ok(())
    }

    /// Determine if rebalancing should be triggered
    fn should_rebalance(&self) -> bool {
        if self.expert_loads.is_empty() {
            return false;
        }

        // Avoid too frequent rebalancing
        if self.last_rebalancing_step < 10 {
            return false;
        }

        let mean_load: f32 = self.expert_loads.iter().sum::<f32>() / self.expert_loads.len() as f32;
        let max_load = self
            .expert_loads
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_load = self
            .expert_loads
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);

        if mean_load <= 0.0 {
            return false;
        }

        let load_imbalance = (max_load - min_load) / mean_load;
        load_imbalance > self.rebalancing_threshold
    }

    /// Trigger rebalancing using appropriate strategy
    fn trigger_rebalancing(&mut self) -> TorshResult<()> {
        info!(" Expert Load Rebalancing: Load imbalance detected, triggering rebalancing");

        let strategy = self.get_rebalancing_strategy();
        let start_variance = self.calculate_load_variance();

        match strategy {
            RebalancingStrategy::RoutingAdjustment => {
                self.apply_routing_adjustment()?;
            }
            RebalancingStrategy::ExpertMigration => {
                self.apply_expert_migration()?;
            }
            RebalancingStrategy::CapacityReallocation => {
                self.apply_capacity_reallocation()?;
            }
            RebalancingStrategy::HybridApproach => {
                self.apply_routing_adjustment()?;
                self.apply_capacity_reallocation()?;
            }
        }

        // Track strategy effectiveness for future decisions
        self.update_strategy_effectiveness(strategy, start_variance);

        self.rebalancing_count += 1;
        self.last_rebalancing_step = 0;

        Ok(())
    }

    /// Determine the best rebalancing strategy for current conditions
    fn get_rebalancing_strategy(&self) -> RebalancingStrategy {
        let mean_load: f32 = self.expert_loads.iter().sum::<f32>() / self.expert_loads.len() as f32;
        let max_load = self
            .expert_loads
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_load = self
            .expert_loads
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);

        let load_variance = self.calculate_load_variance();
        let imbalance_severity = if mean_load > 0.0 {
            (max_load - min_load) / mean_load
        } else {
            0.0
        };

        // Consider historical effectiveness of strategies
        let best_strategy = self
            .strategy_effectiveness
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(strategy, _)| *strategy);

        // Use historical data if available, otherwise use heuristics
        if let Some(strategy) = best_strategy {
            if self.strategy_effectiveness[&strategy] > 0.3 {
                return strategy;
            }
        }

        // Fallback to heuristic-based strategy selection
        if imbalance_severity > 0.5 && load_variance > 0.1 {
            RebalancingStrategy::HybridApproach
        } else if imbalance_severity > 0.3 {
            RebalancingStrategy::ExpertMigration
        } else if load_variance > 0.05 {
            RebalancingStrategy::CapacityReallocation
        } else {
            RebalancingStrategy::RoutingAdjustment
        }
    }

    /// Apply routing probability adjustments
    fn apply_routing_adjustment(&mut self) -> TorshResult<()> {
        let mean_load: f32 = self.expert_loads.iter().sum::<f32>() / self.expert_loads.len() as f32;

        for (expert_id, &load) in self.expert_loads.iter().enumerate() {
            let load_ratio = if mean_load > 0.0 {
                load / mean_load
            } else {
                1.0
            };

            // Calculate routing bias to favor underloaded experts
            let routing_bias = if load_ratio < 0.8 {
                1.2 // Boost underloaded experts
            } else if load_ratio > 1.2 {
                0.8 // Penalize overloaded experts
            } else {
                1.0 // Neutral
            };

            if expert_id < self.routing_adjustments.len() {
                self.routing_adjustments[expert_id] = routing_bias;
            }

            info!(
                " Routing adjustment for expert {}: load={:.3}, bias={:.2}",
                expert_id, load, routing_bias
            );
        }

        // Schedule gradual restoration of normal routing
        self.routing_adjustment_decay = 0.95;

        Ok(())
    }

    /// Apply expert migration strategy
    fn apply_expert_migration(&mut self) -> TorshResult<()> {
        let (overloaded_experts, underloaded_experts) = self.identify_migration_candidates();

        if overloaded_experts.is_empty() || underloaded_experts.is_empty() {
            return Ok(());
        }

        let migration_plan = self.plan_expert_migrations(&overloaded_experts, &underloaded_experts);

        for migration in migration_plan {
            self.schedule_expert_migration(migration)?;
        }

        info!(
            " Scheduled {} expert migrations for load balancing",
            self.pending_migrations.len()
        );

        Ok(())
    }

    /// Apply capacity reallocation strategy
    fn apply_capacity_reallocation(&mut self) -> TorshResult<()> {
        let total_capacity: f32 = self.expert_capacities.iter().sum::<f32>();
        let target_capacity_per_expert = total_capacity / self.expert_loads.len() as f32;

        for (expert_id, &current_load) in self.expert_loads.iter().enumerate() {
            let load_trend = self.calculate_load_trend(expert_id);
            let predicted_load = current_load + 0.3 * load_trend;

            let new_capacity = if predicted_load > target_capacity_per_expert * 1.1 {
                (target_capacity_per_expert * 1.25).min(total_capacity * 0.3)
            } else if predicted_load < target_capacity_per_expert * 0.8 {
                (target_capacity_per_expert * 0.75).max(target_capacity_per_expert * 0.5)
            } else {
                target_capacity_per_expert
            };

            if expert_id < self.expert_capacities.len() {
                self.expert_capacities[expert_id] = new_capacity;
            }

            info!(
                "  Capacity reallocation for expert {}: {:.1} -> {:.1} (trend: {:.3})",
                expert_id, target_capacity_per_expert, new_capacity, load_trend
            );
        }

        Ok(())
    }

    /// Identify candidates for expert migration
    fn identify_migration_candidates(&self) -> (Vec<usize>, Vec<usize>) {
        let mean_load: f32 = self.expert_loads.iter().sum::<f32>() / self.expert_loads.len() as f32;
        let mut overloaded = Vec::new();
        let mut underloaded = Vec::new();

        for (expert_id, &load) in self.expert_loads.iter().enumerate() {
            if load > mean_load * 1.3 {
                overloaded.push(expert_id);
            } else if load < mean_load * 0.7 {
                underloaded.push(expert_id);
            }
        }

        (overloaded, underloaded)
    }

    /// Plan expert migrations between overloaded and underloaded experts
    fn plan_expert_migrations(
        &self,
        overloaded: &[usize],
        underloaded: &[usize],
    ) -> Vec<ExpertMigration> {
        let mut migrations = Vec::new();
        let max_migrations = (overloaded.len().min(underloaded.len())).min(2); // Limit concurrent migrations

        for i in 0..max_migrations {
            let source_expert = overloaded[i];
            let target_expert = underloaded[i % underloaded.len()];

            migrations.push(ExpertMigration::new(
                source_expert,
                target_expert,
                MigrationType::LoadRedistribution,
                self.calculate_migration_priority(source_expert, target_expert),
                self.estimate_migration_duration(source_expert),
            ));
        }

        migrations.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        migrations
    }

    /// Schedule an expert migration for execution
    fn schedule_expert_migration(&mut self, migration: ExpertMigration) -> TorshResult<()> {
        self.pending_migrations.push_back(migration.clone());

        info!(
            " Scheduled migration: Expert {} -> Expert {} (priority: {:.2})",
            migration.source_expert_id, migration.target_expert_id, migration.priority
        );

        Ok(())
    }

    /// Calculate load variance across all experts
    fn calculate_load_variance(&self) -> f32 {
        if self.expert_loads.is_empty() {
            return 0.0;
        }

        let mean: f32 = self.expert_loads.iter().sum::<f32>() / self.expert_loads.len() as f32;
        let variance: f32 = self
            .expert_loads
            .iter()
            .map(|&load| (load - mean).powi(2))
            .sum::<f32>()
            / self.expert_loads.len() as f32;

        variance
    }

    /// Calculate load trend for a specific expert
    fn calculate_load_trend(&self, expert_id: usize) -> f32 {
        if self.load_history.len() < 2 {
            return 0.0;
        }

        let recent_loads: Vec<f32> = self
            .load_history
            .iter()
            .rev()
            .take(5)
            .map(|loads| loads.get(expert_id).copied().unwrap_or(0.0))
            .collect();

        if recent_loads.len() < 2 {
            return 0.0;
        }

        // Simple linear regression for trend calculation
        let n = recent_loads.len() as f32;
        let sum_x: f32 = (0..recent_loads.len()).map(|i| i as f32).sum();
        let sum_y: f32 = recent_loads.iter().sum();
        let sum_xy: f32 = recent_loads
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f32 * y)
            .sum();
        let sum_x2: f32 = (0..recent_loads.len()).map(|i| (i as f32).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < f32::EPSILON {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }

    /// Calculate migration priority based on load difference
    fn calculate_migration_priority(&self, source_expert: usize, target_expert: usize) -> f32 {
        let source_load = self.expert_loads.get(source_expert).copied().unwrap_or(0.0);
        let target_load = self.expert_loads.get(target_expert).copied().unwrap_or(0.0);
        let load_difference = source_load - target_load;

        load_difference * 10.0
    }

    /// Estimate migration duration based on expert characteristics
    fn estimate_migration_duration(&self, expert_id: usize) -> f32 {
        let base_duration = 5.0; // seconds
        let load_factor = self.expert_loads.get(expert_id).copied().unwrap_or(1.0);

        base_duration * (1.0 + load_factor)
    }

    /// Update strategy effectiveness tracking
    fn update_strategy_effectiveness(
        &mut self,
        strategy: RebalancingStrategy,
        start_variance: f32,
    ) {
        let end_variance = self.calculate_load_variance();
        let improvement = (start_variance - end_variance) / start_variance.max(f32::EPSILON);

        self.strategy_effectiveness
            .entry(strategy)
            .and_modify(|e| *e = 0.9 * *e + 0.1 * improvement)
            .or_insert(improvement);
    }

    /// Get current routing adjustment for an expert
    pub fn get_routing_adjustment(&self, expert_id: usize) -> f32 {
        self.routing_adjustments
            .get(expert_id)
            .copied()
            .unwrap_or(1.0)
    }

    /// Apply decay to routing adjustments (called periodically)
    pub fn decay_routing_adjustments(&mut self) {
        for adjustment in self.routing_adjustments.iter_mut() {
            *adjustment =
                *adjustment * self.routing_adjustment_decay + (1.0 - self.routing_adjustment_decay);
        }
    }

    /// Process pending expert migrations
    pub fn process_pending_migrations(&mut self) -> TorshResult<()> {
        if let Some(migration) = self.pending_migrations.pop_front() {
            self.execute_expert_migration(migration)?;
        }
        Ok(())
    }

    /// Execute an expert migration
    fn execute_expert_migration(&mut self, migration: ExpertMigration) -> TorshResult<()> {
        info!(
            " Executing migration: Expert {} -> Expert {} (type: {:?})",
            migration.source_expert_id, migration.target_expert_id, migration.migration_type
        );

        // Update local load estimates after migration
        let load_transfer = migration.expected_load_reduction;
        if let Some(source_load) = self.expert_loads.get_mut(migration.source_expert_id) {
            *source_load = (*source_load * (1.0 - load_transfer)).max(0.0);
        }
        if let Some(target_load) = self.expert_loads.get_mut(migration.target_expert_id) {
            *target_load += load_transfer * migration.expected_load_increase;
        }

        Ok(())
    }

    /// Get load balancing statistics
    pub fn get_stats(&self) -> LoadBalancingStats {
        LoadBalancingStats {
            expert_loads: self.expert_loads.clone(),
            load_variance: self.calculate_load_variance(),
            rebalancing_count: self.rebalancing_count,
            pending_migrations: self.pending_migrations.len(),
            routing_adjustments: self.routing_adjustments.clone(),
            strategy_effectiveness: self.strategy_effectiveness.clone(),
        }
    }

    /// Reset load balancer state
    pub fn reset(&mut self) {
        self.expert_loads.fill(0.0);
        self.load_history.clear();
        self.routing_adjustments.fill(1.0);
        self.pending_migrations.clear();
        self.rebalancing_count = 0;
        self.last_rebalancing_step = 0;
        self.load_variance_history.clear();
        self.strategy_effectiveness.clear();
    }
}

/// Load balancing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingStats {
    /// Current load for each expert
    pub expert_loads: Vec<f32>,
    /// Current load variance across experts
    pub load_variance: f32,
    /// Total number of rebalancing operations performed
    pub rebalancing_count: u64,
    /// Number of pending migrations
    pub pending_migrations: usize,
    /// Current routing adjustments for each expert
    pub routing_adjustments: Vec<f32>,
    /// Effectiveness of different rebalancing strategies
    pub strategy_effectiveness: std::collections::HashMap<RebalancingStrategy, f32>,
}

impl LoadBalancingStats {
    /// Get the coefficient of variation for load distribution
    pub fn load_cv(&self) -> f32 {
        if self.expert_loads.is_empty() {
            return 0.0;
        }

        let mean = self.expert_loads.iter().sum::<f32>() / self.expert_loads.len() as f32;
        if mean > 0.0 {
            self.load_variance.sqrt() / mean
        } else {
            0.0
        }
    }

    /// Get the most effective rebalancing strategy
    pub fn best_strategy(&self) -> Option<RebalancingStrategy> {
        self.strategy_effectiveness
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(strategy, _)| *strategy)
    }

    /// Check if load is well balanced (CV < 0.2)
    pub fn is_well_balanced(&self) -> bool {
        self.load_cv() < 0.2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_parallelism::config::ExpertParallelismConfig;

    #[test]
    fn test_load_balancer_creation() {
        let config = ExpertParallelismConfig::default();
        let load_balancer = LoadBalancer::new(&config);

        assert_eq!(load_balancer.expert_loads.len(), config.num_experts);
        assert_eq!(load_balancer.routing_adjustments.len(), config.num_experts);
    }

    #[test]
    fn test_rebalancing_strategy_properties() {
        assert_eq!(
            RebalancingStrategy::RoutingAdjustment.disruption_level(),
            0.1
        );
        assert!(RebalancingStrategy::RoutingAdjustment.supports_rollback());
        assert!(!RebalancingStrategy::ExpertMigration.supports_rollback());
    }

    #[test]
    fn test_migration_type_complexity() {
        assert_eq!(MigrationType::LoadRedistribution.complexity(), 1);
        assert_eq!(MigrationType::ExpertConsolidation.complexity(), 4);
        assert!(MigrationType::DeviceMigration.requires_parameter_copy());
        assert!(!MigrationType::LoadRedistribution.requires_parameter_copy());
    }

    #[test]
    fn test_expert_migration() {
        let migration = ExpertMigration::new(0, 1, MigrationType::LoadRedistribution, 5.5, 10.0);
        assert_eq!(migration.source_expert_id, 0);
        assert_eq!(migration.target_expert_id, 1);
        assert!(migration.is_high_priority());
        assert!(migration.expected_improvement() > 0.0);
    }

    #[test]
    fn test_load_variance_calculation() {
        let config = ExpertParallelismConfig::default();
        let mut load_balancer = LoadBalancer::new(&config);

        // Set some test loads
        load_balancer.expert_loads = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let variance = load_balancer.calculate_load_variance();
        assert!(variance > 0.0);
    }

    #[test]
    fn test_migration_candidates() {
        let config = ExpertParallelismConfig::default();
        let mut load_balancer = LoadBalancer::new(&config);

        // Set imbalanced loads
        load_balancer.expert_loads = vec![0.05, 0.1, 0.6, 0.8, 0.05, 0.1, 0.05, 0.1]; // Mean ≈ 0.24

        let (overloaded, underloaded) = load_balancer.identify_migration_candidates();
        assert!(!overloaded.is_empty());
        assert!(!underloaded.is_empty());
    }

    #[test]
    fn test_load_balancing_stats() {
        let stats = LoadBalancingStats {
            expert_loads: vec![0.2, 0.2, 0.2, 0.2, 0.2], // Perfectly balanced
            load_variance: 0.0,
            rebalancing_count: 5,
            pending_migrations: 2,
            routing_adjustments: vec![1.0; 5],
            strategy_effectiveness: std::collections::HashMap::new(),
        };

        assert_eq!(stats.load_cv(), 0.0);
        assert!(stats.is_well_balanced());
    }
}
