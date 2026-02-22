//! Expert routing functionality for token-to-expert assignment
//!
//! This module implements the core routing logic for Mixture of Experts (MoE) models,
//! including hierarchical gating networks, capacity constraints, and load balancing.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use super::config::ExpertParallelismConfig;
use super::load_balancer::LoadBalancer;
use super::stats::RoutingStats;
use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use torsh_core::device::DeviceType;
use torsh_tensor::{creation::randn, Tensor};

/// Expert assignment for a single token
///
/// Represents the assignment of a single token to a specific expert,
/// including routing probability and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertAssignment {
    /// ID of the assigned expert
    pub expert_id: usize,
    /// Routing probability (softmax score)
    pub probability: f32,
    /// Index of the token being routed
    pub token_idx: usize,
    /// Rank among selected experts (0 = highest probability)
    pub expert_rank: usize,
}

impl ExpertAssignment {
    /// Create a new expert assignment
    pub fn new(expert_id: usize, probability: f32, token_idx: usize, expert_rank: usize) -> Self {
        Self {
            expert_id,
            probability,
            token_idx,
            expert_rank,
        }
    }

    /// Check if this assignment is valid
    pub fn is_valid(&self) -> bool {
        self.probability >= 0.0 && self.probability <= 1.0
    }

    /// Get the weighted contribution of this assignment
    pub fn weighted_contribution(&self) -> f32 {
        self.probability / (self.expert_rank as f32 + 1.0)
    }
}

/// Complete routing decision for a batch
///
/// Contains all information about how tokens in a batch are routed to experts,
/// including capacity utilization, dropped tokens, and auxiliary losses.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Expert assignments for each token in the batch
    pub expert_assignments: Vec<Vec<ExpertAssignment>>,
    /// Current capacity utilization for each expert
    pub expert_capacities: Vec<usize>,
    /// Total number of tokens in the batch
    pub total_tokens: usize,
    /// Number of tokens that couldn't be assigned due to capacity constraints
    pub tokens_dropped: usize,
    /// Load balancing auxiliary loss
    pub load_balance_loss: f32,
    /// Router z-loss for numerical stability
    pub router_z_loss: f32,
    /// Combined auxiliary loss
    pub auxiliary_loss: f32,
}

impl RoutingDecision {
    /// Create a new routing decision
    pub fn new(
        expert_assignments: Vec<Vec<ExpertAssignment>>,
        expert_capacities: Vec<usize>,
        total_tokens: usize,
        load_balance_loss: f32,
        router_z_loss: f32,
        auxiliary_loss: f32,
    ) -> Self {
        let tokens_dropped = total_tokens.saturating_sub(
            expert_assignments
                .iter()
                .map(|assignments| assignments.len())
                .sum(),
        );

        Self {
            expert_assignments,
            expert_capacities,
            total_tokens,
            tokens_dropped,
            load_balance_loss,
            router_z_loss,
            auxiliary_loss,
        }
    }

    /// Get the routing efficiency (tokens successfully routed / total tokens)
    pub fn routing_efficiency(&self) -> f32 {
        if self.total_tokens == 0 {
            1.0
        } else {
            (self.total_tokens - self.tokens_dropped) as f32 / self.total_tokens as f32
        }
    }

    /// Get the load balance coefficient of variation
    pub fn load_balance_cv(&self) -> f32 {
        if self.expert_capacities.is_empty() {
            0.0
        } else {
            let mean = self.expert_capacities.iter().sum::<usize>() as f32
                / self.expert_capacities.len() as f32;
            let variance = self
                .expert_capacities
                .iter()
                .map(|&cap| {
                    let diff = cap as f32 - mean;
                    diff * diff
                })
                .sum::<f32>()
                / self.expert_capacities.len() as f32;

            if mean > 0.0 {
                variance.sqrt() / mean
            } else {
                0.0
            }
        }
    }

    /// Get expert utilization statistics
    pub fn expert_utilization(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        let total_capacity: usize = self.expert_capacities.iter().sum();

        if total_capacity > 0 {
            stats.insert(
                "min_utilization".to_string(),
                *self.expert_capacities.iter().min().unwrap_or(&0) as f32 / total_capacity as f32,
            );
            stats.insert(
                "max_utilization".to_string(),
                *self.expert_capacities.iter().max().unwrap_or(&0) as f32 / total_capacity as f32,
            );
            stats.insert(
                "mean_utilization".to_string(),
                total_capacity as f32 / (self.expert_capacities.len() * total_capacity) as f32,
            );
        }

        stats.insert(
            "tokens_dropped_rate".to_string(),
            self.tokens_dropped as f32 / self.total_tokens as f32,
        );
        stats.insert("routing_efficiency".to_string(), self.routing_efficiency());

        stats
    }
}

/// Expert router for token-to-expert assignment
///
/// The core component responsible for routing tokens to appropriate experts
/// based on learned router weights and capacity constraints.
pub struct ExpertRouter {
    config: ExpertParallelismConfig,
    router_weights: Tensor<f32>,
    gate_network: Option<GateNetwork>,
    load_balancer: LoadBalancer,
    routing_stats: Arc<Mutex<RoutingStats>>,
}

impl ExpertRouter {
    /// Create a new expert router
    ///
    /// # Arguments
    ///
    /// * `config` - Expert parallelism configuration
    /// * `input_dim` - Dimension of input tokens
    /// * `device_id` - Device ID for computation
    ///
    /// # Returns
    ///
    /// A new ExpertRouter instance
    pub fn new(
        config: ExpertParallelismConfig,
        input_dim: usize,
        device_id: i32,
    ) -> TorshResult<Self> {
        // Initialize router weights (input_dim x num_experts)
        let router_weights = randn(&[input_dim, config.num_experts])?;

        let gate_network = if config.num_experts > 32 {
            // Use hierarchical gating for large number of experts
            Some(GateNetwork::new(input_dim, config.num_experts, device_id)?)
        } else {
            None
        };

        let load_balancer = LoadBalancer::new(&config);

        Ok(Self {
            config,
            router_weights,
            gate_network,
            load_balancer,
            routing_stats: Arc::new(Mutex::new(RoutingStats::new())),
        })
    }

    /// Route tokens to experts and return routing decisions
    ///
    /// # Arguments
    ///
    /// * `input_tokens` - Input token tensor [batch_size, seq_len, input_dim]
    /// * `training` - Whether in training mode (affects dropout and load balancing)
    ///
    /// # Returns
    ///
    /// Routing decision containing expert assignments and statistics
    pub fn route_tokens(
        &mut self,
        input_tokens: &Tensor<f32>,
        training: bool,
    ) -> TorshResult<RoutingDecision> {
        let batch_size = input_tokens.shape().dims()[0];
        let seq_len = input_tokens.shape().dims()[1];
        let total_tokens = batch_size * seq_len;

        // Compute router logits
        let router_logits = if let Some(ref gate_network) = self.gate_network {
            gate_network.forward(input_tokens)?
        } else {
            // Simple linear routing
            input_tokens.matmul(&self.router_weights)?
        };

        // Apply softmax to get probabilities
        let router_probs = router_logits.softmax(-1)?;

        // Select top-k experts per token
        let (top_expert_indices, top_expert_probs) = self.select_top_k_experts(&router_probs)?;

        // Apply capacity constraints and load balancing
        let routing_decision = self.apply_capacity_constraints(
            &top_expert_indices,
            &top_expert_probs,
            total_tokens,
            training,
        )?;

        // Update load balancing statistics
        if training && self.config.enable_load_balancing {
            self.load_balancer.update_expert_load(&routing_decision)?;
        }

        // Record routing statistics
        {
            let mut stats = self
                .routing_stats
                .lock()
                .expect("lock should not be poisoned");
            stats.record_routing(&routing_decision);
        }

        Ok(routing_decision)
    }

    /// Select top-k experts for each token
    fn select_top_k_experts(
        &self,
        router_probs: &Tensor<f32>,
    ) -> TorshResult<(Tensor<i32>, Tensor<f32>)> {
        let k = self.config.num_experts_per_token;
        let shape = router_probs.shape();
        let batch_tokens = shape.dims()[0] * shape.dims()[1];
        let num_experts = shape.dims()[2];

        let prob_data =
            router_probs
                .data()
                .map_err(|_| TorshDistributedError::InvalidArgument {
                    arg: "router_probs".to_string(),
                    reason: "Failed to access tensor data".to_string(),
                    expected: "Valid f32 tensor data".to_string(),
                })?;

        let mut top_indices_data = Vec::with_capacity(batch_tokens * k);
        let mut top_probs_data = Vec::with_capacity(batch_tokens * k);

        // Process each token
        for token_idx in 0..batch_tokens {
            let start_idx = token_idx * num_experts;
            let end_idx = start_idx + num_experts;
            let token_probs = &prob_data[start_idx..end_idx];

            // Create probability-index pairs and sort
            let mut prob_indices: Vec<(f32, i32)> = token_probs
                .iter()
                .enumerate()
                .map(|(idx, &prob)| (prob, idx as i32))
                .collect();

            prob_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

            // Take top-k experts
            for &(prob, index) in prob_indices.iter().take(k) {
                top_indices_data.push(index);
                top_probs_data.push(prob);
            }

            // Fill remaining slots if k > num_experts
            for _ in prob_indices.len()..k {
                top_indices_data.push(0);
                top_probs_data.push(0.0);
            }
        }

        let top_indices =
            Tensor::from_data(top_indices_data, vec![batch_tokens, k], DeviceType::Cpu)?;
        let top_probs = Tensor::from_data(top_probs_data, vec![batch_tokens, k], DeviceType::Cpu)?;

        Ok((top_indices, top_probs))
    }

    /// Apply capacity constraints and create routing decision
    fn apply_capacity_constraints(
        &mut self,
        expert_indices: &Tensor<i32>,
        expert_probs: &Tensor<f32>,
        total_tokens: usize,
        training: bool,
    ) -> TorshResult<RoutingDecision> {
        let capacity_per_expert = self.config.calculate_expert_capacity(total_tokens);

        let mut expert_assignments = Vec::new();
        let mut expert_capacities = vec![0usize; self.config.num_experts];
        let mut load_balance_loss = 0.0f32;
        let mut router_z_loss = 0.0f32;

        let batch_tokens = expert_indices.shape().dims()[0];
        let k = expert_indices.shape().dims()[1];

        for token_idx in 0..batch_tokens {
            let mut token_assignments = Vec::new();

            for expert_rank in 0..k {
                let indices_data = expert_indices.to_vec()?;
                let probs_data = expert_probs.to_vec()?;

                let expert_id = indices_data[token_idx * k + expert_rank] as usize;
                let prob = probs_data[token_idx * k + expert_rank];

                if expert_capacities[expert_id] < capacity_per_expert {
                    expert_capacities[expert_id] += 1;

                    token_assignments.push(ExpertAssignment::new(
                        expert_id,
                        prob,
                        token_idx,
                        expert_rank,
                    ));
                } else if training && self.config.expert_dropout > 0.0 {
                    // Apply expert dropout and try fallback assignment
                    if (token_idx as f32 * 0.1) % 1.0 < self.config.expert_dropout {
                        let alternative_expert = self.find_least_loaded_expert(&expert_capacities);
                        if expert_capacities[alternative_expert] < capacity_per_expert {
                            expert_capacities[alternative_expert] += 1;

                            token_assignments.push(ExpertAssignment::new(
                                alternative_expert,
                                prob * 0.5, // Reduced probability for fallback
                                token_idx,
                                expert_rank,
                            ));
                        }
                    }
                }
            }

            expert_assignments.push(token_assignments);
        }

        // Calculate auxiliary losses
        if training {
            load_balance_loss = self.calculate_load_balance_loss(&expert_capacities, total_tokens);
            router_z_loss = self.calculate_router_z_loss(expert_probs)?;
        }

        let auxiliary_loss = load_balance_loss * self.config.load_balance_loss_coeff
            + router_z_loss * self.config.router_z_loss_coeff;

        Ok(RoutingDecision::new(
            expert_assignments,
            expert_capacities,
            total_tokens,
            load_balance_loss,
            router_z_loss,
            auxiliary_loss,
        ))
    }

    /// Find the expert with the least current load
    fn find_least_loaded_expert(&self, capacities: &[usize]) -> usize {
        capacities
            .iter()
            .enumerate()
            .min_by_key(|(_, &capacity)| capacity)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Calculate load balance loss (coefficient of variation)
    fn calculate_load_balance_loss(&self, capacities: &[usize], total_tokens: usize) -> f32 {
        let mean_load = total_tokens as f32 / self.config.num_experts as f32;
        let variance: f32 = capacities
            .iter()
            .map(|&capacity| {
                let diff = capacity as f32 - mean_load;
                diff * diff
            })
            .sum::<f32>()
            / self.config.num_experts as f32;

        if mean_load > 0.0 {
            variance.sqrt() / mean_load
        } else {
            0.0
        }
    }

    /// Calculate router z-loss for numerical stability
    fn calculate_router_z_loss(&self, expert_probs: &Tensor<f32>) -> TorshResult<f32> {
        let probs_data = expert_probs.to_vec()?;
        let z_loss =
            probs_data.iter().map(|&prob| prob * prob).sum::<f32>() / probs_data.len() as f32;
        Ok(z_loss)
    }

    /// Get routing statistics
    pub fn get_stats(&self) -> RoutingStats {
        self.routing_stats
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Reset routing statistics
    pub fn reset_stats(&self) {
        let mut stats = self
            .routing_stats
            .lock()
            .expect("lock should not be poisoned");
        *stats = RoutingStats::new();
    }

    /// Get current load balancer state
    pub fn get_load_balancer(&self) -> &LoadBalancer {
        &self.load_balancer
    }

    /// Update router weights for fine-tuning
    pub fn update_router_weights(&mut self, new_weights: Tensor<f32>) -> TorshResult<()> {
        if new_weights.shape().dims() == self.router_weights.shape().dims() {
            self.router_weights = new_weights;
            Ok(())
        } else {
            Err(TorshDistributedError::InvalidArgument {
                arg: "new_weights".to_string(),
                reason: "Shape mismatch with existing router weights".to_string(),
                expected: format!("{:?}", self.router_weights.shape().dims()),
            })
        }
    }

    /// Get the number of experts in this router
    pub fn get_num_experts(&self) -> usize {
        self.config.num_experts
    }
}

/// Hierarchical gate network for large numbers of experts
///
/// When dealing with hundreds or thousands of experts, a flat routing approach
/// becomes computationally expensive. This hierarchical approach first routes
/// to expert groups, then to experts within groups.
pub struct GateNetwork {
    input_dim: usize,
    num_experts: usize,
    device_id: i32,
    group_router: Tensor<f32>,
    expert_routers: Vec<Tensor<f32>>,
    num_groups: usize,
    experts_per_group: usize,
}

impl GateNetwork {
    /// Create a new hierarchical gate network
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Dimension of input tokens
    /// * `num_experts` - Total number of experts
    /// * `device_id` - Device ID for computation
    ///
    /// # Returns
    ///
    /// A new GateNetwork instance
    pub fn new(input_dim: usize, num_experts: usize, device_id: i32) -> TorshResult<Self> {
        // Organize experts into groups for hierarchical routing
        let num_groups = (num_experts as f32).sqrt().ceil() as usize;
        let experts_per_group = num_experts.div_ceil(num_groups);

        let group_router = randn(&[input_dim, num_groups])?;
        let expert_routers: Vec<_> = (0..num_groups)
            .map(|_| randn(&[input_dim, experts_per_group]))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            input_dim,
            num_experts,
            device_id,
            group_router,
            expert_routers,
            num_groups,
            experts_per_group,
        })
    }

    /// Forward pass through the hierarchical gate network
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [batch_size, seq_len, input_dim]
    ///
    /// # Returns
    ///
    /// Expert routing probabilities [batch_size, seq_len, num_experts]
    pub fn forward(&self, input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
        // Stage 1: Route to expert groups
        let group_logits = input.matmul(&self.group_router)?;
        let group_probs = group_logits.softmax(-1)?;

        // Stage 2: Route to experts within selected groups
        let batch_size = input.shape().dims()[0];
        let seq_len = input.shape().dims()[1];

        let group_probs_data = group_probs.to_vec()?;
        let input_data = input.to_vec()?;
        let mut output_data = vec![0.0f32; batch_size * seq_len * self.num_experts];

        // Process each token in the batch
        for b in 0..batch_size {
            for s in 0..seq_len {
                let token_idx = b * seq_len + s;
                let input_token_start = token_idx * self.input_dim;
                let input_token_end = input_token_start + self.input_dim;
                let token_input = &input_data[input_token_start..input_token_end];

                // Get group probabilities for this token
                let group_probs_start = token_idx * self.num_groups;
                let group_probs_end = group_probs_start + self.num_groups;
                let token_group_probs = &group_probs_data[group_probs_start..group_probs_end];

                // For each expert group, compute expert probabilities within that group
                for (group_idx, &group_prob) in token_group_probs.iter().enumerate() {
                    // Route within this group using the group-specific expert router
                    let expert_router_data = self.expert_routers[group_idx].to_vec()?;

                    // Compute expert logits within this group
                    let mut expert_logits = vec![0.0f32; self.experts_per_group];
                    for (expert_idx, logit_slot) in expert_logits.iter_mut().enumerate() {
                        let mut logit = 0.0f32;
                        for (input_idx, &input_val) in token_input.iter().enumerate() {
                            let weight_idx = input_idx * self.experts_per_group + expert_idx;
                            if weight_idx < expert_router_data.len() {
                                logit += input_val * expert_router_data[weight_idx];
                            }
                        }
                        *logit_slot = logit;
                    }

                    // Apply softmax to get expert probabilities within the group
                    let max_logit = expert_logits
                        .iter()
                        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_sum: f32 = expert_logits.iter().map(|&x| (x - max_logit).exp()).sum();

                    // Combine group probability with within-group expert probabilities
                    for (expert_idx, &expert_logit) in expert_logits.iter().enumerate() {
                        let global_expert_idx = group_idx * self.experts_per_group + expert_idx;
                        if global_expert_idx < self.num_experts {
                            let expert_prob_within_group = if exp_sum > 0.0 {
                                (expert_logit - max_logit).exp() / exp_sum
                            } else {
                                0.0
                            };
                            let final_expert_prob = group_prob * expert_prob_within_group;

                            let output_idx = token_idx * self.num_experts + global_expert_idx;
                            output_data[output_idx] = final_expert_prob;
                        }
                    }
                }
            }
        }

        // Convert back to tensor
        let output_tensor =
            Tensor::from_vec(output_data, &[batch_size, seq_len, self.num_experts])?;
        Ok(output_tensor)
    }

    /// Get the number of expert groups
    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    /// Get the number of experts per group
    pub fn experts_per_group(&self) -> usize {
        self.experts_per_group
    }

    /// Get hierarchical routing statistics
    pub fn get_hierarchy_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("num_groups".to_string(), self.num_groups);
        stats.insert("experts_per_group".to_string(), self.experts_per_group);
        stats.insert("total_experts".to_string(), self.num_experts);
        stats.insert(
            "group_router_params".to_string(),
            self.input_dim * self.num_groups,
        );

        let expert_router_params: usize = self
            .expert_routers
            .iter()
            .map(|_router| self.input_dim * self.experts_per_group)
            .sum();
        stats.insert("expert_router_params".to_string(), expert_router_params);

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_parallelism::config::ExpertParallelismConfig;

    #[test]
    fn test_expert_assignment() {
        let assignment = ExpertAssignment::new(0, 0.8, 5, 0);
        assert_eq!(assignment.expert_id, 0);
        assert_eq!(assignment.probability, 0.8);
        assert_eq!(assignment.token_idx, 5);
        assert_eq!(assignment.expert_rank, 0);
        assert!(assignment.is_valid());
    }

    #[test]
    fn test_routing_decision() {
        let assignments = vec![
            vec![ExpertAssignment::new(0, 0.8, 0, 0)],
            vec![ExpertAssignment::new(1, 0.6, 1, 0)],
        ];
        let capacities = vec![1, 1, 0, 0];
        let decision = RoutingDecision::new(assignments, capacities, 2, 0.1, 0.05, 0.15);

        assert_eq!(decision.total_tokens, 2);
        assert_eq!(decision.tokens_dropped, 0);
        assert_eq!(decision.routing_efficiency(), 1.0);
    }

    #[test]
    fn test_expert_router_creation() {
        let config = ExpertParallelismConfig::default();
        let router = ExpertRouter::new(config, 128, 0);
        assert!(router.is_ok());
    }

    #[test]
    fn test_gate_network_creation() {
        let gate_network = GateNetwork::new(128, 64, 0);
        assert!(gate_network.is_ok());

        let network = gate_network.unwrap();
        assert_eq!(network.num_groups(), 8); // sqrt(64) = 8
        assert_eq!(network.experts_per_group(), 8); // 64/8 = 8
    }

    #[test]
    fn test_load_balance_cv() {
        let decision = RoutingDecision {
            expert_assignments: vec![],
            expert_capacities: vec![10, 10, 10, 10], // Perfectly balanced
            total_tokens: 40,
            tokens_dropped: 0,
            load_balance_loss: 0.0,
            router_z_loss: 0.0,
            auxiliary_loss: 0.0,
        };

        assert_eq!(decision.load_balance_cv(), 0.0);

        let imbalanced_decision = RoutingDecision {
            expert_assignments: vec![],
            expert_capacities: vec![20, 10, 5, 5], // Imbalanced
            total_tokens: 40,
            tokens_dropped: 0,
            load_balance_loss: 0.0,
            router_z_loss: 0.0,
            auxiliary_loss: 0.0,
        };

        assert!(imbalanced_decision.load_balance_cv() > 0.0);
    }
}
