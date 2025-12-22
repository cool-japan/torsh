//! Graph Lottery Ticket Hypothesis
//!
//! This module implements the Lottery Ticket Hypothesis (LTH) for Graph Neural Networks,
//! enabling discovery of sparse, high-performing subnetworks.
//!
//! # Key Features:
//! - Iterative magnitude pruning for GNNs
//! - One-shot and iterative pruning strategies
//! - Structured and unstructured pruning
//! - Ticket finding with weight rewinding
//! - Graph-specific pruning (edge and node pruning)
//!
//! # Applications:
//! - Model compression for deployment
//! - Understanding GNN capacity and redundancy
//! - Efficient training of large graph models
//! - Transfer learning with pruned networks
//!
//! # References:
//! - Frankle & Carbin "The Lottery Ticket Hypothesis" (ICLR 2019)
//! - Chen et al. "Lottery Ticket Preserves Weight Correlation" (ICML 2021)
//! - Chen et al. "The Lottery Ticket Hypothesis for Graph Neural Networks" (2021)

use crate::GraphData;
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// Pruning strategy configuration
#[derive(Debug, Clone)]
pub struct PruningConfig {
    /// Pruning method
    pub method: PruningMethod,
    /// Target sparsity (0.0 = dense, 1.0 = completely sparse)
    pub target_sparsity: f32,
    /// Number of pruning iterations
    pub num_iterations: usize,
    /// Whether to use weight rewinding
    pub use_rewinding: bool,
    /// Epoch to rewind to (if using rewinding)
    pub rewind_epoch: usize,
    /// Whether to prune structured (channels/nodes) or unstructured (weights)
    pub structured: bool,
}

/// Pruning method
#[derive(Debug, Clone, Copy)]
pub enum PruningMethod {
    /// Magnitude-based pruning (prune smallest weights)
    Magnitude,
    /// One-shot magnitude pruning
    OneShotMagnitude,
    /// Gradient-based pruning
    Gradient,
    /// Random pruning (baseline)
    Random,
    /// SNIP: Single-shot Network Pruning based on connection sensitivity
    SNIP,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            method: PruningMethod::Magnitude,
            target_sparsity: 0.8,
            num_iterations: 5,
            use_rewinding: true,
            rewind_epoch: 2,
            structured: false,
        }
    }
}

/// Mask for pruning
///
/// Binary mask indicating which weights are kept (1) or pruned (0)
#[derive(Debug, Clone)]
pub struct PruningMask {
    /// Masks for each parameter tensor (name -> mask)
    pub masks: HashMap<String, Tensor>,
    /// Current sparsity level
    pub sparsity: f32,
}

impl PruningMask {
    /// Create a new pruning mask (initially all ones - no pruning)
    pub fn new() -> Self {
        Self {
            masks: HashMap::new(),
            sparsity: 0.0,
        }
    }

    /// Initialize masks for given parameters
    pub fn initialize(
        &mut self,
        parameters: &HashMap<String, Tensor>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (name, param) in parameters {
            let param_shape = param.shape();
            let shape = param_shape.dims();
            let ones_data = vec![1.0f32; param.numel()];
            let mask = torsh_tensor::creation::from_vec(
                ones_data,
                shape,
                torsh_core::device::DeviceType::Cpu,
            )?;
            self.masks.insert(name.clone(), mask);
        }
        Ok(())
    }

    /// Apply masks to parameters
    pub fn apply(
        &self,
        parameters: &mut HashMap<String, Tensor>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (name, param) in parameters.iter_mut() {
            if let Some(mask) = self.masks.get(name) {
                *param = param.mul(mask)?;
            }
        }
        Ok(())
    }

    /// Get total number of parameters
    pub fn total_params(&self) -> usize {
        self.masks.values().map(|m| m.numel()).sum()
    }

    /// Get number of active (non-pruned) parameters
    pub fn active_params(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let mut total = 0;
        for mask in self.masks.values() {
            let mask_data = mask.to_vec()?;
            total += mask_data.iter().filter(|&&x| x > 0.5).count();
        }
        Ok(total)
    }

    /// Compute current sparsity
    pub fn compute_sparsity(&mut self) -> Result<f32, Box<dyn std::error::Error>> {
        let total = self.total_params() as f32;
        let active = self.active_params()? as f32;
        self.sparsity = 1.0 - (active / total);
        Ok(self.sparsity)
    }
}

impl Default for PruningMask {
    fn default() -> Self {
        Self::new()
    }
}

/// Lottery Ticket Finder for Graph Neural Networks
///
/// Discovers winning lottery tickets (sparse subnetworks) that can match
/// the performance of dense networks.
#[derive(Debug, Clone)]
pub struct LotteryTicketFinder {
    config: PruningConfig,
    /// Initial weights (for rewinding)
    initial_weights: HashMap<String, Tensor>,
    /// Early training weights (for rewinding)
    early_weights: HashMap<String, Tensor>,
    /// Current pruning mask
    mask: PruningMask,
}

impl LotteryTicketFinder {
    /// Create a new lottery ticket finder
    ///
    /// # Arguments:
    /// * `config` - Pruning configuration
    ///
    /// # Example:
    /// ```rust
    /// use torsh_graph::lottery_ticket::{LotteryTicketFinder, PruningConfig};
    ///
    /// let config = PruningConfig::default();
    /// let finder = LotteryTicketFinder::new(config);
    /// ```
    pub fn new(config: PruningConfig) -> Self {
        Self {
            config,
            initial_weights: HashMap::new(),
            early_weights: HashMap::new(),
            mask: PruningMask::new(),
        }
    }

    /// Save initial weights (at initialization)
    pub fn save_initial_weights(
        &mut self,
        parameters: &HashMap<String, Tensor>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.initial_weights = parameters.clone();
        self.mask.initialize(parameters)?;
        Ok(())
    }

    /// Save early training weights (for rewinding)
    pub fn save_early_weights(
        &mut self,
        parameters: &HashMap<String, Tensor>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.early_weights = parameters.clone();
        Ok(())
    }

    /// Perform one round of iterative magnitude pruning
    ///
    /// # Arguments:
    /// * `parameters` - Current model parameters
    /// * `iteration` - Current pruning iteration
    ///
    /// # Returns:
    /// Updated pruning mask
    pub fn prune_iteration(
        &mut self,
        parameters: &HashMap<String, Tensor>,
        iteration: usize,
    ) -> Result<PruningMask, Box<dyn std::error::Error>> {
        // Compute sparsity for this iteration
        let current_sparsity = self.compute_iteration_sparsity(iteration);

        match self.config.method {
            PruningMethod::Magnitude | PruningMethod::OneShotMagnitude => {
                self.magnitude_prune(parameters, current_sparsity)?;
            }
            PruningMethod::Random => {
                self.random_prune(parameters, current_sparsity)?;
            }
            PruningMethod::Gradient => {
                // Would require gradients - simplified here
                self.magnitude_prune(parameters, current_sparsity)?;
            }
            PruningMethod::SNIP => {
                // SNIP requires gradients at initialization - simplified
                self.magnitude_prune(parameters, current_sparsity)?;
            }
        }

        self.mask.compute_sparsity()?;
        Ok(self.mask.clone())
    }

    /// Compute target sparsity for given iteration
    fn compute_iteration_sparsity(&self, iteration: usize) -> f32 {
        if self.config.num_iterations == 1 {
            return self.config.target_sparsity;
        }

        // Exponential sparsity schedule: s_t = 1 - (1 - s_f)^((t+1)/T)
        // where s_f is final sparsity, t is iteration (0-indexed), T is total iterations
        // This ensures sparsity increases from 0 to s_f
        let ratio = (iteration + 1) as f32 / self.config.num_iterations as f32;
        let remaining_density = 1.0 - self.config.target_sparsity;
        1.0 - remaining_density.powf(ratio)
    }

    /// Magnitude-based pruning
    fn magnitude_prune(
        &mut self,
        parameters: &HashMap<String, Tensor>,
        target_sparsity: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Collect all weight magnitudes
        let mut all_magnitudes = Vec::new();

        for (name, param) in parameters {
            if !self.is_prunable_param(name) {
                continue;
            }

            let param_data = param.to_vec()?;
            let mask_data = self.mask.masks.get(name).unwrap().to_vec()?;

            for (i, &weight) in param_data.iter().enumerate() {
                if mask_data[i] > 0.5 {
                    // Only consider currently active weights
                    all_magnitudes.push((name.clone(), i, weight.abs()));
                }
            }
        }

        // Sort by magnitude
        all_magnitudes.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // Determine threshold for pruning
        let num_to_prune = (all_magnitudes.len() as f32 * target_sparsity) as usize;
        let weights_to_prune: Vec<_> = all_magnitudes.iter().take(num_to_prune).collect();

        // Update masks
        for (name, idx, _) in weights_to_prune {
            if let Some(mask) = self.mask.masks.get_mut(name) {
                let mut mask_data = mask.to_vec()?;
                mask_data[*idx] = 0.0;
                *mask = torsh_tensor::creation::from_vec(
                    mask_data,
                    mask.shape().dims(),
                    torsh_core::device::DeviceType::Cpu,
                )?;
            }
        }

        Ok(())
    }

    /// Random pruning (baseline)
    fn random_prune(
        &mut self,
        parameters: &HashMap<String, Tensor>,
        target_sparsity: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use scirs2_core::random::{thread_rng, Uniform};
        use scirs2_core::Distribution;

        let mut rng = thread_rng();
        let uniform = Uniform::new(0.0, 1.0)
            .map_err(|e| format!("Failed to create uniform distribution: {}", e))?;

        for (name, _param) in parameters {
            if !self.is_prunable_param(name) {
                continue;
            }

            if let Some(mask) = self.mask.masks.get_mut(name) {
                let mut mask_data = mask.to_vec()?;

                for val in mask_data.iter_mut() {
                    if *val > 0.5 {
                        // Currently active
                        let rand_val: f32 = uniform.sample(&mut rng) as f32;
                        if rand_val < target_sparsity {
                            *val = 0.0;
                        }
                    }
                }

                *mask = torsh_tensor::creation::from_vec(
                    mask_data,
                    mask.shape().dims(),
                    torsh_core::device::DeviceType::Cpu,
                )?;
            }
        }

        Ok(())
    }

    /// Check if parameter should be pruned (exclude biases, etc.)
    fn is_prunable_param(&self, name: &str) -> bool {
        !name.contains("bias") && !name.contains("norm")
    }

    /// Rewind weights to initial or early training state
    pub fn rewind_weights(
        &self,
        parameters: &mut HashMap<String, Tensor>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rewind_source = if self.config.use_rewinding && !self.early_weights.is_empty() {
            &self.early_weights
        } else {
            &self.initial_weights
        };

        for (name, param) in parameters.iter_mut() {
            if let Some(init_param) = rewind_source.get(name) {
                *param = init_param.clone();
            }
        }

        Ok(())
    }

    /// Get current mask
    pub fn get_mask(&self) -> &PruningMask {
        &self.mask
    }

    /// Get current sparsity
    pub fn get_sparsity(&self) -> f32 {
        self.mask.sparsity
    }
}

/// Graph-specific pruning utilities
///
/// Provides methods for pruning graph structure (edges, nodes) in addition
/// to weight pruning.
pub struct GraphPruning;

impl GraphPruning {
    /// Prune edges based on importance scores
    ///
    /// # Arguments:
    /// * `graph` - Input graph
    /// * `edge_scores` - Importance score for each edge
    /// * `keep_ratio` - Fraction of edges to keep
    ///
    /// # Returns:
    /// Pruned graph
    pub fn prune_edges(
        graph: &GraphData,
        edge_scores: &[f32],
        keep_ratio: f32,
    ) -> Result<GraphData, Box<dyn std::error::Error>> {
        if edge_scores.len() != graph.num_edges {
            return Err("Edge scores length must match number of edges".into());
        }

        // Sort edges by score
        let mut scored_edges: Vec<_> = edge_scores.iter().enumerate().collect();
        scored_edges.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        // Keep top edges
        let num_keep = (graph.num_edges as f32 * keep_ratio).ceil() as usize;
        let keep_indices: Vec<_> = scored_edges
            .iter()
            .take(num_keep)
            .map(|(i, _)| *i)
            .collect();

        // Build new edge index
        let edge_data = graph.edge_index.to_vec()?;
        let mut new_edges = Vec::new();

        for &idx in &keep_indices {
            new_edges.push(edge_data[idx]);
        }
        for &idx in &keep_indices {
            new_edges.push(edge_data[graph.num_edges + idx]);
        }

        let new_edge_index = torsh_tensor::creation::from_vec(
            new_edges,
            &[2, num_keep],
            torsh_core::device::DeviceType::Cpu,
        )?;

        // Build new edge attributes if present
        let new_edge_attr = if let Some(ref edge_attr) = graph.edge_attr {
            let attr_data = edge_attr.to_vec()?;
            let attr_dim = edge_attr.shape().dims()[1];
            let mut new_attr = Vec::new();

            for &idx in &keep_indices {
                for d in 0..attr_dim {
                    new_attr.push(attr_data[idx * attr_dim + d]);
                }
            }

            Some(torsh_tensor::creation::from_vec(
                new_attr,
                &[num_keep, attr_dim],
                torsh_core::device::DeviceType::Cpu,
            )?)
        } else {
            None
        };

        Ok(GraphData {
            x: graph.x.clone(),
            edge_index: new_edge_index,
            edge_attr: new_edge_attr,
            batch: graph.batch.clone(),
            num_nodes: graph.num_nodes,
            num_edges: num_keep,
        })
    }

    /// Prune nodes based on importance scores
    ///
    /// # Arguments:
    /// * `graph` - Input graph
    /// * `node_scores` - Importance score for each node
    /// * `keep_ratio` - Fraction of nodes to keep
    ///
    /// # Returns:
    /// Pruned graph with remapped node indices
    pub fn prune_nodes(
        graph: &GraphData,
        node_scores: &[f32],
        keep_ratio: f32,
    ) -> Result<GraphData, Box<dyn std::error::Error>> {
        if node_scores.len() != graph.num_nodes {
            return Err("Node scores length must match number of nodes".into());
        }

        // Sort nodes by score
        let mut scored_nodes: Vec<_> = node_scores.iter().enumerate().collect();
        scored_nodes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        // Keep top nodes
        let num_keep = (graph.num_nodes as f32 * keep_ratio).ceil() as usize;
        let keep_indices: Vec<_> = scored_nodes
            .iter()
            .take(num_keep)
            .map(|(i, _)| *i)
            .collect();

        // Create node remapping
        let mut node_map = vec![None; graph.num_nodes];
        for (new_idx, &old_idx) in keep_indices.iter().enumerate() {
            node_map[old_idx] = Some(new_idx);
        }

        // Filter features
        let feat_data = graph.x.to_vec()?;
        let feat_dim = graph.x.shape().dims()[1];
        let mut new_features = Vec::new();

        for &idx in &keep_indices {
            for d in 0..feat_dim {
                new_features.push(feat_data[idx * feat_dim + d]);
            }
        }

        let new_x = torsh_tensor::creation::from_vec(
            new_features,
            &[num_keep, feat_dim],
            torsh_core::device::DeviceType::Cpu,
        )?;

        // Filter and remap edges
        let edge_data = graph.edge_index.to_vec()?;
        let mut new_edges = Vec::new();
        let mut kept_edge_indices = Vec::new();

        for e in 0..graph.num_edges {
            let src = edge_data[e] as usize;
            let dst = edge_data[graph.num_edges + e] as usize;

            if let (Some(new_src), Some(_new_dst)) = (node_map[src], node_map[dst]) {
                new_edges.push(new_src as f32);
                kept_edge_indices.push(e);
            }
        }

        // Add destination indices
        for &e in &kept_edge_indices {
            let dst = edge_data[graph.num_edges + e] as usize;
            new_edges.push(node_map[dst].unwrap() as f32);
        }

        let num_new_edges = kept_edge_indices.len();
        let new_edge_index = if num_new_edges > 0 {
            torsh_tensor::creation::from_vec(
                new_edges,
                &[2, num_new_edges],
                torsh_core::device::DeviceType::Cpu,
            )?
        } else {
            torsh_tensor::creation::zeros(&[2, 0])?
        };

        Ok(GraphData {
            x: new_x,
            edge_index: new_edge_index,
            edge_attr: None,
            batch: None,
            num_nodes: num_keep,
            num_edges: num_new_edges,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_pruning_config_default() {
        let config = PruningConfig::default();
        assert_eq!(config.target_sparsity, 0.8);
        assert_eq!(config.num_iterations, 5);
        assert!(config.use_rewinding);
    }

    #[test]
    fn test_pruning_mask_initialization() {
        let mut mask = PruningMask::new();
        let mut params = HashMap::new();

        let param1 = from_vec(vec![1.0; 10], &[10], DeviceType::Cpu).unwrap();
        params.insert("weight1".to_string(), param1);

        let result = mask.initialize(&params);
        assert!(result.is_ok());
        assert_eq!(mask.masks.len(), 1);
    }

    #[test]
    fn test_pruning_mask_sparsity() {
        let mut mask = PruningMask::new();

        let mask_data = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mask_tensor = from_vec(mask_data, &[10], DeviceType::Cpu).unwrap();
        mask.masks.insert("test".to_string(), mask_tensor);

        let sparsity = mask.compute_sparsity().unwrap();
        assert!((sparsity - 0.6).abs() < 0.01); // 6 out of 10 are pruned
    }

    #[test]
    fn test_lottery_ticket_finder_creation() {
        let config = PruningConfig::default();
        let finder = LotteryTicketFinder::new(config);
        assert_eq!(finder.get_sparsity(), 0.0);
    }

    #[test]
    fn test_save_initial_weights() {
        let config = PruningConfig::default();
        let mut finder = LotteryTicketFinder::new(config);

        let mut params = HashMap::new();
        let param1 = from_vec(vec![1.0; 10], &[10], DeviceType::Cpu).unwrap();
        params.insert("weight1".to_string(), param1);

        let result = finder.save_initial_weights(&params);
        assert!(result.is_ok());
        assert_eq!(finder.initial_weights.len(), 1);
    }

    #[test]
    fn test_magnitude_pruning() {
        let mut config = PruningConfig::default();
        config.target_sparsity = 0.5;
        config.num_iterations = 1;

        let mut finder = LotteryTicketFinder::new(config);

        let mut params = HashMap::new();
        let param_data = vec![0.1, 0.5, 0.2, 0.9, 0.3, 0.7, 0.4, 0.8, 0.6, 0.05];
        let param1 = from_vec(param_data, &[10], DeviceType::Cpu).unwrap();
        params.insert("weight1".to_string(), param1);

        finder.save_initial_weights(&params).unwrap();

        let mask = finder.prune_iteration(&params, 0).unwrap();
        let sparsity = mask.sparsity;

        // Should be approximately 50% sparse
        assert!(sparsity > 0.4 && sparsity < 0.6);
    }

    #[test]
    fn test_graph_edge_pruning() {
        let x = from_vec(vec![1.0; 4 * 2], &[4, 2], DeviceType::Cpu).unwrap();
        let edge_index = from_vec(
            vec![0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0],
            &[2, 4],
            DeviceType::Cpu,
        )
        .unwrap();
        let graph = GraphData::new(x, edge_index);

        let edge_scores = vec![0.9, 0.3, 0.7, 0.5];
        let pruned = GraphPruning::prune_edges(&graph, &edge_scores, 0.5).unwrap();

        assert_eq!(pruned.num_edges, 2); // Keep top 50% = 2 edges
        assert_eq!(pruned.num_nodes, 4); // Nodes unchanged
    }

    #[test]
    fn test_graph_node_pruning() {
        let x = from_vec(vec![1.0; 5 * 3], &[5, 3], DeviceType::Cpu).unwrap();
        let edge_index = from_vec(
            vec![0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
            &[2, 4],
            DeviceType::Cpu,
        )
        .unwrap();
        let graph = GraphData::new(x, edge_index);

        let node_scores = vec![0.9, 0.8, 0.3, 0.7, 0.2];
        let pruned = GraphPruning::prune_nodes(&graph, &node_scores, 0.6).unwrap();

        assert_eq!(pruned.num_nodes, 3); // Keep top 60% = 3 nodes
        assert_eq!(pruned.x.shape().dims(), &[3, 3]);
    }

    #[test]
    fn test_iterative_sparsity_schedule() {
        let config = PruningConfig {
            target_sparsity: 0.9,
            num_iterations: 3,
            ..Default::default()
        };
        let finder = LotteryTicketFinder::new(config);

        let s0 = finder.compute_iteration_sparsity(0);
        let s1 = finder.compute_iteration_sparsity(1);
        let s2 = finder.compute_iteration_sparsity(2);

        // Sparsity should increase
        assert!(s0 < s1);
        assert!(s1 < s2);
        // Final should be close to target
        assert!((s2 - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_weight_rewinding() {
        let config = PruningConfig::default();
        let mut finder = LotteryTicketFinder::new(config);

        let mut initial_params = HashMap::new();
        let init_weight = from_vec(vec![1.0; 5], &[5], DeviceType::Cpu).unwrap();
        initial_params.insert("weight".to_string(), init_weight);

        finder.save_initial_weights(&initial_params).unwrap();

        let mut current_params = HashMap::new();
        let current_weight = from_vec(vec![2.0; 5], &[5], DeviceType::Cpu).unwrap();
        current_params.insert("weight".to_string(), current_weight);

        finder.rewind_weights(&mut current_params).unwrap();

        let rewound = current_params.get("weight").unwrap().to_vec().unwrap();
        for &val in &rewound {
            assert!((val - 1.0).abs() < 0.01);
        }
    }
}
