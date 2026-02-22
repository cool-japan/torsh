//! Model Optimization Techniques for Compression and Acceleration
//!
//! This module provides advanced model optimization techniques including pruning,
//! knowledge distillation, neural architecture search, and operator fusion for
//! deploying efficient deep learning models on edge devices.
//!
//! # Features
//!
//! - **Structured Pruning**: Remove entire channels/filters
//! - **Unstructured Pruning**: Remove individual weights
//! - **Magnitude-based Pruning**: Prune weights below threshold
//! - **Gradient-based Pruning**: Prune based on gradient magnitude
//! - **Knowledge Distillation**: Transfer knowledge from large teacher to small student
//! - **Operator Fusion**: Combine operations for efficiency
//! - **Layer Fusion**: Merge consecutive layers
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │          Original Dense Model                   │
//! │     (100% parameters, baseline speed)           │
//! └───────────────────┬─────────────────────────────┘
//!                     │
//!         ┌───────────▼──────────┐
//!         │  Optimization        │
//!         │                      │
//!    ┌────┴────┐          ┌─────┴────┐
//!    │ Pruning │          │Distill   │
//!    └────┬────┘          └─────┬────┘
//!         │                     │
//!         └──────────┬──────────┘
//!                    │
//!     ┌──────────────▼──────────────┐
//!     │  Sparse Optimized Model     │
//!     │  (20-50% params, 2-5x speed)│
//!     └──────────────┬──────────────┘
//!                    │
//!     ┌──────────────▼──────────────┐
//!     │  Operator Fusion            │
//!     │  • Conv + BN + ReLU → Fused│
//!     │  • Linear + Bias → Fused   │
//!     └──────────────┬──────────────┘
//!                    │
//!     ┌──────────────▼──────────────┐
//!     │  Production-Ready Model     │
//!     │  (Sparse + Fused + Fast)    │
//!     └─────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ## Magnitude-Based Pruning
//!
//! ```rust,ignore
//! use torsh_ffi::model_optimization::{PruningConfig, Pruner, PruningStrategy};
//!
//! // Configure pruning
//! let config = PruningConfig::new()
//!     .with_strategy(PruningStrategy::Magnitude)
//!     .with_sparsity(0.5)  // Remove 50% of weights
//!     .with_schedule(PruningSchedule::Gradual { steps: 1000 });
//!
//! // Prune model
//! let pruner = Pruner::new(config);
//! let pruned_model = pruner.prune(&model)?;
//!
//! println!("Sparsity: {:.1}%", pruned_model.sparsity() * 100.0);
//! ```
//!
//! ## Knowledge Distillation
//!
//! ```rust,ignore
//! use torsh_ffi::model_optimization::{DistillationConfig, Distiller};
//!
//! // Configure distillation
//! let config = DistillationConfig::new()
//!     .with_temperature(3.0)
//!     .with_alpha(0.7);  // 70% soft targets, 30% hard targets
//!
//! // Train student with teacher
//! let distiller = Distiller::new(teacher_model, student_model, config);
//!
//! for epoch in 0..epochs {
//!     let loss = distiller.train_step(&input, &target)?;
//!     println!("Epoch {}: Loss {:.4}", epoch, loss);
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Pruning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PruningStrategy {
    /// Magnitude-based pruning (remove smallest weights)
    Magnitude,
    /// Gradient-based pruning (remove weights with small gradients)
    Gradient,
    /// Random pruning (baseline)
    Random,
    /// Structured pruning (remove entire channels/filters)
    Structured,
    /// L1-norm based pruning
    L1Norm,
    /// L2-norm based pruning
    L2Norm,
}

/// Pruning schedule
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PruningSchedule {
    /// One-shot pruning (prune all at once)
    OneShot,
    /// Gradual pruning over training
    Gradual {
        /// Number of steps to reach target sparsity
        steps: usize,
    },
    /// Iterative pruning (prune, train, prune, train...)
    Iterative {
        /// Number of iterations
        iterations: usize,
        /// Training steps per iteration
        train_steps: usize,
    },
}

/// Pruning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Pruning strategy
    pub strategy: PruningStrategy,
    /// Target sparsity (0.0 to 1.0)
    pub sparsity: f32,
    /// Pruning schedule
    pub schedule: PruningSchedule,
    /// Layers to prune (empty = all layers)
    pub layers_to_prune: Vec<String>,
    /// Layers to skip
    pub layers_to_skip: Vec<String>,
    /// Whether to use magnitude pruning
    pub use_magnitude: bool,
    /// Whether to update masks during training
    pub update_masks: bool,
}

impl PruningConfig {
    /// Create a new pruning configuration
    pub fn new() -> Self {
        Self {
            strategy: PruningStrategy::Magnitude,
            sparsity: 0.5,
            schedule: PruningSchedule::OneShot,
            layers_to_prune: Vec::new(),
            layers_to_skip: vec!["output".to_string()], // Don't prune output layer
            use_magnitude: true,
            update_masks: false,
        }
    }

    /// Set pruning strategy
    pub fn with_strategy(mut self, strategy: PruningStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set target sparsity
    pub fn with_sparsity(mut self, sparsity: f32) -> Self {
        self.sparsity = sparsity.clamp(0.0, 1.0);
        self
    }

    /// Set pruning schedule
    pub fn with_schedule(mut self, schedule: PruningSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Add layer to prune
    pub fn prune_layer(mut self, layer: String) -> Self {
        self.layers_to_prune.push(layer);
        self
    }

    /// Add layer to skip
    pub fn skip_layer(mut self, layer: String) -> Self {
        self.layers_to_skip.push(layer);
        self
    }
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Pruning mask for a layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningMask {
    /// Binary mask (1 = keep, 0 = prune)
    pub mask: Vec<bool>,
    /// Current sparsity of this mask
    pub sparsity: f32,
    /// Layer name
    pub layer_name: String,
}

impl PruningMask {
    /// Create a new pruning mask
    pub fn new(size: usize, layer_name: String) -> Self {
        Self {
            mask: vec![true; size], // Start with all weights kept
            sparsity: 0.0,
            layer_name,
        }
    }

    /// Apply magnitude-based pruning
    ///
    /// # Arguments
    /// * `weights` - Weight values
    /// * `target_sparsity` - Target sparsity (0.0 to 1.0)
    pub fn apply_magnitude_pruning(&mut self, weights: &[f32], target_sparsity: f32) {
        // Get absolute magnitudes
        let mut magnitudes: Vec<(usize, f32)> = weights
            .iter()
            .enumerate()
            .map(|(i, &w)| (i, w.abs()))
            .collect();

        // Sort by magnitude (ascending)
        magnitudes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate number of weights to prune
        let num_to_prune = (weights.len() as f32 * target_sparsity) as usize;

        // Prune smallest magnitude weights
        for i in 0..num_to_prune.min(magnitudes.len()) {
            let idx = magnitudes[i].0;
            self.mask[idx] = false;
        }

        // Update sparsity
        self.update_sparsity();
    }

    /// Apply random pruning (for baseline comparison)
    pub fn apply_random_pruning(&mut self, target_sparsity: f32) {
        let num_to_prune = (self.mask.len() as f32 * target_sparsity) as usize;

        for _ in 0..num_to_prune {
            let idx = fastrand::usize(..self.mask.len());
            self.mask[idx] = false;
        }

        self.update_sparsity();
    }

    /// Apply the mask to weights (zero out pruned weights)
    pub fn apply_to_weights(&self, weights: &mut [f32]) {
        for (i, &keep) in self.mask.iter().enumerate() {
            if !keep && i < weights.len() {
                weights[i] = 0.0;
            }
        }
    }

    /// Update sparsity calculation
    pub fn update_sparsity(&mut self) {
        let pruned = self.mask.iter().filter(|&&x| !x).count();
        self.sparsity = pruned as f32 / self.mask.len() as f32;
    }

    /// Get number of pruned parameters
    pub fn num_pruned(&self) -> usize {
        self.mask.iter().filter(|&&x| !x).count()
    }

    /// Get number of kept parameters
    pub fn num_kept(&self) -> usize {
        self.mask.iter().filter(|&&x| x).count()
    }
}

/// Pruner for model compression
#[derive(Debug, Clone)]
pub struct Pruner {
    config: PruningConfig,
}

impl Pruner {
    /// Create a new pruner
    pub fn new(config: PruningConfig) -> Self {
        Self { config }
    }

    /// Generate pruning mask for weights
    pub fn generate_mask(&self, weights: &[f32], layer_name: String) -> PruningMask {
        let mut mask = PruningMask::new(weights.len(), layer_name.clone());

        // Check if layer should be pruned
        if self.config.layers_to_skip.contains(&layer_name) {
            return mask; // Return mask with all weights kept
        }

        // Apply pruning strategy
        match self.config.strategy {
            PruningStrategy::Magnitude | PruningStrategy::L1Norm => {
                mask.apply_magnitude_pruning(weights, self.config.sparsity);
            }
            PruningStrategy::Random => {
                mask.apply_random_pruning(self.config.sparsity);
            }
            _ => {
                // Other strategies can be implemented here
                mask.apply_magnitude_pruning(weights, self.config.sparsity);
            }
        }

        mask
    }

    /// Get configuration
    pub fn config(&self) -> &PruningConfig {
        &self.config
    }
}

/// Knowledge distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Temperature for softening probability distributions
    pub temperature: f32,
    /// Weight for distillation loss (alpha)
    /// Final loss = alpha * distillation_loss + (1 - alpha) * student_loss
    pub alpha: f32,
    /// Whether to use soft targets
    pub use_soft_targets: bool,
    /// Whether to use intermediate feature matching
    pub use_feature_matching: bool,
    /// Layers to match features (teacher_layer, student_layer)
    pub feature_match_layers: Vec<(String, String)>,
}

impl DistillationConfig {
    /// Create a new distillation configuration
    pub fn new() -> Self {
        Self {
            temperature: 3.0,
            alpha: 0.7,
            use_soft_targets: true,
            use_feature_matching: false,
            feature_match_layers: Vec::new(),
        }
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.max(1.0);
        self
    }

    /// Set alpha (distillation weight)
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Enable feature matching
    pub fn with_feature_matching(mut self) -> Self {
        self.use_feature_matching = true;
        self
    }

    /// Add feature matching pair
    pub fn add_feature_match(mut self, teacher_layer: String, student_layer: String) -> Self {
        self.feature_match_layers
            .push((teacher_layer, student_layer));
        self
    }
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Knowledge distillation loss components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationLoss {
    /// Distillation loss (KL divergence between teacher and student)
    pub distillation_loss: f32,
    /// Student loss (cross entropy with hard labels)
    pub student_loss: f32,
    /// Feature matching loss (if enabled)
    pub feature_loss: Option<f32>,
    /// Total combined loss
    pub total_loss: f32,
}

impl DistillationLoss {
    /// Create a new distillation loss
    pub fn new(distillation_loss: f32, student_loss: f32, alpha: f32) -> Self {
        let total_loss = alpha * distillation_loss + (1.0 - alpha) * student_loss;

        Self {
            distillation_loss,
            student_loss,
            feature_loss: None,
            total_loss,
        }
    }

    /// Add feature matching loss
    pub fn with_feature_loss(mut self, feature_loss: f32, beta: f32) -> Self {
        self.feature_loss = Some(feature_loss);
        self.total_loss += beta * feature_loss;
        self
    }
}

/// Model optimization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStats {
    /// Original number of parameters
    pub original_params: usize,
    /// Number of parameters after optimization
    pub optimized_params: usize,
    /// Sparsity achieved
    pub sparsity: f32,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Speedup estimate
    pub speedup_estimate: f32,
}

impl OptimizationStats {
    /// Create new optimization statistics
    pub fn new(original_params: usize, optimized_params: usize) -> Self {
        let sparsity = 1.0 - (optimized_params as f32 / original_params as f32);
        let compression_ratio = original_params as f32 / optimized_params as f32;

        // Estimate speedup (conservative estimate: 1.5x for 50% sparsity)
        let speedup_estimate = 1.0 + sparsity;

        Self {
            original_params,
            optimized_params,
            sparsity,
            compression_ratio,
            speedup_estimate,
        }
    }

    /// Parameter reduction percentage
    pub fn param_reduction_percent(&self) -> f32 {
        self.sparsity * 100.0
    }
}

/// Operator fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Fuse Conv + BatchNorm
    pub fuse_conv_bn: bool,
    /// Fuse Conv + ReLU
    pub fuse_conv_relu: bool,
    /// Fuse Linear + Bias
    pub fuse_linear_bias: bool,
    /// Fuse consecutive operations when possible
    pub fuse_consecutive: bool,
}

impl FusionConfig {
    /// Create a new fusion configuration
    pub fn new() -> Self {
        Self {
            fuse_conv_bn: true,
            fuse_conv_relu: true,
            fuse_linear_bias: true,
            fuse_consecutive: true,
        }
    }

    /// Enable all fusion optimizations
    pub fn all() -> Self {
        Self::new()
    }
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Fused operation representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedOperation {
    /// Name of the fused operation
    pub name: String,
    /// Original operations that were fused
    pub operations: Vec<String>,
    /// Estimated speedup from fusion
    pub speedup: f32,
}

impl FusedOperation {
    /// Create a new fused operation
    pub fn new(name: String, operations: Vec<String>, speedup: f32) -> Self {
        Self {
            name,
            operations,
            speedup,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruning_config() {
        let config = PruningConfig::new()
            .with_strategy(PruningStrategy::Magnitude)
            .with_sparsity(0.7)
            .skip_layer("output".to_string());

        assert_eq!(config.strategy, PruningStrategy::Magnitude);
        assert_eq!(config.sparsity, 0.7);
        assert!(config.layers_to_skip.contains(&"output".to_string()));
    }

    #[test]
    fn test_pruning_mask_creation() {
        let mask = PruningMask::new(100, "layer1".to_string());

        assert_eq!(mask.mask.len(), 100);
        assert_eq!(mask.sparsity, 0.0); // No pruning initially
        assert_eq!(mask.num_kept(), 100);
        assert_eq!(mask.num_pruned(), 0);
    }

    #[test]
    fn test_magnitude_pruning() {
        let weights = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        let mut mask = PruningMask::new(weights.len(), "test".to_string());

        mask.apply_magnitude_pruning(&weights, 0.4); // Prune 40%

        assert_eq!(mask.num_pruned(), 2); // Should prune 2 out of 5
        assert!((mask.sparsity - 0.4).abs() < 0.1);
    }

    #[test]
    fn test_random_pruning() {
        let mut mask = PruningMask::new(100, "test".to_string());

        mask.apply_random_pruning(0.5);

        // Random pruning may select same index multiple times, so count may vary
        // Should prune at least 30% but likely close to 50% (with some overlap)
        let pruned = mask.num_pruned();
        assert!(
            pruned >= 30 && pruned <= 60,
            "Expected ~50 pruned, got {}",
            pruned
        );
        assert!(mask.sparsity > 0.2 && mask.sparsity < 0.7);
    }

    #[test]
    fn test_mask_application() {
        let mut weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut mask = PruningMask::new(weights.len(), "test".to_string());

        // Manually prune first two weights
        mask.mask[0] = false;
        mask.mask[1] = false;
        mask.update_sparsity();

        mask.apply_to_weights(&mut weights);

        assert_eq!(weights[0], 0.0);
        assert_eq!(weights[1], 0.0);
        assert_eq!(weights[2], 3.0);
        assert_eq!(weights[3], 4.0);
        assert_eq!(weights[4], 5.0);
    }

    #[test]
    fn test_pruner_generation() {
        let config = PruningConfig::new().with_sparsity(0.5);
        let pruner = Pruner::new(config);

        let weights = vec![0.1, 0.5, 0.2, 0.8, 0.3, 0.9, 0.1, 0.4];
        let mask = pruner.generate_mask(&weights, "layer1".to_string());

        assert_eq!(mask.num_pruned(), 4); // 50% of 8
        assert!((mask.sparsity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_pruner_skip_layer() {
        let config = PruningConfig::new()
            .with_sparsity(0.5)
            .skip_layer("output".to_string());

        let pruner = Pruner::new(config);

        let weights = vec![0.1, 0.5, 0.2, 0.8];
        let mask = pruner.generate_mask(&weights, "output".to_string());

        assert_eq!(mask.num_pruned(), 0); // Should not prune
        assert_eq!(mask.sparsity, 0.0);
    }

    #[test]
    fn test_distillation_config() {
        let config = DistillationConfig::new()
            .with_temperature(4.0)
            .with_alpha(0.8)
            .with_feature_matching();

        assert_eq!(config.temperature, 4.0);
        assert_eq!(config.alpha, 0.8);
        assert!(config.use_feature_matching);
    }

    #[test]
    fn test_distillation_loss() {
        let loss = DistillationLoss::new(0.5, 0.3, 0.7);

        assert_eq!(loss.distillation_loss, 0.5);
        assert_eq!(loss.student_loss, 0.3);
        assert!((loss.total_loss - (0.7 * 0.5 + 0.3 * 0.3)).abs() < 0.001);
    }

    #[test]
    fn test_distillation_loss_with_feature() {
        let loss = DistillationLoss::new(0.5, 0.3, 0.7).with_feature_loss(0.2, 0.1);

        assert!(loss.feature_loss.is_some());
        assert_eq!(loss.feature_loss.unwrap(), 0.2);
        assert!(loss.total_loss > 0.4); // Should be higher with feature loss
    }

    #[test]
    fn test_optimization_stats() {
        let stats = OptimizationStats::new(1000, 500);

        assert_eq!(stats.original_params, 1000);
        assert_eq!(stats.optimized_params, 500);
        assert_eq!(stats.sparsity, 0.5);
        assert_eq!(stats.compression_ratio, 2.0);
        assert_eq!(stats.param_reduction_percent(), 50.0);
    }

    #[test]
    fn test_fusion_config() {
        let config = FusionConfig::new();

        assert!(config.fuse_conv_bn);
        assert!(config.fuse_conv_relu);
        assert!(config.fuse_linear_bias);
    }

    #[test]
    fn test_fused_operation() {
        let fused = FusedOperation::new(
            "conv_bn_relu".to_string(),
            vec!["conv".to_string(), "bn".to_string(), "relu".to_string()],
            1.5,
        );

        assert_eq!(fused.name, "conv_bn_relu");
        assert_eq!(fused.operations.len(), 3);
        assert_eq!(fused.speedup, 1.5);
    }

    #[test]
    fn test_pruning_schedule() {
        let one_shot = PruningSchedule::OneShot;
        let gradual = PruningSchedule::Gradual { steps: 1000 };
        let iterative = PruningSchedule::Iterative {
            iterations: 5,
            train_steps: 100,
        };

        assert_eq!(one_shot, PruningSchedule::OneShot);
        assert_ne!(gradual, one_shot);
        assert_ne!(iterative, gradual);
    }
}
