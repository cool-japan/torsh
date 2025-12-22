//! Advanced Gradient Clipping Utilities
//!
//! This module provides comprehensive gradient clipping functionality to prevent
//! gradient explosion during training. It includes multiple clipping strategies
//! and supports both global and per-parameter clipping.
//!
//! ## Features
//!
//! - **Norm-based Clipping**: Clip gradients by global L2 norm
//! - **Value-based Clipping**: Clip gradient values element-wise
//! - **Adaptive Clipping**: Dynamic clipping based on gradient statistics
//! - **Per-Layer Clipping**: Apply different clipping strategies per layer
//! - **Gradient Monitoring**: Track gradient norms and clipping frequency
//!
//! ## Usage
//!
//! ```rust,no_run
//! use torsh_autograd::gradient_clipping::{GradientClipper, ClippingStrategy};
//!
//! # fn example() -> torsh_core::error::Result<()> {
//! // Create gradient clipper
//! let mut clipper = GradientClipper::new(ClippingStrategy::Norm(1.0));
//!
//! // Clip gradients during training
//! // clipper.clip_gradients(&mut gradients)?;
//!
//! // Get clipping statistics
//! let stats = clipper.stats();
//! println!("Clipping frequency: {:.2}%", stats.clip_frequency * 100.0);
//! # Ok(())
//! # }
//! ```

use crate::error_handling::AutogradResult;
use std::collections::HashMap;

/// Gradient clipping strategies
#[derive(Debug, Clone)]
pub enum ClippingStrategy {
    /// Clip by global L2 norm
    Norm(f32),
    /// Clip by global L1 norm
    NormL1(f32),
    /// Clip by value (element-wise)
    Value { min: f32, max: f32 },
    /// Adaptive clipping based on gradient history
    Adaptive { percentile: f32, window_size: usize },
    /// Per-parameter norm clipping
    PerParameter(f32),
    /// Combined strategy
    Combined(Vec<ClippingStrategy>),
}

impl ClippingStrategy {
    /// Get human-readable name
    pub fn name(&self) -> String {
        match self {
            Self::Norm(max_norm) => format!("Norm({})", max_norm),
            Self::NormL1(max_norm) => format!("L1Norm({})", max_norm),
            Self::Value { min, max } => format!("Value({}, {})", min, max),
            Self::Adaptive {
                percentile,
                window_size,
            } => {
                format!("Adaptive(p={}, w={})", percentile, window_size)
            }
            Self::PerParameter(max_norm) => format!("PerParameter({})", max_norm),
            Self::Combined(_) => "Combined".to_string(),
        }
    }
}

/// Statistics about gradient clipping
#[derive(Debug, Clone, Default)]
pub struct ClippingStats {
    /// Total number of clipping operations
    pub total_clips: usize,
    /// Number of times gradients were actually clipped
    pub clips_applied: usize,
    /// Average gradient norm before clipping
    pub avg_norm_before: f32,
    /// Average gradient norm after clipping
    pub avg_norm_after: f32,
    /// Maximum gradient norm observed
    pub max_norm_observed: f32,
    /// Minimum gradient norm observed
    pub min_norm_observed: f32,
    /// Frequency of clipping (0.0 to 1.0)
    pub clip_frequency: f32,
}

impl ClippingStats {
    /// Update statistics with a new clipping operation
    pub fn update(&mut self, norm_before: f32, norm_after: f32, was_clipped: bool) {
        self.total_clips += 1;
        if was_clipped {
            self.clips_applied += 1;
        }

        // Update running averages
        let n = self.total_clips as f32;
        self.avg_norm_before = (self.avg_norm_before * (n - 1.0) + norm_before) / n;
        self.avg_norm_after = (self.avg_norm_after * (n - 1.0) + norm_after) / n;

        // Update min/max
        if self.total_clips == 1 {
            self.max_norm_observed = norm_before;
            self.min_norm_observed = norm_before;
        } else {
            self.max_norm_observed = self.max_norm_observed.max(norm_before);
            self.min_norm_observed = self.min_norm_observed.min(norm_before);
        }

        // Update frequency
        self.clip_frequency = self.clips_applied as f32 / self.total_clips as f32;
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Gradient clipper with comprehensive clipping strategies
pub struct GradientClipper {
    strategy: ClippingStrategy,
    stats: ClippingStats,
    /// History of gradient norms for adaptive clipping
    norm_history: Vec<f32>,
    /// Per-parameter statistics
    per_param_stats: HashMap<String, ClippingStats>,
}

impl GradientClipper {
    /// Create a new gradient clipper
    pub fn new(strategy: ClippingStrategy) -> Self {
        Self {
            strategy,
            stats: ClippingStats::default(),
            norm_history: Vec::new(),
            per_param_stats: HashMap::new(),
        }
    }

    /// Get clipping strategy
    pub fn strategy(&self) -> &ClippingStrategy {
        &self.strategy
    }

    /// Set clipping strategy
    pub fn set_strategy(&mut self, strategy: ClippingStrategy) {
        self.strategy = strategy;
    }

    /// Get statistics
    pub fn stats(&self) -> &ClippingStats {
        &self.stats
    }

    /// Get per-parameter statistics
    pub fn per_param_stats(&self) -> &HashMap<String, ClippingStats> {
        &self.per_param_stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats.reset();
        self.per_param_stats.clear();
        self.norm_history.clear();
    }

    /// Compute L2 norm of gradient data
    pub fn compute_l2_norm(&self, data: &[f32]) -> f32 {
        data.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    /// Compute L1 norm of gradient data
    pub fn compute_l1_norm(&self, data: &[f32]) -> f32 {
        data.iter().map(|&x| x.abs()).sum::<f32>()
    }

    /// Clip gradients by L2 norm
    pub fn clip_by_norm(&mut self, data: &mut [f32], max_norm: f32) -> bool {
        let norm = self.compute_l2_norm(data);

        if norm > max_norm {
            let scale = max_norm / (norm + 1e-6);
            for x in data.iter_mut() {
                *x *= scale;
            }
            self.stats.update(norm, max_norm, true);
            true
        } else {
            self.stats.update(norm, norm, false);
            false
        }
    }

    /// Clip gradients by L1 norm
    pub fn clip_by_l1_norm(&mut self, data: &mut [f32], max_norm: f32) -> bool {
        let norm = self.compute_l1_norm(data);

        if norm > max_norm {
            let scale = max_norm / (norm + 1e-6);
            for x in data.iter_mut() {
                *x *= scale;
            }
            self.stats.update(norm, max_norm, true);
            true
        } else {
            self.stats.update(norm, norm, false);
            false
        }
    }

    /// Clip gradients by value
    pub fn clip_by_value(&mut self, data: &mut [f32], min_val: f32, max_val: f32) -> bool {
        let norm_before = self.compute_l2_norm(data);
        let mut was_clipped = false;

        for x in data.iter_mut() {
            let old_val = *x;
            *x = x.max(min_val).min(max_val);
            if *x != old_val {
                was_clipped = true;
            }
        }

        let norm_after = self.compute_l2_norm(data);
        self.stats.update(norm_before, norm_after, was_clipped);
        was_clipped
    }

    /// Adaptive clipping based on gradient history
    pub fn clip_adaptive(&mut self, data: &mut [f32], percentile: f32, window_size: usize) -> bool {
        let norm = self.compute_l2_norm(data);

        // Update history
        self.norm_history.push(norm);
        if self.norm_history.len() > window_size {
            self.norm_history.remove(0);
        }

        // Compute adaptive threshold
        if self.norm_history.len() < 10 {
            // Not enough data for adaptive clipping
            self.stats.update(norm, norm, false);
            return false;
        }

        let mut sorted_norms = self.norm_history.clone();
        sorted_norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((sorted_norms.len() as f32 * percentile) as usize).min(sorted_norms.len() - 1);
        let threshold = sorted_norms[idx];

        // Apply clipping
        self.clip_by_norm(data, threshold)
    }

    /// Clip gradients using the configured strategy
    pub fn clip(&mut self, data: &mut [f32]) -> AutogradResult<bool> {
        match &self.strategy.clone() {
            ClippingStrategy::Norm(max_norm) => Ok(self.clip_by_norm(data, *max_norm)),
            ClippingStrategy::NormL1(max_norm) => Ok(self.clip_by_l1_norm(data, *max_norm)),
            ClippingStrategy::Value { min, max } => Ok(self.clip_by_value(data, *min, *max)),
            ClippingStrategy::Adaptive {
                percentile,
                window_size,
            } => Ok(self.clip_adaptive(data, *percentile, *window_size)),
            ClippingStrategy::PerParameter(max_norm) => Ok(self.clip_by_norm(data, *max_norm)),
            ClippingStrategy::Combined(strategies) => {
                let mut any_clipped = false;
                for strategy in strategies {
                    let old_strategy = self.strategy.clone();
                    self.strategy = strategy.clone();
                    let clipped = self.clip(data)?;
                    any_clipped |= clipped;
                    self.strategy = old_strategy;
                }
                Ok(any_clipped)
            }
        }
    }

    /// Generate performance report
    pub fn report(&self) -> String {
        format!(
            "Gradient Clipping Statistics:\n\
             - Strategy: {}\n\
             - Total clips: {}\n\
             - Clips applied: {} ({:.1}%)\n\
             - Avg norm before: {:.4}\n\
             - Avg norm after: {:.4}\n\
             - Max norm observed: {:.4}\n\
             - Min norm observed: {:.4}\n\
             - Clip frequency: {:.2}%",
            self.strategy.name(),
            self.stats.total_clips,
            self.stats.clips_applied,
            self.stats.clip_frequency * 100.0,
            self.stats.avg_norm_before,
            self.stats.avg_norm_after,
            self.stats.max_norm_observed,
            self.stats.min_norm_observed,
            self.stats.clip_frequency * 100.0
        )
    }
}

impl Default for GradientClipper {
    fn default() -> Self {
        Self::new(ClippingStrategy::Norm(1.0))
    }
}

/// Global gradient clipper instance
static GLOBAL_CLIPPER: once_cell::sync::Lazy<parking_lot::RwLock<GradientClipper>> =
    once_cell::sync::Lazy::new(|| parking_lot::RwLock::new(GradientClipper::default()));

/// Get the global gradient clipper
pub fn get_global_clipper() -> parking_lot::RwLockReadGuard<'static, GradientClipper> {
    GLOBAL_CLIPPER.read()
}

/// Get mutable access to the global gradient clipper
pub fn get_global_clipper_mut() -> parking_lot::RwLockWriteGuard<'static, GradientClipper> {
    GLOBAL_CLIPPER.write()
}

/// Configure the global gradient clipper
pub fn configure_global_clipper(strategy: ClippingStrategy) {
    let mut clipper = GLOBAL_CLIPPER.write();
    clipper.set_strategy(strategy);
}

/// Clip gradients using the global clipper
pub fn clip_gradients_global(data: &mut [f32]) -> AutogradResult<bool> {
    let mut clipper = GLOBAL_CLIPPER.write();
    clipper.clip(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_by_norm() {
        let mut clipper = GradientClipper::new(ClippingStrategy::Norm(1.0));
        let mut data = vec![3.0, 4.0]; // L2 norm = 5.0

        let was_clipped = clipper.clip_by_norm(&mut data, 1.0);
        assert!(was_clipped);

        let norm = clipper.compute_l2_norm(&data);
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_clip_by_value() {
        let mut clipper = GradientClipper::new(ClippingStrategy::Value {
            min: -1.0,
            max: 1.0,
        });
        let mut data = vec![-2.0, 0.5, 2.0];

        let was_clipped = clipper.clip_by_value(&mut data, -1.0, 1.0);
        assert!(was_clipped);
        assert_eq!(data, vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn test_adaptive_clipping() {
        let mut clipper = GradientClipper::new(ClippingStrategy::Adaptive {
            percentile: 0.95,
            window_size: 100,
        });

        // Build up history
        for _ in 0..20 {
            let mut data = vec![1.0, 1.0];
            clipper.clip_adaptive(&mut data, 0.95, 100);
        }

        assert!(clipper.norm_history.len() > 0);
    }

    #[test]
    fn test_clipping_stats() {
        let mut clipper = GradientClipper::new(ClippingStrategy::Norm(1.0));

        let mut data = vec![3.0, 4.0]; // Will be clipped
        clipper.clip_by_norm(&mut data, 1.0);

        let stats = clipper.stats();
        assert_eq!(stats.total_clips, 1);
        assert_eq!(stats.clips_applied, 1);
        assert_eq!(stats.clip_frequency, 1.0);
    }

    #[test]
    fn test_report() {
        let clipper = GradientClipper::new(ClippingStrategy::Norm(1.0));
        let report = clipper.report();

        assert!(report.contains("Gradient Clipping Statistics"));
        assert!(report.contains("Strategy: Norm(1)"));
    }

    #[test]
    fn test_global_clipper() {
        configure_global_clipper(ClippingStrategy::Norm(2.0));

        let clipper = get_global_clipper();
        match clipper.strategy() {
            ClippingStrategy::Norm(max_norm) => assert_eq!(*max_norm, 2.0),
            _ => panic!("Expected Norm strategy"),
        }
    }
}
