//! Adaptive fusion engine with performance learning
//!
//! This module provides an adaptive fusion engine that learns from
//! runtime performance to make better fusion decisions over time.

use super::{core::FusedOp, engine::OpFusionEngine};

/// Adaptive fusion engine that adjusts fusion decisions based on runtime characteristics
pub struct AdaptiveFusionEngine {
    pub base_engine: OpFusionEngine,
    pub performance_history: Vec<FusionPerformance>,
    pub learning_rate: f64,
}

/// Performance metrics for fusion decisions
#[derive(Debug, Clone)]
pub struct FusionPerformance {
    pub operation: FusedOp,
    pub tensor_size: usize,
    pub execution_time: f64,
    pub memory_usage: usize,
    pub benefit_achieved: f64,
}

impl AdaptiveFusionEngine {
    pub fn new() -> Self {
        Self {
            base_engine: OpFusionEngine::new(),
            performance_history: Vec::new(),
            learning_rate: 0.1,
        }
    }

    /// Record performance metrics for a fusion operation
    pub fn record_performance(&mut self, performance: FusionPerformance) {
        self.performance_history.push(performance);

        // Keep only recent history (last 100 operations)
        if self.performance_history.len() > 100 {
            self.performance_history
                .drain(0..self.performance_history.len() - 100);
        }
    }

    /// Predict fusion benefit based on historical performance
    pub fn predict_fusion_benefit(&self, op: &FusedOp, tensor_size: usize) -> f64 {
        let similar_ops: Vec<_> = self
            .performance_history
            .iter()
            .filter(|perf| {
                std::mem::discriminant(&perf.operation) == std::mem::discriminant(op)
                    && (perf.tensor_size as f64 / tensor_size as f64).abs() < 2.0
            })
            .collect();

        if similar_ops.is_empty() {
            // No historical data, use default heuristic
            0.2
        } else {
            // Weighted average based on tensor size similarity
            let total_weight: f64 = similar_ops
                .iter()
                .map(|perf| 1.0 / (1.0 + (perf.tensor_size as f64 - tensor_size as f64).abs()))
                .sum();

            let weighted_benefit: f64 = similar_ops
                .iter()
                .map(|perf| {
                    let weight = 1.0 / (1.0 + (perf.tensor_size as f64 - tensor_size as f64).abs());
                    weight * perf.benefit_achieved
                })
                .sum();

            weighted_benefit / total_weight
        }
    }

    /// Decide whether to fuse operations based on adaptive learning
    pub fn should_fuse_adaptive(&self, op: &FusedOp, tensor_size: usize) -> bool {
        let predicted_benefit = self.predict_fusion_benefit(op, tensor_size);

        // Adaptive threshold based on operation type and tensor size
        let threshold = match op {
            FusedOp::ReluAdd | FusedOp::AddMul | FusedOp::MulAdd => 0.1,
            FusedOp::SigmoidMul | FusedOp::TanhScale => 0.15,
            FusedOp::AddReluMul => 0.2,
            FusedOp::BatchNormFused | FusedOp::LayerNormFused => 0.25,
        };

        // Adjust threshold based on tensor size
        let size_factor = if tensor_size > 10000 { 0.8 } else { 1.2 };
        let adjusted_threshold = threshold * size_factor;

        predicted_benefit > adjusted_threshold
    }

    /// Get average performance for a specific operation type
    pub fn get_average_performance(&self, op: &FusedOp) -> Option<f64> {
        let matching_ops: Vec<_> = self
            .performance_history
            .iter()
            .filter(|perf| std::mem::discriminant(&perf.operation) == std::mem::discriminant(op))
            .collect();

        if matching_ops.is_empty() {
            None
        } else {
            let total_benefit: f64 = matching_ops.iter().map(|perf| perf.benefit_achieved).sum();
            Some(total_benefit / matching_ops.len() as f64)
        }
    }

    /// Clear performance history
    pub fn clear_history(&mut self) {
        self.performance_history.clear();
    }

    /// Get the number of recorded performance samples
    pub fn history_size(&self) -> usize {
        self.performance_history.len()
    }

    /// Update learning rate for adaptive behavior
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.clamp(0.01, 1.0);
    }
}

impl Default for AdaptiveFusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_fusion_engine_creation() {
        let engine = AdaptiveFusionEngine::new();
        assert_eq!(engine.history_size(), 0);
        assert!((engine.learning_rate - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_performance_recording() {
        let mut engine = AdaptiveFusionEngine::new();

        let performance = FusionPerformance {
            operation: FusedOp::ReluAdd,
            tensor_size: 1000,
            execution_time: 0.001,
            memory_usage: 4000,
            benefit_achieved: 0.3,
        };

        engine.record_performance(performance);
        assert_eq!(engine.history_size(), 1);
    }

    #[test]
    fn test_benefit_prediction() {
        let mut engine = AdaptiveFusionEngine::new();

        // Record some performance data
        let performance = FusionPerformance {
            operation: FusedOp::ReluAdd,
            tensor_size: 1000,
            execution_time: 0.001,
            memory_usage: 4000,
            benefit_achieved: 0.3,
        };

        engine.record_performance(performance);

        // Predict benefit for similar operation
        let predicted = engine.predict_fusion_benefit(&FusedOp::ReluAdd, 950);
        assert!(predicted > 0.0);
        assert!(predicted <= 1.0);
    }

    #[test]
    fn test_adaptive_fusion_decision() {
        let mut engine = AdaptiveFusionEngine::new();

        // Record high-benefit performance
        let performance = FusionPerformance {
            operation: FusedOp::AddReluMul,
            tensor_size: 5000,
            execution_time: 0.002,
            memory_usage: 20000,
            benefit_achieved: 0.5,
        };

        engine.record_performance(performance);

        // Should recommend fusion for similar case
        assert!(engine.should_fuse_adaptive(&FusedOp::AddReluMul, 4800));
    }

    #[test]
    fn test_history_size_limit() {
        let mut engine = AdaptiveFusionEngine::new();

        // Add more than 100 performance records
        for i in 0..150 {
            let performance = FusionPerformance {
                operation: FusedOp::ReluAdd,
                tensor_size: 1000 + i,
                execution_time: 0.001,
                memory_usage: 4000,
                benefit_achieved: 0.2,
            };
            engine.record_performance(performance);
        }

        // Should be capped at 100
        assert_eq!(engine.history_size(), 100);
    }

    #[test]
    fn test_learning_rate_adjustment() {
        let mut engine = AdaptiveFusionEngine::new();

        engine.set_learning_rate(0.5);
        assert!((engine.learning_rate - 0.5).abs() < f64::EPSILON);

        // Test clamping
        engine.set_learning_rate(1.5);
        assert!((engine.learning_rate - 1.0).abs() < f64::EPSILON);

        engine.set_learning_rate(-0.1);
        assert!((engine.learning_rate - 0.01).abs() < f64::EPSILON);
    }
}
