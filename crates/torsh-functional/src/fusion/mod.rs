//! Operation fusion module for optimizing functional operation patterns
//!
//! This module provides functionality to identify and fuse common sequences of
//! tensor operations to improve performance and reduce memory overhead.
//!
//! # Module Organization
//!
//! - `core`: Core types and enums (FusedOp, OpSequence)
//! - `operations`: Individual fused operation implementations
//! - `engine`: Basic fusion engine infrastructure and pattern detection
//! - `analysis`: Advanced pattern analysis and cost-benefit evaluation
//! - `adaptive`: Adaptive fusion engine with performance learning
//!
//! # Examples
//!
//! ```rust
//! use torsh_functional::fusion::*;
//! use torsh_tensor::creation::ones;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     // Basic fused operations
//!     let x = ones(&[100])?;
//!     let y = ones(&[100])?;
//!     let result = fused_relu_add(&x, &y)?;  // relu(x + y)
//!
//!     // Pattern detection
//!     let ops = ["add", "relu", "mul"];
//!     let patterns = detect_fusible_patterns(&ops);
//!
//!     // Adaptive fusion engine
//!     let mut engine = AdaptiveFusionEngine::new();
//!     let should_fuse = engine.should_fuse_adaptive(&FusedOp::ReluAdd, 1000);
//!     Ok(())
//! }
//! ```

pub mod adaptive;
pub mod analysis;
pub mod core;
pub mod engine;
pub mod operations;

// Re-export public types and functions
pub use adaptive::{AdaptiveFusionEngine, FusionPerformance};
pub use analysis::{analyze_fusion_opportunities, FusionOpportunity};
pub use core::{FusedOp, OpSequence};
pub use engine::{detect_fusible_patterns, OpFusionEngine};
pub use operations::{
    fused_add_mul, fused_add_relu_mul, fused_batch_norm, fused_mul_add, fused_relu_add,
    fused_sigmoid_mul, fused_silu, fused_tanh_scale,
};

// Re-export for backwards compatibility

#[cfg(test)]
mod tests {
    use super::*;

    use torsh_tensor::creation::{ones, zeros};

    #[test]
    fn test_fused_relu_add() -> crate::TorshResult<()> {
        let x = ones(&[100])?;
        let y = ones(&[100])?;

        let result = fused_relu_add(&x, &y)?;
        let data = result.data()?;

        // Result should be all 2.0s (relu(1.0 + 1.0) = relu(2.0) = 2.0)
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[99] - 2.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_fused_mul_add() -> crate::TorshResult<()> {
        let x = ones(&[100])?;
        let y = ones(&[100])?;
        let z = ones(&[100])?;

        let result = fused_mul_add(&x, &y, &z)?;
        let data = result.data()?;

        // Result should be all 2.0s (1.0 * 1.0 + 1.0 = 2.0)
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[99] - 2.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_fused_add_mul() -> crate::TorshResult<()> {
        let x = ones(&[100])?;
        let y = ones(&[100])?;
        let z = ones(&[100])?;

        let result = fused_add_mul(&x, &y, &z)?;
        let data = result.data()?;

        // Result should be all 2.0s ((1.0 + 1.0) * 1.0 = 2.0)
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[99] - 2.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_fused_sigmoid_mul() -> crate::TorshResult<()> {
        let x = zeros(&[100])?;
        let y = ones(&[100])?;

        let result = fused_sigmoid_mul(&x, &y)?;
        let data = result.data()?;

        // sigmoid(0) = 0.5, so result should be 0.5 * 1.0 = 0.5
        assert!((data[0] - 0.5).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_fused_silu() -> crate::TorshResult<()> {
        let x = zeros(&[100])?;

        let result = fused_silu(&x)?;
        let data = result.data()?;

        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!(data[0].abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_fused_tanh_scale() -> crate::TorshResult<()> {
        let x = zeros(&[100])?;
        let scale = 2.0;

        let result = fused_tanh_scale(&x, scale)?;
        let data = result.data()?;

        // tanh(0) * 2.0 = 0 * 2.0 = 0
        assert!(data[0].abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_fused_add_relu_mul() -> crate::TorshResult<()> {
        let x = ones(&[100])?;
        let bias = ones(&[100])?;
        let scale = ones(&[100])?;

        let result = fused_add_relu_mul(&x, &bias, &scale)?;
        let data = result.data()?;

        // relu(1.0 + 1.0) * 1.0 = relu(2.0) * 1.0 = 2.0 * 1.0 = 2.0
        assert!((data[0] - 2.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_fused_batch_norm() -> crate::TorshResult<()> {
        let x = ones(&[100])?;
        let mean = zeros(&[100])?;
        let var = ones(&[100])?;
        let gamma = Some(&ones(&[100])?);
        let beta = Some(&zeros(&[100])?);
        let eps = 1e-5;

        let result = fused_batch_norm(&x, &mean, &var, gamma, beta, eps)?;
        let data = result.data()?;

        // (1.0 - 0.0) / sqrt(1.0 + 1e-5) * 1.0 + 0.0 ≈ 1.0
        assert!((data[0] - 1.0).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn test_fusion_engine() {
        let engine = OpFusionEngine::new();
        let ops = ["add", "relu"];

        let fused_ops = engine.analyze_sequence(&ops);
        assert_eq!(fused_ops.len(), 1);
        assert!(matches!(fused_ops[0], FusedOp::ReluAdd));
    }

    #[test]
    fn test_pattern_detection() {
        let operations = ["add", "relu", "mul"];
        let patterns = detect_fusible_patterns(&operations);

        assert!(!patterns.is_empty());
        // Should detect AddReluMul pattern
        assert!(patterns
            .iter()
            .any(|(_, op)| matches!(op, FusedOp::AddReluMul)));
    }

    #[test]
    fn test_fusion_opportunity_analysis() {
        let operations = ["add", "relu"];
        let tensor_sizes = [1000, 1000];
        let memory_bandwidth = 100_000_000.0; // 100 MB/s
        let compute_throughput = 1_000_000_000.0; // 1 GFLOP/s

        let opportunities = analyze_fusion_opportunities(
            &operations,
            &tensor_sizes,
            memory_bandwidth,
            compute_throughput,
        );

        assert!(!opportunities.is_empty());
        assert!(opportunities[0].expected_benefit > 0.0);
    }

    #[test]
    fn test_adaptive_engine_learning() {
        let mut engine = AdaptiveFusionEngine::new();

        // Record a high-performance fusion
        let performance = FusionPerformance {
            operation: FusedOp::ReluAdd,
            tensor_size: 1000,
            execution_time: 0.001,
            memory_usage: 4000,
            benefit_achieved: 0.4,
        };

        engine.record_performance(performance);

        // Engine should be optimistic about similar fusions
        let predicted_benefit = engine.predict_fusion_benefit(&FusedOp::ReluAdd, 950);
        assert!(predicted_benefit > 0.3);
    }

    #[test]
    fn test_op_sequence() {
        let mut sequence = OpSequence::new();
        assert!(sequence.operations.is_empty());
        assert_eq!(sequence.input_count, 0);
        assert_eq!(sequence.output_count, 0);

        sequence.add_operation(FusedOp::ReluAdd);
        assert_eq!(sequence.operations.len(), 1);

        assert!(sequence.is_fusible());

        let benefit = sequence.fusion_benefit_estimate();
        assert!(benefit > 0.0);
    }
}
