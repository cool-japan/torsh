//! Core types and enums for operation fusion
//!
//! This module defines the fundamental types used for operation fusion,
//! including the FusedOp enum that represents different fusion patterns
//! and OpSequence for tracking operation chains.

use crate::TorshResult;

/// A fused operation that combines multiple element-wise operations
#[derive(Debug, Clone)]
pub enum FusedOp {
    /// Fused ReLU + Add: relu(x + y)
    ReluAdd,
    /// Fused Add + Multiply: (x + y) * z
    AddMul,
    /// Fused Multiply + Add: x * y + z (FMADD)
    MulAdd,
    /// Fused Sigmoid + Multiply: sigmoid(x) * y (SiLU when y = x)
    SigmoidMul,
    /// Fused Tanh + Scale: tanh(x) * scale
    TanhScale,
    /// Fused Add + ReLU + Multiply: relu(x + bias) * scale
    AddReluMul,
    /// Fused BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    BatchNormFused,
    /// Fused LayerNorm: similar to BatchNorm but along different dimensions
    LayerNormFused,
}

/// Represents a sequence of operations that can be fused
#[derive(Debug)]
pub struct OpSequence {
    pub operations: Vec<FusedOp>,
    pub input_count: usize,
    pub output_count: usize,
}

impl OpSequence {
    /// Create a new operation sequence
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            input_count: 0,
            output_count: 0,
        }
    }

    /// Add an operation to the sequence
    pub fn add_operation(&mut self, op: FusedOp) {
        self.operations.push(op);
    }

    /// Check if this sequence is fusible
    pub fn is_fusible(&self) -> bool {
        !self.operations.is_empty() && self.operations.len() <= 4
    }

    /// Estimate performance benefit of fusing this sequence
    pub fn fusion_benefit_estimate(&self) -> f32 {
        if self.operations.is_empty() {
            return 0.0;
        }

        // Base benefit from eliminating intermediate memory operations
        let memory_benefit = (self.operations.len() - 1) as f32 * 0.3;

        // Additional benefit for SIMD-friendly operations
        let simd_benefit = self
            .operations
            .iter()
            .map(|op| match op {
                FusedOp::ReluAdd | FusedOp::AddMul | FusedOp::MulAdd => 0.2,
                FusedOp::SigmoidMul | FusedOp::TanhScale => 0.15,
                _ => 0.1,
            })
            .sum::<f32>();

        memory_benefit + simd_benefit
    }
}

impl Default for OpSequence {
    fn default() -> Self {
        Self::new()
    }
}
