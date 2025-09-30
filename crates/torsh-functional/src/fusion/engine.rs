//! Basic fusion engine infrastructure and pattern detection
//!
//! This module contains the core OpFusionEngine and pattern detection
//! functionality for identifying fusible operation sequences.

use super::core::FusedOp;

/// Auto-detection and fusion of operation patterns
pub struct OpFusionEngine {
    pub enabled: bool,
    pub fusion_threshold: usize,
}

impl OpFusionEngine {
    pub fn new() -> Self {
        Self {
            enabled: true,
            fusion_threshold: 2, // Minimum number of operations to consider for fusion
        }
    }

    /// Analyze a sequence of operations and suggest fusion opportunities
    pub fn analyze_sequence(&self, ops: &[&str]) -> Vec<FusedOp> {
        let mut fused_ops = Vec::new();

        for window in ops.windows(2) {
            match window {
                ["add", "relu"] => fused_ops.push(FusedOp::ReluAdd),
                ["mul", "add"] => fused_ops.push(FusedOp::MulAdd),
                ["add", "mul"] => fused_ops.push(FusedOp::AddMul),
                ["sigmoid", "mul"] => fused_ops.push(FusedOp::SigmoidMul),
                ["tanh", "scale"] => fused_ops.push(FusedOp::TanhScale),
                _ => {}
            }
        }

        // Look for longer patterns
        for window in ops.windows(3) {
            match window {
                ["add", "relu", "mul"] => {
                    // Remove the individual operations if they were detected
                    fused_ops.retain(|op| !matches!(op, FusedOp::ReluAdd));
                    fused_ops.push(FusedOp::AddReluMul);
                }
                _ => {}
            }
        }

        fused_ops
    }

    /// Check if fusion is beneficial for the given operation sequence
    pub fn should_fuse(&self, ops: &[&str]) -> bool {
        self.enabled && ops.len() >= self.fusion_threshold
    }
}

impl Default for OpFusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern matcher for common operation sequences
pub fn detect_fusible_patterns(operations: &[&str]) -> Vec<(usize, FusedOp)> {
    let mut patterns = Vec::new();

    // Detect 2-operation patterns
    for (i, window) in operations.windows(2).enumerate() {
        match window {
            ["add", "relu"] => patterns.push((i, FusedOp::ReluAdd)),
            ["mul", "add"] => patterns.push((i, FusedOp::MulAdd)),
            ["add", "mul"] => patterns.push((i, FusedOp::AddMul)),
            ["sigmoid", "mul"] => patterns.push((i, FusedOp::SigmoidMul)),
            ["tanh", "scale"] => patterns.push((i, FusedOp::TanhScale)),
            _ => {}
        }
    }

    // Detect 3-operation patterns (and remove conflicting 2-op patterns)
    for (i, window) in operations.windows(3).enumerate() {
        match window {
            ["add", "relu", "mul"] => {
                // Remove conflicting 2-op patterns
                patterns.retain(|(pos, op)| {
                    !(*pos == i && matches!(op, FusedOp::ReluAdd))
                        && !(*pos == i + 1 && matches!(op, FusedOp::AddMul))
                });
                patterns.push((i, FusedOp::AddReluMul));
            }
            _ => {}
        }
    }

    patterns
}
