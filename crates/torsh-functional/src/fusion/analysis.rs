//! Advanced pattern detection and fusion opportunity analysis
//!
//! This module provides sophisticated analysis of operation sequences
//! to identify fusion opportunities with cost-benefit analysis.

use super::core::FusedOp;

/// Represents a fusion opportunity with cost-benefit analysis
#[derive(Debug)]
pub struct FusionOpportunity {
    pub position: usize,
    pub operation: FusedOp,
    pub expected_benefit: f64, // Percentage improvement (0.0 to 1.0)
    pub memory_savings: usize, // Bytes saved
}

/// Advanced pattern detection with cost-benefit analysis
pub fn analyze_fusion_opportunities(
    operations: &[&str],
    tensor_sizes: &[usize],
    memory_bandwidth: f64,
    compute_throughput: f64,
) -> Vec<FusionOpportunity> {
    let mut opportunities = Vec::new();

    // Analyze 2-op patterns
    for (i, window) in operations.windows(2).enumerate() {
        if let Some(op) = match window {
            ["add", "relu"] => Some(FusedOp::ReluAdd),
            ["mul", "add"] => Some(FusedOp::MulAdd),
            ["add", "mul"] => Some(FusedOp::AddMul),
            ["sigmoid", "mul"] => Some(FusedOp::SigmoidMul),
            ["tanh", "scale"] => Some(FusedOp::TanhScale),
            _ => None,
        } {
            let benefit = calculate_fusion_benefit(
                &op,
                tensor_sizes.get(i).copied().unwrap_or(0),
                memory_bandwidth,
                compute_throughput,
            );

            if benefit > 0.1 {
                // Only consider if benefit > 10%
                opportunities.push(FusionOpportunity {
                    position: i,
                    operation: op.clone(),
                    expected_benefit: benefit,
                    memory_savings: estimate_memory_savings(
                        &op,
                        tensor_sizes.get(i).copied().unwrap_or(0),
                    ),
                });
            }
        }
    }

    // Analyze 3-op patterns
    for (i, window) in operations.windows(3).enumerate() {
        if let Some(op) = match window {
            ["add", "relu", "mul"] => Some(FusedOp::AddReluMul),
            ["batch_norm", "relu", "dropout"] => Some(FusedOp::BatchNormFused),
            _ => None,
        } {
            let benefit = calculate_fusion_benefit(
                &op,
                tensor_sizes.get(i).copied().unwrap_or(0),
                memory_bandwidth,
                compute_throughput,
            );

            if benefit > 0.15 {
                // Higher threshold for 3-op patterns
                opportunities.push(FusionOpportunity {
                    position: i,
                    operation: op.clone(),
                    expected_benefit: benefit,
                    memory_savings: estimate_memory_savings(
                        &op,
                        tensor_sizes.get(i).copied().unwrap_or(0),
                    ),
                });
            }
        }
    }

    // Sort by expected benefit (descending)
    opportunities.sort_by(|a, b| {
        b.expected_benefit
            .partial_cmp(&a.expected_benefit)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    opportunities
}

/// Calculate the expected benefit of fusing an operation
fn calculate_fusion_benefit(
    op: &FusedOp,
    tensor_size: usize,
    memory_bandwidth: f64,
    compute_throughput: f64,
) -> f64 {
    let element_size = 4; // Assume f32
    let total_bytes = tensor_size * element_size;

    // Calculate memory access time savings
    // Unfused: each operation reads inputs and writes output, intermediate results add extra reads/writes
    let unfused_memory_accesses = match op {
        FusedOp::ReluAdd | FusedOp::AddMul | FusedOp::MulAdd => {
            // Two ops: first (2 reads + 1 write) + second (1 read + 1 write) = 4 reads + 2 writes = 6 total
            6.0
        }
        FusedOp::SigmoidMul => {
            // sigmoid (1 read + 1 write) + mul (2 reads + 1 write) = 3 reads + 2 writes = 5 total
            5.0
        }
        FusedOp::TanhScale => {
            // tanh (1 read + 1 write) + scale (1 read + 1 write) = 2 reads + 2 writes = 4 total
            4.0
        }
        FusedOp::AddReluMul => {
            // Three ops with intermediates: add + relu + mul = 8 total
            8.0
        }
        FusedOp::BatchNormFused => {
            // Complex multi-step normalization
            12.0
        }
        FusedOp::LayerNormFused => {
            // Complex multi-step normalization
            10.0
        }
    };

    // Fused: single kernel reads all inputs and writes final output
    let fused_memory_accesses = match op {
        FusedOp::ReluAdd | FusedOp::AddMul | FusedOp::MulAdd => 3.0, // 2 reads + 1 write
        FusedOp::SigmoidMul => 3.0,                                  // 2 reads + 1 write
        FusedOp::TanhScale => 2.0,                                   // 1 read + 1 write
        FusedOp::AddReluMul => 4.0,                                  // 3 reads + 1 write
        FusedOp::BatchNormFused => 6.0,                              // multiple reads + 1 write
        FusedOp::LayerNormFused => 5.0,                              // multiple reads + 1 write
    };

    let memory_time_savings =
        (unfused_memory_accesses - fused_memory_accesses) * total_bytes as f64 / memory_bandwidth;

    // Calculate compute time (relatively small for element-wise ops)
    let compute_time = total_bytes as f64 / compute_throughput;

    // Benefit is the ratio of time saved to total time
    if memory_time_savings + compute_time > 0.0 {
        memory_time_savings / (memory_time_savings + compute_time)
    } else {
        0.0
    }
}

/// Estimate memory savings from fusion
fn estimate_memory_savings(op: &FusedOp, tensor_size: usize) -> usize {
    let element_size = 4; // Assume f32

    match op {
        FusedOp::ReluAdd | FusedOp::AddMul | FusedOp::MulAdd => {
            // Save one intermediate tensor
            tensor_size * element_size
        }
        FusedOp::SigmoidMul => {
            // Save sigmoid intermediate result
            tensor_size * element_size
        }
        FusedOp::AddReluMul => {
            // Save two intermediate tensors
            tensor_size * element_size * 2
        }
        FusedOp::BatchNormFused | FusedOp::LayerNormFused => {
            // Save multiple intermediate tensors
            tensor_size * element_size * 3
        }
        _ => tensor_size * element_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_benefit_calculation() {
        let op = FusedOp::ReluAdd;
        let tensor_size = 1000;
        let memory_bandwidth = 100_000_000.0; // 100 MB/s
        let compute_throughput = 1_000_000_000.0; // 1 GFLOP/s

        let benefit =
            calculate_fusion_benefit(&op, tensor_size, memory_bandwidth, compute_throughput);
        assert!(benefit > 0.0);
        assert!(benefit < 1.0);
    }

    #[test]
    fn test_memory_savings_estimation() {
        let op = FusedOp::AddReluMul;
        let tensor_size = 1000;
        let savings = estimate_memory_savings(&op, tensor_size);

        // Should save 2 intermediate tensors
        assert_eq!(savings, tensor_size * 4 * 2);
    }

    #[test]
    fn test_fusion_opportunities() {
        let operations = ["add", "relu", "mul"];
        let tensor_sizes = [1000, 1000, 1000];
        let memory_bandwidth = 100_000_000.0;
        let compute_throughput = 1_000_000_000.0;

        let opportunities = analyze_fusion_opportunities(
            &operations,
            &tensor_sizes,
            memory_bandwidth,
            compute_throughput,
        );

        assert!(!opportunities.is_empty());

        // Should identify the add-relu-mul pattern
        assert!(opportunities
            .iter()
            .any(|opp| matches!(opp.operation, FusedOp::AddReluMul)));
    }
}
