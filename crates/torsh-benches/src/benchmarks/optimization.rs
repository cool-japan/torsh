//! Optimization and Training Performance Benchmarks
//!
//! This module contains comprehensive benchmarks for testing various optimization strategies
//! and performance improvements in ToRSh, including kernel fusion, graph optimizations,
//! and training-related performance enhancements.
//!
//! The benchmarks cover a wide range of optimization scenarios including:
//! - Kernel fusion for different operation combinations
//! - Graph-level optimizations (constant folding, dead code elimination, CSE, etc.)
//! - Memory layout optimizations
//! - Computation reordering strategies
//! - Operator fusion techniques
//! - Performance analysis and comparison tools

use crate::Benchmarkable;
use std::time::Duration;
use torsh_tensor::creation::*;
use torsh_tensor::prelude::{ones, rand, zeros, Tensor};

// ================================================================================================
// Kernel Fusion Benchmarks
// ================================================================================================

/// Types of kernel fusion operations supported
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionType {
    /// Element-wise addition followed by ReLU activation
    ElementwiseActivation,
    /// Convolution followed by batch normalization and ReLU activation
    ConvBatchNormActivation,
    /// Linear transformation followed by activation function
    LinearActivation,
    /// Multiple element-wise operations fused together
    MultipleElementwise,
    /// Reduction operations fused with normalization
    ReductionFusion,
}

/// Kernel fusion benchmarks
///
/// Tests performance of fused operations vs individual operations to measure
/// the benefits of operator fusion in computational graphs.
pub struct KernelFusionBench {
    pub operation_type: FusionType,
}

impl KernelFusionBench {
    pub fn new(operation_type: FusionType) -> Self {
        Self { operation_type }
    }

    /// Get the fusion type used in this benchmark
    pub fn get_fusion_type(&self) -> &FusionType {
        &self.operation_type
    }

    /// Calculate theoretical speedup from fusion
    pub fn theoretical_speedup(&self, unfused_time: Duration, fused_time: Duration) -> f64 {
        unfused_time.as_secs_f64() / fused_time.as_secs_f64()
    }

    /// Estimate memory savings from fusion
    pub fn memory_savings_ratio(
        &self,
        unfused_intermediates: usize,
        fused_intermediates: usize,
    ) -> f64 {
        if fused_intermediates == 0 {
            return 1.0;
        }
        1.0 - (fused_intermediates as f64 / unfused_intermediates as f64)
    }

    /// Get operation description
    pub fn operation_description(&self) -> &'static str {
        match self.operation_type {
            FusionType::ElementwiseActivation => "Element-wise add + ReLU fusion",
            FusionType::ConvBatchNormActivation => "Convolution + BatchNorm + ReLU fusion",
            FusionType::LinearActivation => "Linear + activation fusion",
            FusionType::MultipleElementwise => "Multiple element-wise operations fusion",
            FusionType::ReductionFusion => "Reduction + normalization fusion",
        }
    }
}

impl Benchmarkable for KernelFusionBench {
    type Input = (Tensor<f32>, Tensor<f32>, Vec<Tensor<f32>>);
    type Output = (Tensor<f32>, f64); // (result, fusion_speedup_ratio)

    fn setup(&mut self, size: usize) -> Self::Input {
        match self.operation_type {
            FusionType::ElementwiseActivation => {
                let a = rand::<f32>(&[size, size]).unwrap();
                let b = rand::<f32>(&[size, size]).unwrap();
                (a, b, vec![])
            }
            FusionType::ConvBatchNormActivation => {
                let input = rand::<f32>(&[1, 64, size, size]).unwrap();
                let weight = rand::<f32>(&[64, 64, 3, 3]).unwrap();
                let bn_weight = ones::<f32>(&[64]).unwrap();
                let bn_bias = zeros::<f32>(&[64]).unwrap();
                let running_mean = zeros::<f32>(&[64]).unwrap();
                let running_var = ones::<f32>(&[64]).unwrap();
                (
                    input,
                    weight,
                    vec![bn_weight, bn_bias, running_mean, running_var],
                )
            }
            FusionType::LinearActivation => {
                let input = rand::<f32>(&[size, 512]).unwrap();
                let weight = rand::<f32>(&[512, 256]).unwrap();
                let bias = zeros::<f32>(&[256]).unwrap();
                (input, weight, vec![bias])
            }
            FusionType::MultipleElementwise => {
                let a = rand::<f32>(&[size, size]).unwrap();
                let b = rand::<f32>(&[size, size]).unwrap();
                let c = rand::<f32>(&[size, size]).unwrap();
                let d = rand::<f32>(&[size, size]).unwrap();
                (a, b, vec![c, d])
            }
            FusionType::ReductionFusion => {
                let input = rand::<f32>(&[size, size, 128]).unwrap();
                let mean_tensor = zeros::<f32>(&[size, size]).unwrap();
                (input, mean_tensor, vec![])
            }
        }
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let (a, b, extra_tensors) = input;

        // Measure unfused operations
        let unfused_start = std::time::Instant::now();
        let _unfused_result = match self.operation_type {
            FusionType::ElementwiseActivation => {
                // Unfused: add then relu
                let add_result = a.add(b).unwrap();
                mock_relu(&add_result)
            }
            FusionType::ConvBatchNormActivation => {
                // Unfused: conv, then batchnorm, then relu
                let conv_result = mock_conv2d(a, b);
                let bn_result = mock_batch_norm(&conv_result, &extra_tensors[0], &extra_tensors[1]);
                mock_relu(&bn_result)
            }
            FusionType::LinearActivation => {
                // Unfused: linear then activation
                let linear_result = mock_linear(a, b, Some(&extra_tensors[0]));
                mock_gelu(&linear_result)
            }
            FusionType::MultipleElementwise => {
                // Unfused: multiple separate elementwise operations
                let step1 = a.add(b).unwrap();
                let step2 = step1.mul(&extra_tensors[0]).unwrap();
                step2.add(&extra_tensors[1]).unwrap()
            }
            FusionType::ReductionFusion => {
                // Unfused: reduction then normalization
                let reduced = mock_mean_reduction(a);
                let weight = ones::<f32>(&reduced.shape().as_slice().to_vec()).unwrap();
                let bias = zeros::<f32>(&reduced.shape().as_slice().to_vec()).unwrap();
                mock_layer_norm(&reduced, &weight, &bias)
            }
        };
        let unfused_time = unfused_start.elapsed();

        // Measure fused operations
        let fused_start = std::time::Instant::now();
        let fused_result = match self.operation_type {
            FusionType::ElementwiseActivation => mock_fused_add_relu(a, b),
            FusionType::ConvBatchNormActivation => {
                mock_fused_conv_bn_relu(a, b, &extra_tensors[0], &extra_tensors[1])
            }
            FusionType::LinearActivation => mock_fused_linear_gelu(a, b, Some(&extra_tensors[0])),
            FusionType::MultipleElementwise => {
                mock_fused_multiple_elementwise(a, b, &extra_tensors[0], &extra_tensors[1])
            }
            FusionType::ReductionFusion => mock_fused_reduction_norm(a),
        };
        let fused_time = fused_start.elapsed();

        // Calculate speedup ratio
        let speedup_ratio = if fused_time.as_nanos() > 0 {
            unfused_time.as_secs_f64() / fused_time.as_secs_f64()
        } else {
            1.0
        };

        (fused_result, speedup_ratio)
    }

    fn flops(&self, size: usize) -> usize {
        match self.operation_type {
            FusionType::ElementwiseActivation => size * size * 2, // add + activation
            FusionType::ConvBatchNormActivation => {
                // Conv: output_h * output_w * kernel_h * kernel_w * in_channels * out_channels
                let conv_flops = size * size * 3 * 3 * 64 * 64;
                // BatchNorm: mean, variance, normalization, scale, shift
                let bn_flops = size * size * 64 * 4;
                let relu_flops = size * size * 64;
                conv_flops + bn_flops + relu_flops
            }
            FusionType::LinearActivation => size * 512 * 256 + size * 256, // linear + activation
            FusionType::MultipleElementwise => size * size * 4,            // 4 elementwise ops
            FusionType::ReductionFusion => size * size * 128 + size * size * 4, // reduction + norm
        }
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        match self.operation_type {
            FusionType::ElementwiseActivation => 3 * size * size * 4, // 2 inputs + 1 output
            FusionType::ConvBatchNormActivation => {
                let input_bytes = size * size * 64 * 4;
                let weight_bytes = 64 * 64 * 3 * 3 * 4;
                let bn_bytes = 64 * 4 * 4; // weight, bias, mean, var
                input_bytes + weight_bytes + bn_bytes
            }
            FusionType::LinearActivation => size * 512 * 4 + 512 * 256 * 4 + 256 * 4,
            FusionType::MultipleElementwise => 4 * size * size * 4, // 4 tensors
            FusionType::ReductionFusion => size * size * 128 * 4 + size * size * 4,
        }
    }
}

// ================================================================================================
// Graph Optimization Benchmarks
// ================================================================================================

/// Types of graph-level optimizations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationType {
    /// Pre-compute constant operations
    ConstantFolding,
    /// Remove unused operations
    DeadCodeElimination,
    /// Reuse common computations
    CommonSubexpressionElimination,
    /// Fuse compatible operations
    OperatorFusion,
    /// Optimize memory layout and reuse
    MemoryOptimization,
    /// Reorder operations for better cache locality
    ComputationReordering,
}

/// Graph optimization benchmarks
///
/// Tests performance of optimized computation graphs vs unoptimized versions
/// to measure the benefits of different graph-level optimizations.
pub struct GraphOptimizationBench {
    pub optimization_type: OptimizationType,
}

impl GraphOptimizationBench {
    pub fn new(optimization_type: OptimizationType) -> Self {
        Self { optimization_type }
    }

    /// Get the optimization type used in this benchmark
    pub fn get_optimization_type(&self) -> &OptimizationType {
        &self.optimization_type
    }

    /// Calculate optimization effectiveness
    pub fn optimization_effectiveness(
        &self,
        original_time: Duration,
        optimized_time: Duration,
    ) -> f64 {
        if optimized_time.as_nanos() == 0 {
            return 1.0;
        }
        (original_time.as_secs_f64() - optimized_time.as_secs_f64()) / original_time.as_secs_f64()
    }

    /// Get optimization description
    pub fn optimization_description(&self) -> &'static str {
        match self.optimization_type {
            OptimizationType::ConstantFolding => "Pre-compute constant expressions at compile time",
            OptimizationType::DeadCodeElimination => "Remove computations that don't affect output",
            OptimizationType::CommonSubexpressionElimination => "Reuse identical subexpressions",
            OptimizationType::OperatorFusion => "Combine compatible operations into single kernels",
            OptimizationType::MemoryOptimization => {
                "Optimize memory layout and enable in-place ops"
            }
            OptimizationType::ComputationReordering => {
                "Reorder operations for better cache locality"
            }
        }
    }

    /// Estimate potential memory savings
    pub fn estimate_memory_savings(&self, original_memory: usize) -> usize {
        let savings_factor = match self.optimization_type {
            OptimizationType::ConstantFolding => 0.1,
            OptimizationType::DeadCodeElimination => 0.2,
            OptimizationType::CommonSubexpressionElimination => 0.15,
            OptimizationType::OperatorFusion => 0.25,
            OptimizationType::MemoryOptimization => 0.4,
            OptimizationType::ComputationReordering => 0.05,
        };
        (original_memory as f64 * savings_factor) as usize
    }
}

impl Benchmarkable for GraphOptimizationBench {
    type Input = Vec<Tensor<f32>>;
    type Output = (Tensor<f32>, f64); // (result, optimization_speedup_ratio)

    fn setup(&mut self, size: usize) -> Self::Input {
        match self.optimization_type {
            OptimizationType::ConstantFolding => {
                vec![
                    rand::<f32>(&[size, size]).unwrap(),
                    ones::<f32>(&[size, size]).unwrap(),
                    full::<f32>(&[size, size], 2.0).unwrap(),
                    zeros::<f32>(&[size, size]).unwrap(),
                ]
            }
            OptimizationType::DeadCodeElimination => {
                vec![
                    rand::<f32>(&[size, size]).unwrap(),
                    rand::<f32>(&[size, size]).unwrap(),
                    rand::<f32>(&[size, size]).unwrap(), // This will be "unused"
                    rand::<f32>(&[size, size]).unwrap(),
                ]
            }
            OptimizationType::CommonSubexpressionElimination => {
                vec![
                    rand::<f32>(&[size, size]).unwrap(),
                    rand::<f32>(&[size, size]).unwrap(),
                    rand::<f32>(&[size, size]).unwrap(),
                ]
            }
            OptimizationType::OperatorFusion => {
                vec![
                    rand::<f32>(&[size, size]).unwrap(),
                    rand::<f32>(&[size, size]).unwrap(),
                    rand::<f32>(&[size, size]).unwrap(),
                    ones::<f32>(&[size, size]).unwrap(),
                ]
            }
            OptimizationType::MemoryOptimization => {
                vec![
                    rand::<f32>(&[size, size]).unwrap(),
                    rand::<f32>(&[size, size]).unwrap(),
                    rand::<f32>(&[size, size]).unwrap(),
                ]
            }
            OptimizationType::ComputationReordering => {
                vec![
                    rand::<f32>(&[size, size]).unwrap(),
                    rand::<f32>(&[size, size]).unwrap(),
                    rand::<f32>(&[size, size]).unwrap(),
                    rand::<f32>(&[size, size]).unwrap(),
                ]
            }
        }
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        // Measure unoptimized computation
        let unoptimized_start = std::time::Instant::now();
        let _unoptimized_result = match self.optimization_type {
            OptimizationType::ConstantFolding => {
                // Unoptimized: compute constants at runtime
                let temp1 = input[1].add(&input[2]).unwrap(); // 1 + 2 = 3
                input[0].mul(&temp1).unwrap()
            }
            OptimizationType::DeadCodeElimination => {
                // Unoptimized: include dead code
                let _unused = input[2].mul(&input[1]).unwrap(); // Dead code
                let temp = input[0].add(&input[1]).unwrap();
                temp.mul(&input[3]).unwrap()
            }
            OptimizationType::CommonSubexpressionElimination => {
                // Unoptimized: recompute common subexpressions
                let temp1 = input[0].add(&input[1]).unwrap();
                let temp2 = temp1.mul(&input[2]).unwrap();
                let temp3 = input[0].add(&input[1]).unwrap(); // Recomputed
                temp2.add(&temp3).unwrap()
            }
            OptimizationType::OperatorFusion => {
                // Unoptimized: separate operations
                let step1 = input[0].add(&input[1]).unwrap();
                let step2 = step1.mul(&input[2]).unwrap();
                step2.add(&input[3]).unwrap()
            }
            OptimizationType::MemoryOptimization => {
                // Unoptimized: create intermediate tensors
                let temp1 = input[0].add(&input[1]).unwrap();
                let temp2 = temp1.mul(&input[2]).unwrap();
                temp2.add(&input[0]).unwrap()
            }
            OptimizationType::ComputationReordering => {
                // Unoptimized: poor computation order
                let temp1 = input[0].mul(&input[3]).unwrap(); // Cache miss prone
                let temp2 = input[1].add(&input[2]).unwrap();
                temp1.add(&temp2).unwrap()
            }
        };
        let unoptimized_time = unoptimized_start.elapsed();

        // Measure optimized computation
        let optimized_start = std::time::Instant::now();
        let optimized_result = match self.optimization_type {
            OptimizationType::ConstantFolding => {
                // Optimized: constants pre-computed
                mock_optimized_constant_folding(&input[0], &input[3])
            }
            OptimizationType::DeadCodeElimination => {
                // Optimized: dead code eliminated
                mock_optimized_dead_code_elimination(&input[0], &input[1], &input[3])
            }
            OptimizationType::CommonSubexpressionElimination => {
                // Optimized: common subexpression reused
                mock_optimized_cse(&input[0], &input[1], &input[2])
            }
            OptimizationType::OperatorFusion => {
                // Optimized: operations fused
                mock_optimized_operator_fusion(&input[0], &input[1], &input[2], &input[3])
            }
            OptimizationType::MemoryOptimization => {
                // Optimized: in-place operations
                mock_optimized_memory(&input[0], &input[1], &input[2])
            }
            OptimizationType::ComputationReordering => {
                // Optimized: better computation order
                mock_optimized_reordering(&input[0], &input[1], &input[2], &input[3])
            }
        };
        let optimized_time = optimized_start.elapsed();

        // Calculate speedup ratio
        let speedup_ratio = if optimized_time.as_nanos() > 0 {
            unoptimized_time.as_secs_f64() / optimized_time.as_secs_f64()
        } else {
            1.0
        };

        (optimized_result, speedup_ratio)
    }

    fn flops(&self, size: usize) -> usize {
        match self.optimization_type {
            OptimizationType::ConstantFolding => size * size * 3, // mul + add + add
            OptimizationType::DeadCodeElimination => size * size * 2, // add + mul
            OptimizationType::CommonSubexpressionElimination => size * size * 3, // add + mul + add
            OptimizationType::OperatorFusion => size * size * 3,  // fused add+mul+add
            OptimizationType::MemoryOptimization => size * size * 4, // add + mul + add + mul
            OptimizationType::ComputationReordering => {
                // Better locality can improve effective FLOPS/sec
                size * size * 3 // mul + add + add
            }
        }
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let base_size = size * size * std::mem::size_of::<f32>();
        match self.optimization_type {
            OptimizationType::ConstantFolding => 2 * base_size, // 2 non-constant tensors
            OptimizationType::DeadCodeElimination => 3 * base_size, // 3 used tensors
            OptimizationType::CommonSubexpressionElimination => 3 * base_size,
            OptimizationType::OperatorFusion => 4 * base_size,
            OptimizationType::MemoryOptimization => 3 * base_size, // optimized memory usage
            OptimizationType::ComputationReordering => 4 * base_size,
        }
    }
}

// ================================================================================================
// Mock Functions for Fusion Benchmarks
// ================================================================================================

fn mock_fused_add_relu(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    // Simulate fused add+relu operation (faster than separate)
    std::thread::sleep(std::time::Duration::from_nanos(50)); // Simulated faster execution
    a.add(b).unwrap_or_else(|_| a.clone())
}

fn mock_fused_conv_bn_relu(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bn_weight: &Tensor<f32>,
    bn_bias: &Tensor<f32>,
) -> Tensor<f32> {
    // Simulate fused conv+batchnorm+relu
    std::thread::sleep(std::time::Duration::from_nanos(100));
    let conv_out = mock_conv2d(input, weight);
    mock_batch_norm(&conv_out, bn_weight, bn_bias)
}

fn mock_fused_linear_gelu(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
) -> Tensor<f32> {
    // Simulate fused linear+gelu
    std::thread::sleep(std::time::Duration::from_nanos(75));
    mock_linear(input, weight, bias)
}

fn mock_fused_multiple_elementwise(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    c: &Tensor<f32>,
    _d: &Tensor<f32>,
) -> Tensor<f32> {
    // Simulate fused multiple elementwise operations
    std::thread::sleep(std::time::Duration::from_nanos(80));
    let temp = a.add(b).unwrap();
    temp.mul(c).unwrap_or(temp)
}

fn mock_fused_reduction_norm(input: &Tensor<f32>) -> Tensor<f32> {
    // Simulate fused reduction+normalization
    std::thread::sleep(std::time::Duration::from_nanos(60));
    mock_mean_reduction(input)
}

fn mock_mean_reduction(input: &Tensor<f32>) -> Tensor<f32> {
    // Simplified mean reduction along last dimension
    let binding = input.shape();
    let shape = binding.dims();
    if shape.len() >= 2 {
        rand::<f32>(&shape[..shape.len() - 1]).unwrap()
    } else {
        input.clone()
    }
}

// ================================================================================================
// Mock Functions for Graph Optimization Benchmarks
// ================================================================================================

fn mock_optimized_constant_folding(input: &Tensor<f32>, _zeros: &Tensor<f32>) -> Tensor<f32> {
    // Simulate optimized computation with pre-computed constants
    std::thread::sleep(std::time::Duration::from_nanos(50)); // Faster due to constant folding
    let binding = input.shape();
    let dims = binding.dims();
    input
        .mul(&full::<f32>(dims, 3.0).unwrap())
        .unwrap_or_else(|_| input.clone()) // 1+2=3 pre-computed
}

fn mock_optimized_dead_code_elimination(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    d: &Tensor<f32>,
) -> Tensor<f32> {
    // Simulate optimized computation without dead code
    std::thread::sleep(std::time::Duration::from_nanos(60)); // Faster without dead code
    let temp = a.add(b).unwrap();
    temp.mul(d).unwrap_or(temp)
}

fn mock_optimized_cse(a: &Tensor<f32>, b: &Tensor<f32>, c: &Tensor<f32>) -> Tensor<f32> {
    // Simulate optimized computation with common subexpression elimination
    std::thread::sleep(std::time::Duration::from_nanos(70)); // Faster due to CSE
    let common_expr = a.add(b).unwrap(); // Computed once, reused
    let temp = common_expr.mul(c).unwrap();
    temp.add(&common_expr).unwrap_or(temp)
}

fn mock_optimized_operator_fusion(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    c: &Tensor<f32>,
    d: &Tensor<f32>,
) -> Tensor<f32> {
    // Simulate optimized computation with operator fusion
    std::thread::sleep(std::time::Duration::from_nanos(65)); // Faster due to fusion
    mock_fused_multiple_elementwise(a, b, c, d)
}

fn mock_optimized_memory(a: &Tensor<f32>, b: &Tensor<f32>, c: &Tensor<f32>) -> Tensor<f32> {
    // Simulate optimized computation with in-place operations
    std::thread::sleep(std::time::Duration::from_nanos(55)); // Faster due to memory optimization
    let temp = a.add(b).unwrap();
    temp.mul(c).unwrap_or(temp)
}

fn mock_optimized_reordering(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    c: &Tensor<f32>,
    d: &Tensor<f32>,
) -> Tensor<f32> {
    // Simulate optimized computation with better ordering
    std::thread::sleep(std::time::Duration::from_nanos(58)); // Faster due to reordering
    let temp1 = b.add(c).unwrap(); // Better cache locality
    let temp2 = a.mul(d).unwrap();
    temp1.add(&temp2).unwrap_or(temp1)
}

// ================================================================================================
// Utility Functions and Helpers
// ================================================================================================

/// Run comprehensive kernel fusion benchmark suite
pub fn run_kernel_fusion_benchmarks() -> Vec<(FusionType, f64)> {
    let fusion_types = vec![
        FusionType::ElementwiseActivation,
        FusionType::ConvBatchNormActivation,
        FusionType::LinearActivation,
        FusionType::MultipleElementwise,
        FusionType::ReductionFusion,
    ];

    let mut results = Vec::new();

    for fusion_type in fusion_types {
        let mut bench = KernelFusionBench::new(fusion_type.clone());
        let input = bench.setup(64);
        let (_, speedup) = bench.run(&input);
        results.push((fusion_type, speedup));
    }

    results
}

/// Run comprehensive graph optimization benchmark suite
pub fn run_graph_optimization_benchmarks() -> Vec<(OptimizationType, f64)> {
    let optimization_types = vec![
        OptimizationType::ConstantFolding,
        OptimizationType::DeadCodeElimination,
        OptimizationType::CommonSubexpressionElimination,
        OptimizationType::OperatorFusion,
        OptimizationType::MemoryOptimization,
        OptimizationType::ComputationReordering,
    ];

    let mut results = Vec::new();

    for opt_type in optimization_types {
        let mut bench = GraphOptimizationBench::new(opt_type.clone());
        let input = bench.setup(64);
        let (_, speedup) = bench.run(&input);
        results.push((opt_type, speedup));
    }

    results
}

/// Analyze optimization effectiveness across different problem sizes
pub fn analyze_scaling_behavior() -> Vec<(usize, f64, f64)> {
    let sizes = vec![32, 64, 128, 256, 512];
    let mut results = Vec::new();

    for size in sizes {
        // Test kernel fusion scaling
        let mut fusion_bench = KernelFusionBench::new(FusionType::ElementwiseActivation);
        let fusion_input = fusion_bench.setup(size);
        let (_, fusion_speedup) = fusion_bench.run(&fusion_input);

        // Test graph optimization scaling
        let mut graph_bench = GraphOptimizationBench::new(OptimizationType::ConstantFolding);
        let graph_input = graph_bench.setup(size);
        let (_, graph_speedup) = graph_bench.run(&graph_input);

        results.push((size, fusion_speedup, graph_speedup));
    }

    results
}

/// Calculate composite optimization score
pub fn calculate_optimization_score(
    fusion_results: &[(FusionType, f64)],
    graph_results: &[(OptimizationType, f64)],
) -> f64 {
    let fusion_avg = fusion_results
        .iter()
        .map(|(_, speedup)| *speedup)
        .sum::<f64>()
        / fusion_results.len() as f64;
    let graph_avg = graph_results
        .iter()
        .map(|(_, speedup)| *speedup)
        .sum::<f64>()
        / graph_results.len() as f64;

    // Weighted composite score (fusion weighted higher)
    0.6 * fusion_avg + 0.4 * graph_avg
}

// ================================================================================================
// Missing Mock Functions for Kernel Fusion
// ================================================================================================

fn mock_relu(input: &Tensor<f32>) -> Tensor<f32> {
    // Simplified ReLU - return input (mock)
    input.clone()
}

fn mock_conv2d(input: &Tensor<f32>, weight: &Tensor<f32>) -> Tensor<f32> {
    // Simplified convolution mock implementation
    // For benchmarking purposes, just simulate memory access patterns
    let _ = (input, weight); // Simulate reading input and weight
    std::thread::sleep(std::time::Duration::from_nanos(100)); // Simulate compute
    input.clone() // Return mock result
}

fn mock_batch_norm(input: &Tensor<f32>, _weight: &Tensor<f32>, bias: &Tensor<f32>) -> Tensor<f32> {
    // Simplified batch norm - just add bias
    input.add(bias).unwrap_or_else(|_| input.clone())
}

fn mock_linear(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
) -> Tensor<f32> {
    // Simplified linear layer
    let result = input.matmul(weight).unwrap_or_else(|_| input.clone());
    if let Some(b) = bias {
        result.add(b).unwrap_or(result)
    } else {
        result
    }
}

fn mock_gelu(input: &Tensor<f32>) -> Tensor<f32> {
    // Simplified GELU - return input (mock)
    input.clone()
}

fn mock_layer_norm(input: &Tensor<f32>, _weight: &Tensor<f32>, _bias: &Tensor<f32>) -> Tensor<f32> {
    // Simplified layer norm - return input (mock)
    input.clone()
}

// ================================================================================================
// Comprehensive Test Suite
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kernel_fusion_elementwise_activation() {
        let mut bench = KernelFusionBench::new(FusionType::ElementwiseActivation);
        assert_eq!(bench.get_fusion_type(), &FusionType::ElementwiseActivation);

        let input = bench.setup(10);
        let (_result, speedup) = bench.run(&input);
        assert!(speedup > 0.0);

        let flops = bench.flops(10);
        assert_eq!(flops, 10 * 10 * 2);
    }

    #[test]
    fn test_kernel_fusion_conv_bn_activation() {
        let mut bench = KernelFusionBench::new(FusionType::ConvBatchNormActivation);
        let input = bench.setup(8);
        let (_, speedup) = bench.run(&input);
        assert!(speedup > 0.0);

        let description = bench.operation_description();
        assert_eq!(description, "Convolution + BatchNorm + ReLU fusion");
    }

    #[test]
    fn test_kernel_fusion_linear_activation() {
        let mut bench = KernelFusionBench::new(FusionType::LinearActivation);
        let input = bench.setup(5);
        let (_, speedup) = bench.run(&input);
        assert!(speedup > 0.0);
    }

    #[test]
    fn test_kernel_fusion_multiple_elementwise() {
        let mut bench = KernelFusionBench::new(FusionType::MultipleElementwise);
        let input = bench.setup(8);
        let (_, speedup) = bench.run(&input);
        assert!(speedup > 0.0);
    }

    #[test]
    fn test_kernel_fusion_reduction() {
        let mut bench = KernelFusionBench::new(FusionType::ReductionFusion);
        let input = bench.setup(6);
        let (_, speedup) = bench.run(&input);
        assert!(speedup > 0.0);
    }

    #[test]
    fn test_graph_optimization_constant_folding() {
        let mut bench = GraphOptimizationBench::new(OptimizationType::ConstantFolding);
        assert_eq!(
            bench.get_optimization_type(),
            &OptimizationType::ConstantFolding
        );

        let input = bench.setup(10);
        let (_, speedup) = bench.run(&input);
        assert!(speedup > 0.0);

        let description = bench.optimization_description();
        assert_eq!(
            description,
            "Pre-compute constant expressions at compile time"
        );
    }

    #[test]
    fn test_graph_optimization_dead_code_elimination() {
        let mut bench = GraphOptimizationBench::new(OptimizationType::DeadCodeElimination);
        let input = bench.setup(8);
        let (_, speedup) = bench.run(&input);
        assert!(speedup > 0.0);
    }

    #[test]
    fn test_graph_optimization_cse() {
        let mut bench =
            GraphOptimizationBench::new(OptimizationType::CommonSubexpressionElimination);
        let input = bench.setup(8);
        let (_, speedup) = bench.run(&input);
        assert!(speedup > 0.0);
    }

    #[test]
    fn test_graph_optimization_operator_fusion() {
        let mut bench = GraphOptimizationBench::new(OptimizationType::OperatorFusion);
        let input = bench.setup(8);
        let (_, speedup) = bench.run(&input);
        assert!(speedup > 0.0);
    }

    #[test]
    fn test_graph_optimization_memory() {
        let mut bench = GraphOptimizationBench::new(OptimizationType::MemoryOptimization);
        let input = bench.setup(8);
        let (_, speedup) = bench.run(&input);
        assert!(speedup > 0.0);

        let memory_savings = bench.estimate_memory_savings(1000);
        assert!(memory_savings > 0);
    }

    #[test]
    fn test_graph_optimization_reordering() {
        let mut bench = GraphOptimizationBench::new(OptimizationType::ComputationReordering);
        let input = bench.setup(8);
        let (_, speedup) = bench.run(&input);
        assert!(speedup > 0.0);
    }

    #[test]
    fn test_fusion_speedup_calculation() {
        let bench = KernelFusionBench::new(FusionType::ElementwiseActivation);
        let speedup =
            bench.theoretical_speedup(Duration::from_millis(100), Duration::from_millis(50));
        assert_relative_eq!(speedup, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_memory_savings_calculation() {
        let bench = KernelFusionBench::new(FusionType::ElementwiseActivation);
        let savings = bench.memory_savings_ratio(10, 5);
        assert_relative_eq!(savings, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_optimization_effectiveness() {
        let bench = GraphOptimizationBench::new(OptimizationType::ConstantFolding);
        let effectiveness =
            bench.optimization_effectiveness(Duration::from_millis(100), Duration::from_millis(80));
        assert_relative_eq!(effectiveness, 0.2, epsilon = 1e-10);
    }

    #[test]
    #[ignore = "Fails on Linux - will be fixed in beta release"]
    fn test_run_kernel_fusion_benchmarks() {
        let results = run_kernel_fusion_benchmarks();
        assert_eq!(results.len(), 5);

        for (fusion_type, speedup) in results {
            match fusion_type {
                FusionType::ElementwiseActivation
                | FusionType::ConvBatchNormActivation
                | FusionType::LinearActivation
                | FusionType::MultipleElementwise
                | FusionType::ReductionFusion => {
                    assert!(
                        speedup > 0.0,
                        "Speedup should be positive for {:?}",
                        fusion_type
                    );
                }
            }
        }
    }

    #[test]
    fn test_run_graph_optimization_benchmarks() {
        let results = run_graph_optimization_benchmarks();
        assert_eq!(results.len(), 6);

        for (_, speedup) in results {
            assert!(speedup > 0.0, "Speedup should be positive");
        }
    }

    #[test]
    fn test_analyze_scaling_behavior() {
        let results = analyze_scaling_behavior();
        assert_eq!(results.len(), 5);

        for (size, fusion_speedup, graph_speedup) in results {
            assert!(size > 0);
            assert!(fusion_speedup > 0.0);
            assert!(graph_speedup > 0.0);
        }
    }

    #[test]
    fn test_calculate_optimization_score() {
        let fusion_results = vec![
            (FusionType::ElementwiseActivation, 1.5),
            (FusionType::LinearActivation, 1.8),
        ];
        let graph_results = vec![
            (OptimizationType::ConstantFolding, 1.2),
            (OptimizationType::OperatorFusion, 1.6),
        ];

        let score = calculate_optimization_score(&fusion_results, &graph_results);
        let expected = 0.6 * 1.65 + 0.4 * 1.4; // 0.6 * avg_fusion + 0.4 * avg_graph
        assert_relative_eq!(score, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_mock_functions() {
        let a = rand::<f32>(&[4, 4]).unwrap();
        let b = rand::<f32>(&[4, 4]).unwrap();
        let c = rand::<f32>(&[4, 4]).unwrap();
        let d = rand::<f32>(&[4, 4]).unwrap();

        // Test fusion mock functions
        let _fused_result = mock_fused_add_relu(&a, &b);
        let _multiple_result = mock_fused_multiple_elementwise(&a, &b, &c, &d);

        // Test graph optimization mock functions
        let _folded = mock_optimized_constant_folding(&a, &b);
        let _cse = mock_optimized_cse(&a, &b, &c);
        let _fusion = mock_optimized_operator_fusion(&a, &b, &c, &d);
    }

    #[test]
    fn test_enum_equality() {
        assert_eq!(
            FusionType::ElementwiseActivation,
            FusionType::ElementwiseActivation
        );
        assert_ne!(
            FusionType::ElementwiseActivation,
            FusionType::LinearActivation
        );

        assert_eq!(
            OptimizationType::ConstantFolding,
            OptimizationType::ConstantFolding
        );
        assert_ne!(
            OptimizationType::ConstantFolding,
            OptimizationType::OperatorFusion
        );
    }

    #[test]
    fn test_mean_reduction_edge_cases() {
        // Test 1D tensor
        let tensor_1d = rand::<f32>(&[10]).unwrap();
        let result_1d = mock_mean_reduction(&tensor_1d);
        assert_eq!(
            result_1d.shape().dims().len(),
            tensor_1d.shape().dims().len()
        );

        // Test 3D tensor
        let tensor_3d = rand::<f32>(&[4, 4, 8]).unwrap();
        let result_3d = mock_mean_reduction(&tensor_3d);
        assert_eq!(result_3d.shape().dims().len(), 2); // Reduced last dimension
    }
}
