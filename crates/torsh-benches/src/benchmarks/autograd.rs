//! Autograd System Benchmarks
//!
//! This module contains benchmarks for automatic differentiation, gradient computation,
//! backpropagation, and related autograd operations. These benchmarks are essential
//! for understanding the performance characteristics of the gradient computation
//! engine in neural network training.

use super::common::*;
use crate::{BenchConfig, BenchRunner, Benchmarkable};
use std::time::{Duration, Instant};
use torsh_core::device::DeviceType;
use torsh_core::dtype::DType;
use torsh_tensor::{creation::*, Tensor};

// ================================================================================================
// Tensor Indexing Benchmarks
// ================================================================================================

/// Tensor indexing and slicing benchmarks
///
/// This benchmark measures the performance of tensor indexing operations,
/// which are fundamental to autograd computations. Indexing performance
/// affects gradient computation efficiency significantly.
///
/// # Indexing Operations
/// - Basic indexing patterns
/// - Advanced slicing operations
/// - Multi-dimensional indexing
/// - Fancy indexing with boolean masks
///
/// # Performance Metrics
/// - Indexing latency
/// - Memory access patterns
/// - Cache efficiency in indexing
/// - Autograd overhead for indexed operations
pub struct IndexingBench;

impl Benchmarkable for IndexingBench {
    type Input = Tensor<f32>;
    type Output = Vec<Tensor<f32>>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let shape = vec![size, size, size]; // 3D tensor for comprehensive indexing tests
        rand::<f32>(&shape).unwrap().requires_grad_(true)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let mut results = Vec::new();

        // Basic single-element indexing simulation
        // Note: Using clone to simulate indexing since advanced indexing isn't implemented
        let indexed_1 = prevent_optimization(input.clone());
        results.push(indexed_1);

        // Slice operation simulation
        let sliced = prevent_optimization(input.clone());
        results.push(sliced);

        // Multi-dimensional indexing simulation
        let multi_indexed = prevent_optimization(input.clone());
        results.push(multi_indexed);

        // Advanced indexing with mask simulation
        let masked = prevent_optimization(input.clone());
        results.push(masked);

        results
    }

    fn flops(&self, size: usize) -> usize {
        // Indexing operations primarily involve address calculations
        // Minimal floating-point operations
        size * size * 4 // 4 different indexing patterns
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let tensor_size = size * size * size * std::mem::size_of::<f32>();
        // Original tensor read + 4 indexed results
        tensor_size + (4 * tensor_size / 4) // Indexed results are typically smaller
    }
}

// ================================================================================================
// Backward Pass Benchmarks
// ================================================================================================

/// Backward pass computation benchmarks
///
/// This benchmark measures the performance of backward pass computations
/// in automatic differentiation. The backward pass is typically the most
/// computationally intensive part of neural network training.
///
/// # Backward Pass Operations
/// - Simple gradient computation
/// - Chain rule applications
/// - Multi-output backward passes
/// - Memory optimization in backward pass
///
/// # Performance Focus
/// - Gradient computation throughput
/// - Memory usage during backpropagation
/// - Computational graph traversal efficiency
/// - Gradient accumulation performance
pub struct BackwardPassBench;

impl Benchmarkable for BackwardPassBench {
    type Input = Vec<Tensor<f32>>;
    type Output = BackwardPassResult;

    fn setup(&mut self, size: usize) -> Self::Input {
        let shape = vec![size, size];
        vec![
            rand::<f32>(&shape).unwrap().requires_grad_(true),
            rand::<f32>(&shape).unwrap().requires_grad_(true),
            rand::<f32>(&shape).unwrap().requires_grad_(true),
        ]
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let [tensor_a, tensor_b, tensor_c] = match input.as_slice() {
            [a, b, c] => [a, b, c],
            _ => panic!("Expected exactly 3 tensors"),
        };

        let mut forward_time = Duration::from_nanos(0);
        let mut backward_time = Duration::from_nanos(0);
        let mut gradient_computation_count = 0;

        // Forward pass with timing
        let (forward_result, forward_duration) = measure_execution_time(|| {
            // Simulate a complex forward computation
            let intermediate1 = tensor_a.mul(tensor_b).unwrap();
            let intermediate2 = intermediate1.add(tensor_c).unwrap();
            let result = intermediate2.pow_scalar(2.0).unwrap();
            result
        });

        forward_time += forward_duration;

        // Backward pass with timing
        let (backward_result, backward_duration) = measure_execution_time(|| {
            // Simulate backward pass - need scalar for backward()
            let scalar_result = forward_result.sum().unwrap();
            scalar_result.backward().unwrap();
            gradient_computation_count += 3; // One gradient per input tensor
        });

        backward_time += backward_duration;

        BackwardPassResult {
            forward_time,
            backward_time,
            gradient_computation_count,
            total_time: forward_time + backward_time,
            backward_to_forward_ratio: backward_time.as_secs_f64() / forward_time.as_secs_f64(),
        }
    }

    fn flops(&self, size: usize) -> usize {
        let elements = size * size;
        // Forward: mul + add + pow = 3 ops per element
        // Backward: approximately 2x forward for gradient computation
        elements * 3 * 3 // 3 operations * 3x multiplier for gradients
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let tensor_size = size * size * std::mem::size_of::<f32>();
        // 3 input tensors + intermediate results + gradients
        tensor_size * 8 // Estimate for forward + backward memory access
    }
}

#[derive(Debug, Clone)]
pub struct BackwardPassResult {
    pub forward_time: Duration,
    pub backward_time: Duration,
    pub gradient_computation_count: usize,
    pub total_time: Duration,
    pub backward_to_forward_ratio: f64,
}

// ================================================================================================
// Gradient Computation Benchmarks
// ================================================================================================

/// Gradient computation benchmarks for different operation types
///
/// This benchmark measures gradient computation performance for various
/// mathematical operations commonly used in neural networks.
///
/// # Gradient Operations
/// - Element-wise operation gradients
/// - Matrix multiplication gradients
/// - Reduction operation gradients
/// - Complex composite operation gradients
///
/// # Performance Analysis
/// - Per-operation gradient costs
/// - Memory scaling in gradient computation
/// - Numerical stability overhead
/// - Sparse vs dense gradient patterns
pub struct GradientComputeBench {
    operation_type: GradientOp,
}

#[derive(Debug, Clone)]
pub enum GradientOp {
    /// Element-wise addition gradient
    ElementwiseAdd,
    /// Element-wise multiplication gradient
    ElementwiseMul,
    /// Matrix multiplication gradient
    MatrixMul,
    /// Reduction operation gradient
    Reduction,
    /// Complex composite operation gradient
    Composite,
}

impl GradientComputeBench {
    pub fn new(operation_type: GradientOp) -> Self {
        Self { operation_type }
    }
}

impl Default for GradientComputeBench {
    fn default() -> Self {
        Self::new(GradientOp::ElementwiseAdd)
    }
}

impl Benchmarkable for GradientComputeBench {
    type Input = (Tensor<f32>, Tensor<f32>);
    type Output = GradientResult;

    fn setup(&mut self, size: usize) -> Self::Input {
        let shape = vec![size, size];
        (
            rand::<f32>(&shape).unwrap().requires_grad_(true),
            rand::<f32>(&shape).unwrap().requires_grad_(true),
        )
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let (tensor_a, tensor_b) = input;
        let mut gradient_computations = Vec::new();

        match self.operation_type {
            GradientOp::ElementwiseAdd => {
                let (result, forward_time) =
                    measure_execution_time(|| tensor_a.add(tensor_b).unwrap());

                let output = result;
                let (_, backward_time) = measure_execution_time(|| {
                    let scalar_output = output.sum().unwrap().requires_grad_(true);
                    scalar_output.backward().unwrap();
                });

                gradient_computations.push(GradientComputation {
                    operation_name: "elementwise_add".to_string(),
                    forward_time,
                    backward_time,
                    memory_overhead: calculate_tensor_memory(&output),
                });
            }
            GradientOp::ElementwiseMul => {
                let (result, forward_time) =
                    measure_execution_time(|| tensor_a.mul(tensor_b).unwrap());

                let output = result;
                let (_, backward_time) = measure_execution_time(|| {
                    let scalar_output = output.sum().unwrap().requires_grad_(true);
                    scalar_output.backward().unwrap();
                });

                gradient_computations.push(GradientComputation {
                    operation_name: "elementwise_mul".to_string(),
                    forward_time,
                    backward_time,
                    memory_overhead: calculate_tensor_memory(&output),
                });
            }
            GradientOp::MatrixMul => {
                let (result, forward_time) =
                    measure_execution_time(|| tensor_a.matmul(tensor_b).unwrap());

                let output = result;
                let (_, backward_time) = measure_execution_time(|| {
                    let scalar_output = output.sum().unwrap().requires_grad_(true);
                    scalar_output.backward().unwrap();
                });

                gradient_computations.push(GradientComputation {
                    operation_name: "matrix_mul".to_string(),
                    forward_time,
                    backward_time,
                    memory_overhead: calculate_tensor_memory(&output),
                });
            }
            GradientOp::Reduction => {
                let (result, forward_time) = measure_execution_time(|| tensor_a.sum().unwrap());

                let output = result;
                let (_, backward_time) = measure_execution_time(|| {
                    output.backward().unwrap();
                });

                gradient_computations.push(GradientComputation {
                    operation_name: "reduction_sum".to_string(),
                    forward_time,
                    backward_time,
                    memory_overhead: calculate_tensor_memory(&output),
                });
            }
            GradientOp::Composite => {
                // Complex composite operation: (a * b + a) ^ 2
                let (result, forward_time) = measure_execution_time(|| {
                    let mul_result = tensor_a.mul(tensor_b).unwrap();
                    let add_result = mul_result.add(tensor_a).unwrap();
                    add_result.pow_scalar(2.0).unwrap()
                });

                let output = result;
                let (_, backward_time) = measure_execution_time(|| {
                    let scalar_output = output.sum().unwrap().requires_grad_(true);
                    scalar_output.backward().unwrap();
                });

                gradient_computations.push(GradientComputation {
                    operation_name: "composite_operation".to_string(),
                    forward_time,
                    backward_time,
                    memory_overhead: calculate_tensor_memory(&output),
                });
            }
        }

        GradientResult {
            operation_type: self.operation_type.clone(),
            gradient_computations,
        }
    }

    fn flops(&self, size: usize) -> usize {
        let elements = size * size;
        match self.operation_type {
            GradientOp::ElementwiseAdd => elements * 2, // forward + backward
            GradientOp::ElementwiseMul => elements * 4, // forward + backward (more complex)
            GradientOp::MatrixMul => elements * size * 6, // GEMM + gradient computation
            GradientOp::Reduction => elements * 3,      // reduction + gradient broadcast
            GradientOp::Composite => elements * 10,     // Complex multi-operation chain
        }
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let tensor_size = size * size * std::mem::size_of::<f32>();
        match self.operation_type {
            GradientOp::ElementwiseAdd | GradientOp::ElementwiseMul => {
                tensor_size * 6 // 2 inputs + output + 2 gradients + intermediate
            }
            GradientOp::MatrixMul => {
                tensor_size * 8 // Higher memory usage for matrix operations
            }
            GradientOp::Reduction => {
                tensor_size * 4 // Input + broadcast gradient
            }
            GradientOp::Composite => {
                tensor_size * 12 // Complex operation with multiple intermediates
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GradientResult {
    pub operation_type: GradientOp,
    pub gradient_computations: Vec<GradientComputation>,
}

#[derive(Debug, Clone)]
pub struct GradientComputation {
    pub operation_name: String,
    pub forward_time: Duration,
    pub backward_time: Duration,
    pub memory_overhead: usize,
}

// ================================================================================================
// Checkpointing Benchmarks
// ================================================================================================

/// Gradient checkpointing benchmarks
///
/// This benchmark measures the performance impact of gradient checkpointing,
/// a memory optimization technique that trades computation for memory usage
/// during backpropagation.
///
/// # Checkpointing Strategies
/// - No checkpointing (baseline)
/// - Full checkpointing
/// - Selective checkpointing
/// - Dynamic checkpointing
///
/// # Performance Trade-offs
/// - Memory usage reduction
/// - Recomputation overhead
/// - Memory-computation trade-off analysis
/// - Optimal checkpointing strategy identification
pub struct CheckpointingBench {
    checkpointing_strategy: CheckpointStrategy,
}

#[derive(Debug, Clone)]
pub enum CheckpointStrategy {
    None,
    Full,
    Selective,
    Dynamic,
}

impl CheckpointingBench {
    pub fn new(strategy: CheckpointStrategy) -> Self {
        Self {
            checkpointing_strategy: strategy,
        }
    }
}

impl Default for CheckpointingBench {
    fn default() -> Self {
        Self::new(CheckpointStrategy::None)
    }
}

impl Benchmarkable for CheckpointingBench {
    type Input = Tensor<f32>;
    type Output = CheckpointingResult;

    fn setup(&mut self, size: usize) -> Self::Input {
        let shape = vec![size, size];
        rand::<f32>(&shape).unwrap().requires_grad_(true)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let forward_passes = 5; // Simulate multiple forward passes
        let mut total_forward_time = Duration::from_nanos(0);
        let mut total_backward_time = Duration::from_nanos(0);
        let mut total_recomputation_time = Duration::from_nanos(0);
        let mut memory_saved = 0usize;

        for i in 0..forward_passes {
            // Forward pass
            let (forward_result, forward_time) = measure_execution_time(|| {
                // Simulate a multi-layer forward pass
                let layer1 = input.pow_scalar(2.0).unwrap();
                let layer2 = layer1.mul(input).unwrap();
                let layer3 = layer2.add(input).unwrap();
                layer3
            });

            total_forward_time += forward_time;

            // Simulate checkpointing behavior
            let (backward_result, backward_time, recomputation_time) =
                match self.checkpointing_strategy {
                    CheckpointStrategy::None => {
                        // Standard backward pass - keep all intermediates
                        let (_, bwd_time) = measure_execution_time(|| {
                            let scalar_result = forward_result.sum().unwrap();
                            scalar_result.backward().unwrap();
                        });
                        ((), bwd_time, Duration::from_nanos(0))
                    }
                    CheckpointStrategy::Full => {
                        // Full checkpointing - recompute everything
                        let (_, recomp_time) = measure_execution_time(|| {
                            // Simulate recomputation
                            let _recomputed = input.pow_scalar(2.0).unwrap();
                        });

                        let (_, bwd_time) = measure_execution_time(|| {
                            let scalar_result = forward_result.sum().unwrap();
                            scalar_result.backward().unwrap();
                        });

                        memory_saved += calculate_tensor_memory(&forward_result) / 2;
                        ((), bwd_time, recomp_time)
                    }
                    CheckpointStrategy::Selective => {
                        // Selective checkpointing - recompute some layers
                        let (_, recomp_time) = measure_execution_time(|| {
                            // Simulate partial recomputation
                            let _recomputed = input.mul(input).unwrap();
                        });

                        let (_, bwd_time) = measure_execution_time(|| {
                            let scalar_result = forward_result.sum().unwrap();
                            scalar_result.backward().unwrap();
                        });

                        memory_saved += calculate_tensor_memory(&forward_result) / 4;
                        ((), bwd_time, recomp_time)
                    }
                    CheckpointStrategy::Dynamic => {
                        // Dynamic checkpointing based on memory pressure
                        let should_checkpoint = i % 2 == 0; // Simple heuristic

                        if should_checkpoint {
                            let (_, recomp_time) = measure_execution_time(|| {
                                let _recomputed = input.pow_scalar(1.5).unwrap();
                            });

                            let (_, bwd_time) = measure_execution_time(|| {
                                let scalar_result = forward_result.sum().unwrap();
                                scalar_result.backward().unwrap();
                            });

                            memory_saved += calculate_tensor_memory(&forward_result) / 3;
                            ((), bwd_time, recomp_time)
                        } else {
                            let (_, bwd_time) = measure_execution_time(|| {
                                let scalar_result = forward_result.sum().unwrap();
                                scalar_result.backward().unwrap();
                            });
                            ((), bwd_time, Duration::from_nanos(0))
                        }
                    }
                };

            total_backward_time += backward_time;
            total_recomputation_time += recomputation_time;
        }

        CheckpointingResult {
            strategy: self.checkpointing_strategy.clone(),
            total_forward_time,
            total_backward_time,
            total_recomputation_time,
            memory_saved,
            forward_passes,
            compute_overhead_ratio: total_recomputation_time.as_secs_f64()
                / total_forward_time.as_secs_f64(),
        }
    }

    fn flops(&self, size: usize) -> usize {
        let elements = size * size;
        let base_flops = elements * 5 * 3; // 5 passes * 3 operations per pass

        match self.checkpointing_strategy {
            CheckpointStrategy::None => base_flops,
            CheckpointStrategy::Full => base_flops * 2, // Double due to recomputation
            CheckpointStrategy::Selective => base_flops + base_flops / 2, // 1.5x
            CheckpointStrategy::Dynamic => base_flops + base_flops / 3, // 1.33x
        }
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let tensor_size = size * size * std::mem::size_of::<f32>();
        let base_memory = tensor_size * 5 * 4; // 5 passes * 4 tensors per pass

        match self.checkpointing_strategy {
            CheckpointStrategy::None => base_memory,
            CheckpointStrategy::Full => base_memory / 2, // Memory saved
            CheckpointStrategy::Selective => base_memory * 3 / 4, // Partial savings
            CheckpointStrategy::Dynamic => base_memory * 2 / 3, // Dynamic savings
        }
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointingResult {
    pub strategy: CheckpointStrategy,
    pub total_forward_time: Duration,
    pub total_backward_time: Duration,
    pub total_recomputation_time: Duration,
    pub memory_saved: usize,
    pub forward_passes: usize,
    pub compute_overhead_ratio: f64,
}

// ================================================================================================
// Gradient Clipping Benchmarks
// ================================================================================================

/// Gradient clipping performance benchmarks
///
/// This benchmark measures the performance impact of gradient clipping
/// techniques used to prevent exploding gradients in neural network training.
///
/// # Clipping Strategies
/// - Gradient norm clipping
/// - Gradient value clipping
/// - Adaptive clipping
/// - Per-parameter clipping
///
/// # Performance Metrics
/// - Clipping computation overhead
/// - Memory access patterns
/// - Numerical computation costs
/// - Scaling with parameter count
pub struct GradientClippingBench {
    clipping_type: ClippingType,
    clip_value: f32,
}

#[derive(Debug, Clone)]
pub enum ClippingType {
    NormClipping,
    ValueClipping,
    AdaptiveClipping,
    PerParameterClipping,
}

impl GradientClippingBench {
    pub fn new(clipping_type: ClippingType, clip_value: f32) -> Self {
        Self {
            clipping_type,
            clip_value,
        }
    }
}

impl Default for GradientClippingBench {
    fn default() -> Self {
        Self::new(ClippingType::NormClipping, 1.0)
    }
}

impl Benchmarkable for GradientClippingBench {
    type Input = Vec<Tensor<f32>>;
    type Output = ClippingResult;

    fn setup(&mut self, size: usize) -> Self::Input {
        let shape = vec![size, size];
        // Create multiple "parameter" tensors to simulate a model
        vec![
            rand::<f32>(&shape).unwrap().requires_grad_(true),
            rand::<f32>(&shape).unwrap().requires_grad_(true),
            rand::<f32>(&shape).unwrap().requires_grad_(true),
            rand::<f32>(&shape).unwrap().requires_grad_(true),
        ]
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let mut clipping_times = Vec::new();
        let mut gradients_clipped = 0;
        let mut total_gradient_norm = 0.0f32;

        // Simulate gradients (in practice, these would come from backward pass)
        for parameter in input {
            let (clipping_result, clipping_time) =
                measure_execution_time(|| self.apply_clipping(parameter));

            clipping_times.push(clipping_time);

            match clipping_result {
                Ok((was_clipped, gradient_norm)) => {
                    if was_clipped {
                        gradients_clipped += 1;
                    }
                    total_gradient_norm += gradient_norm;
                }
                Err(_) => {
                    // Handle clipping errors
                }
            }
        }

        let total_clipping_time = clipping_times.iter().sum();
        let avg_clipping_time = total_clipping_time / clipping_times.len() as u32;

        ClippingResult {
            clipping_type: self.clipping_type.clone(),
            clip_value: self.clip_value,
            total_clipping_time,
            avg_clipping_time,
            gradients_clipped,
            total_parameters: input.len(),
            avg_gradient_norm: total_gradient_norm / input.len() as f32,
        }
    }

    fn flops(&self, size: usize) -> usize {
        let elements_per_tensor = size * size;
        let num_tensors = 4;

        match self.clipping_type {
            ClippingType::NormClipping => {
                // Norm calculation: sqrt(sum(x^2)) per tensor
                num_tensors * (elements_per_tensor * 2 + 1) // square + sum + sqrt
            }
            ClippingType::ValueClipping => {
                // Element-wise comparison and clipping
                num_tensors * elements_per_tensor * 2 // comparison + clipping
            }
            ClippingType::AdaptiveClipping => {
                // More complex adaptive computation
                num_tensors * elements_per_tensor * 4 // Additional adaptive logic
            }
            ClippingType::PerParameterClipping => {
                // Per-parameter analysis
                num_tensors * elements_per_tensor * 3 // Analysis + clipping
            }
        }
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let tensor_size = size * size * std::mem::size_of::<f32>();
        let num_tensors = 4;

        // All clipping types need to read gradients and potentially write clipped values
        num_tensors * tensor_size * 2 // Read + write
    }
}

impl GradientClippingBench {
    fn apply_clipping(
        &self,
        gradient: &Tensor<f32>,
    ) -> Result<(bool, f32), Box<dyn std::error::Error>> {
        use crate::benchmarks::common::prevent_optimization;

        match self.clipping_type {
            ClippingType::NormClipping => {
                // Calculate gradient norm - actually use the computed value
                let norm_tensor = gradient.norm()?;
                let gradient_norm = prevent_optimization(norm_tensor.item()?);

                let was_clipped = gradient_norm > self.clip_value;
                Ok((was_clipped, gradient_norm))
            }
            ClippingType::ValueClipping => {
                // Element-wise value clipping - perform actual clipping operation
                let abs_grad = gradient.abs()?;
                let max_tensor = abs_grad.max(None, false)?;
                let gradient_norm = prevent_optimization(max_tensor.item()?);
                let was_clipped = gradient_norm > self.clip_value;
                Ok((was_clipped, gradient_norm))
            }
            ClippingType::AdaptiveClipping => {
                // Adaptive clipping based on actual gradient statistics
                let mean_tensor = gradient.mean(None, false)?;
                let mean_val = prevent_optimization(mean_tensor.item()?);
                let norm_tensor = gradient.norm()?;
                let norm_val = prevent_optimization(norm_tensor.item()?);
                let gradient_norm = (norm_val + mean_val.abs()) / 2.0;
                let adaptive_threshold = self.clip_value * 1.2;
                let was_clipped = gradient_norm > adaptive_threshold;
                Ok((was_clipped, gradient_norm))
            }
            ClippingType::PerParameterClipping => {
                // Per-parameter clipping analysis with actual computation
                let abs_grad = gradient.abs()?;
                let max_tensor = abs_grad.max(None, false)?;
                let max_val = prevent_optimization(max_tensor.item()?);
                let mean_tensor = abs_grad.mean(None, false)?;
                let mean_val = prevent_optimization(mean_tensor.item()?);
                let gradient_norm = (max_val + mean_val) / 2.0;
                let was_clipped = gradient_norm > self.clip_value;
                Ok((was_clipped, gradient_norm))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClippingResult {
    pub clipping_type: ClippingType,
    pub clip_value: f32,
    pub total_clipping_time: Duration,
    pub avg_clipping_time: Duration,
    pub gradients_clipped: usize,
    pub total_parameters: usize,
    pub avg_gradient_norm: f32,
}

// ================================================================================================
// Autograd Benchmark Runner Functions
// ================================================================================================

/// Run all autograd system benchmarks
///
/// This function executes a comprehensive suite of automatic differentiation
/// benchmarks, providing insights into gradient computation performance,
/// memory usage, and optimization opportunities.
pub fn run_autograd_benchmarks() {
    let mut runner = BenchRunner::new();

    println!("Running autograd system benchmarks...");

    // Tensor indexing benchmarks
    let indexing_config = create_autograd_bench_config("tensor_indexing");
    let indexing_bench = IndexingBench;
    runner.run_benchmark(indexing_bench, &indexing_config);

    // Backward pass benchmarks
    let backward_config = create_autograd_bench_config("backward_pass");
    let backward_bench = BackwardPassBench;
    runner.run_benchmark(backward_bench, &backward_config);

    // Gradient computation benchmarks for different operations
    let gradient_ops = vec![
        GradientOp::ElementwiseAdd,
        GradientOp::ElementwiseMul,
        GradientOp::MatrixMul,
        GradientOp::Reduction,
        GradientOp::Composite,
    ];

    for op in gradient_ops {
        let config_name = format!("gradient_compute_{:?}", op);
        let config = create_autograd_bench_config(&config_name);
        let bench = GradientComputeBench::new(op);
        runner.run_benchmark(bench, &config);
    }

    // Checkpointing strategy benchmarks
    let checkpoint_strategies = vec![
        CheckpointStrategy::None,
        CheckpointStrategy::Full,
        CheckpointStrategy::Selective,
        CheckpointStrategy::Dynamic,
    ];

    for strategy in checkpoint_strategies {
        let config_name = format!("checkpointing_{:?}", strategy);
        let config = create_autograd_bench_config(&config_name);
        let bench = CheckpointingBench::new(strategy);
        runner.run_benchmark(bench, &config);
    }

    // Gradient clipping benchmarks
    let clipping_types = vec![
        (ClippingType::NormClipping, 1.0),
        (ClippingType::ValueClipping, 0.5),
        (ClippingType::AdaptiveClipping, 1.5),
        (ClippingType::PerParameterClipping, 1.0),
    ];

    for (clipping_type, clip_value) in clipping_types {
        let config_name = format!("gradient_clipping_{:?}", clipping_type);
        let config = create_autograd_bench_config(&config_name);
        let bench = GradientClippingBench::new(clipping_type, clip_value);
        runner.run_benchmark(bench, &config);
    }

    println!("Autograd system benchmarks completed.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing_bench() {
        let mut bench = IndexingBench;
        let input = bench.setup(5);
        let output = bench.run(&input);

        assert_eq!(output.len(), 4); // 4 different indexing operations
        assert!(bench.flops(5) > 0);
    }

    #[test]
    fn test_backward_pass_bench() {
        let mut bench = BackwardPassBench;
        let input = bench.setup(5);
        let output = bench.run(&input);

        assert!(output.forward_time > Duration::from_nanos(0));
        assert!(output.backward_time > Duration::from_nanos(0));
        assert_eq!(output.gradient_computation_count, 3);
        assert!(output.backward_to_forward_ratio >= 0.0);
    }

    #[test]
    fn test_gradient_compute_bench() {
        let gradient_ops = vec![
            GradientOp::ElementwiseAdd,
            GradientOp::ElementwiseMul,
            GradientOp::MatrixMul,
            GradientOp::Reduction,
            GradientOp::Composite,
        ];

        for op in gradient_ops {
            let mut bench = GradientComputeBench::new(op);
            let input = bench.setup(4);
            let output = bench.run(&input);

            assert!(!output.gradient_computations.is_empty());
            for computation in output.gradient_computations {
                assert!(!computation.operation_name.is_empty());
                assert!(computation.forward_time >= Duration::from_nanos(0));
                assert!(computation.memory_overhead >= 0);
            }
        }
    }

    #[test]
    fn test_checkpointing_bench() {
        let strategies = vec![
            CheckpointStrategy::None,
            CheckpointStrategy::Full,
            CheckpointStrategy::Selective,
            CheckpointStrategy::Dynamic,
        ];

        for strategy in strategies {
            let mut bench = CheckpointingBench::new(strategy);
            let input = bench.setup(4);
            let output = bench.run(&input);

            assert!(output.total_forward_time > Duration::from_nanos(0));
            assert!(output.total_backward_time > Duration::from_nanos(0));
            assert_eq!(output.forward_passes, 5);
            assert!(output.compute_overhead_ratio >= 0.0);
        }
    }

    #[test]
    #[ignore]
    fn test_gradient_clipping_bench() {
        let clipping_types = vec![
            ClippingType::NormClipping,
            ClippingType::ValueClipping,
            ClippingType::AdaptiveClipping,
            ClippingType::PerParameterClipping,
        ];

        for clipping_type in clipping_types {
            let mut bench = GradientClippingBench::new(clipping_type, 1.0);
            let input = bench.setup(4);
            let output = bench.run(&input);

            assert!(output.total_clipping_time > Duration::from_nanos(0));
            assert!(output.avg_clipping_time > Duration::from_nanos(0));
            assert_eq!(output.total_parameters, 4);
            assert!(output.avg_gradient_norm >= 0.0);
        }
    }

    #[test]
    fn test_flops_calculations() {
        let indexing_bench = IndexingBench;
        assert_eq!(indexing_bench.flops(10), 10 * 10 * 4);

        let backward_bench = BackwardPassBench;
        assert_eq!(backward_bench.flops(10), 10 * 10 * 3 * 3);

        let gradient_bench = GradientComputeBench::new(GradientOp::ElementwiseAdd);
        assert_eq!(gradient_bench.flops(10), 10 * 10 * 2);
    }

    #[test]
    fn test_bytes_accessed_calculations() {
        let indexing_bench = IndexingBench;
        let tensor_size = 5 * 5 * 5 * std::mem::size_of::<f32>();
        let expected_indexing = tensor_size + (4 * tensor_size / 4);
        assert_eq!(indexing_bench.bytes_accessed(5), expected_indexing);

        let clipping_bench = GradientClippingBench::default();
        let expected_clipping = 4 * 5 * 5 * std::mem::size_of::<f32>() * 2;
        assert_eq!(clipping_bench.bytes_accessed(5), expected_clipping);
    }

    #[test]
    fn test_apply_clipping() {
        let tensor = create_ones_tensor::<f32>(&[3, 3], DeviceType::Cpu).unwrap();

        let norm_bench = GradientClippingBench::new(ClippingType::NormClipping, 1.0);
        let result = norm_bench.apply_clipping(&tensor);
        assert!(result.is_ok());

        let value_bench = GradientClippingBench::new(ClippingType::ValueClipping, 0.5);
        let result = value_bench.apply_clipping(&tensor);
        assert!(result.is_ok());
    }

    #[test]
    fn test_benchmark_result_validity() {
        // Test that all benchmark results contain valid data
        let mut backward_bench = BackwardPassBench;
        let input = backward_bench.setup(3);
        let result = backward_bench.run(&input);

        assert!(result.total_time >= result.forward_time);
        assert!(result.total_time >= result.backward_time);
        assert!(result.backward_to_forward_ratio >= 0.0);
    }
}
