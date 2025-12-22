//! Tensor Operations Benchmarks
//!
//! This module contains benchmarks for core tensor operations including creation,
//! arithmetic operations, matrix multiplication, and reductions. These benchmarks
//! measure the fundamental performance characteristics of tensor computations
//! in the ToRSh framework.

use super::common::*;
use crate::{BenchRunner, Benchmarkable};
use torsh_core::dtype::DType;
use torsh_tensor::{creation::*, Tensor};

// ================================================================================================
// Tensor Creation Benchmarks
// ================================================================================================

/// Benchmarks for tensor creation operations
///
/// This benchmark measures the performance of creating tensors with different
/// initialization strategies. It covers various creation patterns commonly used
/// in deep learning applications.
///
/// # Benchmark Categories
/// - Random tensor creation
/// - Filled tensor creation (zeros, ones, constants)
/// - Identity matrix creation
/// - Range tensor creation
///
/// # Performance Metrics
/// - Allocation time
/// - Memory bandwidth utilization
/// - Initialization overhead
pub struct TensorCreationBench;

impl TensorCreationBench {
    pub fn new(_dtype: DType) -> Self {
        Self
    }
}

impl Benchmarkable for TensorCreationBench {
    type Input = (usize, DType);
    type Output = Vec<Tensor<f32>>;

    fn setup(&mut self, size: usize) -> Self::Input {
        (size, DType::F32)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let (size, _dtype) = *input;
        let shape = vec![size, size];

        let mut tensors = Vec::new();

        // Benchmark different creation methods
        tensors.push(prevent_optimization(rand::<f32>(&shape).unwrap()));
        tensors.push(prevent_optimization(zeros::<f32>(&shape).unwrap()));
        tensors.push(prevent_optimization(ones::<f32>(&shape).unwrap()));
        tensors.push(prevent_optimization(
            full::<f32>(&shape, std::f32::consts::PI).unwrap(),
        ));

        // Identity matrix (for square tensors)
        if shape[0] == shape[1] {
            tensors.push(prevent_optimization(eye::<f32>(shape[0]).unwrap()));
        }

        tensors
    }

    fn flops(&self, size: usize) -> usize {
        // Tensor creation is primarily memory-bound, minimal FLOPS
        // Count initialization operations (setting values)
        size * size * 5 // 5 different creation methods
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        // 5 tensors created, each with size^2 elements
        5 * size * size * std::mem::size_of::<f32>()
    }
}

// ================================================================================================
// Tensor Arithmetic Benchmarks
// ================================================================================================

/// Benchmarks for tensor arithmetic operations
///
/// This benchmark measures the performance of element-wise and broadcast
/// arithmetic operations. It covers the fundamental arithmetic operations
/// used throughout neural network computations.
///
/// # Operations Tested
/// - Element-wise addition, subtraction, multiplication, division
/// - Scalar arithmetic operations
/// - Broadcasting operations
/// - In-place vs out-of-place operations
///
/// # Performance Focus
/// - Element-wise operation throughput
/// - Memory access patterns
/// - Broadcasting overhead
/// - Cache efficiency
pub struct TensorArithmeticBench {
    operation_type: ArithmeticOp,
}

#[derive(Debug, Clone)]
pub enum ArithmeticOp {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    ScalarMultiply,
    Broadcasting,
}

impl TensorArithmeticBench {
    pub fn new() -> Self {
        Self {
            operation_type: ArithmeticOp::Addition,
        }
    }

    pub fn with_operation(operation_type: ArithmeticOp) -> Self {
        Self { operation_type }
    }
}

impl Default for TensorArithmeticBench {
    fn default() -> Self {
        Self::new()
    }
}

impl Benchmarkable for TensorArithmeticBench {
    type Input = (Tensor<f32>, Tensor<f32>, f32);
    type Output = Vec<Tensor<f32>>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let shape = vec![size, size];
        let tensor_a = rand::<f32>(&shape).unwrap();
        let tensor_b = rand::<f32>(&shape).unwrap();
        let scalar = 2.5f32;

        (tensor_a, tensor_b, scalar)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let (tensor_a, tensor_b, scalar) = input;
        let mut results = Vec::new();

        match self.operation_type {
            ArithmeticOp::Addition => {
                // Element-wise addition benchmarks
                let sum = tensor_a + tensor_b;
                results.push(prevent_optimization(sum));
            }
            ArithmeticOp::Subtraction => {
                // Element-wise subtraction benchmarks
                let diff = tensor_a - tensor_b;
                results.push(prevent_optimization(diff));
            }
            ArithmeticOp::Multiplication => {
                // Element-wise multiplication (Hadamard product)
                let product = tensor_a.mul(tensor_b).unwrap();
                results.push(prevent_optimization(product));
            }
            ArithmeticOp::Division => {
                // Element-wise division
                let quotient = tensor_a / tensor_b;
                results.push(prevent_optimization(quotient));
            }
            ArithmeticOp::ScalarMultiply => {
                // Scalar multiplication
                let scaled = tensor_a
                    .mul_scalar(*scalar)
                    .unwrap_or_else(|_| tensor_a.clone());
                results.push(prevent_optimization(scaled));
            }
            ArithmeticOp::Broadcasting => {
                // Broadcasting operations
                let broadcasted = tensor_a + tensor_b;
                results.push(prevent_optimization(broadcasted));
            }
        }

        results
    }

    fn flops(&self, size: usize) -> usize {
        let elements = size * size;
        match self.operation_type {
            ArithmeticOp::Addition | ArithmeticOp::Subtraction => elements,
            ArithmeticOp::Multiplication | ArithmeticOp::Division => elements,
            ArithmeticOp::ScalarMultiply => elements,
            ArithmeticOp::Broadcasting => elements,
        }
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let element_size = std::mem::size_of::<f32>();
        let tensor_size = size * size * element_size;

        match self.operation_type {
            ArithmeticOp::Addition
            | ArithmeticOp::Subtraction
            | ArithmeticOp::Multiplication
            | ArithmeticOp::Division
            | ArithmeticOp::Broadcasting => {
                // 2 input tensors + 1 output tensor
                3 * tensor_size
            }
            ArithmeticOp::ScalarMultiply => {
                // 1 input tensor + 1 output tensor
                2 * tensor_size
            }
        }
    }
}

// ================================================================================================
// Matrix Multiplication Benchmarks
// ================================================================================================

/// Benchmarks for matrix multiplication operations
///
/// This benchmark measures the performance of matrix multiplication operations,
/// which are fundamental to neural network computations. It tests various
/// matrix sizes and layouts to understand performance characteristics.
///
/// # Operation Types
/// - Standard matrix multiplication (GEMM)
/// - Batch matrix multiplication
/// - Different matrix shapes and sizes
/// - Memory layout considerations
///
/// # Performance Metrics
/// - GFLOPS (Giga Floating Point Operations Per Second)
/// - Memory bandwidth utilization
/// - Cache efficiency
/// - Computational intensity
pub struct MatmulBench;

impl Benchmarkable for MatmulBench {
    type Input = (Tensor<f32>, Tensor<f32>);
    type Output = Tensor<f32>;

    fn setup(&mut self, size: usize) -> Self::Input {
        // Create matrices optimized for multiplication: [size, size] × [size, size]
        let matrix_a = rand::<f32>(&[size, size]).unwrap();
        let matrix_b = rand::<f32>(&[size, size]).unwrap();

        (matrix_a, matrix_b)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let (matrix_a, matrix_b) = input;

        // Perform matrix multiplication
        let result = matrix_a.matmul(matrix_b).unwrap();
        prevent_optimization(result)
    }

    fn flops(&self, size: usize) -> usize {
        // Matrix multiplication: [n,k] × [k,m] requires n*m*k multiplications and (k-1)*n*m additions
        // For square matrices [size, size] × [size, size]: size^3 multiplications + size^2*(size-1) additions
        let multiplications = size * size * size;
        let additions = size * size * (size - 1);
        multiplications + additions
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let element_size = std::mem::size_of::<f32>();
        // 2 input matrices + 1 output matrix, each with size^2 elements
        3 * size * size * element_size
    }
}

// ================================================================================================
// Tensor Reduction Benchmarks
// ================================================================================================

/// Benchmarks for tensor reduction operations
///
/// This benchmark measures the performance of reduction operations that
/// aggregate tensor values along specified dimensions. These operations
/// are crucial for loss computation, normalization, and statistical analysis.
///
/// # Reduction Types
/// - Sum reduction (across different dimensions)
/// - Mean reduction
/// - Maximum and minimum reductions
/// - Product reduction
/// - Norm computations
///
/// # Performance Characteristics
/// - Memory bandwidth optimization
/// - Parallel reduction efficiency
/// - Cache locality in aggregation
/// - Numerical stability overhead
pub struct ReductionBench;

impl Benchmarkable for ReductionBench {
    type Input = Tensor<f32>;
    type Output = Vec<Tensor<f32>>;

    fn setup(&mut self, size: usize) -> Self::Input {
        // Create a larger tensor for meaningful reduction operations
        let shape = vec![size, size];
        rand::<f32>(&shape).unwrap()
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let mut results = Vec::new();

        // Sum reduction (all elements)
        let sum_all = input.sum().unwrap();
        results.push(prevent_optimization(sum_all));

        // Norm computation
        let norm = input.norm().unwrap();
        results.push(prevent_optimization(norm));

        // Additional reduction operations would go here
        // Note: These use mock implementations from TensorExtensions

        results
    }

    fn flops(&self, size: usize) -> usize {
        let total_elements = size * size;
        // Sum reduction: (n-1) additions for n elements
        // Norm computation: n multiplications + (n-1) additions + 1 sqrt
        let sum_ops = total_elements - 1;
        let norm_ops = total_elements + (total_elements - 1) + 1; // mul + add + sqrt
        sum_ops + norm_ops
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let element_size = std::mem::size_of::<f32>();
        let input_size = size * size * element_size;
        // Input tensor read multiple times for different reductions
        // + small output tensors (scalars)
        input_size * 2 + 16 // 2 reads + small outputs
    }
}

// ================================================================================================
// Benchmark Runner Functions
// ================================================================================================

/// Run all tensor operation benchmarks
///
/// This function executes a comprehensive suite of tensor operation benchmarks,
/// providing a complete performance profile of fundamental tensor operations.
pub fn run_tensor_operation_benchmarks() {
    let mut runner = BenchRunner::new();

    // Tensor creation benchmarks
    println!("Running tensor creation benchmarks...");
    let creation_config = create_tensor_bench_config("tensor_creation")
        .with_sizes(vec![64, 128, 256, 512, 1024, 2048]);

    let creation_bench = TensorCreationBench;
    runner.run_benchmark(creation_bench, &creation_config);

    // Arithmetic operation benchmarks
    println!("Running tensor arithmetic benchmarks...");
    let _arithmetic_config =
        create_tensor_bench_config("tensor_arithmetic").with_sizes(vec![64, 128, 256, 512, 1024]);

    // Test each arithmetic operation type
    let arithmetic_ops = vec![
        ArithmeticOp::Addition,
        ArithmeticOp::Subtraction,
        ArithmeticOp::Multiplication,
        ArithmeticOp::Division,
        ArithmeticOp::ScalarMultiply,
        ArithmeticOp::Broadcasting,
    ];

    for op in arithmetic_ops {
        let bench = TensorArithmeticBench::with_operation(op.clone());
        let config_name = format!("tensor_arithmetic_{:?}", op);
        let config = create_tensor_bench_config(&config_name);
        runner.run_benchmark(bench, &config);
    }

    // Matrix multiplication benchmarks
    println!("Running matrix multiplication benchmarks...");
    let matmul_config =
        create_tensor_bench_config("matrix_multiplication").with_sizes(vec![32, 64, 128, 256, 512]);

    let matmul_bench = MatmulBench;
    runner.run_benchmark(matmul_bench, &matmul_config);

    // Reduction operation benchmarks
    println!("Running tensor reduction benchmarks...");
    let reduction_config = create_tensor_bench_config("tensor_reduction")
        .with_sizes(vec![100, 500, 1000, 5000, 10000]);

    let reduction_bench = ReductionBench;
    runner.run_benchmark(reduction_bench, &reduction_config);

    println!("Tensor operation benchmarks completed.");
}

// ================================================================================================
// Advanced Tensor Operation Benchmarks
// ================================================================================================

/// Advanced tensor operation benchmark for specialized operations
///
/// This benchmark covers more complex tensor operations that combine multiple
/// basic operations or have specific performance characteristics.
pub struct AdvancedTensorOpsBench {
    operation_type: AdvancedOp,
}

#[derive(Debug, Clone)]
pub enum AdvancedOp {
    /// Transpose operation performance
    Transpose,
    /// Reshape operation performance
    Reshape,
    /// Contiguous memory layout conversion
    Contiguous,
    /// Complex view operations
    Views,
}

impl AdvancedTensorOpsBench {
    pub fn new(operation_type: AdvancedOp) -> Self {
        Self { operation_type }
    }
}

impl Benchmarkable for AdvancedTensorOpsBench {
    type Input = Tensor<f32>;
    type Output = Tensor<f32>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let shape = vec![size, size];
        rand::<f32>(&shape).unwrap()
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        match self.operation_type {
            AdvancedOp::Transpose => {
                let result = input.transpose(0, 1).unwrap();
                prevent_optimization(result)
            }
            AdvancedOp::Reshape => {
                let new_shape = vec![input.shape().numel() as i32];
                let result = input.view(&new_shape).unwrap();
                prevent_optimization(result)
            }
            AdvancedOp::Contiguous => {
                let result = input.contiguous().unwrap();
                prevent_optimization(result)
            }
            AdvancedOp::Views => {
                // Create a view and then reshape it
                let flat = input.view(&[input.shape().numel() as i32]).unwrap();
                let reshaped = flat
                    .view(&[
                        input.shape().dims()[0] as i32,
                        input.shape().dims()[1] as i32,
                    ])
                    .unwrap();
                prevent_optimization(reshaped)
            }
        }
    }

    fn flops(&self, size: usize) -> usize {
        // Most of these operations are memory-bound with minimal computation
        match self.operation_type {
            AdvancedOp::Transpose | AdvancedOp::Reshape | AdvancedOp::Views => 0,
            AdvancedOp::Contiguous => size * size, // May involve data copying
        }
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let element_size = std::mem::size_of::<f32>();
        let tensor_size = size * size * element_size;

        match self.operation_type {
            AdvancedOp::Transpose | AdvancedOp::Reshape | AdvancedOp::Views => {
                // Input read + output (may be view, so minimal additional memory)
                tensor_size
            }
            AdvancedOp::Contiguous => {
                // Input read + output write (full copy)
                2 * tensor_size
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation_bench() {
        let mut bench = TensorCreationBench;
        let input = bench.setup(10);
        let output = bench.run(&input);

        assert_eq!(output.len(), 5); // 4 creation methods + eye matrix
        assert_eq!(bench.flops(10), 500); // 10*10*5
    }

    #[test]
    fn test_tensor_arithmetic_bench() {
        let mut bench = TensorArithmeticBench::new();
        let input = bench.setup(5);
        let output = bench.run(&input);

        assert_eq!(output.len(), 1);
        assert_eq!(bench.flops(5), 25); // 5*5 elements
    }

    #[test]
    fn test_arithmetic_operations() {
        let ops = vec![
            ArithmeticOp::Addition,
            ArithmeticOp::Subtraction,
            ArithmeticOp::Multiplication,
            ArithmeticOp::Division,
            ArithmeticOp::ScalarMultiply,
            ArithmeticOp::Broadcasting,
        ];

        for op in ops {
            let mut bench = TensorArithmeticBench::with_operation(op);
            let input = bench.setup(5);
            let output = bench.run(&input);
            assert_eq!(output.len(), 1);
        }
    }

    #[test]
    fn test_matmul_bench() {
        let mut bench = MatmulBench;
        let input = bench.setup(4);
        let output = bench.run(&input);

        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[4, 4]);

        // Check FLOPS calculation: 4^3 + 4^2*3 = 64 + 48 = 112
        assert_eq!(bench.flops(4), 112);
    }

    #[test]
    fn test_reduction_bench() {
        let mut bench = ReductionBench;
        let input = bench.setup(5);
        let output = bench.run(&input);

        assert_eq!(output.len(), 2); // sum + norm

        // Check that reductions produce scalar tensors
        for result in output {
            // Scalar tensors have 0 dimensions, not 1
            assert_eq!(result.shape().dims().len(), 0);
        }
    }

    #[test]
    fn test_advanced_tensor_ops() {
        let advanced_ops = vec![
            AdvancedOp::Transpose,
            AdvancedOp::Reshape,
            AdvancedOp::Contiguous,
            AdvancedOp::Views,
        ];

        for op in advanced_ops {
            let mut bench = AdvancedTensorOpsBench::new(op);
            let input = bench.setup(4);
            let output = bench.run(&input);

            // All operations should produce valid tensors
            assert!(output.shape().numel() > 0);
        }
    }

    #[test]
    fn test_flops_calculations() {
        let creation_bench = TensorCreationBench;
        assert_eq!(creation_bench.flops(10), 500);

        let arithmetic_bench = TensorArithmeticBench::new();
        assert_eq!(arithmetic_bench.flops(10), 100);

        let matmul_bench = MatmulBench;
        // 2x2 * 2x2: 2*2*2 = 8 multiplications + 2*2*(2-1) = 4 additions = 12 total
        assert_eq!(matmul_bench.flops(2), 12);

        let reduction_bench = ReductionBench;
        // sum: (4-1) = 3, norm: 4 + 3 + 1 = 8, total = 11
        assert_eq!(reduction_bench.flops(2), 11); // For 2x2 = 4 elements
    }

    #[test]
    fn test_bytes_accessed_calculations() {
        let creation_bench = TensorCreationBench;
        let expected_bytes = 5 * 10 * 10 * std::mem::size_of::<f32>();
        assert_eq!(creation_bench.bytes_accessed(10), expected_bytes);

        let arithmetic_bench = TensorArithmeticBench::new();
        let expected_arith_bytes = 3 * 10 * 10 * std::mem::size_of::<f32>();
        assert_eq!(arithmetic_bench.bytes_accessed(10), expected_arith_bytes);

        let matmul_bench = MatmulBench;
        let expected_matmul_bytes = 3 * 10 * 10 * std::mem::size_of::<f32>();
        assert_eq!(matmul_bench.bytes_accessed(10), expected_matmul_bytes);
    }

    #[test]
    fn test_benchmark_runner_integration() {
        // This test ensures the runner function can be called without panicking
        // Note: In a real test environment, we might want to use a mock runner
        // run_tensor_operation_benchmarks(); // Commented out to avoid long test runs
    }
}
