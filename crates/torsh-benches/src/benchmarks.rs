//! Core benchmark implementations for ToRSh operations

use crate::{Benchmarkable, BenchConfig, BenchRunner};
use torsh_tensor::{Tensor, creation::*};
use torsh_core::dtype::DType;
use criterion::black_box;

/// Tensor creation benchmarks
pub struct TensorCreationBench;

impl Benchmarkable for TensorCreationBench {
    type Input = Vec<usize>;
    type Output = Tensor<f32>;
    
    fn setup(&mut self, size: usize) -> Self::Input {
        vec![size, size] // Square matrix
    }
    
    fn run(&mut self, input: &Self::Input) -> Self::Output {
        black_box(zeros::<f32>(input))
    }
    
    fn bytes_accessed(&self, size: usize) -> usize {
        size * size * std::mem::size_of::<f32>()
    }
}

/// Tensor arithmetic benchmarks
pub struct TensorArithmeticBench {
    tensor_a: Option<Tensor<f32>>,
    tensor_b: Option<Tensor<f32>>,
}

impl TensorArithmeticBench {
    pub fn new() -> Self {
        Self {
            tensor_a: None,
            tensor_b: None,
        }
    }
}

impl Default for TensorArithmeticBench {
    fn default() -> Self {
        Self::new()
    }
}

impl Benchmarkable for TensorArithmeticBench {
    type Input = (Tensor<f32>, Tensor<f32>);
    type Output = Result<Tensor<f32>, torsh_core::error::TorshError>;
    
    fn setup(&mut self, size: usize) -> Self::Input {
        let shape = vec![size, size];
        let a = rand::<f32>(&shape);
        let b = rand::<f32>(&shape);
        (a, b)
    }
    
    fn run(&mut self, input: &Self::Input) -> Self::Output {
        // Test addition
        input.0.add(&input.1)
    }
    
    fn flops(&self, size: usize) -> usize {
        size * size // One operation per element
    }
    
    fn bytes_accessed(&self, size: usize) -> usize {
        3 * size * size * std::mem::size_of::<f32>() // Read 2, write 1
    }
}

/// Matrix multiplication benchmarks
pub struct MatmulBench;

impl Benchmarkable for MatmulBench {
    type Input = (Tensor<f32>, Tensor<f32>);
    type Output = Result<Tensor<f32>, torsh_core::error::TorshError>;
    
    fn setup(&mut self, size: usize) -> Self::Input {
        let a = rand::<f32>(&[size, size]);
        let b = rand::<f32>(&[size, size]);
        (a, b)
    }
    
    fn run(&mut self, input: &Self::Input) -> Self::Output {
        input.0.matmul(&input.1)
    }
    
    fn flops(&self, size: usize) -> usize {
        2 * size * size * size // Standard matrix multiplication complexity
    }
    
    fn bytes_accessed(&self, size: usize) -> usize {
        3 * size * size * std::mem::size_of::<f32>() // Two input matrices + output
    }
}

/// Tensor reduction benchmarks
pub struct ReductionBench;

impl Benchmarkable for ReductionBench {
    type Input = Tensor<f32>;
    type Output = Result<Tensor<f32>, torsh_core::error::TorshError>;
    
    fn setup(&mut self, size: usize) -> Self::Input {
        rand::<f32>(&[size, size])
    }
    
    fn run(&mut self, input: &Self::Input) -> Self::Output {
        input.sum()
    }
    
    fn flops(&self, size: usize) -> usize {
        size * size - 1 // n-1 additions for sum
    }
    
    fn bytes_accessed(&self, size: usize) -> usize {
        size * size * std::mem::size_of::<f32>() // Read entire tensor
    }
}

/// Memory allocation benchmarks
pub struct MemoryBench;

impl Benchmarkable for MemoryBench {
    type Input = Vec<usize>;
    type Output = Vec<Tensor<f32>>;
    
    fn setup(&mut self, size: usize) -> Self::Input {
        vec![size; 16] // Allocate 16 tensors
    }
    
    fn run(&mut self, input: &Self::Input) -> Self::Output {
        input.iter()
            .map(|&s| zeros::<f32>(&[s, s]))
            .collect()
    }
    
    fn bytes_accessed(&self, size: usize) -> usize {
        16 * size * size * std::mem::size_of::<f32>()
    }
}

/// Tensor indexing benchmarks
pub struct IndexingBench;

impl Benchmarkable for IndexingBench {
    type Input = (Tensor<f32>, Vec<usize>);
    type Output = Vec<Result<f32, torsh_core::error::TorshError>>;
    
    fn setup(&mut self, size: usize) -> Self::Input {
        let tensor = rand::<f32>(&[size]);
        let indices: Vec<usize> = (0..std::cmp::min(size, 1000)).collect();
        (tensor, indices)
    }
    
    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let (tensor, indices) = input;
        indices.iter()
            .map(|&i| tensor.get(i))
            .collect()
    }
    
    fn bytes_accessed(&self, size: usize) -> usize {
        std::cmp::min(size, 1000) * std::mem::size_of::<f32>()
    }
}

/// Benchmark suite runner
pub fn run_tensor_benchmarks() {
    let mut runner = BenchRunner::new();
    
    // Tensor creation benchmarks
    let creation_config = BenchConfig::new("tensor_creation")
        .with_sizes(vec![64, 128, 256, 512, 1024, 2048])
        .with_dtypes(vec![DType::F32, DType::F64]);
    
    let mut creation_bench = TensorCreationBench;
    runner.run_benchmark(creation_bench, &creation_config);
    
    // Arithmetic benchmarks
    let arithmetic_config = BenchConfig::new("tensor_arithmetic")
        .with_sizes(vec![64, 128, 256, 512, 1024])
        .with_dtypes(vec![DType::F32]);
    
    let mut arithmetic_bench = TensorArithmeticBench::new();
    runner.run_benchmark(arithmetic_bench, &arithmetic_config);
    
    // Matrix multiplication benchmarks
    let matmul_config = BenchConfig::new("matrix_multiplication")
        .with_sizes(vec![32, 64, 128, 256, 512])
        .with_dtypes(vec![DType::F32]);
    
    let mut matmul_bench = MatmulBench;
    runner.run_benchmark(matmul_bench, &matmul_config);
    
    // Reduction benchmarks
    let reduction_config = BenchConfig::new("tensor_reduction")
        .with_sizes(vec![100, 500, 1000, 5000, 10000])
        .with_dtypes(vec![DType::F32]);
    
    let mut reduction_bench = ReductionBench;
    runner.run_benchmark(reduction_bench, &reduction_config);
    
    // Memory benchmarks
    let memory_config = BenchConfig::new("memory_operations")
        .with_sizes(vec![64, 128, 256, 512])
        .with_memory_measurement();
    
    let mut memory_bench = MemoryBench;
    runner.run_benchmark(memory_bench, &memory_config);
    
    // Generate report
    runner.generate_report("target/benchmark_reports").unwrap();
    runner.export_csv("target/benchmark_results.csv").unwrap();
}

/// Benchmark configuration presets
pub mod presets {
    use super::*;
    
    /// Quick benchmarks for development
    pub fn quick() -> Vec<BenchConfig> {
        vec![
            BenchConfig::new("quick_arithmetic")
                .with_sizes(vec![64, 128])
                .with_dtypes(vec![DType::F32]),
            BenchConfig::new("quick_matmul")
                .with_sizes(vec![32, 64])
                .with_dtypes(vec![DType::F32]),
        ]
    }
    
    /// Comprehensive benchmarks for CI
    pub fn comprehensive() -> Vec<BenchConfig> {
        vec![
            BenchConfig::new("full_arithmetic")
                .with_sizes(vec![64, 128, 256, 512, 1024, 2048])
                .with_dtypes(vec![DType::F16, DType::F32, DType::F64]),
            BenchConfig::new("full_matmul")
                .with_sizes(vec![32, 64, 128, 256, 512, 1024])
                .with_dtypes(vec![DType::F32, DType::F64]),
            BenchConfig::new("full_reduction")
                .with_sizes(vec![1000, 5000, 10000, 50000, 100000])
                .with_dtypes(vec![DType::F32]),
        ]
    }
    
    /// Performance comparison benchmarks
    pub fn comparison() -> Vec<BenchConfig> {
        vec![
            BenchConfig::new("comparison_matmul")
                .with_sizes(vec![64, 128, 256, 512, 1024])
                .with_dtypes(vec![DType::F32])
                .with_metadata("library", "torsh"),
            BenchConfig::new("comparison_arithmetic")
                .with_sizes(vec![1000, 10000, 100000, 1000000])
                .with_dtypes(vec![DType::F32])
                .with_metadata("library", "torsh"),
        ]
    }
}