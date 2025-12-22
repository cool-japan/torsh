//! Modular Benchmark System for ToRSh
//!
//! This module provides a comprehensive benchmarking framework for the ToRSh deep learning
//! library. The system has been refactored from a single monolithic file into a modular
//! architecture for better maintainability, testability, and extensibility.
//!
//! ## Module Architecture
//!
//! The benchmark system consists of the following specialized modules:
//!
//! - **`common`**: Shared utilities, extensions, and benchmark infrastructure
//! - **`tensor_ops`**: Tensor operation benchmarks (creation, arithmetic, matrix operations)
//! - **`memory`**: Memory management benchmarks (allocation, fragmentation, concurrent access)
//! - **`autograd`**: Automatic differentiation benchmarks (gradient computation, backpropagation)
//! - **`data_loading`**: Data loading and preprocessing benchmarks (dataloaders, sampling, transforms)
//! - **`optimization`**: Performance optimization benchmarks (kernel fusion, graph optimization)
//! - **`advanced_systems`**: Advanced system benchmarks (auto-tuning, ML diagnostics, vectorized metrics, SIMD GNN)
//!
//! ## Usage
//!
//! ### Basic Usage
//! ```rust,ignore
//! use torsh_benches::benchmarks::{TensorCreationBench, Benchmarkable};
//!
//! let mut bench = TensorCreationBench::new(torsh_core::DType::F32);
//! let input = bench.setup(256);
//! let result = bench.run(&input);
//! let flops = bench.flops(256);
//! let bytes = bench.bytes_accessed(256);
//! ```
//!
//! ### Running Benchmark Suites
//! ```rust,ignore
//! use torsh_benches::benchmarks::{run_comprehensive_benchmarks, BenchmarkSuite};
//!
//! // Run all benchmarks
//! run_comprehensive_benchmarks();
//!
//! // Run specific benchmark category
//! let mut suite = BenchmarkSuite::new();
//! suite.run_tensor_operation_benchmarks();
//! suite.run_memory_benchmarks();
//! suite.run_autograd_benchmarks();
//! ```
//!
//! ## Backward Compatibility
//!
//! This modular system maintains 100% backward compatibility with the original monolithic
//! implementation. All existing imports and usage patterns continue to work unchanged.

// ================================================================================================
// Module Declarations and Re-exports
// ================================================================================================

pub mod advanced_systems;
pub mod autograd;
pub mod common;
pub mod data_loading;
pub mod memory;
pub mod optimization;
pub mod tensor_ops;
pub mod ultimate_performance_validation;

// Re-export all public types for backward compatibility
pub use advanced_systems::*;
pub use autograd::*;
pub use common::*;
pub use data_loading::*;
pub use memory::*;
pub use optimization::*;
pub use tensor_ops::*;
pub use ultimate_performance_validation::*;

// Core dependencies used across all benchmark modules
use crate::Benchmarkable;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use torsh_core::DType;

// ================================================================================================
// Unified Benchmark Runner and Suite
// ================================================================================================

/// Comprehensive benchmark suite runner
///
/// Provides a unified interface for running all benchmark categories and generating
/// comprehensive performance reports.
pub struct BenchmarkSuite {
    pub config: BenchmarkConfig,
    pub results: HashMap<String, Vec<BenchmarkResult>>,
}

/// Configuration for benchmark runs
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub sizes: Vec<usize>,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub timeout: Duration,
    pub enable_profiling: bool,
    pub output_directory: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            sizes: vec![32, 64, 128, 256, 512],
            iterations: 10,
            warmup_iterations: 3,
            timeout: Duration::from_secs(300),
            enable_profiling: true,
            output_directory: "target/benchmark_results".to_string(),
        }
    }
}

/// Individual benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub size: usize,
    pub duration: Duration,
    pub flops: usize,
    pub bytes_accessed: usize,
    pub throughput: f64, // Operations per second
    pub bandwidth: f64,  // GB/s
    pub efficiency: f64, // % of theoretical peak
    pub metadata: HashMap<String, String>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite with default configuration
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
            results: HashMap::new(),
        }
    }

    /// Create a benchmark suite with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: HashMap::new(),
        }
    }

    /// Run all tensor operation benchmarks
    pub fn run_tensor_operation_benchmarks(&mut self) {
        println!("Running tensor operation benchmarks...");

        // Run tensor creation benchmarks
        for &size in &self.config.sizes.clone() {
            let mut bench = TensorCreationBench::new(DType::F32);
            let result = self.run_single_benchmark("TensorCreation", &mut bench, size);
            self.store_result("tensor_ops", result);
        }

        // Run tensor arithmetic benchmarks
        for &size in &self.config.sizes.clone() {
            let mut bench =
                TensorArithmeticBench::with_operation(tensor_ops::ArithmeticOp::Addition);
            let result = self.run_single_benchmark("TensorArithmetic_Add", &mut bench, size);
            self.store_result("tensor_ops", result);
        }

        // Run matrix multiplication benchmarks
        for &size in &self.config.sizes.clone() {
            let mut bench = MatmulBench;
            let result = self.run_single_benchmark("MatrixMultiplication", &mut bench, size);
            self.store_result("tensor_ops", result);
        }

        println!("Tensor operation benchmarks completed!");
    }

    /// Run all memory management benchmarks
    pub fn run_memory_benchmarks(&mut self) {
        println!("Running memory management benchmarks...");

        // Run basic memory benchmarks
        for &size in &self.config.sizes.clone() {
            let mut bench = MemoryBench;
            let result = self.run_single_benchmark("Memory_Basic", &mut bench, size);
            self.store_result("memory", result);
        }

        // Run memory fragmentation benchmarks
        for &size in &self.config.sizes.clone() {
            let mut bench = MemoryFragmentationBench::new(1000, 100);
            let result = self.run_single_benchmark("Memory_Fragmentation", &mut bench, size);
            self.store_result("memory", result);
        }

        // Run concurrent memory benchmarks
        for &size in &self.config.sizes.clone() {
            let mut bench = ConcurrentMemoryBench::new(4);
            let result = self.run_single_benchmark("Memory_Concurrent", &mut bench, size);
            self.store_result("memory", result);
        }

        println!("Memory benchmarks completed!");
    }

    /// Run all autograd benchmarks
    pub fn run_autograd_benchmarks(&mut self) {
        println!("Running autograd benchmarks...");

        // Run backward pass benchmarks
        let sizes = self.config.sizes.clone();
        for size in sizes {
            let mut bench = BackwardPassBench;
            let result = self.run_single_benchmark("Autograd_BackwardPass", &mut bench, size);
            self.store_result("autograd", result);
        }

        // Run gradient computation benchmarks
        let sizes = self.config.sizes.clone();
        for size in sizes {
            let mut bench = GradientComputeBench::new(autograd::GradientOp::ElementwiseAdd);
            let result = self.run_single_benchmark("Autograd_GradientCompute", &mut bench, size);
            self.store_result("autograd", result);
        }

        println!("Autograd benchmarks completed!");
    }

    /// Run all data loading benchmarks
    pub fn run_data_loading_benchmarks(&mut self) {
        println!("Running data loading benchmarks...");

        // Run dataloader throughput benchmarks
        let sizes = self.config.sizes.clone();
        for size in sizes {
            let mut bench = DataLoaderThroughputBench;
            let result = self.run_single_benchmark("DataLoader_Throughput", &mut bench, size);
            self.store_result("data_loading", result);
        }

        // Run multi-worker benchmarks
        let sizes = self.config.sizes.clone();
        for size in sizes {
            let mut bench = MultiWorkerDataLoaderBench::new(4);
            let result = self.run_single_benchmark("DataLoader_MultiWorker", &mut bench, size);
            self.store_result("data_loading", result);
        }

        println!("Data loading benchmarks completed!");
    }

    /// Run all optimization benchmarks
    pub fn run_optimization_benchmarks(&mut self) {
        println!("Running optimization benchmarks...");

        // Run kernel fusion benchmarks
        let fusion_types = vec![
            FusionType::ElementwiseActivation,
            FusionType::LinearActivation,
            FusionType::MultipleElementwise,
        ];

        for fusion_type in fusion_types {
            let sizes = self.config.sizes.clone();
            for size in sizes {
                let mut bench = KernelFusionBench::new(fusion_type.clone());
                let result = self.run_single_benchmark(
                    &format!("KernelFusion_{:?}", fusion_type),
                    &mut bench,
                    size,
                );
                self.store_result("optimization", result);
            }
        }

        // Run graph optimization benchmarks
        let opt_types = vec![
            OptimizationType::ConstantFolding,
            OptimizationType::DeadCodeElimination,
            OptimizationType::OperatorFusion,
        ];

        for opt_type in opt_types {
            let sizes = self.config.sizes.clone();
            for size in sizes {
                let mut bench = GraphOptimizationBench::new(opt_type.clone());
                let result = self.run_single_benchmark(
                    &format!("GraphOptimization_{:?}", opt_type),
                    &mut bench,
                    size,
                );
                self.store_result("optimization", result);
            }
        }

        println!("Optimization benchmarks completed!");
    }

    /// Run a single benchmark with timing and statistics
    fn run_single_benchmark<B: Benchmarkable>(
        &self,
        name: &str,
        bench: &mut B,
        size: usize,
    ) -> BenchmarkResult {
        // Warmup runs
        for _ in 0..self.config.warmup_iterations {
            let input = bench.setup(size);
            let _ = bench.run(&input);
        }

        // Actual benchmark runs
        let mut durations = Vec::new();
        for _ in 0..self.config.iterations {
            let input = bench.setup(size);
            let start = Instant::now();
            let _ = bench.run(&input);
            let duration = start.elapsed();
            durations.push(duration);
        }

        // Calculate statistics
        let avg_duration = Duration::from_nanos(
            (durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128) as u64,
        );

        let flops = bench.flops(size);
        let bytes = bench.bytes_accessed(size);

        let throughput = if avg_duration.as_secs_f64() > 0.0 {
            flops as f64 / avg_duration.as_secs_f64()
        } else {
            0.0
        };

        let bandwidth = if avg_duration.as_secs_f64() > 0.0 {
            (bytes as f64 / avg_duration.as_secs_f64()) / (1024.0 * 1024.0 * 1024.0)
        } else {
            0.0
        };

        // Mock efficiency calculation (in real implementation, this would be based on hardware specs)
        let efficiency = (throughput / 1e12).min(1.0) * 100.0; // Assuming 1 TFLOPS peak

        let mut metadata = HashMap::new();
        metadata.insert("iterations".to_string(), self.config.iterations.to_string());
        metadata.insert(
            "warmup_iterations".to_string(),
            self.config.warmup_iterations.to_string(),
        );

        BenchmarkResult {
            name: name.to_string(),
            size,
            duration: avg_duration,
            flops,
            bytes_accessed: bytes,
            throughput,
            bandwidth,
            efficiency,
            metadata,
        }
    }

    /// Store a benchmark result
    fn store_result(&mut self, category: &str, result: BenchmarkResult) {
        self.results
            .entry(category.to_string())
            .or_insert_with(Vec::new)
            .push(result);
    }

    /// Generate comprehensive benchmark report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# ToRSh Benchmark Report\n\n");

        report.push_str(&format!(
            "Generated at: {}\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));
        report.push_str(&format!("Configuration:\n"));
        report.push_str(&format!("- Sizes tested: {:?}\n", self.config.sizes));
        report.push_str(&format!(
            "- Iterations per benchmark: {}\n",
            self.config.iterations
        ));
        report.push_str(&format!(
            "- Warmup iterations: {}\n\n",
            self.config.warmup_iterations
        ));

        for (category, results) in &self.results {
            report.push_str(&format!(
                "## {} Results\n\n",
                category.replace('_', " ").to_uppercase()
            ));

            report.push_str("| Benchmark | Size | Duration (ms) | FLOPS | Throughput (GFLOPS) | Bandwidth (GB/s) | Efficiency (%) |\n");
            report.push_str("|-----------|------|---------------|-------|---------------------|------------------|----------------|\n");

            for result in results {
                report.push_str(&format!(
                    "| {} | {} | {:.3} | {} | {:.2} | {:.2} | {:.1} |\n",
                    result.name,
                    result.size,
                    result.duration.as_secs_f64() * 1000.0,
                    result.flops,
                    result.throughput / 1e9,
                    result.bandwidth,
                    result.efficiency
                ));
            }
            report.push_str("\n");
        }

        report
    }

    /// Export results to CSV
    pub fn export_csv(&self, filename: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filename)?;

        // Write CSV header
        writeln!(file, "Category,Benchmark,Size,Duration_ms,FLOPS,Throughput_GFLOPS,Bandwidth_GB_s,Efficiency_percent")?;

        // Write data
        for (category, results) in &self.results {
            for result in results {
                writeln!(
                    file,
                    "{},{},{},{:.3},{},{:.2},{:.2},{:.1}",
                    category,
                    result.name,
                    result.size,
                    result.duration.as_secs_f64() * 1000.0,
                    result.flops,
                    result.throughput / 1e9,
                    result.bandwidth,
                    result.efficiency
                )?;
            }
        }

        Ok(())
    }

    /// Get summary statistics for a category
    pub fn get_category_summary(&self, category: &str) -> Option<CategorySummary> {
        let results = self.results.get(category)?;

        if results.is_empty() {
            return None;
        }

        let total_benchmarks = results.len();
        let avg_throughput =
            results.iter().map(|r| r.throughput).sum::<f64>() / total_benchmarks as f64;
        let avg_bandwidth =
            results.iter().map(|r| r.bandwidth).sum::<f64>() / total_benchmarks as f64;
        let avg_efficiency =
            results.iter().map(|r| r.efficiency).sum::<f64>() / total_benchmarks as f64;

        let max_throughput = results.iter().map(|r| r.throughput).fold(0.0, f64::max);
        let min_throughput = results
            .iter()
            .map(|r| r.throughput)
            .fold(f64::INFINITY, f64::min);

        Some(CategorySummary {
            category: category.to_string(),
            total_benchmarks,
            avg_throughput,
            avg_bandwidth,
            avg_efficiency,
            max_throughput,
            min_throughput,
        })
    }
}

/// Summary statistics for a benchmark category
#[derive(Debug, Clone)]
pub struct CategorySummary {
    pub category: String,
    pub total_benchmarks: usize,
    pub avg_throughput: f64,
    pub avg_bandwidth: f64,
    pub avg_efficiency: f64,
    pub max_throughput: f64,
    pub min_throughput: f64,
}

// ================================================================================================
// Convenience Functions for Backward Compatibility
// ================================================================================================

/// Run comprehensive benchmark suite (legacy compatibility function)
///
/// This function maintains compatibility with the original monolithic implementation
/// and runs all benchmark categories.
pub fn run_comprehensive_benchmarks() {
    let mut suite = BenchmarkSuite::new();

    suite.run_tensor_operation_benchmarks();
    suite.run_memory_benchmarks();
    suite.run_autograd_benchmarks();
    suite.run_data_loading_benchmarks();
    suite.run_optimization_benchmarks();

    let report = suite.generate_report();
    println!("{}", report);

    // Export results
    if let Err(e) = suite.export_csv("target/comprehensive_benchmark_results.csv") {
        eprintln!("Failed to export CSV: {}", e);
    }
}

/// Run kernel fusion benchmark suite (legacy compatibility)
pub fn run_kernel_fusion_benchmarks() {
    let results = crate::benchmarks::optimization::run_kernel_fusion_benchmarks();

    println!("Kernel Fusion Benchmark Results:");
    for (fusion_type, speedup) in results {
        println!("  {:?}: {:.2}x speedup", fusion_type, speedup);
    }
}

/// Run graph optimization benchmark suite (legacy compatibility)
pub fn run_graph_optimization_benchmarks() {
    let results = crate::benchmarks::optimization::run_graph_optimization_benchmarks();

    println!("Graph Optimization Benchmark Results:");
    for (opt_type, speedup) in results {
        println!("  {:?}: {:.2}x speedup", opt_type, speedup);
    }
}

// ================================================================================================
// Module Metadata and Information
// ================================================================================================

/// Information about the modular benchmark system
pub mod info {
    /// Version of the modular benchmark system
    pub const VERSION: &str = "2.0.0";

    /// Original monolithic file size (lines of code)
    pub const ORIGINAL_LOC: usize = 2041;

    /// Number of specialized modules created
    pub const MODULE_COUNT: usize = 6;

    /// Refactoring phase number
    pub const PHASE: u32 = 47;

    /// Date of modular refactoring completion
    pub const REFACTOR_DATE: &str = "2025-09-22";

    /// Get information about the benchmark system
    pub fn system_info() -> String {
        format!(
            "ToRSh Modular Benchmark System v{}\n\
             Refactored from {} lines into {} specialized modules\n\
             Phase {} completed on {}\n\
             Architecture: common, tensor_ops, memory, autograd, data_loading, optimization",
            VERSION, ORIGINAL_LOC, MODULE_COUNT, PHASE, REFACTOR_DATE
        )
    }
}

// ================================================================================================
// Comprehensive Test Suite
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = BenchmarkSuite::new();
        assert_eq!(suite.config.sizes, vec![32, 64, 128, 256, 512]);
        assert_eq!(suite.config.iterations, 10);
        assert_eq!(suite.config.warmup_iterations, 3);
        assert!(suite.results.is_empty());
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig {
            sizes: vec![16, 32, 64],
            iterations: 5,
            warmup_iterations: 1,
            timeout: Duration::from_secs(60),
            enable_profiling: false,
            output_directory: "/tmp/benchmarks".to_string(),
        };

        let suite = BenchmarkSuite::with_config(config);
        assert_eq!(suite.config.sizes, vec![16, 32, 64]);
        assert_eq!(suite.config.iterations, 5);
    }

    #[test]
    fn test_benchmark_result_creation() {
        let mut metadata = HashMap::new();
        metadata.insert("test".to_string(), "value".to_string());

        let result = BenchmarkResult {
            name: "TestBench".to_string(),
            size: 64,
            duration: Duration::from_millis(10),
            flops: 1000,
            bytes_accessed: 2000,
            throughput: 100.0,
            bandwidth: 50.0,
            efficiency: 85.0,
            metadata,
        };

        assert_eq!(result.name, "TestBench");
        assert_eq!(result.size, 64);
        assert_eq!(result.flops, 1000);
    }

    #[test]
    fn test_single_benchmark_run() {
        let suite = BenchmarkSuite::new();
        let mut bench = TensorCreationBench::new(DType::F32);

        let result = suite.run_single_benchmark("TestTensorCreation", &mut bench, 32);

        assert_eq!(result.name, "TestTensorCreation");
        assert_eq!(result.size, 32);
        assert!(result.duration > Duration::ZERO);
        assert!(result.flops > 0);
    }

    #[test]
    fn test_category_summary() {
        let mut suite = BenchmarkSuite::new();

        // Add some mock results
        let result1 = BenchmarkResult {
            name: "Test1".to_string(),
            size: 64,
            duration: Duration::from_millis(10),
            flops: 1000,
            bytes_accessed: 2000,
            throughput: 100.0,
            bandwidth: 50.0,
            efficiency: 80.0,
            metadata: HashMap::new(),
        };

        let result2 = BenchmarkResult {
            name: "Test2".to_string(),
            size: 64,
            duration: Duration::from_millis(5),
            flops: 2000,
            bytes_accessed: 4000,
            throughput: 200.0,
            bandwidth: 100.0,
            efficiency: 90.0,
            metadata: HashMap::new(),
        };

        suite.store_result("test_category", result1);
        suite.store_result("test_category", result2);

        let summary = suite.get_category_summary("test_category").unwrap();
        assert_eq!(summary.total_benchmarks, 2);
        assert_eq!(summary.avg_throughput, 150.0);
        assert_eq!(summary.avg_bandwidth, 75.0);
        assert_eq!(summary.avg_efficiency, 85.0);
        assert_eq!(summary.max_throughput, 200.0);
        assert_eq!(summary.min_throughput, 100.0);
    }

    #[test]
    fn test_report_generation() {
        let mut suite = BenchmarkSuite::new();

        let result = BenchmarkResult {
            name: "TestBench".to_string(),
            size: 64,
            duration: Duration::from_millis(10),
            flops: 1000,
            bytes_accessed: 2000,
            throughput: 100.0,
            bandwidth: 50.0,
            efficiency: 80.0,
            metadata: HashMap::new(),
        };

        suite.store_result("test", result);
        let report = suite.generate_report();

        assert!(report.contains("ToRSh Benchmark Report"));
        assert!(report.contains("TEST Results"));
        assert!(report.contains("TestBench"));
    }

    #[test]
    fn test_system_info() {
        let info = info::system_info();
        assert!(info.contains("ToRSh Modular Benchmark System"));
        assert!(info.contains("2041 lines"));
        assert!(info.contains("6 specialized modules"));
        assert!(info.contains("Phase 47"));
    }

    #[test]
    fn test_module_constants() {
        assert_eq!(info::VERSION, "2.0.0");
        assert_eq!(info::ORIGINAL_LOC, 2041);
        assert_eq!(info::MODULE_COUNT, 6);
        assert_eq!(info::PHASE, 47);
        assert_eq!(info::REFACTOR_DATE, "2025-09-22");
    }

    #[test]
    fn test_all_modules_accessible() {
        // Test that all benchmark types are accessible through the unified interface
        let _tensor_bench = TensorCreationBench::new(DType::F32);
        let _memory_bench = MemoryBench;
        let _autograd_bench = BackwardPassBench;
        let _data_bench = DataLoaderThroughputBench;
        let _fusion_bench = KernelFusionBench::new(FusionType::ElementwiseActivation);
        let _graph_bench = GraphOptimizationBench::new(OptimizationType::ConstantFolding);
    }

    #[test]
    fn test_legacy_compatibility_functions() {
        // These should not panic and should execute successfully
        let fusion_results = crate::benchmarks::optimization::run_kernel_fusion_benchmarks();
        assert!(!fusion_results.is_empty());

        let graph_results = crate::benchmarks::optimization::run_graph_optimization_benchmarks();
        assert!(!graph_results.is_empty());
    }
}
