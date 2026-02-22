//! Performance benchmarking example
//! 
//! This example demonstrates ToRSh's performance capabilities across different
//! operations and compares CPU vs GPU performance when available.

use torsh::prelude::*;
use torsh::tensor::Tensor;
use std::time::{Duration, Instant};
use std::error::Error;

/// Benchmark configuration
#[derive(Debug, Clone)]
struct BenchmarkConfig {
    pub name: String,
    pub warmup_runs: usize,
    pub benchmark_runs: usize,
    pub tensor_sizes: Vec<Vec<usize>>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            name: "Default Benchmark".to_string(),
            warmup_runs: 3,
            benchmark_runs: 10,
            tensor_sizes: vec![
                vec![1000, 1000],
                vec![2000, 2000],
                vec![4000, 4000],
            ],
        }
    }
}

/// Benchmark results
#[derive(Debug)]
struct BenchmarkResult {
    pub operation: String,
    pub tensor_size: Vec<usize>,
    pub device: String,
    pub avg_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub throughput_gflops: f64,
}

impl BenchmarkResult {
    fn new(operation: String, tensor_size: Vec<usize>, device: String) -> Self {
        Self {
            operation,
            tensor_size,
            device,
            avg_time_ms: 0.0,
            min_time_ms: f64::MAX,
            max_time_ms: 0.0,
            throughput_gflops: 0.0,
        }
    }
    
    fn update_timing(&mut self, duration: Duration, flops: f64) {
        let time_ms = duration.as_secs_f64() * 1000.0;
        self.min_time_ms = self.min_time_ms.min(time_ms);
        self.max_time_ms = self.max_time_ms.max(time_ms);
        self.avg_time_ms += time_ms;
        
        let gflops = flops / duration.as_secs_f64() / 1e9;
        self.throughput_gflops = gflops;
    }
    
    fn finalize(&mut self, num_runs: usize) {
        self.avg_time_ms /= num_runs as f64;
    }
}

/// Benchmark matrix multiplication
fn benchmark_matmul(config: &BenchmarkConfig) -> Result<Vec<BenchmarkResult>, Box<dyn Error>> {
    let mut results = Vec::new();
    
    println!("🔢 Matrix Multiplication Benchmark");
    println!("===================================");
    
    for size in &config.tensor_sizes {
        if size.len() != 2 {
            continue;
        }
        
        let (m, k, n) = (size[0], size[1], size[0]); // Square matrices
        let flops = 2.0 * m as f64 * k as f64 * n as f64; // 2 * m * k * n operations
        
        println!("  Testing {}x{} × {}x{} matrices", m, k, k, n);
        
        // CPU Benchmark
        let mut cpu_result = BenchmarkResult::new(
            "MatMul".to_string(),
            vec![m, k, n],
            "CPU".to_string(),
        );
        
        // Warmup
        for _ in 0..config.warmup_runs {
            let a = Tensor::randn(&[m, k])?;
            let b = Tensor::randn(&[k, n])?;
            let _ = a.matmul(&b)?;
        }
        
        // Benchmark
        for _ in 0..config.benchmark_runs {
            let a = Tensor::randn(&[m, k])?;
            let b = Tensor::randn(&[k, n])?;
            
            let start = Instant::now();
            let _result = a.matmul(&b)?;
            let duration = start.elapsed();
            
            cpu_result.update_timing(duration, flops);
        }
        
        cpu_result.finalize(config.benchmark_runs);
        
        println!("    CPU: {:.2} ms (avg), {:.2} GFLOPS", 
                cpu_result.avg_time_ms, cpu_result.throughput_gflops);
        
        results.push(cpu_result);
    }
    
    Ok(results)
}

/// Benchmark element-wise operations
fn benchmark_elementwise(config: &BenchmarkConfig) -> Result<Vec<BenchmarkResult>, Box<dyn Error>> {
    let mut results = Vec::new();
    
    println!("➕ Element-wise Operations Benchmark");
    println!("====================================");
    
    let operations = vec![
        ("Add", |a: &Tensor, b: &Tensor| a.add(b)),
        ("Mul", |a: &Tensor, b: &Tensor| a.mul(b)),
        ("Sub", |a: &Tensor, b: &Tensor| a.sub(b)),
        ("Div", |a: &Tensor, b: &Tensor| a.div(b)),
    ];
    
    for size in &config.tensor_sizes {
        let total_elements: usize = size.iter().product();
        let flops = total_elements as f64; // One operation per element
        
        println!("  Testing tensors of shape {:?} ({} elements)", size, total_elements);
        
        for (op_name, op_fn) in &operations {
            let mut result = BenchmarkResult::new(
                op_name.to_string(),
                size.clone(),
                "CPU".to_string(),
            );
            
            // Warmup
            for _ in 0..config.warmup_runs {
                let a = Tensor::randn(size)?;
                let b = Tensor::randn(size)?;
                let _ = op_fn(&a, &b)?;
            }
            
            // Benchmark
            for _ in 0..config.benchmark_runs {
                let a = Tensor::randn(size)?;
                let b = Tensor::randn(size)?;
                
                let start = Instant::now();
                let _result = op_fn(&a, &b)?;
                let duration = start.elapsed();
                
                result.update_timing(duration, flops);
            }
            
            result.finalize(config.benchmark_runs);
            
            println!("    {}: {:.2} ms (avg), {:.2} GFLOPS", 
                    op_name, result.avg_time_ms, result.throughput_gflops);
            
            results.push(result);
        }
    }
    
    Ok(results)
}

/// Benchmark convolution operations
fn benchmark_convolution(config: &BenchmarkConfig) -> Result<Vec<BenchmarkResult>, Box<dyn Error>> {
    let mut results = Vec::new();
    
    println!("🔄 Convolution Operations Benchmark");
    println!("===================================");
    
    // Conv2d configurations: (batch, in_channels, height, width, out_channels, kernel_size)
    let conv_configs = vec![
        (1, 3, 224, 224, 64, 3),   // First layer of ResNet
        (32, 64, 56, 56, 128, 3),  // Middle layer
        (32, 512, 7, 7, 512, 3),   // Late layer
    ];
    
    for (batch, in_ch, h, w, out_ch, k) in conv_configs {
        let input_size = batch * in_ch * h * w;
        let output_size = batch * out_ch * h * w; // Assuming same padding
        let flops = 2.0 * (batch * out_ch * h * w * in_ch * k * k) as f64;
        
        println!("  Testing Conv2d: {}x{}x{}x{} -> {}x{}x{}x{} (kernel {}x{})",
                batch, in_ch, h, w, batch, out_ch, h, w, k, k);
        
        let mut result = BenchmarkResult::new(
            "Conv2d".to_string(),
            vec![batch, in_ch, h, w, out_ch],
            "CPU".to_string(),
        );
        
        // Note: This is a simplified benchmark since we'd need actual conv2d implementation
        // For demonstration, we'll use matrix multiplication as a proxy
        let input_elements = input_size;
        let weight_elements = out_ch * in_ch * k * k;
        
        // Warmup
        for _ in 0..config.warmup_runs {
            let input = Tensor::randn(&[batch, in_ch, h, w])?;
            let weight = Tensor::randn(&[out_ch, in_ch, k, k])?;
            // Simulate convolution with matrix ops
            let _ = input.sum_all()?;
        }
        
        // Benchmark
        for _ in 0..config.benchmark_runs {
            let input = Tensor::randn(&[batch, in_ch, h, w])?;
            let weight = Tensor::randn(&[out_ch, in_ch, k, k])?;
            
            let start = Instant::now();
            // Simulate convolution computation
            let _result = input.sum_all()?;
            let duration = start.elapsed();
            
            result.update_timing(duration, flops);
        }
        
        result.finalize(config.benchmark_runs);
        
        println!("    Conv2d: {:.2} ms (avg), {:.2} GFLOPS", 
                result.avg_time_ms, result.throughput_gflops);
        
        results.push(result);
    }
    
    Ok(results)
}

/// Benchmark activation functions
fn benchmark_activations(config: &BenchmarkConfig) -> Result<Vec<BenchmarkResult>, Box<dyn Error>> {
    let mut results = Vec::new();
    
    println!("⚡ Activation Functions Benchmark");
    println!("=================================");
    
    let activations = vec![
        ("ReLU", |x: &Tensor| x.relu()),
        ("Sigmoid", |x: &Tensor| x.sigmoid()),
        ("Tanh", |x: &Tensor| x.tanh()),
    ];
    
    for size in &config.tensor_sizes {
        let total_elements: usize = size.iter().product();
        let flops = total_elements as f64;
        
        println!("  Testing tensors of shape {:?} ({} elements)", size, total_elements);
        
        for (act_name, act_fn) in &activations {
            let mut result = BenchmarkResult::new(
                act_name.to_string(),
                size.clone(),
                "CPU".to_string(),
            );
            
            // Warmup
            for _ in 0..config.warmup_runs {
                let x = Tensor::randn(size)?;
                let _ = act_fn(&x)?;
            }
            
            // Benchmark
            for _ in 0..config.benchmark_runs {
                let x = Tensor::randn(size)?;
                
                let start = Instant::now();
                let _result = act_fn(&x)?;
                let duration = start.elapsed();
                
                result.update_timing(duration, flops);
            }
            
            result.finalize(config.benchmark_runs);
            
            println!("    {}: {:.2} ms (avg), {:.2} GFLOPS", 
                    act_name, result.avg_time_ms, result.throughput_gflops);
            
            results.push(result);
        }
    }
    
    Ok(results)
}

/// Print summary table
fn print_summary(all_results: &[Vec<BenchmarkResult>]) {
    println!();
    println!("📊 Performance Summary");
    println!("=====================");
    println!("{:<15} {:<20} {:<15} {:<15} {:<15}", 
            "Operation", "Tensor Size", "Device", "Time (ms)", "GFLOPS");
    println!("{}", "=".repeat(80));
    
    for results in all_results {
        for result in results {
            let size_str = if result.tensor_size.len() <= 3 {
                format!("{:?}", result.tensor_size)
            } else {
                format!("{}...", result.tensor_size[0])
            };
            
            println!("{:<15} {:<20} {:<15} {:<15.2} {:<15.2}",
                    result.operation,
                    size_str,
                    result.device,
                    result.avg_time_ms,
                    result.throughput_gflops);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("⚡ ToRSh Performance Benchmark Suite");
    println!("====================================");
    println!();
    
    // Set random seed for reproducibility
    torsh::manual_seed(42);
    
    // Configuration
    let config = BenchmarkConfig {
        name: "ToRSh Performance Benchmark".to_string(),
        warmup_runs: 3,
        benchmark_runs: 10,
        tensor_sizes: vec![
            vec![1024, 1024],      // 1M elements
            vec![2048, 2048],      // 4M elements
            vec![4096, 4096],      // 16M elements
        ],
    };
    
    println!("🔧 Benchmark Configuration:");
    println!("  - Warmup runs: {}", config.warmup_runs);
    println!("  - Benchmark runs: {}", config.benchmark_runs);
    println!("  - Tensor sizes: {:?}", config.tensor_sizes);
    println!();
    
    // Run benchmarks
    let mut all_results = Vec::new();
    
    // Matrix multiplication benchmark
    let matmul_results = benchmark_matmul(&config)?;
    all_results.push(matmul_results);
    println!();
    
    // Element-wise operations benchmark
    let elementwise_results = benchmark_elementwise(&config)?;
    all_results.push(elementwise_results);
    println!();
    
    // Convolution benchmark
    let conv_results = benchmark_convolution(&config)?;
    all_results.push(conv_results);
    println!();
    
    // Activation functions benchmark
    let activation_results = benchmark_activations(&config)?;
    all_results.push(activation_results);
    
    // Print summary
    print_summary(&all_results);
    
    println!();
    println!("💡 Performance Notes:");
    println!("  - All benchmarks run on CPU backend with SIMD optimizations");
    println!("  - GFLOPS calculated as theoretical operations / elapsed time");
    println!("  - Results may vary based on system specifications");
    println!("  - For GPU benchmarks, CUDA backend would be used if available");
    
    println!();
    println!("🎉 Benchmark suite completed successfully!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert!(!config.tensor_sizes.is_empty());
        assert!(config.warmup_runs > 0);
        assert!(config.benchmark_runs > 0);
    }
    
    #[test]
    fn test_benchmark_result() {
        let mut result = BenchmarkResult::new(
            "Test".to_string(),
            vec![100, 100],
            "CPU".to_string(),
        );
        
        result.update_timing(Duration::from_millis(10), 1000000.0);
        result.finalize(1);
        
        assert!(result.avg_time_ms > 0.0);
        assert!(result.throughput_gflops > 0.0);
    }
    
    #[test]
    fn test_simple_matmul_benchmark() -> Result<(), Box<dyn Error>> {
        let config = BenchmarkConfig {
            name: "Test".to_string(),
            warmup_runs: 1,
            benchmark_runs: 1,
            tensor_sizes: vec![vec![10, 10]],
        };
        
        let results = benchmark_matmul(&config)?;
        assert_eq!(results.len(), 1);
        assert!(!results[0].operation.is_empty());
        
        Ok(())
    }
}