# ToRSh Benchmarking Methodology

## Overview

This document outlines the systematic approach, methodologies, and best practices used in the ToRSh benchmarking suite to ensure reproducible, reliable, and meaningful performance measurements.

## Core Principles

### 1. **Reproducibility**
All benchmarks must be reproducible across different environments, hardware configurations, and time periods.

**Implementation:**
- Fixed random seeds for data generation
- Deterministic algorithms where possible
- Documented environmental requirements
- Version-locked dependencies

### 2. **Statistical Rigor**
Performance measurements must be statistically valid and account for variance in execution times.

**Implementation:**
- Sufficient sample sizes (minimum 30 iterations)
- Warm-up phases to eliminate cold-start effects
- Outlier detection and handling
- Confidence interval reporting

### 3. **Real-World Relevance**
Benchmarks should reflect actual usage patterns and constraints.

**Implementation:**
- Representative model architectures
- Realistic data distributions
- Production-like batch sizes
- End-to-end workflows

### 4. **Comprehensive Coverage**
The benchmark suite covers all critical performance aspects.

**Implementation:**
- Multiple hardware targets (CPU, GPU, mobile)
- Various precision levels (FP32, FP16, INT8)
- Different problem sizes and batch configurations
- Memory and compute-bound scenarios

## Benchmark Design Methodology

### 1. **Operation Classification**

#### Compute-Bound Operations
**Characteristics:**
- High FLOPS requirements
- Low memory bandwidth utilization
- Examples: Large matrix multiplies, convolutions with small kernels

**Benchmark Focus:**
- FLOPS throughput
- Numerical precision
- Hardware utilization
- Scaling with problem size

```rust
// Example: Compute-bound benchmark
impl Benchmark for ComputeBoundMatMul {
    fn setup(&mut self, config: &BenchConfig) -> Result<()> {
        // Large matrices that exceed cache
        self.a = Tensor::randn(vec![2048, 2048]);
        self.b = Tensor::randn(vec![2048, 2048]);
        Ok(())
    }
    
    fn run_iteration(&mut self, _iteration: usize) -> Result<BenchmarkResult> {
        let start = Instant::now();
        let _result = self.a.matmul(&self.b)?;
        let duration = start.elapsed();
        
        Ok(BenchmarkResult {
            duration,
            flops: Some(2.0 * 2048.0_f64.powi(3)), // 2 * n^3 for matrix multiply
            memory_bandwidth: None, // Not the bottleneck
            ..Default::default()
        })
    }
}
```

#### Memory-Bound Operations
**Characteristics:**
- High memory bandwidth requirements
- Low computational intensity
- Examples: Element-wise operations, large tensor copies

**Benchmark Focus:**
- Memory bandwidth utilization
- Cache efficiency
- Data access patterns
- Memory hierarchy optimization

```rust
// Example: Memory-bound benchmark
impl Benchmark for MemoryBoundElementWise {
    fn setup(&mut self, config: &BenchConfig) -> Result<()> {
        // Large tensors to exceed cache
        let size = 100_000_000; // ~400MB for f32
        self.a = Tensor::randn(vec![size]);
        self.b = Tensor::randn(vec![size]);
        Ok(())
    }
    
    fn run_iteration(&mut self, _iteration: usize) -> Result<BenchmarkResult> {
        let start = Instant::now();
        let _result = self.a.add(&self.b)?;
        let duration = start.elapsed();
        
        let bytes_accessed = 3 * size * 4; // 3 arrays * 4 bytes per f32
        let bandwidth = bytes_accessed as f64 / duration.as_secs_f64() / 1e9;
        
        Ok(BenchmarkResult {
            duration,
            memory_bandwidth: Some(bandwidth),
            flops: None, // Minimal computation
            ..Default::default()
        })
    }
}
```

### 2. **Problem Size Selection**

#### Small Problems (Cache-Friendly)
- **Size Range**: Fits in L1/L2 cache (< 1MB)
- **Purpose**: Test algorithm efficiency and cache optimization
- **Examples**: Small matrix operations, kernel microbenchmarks

#### Medium Problems (Memory Hierarchy)
- **Size Range**: Fits in L3 cache (1-50MB)
- **Purpose**: Test cache management and memory access patterns
- **Examples**: Medium neural network layers, moderate batch sizes

#### Large Problems (Memory-Bound)
- **Size Range**: Exceeds cache (> 50MB)
- **Purpose**: Test memory bandwidth and scalability
- **Examples**: Large model inference, high-throughput training

### 3. **Batch Size Methodology**

#### Latency-Oriented Benchmarks
```rust
// Test single sample inference
let batch_sizes = vec![1];
```

#### Throughput-Oriented Benchmarks
```rust
// Test various batch sizes to find optimal throughput
let batch_sizes = vec![1, 2, 4, 8, 16, 32, 64, 128, 256];
```

#### Memory-Constraint Benchmarks
```rust
// Test maximum feasible batch size
let max_memory = get_available_memory();
let max_batch = calculate_max_batch_size(model_size, max_memory);
let batch_sizes = (1..=max_batch).step_by(max_batch / 10).collect();
```

## Statistical Methodology

### 1. **Timing Measurement**

#### Warm-up Protocol
```rust
impl BenchmarkRunner {
    fn run_benchmark(&mut self, benchmark: &mut dyn Benchmark) -> Result<BenchmarkResult> {
        // Warm-up phase
        for _ in 0..self.config.warmup_iterations {
            benchmark.run_iteration(0)?;
        }
        
        // Clear any performance counters
        self.reset_performance_counters();
        
        // Measurement phase
        let mut measurements = Vec::new();
        for i in 0..self.config.measurement_iterations {
            let result = benchmark.run_iteration(i)?;
            measurements.push(result);
        }
        
        self.aggregate_measurements(measurements)
    }
}
```

#### Outlier Detection
```rust
fn remove_outliers(measurements: &mut Vec<Duration>) {
    measurements.sort();
    let q1 = measurements[measurements.len() / 4];
    let q3 = measurements[3 * measurements.len() / 4];
    let iqr = q3 - q1;
    let lower_bound = q1 - 1.5 * iqr;
    let upper_bound = q3 + 1.5 * iqr;
    
    measurements.retain(|&x| x >= lower_bound && x <= upper_bound);
}
```

#### Statistical Aggregation
```rust
struct StatisticalSummary {
    mean: Duration,
    median: Duration,
    std_dev: Duration,
    min: Duration,
    max: Duration,
    confidence_interval_95: (Duration, Duration),
    sample_size: usize,
}

impl StatisticalSummary {
    fn from_measurements(measurements: &[Duration]) -> Self {
        let mean = measurements.iter().sum::<Duration>() / measurements.len() as u32;
        let median = measurements[measurements.len() / 2];
        
        // Calculate standard deviation
        let variance = measurements.iter()
            .map(|&x| (x.as_nanos() as f64 - mean.as_nanos() as f64).powi(2))
            .sum::<f64>() / measurements.len() as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);
        
        // 95% confidence interval using t-distribution
        let t_value = get_t_value(measurements.len() - 1, 0.05);
        let margin_of_error = t_value * std_dev.as_nanos() as f64 / (measurements.len() as f64).sqrt();
        let confidence_interval_95 = (
            Duration::from_nanos((mean.as_nanos() as f64 - margin_of_error) as u64),
            Duration::from_nanos((mean.as_nanos() as f64 + margin_of_error) as u64),
        );
        
        Self {
            mean,
            median,
            std_dev,
            min: *measurements.iter().min().unwrap(),
            max: *measurements.iter().max().unwrap(),
            confidence_interval_95,
            sample_size: measurements.len(),
        }
    }
}
```

### 2. **Comparative Analysis**

#### Hypothesis Testing
```rust
fn compare_benchmarks(baseline: &[Duration], treatment: &[Duration]) -> ComparisonResult {
    // Welch's t-test for unequal variances
    let t_statistic = calculate_welch_t_statistic(baseline, treatment);
    let p_value = calculate_p_value(t_statistic, baseline.len(), treatment.len());
    
    let baseline_mean = baseline.iter().sum::<Duration>() / baseline.len() as u32;
    let treatment_mean = treatment.iter().sum::<Duration>() / treatment.len() as u32;
    let speedup = baseline_mean.as_nanos() as f64 / treatment_mean.as_nanos() as f64;
    
    ComparisonResult {
        speedup,
        p_value,
        statistically_significant: p_value < 0.05,
        effect_size: calculate_cohens_d(baseline, treatment),
    }
}
```

#### Regression Detection
```rust
struct RegressionDetector {
    baseline_repository: BaselineRepository,
    significance_threshold: f64,
    minimum_effect_size: f64,
}

impl RegressionDetector {
    fn detect_regression(&self, current_results: &BenchmarkResults) -> RegressionReport {
        let baseline = self.baseline_repository.get_baseline(&current_results.benchmark_id)?;
        let comparison = compare_benchmarks(&baseline.measurements, &current_results.measurements);
        
        let is_regression = comparison.speedup < 1.0 
            && comparison.statistically_significant
            && comparison.effect_size > self.minimum_effect_size;
            
        RegressionReport {
            benchmark_id: current_results.benchmark_id.clone(),
            is_regression,
            performance_change: (comparison.speedup - 1.0) * 100.0,
            confidence_level: 1.0 - comparison.p_value,
            recommendation: self.generate_recommendation(&comparison),
        }
    }
}
```

## Hardware-Specific Methodologies

### 1. **GPU Benchmarking**

#### CUDA Kernel Profiling
```rust
impl CudaBenchmark {
    fn run_with_profiling(&mut self) -> Result<CudaBenchmarkResult> {
        // Warm up GPU
        self.warmup_gpu()?;
        
        // Start CUDA profiling
        cuda_profiler_start()?;
        
        // Create CUDA events for precise timing
        let start_event = CudaEvent::new()?;
        let end_event = CudaEvent::new()?;
        
        start_event.record()?;
        
        // Execute kernel
        self.execute_kernel()?;
        
        end_event.record()?;
        end_event.synchronize()?;
        
        let kernel_time = start_event.elapsed_time(&end_event)?;
        
        cuda_profiler_stop()?;
        
        // Get additional metrics
        let occupancy = self.calculate_occupancy()?;
        let memory_throughput = self.measure_memory_throughput()?;
        
        Ok(CudaBenchmarkResult {
            kernel_time,
            occupancy,
            memory_throughput,
            profiling_data: self.extract_profiling_data()?,
        })
    }
}
```

#### Memory Bandwidth Measurement
```rust
fn measure_memory_bandwidth() -> Result<f64> {
    let size = 1_000_000_000; // 1GB
    let iterations = 10;
    
    let src = cuda_malloc(size)?;
    let dst = cuda_malloc(size)?;
    
    // Warm up
    for _ in 0..5 {
        cuda_memcpy(dst, src, size)?;
    }
    
    let start = Instant::now();
    for _ in 0..iterations {
        cuda_memcpy(dst, src, size)?;
    }
    cuda_device_synchronize()?;
    let duration = start.elapsed();
    
    let bytes_transferred = size * iterations;
    let bandwidth = bytes_transferred as f64 / duration.as_secs_f64() / 1e9; // GB/s
    
    cuda_free(src)?;
    cuda_free(dst)?;
    
    Ok(bandwidth)
}
```

### 2. **CPU Benchmarking**

#### SIMD Utilization Detection
```rust
fn measure_simd_utilization() -> SIMDMetrics {
    let performance_counters = PerformanceCounters::new(&[
        "INST_RETIRED.ANY",           // Total instructions
        "FP_ARITH_INST_RETIRED.SCALAR_SINGLE",  // Scalar FP instructions
        "FP_ARITH_INST_RETIRED.256B_PACKED_SINGLE", // AVX instructions
        "FP_ARITH_INST_RETIRED.512B_PACKED_SINGLE", // AVX-512 instructions
    ])?;
    
    performance_counters.start()?;
    
    // Execute benchmark
    run_vector_operation()?;
    
    let counters = performance_counters.read()?;
    
    let total_fp_ops = counters["FP_ARITH_INST_RETIRED.SCALAR_SINGLE"]
        + counters["FP_ARITH_INST_RETIRED.256B_PACKED_SINGLE"] * 8
        + counters["FP_ARITH_INST_RETIRED.512B_PACKED_SINGLE"] * 16;
        
    let vectorized_ops = counters["FP_ARITH_INST_RETIRED.256B_PACKED_SINGLE"] * 8
        + counters["FP_ARITH_INST_RETIRED.512B_PACKED_SINGLE"] * 16;
    
    SIMDMetrics {
        vectorization_ratio: vectorized_ops as f64 / total_fp_ops as f64,
        avx_utilization: counters["FP_ARITH_INST_RETIRED.256B_PACKED_SINGLE"],
        avx512_utilization: counters["FP_ARITH_INST_RETIRED.512B_PACKED_SINGLE"],
    }
}
```

#### Cache Performance Analysis
```rust
fn analyze_cache_performance() -> CacheMetrics {
    let performance_counters = PerformanceCounters::new(&[
        "L1D.REPLACEMENT",
        "L2_RQSTS.ALL_DEMAND_MISS",
        "LLC_MISSES",
        "MEM_LOAD_RETIRED.L1_HIT",
        "MEM_LOAD_RETIRED.L2_HIT",
        "MEM_LOAD_RETIRED.L3_HIT",
    ])?;
    
    performance_counters.start()?;
    run_memory_intensive_operation()?;
    let counters = performance_counters.read()?;
    
    let l1_hits = counters["MEM_LOAD_RETIRED.L1_HIT"];
    let l2_hits = counters["MEM_LOAD_RETIRED.L2_HIT"];
    let l3_hits = counters["MEM_LOAD_RETIRED.L3_HIT"];
    let total_accesses = l1_hits + l2_hits + l3_hits + counters["LLC_MISSES"];
    
    CacheMetrics {
        l1_hit_rate: l1_hits as f64 / total_accesses as f64,
        l2_hit_rate: l2_hits as f64 / total_accesses as f64,
        l3_hit_rate: l3_hits as f64 / total_accesses as f64,
        memory_bandwidth: calculate_memory_bandwidth(&counters),
    }
}
```

## Cross-Framework Comparison Methodology

### 1. **Environment Standardization**

#### Dependency Management
```toml
# Cargo.toml - Lock all versions for reproducibility
[dependencies]
pytorch = "=2.1.0"
tensorflow = "=2.13.0"
jax = "=0.4.13"
numpy = "=1.24.3"
```

#### Environment Setup
```rust
fn setup_comparison_environment() -> Result<ComparisonEnvironment> {
    // Set deterministic behavior
    set_random_seed(42);
    
    // Configure PyTorch
    torch::manual_seed(42);
    torch::cuda::manual_seed_all(42);
    torch::set_num_threads(1); // Single-threaded for fairness
    
    // Configure TensorFlow
    tf::random::set_seed(42);
    tf::config::threading::set_inter_op_parallelism_threads(1);
    tf::config::threading::set_intra_op_parallelism_threads(1);
    
    // Configure JAX
    jax::random::PRNGKey(42);
    jax::config::update("jax_platform_name", "cpu"); // or "gpu"
    
    Ok(ComparisonEnvironment {
        torch_version: get_torch_version(),
        tf_version: get_tf_version(),
        jax_version: get_jax_version(),
        hardware_info: get_hardware_info(),
    })
}
```

### 2. **Operation Equivalence Verification**

#### Numerical Accuracy Checking
```rust
fn verify_operation_equivalence() -> Result<EquivalenceReport> {
    let input = generate_test_tensor();
    
    // Run same operation in all frameworks
    let torsh_result = torsh_operation(&input)?;
    let pytorch_result = pytorch_operation(&input)?;
    let tf_result = tensorflow_operation(&input)?;
    
    // Check numerical equivalence
    let torsh_pytorch_diff = calculate_max_difference(&torsh_result, &pytorch_result);
    let torsh_tf_diff = calculate_max_difference(&torsh_result, &tf_result);
    
    let tolerance = 1e-6; // FP32 precision tolerance
    
    EquivalenceReport {
        torsh_pytorch_equivalent: torsh_pytorch_diff < tolerance,
        torsh_tf_equivalent: torsh_tf_diff < tolerance,
        max_differences: vec![torsh_pytorch_diff, torsh_tf_diff],
        tolerance_used: tolerance,
    }
}
```

### 3. **Fair Comparison Protocols**

#### Timing Synchronization
```rust
impl CrossFrameworkBenchmark {
    fn run_synchronized_comparison(&mut self) -> Result<ComparisonResults> {
        // Ensure all frameworks are warmed up
        self.warmup_all_frameworks()?;
        
        // Synchronize execution
        let barrier = Barrier::new(self.frameworks.len());
        
        let mut results = Vec::new();
        
        for framework in &self.frameworks {
            let framework_clone = framework.clone();
            let barrier_clone = barrier.clone();
            
            let handle = thread::spawn(move || {
                barrier_clone.wait(); // Synchronize start
                framework_clone.run_benchmark()
            });
            
            results.push(handle);
        }
        
        // Collect results
        let framework_results: Vec<_> = results.into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Result<Vec<_>>>()?;
            
        Ok(ComparisonResults {
            framework_results,
            measurement_timestamp: Instant::now(),
            environment: self.environment.clone(),
        })
    }
}
```

## Quality Assurance Methodology

### 1. **Validation Framework**

#### Benchmark Validation
```rust
trait BenchmarkValidator {
    fn validate_setup(&self, benchmark: &dyn Benchmark) -> ValidationResult;
    fn validate_measurements(&self, results: &BenchmarkResults) -> ValidationResult;
    fn validate_statistical_properties(&self, measurements: &[Duration]) -> ValidationResult;
}

struct StandardValidator {
    min_iterations: usize,
    max_coefficient_of_variation: f64,
    outlier_threshold: f64,
}

impl BenchmarkValidator for StandardValidator {
    fn validate_measurements(&self, results: &BenchmarkResults) -> ValidationResult {
        let measurements = &results.measurements;
        
        // Check sample size
        if measurements.len() < self.min_iterations {
            return ValidationResult::Failed(
                format!("Insufficient iterations: {} < {}", measurements.len(), self.min_iterations)
            );
        }
        
        // Check coefficient of variation
        let mean = measurements.iter().sum::<Duration>() / measurements.len() as u32;
        let variance = measurements.iter()
            .map(|&x| (x.as_nanos() as f64 - mean.as_nanos() as f64).powi(2))
            .sum::<f64>() / measurements.len() as f64;
        let cv = variance.sqrt() / mean.as_nanos() as f64;
        
        if cv > self.max_coefficient_of_variation {
            return ValidationResult::Warning(
                format!("High variance detected: CV = {:.3}", cv)
            );
        }
        
        ValidationResult::Passed
    }
}
```

### 2. **Reproducibility Testing**

#### Cross-Session Validation
```rust
fn validate_reproducibility() -> Result<ReproducibilityReport> {
    let benchmark = MatMulBench::new(1024, 1024, 1024);
    let num_sessions = 5;
    
    let mut session_results = Vec::new();
    
    for session in 0..num_sessions {
        // Restart process to ensure clean state
        let result = run_benchmark_in_separate_process(benchmark.clone())?;
        session_results.push(result);
    }
    
    // Analyze consistency across sessions
    let mean_times: Vec<_> = session_results.iter()
        .map(|r| r.statistical_summary.mean)
        .collect();
        
    let overall_mean = mean_times.iter().sum::<Duration>() / mean_times.len() as u32;
    let session_variance = mean_times.iter()
        .map(|&t| (t.as_nanos() as f64 - overall_mean.as_nanos() as f64).powi(2))
        .sum::<f64>() / mean_times.len() as f64;
    let session_cv = session_variance.sqrt() / overall_mean.as_nanos() as f64;
    
    ReproducibilityReport {
        session_results,
        cross_session_cv: session_cv,
        reproducible: session_cv < 0.05, // 5% threshold
    }
}
```

## Documentation and Reporting Methodology

### 1. **Metadata Collection**

#### System Information
```rust
struct BenchmarkEnvironment {
    hardware: HardwareInfo,
    software: SoftwareInfo,
    configuration: BenchmarkConfiguration,
    timestamp: SystemTime,
    git_commit: String,
}

impl BenchmarkEnvironment {
    fn collect() -> Self {
        Self {
            hardware: HardwareInfo {
                cpu: get_cpu_info(),
                memory: get_memory_info(),
                gpu: get_gpu_info(),
                storage: get_storage_info(),
            },
            software: SoftwareInfo {
                os: get_os_info(),
                rust_version: get_rust_version(),
                dependencies: get_dependency_versions(),
            },
            configuration: BenchmarkConfiguration::current(),
            timestamp: SystemTime::now(),
            git_commit: get_git_commit_hash(),
        }
    }
}
```

### 2. **Report Generation**

#### Structured Output Format
```rust
#[derive(Serialize, Deserialize)]
struct BenchmarkReport {
    metadata: BenchmarkEnvironment,
    results: Vec<BenchmarkResults>,
    statistical_analysis: StatisticalAnalysis,
    performance_insights: Vec<PerformanceInsight>,
    recommendations: Vec<OptimizationRecommendation>,
}

impl BenchmarkReport {
    fn generate_html(&self) -> Result<String> {
        let template = include_str!("templates/report.html");
        let context = ReportContext::from(self);
        render_template(template, &context)
    }
    
    fn generate_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| BenchmarkError::SerializationError(e))
    }
    
    fn generate_csv(&self) -> Result<String> {
        let mut csv = String::new();
        
        // Headers
        csv.push_str("benchmark,mean_time,std_dev,throughput,memory_usage\n");
        
        // Data rows
        for result in &self.results {
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                result.benchmark_name,
                result.statistical_summary.mean.as_micros(),
                result.statistical_summary.std_dev.as_micros(),
                result.throughput.unwrap_or(0.0),
                result.peak_memory_usage.unwrap_or(0)
            ));
        }
        
        Ok(csv)
    }
}
```

## Continuous Integration Methodology

### 1. **Automated Benchmarking**

#### CI Pipeline Integration
```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]  # Dedicated benchmark hardware
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
      
    - name: Setup Environment
      run: |
        ./scripts/setup-benchmark-env.sh
        
    - name: Run Benchmarks
      run: |
        cargo bench --package torsh-benches -- \
          --output-format json \
          --confidence-level 0.95 \
          --measurement-time 30 \
          > benchmark-results.json
          
    - name: Analyze Results
      run: |
        ./scripts/analyze-performance.py \
          --results benchmark-results.json \
          --baseline-branch main \
          --regression-threshold 0.05
          
    - name: Generate Report
      run: |
        cargo run --bin benchmark-reporter -- \
          --input benchmark-results.json \
          --output performance-report.html \
          --include-comparisons
          
    - name: Upload Artifacts
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: |
          benchmark-results.json
          performance-report.html
```

### 2. **Performance Regression Detection**

#### Automated Analysis
```python
#!/usr/bin/env python3
# scripts/analyze-performance.py

import json
import sys
import statistics
from scipy import stats

def analyze_performance(results_file, baseline_branch, threshold):
    with open(results_file) as f:
        current_results = json.load(f)
    
    baseline_results = fetch_baseline_results(baseline_branch)
    
    regressions = []
    
    for benchmark in current_results['benchmarks']:
        benchmark_name = benchmark['name']
        current_times = benchmark['measurements']
        
        if benchmark_name in baseline_results:
            baseline_times = baseline_results[benchmark_name]['measurements']
            
            # Perform statistical test
            t_stat, p_value = stats.ttest_ind(baseline_times, current_times)
            
            # Calculate effect size
            current_mean = statistics.mean(current_times)
            baseline_mean = statistics.mean(baseline_times)
            speedup = baseline_mean / current_mean
            
            # Check for significant regression
            if p_value < 0.05 and speedup < (1 - threshold):
                regressions.append({
                    'benchmark': benchmark_name,
                    'regression_percent': (1 - speedup) * 100,
                    'p_value': p_value,
                    'current_mean': current_mean,
                    'baseline_mean': baseline_mean
                })
    
    if regressions:
        print("Performance regressions detected:")
        for reg in regressions:
            print(f"  {reg['benchmark']}: {reg['regression_percent']:.1f}% slower")
        sys.exit(1)
    else:
        print("No significant performance regressions detected")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--baseline-branch', default='main')
    parser.add_argument('--regression-threshold', type=float, default=0.05)
    
    args = parser.parse_args()
    analyze_performance(args.results, args.baseline_branch, args.regression_threshold)
```

This methodology ensures that ToRSh benchmarks provide reliable, reproducible, and actionable performance insights while maintaining scientific rigor and practical relevance.