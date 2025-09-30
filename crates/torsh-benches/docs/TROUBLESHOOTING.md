# ToRSh Benchmarking Troubleshooting Guide

## Overview

This guide helps diagnose and resolve common issues encountered when running ToRSh benchmarks. Issues are organized by category with step-by-step solutions.

## Quick Diagnostic Commands

### System Health Check
```bash
# Check ToRSh installation
cargo --version
rustc --version

# Verify CUDA setup (if using GPU)
nvidia-smi
nvcc --version

# Check available memory
free -h

# Check CPU information
lscpu

# Run basic benchmark test
cargo test --package torsh-benches basic_functionality_test
```

### Environment Validation
```rust
use torsh_benches::diagnostic::SystemDiagnostic;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let diagnostic = SystemDiagnostic::new();
    let report = diagnostic.run_full_check()?;
    
    if report.has_issues() {
        println!("System issues detected:");
        for issue in report.issues() {
            println!("  - {}: {}", issue.category, issue.description);
        }
    } else {
        println!("System ready for benchmarking");
    }
    
    Ok(())
}
```

## Compilation Issues

### Problem: "Failed to compile torsh-benches"

#### Symptoms
```
error[E0432]: unresolved import `torsh_tensor::ops`
error[E0599]: no method named `matmul` found for struct `Tensor`
error: could not compile `torsh-benches` due to previous errors
```

#### Solutions

**1. Update Dependencies**
```bash
# Update all ToRSh dependencies
cargo update

# Clean and rebuild
cargo clean
cargo build --package torsh-benches
```

**2. Check Feature Flags**
```toml
# Cargo.toml
[dependencies]
torsh-tensor = { path = "../torsh-tensor", features = ["all-ops"] }
torsh-nn = { path = "../torsh-nn", features = ["gpu"] }
```

**3. Rust Version Compatibility**
```bash
# Check minimum required Rust version
rustc --version

# Update Rust if needed
rustup update stable
```

**4. Missing System Dependencies**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install openssl-devel

# macOS
xcode-select --install
brew install pkg-config openssl
```

### Problem: "CUDA compilation errors"

#### Symptoms
```
error: CUDA driver version is insufficient for CUDA runtime version
error: cannot find -lcuda
nvcc fatal : No input files specified
```

#### Solutions

**1. CUDA Driver/Runtime Mismatch**
```bash
# Check driver version
nvidia-smi

# Check runtime version
nvcc --version

# Update CUDA driver (requires reboot)
sudo ubuntu-drivers autoinstall
sudo reboot
```

**2. Missing CUDA Development Tools**
```bash
# Install CUDA toolkit
sudo apt-get install nvidia-cuda-toolkit

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**3. Feature Configuration**
```toml
# Disable CUDA if not needed
[dependencies]
torsh-backend = { path = "../torsh-backend", default-features = false, features = ["cpu"] }

# Or ensure CUDA is properly enabled
[dependencies]
torsh-backend = { path = "../torsh-backend", features = ["cuda"] }
```

## Runtime Issues

### Problem: "Benchmark results are inconsistent"

#### Symptoms
- High variance in execution times
- Results change significantly between runs
- Performance degrades over time

#### Diagnostic
```rust
use torsh_benches::diagnostic::ConsistencyChecker;

let checker = ConsistencyChecker::new()
    .with_iterations(100)
    .with_warmup(10)
    .with_cooldown(5);

let consistency_report = checker.check_benchmark(benchmark)?;

if consistency_report.coefficient_of_variation > 0.1 {
    println!("High variance detected: CV = {:.3}", 
        consistency_report.coefficient_of_variation);
    
    for suggestion in consistency_report.suggestions {
        println!("Suggestion: {}", suggestion);
    }
}
```

#### Solutions

**1. System Load Issues**
```bash
# Check running processes
top
htop

# Kill unnecessary processes
sudo pkill chrome
sudo pkill firefox

# Set process priority
sudo nice -n -20 cargo bench
```

**2. Thermal Throttling**
```bash
# Monitor CPU temperature
sensors
watch -n 1 sensors

# Check for throttling
sudo dmesg | grep -i thermal
```

```rust
// Add thermal monitoring to benchmarks
use torsh_benches::monitoring::ThermalMonitor;

let thermal_monitor = ThermalMonitor::new()
    .with_temperature_threshold(80.0) // Celsius
    .with_throttling_detection(true);

thermal_monitor.start();

// Run benchmark
let results = run_benchmark()?;

if thermal_monitor.throttling_detected() {
    println!("Warning: Thermal throttling detected during benchmark");
    println!("Consider improving cooling or reducing workload");
}
```

**3. Memory Pressure**
```bash
# Check memory usage
free -h
cat /proc/meminfo

# Monitor memory during benchmark
watch -n 1 'free -h'
```

```rust
// Add memory monitoring
use torsh_benches::monitoring::MemoryMonitor;

let memory_monitor = MemoryMonitor::new()
    .with_leak_detection(true)
    .with_pressure_threshold(0.9); // 90% memory usage

memory_monitor.start();
let results = run_benchmark()?;

if memory_monitor.pressure_detected() {
    println!("Memory pressure detected. Consider:");
    println!("  - Reducing batch size");
    println!("  - Using gradient checkpointing");
    println!("  - Enabling memory cleanup");
}
```

**4. Background Process Interference**
```bash
# Disable CPU frequency scaling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable turbo boost (for consistent results)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Isolate CPU cores for benchmarking
sudo echo 2-7 > /sys/devices/system/cpu/isolated
```

### Problem: "GPU benchmarks fail or crash"

#### Symptoms
```
CUDA error: out of memory
CUDA error: invalid device function
Segmentation fault (core dumped)
```

#### Diagnostic
```rust
use torsh_backend::cuda::CudaDiagnostic;

let diagnostic = CudaDiagnostic::new();

// Check GPU memory
let memory_info = diagnostic.get_memory_info()?;
println!("GPU Memory: {:.1} GB / {:.1} GB", 
    memory_info.used_gb, memory_info.total_gb);

// Check CUDA capabilities
let capabilities = diagnostic.get_device_capabilities()?;
println!("Compute Capability: {}.{}", 
    capabilities.major, capabilities.minor);

// Test basic CUDA operations
let cuda_test_result = diagnostic.test_basic_operations()?;
if !cuda_test_result.success {
    println!("CUDA basic operations failed: {}", cuda_test_result.error);
}
```

#### Solutions

**1. Out of Memory Errors**
```rust
// Reduce batch size
let original_batch_size = 64;
let safe_batch_size = find_maximum_batch_size(model, available_memory)?;
println!("Reducing batch size from {} to {}", original_batch_size, safe_batch_size);

// Enable gradient checkpointing
use torsh_autograd::checkpoint;

fn memory_efficient_forward(model: &Model, input: &Tensor) -> Result<Tensor> {
    let mut x = input.clone();
    
    for (i, layer) in model.layers.iter().enumerate() {
        if i % 3 == 0 { // Checkpoint every 3 layers
            x = checkpoint::checkpoint(|| layer.forward(&x))?;
        } else {
            x = layer.forward(&x)?;
        }
    }
    
    Ok(x)
}

// Clear GPU cache between runs
fn clear_gpu_cache() -> Result<()> {
    torsh_backend::cuda::empty_cache()?;
    torsh_backend::cuda::synchronize()?;
    Ok(())
}
```

**2. CUDA Version Incompatibility**
```bash
# Check CUDA compatibility
nvidia-smi
cat /usr/local/cuda/version.txt

# Install compatible PyTorch/CUDA versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. Driver Issues**
```bash
# Reinstall NVIDIA drivers
sudo apt-get purge nvidia-*
sudo ubuntu-drivers autoinstall
sudo reboot

# Verify installation
nvidia-smi
```

### Problem: "Benchmarks run too slowly"

#### Symptoms
- Single benchmark takes hours to complete
- CPU usage is low during benchmarks
- GPU utilization is minimal

#### Diagnostic
```rust
use torsh_benches::diagnostic::PerformanceDiagnostic;

let diagnostic = PerformanceDiagnostic::new();
let profile = diagnostic.profile_benchmark(benchmark)?;

println!("CPU Utilization: {:.1}%", profile.cpu_utilization);
println!("GPU Utilization: {:.1}%", profile.gpu_utilization);
println!("Memory Bandwidth: {:.1} GB/s", profile.memory_bandwidth);

// Identify bottlenecks
for bottleneck in profile.bottlenecks {
    println!("Bottleneck: {} ({:.1}% of total time)", 
        bottleneck.operation, bottleneck.percentage);
}
```

#### Solutions

**1. Insufficient Parallelization**
```rust
// Check thread usage
use rayon::current_num_threads;
println!("Using {} threads", current_num_threads());

// Increase thread count
use rayon::ThreadPoolBuilder;
ThreadPoolBuilder::new()
    .num_threads(std::thread::available_parallelism().unwrap().get())
    .build_global()
    .unwrap();
```

**2. Debug Mode Performance**
```bash
# Use release mode for benchmarks
cargo bench --release

# Profile with release optimizations
cargo build --release --package torsh-benches
```

**3. Inefficient Operations**
```rust
// Replace slow operations with optimized versions
// Before: Element-wise operations in loop
for i in 0..tensor.len() {
    result[i] = tensor1[i] + tensor2[i];
}

// After: Vectorized operation
let result = tensor1.add(&tensor2)?;

// Use fused operations
let result = input.conv2d_bn_relu(weight, bias, bn_weight, bn_bias)?;
```

## Memory Issues

### Problem: "Memory leaks during benchmarking"

#### Symptoms
- Memory usage grows continuously
- System becomes unresponsive
- Out of memory errors after many iterations

#### Diagnostic
```rust
use torsh_benches::diagnostic::MemoryLeakDetector;

let detector = MemoryLeakDetector::new()
    .with_sampling_interval(Duration::from_secs(1))
    .with_leak_threshold(100_000_000); // 100MB growth

detector.start();

// Run benchmark
for i in 0..1000 {
    run_single_benchmark_iteration()?;
    
    if i % 100 == 0 {
        let report = detector.get_current_report();
        if report.potential_leak_detected() {
            println!("Potential memory leak at iteration {}", i);
            println!("Memory growth: {:.1} MB", report.memory_growth_mb);
            break;
        }
    }
}
```

#### Solutions

**1. Explicit Memory Management**
```rust
// Clear gradients after each iteration
optimizer.zero_grad();

// Clear autograd graph
torsh_autograd::clear_graph();

// Force garbage collection
std::mem::drop(large_tensor);

// Clear GPU memory
torsh_backend::cuda::empty_cache()?;
```

**2. Use Memory Pools**
```rust
use torsh_backend::memory::MemoryPool;

// Configure memory pool
let pool = MemoryPool::new()
    .with_initial_size(1_000_000_000) // 1GB
    .with_max_size(4_000_000_000)     // 4GB
    .enable_leak_detection(true);

MemoryPool::set_default(pool);
```

**3. Tensor Lifecycle Management**
```rust
// Use scoped tensors for temporary computations
fn compute_with_temporaries(input: &Tensor) -> Result<Tensor> {
    let result = {
        let temp1 = input.relu()?;
        let temp2 = temp1.sigmoid()?;
        temp2.tanh()? // Only this tensor survives the scope
    };
    Ok(result)
}
```

### Problem: "Out of memory errors"

#### Quick Fixes
```rust
// Reduce batch size
let safe_batch_size = calculate_safe_batch_size(model, input_shape)?;

// Use gradient accumulation instead of large batches
let effective_batch = 128;
let micro_batch = 32;
let accumulation_steps = effective_batch / micro_batch;

for step in 0..accumulation_steps {
    let micro_batch_data = get_micro_batch(step);
    let loss = model.forward(&micro_batch_data)? / accumulation_steps as f32;
    loss.backward()?;
}
optimizer.step();

// Enable mixed precision
use torsh_autograd::GradScaler;
let scaler = GradScaler::new();
model = model.to_dtype(DType::F16);
```

## Performance Issues

### Problem: "Benchmarks show poor GPU utilization"

#### Diagnostic
```bash
# Monitor GPU during benchmark
nvidia-smi dmon -s puc -d 1

# Use nvtop for detailed monitoring
nvtop
```

#### Solutions

**1. Increase Batch Size**
```rust
// Find optimal batch size for GPU utilization
fn find_optimal_batch_size(model: &Model) -> Result<usize> {
    let mut best_batch_size = 1;
    let mut best_throughput = 0.0;
    
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        match measure_throughput(model, batch_size) {
            Ok(throughput) => {
                if throughput > best_throughput {
                    best_throughput = throughput;
                    best_batch_size = batch_size;
                }
            }
            Err(_) => break, // OOM, use previous batch size
        }
    }
    
    Ok(best_batch_size)
}
```

**2. Optimize Data Transfer**
```rust
// Pin memory for faster transfers
let pinned_data = input.pin_memory()?;

// Use asynchronous transfers
let stream = CudaStream::new()?;
let gpu_data = pinned_data.to_device_async(&Device::cuda(0), &stream)?;

// Overlap computation and data transfer
stream.synchronize()?;
```

**3. Use Tensor Cores**
```rust
// Ensure FP16 precision for Tensor Core usage
let model = model.to_dtype(DType::F16);
let input = input.to_dtype(DType::F16);

// Align matrix dimensions for Tensor Cores
fn align_for_tensor_cores(size: usize) -> usize {
    (size + 7) / 8 * 8 // Round up to nearest multiple of 8
}
```

### Problem: "Cross-framework comparison shows ToRSh is slower"

#### Diagnostic
```rust
use torsh_benches::comparison::FrameworkComparison;

let comparison = FrameworkComparison::new()
    .add_framework("torsh", torsh_model)
    .add_framework("pytorch", pytorch_model)
    .enable_profiling(true);

let results = comparison.run_comparative_benchmark()?;

for (framework, result) in results.iter() {
    println!("{}: {:.2}ms", framework, result.mean_time_ms);
    
    if result.profiling_enabled {
        println!("  Bottlenecks:");
        for bottleneck in &result.bottlenecks {
            println!("    {}: {:.2}ms", bottleneck.operation, bottleneck.time_ms);
        }
    }
}
```

#### Solutions

**1. Enable Optimizations**
```rust
// Ensure release mode
#[cfg(debug_assertions)]
compile_error!("Benchmarks must be run in release mode");

// Enable all optimizations
use torsh_fx::optimize_model;
let optimized_model = optimize_model(model, &OptimizationConfig::aggressive())?;
```

**2. Fair Comparison Setup**
```rust
// Ensure equivalent configurations
fn setup_fair_comparison() -> Result<()> {
    // Set same number of threads
    std::env::set_var("OMP_NUM_THREADS", "1");
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    
    // Use same precision
    let dtype = DType::F32;
    
    // Same batch size
    let batch_size = 32;
    
    // Same random seed
    torsh_tensor::manual_seed(42);
    
    Ok(())
}
```

**3. Profile Differences**
```rust
// Compare operation-level performance
fn profile_operation_differences() -> Result<()> {
    let operations = vec!["matmul", "conv2d", "relu", "softmax"];
    
    for op in operations {
        let torsh_time = benchmark_torsh_operation(op)?;
        let pytorch_time = benchmark_pytorch_operation(op)?;
        let ratio = pytorch_time / torsh_time;
        
        println!("{}: ToRSh {:.2}ms, PyTorch {:.2}ms, Ratio: {:.2}x", 
            op, torsh_time, pytorch_time, ratio);
    }
    
    Ok(())
}
```

## Debugging Tools and Techniques

### 1. **Verbose Logging**
```rust
use torsh_benches::logging::BenchmarkLogger;

// Enable detailed logging
let logger = BenchmarkLogger::new()
    .with_level(LogLevel::Debug)
    .with_performance_metrics(true)
    .with_memory_tracking(true);

logger.enable();

// Log will show detailed execution information
let results = run_benchmark()?;
```

### 2. **Step-by-Step Debugging**
```rust
use torsh_benches::debug::StepDebugger;

let debugger = StepDebugger::new()
    .break_on_error(true)
    .break_on_slow_operation(Duration::from_millis(100))
    .enable_memory_snapshots(true);

debugger.attach(benchmark);

// Will pause execution at breakpoints
let results = debugger.run_with_debugging()?;
```

### 3. **Performance Profiling**
```bash
# Profile with perf
sudo perf record -g cargo bench --bench my_benchmark
sudo perf report

# Profile with Valgrind
valgrind --tool=callgrind cargo bench --bench my_benchmark
kcachegrind callgrind.out.*

# Profile memory usage
valgrind --tool=massif cargo bench --bench my_benchmark
ms_print massif.out.*
```

### 4. **GPU Profiling**
```bash
# NVIDIA Nsight Systems
nsys profile -o profile.nsys-rep cargo bench --bench gpu_benchmark

# NVIDIA Nsight Compute
ncu --set full -o profile cargo bench --bench gpu_benchmark

# View results
nsys-ui profile.nsys-rep
ncu-ui profile.ncu-rep
```

## Environment-Specific Issues

### 1. **Docker/Container Issues**

#### Problem: Poor performance in containers
```dockerfile
# Optimize Dockerfile for benchmarking
FROM rust:1.70

# Ensure proper CPU access
RUN echo 'performance' > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# For GPU access
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set resource limits
RUN ulimit -n 65536
```

### 2. **Cloud Instance Issues**

#### Problem: Inconsistent performance on cloud instances
```rust
// Detect cloud environment and adjust
use torsh_benches::environment::CloudDetector;

let cloud_info = CloudDetector::detect()?;

match cloud_info.provider {
    CloudProvider::AWS => {
        // AWS-specific optimizations
        if cloud_info.instance_type.contains("c5") {
            // CPU-optimized instance
            configure_for_cpu_optimization();
        }
    }
    CloudProvider::GCP => {
        // GCP-specific optimizations
        if cloud_info.has_tensor_processing_units() {
            configure_for_tpu();
        }
    }
    CloudProvider::Azure => {
        // Azure-specific optimizations
        configure_for_azure();
    }
}
```

## Getting Help

### 1. **Collecting Debug Information**
```rust
use torsh_benches::debug::DebugInfoCollector;

// Collect comprehensive debug information
let debug_info = DebugInfoCollector::new()
    .include_system_info(true)
    .include_benchmark_config(true)
    .include_performance_metrics(true)
    .include_error_logs(true)
    .collect()?;

// Save to file for sharing
debug_info.save_to_file("debug_report.json")?;
```

### 2. **Creating Minimal Reproductions**
```rust
// Create minimal failing example
use torsh_benches::prelude::*;

fn minimal_reproduction() -> Result<(), Box<dyn std::error::Error>> {
    let config = BenchConfig::default();
    let benchmark = MatMulBench::new(256, 256, 256);
    
    let runner = BenchRunner::new(config);
    let results = runner.run_single(Box::new(benchmark))?;
    
    println!("Results: {:?}", results);
    Ok(())
}
```

### 3. **Reporting Issues**
When reporting issues, include:

1. **System Information**
   - OS version and architecture
   - Rust version (`rustc --version`)
   - GPU information (`nvidia-smi`)
   - ToRSh version and commit hash

2. **Benchmark Configuration**
   - Benchmark type and parameters
   - Hardware configuration
   - Environment variables

3. **Error Messages**
   - Complete error output
   - Stack traces if available
   - Debug logs

4. **Reproduction Steps**
   - Minimal code example
   - Commands to reproduce
   - Expected vs. actual behavior

### 4. **Community Resources**
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Ask questions and share experiences  
- **Documentation**: Check official docs for updates
- **Examples**: Look at working examples in the repository

This troubleshooting guide covers the most common issues encountered when benchmarking with ToRSh. For issues not covered here, please create a detailed issue report with the information outlined above.