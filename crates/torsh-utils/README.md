# torsh-utils

Comprehensive utilities and tools for the ToRSh deep learning framework.

## Overview

This crate provides essential utilities for model development, debugging, optimization, and deployment in the ToRSh ecosystem. It includes benchmarking tools, profiling utilities, TensorBoard integration, mobile optimization, and development environment management.

## Features

- **Benchmarking**: Performance analysis and model benchmarking tools
- **Profiling**: Bottleneck detection and performance profiling
- **TensorBoard Integration**: Logging and visualization support
- **Mobile Optimization**: Model optimization for mobile deployment
- **Environment Collection**: Development environment diagnostics
- **C++ Extensions**: Build system for custom C++ operations
- **Model Zoo**: Model repository and management utilities

## Modules

- `benchmark`: Model benchmarking and performance analysis
- `bottleneck`: Performance bottleneck detection and profiling
- `tensorboard`: TensorBoard logging and visualization
- `mobile_optimizer`: Mobile deployment optimization
- `collect_env`: Environment and system information collection
- `cpp_extension`: C++ extension building utilities
- `model_zoo`: Model repository management

## Usage

### Benchmarking

```rust
use torsh_utils::prelude::*;
use torsh_nn::Module;

// Benchmark a model
let config = BenchmarkConfig {
    batch_size: 32,
    warmup_iterations: 10,
    benchmark_iterations: 100,
    measure_memory: true,
    device: DeviceType::Cpu,
};

let result = benchmark_model(&model, &[1, 3, 224, 224], config)?;
println!("Average forward time: {:.2}ms", result.avg_forward_time.as_millis());
```

### Profiling Bottlenecks

```rust
use torsh_utils::prelude::*;

// Profile model bottlenecks
let report = profile_bottlenecks(
    &model,
    &[32, 3, 224, 224], // input shape
    100,                // iterations
    DeviceType::Cpu,
)?;

println!("Bottleneck report:");
for (op, time) in report.operation_times {
    println!("  {}: {:.2}ms", op, time.as_millis());
}
```

### TensorBoard Integration

```rust
use torsh_utils::prelude::*;

// Create TensorBoard writer
let mut writer = SummaryWriter::new("./logs")?;

// Log scalars
writer.add_scalar("train/loss", 0.5, 100)?;
writer.add_scalar("train/accuracy", 0.95, 100)?;

// Log histograms
let weights: Tensor<f32> = model.get_parameter("linear.weight")?;
writer.add_histogram("weights/linear", &weights, 100)?;

writer.close()?;
```

### Mobile Optimization

```rust
use torsh_utils::prelude::*;

// Optimize model for mobile
let config = MobileOptimizerConfig {
    backend: MobileBackend::CoreML,
    export_format: ExportFormat::TorchScript,
    quantize: true,
    quantization_bits: 8,
    optimize_for_inference: true,
    remove_dropout: true,
    fold_batch_norm: true,
};

let optimized_model = optimize_for_mobile(&model, config)?;
```

### Environment Collection

```rust
use torsh_utils::prelude::*;

// Collect environment information
let env_info = collect_env()?;
println!("ToRSh version: {}", env_info.torsh_version);
println!("Rust version: {}", env_info.rust_version);
println!("Available devices: {:?}", env_info.available_devices);
```

## Features

### Default Features
- `std`: Standard library support
- `tensorboard`: TensorBoard integration

### Optional Features
- `profiling`: Advanced profiling capabilities
- `mobile`: Mobile optimization tools
- `cpp-extensions`: C++ extension building

## Dependencies

- `torsh-core`: Core types and device abstraction
- `torsh-tensor`: Tensor operations
- `torsh-nn`: Neural network modules
- `torsh-profiler`: Performance profiling
- `reqwest`: HTTP client for model downloads
- `prometheus`: Metrics collection
- `sysinfo`: System information gathering

## Performance

torsh-utils is optimized for:
- Minimal overhead benchmarking with high-resolution timing
- Efficient memory usage tracking and analysis
- Low-latency profiling with minimal instrumentation impact
- Streaming TensorBoard logging for large-scale training

## Compatibility

Designed to integrate seamlessly with:
- PyTorch TensorBoard logs (compatible format)
- Standard ML development workflows
- CI/CD pipelines for model validation
- Mobile deployment pipelines (iOS/Android)

## Examples

See the `examples/` directory for:
- Comprehensive benchmarking workflows
- Profiling and optimization guides
- TensorBoard integration examples
- Mobile deployment tutorials