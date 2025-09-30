# torsh-profiler

Performance profiling and analysis tools for ToRSh applications.

## Overview

This crate provides comprehensive profiling capabilities for deep learning workloads:

- **Performance Profiling**: CPU/GPU time, memory usage, operation counts
- **Memory Profiling**: Allocation tracking, peak usage, memory leaks
- **Operation Analysis**: Kernel timing, FLOPS counting, bottleneck detection
- **Visualization**: Chrome tracing, TensorBoard integration, custom views
- **Integration**: Works with CUDA profiler, Intel VTune, Apple Instruments

## Usage

### Basic Profiling

```rust
use torsh_profiler::prelude::*;

// Profile a model
let profiler = Profiler::new()
    .record_shapes(true)
    .with_stack(true);

with_profiler(&profiler, || {
    for _ in 0..100 {
        let output = model.forward(&input)?;
        loss = criterion(&output, &target)?;
        loss.backward()?;
        optimizer.step()?;
    }
})?;

// Get results
let report = profiler.report();
println!("{}", report);
```

### Detailed Operation Profiling

```rust
// Profile with categories
let profiler = Profiler::new()
    .activities(&[ProfilerActivity::CPU, ProfilerActivity::CUDA])
    .record_shapes(true)
    .profile_memory(true)
    .with_stack(true);

// Profile specific operations
profiler.start();

profiler.step("data_loading");
let batch = dataloader.next()?;

profiler.step("forward");
let output = model.forward(&batch)?;

profiler.step("loss");
let loss = criterion(&output, &target)?;

profiler.step("backward");
loss.backward()?;

profiler.step("optimizer");
optimizer.step()?;

profiler.stop();

// Export trace
profiler.export_chrome_trace("trace.json")?;
```

### Memory Profiling

```rust
use torsh_profiler::memory::*;

// Track memory allocations
let memory_profiler = MemoryProfiler::new()
    .track_allocations(true)
    .include_stacktraces(true);

memory_profiler.start();

// Your code here
let tensors = (0..1000).map(|_| {
    randn(&[1024, 1024])
}).collect::<Vec<_>>();

memory_profiler.stop();

// Analyze memory usage
let snapshot = memory_profiler.snapshot()?;
println!("Peak memory: {} MB", snapshot.peak_memory_mb());
println!("Current memory: {} MB", snapshot.current_memory_mb());

// Find memory leaks
let leaks = memory_profiler.find_leaks()?;
for leak in leaks {
    println!("Potential leak: {} bytes at {}", leak.size, leak.stack_trace);
}
```

### FLOPS Counting

```rust
use torsh_profiler::flops::*;

// Count FLOPS for a model
let flop_counter = FlopCounter::new(&model);
let input_shape = vec![1, 3, 224, 224];

let total_flops = flop_counter.count(&input_shape)?;
println!("Total FLOPs: {}", format_flops(total_flops));

// Detailed breakdown
let breakdown = flop_counter.breakdown(&input_shape)?;
for (module_name, flops) in breakdown {
    println!("{}: {}", module_name, format_flops(flops));
}
```

### Custom Profiling Regions

```rust
use torsh_profiler::profile;

// Profile specific code regions
profile!("data_preprocessing", {
    let normalized = normalize(&data)?;
    let augmented = augment(&normalized)?;
    augmented
});

// Or with explicit profiler
let profiler = Profiler::current();
let _guard = profiler.record("critical_section");
// Critical code here
// _guard automatically stops profiling when dropped
```

### TensorBoard Integration

```rust
use torsh_profiler::tensorboard::*;

// Export to TensorBoard format
let tb_profiler = TensorBoardProfiler::new("./runs/profile");

tb_profiler.add_scalar("loss", loss.item(), step)?;
tb_profiler.add_histogram("weights", &model.weight, step)?;
tb_profiler.add_graph(&model, &example_input)?;

// Profile and export
with_profiler(&tb_profiler, || {
    // Training loop
})?;
```

### Advanced Analysis

```rust
use torsh_profiler::analysis::*;

// Analyze bottlenecks
let analyzer = ProfileAnalyzer::new(&profiler.events());

let bottlenecks = analyzer.find_bottlenecks()?;
for bottleneck in bottlenecks.iter().take(10) {
    println!("{}: {:.2}% of total time", bottleneck.name, bottleneck.percentage);
}

// Find inefficient operations
let inefficiencies = analyzer.find_inefficiencies()?;
for issue in inefficiencies {
    println!("Inefficiency: {}", issue.description);
    println!("Suggestion: {}", issue.suggestion);
}

// Memory access patterns
let memory_patterns = analyzer.analyze_memory_access()?;
println!("Cache efficiency: {:.2}%", memory_patterns.cache_efficiency * 100.0);
```

### Multi-GPU Profiling

```rust
// Profile distributed training
let profiler = DistributedProfiler::new()
    .rank(rank)
    .world_size(world_size)
    .sync_enabled(true);

with_profiler(&profiler, || {
    // Distributed training
})?;

// Aggregate results from all ranks
if rank == 0 {
    let aggregated = profiler.aggregate_results()?;
    println!("Total communication time: {:?}", aggregated.comm_time);
    println!("Load imbalance: {:.2}%", aggregated.load_imbalance * 100.0);
}
```

### Integration with External Profilers

```rust
// NVIDIA Nsight Systems
#[cfg(feature = "cuda")]
{
    use torsh_profiler::cuda::*;
    
    nvtx::range_push("model_forward");
    let output = model.forward(&input)?;
    nvtx::range_pop();
}

// Intel VTune
#[cfg(feature = "vtune")]
{
    use torsh_profiler::vtune::*;
    
    let domain = vtune::Domain::new("torsh_app");
    let task = domain.begin_task("inference");
    let output = model.forward(&input)?;
    task.end();
}
```

### Profiling Configuration

```rust
// Configure via environment variables
// TORSH_PROFILER_ENABLED=1
// TORSH_PROFILER_OUTPUT=trace.json
// TORSH_PROFILER_ACTIVITIES=cpu,cuda

// Or programmatically
ProfilerConfig::default()
    .enabled(true)
    .output_path("profile.json")
    .activities(vec![Activity::CPU, Activity::CUDA])
    .record_shapes(true)
    .profile_memory(true)
    .with_stack(true)
    .with_flops(true)
    .with_modules(true)
    .export_format(ExportFormat::Chrome)
    .apply()?;
```

## Visualization

The profiler can export data in various formats:

- **Chrome Tracing**: View in chrome://tracing
- **TensorBoard**: Integrated with TensorBoard profiler plugin
- **Perfetto**: Modern trace viewer
- **Custom JSON**: For custom analysis tools

## Performance Tips

1. Profile representative workloads
2. Warm up before profiling (exclude first iterations)
3. Profile both training and inference
4. Look for memory allocation patterns
5. Check for unnecessary synchronizations

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.