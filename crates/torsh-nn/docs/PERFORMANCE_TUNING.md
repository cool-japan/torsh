# Performance Tuning Guide

Comprehensive guide to optimizing ToRSh-NN models for maximum performance.

## Table of Contents

1. [Profiling and Measurement](#profiling-and-measurement)
2. [Memory Optimization](#memory-optimization)
3. [Computational Optimization](#computational-optimization)
4. [Batch Size Tuning](#batch-size-tuning)
5. [Backend Selection](#backend-selection)
6. [Model Architecture Optimization](#model-architecture-optimization)
7. [Training Optimization](#training-optimization)
8. [Inference Optimization](#inference-optimization)
9. [Hardware-Specific Optimizations](#hardware-specific-optimizations)

---

## Profiling and Measurement

### Measure First, Optimize Later

**Always profile before optimizing!**

```rust
use std::time::Instant;
use torsh_nn::profiling::Profiler;

// Basic timing
let start = Instant::now();
let output = model.forward(&input)?;
println!("Forward pass: {:?}", start.elapsed());

// Detailed profiling
#[cfg(feature = "profiling")]
{
    let mut profiler = Profiler::new();

    profiler.start("data_loading");
    let data = load_batch()?;
    profiler.stop("data_loading");

    profiler.start("forward");
    let output = model.forward(&data)?;
    profiler.stop("forward");

    profiler.start("loss_computation");
    let loss = compute_loss(&output, &target)?;
    profiler.stop("loss_computation");

    profiler.start("backward");
    loss.backward()?;
    profiler.stop("backward");

    profiler.start("optimizer_step");
    optimizer.step()?;
    profiler.stop("optimizer_step");

    // Print report
    profiler.report();
}
```

### Memory Profiling

```rust
use torsh_nn::utils::memory_usage;

let baseline = memory_usage();
println!("Baseline memory: {} MB", baseline);

let model = MyModel::new()?;
println!("After model creation: {} MB", memory_usage() - baseline);

let input = randn::<f32>(&[64, 3, 224, 224])?;
println!("After input allocation: {} MB", memory_usage() - baseline);

let output = model.forward(&input)?;
println!("After forward pass: {} MB", memory_usage() - baseline);
```

### Layer-by-Layer Analysis

```rust
use torsh_nn::summary::ModelAnalyzer;

let analyzer = ModelAnalyzer::new(&model);
let summary = analyzer.analyze(&sample_input)?;

// Print layer-wise statistics
for layer_info in summary.layers {
    println!("{}: {} params, {:.2} MB, {:.2} GFLOPs",
        layer_info.name,
        layer_info.num_params,
        layer_info.memory_mb,
        layer_info.flops / 1e9
    );
}

// Identify bottlenecks
let bottlenecks = analyzer.find_bottlenecks()?;
for bottleneck in bottlenecks {
    println!("Bottleneck: {} ({} ms)", bottleneck.layer, bottleneck.time_ms);
}
```

---

## Memory Optimization

### 1. Gradient Checkpointing

Trade computation for memory by recomputing activations during backward pass.

```rust
pub struct CheckpointedBlock {
    layers: Vec<Box<dyn Module>>,
    checkpoint_segments: usize,
}

impl CheckpointedBlock {
    pub fn new(layers: Vec<Box<dyn Module>>, checkpoint_segments: usize) -> Self {
        Self { layers, checkpoint_segments }
    }

    fn forward_segment(&self, input: &Tensor, start: usize, end: usize) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in &self.layers[start..end] {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

impl Module for CheckpointedBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.is_training() {
            // No checkpointing during inference
            return self.forward_segment(input, 0, self.layers.len());
        }

        let segment_size = self.layers.len() / self.checkpoint_segments;
        let mut x = input.clone();

        for i in 0..self.checkpoint_segments {
            let start = i * segment_size;
            let end = ((i + 1) * segment_size).min(self.layers.len());

            // Checkpoint: don't store intermediate activations
            x = self.forward_segment(&x, start, end)?;
            // Detach to save memory
            x = x.detach();
        }

        Ok(x)
    }

    // ... other methods
}
```

### 2. In-Place Operations

Use in-place operations when safe to reduce memory allocations.

```rust
// Standard operation (creates new tensor)
let output = input.add(&other)?;

// In-place operation (modifies input)
input.add_(&other)?;  // input is now modified

// Example: activation in-place
impl Module for InPlaceReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = input.clone();
        relu_inplace(&mut output)?;  // Modifies in-place
        Ok(output)
    }
}
```

### 3. Memory-Efficient Data Loading

```rust
use std::sync::Arc;

pub struct EfficientDataLoader {
    data: Arc<Vec<Tensor>>,  // Shared data
    batch_size: usize,
    shuffle: bool,
}

impl EfficientDataLoader {
    /// Use memory-mapped files for large datasets
    pub fn from_mmap(path: &str, batch_size: usize) -> Result<Self> {
        // Memory-map the data file
        #[cfg(feature = "memory_efficient")]
        {
            use scirs2_core::memory_efficient::create_mmap;
            let mmap_array = create_mmap(path)?;
            // Convert to tensors on-the-fly
            // ...
        }

        Ok(Self {
            data: Arc::new(vec![]),
            batch_size,
            shuffle: false,
        })
    }

    /// Stream data without loading everything into memory
    pub fn stream_batches(&self) -> impl Iterator<Item = Result<Tensor>> + '_ {
        (0..self.data.len())
            .step_by(self.batch_size)
            .map(move |start| {
                let end = (start + self.batch_size).min(self.data.len());
                // Load only one batch at a time
                self.load_batch_range(start, end)
            })
    }

    fn load_batch_range(&self, start: usize, end: usize) -> Result<Tensor> {
        // Load and concatenate batch
        // ...
        unimplemented!()
    }
}
```

### 4. Model Quantization

Reduce memory footprint with quantization.

```rust
use torsh_nn::quantization::{quantize_model, QuantizationConfig};

let model = MyModel::new()?;

// 8-bit quantization
let config = QuantizationConfig::int8();
let quantized = quantize_model(&model, &config, &calibration_data)?;

// Memory usage: ~4x smaller
println!("Original size: {} MB", model.memory_usage());
println!("Quantized size: {} MB", quantized.memory_usage());

// Accuracy: minimal loss (typically <1%)
let accuracy_full = evaluate(&model, &test_data)?;
let accuracy_quantized = evaluate(&quantized, &test_data)?;
println!("Accuracy drop: {:.2}%", accuracy_full - accuracy_quantized);
```

---

## Computational Optimization

### 1. Use Optimized Operations

```rust
// ❌ SLOW: Explicit loops
let mut result = vec![0.0; n];
for i in 0..n {
    result[i] = a[i] + b[i];
}

// ✅ FAST: Vectorized operations (uses SIMD)
let result = a.add(&b)?;  // Automatically optimized

// ✅ FAST: Matrix multiplication (uses BLAS)
let result = a.matmul(&b)?;  // Highly optimized
```

### 2. Fuse Operations

Combine operations to reduce memory traffic.

```rust
// ❌ SLOW: Separate operations
let x = input.add(&bias)?;
let x = relu(&x)?;
let x = dropout(&x, 0.5, true)?;

// ✅ FAST: Fused operations
let x = fused_bias_relu_dropout(&input, &bias, 0.5, true)?;

// Implement fused operations for common patterns
pub fn fused_bias_relu_dropout(
    input: &Tensor,
    bias: &Tensor,
    dropout_p: f32,
    training: bool,
) -> Result<Tensor> {
    // Single pass through data
    let output = input.add(bias)?;
    let output = relu_inplace(&mut output)?;
    if training {
        dropout_inplace(&mut output, dropout_p)?;
    }
    Ok(output)
}
```

### 3. Optimize Attention Mechanisms

```rust
// ❌ SLOW: Standard attention (O(n²) memory)
pub fn standard_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let scores = q.matmul(&k.transpose(-2, -1)?)?;
    let attn_weights = softmax(&scores, Some(-1))?;
    attn_weights.matmul(v)
}

// ✅ FAST: Flash Attention (memory-efficient)
#[cfg(feature = "flash_attention")]
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // Tiled computation with kernel fusion
    // O(n) memory complexity
    use torsh_nn::attention::flash_attention;
    flash_attention(q, k, v, None, None)
}

// ✅ FAST: Sparse Attention (for long sequences)
#[cfg(feature = "sparse_attention")]
pub fn sparse_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &SparseMask,
) -> Result<Tensor> {
    // Only compute attention for relevant positions
    use torsh_nn::attention::sparse_attention;
    sparse_attention(q, k, v, mask)
}
```

### 4. JIT Compilation

```rust
#[cfg(feature = "jit")]
use torsh_jit::{JitCompiler, JitModule};

// Compile module for better performance
let mut compiler = JitCompiler::new();
let jit_model = compiler.compile(&model)?;

// First run: compilation overhead
let output = jit_model.forward(&input)?;

// Subsequent runs: optimized code
let output = jit_model.forward(&input)?;  // Much faster!
```

---

## Batch Size Tuning

### Find Optimal Batch Size

```rust
pub fn find_optimal_batch_size(
    model: &impl Module,
    input_shape: &[usize],
    min_batch: usize,
    max_batch: usize,
) -> Result<usize> {
    let mut best_batch_size = min_batch;
    let mut best_throughput = 0.0;

    for batch_size in (min_batch..=max_batch).step_by(8) {
        // Create test input
        let mut shape = input_shape.to_vec();
        shape[0] = batch_size;
        let input = randn::<f32>(&shape)?;

        // Measure throughput
        let start = Instant::now();
        let num_iters = 10;

        for _ in 0..num_iters {
            model.forward(&input)?;
        }

        let elapsed = start.elapsed().as_secs_f32();
        let throughput = (batch_size * num_iters) as f32 / elapsed;

        println!("Batch size {}: {:.2} samples/sec", batch_size, throughput);

        if throughput > best_throughput {
            best_throughput = throughput;
            best_batch_size = batch_size;
        }

        // Stop if OOM
        if memory_usage() > 0.9 * total_memory() {
            println!("Approaching memory limit at batch size {}", batch_size);
            break;
        }
    }

    println!("Optimal batch size: {}", best_batch_size);
    Ok(best_batch_size)
}
```

### Gradient Accumulation for Large Batches

```rust
pub fn train_with_gradient_accumulation(
    model: &mut impl Module,
    optimizer: &mut impl Optimizer,
    data_loader: &DataLoader,
    effective_batch_size: usize,
    mini_batch_size: usize,
) -> Result<()> {
    let accumulation_steps = effective_batch_size / mini_batch_size;

    for (step, (data, target)) in data_loader.enumerate() {
        // Forward pass
        let output = model.forward(&data)?;
        let loss = compute_loss(&output, &target)?;

        // Scale loss by accumulation steps
        let scaled_loss = loss.div_scalar(accumulation_steps as f32)?;

        // Backward pass (gradients accumulate)
        scaled_loss.backward()?;

        // Update weights every N steps
        if (step + 1) % accumulation_steps == 0 {
            optimizer.step()?;
            optimizer.zero_grad();
        }
    }

    Ok(())
}
```

---

## Backend Selection

### CPU Backend Optimization

```rust
// Enable all CPU features
#[cfg(feature = "backend-cpu")]
use torsh_backend_cpu::CpuBackend;

let backend = CpuBackend::new()
    .with_threads(num_cpus::get())  // Use all cores
    .with_simd(true)                // Enable SIMD
    .with_blas("openblas")          // Use optimized BLAS
    .build()?;

// Set as default backend
set_default_backend(Box::new(backend))?;
```

### GPU Backend Optimization

```rust
#[cfg(feature = "backend-cuda")]
use torsh_backend_cuda::CudaBackend;

let backend = CudaBackend::new()
    .with_device(0)                    // Select GPU
    .with_cudnn(true)                  // Enable cuDNN
    .with_tensor_cores(true)           // Enable tensor cores
    .with_memory_pool(true)            // Use memory pool
    .with_stream_count(4)              // Multiple streams
    .build()?;

set_default_backend(Box::new(backend))?;
```

### Mixed Precision Training

```rust
#[cfg(feature = "mixed_precision")]
use torsh_nn::mixed_precision::{MixedPrecisionTrainer, GradScaler};

let mut trainer = MixedPrecisionTrainer::new(model, optimizer);
let mut scaler = GradScaler::new(init_scale=2.0^16);

for (data, target) in train_loader {
    // Forward in FP16
    let output = trainer.forward_fp16(&data)?;
    let loss = compute_loss(&output, &target)?;

    // Backward with gradient scaling
    scaler.scale(&loss).backward()?;
    scaler.step(&mut trainer.optimizer)?;
    scaler.update();

    trainer.zero_grad();
}
```

---

## Model Architecture Optimization

### 1. Reduce Parameter Count

```rust
// ❌ SLOW: Large model
pub struct HeavyModel {
    fc1: Linear(1024, 2048, true),
    fc2: Linear(2048, 2048, true),
    fc3: Linear(2048, 1024, true),
}
// Total: 1024*2048 + 2048*2048 + 2048*1024 = ~8.4M parameters

// ✅ FAST: Bottleneck architecture
pub struct EfficientModel {
    fc1: Linear(1024, 256, true),   // Bottleneck
    fc2: Linear(256, 256, true),    // Small hidden layer
    fc3: Linear(256, 1024, true),   // Expand
}
// Total: 1024*256 + 256*256 + 256*1024 = ~0.5M parameters
```

### 2. Use Depthwise Separable Convolutions

```rust
// ❌ SLOW: Standard convolution
let conv = Conv2d::new(128, 256, 3, 1, 1, 1, 1, true)?;
// Parameters: 128 * 256 * 3 * 3 = 294,912

// ✅ FAST: Depthwise separable
let depthwise = Conv2d::new(128, 128, 3, 1, 1, 1, 128, true)?;  // groups=128
let pointwise = Conv2d::new(128, 256, 1, 1, 0, 1, 1, true)?;
// Parameters: 128 * 3 * 3 + 128 * 256 = 33,920 (9x fewer!)
```

### 3. Structured Pruning

```rust
use torsh_nn::pruning::{StructuredPruner, PruningScheduler};

let mut pruner = StructuredPruner::new(0.5);  // Remove 50% of filters

// Identify unimportant filters
let importance = pruner.compute_importance(&model, &calibration_data)?;

// Remove least important filters
let pruned_model = pruner.prune(&model, &importance)?;

// Fine-tune pruned model
train(&mut pruned_model, &train_data)?;

// Result: 2x faster, minimal accuracy loss
```

### 4. Knowledge Distillation

```rust
use torsh_nn::distillation::Distiller;

// Large teacher model
let teacher = LargeModel::new()?;
teacher.load("teacher_weights.pth")?;

// Small student model
let mut student = SmallModel::new()?;

// Distill knowledge
let distiller = Distiller::new(teacher, temperature=4.0);

for (data, target) in train_loader {
    // Get soft targets from teacher
    let teacher_logits = distiller.teacher_forward(&data)?;

    // Train student to match teacher
    let student_logits = student.forward(&data)?;
    let distillation_loss = distiller.distillation_loss(
        &student_logits,
        &teacher_logits,
        &target
    )?;

    distillation_loss.backward()?;
    optimizer.step()?;
}

// Result: Student model is small but performs nearly as well
```

---

## Training Optimization

### 1. Learning Rate Scheduling

```rust
use torsh_optim::lr_scheduler::{CosineAnnealingLR, WarmupScheduler};

// Warm-up + cosine annealing
let mut scheduler = WarmupScheduler::new(
    CosineAnnealingLR::new(optimizer, T_max=epochs),
    warmup_epochs=5,
    warmup_factor=0.1,
);

for epoch in 0..epochs {
    train_one_epoch(&mut model, &mut optimizer, &train_loader)?;
    scheduler.step();  // Update learning rate

    let lr = scheduler.get_lr();
    println!("Epoch {}, LR: {:.6}", epoch, lr);
}
```

### 2. Automatic Mixed Precision

```rust
#[cfg(feature = "amp")]
{
    use torsh_nn::amp::{autocast, GradScaler};

    let mut scaler = GradScaler::new();

    for (data, target) in train_loader {
        optimizer.zero_grad();

        // Automatic mixed precision
        let (output, loss) = autocast(|| {
            let output = model.forward(&data)?;
            let loss = loss_fn(&output, &target)?;
            Ok((output, loss))
        })?;

        // Scale gradients to prevent underflow
        scaler.scale(&loss).backward()?;
        scaler.step(&mut optimizer)?;
        scaler.update();
    }
}
```

### 3. Data Prefetching

```rust
use std::sync::mpsc::channel;
use std::thread;

pub struct PrefetchDataLoader {
    loader: DataLoader,
    prefetch_factor: usize,
}

impl PrefetchDataLoader {
    pub fn iter(&self) -> PrefetchIterator {
        let (tx, rx) = channel();
        let loader = self.loader.clone();
        let prefetch_factor = self.prefetch_factor;

        // Spawn background thread for data loading
        thread::spawn(move || {
            for batch in loader.iter() {
                tx.send(batch).unwrap();
            }
        });

        PrefetchIterator {
            receiver: rx,
            buffer: Vec::with_capacity(prefetch_factor),
        }
    }
}

// Usage
let prefetch_loader = PrefetchDataLoader::new(data_loader, prefetch_factor=2);

for (data, target) in prefetch_loader.iter() {
    // Data is already loaded in background
    let output = model.forward(&data)?;
    // ... training step
}
```

---

## Inference Optimization

### 1. Model Fusion

```rust
use torsh_nn::optimization::fuse_modules;

// Before: separate modules
let model = Sequential::new()
    .add(conv)
    .add(batch_norm)
    .add(relu);

// After: fused modules
let fused_model = fuse_modules(&model)?;
// conv + batch_norm fused into single operation
// Up to 2x faster inference
```

### 2. Static Graph Optimization

```rust
#[cfg(feature = "static_graph")]
{
    use torsh_nn::optimization::optimize_graph;

    // Trace model with example input
    let example = randn::<f32>(&[1, 3, 224, 224])?;
    let traced = model.trace(&example)?;

    // Optimize traced graph
    let optimized = optimize_graph(&traced, &[
        "constant_folding",
        "dead_code_elimination",
        "operator_fusion",
    ])?;

    // Use optimized model for inference
    let output = optimized.forward(&input)?;  // Faster!
}
```

### 3. Batch Inference

```rust
// ❌ SLOW: Process one at a time
for sample in samples {
    let output = model.forward(&sample)?;
    results.push(output);
}

// ✅ FAST: Process in batches
let batched = stack_samples(&samples, batch_size=32)?;
for batch in batched {
    let outputs = model.forward(&batch)?;  // Much faster!
    results.extend(outputs.split()?);
}
```

### 4. Model Caching

```rust
use std::collections::LRUCache;

pub struct CachedModel {
    model: Box<dyn Module>,
    cache: RefCell<LRUCache<Vec<u8>, Tensor>>,
}

impl CachedModel {
    pub fn new(model: Box<dyn Module>, cache_size: usize) -> Self {
        Self {
            model,
            cache: RefCell::new(LRUCache::new(cache_size)),
        }
    }

    pub fn forward_cached(&self, input: &Tensor) -> Result<Tensor> {
        // Hash input
        let key = input.to_bytes()?;

        // Check cache
        if let Some(cached) = self.cache.borrow_mut().get(&key) {
            return Ok(cached.clone());
        }

        // Compute and cache
        let output = self.model.forward(input)?;
        self.cache.borrow_mut().put(key, output.clone());

        Ok(output)
    }
}
```

---

## Hardware-Specific Optimizations

### CPU-Specific

```rust
// Use all available cores
std::env::set_var("RAYON_NUM_THREADS", num_cpus::get().to_string());

// Enable MKL or OpenBLAS
#[cfg(feature = "mkl")]
use mkl::set_num_threads;
set_num_threads(num_cpus::get());

// Use SIMD instructions
#[cfg(target_feature = "avx2")]
{
    // AVX2-optimized path
}
```

### GPU-Specific

```rust
#[cfg(feature = "cuda")]
{
    // Pin memory for faster GPU transfers
    let pinned_data = data.pin_memory()?;

    // Use multiple CUDA streams
    let stream1 = CudaStream::new()?;
    let stream2 = CudaStream::new()?;

    // Overlap computation and data transfer
    stream1.async_copy(&data_batch1)?;
    stream2.forward(&model, &data_batch0)?;

    // Enable cuDNN autotuner
    cudnn::set_benchmark_mode(true)?;

    // Use tensor cores for matrix multiplication
    let output = cublas::gemm_tensor_core(&a, &b)?;
}
```

### Multi-GPU

```rust
#[cfg(feature = "distributed")]
{
    use torsh_nn::distributed::DataParallel;

    // Wrap model for multi-GPU training
    let model = DataParallel::new(model, device_ids=vec![0, 1, 2, 3])?;

    // Automatically splits batch across GPUs
    let output = model.forward(&input)?;
}
```

---

## Performance Checklist

Before deployment, verify:

- [ ] Profiled code and identified bottlenecks
- [ ] Optimal batch size selected
- [ ] Using appropriate backend (CPU/GPU)
- [ ] Enabled relevant hardware features (SIMD, cuDNN, etc.)
- [ ] Applied model optimizations (pruning, quantization, distillation)
- [ ] Fused operations where possible
- [ ] Eliminated unnecessary data copies
- [ ] Using efficient data loading
- [ ] Applied mixed precision if on GPU
- [ ] Tested throughput and latency

---

## Quick Wins

1. **Use larger batch sizes**: Often 2-4x speedup
2. **Enable SIMD**: Automatic with SciRS2, verify with profiling
3. **Quantize model**: 4x memory reduction, ~2x speedup
4. **Use mixed precision**: 2-3x speedup on modern GPUs
5. **Batch normalization fusion**: Fold into conv layers for inference
6. **Profile and optimize hotspots**: Focus on top 3 bottlenecks

---

## Benchmarking Template

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_model(c: &mut Criterion) {
    let model = MyModel::new().unwrap();
    let input = randn::<f32>(&[32, 3, 224, 224]).unwrap();

    c.bench_function("model_forward", |b| {
        b.iter(|| {
            let output = model.forward(black_box(&input)).unwrap();
            black_box(output);
        });
    });
}

criterion_group!(benches, benchmark_model);
criterion_main!(benches);
```

---

## Additional Resources

- **Profiling Guide**: See `flamegraph` and `perf` tools
- **CUDA Best Practices**: NVIDIA CUDA C Best Practices Guide
- **CPU Optimization**: Intel Optimization Manual
- **Model Optimization**: See `BEST_PRACTICES.md`

For questions or contributions, visit: https://github.com/cool-japan/torsh
