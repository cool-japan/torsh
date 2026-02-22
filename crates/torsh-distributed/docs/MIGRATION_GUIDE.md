# Migration Guide: From PyTorch Distributed to ToRSh

This comprehensive guide helps you migrate your PyTorch distributed training code to ToRSh. We'll cover common patterns, provide side-by-side comparisons, and offer migration strategies for different use cases.

## Table of Contents

1. [Migration Overview](#migration-overview)
2. [Basic DDP Migration](#basic-ddp-migration)
3. [Advanced Features Migration](#advanced-features-migration)
4. [Framework Integration Migration](#framework-integration-migration)
5. [Configuration Migration](#configuration-migration)
6. [Performance Migration](#performance-migration)
7. [Troubleshooting Migration Issues](#troubleshooting-migration-issues)
8. [Migration Checklist](#migration-checklist)

## Migration Overview

### Key Differences

| Aspect | PyTorch Distributed | ToRSh Distributed |
|--------|-------------------|------------------|
| **Language** | Python | Rust |
| **Memory Safety** | Runtime checks | Compile-time guarantees |
| **Performance** | Good | Excellent (zero-cost abstractions) |
| **Backend Support** | NCCL, Gloo, MPI | NCCL, Gloo, MPI, Mock |
| **Framework Integration** | Native PyTorch | DeepSpeed, Horovod, FairScale, Ray, Dask |
| **Error Handling** | Exceptions | Result types with detailed context |
| **Configuration** | Python dicts/args | Type-safe config structs |

### Migration Strategy

1. **Gradual Migration**: Start with simple DDP, then add advanced features
2. **Side-by-side Testing**: Run both implementations to verify correctness
3. **Performance Validation**: Benchmark before and after migration
4. **Configuration Mapping**: Use our integration layers for easy config migration

## Basic DDP Migration

### Simple DDP Setup

**PyTorch (Before):**
```python
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    setup(rank, world_size)
    
    # Create model
    model = MyModel()
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = ddp_model(batch.to(rank))
            loss = criterion(outputs, targets.to(rank))
            loss.backward()
            optimizer.step()
    
    cleanup()
```

**ToRSh (After):**
```rust
use torsh_distributed::*;
use torsh_nn::Module;

fn main() -> TorshResult<()> {
    // Initialize process group
    let rank = std::env::var("LOCAL_RANK")?.parse()?;
    let world_size = std::env::var("WORLD_SIZE")?.parse()?;
    
    let pg = init_process_group(
        BackendType::Nccl,
        Rank(rank),
        WorldSize(world_size),
        "localhost",
        12355,
    )?;
    
    // Create model
    let model = MyModel::new()?;
    
    // Configure DDP
    let bucket_config = BucketConfig {
        bucket_size_mb: 25.0,
        gradient_accumulation_steps: 1,
        overlap_communication: true,
        gradient_compression: false,
        find_unused_parameters: false,
        static_graph: false,
        backward_passes_per_step: 1,
        gradient_as_bucket_view: false,
    };
    
    let ddp_model = DistributedDataParallel::new(model, bucket_config)?;
    
    // Training loop
    for epoch in 0..num_epochs {
        for batch in dataloader {
            let outputs = ddp_model.forward(&batch.input)?;
            let loss = compute_loss(&outputs, &batch.target)?;
            loss.backward()?;
            ddp_model.step()?;
        }
    }
    
    Ok(())
}
```

### Launch Scripts Migration

**PyTorch (torchrun):**
```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --master_addr=node1 \
    --master_port=29500 \
    train.py --batch_size=32
```

**ToRSh (equivalent):**
```bash
# Set environment variables
export MASTER_ADDR=node1
export MASTER_PORT=29500
export WORLD_SIZE=8
export LOCAL_WORLD_SIZE=4

# Launch on each node
CUDA_VISIBLE_DEVICES=0,1,2,3 cargo run --release --bin train -- --batch_size=32
```

## Advanced Features Migration

### FSDP Migration

**PyTorch FSDP:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp import ShardingStrategy

# Configure FSDP
fsdp_config = {
    "sharding_strategy": ShardingStrategy.FULL_SHARD,
    "cpu_offload": None,
    "auto_wrap_policy": None,
    "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
    "mixed_precision": MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    ),
    "ignored_modules": [],
    "param_init_fn": None,
    "device_id": None,
    "sync_module_states": True,
}

model = FSDP(model, **fsdp_config)
```

**ToRSh FSDP:**
```rust
use torsh_distributed::*;

// Configure FSDP
let fsdp_config = FsdpConfig {
    sharding_strategy: ShardingStrategy::FullShard,
    cpu_offload: false,
    auto_wrap_policy: None,
    backward_prefetch: Some(BackwardPrefetch::BackwardPre),
    mixed_precision: Some(MixedPrecisionConfig {
        param_dtype: "float16".to_string(),
        reduce_dtype: "float16".to_string(),
        buffer_dtype: "float16".to_string(),
        keep_low_precision_grads: false,
        cast_forward_inputs: true,
        cast_root_forward_inputs: true,
    }),
    ignored_modules: Vec::new(),
    param_init_fn: None,
    device_id: None,
    sync_module_states: true,
    forward_prefetch: false,
    limit_all_gathers: true,
    use_orig_params: false,
};

let fsdp_model = FullyShardedDataParallel::new(model, fsdp_config)?;
```

### Gradient Compression Migration

**PyTorch (using external library):**
```python
# Using Horovod or similar
import horovod.torch as hvd

hvd.init()

# Compression
compression = hvd.Compression.fp16  # or hvd.Compression.none

optimizer = hvd.DistributedOptimizer(
    optimizer, 
    named_parameters=model.named_parameters(),
    compression=compression
)
```

**ToRSh (built-in):**
```rust
use torsh_distributed::*;

// Configure compression
let compression_config = CompressionConfig {
    method: CompressionMethod::TopK { k: 0.01 },  // Top 1%
    error_feedback: true,
    compression_period: 1,
    memory_optimization: true,
};

let compressor = GradientCompressor::new(compression_config)?;

// Apply during training
let compressed_gradients = compressor.compress(&gradients)?;
all_reduce(&compressed_gradients, ReduceOp::Sum)?;
let decompressed_gradients = compressor.decompress(&compressed_gradients)?;
```

### Pipeline Parallelism Migration

**PyTorch (using PiPPy or similar):**
```python
from torch.distributed.pipeline.sync import Pipe

# Manual pipeline creation
class MyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model = MyPipeline()
pipe_model = Pipe(model, balance=[2, 1], devices=[0, 1], chunks=8)
```

**ToRSh (built-in):**
```rust
use torsh_distributed::*;

// Configure pipeline
let pipeline_config = PipelineConfig {
    stages: 4,
    micro_batch_size: 8,
    schedule: ScheduleType::OneF1B,
    checkpoint_activation: true,
    enable_backward_prefetch: true,
    enable_forward_prefetch: true,
    accumulate_grads_in_fp32: false,
    overlap_p2p_communication: true,
    use_interleaved_batches: false,
};

// Create pipeline stages
let stages = create_pipeline_stages(&model, pipeline_config.stages)?;
let pipeline_model = PipelineParallel::new(stages, pipeline_config)?;
```

## Framework Integration Migration

### DeepSpeed Migration

**PyTorch + DeepSpeed:**
```python
import deepspeed

# DeepSpeed config
ds_config = {
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 500
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    },
    "fp16": {
        "enabled": True,
        "initial_scale_power": 16
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    config_params=ds_config
)
```

**ToRSh + DeepSpeed Integration:**
```rust
use torsh_distributed::*;

// Load DeepSpeed config from JSON
let deepspeed_config = DeepSpeedIntegration::from_file("deepspeed_config.json")?;

// Initialize integration
let mut deepspeed = DeepSpeedIntegration::new(deepspeed_config);
deepspeed.initialize(rank, world_size)?;

// Convert to ToRSh FSDP
let fsdp_config = deepspeed.to_fsdp_config()?;
let compression_config = deepspeed.to_compression_config()?;

// Use with ToRSh
let fsdp_model = FullyShardedDataParallel::new(model, fsdp_config)?;
if let Some(compression) = compression_config {
    let compressor = GradientCompressor::new(compression)?;
    // Use compressor in training loop
}
```

### Horovod Migration

**Horovod:**
```python
import horovod.torch as hvd

hvd.init()

# Scale learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01 * hvd.size())

# Wrap optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

# Broadcast parameters
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
```

**ToRSh + Horovod Integration:**
```rust
use torsh_distributed::*;

// Create Horovod-compatible configuration
let config = HorovodIntegration::config_with_topk_compression(0.01);
let mut horovod = HorovodIntegration::new(config);
horovod.initialize(rank, world_size, local_rank, local_size)?;

// Convert to ToRSh DDP
let ddp_config = horovod.to_ddp_config()?;
let compression_config = horovod.to_compression_config()?;

let ddp_model = DistributedDataParallel::new(model, ddp_config)?;

// Scale learning rate (equivalent to hvd.size())
let scaled_lr = base_lr * horovod.size() as f32;

// Training loop with Horovod-style operations
for epoch in 0..num_epochs {
    for batch in dataloader {
        let output = ddp_model.forward(&batch.input)?;
        let loss = compute_loss(&output, &batch.target)?;
        loss.backward()?;
        
        // Simulate Horovod allreduce
        horovod.allreduce("gradients", gradient_size)?;
        
        ddp_model.step()?;
    }
}
```

## Configuration Migration

### Environment Variables Mapping

| PyTorch Environment | ToRSh Environment | Notes |
|--------------------|------------------|-------|
| `RANK` | `RANK` | Global rank |
| `LOCAL_RANK` | `LOCAL_RANK` | Local rank within node |
| `WORLD_SIZE` | `WORLD_SIZE` | Total number of processes |
| `MASTER_ADDR` | `MASTER_ADDR` | Master node address |
| `MASTER_PORT` | `MASTER_PORT` | Master node port |
| `NCCL_DEBUG` | `NCCL_DEBUG` | NCCL debugging |
| `CUDA_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` | GPU visibility |

### Configuration File Migration

**PyTorch Config (YAML/JSON):**
```yaml
# pytorch_config.yaml
distributed:
  backend: nccl
  init_method: env://
  world_size: 8
  rank: 0

model:
  batch_size: 32
  learning_rate: 0.001

training:
  epochs: 100
  gradient_clipping: 1.0
```

**ToRSh Config (TOML/JSON):**
```toml
# torsh_config.toml
[distributed]
backend = "Nccl"
master_addr = "localhost"
master_port = 29500
world_size = 8
rank = 0

[ddp]
bucket_size_mb = 25.0
overlap_communication = true
gradient_compression = false

[training]
epochs = 100
batch_size = 32
learning_rate = 0.001
gradient_clipping = 1.0
```

**Loading in ToRSh:**
```rust
use serde::{Deserialize, Serialize};
use torsh_distributed::*;

#[derive(Debug, Deserialize)]
struct TrainingConfig {
    distributed: DistributedConfig,
    ddp: DdpConfig,
    training: TrainingParams,
}

#[derive(Debug, Deserialize)]
struct DistributedConfig {
    backend: String,
    master_addr: String,
    master_port: u16,
    world_size: u32,
    rank: u32,
}

#[derive(Debug, Deserialize)]
struct DdpConfig {
    bucket_size_mb: f32,
    overlap_communication: bool,
    gradient_compression: bool,
}

#[derive(Debug, Deserialize)]
struct TrainingParams {
    epochs: u32,
    batch_size: u32,
    learning_rate: f32,
    gradient_clipping: f32,
}

fn load_config() -> TorshResult<TrainingConfig> {
    let config_str = std::fs::read_to_string("torsh_config.toml")?;
    let config: TrainingConfig = toml::from_str(&config_str)?;
    Ok(config)
}

fn main() -> TorshResult<()> {
    let config = load_config()?;
    
    // Initialize process group
    let backend = match config.distributed.backend.as_str() {
        "Nccl" => BackendType::Nccl,
        "Gloo" => BackendType::Gloo,
        "Mpi" => BackendType::Mpi,
        _ => BackendType::Mock,
    };
    
    let pg = init_process_group(
        backend,
        Rank(config.distributed.rank),
        WorldSize(config.distributed.world_size),
        &config.distributed.master_addr,
        config.distributed.master_port,
    )?;
    
    // Configure DDP
    let bucket_config = BucketConfig {
        bucket_size_mb: config.ddp.bucket_size_mb,
        overlap_communication: config.ddp.overlap_communication,
        gradient_compression: config.ddp.gradient_compression,
        ..Default::default()
    };
    
    // Training setup...
    Ok(())
}
```

## Performance Migration

### Optimization Settings Migration

**PyTorch Optimizations:**
```python
# PyTorch optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# NCCL optimizations
os.environ['NCCL_ALGO'] = 'Ring'
os.environ['NCCL_MIN_NCHANNELS'] = '4'
os.environ['NCCL_MAX_NCHANNELS'] = '16'

# Memory optimizations
torch.cuda.empty_cache()
```

**ToRSh Optimizations:**
```rust
use torsh_distributed::*;

fn configure_optimizations() -> TorshResult<()> {
    // NCCL optimizations
    std::env::set_var("NCCL_ALGO", "Ring");
    std::env::set_var("NCCL_MIN_NCHANNELS", "4");
    std::env::set_var("NCCL_MAX_NCHANNELS", "16");
    
    // Configure ToRSh-specific optimizations
    #[cfg(feature = "nccl")]
    {
        use torsh_distributed::nccl_optimization::*;
        
        let scheduler = NcclScheduler::new()?;
        scheduler.optimize_communication_patterns()?;
        scheduler.enable_kernel_fusion(true)?;
        
        let memory_pool = GpuMemoryPool::new(1024 * 1024 * 1024)?; // 1GB pool
        memory_pool.enable_defragmentation(true)?;
    }
    
    Ok(())
}

fn main() -> TorshResult<()> {
    configure_optimizations()?;
    
    // Your training code...
    Ok(())
}
```

### Memory Optimization Migration

**PyTorch:**
```python
# Gradient checkpointing
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return self.output_layer(x)

# Mixed precision
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**ToRSh:**
```rust
use torsh_distributed::*;

// Enable gradient checkpointing in FSDP
let fsdp_config = FsdpConfig {
    // Checkpointing is handled automatically by FSDP
    backward_prefetch: Some(BackwardPrefetch::BackwardPre),
    forward_prefetch: true,
    limit_all_gathers: true,
    // ... other config
};

// Mixed precision with automatic scaling
let mixed_precision_config = MixedPrecisionConfig {
    param_dtype: "float16".to_string(),
    reduce_dtype: "float16".to_string(),
    buffer_dtype: "float32".to_string(),
    keep_low_precision_grads: false,
    cast_forward_inputs: true,
    cast_root_forward_inputs: true,
};

let fsdp_config = FsdpConfig {
    mixed_precision: Some(mixed_precision_config),
    // ... other config
};

let fsdp_model = FullyShardedDataParallel::new(model, fsdp_config)?;

// Training loop with automatic mixed precision
for batch in dataloader {
    let outputs = fsdp_model.forward(&batch.input)?; // Automatic casting
    let loss = compute_loss(&outputs, &batch.target)?;
    loss.backward()?; // Automatic scaling
    fsdp_model.step()?; // Automatic unscaling and step
}
```

## Troubleshooting Migration Issues

### Common Migration Problems and Solutions

#### 1. Performance Regression

**Problem**: ToRSh training is slower than PyTorch equivalent.

**Diagnosis**:
```rust
use torsh_distributed::*;

fn diagnose_performance() -> TorshResult<()> {
    // Enable comprehensive profiling
    let profiling_config = ProfilingConfig {
        enable_memory_profiling: true,
        enable_communication_profiling: true,
        enable_computation_profiling: true,
        sampling_interval_ms: 10,
        output_file: "performance_profile.json".to_string(),
        ..Default::default()
    };
    
    init_global_profiler(profiling_config)?;
    
    // Run training with profiling
    // Analysis will be in the output file
    
    Ok(())
}
```

**Solutions**:
1. Enable communication overlap
2. Optimize bucket sizes
3. Use gradient compression
4. Enable NCCL optimizations

#### 2. Memory Usage Differences

**Problem**: Higher memory usage compared to PyTorch.

**Solutions**:
```rust
// Enable aggressive memory optimization
let memory_config = MemoryConfig {
    enable_checkpointing: true,
    checkpoint_ratio: 0.5,
    offload_activations: true,
    gradient_accumulation: true,
};

// Use CPU offloading
let zero3_config = Zero3CpuOffloadConfig {
    offload_optimizer_state: true,
    offload_parameters: true,
    compression_method: Some(CpuCompressionMethod::FP16),
    memory_strategy: AutoMemoryStrategy::Aggressive,
    ..Default::default()
};
```

#### 3. Configuration Compatibility

**Problem**: PyTorch configurations don't translate directly.

**Solution**: Use migration utilities:
```rust
use torsh_distributed::*;

fn migrate_pytorch_config(pytorch_config: &PyTorchConfig) -> TorshResult<FsdpConfig> {
    let torsh_config = FsdpConfig {
        sharding_strategy: match pytorch_config.sharding_strategy {
            "FULL_SHARD" => ShardingStrategy::FullShard,
            "SHARD_GRAD_OP" => ShardingStrategy::ShardGradOp,
            "NO_SHARD" => ShardingStrategy::NoShard,
            _ => ShardingStrategy::FullShard,
        },
        cpu_offload: pytorch_config.cpu_offload.unwrap_or(false),
        mixed_precision: pytorch_config.mixed_precision.as_ref().map(|mp| {
            MixedPrecisionConfig {
                param_dtype: mp.param_dtype.clone(),
                reduce_dtype: mp.reduce_dtype.clone(),
                buffer_dtype: mp.buffer_dtype.clone(),
                ..Default::default()
            }
        }),
        ..Default::default()
    };
    
    Ok(torsh_config)
}
```

## Migration Checklist

### Pre-Migration Checklist

- [ ] **Inventory Current Setup**
  - [ ] List all PyTorch distributed features used
  - [ ] Document current performance benchmarks
  - [ ] Identify third-party dependencies (Horovod, DeepSpeed, etc.)
  - [ ] Review custom CUDA kernels or C++ extensions

- [ ] **Environment Preparation**
  - [ ] Install Rust toolchain (1.70.0+)
  - [ ] Set up CUDA environment (if using GPUs)
  - [ ] Install NCCL/MPI libraries
  - [ ] Prepare test datasets

- [ ] **Code Analysis**
  - [ ] Identify distributed-specific code sections
  - [ ] Map PyTorch APIs to ToRSh equivalents
  - [ ] Plan gradual migration strategy

### Migration Execution Checklist

- [ ] **Basic Setup Migration**
  - [ ] Convert process group initialization
  - [ ] Migrate basic DDP setup
  - [ ] Test single-node multi-GPU training
  - [ ] Verify gradient synchronization

- [ ] **Advanced Features Migration**
  - [ ] Migrate FSDP configurations
  - [ ] Convert gradient compression setups
  - [ ] Migrate pipeline parallelism
  - [ ] Test mixed precision training

- [ ] **Framework Integration Migration**
  - [ ] Migrate DeepSpeed configurations
  - [ ] Convert Horovod setups
  - [ ] Test framework compatibility layers

- [ ] **Performance Optimization**
  - [ ] Apply ToRSh-specific optimizations
  - [ ] Tune communication parameters
  - [ ] Optimize memory usage
  - [ ] Configure profiling and monitoring

### Post-Migration Validation

- [ ] **Correctness Validation**
  - [ ] Compare training curves with PyTorch baseline
  - [ ] Verify model convergence behavior
  - [ ] Test checkpoint save/restore functionality
  - [ ] Validate multi-node training

- [ ] **Performance Validation**
  - [ ] Benchmark training throughput
  - [ ] Measure memory usage
  - [ ] Profile communication overhead
  - [ ] Compare scaling efficiency

- [ ] **Production Readiness**
  - [ ] Set up monitoring and alerting
  - [ ] Configure fault tolerance
  - [ ] Document new operational procedures
  - [ ] Train team on ToRSh-specific features

### Migration Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Training Throughput** | ≥ PyTorch baseline | Samples/second |
| **Memory Usage** | ≤ 110% of PyTorch | Peak GPU memory |
| **Model Accuracy** | Within 1% of PyTorch | Validation metrics |
| **Time to Convergence** | ≤ 105% of PyTorch | Training steps |
| **Scaling Efficiency** | ≥ 90% of PyTorch | Strong scaling curves |
| **Fault Recovery Time** | < 60 seconds | Checkpoint restore time |

### Common Migration Patterns

```rust
// Pattern 1: PyTorch DDP → ToRSh DDP
fn migrate_basic_ddp() -> TorshResult<()> {
    // Before (PyTorch concept):
    // model = DDP(model, device_ids=[rank])
    
    // After (ToRSh):
    let bucket_config = BucketConfig::default();
    let ddp_model = DistributedDataParallel::new(model, bucket_config)?;
    Ok(())
}

// Pattern 2: PyTorch FSDP → ToRSh FSDP
fn migrate_fsdp() -> TorshResult<()> {
    // Before (PyTorch concept):
    // model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
    
    // After (ToRSh):
    let fsdp_config = FsdpConfig {
        sharding_strategy: ShardingStrategy::FullShard,
        ..Default::default()
    };
    let fsdp_model = FullyShardedDataParallel::new(model, fsdp_config)?;
    Ok(())
}

// Pattern 3: Custom Hooks → ToRSh Callbacks
fn migrate_custom_hooks() -> TorshResult<()> {
    // Before (PyTorch concept):
    // model.register_comm_hook(state, hook_fn)
    
    // After (ToRSh):
    let ddp_model = DistributedDataParallel::new(model, bucket_config)?;
    ddp_model.register_gradient_hook(|gradients| {
        // Custom gradient processing
        Ok(gradients)
    })?;
    Ok(())
}
```

This migration guide provides a comprehensive roadmap for transitioning from PyTorch distributed training to ToRSh. The key is to migrate incrementally, validate at each step, and leverage ToRSh's integration layers for smoother transitions from existing frameworks.