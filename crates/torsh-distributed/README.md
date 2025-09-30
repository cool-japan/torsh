# torsh-distributed

Distributed training support for ToRSh with PyTorch-compatible API.

## Overview

This crate provides distributed and parallel training capabilities including:

- **Data Parallel Training**: DistributedDataParallel (DDP)
- **Model Parallel Training**: Pipeline and tensor parallelism
- **Communication Backends**: NCCL, Gloo, MPI support
- **RPC Framework**: Remote procedure calls for distributed computing
- **Collective Operations**: All-reduce, broadcast, gather, scatter

## Usage

### Basic Distributed Training

```rust
use torsh_distributed::prelude::*;
use torsh_nn::prelude::*;

// Initialize process group
init_process_group(
    Backend::NCCL,
    InitMethod::Env,
    None,
    None,
)?;

// Get rank and world size
let rank = get_rank();
let world_size = get_world_size();

// Create model and wrap with DDP
let model = create_model();
let ddp_model = DistributedDataParallel::new(
    model,
    vec![rank as i32], // device_ids
    rank as i32,       // output_device
    vec![],           // broadcast_buffers
    true,             // find_unused_parameters
)?;

// Distributed optimizer
let optimizer = DistributedOptimizer::new(
    SGD::new(ddp_model.parameters(), 0.1, None, None, None, false),
)?;

// Training loop
for epoch in 0..num_epochs {
    for batch in dataloader {
        let output = ddp_model.forward(&batch.input)?;
        let loss = compute_loss(&output, &batch.target)?;
        
        loss.backward()?;
        optimizer.step()?;
        optimizer.zero_grad();
    }
}

// Cleanup
destroy_process_group()?;
```

### Collective Operations

```rust
use torsh_distributed::collectives::*;

// All-reduce: sum tensors across all processes
let tensor = create_tensor();
all_reduce(&mut tensor, ReduceOp::Sum)?;

// Broadcast: send tensor from rank 0 to all others
broadcast(&mut tensor, 0)?;

// Gather: collect tensors from all ranks
let gathered = all_gather(&tensor)?;

// Scatter: distribute chunks to different ranks
let chunks = scatter(&tensor, 0)?;

// Reduce: aggregate to specific rank
reduce(&mut tensor, ReduceOp::Sum, 0)?;
```

### RPC Framework

```rust
use torsh_distributed::rpc::*;

// Initialize RPC
init_rpc(
    "worker",
    rank,
    world_size,
    Some(RpcBackendOptions::default()),
)?;

// Remote procedure call
let future = rpc_async(
    "worker1",
    "process_data",
    &[tensor.clone()],
)?;

// Get result
let result = future.wait()?;

// Remote reference
let rref = remote(&tensor, "worker2")?;
let local_value = rref.to_here()?;

// Shutdown RPC
shutdown_rpc()?;
```

### Pipeline Parallelism

```rust
use torsh_distributed::pipeline::*;

// Split model into stages
let stages = vec![
    stage1_layers,
    stage2_layers,
    stage3_layers,
    stage4_layers,
];

// Create pipeline
let pipeline = PipelineParallel::new(
    stages,
    num_microbatches,
    device_placement,
)?;

// Forward with micro-batching
let output = pipeline.forward(input)?;
```

### Model Parallel

```rust
use torsh_distributed::model_parallel::*;

// Tensor parallel linear layer
let tp_linear = ColumnParallelLinear::new(
    in_features,
    out_features,
    bias,
    gather_output,
)?;

// Attention with tensor parallelism
let tp_attention = ParallelAttention::new(
    embed_dim,
    num_heads,
    dropout,
)?;
```

### Gradient Compression

```rust
use torsh_distributed::compression::*;

// Configure gradient compression
let compressor = GradientCompressor::new()
    .algorithm(CompressionAlgorithm::TopK(0.1))
    .memory(CompressorMemory::Residual);

// Apply to DDP
let ddp_model = DistributedDataParallel::new(model, ...)
    .with_compression(compressor)?;
```

### Fault Tolerance

```rust
use torsh_distributed::elastic::*;

// Elastic training with dynamic workers
let elastic_agent = ElasticAgent::new()
    .min_workers(2)
    .max_workers(8)
    .checkpoint_dir("./checkpoints");

elastic_agent.run(train_fn)?;
```

### Monitoring

```rust
use torsh_distributed::monitoring::*;

// Track distributed metrics
let monitor = DistributedMonitor::new();

// Log communication time
monitor.log_comm_time("all_reduce", duration);

// Get statistics
let stats = monitor.get_stats();
println!("Total communication time: {:?}", stats.total_comm_time);
```

## Backends

### NCCL (NVIDIA GPUs)
- Optimized for NVIDIA GPU communication
- Supports GPUDirect and NVLink
- Best for single-node multi-GPU

### Gloo (CPU and GPU)
- Cross-platform communication
- Supports both TCP and InfiniBand
- Good for CPU training

### MPI (HPC environments)
- Integration with MPI implementations
- Optimized for HPC clusters
- Supports various interconnects

## Environment Variables

```bash
# Basic setup
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=4

# NCCL specific
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# Gloo specific
export GLOO_SOCKET_IFNAME=eth0
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.