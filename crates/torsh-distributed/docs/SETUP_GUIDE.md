# ToRSh Distributed Training Setup Guide

This comprehensive guide will help you set up ToRSh distributed training in various environments and configurations.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Basic Setup](#basic-setup)
4. [Multi-Node Setup](#multi-node-setup)
5. [Backend Configuration](#backend-configuration)
6. [Framework Integrations](#framework-integrations)
7. [Docker Deployment](#docker-deployment)
8. [Kubernetes Deployment](#kubernetes-deployment)
9. [HPC Environments](#hpc-environments)
10. [Cloud Deployment](#cloud-deployment)
11. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+, RHEL 7+, CentOS 7+)
- **Rust**: 1.70.0 or higher
- **CPU**: Multi-core processor (4+ cores recommended)
- **Memory**: 8GB RAM minimum (16GB+ recommended for large models)
- **Network**: High-bandwidth, low-latency network for multi-node training
- **Storage**: Fast SSD storage recommended for checkpoints and data

### Optional Dependencies

- **CUDA**: 11.8+ for GPU acceleration (with compatible NVIDIA drivers)
- **NCCL**: 2.18+ for NVIDIA GPU collective operations
- **MPI**: OpenMPI 4.0+ or MPICH 3.3+ for MPI backend
- **InfiniBand**: For high-performance networking in HPC environments

### Development Tools

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install build essentials
sudo apt update
sudo apt install build-essential cmake pkg-config libssl-dev

# For CUDA support (optional)
# Download and install CUDA toolkit from NVIDIA website
# Add CUDA to PATH and LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/torsh-org/torsh.git
cd torsh

# Build with default features
cargo build --release

# Build with GPU support
cargo build --release --features="cuda,nccl"

# Build with MPI support
cargo build --release --features="mpi"

# Build with all features
cargo build --release --features="cuda,nccl,mpi,gpu"
```

### Using Cargo

```bash
# Add to your Cargo.toml
[dependencies]
torsh-distributed = "0.1.0-alpha.1"

# With features
[dependencies]
torsh-distributed = { version = "0.1.0-alpha.1", features = ["cuda", "nccl"] }
```

### Verify Installation

```bash
# Test basic functionality
cargo test --package torsh-distributed

# Test with specific backend
cargo test --package torsh-distributed --features="nccl" -- nccl

# Run benchmarks
cargo bench --package torsh-distributed
```

## Basic Setup

### Single-Node Training

```rust
use torsh_distributed::*;

fn main() -> TorshResult<()> {
    // Initialize process group for single-node, multi-GPU training
    let pg = init_process_group(
        BackendType::Nccl,  // or BackendType::Gloo for CPU
        Rank(0),
        WorldSize(4),  // Number of GPUs
        "localhost",
        29500,
    )?;

    // Your training code here
    println!("Process group initialized with {} workers", pg.world_size().0);
    
    Ok(())
}
```

### Basic DDP Setup

```rust
use torsh_distributed::*;
use torsh_nn::Module;

fn setup_ddp_training() -> TorshResult<()> {
    // Initialize distributed training
    let pg = init_process_group(
        BackendType::Nccl,
        Rank(0),
        WorldSize(4),
        "localhost",
        29500,
    )?;

    // Create your model
    let model = create_model()?;
    
    // Wrap with DDP
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
            let output = ddp_model.forward(&batch.input)?;
            let loss = compute_loss(&output, &batch.target)?;
            loss.backward()?;
            ddp_model.step()?;
        }
    }
    
    Ok(())
}
```

## Multi-Node Setup

### Network Configuration

```bash
# Ensure all nodes can communicate
# Open necessary ports in firewall
sudo ufw allow 29500:29510/tcp  # Default port range

# For NCCL (if using GPU)
sudo ufw allow 61000:61010/tcp  # NCCL port range

# Test connectivity between nodes
ping node1
ping node2
telnet node1 29500
```

### SSH Key Setup

```bash
# Generate SSH keys on master node
ssh-keygen -t rsa -b 4096 -f ~/.ssh/torsh_cluster

# Copy public key to all nodes
ssh-copy-id -i ~/.ssh/torsh_cluster.pub user@node1
ssh-copy-id -i ~/.ssh/torsh_cluster.pub user@node2

# Test passwordless SSH
ssh -i ~/.ssh/torsh_cluster user@node1
ssh -i ~/.ssh/torsh_cluster user@node2
```

### Multi-Node Launch Script

```bash
#!/bin/bash
# launch_distributed.sh

MASTER_NODE="node1"
MASTER_PORT="29500"
WORLD_SIZE=8
NODES=("node1" "node2")
GPUS_PER_NODE=4

for i in "${!NODES[@]}"; do
    NODE_RANK=$((i * GPUS_PER_NODE))
    
    ssh ${NODES[$i]} "
        cd /path/to/torsh &&
        MASTER_ADDR=$MASTER_NODE \
        MASTER_PORT=$MASTER_PORT \
        WORLD_SIZE=$WORLD_SIZE \
        NODE_RANK=$NODE_RANK \
        LOCAL_RANK=0 \
        cargo run --release --bin distributed_training
    " &
done

wait  # Wait for all processes to complete
```

### Environment Variables

```bash
# Required environment variables for multi-node setup
export MASTER_ADDR="192.168.1.10"  # IP of master node
export MASTER_PORT="29500"          # Port for communication
export WORLD_SIZE="8"               # Total number of processes
export NODE_RANK="0"                # Rank of current node (0, 1, 2, ...)
export LOCAL_RANK="0"               # Local rank within node (0-3 for 4 GPUs)
export LOCAL_WORLD_SIZE="4"         # Number of processes on this node
```

## Backend Configuration

### NCCL Backend (GPU)

```rust
use torsh_distributed::*;

fn setup_nccl_backend() -> TorshResult<()> {
    // Check NCCL availability
    if !is_nccl_available() {
        return Err(TorshDistributedError::feature_not_available(
            "NCCL", 
            "cuda,nccl"
        ));
    }
    
    // Initialize with NCCL backend
    let pg = init_process_group(
        BackendType::Nccl,
        Rank(std::env::var("LOCAL_RANK")?.parse()?),
        WorldSize(std::env::var("WORLD_SIZE")?.parse()?),
        &std::env::var("MASTER_ADDR")?,
        std::env::var("MASTER_PORT")?.parse()?,
    )?;
    
    // Configure NCCL optimizations
    #[cfg(feature = "nccl")]
    {
        use torsh_distributed::nccl_optimization::*;
        
        let scheduler = NcclScheduler::new()?;
        scheduler.optimize_communication_patterns()?;
        scheduler.enable_kernel_fusion(true)?;
    }
    
    Ok(())
}
```

### Gloo Backend (CPU)

```rust
use torsh_distributed::*;

fn setup_gloo_backend() -> TorshResult<()> {
    // Initialize with Gloo backend for CPU training
    let pg = init_process_group(
        BackendType::Gloo,
        Rank(std::env::var("LOCAL_RANK")?.parse()?),
        WorldSize(std::env::var("WORLD_SIZE")?.parse()?),
        &std::env::var("MASTER_ADDR")?,
        std::env::var("MASTER_PORT")?.parse()?,
    )?;
    
    // Gloo works well for CPU-only distributed training
    println!("Gloo backend initialized for CPU training");
    
    Ok(())
}
```

### MPI Backend

```rust
use torsh_distributed::*;

fn setup_mpi_backend() -> TorshResult<()> {
    // Check MPI availability
    if !is_mpi_available() {
        return Err(TorshDistributedError::feature_not_available(
            "MPI", 
            "mpi"
        ));
    }
    
    // MPI initialization is typically handled by mpirun
    let pg = init_process_group(
        BackendType::Mpi,
        Rank(0),  // MPI handles rank assignment
        WorldSize(1),  // MPI handles world size
        "localhost",
        0,  // Port not used for MPI
    )?;
    
    Ok(())
}
```

### Launch with MPI

```bash
# Launch MPI job
mpirun -np 8 \
    --hostfile hosts.txt \
    --bind-to core \
    --map-by ppr:4:node \
    cargo run --release --features="mpi" --bin distributed_training

# hosts.txt example:
# node1 slots=4
# node2 slots=4
```

## Framework Integrations

### DeepSpeed Integration

```rust
use torsh_distributed::*;

fn setup_deepspeed() -> TorshResult<()> {
    // Load DeepSpeed configuration
    let config = DeepSpeedConfig::from_file("deepspeed_config.json")?;
    
    // Initialize DeepSpeed integration
    let mut deepspeed = DeepSpeedIntegration::new(config);
    deepspeed.initialize(
        Rank(0),
        WorldSize(4),
    )?;
    
    // Convert to ToRSh FSDP configuration
    let fsdp_config = deepspeed.to_fsdp_config()?;
    
    // Use with your model
    let model = create_model()?;
    let fsdp_model = FullyShardedDataParallel::new(model, fsdp_config)?;
    
    Ok(())
}
```

DeepSpeed configuration example (`deepspeed_config.json`):

```json
{
  "zero_optimization": {
    "stage": 3,
    "allgather_bucket_size": 200000000,
    "reduce_bucket_size": 200000000,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1000000000,
    "stage3_max_reuse_distance": 1000,
    "stage3_prefetch_bucket_size": 500000000,
    "offload_optimizer": {
      "device": "cpu"
    },
    "offload_param": {
      "device": "cpu"
    }
  },
  "gradient_clipping": 1.0,
  "gradient_accumulation_steps": 4,
  "fp16": {
    "enabled": true,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  }
}
```

### Horovod Integration

```rust
use torsh_distributed::*;

fn setup_horovod() -> TorshResult<()> {
    // Create Horovod configuration with gradient compression
    let config = HorovodIntegration::config_with_topk_compression(0.01);
    
    // Initialize Horovod integration
    let mut horovod = HorovodIntegration::new(config);
    horovod.initialize(
        rank(),
        size(),
        local_rank(),
        local_size(),
    )?;
    
    // Convert to ToRSh DDP configuration
    let ddp_config = horovod.to_ddp_config()?;
    let compression_config = horovod.to_compression_config()?;
    
    // Use with your model
    let model = create_model()?;
    let ddp_model = DistributedDataParallel::new(model, ddp_config)?;
    
    Ok(())
}
```

### Ray Integration

```rust
use torsh_distributed::*;

fn setup_ray_training() -> TorshResult<()> {
    // Create Ray configuration for distributed training
    let config = RayIntegration::default_config();
    
    // Initialize Ray integration
    let mut ray = RayIntegration::new(config);
    ray.initialize(0, 4, 0, 1)?;
    
    // Run distributed training
    ray.run_training("my_training_function", 100)?;
    
    // For hyperparameter tuning
    let tune_config = RayIntegration::config_with_tune(50, RaySearchAlgorithm::BayesOpt);
    let mut ray_tune = RayIntegration::new(tune_config);
    ray_tune.initialize(0, 4, 0, 1)?;
    ray_tune.run_tuning("hyperparameter_search")?;
    
    Ok(())
}
```

### Dask Integration

```rust
use torsh_distributed::*;

fn setup_dask_computing() -> TorshResult<()> {
    // Create Dask configuration for large-scale computing
    let config = DaskIntegration::config_with_large_scale(16, "8GB");
    
    // Initialize Dask integration
    let mut dask = DaskIntegration::new(config);
    dask.initialize(0, 16, 0, 4)?;
    
    // Submit tasks to Dask cluster
    let task_id = dask.submit_task("preprocess_data", 1024 * 1024)?;
    println!("Submitted task: {}", task_id);
    
    // Compute collections
    dask.compute("training_dataset")?;
    
    // Scale cluster dynamically
    dask.scale_cluster(32)?;
    
    Ok(())
}
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
WORKDIR /app
COPY . .

# Build with CUDA support
RUN cargo build --release --features="cuda,nccl"

# Expose ports
EXPOSE 29500

# Entry point
CMD ["cargo", "run", "--release", "--features=cuda,nccl", "--bin", "distributed_training"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  master:
    build: .
    environment:
      - MASTER_ADDR=master
      - MASTER_PORT=29500
      - WORLD_SIZE=4
      - NODE_RANK=0
      - LOCAL_RANK=0
    ports:
      - "29500:29500"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]

  worker1:
    build: .
    environment:
      - MASTER_ADDR=master
      - MASTER_PORT=29500
      - WORLD_SIZE=4
      - NODE_RANK=1
      - LOCAL_RANK=0
    depends_on:
      - master
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
```

### Launch with Docker

```bash
# Build the image
docker build -t torsh-distributed .

# Run single container
docker run --gpus all \
    -e MASTER_ADDR=localhost \
    -e MASTER_PORT=29500 \
    -e WORLD_SIZE=1 \
    -e NODE_RANK=0 \
    -e LOCAL_RANK=0 \
    -p 29500:29500 \
    torsh-distributed

# Launch with docker-compose
docker-compose up --scale worker1=3
```

## Kubernetes Deployment

### Kubernetes Manifests

```yaml
# torsh-distributed-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: torsh-distributed-training
spec:
  parallelism: 4
  completions: 4
  template:
    metadata:
      labels:
        app: torsh-distributed
    spec:
      restartPolicy: OnFailure
      containers:
      - name: torsh-worker
        image: torsh-distributed:latest
        env:
        - name: MASTER_ADDR
          value: "torsh-master-service"
        - name: MASTER_PORT
          value: "29500"
        - name: WORLD_SIZE
          value: "4"
        - name: NODE_RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
        ports:
        - containerPort: 29500
---
apiVersion: v1
kind: Service
metadata:
  name: torsh-master-service
spec:
  selector:
    app: torsh-distributed
  ports:
  - port: 29500
    targetPort: 29500
  clusterIP: None  # Headless service
```

### Deploy to Kubernetes

```bash
# Apply the manifests
kubectl apply -f torsh-distributed-job.yaml

# Monitor the job
kubectl get jobs
kubectl get pods

# Check logs
kubectl logs -l app=torsh-distributed

# Delete the job
kubectl delete job torsh-distributed-training
```

## HPC Environments

### SLURM Integration

```bash
#!/bin/bash
#SBATCH --job-name=torsh-distributed
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# Load modules
module load cuda/11.8
module load rust/1.70

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# Launch distributed training
srun cargo run --release --features="cuda,nccl" --bin distributed_training
```

### PBS Integration

```bash
#!/bin/bash
#PBS -N torsh-distributed
#PBS -l nodes=4:ppn=4:gpus=4
#PBS -l walltime=12:00:00
#PBS -q gpu

cd $PBS_O_WORKDIR

# Get node list
NODELIST=$(cat $PBS_NODEFILE | uniq)
MASTER_NODE=$(echo $NODELIST | head -n 1)

export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500
export WORLD_SIZE=$(wc -l < $PBS_NODEFILE)

# Launch on all nodes
mpirun -machinefile $PBS_NODEFILE \
    cargo run --release --features="cuda,nccl" --bin distributed_training
```

## Cloud Deployment

### AWS Setup

```bash
# Launch EC2 instances with Deep Learning AMI
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --count 4 \
    --instance-type p3.2xlarge \
    --key-name my-key-pair \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678

# Configure security group for distributed training
aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 29500-29510 \
    --source-group sg-12345678
```

### Azure Setup

```bash
# Create resource group
az group create --name torsh-rg --location eastus

# Create VM scale set
az vmss create \
    --resource-group torsh-rg \
    --name torsh-vmss \
    --image UbuntuLTS \
    --vm-sku Standard_NC6s_v3 \
    --instance-count 4 \
    --vnet-name torsh-vnet \
    --subnet torsh-subnet \
    --lb torsh-lb \
    --admin-username azureuser \
    --generate-ssh-keys
```

### GCP Setup

```bash
# Create instance template
gcloud compute instance-templates create torsh-template \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    --restart-on-failure

# Create managed instance group
gcloud compute instance-groups managed create torsh-group \
    --template=torsh-template \
    --size=4 \
    --zone=us-central1-a
```

## Troubleshooting

### Common Issues

#### 1. NCCL Initialization Timeout

```bash
# Increase NCCL timeout
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600  # 1 hour

# Debug NCCL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

#### 2. Network Connectivity Issues

```bash
# Test network connectivity
nc -zv master_node 29500

# Check firewall
sudo ufw status
sudo iptables -L

# For InfiniBand networks
ibstat
ibv_devinfo
```

#### 3. CUDA/GPU Issues

```bash
# Check GPU availability
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify NCCL installation
ldconfig -p | grep nccl
```

#### 4. Memory Issues

```bash
# Monitor memory usage
nvidia-smi -l 1
htop

# Reduce batch size or enable gradient checkpointing
# Enable CPU offloading for large models
```

### Debug Mode

```rust
use torsh_distributed::*;

fn enable_debug_logging() -> TorshResult<()> {
    // Enable debug logging
    std::env::set_var("RUST_LOG", "torsh_distributed=debug");
    env_logger::init();
    
    // Initialize debugging
    let debug_config = DebugConfig {
        enable_profiling: true,
        log_communication: true,
        save_timeline: true,
        timeline_file: "torsh_timeline.json".to_string(),
        memory_profiling: true,
        bottleneck_detection: true,
    };
    
    init_global_debugger(debug_config)?;
    
    Ok(())
}
```

### Performance Monitoring

```rust
use torsh_distributed::*;

fn setup_monitoring() -> TorshResult<()> {
    // Initialize metrics collection
    let metrics_config = MetricsConfig {
        collection_interval_ms: 1000,
        enable_system_metrics: true,
        enable_communication_metrics: true,
        enable_training_metrics: true,
        export_prometheus: true,
        prometheus_port: 9090,
    };
    
    init_global_metrics_collector(metrics_config)?;
    start_global_metrics_collection()?;
    
    // Initialize bottleneck detection
    let bottleneck_config = BottleneckDetectionConfig {
        detection_interval_ms: 5000,
        thresholds: BottleneckThresholds {
            cpu_utilization: 0.9,
            memory_utilization: 0.8,
            network_utilization: 0.8,
            gpu_utilization: 0.9,
        },
        enable_auto_scaling: true,
        enable_notifications: true,
    };
    
    init_global_bottleneck_detector(bottleneck_config)?;
    
    Ok(())
}
```

### Getting Help

- **Documentation**: Check the [API documentation](https://docs.rs/torsh-distributed)
- **GitHub Issues**: Report bugs and feature requests at [GitHub](https://github.com/torsh-org/torsh/issues)
- **Community Discord**: Join our Discord server for real-time help
- **Stack Overflow**: Tag questions with `torsh` and `distributed-training`

### Quick Diagnostic Script

```bash
#!/bin/bash
# diagnose_torsh.sh

echo "=== ToRSh Distributed Training Diagnostics ==="

echo "1. Rust Version:"
rustc --version

echo "2. CUDA Version:"
nvcc --version 2>/dev/null || echo "CUDA not found"

echo "3. GPU Information:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not found"

echo "4. Network Interfaces:"
ip addr show

echo "5. Available Ports:"
ss -tuln | grep :29500

echo "6. Environment Variables:"
env | grep -E "(MASTER_|WORLD_|LOCAL_|NODE_|RANK)"

echo "7. ToRSh Features:"
cargo metadata --format-version 1 | jq '.packages[] | select(.name == "torsh-distributed") | .features'

echo "Diagnostics complete."
```

This setup guide provides comprehensive instructions for deploying ToRSh distributed training in various environments. Follow the sections relevant to your setup, and refer to the troubleshooting section if you encounter issues.