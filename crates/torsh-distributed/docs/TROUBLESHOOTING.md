# ToRSh Distributed Training Troubleshooting Guide

This comprehensive troubleshooting guide helps you diagnose and resolve common issues encountered when using ToRSh distributed training.

## Table of Contents

1. [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
2. [Installation Issues](#installation-issues)
3. [Network and Communication Issues](#network-and-communication-issues)
4. [Backend-Specific Issues](#backend-specific-issues)
5. [Memory and Performance Issues](#memory-and-performance-issues)
6. [Multi-Node Setup Issues](#multi-node-setup-issues)
7. [Framework Integration Issues](#framework-integration-issues)
8. [Debugging Tools and Techniques](#debugging-tools-and-techniques)
9. [Error Reference](#error-reference)
10. [Getting Additional Help](#getting-additional-help)

## Quick Diagnostic Checklist

Before diving into specific troubleshooting sections, run through this quick checklist:

### Environment Check

```bash
# 1. Verify Rust installation
rustc --version  # Should be 1.70.0+

# 2. Check ToRSh compilation
cargo check --package torsh-distributed

# 3. Verify network connectivity (for multi-node)
ping <master_node_ip>
telnet <master_node_ip> 29500

# 4. Check GPU availability (if using CUDA)
nvidia-smi

# 5. Verify environment variables
echo $MASTER_ADDR
echo $MASTER_PORT
echo $WORLD_SIZE
echo $RANK
```

### Quick Test

```rust
// test_basic_setup.rs
use torsh_distributed::*;

fn main() -> TorshResult<()> {
    println!("Testing ToRSh distributed setup...");
    
    // Test availability
    println!("Distributed available: {}", is_available());
    println!("NCCL available: {}", is_nccl_available());
    println!("MPI available: {}", is_mpi_available());
    println!("Gloo available: {}", is_gloo_available());
    
    // Test basic initialization
    let pg = init_process_group(
        BackendType::Mock,  // Use mock for testing
        Rank(0),
        WorldSize(1),
        "localhost",
        29500,
    )?;
    
    println!("✅ Basic setup test passed!");
    Ok(())
}
```

## Installation Issues

### Issue: Compilation Fails with Missing Dependencies

**Error:**
```
error: failed to run custom build command for `torsh-distributed`
note: ld: library not found for -lcuda
```

**Solutions:**

1. **Install CUDA Toolkit:**
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Set environment variables
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

2. **Install NCCL:**
```bash
# Download NCCL from NVIDIA website
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/libnccl2_2.18.3-1+cuda11.8_amd64.deb
sudo dpkg -i libnccl2_2.18.3-1+cuda11.8_amd64.deb

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/libnccl-dev_2.18.3-1+cuda11.8_amd64.deb
sudo dpkg -i libnccl-dev_2.18.3-1+cuda11.8_amd64.deb
```

3. **Build without CUDA features:**
```bash
cargo build --release --no-default-features
```

### Issue: Feature Flags Not Working

**Error:**
```
error: feature 'nccl' not available in this build
```

**Solutions:**

1. **Check available features:**
```bash
cargo metadata --format-version 1 | jq '.packages[] | select(.name == "torsh-distributed") | .features'
```

2. **Build with specific features:**
```bash
# For GPU support
cargo build --release --features="cuda,nccl"

# For MPI support
cargo build --release --features="mpi"

# For all features
cargo build --release --features="cuda,nccl,mpi,gpu"
```

3. **Update Cargo.toml:**
```toml
[dependencies]
torsh-distributed = { version = "0.1.0-alpha.2", features = ["cuda", "nccl"] }
```

### Issue: Linker Errors on Different Platforms

**Error (macOS):**
```
ld: library not found for -lssl
```

**Solutions:**

1. **macOS:**
```bash
# Install OpenSSL
brew install openssl
export PKG_CONFIG_PATH="/usr/local/opt/openssl/lib/pkgconfig"

# For Apple Silicon
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"
```

2. **CentOS/RHEL:**
```bash
sudo yum groupinstall "Development Tools"
sudo yum install openssl-devel
```

3. **Alpine Linux:**
```bash
apk add --no-cache musl-dev openssl-dev
```

## Network and Communication Issues

### Issue: Process Group Initialization Timeout

**Error:**
```
TorshDistributedError::OperationTimeout { operation: "init_process_group", timeout_secs: 300 }
```

**Diagnosis:**
```bash
# Check if master node is reachable
ping $MASTER_ADDR

# Check if port is open
telnet $MASTER_ADDR $MASTER_PORT
nc -zv $MASTER_ADDR $MASTER_PORT

# Check firewall
sudo ufw status
sudo iptables -L -n
```

**Solutions:**

1. **Configure Firewall:**
```bash
# Open required ports
sudo ufw allow 29500:29510/tcp
sudo ufw allow from <worker_node_ip> to any port 29500:29510

# For NCCL
sudo ufw allow 61000:61010/tcp
```

2. **Increase Timeout:**
```rust
use torsh_distributed::*;

// Set longer timeout
std::env::set_var("TORSH_INIT_TIMEOUT", "600"); // 10 minutes

// Or use retry mechanism
let mut retries = 3;
let pg = loop {
    match init_process_group(
        BackendType::Nccl,
        Rank(0),
        WorldSize(4),
        &master_addr,
        master_port,
    ) {
        Ok(pg) => break pg,
        Err(e) if retries > 0 => {
            eprintln!("Init failed, retrying... {}", e);
            retries -= 1;
            std::thread::sleep(std::time::Duration::from_secs(10));
        }
        Err(e) => return Err(e),
    }
};
```

3. **Check Network Configuration:**
```bash
# Show network interfaces
ip addr show

# Check routing
ip route show

# For multi-node setups, ensure all nodes use same network interface
export NCCL_SOCKET_IFNAME=eth0  # or your network interface
```

### Issue: Intermittent Communication Failures

**Error:**
```
TorshDistributedError::CommunicationError { operation: "all_reduce", cause: "Connection reset by peer" }
```

**Diagnosis:**
```bash
# Check network stability
ping -c 100 $MASTER_ADDR
mtr $MASTER_ADDR

# Monitor network traffic
sudo tcpdump -i eth0 port 29500

# Check system logs
journalctl -f
dmesg | tail
```

**Solutions:**

1. **Enable Connection Retry:**
```rust
use torsh_distributed::*;

// Configure retry policy
let retry_config = RetryConfig {
    max_retries: 3,
    initial_delay_ms: 1000,
    max_delay_ms: 10000,
    backoff_factor: 2.0,
    jitter: true,
};

let retry_executor = RetryExecutor::new(retry_config);

// Use with communication operations
retry_executor.execute(|| {
    all_reduce(&tensor, ReduceOp::Sum)
})?;
```

2. **Tune Network Parameters:**
```bash
# Increase TCP buffer sizes
echo 16777216 | sudo tee /proc/sys/net/core/rmem_max
echo 16777216 | sudo tee /proc/sys/net/core/wmem_max

# For NCCL
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=2
```

3. **Use Health Monitoring:**
```rust
use torsh_distributed::*;

let health_checker = HealthChecker::new()?;
health_checker.start_monitoring(std::time::Duration::from_secs(30))?;

// Check health before operations
if health_checker.get_status()? != HealthStatus::Healthy {
    eprintln!("Warning: Unhealthy workers detected");
}
```

## Backend-Specific Issues

### NCCL Issues

#### Issue: NCCL Initialization Fails

**Error:**
```
NCCL WARN Bootstrap : no socket interface found
```

**Solutions:**

1. **Specify Network Interface:**
```bash
export NCCL_SOCKET_IFNAME=eth0,ib0  # Specify your network interfaces
export NCCL_IB_DISABLE=1            # Disable InfiniBand if not available
```

2. **Debug NCCL:**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

3. **Check GPU Topology:**
```bash
nvidia-smi topo -m
```

#### Issue: NCCL Hangs During Communication

**Error:**
Process hangs without error message during collective operations.

**Solutions:**

1. **Enable Timeout:**
```bash
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600  # 1 hour timeout
```

2. **Optimize NCCL Parameters:**
```bash
# For small clusters
export NCCL_TREE_THRESHOLD=0

# For large models
export NCCL_LL_THRESHOLD=0
export NCCL_LL128_THRESHOLD=0
```

### MPI Issues

#### Issue: MPI Not Found

**Error:**
```
mpirun: command not found
```

**Solutions:**

1. **Install MPI:**
```bash
# Ubuntu/Debian
sudo apt install openmpi-bin openmpi-dev

# CentOS/RHEL
sudo yum install openmpi openmpi-devel

# Load MPI module (on HPC systems)
module load mpi/openmpi
```

2. **Set MPI Environment:**
```bash
export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH
```

#### Issue: MPI Process Mapping

**Error:**
```
There are not enough slots available in the system
```

**Solutions:**

1. **Create Hostfile:**
```bash
# hosts.txt
node1 slots=4
node2 slots=4
node3 slots=4
```

2. **Use Proper Mapping:**
```bash
mpirun -np 12 \
    --hostfile hosts.txt \
    --map-by ppr:4:node \
    --bind-to core \
    your_program
```

### Gloo Issues

#### Issue: Gloo Backend Timeout

**Error:**
```
Gloo process group initialization timeout
```

**Solutions:**

1. **Increase Timeout:**
```bash
export GLOO_TIMEOUT_MS=300000  # 5 minutes
```

2. **Check File Descriptor Limits:**
```bash
ulimit -n 65536  # Increase file descriptor limit
```

## Memory and Performance Issues

### Issue: Out of Memory (OOM)

**Error:**
```
TorshDistributedError::MemoryAllocationFailed { requested_bytes: 4294967296, context: "gradient_buffer" }
```

**Diagnosis:**
```bash
# Check memory usage
free -h
nvidia-smi

# Monitor memory during training
watch -n 1 nvidia-smi
```

**Solutions:**

1. **Enable Gradient Checkpointing:**
```rust
use torsh_distributed::*;

let fsdp_config = FsdpConfig {
    // ... other config
    backward_prefetch: Some(BackwardPrefetch::BackwardPre),
    // Enable CPU offloading
    cpu_offload: true,
    // ... other config
};
```

2. **Use Gradient Compression:**
```rust
use torsh_distributed::*;

let compression_config = CompressionConfig {
    method: CompressionMethod::TopK { k: 0.01 },  // Compress to 1%
    error_feedback: true,
    compression_period: 1,
    memory_optimization: true,
};

let compressor = GradientCompressor::new(compression_config)?;
```

3. **Reduce Batch Size:**
```rust
// Use gradient accumulation to maintain effective batch size
let bucket_config = BucketConfig {
    gradient_accumulation_steps: 4,  // Accumulate over 4 steps
    // ... other config
};
```

4. **Enable CPU Offloading:**
```rust
use torsh_distributed::*;

// For ZeRO-3 style offloading
let zero3_config = Zero3CpuOffloadConfig {
    offload_optimizer_state: true,
    offload_parameters: true,
    compression_method: Some(CpuCompressionMethod::FP16),
    memory_strategy: AutoMemoryStrategy::Aggressive,
    // ... other config
};

let zero3_manager = Zero3CpuOffloadManager::new(zero3_config)?;
```

### Issue: Poor Performance

**Symptoms:**
- Low GPU utilization
- High communication overhead
- Slow training progress

**Diagnosis:**
```rust
use torsh_distributed::*;

// Enable profiling
let profiling_config = ProfilingConfig {
    enable_memory_profiling: true,
    enable_communication_profiling: true,
    enable_computation_profiling: true,
    sampling_interval_ms: 100,
    output_file: "torsh_profile.json".to_string(),
};

init_global_profiler(profiling_config)?;

// Monitor bottlenecks
let bottleneck_detector = BottleneckDetector::new()?;
bottleneck_detector.start_monitoring()?;
```

**Solutions:**

1. **Optimize Communication:**
```rust
// Enable communication overlap
let ddp_config = BucketConfig {
    overlap_communication: true,
    bucket_size_mb: 25.0,  // Tune bucket size
    // ... other config
};

// Use communication scheduling
let scheduler_config = SchedulerConfig {
    strategy: SchedulingStrategy::PriorityBased,
    enable_fusion: true,
    max_concurrent_ops: 4,
    // ... other config
};
```

2. **Tune NCCL Parameters:**
```bash
# For bandwidth optimization
export NCCL_ALGO=Ring  # or Tree
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=16

# For latency optimization
export NCCL_BUFFSIZE=2097152  # 2MB buffer
```

3. **Use Mixed Precision:**
```rust
let mixed_precision_config = MixedPrecisionConfig {
    param_dtype: "float16".to_string(),
    reduce_dtype: "float16".to_string(),
    buffer_dtype: "float32".to_string(),
    keep_low_precision_grads: false,
    cast_forward_inputs: true,
    cast_root_forward_inputs: true,
};
```

## Multi-Node Setup Issues

### Issue: Inconsistent Environment Across Nodes

**Error:**
```
TorshDistributedError::ConfigurationError { message: "Mismatched world size across nodes" }
```

**Solutions:**

1. **Use Consistent Launch Script:**
```bash
#!/bin/bash
# launch_all_nodes.sh

NODES=("node1" "node2" "node3" "node4")
MASTER_NODE="node1"
MASTER_PORT="29500"
WORLD_SIZE=16  # 4 nodes × 4 GPUs each

for i in "${!NODES[@]}"; do
    NODE_RANK=$i
    
    ssh ${NODES[$i]} "
        export MASTER_ADDR=$MASTER_NODE
        export MASTER_PORT=$MASTER_PORT
        export WORLD_SIZE=$WORLD_SIZE
        export NODE_RANK=$NODE_RANK
        export LOCAL_WORLD_SIZE=4
        
        cd /path/to/torsh &&
        ./run_training.sh
    " &
done

wait
```

2. **Validate Environment:**
```rust
use torsh_distributed::*;

fn validate_distributed_env() -> TorshResult<()> {
    let master_addr = std::env::var("MASTER_ADDR")
        .map_err(|_| TorshDistributedError::configuration_error("MASTER_ADDR not set"))?;
    
    let world_size: u32 = std::env::var("WORLD_SIZE")
        .map_err(|_| TorshDistributedError::configuration_error("WORLD_SIZE not set"))?
        .parse()
        .map_err(|_| TorshDistributedError::configuration_error("Invalid WORLD_SIZE"))?;
    
    println!("Environment validated: master={}, world_size={}", master_addr, world_size);
    Ok(())
}
```

### Issue: SSH Authentication Failures

**Error:**
```
Permission denied (publickey)
```

**Solutions:**

1. **Setup SSH Keys:**
```bash
# Generate SSH key pair
ssh-keygen -t rsa -b 4096 -f ~/.ssh/torsh_cluster

# Copy to all nodes
for node in node1 node2 node3 node4; do
    ssh-copy-id -i ~/.ssh/torsh_cluster.pub user@$node
done

# Test passwordless SSH
for node in node1 node2 node3 node4; do
    ssh -i ~/.ssh/torsh_cluster user@$node hostname
done
```

2. **Configure SSH Config:**
```bash
# ~/.ssh/config
Host node*
    User torshuser
    IdentityFile ~/.ssh/torsh_cluster
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

### Issue: Time Synchronization

**Error:**
Operations timeout due to clock skew between nodes.

**Solutions:**

1. **Install NTP:**
```bash
# Ubuntu/Debian
sudo apt install ntp
sudo systemctl enable ntp
sudo systemctl start ntp

# CentOS/RHEL
sudo yum install ntp
sudo systemctl enable ntpd
sudo systemctl start ntpd
```

2. **Verify Time Sync:**
```bash
# Check time on all nodes
for node in node1 node2 node3 node4; do
    echo "$node: $(ssh $node date)"
done

# Check NTP status
ntpq -p
```

## Framework Integration Issues

### DeepSpeed Integration Issues

#### Issue: DeepSpeed Config Validation Fails

**Error:**
```
TorshDistributedError::ConfigurationError { message: "ZeRO Stage 3 requires stage3_max_live_parameters to be set" }
```

**Solutions:**

1. **Complete DeepSpeed Config:**
```json
{
  "zero_optimization": {
    "stage": 3,
    "stage3_max_live_parameters": 1000000000,
    "stage3_max_reuse_distance": 1000,
    "stage3_prefetch_bucket_size": 500000000,
    "allgather_bucket_size": 200000000,
    "reduce_bucket_size": 200000000,
    "overlap_comm": true,
    "contiguous_gradients": true
  }
}
```

2. **Validate Config Programmatically:**
```rust
use torsh_distributed::*;

let mut deepspeed = DeepSpeedIntegration::from_file("deepspeed_config.json")?;

// Validate before initialization
if !deepspeed.config().zero_optimization.stage3_max_live_parameters.is_some() {
    eprintln!("Warning: Adding default stage3_max_live_parameters");
    // Set default values
}

deepspeed.initialize(rank, world_size)?;
```

### Horovod Integration Issues

#### Issue: Gradient Compression Not Working

**Error:**
No compression observed, full gradients being transmitted.

**Solutions:**

1. **Verify Compression Config:**
```rust
use torsh_distributed::*;

let config = HorovodConfig {
    gradient_compression: Some(HorovodCompressionConfig {
        compression_type: HorovodCompressionType::TopK,
        compression_params: {
            let mut params = std::collections::HashMap::new();
            params.insert("k".to_string(), 0.01);  // 1% sparsity
            params
        },
        memory_optimization: Some(true),
        compression_period: Some(1),
    }),
    // ... other config
};

let mut horovod = HorovodIntegration::new(config);
```

2. **Monitor Compression Stats:**
```rust
// Check compression statistics
let stats = horovod.stats();
println!("Compression ratio: {:.2}", stats.compression_ratio);
println!("Compressed bytes: {}", stats.compressed_bytes);
println!("Uncompressed bytes: {}", stats.uncompressed_bytes);
```

### Ray Integration Issues

#### Issue: Ray Cluster Connection Fails

**Error:**
```
RayConnectionError: Could not connect to Ray cluster
```

**Solutions:**

1. **Start Ray Cluster:**
```bash
# Start Ray head node
ray start --head --port=10001 --dashboard-host=0.0.0.0

# Connect worker nodes
ray start --address='ray://head-node-ip:10001'
```

2. **Configure Ray in Code:**
```rust
use torsh_distributed::*;

let config = RayConfig {
    cluster: Some(RayClusterConfig {
        address: Some("ray://head-node-ip:10001".to_string()),
        // ... other config
    }),
    // ... other config
};
```

### Dask Integration Issues

#### Issue: Dask Scheduler Not Responsive

**Error:**
Tasks submitted but not executing.

**Solutions:**

1. **Check Dask Dashboard:**
```bash
# Access dashboard at http://scheduler-ip:8787
curl http://scheduler-ip:8787/status
```

2. **Scale Dask Cluster:**
```rust
use torsh_distributed::*;

let mut dask = DaskIntegration::new(config);
dask.initialize(rank, world_size, local_rank, local_size)?;

// Scale to more workers
dask.scale_cluster(16)?;

// Check worker status
println!("Connected workers: {}", dask.stats().workers_connected);
```

## Debugging Tools and Techniques

### Enable Comprehensive Logging

```rust
use torsh_distributed::*;

// Enable debug logging
std::env::set_var("RUST_LOG", "torsh_distributed=debug,torsh_core=info");
env_logger::init();

// Initialize debugging
let debug_config = DebugConfig {
    enable_profiling: true,
    log_communication: true,
    save_timeline: true,
    timeline_file: "torsh_timeline.json".to_string(),
    memory_profiling: true,
    bottleneck_detection: true,
    log_level: LogLevel::Debug,
    max_log_entries: 10000,
};

init_global_debugger(debug_config)?;
```

### System State Snapshot

```rust
use torsh_distributed::*;

// Take system snapshot for debugging
let debugger = get_global_debugger()?;
let snapshot = debugger.take_system_snapshot()?;

println!("System State:");
println!("- Memory usage: {} MB", snapshot.memory_usage_mb);
println!("- CPU usage: {:.1}%", snapshot.cpu_usage_percent);
println!("- Network usage: {} Mbps", snapshot.network_usage_mbps);
println!("- Active operations: {}", snapshot.active_operations.len());

// Save snapshot to file
debugger.save_snapshot(&snapshot, "debug_snapshot.json")?;
```

### Communication Tracing

```rust
use torsh_distributed::*;

// Enable communication tracing
let profiler = get_global_profiler()?;
profiler.start_communication_tracing()?;

// Perform operations
all_reduce(&tensor, ReduceOp::Sum)?;

// Get trace results
let events = profiler.get_communication_events()?;
for event in events {
    println!("Operation: {}, Duration: {:.2}ms, Size: {} bytes",
             event.operation_type, event.duration_ms, event.data_size_bytes);
}
```

### Performance Analysis

```rust
use torsh_distributed::*;

// Enable bottleneck detection
let detector = get_global_bottleneck_detector()?;
detector.start_analysis()?;

// Run training
train_model()?;

// Get bottleneck report
let bottlenecks = detector.get_detected_bottlenecks()?;
for bottleneck in bottlenecks {
    println!("Bottleneck: {:?}, Severity: {:?}, Suggestions: {:?}",
             bottleneck.bottleneck_type, bottleneck.severity, bottleneck.suggestions);
}
```

## Error Reference

### Common Error Codes and Solutions

| Error Code | Description | Common Causes | Solutions |
|------------|-------------|---------------|-----------|
| `BackendNotInitialized` | Process group not initialized | Forgot to call `init_process_group()` | Initialize process group before operations |
| `RankOutOfBounds` | Invalid rank specified | Rank >= world_size | Check rank assignment |
| `CommunicationError` | Network communication failed | Network issues, firewall | Check network connectivity |
| `OperationTimeout` | Operation timed out | Network latency, system load | Increase timeout, check resources |
| `TensorShapeMismatch` | Tensor shapes don't match | Inconsistent tensor creation | Ensure consistent shapes across processes |
| `MemoryAllocationFailed` | Out of memory | Large models, insufficient RAM/VRAM | Reduce batch size, enable offloading |
| `FeatureNotAvailable` | Feature not compiled | Missing feature flags | Rebuild with required features |
| `ProcessFailure` | Worker process failed | System crash, OOM | Check system resources, logs |
| `ConfigurationError` | Invalid configuration | Wrong config values | Validate configuration |
| `CheckpointError` | Checkpoint save/load failed | Disk space, permissions | Check storage and permissions |

### Debug Error Messages

```rust
use torsh_distributed::*;

// Get detailed error information
match operation_result {
    Err(e) => {
        eprintln!("Error: {}", e);
        eprintln!("Retryable: {}", e.is_retryable());
        eprintln!("Recovery suggestions:");
        for suggestion in e.recovery_suggestions() {
            eprintln!("  - {}", suggestion);
        }
    }
    Ok(result) => {
        // Handle success
    }
}
```

## Getting Additional Help

### Diagnostic Information to Collect

When reporting issues, include the following information:

```bash
#!/bin/bash
# collect_debug_info.sh

echo "=== ToRSh Distributed Debug Information ==="
echo "Date: $(date)"
echo

echo "=== System Information ==="
uname -a
lsb_release -a 2>/dev/null || cat /etc/os-release

echo "=== Hardware Information ==="
lscpu | head -20
free -h
nvidia-smi 2>/dev/null || echo "No NVIDIA GPUs found"

echo "=== Software Versions ==="
rustc --version
cargo --version
nvcc --version 2>/dev/null || echo "CUDA not found"

echo "=== Network Configuration ==="
ip addr show
ss -tuln | grep -E ":(29500|8786|8787)"

echo "=== Environment Variables ==="
env | grep -E "(TORSH|MASTER|WORLD|RANK|NCCL|CUDA|LD_LIBRARY_PATH)"

echo "=== Recent Logs ==="
journalctl --since "1 hour ago" | grep -i "torsh\|distributed\|nccl\|cuda" | tail -20

echo "=== ToRSh Configuration ==="
find . -name "Cargo.toml" -exec grep -l "torsh-distributed" {} \; | head -1 | xargs cat
```

### Community Resources

1. **GitHub Issues**: https://github.com/torsh-org/torsh/issues
2. **Documentation**: https://docs.rs/torsh-distributed
3. **Discord Community**: [Join our Discord]
4. **Stack Overflow**: Tag with `torsh`, `distributed-training`, `rust`

### Professional Support

For enterprise support and consulting:
- Email: support@torsh.org
- Enterprise support portal: https://support.torsh.org

### Contributing Bug Fixes

If you find and fix a bug:

1. Create a test case that reproduces the issue
2. Implement the fix
3. Add documentation updates
4. Submit a pull request with:
   - Clear description of the problem
   - Explanation of the solution
   - Test coverage for the fix

This troubleshooting guide should help you resolve most common issues with ToRSh distributed training. If you encounter problems not covered here, please don't hesitate to reach out to the community or file an issue with detailed diagnostic information.