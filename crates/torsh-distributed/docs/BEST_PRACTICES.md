# ToRSh Distributed Training Best Practices

This guide covers best practices for efficient and scalable distributed training with ToRSh. Following these practices will help you achieve optimal performance, reliability, and maintainability in your distributed training workflows.

## Table of Contents

1. [Architecture and Design Patterns](#architecture-and-design-patterns)
2. [Performance Optimization](#performance-optimization)
3. [Memory Management](#memory-management)
4. [Communication Optimization](#communication-optimization)
5. [Fault Tolerance and Reliability](#fault-tolerance-and-reliability)
6. [Model Design for Distribution](#model-design-for-distribution)
7. [Data Management](#data-management)
8. [Monitoring and Debugging](#monitoring-and-debugging)
9. [Framework-Specific Best Practices](#framework-specific-best-practices)
10. [Production Deployment](#production-deployment)

## Architecture and Design Patterns

### Choose the Right Parallelism Strategy

```rust
use torsh_distributed::*;

fn select_parallelism_strategy(
    model_size: usize,
    available_memory: usize,
    num_nodes: usize,
    num_gpus_per_node: usize,
) -> TorshResult<ParallelismConfig> {
    
    let total_gpus = num_nodes * num_gpus_per_node;
    
    match (model_size, total_gpus) {
        // Small models: Use DDP
        (size, _) if size < 1_000_000_000 => {
            Ok(ParallelismConfig::DataParallel {
                bucket_config: BucketConfig {
                    bucket_size_mb: 25.0,
                    overlap_communication: true,
                    gradient_compression: false,
                    ..Default::default()
                }
            })
        }
        
        // Medium models: Use FSDP
        (size, gpus) if size < 10_000_000_000 && gpus <= 32 => {
            Ok(ParallelismConfig::FullySharded {
                fsdp_config: FsdpConfig {
                    sharding_strategy: ShardingStrategy::FullShard,
                    cpu_offload: size > available_memory * gpus as usize,
                    mixed_precision: Some(MixedPrecisionConfig::default()),
                    ..Default::default()
                }
            })
        }
        
        // Large models: Use 3D parallelism
        (_, gpus) if gpus > 32 => {
            Ok(ParallelismConfig::ThreeD {
                config: ThreeDParallelismConfig {
                    data_parallel_size: num_nodes as u32,
                    tensor_parallel_size: num_gpus_per_node as u32,
                    pipeline_parallel_size: 4,
                    memory_optimization: MemoryOptimizationStrategy::Aggressive,
                    communication_strategy: CommunicationStrategy::Hierarchical,
                }
            })
        }
        
        _ => Ok(ParallelismConfig::DataParallel {
            bucket_config: BucketConfig::default()
        })
    }
}

enum ParallelismConfig {
    DataParallel { bucket_config: BucketConfig },
    FullySharded { fsdp_config: FsdpConfig },
    ThreeD { config: ThreeDParallelismConfig },
}
```

### Layer-wise Distributed Model Design

```rust
use torsh_distributed::*;
use torsh_nn::Module;

trait DistributedModule: Module {
    /// Configure which layers should be sharded
    fn shard_config(&self) -> Vec<ShardingDecision>;
    
    /// Configure communication patterns
    fn communication_config(&self) -> CommunicationConfig;
    
    /// Optimize memory usage patterns
    fn memory_config(&self) -> MemoryConfig;
}

#[derive(Debug)]
enum ShardingDecision {
    /// Keep layer on single device
    NoShard,
    /// Shard parameters across devices
    ParameterShard,
    /// Shard along input dimension
    InputShard,
    /// Shard along output dimension
    OutputShard,
    /// Pipeline stage boundary
    PipelineBoundary,
}

impl DistributedModule for TransformerBlock {
    fn shard_config(&self) -> Vec<ShardingDecision> {
        vec![
            ShardingDecision::InputShard,      // Input embeddings
            ShardingDecision::ParameterShard,  // Attention weights
            ShardingDecision::ParameterShard,  // MLP weights
            ShardingDecision::OutputShard,     // Output projection
        ]
    }
    
    fn communication_config(&self) -> CommunicationConfig {
        CommunicationConfig {
            overlap_computation: true,
            fuse_small_ops: true,
            use_async_collective: true,
            compression_threshold_mb: 1.0,
        }
    }
    
    fn memory_config(&self) -> MemoryConfig {
        MemoryConfig {
            enable_checkpointing: true,
            checkpoint_ratio: 0.5,
            offload_activations: false,
            gradient_accumulation: true,
        }
    }
}
```

### Hierarchical Communication Patterns

```rust
use torsh_distributed::*;

fn setup_hierarchical_communication(
    world_size: u32,
    local_size: u32,
) -> TorshResult<CommunicationTopology> {
    
    let num_nodes = world_size / local_size;
    
    // Create node-local groups for fast intra-node communication
    let mut local_groups = Vec::new();
    for node in 0..num_nodes {
        let start_rank = node * local_size;
        let end_rank = start_rank + local_size;
        let ranks: Vec<u32> = (start_rank..end_rank).collect();
        local_groups.push(CommunicationGroup::new(ranks)?);
    }
    
    // Create inter-node groups for cross-node communication
    let mut inter_node_groups = Vec::new();
    for local_rank in 0..local_size {
        let mut ranks = Vec::new();
        for node in 0..num_nodes {
            ranks.push(node * local_size + local_rank);
        }
        inter_node_groups.push(CommunicationGroup::new(ranks)?);
    }
    
    Ok(CommunicationTopology {
        local_groups,
        inter_node_groups,
        topology_type: TopologyType::Hierarchical,
    })
}

#[derive(Debug)]
struct CommunicationTopology {
    local_groups: Vec<CommunicationGroup>,
    inter_node_groups: Vec<CommunicationGroup>,
    topology_type: TopologyType,
}

#[derive(Debug)]
enum TopologyType {
    Flat,
    Hierarchical,
    Tree,
    Ring,
}
```

## Performance Optimization

### Optimal Batch Size Selection

```rust
use torsh_distributed::*;

fn calculate_optimal_batch_size(
    model_params: usize,
    available_memory_gb: f32,
    world_size: u32,
    target_throughput: f32,
) -> TorshResult<BatchSizeConfig> {
    
    // Rule of thumb: 70% of memory for model, 20% for activations, 10% buffer
    let memory_for_batch = available_memory_gb * 0.2 * 1024.0 * 1024.0 * 1024.0; // Convert to bytes
    
    // Estimate memory per sample (rough heuristic)
    let memory_per_sample = model_params * 4; // 4 bytes per parameter (FP32)
    let max_local_batch_size = (memory_for_batch / memory_per_sample as f32) as u32;
    
    // Consider gradient accumulation for larger effective batch sizes
    let effective_batch_size = calculate_effective_batch_size(target_throughput, world_size);
    let gradient_accumulation_steps = effective_batch_size / (max_local_batch_size * world_size);
    
    Ok(BatchSizeConfig {
        local_batch_size: max_local_batch_size,
        gradient_accumulation_steps: gradient_accumulation_steps.max(1),
        effective_batch_size,
        micro_batch_size: max_local_batch_size / gradient_accumulation_steps.max(1),
    })
}

#[derive(Debug)]
struct BatchSizeConfig {
    local_batch_size: u32,
    gradient_accumulation_steps: u32,
    effective_batch_size: u32,
    micro_batch_size: u32,
}

fn calculate_effective_batch_size(target_throughput: f32, world_size: u32) -> u32 {
    // Scale batch size with number of workers, but with diminishing returns
    let base_batch_size = 32;
    let scaling_factor = (world_size as f32).sqrt();
    (base_batch_size as f32 * scaling_factor) as u32
}
```

### Adaptive Learning Rate Scaling

```rust
use torsh_distributed::*;

fn scale_learning_rate_for_distributed(
    base_lr: f32,
    effective_batch_size: u32,
    world_size: u32,
    scaling_strategy: LRScalingStrategy,
) -> f32 {
    match scaling_strategy {
        LRScalingStrategy::Linear => {
            // Linear scaling: lr = base_lr * world_size
            base_lr * world_size as f32
        }
        
        LRScalingStrategy::SquareRoot => {
            // Square root scaling: lr = base_lr * sqrt(world_size)
            base_lr * (world_size as f32).sqrt()
        }
        
        LRScalingStrategy::Adaptive => {
            // Adaptive scaling based on batch size
            let batch_scale = (effective_batch_size as f32 / 32.0).sqrt();
            base_lr * batch_scale
        }
        
        LRScalingStrategy::Warmup => {
            // Use linear scaling with warmup
            base_lr * world_size as f32
        }
    }
}

#[derive(Debug)]
enum LRScalingStrategy {
    Linear,
    SquareRoot,
    Adaptive,
    Warmup,
}

fn setup_lr_scheduler_with_warmup(
    optimizer: &mut Optimizer,
    base_lr: f32,
    world_size: u32,
    warmup_steps: u32,
) -> TorshResult<LRScheduler> {
    let scaled_lr = scale_learning_rate_for_distributed(
        base_lr,
        0, // Will be set dynamically
        world_size,
        LRScalingStrategy::Warmup,
    );
    
    Ok(LRScheduler {
        base_lr,
        target_lr: scaled_lr,
        warmup_steps,
        current_step: 0,
        scheduler_type: SchedulerType::LinearWarmup,
    })
}
```

### Communication-Computation Overlap

```rust
use torsh_distributed::*;

struct OverlapManager {
    communication_queue: Vec<CommunicationOp>,
    computation_queue: Vec<ComputationOp>,
    overlap_enabled: bool,
}

impl OverlapManager {
    fn new(enable_overlap: bool) -> Self {
        Self {
            communication_queue: Vec::new(),
            computation_queue: Vec::new(),
            overlap_enabled: enable_overlap,
        }
    }
    
    fn schedule_backward_pass(
        &mut self,
        layers: &[Layer],
        gradients: &[Tensor],
    ) -> TorshResult<()> {
        if !self.overlap_enabled {
            // Sequential execution
            for (layer, grad) in layers.iter().zip(gradients.iter()) {
                layer.backward(grad)?;
                self.all_reduce_gradient(grad)?;
            }
            return Ok(());
        }
        
        // Overlapped execution
        for (i, (layer, grad)) in layers.iter().zip(gradients.iter()).enumerate() {
            // Start computation for current layer
            let comp_op = ComputationOp::Backward {
                layer_id: i,
                gradient: grad.clone(),
            };
            self.computation_queue.push(comp_op);
            
            // Start communication for previous layer (if available)
            if i > 0 {
                let comm_op = CommunicationOp::AllReduce {
                    tensor: gradients[i - 1].clone(),
                    reduce_op: ReduceOp::Sum,
                };
                self.communication_queue.push(comm_op);
            }
            
            // Execute overlapped operations
            self.execute_overlapped_ops()?;
        }
        
        // Handle last layer communication
        if !gradients.is_empty() {
            let last_grad = &gradients[gradients.len() - 1];
            self.all_reduce_gradient(last_grad)?;
        }
        
        Ok(())
    }
    
    fn execute_overlapped_ops(&mut self) -> TorshResult<()> {
        // Execute communication and computation concurrently
        let comm_handle = std::thread::spawn(move || {
            // Execute communication operations
        });
        
        let comp_handle = std::thread::spawn(move || {
            // Execute computation operations
        });
        
        comm_handle.join().map_err(|_| {
            TorshDistributedError::communication_error("comm_thread", "Thread join failed")
        })??;
        
        comp_handle.join().map_err(|_| {
            TorshDistributedError::communication_error("comp_thread", "Thread join failed")
        })??;
        
        Ok(())
    }
    
    fn all_reduce_gradient(&self, gradient: &Tensor) -> TorshResult<()> {
        all_reduce(gradient, ReduceOp::Sum)
    }
}
```

## Memory Management

### Smart Memory Allocation Strategies

```rust
use torsh_distributed::*;

struct MemoryManager {
    memory_pool: MemoryPool,
    offload_manager: OffloadManager,
    compression_enabled: bool,
}

impl MemoryManager {
    fn new(total_memory_gb: f32) -> Self {
        let pool_size = (total_memory_gb * 0.8 * 1024.0 * 1024.0 * 1024.0) as usize;
        
        Self {
            memory_pool: MemoryPool::new(pool_size),
            offload_manager: OffloadManager::new(),
            compression_enabled: true,
        }
    }
    
    fn allocate_model_memory(
        &mut self,
        model: &Model,
        sharding_config: &ShardingConfig,
    ) -> TorshResult<MemoryLayout> {
        let mut layout = MemoryLayout::new();
        
        for (layer_id, layer) in model.layers().iter().enumerate() {
            let layer_memory = self.calculate_layer_memory(layer)?;
            
            match sharding_config.get_strategy(layer_id) {
                ShardingStrategy::FullShard => {
                    // Distribute parameters across all devices
                    let shard_size = layer_memory / get_world_size() as usize;
                    let allocation = self.memory_pool.allocate(shard_size)?;
                    layout.add_layer(layer_id, allocation);
                }
                
                ShardingStrategy::NoShard => {
                    // Keep full parameters on each device
                    let allocation = self.memory_pool.allocate(layer_memory)?;
                    layout.add_layer(layer_id, allocation);
                }
                
                ShardingStrategy::OffloadCpu => {
                    // Offload to CPU memory
                    let cpu_allocation = self.offload_manager.allocate_cpu(layer_memory)?;
                    layout.add_cpu_layer(layer_id, cpu_allocation);
                }
            }
        }
        
        Ok(layout)
    }
    
    fn manage_activation_memory(
        &mut self,
        forward_pass_layers: &[LayerInfo],
    ) -> TorshResult<ActivationPlan> {
        let mut plan = ActivationPlan::new();
        
        for (i, layer_info) in forward_pass_layers.iter().enumerate() {
            let activation_size = layer_info.output_size;
            
            // Decide whether to checkpoint this activation
            let should_checkpoint = self.should_checkpoint_activation(
                i,
                activation_size,
                forward_pass_layers.len(),
            );
            
            if should_checkpoint {
                plan.add_checkpoint(i, activation_size);
            } else {
                // Allocate memory for storing activation
                let allocation = self.memory_pool.allocate(activation_size)?;
                plan.add_stored_activation(i, allocation);
            }
        }
        
        Ok(plan)
    }
    
    fn should_checkpoint_activation(
        &self,
        layer_index: usize,
        activation_size: usize,
        total_layers: usize,
    ) -> bool {
        // Checkpoint every 4th layer, or large activations
        layer_index % 4 == 0 || activation_size > 100 * 1024 * 1024 // 100MB
    }
    
    fn optimize_memory_layout(&mut self, model: &mut Model) -> TorshResult<()> {
        // Analyze memory access patterns
        let access_patterns = self.analyze_memory_access(model)?;
        
        // Reorganize memory layout for better cache locality
        for pattern in access_patterns {
            if pattern.should_colocate() {
                self.memory_pool.colocate_tensors(&pattern.tensor_ids)?;
            }
        }
        
        // Enable memory compression for inactive tensors
        if self.compression_enabled {
            self.compress_inactive_tensors(model)?;
        }
        
        Ok(())
    }
    
    fn analyze_memory_access(&self, model: &Model) -> TorshResult<Vec<AccessPattern>> {
        // Analyze which tensors are accessed together
        // This is a simplified example
        Ok(Vec::new())
    }
    
    fn compress_inactive_tensors(&mut self, model: &mut Model) -> TorshResult<()> {
        for layer in model.layers_mut() {
            if !layer.is_active() {
                layer.compress_parameters()?;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct MemoryLayout {
    layer_allocations: std::collections::HashMap<usize, MemoryAllocation>,
    cpu_allocations: std::collections::HashMap<usize, CpuAllocation>,
}

#[derive(Debug)]
struct ActivationPlan {
    checkpoints: Vec<(usize, usize)>,  // (layer_index, size)
    stored_activations: std::collections::HashMap<usize, MemoryAllocation>,
}

#[derive(Debug)]
struct AccessPattern {
    tensor_ids: Vec<usize>,
    access_frequency: f32,
    temporal_locality: f32,
}

impl AccessPattern {
    fn should_colocate(&self) -> bool {
        self.temporal_locality > 0.8 && self.tensor_ids.len() > 1
    }
}
```

### Gradient Compression Strategies

```rust
use torsh_distributed::*;

fn configure_gradient_compression(
    model_size: usize,
    network_bandwidth_gbps: f32,
    compression_tolerance: f32,
) -> TorshResult<CompressionConfig> {
    
    let compression_method = if network_bandwidth_gbps < 10.0 {
        // Low bandwidth: use aggressive compression
        CompressionMethod::TopK { k: 0.001 }  // 0.1% sparsity
    } else if network_bandwidth_gbps < 25.0 {
        // Medium bandwidth: moderate compression
        CompressionMethod::TopK { k: 0.01 }   // 1% sparsity
    } else {
        // High bandwidth: light compression or none
        if compression_tolerance > 0.1 {
            CompressionMethod::Quantization { bits: 8 }
        } else {
            CompressionMethod::None
        }
    };
    
    Ok(CompressionConfig {
        method: compression_method,
        error_feedback: true,
        compression_period: calculate_compression_period(model_size, network_bandwidth_gbps),
        memory_optimization: true,
    })
}

fn calculate_compression_period(model_size: usize, bandwidth_gbps: f32) -> u32 {
    // Compress every N steps based on model size and bandwidth
    let base_period = if model_size > 10_000_000_000 { 1 } else { 4 };
    let bandwidth_factor = (10.0 / bandwidth_gbps).max(1.0) as u32;
    base_period * bandwidth_factor
}

fn adaptive_compression_scheduler(
    compressor: &mut GradientCompressor,
    training_step: u32,
    loss_trend: &[f32],
) -> TorshResult<()> {
    
    // Analyze training stability
    let stability_metric = calculate_training_stability(loss_trend);
    
    // Adjust compression aggressiveness based on stability
    let new_config = match stability_metric {
        stability if stability > 0.9 => {
            // Training is stable, can use more aggressive compression
            CompressionConfig {
                method: CompressionMethod::TopK { k: 0.001 },
                error_feedback: true,
                compression_period: 1,
                memory_optimization: true,
            }
        }
        
        stability if stability > 0.7 => {
            // Moderate stability
            CompressionConfig {
                method: CompressionMethod::TopK { k: 0.01 },
                error_feedback: true,
                compression_period: 2,
                memory_optimization: true,
            }
        }
        
        _ => {
            // Training is unstable, reduce compression
            CompressionConfig {
                method: CompressionMethod::Quantization { bits: 16 },
                error_feedback: true,
                compression_period: 4,
                memory_optimization: false,
            }
        }
    };
    
    compressor.update_config(new_config)?;
    Ok(())
}

fn calculate_training_stability(loss_values: &[f32]) -> f32 {
    if loss_values.len() < 10 {
        return 1.0; // Assume stable if not enough data
    }
    
    // Calculate coefficient of variation (std/mean) for recent losses
    let recent_losses = &loss_values[loss_values.len() - 10..];
    let mean = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
    let variance = recent_losses.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / recent_losses.len() as f32;
    let std_dev = variance.sqrt();
    
    let cv = std_dev / mean.abs();
    
    // Convert to stability metric (1.0 = very stable, 0.0 = very unstable)
    (1.0 - cv.min(1.0)).max(0.0)
}
```

## Communication Optimization

### Bandwidth-Aware Communication Scheduling

```rust
use torsh_distributed::*;

struct BandwidthAwareCommunicationScheduler {
    available_bandwidth: f32,  // GB/s
    communication_queue: PriorityQueue<CommunicationTask>,
    bandwidth_monitor: BandwidthMonitor,
}

impl BandwidthAwareCommunicationScheduler {
    fn new(initial_bandwidth: f32) -> Self {
        Self {
            available_bandwidth: initial_bandwidth,
            communication_queue: PriorityQueue::new(),
            bandwidth_monitor: BandwidthMonitor::new(),
        }
    }
    
    fn schedule_communication(
        &mut self,
        operation: CommunicationOp,
        priority: Priority,
        size_bytes: usize,
    ) -> TorshResult<TaskId> {
        
        // Estimate communication time
        let estimated_time = size_bytes as f32 / (self.available_bandwidth * 1e9);
        
        // Create task with priority adjustment based on size and urgency
        let adjusted_priority = self.calculate_adjusted_priority(
            priority,
            size_bytes,
            estimated_time,
        );
        
        let task = CommunicationTask {
            id: TaskId::new(),
            operation,
            priority: adjusted_priority,
            size_bytes,
            estimated_time_ms: (estimated_time * 1000.0) as u64,
            created_at: std::time::Instant::now(),
        };
        
        let task_id = task.id;
        self.communication_queue.push(task);
        
        Ok(task_id)
    }
    
    fn execute_scheduled_operations(&mut self) -> TorshResult<Vec<TaskResult>> {
        let mut results = Vec::new();
        let mut current_bandwidth = self.available_bandwidth;
        
        // Execute tasks in priority order, considering bandwidth constraints
        while let Some(task) = self.communication_queue.pop() {
            // Check if we have enough bandwidth for this task
            let required_bandwidth = task.size_bytes as f32 / (task.estimated_time_ms as f32 / 1000.0);
            
            if required_bandwidth <= current_bandwidth {
                // Execute the task
                let start_time = std::time::Instant::now();
                let result = self.execute_task(&task)?;
                let actual_time = start_time.elapsed();
                
                // Update bandwidth estimation
                self.bandwidth_monitor.record_transfer(
                    task.size_bytes,
                    actual_time,
                );
                
                current_bandwidth = self.bandwidth_monitor.get_current_bandwidth();
                self.available_bandwidth = current_bandwidth;
                
                results.push(TaskResult {
                    task_id: task.id,
                    success: result.is_ok(),
                    actual_time_ms: actual_time.as_millis() as u64,
                    estimated_time_ms: task.estimated_time_ms,
                });
            } else {
                // Re-queue task for later
                self.communication_queue.push(task);
                break; // Try again later when bandwidth is available
            }
        }
        
        Ok(results)
    }
    
    fn calculate_adjusted_priority(
        &self,
        base_priority: Priority,
        size_bytes: usize,
        estimated_time: f32,
    ) -> Priority {
        let size_factor = if size_bytes > 100_000_000 { -10 } else { 0 }; // Large ops get lower priority
        let urgency_factor = if estimated_time > 1.0 { -5 } else { 5 }; // Fast ops get higher priority
        
        let adjusted = base_priority as i32 + size_factor + urgency_factor;
        Priority::from_i32(adjusted.clamp(0, 100))
    }
    
    fn execute_task(&self, task: &CommunicationTask) -> TorshResult<()> {
        match &task.operation {
            CommunicationOp::AllReduce { tensor, reduce_op } => {
                all_reduce(tensor, *reduce_op)
            }
            CommunicationOp::Broadcast { tensor, root } => {
                broadcast(tensor, *root)
            }
            CommunicationOp::AllGather { tensors } => {
                all_gather(tensors)
            }
            // ... other operations
        }
    }
}

struct BandwidthMonitor {
    recent_transfers: Vec<TransferRecord>,
    window_size: usize,
}

impl BandwidthMonitor {
    fn new() -> Self {
        Self {
            recent_transfers: Vec::new(),
            window_size: 10,
        }
    }
    
    fn record_transfer(&mut self, bytes: usize, duration: std::time::Duration) {
        let record = TransferRecord {
            bytes,
            duration,
            timestamp: std::time::Instant::now(),
        };
        
        self.recent_transfers.push(record);
        
        // Keep only recent transfers
        if self.recent_transfers.len() > self.window_size {
            self.recent_transfers.remove(0);
        }
    }
    
    fn get_current_bandwidth(&self) -> f32 {
        if self.recent_transfers.is_empty() {
            return 1.0; // Default 1 GB/s
        }
        
        let total_bytes: usize = self.recent_transfers.iter().map(|r| r.bytes).sum();
        let total_time: f32 = self.recent_transfers
            .iter()
            .map(|r| r.duration.as_secs_f32())
            .sum();
        
        if total_time > 0.0 {
            (total_bytes as f32) / total_time / 1e9  // Convert to GB/s
        } else {
            1.0
        }
    }
}

#[derive(Debug)]
struct TransferRecord {
    bytes: usize,
    duration: std::time::Duration,
    timestamp: std::time::Instant,
}

#[derive(Debug)]
struct TaskResult {
    task_id: TaskId,
    success: bool,
    actual_time_ms: u64,
    estimated_time_ms: u64,
}
```

### Network Topology-Aware Communication

```rust
use torsh_distributed::*;

fn optimize_communication_for_topology(
    topology: &NetworkTopology,
    world_size: u32,
) -> TorshResult<CommunicationStrategy> {
    
    match topology {
        NetworkTopology::SingleNode { num_gpus } => {
            // Use NVLink for intra-node communication
            Ok(CommunicationStrategy::NVLink {
                use_p2p: true,
                enable_compression: false,
                buffer_size_mb: 256,
            })
        }
        
        NetworkTopology::MultiNode { 
            nodes, 
            gpus_per_node,
            interconnect 
        } => {
            match interconnect {
                Interconnect::InfiniBand => {
                    Ok(CommunicationStrategy::Hierarchical {
                        intra_node: Box::new(CommunicationStrategy::NVLink {
                            use_p2p: true,
                            enable_compression: false,
                            buffer_size_mb: 256,
                        }),
                        inter_node: Box::new(CommunicationStrategy::InfiniBand {
                            use_rdma: true,
                            enable_compression: false,
                            ring_size: *nodes,
                        }),
                    })
                }
                
                Interconnect::Ethernet => {
                    Ok(CommunicationStrategy::Hierarchical {
                        intra_node: Box::new(CommunicationStrategy::NVLink {
                            use_p2p: true,
                            enable_compression: false,
                            buffer_size_mb: 128,
                        }),
                        inter_node: Box::new(CommunicationStrategy::Ethernet {
                            enable_compression: true,  // Compress for ethernet
                            compression_method: CompressionMethod::TopK { k: 0.01 },
                            tcp_window_size: 65536,
                        }),
                    })
                }
            }
        }
        
        NetworkTopology::Cloud { provider, region } => {
            // Optimize for cloud environment
            Ok(CommunicationStrategy::Cloud {
                provider: *provider,
                enable_compression: true,
                adaptive_batching: true,
                redundant_paths: false,
            })
        }
    }
}

#[derive(Debug)]
enum NetworkTopology {
    SingleNode { num_gpus: u32 },
    MultiNode { 
        nodes: u32, 
        gpus_per_node: u32,
        interconnect: Interconnect,
    },
    Cloud { provider: CloudProvider, region: String },
}

#[derive(Debug)]
enum Interconnect {
    InfiniBand,
    Ethernet,
    Custom(String),
}

#[derive(Debug, Clone, Copy)]
enum CloudProvider {
    AWS,
    Azure,
    GCP,
    Custom,
}

#[derive(Debug)]
enum CommunicationStrategy {
    NVLink {
        use_p2p: bool,
        enable_compression: bool,
        buffer_size_mb: u32,
    },
    InfiniBand {
        use_rdma: bool,
        enable_compression: bool,
        ring_size: u32,
    },
    Ethernet {
        enable_compression: bool,
        compression_method: CompressionMethod,
        tcp_window_size: u32,
    },
    Hierarchical {
        intra_node: Box<CommunicationStrategy>,
        inter_node: Box<CommunicationStrategy>,
    },
    Cloud {
        provider: CloudProvider,
        enable_compression: bool,
        adaptive_batching: bool,
        redundant_paths: bool,
    },
}
```

## Fault Tolerance and Reliability

### Comprehensive Checkpointing Strategy

```rust
use torsh_distributed::*;

struct CheckpointManager {
    config: CheckpointConfig,
    checkpoint_queue: Vec<CheckpointTask>,
    last_checkpoint_time: std::time::Instant,
    health_monitor: HealthMonitor,
}

impl CheckpointManager {
    fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            checkpoint_queue: Vec::new(),
            last_checkpoint_time: std::time::Instant::now(),
            health_monitor: HealthMonitor::new(),
        }
    }
    
    fn should_checkpoint(
        &self,
        current_step: u32,
        loss_value: f32,
        system_health: HealthStatus,
    ) -> bool {
        
        // Regular interval checkpointing
        let interval_checkpoint = current_step % self.config.checkpoint_interval == 0;
        
        // Performance-based checkpointing (best model so far)
        let performance_checkpoint = self.is_best_performance(loss_value);
        
        // Health-based checkpointing (when system becomes unhealthy)
        let health_checkpoint = matches!(system_health, HealthStatus::Degraded | HealthStatus::Critical);
        
        // Time-based checkpointing (maximum time since last checkpoint)
        let time_checkpoint = self.last_checkpoint_time.elapsed() 
            > std::time::Duration::from_secs(self.config.max_time_between_checkpoints_sec);
        
        interval_checkpoint || performance_checkpoint || health_checkpoint || time_checkpoint
    }
    
    fn create_checkpoint(
        &mut self,
        model: &Model,
        optimizer: &Optimizer,
        training_state: &TrainingState,
        metadata: CheckpointMetadata,
    ) -> TorshResult<CheckpointId> {
        
        let checkpoint_id = CheckpointId::new();
        
        // Create checkpoint asynchronously to avoid blocking training
        let task = CheckpointTask {
            id: checkpoint_id,
            model_state: model.state_dict()?,
            optimizer_state: optimizer.state_dict()?,
            training_state: training_state.clone(),
            metadata,
            created_at: std::time::Instant::now(),
        };
        
        self.checkpoint_queue.push(task);
        
        // Trigger async checkpoint saving
        self.process_checkpoint_queue()?;
        
        self.last_checkpoint_time = std::time::Instant::now();
        Ok(checkpoint_id)
    }
    
    fn process_checkpoint_queue(&mut self) -> TorshResult<()> {
        // Process checkpoints asynchronously
        while let Some(task) = self.checkpoint_queue.pop() {
            self.save_checkpoint_async(task)?;
        }
        Ok(())
    }
    
    fn save_checkpoint_async(&self, task: CheckpointTask) -> TorshResult<()> {
        let config = self.config.clone();
        
        std::thread::spawn(move || {
            let result = Self::save_checkpoint_to_storage(&task, &config);
            if let Err(e) = result {
                eprintln!("Checkpoint save failed: {}", e);
            }
        });
        
        Ok(())
    }
    
    fn save_checkpoint_to_storage(
        task: &CheckpointTask,
        config: &CheckpointConfig,
    ) -> TorshResult<()> {
        
        let checkpoint_path = format!(
            "{}/checkpoint_{}.pt",
            config.checkpoint_dir,
            task.id.to_string()
        );
        
        // Save with compression if enabled
        if config.enable_compression {
            let compressed_data = compress_checkpoint_data(task)?;
            std::fs::write(&checkpoint_path, compressed_data)?;
        } else {
            let serialized_data = serialize_checkpoint_data(task)?;
            std::fs::write(&checkpoint_path, serialized_data)?;
        }
        
        // Verify checkpoint integrity
        Self::verify_checkpoint_integrity(&checkpoint_path)?;
        
        // Clean up old checkpoints
        Self::cleanup_old_checkpoints(config)?;
        
        Ok(())
    }
    
    fn restore_from_checkpoint(
        &self,
        checkpoint_id: CheckpointId,
        model: &mut Model,
        optimizer: &mut Optimizer,
    ) -> TorshResult<TrainingState> {
        
        let checkpoint_path = format!(
            "{}/checkpoint_{}.pt",
            self.config.checkpoint_dir,
            checkpoint_id.to_string()
        );
        
        // Load and decompress if needed
        let checkpoint_data = if self.config.enable_compression {
            let compressed_data = std::fs::read(&checkpoint_path)?;
            decompress_checkpoint_data(&compressed_data)?
        } else {
            std::fs::read(&checkpoint_path)?
        };
        
        let task = deserialize_checkpoint_data(&checkpoint_data)?;
        
        // Restore model and optimizer states
        model.load_state_dict(&task.model_state)?;
        optimizer.load_state_dict(&task.optimizer_state)?;
        
        Ok(task.training_state)
    }
    
    fn is_best_performance(&self, current_loss: f32) -> bool {
        // Implement logic to determine if this is the best performance so far
        // This would typically involve comparing with historical best loss
        false // Simplified for this example
    }
    
    fn verify_checkpoint_integrity(checkpoint_path: &str) -> TorshResult<()> {
        // Verify that the checkpoint can be loaded successfully
        let data = std::fs::read(checkpoint_path)?;
        if data.is_empty() {
            return Err(TorshDistributedError::checkpoint_error(
                "verification",
                "Checkpoint file is empty"
            ));
        }
        
        // Additional integrity checks (checksum, magic numbers, etc.)
        Ok(())
    }
    
    fn cleanup_old_checkpoints(config: &CheckpointConfig) -> TorshResult<()> {
        // Keep only the most recent N checkpoints
        let mut checkpoint_files = std::fs::read_dir(&config.checkpoint_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_name()
                    .to_string_lossy()
                    .starts_with("checkpoint_")
            })
            .collect::<Vec<_>>();
        
        // Sort by modification time
        checkpoint_files.sort_by_key(|entry| {
            entry.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        
        // Remove old checkpoints
        let num_to_keep = config.max_checkpoints_to_keep;
        if checkpoint_files.len() > num_to_keep {
            for file_to_remove in checkpoint_files.iter().take(checkpoint_files.len() - num_to_keep) {
                std::fs::remove_file(file_to_remove.path())?;
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct CheckpointConfig {
    checkpoint_dir: String,
    checkpoint_interval: u32,
    max_checkpoints_to_keep: usize,
    enable_compression: bool,
    max_time_between_checkpoints_sec: u64,
    verify_integrity: bool,
}

#[derive(Debug)]
struct CheckpointTask {
    id: CheckpointId,
    model_state: ModelStateDict,
    optimizer_state: OptimizerStateDict,
    training_state: TrainingState,
    metadata: CheckpointMetadata,
    created_at: std::time::Instant,
}

#[derive(Debug, Clone)]
struct CheckpointMetadata {
    step: u32,
    epoch: u32,
    loss: f32,
    learning_rate: f32,
    timestamp: String,
    git_commit: Option<String>,
    config_hash: String,
}

#[derive(Debug, Clone)]
struct TrainingState {
    current_step: u32,
    current_epoch: u32,
    best_loss: f32,
    random_state: RandomState,
}

// Helper functions
fn compress_checkpoint_data(task: &CheckpointTask) -> TorshResult<Vec<u8>> {
    // Implement compression logic
    Ok(Vec::new())
}

fn decompress_checkpoint_data(data: &[u8]) -> TorshResult<Vec<u8>> {
    // Implement decompression logic
    Ok(Vec::new())
}

fn serialize_checkpoint_data(task: &CheckpointTask) -> TorshResult<Vec<u8>> {
    // Implement serialization logic
    Ok(Vec::new())
}

fn deserialize_checkpoint_data(data: &[u8]) -> TorshResult<CheckpointTask> {
    // Implement deserialization logic
    todo!()
}
```

### Elastic Training and Auto-Recovery

```rust
use torsh_distributed::*;

struct ElasticTrainingManager {
    config: ElasticConfig,
    active_workers: Vec<WorkerInfo>,
    failed_workers: Vec<WorkerInfo>,
    scaling_strategy: ScalingStrategy,
    recovery_manager: RecoveryManager,
}

impl ElasticTrainingManager {
    fn new(config: ElasticConfig) -> Self {
        Self {
            config,
            active_workers: Vec::new(),
            failed_workers: Vec::new(),
            scaling_strategy: ScalingStrategy::Conservative,
            recovery_manager: RecoveryManager::new(),
        }
    }
    
    fn monitor_worker_health(&mut self) -> TorshResult<Vec<HealthEvent>> {
        let mut events = Vec::new();
        
        for worker in &mut self.active_workers {
            let health = self.check_worker_health(worker)?;
            
            match health {
                HealthStatus::Healthy => {
                    worker.consecutive_failures = 0;
                }
                
                HealthStatus::Degraded => {
                    worker.consecutive_failures += 1;
                    events.push(HealthEvent::WorkerDegraded {
                        worker_id: worker.id,
                        reason: "Performance degradation detected".to_string(),
                    });
                    
                    if worker.consecutive_failures >= 3 {
                        events.push(HealthEvent::WorkerFailed {
                            worker_id: worker.id,
                            reason: "Consecutive failures exceeded threshold".to_string(),
                        });
                    }
                }
                
                HealthStatus::Critical | HealthStatus::Failed => {
                    events.push(HealthEvent::WorkerFailed {
                        worker_id: worker.id,
                        reason: "Worker unresponsive or crashed".to_string(),
                    });
                }
            }
        }
        
        Ok(events)
    }
    
    fn handle_health_events(&mut self, events: Vec<HealthEvent>) -> TorshResult<()> {
        for event in events {
            match event {
                HealthEvent::WorkerFailed { worker_id, reason } => {
                    self.handle_worker_failure(worker_id, &reason)?;
                }
                
                HealthEvent::WorkerDegraded { worker_id, reason } => {
                    self.handle_worker_degradation(worker_id, &reason)?;
                }
                
                HealthEvent::WorkerRecovered { worker_id } => {
                    self.handle_worker_recovery(worker_id)?;
                }
                
                HealthEvent::ScalingTriggered { direction, reason } => {
                    self.handle_scaling_event(direction, &reason)?;
                }
            }
        }
        Ok(())
    }
    
    fn handle_worker_failure(&mut self, worker_id: WorkerId, reason: &str) -> TorshResult<()> {
        // Move worker from active to failed list
        if let Some(pos) = self.active_workers.iter().position(|w| w.id == worker_id) {
            let mut failed_worker = self.active_workers.remove(pos);
            failed_worker.failure_time = Some(std::time::Instant::now());
            failed_worker.failure_reason = Some(reason.to_string());
            self.failed_workers.push(failed_worker);
        }
        
        // Check if we need to scale up to replace the failed worker
        let current_workers = self.active_workers.len() as u32;
        if current_workers < self.config.min_workers {
            self.trigger_scale_up("Worker failure below minimum threshold")?;
        }
        
        // Redistribute work from failed worker
        self.redistribute_work_from_failed_worker(worker_id)?;
        
        // Update process group
        self.update_process_group_after_failure(worker_id)?;
        
        Ok(())
    }
    
    fn trigger_scale_up(&mut self, reason: &str) -> TorshResult<()> {
        let target_workers = (self.active_workers.len() as u32 + 1)
            .min(self.config.max_workers);
        
        tracing::info!("Scaling up to {} workers: {}", target_workers, reason);
        
        // Request new worker from resource manager
        let new_worker = self.request_new_worker()?;
        
        // Initialize new worker
        self.initialize_new_worker(&new_worker)?;
        
        // Add to active workers
        self.active_workers.push(new_worker);
        
        Ok(())
    }
    
    fn trigger_scale_down(&mut self, reason: &str) -> TorshResult<()> {
        if self.active_workers.len() as u32 <= self.config.min_workers {
            return Ok(()); // Don't scale below minimum
        }
        
        tracing::info!("Scaling down: {}", reason);
        
        // Select worker to remove (least loaded)
        let worker_to_remove = self.select_worker_for_removal()?;
        
        // Gracefully shutdown worker
        self.gracefully_shutdown_worker(worker_to_remove)?;
        
        // Remove from active workers
        self.active_workers.retain(|w| w.id != worker_to_remove);
        
        Ok(())
    }
    
    fn auto_scale_based_on_load(&mut self) -> TorshResult<()> {
        let current_load = self.calculate_average_load()?;
        let current_workers = self.active_workers.len() as u32;
        
        match self.scaling_strategy {
            ScalingStrategy::Aggressive => {
                if current_load > 0.8 && current_workers < self.config.max_workers {
                    self.trigger_scale_up("High load detected")?;
                } else if current_load < 0.3 && current_workers > self.config.min_workers {
                    self.trigger_scale_down("Low load detected")?;
                }
            }
            
            ScalingStrategy::Conservative => {
                if current_load > 0.9 && current_workers < self.config.max_workers {
                    self.trigger_scale_up("Very high load detected")?;
                } else if current_load < 0.2 && current_workers > self.config.min_workers {
                    self.trigger_scale_down("Very low load detected")?;
                }
            }
            
            ScalingStrategy::Reactive => {
                // Only scale in response to failures or explicit requests
            }
        }
        
        Ok(())
    }
    
    fn redistribute_work_from_failed_worker(&mut self, failed_worker_id: WorkerId) -> TorshResult<()> {
        // Get unfinished work from failed worker
        let unfinished_work = self.recovery_manager.get_unfinished_work(failed_worker_id)?;
        
        // Redistribute among active workers
        let num_active = self.active_workers.len();
        if num_active == 0 {
            return Err(TorshDistributedError::process_failure(
                failed_worker_id.0,
                "work_redistribution",
                "No active workers available for redistribution"
            ));
        }
        
        let work_per_worker = unfinished_work.len() / num_active;
        let remainder = unfinished_work.len() % num_active;
        
        let mut work_index = 0;
        for (i, worker) in self.active_workers.iter_mut().enumerate() {
            let work_count = work_per_worker + if i < remainder { 1 } else { 0 };
            let worker_work = &unfinished_work[work_index..work_index + work_count];
            
            for work_item in worker_work {
                worker.pending_work.push(work_item.clone());
            }
            
            work_index += work_count;
        }
        
        Ok(())
    }
    
    // Helper methods
    fn check_worker_health(&self, worker: &WorkerInfo) -> TorshResult<HealthStatus> {
        // Implement health check logic
        Ok(HealthStatus::Healthy)
    }
    
    fn request_new_worker(&self) -> TorshResult<WorkerInfo> {
        // Implement worker provisioning logic
        Ok(WorkerInfo::new())
    }
    
    fn initialize_new_worker(&self, worker: &WorkerInfo) -> TorshResult<()> {
        // Implement worker initialization logic
        Ok(())
    }
    
    fn select_worker_for_removal(&self) -> TorshResult<WorkerId> {
        // Select least loaded worker
        self.active_workers
            .iter()
            .min_by_key(|w| w.current_load)
            .map(|w| w.id)
            .ok_or_else(|| TorshDistributedError::configuration_error("No workers available for removal"))
    }
    
    fn gracefully_shutdown_worker(&self, worker_id: WorkerId) -> TorshResult<()> {
        // Implement graceful shutdown logic
        Ok(())
    }
    
    fn calculate_average_load(&self) -> TorshResult<f32> {
        if self.active_workers.is_empty() {
            return Ok(0.0);
        }
        
        let total_load: f32 = self.active_workers.iter().map(|w| w.current_load).sum();
        Ok(total_load / self.active_workers.len() as f32)
    }
    
    fn update_process_group_after_failure(&self, failed_worker_id: WorkerId) -> TorshResult<()> {
        // Update process group configuration after worker failure
        Ok(())
    }
    
    fn handle_worker_degradation(&mut self, worker_id: WorkerId, reason: &str) -> TorshResult<()> {
        // Handle worker performance degradation
        tracing::warn!("Worker {} degraded: {}", worker_id.0, reason);
        Ok(())
    }
    
    fn handle_worker_recovery(&mut self, worker_id: WorkerId) -> TorshResult<()> {
        // Handle worker recovery
        tracing::info!("Worker {} recovered", worker_id.0);
        Ok(())
    }
    
    fn handle_scaling_event(&mut self, direction: ScalingDirection, reason: &str) -> TorshResult<()> {
        match direction {
            ScalingDirection::Up => self.trigger_scale_up(reason),
            ScalingDirection::Down => self.trigger_scale_down(reason),
        }
    }
}

#[derive(Debug)]
struct WorkerInfo {
    id: WorkerId,
    rank: u32,
    address: String,
    current_load: f32,
    consecutive_failures: u32,
    failure_time: Option<std::time::Instant>,
    failure_reason: Option<String>,
    pending_work: Vec<WorkItem>,
}

impl WorkerInfo {
    fn new() -> Self {
        Self {
            id: WorkerId::new(),
            rank: 0,
            address: String::new(),
            current_load: 0.0,
            consecutive_failures: 0,
            failure_time: None,
            failure_reason: None,
            pending_work: Vec::new(),
        }
    }
}

#[derive(Debug)]
enum HealthEvent {
    WorkerFailed { worker_id: WorkerId, reason: String },
    WorkerDegraded { worker_id: WorkerId, reason: String },
    WorkerRecovered { worker_id: WorkerId },
    ScalingTriggered { direction: ScalingDirection, reason: String },
}

#[derive(Debug)]
enum ScalingStrategy {
    Aggressive,   // Scale quickly based on load
    Conservative, // Scale only when necessary
    Reactive,     // Scale only in response to failures
}

#[derive(Debug)]
enum ScalingDirection {
    Up,
    Down,
}

#[derive(Debug)]
struct WorkItem {
    id: String,
    data: Vec<u8>,
    priority: u32,
}

struct RecoveryManager {
    work_tracker: std::collections::HashMap<WorkerId, Vec<WorkItem>>,
}

impl RecoveryManager {
    fn new() -> Self {
        Self {
            work_tracker: std::collections::HashMap::new(),
        }
    }
    
    fn get_unfinished_work(&self, worker_id: WorkerId) -> TorshResult<Vec<WorkItem>> {
        Ok(self.work_tracker.get(&worker_id).cloned().unwrap_or_default())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct WorkerId(u32);

impl WorkerId {
    fn new() -> Self {
        Self(rand::random())
    }
}
```

This comprehensive best practices guide covers the most important aspects of distributed training with ToRSh. Following these patterns and recommendations will help you build robust, scalable, and efficient distributed training systems.

Key takeaways:
1. **Choose the right parallelism strategy** based on model size and available resources
2. **Optimize memory usage** through smart allocation and compression strategies  
3. **Design communication patterns** that match your network topology
4. **Implement comprehensive fault tolerance** with checkpointing and elastic scaling
5. **Monitor and debug** your distributed training jobs proactively
6. **Follow framework-specific best practices** when integrating with other systems
7. **Plan for production deployment** with proper monitoring and automation

Remember that distributed training is complex, and these practices should be adapted to your specific use case, hardware configuration, and performance requirements.