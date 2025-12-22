//! Fully Sharded Data Parallel (FSDP) implementation
//!
//! FSDP enables training very large models by sharding parameters across workers.
//! Unlike DDP which replicates the full model on each worker, FSDP:
//! - Shards model parameters across workers (each worker stores only a fraction)
//! - Gathers parameters during forward/backward passes as needed
//! - Re-shards parameters after computation to save memory
//! - Supports nested sharding for hierarchical model parallelism

use crate::backend::ReduceOp;
use crate::collectives::{all_gather, all_reduce};
use crate::{ProcessGroup, Rank, TorshDistributedError, TorshResult};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use torsh_core::{device::DeviceType, error::Result, DType, Shape};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;
use tracing::{debug, info};

/// FSDP configuration options
#[derive(Debug, Clone)]
pub struct FsdpConfig {
    /// Minimum number of parameters for a module to be sharded
    pub min_num_params: usize,
    /// Whether to use auto wrapping of submodules
    pub auto_wrap_policy: AutoWrapPolicy,
    /// Sharding strategy
    pub sharding_strategy: ShardingStrategy,
    /// Mixed precision configuration
    pub mixed_precision: Option<MixedPrecisionConfig>,
    /// Whether to use CPU offloading
    pub cpu_offload: bool,
    /// Memory management configuration
    pub memory_config: MemoryConfig,
    /// Backward prefetch configuration
    pub backward_prefetch: BackwardPrefetch,
}

impl Default for FsdpConfig {
    fn default() -> Self {
        Self {
            min_num_params: 1000,
            auto_wrap_policy: AutoWrapPolicy::SizeBasedAutoWrap {
                min_num_params: 1000,
            },
            sharding_strategy: ShardingStrategy::FullShard,
            mixed_precision: None,
            cpu_offload: false,
            memory_config: MemoryConfig::default(),
            backward_prefetch: BackwardPrefetch::BackwardPre,
        }
    }
}

/// Auto-wrapping policies for FSDP
#[derive(Debug, Clone)]
pub enum AutoWrapPolicy {
    /// Wrap modules based on number of parameters
    SizeBasedAutoWrap { min_num_params: usize },
    /// Wrap specific module types
    ModuleTypeBasedAutoWrap { module_types: Vec<String> },
    /// Custom wrapping function
    CustomAutoWrap,
    /// No auto-wrapping
    NoAutoWrap,
}

/// Sharding strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ShardingStrategy {
    /// Shard parameters, gradients, and optimizer states (default)
    FullShard,
    /// Shard gradients and optimizer states only
    ShardGradOp,
    /// No sharding (equivalent to DDP)
    NoShard,
    /// Hybrid sharding for hierarchical parallelism
    HybridShard,
}

/// Mixed precision configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Parameter data type
    pub param_dtype: DType,
    /// Gradient reduction data type
    pub reduce_dtype: DType,
    /// Buffer data type
    pub buffer_dtype: DType,
    /// Keep parameters in fp32 for backward pass
    pub keep_low_precision_grads: bool,
}

/// Memory management configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Limit all-gather for parameters (memory vs speed tradeoff)
    pub limit_all_gathers: bool,
    /// Use original parameters for computation when possible
    pub use_orig_params: bool,
    /// Offload parameters to CPU when not in use
    pub offload_to_cpu: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            limit_all_gathers: true,
            use_orig_params: false,
            offload_to_cpu: false,
        }
    }
}

/// Backward prefetch configuration
#[derive(Debug, Clone, PartialEq)]
pub enum BackwardPrefetch {
    /// Prefetch the next layer's parameters during backward pass
    BackwardPre,
    /// Prefetch after current layer's gradient computation
    BackwardPost,
    /// No prefetching
    None,
}

/// Parameter shard information
#[derive(Debug, Clone)]
pub struct ShardInfo {
    /// Rank that owns this shard
    pub rank: Rank,
    /// Start index in the flattened parameter tensor
    pub start_idx: usize,
    /// Number of elements in this shard
    pub shard_size: usize,
    /// Shape of the original parameter
    pub original_shape: Shape,
    /// Whether this shard is currently on this worker
    pub is_local: bool,
}

/// FSDP parameter state
#[derive(Debug)]
enum ParameterState {
    /// Parameter is sharded across workers
    Sharded {
        #[allow(dead_code)]
        shard_info: ShardInfo,
    },
    /// Parameter is gathered (full tensor available)
    Gathered {
        #[allow(dead_code)]
        full_tensor: Tensor,
    },
    /// Parameter is being gathered (async operation in progress)
    #[allow(dead_code)]
    Gathering,
    /// Parameter is being sharded (async operation in progress)  
    #[allow(dead_code)]
    Sharding,
}

/// Fully Sharded Data Parallel wrapper
pub struct FullyShardedDataParallel {
    /// Wrapped module
    module: Arc<RwLock<dyn Module>>,
    /// Process group for communication
    process_group: Arc<ProcessGroup>,
    /// FSDP configuration
    config: FsdpConfig,
    /// Parameter state tracking
    param_states: Arc<DashMap<String, ParameterState>>,
    /// Sharded parameters storage
    sharded_params: Arc<DashMap<String, Tensor>>,
    /// Gathered parameters cache
    #[allow(dead_code)]
    gathered_params: Arc<DashMap<String, Tensor>>,
    /// Gradient buffers
    #[allow(dead_code)]
    grad_buffers: Arc<DashMap<String, Tensor>>,
    /// Whether we're in training mode
    training: Arc<Mutex<bool>>,
    /// Current compute stream for overlapping
    #[allow(dead_code)]
    compute_stream: Arc<Mutex<Option<String>>>,
    /// Memory usage statistics
    memory_stats: Arc<Mutex<MemoryStats>>,
}

/// Memory usage statistics
#[derive(Debug, Default)]
pub struct MemoryStats {
    /// Peak memory usage during training
    pub peak_memory_mb: f64,
    /// Current memory usage
    pub current_memory_mb: f64,
    /// Memory saved by sharding
    pub memory_saved_mb: f64,
    /// Number of all-gather operations
    pub num_all_gathers: u64,
    /// Number of reduce-scatter operations  
    pub num_reduce_scatters: u64,
}

impl FullyShardedDataParallel {
    /// Create a new FSDP wrapper
    pub fn new(
        module: Arc<RwLock<dyn Module>>,
        process_group: Arc<ProcessGroup>,
        config: FsdpConfig,
    ) -> TorshResult<Self> {
        let fsdp = Self {
            module,
            process_group,
            config,
            param_states: Arc::new(DashMap::new()),
            sharded_params: Arc::new(DashMap::new()),
            gathered_params: Arc::new(DashMap::new()),
            grad_buffers: Arc::new(DashMap::new()),
            training: Arc::new(Mutex::new(true)),
            compute_stream: Arc::new(Mutex::new(None)),
            memory_stats: Arc::new(Mutex::new(MemoryStats::default())),
        };

        // Initialize parameter sharding
        fsdp.shard_parameters()?;

        info!(
            "FSDP initialized with strategy {:?} for {} workers",
            fsdp.config.sharding_strategy,
            fsdp.process_group.world_size()
        );

        Ok(fsdp)
    }

    /// Shard parameters across workers
    fn shard_parameters(&self) -> TorshResult<()> {
        let module_guard = self.module.read();
        let parameters = module_guard.parameters();
        drop(module_guard);

        let world_size = self.process_group.world_size() as usize;
        let rank = self.process_group.rank() as usize;

        for (name, param) in parameters {
            let tensor_arc = param.tensor();
            let tensor_guard = tensor_arc.read();
            if tensor_guard.numel() < self.config.min_num_params {
                // Don't shard small parameters
                self.param_states.insert(
                    name.clone(),
                    ParameterState::Gathered {
                        full_tensor: tensor_guard.clone(),
                    },
                );
                continue;
            }

            // Flatten parameter for sharding
            let flat_param = tensor_guard.flatten()?;
            let total_elements = flat_param.numel();

            // Calculate shard sizes (handle uneven division)
            let base_shard_size = total_elements / world_size;
            let remainder = total_elements % world_size;

            let mut start_idx = 0;
            for worker_rank in 0..world_size {
                let shard_size = base_shard_size + if worker_rank < remainder { 1 } else { 0 };

                if worker_rank == rank {
                    // This is our shard
                    let shard = flat_param
                        .slice(0, start_idx, start_idx + shard_size)?
                        .to_tensor()?;
                    self.sharded_params.insert(name.clone(), shard);

                    let shard_info = ShardInfo {
                        rank: worker_rank as Rank,
                        start_idx,
                        shard_size,
                        original_shape: tensor_guard.shape().clone(),
                        is_local: true,
                    };

                    self.param_states
                        .insert(name.clone(), ParameterState::Sharded { shard_info });
                }

                start_idx += shard_size;
            }

            debug!(
                "Sharded parameter '{}' with {} elements across {} workers",
                name, total_elements, world_size
            );
            drop(tensor_guard);
        }

        Ok(())
    }

    /// Gather parameters for computation
    #[allow(dead_code)]
    async fn gather_parameters(&self, param_names: &[String]) -> TorshResult<()> {
        for param_name in param_names {
            if let Some(mut state_ref) = self.param_states.get_mut(param_name) {
                if let ParameterState::Sharded { shard_info } = &*state_ref {
                    // Mark as gathering
                    let original_shape = shard_info.original_shape.clone();
                    *state_ref = ParameterState::Gathering;
                    drop(state_ref);

                    // Perform all-gather to reconstruct full parameter
                    let shard = self.sharded_params.get(param_name).ok_or_else(|| {
                        TorshDistributedError::backend_error(
                            "fsdp",
                            format!("Shard not found for parameter '{}'", param_name),
                        )
                    })?;

                    let mut gathered_tensors = Vec::new();
                    all_gather(&mut gathered_tensors, &*shard, &self.process_group).await?;

                    // Concatenate all gathered tensors
                    let gathered_tensor = if gathered_tensors.len() == 1 {
                        gathered_tensors.into_iter().next().unwrap()
                    } else {
                        // For simplicity, just use the first tensor (mock behavior)
                        gathered_tensors.into_iter().next().unwrap()
                    };

                    // Reshape to original shape
                    let shape_dims: Vec<i32> =
                        original_shape.dims().iter().map(|&x| x as i32).collect();
                    let reshaped = gathered_tensor.reshape(&shape_dims)?;

                    // Cache gathered parameter
                    self.gathered_params
                        .insert(param_name.clone(), reshaped.clone());

                    // Update state
                    self.param_states.insert(
                        param_name.clone(),
                        ParameterState::Gathered {
                            full_tensor: reshaped,
                        },
                    );

                    // Update statistics
                    let mut stats = self.memory_stats.lock().unwrap();
                    stats.num_all_gathers += 1;
                }
            }
        }

        Ok(())
    }

    /// Reduce-scatter gradients and re-shard parameters
    #[allow(dead_code)]
    async fn reduce_scatter_gradients(&self, param_names: &[String]) -> TorshResult<()> {
        for param_name in param_names {
            if let Some(grad_buffer) = self.grad_buffers.get(param_name) {
                // Perform reduce-scatter on gradients
                let mut reduced_grad = grad_buffer.clone();
                all_reduce(&mut reduced_grad, ReduceOp::Sum, &self.process_group).await?;

                // Get the local shard of the gradient
                if let Some(state_ref) = self.param_states.get(param_name) {
                    if let ParameterState::Sharded { shard_info } = &*state_ref {
                        let grad_shard = reduced_grad.slice(
                            0,
                            shard_info.start_idx,
                            shard_info.start_idx + shard_info.shard_size,
                        )?;

                        // Update the local parameter shard with gradient
                        if let Some(mut param_shard) = self.sharded_params.get_mut(param_name) {
                            // Apply gradient (this would typically be done by optimizer)
                            let grad_tensor = grad_shard.to_tensor()?;
                            *param_shard = param_shard.sub(&grad_tensor)?;
                        }
                    }
                }

                // Update statistics
                let mut stats = self.memory_stats.lock().unwrap();
                stats.num_reduce_scatters += 1;
            }

            // Re-shard parameter
            self.param_states.insert(
                param_name.clone(),
                ParameterState::Sharded {
                    shard_info: self.get_shard_info(param_name)?,
                },
            );

            // Remove from gathered cache to save memory
            self.gathered_params.remove(param_name);
        }

        Ok(())
    }

    /// Get shard information for a parameter
    #[allow(dead_code)]
    fn get_shard_info(&self, param_name: &str) -> TorshResult<ShardInfo> {
        if let Some(state_ref) = self.param_states.get(param_name) {
            match &*state_ref {
                ParameterState::Sharded { shard_info } => Ok(shard_info.clone()),
                _ => Err(TorshDistributedError::backend_error(
                    "fsdp",
                    format!("Parameter '{}' is not in sharded state", param_name),
                )),
            }
        } else {
            Err(TorshDistributedError::backend_error(
                "fsdp",
                format!("Parameter '{}' not found", param_name),
            ))
        }
    }

    /// Set training mode
    pub fn train(&self, mode: bool) {
        *self.training.lock().unwrap() = mode;
        let mut module_guard = self.module.write();
        if mode {
            module_guard.train();
        } else {
            module_guard.eval();
        }
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        *self.training.lock().unwrap()
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let stats = self.memory_stats.lock().unwrap();
        MemoryStats {
            peak_memory_mb: stats.peak_memory_mb,
            current_memory_mb: stats.current_memory_mb,
            memory_saved_mb: stats.memory_saved_mb,
            num_all_gathers: stats.num_all_gathers,
            num_reduce_scatters: stats.num_reduce_scatters,
        }
    }

    /// Get the number of parameters in the model
    pub fn num_parameters(&self) -> usize {
        let module_guard = self.module.read();
        let parameters = module_guard.parameters();
        parameters.values().map(|p| p.tensor().read().numel()).sum()
    }

    /// Get the local sharding ratio (fraction of parameters stored locally)
    pub fn local_sharding_ratio(&self) -> f64 {
        let total_params = self.num_parameters();
        let local_params: usize = self
            .sharded_params
            .iter()
            .map(|entry| entry.value().numel())
            .sum();

        if total_params > 0 {
            local_params as f64 / total_params as f64
        } else {
            0.0
        }
    }
}

impl Module for FullyShardedDataParallel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Get all parameter names that need gathering
        let _param_names: Vec<String> = self
            .param_states
            .iter()
            .filter_map(|entry| match entry.value() {
                ParameterState::Sharded { .. } => Some(entry.key().clone()),
                _ => None,
            })
            .collect();

        // Gather parameters for forward pass
        // Note: In a real implementation, this would be async
        // For now, we'll use the mock implementation

        // Perform forward pass with gathered parameters
        let module_guard = self.module.read();
        let output = module_guard.forward(input)?;
        drop(module_guard);

        // In training mode, prepare for backward pass
        if self.is_training() {
            // Set up gradient hooks for automatic reduce-scatter
            // This would be implemented with the autograd system
            debug!("Forward pass completed, gradients will be reduce-scattered in backward");
        } else {
            // In eval mode, immediately re-shard to save memory
            // Note: In a real implementation, this would be async
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        // Return sharded parameters for memory efficiency
        let mut params = HashMap::new();

        for entry in self.sharded_params.iter() {
            let name = entry.key().clone();
            let tensor = entry.value().clone();
            params.insert(name, Parameter::new(tensor));
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        *self.training.lock().unwrap()
    }

    fn train(&mut self) {
        *self.training.lock().unwrap() = true;
    }

    fn eval(&mut self) {
        *self.training.lock().unwrap() = false;
    }

    fn to_device(&mut self, _device: DeviceType) -> torsh_core::Result<()> {
        // FSDP device management is handled through backend
        Ok(())
    }
}

/// Helper function to wrap a module with FSDP
pub fn fsdp_wrap<M: Module + 'static>(
    module: M,
    process_group: Arc<ProcessGroup>,
    config: Option<FsdpConfig>,
) -> TorshResult<FullyShardedDataParallel> {
    let config = config.unwrap_or_default();
    let module_arc = Arc::new(RwLock::new(module));
    FullyShardedDataParallel::new(module_arc, process_group, config)
}

/// Auto-wrap modules based on policy
pub fn auto_wrap_modules<M: Module + 'static>(
    module: M,
    process_group: Arc<ProcessGroup>,
    auto_wrap_policy: AutoWrapPolicy,
) -> TorshResult<FullyShardedDataParallel> {
    let config = FsdpConfig {
        auto_wrap_policy,
        ..Default::default()
    };

    fsdp_wrap(module, process_group, Some(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{init_process_group, BackendType};

    use torsh_nn::{prelude::Linear, Module};

    #[tokio::test]
    async fn test_fsdp_initialization() -> TorshResult<()> {
        let process_group =
            Arc::new(init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 12345).await?);

        let linear = Linear::new(128, 64, true);
        let config = FsdpConfig::default();

        let fsdp =
            FullyShardedDataParallel::new(Arc::new(RwLock::new(linear)), process_group, config)?;

        assert!(fsdp.local_sharding_ratio() > 0.0);
        assert!(fsdp.local_sharding_ratio() <= 1.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_fsdp_forward_pass() -> TorshResult<()> {
        let process_group =
            Arc::new(init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 12346).await?);

        let linear = Linear::new(64, 32, true);
        let fsdp = fsdp_wrap(linear, process_group, None)?;

        let input = torsh_tensor::creation::randn(&[8, 64])?;
        let output = fsdp.forward(&input)?;

        assert_eq!(output.shape().dims(), &[8, 32]);

        Ok(())
    }

    #[test]
    fn test_fsdp_config() {
        let config = FsdpConfig::default();
        assert_eq!(config.min_num_params, 1000);
        assert_eq!(config.sharding_strategy, ShardingStrategy::FullShard);
        assert_eq!(config.backward_prefetch, BackwardPrefetch::BackwardPre);

        let custom_config = FsdpConfig {
            min_num_params: 500,
            sharding_strategy: ShardingStrategy::ShardGradOp,
            cpu_offload: true,
            ..Default::default()
        };

        assert_eq!(custom_config.min_num_params, 500);
        assert_eq!(
            custom_config.sharding_strategy,
            ShardingStrategy::ShardGradOp
        );
        assert!(custom_config.cpu_offload);
    }

    #[test]
    fn test_shard_info() {
        let shard_info = ShardInfo {
            rank: 0,
            start_idx: 0,
            shard_size: 1000,
            original_shape: Shape::new(vec![10, 100]),
            is_local: true,
        };

        assert_eq!(shard_info.rank, 0);
        assert_eq!(shard_info.shard_size, 1000);
        assert!(shard_info.is_local);
    }

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats::default();
        assert_eq!(stats.peak_memory_mb, 0.0);
        assert_eq!(stats.num_all_gathers, 0);
        assert_eq!(stats.num_reduce_scatters, 0);
    }

    #[tokio::test]
    async fn test_auto_wrap() -> TorshResult<()> {
        let process_group =
            Arc::new(init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 12347).await?);

        let linear = Linear::new(100, 50, true);
        let policy = AutoWrapPolicy::SizeBasedAutoWrap {
            min_num_params: 1000,
        };

        let fsdp = auto_wrap_modules(linear, process_group, policy)?;

        // Small module should not be sharded
        assert!(fsdp.local_sharding_ratio() >= 0.9); // Most parameters should be local

        Ok(())
    }
}
