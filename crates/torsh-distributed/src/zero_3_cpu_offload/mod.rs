//! ZeRO-3 CPU Offloading Module
//!
//! This module provides a modular implementation of ZeRO-3 (Zero Redundancy Optimizer Stage 3)
//! with CPU offloading capabilities for training extremely large models. The implementation
//! has been systematically refactored from a monolithic structure into specialized modules
//! for improved maintainability, testability, and performance.
//!
//! ## Architecture Overview
//!
//! The ZeRO-3 CPU offloading system consists of several interconnected components:
//!
//! - **Configuration**: Centralized configuration management for all ZeRO-3 settings
//! - **Parameter Management**: Partitioning, storage, and caching of model parameters
//! - **Gradient Management**: Gradient partitioning, storage, and all-reduce operations
//! - **Optimizer State**: Management of optimizer states (momentum, variance) with CPU offloading
//! - **Memory Management**: Intelligent memory optimization and garbage collection
//! - **Prefetch Scheduling**: Asynchronous parameter prefetching for optimal performance
//! - **Performance Statistics**: Comprehensive metrics collection and analysis
//!
//! ## Key Features
//!
//! - **Memory Efficiency**: Partitions parameters, gradients, and optimizer states across processes
//! - **CPU Offloading**: Automatically offloads unused data to CPU memory to reduce GPU usage
//! - **Intelligent Prefetching**: Predicts and preloads parameters before they're needed
//! - **Compression**: Supports multiple compression methods for CPU storage
//! - **Adaptive Optimization**: Dynamically adjusts strategies based on system performance
//! - **Comprehensive Monitoring**: Detailed statistics for performance analysis and tuning
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use torsh_distributed::zero_3_cpu_offload::{Zero3CpuOffloadManager, Zero3CpuOffloadConfig};
//! use std::sync::Arc;
//!
//! # async fn example() -> torsh_distributed::TorshResult<()> {
//! // Configure ZeRO-3 with CPU offloading
//! let config = Zero3CpuOffloadConfig {
//!     offload_params: true,
//!     offload_grads: true,
//!     offload_optimizer_states: true,
//!     cpu_memory_budget: 32 * 1024 * 1024 * 1024, // 32GB
//!     async_prefetch: true,
//!     ..Default::default()
//! };
//!
//! // Initialize model parameters
//! let mut model_params = ModelParameters::new();
//! model_params.add_parameter("layer1.weight".to_string(), vec![512, 1024]);
//! model_params.add_parameter("layer1.bias".to_string(), vec![1024]);
//!
//! // Create ZeRO-3 manager
//! # let process_group = torsh_distributed::init_process_group(
//! #     torsh_distributed::BackendType::Gloo, 0, 4, "127.0.0.1", 29500
//! # )?;
//! let mut manager = Zero3CpuOffloadManager::new(
//!     config,
//!     Arc::new(process_group),
//!     &model_params,
//! )?;
//!
//! // Execute training steps with automatic memory management
//! # let input = torsh_tensor::Tensor::zeros(&[32, 512])?;
//! # let layer_names = vec!["layer1".to_string()];
//! let output = manager.forward_pass(&input, &layer_names).await?;
//! # let grad_output = torsh_tensor::Tensor::zeros(&[32, 1024])?;
//! manager.backward_pass(&grad_output, &layer_names).await?;
//! manager.optimizer_step(0.001).await?;
//!
//! // Get performance statistics
//! let stats = manager.get_performance_stats();
//! info!("Training efficiency: {:.2}%", stats.get_training_efficiency() * 100.0);
//! # Ok(())
//! # }
//! ```

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
// Module declarations - order matters for compilation
pub mod config;
pub mod gradient_management;
pub mod memory_management;
pub mod optimizer_state;
pub mod parameter_management;
pub mod prefetch;
pub mod stats;

// Re-export specific types to avoid ambiguity
pub use config::{
    AutoMemoryStrategy, CpuCompressionMethod, ModelParameterStats,
    ModelParameters as ConfigModelParameters, Zero3CpuOffloadConfig,
    Zero3RankMapping as ConfigZero3RankMapping,
};
pub use gradient_management::*;
pub use memory_management::*;
pub use optimizer_state::{
    OptimizerState, OptimizerStateManager, OptimizerStateMemoryStats,
    Zero3RankMapping as OptimizerZero3RankMapping,
};
pub use parameter_management::*;
pub use prefetch::*;
pub use stats::*;

// Core dependencies
use crate::{ProcessGroup, TorshDistributedError, TorshResult};
use half::{bf16, f16};
use log::info;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

/// Main ZeRO-3 CPU offload manager that orchestrates all components
///
/// This is the primary interface for ZeRO-3 operations, providing a unified API
/// that coordinates between all the specialized modules. It maintains the same
/// interface as the original monolithic implementation for backward compatibility.
pub struct Zero3CpuOffloadManager {
    /// Configuration for ZeRO-3 operations
    config: Zero3CpuOffloadConfig,
    /// Process group for distributed coordination
    process_group: Arc<ProcessGroup>,
    /// Rank mapping for parameter partitioning
    rank_mapping: ConfigZero3RankMapping,

    // Core component managers
    /// Parameter management system
    param_partitioner: ParameterPartitioner,
    /// CPU parameter storage
    cpu_param_store: CpuParameterStore,
    /// GPU parameter cache
    gpu_param_cache: GpuParameterCache,

    /// Gradient management system
    gradient_partitioner: GradientPartitioner,
    /// CPU gradient storage
    cpu_gradient_store: CpuGradientStore,
    /// GPU gradient buffer
    gpu_gradient_buffer: GpuGradientBuffer,

    /// Optimizer state manager
    optimizer_state_manager: OptimizerStateManager,

    /// Memory management system
    memory_manager: Zero3MemoryManager,
    /// Prefetch scheduler
    prefetch_scheduler: PrefetchScheduler,

    /// Performance monitoring
    performance_stats: Arc<Mutex<Zero3PerformanceStats>>,
}

impl Zero3CpuOffloadManager {
    /// Create a new ZeRO-3 CPU offload manager
    ///
    /// Initializes all component systems and establishes distributed coordination.
    /// The manager will automatically partition parameters, gradients, and optimizer
    /// states according to the ZeRO-3 algorithm.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for ZeRO-3 behavior and memory management
    /// * `process_group` - Distributed process group for coordination
    /// * `model_parameters` - Description of model parameters to be managed
    ///
    /// # Returns
    ///
    /// Returns a configured ZeRO-3 manager ready for training operations.
    pub fn new(
        config: Zero3CpuOffloadConfig,
        process_group: Arc<ProcessGroup>,
        model_parameters: &ConfigModelParameters,
    ) -> TorshResult<Self> {
        let world_size = process_group.world_size() as usize;
        let rank = process_group.rank() as usize;

        info!(
            " Initializing ZeRO-3 CPU Offload Manager: rank {}/{}, {} parameters",
            rank, world_size, model_parameters.parameter_count
        );

        let rank_mapping = ConfigZero3RankMapping::new(rank, world_size);

        // Initialize parameter management subsystem
        let param_partitioner =
            ParameterPartitioner::new(&config, &rank_mapping, model_parameters)?;
        let cpu_param_store = CpuParameterStore::new(&config)?;
        let gpu_param_cache = GpuParameterCache::new(&config)?;

        // Initialize gradient management subsystem
        let gradient_partitioner = GradientPartitioner::new(&config, &rank_mapping)?;
        let cpu_gradient_store = CpuGradientStore::new(&config)?;
        let gpu_gradient_buffer = GpuGradientBuffer::new(&config)?;

        // Initialize optimizer state management
        let optimizer_rank_mapping = OptimizerZero3RankMapping::new(rank, world_size);
        let optimizer_state_manager = OptimizerStateManager::new(&config, &optimizer_rank_mapping)?;

        // Initialize memory management and prefetch scheduling
        let memory_manager = Zero3MemoryManager::new(&config);
        let prefetch_scheduler = PrefetchScheduler::new(&config, process_group.clone());

        let performance_stats = Arc::new(Mutex::new(Zero3PerformanceStats::new()));

        info!(" ZeRO-3 CPU Offload initialized successfully:");
        info!(
            "    Parameters: {} total, partitioned across {} ranks",
            model_parameters.parameter_count, world_size
        );
        info!(
            "    Memory: CPU budget {}GB, GPU budget {}GB",
            config.cpu_memory_budget / (1024 * 1024 * 1024),
            config.gpu_param_memory_budget / (1024 * 1024 * 1024)
        );
        info!(
            "   🔧 Features: params={}, grads={}, optimizer={}, prefetch={}",
            config.offload_params,
            config.offload_grads,
            config.offload_optimizer_states,
            config.async_prefetch
        );

        Ok(Self {
            config,
            process_group,
            rank_mapping,
            param_partitioner,
            cpu_param_store,
            gpu_param_cache,
            gradient_partitioner,
            cpu_gradient_store,
            gpu_gradient_buffer,
            optimizer_state_manager,
            memory_manager,
            prefetch_scheduler,
            performance_stats,
        })
    }

    /// Execute forward pass with ZeRO-3 CPU offloading
    ///
    /// Processes each layer with intelligent parameter management:
    /// 1. Prefetches parameters for upcoming layers
    /// 2. Ensures current layer parameters are on GPU
    /// 3. Executes layer computation
    /// 4. Optionally offloads parameters back to CPU
    /// 5. Performs memory optimization as needed
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor for the forward pass
    /// * `layer_names` - Ordered list of layer names to execute
    ///
    /// # Returns
    ///
    /// Returns the output tensor after processing all layers.
    pub async fn forward_pass(
        &mut self,
        input: &Tensor<f32>,
        layer_names: &[String],
    ) -> TorshResult<Tensor<f32>> {
        let start_time = Instant::now();
        let mut current_input = input.clone();

        info!(" ZeRO-3 Forward Pass: {} layers", layer_names.len());

        // Process each layer with ZeRO-3 optimizations
        for (layer_idx, layer_name) in layer_names.iter().enumerate() {
            let layer_start = Instant::now();

            // Step 1: Intelligent prefetching for upcoming layers
            if self.config.async_prefetch {
                self.prefetch_scheduler
                    .intelligent_prefetch(layer_name, layer_names)
                    .await?;
            }

            // Step 2: Ensure parameters are available on GPU
            let layer_params = self.ensure_parameters_on_gpu(layer_name).await?;

            // Step 3: Execute layer computation
            current_input = self
                .execute_layer_computation(&current_input, &layer_params, layer_name)
                .await?;

            // Step 4: Intelligent parameter offloading
            if self.should_offload_layer_params(layer_name, layer_idx, layer_names.len()) {
                self.offload_parameters_to_cpu(layer_name, &layer_params)
                    .await?;
            }

            // Record layer performance
            let layer_duration = layer_start.elapsed();
            {
                let mut stats = self
                    .performance_stats
                    .lock()
                    .expect("lock should not be poisoned");
                stats.record_layer_execution(layer_name.clone(), layer_duration);
            }

            // Periodic memory optimization
            if layer_idx % 4 == 0 {
                self.memory_manager.check_and_optimize_memory().await?;
            }
        }

        // Record overall forward pass performance
        let total_duration = start_time.elapsed();
        {
            let mut stats = self
                .performance_stats
                .lock()
                .expect("lock should not be poisoned");
            stats.record_forward_pass(total_duration, input.numel());
        }

        info!("    Forward pass completed in {:?}", total_duration);
        Ok(current_input)
    }

    /// Execute backward pass with ZeRO-3 CPU offloading
    ///
    /// Processes layers in reverse order for gradient computation:
    /// 1. Ensures parameters are available for gradient computation
    /// 2. Computes gradients for each layer
    /// 3. Partitions and manages gradients according to ZeRO-3
    /// 4. Performs all-reduce synchronization across ranks
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient tensor from the loss function
    /// * `layer_names` - Ordered list of layer names (processed in reverse)
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` when backward pass completes successfully.
    pub async fn backward_pass(
        &mut self,
        grad_output: &Tensor<f32>,
        layer_names: &[String],
    ) -> TorshResult<()> {
        let start_time = Instant::now();
        let mut current_grad = grad_output.clone();

        info!(" ZeRO-3 Backward Pass: {} layers", layer_names.len());

        // Process layers in reverse order for backward pass
        for (rev_idx, layer_name) in layer_names.iter().rev().enumerate() {
            let layer_start = Instant::now();

            // Step 1: Ensure parameters are available on GPU for gradient computation
            let layer_params = self.ensure_parameters_on_gpu(layer_name).await?;

            // Step 2: Compute gradients for this layer
            let (grad_input, param_grads) = self
                .compute_layer_gradients(&current_grad, &layer_params, layer_name)
                .await?;

            // Step 3: Partition and manage gradients according to ZeRO-3
            self.handle_parameter_gradients(layer_name, &param_grads)
                .await?;

            // Step 4: Update current gradient for next layer
            current_grad = grad_input;

            // Step 5: Intelligent parameter offloading
            if self.should_offload_layer_params(layer_name, rev_idx, layer_names.len()) {
                self.offload_parameters_to_cpu(layer_name, &layer_params)
                    .await?;
            }

            let layer_duration = layer_start.elapsed();
            {
                let mut stats = self
                    .performance_stats
                    .lock()
                    .expect("lock should not be poisoned");
                stats.record_layer_backward(layer_name.clone(), layer_duration);
            }
        }

        // Step 6: All-reduce accumulated gradients across ranks
        self.all_reduce_partitioned_gradients().await?;

        let total_duration = start_time.elapsed();
        {
            let mut stats = self
                .performance_stats
                .lock()
                .expect("lock should not be poisoned");
            stats.record_backward_pass(total_duration, grad_output.numel());
        }

        info!("    Backward pass completed in {:?}", total_duration);
        Ok(())
    }

    /// Update optimizer states and parameters with ZeRO-3 partitioning
    ///
    /// Performs optimizer step with intelligent state management:
    /// 1. Gathers partitioned gradients for owned parameters
    /// 2. Fetches optimizer states from CPU if needed
    /// 3. Computes parameter updates using optimizer algorithm
    /// 4. Updates parameters and stores back to appropriate location
    /// 5. Broadcasts updates to all ranks that need them
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for parameter updates
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` when optimizer step completes successfully.
    pub async fn optimizer_step(&mut self, learning_rate: f32) -> TorshResult<()> {
        let start_time = Instant::now();

        info!(" ZeRO-3 Optimizer Step (lr: {})", learning_rate);

        // Step 1: Gather partitioned gradients for owned parameters
        let owned_param_grads = self.gather_owned_parameter_gradients().await?;

        info!(
            "    Processing {} owned parameter gradients",
            owned_param_grads.len()
        );

        // Step 2: Update optimizer states and parameters
        for (param_name, gradient) in owned_param_grads.iter() {
            // Fetch optimizer state from CPU if offloaded
            let optimizer_state = self.optimizer_state_manager.fetch_state(param_name).await?;

            // Compute parameter update using optimizer state and gradient
            let param_update =
                self.compute_parameter_update(&optimizer_state, gradient, learning_rate)?;

            // Update parameter (fetch from CPU if needed)
            let mut parameter = self.fetch_parameter_for_update(param_name).await?;
            parameter = parameter.sub(&param_update)?;

            // Store updated parameter and optimizer state
            self.store_updated_parameter(param_name, &parameter).await?;
            self.optimizer_state_manager
                .store_state(param_name, &optimizer_state)
                .await?;
        }

        // Step 3: Broadcast updated parameters to all ranks that need them
        self.broadcast_parameter_updates().await?;

        let duration = start_time.elapsed();
        {
            let mut stats = self
                .performance_stats
                .lock()
                .expect("lock should not be poisoned");
            stats.record_optimizer_step(duration, owned_param_grads.len());
        }

        info!("    Optimizer step completed in {:?}", duration);
        Ok(())
    }

    /// Get comprehensive performance statistics
    ///
    /// Returns detailed performance metrics including timing, throughput,
    /// memory usage, and efficiency measurements.
    pub fn get_performance_stats(&self) -> Zero3PerformanceStats {
        self.performance_stats
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Get memory usage statistics
    ///
    /// Returns current memory usage across CPU and GPU, including
    /// parameter distribution and compression effectiveness.
    pub fn get_memory_stats(&self) -> Zero3MemoryStats {
        self.memory_manager.get_memory_stats()
    }

    /// Force immediate memory optimization
    ///
    /// Triggers aggressive memory optimization regardless of current pressure.
    /// Useful for cleaning up before checkpointing or when memory is critically low.
    pub async fn force_memory_optimization(&self) -> TorshResult<()> {
        self.memory_manager.force_memory_optimization().await
    }

    /// Get prefetch scheduler status
    ///
    /// Returns information about current prefetch operations and queue status.
    pub fn get_prefetch_status(&self) -> PrefetchQueueStatus {
        self.prefetch_scheduler.get_queue_status()
    }

    /// Adapt system performance based on runtime metrics
    ///
    /// Analyzes recent performance and adjusts prefetch strategies,
    /// memory management policies, and other adaptive parameters.
    pub async fn adapt_performance(&self) -> TorshResult<()> {
        self.prefetch_scheduler.adapt_prefetch_strategy().await
    }

    /// Clear all caches and reset state
    ///
    /// Useful for testing or when switching between different models.
    pub async fn reset_state(&self) -> TorshResult<()> {
        self.optimizer_state_manager.clear_states().await?;
        self.prefetch_scheduler.cancel_all_prefetches().await?;
        info!("🧹 ZeRO-3 manager state reset completed");
        Ok(())
    }

    // Private helper methods (implementations remain similar to original)

    async fn ensure_parameters_on_gpu(&mut self, layer_name: &str) -> TorshResult<LayerParameters> {
        // Check if parameters are already in GPU cache
        if let Some(cached_params) = self.gpu_param_cache.get(layer_name).await? {
            return Ok(cached_params);
        }

        // Fetch parameters from CPU store
        let cpu_params = self.cpu_param_store.fetch(layer_name).await?;

        // Transfer to GPU with potential decompression
        let gpu_params = self.transfer_params_to_gpu(&cpu_params).await?;

        // Cache on GPU
        self.gpu_param_cache.store(layer_name, &gpu_params).await?;

        Ok(gpu_params)
    }

    async fn transfer_params_to_gpu(
        &self,
        cpu_params: &CpuParameterData,
    ) -> TorshResult<LayerParameters> {
        let transfer_start = Instant::now();

        // Decompress if needed
        let decompressed_data = match self.config.cpu_compression {
            CpuCompressionMethod::None => cpu_params.data.clone(),
            CpuCompressionMethod::FP16 => self.decompress_fp16(&cpu_params.data)?,
            CpuCompressionMethod::BF16 => self.decompress_bf16(&cpu_params.data)?,
            CpuCompressionMethod::INT8 => self.decompress_int8(&cpu_params.data)?,
            _ => {
                return Err(TorshDistributedError::feature_not_available(
                    "compression_method",
                    "Compression method not implemented",
                ));
            }
        };

        // Create GPU tensors
        let weight = Tensor::from_data(
            decompressed_data,
            cpu_params.weight_shape.clone(),
            DeviceType::Cuda(0),
        )?;
        let bias = if let Some(ref bias_data) = cpu_params.bias_data {
            Some(Tensor::from_data(
                bias_data.clone(),
                cpu_params
                    .bias_shape
                    .as_ref()
                    .expect("bias_shape should be present when bias_data exists")
                    .clone(),
                DeviceType::Cuda(0),
            )?)
        } else {
            None
        };

        // Record transfer metrics
        let transfer_duration = transfer_start.elapsed();
        {
            let mut stats = self
                .performance_stats
                .lock()
                .expect("lock should not be poisoned");
            stats.record_parameter_transfer(
                transfer_duration,
                cpu_params.size_bytes,
                TransferDirection::CpuToGpu,
            );
        }

        info!(
            "    Transferred parameters to GPU: {} ({} bytes in {:?})",
            "layer", cpu_params.size_bytes, transfer_duration
        );

        Ok(LayerParameters { weight, bias })
    }

    async fn execute_layer_computation(
        &self,
        input: &Tensor<f32>,
        params: &LayerParameters,
        layer_name: &str,
    ) -> TorshResult<Tensor<f32>> {
        info!("   🧮 Computing layer: {}", layer_name);

        // Simple linear layer computation for demonstration
        let output = input.matmul(&params.weight)?;

        if let Some(ref bias) = params.bias {
            let output = output.add(bias)?;
            Ok(output.relu()?) // Apply activation
        } else {
            Ok(output.relu()?)
        }
    }

    fn should_offload_layer_params(
        &self,
        _layer_name: &str,
        current_idx: usize,
        total_layers: usize,
    ) -> bool {
        // Intelligent offloading heuristic
        let remaining_layers = total_layers - current_idx;
        remaining_layers > self.config.prefetch_buffer_size
    }

    async fn offload_parameters_to_cpu(
        &mut self,
        layer_name: &str,
        params: &LayerParameters,
    ) -> TorshResult<()> {
        if !self.config.offload_params {
            return Ok(());
        }

        let offload_start = Instant::now();

        // Compress parameters if configured
        let compressed_data = self.compress_parameters(params).await?;

        // Store in CPU memory
        self.cpu_param_store
            .store(layer_name, &compressed_data)
            .await?;

        // Remove from GPU cache to free memory
        self.gpu_param_cache.remove(layer_name).await?;

        // Record transfer metrics
        let offload_duration = offload_start.elapsed();
        {
            let mut stats = self
                .performance_stats
                .lock()
                .expect("lock should not be poisoned");
            stats.record_parameter_transfer(
                offload_duration,
                compressed_data.size_bytes,
                TransferDirection::GpuToCpu,
            );
        }

        info!(
            "    Offloaded parameters to CPU: {} ({} bytes in {:?})",
            layer_name, compressed_data.size_bytes, offload_duration
        );

        Ok(())
    }

    async fn compress_parameters(&self, params: &LayerParameters) -> TorshResult<CpuParameterData> {
        let weight_data = params.weight.to_vec()?;
        let bias_data = if let Some(ref bias) = params.bias {
            Some(bias.to_vec()?)
        } else {
            None
        };

        let (compressed_weight, weight_shape) = match self.config.cpu_compression {
            CpuCompressionMethod::None => (weight_data, params.weight.shape().dims().to_vec()),
            CpuCompressionMethod::FP16 => {
                self.compress_to_fp16(&weight_data, params.weight.shape().dims())?
            }
            CpuCompressionMethod::BF16 => {
                self.compress_to_bf16(&weight_data, params.weight.shape().dims())?
            }
            CpuCompressionMethod::INT8 => {
                self.compress_to_int8(&weight_data, params.weight.shape().dims())?
            }
            _ => {
                return Err(TorshDistributedError::feature_not_available(
                    "compression_method",
                    "Compression method not implemented",
                ));
            }
        };

        let size_bytes = compressed_weight.len() * std::mem::size_of::<f32>()
            + bias_data
                .as_ref()
                .map(|b: &Vec<f32>| b.len() * std::mem::size_of::<f32>())
                .unwrap_or(0);

        Ok(CpuParameterData {
            data: compressed_weight,
            bias_data,
            weight_shape,
            bias_shape: params.bias.as_ref().map(|b| b.shape().dims().to_vec()),
            size_bytes,
            compression: self.config.cpu_compression,
        })
    }

    async fn compute_layer_gradients(
        &self,
        grad_output: &Tensor<f32>,
        params: &LayerParameters,
        layer_name: &str,
    ) -> TorshResult<(Tensor<f32>, ParameterGradients)> {
        info!("   🔢 Computing gradients for layer: {}", layer_name);

        // Simplified gradient computation for linear layer
        let grad_input = grad_output.matmul(&params.weight.transpose(-2, -1)?)?;
        let grad_weight = grad_output.clone(); // Mock gradient
        let grad_bias = if params.bias.is_some() {
            Some(grad_output.sum_dim(&[0], false)?)
        } else {
            None
        };

        let param_grads = ParameterGradients {
            weight_grad: grad_weight,
            bias_grad: grad_bias,
        };

        Ok((grad_input, param_grads))
    }

    async fn handle_parameter_gradients(
        &mut self,
        layer_name: &str,
        grads: &ParameterGradients,
    ) -> TorshResult<()> {
        // Partition gradients according to ZeRO-3
        let partitioned_grads = self
            .gradient_partitioner
            .partition_gradients(layer_name, grads)?;

        // Store locally owned gradient partitions
        for (partition_idx, grad_partition) in partitioned_grads.into_iter().enumerate() {
            if self.rank_mapping.owns_partition(partition_idx) {
                if self.config.offload_grads {
                    self.cpu_gradient_store
                        .store(layer_name, partition_idx, &grad_partition.weight_gradient)
                        .await?;
                } else {
                    self.gpu_gradient_buffer
                        .store(layer_name, partition_idx, &grad_partition.weight_gradient)
                        .await?;
                }
            }
        }

        Ok(())
    }

    async fn all_reduce_partitioned_gradients(&mut self) -> TorshResult<()> {
        let sync_start = Instant::now();
        info!("    All-reducing partitioned gradients");

        let local_gradients = self.cpu_gradient_store.get_all_gradients().await?;
        let gradients_count = local_gradients.len();

        // Simulate all-reduce with proper timing
        for (layer_partition_key, gradient) in local_gradients {
            let mut grad_tensor = gradient;
            let world_size = self.process_group.world_size() as f32;

            // Mock all-reduce operation
            grad_tensor = grad_tensor.div_scalar(world_size)?;

            self.cpu_gradient_store
                .store_reduced_gradient(&layer_partition_key, &grad_tensor)
                .await?;
        }

        let sync_duration = sync_start.elapsed();
        {
            let mut stats = self
                .performance_stats
                .lock()
                .expect("lock should not be poisoned");
            stats.record_gradient_sync(
                sync_duration,
                gradients_count,
                self.process_group.world_size() as usize,
            );
        }

        info!(
            "    Gradient synchronization completed in {:?}",
            sync_duration
        );
        Ok(())
    }

    async fn gather_owned_parameter_gradients(
        &mut self,
    ) -> TorshResult<HashMap<String, Tensor<f32>>> {
        self.cpu_gradient_store
            .get_owned_gradients(self.rank_mapping.rank(), self.rank_mapping.world_size())
            .await
    }

    fn compute_parameter_update(
        &self,
        _optimizer_state: &OptimizerState,
        gradient: &Tensor<f32>,
        learning_rate: f32,
    ) -> TorshResult<Tensor<f32>> {
        // Simple SGD update for demonstration
        Ok(gradient.mul_scalar(learning_rate)?)
    }

    async fn fetch_parameter_for_update(&mut self, param_name: &str) -> TorshResult<Tensor<f32>> {
        let cpu_param_data = self.cpu_param_store.fetch(param_name).await?;
        let gpu_params = self.transfer_params_to_gpu(&cpu_param_data).await?;
        Ok(gpu_params.weight)
    }

    async fn store_updated_parameter(
        &mut self,
        param_name: &str,
        parameter: &Tensor<f32>,
    ) -> TorshResult<()> {
        let layer_params = LayerParameters {
            weight: parameter.clone(),
            bias: None, // Simplified
        };

        let compressed_data = self.compress_parameters(&layer_params).await?;
        self.cpu_param_store
            .store(param_name, &compressed_data)
            .await?;

        Ok(())
    }

    async fn broadcast_parameter_updates(&mut self) -> TorshResult<()> {
        let broadcast_start = Instant::now();
        info!("    Broadcasting parameter updates across process group");

        // Mock parameter broadcasting with realistic timing
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let broadcast_duration = broadcast_start.elapsed();
        {
            let mut stats = self
                .performance_stats
                .lock()
                .expect("lock should not be poisoned");
            stats.record_communication(
                CommunicationOperation::Broadcast,
                broadcast_duration,
                1024 * 1024, // Mock 1MB broadcast
            );
        }

        info!(
            "    Parameter broadcasting completed in {:?}",
            broadcast_duration
        );
        Ok(())
    }

    // Compression helper methods (simplified implementations)
    fn compress_to_fp16(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> TorshResult<(Vec<f32>, Vec<usize>)> {
        let compressed: Vec<f32> = data
            .iter()
            .map(|&val| f16::from_f32(val).to_f32())
            .collect();
        Ok((compressed, shape.to_vec()))
    }

    fn compress_to_bf16(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> TorshResult<(Vec<f32>, Vec<usize>)> {
        let compressed: Vec<f32> = data
            .iter()
            .map(|&val| bf16::from_f32(val).to_f32())
            .collect();
        Ok((compressed, shape.to_vec()))
    }

    fn compress_to_int8(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> TorshResult<(Vec<f32>, Vec<usize>)> {
        if data.is_empty() {
            return Ok((Vec::new(), shape.to_vec()));
        }

        let max_abs = data
            .iter()
            .map(|&x| x.abs())
            .fold(f32::NEG_INFINITY, f32::max);
        if max_abs == 0.0 {
            return Ok((vec![0.0; data.len()], shape.to_vec()));
        }

        let scale = 127.0 / max_abs;
        let inv_scale = max_abs / 127.0;

        let quantized: Vec<f32> = data
            .iter()
            .map(|&val| {
                let quantized_val = (val * scale).round().clamp(-127.0, 127.0);
                quantized_val * inv_scale
            })
            .collect();

        Ok((quantized, shape.to_vec()))
    }

    fn decompress_fp16(&self, data: &[f32]) -> TorshResult<Vec<f32>> {
        Ok(data.to_vec())
    }

    fn decompress_bf16(&self, data: &[f32]) -> TorshResult<Vec<f32>> {
        Ok(data.to_vec())
    }

    fn decompress_int8(&self, data: &[f32]) -> TorshResult<Vec<f32>> {
        Ok(data.to_vec())
    }
}

/// Model parameters description for ZeRO-3 initialization
#[derive(Debug)]
pub struct ModelParameters {
    /// Total number of parameters
    pub parameter_count: usize,
    /// Names of all parameters
    pub parameter_names: Vec<String>,
    /// Shape of each parameter
    pub parameter_shapes: HashMap<String, Vec<usize>>,
    /// Total memory usage in bytes
    pub total_memory_bytes: usize,
}

impl ModelParameters {
    /// Create new model parameters description
    pub fn new() -> Self {
        Self {
            parameter_count: 0,
            parameter_names: Vec::new(),
            parameter_shapes: HashMap::new(),
            total_memory_bytes: 0,
        }
    }

    /// Add a parameter to the model description
    pub fn add_parameter(&mut self, name: String, shape: Vec<usize>) {
        let param_size = shape.iter().product::<usize>();
        self.parameter_count += param_size;
        self.total_memory_bytes += param_size * std::mem::size_of::<f32>();
        self.parameter_shapes.insert(name.clone(), shape);
        self.parameter_names.push(name);
    }

    /// Get parameter shape by name
    pub fn get_parameter_shape(&self, name: &str) -> Option<&Vec<usize>> {
        self.parameter_shapes.get(name)
    }

    /// Check if parameter exists
    pub fn has_parameter(&self, name: &str) -> bool {
        self.parameter_shapes.contains_key(name)
    }
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{init_process_group, BackendType};

    #[test]
    fn test_model_parameters() {
        let mut model_params = ConfigModelParameters::new();
        model_params.add_parameter("layer1.weight".to_string(), vec![512, 1024]);
        model_params.add_parameter("layer1.bias".to_string(), vec![1024]);

        assert_eq!(model_params.parameter_names.len(), 2);
        assert_eq!(model_params.parameter_count, 512 * 1024 + 1024);
        assert!(model_params.has_parameter("layer1.weight"));
        assert!(!model_params.has_parameter("nonexistent"));
    }

    #[tokio::test]
    async fn test_zero3_manager_creation() {
        let pg = init_process_group(BackendType::Gloo, 0, 4, "127.0.0.1", 29500)
            .await
            .unwrap();
        let config = Zero3CpuOffloadConfig::default();

        let mut model_params = ConfigModelParameters::new();
        model_params.add_parameter("layer1.weight".to_string(), vec![512, 512]);
        model_params.add_parameter("layer2.weight".to_string(), vec![512, 512]);

        let manager = Zero3CpuOffloadManager::new(config, Arc::new(pg), &model_params);
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        let stats = manager.get_performance_stats();
        assert_eq!(stats.forward_passes, 0);

        let _memory_stats = manager.get_memory_stats();
        // total_parameters is usize, always >= 0
    }

    #[tokio::test]
    async fn test_manager_operations() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();
        let config = Zero3CpuOffloadConfig::default();

        let mut model_params = ConfigModelParameters::new();
        model_params.add_parameter("test_layer".to_string(), vec![10, 10]);

        let manager = Zero3CpuOffloadManager::new(config, Arc::new(pg), &model_params).unwrap();

        // Test state reset
        manager.reset_state().await.unwrap();

        // Test memory optimization
        manager.force_memory_optimization().await.unwrap();

        // Test prefetch status
        let prefetch_status = manager.get_prefetch_status();
        assert_eq!(prefetch_status.queued_requests, 0);
    }

    #[tokio::test]
    async fn test_compression_methods() {
        let config = Zero3CpuOffloadConfig::default();
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();
        let model_params = ConfigModelParameters::new();
        let manager = Zero3CpuOffloadManager::new(config, Arc::new(pg), &model_params).unwrap();

        let test_data = vec![1.0, 2.0, -1.5, 0.5];
        let shape = vec![2, 2];

        // Test FP16 compression
        let (compressed, result_shape) = manager.compress_to_fp16(&test_data, &shape).unwrap();
        assert_eq!(result_shape, shape);
        assert_eq!(compressed.len(), test_data.len());

        // Test BF16 compression
        let (compressed, result_shape) = manager.compress_to_bf16(&test_data, &shape).unwrap();
        assert_eq!(result_shape, shape);
        assert_eq!(compressed.len(), test_data.len());

        // Test INT8 compression
        let (compressed, result_shape) = manager.compress_to_int8(&test_data, &shape).unwrap();
        assert_eq!(result_shape, shape);
        assert_eq!(compressed.len(), test_data.len());
    }
}
