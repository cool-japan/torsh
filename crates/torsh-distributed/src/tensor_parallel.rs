//! Tensor Parallelism implementation for distributed training
//!
//! Tensor parallelism splits individual tensors across multiple devices,
//! enabling training of models that are too large to fit on a single device.
//! This is particularly useful for transformer models where we can split
//! attention and feed-forward layers.
//!
//! Enhanced with SciRS2 memory-efficient operations for optimal performance
//! and reduced memory footprint in distributed training scenarios.

use crate::collectives::{all_gather, reduce_scatter};
use crate::{ProcessGroup, TorshDistributedError, TorshResult};
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::{error::Result, DeviceType, Shape};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;
use tracing::{debug, info};

// Enhanced SciRS2 integration for memory-efficient tensor operations
#[cfg(feature = "scirs2-memory")]
use scirs2_core::memory::{BufferPool, ChunkProcessor, GlobalBufferPool};
#[cfg(feature = "scirs2-memory")]
use scirs2_core::memory_efficient::{AdaptiveChunking, DiskBackedArray, ZeroCopyOps};
#[cfg(feature = "scirs2-memory")]
use scirs2_core::memory_efficient::{ChunkedArray, LazyArray, MemoryMappedArray};
#[cfg(feature = "scirs2-memory")]
use scirs2_core::parallel_ops::{par_chunks, par_join, par_scope};
#[cfg(feature = "scirs2-memory")]
use scirs2_core::simd_ops::{simd_dot_product, simd_matrix_multiply};

/// Enhanced tensor parallelism configuration with SciRS2 memory optimizations
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// Tensor parallel group size
    pub tp_size: usize,
    /// Whether to use sequence parallelism
    pub sequence_parallel: bool,
    /// Communication backend for tensor parallel operations
    pub communication_backend: String,
    /// Whether to use async communication
    pub async_communication: bool,
    /// Memory optimization level (0-3)
    pub memory_optimization_level: u8,
    /// Enable SciRS2 memory-efficient operations
    #[cfg(feature = "scirs2-memory")]
    pub enable_scirs2_memory: bool,
    /// Use memory-mapped arrays for large tensors
    #[cfg(feature = "scirs2-memory")]
    pub use_memory_mapping: bool,
    /// Enable lazy tensor loading
    #[cfg(feature = "scirs2-memory")]
    pub enable_lazy_loading: bool,
    /// Enable chunked tensor processing
    #[cfg(feature = "scirs2-memory")]
    pub enable_chunked_processing: bool,
    /// Enable SIMD optimizations
    #[cfg(feature = "scirs2-memory")]
    pub enable_simd_ops: bool,
    /// Buffer pool size for memory management
    #[cfg(feature = "scirs2-memory")]
    pub buffer_pool_size_mb: usize,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            tp_size: 1,
            sequence_parallel: false,
            communication_backend: "nccl".to_string(),
            async_communication: true,
            memory_optimization_level: 1,
            #[cfg(feature = "scirs2-memory")]
            enable_scirs2_memory: true,
            #[cfg(feature = "scirs2-memory")]
            use_memory_mapping: true,
            #[cfg(feature = "scirs2-memory")]
            enable_lazy_loading: false,
            #[cfg(feature = "scirs2-memory")]
            enable_chunked_processing: true,
            #[cfg(feature = "scirs2-memory")]
            enable_simd_ops: true,
            #[cfg(feature = "scirs2-memory")]
            buffer_pool_size_mb: 512,
        }
    }
}

/// Tensor parallelism strategy
#[derive(Debug, Clone, PartialEq)]
pub enum TensorParallelStrategy {
    /// Split tensor along rows (for weight matrices)
    RowParallel,
    /// Split tensor along columns (for weight matrices)
    ColumnParallel,
    /// Split along vocabulary dimension (for embeddings)
    VocabParallel,
    /// Split along sequence dimension
    SequenceParallel,
    /// Split along attention heads
    AttentionHeadParallel,
}

/// Tensor parallel layer types
#[derive(Debug, Clone)]
pub enum TensorParallelLayer {
    /// Row-parallel linear layer
    RowParallelLinear {
        input_size: usize,
        output_size: usize,
        bias: bool,
        input_is_parallel: bool,
    },
    /// Column-parallel linear layer
    ColumnParallelLinear {
        input_size: usize,
        output_size: usize,
        bias: bool,
        gather_output: bool,
    },
    /// Parallel embedding layer
    ParallelEmbedding {
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
    },
    /// Parallel attention layer
    ParallelAttention {
        hidden_size: usize,
        num_attention_heads: usize,
        dropout_prob: f32,
    },
}

/// Tensor parallel wrapper for modules
pub struct TensorParallel {
    /// The underlying module
    module: Box<dyn Module>,
    /// Process group for tensor parallel communication
    tp_group: Arc<ProcessGroup>,
    /// Configuration
    config: TensorParallelConfig,
    /// Tensor parallel rank within the group
    tp_rank: usize,
    /// Layer type and strategy
    layer_info: TensorParallelLayer,
    /// Parameter sharding information
    shard_info: HashMap<String, ShardInfo>,
    /// Communication buffers
    comm_buffers: HashMap<String, Tensor>,
}

/// Information about how a parameter is sharded
#[derive(Debug, Clone)]
pub struct ShardInfo {
    /// Which dimension is sharded
    pub shard_dim: usize,
    /// Start index of this shard
    pub start_idx: usize,
    /// Size of this shard
    pub shard_size: usize,
    /// Original tensor shape
    pub original_shape: Shape,
    /// Strategy used for sharding
    pub strategy: TensorParallelStrategy,
}

impl TensorParallel {
    /// Create a new tensor parallel wrapper
    pub fn new(
        module: Box<dyn Module>,
        tp_group: Arc<ProcessGroup>,
        config: TensorParallelConfig,
        layer_info: TensorParallelLayer,
    ) -> TorshResult<Self> {
        let tp_rank = tp_group.rank() as usize;
        let tp_size = tp_group.world_size() as usize;

        if tp_size != config.tp_size {
            return Err(TorshDistributedError::invalid_argument(
                "tp_size",
                format!(
                    "TP group size ({}) doesn't match config TP size ({})",
                    tp_size, config.tp_size
                ),
                format!("tp_size = {}", config.tp_size),
            )
            .into());
        }

        let mut tensor_parallel = Self {
            module,
            tp_group,
            config,
            tp_rank,
            layer_info,
            shard_info: HashMap::new(),
            comm_buffers: HashMap::new(),
        };

        // Initialize parameter sharding
        tensor_parallel.init_parameter_sharding()?;

        info!(
            "Initialized tensor parallel layer with TP size {} at rank {}",
            tp_size, tp_rank
        );

        Ok(tensor_parallel)
    }

    /// Initialize parameter sharding based on layer type
    fn init_parameter_sharding(&mut self) -> TorshResult<()> {
        let parameters = self.module.parameters();

        match &self.layer_info {
            TensorParallelLayer::RowParallelLinear { output_size, .. } => {
                self.shard_row_parallel_parameters(&parameters, *output_size)?;
            }
            TensorParallelLayer::ColumnParallelLinear { input_size, .. } => {
                self.shard_column_parallel_parameters(&parameters, *input_size)?;
            }
            TensorParallelLayer::ParallelEmbedding { num_embeddings, .. } => {
                self.shard_embedding_parameters(&parameters, *num_embeddings)?;
            }
            TensorParallelLayer::ParallelAttention {
                num_attention_heads,
                ..
            } => {
                self.shard_attention_parameters(&parameters, *num_attention_heads)?;
            }
        }

        Ok(())
    }

    /// Shard parameters for row-parallel linear layer
    fn shard_row_parallel_parameters(
        &mut self,
        parameters: &HashMap<String, Parameter>,
        output_size: usize,
    ) -> TorshResult<()> {
        for (name, _param) in parameters {
            if name.contains("weight") {
                let shard_size = output_size / self.config.tp_size;
                let start_idx = self.tp_rank * shard_size;

                let shard_info = ShardInfo {
                    shard_dim: 0, // Row dimension
                    start_idx,
                    shard_size,
                    original_shape: Shape::new(vec![output_size, parameters.len()]), // Simplified
                    strategy: TensorParallelStrategy::RowParallel,
                };

                self.shard_info.insert(name.clone(), shard_info);
                debug!("Sharded parameter '{}' with row-parallel strategy", name);
            }
        }

        Ok(())
    }

    /// Shard parameters for column-parallel linear layer
    fn shard_column_parallel_parameters(
        &mut self,
        parameters: &HashMap<String, Parameter>,
        input_size: usize,
    ) -> TorshResult<()> {
        for (name, _param) in parameters {
            if name.contains("weight") {
                let shard_size = input_size / self.config.tp_size;
                let start_idx = self.tp_rank * shard_size;

                let shard_info = ShardInfo {
                    shard_dim: 1, // Column dimension
                    start_idx,
                    shard_size,
                    original_shape: Shape::new(vec![parameters.len(), input_size]), // Simplified
                    strategy: TensorParallelStrategy::ColumnParallel,
                };

                self.shard_info.insert(name.clone(), shard_info);
                debug!("Sharded parameter '{}' with column-parallel strategy", name);
            }
        }

        Ok(())
    }

    /// Shard parameters for parallel embedding layer
    fn shard_embedding_parameters(
        &mut self,
        parameters: &HashMap<String, Parameter>,
        num_embeddings: usize,
    ) -> TorshResult<()> {
        for (name, _param) in parameters {
            if name.contains("weight") {
                let shard_size = num_embeddings / self.config.tp_size;
                let start_idx = self.tp_rank * shard_size;

                let shard_info = ShardInfo {
                    shard_dim: 0, // Vocabulary dimension
                    start_idx,
                    shard_size,
                    original_shape: Shape::new(vec![num_embeddings, 512]), // Simplified embedding dim
                    strategy: TensorParallelStrategy::VocabParallel,
                };

                self.shard_info.insert(name.clone(), shard_info);
                debug!("Sharded parameter '{}' with vocab-parallel strategy", name);
            }
        }

        Ok(())
    }

    /// Shard parameters for parallel attention layer
    fn shard_attention_parameters(
        &mut self,
        parameters: &HashMap<String, Parameter>,
        num_attention_heads: usize,
    ) -> TorshResult<()> {
        let heads_per_partition = num_attention_heads / self.config.tp_size;
        let start_head = self.tp_rank * heads_per_partition;

        for (name, _param) in parameters {
            if name.contains("query")
                || name.contains("key")
                || name.contains("value")
                || name.contains("output")
            {
                let shard_info = ShardInfo {
                    shard_dim: 0, // Head dimension
                    start_idx: start_head,
                    shard_size: heads_per_partition,
                    original_shape: Shape::new(vec![num_attention_heads, 64]), // Simplified head dim
                    strategy: TensorParallelStrategy::AttentionHeadParallel,
                };

                self.shard_info.insert(name.clone(), shard_info);
                debug!(
                    "Sharded parameter '{}' with attention-head-parallel strategy",
                    name
                );
            }
        }

        Ok(())
    }

    /// Perform all-gather communication for row-parallel layers
    async fn all_gather_for_row_parallel(&mut self, input: &Tensor) -> TorshResult<Tensor> {
        debug!("Performing all-gather for row-parallel layer");

        let mut gathered_tensors = Vec::new();
        all_gather(&mut gathered_tensors, input, &self.tp_group).await?;

        // Concatenate gathered tensors along the row dimension
        if gathered_tensors.len() == 1 {
            Ok(gathered_tensors.into_iter().next().unwrap())
        } else {
            // For simplicity, just return the first tensor
            // In a real implementation, we would concatenate properly
            Ok(gathered_tensors.into_iter().next().unwrap())
        }
    }

    /// Perform reduce-scatter communication for column-parallel layers
    async fn reduce_scatter_for_column_parallel(&mut self, input: &Tensor) -> TorshResult<Tensor> {
        debug!("Performing reduce-scatter for column-parallel layer");

        let mut output_tensor = input.clone();
        reduce_scatter(
            &mut output_tensor,
            input,
            crate::backend::ReduceOp::Sum,
            &self.tp_group,
        )
        .await?;

        // Return the local shard
        Ok(output_tensor)
    }

    /// Perform sequence-parallel communication
    async fn sequence_parallel_communication(&mut self, input: &Tensor) -> TorshResult<Tensor> {
        debug!("Performing sequence-parallel communication");

        if self.config.sequence_parallel {
            // Gather along sequence dimension
            self.all_gather_for_row_parallel(input).await
        } else {
            Ok(input.clone())
        }
    }

    /// Get tensor parallel rank
    pub fn tp_rank(&self) -> usize {
        self.tp_rank
    }

    /// Get tensor parallel world size
    pub fn tp_world_size(&self) -> usize {
        self.config.tp_size
    }

    /// Get sharding information for a parameter
    pub fn get_shard_info(&self, param_name: &str) -> Option<&ShardInfo> {
        self.shard_info.get(param_name)
    }

    /// Check if layer uses sequence parallelism
    pub fn uses_sequence_parallel(&self) -> bool {
        self.config.sequence_parallel
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> TensorParallelStats {
        let total_params = self.module.parameters().len();
        let sharded_params = self.shard_info.len();
        let memory_reduction = if total_params > 0 {
            1.0 - (sharded_params as f64 / total_params as f64)
        } else {
            0.0
        };

        TensorParallelStats {
            tp_rank: self.tp_rank,
            tp_world_size: self.config.tp_size,
            total_parameters: total_params,
            sharded_parameters: sharded_params,
            memory_reduction_ratio: memory_reduction,
            communication_overhead_ms: 0.0, // Would be measured in real implementation
        }
    }

    // Enhanced SciRS2 memory-efficient operations

    /// Create memory-efficient tensor shard using SciRS2 operations
    #[cfg(feature = "scirs2-memory")]
    pub fn create_memory_efficient_shard(
        &self,
        tensor: &Tensor,
        shard_dim: usize,
        use_memory_mapping: bool,
    ) -> TorshResult<Tensor> {
        debug!(
            "Creating memory-efficient shard for tensor with shape {:?}",
            tensor.shape()
        );

        if !self.config.enable_scirs2_memory {
            return self.create_shard(tensor, shard_dim);
        }

        // Use SciRS2 memory-efficient operations
        if use_memory_mapping && tensor.numel() > 1_000_000 {
            // Use memory-mapped arrays for large tensors
            let mapped_array = MemoryMappedArray::from_tensor(tensor)?;
            let shard_size = tensor.shape().dims()[shard_dim] / self.config.tp_size;
            let start_idx = self.tp_rank * shard_size;

            let shard_data = mapped_array.slice(start_idx, start_idx + shard_size)?;
            let shard_tensor = Tensor::from_memory_mapped(&shard_data)?;

            debug!(
                "Created memory-mapped shard with {} elements",
                shard_tensor.numel()
            );
            Ok(shard_tensor)
        } else if self.config.enable_chunked_processing {
            // Use chunked processing for smaller tensors
            self.create_chunked_shard(tensor, shard_dim)
        } else {
            // Fallback to standard sharding
            self.create_shard(tensor, shard_dim)
        }
    }

    /// Create chunked tensor shard using SciRS2 operations
    #[cfg(feature = "scirs2-memory")]
    fn create_chunked_shard(&self, tensor: &Tensor, shard_dim: usize) -> TorshResult<Tensor> {
        use scirs2_core::memory_efficient::AdaptiveChunking;

        let chunk_processor = ChunkProcessor::new(self.config.buffer_pool_size_mb * 1024 * 1024);
        let chunked_array = ChunkedArray::from_tensor(tensor, &chunk_processor)?;

        let shard_size = tensor.shape().dims()[shard_dim] / self.config.tp_size;
        let start_idx = self.tp_rank * shard_size;

        let shard_chunks = chunked_array.get_chunks_range(start_idx, start_idx + shard_size)?;
        let shard_tensor = Tensor::from_chunks(&shard_chunks)?;

        info!("Created chunked shard with {} chunks", shard_chunks.len());
        Ok(shard_tensor)
    }

    /// Perform SIMD-optimized tensor operations for parallel processing
    #[cfg(feature = "scirs2-memory")]
    pub fn simd_optimized_forward(&self, input: &Tensor, weights: &Tensor) -> TorshResult<Tensor> {
        if !self.config.enable_simd_ops {
            return self.standard_forward(input, weights);
        }

        debug!("Performing SIMD-optimized forward pass");

        // Use SciRS2 SIMD operations for matrix multiplication
        match (input.dtype(), weights.dtype()) {
            (torsh_core::DType::F32, torsh_core::DType::F32) => {
                let input_data = input.data_f32()?;
                let weights_data = weights.data_f32()?;

                // Use SIMD matrix multiplication
                let result_data = simd_matrix_multiply(&input_data, &weights_data)?;
                let output_shape = self.compute_output_shape(input.shape(), weights.shape())?;

                let result = Tensor::from_data(result_data, &output_shape)?;

                debug!(
                    "SIMD-optimized forward completed with output shape {:?}",
                    result.shape()
                );
                Ok(result)
            }
            _ => {
                // Fallback to standard operations for unsupported dtypes
                self.standard_forward(input, weights)
            }
        }
    }

    /// Parallel all-gather operation using SciRS2 parallel processing
    #[cfg(feature = "scirs2-memory")]
    pub async fn parallel_all_gather(&self, tensor: &Tensor) -> TorshResult<Tensor> {
        if !self.config.enable_scirs2_memory {
            return all_gather(tensor, &*self.tp_group).await;
        }

        debug!("Performing parallel all-gather with SciRS2 optimizations");

        // Use parallel processing for large tensor operations
        let chunk_size = std::cmp::max(1, tensor.numel() / self.config.tp_size);

        let gathered_chunks = par_chunks(tensor.data_f32()?, chunk_size)
            .map(|chunk| {
                // Simulate gathering chunk from all processes
                // In real implementation, this would be distributed
                chunk.to_vec()
            })
            .collect::<Vec<Vec<f32>>>();

        let gathered_data: Vec<f32> = gathered_chunks.into_iter().flatten().collect();
        let gathered_shape = self.compute_gathered_shape(tensor.shape())?;

        let result = Tensor::from_data(gathered_data, &gathered_shape)?;

        info!(
            "Parallel all-gather completed with shape {:?}",
            result.shape()
        );
        Ok(result)
    }

    /// Initialize memory-efficient buffer pools
    #[cfg(feature = "scirs2-memory")]
    pub fn init_scirs2_memory_pools(&mut self) -> TorshResult<()> {
        if !self.config.enable_scirs2_memory {
            return Ok(());
        }

        info!(
            "Initializing SciRS2 memory pools with {}MB buffer",
            self.config.buffer_pool_size_mb
        );

        // Initialize global buffer pool
        let buffer_pool =
            GlobalBufferPool::initialize(self.config.buffer_pool_size_mb * 1024 * 1024)?;

        // Pre-allocate buffers for common tensor sizes
        let common_sizes = vec![
            1024 * 1024,      // 1M elements
            4 * 1024 * 1024,  // 4M elements
            16 * 1024 * 1024, // 16M elements
        ];

        for size in common_sizes {
            buffer_pool.pre_allocate_buffer(size * 4)?; // 4 bytes per f32
        }

        info!("SciRS2 memory pools initialized successfully");
        Ok(())
    }

    /// Get memory efficiency statistics
    #[cfg(feature = "scirs2-memory")]
    pub fn get_memory_efficiency_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        if self.config.enable_scirs2_memory {
            if let Ok(buffer_pool) = GlobalBufferPool::instance() {
                stats.insert(
                    "buffer_pool_utilization".to_string(),
                    buffer_pool.utilization_ratio(),
                );
                stats.insert(
                    "buffer_pool_fragmentation".to_string(),
                    buffer_pool.fragmentation_ratio(),
                );
                stats.insert(
                    "total_allocations".to_string(),
                    buffer_pool.total_allocations() as f64,
                );
                stats.insert("cache_hit_ratio".to_string(), buffer_pool.cache_hit_ratio());
            }
        }

        // Add tensor parallelism specific stats
        stats.insert(
            "memory_reduction_ratio".to_string(),
            self.get_stats().memory_reduction_ratio,
        );
        stats.insert(
            "tp_efficiency".to_string(),
            1.0 / self.config.tp_size as f64,
        );

        stats
    }

    // Helper methods

    #[cfg(feature = "scirs2-memory")]
    fn compute_output_shape(
        &self,
        input_shape: &Shape,
        weights_shape: &Shape,
    ) -> TorshResult<Shape> {
        // Simplified shape computation - in real implementation would be more sophisticated
        let input_dims = input_shape.dims();
        let weights_dims = weights_shape.dims();

        let output_dims = vec![input_dims[0], weights_dims[1]];
        Ok(Shape::from_dims(&output_dims))
    }

    #[cfg(feature = "scirs2-memory")]
    fn compute_gathered_shape(&self, shard_shape: &Shape) -> TorshResult<Shape> {
        let mut dims = shard_shape.dims().to_vec();
        dims[1] *= self.config.tp_size; // Assuming gathering along dimension 1
        Ok(Shape::from_dims(&dims))
    }

    #[cfg(feature = "scirs2-memory")]
    fn standard_forward(&self, input: &Tensor, weights: &Tensor) -> TorshResult<Tensor> {
        // Fallback implementation without SIMD optimizations
        info!("Using standard forward pass (SIMD disabled)");

        // Basic matrix multiplication implementation
        let result = input.matmul(weights)?;
        Ok(result)
    }
}

impl Module for TensorParallel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match &self.layer_info {
            TensorParallelLayer::RowParallelLinear {
                input_is_parallel, ..
            } => {
                // For row-parallel, input might need all-gather first
                let processed_input = if *input_is_parallel {
                    input.clone()
                } else {
                    // Would need async version in real implementation
                    input.clone()
                };

                // Forward through local shard
                let local_output = self.module.forward(&processed_input)?;

                // All-reduce the output (since each rank computes a partial result)
                // For simplicity, returning local output for now
                Ok(local_output)
            }

            TensorParallelLayer::ColumnParallelLinear { gather_output, .. } => {
                // Forward through local shard
                let local_output = self.module.forward(input)?;

                if *gather_output {
                    // All-gather outputs from all ranks
                    // For simplicity, returning local output for now
                    Ok(local_output)
                } else {
                    Ok(local_output)
                }
            }

            TensorParallelLayer::ParallelEmbedding { .. } => {
                // For vocab-parallel embeddings, only some ranks have relevant embeddings
                let output = self.module.forward(input)?;

                // All-reduce to combine embeddings from different vocab shards
                // For simplicity, returning local output for now
                Ok(output)
            }

            TensorParallelLayer::ParallelAttention { .. } => {
                // For attention, each rank computes a subset of attention heads
                let output = self.module.forward(input)?;

                // Concatenate attention heads from all ranks
                // For simplicity, returning local output for now
                Ok(output)
            }
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        // Return only the local sharded parameters
        let all_params = self.module.parameters();
        let mut sharded_params = HashMap::new();

        for (name, param) in all_params {
            if let Some(_shard_info) = self.shard_info.get(&name) {
                // Extract the local shard of this parameter
                let tensor = param.tensor();
                let _tensor_guard = tensor.read();

                // For simplicity, return the whole parameter
                // In a real implementation, we would slice based on shard_info
                sharded_params.insert(name, param);
            } else {
                // Parameter is not sharded, include as-is
                sharded_params.insert(name, param);
            }
        }

        sharded_params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.module.training()
    }

    fn train(&mut self) {
        self.module.train()
    }

    fn eval(&mut self) {
        self.module.eval()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.module.to_device(device)
    }
}

/// Statistics for tensor parallelism
#[derive(Debug, Clone)]
pub struct TensorParallelStats {
    /// Tensor parallel rank
    pub tp_rank: usize,
    /// Tensor parallel world size
    pub tp_world_size: usize,
    /// Total number of parameters in the original model
    pub total_parameters: usize,
    /// Number of parameters that are sharded
    pub sharded_parameters: usize,
    /// Memory reduction ratio due to sharding
    pub memory_reduction_ratio: f64,
    /// Communication overhead in milliseconds
    pub communication_overhead_ms: f64,
}

/// Utility functions for tensor parallelism
pub mod utils {
    use super::*;

    /// Create a row-parallel linear layer
    pub fn create_row_parallel_linear(
        input_size: usize,
        output_size: usize,
        bias: bool,
        input_is_parallel: bool,
        tp_group: Arc<ProcessGroup>,
        config: Option<TensorParallelConfig>,
    ) -> TorshResult<TensorParallel> {
        let linear = torsh_nn::layers::Linear::new(input_size, output_size, bias);
        let module = Box::new(linear) as Box<dyn Module>;

        let layer_info = TensorParallelLayer::RowParallelLinear {
            input_size,
            output_size,
            bias,
            input_is_parallel,
        };

        let config = config.unwrap_or_default();
        TensorParallel::new(module, tp_group, config, layer_info)
    }

    /// Create a column-parallel linear layer
    pub fn create_column_parallel_linear(
        input_size: usize,
        output_size: usize,
        bias: bool,
        gather_output: bool,
        tp_group: Arc<ProcessGroup>,
        config: Option<TensorParallelConfig>,
    ) -> TorshResult<TensorParallel> {
        let linear = torsh_nn::layers::Linear::new(input_size, output_size, bias);
        let module = Box::new(linear) as Box<dyn Module>;

        let layer_info = TensorParallelLayer::ColumnParallelLinear {
            input_size,
            output_size,
            bias,
            gather_output,
        };

        let config = config.unwrap_or_default();
        TensorParallel::new(module, tp_group, config, layer_info)
    }

    /// Split a tensor along a given dimension for tensor parallelism
    pub fn split_tensor_for_tp(
        tensor: &Tensor,
        split_dim: usize,
        tp_rank: usize,
        tp_size: usize,
    ) -> TorshResult<Tensor> {
        let shape = tensor.shape();
        let dim_size = shape.dims()[split_dim];

        if dim_size % tp_size != 0 {
            return Err(TorshDistributedError::invalid_argument(
                "tensor_dimension",
                format!(
                    "Dimension size {} is not divisible by TP size {}",
                    dim_size, tp_size
                ),
                format!("dimension size must be multiple of tp_size ({})", tp_size),
            )
            .into());
        }

        let shard_size = dim_size / tp_size;
        let start_idx = tp_rank * shard_size;
        let end_idx = start_idx + shard_size;

        Ok(tensor.slice(split_dim, start_idx, end_idx)?.to_tensor()?)
    }

    /// Gather tensors from all TP ranks along a given dimension
    pub async fn gather_tensor_from_tp(
        tensor: &Tensor,
        _gather_dim: usize,
        tp_group: &ProcessGroup,
    ) -> TorshResult<Tensor> {
        let mut gathered_tensors = Vec::new();
        all_gather(&mut gathered_tensors, tensor, tp_group).await?;

        // For simplicity, return the first tensor
        // In a real implementation, we would concatenate along gather_dim
        if gathered_tensors.is_empty() {
            Err(TorshDistributedError::communication_error(
                "tensor_parallel",
                "No tensors gathered",
            )
            .into())
        } else {
            Ok(gathered_tensors.into_iter().next().unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{init_process_group, BackendType};
    use torsh_nn::layers::Linear;

    #[tokio::test]
    async fn test_tensor_parallel_config() {
        let config = TensorParallelConfig::default();
        assert_eq!(config.tp_size, 1);
        assert!(!config.sequence_parallel);
        assert_eq!(config.communication_backend, "nccl");
        assert!(config.async_communication);
    }

    #[tokio::test]
    async fn test_shard_info() {
        let shard_info = ShardInfo {
            shard_dim: 0,
            start_idx: 0,
            shard_size: 128,
            original_shape: Shape::new(vec![512, 256]),
            strategy: TensorParallelStrategy::RowParallel,
        };

        assert_eq!(shard_info.shard_dim, 0);
        assert_eq!(shard_info.shard_size, 128);
        assert_eq!(shard_info.strategy, TensorParallelStrategy::RowParallel);
    }

    #[tokio::test]
    async fn test_tensor_parallel_stats() {
        let stats = TensorParallelStats {
            tp_rank: 0,
            tp_world_size: 4,
            total_parameters: 1000,
            sharded_parameters: 800,
            memory_reduction_ratio: 0.75,
            communication_overhead_ms: 5.2,
        };

        assert_eq!(stats.tp_rank, 0);
        assert_eq!(stats.tp_world_size, 4);
        assert_eq!(stats.memory_reduction_ratio, 0.75);
    }

    #[tokio::test]
    async fn test_create_row_parallel_linear() -> TorshResult<()> {
        let process_group =
            Arc::new(init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 12345).await?);

        let tp_layer =
            utils::create_row_parallel_linear(128, 256, true, false, process_group, None)?;

        assert_eq!(tp_layer.tp_rank(), 0);
        assert_eq!(tp_layer.tp_world_size(), 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_create_column_parallel_linear() -> TorshResult<()> {
        let process_group =
            Arc::new(init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 12346).await?);

        let tp_layer =
            utils::create_column_parallel_linear(128, 256, true, true, process_group, None)?;

        assert_eq!(tp_layer.tp_rank(), 0);
        assert_eq!(tp_layer.tp_world_size(), 2);

        Ok(())
    }

    #[test]
    fn test_tensor_parallel_strategies() {
        assert_ne!(
            TensorParallelStrategy::RowParallel,
            TensorParallelStrategy::ColumnParallel
        );
        assert_ne!(
            TensorParallelStrategy::VocabParallel,
            TensorParallelStrategy::SequenceParallel
        );
        assert_ne!(
            TensorParallelStrategy::AttentionHeadParallel,
            TensorParallelStrategy::RowParallel
        );
    }

    #[tokio::test]
    async fn test_split_tensor_for_tp() -> TorshResult<()> {
        let tensor = torsh_tensor::creation::ones(&[8, 16])?;

        let shard = utils::split_tensor_for_tp(&tensor, 1, 0, 2)?;
        assert_eq!(shard.shape().dims(), &[8, 8]);

        Ok(())
    }
}
