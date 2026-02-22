//! Distributed gradient accumulator for efficient gradient computation and communication
//!
//! This module provides the core gradient accumulation functionality with support for
//! multiple communication patterns and compression strategies.

use crate::compression::{
    CompressedGradient, CompressionAlgorithm, CompressionConfig, CompressionMetadata,
    GradientCompressor,
};
use crate::simd_ops::{F32SimdAccumulator, SimdGradAccumulator, SimdLevel};
use scirs2_core::numeric::{FromPrimitive, ToPrimitive};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use torsh_core::dtype::FloatElement;
use torsh_core::error::{Result, TorshError};

use super::config::{
    CommunicationPattern, CompressionStrategy, DistributedConfig, DistributedStats, ReductionOp,
};

/// Distributed gradient accumulator with advanced features
pub struct DistributedGradAccumulator<T: FloatElement> {
    /// Configuration
    config: DistributedConfig,
    /// Local gradient accumulators per parameter group
    local_accumulators: HashMap<String, Box<dyn SimdGradAccumulator<T>>>,
    /// Gradient buckets for communication efficiency
    gradient_buckets: Vec<GradientBucket<T>>,
    /// Current accumulation step
    accumulation_step: usize,
    /// Communication context
    #[allow(dead_code)]
    comm_context: Arc<Mutex<Option<CommunicationContext>>>,
    /// Statistics tracking
    stats: Arc<RwLock<DistributedStats>>,
    /// Gradient compressor for bandwidth optimization
    #[allow(dead_code)]
    compressor: GradientCompressor<T>,
}

/// Gradient bucket for efficient communication
#[derive(Debug)]
struct GradientBucket<T: FloatElement> {
    /// Bucket identifier
    #[allow(dead_code)]
    id: usize,
    /// Parameter names in this bucket
    parameter_names: Vec<String>,
    /// Flattened gradient data
    data: Vec<T>,
    /// Sizes of individual gradients
    gradient_sizes: Vec<usize>,
    /// Whether this bucket is ready for communication
    ready: bool,
    /// SIMD level for this bucket
    #[allow(dead_code)]
    simd_level: SimdLevel,
}

// GradientBucket is Send + Sync when T is Send + Sync
unsafe impl<T: FloatElement + Send> Send for GradientBucket<T> {}
unsafe impl<T: FloatElement + Send + Sync> Sync for GradientBucket<T> {}

/// Communication context for distributed operations
#[derive(Debug)]
struct CommunicationContext {
    /// Backend-specific handle
    #[allow(dead_code)]
    backend_handle: usize,
    /// Communication streams/contexts
    #[allow(dead_code)]
    streams: Vec<usize>,
    /// Pending operations
    #[allow(dead_code)]
    pending_ops: Vec<PendingOperation>,
}

// CommunicationContext is Send + Sync
unsafe impl Send for CommunicationContext {}
unsafe impl Sync for CommunicationContext {}

/// Pending communication operation
#[derive(Debug)]
struct PendingOperation {
    /// Operation identifier
    #[allow(dead_code)]
    id: usize,
    /// Operation type
    #[allow(dead_code)]
    op_type: CommunicationPattern,
    /// Start time
    #[allow(dead_code)]
    start_time: Instant,
    /// Data size in bytes
    #[allow(dead_code)]
    data_size: usize,
}

// PendingOperation is Send + Sync
unsafe impl Send for PendingOperation {}
unsafe impl Sync for PendingOperation {}

// DistributedGradAccumulator is Send + Sync when T is Send + Sync
unsafe impl<T: FloatElement + FromPrimitive + ToPrimitive + Send> Send
    for DistributedGradAccumulator<T>
{
}
unsafe impl<T: FloatElement + FromPrimitive + ToPrimitive + Send + Sync> Sync
    for DistributedGradAccumulator<T>
{
}

impl<T: FloatElement + FromPrimitive + ToPrimitive> DistributedGradAccumulator<T> {
    /// Create a new distributed gradient accumulator
    pub fn new(config: DistributedConfig) -> Result<Self> {
        // Create compression config based on distributed strategy
        let compression_config = match config.compression {
            CompressionStrategy::None => CompressionConfig {
                algorithm: CompressionAlgorithm::None,
                ..Default::default()
            },
            CompressionStrategy::Quantization => CompressionConfig {
                algorithm: CompressionAlgorithm::Quantization8Bit,
                target_ratio: 0.25,
                ..Default::default()
            },
            CompressionStrategy::Sparsification => CompressionConfig {
                algorithm: CompressionAlgorithm::TopKSparsification,
                sparsity_threshold: 0.01,
                ..Default::default()
            },
            CompressionStrategy::ErrorFeedback => CompressionConfig {
                algorithm: CompressionAlgorithm::ErrorFeedback,
                ..Default::default()
            },
            CompressionStrategy::Sketching => CompressionConfig {
                algorithm: CompressionAlgorithm::GradientSketching,
                ..Default::default()
            },
        };

        let mut accumulator = Self {
            config,
            local_accumulators: HashMap::new(),
            gradient_buckets: Vec::new(),
            accumulation_step: 0,
            comm_context: Arc::new(Mutex::new(None)),
            stats: Arc::new(RwLock::new(DistributedStats::default())),
            compressor: GradientCompressor::new(compression_config),
        };

        accumulator.initialize_communication()?;
        Ok(accumulator)
    }

    /// Initialize communication backend
    fn initialize_communication(&mut self) -> Result<()> {
        use super::config::DistributedBackend;

        match self.config.backend {
            DistributedBackend::None => {
                // Single device mode - no communication needed
                Ok(())
            }
            DistributedBackend::Nccl => self.initialize_nccl(),
            DistributedBackend::Gloo => self.initialize_gloo(),
            DistributedBackend::Mpi => self.initialize_mpi(),
            DistributedBackend::Custom => self.initialize_custom(),
        }
    }

    /// Initialize NCCL backend
    fn initialize_nccl(&mut self) -> Result<()> {
        // Placeholder for NCCL initialization
        // In a real implementation, this would:
        // 1. Initialize NCCL communicator
        // 2. Set up CUDA streams
        // 3. Configure NCCL topology

        tracing::info!("Initializing NCCL backend (placeholder implementation)");
        Ok(())
    }

    /// Initialize Gloo backend
    fn initialize_gloo(&mut self) -> Result<()> {
        // Placeholder for Gloo initialization
        // In a real implementation, this would:
        // 1. Set up TCP/InfiniBand connections
        // 2. Create process group
        // 3. Configure rendezvous

        tracing::info!("Initializing Gloo backend (placeholder implementation)");
        Ok(())
    }

    /// Initialize MPI backend
    fn initialize_mpi(&mut self) -> Result<()> {
        // Placeholder for MPI initialization
        // In a real implementation, this would:
        // 1. Initialize MPI environment
        // 2. Get rank and world size
        // 3. Set up MPI communicators

        tracing::info!("Initializing MPI backend (placeholder implementation)");
        Ok(())
    }

    /// Initialize custom backend
    fn initialize_custom(&mut self) -> Result<()> {
        // Placeholder for custom backend initialization
        tracing::info!("Initializing custom backend (placeholder implementation)");
        Ok(())
    }

    /// Register a parameter group for gradient accumulation
    pub fn register_parameter_group(&mut self, name: String, size: usize) -> Result<()> {
        // Create appropriate accumulator based on type
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => {
                let accumulator = F32SimdAccumulator::new(size);
                let boxed: Box<dyn SimdGradAccumulator<T>> = unsafe {
                    std::mem::transmute(Box::new(accumulator) as Box<dyn SimdGradAccumulator<f32>>)
                };
                self.local_accumulators.insert(name.clone(), boxed);
            }
            _ => {
                return Err(TorshError::AutogradError(
                    "Unsupported element type for distributed accumulation".to_string(),
                ));
            }
        }

        // Add to appropriate bucket
        self.add_to_bucket(name, size)?;

        Ok(())
    }

    /// Add parameter to a gradient bucket
    fn add_to_bucket(&mut self, parameter_name: String, size: usize) -> Result<()> {
        let element_size = std::mem::size_of::<T>();
        let parameter_bytes = size * element_size;

        // Find bucket with space or create new one
        let bucket_idx = self.find_or_create_bucket(parameter_bytes)?;
        self.gradient_buckets[bucket_idx]
            .parameter_names
            .push(parameter_name);
        self.gradient_buckets[bucket_idx].gradient_sizes.push(size);

        Ok(())
    }

    /// Find bucket with space or create a new one
    fn find_or_create_bucket(&mut self, required_bytes: usize) -> Result<usize> {
        // Try to find existing bucket with space
        for (idx, bucket) in self.gradient_buckets.iter().enumerate() {
            let current_size = std::mem::size_of_val(&bucket.data);
            if current_size + required_bytes <= self.config.bucket_size {
                return Ok(idx);
            }
        }

        // Create new bucket
        let bucket_id = self.gradient_buckets.len();
        let new_bucket = GradientBucket {
            id: bucket_id,
            parameter_names: Vec::new(),
            data: Vec::new(),
            gradient_sizes: Vec::new(),
            ready: false,
            simd_level: crate::simd_ops::detect_simd_level(),
        };

        self.gradient_buckets.push(new_bucket);
        Ok(bucket_id)
    }

    /// Accumulate gradients for a parameter group
    pub fn accumulate_gradients(&mut self, parameter_name: &str, gradients: &[T]) -> Result<()> {
        // Get local accumulator
        let accumulator = self
            .local_accumulators
            .get_mut(parameter_name)
            .ok_or_else(|| {
                TorshError::AutogradError(format!(
                    "Parameter group '{parameter_name}' not registered"
                ))
            })?;

        // Accumulate locally with SIMD optimization
        accumulator.accumulate_simd(gradients, None)?;

        // Update accumulation step
        self.accumulation_step += 1;

        // Check if we need to synchronize
        if self.accumulation_step % self.config.gradient_accumulation_steps == 0 {
            self.synchronize_gradients()?;
        }

        Ok(())
    }

    /// Synchronize gradients across all devices
    pub fn synchronize_gradients(&mut self) -> Result<()> {
        if self.config.world_size <= 1 {
            // Single device - no synchronization needed
            return Ok(());
        }

        let start_time = Instant::now();

        // Prepare gradient buckets
        self.prepare_buckets()?;

        // Communicate gradients
        match self.config.communication_pattern {
            CommunicationPattern::AllReduce => self.all_reduce()?,
            CommunicationPattern::ReduceScatter => self.reduce_scatter()?,
            CommunicationPattern::AllGather => self.all_gather()?,
            CommunicationPattern::ParameterServer => self.parameter_server_sync()?,
            CommunicationPattern::Ring => self.ring_reduce()?,
            CommunicationPattern::Tree => self.tree_reduce()?,
        }

        // Update statistics
        let sync_time = start_time.elapsed();
        let mut stats = self.stats.write().expect("lock should not be poisoned");
        stats.total_communications += 1;
        stats.total_comm_time += sync_time;
        stats.sync_overhead += sync_time;

        Ok(())
    }

    /// Prepare gradient buckets for communication
    fn prepare_buckets(&mut self) -> Result<()> {
        for bucket in &mut self.gradient_buckets {
            // Flatten gradients into bucket
            bucket.data.clear();

            for param_name in &bucket.parameter_names {
                if let Some(accumulator) = self.local_accumulators.get(param_name) {
                    let accumulated = accumulator.get_accumulated();

                    // Convert to our element type (this is a simplified conversion)
                    for &val in accumulated {
                        // This is a placeholder - proper type conversion would be needed
                        let converted_val = unsafe { std::mem::transmute_copy(&val) };
                        bucket.data.push(converted_val);
                    }
                }
            }

            bucket.ready = true;
        }

        Ok(())
    }

    /// All-reduce implementation
    fn all_reduce(&mut self) -> Result<()> {
        tracing::debug!(
            "Performing all-reduce operation with compression strategy: {:?}",
            self.config.compression
        );

        let reduction_op = self.config.reduction_op;
        let world_size = self.config.world_size;

        // Process buckets in two phases to avoid borrow checker issues
        let mut processed_data = Vec::new();

        // Phase 1: Compression (read-only access to buckets)
        for (_i, bucket) in self.gradient_buckets.iter().enumerate() {
            if !bucket.ready {
                processed_data.push(None);
                continue;
            }

            let start_time = std::time::Instant::now();

            // Step 1: Apply compression using static method to avoid mutable borrow
            let compressed = match self.config.compression {
                CompressionStrategy::None => {
                    // No compression - create a dummy compressed gradient
                    let data = unsafe {
                        std::slice::from_raw_parts(
                            bucket.data.as_ptr() as *const u8,
                            std::mem::size_of_val(&bucket.data),
                        )
                    };
                    CompressedGradient {
                        original_shape: vec![bucket.data.len()],
                        data: data.to_vec(),
                        metadata: CompressionMetadata {
                            scale: 1.0,
                            zero_point: 0,
                            indices: Vec::new(),
                            seed: 0,
                            rank: 0,
                            error: Vec::new(),
                        },
                        algorithm: CompressionAlgorithm::None,
                    }
                }
                _ => {
                    // Use compressor for actual compression
                    let mut temp_compressor = GradientCompressor::new(CompressionConfig {
                        algorithm: CompressionAlgorithm::TopKSparsification, // Default algorithm
                        ..Default::default()
                    });
                    temp_compressor.compress(&bucket.data, "bucket")?
                }
            };

            let original_size = std::mem::size_of_val(&bucket.data);
            let compressed_size = compressed.data.len();
            let compression_ratio = compressed_size as f64 / original_size as f64;

            tracing::debug!(
                "Compressed gradient data from {} bytes to {} bytes (ratio: {:.2})",
                original_size,
                compressed_size,
                compression_ratio
            );

            // Step 2: Simulate communication delay based on compressed size
            let comm_latency = Duration::from_micros((compressed_size / 1024).max(1) as u64 * 10);
            std::thread::sleep(comm_latency);

            processed_data.push(Some((compressed, start_time)));
        }

        // Phase 2: Decompression and updating buckets (mutable access)
        for (_i, (bucket, processed)) in self
            .gradient_buckets
            .iter_mut()
            .zip(processed_data.iter())
            .enumerate()
        {
            if let Some((compressed, start_time)) = processed {
                // Step 3: Decompress received gradients
                let decompressed = match self.config.compression {
                    CompressionStrategy::None => {
                        // Simple byte copy back for no compression
                        let data_ptr = compressed.data.as_ptr() as *const T;
                        let data_len = compressed.data.len() / std::mem::size_of::<T>();
                        unsafe { std::slice::from_raw_parts(data_ptr, data_len) }.to_vec()
                    }
                    _ => {
                        let mut temp_compressor = GradientCompressor::new(CompressionConfig {
                            algorithm: CompressionAlgorithm::TopKSparsification,
                            ..Default::default()
                        });
                        temp_compressor.decompress(compressed)?
                    }
                };
                bucket.data = decompressed;

                // Step 4: Apply reduction operation
                Self::apply_reduction_static(&mut bucket.data, reduction_op, world_size)?;

                let total_time = start_time.elapsed();
                let compressed_size = compressed.data.len();
                let original_size = std::mem::size_of_val(&bucket.data);
                let compression_ratio = compressed_size as f64 / original_size as f64;

                // Update statistics
                let mut stats = self.stats.write().expect("lock should not be poisoned");
                stats.total_communications += 1;
                stats.total_data_communicated += compressed_size; // Use compressed size for bandwidth calculation
                stats.total_comm_time += total_time;
                stats.compression_ratio = compression_ratio;

                // Calculate bandwidth based on compressed data
                let data_mb = compressed_size as f64 / (1024.0 * 1024.0);
                let comm_latency =
                    Duration::from_micros((compressed_size / 1024).max(1) as u64 * 10);
                let time_s = comm_latency.as_secs_f64();
                let bandwidth = if time_s > 0.0 { data_mb / time_s } else { 0.0 };
                stats.avg_bandwidth_mbps = (stats.avg_bandwidth_mbps + bandwidth) / 2.0;
                // Running average
            }
        }

        Ok(())
    }

    /// Reduce-scatter implementation
    fn reduce_scatter(&mut self) -> Result<()> {
        tracing::debug!("Performing reduce-scatter operation");

        let world_size = self.config.world_size;
        let rank = self.config.rank;

        for bucket in &mut self.gradient_buckets {
            if !bucket.ready {
                continue;
            }

            // In reduce-scatter, each rank gets a portion of the reduced result
            let chunk_size = bucket.data.len() / world_size;
            let start_idx = rank * chunk_size;
            let end_idx = if rank == world_size - 1 {
                bucket.data.len() // Last rank gets remainder
            } else {
                start_idx + chunk_size
            };

            // Simulate reduction for our chunk
            for i in start_idx..end_idx {
                if let Some(val) = bucket.data.get_mut(i) {
                    // Apply reduction - for simulation, scale by world size
                    *val = *val * T::from_usize(world_size).expect("numeric conversion should succeed");

                    // Apply division for mean operation
                    if self.config.reduction_op == ReductionOp::Mean {
                        *val = *val / T::from_usize(world_size).expect("numeric conversion should succeed");
                    }
                }
            }

            // Keep only our chunk (in real implementation, other chunks would be on other ranks)
            let our_chunk: Vec<T> = bucket.data[start_idx..end_idx].to_vec();
            bucket.data = our_chunk;
        }

        Ok(())
    }

    /// All-gather implementation
    fn all_gather(&mut self) -> Result<()> {
        tracing::debug!("Performing all-gather operation");

        let world_size = self.config.world_size;

        for bucket in &mut self.gradient_buckets {
            if !bucket.ready {
                continue;
            }

            // In all-gather, we simulate gathering data from all ranks
            let original_size = bucket.data.len();
            let mut gathered_data = Vec::with_capacity(original_size * world_size);

            // Simulate gathering data from each rank (including ourselves)
            for rank in 0..world_size {
                // For simulation, we'll just replicate our data with slight variations
                for (_i, &val) in bucket.data.iter().enumerate() {
                    let simulated_val = if rank == self.config.rank {
                        val // Our actual data
                    } else {
                        // Simulate slightly different data from other ranks
                        val * <T as torsh_core::TensorElement>::from_f64(1.0 + (rank as f64 * 0.01))
                            .expect("f64 conversion should succeed")
                    };
                    gathered_data.push(simulated_val);
                }
            }

            bucket.data = gathered_data;

            // Update statistics
            let mut stats = self.stats.write().expect("lock should not be poisoned");
            stats.total_data_communicated += std::mem::size_of_val(&bucket.data);
        }

        Ok(())
    }

    /// Parameter server synchronization
    fn parameter_server_sync(&mut self) -> Result<()> {
        tracing::debug!("Performing parameter server synchronization");

        let is_server = self.config.rank == 0; // Rank 0 acts as parameter server

        for bucket in &mut self.gradient_buckets {
            if !bucket.ready {
                continue;
            }

            if is_server {
                // Server side: aggregate gradients from all workers
                // For simulation, we'll apply the reduction operation
                match self.config.reduction_op {
                    ReductionOp::Sum => {
                        // Server accumulates gradients from all workers
                        for val in bucket.data.iter_mut() {
                            *val = *val * T::from_usize(self.config.world_size).expect("numeric conversion should succeed");
                        }
                    }
                    ReductionOp::Mean => {
                        // Server averages gradients from all workers
                        // For simulation, we don't modify the data since it's already averaged
                    }
                    ReductionOp::Max | ReductionOp::Min => {
                        // Server finds max/min across workers
                        // For simulation, we keep the data as-is
                    }
                }

                // Send updated parameters back to workers (simulated)
                tracing::debug!("Parameter server sending updated parameters to workers");
            } else {
                // Worker side: send gradients to server and receive updated parameters
                tracing::debug!(
                    "Worker {} sending gradients to parameter server",
                    self.config.rank
                );

                // Simulate receiving updated parameters from server
                // In a real implementation, this would wait for server response
                std::thread::sleep(Duration::from_micros(50)); // Simulate network latency
            }

            // Update statistics
            let mut stats = self.stats.write().expect("lock should not be poisoned");
            stats.total_data_communicated += std::mem::size_of_val(&bucket.data);
        }

        Ok(())
    }

    /// Ring reduce implementation
    fn ring_reduce(&mut self) -> Result<()> {
        tracing::debug!("Performing ring reduce operation");

        let world_size = self.config.world_size;
        let rank = self.config.rank;

        if world_size <= 1 {
            return Ok(());
        }

        for bucket in &mut self.gradient_buckets {
            if !bucket.ready {
                continue;
            }

            // Ring reduce operates in two phases:
            // 1. Reduce-scatter phase: each rank reduces a chunk
            // 2. All-gather phase: each rank gathers all chunks

            let chunk_size = bucket.data.len() / world_size;

            // Phase 1: Reduce-scatter
            for step in 0..world_size - 1 {
                let send_chunk = (rank + world_size - step) % world_size;
                let recv_chunk = (rank + world_size - step - 1) % world_size;

                let send_start = send_chunk * chunk_size;
                let _send_end = if send_chunk == world_size - 1 {
                    bucket.data.len()
                } else {
                    send_start + chunk_size
                };

                // Simulate sending to next rank and receiving from previous rank
                tracing::debug!(
                    "Ring reduce step {}: rank {} processing chunk {}",
                    step,
                    rank,
                    recv_chunk
                );

                // Apply reduction to received chunk (simulated)
                let recv_start = recv_chunk * chunk_size;
                let recv_end = if recv_chunk == world_size - 1 {
                    bucket.data.len()
                } else {
                    recv_start + chunk_size
                };

                for i in recv_start..recv_end {
                    if let Some(val) = bucket.data.get_mut(i) {
                        // Simulate accumulating gradients from neighbor
                        *val = *val
                            + (*val * <T as torsh_core::TensorElement>::from_f64(0.1).expect("f64 conversion should succeed"));
                        // Small increment to simulate accumulation
                    }
                }

                // Simulate communication latency
                std::thread::sleep(Duration::from_micros(10));
            }

            // Phase 2: All-gather
            for step in 0..world_size - 1 {
                let send_chunk = (rank + 1 - step + world_size) % world_size;
                let _recv_chunk = (rank - step + world_size) % world_size;

                tracing::debug!(
                    "Ring all-gather step {}: rank {} sharing chunk {}",
                    step,
                    rank,
                    send_chunk
                );

                // Simulate sharing final reduced values around the ring
                std::thread::sleep(Duration::from_micros(10));
            }

            // Apply final reduction operation
            match self.config.reduction_op {
                ReductionOp::Mean => {
                    for val in bucket.data.iter_mut() {
                        *val = *val / T::from_usize(world_size).expect("numeric conversion should succeed");
                    }
                }
                _ => {} // Other ops handled during reduce phase
            }

            // Update statistics
            let mut stats = self.stats.write().expect("lock should not be poisoned");
            stats.total_data_communicated += std::mem::size_of_val(&bucket.data) * 2;
            // Two phases
        }

        Ok(())
    }

    /// Tree reduce implementation
    fn tree_reduce(&mut self) -> Result<()> {
        tracing::debug!("Performing tree reduce operation");

        let world_size = self.config.world_size;
        let rank = self.config.rank;

        if world_size <= 1 {
            return Ok(());
        }

        for bucket in &mut self.gradient_buckets {
            if !bucket.ready {
                continue;
            }

            // Tree reduce uses a binary tree structure
            // Each rank has at most two children: 2*rank+1 and 2*rank+2
            // Parent of rank is (rank-1)/2

            let tree_height = (world_size as f64).log2().ceil() as usize;

            // Phase 1: Reduce up the tree (leaf to root)
            for level in (0..tree_height).rev() {
                let level_start = (1 << level) - 1;
                let level_size = 1 << level;

                if rank >= level_start && rank < level_start + level_size {
                    // This rank is active at this level
                    let _local_rank = rank - level_start;
                    let child1 = 2 * rank + 1;
                    let child2 = 2 * rank + 2;

                    // Receive from children if they exist
                    if child1 < world_size {
                        tracing::debug!("Rank {} receiving from child {}", rank, child1);

                        // Simulate receiving and accumulating gradients from child1
                        for val in bucket.data.iter_mut() {
                            *val = *val
                                + (*val * <T as torsh_core::TensorElement>::from_f64(0.5).expect("f64 conversion should succeed"));
                            // Simulate child contribution
                        }

                        std::thread::sleep(Duration::from_micros(5));
                    }

                    if child2 < world_size {
                        tracing::debug!("Rank {} receiving from child {}", rank, child2);

                        // Simulate receiving and accumulating gradients from child2
                        for val in bucket.data.iter_mut() {
                            *val = *val
                                + (*val * <T as torsh_core::TensorElement>::from_f64(0.5).expect("f64 conversion should succeed"));
                            // Simulate child contribution
                        }

                        std::thread::sleep(Duration::from_micros(5));
                    }

                    // Send to parent if not root
                    if rank > 0 {
                        let parent = (rank - 1) / 2;
                        tracing::debug!("Rank {} sending to parent {}", rank, parent);
                        std::thread::sleep(Duration::from_micros(5));
                    }
                }
            }

            // Phase 2: Broadcast down the tree (root to leaf)
            for level in 0..tree_height {
                let level_start = (1 << level) - 1;
                let level_size = 1 << level;

                if rank >= level_start && rank < level_start + level_size {
                    // This rank is active at this level
                    let child1 = 2 * rank + 1;
                    let child2 = 2 * rank + 2;

                    // Send to children if they exist
                    if child1 < world_size {
                        tracing::debug!("Rank {} broadcasting to child {}", rank, child1);
                        std::thread::sleep(Duration::from_micros(5));
                    }

                    if child2 < world_size {
                        tracing::debug!("Rank {} broadcasting to child {}", rank, child2);
                        std::thread::sleep(Duration::from_micros(5));
                    }
                }
            }

            // Apply final reduction operation
            match self.config.reduction_op {
                ReductionOp::Mean => {
                    for val in bucket.data.iter_mut() {
                        *val = *val / T::from_usize(world_size).expect("numeric conversion should succeed");
                    }
                }
                ReductionOp::Sum => {
                    // Sum is accumulated during the tree traversal
                }
                ReductionOp::Max | ReductionOp::Min => {
                    // For tree reduce, these would need element-wise comparison
                    // For simulation, we keep the data as-is
                }
            }

            // Update statistics
            let mut stats = self.stats.write().expect("lock should not be poisoned");
            stats.total_data_communicated += std::mem::size_of_val(&bucket.data) * tree_height * 2;
            // Up and down
        }

        Ok(())
    }

    /// Compress gradients according to strategy (legacy method)
    #[allow(dead_code)]
    fn compress_gradients_legacy(&self, data: &[T]) -> Result<Vec<u8>> {
        Self::compress_gradients_static(data, self.config.compression)
    }

    /// Compress gradients using the configured compressor
    #[allow(dead_code)]
    fn compress_gradients(&mut self, data: &[T]) -> Result<CompressedGradient> {
        let compressed = self.compressor.compress(data, "gradient_bucket")?;

        // Update compression statistics
        let original_size = std::mem::size_of_val(data);
        let compressed_size = compressed.data.len();
        let compression_ratio = compressed_size as f64 / original_size as f64;

        {
            let mut stats = self.stats.write().expect("lock should not be poisoned");
            stats.compression_ratio = (stats.compression_ratio + compression_ratio) / 2.0;
            // Running average
        }

        Ok(compressed)
    }

    /// Decompress gradients using the configured compressor
    #[allow(dead_code)]
    fn decompress_gradients(&mut self, compressed: &CompressedGradient) -> Result<Vec<T>> {
        self.compressor.decompress(compressed)
    }

    /// Static version of compress_gradients for borrow checker
    fn compress_gradients_static(data: &[T], compression: CompressionStrategy) -> Result<Vec<u8>> {
        // Create temporary compressor for static compression
        let compression_config = match compression {
            CompressionStrategy::None => CompressionConfig {
                algorithm: CompressionAlgorithm::None,
                ..Default::default()
            },
            CompressionStrategy::Quantization => CompressionConfig {
                algorithm: CompressionAlgorithm::Quantization8Bit,
                target_ratio: 0.25,
                ..Default::default()
            },
            CompressionStrategy::Sparsification => CompressionConfig {
                algorithm: CompressionAlgorithm::TopKSparsification,
                sparsity_threshold: 0.01,
                ..Default::default()
            },
            CompressionStrategy::ErrorFeedback => CompressionConfig {
                algorithm: CompressionAlgorithm::ErrorFeedback,
                ..Default::default()
            },
            CompressionStrategy::Sketching => CompressionConfig {
                algorithm: CompressionAlgorithm::GradientSketching,
                ..Default::default()
            },
        };

        let mut compressor = GradientCompressor::new(compression_config);
        let compressed = compressor.compress(data, "sketching_data")?;
        Ok(compressed.data)
    }

    /// Apply compression to gradient bucket data
    #[allow(dead_code)]
    fn apply_compression_to_bucket(
        &mut self,
        bucket: &mut GradientBucket<T>,
    ) -> Result<CompressedGradient> {
        if self.config.compression == CompressionStrategy::None {
            // No compression - create a dummy compressed gradient
            let data = unsafe {
                std::slice::from_raw_parts(
                    bucket.data.as_ptr() as *const u8,
                    std::mem::size_of_val(&bucket.data),
                )
            };
            return Ok(CompressedGradient {
                original_shape: vec![bucket.data.len()],
                data: data.to_vec(),
                metadata: crate::compression::CompressionMetadata::default(),
                algorithm: CompressionAlgorithm::None,
            });
        }

        self.compress_gradients(&bucket.data)
    }

    /// Apply decompression to received gradient data
    #[allow(dead_code)]
    fn apply_decompression(&mut self, compressed: &CompressedGradient) -> Result<Vec<T>> {
        self.decompress_gradients(compressed)
    }

    /// Apply reduction operation to gradient data
    #[allow(dead_code)]
    fn apply_reduction(&self, data: &mut [T]) -> Result<()> {
        Self::apply_reduction_static(data, self.config.reduction_op, self.config.world_size)
    }

    /// Static version of apply_reduction for borrow checker
    fn apply_reduction_static(
        data: &mut [T],
        reduction_op: ReductionOp,
        world_size: usize,
    ) -> Result<()> {
        match reduction_op {
            ReductionOp::Sum => {
                // For simulation, we just multiply by world size to simulate sum
                // In real implementation, this would be the result of actual communication
                for val in data.iter_mut() {
                    *val = *val * T::from_usize(world_size).expect("numeric conversion should succeed");
                }
            }
            ReductionOp::Mean => {
                // For mean, we don't need to do anything in this simulation
                // In real implementation, sum would be divided by world size
            }
            ReductionOp::Max | ReductionOp::Min => {
                // Placeholder - would need actual communication to implement
            }
        }

        Ok(())
    }

    /// Get accumulated gradients for a parameter
    pub fn get_accumulated_gradients(&self, parameter_name: &str) -> Result<&[T]> {
        let accumulator = self.local_accumulators.get(parameter_name).ok_or_else(|| {
            TorshError::AutogradError(format!("Parameter group '{parameter_name}' not found"))
        })?;

        Ok(accumulator.get_accumulated())
    }

    /// Reset all accumulators
    pub fn reset(&mut self) {
        for accumulator in self.local_accumulators.values_mut() {
            accumulator.reset();
        }

        for bucket in &mut self.gradient_buckets {
            bucket.data.clear();
            bucket.ready = false;
        }

        self.accumulation_step = 0;
    }

    /// Get distributed training statistics
    pub fn get_stats(&self) -> DistributedStats {
        self.stats.read().expect("lock should not be poisoned").clone()
    }

    /// Check if this is the master rank
    pub fn is_master(&self) -> bool {
        self.config.rank == 0
    }

    /// Get world size
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }

    /// Get local rank
    pub fn rank(&self) -> usize {
        self.config.rank
    }
}