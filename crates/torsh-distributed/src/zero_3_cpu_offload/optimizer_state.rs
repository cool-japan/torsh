//! Optimizer State Management for ZeRO-3 CPU Offloading
//!
//! This module implements optimizer state management for ZeRO-3 (Zero Redundancy
//! Optimizer Stage 3) with CPU offloading capabilities. It handles the storage,
//! retrieval, and synchronization of optimizer states (momentum, variance, etc.)
//! across distributed workers while supporting offloading to CPU memory.

use crate::{TorshDistributedError, TorshResult};
use log::{debug, info, warn};
use std::collections::HashMap;
use std::sync::RwLock;
use torsh_tensor::Tensor;

use super::config::Zero3CpuOffloadConfig;

/// ZeRO-3 rank mapping for parameter partitioning
#[derive(Debug, Clone)]
pub struct Zero3RankMapping {
    pub rank: usize,
    pub world_size: usize,
}

impl Zero3RankMapping {
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }

    pub fn owns_partition(&self, partition_idx: usize) -> bool {
        partition_idx % self.world_size == self.rank
    }

    pub fn get_parameter_owner(&self, param_idx: usize) -> usize {
        param_idx % self.world_size
    }
}

/// Optimizer state manager for ZeRO-3
///
/// Manages the storage and retrieval of optimizer states across distributed workers.
/// Supports CPU offloading to reduce GPU memory usage for large models.
/// Each rank owns a subset of optimizer states according to ZeRO-3 partitioning.
pub struct OptimizerStateManager {
    /// Configuration for ZeRO-3 CPU offloading
    config: Zero3CpuOffloadConfig,
    /// Rank mapping for parameter partitioning
    rank_mapping: Zero3RankMapping,
    /// Storage for optimizer states indexed by parameter name
    optimizer_states: RwLock<HashMap<String, OptimizerState>>,
    /// CPU storage for offloaded optimizer states
    cpu_optimizer_states: RwLock<HashMap<String, CpuOptimizerState>>,
    /// GPU cache for frequently accessed optimizer states
    gpu_optimizer_cache: RwLock<HashMap<String, OptimizerState>>,
    /// Memory usage tracking
    memory_used_cpu: std::sync::atomic::AtomicUsize,
    memory_used_gpu: std::sync::atomic::AtomicUsize,
}

impl OptimizerStateManager {
    /// Create a new optimizer state manager
    pub fn new(
        config: &Zero3CpuOffloadConfig,
        rank_mapping: &Zero3RankMapping,
    ) -> TorshResult<Self> {
        Ok(Self {
            config: config.clone(),
            rank_mapping: rank_mapping.clone(),
            optimizer_states: RwLock::new(HashMap::new()),
            cpu_optimizer_states: RwLock::new(HashMap::new()),
            gpu_optimizer_cache: RwLock::new(HashMap::new()),
            memory_used_cpu: std::sync::atomic::AtomicUsize::new(0),
            memory_used_gpu: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Fetch optimizer state for a parameter
    ///
    /// First checks GPU cache, then CPU storage, creates new state if not found.
    /// Automatically manages memory by promoting frequently accessed states to GPU.
    pub async fn fetch_state(&self, param_name: &str) -> TorshResult<OptimizerState> {
        // Check if this rank owns the optimizer state for this parameter
        let param_hash = self.hash_parameter_name(param_name);
        let owner_rank = param_hash % self.rank_mapping.world_size;

        if owner_rank != self.rank_mapping.rank {
            return Err(TorshDistributedError::invalid_argument(
                "param_name",
                format!(
                    "Parameter {} is owned by rank {}, not {}",
                    param_name, owner_rank, self.rank_mapping.rank
                ),
                "parameter owned by this rank",
            )
            .into());
        }

        // First check GPU cache
        {
            let gpu_cache = self.gpu_optimizer_cache.read().unwrap();
            if let Some(state) = gpu_cache.get(param_name) {
                info!(
                    "    Retrieved optimizer state from GPU cache: {}",
                    param_name
                );
                return Ok(state.clone());
            }
        }

        // Check CPU storage if GPU cache miss
        if self.config.offload_optimizer_states {
            let cpu_states = self.cpu_optimizer_states.read().unwrap();
            if let Some(cpu_state) = cpu_states.get(param_name) {
                info!(
                    "    Retrieved optimizer state from CPU storage: {}",
                    param_name
                );
                let state = self.decompress_optimizer_state(cpu_state)?;

                // Promote to GPU cache if there's space
                if self.should_promote_to_gpu_cache(param_name) {
                    self.promote_state_to_gpu(param_name, &state).await?;
                }

                return Ok(state);
            }
        }

        // Check in-memory storage for backward compatibility
        {
            let states = self.optimizer_states.read().unwrap();
            if let Some(state) = states.get(param_name) {
                info!("    Retrieved optimizer state from memory: {}", param_name);
                return Ok(state.clone());
            }
        }

        // Create new optimizer state if not found
        info!("    Creating new optimizer state: {}", param_name);
        Ok(OptimizerState::new())
    }

    /// Store optimizer state for a parameter
    ///
    /// Automatically determines whether to store on CPU or GPU based on configuration
    /// and memory pressure. Implements intelligent caching strategies.
    pub async fn store_state(&self, param_name: &str, state: &OptimizerState) -> TorshResult<()> {
        // Verify ownership
        let param_hash = self.hash_parameter_name(param_name);
        let owner_rank = param_hash % self.rank_mapping.world_size;

        if owner_rank != self.rank_mapping.rank {
            return Err(TorshDistributedError::invalid_argument(
                "param_name",
                format!(
                    "Cannot store state for parameter {} owned by rank {}",
                    param_name, owner_rank
                ),
                "parameter owned by this rank",
            )
            .into());
        }

        if self.config.offload_optimizer_states {
            // Store on CPU with compression
            let cpu_state = self.compress_optimizer_state(state)?;
            let state_size = cpu_state.size_bytes;

            {
                let mut cpu_states = self.cpu_optimizer_states.write().unwrap();
                if let Some(old_state) = cpu_states.insert(param_name.to_string(), cpu_state) {
                    // Update memory tracking
                    self.memory_used_cpu
                        .fetch_sub(old_state.size_bytes, std::sync::atomic::Ordering::SeqCst);
                }
                self.memory_used_cpu
                    .fetch_add(state_size, std::sync::atomic::Ordering::SeqCst);
            }

            // Also keep in GPU cache if recently accessed
            if self.should_cache_on_gpu(param_name) {
                self.store_state_on_gpu(param_name, state).await?;
            }

            info!(
                "    Stored optimizer state on CPU: {} ({} bytes)",
                param_name, state_size
            );
        } else {
            // Store in GPU memory or regular memory
            self.store_state_on_gpu(param_name, state).await?;
        }

        Ok(())
    }

    /// Store optimizer state in GPU cache/memory
    async fn store_state_on_gpu(
        &self,
        param_name: &str,
        state: &OptimizerState,
    ) -> TorshResult<()> {
        let state_size = self.calculate_state_size(state);

        // Check if we need to evict states to make room
        while self
            .memory_used_gpu
            .load(std::sync::atomic::Ordering::SeqCst)
            + state_size
            > self.config.gpu_param_memory_budget
        {
            self.evict_lru_gpu_state().await?;
        }

        {
            let mut gpu_cache = self.gpu_optimizer_cache.write().unwrap();
            gpu_cache.insert(param_name.to_string(), state.clone());
        }

        self.memory_used_gpu
            .fetch_add(state_size, std::sync::atomic::Ordering::SeqCst);
        info!(
            "    Stored optimizer state on GPU: {} ({} bytes)",
            param_name, state_size
        );
        Ok(())
    }

    /// Promote frequently accessed state to GPU cache
    async fn promote_state_to_gpu(
        &self,
        param_name: &str,
        state: &OptimizerState,
    ) -> TorshResult<()> {
        if self.has_gpu_cache_space(state) {
            self.store_state_on_gpu(param_name, state).await?;
            info!(
                "   ⬆️  Promoted optimizer state to GPU cache: {}",
                param_name
            );
        }
        Ok(())
    }

    /// Evict least recently used state from GPU cache
    async fn evict_lru_gpu_state(&self) -> TorshResult<()> {
        // Simple LRU eviction - in practice you'd track access patterns
        let state_to_evict = {
            let gpu_cache = self.gpu_optimizer_cache.read().unwrap();
            gpu_cache.keys().next().cloned()
        };

        if let Some(param_name) = state_to_evict {
            let state_size = {
                let mut gpu_cache = self.gpu_optimizer_cache.write().unwrap();
                if let Some(state) = gpu_cache.remove(&param_name) {
                    self.calculate_state_size(&state)
                } else {
                    0
                }
            };

            self.memory_used_gpu
                .fetch_sub(state_size, std::sync::atomic::Ordering::SeqCst);
            info!(
                "   📤 Evicted optimizer state from GPU cache: {} ({} bytes)",
                param_name, state_size
            );
        }

        Ok(())
    }

    /// Compress optimizer state for CPU storage
    fn compress_optimizer_state(&self, state: &OptimizerState) -> TorshResult<CpuOptimizerState> {
        let mut total_size = std::mem::size_of::<usize>(); // step_count

        let momentum_data = if let Some(ref momentum) = state.momentum {
            let data = momentum.to_vec()?;
            total_size += data.len() * std::mem::size_of::<f32>();
            Some(data)
        } else {
            None
        };

        let variance_data = if let Some(ref variance) = state.variance {
            let data = variance.to_vec()?;
            total_size += data.len() * std::mem::size_of::<f32>();
            Some(data)
        } else {
            None
        };

        // Apply compression based on configuration
        let (compressed_momentum, compressed_variance) = match self.config.cpu_compression {
            super::config::CpuCompressionMethod::None => (momentum_data, variance_data),
            super::config::CpuCompressionMethod::FP16 => {
                let compressed_momentum = momentum_data
                    .map(|data| self.compress_to_fp16(&data))
                    .transpose()?;
                let compressed_variance = variance_data
                    .map(|data| self.compress_to_fp16(&data))
                    .transpose()?;
                (compressed_momentum, compressed_variance)
            }
            super::config::CpuCompressionMethod::BF16 => {
                let compressed_momentum = momentum_data
                    .map(|data| self.compress_to_bf16(&data))
                    .transpose()?;
                let compressed_variance = variance_data
                    .map(|data| self.compress_to_bf16(&data))
                    .transpose()?;
                (compressed_momentum, compressed_variance)
            }
            _ => {
                return Err(TorshDistributedError::feature_not_available(
                    "Compression method",
                    "Compression method not implemented for optimizer states",
                ));
            }
        };

        Ok(CpuOptimizerState {
            momentum_data: compressed_momentum,
            variance_data: compressed_variance,
            momentum_shape: state.momentum.as_ref().map(|m| m.shape().dims().to_vec()),
            variance_shape: state.variance.as_ref().map(|v| v.shape().dims().to_vec()),
            step_count: state.step_count,
            size_bytes: total_size,
            compression: self.config.cpu_compression,
        })
    }

    /// Decompress optimizer state from CPU storage
    fn decompress_optimizer_state(
        &self,
        cpu_state: &CpuOptimizerState,
    ) -> TorshResult<OptimizerState> {
        let momentum = if let Some(ref momentum_data) = cpu_state.momentum_data {
            let decompressed_data = match cpu_state.compression {
                super::config::CpuCompressionMethod::None => momentum_data.clone(),
                super::config::CpuCompressionMethod::FP16 => {
                    self.decompress_from_fp16(momentum_data)?
                }
                super::config::CpuCompressionMethod::BF16 => {
                    self.decompress_from_bf16(momentum_data)?
                }
                _ => {
                    return Err(TorshDistributedError::feature_not_available(
                        "Decompression method",
                        "Compression method not implemented for decompression",
                    ))
                }
            };

            if let Some(ref shape) = cpu_state.momentum_shape {
                Some(Tensor::from_vec(decompressed_data, shape)?)
            } else {
                None
            }
        } else {
            None
        };

        let variance = if let Some(ref variance_data) = cpu_state.variance_data {
            let decompressed_data = match cpu_state.compression {
                super::config::CpuCompressionMethod::None => variance_data.clone(),
                super::config::CpuCompressionMethod::FP16 => {
                    self.decompress_from_fp16(variance_data)?
                }
                super::config::CpuCompressionMethod::BF16 => {
                    self.decompress_from_bf16(variance_data)?
                }
                _ => {
                    return Err(TorshDistributedError::feature_not_available(
                        "Decompression method",
                        "Compression method not implemented for decompression",
                    ))
                }
            };

            if let Some(ref shape) = cpu_state.variance_shape {
                Some(Tensor::from_vec(decompressed_data, shape)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(OptimizerState {
            momentum,
            variance,
            step_count: cpu_state.step_count,
        })
    }

    /// Update optimizer state in-place (for Adam, RMSprop, etc.)
    pub async fn update_state_inplace(
        &self,
        param_name: &str,
        gradient: &Tensor<f32>,
        learning_rate: f32,
        beta1: Option<f32>,
        beta2: Option<f32>,
        epsilon: f32,
    ) -> TorshResult<()> {
        let mut state = self.fetch_state(param_name).await?;

        // Update step count
        state.step_count += 1;

        // Update momentum (first moment)
        if let Some(beta1) = beta1 {
            if state.momentum.is_none() {
                state.momentum = Some(Tensor::zeros_like(gradient)?);
            }

            if let Some(ref mut momentum) = state.momentum {
                // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                *momentum = momentum
                    .mul_scalar(beta1)?
                    .add(&gradient.mul_scalar(1.0 - beta1)?)?;
            }
        }

        // Update variance (second moment)
        if let Some(beta2) = beta2 {
            if state.variance.is_none() {
                state.variance = Some(Tensor::zeros_like(gradient)?);
            }

            if let Some(ref mut variance) = state.variance {
                // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                let grad_squared = gradient.mul(gradient)?;
                *variance = variance
                    .mul_scalar(beta2)?
                    .add(&grad_squared.mul_scalar(1.0 - beta2)?)?;
            }
        }

        // Store updated state
        self.store_state(param_name, &state).await?;
        Ok(())
    }

    /// Get all optimizer states owned by this rank
    pub async fn get_owned_states(&self) -> TorshResult<HashMap<String, OptimizerState>> {
        let mut owned_states = HashMap::new();

        // Collect from GPU cache
        {
            let gpu_cache = self.gpu_optimizer_cache.read().unwrap();
            for (param_name, state) in gpu_cache.iter() {
                if self.owns_parameter(param_name) {
                    owned_states.insert(param_name.clone(), state.clone());
                }
            }
        }

        // Collect from CPU storage
        if self.config.offload_optimizer_states {
            let cpu_states = self.cpu_optimizer_states.read().unwrap();
            for (param_name, cpu_state) in cpu_states.iter() {
                if self.owns_parameter(param_name) && !owned_states.contains_key(param_name) {
                    let state = self.decompress_optimizer_state(cpu_state)?;
                    owned_states.insert(param_name.clone(), state);
                }
            }
        }

        // Collect from regular memory storage
        {
            let states = self.optimizer_states.read().unwrap();
            for (param_name, state) in states.iter() {
                if self.owns_parameter(param_name) && !owned_states.contains_key(param_name) {
                    owned_states.insert(param_name.clone(), state.clone());
                }
            }
        }

        Ok(owned_states)
    }

    /// Clear all optimizer states (reset for new training)
    pub async fn clear_states(&self) -> TorshResult<()> {
        {
            let mut states = self.optimizer_states.write().unwrap();
            states.clear();
        }

        {
            let mut cpu_states = self.cpu_optimizer_states.write().unwrap();
            cpu_states.clear();
        }

        {
            let mut gpu_cache = self.gpu_optimizer_cache.write().unwrap();
            gpu_cache.clear();
        }

        self.memory_used_cpu
            .store(0, std::sync::atomic::Ordering::SeqCst);
        self.memory_used_gpu
            .store(0, std::sync::atomic::Ordering::SeqCst);

        info!("   🧹 Cleared all optimizer states");
        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> OptimizerStateMemoryStats {
        OptimizerStateMemoryStats {
            cpu_memory_used: self
                .memory_used_cpu
                .load(std::sync::atomic::Ordering::SeqCst),
            gpu_memory_used: self
                .memory_used_gpu
                .load(std::sync::atomic::Ordering::SeqCst),
            states_on_cpu: self.cpu_optimizer_states.read().unwrap().len(),
            states_on_gpu: self.gpu_optimizer_cache.read().unwrap().len(),
            total_states: self.optimizer_states.read().unwrap().len()
                + self.cpu_optimizer_states.read().unwrap().len()
                + self.gpu_optimizer_cache.read().unwrap().len(),
        }
    }

    // Helper methods

    fn hash_parameter_name(&self, param_name: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        param_name.hash(&mut hasher);
        hasher.finish() as usize
    }

    fn owns_parameter(&self, param_name: &str) -> bool {
        let param_hash = self.hash_parameter_name(param_name);
        let owner_rank = param_hash % self.rank_mapping.world_size;
        owner_rank == self.rank_mapping.rank
    }

    fn should_promote_to_gpu_cache(&self, _param_name: &str) -> bool {
        // Simple heuristic - promote if GPU cache has space
        self.memory_used_gpu
            .load(std::sync::atomic::Ordering::SeqCst)
            < self.config.gpu_param_memory_budget / 2
    }

    fn should_cache_on_gpu(&self, _param_name: &str) -> bool {
        // Cache frequently accessed parameters
        true // Simplified - in practice would track access patterns
    }

    fn has_gpu_cache_space(&self, state: &OptimizerState) -> bool {
        let state_size = self.calculate_state_size(state);
        self.memory_used_gpu
            .load(std::sync::atomic::Ordering::SeqCst)
            + state_size
            <= self.config.gpu_param_memory_budget
    }

    fn calculate_state_size(&self, state: &OptimizerState) -> usize {
        let mut size = std::mem::size_of::<usize>(); // step_count

        if let Some(ref momentum) = state.momentum {
            size += momentum.numel() * std::mem::size_of::<f32>();
        }

        if let Some(ref variance) = state.variance {
            size += variance.numel() * std::mem::size_of::<f32>();
        }

        size
    }

    // Compression helper methods
    fn compress_to_fp16(&self, data: &[f32]) -> TorshResult<Vec<f32>> {
        use half::f16;
        Ok(data
            .iter()
            .map(|&val| f16::from_f32(val).to_f32())
            .collect())
    }

    fn compress_to_bf16(&self, data: &[f32]) -> TorshResult<Vec<f32>> {
        use half::bf16;
        Ok(data
            .iter()
            .map(|&val| bf16::from_f32(val).to_f32())
            .collect())
    }

    fn decompress_from_fp16(&self, data: &[f32]) -> TorshResult<Vec<f32>> {
        Ok(data.to_vec()) // Already decompressed during compression
    }

    fn decompress_from_bf16(&self, data: &[f32]) -> TorshResult<Vec<f32>> {
        Ok(data.to_vec()) // Already decompressed during compression
    }
}

/// Optimizer state for a single parameter
///
/// Contains momentum, variance, and step count for optimizers like Adam, AdamW, RMSprop.
/// Supports automatic CPU/GPU management through the OptimizerStateManager.
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// First moment estimate (momentum) for Adam-style optimizers
    pub momentum: Option<Tensor<f32>>,
    /// Second moment estimate (variance) for Adam-style optimizers
    pub variance: Option<Tensor<f32>>,
    /// Step count for bias correction
    pub step_count: usize,
}

impl OptimizerState {
    /// Create a new empty optimizer state
    pub fn new() -> Self {
        Self {
            momentum: None,
            variance: None,
            step_count: 0,
        }
    }

    /// Initialize momentum tensor with zeros matching parameter shape
    pub fn init_momentum(&mut self, param_shape: &[usize]) -> TorshResult<()> {
        self.momentum = Some(Tensor::zeros(param_shape, torsh_core::DeviceType::Cpu)?);
        Ok(())
    }

    /// Initialize variance tensor with zeros matching parameter shape
    pub fn init_variance(&mut self, param_shape: &[usize]) -> TorshResult<()> {
        self.variance = Some(Tensor::zeros(param_shape, torsh_core::DeviceType::Cpu)?);
        Ok(())
    }

    /// Get bias-corrected momentum for Adam optimizer
    pub fn get_bias_corrected_momentum(&self, beta1: f32) -> TorshResult<Option<Tensor<f32>>> {
        if let Some(ref momentum) = self.momentum {
            let correction = 1.0 - beta1.powi(self.step_count as i32);
            Ok(Some(momentum.div_scalar(correction)?))
        } else {
            Ok(None)
        }
    }

    /// Get bias-corrected variance for Adam optimizer
    pub fn get_bias_corrected_variance(&self, beta2: f32) -> TorshResult<Option<Tensor<f32>>> {
        if let Some(ref variance) = self.variance {
            let correction = 1.0 - beta2.powi(self.step_count as i32);
            Ok(Some(variance.div_scalar(correction)?))
        } else {
            Ok(None)
        }
    }

    /// Check if state has momentum
    pub fn has_momentum(&self) -> bool {
        self.momentum.is_some()
    }

    /// Check if state has variance
    pub fn has_variance(&self) -> bool {
        self.variance.is_some()
    }

    /// Reset step count and optionally clear moment estimates
    pub fn reset(&mut self, clear_moments: bool) {
        self.step_count = 0;
        if clear_moments {
            self.momentum = None;
            self.variance = None;
        }
    }
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU-stored optimizer state with compression
#[derive(Debug, Clone)]
struct CpuOptimizerState {
    /// Compressed momentum data
    momentum_data: Option<Vec<f32>>,
    /// Compressed variance data
    variance_data: Option<Vec<f32>>,
    /// Original momentum shape
    momentum_shape: Option<Vec<usize>>,
    /// Original variance shape
    variance_shape: Option<Vec<usize>>,
    /// Step count
    step_count: usize,
    /// Total size in bytes
    size_bytes: usize,
    /// Compression method used
    compression: super::config::CpuCompressionMethod,
}

/// Memory usage statistics for optimizer states
#[derive(Debug, Clone)]
pub struct OptimizerStateMemoryStats {
    /// Memory used by CPU-stored optimizer states
    pub cpu_memory_used: usize,
    /// Memory used by GPU-cached optimizer states
    pub gpu_memory_used: usize,
    /// Number of optimizer states stored on CPU
    pub states_on_cpu: usize,
    /// Number of optimizer states cached on GPU
    pub states_on_gpu: usize,
    /// Total number of optimizer states
    pub total_states: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_state_creation() {
        let state = OptimizerState::new();
        assert!(!state.has_momentum());
        assert!(!state.has_variance());
        assert_eq!(state.step_count, 0);
    }

    #[test]
    fn test_rank_mapping() {
        let mapping = Zero3RankMapping::new(2, 8);
        assert_eq!(mapping.rank, 2);
        assert_eq!(mapping.world_size, 8);
        assert!(mapping.owns_partition(2)); // 2 % 8 == 2
        assert!(!mapping.owns_partition(3)); // 3 % 8 != 2
    }

    #[tokio::test]
    async fn test_optimizer_state_manager() {
        let config = Zero3CpuOffloadConfig::default();
        let rank_mapping = Zero3RankMapping::new(0, 4);
        let manager = OptimizerStateManager::new(&config, &rank_mapping).unwrap();

        // Test fetching non-existent state (should create new)
        let state = manager.fetch_state("test_param").await.unwrap();
        assert_eq!(state.step_count, 0);

        // Test storing and retrieving state
        let mut test_state = OptimizerState::new();
        test_state.step_count = 5;
        manager
            .store_state("test_param", &test_state)
            .await
            .unwrap();

        let retrieved_state = manager.fetch_state("test_param").await.unwrap();
        assert_eq!(retrieved_state.step_count, 5);
    }

    #[test]
    fn test_optimizer_state_initialization() {
        let mut state = OptimizerState::new();

        // Test momentum initialization
        state.init_momentum(&[10, 10]).unwrap();
        assert!(state.has_momentum());

        // Test variance initialization
        state.init_variance(&[10, 10]).unwrap();
        assert!(state.has_variance());

        // Test reset
        state.reset(true);
        assert_eq!(state.step_count, 0);
        assert!(!state.has_momentum());
        assert!(!state.has_variance());
    }

    #[tokio::test]
    async fn test_memory_stats() {
        let config = Zero3CpuOffloadConfig::default();
        let rank_mapping = Zero3RankMapping::new(0, 4);
        let manager = OptimizerStateManager::new(&config, &rank_mapping).unwrap();

        let stats = manager.get_memory_stats();
        assert_eq!(stats.cpu_memory_used, 0);
        assert_eq!(stats.gpu_memory_used, 0);
        assert_eq!(stats.total_states, 0);
    }
}
