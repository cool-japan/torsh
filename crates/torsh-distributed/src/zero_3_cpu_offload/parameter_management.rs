//! Parameter Management for ZeRO-3 CPU Offloading
//!
//! This module provides parameter partitioning, CPU storage, and GPU caching
//! functionality for ZeRO-3 distributed training with CPU offloading.
//! It manages the complex process of partitioning model parameters across
//! distributed workers and efficiently moving them between CPU and GPU memory.

use crate::{TorshDistributedError, TorshResult};
use log::info;
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, RwLock};
use torsh_tensor::Tensor;

use super::config::{
    CpuCompressionMethod, ModelParameters, Zero3CpuOffloadConfig, Zero3RankMapping,
};

/// Parameter partitioner for ZeRO-3 distributed training
///
/// Handles the partitioning of model parameters across distributed workers,
/// ensuring efficient load balancing and memory utilization.
pub struct ParameterPartitioner {
    config: Zero3CpuOffloadConfig,
    rank_mapping: Zero3RankMapping,
    partition_map: HashMap<String, Vec<ParameterPartition>>,
    total_parameters: usize,
}

impl ParameterPartitioner {
    /// Create a new parameter partitioner
    pub fn new(
        config: &Zero3CpuOffloadConfig,
        rank_mapping: &Zero3RankMapping,
        model_params: &ModelParameters,
    ) -> TorshResult<Self> {
        let mut partition_map = HashMap::new();

        // Create partitions for each parameter
        for param_name in &model_params.parameter_names {
            let param_shape = model_params.parameter_shapes.get(param_name).unwrap();
            let partitions = Self::create_parameter_partitions(
                param_name,
                param_shape,
                rank_mapping.world_size(),
            );
            partition_map.insert(param_name.clone(), partitions);
        }

        info!(
            " Parameter partitioner initialized: {} parameters across {} ranks",
            model_params.parameter_names.len(),
            rank_mapping.world_size()
        );

        Ok(Self {
            config: config.clone(),
            rank_mapping: rank_mapping.clone(),
            partition_map,
            total_parameters: model_params.parameter_count,
        })
    }

    /// Create parameter partitions for a given parameter
    fn create_parameter_partitions(
        param_name: &str,
        shape: &[usize],
        world_size: usize,
    ) -> Vec<ParameterPartition> {
        let total_elements = shape.iter().product::<usize>();
        let elements_per_partition = total_elements.div_ceil(world_size);

        let mut partitions = Vec::new();
        for rank in 0..world_size {
            let start_idx = rank * elements_per_partition;
            let end_idx = ((rank + 1) * elements_per_partition).min(total_elements);

            if start_idx < total_elements {
                partitions.push(ParameterPartition {
                    param_name: param_name.to_string(),
                    owner_rank: rank,
                    start_idx,
                    end_idx,
                    size_elements: end_idx - start_idx,
                });
            }
        }

        partitions
    }

    /// Get the partitions for a specific parameter
    pub fn get_parameter_partitions(&self, param_name: &str) -> Option<&Vec<ParameterPartition>> {
        self.partition_map.get(param_name)
    }

    /// Get the partition owned by this rank for a parameter
    pub fn get_owned_partition(&self, param_name: &str) -> Option<&ParameterPartition> {
        if let Some(partitions) = self.partition_map.get(param_name) {
            partitions
                .iter()
                .find(|p| p.owner_rank == self.rank_mapping.rank())
        } else {
            None
        }
    }

    /// Get all partitions owned by this rank
    pub fn get_all_owned_partitions(&self) -> Vec<&ParameterPartition> {
        self.partition_map
            .values()
            .flatten()
            .filter(|p| p.owner_rank == self.rank_mapping.rank())
            .collect()
    }

    /// Get the total number of parameters
    pub fn total_parameter_count(&self) -> usize {
        self.total_parameters
    }

    /// Get the number of partitions owned by this rank
    pub fn owned_partition_count(&self) -> usize {
        self.get_all_owned_partitions().len()
    }

    /// Calculate memory requirements for owned partitions
    pub fn calculate_owned_memory_requirement(&self) -> usize {
        self.get_all_owned_partitions()
            .iter()
            .map(|p| p.size_elements * std::mem::size_of::<f32>())
            .sum()
    }

    /// Get partitioner statistics
    pub fn get_statistics(&self) -> ParameterPartitionerStats {
        let owned_partitions = self.get_all_owned_partitions();
        let total_partitions: usize = self.partition_map.values().map(|v| v.len()).sum();

        ParameterPartitionerStats {
            total_parameters: self.partition_map.len(),
            total_partitions,
            owned_partitions: owned_partitions.len(),
            owned_elements: owned_partitions.iter().map(|p| p.size_elements).sum(),
            memory_requirement_bytes: self.calculate_owned_memory_requirement(),
            rank: self.rank_mapping.rank(),
            world_size: self.rank_mapping.world_size(),
        }
    }
}

/// Represents a partition of a parameter owned by a specific rank
#[derive(Debug, Clone)]
pub struct ParameterPartition {
    pub param_name: String,
    pub owner_rank: usize,
    pub start_idx: usize,
    pub end_idx: usize,
    pub size_elements: usize,
}

impl ParameterPartition {
    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_elements * std::mem::size_of::<f32>()
    }

    /// Check if this partition contains a specific element index
    pub fn contains_element(&self, element_idx: usize) -> bool {
        element_idx >= self.start_idx && element_idx < self.end_idx
    }

    /// Get the local index within this partition for a global element index
    pub fn global_to_local_index(&self, global_idx: usize) -> Option<usize> {
        if self.contains_element(global_idx) {
            Some(global_idx - self.start_idx)
        } else {
            None
        }
    }

    /// Get the global index for a local element index within this partition
    pub fn local_to_global_index(&self, local_idx: usize) -> Option<usize> {
        if local_idx < self.size_elements {
            Some(self.start_idx + local_idx)
        } else {
            None
        }
    }
}

/// Statistics about parameter partitioning
#[derive(Debug, Clone)]
pub struct ParameterPartitionerStats {
    pub total_parameters: usize,
    pub total_partitions: usize,
    pub owned_partitions: usize,
    pub owned_elements: usize,
    pub memory_requirement_bytes: usize,
    pub rank: usize,
    pub world_size: usize,
}

/// CPU parameter store with compression support
///
/// Manages storage of parameters in CPU memory with optional compression
/// to optimize memory usage while maintaining fast access patterns.
pub struct CpuParameterStore {
    config: Zero3CpuOffloadConfig,
    stored_parameters: RwLock<HashMap<String, CpuParameterData>>,
    memory_used: std::sync::atomic::AtomicUsize,
}

impl CpuParameterStore {
    /// Create a new CPU parameter store
    pub fn new(config: &Zero3CpuOffloadConfig) -> TorshResult<Self> {
        info!(
            " CPU Parameter Store initialized with {} MB budget",
            config.cpu_memory_budget / (1024 * 1024)
        );

        Ok(Self {
            config: config.clone(),
            stored_parameters: RwLock::new(HashMap::new()),
            memory_used: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Store a parameter in CPU memory
    pub async fn store(&self, param_name: &str, data: &CpuParameterData) -> TorshResult<()> {
        let mut params = self.stored_parameters.write().unwrap();

        // Check memory budget
        let new_memory_usage = self.memory_used() + data.size_bytes;
        if new_memory_usage > self.config.cpu_memory_budget {
            return Err(TorshDistributedError::memory_allocation_failed(
                new_memory_usage,
                "CPU memory budget exceeded",
            ));
        }

        if let Some(old_data) = params.insert(param_name.to_string(), data.clone()) {
            // Subtract old memory usage
            self.memory_used
                .fetch_sub(old_data.size_bytes, std::sync::atomic::Ordering::SeqCst);
        }

        // Add new memory usage
        self.memory_used
            .fetch_add(data.size_bytes, std::sync::atomic::Ordering::SeqCst);

        info!(
            "    Stored parameter '{}' in CPU ({} bytes)",
            param_name, data.size_bytes
        );

        Ok(())
    }

    /// Fetch a parameter from CPU memory
    pub async fn fetch(&self, param_name: &str) -> TorshResult<CpuParameterData> {
        let params = self.stored_parameters.read().unwrap();
        params.get(param_name).cloned().ok_or_else(|| {
            TorshDistributedError::invalid_argument(
                "param_name",
                format!("Parameter {} not found in CPU store", param_name),
                "valid parameter name that exists in CPU store",
            )
        })
    }

    /// Remove a parameter from CPU memory
    pub async fn remove(&self, param_name: &str) -> TorshResult<Option<CpuParameterData>> {
        let mut params = self.stored_parameters.write().unwrap();
        if let Some(data) = params.remove(param_name) {
            self.memory_used
                .fetch_sub(data.size_bytes, std::sync::atomic::Ordering::SeqCst);
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }

    /// Check if a parameter exists in the store
    pub fn contains(&self, param_name: &str) -> bool {
        self.stored_parameters
            .read()
            .unwrap()
            .contains_key(param_name)
    }

    /// Get the current memory usage
    pub fn memory_used(&self) -> usize {
        self.memory_used.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get the number of stored parameters
    pub fn parameter_count(&self) -> usize {
        self.stored_parameters.read().unwrap().len()
    }

    /// Get memory utilization as a percentage
    pub fn memory_utilization(&self) -> f32 {
        (self.memory_used() as f32) / (self.config.cpu_memory_budget as f32)
    }

    /// Get list of all stored parameter names
    pub fn get_parameter_names(&self) -> Vec<String> {
        self.stored_parameters
            .read()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    }

    /// Clear all stored parameters
    pub async fn clear(&self) -> TorshResult<()> {
        let mut params = self.stored_parameters.write().unwrap();
        params.clear();
        self.memory_used
            .store(0, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Get CPU store statistics
    pub fn get_statistics(&self) -> CpuParameterStoreStats {
        let params = self.stored_parameters.read().unwrap();
        let compression_ratios: Vec<f32> = params
            .values()
            .map(|data| data.compression.ratio())
            .collect();

        let avg_compression_ratio = if compression_ratios.is_empty() {
            1.0
        } else {
            compression_ratios.iter().sum::<f32>() / compression_ratios.len() as f32
        };

        CpuParameterStoreStats {
            parameter_count: params.len(),
            memory_used_bytes: self.memory_used(),
            memory_budget_bytes: self.config.cpu_memory_budget,
            memory_utilization: self.memory_utilization(),
            average_compression_ratio: avg_compression_ratio,
        }
    }
}

/// CPU parameter data with compression support
#[derive(Debug, Clone)]
pub struct CpuParameterData {
    pub data: Vec<f32>,
    pub bias_data: Option<Vec<f32>>,
    pub weight_shape: Vec<usize>,
    pub bias_shape: Option<Vec<usize>>,
    pub size_bytes: usize,
    pub compression: CpuCompressionMethod,
}

impl CpuParameterData {
    /// Create new CPU parameter data
    pub fn new(
        data: Vec<f32>,
        weight_shape: Vec<usize>,
        bias_data: Option<Vec<f32>>,
        bias_shape: Option<Vec<usize>>,
        compression: CpuCompressionMethod,
    ) -> Self {
        let size_bytes = data.len() * std::mem::size_of::<f32>()
            + bias_data
                .as_ref()
                .map(|b| b.len() * std::mem::size_of::<f32>())
                .unwrap_or(0);

        Self {
            data,
            bias_data,
            weight_shape,
            bias_shape,
            size_bytes,
            compression,
        }
    }

    /// Get the number of weight elements
    pub fn weight_elements(&self) -> usize {
        self.data.len()
    }

    /// Get the number of bias elements
    pub fn bias_elements(&self) -> usize {
        self.bias_data.as_ref().map(|b| b.len()).unwrap_or(0)
    }

    /// Get the total number of elements
    pub fn total_elements(&self) -> usize {
        self.weight_elements() + self.bias_elements()
    }

    /// Get the effective compression ratio
    pub fn effective_compression_ratio(&self) -> f32 {
        self.compression.ratio()
    }
}

/// Statistics about CPU parameter store
#[derive(Debug, Clone)]
pub struct CpuParameterStoreStats {
    pub parameter_count: usize,
    pub memory_used_bytes: usize,
    pub memory_budget_bytes: usize,
    pub memory_utilization: f32,
    pub average_compression_ratio: f32,
}

/// GPU parameter cache for active parameters
///
/// Maintains a cache of frequently accessed parameters in GPU memory
/// with LRU eviction policy to optimize memory usage and access patterns.
pub struct GpuParameterCache {
    config: Zero3CpuOffloadConfig,
    cached_parameters: RwLock<HashMap<String, LayerParameters>>,
    memory_used: std::sync::atomic::AtomicUsize,
    cache_lru: Mutex<VecDeque<String>>,
}

impl GpuParameterCache {
    /// Create a new GPU parameter cache
    pub fn new(config: &Zero3CpuOffloadConfig) -> TorshResult<Self> {
        info!(
            " GPU Parameter Cache initialized with {} MB budget",
            config.gpu_param_memory_budget / (1024 * 1024)
        );

        Ok(Self {
            config: config.clone(),
            cached_parameters: RwLock::new(HashMap::new()),
            memory_used: std::sync::atomic::AtomicUsize::new(0),
            cache_lru: Mutex::new(VecDeque::new()),
        })
    }

    /// Get a parameter from the cache
    pub async fn get(&self, param_name: &str) -> TorshResult<Option<LayerParameters>> {
        let params = self.cached_parameters.read().unwrap();
        if let Some(layer_params) = params.get(param_name) {
            // Update LRU
            let mut lru = self.cache_lru.lock().unwrap();
            if let Some(pos) = lru.iter().position(|x| x == param_name) {
                lru.remove(pos);
            }
            lru.push_back(param_name.to_string());

            Ok(Some(layer_params.clone()))
        } else {
            Ok(None)
        }
    }

    /// Store a parameter in the cache
    pub async fn store(&self, param_name: &str, params: &LayerParameters) -> TorshResult<()> {
        let param_size = params.weight.numel() * std::mem::size_of::<f32>()
            + params
                .bias
                .as_ref()
                .map(|b| b.numel() * std::mem::size_of::<f32>())
                .unwrap_or(0);

        // Check if we need to evict parameters to make space
        while self.memory_used.load(std::sync::atomic::Ordering::SeqCst) + param_size
            > self.config.gpu_param_memory_budget
        {
            self.evict_lru_parameter().await?;
        }

        {
            let mut cached = self.cached_parameters.write().unwrap();
            cached.insert(param_name.to_string(), params.clone());
        }

        {
            let mut lru = self.cache_lru.lock().unwrap();
            lru.push_back(param_name.to_string());
        }

        self.memory_used
            .fetch_add(param_size, std::sync::atomic::Ordering::SeqCst);

        info!(
            "    Cached parameter '{}' in GPU ({} bytes)",
            param_name, param_size
        );

        Ok(())
    }

    /// Remove a parameter from the cache
    pub async fn remove(&self, param_name: &str) -> TorshResult<()> {
        let mut cached = self.cached_parameters.write().unwrap();
        if let Some(params) = cached.remove(param_name) {
            let param_size = params.weight.numel() * std::mem::size_of::<f32>()
                + params
                    .bias
                    .as_ref()
                    .map(|b| b.numel() * std::mem::size_of::<f32>())
                    .unwrap_or(0);
            self.memory_used
                .fetch_sub(param_size, std::sync::atomic::Ordering::SeqCst);
        }

        let mut lru = self.cache_lru.lock().unwrap();
        if let Some(pos) = lru.iter().position(|x| x == param_name) {
            lru.remove(pos);
        }

        Ok(())
    }

    /// Evict the least recently used parameter
    async fn evict_lru_parameter(&self) -> TorshResult<()> {
        let param_to_evict = {
            let mut lru = self.cache_lru.lock().unwrap();
            lru.pop_front()
        };

        if let Some(param_name) = param_to_evict {
            info!("   🗑️  Evicting LRU parameter: {}", param_name);
            self.remove(&param_name).await?;
        }

        Ok(())
    }

    /// Check if a parameter is cached
    pub fn contains(&self, param_name: &str) -> bool {
        self.cached_parameters
            .read()
            .unwrap()
            .contains_key(param_name)
    }

    /// Get the current memory usage
    pub fn memory_used(&self) -> usize {
        self.memory_used.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get the number of cached parameters
    pub fn parameter_count(&self) -> usize {
        self.cached_parameters.read().unwrap().len()
    }

    /// Get memory utilization as a percentage
    pub fn memory_utilization(&self) -> f32 {
        (self.memory_used() as f32) / (self.config.gpu_param_memory_budget as f32)
    }

    /// Get cache hit rate (requires tracking)
    pub fn get_cache_statistics(&self) -> GpuParameterCacheStats {
        let lru = self.cache_lru.lock().unwrap();

        GpuParameterCacheStats {
            parameter_count: self.parameter_count(),
            memory_used_bytes: self.memory_used(),
            memory_budget_bytes: self.config.gpu_param_memory_budget,
            memory_utilization: self.memory_utilization(),
            lru_queue_length: lru.len(),
        }
    }

    /// Clear the entire cache
    pub async fn clear(&self) -> TorshResult<()> {
        let mut cached = self.cached_parameters.write().unwrap();
        cached.clear();

        let mut lru = self.cache_lru.lock().unwrap();
        lru.clear();

        self.memory_used
            .store(0, std::sync::atomic::Ordering::SeqCst);

        Ok(())
    }
}

/// Layer parameters stored in GPU cache
#[derive(Debug, Clone)]
pub struct LayerParameters {
    pub weight: Tensor<f32>,
    pub bias: Option<Tensor<f32>>,
}

impl LayerParameters {
    /// Create new layer parameters
    pub fn new(weight: Tensor<f32>, bias: Option<Tensor<f32>>) -> Self {
        Self { weight, bias }
    }

    /// Get the total memory size of these parameters
    pub fn memory_size(&self) -> usize {
        let weight_size = self.weight.numel() * std::mem::size_of::<f32>();
        let bias_size = self
            .bias
            .as_ref()
            .map(|b| b.numel() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        weight_size + bias_size
    }

    /// Get the total number of elements
    pub fn total_elements(&self) -> usize {
        let weight_elements = self.weight.numel();
        let bias_elements = self.bias.as_ref().map(|b| b.numel()).unwrap_or(0);
        weight_elements + bias_elements
    }

    /// Check if this layer has bias
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

/// Statistics about GPU parameter cache
#[derive(Debug, Clone)]
pub struct GpuParameterCacheStats {
    pub parameter_count: usize,
    pub memory_used_bytes: usize,
    pub memory_budget_bytes: usize,
    pub memory_utilization: f32,
    pub lru_queue_length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_partition() {
        let partition = ParameterPartition {
            param_name: "layer1.weight".to_string(),
            owner_rank: 0,
            start_idx: 0,
            end_idx: 100,
            size_elements: 100,
        };

        assert_eq!(partition.size_bytes(), 400); // 100 * 4 bytes
        assert!(partition.contains_element(50));
        assert!(!partition.contains_element(150));
        assert_eq!(partition.global_to_local_index(25), Some(25));
        assert_eq!(partition.local_to_global_index(25), Some(25));
    }

    #[test]
    fn test_parameter_partitioner() {
        let config = Zero3CpuOffloadConfig::default();
        let rank_mapping = Zero3RankMapping::new(0, 4);

        let mut model_params = ModelParameters::new();
        model_params.add_parameter("layer1.weight".to_string(), vec![100, 50]);
        model_params.add_parameter("layer1.bias".to_string(), vec![50]);

        let partitioner = ParameterPartitioner::new(&config, &rank_mapping, &model_params).unwrap();

        assert_eq!(partitioner.partition_map.len(), 2);

        let weight_partitions = partitioner
            .get_parameter_partitions("layer1.weight")
            .unwrap();
        assert_eq!(weight_partitions.len(), 4); // 4 ranks

        let owned_partition = partitioner.get_owned_partition("layer1.weight").unwrap();
        assert_eq!(owned_partition.owner_rank, 0);

        let stats = partitioner.get_statistics();
        assert_eq!(stats.total_parameters, 2);
        assert_eq!(stats.rank, 0);
        assert_eq!(stats.world_size, 4);
    }

    #[test]
    fn test_cpu_parameter_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let bias_data = Some(vec![0.1, 0.2]);
        let weight_shape = vec![2, 2];
        let bias_shape = Some(vec![2]);

        let cpu_data = CpuParameterData::new(
            data,
            weight_shape,
            bias_data,
            bias_shape,
            CpuCompressionMethod::None,
        );

        assert_eq!(cpu_data.weight_elements(), 4);
        assert_eq!(cpu_data.bias_elements(), 2);
        assert_eq!(cpu_data.total_elements(), 6);
        assert_eq!(cpu_data.effective_compression_ratio(), 1.0);
    }

    #[tokio::test]
    async fn test_cpu_parameter_store() {
        let config = Zero3CpuOffloadConfig::default();
        let store = CpuParameterStore::new(&config).unwrap();

        let data = CpuParameterData::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            None,
            None,
            CpuCompressionMethod::None,
        );

        // Test store and fetch
        store.store("test_param", &data).await.unwrap();
        assert!(store.contains("test_param"));
        assert_eq!(store.parameter_count(), 1);

        let fetched = store.fetch("test_param").await.unwrap();
        assert_eq!(fetched.data, data.data);

        // Test remove
        let removed = store.remove("test_param").await.unwrap();
        assert!(removed.is_some());
        assert!(!store.contains("test_param"));
        assert_eq!(store.parameter_count(), 0);
    }

    #[test]
    fn test_layer_parameters() -> Result<(), Box<dyn std::error::Error>> {
        use torsh_tensor::Tensor;

        let weight = Tensor::zeros(&[10, 5], torsh_core::DeviceType::Cpu)?;
        let bias = Some(Tensor::zeros(&[5], torsh_core::DeviceType::Cpu)?);

        let layer_params = LayerParameters::new(weight, bias);

        assert!(layer_params.has_bias());
        assert_eq!(layer_params.total_elements(), 55); // 10*5 + 5
        Ok(())
    }
}
