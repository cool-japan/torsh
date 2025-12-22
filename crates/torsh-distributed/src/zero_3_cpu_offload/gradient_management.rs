//! Gradient Management for ZeRO-3 CPU Offloading
//!
//! This module provides gradient partitioning, CPU storage, and GPU buffering
//! functionality for ZeRO-3 distributed training. It manages the complex process
//! of partitioning gradients across distributed workers and efficiently handling
//! gradient synchronization and communication.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::{TorshDistributedError, TorshResult};
use log::info;
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};
use torsh_tensor::Tensor;

use super::config::{Zero3CpuOffloadConfig, Zero3RankMapping};

/// Gradient partitioner for ZeRO-3 distributed training
///
/// Handles the partitioning of gradients across distributed workers,
/// ensuring efficient load balancing and communication patterns.
pub struct GradientPartitioner {
    config: Zero3CpuOffloadConfig,
    rank_mapping: Zero3RankMapping,
    partition_metadata: Mutex<HashMap<String, GradientPartitionMetadata>>,
}

impl GradientPartitioner {
    /// Create a new gradient partitioner
    pub fn new(
        config: &Zero3CpuOffloadConfig,
        rank_mapping: &Zero3RankMapping,
    ) -> TorshResult<Self> {
        info!(
            " Gradient Partitioner initialized for rank {}/{}",
            rank_mapping.rank(),
            rank_mapping.world_size()
        );

        Ok(Self {
            config: config.clone(),
            rank_mapping: rank_mapping.clone(),
            partition_metadata: Mutex::new(HashMap::new()),
        })
    }

    /// Partition gradients for a layer across all ranks
    pub fn partition_gradients(
        &self,
        layer_name: &str,
        grads: &ParameterGradients,
    ) -> TorshResult<Vec<GradientPartition>> {
        let mut partitions = Vec::new();

        // Get weight gradient data
        let weight_grad = &grads.weight_grad;
        let grad_data = weight_grad.to_vec()?;
        let grad_shape_binding = weight_grad.shape();
        let _grad_shape = grad_shape_binding.dims();

        // Calculate elements per partition
        let total_elements = grad_data.len();
        let elements_per_partition = total_elements.div_ceil(self.rank_mapping.world_size());

        info!(
            "    Partitioning gradients for '{}': {} elements across {} ranks",
            layer_name,
            total_elements,
            self.rank_mapping.world_size()
        );

        // Create actual partitions by slicing the gradient tensor
        for rank in 0..self.rank_mapping.world_size() {
            let start_idx = rank * elements_per_partition;
            let end_idx = ((rank + 1) * elements_per_partition).min(total_elements);

            if start_idx < total_elements {
                // Extract partition data
                let partition_data = grad_data[start_idx..end_idx].to_vec();
                let partition_size = end_idx - start_idx;

                // Calculate partition shape (simplified: keep original shape, adjust first dimension)
                let partition_shape = vec![partition_size];

                // Create partition tensor
                let partition_tensor = Tensor::from_vec(partition_data, &partition_shape)?;

                partitions.push(GradientPartition {
                    layer_name: layer_name.to_string(),
                    rank,
                    partition_idx: rank,
                    start_idx,
                    end_idx,
                    size_elements: partition_size,
                    weight_gradient: partition_tensor,
                    bias_gradient: None, // Will be handled separately below
                });
            } else {
                // Create empty partition for ranks beyond the data
                let empty_tensor = Tensor::from_vec(vec![], &[0])?;
                partitions.push(GradientPartition {
                    layer_name: layer_name.to_string(),
                    rank,
                    partition_idx: rank,
                    start_idx: total_elements,
                    end_idx: total_elements,
                    size_elements: 0,
                    weight_gradient: empty_tensor,
                    bias_gradient: None,
                });
            }
        }

        // Handle bias gradients if present
        if let Some(ref bias_grad) = grads.bias_grad {
            let bias_data = bias_grad.to_vec()?;
            let bias_elements = bias_data.len();
            let bias_elements_per_partition =
                bias_elements.div_ceil(self.rank_mapping.world_size());

            for (rank, partition) in partitions.iter_mut().enumerate() {
                let bias_start = rank * bias_elements_per_partition;
                let bias_end = ((rank + 1) * bias_elements_per_partition).min(bias_elements);

                if bias_start < bias_elements {
                    let bias_partition_data = bias_data[bias_start..bias_end].to_vec();
                    let bias_partition_shape = vec![bias_partition_data.len()];
                    let bias_partition_tensor =
                        Tensor::from_vec(bias_partition_data, &bias_partition_shape)?;
                    partition.bias_gradient = Some(bias_partition_tensor);
                }
            }
        }

        // Store metadata for this layer
        let metadata = GradientPartitionMetadata {
            layer_name: layer_name.to_string(),
            total_weight_elements: total_elements,
            total_bias_elements: grads.bias_grad.as_ref().map(|b| b.numel()).unwrap_or(0),
            elements_per_partition,
            world_size: self.rank_mapping.world_size(),
        };

        {
            let mut meta = self.partition_metadata.lock().unwrap();
            meta.insert(layer_name.to_string(), metadata);
        }

        Ok(partitions)
    }

    /// Get the partition owned by this rank for a layer
    pub fn get_owned_partition<'a>(
        &self,
        layer_name: &str,
        partitions: &'a [GradientPartition],
    ) -> Option<&'a GradientPartition> {
        partitions
            .iter()
            .find(|p| p.rank == self.rank_mapping.rank() && p.layer_name == layer_name)
    }

    /// Get metadata for a layer's gradient partitioning
    pub fn get_layer_metadata(&self, layer_name: &str) -> Option<GradientPartitionMetadata> {
        let meta = self.partition_metadata.lock().unwrap();
        meta.get(layer_name).cloned()
    }

    /// Calculate total gradient memory for owned partitions
    pub fn calculate_owned_gradient_memory(
        &self,
        layer_gradients: &HashMap<String, Vec<GradientPartition>>,
    ) -> usize {
        let mut total_memory = 0;
        for partitions in layer_gradients.values() {
            if let Some(owned_partition) = partitions
                .iter()
                .find(|p| p.rank == self.rank_mapping.rank())
            {
                total_memory += owned_partition.memory_size();
            }
        }
        total_memory
    }

    /// Get partitioner statistics
    pub fn get_statistics(&self) -> GradientPartitionerStats {
        let meta = self.partition_metadata.lock().unwrap();
        let total_layers = meta.len();
        let total_elements: usize = meta
            .values()
            .map(|m| m.total_weight_elements + m.total_bias_elements)
            .sum();

        GradientPartitionerStats {
            total_layers,
            total_elements,
            rank: self.rank_mapping.rank(),
            world_size: self.rank_mapping.world_size(),
        }
    }
}

/// Metadata about gradient partitioning for a layer
#[derive(Debug, Clone)]
pub struct GradientPartitionMetadata {
    pub layer_name: String,
    pub total_weight_elements: usize,
    pub total_bias_elements: usize,
    pub elements_per_partition: usize,
    pub world_size: usize,
}

/// Represents a gradient partition for a specific rank
#[derive(Debug, Clone)]
pub struct GradientPartition {
    pub layer_name: String,
    pub rank: usize,
    pub partition_idx: usize,
    pub start_idx: usize,
    pub end_idx: usize,
    pub size_elements: usize,
    pub weight_gradient: Tensor<f32>,
    pub bias_gradient: Option<Tensor<f32>>,
}

impl GradientPartition {
    /// Get the total memory size of this partition
    pub fn memory_size(&self) -> usize {
        let weight_size = self.weight_gradient.numel() * std::mem::size_of::<f32>();
        let bias_size = self
            .bias_gradient
            .as_ref()
            .map(|b| b.numel() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        weight_size + bias_size
    }

    /// Check if this partition is empty
    pub fn is_empty(&self) -> bool {
        self.size_elements == 0
    }

    /// Get the total number of elements in this partition
    pub fn total_elements(&self) -> usize {
        let weight_elements = self.weight_gradient.numel();
        let bias_elements = self.bias_gradient.as_ref().map(|b| b.numel()).unwrap_or(0);
        weight_elements + bias_elements
    }

    /// Check if this partition has bias gradients
    pub fn has_bias(&self) -> bool {
        self.bias_gradient.is_some()
    }
}

/// Statistics about gradient partitioning
#[derive(Debug, Clone)]
pub struct GradientPartitionerStats {
    pub total_layers: usize,
    pub total_elements: usize,
    pub rank: usize,
    pub world_size: usize,
}

/// Parameter gradients for a layer
#[derive(Debug, Clone)]
pub struct ParameterGradients {
    pub weight_grad: Tensor<f32>,
    pub bias_grad: Option<Tensor<f32>>,
}

impl ParameterGradients {
    /// Create new parameter gradients
    pub fn new(weight_grad: Tensor<f32>, bias_grad: Option<Tensor<f32>>) -> Self {
        Self {
            weight_grad,
            bias_grad,
        }
    }

    /// Get the total memory size of these gradients
    pub fn memory_size(&self) -> usize {
        let weight_size = self.weight_grad.numel() * std::mem::size_of::<f32>();
        let bias_size = self
            .bias_grad
            .as_ref()
            .map(|b| b.numel() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        weight_size + bias_size
    }

    /// Get the total number of elements
    pub fn total_elements(&self) -> usize {
        let weight_elements = self.weight_grad.numel();
        let bias_elements = self.bias_grad.as_ref().map(|b| b.numel()).unwrap_or(0);
        weight_elements + bias_elements
    }

    /// Check if this has bias gradients
    pub fn has_bias(&self) -> bool {
        self.bias_grad.is_some()
    }
}

/// CPU gradient store for offloaded gradients
///
/// Manages storage of gradient partitions in CPU memory with
/// efficient retrieval and communication support.
pub struct CpuGradientStore {
    config: Zero3CpuOffloadConfig,
    stored_gradients: RwLock<HashMap<String, Tensor<f32>>>,
    memory_used: std::sync::atomic::AtomicUsize,
    gradient_metadata: Mutex<HashMap<String, GradientStoreMetadata>>,
}

impl CpuGradientStore {
    /// Create a new CPU gradient store
    pub fn new(config: &Zero3CpuOffloadConfig) -> TorshResult<Self> {
        info!(
            " CPU Gradient Store initialized with {} MB budget",
            config.cpu_memory_budget / (1024 * 1024)
        );

        Ok(Self {
            config: config.clone(),
            stored_gradients: RwLock::new(HashMap::new()),
            memory_used: std::sync::atomic::AtomicUsize::new(0),
            gradient_metadata: Mutex::new(HashMap::new()),
        })
    }

    /// Store a gradient partition
    pub async fn store(
        &self,
        layer_name: &str,
        partition_idx: usize,
        gradient: &Tensor<f32>,
    ) -> TorshResult<()> {
        let key = format!("{}_{}", layer_name, partition_idx);
        let grad_size = gradient.numel() * std::mem::size_of::<f32>();

        // Check memory budget
        let new_memory_usage = self.memory_used() + grad_size;
        if new_memory_usage > self.config.cpu_memory_budget {
            return Err(TorshDistributedError::memory_allocation_failed(
                new_memory_usage,
                "CPU memory budget exceeded for gradient storage",
            ));
        }

        {
            let mut grads = self.stored_gradients.write().unwrap();
            grads.insert(key.clone(), gradient.clone());
        }

        {
            let mut metadata = self.gradient_metadata.lock().unwrap();
            metadata.insert(
                key.clone(),
                GradientStoreMetadata {
                    layer_name: layer_name.to_string(),
                    partition_idx,
                    size_bytes: grad_size,
                    elements: gradient.numel(),
                },
            );
        }

        self.memory_used
            .fetch_add(grad_size, std::sync::atomic::Ordering::SeqCst);

        info!(
            "    Stored gradient partition '{}_{}' in CPU ({} bytes)",
            layer_name, partition_idx, grad_size
        );

        Ok(())
    }

    /// Get a specific gradient partition
    pub async fn get_gradient(
        &self,
        layer_name: &str,
        partition_idx: usize,
    ) -> TorshResult<Option<Tensor<f32>>> {
        let key = format!("{}_{}", layer_name, partition_idx);
        let grads = self.stored_gradients.read().unwrap();
        Ok(grads.get(&key).cloned())
    }

    /// Get all gradients
    pub async fn get_all_gradients(&self) -> TorshResult<HashMap<String, Tensor<f32>>> {
        let grads = self.stored_gradients.read().unwrap();
        Ok(grads.clone())
    }

    /// Get gradients owned by a specific rank
    pub async fn get_owned_gradients(
        &self,
        rank: usize,
        world_size: usize,
    ) -> TorshResult<HashMap<String, Tensor<f32>>> {
        let grads = self.stored_gradients.read().unwrap();

        // Filter gradients owned by this rank
        let owned_grads: HashMap<String, Tensor<f32>> = grads
            .iter()
            .filter(|(key, _)| {
                // Extract partition index from key and check ownership
                if let Some((_layer, partition_str)) = key.split_once('_') {
                    if let Ok(partition_idx) = partition_str.parse::<usize>() {
                        return partition_idx % world_size == rank;
                    }
                }
                false
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        Ok(owned_grads)
    }

    /// Store a reduced (aggregated) gradient
    pub async fn store_reduced_gradient(
        &self,
        key: &str,
        gradient: &Tensor<f32>,
    ) -> TorshResult<()> {
        let grad_size = gradient.numel() * std::mem::size_of::<f32>();

        {
            let mut grads = self.stored_gradients.write().unwrap();
            if let Some(old_gradient) = grads.insert(key.to_string(), gradient.clone()) {
                // Subtract old memory usage
                let old_size = old_gradient.numel() * std::mem::size_of::<f32>();
                self.memory_used
                    .fetch_sub(old_size, std::sync::atomic::Ordering::SeqCst);
            } else {
                // Add new memory usage
                self.memory_used
                    .fetch_add(grad_size, std::sync::atomic::Ordering::SeqCst);
            }
        }

        info!(
            "    Stored reduced gradient '{}' in CPU ({} bytes)",
            key, grad_size
        );

        Ok(())
    }

    /// Remove a gradient partition
    pub async fn remove_gradient(
        &self,
        layer_name: &str,
        partition_idx: usize,
    ) -> TorshResult<Option<Tensor<f32>>> {
        let key = format!("{}_{}", layer_name, partition_idx);

        let removed_gradient = {
            let mut grads = self.stored_gradients.write().unwrap();
            grads.remove(&key)
        };

        if let Some(ref gradient) = removed_gradient {
            let grad_size = gradient.numel() * std::mem::size_of::<f32>();
            self.memory_used
                .fetch_sub(grad_size, std::sync::atomic::Ordering::SeqCst);

            let mut metadata = self.gradient_metadata.lock().unwrap();
            metadata.remove(&key);
        }

        Ok(removed_gradient)
    }

    /// Clear all stored gradients
    pub async fn clear(&self) -> TorshResult<()> {
        {
            let mut grads = self.stored_gradients.write().unwrap();
            grads.clear();
        }

        {
            let mut metadata = self.gradient_metadata.lock().unwrap();
            metadata.clear();
        }

        self.memory_used
            .store(0, std::sync::atomic::Ordering::SeqCst);
        info!("   🗑️  Cleared all gradients from CPU store");

        Ok(())
    }

    /// Get current memory usage
    pub fn memory_used(&self) -> usize {
        self.memory_used.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get number of stored gradients
    pub fn gradient_count(&self) -> usize {
        self.stored_gradients.read().unwrap().len()
    }

    /// Get memory utilization as a percentage
    pub fn memory_utilization(&self) -> f32 {
        (self.memory_used() as f32) / (self.config.cpu_memory_budget as f32)
    }

    /// Get CPU gradient store statistics
    pub fn get_statistics(&self) -> CpuGradientStoreStats {
        let grads = self.stored_gradients.read().unwrap();
        let metadata = self.gradient_metadata.lock().unwrap();

        let total_elements: usize = metadata.values().map(|m| m.elements).sum();

        CpuGradientStoreStats {
            gradient_count: grads.len(),
            memory_used_bytes: self.memory_used(),
            memory_budget_bytes: self.config.cpu_memory_budget,
            memory_utilization: self.memory_utilization(),
            total_elements,
        }
    }
}

/// Metadata for stored gradients
#[derive(Debug, Clone)]
struct GradientStoreMetadata {
    layer_name: String,
    partition_idx: usize,
    size_bytes: usize,
    elements: usize,
}

/// Statistics about CPU gradient store
#[derive(Debug, Clone)]
pub struct CpuGradientStoreStats {
    pub gradient_count: usize,
    pub memory_used_bytes: usize,
    pub memory_budget_bytes: usize,
    pub memory_utilization: f32,
    pub total_elements: usize,
}

/// GPU gradient buffer for keeping gradients on GPU
///
/// Maintains gradients in GPU memory for immediate use during
/// backward passes and optimization steps.
pub struct GpuGradientBuffer {
    config: Zero3CpuOffloadConfig,
    stored_gradients: RwLock<HashMap<String, Tensor<f32>>>,
    memory_used: std::sync::atomic::AtomicUsize,
    buffer_metadata: Mutex<HashMap<String, GradientBufferMetadata>>,
}

impl GpuGradientBuffer {
    /// Create a new GPU gradient buffer
    pub fn new(config: &Zero3CpuOffloadConfig) -> TorshResult<Self> {
        info!(
            " GPU Gradient Buffer initialized with {} MB budget",
            config.gpu_param_memory_budget / (1024 * 1024)
        );

        Ok(Self {
            config: config.clone(),
            stored_gradients: RwLock::new(HashMap::new()),
            memory_used: std::sync::atomic::AtomicUsize::new(0),
            buffer_metadata: Mutex::new(HashMap::new()),
        })
    }

    /// Store a gradient partition in GPU buffer
    pub async fn store(
        &self,
        layer_name: &str,
        partition_idx: usize,
        gradient: &Tensor<f32>,
    ) -> TorshResult<()> {
        let key = format!("{}_{}", layer_name, partition_idx);
        let grad_size = gradient.numel() * std::mem::size_of::<f32>();

        // Check memory budget
        let new_memory_usage = self.memory_used() + grad_size;
        if new_memory_usage > self.config.gpu_param_memory_budget {
            return Err(TorshDistributedError::memory_allocation_failed(
                new_memory_usage,
                "GPU memory budget exceeded for gradient buffer",
            ));
        }

        {
            let mut grads = self.stored_gradients.write().unwrap();
            grads.insert(key.clone(), gradient.clone());
        }

        {
            let mut metadata = self.buffer_metadata.lock().unwrap();
            metadata.insert(
                key.clone(),
                GradientBufferMetadata {
                    layer_name: layer_name.to_string(),
                    partition_idx,
                    size_bytes: grad_size,
                    elements: gradient.numel(),
                },
            );
        }

        self.memory_used
            .fetch_add(grad_size, std::sync::atomic::Ordering::SeqCst);

        info!(
            "    Buffered gradient partition '{}_{}' in GPU ({} bytes)",
            layer_name, partition_idx, grad_size
        );

        Ok(())
    }

    /// Get a specific gradient partition
    pub async fn get_gradient(
        &self,
        layer_name: &str,
        partition_idx: usize,
    ) -> TorshResult<Option<Tensor<f32>>> {
        let key = format!("{}_{}", layer_name, partition_idx);
        let grads = self.stored_gradients.read().unwrap();
        Ok(grads.get(&key).cloned())
    }

    /// Get all gradients in the buffer
    pub async fn get_all_gradients(&self) -> TorshResult<HashMap<String, Tensor<f32>>> {
        let grads = self.stored_gradients.read().unwrap();
        Ok(grads.clone())
    }

    /// Remove a gradient partition from the buffer
    pub async fn remove_gradient(
        &self,
        layer_name: &str,
        partition_idx: usize,
    ) -> TorshResult<Option<Tensor<f32>>> {
        let key = format!("{}_{}", layer_name, partition_idx);

        let removed_gradient = {
            let mut grads = self.stored_gradients.write().unwrap();
            grads.remove(&key)
        };

        if let Some(ref gradient) = removed_gradient {
            let grad_size = gradient.numel() * std::mem::size_of::<f32>();
            self.memory_used
                .fetch_sub(grad_size, std::sync::atomic::Ordering::SeqCst);

            let mut metadata = self.buffer_metadata.lock().unwrap();
            metadata.remove(&key);
        }

        Ok(removed_gradient)
    }

    /// Get current memory usage
    pub fn memory_used(&self) -> usize {
        self.memory_used.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get number of buffered gradients
    pub fn gradient_count(&self) -> usize {
        self.stored_gradients.read().unwrap().len()
    }

    /// Clear all gradients from the buffer
    pub fn clear(&self) -> TorshResult<()> {
        {
            let mut grads = self.stored_gradients.write().unwrap();
            grads.clear();
        }

        {
            let mut metadata = self.buffer_metadata.lock().unwrap();
            metadata.clear();
        }

        self.memory_used
            .store(0, std::sync::atomic::Ordering::SeqCst);

        info!("   🗑️  Cleared all gradients from GPU buffer");

        Ok(())
    }

    /// Get memory utilization as a percentage
    pub fn memory_utilization(&self) -> f32 {
        (self.memory_used() as f32) / (self.config.gpu_param_memory_budget as f32)
    }

    /// Get GPU gradient buffer statistics
    pub fn get_statistics(&self) -> GpuGradientBufferStats {
        let grads = self.stored_gradients.read().unwrap();
        let metadata = self.buffer_metadata.lock().unwrap();

        let total_elements: usize = metadata.values().map(|m| m.elements).sum();

        GpuGradientBufferStats {
            gradient_count: grads.len(),
            memory_used_bytes: self.memory_used(),
            memory_budget_bytes: self.config.gpu_param_memory_budget,
            memory_utilization: self.memory_utilization(),
            total_elements,
        }
    }
}

/// Metadata for buffered gradients
#[derive(Debug, Clone)]
struct GradientBufferMetadata {
    layer_name: String,
    partition_idx: usize,
    size_bytes: usize,
    elements: usize,
}

/// Statistics about GPU gradient buffer
#[derive(Debug, Clone)]
pub struct GpuGradientBufferStats {
    pub gradient_count: usize,
    pub memory_used_bytes: usize,
    pub memory_budget_bytes: usize,
    pub memory_utilization: f32,
    pub total_elements: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_gradients() -> Result<(), Box<dyn std::error::Error>> {
        use torsh_tensor::Tensor;

        let weight_grad = Tensor::zeros(&[10, 5], torsh_core::DeviceType::Cpu)?;
        let bias_grad = Some(Tensor::zeros(&[5], torsh_core::DeviceType::Cpu)?);

        let param_grads = ParameterGradients::new(weight_grad, bias_grad);

        assert!(param_grads.has_bias());
        assert_eq!(param_grads.total_elements(), 55); // 10*5 + 5
        Ok(())
    }

    #[test]
    fn test_gradient_partition() -> Result<(), Box<dyn std::error::Error>> {
        use torsh_tensor::Tensor;

        let weight_grad = Tensor::zeros(&[20], torsh_core::DeviceType::Cpu)?;
        let bias_grad = Some(Tensor::zeros(&[5], torsh_core::DeviceType::Cpu)?);

        let partition = GradientPartition {
            layer_name: "layer1".to_string(),
            rank: 0,
            partition_idx: 0,
            start_idx: 0,
            end_idx: 20,
            size_elements: 20,
            weight_gradient: weight_grad,
            bias_gradient: bias_grad,
        };

        assert!(partition.has_bias());
        assert_eq!(partition.total_elements(), 25); // 20 + 5
        assert!(!partition.is_empty());
        Ok(())
    }

    #[test]
    fn test_gradient_partitioner() -> Result<(), Box<dyn std::error::Error>> {
        use torsh_tensor::Tensor;

        let config = Zero3CpuOffloadConfig::default();
        let rank_mapping = Zero3RankMapping::new(0, 4);

        let partitioner = GradientPartitioner::new(&config, &rank_mapping).unwrap();

        let weight_grad = Tensor::ones(&[100], torsh_core::DeviceType::Cpu)?;
        let param_grads = ParameterGradients::new(weight_grad, None);

        let partitions = partitioner
            .partition_gradients("layer1", &param_grads)
            .unwrap();

        assert_eq!(partitions.len(), 4); // 4 ranks
        assert_eq!(partitions[0].rank, 0);
        assert_eq!(partitions[0].size_elements, 25); // 100 / 4

        let owned = partitioner.get_owned_partition("layer1", &partitions);
        assert!(owned.is_some());
        assert_eq!(owned.unwrap().rank, 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_cpu_gradient_store() -> Result<(), Box<dyn std::error::Error>> {
        use torsh_tensor::Tensor;

        let config = Zero3CpuOffloadConfig::default();
        let store = CpuGradientStore::new(&config).unwrap();

        let gradient = Tensor::ones(&[100], torsh_core::DeviceType::Cpu)?;

        // Test store and get
        store.store("layer1", 0, &gradient).await.unwrap();
        assert_eq!(store.gradient_count(), 1);

        let retrieved = store.get_gradient("layer1", 0).await.unwrap();
        assert!(retrieved.is_some());

        // Test remove
        let removed = store.remove_gradient("layer1", 0).await.unwrap();
        assert!(removed.is_some());
        assert_eq!(store.gradient_count(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_gradient_buffer() -> Result<(), Box<dyn std::error::Error>> {
        use torsh_tensor::Tensor;

        let config = Zero3CpuOffloadConfig::default();
        let buffer = GpuGradientBuffer::new(&config).unwrap();

        let gradient = Tensor::ones(&[50], torsh_core::DeviceType::Cpu)?;

        // Test store and get
        buffer.store("layer1", 0, &gradient).await.unwrap();
        assert_eq!(buffer.gradient_count(), 1);

        let retrieved = buffer.get_gradient("layer1", 0).await.unwrap();
        assert!(retrieved.is_some());

        // Test clear
        buffer.clear().unwrap();
        assert_eq!(buffer.gradient_count(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_gradient_store_owned_gradients() -> Result<(), Box<dyn std::error::Error>> {
        use torsh_tensor::Tensor;

        let config = Zero3CpuOffloadConfig::default();
        let store = CpuGradientStore::new(&config).unwrap();

        // Store gradients for different partitions
        for i in 0..8 {
            let gradient = Tensor::ones(&[10], torsh_core::DeviceType::Cpu)?;
            store.store("layer1", i, &gradient).await.unwrap();
        }

        // Get gradients owned by rank 0 (partitions 0, 4 for world_size=4)
        let owned = store.get_owned_gradients(0, 4).await.unwrap();
        assert_eq!(owned.len(), 2); // Should own partitions 0 and 4

        // Check that we got the right partitions
        assert!(owned.contains_key("layer1_0"));
        assert!(owned.contains_key("layer1_4"));
        Ok(())
    }
}
