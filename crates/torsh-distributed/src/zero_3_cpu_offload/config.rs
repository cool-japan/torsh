//! ZeRO-3 Configuration and Core Types
//!
//! This module provides core configuration types and rank mapping functionality for
//! ZeRO-3 CPU offloading optimizations. It defines the configuration parameters,
//! compression methods, memory management strategies, and distributed rank mapping.

use std::collections::HashMap;

/// ZeRO-3 CPU offloading configuration
#[derive(Debug, Clone)]
pub struct Zero3CpuOffloadConfig {
    /// Enable parameter offloading to CPU
    pub offload_params: bool,
    /// Enable gradient offloading to CPU
    pub offload_grads: bool,
    /// Enable optimizer state offloading to CPU
    pub offload_optimizer_states: bool,
    /// CPU memory buffer size in bytes
    pub cpu_memory_budget: usize,
    /// GPU memory budget for parameters in bytes
    pub gpu_param_memory_budget: usize,
    /// Maximum GPU memory in MB (for memory pressure calculation)
    pub max_gpu_memory_mb: usize,
    /// Maximum CPU memory in MB (for memory pressure calculation)
    pub max_cpu_memory_mb: usize,
    /// Prefetch buffer size (number of parameters to prefetch)
    pub prefetch_buffer_size: usize,
    /// Enable asynchronous parameter prefetching
    pub async_prefetch: bool,
    /// Enable parameter overlapping (prefetch while computing)
    pub overlap_computation: bool,
    /// Pin CPU memory for faster transfers
    pub pin_cpu_memory: bool,
    /// Compression for CPU-stored parameters
    pub cpu_compression: CpuCompressionMethod,
    /// Automatic memory management strategy
    pub auto_memory_management: AutoMemoryStrategy,
}

impl Default for Zero3CpuOffloadConfig {
    fn default() -> Self {
        Self {
            offload_params: true,
            offload_grads: true,
            offload_optimizer_states: true,
            cpu_memory_budget: 32 * 1024 * 1024 * 1024, // 32GB
            gpu_param_memory_budget: 2 * 1024 * 1024 * 1024, // 2GB
            max_gpu_memory_mb: 8 * 1024,                // 8GB
            max_cpu_memory_mb: 64 * 1024,               // 64GB
            prefetch_buffer_size: 16,
            async_prefetch: true,
            overlap_computation: true,
            pin_cpu_memory: true,
            cpu_compression: CpuCompressionMethod::None,
            auto_memory_management: AutoMemoryStrategy::Aggressive,
        }
    }
}

impl Zero3CpuOffloadConfig {
    /// Create a new configuration with custom settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set parameter offloading option
    pub fn with_offload_params(mut self, offload: bool) -> Self {
        self.offload_params = offload;
        self
    }

    /// Set gradient offloading option
    pub fn with_offload_grads(mut self, offload: bool) -> Self {
        self.offload_grads = offload;
        self
    }

    /// Set optimizer state offloading option
    pub fn with_offload_optimizer_states(mut self, offload: bool) -> Self {
        self.offload_optimizer_states = offload;
        self
    }

    /// Set CPU memory budget
    pub fn with_cpu_memory_budget(mut self, budget: usize) -> Self {
        self.cpu_memory_budget = budget;
        self
    }

    /// Set GPU parameter memory budget
    pub fn with_gpu_param_memory_budget(mut self, budget: usize) -> Self {
        self.gpu_param_memory_budget = budget;
        self
    }

    /// Set prefetch buffer size
    pub fn with_prefetch_buffer_size(mut self, size: usize) -> Self {
        self.prefetch_buffer_size = size;
        self
    }

    /// Set compression method
    pub fn with_compression(mut self, compression: CpuCompressionMethod) -> Self {
        self.cpu_compression = compression;
        self
    }

    /// Set memory management strategy
    pub fn with_memory_strategy(mut self, strategy: AutoMemoryStrategy) -> Self {
        self.auto_memory_management = strategy;
        self
    }

    /// Enable asynchronous prefetching
    pub fn with_async_prefetch(mut self, async_prefetch: bool) -> Self {
        self.async_prefetch = async_prefetch;
        self
    }

    /// Enable computation overlap
    pub fn with_overlap_computation(mut self, overlap: bool) -> Self {
        self.overlap_computation = overlap;
        self
    }

    /// Enable CPU memory pinning
    pub fn with_pin_cpu_memory(mut self, pin: bool) -> Self {
        self.pin_cpu_memory = pin;
        self
    }

    /// Validate the configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.cpu_memory_budget == 0 {
            return Err("CPU memory budget cannot be zero".to_string());
        }

        if self.gpu_param_memory_budget == 0 {
            return Err("GPU parameter memory budget cannot be zero".to_string());
        }

        if self.prefetch_buffer_size == 0 {
            return Err("Prefetch buffer size cannot be zero".to_string());
        }

        if self.max_gpu_memory_mb == 0 {
            return Err("Maximum GPU memory cannot be zero".to_string());
        }

        if self.max_cpu_memory_mb == 0 {
            return Err("Maximum CPU memory cannot be zero".to_string());
        }

        // Check that GPU memory budget doesn't exceed maximum GPU memory
        let gpu_budget_mb = self.gpu_param_memory_budget / (1024 * 1024);
        if gpu_budget_mb > self.max_gpu_memory_mb {
            return Err(format!(
                "GPU parameter memory budget ({} MB) exceeds maximum GPU memory ({} MB)",
                gpu_budget_mb, self.max_gpu_memory_mb
            ));
        }

        // Check that CPU memory budget doesn't exceed maximum CPU memory
        let cpu_budget_mb = self.cpu_memory_budget / (1024 * 1024);
        if cpu_budget_mb > self.max_cpu_memory_mb {
            return Err(format!(
                "CPU memory budget ({} MB) exceeds maximum CPU memory ({} MB)",
                cpu_budget_mb, self.max_cpu_memory_mb
            ));
        }

        Ok(())
    }

    /// Get the effective compression ratio for CPU storage
    pub fn compression_ratio(&self) -> f32 {
        match self.cpu_compression {
            CpuCompressionMethod::None => 1.0,
            CpuCompressionMethod::FP16 => 0.5,
            CpuCompressionMethod::BF16 => 0.5,
            CpuCompressionMethod::INT8 => 0.25,
            CpuCompressionMethod::Quantization => 0.25,
            CpuCompressionMethod::LosslessCompression => 0.7, // Typical lossless compression ratio
        }
    }

    /// Get the effective CPU memory budget after compression
    pub fn effective_cpu_memory_budget(&self) -> usize {
        (self.cpu_memory_budget as f32 / self.compression_ratio()) as usize
    }
}

/// Compression methods for CPU-stored data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuCompressionMethod {
    /// No compression
    None,
    /// 16-bit floating point compression
    FP16,
    /// BFloat16 compression
    BF16,
    /// 8-bit integer quantization
    INT8,
    /// Advanced quantization schemes
    Quantization,
    /// Lossless compression (e.g., LZ4, Snappy)
    LosslessCompression,
}

impl CpuCompressionMethod {
    /// Get the compression ratio (0.0 to 1.0, where 1.0 means no compression)
    pub fn ratio(&self) -> f32 {
        match self {
            CpuCompressionMethod::None => 1.0,
            CpuCompressionMethod::FP16 => 0.5,
            CpuCompressionMethod::BF16 => 0.5,
            CpuCompressionMethod::INT8 => 0.25,
            CpuCompressionMethod::Quantization => 0.25,
            CpuCompressionMethod::LosslessCompression => 0.7,
        }
    }

    /// Check if this compression method is lossy
    pub fn is_lossy(&self) -> bool {
        matches!(
            self,
            CpuCompressionMethod::FP16
                | CpuCompressionMethod::BF16
                | CpuCompressionMethod::INT8
                | CpuCompressionMethod::Quantization
        )
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            CpuCompressionMethod::None => "No compression",
            CpuCompressionMethod::FP16 => "16-bit floating point",
            CpuCompressionMethod::BF16 => "BFloat16",
            CpuCompressionMethod::INT8 => "8-bit integer quantization",
            CpuCompressionMethod::Quantization => "Advanced quantization",
            CpuCompressionMethod::LosslessCompression => "Lossless compression",
        }
    }
}

/// Automatic memory management strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoMemoryStrategy {
    /// Conservative memory management - minimal offloading
    Conservative,
    /// Balanced approach - moderate offloading based on memory pressure
    Balanced,
    /// Aggressive offloading - maximize CPU utilization
    Aggressive,
    /// Extreme offloading - offload everything possible
    Extreme,
}

impl AutoMemoryStrategy {
    /// Get the memory pressure threshold for triggering offloading
    pub fn pressure_threshold(&self) -> f32 {
        match self {
            AutoMemoryStrategy::Conservative => 0.9, // 90% memory pressure
            AutoMemoryStrategy::Balanced => 0.75,    // 75% memory pressure
            AutoMemoryStrategy::Aggressive => 0.6,   // 60% memory pressure
            AutoMemoryStrategy::Extreme => 0.4,      // 40% memory pressure
        }
    }

    /// Get the offloading aggressiveness factor (0.0 to 1.0)
    pub fn aggressiveness(&self) -> f32 {
        match self {
            AutoMemoryStrategy::Conservative => 0.2,
            AutoMemoryStrategy::Balanced => 0.5,
            AutoMemoryStrategy::Aggressive => 0.8,
            AutoMemoryStrategy::Extreme => 1.0,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            AutoMemoryStrategy::Conservative => "Conservative - minimal offloading",
            AutoMemoryStrategy::Balanced => "Balanced - moderate offloading",
            AutoMemoryStrategy::Aggressive => "Aggressive - maximize CPU utilization",
            AutoMemoryStrategy::Extreme => "Extreme - offload everything possible",
        }
    }
}

/// ZeRO-3 rank mapping for parameter partitioning
#[derive(Debug, Clone)]
pub struct Zero3RankMapping {
    rank: usize,
    world_size: usize,
}

impl Zero3RankMapping {
    /// Create a new rank mapping
    pub fn new(rank: usize, world_size: usize) -> Self {
        assert!(rank < world_size, "Rank must be less than world size");
        Self { rank, world_size }
    }

    /// Get the current rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Check if this rank owns a specific partition
    pub fn owns_partition(&self, partition_idx: usize) -> bool {
        partition_idx % self.world_size == self.rank
    }

    /// Get the owner rank for a parameter
    pub fn get_parameter_owner(&self, param_idx: usize) -> usize {
        param_idx % self.world_size
    }

    /// Get all partitions owned by this rank
    pub fn owned_partitions(&self, total_partitions: usize) -> Vec<usize> {
        (0..total_partitions)
            .filter(|&i| self.owns_partition(i))
            .collect()
    }

    /// Get the number of partitions owned by this rank
    pub fn owned_partition_count(&self, total_partitions: usize) -> usize {
        let base_count = total_partitions / self.world_size;
        let remainder = total_partitions % self.world_size;

        if self.rank < remainder {
            base_count + 1
        } else {
            base_count
        }
    }

    /// Map a global parameter index to a local partition index
    pub fn global_to_local_partition(&self, global_idx: usize) -> Option<usize> {
        if self.owns_partition(global_idx) {
            Some(global_idx / self.world_size)
        } else {
            None
        }
    }

    /// Map a local partition index to a global parameter index
    pub fn local_to_global_partition(&self, local_idx: usize) -> usize {
        local_idx * self.world_size + self.rank
    }

    /// Get ranks that need to participate in communication for a given parameter set
    pub fn communication_group(&self, param_indices: &[usize]) -> Vec<usize> {
        let mut ranks = std::collections::HashSet::new();
        for &param_idx in param_indices {
            ranks.insert(self.get_parameter_owner(param_idx));
        }
        let mut result: Vec<usize> = ranks.into_iter().collect();
        result.sort();
        result
    }
}

/// Model parameters for ZeRO-3 initialization
#[derive(Debug)]
pub struct ModelParameters {
    pub parameter_count: usize,
    pub parameter_names: Vec<String>,
    pub parameter_shapes: HashMap<String, Vec<usize>>,
    pub total_memory_bytes: usize,
}

impl ModelParameters {
    /// Create a new empty model parameters collection
    pub fn new() -> Self {
        Self {
            parameter_count: 0,
            parameter_names: Vec::new(),
            parameter_shapes: HashMap::new(),
            total_memory_bytes: 0,
        }
    }

    /// Add a parameter to the collection
    pub fn add_parameter(&mut self, name: String, shape: Vec<usize>) {
        let param_size = shape.iter().product::<usize>();
        self.parameter_count += param_size;
        self.total_memory_bytes += param_size * std::mem::size_of::<f32>();
        self.parameter_shapes.insert(name.clone(), shape);
        self.parameter_names.push(name);
    }

    /// Check if a parameter exists
    pub fn has_parameter(&self, name: &str) -> bool {
        self.parameter_shapes.contains_key(name)
    }

    /// Add a parameter with custom element size
    pub fn add_parameter_with_size(
        &mut self,
        name: String,
        shape: Vec<usize>,
        element_size: usize,
    ) {
        let param_size = shape.iter().product::<usize>();
        self.parameter_count += param_size;
        self.total_memory_bytes += param_size * element_size;
        self.parameter_shapes.insert(name.clone(), shape);
        self.parameter_names.push(name);
    }

    /// Get the shape of a parameter by name
    pub fn get_parameter_shape(&self, name: &str) -> Option<&Vec<usize>> {
        self.parameter_shapes.get(name)
    }

    /// Get the number of elements in a parameter
    pub fn get_parameter_size(&self, name: &str) -> Option<usize> {
        self.parameter_shapes
            .get(name)
            .map(|shape| shape.iter().product::<usize>())
    }

    /// Get total number of parameters
    pub fn total_parameters(&self) -> usize {
        self.parameter_names.len()
    }

    /// Get memory usage in MB
    pub fn memory_usage_mb(&self) -> f64 {
        self.total_memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get parameter statistics
    pub fn get_statistics(&self) -> ModelParameterStats {
        if self.parameter_names.is_empty() {
            return ModelParameterStats::default();
        }

        let mut sizes: Vec<usize> = self
            .parameter_shapes
            .values()
            .map(|shape| shape.iter().product::<usize>())
            .collect();
        sizes.sort();

        let total_elements = sizes.iter().sum::<usize>();
        let mean_size = total_elements as f64 / sizes.len() as f64;
        let median_size = if sizes.len() % 2 == 0 {
            (sizes[sizes.len() / 2 - 1] + sizes[sizes.len() / 2]) as f64 / 2.0
        } else {
            sizes[sizes.len() / 2] as f64
        };

        ModelParameterStats {
            total_parameters: self.parameter_names.len(),
            total_elements,
            mean_parameter_size: mean_size,
            median_parameter_size: median_size,
            min_parameter_size: *sizes.first().unwrap_or(&0),
            max_parameter_size: *sizes.last().unwrap_or(&0),
            total_memory_bytes: self.total_memory_bytes,
        }
    }
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about model parameters
#[derive(Debug, Clone)]
pub struct ModelParameterStats {
    pub total_parameters: usize,
    pub total_elements: usize,
    pub mean_parameter_size: f64,
    pub median_parameter_size: f64,
    pub min_parameter_size: usize,
    pub max_parameter_size: usize,
    pub total_memory_bytes: usize,
}

impl Default for ModelParameterStats {
    fn default() -> Self {
        Self {
            total_parameters: 0,
            total_elements: 0,
            mean_parameter_size: 0.0,
            median_parameter_size: 0.0,
            min_parameter_size: 0,
            max_parameter_size: 0,
            total_memory_bytes: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero3_config_default() {
        let config = Zero3CpuOffloadConfig::default();
        assert!(config.offload_params);
        assert!(config.offload_grads);
        assert!(config.offload_optimizer_states);
        assert!(config.async_prefetch);
        assert_eq!(config.cpu_compression, CpuCompressionMethod::None);
        assert_eq!(
            config.auto_memory_management,
            AutoMemoryStrategy::Aggressive
        );
    }

    #[test]
    fn test_zero3_config_builder() {
        let config = Zero3CpuOffloadConfig::new()
            .with_offload_params(false)
            .with_compression(CpuCompressionMethod::FP16)
            .with_memory_strategy(AutoMemoryStrategy::Conservative)
            .with_prefetch_buffer_size(32);

        assert!(!config.offload_params);
        assert_eq!(config.cpu_compression, CpuCompressionMethod::FP16);
        assert_eq!(
            config.auto_memory_management,
            AutoMemoryStrategy::Conservative
        );
        assert_eq!(config.prefetch_buffer_size, 32);
    }

    #[test]
    fn test_zero3_config_validation() {
        let config = Zero3CpuOffloadConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.cpu_memory_budget = 0;
        assert!(invalid_config.validate().is_err());

        let mut invalid_config = config.clone();
        invalid_config.gpu_param_memory_budget = 0;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_compression_methods() {
        assert_eq!(CpuCompressionMethod::None.ratio(), 1.0);
        assert_eq!(CpuCompressionMethod::FP16.ratio(), 0.5);
        assert_eq!(CpuCompressionMethod::INT8.ratio(), 0.25);

        assert!(!CpuCompressionMethod::None.is_lossy());
        assert!(CpuCompressionMethod::FP16.is_lossy());
        assert!(!CpuCompressionMethod::LosslessCompression.is_lossy());
    }

    #[test]
    fn test_memory_strategies() {
        assert_eq!(AutoMemoryStrategy::Conservative.pressure_threshold(), 0.9);
        assert_eq!(AutoMemoryStrategy::Aggressive.pressure_threshold(), 0.6);

        assert_eq!(AutoMemoryStrategy::Conservative.aggressiveness(), 0.2);
        assert_eq!(AutoMemoryStrategy::Extreme.aggressiveness(), 1.0);
    }

    #[test]
    fn test_rank_mapping() {
        let mapping = Zero3RankMapping::new(1, 4);

        assert_eq!(mapping.rank(), 1);
        assert_eq!(mapping.world_size(), 4);

        assert!(mapping.owns_partition(1)); // 1 % 4 == 1
        assert!(mapping.owns_partition(5)); // 5 % 4 == 1
        assert!(!mapping.owns_partition(0)); // 0 % 4 != 1
        assert!(!mapping.owns_partition(2)); // 2 % 4 != 1

        assert_eq!(mapping.get_parameter_owner(5), 1);
        assert_eq!(mapping.get_parameter_owner(8), 0);

        let owned = mapping.owned_partitions(10);
        assert_eq!(owned, vec![1, 5, 9]);

        assert_eq!(mapping.owned_partition_count(10), 3); // 10 partitions, rank 1 gets 3
        assert_eq!(mapping.owned_partition_count(8), 2); // 8 partitions, rank 1 gets 2
    }

    #[test]
    fn test_model_parameters() {
        let mut params = ModelParameters::new();

        params.add_parameter("layer1.weight".to_string(), vec![100, 50]);
        params.add_parameter("layer1.bias".to_string(), vec![50]);

        assert_eq!(params.total_parameters(), 2);
        assert_eq!(params.parameter_count, 5050); // 100*50 + 50
        assert_eq!(params.get_parameter_size("layer1.weight"), Some(5000));
        assert_eq!(params.get_parameter_size("layer1.bias"), Some(50));

        let stats = params.get_statistics();
        assert_eq!(stats.total_parameters, 2);
        assert_eq!(stats.total_elements, 5050);
        assert_eq!(stats.min_parameter_size, 50);
        assert_eq!(stats.max_parameter_size, 5000);
    }

    #[test]
    fn test_rank_mapping_communication_group() {
        let mapping = Zero3RankMapping::new(1, 4);
        let param_indices = vec![0, 1, 4, 5, 8, 9];
        let comm_group = mapping.communication_group(&param_indices);

        // Parameters owned by: 0->rank0, 1->rank1, 4->rank0, 5->rank1, 8->rank0, 9->rank1
        // So communication group should be [0, 1]
        assert_eq!(comm_group, vec![0, 1]);
    }

    #[test]
    fn test_effective_cpu_memory_budget() {
        let config = Zero3CpuOffloadConfig::new()
            .with_cpu_memory_budget(1000)
            .with_compression(CpuCompressionMethod::FP16);

        // With FP16 compression (0.5 ratio), effective budget should be 2000
        assert_eq!(config.effective_cpu_memory_budget(), 2000);

        let config_no_compression = Zero3CpuOffloadConfig::new()
            .with_cpu_memory_budget(1000)
            .with_compression(CpuCompressionMethod::None);

        assert_eq!(config_no_compression.effective_cpu_memory_budget(), 1000);
    }
}
