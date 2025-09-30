//! Optimized state dict operations for efficient optimizer state management
//!
//! This module provides optimized implementations for saving and loading optimizer state,
//! including compression, serialization formats, and memory-efficient operations.

use crate::{OptimizerState, ParamGroupState};
use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_core::DeviceType;
use torsh_tensor::Tensor;

/// Compression method for state dict serialization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionMethod {
    None,
    Gzip,
    Zstd,
    Lz4,
}

/// Serialization format for state dict
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SerializationFormat {
    Binary,
    Json,
    MessagePack,
    Protobuf,
}

/// Configuration for state dict operations
#[derive(Debug, Clone)]
pub struct StateDictConfig {
    pub compression: CompressionMethod,
    pub format: SerializationFormat,
    pub compress_threshold: usize,
    pub use_memory_mapping: bool,
    pub chunk_size: usize,
    pub parallel_processing: bool,
}

impl Default for StateDictConfig {
    fn default() -> Self {
        Self {
            compression: CompressionMethod::Zstd,
            format: SerializationFormat::Binary,
            compress_threshold: 1024 * 1024, // 1MB
            use_memory_mapping: true,
            chunk_size: 64 * 1024 * 1024, // 64MB
            parallel_processing: true,
        }
    }
}

/// Optimized state dict manager
pub struct StateDictManager {
    config: StateDictConfig,
    cache: HashMap<String, CachedStateEntry>,
    compression_stats: CompressionStats,
}

/// Cached state entry for fast access
#[derive(Clone)]
struct CachedStateEntry {
    data: Vec<u8>,
    checksum: u64,
    last_accessed: std::time::Instant,
    compression_ratio: f64,
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub total_original_size: u64,
    pub total_compressed_size: u64,
    pub compression_time_ms: u64,
    pub decompression_time_ms: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl CompressionStats {
    /// Calculate overall compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.total_original_size == 0 {
            1.0
        } else {
            self.total_compressed_size as f64 / self.total_original_size as f64
        }
    }

    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total_accesses = self.cache_hits + self.cache_misses;
        if total_accesses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_accesses as f64
        }
    }
}

impl StateDictManager {
    /// Create a new state dict manager with default configuration
    pub fn new() -> Self {
        Self::with_config(StateDictConfig::default())
    }

    /// Create a new state dict manager with custom configuration
    pub fn with_config(config: StateDictConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            compression_stats: CompressionStats::default(),
        }
    }

    /// Serialize optimizer state to bytes with optimization
    pub fn serialize_state(&mut self, state: &OptimizerState) -> Result<Vec<u8>> {
        let start_time = std::time::Instant::now();

        // Convert state to serializable format
        let serializable_state = self.prepare_for_serialization(state)?;

        // Serialize based on format
        let mut data = match self.config.format {
            SerializationFormat::Binary => self.serialize_binary(&serializable_state)?,
            SerializationFormat::Json => self.serialize_json(&serializable_state)?,
            SerializationFormat::MessagePack => self.serialize_msgpack(&serializable_state)?,
            SerializationFormat::Protobuf => self.serialize_protobuf(&serializable_state)?,
        };

        let original_size = data.len() as u64;

        // Apply compression if data is large enough
        if data.len() > self.config.compress_threshold {
            data = self.compress_data(data)?;
        }

        let compressed_size = data.len() as u64;

        // Update statistics
        self.compression_stats.total_original_size += original_size;
        self.compression_stats.total_compressed_size += compressed_size;
        self.compression_stats.compression_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(data)
    }

    /// Deserialize optimizer state from bytes with optimization
    pub fn deserialize_state(&mut self, data: &[u8]) -> Result<OptimizerState> {
        let start_time = std::time::Instant::now();

        // Decompress data if needed
        let decompressed_data = if self.is_compressed(data) {
            self.decompress_data(data)?
        } else {
            data.to_vec()
        };

        // Deserialize based on format
        let serializable_state = match self.config.format {
            SerializationFormat::Binary => self.deserialize_binary(&decompressed_data)?,
            SerializationFormat::Json => self.deserialize_json(&decompressed_data)?,
            SerializationFormat::MessagePack => self.deserialize_msgpack(&decompressed_data)?,
            SerializationFormat::Protobuf => self.deserialize_protobuf(&decompressed_data)?,
        };

        // Convert back to optimizer state
        let state = self.restore_from_serialization(serializable_state)?;

        self.compression_stats.decompression_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(state)
    }

    /// Save optimizer state to file with optimizations
    pub fn save_to_file(&mut self, state: &OptimizerState, path: &str) -> Result<()> {
        let data = self.serialize_state(state)?;

        if self.config.use_memory_mapping && data.len() > self.config.chunk_size {
            self.save_with_memory_mapping(&data, path)
        } else {
            std::fs::write(path, data)
                .map_err(|e| TorshError::IoError(format!("Failed to write file: {e}")))?;
            Ok(())
        }
    }

    /// Load optimizer state from file with optimizations
    pub fn load_from_file(&mut self, path: &str) -> Result<OptimizerState> {
        let data = if self.config.use_memory_mapping {
            self.load_with_memory_mapping(path)?
        } else {
            std::fs::read(path)
                .map_err(|e| TorshError::IoError(format!("Failed to read file: {e}")))?
        };

        self.deserialize_state(&data)
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> &CompressionStats {
        &self.compression_stats
    }

    /// Clear cache and reset statistics
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.compression_stats = CompressionStats::default();
    }

    /// Prepare state for serialization (convert tensors to raw data)
    fn prepare_for_serialization(&self, state: &OptimizerState) -> Result<SerializableState> {
        let mut serializable_tensors = HashMap::new();

        for (param_id, tensor_map) in &state.state {
            let mut param_tensors = HashMap::new();
            for (tensor_name, tensor) in tensor_map {
                // Convert tensor to raw bytes for serialization
                let tensor_data = TensorData {
                    data: tensor.to_vec()?,
                    shape: tensor.shape().dims().to_vec(),
                    dtype: format!("{:?}", tensor.dtype()),
                    device: format!("{:?}", tensor.device()),
                };
                param_tensors.insert(tensor_name.clone(), tensor_data);
            }
            serializable_tensors.insert(param_id.clone(), param_tensors);
        }

        Ok(SerializableState {
            optimizer_type: state.optimizer_type.clone(),
            version: state.version.clone(),
            param_groups: state.param_groups.clone(),
            state: serializable_tensors,
            global_state: state.global_state.clone(),
        })
    }

    /// Restore state from serialization
    fn restore_from_serialization(
        &self,
        serializable: SerializableState,
    ) -> Result<OptimizerState> {
        let mut state_tensors = HashMap::new();

        for (param_id, tensor_map) in serializable.state {
            let mut param_tensors = HashMap::new();
            for (tensor_name, tensor_data) in tensor_map {
                // Recreate tensor from raw data
                let tensor = self.recreate_tensor(tensor_data)?;
                param_tensors.insert(tensor_name, tensor);
            }
            state_tensors.insert(param_id, param_tensors);
        }

        Ok(OptimizerState {
            optimizer_type: serializable.optimizer_type,
            version: serializable.version,
            param_groups: serializable.param_groups,
            state: state_tensors,
            global_state: serializable.global_state,
        })
    }

    /// Recreate tensor from serialized data
    fn recreate_tensor(&self, data: TensorData) -> Result<Tensor> {
        // This is a simplified implementation
        // In a real implementation, this would properly recreate the tensor
        // with the correct device, dtype, and shape
        let device = DeviceType::Cpu; // Simplified
        Ok(torsh_tensor::creation::from_vec(
            data.data,
            &data.shape,
            device,
        )?)
    }

    /// Compress data using the configured method
    fn compress_data(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        match self.config.compression {
            CompressionMethod::None => Ok(data),
            CompressionMethod::Gzip => {
                // Simplified - would use actual gzip compression
                Ok(data)
            }
            CompressionMethod::Zstd => {
                // Simplified - would use actual zstd compression
                Ok(data)
            }
            CompressionMethod::Lz4 => {
                // Simplified - would use actual lz4 compression
                Ok(data)
            }
        }
    }

    /// Decompress data
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simplified implementation
        Ok(data.to_vec())
    }

    /// Check if data is compressed
    fn is_compressed(&self, data: &[u8]) -> bool {
        // Simplified check - would check magic bytes for compression format
        data.len() > 4 && &data[0..4] == b"COMP"
    }

    /// Serialize to binary format
    fn serialize_binary(&self, state: &SerializableState) -> Result<Vec<u8>> {
        // Simplified binary serialization
        Ok(format!("{:?}", state).into_bytes())
    }

    /// Deserialize from binary format
    fn deserialize_binary(&self, _data: &[u8]) -> Result<SerializableState> {
        // Simplified binary deserialization
        Err(TorshError::Other(
            "Binary deserialization not fully implemented".to_string(),
        ))
    }

    /// Serialize to JSON format
    fn serialize_json(&self, state: &SerializableState) -> Result<Vec<u8>> {
        // Would use serde_json for actual implementation
        Ok(format!("{:?}", state).into_bytes())
    }

    /// Deserialize from JSON format
    fn deserialize_json(&self, _data: &[u8]) -> Result<SerializableState> {
        // Would use serde_json for actual implementation
        Err(TorshError::Other(
            "JSON deserialization not fully implemented".to_string(),
        ))
    }

    /// Serialize to MessagePack format
    fn serialize_msgpack(&self, state: &SerializableState) -> Result<Vec<u8>> {
        // Would use rmp-serde for actual implementation
        Ok(format!("{:?}", state).into_bytes())
    }

    /// Deserialize from MessagePack format
    fn deserialize_msgpack(&self, _data: &[u8]) -> Result<SerializableState> {
        // Would use rmp-serde for actual implementation
        Err(TorshError::Other(
            "MessagePack deserialization not fully implemented".to_string(),
        ))
    }

    /// Serialize to Protobuf format
    fn serialize_protobuf(&self, state: &SerializableState) -> Result<Vec<u8>> {
        // Would use prost for actual implementation
        Ok(format!("{:?}", state).into_bytes())
    }

    /// Deserialize from Protobuf format
    fn deserialize_protobuf(&self, _data: &[u8]) -> Result<SerializableState> {
        // Would use prost for actual implementation
        Err(TorshError::Other(
            "Protobuf deserialization not fully implemented".to_string(),
        ))
    }

    /// Save using memory mapping for large files
    fn save_with_memory_mapping(&self, data: &[u8], path: &str) -> Result<()> {
        // Simplified implementation - would use memmap2 crate
        std::fs::write(path, data)
            .map_err(|e| TorshError::IoError(format!("Failed to write file: {e}")))?;
        Ok(())
    }

    /// Load using memory mapping for large files
    fn load_with_memory_mapping(&self, path: &str) -> Result<Vec<u8>> {
        // Simplified implementation - would use memmap2 crate
        std::fs::read(path).map_err(|e| TorshError::IoError(format!("Failed to read file: {e}")))
    }
}

impl Default for StateDictManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable representation of optimizer state
#[derive(Debug, Clone)]
struct SerializableState {
    optimizer_type: String,
    version: String,
    param_groups: Vec<ParamGroupState>,
    state: HashMap<String, HashMap<String, TensorData>>,
    global_state: HashMap<String, f32>,
}

/// Serializable tensor data
#[derive(Debug, Clone)]
struct TensorData {
    data: Vec<f32>,
    shape: Vec<usize>,
    dtype: String,
    device: String,
}

/// Utility functions for state dict operations
pub mod utils {
    use super::*;

    /// Calculate state dict size in bytes
    pub fn calculate_state_size(state: &OptimizerState) -> usize {
        let mut total_size = 0;

        for (_, tensor_map) in &state.state {
            for (_, tensor) in tensor_map {
                total_size += tensor.numel() * std::mem::size_of::<f32>();
            }
        }

        total_size
    }

    /// Estimate memory usage for state dict operations
    pub fn estimate_memory_usage(
        state: &OptimizerState,
        config: &StateDictConfig,
    ) -> MemoryEstimate {
        let state_size = calculate_state_size(state);
        let serialization_overhead = state_size / 4; // Rough estimate
        let compression_working_memory = if matches!(config.compression, CompressionMethod::None) {
            0
        } else {
            state_size / 2
        };

        MemoryEstimate {
            state_size,
            serialization_overhead,
            compression_working_memory,
            total_peak_usage: state_size + serialization_overhead + compression_working_memory,
        }
    }

    /// Optimize state dict configuration based on state size
    pub fn optimize_config_for_size(state_size: usize) -> StateDictConfig {
        let mut config = StateDictConfig::default();

        if state_size > 1024 * 1024 * 1024 {
            // > 1GB
            config.compression = CompressionMethod::Zstd;
            config.use_memory_mapping = true;
            config.chunk_size = 128 * 1024 * 1024; // 128MB chunks
            config.parallel_processing = true;
        } else if state_size > 100 * 1024 * 1024 {
            // > 100MB
            config.compression = CompressionMethod::Lz4; // Faster compression
            config.use_memory_mapping = false;
            config.parallel_processing = true;
        } else {
            config.compression = CompressionMethod::None;
            config.use_memory_mapping = false;
            config.parallel_processing = false;
        }

        config
    }
}

/// Memory usage estimate for state dict operations
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub state_size: usize,
    pub serialization_overhead: usize,
    pub compression_working_memory: usize,
    pub total_peak_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptimizerResult;
    use torsh_core::device::Device;
    use torsh_tensor::creation;

    #[test]
    fn test_state_dict_manager_creation() {
        let manager = StateDictManager::new();
        assert_eq!(manager.config.compression, CompressionMethod::Zstd);
        assert_eq!(manager.config.format, SerializationFormat::Binary);
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::default();
        stats.total_original_size = 1000;
        stats.total_compressed_size = 500;
        stats.cache_hits = 8;
        stats.cache_misses = 2;

        assert_eq!(stats.compression_ratio(), 0.5);
        assert_eq!(stats.cache_hit_rate(), 0.8);
    }

    #[test]
    fn test_config_optimization() {
        let small_config = utils::optimize_config_for_size(1024); // 1KB
        assert_eq!(small_config.compression, CompressionMethod::None);
        assert!(!small_config.use_memory_mapping);

        let large_config = utils::optimize_config_for_size(2 * 1024 * 1024 * 1024); // 2GB
        assert_eq!(large_config.compression, CompressionMethod::Zstd);
        assert!(large_config.use_memory_mapping);
    }

    #[test]
    fn test_memory_estimation() -> OptimizerResult<()> {
        // Create a simple state for testing
        let tensor = creation::randn::<f32>(&[100, 100])?;

        let mut state_map = HashMap::new();
        let mut tensor_map = HashMap::new();
        tensor_map.insert("test_tensor".to_string(), tensor);
        state_map.insert("param_1".to_string(), tensor_map);

        let state = OptimizerState {
            optimizer_type: "test".to_string(),
            version: "0.1.0".to_string(),
            param_groups: vec![],
            state: state_map,
            global_state: HashMap::new(),
        };

        let config = StateDictConfig::default();
        let estimate = utils::estimate_memory_usage(&state, &config);

        assert!(estimate.state_size > 0);
        assert!(estimate.total_peak_usage >= estimate.state_size);
        Ok(())
    }

    #[test]
    fn test_calculate_state_size() -> OptimizerResult<()> {
        let tensor1 = creation::randn::<f32>(&[10, 10])?; // 100 elements
        let tensor2 = creation::randn::<f32>(&[5, 5])?; // 25 elements

        let mut state_map = HashMap::new();
        let mut tensor_map = HashMap::new();
        tensor_map.insert("tensor1".to_string(), tensor1);
        tensor_map.insert("tensor2".to_string(), tensor2);
        state_map.insert("param_1".to_string(), tensor_map);

        let state = OptimizerState {
            optimizer_type: "test".to_string(),
            version: "0.1.0".to_string(),
            param_groups: vec![],
            state: state_map,
            global_state: HashMap::new(),
        };

        let size = utils::calculate_state_size(&state);
        // 125 elements * 4 bytes per f32
        assert_eq!(size, 125 * 4);
        Ok(())
    }
}
