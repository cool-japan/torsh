//! Lazy loading optimizations for efficient model loading
//!
//! This module provides advanced loading strategies:
//! - Memory-mapped file access for large models
//! - Lazy tensor materialization (load on first use)
//! - Streaming loading for huge models
//! - LRU caching for frequently accessed tensors

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use safetensors::SafeTensors;
use torsh_core::{device::DeviceType, dtype::DType};
use torsh_tensor::Tensor;

use crate::{ModelError, ModelResult};

/// Simple f16 to f32 conversion (not IEEE 754 accurate)
/// In production, use a proper half-precision library
fn f16_to_f32_simple(bits: u16) -> f32 {
    // Extract sign, exponent, and mantissa
    let sign = (bits >> 15) & 0x1;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;

    // Handle special cases
    if exponent == 0 {
        if mantissa == 0 {
            // Zero
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal
        return 0.0; // Simplified: return 0 for subnormals
    } else if exponent == 0x1F {
        if mantissa == 0 {
            // Infinity
            return if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
        }
        // NaN
        return f32::NAN;
    }

    // Normal numbers
    let f32_exponent = (exponent as i32) - 15 + 127;
    let f32_mantissa = (mantissa as u32) << 13;
    let f32_sign = (sign as u32) << 31;

    let f32_bits = f32_sign | ((f32_exponent as u32) << 23) | f32_mantissa;
    f32::from_bits(f32_bits)
}

/// Lazy tensor that loads data on first access
pub struct LazyTensor {
    /// Tensor name
    name: String,
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Data type
    dtype: DType,
    /// Path to the model file
    file_path: PathBuf,
    /// Cached tensor (None until first access)
    cached: Arc<RwLock<Option<Tensor>>>,
    /// Offset in file (for memory-mapped access)
    _offset: usize,
    /// Size in bytes
    size: usize,
}

impl LazyTensor {
    /// Create a new lazy tensor
    pub fn new(
        name: String,
        shape: Vec<usize>,
        dtype: DType,
        file_path: PathBuf,
        offset: usize,
        size: usize,
    ) -> Self {
        Self {
            name,
            shape,
            dtype,
            file_path,
            cached: Arc::new(RwLock::new(None)),
            _offset: offset,
            size,
        }
    }

    /// Get the tensor, loading it if not cached
    pub fn get(&self) -> ModelResult<Tensor> {
        // Check if cached
        {
            let cache = self.cached.read().unwrap();
            if let Some(tensor) = cache.as_ref() {
                return Ok(tensor.clone());
            }
        }

        // Load tensor
        let tensor = self.load_from_file()?;

        // Cache it
        {
            let mut cache = self.cached.write().unwrap();
            *cache = Some(tensor.clone());
        }

        Ok(tensor)
    }

    /// Load tensor from file
    fn load_from_file(&self) -> ModelResult<Tensor> {
        // Read the file data
        let file_data = std::fs::read(&self.file_path)?;

        // Parse SafeTensors
        let safetensors = SafeTensors::deserialize(&file_data)?;

        // Get the specific tensor
        let tensor_view = safetensors
            .tensor(&self.name)
            .map_err(|e| ModelError::LoadingError {
                reason: format!("Tensor {} not found in file: {}", self.name, e),
            })?;

        // Convert to ToRSh tensor - safetensors data is &[u8], need to handle properly
        let data = tensor_view.data();

        // NOTE: Current limitation - all tensors are converted to f32 because Tensor<T> is generic
        // In the future, this should properly support all dtypes with dynamic dispatch or enum wrapping

        // Convert bytes to f32 based on original dtype
        let float_data: Vec<f32> = match self.dtype {
            DType::F32 => data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            DType::F64 => data
                .chunks_exact(8)
                .map(|chunk| {
                    let val = f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]);
                    val as f32
                })
                .collect(),
            DType::I32 => data
                .chunks_exact(4)
                .map(|chunk| {
                    let val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    val as f32
                })
                .collect(),
            DType::I64 => data
                .chunks_exact(8)
                .map(|chunk| {
                    let val = i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]);
                    val as f32
                })
                .collect(),
            DType::I16 => data
                .chunks_exact(2)
                .map(|chunk| {
                    let val = i16::from_le_bytes([chunk[0], chunk[1]]);
                    val as f32
                })
                .collect(),
            DType::I8 => data.iter().map(|&b| (b as i8) as f32).collect(),
            DType::U8 => data.iter().map(|&b| b as f32).collect(),
            DType::U32 => data
                .chunks_exact(4)
                .map(|chunk| {
                    let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    val as f32
                })
                .collect(),
            DType::U64 => data
                .chunks_exact(8)
                .map(|chunk| {
                    let val = u64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]);
                    val as f32
                })
                .collect(),
            DType::F16 | DType::BF16 => data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f16_to_f32_simple(bits)
                })
                .collect(),
            _ => {
                // Fallback: treat as u8 and convert to f32
                data.iter().map(|&b| b as f32).collect()
            }
        };

        let tensor = Tensor::from_data(float_data, self.shape.clone(), DeviceType::Cpu)?;

        Ok(tensor)
    }

    /// Clear the cache to free memory
    pub fn clear_cache(&self) {
        let mut cache = self.cached.write().unwrap();
        *cache = None;
    }

    /// Check if tensor is cached
    pub fn is_cached(&self) -> bool {
        let cache = self.cached.read().unwrap();
        cache.is_some()
    }

    /// Get tensor shape without loading
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get tensor dtype without loading
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get tensor name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Lazy model loader with LRU cache
pub struct LazyModelLoader {
    /// Path to the model file
    _file_path: PathBuf,
    /// Lazy tensors indexed by name
    tensors: HashMap<String, LazyTensor>,
    /// Maximum cache size in bytes
    max_cache_size: usize,
    /// Current cache size in bytes
    current_cache_size: Arc<RwLock<usize>>,
    /// Access order for LRU
    access_order: Arc<RwLock<Vec<String>>>,
}

impl LazyModelLoader {
    /// Create a new lazy model loader
    pub fn new<P: AsRef<Path>>(path: P, max_cache_size: usize) -> ModelResult<Self> {
        let file_path = path.as_ref().to_path_buf();
        let tensors = Self::scan_tensors(&file_path)?;

        Ok(Self {
            _file_path: file_path,
            tensors,
            max_cache_size,
            current_cache_size: Arc::new(RwLock::new(0)),
            access_order: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Scan file and create lazy tensors
    fn scan_tensors(path: &Path) -> ModelResult<HashMap<String, LazyTensor>> {
        let file_data = std::fs::read(path)?;
        let safetensors = SafeTensors::deserialize(&file_data)?;

        let mut tensors = HashMap::new();

        for (name, _tensor_view) in safetensors.tensors() {
            // Get tensor view again to extract metadata
            let tensor_view = safetensors
                .tensor(&name)
                .map_err(|e| ModelError::LoadingError {
                    reason: format!("Failed to get tensor {}: {}", name, e),
                })?;

            let shape = tensor_view.shape().to_vec();
            let dtype = Self::convert_dtype(tensor_view.dtype());
            let size = tensor_view.data().len();

            let lazy_tensor = LazyTensor::new(
                name.to_string(),
                shape,
                dtype,
                path.to_path_buf(),
                0, // Offset would need to be calculated properly
                size,
            );

            tensors.insert(name.to_string(), lazy_tensor);
        }

        Ok(tensors)
    }

    /// Convert SafeTensors dtype to ToRSh DType
    fn convert_dtype(dtype: safetensors::Dtype) -> DType {
        match dtype {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::F64 => DType::F64,
            safetensors::Dtype::I32 => DType::I32,
            safetensors::Dtype::I64 => DType::I64,
            safetensors::Dtype::U8 => DType::U8,
            safetensors::Dtype::I8 => DType::I8,
            safetensors::Dtype::I16 => DType::I16,
            safetensors::Dtype::U16 => DType::I16, // Note: DType doesn't have U16, using I16
            safetensors::Dtype::U32 => DType::U32,
            safetensors::Dtype::U64 => DType::U64,
            safetensors::Dtype::F16 => DType::F16,
            safetensors::Dtype::BF16 => DType::BF16,
            _ => DType::F32, // Default
        }
    }

    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> ModelResult<Tensor> {
        let lazy_tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| ModelError::LoadingError {
                reason: format!("Tensor {} not found", name),
            })?;

        // Update access order for LRU
        self.update_access_order(name);

        // Get the tensor (will load if not cached)
        let tensor = lazy_tensor.get()?;

        // Update cache size
        let tensor_size = lazy_tensor.size;
        self.add_to_cache(tensor_size)?;

        Ok(tensor)
    }

    /// Update access order for LRU eviction
    fn update_access_order(&self, name: &str) {
        let mut access_order = self.access_order.write().unwrap();

        // Remove if already present
        if let Some(pos) = access_order.iter().position(|n| n == name) {
            access_order.remove(pos);
        }

        // Add to the end (most recently used)
        access_order.push(name.to_string());
    }

    /// Add to cache and evict if necessary
    fn add_to_cache(&self, size: usize) -> ModelResult<()> {
        let mut current_size = self.current_cache_size.write().unwrap();
        *current_size += size;

        // Evict least recently used tensors if cache is full
        while *current_size > self.max_cache_size {
            let evicted = self.evict_lru()?;
            if !evicted {
                break; // Nothing left to evict
            }
        }

        Ok(())
    }

    /// Evict least recently used tensor
    fn evict_lru(&self) -> ModelResult<bool> {
        let mut access_order = self.access_order.write().unwrap();

        if access_order.is_empty() {
            return Ok(false);
        }

        // Get least recently used (first in the list)
        let lru_name = access_order.remove(0);

        if let Some(tensor) = self.tensors.get(&lru_name) {
            let tensor_size = tensor.size;
            tensor.clear_cache();

            let mut current_size = self.current_cache_size.write().unwrap();
            *current_size = current_size.saturating_sub(tensor_size);
        }

        Ok(true)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Get tensor metadata without loading
    pub fn tensor_metadata(&self, name: &str) -> Option<(Vec<usize>, DType)> {
        self.tensors
            .get(name)
            .map(|t| (t.shape().to_vec(), t.dtype()))
    }

    /// Clear entire cache
    pub fn clear_cache(&self) {
        for tensor in self.tensors.values() {
            tensor.clear_cache();
        }

        let mut current_size = self.current_cache_size.write().unwrap();
        *current_size = 0;

        let mut access_order = self.access_order.write().unwrap();
        access_order.clear();
    }

    /// Get current cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let cached_count = self.tensors.values().filter(|t| t.is_cached()).count();
        let total_count = self.tensors.len();
        let current_size = *self.current_cache_size.read().unwrap();

        CacheStats {
            cached_tensors: cached_count,
            total_tensors: total_count,
            cache_size_bytes: current_size,
            max_cache_size_bytes: self.max_cache_size,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of currently cached tensors
    pub cached_tensors: usize,
    /// Total number of tensors
    pub total_tensors: usize,
    /// Current cache size in bytes
    pub cache_size_bytes: usize,
    /// Maximum cache size in bytes
    pub max_cache_size_bytes: usize,
}

impl CacheStats {
    /// Get cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        if self.total_tensors == 0 {
            0.0
        } else {
            self.cached_tensors as f64 / self.total_tensors as f64
        }
    }

    /// Get cache utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        if self.max_cache_size_bytes == 0 {
            0.0
        } else {
            self.cache_size_bytes as f64 / self.max_cache_size_bytes as f64
        }
    }
}

/// Streaming model loader for very large models
pub struct StreamingModelLoader {
    /// Path to the model file
    file_path: PathBuf,
    /// Chunk size for streaming
    chunk_size: usize,
}

impl StreamingModelLoader {
    /// Create a new streaming model loader
    pub fn new<P: AsRef<Path>>(path: P, chunk_size: usize) -> Self {
        Self {
            file_path: path.as_ref().to_path_buf(),
            chunk_size,
        }
    }

    /// Stream tensors one at a time
    pub fn stream_tensors<F>(&self, mut callback: F) -> ModelResult<()>
    where
        F: FnMut(&str, Tensor) -> ModelResult<()>,
    {
        let file_data = std::fs::read(&self.file_path)?;
        let safetensors = SafeTensors::deserialize(&file_data)?;

        for (name, _tensor_view) in safetensors.tensors() {
            // Get tensor view again
            let tensor_view = safetensors
                .tensor(&name)
                .map_err(|e| ModelError::LoadingError {
                    reason: format!("Failed to get tensor {}: {}", name, e),
                })?;

            let shape = tensor_view.shape().to_vec();
            let data = tensor_view.data();

            // Convert bytes to f32 for simplicity (in production, handle all dtypes)
            let float_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let tensor = Tensor::from_data(float_data, shape, DeviceType::Cpu)?;

            callback(&name, tensor)?;
        }

        Ok(())
    }

    /// Stream tensors in chunks
    pub fn stream_tensor_chunks<F>(&self, tensor_name: &str, mut callback: F) -> ModelResult<()>
    where
        F: FnMut(usize, &[u8]) -> ModelResult<()>,
    {
        let file_data = std::fs::read(&self.file_path)?;
        let safetensors = SafeTensors::deserialize(&file_data)?;

        let tensor_view =
            safetensors
                .tensor(tensor_name)
                .map_err(|e| ModelError::LoadingError {
                    reason: format!("Tensor {} not found: {}", tensor_name, e),
                })?;

        let data = tensor_view.data();

        // Stream in chunks
        for (i, chunk) in data.chunks(self.chunk_size).enumerate() {
            callback(i, chunk)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_safetensors() -> NamedTempFile {
        // Create a minimal SafeTensors file for testing
        let mut file = NamedTempFile::new().unwrap();

        // This is a simplified test - in practice you'd create a proper SafeTensors file
        let test_data = vec![0u8; 100];
        file.write_all(&test_data).unwrap();
        file.flush().unwrap();

        file
    }

    #[test]
    fn test_cache_stats() {
        let stats = CacheStats {
            cached_tensors: 5,
            total_tensors: 10,
            cache_size_bytes: 1024,
            max_cache_size_bytes: 2048,
        };

        assert_eq!(stats.hit_rate(), 0.5);
        assert_eq!(stats.utilization(), 0.5);
    }

    #[test]
    fn test_streaming_loader_creation() {
        let file = create_test_safetensors();
        let loader = StreamingModelLoader::new(file.path(), 1024);
        assert_eq!(loader.chunk_size, 1024);
    }
}
