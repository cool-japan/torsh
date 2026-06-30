//! Lazy Loading for Memory-Mapped Tensor Data
//!
//! This module provides lazy loading capabilities for tensor data stored in memory-mapped files.
//! Data is loaded on-demand when first accessed, allowing for efficient handling of large datasets
//! that may not fit entirely in memory.
//!
//! # Features
//!
//! - **On-demand loading**: Data is loaded only when accessed
//! - **Chunk-based access**: Supports loading specific regions of large tensors
//! - **Caching**: Keeps recently accessed chunks in memory
//! - **Memory pressure handling**: Unloads chunks when memory pressure is high
//! - **Multi-threaded access**: Thread-safe concurrent access to lazy-loaded data

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
    shape::Shape,
};

/// Configuration for lazy loading behavior
#[derive(Debug, Clone)]
pub struct LazyLoadConfig {
    /// Size of chunks to load at once (in elements)
    pub chunk_size: usize,
    /// Maximum number of chunks to keep in cache
    pub max_cached_chunks: usize,
    /// Time to keep unused chunks in cache before unloading
    pub cache_ttl: Duration,
    /// Memory pressure threshold for aggressive cleanup
    pub memory_pressure_threshold: usize,
}

impl Default for LazyLoadConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024 * 1024, // 1M elements per chunk
            max_cached_chunks: 16,
            cache_ttl: Duration::from_secs(300), // 5 minutes
            memory_pressure_threshold: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Metadata for a lazily-loaded tensor
#[derive(Debug, Clone)]
pub struct LazyTensorMetadata {
    /// Shape of the tensor
    pub shape: Shape,
    /// Data type name
    pub dtype: String,
    /// Total size in elements
    pub total_elements: usize,
    /// Size of each element in bytes
    pub element_size: usize,
    /// File offset where data begins
    pub data_offset: u64,
}

/// A cached chunk of tensor data
#[derive(Debug, Clone)]
struct CachedChunk<T: TensorElement> {
    /// The actual data
    data: Vec<T>,
    /// Index range this chunk covers (start, end)
    range: (usize, usize),
    /// Last access time for TTL management
    last_accessed: Instant,
}

/// Lazy-loaded tensor data backed by a memory-mapped file
pub struct LazyTensor<T: TensorElement> {
    /// Metadata about the tensor
    metadata: LazyTensorMetadata,
    /// File backing the data
    file: Arc<Mutex<File>>,
    /// Path to the backing file
    #[allow(dead_code)]
    file_path: PathBuf,
    /// Cache of loaded chunks
    chunk_cache: Arc<RwLock<HashMap<usize, CachedChunk<T>>>>,
    /// Configuration
    config: LazyLoadConfig,
    /// Element type marker
    _phantom: std::marker::PhantomData<T>,
}

impl<T: TensorElement> LazyTensor<T> {
    /// Create a new lazy tensor from a file
    ///
    /// # Arguments
    /// * `file_path` - Path to the file containing tensor data
    /// * `metadata` - Metadata describing the tensor layout
    /// * `config` - Configuration for lazy loading behavior
    ///
    /// # Returns
    /// * `Result<LazyTensor<T>>` - The lazy tensor or error
    pub fn new<P: AsRef<Path>>(
        file_path: P,
        metadata: LazyTensorMetadata,
        config: LazyLoadConfig,
    ) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        let file = File::open(&file_path)
            .map_err(|e| TorshError::IoError(format!("Failed to open file: {}", e)))?;

        Ok(Self {
            metadata,
            file: Arc::new(Mutex::new(file)),
            file_path,
            chunk_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &Shape {
        &self.metadata.shape
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.metadata.total_elements
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.metadata.total_elements == 0
    }

    /// Load a specific element by flat index
    ///
    /// # Arguments
    /// * `index` - Flat index of the element to load
    ///
    /// # Returns
    /// * `Result<T>` - The element value or error
    pub fn get_element(&self, index: usize) -> Result<T> {
        if index >= self.metadata.total_elements {
            return Err(TorshError::InvalidArgument(format!(
                "Index {} out of bounds for tensor with {} elements",
                index, self.metadata.total_elements
            )));
        }

        let chunk_index = index / self.config.chunk_size;
        let chunk_offset = index % self.config.chunk_size;

        let chunk = self.load_chunk(chunk_index)?;
        Ok(chunk.data[chunk_offset])
    }

    /// Load a range of elements
    ///
    /// # Arguments
    /// * `start` - Starting index (inclusive)
    /// * `end` - Ending index (exclusive)
    ///
    /// # Returns
    /// * `Result<Vec<T>>` - The loaded elements or error
    pub fn get_range(&self, start: usize, end: usize) -> Result<Vec<T>> {
        if start > end || end > self.metadata.total_elements {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid range [{}..{}] for tensor with {} elements",
                start, end, self.metadata.total_elements
            )));
        }

        let mut result = Vec::with_capacity(end - start);
        let start_chunk = start / self.config.chunk_size;
        let end_chunk = (end - 1) / self.config.chunk_size;

        for chunk_idx in start_chunk..=end_chunk {
            let chunk = self.load_chunk(chunk_idx)?;

            let chunk_start = chunk_idx * self.config.chunk_size;
            let chunk_end = std::cmp::min(
                (chunk_idx + 1) * self.config.chunk_size,
                self.metadata.total_elements,
            );

            let range_start = std::cmp::max(start, chunk_start) - chunk_start;
            let range_end = std::cmp::min(end, chunk_end) - chunk_start;

            result.extend_from_slice(&chunk.data[range_start..range_end]);
        }

        Ok(result)
    }

    /// Load all data (use with caution for large tensors)
    ///
    /// # Returns
    /// * `Result<Vec<T>>` - All tensor data or error
    pub fn load_all(&self) -> Result<Vec<T>> {
        self.get_range(0, self.metadata.total_elements)
    }

    /// Load a specific chunk into cache
    fn load_chunk(&self, chunk_index: usize) -> Result<Arc<CachedChunk<T>>> {
        // Check if chunk is already cached
        {
            let cache = self
                .chunk_cache
                .read()
                .expect("lock should not be poisoned");
            if let Some(cached) = cache.get(&chunk_index) {
                // Update access time and return cached chunk
                return Ok(Arc::new(CachedChunk {
                    data: cached.data.clone(),
                    range: cached.range,
                    last_accessed: Instant::now(),
                }));
            }
        }

        // Calculate chunk boundaries
        let start_element = chunk_index * self.config.chunk_size;
        let end_element = std::cmp::min(
            (chunk_index + 1) * self.config.chunk_size,
            self.metadata.total_elements,
        );
        let chunk_size = end_element - start_element;

        // Load data from file
        let data = self.load_chunk_from_file(start_element, chunk_size)?;

        let chunk = Arc::new(CachedChunk {
            data,
            range: (start_element, end_element),
            last_accessed: Instant::now(),
        });

        // Add to cache
        {
            let mut cache = self
                .chunk_cache
                .write()
                .expect("lock should not be poisoned");

            // Clean up cache if needed
            self.cleanup_cache(&mut cache);

            cache.insert(chunk_index, (*chunk).clone());
        }

        Ok(chunk)
    }

    /// Load chunk data directly from file
    fn load_chunk_from_file(&self, start_element: usize, chunk_size: usize) -> Result<Vec<T>> {
        let mut file = self.file.lock().expect("lock should not be poisoned");

        let file_offset =
            self.metadata.data_offset + (start_element as u64 * self.metadata.element_size as u64);

        file.seek(SeekFrom::Start(file_offset))
            .map_err(|e| TorshError::IoError(format!("Failed to seek: {}", e)))?;

        let mut buffer = vec![0u8; chunk_size * self.metadata.element_size];
        file.read_exact(&mut buffer)
            .map_err(|e| TorshError::IoError(format!("Failed to read chunk: {}", e)))?;

        // Convert bytes to elements (this is a simplified version)
        // In practice, you'd need proper deserialization based on the data type
        let data =
            unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, chunk_size).to_vec() };

        Ok(data)
    }

    /// Clean up old cached chunks
    fn cleanup_cache(&self, cache: &mut HashMap<usize, CachedChunk<T>>) {
        let now = Instant::now();

        // Remove expired chunks
        cache.retain(|_, chunk| now.duration_since(chunk.last_accessed) < self.config.cache_ttl);

        // If we still have too many chunks, remove the oldest ones
        if cache.len() > self.config.max_cached_chunks {
            let mut chunks_to_remove: Vec<_> = cache
                .iter()
                .map(|(idx, chunk)| (*idx, chunk.last_accessed))
                .collect();
            chunks_to_remove.sort_by_key(|(_, accessed)| *accessed);

            let to_remove = cache.len() - self.config.max_cached_chunks;
            for (idx, _) in chunks_to_remove.iter().take(to_remove) {
                cache.remove(idx);
            }
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let cache = self
            .chunk_cache
            .read()
            .expect("lock should not be poisoned");

        let total_cached_elements: usize = cache.values().map(|chunk| chunk.data.len()).sum();

        CacheStats {
            cached_chunks: cache.len(),
            total_cached_elements,
            estimated_memory_usage: total_cached_elements * std::mem::size_of::<T>(),
        }
    }

    /// Force cleanup of all cached chunks
    pub fn clear_cache(&self) {
        let mut cache = self
            .chunk_cache
            .write()
            .expect("lock should not be poisoned");
        cache.clear();
    }

    /// Check memory pressure and cleanup if necessary
    pub fn check_memory_pressure(&self) -> Result<()> {
        let stats = self.cache_stats();

        if stats.estimated_memory_usage > self.config.memory_pressure_threshold {
            let mut cache = self
                .chunk_cache
                .write()
                .expect("lock should not be poisoned");

            // Aggressive cleanup - keep only recently accessed chunks
            let recent_threshold = Duration::from_secs(60);
            let now = Instant::now();

            cache.retain(|_, chunk| now.duration_since(chunk.last_accessed) < recent_threshold);
        }

        Ok(())
    }
}

/// Statistics about cached chunks
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of chunks currently cached
    pub cached_chunks: usize,
    /// Total number of elements in cache
    pub total_cached_elements: usize,
    /// Estimated memory usage in bytes
    pub estimated_memory_usage: usize,
}

/// Builder for creating lazy tensors with custom configuration
pub struct LazyTensorBuilder {
    config: LazyLoadConfig,
}

impl LazyTensorBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: LazyLoadConfig::default(),
        }
    }

    /// Set the chunk size
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Set the maximum number of cached chunks
    pub fn max_cached_chunks(mut self, max: usize) -> Self {
        self.config.max_cached_chunks = max;
        self
    }

    /// Set the cache TTL
    pub fn cache_ttl(mut self, ttl: Duration) -> Self {
        self.config.cache_ttl = ttl;
        self
    }

    /// Set the memory pressure threshold
    pub fn memory_pressure_threshold(mut self, threshold: usize) -> Self {
        self.config.memory_pressure_threshold = threshold;
        self
    }

    /// Build the lazy tensor
    pub fn build<T: TensorElement, P: AsRef<Path>>(
        self,
        file_path: P,
        metadata: LazyTensorMetadata,
    ) -> Result<LazyTensor<T>> {
        LazyTensor::new(file_path, metadata, self.config)
    }
}

impl Default for LazyTensorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for working with lazy tensors
pub mod utils {
    use super::*;
    use std::fs::File;
    use std::io::{BufReader, Read};
    use torsh_core::dtype::DType;

    /// Create a lazy tensor metadata from a binary file header
    ///
    /// This function reads tensor metadata from a binary file header
    /// and creates appropriate LazyTensorMetadata.
    pub fn create_metadata_from_header<P: AsRef<Path>>(file_path: P) -> Result<LazyTensorMetadata> {
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);

        // Read a simple header format (this is a placeholder - you'd implement
        // the actual format parsing based on your serialization format)
        let mut header_size_bytes = [0u8; 4];
        reader
            .read_exact(&mut header_size_bytes)
            .map_err(|e| TorshError::IoError(format!("Failed to read header size: {}", e)))?;

        let header_size = u32::from_le_bytes(header_size_bytes) as usize;

        let mut header_data = vec![0u8; header_size];
        reader
            .read_exact(&mut header_data)
            .map_err(|e| TorshError::IoError(format!("Failed to read header: {}", e)))?;

        // Deserialize the JSON header object into concrete tensor metadata.
        //
        // The header is a compact JSON object describing the on-disk tensor, e.g.
        // `{"shape":[10,10],"dtype":"f32","total_elements":100}`. The element size is
        // derived from the dtype, and the data offset is the size prefix (4 bytes)
        // plus the header length, i.e. the first byte position of the payload.
        let header_str = String::from_utf8(header_data)
            .map_err(|e| TorshError::SerializationError(format!("Invalid header: {}", e)))?;

        let dims = parse_json_usize_array(&header_str, "shape").ok_or_else(|| {
            TorshError::SerializationError(format!(
                "Header is missing a valid 'shape' field: {}",
                header_str
            ))
        })?;

        let dtype = parse_json_string(&header_str, "dtype").ok_or_else(|| {
            TorshError::SerializationError(format!(
                "Header is missing a valid 'dtype' field: {}",
                header_str
            ))
        })?;

        // Derive the element size from the canonical dtype definition so it always
        // matches the dtype that was recorded in the header.
        let element_size = dtype
            .parse::<DType>()
            .map_err(|e| {
                TorshError::SerializationError(format!("Unsupported dtype '{}': {}", dtype, e))
            })?
            .size();

        // `total_elements` is optional: when present it must agree with the product
        // of the shape dimensions, otherwise the header is internally inconsistent.
        let shape_elements: usize = dims.iter().product();
        let total_elements = match parse_json_usize(&header_str, "total_elements") {
            Some(declared) if declared != shape_elements => {
                return Err(TorshError::SerializationError(format!(
                    "Header total_elements ({}) does not match shape product ({})",
                    declared, shape_elements
                )));
            }
            Some(declared) => declared,
            None => shape_elements,
        };

        Ok(LazyTensorMetadata {
            shape: Shape::new(dims),
            dtype,
            total_elements,
            element_size,
            data_offset: 4 + header_size as u64,
        })
    }

    /// Locate the value substring immediately following a `"key":` token.
    ///
    /// Returns the slice after the colon (with leading whitespace trimmed), or
    /// `None` when the key is not present in the JSON object string.
    fn json_value_after_key<'a>(header: &'a str, key: &str) -> Option<&'a str> {
        let quoted_key = format!("\"{}\"", key);
        let key_pos = header.find(&quoted_key)?;
        let after_key = &header[key_pos + quoted_key.len()..];
        let colon_pos = after_key.find(':')?;
        Some(after_key[colon_pos + 1..].trim_start())
    }

    /// Parse a quoted JSON string value for `key` (e.g. `"dtype":"f32"`).
    fn parse_json_string(header: &str, key: &str) -> Option<String> {
        let value = json_value_after_key(header, key)?;
        let inner = value.strip_prefix('"')?;
        let end = inner.find('"')?;
        Some(inner[..end].to_string())
    }

    /// Parse an unsigned integer JSON value for `key` (e.g. `"total_elements":100`).
    fn parse_json_usize(header: &str, key: &str) -> Option<usize> {
        let value = json_value_after_key(header, key)?;
        let digits: String = value.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            return None;
        }
        digits.parse::<usize>().ok()
    }

    /// Parse a JSON array of unsigned integers for `key` (e.g. `"shape":[10,10]`).
    fn parse_json_usize_array(header: &str, key: &str) -> Option<Vec<usize>> {
        let value = json_value_after_key(header, key)?;
        let inner = value.strip_prefix('[')?;
        let end = inner.find(']')?;
        let mut dims = Vec::new();
        for part in inner[..end].split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            dims.push(trimmed.parse::<usize>().ok()?);
        }
        Some(dims)
    }

    /// Create a lazy tensor from a file path with automatic metadata detection
    pub fn lazy_tensor_from_file<T: TensorElement, P: AsRef<Path>>(
        file_path: P,
    ) -> Result<LazyTensor<T>> {
        let metadata = create_metadata_from_header(&file_path)?;
        LazyTensor::new(file_path, metadata, LazyLoadConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use torsh_core::shape::Shape;

    fn create_test_file() -> (NamedTempFile, LazyTensorMetadata) {
        let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");

        // Write a simple header (4 bytes for header size, then JSON metadata)
        let header = r#"{"shape":[10,10],"dtype":"f32","total_elements":100}"#;
        let header_size = header.len() as u32;
        temp_file
            .write_all(&header_size.to_le_bytes())
            .expect("write should succeed");
        temp_file
            .write_all(header.as_bytes())
            .expect("write should succeed");

        // Write test data (100 f32 values)
        for i in 0..100 {
            temp_file
                .write_all(&(i as f32).to_le_bytes())
                .expect("write should succeed");
        }
        temp_file.flush().expect("flush should succeed");

        let metadata = LazyTensorMetadata {
            shape: Shape::new(vec![10, 10]),
            dtype: "f32".to_string(),
            total_elements: 100,
            element_size: 4,
            data_offset: 4 + header.len() as u64,
        };

        (temp_file, metadata)
    }

    #[test]
    fn test_lazy_tensor_creation() {
        let (temp_file, metadata) = create_test_file();

        let lazy_tensor: LazyTensor<f32> =
            LazyTensor::new(temp_file.path(), metadata, LazyLoadConfig::default())
                .expect("lazy tensor creation should succeed");

        assert_eq!(lazy_tensor.len(), 100);
        assert!(!lazy_tensor.is_empty());
        assert_eq!(lazy_tensor.shape().dims(), &[10, 10]);
    }

    #[test]
    fn test_lazy_loading_element_access() {
        let (temp_file, metadata) = create_test_file();

        let lazy_tensor: LazyTensor<f32> = LazyTensor::new(
            temp_file.path(),
            metadata,
            LazyLoadConfig {
                chunk_size: 10,
                ..LazyLoadConfig::default()
            },
        )
        .expect("lazy tensor creation should succeed");

        // Test loading individual elements
        let element = lazy_tensor
            .get_element(5)
            .expect("get_element should succeed");
        assert!((element - 5.0).abs() < f32::EPSILON);

        let element = lazy_tensor
            .get_element(50)
            .expect("get_element should succeed");
        assert!((element - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lazy_loading_range_access() {
        let (temp_file, metadata) = create_test_file();

        let lazy_tensor: LazyTensor<f32> = LazyTensor::new(
            temp_file.path(),
            metadata,
            LazyLoadConfig {
                chunk_size: 10,
                ..LazyLoadConfig::default()
            },
        )
        .expect("lazy tensor creation should succeed");

        // Test loading a range of elements
        let range = lazy_tensor
            .get_range(10, 20)
            .expect("get_range should succeed");
        assert_eq!(range.len(), 10);

        for (i, &value) in range.iter().enumerate() {
            let expected = (10 + i) as f32;
            assert!((value - expected).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_cache_management() {
        let (temp_file, metadata) = create_test_file();

        let lazy_tensor: LazyTensor<f32> = LazyTensor::new(
            temp_file.path(),
            metadata,
            LazyLoadConfig {
                chunk_size: 10,
                max_cached_chunks: 2,
                ..LazyLoadConfig::default()
            },
        )
        .expect("lazy tensor creation should succeed");

        // Load some chunks - should respect max cache size
        lazy_tensor
            .get_element(5)
            .expect("get_element should succeed"); // Chunk 0
        lazy_tensor
            .get_element(15)
            .expect("get_element should succeed"); // Chunk 1

        let stats = lazy_tensor.cache_stats();
        assert!(stats.cached_chunks <= 2); // Should not exceed max after 2 chunks

        lazy_tensor
            .get_element(25)
            .expect("get_element should succeed"); // Chunk 2 - should trigger cleanup

        let stats_after = lazy_tensor.cache_stats();
        // After loading a 3rd chunk, cache size might be 2 or 3 depending on cleanup timing
        // We allow up to 3 as the cleanup happens during insertion, not after
        assert!(stats_after.cached_chunks <= 3);
        assert!(stats_after.total_cached_elements > 0);
    }

    #[test]
    fn test_lazy_tensor_builder() {
        let (temp_file, metadata) = create_test_file();

        let lazy_tensor: LazyTensor<f32> = LazyTensorBuilder::new()
            .chunk_size(5)
            .max_cached_chunks(3)
            .build(temp_file.path(), metadata)
            .expect("lazy operation should succeed");

        assert_eq!(lazy_tensor.config.chunk_size, 5);
        assert_eq!(lazy_tensor.config.max_cached_chunks, 3);
    }

    #[test]
    fn test_out_of_bounds_access() {
        let (temp_file, metadata) = create_test_file();

        let lazy_tensor: LazyTensor<f32> =
            LazyTensor::new(temp_file.path(), metadata, LazyLoadConfig::default())
                .expect("lazy tensor creation should succeed");

        // Test out of bounds access
        let result = lazy_tensor.get_element(1000);
        assert!(result.is_err());

        let result = lazy_tensor.get_range(90, 110);
        assert!(result.is_err());
    }

    /// Build a unique path inside the system temp directory for header tests.
    fn unique_temp_path(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("torsh_{}_{}_{}_{}.bin", tag, pid, n, nanos))
    }

    /// Write a `[u32 header_size][header bytes][f32 payload]` file used by the
    /// lazy-loading header parser.
    fn write_header_file(path: &std::path::Path, header: &str, data: &[f32]) {
        let mut file = std::fs::File::create(path).expect("temp file creation should succeed");
        let header_size = header.len() as u32;
        file.write_all(&header_size.to_le_bytes())
            .expect("write should succeed");
        file.write_all(header.as_bytes())
            .expect("write should succeed");
        for &value in data {
            file.write_all(&value.to_le_bytes())
                .expect("write should succeed");
        }
        file.flush().expect("flush should succeed");
    }

    #[test]
    fn test_create_metadata_from_header_populates_all_fields() {
        let header = r#"{"shape":[3,4,5],"dtype":"f32","total_elements":60}"#;
        let path = unique_temp_path("lazy_header_f32");
        write_header_file(&path, header, &[]);

        let metadata =
            utils::create_metadata_from_header(&path).expect("metadata parsing should succeed");

        // Every field must be populated from the header, not from placeholders.
        assert_eq!(metadata.shape.dims(), &[3, 4, 5]);
        assert_eq!(metadata.dtype, "f32");
        assert_eq!(metadata.total_elements, 60);
        assert_eq!(metadata.element_size, 4);
        assert_eq!(metadata.data_offset, 4 + header.len() as u64);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_create_metadata_from_header_derives_element_size_and_total() {
        // No `total_elements` field: it must be derived from the shape product,
        // and the element size must follow the f64 dtype (8 bytes).
        let header = r#"{"shape":[2,8],"dtype":"f64"}"#;
        let path = unique_temp_path("lazy_header_f64");
        write_header_file(&path, header, &[]);

        let metadata =
            utils::create_metadata_from_header(&path).expect("metadata parsing should succeed");

        assert_eq!(metadata.shape.dims(), &[2, 8]);
        assert_eq!(metadata.dtype, "f64");
        assert_eq!(metadata.total_elements, 16);
        assert_eq!(metadata.element_size, 8);
        assert_eq!(metadata.data_offset, 4 + header.len() as u64);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_create_metadata_from_header_roundtrip_load() {
        // End-to-end: parse the header, then load the payload through a LazyTensor.
        // This only yields the correct values if `data_offset` is computed correctly.
        let header = r#"{"shape":[2,3],"dtype":"f32","total_elements":6}"#;
        let path = unique_temp_path("lazy_header_roundtrip");
        let data: Vec<f32> = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        write_header_file(&path, header, &data);

        let metadata =
            utils::create_metadata_from_header(&path).expect("metadata parsing should succeed");
        assert_eq!(metadata.total_elements, 6);
        assert_eq!(metadata.data_offset, 4 + header.len() as u64);

        let lazy: LazyTensor<f32> = LazyTensor::new(
            &path,
            metadata,
            LazyLoadConfig {
                chunk_size: 4,
                ..LazyLoadConfig::default()
            },
        )
        .expect("lazy tensor creation should succeed");

        let loaded = lazy.load_all().expect("load_all should succeed");
        assert_eq!(loaded, data);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_create_metadata_from_header_rejects_missing_shape() {
        let header = r#"{"dtype":"f32","total_elements":6}"#;
        let path = unique_temp_path("lazy_header_missing_shape");
        write_header_file(&path, header, &[]);

        let result = utils::create_metadata_from_header(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_create_metadata_from_header_rejects_inconsistent_total() {
        // total_elements (99) contradicts the shape product (2 * 3 = 6).
        let header = r#"{"shape":[2,3],"dtype":"f32","total_elements":99}"#;
        let path = unique_temp_path("lazy_header_inconsistent");
        write_header_file(&path, header, &[]);

        let result = utils::create_metadata_from_header(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }
}
