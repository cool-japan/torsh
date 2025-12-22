//! Lazy loading resource system for packages
//!
//! This module provides lazy loading capabilities for package resources,
//! allowing packages to load only the resources that are actually needed.
//! Features include:
//! - Lazy loading from files and archives
//! - Memory mapping for large files
//! - Streaming support for processing large resources
//! - Cache management with eviction strategies

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use torsh_core::error::{Result, TorshError};

use crate::resources::{Resource, ResourceType};

/// Lazy resource that loads data on-demand
#[derive(Debug, Clone)]
pub struct LazyResource {
    /// Resource name (unique identifier)
    pub name: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Resource storage strategy
    storage: LazyStorage,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Storage strategy for lazy resources
#[derive(Debug, Clone)]
enum LazyStorage {
    /// Data is already loaded in memory
    InMemory(Vec<u8>),
    /// Data is stored on disk and loaded on-demand
    LazyFile {
        /// Path to the file containing the data
        file_path: PathBuf,
        /// File offset where the data starts
        offset: u64,
        /// Size of the data in bytes
        size: u64,
        /// Cached data (loaded when first accessed)
        cached_data: Arc<RwLock<Option<Vec<u8>>>>,
    },
    /// Data is stored in a compressed archive
    LazyArchive {
        /// Path to the archive file
        archive_path: PathBuf,
        /// Entry name within the archive
        entry_name: String,
        /// Cached data (loaded when first accessed)
        cached_data: Arc<RwLock<Option<Vec<u8>>>>,
    },
}

/// Resource manager with lazy loading capabilities
#[derive(Debug)]
pub struct LazyResourceManager {
    /// Collection of lazy resources
    resources: HashMap<String, LazyResource>,
    /// Memory usage limit in bytes
    memory_limit: Option<usize>,
    /// Current memory usage
    current_memory_usage: Arc<RwLock<usize>>,
    /// Eviction strategy when memory limit is reached
    eviction_strategy: EvictionStrategy,
}

/// Eviction strategy for managing memory usage
#[derive(Debug, Clone, Copy)]
pub enum EvictionStrategy {
    /// Least recently used
    LRU,
    /// Largest resources first
    LargestFirst,
    /// Random eviction
    Random,
}

impl LazyResource {
    /// Create a new lazy resource with in-memory data
    pub fn new_in_memory(name: String, resource_type: ResourceType, data: Vec<u8>) -> Self {
        Self {
            name,
            resource_type,
            storage: LazyStorage::InMemory(data),
            metadata: HashMap::new(),
        }
    }

    /// Create a new lazy resource with file-based loading
    pub fn new_lazy_file<P: Into<PathBuf>>(
        name: String,
        resource_type: ResourceType,
        file_path: P,
        offset: u64,
        size: u64,
    ) -> Self {
        Self {
            name,
            resource_type,
            storage: LazyStorage::LazyFile {
                file_path: file_path.into(),
                offset,
                size,
                cached_data: Arc::new(RwLock::new(None)),
            },
            metadata: HashMap::new(),
        }
    }

    /// Create a new lazy resource with archive-based loading
    pub fn new_lazy_archive<P: Into<PathBuf>>(
        name: String,
        resource_type: ResourceType,
        archive_path: P,
        entry_name: String,
    ) -> Self {
        Self {
            name,
            resource_type,
            storage: LazyStorage::LazyArchive {
                archive_path: archive_path.into(),
                entry_name,
                cached_data: Arc::new(RwLock::new(None)),
            },
            metadata: HashMap::new(),
        }
    }

    /// Create from existing Resource
    pub fn from_resource(resource: Resource) -> Self {
        Self {
            name: resource.name,
            resource_type: resource.resource_type,
            storage: LazyStorage::InMemory(resource.data),
            metadata: resource.metadata,
        }
    }

    /// Convert to regular Resource (loads data if needed)
    pub fn to_resource(&self) -> Result<Resource> {
        let data = self.data()?;
        Ok(Resource {
            name: self.name.clone(),
            resource_type: self.resource_type,
            data,
            metadata: self.metadata.clone(),
        })
    }

    /// Get the resource data, loading it if necessary
    pub fn data(&self) -> Result<Vec<u8>> {
        match &self.storage {
            LazyStorage::InMemory(data) => Ok(data.clone()),
            LazyStorage::LazyFile {
                file_path,
                offset,
                size,
                cached_data,
            } => {
                // Check if data is already cached
                {
                    let cache_read = cached_data.read().map_err(|e| {
                        TorshError::InvalidArgument(format!("Failed to acquire read lock: {}", e))
                    })?;

                    if let Some(ref cached) = *cache_read {
                        return Ok(cached.clone());
                    }
                }

                // Load data from file
                let data = self.load_file_data(file_path, *offset, *size)?;

                // Cache the data
                {
                    let mut cache_write = cached_data.write().map_err(|e| {
                        TorshError::InvalidArgument(format!("Failed to acquire write lock: {}", e))
                    })?;
                    *cache_write = Some(data.clone());
                }

                Ok(data)
            }
            LazyStorage::LazyArchive {
                archive_path,
                entry_name,
                cached_data,
            } => {
                // Check if data is already cached
                {
                    let cache_read = cached_data.read().map_err(|e| {
                        TorshError::InvalidArgument(format!("Failed to acquire read lock: {}", e))
                    })?;

                    if let Some(ref cached) = *cache_read {
                        return Ok(cached.clone());
                    }
                }

                // Load data from archive
                let data = self.load_archive_data(archive_path, entry_name)?;

                // Cache the data
                {
                    let mut cache_write = cached_data.write().map_err(|e| {
                        TorshError::InvalidArgument(format!("Failed to acquire write lock: {}", e))
                    })?;
                    *cache_write = Some(data.clone());
                }

                Ok(data)
            }
        }
    }

    /// Check if the resource is loaded in memory
    pub fn is_loaded(&self) -> bool {
        match &self.storage {
            LazyStorage::InMemory(_) => true,
            LazyStorage::LazyFile { cached_data, .. }
            | LazyStorage::LazyArchive { cached_data, .. } => {
                cached_data.read().map_or(false, |cache| cache.is_some())
            }
        }
    }

    /// Get the size of the resource data
    pub fn size(&self) -> Result<u64> {
        match &self.storage {
            LazyStorage::InMemory(data) => Ok(data.len() as u64),
            LazyStorage::LazyFile { size, .. } => Ok(*size),
            LazyStorage::LazyArchive {
                cached_data,
                archive_path,
                entry_name,
            } => {
                // If cached, return cached size
                {
                    let cache_read = cached_data.read().map_err(|e| {
                        TorshError::InvalidArgument(format!("Failed to acquire read lock: {}", e))
                    })?;

                    if let Some(ref cached) = *cache_read {
                        return Ok(cached.len() as u64);
                    }
                }

                // Otherwise, get size from archive metadata
                self.get_archive_entry_size(archive_path, entry_name)
            }
        }
    }

    /// Preload the resource data into memory
    pub fn preload(&self) -> Result<()> {
        self.data()?;
        Ok(())
    }

    /// Evict cached data to free memory
    pub fn evict_cache(&self) -> Result<()> {
        match &self.storage {
            LazyStorage::InMemory(_) => Ok(()),
            LazyStorage::LazyFile { cached_data, .. }
            | LazyStorage::LazyArchive { cached_data, .. } => {
                let mut cache_write = cached_data.write().map_err(|e| {
                    TorshError::InvalidArgument(format!("Failed to acquire write lock: {}", e))
                })?;
                *cache_write = None;
                Ok(())
            }
        }
    }

    /// Load data from a file at a specific offset
    fn load_file_data(&self, file_path: &Path, offset: u64, size: u64) -> Result<Vec<u8>> {
        let file = fs::File::open(file_path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to open file {:?}: {}", file_path, e))
        })?;

        let mut reader = std::io::BufReader::new(file);

        // Seek to the offset
        std::io::Seek::seek(&mut reader, std::io::SeekFrom::Start(offset)).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to seek to offset {}: {}", offset, e))
        })?;

        // Read the specified amount of data
        let mut buffer = vec![0u8; size as usize];
        reader.read_exact(&mut buffer).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to read {} bytes: {}", size, e))
        })?;

        Ok(buffer)
    }

    /// Load data from an archive entry
    fn load_archive_data(&self, archive_path: &Path, entry_name: &str) -> Result<Vec<u8>> {
        let file = fs::File::open(archive_path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to open archive {:?}: {}", archive_path, e))
        })?;

        let mut archive = zip::ZipArchive::new(file).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to read ZIP archive: {}", e))
        })?;

        let mut entry = archive.by_name(entry_name).map_err(|e| {
            TorshError::InvalidArgument(format!(
                "Failed to find entry '{}' in archive: {}",
                entry_name, e
            ))
        })?;

        let mut buffer = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buffer).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to read archive entry: {}", e))
        })?;

        Ok(buffer)
    }

    /// Get the size of an archive entry
    fn get_archive_entry_size(&self, archive_path: &Path, entry_name: &str) -> Result<u64> {
        let file = fs::File::open(archive_path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to open archive {:?}: {}", archive_path, e))
        })?;

        let mut archive = zip::ZipArchive::new(file).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to read ZIP archive: {}", e))
        })?;

        let entry = archive.by_name(entry_name).map_err(|e| {
            TorshError::InvalidArgument(format!(
                "Failed to find entry '{}' in archive: {}",
                entry_name, e
            ))
        })?;

        Ok(entry.size())
    }
}

impl Default for LazyResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl LazyResourceManager {
    /// Create a new lazy resource manager
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            memory_limit: None,
            current_memory_usage: Arc::new(RwLock::new(0)),
            eviction_strategy: EvictionStrategy::LRU,
        }
    }

    /// Set memory limit in bytes
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = Some(limit);
        self
    }

    /// Set eviction strategy
    pub fn with_eviction_strategy(mut self, strategy: EvictionStrategy) -> Self {
        self.eviction_strategy = strategy;
        self
    }

    /// Add a lazy resource
    pub fn add_resource(&mut self, resource: LazyResource) -> Result<()> {
        if self.resources.contains_key(&resource.name) {
            return Err(TorshError::InvalidArgument(format!(
                "Resource '{}' already exists",
                resource.name
            )));
        }

        self.resources.insert(resource.name.clone(), resource);
        Ok(())
    }

    /// Get a lazy resource by name
    pub fn get_resource(&self, name: &str) -> Option<&LazyResource> {
        self.resources.get(name)
    }

    /// Load a resource's data (triggering lazy loading if necessary)
    pub fn load_resource_data(&self, name: &str) -> Result<Vec<u8>> {
        let resource = self
            .resources
            .get(name)
            .ok_or_else(|| TorshError::InvalidArgument(format!("Resource '{}' not found", name)))?;

        let data = resource.data()?;

        // Update memory usage if memory limit is set
        if self.memory_limit.is_some() {
            self.update_memory_usage(data.len())?;
        }

        Ok(data)
    }

    /// Preload multiple resources
    pub fn preload_resources(&self, names: &[&str]) -> Result<()> {
        for name in names {
            if let Some(resource) = self.resources.get(*name) {
                resource.preload()?;
            }
        }
        Ok(())
    }

    /// Evict all cached data
    pub fn evict_all_cache(&self) -> Result<()> {
        for resource in self.resources.values() {
            resource.evict_cache()?;
        }

        // Reset memory usage
        if let Ok(mut usage) = self.current_memory_usage.write() {
            *usage = 0;
        }

        Ok(())
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.current_memory_usage.read().map_or(0, |usage| *usage)
    }

    /// Get list of loaded resources
    pub fn loaded_resources(&self) -> Vec<String> {
        self.resources
            .iter()
            .filter(|(_, resource)| resource.is_loaded())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Update memory usage and potentially evict resources
    fn update_memory_usage(&self, additional_bytes: usize) -> Result<()> {
        if let Some(limit) = self.memory_limit {
            let mut usage = self.current_memory_usage.write().map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to acquire write lock: {}", e))
            })?;

            *usage += additional_bytes;

            if *usage > limit {
                // Need to evict some resources
                drop(usage); // Release the lock before calling evict
                self.evict_resources_to_limit()?;
            }
        }

        Ok(())
    }

    /// Evict resources according to the eviction strategy
    fn evict_resources_to_limit(&self) -> Result<()> {
        let limit = self.memory_limit.unwrap_or(usize::MAX);

        while self.memory_usage() > limit {
            let resource_to_evict = match self.eviction_strategy {
                EvictionStrategy::LRU => self.find_lru_resource(),
                EvictionStrategy::LargestFirst => self.find_largest_resource()?,
                EvictionStrategy::Random => self.find_random_resource(),
            };

            if let Some(resource_name) = resource_to_evict {
                if let Some(resource) = self.resources.get(&resource_name) {
                    let size = resource.size().unwrap_or(0) as usize;
                    resource.evict_cache()?;

                    // Update memory usage
                    if let Ok(mut usage) = self.current_memory_usage.write() {
                        *usage = usage.saturating_sub(size);
                    }
                } else {
                    break; // No more resources to evict
                }
            } else {
                break; // No resource found to evict
            }
        }

        Ok(())
    }

    /// Find least recently used resource (simplified implementation)
    fn find_lru_resource(&self) -> Option<String> {
        // For simplicity, just return the first loaded resource
        // In a production implementation, you'd track access times
        for (name, resource) in &self.resources {
            if resource.is_loaded() {
                return Some(name.clone());
            }
        }
        None
    }

    /// Find the largest loaded resource
    fn find_largest_resource(&self) -> Result<Option<String>> {
        let mut largest_size = 0u64;
        let mut largest_name = None;

        for (name, resource) in &self.resources {
            if resource.is_loaded() {
                let size = resource.size()?;
                if size > largest_size {
                    largest_size = size;
                    largest_name = Some(name.clone());
                }
            }
        }

        Ok(largest_name)
    }

    /// Find a random loaded resource
    fn find_random_resource(&self) -> Option<String> {
        let loaded_resources: Vec<_> = self
            .resources
            .iter()
            .filter(|(_, resource)| resource.is_loaded())
            .map(|(name, _)| name.clone())
            .collect();

        if loaded_resources.is_empty() {
            None
        } else {
            // For simplicity, just return the first one
            // In a production implementation, you'd use proper randomization
            loaded_resources.into_iter().next()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_lazy_resource_in_memory() {
        let data = b"test data".to_vec();
        let resource =
            LazyResource::new_in_memory("test".to_string(), ResourceType::Data, data.clone());

        assert!(resource.is_loaded());
        assert_eq!(resource.data().unwrap(), data);
        assert_eq!(resource.size().unwrap(), data.len() as u64);
    }

    #[test]
    fn test_lazy_resource_file() -> std::io::Result<()> {
        // Create a temporary file
        let mut temp_file = NamedTempFile::new()?;
        let test_data = b"test file data";
        temp_file.write_all(test_data)?;

        let resource = LazyResource::new_lazy_file(
            "test".to_string(),
            ResourceType::Data,
            temp_file.path(),
            0,
            test_data.len() as u64,
        );

        assert!(!resource.is_loaded());
        assert_eq!(resource.size().unwrap(), test_data.len() as u64);
        assert_eq!(resource.data().unwrap(), test_data);
        assert!(resource.is_loaded());

        Ok(())
    }

    #[test]
    fn test_lazy_resource_manager() {
        let mut manager = LazyResourceManager::new();

        let resource = LazyResource::new_in_memory(
            "test".to_string(),
            ResourceType::Data,
            b"test data".to_vec(),
        );

        manager.add_resource(resource).unwrap();
        assert!(manager.get_resource("test").is_some());

        let data = manager.load_resource_data("test").unwrap();
        assert_eq!(data, b"test data");
    }

    #[test]
    fn test_memory_limit() {
        let mut manager = LazyResourceManager::new().with_memory_limit(100);

        let large_resource = LazyResource::new_in_memory(
            "large".to_string(),
            ResourceType::Data,
            vec![0u8; 150], // Larger than the limit
        );

        manager.add_resource(large_resource).unwrap();

        // Loading the data should trigger eviction logic
        manager.load_resource_data("large").unwrap();
    }

    #[test]
    fn test_conversion_to_regular_resource() {
        let lazy_resource = LazyResource::new_in_memory(
            "test".to_string(),
            ResourceType::Data,
            b"test data".to_vec(),
        );

        let regular_resource = lazy_resource.to_resource().unwrap();
        assert_eq!(regular_resource.name, "test");
        assert_eq!(regular_resource.data, b"test data");
    }
}

/// Memory-mapped resource for efficient large file access
pub struct MappedResource {
    /// Resource name
    pub name: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Memory-mapped file
    mmap: Arc<Mutex<Option<Mmap>>>,
    /// Path to the file
    file_path: PathBuf,
    /// File offset
    offset: u64,
    /// File size
    size: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl MappedResource {
    /// Create a new memory-mapped resource
    pub fn new<P: Into<PathBuf>>(
        name: String,
        resource_type: ResourceType,
        file_path: P,
        offset: u64,
        size: u64,
    ) -> Self {
        Self {
            name,
            resource_type,
            mmap: Arc::new(Mutex::new(None)),
            file_path: file_path.into(),
            offset,
            size,
            metadata: HashMap::new(),
        }
    }

    /// Map the file into memory
    pub fn map(&self) -> Result<()> {
        let file = fs::File::open(&self.file_path)
            .map_err(|e| TorshError::IoError(format!("Failed to open file: {}", e)))?;

        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| TorshError::IoError(format!("Failed to map file: {}", e)))?;

        let mut guard = self
            .mmap
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to acquire lock: {}", e)))?;

        *guard = Some(mmap);
        Ok(())
    }

    /// Unmap the file from memory
    pub fn unmap(&self) -> Result<()> {
        let mut guard = self
            .mmap
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to acquire lock: {}", e)))?;

        *guard = None;
        Ok(())
    }

    /// Check if the file is currently mapped
    pub fn is_mapped(&self) -> bool {
        self.mmap.lock().map_or(false, |guard| guard.is_some())
    }

    /// Get a slice of the mapped data
    pub fn data(&self) -> Result<Vec<u8>> {
        if !self.is_mapped() {
            self.map()?;
        }

        let guard = self
            .mmap
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to acquire lock: {}", e)))?;

        let mmap = guard
            .as_ref()
            .ok_or_else(|| TorshError::InvalidArgument("File not mapped".to_string()))?;

        let start = self.offset as usize;
        let end = (self.offset + self.size) as usize;

        if end > mmap.len() {
            return Err(TorshError::InvalidArgument(
                "Requested range exceeds file size".to_string(),
            ));
        }

        Ok(mmap[start..end].to_vec())
    }

    /// Get the size of the resource
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// Streaming resource for processing large files chunk by chunk
pub struct StreamingResource {
    /// Resource name
    pub name: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Path to the file
    file_path: PathBuf,
    /// File offset
    offset: u64,
    /// File size
    size: u64,
    /// Chunk size for streaming
    chunk_size: usize,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl StreamingResource {
    /// Create a new streaming resource
    pub fn new<P: Into<PathBuf>>(
        name: String,
        resource_type: ResourceType,
        file_path: P,
        offset: u64,
        size: u64,
    ) -> Self {
        Self {
            name,
            resource_type,
            file_path: file_path.into(),
            offset,
            size,
            chunk_size: 1024 * 1024, // 1MB default chunk size
            metadata: HashMap::new(),
        }
    }

    /// Set chunk size for streaming
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Stream data through a callback function
    pub fn stream<F>(&self, mut callback: F) -> Result<()>
    where
        F: FnMut(&[u8]) -> Result<()>,
    {
        let mut file = fs::File::open(&self.file_path)
            .map_err(|e| TorshError::IoError(format!("Failed to open file: {}", e)))?;

        file.seek(std::io::SeekFrom::Start(self.offset))
            .map_err(|e| TorshError::IoError(format!("Failed to seek: {}", e)))?;

        let mut remaining = self.size as usize;
        let mut buffer = vec![0u8; self.chunk_size.min(remaining)];

        while remaining > 0 {
            let to_read = self.chunk_size.min(remaining);
            let bytes_read = file
                .read(&mut buffer[..to_read])
                .map_err(|e| TorshError::IoError(format!("Failed to read: {}", e)))?;

            if bytes_read == 0 {
                break;
            }

            callback(&buffer[..bytes_read])?;
            remaining -= bytes_read;
        }

        Ok(())
    }

    /// Stream data and collect into a vector (for convenience)
    pub fn collect(&self) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(self.size as usize);

        self.stream(|chunk| {
            result.extend_from_slice(chunk);
            Ok(())
        })?;

        Ok(result)
    }

    /// Process stream in parallel chunks using scirs2-core
    pub fn stream_parallel<F>(&self, callback: F) -> Result<()>
    where
        F: Fn(&[u8]) -> Result<()> + Send + Sync,
    {
        // Read all data first (in real implementation, you'd want to stream this)
        let data = self.collect()?;

        // Split into chunks
        let num_chunks = (data.len() + self.chunk_size - 1) / self.chunk_size;
        let chunks: Vec<&[u8]> = (0..num_chunks)
            .map(|i| {
                let start = i * self.chunk_size;
                let end = (start + self.chunk_size).min(data.len());
                &data[start..end]
            })
            .collect();

        // Process chunks in parallel
        use scirs2_core::parallel_ops::{IntoParallelIterator, ParallelIterator};

        let results: Vec<_> = chunks
            .into_par_iter()
            .map(|chunk| callback(chunk))
            .collect();

        for result in results {
            result?;
        }

        Ok(())
    }

    /// Get the size of the resource
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// Resource stream writer for creating large resources incrementally
pub struct ResourceStreamWriter {
    /// Resource name
    pub name: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Output file
    file: fs::File,
    /// Bytes written
    bytes_written: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ResourceStreamWriter {
    /// Create a new resource stream writer
    pub fn new<P: AsRef<Path>>(
        name: String,
        resource_type: ResourceType,
        output_path: P,
    ) -> Result<Self> {
        let file = fs::File::create(output_path)
            .map_err(|e| TorshError::IoError(format!("Failed to create file: {}", e)))?;

        Ok(Self {
            name,
            resource_type,
            file,
            bytes_written: 0,
            metadata: HashMap::new(),
        })
    }

    /// Write a chunk of data
    pub fn write_chunk(&mut self, data: &[u8]) -> Result<()> {
        self.file
            .write_all(data)
            .map_err(|e| TorshError::IoError(format!("Failed to write: {}", e)))?;

        self.bytes_written += data.len() as u64;
        Ok(())
    }

    /// Finalize the stream and return the resource info
    pub fn finalize(self) -> Result<(String, u64)> {
        Ok((self.name, self.bytes_written))
    }

    /// Get the number of bytes written
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_streaming_resource() -> std::io::Result<()> {
        // Create a temporary file
        let mut temp_file = NamedTempFile::new()?;
        let test_data = b"This is streaming test data with multiple chunks";
        temp_file.write_all(test_data)?;
        temp_file.flush()?;

        let streaming_resource = StreamingResource::new(
            "test_stream".to_string(),
            ResourceType::Data,
            temp_file.path(),
            0,
            test_data.len() as u64,
        )
        .with_chunk_size(10);

        let mut chunks = Vec::new();
        streaming_resource
            .stream(|chunk| {
                chunks.push(chunk.to_vec());
                Ok(())
            })
            .unwrap();

        // Verify chunks were read
        assert!(!chunks.is_empty());

        // Verify total data matches
        let collected: Vec<u8> = chunks.into_iter().flatten().collect();
        assert_eq!(collected, test_data);

        Ok(())
    }

    #[test]
    fn test_streaming_collect() -> std::io::Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        let test_data = b"Collect all streaming data";
        temp_file.write_all(test_data)?;
        temp_file.flush()?;

        let streaming_resource = StreamingResource::new(
            "test".to_string(),
            ResourceType::Data,
            temp_file.path(),
            0,
            test_data.len() as u64,
        );

        let collected = streaming_resource.collect().unwrap();
        assert_eq!(collected, test_data);

        Ok(())
    }

    #[test]
    fn test_memory_mapped_resource() -> std::io::Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        let test_data = b"Memory mapped test data";
        temp_file.write_all(test_data)?;
        temp_file.flush()?;

        let mapped_resource = MappedResource::new(
            "test_mmap".to_string(),
            ResourceType::Data,
            temp_file.path(),
            0,
            test_data.len() as u64,
        );

        assert!(!mapped_resource.is_mapped());

        mapped_resource.map().unwrap();
        assert!(mapped_resource.is_mapped());

        let data = mapped_resource.data().unwrap();
        assert_eq!(data, test_data);

        mapped_resource.unmap().unwrap();
        assert!(!mapped_resource.is_mapped());

        Ok(())
    }

    #[test]
    fn test_resource_stream_writer() -> std::io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path().to_path_buf();

        let mut writer =
            ResourceStreamWriter::new("test_writer".to_string(), ResourceType::Data, &path)
                .unwrap();

        writer.write_chunk(b"First chunk").unwrap();
        writer.write_chunk(b" Second chunk").unwrap();
        writer.write_chunk(b" Third chunk").unwrap();

        assert_eq!(writer.bytes_written(), 36);

        let (name, size) = writer.finalize().unwrap();
        assert_eq!(name, "test_writer");
        assert_eq!(size, 36);

        // Verify file contents
        let contents = fs::read(&path)?;
        assert_eq!(contents, b"First chunk Second chunk Third chunk");

        Ok(())
    }
}
