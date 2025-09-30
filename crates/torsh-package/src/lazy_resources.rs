//! Lazy loading resource system for packages
//!
//! This module provides lazy loading capabilities for package resources,
//! allowing packages to load only the resources that are actually needed.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
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
