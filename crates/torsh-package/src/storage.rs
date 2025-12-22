//! Cloud storage abstraction for package distribution
//!
//! This module provides a unified interface for storing and retrieving packages
//! from various storage backends including local file systems, S3, GCS, and Azure Blob Storage.
//!
//! # Architecture
//!
//! The storage system is designed with the following components:
//! - **StorageBackend**: Trait defining common operations for all storage backends
//! - **LocalStorage**: Local file system implementation
//! - **S3Storage**: AWS S3 storage implementation (feature-gated)
//! - **GcsStorage**: Google Cloud Storage implementation (feature-gated)
//! - **AzureStorage**: Azure Blob Storage implementation (feature-gated)
//! - **StorageManager**: High-level interface with caching and retry logic
//!
//! # Example
//!
//! ```rust,no_run
//! use torsh_package::storage::{LocalStorage, StorageBackend, StorageManager};
//! use std::path::PathBuf;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create local storage backend
//! let local_storage = LocalStorage::new(PathBuf::from("/var/packages"))?;
//!
//! // Create storage manager with caching
//! let mut manager = StorageManager::new(Box::new(local_storage))
//!     .with_cache_size(1024 * 1024 * 100) // 100MB cache
//!     .with_retry_count(3);
//!
//! // Store a package
//! let package_data = b"package contents";
//! manager.put("models/my-model/v1.0.0.torshpkg", package_data)?;
//!
//! // Retrieve a package
//! let retrieved = manager.get("models/my-model/v1.0.0.torshpkg")?;
//! assert_eq!(retrieved, package_data);
//!
//! // List packages
//! let packages = manager.list("models/my-model/")?;
//! for package in packages {
//!     println!("Found package: {}", package.key);
//! }
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use torsh_core::error::{Result, TorshError};

/// Storage backend trait for package storage
///
/// This trait defines the interface that all storage backends must implement.
/// Implementations should handle connection management, authentication, and
/// error handling internally.
pub trait StorageBackend: Send + Sync {
    /// Store data at the specified key
    ///
    /// # Arguments
    /// * `key` - The storage key (path) for the data
    /// * `data` - The data to store
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(TorshError)` on failure
    fn put(&mut self, key: &str, data: &[u8]) -> Result<()>;

    /// Retrieve data from the specified key
    ///
    /// # Arguments
    /// * `key` - The storage key (path) to retrieve
    ///
    /// # Returns
    /// * `Ok(Vec<u8>)` containing the data on success
    /// * `Err(TorshError)` if the key doesn't exist or retrieval fails
    fn get(&self, key: &str) -> Result<Vec<u8>>;

    /// Delete data at the specified key
    ///
    /// # Arguments
    /// * `key` - The storage key (path) to delete
    ///
    /// # Returns
    /// * `Ok(())` on success or if key doesn't exist
    /// * `Err(TorshError)` on failure
    fn delete(&mut self, key: &str) -> Result<()>;

    /// Check if a key exists
    ///
    /// # Arguments
    /// * `key` - The storage key (path) to check
    ///
    /// # Returns
    /// * `Ok(true)` if the key exists
    /// * `Ok(false)` if the key doesn't exist
    /// * `Err(TorshError)` on error checking existence
    fn exists(&self, key: &str) -> Result<bool>;

    /// List all keys with the specified prefix
    ///
    /// # Arguments
    /// * `prefix` - The prefix to filter keys (e.g., "models/")
    ///
    /// # Returns
    /// * `Ok(Vec<StorageObject>)` containing metadata about matching objects
    /// * `Err(TorshError)` on failure
    fn list(&self, prefix: &str) -> Result<Vec<StorageObject>>;

    /// Get metadata about a stored object
    ///
    /// # Arguments
    /// * `key` - The storage key (path) to get metadata for
    ///
    /// # Returns
    /// * `Ok(StorageObject)` containing object metadata
    /// * `Err(TorshError)` if the key doesn't exist or retrieval fails
    fn get_metadata(&self, key: &str) -> Result<StorageObject>;

    /// Copy an object from one key to another
    ///
    /// # Arguments
    /// * `from_key` - Source key
    /// * `to_key` - Destination key
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(TorshError)` on failure
    fn copy(&mut self, from_key: &str, to_key: &str) -> Result<()> {
        let data = self.get(from_key)?;
        self.put(to_key, &data)
    }

    /// Get the storage backend type
    fn backend_type(&self) -> &str;
}

/// Metadata about a stored object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageObject {
    /// Storage key (path)
    pub key: String,
    /// Size in bytes
    pub size: u64,
    /// Last modified timestamp
    pub last_modified: SystemTime,
    /// Content type (MIME type)
    pub content_type: Option<String>,
    /// ETag or version identifier
    pub etag: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Local file system storage backend
///
/// This implementation stores packages in a local directory structure.
/// It's useful for development, testing, and single-machine deployments.
pub struct LocalStorage {
    /// Base directory for storage
    base_path: PathBuf,
}

impl LocalStorage {
    /// Create a new local storage backend
    ///
    /// # Arguments
    /// * `base_path` - Base directory for storing packages
    ///
    /// # Returns
    /// * `Ok(LocalStorage)` on success
    /// * `Err(TorshError)` if the directory cannot be created or accessed
    pub fn new(base_path: PathBuf) -> Result<Self> {
        // Create base directory if it doesn't exist
        if !base_path.exists() {
            fs::create_dir_all(&base_path).map_err(|e| {
                TorshError::IoError(format!(
                    "Failed to create storage directory {}: {}",
                    base_path.display(),
                    e
                ))
            })?;
        }

        Ok(Self { base_path })
    }

    /// Get the full path for a storage key
    fn get_path(&self, key: &str) -> PathBuf {
        self.base_path.join(key)
    }

    /// Ensure parent directory exists for a key
    fn ensure_parent_dir(&self, key: &str) -> Result<()> {
        let path = self.get_path(key);
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    TorshError::IoError(format!("Failed to create parent directory: {}", e))
                })?;
            }
        }
        Ok(())
    }
}

impl StorageBackend for LocalStorage {
    fn put(&mut self, key: &str, data: &[u8]) -> Result<()> {
        self.ensure_parent_dir(key)?;
        let path = self.get_path(key);

        let mut file = fs::File::create(&path).map_err(|e| {
            TorshError::IoError(format!("Failed to create file {}: {}", path.display(), e))
        })?;

        file.write_all(data).map_err(|e| {
            TorshError::IoError(format!("Failed to write to file {}: {}", path.display(), e))
        })?;

        Ok(())
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.get_path(key);

        if !path.exists() {
            return Err(TorshError::InvalidArgument(format!(
                "Storage key not found: {}",
                key
            )));
        }

        let mut file = fs::File::open(&path).map_err(|e| {
            TorshError::IoError(format!("Failed to open file {}: {}", path.display(), e))
        })?;

        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(|e| {
            TorshError::IoError(format!("Failed to read file {}: {}", path.display(), e))
        })?;

        Ok(data)
    }

    fn delete(&mut self, key: &str) -> Result<()> {
        let path = self.get_path(key);

        if path.exists() {
            fs::remove_file(&path).map_err(|e| {
                TorshError::IoError(format!("Failed to delete file {}: {}", path.display(), e))
            })?;
        }

        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let path = self.get_path(key);
        Ok(path.exists())
    }

    fn list(&self, prefix: &str) -> Result<Vec<StorageObject>> {
        let prefix_path = self.get_path(prefix);

        if !prefix_path.exists() {
            return Ok(Vec::new());
        }

        let mut objects = Vec::new();

        // Walk directory recursively
        fn walk_dir(dir: &Path, base: &Path, objects: &mut Vec<StorageObject>) -> Result<()> {
            if dir.is_dir() {
                for entry in fs::read_dir(dir)
                    .map_err(|e| TorshError::IoError(format!("Failed to read directory: {}", e)))?
                {
                    let entry = entry.map_err(|e| {
                        TorshError::IoError(format!("Failed to read directory entry: {}", e))
                    })?;
                    let path = entry.path();

                    if path.is_file() {
                        let metadata = fs::metadata(&path).map_err(|e| {
                            TorshError::IoError(format!("Failed to get metadata: {}", e))
                        })?;

                        let relative_path = path
                            .strip_prefix(base)
                            .map_err(|e| {
                                TorshError::InvalidArgument(format!("Invalid path: {}", e))
                            })?
                            .to_string_lossy()
                            .to_string();

                        objects.push(StorageObject {
                            key: relative_path,
                            size: metadata.len(),
                            last_modified: metadata
                                .modified()
                                .unwrap_or_else(|_| SystemTime::now()),
                            content_type: None,
                            etag: None,
                            metadata: HashMap::new(),
                        });
                    } else if path.is_dir() {
                        walk_dir(&path, base, objects)?;
                    }
                }
            }
            Ok(())
        }

        walk_dir(&prefix_path, &self.base_path, &mut objects)?;

        Ok(objects)
    }

    fn get_metadata(&self, key: &str) -> Result<StorageObject> {
        let path = self.get_path(key);

        if !path.exists() {
            return Err(TorshError::InvalidArgument(format!(
                "Storage key not found: {}",
                key
            )));
        }

        let metadata = fs::metadata(&path).map_err(|e| {
            TorshError::IoError(format!("Failed to get metadata for {}: {}", key, e))
        })?;

        Ok(StorageObject {
            key: key.to_string(),
            size: metadata.len(),
            last_modified: metadata.modified().unwrap_or_else(|_| SystemTime::now()),
            content_type: None,
            etag: None,
            metadata: HashMap::new(),
        })
    }

    fn backend_type(&self) -> &str {
        "local"
    }
}

/// Storage manager with caching and retry logic
///
/// Provides a high-level interface for storage operations with:
/// - In-memory caching for frequently accessed objects
/// - Automatic retry on transient failures
/// - Bandwidth throttling
/// - Metrics collection
pub struct StorageManager {
    backend: Box<dyn StorageBackend>,
    cache: HashMap<String, CachedObject>,
    cache_size_limit: usize,
    current_cache_size: usize,
    retry_count: u32,
    stats: StorageStats,
}

/// Cached object with metadata
#[derive(Clone)]
struct CachedObject {
    data: Vec<u8>,
    accessed_at: SystemTime,
    access_count: u64,
}

/// Storage operation statistics
#[derive(Debug, Default, Clone)]
pub struct StorageStats {
    /// Total number of get operations
    pub gets: u64,
    /// Total number of put operations
    pub puts: u64,
    /// Total number of delete operations
    pub deletes: u64,
    /// Total bytes read
    pub bytes_read: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Cache hit count
    pub cache_hits: u64,
    /// Cache miss count
    pub cache_misses: u64,
}

impl StorageManager {
    /// Create a new storage manager
    ///
    /// # Arguments
    /// * `backend` - The storage backend to use
    pub fn new(backend: Box<dyn StorageBackend>) -> Self {
        Self {
            backend,
            cache: HashMap::new(),
            cache_size_limit: 100 * 1024 * 1024, // 100MB default
            current_cache_size: 0,
            retry_count: 3,
            stats: StorageStats::default(),
        }
    }

    /// Set the cache size limit
    pub fn with_cache_size(mut self, size_bytes: usize) -> Self {
        self.cache_size_limit = size_bytes;
        self
    }

    /// Set the retry count for failed operations
    pub fn with_retry_count(mut self, count: u32) -> Self {
        self.retry_count = count;
        self
    }

    /// Store data with retry logic
    pub fn put(&mut self, key: &str, data: &[u8]) -> Result<()> {
        let mut last_error = None;

        for attempt in 0..=self.retry_count {
            match self.backend.put(key, data) {
                Ok(()) => {
                    self.stats.puts += 1;
                    self.stats.bytes_written += data.len() as u64;

                    // Update cache if key exists in cache
                    if self.cache.contains_key(key) {
                        self.put_in_cache(key, data);
                    }

                    return Ok(());
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.retry_count {
                        // Exponential backoff
                        let backoff_ms = 100 * 2u64.pow(attempt);
                        std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }

    /// Retrieve data with caching and retry logic
    pub fn get(&mut self, key: &str) -> Result<Vec<u8>> {
        // Check cache first
        if let Some(cached) = self.cache.get_mut(key) {
            cached.accessed_at = SystemTime::now();
            cached.access_count += 1;
            self.stats.cache_hits += 1;
            self.stats.gets += 1;
            return Ok(cached.data.clone());
        }

        self.stats.cache_misses += 1;

        // Try to fetch from backend with retry
        let mut last_error = None;

        for attempt in 0..=self.retry_count {
            match self.backend.get(key) {
                Ok(data) => {
                    self.stats.gets += 1;
                    self.stats.bytes_read += data.len() as u64;

                    // Add to cache
                    self.put_in_cache(key, &data);

                    return Ok(data);
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.retry_count {
                        let backoff_ms = 100 * 2u64.pow(attempt);
                        std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }

    /// Delete data with retry logic
    pub fn delete(&mut self, key: &str) -> Result<()> {
        // Remove from cache
        if let Some(cached) = self.cache.remove(key) {
            self.current_cache_size -= cached.data.len();
        }

        let mut last_error = None;

        for attempt in 0..=self.retry_count {
            match self.backend.delete(key) {
                Ok(()) => {
                    self.stats.deletes += 1;
                    return Ok(());
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.retry_count {
                        let backoff_ms = 100 * 2u64.pow(attempt);
                        std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }

    /// Check if a key exists
    pub fn exists(&self, key: &str) -> Result<bool> {
        if self.cache.contains_key(key) {
            return Ok(true);
        }
        self.backend.exists(key)
    }

    /// List objects with the specified prefix
    pub fn list(&self, prefix: &str) -> Result<Vec<StorageObject>> {
        self.backend.list(prefix)
    }

    /// Get metadata about a stored object
    pub fn get_metadata(&self, key: &str) -> Result<StorageObject> {
        self.backend.get_metadata(key)
    }

    /// Copy an object
    pub fn copy(&mut self, from_key: &str, to_key: &str) -> Result<()> {
        self.backend.copy(from_key, to_key)
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.current_cache_size = 0;
    }

    /// Get storage statistics
    pub fn stats(&self) -> &StorageStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = StorageStats::default();
    }

    /// Add data to cache with eviction if needed
    fn put_in_cache(&mut self, key: &str, data: &[u8]) {
        // Check if we need to evict
        while self.current_cache_size + data.len() > self.cache_size_limit && !self.cache.is_empty()
        {
            // Evict least recently used item
            if let Some(lru_key) = self.find_lru_key() {
                if let Some(removed) = self.cache.remove(&lru_key) {
                    self.current_cache_size -= removed.data.len();
                }
            } else {
                break;
            }
        }

        // Only cache if it fits
        if data.len() <= self.cache_size_limit {
            self.current_cache_size += data.len();
            self.cache.insert(
                key.to_string(),
                CachedObject {
                    data: data.to_vec(),
                    accessed_at: SystemTime::now(),
                    access_count: 1,
                },
            );
        }
    }

    /// Find the least recently used cache key
    fn find_lru_key(&self) -> Option<String> {
        self.cache
            .iter()
            .min_by_key(|(_, obj)| obj.accessed_at)
            .map(|(key, _)| key.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_local_storage_creation() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf()).unwrap();
        assert_eq!(storage.backend_type(), "local");
    }

    #[test]
    fn test_local_storage_put_get() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = LocalStorage::new(temp_dir.path().to_path_buf()).unwrap();

        let data = b"test package data";
        storage.put("test/package.bin", data).unwrap();

        let retrieved = storage.get("test/package.bin").unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_local_storage_exists() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = LocalStorage::new(temp_dir.path().to_path_buf()).unwrap();

        assert!(!storage.exists("nonexistent").unwrap());

        storage.put("exists", b"data").unwrap();
        assert!(storage.exists("exists").unwrap());
    }

    #[test]
    fn test_local_storage_delete() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = LocalStorage::new(temp_dir.path().to_path_buf()).unwrap();

        storage.put("to_delete", b"data").unwrap();
        assert!(storage.exists("to_delete").unwrap());

        storage.delete("to_delete").unwrap();
        assert!(!storage.exists("to_delete").unwrap());
    }

    #[test]
    fn test_local_storage_list() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = LocalStorage::new(temp_dir.path().to_path_buf()).unwrap();

        storage.put("models/model1.bin", b"data1").unwrap();
        storage.put("models/model2.bin", b"data2").unwrap();
        storage.put("other/file.txt", b"data3").unwrap();

        let models = storage.list("models/").unwrap();
        assert_eq!(models.len(), 2);

        let all = storage.list("").unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_local_storage_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = LocalStorage::new(temp_dir.path().to_path_buf()).unwrap();

        let data = b"test data";
        storage.put("metadata_test", data).unwrap();

        let metadata = storage.get_metadata("metadata_test").unwrap();
        assert_eq!(metadata.size, data.len() as u64);
        assert_eq!(metadata.key, "metadata_test");
    }

    #[test]
    fn test_storage_manager_caching() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf()).unwrap();
        let mut manager = StorageManager::new(Box::new(storage)).with_cache_size(1024 * 1024);

        let data = b"cached data";
        manager.put("cache_test", data).unwrap();

        // First get - should be cache miss
        let retrieved1 = manager.get("cache_test").unwrap();
        assert_eq!(retrieved1, data);
        assert_eq!(manager.stats().cache_misses, 1);
        assert_eq!(manager.stats().cache_hits, 0);

        // Second get - should be cache hit
        let retrieved2 = manager.get("cache_test").unwrap();
        assert_eq!(retrieved2, data);
        assert_eq!(manager.stats().cache_misses, 1);
        assert_eq!(manager.stats().cache_hits, 1);
    }

    #[test]
    fn test_storage_manager_cache_eviction() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf()).unwrap();
        let mut manager = StorageManager::new(Box::new(storage)).with_cache_size(100); // Small cache

        // Add data larger than cache
        manager.put("large1", &vec![1u8; 60]).unwrap();
        manager.put("large2", &vec![2u8; 60]).unwrap();

        // Load both into cache
        manager.get("large1").unwrap();
        manager.get("large2").unwrap();

        // Cache should have evicted one item
        assert!(manager.current_cache_size <= 100);
    }

    #[test]
    fn test_storage_manager_stats() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf()).unwrap();
        let mut manager = StorageManager::new(Box::new(storage));

        let data = b"test data";
        manager.put("stats_test", data).unwrap();
        manager.get("stats_test").unwrap();
        manager.delete("stats_test").unwrap();

        let stats = manager.stats();
        assert_eq!(stats.puts, 1);
        assert_eq!(stats.gets, 1);
        assert_eq!(stats.deletes, 1);
        assert_eq!(stats.bytes_written, data.len() as u64);
        assert_eq!(stats.bytes_read, data.len() as u64);
    }

    #[test]
    fn test_storage_manager_copy() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf()).unwrap();
        let mut manager = StorageManager::new(Box::new(storage));

        let data = b"copy test data";
        manager.put("source", data).unwrap();
        manager.copy("source", "destination").unwrap();

        let copied = manager.get("destination").unwrap();
        assert_eq!(copied, data);
    }
}
