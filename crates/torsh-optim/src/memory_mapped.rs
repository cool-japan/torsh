use crate::OptimizerError;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

/// Configuration for memory-mapped optimizer states
#[derive(Debug, Clone)]
pub struct MemoryMappedConfig {
    /// Base directory for memory-mapped files
    pub base_dir: PathBuf,
    /// Initial size for memory-mapped files (in bytes)
    pub initial_size: usize,
    /// Growth factor when expanding files
    pub growth_factor: f32,
    /// Whether to sync to disk automatically
    pub auto_sync: bool,
    /// Sync frequency (every N operations)
    pub sync_frequency: usize,
    /// Whether to use advisory locking
    pub use_locking: bool,
    /// Whether to prefault pages
    pub prefault: bool,
}

impl Default for MemoryMappedConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("mmap_states"),
            initial_size: 1024 * 1024, // 1MB
            growth_factor: 2.0,
            auto_sync: true,
            sync_frequency: 100,
            use_locking: true,
            prefault: false,
        }
    }
}

/// Memory-mapped file wrapper
pub struct MemoryMappedFile {
    file: File,
    path: PathBuf,
    size: usize,
    capacity: usize,
    sync_counter: usize,
    config: MemoryMappedConfig,
}

impl MemoryMappedFile {
    /// Create a new memory-mapped file
    pub fn new(path: PathBuf, config: MemoryMappedConfig) -> Result<Self, OptimizerError> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                OptimizerError::MemoryMapError(format!("Failed to create directory: {}", e))
            })?;
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .map_err(|e| OptimizerError::MemoryMapError(format!("Failed to open file: {}", e)))?;

        // Set initial file size
        let initial_size = config.initial_size;
        file.set_len(initial_size as u64).map_err(|e| {
            OptimizerError::MemoryMapError(format!("Failed to set file size: {}", e))
        })?;

        Ok(Self {
            file,
            path,
            size: 0,
            capacity: initial_size,
            sync_counter: 0,
            config,
        })
    }

    /// Open an existing memory-mapped file
    pub fn open(path: PathBuf, config: MemoryMappedConfig) -> Result<Self, OptimizerError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| OptimizerError::MemoryMapError(format!("Failed to open file: {}", e)))?;

        let metadata = file.metadata().map_err(|e| {
            OptimizerError::MemoryMapError(format!("Failed to get file metadata: {}", e))
        })?;

        let capacity = metadata.len() as usize;

        Ok(Self {
            file,
            path,
            size: capacity, // Assume file is fully used when opening
            capacity,
            sync_counter: 0,
            config,
        })
    }

    /// Write data to the memory-mapped file
    pub fn write(&mut self, data: &[u8]) -> Result<usize, OptimizerError> {
        // Check if we need to expand the file
        if self.size + data.len() > self.capacity {
            self.expand(data.len())?;
        }

        // Seek to the end of the used data
        self.file
            .seek(SeekFrom::Start(self.size as u64))
            .map_err(|e| OptimizerError::MemoryMapError(format!("Failed to seek: {}", e)))?;

        // Write the data
        let bytes_written = self
            .file
            .write(data)
            .map_err(|e| OptimizerError::MemoryMapError(format!("Failed to write: {}", e)))?;

        self.size += bytes_written;
        self.sync_counter += 1;

        // Auto-sync if configured
        if self.config.auto_sync && self.sync_counter >= self.config.sync_frequency {
            self.sync()?;
        }

        Ok(bytes_written)
    }

    /// Read data from the memory-mapped file
    pub fn read(&mut self, offset: usize, length: usize) -> Result<Vec<u8>, OptimizerError> {
        if offset + length > self.size {
            return Err(OptimizerError::MemoryMapError(
                "Read extends beyond file size".to_string(),
            ));
        }

        self.file
            .seek(SeekFrom::Start(offset as u64))
            .map_err(|e| OptimizerError::MemoryMapError(format!("Failed to seek: {}", e)))?;

        let mut buffer = vec![0u8; length];
        self.file
            .read_exact(&mut buffer)
            .map_err(|e| OptimizerError::MemoryMapError(format!("Failed to read: {}", e)))?;

        Ok(buffer)
    }

    /// Read all data from the file
    pub fn read_all(&mut self) -> Result<Vec<u8>, OptimizerError> {
        self.read(0, self.size)
    }

    /// Overwrite data at a specific offset
    pub fn write_at(&mut self, offset: usize, data: &[u8]) -> Result<(), OptimizerError> {
        if offset + data.len() > self.capacity {
            self.expand(offset + data.len() - self.capacity)?;
        }

        self.file
            .seek(SeekFrom::Start(offset as u64))
            .map_err(|e| OptimizerError::MemoryMapError(format!("Failed to seek: {}", e)))?;

        self.file
            .write_all(data)
            .map_err(|e| OptimizerError::MemoryMapError(format!("Failed to write: {}", e)))?;

        // Update size if we wrote beyond the current end
        if offset + data.len() > self.size {
            self.size = offset + data.len();
        }

        self.sync_counter += 1;

        if self.config.auto_sync && self.sync_counter >= self.config.sync_frequency {
            self.sync()?;
        }

        Ok(())
    }

    /// Sync the file to disk
    pub fn sync(&mut self) -> Result<(), OptimizerError> {
        self.file
            .sync_all()
            .map_err(|e| OptimizerError::MemoryMapError(format!("Failed to sync: {}", e)))?;
        self.sync_counter = 0;
        Ok(())
    }

    /// Truncate the file to a specific size
    pub fn truncate(&mut self, size: usize) -> Result<(), OptimizerError> {
        self.file
            .set_len(size as u64)
            .map_err(|e| OptimizerError::MemoryMapError(format!("Failed to truncate: {}", e)))?;

        self.size = size.min(self.size);
        self.capacity = size;

        Ok(())
    }

    /// Get the current file size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the current file capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    // Private methods

    fn expand(&mut self, min_additional: usize) -> Result<(), OptimizerError> {
        let new_capacity =
            ((self.capacity + min_additional) as f32 * self.config.growth_factor) as usize;

        self.file
            .set_len(new_capacity as u64)
            .map_err(|e| OptimizerError::MemoryMapError(format!("Failed to expand file: {}", e)))?;

        self.capacity = new_capacity;

        Ok(())
    }
}

/// Memory-mapped optimizer state storage
pub struct MemoryMappedStateStorage {
    config: MemoryMappedConfig,
    files: HashMap<String, Arc<Mutex<MemoryMappedFile>>>,
    metadata: Arc<RwLock<StateMetadata>>,
}

#[derive(Debug, Clone, Default)]
struct StateMetadata {
    entries: HashMap<String, StateEntry>,
}

#[derive(Debug, Clone)]
struct StateEntry {
    file_key: String,
    offset: usize,
    length: usize,
    data_type: String,
    timestamp: u64,
}

impl MemoryMappedStateStorage {
    /// Create a new memory-mapped state storage
    pub fn new(config: MemoryMappedConfig) -> Result<Self, OptimizerError> {
        // Create base directory
        std::fs::create_dir_all(&config.base_dir).map_err(|e| {
            OptimizerError::MemoryMapError(format!("Failed to create base directory: {}", e))
        })?;

        Ok(Self {
            config,
            files: HashMap::new(),
            metadata: Arc::new(RwLock::new(StateMetadata::default())),
        })
    }

    /// Store optimizer state data
    pub fn store<T: serde::Serialize>(
        &mut self,
        key: &str,
        data: &T,
    ) -> Result<(), OptimizerError> {
        let serialized =
            bincode::serde::encode_to_vec(data, bincode::config::standard()).map_err(|e| {
                OptimizerError::MemoryMapError(format!("Failed to serialize data: {}", e))
            })?;

        self.store_raw(key, &serialized, std::any::type_name::<T>())
    }

    /// Store raw bytes
    pub fn store_raw(
        &mut self,
        key: &str,
        data: &[u8],
        data_type: &str,
    ) -> Result<(), OptimizerError> {
        let file_key = format!("state_{}", key.replace('/', "_"));
        let file_path = self.config.base_dir.join(format!("{}.mmap", file_key));

        // Get or create the memory-mapped file
        let file_mutex = if let Some(existing) = self.files.get(&file_key) {
            existing.clone()
        } else {
            let mmap_file = MemoryMappedFile::new(file_path, self.config.clone())?;
            let file_mutex = Arc::new(Mutex::new(mmap_file));
            self.files.insert(file_key.clone(), file_mutex.clone());
            file_mutex
        };

        // Write data to the file
        let offset = {
            let mut file = file_mutex.lock().unwrap();
            let offset = file.size();
            file.write(data)?;
            offset
        };

        // Update metadata
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let entry = StateEntry {
            file_key: file_key.clone(),
            offset,
            length: data.len(),
            data_type: data_type.to_string(),
            timestamp,
        };

        let mut metadata = self.metadata.write().unwrap();
        metadata.entries.insert(key.to_string(), entry);

        Ok(())
    }

    /// Load optimizer state data
    pub fn load<T: serde::de::DeserializeOwned>(
        &mut self,
        key: &str,
    ) -> Result<Option<T>, OptimizerError> {
        if let Some(data) = self.load_raw(key)? {
            let (deserialized, _): (T, usize) =
                bincode::serde::decode_from_slice(&data, bincode::config::standard()).map_err(
                    |e| {
                        OptimizerError::MemoryMapError(format!("Failed to deserialize data: {}", e))
                    },
                )?;
            Ok(Some(deserialized))
        } else {
            Ok(None)
        }
    }

    /// Load raw bytes
    pub fn load_raw(&mut self, key: &str) -> Result<Option<Vec<u8>>, OptimizerError> {
        let metadata = self.metadata.read().unwrap();
        let entry = if let Some(entry) = metadata.entries.get(key) {
            entry.clone()
        } else {
            return Ok(None);
        };
        drop(metadata);

        // Get the memory-mapped file
        let file_mutex = self
            .files
            .get(&entry.file_key)
            .ok_or_else(|| OptimizerError::MemoryMapError("File not found".to_string()))?
            .clone();

        // Read data from the file
        let mut file = file_mutex.lock().unwrap();
        let data = file.read(entry.offset, entry.length)?;

        Ok(Some(data))
    }

    /// Update existing state data
    pub fn update<T: serde::Serialize>(
        &mut self,
        key: &str,
        data: &T,
    ) -> Result<(), OptimizerError> {
        let serialized =
            bincode::serde::encode_to_vec(data, bincode::config::standard()).map_err(|e| {
                OptimizerError::MemoryMapError(format!("Failed to serialize data: {}", e))
            })?;

        self.update_raw(key, &serialized)
    }

    /// Update raw bytes
    pub fn update_raw(&mut self, key: &str, data: &[u8]) -> Result<(), OptimizerError> {
        let metadata = self.metadata.read().unwrap();
        let mut entry = if let Some(entry) = metadata.entries.get(key) {
            entry.clone()
        } else {
            drop(metadata);
            return self.store_raw(key, data, "unknown");
        };
        drop(metadata);

        let file_mutex = self
            .files
            .get(&entry.file_key)
            .ok_or_else(|| OptimizerError::MemoryMapError("File not found".to_string()))?
            .clone();

        // If new data fits in the existing space, overwrite
        if data.len() <= entry.length {
            let mut file = file_mutex.lock().unwrap();
            file.write_at(entry.offset, data)?;
        } else {
            // Otherwise, append new data and update metadata
            let mut file = file_mutex.lock().unwrap();
            entry.offset = file.size();
            entry.length = data.len();
            file.write(data)?;
        }

        // Update timestamp
        entry.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut metadata = self.metadata.write().unwrap();
        metadata.entries.insert(key.to_string(), entry);

        Ok(())
    }

    /// Remove state data
    pub fn remove(&mut self, key: &str) -> Result<bool, OptimizerError> {
        let mut metadata = self.metadata.write().unwrap();
        Ok(metadata.entries.remove(key).is_some())
    }

    /// List all stored keys
    pub fn keys(&self) -> Vec<String> {
        let metadata = self.metadata.read().unwrap();
        metadata.entries.keys().cloned().collect()
    }

    /// Check if a key exists
    pub fn contains_key(&self, key: &str) -> bool {
        let metadata = self.metadata.read().unwrap();
        metadata.entries.contains_key(key)
    }

    /// Get storage statistics
    pub fn statistics(&self) -> StorageStatistics {
        let metadata = self.metadata.read().unwrap();
        let total_entries = metadata.entries.len();

        let total_size: usize = self
            .files
            .values()
            .map(|file_mutex| {
                let file = file_mutex.lock().unwrap();
                file.size()
            })
            .sum();

        let total_capacity: usize = self
            .files
            .values()
            .map(|file_mutex| {
                let file = file_mutex.lock().unwrap();
                file.capacity()
            })
            .sum();

        StorageStatistics {
            total_entries,
            total_size,
            total_capacity,
            utilization: if total_capacity > 0 {
                total_size as f32 / total_capacity as f32
            } else {
                0.0
            },
            num_files: self.files.len(),
        }
    }

    /// Sync all files to disk
    pub fn sync_all(&mut self) -> Result<(), OptimizerError> {
        for file_mutex in self.files.values() {
            let mut file = file_mutex.lock().unwrap();
            file.sync()?;
        }
        Ok(())
    }

    /// Compact storage by removing unused space
    pub fn compact(&mut self) -> Result<(), OptimizerError> {
        // Implementation would involve rewriting files to remove gaps
        // For now, just sync all files
        self.sync_all()
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStatistics {
    pub total_entries: usize,
    pub total_size: usize,
    pub total_capacity: usize,
    pub utilization: f32,
    pub num_files: usize,
}

impl std::fmt::Display for StorageStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Memory-Mapped Storage Statistics:")?;
        writeln!(f, "  Total Entries: {}", self.total_entries)?;
        writeln!(
            f,
            "  Total Size: {:.2} MB",
            self.total_size as f64 / 1024.0 / 1024.0
        )?;
        writeln!(
            f,
            "  Total Capacity: {:.2} MB",
            self.total_capacity as f64 / 1024.0 / 1024.0
        )?;
        writeln!(f, "  Utilization: {:.1}%", self.utilization * 100.0)?;
        writeln!(f, "  Number of Files: {}", self.num_files)?;
        Ok(())
    }
}

/// Trait for optimizers that support memory-mapped states
pub trait MemoryMappedSupport {
    /// Save state to memory-mapped storage
    fn save_to_mmap(
        &self,
        storage: &mut MemoryMappedStateStorage,
        prefix: &str,
    ) -> Result<(), OptimizerError>;

    /// Load state from memory-mapped storage
    fn load_from_mmap(
        &mut self,
        storage: &mut MemoryMappedStateStorage,
        prefix: &str,
    ) -> Result<(), OptimizerError>;

    /// Get list of state keys for this optimizer
    fn mmap_state_keys(&self, prefix: &str) -> Vec<String>;
}

/// Wrapper optimizer that uses memory-mapped storage
pub struct MemoryMappedOptimizer<T> {
    inner: T,
    storage: MemoryMappedStateStorage,
    prefix: String,
    auto_save: bool,
    save_frequency: usize,
    step_count: usize,
}

impl<T> MemoryMappedOptimizer<T>
where
    T: MemoryMappedSupport,
{
    /// Create a new memory-mapped optimizer wrapper
    pub fn new(
        inner: T,
        config: MemoryMappedConfig,
        prefix: String,
    ) -> Result<Self, OptimizerError> {
        let storage = MemoryMappedStateStorage::new(config)?;

        Ok(Self {
            inner,
            storage,
            prefix,
            auto_save: true,
            save_frequency: 100,
            step_count: 0,
        })
    }

    /// Enable or disable auto-save
    pub fn set_auto_save(&mut self, enabled: bool, frequency: usize) {
        self.auto_save = enabled;
        self.save_frequency = frequency;
    }

    /// Get the inner optimizer
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get the inner optimizer mutably
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Get the storage
    pub fn storage(&self) -> &MemoryMappedStateStorage {
        &self.storage
    }

    /// Get the storage mutably
    pub fn storage_mut(&mut self) -> &mut MemoryMappedStateStorage {
        &mut self.storage
    }

    /// Manually save state
    pub fn save_state(&mut self) -> Result<(), OptimizerError> {
        self.inner.save_to_mmap(&mut self.storage, &self.prefix)
    }

    /// Load state
    pub fn load_state(&mut self) -> Result<(), OptimizerError> {
        self.inner.load_from_mmap(&mut self.storage, &self.prefix)
    }

    /// Step the optimizer and handle auto-save
    pub fn step_with_mmap(&mut self) -> Result<(), OptimizerError> {
        self.step_count += 1;

        if self.auto_save && (self.step_count % self.save_frequency == 0) {
            self.save_state()?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_memory_mapped_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = MemoryMappedConfig::default();
        let path = temp_dir.path().join("test.mmap");

        let mut file = MemoryMappedFile::new(path.clone(), config).unwrap();

        // Test writing
        let data = b"Hello, world!";
        let bytes_written = file.write(data).unwrap();
        assert_eq!(bytes_written, data.len());

        // Test reading
        let read_data = file.read(0, data.len()).unwrap();
        assert_eq!(read_data, data);

        // Test write_at
        let new_data = b"Hi";
        file.write_at(0, new_data).unwrap();
        let read_data = file.read(0, new_data.len()).unwrap();
        assert_eq!(read_data, new_data);
    }

    #[test]
    fn test_memory_mapped_storage() -> Result<(), OptimizerError> {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = MemoryMappedConfig {
            base_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let mut storage = MemoryMappedStateStorage::new(config).unwrap();

        // Test storing and loading data
        let test_data = HashMap::from([
            ("lr".to_string(), 0.01f32),
            ("momentum".to_string(), 0.9f32),
        ]);

        storage.store("optimizer_params", &test_data).unwrap();
        let loaded_data: HashMap<String, f32> = storage.load("optimizer_params").unwrap().unwrap();

        assert_eq!(test_data, loaded_data);

        // Test updating
        let updated_data = HashMap::from([
            ("lr".to_string(), 0.001f32),
            ("momentum".to_string(), 0.95f32),
        ]);

        storage.update("optimizer_params", &updated_data).unwrap();
        let loaded_updated: HashMap<String, f32> =
            storage.load("optimizer_params").unwrap().unwrap();

        assert_eq!(updated_data, loaded_updated);

        // Test statistics
        let stats = storage.statistics();
        assert_eq!(stats.total_entries, 1);
        assert!(stats.total_size > 0);
        Ok(())
    }

    #[derive(Debug)]
    struct MockOptimizer {
        lr: f32,
        momentum: f32,
    }

    impl MemoryMappedSupport for MockOptimizer {
        fn save_to_mmap(
            &self,
            storage: &mut MemoryMappedStateStorage,
            prefix: &str,
        ) -> Result<(), OptimizerError> {
            storage.store(&format!("{}_lr", prefix), &self.lr)?;
            storage.store(&format!("{}_momentum", prefix), &self.momentum)?;
            Ok(())
        }

        fn load_from_mmap(
            &mut self,
            storage: &mut MemoryMappedStateStorage,
            prefix: &str,
        ) -> Result<(), OptimizerError> {
            if let Some(lr) = storage.load(&format!("{}_lr", prefix))? {
                self.lr = lr;
            }
            if let Some(momentum) = storage.load(&format!("{}_momentum", prefix))? {
                self.momentum = momentum;
            }
            Ok(())
        }

        fn mmap_state_keys(&self, prefix: &str) -> Vec<String> {
            vec![format!("{}_lr", prefix), format!("{}_momentum", prefix)]
        }
    }

    #[test]
    fn test_memory_mapped_optimizer() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = MemoryMappedConfig {
            base_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let optimizer = MockOptimizer {
            lr: 0.01,
            momentum: 0.9,
        };
        let mut mmap_optimizer =
            MemoryMappedOptimizer::new(optimizer, config, "test_opt".to_string()).unwrap();

        // Save state
        mmap_optimizer.save_state().unwrap();

        // Modify optimizer
        mmap_optimizer.inner_mut().lr = 0.001;
        mmap_optimizer.inner_mut().momentum = 0.95;

        // Load state (should restore original values)
        mmap_optimizer.load_state().unwrap();

        assert_eq!(mmap_optimizer.inner().lr, 0.01);
        assert_eq!(mmap_optimizer.inner().momentum, 0.9);
    }
}
