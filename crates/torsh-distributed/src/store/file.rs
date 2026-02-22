//! File-based store implementation for single-node multi-process coordination

use super::{
    store_trait::Store,
    types::{StoreValue, DEFAULT_TIMEOUT},
};
use crate::{TorshDistributedError, TorshResult};
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// File-based store implementation
#[derive(Debug)]
pub struct FileStore {
    file_path: String,
    data: Arc<RwLock<HashMap<String, StoreValue>>>,
}

impl FileStore {
    pub fn new(file_path: String) -> TorshResult<Self> {
        let store = Self {
            file_path,
            data: Arc::new(RwLock::new(HashMap::new())),
        };

        // Try to load existing data
        if store.load_from_file().is_err() {
            // If loading fails, start with empty store
        }

        Ok(store)
    }

    fn load_from_file(&self) -> TorshResult<()> {
        if std::path::Path::new(&self.file_path).exists() {
            let contents = std::fs::read_to_string(&self.file_path).map_err(|e| {
                TorshDistributedError::backend_error(
                    "FileStore",
                    format!("Failed to read store file: {}", e),
                )
            })?;

            let data: HashMap<String, StoreValue> =
                serde_json::from_str(&contents).map_err(|e| {
                    TorshDistributedError::backend_error(
                        "FileStore",
                        format!("Failed to parse store file: {}", e),
                    )
                })?;

            *self.data.write() = data;
        }
        Ok(())
    }

    fn save_to_file(&self) -> TorshResult<()> {
        let data = self.data.read();
        let contents = serde_json::to_string_pretty(&*data).map_err(|e| {
            TorshDistributedError::backend_error(
                "FileStore",
                format!("Failed to serialize store: {}", e),
            )
        })?;

        std::fs::write(&self.file_path, contents).map_err(|e| {
            TorshDistributedError::backend_error(
                "FileStore",
                format!("Failed to write store file: {}", e),
            )
        })?;

        Ok(())
    }
}

#[async_trait]
impl Store for FileStore {
    async fn set(&self, key: &str, value: &[u8]) -> TorshResult<()> {
        let store_value = StoreValue::new(value.to_vec());
        self.data.write().insert(key.to_string(), store_value);
        self.save_to_file()?;
        Ok(())
    }

    async fn get(&self, key: &str) -> TorshResult<Option<Vec<u8>>> {
        self.load_from_file()?;
        Ok(self.data.read().get(key).map(|v| v.data().to_vec()))
    }

    async fn wait(&self, keys: &[String]) -> TorshResult<()> {
        let start = Instant::now();

        loop {
            self.load_from_file()?;
            let all_present = {
                let data = self.data.read();
                keys.iter().all(|key| data.contains_key(key))
            }; // RwLockReadGuard is dropped here

            if all_present {
                return Ok(());
            }

            if start.elapsed() > DEFAULT_TIMEOUT {
                return Err(TorshDistributedError::communication_error(
                    "Store wait",
                    "Timeout waiting for keys",
                ));
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn delete(&self, key: &str) -> TorshResult<()> {
        self.data.write().remove(key);
        self.save_to_file()?;
        Ok(())
    }

    async fn num_keys(&self) -> TorshResult<usize> {
        self.load_from_file()?;
        Ok(self.data.read().len())
    }

    async fn contains(&self, key: &str) -> TorshResult<bool> {
        self.load_from_file()?;
        Ok(self.data.read().contains_key(key))
    }

    async fn set_with_expiry(&self, key: &str, value: &[u8], _ttl: Duration) -> TorshResult<()> {
        // File store doesn't support TTL, just set normally
        self.set(key, value).await
    }

    async fn compare_and_swap(
        &self,
        key: &str,
        expected: Option<&[u8]>,
        value: &[u8],
    ) -> TorshResult<bool> {
        self.load_from_file()?;
        let mut data = self.data.write();

        match expected {
            Some(expected_val) => {
                if let Some(current) = data.get(key) {
                    if current.data() == expected_val {
                        let store_value = StoreValue::new(value.to_vec());
                        data.insert(key.to_string(), store_value);
                        drop(data);
                        self.save_to_file()?;
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            None => {
                if data.contains_key(key) {
                    Ok(false)
                } else {
                    let store_value = StoreValue::new(value.to_vec());
                    data.insert(key.to_string(), store_value);
                    drop(data);
                    self.save_to_file()?;
                    Ok(true)
                }
            }
        }
    }

    async fn add(&self, key: &str, value: i64) -> TorshResult<i64> {
        self.load_from_file()?;
        let mut data = self.data.write();

        let new_value = if let Some(existing) = data.get(key) {
            let current = i64::from_le_bytes(existing.data()[..8].try_into().map_err(|_| {
                TorshDistributedError::invalid_argument(
                    "value",
                    "Failed to convert stored bytes to i64",
                    "8 bytes representing a valid i64 value",
                )
            })?);
            current + value
        } else {
            value
        };

        let store_value = StoreValue::new(new_value.to_le_bytes().to_vec());
        data.insert(key.to_string(), store_value);
        drop(data);
        self.save_to_file()?;
        Ok(new_value)
    }
}
