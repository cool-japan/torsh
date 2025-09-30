//! In-memory store implementation for testing and single-node coordination

use super::{
    store_trait::Store,
    types::{StoreValue, DEFAULT_TIMEOUT},
};
use crate::{TorshDistributedError, TorshResult};
use async_trait::async_trait;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// In-memory store implementation
#[derive(Debug)]
pub struct MemoryStore {
    data: Arc<DashMap<String, StoreValue>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            data: Arc::new(DashMap::new()),
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Store for MemoryStore {
    async fn set(&self, key: &str, value: &[u8]) -> TorshResult<()> {
        let store_value = StoreValue::new(value.to_vec());
        self.data.insert(key.to_string(), store_value);
        Ok(())
    }

    async fn get(&self, key: &str) -> TorshResult<Option<Vec<u8>>> {
        Ok(self.data.get(key).map(|v| v.data().to_vec()))
    }

    async fn wait(&self, keys: &[String]) -> TorshResult<()> {
        let start = Instant::now();

        loop {
            let all_present = keys.iter().all(|key| self.data.contains_key(key));

            if all_present {
                return Ok(());
            }

            if start.elapsed() > DEFAULT_TIMEOUT {
                return Err(TorshDistributedError::communication_error(
                    "Store wait",
                    "Timeout waiting for keys",
                )
                .into());
            }

            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    async fn delete(&self, key: &str) -> TorshResult<()> {
        self.data.remove(key);
        Ok(())
    }

    async fn num_keys(&self) -> TorshResult<usize> {
        Ok(self.data.len())
    }

    async fn contains(&self, key: &str) -> TorshResult<bool> {
        Ok(self.data.contains_key(key))
    }

    async fn set_with_expiry(&self, key: &str, value: &[u8], _ttl: Duration) -> TorshResult<()> {
        // Memory store doesn't support TTL, just set normally
        self.set(key, value).await
    }

    async fn compare_and_swap(
        &self,
        key: &str,
        expected: Option<&[u8]>,
        value: &[u8],
    ) -> TorshResult<bool> {
        match expected {
            Some(expected_val) => {
                if let Some(current) = self.data.get(key) {
                    if current.data() == expected_val {
                        let store_value = StoreValue::new(value.to_vec());
                        self.data.insert(key.to_string(), store_value);
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            None => {
                // Expected value is None, so set only if key doesn't exist
                if self.data.contains_key(key) {
                    Ok(false)
                } else {
                    let store_value = StoreValue::new(value.to_vec());
                    self.data.insert(key.to_string(), store_value);
                    Ok(true)
                }
            }
        }
    }

    async fn add(&self, key: &str, value: i64) -> TorshResult<i64> {
        let new_value = if let Some(existing) = self.data.get(key) {
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
        self.data.insert(key.to_string(), store_value);
        Ok(new_value)
    }
}
