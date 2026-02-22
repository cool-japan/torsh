//! Store trait definition

use crate::TorshResult;
use async_trait::async_trait;
use std::time::Duration;

/// Main store trait for distributed coordination
#[async_trait]
pub trait Store: Send + Sync {
    /// Set a key-value pair
    async fn set(&self, key: &str, value: &[u8]) -> TorshResult<()>;

    /// Get a value by key
    async fn get(&self, key: &str) -> TorshResult<Option<Vec<u8>>>;

    /// Wait for a key to become available
    async fn wait(&self, keys: &[String]) -> TorshResult<()>;

    /// Delete a key
    async fn delete(&self, key: &str) -> TorshResult<()>;

    /// Get the number of keys in the store
    async fn num_keys(&self) -> TorshResult<usize>;

    /// Check if a key exists
    async fn contains(&self, key: &str) -> TorshResult<bool>;

    /// Set a key with expiration
    async fn set_with_expiry(&self, key: &str, value: &[u8], ttl: Duration) -> TorshResult<()>;

    /// Atomic compare and swap
    async fn compare_and_swap(
        &self,
        key: &str,
        expected: Option<&[u8]>,
        value: &[u8],
    ) -> TorshResult<bool>;

    /// Add to a numeric value (atomic)
    async fn add(&self, key: &str, value: i64) -> TorshResult<i64>;
}
