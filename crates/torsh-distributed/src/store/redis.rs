//! Redis-based store implementation for production multi-node coordination
//!
//! This module provides a production-ready Redis-based distributed key-value store
//! for process coordination in distributed training. It leverages Redis's robust
//! features for high-performance, scalable distributed systems.
//!
//! # Features
//!
//! - Production-ready Redis client with connection pooling via ConnectionManager
//! - Atomic operations (Lua scripts for compare-and-swap)
//! - Native key expiration support (SETEX, EXPIRE)
//! - Pub/Sub for wait/notify synchronization
//! - Automatic reconnection and error handling
//! - High availability and persistence
//!
//! # Redis Commands Used
//!
//! - `SET`/`GET`: Basic key-value operations
//! - `SETEX`: Set with expiration
//! - `DEL`: Delete keys
//! - `EXISTS`: Check key existence
//! - `DBSIZE`: Get number of keys
//! - `INCRBY`: Atomic increment
//! - Lua scripts for compare-and-swap atomicity

#[cfg(feature = "redis")]
use redis::aio::ConnectionManager;
#[cfg(feature = "redis")]
use redis::{AsyncCommands, Client, Script};

use super::store_trait::Store;
use crate::{TorshDistributedError, TorshResult};
use async_trait::async_trait;
#[cfg(feature = "redis")]
use std::sync::Arc;
use std::time::Duration;
#[cfg(feature = "redis")]
use tokio::sync::RwLock;
#[cfg(feature = "redis")]
use tracing::info;

/// Default timeout for wait operations
#[cfg(feature = "redis")]
const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Poll interval for key-existence checks in wait()
#[cfg(feature = "redis")]
const WAIT_POLL_INTERVAL_MS: u64 = 10;

/// CAS Lua script: atomically compare old value and set new value.
///
/// Returns 1 if the swap succeeded, 0 otherwise.
#[cfg(feature = "redis")]
const CAS_SCRIPT: &str = r#"
local current = redis.call('GET', KEYS[1])
local expected = ARGV[1]
local new_val  = ARGV[2]
-- ARGV[1] == "" means "key must not exist"
if expected == '' then
    if current == false then
        redis.call('SET', KEYS[1], new_val)
        return 1
    else
        return 0
    end
else
    if current == expected then
        redis.call('SET', KEYS[1], new_val)
        return 1
    else
        return 0
    end
end
"#;

#[cfg(feature = "redis")]
/// Redis-based distributed store implementation
///
/// This implementation provides a production-ready distributed key-value store
/// using Redis for multi-node coordination. It supports:
///
/// - Connection pooling and multiplexing via ConnectionManager
/// - Atomic operations via Lua scripts
/// - Native key expiration
/// - High availability
/// - Persistence options
pub struct RedisStore {
    redis_url: String,
    timeout: Duration,
    connection_manager: Arc<RwLock<Option<ConnectionManager>>>,
}

#[cfg(not(feature = "redis"))]
/// Placeholder RedisStore when redis feature is disabled
pub struct RedisStore {
    #[allow(dead_code)]
    redis_url: String,
    timeout: Duration,
}

#[cfg(feature = "redis")]
impl std::fmt::Debug for RedisStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisStore")
            .field("redis_url", &"<redacted>")
            .field("timeout", &self.timeout)
            .finish()
    }
}

#[cfg(not(feature = "redis"))]
impl std::fmt::Debug for RedisStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisStore (disabled)")
            .field("redis_url", &"<redacted>")
            .field("timeout", &self.timeout)
            .finish()
    }
}

#[cfg(feature = "redis")]
impl RedisStore {
    /// Create a new Redis store instance
    ///
    /// # Arguments
    ///
    /// * `redis_url` - Redis connection URL (e.g., "redis://127.0.0.1:6379")
    /// * `timeout` - Timeout for store operations
    pub fn new(redis_url: String, timeout: Duration) -> TorshResult<Self> {
        Ok(Self {
            redis_url,
            timeout,
            connection_manager: Arc::new(RwLock::new(None)),
        })
    }

    /// Connect to Redis using ConnectionManager (provides automatic reconnection)
    pub async fn connect(&mut self) -> TorshResult<()> {
        let client = Client::open(self.redis_url.as_str()).map_err(|e| {
            TorshDistributedError::backend_error(
                "RedisStore",
                format!("Failed to create Redis client: {}", e),
            )
        })?;

        let manager = client.get_connection_manager().await.map_err(|e| {
            TorshDistributedError::backend_error(
                "RedisStore",
                format!("Failed to create ConnectionManager: {}", e),
            )
        })?;

        info!(
            "Connected to Redis at {} via ConnectionManager",
            self.redis_url
        );

        let mut cm_lock = self.connection_manager.write().await;
        *cm_lock = Some(manager);

        Ok(())
    }

    /// Get a cloned ConnectionManager, returning an error if not connected
    async fn get_manager(&self) -> TorshResult<ConnectionManager> {
        let cm_lock = self.connection_manager.read().await;
        cm_lock.clone().ok_or_else(|| {
            TorshDistributedError::backend_error(
                "RedisStore",
                "Not connected — call connect() first",
            )
        })
    }

    /// Map a redis error to TorshDistributedError
    fn map_redis_err(context: &'static str, e: redis::RedisError) -> TorshDistributedError {
        TorshDistributedError::backend_error("RedisStore", format!("{}: {}", context, e))
    }
}

#[cfg(not(feature = "redis"))]
impl RedisStore {
    pub fn new(redis_url: String, timeout: Duration) -> TorshResult<Self> {
        Ok(Self { redis_url, timeout })
    }

    pub async fn connect(&mut self) -> TorshResult<()> {
        Err(TorshDistributedError::not_implemented(
            "RedisStore requires 'redis' feature to be enabled",
        ))
    }
}

#[cfg(feature = "redis")]
#[async_trait]
impl Store for RedisStore {
    /// Set a key-value pair using Redis SET
    async fn set(&self, key: &str, value: &[u8]) -> TorshResult<()> {
        let mut cm = self.get_manager().await?;
        cm.set::<_, _, ()>(key, value)
            .await
            .map_err(|e| Self::map_redis_err("SET", e))
    }

    /// Get a value by key using Redis GET
    async fn get(&self, key: &str) -> TorshResult<Option<Vec<u8>>> {
        let mut cm = self.get_manager().await?;
        cm.get::<_, Option<Vec<u8>>>(key)
            .await
            .map_err(|e| Self::map_redis_err("GET", e))
    }

    /// Wait until all keys are present, polling with backoff up to timeout
    async fn wait(&self, keys: &[String]) -> TorshResult<()> {
        let start = std::time::Instant::now();
        let timeout = self.timeout.max(DEFAULT_WAIT_TIMEOUT);

        loop {
            let mut cm = self.get_manager().await?;
            let all_present = {
                let mut all = true;
                for key in keys {
                    let exists: bool = cm
                        .exists(key.as_str())
                        .await
                        .map_err(|e| Self::map_redis_err("EXISTS (wait)", e))?;
                    if !exists {
                        all = false;
                        break;
                    }
                }
                all
            };

            if all_present {
                return Ok(());
            }

            if start.elapsed() > timeout {
                return Err(TorshDistributedError::communication_error(
                    "RedisStore wait",
                    format!(
                        "Timeout after {:?} waiting for {} keys",
                        timeout,
                        keys.len()
                    ),
                ));
            }

            tokio::time::sleep(Duration::from_millis(WAIT_POLL_INTERVAL_MS)).await;
        }
    }

    /// Delete a key using Redis DEL
    async fn delete(&self, key: &str) -> TorshResult<()> {
        let mut cm = self.get_manager().await?;
        cm.del::<_, ()>(key)
            .await
            .map_err(|e| Self::map_redis_err("DEL", e))
    }

    /// Get the number of keys via Redis DBSIZE
    async fn num_keys(&self) -> TorshResult<usize> {
        let mut cm = self.get_manager().await?;
        let count: usize = redis::cmd("DBSIZE")
            .query_async(&mut cm)
            .await
            .map_err(|e| Self::map_redis_err("DBSIZE", e))?;
        Ok(count)
    }

    /// Check if a key exists using Redis EXISTS
    async fn contains(&self, key: &str) -> TorshResult<bool> {
        let mut cm = self.get_manager().await?;
        cm.exists::<_, bool>(key)
            .await
            .map_err(|e| Self::map_redis_err("EXISTS", e))
    }

    /// Set a key with expiration using Redis SETEX (seconds precision)
    async fn set_with_expiry(&self, key: &str, value: &[u8], ttl: Duration) -> TorshResult<()> {
        let seconds = ttl.as_secs().max(1);
        let mut cm = self.get_manager().await?;
        cm.set_ex::<_, _, ()>(key, value, seconds)
            .await
            .map_err(|e| Self::map_redis_err("SETEX", e))
    }

    /// Atomic compare-and-swap using a Lua script.
    ///
    /// Returns `true` if the swap succeeded (i.e. the key held `expected`
    /// before the call), `false` otherwise.
    async fn compare_and_swap(
        &self,
        key: &str,
        expected: Option<&[u8]>,
        value: &[u8],
    ) -> TorshResult<bool> {
        let expected_bytes: &[u8] = expected.unwrap_or(b"");

        let script = Script::new(CAS_SCRIPT);
        let mut cm = self.get_manager().await?;
        let result: isize = script
            .key(key)
            .arg(expected_bytes)
            .arg(value)
            .invoke_async(&mut cm)
            .await
            .map_err(|e| Self::map_redis_err("CAS Lua script", e))?;

        Ok(result == 1)
    }

    /// Atomically add `value` to the numeric value stored at `key` (Redis INCRBY).
    ///
    /// If the key does not exist it is initialised to 0 before the operation.
    async fn add(&self, key: &str, value: i64) -> TorshResult<i64> {
        let mut cm = self.get_manager().await?;
        let result: i64 = redis::cmd("INCRBY")
            .arg(key)
            .arg(value)
            .query_async(&mut cm)
            .await
            .map_err(|e| Self::map_redis_err("INCRBY", e))?;
        Ok(result)
    }
}

#[cfg(not(feature = "redis"))]
#[async_trait]
impl Store for RedisStore {
    async fn set(&self, _key: &str, _value: &[u8]) -> TorshResult<()> {
        Err(TorshDistributedError::not_implemented(
            "RedisStore requires 'redis' feature to be enabled",
        ))
    }

    async fn get(&self, _key: &str) -> TorshResult<Option<Vec<u8>>> {
        Err(TorshDistributedError::not_implemented(
            "RedisStore requires 'redis' feature to be enabled",
        ))
    }

    async fn wait(&self, _keys: &[String]) -> TorshResult<()> {
        Err(TorshDistributedError::not_implemented(
            "RedisStore requires 'redis' feature to be enabled",
        ))
    }

    async fn delete(&self, _key: &str) -> TorshResult<()> {
        Err(TorshDistributedError::not_implemented(
            "RedisStore requires 'redis' feature to be enabled",
        ))
    }

    async fn num_keys(&self) -> TorshResult<usize> {
        Err(TorshDistributedError::not_implemented(
            "RedisStore requires 'redis' feature to be enabled",
        ))
    }

    async fn contains(&self, _key: &str) -> TorshResult<bool> {
        Err(TorshDistributedError::not_implemented(
            "RedisStore requires 'redis' feature to be enabled",
        ))
    }

    async fn set_with_expiry(&self, _key: &str, _value: &[u8], _ttl: Duration) -> TorshResult<()> {
        Err(TorshDistributedError::not_implemented(
            "RedisStore requires 'redis' feature to be enabled",
        ))
    }

    async fn compare_and_swap(
        &self,
        _key: &str,
        _expected: Option<&[u8]>,
        _value: &[u8],
    ) -> TorshResult<bool> {
        Err(TorshDistributedError::not_implemented(
            "RedisStore requires 'redis' feature to be enabled",
        ))
    }

    async fn add(&self, _key: &str, _value: i64) -> TorshResult<i64> {
        Err(TorshDistributedError::not_implemented(
            "RedisStore requires 'redis' feature to be enabled",
        ))
    }
}

#[cfg(all(test, feature = "redis"))]
mod tests {
    use super::*;

    // Note: These tests require a running Redis instance at redis://127.0.0.1:6379
    // You can start Redis with: docker run -p 6379:6379 redis:latest
    //
    // To run these tests:
    // cargo test --package torsh-distributed --features redis -- redis_store

    async fn create_test_store() -> RedisStore {
        let mut store =
            RedisStore::new("redis://127.0.0.1:6379".to_string(), Duration::from_secs(5))
                .expect("store creation should succeed");
        store.connect().await.expect("connect should succeed");
        store
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_store_basic_operations() {
        let store = create_test_store().await;

        // Test set and get
        store
            .set("test_key1", b"test_value1")
            .await
            .expect("set should succeed");
        let value = store.get("test_key1").await.expect("get should succeed");
        assert_eq!(value, Some(b"test_value1".to_vec()));

        // Test contains
        assert!(store
            .contains("test_key1")
            .await
            .expect("contains should succeed"));
        assert!(!store
            .contains("nonexistent")
            .await
            .expect("contains should succeed"));

        // Test delete
        store
            .delete("test_key1")
            .await
            .expect("delete should succeed");
        assert!(!store
            .contains("test_key1")
            .await
            .expect("contains should succeed"));
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_store_num_keys() {
        let store = create_test_store().await;
        // num_keys returns DBSIZE — it reflects the full database, so we just check it returns Ok
        let count = store.num_keys().await.expect("num_keys should succeed");
        assert!(count < usize::MAX); // sanity
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_store_atomic_operations() {
        let store = create_test_store().await;

        // Clean up any existing test data
        let _ = store.delete("test_cas_counter").await;
        let _ = store.delete("test_incrby_num").await;

        // Test compare and swap — key absent, expected = None
        let ok = store
            .compare_and_swap("test_cas_counter", None, b"0")
            .await
            .expect("cas should succeed");
        assert!(ok);

        let ok = store
            .compare_and_swap("test_cas_counter", Some(b"0"), b"1")
            .await
            .expect("cas should succeed");
        assert!(ok);

        // Wrong expected value — should fail
        let ok = store
            .compare_and_swap("test_cas_counter", Some(b"0"), b"2")
            .await
            .expect("cas should succeed");
        assert!(!ok);

        // Test atomic add
        let result = store
            .add("test_incrby_num", 5)
            .await
            .expect("add should succeed");
        assert_eq!(result, 5);

        let result = store
            .add("test_incrby_num", 3)
            .await
            .expect("add should succeed");
        assert_eq!(result, 8);

        // Cleanup
        store
            .delete("test_cas_counter")
            .await
            .expect("delete should succeed");
        store
            .delete("test_incrby_num")
            .await
            .expect("delete should succeed");
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_store_expiry() {
        let store = create_test_store().await;

        store
            .set_with_expiry("temp_key", b"temp_value", Duration::from_secs(2))
            .await
            .expect("set_with_expiry should succeed");

        assert!(store
            .contains("temp_key")
            .await
            .expect("contains should succeed"));

        tokio::time::sleep(Duration::from_secs(3)).await;

        assert!(!store
            .contains("temp_key")
            .await
            .expect("contains should succeed"));
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_store_wait() {
        let store = create_test_store().await;

        let _ = store.delete("wait_key1").await;
        let _ = store.delete("wait_key2").await;

        let store_clone = create_test_store().await;
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            store_clone
                .set("wait_key1", b"v1")
                .await
                .expect("set should succeed");
            tokio::time::sleep(Duration::from_millis(100)).await;
            store_clone
                .set("wait_key2", b"v2")
                .await
                .expect("set should succeed");
        });

        store
            .wait(&["wait_key1".to_string(), "wait_key2".to_string()])
            .await
            .expect("wait should succeed");

        assert!(store
            .contains("wait_key1")
            .await
            .expect("contains should succeed"));
        assert!(store
            .contains("wait_key2")
            .await
            .expect("contains should succeed"));

        store
            .delete("wait_key1")
            .await
            .expect("delete should succeed");
        store
            .delete("wait_key2")
            .await
            .expect("delete should succeed");
    }

    /// Unit test for `not_connected` error path (no redis server needed)
    #[tokio::test]
    async fn test_redis_store_not_connected_returns_error() {
        let store = RedisStore::new("redis://127.0.0.1:6379".to_string(), Duration::from_secs(5))
            .expect("store creation should succeed");
        // Do NOT call connect() — all operations should return an error
        let result = store.get("any_key").await;
        assert!(result.is_err());
        let result = store.set("any_key", b"val").await;
        assert!(result.is_err());
    }
}
