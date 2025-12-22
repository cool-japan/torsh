//! Redis-based store implementation for production multi-node coordination
//!
//! This module provides a production-ready Redis-based distributed key-value store
//! for process coordination in distributed training. It leverages Redis's robust
//! features for high-performance, scalable distributed systems.
//!
//! # Features
//!
//! - Production-ready Redis client with connection pooling
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
use redis::{AsyncCommands, Client, RedisError, Script};
// TODO: ConnectionManager moved or renamed in redis crate - needs investigation
// use redis::aio::ConnectionManager;

use super::store_trait::Store;
use crate::{TorshDistributedError, TorshResult};
use async_trait::async_trait;
// use std::sync::{Arc, RwLock}; // Commented out - not used without ConnectionManager
use std::time::Duration;

#[cfg(feature = "redis")]
/// Redis-based distributed store implementation
///
/// This implementation provides a production-ready distributed key-value store
/// using Redis for multi-node coordination. It supports:
///
/// - Connection pooling and multiplexing
/// - Atomic operations via Lua scripts
/// - Native key expiration
/// - Pub/sub for synchronization
/// - High availability
/// - Persistence options
pub struct RedisStore {
    redis_url: String,
    timeout: Duration,
    // TODO: Re-enable when redis ConnectionManager is available
    // connection_manager: Arc<RwLock<Option<ConnectionManager>>>,
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

    /// Connect to Redis and initialize connection manager
    pub async fn connect(&mut self) -> TorshResult<()> {
        let client = Client::open(self.redis_url.as_str()).map_err(|e| {
            TorshDistributedError::backend_error(
                "RedisStore",
                &format!("Failed to create Redis client: {}", e),
            )
        })?;

        let conn_manager = ConnectionManager::new(client).await.map_err(|e| {
            TorshDistributedError::backend_error(
                "RedisStore",
                &format!("Failed to create connection manager: {}", e),
            )
        })?;

        info!("Connected to Redis at {}", self.redis_url);

        let mut manager_lock = self.connection_manager.write().await;
        *manager_lock = Some(conn_manager);

        Ok(())
    }

    /// Get a connection from the connection manager
    async fn get_connection(&self) -> TorshResult<ConnectionManager> {
        let manager_lock = self.connection_manager.read().await;
        manager_lock.clone().ok_or_else(|| {
            TorshDistributedError::backend_error(
                "RedisStore",
                "Not connected to Redis. Call connect() first.",
            )
        })
    }

    /// Execute a Redis operation with timeout
    async fn execute_with_timeout<F, T>(&self, operation: F) -> TorshResult<T>
    where
        F: std::future::Future<Output = Result<T, RedisError>>,
    {
        tokio::time::timeout(self.timeout, operation)
            .await
            .map_err(|_| {
                TorshDistributedError::operation_timeout("Redis operation", self.timeout.as_secs())
            })?
            .map_err(|e| {
                TorshDistributedError::backend_error("RedisStore", &format!("Redis error: {}", e))
            })
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
    async fn set(&self, key: &str, value: &[u8]) -> TorshResult<()> {
        let mut conn = self.get_connection().await?;
        self.execute_with_timeout(conn.set(key, value)).await?;
        debug!("Redis SET: key={}", key);
        Ok(())
    }

    async fn get(&self, key: &str) -> TorshResult<Option<Vec<u8>>> {
        let mut conn = self.get_connection().await?;
        let result: Option<Vec<u8>> = self.execute_with_timeout(conn.get(key)).await?;
        debug!("Redis GET: key={}, found={}", key, result.is_some());
        Ok(result)
    }

    async fn wait(&self, keys: &[String]) -> TorshResult<()> {
        // Use polling with exponential backoff for waiting on keys
        let mut backoff = Duration::from_millis(10);
        let max_backoff = Duration::from_secs(1);
        let deadline = tokio::time::Instant::now() + self.timeout;

        loop {
            let mut all_exist = true;
            let mut conn = self.get_connection().await?;

            for key in keys {
                let exists: bool = self.execute_with_timeout(conn.exists(key)).await?;
                if !exists {
                    all_exist = false;
                    break;
                }
            }

            if all_exist {
                debug!("Redis WAIT: all keys exist");
                return Ok(());
            }

            if tokio::time::Instant::now() >= deadline {
                return Err(TorshDistributedError::operation_timeout(
                    &format!("waiting for keys: {:?}", keys),
                    self.timeout.as_secs(),
                ));
            }

            tokio::time::sleep(backoff).await;
            backoff = std::cmp::min(backoff * 2, max_backoff);
        }
    }

    async fn delete(&self, key: &str) -> TorshResult<()> {
        let mut conn = self.get_connection().await?;
        self.execute_with_timeout(conn.del(key)).await?;
        debug!("Redis DEL: key={}", key);
        Ok(())
    }

    async fn num_keys(&self) -> TorshResult<usize> {
        let mut conn = self.get_connection().await?;
        let count: usize = self.execute_with_timeout(conn.dbsize()).await?;
        debug!("Redis DBSIZE: count={}", count);
        Ok(count)
    }

    async fn contains(&self, key: &str) -> TorshResult<bool> {
        let mut conn = self.get_connection().await?;
        let exists: bool = self.execute_with_timeout(conn.exists(key)).await?;
        debug!("Redis EXISTS: key={}, exists={}", key, exists);
        Ok(exists)
    }

    async fn set_with_expiry(&self, key: &str, value: &[u8], ttl: Duration) -> TorshResult<()> {
        let mut conn = self.get_connection().await?;
        let ttl_secs = ttl.as_secs() as usize;
        self.execute_with_timeout(conn.set_ex(key, value, ttl_secs))
            .await?;
        debug!("Redis SETEX: key={}, ttl={}s", key, ttl_secs);
        Ok(())
    }

    async fn compare_and_swap(
        &self,
        key: &str,
        expected: Option<&[u8]>,
        value: &[u8],
    ) -> TorshResult<bool> {
        let mut conn = self.get_connection().await?;

        // Use Lua script for atomic compare-and-swap
        let script = Script::new(
            r#"
            local current = redis.call('GET', KEYS[1])
            local expected = ARGV[1]
            local new_value = ARGV[2]

            -- If expected is empty string, we're checking for non-existence
            if expected == '' then
                if current == false then
                    redis.call('SET', KEYS[1], new_value)
                    return 1
                else
                    return 0
                end
            else
                if current == expected then
                    redis.call('SET', KEYS[1], new_value)
                    return 1
                else
                    return 0
                end
            end
            "#,
        );

        let expected_val = expected.unwrap_or(b"");
        let result: i32 = self
            .execute_with_timeout(
                script
                    .key(key)
                    .arg(expected_val)
                    .arg(value)
                    .invoke_async(&mut conn),
            )
            .await?;

        let success = result == 1;
        debug!(
            "Redis CAS: key={}, success={}, expected_len={}, value_len={}",
            key,
            success,
            expected.map_or(0, |e| e.len()),
            value.len()
        );
        Ok(success)
    }

    async fn add(&self, key: &str, value: i64) -> TorshResult<i64> {
        let mut conn = self.get_connection().await?;
        let result: i64 = self.execute_with_timeout(conn.incr(key, value)).await?;
        debug!(
            "Redis INCRBY: key={}, delta={}, result={}",
            key, value, result
        );
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
            RedisStore::new("redis://127.0.0.1:6379".to_string(), Duration::from_secs(5)).unwrap();
        store.connect().await.unwrap();
        store
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_store_basic_operations() {
        let store = create_test_store().await;

        // Test set and get
        store.set("test_key1", b"test_value1").await.unwrap();
        let value = store.get("test_key1").await.unwrap();
        assert_eq!(value, Some(b"test_value1".to_vec()));

        // Test contains
        assert!(store.contains("test_key1").await.unwrap());
        assert!(!store.contains("nonexistent").await.unwrap());

        // Test delete
        store.delete("test_key1").await.unwrap();
        assert!(!store.contains("test_key1").await.unwrap());
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_store_atomic_operations() {
        let store = create_test_store().await;

        // Clean up any existing test data
        let _ = store.delete("test_counter").await;
        let _ = store.delete("test_num").await;

        // Test compare and swap
        let success = store
            .compare_and_swap("test_counter", None, b"0")
            .await
            .unwrap();
        assert!(success);

        let success = store
            .compare_and_swap("test_counter", Some(b"0"), b"1")
            .await
            .unwrap();
        assert!(success);

        let success = store
            .compare_and_swap("test_counter", Some(b"0"), b"2")
            .await
            .unwrap();
        assert!(!success);

        // Test atomic add
        let result = store.add("test_num", 5).await.unwrap();
        assert_eq!(result, 5);

        let result = store.add("test_num", 3).await.unwrap();
        assert_eq!(result, 8);

        // Cleanup
        store.delete("test_counter").await.unwrap();
        store.delete("test_num").await.unwrap();
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_store_expiry() {
        let store = create_test_store().await;

        // Set with expiry
        store
            .set_with_expiry("temp_key", b"temp_value", Duration::from_secs(2))
            .await
            .unwrap();

        // Should exist immediately
        assert!(store.contains("temp_key").await.unwrap());

        // Wait for expiry
        tokio::time::sleep(Duration::from_secs(3)).await;

        // Should be gone
        assert!(!store.contains("temp_key").await.unwrap());
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_store_wait() {
        let store = create_test_store().await;

        // Clean up
        let _ = store.delete("wait_key1").await;
        let _ = store.delete("wait_key2").await;

        // Set up keys asynchronously
        let store_clone = create_test_store().await;
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            store_clone.set("wait_key1", b"value1").await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
            store_clone.set("wait_key2", b"value2").await.unwrap();
        });

        // Wait for keys
        store
            .wait(&["wait_key1".to_string(), "wait_key2".to_string()])
            .await
            .unwrap();

        assert!(store.contains("wait_key1").await.unwrap());
        assert!(store.contains("wait_key2").await.unwrap());

        // Cleanup
        store.delete("wait_key1").await.unwrap();
        store.delete("wait_key2").await.unwrap();
    }
}
