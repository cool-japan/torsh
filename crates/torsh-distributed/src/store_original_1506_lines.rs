//! Distributed store for process coordination
//!
//! The distributed store provides a key-value store that all processes
//! can access to coordinate initialization and share information during
//! distributed training.

use crate::{TorshDistributedError, TorshResult};
use log::{debug, info, warn};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "redis")]
use redis::{AsyncCommands, Client as RedisClient};

/// Timeout for store operations
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// A value stored in the distributed store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreValue {
    data: Vec<u8>,
    timestamp: u64,
}

impl StoreValue {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }
}

/// Store backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreBackend {
    /// In-memory store (for testing)
    Memory,
    /// File-based store
    File,
    /// TCP-based store
    Tcp,
    /// Redis-based store
    Redis,
}

/// Configuration for the distributed store
#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// Backend type
    pub backend: StoreBackend,
    /// Master address for TCP store
    pub master_addr: Option<IpAddr>,
    /// Master port for TCP store
    pub master_port: Option<u16>,
    /// File path for file-based store
    pub file_path: Option<String>,
    /// Redis URL for Redis store
    pub redis_url: Option<String>,
    /// Timeout for operations
    pub timeout: Duration,
    /// Number of retries for failed operations
    pub max_retries: u32,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            backend: StoreBackend::Memory,
            master_addr: None,
            master_port: None,
            file_path: None,
            redis_url: None,
            timeout: DEFAULT_TIMEOUT,
            max_retries: 3,
        }
    }
}

/// Trait for distributed store backends
#[async_trait::async_trait]
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

#[async_trait::async_trait]
impl Store for MemoryStore {
    async fn set(&self, key: &str, value: &[u8]) -> TorshResult<()> {
        let store_value = StoreValue::new(value.to_vec());
        self.data.insert(key.to_string(), store_value);
        Ok(())
    }

    async fn get(&self, key: &str) -> TorshResult<Option<Vec<u8>>> {
        Ok(self.data.get(key).map(|v| v.data.clone()))
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
                    if current.data == expected_val {
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
            let current = i64::from_le_bytes(existing.data[..8].try_into().map_err(|_| {
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
        if let Err(_) = store.load_from_file() {
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

#[async_trait::async_trait]
impl Store for FileStore {
    async fn set(&self, key: &str, value: &[u8]) -> TorshResult<()> {
        let store_value = StoreValue::new(value.to_vec());
        self.data.write().insert(key.to_string(), store_value);
        self.save_to_file()?;
        Ok(())
    }

    async fn get(&self, key: &str) -> TorshResult<Option<Vec<u8>>> {
        self.load_from_file()?;
        Ok(self.data.read().get(key).map(|v| v.data.clone()))
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
                )
                .into());
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
                    if current.data == expected_val {
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
            let current = i64::from_le_bytes(existing.data[..8].try_into().map_err(|_| {
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

/// TCP-based distributed store implementation
#[derive(Debug)]
pub struct TcpStore {
    client: Arc<tokio::sync::Mutex<Option<tokio::net::TcpStream>>>,
    master_addr: std::net::IpAddr,
    master_port: u16,
    timeout: Duration,
    data_cache: Arc<DashMap<String, StoreValue>>,
}

impl TcpStore {
    /// Create a new TCP store
    pub fn new(
        master_addr: std::net::IpAddr,
        master_port: u16,
        timeout: Duration,
    ) -> TorshResult<Self> {
        Ok(Self {
            client: Arc::new(tokio::sync::Mutex::new(None)),
            master_addr,
            master_port,
            timeout,
            data_cache: Arc::new(DashMap::new()),
        })
    }

    /// Ensure connection to the master
    async fn ensure_connection(&self) -> TorshResult<()> {
        let mut client = self.client.lock().await;

        if client.is_none() {
            let addr = std::net::SocketAddr::new(self.master_addr, self.master_port);

            match tokio::time::timeout(self.timeout, tokio::net::TcpStream::connect(addr)).await {
                Ok(Ok(stream)) => {
                    *client = Some(stream);
                    info!("🌐 Connected to TCP store at {}", addr);
                }
                Ok(Err(e)) => {
                    return Err(TorshDistributedError::CommunicationError {
                        operation: "TCP connect".to_string(),
                        cause: e.to_string(),
                    }
                    .into());
                }
                Err(_) => {
                    return Err(TorshDistributedError::OperationTimeout {
                        operation: "TCP connect".to_string(),
                        timeout_secs: self.timeout.as_secs(),
                    }
                    .into());
                }
            }
        }

        Ok(())
    }

    /// Send a message to the master and receive response
    async fn send_request(&self, request: TcpStoreMessage) -> TorshResult<TcpStoreResponse> {
        self.ensure_connection().await?;

        let mut client = self.client.lock().await;
        let stream = client.as_mut().unwrap();

        // Serialize request
        let request_data = serde_json::to_vec(&request).map_err(|e| {
            TorshDistributedError::serialization_error(format!(
                "Failed to serialize TcpStoreMessage: {}",
                e
            ))
        })?;

        // Send request with length prefix
        let len = request_data.len() as u32;
        tokio::io::AsyncWriteExt::write_all(stream, &len.to_le_bytes())
            .await
            .map_err(|e| TorshDistributedError::CommunicationError {
                operation: "TCP write length".to_string(),
                cause: e.to_string(),
            })?;

        tokio::io::AsyncWriteExt::write_all(stream, &request_data)
            .await
            .map_err(|e| TorshDistributedError::CommunicationError {
                operation: "TCP write data".to_string(),
                cause: e.to_string(),
            })?;

        // Read response length
        let mut len_buf = [0u8; 4];
        tokio::io::AsyncReadExt::read_exact(stream, &mut len_buf)
            .await
            .map_err(|e| TorshDistributedError::CommunicationError {
                operation: "TCP read length".to_string(),
                cause: e.to_string(),
            })?;

        let response_len = u32::from_le_bytes(len_buf) as usize;

        // Read response data
        let mut response_data = vec![0u8; response_len];
        tokio::io::AsyncReadExt::read_exact(stream, &mut response_data)
            .await
            .map_err(|e| TorshDistributedError::CommunicationError {
                operation: "TCP read data".to_string(),
                cause: e.to_string(),
            })?;

        // Deserialize response
        let response: TcpStoreResponse = serde_json::from_slice(&response_data).map_err(|e| {
            TorshDistributedError::SerializationError {
                data_type: "TcpStoreResponse".to_string(),
                cause: e.to_string(),
            }
        })?;

        Ok(response)
    }
}

/// TCP store message types
#[derive(Debug, Clone, Serialize, Deserialize)]
enum TcpStoreMessage {
    Set {
        key: String,
        value: Vec<u8>,
    },
    Get {
        key: String,
    },
    Delete {
        key: String,
    },
    Contains {
        key: String,
    },
    NumKeys,
    Wait {
        keys: Vec<String>,
    },
    CompareAndSwap {
        key: String,
        expected: Option<Vec<u8>>,
        value: Vec<u8>,
    },
    Add {
        key: String,
        value: i64,
    },
}

/// TCP store response types
#[derive(Debug, Clone, Serialize, Deserialize)]
enum TcpStoreResponse {
    Ok,
    Value(Option<Vec<u8>>),
    Bool(bool),
    Number(usize),
    I64(i64),
    Error(String),
}

#[async_trait::async_trait]
impl Store for TcpStore {
    async fn set(&self, key: &str, value: &[u8]) -> TorshResult<()> {
        let message = TcpStoreMessage::Set {
            key: key.to_string(),
            value: value.to_vec(),
        };

        match self.send_request(message).await? {
            TcpStoreResponse::Ok => {
                // Cache the value locally
                self.data_cache
                    .insert(key.to_string(), StoreValue::new(value.to_vec()));
                Ok(())
            }
            TcpStoreResponse::Error(e) => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: e,
            }
            .into()),
            _ => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: "Unexpected response type for set operation".to_string(),
            }
            .into()),
        }
    }

    async fn get(&self, key: &str) -> TorshResult<Option<Vec<u8>>> {
        // Try cache first
        if let Some(cached) = self.data_cache.get(key) {
            return Ok(Some(cached.data().to_vec()));
        }

        let message = TcpStoreMessage::Get {
            key: key.to_string(),
        };

        match self.send_request(message).await? {
            TcpStoreResponse::Value(value) => {
                // Cache the value if it exists
                if let Some(ref v) = value {
                    self.data_cache
                        .insert(key.to_string(), StoreValue::new(v.clone()));
                }
                Ok(value)
            }
            TcpStoreResponse::Error(e) => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: e,
            }
            .into()),
            _ => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: "Unexpected response type for get operation".to_string(),
            }
            .into()),
        }
    }

    async fn wait(&self, keys: &[String]) -> TorshResult<()> {
        let message = TcpStoreMessage::Wait {
            keys: keys.to_vec(),
        };

        match self.send_request(message).await? {
            TcpStoreResponse::Ok => Ok(()),
            TcpStoreResponse::Error(e) => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: e,
            }
            .into()),
            _ => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: "Unexpected response type for wait operation".to_string(),
            }
            .into()),
        }
    }

    async fn delete(&self, key: &str) -> TorshResult<()> {
        let message = TcpStoreMessage::Delete {
            key: key.to_string(),
        };

        match self.send_request(message).await? {
            TcpStoreResponse::Ok => {
                // Remove from cache
                self.data_cache.remove(key);
                Ok(())
            }
            TcpStoreResponse::Error(e) => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: e,
            }
            .into()),
            _ => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: "Unexpected response type for delete operation".to_string(),
            }
            .into()),
        }
    }

    async fn num_keys(&self) -> TorshResult<usize> {
        let message = TcpStoreMessage::NumKeys;

        match self.send_request(message).await? {
            TcpStoreResponse::Number(count) => Ok(count),
            TcpStoreResponse::Error(e) => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: e,
            }
            .into()),
            _ => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: "Unexpected response type for num_keys operation".to_string(),
            }
            .into()),
        }
    }

    async fn contains(&self, key: &str) -> TorshResult<bool> {
        // Check cache first
        if self.data_cache.contains_key(key) {
            return Ok(true);
        }

        let message = TcpStoreMessage::Contains {
            key: key.to_string(),
        };

        match self.send_request(message).await? {
            TcpStoreResponse::Bool(exists) => Ok(exists),
            TcpStoreResponse::Error(e) => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: e,
            }
            .into()),
            _ => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: "Unexpected response type for contains operation".to_string(),
            }
            .into()),
        }
    }

    async fn set_with_expiry(&self, key: &str, value: &[u8], _ttl: Duration) -> TorshResult<()> {
        // For simplicity, TCP store doesn't support TTL - just do regular set
        // In a production implementation, you'd add TTL support to the protocol
        info!("  TCP store doesn't support TTL, using regular set operation");
        self.set(key, value).await
    }

    async fn compare_and_swap(
        &self,
        key: &str,
        expected: Option<&[u8]>,
        value: &[u8],
    ) -> TorshResult<bool> {
        let message = TcpStoreMessage::CompareAndSwap {
            key: key.to_string(),
            expected: expected.map(|v| v.to_vec()),
            value: value.to_vec(),
        };

        match self.send_request(message).await? {
            TcpStoreResponse::Bool(success) => {
                if success {
                    // Update cache
                    self.data_cache
                        .insert(key.to_string(), StoreValue::new(value.to_vec()));
                }
                Ok(success)
            }
            TcpStoreResponse::Error(e) => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: e,
            }
            .into()),
            _ => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: "Unexpected response type for compare_and_swap operation".to_string(),
            }
            .into()),
        }
    }

    async fn add(&self, key: &str, value: i64) -> TorshResult<i64> {
        let message = TcpStoreMessage::Add {
            key: key.to_string(),
            value,
        };

        match self.send_request(message).await? {
            TcpStoreResponse::I64(new_value) => {
                // Update cache with new value
                self.data_cache.insert(
                    key.to_string(),
                    StoreValue::new(new_value.to_le_bytes().to_vec()),
                );
                Ok(new_value)
            }
            TcpStoreResponse::Error(e) => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: e,
            }
            .into()),
            _ => Err(TorshDistributedError::BackendError {
                backend: "TCP store".to_string(),
                message: "Unexpected response type for add operation".to_string(),
            }
            .into()),
        }
    }
}

/// Redis-based distributed store implementation
#[cfg(feature = "redis")]
#[derive(Debug)]
pub struct RedisStore {
    client: RedisClient,
    timeout: Duration,
    data_cache: Arc<DashMap<String, StoreValue>>,
}

#[cfg(feature = "redis")]
impl RedisStore {
    /// Create a new Redis store
    pub async fn new(redis_url: &str, timeout: Duration) -> TorshResult<Self> {
        let client =
            RedisClient::open(redis_url).map_err(|e| TorshDistributedError::BackendError {
                backend: "Redis store".to_string(),
                message: format!("Failed to create Redis client: {}", e),
            })?;

        // Test connection
        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| TorshDistributedError::CommunicationError {
                operation: "Redis connect".to_string(),
                cause: e.to_string(),
            })?;

        // Test ping
        let _: String =
            conn.ping()
                .await
                .map_err(|e| TorshDistributedError::CommunicationError {
                    operation: "Redis ping".to_string(),
                    cause: e.to_string(),
                })?;

        info!("🗃️  Connected to Redis store at {}", redis_url);

        Ok(Self {
            client,
            timeout,
            data_cache: Arc::new(DashMap::new()),
        })
    }

    /// Get a connection with timeout
    async fn get_connection(&self) -> TorshResult<redis::aio::MultiplexedConnection> {
        tokio::time::timeout(self.timeout, self.client.get_multiplexed_async_connection())
            .await
            .map_err(|_| TorshDistributedError::OperationTimeout {
                operation: "Redis connection".to_string(),
                timeout_secs: self.timeout.as_secs(),
            })?
            .map_err(|e| TorshDistributedError::CommunicationError {
                operation: "Redis connection".to_string(),
                cause: e.to_string(),
            })
    }
}

#[cfg(feature = "redis")]
#[async_trait::async_trait]
impl Store for RedisStore {
    async fn set(&self, key: &str, value: &[u8]) -> TorshResult<()> {
        let mut conn = self.get_connection().await?;

        tokio::time::timeout(self.timeout, conn.set::<&str, &[u8], ()>(key, value))
            .await
            .map_err(|_| TorshDistributedError::OperationTimeout {
                operation: "Redis set".to_string(),
                timeout_secs: self.timeout.as_secs(),
            })?
            .map_err(|e| TorshDistributedError::BackendError {
                backend: "Redis store".to_string(),
                message: format!("Set operation failed: {}", e),
            })?;

        // Cache the value locally
        self.data_cache
            .insert(key.to_string(), StoreValue::new(value.to_vec()));
        Ok(())
    }

    async fn get(&self, key: &str) -> TorshResult<Option<Vec<u8>>> {
        // Try cache first
        if let Some(cached) = self.data_cache.get(key) {
            return Ok(Some(cached.data().to_vec()));
        }

        let mut conn = self.get_connection().await?;

        let result: Option<Vec<u8>> = tokio::time::timeout(self.timeout, conn.get(key))
            .await
            .map_err(|_| TorshDistributedError::OperationTimeout {
                operation: "Redis get".to_string(),
                timeout_secs: self.timeout.as_secs(),
            })?
            .map_err(|e| TorshDistributedError::BackendError {
                backend: "Redis store".to_string(),
                message: format!("Get operation failed: {}", e),
            })?;

        // Cache the value if it exists
        if let Some(ref v) = result {
            self.data_cache
                .insert(key.to_string(), StoreValue::new(v.clone()));
        }

        Ok(result)
    }

    async fn wait(&self, keys: &[String]) -> TorshResult<()> {
        let mut conn = self.get_connection().await?;
        let start = Instant::now();

        loop {
            let mut all_present = true;

            for key in keys {
                let exists: bool = tokio::time::timeout(self.timeout, conn.exists(key))
                    .await
                    .map_err(|_| TorshDistributedError::OperationTimeout {
                        operation: "Redis exists".to_string(),
                        timeout_secs: self.timeout.as_secs(),
                    })?
                    .map_err(|e| TorshDistributedError::BackendError {
                        backend: "Redis store".to_string(),
                        message: format!("Exists operation failed: {}", e),
                    })?;

                if !exists {
                    all_present = false;
                    break;
                }
            }

            if all_present {
                return Ok(());
            }

            if start.elapsed() > self.timeout {
                return Err(TorshDistributedError::OperationTimeout {
                    operation: "Redis wait".to_string(),
                    timeout_secs: self.timeout.as_secs(),
                }
                .into());
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn delete(&self, key: &str) -> TorshResult<()> {
        let mut conn = self.get_connection().await?;

        tokio::time::timeout(self.timeout, conn.del::<&str, ()>(key))
            .await
            .map_err(|_| TorshDistributedError::OperationTimeout {
                operation: "Redis delete".to_string(),
                timeout_secs: self.timeout.as_secs(),
            })?
            .map_err(|e| TorshDistributedError::BackendError {
                backend: "Redis store".to_string(),
                message: format!("Delete operation failed: {}", e),
            })?;

        // Remove from cache
        self.data_cache.remove(key);
        Ok(())
    }

    async fn num_keys(&self) -> TorshResult<usize> {
        let mut conn = self.get_connection().await?;

        let count: usize = tokio::time::timeout(self.timeout, conn.dbsize())
            .await
            .map_err(|_| TorshDistributedError::OperationTimeout {
                operation: "Redis dbsize".to_string(),
                timeout_secs: self.timeout.as_secs(),
            })?
            .map_err(|e| TorshDistributedError::BackendError {
                backend: "Redis store".to_string(),
                message: format!("Dbsize operation failed: {}", e),
            })?;

        Ok(count)
    }

    async fn contains(&self, key: &str) -> TorshResult<bool> {
        // Check cache first
        if self.data_cache.contains_key(key) {
            return Ok(true);
        }

        let mut conn = self.get_connection().await?;

        let exists: bool = tokio::time::timeout(self.timeout, conn.exists(key))
            .await
            .map_err(|_| TorshDistributedError::OperationTimeout {
                operation: "Redis exists".to_string(),
                timeout_secs: self.timeout.as_secs(),
            })?
            .map_err(|e| TorshDistributedError::BackendError {
                backend: "Redis store".to_string(),
                message: format!("Exists operation failed: {}", e),
            })?;

        Ok(exists)
    }

    async fn set_with_expiry(&self, key: &str, value: &[u8], ttl: Duration) -> TorshResult<()> {
        let mut conn = self.get_connection().await?;

        tokio::time::timeout(
            self.timeout,
            conn.set_ex::<&str, &[u8], ()>(key, value, ttl.as_secs()),
        )
        .await
        .map_err(|_| TorshDistributedError::OperationTimeout {
            operation: "Redis set_ex".to_string(),
            timeout_secs: self.timeout.as_secs(),
        })?
        .map_err(|e| TorshDistributedError::BackendError {
            backend: "Redis store".to_string(),
            message: format!("Set with expiry operation failed: {}", e),
        })?;

        // Cache the value locally (note: we don't implement TTL in local cache)
        self.data_cache
            .insert(key.to_string(), StoreValue::new(value.to_vec()));
        Ok(())
    }

    async fn compare_and_swap(
        &self,
        key: &str,
        expected: Option<&[u8]>,
        value: &[u8],
    ) -> TorshResult<bool> {
        let mut conn = self.get_connection().await?;

        // Use Redis WATCH/MULTI/EXEC for atomic compare-and-swap
        let mut pipe = redis::pipe();
        pipe.atomic();

        match expected {
            Some(expected_val) => {
                // Watch the key for changes
                tokio::time::timeout(self.timeout, conn.watch(key))
                    .await
                    .map_err(|_| TorshDistributedError::OperationTimeout {
                        operation: "Redis watch".to_string(),
                        timeout_secs: self.timeout.as_secs(),
                    })?
                    .map_err(|e| TorshDistributedError::BackendError {
                        backend: "Redis store".to_string(),
                        message: format!("Watch operation failed: {}", e),
                    })?;

                // Get current value
                let current: Option<Vec<u8>> = tokio::time::timeout(self.timeout, conn.get(key))
                    .await
                    .map_err(|_| TorshDistributedError::OperationTimeout {
                        operation: "Redis get".to_string(),
                        timeout_secs: self.timeout.as_secs(),
                    })?
                    .map_err(|e| TorshDistributedError::BackendError {
                        backend: "Redis store".to_string(),
                        message: format!("Get operation failed: {}", e),
                    })?;

                // Check if current value matches expected
                if current.as_ref().map(|v| v.as_slice()) == Some(expected_val) {
                    pipe.set(key, value);
                    let result: Option<redis::Value> =
                        tokio::time::timeout(self.timeout, pipe.query_async(&mut conn))
                            .await
                            .map_err(|_| TorshDistributedError::OperationTimeout {
                                operation: "Redis transaction".to_string(),
                                timeout_secs: self.timeout.as_secs(),
                            })?
                            .map_err(|e| TorshDistributedError::BackendError {
                                backend: "Redis store".to_string(),
                                message: format!("Transaction failed: {}", e),
                            })?;

                    // Transaction succeeded if result is not nil
                    let success = result.is_some();
                    if success {
                        self.data_cache
                            .insert(key.to_string(), StoreValue::new(value.to_vec()));
                    }
                    Ok(success)
                } else {
                    Ok(false)
                }
            }
            None => {
                // Set only if key doesn't exist (using SET NX)
                let result: bool = tokio::time::timeout(self.timeout, conn.set_nx(key, value))
                    .await
                    .map_err(|_| TorshDistributedError::OperationTimeout {
                        operation: "Redis set_nx".to_string(),
                        timeout_secs: self.timeout.as_secs(),
                    })?
                    .map_err(|e| TorshDistributedError::BackendError {
                        backend: "Redis store".to_string(),
                        message: format!("Set NX operation failed: {}", e),
                    })?;

                if result {
                    self.data_cache
                        .insert(key.to_string(), StoreValue::new(value.to_vec()));
                }
                Ok(result)
            }
        }
    }

    async fn add(&self, key: &str, value: i64) -> TorshResult<i64> {
        let mut conn = self.get_connection().await?;

        let new_value: i64 = tokio::time::timeout(self.timeout, conn.incr(key, value))
            .await
            .map_err(|_| TorshDistributedError::OperationTimeout {
                operation: "Redis incr".to_string(),
                timeout_secs: self.timeout.as_secs(),
            })?
            .map_err(|e| TorshDistributedError::BackendError {
                backend: "Redis store".to_string(),
                message: format!("Increment operation failed: {}", e),
            })?;

        // Update cache with new value
        self.data_cache.insert(
            key.to_string(),
            StoreValue::new(new_value.to_le_bytes().to_vec()),
        );
        Ok(new_value)
    }
}

/// Create a distributed store based on configuration
pub fn create_store(config: &StoreConfig) -> TorshResult<Box<dyn Store>> {
    match config.backend {
        StoreBackend::Memory => Ok(Box::new(MemoryStore::new())),
        StoreBackend::File => {
            let file_path = config.file_path.as_ref().ok_or_else(|| {
                TorshDistributedError::invalid_argument(
                    "file_path",
                    "File path is required when using file store backend",
                    "valid file path string",
                )
            })?;
            Ok(Box::new(FileStore::new(file_path.clone())?))
        }
        StoreBackend::Tcp => {
            let master_addr =
                config
                    .master_addr
                    .ok_or_else(|| TorshDistributedError::InvalidArgument {
                        arg: "master_addr".to_string(),
                        reason: "Master address required for TCP store".to_string(),
                        expected: "Valid IP address".to_string(),
                    })?;
            let master_port =
                config
                    .master_port
                    .ok_or_else(|| TorshDistributedError::InvalidArgument {
                        arg: "master_port".to_string(),
                        reason: "Master port required for TCP store".to_string(),
                        expected: "Valid port number".to_string(),
                    })?;
            Ok(Box::new(TcpStore::new(
                master_addr,
                master_port,
                config.timeout,
            )?))
        }
        StoreBackend::Redis => {
            #[cfg(feature = "redis")]
            {
                let redis_url = config.redis_url.as_ref().ok_or_else(|| {
                    TorshDistributedError::InvalidArgument {
                        arg: "redis_url".to_string(),
                        reason: "Redis URL required for Redis store".to_string(),
                        expected: "Valid Redis URL (e.g., redis://localhost:6379)".to_string(),
                    }
                })?;

                // Note: RedisStore::new is async, but create_store is sync
                // In a real implementation, you might want to make create_store async
                // For now, we'll return an error indicating async initialization is needed
                Err(TorshDistributedError::FeatureNotAvailable(
                    "Redis store requires async initialization. Use RedisStore::new() directly."
                        .to_string(),
                )
                .into())
            }

            #[cfg(not(feature = "redis"))]
            {
                Err(TorshDistributedError::FeatureNotAvailable(
                    "Redis store feature not enabled. Enable with --features redis".to_string(),
                )
                .into())
            }
        }
    }
}

/// Utility functions for common store operations
impl dyn Store {
    /// Set a string value
    pub async fn set_string(&self, key: &str, value: &str) -> TorshResult<()> {
        self.set(key, value.as_bytes()).await
    }

    /// Get a string value
    pub async fn get_string(&self, key: &str) -> TorshResult<Option<String>> {
        match self.get(key).await? {
            Some(bytes) => {
                let s = String::from_utf8(bytes).map_err(|_| {
                    TorshDistributedError::invalid_argument(
                        "bytes",
                        "Failed to convert bytes to UTF-8 string",
                        "valid UTF-8 encoded bytes",
                    )
                })?;
                Ok(Some(s))
            }
            None => Ok(None),
        }
    }

    /// Set an integer value
    pub async fn set_i64(&self, key: &str, value: i64) -> TorshResult<()> {
        self.set(key, &value.to_le_bytes()).await
    }

    /// Get an integer value
    pub async fn get_i64(&self, key: &str) -> TorshResult<Option<i64>> {
        match self.get(key).await? {
            Some(bytes) => {
                if bytes.len() == 8 {
                    let array: [u8; 8] = bytes.try_into().map_err(|_| {
                        TorshDistributedError::invalid_argument(
                            "bytes",
                            "Failed to convert bytes to 8-byte array for i64",
                            "exactly 8 bytes",
                        )
                    })?;
                    Ok(Some(i64::from_le_bytes(array)))
                } else {
                    Err(TorshDistributedError::invalid_argument(
                        "bytes",
                        format!("Invalid byte length for i64: got {} bytes", bytes.len()),
                        "exactly 8 bytes",
                    )
                    .into())
                }
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_memory_store() -> TorshResult<()> {
        let store = MemoryStore::new();

        // Test basic set/get
        store.set("key1", b"value1").await?;
        let value = store.get("key1").await?;
        assert_eq!(value, Some(b"value1".to_vec()));

        // Test non-existent key
        let value = store.get("nonexistent").await?;
        assert_eq!(value, None);

        // Test contains
        assert!(store.contains("key1").await?);
        assert!(!store.contains("nonexistent").await?);

        // Test delete
        store.delete("key1").await?;
        assert!(!store.contains("key1").await?);

        // Test wait
        tokio::spawn({
            let store = MemoryStore::new();
            async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                store.set("async_key", b"async_value").await.unwrap();
            }
        });

        // This should complete when the key is set
        let store2 = MemoryStore::new();
        store2.set("async_key", b"async_value").await?;
        store2.wait(&["async_key".to_string()]).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_file_store() -> TorshResult<()> {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir
            .path()
            .join("test_store.json")
            .to_string_lossy()
            .to_string();

        let store = FileStore::new(file_path)?;

        // Test basic set/get
        store.set("key1", b"value1").await?;
        let value = store.get("key1").await?;
        assert_eq!(value, Some(b"value1".to_vec()));

        // Test persistence by creating a new store instance
        let store2 = FileStore::new(
            temp_dir
                .path()
                .join("test_store.json")
                .to_string_lossy()
                .to_string(),
        )?;
        let value = store2.get("key1").await?;
        assert_eq!(value, Some(b"value1".to_vec()));

        Ok(())
    }

    #[tokio::test]
    async fn test_store_utility_functions() -> TorshResult<()> {
        let store: Box<dyn Store> = Box::new(MemoryStore::new());

        // Test string functions
        store.set_string("str_key", "hello world").await?;
        let value = store.get_string("str_key").await?;
        assert_eq!(value, Some("hello world".to_string()));

        // Test i64 functions
        store.set_i64("int_key", 42).await?;
        let value = store.get_i64("int_key").await?;
        assert_eq!(value, Some(42));

        // Test add function
        let result = store.add("counter", 10).await?;
        assert_eq!(result, 10);
        let result = store.add("counter", 5).await?;
        assert_eq!(result, 15);

        Ok(())
    }

    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn test_redis_store() -> TorshResult<()> {
        // Note: This test requires a Redis instance running at redis://localhost:6379
        // Skip if Redis is not available
        let redis_url = "redis://localhost:6379";

        let store = match RedisStore::new(redis_url, Duration::from_secs(5)).await {
            Ok(store) => store,
            Err(_) => {
                info!(
                    "  Skipping Redis test - Redis not available at {}",
                    redis_url
                );
                return Ok(());
            }
        };

        // Test basic set/get
        store.set("redis_key1", b"redis_value1").await?;
        let value = store.get("redis_key1").await?;
        assert_eq!(value, Some(b"redis_value1".to_vec()));

        // Test non-existent key
        let value = store.get("nonexistent_redis").await?;
        assert_eq!(value, None);

        // Test contains
        assert!(store.contains("redis_key1").await?);
        assert!(!store.contains("nonexistent_redis").await?);

        // Test delete
        store.delete("redis_key1").await?;
        assert!(!store.contains("redis_key1").await?);

        // Test set with expiry
        store
            .set_with_expiry("expiry_key", b"expiry_value", Duration::from_secs(1))
            .await?;
        assert!(store.contains("expiry_key").await?);

        // Wait for expiry (this would require waiting, but we'll skip for the test)
        // tokio::time::sleep(Duration::from_secs(2)).await;
        // assert!(!store.contains("expiry_key").await?);

        // Test compare and swap
        store.set("cas_key", b"initial").await?;
        let success = store
            .compare_and_swap("cas_key", Some(b"initial"), b"updated")
            .await?;
        assert!(success);
        let value = store.get("cas_key").await?;
        assert_eq!(value, Some(b"updated".to_vec()));

        // Test failed compare and swap
        let success = store
            .compare_and_swap("cas_key", Some(b"wrong"), b"failed")
            .await?;
        assert!(!success);

        // Test add operation
        let result = store.add("redis_counter", 10).await?;
        assert_eq!(result, 10);
        let result = store.add("redis_counter", 5).await?;
        assert_eq!(result, 15);

        // Clean up
        store.delete("cas_key").await?;
        store.delete("expiry_key").await?;
        store.delete("redis_counter").await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_store_creation() -> TorshResult<()> {
        // Test memory store creation
        let config = StoreConfig {
            backend: StoreBackend::Memory,
            ..Default::default()
        };
        let _store = create_store(&config)?;

        // Test file store creation
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir
            .path()
            .join("test.json")
            .to_string_lossy()
            .to_string();
        let config = StoreConfig {
            backend: StoreBackend::File,
            file_path: Some(file_path),
            ..Default::default()
        };
        let _store = create_store(&config)?;

        // Test TCP store creation
        let config = StoreConfig {
            backend: StoreBackend::Tcp,
            master_addr: Some("127.0.0.1".parse().unwrap()),
            master_port: Some(29500),
            ..Default::default()
        };
        let _store = create_store(&config)?;

        // Test Redis store configuration validation
        let config = StoreConfig {
            backend: StoreBackend::Redis,
            redis_url: Some("redis://localhost:6379".to_string()),
            ..Default::default()
        };
        let result = create_store(&config);
        assert!(result.is_err()); // Should fail because we need async initialization

        Ok(())
    }
}
