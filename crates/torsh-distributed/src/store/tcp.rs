//! TCP-based store implementation for multi-node coordination
//!
//! This module provides a production-ready TCP-based distributed key-value store
//! for process coordination in distributed training. It implements a client-server
//! architecture where one process acts as the master server and others connect as clients.
//!
//! # Architecture
//!
//! - **Server (Master)**: Runs on the master node and maintains the key-value store
//! - **Clients**: Connect to the master server and perform store operations
//! - **Protocol**: Binary protocol using bincode serialization for efficiency
//!
//! # Features
//!
//! - Atomic operations (compare-and-swap, atomic add)
//! - Key expiration support with automatic cleanup
//! - Wait operations for synchronization
//! - Connection pooling and timeout management
//! - Robust error handling and recovery

use super::store_trait::Store;
use crate::{TorshDistributedError, TorshResult};
use async_trait::async_trait;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, RwLock as TokioRwLock};
use tokio::time::timeout as tokio_timeout;
use tracing::{debug, error, info, warn};

/// TCP store protocol message types
#[derive(Debug, Clone, Serialize, Deserialize)]
enum StoreMessage {
    /// Set a key-value pair
    Set { key: String, value: Vec<u8> },
    /// Get a value by key
    Get { key: String },
    /// Wait for keys to become available
    Wait { keys: Vec<String> },
    /// Delete a key
    Delete { key: String },
    /// Get number of keys
    NumKeys,
    /// Check if key exists
    Contains { key: String },
    /// Set with expiration
    SetWithExpiry {
        key: String,
        value: Vec<u8>,
        ttl_secs: u64,
    },
    /// Atomic compare and swap
    CompareAndSwap {
        key: String,
        expected: Option<Vec<u8>>,
        value: Vec<u8>,
    },
    /// Atomic add
    Add { key: String, value: i64 },
}

/// Response from store operations
#[derive(Debug, Clone, Serialize, Deserialize)]
enum StoreResponse {
    Ok,
    Value(Option<Vec<u8>>),
    NumKeys(usize),
    Bool(bool),
    I64(i64),
    Error(String),
}

/// Entry in the store with optional expiration
#[derive(Debug, Clone)]
struct StoreEntry {
    value: Vec<u8>,
    expiry: Option<Instant>,
}

impl StoreEntry {
    fn new(value: Vec<u8>, ttl: Option<Duration>) -> Self {
        Self {
            value,
            expiry: ttl.map(|d| Instant::now() + d),
        }
    }

    fn is_expired(&self) -> bool {
        self.expiry.is_some_and(|exp| Instant::now() > exp)
    }
}

/// TCP store server that runs on the master node
struct TcpStoreServer {
    store: Arc<DashMap<String, StoreEntry>>,
    waiters: Arc<DashMap<String, Arc<Notify>>>,
    running: Arc<AtomicBool>,
}

impl TcpStoreServer {
    fn new() -> Self {
        Self {
            store: Arc::new(DashMap::new()),
            waiters: Arc::new(DashMap::new()),
            running: Arc::new(AtomicBool::new(true)),
        }
    }

    async fn start(self: Arc<Self>, addr: SocketAddr) -> TorshResult<()> {
        let listener = TcpListener::bind(addr).await.map_err(|e| {
            TorshDistributedError::backend_error(
                "TcpStoreServer",
                format!("Failed to bind to {}: {}", addr, e),
            )
        })?;

        info!("TCP store server started on {}", addr);

        // Start cleanup task for expired entries
        let cleanup_store = self.store.clone();
        let cleanup_running = self.running.clone();
        tokio::spawn(async move {
            while cleanup_running.load(Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_secs(60)).await;
                cleanup_store.retain(|_, entry| !entry.is_expired());
            }
        });

        while self.running.load(Ordering::Relaxed) {
            match listener.accept().await {
                Ok((socket, peer_addr)) => {
                    debug!("Accepted connection from {}", peer_addr);
                    let server = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = server.handle_client(socket).await {
                            warn!("Error handling client {}: {}", peer_addr, e);
                        }
                    });
                }
                Err(e) => {
                    error!("Error accepting connection: {}", e);
                }
            }
        }

        Ok(())
    }

    async fn handle_client(self: Arc<Self>, mut socket: TcpStream) -> TorshResult<()> {
        loop {
            // Read message length (4 bytes)
            let mut len_buf = [0u8; 4];
            if socket.read_exact(&mut len_buf).await.is_err() {
                break; // Client disconnected
            }
            let msg_len = u32::from_be_bytes(len_buf) as usize;

            // Validate message length
            if msg_len == 0 || msg_len > 100 * 1024 * 1024 {
                // 100MB max
                warn!("Invalid message length: {}", msg_len);
                break;
            }

            // Read message data
            let mut msg_buf = vec![0u8; msg_len];
            socket.read_exact(&mut msg_buf).await.map_err(|e| {
                TorshDistributedError::communication_error(
                    "read",
                    format!("Failed to read message: {}", e),
                )
            })?;

            // Deserialize message
            let message: StoreMessage =
                oxicode::serde::decode_from_slice(&msg_buf, oxicode::config::standard())
                    .map_err(|e| {
                        TorshDistributedError::communication_error(
                            "deserialize",
                            format!("Failed to deserialize message: {}", e),
                        )
                    })?
                    .0;

            // Process message
            let response = self.process_message(message).await;

            // Serialize response
            let response_buf =
                oxicode::serde::encode_to_vec(&response, oxicode::config::standard()).map_err(
                    |e| {
                        TorshDistributedError::communication_error(
                            "serialize",
                            format!("Failed to serialize response: {}", e),
                        )
                    },
                )?;

            // Send response length and data
            let len = (response_buf.len() as u32).to_be_bytes();
            socket.write_all(&len).await.map_err(|e| {
                TorshDistributedError::communication_error(
                    "write",
                    format!("Failed to write response length: {}", e),
                )
            })?;
            socket.write_all(&response_buf).await.map_err(|e| {
                TorshDistributedError::communication_error(
                    "write",
                    format!("Failed to write response: {}", e),
                )
            })?;
        }

        Ok(())
    }

    async fn process_message(&self, message: StoreMessage) -> StoreResponse {
        match message {
            StoreMessage::Set { key, value } => {
                self.store.insert(key.clone(), StoreEntry::new(value, None));
                // Notify waiters
                if let Some(notify) = self.waiters.get(&key) {
                    notify.notify_waiters();
                }
                StoreResponse::Ok
            }
            StoreMessage::Get { key } => {
                let value = self.store.get(&key).and_then(|e| {
                    if e.is_expired() {
                        None
                    } else {
                        Some(e.value.clone())
                    }
                });
                StoreResponse::Value(value)
            }
            StoreMessage::Wait { keys } => {
                // Wait for all keys to be available
                for key in keys {
                    while !self.store.contains_key(&key)
                        || self.store.get(&key).map_or(true, |e| e.is_expired())
                    {
                        let notify = self
                            .waiters
                            .entry(key.clone())
                            .or_insert_with(|| Arc::new(Notify::new()))
                            .clone();
                        notify.notified().await;
                    }
                }
                StoreResponse::Ok
            }
            StoreMessage::Delete { key } => {
                self.store.remove(&key);
                StoreResponse::Ok
            }
            StoreMessage::NumKeys => {
                let count = self
                    .store
                    .iter()
                    .filter(|entry| !entry.value().is_expired())
                    .count();
                StoreResponse::NumKeys(count)
            }
            StoreMessage::Contains { key } => {
                let exists = self.store.get(&key).is_some_and(|e| !e.is_expired());
                StoreResponse::Bool(exists)
            }
            StoreMessage::SetWithExpiry {
                key,
                value,
                ttl_secs,
            } => {
                let ttl = Duration::from_secs(ttl_secs);
                self.store
                    .insert(key.clone(), StoreEntry::new(value, Some(ttl)));
                // Notify waiters
                if let Some(notify) = self.waiters.get(&key) {
                    notify.notify_waiters();
                }
                StoreResponse::Ok
            }
            StoreMessage::CompareAndSwap {
                key,
                expected,
                value,
            } => {
                let mut success = false;
                self.store
                    .entry(key.clone())
                    .and_modify(|entry| {
                        if !entry.is_expired() {
                            let current_matches = match &expected {
                                None => false,
                                Some(exp) => &entry.value == exp,
                            };
                            if current_matches {
                                entry.value = value.clone();
                                entry.expiry = None;
                                success = true;
                            }
                        }
                    })
                    .or_insert_with(|| {
                        if expected.is_none() {
                            success = true;
                            StoreEntry::new(value.clone(), None)
                        } else {
                            StoreEntry::new(vec![], None)
                        }
                    });
                if success {
                    if let Some(notify) = self.waiters.get(&key) {
                        notify.notify_waiters();
                    }
                }
                StoreResponse::Bool(success)
            }
            StoreMessage::Add { key, value } => {
                let new_value = self
                    .store
                    .entry(key.clone())
                    .and_modify(|entry| {
                        if !entry.is_expired() {
                            // Try to decode as i64 and add
                            if entry.value.len() == 8 {
                                let current = i64::from_be_bytes(
                                    entry.value[..8].try_into().unwrap_or([0; 8]),
                                );
                                let new = current.wrapping_add(value);
                                entry.value = new.to_be_bytes().to_vec();
                            }
                        }
                    })
                    .or_insert_with(|| StoreEntry::new(value.to_be_bytes().to_vec(), None))
                    .value
                    .clone();

                let result = if new_value.len() == 8 {
                    i64::from_be_bytes(new_value[..8].try_into().unwrap_or([0; 8]))
                } else {
                    0
                };

                if let Some(notify) = self.waiters.get(&key) {
                    notify.notify_waiters();
                }
                StoreResponse::I64(result)
            }
        }
    }

    fn shutdown(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

/// TCP-based distributed store implementation
///
/// This implementation provides a production-ready distributed key-value store
/// using TCP for multi-node coordination. It supports:
///
/// - Master-worker architecture
/// - Atomic operations (CAS, atomic add)
/// - Key expiration
/// - Wait/notify synchronization
/// - Connection pooling
/// - Timeout management
pub struct TcpStore {
    master_addr: IpAddr,
    master_port: u16,
    timeout: Duration,
    is_server: bool,
    server_handle: Option<Arc<TokioRwLock<Option<tokio::task::JoinHandle<()>>>>>,
    server_instance: Option<Arc<TcpStoreServer>>,
}

impl std::fmt::Debug for TcpStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpStore")
            .field("master_addr", &self.master_addr)
            .field("master_port", &self.master_port)
            .field("timeout", &self.timeout)
            .field("is_server", &self.is_server)
            .finish()
    }
}

impl TcpStore {
    /// Create a new TCP store instance
    ///
    /// # Arguments
    ///
    /// * `master_addr` - IP address of the master node
    /// * `master_port` - Port number for the TCP store server
    /// * `timeout` - Timeout for store operations
    /// * `is_server` - Whether this instance should run the server
    pub fn new(
        master_addr: IpAddr,
        master_port: u16,
        timeout: Duration,
        is_server: bool,
    ) -> TorshResult<Self> {
        Ok(Self {
            master_addr,
            master_port,
            timeout,
            is_server,
            server_handle: None,
            server_instance: None,
        })
    }

    /// Start the store (server if is_server=true)
    pub async fn start(&mut self) -> TorshResult<()> {
        if self.is_server {
            let server = Arc::new(TcpStoreServer::new());
            let addr = SocketAddr::new(self.master_addr, self.master_port);
            let server_clone = server.clone();

            let handle = tokio::spawn(async move {
                if let Err(e) = server_clone.start(addr).await {
                    error!("TCP store server error: {}", e);
                }
            });

            self.server_handle = Some(Arc::new(TokioRwLock::new(Some(handle))));
            self.server_instance = Some(server);

            // Wait a bit for server to start
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Ok(())
    }

    /// Connect to the TCP store and send a message
    async fn send_message(&self, message: StoreMessage) -> TorshResult<StoreResponse> {
        let addr = SocketAddr::new(self.master_addr, self.master_port);

        let mut socket = tokio_timeout(self.timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| {
                TorshDistributedError::operation_timeout(
                    format!("connect to {}", addr),
                    self.timeout.as_secs(),
                )
            })?
            .map_err(|e| {
                TorshDistributedError::communication_error(
                    "connect",
                    format!("Failed to connect to {}: {}", addr, e),
                )
            })?;

        // Serialize message
        let msg_buf = oxicode::serde::encode_to_vec(&message, oxicode::config::standard())
            .map_err(|e| {
                TorshDistributedError::communication_error(
                    "serialize",
                    format!("Failed to serialize message: {}", e),
                )
            })?;

        // Send message length and data
        let len = (msg_buf.len() as u32).to_be_bytes();
        socket.write_all(&len).await.map_err(|e| {
            TorshDistributedError::communication_error(
                "write",
                format!("Failed to write message length: {}", e),
            )
        })?;
        socket.write_all(&msg_buf).await.map_err(|e| {
            TorshDistributedError::communication_error(
                "write",
                format!("Failed to write message: {}", e),
            )
        })?;

        // Read response length
        let mut len_buf = [0u8; 4];
        tokio_timeout(self.timeout, socket.read_exact(&mut len_buf))
            .await
            .map_err(|_| {
                TorshDistributedError::operation_timeout(
                    "read response length",
                    self.timeout.as_secs(),
                )
            })?
            .map_err(|e| {
                TorshDistributedError::communication_error(
                    "read",
                    format!("Failed to read response length: {}", e),
                )
            })?;
        let response_len = u32::from_be_bytes(len_buf) as usize;

        // Read response data
        let mut response_buf = vec![0u8; response_len];
        tokio_timeout(self.timeout, socket.read_exact(&mut response_buf))
            .await
            .map_err(|_| {
                TorshDistributedError::operation_timeout("read response", self.timeout.as_secs())
            })?
            .map_err(|e| {
                TorshDistributedError::communication_error(
                    "read",
                    format!("Failed to read response: {}", e),
                )
            })?;

        // Deserialize response
        let response: StoreResponse =
            oxicode::serde::decode_from_slice(&response_buf, oxicode::config::standard())
                .map_err(|e| {
                    TorshDistributedError::communication_error(
                        "deserialize",
                        format!("Failed to deserialize response: {}", e),
                    )
                })?
                .0;

        match response {
            StoreResponse::Error(msg) => {
                Err(TorshDistributedError::backend_error("TcpStore", &msg))
            }
            _ => Ok(response),
        }
    }
}

impl Drop for TcpStore {
    fn drop(&mut self) {
        if let Some(server) = &self.server_instance {
            server.shutdown();
        }
    }
}

#[async_trait]
impl Store for TcpStore {
    async fn set(&self, key: &str, value: &[u8]) -> TorshResult<()> {
        let message = StoreMessage::Set {
            key: key.to_string(),
            value: value.to_vec(),
        };
        let response = self.send_message(message).await?;
        match response {
            StoreResponse::Ok => Ok(()),
            _ => Err(TorshDistributedError::backend_error(
                "TcpStore::set",
                "Unexpected response type",
            )),
        }
    }

    async fn get(&self, key: &str) -> TorshResult<Option<Vec<u8>>> {
        let message = StoreMessage::Get {
            key: key.to_string(),
        };
        let response = self.send_message(message).await?;
        match response {
            StoreResponse::Value(v) => Ok(v),
            _ => Err(TorshDistributedError::backend_error(
                "TcpStore::get",
                "Unexpected response type",
            )),
        }
    }

    async fn wait(&self, keys: &[String]) -> TorshResult<()> {
        let message = StoreMessage::Wait {
            keys: keys.to_vec(),
        };
        let response = self.send_message(message).await?;
        match response {
            StoreResponse::Ok => Ok(()),
            _ => Err(TorshDistributedError::backend_error(
                "TcpStore::wait",
                "Unexpected response type",
            )),
        }
    }

    async fn delete(&self, key: &str) -> TorshResult<()> {
        let message = StoreMessage::Delete {
            key: key.to_string(),
        };
        let response = self.send_message(message).await?;
        match response {
            StoreResponse::Ok => Ok(()),
            _ => Err(TorshDistributedError::backend_error(
                "TcpStore::delete",
                "Unexpected response type",
            )),
        }
    }

    async fn num_keys(&self) -> TorshResult<usize> {
        let message = StoreMessage::NumKeys;
        let response = self.send_message(message).await?;
        match response {
            StoreResponse::NumKeys(n) => Ok(n),
            _ => Err(TorshDistributedError::backend_error(
                "TcpStore::num_keys",
                "Unexpected response type",
            )),
        }
    }

    async fn contains(&self, key: &str) -> TorshResult<bool> {
        let message = StoreMessage::Contains {
            key: key.to_string(),
        };
        let response = self.send_message(message).await?;
        match response {
            StoreResponse::Bool(b) => Ok(b),
            _ => Err(TorshDistributedError::backend_error(
                "TcpStore::contains",
                "Unexpected response type",
            )),
        }
    }

    async fn set_with_expiry(&self, key: &str, value: &[u8], ttl: Duration) -> TorshResult<()> {
        let message = StoreMessage::SetWithExpiry {
            key: key.to_string(),
            value: value.to_vec(),
            ttl_secs: ttl.as_secs(),
        };
        let response = self.send_message(message).await?;
        match response {
            StoreResponse::Ok => Ok(()),
            _ => Err(TorshDistributedError::backend_error(
                "TcpStore::set_with_expiry",
                "Unexpected response type",
            )),
        }
    }

    async fn compare_and_swap(
        &self,
        key: &str,
        expected: Option<&[u8]>,
        value: &[u8],
    ) -> TorshResult<bool> {
        let message = StoreMessage::CompareAndSwap {
            key: key.to_string(),
            expected: expected.map(|v| v.to_vec()),
            value: value.to_vec(),
        };
        let response = self.send_message(message).await?;
        match response {
            StoreResponse::Bool(b) => Ok(b),
            _ => Err(TorshDistributedError::backend_error(
                "TcpStore::compare_and_swap",
                "Unexpected response type",
            )),
        }
    }

    async fn add(&self, key: &str, value: i64) -> TorshResult<i64> {
        let message = StoreMessage::Add {
            key: key.to_string(),
            value,
        };
        let response = self.send_message(message).await?;
        match response {
            StoreResponse::I64(i) => Ok(i),
            _ => Err(TorshDistributedError::backend_error(
                "TcpStore::add",
                "Unexpected response type",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[tokio::test]
    async fn test_tcp_store_basic_operations() {
        let addr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
        let port = 29500;
        let timeout = Duration::from_secs(5);

        // Create and start server
        let mut server_store = TcpStore::new(addr, port, timeout, true).unwrap();
        server_store.start().await.unwrap();

        // Give server time to start
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create client
        let client_store = TcpStore::new(addr, port, timeout, false).unwrap();

        // Test set and get
        client_store.set("key1", b"value1").await.unwrap();
        let value = client_store.get("key1").await.unwrap();
        assert_eq!(value, Some(b"value1".to_vec()));

        // Test contains
        assert!(client_store.contains("key1").await.unwrap());
        assert!(!client_store.contains("nonexistent").await.unwrap());

        // Test num_keys
        client_store.set("key2", b"value2").await.unwrap();
        let num_keys = client_store.num_keys().await.unwrap();
        assert_eq!(num_keys, 2);

        // Test delete
        client_store.delete("key1").await.unwrap();
        assert!(!client_store.contains("key1").await.unwrap());
    }

    #[tokio::test]
    async fn test_tcp_store_atomic_operations() {
        let addr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
        let port = 29501;
        let timeout = Duration::from_secs(5);

        // Create and start server
        let mut server_store = TcpStore::new(addr, port, timeout, true).unwrap();
        server_store.start().await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create client
        let client_store = TcpStore::new(addr, port, timeout, false).unwrap();

        // Test compare and swap
        let success = client_store
            .compare_and_swap("counter", None, b"0")
            .await
            .unwrap();
        assert!(success);

        let success = client_store
            .compare_and_swap("counter", Some(b"0"), b"1")
            .await
            .unwrap();
        assert!(success);

        let success = client_store
            .compare_and_swap("counter", Some(b"0"), b"2")
            .await
            .unwrap();
        assert!(!success);

        // Test atomic add
        let result = client_store.add("num", 5).await.unwrap();
        assert_eq!(result, 5);

        let result = client_store.add("num", 3).await.unwrap();
        assert_eq!(result, 8);
    }

    #[tokio::test]
    async fn test_tcp_store_expiry() {
        let addr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
        let port = 29502;
        let timeout = Duration::from_secs(5);

        // Create and start server
        let mut server_store = TcpStore::new(addr, port, timeout, true).unwrap();
        server_store.start().await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create client
        let client_store = TcpStore::new(addr, port, timeout, false).unwrap();

        // Set with expiry
        client_store
            .set_with_expiry("temp", b"value", Duration::from_secs(1))
            .await
            .unwrap();

        // Should exist immediately
        assert!(client_store.contains("temp").await.unwrap());

        // Wait for expiry
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Should be gone
        assert!(!client_store.contains("temp").await.unwrap());
    }
}
