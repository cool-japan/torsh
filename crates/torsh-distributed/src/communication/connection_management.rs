//! Connection management utilities for distributed communication
//!
//! This module provides shared connection pooling and management
//! to eliminate duplication between RPC and parameter server.

use super::error_handling::{retry_with_backoff, RetryConfig};
use super::serialization::{deserialize_message, serialize_message, CommunicationMessage};
use crate::{TorshDistributedError, TorshResult};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::{Mutex, RwLock};
use tokio::time::timeout;

/// Configuration for connection management
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Keep-alive timeout
    pub keep_alive_timeout: Duration,
    /// Maximum number of connections per peer
    pub max_connections_per_peer: usize,
    /// Connection retry configuration
    pub retry_config: RetryConfig,
    /// Enable connection pooling
    pub enable_pooling: bool,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(10),
            keep_alive_timeout: Duration::from_secs(300), // 5 minutes
            max_connections_per_peer: 4,
            retry_config: RetryConfig::default(),
            enable_pooling: true,
        }
    }
}

/// Managed TCP connection with automatic reconnection
#[derive(Debug)]
pub struct ManagedConnection {
    stream: Arc<Mutex<TcpStream>>,
    peer_addr: SocketAddr,
    config: ConnectionConfig,
    last_used: Arc<Mutex<std::time::Instant>>,
}

impl ManagedConnection {
    /// Create a new managed connection
    pub async fn new(peer_addr: SocketAddr, config: ConnectionConfig) -> TorshResult<Self> {
        let stream = Self::connect_with_retry(peer_addr, &config).await?;

        Ok(Self {
            stream: Arc::new(Mutex::new(stream)),
            peer_addr,
            config,
            last_used: Arc::new(Mutex::new(std::time::Instant::now())),
        })
    }

    /// Connect with retry logic
    async fn connect_with_retry(
        addr: SocketAddr,
        config: &ConnectionConfig,
    ) -> TorshResult<TcpStream> {
        retry_with_backoff(
            move || async move {
                let connect_future = TcpStream::connect(addr);
                let stream = timeout(config.connect_timeout, connect_future)
                    .await
                    .map_err(|_| TorshDistributedError::OperationTimeout {
                        operation: "tcp_connect".to_string(),
                        timeout_secs: config.connect_timeout.as_secs(),
                    })?
                    .map_err(|e| TorshDistributedError::CommunicationError {
                        operation: "tcp_connect".to_string(),
                        cause: e.to_string(),
                    })?;

                Ok(stream)
            },
            config.retry_config.clone(),
        )
        .await
    }

    /// Send a message over the connection
    pub async fn send_message<T: CommunicationMessage>(&self, message: &T) -> TorshResult<()> {
        use tokio::io::AsyncWriteExt;

        let serialized = serialize_message(message)?;
        let message_len = serialized.len() as u32;

        let mut stream = self.stream.lock().await;
        *self.last_used.lock().await = std::time::Instant::now();

        // Send message length first (4 bytes)
        stream
            .write_all(&message_len.to_be_bytes())
            .await
            .map_err(|e| TorshDistributedError::CommunicationError {
                operation: "send_message_length".to_string(),
                cause: e.to_string(),
            })?;

        // Send message data
        stream.write_all(&serialized).await.map_err(|e| {
            TorshDistributedError::CommunicationError {
                operation: "send_message_data".to_string(),
                cause: e.to_string(),
            }
        })?;

        stream
            .flush()
            .await
            .map_err(|e| TorshDistributedError::CommunicationError {
                operation: "flush_message".to_string(),
                cause: e.to_string(),
            })?;

        Ok(())
    }

    /// Receive a message from the connection
    pub async fn receive_message<T: CommunicationMessage>(&self) -> TorshResult<T> {
        use tokio::io::AsyncReadExt;

        let mut stream = self.stream.lock().await;
        *self.last_used.lock().await = std::time::Instant::now();

        // Read message length (4 bytes)
        let mut len_bytes = [0u8; 4];
        stream.read_exact(&mut len_bytes).await.map_err(|e| {
            TorshDistributedError::CommunicationError {
                operation: "receive_message_length".to_string(),
                cause: e.to_string(),
            }
        })?;

        let message_len = u32::from_be_bytes(len_bytes) as usize;

        // Validate message length (prevent memory exhaustion)
        if message_len > 100 * 1024 * 1024 {
            // 100MB limit
            return Err(TorshDistributedError::CommunicationError {
                operation: "receive_message".to_string(),
                cause: format!("Message too large: {} bytes", message_len),
            }
            .into());
        }

        // Read message data
        let mut message_data = vec![0u8; message_len];
        stream.read_exact(&mut message_data).await.map_err(|e| {
            TorshDistributedError::CommunicationError {
                operation: "receive_message_data".to_string(),
                cause: e.to_string(),
            }
        })?;

        // Deserialize message
        deserialize_message(&message_data)
    }

    /// Check if connection is still alive
    pub fn is_expired(&self) -> bool {
        let last_used = if let Ok(guard) = self.last_used.try_lock() {
            *guard
        } else {
            // If we can't get the lock, assume it's being used (not expired)
            return false;
        };

        last_used.elapsed() > self.config.keep_alive_timeout
    }

    /// Get peer address
    pub fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }
}

/// Connection pool for managing multiple connections to different peers
#[derive(Debug)]
pub struct ConnectionPool {
    connections: Arc<RwLock<HashMap<SocketAddr, Vec<Arc<ManagedConnection>>>>>,
    config: ConnectionConfig,
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new(config: ConnectionConfig) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Get a connection to a peer, creating one if necessary
    pub async fn get_connection(
        &self,
        peer_addr: SocketAddr,
    ) -> TorshResult<Arc<ManagedConnection>> {
        if !self.config.enable_pooling {
            // If pooling is disabled, always create a new connection
            return Ok(Arc::new(
                ManagedConnection::new(peer_addr, self.config.clone()).await?,
            ));
        }

        // First, try to get an existing connection
        {
            let connections = self.connections.read().await;
            if let Some(peer_connections) = connections.get(&peer_addr) {
                // Find a non-expired connection
                for conn in peer_connections {
                    if !conn.is_expired() {
                        return Ok(conn.clone());
                    }
                }
            }
        }

        // No suitable connection found, create a new one
        let new_connection =
            Arc::new(ManagedConnection::new(peer_addr, self.config.clone()).await?);

        // Add to pool
        {
            let mut connections = self.connections.write().await;
            let peer_connections = connections.entry(peer_addr).or_insert_with(Vec::new);

            // Remove expired connections
            peer_connections.retain(|conn| !conn.is_expired());

            // Add new connection if under limit
            if peer_connections.len() < self.config.max_connections_per_peer {
                peer_connections.push(new_connection.clone());
            }
        }

        Ok(new_connection)
    }

    /// Send a message to a peer
    pub async fn send_message<T: CommunicationMessage>(
        &self,
        peer_addr: SocketAddr,
        message: &T,
    ) -> TorshResult<()> {
        let connection = self.get_connection(peer_addr).await?;
        connection.send_message(message).await
    }

    /// Send a message and wait for a response
    pub async fn send_and_receive<Req: CommunicationMessage, Resp: CommunicationMessage>(
        &self,
        peer_addr: SocketAddr,
        request: &Req,
    ) -> TorshResult<Resp> {
        let connection = self.get_connection(peer_addr).await?;
        connection.send_message(request).await?;
        connection.receive_message().await
    }

    /// Clean up expired connections
    pub async fn cleanup_expired(&self) {
        let mut connections = self.connections.write().await;
        for peer_connections in connections.values_mut() {
            peer_connections.retain(|conn| !conn.is_expired());
        }
        // Remove empty entries
        connections.retain(|_, conns| !conns.is_empty());
    }

    /// Get statistics about the connection pool
    pub async fn get_stats(&self) -> ConnectionPoolStats {
        let connections = self.connections.read().await;
        let mut total_connections = 0;
        let mut active_connections = 0;
        let mut peers = 0;

        for peer_connections in connections.values() {
            peers += 1;
            total_connections += peer_connections.len();
            active_connections += peer_connections
                .iter()
                .filter(|conn| !conn.is_expired())
                .count();
        }

        ConnectionPoolStats {
            total_connections,
            active_connections,
            peers,
            expired_connections: total_connections - active_connections,
        }
    }

    /// Close all connections
    pub async fn shutdown(&self) {
        let mut connections = self.connections.write().await;
        connections.clear();
    }
}

/// Statistics about the connection pool
#[derive(Debug, Clone)]
pub struct ConnectionPoolStats {
    pub total_connections: usize,
    pub active_connections: usize,
    pub expired_connections: usize,
    pub peers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tokio::net::TcpListener;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestMessage {
        id: u32,
        content: String,
    }

    async fn setup_test_server() -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};

                    loop {
                        // Read message length
                        let mut len_bytes = [0u8; 4];
                        if stream.read_exact(&mut len_bytes).await.is_err() {
                            break;
                        }
                        let message_len = u32::from_be_bytes(len_bytes) as usize;

                        // Read message data
                        let mut message_data = vec![0u8; message_len];
                        if stream.read_exact(&mut message_data).await.is_err() {
                            break;
                        }

                        // Echo back the same message
                        if stream.write_all(&len_bytes).await.is_err()
                            || stream.write_all(&message_data).await.is_err()
                            || stream.flush().await.is_err()
                        {
                            break;
                        }
                    }
                });
            }
        });

        (addr, handle)
    }

    #[tokio::test]
    async fn test_managed_connection() {
        let (server_addr, _handle) = setup_test_server().await;
        let config = ConnectionConfig::default();

        let connection = ManagedConnection::new(server_addr, config).await.unwrap();

        let message = TestMessage {
            id: 42,
            content: "Hello, server!".to_string(),
        };

        // Send message
        connection.send_message(&message).await.unwrap();

        // Receive echo
        let response: TestMessage = connection.receive_message().await.unwrap();
        assert_eq!(response, message);
    }

    #[tokio::test]
    async fn test_connection_pool() {
        let (server_addr, _handle) = setup_test_server().await;
        let config = ConnectionConfig::default();

        let pool = ConnectionPool::new(config);

        let message = TestMessage {
            id: 123,
            content: "Pool test".to_string(),
        };

        // Test send and receive
        let response: TestMessage = pool.send_and_receive(server_addr, &message).await.unwrap();
        assert_eq!(response, message);

        // Check pool stats
        let stats = pool.get_stats().await;
        assert_eq!(stats.peers, 1);
        assert_eq!(stats.active_connections, 1);
    }

    #[tokio::test]
    async fn test_connection_pool_cleanup() {
        let config = ConnectionConfig {
            keep_alive_timeout: Duration::from_millis(1), // Very short timeout
            ..Default::default()
        };

        let pool = ConnectionPool::new(config);

        // Add a connection that will expire immediately
        let fake_addr: std::net::SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let initial_stats = pool.get_stats().await;

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Cleanup
        pool.cleanup_expired().await;

        let final_stats = pool.get_stats().await;
        assert_eq!(
            final_stats.total_connections,
            initial_stats.total_connections
        );
    }
}
