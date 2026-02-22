//! Core types for distributed store

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Timeout for store operations
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// A value stored in the distributed store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreValue {
    data: Vec<u8>,
    timestamp: u64,
}

impl StoreValue {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time should be after UNIX_EPOCH")
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
    /// In-memory store (for single-node testing)
    Memory,
    /// File-based store (for single-node multi-process)
    File,
    /// TCP-based store (for multi-node)
    Tcp,
    /// Redis-based store (for production multi-node)
    Redis,
}

/// Configuration for the distributed store
#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// Backend type
    pub backend: StoreBackend,
    /// Master address for TCP store
    pub master_addr: Option<std::net::IpAddr>,
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
    /// Whether this instance is the server (for TCP store)
    pub is_server: bool,
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
            is_server: false,
        }
    }
}
