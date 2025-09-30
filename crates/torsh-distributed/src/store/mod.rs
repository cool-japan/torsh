//! Distributed store for process coordination
//!
//! The distributed store provides a key-value store that all processes
//! can access to coordinate initialization and share information during
//! distributed training.
//!
//! This module has been refactored from a single large file (1506 lines)
//! into logical submodules following the 2000-line refactoring policy.

pub mod file;
pub mod memory;
pub mod redis;
pub mod store_trait;
pub mod tcp;
pub mod types;

// Re-export public types and traits for backward compatibility
pub use file::FileStore;
pub use memory::MemoryStore;
pub use redis::RedisStore;
pub use store_trait::Store;
pub use tcp::TcpStore;
pub use types::{StoreBackend, StoreConfig, StoreValue, DEFAULT_TIMEOUT};

use crate::{TorshDistributedError, TorshResult};

/// Factory function to create store instances based on configuration
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
            let master_addr = config.master_addr.ok_or_else(|| {
                TorshDistributedError::invalid_argument(
                    "master_addr",
                    "Master address required for TCP store",
                    "Valid IP address",
                )
            })?;
            let master_port = config.master_port.ok_or_else(|| {
                TorshDistributedError::invalid_argument(
                    "master_port",
                    "Master port required for TCP store",
                    "Valid port number",
                )
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
                    TorshDistributedError::invalid_argument(
                        "redis_url",
                        "Redis URL is required when using redis store backend",
                        "valid Redis connection string",
                    )
                })?;
                Ok(Box::new(RedisStore::new(
                    redis_url.clone(),
                    config.timeout,
                )?))
            }
            #[cfg(not(feature = "redis"))]
            {
                Err(TorshDistributedError::backend_error(
                    "RedisStore",
                    "Redis support not compiled in. Enable 'redis' feature to use Redis store.",
                )
                .into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_memory_store() {
        let config = StoreConfig {
            backend: StoreBackend::Memory,
            ..Default::default()
        };

        let store = create_store(&config);
        assert!(store.is_ok());
    }

    #[test]
    fn test_create_file_store() {
        let config = StoreConfig {
            backend: StoreBackend::File,
            file_path: Some("/tmp/test_store.json".to_string()),
            ..Default::default()
        };

        let store = create_store(&config);
        assert!(store.is_ok());
    }

    #[test]
    fn test_create_file_store_missing_path() {
        let config = StoreConfig {
            backend: StoreBackend::File,
            file_path: None,
            ..Default::default()
        };

        let store = create_store(&config);
        assert!(store.is_err());
    }
}
