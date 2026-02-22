//! Process group management for distributed training

#![allow(unexpected_cfgs)]

use crate::backend::{Backend, BackendConfig, BackendType, MockBackend};
use crate::{TorshDistributedError, TorshResult};
use parking_lot::RwLock;
use std::sync::Arc;

/// Process rank type
pub type Rank = u32;

/// World size type
pub type WorldSize = u32;

/// Process group for distributed communication
pub struct ProcessGroup {
    backend: Arc<RwLock<Box<dyn Backend>>>,
    rank: Rank,
    world_size: WorldSize,
    #[allow(dead_code)]
    master_addr: String,
    #[allow(dead_code)]
    master_port: u16,
}

impl ProcessGroup {
    /// Create a new process group
    pub async fn new(
        backend_type: BackendType,
        rank: Rank,
        world_size: WorldSize,
        master_addr: &str,
        master_port: u16,
    ) -> TorshResult<Self> {
        let mut backend = create_backend(backend_type, rank, world_size)?;

        // Initialize the backend with default config
        let config = BackendConfig::default();
        backend.init(config).await?;

        let pg = Self {
            backend: Arc::new(RwLock::new(backend)),
            rank,
            world_size,
            master_addr: master_addr.to_string(),
            master_port,
        };

        Ok(pg)
    }

    /// Get the rank of this process
    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// Get the world size
    pub fn world_size(&self) -> WorldSize {
        self.world_size
    }

    /// Get the backend type
    pub fn backend_type(&self) -> BackendType {
        self.backend.read().backend_type()
    }

    /// Get a reference to the backend
    pub fn backend(&self) -> &Arc<RwLock<Box<dyn Backend>>> {
        &self.backend
    }
}

/// Create a backend based on the type
fn create_backend(
    backend_type: BackendType,
    rank: Rank,
    world_size: WorldSize,
) -> TorshResult<Box<dyn Backend>> {
    match backend_type {
        #[cfg(feature = "nccl")]
        BackendType::Nccl => {
            // For now, use mock backend - NCCL backend needs implementation
            Ok(Box::new(MockBackend::new(rank, world_size)))
        }
        #[cfg(not(feature = "nccl"))]
        BackendType::Nccl => Err(TorshDistributedError::feature_not_available(
            "NCCL backend",
            "nccl",
        )),
        #[cfg(feature = "mpi")]
        BackendType::Mpi => {
            // For now, use mock backend - MPI backend needs implementation
            Ok(Box::new(MockBackend::new(rank, world_size)))
        }
        #[cfg(not(feature = "mpi"))]
        BackendType::Mpi => Err(TorshDistributedError::feature_not_available(
            "MPI backend",
            "mpi",
        )),
        BackendType::Gloo => {
            // Use mock backend for now
            Ok(Box::new(MockBackend::new(rank, world_size)))
        }
        BackendType::Custom(name) => Err(TorshDistributedError::feature_not_available(
            format!("Custom backend: {}", name),
            "custom backend implementation",
        )),
    }
}
