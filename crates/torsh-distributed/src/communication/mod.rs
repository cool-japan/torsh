//! Communication utilities for consolidating common patterns across distributed modules
//!
//! This module provides shared utilities to reduce code duplication and improve
//! consistency across the distributed communication modules.

pub mod connection_management;
pub mod error_handling;
pub mod primitives;
pub mod serialization;
pub mod statistics;

// Re-export common utilities for easy access
pub use error_handling::{retry_with_backoff, wrap_communication_error};
pub use primitives::{
    validate_backend_initialized, validate_rank, with_backend_read, with_backend_write,
};
pub use serialization::{
    deserialize_message, deserialize_tensor, serialize_message, serialize_tensor,
};
pub use statistics::{CommunicationStats, OperationStats, StatsCollector};
