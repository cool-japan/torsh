//! Pooling functional operations
//!
//! This module contains functional pooling operations (max pool, avg pool, adaptive pool,
//! padding operations, etc.), split from the original monolithic pooling.rs.

// Main pooling functions
pub mod functions;

// Tests
#[cfg(test)]
mod functions_2;

// Re-export all public functions
pub use functions::*;
