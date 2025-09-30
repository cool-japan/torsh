//! Pooling operations organized by functionality
//!
//! This module contains various pooling operations for neural networks,
//! organized into logical sub-modules for better maintainability.

pub mod adaptive;
pub mod advanced;
pub mod basic;
pub mod global;
pub mod unpool;

// Re-export all public functions for backward compatibility
pub use adaptive::*;
pub use advanced::*;
pub use basic::*;
pub use global::*;
pub use unpool::*;
