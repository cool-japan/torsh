//! Tensor module - PyTorch-compatible tensor operations in Python bindings
//!
//! This module provides a modular structure for tensor operations:
//! - `core` - Core PyTensor struct and basic operations
//! - `creation` - Tensor creation functions (zeros, ones, randn, etc.)

pub mod core;
pub mod creation;

// Re-export the main types
pub use core::PyTensor;
pub use creation::register_creation_functions;
