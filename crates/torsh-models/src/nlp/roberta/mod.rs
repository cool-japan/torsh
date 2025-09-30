//! RoBERTa model family - Robustly Optimized BERT Pretraining Approach
//!
//! This module contains all RoBERTa-related models and configurations, including:
//! - Base RoBERTa models
//! - RoBERTa for various downstream tasks
//! - Configuration and embedding components

pub mod attention;
pub mod config;
pub mod embeddings;
pub mod layers;
pub mod models;

// Re-export main components
pub use attention::*;
pub use config::*;
pub use embeddings::*;
pub use layers::*;
pub use models::*;
