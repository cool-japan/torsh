//! BigBird model family - Sparse Attention Transformers
//!
//! This module contains all BigBird-related models and configurations.
//!
//! # Architecture Overview
//!
//! BigBird uses sparse attention mechanisms:
//! - **Random Attention**: Random token pairs attend to each other
//! - **Window Attention**: Local sliding window attention
//! - **Global Attention**: Selected tokens attend to all positions
//! - **Block Sparse Structure**: Efficient O(n) complexity
//!
//! # Key Features
//!
//! - Efficient processing of long documents (up to 4096 tokens)
//! - Provably theoretically sound sparse attention
//! - Suitable for document classification, QA, summarization
//! - Compatible with BERT-style pretraining
//!
//! # Usage Examples
//!
//! ```rust,no_run
//! use torsh_models::nlp::bigbird::*;
//!
//! // Create BigBird-base model
//! let model = BigBirdModel::bigbird_base()?;
//!
//! // Create BigBird for sequence classification
//! let classifier = BigBirdForSequenceClassification::bigbird_base_for_classification(2)?;
//! # Ok::<(), torsh_core::error::TorshError>(())
//! ```

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
