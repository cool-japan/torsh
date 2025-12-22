//! Longformer model family - Long Document Transformers
//!
//! This module contains all Longformer-related models and configurations.
//!
//! # Architecture Overview
//!
//! Longformer extends BERT with efficient attention for long documents:
//! - **Sliding Window Attention**: O(n) complexity with local attention
//! - **Global Attention**: Selected tokens can attend globally
//! - **Extended Positions**: Supports up to 4096 tokens (vs BERT's 512)
//! - **Dilated Windows**: Multi-scale attention patterns
//!
//! # Key Features
//!
//! - Efficient processing of long documents
//! - Configurable attention window sizes per layer
//! - Compatible with RoBERTa tokenizer and vocabulary
//! - Suitable for document classification, QA, summarization
//!
//! # Usage Examples
//!
//! ```rust,no_run
//! use torsh_models::nlp::longformer::*;
//!
//! // Create Longformer-base model
//! let model = LongformerModel::longformer_base()?;
//!
//! // Create Longformer for sequence classification
//! let classifier = LongformerForSequenceClassification::longformer_base_for_classification(2)?;
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
