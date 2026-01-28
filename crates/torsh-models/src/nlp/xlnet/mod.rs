//! XLNet model family - Generalized Autoregressive Pretraining
//!
//! This module contains all XLNet-related models and configurations, including:
//! - Base XLNet models with permutation language modeling
//! - XLNet for sequence classification
//! - Relative position attention and two-stream attention
//! - Configuration and embedding components
//!
//! # Architecture Overview
//!
//! XLNet introduces several key innovations:
//! - **Permutation Language Modeling**: Learns bidirectional context while maintaining
//!   autoregressive formulation
//! - **Two-Stream Self-Attention**: Separates content and query representations
//! - **Relative Position Encoding**: From Transformer-XL for better long-range dependencies
//! - **Segment Recurrence**: Memory mechanism for processing long documents
//!
//! # Key Differences from BERT
//!
//! - Uses permutation LM instead of masked LM
//! - No \[MASK\] token corruption
//! - Better at handling long-range dependencies
//! - More effective for generation tasks
//!
//! # Usage Examples
//!
//! ```rust,no_run
//! use torsh_models::nlp::xlnet::*;
//!
//! // Create XLNet-base model
//! let model = XLNetModel::xlnet_base()?;
//!
//! // Create XLNet for sequence classification
//! let classifier = XLNetForSequenceClassification::xlnet_base_for_classification(2)?;
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
