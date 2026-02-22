//! NLP models organized by model family
//!
//! This module provides a comprehensive set of NLP model architectures organized into
//! logical families for better maintainability and discoverability.
//!
//! # Modular Structure
//!
//! - `common` - Shared utilities, types, and preprocessing components
//! - `roberta` - RoBERTa model family (BERT-based bidirectional models)
//! - `t5` - T5 model family (Text-to-Text Transfer Transformer)
//! - `xlnet` - XLNet model family (Generalized autoregressive pretraining)
//! - `longformer` - Longformer model family (Long document transformers)
//! - `bigbird` - BigBird model family (Sparse attention transformers)
//!
//! # Backward Compatibility
//!
//! All existing models continue to work exactly as before. The modular structure
//! is an addition that doesn't break any existing code.

// ========================================
// NEW MODULAR STRUCTURE
// ========================================

// Core model families
pub mod bigbird;
pub mod longformer;
pub mod nlp_common;
pub mod roberta;
pub mod t5;
pub mod xlnet;

// Re-export common components for easy access
pub use nlp_common::preprocessing::*;
pub use nlp_common::types::*;
pub use nlp_common::utils::*;

// Re-export main model structs from each family (avoiding ambiguous submodule re-exports)
pub use bigbird::{
    BigBirdConfig, BigBirdEmbeddings, BigBirdEncoder, BigBirdForSequenceClassification,
    BigBirdLayer, BigBirdModel, BigBirdSparseAttention,
};
pub use longformer::{
    LongformerConfig, LongformerEmbeddings, LongformerEncoder, LongformerForSequenceClassification,
    LongformerLayer, LongformerModel, LongformerSlidingWindowAttention,
};
pub use roberta::*;
pub use t5::*;
pub use xlnet::{
    XLNetConfig, XLNetEmbeddings, XLNetEncoder, XLNetForSequenceClassification, XLNetLayer,
    XLNetModel, XLNetRelativeAttention, XLNetTwoStreamAttention,
};

// ========================================
// LEGACY RE-EXPORTS (for backward compatibility)
// ========================================

// Re-export everything for backward compatibility
// This ensures existing code continues to work without modification
