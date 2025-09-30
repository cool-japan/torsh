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
pub mod common;
pub mod roberta;
pub mod t5;
// TODO: Implement these modules when they have content
// pub mod xlnet;
// pub mod longformer;
// pub mod bigbird;

// Re-export common components for easy access
pub use common::preprocessing::*;
pub use common::types::*;
pub use common::utils::*;

// Re-export all model families
// TODO: Enable when modules are implemented
// pub use bigbird::*;
// pub use longformer::*;
pub use roberta::*;
pub use t5::*;
// pub use xlnet::*;

// ========================================
// LEGACY RE-EXPORTS (for backward compatibility)
// ========================================

// Re-export everything for backward compatibility
// This ensures existing code continues to work without modification
