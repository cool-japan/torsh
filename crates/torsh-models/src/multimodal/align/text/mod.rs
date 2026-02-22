//! ALIGN Text Components
//!
//! BERT-based text encoder with embeddings, transformer layers, attention mechanisms,
//! and feed-forward networks for ALIGN text processing.

pub mod bert;
pub mod embeddings;
pub mod encoder;

// Re-export key components
pub use bert::{
    ALIGNBertAttention, ALIGNBertEncoder, ALIGNBertIntermediate, ALIGNBertLayer, ALIGNBertOutput,
    ALIGNBertSelfOutput,
};
pub use embeddings::ALIGNTextEmbeddings;
pub use encoder::ALIGNTextEncoder;
