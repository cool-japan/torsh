//! Natural language processing operations for ToRSh
//!
//! This crate provides PyTorch-compatible NLP functionality including:
//! - Tokenization (BPE, WordPiece, SentencePiece)
//! - Text embeddings (Word2Vec, GloVe, FastText)
//! - Text generation and beam search
//! - Pre-trained language models
//! - Text datasets and data loaders
//! - Analysis tools (sentiment, coherence, fluency)
//!
//! Built on top of the SciRS2 ecosystem for high-performance text processing.
//!
//! # Examples
//!
//! ```rust,ignore
//! use torsh_text::tokenization::*;
//!
//! // Create a tokenizer
//! let tokenizer = BPETokenizer::from_pretrained("gpt2")?;
//! let tokens = tokenizer.encode("Hello, world!")?;
//! ```

#![allow(clippy::too_many_arguments)]
#![allow(clippy::module_inception)]
#![allow(clippy::large_enum_variant)]
// Allow dead_code for intentional placeholders and future implementations
#![allow(dead_code)]
// Note: Some unused imports remain from auto-generated code - will clean up in future refactoring

pub mod analysis;
pub mod convenience;
pub mod datasets;
pub mod embeddings;
pub mod generation;
// pub mod metrics;  // Has complex import issues - needs significant refactoring (deferred)
pub mod models;
pub mod prelude;
pub mod scirs2_ops;
pub mod scirs2_text_integration; // Re-enabled for checking
pub mod tokenization;
pub mod utils;
pub mod vocab;

#[cfg(test)]
mod test_utils;

pub use analysis::*;
pub use convenience::*;
pub use datasets::*;
pub use embeddings::*;
pub use generation::{
    BeamHypothesis, BeamSearchDecoder, GenerationConfig as TextGenerationConfig,
    NGramRepetitionFilter, RepetitionPenalty, TextGenerator, TextSampler,
};
// pub use metrics::*;  // Disabled - needs significant refactoring (deferred)
pub use models::*;
pub use scirs2_ops::advanced_analytics::{
    compute_advanced_stats, AdvancedTextSampler, AdvancedTextStats, ComplexityAnalyzer,
    ComplexityMetrics,
};
pub use scirs2_ops::performance::{PerformanceMetrics, PerformanceMonitor};
pub use scirs2_ops::*;
pub use scirs2_text_integration::{
    advanced_ops::{cluster_documents, extract_topics, paraphrase_text},
    ClassificationResult, ClusterResult, DeviceType as TextDeviceType, EntityType,
    LanguageDetection, LanguageModel, NamedEntity, PrecisionLevel, SciRS2TextProcessor,
    SentimentLabel, SentimentResult, TextConfig, TextEmbeddings, Topic,
};
pub use tokenization::*;
pub use utils::*;
pub use vocab::*;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

#[derive(Debug, thiserror::Error)]
pub enum TextError {
    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Vocabulary error: {0}")]
    VocabError(String),

    #[error("Dataset error: {0}")]
    DatasetError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Empty input provided where non-empty input is required")]
    EmptyInput,

    #[error("Invalid parameter: {parameter} = {value}, expected {expected}")]
    InvalidParameter {
        parameter: String,
        value: String,
        expected: String,
    },

    #[error("Processing failed for {item}: {reason}")]
    ProcessingError { item: String, reason: String },

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Tensor error: {0}")]
    TensorError(#[from] torsh_core::TorshError),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, TextError>;

impl From<TextError> for torsh_core::TorshError {
    fn from(error: TextError) -> Self {
        torsh_core::TorshError::Other(error.to_string())
    }
}
