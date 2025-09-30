//! Prelude module for torsh-text
//!
//! This module re-exports the most commonly used types and traits from torsh-text,
//! allowing users to import everything they need with a single `use torsh_text::prelude::*;`

// Core types and traits
pub use crate::{Result, TextError};

// Tokenization essentials
pub use crate::tokenization::{
    BPETokenizer, CharTokenizer, SubwordTokenizer, Tokenizer, WhitespaceTokenizer,
};

// Advanced tokenization
pub use crate::tokenization::advanced::FastTokenizer;

// Unified tokenization
pub use crate::tokenization::unified::{
    EfficientUnifiedTokenizer, TokenizerConfig, TokenizerFactory, UnifiedTokenizer,
};

// Vocabulary management
pub use crate::vocab::{SpecialTokens, Vocabulary};

// Text preprocessing
pub use crate::utils::{
    BatchProcessor, CustomStep, OptimizedBatchOps, PreprocessingStep, StreamingBatchProcessor,
    TextAugmenter, TextCleaner, TextNormalizer, TextPreprocessingPipeline,
};

// Embeddings
pub use crate::embeddings::{
    CombinedEmbeddings, EmbeddingUtils, PositionalEncoding, WordEmbedding,
};

// Datasets
pub use crate::datasets::{
    AgNewsDataset, ClassificationDataset, ConsolidatedDataset, Dataset, DatasetConfig,
    DatasetDownloader, DatasetUtils, ImdbDataset, LanguageModelingDataset, Multi30kDataset,
    SequenceLabelingDataset, TranslationDataset, UnifiedDatasetLoader, WikiTextDataset,
};

// Text generation
pub use crate::generation::{
    BeamHypothesis, BeamSearchDecoder, GenerationConfig, NGramRepetitionFilter, RepetitionPenalty,
    TextGenerator, TextSampler,
};

// Analysis tools
pub use crate::analysis::{NgramExtractor, TextSimilarity, TextStatistics, TfIdfCalculator};

// Metrics - Temporarily disabled due to module issues
// pub use crate::metrics::{
//     BertScore, BertScoreResult, BleuScore, EditDistance, PerplexityCalculator, RougeMetrics,
//     RougeScore, RougeType, SemanticSimilarity,
// };

// Custom metrics - Temporarily disabled due to module issues
// pub use crate::metrics::custom::{
//     CompositeMetric, CustomMetric, EvaluationFramework, FluencyMetric, MetricRegistry,
//     SemanticCoherenceMetric, WordOverlapMetric,
// };

// Models
pub use crate::models::{
    GenerationConfig as ModelGenerationConfig, ModelRegistry, TextDecoder, TextEncoder, TextModel,
};

// Model registry
pub use crate::models::registry::{create_model, get_config, get_global_registry, list_configs};

// SciRS2 operations
pub use crate::scirs2_ops::SciRS2TextOps;

// SciRS2 string operations
pub use crate::scirs2_ops::string_ops::*;

// Advanced analytics
pub use crate::scirs2_ops::advanced_analytics::{
    compute_advanced_stats, AdvancedTextSampler, AdvancedTextStats, ComplexityAnalyzer,
    ComplexityMetrics,
};

// Performance monitoring
pub use crate::scirs2_ops::performance::{PerformanceMetrics, PerformanceMonitor};

// SciRS2 vectorized operations
pub use crate::scirs2_ops::vectorized_ops::*;

// SciRS2 indexing
pub use crate::scirs2_ops::indexing::*;

// SciRS2 memory optimization
pub use crate::scirs2_ops::memory::*;

// Convenience utilities
pub use crate::convenience::{
    BatchTextProcessor, ComprehensiveTextReport, EnhancedTextAnalyzer, LanguageDetector,
    QuickTextProcessor, TextQualityAssessor,
};

// Re-export commonly used external types
pub use torsh_core::{DType, Device, Shape};
pub use torsh_tensor::Tensor;

/// Convenience macro for creating a preprocessing pipeline
///
/// # Examples
///
/// ```rust
/// use torsh_text::prelude::*;
///
/// let pipeline = preprocessing_pipeline! {
///     normalize: (unicode: true, accents: true, punctuation: false),
///     clean: (urls: true, emails: true, html: true),
///     custom: |text| text.to_lowercase()
/// };
/// ```
#[macro_export]
macro_rules! preprocessing_pipeline {
    (
        normalize: (unicode: $unicode:expr, accents: $accents:expr, punctuation: $punct:expr),
        clean: (urls: $urls:expr, emails: $emails:expr, html: $html:expr)
        $(, custom: $custom:expr)*
    ) => {{
        let normalizer = $crate::utils::TextNormalizer::default()
            .normalize_unicode($unicode)
            .remove_accents($accents)
            .remove_punctuation($punct);
        let cleaner = $crate::utils::TextCleaner::default()
            .remove_urls($urls)
            .remove_emails($emails)
            .remove_html($html);
        let mut pipeline = $crate::utils::TextPreprocessingPipeline::new()
            .with_normalization(normalizer)
            .with_cleaning(cleaner);

        $(
            pipeline = pipeline.add_custom_step(Box::new($crate::utils::CustomStep::new($custom, "custom".to_string())));
        )*

        pipeline
    }};
}

/// Convenience macro for creating a vocabulary with special tokens
///
/// # Examples
///
/// ```rust
/// use torsh_text::prelude::*;
///
/// let vocab = vocabulary! {
///     special_tokens: {
///         pad: "<pad>",
///         unk: "<unk>",
///         bos: "<s>",
///         eos: "</s>"
///     },
///     min_freq: 5
/// };
/// ```
#[macro_export]
macro_rules! vocabulary {
    (
        special_tokens: {
            $($name:ident: $token:expr),* $(,)?
        }
        $(, min_freq: $min_freq:expr)?
    ) => {{
        let mut special_tokens = $crate::vocab::SpecialTokens::default();
        $(
            match stringify!($name) {
                "pad" => special_tokens.pad = $token.to_string(),
                "unk" => special_tokens.unk = $token.to_string(),
                "bos" => special_tokens.bos = $token.to_string(),
                "eos" => special_tokens.eos = $token.to_string(),
                "sep" => special_tokens.sep = $token.to_string(),
                "cls" => special_tokens.cls = $token.to_string(),
                "mask" => special_tokens.mask = $token.to_string(),
                _ => {}
            }
        )*

        let vocab = $crate::vocab::Vocabulary::new(Some(special_tokens));
        vocab
    }};
}

/// Quick text processing function for common use cases
///
/// # Examples
///
/// ```rust
/// use torsh_text::prelude::*;
///
/// let processed = quick_process!(
///     "Hello, world! Visit https://example.com",
///     normalize: true,
///     clean_urls: true,
///     lowercase: true
/// );
/// ```
#[macro_export]
macro_rules! quick_process {
    (
        $text:expr
        $(, normalize: $normalize:expr)?
        $(, clean_urls: $clean_urls:expr)?
        $(, clean_emails: $clean_emails:expr)?
        $(, clean_html: $clean_html:expr)?
        $(, lowercase: $lowercase:expr)?
    ) => {{
        let mut pipeline = $crate::utils::TextPreprocessingPipeline::new();

        $(
            if $normalize {
                let normalizer = $crate::utils::TextNormalizer::default();
                pipeline = pipeline.with_normalization(normalizer);
            }
        )?

        $(
            if $clean_urls {
                let cleaner = $crate::utils::TextCleaner::default().remove_urls(true).remove_emails(false).remove_html(false);
                pipeline = pipeline.with_cleaning(cleaner);
            }
        )?

        $(
            if $clean_emails {
                let cleaner = $crate::utils::TextCleaner::default().remove_urls(false).remove_emails(true).remove_html(false);
                pipeline = pipeline.with_cleaning(cleaner);
            }
        )?

        $(
            if $clean_html {
                let cleaner = $crate::utils::TextCleaner::default().remove_urls(false).remove_emails(false).remove_html(true);
                pipeline = pipeline.with_cleaning(cleaner);
            }
        )?

        $(
            if $lowercase {
                pipeline = pipeline.add_custom_step(Box::new($crate::utils::CustomStep::new(|text: &str| text.to_lowercase(), "lowercase".to_string())));
            }
        )?

        pipeline.process_text($text)
    }};
}

// Re-export macros
pub use preprocessing_pipeline;
pub use quick_process;
pub use vocabulary;
