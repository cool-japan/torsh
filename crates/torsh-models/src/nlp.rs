//! NLP models and utilities

#[cfg(feature = "nlp")]
use std::collections::HashMap;

use crate::{ModelError, ModelResult};

/// NLP model architectures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NlpArchitecture {
    BERT,
    GPT,
    T5,
    RoBERTa,
    DistilBERT,
    ELECTRA,
    ALBERT,
    XLNet,
    Transformer,
    LSTM,
    GRU,
}

impl NlpArchitecture {
    /// Get architecture name
    pub fn name(&self) -> &'static str {
        match self {
            NlpArchitecture::BERT => "BERT",
            NlpArchitecture::GPT => "GPT",
            NlpArchitecture::T5 => "T5",
            NlpArchitecture::RoBERTa => "RoBERTa",
            NlpArchitecture::DistilBERT => "DistilBERT",
            NlpArchitecture::ELECTRA => "ELECTRA",
            NlpArchitecture::ALBERT => "ALBERT",
            NlpArchitecture::XLNet => "XLNet",
            NlpArchitecture::Transformer => "Transformer",
            NlpArchitecture::LSTM => "LSTM",
            NlpArchitecture::GRU => "GRU",
        }
    }
    
    /// Get typical max sequence length for architecture
    pub fn default_max_length(&self) -> usize {
        match self {
            NlpArchitecture::BERT => 512,
            NlpArchitecture::GPT => 1024,
            NlpArchitecture::T5 => 512,
            NlpArchitecture::RoBERTa => 512,
            NlpArchitecture::DistilBERT => 512,
            NlpArchitecture::ELECTRA => 512,
            NlpArchitecture::ALBERT => 512,
            NlpArchitecture::XLNet => 512,
            NlpArchitecture::Transformer => 512,
            NlpArchitecture::LSTM => 256,
            NlpArchitecture::GRU => 256,
        }
    }
}

/// NLP model variants
#[derive(Debug, Clone)]
pub struct NlpModelVariant {
    /// Architecture type
    pub architecture: NlpArchitecture,
    /// Model size/variant
    pub variant: String,
    /// Number of parameters
    pub parameters: u64,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Supported languages
    pub languages: Vec<String>,
    /// Training datasets
    pub training_data: Vec<String>,
    /// Performance metrics (task-specific)
    pub metrics: HashMap<String, f32>,
}

/// Common NLP model variants
pub fn get_common_nlp_models() -> Vec<NlpModelVariant> {
    vec![
        // BERT variants
        NlpModelVariant {
            architecture: NlpArchitecture::BERT,
            variant: "bert-base-uncased".to_string(),
            parameters: 110_000_000,
            vocab_size: 30522,
            hidden_size: 768,
            num_layers: 12,
            max_length: 512,
            languages: vec!["english".to_string()],
            training_data: vec!["BookCorpus".to_string(), "English Wikipedia".to_string()],
            metrics: {
                let mut m = HashMap::new();
                m.insert("glue_avg".to_string(), 79.6);
                m.insert("squad_v1_f1".to_string(), 88.5);
                m.insert("squad_v2_f1".to_string(), 76.3);
                m
            },
        },
        NlpModelVariant {
            architecture: NlpArchitecture::BERT,
            variant: "bert-large-uncased".to_string(),
            parameters: 340_000_000,
            vocab_size: 30522,
            hidden_size: 1024,
            num_layers: 24,
            max_length: 512,
            languages: vec!["english".to_string()],
            training_data: vec!["BookCorpus".to_string(), "English Wikipedia".to_string()],
            metrics: {
                let mut m = HashMap::new();
                m.insert("glue_avg".to_string(), 82.1);
                m.insert("squad_v1_f1".to_string(), 92.2);
                m.insert("squad_v2_f1".to_string(), 81.8);
                m
            },
        },
        
        // RoBERTa variants
        NlpModelVariant {
            architecture: NlpArchitecture::RoBERTa,
            variant: "roberta-base".to_string(),
            parameters: 125_000_000,
            vocab_size: 50265,
            hidden_size: 768,
            num_layers: 12,
            max_length: 512,
            languages: vec!["english".to_string()],
            training_data: vec!["BookCorpus".to_string(), "English Wikipedia".to_string(), "CC-News".to_string(), "OpenWebText".to_string()],
            metrics: {
                let mut m = HashMap::new();
                m.insert("glue_avg".to_string(), 83.2);
                m.insert("squad_v1_f1".to_string(), 91.5);
                m.insert("squad_v2_f1".to_string(), 83.7);
                m
            },
        },
        NlpModelVariant {
            architecture: NlpArchitecture::RoBERTa,
            variant: "roberta-large".to_string(),
            parameters: 355_000_000,
            vocab_size: 50265,
            hidden_size: 1024,
            num_layers: 24,
            max_length: 512,
            languages: vec!["english".to_string()],
            training_data: vec!["BookCorpus".to_string(), "English Wikipedia".to_string(), "CC-News".to_string(), "OpenWebText".to_string()],
            metrics: {
                let mut m = HashMap::new();
                m.insert("glue_avg".to_string(), 86.4);
                m.insert("squad_v1_f1".to_string(), 94.6);
                m.insert("squad_v2_f1".to_string(), 89.4);
                m
            },
        },
        
        // DistilBERT
        NlpModelVariant {
            architecture: NlpArchitecture::DistilBERT,
            variant: "distilbert-base-uncased".to_string(),
            parameters: 66_000_000,
            vocab_size: 30522,
            hidden_size: 768,
            num_layers: 6,
            max_length: 512,
            languages: vec!["english".to_string()],
            training_data: vec!["Same as BERT".to_string()],
            metrics: {
                let mut m = HashMap::new();
                m.insert("glue_avg".to_string(), 77.0);
                m.insert("squad_v1_f1".to_string(), 86.9);
                m
            },
        },
        
        // GPT variants
        NlpModelVariant {
            architecture: NlpArchitecture::GPT,
            variant: "gpt2".to_string(),
            parameters: 117_000_000,
            vocab_size: 50257,
            hidden_size: 768,
            num_layers: 12,
            max_length: 1024,
            languages: vec!["english".to_string()],
            training_data: vec!["WebText".to_string()],
            metrics: {
                let mut m = HashMap::new();
                m.insert("perplexity".to_string(), 35.76);
                m
            },
        },
        NlpModelVariant {
            architecture: NlpArchitecture::GPT,
            variant: "gpt2-medium".to_string(),
            parameters: 345_000_000,
            vocab_size: 50257,
            hidden_size: 1024,
            num_layers: 24,
            max_length: 1024,
            languages: vec!["english".to_string()],
            training_data: vec!["WebText".to_string()],
            metrics: {
                let mut m = HashMap::new();
                m.insert("perplexity".to_string(), 26.37);
                m
            },
        },
        NlpModelVariant {
            architecture: NlpArchitecture::GPT,
            variant: "gpt2-large".to_string(),
            parameters: 762_000_000,
            vocab_size: 50257,
            hidden_size: 1280,
            num_layers: 36,
            max_length: 1024,
            languages: vec!["english".to_string()],
            training_data: vec!["WebText".to_string()],
            metrics: {
                let mut m = HashMap::new();
                m.insert("perplexity".to_string(), 22.05);
                m
            },
        },
        
        // T5 variants
        NlpModelVariant {
            architecture: NlpArchitecture::T5,
            variant: "t5-small".to_string(),
            parameters: 60_000_000,
            vocab_size: 32128,
            hidden_size: 512,
            num_layers: 6,
            max_length: 512,
            languages: vec!["english".to_string()],
            training_data: vec!["C4".to_string()],
            metrics: {
                let mut m = HashMap::new();
                m.insert("glue_avg".to_string(), 76.3);
                m
            },
        },
        NlpModelVariant {
            architecture: NlpArchitecture::T5,
            variant: "t5-base".to_string(),
            parameters: 220_000_000,
            vocab_size: 32128,
            hidden_size: 768,
            num_layers: 12,
            max_length: 512,
            languages: vec!["english".to_string()],
            training_data: vec!["C4".to_string()],
            metrics: {
                let mut m = HashMap::new();
                m.insert("glue_avg".to_string(), 82.1);
                m
            },
        },
    ]
}

/// Text preprocessing utilities
pub struct TextPreprocessor {
    /// Maximum sequence length
    pub max_length: usize,
    /// Whether to truncate long sequences
    pub truncate: bool,
    /// Whether to pad short sequences
    pub pad: bool,
    /// Padding token ID
    pub pad_token_id: u32,
    /// Whether to add special tokens
    pub add_special_tokens: bool,
}

impl TextPreprocessor {
    /// Create BERT-style preprocessor
    pub fn bert() -> Self {
        Self {
            max_length: 512,
            truncate: true,
            pad: true,
            pad_token_id: 0,
            add_special_tokens: true,
        }
    }
    
    /// Create GPT-style preprocessor
    pub fn gpt() -> Self {
        Self {
            max_length: 1024,
            truncate: true,
            pad: true,
            pad_token_id: 50256,
            add_special_tokens: false,
        }
    }
    
    /// Create custom preprocessor
    pub fn custom(max_length: usize, pad_token_id: u32) -> Self {
        Self {
            max_length,
            truncate: true,
            pad: true,
            pad_token_id,
            add_special_tokens: false,
        }
    }
    
    /// Preprocess token sequence (placeholder implementation)
    pub fn preprocess(&self, _tokens: &[u32]) -> ModelResult<Vec<u32>> {
        // TODO: Implement actual text preprocessing
        // This would involve:
        // 1. Add special tokens if needed (CLS, SEP for BERT)
        // 2. Truncate to max_length if needed
        // 3. Pad to max_length if needed
        // 4. Create attention masks
        
        Err(ModelError::LoadingError {
            reason: "Text preprocessing not yet implemented".to_string(),
        })
    }
}

/// Tokenizer interface (placeholder)
pub trait Tokenizer: Send + Sync {
    /// Tokenize text to token IDs
    fn tokenize(&self, text: &str) -> ModelResult<Vec<u32>>;
    
    /// Decode token IDs to text
    fn decode(&self, tokens: &[u32]) -> ModelResult<String>;
    
    /// Get vocabulary size
    fn vocab_size(&self) -> usize;
    
    /// Get special token IDs
    fn special_tokens(&self) -> HashMap<String, u32>;
}

/// Simple whitespace tokenizer (placeholder)
pub struct WhitespaceTokenizer {
    vocab: HashMap<String, u32>,
    inverse_vocab: HashMap<u32, String>,
}

impl WhitespaceTokenizer {
    pub fn new() -> Self {
        // Create a minimal vocabulary
        let vocab = vec![
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "the", "and", "of", "to", "a", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
        ];
        
        let vocab_map: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, token)| (token.to_string(), i as u32))
            .collect();
        
        let inverse_vocab: HashMap<u32, String> = vocab
            .iter()
            .enumerate()
            .map(|(i, token)| (i as u32, token.to_string()))
            .collect();
        
        Self {
            vocab: vocab_map,
            inverse_vocab,
        }
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> ModelResult<Vec<u32>> {
        let tokens: Vec<u32> = text
            .split_whitespace()
            .map(|word| {
                self.vocab.get(word).copied().unwrap_or(1) // UNK token
            })
            .collect();
        
        Ok(tokens)
    }
    
    fn decode(&self, tokens: &[u32]) -> ModelResult<String> {
        let words: Vec<String> = tokens
            .iter()
            .map(|&token_id| {
                self.inverse_vocab
                    .get(&token_id)
                    .cloned()
                    .unwrap_or_else(|| "[UNK]".to_string())
            })
            .collect();
        
        Ok(words.join(" "))
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    fn special_tokens(&self) -> HashMap<String, u32> {
        let mut special = HashMap::new();
        special.insert("pad_token".to_string(), 0);
        special.insert("unk_token".to_string(), 1);
        special.insert("cls_token".to_string(), 2);
        special.insert("sep_token".to_string(), 3);
        special.insert("mask_token".to_string(), 4);
        special
    }
}

/// NLP model utilities
pub struct NlpModelUtils;

impl NlpModelUtils {
    /// Get recommended preprocessor for a model
    pub fn get_preprocessor(model_name: &str) -> TextPreprocessor {
        match model_name {
            name if name.contains("bert") || name.contains("roberta") => TextPreprocessor::bert(),
            name if name.contains("gpt") => TextPreprocessor::gpt(),
            name if name.contains("t5") => TextPreprocessor::bert(), // T5 uses similar preprocessing
            _ => TextPreprocessor::bert(), // Default to BERT-style
        }
    }
    
    /// Get model variant by name
    pub fn get_model_variant(name: &str) -> Option<NlpModelVariant> {
        let models = get_common_nlp_models();
        models.into_iter().find(|m| m.variant == name)
    }
    
    /// List models by architecture
    pub fn list_models_by_architecture(arch: NlpArchitecture) -> Vec<NlpModelVariant> {
        let models = get_common_nlp_models();
        models.into_iter().filter(|m| m.architecture == arch).collect()
    }
    
    /// Get models by parameter count range
    pub fn get_models_by_size(min_params: u64, max_params: u64) -> Vec<NlpModelVariant> {
        let models = get_common_nlp_models();
        models
            .into_iter()
            .filter(|m| m.parameters >= min_params && m.parameters <= max_params)
            .collect()
    }
    
    /// Apply softmax to logits
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_values.iter().sum();
        exp_values.iter().map(|&x| x / sum).collect()
    }
    
    /// Get top-k tokens from logits
    pub fn get_top_k_tokens(
        logits: &[f32],
        k: usize,
        tokenizer: Option<&dyn Tokenizer>,
    ) -> Vec<(u32, f32, Option<String>)> {
        let mut predictions: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &score)| (i as u32, score))
            .collect();
        
        // Sort by score descending
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top-k
        predictions
            .into_iter()
            .take(k)
            .map(|(token_id, score)| {
                let token_text = tokenizer
                    .and_then(|t| t.decode(&[token_id]).ok());
                (token_id, score, token_text)
            })
            .collect()
    }
    
    /// Simple text generation (greedy decoding)
    pub fn generate_text_greedy(
        initial_tokens: &[u32],
        model_fn: &dyn Fn(&[u32]) -> ModelResult<Vec<f32>>,
        max_length: usize,
        eos_token_id: Option<u32>,
    ) -> ModelResult<Vec<u32>> {
        let mut tokens = initial_tokens.to_vec();
        
        for _ in 0..(max_length - initial_tokens.len()) {
            // Get model predictions
            let logits = model_fn(&tokens)?;
            
            // Get most likely next token (greedy)
            let next_token_id = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            
            tokens.push(next_token_id);
            
            // Check for end-of-sequence
            if let Some(eos_id) = eos_token_id {
                if next_token_id == eos_id {
                    break;
                }
            }
        }
        
        Ok(tokens)
    }
}

/// Common NLP tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NlpTask {
    TextClassification,
    TokenClassification,
    QuestionAnswering,
    TextGeneration,
    Summarization,
    Translation,
    SentimentAnalysis,
    NamedEntityRecognition,
    PartOfSpeechTagging,
    LanguageModeling,
}

impl NlpTask {
    /// Get task name
    pub fn name(&self) -> &'static str {
        match self {
            NlpTask::TextClassification => "Text Classification",
            NlpTask::TokenClassification => "Token Classification",
            NlpTask::QuestionAnswering => "Question Answering",
            NlpTask::TextGeneration => "Text Generation",
            NlpTask::Summarization => "Summarization",
            NlpTask::Translation => "Translation",
            NlpTask::SentimentAnalysis => "Sentiment Analysis",
            NlpTask::NamedEntityRecognition => "Named Entity Recognition",
            NlpTask::PartOfSpeechTagging => "Part-of-Speech Tagging",
            NlpTask::LanguageModeling => "Language Modeling",
        }
    }
    
    /// Get suitable architectures for task
    pub fn suitable_architectures(&self) -> Vec<NlpArchitecture> {
        match self {
            NlpTask::TextClassification => vec![
                NlpArchitecture::BERT,
                NlpArchitecture::RoBERTa,
                NlpArchitecture::DistilBERT,
                NlpArchitecture::ELECTRA,
            ],
            NlpTask::TokenClassification => vec![
                NlpArchitecture::BERT,
                NlpArchitecture::RoBERTa,
                NlpArchitecture::ELECTRA,
            ],
            NlpTask::QuestionAnswering => vec![
                NlpArchitecture::BERT,
                NlpArchitecture::RoBERTa,
                NlpArchitecture::ELECTRA,
            ],
            NlpTask::TextGeneration => vec![
                NlpArchitecture::GPT,
                NlpArchitecture::T5,
            ],
            NlpTask::Summarization => vec![
                NlpArchitecture::T5,
                NlpArchitecture::BERT,
            ],
            NlpTask::Translation => vec![
                NlpArchitecture::T5,
                NlpArchitecture::Transformer,
            ],
            NlpTask::SentimentAnalysis => vec![
                NlpArchitecture::BERT,
                NlpArchitecture::RoBERTa,
                NlpArchitecture::DistilBERT,
                NlpArchitecture::LSTM,
            ],
            NlpTask::NamedEntityRecognition => vec![
                NlpArchitecture::BERT,
                NlpArchitecture::RoBERTa,
                NlpArchitecture::ELECTRA,
            ],
            NlpTask::PartOfSpeechTagging => vec![
                NlpArchitecture::BERT,
                NlpArchitecture::RoBERTa,
                NlpArchitecture::LSTM,
                NlpArchitecture::GRU,
            ],
            NlpTask::LanguageModeling => vec![
                NlpArchitecture::GPT,
                NlpArchitecture::BERT,
                NlpArchitecture::LSTM,
                NlpArchitecture::GRU,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nlp_architecture_name() {
        assert_eq!(NlpArchitecture::BERT.name(), "BERT");
        assert_eq!(NlpArchitecture::GPT.name(), "GPT");
    }
    
    #[test]
    fn test_get_common_nlp_models() {
        let models = get_common_nlp_models();
        assert!(!models.is_empty());
        
        // Check if BERT-base is present
        let bert_base = models.iter().find(|m| m.variant == "bert-base-uncased");
        assert!(bert_base.is_some());
        
        let bert_base = bert_base.unwrap();
        assert_eq!(bert_base.architecture, NlpArchitecture::BERT);
        assert_eq!(bert_base.parameters, 110_000_000);
    }
    
    #[test]
    fn test_text_preprocessor() {
        let preprocessor = TextPreprocessor::bert();
        assert_eq!(preprocessor.max_length, 512);
        assert_eq!(preprocessor.pad_token_id, 0);
        assert!(preprocessor.add_special_tokens);
    }
    
    #[test]
    fn test_whitespace_tokenizer() {
        let tokenizer = WhitespaceTokenizer::new();
        assert!(tokenizer.vocab_size() > 0);
        
        let special_tokens = tokenizer.special_tokens();
        assert!(special_tokens.contains_key("pad_token"));
        assert!(special_tokens.contains_key("unk_token"));
    }
    
    #[test]
    fn test_tokenizer_roundtrip() {
        let tokenizer = WhitespaceTokenizer::new();
        let text = "the quick brown";
        
        if let Ok(tokens) = tokenizer.tokenize(text) {
            if let Ok(decoded) = tokenizer.decode(&tokens) {
                // Should be close to original (may differ due to UNK tokens)
                assert!(decoded.contains("the"));
            }
        }
    }
    
    #[test]
    fn test_nlp_model_utils() {
        let variant = NlpModelUtils::get_model_variant("bert-base-uncased");
        assert!(variant.is_some());
        
        let variant = variant.unwrap();
        assert_eq!(variant.architecture, NlpArchitecture::BERT);
    }
    
    #[test]
    fn test_nlp_task_architectures() {
        let task = NlpTask::TextClassification;
        let archs = task.suitable_architectures();
        assert!(archs.contains(&NlpArchitecture::BERT));
        assert!(archs.contains(&NlpArchitecture::RoBERTa));
    }
    
    #[test]
    fn test_top_k_tokens() {
        let logits = vec![0.1, 0.8, 0.3, 0.9, 0.2];
        let top_3 = NlpModelUtils::get_top_k_tokens(&logits, 3, None);
        
        assert_eq!(top_3.len(), 3);
        assert_eq!(top_3[0].0, 3); // Index of highest score
        assert_eq!(top_3[1].0, 1); // Index of second highest
        assert_eq!(top_3[2].0, 2); // Index of third highest
    }
}