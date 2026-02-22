//! Common NLP types and enums

#[cfg(feature = "nlp")]
use std::collections::HashMap;

/// NLP model architectures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NlpArchitecture {
    BERT,
    GPT,
    T5,
    RoBERTa,
    BART,
    DistilBERT,
    ELECTRA,
    ALBERT,
    XLNet,
    DeBERTa,
    Longformer,
    BigBird,
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
            NlpArchitecture::BART => "BART",
            NlpArchitecture::DistilBERT => "DistilBERT",
            NlpArchitecture::ELECTRA => "ELECTRA",
            NlpArchitecture::ALBERT => "ALBERT",
            NlpArchitecture::XLNet => "XLNet",
            NlpArchitecture::DeBERTa => "DeBERTa",
            NlpArchitecture::Longformer => "Longformer",
            NlpArchitecture::BigBird => "BigBird",
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
            NlpArchitecture::BART => 1024,
            NlpArchitecture::DistilBERT => 512,
            NlpArchitecture::ELECTRA => 512,
            NlpArchitecture::ALBERT => 512,
            NlpArchitecture::XLNet => 512,
            NlpArchitecture::DeBERTa => 512,
            NlpArchitecture::Longformer => 4096,
            NlpArchitecture::BigBird => 4096,
            NlpArchitecture::Transformer => 512,
            NlpArchitecture::LSTM => 256,
            NlpArchitecture::GRU => 256,
        }
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
                NlpArchitecture::Longformer,
                NlpArchitecture::BigBird,
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
                NlpArchitecture::Longformer,
                NlpArchitecture::BigBird,
            ],
            NlpTask::TextGeneration => vec![NlpArchitecture::GPT, NlpArchitecture::T5],
            NlpTask::Summarization => vec![
                NlpArchitecture::T5,
                NlpArchitecture::BERT,
                NlpArchitecture::Longformer,
            ],
            NlpTask::Translation => vec![NlpArchitecture::T5, NlpArchitecture::Transformer],
            NlpTask::SentimentAnalysis => vec![
                NlpArchitecture::BERT,
                NlpArchitecture::RoBERTa,
                NlpArchitecture::DistilBERT,
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
            ],
            NlpTask::LanguageModeling => vec![
                NlpArchitecture::GPT,
                NlpArchitecture::BERT,
                NlpArchitecture::XLNet,
                NlpArchitecture::LSTM,
                NlpArchitecture::GRU,
            ],
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
            training_data: vec![
                "BookCorpus".to_string(),
                "English Wikipedia".to_string(),
                "CC-News".to_string(),
                "OpenWebText".to_string(),
            ],
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
            training_data: vec![
                "BookCorpus".to_string(),
                "English Wikipedia".to_string(),
                "CC-News".to_string(),
                "OpenWebText".to_string(),
            ],
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
