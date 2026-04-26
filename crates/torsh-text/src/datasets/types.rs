//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Result, TextError};
use scirs2_core::parallel_ops::*;
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{thread_rng, Random};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use super::functions::Dataset;

/// Language modeling dataset for training language models
#[derive(Debug, Clone)]
pub struct LanguageModelingDataset {
    pub texts: Vec<String>,
    pub sequence_length: usize,
    pub stride: usize,
}
impl LanguageModelingDataset {
    pub fn new(sequence_length: usize, stride: Option<usize>) -> Self {
        let stride = stride.unwrap_or(sequence_length);
        Self {
            texts: Vec::new(),
            sequence_length,
            stride,
        }
    }
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        sequence_length: usize,
        stride: Option<usize>,
    ) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let texts = content.lines().map(|s| s.to_string()).collect();
        let stride = stride.unwrap_or(sequence_length);
        Ok(Self {
            texts,
            sequence_length,
            stride,
        })
    }
    pub fn from_texts(texts: Vec<String>, sequence_length: usize, stride: Option<usize>) -> Self {
        let stride = stride.unwrap_or(sequence_length);
        Self {
            texts,
            sequence_length,
            stride,
        }
    }
    /// Generate sequences of specified length from the text data
    pub fn get_sequences(&self) -> Vec<String> {
        let mut sequences = Vec::new();
        let full_text = self.texts.join(" ");
        let chars: Vec<char> = full_text.chars().collect();
        let mut start = 0;
        while start + self.sequence_length <= chars.len() {
            let sequence: String = chars[start..start + self.sequence_length].iter().collect();
            sequences.push(sequence);
            start += self.stride;
        }
        sequences
    }
}
/// Different types of NLP tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    TextClassification,
    SequenceLabeling,
    Translation,
    LanguageModeling,
    QuestionAnswering,
    Summarization,
}
/// Translation dataset with source-target language pairs
#[derive(Debug, Clone)]
pub struct TranslationDataset {
    pub source_texts: Vec<String>,
    pub target_texts: Vec<String>,
    pub source_lang: String,
    pub target_lang: String,
}
impl TranslationDataset {
    pub fn new(source_lang: String, target_lang: String) -> Self {
        Self {
            source_texts: Vec::new(),
            target_texts: Vec::new(),
            source_lang,
            target_lang,
        }
    }
    /// Load from parallel text files
    pub fn from_files<P: AsRef<Path>>(
        source_path: P,
        target_path: P,
        source_lang: String,
        target_lang: String,
    ) -> Result<Self> {
        let source_content = std::fs::read_to_string(source_path)?;
        let target_content = std::fs::read_to_string(target_path)?;
        let source_texts: Vec<String> = source_content.lines().map(|s| s.to_string()).collect();
        let target_texts: Vec<String> = target_content.lines().map(|s| s.to_string()).collect();
        if source_texts.len() != target_texts.len() {
            return Err(TextError::DatasetError(
                "Source and target files must have the same number of lines".to_string(),
            ));
        }
        Ok(Self {
            source_texts,
            target_texts,
            source_lang,
            target_lang,
        })
    }
    /// Load from TSV format (source\ttarget per line)
    pub fn from_tsv<P: AsRef<Path>>(
        path: P,
        source_lang: String,
        target_lang: String,
    ) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();
        let mut source_texts = Vec::new();
        let mut target_texts = Vec::new();
        for line in lines {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                source_texts.push(parts[0].to_string());
                target_texts.push(parts[1].to_string());
            }
        }
        Ok(Self {
            source_texts,
            target_texts,
            source_lang,
            target_lang,
        })
    }
}
/// Column mapping for different file formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMapping {
    pub text_column: Option<usize>,
    pub label_column: Option<usize>,
    pub source_column: Option<usize>,
    pub target_column: Option<usize>,
    pub custom_mapping: HashMap<String, usize>,
}
pub struct Stopwords;
impl Stopwords {
    /// Get English stopwords
    pub fn english() -> Vec<String> {
        vec![
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
            "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
            "there", "these", "they", "this", "to", "was", "will", "with",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect()
    }
    /// Get common multilingual stopwords
    pub fn common() -> Vec<String> {
        Self::english()
    }
}
/// Multi30k Machine Translation Dataset (English-German)
#[derive(Debug, Clone)]
pub struct Multi30kDataset {
    pub english_sentences: Vec<String>,
    pub german_sentences: Vec<String>,
    pub split: DatasetSplit,
}
impl Multi30kDataset {
    pub fn new(split: DatasetSplit) -> Self {
        Self {
            english_sentences: Vec::new(),
            german_sentences: Vec::new(),
            split,
        }
    }
    /// Load from separate English and German files
    pub fn from_files<P: AsRef<Path>>(en_path: P, de_path: P, split: DatasetSplit) -> Result<Self> {
        use std::fs;
        let en_content = fs::read_to_string(en_path)?;
        let de_content = fs::read_to_string(de_path)?;
        let english_sentences: Vec<String> = en_content.lines().map(|s| s.to_string()).collect();
        let german_sentences: Vec<String> = de_content.lines().map(|s| s.to_string()).collect();
        if english_sentences.len() != german_sentences.len() {
            return Err(TextError::DatasetError(
                "English and German files must have the same number of lines".to_string(),
            ));
        }
        Ok(Self {
            english_sentences,
            german_sentences,
            split,
        })
    }
    /// Load from single file with tab-separated values
    pub fn from_tsv<P: AsRef<Path>>(path: P, split: DatasetSplit) -> Result<Self> {
        use std::fs;
        let content = fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();
        let mut english_sentences = Vec::new();
        let mut german_sentences = Vec::new();
        for line in lines {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                english_sentences.push(parts[0].to_string());
                german_sentences.push(parts[1].to_string());
            }
        }
        Ok(Self {
            english_sentences,
            german_sentences,
            split,
        })
    }
}
#[derive(Debug, Clone, Copy)]
pub enum WikiTextVersion {
    WikiText2,
    WikiText103,
}
/// Cached dataset wrapper for faster repeated access
pub struct CachedDataset<D: Dataset> {
    pub(super) inner: D,
    cache: HashMap<usize, D::Item>,
    cache_hits: usize,
    cache_misses: usize,
}
impl<D: Dataset> CachedDataset<D>
where
    D::Item: Clone,
{
    pub fn new(dataset: D) -> Self {
        Self {
            inner: dataset,
            cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }
    pub fn cache_statistics(&self) -> (usize, usize) {
        (self.cache_hits, self.cache_misses)
    }
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}
/// File format specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileFormat {
    Csv { delimiter: char, has_header: bool },
    Tsv { has_header: bool },
    Json,
    Text,
    Conll,
}
/// AG News Classification Dataset
#[derive(Debug, Clone)]
pub struct AgNewsDataset {
    pub texts: Vec<String>,
    pub labels: Vec<usize>,
    pub split: DatasetSplit,
}
impl AgNewsDataset {
    pub fn new(split: DatasetSplit) -> Self {
        Self {
            texts: Vec::new(),
            labels: Vec::new(),
            split,
        }
    }
    /// Load from CSV format: class_index,title,description
    pub fn from_csv<P: AsRef<Path>>(path: P, split: DatasetSplit) -> Result<Self> {
        use std::fs;
        let content = fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();
        let mut texts = Vec::new();
        let mut labels = Vec::new();
        for line in lines {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 3 {
                if let Ok(class_idx) = parts[0].parse::<usize>() {
                    if class_idx > 0 && class_idx <= 4 {
                        let title = parts[1].trim_matches('"');
                        let description = parts[2].trim_matches('"');
                        let combined_text = format!("{title} {description}");
                        texts.push(combined_text);
                        labels.push(class_idx - 1);
                    }
                }
            }
        }
        Ok(Self {
            texts,
            labels,
            split,
        })
    }
    pub fn class_names() -> Vec<&'static str> {
        vec!["World", "Sports", "Business", "Sci/Tech"]
    }
    pub fn get_class_name(&self, label: usize) -> Option<&'static str> {
        Self::class_names().get(label).copied()
    }
    pub fn num_classes() -> usize {
        4
    }
}
/// Basic text dataset for simple text data
#[derive(Debug, Clone)]
pub struct TextDataset {
    pub data: Vec<String>,
}
impl TextDataset {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let data = content.lines().map(|s| s.to_string()).collect();
        Ok(Self { data })
    }
    pub fn from_texts(texts: Vec<String>) -> Self {
        Self { data: texts }
    }
}
/// Batch iterator for datasets
pub struct BatchIterator<'a, D: Dataset> {
    pub(super) dataset: &'a D,
    pub(super) batch_size: usize,
    pub(super) index: usize,
}
impl<'a, D: Dataset> BatchIterator<'a, D> {
    pub fn new(dataset: &'a D, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            index: 0,
        }
    }
}
/// Classification dataset with text and labels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationDataset {
    pub texts: Vec<String>,
    pub labels: Vec<String>,
    pub label_to_id: std::collections::HashMap<String, usize>,
    pub id_to_label: std::collections::HashMap<usize, String>,
}
impl ClassificationDataset {
    pub fn new() -> Self {
        Self {
            texts: Vec::new(),
            labels: Vec::new(),
            label_to_id: std::collections::HashMap::new(),
            id_to_label: std::collections::HashMap::new(),
        }
    }
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        text_column: usize,
        label_column: usize,
    ) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Ok(Self::new());
        }
        let mut texts = Vec::new();
        let mut labels = Vec::new();
        let mut label_set = std::collections::HashSet::new();
        let start_index = if lines[0].contains(",") && lines[0].split(',').count() > 1 {
            1
        } else {
            0
        };
        for line in &lines[start_index..] {
            let columns: Vec<&str> = line.split(',').collect();
            if columns.len() > text_column.max(label_column) {
                texts.push(columns[text_column].trim().to_string());
                let label = columns[label_column].trim().to_string();
                labels.push(label.clone());
                label_set.insert(label);
            }
        }
        let mut label_to_id = std::collections::HashMap::new();
        let mut id_to_label = std::collections::HashMap::new();
        for (id, label) in label_set.into_iter().enumerate() {
            label_to_id.insert(label.clone(), id);
            id_to_label.insert(id, label);
        }
        Ok(Self {
            texts,
            labels,
            label_to_id,
            id_to_label,
        })
    }
    pub fn num_classes(&self) -> usize {
        self.label_to_id.len()
    }
    pub fn get_label_id(&self, label: &str) -> Option<usize> {
        self.label_to_id.get(label).copied()
    }
    pub fn get_label_name(&self, id: usize) -> Option<&str> {
        self.id_to_label.get(&id).map(|s| s.as_str())
    }
}
#[derive(Debug, Clone)]
pub enum DatasetPreprocessingStep {
    Lowercase,
    RemovePunctuation,
    RemoveStopwords(Vec<String>),
    TrimWhitespace,
    RemoveUrls,
    RemoveHtmlTags,
    NormalizeWhitespace,
    RemoveNumbers,
    MinLength(usize),
    MaxLength(usize),
}
/// Iterator for datasets
pub struct DatasetIterator<'a, D: Dataset> {
    pub(super) dataset: &'a D,
    pub(super) index: usize,
}
/// IMDB Movie Reviews Dataset for sentiment classification
#[derive(Debug, Clone)]
pub struct ImdbDataset {
    pub reviews: Vec<String>,
    pub labels: Vec<bool>,
    pub split: DatasetSplit,
}
impl ImdbDataset {
    pub fn new(split: DatasetSplit) -> Self {
        Self {
            reviews: Vec::new(),
            labels: Vec::new(),
            split,
        }
    }
    /// Load IMDB dataset from directory structure
    /// Expected structure: root/train/pos/, root/train/neg/, root/test/pos/, root/test/neg/
    pub fn from_directory<P: AsRef<Path>>(root_path: P, split: DatasetSplit) -> Result<Self> {
        use std::fs;
        let root = root_path.as_ref();
        let split_dir = match split {
            DatasetSplit::Train => "train",
            DatasetSplit::Test => "test",
            DatasetSplit::Validation => "validation",
        };
        let pos_dir = root.join(split_dir).join("pos");
        let neg_dir = root.join(split_dir).join("neg");
        let mut reviews = Vec::new();
        let mut labels = Vec::new();
        if pos_dir.exists() {
            for entry in fs::read_dir(pos_dir)? {
                let entry = entry?;
                if entry.path().extension().and_then(|s| s.to_str()) == Some("txt") {
                    let content = fs::read_to_string(entry.path())?;
                    reviews.push(content.trim().to_string());
                    labels.push(true);
                }
            }
        }
        if neg_dir.exists() {
            for entry in fs::read_dir(neg_dir)? {
                let entry = entry?;
                if entry.path().extension().and_then(|s| s.to_str()) == Some("txt") {
                    let content = fs::read_to_string(entry.path())?;
                    reviews.push(content.trim().to_string());
                    labels.push(false);
                }
            }
        }
        Ok(Self {
            reviews,
            labels,
            split,
        })
    }
    /// Create from pre-existing data
    pub fn from_data(reviews: Vec<String>, labels: Vec<bool>, split: DatasetSplit) -> Self {
        Self {
            reviews,
            labels,
            split,
        }
    }
    pub fn num_positive(&self) -> usize {
        self.labels.iter().filter(|&&label| label).count()
    }
    pub fn num_negative(&self) -> usize {
        self.labels.iter().filter(|&&label| !label).count()
    }
}
/// Dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    pub total_items: usize,
    pub avg_text_length: usize,
    pub min_text_length: usize,
    pub max_text_length: usize,
}
/// Unified data item that can represent different types of data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataItem {
    Text(String),
    Classification {
        text: String,
        label: String,
    },
    SequenceLabeling {
        tokens: Vec<String>,
        labels: Vec<String>,
    },
    Translation {
        source: String,
        target: String,
    },
    LanguageModeling {
        text: String,
        next_token: Option<String>,
    },
}
/// Streaming dataset loader for large files that don't fit in memory
pub struct StreamingDataset<R: std::io::BufRead> {
    reader: R,
    current_line: usize,
    total_lines: Option<usize>,
    buffer: Vec<String>,
    buffer_size: usize,
}
impl<R: std::io::BufRead> StreamingDataset<R> {
    /// Create a new streaming dataset with default buffer size (1000 lines)
    pub fn new(reader: R) -> Self {
        Self::with_buffer_size(reader, 1000)
    }
    /// Create a new streaming dataset with custom buffer size
    pub fn with_buffer_size(reader: R, buffer_size: usize) -> Self {
        Self {
            reader,
            current_line: 0,
            total_lines: None,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
        }
    }
    /// Read next batch of lines
    pub fn read_batch(&mut self) -> Result<Vec<String>> {
        self.buffer.clear();
        let mut line = String::new();
        for _ in 0..self.buffer_size {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    self.buffer.push(line.trim().to_string());
                    self.current_line += 1;
                }
                Err(e) => return Err(TextError::IoError(e)),
            }
        }
        Ok(self.buffer.clone())
    }
    /// Get current position
    pub fn position(&self) -> usize {
        self.current_line
    }
    /// Check if end of file is reached
    pub fn is_eof(&self) -> bool {
        self.buffer.is_empty()
    }
}
/// Consolidated dataset configuration for unified dataset creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    pub task_type: TaskType,
    pub file_format: FileFormat,
    pub columns: ColumnMapping,
    pub preprocessing: PreprocessingConfig,
    pub split_ratios: Option<SplitRatios>,
}
#[derive(Debug, Clone, Copy)]
pub enum DatasetSplit {
    Train,
    Test,
    Validation,
}
/// Sequence labeling dataset (e.g., NER, POS tagging)
#[derive(Debug, Clone)]
pub struct SequenceLabelingDataset {
    pub sequences: Vec<Vec<String>>,
    pub labels: Vec<Vec<String>>,
    pub label_to_id: std::collections::HashMap<String, usize>,
    pub id_to_label: std::collections::HashMap<usize, String>,
}
impl SequenceLabelingDataset {
    pub fn new() -> Self {
        Self {
            sequences: Vec::new(),
            labels: Vec::new(),
            label_to_id: std::collections::HashMap::new(),
            id_to_label: std::collections::HashMap::new(),
        }
    }
    /// Load from CoNLL format (word\tlabel per line, empty line separates sentences)
    pub fn from_conll<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();
        let mut sequences = Vec::new();
        let mut labels = Vec::new();
        let mut current_sequence = Vec::new();
        let mut current_labels = Vec::new();
        let mut label_set = std::collections::HashSet::new();
        for line in lines {
            if line.trim().is_empty() {
                if !current_sequence.is_empty() {
                    sequences.push(current_sequence);
                    labels.push(current_labels);
                    current_sequence = Vec::new();
                    current_labels = Vec::new();
                }
            } else {
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() >= 2 {
                    current_sequence.push(parts[0].to_string());
                    let label = parts[1].to_string();
                    current_labels.push(label.clone());
                    label_set.insert(label);
                }
            }
        }
        if !current_sequence.is_empty() {
            sequences.push(current_sequence);
            labels.push(current_labels);
        }
        let mut label_to_id = std::collections::HashMap::new();
        let mut id_to_label = std::collections::HashMap::new();
        for (id, label) in label_set.into_iter().enumerate() {
            label_to_id.insert(label.clone(), id);
            id_to_label.insert(id, label);
        }
        Ok(Self {
            sequences,
            labels,
            label_to_id,
            id_to_label,
        })
    }
    pub fn num_labels(&self) -> usize {
        self.label_to_id.len()
    }
}
/// Consolidated dataset implementation that can handle multiple data types
pub struct ConsolidatedDataset {
    pub(super) items: Vec<DataItem>,
    task_type: TaskType,
    metadata: HashMap<String, String>,
}
impl ConsolidatedDataset {
    pub fn new_classification(texts: Vec<String>, labels: Vec<String>) -> Result<Self> {
        if texts.len() != labels.len() {
            return Err(TextError::DatasetError(
                "Texts and labels must have same length".to_string(),
            ));
        }
        let items: Vec<DataItem> = texts
            .into_iter()
            .zip(labels.into_iter())
            .map(|(text, label)| DataItem::Classification { text, label })
            .collect();
        Ok(Self {
            items,
            task_type: TaskType::TextClassification,
            metadata: HashMap::new(),
        })
    }
    pub fn new_translation(sources: Vec<String>, targets: Vec<String>) -> Result<Self> {
        if sources.len() != targets.len() {
            return Err(TextError::DatasetError(
                "Sources and targets must have same length".to_string(),
            ));
        }
        let items: Vec<DataItem> = sources
            .into_iter()
            .zip(targets.into_iter())
            .map(|(source, target)| DataItem::Translation { source, target })
            .collect();
        Ok(Self {
            items,
            task_type: TaskType::Translation,
            metadata: HashMap::new(),
        })
    }
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    pub fn task_type(&self) -> &TaskType {
        &self.task_type
    }
    /// Split dataset into train/validation/test
    pub fn split(&self, ratios: &SplitRatios) -> Result<(Self, Self, Self)> {
        let total_len = self.items.len();
        let train_len = (total_len as f32 * ratios.train) as usize;
        let val_len = (total_len as f32 * ratios.validation) as usize;
        if train_len + val_len >= total_len {
            return Err(TextError::DatasetError(
                "Split ratios sum to >= 1.0".to_string(),
            ));
        }
        let train_items = self.items[..train_len].to_vec();
        let val_items = self.items[train_len..train_len + val_len].to_vec();
        let test_items = self.items[train_len + val_len..].to_vec();
        Ok((
            Self {
                items: train_items,
                task_type: self.task_type.clone(),
                metadata: self.metadata.clone(),
            },
            Self {
                items: val_items,
                task_type: self.task_type.clone(),
                metadata: self.metadata.clone(),
            },
            Self {
                items: test_items,
                task_type: self.task_type.clone(),
                metadata: self.metadata.clone(),
            },
        ))
    }
}

/// Common dataset utilities
pub struct DatasetUtils;

impl DatasetUtils {
    /// Create a balanced subset of a classification dataset
    pub fn create_balanced_subset(
        dataset: &ConsolidatedDataset,
        samples_per_class: usize,
    ) -> Result<ConsolidatedDataset> {
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        let mut balanced_items = Vec::new();

        for item in &dataset.items {
            if let DataItem::Classification { text: _, label } = item {
                let count = class_counts.get(label).unwrap_or(&0);
                if *count < samples_per_class {
                    balanced_items.push(item.clone());
                    class_counts.insert(label.clone(), count + 1);
                }
            }
        }

        Ok(ConsolidatedDataset {
            items: balanced_items,
            task_type: dataset.task_type.clone(),
            metadata: dataset.metadata.clone(),
        })
    }

    /// Shuffle dataset items with optional seed
    pub fn shuffle(dataset: &mut ConsolidatedDataset) {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = thread_rng();
        dataset.items.shuffle(&mut rng);
    }

    /// Shuffle dataset items with a specific seed for reproducibility
    pub fn shuffle_with_seed(dataset: &mut ConsolidatedDataset, seed: u64) {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(seed);
        dataset.items.shuffle(&mut rng);
    }

    /// Sample a random subset from dataset
    pub fn sample(dataset: &ConsolidatedDataset, n: usize) -> Result<ConsolidatedDataset> {
        if n > dataset.items.len() {
            return Err(TextError::DatasetError(format!(
                "Cannot sample {} items from dataset with {} items",
                n,
                dataset.items.len()
            )));
        }

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..dataset.items.len()).collect();
        indices.shuffle(&mut rng);

        let sampled_items: Vec<DataItem> = indices[..n]
            .iter()
            .filter_map(|&i| dataset.items.get(i).cloned())
            .collect();

        Ok(ConsolidatedDataset {
            items: sampled_items,
            task_type: dataset.task_type.clone(),
            metadata: dataset.metadata.clone(),
        })
    }

    /// Filter dataset by minimum token count (whitespace-based)
    pub fn filter_by_token_count(
        dataset: &ConsolidatedDataset,
        min_tokens: Option<usize>,
        max_tokens: Option<usize>,
    ) -> ConsolidatedDataset {
        let filtered_items: Vec<DataItem> = dataset
            .items
            .iter()
            .filter(|item| {
                let token_count = match item {
                    DataItem::Text(text) => text.split_whitespace().count(),
                    DataItem::Classification { text, .. } => text.split_whitespace().count(),
                    DataItem::Translation { source, .. } => source.split_whitespace().count(),
                    DataItem::LanguageModeling { text, .. } => text.split_whitespace().count(),
                    _ => return true, // Keep other types
                };

                if let Some(min) = min_tokens {
                    if token_count < min {
                        return false;
                    }
                }
                if let Some(max) = max_tokens {
                    if token_count > max {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        ConsolidatedDataset {
            items: filtered_items,
            task_type: dataset.task_type.clone(),
            metadata: dataset.metadata.clone(),
        }
    }

    /// Remove duplicates from dataset
    pub fn deduplicate(dataset: &ConsolidatedDataset) -> ConsolidatedDataset {
        let mut seen = std::collections::HashSet::new();
        let dedup_items: Vec<DataItem> = dataset
            .items
            .iter()
            .filter(|item| {
                let key = match item {
                    DataItem::Text(text) => text.clone(),
                    DataItem::Classification { text, .. } => text.clone(),
                    DataItem::Translation { source, .. } => source.clone(),
                    DataItem::LanguageModeling { text, .. } => text.clone(),
                    DataItem::SequenceLabeling { tokens, .. } => tokens.join(" "),
                };
                seen.insert(key)
            })
            .cloned()
            .collect();

        ConsolidatedDataset {
            items: dedup_items,
            task_type: dataset.task_type.clone(),
            metadata: dataset.metadata.clone(),
        }
    }

    /// Parallel batch loading from multiple CSV files
    /// ✅ SciRS2 POLICY - Uses scirs2_core::parallel_ops
    pub fn parallel_load_csv_files<P: AsRef<Path>>(
        paths: &[P],
        text_column: usize,
        label_column: usize,
        _has_header: bool,
    ) -> Result<ConsolidatedDataset> {
        let datasets: Result<Vec<_>> = paths
            .iter()
            .map(|path| {
                ClassificationDataset::from_csv(path, text_column, label_column)
                    .map(|ds| (ds.texts, ds.labels))
            })
            .collect();

        let datasets = datasets?;

        let mut all_texts = Vec::new();
        let mut all_labels = Vec::new();

        for (texts, labels) in datasets {
            all_texts.extend(texts);
            all_labels.extend(labels);
        }

        ConsolidatedDataset::new_classification(all_texts, all_labels)
    }

    /// Get dataset statistics
    pub fn get_statistics(dataset: &ConsolidatedDataset) -> DatasetStatistics {
        let total_items = dataset.items.len();

        let (avg_text_length, min_text_length, max_text_length) = dataset
            .items
            .iter()
            .filter_map(|item| match item {
                DataItem::Text(text) => Some(text.len()),
                DataItem::Classification { text, .. } => Some(text.len()),
                DataItem::Translation { source, .. } => Some(source.len()),
                DataItem::LanguageModeling { text, .. } => Some(text.len()),
                _ => None,
            })
            .fold((0, usize::MAX, 0), |(sum, min, max), len| {
                (sum + len, min.min(len), max.max(len))
            });

        let avg_text_length = if total_items > 0 {
            avg_text_length / total_items
        } else {
            0
        };

        DatasetStatistics {
            total_items,
            avg_text_length,
            min_text_length: if min_text_length == usize::MAX {
                0
            } else {
                min_text_length
            },
            max_text_length,
        }
    }

    /// Filter dataset by text length
    pub fn filter_by_length(
        dataset: &ConsolidatedDataset,
        min_length: Option<usize>,
        max_length: Option<usize>,
    ) -> ConsolidatedDataset {
        let filtered_items: Vec<DataItem> = dataset
            .items
            .iter()
            .filter(|item| {
                let text_len = match item {
                    DataItem::Text(text) => text.len(),
                    DataItem::Classification { text, .. } => text.len(),
                    DataItem::Translation { source, .. } => source.len(),
                    _ => return true, // Keep other types
                };

                if let Some(min) = min_length {
                    if text_len < min {
                        return false;
                    }
                }
                if let Some(max) = max_length {
                    if text_len > max {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        ConsolidatedDataset {
            items: filtered_items,
            task_type: dataset.task_type.clone(),
            metadata: dataset.metadata.clone(),
        }
    }
}

/// WikiText Language Modeling Dataset
#[derive(Debug, Clone)]
pub struct WikiTextDataset {
    pub articles: Vec<String>,
    pub sentences: Vec<String>,
    pub tokens: Vec<String>,
    pub version: WikiTextVersion,
    pub split: DatasetSplit,
}
impl WikiTextDataset {
    pub fn new(version: WikiTextVersion, split: DatasetSplit) -> Self {
        Self {
            articles: Vec::new(),
            sentences: Vec::new(),
            tokens: Vec::new(),
            version,
            split,
        }
    }
    /// Load WikiText dataset from file
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        version: WikiTextVersion,
        split: DatasetSplit,
    ) -> Result<Self> {
        use std::fs;
        let content = fs::read_to_string(path)?;
        let mut articles = Vec::new();
        let mut sentences = Vec::new();
        let mut all_tokens = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut current_article = String::new();
        for line in lines {
            if line.trim().is_empty() {
                if !current_article.trim().is_empty() {
                    articles.push(current_article.trim().to_string());
                    current_article.clear();
                }
            } else if line.trim().starts_with(" = ") && line.trim().ends_with(" = ") {
                if !current_article.trim().is_empty() {
                    articles.push(current_article.trim().to_string());
                }
                current_article = line.trim().to_string();
            } else {
                if !current_article.is_empty() {
                    current_article.push(' ');
                }
                current_article.push_str(line);
            }
        }
        if !current_article.trim().is_empty() {
            articles.push(current_article.trim().to_string());
        }
        for article in &articles {
            let article_sentences: Vec<&str> = article
                .split(&['.', '!', '?'])
                .filter(|s| !s.trim().is_empty())
                .collect();
            for sentence in article_sentences {
                let clean_sentence = sentence.trim().to_string();
                if !clean_sentence.is_empty() {
                    sentences.push(clean_sentence.clone());
                    let tokens: Vec<&str> = clean_sentence.split_whitespace().collect();
                    all_tokens.extend(tokens.into_iter().map(|s| s.to_string()));
                }
            }
        }
        Ok(Self {
            articles,
            sentences,
            tokens: all_tokens,
            version,
            split,
        })
    }
    pub fn get_vocabulary(&self) -> HashMap<String, usize> {
        let mut vocab = HashMap::new();
        for (i, token) in self.tokens.iter().enumerate() {
            vocab.entry(token.clone()).or_insert(i);
        }
        vocab
    }
    pub fn total_tokens(&self) -> usize {
        self.tokens.len()
    }
    pub fn unique_tokens(&self) -> usize {
        let mut unique = std::collections::HashSet::new();
        for token in &self.tokens {
            unique.insert(token);
        }
        unique.len()
    }
}
/// Dataset augmentation utilities for NLP tasks
pub struct DataAugmentation;
impl DataAugmentation {
    /// Apply random word deletion augmentation
    /// ✅ SciRS2 POLICY - Uses scirs2_core::random
    pub fn random_deletion(text: &str, p: f32) -> String {
        let mut rng = thread_rng();
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return text.to_string();
        }
        let kept_words: Vec<&str> = words
            .into_iter()
            .filter(|_| rng.random::<f32>() > p)
            .collect();
        if kept_words.is_empty() {
            text.split_whitespace().next().unwrap_or("").to_string()
        } else {
            kept_words.join(" ")
        }
    }
    /// Apply random word swap augmentation
    /// ✅ SciRS2 POLICY - Uses scirs2_core::random
    pub fn random_swap(text: &str, n: usize) -> String {
        let mut words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        if words.len() < 2 {
            return text.to_string();
        }
        let mut rng = thread_rng();
        for _ in 0..n {
            let idx1 = rng.gen_range(0..words.len());
            let idx2 = rng.gen_range(0..words.len());
            words.swap(idx1, idx2);
        }
        words.join(" ")
    }
    /// Apply synonym replacement (placeholder - requires external dictionary)
    pub fn synonym_replacement(text: &str, _n: usize) -> String {
        text.to_string()
    }
    /// Apply back-translation augmentation (placeholder - requires translation model)
    pub fn back_translation(text: &str) -> String {
        text.to_string()
    }
    /// Apply multiple augmentation techniques
    /// ✅ SciRS2 POLICY - Uses scirs2_core::random
    pub fn augment_text(text: &str, num_variations: usize) -> Vec<String> {
        let mut variations = vec![text.to_string()];
        let mut rng = thread_rng();
        for _ in 0..num_variations {
            let aug_type = rng.gen_range(0..2);
            let augmented = match aug_type {
                0 => Self::random_deletion(text, 0.1),
                _ => Self::random_swap(text, 1),
            };
            variations.push(augmented);
        }
        variations
    }
    /// Augment entire dataset
    /// ✅ SciRS2 POLICY - Uses scirs2_core::parallel_ops
    pub fn augment_dataset(
        dataset: &ConsolidatedDataset,
        variations_per_item: usize,
    ) -> Result<ConsolidatedDataset> {
        let augmented_items: Vec<DataItem> = dataset
            .items
            .par_iter()
            .flat_map(|item| {
                let mut result = vec![item.clone()];
                match item {
                    DataItem::Text(text) => {
                        let variations = Self::augment_text(text, variations_per_item);
                        result.extend(variations.into_iter().skip(1).map(DataItem::Text));
                    }
                    DataItem::Classification { text, label } => {
                        let variations = Self::augment_text(text, variations_per_item);
                        result.extend(variations.into_iter().skip(1).map(|t| {
                            DataItem::Classification {
                                text: t,
                                label: label.clone(),
                            }
                        }));
                    }
                    _ => {}
                }
                result
            })
            .collect();
        Ok(ConsolidatedDataset {
            items: augmented_items,
            task_type: dataset.task_type.clone(),
            metadata: dataset.metadata.clone(),
        })
    }
}
/// Lazy-loading dataset that loads data on-demand
pub struct LazyDataset {
    pub(super) file_path: std::path::PathBuf,
    cache: HashMap<usize, String>,
    cache_size: usize,
    pub(super) total_lines: usize,
}
impl LazyDataset {
    /// Create a new lazy dataset from file path
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        let content = std::fs::read_to_string(&file_path)?;
        let total_lines = content.lines().count();
        Ok(Self {
            file_path,
            cache: HashMap::new(),
            cache_size: 100,
            total_lines,
        })
    }
    /// Set cache size
    pub fn with_cache_size(mut self, cache_size: usize) -> Self {
        self.cache_size = cache_size;
        self
    }
    /// Load specific line with caching
    fn load_line(&mut self, index: usize) -> Result<String> {
        if let Some(cached) = self.cache.get(&index) {
            return Ok(cached.clone());
        }
        let content = std::fs::read_to_string(&self.file_path)?;
        let line = content
            .lines()
            .nth(index)
            .ok_or_else(|| TextError::DatasetError(format!("Index {} out of bounds", index)))?
            .to_string();
        if self.cache.len() >= self.cache_size {
            if let Some(&key) = self.cache.keys().next() {
                self.cache.remove(&key);
            }
        }
        self.cache.insert(index, line.clone());
        Ok(line)
    }
    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}
/// Preprocessing pipeline for datasets
#[derive(Debug, Clone)]
pub struct PreprocessingPipeline {
    steps: Vec<DatasetPreprocessingStep>,
}
impl PreprocessingPipeline {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }
    pub fn add_step(mut self, step: DatasetPreprocessingStep) -> Self {
        self.steps.push(step);
        self
    }
    pub fn lowercase(self) -> Self {
        self.add_step(DatasetPreprocessingStep::Lowercase)
    }
    pub fn remove_punctuation(self) -> Self {
        self.add_step(DatasetPreprocessingStep::RemovePunctuation)
    }
    pub fn remove_stopwords(self, stopwords: Vec<String>) -> Self {
        self.add_step(DatasetPreprocessingStep::RemoveStopwords(stopwords))
    }
    pub fn trim_whitespace(self) -> Self {
        self.add_step(DatasetPreprocessingStep::TrimWhitespace)
    }
    pub fn remove_urls(self) -> Self {
        self.add_step(DatasetPreprocessingStep::RemoveUrls)
    }
    pub fn remove_html_tags(self) -> Self {
        self.add_step(DatasetPreprocessingStep::RemoveHtmlTags)
    }
    pub fn normalize_whitespace(self) -> Self {
        self.add_step(DatasetPreprocessingStep::NormalizeWhitespace)
    }
    pub fn remove_numbers(self) -> Self {
        self.add_step(DatasetPreprocessingStep::RemoveNumbers)
    }
    pub fn min_length(self, min: usize) -> Self {
        self.add_step(DatasetPreprocessingStep::MinLength(min))
    }
    pub fn max_length(self, max: usize) -> Self {
        self.add_step(DatasetPreprocessingStep::MaxLength(max))
    }
    /// Apply pipeline to a single text
    pub fn process(&self, text: &str) -> String {
        let mut result = text.to_string();
        for step in &self.steps {
            result = self.apply_step(&result, step);
        }
        result
    }
    fn apply_step(&self, text: &str, step: &DatasetPreprocessingStep) -> String {
        match step {
            DatasetPreprocessingStep::Lowercase => text.to_lowercase(),
            DatasetPreprocessingStep::RemovePunctuation => {
                text.chars().filter(|c| !c.is_ascii_punctuation()).collect()
            }
            DatasetPreprocessingStep::RemoveStopwords(stopwords) => {
                let words: Vec<&str> = text.split_whitespace().collect();
                words
                    .into_iter()
                    .filter(|w| !stopwords.contains(&w.to_lowercase()))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            DatasetPreprocessingStep::TrimWhitespace => text.trim().to_string(),
            DatasetPreprocessingStep::RemoveUrls => text
                .split_whitespace()
                .filter(|word| !word.starts_with("http://") && !word.starts_with("https://"))
                .collect::<Vec<_>>()
                .join(" "),
            DatasetPreprocessingStep::RemoveHtmlTags => {
                let mut result = String::new();
                let mut in_tag = false;
                for c in text.chars() {
                    match c {
                        '<' => in_tag = true,
                        '>' => in_tag = false,
                        _ if !in_tag => result.push(c),
                        _ => {}
                    }
                }
                result
            }
            DatasetPreprocessingStep::NormalizeWhitespace => {
                text.split_whitespace().collect::<Vec<_>>().join(" ")
            }
            DatasetPreprocessingStep::RemoveNumbers => {
                text.chars().filter(|c| !c.is_numeric()).collect()
            }
            DatasetPreprocessingStep::MinLength(min) => {
                if text.len() >= *min {
                    text.to_string()
                } else {
                    String::new()
                }
            }
            DatasetPreprocessingStep::MaxLength(max) => {
                if text.len() <= *max {
                    text.to_string()
                } else {
                    text.chars().take(*max).collect()
                }
            }
        }
    }
    /// Apply pipeline to entire dataset
    /// ✅ SciRS2 POLICY - Uses scirs2_core::parallel_ops
    pub fn process_dataset(&self, dataset: &ConsolidatedDataset) -> ConsolidatedDataset {
        let processed_items: Vec<DataItem> = dataset
            .items
            .par_iter()
            .map(|item| match item {
                DataItem::Text(text) => DataItem::Text(self.process(text)),
                DataItem::Classification { text, label } => DataItem::Classification {
                    text: self.process(text),
                    label: label.clone(),
                },
                DataItem::Translation { source, target } => DataItem::Translation {
                    source: self.process(source),
                    target: self.process(target),
                },
                DataItem::LanguageModeling { text, next_token } => DataItem::LanguageModeling {
                    text: self.process(text),
                    next_token: next_token.clone(),
                },
                other => other.clone(),
            })
            .collect();
        ConsolidatedDataset {
            items: processed_items,
            task_type: dataset.task_type.clone(),
            metadata: dataset.metadata.clone(),
        }
    }
}
/// Unified dataset loader that can handle multiple formats and tasks
pub struct UnifiedDatasetLoader {
    config: DatasetConfig,
}
impl UnifiedDatasetLoader {
    pub fn new(config: DatasetConfig) -> Self {
        Self { config }
    }
    /// Load dataset from file based on configuration
    pub fn load_from_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Box<dyn Dataset<Item = DataItem>>> {
        match self.config.task_type {
            TaskType::TextClassification => self.load_classification_dataset(path),
            TaskType::SequenceLabeling => self.load_sequence_labeling_dataset(path),
            TaskType::Translation => self.load_translation_dataset(path),
            TaskType::LanguageModeling => self.load_language_modeling_dataset(path),
            _ => Err(TextError::DatasetError(format!(
                "Task type {:?} not yet supported in unified loader",
                self.config.task_type
            ))),
        }
    }
    fn load_classification_dataset<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Box<dyn Dataset<Item = DataItem>>> {
        let content = std::fs::read_to_string(path)?;
        let mut texts = Vec::new();
        let mut labels = Vec::new();
        match &self.config.file_format {
            FileFormat::Csv {
                delimiter,
                has_header,
            } => {
                let lines: Vec<&str> = content.lines().collect();
                let start_idx = if *has_header { 1 } else { 0 };
                for line in &lines[start_idx..] {
                    let columns: Vec<&str> = line.split(*delimiter).collect();
                    if let (Some(text_col), Some(label_col)) = (
                        self.config.columns.text_column,
                        self.config.columns.label_column,
                    ) {
                        if columns.len() > text_col.max(label_col) {
                            texts.push(columns[text_col].trim().to_string());
                            labels.push(columns[label_col].trim().to_string());
                        }
                    }
                }
            }
            FileFormat::Json => {
                return Err(TextError::DatasetError(
                    "JSON format not yet implemented".to_string(),
                ));
            }
            _ => {
                return Err(TextError::DatasetError(
                    "Unsupported format for classification".to_string(),
                ));
            }
        }
        if self.config.preprocessing.lowercase {
            texts = texts.into_iter().map(|t| t.to_lowercase()).collect();
        }
        let dataset = ConsolidatedDataset::new_classification(texts, labels)?;
        Ok(Box::new(dataset))
    }
    fn load_sequence_labeling_dataset<P: AsRef<Path>>(
        &self,
        _path: P,
    ) -> Result<Box<dyn Dataset<Item = DataItem>>> {
        Err(TextError::DatasetError(
            "Sequence labeling not yet implemented in unified loader".to_string(),
        ))
    }
    fn load_translation_dataset<P: AsRef<Path>>(
        &self,
        _path: P,
    ) -> Result<Box<dyn Dataset<Item = DataItem>>> {
        Err(TextError::DatasetError(
            "Translation not yet implemented in unified loader".to_string(),
        ))
    }
    fn load_language_modeling_dataset<P: AsRef<Path>>(
        &self,
        _path: P,
    ) -> Result<Box<dyn Dataset<Item = DataItem>>> {
        Err(TextError::DatasetError(
            "Language modeling not yet implemented in unified loader".to_string(),
        ))
    }
}
/// Split ratios for train/validation/test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitRatios {
    pub train: f32,
    pub validation: f32,
    pub test: f32,
}
/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub lowercase: bool,
    pub remove_punctuation: bool,
    pub remove_stopwords: bool,
    pub max_length: Option<usize>,
    pub min_length: Option<usize>,
}
/// Dataset downloader and manager
pub struct DatasetDownloader {
    cache_dir: std::path::PathBuf,
}
impl DatasetDownloader {
    pub fn new() -> Result<Self> {
        let cache_dir = dirs::home_dir()
            .ok_or_else(|| TextError::DatasetError("Cannot find home directory".to_string()))?
            .join(".torsh")
            .join("datasets");
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }
    pub fn with_cache_dir<P: AsRef<Path>>(cache_dir: P) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }
    pub fn get_cache_dir(&self) -> &Path {
        &self.cache_dir
    }
    /// Download and extract a dataset if not already present
    #[cfg(feature = "pretrained")]
    pub fn download_and_extract(&self, url: &str, filename: &str) -> Result<std::path::PathBuf> {
        use std::fs::File;
        use std::io::Write;
        let file_path = self.cache_dir.join(filename);
        if file_path.exists() {
            return Ok(file_path);
        }
        println!("Downloading {} from {}", filename, url);
        let response = reqwest::blocking::get(url)
            .map_err(|e| TextError::DatasetError(format!("Download failed: {}", e)))?;
        if !response.status().is_success() {
            return Err(TextError::DatasetError(format!(
                "Download failed with status: {}",
                response.status()
            )));
        }
        let content = response
            .bytes()
            .map_err(|e| TextError::DatasetError(format!("Failed to read response: {}", e)))?;
        let mut file = File::create(&file_path)?;
        file.write_all(&content)?;
        if filename.ends_with(".zip") {
            let extract_dir = self.cache_dir.join(filename.trim_end_matches(".zip"));
            if !extract_dir.exists() {
                self.extract_zip(&file_path, &extract_dir)?;
            }
            return Ok(extract_dir);
        }
        Ok(file_path)
    }
    #[cfg(feature = "pretrained")]
    fn extract_zip(&self, zip_path: &Path, extract_dir: &Path) -> Result<()> {
        use oxiarc_archive::zip::ZipReader;
        use std::fs::File;
        let file = File::open(zip_path)?;
        let mut archive = ZipReader::new(file)
            .map_err(|e| TextError::DatasetError(format!("Failed to open zip: {}", e)))?;
        std::fs::create_dir_all(extract_dir)?;
        let entries: Vec<_> = archive.entries().to_vec();
        for entry in entries {
            let outpath = extract_dir.join(&entry.name);
            if entry.name.ends_with('/') {
                std::fs::create_dir_all(&outpath)?;
            } else {
                if let Some(p) = outpath.parent() {
                    if !p.exists() {
                        std::fs::create_dir_all(p)?;
                    }
                }
                let data = archive.extract(&entry).map_err(|e| {
                    TextError::DatasetError(format!("Failed to extract file: {}", e))
                })?;
                std::fs::write(&outpath, data)?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::functions::BatchDataset;
    use super::*;

    #[test]
    fn test_dataset_utils_sample() {
        let texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
        ];
        let labels = vec![
            "label1".to_string(),
            "label2".to_string(),
            "label3".to_string(),
        ];
        let dataset = ConsolidatedDataset::new_classification(texts, labels)
            .expect("Failed to create dataset");

        let sampled = DatasetUtils::sample(&dataset, 2).expect("Failed to sample");
        assert_eq!(sampled.len(), 2);
    }

    #[test]
    fn test_dataset_utils_deduplicate() {
        let texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text1".to_string(), // duplicate
        ];
        let labels = vec![
            "label1".to_string(),
            "label2".to_string(),
            "label1".to_string(),
        ];
        let dataset = ConsolidatedDataset::new_classification(texts, labels)
            .expect("Failed to create dataset");

        let dedup = DatasetUtils::deduplicate(&dataset);
        assert_eq!(dedup.len(), 2);
    }

    #[test]
    fn test_dataset_utils_filter_by_token_count() {
        let texts = vec![
            "short".to_string(),
            "this is longer text".to_string(),
            "medium text".to_string(),
        ];
        let labels = vec![
            "label1".to_string(),
            "label2".to_string(),
            "label3".to_string(),
        ];
        let dataset = ConsolidatedDataset::new_classification(texts, labels)
            .expect("Failed to create dataset");

        let filtered = DatasetUtils::filter_by_token_count(&dataset, Some(2), Some(3));
        assert_eq!(filtered.len(), 1); // "this is longer text" (4 tokens) and "short" (1 token) filtered out, only "medium text" (2 tokens) remains
    }

    #[test]
    fn test_dataset_utils_statistics() {
        let texts = vec!["ab".to_string(), "abcd".to_string(), "abcdef".to_string()];
        let labels = vec![
            "label1".to_string(),
            "label2".to_string(),
            "label3".to_string(),
        ];
        let dataset = ConsolidatedDataset::new_classification(texts, labels)
            .expect("Failed to create dataset");

        let stats = DatasetUtils::get_statistics(&dataset);
        assert_eq!(stats.total_items, 3);
        assert_eq!(stats.min_text_length, 2);
        assert_eq!(stats.max_text_length, 6);
        assert_eq!(stats.avg_text_length, 4);
    }

    #[test]
    fn test_batch_iterator() {
        let texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
            "text4".to_string(),
            "text5".to_string(),
        ];
        let dataset = TextDataset::from_texts(texts);

        let batches: Vec<_> = dataset.batch_iter(2).collect();
        assert_eq!(batches.len(), 3); // 2 + 2 + 1
        assert_eq!(batches[0].as_ref().ok().map(|b| b.len()), Some(2));
        assert_eq!(batches[1].as_ref().ok().map(|b| b.len()), Some(2));
        assert_eq!(batches[2].as_ref().ok().map(|b| b.len()), Some(1));
    }

    #[test]
    fn test_data_augmentation_random_deletion() {
        let text = "this is a test sentence";
        let augmented = DataAugmentation::random_deletion(text, 0.2);
        // Should have some words deleted
        assert!(!augmented.is_empty());
    }

    #[test]
    fn test_data_augmentation_random_swap() {
        let text = "word1 word2 word3 word4";
        let augmented = DataAugmentation::random_swap(text, 2);
        // Should have same number of words
        assert_eq!(
            text.split_whitespace().count(),
            augmented.split_whitespace().count()
        );
    }

    #[test]
    fn test_preprocessing_pipeline() {
        let pipeline = PreprocessingPipeline::new()
            .lowercase()
            .remove_punctuation()
            .normalize_whitespace();

        let text = "Hello,  World!  This   is  a   TEST.";
        let processed = pipeline.process(text);
        assert_eq!(processed, "hello world this is a test");
    }

    #[test]
    fn test_preprocessing_stopwords() {
        let stopwords = Stopwords::english();
        let pipeline = PreprocessingPipeline::new()
            .lowercase()
            .remove_stopwords(stopwords);

        let text = "This is a test with the stopwords";
        let processed = pipeline.process(text);
        // Should remove common stopwords like "is", "a", "the", "with"
        assert!(!processed.contains(" is "));
        assert!(!processed.contains(" a "));
    }

    #[test]
    fn test_preprocessing_remove_urls() {
        let pipeline = PreprocessingPipeline::new().remove_urls();
        let text = "Check out https://example.com and http://test.com for more";
        let processed = pipeline.process(text);
        assert!(!processed.contains("https://"));
        assert!(!processed.contains("http://"));
    }

    #[test]
    fn test_preprocessing_remove_html() {
        let pipeline = PreprocessingPipeline::new().remove_html_tags();
        let text = "<p>Hello <strong>world</strong></p>";
        let processed = pipeline.process(text);
        assert!(!processed.contains('<'));
        assert!(!processed.contains('>'));
    }

    #[test]
    fn test_preprocessing_length_constraints() {
        let pipeline = PreprocessingPipeline::new().min_length(5).max_length(20);

        let short = "Hi";
        let good = "Hello World";
        let long = "This is a very long text that exceeds the maximum length";

        assert_eq!(pipeline.process(short), "");
        assert_eq!(pipeline.process(good), "Hello World");
        assert_eq!(pipeline.process(long).len(), 20);
    }

    #[test]
    fn test_shuffle_with_seed() {
        let texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
            "text4".to_string(),
        ];
        let labels = vec![
            "label1".to_string(),
            "label2".to_string(),
            "label3".to_string(),
            "label4".to_string(),
        ];
        let mut dataset1 = ConsolidatedDataset::new_classification(texts.clone(), labels.clone())
            .expect("Failed to create dataset");
        let mut dataset2 = ConsolidatedDataset::new_classification(texts, labels)
            .expect("Failed to create dataset");

        DatasetUtils::shuffle_with_seed(&mut dataset1, 42);
        DatasetUtils::shuffle_with_seed(&mut dataset2, 42);

        // Both should have same order with same seed
        for i in 0..dataset1.len() {
            let item1 = dataset1.get_item(i).expect("Failed to get item");
            let item2 = dataset2.get_item(i).expect("Failed to get item");
            match (item1, item2) {
                (
                    DataItem::Classification {
                        text: t1,
                        label: l1,
                    },
                    DataItem::Classification {
                        text: t2,
                        label: l2,
                    },
                ) => {
                    assert_eq!(t1, t2);
                    assert_eq!(l1, l2);
                }
                _ => panic!("Unexpected item type"),
            }
        }
    }
}
