// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::{Result, TextError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

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

/// File format specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileFormat {
    Csv { delimiter: char, has_header: bool },
    Tsv { has_header: bool },
    Json,
    Text,
    Conll,
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

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub lowercase: bool,
    pub remove_punctuation: bool,
    pub remove_stopwords: bool,
    pub max_length: Option<usize>,
    pub min_length: Option<usize>,
}

/// Split ratios for train/validation/test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitRatios {
    pub train: f32,
    pub validation: f32,
    pub test: f32,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            name: "default_dataset".to_string(),
            task_type: TaskType::TextClassification,
            file_format: FileFormat::Csv {
                delimiter: ',',
                has_header: true,
            },
            columns: ColumnMapping::default(),
            preprocessing: PreprocessingConfig::default(),
            split_ratios: Some(SplitRatios {
                train: 0.8,
                validation: 0.1,
                test: 0.1,
            }),
        }
    }
}

impl Default for ColumnMapping {
    fn default() -> Self {
        Self {
            text_column: Some(0),
            label_column: Some(1),
            source_column: None,
            target_column: None,
            custom_mapping: HashMap::new(),
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            lowercase: false,
            remove_punctuation: false,
            remove_stopwords: false,
            max_length: None,
            min_length: None,
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
                // JSON parsing would go here
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

        // Apply preprocessing
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
        // Placeholder for sequence labeling
        Err(TextError::DatasetError(
            "Sequence labeling not yet implemented in unified loader".to_string(),
        ))
    }

    fn load_translation_dataset<P: AsRef<Path>>(
        &self,
        _path: P,
    ) -> Result<Box<dyn Dataset<Item = DataItem>>> {
        // Placeholder for translation
        Err(TextError::DatasetError(
            "Translation not yet implemented in unified loader".to_string(),
        ))
    }

    fn load_language_modeling_dataset<P: AsRef<Path>>(
        &self,
        _path: P,
    ) -> Result<Box<dyn Dataset<Item = DataItem>>> {
        // Placeholder for language modeling
        Err(TextError::DatasetError(
            "Language modeling not yet implemented in unified loader".to_string(),
        ))
    }
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

/// Consolidated dataset implementation that can handle multiple data types
pub struct ConsolidatedDataset {
    items: Vec<DataItem>,
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

impl Dataset for ConsolidatedDataset {
    type Item = DataItem;

    fn len(&self) -> usize {
        self.items.len()
    }

    fn get_item(&self, index: usize) -> Result<Self::Item> {
        self.items
            .get(index)
            .cloned()
            .ok_or_else(|| TextError::DatasetError(format!("Index {} out of bounds", index)))
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

    /// Shuffle dataset items
    pub fn shuffle(dataset: &mut ConsolidatedDataset) {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::rand_prelude::SliceRandom;
        use scirs2_core::random::Random;
        let mut rng = Random::seed(42);
        dataset.items.shuffle(&mut rng);
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

/// Base trait for all text datasets
pub trait Dataset {
    type Item;

    /// Get the length of the dataset
    fn len(&self) -> usize;

    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an item by index
    fn get_item(&self, index: usize) -> Result<Self::Item>;

    /// Iterator over the dataset
    fn iter(&self) -> DatasetIterator<'_, Self>
    where
        Self: Sized,
    {
        DatasetIterator {
            dataset: self,
            index: 0,
        }
    }
}

/// Iterator for datasets
pub struct DatasetIterator<'a, D: Dataset> {
    dataset: &'a D,
    index: usize,
}

impl<'a, D: Dataset> Iterator for DatasetIterator<'a, D> {
    type Item = Result<D::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.dataset.len() {
            let item = self.dataset.get_item(self.index);
            self.index += 1;
            Some(item)
        } else {
            None
        }
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

impl Dataset for TextDataset {
    type Item = String;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get_item(&self, index: usize) -> Result<Self::Item> {
        self.data
            .get(index)
            .cloned()
            .ok_or_else(|| TextError::DatasetError(format!("Index {} out of bounds", index)))
    }
}

impl Default for TextDataset {
    fn default() -> Self {
        Self::new()
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

        // Skip header if present
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

        // Create label mappings
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

impl Dataset for ClassificationDataset {
    type Item = (String, String);

    fn len(&self) -> usize {
        self.texts.len()
    }

    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index < self.texts.len() {
            Ok((self.texts[index].clone(), self.labels[index].clone()))
        } else {
            Err(TextError::DatasetError(format!(
                "Index {} out of bounds",
                index
            )))
        }
    }
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

        // Add final sequence if not empty
        if !current_sequence.is_empty() {
            sequences.push(current_sequence);
            labels.push(current_labels);
        }

        // Create label mappings
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

impl Dataset for SequenceLabelingDataset {
    type Item = (Vec<String>, Vec<String>);

    fn len(&self) -> usize {
        self.sequences.len()
    }

    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index < self.sequences.len() {
            Ok((self.sequences[index].clone(), self.labels[index].clone()))
        } else {
            Err(TextError::DatasetError(format!(
                "Index {} out of bounds",
                index
            )))
        }
    }
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

impl Dataset for TranslationDataset {
    type Item = (String, String);

    fn len(&self) -> usize {
        self.source_texts.len()
    }

    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index < self.source_texts.len() {
            Ok((
                self.source_texts[index].clone(),
                self.target_texts[index].clone(),
            ))
        } else {
            Err(TextError::DatasetError(format!(
                "Index {} out of bounds",
                index
            )))
        }
    }
}

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

impl Dataset for LanguageModelingDataset {
    type Item = String;

    fn len(&self) -> usize {
        let full_text = self.texts.join(" ");
        let text_len = full_text.chars().count();
        if text_len < self.sequence_length {
            0
        } else {
            (text_len - self.sequence_length) / self.stride + 1
        }
    }

    fn get_item(&self, index: usize) -> Result<Self::Item> {
        let sequences = self.get_sequences();
        sequences
            .get(index)
            .cloned()
            .ok_or_else(|| TextError::DatasetError(format!("Index {} out of bounds", index)))
    }
}

// ============================================================================
// Common Datasets
// ============================================================================

/// IMDB Movie Reviews Dataset for sentiment classification
#[derive(Debug, Clone)]
pub struct ImdbDataset {
    pub reviews: Vec<String>,
    pub labels: Vec<bool>, // true for positive, false for negative
    pub split: DatasetSplit,
}

#[derive(Debug, Clone, Copy)]
pub enum DatasetSplit {
    Train,
    Test,
    Validation,
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
            DatasetSplit::Validation => "validation", // fallback to test if validation doesn't exist
        };

        let pos_dir = root.join(split_dir).join("pos");
        let neg_dir = root.join(split_dir).join("neg");

        let mut reviews = Vec::new();
        let mut labels = Vec::new();

        // Load positive reviews
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

        // Load negative reviews
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

impl Dataset for ImdbDataset {
    type Item = (String, bool);

    fn len(&self) -> usize {
        self.reviews.len()
    }

    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index < self.reviews.len() {
            Ok((self.reviews[index].clone(), self.labels[index]))
        } else {
            Err(TextError::DatasetError(format!(
                "Index {} out of bounds",
                index
            )))
        }
    }
}

/// AG News Classification Dataset
#[derive(Debug, Clone)]
pub struct AgNewsDataset {
    pub texts: Vec<String>,
    pub labels: Vec<usize>, // 0: World, 1: Sports, 2: Business, 3: Sci/Tech
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
                // Parse class index (1-based in AG News, convert to 0-based)
                if let Ok(class_idx) = parts[0].parse::<usize>() {
                    if class_idx > 0 && class_idx <= 4 {
                        let title = parts[1].trim_matches('"');
                        let description = parts[2].trim_matches('"');
                        let combined_text = format!("{title} {description}");

                        texts.push(combined_text);
                        labels.push(class_idx - 1); // Convert to 0-based indexing
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

impl Dataset for AgNewsDataset {
    type Item = (String, usize);

    fn len(&self) -> usize {
        self.texts.len()
    }

    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index < self.texts.len() {
            Ok((self.texts[index].clone(), self.labels[index]))
        } else {
            Err(TextError::DatasetError(format!(
                "Index {} out of bounds",
                index
            )))
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

#[derive(Debug, Clone, Copy)]
pub enum WikiTextVersion {
    WikiText2,
    WikiText103,
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

        // Split content into articles (separated by empty lines and titles starting with " = ")
        let lines: Vec<&str> = content.lines().collect();
        let mut current_article = String::new();

        for line in lines {
            if line.trim().is_empty() {
                if !current_article.trim().is_empty() {
                    articles.push(current_article.trim().to_string());
                    current_article.clear();
                }
            } else if line.trim().starts_with(" = ") && line.trim().ends_with(" = ") {
                // Article title - start new article
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

        // Add final article
        if !current_article.trim().is_empty() {
            articles.push(current_article.trim().to_string());
        }

        // Extract sentences and tokens
        for article in &articles {
            // Simple sentence splitting (could be improved with proper NLP library)
            let article_sentences: Vec<&str> = article
                .split(&['.', '!', '?'])
                .filter(|s| !s.trim().is_empty())
                .collect();

            for sentence in article_sentences {
                let clean_sentence = sentence.trim().to_string();
                if !clean_sentence.is_empty() {
                    sentences.push(clean_sentence.clone());

                    // Tokenize (simple whitespace splitting)
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

impl Dataset for WikiTextDataset {
    type Item = String;

    fn len(&self) -> usize {
        self.articles.len()
    }

    fn get_item(&self, index: usize) -> Result<Self::Item> {
        self.articles
            .get(index)
            .cloned()
            .ok_or_else(|| TextError::DatasetError(format!("Index {} out of bounds", index)))
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

impl Dataset for Multi30kDataset {
    type Item = (String, String); // (English, German)

    fn len(&self) -> usize {
        self.english_sentences.len()
    }

    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index < self.english_sentences.len() {
            Ok((
                self.english_sentences[index].clone(),
                self.german_sentences[index].clone(),
            ))
        } else {
            Err(TextError::DatasetError(format!(
                "Index {} out of bounds",
                index
            )))
        }
    }
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

        // Extract if it's a zip file
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
        use std::fs::File;
        use zip::ZipArchive;

        let file = File::open(zip_path)?;
        let mut archive = ZipArchive::new(file)
            .map_err(|e| TextError::DatasetError(format!("Failed to open zip: {}", e)))?;

        std::fs::create_dir_all(extract_dir)?;

        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| TextError::DatasetError(format!("Failed to extract file: {}", e)))?;

            let outpath = extract_dir.join(file.name());

            if file.name().ends_with('/') {
                std::fs::create_dir_all(&outpath)?;
            } else {
                if let Some(p) = outpath.parent() {
                    if !p.exists() {
                        std::fs::create_dir_all(p)?;
                    }
                }
                let mut outfile = File::create(&outpath)?;
                std::io::copy(&mut file, &mut outfile)?;
            }
        }

        Ok(())
    }
}

impl Default for DatasetDownloader {
    fn default() -> Self {
        Self::new().expect("Failed to create dataset downloader")
    }
}
