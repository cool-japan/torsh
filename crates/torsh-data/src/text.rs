//! Text preprocessing and NLP dataset utilities

use crate::{dataset::Dataset, transforms::Transform};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, collections::BTreeMap as HashMap, string::String, vec::Vec};
#[cfg(feature = "std")]
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Text sequence data container
#[derive(Debug, Clone)]
pub struct TextSequence {
    pub text: String,
    pub tokens: Option<Vec<String>>,
    pub token_ids: Option<Vec<usize>>,
}

impl TextSequence {
    pub fn new(text: String) -> Self {
        Self {
            text,
            tokens: None,
            token_ids: None,
        }
    }

    pub fn with_tokens(mut self, tokens: Vec<String>) -> Self {
        self.tokens = Some(tokens);
        self
    }

    pub fn with_token_ids(mut self, token_ids: Vec<usize>) -> Self {
        self.token_ids = Some(token_ids);
        self
    }

    pub fn len(&self) -> usize {
        if let Some(ref tokens) = self.tokens {
            tokens.len()
        } else if let Some(ref token_ids) = self.token_ids {
            token_ids.len()
        } else {
            self.text.split_whitespace().count()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// Simple vocabulary for text tokenization
#[derive(Debug, Clone)]
pub struct Vocabulary {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
    special_tokens: HashMap<String, usize>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            special_tokens: HashMap::new(),
        }
    }

    /// Build vocabulary from text corpus
    pub fn build_from_texts(&mut self, texts: &[String], min_freq: usize) -> Result<()> {
        // Count token frequencies
        let mut token_counts = HashMap::new();

        for text in texts {
            for token in Self::simple_tokenize(text) {
                *token_counts.entry(token).or_insert(0) += 1;
            }
        }

        // Add special tokens first
        self.add_special_token("<UNK>".to_string());
        self.add_special_token("<PAD>".to_string());
        self.add_special_token("<SOS>".to_string());
        self.add_special_token("<EOS>".to_string());

        // Add tokens that meet frequency threshold
        let mut sorted_tokens: Vec<(String, usize)> = token_counts.into_iter().collect();
        sorted_tokens.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency (descending)

        for (token, count) in sorted_tokens {
            if count >= min_freq && !self.token_to_id.contains_key(&token) {
                self.add_token(token);
            }
        }

        Ok(())
    }

    /// Add a regular token
    pub fn add_token(&mut self, token: String) {
        if !self.token_to_id.contains_key(&token) {
            let id = self.id_to_token.len();
            self.token_to_id.insert(token.clone(), id);
            self.id_to_token.push(token);
        }
    }

    /// Add a special token
    pub fn add_special_token(&mut self, token: String) {
        if !self.token_to_id.contains_key(&token) {
            let id = self.id_to_token.len();
            self.token_to_id.insert(token.clone(), id);
            self.special_tokens.insert(token.clone(), id);
            self.id_to_token.push(token);
        }
    }

    /// Convert token to ID
    pub fn token_to_id(&self, token: &str) -> usize {
        self.token_to_id
            .get(token)
            .copied()
            .unwrap_or_else(|| self.unk_id())
    }

    /// Convert ID to token
    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    /// Get unknown token ID
    pub fn unk_id(&self) -> usize {
        self.special_tokens.get("<UNK>").copied().unwrap_or(0)
    }

    /// Get padding token ID
    pub fn pad_id(&self) -> usize {
        self.special_tokens.get("<PAD>").copied().unwrap_or(1)
    }

    /// Get start of sequence token ID
    pub fn sos_id(&self) -> usize {
        self.special_tokens.get("<SOS>").copied().unwrap_or(2)
    }

    /// Get end of sequence token ID
    pub fn eos_id(&self) -> usize {
        self.special_tokens.get("<EOS>").copied().unwrap_or(3)
    }

    /// Get vocabulary size
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// Simple whitespace tokenization
    fn simple_tokenize(text: &str) -> Vec<String> {
        text.split_whitespace().map(|s| s.to_lowercase()).collect()
    }

    /// Tokenize text and convert to IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        Self::simple_tokenize(text)
            .into_iter()
            .map(|token| self.token_to_id(&token))
            .collect()
    }

    /// Convert IDs back to text
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_token(id))
            .filter(|&token| !self.special_tokens.contains_key(token) || token == "<UNK>")
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Text dataset for classification tasks
pub struct TextClassificationDataset {
    texts: Vec<String>,
    labels: Vec<usize>,
    vocabulary: Vocabulary,
    max_length: Option<usize>,
    transform: Option<Box<dyn Transform<TextSequence, Output = Tensor<f32>>>>,
}

impl TextClassificationDataset {
    /// Create a new text classification dataset
    pub fn new(texts: Vec<String>, labels: Vec<usize>) -> Result<Self> {
        if texts.len() != labels.len() {
            return Err(TorshError::InvalidArgument(
                "Number of texts must match number of labels".to_string(),
            ));
        }

        let mut vocabulary = Vocabulary::new();
        vocabulary.build_from_texts(&texts, 1)?;

        Ok(Self {
            texts,
            labels,
            vocabulary,
            max_length: None,
            transform: None,
        })
    }

    /// Set maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Set transform
    pub fn with_transform<T>(mut self, transform: T) -> Self
    where
        T: Transform<TextSequence, Output = Tensor<f32>> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }

    /// Get vocabulary reference
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Get number of classes
    pub fn num_classes(&self) -> usize {
        self.labels.iter().max().map(|&x| x + 1).unwrap_or(0)
    }
}

impl Dataset for TextClassificationDataset {
    type Item = (Tensor<f32>, usize);

    fn len(&self) -> usize {
        self.texts.len()
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.texts.len() {
            return Err(TorshError::IndexError {
                index,
                size: self.texts.len(),
            });
        }

        let text = &self.texts[index];
        let label = self.labels[index];

        // Tokenize and encode
        let token_ids = self.vocabulary.encode(text);
        let tokens = Vocabulary::simple_tokenize(text);

        let mut sequence = TextSequence::new(text.clone())
            .with_tokens(tokens)
            .with_token_ids(token_ids);

        // Apply max length if specified
        if let Some(max_len) = self.max_length {
            if let Some(ref mut token_ids) = sequence.token_ids {
                if token_ids.len() > max_len {
                    token_ids.truncate(max_len);
                } else {
                    // Pad with padding token
                    let pad_id = self.vocabulary.pad_id();
                    token_ids.resize(max_len, pad_id);
                }
            }
        }

        let tensor = if let Some(ref transform) = self.transform {
            transform.transform(sequence)?
        } else {
            // Default: convert token IDs to tensor
            TokenIdsToTensor.transform(sequence)?
        };

        Ok((tensor, label))
    }
}

/// Text file dataset for reading text files from directories
pub struct TextFileDataset {
    files: Vec<(PathBuf, usize)>,
    classes: Vec<String>,
    vocabulary: Vocabulary,
    max_length: Option<usize>,
    transform: Option<Box<dyn Transform<TextSequence, Output = Tensor<f32>>>>,
}

impl TextFileDataset {
    /// Create a new text file dataset
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self> {
        let root = root.as_ref().to_path_buf();

        if !root.exists() {
            return Err(TorshError::IoError(format!(
                "Directory does not exist: {root:?}"
            )));
        }

        let mut classes = Vec::new();
        let mut files = Vec::new();
        let mut all_texts = Vec::new();

        // Scan subdirectories for classes
        for entry in std::fs::read_dir(&root).map_err(|e| TorshError::IoError(e.to_string()))? {
            let entry = entry.map_err(|e| TorshError::IoError(e.to_string()))?;
            let path = entry.path();

            if path.is_dir() {
                let class_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .ok_or_else(|| TorshError::IoError("Invalid class directory name".to_string()))?
                    .to_string();

                let class_idx = classes.len();
                classes.push(class_name);

                // Scan text files in class directory
                for file_entry in
                    std::fs::read_dir(&path).map_err(|e| TorshError::IoError(e.to_string()))?
                {
                    let file_entry = file_entry.map_err(|e| TorshError::IoError(e.to_string()))?;
                    let file_path = file_entry.path();

                    if Self::is_text_file(&file_path) {
                        files.push((file_path.clone(), class_idx));

                        // Read file content for vocabulary building
                        if let Ok(content) = std::fs::read_to_string(&file_path) {
                            all_texts.push(content);
                        }
                    }
                }
            }
        }

        // Build vocabulary
        let mut vocabulary = Vocabulary::new();
        vocabulary.build_from_texts(&all_texts, 2)?;

        Ok(Self {
            files,
            classes,
            vocabulary,
            max_length: None,
            transform: None,
        })
    }

    /// Check if file is a text file
    fn is_text_file(path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            matches!(
                extension.to_lowercase().as_str(),
                "txt" | "text" | "md" | "rst" | "csv" | "json"
            )
        } else {
            false
        }
    }

    /// Set maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Set transform
    pub fn with_transform<T>(mut self, transform: T) -> Self
    where
        T: Transform<TextSequence, Output = Tensor<f32>> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }

    /// Get class names
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Get vocabulary reference
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }
}

impl Dataset for TextFileDataset {
    type Item = (Tensor<f32>, usize);

    fn len(&self) -> usize {
        self.files.len()
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.files.len() {
            return Err(TorshError::IndexError {
                index,
                size: self.files.len(),
            });
        }

        let (ref path, class_idx) = self.files[index];

        // Read file content
        let text = std::fs::read_to_string(path)
            .map_err(|e| TorshError::IoError(format!("Failed to read file {path:?}: {e}")))?;

        // Tokenize and encode
        let token_ids = self.vocabulary.encode(&text);
        let tokens = Vocabulary::simple_tokenize(&text);

        let mut sequence = TextSequence::new(text)
            .with_tokens(tokens)
            .with_token_ids(token_ids);

        // Apply max length if specified
        if let Some(max_len) = self.max_length {
            if let Some(ref mut token_ids) = sequence.token_ids {
                if token_ids.len() > max_len {
                    token_ids.truncate(max_len);
                } else {
                    // Pad with padding token
                    let pad_id = self.vocabulary.pad_id();
                    token_ids.resize(max_len, pad_id);
                }
            }
        }

        let tensor = if let Some(ref transform) = self.transform {
            transform.transform(sequence)?
        } else {
            // Default: convert token IDs to tensor
            TokenIdsToTensor.transform(sequence)?
        };

        Ok((tensor, class_idx))
    }
}

/// Transform to convert token IDs to tensor
pub struct TokenIdsToTensor;

impl Transform<TextSequence> for TokenIdsToTensor {
    type Output = Tensor<f32>;

    fn transform(&self, input: TextSequence) -> Result<Self::Output> {
        if let Some(token_ids) = input.token_ids {
            // Convert token IDs to f32 tensor
            let len = token_ids.len();
            let data: Vec<f32> = token_ids.into_iter().map(|id| id as f32).collect();
            Tensor::from_data(data, vec![len], torsh_core::device::DeviceType::Cpu)
        } else {
            Err(TorshError::InvalidArgument(
                "TextSequence must have token_ids for tensor conversion".to_string(),
            ))
        }
    }
}

/// Text preprocessing transforms
pub mod transforms {
    use super::*;
    use crate::transforms::Transform;

    /// Convert text to lowercase
    pub struct ToLowercase;

    impl Transform<TextSequence> for ToLowercase {
        type Output = TextSequence;

        fn transform(&self, mut input: TextSequence) -> Result<Self::Output> {
            input.text = input.text.to_lowercase();
            if let Some(ref mut tokens) = input.tokens {
                for token in tokens.iter_mut() {
                    *token = token.to_lowercase();
                }
            }
            Ok(input)
        }
    }

    /// Remove punctuation from text
    pub struct RemovePunctuation;

    impl Transform<TextSequence> for RemovePunctuation {
        type Output = TextSequence;

        fn transform(&self, mut input: TextSequence) -> Result<Self::Output> {
            input.text = input
                .text
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect();

            if let Some(ref mut tokens) = input.tokens {
                for token in tokens.iter_mut() {
                    *token = token.chars().filter(|c| c.is_alphanumeric()).collect();
                }
                // Remove empty tokens
                tokens.retain(|token| !token.is_empty());
            }
            Ok(input)
        }
    }

    /// Truncate or pad sequence to fixed length
    pub struct FixedLength {
        length: usize,
        pad_token_id: usize,
    }

    impl FixedLength {
        pub fn new(length: usize, pad_token_id: usize) -> Self {
            Self {
                length,
                pad_token_id,
            }
        }
    }

    impl Transform<TextSequence> for FixedLength {
        type Output = TextSequence;

        fn transform(&self, mut input: TextSequence) -> Result<Self::Output> {
            if let Some(ref mut token_ids) = input.token_ids {
                if token_ids.len() > self.length {
                    token_ids.truncate(self.length);
                } else {
                    token_ids.resize(self.length, self.pad_token_id);
                }
            }

            if let Some(ref mut tokens) = input.tokens {
                if tokens.len() > self.length {
                    tokens.truncate(self.length);
                } else {
                    tokens.resize(self.length, "<PAD>".to_string());
                }
            }

            Ok(input)
        }
    }

    /// Add start and end tokens
    pub struct AddSpecialTokens {
        sos_token_id: usize,
        eos_token_id: usize,
    }

    impl AddSpecialTokens {
        pub fn new(sos_token_id: usize, eos_token_id: usize) -> Self {
            Self {
                sos_token_id,
                eos_token_id,
            }
        }
    }

    impl Transform<TextSequence> for AddSpecialTokens {
        type Output = TextSequence;

        fn transform(&self, mut input: TextSequence) -> Result<Self::Output> {
            if let Some(ref mut token_ids) = input.token_ids {
                token_ids.insert(0, self.sos_token_id);
                token_ids.push(self.eos_token_id);
            }

            if let Some(ref mut tokens) = input.tokens {
                tokens.insert(0, "<SOS>".to_string());
                tokens.push("<EOS>".to_string());
            }

            Ok(input)
        }
    }

    /// Simple n-gram extraction
    pub struct NGrams {
        n: usize,
    }

    impl NGrams {
        pub fn new(n: usize) -> Self {
            assert!(n > 0, "n must be positive");
            Self { n }
        }
    }

    impl Transform<TextSequence> for NGrams {
        type Output = TextSequence;

        fn transform(&self, input: TextSequence) -> Result<Self::Output> {
            let tokens = if let Some(tokens) = input.tokens {
                tokens
            } else {
                Vocabulary::simple_tokenize(&input.text)
            };

            let mut ngrams = Vec::new();
            for window in tokens.windows(self.n) {
                let ngram = window.join("_");
                ngrams.push(ngram);
            }

            let ngram_text = ngrams.join(" ");
            Ok(TextSequence::new(ngram_text).with_tokens(ngrams))
        }
    }

    /// Character-level tokenization
    pub struct CharTokenizer;

    impl Transform<TextSequence> for CharTokenizer {
        type Output = TextSequence;

        fn transform(&self, input: TextSequence) -> Result<Self::Output> {
            let chars: Vec<String> = input.text.chars().map(|c| c.to_string()).collect();
            Ok(input.with_tokens(chars))
        }
    }

    /// Byte Pair Encoding (simplified version)
    pub struct SimpleBPE {
        #[allow(dead_code)]
        vocab_size: usize,
    }

    impl SimpleBPE {
        pub fn new(vocab_size: usize) -> Self {
            Self { vocab_size }
        }
    }

    impl Transform<TextSequence> for SimpleBPE {
        type Output = TextSequence;

        fn transform(&self, input: TextSequence) -> Result<Self::Output> {
            // Simplified BPE - in practice you'd need a trained BPE model
            // For now, just do character-level tokenization with word boundaries
            let mut tokens = Vec::new();

            for word in input.text.split_whitespace() {
                // Add word-level tokens for short words
                if word.len() <= 3 {
                    tokens.push(word.to_string());
                } else {
                    // Split longer words into subwords (simplified)
                    let chars: Vec<char> = word.chars().collect();
                    for chunk in chars.chunks(2) {
                        let subword: String = chunk.iter().collect();
                        tokens.push(subword);
                    }
                }
            }

            Ok(input.with_tokens(tokens))
        }
    }
}

/// Common NLP datasets
pub mod datasets {
    use super::*;

    /// IMDB movie reviews dataset (simplified)
    pub struct IMDB {
        #[allow(dead_code)]
        root: PathBuf,
        #[allow(dead_code)]
        split: String,
        vocabulary: Vocabulary,
        samples: Vec<(String, usize)>, // (text, label)
        transform: Option<Box<dyn Transform<TextSequence, Output = Tensor<f32>>>>,
    }

    impl IMDB {
        /// Create IMDB dataset
        pub fn new<P: AsRef<Path>>(root: P, split: &str) -> Result<Self> {
            let root = root.as_ref().to_path_buf();

            // In a real implementation, you would:
            // 1. Download IMDB dataset from official source
            // 2. Parse the data files
            // 3. Load reviews and sentiment labels

            // For now, create dummy data
            let samples = vec![
                ("This movie is great!".to_string(), 1),          // positive
                ("Terrible film, waste of time.".to_string(), 0), // negative
                ("Amazing cinematography and acting.".to_string(), 1),
                ("Boring and predictable plot.".to_string(), 0),
            ];

            let texts: Vec<String> = samples.iter().map(|(text, _)| text.clone()).collect();
            let mut vocabulary = Vocabulary::new();
            vocabulary.build_from_texts(&texts, 1)?;

            Ok(Self {
                root,
                split: split.to_string(),
                vocabulary,
                samples,
                transform: None,
            })
        }

        /// Set transform
        pub fn with_transform<T>(mut self, transform: T) -> Self
        where
            T: Transform<TextSequence, Output = Tensor<f32>> + 'static,
        {
            self.transform = Some(Box::new(transform));
            self
        }

        /// Get vocabulary reference
        pub fn vocabulary(&self) -> &Vocabulary {
            &self.vocabulary
        }
    }

    impl Dataset for IMDB {
        type Item = (Tensor<f32>, usize);

        fn len(&self) -> usize {
            self.samples.len()
        }

        fn get(&self, index: usize) -> Result<Self::Item> {
            if index >= self.samples.len() {
                return Err(TorshError::IndexError {
                    index,
                    size: self.samples.len(),
                });
            }

            let (ref text, label) = self.samples[index];

            // Tokenize and encode
            let token_ids = self.vocabulary.encode(text);
            let tokens = Vocabulary::simple_tokenize(text);

            let sequence = TextSequence::new(text.clone())
                .with_tokens(tokens)
                .with_token_ids(token_ids);

            let tensor = if let Some(ref transform) = self.transform {
                transform.transform(sequence)?
            } else {
                TokenIdsToTensor.transform(sequence)?
            };

            Ok((tensor, label))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary() {
        let texts = vec![
            "hello world".to_string(),
            "world hello".to_string(),
            "foo bar".to_string(),
        ];

        let mut vocab = Vocabulary::new();
        vocab.build_from_texts(&texts, 1).unwrap();

        // Should have special tokens + unique words
        assert!(vocab.len() >= 6); // 4 special + hello, world, foo, bar

        // Test encoding/decoding
        let ids = vocab.encode("hello world");
        let decoded = vocab.decode(&ids);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_text_sequence() {
        let seq = TextSequence::new("hello world".to_string())
            .with_tokens(vec!["hello".to_string(), "world".to_string()])
            .with_token_ids(vec![1, 2]);

        assert_eq!(seq.len(), 2);
        assert!(!seq.is_empty());
    }

    #[test]
    fn test_text_classification_dataset() {
        let texts = vec![
            "positive example".to_string(),
            "negative example".to_string(),
        ];
        let labels = vec![1, 0];

        let dataset = TextClassificationDataset::new(texts, labels).unwrap();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.num_classes(), 2);

        let (tensor, label) = dataset.get(0).unwrap();
        assert_eq!(label, 1);
        assert!(tensor.ndim() > 0);
    }

    #[test]
    fn test_token_ids_to_tensor() {
        let seq = TextSequence::new("test".to_string()).with_token_ids(vec![1, 2, 3]);

        let transform = TokenIdsToTensor;
        let result = transform.transform(seq).unwrap();

        assert_eq!(result.shape().dims(), &[3]);
        let data = result.to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_text_transforms() {
        use transforms::*;

        let seq = TextSequence::new("Hello, World!".to_string())
            .with_tokens(vec!["Hello,".to_string(), "World!".to_string()]);

        // Test lowercase transform
        let lowercase = ToLowercase;
        let result = lowercase.transform(seq.clone()).unwrap();
        assert_eq!(result.text, "hello, world!");

        // Test punctuation removal
        let remove_punct = RemovePunctuation;
        let result = remove_punct.transform(seq.clone()).unwrap();
        assert_eq!(result.text, "Hello World");

        // Test fixed length
        let seq_with_ids = seq.with_token_ids(vec![1, 2]);
        let fixed_len = FixedLength::new(4, 0);
        let result = fixed_len.transform(seq_with_ids).unwrap();
        assert_eq!(result.token_ids.unwrap(), vec![1, 2, 0, 0]);

        // Test special tokens
        let add_special = AddSpecialTokens::new(100, 101);
        let seq_with_ids = TextSequence::new("test".to_string()).with_token_ids(vec![1, 2]);
        let result = add_special.transform(seq_with_ids).unwrap();
        assert_eq!(result.token_ids.unwrap(), vec![100, 1, 2, 101]);
    }

    #[test]
    fn test_ngrams() {
        use transforms::*;

        let seq = TextSequence::new("the quick brown fox".to_string()).with_tokens(vec![
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
        ]);

        let bigrams = NGrams::new(2);
        let result = bigrams.transform(seq).unwrap();

        let expected_tokens = vec![
            "the_quick".to_string(),
            "quick_brown".to_string(),
            "brown_fox".to_string(),
        ];
        assert_eq!(result.tokens.unwrap(), expected_tokens);
    }

    #[test]
    fn test_imdb_dataset() {
        use datasets::*;

        let dataset = IMDB::new("/tmp", "train").unwrap();
        assert_eq!(dataset.len(), 4);

        let (tensor, label) = dataset.get(0).unwrap();
        assert_eq!(label, 1); // positive
        assert!(tensor.ndim() > 0);
    }

    #[test]
    fn test_char_tokenizer() {
        use transforms::*;

        let seq = TextSequence::new("abc".to_string());
        let char_tokenizer = CharTokenizer;
        let result = char_tokenizer.transform(seq).unwrap();

        assert_eq!(
            result.tokens.unwrap(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_simple_bpe() {
        use transforms::*;

        let seq = TextSequence::new("hello world".to_string());
        let bpe = SimpleBPE::new(1000);
        let result = bpe.transform(seq).unwrap();

        // Should have some form of subword tokenization
        assert!(result.tokens.is_some());
        assert!(!result.tokens.unwrap().is_empty());
    }
}
