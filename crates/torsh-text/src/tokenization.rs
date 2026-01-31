use crate::{Result, TextError};
use scirs2_core::random::{Random, Rng};
use std::cmp::Ordering;
use std::collections::HashMap;
use tokenizers::{
    models::{bpe::BPE, unigram::Unigram, wordpiece::WordPiece},
    normalizers::BertNormalizer,
    pre_tokenizers::{bert::BertPreTokenizer, byte_level::ByteLevel},
    processors::bert::BertProcessing,
    tokenizer::Tokenizer as HFTokenizer,
};
use unicode_segmentation::UnicodeSegmentation;

/// Type alias for complex tensor output from tokenizers
type TensorOutput = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);

/// Tokenizer trait for text tokenization
pub trait Tokenizer: Send + Sync {
    /// Tokenize a string into tokens
    fn tokenize(&self, text: &str) -> Result<Vec<String>>;

    /// Convert tokens to IDs
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Convert IDs back to tokens
    fn decode(&self, ids: &[u32]) -> Result<String>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;
}

/// Simple whitespace tokenizer
pub struct WhitespaceTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
}

impl Default for WhitespaceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl WhitespaceTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("<unk>".to_string(), 1);
        vocab.insert("<bos>".to_string(), 2);
        vocab.insert("<eos>".to_string(), 3);

        let reverse_vocab = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        Self {
            vocab,
            reverse_vocab,
        }
    }

    pub fn from_texts(texts: &[String], min_freq: usize) -> Self {
        let mut word_counts = HashMap::new();

        for text in texts {
            for word in text.split_whitespace() {
                *word_counts.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        let mut vocab = HashMap::new();
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("<unk>".to_string(), 1);
        vocab.insert("<bos>".to_string(), 2);
        vocab.insert("<eos>".to_string(), 3);

        let mut idx = 4;
        for (word, count) in word_counts {
            if count >= min_freq {
                vocab.insert(word, idx);
                idx += 1;
            }
        }

        let reverse_vocab = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        Self {
            vocab,
            reverse_vocab,
        }
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        Ok(text.split_whitespace().map(|s| s.to_string()).collect())
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let tokens = self.tokenize(text)?;
        let mut ids = Vec::new();

        for token in tokens {
            let id = self.vocab.get(&token).unwrap_or(&1); // Use <unk> for unknown tokens
            ids.push(*id);
        }

        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut tokens = Vec::new();

        for &id in ids {
            if let Some(token) = self.reverse_vocab.get(&id) {
                if !["<pad>", "<unk>", "<bos>", "<eos>"].contains(&token.as_str()) {
                    tokens.push(token.clone());
                }
            }
        }

        Ok(tokens.join(" "))
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Character-level tokenizer
pub struct CharTokenizer {
    vocab: HashMap<char, u32>,
    reverse_vocab: HashMap<u32, char>,
}

impl CharTokenizer {
    pub fn new(chars: Option<Vec<char>>) -> Self {
        let mut vocab = HashMap::new();
        vocab.insert('\0', 0); // padding
        vocab.insert('\x01', 1); // unknown
        vocab.insert('\x02', 2); // start
        vocab.insert('\x03', 3); // end

        let chars = chars.unwrap_or_else(|| {
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\""
                .chars()
                .collect()
        });

        let mut idx = 4;
        for ch in chars {
            if !vocab.contains_key(&ch) {
                vocab.insert(ch, idx);
                idx += 1;
            }
        }

        let reverse_vocab = vocab.iter().map(|(&k, &v)| (v, k)).collect();

        Self {
            vocab,
            reverse_vocab,
        }
    }
}

impl Tokenizer for CharTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        Ok(text.chars().map(|c| c.to_string()).collect())
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();

        for ch in text.chars() {
            let id = self.vocab.get(&ch).unwrap_or(&1); // Use unknown token
            ids.push(*id);
        }

        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut chars = Vec::new();

        for &id in ids {
            if let Some(&ch) = self.reverse_vocab.get(&id) {
                if ch != '\0' && ch != '\x01' && ch != '\x02' && ch != '\x03' {
                    chars.push(ch);
                }
            }
        }

        Ok(chars.into_iter().collect())
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Subword tokenizer using Hugging Face tokenizers
pub struct SubwordTokenizer {
    tokenizer: HFTokenizer,
}

impl SubwordTokenizer {
    /// Create a BERT tokenizer
    pub fn bert(vocab_path: &str) -> Result<Self> {
        let mut tokenizer = HFTokenizer::new(
            WordPiece::from_file(vocab_path)
                .build()
                .map_err(|e| TextError::TokenizationError(e.to_string()))?,
        );

        let normalizer = BertNormalizer::new(
            true,        // clean_text
            true,        // handle_chinese_chars
            Some(false), // strip_accents
            true,        // lowercase
        );
        tokenizer.with_normalizer(Some(normalizer));

        tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));

        tokenizer.with_post_processor(Some(BertProcessing::new(
            ("[SEP]".to_string(), 102),
            ("[CLS]".to_string(), 101),
        )));

        Ok(Self { tokenizer })
    }

    /// Create a GPT-2/RoBERTa style tokenizer
    pub fn byte_level_bpe(vocab_path: &str, merges_path: &str) -> Result<Self> {
        let bpe = BPE::from_file(vocab_path, merges_path)
            .build()
            .map_err(|e| TextError::TokenizationError(e.to_string()))?;

        let mut tokenizer = HFTokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));

        Ok(Self { tokenizer })
    }

    /// Create a SentencePiece tokenizer
    pub fn sentencepiece(_model_path: &str) -> Result<Self> {
        // Create a placeholder unigram model - actual implementation would load from file
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("<s>".to_string(), 0.0),
            ("</s>".to_string(), 0.0),
        ];
        let unigram = Unigram::from(vocab, Some(0), false)
            .map_err(|e| TextError::TokenizationError(e.to_string()))?;

        let tokenizer = HFTokenizer::new(unigram);

        Ok(Self { tokenizer })
    }
}

impl Tokenizer for SubwordTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| TextError::TokenizationError(e.to_string()))?;

        Ok(encoding.get_tokens().to_vec())
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| TextError::TokenizationError(e.to_string()))?;

        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| TextError::TokenizationError(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }
}

/// Byte Pair Encoding (BPE) tokenizer
pub struct BPETokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    merges: Vec<(String, String)>,
}

impl Default for BPETokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BPETokenizer {
    /// Create a new BPE tokenizer with empty vocabulary
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("<unk>".to_string(), 1);
        vocab.insert("<bos>".to_string(), 2);
        vocab.insert("<eos>".to_string(), 3);

        let reverse_vocab = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        Self {
            vocab,
            reverse_vocab,
            merges: Vec::new(),
        }
    }

    /// Train BPE from texts
    pub fn from_texts(texts: &[String], vocab_size: usize) -> Result<Self> {
        let mut tokenizer = Self::new();

        // Count character frequencies
        let mut char_counts = HashMap::new();
        let mut word_counts = HashMap::new();

        for text in texts {
            for word in text.split_whitespace() {
                *word_counts.entry(word.to_string()).or_insert(0) += 1;
                for ch in word.chars() {
                    *char_counts.entry(ch.to_string()).or_insert(0) += 1;
                }
            }
        }

        // Add characters to vocabulary
        let mut current_id = 4;
        for (ch, _) in char_counts {
            if !tokenizer.vocab.contains_key(&ch) {
                tokenizer.vocab.insert(ch.clone(), current_id);
                tokenizer.reverse_vocab.insert(current_id, ch);
                current_id += 1;
            }
        }

        // Learn BPE merges
        while tokenizer.vocab.len() < vocab_size {
            let mut pair_counts = HashMap::new();

            for (word, freq) in &word_counts {
                let tokens = Self::get_word_tokens(word);
                for i in 0..tokens.len().saturating_sub(1) {
                    let pair = (tokens[i].clone(), tokens[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += freq;
                }
            }

            if let Some((best_pair, _)) = pair_counts.iter().max_by_key(|(_, &count)| count) {
                let new_token = best_pair.0.clone() + &best_pair.1;
                tokenizer.vocab.insert(new_token.clone(), current_id);
                tokenizer.reverse_vocab.insert(current_id, new_token);
                tokenizer
                    .merges
                    .push((best_pair.0.clone(), best_pair.1.clone()));
                current_id += 1;
            } else {
                break;
            }
        }

        Ok(tokenizer)
    }

    /// Apply BPE algorithm to tokenize text
    fn apply_bpe(&self, word: &str) -> Vec<String> {
        let mut tokens = Self::get_word_tokens(word);

        for (left, right) in &self.merges {
            let mut new_tokens = Vec::new();
            let mut i = 0;

            while i < tokens.len() {
                if i < tokens.len() - 1 && tokens[i] == *left && tokens[i + 1] == *right {
                    new_tokens.push(left.to_string() + right);
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        tokens
    }

    /// Get individual character tokens for a word
    fn get_word_tokens(word: &str) -> Vec<String> {
        word.chars().map(|c| c.to_string()).collect()
    }
}

impl Tokenizer for BPETokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        let mut result = Vec::new();

        for word in text.split_whitespace() {
            let tokens = self.apply_bpe(word);
            result.extend(tokens);
        }

        Ok(result)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let tokens = self.tokenize(text)?;
        let mut ids = Vec::new();

        for token in tokens {
            let id = self.vocab.get(&token).unwrap_or(&1); // Use <unk> for unknown tokens
            ids.push(*id);
        }

        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut tokens = Vec::new();

        for &id in ids {
            if let Some(token) = self.reverse_vocab.get(&id) {
                if !["<pad>", "<unk>", "<bos>", "<eos>"].contains(&token.as_str()) {
                    tokens.push(token.clone());
                }
            }
        }

        Ok(tokens.join(""))
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Text preprocessing utilities
pub mod preprocessing {
    use super::*;
    use regex::Regex;

    /// Clean text by removing extra whitespace and special characters
    pub fn clean_text(text: &str, remove_special: bool) -> String {
        let mut cleaned = text.trim().to_string();

        // Replace multiple spaces with single space
        let space_re =
            Regex::new(r"\s+").expect("clean_text: compile-time constant regex should be valid");
        cleaned = space_re.replace_all(&cleaned, " ").to_string();

        if remove_special {
            let special_re = Regex::new(r"[^\w\s]")
                .expect("clean_text: compile-time constant regex should be valid");
            cleaned = special_re.replace_all(&cleaned, "").to_string();
        }

        cleaned
    }

    /// Normalize text (lowercase, remove accents, etc.)
    pub fn normalize_text(text: &str) -> String {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect()
    }

    /// Split text into sentences
    pub fn split_sentences(text: &str) -> Vec<String> {
        // Simple sentence splitting - could be improved with more sophisticated rules
        let sentence_re = Regex::new(r"[.!?]+")
            .expect("split_sentences: compile-time constant regex should be valid");
        sentence_re
            .split(text)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Split text into words using Unicode word boundaries
    pub fn split_words(text: &str) -> Vec<String> {
        text.unicode_words().map(|s| s.to_string()).collect()
    }
}

/// Advanced tokenization features
pub mod advanced {
    use super::*;

    /// Subword regularization for BPE and Unigram tokenizers
    pub struct SubwordRegularizer {
        dropout_prob: f32,
        use_sampling: bool,
    }

    impl SubwordRegularizer {
        pub fn new(dropout_prob: f32, use_sampling: bool) -> Self {
            Self {
                dropout_prob,
                use_sampling,
            }
        }

        /// Apply subword regularization to merge operations
        pub fn regularize_merges(&self, merges: &[(String, String)]) -> Vec<(String, String)> {
            if !self.use_sampling || self.dropout_prob <= 0.0 {
                return merges.to_vec();
            }

            let mut rng = Random::seed(0);
            merges
                .iter()
                .filter(|_| rng.random::<f32>() > self.dropout_prob)
                .cloned()
                .collect()
        }

        /// Apply sampling to vocabulary during tokenization
        pub fn sample_vocabulary(&self, vocab: &HashMap<String, f32>) -> HashMap<String, f32> {
            if !self.use_sampling || self.dropout_prob <= 0.0 {
                return vocab.clone();
            }

            let mut rng = Random::seed(0);
            vocab
                .iter()
                .filter(|(_, _)| rng.random::<f32>() > self.dropout_prob)
                .map(|(k, v)| (k.clone(), *v))
                .collect()
        }
    }

    /// Enhanced Byte-Level BPE tokenizer with regularization
    pub struct ByteLevelBPETokenizer {
        vocab: HashMap<String, u32>,
        reverse_vocab: HashMap<u32, String>,
        merges: Vec<(String, String)>,
        byte_encoder: HashMap<u8, char>,
        byte_decoder: HashMap<char, u8>,
        regularizer: Option<SubwordRegularizer>,
    }

    impl Default for ByteLevelBPETokenizer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ByteLevelBPETokenizer {
        pub fn new() -> Self {
            let (byte_encoder, byte_decoder) = Self::create_byte_maps();

            let mut vocab = HashMap::new();
            vocab.insert("<|endoftext|>".to_string(), 0);
            vocab.insert("<|unk|>".to_string(), 1);
            vocab.insert("<|pad|>".to_string(), 2);

            let reverse_vocab = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

            Self {
                vocab,
                reverse_vocab,
                merges: Vec::new(),
                byte_encoder,
                byte_decoder,
                regularizer: None,
            }
        }

        pub fn with_regularization(mut self, regularizer: SubwordRegularizer) -> Self {
            self.regularizer = Some(regularizer);
            self
        }

        /// Train byte-level BPE from texts
        pub fn from_texts(texts: &[String], vocab_size: usize) -> Result<Self> {
            let mut tokenizer = Self::new();

            // Initialize vocabulary with all bytes
            let mut current_id = 3;
            for i in 0..256 {
                let byte_char = tokenizer.byte_encoder[&(i as u8)];
                let token = byte_char.to_string();
                if !tokenizer.vocab.contains_key(&token) {
                    tokenizer.vocab.insert(token.clone(), current_id);
                    tokenizer.reverse_vocab.insert(current_id, token);
                    current_id += 1;
                }
            }

            // Count byte pair frequencies
            let mut pair_counts = HashMap::new();
            for text in texts {
                let byte_tokens = tokenizer.text_to_bytes(text);
                for window in byte_tokens.windows(2) {
                    if let [a, b] = window {
                        let pair = (a.clone(), b.clone());
                        *pair_counts.entry(pair).or_insert(0) += 1;
                    }
                }
            }

            // Learn merges
            while tokenizer.vocab.len() < vocab_size {
                if let Some((best_pair, _)) = pair_counts.iter().max_by_key(|(_, &count)| count) {
                    let best_pair = best_pair.clone(); // Clone to avoid borrow checker issues
                    let new_token = best_pair.0.clone() + &best_pair.1;
                    tokenizer.vocab.insert(new_token.clone(), current_id);
                    tokenizer.reverse_vocab.insert(current_id, new_token);
                    tokenizer
                        .merges
                        .push((best_pair.0.clone(), best_pair.1.clone()));
                    current_id += 1;

                    // Update pair counts for next iteration
                    pair_counts.remove(&best_pair);
                } else {
                    break;
                }
            }

            Ok(tokenizer)
        }

        /// Convert text to byte tokens
        fn text_to_bytes(&self, text: &str) -> Vec<String> {
            text.bytes()
                .map(|b| self.byte_encoder[&b].to_string())
                .collect()
        }

        /// Apply BPE merges with optional regularization
        fn apply_bpe(&self, tokens: Vec<String>) -> Vec<String> {
            let merges = if let Some(reg) = &self.regularizer {
                reg.regularize_merges(&self.merges)
            } else {
                self.merges.clone()
            };

            let mut current_tokens = tokens;
            for (left, right) in merges {
                let mut new_tokens = Vec::new();
                let mut i = 0;

                while i < current_tokens.len() {
                    if i < current_tokens.len() - 1
                        && current_tokens[i] == left
                        && current_tokens[i + 1] == right
                    {
                        new_tokens.push(left.to_string() + &right);
                        i += 2;
                    } else {
                        new_tokens.push(current_tokens[i].clone());
                        i += 1;
                    }
                }
                current_tokens = new_tokens;
            }

            current_tokens
        }

        /// Create byte encoding maps
        fn create_byte_maps() -> (HashMap<u8, char>, HashMap<char, u8>) {
            let mut byte_encoder = HashMap::new();
            let mut byte_decoder = HashMap::new();

            // Printable ASCII characters
            for i in 33..=126 {
                byte_encoder.insert(i, i as char);
                byte_decoder.insert(i as char, i);
            }

            // Extended characters for non-printable bytes
            let mut char_code = 256;
            for i in 0..=255u8 {
                if !byte_encoder.contains_key(&i) {
                    let ch = char::from_u32(char_code)
                        .expect("char_code should be valid unicode scalar value");
                    byte_encoder.insert(i, ch);
                    byte_decoder.insert(ch, i);
                    char_code += 1;
                }
            }

            (byte_encoder, byte_decoder)
        }
    }

    impl Tokenizer for ByteLevelBPETokenizer {
        fn tokenize(&self, text: &str) -> Result<Vec<String>> {
            let byte_tokens = self.text_to_bytes(text);
            Ok(self.apply_bpe(byte_tokens))
        }

        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            let tokens = self.tokenize(text)?;
            let mut ids = Vec::new();

            for token in tokens {
                let id = self.vocab.get(&token).unwrap_or(&1);
                ids.push(*id);
            }

            Ok(ids)
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            let mut byte_tokens = Vec::new();

            for &id in ids {
                if let Some(token) = self.reverse_vocab.get(&id) {
                    if !["<|endoftext|>", "<|unk|>", "<|pad|>"].contains(&token.as_str()) {
                        byte_tokens.extend(token.chars());
                    }
                }
            }

            // Convert back to bytes then to string
            let bytes: Vec<u8> = byte_tokens
                .iter()
                .filter_map(|&ch| self.byte_decoder.get(&ch))
                .copied()
                .collect();

            String::from_utf8(bytes).map_err(|e| TextError::TokenizationError(e.to_string()))
        }

        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }
    }

    /// Unigram Language Model tokenizer
    pub struct UnigramTokenizer {
        vocab: HashMap<String, f32>, // token -> log probability
        id_to_token: HashMap<u32, String>,
        token_to_id: HashMap<String, u32>,
        unk_id: u32,
        regularizer: Option<SubwordRegularizer>,
    }

    #[derive(Debug, PartialEq)]
    struct TokenCandidate {
        token: String,
        score: f32,
        start: usize,
        end: usize,
    }

    impl Eq for TokenCandidate {}

    impl Ord for TokenCandidate {
        fn cmp(&self, other: &Self) -> Ordering {
            self.score
                .partial_cmp(&other.score)
                .unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for TokenCandidate {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Default for UnigramTokenizer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl UnigramTokenizer {
        pub fn new() -> Self {
            let mut vocab = HashMap::new();
            let mut token_to_id = HashMap::new();
            let mut id_to_token = HashMap::new();

            // Special tokens
            vocab.insert("<unk>".to_string(), -10.0);
            vocab.insert("<s>".to_string(), -10.0);
            vocab.insert("</s>".to_string(), -10.0);

            token_to_id.insert("<unk>".to_string(), 0);
            token_to_id.insert("<s>".to_string(), 1);
            token_to_id.insert("</s>".to_string(), 2);

            id_to_token.insert(0, "<unk>".to_string());
            id_to_token.insert(1, "<s>".to_string());
            id_to_token.insert(2, "</s>".to_string());

            Self {
                vocab,
                id_to_token,
                token_to_id,
                unk_id: 0,
                regularizer: None,
            }
        }

        pub fn with_regularization(mut self, regularizer: SubwordRegularizer) -> Self {
            self.regularizer = Some(regularizer);
            self
        }

        /// Train Unigram model from texts
        pub fn from_texts(texts: &[String], vocab_size: usize) -> Result<Self> {
            let mut tokenizer = Self::new();

            // Initialize with character vocabulary
            let mut char_counts = HashMap::new();
            let mut total_chars = 0;

            for text in texts {
                for ch in text.chars() {
                    *char_counts.entry(ch.to_string()).or_insert(0) += 1;
                    total_chars += 1;
                }
            }

            let mut current_id = 3;
            for (ch, count) in char_counts {
                let prob = (count as f32 / total_chars as f32).ln();
                tokenizer.vocab.insert(ch.clone(), prob);
                tokenizer.token_to_id.insert(ch.clone(), current_id);
                tokenizer.id_to_token.insert(current_id, ch);
                current_id += 1;
            }

            // Build subword vocabulary using EM algorithm (simplified)
            let max_len = 10; // Maximum subword length
            for text in texts {
                for window_size in 2..=max_len.min(text.len()) {
                    for start in 0..=text.len().saturating_sub(window_size) {
                        let substr = text[start..start + window_size].to_string();
                        if !tokenizer.vocab.contains_key(&substr)
                            && tokenizer.vocab.len() < vocab_size
                        {
                            // Simple frequency-based scoring
                            let freq = texts
                                .iter()
                                .map(|t| t.matches(&substr).count())
                                .sum::<usize>();

                            if freq > 1 {
                                let prob = (freq as f32 / texts.len() as f32).ln();
                                tokenizer.vocab.insert(substr.clone(), prob);
                                tokenizer.token_to_id.insert(substr.clone(), current_id);
                                tokenizer.id_to_token.insert(current_id, substr);
                                current_id += 1;
                            }
                        }
                    }
                }
            }

            Ok(tokenizer)
        }

        /// Tokenize using Viterbi algorithm
        fn viterbi_tokenize(&self, text: &str) -> Vec<String> {
            if text.is_empty() {
                return Vec::new();
            }

            let vocab = if let Some(reg) = &self.regularizer {
                reg.sample_vocabulary(&self.vocab)
            } else {
                self.vocab.clone()
            };

            let chars: Vec<char> = text.chars().collect();
            let n = chars.len();

            // Dynamic programming arrays
            let mut best_score = vec![f32::NEG_INFINITY; n + 1];
            let mut best_tokens = vec![Vec::new(); n + 1];

            best_score[0] = 0.0;

            for i in 0..n {
                if best_score[i] == f32::NEG_INFINITY {
                    continue;
                }

                // Try all possible tokens starting at position i
                for end in i + 1..=n {
                    let substr: String = chars[i..end].iter().collect();

                    let score = vocab
                        .get(&substr)
                        .copied()
                        .unwrap_or_else(|| vocab.get("<unk>").copied().unwrap_or(-20.0));

                    let new_score = best_score[i] + score;

                    if new_score > best_score[end] {
                        best_score[end] = new_score;
                        let mut new_tokens = best_tokens[i].clone();
                        new_tokens.push(substr);
                        best_tokens[end] = new_tokens;
                    }
                }
            }

            best_tokens[n].clone()
        }
    }

    impl Tokenizer for UnigramTokenizer {
        fn tokenize(&self, text: &str) -> Result<Vec<String>> {
            Ok(self.viterbi_tokenize(text))
        }

        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            let tokens = self.tokenize(text)?;
            let mut ids = Vec::new();

            for token in tokens {
                let id = self.token_to_id.get(&token).unwrap_or(&self.unk_id);
                ids.push(*id);
            }

            Ok(ids)
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            let mut tokens = Vec::new();

            for &id in ids {
                if let Some(token) = self.id_to_token.get(&id) {
                    if !["<s>", "</s>"].contains(&token.as_str()) {
                        if token == "<unk>" {
                            tokens.push("�".to_string()); // Unicode replacement character
                        } else {
                            tokens.push(token.clone());
                        }
                    }
                }
            }

            Ok(tokens.join(""))
        }

        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }
    }

    /// Custom tokenizer that can combine multiple tokenization strategies
    pub struct CustomTokenizer {
        primary: Box<dyn Tokenizer>,
        fallback: Option<Box<dyn Tokenizer>>,
        preprocessing: Option<fn(&str) -> String>,
        postprocessing: Option<fn(Vec<String>) -> Vec<String>>,
    }

    impl CustomTokenizer {
        pub fn new(primary: Box<dyn Tokenizer>) -> Self {
            Self {
                primary,
                fallback: None,
                preprocessing: None,
                postprocessing: None,
            }
        }

        pub fn with_fallback(mut self, fallback: Box<dyn Tokenizer>) -> Self {
            self.fallback = Some(fallback);
            self
        }

        pub fn with_preprocessing(mut self, preprocessing: fn(&str) -> String) -> Self {
            self.preprocessing = Some(preprocessing);
            self
        }

        pub fn with_postprocessing(
            mut self,
            postprocessing: fn(Vec<String>) -> Vec<String>,
        ) -> Self {
            self.postprocessing = Some(postprocessing);
            self
        }
    }

    impl Tokenizer for CustomTokenizer {
        fn tokenize(&self, text: &str) -> Result<Vec<String>> {
            let processed_text = if let Some(preprocess) = self.preprocessing {
                preprocess(text)
            } else {
                text.to_string()
            };

            let tokens = match self.primary.tokenize(&processed_text) {
                Ok(tokens) => tokens,
                Err(_) => {
                    if let Some(fallback) = &self.fallback {
                        fallback.tokenize(&processed_text)?
                    } else {
                        return Err(TextError::TokenizationError(
                            "Primary tokenization failed and no fallback provided".to_string(),
                        ));
                    }
                }
            };

            let final_tokens = if let Some(postprocess) = self.postprocessing {
                postprocess(tokens)
            } else {
                tokens
            };

            Ok(final_tokens)
        }

        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            let tokens = self.tokenize(text)?;
            // Use primary tokenizer for encoding
            let token_text = tokens.join(" ");
            self.primary.encode(&token_text)
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            self.primary.decode(ids)
        }

        fn vocab_size(&self) -> usize {
            self.primary.vocab_size()
        }
    }

    /// Fast tokenizer wrapper with caching and optimizations
    pub struct FastTokenizer {
        inner: Box<dyn Tokenizer>,
        cache: std::sync::Mutex<HashMap<String, Vec<String>>>,
        cache_size_limit: usize,
    }

    impl FastTokenizer {
        pub fn new(tokenizer: Box<dyn Tokenizer>) -> Self {
            Self {
                inner: tokenizer,
                cache: std::sync::Mutex::new(HashMap::new()),
                cache_size_limit: 10000,
            }
        }

        pub fn with_cache_limit(mut self, limit: usize) -> Self {
            self.cache_size_limit = limit;
            self
        }

        /// Clear the tokenization cache
        pub fn clear_cache(&self) {
            if let Ok(mut cache) = self.cache.lock() {
                cache.clear();
            }
        }
    }

    impl Tokenizer for FastTokenizer {
        fn tokenize(&self, text: &str) -> Result<Vec<String>> {
            // Check cache first
            if let Ok(cache) = self.cache.lock() {
                if let Some(cached_tokens) = cache.get(text) {
                    return Ok(cached_tokens.clone());
                }
            }

            // Tokenize and cache result
            let tokens = self.inner.tokenize(text)?;

            if let Ok(mut cache) = self.cache.lock() {
                if cache.len() >= self.cache_size_limit {
                    // Simple LRU: clear cache when full
                    cache.clear();
                }
                cache.insert(text.to_string(), tokens.clone());
            }

            Ok(tokens)
        }

        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            self.inner.encode(text)
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            self.inner.decode(ids)
        }

        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }
    }
}

/// Unified tokenizer API and memory optimizations
pub mod unified {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Unified configuration for all tokenizer types
    #[derive(Debug, Clone)]
    pub struct TokenizerConfig {
        pub tokenizer_type: TokenizerType,
        pub vocab_size: usize,
        pub max_length: Option<usize>,
        pub padding: bool,
        pub truncation: bool,
        pub add_special_tokens: bool,
        pub return_attention_mask: bool,
        pub return_token_type_ids: bool,
        pub model_path: Option<String>,
        pub vocab_path: Option<String>,
        pub merges_path: Option<String>,
        pub special_tokens: HashMap<String, u32>,
        pub regularization: Option<RegularizationConfig>,
    }

    #[derive(Debug, Clone)]
    pub enum TokenizerType {
        Whitespace,
        Character,
        BPE,
        ByteLevelBPE,
        WordPiece,
        SentencePiece,
        Unigram,
        Custom(String),
    }

    #[derive(Debug, Clone)]
    pub struct RegularizationConfig {
        pub dropout_prob: f32,
        pub use_sampling: bool,
    }

    impl Default for TokenizerConfig {
        fn default() -> Self {
            Self {
                tokenizer_type: TokenizerType::Whitespace,
                vocab_size: 30000,
                max_length: Some(512),
                padding: true,
                truncation: true,
                add_special_tokens: true,
                return_attention_mask: true,
                return_token_type_ids: false,
                model_path: None,
                vocab_path: None,
                merges_path: None,
                special_tokens: HashMap::new(),
                regularization: None,
            }
        }
    }

    /// Tokenization result with all possible outputs
    #[derive(Debug, Clone)]
    pub struct TokenizationResult {
        pub input_ids: Vec<u32>,
        pub attention_mask: Option<Vec<u32>>,
        pub token_type_ids: Option<Vec<u32>>,
        pub tokens: Vec<String>,
        pub special_tokens_mask: Option<Vec<u32>>,
        pub offset_mapping: Option<Vec<(usize, usize)>>,
    }

    impl TokenizationResult {
        pub fn new(input_ids: Vec<u32>, tokens: Vec<String>) -> Self {
            Self {
                input_ids,
                attention_mask: None,
                token_type_ids: None,
                tokens,
                special_tokens_mask: None,
                offset_mapping: None,
            }
        }

        pub fn with_attention_mask(mut self, mask: Vec<u32>) -> Self {
            self.attention_mask = Some(mask);
            self
        }

        pub fn with_token_type_ids(mut self, type_ids: Vec<u32>) -> Self {
            self.token_type_ids = Some(type_ids);
            self
        }

        pub fn with_special_tokens_mask(mut self, mask: Vec<u32>) -> Self {
            self.special_tokens_mask = Some(mask);
            self
        }

        pub fn with_offset_mapping(mut self, offsets: Vec<(usize, usize)>) -> Self {
            self.offset_mapping = Some(offsets);
            self
        }
    }

    /// Unified tokenizer interface
    pub trait UnifiedTokenizer: Send + Sync {
        /// Tokenize a single text
        fn tokenize_single(
            &self,
            text: &str,
            config: &TokenizerConfig,
        ) -> Result<TokenizationResult>;

        /// Tokenize multiple texts efficiently
        fn tokenize_batch(
            &self,
            texts: &[String],
            config: &TokenizerConfig,
        ) -> Result<Vec<TokenizationResult>>;

        /// Encode text to tensor-ready format
        fn encode_batch(&self, texts: &[String], config: &TokenizerConfig)
            -> Result<BatchEncoding>;

        /// Decode token IDs back to text
        fn decode(&self, token_ids: &[u32]) -> Result<String>;

        /// Decode multiple sequences
        fn decode_batch(&self, token_ids_batch: &[Vec<u32>]) -> Result<Vec<String>>;

        /// Get vocabulary size
        fn vocab_size(&self) -> usize;

        /// Get special token IDs
        fn special_tokens(&self) -> &HashMap<String, u32>;

        /// Save tokenizer state
        fn save(&self, path: &str) -> Result<()>;

        /// Load tokenizer state
        fn load(path: &str) -> Result<Self>
        where
            Self: Sized;
    }

    /// Batch encoding result optimized for tensor operations
    #[derive(Debug, Clone)]
    pub struct BatchEncoding {
        pub input_ids: Vec<Vec<u32>>,
        pub attention_mask: Option<Vec<Vec<u32>>>,
        pub token_type_ids: Option<Vec<Vec<u32>>>,
        pub batch_size: usize,
        pub max_length: usize,
    }

    impl BatchEncoding {
        pub fn new(input_ids: Vec<Vec<u32>>) -> Self {
            let batch_size = input_ids.len();
            let max_length = input_ids.iter().map(|seq| seq.len()).max().unwrap_or(0);

            Self {
                input_ids,
                attention_mask: None,
                token_type_ids: None,
                batch_size,
                max_length,
            }
        }

        /// Convert to tensor-ready flat vectors with padding
        pub fn to_tensors(&self, pad_token_id: u32) -> Result<TensorOutput> {
            let mut padded_ids = Vec::with_capacity(self.batch_size * self.max_length);
            let mut padded_mask = if self.attention_mask.is_some() {
                Some(Vec::with_capacity(self.batch_size * self.max_length))
            } else {
                None
            };
            let mut padded_type_ids = if self.token_type_ids.is_some() {
                Some(Vec::with_capacity(self.batch_size * self.max_length))
            } else {
                None
            };

            for i in 0..self.batch_size {
                let seq = &self.input_ids[i];
                let seq_len = seq.len();

                // Add sequence tokens
                padded_ids.extend_from_slice(seq);

                // Pad to max_length
                for _ in seq_len..self.max_length {
                    padded_ids.push(pad_token_id);
                }

                // Handle attention mask
                if let (Some(ref mask_data), Some(ref mut padded_mask)) =
                    (&self.attention_mask, &mut padded_mask)
                {
                    let mask_seq = &mask_data[i];
                    padded_mask.extend_from_slice(mask_seq);

                    // Pad mask with 0s
                    for _ in seq_len..self.max_length {
                        padded_mask.push(0);
                    }
                }

                // Handle token type IDs
                if let (Some(ref type_data), Some(ref mut padded_type_ids)) =
                    (&self.token_type_ids, &mut padded_type_ids)
                {
                    let type_seq = &type_data[i];
                    padded_type_ids.extend_from_slice(type_seq);

                    // Pad type IDs with 0s
                    for _ in seq_len..self.max_length {
                        padded_type_ids.push(0);
                    }
                }
            }

            Ok((padded_ids, padded_mask, padded_type_ids))
        }
    }

    /// Memory-efficient unified tokenizer implementation
    pub struct EfficientUnifiedTokenizer {
        inner: Box<dyn Tokenizer>,
        config: TokenizerConfig,
        special_tokens: HashMap<String, u32>,
        cache: std::sync::Mutex<HashMap<String, Vec<u32>>>,
    }

    impl EfficientUnifiedTokenizer {
        pub fn new(config: TokenizerConfig) -> Result<Self> {
            let inner: Box<dyn Tokenizer> = match config.tokenizer_type {
                TokenizerType::Whitespace => Box::new(WhitespaceTokenizer::new()),
                TokenizerType::Character => Box::new(CharTokenizer::new(None)),
                TokenizerType::BPE => Box::new(BPETokenizer::new()),
                TokenizerType::ByteLevelBPE => Box::new(advanced::ByteLevelBPETokenizer::new()),
                TokenizerType::Unigram => Box::new(advanced::UnigramTokenizer::new()),
                _ => {
                    return Err(TextError::TokenizationError(
                        "Unsupported tokenizer type".to_string(),
                    ))
                }
            };

            // Create cache with reasonable size
            let cache = std::sync::Mutex::new(HashMap::new());

            Ok(Self {
                inner,
                special_tokens: config.special_tokens.clone(),
                config,
                cache,
            })
        }

        pub fn from_pretrained(_model_name: &str) -> Result<Self> {
            // Load configuration from model registry or files
            let config = TokenizerConfig::default(); // Simplified
            Self::new(config)
        }

        /// Apply padding and truncation
        fn apply_padding_truncation(&self, mut tokens: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
            let max_length = self.config.max_length.unwrap_or(512);
            let mut attention_mask = vec![1u32; tokens.len()];

            // Truncation
            if self.config.truncation && tokens.len() > max_length {
                tokens.truncate(max_length);
                attention_mask.truncate(max_length);
            }

            // Padding
            if self.config.padding && tokens.len() < max_length {
                let pad_token_id = self.special_tokens.get("pad").copied().unwrap_or(0);
                let padding_length = max_length - tokens.len();

                tokens.extend(vec![pad_token_id; padding_length]);
                attention_mask.extend(vec![0u32; padding_length]);
            }

            (tokens, attention_mask)
        }

        /// Add special tokens if configured
        fn add_special_tokens(&self, mut tokens: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
            let mut special_mask = vec![0u32; tokens.len()];

            if self.config.add_special_tokens {
                // Add BOS token
                if let Some(&bos_id) = self.special_tokens.get("bos") {
                    tokens.insert(0, bos_id);
                    special_mask.insert(0, 1);
                }

                // Add EOS token
                if let Some(&eos_id) = self.special_tokens.get("eos") {
                    tokens.push(eos_id);
                    special_mask.push(1);
                }
            }

            (tokens, special_mask)
        }
    }

    impl UnifiedTokenizer for EfficientUnifiedTokenizer {
        fn tokenize_single(
            &self,
            text: &str,
            config: &TokenizerConfig,
        ) -> Result<TokenizationResult> {
            // Check cache first
            if let Ok(cache) = self.cache.lock() {
                if let Some(cached_tokens) = cache.get(text) {
                    let tokens = self.inner.tokenize(text)?;
                    return Ok(TokenizationResult::new(cached_tokens.clone(), tokens));
                }
            }

            // Tokenize
            let tokens = self.inner.tokenize(text)?;
            let token_ids = self.inner.encode(text)?;

            // Apply special tokens
            let (token_ids, special_mask) = self.add_special_tokens(token_ids);

            // Apply padding and truncation
            let (token_ids, attention_mask) = self.apply_padding_truncation(token_ids);

            // Cache result
            if let Ok(mut cache) = self.cache.lock() {
                cache.insert(text.to_string(), token_ids.clone());
            }

            let mut result = TokenizationResult::new(token_ids, tokens);

            if config.return_attention_mask {
                result = result.with_attention_mask(attention_mask);
            }

            if config.add_special_tokens {
                result = result.with_special_tokens_mask(special_mask);
            }

            Ok(result)
        }

        fn tokenize_batch(
            &self,
            texts: &[String],
            config: &TokenizerConfig,
        ) -> Result<Vec<TokenizationResult>> {
            let mut results = Vec::with_capacity(texts.len());

            for text in texts {
                results.push(self.tokenize_single(text, config)?);
            }

            Ok(results)
        }

        fn encode_batch(
            &self,
            texts: &[String],
            config: &TokenizerConfig,
        ) -> Result<BatchEncoding> {
            let results = self.tokenize_batch(texts, config)?;

            let input_ids: Vec<Vec<u32>> = results.iter().map(|r| r.input_ids.clone()).collect();
            let mut encoding = BatchEncoding::new(input_ids);

            if config.return_attention_mask {
                let attention_masks: Vec<Vec<u32>> = results
                    .iter()
                    .map(|r| r.attention_mask.clone().unwrap_or_default())
                    .collect();
                encoding.attention_mask = Some(attention_masks);
            }

            if config.return_token_type_ids {
                let token_type_ids: Vec<Vec<u32>> = results
                    .iter()
                    .map(|r| r.token_type_ids.clone().unwrap_or_default())
                    .collect();
                encoding.token_type_ids = Some(token_type_ids);
            }

            Ok(encoding)
        }

        fn decode(&self, token_ids: &[u32]) -> Result<String> {
            self.inner.decode(token_ids)
        }

        fn decode_batch(&self, token_ids_batch: &[Vec<u32>]) -> Result<Vec<String>> {
            let mut results = Vec::with_capacity(token_ids_batch.len());

            for token_ids in token_ids_batch {
                results.push(self.decode(token_ids)?);
            }

            Ok(results)
        }

        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }

        fn special_tokens(&self) -> &HashMap<String, u32> {
            &self.special_tokens
        }

        fn save(&self, _path: &str) -> Result<()> {
            // Implementation for saving tokenizer state
            Ok(())
        }

        fn load(_path: &str) -> Result<Self> {
            // Implementation for loading tokenizer state
            Err(TextError::TokenizationError(
                "Load not implemented".to_string(),
            ))
        }
    }

    /// Factory for creating unified tokenizers
    pub struct TokenizerFactory {
        _memory_pool: Arc<()>, // Simplified for now
    }

    impl TokenizerFactory {
        pub fn new() -> Self {
            Self {
                _memory_pool: Arc::new(()),
            }
        }

        /// Create a tokenizer from configuration
        pub fn create(&self, config: TokenizerConfig) -> Result<EfficientUnifiedTokenizer> {
            EfficientUnifiedTokenizer::new(config)
        }

        /// Create a tokenizer from pretrained model
        pub fn from_pretrained(&self, model_name: &str) -> Result<EfficientUnifiedTokenizer> {
            EfficientUnifiedTokenizer::from_pretrained(model_name)
        }

        /// Create configuration for common tokenizer types
        pub fn bert_config() -> TokenizerConfig {
            let mut config = TokenizerConfig {
                tokenizer_type: TokenizerType::WordPiece,
                add_special_tokens: true,
                return_attention_mask: true,
                return_token_type_ids: true,
                ..Default::default()
            };

            // Add BERT special tokens
            config.special_tokens.insert("pad".to_string(), 0);
            config.special_tokens.insert("unk".to_string(), 100);
            config.special_tokens.insert("cls".to_string(), 101);
            config.special_tokens.insert("sep".to_string(), 102);
            config.special_tokens.insert("mask".to_string(), 103);

            config
        }

        pub fn gpt2_config() -> TokenizerConfig {
            let mut config = TokenizerConfig {
                tokenizer_type: TokenizerType::ByteLevelBPE,
                add_special_tokens: true,
                return_attention_mask: true,
                return_token_type_ids: false,
                ..Default::default()
            };

            // Add GPT-2 special tokens
            config.special_tokens.insert("unk".to_string(), 0);
            config.special_tokens.insert("bos".to_string(), 1);
            config.special_tokens.insert("eos".to_string(), 2);
            config.special_tokens.insert("pad".to_string(), 3);

            config
        }

        pub fn t5_config() -> TokenizerConfig {
            let mut config = TokenizerConfig {
                tokenizer_type: TokenizerType::SentencePiece,
                add_special_tokens: true,
                return_attention_mask: true,
                return_token_type_ids: false,
                ..Default::default()
            };

            // Add T5 special tokens
            config.special_tokens.insert("pad".to_string(), 0);
            config.special_tokens.insert("eos".to_string(), 1);
            config.special_tokens.insert("unk".to_string(), 2);

            config
        }
    }

    impl Default for TokenizerFactory {
        fn default() -> Self {
            Self::new()
        }
    }
}
