use crate::{Result, TextError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Vocabulary for text processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_tokens: SpecialTokens,
    unk_token: String,
    pad_token: String,
}

/// Special tokens used in text processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub pad: String,
    pub unk: String,
    pub bos: String,
    pub eos: String,
    pub sep: String,
    pub cls: String,
    pub mask: String,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            pad: "<pad>".to_string(),
            unk: "<unk>".to_string(),
            bos: "<bos>".to_string(),
            eos: "<eos>".to_string(),
            sep: "<sep>".to_string(),
            cls: "<cls>".to_string(),
            mask: "<mask>".to_string(),
        }
    }
}

impl Vocabulary {
    /// Create a new vocabulary
    pub fn new(special_tokens: Option<SpecialTokens>) -> Self {
        let special_tokens = special_tokens.unwrap_or_default();
        let mut vocab = Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: special_tokens.clone(),
            unk_token: special_tokens.unk.clone(),
            pad_token: special_tokens.pad.clone(),
        };

        // Add special tokens
        vocab.add_token(&special_tokens.pad);
        vocab.add_token(&special_tokens.unk);
        vocab.add_token(&special_tokens.bos);
        vocab.add_token(&special_tokens.eos);
        vocab.add_token(&special_tokens.sep);
        vocab.add_token(&special_tokens.cls);
        vocab.add_token(&special_tokens.mask);

        vocab
    }

    /// Build vocabulary from texts
    pub fn from_texts(texts: &[String], min_freq: usize, max_size: Option<usize>) -> Self {
        let mut word_counts = HashMap::new();

        // Count word frequencies
        for text in texts {
            for word in text.split_whitespace() {
                *word_counts.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Sort by frequency
        let mut word_freq: Vec<_> = word_counts.into_iter().collect();
        word_freq.sort_by(|a, b| b.1.cmp(&a.1));

        // Create vocabulary
        let mut vocab = Self::new(None);

        for (word, freq) in word_freq {
            if freq < min_freq {
                break;
            }
            if let Some(max) = max_size {
                if vocab.size() >= max {
                    break;
                }
            }
            vocab.add_token(&word);
        }

        vocab
    }

    /// Add a token to the vocabulary
    pub fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }

        let id = self.token_to_id.len() as u32;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        id
    }

    /// Get token ID
    pub fn token_to_id(&self, token: &str) -> u32 {
        self.token_to_id
            .get(token)
            .copied()
            .unwrap_or_else(|| self.token_to_id[&self.unk_token])
    }

    /// Get token from ID
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Convert tokens to IDs
    pub fn tokens_to_ids(&self, tokens: &[String]) -> Vec<u32> {
        tokens.iter().map(|t| self.token_to_id(t)).collect()
    }

    /// Convert IDs to tokens
    pub fn ids_to_tokens(&self, ids: &[u32]) -> Vec<String> {
        ids.iter()
            .filter_map(|&id| self.id_to_token(id).map(|s| s.to_string()))
            .collect()
    }

    /// Get vocabulary size
    pub fn size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Get vocabulary size (alias for size())
    pub fn len(&self) -> usize {
        self.size()
    }

    /// Check if vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    /// Get token ID (alias for token_to_id())
    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get special token IDs
    pub fn get_special_token_ids(&self) -> HashMap<String, u32> {
        let mut special_ids = HashMap::new();
        special_ids.insert(
            "pad".to_string(),
            self.token_to_id(&self.special_tokens.pad),
        );
        special_ids.insert(
            "unk".to_string(),
            self.token_to_id(&self.special_tokens.unk),
        );
        special_ids.insert(
            "bos".to_string(),
            self.token_to_id(&self.special_tokens.bos),
        );
        special_ids.insert(
            "eos".to_string(),
            self.token_to_id(&self.special_tokens.eos),
        );
        special_ids.insert(
            "sep".to_string(),
            self.token_to_id(&self.special_tokens.sep),
        );
        special_ids.insert(
            "cls".to_string(),
            self.token_to_id(&self.special_tokens.cls),
        );
        special_ids.insert(
            "mask".to_string(),
            self.token_to_id(&self.special_tokens.mask),
        );
        special_ids
    }

    /// Save vocabulary to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self).map_err(|e| TextError::Other(e.into()))
    }

    /// Load vocabulary from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        serde_json::from_reader(file).map_err(|e| TextError::Other(e.into()))
    }

    /// Save vocabulary in simple text format (one token per line)
    pub fn save_txt<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = File::create(path)?;

        // Sort by ID to maintain order
        let mut tokens: Vec<_> = self.id_to_token.iter().collect();
        tokens.sort_by_key(|(id, _)| *id);

        for (_, token) in tokens {
            writeln!(file, "{token}")?;
        }

        Ok(())
    }

    /// Load vocabulary from simple text format
    pub fn load_txt<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut vocab = Self::new(None);
        vocab.token_to_id.clear();
        vocab.id_to_token.clear();

        for (idx, line) in reader.lines().enumerate() {
            let token = line?;
            vocab.token_to_id.insert(token.clone(), idx as u32);
            vocab.id_to_token.insert(idx as u32, token);
        }

        Ok(vocab)
    }

    /// Merge another vocabulary into this one
    pub fn merge(&mut self, other: &Vocabulary) -> Result<()> {
        for token in other.token_to_id.keys() {
            // Skip special tokens to avoid duplicates
            if !self.is_special_token(token) {
                self.add_token(token);
            }
        }
        Ok(())
    }

    /// Check if a token is a special token
    fn is_special_token(&self, token: &str) -> bool {
        token == self.special_tokens.pad
            || token == self.special_tokens.unk
            || token == self.special_tokens.bos
            || token == self.special_tokens.eos
            || token == self.special_tokens.sep
            || token == self.special_tokens.cls
            || token == self.special_tokens.mask
    }

    /// Create a merged vocabulary from multiple vocabularies
    pub fn merge_vocabularies(vocabs: &[&Vocabulary]) -> Result<Self> {
        if vocabs.is_empty() {
            return Ok(Self::new(None));
        }

        let mut merged = vocabs[0].clone();
        for vocab in &vocabs[1..] {
            merged.merge(vocab)?;
        }
        Ok(merged)
    }
}

/// Subword vocabulary utilities
pub mod subword {
    use super::*;

    /// Byte Pair Encoding (BPE) vocabulary
    pub struct BPEVocabulary {
        vocab: Vocabulary,
        merges: Vec<(String, String)>,
    }

    impl BPEVocabulary {
        pub fn new(vocab: Vocabulary, merges: Vec<(String, String)>) -> Self {
            Self { vocab, merges }
        }

        /// Learn BPE from texts
        pub fn learn_bpe(texts: &[String], vocab_size: usize) -> Result<Self> {
            // Initialize character-level vocabulary
            let mut vocab = Vocabulary::new(None);
            let mut word_freq = HashMap::new();

            // Count word frequencies and initialize character vocabulary
            for text in texts {
                for word in text.split_whitespace() {
                    *word_freq.entry(word.to_string()).or_insert(0) += 1;
                    // Add individual characters to vocab
                    for ch in word.chars() {
                        vocab.add_token(&ch.to_string());
                    }
                }
            }

            let mut merges = Vec::new();
            let mut current_vocab_size = vocab.size();

            while current_vocab_size < vocab_size {
                // Find the most frequent pair
                let mut pair_counts = HashMap::new();

                for (word, freq) in &word_freq {
                    let tokens = Self::get_word_tokens(word);
                    for i in 0..tokens.len().saturating_sub(1) {
                        let pair = (tokens[i].clone(), tokens[i + 1].clone());
                        *pair_counts.entry(pair).or_insert(0) += freq;
                    }
                }

                if let Some((best_pair, _)) = pair_counts.iter().max_by_key(|(_, &count)| count) {
                    let new_token = format!("{}{}", best_pair.0, best_pair.1);
                    vocab.add_token(&new_token);
                    merges.push((best_pair.0.clone(), best_pair.1.clone()));
                    current_vocab_size += 1;
                } else {
                    break;
                }
            }

            Ok(Self { vocab, merges })
        }

        /// Apply BPE to text
        pub fn apply_bpe(&self, text: &str) -> Vec<String> {
            let mut result = Vec::new();

            for word in text.split_whitespace() {
                let mut tokens = Self::get_word_tokens(word);

                // Apply merges in order
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

                result.extend(tokens);
            }

            result
        }

        /// Get individual character tokens for a word
        fn get_word_tokens(word: &str) -> Vec<String> {
            word.chars().map(|c| c.to_string()).collect()
        }
    }

    /// SentencePiece vocabulary wrapper
    pub struct SentencePieceVocabulary {
        model_path: String,
    }

    impl SentencePieceVocabulary {
        pub fn new(model_path: String) -> Self {
            Self { model_path }
        }

        /// Train SentencePiece model
        ///
        /// This is a simplified implementation framework. A complete implementation
        /// would require integration with the actual SentencePiece library.
        pub fn train(texts: &[String], vocab_size: usize, model_prefix: &str) -> Result<Self> {
            // Basic SentencePiece training framework
            // In a real implementation, this would:
            // 1. Preprocess texts (normalize, segment)
            // 2. Build character-level vocabulary
            // 3. Apply unigram language model training
            // 4. Optimize vocabulary using EM algorithm
            // 5. Save model and vocabulary files

            if texts.is_empty() {
                return Err(TextError::Other(anyhow::anyhow!(
                    "Cannot train on empty text corpus"
                )));
            }

            if vocab_size < 100 {
                return Err(TextError::Other(anyhow::anyhow!(
                    "Vocabulary size must be at least 100"
                )));
            }

            // Collect all unique characters from the texts
            let mut char_freq = HashMap::new();
            for text in texts {
                for ch in text.chars() {
                    *char_freq.entry(ch.to_string()).or_insert(0) += 1;
                }
            }

            // Initialize with character-level tokens
            let mut candidates: Vec<(String, i32)> = char_freq.into_iter().collect();
            candidates.sort_by(|a, b| b.1.cmp(&a.1));

            // Add special tokens
            candidates.insert(0, ("<unk>".to_string(), i32::MAX));
            candidates.insert(1, ("<s>".to_string(), i32::MAX - 1));
            candidates.insert(2, ("</s>".to_string(), i32::MAX - 2));

            // In a real implementation, this would use the unigram algorithm
            // to iteratively build subword pieces. For now, we'll use a simplified
            // approach that combines frequent character sequences.

            // Combine frequent adjacent characters (simplified bigram approach)
            let mut bigram_freq = HashMap::new();
            for text in texts {
                let chars: Vec<char> = text.chars().collect();
                for window in chars.windows(2) {
                    let bigram = window[0].to_string() + &window[1].to_string();
                    *bigram_freq.entry(bigram).or_insert(0) += 1;
                }
            }

            // Add frequent bigrams to candidates
            for (bigram, freq) in bigram_freq {
                if freq > 10 && candidates.len() < vocab_size {
                    // Threshold for frequent bigrams
                    candidates.push((bigram, freq));
                }
            }

            // Sort by frequency and take top vocab_size candidates
            candidates.sort_by(|a, b| b.1.cmp(&a.1));
            candidates.truncate(vocab_size);

            // Create model file path
            let model_path = model_prefix.to_string() + ".model";

            // In a real implementation, this would save the actual SentencePiece model
            // For now, we'll save a simple vocabulary list
            let vocab_path = model_prefix.to_string() + ".vocab";
            let mut vocab_file = std::fs::File::create(&vocab_path)?;
            for (token, score) in &candidates {
                writeln!(vocab_file, "{token}\t{score}")?;
            }

            // Create a placeholder model file
            let mut model_file = std::fs::File::create(&model_path)?;
            writeln!(model_file, "# Simplified SentencePiece model placeholder")?;
            writeln!(model_file, "# Vocabulary size: {vocab_size}")?;
            writeln!(
                model_file,
                "# Training texts: {documents} documents",
                documents = texts.len()
            )?;

            Ok(Self { model_path })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_creation() {
        let mut vocab = Vocabulary::new(None);

        // Check special tokens are added
        assert_eq!(vocab.size(), 7);
        assert_eq!(vocab.token_to_id("<pad>"), 0);
        assert_eq!(vocab.token_to_id("<unk>"), 1);

        // Add new tokens
        let id1 = vocab.add_token("hello");
        let id2 = vocab.add_token("world");

        assert_eq!(vocab.size(), 9);
        assert_eq!(vocab.token_to_id("hello"), id1);
        assert_eq!(vocab.token_to_id("world"), id2);

        // Test unknown token
        assert_eq!(vocab.token_to_id("unknown"), 1);
    }

    #[test]
    fn test_vocabulary_from_texts() {
        let texts = vec![
            "hello world".to_string(),
            "hello again".to_string(),
            "world of rust".to_string(),
        ];

        let vocab = Vocabulary::from_texts(&texts, 1, None);

        // Should have special tokens + words that appear at least once
        assert!(vocab.token_to_id.contains_key("hello"));
        assert!(vocab.token_to_id.contains_key("world"));
        assert!(vocab.token_to_id.contains_key("again"));
        assert!(vocab.token_to_id.contains_key("of"));
        assert!(vocab.token_to_id.contains_key("rust"));
    }
}
