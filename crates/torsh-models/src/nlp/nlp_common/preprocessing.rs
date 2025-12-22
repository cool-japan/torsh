//! Text preprocessing utilities and tokenizers

use crate::ModelResult;
use std::collections::HashMap;

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

    /// Create T5-style preprocessor
    pub fn t5() -> Self {
        Self {
            max_length: 512,
            truncate: true,
            pad: true,
            pad_token_id: 0,
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

    /// Preprocess token sequence
    pub fn preprocess(&self, tokens: &[u32]) -> ModelResult<Vec<u32>> {
        let mut processed_tokens = tokens.to_vec();

        // 1. Add special tokens if needed (CLS, SEP for BERT)
        if self.add_special_tokens {
            // Add CLS token at beginning (token ID 2 for BERT-style)
            processed_tokens.insert(0, 2);
            // Add SEP token at end (token ID 3 for BERT-style)
            processed_tokens.push(3);
        }

        // 2. Truncate to max_length if needed
        if self.truncate && processed_tokens.len() > self.max_length {
            processed_tokens.truncate(self.max_length);
            // Ensure SEP token is at the end if we added special tokens
            if self.add_special_tokens && processed_tokens.len() > 0 {
                let len = processed_tokens.len();
                processed_tokens[len - 1] = 3; // SEP token
            }
        }

        // 3. Pad to max_length if needed
        if self.pad && processed_tokens.len() < self.max_length {
            let padding_needed = self.max_length - processed_tokens.len();
            processed_tokens.extend(vec![self.pad_token_id; padding_needed]);
        }

        // 4. Attention masks would be created separately in a full implementation
        // For now, we just return the processed tokens

        Ok(processed_tokens)
    }

    /// Create attention mask for the processed tokens
    pub fn create_attention_mask(&self, tokens: &[u32]) -> Vec<u32> {
        tokens
            .iter()
            .map(|&token| if token == self.pad_token_id { 0 } else { 1 })
            .collect()
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
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "and", "of", "to", "a", "in",
            "is", "it", "you", "that", "he", "was", "for", "on", "are", "as", "with", "his",
            "they", "i", "we", "you", "be", "have", "do", "say", "get", "make", "go", "know",
            "take", "see", "come", "think", "look", "want", "give", "use", "find", "tell", "ask",
            "work", "seem", "feel", "try", "leave", "call",
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

impl Default for WhitespaceTokenizer {
    fn default() -> Self {
        Self::new()
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
