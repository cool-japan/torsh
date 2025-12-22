use crate::{Result, TextError};
// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use regex::Regex;
use scirs2_core::random::{Random, Rng};
use std::collections::HashMap;

// Static regex patterns for performance optimization
lazy_static::lazy_static! {
    /// Regex pattern for matching multiple whitespace characters
    static ref WHITESPACE_RE: Regex = Regex::new(r"\s+").unwrap();

    /// Regex pattern for matching URLs
    static ref URL_RE: Regex = Regex::new(r"https?://[^\s]+").unwrap();

    /// Regex pattern for matching email addresses
    static ref EMAIL_RE: Regex = Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap();

    /// Regex pattern for matching HTML tags
    static ref HTML_RE: Regex = Regex::new(r"<[^>]+>").unwrap();

    /// Regex pattern for matching mentions (@username)
    static ref MENTION_RE: Regex = Regex::new(r"@\w+").unwrap();

    /// Regex pattern for matching hashtags (#hashtag)
    static ref HASHTAG_RE: Regex = Regex::new(r"#\w+").unwrap();
}

// ============================================================================
// Text Normalization
// ============================================================================

#[derive(Debug, Clone)]
pub struct TextNormalizer {
    lowercase: bool,
    remove_accents: bool,
    remove_punctuation: bool,
    remove_digits: bool,
    remove_extra_spaces: bool,
    normalize_unicode: bool,
}

impl Default for TextNormalizer {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_accents: false,
            remove_punctuation: false,
            remove_digits: false,
            remove_extra_spaces: true,
            normalize_unicode: true,
        }
    }
}

impl TextNormalizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lowercase(mut self, value: bool) -> Self {
        self.lowercase = value;
        self
    }

    pub fn remove_accents(mut self, value: bool) -> Self {
        self.remove_accents = value;
        self
    }

    pub fn remove_punctuation(mut self, value: bool) -> Self {
        self.remove_punctuation = value;
        self
    }

    pub fn remove_digits(mut self, value: bool) -> Self {
        self.remove_digits = value;
        self
    }

    pub fn remove_extra_spaces(mut self, value: bool) -> Self {
        self.remove_extra_spaces = value;
        self
    }

    pub fn normalize_unicode(mut self, value: bool) -> Self {
        self.normalize_unicode = value;
        self
    }

    pub fn normalize(&self, text: &str) -> String {
        let mut result = text.to_string();

        if self.normalize_unicode {
            result = self.normalize_unicode_text(&result);
        }

        if self.lowercase {
            result = result.to_lowercase();
        }

        if self.remove_accents {
            result = self.remove_accents_text(&result);
        }

        if self.remove_punctuation {
            result = self.remove_punctuation_text(&result);
        }

        if self.remove_digits {
            result = self.remove_digits_text(&result);
        }

        if self.remove_extra_spaces {
            result = self.remove_extra_spaces_text(&result);
        }

        result.trim().to_string()
    }

    fn normalize_unicode_text(&self, text: &str) -> String {
        // Basic Unicode normalization (NFD)
        let mut result = String::new();
        for c in text.chars() {
            match c {
                '\u{2018}' | '\u{2019}' => result.push('\''), // Smart quotes
                '\u{201C}' | '\u{201D}' => result.push('"'),  // Smart quotes
                '\u{2013}' | '\u{2014}' => result.push('-'),  // En dash, em dash
                '\u{2026}' => result.push_str("..."),         // Ellipsis
                _ => result.push(c),
            }
        }
        result
    }

    fn remove_accents_text(&self, text: &str) -> String {
        // Basic accent removal mapping
        let accent_map: HashMap<char, char> = [
            ('à', 'a'),
            ('á', 'a'),
            ('â', 'a'),
            ('ã', 'a'),
            ('ä', 'a'),
            ('å', 'a'),
            ('è', 'e'),
            ('é', 'e'),
            ('ê', 'e'),
            ('ë', 'e'),
            ('ì', 'i'),
            ('í', 'i'),
            ('î', 'i'),
            ('ï', 'i'),
            ('ò', 'o'),
            ('ó', 'o'),
            ('ô', 'o'),
            ('õ', 'o'),
            ('ö', 'o'),
            ('ù', 'u'),
            ('ú', 'u'),
            ('û', 'u'),
            ('ü', 'u'),
            ('ý', 'y'),
            ('ÿ', 'y'),
            ('ñ', 'n'),
            ('ç', 'c'),
            ('À', 'A'),
            ('Á', 'A'),
            ('Â', 'A'),
            ('Ã', 'A'),
            ('Ä', 'A'),
            ('Å', 'A'),
            ('È', 'E'),
            ('É', 'E'),
            ('Ê', 'E'),
            ('Ë', 'E'),
            ('Ì', 'I'),
            ('Í', 'I'),
            ('Î', 'I'),
            ('Ï', 'I'),
            ('Ò', 'O'),
            ('Ó', 'O'),
            ('Ô', 'O'),
            ('Õ', 'O'),
            ('Ö', 'O'),
            ('Ù', 'U'),
            ('Ú', 'U'),
            ('Û', 'U'),
            ('Ü', 'U'),
            ('Ý', 'Y'),
            ('Ÿ', 'Y'),
            ('Ñ', 'N'),
            ('Ç', 'C'),
        ]
        .iter()
        .cloned()
        .collect();

        text.chars()
            .map(|c| accent_map.get(&c).copied().unwrap_or(c))
            .collect()
    }

    fn remove_punctuation_text(&self, text: &str) -> String {
        text.chars().filter(|c| !c.is_ascii_punctuation()).collect()
    }

    fn remove_digits_text(&self, text: &str) -> String {
        text.chars().filter(|c| !c.is_ascii_digit()).collect()
    }

    fn remove_extra_spaces_text(&self, text: &str) -> String {
        WHITESPACE_RE.replace_all(text, " ").to_string()
    }
}

// ============================================================================
// Text Cleaning
// ============================================================================

#[derive(Debug, Clone)]
pub struct TextCleaner {
    remove_urls: bool,
    remove_emails: bool,
    remove_html: bool,
    remove_mentions: bool,
    remove_hashtags: bool,
    remove_special_chars: bool,
}

impl Default for TextCleaner {
    fn default() -> Self {
        Self {
            remove_urls: true,
            remove_emails: true,
            remove_html: true,
            remove_mentions: false,
            remove_hashtags: false,
            remove_special_chars: false,
        }
    }
}

impl TextCleaner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn remove_urls(mut self, value: bool) -> Self {
        self.remove_urls = value;
        self
    }

    pub fn remove_emails(mut self, value: bool) -> Self {
        self.remove_emails = value;
        self
    }

    pub fn remove_html(mut self, value: bool) -> Self {
        self.remove_html = value;
        self
    }

    pub fn remove_mentions(mut self, value: bool) -> Self {
        self.remove_mentions = value;
        self
    }

    pub fn remove_hashtags(mut self, value: bool) -> Self {
        self.remove_hashtags = value;
        self
    }

    pub fn remove_special_chars(mut self, value: bool) -> Self {
        self.remove_special_chars = value;
        self
    }

    pub fn clean(&self, text: &str) -> String {
        let mut result = text.to_string();

        if self.remove_urls {
            result = self.remove_urls_from_text(&result);
        }

        if self.remove_emails {
            result = self.remove_emails_from_text(&result);
        }

        if self.remove_html {
            result = self.remove_html_from_text(&result);
        }

        if self.remove_mentions {
            result = self.remove_mentions_from_text(&result);
        }

        if self.remove_hashtags {
            result = self.remove_hashtags_from_text(&result);
        }

        if self.remove_special_chars {
            result = self.remove_special_chars_from_text(&result);
        }

        // Clean up extra spaces
        WHITESPACE_RE.replace_all(&result, " ").trim().to_string()
    }

    fn remove_urls_from_text(&self, text: &str) -> String {
        URL_RE.replace_all(text, "").to_string()
    }

    fn remove_emails_from_text(&self, text: &str) -> String {
        EMAIL_RE.replace_all(text, "").to_string()
    }

    fn remove_html_from_text(&self, text: &str) -> String {
        HTML_RE.replace_all(text, "").to_string()
    }

    fn remove_mentions_from_text(&self, text: &str) -> String {
        MENTION_RE.replace_all(text, "").to_string()
    }

    fn remove_hashtags_from_text(&self, text: &str) -> String {
        HASHTAG_RE.replace_all(text, "").to_string()
    }

    fn remove_special_chars_from_text(&self, text: &str) -> String {
        text.chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect()
    }
}

// ============================================================================
// Text Augmentation
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct TextAugmenter {
    // RNG is created locally in methods to avoid Send/Sync issues with ThreadRng
}

impl TextAugmenter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn synonym_replacement(&self, text: &str, replacement_prob: f32) -> String {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);
        // Simple word replacement (would need a synonym dictionary in practice)
        let synonyms: HashMap<&str, Vec<&str>> = [
            ("good", vec!["great", "excellent", "wonderful"]),
            ("bad", vec!["terrible", "awful", "horrible"]),
            ("big", vec!["large", "huge", "enormous"]),
            ("small", vec!["tiny", "little", "miniature"]),
        ]
        .iter()
        .cloned()
        .collect();

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result_words = Vec::new();

        for word in words {
            if rng.random::<f32>() < replacement_prob {
                if let Some(syns) = synonyms.get(word.to_lowercase().as_str()) {
                    let idx = rng.gen_range(0..syns.len());
                    result_words.push(syns[idx].to_string());
                } else {
                    result_words.push(word.to_string());
                }
            } else {
                result_words.push(word.to_string());
            }
        }

        result_words.join(" ")
    }

    pub fn random_insertion(&self, text: &str, insertion_prob: f32) -> String {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result_words = Vec::new();

        // Simple words to insert (would need better vocabulary in practice)
        let insert_words = ["the", "a", "an", "very", "really", "quite"];

        for word in words {
            result_words.push(word.to_string());

            if rng.random::<f32>() < insertion_prob {
                let idx = rng.gen_range(0..insert_words.len());
                result_words.push(insert_words[idx].to_string());
            }
        }

        result_words.join(" ")
    }

    pub fn random_deletion(&self, text: &str, deletion_prob: f32) -> String {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result_words = Vec::new();

        for word in words {
            if rng.random::<f32>() >= deletion_prob {
                result_words.push(word.to_string());
            }
        }

        if result_words.is_empty() {
            text.to_string()
        } else {
            result_words.join(" ")
        }
    }

    pub fn random_swap(&self, text: &str, swap_prob: f32) -> String {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);
        let mut words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();

        if words.len() < 2 {
            return text.to_string();
        }

        for i in 0..words.len() {
            if rng.random::<f32>() < swap_prob {
                let j = rng.gen_range(0..words.len());
                words.swap(i, j);
            }
        }

        words.join(" ")
    }

    pub fn back_translation_simulation(&self, text: &str) -> String {
        // Simulate back translation by introducing small changes
        let mut result = self.synonym_replacement(text, 0.1);
        result = self.random_swap(&result, 0.05);
        result
    }

    /// Apply augmentation to text with default parameters
    pub fn augment(&self, text: &str) -> String {
        // Apply a combination of augmentation techniques with low probabilities
        let mut result = self.synonym_replacement(text, 0.1);
        result = self.random_insertion(&result, 0.05);
        result = self.random_deletion(&result, 0.05);
        result = self.random_swap(&result, 0.05);
        result
    }
}

// ============================================================================
// Padding and Truncation
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub enum PaddingStrategy {
    Left,
    Right,
    Center,
}

#[derive(Debug, Clone, Copy)]
pub enum TruncationStrategy {
    Left,
    Right,
    Center,
}

pub fn pad_sequence(
    tokens: &[u32],
    max_length: usize,
    pad_token_id: u32,
    strategy: PaddingStrategy,
) -> Vec<u32> {
    if tokens.len() >= max_length {
        return tokens.to_vec();
    }

    let padding_needed = max_length - tokens.len();
    let mut result = Vec::with_capacity(max_length);

    match strategy {
        PaddingStrategy::Left => {
            result.extend(vec![pad_token_id; padding_needed]);
            result.extend_from_slice(tokens);
        }
        PaddingStrategy::Right => {
            result.extend_from_slice(tokens);
            result.extend(vec![pad_token_id; padding_needed]);
        }
        PaddingStrategy::Center => {
            let left_padding = padding_needed / 2;
            let right_padding = padding_needed - left_padding;
            result.extend(vec![pad_token_id; left_padding]);
            result.extend_from_slice(tokens);
            result.extend(vec![pad_token_id; right_padding]);
        }
    }

    result
}

pub fn truncate_sequence(
    tokens: &[u32],
    max_length: usize,
    strategy: TruncationStrategy,
) -> Vec<u32> {
    if tokens.len() <= max_length {
        return tokens.to_vec();
    }

    match strategy {
        TruncationStrategy::Left => tokens[tokens.len() - max_length..].to_vec(),
        TruncationStrategy::Right => tokens[..max_length].to_vec(),
        TruncationStrategy::Center => {
            let remove_from_each_side = (tokens.len() - max_length) / 2;
            let start = remove_from_each_side;
            let end = tokens.len() - (tokens.len() - max_length - remove_from_each_side);
            tokens[start..end].to_vec()
        }
    }
}

pub fn pad_and_truncate_sequences(
    sequences: &[Vec<u32>],
    max_length: Option<usize>,
    pad_token_id: u32,
    padding_strategy: PaddingStrategy,
    truncation_strategy: TruncationStrategy,
) -> Vec<Vec<u32>> {
    let max_len =
        max_length.unwrap_or_else(|| sequences.iter().map(|seq| seq.len()).max().unwrap_or(0));

    sequences
        .iter()
        .map(|seq| {
            let truncated = truncate_sequence(seq, max_len, truncation_strategy);
            pad_sequence(&truncated, max_len, pad_token_id, padding_strategy)
        })
        .collect()
}

// ============================================================================
// Encoding Schemes
// ============================================================================

pub fn one_hot_encode(token_ids: &[u32], vocab_size: usize) -> Vec<Vec<f32>> {
    token_ids
        .iter()
        .map(|&token_id| {
            let mut encoding = vec![0.0; vocab_size];
            if (token_id as usize) < vocab_size {
                encoding[token_id as usize] = 1.0;
            }
            encoding
        })
        .collect()
}

pub fn label_encode(labels: &[String]) -> (Vec<u32>, HashMap<String, u32>) {
    let mut label_to_id = HashMap::new();
    let mut id_counter = 0u32;

    let encoded: Vec<u32> = labels
        .iter()
        .map(|label| {
            if let Some(&id) = label_to_id.get(label) {
                id
            } else {
                let id = id_counter;
                label_to_id.insert(label.clone(), id);
                id_counter += 1;
                id
            }
        })
        .collect();

    (encoded, label_to_id)
}

// ============================================================================
// Unified Preprocessing Pipeline
// ============================================================================

/// Unified text preprocessing pipeline that combines all preprocessing steps
#[derive(Debug)]
pub struct TextPreprocessingPipeline {
    normalizer: Option<TextNormalizer>,
    cleaner: Option<TextCleaner>,
    augmenter: Option<TextAugmenter>,
    custom_steps: Vec<Box<dyn PreprocessingStep>>,
}

/// Trait for custom preprocessing steps
pub trait PreprocessingStep: std::fmt::Debug + Send + Sync {
    fn process(&self, text: &str) -> Result<String>;
    fn name(&self) -> &str;
}

/// Wrapper for custom closures that implement PreprocessingStep
pub struct CustomStep<F>
where
    F: Fn(&str) -> String + Send + Sync + 'static,
{
    function: F,
    name: String,
}

impl<F> CustomStep<F>
where
    F: Fn(&str) -> String + Send + Sync + 'static,
{
    pub fn new(function: F, name: String) -> Self {
        Self { function, name }
    }
}

impl<F> std::fmt::Debug for CustomStep<F>
where
    F: Fn(&str) -> String + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomStep")
            .field("name", &self.name)
            .finish()
    }
}

impl<F> PreprocessingStep for CustomStep<F>
where
    F: Fn(&str) -> String + Send + Sync + 'static,
{
    fn process(&self, text: &str) -> Result<String> {
        Ok((self.function)(text))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Clone for TextPreprocessingPipeline {
    fn clone(&self) -> Self {
        Self {
            normalizer: self.normalizer.clone(),
            cleaner: self.cleaner.clone(),
            augmenter: self.augmenter.clone(),
            custom_steps: Vec::new(), // Custom steps can't be cloned, so we create an empty vec
        }
    }
}

impl TextPreprocessingPipeline {
    pub fn new() -> Self {
        Self {
            normalizer: None,
            cleaner: None,
            augmenter: None,
            custom_steps: Vec::new(),
        }
    }

    pub fn with_normalization(mut self, normalizer: TextNormalizer) -> Self {
        self.normalizer = Some(normalizer);
        self
    }

    pub fn with_cleaning(mut self, cleaner: TextCleaner) -> Self {
        self.cleaner = Some(cleaner);
        self
    }

    pub fn with_augmentation(mut self, augmenter: TextAugmenter) -> Self {
        self.augmenter = Some(augmenter);
        self
    }

    pub fn add_custom_step(mut self, step: Box<dyn PreprocessingStep>) -> Self {
        self.custom_steps.push(step);
        self
    }

    /// Process a single text through the entire pipeline
    pub fn process_text(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();

        // Apply normalization
        if let Some(normalizer) = &self.normalizer {
            result = normalizer.normalize(&result);
        }

        // Apply cleaning
        if let Some(cleaner) = &self.cleaner {
            result = cleaner.clean(&result);
        }

        // Apply custom steps
        for step in &self.custom_steps {
            result = step.process(&result)?;
        }

        // Apply augmentation (typically for training data)
        if let Some(augmenter) = &self.augmenter {
            result = augmenter.augment(&result);
        }

        Ok(result)
    }

    /// Process multiple texts in batch
    pub fn process_batch(&self, texts: &[String]) -> Result<Vec<String>> {
        texts.iter().map(|text| self.process_text(text)).collect()
    }

    /// Process texts in parallel for better performance
    pub fn process_batch_parallel(&self, texts: &[String]) -> Result<Vec<String>> {
        use scirs2_core::parallel_ops::*;

        texts
            .par_iter()
            .map(|text| self.process_text(text))
            .collect()
    }

    /// Get a summary of the pipeline steps
    pub fn summary(&self) -> Vec<String> {
        let mut steps = Vec::new();

        if self.normalizer.is_some() {
            steps.push("Text Normalization".to_string());
        }

        if self.cleaner.is_some() {
            steps.push("Text Cleaning".to_string());
        }

        for step in &self.custom_steps {
            steps.push(format!("Custom: {}", step.name()));
        }

        if self.augmenter.is_some() {
            steps.push("Text Augmentation".to_string());
        }

        steps
    }
}

impl Default for TextPreprocessingPipeline {
    fn default() -> Self {
        Self::new()
            .with_normalization(TextNormalizer::default())
            .with_cleaning(TextCleaner::default())
    }
}

/// Common preprocessing steps as implementations of PreprocessingStep
#[derive(Debug)]
pub struct RemoveExtraWhitespaceStep;

impl PreprocessingStep for RemoveExtraWhitespaceStep {
    fn process(&self, text: &str) -> Result<String> {
        Ok(WHITESPACE_RE.replace_all(text, " ").trim().to_string())
    }

    fn name(&self) -> &str {
        "Remove Extra Whitespace"
    }
}

#[derive(Debug)]
pub struct MinLengthFilterStep {
    min_length: usize,
}

impl MinLengthFilterStep {
    pub fn new(min_length: usize) -> Self {
        Self { min_length }
    }
}

impl PreprocessingStep for MinLengthFilterStep {
    fn process(&self, text: &str) -> Result<String> {
        if text.len() < self.min_length {
            Err(TextError::ValidationError(format!(
                "Text too short: {} < {}",
                text.len(),
                self.min_length
            )))
        } else {
            Ok(text.to_string())
        }
    }

    fn name(&self) -> &str {
        "Minimum Length Filter"
    }
}

#[derive(Debug)]
pub struct MaxLengthTruncateStep {
    max_length: usize,
}

impl MaxLengthTruncateStep {
    pub fn new(max_length: usize) -> Self {
        Self { max_length }
    }
}

impl PreprocessingStep for MaxLengthTruncateStep {
    fn process(&self, text: &str) -> Result<String> {
        if text.len() > self.max_length {
            Ok(text.chars().take(self.max_length).collect())
        } else {
            Ok(text.to_string())
        }
    }

    fn name(&self) -> &str {
        "Maximum Length Truncate"
    }
}

/// Preprocessing utilities for common operations
pub struct PreprocessingUtils;

impl PreprocessingUtils {
    /// Create a basic preprocessing pipeline for classification tasks
    pub fn classification_pipeline() -> TextPreprocessingPipeline {
        TextPreprocessingPipeline::new()
            .with_normalization(
                TextNormalizer::new()
                    .lowercase(true)
                    .remove_extra_spaces(true)
                    .normalize_unicode(true),
            )
            .with_cleaning(
                TextCleaner::new()
                    .remove_urls(true)
                    .remove_emails(true)
                    .remove_special_chars(true),
            )
            .add_custom_step(Box::new(RemoveExtraWhitespaceStep))
    }

    /// Create a preprocessing pipeline for language modeling
    pub fn language_modeling_pipeline() -> TextPreprocessingPipeline {
        TextPreprocessingPipeline::new()
            .with_normalization(
                TextNormalizer::new()
                    .normalize_unicode(true)
                    .remove_extra_spaces(true),
            )
            .add_custom_step(Box::new(RemoveExtraWhitespaceStep))
            .add_custom_step(Box::new(MinLengthFilterStep::new(10)))
    }

    /// Create a preprocessing pipeline for machine translation
    pub fn translation_pipeline() -> TextPreprocessingPipeline {
        TextPreprocessingPipeline::new()
            .with_normalization(
                TextNormalizer::new()
                    .normalize_unicode(true)
                    .remove_extra_spaces(true),
            )
            .add_custom_step(Box::new(RemoveExtraWhitespaceStep))
            .add_custom_step(Box::new(MaxLengthTruncateStep::new(512)))
    }

    /// Validate and filter texts based on criteria
    pub fn filter_texts(
        texts: &[String],
        min_length: Option<usize>,
        max_length: Option<usize>,
        allowed_chars: Option<&str>,
    ) -> Vec<String> {
        texts
            .iter()
            .filter(|text| {
                // Length checks
                if let Some(min) = min_length {
                    if text.len() < min {
                        return false;
                    }
                }
                if let Some(max) = max_length {
                    if text.len() > max {
                        return false;
                    }
                }

                // Character validation
                if let Some(allowed) = allowed_chars {
                    let allowed_set: std::collections::HashSet<char> = allowed.chars().collect();
                    for ch in text.chars() {
                        if !allowed_set.contains(&ch) && !ch.is_whitespace() {
                            return false;
                        }
                    }
                }

                true
            })
            .cloned()
            .collect()
    }

    /// Batch statistics for analyzing preprocessing effects
    pub fn compute_batch_stats(texts: &[String]) -> PreprocessingStats {
        let total_texts = texts.len();
        let total_chars: usize = texts.iter().map(|t| t.len()).sum();
        let total_words: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();

        let avg_chars = if total_texts > 0 {
            total_chars as f32 / total_texts as f32
        } else {
            0.0
        };
        let avg_words = if total_texts > 0 {
            total_words as f32 / total_texts as f32
        } else {
            0.0
        };

        let min_chars = texts.iter().map(|t| t.len()).min().unwrap_or(0);
        let max_chars = texts.iter().map(|t| t.len()).max().unwrap_or(0);

        PreprocessingStats {
            total_texts,
            total_chars,
            total_words,
            avg_chars_per_text: avg_chars,
            avg_words_per_text: avg_words,
            min_text_length: min_chars,
            max_text_length: max_chars,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreprocessingStats {
    pub total_texts: usize,
    pub total_chars: usize,
    pub total_words: usize,
    pub avg_chars_per_text: f32,
    pub avg_words_per_text: f32,
    pub min_text_length: usize,
    pub max_text_length: usize,
}

// ============================================================================
// Legacy functions (deprecated - use TextPreprocessingPipeline instead)
// ============================================================================

#[deprecated(
    note = "Use TextPreprocessingPipeline::classification_pipeline().process_text() instead"
)]
pub fn normalize_text(text: &str) -> String {
    TextNormalizer::default().normalize(text)
}

#[deprecated(note = "Use proper sentence segmentation libraries instead")]
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            let sentence = current.trim().to_string();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            current.clear();
        }
    }

    // Add any remaining text as a sentence
    let remaining = current.trim().to_string();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }

    sentences
}

pub fn count_words(text: &str) -> usize {
    text.split_whitespace().count()
}

#[deprecated(
    note = "Use TextPreprocessingPipeline::classification_pipeline().process_text() instead"
)]
pub fn clean_text(text: &str) -> String {
    TextCleaner::default().clean(text)
}

// ============================================================================
// Optimized Batch Processing
// ============================================================================

/// High-performance batch processor for text operations
pub struct BatchProcessor {
    chunk_size: usize,
    parallel: bool,
    cache_enabled: bool,
    cache: Option<std::collections::HashMap<String, String>>,
}

impl BatchProcessor {
    pub fn new() -> Self {
        Self {
            chunk_size: 1000,
            parallel: true,
            cache_enabled: false,
            cache: None,
        }
    }

    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn with_cache(mut self, enable: bool) -> Self {
        self.cache_enabled = enable;
        if enable {
            self.cache = Some(std::collections::HashMap::new());
        } else {
            self.cache = None;
        }
        self
    }

    /// Process texts in optimized batches with a processing function
    pub fn process_with_function<F, T>(
        &mut self,
        texts: &[String],
        mut processor: F,
    ) -> Result<Vec<T>>
    where
        F: FnMut(&str) -> Result<T> + Send + Sync,
        T: Send + Sync,
    {
        if self.parallel && texts.len() > self.chunk_size {
            self.process_parallel_chunked(texts, processor)
        } else {
            texts.iter().map(|text| processor(text)).collect()
        }
    }

    /// Process texts with caching support
    pub fn process_with_cache<F>(
        &mut self,
        texts: &[String],
        mut processor: F,
    ) -> Result<Vec<String>>
    where
        F: FnMut(&str) -> Result<String> + Send + Sync,
    {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            if self.cache_enabled {
                if let Some(cache) = &self.cache {
                    if let Some(cached_result) = cache.get(text) {
                        results.push(cached_result.clone());
                        continue;
                    }
                }
            }

            let result = processor(text)?;

            if self.cache_enabled {
                if let Some(cache) = &mut self.cache {
                    cache.insert(text.clone(), result.clone());
                }
            }

            results.push(result);
        }

        Ok(results)
    }

    fn process_parallel_chunked<F, T>(&self, texts: &[String], processor: F) -> Result<Vec<T>>
    where
        F: FnMut(&str) -> Result<T> + Send + Sync,
        T: Send + Sync,
    {
        use scirs2_core::parallel_ops::*;
        use std::sync::Mutex;

        let processor = Mutex::new(processor);

        texts
            .par_chunks(self.chunk_size)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|text| {
                        let mut proc = processor.lock().unwrap();
                        proc(text)
                    })
                    .collect::<Result<Vec<T>>>()
            })
            .collect::<Result<Vec<Vec<T>>>>()
            .map(|chunks| chunks.into_iter().flatten().collect())
    }

    /// Clear cache if enabled
    pub fn clear_cache(&mut self) {
        if let Some(cache) = &mut self.cache {
            cache.clear();
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<(usize, usize)> {
        self.cache
            .as_ref()
            .map(|cache| (cache.len(), cache.capacity()))
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimized batch operations for common text processing tasks
pub struct OptimizedBatchOps;

impl OptimizedBatchOps {
    /// Optimized batch tokenization
    pub fn batch_tokenize(
        texts: &[String],
        tokenizer: &dyn crate::tokenization::Tokenizer,
        parallel: bool,
    ) -> Result<Vec<Vec<u32>>> {
        if parallel && texts.len() > 100 {
            use scirs2_core::parallel_ops::*;
            texts
                .par_iter()
                .map(|text| tokenizer.encode(text))
                .collect()
        } else {
            texts.iter().map(|text| tokenizer.encode(text)).collect()
        }
    }

    /// Optimized batch text cleaning with memory pooling
    pub fn batch_clean(texts: &[String], cleaner: &TextCleaner) -> Vec<String> {
        let mut processor = BatchProcessor::new()
            .with_parallel(true)
            .with_chunk_size(500)
            .with_cache(texts.len() > 1000);

        processor
            .process_with_cache(texts, |text| Ok(cleaner.clean(text)))
            .unwrap_or_else(|_| texts.iter().map(|t| cleaner.clean(t)).collect())
    }

    /// Optimized batch normalization
    pub fn batch_normalize(texts: &[String], normalizer: &TextNormalizer) -> Vec<String> {
        use scirs2_core::parallel_ops::*;

        if texts.len() > 100 {
            texts
                .par_iter()
                .map(|text| normalizer.normalize(text))
                .collect()
        } else {
            texts
                .iter()
                .map(|text| normalizer.normalize(text))
                .collect()
        }
    }

    /// Memory-efficient batch statistics computation
    pub fn batch_statistics(texts: &[String]) -> BatchTextStats {
        use scirs2_core::parallel_ops::*;

        let chunk_size = 1000;

        if texts.len() > chunk_size {
            // Process in parallel chunks to avoid memory pressure
            let partial_stats: Vec<BatchTextStats> = texts
                .par_chunks(chunk_size)
                .map(Self::compute_chunk_stats)
                .collect();

            // Merge partial statistics
            Self::merge_stats(partial_stats)
        } else {
            Self::compute_chunk_stats(texts)
        }
    }

    fn compute_chunk_stats(texts: &[String]) -> BatchTextStats {
        let mut total_chars = 0;
        let mut total_words = 0;
        let mut min_length = usize::MAX;
        let mut max_length = 0;
        let mut char_distribution = std::collections::HashMap::new();

        for text in texts {
            let char_count = text.chars().count();
            let word_count = text.split_whitespace().count();

            total_chars += char_count;
            total_words += word_count;
            min_length = min_length.min(char_count);
            max_length = max_length.max(char_count);

            // Sample character distribution (for performance)
            if texts.len() < 10000 {
                for ch in text.chars() {
                    *char_distribution.entry(ch).or_insert(0) += 1;
                }
            }
        }

        if min_length == usize::MAX {
            min_length = 0;
        }

        BatchTextStats {
            text_count: texts.len(),
            total_chars,
            total_words,
            min_length,
            max_length,
            avg_length: if texts.is_empty() {
                0.0
            } else {
                total_chars as f64 / texts.len() as f64
            },
            avg_words: if texts.is_empty() {
                0.0
            } else {
                total_words as f64 / texts.len() as f64
            },
            char_distribution,
        }
    }

    fn merge_stats(stats: Vec<BatchTextStats>) -> BatchTextStats {
        let mut merged = BatchTextStats::default();

        for stat in stats {
            merged.text_count += stat.text_count;
            merged.total_chars += stat.total_chars;
            merged.total_words += stat.total_words;
            merged.min_length = merged.min_length.min(stat.min_length);
            merged.max_length = merged.max_length.max(stat.max_length);

            // Merge character distributions
            for (ch, count) in stat.char_distribution {
                *merged.char_distribution.entry(ch).or_insert(0) += count;
            }
        }

        // Recalculate averages
        if merged.text_count > 0 {
            merged.avg_length = merged.total_chars as f64 / merged.text_count as f64;
            merged.avg_words = merged.total_words as f64 / merged.text_count as f64;
        }

        merged
    }

    /// Optimized batch filtering with early termination
    pub fn batch_filter<F>(texts: &[String], predicate: F) -> Vec<String>
    where
        F: Fn(&str) -> bool + Send + Sync,
    {
        use scirs2_core::parallel_ops::*;

        if texts.len() > 1000 {
            texts
                .par_iter()
                .filter(|text| predicate(text))
                .cloned()
                .collect()
        } else {
            texts
                .iter()
                .filter(|text| predicate(text))
                .cloned()
                .collect()
        }
    }

    /// Memory-mapped batch processing for very large datasets
    pub fn process_large_file<F>(
        file_path: &std::path::Path,
        processor: F,
        output_path: &std::path::Path,
    ) -> Result<()>
    where
        F: Fn(&str) -> String + Send + Sync,
    {
        use std::fs::File;
        use std::io::{BufRead, BufReader, BufWriter, Write};

        let input_file = File::open(file_path)?;
        let reader = BufReader::new(input_file);

        let output_file = File::create(output_path)?;
        let mut writer = BufWriter::new(output_file);

        const BATCH_SIZE: usize = 1000;
        let mut batch = Vec::with_capacity(BATCH_SIZE);

        for line in reader.lines() {
            let line = line?;
            batch.push(line);

            if batch.len() >= BATCH_SIZE {
                // Process batch
                let processed: Vec<String> = if batch.len() > 100 {
                    use scirs2_core::parallel_ops::*;
                    batch.par_iter().map(|text| processor(text)).collect()
                } else {
                    batch.iter().map(|text| processor(text)).collect()
                };

                // Write results
                for result in processed {
                    writeln!(writer, "{result}")?;
                }

                batch.clear();
            }
        }

        // Process remaining items
        if !batch.is_empty() {
            let processed: Vec<String> = batch.iter().map(|text| processor(text)).collect();
            for result in processed {
                writeln!(writer, "{result}")?;
            }
        }

        writer.flush()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct BatchTextStats {
    pub text_count: usize,
    pub total_chars: usize,
    pub total_words: usize,
    pub min_length: usize,
    pub max_length: usize,
    pub avg_length: f64,
    pub avg_words: f64,
    pub char_distribution: std::collections::HashMap<char, usize>,
}

/// Type alias for streaming batch processor function
type StreamingProcessorFn<T> = Box<dyn FnMut(&[T]) -> Result<Vec<T>>>;

/// Streaming batch processor for memory-efficient processing of large datasets
pub struct StreamingBatchProcessor<T> {
    batch_size: usize,
    buffer: Vec<T>,
    processor: StreamingProcessorFn<T>,
}

impl<T> StreamingBatchProcessor<T> {
    pub fn new<F>(batch_size: usize, processor: F) -> Self
    where
        F: FnMut(&[T]) -> Result<Vec<T>> + 'static,
    {
        Self {
            batch_size,
            buffer: Vec::with_capacity(batch_size),
            processor: Box::new(processor),
        }
    }

    pub fn add_item(&mut self, item: T) -> Result<Option<Vec<T>>> {
        self.buffer.push(item);

        if self.buffer.len() >= self.batch_size {
            let result = (self.processor)(&self.buffer)?;
            self.buffer.clear();
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    pub fn finish(mut self) -> Result<Vec<T>> {
        if !self.buffer.is_empty() {
            (self.processor)(&self.buffer)
        } else {
            Ok(Vec::new())
        }
    }
}
