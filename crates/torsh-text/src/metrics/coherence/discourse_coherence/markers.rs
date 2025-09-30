//! Discourse marker analysis and extraction
//!
//! This module provides comprehensive discourse marker analysis including marker detection,
//! classification, context analysis, and effectiveness evaluation for various marker types.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use thiserror::Error;

use super::config::{DiscourseCoherenceError, DiscourseMarkerConfig, DiscourseMarkerType};
use super::results::{
    ContextAnalysis, DiscourseMarker, DiscourseMarkerAnalysis, MultiwordMarkerStats,
    SyntacticPosition,
};

/// Errors specific to discourse marker analysis
#[derive(Debug, Error)]
pub enum DiscourseMarkerError {
    #[error("Failed to extract discourse markers: {0}")]
    ExtractionFailed(String),
    #[error("Marker classification error: {0}")]
    ClassificationError(String),
    #[error("Context analysis failed: {0}")]
    ContextAnalysisFailed(String),
    #[error("Invalid marker configuration: {0}")]
    InvalidConfiguration(String),
}

/// Specialized analyzer for discourse markers
pub struct DiscourseMarkerAnalyzer {
    config: DiscourseMarkerConfig,
    discourse_markers: HashMap<String, DiscourseMarkerType>,
    multiword_patterns: HashMap<String, DiscourseMarkerType>,
    marker_cache: Arc<RwLock<HashMap<String, Vec<DiscourseMarker>>>>,
    context_window_cache: Arc<RwLock<HashMap<String, ContextAnalysis>>>,
}

impl DiscourseMarkerAnalyzer {
    /// Create a new discourse marker analyzer
    pub fn new(config: DiscourseMarkerConfig) -> Self {
        let discourse_markers = Self::build_discourse_markers(&config);
        let multiword_patterns = Self::build_multiword_patterns(&config);
        let marker_cache = Arc::new(RwLock::new(HashMap::new()));
        let context_window_cache = Arc::new(RwLock::new(HashMap::new()));

        Self {
            config,
            discourse_markers,
            multiword_patterns,
            marker_cache,
            context_window_cache,
        }
    }

    /// Extract and analyze discourse markers from sentences
    pub fn analyze_markers(
        &self,
        sentences: &[String],
    ) -> Result<Vec<DiscourseMarker>, DiscourseMarkerError> {
        let mut all_markers = Vec::new();

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let sentence_markers =
                self.extract_markers_from_sentence(sent_idx, sentence, sentences)?;
            all_markers.extend(sentence_markers);
        }

        // Sort markers by position for consistent ordering
        all_markers.sort_by_key(|m| (m.sentence_index, m.word_index));

        Ok(all_markers)
    }

    /// Extract markers from a single sentence
    fn extract_markers_from_sentence(
        &self,
        sent_idx: usize,
        sentence: &str,
        all_sentences: &[String],
    ) -> Result<Vec<DiscourseMarker>, DiscourseMarkerError> {
        let mut markers = Vec::new();
        let words: Vec<&str> = sentence.split_whitespace().collect();

        // Single-word marker detection
        for (word_idx, word) in words.iter().enumerate() {
            if let Some(marker_type) = self.classify_single_word_marker(word) {
                let marker = self.create_discourse_marker(
                    marker_type,
                    word,
                    sent_idx,
                    word_idx,
                    all_sentences,
                )?;
                markers.push(marker);
            }
        }

        // Multiword marker detection
        if self.config.analyze_multiword_markers {
            let multiword_markers =
                self.extract_multiword_markers(sent_idx, &words, all_sentences)?;
            markers.extend(multiword_markers);
        }

        Ok(markers)
    }

    /// Classify single-word discourse marker
    fn classify_single_word_marker(&self, word: &str) -> Option<DiscourseMarkerType> {
        let normalized_word = word
            .to_lowercase()
            .trim_matches(|c: char| c.is_ascii_punctuation())
            .to_string();

        // Check custom markers first
        if let Some(marker_type) = self.config.custom_markers.get(&normalized_word) {
            return Some(marker_type.clone());
        }

        // Check built-in markers
        self.discourse_markers.get(&normalized_word).cloned()
    }

    /// Extract multiword discourse markers
    fn extract_multiword_markers(
        &self,
        sent_idx: usize,
        words: &[&str],
        all_sentences: &[String],
    ) -> Result<Vec<DiscourseMarker>, DiscourseMarkerError> {
        let mut markers = Vec::new();

        // Check for 2-5 word patterns
        for pattern_length in 2..=5.min(words.len()) {
            for start_idx in 0..=(words.len() - pattern_length) {
                let pattern = words[start_idx..start_idx + pattern_length].join(" ");
                let normalized_pattern = pattern.to_lowercase();

                if let Some(marker_type) = self.multiword_patterns.get(&normalized_pattern) {
                    let marker = self.create_discourse_marker(
                        marker_type.clone(),
                        &pattern,
                        sent_idx,
                        start_idx,
                        all_sentences,
                    )?;
                    markers.push(marker);
                }
            }
        }

        Ok(markers)
    }

    /// Create a discourse marker with full analysis
    fn create_discourse_marker(
        &self,
        marker_type: DiscourseMarkerType,
        text: &str,
        sent_idx: usize,
        word_idx: usize,
        all_sentences: &[String],
    ) -> Result<DiscourseMarker, DiscourseMarkerError> {
        let context = if self.config.include_context_analysis {
            self.analyze_marker_context(sent_idx, word_idx, all_sentences)?
        } else {
            ContextAnalysis::default()
        };

        let confidence =
            self.calculate_marker_confidence(&marker_type, text, sent_idx, word_idx, &context);

        if confidence < self.config.min_confidence_threshold {
            return Err(DiscourseMarkerError::ClassificationError(format!(
                "Marker '{}' below confidence threshold",
                text
            )));
        }

        let rhetorical_strength = self.calculate_rhetorical_strength(&marker_type, &context);
        let scope = self.calculate_marker_scope(sent_idx, word_idx, all_sentences);
        let position = self.calculate_text_position(sent_idx, word_idx, text, all_sentences);
        let syntactic_position = self.analyze_syntactic_position(text, sent_idx, word_idx);

        Ok(DiscourseMarker {
            marker_type,
            text: text.to_string(),
            position,
            sentence_index: sent_idx,
            word_index: word_idx,
            context,
            confidence,
            rhetorical_strength,
            scope,
            syntactic_position,
        })
    }

    /// Analyze context around a discourse marker
    fn analyze_marker_context(
        &self,
        sent_idx: usize,
        word_idx: usize,
        sentences: &[String],
    ) -> Result<ContextAnalysis, DiscourseMarkerError> {
        // Extract preceding context
        let preceding_context = self.extract_preceding_context(sent_idx, word_idx, sentences);

        // Extract following context
        let following_context = self.extract_following_context(sent_idx, word_idx, sentences);

        // Calculate semantic coherence
        let semantic_coherence_preceding =
            self.calculate_semantic_coherence_with_context(&preceding_context);
        let semantic_coherence_following =
            self.calculate_semantic_coherence_with_context(&following_context);

        // Calculate pragmatic appropriateness
        let pragmatic_appropriateness =
            self.calculate_pragmatic_appropriateness(&preceding_context, &following_context);

        Ok(ContextAnalysis {
            preceding_context,
            following_context,
            semantic_coherence_preceding,
            semantic_coherence_following,
            pragmatic_appropriateness,
        })
    }

    /// Extract preceding context words
    fn extract_preceding_context(
        &self,
        sent_idx: usize,
        word_idx: usize,
        sentences: &[String],
    ) -> Vec<String> {
        let mut context = Vec::new();
        let window_size = self.config.context_window_size;

        // Words before the marker in the same sentence
        if let Some(sentence) = sentences.get(sent_idx) {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let start_idx = word_idx.saturating_sub(window_size / 2);
            for i in start_idx..word_idx {
                if let Some(word) = words.get(i) {
                    context.push(word.to_string());
                }
            }
        }

        // Add words from previous sentences if needed
        if context.len() < window_size {
            let needed = window_size - context.len();
            for prev_sent_idx in (0..sent_idx).rev() {
                if let Some(prev_sentence) = sentences.get(prev_sent_idx) {
                    let prev_words: Vec<&str> = prev_sentence.split_whitespace().collect();
                    for word in prev_words.iter().rev().take(needed) {
                        context.insert(0, word.to_string());
                        if context.len() >= window_size {
                            break;
                        }
                    }
                    if context.len() >= window_size {
                        break;
                    }
                }
            }
        }

        context
    }

    /// Extract following context words
    fn extract_following_context(
        &self,
        sent_idx: usize,
        word_idx: usize,
        sentences: &[String],
    ) -> Vec<String> {
        let mut context = Vec::new();
        let window_size = self.config.context_window_size;

        // Words after the marker in the same sentence
        if let Some(sentence) = sentences.get(sent_idx) {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            for i in (word_idx + 1)..words.len().min(word_idx + 1 + window_size / 2) {
                if let Some(word) = words.get(i) {
                    context.push(word.to_string());
                }
            }
        }

        // Add words from following sentences if needed
        if context.len() < window_size {
            let needed = window_size - context.len();
            for next_sent_idx in (sent_idx + 1)..sentences.len() {
                if let Some(next_sentence) = sentences.get(next_sent_idx) {
                    let next_words: Vec<&str> = next_sentence.split_whitespace().collect();
                    for word in next_words.iter().take(needed) {
                        context.push(word.to_string());
                        if context.len() >= window_size {
                            break;
                        }
                    }
                    if context.len() >= window_size {
                        break;
                    }
                }
            }
        }

        context
    }

    /// Calculate semantic coherence with context
    fn calculate_semantic_coherence_with_context(&self, context: &[String]) -> f64 {
        if context.is_empty() {
            return 0.0;
        }

        // Simple semantic similarity based on word overlap and semantic fields
        let mut coherence_sum = 0.0;
        let mut pair_count = 0;

        for i in 0..context.len() {
            for j in (i + 1)..context.len() {
                coherence_sum += self.calculate_word_similarity(&context[i], &context[j]);
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            coherence_sum / pair_count as f64
        } else {
            0.0
        }
    }

    /// Calculate pragmatic appropriateness of marker in context
    fn calculate_pragmatic_appropriateness(
        &self,
        preceding: &[String],
        following: &[String],
    ) -> f64 {
        if preceding.is_empty() || following.is_empty() {
            return 0.5; // Neutral score when context is incomplete
        }

        // Analyze the appropriateness based on content word patterns
        let preceding_content = self.extract_content_words_from_context(preceding);
        let following_content = self.extract_content_words_from_context(following);

        // Calculate semantic consistency
        self.calculate_context_semantic_consistency(
            &[preceding_content, following_content].concat(),
        )
    }

    /// Extract content words from context
    fn extract_content_words_from_context(&self, context: &[String]) -> Vec<String> {
        let function_words: HashSet<&str> = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after", "above",
            "below", "over", "under", "between", "is", "am", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it",
            "we", "they", "my", "your", "his", "her", "its", "our", "their",
        ]
        .iter()
        .collect();

        context
            .iter()
            .filter_map(|word| {
                let normalized = word.to_lowercase();
                if !function_words.contains(normalized.as_str()) && normalized.len() > 2 {
                    Some(normalized)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Calculate semantic consistency of context words
    fn calculate_context_semantic_consistency(&self, words: &[String]) -> f64 {
        if words.len() < 2 {
            return 0.5;
        }

        let mut similarity_sum = 0.0;
        let mut pair_count = 0;

        // Calculate pairwise semantic similarities
        for i in 0..words.len() {
            for j in (i + 1)..words.len() {
                similarity_sum += self.calculate_word_similarity(&words[i], &words[j]);
                pair_count += 1;
            }
        }

        similarity_sum / pair_count as f64
    }

    /// Calculate semantic similarity between two words
    fn calculate_word_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }

        // Simple similarity measures
        let mut similarity = 0.0;

        // Morphological similarity (shared prefixes/suffixes)
        if word1.len() > 3 && word2.len() > 3 {
            let prefix_sim = self.calculate_prefix_similarity(word1, word2);
            let suffix_sim = self.calculate_suffix_similarity(word1, word2);
            similarity += (prefix_sim + suffix_sim) * 0.3;
        }

        // Length similarity
        let length_ratio =
            (word1.len().min(word2.len()) as f64) / (word1.len().max(word2.len()) as f64);
        similarity += length_ratio * 0.1;

        // Character overlap
        let char_overlap = self.calculate_character_overlap(word1, word2);
        similarity += char_overlap * 0.2;

        // Semantic field similarity (simplified)
        similarity += self.calculate_semantic_field_similarity(word1, word2) * 0.4;

        similarity.min(1.0)
    }

    /// Calculate prefix similarity
    fn calculate_prefix_similarity(&self, word1: &str, word2: &str) -> f64 {
        let min_len = word1.len().min(word2.len()).min(4);
        let mut common_len = 0;

        for i in 0..min_len {
            if word1.chars().nth(i) == word2.chars().nth(i) {
                common_len += 1;
            } else {
                break;
            }
        }

        if min_len > 0 {
            common_len as f64 / min_len as f64
        } else {
            0.0
        }
    }

    /// Calculate suffix similarity
    fn calculate_suffix_similarity(&self, word1: &str, word2: &str) -> f64 {
        let min_len = word1.len().min(word2.len()).min(4);
        let mut common_len = 0;

        let chars1: Vec<char> = word1.chars().collect();
        let chars2: Vec<char> = word2.chars().collect();

        for i in 0..min_len {
            if chars1.get(chars1.len() - 1 - i) == chars2.get(chars2.len() - 1 - i) {
                common_len += 1;
            } else {
                break;
            }
        }

        if min_len > 0 {
            common_len as f64 / min_len as f64
        } else {
            0.0
        }
    }

    /// Calculate character overlap
    fn calculate_character_overlap(&self, word1: &str, word2: &str) -> f64 {
        let chars1: HashSet<char> = word1.chars().collect();
        let chars2: HashSet<char> = word2.chars().collect();

        let intersection = chars1.intersection(&chars2).count();
        let union = chars1.union(&chars2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    /// Calculate semantic field similarity (simplified)
    fn calculate_semantic_field_similarity(&self, word1: &str, word2: &str) -> f64 {
        // This is a simplified implementation
        // In a real system, you would use word embeddings or semantic networks

        let semantic_fields = self.get_semantic_fields();
        let fields1 = self.get_word_semantic_fields(word1, &semantic_fields);
        let fields2 = self.get_word_semantic_fields(word2, &semantic_fields);

        if fields1.is_empty() || fields2.is_empty() {
            return 0.1; // Small baseline similarity
        }

        let intersection: HashSet<_> = fields1.intersection(&fields2).collect();
        let union: HashSet<_> = fields1.union(&fields2).collect();

        intersection.len() as f64 / union.len() as f64
    }

    /// Get predefined semantic fields
    fn get_semantic_fields(&self) -> HashMap<String, HashSet<String>> {
        let mut fields = HashMap::new();

        // Emotion field
        let emotion_words: HashSet<String> = [
            "happy",
            "sad",
            "angry",
            "excited",
            "frustrated",
            "pleased",
            "upset",
            "joy",
            "fear",
            "surprise",
            "disgust",
            "anticipation",
            "trust",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        fields.insert("emotion".to_string(), emotion_words);

        // Time field
        let time_words: HashSet<String> = [
            "before",
            "after",
            "during",
            "while",
            "when",
            "then",
            "now",
            "later",
            "earlier",
            "subsequently",
            "previously",
            "meanwhile",
            "simultaneously",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        fields.insert("time".to_string(), time_words);

        // Causation field
        let causation_words: HashSet<String> = [
            "because",
            "since",
            "due",
            "caused",
            "result",
            "consequence",
            "therefore",
            "thus",
            "hence",
            "accordingly",
            "consequently",
            "effect",
            "reason",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        fields.insert("causation".to_string(), causation_words);

        // Contrast field
        let contrast_words: HashSet<String> = [
            "however",
            "but",
            "although",
            "despite",
            "nevertheless",
            "nonetheless",
            "whereas",
            "while",
            "conversely",
            "alternatively",
            "instead",
            "rather",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        fields.insert("contrast".to_string(), contrast_words);

        fields
    }

    /// Get semantic fields for a word
    fn get_word_semantic_fields(
        &self,
        word: &str,
        fields: &HashMap<String, HashSet<String>>,
    ) -> HashSet<String> {
        let mut word_fields = HashSet::new();
        let normalized_word = word.to_lowercase();

        for (field_name, field_words) in fields {
            if field_words.contains(&normalized_word) {
                word_fields.insert(field_name.clone());
            }
        }

        word_fields
    }

    /// Calculate marker confidence score
    fn calculate_marker_confidence(
        &self,
        marker_type: &DiscourseMarkerType,
        text: &str,
        sent_idx: usize,
        word_idx: usize,
        context: &ContextAnalysis,
    ) -> f64 {
        let mut confidence = 0.7; // Base confidence

        // Adjust based on marker type specificity
        confidence += match marker_type {
            DiscourseMarkerType::Cause | DiscourseMarkerType::Contrast => 0.2,
            DiscourseMarkerType::Temporal | DiscourseMarkerType::Summary => 0.15,
            DiscourseMarkerType::Addition | DiscourseMarkerType::Elaboration => 0.1,
            _ => 0.05,
        };

        // Adjust based on text length (multiword markers are often more reliable)
        if text.split_whitespace().count() > 1 {
            confidence += 0.1;
        }

        // Adjust based on context coherence
        let context_coherence = (context.semantic_coherence_preceding
            + context.semantic_coherence_following
            + context.pragmatic_appropriateness)
            / 3.0;
        confidence += context_coherence * 0.2;

        // Adjust based on position (markers at sentence boundaries are often more reliable)
        if word_idx == 0 {
            confidence += 0.1; // Sentence-initial markers
        }

        // Apply marker type weights from configuration
        if let Some(&weight) = self.config.marker_weights.get(marker_type) {
            confidence *= weight;
        }

        confidence.min(1.0)
    }

    /// Calculate rhetorical strength of marker
    fn calculate_rhetorical_strength(
        &self,
        marker_type: &DiscourseMarkerType,
        context: &ContextAnalysis,
    ) -> f64 {
        let base_strength = match marker_type {
            DiscourseMarkerType::Contrast => 0.9,
            DiscourseMarkerType::Cause => 0.85,
            DiscourseMarkerType::Concession => 0.8,
            DiscourseMarkerType::Summary => 0.75,
            DiscourseMarkerType::Emphasis => 0.7,
            DiscourseMarkerType::Temporal => 0.65,
            DiscourseMarkerType::Comparison => 0.6,
            DiscourseMarkerType::Conditional => 0.6,
            DiscourseMarkerType::Exemplification => 0.55,
            DiscourseMarkerType::Elaboration => 0.5,
            DiscourseMarkerType::Addition => 0.45,
            DiscourseMarkerType::Alternative => 0.4,
            DiscourseMarkerType::Reformulation => 0.4,
        };

        // Adjust based on context appropriateness
        let context_factor = context.pragmatic_appropriateness;
        base_strength * (0.5 + 0.5 * context_factor)
    }

    /// Calculate scope of marker influence
    fn calculate_marker_scope(
        &self,
        sent_idx: usize,
        word_idx: usize,
        sentences: &[String],
    ) -> usize {
        // Simple heuristic: stronger markers have broader scope
        let base_scope = match word_idx {
            0 => 2, // Sentence-initial markers often have broader scope
            _ => 1, // Other positions have more local scope
        };

        // Extend scope based on marker type and position
        let current_sentence = sentences.get(sent_idx).map(|s| s.len()).unwrap_or(0);
        let position_factor = if word_idx < 3 { 1.5 } else { 1.0 };

        (base_scope as f64 * position_factor) as usize
    }

    /// Calculate text position coordinates
    fn calculate_text_position(
        &self,
        sent_idx: usize,
        word_idx: usize,
        text: &str,
        sentences: &[String],
    ) -> (usize, usize) {
        let mut char_start = 0;

        // Add characters from previous sentences
        for i in 0..sent_idx {
            if let Some(sentence) = sentences.get(i) {
                char_start += sentence.len() + 1; // +1 for sentence separator
            }
        }

        // Add characters from previous words in current sentence
        if let Some(current_sentence) = sentences.get(sent_idx) {
            let words: Vec<&str> = current_sentence.split_whitespace().collect();
            for i in 0..word_idx {
                if let Some(word) = words.get(i) {
                    char_start += word.len() + 1; // +1 for space
                }
            }
        }

        let char_end = char_start + text.len();
        (char_start, char_end)
    }

    /// Analyze syntactic position of marker
    fn analyze_syntactic_position(
        &self,
        text: &str,
        sent_idx: usize,
        word_idx: usize,
    ) -> SyntacticPosition {
        // Simplified syntactic analysis
        let sentence_position = match word_idx {
            0 => "beginning".to_string(),
            _ => {
                // This would need more sophisticated sentence parsing
                if text.ends_with('.') || text.ends_with('!') || text.ends_with('?') {
                    "end".to_string()
                } else {
                    "middle".to_string()
                }
            }
        };

        let clause_boundary = word_idx == 0 || text.contains(',') || text.contains(';');

        // Simple POS tagging based on marker patterns
        let pos_tag = match text.to_lowercase().as_str() {
            "however" | "therefore" | "furthermore" | "moreover" => "ADV".to_string(),
            "but" | "and" | "or" | "because" | "although" => "CONJ".to_string(),
            "first" | "second" | "finally" | "next" => "ADV".to_string(),
            _ => "CONJ".to_string(), // Default assumption for discourse markers
        };

        let dependency_relation = if word_idx == 0 {
            "discourse".to_string()
        } else {
            "advmod".to_string()
        };

        SyntacticPosition {
            sentence_position,
            clause_boundary,
            pos_tag,
            dependency_relation,
        }
    }

    /// Generate comprehensive marker analysis
    pub fn generate_analysis(&self, markers: &[DiscourseMarker]) -> DiscourseMarkerAnalysis {
        let total_markers = markers.len();
        let marker_density = self.calculate_marker_density(markers);
        let type_distribution = self.calculate_type_distribution(markers);
        let average_confidence = self.calculate_average_confidence(markers);
        let effectiveness_score = self.calculate_effectiveness_score(markers);
        let context_integration = self.calculate_context_integration(markers);
        let multiword_statistics = self.calculate_multiword_statistics(markers);

        DiscourseMarkerAnalysis {
            total_markers,
            marker_density,
            type_distribution,
            average_confidence,
            effectiveness_score,
            context_integration,
            multiword_statistics,
        }
    }

    /// Calculate marker density
    fn calculate_marker_density(&self, markers: &[DiscourseMarker]) -> f64 {
        if markers.is_empty() {
            return 0.0;
        }

        // Estimate total word count from marker positions
        let max_sentence = markers.iter().map(|m| m.sentence_index).max().unwrap_or(0);
        let estimated_word_count = (max_sentence + 1) * 15; // Rough estimate

        (markers.len() as f64 / estimated_word_count as f64) * 100.0
    }

    /// Calculate type distribution
    fn calculate_type_distribution(
        &self,
        markers: &[DiscourseMarker],
    ) -> HashMap<DiscourseMarkerType, usize> {
        let mut distribution = HashMap::new();

        for marker in markers {
            *distribution.entry(marker.marker_type.clone()).or_insert(0) += 1;
        }

        distribution
    }

    /// Calculate average confidence
    fn calculate_average_confidence(&self, markers: &[DiscourseMarker]) -> f64 {
        if markers.is_empty() {
            return 0.0;
        }

        markers.iter().map(|m| m.confidence).sum::<f64>() / markers.len() as f64
    }

    /// Calculate effectiveness score
    fn calculate_effectiveness_score(&self, markers: &[DiscourseMarker]) -> f64 {
        if markers.is_empty() {
            return 0.0;
        }

        markers.iter().map(|m| m.rhetorical_strength).sum::<f64>() / markers.len() as f64
    }

    /// Calculate context integration score
    fn calculate_context_integration(&self, markers: &[DiscourseMarker]) -> f64 {
        if markers.is_empty() {
            return 0.0;
        }

        let integration_sum: f64 = markers
            .iter()
            .map(|m| {
                (m.context.semantic_coherence_preceding
                    + m.context.semantic_coherence_following
                    + m.context.pragmatic_appropriateness)
                    / 3.0
            })
            .sum();

        integration_sum / markers.len() as f64
    }

    /// Calculate multiword marker statistics
    fn calculate_multiword_statistics(&self, markers: &[DiscourseMarker]) -> MultiwordMarkerStats {
        let multiword_markers: Vec<_> = markers
            .iter()
            .filter(|m| m.text.split_whitespace().count() > 1)
            .collect();

        let multiword_count = multiword_markers.len();

        let average_length = if multiword_count > 0 {
            multiword_markers
                .iter()
                .map(|m| m.text.split_whitespace().count())
                .sum::<usize>() as f64
                / multiword_count as f64
        } else {
            0.0
        };

        // Count pattern frequencies
        let mut pattern_counts: HashMap<String, usize> = HashMap::new();
        for marker in &multiword_markers {
            *pattern_counts.entry(marker.text.clone()).or_insert(0) += 1;
        }

        let mut common_patterns: Vec<(String, usize)> = pattern_counts.into_iter().collect();
        common_patterns.sort_by(|a, b| b.1.cmp(&a.1));
        common_patterns.truncate(10); // Top 10 patterns

        MultiwordMarkerStats {
            multiword_count,
            average_length,
            common_patterns,
        }
    }

    /// Build discourse markers dictionary
    fn build_discourse_markers(
        config: &DiscourseMarkerConfig,
    ) -> HashMap<String, DiscourseMarkerType> {
        let mut markers = HashMap::new();

        // Addition markers
        for word in &[
            "also",
            "furthermore",
            "moreover",
            "additionally",
            "besides",
            "plus",
        ] {
            markers.insert(word.to_string(), DiscourseMarkerType::Addition);
        }

        // Contrast markers
        for word in &[
            "however",
            "but",
            "nevertheless",
            "nonetheless",
            "yet",
            "still",
            "though",
        ] {
            markers.insert(word.to_string(), DiscourseMarkerType::Contrast);
        }

        // Cause markers
        for word in &[
            "therefore",
            "thus",
            "consequently",
            "hence",
            "accordingly",
            "so",
        ] {
            markers.insert(word.to_string(), DiscourseMarkerType::Cause);
        }

        // Temporal markers
        for word in &[
            "then",
            "next",
            "afterwards",
            "subsequently",
            "meanwhile",
            "simultaneously",
        ] {
            markers.insert(word.to_string(), DiscourseMarkerType::Temporal);
        }

        // Conditional markers
        for word in &["if", "unless", "provided", "assuming", "suppose"] {
            markers.insert(word.to_string(), DiscourseMarkerType::Conditional);
        }

        // Add custom markers from configuration
        for (marker, marker_type) in &config.custom_markers {
            markers.insert(marker.clone(), marker_type.clone());
        }

        markers
    }

    /// Build multiword patterns dictionary
    fn build_multiword_patterns(
        config: &DiscourseMarkerConfig,
    ) -> HashMap<String, DiscourseMarkerType> {
        let mut patterns = HashMap::new();

        // Addition patterns
        for pattern in &["in addition", "what is more", "not only but also"] {
            patterns.insert(pattern.to_string(), DiscourseMarkerType::Addition);
        }

        // Contrast patterns
        for pattern in &[
            "on the other hand",
            "by contrast",
            "in contrast",
            "on the contrary",
        ] {
            patterns.insert(pattern.to_string(), DiscourseMarkerType::Contrast);
        }

        // Cause patterns
        for pattern in &[
            "as a result",
            "for this reason",
            "due to this",
            "because of this",
        ] {
            patterns.insert(pattern.to_string(), DiscourseMarkerType::Cause);
        }

        // Summary patterns
        for pattern in &["in conclusion", "to sum up", "in summary", "all in all"] {
            patterns.insert(pattern.to_string(), DiscourseMarkerType::Summary);
        }

        patterns
    }
}

impl Default for ContextAnalysis {
    fn default() -> Self {
        Self {
            preceding_context: Vec::new(),
            following_context: Vec::new(),
            semantic_coherence_preceding: 0.5,
            semantic_coherence_following: 0.5,
            pragmatic_appropriateness: 0.5,
        }
    }
}
