//! Rhetorical structure analysis for discourse coherence
//!
//! This module provides comprehensive rhetorical structure analysis including
//! rhetorical relation identification, discourse tree construction, and
//! nucleus-satellite analysis for understanding discourse organization.

use std::collections::{HashMap, VecDeque};
use thiserror::Error;

use super::config::{DiscourseCoherenceError, RhetoricalRelationType, RhetoricalStructureConfig};
use super::markers::{DiscourseMarker, DiscourseMarkerAnalyzer};
use super::results::{
    DiscourseNode, DiscourseTree, EmbeddingStats, NuclearChain, NucleusSatelliteAnalysis,
    RhetoricalStructureAnalysis,
};

/// Errors specific to rhetorical structure analysis
#[derive(Debug, Error)]
pub enum RhetoricalAnalysisError {
    #[error("Failed to identify rhetorical relations: {0}")]
    RelationIdentificationFailed(String),
    #[error("Discourse tree construction failed: {0}")]
    TreeConstructionFailed(String),
    #[error("Nucleus-satellite analysis failed: {0}")]
    NucleusSatelliteAnalysisFailed(String),
    #[error("Invalid rhetorical configuration: {0}")]
    InvalidConfiguration(String),
}

/// Specialized analyzer for rhetorical structure
pub struct RhetoricalStructureAnalyzer {
    config: RhetoricalStructureConfig,
    relation_patterns: HashMap<String, RhetoricalRelationType>,
    signal_phrases: HashMap<RhetoricalRelationType, Vec<String>>,
    tree_builder: DiscourseTreeBuilder,
    nucleus_analyzer: NucleusSatelliteAnalyzer,
}

impl RhetoricalStructureAnalyzer {
    /// Create a new rhetorical structure analyzer
    pub fn new(config: RhetoricalStructureConfig) -> Self {
        let relation_patterns = Self::build_relation_patterns();
        let signal_phrases = Self::build_signal_phrases();
        let tree_builder = DiscourseTreeBuilder::new(config.clone());
        let nucleus_analyzer = NucleusSatelliteAnalyzer::new(config.clone());

        Self {
            config,
            relation_patterns,
            signal_phrases,
            tree_builder,
            nucleus_analyzer,
        }
    }

    /// Analyze rhetorical structure of text
    pub fn analyze_rhetorical_structure(
        &self,
        sentences: &[String],
        markers: &[DiscourseMarker],
    ) -> Result<RhetoricalStructureAnalysis, RhetoricalAnalysisError> {
        // Identify rhetorical relations
        let relations = self.identify_rhetorical_relations(sentences, markers)?;

        // Calculate relation distribution score
        let relation_distribution_score = self.calculate_relation_distribution_score(&relations);

        // Calculate structural complexity
        let structural_complexity =
            self.calculate_structural_complexity(&relations, sentences.len());

        // Build discourse tree if enabled
        let discourse_tree = if self.config.build_discourse_tree {
            Some(
                self.tree_builder
                    .build_tree(sentences, &relations, markers)?,
            )
        } else {
            None
        };

        // Perform nucleus-satellite analysis if enabled
        let nucleus_satellite_analysis = if self.config.nucleus_satellite_detection {
            Some(
                self.nucleus_analyzer
                    .analyze(sentences, &relations, markers)?,
            )
        } else {
            None
        };

        // Calculate relation confidence scores
        let relation_confidences =
            self.calculate_relation_confidences(&relations, sentences, markers);

        Ok(RhetoricalStructureAnalysis {
            relations,
            relation_distribution_score,
            structural_complexity,
            discourse_tree,
            nucleus_satellite_analysis,
            relation_confidences,
        })
    }

    /// Identify rhetorical relations in text
    fn identify_rhetorical_relations(
        &self,
        sentences: &[String],
        markers: &[DiscourseMarker],
    ) -> Result<HashMap<RhetoricalRelationType, usize>, RhetoricalAnalysisError> {
        let mut relations: HashMap<RhetoricalRelationType, usize> = HashMap::new();

        // Analyze based on discourse markers
        for marker in markers {
            if let Some(relation_type) = self.marker_to_relation(&marker.marker_type) {
                *relations.entry(relation_type).or_insert(0) += 1;
            }
        }

        // Analyze based on textual patterns
        for (i, sentence) in sentences.iter().enumerate() {
            let detected_relations = self.detect_pattern_based_relations(sentence, i, sentences)?;
            for relation_type in detected_relations {
                *relations.entry(relation_type).or_insert(0) += 1;
            }
        }

        // Analyze sentence pairs for implicit relations
        for i in 0..(sentences.len().saturating_sub(1)) {
            if let Some(relation_type) =
                self.detect_implicit_relation(&sentences[i], &sentences[i + 1])?
            {
                *relations.entry(relation_type).or_insert(0) += 1;
            }
        }

        Ok(relations)
    }

    /// Map discourse marker types to rhetorical relations
    fn marker_to_relation(
        &self,
        marker_type: &super::config::DiscourseMarkerType,
    ) -> Option<RhetoricalRelationType> {
        use super::config::DiscourseMarkerType;

        match marker_type {
            DiscourseMarkerType::Cause => Some(RhetoricalRelationType::Evidence),
            DiscourseMarkerType::Contrast => Some(RhetoricalRelationType::Contrast),
            DiscourseMarkerType::Concession => Some(RhetoricalRelationType::Concession),
            DiscourseMarkerType::Temporal => Some(RhetoricalRelationType::Sequence),
            DiscourseMarkerType::Elaboration => Some(RhetoricalRelationType::Elaboration),
            DiscourseMarkerType::Exemplification => Some(RhetoricalRelationType::Elaboration),
            DiscourseMarkerType::Summary => Some(RhetoricalRelationType::Summary),
            DiscourseMarkerType::Addition => Some(RhetoricalRelationType::Joint),
            DiscourseMarkerType::Alternative => Some(RhetoricalRelationType::Joint),
            DiscourseMarkerType::Conditional => Some(RhetoricalRelationType::Enablement),
            _ => None,
        }
    }

    /// Detect relations based on textual patterns
    fn detect_pattern_based_relations(
        &self,
        sentence: &str,
        index: usize,
        all_sentences: &[String],
    ) -> Result<Vec<RhetoricalRelationType>, RhetoricalAnalysisError> {
        let mut relations = Vec::new();
        let normalized = sentence.to_lowercase();

        // Check for signal phrases
        for (relation_type, phrases) in &self.signal_phrases {
            for phrase in phrases {
                if normalized.contains(phrase) {
                    relations.push(relation_type.clone());
                    break;
                }
            }
        }

        // Check for structural patterns
        relations.extend(self.detect_structural_patterns(&normalized, index, all_sentences));

        Ok(relations)
    }

    /// Detect structural patterns indicating rhetorical relations
    fn detect_structural_patterns(
        &self,
        sentence: &str,
        index: usize,
        all_sentences: &[String],
    ) -> Vec<RhetoricalRelationType> {
        let mut relations = Vec::new();

        // Question-answer patterns
        if index > 0 && all_sentences[index - 1].trim_end().ends_with('?') {
            relations.push(RhetoricalRelationType::Solutionhood);
        }

        // List patterns (numbered or bulleted)
        if sentence
            .trim_start()
            .chars()
            .next()
            .map_or(false, |c| c.is_ascii_digit())
        {
            relations.push(RhetoricalRelationType::List);
        }

        // Definition patterns
        if sentence.contains(" is ")
            || sentence.contains(" means ")
            || sentence.contains(" refers to ")
        {
            relations.push(RhetoricalRelationType::Elaboration);
        }

        // Evidence patterns
        if sentence.contains("study") || sentence.contains("research") || sentence.contains("data")
        {
            relations.push(RhetoricalRelationType::Evidence);
        }

        // Background patterns
        if sentence.contains("historically")
            || sentence.contains("traditionally")
            || sentence.contains("previously")
        {
            relations.push(RhetoricalRelationType::Background);
        }

        relations
    }

    /// Detect implicit rhetorical relations between sentence pairs
    fn detect_implicit_relation(
        &self,
        sent1: &str,
        sent2: &str,
    ) -> Result<Option<RhetoricalRelationType>, RhetoricalAnalysisError> {
        let norm1 = sent1.to_lowercase();
        let norm2 = sent2.to_lowercase();

        // Semantic similarity suggests elaboration
        let semantic_similarity = self.calculate_semantic_similarity(&norm1, &norm2);
        if semantic_similarity > 0.7 {
            return Ok(Some(RhetoricalRelationType::Elaboration));
        }

        // Lexical repetition suggests restatement
        let lexical_overlap = self.calculate_lexical_overlap(&norm1, &norm2);
        if lexical_overlap > 0.5 {
            return Ok(Some(RhetoricalRelationType::Restatement));
        }

        // Temporal sequence detection
        if self.has_temporal_progression(&norm1, &norm2) {
            return Ok(Some(RhetoricalRelationType::Sequence));
        }

        // Contrast detection based on sentiment or polarity
        if self.has_contrasting_sentiment(&norm1, &norm2) {
            return Ok(Some(RhetoricalRelationType::Contrast));
        }

        Ok(None)
    }

    /// Calculate semantic similarity between sentences
    fn calculate_semantic_similarity(&self, sent1: &str, sent2: &str) -> f64 {
        let words1: std::collections::HashSet<&str> = sent1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = sent2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    /// Calculate lexical overlap between sentences
    fn calculate_lexical_overlap(&self, sent1: &str, sent2: &str) -> f64 {
        let content_words1 = self.extract_content_words(sent1);
        let content_words2 = self.extract_content_words(sent2);

        if content_words1.is_empty() || content_words2.is_empty() {
            return 0.0;
        }

        let set1: std::collections::HashSet<_> = content_words1.iter().collect();
        let set2: std::collections::HashSet<_> = content_words2.iter().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        intersection as f64 / union as f64
    }

    /// Extract content words from sentence
    fn extract_content_words(&self, sentence: &str) -> Vec<String> {
        let stop_words: std::collections::HashSet<&str> = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after", "above",
            "below", "over", "under", "between", "is", "am", "are", "was", "were", "be", "been",
            "being", "have", "has", "had",
        ]
        .iter()
        .cloned()
        .collect();

        sentence
            .split_whitespace()
            .filter_map(|word| {
                let clean_word = word
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();
                if clean_word.len() > 2 && !stop_words.contains(clean_word.as_str()) {
                    Some(clean_word)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check for temporal progression between sentences
    fn has_temporal_progression(&self, sent1: &str, sent2: &str) -> bool {
        let temporal_markers1 = self.count_temporal_markers(sent1);
        let temporal_markers2 = self.count_temporal_markers(sent2);

        temporal_markers1 > 0 && temporal_markers2 > 0
    }

    /// Count temporal markers in sentence
    fn count_temporal_markers(&self, sentence: &str) -> usize {
        let temporal_words = [
            "first",
            "second",
            "third",
            "then",
            "next",
            "after",
            "before",
            "during",
            "while",
            "when",
            "subsequently",
            "finally",
            "later",
        ];

        let normalized = sentence.to_lowercase();
        temporal_words
            .iter()
            .filter(|&&word| normalized.contains(word))
            .count()
    }

    /// Check for contrasting sentiment between sentences
    fn has_contrasting_sentiment(&self, sent1: &str, sent2: &str) -> bool {
        let positive_words = [
            "good",
            "great",
            "excellent",
            "positive",
            "beneficial",
            "advantage",
        ];
        let negative_words = [
            "bad",
            "poor",
            "terrible",
            "negative",
            "harmful",
            "disadvantage",
        ];

        let pos_count1 = self.count_sentiment_words(sent1, &positive_words);
        let neg_count1 = self.count_sentiment_words(sent1, &negative_words);

        let pos_count2 = self.count_sentiment_words(sent2, &positive_words);
        let neg_count2 = self.count_sentiment_words(sent2, &negative_words);

        // Check if sentiments are opposite
        (pos_count1 > neg_count1 && neg_count2 > pos_count2)
            || (neg_count1 > pos_count1 && pos_count2 > neg_count2)
    }

    /// Count sentiment words in sentence
    fn count_sentiment_words(&self, sentence: &str, words: &[&str]) -> usize {
        let normalized = sentence.to_lowercase();
        words
            .iter()
            .filter(|&&word| normalized.contains(word))
            .count()
    }

    /// Calculate relation distribution score
    fn calculate_relation_distribution_score(
        &self,
        relations: &HashMap<RhetoricalRelationType, usize>,
    ) -> f64 {
        if relations.is_empty() {
            return 0.0;
        }

        let total_relations: usize = relations.values().sum();
        let unique_relations = relations.len();

        // Calculate entropy-based diversity score
        let mut entropy = 0.0;
        for &count in relations.values() {
            let probability = count as f64 / total_relations as f64;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }

        // Normalize by maximum possible entropy
        let max_entropy = (unique_relations as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Calculate structural complexity
    fn calculate_structural_complexity(
        &self,
        relations: &HashMap<RhetoricalRelationType, usize>,
        sentence_count: usize,
    ) -> f64 {
        if sentence_count == 0 {
            return 0.0;
        }

        let total_relations: usize = relations.values().sum();
        let relation_density = total_relations as f64 / sentence_count as f64;

        // Weight different relation types by complexity
        let mut complexity_score = 0.0;
        for (relation_type, &count) in relations {
            let complexity_weight = self.get_relation_complexity_weight(relation_type);
            complexity_score += count as f64 * complexity_weight;
        }

        // Normalize by total relations and apply density factor
        if total_relations > 0 {
            (complexity_score / total_relations as f64) * (1.0 + relation_density * 0.5)
        } else {
            0.0
        }
    }

    /// Get complexity weight for relation type
    fn get_relation_complexity_weight(&self, relation_type: &RhetoricalRelationType) -> f64 {
        match relation_type {
            RhetoricalRelationType::Antithesis => 1.0,
            RhetoricalRelationType::Concession => 0.9,
            RhetoricalRelationType::Evidence => 0.8,
            RhetoricalRelationType::Contrast => 0.7,
            RhetoricalRelationType::Justify => 0.7,
            RhetoricalRelationType::Motivation => 0.6,
            RhetoricalRelationType::Enablement => 0.6,
            RhetoricalRelationType::Solutionhood => 0.6,
            RhetoricalRelationType::Background => 0.5,
            RhetoricalRelationType::Circumstance => 0.5,
            RhetoricalRelationType::Sequence => 0.4,
            RhetoricalRelationType::Elaboration => 0.3,
            RhetoricalRelationType::Summary => 0.3,
            RhetoricalRelationType::Restatement => 0.2,
            RhetoricalRelationType::Joint => 0.2,
            RhetoricalRelationType::List => 0.1,
        }
    }

    /// Calculate confidence scores for relations
    fn calculate_relation_confidences(
        &self,
        relations: &HashMap<RhetoricalRelationType, usize>,
        sentences: &[String],
        markers: &[DiscourseMarker],
    ) -> HashMap<RhetoricalRelationType, f64> {
        let mut confidences = HashMap::new();

        for (relation_type, &count) in relations {
            let base_confidence = 0.6; // Base confidence level

            // Adjust based on supporting discourse markers
            let marker_support = self.calculate_marker_support(relation_type, markers);

            // Adjust based on textual evidence strength
            let textual_evidence = self.calculate_textual_evidence(relation_type, sentences);

            // Adjust based on relation frequency (more frequent = higher confidence)
            let frequency_factor = (count as f64).ln() / 10.0; // Logarithmic scaling

            let confidence =
                base_confidence + marker_support * 0.3 + textual_evidence * 0.2 + frequency_factor;
            confidences.insert(relation_type.clone(), confidence.min(1.0));
        }

        confidences
    }

    /// Calculate marker support for relation type
    fn calculate_marker_support(
        &self,
        relation_type: &RhetoricalRelationType,
        markers: &[DiscourseMarker],
    ) -> f64 {
        let supporting_markers = markers
            .iter()
            .filter(|marker| {
                self.marker_to_relation(&marker.marker_type) == Some(relation_type.clone())
            })
            .count();

        if markers.is_empty() {
            0.0
        } else {
            supporting_markers as f64 / markers.len() as f64
        }
    }

    /// Calculate textual evidence strength for relation type
    fn calculate_textual_evidence(
        &self,
        relation_type: &RhetoricalRelationType,
        sentences: &[String],
    ) -> f64 {
        let signal_phrases = self
            .signal_phrases
            .get(relation_type)
            .unwrap_or(&Vec::new());
        if signal_phrases.is_empty() {
            return 0.5; // Neutral score when no signal phrases defined
        }

        let mut evidence_count = 0;
        let total_sentences = sentences.len();

        for sentence in sentences {
            let normalized = sentence.to_lowercase();
            for phrase in signal_phrases {
                if normalized.contains(phrase) {
                    evidence_count += 1;
                    break; // Count each sentence at most once per relation type
                }
            }
        }

        if total_sentences > 0 {
            evidence_count as f64 / total_sentences as f64
        } else {
            0.0
        }
    }

    /// Build relation patterns dictionary
    fn build_relation_patterns() -> HashMap<String, RhetoricalRelationType> {
        let mut patterns = HashMap::new();

        // Elaboration patterns
        patterns.insert("that is".to_string(), RhetoricalRelationType::Elaboration);
        patterns.insert(
            "in other words".to_string(),
            RhetoricalRelationType::Elaboration,
        );
        patterns.insert(
            "specifically".to_string(),
            RhetoricalRelationType::Elaboration,
        );

        // Evidence patterns
        patterns.insert("studies show".to_string(), RhetoricalRelationType::Evidence);
        patterns.insert(
            "research indicates".to_string(),
            RhetoricalRelationType::Evidence,
        );
        patterns.insert("according to".to_string(), RhetoricalRelationType::Evidence);

        // Contrast patterns
        patterns.insert("in contrast".to_string(), RhetoricalRelationType::Contrast);
        patterns.insert(
            "on the contrary".to_string(),
            RhetoricalRelationType::Contrast,
        );
        patterns.insert(
            "different from".to_string(),
            RhetoricalRelationType::Contrast,
        );

        patterns
    }

    /// Build signal phrases for each relation type
    fn build_signal_phrases() -> HashMap<RhetoricalRelationType, Vec<String>> {
        let mut phrases = HashMap::new();

        phrases.insert(
            RhetoricalRelationType::Elaboration,
            vec![
                "that is".to_string(),
                "in other words".to_string(),
                "specifically".to_string(),
                "for example".to_string(),
                "for instance".to_string(),
                "namely".to_string(),
            ],
        );

        phrases.insert(
            RhetoricalRelationType::Evidence,
            vec![
                "studies show".to_string(),
                "research indicates".to_string(),
                "data suggests".to_string(),
                "according to".to_string(),
                "evidence suggests".to_string(),
            ],
        );

        phrases.insert(
            RhetoricalRelationType::Contrast,
            vec![
                "however".to_string(),
                "in contrast".to_string(),
                "on the contrary".to_string(),
                "conversely".to_string(),
                "alternatively".to_string(),
            ],
        );

        phrases.insert(
            RhetoricalRelationType::Sequence,
            vec![
                "first".to_string(),
                "then".to_string(),
                "next".to_string(),
                "finally".to_string(),
                "subsequently".to_string(),
            ],
        );

        phrases.insert(
            RhetoricalRelationType::Summary,
            vec![
                "in conclusion".to_string(),
                "to sum up".to_string(),
                "in summary".to_string(),
                "overall".to_string(),
                "in brief".to_string(),
            ],
        );

        phrases
    }
}

/// Builder for discourse trees
struct DiscourseTreeBuilder {
    config: RhetoricalStructureConfig,
}

impl DiscourseTreeBuilder {
    fn new(config: RhetoricalStructureConfig) -> Self {
        Self { config }
    }

    fn build_tree(
        &self,
        sentences: &[String],
        relations: &HashMap<RhetoricalRelationType, usize>,
        markers: &[DiscourseMarker],
    ) -> Result<DiscourseTree, RhetoricalAnalysisError> {
        if sentences.is_empty() {
            return Err(RhetoricalAnalysisError::TreeConstructionFailed(
                "No sentences provided".to_string(),
            ));
        }

        // Build tree structure bottom-up
        let mut nodes = self.create_leaf_nodes(sentences);
        let root = self.build_tree_recursive(&mut nodes, relations, markers)?;

        let depth = self.calculate_tree_depth(&root);
        let node_count = self.count_nodes(&root);
        let balance_score = self.calculate_balance_score(&root);
        let complexity_score = self.calculate_tree_complexity(&root, relations);

        Ok(DiscourseTree {
            root,
            depth,
            node_count,
            balance_score,
            complexity_score,
        })
    }

    fn create_leaf_nodes(&self, sentences: &[String]) -> Vec<DiscourseNode> {
        sentences
            .iter()
            .enumerate()
            .map(|(i, sentence)| DiscourseNode {
                node_id: i,
                relation_type: RhetoricalRelationType::Elaboration, // Default for leaves
                nuclearity: "nucleus".to_string(),                  // Default nuclearity
                text_span: (0, sentence.len()),                     // Simplified span
                children: Vec::new(),
                confidence: 0.8, // Base confidence for leaf nodes
                salience: self.calculate_sentence_salience(sentence),
            })
            .collect()
    }

    fn build_tree_recursive(
        &self,
        nodes: &mut Vec<DiscourseNode>,
        relations: &HashMap<RhetoricalRelationType, usize>,
        _markers: &[DiscourseMarker],
    ) -> Result<DiscourseNode, RhetoricalAnalysisError> {
        if nodes.len() == 1 {
            return Ok(nodes.pop().unwrap());
        }

        // Find best pair to combine based on rhetorical relations
        let (left_idx, right_idx, relation_type) = self.find_best_combination(nodes, relations)?;

        // Create parent node
        let left_node = nodes.remove(left_idx);
        let right_node = nodes.remove(if right_idx > left_idx {
            right_idx - 1
        } else {
            right_idx
        });

        let parent_node = DiscourseNode {
            node_id: nodes.len() + 100, // Ensure unique ID
            relation_type,
            nuclearity: "nucleus".to_string(),
            text_span: (
                left_node.text_span.0.min(right_node.text_span.0),
                left_node.text_span.1.max(right_node.text_span.1),
            ),
            children: vec![left_node, right_node],
            confidence: 0.7,
            salience: 0.5,
        };

        nodes.push(parent_node);

        self.build_tree_recursive(nodes, relations, _markers)
    }

    fn find_best_combination(
        &self,
        nodes: &[DiscourseNode],
        relations: &HashMap<RhetoricalRelationType, usize>,
    ) -> Result<(usize, usize, RhetoricalRelationType), RhetoricalAnalysisError> {
        if nodes.len() < 2 {
            return Err(RhetoricalAnalysisError::TreeConstructionFailed(
                "Not enough nodes to combine".to_string(),
            ));
        }

        // Simple heuristic: combine adjacent nodes with the most common relation type
        let most_common_relation = relations
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(relation, _)| relation.clone())
            .unwrap_or(RhetoricalRelationType::Elaboration);

        Ok((0, 1, most_common_relation))
    }

    fn calculate_sentence_salience(&self, sentence: &str) -> f64 {
        // Simple salience calculation based on sentence length and content words
        let word_count = sentence.split_whitespace().count();
        let base_salience = (word_count as f64).ln() / 10.0;

        // Boost salience for sentences with important markers
        let important_phrases = ["important", "significant", "key", "main", "primary"];
        let importance_boost = if important_phrases
            .iter()
            .any(|phrase| sentence.to_lowercase().contains(phrase))
        {
            0.2
        } else {
            0.0
        };

        (base_salience + importance_boost).min(1.0)
    }

    fn calculate_tree_depth(&self, node: &DiscourseNode) -> usize {
        if node.children.is_empty() {
            1
        } else {
            1 + node
                .children
                .iter()
                .map(|child| self.calculate_tree_depth(child))
                .max()
                .unwrap_or(0)
        }
    }

    fn count_nodes(&self, node: &DiscourseNode) -> usize {
        1 + node
            .children
            .iter()
            .map(|child| self.count_nodes(child))
            .sum::<usize>()
    }

    fn calculate_balance_score(&self, node: &DiscourseNode) -> f64 {
        if node.children.is_empty() {
            return 1.0; // Leaf nodes are perfectly balanced
        }

        let child_sizes: Vec<usize> = node
            .children
            .iter()
            .map(|child| self.count_nodes(child))
            .collect();

        if child_sizes.len() < 2 {
            return 1.0;
        }

        // Calculate balance as inverse of size variance
        let mean_size = child_sizes.iter().sum::<usize>() as f64 / child_sizes.len() as f64;
        let variance = child_sizes
            .iter()
            .map(|&size| (size as f64 - mean_size).powi(2))
            .sum::<f64>()
            / child_sizes.len() as f64;

        // Convert variance to balance score (lower variance = higher balance)
        if variance == 0.0 {
            1.0
        } else {
            1.0 / (1.0 + variance.sqrt())
        }
    }

    fn calculate_tree_complexity(
        &self,
        node: &DiscourseNode,
        relations: &HashMap<RhetoricalRelationType, usize>,
    ) -> f64 {
        let depth_factor = self.calculate_tree_depth(node) as f64 / 10.0;
        let relation_diversity = relations.len() as f64 / 16.0; // Max 16 relation types
        let node_density = self.count_nodes(node) as f64 / 100.0; // Normalize by expected max

        (depth_factor + relation_diversity + node_density) / 3.0
    }
}

/// Analyzer for nucleus-satellite structures
struct NucleusSatelliteAnalyzer {
    config: RhetoricalStructureConfig,
}

impl NucleusSatelliteAnalyzer {
    fn new(config: RhetoricalStructureConfig) -> Self {
        Self { config }
    }

    fn analyze(
        &self,
        sentences: &[String],
        relations: &HashMap<RhetoricalRelationType, usize>,
        markers: &[DiscourseMarker],
    ) -> Result<NucleusSatelliteAnalysis, RhetoricalAnalysisError> {
        let nucleus_accuracy = self.calculate_nucleus_accuracy(sentences, markers);
        let satellite_patterns = self.analyze_satellite_patterns(sentences, relations);
        let nuclear_chains = self.identify_nuclear_chains(sentences, markers)?;
        let embedding_statistics = self.calculate_embedding_statistics(&nuclear_chains);

        Ok(NucleusSatelliteAnalysis {
            nucleus_accuracy,
            satellite_patterns,
            nuclear_chains,
            embedding_statistics,
        })
    }

    fn calculate_nucleus_accuracy(&self, sentences: &[String], markers: &[DiscourseMarker]) -> f64 {
        // Simplified nucleus detection accuracy
        // In a real implementation, this would compare against gold standard annotations

        let potential_nuclei = sentences.len();
        let marker_supported_nuclei = markers
            .iter()
            .filter(|m| self.is_nucleus_supporting_marker(&m.marker_type))
            .count();

        if potential_nuclei > 0 {
            marker_supported_nuclei as f64 / potential_nuclei as f64
        } else {
            0.0
        }
    }

    fn is_nucleus_supporting_marker(
        &self,
        marker_type: &super::config::DiscourseMarkerType,
    ) -> bool {
        matches!(
            marker_type,
            super::config::DiscourseMarkerType::Summary
                | super::config::DiscourseMarkerType::Emphasis
                | super::config::DiscourseMarkerType::Cause
        )
    }

    fn analyze_satellite_patterns(
        &self,
        sentences: &[String],
        relations: &HashMap<RhetoricalRelationType, usize>,
    ) -> HashMap<String, usize> {
        let mut patterns = HashMap::new();

        // Analyze common satellite attachment patterns
        for (relation_type, &count) in relations {
            let pattern_name = match relation_type {
                RhetoricalRelationType::Elaboration => "elaborative_satellite",
                RhetoricalRelationType::Evidence => "evidential_satellite",
                RhetoricalRelationType::Background => "background_satellite",
                RhetoricalRelationType::Circumstance => "circumstantial_satellite",
                _ => "other_satellite",
            };

            *patterns.entry(pattern_name.to_string()).or_insert(0) += count;
        }

        patterns
    }

    fn identify_nuclear_chains(
        &self,
        sentences: &[String],
        _markers: &[DiscourseMarker],
    ) -> Result<Vec<NuclearChain>, RhetoricalAnalysisError> {
        let mut chains = Vec::new();

        // Simplified nuclear chain identification
        // In practice, this would involve sophisticated discourse parsing

        let chain = NuclearChain {
            chain_id: 0,
            nuclear_elements: (0..sentences.len()).collect(),
            satellites: Vec::new(),
            coherence_score: 0.7, // Placeholder
        };

        chains.push(chain);
        Ok(chains)
    }

    fn calculate_embedding_statistics(&self, chains: &[NuclearChain]) -> EmbeddingStats {
        let max_depth = chains.len(); // Simplified calculation
        let average_depth = if chains.is_empty() {
            0.0
        } else {
            chains.len() as f64
        };

        let mut depth_distribution = HashMap::new();
        for i in 0..chains.len() {
            *depth_distribution.entry(i + 1).or_insert(0) += 1;
        }

        EmbeddingStats {
            max_depth,
            average_depth,
            depth_distribution,
        }
    }
}
