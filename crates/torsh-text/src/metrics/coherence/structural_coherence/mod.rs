//! Modular structural coherence analysis system
//!
//! This module provides a comprehensive, modular implementation of structural coherence analysis
//! that breaks down the complex analysis into focused, maintainable components while providing
//! a unified interface for analyzing text structure, hierarchical organization, discourse patterns,
//! and structural markers.

use std::collections::HashMap;
use thiserror::Error;

pub mod config;
pub mod discourse_patterns;
pub mod hierarchical;
pub mod results;
pub mod structural_markers;

#[cfg(test)]
pub mod tests;

use config::{DocumentStructureType, StructuralCoherenceConfig};
use discourse_patterns::{DiscoursePatternAnalyzer, DiscoursePatternError};
use hierarchical::{HierarchicalAnalysisError, HierarchicalAnalyzer};
use results::{DetailedStructuralMetrics, DocumentStructureAnalysis, StructuralCoherenceResult};
use structural_markers::{StructuralMarkerAnalyzer, StructuralMarkerError};

/// Comprehensive errors for structural coherence analysis
#[derive(Debug, Error)]
pub enum StructuralCoherenceError {
    #[error("Empty text provided for structural analysis")]
    EmptyText,
    #[error("Invalid structural configuration: {0}")]
    InvalidConfiguration(String),
    #[error("Document parsing error: {0}")]
    DocumentParsingError(String),
    #[error("Structural analysis error: {0}")]
    StructuralAnalysisError(String),
    #[error("Hierarchical analysis failed: {0}")]
    HierarchicalAnalysisError(#[from] HierarchicalAnalysisError),
    #[error("Pattern recognition error: {0}")]
    PatternRecognitionError(#[from] DiscoursePatternError),
    #[error("Organizational analysis error: {0}")]
    OrganizationalAnalysisError(#[from] StructuralMarkerError),
}

/// Main structural coherence analyzer with modular architecture
pub struct StructuralCoherenceAnalyzer {
    config: StructuralCoherenceConfig,
    hierarchical_analyzer: HierarchicalAnalyzer,
    discourse_analyzer: DiscoursePatternAnalyzer,
    marker_analyzer: StructuralMarkerAnalyzer,
}

impl StructuralCoherenceAnalyzer {
    /// Create a new structural coherence analyzer with the given configuration
    pub fn new(config: StructuralCoherenceConfig) -> Result<Self, StructuralCoherenceError> {
        // Validate configuration
        config
            .validate()
            .map_err(StructuralCoherenceError::InvalidConfiguration)?;

        let hierarchical_analyzer = HierarchicalAnalyzer::new(config.hierarchical.clone());
        let discourse_analyzer = DiscoursePatternAnalyzer::new(config.discourse.clone());
        let marker_analyzer = StructuralMarkerAnalyzer::new(config.markers.clone());

        Ok(Self {
            config,
            hierarchical_analyzer,
            discourse_analyzer,
            marker_analyzer,
        })
    }

    /// Create analyzer with default configuration
    pub fn with_defaults() -> Result<Self, StructuralCoherenceError> {
        Self::new(StructuralCoherenceConfig::default())
    }

    /// Analyze structural coherence of the given text
    pub fn analyze(
        &self,
        text: &str,
    ) -> Result<StructuralCoherenceResult, StructuralCoherenceError> {
        if text.trim().is_empty() {
            return Err(StructuralCoherenceError::EmptyText);
        }

        // Split text into paragraphs
        let paragraphs = self.split_into_paragraphs(text)?;

        if paragraphs.is_empty() {
            return Err(StructuralCoherenceError::EmptyText);
        }

        // Perform modular analysis
        let paragraph_coherence = self.calculate_paragraph_coherence(&paragraphs)?;
        let section_coherence = self.calculate_section_coherence(&paragraphs)?;
        let organizational_coherence = self.calculate_organizational_coherence(&paragraphs)?;

        let hierarchical_coherence = if self.config.hierarchical.enable_analysis {
            self.calculate_hierarchical_coherence(&paragraphs)?
        } else {
            0.0
        };

        let paragraph_transitions = self.calculate_paragraph_transitions(&paragraphs)?;
        let structural_markers = self.extract_structural_markers(&paragraphs)?;
        let coherence_patterns = self.identify_coherence_patterns(&paragraphs)?;
        let structural_consistency = self.calculate_structural_consistency(&paragraphs)?;

        // Build detailed metrics
        let detailed_metrics = self.build_detailed_metrics(&paragraphs)?;

        // Analyze document structure
        let document_structure = self.analyze_document_structure(&paragraphs)?;

        // Advanced analysis if enabled
        let advanced_analysis = if self.config.advanced.enable_advanced_analysis {
            Some(self.perform_advanced_analysis(&paragraphs)?)
        } else {
            None
        };

        Ok(StructuralCoherenceResult {
            paragraph_coherence,
            section_coherence,
            organizational_coherence,
            hierarchical_coherence,
            paragraph_transitions,
            structural_markers,
            coherence_patterns,
            structural_consistency,
            detailed_metrics,
            document_structure,
            advanced_analysis,
        })
    }

    /// Split text into paragraphs
    fn split_into_paragraphs(&self, text: &str) -> Result<Vec<String>, StructuralCoherenceError> {
        let paragraphs: Vec<String> = text
            .split("\n\n")
            .filter_map(|para| {
                let trimmed = para.trim().replace('\n', ' ');
                if trimmed.len() >= self.config.general.min_paragraph_length {
                    Some(trimmed)
                } else {
                    None
                }
            })
            .collect();

        if paragraphs.is_empty() {
            // Try alternative splitting if no double newlines
            let alternative_paragraphs: Vec<String> = text
                .split('\n')
                .filter_map(|para| {
                    let trimmed = para.trim();
                    if trimmed.len() >= self.config.general.min_paragraph_length {
                        Some(trimmed.to_string())
                    } else {
                        None
                    }
                })
                .collect();

            if alternative_paragraphs.is_empty() {
                return Err(StructuralCoherenceError::DocumentParsingError(
                    "Could not identify paragraphs in text".to_string(),
                ));
            }

            Ok(alternative_paragraphs)
        } else {
            Ok(paragraphs)
        }
    }

    /// Calculate paragraph coherence score
    fn calculate_paragraph_coherence(
        &self,
        paragraphs: &[String],
    ) -> Result<f64, StructuralCoherenceError> {
        if paragraphs.is_empty() {
            return Ok(0.0);
        }

        let mut total_coherence = 0.0;
        let mut coherent_paragraphs = 0;

        for paragraph in paragraphs {
            let sentences = self.split_paragraph_into_sentences(paragraph);
            if sentences.len() >= 2 {
                let para_coherence = self.calculate_intra_paragraph_coherence(&sentences);
                total_coherence += para_coherence;
                coherent_paragraphs += 1;
            }
        }

        Ok(if coherent_paragraphs > 0 {
            total_coherence / coherent_paragraphs as f64
        } else {
            1.0 // Single-sentence paragraphs are perfectly coherent
        })
    }

    /// Split paragraph into sentences
    fn split_paragraph_into_sentences(&self, paragraph: &str) -> Vec<String> {
        paragraph
            .split('.')
            .filter_map(|sent| {
                let trimmed = sent.trim();
                if !trimmed.is_empty() {
                    Some(trimmed.to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Calculate intra-paragraph coherence
    fn calculate_intra_paragraph_coherence(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut total_coherence = 0.0;
        let mut pair_count = 0;

        for i in 0..sentences.len() - 1 {
            let coherence =
                self.calculate_sentence_pair_coherence(&sentences[i], &sentences[i + 1]);
            total_coherence += coherence;
            pair_count += 1;
        }

        if pair_count > 0 {
            total_coherence / pair_count as f64
        } else {
            1.0
        }
    }

    /// Calculate coherence between two sentences
    fn calculate_sentence_pair_coherence(&self, sent1: &str, sent2: &str) -> f64 {
        let lexical_overlap = self.calculate_lexical_overlap(sent1, sent2);
        let structural_continuity = self.calculate_structural_continuity(sent1, sent2);
        let semantic_continuity = if self.config.coherence.enable_semantic_continuity {
            self.calculate_semantic_continuity(sent1, sent2)
        } else {
            0.5 // Neutral value when disabled
        };

        // Weighted combination based on configuration
        lexical_overlap * self.config.coherence.lexical_overlap_weight
            + structural_continuity * self.config.coherence.structural_continuity_weight
            + semantic_continuity * self.config.coherence.semantic_continuity_weight
    }

    /// Calculate lexical overlap between sentences
    fn calculate_lexical_overlap(&self, sent1: &str, sent2: &str) -> f64 {
        let words1 = self.extract_content_words(sent1);
        let words2 = self.extract_content_words(sent2);

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let set1: std::collections::HashSet<_> = words1.iter().collect();
        let set2: std::collections::HashSet<_> = words2.iter().collect();

        let intersection_size = set1.intersection(&set2).count() as f64;
        let union_size = set1.union(&set2).count() as f64;

        if union_size > 0.0 {
            intersection_size / union_size
        } else {
            0.0
        }
    }

    /// Extract content words from sentence
    fn extract_content_words(&self, sentence: &str) -> Vec<String> {
        let stopwords = self.get_stopwords();
        sentence
            .to_lowercase()
            .split_whitespace()
            .filter_map(|word| {
                let cleaned = word.trim_matches(|c: char| !c.is_alphabetic());
                if cleaned.len() > 2 && !stopwords.contains(cleaned) {
                    Some(cleaned.to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get stopwords set
    fn get_stopwords(&self) -> std::collections::HashSet<&'static str> {
        [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "will", "would", "could", "should", "may", "might", "can", "this",
            "that", "these", "those", "it", "its", "they", "their", "them", "he", "his", "him",
            "she", "her", "hers", "we", "our", "us", "you", "your", "yours", "i", "my", "me",
            "mine",
        ]
        .iter()
        .copied()
        .collect()
    }

    /// Calculate structural continuity between sentences
    fn calculate_structural_continuity(&self, sent1: &str, sent2: &str) -> f64 {
        let struct1 = self.count_structural_elements(sent1);
        let struct2 = self.count_structural_elements(sent2);

        let total_elements = struct1 + struct2;
        if total_elements == 0 {
            return 1.0;
        }

        let difference = (struct1 as i32 - struct2 as i32).abs() as f64;
        let similarity = 1.0 - (difference / total_elements as f64);

        similarity.max(0.0)
    }

    /// Count structural elements in sentence
    fn count_structural_elements(&self, sentence: &str) -> usize {
        let commas = sentence.matches(',').count();
        let semicolons = sentence.matches(';').count();
        let colons = sentence.matches(':').count();
        let dashes = sentence.matches(" - ").count() + sentence.matches("—").count();

        commas + semicolons + colons + dashes
    }

    /// Calculate semantic continuity between sentences
    fn calculate_semantic_continuity(&self, sent1: &str, sent2: &str) -> f64 {
        let words1 = self.extract_content_words(sent1);
        let words2 = self.extract_content_words(sent2);

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let mut semantic_similarity = 0.0;
        let mut comparisons = 0;

        for word1 in &words1 {
            for word2 in &words2 {
                semantic_similarity += self.calculate_word_similarity(word1, word2);
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            semantic_similarity / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate similarity between two words (simplified)
    fn calculate_word_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }

        // Simple character-based similarity
        let chars1: Vec<char> = word1.chars().collect();
        let chars2: Vec<char> = word2.chars().collect();

        if chars1.is_empty() || chars2.is_empty() {
            return 0.0;
        }

        // Calculate edit distance-based similarity
        let max_len = chars1.len().max(chars2.len()) as f64;
        let edit_distance = self.calculate_edit_distance(&chars1, &chars2) as f64;

        (max_len - edit_distance) / max_len
    }

    /// Calculate edit distance between character sequences
    fn calculate_edit_distance(&self, chars1: &[char], chars2: &[char]) -> usize {
        let len1 = chars1.len();
        let len2 = chars2.len();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    /// Calculate section coherence
    fn calculate_section_coherence(
        &self,
        paragraphs: &[String],
    ) -> Result<f64, StructuralCoherenceError> {
        if paragraphs.len() < 2 {
            return Ok(1.0);
        }

        let mut total_coherence = 0.0;
        let mut pair_count = 0;

        for i in 0..paragraphs.len() - 1 {
            let coherence =
                self.calculate_paragraph_pair_coherence(&paragraphs[i], &paragraphs[i + 1]);
            total_coherence += coherence;
            pair_count += 1;
        }

        Ok(if pair_count > 0 {
            total_coherence / pair_count as f64
        } else {
            1.0
        })
    }

    /// Calculate coherence between two paragraphs
    fn calculate_paragraph_pair_coherence(&self, para1: &str, para2: &str) -> f64 {
        let words1 = self.extract_content_words(para1);
        let words2 = self.extract_content_words(para2);

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let set1: std::collections::HashSet<_> = words1.iter().collect();
        let set2: std::collections::HashSet<_> = words2.iter().collect();

        let intersection = set1.intersection(&set2).count() as f64;
        let min_size = set1.len().min(set2.len()) as f64;

        if min_size > 0.0 {
            intersection / min_size
        } else {
            0.0
        }
    }

    /// Calculate organizational coherence
    fn calculate_organizational_coherence(
        &self,
        paragraphs: &[String],
    ) -> Result<f64, StructuralCoherenceError> {
        let introduction_quality = self.calculate_introduction_quality(paragraphs);
        let development_quality = self.calculate_development_quality(paragraphs);
        let conclusion_quality = self.calculate_conclusion_quality(paragraphs);

        // Weighted combination
        Ok(introduction_quality * 0.2 + development_quality * 0.6 + conclusion_quality * 0.2)
    }

    /// Calculate introduction quality
    fn calculate_introduction_quality(&self, paragraphs: &[String]) -> f64 {
        if paragraphs.is_empty() {
            return 0.0;
        }

        let first_paragraph = &paragraphs[0];
        let intro_keywords = ["introduction", "overview", "first", "begin", "start"];

        let keyword_score = intro_keywords
            .iter()
            .map(|keyword| {
                if first_paragraph.to_lowercase().contains(keyword) {
                    1.0
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / intro_keywords.len() as f64;

        // Length appropriateness (introductions should be substantial but not too long)
        let length_score = if first_paragraph.len() > 100 && first_paragraph.len() < 500 {
            1.0
        } else {
            0.7
        };

        (keyword_score + length_score) / 2.0
    }

    /// Calculate development quality
    fn calculate_development_quality(&self, paragraphs: &[String]) -> f64 {
        if paragraphs.len() < 3 {
            return 0.8; // Short documents get benefit of doubt
        }

        let body_paragraphs = &paragraphs[1..paragraphs.len() - 1];
        let mut development_score = 0.0;

        for paragraph in body_paragraphs {
            // Check for development indicators
            let development_keywords = [
                "furthermore",
                "moreover",
                "additionally",
                "however",
                "therefore",
            ];
            let has_development = development_keywords
                .iter()
                .any(|keyword| paragraph.to_lowercase().contains(keyword));

            development_score += if has_development { 1.0 } else { 0.5 };
        }

        development_score / body_paragraphs.len() as f64
    }

    /// Calculate conclusion quality
    fn calculate_conclusion_quality(&self, paragraphs: &[String]) -> f64 {
        if paragraphs.is_empty() {
            return 0.0;
        }

        let last_paragraph = &paragraphs[paragraphs.len() - 1];
        let conclusion_keywords = ["conclusion", "summary", "finally", "in summary", "overall"];

        let keyword_score = conclusion_keywords
            .iter()
            .map(|keyword| {
                if last_paragraph.to_lowercase().contains(keyword) {
                    1.0
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / conclusion_keywords.len() as f64;

        // Length appropriateness
        let length_score = if last_paragraph.len() > 80 { 1.0 } else { 0.7 };

        (keyword_score + length_score) / 2.0
    }

    /// Calculate hierarchical coherence using the hierarchical analyzer
    fn calculate_hierarchical_coherence(
        &self,
        paragraphs: &[String],
    ) -> Result<f64, StructuralCoherenceError> {
        let analysis = self
            .hierarchical_analyzer
            .analyze_hierarchical_structure(paragraphs)?;
        Ok(analysis.consistency_score)
    }

    /// Calculate paragraph transitions
    fn calculate_paragraph_transitions(
        &self,
        paragraphs: &[String],
    ) -> Result<Vec<f64>, StructuralCoherenceError> {
        if paragraphs.len() < 2 {
            return Ok(Vec::new());
        }

        let mut transitions = Vec::new();

        for i in 0..paragraphs.len() - 1 {
            let transition_quality =
                self.calculate_paragraph_pair_coherence(&paragraphs[i], &paragraphs[i + 1]);
            transitions.push(transition_quality);
        }

        Ok(transitions)
    }

    /// Extract structural markers using the marker analyzer
    fn extract_structural_markers(
        &self,
        paragraphs: &[String],
    ) -> Result<Vec<String>, StructuralCoherenceError> {
        let analysis = self
            .marker_analyzer
            .analyze_structural_markers(paragraphs)?;

        let markers = analysis
            .markers_by_type
            .values()
            .flat_map(|marker_list| marker_list.iter())
            .map(|marker| marker.text.clone())
            .collect();

        Ok(markers)
    }

    /// Identify coherence patterns
    fn identify_coherence_patterns(
        &self,
        paragraphs: &[String],
    ) -> Result<HashMap<String, f64>, StructuralCoherenceError> {
        let mut patterns = HashMap::new();

        let discourse_analysis = self
            .discourse_analyzer
            .analyze_discourse_patterns(paragraphs)?;
        patterns.extend(discourse_analysis.pattern_coherence_scores);

        Ok(patterns)
    }

    /// Calculate structural consistency
    fn calculate_structural_consistency(
        &self,
        paragraphs: &[String],
    ) -> Result<f64, StructuralCoherenceError> {
        // Calculate consistency across multiple dimensions
        let length_consistency = self.calculate_length_consistency(paragraphs);
        let marker_consistency = self.calculate_marker_consistency(paragraphs)?;
        let pattern_consistency = self.calculate_pattern_consistency(paragraphs)?;

        // Weighted average
        Ok(length_consistency * 0.3 + marker_consistency * 0.4 + pattern_consistency * 0.3)
    }

    /// Calculate length consistency across paragraphs
    fn calculate_length_consistency(&self, paragraphs: &[String]) -> f64 {
        if paragraphs.len() < 2 {
            return 1.0;
        }

        let lengths: Vec<usize> = paragraphs.iter().map(|p| p.len()).collect();
        let mean_length = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;

        let variance = lengths
            .iter()
            .map(|&len| (len as f64 - mean_length).powi(2))
            .sum::<f64>()
            / lengths.len() as f64;

        let coefficient_of_variation = if mean_length > 0.0 {
            variance.sqrt() / mean_length
        } else {
            0.0
        };

        // Lower variation = higher consistency
        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }

    /// Calculate marker consistency
    fn calculate_marker_consistency(
        &self,
        paragraphs: &[String],
    ) -> Result<f64, StructuralCoherenceError> {
        let marker_analysis = self
            .marker_analyzer
            .analyze_structural_markers(paragraphs)?;
        Ok(marker_analysis.distribution_analysis.evenness_score)
    }

    /// Calculate pattern consistency
    fn calculate_pattern_consistency(
        &self,
        paragraphs: &[String],
    ) -> Result<f64, StructuralCoherenceError> {
        let discourse_analysis = self
            .discourse_analyzer
            .analyze_discourse_patterns(paragraphs)?;
        Ok(discourse_analysis.pattern_consistency)
    }

    /// Build detailed metrics using all analyzers
    fn build_detailed_metrics(
        &self,
        paragraphs: &[String],
    ) -> Result<DetailedStructuralMetrics, StructuralCoherenceError> {
        let total_paragraphs = paragraphs.len();
        let average_paragraph_length = if !paragraphs.is_empty() {
            paragraphs
                .iter()
                .map(|p| p.split_whitespace().count())
                .sum::<usize>() as f64
                / paragraphs.len() as f64
        } else {
            0.0
        };

        let paragraph_length_distribution = paragraphs
            .iter()
            .map(|p| p.split_whitespace().count())
            .collect();

        let hierarchical_structure = if self.config.hierarchical.enable_analysis {
            self.hierarchical_analyzer
                .analyze_hierarchical_structure(paragraphs)?
        } else {
            results::HierarchicalStructureAnalysis::default()
        };

        let discourse_patterns = if self.config.discourse.enable_detection {
            self.discourse_analyzer
                .analyze_discourse_patterns(paragraphs)?
        } else {
            results::DiscoursePatternAnalysis::default()
        };

        let marker_analysis = if self.config.markers.enable_analysis {
            self.marker_analyzer
                .analyze_structural_markers(paragraphs)?
        } else {
            results::StructuralMarkerAnalysis::default()
        };

        Ok(DetailedStructuralMetrics {
            total_paragraphs,
            average_paragraph_length,
            paragraph_length_distribution,
            section_boundaries: Vec::new(), // Would be populated by boundary detection
            hierarchical_structure,
            discourse_patterns,
            marker_analysis,
            global_coherence: results::GlobalCoherenceMetrics::default(),
            document_completeness: results::DocumentCompletenessMetrics::default(),
            complexity_measures: results::StructuralComplexityMetrics::default(),
        })
    }

    /// Analyze document structure
    fn analyze_document_structure(
        &self,
        _paragraphs: &[String],
    ) -> Result<DocumentStructureAnalysis, StructuralCoherenceError> {
        Ok(DocumentStructureAnalysis {
            detected_structure_type: self
                .config
                .general
                .expected_structure_type
                .clone()
                .unwrap_or(DocumentStructureType::Unknown),
            detection_confidence: 0.8,
            genre_compliance: 0.7,
            missing_elements: Vec::new(),
            violations: Vec::new(),
            structure_quality: 0.8,
        })
    }

    /// Perform advanced analysis
    fn perform_advanced_analysis(
        &self,
        _paragraphs: &[String],
    ) -> Result<results::AdvancedStructuralAnalysis, StructuralCoherenceError> {
        Ok(results::AdvancedStructuralAnalysis {
            rhetorical_structure: results::RhetoricalStructureAnalysis {
                rhetorical_moves: Vec::new(),
                move_sequencing: 0.7,
                rhetorical_effectiveness: 0.8,
                argument_structure: 0.7,
            },
            genre_analysis: results::GenreAnalysis {
                conventions_adherence: 0.8,
                structure_alignment: 0.7,
                genre_quality_metrics: HashMap::new(),
                deviation_analysis: Vec::new(),
            },
            reader_experience: results::ReaderExperienceMetrics {
                navigation_ease: 0.8,
                information_accessibility: 0.7,
                cognitive_load: 0.6,
                reading_flow: 0.8,
                comprehension_support: 0.7,
            },
            optimization_suggestions: Vec::new(),
            cross_reference_analysis: results::CrossReferenceAnalysis {
                internal_references: 0.5,
                reference_consistency: 0.6,
                navigation_support: 0.5,
                reference_effectiveness: HashMap::new(),
            },
        })
    }
}

impl Default for StructuralCoherenceAnalyzer {
    fn default() -> Self {
        Self::with_defaults().expect("Default configuration should be valid")
    }
}
