//! Semantic Meaning Analysis Module
//!
//! This module provides comprehensive meaning preservation and analysis capabilities
//! for semantic fluency evaluation. It focuses on evaluating how well semantic
//! meaning is preserved and expressed throughout text.

use super::config::MeaningAnalysisConfig;
use super::results::{
    ConceptualMapping, CoreMeaningAnalysis, MeaningDistortion, MeaningPreservationMetrics,
    MeaningStabilityMetrics, SemanticDrift,
};
use crate::error::TextAnalysisError;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MeaningAnalysisError {
    #[error("Invalid meaning analysis configuration: {0}")]
    ConfigError(String),
    #[error("Meaning calculation failed: {0}")]
    CalculationError(String),
    #[error("Semantic meaning analysis error: {0}")]
    AnalysisError(String),
}

pub type MeaningResult<T> = Result<T, MeaningAnalysisError>;

/// Core semantic meaning analyzer providing comprehensive meaning preservation analysis
#[derive(Debug, Clone)]
pub struct SemanticMeaningAnalyzer {
    config: MeaningAnalysisConfig,
    concept_mappings: HashMap<String, ConceptualMapping>,
    meaning_hierarchies: BTreeMap<String, Vec<String>>,
    semantic_relationships: HashMap<String, HashMap<String, f64>>,
    meaning_cache: HashMap<u64, MeaningPreservationMetrics>,
    drift_patterns: Vec<SemanticDrift>,
    stability_tracker: HashMap<String, f64>,
}

impl SemanticMeaningAnalyzer {
    /// Create new meaning analyzer with configuration
    pub fn new(config: MeaningAnalysisConfig) -> MeaningResult<Self> {
        Self::validate_config(&config)?;

        Ok(Self {
            config,
            concept_mappings: HashMap::new(),
            meaning_hierarchies: BTreeMap::new(),
            semantic_relationships: HashMap::new(),
            meaning_cache: HashMap::new(),
            drift_patterns: Vec::new(),
            stability_tracker: HashMap::new(),
        })
    }

    /// Analyze meaning preservation in text
    pub fn analyze_meaning_preservation(
        &mut self,
        text: &str,
        reference_text: Option<&str>,
    ) -> MeaningResult<MeaningPreservationMetrics> {
        let cache_key = self.generate_cache_key(text, reference_text);
        if let Some(cached) = self.meaning_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let sentences = self.extract_sentences(text);
        let mut metrics = MeaningPreservationMetrics::default();

        // Core meaning preservation analysis
        metrics.overall_preservation =
            self.calculate_overall_preservation(&sentences, reference_text)?;
        metrics.conceptual_stability = self.analyze_conceptual_stability(&sentences)?;
        metrics.semantic_consistency = self.evaluate_semantic_consistency(&sentences)?;
        metrics.meaning_density = self.calculate_meaning_density(&sentences)?;

        // Advanced meaning analysis
        if self.config.use_advanced_analysis {
            metrics.conceptual_mappings = self.build_conceptual_mappings(&sentences)?;
            metrics.semantic_drift_patterns = self.detect_semantic_drift(&sentences)?;
            metrics.meaning_distortions = self.identify_meaning_distortions(&sentences)?;
            metrics.core_meaning_analysis = self.perform_core_meaning_analysis(&sentences)?;
            metrics.stability_metrics = self.calculate_stability_metrics(&sentences)?;
        }

        // Contextual meaning evaluation
        if self.config.analyze_contextual_meaning {
            metrics.contextual_preservation = self.analyze_contextual_preservation(&sentences)?;
            metrics.meaning_transfer_quality = self.evaluate_meaning_transfer(&sentences)?;
        }

        // Cache results for performance
        self.meaning_cache.insert(cache_key, metrics.clone());

        Ok(metrics)
    }

    /// Calculate overall meaning preservation score
    fn calculate_overall_preservation(
        &self,
        sentences: &[String],
        reference_text: Option<&str>,
    ) -> MeaningResult<f64> {
        if sentences.is_empty() {
            return Ok(0.0);
        }

        let mut preservation_scores = Vec::new();

        for (i, sentence) in sentences.iter().enumerate() {
            let sentence_score = match reference_text {
                Some(ref_text) => self.calculate_reference_preservation(sentence, ref_text)?,
                None => self.calculate_intrinsic_preservation(sentence, sentences)?,
            };

            preservation_scores.push(sentence_score);
        }

        // Weight preservation by sentence importance
        let weighted_score = if self.config.weight_by_importance {
            self.calculate_weighted_preservation(&preservation_scores, sentences)?
        } else {
            preservation_scores.iter().sum::<f64>() / preservation_scores.len() as f64
        };

        Ok(weighted_score.max(0.0).min(1.0))
    }

    /// Analyze conceptual stability throughout text
    fn analyze_conceptual_stability(&mut self, sentences: &[String]) -> MeaningResult<f64> {
        if sentences.len() < 2 {
            return Ok(1.0);
        }

        let mut stability_scores = Vec::new();
        let concepts = self.extract_core_concepts(sentences)?;

        for i in 1..sentences.len() {
            let prev_concepts = self.extract_sentence_concepts(&sentences[i - 1])?;
            let curr_concepts = self.extract_sentence_concepts(&sentences[i])?;

            let stability = self.calculate_conceptual_overlap(&prev_concepts, &curr_concepts)?;
            stability_scores.push(stability);
        }

        let average_stability =
            stability_scores.iter().sum::<f64>() / stability_scores.len() as f64;

        // Update stability tracker
        for concept in concepts {
            self.stability_tracker.insert(concept, average_stability);
        }

        Ok(average_stability)
    }

    /// Evaluate semantic consistency across text
    fn evaluate_semantic_consistency(&self, sentences: &[String]) -> MeaningResult<f64> {
        if sentences.len() < 2 {
            return Ok(1.0);
        }

        let mut consistency_scores = Vec::new();

        // Analyze pairwise semantic consistency
        for i in 0..sentences.len() {
            for j in (i + 1)..sentences.len() {
                let consistency =
                    self.calculate_semantic_consistency(&sentences[i], &sentences[j])?;
                consistency_scores.push(consistency);
            }
        }

        // Calculate overall consistency with variance consideration
        let mean_consistency =
            consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64;
        let variance = self.calculate_variance(&consistency_scores, mean_consistency);

        // Penalize high variance in consistency
        let consistency_penalty = if variance > self.config.consistency_variance_threshold {
            1.0 - (variance - self.config.consistency_variance_threshold)
        } else {
            1.0
        };

        Ok((mean_consistency * consistency_penalty).max(0.0).min(1.0))
    }

    /// Calculate meaning density in text
    fn calculate_meaning_density(&self, sentences: &[String]) -> MeaningResult<f64> {
        let total_words: usize = sentences.iter().map(|s| s.split_whitespace().count()).sum();

        if total_words == 0 {
            return Ok(0.0);
        }

        let mut meaningful_words = 0;
        let mut semantic_weight_sum = 0.0;

        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();

            for word in words {
                if self.is_meaningful_word(word) {
                    meaningful_words += 1;
                    semantic_weight_sum += self.calculate_semantic_weight(word)?;
                }
            }
        }

        let density_ratio = meaningful_words as f64 / total_words as f64;
        let weight_density = if meaningful_words > 0 {
            semantic_weight_sum / meaningful_words as f64
        } else {
            0.0
        };

        // Combine ratio and weight considerations
        Ok((density_ratio * self.config.density_ratio_weight
            + weight_density * self.config.weight_density_factor)
            .min(1.0))
    }

    /// Build conceptual mappings between sentences
    fn build_conceptual_mappings(
        &mut self,
        sentences: &[String],
    ) -> MeaningResult<Vec<ConceptualMapping>> {
        let mut mappings = Vec::new();

        for (i, sentence) in sentences.iter().enumerate() {
            let concepts = self.extract_sentence_concepts(sentence)?;

            for concept in concepts {
                let mapping = ConceptualMapping {
                    concept: concept.clone(),
                    source_sentence: i,
                    target_sentences: self.find_related_sentences(&concept, sentences, i)?,
                    mapping_strength: self.calculate_mapping_strength(&concept, sentences)?,
                    semantic_distance: self.calculate_concept_distance(&concept, sentences)?,
                    preservation_score: self.calculate_concept_preservation(&concept, sentences)?,
                };

                mappings.push(mapping);
                self.concept_mappings.insert(concept, mapping.clone());
            }
        }

        Ok(mappings)
    }

    /// Detect semantic drift patterns in text
    fn detect_semantic_drift(&mut self, sentences: &[String]) -> MeaningResult<Vec<SemanticDrift>> {
        let mut drift_patterns = Vec::new();

        if sentences.len() < 3 {
            return Ok(drift_patterns);
        }

        let window_size = self.config.drift_window_size.min(sentences.len());

        for window_start in 0..=(sentences.len() - window_size) {
            let window = &sentences[window_start..(window_start + window_size)];

            let drift = self.analyze_window_drift(window, window_start)?;

            if drift.drift_magnitude > self.config.drift_threshold {
                drift_patterns.push(drift);
            }
        }

        // Store for future analysis
        self.drift_patterns = drift_patterns.clone();

        Ok(drift_patterns)
    }

    /// Identify meaning distortions in text
    fn identify_meaning_distortions(
        &self,
        sentences: &[String],
    ) -> MeaningResult<Vec<MeaningDistortion>> {
        let mut distortions = Vec::new();

        for (i, sentence) in sentences.iter().enumerate() {
            let sentence_concepts = self.extract_sentence_concepts(sentence)?;

            for concept in sentence_concepts {
                // Check for semantic inconsistencies
                if let Some(expected_meaning) = self.get_expected_meaning(&concept) {
                    let actual_meaning = self.extract_concept_meaning(sentence, &concept)?;
                    let distortion_level =
                        self.calculate_meaning_distortion(&expected_meaning, &actual_meaning)?;

                    if distortion_level > self.config.distortion_threshold {
                        let distortion = MeaningDistortion {
                            concept: concept.clone(),
                            sentence_index: i,
                            distortion_type: self
                                .classify_distortion_type(&expected_meaning, &actual_meaning)?,
                            severity: distortion_level,
                            description: self.generate_distortion_description(
                                &concept,
                                &expected_meaning,
                                &actual_meaning,
                            )?,
                        };

                        distortions.push(distortion);
                    }
                }
            }
        }

        Ok(distortions)
    }

    /// Perform core meaning analysis
    fn perform_core_meaning_analysis(
        &self,
        sentences: &[String],
    ) -> MeaningResult<CoreMeaningAnalysis> {
        let core_concepts = self.identify_core_concepts(sentences)?;
        let meaning_relationships = self.analyze_meaning_relationships(sentences)?;
        let conceptual_hierarchy = self.build_conceptual_hierarchy(sentences)?;
        let meaning_evolution = self.trace_meaning_evolution(sentences)?;

        Ok(CoreMeaningAnalysis {
            core_concepts,
            primary_meaning_threads: self.identify_primary_threads(sentences)?,
            meaning_relationships,
            conceptual_hierarchy,
            meaning_evolution,
            semantic_anchors: self.identify_semantic_anchors(sentences)?,
            meaning_coherence_score: self.calculate_meaning_coherence(sentences)?,
        })
    }

    /// Calculate stability metrics for meanings
    fn calculate_stability_metrics(
        &self,
        sentences: &[String],
    ) -> MeaningResult<MeaningStabilityMetrics> {
        let concept_stability = self.analyze_concept_stability(sentences)?;
        let relationship_stability = self.analyze_relationship_stability(sentences)?;
        let temporal_consistency = self.calculate_temporal_consistency(sentences)?;
        let semantic_momentum = self.calculate_semantic_momentum(sentences)?;

        Ok(MeaningStabilityMetrics {
            concept_stability,
            relationship_stability,
            temporal_consistency,
            semantic_momentum,
            stability_variance: self.calculate_stability_variance(sentences)?,
            drift_resistance: self.calculate_drift_resistance(sentences)?,
        })
    }

    /// Analyze contextual meaning preservation
    fn analyze_contextual_preservation(&self, sentences: &[String]) -> MeaningResult<f64> {
        if sentences.is_empty() {
            return Ok(0.0);
        }

        let mut context_scores = Vec::new();

        for (i, sentence) in sentences.iter().enumerate() {
            let context_before = if i > 0 { Some(&sentences[0..i]) } else { None };
            let context_after = if i < sentences.len() - 1 {
                Some(&sentences[i + 1..])
            } else {
                None
            };

            let preservation_score = self.calculate_contextual_preservation_score(
                sentence,
                context_before,
                context_after,
            )?;

            context_scores.push(preservation_score);
        }

        Ok(context_scores.iter().sum::<f64>() / context_scores.len() as f64)
    }

    /// Evaluate meaning transfer quality
    fn evaluate_meaning_transfer(&self, sentences: &[String]) -> MeaningResult<f64> {
        if sentences.len() < 2 {
            return Ok(1.0);
        }

        let mut transfer_scores = Vec::new();

        for i in 1..sentences.len() {
            let transfer_quality =
                self.calculate_meaning_transfer_quality(&sentences[i - 1], &sentences[i])?;
            transfer_scores.push(transfer_quality);
        }

        Ok(transfer_scores.iter().sum::<f64>() / transfer_scores.len() as f64)
    }

    // Helper methods for semantic meaning analysis

    fn validate_config(config: &MeaningAnalysisConfig) -> MeaningResult<()> {
        if config.meaning_preservation_threshold < 0.0
            || config.meaning_preservation_threshold > 1.0
        {
            return Err(MeaningAnalysisError::ConfigError(
                "meaning_preservation_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if config.drift_window_size < 2 {
            return Err(MeaningAnalysisError::ConfigError(
                "drift_window_size must be at least 2".to_string(),
            ));
        }

        Ok(())
    }

    fn extract_sentences(&self, text: &str) -> Vec<String> {
        text.split(&self.config.sentence_delimiters)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn generate_cache_key(&self, text: &str, reference_text: Option<&str>) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        if let Some(ref_text) = reference_text {
            ref_text.hash(&mut hasher);
        }
        self.config.hash(&mut hasher);
        hasher.finish()
    }

    fn extract_core_concepts(&self, sentences: &[String]) -> MeaningResult<Vec<String>> {
        let mut concept_counts: HashMap<String, usize> = HashMap::new();

        for sentence in sentences {
            let concepts = self.extract_sentence_concepts(sentence)?;
            for concept in concepts {
                *concept_counts.entry(concept).or_insert(0) += 1;
            }
        }

        let mut concepts: Vec<_> = concept_counts.into_iter().collect();
        concepts.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(concepts
            .into_iter()
            .take(self.config.max_core_concepts)
            .map(|(concept, _)| concept)
            .collect())
    }

    fn extract_sentence_concepts(&self, sentence: &str) -> MeaningResult<Vec<String>> {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut concepts = Vec::new();

        for word in words {
            if self.is_meaningful_word(word) && !self.is_stop_word(word) {
                concepts.push(word.to_lowercase());
            }
        }

        // Extract multi-word concepts if enabled
        if self.config.extract_multiword_concepts {
            concepts.extend(self.extract_multiword_concepts(sentence)?);
        }

        Ok(concepts)
    }

    fn is_meaningful_word(&self, word: &str) -> bool {
        word.len() >= self.config.min_word_length && word.chars().any(|c| c.is_alphabetic())
    }

    fn is_stop_word(&self, word: &str) -> bool {
        self.config.stop_words.contains(&word.to_lowercase())
    }

    fn calculate_semantic_weight(&self, word: &str) -> MeaningResult<f64> {
        // Simple frequency-based weighting - can be enhanced with semantic databases
        let base_weight = 1.0;
        let length_factor = (word.len() as f64).min(10.0) / 10.0;
        let rarity_factor = 1.0; // Would use corpus frequency in real implementation

        Ok(base_weight * length_factor * rarity_factor)
    }

    fn calculate_variance(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let variance_sum: f64 = values.iter().map(|value| (value - mean).powi(2)).sum();

        variance_sum / (values.len() - 1) as f64
    }
}

impl Default for SemanticMeaningAnalyzer {
    fn default() -> Self {
        Self::new(MeaningAnalysisConfig::default()).unwrap()
    }
}

// Additional implementation methods would continue here...
// This represents the core structure for comprehensive meaning analysis
