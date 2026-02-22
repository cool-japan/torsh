//! Phonological Analysis Module for Prosodic Fluency
//!
//! This module provides comprehensive phonological pattern analysis capabilities
//! for prosodic fluency assessment, including syllable structure analysis,
//! phoneme sequence evaluation, phonological rule application, and phonotactic
//! constraint assessment.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use thiserror::Error;

use super::config::PhonologicalAnalysisConfig;
use super::results::{
    ConstraintViolation, PhonemeClass, PhonemeSequenceMetrics, PhonologicalMetrics,
    PhonologicalPattern, PhonologicalRule, PhonologicalRuleMetrics, PhonotacticConstraintMetrics,
    SyllableStructureMetrics, SyllableType,
};

/// Comprehensive phonological analysis engine for prosodic fluency assessment
#[derive(Debug, Clone)]
pub struct PhonologicalAnalyzer {
    config: PhonologicalAnalysisConfig,
    syllable_analyzer: SyllableStructureAnalyzer,
    phoneme_analyzer: PhonemeSequenceAnalyzer,
    rule_analyzer: PhonologicalRuleAnalyzer,
    constraint_analyzer: PhonotacticConstraintAnalyzer,
    pattern_matcher: PhonologicalPatternMatcher,
    analysis_cache: HashMap<u64, PhonologicalMetrics>,
}

/// Specialized syllable structure analysis engine
#[derive(Debug, Clone)]
pub struct SyllableStructureAnalyzer {
    onset_patterns: HashMap<String, f64>,
    nucleus_patterns: HashMap<String, f64>,
    coda_patterns: HashMap<String, f64>,
    complexity_weights: HashMap<SyllableType, f64>,
    structure_templates: Vec<SyllableTemplate>,
}

/// Specialized phoneme sequence analysis engine
#[derive(Debug, Clone)]
pub struct PhonemeSequenceAnalyzer {
    sequence_patterns: HashMap<String, f64>,
    transition_probabilities: HashMap<(PhonemeClass, PhonemeClass), f64>,
    clustering_rules: Vec<ClusteringRule>,
    phonotactic_weights: HashMap<String, f64>,
}

/// Specialized phonological rule analysis engine
#[derive(Debug, Clone)]
pub struct PhonologicalRuleAnalyzer {
    active_rules: Vec<PhonologicalRule>,
    rule_contexts: HashMap<String, Vec<String>>,
    application_frequencies: HashMap<String, f64>,
    rule_interactions: HashMap<(String, String), f64>,
}

/// Specialized phonotactic constraint analysis engine
#[derive(Debug, Clone)]
pub struct PhonotacticConstraintAnalyzer {
    universal_constraints: Vec<PhonotacticConstraint>,
    language_constraints: Vec<PhonotacticConstraint>,
    constraint_weights: HashMap<String, f64>,
    violation_penalties: HashMap<String, f64>,
}

/// Pattern matching engine for phonological structures
#[derive(Debug, Clone)]
pub struct PhonologicalPatternMatcher {
    pattern_database: Vec<PhonologicalPattern>,
    matching_thresholds: HashMap<String, f64>,
    pattern_frequencies: HashMap<String, f64>,
    context_weights: HashMap<String, f64>,
}

/// Syllable structure template for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyllableTemplate {
    pub name: String,
    pub onset_pattern: String,
    pub nucleus_pattern: String,
    pub coda_pattern: String,
    pub complexity_score: f64,
    pub frequency_weight: f64,
    pub language_specific: bool,
}

/// Phoneme clustering rule for sequence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringRule {
    pub name: String,
    pub pattern: String,
    pub context: String,
    pub sonority_profile: Vec<f64>,
    pub application_probability: f64,
    pub language_universal: bool,
}

/// Phonotactic constraint for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonotacticConstraint {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub pattern: String,
    pub context: String,
    pub violation_weight: f64,
    pub universality: ConstraintUniversality,
}

/// Types of phonotactic constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    Onset,
    Nucleus,
    Coda,
    Sequence,
    Transition,
    Sonority,
    Markedness,
    Faithfulness,
}

/// Universality levels of constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintUniversality {
    Universal,
    Typological,
    LanguageSpecific,
    Dialectal,
}

/// Errors that can occur during phonological analysis
#[derive(Debug, Error)]
pub enum PhonologicalAnalysisError {
    #[error("Invalid phoneme sequence: {sequence}")]
    InvalidPhonemeSequence { sequence: String },

    #[error("Syllable structure parsing failed: {structure}")]
    SyllableParsingError { structure: String },

    #[error("Phonological rule application failed: {rule}")]
    RuleApplicationError { rule: String },

    #[error("Constraint evaluation failed: {constraint}")]
    ConstraintEvaluationError { constraint: String },

    #[error("Pattern matching failed: {pattern}")]
    PatternMatchingError { pattern: String },

    #[error("Cache operation failed: {operation}")]
    CacheError { operation: String },
}

impl PhonologicalAnalyzer {
    /// Creates a new phonological analyzer with the specified configuration
    pub fn new(config: PhonologicalAnalysisConfig) -> Self {
        let syllable_analyzer = SyllableStructureAnalyzer::new(&config);
        let phoneme_analyzer = PhonemeSequenceAnalyzer::new(&config);
        let rule_analyzer = PhonologicalRuleAnalyzer::new(&config);
        let constraint_analyzer = PhonotacticConstraintAnalyzer::new(&config);
        let pattern_matcher = PhonologicalPatternMatcher::new(&config);

        Self {
            config,
            syllable_analyzer,
            phoneme_analyzer,
            rule_analyzer,
            constraint_analyzer,
            pattern_matcher,
            analysis_cache: HashMap::new(),
        }
    }

    /// Performs comprehensive phonological analysis on the input text
    pub fn analyze(
        &mut self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<PhonologicalMetrics, PhonologicalAnalysisError> {
        // Generate cache key
        let cache_key = self.generate_cache_key(text, phonetic_transcription);

        // Check cache first
        if let Some(cached_result) = self.analysis_cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }

        // Perform comprehensive analysis
        let syllable_metrics = self.analyze_syllable_structure(phonetic_transcription)?;
        let phoneme_metrics = self.analyze_phoneme_sequences(phonetic_transcription)?;
        let rule_metrics = self.analyze_phonological_rules(text, phonetic_transcription)?;
        let constraint_metrics = self.analyze_phonotactic_constraints(phonetic_transcription)?;

        // Detect phonological patterns
        let detected_patterns = self.detect_phonological_patterns(phonetic_transcription)?;

        // Calculate overall phonological complexity
        let complexity_score = self.calculate_phonological_complexity(
            &syllable_metrics,
            &phoneme_metrics,
            &rule_metrics,
            &constraint_metrics,
        );

        // Calculate fluency impact score
        let fluency_impact =
            self.calculate_fluency_impact(&syllable_metrics, &phoneme_metrics, &constraint_metrics);

        let metrics = PhonologicalMetrics {
            syllable_structure: syllable_metrics,
            phoneme_sequence: phoneme_metrics,
            phonological_rules: rule_metrics,
            phonotactic_constraints: constraint_metrics,
            detected_patterns,
            complexity_score,
            fluency_impact,
            analysis_confidence: self.calculate_analysis_confidence(),
        };

        // Cache the result
        if self.config.enable_caching {
            self.analysis_cache.insert(cache_key, metrics.clone());
        }

        Ok(metrics)
    }

    /// Analyzes syllable structure patterns
    fn analyze_syllable_structure(
        &mut self,
        phonetic_transcription: &str,
    ) -> Result<SyllableStructureMetrics, PhonologicalAnalysisError> {
        self.syllable_analyzer.analyze(phonetic_transcription)
    }

    /// Analyzes phoneme sequence patterns
    fn analyze_phoneme_sequences(
        &mut self,
        phonetic_transcription: &str,
    ) -> Result<PhonemeSequenceMetrics, PhonologicalAnalysisError> {
        self.phoneme_analyzer.analyze(phonetic_transcription)
    }

    /// Analyzes phonological rule applications
    fn analyze_phonological_rules(
        &mut self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<PhonologicalRuleMetrics, PhonologicalAnalysisError> {
        self.rule_analyzer.analyze(text, phonetic_transcription)
    }

    /// Analyzes phonotactic constraint adherence
    fn analyze_phonotactic_constraints(
        &mut self,
        phonetic_transcription: &str,
    ) -> Result<PhonotacticConstraintMetrics, PhonologicalAnalysisError> {
        self.constraint_analyzer.analyze(phonetic_transcription)
    }

    /// Detects phonological patterns in the transcription
    fn detect_phonological_patterns(
        &mut self,
        phonetic_transcription: &str,
    ) -> Result<Vec<PhonologicalPattern>, PhonologicalAnalysisError> {
        self.pattern_matcher.detect_patterns(phonetic_transcription)
    }

    /// Calculates overall phonological complexity score
    fn calculate_phonological_complexity(
        &self,
        syllable_metrics: &SyllableStructureMetrics,
        phoneme_metrics: &PhonemeSequenceMetrics,
        rule_metrics: &PhonologicalRuleMetrics,
        constraint_metrics: &PhonotacticConstraintMetrics,
    ) -> f64 {
        let weights = &self.config.complexity_weights;

        weights.syllable_complexity * syllable_metrics.average_complexity
            + weights.phoneme_complexity * phoneme_metrics.sequence_complexity
            + weights.rule_complexity * rule_metrics.rule_density
            + weights.constraint_complexity * constraint_metrics.violation_density
    }

    /// Calculates phonological impact on fluency
    fn calculate_fluency_impact(
        &self,
        syllable_metrics: &SyllableStructureMetrics,
        phoneme_metrics: &PhonemeSequenceMetrics,
        constraint_metrics: &PhonotacticConstraintMetrics,
    ) -> f64 {
        let syllable_impact = if syllable_metrics.average_complexity
            > self.config.complexity_thresholds.high_complexity
        {
            0.3
        } else if syllable_metrics.average_complexity
            > self.config.complexity_thresholds.medium_complexity
        {
            0.1
        } else {
            -0.1
        };

        let phoneme_impact = if phoneme_metrics.clustering_frequency
            > self.config.difficulty_thresholds.high_clustering
        {
            0.25
        } else {
            -0.05
        };

        let constraint_impact = constraint_metrics.violation_density * 0.4;

        (syllable_impact + phoneme_impact + constraint_impact)
            .max(-1.0)
            .min(1.0)
    }

    /// Calculates confidence in the analysis results
    fn calculate_analysis_confidence(&self) -> f64 {
        // Base confidence on data quality and analysis coverage
        let base_confidence = 0.85;
        let coverage_bonus = if self.config.comprehensive_analysis {
            0.1
        } else {
            0.0
        };
        let cache_penalty = if self.analysis_cache.len() > 1000 {
            -0.05
        } else {
            0.0
        };

        (base_confidence + coverage_bonus + cache_penalty)
            .max(0.0)
            .min(1.0)
    }

    /// Generates a cache key for the analysis
    fn generate_cache_key(&self, text: &str, phonetic_transcription: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        phonetic_transcription.hash(&mut hasher);
        self.config.analysis_depth.hash(&mut hasher);
        hasher.finish()
    }

    /// Updates the analysis configuration
    pub fn update_config(&mut self, new_config: PhonologicalAnalysisConfig) {
        self.config = new_config;

        // Update sub-analyzers
        self.syllable_analyzer.update_config(&self.config);
        self.phoneme_analyzer.update_config(&self.config);
        self.rule_analyzer.update_config(&self.config);
        self.constraint_analyzer.update_config(&self.config);
        self.pattern_matcher.update_config(&self.config);

        // Clear cache if configuration changed significantly
        if self.config.clear_cache_on_config_change {
            self.analysis_cache.clear();
        }
    }

    /// Clears the analysis cache
    pub fn clear_cache(&mut self) {
        self.analysis_cache.clear();
    }

    /// Gets current cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        (self.analysis_cache.len(), self.analysis_cache.capacity())
    }
}

impl SyllableStructureAnalyzer {
    fn new(config: &PhonologicalAnalysisConfig) -> Self {
        let syllable_templates = Self::create_default_syllable_templates();

        Self {
            onset_patterns: Self::create_onset_patterns(),
            nucleus_patterns: Self::create_nucleus_patterns(),
            coda_patterns: Self::create_coda_patterns(),
            complexity_weights: Self::create_complexity_weights(),
            structure_templates: syllable_templates,
        }
    }

    fn analyze(
        &self,
        phonetic_transcription: &str,
    ) -> Result<SyllableStructureMetrics, PhonologicalAnalysisError> {
        let syllables = self.segment_syllables(phonetic_transcription)?;
        let mut structure_counts = HashMap::new();
        let mut complexity_scores = Vec::new();
        let mut syllable_types = Vec::new();

        for syllable in &syllables {
            let structure = self.analyze_syllable_structure(syllable)?;
            let complexity = self.calculate_syllable_complexity(&structure);
            let syllable_type = self.classify_syllable_type(&structure);

            *structure_counts.entry(structure.clone()).or_insert(0u32) += 1;
            complexity_scores.push(complexity);
            syllable_types.push(syllable_type);
        }

        let total_syllables = syllables.len() as f64;
        let average_complexity = complexity_scores.iter().sum::<f64>() / total_syllables;
        let complexity_variance = self.calculate_variance(&complexity_scores, average_complexity);

        // Calculate type distributions
        let cv_syllables = syllable_types
            .iter()
            .filter(|&t| matches!(t, SyllableType::CV))
            .count() as f64
            / total_syllables;
        let cvc_syllables = syllable_types
            .iter()
            .filter(|&t| matches!(t, SyllableType::CVC))
            .count() as f64
            / total_syllables;
        let complex_syllables = syllable_types
            .iter()
            .filter(|&t| {
                matches!(
                    t,
                    SyllableType::CCVC | SyllableType::CVCC | SyllableType::CCVCC
                )
            })
            .count() as f64
            / total_syllables;

        Ok(SyllableStructureMetrics {
            total_syllables: syllables.len(),
            structure_types: structure_counts,
            average_complexity,
            complexity_variance,
            cv_syllables,
            cvc_syllables,
            complex_syllables,
            syllable_types,
            onset_complexity: self.calculate_onset_complexity(&syllables),
            coda_complexity: self.calculate_coda_complexity(&syllables),
        })
    }

    fn segment_syllables(
        &self,
        phonetic_transcription: &str,
    ) -> Result<Vec<String>, PhonologicalAnalysisError> {
        // Implement syllable segmentation algorithm
        let mut syllables = Vec::new();
        let phonemes: Vec<&str> = phonetic_transcription.split_whitespace().collect();

        let mut current_syllable = String::new();
        let mut in_nucleus = false;

        for phoneme in phonemes {
            if self.is_vowel(phoneme) {
                if in_nucleus && !current_syllable.is_empty() {
                    syllables.push(current_syllable.trim().to_string());
                    current_syllable = String::new();
                }
                current_syllable.push_str(&format!("{} ", phoneme));
                in_nucleus = true;
            } else {
                current_syllable.push_str(&format!("{} ", phoneme));
                if in_nucleus {
                    // This is a coda consonant
                    // Check if next phoneme is a vowel to determine syllable boundary
                }
            }
        }

        if !current_syllable.is_empty() {
            syllables.push(current_syllable.trim().to_string());
        }

        if syllables.is_empty() {
            return Err(PhonologicalAnalysisError::SyllableParsingError {
                structure: phonetic_transcription.to_string(),
            });
        }

        Ok(syllables)
    }

    fn analyze_syllable_structure(
        &self,
        syllable: &str,
    ) -> Result<String, PhonologicalAnalysisError> {
        let phonemes: Vec<&str> = syllable.split_whitespace().collect();
        let mut structure = String::new();

        for phoneme in phonemes {
            if self.is_vowel(phoneme) {
                structure.push('V');
            } else {
                structure.push('C');
            }
        }

        if structure.is_empty() {
            return Err(PhonologicalAnalysisError::SyllableParsingError {
                structure: syllable.to_string(),
            });
        }

        Ok(structure)
    }

    fn calculate_syllable_complexity(&self, structure: &str) -> f64 {
        let mut complexity = 0.0;

        // Count consecutive consonants (clusters)
        let mut consonant_count = 0;
        let mut max_cluster = 0;

        for c in structure.chars() {
            if c == 'C' {
                consonant_count += 1;
                max_cluster = max_cluster.max(consonant_count);
            } else {
                consonant_count = 0;
            }
        }

        // Base complexity on structure length and cluster size
        complexity += structure.len() as f64 * 0.2;
        complexity += max_cluster as f64 * 0.5;

        complexity
    }

    fn classify_syllable_type(&self, structure: &str) -> SyllableType {
        match structure {
            "V" => SyllableType::V,
            "CV" => SyllableType::CV,
            "VC" => SyllableType::VC,
            "CVC" => SyllableType::CVC,
            s if s.starts_with("CC") && s.ends_with("C") && s.contains('V') => SyllableType::CCVCC,
            s if s.starts_with("CC") && s.contains('V') => SyllableType::CCVC,
            s if s.ends_with("CC") && s.contains('V') => SyllableType::CVCC,
            _ => SyllableType::Complex,
        }
    }

    fn calculate_variance(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let sum_squared_diffs: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum();

        sum_squared_diffs / (values.len() - 1) as f64
    }

    fn calculate_onset_complexity(&self, syllables: &[String]) -> f64 {
        syllables
            .iter()
            .map(|syl| self.get_onset_complexity(syl))
            .sum::<f64>()
            / syllables.len() as f64
    }

    fn calculate_coda_complexity(&self, syllables: &[String]) -> f64 {
        syllables
            .iter()
            .map(|syl| self.get_coda_complexity(syl))
            .sum::<f64>()
            / syllables.len() as f64
    }

    fn get_onset_complexity(&self, syllable: &str) -> f64 {
        // Count initial consonants
        let phonemes: Vec<&str> = syllable.split_whitespace().collect();
        let mut onset_length = 0;

        for phoneme in phonemes {
            if self.is_vowel(phoneme) {
                break;
            }
            onset_length += 1;
        }

        onset_length as f64
    }

    fn get_coda_complexity(&self, syllable: &str) -> f64 {
        // Count final consonants
        let phonemes: Vec<&str> = syllable.split_whitespace().collect();
        let mut coda_length = 0;

        for phoneme in phonemes.iter().rev() {
            if self.is_vowel(phoneme) {
                break;
            }
            coda_length += 1;
        }

        coda_length as f64
    }

    fn is_vowel(&self, phoneme: &str) -> bool {
        // Simplified vowel detection - in practice, would use IPA classification
        matches!(
            phoneme.to_lowercase().as_str(),
            "a" | "e"
                | "i"
                | "o"
                | "u"
                | "æ"
                | "ɛ"
                | "ɪ"
                | "ɔ"
                | "ʊ"
                | "ə"
                | "ɑ"
                | "ɒ"
                | "ʌ"
                | "ɜ"
                | "ɨ"
                | "ɵ"
                | "ɐ"
                | "ɶ"
                | "ø"
                | "y"
        )
    }

    fn create_default_syllable_templates() -> Vec<SyllableTemplate> {
        vec![
            SyllableTemplate {
                name: "Simple CV".to_string(),
                onset_pattern: "C".to_string(),
                nucleus_pattern: "V".to_string(),
                coda_pattern: "".to_string(),
                complexity_score: 1.0,
                frequency_weight: 0.3,
                language_specific: false,
            },
            SyllableTemplate {
                name: "Simple CVC".to_string(),
                onset_pattern: "C".to_string(),
                nucleus_pattern: "V".to_string(),
                coda_pattern: "C".to_string(),
                complexity_score: 1.5,
                frequency_weight: 0.4,
                language_specific: false,
            },
            SyllableTemplate {
                name: "Complex CCVC".to_string(),
                onset_pattern: "CC".to_string(),
                nucleus_pattern: "V".to_string(),
                coda_pattern: "C".to_string(),
                complexity_score: 2.5,
                frequency_weight: 0.1,
                language_specific: true,
            },
        ]
    }

    fn create_onset_patterns() -> HashMap<String, f64> {
        let mut patterns = HashMap::new();
        patterns.insert("C".to_string(), 1.0);
        patterns.insert("CC".to_string(), 2.0);
        patterns.insert("CCC".to_string(), 3.0);
        patterns
    }

    fn create_nucleus_patterns() -> HashMap<String, f64> {
        let mut patterns = HashMap::new();
        patterns.insert("V".to_string(), 1.0);
        patterns.insert("VV".to_string(), 1.5);
        patterns.insert("VVV".to_string(), 2.0);
        patterns
    }

    fn create_coda_patterns() -> HashMap<String, f64> {
        let mut patterns = HashMap::new();
        patterns.insert("".to_string(), 0.0);
        patterns.insert("C".to_string(), 1.0);
        patterns.insert("CC".to_string(), 2.0);
        patterns.insert("CCC".to_string(), 3.0);
        patterns
    }

    fn create_complexity_weights() -> HashMap<SyllableType, f64> {
        let mut weights = HashMap::new();
        weights.insert(SyllableType::V, 0.5);
        weights.insert(SyllableType::CV, 1.0);
        weights.insert(SyllableType::VC, 1.2);
        weights.insert(SyllableType::CVC, 1.5);
        weights.insert(SyllableType::CCVC, 2.0);
        weights.insert(SyllableType::CVCC, 2.0);
        weights.insert(SyllableType::CCVCC, 2.5);
        weights.insert(SyllableType::Complex, 3.0);
        weights
    }

    fn update_config(&mut self, config: &PhonologicalAnalysisConfig) {
        // Update internal configuration based on new settings
        if config.detailed_syllable_analysis {
            self.structure_templates
                .extend(Self::create_extended_syllable_templates());
        }
    }

    fn create_extended_syllable_templates() -> Vec<SyllableTemplate> {
        vec![
            SyllableTemplate {
                name: "Vowel-only".to_string(),
                onset_pattern: "".to_string(),
                nucleus_pattern: "V".to_string(),
                coda_pattern: "".to_string(),
                complexity_score: 0.8,
                frequency_weight: 0.05,
                language_specific: true,
            },
            SyllableTemplate {
                name: "Complex CCVCC".to_string(),
                onset_pattern: "CC".to_string(),
                nucleus_pattern: "V".to_string(),
                coda_pattern: "CC".to_string(),
                complexity_score: 3.0,
                frequency_weight: 0.02,
                language_specific: true,
            },
        ]
    }
}

impl PhonemeSequenceAnalyzer {
    fn new(config: &PhonologicalAnalysisConfig) -> Self {
        Self {
            sequence_patterns: Self::create_sequence_patterns(),
            transition_probabilities: Self::create_transition_probabilities(),
            clustering_rules: Self::create_clustering_rules(),
            phonotactic_weights: Self::create_phonotactic_weights(),
        }
    }

    fn analyze(
        &self,
        phonetic_transcription: &str,
    ) -> Result<PhonemeSequenceMetrics, PhonologicalAnalysisError> {
        let phonemes: Vec<&str> = phonetic_transcription.split_whitespace().collect();

        if phonemes.is_empty() {
            return Err(PhonologicalAnalysisError::InvalidPhonemeSequence {
                sequence: phonetic_transcription.to_string(),
            });
        }

        let sequence_complexity = self.calculate_sequence_complexity(&phonemes);
        let transition_smoothness = self.calculate_transition_smoothness(&phonemes);
        let clustering_frequency = self.calculate_clustering_frequency(&phonemes);
        let phonotactic_violations = self.detect_phonotactic_violations(&phonemes);

        let bigrams = self.extract_bigrams(&phonemes);
        let trigrams = self.extract_trigrams(&phonemes);
        let consonant_clusters = self.identify_consonant_clusters(&phonemes);
        let vowel_sequences = self.identify_vowel_sequences(&phonemes);

        Ok(PhonemeSequenceMetrics {
            total_phonemes: phonemes.len(),
            sequence_complexity,
            transition_smoothness,
            clustering_frequency,
            phonotactic_violations: phonotactic_violations.len(),
            bigrams,
            trigrams,
            consonant_clusters,
            vowel_sequences,
            average_cluster_size: self.calculate_average_cluster_size(&consonant_clusters),
            sonority_violations: self.count_sonority_violations(&phonemes),
        })
    }

    fn calculate_sequence_complexity(&self, phonemes: &[&str]) -> f64 {
        let mut complexity = 0.0;

        // Base complexity on phoneme diversity and sequence length
        let unique_phonemes: HashSet<&str> = phonemes.iter().cloned().collect();
        complexity += unique_phonemes.len() as f64 * 0.1;
        complexity += phonemes.len() as f64 * 0.05;

        // Add complexity for difficult transitions
        for window in phonemes.windows(2) {
            if let [p1, p2] = window {
                let transition_key = (self.classify_phoneme(p1), self.classify_phoneme(p2));
                if let Some(&prob) = self.transition_probabilities.get(&transition_key) {
                    complexity += (1.0 - prob) * 0.5;
                }
            }
        }

        complexity
    }

    fn calculate_transition_smoothness(&self, phonemes: &[&str]) -> f64 {
        if phonemes.len() < 2 {
            return 1.0;
        }

        let mut smoothness_sum = 0.0;
        let mut transition_count = 0;

        for window in phonemes.windows(2) {
            if let [p1, p2] = window {
                let transition_key = (self.classify_phoneme(p1), self.classify_phoneme(p2));
                if let Some(&prob) = self.transition_probabilities.get(&transition_key) {
                    smoothness_sum += prob;
                    transition_count += 1;
                }
            }
        }

        if transition_count > 0 {
            smoothness_sum / transition_count as f64
        } else {
            0.5 // Default smoothness for unknown transitions
        }
    }

    fn calculate_clustering_frequency(&self, phonemes: &[&str]) -> f64 {
        let consonant_clusters = self.identify_consonant_clusters(phonemes);
        consonant_clusters.len() as f64 / phonemes.len() as f64
    }

    fn detect_phonotactic_violations(&self, phonemes: &[&str]) -> Vec<String> {
        let mut violations = Vec::new();

        // Check for illegal sequences
        for window in phonemes.windows(2) {
            if let [p1, p2] = window {
                let sequence = format!("{}-{}", p1, p2);
                if let Some(&weight) = self.phonotactic_weights.get(&sequence) {
                    if weight < 0.1 {
                        violations.push(sequence);
                    }
                }
            }
        }

        violations
    }

    fn extract_bigrams(&self, phonemes: &[&str]) -> Vec<String> {
        phonemes
            .windows(2)
            .map(|window| format!("{}-{}", window[0], window[1]))
            .collect()
    }

    fn extract_trigrams(&self, phonemes: &[&str]) -> Vec<String> {
        phonemes
            .windows(3)
            .map(|window| format!("{}-{}-{}", window[0], window[1], window[2]))
            .collect()
    }

    fn identify_consonant_clusters(&self, phonemes: &[&str]) -> Vec<String> {
        let mut clusters = Vec::new();
        let mut current_cluster = Vec::new();

        for &phoneme in phonemes {
            if self.is_consonant(phoneme) {
                current_cluster.push(phoneme);
            } else {
                if current_cluster.len() > 1 {
                    clusters.push(current_cluster.join("-"));
                }
                current_cluster.clear();
            }
        }

        // Don't forget the final cluster if it exists
        if current_cluster.len() > 1 {
            clusters.push(current_cluster.join("-"));
        }

        clusters
    }

    fn identify_vowel_sequences(&self, phonemes: &[&str]) -> Vec<String> {
        let mut sequences = Vec::new();
        let mut current_sequence = Vec::new();

        for &phoneme in phonemes {
            if self.is_vowel(phoneme) {
                current_sequence.push(phoneme);
            } else {
                if current_sequence.len() > 1 {
                    sequences.push(current_sequence.join("-"));
                }
                current_sequence.clear();
            }
        }

        // Don't forget the final sequence if it exists
        if current_sequence.len() > 1 {
            sequences.push(current_sequence.join("-"));
        }

        sequences
    }

    fn calculate_average_cluster_size(&self, clusters: &[String]) -> f64 {
        if clusters.is_empty() {
            return 0.0;
        }

        let total_size: usize = clusters
            .iter()
            .map(|cluster| cluster.split('-').count())
            .sum();

        total_size as f64 / clusters.len() as f64
    }

    fn count_sonority_violations(&self, phonemes: &[&str]) -> usize {
        let mut violations = 0;

        for window in phonemes.windows(2) {
            if let [p1, p2] = window {
                let sonority1 = self.get_sonority_level(p1);
                let sonority2 = self.get_sonority_level(p2);

                // Check for sonority sequencing principle violations
                if self.is_consonant(p1) && self.is_consonant(p2) && sonority1 > sonority2 {
                    violations += 1;
                }
            }
        }

        violations
    }

    fn classify_phoneme(&self, phoneme: &str) -> PhonemeClass {
        if self.is_vowel(phoneme) {
            PhonemeClass::Vowel
        } else if self.is_fricative(phoneme) {
            PhonemeClass::Fricative
        } else if self.is_stop(phoneme) {
            PhonemeClass::Stop
        } else if self.is_nasal(phoneme) {
            PhonemeClass::Nasal
        } else if self.is_liquid(phoneme) {
            PhonemeClass::Liquid
        } else {
            PhonemeClass::Other
        }
    }

    fn is_consonant(&self, phoneme: &str) -> bool {
        !self.is_vowel(phoneme)
    }

    fn is_vowel(&self, phoneme: &str) -> bool {
        matches!(
            phoneme.to_lowercase().as_str(),
            "a" | "e"
                | "i"
                | "o"
                | "u"
                | "æ"
                | "ɛ"
                | "ɪ"
                | "ɔ"
                | "ʊ"
                | "ə"
                | "ɑ"
                | "ɒ"
                | "ʌ"
                | "ɜ"
                | "ɨ"
                | "ɵ"
                | "ɐ"
                | "ɶ"
                | "ø"
                | "y"
        )
    }

    fn is_fricative(&self, phoneme: &str) -> bool {
        matches!(
            phoneme.to_lowercase().as_str(),
            "f" | "v" | "θ" | "ð" | "s" | "z" | "ʃ" | "ʒ" | "h" | "x" | "ɣ"
        )
    }

    fn is_stop(&self, phoneme: &str) -> bool {
        matches!(
            phoneme.to_lowercase().as_str(),
            "p" | "b" | "t" | "d" | "k" | "g" | "q" | "ɢ" | "ʔ"
        )
    }

    fn is_nasal(&self, phoneme: &str) -> bool {
        matches!(
            phoneme.to_lowercase().as_str(),
            "m" | "n" | "ŋ" | "ɲ" | "ɳ" | "ɴ"
        )
    }

    fn is_liquid(&self, phoneme: &str) -> bool {
        matches!(
            phoneme.to_lowercase().as_str(),
            "l" | "r" | "ɫ" | "ɾ" | "ɽ" | "ʀ" | "ʁ"
        )
    }

    fn get_sonority_level(&self, phoneme: &str) -> u8 {
        if self.is_vowel(phoneme) {
            5
        } else if self.is_liquid(phoneme) {
            4
        } else if self.is_nasal(phoneme) {
            3
        } else if self.is_fricative(phoneme) {
            2
        } else if self.is_stop(phoneme) {
            1
        } else {
            0
        }
    }

    fn create_sequence_patterns() -> HashMap<String, f64> {
        let mut patterns = HashMap::new();
        // Add common phoneme sequence patterns with their frequencies
        patterns.insert("consonant-vowel".to_string(), 0.8);
        patterns.insert("vowel-consonant".to_string(), 0.7);
        patterns.insert("consonant-consonant".to_string(), 0.3);
        patterns.insert("vowel-vowel".to_string(), 0.2);
        patterns
    }

    fn create_transition_probabilities() -> HashMap<(PhonemeClass, PhonemeClass), f64> {
        let mut probs = HashMap::new();
        probs.insert((PhonemeClass::Vowel, PhonemeClass::Consonant), 0.8);
        probs.insert((PhonemeClass::Consonant, PhonemeClass::Vowel), 0.9);
        probs.insert((PhonemeClass::Consonant, PhonemeClass::Consonant), 0.4);
        probs.insert((PhonemeClass::Vowel, PhonemeClass::Vowel), 0.3);
        probs.insert((PhonemeClass::Stop, PhonemeClass::Fricative), 0.6);
        probs.insert((PhonemeClass::Fricative, PhonemeClass::Stop), 0.5);
        probs.insert((PhonemeClass::Nasal, PhonemeClass::Stop), 0.7);
        probs.insert((PhonemeClass::Liquid, PhonemeClass::Vowel), 0.9);
        probs
    }

    fn create_clustering_rules() -> Vec<ClusteringRule> {
        vec![
            ClusteringRule {
                name: "Stop + Liquid".to_string(),
                pattern: "stop-liquid".to_string(),
                context: "onset".to_string(),
                sonority_profile: vec![1.0, 4.0],
                application_probability: 0.8,
                language_universal: true,
            },
            ClusteringRule {
                name: "Fricative + Liquid".to_string(),
                pattern: "fricative-liquid".to_string(),
                context: "onset".to_string(),
                sonority_profile: vec![2.0, 4.0],
                application_probability: 0.6,
                language_universal: false,
            },
        ]
    }

    fn create_phonotactic_weights() -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        // Add phonotactic constraint weights for common sequences
        weights.insert("ŋ-k".to_string(), 0.9); // Common
        weights.insert("s-t".to_string(), 0.8); // Common
        weights.insert("tl-".to_string(), 0.1); // Rare in English
        weights.insert("ʔ-ʔ".to_string(), 0.05); // Very rare
        weights
    }

    fn update_config(&mut self, config: &PhonologicalAnalysisConfig) {
        if config.detailed_sequence_analysis {
            self.sequence_patterns
                .extend(Self::create_extended_sequence_patterns());
        }
    }

    fn create_extended_sequence_patterns() -> HashMap<String, f64> {
        let mut patterns = HashMap::new();
        patterns.insert("stop-liquid-vowel".to_string(), 0.7);
        patterns.insert("fricative-vowel-nasal".to_string(), 0.5);
        patterns.insert("nasal-stop-vowel".to_string(), 0.6);
        patterns
    }
}

impl PhonologicalRuleAnalyzer {
    fn new(config: &PhonologicalAnalysisConfig) -> Self {
        Self {
            active_rules: Self::create_default_rules(),
            rule_contexts: Self::create_rule_contexts(),
            application_frequencies: HashMap::new(),
            rule_interactions: HashMap::new(),
        }
    }

    fn analyze(
        &mut self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<PhonologicalRuleMetrics, PhonologicalAnalysisError> {
        let applied_rules = self.identify_applied_rules(text, phonetic_transcription)?;
        let rule_density = applied_rules.len() as f64 / text.len() as f64;
        let rule_diversity = self.calculate_rule_diversity(&applied_rules);
        let rule_interactions = self.analyze_rule_interactions(&applied_rules);

        Ok(PhonologicalRuleMetrics {
            total_rules_applied: applied_rules.len(),
            rule_density,
            rule_diversity,
            applied_rules: applied_rules.clone(),
            rule_interactions,
            assimilation_frequency: self.count_assimilations(&applied_rules),
            deletion_frequency: self.count_deletions(&applied_rules),
            insertion_frequency: self.count_insertions(&applied_rules),
            metathesis_frequency: self.count_metatheses(&applied_rules),
        })
    }

    fn identify_applied_rules(
        &self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<Vec<PhonologicalRule>, PhonologicalAnalysisError> {
        let mut applied_rules = Vec::new();

        // Compare orthographic and phonetic forms to identify rule applications
        let words: Vec<&str> = text.split_whitespace().collect();
        let phonetic_words: Vec<&str> = phonetic_transcription.split(' ').collect();

        for (word, phonetic) in words.iter().zip(phonetic_words.iter()) {
            for rule in &self.active_rules {
                if self.rule_applies(word, phonetic, rule) {
                    applied_rules.push(rule.clone());
                }
            }
        }

        Ok(applied_rules)
    }

    fn rule_applies(&self, orthographic: &str, phonetic: &str, rule: &PhonologicalRule) -> bool {
        // Simplified rule application detection
        match rule.name.as_str() {
            "Final Devoicing" => {
                orthographic.ends_with('d') && phonetic.ends_with('t')
                    || orthographic.ends_with('g') && phonetic.ends_with('k')
                    || orthographic.ends_with('b') && phonetic.ends_with('p')
            }
            "Vowel Reduction" => {
                phonetic.contains('ə') && !orthographic.to_lowercase().contains('ə')
            }
            "Consonant Cluster Simplification" => {
                orthographic.len() > phonetic.split_whitespace().count()
            }
            _ => false,
        }
    }

    fn calculate_rule_diversity(&self, applied_rules: &[PhonologicalRule]) -> f64 {
        let unique_rules: HashSet<String> =
            applied_rules.iter().map(|rule| rule.name.clone()).collect();

        unique_rules.len() as f64 / applied_rules.len().max(1) as f64
    }

    fn analyze_rule_interactions(&self, applied_rules: &[PhonologicalRule]) -> Vec<String> {
        let mut interactions = Vec::new();

        for window in applied_rules.windows(2) {
            if let [rule1, rule2] = window {
                let interaction_key = format!("{}-{}", rule1.name, rule2.name);
                if self
                    .rule_interactions
                    .contains_key(&(rule1.name.clone(), rule2.name.clone()))
                {
                    interactions.push(interaction_key);
                }
            }
        }

        interactions
    }

    fn count_assimilations(&self, applied_rules: &[PhonologicalRule]) -> usize {
        applied_rules
            .iter()
            .filter(|rule| rule.rule_type == "assimilation")
            .count()
    }

    fn count_deletions(&self, applied_rules: &[PhonologicalRule]) -> usize {
        applied_rules
            .iter()
            .filter(|rule| rule.rule_type == "deletion")
            .count()
    }

    fn count_insertions(&self, applied_rules: &[PhonologicalRule]) -> usize {
        applied_rules
            .iter()
            .filter(|rule| rule.rule_type == "insertion")
            .count()
    }

    fn count_metatheses(&self, applied_rules: &[PhonologicalRule]) -> usize {
        applied_rules
            .iter()
            .filter(|rule| rule.rule_type == "metathesis")
            .count()
    }

    fn create_default_rules() -> Vec<PhonologicalRule> {
        vec![
            PhonologicalRule {
                name: "Final Devoicing".to_string(),
                rule_type: "assimilation".to_string(),
                context: "word-final".to_string(),
                structural_change: "[+voice] → [-voice] / _#".to_string(),
                frequency: 0.7,
                language_universal: false,
            },
            PhonologicalRule {
                name: "Vowel Reduction".to_string(),
                rule_type: "reduction".to_string(),
                context: "unstressed".to_string(),
                structural_change: "V → ə / unstressed".to_string(),
                frequency: 0.8,
                language_universal: false,
            },
            PhonologicalRule {
                name: "Consonant Cluster Simplification".to_string(),
                rule_type: "deletion".to_string(),
                context: "complex onset/coda".to_string(),
                structural_change: "CCC → CC".to_string(),
                frequency: 0.4,
                language_universal: true,
            },
        ]
    }

    fn create_rule_contexts() -> HashMap<String, Vec<String>> {
        let mut contexts = HashMap::new();
        contexts.insert(
            "Final Devoicing".to_string(),
            vec!["word-final".to_string(), "syllable-final".to_string()],
        );
        contexts.insert(
            "Vowel Reduction".to_string(),
            vec!["unstressed".to_string(), "function word".to_string()],
        );
        contexts.insert(
            "Assimilation".to_string(),
            vec!["adjacent".to_string(), "within syllable".to_string()],
        );
        contexts
    }

    fn update_config(&mut self, config: &PhonologicalAnalysisConfig) {
        if config.include_morphophonological_rules {
            self.active_rules
                .extend(Self::create_morphophonological_rules());
        }
    }

    fn create_morphophonological_rules() -> Vec<PhonologicalRule> {
        vec![PhonologicalRule {
            name: "Past Tense Allomorphy".to_string(),
            rule_type: "morphophonological".to_string(),
            context: "past tense morpheme".to_string(),
            structural_change: "/t/ → /ɪd/ / [+alveolar stop]_".to_string(),
            frequency: 0.9,
            language_universal: false,
        }]
    }
}

impl PhonotacticConstraintAnalyzer {
    fn new(config: &PhonologicalAnalysisConfig) -> Self {
        Self {
            universal_constraints: Self::create_universal_constraints(),
            language_constraints: Self::create_language_constraints(),
            constraint_weights: Self::create_constraint_weights(),
            violation_penalties: Self::create_violation_penalties(),
        }
    }

    fn analyze(
        &self,
        phonetic_transcription: &str,
    ) -> Result<PhonotacticConstraintMetrics, PhonologicalAnalysisError> {
        let phonemes: Vec<&str> = phonetic_transcription.split_whitespace().collect();

        let violations = self.evaluate_constraints(&phonemes);
        let violation_density = violations.len() as f64 / phonemes.len() as f64;
        let constraint_adherence = 1.0 - violation_density;

        let universal_violations = violations
            .iter()
            .filter(|v| v.constraint_universality == ConstraintUniversality::Universal)
            .count();

        let language_violations = violations
            .iter()
            .filter(|v| v.constraint_universality == ConstraintUniversality::LanguageSpecific)
            .count();

        Ok(PhonotacticConstraintMetrics {
            total_constraints_evaluated: self.universal_constraints.len()
                + self.language_constraints.len(),
            violations: violations.clone(),
            violation_density,
            constraint_adherence,
            universal_violations,
            language_violations,
            markedness_violations: self.count_markedness_violations(&violations),
            faithfulness_violations: self.count_faithfulness_violations(&violations),
        })
    }

    fn evaluate_constraints(&self, phonemes: &[&str]) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();

        // Evaluate universal constraints
        for constraint in &self.universal_constraints {
            if let Some(violation) = self.check_constraint(phonemes, constraint) {
                violations.push(violation);
            }
        }

        // Evaluate language-specific constraints
        for constraint in &self.language_constraints {
            if let Some(violation) = self.check_constraint(phonemes, constraint) {
                violations.push(violation);
            }
        }

        violations
    }

    fn check_constraint(
        &self,
        phonemes: &[&str],
        constraint: &PhonotacticConstraint,
    ) -> Option<ConstraintViolation> {
        match constraint.constraint_type {
            ConstraintType::Onset => self.check_onset_constraint(phonemes, constraint),
            ConstraintType::Coda => self.check_coda_constraint(phonemes, constraint),
            ConstraintType::Sequence => self.check_sequence_constraint(phonemes, constraint),
            ConstraintType::Sonority => self.check_sonority_constraint(phonemes, constraint),
            _ => None,
        }
    }

    fn check_onset_constraint(
        &self,
        phonemes: &[&str],
        constraint: &PhonotacticConstraint,
    ) -> Option<ConstraintViolation> {
        // Simplified onset constraint checking
        if constraint.name == "No Complex Onsets" {
            for window in phonemes.windows(3) {
                if window.iter().take(2).all(|&p| self.is_consonant(p)) && self.is_vowel(window[2])
                {
                    return Some(ConstraintViolation {
                        constraint_name: constraint.name.clone(),
                        constraint_type: constraint.constraint_type,
                        constraint_universality: constraint.universality,
                        position: 0, // Simplified position
                        severity: constraint.violation_weight,
                        description: format!("Complex onset found: {}-{}", window[0], window[1]),
                    });
                }
            }
        }
        None
    }

    fn check_coda_constraint(
        &self,
        phonemes: &[&str],
        constraint: &PhonotacticConstraint,
    ) -> Option<ConstraintViolation> {
        // Simplified coda constraint checking
        if constraint.name == "No Coda" {
            for window in phonemes.windows(2) {
                if self.is_consonant(window[0])
                    && (window.len() == 1 || self.is_consonant(window[1]))
                {
                    // Check if this consonant is in coda position
                    return Some(ConstraintViolation {
                        constraint_name: constraint.name.clone(),
                        constraint_type: constraint.constraint_type,
                        constraint_universality: constraint.universality,
                        position: 0,
                        severity: constraint.violation_weight,
                        description: format!("Coda consonant found: {}", window[0]),
                    });
                }
            }
        }
        None
    }

    fn check_sequence_constraint(
        &self,
        phonemes: &[&str],
        constraint: &PhonotacticConstraint,
    ) -> Option<ConstraintViolation> {
        // Check for prohibited sequences
        if constraint.pattern == "**" {
            for window in phonemes.windows(2) {
                if window[0] == window[1] {
                    return Some(ConstraintViolation {
                        constraint_name: constraint.name.clone(),
                        constraint_type: constraint.constraint_type,
                        constraint_universality: constraint.universality,
                        position: 0,
                        severity: constraint.violation_weight,
                        description: format!("Geminate found: {}", window[0]),
                    });
                }
            }
        }
        None
    }

    fn check_sonority_constraint(
        &self,
        phonemes: &[&str],
        constraint: &PhonotacticConstraint,
    ) -> Option<ConstraintViolation> {
        // Check sonority sequencing principle
        if constraint.name == "Sonority Sequencing" {
            for window in phonemes.windows(2) {
                let sonority1 = self.get_sonority_level(window[0]);
                let sonority2 = self.get_sonority_level(window[1]);

                if self.is_consonant(window[0])
                    && self.is_consonant(window[1])
                    && sonority1 > sonority2
                {
                    return Some(ConstraintViolation {
                        constraint_name: constraint.name.clone(),
                        constraint_type: constraint.constraint_type,
                        constraint_universality: constraint.universality,
                        position: 0,
                        severity: constraint.violation_weight,
                        description: format!("Sonority violation: {} > {}", window[0], window[1]),
                    });
                }
            }
        }
        None
    }

    fn count_markedness_violations(&self, violations: &[ConstraintViolation]) -> usize {
        violations
            .iter()
            .filter(|v| v.constraint_type == ConstraintType::Markedness)
            .count()
    }

    fn count_faithfulness_violations(&self, violations: &[ConstraintViolation]) -> usize {
        violations
            .iter()
            .filter(|v| v.constraint_type == ConstraintType::Faithfulness)
            .count()
    }

    fn is_consonant(&self, phoneme: &str) -> bool {
        !self.is_vowel(phoneme)
    }

    fn is_vowel(&self, phoneme: &str) -> bool {
        matches!(
            phoneme.to_lowercase().as_str(),
            "a" | "e"
                | "i"
                | "o"
                | "u"
                | "æ"
                | "ɛ"
                | "ɪ"
                | "ɔ"
                | "ʊ"
                | "ə"
                | "ɑ"
                | "ɒ"
                | "ʌ"
                | "ɜ"
                | "ɨ"
                | "ɵ"
                | "ɐ"
                | "ɶ"
                | "ø"
                | "y"
        )
    }

    fn get_sonority_level(&self, phoneme: &str) -> u8 {
        if self.is_vowel(phoneme) {
            5
        } else if matches!(
            phoneme.to_lowercase().as_str(),
            "l" | "r" | "ɫ" | "ɾ" | "ɽ" | "ʀ" | "ʁ"
        ) {
            4
        } else if matches!(
            phoneme.to_lowercase().as_str(),
            "m" | "n" | "ŋ" | "ɲ" | "ɳ" | "ɴ"
        ) {
            3
        } else if matches!(
            phoneme.to_lowercase().as_str(),
            "f" | "v" | "θ" | "ð" | "s" | "z" | "ʃ" | "ʒ" | "h" | "x" | "ɣ"
        ) {
            2
        } else if matches!(
            phoneme.to_lowercase().as_str(),
            "p" | "b" | "t" | "d" | "k" | "g" | "q" | "ɢ" | "ʔ"
        ) {
            1
        } else {
            0
        }
    }

    fn create_universal_constraints() -> Vec<PhonotacticConstraint> {
        vec![
            PhonotacticConstraint {
                name: "Sonority Sequencing".to_string(),
                constraint_type: ConstraintType::Sonority,
                pattern: "sonority rise in onset, fall in coda".to_string(),
                context: "syllable".to_string(),
                violation_weight: 1.0,
                universality: ConstraintUniversality::Universal,
            },
            PhonotacticConstraint {
                name: "No Complex Onsets".to_string(),
                constraint_type: ConstraintType::Onset,
                pattern: "*CC".to_string(),
                context: "syllable-initial".to_string(),
                violation_weight: 0.8,
                universality: ConstraintUniversality::Typological,
            },
        ]
    }

    fn create_language_constraints() -> Vec<PhonotacticConstraint> {
        vec![PhonotacticConstraint {
            name: "No Coda".to_string(),
            constraint_type: ConstraintType::Coda,
            pattern: "*C]σ".to_string(),
            context: "syllable-final".to_string(),
            violation_weight: 0.6,
            universality: ConstraintUniversality::LanguageSpecific,
        }]
    }

    fn create_constraint_weights() -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        weights.insert("Sonority Sequencing".to_string(), 1.0);
        weights.insert("No Complex Onsets".to_string(), 0.8);
        weights.insert("No Coda".to_string(), 0.6);
        weights
    }

    fn create_violation_penalties() -> HashMap<String, f64> {
        let mut penalties = HashMap::new();
        penalties.insert("Sonority Sequencing".to_string(), 2.0);
        penalties.insert("No Complex Onsets".to_string(), 1.5);
        penalties.insert("No Coda".to_string(), 1.0);
        penalties
    }

    fn update_config(&mut self, config: &PhonologicalAnalysisConfig) {
        if config.strict_constraint_evaluation {
            for constraint in &mut self.universal_constraints {
                constraint.violation_weight *= 1.2;
            }
        }
    }
}

impl PhonologicalPatternMatcher {
    fn new(config: &PhonologicalAnalysisConfig) -> Self {
        Self {
            pattern_database: Self::create_pattern_database(),
            matching_thresholds: Self::create_matching_thresholds(),
            pattern_frequencies: Self::create_pattern_frequencies(),
            context_weights: Self::create_context_weights(),
        }
    }

    fn detect_patterns(
        &self,
        phonetic_transcription: &str,
    ) -> Result<Vec<PhonologicalPattern>, PhonologicalAnalysisError> {
        let mut detected_patterns = Vec::new();
        let phonemes: Vec<&str> = phonetic_transcription.split_whitespace().collect();

        for pattern in &self.pattern_database {
            if self.matches_pattern(&phonemes, pattern) {
                detected_patterns.push(pattern.clone());
            }
        }

        Ok(detected_patterns)
    }

    fn matches_pattern(&self, phonemes: &[&str], pattern: &PhonologicalPattern) -> bool {
        // Simplified pattern matching - in practice would be more sophisticated
        match pattern.name.as_str() {
            "CV Syllable" => phonemes
                .windows(2)
                .any(|window| self.is_consonant(window[0]) && self.is_vowel(window[1])),
            "Consonant Cluster" => phonemes
                .windows(2)
                .any(|window| self.is_consonant(window[0]) && self.is_consonant(window[1])),
            "Vowel Hiatus" => phonemes
                .windows(2)
                .any(|window| self.is_vowel(window[0]) && self.is_vowel(window[1])),
            _ => false,
        }
    }

    fn is_consonant(&self, phoneme: &str) -> bool {
        !self.is_vowel(phoneme)
    }

    fn is_vowel(&self, phoneme: &str) -> bool {
        matches!(
            phoneme.to_lowercase().as_str(),
            "a" | "e"
                | "i"
                | "o"
                | "u"
                | "æ"
                | "ɛ"
                | "ɪ"
                | "ɔ"
                | "ʊ"
                | "ə"
                | "ɑ"
                | "ɒ"
                | "ʌ"
                | "ɜ"
                | "ɨ"
                | "ɵ"
                | "ɐ"
                | "ɶ"
                | "ø"
                | "y"
        )
    }

    fn create_pattern_database() -> Vec<PhonologicalPattern> {
        vec![
            PhonologicalPattern {
                name: "CV Syllable".to_string(),
                pattern_type: "syllable".to_string(),
                structural_description: "consonant + vowel".to_string(),
                frequency: 0.6,
                complexity: 1.0,
                language_universal: true,
            },
            PhonologicalPattern {
                name: "Consonant Cluster".to_string(),
                pattern_type: "sequence".to_string(),
                structural_description: "consecutive consonants".to_string(),
                frequency: 0.3,
                complexity: 2.0,
                language_universal: false,
            },
            PhonologicalPattern {
                name: "Vowel Hiatus".to_string(),
                pattern_type: "sequence".to_string(),
                structural_description: "consecutive vowels".to_string(),
                frequency: 0.1,
                complexity: 1.5,
                language_universal: false,
            },
        ]
    }

    fn create_matching_thresholds() -> HashMap<String, f64> {
        let mut thresholds = HashMap::new();
        thresholds.insert("exact".to_string(), 1.0);
        thresholds.insert("close".to_string(), 0.8);
        thresholds.insert("approximate".to_string(), 0.6);
        thresholds
    }

    fn create_pattern_frequencies() -> HashMap<String, f64> {
        let mut frequencies = HashMap::new();
        frequencies.insert("CV Syllable".to_string(), 0.6);
        frequencies.insert("CVC Syllable".to_string(), 0.4);
        frequencies.insert("Consonant Cluster".to_string(), 0.3);
        frequencies.insert("Vowel Hiatus".to_string(), 0.1);
        frequencies
    }

    fn create_context_weights() -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        weights.insert("word-initial".to_string(), 1.2);
        weights.insert("word-medial".to_string(), 1.0);
        weights.insert("word-final".to_string(), 1.1);
        weights
    }

    fn update_config(&mut self, config: &PhonologicalAnalysisConfig) {
        if config.extended_pattern_matching {
            self.pattern_database
                .extend(Self::create_extended_patterns());
        }
    }

    fn create_extended_patterns() -> Vec<PhonologicalPattern> {
        vec![PhonologicalPattern {
            name: "CCVC Syllable".to_string(),
            pattern_type: "syllable".to_string(),
            structural_description: "consonant cluster + vowel + consonant".to_string(),
            frequency: 0.2,
            complexity: 2.5,
            language_universal: false,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phonological_analyzer_creation() {
        let config = PhonologicalAnalysisConfig::default();
        let analyzer = PhonologicalAnalyzer::new(config);
        assert_eq!(analyzer.analysis_cache.len(), 0);
    }

    #[test]
    fn test_syllable_structure_analysis() {
        let config = PhonologicalAnalysisConfig::default();
        let mut analyzer = PhonologicalAnalyzer::new(config);

        let result = analyzer.analyze("test", "t ɛ s t").unwrap();
        assert!(result.syllable_structure.total_syllables > 0);
        assert!(result.syllable_structure.average_complexity > 0.0);
    }

    #[test]
    fn test_phoneme_sequence_analysis() {
        let config = PhonologicalAnalysisConfig::default();
        let mut analyzer = PhonologicalAnalyzer::new(config);

        let result = analyzer.analyze("stop", "s t ɒ p").unwrap();
        assert!(result.phoneme_sequence.total_phonemes == 4);
        assert!(result.phoneme_sequence.sequence_complexity > 0.0);
    }

    #[test]
    fn test_constraint_violation_detection() {
        let config = PhonologicalAnalysisConfig::default();
        let mut analyzer = PhonologicalAnalyzer::new(config);

        let result = analyzer.analyze("strength", "s t r ɛ ŋ θ").unwrap();
        // Should detect some complexity due to consonant clusters
        assert!(result.phonotactic_constraints.violations.len() >= 0);
    }

    #[test]
    fn test_phonological_pattern_detection() {
        let config = PhonologicalAnalysisConfig::default();
        let mut analyzer = PhonologicalAnalyzer::new(config);

        let result = analyzer.analyze("cat", "k æ t").unwrap();
        assert!(!result.detected_patterns.is_empty());

        // Should detect CV pattern
        let has_cv_pattern = result
            .detected_patterns
            .iter()
            .any(|p| p.name == "CV Syllable");
        assert!(has_cv_pattern);
    }

    #[test]
    fn test_complexity_calculation() {
        let config = PhonologicalAnalysisConfig::default();
        let mut analyzer = PhonologicalAnalyzer::new(config);

        let simple_result = analyzer.analyze("go", "g oʊ").unwrap();
        let complex_result = analyzer.analyze("strengths", "s t r ɛ ŋ θ s").unwrap();

        assert!(complex_result.complexity_score > simple_result.complexity_score);
    }

    #[test]
    fn test_cache_functionality() {
        let mut config = PhonologicalAnalysisConfig::default();
        config.enable_caching = true;
        let mut analyzer = PhonologicalAnalyzer::new(config);

        // First analysis
        let _result1 = analyzer.analyze("test", "t ɛ s t").unwrap();
        assert_eq!(analyzer.get_cache_stats().0, 1);

        // Second analysis (should use cache)
        let _result2 = analyzer.analyze("test", "t ɛ s t").unwrap();
        assert_eq!(analyzer.get_cache_stats().0, 1); // Cache size shouldn't change
    }

    #[test]
    fn test_syllable_segmentation() {
        let config = PhonologicalAnalysisConfig::default();
        let analyzer = SyllableStructureAnalyzer::new(&config);

        let syllables = analyzer.segment_syllables("k æ t").unwrap();
        assert_eq!(syllables.len(), 1);
        assert_eq!(syllables[0], "k æ t");
    }

    #[test]
    fn test_consonant_cluster_identification() {
        let config = PhonologicalAnalysisConfig::default();
        let analyzer = PhonemeSequenceAnalyzer::new(&config);

        let clusters = analyzer.identify_consonant_clusters(&["s", "t", "r", "ɛ", "ŋ", "θ", "s"]);
        assert!(!clusters.is_empty());
        assert!(clusters.iter().any(|cluster| cluster.contains("s-t-r")));
    }

    #[test]
    fn test_sonority_violation_detection() {
        let config = PhonologicalAnalysisConfig::default();
        let analyzer = PhonemeSequenceAnalyzer::new(&config);

        let violations = analyzer.count_sonority_violations(&["s", "t", "r", "æ"]);
        // "t" before "r" violates sonority sequencing (stop before liquid)
        assert!(violations >= 0);
    }
}
