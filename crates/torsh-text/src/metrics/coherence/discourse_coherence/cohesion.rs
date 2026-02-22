//! Cohesion analysis for discourse coherence
//!
//! This module provides comprehensive cohesion analysis including reference cohesion,
//! lexical cohesion, conjunctive cohesion, and temporal coherence analysis.

use std::collections::{HashMap, HashSet};
use thiserror::Error;

use super::config::{CohesionAnalysisConfig, CohesiveDeviceType, DiscourseCoherenceError};
use super::results::{
    CohesionAnalysis, CohesiveDevice, ComplexConjunction, ConjunctiveCohesionMetrics,
    LexicalCohesionMetrics, ReferenceChain, ReferenceCohesionMetrics, RepetitionAnalysis,
    SynonymCluster, SynonymNetworkMetrics, TemporalChain, TemporalCoherenceMetrics,
};

/// Errors specific to cohesion analysis
#[derive(Debug, Error)]
pub enum CohesionAnalysisError {
    #[error("Failed to analyze cohesive devices: {0}")]
    CohesiveDeviceAnalysisFailed(String),
    #[error("Reference resolution failed: {0}")]
    ReferenceResolutionFailed(String),
    #[error("Lexical cohesion analysis failed: {0}")]
    LexicalCohesionFailed(String),
    #[error("Temporal coherence analysis failed: {0}")]
    TemporalCoherenceFailed(String),
}

/// Specialized analyzer for cohesion
pub struct CohesionAnalyzer {
    config: CohesionAnalysisConfig,
    reference_patterns: HashSet<String>,
    conjunction_patterns: HashMap<String, String>,
    temporal_markers: HashSet<String>,
}

impl CohesionAnalyzer {
    /// Create a new cohesion analyzer
    pub fn new(config: CohesionAnalysisConfig) -> Self {
        let reference_patterns = Self::build_reference_patterns();
        let conjunction_patterns = Self::build_conjunction_patterns();
        let temporal_markers = Self::build_temporal_markers();

        Self {
            config,
            reference_patterns,
            conjunction_patterns,
            temporal_markers,
        }
    }

    /// Analyze cohesion in text
    pub fn analyze_cohesion(
        &self,
        sentences: &[String],
    ) -> Result<CohesionAnalysis, CohesionAnalysisError> {
        let cohesive_devices = self.identify_cohesive_devices(sentences)?;
        let overall_cohesion_score =
            self.calculate_overall_cohesion_score(&cohesive_devices, sentences);

        let reference_cohesion = if self.config.analyze_reference_cohesion {
            self.analyze_reference_cohesion(sentences)?
        } else {
            ReferenceCohesionMetrics::default()
        };

        let lexical_cohesion = if self.config.analyze_lexical_cohesion {
            self.analyze_lexical_cohesion(sentences)?
        } else {
            LexicalCohesionMetrics::default()
        };

        let conjunctive_cohesion = if self.config.analyze_conjunctive_cohesion {
            self.analyze_conjunctive_cohesion(sentences)?
        } else {
            ConjunctiveCohesionMetrics::default()
        };

        let temporal_coherence = self.analyze_temporal_coherence(sentences)?;

        Ok(CohesionAnalysis {
            overall_cohesion_score,
            cohesive_devices,
            reference_cohesion,
            lexical_cohesion,
            conjunctive_cohesion,
            temporal_coherence,
        })
    }

    /// Identify cohesive devices in text
    fn identify_cohesive_devices(
        &self,
        sentences: &[String],
    ) -> Result<Vec<CohesiveDevice>, CohesionAnalysisError> {
        let mut devices = Vec::new();

        // Analyze reference devices
        devices.extend(self.analyze_reference_devices(sentences));

        // Analyze conjunction devices
        devices.extend(self.analyze_conjunction_devices(sentences));

        // Analyze lexical cohesion devices
        devices.extend(self.analyze_lexical_cohesion_devices(sentences));

        Ok(devices)
    }

    /// Analyze reference cohesive devices
    fn analyze_reference_devices(&self, sentences: &[String]) -> Vec<CohesiveDevice> {
        let mut devices = Vec::new();

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let words: Vec<&str> = sentence.split_whitespace().collect();

            for (word_idx, word) in words.iter().enumerate() {
                if let Some(device_type) = self.classify_reference_device(word) {
                    let device = self.create_cohesive_device(
                        device_type,
                        word.to_string(),
                        sent_idx,
                        word_idx,
                        sentences,
                    );
                    devices.push(device);
                }
            }
        }

        devices
    }

    /// Classify reference cohesive device
    fn classify_reference_device(&self, word: &str) -> Option<CohesiveDeviceType> {
        let normalized = word
            .to_lowercase()
            .trim_matches(|c: char| !c.is_alphabetic())
            .to_string();

        match normalized.as_str() {
            "he" | "she" | "it" | "they" | "him" | "her" | "them" | "his" | "hers" | "its"
            | "their" => Some(CohesiveDeviceType::PersonalPronoun),
            "this" | "that" | "these" | "those" => Some(CohesiveDeviceType::Demonstrative),
            "such" | "same" | "other" | "another" | "similar" | "different" => {
                Some(CohesiveDeviceType::Comparative)
            }
            "one" | "ones" | "so" | "not" => Some(CohesiveDeviceType::Substitution),
            _ => None,
        }
    }

    /// Analyze conjunction cohesive devices
    fn analyze_conjunction_devices(&self, sentences: &[String]) -> Vec<CohesiveDevice> {
        let mut devices = Vec::new();

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let words: Vec<&str> = sentence.split_whitespace().collect();

            for (word_idx, word) in words.iter().enumerate() {
                if let Some(conjunction_type) = self.conjunction_patterns.get(&word.to_lowercase())
                {
                    let device = CohesiveDevice {
                        device_type: CohesiveDeviceType::Conjunction,
                        elements: vec![word.to_string()],
                        positions: vec![(sent_idx, word_idx)],
                        strength: self.calculate_conjunction_strength(word),
                        local_contribution: 0.7,
                        global_contribution: 0.5,
                        resolution_confidence: 0.8,
                        distance: 0, // Conjunctions typically have local effect
                    };
                    devices.push(device);
                }
            }

            // Check for multiword conjunctions
            devices.extend(self.find_multiword_conjunctions(sent_idx, &words));
        }

        devices
    }

    /// Find multiword conjunction patterns
    fn find_multiword_conjunctions(&self, sent_idx: usize, words: &[&str]) -> Vec<CohesiveDevice> {
        let mut devices = Vec::new();
        let multiword_conjunctions = [
            "in addition",
            "on the other hand",
            "as a result",
            "for example",
            "in contrast",
            "furthermore",
            "moreover",
            "however",
        ];

        let sentence = words.join(" ").to_lowercase();
        for pattern in &multiword_conjunctions {
            if sentence.contains(pattern) {
                let device = CohesiveDevice {
                    device_type: CohesiveDeviceType::Conjunction,
                    elements: vec![pattern.to_string()],
                    positions: vec![(sent_idx, 0)], // Simplified position
                    strength: 0.8,                  // Multiword conjunctions are typically strong
                    local_contribution: 0.8,
                    global_contribution: 0.7,
                    resolution_confidence: 0.9,
                    distance: 1,
                };
                devices.push(device);
            }
        }

        devices
    }

    /// Calculate conjunction strength
    fn calculate_conjunction_strength(&self, conjunction: &str) -> f64 {
        match conjunction.to_lowercase().as_str() {
            "however" | "nevertheless" | "nonetheless" => 0.9,
            "therefore" | "thus" | "consequently" => 0.85,
            "furthermore" | "moreover" | "additionally" => 0.8,
            "but" | "yet" | "although" => 0.75,
            "and" | "or" => 0.6,
            "so" | "then" => 0.7,
            _ => 0.5,
        }
    }

    /// Analyze lexical cohesion devices
    fn analyze_lexical_cohesion_devices(&self, sentences: &[String]) -> Vec<CohesiveDevice> {
        let mut devices = Vec::new();

        // Find repetitions
        devices.extend(self.find_lexical_repetitions(sentences));

        // Find synonymy relations
        devices.extend(self.find_synonym_relations(sentences));

        // Find collocation patterns
        devices.extend(self.find_collocation_patterns(sentences));

        devices
    }

    /// Find lexical repetitions
    fn find_lexical_repetitions(&self, sentences: &[String]) -> Vec<CohesiveDevice> {
        let mut devices = Vec::new();
        let mut word_positions: HashMap<String, Vec<(usize, usize)>> = HashMap::new();

        // Collect word positions
        for (sent_idx, sentence) in sentences.iter().enumerate() {
            for (word_idx, word) in sentence.split_whitespace().enumerate() {
                let normalized = word
                    .to_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();
                if normalized.len() > 3 {
                    // Only consider content words
                    word_positions
                        .entry(normalized)
                        .or_insert_with(Vec::new)
                        .push((sent_idx, word_idx));
                }
            }
        }

        // Create devices for repeated words
        for (word, positions) in word_positions {
            if positions.len() > 1 {
                let distance = if positions.len() > 1 {
                    positions[positions.len() - 1].0 - positions[0].0
                } else {
                    0
                };

                let device = CohesiveDevice {
                    device_type: CohesiveDeviceType::Repetition,
                    elements: vec![word],
                    positions,
                    strength: 0.7,
                    local_contribution: 0.6,
                    global_contribution: 0.4,
                    resolution_confidence: 1.0, // Repetition is always certain
                    distance,
                };
                devices.push(device);
            }
        }

        devices
    }

    /// Find synonym relations (simplified)
    fn find_synonym_relations(&self, sentences: &[String]) -> Vec<CohesiveDevice> {
        let mut devices = Vec::new();
        let synonym_pairs = self.get_common_synonym_pairs();

        for (word1, word2) in synonym_pairs {
            let positions1 = self.find_word_positions(sentences, &word1);
            let positions2 = self.find_word_positions(sentences, &word2);

            if !positions1.is_empty() && !positions2.is_empty() {
                let all_positions = [positions1, positions2].concat();
                let distance = self.calculate_max_distance(&all_positions);

                let device = CohesiveDevice {
                    device_type: CohesiveDeviceType::Synonymy,
                    elements: vec![word1, word2],
                    positions: all_positions,
                    strength: 0.8,
                    local_contribution: 0.7,
                    global_contribution: 0.6,
                    resolution_confidence: 0.7,
                    distance,
                };
                devices.push(device);
            }
        }

        devices
    }

    /// Get common synonym pairs
    fn get_common_synonym_pairs(&self) -> Vec<(String, String)> {
        vec![
            ("big".to_string(), "large".to_string()),
            ("small".to_string(), "little".to_string()),
            ("good".to_string(), "great".to_string()),
            ("bad".to_string(), "poor".to_string()),
            ("start".to_string(), "begin".to_string()),
            ("end".to_string(), "finish".to_string()),
            ("important".to_string(), "significant".to_string()),
            ("problem".to_string(), "issue".to_string()),
        ]
    }

    /// Find collocation patterns
    fn find_collocation_patterns(&self, sentences: &[String]) -> Vec<CohesiveDevice> {
        let mut devices = Vec::new();
        let collocations = self.get_common_collocations();

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let normalized = sentence.to_lowercase();
            for (word1, word2) in &collocations {
                if normalized.contains(word1) && normalized.contains(word2) {
                    let device = CohesiveDevice {
                        device_type: CohesiveDeviceType::Collocation,
                        elements: vec![word1.clone(), word2.clone()],
                        positions: vec![(sent_idx, 0)], // Simplified positioning
                        strength: 0.6,
                        local_contribution: 0.5,
                        global_contribution: 0.3,
                        resolution_confidence: 0.6,
                        distance: 0,
                    };
                    devices.push(device);
                }
            }
        }

        devices
    }

    /// Get common collocations
    fn get_common_collocations(&self) -> Vec<(String, String)> {
        vec![
            ("make".to_string(), "decision".to_string()),
            ("take".to_string(), "action".to_string()),
            ("pay".to_string(), "attention".to_string()),
            ("conduct".to_string(), "research".to_string()),
            ("strong".to_string(), "evidence".to_string()),
            ("clear".to_string(), "example".to_string()),
        ]
    }

    /// Find positions of a word in sentences
    fn find_word_positions(&self, sentences: &[String], target_word: &str) -> Vec<(usize, usize)> {
        let mut positions = Vec::new();

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            for (word_idx, word) in sentence.split_whitespace().enumerate() {
                let normalized = word
                    .to_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();
                if normalized == target_word.to_lowercase() {
                    positions.push((sent_idx, word_idx));
                }
            }
        }

        positions
    }

    /// Calculate maximum distance between positions
    fn calculate_max_distance(&self, positions: &[(usize, usize)]) -> usize {
        if positions.len() < 2 {
            return 0;
        }

        let min_sent = positions.iter().map(|(sent, _)| *sent).min().unwrap_or(0);
        let max_sent = positions.iter().map(|(sent, _)| *sent).max().unwrap_or(0);
        max_sent - min_sent
    }

    /// Create a cohesive device
    fn create_cohesive_device(
        &self,
        device_type: CohesiveDeviceType,
        element: String,
        sent_idx: usize,
        word_idx: usize,
        sentences: &[String],
    ) -> CohesiveDevice {
        let strength = self.calculate_device_strength(&device_type);
        let (local_contribution, global_contribution) = self.calculate_contributions(&device_type);
        let resolution_confidence =
            self.calculate_resolution_confidence(&device_type, &element, sent_idx, sentences);
        let distance = self.calculate_reference_distance(&device_type, sent_idx, sentences);

        CohesiveDevice {
            device_type,
            elements: vec![element],
            positions: vec![(sent_idx, word_idx)],
            strength,
            local_contribution,
            global_contribution,
            resolution_confidence,
            distance,
        }
    }

    /// Calculate device strength based on type
    fn calculate_device_strength(&self, device_type: &CohesiveDeviceType) -> f64 {
        self.config
            .device_weights
            .get(device_type)
            .cloned()
            .unwrap_or(0.5)
    }

    /// Calculate local and global contributions
    fn calculate_contributions(&self, device_type: &CohesiveDeviceType) -> (f64, f64) {
        match device_type {
            CohesiveDeviceType::PersonalPronoun => (0.8, 0.6),
            CohesiveDeviceType::Demonstrative => (0.9, 0.5),
            CohesiveDeviceType::Conjunction => (0.7, 0.8),
            CohesiveDeviceType::Repetition => (0.6, 0.7),
            CohesiveDeviceType::Synonymy => (0.7, 0.8),
            _ => (0.5, 0.5),
        }
    }

    /// Calculate resolution confidence
    fn calculate_resolution_confidence(
        &self,
        device_type: &CohesiveDeviceType,
        element: &str,
        sent_idx: usize,
        sentences: &[String],
    ) -> f64 {
        match device_type {
            CohesiveDeviceType::PersonalPronoun => {
                // Higher confidence if there are clear antecedents nearby
                self.calculate_antecedent_clarity(element, sent_idx, sentences)
            }
            CohesiveDeviceType::Demonstrative => 0.7,
            CohesiveDeviceType::Conjunction => 0.9,
            CohesiveDeviceType::Repetition => 1.0,
            _ => 0.6,
        }
    }

    /// Calculate antecedent clarity for pronouns
    fn calculate_antecedent_clarity(
        &self,
        pronoun: &str,
        sent_idx: usize,
        sentences: &[String],
    ) -> f64 {
        // Simplified antecedent detection
        let search_window = 3; // Look back up to 3 sentences
        let start_idx = sent_idx.saturating_sub(search_window);

        let mut potential_antecedents = 0;
        for i in start_idx..sent_idx {
            if let Some(sentence) = sentences.get(i) {
                potential_antecedents += self.count_potential_antecedents(sentence, pronoun);
            }
        }

        // More potential antecedents = lower clarity (more ambiguity)
        match potential_antecedents {
            0 => 0.3, // No clear antecedent
            1 => 0.9, // Clear single antecedent
            2 => 0.6, // Some ambiguity
            _ => 0.4, // High ambiguity
        }
    }

    /// Count potential antecedents for a pronoun
    fn count_potential_antecedents(&self, sentence: &str, pronoun: &str) -> usize {
        let normalized_pronoun = pronoun.to_lowercase();
        let words: Vec<&str> = sentence.split_whitespace().collect();

        words
            .iter()
            .filter(|word| {
                let normalized = word.to_lowercase();
                match normalized_pronoun.as_str() {
                    "he" | "him" | "his" => self.is_masculine_noun(&normalized),
                    "she" | "her" | "hers" => self.is_feminine_noun(&normalized),
                    "it" | "its" => self.is_neuter_noun(&normalized),
                    "they" | "them" | "their" => self.is_plural_noun(&normalized),
                    _ => false,
                }
            })
            .count()
    }

    /// Check if word is a masculine noun (simplified)
    fn is_masculine_noun(&self, word: &str) -> bool {
        ["man", "boy", "father", "brother", "son", "grandfather"].contains(&word)
    }

    /// Check if word is a feminine noun (simplified)
    fn is_feminine_noun(&self, word: &str) -> bool {
        [
            "woman",
            "girl",
            "mother",
            "sister",
            "daughter",
            "grandmother",
        ]
        .contains(&word)
    }

    /// Check if word is a neuter noun (simplified)
    fn is_neuter_noun(&self, word: &str) -> bool {
        ["thing", "object", "item", "concept", "idea", "system"].contains(&word)
    }

    /// Check if word is a plural noun (simplified)
    fn is_plural_noun(&self, word: &str) -> bool {
        word.ends_with('s') && !word.ends_with("ss") && word.len() > 3
    }

    /// Calculate reference distance
    fn calculate_reference_distance(
        &self,
        device_type: &CohesiveDeviceType,
        sent_idx: usize,
        _sentences: &[String],
    ) -> usize {
        match device_type {
            CohesiveDeviceType::PersonalPronoun | CohesiveDeviceType::Demonstrative => {
                // These typically refer to something in previous sentences
                1 + sent_idx.saturating_sub(1)
            }
            _ => 0,
        }
    }

    /// Calculate overall cohesion score
    fn calculate_overall_cohesion_score(
        &self,
        devices: &[CohesiveDevice],
        sentences: &[String],
    ) -> f64 {
        if devices.is_empty() || sentences.is_empty() {
            return 0.0;
        }

        let total_strength: f64 = devices.iter().map(|d| d.strength).sum();
        let device_density = devices.len() as f64 / sentences.len() as f64;

        // Normalize by expected values
        let normalized_strength = total_strength / devices.len() as f64;
        let normalized_density = device_density.min(2.0) / 2.0; // Cap at 2 devices per sentence

        (normalized_strength * 0.7 + normalized_density * 0.3).min(1.0)
    }

    /// Analyze reference cohesion
    fn analyze_reference_cohesion(
        &self,
        sentences: &[String],
    ) -> Result<ReferenceCohesionMetrics, CohesionAnalysisError> {
        let reference_chains = self.build_reference_chains(sentences)?;
        let total_references = self.count_total_references(sentences);
        let reference_density = total_references as f64 / sentences.len() as f64;
        let resolution_success_rate = self.calculate_resolution_success_rate(&reference_chains);
        let average_reference_distance =
            self.calculate_average_reference_distance(&reference_chains);
        let ambiguous_references = self.count_ambiguous_references(sentences);
        let complexity_score = self.calculate_reference_complexity(&reference_chains);

        Ok(ReferenceCohesionMetrics {
            total_references,
            reference_density,
            resolution_success_rate,
            average_reference_distance,
            ambiguous_references,
            complexity_score,
            reference_chains,
        })
    }

    /// Build reference chains (simplified)
    fn build_reference_chains(
        &self,
        sentences: &[String],
    ) -> Result<Vec<ReferenceChain>, CohesionAnalysisError> {
        let mut chains = Vec::new();

        // This is a simplified implementation
        // A full implementation would use coreference resolution

        for (chain_id, sentence) in sentences.iter().enumerate() {
            if self.contains_pronouns(sentence) {
                let chain = ReferenceChain {
                    chain_id,
                    entity: format!("entity_{}", chain_id),
                    referring_expressions: vec![sentence.clone()],
                    positions: vec![(chain_id, 0)],
                    coherence_score: 0.7,
                    completeness_score: 0.6,
                };
                chains.push(chain);
            }
        }

        Ok(chains)
    }

    /// Check if sentence contains pronouns
    fn contains_pronouns(&self, sentence: &str) -> bool {
        let pronouns = [
            "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their",
        ];
        let words: Vec<&str> = sentence.split_whitespace().collect();
        words
            .iter()
            .any(|word| pronouns.contains(&word.to_lowercase().as_str()))
    }

    /// Count total references in text
    fn count_total_references(&self, sentences: &[String]) -> usize {
        sentences
            .iter()
            .map(|sentence| self.count_references_in_sentence(sentence))
            .sum()
    }

    /// Count references in a single sentence
    fn count_references_in_sentence(&self, sentence: &str) -> usize {
        let reference_words = ["he", "she", "it", "they", "this", "that", "these", "those"];
        sentence
            .split_whitespace()
            .filter(|word| reference_words.contains(&word.to_lowercase().as_str()))
            .count()
    }

    /// Calculate resolution success rate
    fn calculate_resolution_success_rate(&self, chains: &[ReferenceChain]) -> f64 {
        if chains.is_empty() {
            return 0.0;
        }

        let successful_chains = chains
            .iter()
            .filter(|chain| chain.coherence_score > 0.5)
            .count();
        successful_chains as f64 / chains.len() as f64
    }

    /// Calculate average reference distance
    fn calculate_average_reference_distance(&self, chains: &[ReferenceChain]) -> f64 {
        if chains.is_empty() {
            return 0.0;
        }

        let total_distance: usize = chains
            .iter()
            .map(|chain| {
                if chain.positions.len() > 1 {
                    let first_pos = chain.positions[0].0;
                    let last_pos = chain.positions[chain.positions.len() - 1].0;
                    last_pos - first_pos
                } else {
                    0
                }
            })
            .sum();

        total_distance as f64 / chains.len() as f64
    }

    /// Count ambiguous references
    fn count_ambiguous_references(&self, sentences: &[String]) -> usize {
        // Simplified: count sentences with multiple potential antecedents
        sentences
            .iter()
            .enumerate()
            .filter(|(i, sentence)| {
                self.contains_pronouns(sentence)
                    && self.has_multiple_potential_antecedents(*i, sentences)
            })
            .count()
    }

    /// Check if there are multiple potential antecedents
    fn has_multiple_potential_antecedents(&self, sent_idx: usize, sentences: &[String]) -> bool {
        if sent_idx == 0 {
            return false;
        }

        let search_window = 2;
        let start_idx = sent_idx.saturating_sub(search_window);

        let mut antecedent_count = 0;
        for i in start_idx..sent_idx {
            if let Some(sentence) = sentences.get(i) {
                antecedent_count += self.count_potential_antecedents_general(sentence);
            }
        }

        antecedent_count > 1
    }

    /// Count general potential antecedents in sentence
    fn count_potential_antecedents_general(&self, sentence: &str) -> usize {
        sentence
            .split_whitespace()
            .filter(|word| {
                let normalized = word.to_lowercase();
                self.is_potential_antecedent(&normalized)
            })
            .count()
    }

    /// Check if word could be a potential antecedent
    fn is_potential_antecedent(&self, word: &str) -> bool {
        // Simplified: nouns that could be referred to by pronouns
        word.len() > 3
            && ![
                "the", "and", "that", "this", "with", "from", "they", "have", "been", "were",
            ]
            .contains(&word)
            && word.chars().all(|c| c.is_alphabetic())
    }

    /// Calculate reference complexity
    fn calculate_reference_complexity(&self, chains: &[ReferenceChain]) -> f64 {
        if chains.is_empty() {
            return 0.0;
        }

        let avg_chain_length: f64 = chains
            .iter()
            .map(|chain| chain.referring_expressions.len() as f64)
            .sum::<f64>()
            / chains.len() as f64;

        let avg_distance = self.calculate_average_reference_distance(chains);

        // Complexity increases with longer chains and greater distances
        (avg_chain_length / 5.0 + avg_distance / 10.0).min(1.0)
    }

    /// Analyze lexical cohesion
    fn analyze_lexical_cohesion(
        &self,
        sentences: &[String],
    ) -> Result<LexicalCohesionMetrics, CohesionAnalysisError> {
        let lexical_ties = self.count_lexical_ties(sentences);
        let lexical_density = lexical_ties as f64 / sentences.len() as f64;
        let repetition_analysis = self.analyze_repetition_patterns(sentences);
        let synonym_networks = self.analyze_synonym_networks(sentences);
        let semantic_field_coherence = self.calculate_semantic_field_coherence(sentences);
        let sophistication_score = self.calculate_lexical_sophistication(sentences);

        Ok(LexicalCohesionMetrics {
            lexical_ties,
            lexical_density,
            repetition_analysis,
            synonym_networks,
            semantic_field_coherence,
            sophistication_score,
        })
    }

    /// Count lexical ties between sentences
    fn count_lexical_ties(&self, sentences: &[String]) -> usize {
        let mut ties = 0;
        let mut word_positions: HashMap<String, Vec<usize>> = HashMap::new();

        // Collect word positions
        for (sent_idx, sentence) in sentences.iter().enumerate() {
            for word in sentence.split_whitespace() {
                let normalized = word
                    .to_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();
                if normalized.len() > 3 {
                    word_positions
                        .entry(normalized)
                        .or_insert_with(Vec::new)
                        .push(sent_idx);
                }
            }
        }

        // Count ties (words appearing in multiple sentences)
        for positions in word_positions.values() {
            if positions.len() > 1 {
                ties += positions.len() - 1; // n positions create n-1 ties
            }
        }

        ties
    }

    /// Analyze repetition patterns
    fn analyze_repetition_patterns(&self, sentences: &[String]) -> RepetitionAnalysis {
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut morphological_variants: HashMap<String, HashSet<String>> = HashMap::new();

        for sentence in sentences {
            for word in sentence.split_whitespace() {
                let normalized = word
                    .to_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();
                if normalized.len() > 3 {
                    *word_counts.entry(normalized.clone()).or_insert(0) += 1;

                    // Track morphological variants (simplified)
                    let stem = self.simple_stem(&normalized);
                    morphological_variants
                        .entry(stem)
                        .or_insert_with(HashSet::new)
                        .insert(normalized);
                }
            }
        }

        let exact_repetitions = word_counts.values().filter(|&&count| count > 1).count();
        let morphological_variations = morphological_variants
            .values()
            .filter(|variants| variants.len() > 1)
            .count();

        let mut frequent_terms: Vec<(String, usize)> = word_counts
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .collect();
        frequent_terms.sort_by(|a, b| b.1.cmp(&a.1));
        frequent_terms.truncate(10);

        let distribution_score = self.calculate_repetition_distribution(&frequent_terms);

        RepetitionAnalysis {
            exact_repetitions,
            morphological_variations,
            frequent_terms,
            distribution_score,
        }
    }

    /// Simple stemming function
    fn simple_stem(&self, word: &str) -> String {
        if word.ends_with("ing") && word.len() > 6 {
            word[..word.len() - 3].to_string()
        } else if word.ends_with("ed") && word.len() > 5 {
            word[..word.len() - 2].to_string()
        } else if word.ends_with("s") && word.len() > 4 {
            word[..word.len() - 1].to_string()
        } else {
            word.to_string()
        }
    }

    /// Calculate repetition distribution score
    fn calculate_repetition_distribution(&self, frequent_terms: &[(String, usize)]) -> f64 {
        if frequent_terms.is_empty() {
            return 0.0;
        }

        let total_repetitions: usize = frequent_terms.iter().map(|(_, count)| count).sum();
        let unique_terms = frequent_terms.len();

        // Calculate entropy of distribution
        let mut entropy = 0.0;
        for (_, count) in frequent_terms {
            let probability = *count as f64 / total_repetitions as f64;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }

        let max_entropy = (unique_terms as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Analyze synonym networks (simplified)
    fn analyze_synonym_networks(&self, sentences: &[String]) -> SynonymNetworkMetrics {
        let synonym_pairs = self.get_common_synonym_pairs();
        let mut clusters = Vec::new();
        let mut cluster_id = 0;

        for (word1, word2) in synonym_pairs {
            let positions1 = self.find_word_positions(sentences, &word1);
            let positions2 = self.find_word_positions(sentences, &word2);

            if !positions1.is_empty() && !positions2.is_empty() {
                let cluster = SynonymCluster {
                    cluster_id,
                    words: vec![word1, word2],
                    coherence_score: 0.8,
                    similarity_threshold: 0.7,
                };
                clusters.push(cluster);
                cluster_id += 1;
            }
        }

        let cluster_count = clusters.len();
        let average_cluster_size = if cluster_count > 0 {
            clusters.iter().map(|c| c.words.len()).sum::<usize>() as f64 / cluster_count as f64
        } else {
            0.0
        };

        let connectivity_score = cluster_count as f64 / 10.0; // Normalize by expected max
        let major_clusters = clusters.into_iter().take(5).collect();

        SynonymNetworkMetrics {
            cluster_count,
            average_cluster_size,
            connectivity_score: connectivity_score.min(1.0),
            major_clusters,
        }
    }

    /// Calculate semantic field coherence
    fn calculate_semantic_field_coherence(&self, sentences: &[String]) -> f64 {
        // This is a simplified implementation
        // A full implementation would use semantic similarity models
        0.6 // Placeholder value
    }

    /// Calculate lexical sophistication
    fn calculate_lexical_sophistication(&self, sentences: &[String]) -> f64 {
        let mut total_words = 0;
        let mut sophisticated_words = 0;
        let sophisticated_patterns = ["tion", "sion", "ment", "ness", "ity", "ency"];

        for sentence in sentences {
            for word in sentence.split_whitespace() {
                let normalized = word
                    .to_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();
                if normalized.len() > 3 {
                    total_words += 1;
                    if normalized.len() > 7
                        || sophisticated_patterns
                            .iter()
                            .any(|pattern| normalized.contains(pattern))
                    {
                        sophisticated_words += 1;
                    }
                }
            }
        }

        if total_words > 0 {
            sophisticated_words as f64 / total_words as f64
        } else {
            0.0
        }
    }

    /// Analyze conjunctive cohesion
    fn analyze_conjunctive_cohesion(
        &self,
        sentences: &[String],
    ) -> Result<ConjunctiveCohesionMetrics, CohesionAnalysisError> {
        let conjunction_counts = self.count_conjunctions_by_type(sentences);
        let conjunctive_density =
            self.calculate_conjunctive_density(&conjunction_counts, sentences.len());
        let logical_flow_score = self.calculate_logical_flow_score(sentences);
        let conjunction_effectiveness = self.calculate_conjunction_effectiveness(sentences);
        let complex_conjunctions = self.analyze_complex_conjunctions(sentences);

        Ok(ConjunctiveCohesionMetrics {
            conjunction_counts,
            conjunctive_density,
            logical_flow_score,
            conjunction_effectiveness,
            complex_conjunctions,
        })
    }

    /// Count conjunctions by type
    fn count_conjunctions_by_type(&self, sentences: &[String]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        for sentence in sentences {
            for word in sentence.split_whitespace() {
                let normalized = word
                    .to_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();
                if let Some(conjunction_type) = self.conjunction_patterns.get(&normalized) {
                    *counts.entry(conjunction_type.clone()).or_insert(0) += 1;
                }
            }
        }

        counts
    }

    /// Calculate conjunctive density
    fn calculate_conjunctive_density(
        &self,
        conjunction_counts: &HashMap<String, usize>,
        sentence_count: usize,
    ) -> f64 {
        let total_conjunctions: usize = conjunction_counts.values().sum();
        if sentence_count > 0 {
            total_conjunctions as f64 / sentence_count as f64
        } else {
            0.0
        }
    }

    /// Calculate logical flow score
    fn calculate_logical_flow_score(&self, sentences: &[String]) -> f64 {
        // Simplified logical flow analysis
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut flow_score = 0.0;
        for i in 1..sentences.len() {
            flow_score += self.calculate_sentence_pair_flow(&sentences[i - 1], &sentences[i]);
        }

        flow_score / (sentences.len() - 1) as f64
    }

    /// Calculate flow between sentence pair
    fn calculate_sentence_pair_flow(&self, sent1: &str, sent2: &str) -> f64 {
        // Check for logical connectors at the beginning of the second sentence
        let logical_connectors = [
            "however",
            "therefore",
            "furthermore",
            "moreover",
            "consequently",
            "thus",
        ];
        let sent2_lower = sent2.to_lowercase();

        for connector in &logical_connectors {
            if sent2_lower.starts_with(connector) {
                return 0.8; // High flow with explicit connector
            }
        }

        // Check for implicit logical flow (simplified)
        let lexical_overlap = self.calculate_lexical_overlap_simple(sent1, sent2);
        lexical_overlap * 0.6 // Implicit flow based on lexical connection
    }

    /// Simple lexical overlap calculation
    fn calculate_lexical_overlap_simple(&self, sent1: &str, sent2: &str) -> f64 {
        let words1: HashSet<&str> = sent1.split_whitespace().collect();
        let words2: HashSet<&str> = sent2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    /// Calculate conjunction effectiveness
    fn calculate_conjunction_effectiveness(&self, sentences: &[String]) -> f64 {
        let mut total_effectiveness = 0.0;
        let mut conjunction_count = 0;

        for sentence in sentences {
            for word in sentence.split_whitespace() {
                let normalized = word.to_lowercase();
                if self.conjunction_patterns.contains_key(&normalized) {
                    total_effectiveness += self.calculate_conjunction_strength(&word);
                    conjunction_count += 1;
                }
            }
        }

        if conjunction_count > 0 {
            total_effectiveness / conjunction_count as f64
        } else {
            0.0
        }
    }

    /// Analyze complex conjunctions
    fn analyze_complex_conjunctions(&self, sentences: &[String]) -> Vec<ComplexConjunction> {
        let mut complex_conjunctions = Vec::new();
        let complex_patterns = [
            ("on the other hand", "contrast"),
            ("in addition to", "addition"),
            ("as a result of", "causation"),
            ("in spite of", "concession"),
            ("for the purpose of", "purpose"),
        ];

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let normalized = sentence.to_lowercase();
            for (pattern, relation) in &complex_patterns {
                if normalized.contains(pattern) {
                    let conjunction = ComplexConjunction {
                        text: pattern.to_string(),
                        logical_relation: relation.to_string(),
                        position: (sent_idx, 0), // Simplified position
                        effectiveness: 0.8,
                        scope: 2, // Affects current and next sentence
                    };
                    complex_conjunctions.push(conjunction);
                }
            }
        }

        complex_conjunctions
    }

    /// Analyze temporal coherence
    fn analyze_temporal_coherence(
        &self,
        sentences: &[String],
    ) -> Result<TemporalCoherenceMetrics, CohesionAnalysisError> {
        let temporal_marker_frequency = self.calculate_temporal_marker_frequency(sentences);
        let sequence_coherence = self.calculate_sequence_coherence(sentences);
        let anchoring_score = self.calculate_temporal_anchoring(sentences);
        let timeline_consistency = self.calculate_timeline_consistency(sentences);
        let temporal_disruptions = self.count_temporal_disruptions(sentences);
        let temporal_chains = self.identify_temporal_chains(sentences)?;

        Ok(TemporalCoherenceMetrics {
            temporal_marker_frequency,
            sequence_coherence,
            anchoring_score,
            timeline_consistency,
            temporal_disruptions,
            temporal_chains,
        })
    }

    /// Calculate temporal marker frequency
    fn calculate_temporal_marker_frequency(&self, sentences: &[String]) -> f64 {
        let mut temporal_marker_count = 0;
        let total_sentences = sentences.len();

        for sentence in sentences {
            for word in sentence.split_whitespace() {
                if self.temporal_markers.contains(&word.to_lowercase()) {
                    temporal_marker_count += 1;
                    break; // Count at most one per sentence
                }
            }
        }

        if total_sentences > 0 {
            temporal_marker_count as f64 / total_sentences as f64
        } else {
            0.0
        }
    }

    /// Calculate sequence coherence
    fn calculate_sequence_coherence(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut coherence_sum = 0.0;
        for i in 1..sentences.len() {
            coherence_sum +=
                self.calculate_temporal_coherence_pair(&sentences[i - 1], &sentences[i]);
        }

        coherence_sum / (sentences.len() - 1) as f64
    }

    /// Calculate temporal coherence between sentence pair
    fn calculate_temporal_coherence_pair(&self, sent1: &str, sent2: &str) -> f64 {
        let temporal_score1 = self.calculate_sentence_temporal_score(sent1);
        let temporal_score2 = self.calculate_sentence_temporal_score(sent2);

        // Higher coherence if both sentences have temporal elements
        if temporal_score1 > 0.0 && temporal_score2 > 0.0 {
            0.8
        } else if temporal_score1 > 0.0 || temporal_score2 > 0.0 {
            0.6
        } else {
            0.4 // Neutral temporal coherence
        }
    }

    /// Calculate temporal score for a sentence
    fn calculate_sentence_temporal_score(&self, sentence: &str) -> f64 {
        let temporal_markers_count = sentence
            .split_whitespace()
            .filter(|word| self.temporal_markers.contains(&word.to_lowercase()))
            .count();

        (temporal_markers_count as f64).min(1.0)
    }

    /// Calculate temporal anchoring
    fn calculate_temporal_anchoring(&self, sentences: &[String]) -> f64 {
        // Count sentences with specific temporal references
        let anchored_sentences = sentences
            .iter()
            .filter(|sentence| self.has_specific_temporal_reference(sentence))
            .count();

        if sentences.is_empty() {
            0.0
        } else {
            anchored_sentences as f64 / sentences.len() as f64
        }
    }

    /// Check if sentence has specific temporal reference
    fn has_specific_temporal_reference(&self, sentence: &str) -> bool {
        let specific_temporal_patterns = [
            "yesterday",
            "today",
            "tomorrow",
            "monday",
            "tuesday",
            "wednesday",
            "january",
            "february",
            "march",
            "2023",
            "2024",
            "morning",
            "evening",
        ];

        let normalized = sentence.to_lowercase();
        specific_temporal_patterns
            .iter()
            .any(|pattern| normalized.contains(pattern))
    }

    /// Calculate timeline consistency
    fn calculate_timeline_consistency(&self, sentences: &[String]) -> f64 {
        // This is a simplified implementation
        // A full implementation would track temporal references and check for consistency
        0.7 // Placeholder value
    }

    /// Count temporal disruptions
    fn count_temporal_disruptions(&self, sentences: &[String]) -> usize {
        // Simplified: count abrupt temporal shifts
        let mut disruptions = 0;

        for i in 1..sentences.len() {
            if self.has_temporal_disruption(&sentences[i - 1], &sentences[i]) {
                disruptions += 1;
            }
        }

        disruptions
    }

    /// Check for temporal disruption between sentences
    fn has_temporal_disruption(&self, sent1: &str, sent2: &str) -> bool {
        let past_indicators = ["was", "were", "had", "did", "yesterday", "before"];
        let future_indicators = ["will", "shall", "tomorrow", "later", "next"];

        let sent1_is_past = past_indicators
            .iter()
            .any(|indicator| sent1.to_lowercase().contains(indicator));
        let sent1_is_future = future_indicators
            .iter()
            .any(|indicator| sent1.to_lowercase().contains(indicator));

        let sent2_is_past = past_indicators
            .iter()
            .any(|indicator| sent2.to_lowercase().contains(indicator));
        let sent2_is_future = future_indicators
            .iter()
            .any(|indicator| sent2.to_lowercase().contains(indicator));

        // Disruption if there's a clear temporal shift without transition
        (sent1_is_past && sent2_is_future) || (sent1_is_future && sent2_is_past)
    }

    /// Identify temporal chains
    fn identify_temporal_chains(
        &self,
        sentences: &[String],
    ) -> Result<Vec<TemporalChain>, CohesionAnalysisError> {
        let mut chains = Vec::new();

        // Simplified temporal chain identification
        let mut current_chain: Vec<String> = Vec::new();
        let mut current_positions: Vec<usize> = Vec::new();

        for (i, sentence) in sentences.iter().enumerate() {
            if self.calculate_sentence_temporal_score(sentence) > 0.0 {
                current_chain.push(sentence.clone());
                current_positions.push(i);
            } else if !current_chain.is_empty() {
                // End current chain
                let chain = TemporalChain {
                    chain_id: chains.len(),
                    expressions: current_chain.clone(),
                    ordering: current_positions.clone(),
                    consistency_score: 0.7, // Simplified calculation
                };
                chains.push(chain);
                current_chain.clear();
                current_positions.clear();
            }
        }

        // Add final chain if exists
        if !current_chain.is_empty() {
            let chain = TemporalChain {
                chain_id: chains.len(),
                expressions: current_chain,
                ordering: current_positions,
                consistency_score: 0.7,
            };
            chains.push(chain);
        }

        Ok(chains)
    }

    /// Build reference patterns
    fn build_reference_patterns() -> HashSet<String> {
        [
            "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their", "this",
            "that", "these", "those",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Build conjunction patterns
    fn build_conjunction_patterns() -> HashMap<String, String> {
        let mut patterns = HashMap::new();

        // Additive
        patterns.insert("and".to_string(), "additive".to_string());
        patterns.insert("also".to_string(), "additive".to_string());
        patterns.insert("furthermore".to_string(), "additive".to_string());

        // Adversative
        patterns.insert("but".to_string(), "adversative".to_string());
        patterns.insert("however".to_string(), "adversative".to_string());
        patterns.insert("nevertheless".to_string(), "adversative".to_string());

        // Causal
        patterns.insert("therefore".to_string(), "causal".to_string());
        patterns.insert("thus".to_string(), "causal".to_string());
        patterns.insert("consequently".to_string(), "causal".to_string());

        // Temporal
        patterns.insert("then".to_string(), "temporal".to_string());
        patterns.insert("next".to_string(), "temporal".to_string());
        patterns.insert("finally".to_string(), "temporal".to_string());

        patterns
    }

    /// Build temporal markers
    fn build_temporal_markers() -> HashSet<String> {
        [
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
            "first",
            "second",
            "third",
            "finally",
            "next",
            "last",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}

impl Default for ReferenceCohesionMetrics {
    fn default() -> Self {
        Self {
            total_references: 0,
            reference_density: 0.0,
            resolution_success_rate: 0.0,
            average_reference_distance: 0.0,
            ambiguous_references: 0,
            complexity_score: 0.0,
            reference_chains: Vec::new(),
        }
    }
}

impl Default for LexicalCohesionMetrics {
    fn default() -> Self {
        Self {
            lexical_ties: 0,
            lexical_density: 0.0,
            repetition_analysis: RepetitionAnalysis {
                exact_repetitions: 0,
                morphological_variations: 0,
                frequent_terms: Vec::new(),
                distribution_score: 0.0,
            },
            synonym_networks: SynonymNetworkMetrics {
                cluster_count: 0,
                average_cluster_size: 0.0,
                connectivity_score: 0.0,
                major_clusters: Vec::new(),
            },
            semantic_field_coherence: 0.0,
            sophistication_score: 0.0,
        }
    }
}

impl Default for ConjunctiveCohesionMetrics {
    fn default() -> Self {
        Self {
            conjunction_counts: HashMap::new(),
            conjunctive_density: 0.0,
            logical_flow_score: 0.0,
            conjunction_effectiveness: 0.0,
            complex_conjunctions: Vec::new(),
        }
    }
}
