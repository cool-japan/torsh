//! Discourse pattern analysis for structural coherence
//!
//! This module provides comprehensive discourse pattern detection and analysis,
//! including pattern classification, strength calculation, and transition analysis.

use crate::metrics::coherence::structural_coherence::{
    config::{DiscoursePatternConfig, DiscoursePatternType},
    results::{DetectedPattern, DiscoursePatternAnalysis, PatternTransition},
};
use std::collections::HashMap;
use thiserror::Error;

/// Errors specific to discourse pattern analysis
#[derive(Debug, Error)]
pub enum DiscoursePatternError {
    #[error("Invalid discourse pattern configuration: {0}")]
    InvalidConfiguration(String),
    #[error("Pattern detection failed: {0}")]
    PatternDetectionError(String),
    #[error("Insufficient content for pattern analysis")]
    InsufficientContent,
}

/// Discourse pattern analyzer
pub struct DiscoursePatternAnalyzer {
    config: DiscoursePatternConfig,
    pattern_keywords: HashMap<DiscoursePatternType, Vec<String>>,
    pattern_structures: HashMap<DiscoursePatternType, Vec<String>>,
    transition_rules: HashMap<(DiscoursePatternType, DiscoursePatternType), f64>,
}

impl DiscoursePatternAnalyzer {
    /// Create a new discourse pattern analyzer
    pub fn new(config: DiscoursePatternConfig) -> Self {
        Self {
            config,
            pattern_keywords: Self::build_pattern_keywords(),
            pattern_structures: Self::build_pattern_structures(),
            transition_rules: Self::build_transition_rules(),
        }
    }

    /// Analyze discourse patterns in paragraphs
    pub fn analyze_discourse_patterns(
        &self,
        paragraphs: &[String],
    ) -> Result<DiscoursePatternAnalysis, DiscoursePatternError> {
        if !self.config.enable_detection {
            return Ok(DiscoursePatternAnalysis::default());
        }

        if paragraphs.len() < 2 {
            return Err(DiscoursePatternError::InsufficientContent);
        }

        let detected_patterns = self.detect_patterns(paragraphs)?;
        let pattern_distribution = self.calculate_pattern_distribution(&detected_patterns);
        let pattern_coherence_scores = self.calculate_pattern_coherence_scores(&detected_patterns);
        let pattern_transitions = self.analyze_pattern_transitions(&detected_patterns)?;
        let pattern_consistency = self.calculate_pattern_consistency(&detected_patterns);

        Ok(DiscoursePatternAnalysis {
            detected_patterns,
            pattern_distribution,
            pattern_coherence_scores,
            pattern_transitions,
            pattern_consistency,
        })
    }

    /// Detect discourse patterns in paragraphs
    fn detect_patterns(
        &self,
        paragraphs: &[String],
    ) -> Result<Vec<DetectedPattern>, DiscoursePatternError> {
        let mut patterns = Vec::new();

        for (i, paragraph) in paragraphs.iter().enumerate() {
            let pattern_type = self.classify_discourse_pattern(paragraph);
            let strength = self.calculate_pattern_strength(paragraph, &pattern_type);

            if strength >= self.config.min_pattern_strength {
                let completeness = self.calculate_pattern_completeness(paragraph, &pattern_type);
                let evidence = self.collect_pattern_evidence(paragraph, &pattern_type);
                let quality_score = self.calculate_pattern_quality(
                    paragraph,
                    &pattern_type,
                    strength,
                    completeness,
                );

                patterns.push(DetectedPattern {
                    pattern_type,
                    strength,
                    span: (i, i),
                    completeness,
                    evidence,
                    quality_score,
                });
            }
        }

        // Merge adjacent similar patterns
        patterns = self.merge_adjacent_patterns(patterns, paragraphs.len());

        // Limit to maximum patterns per document
        if patterns.len() > self.config.max_patterns_per_document {
            patterns.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap());
            patterns.truncate(self.config.max_patterns_per_document);
        }

        Ok(patterns)
    }

    /// Classify discourse pattern for a paragraph
    fn classify_discourse_pattern(&self, paragraph: &str) -> DiscoursePatternType {
        let paragraph_lower = paragraph.to_lowercase();
        let mut pattern_scores = HashMap::new();

        // Calculate scores for each pattern type
        for (pattern_type, keywords) in &self.pattern_keywords {
            let score = self.calculate_pattern_match_score(&paragraph_lower, keywords);
            pattern_scores.insert(pattern_type.clone(), score);
        }

        // Also check structural indicators
        for (pattern_type, structures) in &self.pattern_structures {
            let structural_score =
                self.calculate_structural_match_score(&paragraph_lower, structures);
            let current_score = pattern_scores.get(pattern_type).unwrap_or(&0.0);
            pattern_scores.insert(pattern_type.clone(), current_score + structural_score * 0.5);
        }

        // Return the pattern with the highest score
        pattern_scores
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(pattern_type, _)| pattern_type)
            .unwrap_or(DiscoursePatternType::Mixed)
    }

    /// Calculate pattern match score based on keywords
    fn calculate_pattern_match_score(&self, paragraph: &str, keywords: &[String]) -> f64 {
        if keywords.is_empty() {
            return 0.0;
        }

        let words: Vec<&str> = paragraph.split_whitespace().collect();
        let word_count = words.len() as f64;

        if word_count == 0.0 {
            return 0.0;
        }

        let mut matches = 0;
        for keyword in keywords {
            if paragraph.contains(keyword) {
                matches += 1;
            }
        }

        (matches as f64 / keywords.len() as f64) * (matches as f64 / word_count).min(1.0)
    }

    /// Calculate structural match score based on structural patterns
    fn calculate_structural_match_score(&self, paragraph: &str, structures: &[String]) -> f64 {
        let mut score = 0.0;

        for structure in structures {
            if paragraph.contains(structure) {
                score += 1.0;
            }
        }

        if !structures.is_empty() {
            score / structures.len() as f64
        } else {
            0.0
        }
    }

    /// Calculate strength of a detected pattern
    fn calculate_pattern_strength(
        &self,
        paragraph: &str,
        pattern_type: &DiscoursePatternType,
    ) -> f64 {
        let keywords = self
            .pattern_keywords
            .get(pattern_type)
            .unwrap_or(&Vec::new());
        let structures = self
            .pattern_structures
            .get(pattern_type)
            .unwrap_or(&Vec::new());

        let keyword_strength =
            self.calculate_pattern_match_score(&paragraph.to_lowercase(), keywords);
        let structural_strength =
            self.calculate_structural_match_score(&paragraph.to_lowercase(), structures);

        // Additional contextual strength factors
        let length_factor = self.calculate_length_appropriateness_factor(paragraph, pattern_type);
        let position_factor = 1.0; // Could be calculated based on paragraph position in document

        ((keyword_strength * 0.4
            + structural_strength * 0.3
            + length_factor * 0.2
            + position_factor * 0.1)
            * self.config.detection_sensitivity)
            .min(1.0)
    }

    /// Calculate pattern completeness
    fn calculate_pattern_completeness(
        &self,
        paragraph: &str,
        pattern_type: &DiscoursePatternType,
    ) -> f64 {
        match pattern_type {
            DiscoursePatternType::ProblemSolution => {
                let has_problem = paragraph.to_lowercase().contains("problem")
                    || paragraph.to_lowercase().contains("issue")
                    || paragraph.to_lowercase().contains("challenge");
                let has_solution = paragraph.to_lowercase().contains("solution")
                    || paragraph.to_lowercase().contains("resolve")
                    || paragraph.to_lowercase().contains("address");

                match (has_problem, has_solution) {
                    (true, true) => 1.0,
                    (true, false) | (false, true) => 0.6,
                    (false, false) => 0.2,
                }
            }
            DiscoursePatternType::CauseEffect => {
                let has_cause = paragraph.to_lowercase().contains("because")
                    || paragraph.to_lowercase().contains("due to")
                    || paragraph.to_lowercase().contains("caused by");
                let has_effect = paragraph.to_lowercase().contains("result")
                    || paragraph.to_lowercase().contains("therefore")
                    || paragraph.to_lowercase().contains("consequently");

                match (has_cause, has_effect) {
                    (true, true) => 1.0,
                    (true, false) | (false, true) => 0.7,
                    (false, false) => 0.3,
                }
            }
            DiscoursePatternType::CompareContrast => {
                let has_compare = paragraph.to_lowercase().contains("similar")
                    || paragraph.to_lowercase().contains("likewise")
                    || paragraph.to_lowercase().contains("compared to");
                let has_contrast = paragraph.to_lowercase().contains("however")
                    || paragraph.to_lowercase().contains("in contrast")
                    || paragraph.to_lowercase().contains("different");

                if has_compare && has_contrast {
                    1.0
                } else if has_compare || has_contrast {
                    0.8
                } else {
                    0.4
                }
            }
            DiscoursePatternType::Chronological => {
                let time_markers = [
                    "first", "then", "next", "finally", "after", "before", "during",
                ];
                let marker_count = time_markers
                    .iter()
                    .filter(|marker| paragraph.to_lowercase().contains(*marker))
                    .count();

                (marker_count as f64 / 3.0).min(1.0)
            }
            DiscoursePatternType::Classification => {
                let classification_markers =
                    ["types", "kinds", "categories", "classified", "divided"];
                let marker_count = classification_markers
                    .iter()
                    .filter(|marker| paragraph.to_lowercase().contains(*marker))
                    .count();

                (marker_count as f64 / 2.0).min(1.0)
            }
            DiscoursePatternType::Definition => {
                let has_definition = paragraph.to_lowercase().contains("defined as")
                    || paragraph.to_lowercase().contains("refers to")
                    || paragraph.to_lowercase().contains("means");

                if has_definition {
                    0.9
                } else {
                    0.5
                }
            }
            DiscoursePatternType::Process => {
                let process_markers = ["step", "stage", "phase", "procedure", "method"];
                let marker_count = process_markers
                    .iter()
                    .filter(|marker| paragraph.to_lowercase().contains(*marker))
                    .count();

                (marker_count as f64 / 2.0).min(1.0)
            }
            _ => 0.5, // Default completeness for other patterns
        }
    }

    /// Calculate length appropriateness factor
    fn calculate_length_appropriateness_factor(
        &self,
        paragraph: &str,
        pattern_type: &DiscoursePatternType,
    ) -> f64 {
        let length = paragraph.len();

        let (ideal_min, ideal_max) = match pattern_type {
            DiscoursePatternType::Definition => (50, 200),
            DiscoursePatternType::ProblemSolution => (100, 500),
            DiscoursePatternType::CauseEffect => (80, 300),
            DiscoursePatternType::CompareContrast => (150, 400),
            DiscoursePatternType::Chronological => (100, 400),
            DiscoursePatternType::Process => (100, 350),
            DiscoursePatternType::Classification => (80, 300),
            _ => (50, 300),
        };

        if length < ideal_min {
            length as f64 / ideal_min as f64
        } else if length > ideal_max {
            ideal_max as f64 / length as f64
        } else {
            1.0
        }
    }

    /// Collect evidence for pattern detection
    fn collect_pattern_evidence(
        &self,
        paragraph: &str,
        pattern_type: &DiscoursePatternType,
    ) -> Vec<String> {
        let mut evidence = Vec::new();
        let paragraph_lower = paragraph.to_lowercase();

        if let Some(keywords) = self.pattern_keywords.get(pattern_type) {
            for keyword in keywords {
                if paragraph_lower.contains(keyword) {
                    evidence.push(keyword.clone());
                }
            }
        }

        if let Some(structures) = self.pattern_structures.get(pattern_type) {
            for structure in structures {
                if paragraph_lower.contains(structure) {
                    evidence.push(structure.clone());
                }
            }
        }

        evidence.truncate(10); // Limit evidence to avoid clutter
        evidence
    }

    /// Calculate pattern quality score
    fn calculate_pattern_quality(
        &self,
        _paragraph: &str,
        _pattern_type: &DiscoursePatternType,
        strength: f64,
        completeness: f64,
    ) -> f64 {
        // Weighted combination of strength and completeness
        strength * 0.6 + completeness * 0.4
    }

    /// Merge adjacent patterns of the same type
    fn merge_adjacent_patterns(
        &self,
        mut patterns: Vec<DetectedPattern>,
        _total_paragraphs: usize,
    ) -> Vec<DetectedPattern> {
        if patterns.len() < 2 {
            return patterns;
        }

        // Sort by position
        patterns.sort_by_key(|p| p.span.0);

        let mut merged = Vec::new();
        let mut current_pattern = patterns.remove(0);

        for next_pattern in patterns {
            if current_pattern.pattern_type == next_pattern.pattern_type
                && next_pattern.span.0 <= current_pattern.span.1 + 2
            {
                // Merge patterns
                current_pattern.span.1 = next_pattern.span.1;
                current_pattern.strength = (current_pattern.strength + next_pattern.strength) / 2.0;
                current_pattern.completeness =
                    current_pattern.completeness.max(next_pattern.completeness);
                current_pattern.quality_score =
                    (current_pattern.quality_score + next_pattern.quality_score) / 2.0;
                current_pattern.evidence.extend(next_pattern.evidence);
                current_pattern.evidence.sort();
                current_pattern.evidence.dedup();
            } else {
                merged.push(current_pattern);
                current_pattern = next_pattern;
            }
        }
        merged.push(current_pattern);

        merged
    }

    /// Calculate pattern distribution across document
    fn calculate_pattern_distribution(&self, patterns: &[DetectedPattern]) -> HashMap<String, f64> {
        let mut distribution = HashMap::new();
        let total_patterns = patterns.len() as f64;

        if total_patterns == 0.0 {
            return distribution;
        }

        for pattern in patterns {
            let pattern_name = format!("{:?}", pattern.pattern_type);
            *distribution.entry(pattern_name).or_insert(0.0) += 1.0;
        }

        // Normalize to percentages
        for value in distribution.values_mut() {
            *value /= total_patterns;
        }

        distribution
    }

    /// Calculate coherence scores for each pattern type
    fn calculate_pattern_coherence_scores(
        &self,
        patterns: &[DetectedPattern],
    ) -> HashMap<String, f64> {
        let mut scores = HashMap::new();

        for pattern in patterns {
            let pattern_name = format!("{:?}", pattern.pattern_type);
            let entry = scores.entry(pattern_name).or_insert_with(|| (0.0, 0));
            entry.0 += pattern.quality_score;
            entry.1 += 1;
        }

        // Convert to averages
        scores
            .into_iter()
            .map(|(name, (total_score, count))| {
                let avg_score = if count > 0 {
                    total_score / count as f64
                } else {
                    0.0
                };
                (name, avg_score)
            })
            .collect()
    }

    /// Analyze transitions between patterns
    fn analyze_pattern_transitions(
        &self,
        patterns: &[DetectedPattern],
    ) -> Result<Vec<PatternTransition>, DiscoursePatternError> {
        if !self.config.analyze_transitions {
            return Ok(Vec::new());
        }

        let mut transitions = Vec::new();

        for window in patterns.windows(2) {
            let from_pattern = &window[0].pattern_type;
            let to_pattern = &window[1].pattern_type;
            let position = window[1].span.0;

            let smoothness = self.calculate_pattern_transition_smoothness(from_pattern, to_pattern);
            let appropriateness =
                self.calculate_pattern_transition_appropriateness(from_pattern, to_pattern);

            transitions.push(PatternTransition {
                from_pattern: from_pattern.clone(),
                to_pattern: to_pattern.clone(),
                position,
                smoothness,
                appropriateness,
            });
        }

        Ok(transitions)
    }

    /// Calculate smoothness of pattern transition
    fn calculate_pattern_transition_smoothness(
        &self,
        from_pattern: &DiscoursePatternType,
        to_pattern: &DiscoursePatternType,
    ) -> f64 {
        if from_pattern == to_pattern {
            return 1.0; // Same pattern is perfectly smooth
        }

        self.transition_rules
            .get(&(from_pattern.clone(), to_pattern.clone()))
            .copied()
            .unwrap_or(0.5) // Default smoothness for undefined transitions
    }

    /// Calculate appropriateness of pattern transition
    fn calculate_pattern_transition_appropriateness(
        &self,
        from_pattern: &DiscoursePatternType,
        to_pattern: &DiscoursePatternType,
    ) -> f64 {
        // Some transitions are more appropriate than others in academic/technical writing
        match (from_pattern, to_pattern) {
            (DiscoursePatternType::Definition, DiscoursePatternType::Process) => 0.9,
            (DiscoursePatternType::ProblemSolution, DiscoursePatternType::CauseEffect) => 0.8,
            (DiscoursePatternType::CompareContrast, DiscoursePatternType::Classification) => 0.8,
            (DiscoursePatternType::Chronological, DiscoursePatternType::Process) => 0.9,
            (DiscoursePatternType::Classification, DiscoursePatternType::CompareContrast) => 0.7,
            (DiscoursePatternType::Definition, DiscoursePatternType::CompareContrast) => 0.7,
            _ => 0.6, // Default appropriateness
        }
    }

    /// Calculate overall pattern consistency
    fn calculate_pattern_consistency(&self, patterns: &[DetectedPattern]) -> f64 {
        if patterns.len() < 2 {
            return 1.0;
        }

        let mut consistency_sum = 0.0;
        let mut total_patterns = 0;

        // Group patterns by type and calculate internal consistency
        let mut pattern_groups: HashMap<DiscoursePatternType, Vec<&DetectedPattern>> =
            HashMap::new();

        for pattern in patterns {
            pattern_groups
                .entry(pattern.pattern_type.clone())
                .or_insert_with(Vec::new)
                .push(pattern);
        }

        for group in pattern_groups.values() {
            if group.len() > 1 {
                let group_consistency = self.calculate_group_consistency(group);
                consistency_sum += group_consistency * group.len() as f64;
                total_patterns += group.len();
            }
        }

        if total_patterns > 0 {
            consistency_sum / total_patterns as f64
        } else {
            1.0
        }
    }

    /// Calculate consistency within a group of same-type patterns
    fn calculate_group_consistency(&self, patterns: &[&DetectedPattern]) -> f64 {
        if patterns.len() < 2 {
            return 1.0;
        }

        // Calculate variance in quality scores
        let scores: Vec<f64> = patterns.iter().map(|p| p.quality_score).collect();
        let mean_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;

        let variance = scores
            .iter()
            .map(|score| (score - mean_score).powi(2))
            .sum::<f64>()
            / scores.len() as f64;

        // Convert variance to consistency (lower variance = higher consistency)
        let std_dev = variance.sqrt();
        (1.0 - std_dev).max(0.0)
    }

    /// Build pattern keywords mapping
    fn build_pattern_keywords() -> HashMap<DiscoursePatternType, Vec<String>> {
        let mut keywords = HashMap::new();

        keywords.insert(
            DiscoursePatternType::ProblemSolution,
            vec![
                "problem".to_string(),
                "issue".to_string(),
                "challenge".to_string(),
                "solution".to_string(),
                "resolve".to_string(),
                "address".to_string(),
                "fix".to_string(),
                "solve".to_string(),
                "overcome".to_string(),
            ],
        );

        keywords.insert(
            DiscoursePatternType::CauseEffect,
            vec![
                "because".to_string(),
                "due to".to_string(),
                "caused by".to_string(),
                "result".to_string(),
                "therefore".to_string(),
                "consequently".to_string(),
                "leads to".to_string(),
                "results in".to_string(),
                "effect".to_string(),
            ],
        );

        keywords.insert(
            DiscoursePatternType::CompareContrast,
            vec![
                "similar".to_string(),
                "likewise".to_string(),
                "compared to".to_string(),
                "however".to_string(),
                "in contrast".to_string(),
                "different".to_string(),
                "whereas".to_string(),
                "on the other hand".to_string(),
                "although".to_string(),
            ],
        );

        keywords.insert(
            DiscoursePatternType::Chronological,
            vec![
                "first".to_string(),
                "then".to_string(),
                "next".to_string(),
                "finally".to_string(),
                "after".to_string(),
                "before".to_string(),
                "during".to_string(),
                "subsequently".to_string(),
                "previously".to_string(),
            ],
        );

        keywords.insert(
            DiscoursePatternType::Spatial,
            vec![
                "above".to_string(),
                "below".to_string(),
                "near".to_string(),
                "far".to_string(),
                "left".to_string(),
                "right".to_string(),
                "adjacent".to_string(),
                "opposite".to_string(),
                "parallel".to_string(),
            ],
        );

        keywords.insert(
            DiscoursePatternType::Classification,
            vec![
                "types".to_string(),
                "kinds".to_string(),
                "categories".to_string(),
                "classified".to_string(),
                "divided".to_string(),
                "grouped".to_string(),
                "class".to_string(),
                "category".to_string(),
                "type".to_string(),
            ],
        );

        keywords.insert(
            DiscoursePatternType::Definition,
            vec![
                "defined as".to_string(),
                "refers to".to_string(),
                "means".to_string(),
                "is".to_string(),
                "definition".to_string(),
                "term".to_string(),
                "concept".to_string(),
                "characterized by".to_string(),
            ],
        );

        keywords.insert(
            DiscoursePatternType::Process,
            vec![
                "step".to_string(),
                "stage".to_string(),
                "phase".to_string(),
                "procedure".to_string(),
                "method".to_string(),
                "process".to_string(),
                "technique".to_string(),
                "approach".to_string(),
                "methodology".to_string(),
            ],
        );

        keywords
    }

    /// Build pattern structures mapping
    fn build_pattern_structures() -> HashMap<DiscoursePatternType, Vec<String>> {
        let mut structures = HashMap::new();

        structures.insert(
            DiscoursePatternType::ProblemSolution,
            vec![
                "the problem is".to_string(),
                "one solution is".to_string(),
                "to solve this".to_string(),
            ],
        );

        structures.insert(
            DiscoursePatternType::CauseEffect,
            vec![
                "as a result of".to_string(),
                "the effect of".to_string(),
                "this leads to".to_string(),
            ],
        );

        structures.insert(
            DiscoursePatternType::CompareContrast,
            vec![
                "in comparison".to_string(),
                "on the contrary".to_string(),
                "similar to".to_string(),
            ],
        );

        structures.insert(
            DiscoursePatternType::Chronological,
            vec![
                "the first step".to_string(),
                "the next phase".to_string(),
                "in the end".to_string(),
            ],
        );

        structures.insert(
            DiscoursePatternType::Definition,
            vec![
                "can be defined as".to_string(),
                "the definition of".to_string(),
                "this means that".to_string(),
            ],
        );

        structures
    }

    /// Build transition rules for pattern smoothness
    fn build_transition_rules() -> HashMap<(DiscoursePatternType, DiscoursePatternType), f64> {
        let mut rules = HashMap::new();

        // High-quality transitions
        rules.insert(
            (
                DiscoursePatternType::Definition,
                DiscoursePatternType::Process,
            ),
            0.9,
        );
        rules.insert(
            (
                DiscoursePatternType::ProblemSolution,
                DiscoursePatternType::CauseEffect,
            ),
            0.9,
        );
        rules.insert(
            (
                DiscoursePatternType::Classification,
                DiscoursePatternType::CompareContrast,
            ),
            0.8,
        );
        rules.insert(
            (
                DiscoursePatternType::Chronological,
                DiscoursePatternType::Process,
            ),
            0.8,
        );

        // Medium-quality transitions
        rules.insert(
            (
                DiscoursePatternType::Definition,
                DiscoursePatternType::Classification,
            ),
            0.7,
        );
        rules.insert(
            (
                DiscoursePatternType::CompareContrast,
                DiscoursePatternType::Classification,
            ),
            0.7,
        );
        rules.insert(
            (
                DiscoursePatternType::CauseEffect,
                DiscoursePatternType::ProblemSolution,
            ),
            0.6,
        );

        // Same pattern transitions
        for pattern in [
            DiscoursePatternType::ProblemSolution,
            DiscoursePatternType::CauseEffect,
            DiscoursePatternType::CompareContrast,
            DiscoursePatternType::Chronological,
            DiscoursePatternType::Classification,
            DiscoursePatternType::Definition,
            DiscoursePatternType::Process,
        ] {
            rules.insert((pattern.clone(), pattern), 1.0);
        }

        rules
    }
}
