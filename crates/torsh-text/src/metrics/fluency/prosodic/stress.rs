//! Stress Analysis Module
//!
//! This module provides comprehensive stress pattern analysis for prosodic fluency,
//! including metrical structure analysis, prominence detection, stress placement
//! evaluation, and prosodic foot analysis with sophisticated pattern recognition.

use super::config::StressAnalysisConfig;
use super::results::{
    AccentPattern, AccentType, ComplexFootPattern, ContrastiveStress, FocusMarker, FocusType,
    FootAnalysis, FootType, MetricalFoot, MetricalStructure, MetricalViolation, ProminenceAnalysis,
    ProminenceLevel, ProminenceType, StressClash, StressMetrics, TonalMovement, TonalProperties,
};
use crate::error::TextAnalysisError;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StressAnalysisError {
    #[error("Invalid stress analysis configuration: {0}")]
    ConfigError(String),
    #[error("Stress calculation failed: {0}")]
    CalculationError(String),
    #[error("Metrical analysis error: {0}")]
    MetricalError(String),
    #[error("Prominence analysis error: {0}")]
    ProminenceError(String),
}

pub type StressResult<T> = Result<T, StressAnalysisError>;

/// Stress pattern representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressPattern {
    /// Syllable positions
    pub positions: Vec<usize>,
    /// Stress levels (0.0 to 1.0)
    pub levels: Vec<f64>,
    /// Stress types at each position
    pub types: Vec<StressType>,
    /// Pattern confidence
    pub confidence: f64,
    /// Pattern context
    pub context: String,
}

/// Types of prosodic stress
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StressType {
    /// Primary word stress
    Primary,
    /// Secondary word stress
    Secondary,
    /// Tertiary stress
    Tertiary,
    /// Unstressed syllable
    Unstressed,
    /// Reduced vowel stress
    Reduced,
}

/// Metrical foot analyzer for prosodic structure
#[derive(Debug, Clone)]
pub struct MetricalFootAnalyzer {
    /// Configuration settings
    config: StressAnalysisConfig,
    /// Foot identification rules
    foot_rules: Vec<FootRule>,
    /// Foot type probabilities
    type_probabilities: HashMap<FootType, f64>,
    /// Boundary detection parameters
    boundary_params: BoundaryDetectionParams,
}

#[derive(Debug, Clone)]
pub struct FootRule {
    /// Rule name
    pub name: String,
    /// Stress pattern for foot type
    pub pattern: Vec<StressType>,
    /// Rule confidence weight
    pub weight: f64,
    /// Context requirements
    pub context_requirements: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BoundaryDetectionParams {
    /// Minimum foot length
    pub min_foot_length: usize,
    /// Maximum foot length
    pub max_foot_length: usize,
    /// Boundary strength threshold
    pub boundary_threshold: f64,
    /// Context window size
    pub context_window: usize,
}

impl Default for BoundaryDetectionParams {
    fn default() -> Self {
        Self {
            min_foot_length: 1,
            max_foot_length: 4,
            boundary_threshold: 0.6,
            context_window: 3,
        }
    }
}

impl MetricalFootAnalyzer {
    /// Create new metrical foot analyzer
    pub fn new(config: StressAnalysisConfig) -> Self {
        let foot_rules = Self::create_foot_rules();
        let type_probabilities = Self::calculate_type_probabilities();

        Self {
            config,
            foot_rules,
            type_probabilities,
            boundary_params: BoundaryDetectionParams::default(),
        }
    }

    /// Create foot identification rules
    fn create_foot_rules() -> Vec<FootRule> {
        vec![
            FootRule {
                name: "trochee".to_string(),
                pattern: vec![StressType::Primary, StressType::Unstressed],
                weight: 1.0,
                context_requirements: vec![],
            },
            FootRule {
                name: "iamb".to_string(),
                pattern: vec![StressType::Unstressed, StressType::Primary],
                weight: 1.0,
                context_requirements: vec![],
            },
            FootRule {
                name: "dactyl".to_string(),
                pattern: vec![
                    StressType::Primary,
                    StressType::Unstressed,
                    StressType::Unstressed,
                ],
                weight: 0.8,
                context_requirements: vec![],
            },
            FootRule {
                name: "anapest".to_string(),
                pattern: vec![
                    StressType::Unstressed,
                    StressType::Unstressed,
                    StressType::Primary,
                ],
                weight: 0.8,
                context_requirements: vec![],
            },
            FootRule {
                name: "spondee".to_string(),
                pattern: vec![StressType::Primary, StressType::Primary],
                weight: 0.6,
                context_requirements: vec!["emphatic".to_string()],
            },
            FootRule {
                name: "pyrrhic".to_string(),
                pattern: vec![StressType::Unstressed, StressType::Unstressed],
                weight: 0.4,
                context_requirements: vec!["function_words".to_string()],
            },
        ]
    }

    /// Calculate foot type probabilities based on language characteristics
    fn calculate_type_probabilities() -> HashMap<FootType, f64> {
        let mut probabilities = HashMap::new();

        // English typically favors trochaic patterns
        probabilities.insert(FootType::Trochee, 0.35);
        probabilities.insert(FootType::Iamb, 0.30);
        probabilities.insert(FootType::Dactyl, 0.15);
        probabilities.insert(FootType::Anapest, 0.10);
        probabilities.insert(FootType::Spondee, 0.05);
        probabilities.insert(FootType::Pyrrhic, 0.05);

        probabilities
    }

    /// Analyze metrical structure of text
    pub fn analyze_metrical_structure(
        &self,
        syllables: &[String],
        stress_patterns: &[StressPattern],
    ) -> StressResult<MetricalStructure> {
        let metrical_feet = self.identify_metrical_feet(syllables, stress_patterns)?;
        let consistency_score = self.calculate_consistency(&metrical_feet);
        let dominant_foot_type = self.determine_dominant_foot_type(&metrical_feet);
        let pattern_regularity = self.calculate_pattern_regularity(&metrical_feet);
        let violations = self.detect_metrical_violations(&metrical_feet, syllables)?;

        Ok(MetricalStructure {
            metrical_feet,
            consistency_score,
            dominant_foot_type,
            pattern_regularity,
            violations,
        })
    }

    /// Identify metrical feet in syllable sequence
    fn identify_metrical_feet(
        &self,
        syllables: &[String],
        stress_patterns: &[StressPattern],
    ) -> StressResult<Vec<MetricalFoot>> {
        let mut feet = Vec::new();
        let mut current_position = 0;

        // Combine all stress information
        let combined_stress = self.combine_stress_patterns(stress_patterns)?;

        while current_position < syllables.len() {
            if let Some(foot) =
                self.identify_foot_at_position(current_position, syllables, &combined_stress)?
            {
                current_position = foot.boundaries.1;
                feet.push(foot);
            } else {
                current_position += 1;
            }
        }

        Ok(feet)
    }

    /// Combine multiple stress patterns into unified representation
    fn combine_stress_patterns(
        &self,
        patterns: &[StressPattern],
    ) -> StressResult<Vec<(StressType, f64)>> {
        let mut combined = Vec::new();

        if patterns.is_empty() {
            return Ok(combined);
        }

        // Find maximum position to determine sequence length
        let max_position = patterns
            .iter()
            .flat_map(|p| p.positions.iter())
            .max()
            .copied()
            .unwrap_or(0);

        // Initialize with unstressed
        for _ in 0..=max_position {
            combined.push((StressType::Unstressed, 0.0));
        }

        // Overlay patterns with confidence weighting
        for pattern in patterns {
            for (i, &position) in pattern.positions.iter().enumerate() {
                if position < combined.len() && i < pattern.levels.len() && i < pattern.types.len()
                {
                    let current_strength = combined[position].1;
                    let pattern_strength = pattern.levels[i] * pattern.confidence;

                    if pattern_strength > current_strength {
                        combined[position] = (pattern.types[i].clone(), pattern_strength);
                    }
                }
            }
        }

        Ok(combined)
    }

    /// Identify foot at specific position
    fn identify_foot_at_position(
        &self,
        position: usize,
        syllables: &[String],
        stress_info: &[(StressType, f64)],
    ) -> StressResult<Option<MetricalFoot>> {
        let mut best_match: Option<MetricalFoot> = None;
        let mut best_score = 0.0;

        // Try different foot lengths
        for foot_length in
            self.boundary_params.min_foot_length..=self.boundary_params.max_foot_length
        {
            if position + foot_length <= syllables.len() {
                if let Some(foot) =
                    self.try_foot_match(position, foot_length, syllables, stress_info)?
                {
                    if foot.strength > best_score {
                        best_score = foot.strength;
                        best_match = Some(foot);
                    }
                }
            }
        }

        Ok(best_match)
    }

    /// Try to match foot pattern at position
    fn try_foot_match(
        &self,
        start: usize,
        length: usize,
        syllables: &[String],
        stress_info: &[(StressType, f64)],
    ) -> StressResult<Option<MetricalFoot>> {
        if start + length > syllables.len() || start + length > stress_info.len() {
            return Ok(None);
        }

        let foot_syllables: Vec<String> = syllables[start..start + length].to_vec();
        let foot_stress_types: Vec<StressType> = stress_info[start..start + length]
            .iter()
            .map(|(stress_type, _)| stress_type.clone())
            .collect();
        let foot_stress_pattern: Vec<bool> = foot_stress_types
            .iter()
            .map(|stress_type| !matches!(stress_type, StressType::Unstressed | StressType::Reduced))
            .collect();

        // Find best matching foot rule
        let mut best_rule: Option<&FootRule> = None;
        let mut best_match_score = 0.0;

        for rule in &self.foot_rules {
            if rule.pattern.len() == length {
                let match_score = self.calculate_rule_match_score(rule, &foot_stress_types);
                if match_score > best_match_score {
                    best_match_score = match_score;
                    best_rule = Some(rule);
                }
            }
        }

        if let Some(rule) = best_rule {
            if best_match_score >= self.config.foot_boundary_accuracy {
                let foot_type = self.rule_to_foot_type(rule);
                let strength = best_match_score * rule.weight;

                return Ok(Some(MetricalFoot {
                    foot_type,
                    syllables: foot_syllables,
                    stress_pattern: foot_stress_pattern,
                    boundaries: (start, start + length),
                    strength,
                }));
            }
        }

        Ok(None)
    }

    /// Calculate rule match score
    fn calculate_rule_match_score(&self, rule: &FootRule, stress_types: &[StressType]) -> f64 {
        if rule.pattern.len() != stress_types.len() {
            return 0.0;
        }

        let mut matches = 0;
        let total = rule.pattern.len();

        for (expected, actual) in rule.pattern.iter().zip(stress_types.iter()) {
            if self.stress_types_compatible(expected, actual) {
                matches += 1;
            }
        }

        matches as f64 / total as f64
    }

    /// Check if stress types are compatible
    fn stress_types_compatible(&self, expected: &StressType, actual: &StressType) -> bool {
        match (expected, actual) {
            (StressType::Primary, StressType::Primary) => true,
            (StressType::Primary, StressType::Secondary) => true, // Allow secondary for primary
            (StressType::Secondary, StressType::Secondary) => true,
            (StressType::Secondary, StressType::Tertiary) => true,
            (StressType::Unstressed, StressType::Unstressed) => true,
            (StressType::Unstressed, StressType::Reduced) => true,
            _ => false,
        }
    }

    /// Convert rule to foot type
    fn rule_to_foot_type(&self, rule: &FootRule) -> FootType {
        match rule.name.as_str() {
            "trochee" => FootType::Trochee,
            "iamb" => FootType::Iamb,
            "dactyl" => FootType::Dactyl,
            "anapest" => FootType::Anapest,
            "spondee" => FootType::Spondee,
            "pyrrhic" => FootType::Pyrrhic,
            _ => FootType::Trochee, // Default
        }
    }

    /// Calculate metrical consistency
    fn calculate_consistency(&self, feet: &[MetricalFoot]) -> f64 {
        if feet.len() < 2 {
            return 1.0;
        }

        // Count foot types
        let mut type_counts: HashMap<FootType, usize> = HashMap::new();
        for foot in feet {
            *type_counts.entry(foot.foot_type.clone()).or_insert(0) += 1;
        }

        // Calculate entropy (lower entropy = more consistent)
        let total = feet.len() as f64;
        let entropy: f64 = type_counts
            .values()
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.ln()
            })
            .sum();

        let max_entropy = (type_counts.len() as f64).ln();
        if max_entropy > 0.0 {
            1.0 - (entropy / max_entropy)
        } else {
            1.0
        }
    }

    /// Determine dominant foot type
    fn determine_dominant_foot_type(&self, feet: &[MetricalFoot]) -> FootType {
        let mut type_counts: HashMap<FootType, usize> = HashMap::new();

        for foot in feet {
            *type_counts.entry(foot.foot_type.clone()).or_insert(0) += 1;
        }

        type_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(foot_type, _)| foot_type)
            .unwrap_or(FootType::Trochee)
    }

    /// Calculate pattern regularity
    fn calculate_pattern_regularity(&self, feet: &[MetricalFoot]) -> f64 {
        if feet.len() < 2 {
            return 1.0;
        }

        // Check for consistent foot lengths
        let lengths: Vec<usize> = feet.iter().map(|foot| foot.syllables.len()).collect();
        let mean_length = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
        let length_variance = lengths
            .iter()
            .map(|&len| (len as f64 - mean_length).powi(2))
            .sum::<f64>()
            / lengths.len() as f64;

        let length_regularity = 1.0 / (1.0 + length_variance.sqrt());

        // Check for consistent stress patterns within same foot type
        let mut type_pattern_consistency = HashMap::new();

        for foot in feet {
            let patterns = type_pattern_consistency
                .entry(foot.foot_type.clone())
                .or_insert_with(Vec::new);
            patterns.push(&foot.stress_pattern);
        }

        let mut pattern_consistencies = Vec::new();
        for (_, patterns) in type_pattern_consistency {
            if patterns.len() > 1 {
                let consistency = self.calculate_pattern_group_consistency(&patterns);
                pattern_consistencies.push(consistency);
            }
        }

        let pattern_regularity = if pattern_consistencies.is_empty() {
            1.0
        } else {
            pattern_consistencies.iter().sum::<f64>() / pattern_consistencies.len() as f64
        };

        (length_regularity + pattern_regularity) / 2.0
    }

    /// Calculate consistency within pattern group
    fn calculate_pattern_group_consistency(&self, patterns: &[&Vec<bool>]) -> f64 {
        if patterns.len() < 2 {
            return 1.0;
        }

        let reference = patterns[0];
        let mut total_similarity = 0.0;

        for pattern in patterns.iter().skip(1) {
            let similarity = self.calculate_pattern_similarity(reference, pattern);
            total_similarity += similarity;
        }

        total_similarity / (patterns.len() - 1) as f64
    }

    /// Calculate similarity between two stress patterns
    fn calculate_pattern_similarity(&self, pattern1: &[bool], pattern2: &[bool]) -> f64 {
        let min_len = pattern1.len().min(pattern2.len());
        if min_len == 0 {
            return 0.0;
        }

        let matches = pattern1
            .iter()
            .zip(pattern2.iter())
            .take(min_len)
            .filter(|(a, b)| a == b)
            .count();

        matches as f64 / min_len as f64
    }

    /// Detect metrical violations
    fn detect_metrical_violations(
        &self,
        feet: &[MetricalFoot],
        syllables: &[String],
    ) -> StressResult<Vec<MetricalViolation>> {
        let mut violations = Vec::new();

        // Check for stress clashes
        violations.extend(self.detect_stress_clashes(feet)?);

        // Check for lapse violations (too many unstressed syllables)
        violations.extend(self.detect_lapse_violations(feet, syllables)?);

        // Check for foot boundary violations
        violations.extend(self.detect_boundary_violations(feet)?);

        Ok(violations)
    }

    /// Detect stress clash violations
    fn detect_stress_clashes(&self, feet: &[MetricalFoot]) -> StressResult<Vec<MetricalViolation>> {
        let mut violations = Vec::new();

        for i in 1..feet.len() {
            let prev_foot = &feet[i - 1];
            let curr_foot = &feet[i];

            // Check if last syllable of previous foot and first syllable of current foot are both stressed
            if let (Some(&prev_last_stress), Some(&curr_first_stress)) = (
                prev_foot.stress_pattern.last(),
                curr_foot.stress_pattern.first(),
            ) {
                if prev_last_stress && curr_first_stress {
                    violations.push(MetricalViolation {
                        violation_type: "stress_clash".to_string(),
                        position: prev_foot.boundaries.1,
                        severity: 0.8,
                        description: "Adjacent stressed syllables create stress clash".to_string(),
                        correction: Some("Consider stress shift or pause insertion".to_string()),
                    });
                }
            }
        }

        Ok(violations)
    }

    /// Detect lapse violations (too many unstressed syllables)
    fn detect_lapse_violations(
        &self,
        feet: &[MetricalFoot],
        syllables: &[String],
    ) -> StressResult<Vec<MetricalViolation>> {
        let mut violations = Vec::new();
        let max_lapse_length = 3; // Maximum consecutive unstressed syllables

        // Create continuous stress pattern
        let mut continuous_stress = Vec::new();
        for foot in feet {
            continuous_stress.extend(&foot.stress_pattern);
        }

        let mut unstressed_count = 0;
        let mut lapse_start = 0;

        for (i, &is_stressed) in continuous_stress.iter().enumerate() {
            if is_stressed {
                if unstressed_count > max_lapse_length {
                    violations.push(MetricalViolation {
                        violation_type: "lapse".to_string(),
                        position: lapse_start,
                        severity: 0.6,
                        description: format!("Lapse of {} unstressed syllables", unstressed_count),
                        correction: Some("Consider adding secondary stress".to_string()),
                    });
                }
                unstressed_count = 0;
            } else {
                if unstressed_count == 0 {
                    lapse_start = i;
                }
                unstressed_count += 1;
            }
        }

        Ok(violations)
    }

    /// Detect foot boundary violations
    fn detect_boundary_violations(
        &self,
        feet: &[MetricalFoot],
    ) -> StressResult<Vec<MetricalViolation>> {
        let mut violations = Vec::new();

        for foot in feet {
            // Check if foot has at least one stressed syllable
            if !foot.stress_pattern.iter().any(|&stressed| stressed) {
                violations.push(MetricalViolation {
                    violation_type: "headless_foot".to_string(),
                    position: foot.boundaries.0,
                    severity: 0.7,
                    description: "Foot contains no stressed syllables".to_string(),
                    correction: Some("Restructure foot boundaries or add stress".to_string()),
                });
            }

            // Check foot length constraints
            if foot.syllables.len() > self.boundary_params.max_foot_length {
                violations.push(MetricalViolation {
                    violation_type: "oversized_foot".to_string(),
                    position: foot.boundaries.0,
                    severity: 0.5,
                    description: format!(
                        "Foot exceeds maximum length of {}",
                        self.boundary_params.max_foot_length
                    ),
                    correction: Some("Consider splitting into multiple feet".to_string()),
                });
            }
        }

        Ok(violations)
    }
}

/// Prominence analyzer for detecting stress prominence
#[derive(Debug, Clone)]
pub struct ProminenceAnalyzer {
    /// Configuration settings
    config: StressAnalysisConfig,
    /// Prominence detection thresholds
    thresholds: ProminenceThresholds,
    /// Contextual analysis parameters
    context_params: ContextAnalysisParams,
}

#[derive(Debug, Clone)]
pub struct ProminenceThresholds {
    /// Primary prominence threshold
    pub primary_threshold: f64,
    /// Secondary prominence threshold
    pub secondary_threshold: f64,
    /// Focus prominence threshold
    pub focus_threshold: f64,
    /// Contrastive prominence threshold
    pub contrastive_threshold: f64,
}

impl Default for ProminenceThresholds {
    fn default() -> Self {
        Self {
            primary_threshold: 0.8,
            secondary_threshold: 0.6,
            focus_threshold: 0.9,
            contrastive_threshold: 0.85,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextAnalysisParams {
    /// Context window size for analysis
    pub window_size: usize,
    /// Focus detection parameters
    pub focus_detection_sensitivity: f64,
    /// Contrast analysis depth
    pub contrast_analysis_depth: usize,
}

impl Default for ContextAnalysisParams {
    fn default() -> Self {
        Self {
            window_size: 5,
            focus_detection_sensitivity: 0.7,
            contrast_analysis_depth: 3,
        }
    }
}

impl ProminenceAnalyzer {
    /// Create new prominence analyzer
    pub fn new(config: StressAnalysisConfig) -> Self {
        Self {
            config,
            thresholds: ProminenceThresholds::default(),
            context_params: ContextAnalysisParams::default(),
        }
    }

    /// Analyze prominence patterns in text
    pub fn analyze_prominence(
        &self,
        syllables: &[String],
        stress_levels: &[f64],
        contextual_info: Option<&HashMap<String, String>>,
    ) -> StressResult<ProminenceAnalysis> {
        let primary_positions = self.identify_primary_stress(stress_levels)?;
        let secondary_positions = self.identify_secondary_stress(stress_levels)?;
        let prominence_hierarchy = self.build_prominence_hierarchy(syllables, stress_levels)?;
        let focus_prominence = if self.config.enable_prominence_analysis {
            self.detect_focus_prominence(syllables, stress_levels, contextual_info)?
        } else {
            Vec::new()
        };
        let contrastive_stress = if self.config.detect_stress_clashes {
            self.detect_contrastive_stress(syllables, stress_levels)?
        } else {
            Vec::new()
        };

        Ok(ProminenceAnalysis {
            primary_stress_positions: primary_positions,
            secondary_stress_positions: secondary_positions,
            prominence_hierarchy,
            focus_prominence,
            contrastive_stress,
        })
    }

    /// Identify primary stress positions
    fn identify_primary_stress(&self, stress_levels: &[f64]) -> StressResult<Vec<usize>> {
        let mut primary_positions = Vec::new();

        for (i, &level) in stress_levels.iter().enumerate() {
            if level >= self.thresholds.primary_threshold {
                primary_positions.push(i);
            }
        }

        Ok(primary_positions)
    }

    /// Identify secondary stress positions
    fn identify_secondary_stress(&self, stress_levels: &[f64]) -> StressResult<Vec<usize>> {
        let mut secondary_positions = Vec::new();

        for (i, &level) in stress_levels.iter().enumerate() {
            if level >= self.thresholds.secondary_threshold
                && level < self.thresholds.primary_threshold
            {
                secondary_positions.push(i);
            }
        }

        Ok(secondary_positions)
    }

    /// Build prominence hierarchy
    fn build_prominence_hierarchy(
        &self,
        syllables: &[String],
        stress_levels: &[f64],
    ) -> StressResult<Vec<ProminenceLevel>> {
        let mut hierarchy = Vec::new();

        for (i, (&level, syllable)) in stress_levels.iter().zip(syllables.iter()).enumerate() {
            if level > 0.1 {
                // Only include syllables with some prominence
                let prominence_type = self.classify_prominence_type(level);
                let factors = self.analyze_prominence_factors(i, syllables, stress_levels);

                hierarchy.push(ProminenceLevel {
                    position: i,
                    strength: level,
                    prominence_type,
                    factors,
                });
            }
        }

        // Sort by prominence strength
        hierarchy.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());

        Ok(hierarchy)
    }

    /// Classify prominence type based on strength
    fn classify_prominence_type(&self, strength: f64) -> ProminenceType {
        if strength >= self.thresholds.primary_threshold {
            ProminenceType::Primary
        } else if strength >= self.thresholds.secondary_threshold {
            ProminenceType::Secondary
        } else if strength >= 0.4 {
            ProminenceType::Phrasal
        } else {
            ProminenceType::Contrastive
        }
    }

    /// Analyze factors contributing to prominence
    fn analyze_prominence_factors(
        &self,
        position: usize,
        syllables: &[String],
        stress_levels: &[f64],
    ) -> Vec<String> {
        let mut factors = Vec::new();

        // Check if it's a peak in local context
        let window_start = position.saturating_sub(self.context_params.window_size / 2);
        let window_end =
            (position + self.context_params.window_size / 2 + 1).min(stress_levels.len());

        let local_max = stress_levels[window_start..window_end]
            .iter()
            .fold(0.0, |max, &x| max.max(x));

        if stress_levels[position] == local_max && local_max > 0.5 {
            factors.push("local_peak".to_string());
        }

        // Check syllable characteristics
        if let Some(syllable) = syllables.get(position) {
            if syllable.len() > 3 {
                factors.push("long_syllable".to_string());
            }
            if syllable.chars().any(|c| "aeiouAEIOU".contains(c)) {
                factors.push("vowel_prominence".to_string());
            }
        }

        // Check positional factors
        if position == 0 {
            factors.push("initial_position".to_string());
        } else if position == syllables.len() - 1 {
            factors.push("final_position".to_string());
        }

        factors
    }

    /// Detect focus prominence markers
    fn detect_focus_prominence(
        &self,
        syllables: &[String],
        stress_levels: &[f64],
        contextual_info: Option<&HashMap<String, String>>,
    ) -> StressResult<Vec<FocusMarker>> {
        let mut focus_markers = Vec::new();

        // Look for unusually high prominence that might indicate focus
        for (i, &level) in stress_levels.iter().enumerate() {
            if level >= self.thresholds.focus_threshold {
                let focus_type =
                    self.classify_focus_type(i, syllables, stress_levels, contextual_info);
                let strength = level;
                let scope = self.determine_focus_scope(i, syllables, stress_levels);

                focus_markers.push(FocusMarker {
                    position: i,
                    focus_type,
                    strength,
                    scope,
                });
            }
        }

        Ok(focus_markers)
    }

    /// Classify type of focus
    fn classify_focus_type(
        &self,
        position: usize,
        syllables: &[String],
        stress_levels: &[f64],
        contextual_info: Option<&HashMap<String, String>>,
    ) -> FocusType {
        // Simple heuristic classification - would be enhanced with syntactic/semantic analysis
        if let Some(context) = contextual_info {
            if context.contains_key("contrast") {
                return FocusType::Contrastive;
            }
            if context.contains_key("correction") {
                return FocusType::Corrective;
            }
        }

        // Check for local prominence patterns
        let local_prominence = self.calculate_local_prominence_ratio(position, stress_levels);

        if local_prominence > 2.0 {
            FocusType::Emphatic
        } else {
            FocusType::Information
        }
    }

    /// Calculate local prominence ratio
    fn calculate_local_prominence_ratio(&self, position: usize, stress_levels: &[f64]) -> f64 {
        let window_start = position.saturating_sub(2);
        let window_end = (position + 3).min(stress_levels.len());

        let local_context = &stress_levels[window_start..window_end];
        let local_mean = local_context.iter().sum::<f64>() / local_context.len() as f64;

        if local_mean > 0.0 {
            stress_levels[position] / local_mean
        } else {
            1.0
        }
    }

    /// Determine scope of focus
    fn determine_focus_scope(
        &self,
        focus_position: usize,
        syllables: &[String],
        stress_levels: &[f64],
    ) -> Vec<usize> {
        let mut scope = vec![focus_position];

        // Extend scope to adjacent stressed syllables
        let scope_threshold = stress_levels[focus_position] * 0.7;

        // Extend backward
        for i in (0..focus_position).rev() {
            if stress_levels[i] >= scope_threshold {
                scope.push(i);
            } else {
                break;
            }
        }

        // Extend forward
        for i in (focus_position + 1)..stress_levels.len() {
            if stress_levels[i] >= scope_threshold {
                scope.push(i);
            } else {
                break;
            }
        }

        scope.sort();
        scope
    }

    /// Detect contrastive stress patterns
    fn detect_contrastive_stress(
        &self,
        syllables: &[String],
        stress_levels: &[f64],
    ) -> StressResult<Vec<ContrastiveStress>> {
        let mut contrastive_markers = Vec::new();

        // Look for pairs of high prominence that might indicate contrast
        for i in 0..stress_levels.len() {
            if stress_levels[i] >= self.thresholds.contrastive_threshold {
                // Look for contrasting element within analysis window
                let window_start = (i + 1).min(stress_levels.len());
                let window_end =
                    (i + self.context_params.contrast_analysis_depth + 1).min(stress_levels.len());

                for j in window_start..window_end {
                    if stress_levels[j] >= self.thresholds.contrastive_threshold {
                        let strength = (stress_levels[i] + stress_levels[j]) / 2.0;
                        let contrasted_elements = vec![
                            syllables.get(i).unwrap_or(&"".to_string()).clone(),
                            syllables.get(j).unwrap_or(&"".to_string()).clone(),
                        ];
                        let context = self.build_contrast_context(i, j, syllables);

                        contrastive_markers.push(ContrastiveStress {
                            position: i,
                            strength,
                            contrasted_elements,
                            context,
                        });
                    }
                }
            }
        }

        Ok(contrastive_markers)
    }

    /// Build context description for contrastive stress
    fn build_contrast_context(&self, pos1: usize, pos2: usize, syllables: &[String]) -> String {
        let context_start = pos1.saturating_sub(2);
        let context_end = (pos2 + 3).min(syllables.len());

        syllables[context_start..context_end].join(" ")
    }
}

/// Main stress analyzer orchestrating all stress analysis components
#[derive(Debug, Clone)]
pub struct StressAnalyzer {
    config: StressAnalysisConfig,
    metrical_analyzer: MetricalFootAnalyzer,
    prominence_analyzer: ProminenceAnalyzer,
    analysis_cache: HashMap<u64, StressMetrics>,
}

impl StressAnalyzer {
    /// Create new stress analyzer
    pub fn new(config: StressAnalysisConfig) -> StressResult<Self> {
        Self::validate_config(&config)?;

        let metrical_analyzer = MetricalFootAnalyzer::new(config.clone());
        let prominence_analyzer = ProminenceAnalyzer::new(config.clone());

        Ok(Self {
            config,
            metrical_analyzer,
            prominence_analyzer,
            analysis_cache: HashMap::new(),
        })
    }

    /// Analyze stress patterns in text
    pub fn analyze_stress(
        &mut self,
        sentences: &[String],
        contextual_info: Option<&HashMap<String, String>>,
    ) -> StressResult<StressMetrics> {
        let cache_key = self.generate_cache_key(sentences, contextual_info);
        if let Some(cached) = self.analysis_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let (syllables, stress_patterns) = self.extract_stress_information(sentences)?;

        // Pattern naturalness analysis
        let pattern_naturalness = self.calculate_pattern_naturalness(&stress_patterns)?;

        // Metrical structure analysis
        let metrical_structure = if self.config.enable_metrical_analysis {
            self.metrical_analyzer
                .analyze_metrical_structure(&syllables, &stress_patterns)?
        } else {
            MetricalStructure::default()
        };

        let metrical_quality = self.evaluate_metrical_quality(&metrical_structure);

        // Prominence analysis
        let stress_levels: Vec<f64> = stress_patterns
            .iter()
            .flat_map(|pattern| pattern.levels.iter())
            .copied()
            .collect();

        let prominence_analysis = if self.config.enable_prominence_analysis {
            self.prominence_analyzer.analyze_prominence(
                &syllables,
                &stress_levels,
                contextual_info,
            )?
        } else {
            ProminenceAnalysis::default()
        };

        // Stress placement accuracy
        let placement_accuracy =
            self.calculate_placement_accuracy(&stress_patterns, &metrical_structure);

        // Stress clash detection
        let stress_clashes = if self.config.detect_stress_clashes {
            self.detect_stress_clashes(&syllables, &stress_levels)?
        } else {
            Vec::new()
        };

        // Foot analysis
        let foot_analysis = self.analyze_feet(&metrical_structure);

        // Accent pattern analysis
        let accent_patterns = if self.config.enable_accent_analysis {
            self.analyze_accent_patterns(&syllables, &stress_levels)?
        } else {
            Vec::new()
        };

        // Stress prediction accuracy (if enabled)
        let prediction_accuracy = if self.config.enable_stress_prediction {
            Some(self.evaluate_stress_prediction_accuracy(&stress_patterns))
        } else {
            None
        };

        // Calculate overall stress score
        let overall_score = self.calculate_overall_stress_score(
            pattern_naturalness,
            metrical_quality,
            placement_accuracy,
            &prominence_analysis,
        );

        let metrics = StressMetrics {
            overall_stress_score: overall_score,
            pattern_naturalness,
            metrical_quality,
            placement_accuracy,
            prominence_analysis,
            metrical_structure,
            stress_clashes,
            foot_analysis,
            accent_patterns,
            prediction_accuracy,
        };

        // Cache results
        self.analysis_cache.insert(cache_key, metrics.clone());

        Ok(metrics)
    }

    // Helper methods for stress analysis

    fn validate_config(config: &StressAnalysisConfig) -> StressResult<()> {
        if config.stress_weight < 0.0 {
            return Err(StressAnalysisError::ConfigError(
                "Stress weight must be non-negative".to_string(),
            ));
        }

        if config.metrical_consistency_threshold < 0.0
            || config.metrical_consistency_threshold > 1.0
        {
            return Err(StressAnalysisError::ConfigError(
                "Metrical consistency threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    fn generate_cache_key(
        &self,
        sentences: &[String],
        contextual_info: Option<&HashMap<String, String>>,
    ) -> u64 {
        let mut hasher = DefaultHasher::new();

        for sentence in sentences {
            sentence.hash(&mut hasher);
        }

        if let Some(context) = contextual_info {
            for (k, v) in context {
                k.hash(&mut hasher);
                v.hash(&mut hasher);
            }
        }

        self.config.enabled.hash(&mut hasher);
        hasher.finish()
    }

    fn extract_stress_information(
        &self,
        sentences: &[String],
    ) -> StressResult<(Vec<String>, Vec<StressPattern>)> {
        let mut all_syllables = Vec::new();
        let mut stress_patterns = Vec::new();

        for (sentence_idx, sentence) in sentences.iter().enumerate() {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let mut sentence_syllables = Vec::new();
            let mut sentence_positions = Vec::new();
            let mut sentence_levels = Vec::new();
            let mut sentence_types = Vec::new();

            for word in words {
                let word_syllables = self.extract_syllables(word);
                let (word_positions, word_levels, word_types) =
                    self.extract_word_stress(word, &word_syllables, all_syllables.len())?;

                sentence_syllables.extend(word_syllables);
                sentence_positions.extend(word_positions);
                sentence_levels.extend(word_levels);
                sentence_types.extend(word_types);
            }

            all_syllables.extend(sentence_syllables);

            if !sentence_positions.is_empty() {
                stress_patterns.push(StressPattern {
                    positions: sentence_positions,
                    levels: sentence_levels,
                    types: sentence_types,
                    confidence: 0.8, // Default confidence
                    context: format!("sentence_{}", sentence_idx),
                });
            }
        }

        Ok((all_syllables, stress_patterns))
    }

    fn extract_syllables(&self, word: &str) -> Vec<String> {
        // Simplified syllable extraction (would use proper syllabification in production)
        let vowels = "aeiouAEIOU";
        let mut syllables = Vec::new();
        let chars: Vec<char> = word.chars().collect();

        let mut current_syllable = String::new();
        let mut has_vowel = false;

        for ch in chars {
            current_syllable.push(ch);

            if vowels.contains(ch) {
                if has_vowel {
                    // Diphthong - continue current syllable
                    continue;
                }
                has_vowel = true;
            } else if has_vowel {
                // End of syllable
                syllables.push(current_syllable.trim().to_string());
                current_syllable = String::new();
                has_vowel = false;
            }
        }

        if !current_syllable.is_empty() {
            syllables.push(current_syllable);
        }

        if syllables.is_empty() {
            vec![word.to_string()]
        } else {
            syllables
        }
    }

    fn extract_word_stress(
        &self,
        word: &str,
        syllables: &[String],
        global_offset: usize,
    ) -> StressResult<(Vec<usize>, Vec<f64>, Vec<StressType>)> {
        let mut positions = Vec::new();
        let mut levels = Vec::new();
        let mut types = Vec::new();

        // Simple stress assignment heuristics (would use dictionary/rules in production)
        for (i, syllable) in syllables.iter().enumerate() {
            positions.push(global_offset + i);

            // Heuristic stress assignment
            let (stress_level, stress_type) = if syllables.len() == 1 {
                // Monosyllabic words - check if content word
                if self.is_content_word(word) {
                    (0.8, StressType::Primary)
                } else {
                    (0.2, StressType::Unstressed)
                }
            } else if i == 0 && syllables.len() <= 3 {
                // Short words - first syllable stress
                (0.8, StressType::Primary)
            } else if i == 1 && syllables.len() > 3 {
                // Longer words - second syllable stress
                (0.8, StressType::Primary)
            } else if i == 0 && syllables.len() > 3 {
                // Longer words - secondary stress on first
                (0.6, StressType::Secondary)
            } else {
                (0.2, StressType::Unstressed)
            };

            levels.push(stress_level);
            types.push(stress_type);
        }

        Ok((positions, levels, types))
    }

    fn is_content_word(&self, word: &str) -> bool {
        // Simple heuristic - would use POS tagging in production
        const FUNCTION_WORDS: &[&str] = &[
            "the", "a", "an", "and", "or", "but", "of", "to", "for", "with", "by", "from", "in",
            "on", "at", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "must",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
            "your", "his", "her", "its", "our", "their", "this", "that", "these", "those",
        ];

        !FUNCTION_WORDS.contains(&word.to_lowercase().as_str())
    }

    fn calculate_pattern_naturalness(&self, patterns: &[StressPattern]) -> StressResult<f64> {
        if patterns.is_empty() {
            return Ok(1.0);
        }

        let mut naturalness_scores = Vec::new();

        for pattern in patterns {
            let pattern_score = self.evaluate_pattern_naturalness(pattern)?;
            naturalness_scores.push(pattern_score * pattern.confidence);
        }

        let overall_naturalness =
            naturalness_scores.iter().sum::<f64>() / naturalness_scores.len() as f64;
        Ok(overall_naturalness)
    }

    fn evaluate_pattern_naturalness(&self, pattern: &StressPattern) -> StressResult<f64> {
        if pattern.types.is_empty() {
            return Ok(0.0);
        }

        // Check for natural stress patterns
        let mut score = 0.0;
        let mut factors = 0;

        // Prefer alternating stress patterns
        let alternation_score = self.calculate_alternation_score(&pattern.levels);
        score += alternation_score;
        factors += 1;

        // Check for appropriate primary stress placement
        let primary_count = pattern
            .types
            .iter()
            .filter(|&t| *t == StressType::Primary)
            .count();
        let primary_ratio = primary_count as f64 / pattern.types.len() as f64;

        // Prefer one primary stress per content word/phrase
        let primary_score = if primary_ratio > 0.0 && primary_ratio <= 0.5 {
            1.0 - (primary_ratio - 0.3).abs() * 2.0 // Optimal around 0.3
        } else {
            0.5
        };

        score += primary_score;
        factors += 1;

        if factors > 0 {
            Ok(score / factors as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_alternation_score(&self, levels: &[f64]) -> f64 {
        if levels.len() < 2 {
            return 1.0;
        }

        let mut alternation_changes = 0;
        let mut total_comparisons = 0;

        for i in 1..levels.len() {
            let prev_high = levels[i - 1] > 0.5;
            let curr_high = levels[i] > 0.5;

            if prev_high != curr_high {
                alternation_changes += 1;
            }
            total_comparisons += 1;
        }

        if total_comparisons > 0 {
            alternation_changes as f64 / total_comparisons as f64
        } else {
            0.0
        }
    }

    fn evaluate_metrical_quality(&self, structure: &MetricalStructure) -> f64 {
        let mut quality_components = Vec::new();

        // Consistency component
        quality_components.push(structure.consistency_score);

        // Regularity component
        quality_components.push(structure.pattern_regularity);

        // Violation penalty
        let violation_penalty = if structure.violations.is_empty() {
            1.0
        } else {
            let total_severity: f64 = structure.violations.iter().map(|v| v.severity).sum();
            let avg_severity = total_severity / structure.violations.len() as f64;
            1.0 - avg_severity.min(0.5) // Cap penalty at 0.5
        };
        quality_components.push(violation_penalty);

        quality_components.iter().sum::<f64>() / quality_components.len() as f64
    }

    fn calculate_placement_accuracy(
        &self,
        patterns: &[StressPattern],
        metrical_structure: &MetricalStructure,
    ) -> f64 {
        // Evaluate how well stress placement aligns with metrical structure
        let mut accuracy_scores = Vec::new();

        for foot in &metrical_structure.metrical_feet {
            let foot_accuracy = self.evaluate_foot_stress_accuracy(foot, patterns);
            accuracy_scores.push(foot_accuracy);
        }

        if accuracy_scores.is_empty() {
            0.5 // Default moderate accuracy
        } else {
            accuracy_scores.iter().sum::<f64>() / accuracy_scores.len() as f64
        }
    }

    fn evaluate_foot_stress_accuracy(
        &self,
        foot: &MetricalFoot,
        patterns: &[StressPattern],
    ) -> f64 {
        let expected_pattern = self.get_expected_foot_pattern(&foot.foot_type);

        if expected_pattern.len() != foot.stress_pattern.len() {
            return 0.5; // Partial accuracy for length mismatch
        }

        let matches = expected_pattern
            .iter()
            .zip(foot.stress_pattern.iter())
            .filter(|(expected, actual)| **expected == **actual)
            .count();

        matches as f64 / expected_pattern.len() as f64
    }

    fn get_expected_foot_pattern(&self, foot_type: &FootType) -> Vec<bool> {
        match foot_type {
            FootType::Trochee => vec![true, false],
            FootType::Iamb => vec![false, true],
            FootType::Dactyl => vec![true, false, false],
            FootType::Anapest => vec![false, false, true],
            FootType::Spondee => vec![true, true],
            FootType::Pyrrhic => vec![false, false],
        }
    }

    fn detect_stress_clashes(
        &self,
        syllables: &[String],
        stress_levels: &[f64],
    ) -> StressResult<Vec<StressClash>> {
        let mut clashes = Vec::new();

        for i in 1..stress_levels.len() {
            if stress_levels[i - 1] >= 0.7 && stress_levels[i] >= 0.7 {
                let positions = vec![i - 1, i];
                let severity = (stress_levels[i - 1] + stress_levels[i]) / 2.0;
                let context = if i >= 2 && i < syllables.len() - 1 {
                    syllables[i - 2..i + 2].join(" ")
                } else {
                    syllables[i - 1..i + 1].join(" ")
                };

                let resolutions = vec![
                    "Reduce stress on first syllable".to_string(),
                    "Insert pause between syllables".to_string(),
                    "Apply stress shift".to_string(),
                ];

                clashes.push(StressClash {
                    positions,
                    severity,
                    context,
                    resolutions,
                });
            }
        }

        Ok(clashes)
    }

    fn analyze_feet(&self, metrical_structure: &MetricalStructure) -> FootAnalysis {
        let feet = &metrical_structure.metrical_feet;

        if feet.is_empty() {
            return FootAnalysis::default();
        }

        // Calculate boundary accuracy (simplified)
        let boundary_accuracy = metrical_structure.consistency_score;

        // Calculate type distribution
        let mut type_counts: HashMap<FootType, usize> = HashMap::new();
        for foot in feet {
            *type_counts.entry(foot.foot_type.clone()).or_insert(0) += 1;
        }

        let total_feet = feet.len() as f64;
        let type_distribution: HashMap<FootType, f64> = type_counts
            .into_iter()
            .map(|(foot_type, count)| (foot_type, count as f64 / total_feet))
            .collect();

        // Calculate average foot length
        let average_length =
            feet.iter().map(|f| f.syllables.len()).sum::<usize>() as f64 / total_feet;

        // Regularity score (from metrical structure)
        let regularity_score = metrical_structure.pattern_regularity;

        // Complex patterns (simplified)
        let mut complex_patterns = Vec::new();
        let mut pattern_counts: HashMap<String, (usize, Vec<usize>)> = HashMap::new();

        for (i, foot) in feet.iter().enumerate() {
            let pattern_key = format!("{:?}_{}", foot.foot_type, foot.syllables.len());
            let entry = pattern_counts
                .entry(pattern_key.clone())
                .or_insert((0, Vec::new()));
            entry.0 += 1;
            entry.1.push(i);
        }

        for (pattern, (count, positions)) in pattern_counts {
            if count > 2 {
                // Patterns that appear multiple times
                complex_patterns.push(ComplexFootPattern {
                    pattern: pattern.clone(),
                    count,
                    complexity: self.calculate_pattern_complexity(&pattern),
                    positions,
                });
            }
        }

        FootAnalysis {
            boundary_accuracy,
            type_distribution,
            average_length,
            regularity_score,
            complex_patterns,
        }
    }

    fn calculate_pattern_complexity(&self, pattern: &str) -> f64 {
        // Simple complexity measure based on pattern description length
        pattern.len() as f64 / 20.0 // Normalize to 0-1 range approximately
    }

    fn analyze_accent_patterns(
        &self,
        syllables: &[String],
        stress_levels: &[f64],
    ) -> StressResult<Vec<AccentPattern>> {
        let mut accent_patterns = Vec::new();

        for (i, &level) in stress_levels.iter().enumerate() {
            if level >= 0.6 {
                // Accented syllable threshold
                let accent_type = self.classify_accent_type(i, stress_levels);
                let tonal_properties = None; // Would be enhanced with pitch analysis

                accent_patterns.push(AccentPattern {
                    accent_type,
                    position: i,
                    strength: level,
                    tonal_properties,
                });
            }
        }

        Ok(accent_patterns)
    }

    fn classify_accent_type(&self, position: usize, stress_levels: &[f64]) -> AccentType {
        // Simple classification based on local context
        let current_level = stress_levels[position];

        // Check preceding and following levels
        let prev_level = if position > 0 {
            stress_levels[position - 1]
        } else {
            0.0
        };
        let next_level = if position + 1 < stress_levels.len() {
            stress_levels[position + 1]
        } else {
            0.0
        };

        if current_level > prev_level && current_level > next_level {
            if current_level >= 0.9 {
                AccentType::HighTone
            } else {
                AccentType::Rising
            }
        } else if current_level > prev_level {
            AccentType::Rising
        } else if current_level > next_level {
            AccentType::Falling
        } else {
            AccentType::Complex
        }
    }

    fn evaluate_stress_prediction_accuracy(&self, patterns: &[StressPattern]) -> f64 {
        // Placeholder for stress prediction accuracy evaluation
        // Would compare predicted vs. actual stress in production system
        let mut accuracy_sum = 0.0;
        let mut total_patterns = 0;

        for pattern in patterns {
            // Simple confidence-based accuracy estimate
            accuracy_sum += pattern.confidence;
            total_patterns += 1;
        }

        if total_patterns > 0 {
            accuracy_sum / total_patterns as f64
        } else {
            0.0
        }
    }

    fn calculate_overall_stress_score(
        &self,
        pattern_naturalness: f64,
        metrical_quality: f64,
        placement_accuracy: f64,
        prominence_analysis: &ProminenceAnalysis,
    ) -> f64 {
        let naturalness_weight = 0.3;
        let metrical_weight = 0.25;
        let placement_weight = 0.25;
        let prominence_weight = 0.2;

        // Prominence score based on hierarchy quality
        let prominence_score = if prominence_analysis.prominence_hierarchy.is_empty() {
            0.5
        } else {
            let avg_prominence = prominence_analysis
                .prominence_hierarchy
                .iter()
                .map(|p| p.strength)
                .sum::<f64>()
                / prominence_analysis.prominence_hierarchy.len() as f64;
            avg_prominence
        };

        pattern_naturalness * naturalness_weight
            + metrical_quality * metrical_weight
            + placement_accuracy * placement_weight
            + prominence_score * prominence_weight
    }
}

impl Default for StressAnalyzer {
    fn default() -> Self {
        Self::new(StressAnalysisConfig {
            enabled: true,
            stress_weight: 0.20,
            enable_stress_prediction: false,
            enable_metrical_analysis: true,
            metrical_consistency_threshold: 0.7,
            enable_prominence_analysis: true,
            foot_boundary_accuracy: 0.7,
            detect_stress_clashes: true,
            primary_stress_preference: 0.8,
            enable_accent_analysis: true,
        })
        .unwrap()
    }
}
