//! Rhythm Analysis Module
//!
//! This module provides comprehensive rhythmic flow analysis for prosodic fluency,
//! including beat pattern detection, rhythm template matching, alternation scoring,
//! and rhythm classification with sophisticated pattern analysis capabilities.

use super::config::RhythmAnalysisConfig;
use super::results::{
    BeatPattern, BeatType, RhythmClass, RhythmClassification, RhythmMetrics, RhythmTemplateMatch,
    TempoAnalysis, TempoClass,
};
use crate::error::TextAnalysisError;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RhythmAnalysisError {
    #[error("Invalid rhythm analysis configuration: {0}")]
    ConfigError(String),
    #[error("Rhythm calculation failed: {0}")]
    CalculationError(String),
    #[error("Rhythm pattern analysis error: {0}")]
    PatternError(String),
    #[error("Beat detection error: {0}")]
    BeatError(String),
}

pub type RhythmResult<T> = Result<T, RhythmAnalysisError>;

/// Rhythm template for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmTemplate {
    /// Template name identifier
    pub name: String,
    /// Syllable pattern (stress levels)
    pub pattern: Vec<f64>,
    /// Template weight for scoring
    pub weight: f64,
    /// Context requirements for applicability
    pub context: Vec<String>,
    /// Template complexity score
    pub complexity: f64,
    /// Usage frequency in language
    pub frequency: f64,
}

impl RhythmTemplate {
    /// Create a new rhythm template
    pub fn new(name: String, pattern: Vec<f64>, weight: f64) -> Self {
        let complexity = Self::calculate_complexity(&pattern);
        Self {
            name,
            pattern,
            weight,
            context: Vec::new(),
            complexity,
            frequency: 1.0,
        }
    }

    /// Calculate pattern complexity
    fn calculate_complexity(pattern: &[f64]) -> f64 {
        if pattern.is_empty() {
            return 0.0;
        }

        // Calculate variance as complexity measure
        let mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
        let variance =
            pattern.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / pattern.len() as f64;

        variance.sqrt()
    }

    /// Check if template matches given pattern with tolerance
    pub fn matches(&self, pattern: &[f64], tolerance: f64) -> bool {
        if self.pattern.len() != pattern.len() {
            return false;
        }

        self.pattern
            .iter()
            .zip(pattern.iter())
            .all(|(template_val, pattern_val)| (template_val - pattern_val).abs() <= tolerance)
    }

    /// Calculate similarity to given pattern
    pub fn similarity(&self, pattern: &[f64]) -> f64 {
        if self.pattern.is_empty() || pattern.is_empty() {
            return 0.0;
        }

        let min_len = self.pattern.len().min(pattern.len());
        let max_len = self.pattern.len().max(pattern.len());

        // Calculate normalized correlation
        let correlation = self.pattern[..min_len]
            .iter()
            .zip(pattern[..min_len].iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

        let norm_a = self.pattern[..min_len]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        let norm_b = pattern[..min_len].iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        let similarity = correlation / (norm_a * norm_b);
        let length_penalty = min_len as f64 / max_len as f64;

        similarity * length_penalty
    }
}

/// Beat pattern analyzer for rhythm detection
#[derive(Debug, Clone)]
pub struct BeatPatternAnalyzer {
    /// Beat detection sensitivity
    pub sensitivity: f64,
    /// Minimum beat strength threshold
    pub min_strength_threshold: f64,
    /// Beat classification thresholds
    pub classification_thresholds: BeatClassificationThresholds,
    /// Temporal window for beat analysis
    pub temporal_window: usize,
}

#[derive(Debug, Clone)]
pub struct BeatClassificationThresholds {
    /// Strong beat threshold
    pub strong_threshold: f64,
    /// Weak beat threshold
    pub weak_threshold: f64,
    /// Intermediate beat threshold
    pub intermediate_threshold: f64,
}

impl Default for BeatClassificationThresholds {
    fn default() -> Self {
        Self {
            strong_threshold: 0.8,
            weak_threshold: 0.3,
            intermediate_threshold: 0.6,
        }
    }
}

impl BeatPatternAnalyzer {
    /// Create new beat pattern analyzer
    pub fn new(sensitivity: f64) -> Self {
        Self {
            sensitivity,
            min_strength_threshold: 0.1,
            classification_thresholds: BeatClassificationThresholds::default(),
            temporal_window: 5,
        }
    }

    /// Detect beat patterns in syllable sequence
    pub fn detect_beats(
        &self,
        syllables: &[String],
        stress_levels: &[f64],
    ) -> RhythmResult<Vec<BeatPattern>> {
        if syllables.len() != stress_levels.len() {
            return Err(RhythmAnalysisError::BeatError(
                "Syllable count must match stress level count".to_string(),
            ));
        }

        let mut beats = Vec::new();

        for (i, (syllable, &stress)) in syllables.iter().zip(stress_levels.iter()).enumerate() {
            let adjusted_stress = self.adjust_stress_for_context(stress, i, stress_levels);

            if adjusted_stress >= self.min_strength_threshold {
                let beat_type = self.classify_beat(adjusted_stress);
                let context = self.build_beat_context(i, syllables);

                beats.push(BeatPattern {
                    beat_type,
                    position: i,
                    strength: adjusted_stress,
                    context,
                });
            }
        }

        Ok(beats)
    }

    /// Adjust stress level based on local context
    fn adjust_stress_for_context(
        &self,
        stress: f64,
        position: usize,
        stress_levels: &[f64],
    ) -> f64 {
        let window_start = position.saturating_sub(self.temporal_window / 2);
        let window_end = (position + self.temporal_window / 2 + 1).min(stress_levels.len());

        let local_context = &stress_levels[window_start..window_end];
        let local_mean = local_context.iter().sum::<f64>() / local_context.len() as f64;
        let local_max = local_context.iter().fold(0.0, |max, &x| max.max(x));

        // Normalize stress relative to local context
        let normalized_stress = if local_max > 0.0 {
            stress / local_max
        } else {
            stress
        };

        // Apply sensitivity adjustment
        normalized_stress * self.sensitivity
    }

    /// Classify beat type based on strength
    fn classify_beat(&self, strength: f64) -> BeatType {
        if strength >= self.classification_thresholds.strong_threshold {
            BeatType::Strong
        } else if strength >= self.classification_thresholds.intermediate_threshold {
            BeatType::Intermediate
        } else if strength >= self.classification_thresholds.weak_threshold {
            BeatType::Weak
        } else {
            BeatType::Silent
        }
    }

    /// Build contextual information for beat
    fn build_beat_context(&self, position: usize, syllables: &[String]) -> String {
        let context_start = position.saturating_sub(2);
        let context_end = (position + 3).min(syllables.len());

        syllables[context_start..context_end].join(" ")
    }

    /// Calculate beat consistency across sequence
    pub fn calculate_beat_consistency(&self, beats: &[BeatPattern]) -> f64 {
        if beats.len() < 2 {
            return 1.0;
        }

        // Calculate intervals between beats
        let mut intervals = Vec::new();
        for window in beats.windows(2) {
            let interval = (window[1].position - window[0].position) as f64;
            intervals.push(interval);
        }

        if intervals.is_empty() {
            return 1.0;
        }

        // Calculate coefficient of variation
        let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let variance = intervals
            .iter()
            .map(|&interval| (interval - mean_interval).powi(2))
            .sum::<f64>()
            / intervals.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean_interval > 0.0 {
            std_dev / mean_interval
        } else {
            1.0
        };

        // Consistency is inverse of coefficient of variation
        1.0 / (1.0 + coefficient_of_variation)
    }
}

/// Rhythm template matcher for pattern recognition
#[derive(Debug, Clone)]
pub struct RhythmTemplateMatcher {
    /// Collection of rhythm templates
    pub templates: Vec<RhythmTemplate>,
    /// Matching tolerance
    pub tolerance: f64,
    /// Minimum match confidence threshold
    pub min_confidence: f64,
    /// Template usage statistics
    pub usage_stats: HashMap<String, usize>,
}

impl RhythmTemplateMatcher {
    /// Create new template matcher with default templates
    pub fn new(tolerance: f64) -> Self {
        let templates = Self::create_default_templates();

        Self {
            templates,
            tolerance,
            min_confidence: 0.3,
            usage_stats: HashMap::new(),
        }
    }

    /// Create default rhythm templates for common patterns
    fn create_default_templates() -> Vec<RhythmTemplate> {
        vec![
            // Iambic patterns
            RhythmTemplate::new("iambic_dimeter".to_string(), vec![0.2, 0.8, 0.2, 0.8], 1.0),
            RhythmTemplate::new(
                "iambic_trimeter".to_string(),
                vec![0.2, 0.8, 0.2, 0.8, 0.2, 0.8],
                1.0,
            ),
            RhythmTemplate::new(
                "iambic_tetrameter".to_string(),
                vec![0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8],
                1.0,
            ),
            RhythmTemplate::new(
                "iambic_pentameter".to_string(),
                vec![0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8],
                1.0,
            ),
            // Trochaic patterns
            RhythmTemplate::new(
                "trochaic_dimeter".to_string(),
                vec![0.8, 0.2, 0.8, 0.2],
                1.0,
            ),
            RhythmTemplate::new(
                "trochaic_trimeter".to_string(),
                vec![0.8, 0.2, 0.8, 0.2, 0.8, 0.2],
                1.0,
            ),
            RhythmTemplate::new(
                "trochaic_tetrameter".to_string(),
                vec![0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2],
                1.0,
            ),
            // Dactylic patterns
            RhythmTemplate::new(
                "dactylic_dimeter".to_string(),
                vec![0.8, 0.3, 0.3, 0.8, 0.3, 0.3],
                1.0,
            ),
            RhythmTemplate::new(
                "dactylic_trimeter".to_string(),
                vec![0.8, 0.3, 0.3, 0.8, 0.3, 0.3, 0.8, 0.3, 0.3],
                1.0,
            ),
            // Anapestic patterns
            RhythmTemplate::new(
                "anapestic_dimeter".to_string(),
                vec![0.3, 0.3, 0.8, 0.3, 0.3, 0.8],
                1.0,
            ),
            RhythmTemplate::new(
                "anapestic_trimeter".to_string(),
                vec![0.3, 0.3, 0.8, 0.3, 0.3, 0.8, 0.3, 0.3, 0.8],
                1.0,
            ),
            // Mixed and complex patterns
            RhythmTemplate::new(
                "alternating_strong".to_string(),
                vec![0.8, 0.2, 0.8, 0.2, 0.8],
                0.8,
            ),
            RhythmTemplate::new(
                "alternating_weak".to_string(),
                vec![0.2, 0.8, 0.2, 0.8, 0.2],
                0.8,
            ),
            RhythmTemplate::new("triple_meter".to_string(), vec![0.8, 0.4, 0.4], 0.7),
            RhythmTemplate::new(
                "compound_duple".to_string(),
                vec![0.8, 0.3, 0.6, 0.8, 0.3, 0.6],
                0.7,
            ),
        ]
    }

    /// Find template matches in stress pattern
    pub fn find_matches(
        &mut self,
        stress_pattern: &[f64],
    ) -> RhythmResult<Vec<RhythmTemplateMatch>> {
        let mut matches = Vec::new();

        for template in &self.templates {
            let match_result = self.match_template(template, stress_pattern)?;
            if match_result.confidence >= self.min_confidence {
                matches.push(match_result);

                // Update usage statistics
                *self.usage_stats.entry(template.name.clone()).or_insert(0) += 1;
            }
        }

        // Sort matches by confidence
        matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(matches)
    }

    /// Match a single template against stress pattern
    fn match_template(
        &self,
        template: &RhythmTemplate,
        stress_pattern: &[f64],
    ) -> RhythmResult<RhythmTemplateMatch> {
        let mut best_confidence = 0.0;
        let mut best_coverage = 0.0;
        let mut match_positions = Vec::new();

        // Try matching template at different positions
        if template.pattern.len() <= stress_pattern.len() {
            for start_pos in 0..=(stress_pattern.len() - template.pattern.len()) {
                let segment = &stress_pattern[start_pos..start_pos + template.pattern.len()];
                let similarity = template.similarity(segment);

                if similarity > best_confidence {
                    best_confidence = similarity;
                    match_positions = vec![start_pos];
                } else if (similarity - best_confidence).abs() < 0.01 {
                    // Add position for similar confidence
                    match_positions.push(start_pos);
                }
            }

            // Calculate coverage
            if !match_positions.is_empty() {
                let covered_positions = match_positions.len() * template.pattern.len();
                best_coverage = covered_positions as f64 / stress_pattern.len() as f64;
            }
        }

        // Adjust confidence based on template weight and coverage
        let adjusted_confidence = best_confidence * template.weight * (0.5 + 0.5 * best_coverage);

        Ok(RhythmTemplateMatch {
            template_name: template.name.clone(),
            confidence: adjusted_confidence,
            coverage: best_coverage,
            pattern: template.pattern.clone(),
            positions: match_positions,
        })
    }

    /// Add custom template to matcher
    pub fn add_template(&mut self, template: RhythmTemplate) {
        self.templates.push(template);
    }

    /// Get most frequently used templates
    pub fn get_popular_templates(&self, limit: usize) -> Vec<(String, usize)> {
        let mut templates_with_usage: Vec<_> = self
            .usage_stats
            .iter()
            .map(|(name, &count)| (name.clone(), count))
            .collect();

        templates_with_usage.sort_by(|a, b| b.1.cmp(&a.1));
        templates_with_usage.truncate(limit);
        templates_with_usage
    }
}

/// Rhythm classifier for categorizing rhythm types
#[derive(Debug, Clone)]
pub struct RhythmClassifier {
    /// Classification confidence threshold
    pub confidence_threshold: f64,
    /// Feature extraction parameters
    pub feature_params: RhythmFeatureParameters,
    /// Classification models or rules
    pub classification_rules: HashMap<RhythmClass, ClassificationRule>,
}

#[derive(Debug, Clone)]
pub struct RhythmFeatureParameters {
    /// Window size for feature extraction
    pub window_size: usize,
    /// Number of timing intervals to analyze
    pub timing_intervals: usize,
    /// Stress level quantization bins
    pub stress_bins: usize,
}

impl Default for RhythmFeatureParameters {
    fn default() -> Self {
        Self {
            window_size: 10,
            timing_intervals: 5,
            stress_bins: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClassificationRule {
    /// Rule conditions
    pub conditions: Vec<FeatureCondition>,
    /// Rule confidence weight
    pub weight: f64,
    /// Required feature thresholds
    pub feature_thresholds: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct FeatureCondition {
    /// Feature name
    pub feature: String,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Threshold value
    pub threshold: f64,
}

#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    Range(f64, f64),
}

impl RhythmClassifier {
    /// Create new rhythm classifier
    pub fn new(confidence_threshold: f64) -> Self {
        let classification_rules = Self::create_default_rules();

        Self {
            confidence_threshold,
            feature_params: RhythmFeatureParameters::default(),
            classification_rules,
        }
    }

    /// Create default classification rules
    fn create_default_rules() -> HashMap<RhythmClass, ClassificationRule> {
        let mut rules = HashMap::new();

        // Stress-timed rhythm characteristics
        rules.insert(
            RhythmClass::StressTimed,
            ClassificationRule {
                conditions: vec![
                    FeatureCondition {
                        feature: "stress_regularity".to_string(),
                        operator: ComparisonOperator::GreaterThan,
                        threshold: 0.7,
                    },
                    FeatureCondition {
                        feature: "vowel_reduction".to_string(),
                        operator: ComparisonOperator::GreaterThan,
                        threshold: 0.5,
                    },
                ],
                weight: 1.0,
                feature_thresholds: [
                    ("isochrony".to_string(), 0.6),
                    ("stress_prominence".to_string(), 0.8),
                ]
                .iter()
                .cloned()
                .collect(),
            },
        );

        // Syllable-timed rhythm characteristics
        rules.insert(
            RhythmClass::SyllableTimed,
            ClassificationRule {
                conditions: vec![
                    FeatureCondition {
                        feature: "syllable_regularity".to_string(),
                        operator: ComparisonOperator::GreaterThan,
                        threshold: 0.7,
                    },
                    FeatureCondition {
                        feature: "vowel_reduction".to_string(),
                        operator: ComparisonOperator::LessThan,
                        threshold: 0.3,
                    },
                ],
                weight: 1.0,
                feature_thresholds: [
                    ("syllable_isochrony".to_string(), 0.6),
                    ("duration_variability".to_string(), 0.4),
                ]
                .iter()
                .cloned()
                .collect(),
            },
        );

        // Mixed rhythm characteristics
        rules.insert(
            RhythmClass::Mixed,
            ClassificationRule {
                conditions: vec![FeatureCondition {
                    feature: "rhythm_variability".to_string(),
                    operator: ComparisonOperator::Range(0.4, 0.7),
                    threshold: 0.0, // Not used for Range
                }],
                weight: 0.8,
                feature_thresholds: HashMap::new(),
            },
        );

        // Irregular rhythm characteristics
        rules.insert(
            RhythmClass::Irregular,
            ClassificationRule {
                conditions: vec![FeatureCondition {
                    feature: "pattern_entropy".to_string(),
                    operator: ComparisonOperator::GreaterThan,
                    threshold: 0.8,
                }],
                weight: 0.9,
                feature_thresholds: [("regularity".to_string(), 0.3)].iter().cloned().collect(),
            },
        );

        rules
    }

    /// Classify rhythm based on features
    pub fn classify_rhythm(
        &self,
        stress_pattern: &[f64],
        beat_patterns: &[BeatPattern],
    ) -> RhythmResult<RhythmClassification> {
        let features = self.extract_features(stress_pattern, beat_patterns)?;
        let mut class_scores = HashMap::new();

        for (rhythm_class, rule) in &self.classification_rules {
            let score = self.evaluate_rule(rule, &features);
            class_scores.insert(rhythm_class.clone(), score);
        }

        // Find best classification
        let (primary_class, primary_score) = class_scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(class, &score)| (class.clone(), score))
            .unwrap_or((RhythmClass::Irregular, 0.0));

        // Get secondary classes
        let mut secondary_classes = HashMap::new();
        for (class, &score) in &class_scores {
            if class != &primary_class && score > self.confidence_threshold * 0.5 {
                secondary_classes.insert(class.clone(), score);
            }
        }

        Ok(RhythmClassification {
            primary_class,
            secondary_classes,
            confidence: primary_score,
            features,
        })
    }

    /// Extract rhythm features for classification
    fn extract_features(
        &self,
        stress_pattern: &[f64],
        beat_patterns: &[BeatPattern],
    ) -> RhythmResult<HashMap<String, f64>> {
        let mut features = HashMap::new();

        // Stress-related features
        features.insert(
            "stress_regularity".to_string(),
            self.calculate_stress_regularity(stress_pattern),
        );
        features.insert(
            "stress_prominence".to_string(),
            self.calculate_stress_prominence(stress_pattern),
        );
        features.insert(
            "vowel_reduction".to_string(),
            self.estimate_vowel_reduction(stress_pattern),
        );

        // Timing-related features
        features.insert(
            "syllable_regularity".to_string(),
            self.calculate_syllable_regularity(beat_patterns),
        );
        features.insert(
            "isochrony".to_string(),
            self.calculate_isochrony(beat_patterns),
        );
        features.insert(
            "duration_variability".to_string(),
            self.calculate_duration_variability(beat_patterns),
        );

        // Pattern-related features
        features.insert(
            "pattern_entropy".to_string(),
            self.calculate_pattern_entropy(stress_pattern),
        );
        features.insert(
            "rhythm_variability".to_string(),
            self.calculate_rhythm_variability(stress_pattern),
        );

        Ok(features)
    }

    /// Evaluate classification rule against features
    fn evaluate_rule(&self, rule: &ClassificationRule, features: &HashMap<String, f64>) -> f64 {
        let mut condition_scores = Vec::new();

        for condition in &rule.conditions {
            if let Some(&feature_value) = features.get(&condition.feature) {
                let score = self.evaluate_condition(condition, feature_value);
                condition_scores.push(score);
            }
        }

        if condition_scores.is_empty() {
            return 0.0;
        }

        // Average condition scores and apply rule weight
        let average_score = condition_scores.iter().sum::<f64>() / condition_scores.len() as f64;
        average_score * rule.weight
    }

    /// Evaluate single condition
    fn evaluate_condition(&self, condition: &FeatureCondition, value: f64) -> f64 {
        match &condition.operator {
            ComparisonOperator::GreaterThan => {
                if value > condition.threshold {
                    1.0
                } else {
                    value / condition.threshold
                }
            }
            ComparisonOperator::LessThan => {
                if value < condition.threshold {
                    1.0
                } else {
                    condition.threshold / value.max(0.001)
                }
            }
            ComparisonOperator::Equal => {
                let diff = (value - condition.threshold).abs();
                1.0 / (1.0 + diff)
            }
            ComparisonOperator::Range(min, max) => {
                if value >= *min && value <= *max {
                    1.0
                } else if value < *min {
                    value / min
                } else {
                    max / value
                }
            }
        }
    }

    // Feature calculation methods
    fn calculate_stress_regularity(&self, stress_pattern: &[f64]) -> f64 {
        if stress_pattern.len() < 2 {
            return 1.0;
        }

        let mean_stress = stress_pattern.iter().sum::<f64>() / stress_pattern.len() as f64;
        let variance = stress_pattern
            .iter()
            .map(|&x| (x - mean_stress).powi(2))
            .sum::<f64>()
            / stress_pattern.len() as f64;

        1.0 / (1.0 + variance.sqrt())
    }

    fn calculate_stress_prominence(&self, stress_pattern: &[f64]) -> f64 {
        if stress_pattern.is_empty() {
            return 0.0;
        }

        let max_stress = stress_pattern.iter().fold(0.0, |max, &x| max.max(x));
        let mean_stress = stress_pattern.iter().sum::<f64>() / stress_pattern.len() as f64;

        if mean_stress > 0.0 {
            max_stress / mean_stress / stress_pattern.len() as f64
        } else {
            0.0
        }
    }

    fn estimate_vowel_reduction(&self, stress_pattern: &[f64]) -> f64 {
        // Estimate based on stress contrast
        if stress_pattern.len() < 2 {
            return 0.0;
        }

        let max_stress = stress_pattern.iter().fold(0.0, |max, &x| max.max(x));
        let min_stress = stress_pattern.iter().fold(1.0, |min, &x| min.min(x));

        if max_stress > 0.0 {
            (max_stress - min_stress) / max_stress
        } else {
            0.0
        }
    }

    fn calculate_syllable_regularity(&self, beat_patterns: &[BeatPattern]) -> f64 {
        if beat_patterns.len() < 2 {
            return 1.0;
        }

        // Calculate intervals between beats
        let intervals: Vec<f64> = beat_patterns
            .windows(2)
            .map(|window| (window[1].position - window[0].position) as f64)
            .collect();

        if intervals.is_empty() {
            return 1.0;
        }

        let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let variance = intervals
            .iter()
            .map(|&x| (x - mean_interval).powi(2))
            .sum::<f64>()
            / intervals.len() as f64;

        1.0 / (1.0 + variance.sqrt())
    }

    fn calculate_isochrony(&self, beat_patterns: &[BeatPattern]) -> f64 {
        if beat_patterns.len() < 3 {
            return 1.0;
        }

        // Measure regularity of strong beats
        let strong_beats: Vec<_> = beat_patterns
            .iter()
            .filter(|beat| matches!(beat.beat_type, BeatType::Strong))
            .collect();

        if strong_beats.len() < 2 {
            return 0.5;
        }

        let intervals: Vec<f64> = strong_beats
            .windows(2)
            .map(|window| (window[1].position - window[0].position) as f64)
            .collect();

        let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let coefficient_of_variation = if mean_interval > 0.0 {
            let std_dev = intervals
                .iter()
                .map(|&x| (x - mean_interval).powi(2))
                .sum::<f64>()
                .sqrt()
                / intervals.len() as f64;
            std_dev / mean_interval
        } else {
            1.0
        };

        1.0 / (1.0 + coefficient_of_variation)
    }

    fn calculate_duration_variability(&self, beat_patterns: &[BeatPattern]) -> f64 {
        if beat_patterns.is_empty() {
            return 0.0;
        }

        let strengths: Vec<f64> = beat_patterns.iter().map(|beat| beat.strength).collect();
        let mean_strength = strengths.iter().sum::<f64>() / strengths.len() as f64;

        if mean_strength == 0.0 {
            return 0.0;
        }

        let variance = strengths
            .iter()
            .map(|&x| (x - mean_strength).powi(2))
            .sum::<f64>()
            / strengths.len() as f64;

        variance.sqrt() / mean_strength
    }

    fn calculate_pattern_entropy(&self, stress_pattern: &[f64]) -> f64 {
        if stress_pattern.is_empty() {
            return 0.0;
        }

        // Discretize stress values into bins
        let bins = self.feature_params.stress_bins;
        let max_stress = stress_pattern.iter().fold(0.0, |max, &x| max.max(x));

        if max_stress == 0.0 {
            return 0.0;
        }

        let mut bin_counts = vec![0; bins];
        for &stress in stress_pattern {
            let bin_index = ((stress / max_stress) * (bins - 1) as f64).floor() as usize;
            let bin_index = bin_index.min(bins - 1);
            bin_counts[bin_index] += 1;
        }

        // Calculate entropy
        let total = stress_pattern.len() as f64;
        let entropy = bin_counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.ln()
            })
            .sum::<f64>();

        entropy / (bins as f64).ln() // Normalize by maximum possible entropy
    }

    fn calculate_rhythm_variability(&self, stress_pattern: &[f64]) -> f64 {
        if stress_pattern.len() < self.feature_params.window_size {
            return self.calculate_pattern_entropy(stress_pattern);
        }

        // Calculate variability across windows
        let mut window_entropies = Vec::new();

        for window_start in 0..=(stress_pattern.len() - self.feature_params.window_size) {
            let window =
                &stress_pattern[window_start..window_start + self.feature_params.window_size];
            let entropy = self.calculate_pattern_entropy(window);
            window_entropies.push(entropy);
        }

        if window_entropies.is_empty() {
            return 0.0;
        }

        let mean_entropy = window_entropies.iter().sum::<f64>() / window_entropies.len() as f64;
        let variance = window_entropies
            .iter()
            .map(|&x| (x - mean_entropy).powi(2))
            .sum::<f64>()
            / window_entropies.len() as f64;

        variance.sqrt()
    }
}

/// Main rhythm analyzer orchestrating all rhythm analysis components
#[derive(Debug, Clone)]
pub struct RhythmAnalyzer {
    config: RhythmAnalysisConfig,
    beat_analyzer: BeatPatternAnalyzer,
    template_matcher: RhythmTemplateMatcher,
    rhythm_classifier: RhythmClassifier,
    analysis_cache: HashMap<u64, RhythmMetrics>,
}

impl RhythmAnalyzer {
    /// Create new rhythm analyzer
    pub fn new(config: RhythmAnalysisConfig) -> RhythmResult<Self> {
        Self::validate_config(&config)?;

        let beat_analyzer = BeatPatternAnalyzer::new(config.beat_detection_sensitivity);
        let template_matcher = RhythmTemplateMatcher::new(0.1);
        let rhythm_classifier = RhythmClassifier::new(0.3);

        Ok(Self {
            config,
            beat_analyzer,
            template_matcher,
            rhythm_classifier,
            analysis_cache: HashMap::new(),
        })
    }

    /// Analyze rhythmic patterns in text
    pub fn analyze_rhythm(&mut self, sentences: &[String]) -> RhythmResult<RhythmMetrics> {
        let cache_key = self.generate_cache_key(sentences);
        if let Some(cached) = self.analysis_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let (syllables, stress_levels) = self.extract_syllables_and_stress(sentences)?;

        // Beat pattern detection
        let beat_patterns = self
            .beat_analyzer
            .detect_beats(&syllables, &stress_levels)?;
        let beat_consistency = self
            .beat_analyzer
            .calculate_beat_consistency(&beat_patterns);

        // Template matching
        let template_matches = self.template_matcher.find_matches(&stress_levels)?;

        // Rhythm classification
        let rhythm_classification = if self.config.enable_rhythm_classification {
            Some(
                self.rhythm_classifier
                    .classify_rhythm(&stress_levels, &beat_patterns)?,
            )
        } else {
            None
        };

        // Calculate rhythm metrics
        let rhythm_regularity = self.calculate_rhythm_regularity(&stress_levels);
        let alternation_quality = self.calculate_alternation_quality(&stress_levels);
        let timing_variance = self.calculate_timing_variance(&beat_patterns);
        let rhythmic_complexity =
            self.calculate_rhythmic_complexity(&stress_levels, &beat_patterns);
        let pattern_entropy = self.calculate_pattern_entropy(&stress_levels);

        // Overall rhythm score
        let overall_score = self.calculate_overall_rhythm_score(
            rhythm_regularity,
            beat_consistency,
            alternation_quality,
            &template_matches,
            rhythmic_complexity,
        );

        let metrics = RhythmMetrics {
            overall_rhythm_score: overall_score,
            rhythm_regularity,
            beat_consistency,
            alternation_quality,
            template_matches,
            beat_patterns,
            rhythm_classification,
            timing_variance,
            rhythmic_complexity,
            pattern_entropy,
        };

        // Cache results
        self.analysis_cache.insert(cache_key, metrics.clone());

        Ok(metrics)
    }

    /// Extract syllables and stress levels from sentences
    fn extract_syllables_and_stress(
        &self,
        sentences: &[String],
    ) -> RhythmResult<(Vec<String>, Vec<f64>)> {
        let mut syllables = Vec::new();
        let mut stress_levels = Vec::new();

        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();

            for word in words {
                // Simple syllable extraction (would be enhanced with proper syllabification)
                let word_syllables = self.extract_word_syllables(word);
                let word_stress = self.extract_word_stress(word, &word_syllables);

                syllables.extend(word_syllables);
                stress_levels.extend(word_stress);
            }
        }

        Ok((syllables, stress_levels))
    }

    /// Simple syllable extraction (placeholder for proper implementation)
    fn extract_word_syllables(&self, word: &str) -> Vec<String> {
        // Simplified syllable counting based on vowel clusters
        let vowels = "aeiouAEIOU";
        let mut syllables = Vec::new();
        let chars: Vec<char> = word.chars().collect();

        let mut current_syllable = String::new();
        let mut vowel_found = false;

        for &ch in &chars {
            current_syllable.push(ch);

            if vowels.contains(ch) {
                if vowel_found {
                    continue; // Diphthong or vowel cluster
                }
                vowel_found = true;
            } else if vowel_found {
                // End of syllable
                if !current_syllable.is_empty() {
                    syllables.push(current_syllable.trim().to_string());
                    current_syllable = String::new();
                }
                vowel_found = false;
            }
        }

        if !current_syllable.is_empty() {
            syllables.push(current_syllable.trim().to_string());
        }

        if syllables.is_empty() {
            syllables.push(word.to_string());
        }

        syllables
    }

    /// Simple stress extraction (placeholder for proper implementation)
    fn extract_word_stress(&self, word: &str, syllables: &[String]) -> Vec<f64> {
        let mut stress = vec![0.3; syllables.len()]; // Default unstressed

        // Simple heuristic: first syllable gets primary stress in short words
        if syllables.len() <= 2 && !syllables.is_empty() {
            stress[0] = 0.8;
        } else if syllables.len() > 2 {
            // Second syllable gets stress in longer words (very simplified)
            stress[1] = 0.8;
        }

        stress
    }

    // Helper methods for rhythm analysis

    fn validate_config(config: &RhythmAnalysisConfig) -> RhythmResult<()> {
        if config.rhythm_weight < 0.0 {
            return Err(RhythmAnalysisError::ConfigError(
                "Rhythm weight must be non-negative".to_string(),
            ));
        }

        if config.beat_detection_sensitivity < 0.0 || config.beat_detection_sensitivity > 1.0 {
            return Err(RhythmAnalysisError::ConfigError(
                "Beat detection sensitivity must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    fn generate_cache_key(&self, sentences: &[String]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for sentence in sentences {
            sentence.hash(&mut hasher);
        }
        self.config.enabled.hash(&mut hasher);
        hasher.finish()
    }

    fn calculate_rhythm_regularity(&self, stress_levels: &[f64]) -> f64 {
        if stress_levels.len() < 2 {
            return 1.0;
        }

        // Calculate autocorrelation for different lags
        let max_lag = (stress_levels.len() / 4).max(1);
        let mut max_correlation = 0.0;

        for lag in 1..=max_lag {
            let correlation = self.calculate_autocorrelation(stress_levels, lag);
            max_correlation = max_correlation.max(correlation);
        }

        max_correlation
    }

    fn calculate_autocorrelation(&self, signal: &[f64], lag: usize) -> f64 {
        if lag >= signal.len() {
            return 0.0;
        }

        let n = signal.len() - lag;
        if n == 0 {
            return 0.0;
        }

        let mean = signal.iter().sum::<f64>() / signal.len() as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let x = signal[i] - mean;
            let y = signal[i + lag] - mean;
            numerator += x * y;
            denominator += x * x;
        }

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn calculate_alternation_quality(&self, stress_levels: &[f64]) -> f64 {
        if stress_levels.len() < 2 {
            return 1.0;
        }

        let mut alternation_score = 0.0;
        let mut comparisons = 0;

        for i in 1..stress_levels.len() {
            let current = stress_levels[i];
            let previous = stress_levels[i - 1];

            // Good alternation: high stress followed by low stress or vice versa
            let difference = (current - previous).abs();
            alternation_score += difference;
            comparisons += 1;
        }

        if comparisons > 0 {
            alternation_score / comparisons as f64
        } else {
            0.0
        }
    }

    fn calculate_timing_variance(&self, beat_patterns: &[BeatPattern]) -> f64 {
        if beat_patterns.len() < 2 {
            return 0.0;
        }

        let intervals: Vec<f64> = beat_patterns
            .windows(2)
            .map(|window| (window[1].position - window[0].position) as f64)
            .collect();

        if intervals.is_empty() {
            return 0.0;
        }

        let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let variance =
            intervals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / intervals.len() as f64;

        variance.sqrt()
    }

    fn calculate_rhythmic_complexity(
        &self,
        stress_levels: &[f64],
        beat_patterns: &[BeatPattern],
    ) -> f64 {
        if stress_levels.is_empty() {
            return 0.0;
        }

        // Combine stress pattern complexity with beat pattern complexity
        let stress_entropy = self.calculate_pattern_entropy(stress_levels);

        let beat_type_variety = {
            let mut type_counts = HashMap::new();
            for beat in beat_patterns {
                *type_counts.entry(&beat.beat_type).or_insert(0) += 1;
            }
            type_counts.len() as f64 / 4.0 // Normalize by max possible beat types
        };

        (stress_entropy + beat_type_variety) / 2.0
    }

    fn calculate_pattern_entropy(&self, pattern: &[f64]) -> f64 {
        if pattern.is_empty() {
            return 0.0;
        }

        // Discretize into bins for entropy calculation
        let bins = 5;
        let max_val = pattern.iter().fold(0.0, |max, &x| max.max(x));

        if max_val == 0.0 {
            return 0.0;
        }

        let mut bin_counts = vec![0; bins];
        for &value in pattern {
            let bin = ((value / max_val) * (bins - 1) as f64).floor() as usize;
            let bin = bin.min(bins - 1);
            bin_counts[bin] += 1;
        }

        // Calculate Shannon entropy
        let total = pattern.len() as f64;
        let entropy = bin_counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.ln()
            })
            .sum::<f64>();

        entropy / (bins as f64).ln() // Normalize
    }

    fn calculate_overall_rhythm_score(
        &self,
        regularity: f64,
        consistency: f64,
        alternation: f64,
        template_matches: &[RhythmTemplateMatch],
        complexity: f64,
    ) -> f64 {
        // Weight different components
        let regularity_weight = 0.3;
        let consistency_weight = 0.25;
        let alternation_weight = 0.2;
        let template_weight = 0.15;
        let complexity_weight = 0.1;

        // Template match score (best match confidence)
        let template_score = template_matches
            .first()
            .map(|m| m.confidence)
            .unwrap_or(0.0);

        // Complexity contributes inversely (simpler is better for rhythm)
        let complexity_score = 1.0 - complexity.min(1.0);

        regularity * regularity_weight
            + consistency * consistency_weight
            + alternation * alternation_weight
            + template_score * template_weight
            + complexity_score * complexity_weight
    }
}

impl Default for RhythmAnalyzer {
    fn default() -> Self {
        Self::new(RhythmAnalysisConfig {
            enabled: true,
            rhythm_weight: 0.25,
            prefer_regular_rhythm: true,
            enable_template_matching: true,
            max_rhythm_templates: 20,
            beat_detection_sensitivity: 0.7,
            alternation_preference: 0.6,
            enable_rhythm_classification: true,
            rhythm_complexity_threshold: 0.5,
            min_pattern_length: 3,
        })
        .unwrap()
    }
}
