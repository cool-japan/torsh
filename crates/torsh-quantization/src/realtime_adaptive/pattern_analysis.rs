//! Workload pattern analysis and recognition
//!
//! This module provides pattern recognition capabilities to identify and classify
//! different computation patterns for adaptive quantization optimization.

use super::config::PerformanceProfile;
use crate::{QuantConfig, TorshResult};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Workload pattern analyzer
#[derive(Debug, Clone)]
pub struct WorkloadPatternAnalyzer {
    /// Detected patterns
    patterns: HashMap<String, WorkloadPattern>,
    /// Current pattern classification
    current_pattern: Option<String>,
    /// Pattern history
    pattern_history: VecDeque<PatternTransition>,
    /// Pattern classifier
    classifier: PatternClassifier,
}

/// Workload pattern definition
#[derive(Debug, Clone)]
pub struct WorkloadPattern {
    /// Pattern name
    pub name: String,
    /// Characteristic features
    pub features: Vec<f32>,
    /// Optimal quantization configuration
    pub optimal_config: QuantConfig,
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
    /// Frequency of occurrence
    pub frequency: usize,
}

/// Pattern transition tracking
#[derive(Debug, Clone)]
pub struct PatternTransition {
    /// Previous pattern
    pub from_pattern: Option<String>,
    /// New pattern
    pub to_pattern: String,
    /// Transition timestamp
    pub timestamp: Instant,
    /// Transition cost
    pub transition_cost: f32,
}

/// Pattern classifier for workload analysis
#[derive(Debug, Clone)]
pub struct PatternClassifier {
    /// Clustering centers for pattern recognition
    cluster_centers: Vec<Vec<f32>>,
    /// Classification confidence threshold
    #[allow(dead_code)]
    confidence_threshold: f32,
    /// Feature weights for classification
    feature_weights: Vec<f32>,
}

impl WorkloadPatternAnalyzer {
    /// Create new workload pattern analyzer
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // Initialize with common patterns
        patterns.insert(
            "compute_intensive".to_string(),
            WorkloadPattern {
                name: "compute_intensive".to_string(),
                features: vec![
                    0.8, 0.2, 0.1, 0.9, 0.7, 0.3, 0.5, 0.8, 0.6, 0.4, 0.3, 0.7, 0.5, 0.2, 0.8, 0.6,
                ],
                optimal_config: QuantConfig::default(),
                performance_profile: PerformanceProfile {
                    avg_execution_time: 5.0,
                    memory_usage: 200.0,
                    energy_consumption: 25.0,
                    cache_efficiency: 0.6,
                },
                frequency: 0,
            },
        );

        patterns.insert(
            "memory_bound".to_string(),
            WorkloadPattern {
                name: "memory_bound".to_string(),
                features: vec![
                    0.3, 0.8, 0.9, 0.2, 0.1, 0.7, 0.8, 0.3, 0.4, 0.9, 0.7, 0.2, 0.5, 0.8, 0.3, 0.6,
                ],
                optimal_config: QuantConfig::default(),
                performance_profile: PerformanceProfile {
                    avg_execution_time: 2.0,
                    memory_usage: 500.0,
                    energy_consumption: 15.0,
                    cache_efficiency: 0.9,
                },
                frequency: 0,
            },
        );

        patterns.insert(
            "balanced".to_string(),
            WorkloadPattern {
                name: "balanced".to_string(),
                features: vec![0.5; 16],
                optimal_config: QuantConfig::default(),
                performance_profile: PerformanceProfile::default(),
                frequency: 0,
            },
        );

        Self {
            patterns,
            current_pattern: None,
            pattern_history: VecDeque::new(),
            classifier: PatternClassifier::new(),
        }
    }

    /// Analyze pattern from features
    pub fn analyze_pattern(&mut self, features: &[f32]) -> TorshResult<Option<String>> {
        let classified_pattern = self.classifier.classify_pattern(features, &self.patterns)?;

        if let Some(ref pattern_name) = classified_pattern {
            // Update pattern frequency
            if let Some(pattern) = self.patterns.get_mut(pattern_name) {
                pattern.frequency += 1;
            }

            // Record pattern transition
            if self.current_pattern.as_ref() != Some(pattern_name) {
                let transition = PatternTransition {
                    from_pattern: self.current_pattern.clone(),
                    to_pattern: pattern_name.clone(),
                    timestamp: Instant::now(),
                    transition_cost: self
                        .calculate_transition_cost(&self.current_pattern, pattern_name),
                };

                self.pattern_history.push_back(transition);
                if self.pattern_history.len() > 100 {
                    self.pattern_history.pop_front();
                }

                self.current_pattern = Some(pattern_name.clone());
            }
        }

        Ok(classified_pattern)
    }

    /// Calculate cost of transitioning between patterns
    fn calculate_transition_cost(&self, from: &Option<String>, to: &String) -> f32 {
        match from {
            Some(from_pattern) if from_pattern == to => 0.0, // No transition
            Some(from_pattern) => {
                // Different patterns - calculate transition cost
                match (from_pattern.as_str(), to.as_str()) {
                    ("compute_intensive", "memory_bound") => 0.3,
                    ("memory_bound", "compute_intensive") => 0.4,
                    ("balanced", _) => 0.1,
                    (_, "balanced") => 0.1,
                    _ => 0.2,
                }
            }
            None => 0.0, // Initial pattern
        }
    }

    /// Get current pattern
    pub fn get_current_pattern(&self) -> &Option<String> {
        &self.current_pattern
    }

    /// Get pattern information
    pub fn get_pattern(&self, name: &str) -> Option<&WorkloadPattern> {
        self.patterns.get(name)
    }

    /// Get all patterns
    pub fn get_all_patterns(&self) -> &HashMap<String, WorkloadPattern> {
        &self.patterns
    }

    /// Add or update pattern
    pub fn add_pattern(&mut self, pattern: WorkloadPattern) {
        self.patterns.insert(pattern.name.clone(), pattern);
    }

    /// Get pattern transition history
    pub fn get_pattern_history(&self) -> &VecDeque<PatternTransition> {
        &self.pattern_history
    }

    /// Get pattern statistics
    pub fn get_pattern_statistics(&self) -> PatternStatistics {
        let total_frequency: usize = self.patterns.values().map(|p| p.frequency).sum();
        let most_common_pattern = self
            .patterns
            .iter()
            .max_by_key(|(_, pattern)| pattern.frequency)
            .map(|(name, _)| name.clone());

        let transition_count = self.pattern_history.len();
        let avg_transition_cost = if transition_count > 0 {
            self.pattern_history
                .iter()
                .map(|t| t.transition_cost)
                .sum::<f32>()
                / transition_count as f32
        } else {
            0.0
        };

        PatternStatistics {
            total_patterns: self.patterns.len(),
            total_frequency,
            most_common_pattern,
            current_pattern: self.current_pattern.clone(),
            transition_count,
            avg_transition_cost,
        }
    }

    /// Learn new pattern from features and performance data
    pub fn learn_pattern(
        &mut self,
        name: String,
        features: Vec<f32>,
        performance: PerformanceProfile,
    ) {
        let pattern = WorkloadPattern {
            name: name.clone(),
            features,
            optimal_config: QuantConfig::default(), // Would learn optimal config
            performance_profile: performance,
            frequency: 1,
        };

        self.patterns.insert(name, pattern);
        self.classifier.update_clusters(&self.patterns);
    }
}

impl Default for WorkloadPatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternClassifier {
    /// Create new pattern classifier
    fn new() -> Self {
        Self {
            cluster_centers: Vec::new(),
            confidence_threshold: 0.8,
            feature_weights: vec![1.0; 16], // Uniform weights initially
        }
    }

    /// Classify pattern based on features
    fn classify_pattern(
        &self,
        features: &[f32],
        patterns: &HashMap<String, WorkloadPattern>,
    ) -> TorshResult<Option<String>> {
        if patterns.is_empty() {
            return Ok(None);
        }

        let mut best_match = None;
        let mut best_distance = f32::INFINITY;

        for (name, pattern) in patterns {
            let distance = self.calculate_distance(features, &pattern.features);
            if distance < best_distance {
                best_distance = distance;
                best_match = Some(name.clone());
            }
        }

        // Return best match if distance is reasonable
        if best_distance < 2.0 {
            // Threshold for classification
            Ok(best_match)
        } else {
            Ok(Some("unknown".to_string()))
        }
    }

    /// Calculate weighted Euclidean distance
    fn calculate_distance(&self, features1: &[f32], features2: &[f32]) -> f32 {
        let min_len = features1.len().min(features2.len());
        let mut distance = 0.0;

        for i in 0..min_len {
            let weight = if i < self.feature_weights.len() {
                self.feature_weights[i]
            } else {
                1.0
            };
            let diff = features1[i] - features2[i];
            distance += weight * diff * diff;
        }

        distance.sqrt()
    }

    /// Update clustering centers based on current patterns
    fn update_clusters(&mut self, patterns: &HashMap<String, WorkloadPattern>) {
        self.cluster_centers.clear();
        for pattern in patterns.values() {
            self.cluster_centers.push(pattern.features.clone());
        }
    }

    /// Update feature weights based on pattern discrimination
    pub fn update_feature_weights(&mut self, _patterns: &HashMap<String, WorkloadPattern>) {
        // Simplified weight update - in practice would use more sophisticated methods
        // like mutual information or feature importance analysis
        for weight in &mut self.feature_weights {
            *weight = (*weight * 0.9 + 0.1).max(0.1).min(2.0);
        }
    }
}

/// Pattern analysis statistics
#[derive(Debug, Clone)]
pub struct PatternStatistics {
    pub total_patterns: usize,
    pub total_frequency: usize,
    pub most_common_pattern: Option<String>,
    pub current_pattern: Option<String>,
    pub transition_count: usize,
    pub avg_transition_cost: f32,
}

impl Default for PatternStatistics {
    fn default() -> Self {
        Self {
            total_patterns: 0,
            total_frequency: 0,
            most_common_pattern: None,
            current_pattern: None,
            transition_count: 0,
            avg_transition_cost: 0.0,
        }
    }
}
