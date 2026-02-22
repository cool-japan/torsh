//! Modular discourse coherence analysis system
//!
//! This module provides a comprehensive, modular implementation of discourse coherence analysis
//! that breaks down the complex analysis into focused, maintainable components while providing
//! a unified interface for analyzing discourse structure, coherence relations, and text flow.

use std::collections::HashMap;
use thiserror::Error;

pub mod cohesion;
pub mod config;
pub mod markers;
pub mod results;
pub mod rhetorical;

use cohesion::{CohesionAnalysisError, CohesionAnalyzer};
use config::{DiscourseCoherenceConfig, DiscourseCoherenceError};
use markers::{DiscourseMarkerAnalyzer, DiscourseMarkerError};
use results::{DetailedDiscourseMetrics, DiscourseCoherenceResult};
use rhetorical::{RhetoricalAnalysisError, RhetoricalStructureAnalyzer};

/// Comprehensive errors for discourse coherence analysis
#[derive(Debug, Error)]
pub enum ModularDiscourseError {
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Text processing error: {0}")]
    TextProcessingError(String),
    #[error("Marker analysis failed: {0}")]
    MarkerAnalysisError(#[from] DiscourseMarkerError),
    #[error("Rhetorical analysis failed: {0}")]
    RhetoricalAnalysisError(#[from] RhetoricalAnalysisError),
    #[error("Cohesion analysis failed: {0}")]
    CohesionAnalysisError(#[from] CohesionAnalysisError),
    #[error("Analysis integration error: {0}")]
    IntegrationError(String),
}

/// Main discourse coherence analyzer with modular architecture
pub struct DiscourseCoherenceAnalyzer {
    config: DiscourseCoherenceConfig,
    marker_analyzer: DiscourseMarkerAnalyzer,
    rhetorical_analyzer: RhetoricalStructureAnalyzer,
    cohesion_analyzer: CohesionAnalyzer,
}

impl DiscourseCoherenceAnalyzer {
    /// Create a new discourse coherence analyzer with default configuration
    pub fn new() -> Self {
        Self::with_config(DiscourseCoherenceConfig::default())
    }

    /// Create a new discourse coherence analyzer with custom configuration
    pub fn with_config(config: DiscourseCoherenceConfig) -> Self {
        let marker_analyzer = DiscourseMarkerAnalyzer::new(config.markers.clone());
        let rhetorical_analyzer = RhetoricalStructureAnalyzer::new(config.rhetorical.clone());
        let cohesion_analyzer = CohesionAnalyzer::new(config.cohesion.clone());

        Self {
            config,
            marker_analyzer,
            rhetorical_analyzer,
            cohesion_analyzer,
        }
    }

    /// Analyze discourse coherence of the given text
    pub fn analyze_discourse_coherence(
        &self,
        text: &str,
    ) -> Result<DiscourseCoherenceResult, ModularDiscourseError> {
        if text.trim().is_empty() {
            return Err(ModularDiscourseError::TextProcessingError(
                "Empty text provided".to_string(),
            ));
        }

        // Split text into sentences
        let sentences = self.split_into_sentences(text)?;

        // Analyze discourse markers
        let discourse_markers = self.marker_analyzer.analyze_markers(&sentences)?;

        // Calculate marker coherence score
        let marker_coherence_score = self.calculate_marker_coherence(&discourse_markers);

        // Analyze rhetorical structure
        let rhetorical_analysis = self
            .rhetorical_analyzer
            .analyze_rhetorical_structure(&sentences, &discourse_markers)?;
        let rhetorical_structure_score = rhetorical_analysis.relation_distribution_score;

        // Extract rhetorical relations for compatibility
        let rhetorical_relations: HashMap<String, usize> = rhetorical_analysis
            .relations
            .iter()
            .map(|(rel_type, &count)| (format!("{:?}", rel_type), count))
            .collect();

        // Analyze cohesion
        let cohesion_analysis = self.cohesion_analyzer.analyze_cohesion(&sentences)?;
        let cohesion_score = cohesion_analysis.overall_cohesion_score;

        // Calculate transition scores (simplified from cohesion analysis)
        let transition_scores = self.extract_transition_scores(&cohesion_analysis);

        // Calculate overall coherence score
        let overall_coherence_score = self.calculate_overall_coherence_score(
            marker_coherence_score,
            rhetorical_structure_score,
            cohesion_score,
        );

        // Build detailed metrics if enabled
        let detailed_metrics = if self.config.general.detailed_metrics {
            Some(self.build_detailed_metrics(
                &discourse_markers,
                &rhetorical_analysis,
                &cohesion_analysis,
            )?)
        } else {
            None
        };

        Ok(DiscourseCoherenceResult {
            overall_coherence_score,
            marker_coherence_score,
            rhetorical_structure_score,
            cohesion_score,
            transition_scores,
            discourse_markers,
            rhetorical_relations,
            detailed_metrics,
        })
    }

    /// Split text into sentences
    fn split_into_sentences(&self, text: &str) -> Result<Vec<String>, ModularDiscourseError> {
        // Simple sentence splitting (could be enhanced with proper sentence segmentation)
        let sentences: Vec<String> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() >= self.config.general.min_sentence_length)
            .collect();

        if sentences.is_empty() {
            return Err(ModularDiscourseError::TextProcessingError(
                "No valid sentences found".to_string(),
            ));
        }

        Ok(sentences)
    }

    /// Calculate marker coherence score
    fn calculate_marker_coherence(&self, markers: &[results::DiscourseMarker]) -> f64 {
        if markers.is_empty() {
            return 0.3; // Low but not zero for texts without explicit markers
        }

        let total_strength: f64 = markers.iter().map(|m| m.rhetorical_strength).sum();
        let average_confidence: f64 =
            markers.iter().map(|m| m.confidence).sum::<f64>() / markers.len() as f64;

        let strength_score = (total_strength / markers.len() as f64).min(1.0);

        // Combine strength and confidence with appropriate weighting
        (strength_score * 0.6 + average_confidence * 0.4).min(1.0)
    }

    /// Extract transition scores from cohesion analysis
    fn extract_transition_scores(&self, cohesion_analysis: &results::CohesionAnalysis) -> Vec<f64> {
        // Extract transition-relevant information from cohesion metrics
        let base_score = cohesion_analysis.overall_cohesion_score;
        let reference_score = cohesion_analysis.reference_cohesion.resolution_success_rate;
        let conjunctive_score = cohesion_analysis.conjunctive_cohesion.logical_flow_score;

        // Create transition scores (simplified approach)
        let transition_count = (cohesion_analysis.cohesive_devices.len() / 2).max(1);
        vec![
            (base_score * 0.4 + reference_score * 0.3 + conjunctive_score * 0.3).min(1.0);
            transition_count
        ]
    }

    /// Calculate overall coherence score
    fn calculate_overall_coherence_score(
        &self,
        marker_score: f64,
        rhetorical_score: f64,
        cohesion_score: f64,
    ) -> f64 {
        // Weighted combination of component scores
        let marker_weight = 0.3;
        let rhetorical_weight = 0.4;
        let cohesion_weight = 0.3;

        (marker_score * marker_weight
            + rhetorical_score * rhetorical_weight
            + cohesion_score * cohesion_weight)
            .min(1.0)
    }

    /// Build detailed metrics
    fn build_detailed_metrics(
        &self,
        discourse_markers: &[results::DiscourseMarker],
        rhetorical_analysis: &results::RhetoricalStructureAnalysis,
        cohesion_analysis: &results::CohesionAnalysis,
    ) -> Result<DetailedDiscourseMetrics, ModularDiscourseError> {
        // Generate marker analysis
        let marker_analysis = Some(self.marker_analyzer.generate_analysis(discourse_markers));

        // Use rhetorical analysis directly
        let rhetorical_analysis = Some(rhetorical_analysis.clone());

        // Use cohesion analysis directly
        let cohesion_analysis = Some(cohesion_analysis.clone());

        // Placeholder implementations for other analyses
        let transition_analysis = None; // Would be implemented with dedicated transition analyzer
        let information_structure = None; // Would be implemented with information structure analyzer
        let advanced_analysis = None; // Would be implemented with advanced analyzer

        Ok(DetailedDiscourseMetrics {
            marker_analysis,
            rhetorical_analysis,
            cohesion_analysis,
            transition_analysis,
            information_structure,
            advanced_analysis,
        })
    }

    /// Get current configuration
    pub fn get_configuration(&self) -> &DiscourseCoherenceConfig {
        &self.config
    }

    /// Validate input text
    pub fn validate_input(&self, text: &str) -> Result<(), ModularDiscourseError> {
        if text.trim().is_empty() {
            return Err(ModularDiscourseError::TextProcessingError(
                "Empty text provided".to_string(),
            ));
        }

        let word_count = text.split_whitespace().count();
        if word_count < self.config.general.min_sentence_length {
            return Err(ModularDiscourseError::TextProcessingError(format!(
                "Text too short: {} words",
                word_count
            )));
        }

        Ok(())
    }

    /// Analyze individual components (utility methods for fine-grained analysis)
    pub fn analyze_markers_only(
        &self,
        text: &str,
    ) -> Result<Vec<results::DiscourseMarker>, ModularDiscourseError> {
        let sentences = self.split_into_sentences(text)?;
        Ok(self.marker_analyzer.analyze_markers(&sentences)?)
    }

    pub fn analyze_rhetorical_structure_only(
        &self,
        text: &str,
    ) -> Result<results::RhetoricalStructureAnalysis, ModularDiscourseError> {
        let sentences = self.split_into_sentences(text)?;
        let markers = self.marker_analyzer.analyze_markers(&sentences)?;
        Ok(self
            .rhetorical_analyzer
            .analyze_rhetorical_structure(&sentences, &markers)?)
    }

    pub fn analyze_cohesion_only(
        &self,
        text: &str,
    ) -> Result<results::CohesionAnalysis, ModularDiscourseError> {
        let sentences = self.split_into_sentences(text)?;
        Ok(self.cohesion_analyzer.analyze_cohesion(&sentences)?)
    }
}

impl Default for DiscourseCoherenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export key types for backward compatibility
pub use config::{
    AdvancedAnalysisConfig, CohesionAnalysisConfig, CohesiveDeviceType, DiscourseMarkerConfig,
    DiscourseMarkerType, InformationStructureConfig, RhetoricalRelationType,
    RhetoricalStructureConfig, TransitionAnalysisConfig, TransitionQuality,
};

pub use results::{
    CohesionAnalysis, CohesiveDevice, ConjunctiveCohesionMetrics, ContextAnalysis, DiscourseMarker,
    DiscourseNode, DiscourseTree, LexicalCohesionMetrics, ReferenceCohesionMetrics,
    RhetoricalStructureAnalysis, SyntacticPosition, TemporalCoherenceMetrics,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discourse_coherence_analyzer_creation() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        assert!(!format!("{:?}", analyzer.get_configuration()).is_empty());
    }

    #[test]
    fn test_basic_discourse_coherence_analysis() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let text = "This is a test sentence. However, this is another sentence. Therefore, we can conclude something.";

        let result = analyzer.analyze_discourse_coherence(text);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.overall_coherence_score >= 0.0);
        assert!(analysis.overall_coherence_score <= 1.0);
    }

    #[test]
    fn test_individual_component_analysis() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let text = "First, we consider the problem. Then, we propose a solution. Finally, we evaluate the results.";

        // Test marker analysis
        let markers_result = analyzer.analyze_markers_only(text);
        assert!(markers_result.is_ok());

        // Test rhetorical analysis
        let rhetorical_result = analyzer.analyze_rhetorical_structure_only(text);
        assert!(rhetorical_result.is_ok());

        // Test cohesion analysis
        let cohesion_result = analyzer.analyze_cohesion_only(text);
        assert!(cohesion_result.is_ok());
    }

    #[test]
    fn test_empty_text_handling() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let result = analyzer.analyze_discourse_coherence("");
        assert!(result.is_err());
    }

    #[test]
    fn test_configuration_variants() {
        let minimal_config = DiscourseCoherenceConfig::minimal();
        let minimal_analyzer = DiscourseCoherenceAnalyzer::with_config(minimal_config);

        let comprehensive_config = DiscourseCoherenceConfig::comprehensive();
        let comprehensive_analyzer = DiscourseCoherenceAnalyzer::with_config(comprehensive_config);

        let text = "This is a test. However, it is simple.";

        let minimal_result = minimal_analyzer.analyze_discourse_coherence(text);
        let comprehensive_result = comprehensive_analyzer.analyze_discourse_coherence(text);

        assert!(minimal_result.is_ok());
        assert!(comprehensive_result.is_ok());
    }

    #[test]
    fn test_input_validation() {
        let analyzer = DiscourseCoherenceAnalyzer::new();

        assert!(analyzer.validate_input("").is_err());
        assert!(analyzer.validate_input("a").is_err()); // Too short
        assert!(analyzer
            .validate_input("This is a proper sentence.")
            .is_ok());
    }
}
