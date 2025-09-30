//! Structural marker analysis for structural coherence
//!
//! This module provides comprehensive structural marker detection and analysis,
//! including marker categorization, density analysis, and effectiveness evaluation.

use crate::metrics::coherence::structural_coherence::{
    config::{StructuralMarkerConfig, StructuralMarkerType},
    results::{MarkerDistributionAnalysis, StructuralMarker, StructuralMarkerAnalysis},
};
use std::collections::HashMap;
use thiserror::Error;

/// Errors specific to structural marker analysis
#[derive(Debug, Error)]
pub enum StructuralMarkerError {
    #[error("Invalid structural marker configuration: {0}")]
    InvalidConfiguration(String),
    #[error("Marker detection failed: {0}")]
    MarkerDetectionError(String),
    #[error("Insufficient content for marker analysis")]
    InsufficientContent,
}

/// Structural marker analyzer
pub struct StructuralMarkerAnalyzer {
    config: StructuralMarkerConfig,
    marker_patterns: HashMap<StructuralMarkerType, Vec<String>>,
    effectiveness_rules: HashMap<StructuralMarkerType, f64>,
}

impl StructuralMarkerAnalyzer {
    /// Create a new structural marker analyzer
    pub fn new(config: StructuralMarkerConfig) -> Self {
        Self {
            config,
            marker_patterns: Self::build_marker_patterns(),
            effectiveness_rules: Self::build_effectiveness_rules(),
        }
    }

    /// Analyze structural markers in paragraphs
    pub fn analyze_structural_markers(
        &self,
        paragraphs: &[String],
    ) -> Result<StructuralMarkerAnalysis, StructuralMarkerError> {
        if !self.config.enable_analysis {
            return Ok(StructuralMarkerAnalysis::default());
        }

        if paragraphs.is_empty() {
            return Err(StructuralMarkerError::InsufficientContent);
        }

        let markers_by_type = self.categorize_markers_by_type(paragraphs)?;
        let marker_density = self.calculate_marker_density(paragraphs, &markers_by_type);
        let distribution_analysis =
            self.analyze_marker_distribution(paragraphs, &markers_by_type)?;
        let effectiveness_scores = if self.config.analyze_effectiveness {
            self.calculate_marker_effectiveness(paragraphs, &markers_by_type)?
        } else {
            HashMap::new()
        };
        let missing_markers = if self.config.detect_missing_markers {
            self.identify_missing_markers(paragraphs)?
        } else {
            Vec::new()
        };

        Ok(StructuralMarkerAnalysis {
            markers_by_type,
            marker_density,
            distribution_analysis,
            effectiveness_scores,
            missing_markers,
        })
    }

    /// Categorize markers by type across paragraphs
    fn categorize_markers_by_type(
        &self,
        paragraphs: &[String],
    ) -> Result<HashMap<String, Vec<StructuralMarker>>, StructuralMarkerError> {
        let mut markers_by_type = HashMap::new();

        for (paragraph_index, paragraph) in paragraphs.iter().enumerate() {
            let detected_markers = self.detect_markers_in_paragraph(paragraph, paragraph_index)?;

            for marker in detected_markers {
                let type_name = format!("{:?}", marker.marker_type);
                markers_by_type
                    .entry(type_name)
                    .or_insert_with(Vec::new)
                    .push(marker);
            }
        }

        // Add custom markers if any
        for custom_marker in &self.config.custom_markers {
            self.detect_custom_marker(paragraphs, custom_marker, &mut markers_by_type)?;
        }

        Ok(markers_by_type)
    }

    /// Detect markers in a single paragraph
    fn detect_markers_in_paragraph(
        &self,
        paragraph: &str,
        paragraph_index: usize,
    ) -> Result<Vec<StructuralMarker>, StructuralMarkerError> {
        let mut markers = Vec::new();
        let paragraph_lower = paragraph.to_lowercase();

        for (marker_type, patterns) in &self.marker_patterns {
            for pattern in patterns {
                if paragraph_lower.contains(pattern) {
                    let strength = self.calculate_marker_strength(paragraph, pattern, marker_type);

                    if strength >= self.config.detection_sensitivity {
                        let context = self.extract_marker_context(paragraph, pattern);
                        let effectiveness = self.calculate_individual_marker_effectiveness(
                            paragraph,
                            pattern,
                            marker_type,
                        );

                        markers.push(StructuralMarker {
                            text: pattern.clone(),
                            marker_type: marker_type.clone(),
                            position: paragraph_index,
                            strength,
                            context,
                            effectiveness,
                        });
                    }
                }
            }
        }

        Ok(markers)
    }

    /// Detect custom markers
    fn detect_custom_marker(
        &self,
        paragraphs: &[String],
        custom_pattern: &str,
        markers_by_type: &mut HashMap<String, Vec<StructuralMarker>>,
    ) -> Result<(), StructuralMarkerError> {
        for (paragraph_index, paragraph) in paragraphs.iter().enumerate() {
            if paragraph
                .to_lowercase()
                .contains(&custom_pattern.to_lowercase())
            {
                let strength = 0.8; // Default strength for custom markers
                let context = self.extract_marker_context(paragraph, custom_pattern);
                let effectiveness = 0.7; // Default effectiveness for custom markers

                let marker = StructuralMarker {
                    text: custom_pattern.clone(),
                    marker_type: StructuralMarkerType::Section, // Default type for custom
                    position: paragraph_index,
                    strength,
                    context,
                    effectiveness,
                };

                markers_by_type
                    .entry("Custom".to_string())
                    .or_insert_with(Vec::new)
                    .push(marker);
            }
        }

        Ok(())
    }

    /// Calculate strength of a detected marker
    fn calculate_marker_strength(
        &self,
        paragraph: &str,
        pattern: &str,
        marker_type: &StructuralMarkerType,
    ) -> f64 {
        let paragraph_lower = paragraph.to_lowercase();
        let pattern_lower = pattern.to_lowercase();

        // Base strength from pattern match
        let mut strength = if paragraph_lower.contains(&pattern_lower) {
            1.0
        } else {
            0.0
        };

        // Adjust based on context
        strength *= self.calculate_context_strength_multiplier(paragraph, pattern);

        // Adjust based on position in paragraph
        strength *= self.calculate_position_strength_multiplier(paragraph, pattern);

        // Adjust based on marker type appropriateness
        strength *= self.calculate_type_appropriateness_multiplier(paragraph, marker_type);

        strength.min(1.0)
    }

    /// Calculate context-based strength multiplier
    fn calculate_context_strength_multiplier(&self, paragraph: &str, pattern: &str) -> f64 {
        let pattern_lower = pattern.to_lowercase();
        let paragraph_lower = paragraph.to_lowercase();

        // Find the pattern in the paragraph
        if let Some(index) = paragraph_lower.find(&pattern_lower) {
            let before_context = &paragraph_lower[..index];
            let after_context = &paragraph_lower[index + pattern_lower.len()..];

            let mut multiplier = 1.0;

            // Check for supporting context
            if before_context.ends_with(' ') || before_context.is_empty() {
                multiplier += 0.1; // Word boundary before
            }
            if after_context.starts_with(' ')
                || after_context.starts_with('.')
                || after_context.starts_with(',')
            {
                multiplier += 0.1; // Word boundary after
            }

            // Check for sentence position
            if before_context.trim().is_empty() || before_context.ends_with('.') {
                multiplier += 0.2; // Beginning of sentence
            }

            multiplier.min(1.5)
        } else {
            1.0
        }
    }

    /// Calculate position-based strength multiplier
    fn calculate_position_strength_multiplier(&self, paragraph: &str, pattern: &str) -> f64 {
        let pattern_lower = pattern.to_lowercase();
        let paragraph_lower = paragraph.to_lowercase();

        if let Some(index) = paragraph_lower.find(&pattern_lower) {
            let position_ratio = index as f64 / paragraph_lower.len() as f64;

            // Some markers are more effective at certain positions
            match position_ratio {
                r if r < 0.2 => 1.2, // Beginning of paragraph
                r if r > 0.8 => 1.1, // End of paragraph
                _ => 1.0,            // Middle of paragraph
            }
        } else {
            1.0
        }
    }

    /// Calculate type appropriateness multiplier
    fn calculate_type_appropriateness_multiplier(
        &self,
        paragraph: &str,
        marker_type: &StructuralMarkerType,
    ) -> f64 {
        let paragraph_lower = paragraph.to_lowercase();

        match marker_type {
            StructuralMarkerType::Introduction => {
                if paragraph_lower.contains("introduce") || paragraph_lower.contains("begin") {
                    1.3
                } else {
                    1.0
                }
            }
            StructuralMarkerType::Conclusion => {
                if paragraph_lower.contains("conclude") || paragraph_lower.contains("summary") {
                    1.3
                } else {
                    1.0
                }
            }
            StructuralMarkerType::Transition => {
                if paragraph.len() < 500 {
                    // Short paragraphs are often transitions
                    1.2
                } else {
                    1.0
                }
            }
            StructuralMarkerType::Example => {
                if paragraph_lower.contains("instance") || paragraph_lower.contains("example") {
                    1.2
                } else {
                    1.0
                }
            }
            _ => 1.0,
        }
    }

    /// Extract context around marker
    fn extract_marker_context(&self, paragraph: &str, pattern: &str) -> String {
        let pattern_lower = pattern.to_lowercase();
        let paragraph_lower = paragraph.to_lowercase();

        if let Some(index) = paragraph_lower.find(&pattern_lower) {
            let context_start = index.saturating_sub(50);
            let context_end = (index + pattern_lower.len() + 50).min(paragraph.len());
            paragraph[context_start..context_end].to_string()
        } else {
            paragraph.chars().take(100).collect()
        }
    }

    /// Calculate effectiveness of individual marker
    fn calculate_individual_marker_effectiveness(
        &self,
        paragraph: &str,
        _pattern: &str,
        marker_type: &StructuralMarkerType,
    ) -> f64 {
        let base_effectiveness = self
            .effectiveness_rules
            .get(marker_type)
            .copied()
            .unwrap_or(0.5);

        // Adjust based on paragraph characteristics
        let length_factor = if paragraph.len() > 200 { 1.1 } else { 0.9 };
        let complexity_factor = self.calculate_paragraph_complexity_factor(paragraph);

        base_effectiveness * length_factor * complexity_factor
    }

    /// Calculate paragraph complexity factor for effectiveness
    fn calculate_paragraph_complexity_factor(&self, paragraph: &str) -> f64 {
        let sentences = paragraph
            .split('.')
            .filter(|s| !s.trim().is_empty())
            .count();
        let words = paragraph.split_whitespace().count();

        let avg_sentence_length = if sentences > 0 {
            words as f64 / sentences as f64
        } else {
            0.0
        };

        // Markers are more effective in complex paragraphs
        match avg_sentence_length {
            len if len > 25.0 => 1.2, // Complex sentences
            len if len > 15.0 => 1.1, // Medium sentences
            _ => 1.0,                 // Simple sentences
        }
    }

    /// Calculate overall marker density
    fn calculate_marker_density(
        &self,
        paragraphs: &[String],
        markers_by_type: &HashMap<String, Vec<StructuralMarker>>,
    ) -> f64 {
        let total_markers: usize = markers_by_type.values().map(|markers| markers.len()).sum();
        let total_words: usize = paragraphs
            .iter()
            .map(|p| p.split_whitespace().count())
            .sum();

        if total_words > 0 {
            total_markers as f64 / total_words as f64 * 100.0 // Markers per 100 words
        } else {
            0.0
        }
    }

    /// Analyze marker distribution across document
    fn analyze_marker_distribution(
        &self,
        paragraphs: &[String],
        markers_by_type: &HashMap<String, Vec<StructuralMarker>>,
    ) -> Result<MarkerDistributionAnalysis, StructuralMarkerError> {
        let total_markers: usize = markers_by_type.values().map(|markers| markers.len()).sum();

        if total_markers == 0 {
            return Ok(MarkerDistributionAnalysis {
                evenness_score: 1.0,
                markers_per_section: vec![0; paragraphs.len()],
                clustering_analysis: HashMap::new(),
                distribution_quality: 0.0,
            });
        }

        let markers_per_section =
            self.calculate_markers_per_section(paragraphs.len(), markers_by_type);
        let evenness_score = self.calculate_distribution_evenness(&markers_per_section);
        let clustering_analysis = self.analyze_marker_clustering(markers_by_type);
        let distribution_quality =
            self.calculate_distribution_quality(&markers_per_section, total_markers);

        Ok(MarkerDistributionAnalysis {
            evenness_score,
            markers_per_section,
            clustering_analysis,
            distribution_quality,
        })
    }

    /// Calculate markers per section
    fn calculate_markers_per_section(
        &self,
        section_count: usize,
        markers_by_type: &HashMap<String, Vec<StructuralMarker>>,
    ) -> Vec<usize> {
        let mut markers_per_section = vec![0; section_count];

        for markers in markers_by_type.values() {
            for marker in markers {
                if marker.position < section_count {
                    markers_per_section[marker.position] += 1;
                }
            }
        }

        markers_per_section
    }

    /// Calculate distribution evenness score
    fn calculate_distribution_evenness(&self, markers_per_section: &[usize]) -> f64 {
        if markers_per_section.is_empty() {
            return 1.0;
        }

        let total_markers: usize = markers_per_section.iter().sum();
        if total_markers == 0 {
            return 1.0;
        }

        let expected_per_section = total_markers as f64 / markers_per_section.len() as f64;

        // Calculate variance from expected distribution
        let variance = markers_per_section
            .iter()
            .map(|&count| (count as f64 - expected_per_section).powi(2))
            .sum::<f64>()
            / markers_per_section.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if expected_per_section > 0.0 {
            std_dev / expected_per_section
        } else {
            0.0
        };

        // Convert to evenness score (lower variation = higher evenness)
        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }

    /// Analyze marker clustering patterns
    fn analyze_marker_clustering(
        &self,
        markers_by_type: &HashMap<String, Vec<StructuralMarker>>,
    ) -> HashMap<String, f64> {
        let mut clustering_analysis = HashMap::new();

        for (marker_type, markers) in markers_by_type {
            if markers.len() < 2 {
                clustering_analysis.insert(marker_type.clone(), 0.0);
                continue;
            }

            let positions: Vec<usize> = markers.iter().map(|m| m.position).collect();
            let clustering_score = self.calculate_clustering_score(&positions);
            clustering_analysis.insert(marker_type.clone(), clustering_score);
        }

        clustering_analysis
    }

    /// Calculate clustering score for marker positions
    fn calculate_clustering_score(&self, positions: &[usize]) -> f64 {
        if positions.len() < 2 {
            return 0.0;
        }

        let mut sorted_positions = positions.to_vec();
        sorted_positions.sort_unstable();

        // Calculate gaps between consecutive markers
        let gaps: Vec<usize> = sorted_positions
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect();

        let mean_gap = gaps.iter().sum::<usize>() as f64 / gaps.len() as f64;

        // Calculate variance in gaps
        let gap_variance = gaps
            .iter()
            .map(|&gap| (gap as f64 - mean_gap).powi(2))
            .sum::<f64>()
            / gaps.len() as f64;

        let gap_std_dev = gap_variance.sqrt();

        // Higher clustering score means more clustering (less even distribution)
        if mean_gap > 0.0 {
            gap_std_dev / mean_gap
        } else {
            0.0
        }
    }

    /// Calculate overall distribution quality
    fn calculate_distribution_quality(
        &self,
        markers_per_section: &[usize],
        total_markers: usize,
    ) -> f64 {
        if total_markers == 0 || markers_per_section.is_empty() {
            return 0.0;
        }

        // Quality is based on having markers in most sections, but not too clustered
        let sections_with_markers = markers_per_section
            .iter()
            .filter(|&&count| count > 0)
            .count();
        let coverage_score = sections_with_markers as f64 / markers_per_section.len() as f64;

        // Balance score (prefer moderate marker counts per section)
        let balance_score = self.calculate_distribution_evenness(markers_per_section);

        // Density score (prefer reasonable marker density)
        let avg_markers_per_section = total_markers as f64 / markers_per_section.len() as f64;
        let density_score = if avg_markers_per_section > 5.0 {
            0.8 // Too many markers
        } else if avg_markers_per_section > 2.0 {
            1.0 // Good marker density
        } else if avg_markers_per_section > 0.5 {
            0.9 // Adequate marker density
        } else {
            0.6 // Low marker density
        };

        // Combine scores
        (coverage_score * 0.4 + balance_score * 0.4 + density_score * 0.2)
    }

    /// Calculate effectiveness scores for each marker type
    fn calculate_marker_effectiveness(
        &self,
        paragraphs: &[String],
        markers_by_type: &HashMap<String, Vec<StructuralMarker>>,
    ) -> Result<HashMap<String, f64>, StructuralMarkerError> {
        let mut effectiveness_scores = HashMap::new();

        for (marker_type, markers) in markers_by_type {
            if markers.is_empty() {
                effectiveness_scores.insert(marker_type.clone(), 0.0);
                continue;
            }

            let total_effectiveness: f64 = markers.iter().map(|m| m.effectiveness).sum();
            let avg_effectiveness = total_effectiveness / markers.len() as f64;

            // Adjust for context and distribution
            let context_adjustment =
                self.calculate_context_effectiveness_adjustment(markers, paragraphs);
            let distribution_adjustment =
                self.calculate_distribution_effectiveness_adjustment(markers, paragraphs.len());

            let final_effectiveness =
                (avg_effectiveness * context_adjustment * distribution_adjustment).min(1.0);
            effectiveness_scores.insert(marker_type.clone(), final_effectiveness);
        }

        Ok(effectiveness_scores)
    }

    /// Calculate context-based effectiveness adjustment
    fn calculate_context_effectiveness_adjustment(
        &self,
        markers: &[StructuralMarker],
        _paragraphs: &[String],
    ) -> f64 {
        // Markers are more effective when they have good strength
        let avg_strength: f64 =
            markers.iter().map(|m| m.strength).sum::<f64>() / markers.len() as f64;

        // Convert strength to effectiveness multiplier
        0.5 + avg_strength * 0.5
    }

    /// Calculate distribution-based effectiveness adjustment
    fn calculate_distribution_effectiveness_adjustment(
        &self,
        markers: &[StructuralMarker],
        total_paragraphs: usize,
    ) -> f64 {
        if markers.is_empty() || total_paragraphs == 0 {
            return 1.0;
        }

        let positions: Vec<usize> = markers.iter().map(|m| m.position).collect();
        let coverage = positions.len() as f64 / total_paragraphs as f64;

        // Better effectiveness with good coverage, but not perfect coverage (which might indicate overuse)
        match coverage {
            c if c > 0.8 => 0.9, // Too frequent
            c if c > 0.4 => 1.1, // Good distribution
            c if c > 0.2 => 1.0, // Adequate distribution
            _ => 0.8,            // Too sparse
        }
    }

    /// Identify missing markers that could improve structure
    fn identify_missing_markers(
        &self,
        paragraphs: &[String],
    ) -> Result<Vec<String>, StructuralMarkerError> {
        let mut missing_markers = Vec::new();

        // Analyze document structure to identify missing markers
        let has_introduction = self.has_introduction_markers(paragraphs);
        let has_conclusion = self.has_conclusion_markers(paragraphs);
        let has_transitions = self.has_sufficient_transitions(paragraphs);
        let has_examples = self.has_example_markers(paragraphs);

        if !has_introduction {
            missing_markers
                .push("Introduction markers (e.g., 'First', 'To begin with')".to_string());
        }

        if !has_conclusion {
            missing_markers
                .push("Conclusion markers (e.g., 'In conclusion', 'Finally')".to_string());
        }

        if !has_transitions {
            missing_markers.push("Transition markers (e.g., 'However', 'Therefore')".to_string());
        }

        if !has_examples {
            missing_markers.push("Example markers (e.g., 'For example', 'Such as')".to_string());
        }

        // Check for section markers in longer documents
        if paragraphs.len() > 10 && !self.has_section_markers(paragraphs) {
            missing_markers.push("Section markers (e.g., 'Next', 'Another aspect')".to_string());
        }

        Ok(missing_markers)
    }

    /// Check if document has introduction markers
    fn has_introduction_markers(&self, paragraphs: &[String]) -> bool {
        if paragraphs.is_empty() {
            return false;
        }

        let first_few_paragraphs = &paragraphs[..paragraphs.len().min(3)];
        let introduction_patterns = ["first", "begin", "start", "introduce", "overview"];

        first_few_paragraphs.iter().any(|paragraph| {
            let paragraph_lower = paragraph.to_lowercase();
            introduction_patterns
                .iter()
                .any(|pattern| paragraph_lower.contains(pattern))
        })
    }

    /// Check if document has conclusion markers
    fn has_conclusion_markers(&self, paragraphs: &[String]) -> bool {
        if paragraphs.is_empty() {
            return false;
        }

        let last_few_paragraphs = &paragraphs[paragraphs.len().saturating_sub(3)..];
        let conclusion_patterns = ["conclusion", "finally", "in summary", "to conclude"];

        last_few_paragraphs.iter().any(|paragraph| {
            let paragraph_lower = paragraph.to_lowercase();
            conclusion_patterns
                .iter()
                .any(|pattern| paragraph_lower.contains(pattern))
        })
    }

    /// Check if document has sufficient transition markers
    fn has_sufficient_transitions(&self, paragraphs: &[String]) -> bool {
        let transition_patterns = [
            "however",
            "therefore",
            "moreover",
            "furthermore",
            "nevertheless",
        ];
        let transition_count = paragraphs
            .iter()
            .map(|paragraph| {
                let paragraph_lower = paragraph.to_lowercase();
                transition_patterns
                    .iter()
                    .filter(|pattern| paragraph_lower.contains(*pattern))
                    .count()
            })
            .sum::<usize>();

        // Expect at least one transition per 5 paragraphs
        transition_count >= (paragraphs.len() / 5).max(1)
    }

    /// Check if document has example markers
    fn has_example_markers(&self, paragraphs: &[String]) -> bool {
        let example_patterns = ["for example", "such as", "for instance", "including"];

        paragraphs.iter().any(|paragraph| {
            let paragraph_lower = paragraph.to_lowercase();
            example_patterns
                .iter()
                .any(|pattern| paragraph_lower.contains(pattern))
        })
    }

    /// Check if document has section markers
    fn has_section_markers(&self, paragraphs: &[String]) -> bool {
        let section_patterns = ["next", "another", "second", "third", "additionally"];
        let section_count = paragraphs
            .iter()
            .map(|paragraph| {
                let paragraph_lower = paragraph.to_lowercase();
                section_patterns
                    .iter()
                    .filter(|pattern| paragraph_lower.contains(*pattern))
                    .count()
            })
            .sum::<usize>();

        // Expect at least one section marker per 8 paragraphs
        section_count >= (paragraphs.len() / 8).max(1)
    }

    /// Build marker patterns for each type
    fn build_marker_patterns() -> HashMap<StructuralMarkerType, Vec<String>> {
        let mut patterns = HashMap::new();

        patterns.insert(
            StructuralMarkerType::Introduction,
            vec![
                "first".to_string(),
                "to begin with".to_string(),
                "initially".to_string(),
                "introduction".to_string(),
                "starting with".to_string(),
                "first of all".to_string(),
                "at the outset".to_string(),
                "to start".to_string(),
            ],
        );

        patterns.insert(
            StructuralMarkerType::Transition,
            vec![
                "however".to_string(),
                "therefore".to_string(),
                "moreover".to_string(),
                "furthermore".to_string(),
                "nevertheless".to_string(),
                "consequently".to_string(),
                "in addition".to_string(),
                "on the other hand".to_string(),
                "meanwhile".to_string(),
                "subsequently".to_string(),
                "alternatively".to_string(),
            ],
        );

        patterns.insert(
            StructuralMarkerType::Conclusion,
            vec![
                "in conclusion".to_string(),
                "finally".to_string(),
                "in summary".to_string(),
                "to conclude".to_string(),
                "overall".to_string(),
                "in the end".to_string(),
                "ultimately".to_string(),
                "to sum up".to_string(),
            ],
        );

        patterns.insert(
            StructuralMarkerType::Enumeration,
            vec![
                "first".to_string(),
                "second".to_string(),
                "third".to_string(),
                "next".to_string(),
                "then".to_string(),
                "finally".to_string(),
                "1.".to_string(),
                "2.".to_string(),
                "3.".to_string(),
                "firstly".to_string(),
                "secondly".to_string(),
                "thirdly".to_string(),
            ],
        );

        patterns.insert(
            StructuralMarkerType::Example,
            vec![
                "for example".to_string(),
                "for instance".to_string(),
                "such as".to_string(),
                "including".to_string(),
                "namely".to_string(),
                "specifically".to_string(),
                "to illustrate".to_string(),
                "as an example".to_string(),
            ],
        );

        patterns.insert(
            StructuralMarkerType::Emphasis,
            vec![
                "importantly".to_string(),
                "significantly".to_string(),
                "notably".to_string(),
                "particularly".to_string(),
                "especially".to_string(),
                "crucially".to_string(),
                "above all".to_string(),
                "most importantly".to_string(),
            ],
        );

        patterns.insert(
            StructuralMarkerType::Reference,
            vec![
                "as mentioned".to_string(),
                "as discussed".to_string(),
                "previously".to_string(),
                "earlier".to_string(),
                "as noted".to_string(),
                "as stated".to_string(),
                "referring to".to_string(),
                "according to".to_string(),
            ],
        );

        patterns.insert(
            StructuralMarkerType::Section,
            vec![
                "section".to_string(),
                "chapter".to_string(),
                "part".to_string(),
                "another aspect".to_string(),
                "turning to".to_string(),
                "moving on".to_string(),
                "the next".to_string(),
                "additionally".to_string(),
            ],
        );

        patterns
    }

    /// Build effectiveness rules for each marker type
    fn build_effectiveness_rules() -> HashMap<StructuralMarkerType, f64> {
        let mut rules = HashMap::new();

        rules.insert(StructuralMarkerType::Introduction, 0.9);
        rules.insert(StructuralMarkerType::Transition, 0.8);
        rules.insert(StructuralMarkerType::Conclusion, 0.9);
        rules.insert(StructuralMarkerType::Enumeration, 0.7);
        rules.insert(StructuralMarkerType::Example, 0.6);
        rules.insert(StructuralMarkerType::Emphasis, 0.7);
        rules.insert(StructuralMarkerType::Reference, 0.6);
        rules.insert(StructuralMarkerType::Section, 0.8);

        rules
    }
}
