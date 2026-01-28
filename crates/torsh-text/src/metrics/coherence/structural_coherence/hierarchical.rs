//! Hierarchical structure analysis for structural coherence
//!
//! This module provides comprehensive hierarchical structure analysis including
//! level detection, transition analysis, structural tree building, and balance scoring.

use crate::metrics::coherence::structural_coherence::{
    config::{HierarchicalAnalysisConfig, HierarchicalLevel},
    results::{
        HierarchicalStructureAnalysis, LevelTransition, NodeMetrics, StructuralNode, StructuralTree,
    },
};
use std::collections::HashMap;
use thiserror::Error;

/// Errors specific to hierarchical analysis
#[derive(Debug, Error)]
pub enum HierarchicalAnalysisError {
    #[error("Invalid hierarchical configuration: {0}")]
    InvalidConfiguration(String),
    #[error("Level detection failed: {0}")]
    LevelDetectionError(String),
    #[error("Tree construction failed: {0}")]
    TreeConstructionError(String),
    #[error("Insufficient content for hierarchical analysis")]
    InsufficientContent,
}

/// Hierarchical structure analyzer
pub struct HierarchicalAnalyzer {
    config: HierarchicalAnalysisConfig,
    level_patterns: HashMap<HierarchicalLevel, Vec<String>>,
    transition_rules: HashMap<(HierarchicalLevel, HierarchicalLevel), f64>,
}

impl HierarchicalAnalyzer {
    /// Create a new hierarchical analyzer
    pub fn new(config: HierarchicalAnalysisConfig) -> Self {
        Self {
            config,
            level_patterns: Self::build_level_patterns(),
            transition_rules: Self::build_transition_rules(),
        }
    }

    /// Analyze hierarchical structure of paragraphs
    pub fn analyze_hierarchical_structure(
        &self,
        paragraphs: &[String],
    ) -> Result<HierarchicalStructureAnalysis, HierarchicalAnalysisError> {
        if !self.config.enable_analysis {
            return Ok(HierarchicalStructureAnalysis::default());
        }

        if paragraphs.len() < 2 {
            return Err(HierarchicalAnalysisError::InsufficientContent);
        }

        let detected_levels = self.detect_hierarchical_levels(paragraphs)?;
        let level_transitions = self.analyze_level_transitions(&detected_levels)?;
        let balance_score = self.calculate_hierarchical_balance(&detected_levels);
        let depth_distribution = self.calculate_depth_distribution(&detected_levels);
        let consistency_score = self.calculate_hierarchical_consistency(&level_transitions);

        let structural_tree = if self.config.generate_structural_tree {
            Some(self.build_structural_tree(paragraphs, &detected_levels)?)
        } else {
            None
        };

        Ok(HierarchicalStructureAnalysis {
            detected_levels,
            level_transitions,
            balance_score,
            depth_distribution,
            structural_tree,
            consistency_score,
        })
    }

    /// Detect hierarchical levels in paragraphs
    fn detect_hierarchical_levels(
        &self,
        paragraphs: &[String],
    ) -> Result<Vec<HierarchicalLevel>, HierarchicalAnalysisError> {
        let mut levels = Vec::new();

        for paragraph in paragraphs {
            let level = self.determine_hierarchical_level(paragraph);
            levels.push(level);
        }

        // Validate level sequence
        self.validate_level_sequence(&levels)?;

        Ok(levels)
    }

    /// Determine the hierarchical level of a single paragraph
    fn determine_hierarchical_level(&self, paragraph: &str) -> HierarchicalLevel {
        let paragraph_lower = paragraph.to_lowercase();

        // Check for document-level indicators
        if self.is_document_level(&paragraph_lower) {
            return HierarchicalLevel::Document;
        }

        // Check for chapter-level indicators
        if self.is_chapter_level(&paragraph_lower) {
            return HierarchicalLevel::Chapter;
        }

        // Check for section-level indicators
        if self.is_section_level(&paragraph_lower) {
            return HierarchicalLevel::Section;
        }

        // Check for subsection-level indicators
        if self.is_subsection_level(&paragraph_lower) {
            return HierarchicalLevel::Subsection;
        }

        // Check sentence-level content
        if paragraph
            .split('.')
            .filter(|s| !s.trim().is_empty())
            .count()
            <= 1
        {
            return HierarchicalLevel::Sentence;
        }

        // Default to paragraph level
        HierarchicalLevel::Paragraph
    }

    /// Check if paragraph represents document level
    fn is_document_level(&self, paragraph: &str) -> bool {
        let document_indicators = [
            "abstract",
            "summary",
            "title",
            "introduction",
            "conclusion",
            "executive summary",
            "overview",
            "executive overview",
        ];

        document_indicators.iter().any(|indicator| {
            paragraph.starts_with(indicator) || paragraph.contains(&format!("{}:", indicator))
        })
    }

    /// Check if paragraph represents chapter level
    fn is_chapter_level(&self, paragraph: &str) -> bool {
        let chapter_patterns = [
            r"chapter \d+",
            r"part \d+",
            r"\d+\.",
            r"\d+\s+\w+",
            "methodology",
            "results",
            "discussion",
            "literature review",
        ];

        chapter_patterns.iter().any(|pattern| {
            paragraph.contains(pattern)
                || (paragraph.len() < 100 && paragraph.split_whitespace().count() < 10)
        })
    }

    /// Check if paragraph represents section level
    fn is_section_level(&self, paragraph: &str) -> bool {
        let section_patterns = [
            r"\d+\.\d+",
            r"section \d+",
            r"part \w+",
            "background",
            "methods",
            "analysis",
            "findings",
        ];

        section_patterns
            .iter()
            .any(|pattern| paragraph.contains(pattern))
            || (paragraph.len() < 200 && paragraph.ends_with(':'))
    }

    /// Check if paragraph represents subsection level
    fn is_subsection_level(&self, paragraph: &str) -> bool {
        let subsection_patterns = [
            r"\d+\.\d+\.\d+",
            r"subsection",
            r"\w+\s*:",
            "data collection",
            "data analysis",
            "participants",
        ];

        subsection_patterns
            .iter()
            .any(|pattern| paragraph.contains(pattern))
            || (paragraph.len() < 150 && paragraph.split(':').count() > 1)
    }

    /// Validate the sequence of hierarchical levels
    fn validate_level_sequence(
        &self,
        levels: &[HierarchicalLevel],
    ) -> Result<(), HierarchicalAnalysisError> {
        for window in levels.windows(2) {
            let from_level = &window[0];
            let to_level = &window[1];

            if !self.is_valid_transition(from_level, to_level) {
                let confidence = self
                    .transition_rules
                    .get(&(from_level.clone(), to_level.clone()))
                    .unwrap_or(&0.0);

                if *confidence < self.config.min_level_confidence {
                    return Err(HierarchicalAnalysisError::LevelDetectionError(format!(
                        "Invalid transition from {:?} to {:?}",
                        from_level, to_level
                    )));
                }
            }
        }

        Ok(())
    }

    /// Check if a level transition is valid
    fn is_valid_transition(&self, from: &HierarchicalLevel, to: &HierarchicalLevel) -> bool {
        let from_depth = self.hierarchical_level_to_number(from);
        let to_depth = self.hierarchical_level_to_number(to);

        // Allow transitions within reasonable depth differences
        (to_depth as i32 - from_depth as i32).abs() <= 2
    }

    /// Convert hierarchical level to numeric depth
    fn hierarchical_level_to_number(&self, level: &HierarchicalLevel) -> usize {
        match level {
            HierarchicalLevel::Document => 0,
            HierarchicalLevel::Chapter => 1,
            HierarchicalLevel::Section => 2,
            HierarchicalLevel::Subsection => 3,
            HierarchicalLevel::Paragraph => 4,
            HierarchicalLevel::Sentence => 5,
            HierarchicalLevel::Phrase => 6,
        }
    }

    /// Analyze transitions between hierarchical levels
    fn analyze_level_transitions(
        &self,
        levels: &[HierarchicalLevel],
    ) -> Result<Vec<LevelTransition>, HierarchicalAnalysisError> {
        let mut transitions = Vec::new();

        for (i, window) in levels.windows(2).enumerate() {
            let from_level = window[0].clone();
            let to_level = window[1].clone();

            let quality = self.calculate_transition_quality(&from_level, &to_level);
            let appropriateness = self.calculate_transition_appropriateness(&from_level, &to_level);

            transitions.push(LevelTransition {
                from_level,
                to_level,
                position: i,
                quality,
                appropriateness,
            });
        }

        Ok(transitions)
    }

    /// Calculate quality of a level transition
    fn calculate_transition_quality(
        &self,
        from_level: &HierarchicalLevel,
        to_level: &HierarchicalLevel,
    ) -> f64 {
        self.transition_rules
            .get(&(from_level.clone(), to_level.clone()))
            .copied()
            .unwrap_or_else(|| {
                // Calculate based on level distance
                let from_depth = self.hierarchical_level_to_number(from_level);
                let to_depth = self.hierarchical_level_to_number(to_level);
                let distance = (to_depth as i32 - from_depth as i32).abs() as f64;

                // Closer transitions are higher quality
                (3.0 - distance.min(3.0)) / 3.0
            })
    }

    /// Calculate appropriateness of a level transition
    fn calculate_transition_appropriateness(
        &self,
        from_level: &HierarchicalLevel,
        to_level: &HierarchicalLevel,
    ) -> f64 {
        let from_depth = self.hierarchical_level_to_number(from_level);
        let to_depth = self.hierarchical_level_to_number(to_level);

        match to_depth as i32 - from_depth as i32 {
            0 => 1.0,  // Same level - perfectly appropriate
            1 => 0.9,  // Going one level deeper - very appropriate
            -1 => 0.8, // Going one level up - appropriate
            2 => 0.6,  // Skipping one level down - moderately appropriate
            -2 => 0.5, // Skipping one level up - less appropriate
            distance => {
                // Large jumps - inappropriate
                (5.0 - distance.abs() as f64).max(0.0) / 5.0
            }
        }
    }

    /// Calculate hierarchical balance score
    fn calculate_hierarchical_balance(&self, levels: &[HierarchicalLevel]) -> f64 {
        if levels.is_empty() {
            return 0.0;
        }

        let mut level_counts = HashMap::new();
        for level in levels {
            *level_counts.entry(level.clone()).or_insert(0) += 1;
        }

        // Calculate entropy-based balance
        let total = levels.len() as f64;
        let mut entropy = 0.0;

        for count in level_counts.values() {
            let p = *count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        // Normalize by maximum possible entropy
        let max_entropy = (level_counts.len() as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Calculate depth distribution
    fn calculate_depth_distribution(&self, levels: &[HierarchicalLevel]) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for level in levels {
            let level_name = format!("{:?}", level);
            *distribution.entry(level_name).or_insert(0) += 1;
        }

        distribution
    }

    /// Calculate hierarchical consistency score
    fn calculate_hierarchical_consistency(&self, transitions: &[LevelTransition]) -> f64 {
        if transitions.is_empty() {
            return 1.0;
        }

        let total_quality: f64 = transitions
            .iter()
            .map(|t| t.quality * t.appropriateness)
            .sum();

        total_quality / transitions.len() as f64
    }

    /// Build structural tree representation
    fn build_structural_tree(
        &self,
        paragraphs: &[String],
        levels: &[HierarchicalLevel],
    ) -> Result<StructuralTree, HierarchicalAnalysisError> {
        if paragraphs.len() != levels.len() {
            return Err(HierarchicalAnalysisError::TreeConstructionError(
                "Paragraph count doesn't match level count".to_string(),
            ));
        }

        let root = self.build_tree_recursive(paragraphs, levels, 0, 0)?;
        let depth = self.calculate_tree_depth(&root);
        let node_count = self.calculate_node_count(&root);
        let balance_score = self.calculate_tree_balance(&root);

        Ok(StructuralTree {
            root,
            depth,
            node_count,
            balance_score,
        })
    }

    /// Recursively build structural tree
    fn build_tree_recursive(
        &self,
        paragraphs: &[String],
        levels: &[HierarchicalLevel],
        start_index: usize,
        node_id: usize,
    ) -> Result<StructuralNode, HierarchicalAnalysisError> {
        if start_index >= paragraphs.len() {
            return Err(HierarchicalAnalysisError::TreeConstructionError(
                "Invalid start index for tree construction".to_string(),
            ));
        }

        let current_level = levels[start_index].clone();
        let current_depth = self.hierarchical_level_to_number(&current_level);

        // Find the end of this section
        let mut end_index = start_index + 1;
        while end_index < levels.len() {
            let next_depth = self.hierarchical_level_to_number(&levels[end_index]);
            if next_depth <= current_depth {
                break;
            }
            end_index += 1;
        }

        let content_span = (start_index, end_index.saturating_sub(1));
        let title = self.extract_title_from_paragraph(&paragraphs[start_index]);

        // Build child nodes
        let mut children = Vec::new();
        let mut child_index = start_index + 1;
        let mut child_node_id = node_id + 1;

        while child_index < end_index {
            let child_depth = self.hierarchical_level_to_number(&levels[child_index]);

            // Only create child nodes for immediate children
            if child_depth == current_depth + 1 {
                let child =
                    self.build_tree_recursive(paragraphs, levels, child_index, child_node_id)?;

                // Skip to next sibling
                child_index = self.find_next_sibling(levels, child_index, child_depth);
                child_node_id = self.get_next_node_id(&child) + 1;
                children.push(child);
            } else {
                child_index += 1;
            }
        }

        let metrics = self.calculate_node_metrics(paragraphs, &content_span, &children);

        Ok(StructuralNode {
            node_id,
            level: current_level,
            content_span,
            title,
            children,
            metrics,
        })
    }

    /// Find the index of the next sibling at the same depth
    fn find_next_sibling(
        &self,
        levels: &[HierarchicalLevel],
        current_index: usize,
        target_depth: usize,
    ) -> usize {
        let mut index = current_index + 1;

        while index < levels.len() {
            let depth = self.hierarchical_level_to_number(&levels[index]);
            if depth <= target_depth {
                return index;
            }
            index += 1;
        }

        levels.len()
    }

    /// Get the maximum node ID from a subtree
    fn get_next_node_id(&self, node: &StructuralNode) -> usize {
        let mut max_id = node.node_id;

        for child in &node.children {
            max_id = max_id.max(self.get_next_node_id(child));
        }

        max_id
    }

    /// Extract title from paragraph
    fn extract_title_from_paragraph(&self, paragraph: &str) -> Option<String> {
        // Look for title patterns
        if paragraph.len() < 200 {
            // Likely a title if short
            let trimmed = paragraph.trim();

            // Remove common prefixes
            let title = trimmed
                .trim_start_matches(|c: char| c.is_digit(10) || c == '.' || c == ' ')
                .trim_end_matches(':')
                .trim();

            if !title.is_empty() && title.len() < 100 {
                return Some(title.to_string());
            }
        }

        None
    }

    /// Calculate metrics for a tree node
    fn calculate_node_metrics(
        &self,
        paragraphs: &[String],
        content_span: &(usize, usize),
        children: &[StructuralNode],
    ) -> NodeMetrics {
        let content_length = (content_span.0..=content_span.1)
            .map(|i| paragraphs.get(i).map(|p| p.len()).unwrap_or(0))
            .sum();

        let internal_coherence = self.calculate_internal_coherence(paragraphs, content_span);
        let parent_connection = 0.8; // Placeholder - would be calculated based on parent relationship
        let children_connections = children
            .iter()
            .map(|_| 0.7) // Placeholder - would be calculated based on child relationships
            .collect();

        NodeMetrics {
            content_length,
            internal_coherence,
            parent_connection,
            children_connections,
        }
    }

    /// Calculate internal coherence of a content span
    fn calculate_internal_coherence(&self, paragraphs: &[String], span: &(usize, usize)) -> f64 {
        if span.0 == span.1 {
            return 1.0; // Single paragraph has perfect internal coherence
        }

        let content: Vec<&String> = (span.0..=span.1)
            .filter_map(|i| paragraphs.get(i))
            .collect();

        if content.len() < 2 {
            return 1.0;
        }

        // Calculate pairwise coherence
        let mut total_coherence = 0.0;
        let mut pair_count = 0;

        for i in 0..content.len() {
            for j in i + 1..content.len() {
                total_coherence += self.calculate_paragraph_coherence(content[i], content[j]);
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_coherence / pair_count as f64
        } else {
            1.0
        }
    }

    /// Calculate coherence between two paragraphs
    fn calculate_paragraph_coherence(&self, para1: &str, para2: &str) -> f64 {
        // Simplified coherence calculation based on word overlap
        let words1: std::collections::HashSet<&str> = para1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = para2.split_whitespace().collect();

        let intersection_size = words1.intersection(&words2).count() as f64;
        let union_size = words1.union(&words2).count() as f64;

        if union_size > 0.0 {
            intersection_size / union_size
        } else {
            0.0
        }
    }

    /// Calculate tree depth
    fn calculate_tree_depth(&self, node: &StructuralNode) -> usize {
        if node.children.is_empty() {
            1
        } else {
            1 + node
                .children
                .iter()
                .map(|child| self.calculate_tree_depth(child))
                .max()
                .unwrap_or(0)
        }
    }

    /// Calculate total node count in tree
    fn calculate_node_count(&self, node: &StructuralNode) -> usize {
        1 + node
            .children
            .iter()
            .map(|child| self.calculate_node_count(child))
            .sum::<usize>()
    }

    /// Calculate tree balance score
    fn calculate_tree_balance(&self, node: &StructuralNode) -> f64 {
        if node.children.is_empty() {
            return 1.0;
        }

        // Calculate balance based on child distribution
        let child_counts: Vec<usize> = node
            .children
            .iter()
            .map(|child| self.calculate_node_count(child))
            .collect();

        if child_counts.is_empty() {
            return 1.0;
        }

        let max_count = *child_counts.iter().max().expect("reduction should succeed") as f64;
        let min_count = *child_counts.iter().min().expect("reduction should succeed") as f64;

        // Balance score is how evenly distributed the children are
        let balance = if max_count > 0.0 {
            min_count / max_count
        } else {
            1.0
        };

        // Recursively calculate balance for children
        let child_balance_sum: f64 = node
            .children
            .iter()
            .map(|child| self.calculate_tree_balance(child))
            .sum();

        let child_balance_avg = if !node.children.is_empty() {
            child_balance_sum / node.children.len() as f64
        } else {
            1.0
        };

        // Combine current level balance with children balance
        (balance + child_balance_avg) / 2.0
    }

    /// Build level detection patterns
    fn build_level_patterns() -> HashMap<HierarchicalLevel, Vec<String>> {
        let mut patterns = HashMap::new();

        patterns.insert(
            HierarchicalLevel::Document,
            vec![
                "title".to_string(),
                "abstract".to_string(),
                "summary".to_string(),
                "introduction".to_string(),
                "conclusion".to_string(),
            ],
        );

        patterns.insert(
            HierarchicalLevel::Chapter,
            vec![
                "chapter".to_string(),
                "part".to_string(),
                "methodology".to_string(),
                "results".to_string(),
                "discussion".to_string(),
            ],
        );

        patterns.insert(
            HierarchicalLevel::Section,
            vec![
                "section".to_string(),
                "background".to_string(),
                "methods".to_string(),
                "analysis".to_string(),
                "findings".to_string(),
            ],
        );

        patterns.insert(
            HierarchicalLevel::Subsection,
            vec![
                "subsection".to_string(),
                "data collection".to_string(),
                "data analysis".to_string(),
                "participants".to_string(),
            ],
        );

        patterns
    }

    /// Build transition quality rules
    fn build_transition_rules() -> HashMap<(HierarchicalLevel, HierarchicalLevel), f64> {
        let mut rules = HashMap::new();

        // Same level transitions
        rules.insert(
            (HierarchicalLevel::Document, HierarchicalLevel::Document),
            0.9,
        );
        rules.insert(
            (HierarchicalLevel::Chapter, HierarchicalLevel::Chapter),
            0.9,
        );
        rules.insert(
            (HierarchicalLevel::Section, HierarchicalLevel::Section),
            0.9,
        );
        rules.insert(
            (HierarchicalLevel::Subsection, HierarchicalLevel::Subsection),
            0.9,
        );
        rules.insert(
            (HierarchicalLevel::Paragraph, HierarchicalLevel::Paragraph),
            1.0,
        );

        // Natural downward transitions
        rules.insert(
            (HierarchicalLevel::Document, HierarchicalLevel::Chapter),
            0.95,
        );
        rules.insert(
            (HierarchicalLevel::Chapter, HierarchicalLevel::Section),
            0.95,
        );
        rules.insert(
            (HierarchicalLevel::Section, HierarchicalLevel::Subsection),
            0.95,
        );
        rules.insert(
            (HierarchicalLevel::Subsection, HierarchicalLevel::Paragraph),
            0.9,
        );
        rules.insert(
            (HierarchicalLevel::Paragraph, HierarchicalLevel::Sentence),
            0.8,
        );

        // Natural upward transitions
        rules.insert(
            (HierarchicalLevel::Chapter, HierarchicalLevel::Document),
            0.8,
        );
        rules.insert(
            (HierarchicalLevel::Section, HierarchicalLevel::Chapter),
            0.8,
        );
        rules.insert(
            (HierarchicalLevel::Subsection, HierarchicalLevel::Section),
            0.8,
        );
        rules.insert(
            (HierarchicalLevel::Paragraph, HierarchicalLevel::Subsection),
            0.7,
        );

        // Skip-level transitions (less common but acceptable)
        rules.insert(
            (HierarchicalLevel::Document, HierarchicalLevel::Section),
            0.6,
        );
        rules.insert(
            (HierarchicalLevel::Chapter, HierarchicalLevel::Subsection),
            0.6,
        );
        rules.insert(
            (HierarchicalLevel::Section, HierarchicalLevel::Paragraph),
            0.7,
        );

        rules
    }
}
