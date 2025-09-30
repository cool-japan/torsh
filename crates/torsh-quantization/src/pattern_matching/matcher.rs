//! Pattern Matching Engine
//!
//! This module provides the core pattern matching engine for computational graphs.
//! It includes algorithms for finding pattern matches, managing pattern collections,
//! and optimizing the matching process for performance.

use super::graph::{ComputationGraph, GraphNode};
use super::patterns::{CommonPatterns, GraphPattern, PatternCollection, PatternNode};
use crate::{QuantConfig, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use torsh_core::TorshError;

// =============================================================================
// Pattern Match Result
// =============================================================================

/// Represents a successful pattern match
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    /// Name of the matched pattern
    pub pattern_name: String,
    /// IDs of nodes that matched the pattern
    pub matched_node_ids: Vec<String>,
    /// Quantization config associated with this pattern
    pub qconfig: Option<QuantConfig>,
    /// Confidence score of the match (0.0 to 1.0)
    pub confidence: f64,
    /// Metadata about the match
    pub metadata: HashMap<String, String>,
}

impl PatternMatch {
    /// Create a new pattern match
    pub fn new(pattern_name: String, matched_node_ids: Vec<String>) -> Self {
        Self {
            pattern_name,
            matched_node_ids,
            qconfig: None,
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Create a pattern match with quantization config
    pub fn with_qconfig(mut self, qconfig: QuantConfig) -> Self {
        self.qconfig = Some(qconfig);
        self
    }

    /// Set the confidence score
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get the number of nodes in this match
    pub fn node_count(&self) -> usize {
        self.matched_node_ids.len()
    }

    /// Check if this match overlaps with another match
    pub fn overlaps_with(&self, other: &PatternMatch) -> bool {
        let self_set: HashSet<&String> = self.matched_node_ids.iter().collect();
        let other_set: HashSet<&String> = other.matched_node_ids.iter().collect();

        !self_set.is_disjoint(&other_set)
    }

    /// Get the overlap count with another match
    pub fn overlap_count(&self, other: &PatternMatch) -> usize {
        let self_set: HashSet<&String> = self.matched_node_ids.iter().collect();
        let other_set: HashSet<&String> = other.matched_node_ids.iter().collect();

        self_set.intersection(&other_set).count()
    }

    /// Check if this match contains all nodes of another match
    pub fn contains(&self, other: &PatternMatch) -> bool {
        let self_set: HashSet<&String> = self.matched_node_ids.iter().collect();
        other
            .matched_node_ids
            .iter()
            .all(|node_id| self_set.contains(node_id))
    }

    /// Get a textual representation of the match
    pub fn to_string(&self) -> String {
        format!(
            "Match '{}': {} nodes [{}] (confidence: {:.2})",
            self.pattern_name,
            self.node_count(),
            self.matched_node_ids.join(", "),
            self.confidence
        )
    }
}

// =============================================================================
// Pattern Matching Engine
// =============================================================================

/// Advanced pattern matching engine with configurable algorithms
#[derive(Debug)]
pub struct PatternMatcher {
    /// Available patterns to match
    patterns: Vec<GraphPattern>,
    /// Matching configuration
    config: MatchingConfig,
    /// Pattern cache for performance
    pattern_cache: HashMap<String, Vec<PatternMatch>>,
}

/// Configuration for pattern matching behavior
#[derive(Debug, Clone)]
pub struct MatchingConfig {
    /// Maximum recursion depth for pattern matching
    pub max_depth: usize,
    /// Whether to enable overlap detection
    pub detect_overlaps: bool,
    /// Minimum confidence threshold for matches
    pub min_confidence: f64,
    /// Whether to cache pattern matching results
    pub enable_caching: bool,
    /// Maximum number of matches to return per pattern
    pub max_matches_per_pattern: usize,
}

impl Default for MatchingConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            detect_overlaps: true,
            min_confidence: 0.8,
            enable_caching: true,
            max_matches_per_pattern: 100,
        }
    }
}

impl PatternMatcher {
    /// Create a new pattern matcher with default patterns
    pub fn new() -> Self {
        let mut matcher = Self {
            patterns: Vec::new(),
            config: MatchingConfig::default(),
            pattern_cache: HashMap::new(),
        };

        // Add default patterns from common patterns
        for pattern in CommonPatterns::all_patterns() {
            matcher.add_pattern(pattern);
        }

        matcher
    }

    /// Create a pattern matcher with custom configuration
    pub fn with_config(config: MatchingConfig) -> Self {
        Self {
            patterns: Vec::new(),
            config,
            pattern_cache: HashMap::new(),
        }
    }

    /// Create a pattern matcher from a pattern collection
    pub fn from_collection(collection: PatternCollection) -> Self {
        let mut matcher = Self::new();
        matcher.patterns = collection.patterns;
        matcher
    }

    /// Add a pattern to the matcher
    pub fn add_pattern(&mut self, pattern: GraphPattern) {
        self.patterns.push(pattern);
        // Clear cache since patterns changed
        self.pattern_cache.clear();
    }

    /// Add multiple patterns
    pub fn add_patterns(&mut self, patterns: Vec<GraphPattern>) {
        self.patterns.extend(patterns);
        self.pattern_cache.clear();
    }

    /// Set the matching configuration
    pub fn set_config(&mut self, config: MatchingConfig) {
        self.config = config;
        self.pattern_cache.clear();
    }

    /// Clear all patterns
    pub fn clear_patterns(&mut self) {
        self.patterns.clear();
        self.pattern_cache.clear();
    }

    /// Find all pattern matches in a graph
    pub fn find_matches(&mut self, graph: &ComputationGraph) -> TorshResult<Vec<PatternMatch>> {
        // Check cache first
        let graph_key = self.compute_graph_key(graph);
        if self.config.enable_caching {
            if let Some(cached_matches) = self.pattern_cache.get(&graph_key) {
                return Ok(cached_matches.clone());
            }
        }

        let mut all_matches = Vec::new();

        // Sort patterns by priority (higher priority first)
        let mut sorted_patterns: Vec<&GraphPattern> = self.patterns.iter().collect();
        sorted_patterns.sort_by(|a, b| b.priority.cmp(&a.priority));

        for pattern in sorted_patterns {
            let pattern_matches = self.find_pattern_matches(graph, pattern)?;
            all_matches.extend(pattern_matches);
        }

        // Filter by confidence and remove overlaps if enabled
        let filtered_matches = self.filter_matches(all_matches)?;

        // Cache the results
        if self.config.enable_caching {
            self.pattern_cache
                .insert(graph_key, filtered_matches.clone());
        }

        Ok(filtered_matches)
    }

    /// Find matches for a specific pattern
    pub fn find_pattern_matches(
        &self,
        graph: &ComputationGraph,
        pattern: &GraphPattern,
    ) -> TorshResult<Vec<PatternMatch>> {
        let mut matches = Vec::new();
        let mut match_count = 0;

        // Try to match the pattern starting from each node in the graph
        for start_node_id in graph.nodes.keys() {
            if match_count >= self.config.max_matches_per_pattern {
                break;
            }

            if let Some(pattern_match) =
                self.try_match_pattern_at_node(graph, pattern, start_node_id)?
            {
                matches.push(pattern_match);
                match_count += 1;
            }
        }

        Ok(matches)
    }

    /// Try to match a pattern starting at a specific node
    fn try_match_pattern_at_node(
        &self,
        graph: &ComputationGraph,
        pattern: &GraphPattern,
        start_node_id: &str,
    ) -> TorshResult<Option<PatternMatch>> {
        if pattern.nodes.is_empty() {
            return Ok(None);
        }

        let start_node = graph
            .get_node(start_node_id)
            .ok_or_else(|| TorshError::InvalidArgument("Start node not found".to_string()))?;

        // Check if the start node matches the first pattern node
        if !self.node_matches_pattern(start_node, &pattern.nodes[0]) {
            return Ok(None);
        }

        // Try to match the rest of the pattern
        let mut matched_nodes = vec![start_node_id.to_string()];
        let mut visited = HashSet::new();
        visited.insert(start_node_id.to_string());

        let match_result =
            self.match_remaining_pattern(graph, pattern, &mut matched_nodes, &mut visited, 1, 0)?;

        if match_result {
            let confidence = self.compute_match_confidence(graph, pattern, &matched_nodes);

            if confidence >= self.config.min_confidence {
                let mut pattern_match = PatternMatch::new(pattern.name.clone(), matched_nodes)
                    .with_confidence(confidence);

                if let Some(ref qconfig) = pattern.qconfig {
                    pattern_match = pattern_match.with_qconfig(qconfig.clone());
                }

                // Add metadata
                pattern_match = pattern_match
                    .with_metadata(
                        "pattern_type".to_string(),
                        if pattern.iterative {
                            "iterative".to_string()
                        } else {
                            "single".to_string()
                        },
                    )
                    .with_metadata("priority".to_string(), pattern.priority.to_string());

                Ok(Some(pattern_match))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Recursively match the remaining part of a pattern
    fn match_remaining_pattern(
        &self,
        graph: &ComputationGraph,
        pattern: &GraphPattern,
        matched_nodes: &mut Vec<String>,
        visited: &mut HashSet<String>,
        pattern_index: usize,
        depth: usize,
    ) -> TorshResult<bool> {
        if depth > self.config.max_depth {
            return Ok(false); // Prevent infinite recursion
        }

        if pattern_index >= pattern.nodes.len() {
            return Ok(true); // All pattern nodes matched
        }

        let current_pattern_node = &pattern.nodes[pattern_index];

        // Find edges in the pattern that lead to the current pattern node
        let incoming_edges: Vec<usize> = pattern
            .edges
            .iter()
            .filter(|(_, to)| *to == pattern_index)
            .map(|(from, _)| *from)
            .collect();

        // For each incoming edge, try to find a matching node in the graph
        for &from_pattern_index in &incoming_edges {
            if from_pattern_index < matched_nodes.len() {
                let from_node_id = &matched_nodes[from_pattern_index];
                let from_node = graph.get_node(from_node_id).unwrap();

                // Check all output nodes of the current matched node
                for output_id in &from_node.outputs {
                    if visited.contains(output_id) {
                        continue;
                    }

                    let output_node = graph.get_node(output_id).unwrap();

                    if self.node_matches_pattern(output_node, current_pattern_node) {
                        matched_nodes.push(output_id.clone());
                        visited.insert(output_id.clone());

                        if self.match_remaining_pattern(
                            graph,
                            pattern,
                            matched_nodes,
                            visited,
                            pattern_index + 1,
                            depth + 1,
                        )? {
                            return Ok(true);
                        }

                        // Backtrack
                        matched_nodes.pop();
                        visited.remove(output_id);
                    }
                }
            }
        }

        // If this pattern node is optional, try to skip it
        if current_pattern_node.optional {
            return self.match_remaining_pattern(
                graph,
                pattern,
                matched_nodes,
                visited,
                pattern_index + 1,
                depth + 1,
            );
        }

        Ok(false)
    }

    /// Check if a graph node matches a pattern node
    fn node_matches_pattern(&self, node: &GraphNode, pattern_node: &PatternNode) -> bool {
        pattern_node.matches(&node.op_type, &node.attributes)
    }

    /// Compute a confidence score for a match
    fn compute_match_confidence(
        &self,
        graph: &ComputationGraph,
        pattern: &GraphPattern,
        matched_nodes: &[String],
    ) -> f64 {
        if matched_nodes.is_empty() || pattern.nodes.is_empty() {
            return 0.0;
        }

        // Base confidence based on the ratio of matched vs expected nodes
        let node_ratio = matched_nodes.len() as f64 / pattern.nodes.len() as f64;

        // Attribute matching bonus
        let mut attribute_score = 0.0;
        let mut total_attributes = 0;

        for (i, node_id) in matched_nodes.iter().enumerate() {
            if i < pattern.nodes.len() {
                if let Some(node) = graph.get_node(node_id) {
                    let pattern_node = &pattern.nodes[i];
                    for (key, expected_value) in &pattern_node.attributes {
                        total_attributes += 1;
                        if let Some(actual_value) = node.get_attribute(key) {
                            if actual_value == expected_value {
                                attribute_score += 1.0;
                            }
                        }
                    }
                }
            }
        }

        let attribute_ratio = if total_attributes > 0 {
            attribute_score / total_attributes as f64
        } else {
            1.0 // No attributes to match, perfect score
        };

        // Combine scores
        (node_ratio * 0.7 + attribute_ratio * 0.3).min(1.0)
    }

    /// Filter matches based on configuration
    fn filter_matches(&self, matches: Vec<PatternMatch>) -> TorshResult<Vec<PatternMatch>> {
        let mut filtered: Vec<PatternMatch> = matches
            .into_iter()
            .filter(|m| m.confidence >= self.config.min_confidence)
            .collect();

        // Remove overlaps if enabled
        if self.config.detect_overlaps {
            filtered = self.remove_overlapping_matches(filtered);
        }

        Ok(filtered)
    }

    /// Remove overlapping matches, keeping higher priority/confidence ones
    fn remove_overlapping_matches(&self, matches: Vec<PatternMatch>) -> Vec<PatternMatch> {
        let mut result = Vec::new();
        let mut sorted_matches = matches;

        // Sort by confidence (highest first)
        sorted_matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for candidate in sorted_matches {
            let has_overlap = result
                .iter()
                .any(|existing| existing.overlaps_with(&candidate));

            if !has_overlap {
                result.push(candidate);
            }
        }

        result
    }

    /// Compute a unique key for a graph (for caching)
    fn compute_graph_key(&self, graph: &ComputationGraph) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash node count and execution order
        graph.nodes.len().hash(&mut hasher);
        graph.execution_order.hash(&mut hasher);

        // Hash a subset of node information for efficiency
        for node_id in graph.execution_order.iter().take(20) {
            if let Some(node) = graph.get_node(node_id) {
                node.op_type.hash(&mut hasher);
                node.inputs.len().hash(&mut hasher);
                node.outputs.len().hash(&mut hasher);
            }
        }

        format!("graph_{:x}", hasher.finish())
    }

    /// Get available patterns
    pub fn get_patterns(&self) -> &[GraphPattern] {
        &self.patterns
    }

    /// Get matching statistics
    pub fn get_statistics(&self) -> MatcherStatistics {
        MatcherStatistics {
            pattern_count: self.patterns.len(),
            cache_size: self.pattern_cache.len(),
            cache_enabled: self.config.enable_caching,
        }
    }

    /// Clear the pattern cache
    pub fn clear_cache(&mut self) {
        self.pattern_cache.clear();
    }
}

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Matcher Statistics
// =============================================================================

/// Statistics about the pattern matcher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatcherStatistics {
    /// Number of patterns loaded
    pub pattern_count: usize,
    /// Size of the pattern cache
    pub cache_size: usize,
    /// Whether caching is enabled
    pub cache_enabled: bool,
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Create a pattern matcher optimized for fusion patterns
pub fn create_fusion_matcher() -> PatternMatcher {
    PatternMatcher::from_collection(PatternCollection::fusion_only())
}

/// Create a pattern matcher optimized for elimination patterns
pub fn create_elimination_matcher() -> PatternMatcher {
    PatternMatcher::from_collection(PatternCollection::elimination_only())
}

/// Create a high-performance pattern matcher
pub fn create_performance_matcher() -> PatternMatcher {
    let config = MatchingConfig {
        max_depth: 5,
        detect_overlaps: true,
        min_confidence: 0.9,
        enable_caching: true,
        max_matches_per_pattern: 50,
    };

    PatternMatcher::with_config(config)
}

/// Find and rank pattern matches by confidence
pub fn find_ranked_matches(
    matcher: &mut PatternMatcher,
    graph: &ComputationGraph,
) -> TorshResult<Vec<PatternMatch>> {
    let mut matches = matcher.find_matches(graph)?;
    matches.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern_matching::graph::{create_branching_graph, create_linear_graph};

    #[test]
    fn test_pattern_match_creation() {
        let nodes = vec!["node1".to_string(), "node2".to_string()];
        let match_result = PatternMatch::new("test_pattern".to_string(), nodes)
            .with_confidence(0.95)
            .with_metadata("type".to_string(), "fusion".to_string());

        assert_eq!(match_result.pattern_name, "test_pattern");
        assert_eq!(match_result.node_count(), 2);
        assert_eq!(match_result.confidence, 0.95);
        assert!(match_result.metadata.contains_key("type"));
    }

    #[test]
    fn test_pattern_match_overlap() {
        let match1 = PatternMatch::new(
            "pattern1".to_string(),
            vec!["a".to_string(), "b".to_string()],
        );
        let match2 = PatternMatch::new(
            "pattern2".to_string(),
            vec!["b".to_string(), "c".to_string()],
        );
        let match3 = PatternMatch::new(
            "pattern3".to_string(),
            vec!["d".to_string(), "e".to_string()],
        );

        assert!(match1.overlaps_with(&match2));
        assert!(!match1.overlaps_with(&match3));
        assert_eq!(match1.overlap_count(&match2), 1);
        assert_eq!(match1.overlap_count(&match3), 0);
    }

    #[test]
    fn test_pattern_matcher_creation() {
        let matcher = PatternMatcher::new();
        assert!(!matcher.patterns.is_empty());

        let stats = matcher.get_statistics();
        assert!(stats.pattern_count > 0);
        assert!(stats.cache_enabled);
    }

    #[test]
    fn test_pattern_matching_with_linear_graph() {
        let graph = create_linear_graph(&["conv2d", "relu", "pool"]);
        let mut matcher = PatternMatcher::new();

        let matches = matcher.find_matches(&graph).unwrap();

        // Should find conv_relu pattern
        let conv_relu_matches: Vec<&PatternMatch> = matches
            .iter()
            .filter(|m| m.pattern_name.contains("conv_relu"))
            .collect();

        assert!(!conv_relu_matches.is_empty());
    }

    #[test]
    fn test_custom_matching_config() {
        let config = MatchingConfig {
            max_depth: 3,
            detect_overlaps: false,
            min_confidence: 0.5,
            enable_caching: false,
            max_matches_per_pattern: 10,
        };

        let matcher = PatternMatcher::with_config(config);
        let stats = matcher.get_statistics();
        assert!(!stats.cache_enabled);
    }

    #[test]
    fn test_confidence_scoring() {
        let mut matcher = PatternMatcher::new();
        let graph = create_linear_graph(&["conv2d", "relu"]);

        let matches = matcher.find_matches(&graph).unwrap();
        for match_result in matches {
            assert!(match_result.confidence >= 0.0 && match_result.confidence <= 1.0);
        }
    }

    #[test]
    fn test_overlap_removal() {
        let match1 = PatternMatch::new(
            "pattern1".to_string(),
            vec!["a".to_string(), "b".to_string()],
        )
        .with_confidence(0.9);
        let match2 = PatternMatch::new(
            "pattern2".to_string(),
            vec!["b".to_string(), "c".to_string()],
        )
        .with_confidence(0.8);
        let match3 = PatternMatch::new(
            "pattern3".to_string(),
            vec!["d".to_string(), "e".to_string()],
        )
        .with_confidence(0.7);

        let matcher = PatternMatcher::new();
        let filtered = matcher.remove_overlapping_matches(vec![match1, match2, match3]);

        // Should keep match1 (highest confidence) and match3 (no overlap)
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().any(|m| m.pattern_name == "pattern1"));
        assert!(filtered.iter().any(|m| m.pattern_name == "pattern3"));
        assert!(!filtered.iter().any(|m| m.pattern_name == "pattern2"));
    }

    #[test]
    fn test_specialized_matchers() {
        let fusion_matcher = create_fusion_matcher();
        let elimination_matcher = create_elimination_matcher();
        let performance_matcher = create_performance_matcher();

        assert!(!fusion_matcher.patterns.is_empty());
        assert!(!elimination_matcher.patterns.is_empty());
        assert!(!performance_matcher.patterns.is_empty());
    }
}
