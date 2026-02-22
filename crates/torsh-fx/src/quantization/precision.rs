//! Automatic precision selection for quantization

use super::context::QuantizationContext;
use super::types::{QuantizationAnnotation, QuantizationParams, QuantizationScheme};
use crate::{FxGraph, Node, TorshResult};
use petgraph::graph::NodeIndex;
use petgraph::visit::IntoNodeReferences;
use std::collections::HashMap;

/// Automatic precision selection criteria
#[derive(Debug, Clone, Copy)]
pub enum PrecisionCriteria {
    /// Maximize performance, minimal accuracy loss
    Performance,
    /// Balance performance and accuracy
    Balanced,
    /// Maximize accuracy, minimal performance loss
    Accuracy,
    /// Custom criteria with specified thresholds
    Custom {
        max_accuracy_loss: f32,
        min_speedup: f32,
    },
}

/// Precision selection result for a node
#[derive(Debug, Clone)]
pub struct PrecisionRecommendation {
    /// Recommended quantization scheme
    pub scheme: QuantizationScheme,
    /// Expected accuracy loss (0.0 = no loss, 1.0 = total loss)
    pub accuracy_loss: f32,
    /// Expected speedup ratio (1.0 = no speedup, 2.0 = 2x faster)
    pub speedup_ratio: f32,
    /// Confidence in recommendation (0.0 = low, 1.0 = high)
    pub confidence: f32,
    /// Reasoning for the recommendation
    pub reasoning: String,
}

/// Precision selection strategy
#[derive(Debug, Clone)]
pub struct PrecisionStrategy {
    /// Priority for different data types
    pub int8_priority: f32,
    /// Priority for int16 (usually lower than int8)
    pub int16_priority: f32,
    /// Priority for dynamic quantization
    pub dynamic_priority: f32,
    /// Priority for keeping full precision
    pub fp32_priority: f32,
    /// Performance sensitivity factor
    pub performance_weight: f32,
    /// Accuracy sensitivity factor
    pub accuracy_weight: f32,
}

impl Default for PrecisionStrategy {
    fn default() -> Self {
        Self {
            int8_priority: 0.8,
            int16_priority: 0.6,
            dynamic_priority: 0.4,
            fp32_priority: 0.2,
            performance_weight: 0.5,
            accuracy_weight: 0.5,
        }
    }
}

/// Automatic precision selector
pub struct AutomaticPrecisionSelector {
    /// Selection criteria
    pub criteria: PrecisionCriteria,
    /// Selection strategy
    pub strategy: PrecisionStrategy,
    /// Operation-specific precision profiles
    pub operation_profiles: HashMap<String, PrecisionProfile>,
}

/// Precision profile for specific operations
#[derive(Debug, Clone)]
pub struct PrecisionProfile {
    /// Recommended scheme for this operation
    pub recommended_scheme: QuantizationScheme,
    /// Expected accuracy impact for each scheme
    pub accuracy_impact: HashMap<QuantizationScheme, f32>,
    /// Expected performance gain for each scheme
    pub performance_gain: HashMap<QuantizationScheme, f32>,
    /// Whether this operation is quantization-sensitive
    pub quantization_sensitive: bool,
}

impl AutomaticPrecisionSelector {
    /// Create new precision selector
    pub fn new(criteria: PrecisionCriteria) -> Self {
        Self {
            criteria,
            strategy: PrecisionStrategy::default(),
            operation_profiles: Self::create_default_profiles(),
        }
    }

    /// Create precision selector with custom strategy
    pub fn with_strategy(criteria: PrecisionCriteria, strategy: PrecisionStrategy) -> Self {
        Self {
            criteria,
            strategy,
            operation_profiles: Self::create_default_profiles(),
        }
    }

    /// Analyze graph and recommend precision for each operation
    pub fn analyze_graph(
        &self,
        graph: &FxGraph,
    ) -> TorshResult<HashMap<NodeIndex, PrecisionRecommendation>> {
        let mut recommendations = HashMap::new();

        // Analyze each node in the graph
        for (node_idx, node) in graph.graph.node_references() {
            if let Node::Call(op_name, _args) = node {
                let recommendation = self.analyze_operation(&op_name, node_idx, graph)?;
                recommendations.insert(node_idx, recommendation);
            }
        }

        // Post-process recommendations to ensure graph-level consistency
        self.optimize_graph_precision(&mut recommendations, graph)?;

        Ok(recommendations)
    }

    /// Analyze a specific operation and recommend precision
    fn analyze_operation(
        &self,
        op_name: &str,
        node_idx: NodeIndex,
        graph: &FxGraph,
    ) -> TorshResult<PrecisionRecommendation> {
        let profile = self
            .operation_profiles
            .get(op_name)
            .cloned()
            .unwrap_or_else(|| self.create_generic_profile(op_name));

        // Calculate scores for different precision schemes
        let mut best_score = f32::NEG_INFINITY;
        let mut best_scheme = None;
        let mut best_reasoning = String::new();

        for &scheme in &[
            QuantizationScheme::Int8,
            QuantizationScheme::Int16,
            QuantizationScheme::Dynamic,
        ] {
            let score = self.calculate_precision_score(&profile, scheme, node_idx, graph)?;

            if score > best_score && score != f32::NEG_INFINITY {
                best_score = score;
                best_scheme = Some(scheme);
                best_reasoning = self.generate_reasoning(op_name, scheme, &profile);
            }
        }

        // Use the best scheme or fallback to a conservative default
        let selected_scheme = best_scheme.unwrap_or_else(|| {
            // If no scheme meets the criteria, use the most conservative option
            if matches!(self.criteria, PrecisionCriteria::Custom { .. }) {
                // For custom criteria, try to find a scheme that at least meets accuracy requirements
                for &scheme in &[
                    QuantizationScheme::Int16,
                    QuantizationScheme::Dynamic,
                    QuantizationScheme::Int8,
                ] {
                    let accuracy_loss =
                        profile.accuracy_impact.get(&scheme).copied().unwrap_or(0.1);
                    if let PrecisionCriteria::Custom {
                        max_accuracy_loss, ..
                    } = self.criteria
                    {
                        if accuracy_loss <= max_accuracy_loss {
                            return scheme;
                        }
                    }
                }
            }
            QuantizationScheme::Int16 // Most conservative fallback
        });

        // Get metrics for the selected scheme
        let accuracy_loss = profile
            .accuracy_impact
            .get(&selected_scheme)
            .copied()
            .unwrap_or(0.1);
        let speedup_ratio = profile
            .performance_gain
            .get(&selected_scheme)
            .copied()
            .unwrap_or(1.2);
        let confidence = self.calculate_confidence(&profile, selected_scheme);

        Ok(PrecisionRecommendation {
            scheme: selected_scheme,
            accuracy_loss,
            speedup_ratio,
            confidence,
            reasoning: if best_scheme.is_some() {
                best_reasoning
            } else {
                format!(
                    "Fallback selection for '{}' due to constraint violations",
                    op_name
                )
            },
        })
    }

    /// Calculate precision score for a specific scheme
    fn calculate_precision_score(
        &self,
        profile: &PrecisionProfile,
        scheme: QuantizationScheme,
        _node_idx: NodeIndex,
        _graph: &FxGraph,
    ) -> TorshResult<f32> {
        let accuracy_loss = profile.accuracy_impact.get(&scheme).copied().unwrap_or(0.1);
        let performance_gain = profile
            .performance_gain
            .get(&scheme)
            .copied()
            .unwrap_or(1.1);

        // Calculate base score
        let accuracy_score = (1.0 - accuracy_loss) * self.strategy.accuracy_weight;
        let performance_score = (performance_gain - 1.0) * self.strategy.performance_weight;

        // Apply criteria-specific adjustments
        let adjusted_score = match self.criteria {
            PrecisionCriteria::Performance => performance_score * 2.0 + accuracy_score,
            PrecisionCriteria::Accuracy => {
                // For accuracy-focused selection, heavily penalize accuracy loss
                // and favor schemes that preserve accuracy, especially for sensitive operations
                if profile.quantization_sensitive {
                    // For sensitive operations, strongly prefer Int16 over Int8
                    let sensitivity_bonus = match scheme {
                        QuantizationScheme::Int16 => 2.0,
                        QuantizationScheme::Int8 => -1.0,
                        _ => 0.0,
                    };
                    accuracy_score * 3.0 + performance_score * 0.5 + sensitivity_bonus
                } else {
                    accuracy_score * 2.0 + performance_score
                }
            }
            PrecisionCriteria::Balanced => {
                // For balanced selection, also consider operation sensitivity
                if profile.quantization_sensitive {
                    // For sensitive operations, prefer Int16 over Int8
                    let sensitivity_bonus = match scheme {
                        QuantizationScheme::Int16 => 1.0,
                        QuantizationScheme::Int8 => -0.5,
                        _ => 0.0,
                    };
                    accuracy_score + performance_score + sensitivity_bonus
                } else {
                    accuracy_score + performance_score
                }
            }
            PrecisionCriteria::Custom {
                max_accuracy_loss,
                min_speedup,
            } => {
                if accuracy_loss > max_accuracy_loss || performance_gain < min_speedup {
                    return Ok(f32::NEG_INFINITY);
                }
                accuracy_score + performance_score
            }
        };

        // Apply scheme-specific priority
        let priority = match scheme {
            QuantizationScheme::Int8 => self.strategy.int8_priority,
            QuantizationScheme::Int16 => self.strategy.int16_priority,
            QuantizationScheme::Dynamic => self.strategy.dynamic_priority,
            QuantizationScheme::Fake => self.strategy.fp32_priority,
        };

        Ok(adjusted_score * priority)
    }

    /// Generate reasoning for precision recommendation
    fn generate_reasoning(
        &self,
        op_name: &str,
        scheme: QuantizationScheme,
        profile: &PrecisionProfile,
    ) -> String {
        let scheme_name = match scheme {
            QuantizationScheme::Int8 => "INT8",
            QuantizationScheme::Int16 => "INT16",
            QuantizationScheme::Dynamic => "Dynamic",
            QuantizationScheme::Fake => "FP32",
        };

        if profile.quantization_sensitive {
            format!("Operation '{op_name}' is quantization-sensitive. {scheme_name} provides good balance of performance and accuracy.")
        } else {
            format!("Operation '{op_name}' is quantization-friendly. {scheme_name} offers optimal performance with minimal accuracy loss.")
        }
    }

    /// Calculate confidence in recommendation
    fn calculate_confidence(&self, profile: &PrecisionProfile, scheme: QuantizationScheme) -> f32 {
        let base_confidence = if profile.quantization_sensitive {
            0.75
        } else {
            0.9
        };

        // Higher confidence for well-supported schemes
        let scheme_confidence = match scheme {
            QuantizationScheme::Int8 => 0.9,
            QuantizationScheme::Int16 => 0.85,
            QuantizationScheme::Dynamic => 0.7,
            QuantizationScheme::Fake => 0.6,
        };

        // Additional bonus for schemes that match the operation's recommended scheme
        let recommendation_bonus = if scheme == profile.recommended_scheme {
            1.1
        } else {
            1.0
        };

        let confidence: f32 = base_confidence * scheme_confidence * recommendation_bonus;
        confidence.min(1.0)
    }

    /// Optimize precision recommendations at graph level
    fn optimize_graph_precision(
        &self,
        recommendations: &mut HashMap<NodeIndex, PrecisionRecommendation>,
        _graph: &FxGraph,
    ) -> TorshResult<()> {
        // For now, just ensure consistency - in practice, this would:
        // 1. Minimize precision mismatches between connected operations
        // 2. Avoid unnecessary conversions
        // 3. Consider memory bandwidth and cache efficiency

        // Simple optimization: prefer consistent precision in chains
        for recommendation in recommendations.values_mut() {
            if recommendation.confidence < 0.5 {
                // Low confidence - use conservative INT16
                recommendation.scheme = QuantizationScheme::Int16;
                recommendation.reasoning = format!(
                    "Conservative choice due to low confidence: {}",
                    recommendation.reasoning
                );
            }
        }

        Ok(())
    }

    /// Create default precision profiles for common operations
    fn create_default_profiles() -> HashMap<String, PrecisionProfile> {
        let mut profiles = HashMap::new();

        // Matrix operations - generally quantization-friendly
        profiles.insert(
            "matmul".to_string(),
            PrecisionProfile {
                recommended_scheme: QuantizationScheme::Int8,
                accuracy_impact: [
                    (QuantizationScheme::Int8, 0.015),
                    (QuantizationScheme::Int16, 0.005),
                    (QuantizationScheme::Dynamic, 0.008),
                    (QuantizationScheme::Fake, 0.0),
                ]
                .iter()
                .cloned()
                .collect(),
                performance_gain: [
                    (QuantizationScheme::Int8, 2.5),
                    (QuantizationScheme::Int16, 2.2),
                    (QuantizationScheme::Dynamic, 2.1),
                    (QuantizationScheme::Fake, 1.0),
                ]
                .iter()
                .cloned()
                .collect(),
                quantization_sensitive: false,
            },
        );

        // Convolution operations - quantization-friendly
        profiles.insert(
            "conv2d".to_string(),
            PrecisionProfile {
                recommended_scheme: QuantizationScheme::Int8,
                accuracy_impact: [
                    (QuantizationScheme::Int8, 0.03),
                    (QuantizationScheme::Int16, 0.008),
                    (QuantizationScheme::Dynamic, 0.015),
                    (QuantizationScheme::Fake, 0.0),
                ]
                .iter()
                .cloned()
                .collect(),
                performance_gain: [
                    (QuantizationScheme::Int8, 3.0),
                    (QuantizationScheme::Int16, 2.0),
                    (QuantizationScheme::Dynamic, 1.5),
                    (QuantizationScheme::Fake, 1.0),
                ]
                .iter()
                .cloned()
                .collect(),
                quantization_sensitive: false,
            },
        );

        // Attention operations - more quantization-sensitive
        profiles.insert(
            "attention".to_string(),
            PrecisionProfile {
                recommended_scheme: QuantizationScheme::Int16,
                accuracy_impact: [
                    (QuantizationScheme::Int8, 0.08),
                    (QuantizationScheme::Int16, 0.02),
                    (QuantizationScheme::Dynamic, 0.04),
                    (QuantizationScheme::Fake, 0.0),
                ]
                .iter()
                .cloned()
                .collect(),
                performance_gain: [
                    (QuantizationScheme::Int8, 2.0),
                    (QuantizationScheme::Int16, 1.6),
                    (QuantizationScheme::Dynamic, 1.3),
                    (QuantizationScheme::Fake, 1.0),
                ]
                .iter()
                .cloned()
                .collect(),
                quantization_sensitive: true,
            },
        );

        // Activation functions - generally quantization-friendly
        profiles.insert(
            "relu".to_string(),
            PrecisionProfile {
                recommended_scheme: QuantizationScheme::Int8,
                accuracy_impact: [
                    (QuantizationScheme::Int8, 0.001),
                    (QuantizationScheme::Int16, 0.0005),
                    (QuantizationScheme::Dynamic, 0.0008),
                    (QuantizationScheme::Fake, 0.0),
                ]
                .iter()
                .cloned()
                .collect(),
                performance_gain: [
                    (QuantizationScheme::Int8, 1.8),
                    (QuantizationScheme::Int16, 1.4),
                    (QuantizationScheme::Dynamic, 1.2),
                    (QuantizationScheme::Fake, 1.0),
                ]
                .iter()
                .cloned()
                .collect(),
                quantization_sensitive: false,
            },
        );

        profiles
    }

    /// Create generic profile for unknown operations
    fn create_generic_profile(&self, _op_name: &str) -> PrecisionProfile {
        PrecisionProfile {
            recommended_scheme: QuantizationScheme::Int16, // Conservative default
            accuracy_impact: [
                (QuantizationScheme::Int8, 0.015),
                (QuantizationScheme::Int16, 0.005),
                (QuantizationScheme::Dynamic, 0.01),
                (QuantizationScheme::Fake, 0.0),
            ]
            .iter()
            .cloned()
            .collect(),
            performance_gain: [
                (QuantizationScheme::Int8, 2.0),
                (QuantizationScheme::Int16, 1.5),
                (QuantizationScheme::Dynamic, 1.3),
                (QuantizationScheme::Fake, 1.0),
            ]
            .iter()
            .cloned()
            .collect(),
            quantization_sensitive: true, // Conservative default
        }
    }
}

/// Convenience function to perform automatic precision selection
pub fn select_automatic_precision(
    graph: &FxGraph,
    criteria: PrecisionCriteria,
) -> TorshResult<HashMap<NodeIndex, PrecisionRecommendation>> {
    let selector = AutomaticPrecisionSelector::new(criteria);
    selector.analyze_graph(graph)
}

/// Apply automatic precision selection to a graph
pub fn apply_automatic_precision(
    graph: &mut FxGraph,
    criteria: PrecisionCriteria,
) -> TorshResult<QuantizationContext> {
    let recommendations = select_automatic_precision(graph, criteria)?;

    let mut context = QuantizationContext::new(QuantizationScheme::Int8);

    // Apply recommendations to the graph
    for (node_idx, recommendation) in recommendations {
        let params = QuantizationParams::symmetric(recommendation.scheme, 0.1);
        let annotation = QuantizationAnnotation {
            input_params: vec![Some(params.clone())],
            output_params: Some(params),
            calibration_data: None,
        };

        context.annotate_node(node_idx, annotation);
    }

    Ok(context)
}
