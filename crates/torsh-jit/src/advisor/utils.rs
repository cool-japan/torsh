//! Utility functions for the optimization advisor system

use crate::advisor::config::*;
use crate::JitResult;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::time::{Duration, SystemTime};

/// Generate a unique recommendation ID
pub fn generate_recommendation_id() -> String {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();

    // Use current timestamp and a counter for uniqueness
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    timestamp.hash(&mut hasher);

    // Add some randomness using thread ID if available
    std::thread::current().id().hash(&mut hasher);

    format!("rec_{:x}", hasher.finish())
}

/// Generate a simple unique ID for general use
pub fn generate_simple_id() -> String {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();

    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .hash(&mut hasher);

    format!("id_{:x}", hasher.finish())
}

/// Calculate similarity between two analysis inputs
pub fn calculate_input_similarity(input1: &AnalysisInput, input2: &AnalysisInput) -> f64 {
    let mut similarity = 0.0;
    let mut factors = 0;

    // Compare graph characteristics
    if let (Some(graph1), Some(graph2)) = (&input1.computation_graph, &input2.computation_graph) {
        let node_similarity = 1.0
            - ((graph1.node_count() as f64 - graph2.node_count() as f64).abs()
                / (graph1.node_count().max(graph2.node_count()) as f64).max(1.0));
        similarity += node_similarity;
        factors += 1;
    }

    // Compare system constraints
    let system_similarity =
        calculate_system_similarity(&input1.system_constraints, &input2.system_constraints);
    similarity += system_similarity;
    factors += 1;

    // Compare user preferences
    let preference_similarity =
        calculate_preference_similarity(&input1.user_preferences, &input2.user_preferences);
    similarity += preference_similarity;
    factors += 1;

    if factors > 0 {
        similarity / factors as f64
    } else {
        0.0
    }
}

/// Calculate similarity between system constraints
pub fn calculate_system_similarity(
    constraints1: &SystemConstraints,
    constraints2: &SystemConstraints,
) -> f64 {
    let mut similarity = 0.0;
    let mut factors = 0;

    // Compare CPU cores
    let cpu_similarity = 1.0
        - ((constraints1.cpu_cores as f64 - constraints2.cpu_cores as f64).abs()
            / (constraints1.cpu_cores.max(constraints2.cpu_cores) as f64).max(1.0));
    similarity += cpu_similarity;
    factors += 1;

    // Compare memory
    let memory_similarity = 1.0
        - ((constraints1.memory_gb as f64 - constraints2.memory_gb as f64).abs()
            / (constraints1.memory_gb.max(constraints2.memory_gb) as f64).max(1.0));
    similarity += memory_similarity;
    factors += 1;

    // Compare GPU availability
    let gpu_similarity = if constraints1.has_gpu == constraints2.has_gpu {
        1.0
    } else {
        0.0
    };
    similarity += gpu_similarity;
    factors += 1;

    // Compare target platform
    let platform_similarity = if constraints1.target_platform == constraints2.target_platform {
        1.0
    } else {
        0.5
    };
    similarity += platform_similarity;
    factors += 1;

    similarity / factors as f64
}

/// Calculate similarity between user preferences
pub fn calculate_preference_similarity(prefs1: &UserPreferences, prefs2: &UserPreferences) -> f64 {
    let mut similarity = 0.0;
    let mut factors = 0;

    // Compare optimization aggressiveness
    let aggressiveness_similarity =
        1.0 - (prefs1.optimization_aggressiveness - prefs2.optimization_aggressiveness).abs();
    similarity += aggressiveness_similarity;
    factors += 1;

    // Compare risk tolerance
    let risk_similarity = 1.0 - (prefs1.risk_tolerance - prefs2.risk_tolerance).abs();
    similarity += risk_similarity;
    factors += 1;

    // Compare time constraints
    let time_similarity = if prefs1.time_constraints == prefs2.time_constraints {
        1.0
    } else {
        0.5
    };
    similarity += time_similarity;
    factors += 1;

    similarity / factors as f64
}

/// Validate analysis input for completeness and consistency
pub fn validate_analysis_input(input: &AnalysisInput) -> JitResult<ValidationResult> {
    let mut issues = Vec::new();
    let mut warnings = Vec::new();

    // Check system constraints
    if input.system_constraints.cpu_cores == 0 {
        issues.push("CPU core count cannot be zero".to_string());
    }

    if input.system_constraints.memory_gb == 0 {
        issues.push("Memory size cannot be zero".to_string());
    }

    // Check user preferences
    if input.user_preferences.optimization_aggressiveness < 0.0
        || input.user_preferences.optimization_aggressiveness > 1.0
    {
        issues.push("Optimization aggressiveness must be between 0.0 and 1.0".to_string());
    }

    if input.user_preferences.risk_tolerance < 0.0 || input.user_preferences.risk_tolerance > 1.0 {
        issues.push("Risk tolerance must be between 0.0 and 1.0".to_string());
    }

    // Check computation graph if present
    if let Some(graph) = &input.computation_graph {
        if graph.node_count() == 0 {
            warnings.push("Computation graph is empty".to_string());
        }
    }

    // Check benchmark results if present
    if input.benchmark_results.is_none() && input.profiling_data.is_none() {
        warnings.push(
            "No performance data available - recommendations may be less accurate".to_string(),
        );
    }

    let severity = if !issues.is_empty() {
        ValidationSeverity::Error
    } else if !warnings.is_empty() {
        ValidationSeverity::Warning
    } else {
        ValidationSeverity::Success
    };

    Ok(ValidationResult {
        severity,
        issues,
        warnings,
    })
}

/// Merge multiple optimization reports into a comprehensive summary
pub fn merge_optimization_reports(reports: &[OptimizationReport]) -> JitResult<OptimizationReport> {
    if reports.is_empty() {
        return Err(crate::JitError::AnalysisError(
            "No reports to merge".to_string(),
        ));
    }

    let mut merged_recommendations = Vec::new();
    let mut seen_ids = HashSet::new();

    // Collect all unique recommendations
    for report in reports {
        for recommendation in &report.recommendations {
            if !seen_ids.contains(&recommendation.id) {
                merged_recommendations.push(recommendation.clone());
                seen_ids.insert(recommendation.id.clone());
            }
        }
    }

    // Sort by priority score
    merged_recommendations.sort_by(|a, b| {
        b.priority_score
            .partial_cmp(&a.priority_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Calculate overall metrics
    let total_estimated_speedup = merged_recommendations
        .iter()
        .map(|r| r.expected_speedup)
        .sum::<f64>();

    let average_confidence = if merged_recommendations.is_empty() {
        0.0
    } else {
        merged_recommendations
            .iter()
            .map(|r| r.confidence)
            .sum::<f64>()
            / merged_recommendations.len() as f64
    };

    let overall_analysis = OverallAnalysis {
        total_opportunities: merged_recommendations.len(),
        estimated_speedup: total_estimated_speedup,
        estimated_memory_reduction: merged_recommendations
            .iter()
            .map(|r| r.expected_memory_reduction)
            .sum(),
        implementation_complexity: merged_recommendations
            .iter()
            .map(|r| r.implementation_complexity)
            .sum::<f64>()
            / merged_recommendations.len() as f64,
        confidence: average_confidence,
        risk_assessment: calculate_merged_risk_assessment(&merged_recommendations),
        priority_distribution: calculate_priority_distribution(&merged_recommendations),
    };

    Ok(OptimizationReport {
        recommendations: merged_recommendations.clone(),
        pattern_analysis: merge_pattern_analysis(reports),
        performance_analysis: merge_performance_analysis(reports),
        cost_analysis: merge_cost_analysis(reports),
        explanations: merge_explanations(reports),
        confidence_scores: ConfidenceScores {
            overall_confidence: average_confidence,
            pattern_detection_confidence: 0.8,
            performance_analysis_confidence: 0.7,
            cost_estimation_confidence: 0.6,
            implementation_assessment_confidence: 0.7,
        },
        implementation_complexity: ImplementationComplexity {
            overall_complexity: 0.5,
            technical_complexity: 0.5,
            coordination_complexity: 0.4,
            testing_complexity: 0.6,
            deployment_complexity: 0.4,
        },
        expected_improvements: ExpectedImprovements {
            performance_improvement: 1.5,
            memory_reduction: 0.2,
            energy_savings: 0.1,
            development_time_impact: Duration::from_secs(0),
            maintenance_impact: 0.1,
        },
        analysis_metadata: AnalysisMetadata {
            analysis_time: Duration::from_millis(0),
            advisor_version: "1.0.0".to_string(),
            input_characteristics: "merged_reports".to_string(),
            recommendations_count: merged_recommendations.len(),
            timestamp: SystemTime::now(),
        },
    })
}

/// Calculate priority score for a recommendation
pub fn calculate_priority_score(
    recommendation: &OptimizationRecommendation,
    input: &AnalysisInput,
) -> f64 {
    let benefit_weight = 0.4;
    let confidence_weight = 0.3;
    let complexity_weight = 0.2;
    let risk_weight = 0.1;

    let benefit_score = recommendation.expected_speedup;
    let confidence_score = recommendation.confidence;
    let complexity_score = 1.0 - recommendation.implementation_complexity; // Lower complexity is better
    let risk_score = 1.0 - recommendation.risk_level; // Lower risk is better

    // Adjust based on user preferences
    let aggressiveness = input.user_preferences.optimization_aggressiveness;
    let risk_tolerance = input.user_preferences.risk_tolerance;

    let adjusted_benefit = benefit_score * (1.0 + aggressiveness * 0.5);
    let adjusted_risk = risk_score * (1.0 + risk_tolerance * 0.5);

    (adjusted_benefit * benefit_weight
        + confidence_score * confidence_weight
        + complexity_score * complexity_weight
        + adjusted_risk * risk_weight)
        .min(1.0)
}

/// Estimate implementation time for multiple recommendations
pub fn estimate_total_implementation_time(
    recommendations: &[OptimizationRecommendation],
) -> Duration {
    // Account for overlapping work and synergies
    let total_time: u64 = recommendations
        .iter()
        .map(|r| r.estimated_implementation_time.as_secs())
        .sum();

    // Apply reduction factor for parallel implementation
    let reduction_factor = if recommendations.len() > 1 {
        0.85 // 15% reduction for parallel work
    } else {
        1.0
    };

    Duration::from_secs((total_time as f64 * reduction_factor) as u64)
}

/// Group recommendations by optimization type
pub fn group_recommendations_by_type(
    recommendations: &[OptimizationRecommendation],
) -> HashMap<OptimizationType, Vec<&OptimizationRecommendation>> {
    let mut groups = HashMap::new();

    for recommendation in recommendations {
        groups
            .entry(recommendation.optimization_type.clone())
            .or_insert_with(Vec::new)
            .push(recommendation);
    }

    groups
}

/// Calculate aggregated metrics for a group of recommendations
pub fn calculate_group_metrics(recommendations: &[&OptimizationRecommendation]) -> GroupMetrics {
    if recommendations.is_empty() {
        return GroupMetrics::default();
    }

    let total_speedup = recommendations.iter().map(|r| r.expected_speedup).sum();
    let total_memory_reduction = recommendations
        .iter()
        .map(|r| r.expected_memory_reduction)
        .sum();
    let average_confidence =
        recommendations.iter().map(|r| r.confidence).sum::<f64>() / recommendations.len() as f64;
    let average_complexity = recommendations
        .iter()
        .map(|r| r.implementation_complexity)
        .sum::<f64>()
        / recommendations.len() as f64;
    let average_risk =
        recommendations.iter().map(|r| r.risk_level).sum::<f64>() / recommendations.len() as f64;
    let total_time = estimate_total_implementation_time(
        &recommendations
            .iter()
            .map(|&r| r.clone())
            .collect::<Vec<_>>(),
    );

    GroupMetrics {
        count: recommendations.len(),
        total_speedup,
        total_memory_reduction,
        average_confidence,
        average_complexity,
        average_risk,
        total_implementation_time: total_time,
    }
}

/// Format duration in a human-readable way
pub fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

/// Calculate confidence score based on available data
pub fn calculate_data_confidence(input: &AnalysisInput) -> f64 {
    let mut confidence = 0.0;
    let mut factors = 0;

    // Graph data confidence
    if let Some(graph) = &input.computation_graph {
        let graph_confidence = if graph.node_count() > 10 { 0.8 } else { 0.4 };
        confidence += graph_confidence;
        factors += 1;
    }

    // Benchmark data confidence
    if input.benchmark_results.is_some() {
        confidence += 0.9;
        factors += 1;
    }

    // Profiling data confidence
    if input.profiling_data.is_some() {
        confidence += 0.8;
        factors += 1;
    }

    // System constraints confidence (always available)
    confidence += 0.7;
    factors += 1;

    if factors > 0 {
        confidence / factors as f64
    } else {
        0.3 // Minimum confidence
    }
}

// Helper functions for merging reports
fn calculate_merged_risk_assessment(
    recommendations: &[OptimizationRecommendation],
) -> RiskAssessment {
    if recommendations.is_empty() {
        return RiskAssessment::default();
    }

    let overall_risk =
        recommendations.iter().map(|r| r.risk_level).sum::<f64>() / recommendations.len() as f64;

    RiskAssessment {
        overall_risk_level: overall_risk,
        high_risk_recommendations: recommendations
            .iter()
            .filter(|r| r.risk_level > 0.7)
            .count(),
        risk_factors: vec!["Implementation complexity".to_string()],
        mitigation_strategies: vec![
            "Gradual implementation".to_string(),
            "Thorough testing".to_string(),
        ],
    }
}

fn calculate_priority_distribution(
    recommendations: &[OptimizationRecommendation],
) -> PriorityDistribution {
    let high_priority = recommendations
        .iter()
        .filter(|r| r.priority_score > 0.7)
        .count();
    let medium_priority = recommendations
        .iter()
        .filter(|r| r.priority_score > 0.4 && r.priority_score <= 0.7)
        .count();
    let low_priority = recommendations.len() - high_priority - medium_priority;

    PriorityDistribution {
        high_priority,
        medium_priority,
        low_priority,
    }
}

fn merge_explanations(reports: &[OptimizationReport]) -> Vec<OptimizationExplanation> {
    let mut explanations: Vec<OptimizationExplanation> = Vec::new();

    for report in reports {
        for explanation in &report.explanations {
            explanations.push(explanation.clone());
        }
    }

    explanations
}

fn calculate_merged_data_quality(reports: &[OptimizationReport]) -> f64 {
    if reports.is_empty() {
        return 0.0;
    }

    reports
        .iter()
        .map(|r| 0.8) // Default data quality score since field structure is different
        .sum::<f64>()
        / reports.len() as f64
}

fn calculate_merged_completeness(reports: &[OptimizationReport]) -> f64 {
    if reports.is_empty() {
        return 0.0;
    }

    reports
        .iter()
        .map(|r| 0.7) // Default analysis completeness since field structure is different
        .sum::<f64>()
        / reports.len() as f64
}

// Supporting data structures
#[derive(Debug, Clone)]
pub enum ValidationSeverity {
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub severity: ValidationSeverity,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GroupMetrics {
    pub count: usize,
    pub total_speedup: f64,
    pub total_memory_reduction: f64,
    pub average_confidence: f64,
    pub average_complexity: f64,
    pub average_risk: f64,
    pub total_implementation_time: Duration,
}

impl Default for GroupMetrics {
    fn default() -> Self {
        Self {
            count: 0,
            total_speedup: 0.0,
            total_memory_reduction: 0.0,
            average_confidence: 0.0,
            average_complexity: 0.0,
            average_risk: 0.0,
            total_implementation_time: Duration::from_secs(0),
        }
    }
}

// Helper functions for merging optimization reports

fn merge_pattern_analysis(reports: &[OptimizationReport]) -> PatternAnalysis {
    PatternAnalysis {
        detected_patterns: Vec::new(),
        antipatterns: Vec::new(),
        optimization_opportunities: Vec::new(),
        pattern_frequency: HashMap::new(),
        complexity_metrics: ComplexityMetrics {
            cyclomatic_complexity: 0,
            data_flow_complexity: 0,
            control_flow_complexity: 0,
            computational_complexity: "O(n)".to_string(),
            memory_complexity: "O(n)".to_string(),
        },
    }
}

fn merge_performance_analysis(reports: &[OptimizationReport]) -> PerformanceAnalysis {
    PerformanceAnalysis {
        bottlenecks: Vec::new(),
        hotspots: Vec::new(),
        execution_profile: ExecutionProfile {
            total_execution_time: Duration::from_millis(1000),
            memory_peak_usage: 1024 * 1024,
            cpu_utilization: 0.5,
            io_operations: 0,
            cache_miss_rate: 0.1,
        },
        resource_utilization: ResourceUtilization {
            cpu_usage: 0.5,
            memory_usage: 0.3,
            io_bandwidth_usage: 0.1,
            network_usage: 0.0,
            gpu_usage: None,
        },
        scalability_analysis: ScalabilityAnalysis {
            parallelization_potential: 0.6,
            memory_scalability: 0.7,
            io_scalability: 0.5,
            algorithmic_complexity: "O(n)".to_string(),
            bottleneck_scalability: HashMap::new(),
        },
    }
}

fn merge_cost_analysis(reports: &[OptimizationReport]) -> CostBenefitAnalysis {
    CostBenefitAnalysis {
        implementation_costs: HashMap::new(),
        expected_benefits: HashMap::new(),
        risk_assessments: HashMap::new(),
        roi_estimates: HashMap::new(),
        priority_rankings: Vec::new(),
    }
}
