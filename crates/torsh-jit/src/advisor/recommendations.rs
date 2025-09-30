//! Recommendation engine for generating optimization suggestions

use crate::advisor::config::*;
use crate::advisor::utils::*;
use crate::JitResult;
use std::time::Duration;

/// Recommendation engine for generating optimization suggestions
pub struct RecommendationEngine {
    config: AdvisorConfig,
}

impl RecommendationEngine {
    pub fn new(config: AdvisorConfig) -> Self {
        Self { config }
    }

    pub fn generate_from_opportunity(
        &self,
        opportunity: &OptimizationOpportunity,
        input: &AnalysisInput,
        performance_analysis: &PerformanceAnalysis,
        cost_analysis: &CostBenefitAnalysis,
    ) -> JitResult<Option<OptimizationRecommendation>> {
        if opportunity.estimated_benefit < self.config.min_benefit_threshold {
            return Ok(None);
        }

        let optimization_type = opportunity.opportunity_type.to_optimization_type();

        let recommendation = OptimizationRecommendation {
            id: generate_recommendation_id(),
            optimization_type: optimization_type.clone(),
            title: self.generate_title(&optimization_type),
            description: opportunity.description.clone(),
            expected_speedup: opportunity.estimated_benefit,
            expected_memory_reduction: opportunity.estimated_benefit * 0.5,
            implementation_complexity: opportunity.implementation_complexity,
            confidence: self.calculate_recommendation_confidence(opportunity, performance_analysis),
            risk_level: opportunity.implementation_complexity * 0.8,
            priority_score: 0.0, // Will be calculated later
            affected_components: self.identify_affected_components(opportunity, input),
            prerequisites: opportunity.prerequisites.clone(),
            estimated_implementation_time: self.estimate_implementation_time(opportunity),
        };

        Ok(Some(recommendation))
    }

    pub fn generate_from_bottleneck(
        &self,
        bottleneck: &PerformanceBottleneck,
        input: &AnalysisInput,
        _pattern_analysis: &PatternAnalysis,
        _cost_analysis: &CostBenefitAnalysis,
    ) -> JitResult<Option<OptimizationRecommendation>> {
        let optimization_type = match bottleneck.bottleneck_type {
            BottleneckType::Memory => OptimizationType::MemoryOptimization,
            BottleneckType::Computation => OptimizationType::ComputationOptimization,
            BottleneckType::IO => OptimizationType::IOOptimization,
            BottleneckType::Synchronization => OptimizationType::ParallelizationOptimization,
        };

        let recommendation = OptimizationRecommendation {
            id: generate_recommendation_id(),
            optimization_type: optimization_type.clone(),
            title: format!(
                "Address {} Bottleneck",
                bottleneck.bottleneck_type.description()
            ),
            description: format!(
                "Optimize {} bottleneck: {}",
                bottleneck.bottleneck_type.description(),
                bottleneck.description
            ),
            expected_speedup: bottleneck.severity * 0.4,
            expected_memory_reduction: if matches!(
                bottleneck.bottleneck_type,
                BottleneckType::Memory
            ) {
                bottleneck.severity * 0.3
            } else {
                0.0
            },
            implementation_complexity: 0.5,
            confidence: 0.8,
            risk_level: 0.3,
            priority_score: 0.0,
            affected_components: vec![bottleneck.location.clone()],
            prerequisites: self.identify_prerequisites(&optimization_type),
            estimated_implementation_time: Duration::from_secs(20 * 3600),
        };

        Ok(Some(recommendation))
    }

    pub fn generate_from_antipattern(
        &self,
        antipattern: &DetectedAntipattern,
        _input: &AnalysisInput,
        _cost_analysis: &CostBenefitAnalysis,
    ) -> JitResult<Option<OptimizationRecommendation>> {
        let optimization_type = match antipattern.antipattern_type {
            AntipatternType::RedundantComputation => {
                OptimizationType::DeadCodeEliminationOptimization
            }
            AntipatternType::PoorMemoryLocality => OptimizationType::MemoryOptimization,
            AntipatternType::InefficientAlgorithm => OptimizationType::ComputationOptimization,
            AntipatternType::ExcessiveAllocation => OptimizationType::MemoryOptimization,
        };

        let recommendation = OptimizationRecommendation {
            id: generate_recommendation_id(),
            optimization_type: optimization_type.clone(),
            title: format!("Fix {}", antipattern.antipattern_type.description()),
            description: format!("Address antipattern: {}", antipattern.description),
            expected_speedup: antipattern.severity * 0.3,
            expected_memory_reduction: if matches!(
                antipattern.antipattern_type,
                AntipatternType::ExcessiveAllocation
            ) {
                antipattern.severity * 0.4
            } else {
                antipattern.severity * 0.1
            },
            implementation_complexity: 0.4,
            confidence: 0.7,
            risk_level: 0.2,
            priority_score: 0.0,
            affected_components: self.extract_components_from_location(&antipattern.location),
            prerequisites: self.identify_prerequisites(&optimization_type),
            estimated_implementation_time: Duration::from_secs(12 * 3600),
        };

        Ok(Some(recommendation))
    }

    pub fn generate_holistic_recommendations(
        &self,
        input: &AnalysisInput,
        pattern_analysis: &PatternAnalysis,
        performance_analysis: &PerformanceAnalysis,
        _cost_analysis: &CostBenefitAnalysis,
    ) -> JitResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        // Generate system-wide recommendations
        if self.should_recommend_compilation_optimization(input, performance_analysis) {
            recommendations.push(self.create_compilation_optimization_recommendation()?);
        }

        if self.should_recommend_architecture_optimization(input, pattern_analysis) {
            recommendations.push(self.create_architecture_optimization_recommendation()?);
        }

        // Generate composite optimizations
        if pattern_analysis.detected_patterns.len() > 5 {
            recommendations.push(self.create_comprehensive_optimization_recommendation()?);
        }

        Ok(recommendations)
    }

    pub fn calculate_confidence(&self) -> f64 {
        // Recommendation engine confidence
        0.8
    }

    // Helper methods
    fn generate_title(&self, optimization_type: &OptimizationType) -> String {
        match optimization_type {
            OptimizationType::FusionOptimization => "Implement Operation Fusion".to_string(),
            OptimizationType::MemoryOptimization => "Optimize Memory Access Patterns".to_string(),
            OptimizationType::ParallelizationOptimization => "Enable Parallelization".to_string(),
            OptimizationType::VectorizationOptimization => "Apply Vectorization".to_string(),
            OptimizationType::ConstantFoldingOptimization => "Perform Constant Folding".to_string(),
            OptimizationType::DeadCodeEliminationOptimization => "Eliminate Dead Code".to_string(),
            OptimizationType::ComputationOptimization => "Optimize Computation".to_string(),
            OptimizationType::IOOptimization => "Optimize I/O Operations".to_string(),
            OptimizationType::CompilationOptimization => "Improve Compilation".to_string(),
            OptimizationType::ArchitectureOptimization => "Optimize Architecture".to_string(),
        }
    }

    fn calculate_recommendation_confidence(
        &self,
        opportunity: &OptimizationOpportunity,
        _performance_analysis: &PerformanceAnalysis,
    ) -> f64 {
        let base_confidence = 0.7;
        let complexity_penalty = opportunity.implementation_complexity * 0.2;
        let benefit_boost = opportunity.estimated_benefit * 0.2;

        (base_confidence - complexity_penalty + benefit_boost)
            .max(0.1)
            .min(1.0)
    }

    fn identify_affected_components(
        &self,
        opportunity: &OptimizationOpportunity,
        _input: &AnalysisInput,
    ) -> Vec<String> {
        match &opportunity.location {
            PatternLocation::Node(node_id) => vec![format!("Node_{:?}", node_id)],
            PatternLocation::Nodes(node_ids) => {
                node_ids.iter().map(|id| format!("Node_{:?}", id)).collect()
            }
            PatternLocation::Subgraph(subgraphs) => subgraphs
                .iter()
                .enumerate()
                .map(|(i, _)| format!("Subgraph_{}", i))
                .collect(),
            PatternLocation::Global => vec!["Global".to_string()],
        }
    }

    fn identify_prerequisites(&self, optimization_type: &OptimizationType) -> Vec<String> {
        match optimization_type {
            OptimizationType::FusionOptimization => vec![
                "Data dependency analysis".to_string(),
                "Memory access pattern analysis".to_string(),
            ],
            OptimizationType::MemoryOptimization => {
                vec!["Memory profiling".to_string(), "Cache analysis".to_string()]
            }
            OptimizationType::ParallelizationOptimization => vec![
                "Thread safety analysis".to_string(),
                "Load balancing analysis".to_string(),
            ],
            OptimizationType::VectorizationOptimization => vec![
                "SIMD capability detection".to_string(),
                "Data alignment verification".to_string(),
            ],
            OptimizationType::ConstantFoldingOptimization => {
                vec!["Constant propagation analysis".to_string()]
            }
            OptimizationType::DeadCodeEliminationOptimization => vec![
                "Live variable analysis".to_string(),
                "Reachability analysis".to_string(),
            ],
            _ => vec!["Performance profiling".to_string()],
        }
    }

    fn estimate_implementation_time(&self, opportunity: &OptimizationOpportunity) -> Duration {
        let base_hours = match opportunity.opportunity_type {
            OpportunityType::ConstantFolding => 4,
            OpportunityType::DeadCodeElimination => 6,
            OpportunityType::FusionOptimization => 16,
            OpportunityType::VectorizationOptimization => 20,
            OpportunityType::MemoryOptimization => 24,
            OpportunityType::ComputationOptimization => 32,
            OpportunityType::ParallelizationOptimization => 40,
        };

        let complexity_multiplier = 1.0 + opportunity.implementation_complexity;
        let total_hours = (base_hours as f64 * complexity_multiplier) as u64;

        Duration::from_secs(total_hours * 3600)
    }

    fn extract_components_from_location(&self, location: &PatternLocation) -> Vec<String> {
        match location {
            PatternLocation::Node(node_id) => vec![format!("Node_{:?}", node_id)],
            PatternLocation::Nodes(node_ids) => {
                node_ids.iter().map(|id| format!("Node_{:?}", id)).collect()
            }
            PatternLocation::Subgraph(subgraphs) => subgraphs
                .iter()
                .enumerate()
                .map(|(i, _)| format!("Subgraph_{}", i))
                .collect(),
            PatternLocation::Global => vec!["Global".to_string()],
        }
    }

    fn should_recommend_compilation_optimization(
        &self,
        input: &AnalysisInput,
        performance_analysis: &PerformanceAnalysis,
    ) -> bool {
        // Recommend compilation optimization if compilation takes a significant time
        performance_analysis.execution_profile.total_execution_time > Duration::from_secs(10)
            && input.system_constraints.target_platform == TargetPlatform::Desktop
    }

    fn should_recommend_architecture_optimization(
        &self,
        input: &AnalysisInput,
        pattern_analysis: &PatternAnalysis,
    ) -> bool {
        // Recommend architecture optimization for complex graphs
        input
            .computation_graph
            .as_ref()
            .map_or(false, |g| g.node_count() > 100)
            && pattern_analysis.detected_patterns.len() > 3
    }

    fn create_compilation_optimization_recommendation(
        &self,
    ) -> JitResult<OptimizationRecommendation> {
        Ok(OptimizationRecommendation {
            id: generate_recommendation_id(),
            optimization_type: OptimizationType::CompilationOptimization,
            title: "Optimize Compilation Process".to_string(),
            description: "Improve compilation speed and output quality".to_string(),
            expected_speedup: 0.2,
            expected_memory_reduction: 0.1,
            implementation_complexity: 0.6,
            confidence: 0.7,
            risk_level: 0.3,
            priority_score: 0.0,
            affected_components: vec!["Compiler".to_string()],
            prerequisites: vec!["Compilation profiling".to_string()],
            estimated_implementation_time: Duration::from_secs(30 * 3600),
        })
    }

    fn create_architecture_optimization_recommendation(
        &self,
    ) -> JitResult<OptimizationRecommendation> {
        Ok(OptimizationRecommendation {
            id: generate_recommendation_id(),
            optimization_type: OptimizationType::ArchitectureOptimization,
            title: "Optimize System Architecture".to_string(),
            description: "Improve overall system design for better performance".to_string(),
            expected_speedup: 0.3,
            expected_memory_reduction: 0.2,
            implementation_complexity: 0.8,
            confidence: 0.6,
            risk_level: 0.5,
            priority_score: 0.0,
            affected_components: vec!["Architecture".to_string()],
            prerequisites: vec!["Architecture analysis".to_string()],
            estimated_implementation_time: Duration::from_secs(80 * 3600),
        })
    }

    fn create_comprehensive_optimization_recommendation(
        &self,
    ) -> JitResult<OptimizationRecommendation> {
        Ok(OptimizationRecommendation {
            id: generate_recommendation_id(),
            optimization_type: OptimizationType::ComputationOptimization,
            title: "Comprehensive Performance Optimization".to_string(),
            description: "Apply multiple optimization techniques systematically".to_string(),
            expected_speedup: 0.4,
            expected_memory_reduction: 0.3,
            implementation_complexity: 0.9,
            confidence: 0.8,
            risk_level: 0.4,
            priority_score: 0.0,
            affected_components: vec!["Entire System".to_string()],
            prerequisites: vec!["Comprehensive analysis".to_string()],
            estimated_implementation_time: Duration::from_secs(120 * 3600),
        })
    }
}
