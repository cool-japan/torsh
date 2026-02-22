//! Cost model for optimization analysis

use crate::advisor::config::*;
use crate::JitResult;
use std::collections::HashMap;

/// Cost model for optimization analysis
pub struct CostModel;

impl CostModel {
    pub fn new() -> Self {
        Self
    }

    pub fn calculate_implementation_cost(
        &self,
        opportunity: &OptimizationOpportunity,
        input: &AnalysisInput,
    ) -> JitResult<f64> {
        let base_cost = match opportunity.opportunity_type {
            OpportunityType::FusionOptimization => 0.4,
            OpportunityType::MemoryOptimization => 0.6,
            OpportunityType::ParallelizationOptimization => 0.8,
            OpportunityType::VectorizationOptimization => 0.5,
            OpportunityType::ConstantFolding => 0.2,
            OpportunityType::DeadCodeElimination => 0.1,
            OpportunityType::ComputationOptimization => 0.7,
        };

        // Adjust based on complexity
        let complexity_factor = opportunity.implementation_complexity;
        let system_factor = self.calculate_system_complexity_factor(input);

        let total_cost = base_cost * (1.0 + complexity_factor) * system_factor;
        Ok(total_cost.min(1.0))
    }

    pub fn estimate_performance_benefit(
        &self,
        opportunity: &OptimizationOpportunity,
        input: &AnalysisInput,
    ) -> JitResult<f64> {
        let base_benefit = opportunity.estimated_benefit;

        // Adjust based on system characteristics
        let system_multiplier = if input.system_constraints.has_gpu {
            1.2 // GPU can amplify benefits
        } else {
            1.0
        };

        let cpu_multiplier = match input.system_constraints.cpu_cores {
            1..=2 => 0.8,
            3..=8 => 1.0,
            9..=16 => 1.2,
            _ => 1.4,
        };

        let adjusted_benefit = base_benefit * system_multiplier * cpu_multiplier;
        Ok(adjusted_benefit.min(1.0))
    }

    pub fn evaluate_risks(
        &self,
        opportunity: &OptimizationOpportunity,
        input: &AnalysisInput,
    ) -> JitResult<f64> {
        let mut risk_score = 0.0;

        // Base risk based on optimization type
        risk_score += match opportunity.opportunity_type {
            OpportunityType::FusionOptimization => 0.3,
            OpportunityType::MemoryOptimization => 0.4,
            OpportunityType::ParallelizationOptimization => 0.6,
            OpportunityType::VectorizationOptimization => 0.2,
            OpportunityType::ConstantFolding => 0.1,
            OpportunityType::DeadCodeElimination => 0.1,
            OpportunityType::ComputationOptimization => 0.5,
        };

        // Complexity risk
        risk_score += opportunity.implementation_complexity * 0.3;

        // System-specific risks
        if matches!(
            input.system_constraints.target_platform,
            TargetPlatform::Embedded
        ) {
            risk_score += 0.2; // Higher risk for embedded systems
        }

        // User preference risk adjustment
        let aggressiveness = input.user_preferences.optimization_aggressiveness;
        if aggressiveness > 0.8 {
            risk_score *= 0.8; // User accepts higher risk
        } else if aggressiveness < 0.3 {
            risk_score *= 1.2; // User prefers lower risk
        }

        Ok(risk_score.min(1.0))
    }

    pub fn calculate_roi_estimates(
        &self,
        costs: &HashMap<String, f64>,
        benefits: &HashMap<String, f64>,
    ) -> JitResult<HashMap<String, f64>> {
        let mut roi_estimates = HashMap::new();

        for optimization_id in costs.keys() {
            let cost = costs.get(optimization_id).unwrap_or(&1.0);
            let benefit = benefits.get(optimization_id).unwrap_or(&0.0);

            let roi = if *cost > 0.0 { benefit / cost } else { 0.0 };

            roi_estimates.insert(optimization_id.clone(), roi);
        }

        Ok(roi_estimates)
    }

    pub fn generate_priority_rankings(
        &self,
        costs: &HashMap<String, f64>,
        benefits: &HashMap<String, f64>,
        risks: &HashMap<String, f64>,
    ) -> JitResult<Vec<(String, f64)>> {
        let mut rankings = Vec::new();

        for optimization_id in costs.keys() {
            let cost = costs.get(optimization_id).unwrap_or(&1.0);
            let benefit = benefits.get(optimization_id).unwrap_or(&0.0);
            let risk = risks.get(optimization_id).unwrap_or(&0.5);

            // Priority score: benefit / (cost + risk)
            let priority_score = if *cost + *risk > 0.0 {
                benefit / (cost + risk)
            } else {
                0.0
            };

            rankings.push((optimization_id.clone(), priority_score));
        }

        // Sort by priority score (highest first)
        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(rankings)
    }

    pub fn calculate_confidence(&self) -> f64 {
        // Cost model confidence based on available data and heuristics
        0.7
    }

    // Helper methods
    fn calculate_system_complexity_factor(&self, input: &AnalysisInput) -> f64 {
        let mut factor = 1.0;

        // Graph complexity
        if let Some(graph) = &input.computation_graph {
            let node_count = graph.node_count();
            factor *= match node_count {
                0..=50 => 1.0,
                51..=200 => 1.1,
                201..=500 => 1.2,
                _ => 1.3,
            };
        }

        // Platform complexity
        factor *= match input.system_constraints.target_platform {
            TargetPlatform::Desktop => 1.0,
            TargetPlatform::Server => 1.1,
            TargetPlatform::Mobile => 1.2,
            TargetPlatform::Embedded => 1.4,
        };

        factor
    }

    pub fn estimate_development_time(
        &self,
        opportunity: &OptimizationOpportunity,
    ) -> std::time::Duration {
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

        std::time::Duration::from_secs(total_hours * 3600)
    }

    pub fn calculate_maintenance_cost(&self, opportunity: &OptimizationOpportunity) -> f64 {
        // Maintenance cost as a fraction of implementation cost
        let base_maintenance = 0.2; // 20% of implementation cost annually

        let complexity_factor = 1.0 + opportunity.implementation_complexity * 0.5;

        // Some optimizations have higher maintenance costs
        let type_factor = match opportunity.opportunity_type {
            OpportunityType::ParallelizationOptimization => 1.5, // Parallel code is harder to maintain
            OpportunityType::MemoryOptimization => 1.3,
            OpportunityType::ComputationOptimization => 1.2,
            _ => 1.0,
        };

        base_maintenance * complexity_factor * type_factor
    }

    pub fn assess_technical_debt(&self, opportunity: &OptimizationOpportunity) -> f64 {
        // Assess how much technical debt this optimization might introduce
        let base_debt = match opportunity.opportunity_type {
            OpportunityType::FusionOptimization => 0.3, // Can make code less readable
            OpportunityType::ParallelizationOptimization => 0.5, // Complex concurrency
            OpportunityType::MemoryOptimization => 0.4, // Complex memory management
            OpportunityType::VectorizationOptimization => 0.2, // Usually well-contained
            OpportunityType::ConstantFolding => 0.1,    // Low debt
            OpportunityType::DeadCodeElimination => 0.05, // Very low debt
            OpportunityType::ComputationOptimization => 0.3,
        };

        // Complexity amplifies technical debt
        base_debt * (1.0 + opportunity.implementation_complexity)
    }
}
