//! Multi-objective optimization for adaptive quantization
//!
//! This module provides Pareto-optimal solutions for balancing accuracy,
//! performance, and energy efficiency in quantization parameter selection.

use super::config::{
    AdaptiveQuantConfig, ConstraintHandler, OptimizationTarget, QuantizationParameters,
};
use crate::TorshResult;
use std::time::Instant;

/// Multi-objective optimizer
#[derive(Debug, Clone)]
pub struct MultiObjectiveOptimizer {
    /// Pareto frontier solutions
    #[allow(dead_code)]
    pareto_solutions: Vec<ParetoSolution>,
    /// Current optimization target
    current_target: OptimizationTarget,
    /// Optimization history
    optimization_history: Vec<OptimizationStep>,
    /// Constraint handler
    #[allow(dead_code)]
    constraint_handler: ConstraintHandler,
}

/// Pareto-optimal solution
#[derive(Debug, Clone)]
pub struct ParetoSolution {
    /// Quantization parameters
    pub parameters: QuantizationParameters,
    /// Objective values [accuracy, performance, energy]
    pub objectives: [f32; 3],
    /// Dominance rank
    pub rank: usize,
    /// Crowding distance
    pub crowding_distance: f32,
}

/// Optimization step record
#[derive(Debug, Clone)]
pub struct OptimizationStep {
    /// Parameters before optimization
    pub before: QuantizationParameters,
    /// Parameters after optimization
    pub after: QuantizationParameters,
    /// Improvement achieved
    pub improvement: [f32; 3],
    /// Step timestamp
    pub timestamp: Instant,
}

impl MultiObjectiveOptimizer {
    /// Create new multi-objective optimizer
    pub fn new() -> Self {
        Self {
            pareto_solutions: Vec::new(),
            current_target: OptimizationTarget::default(),
            optimization_history: Vec::new(),
            constraint_handler: ConstraintHandler::default(),
        }
    }

    /// Optimize parameters using multi-objective approach
    pub fn optimize_parameters(
        &mut self,
        initial_params: &QuantizationParameters,
        current_pattern: &Option<String>,
        config: &AdaptiveQuantConfig,
    ) -> TorshResult<QuantizationParameters> {
        let before = initial_params.clone();

        // Generate candidate solutions
        let candidates = self.generate_candidates(initial_params, current_pattern, config)?;

        // Evaluate objectives for each candidate
        let mut solutions = Vec::new();
        for candidate in candidates {
            let objectives = self.evaluate_objectives(&candidate, config);
            solutions.push(ParetoSolution {
                parameters: candidate,
                objectives,
                rank: 0,
                crowding_distance: 0.0,
            });
        }

        // Perform Pareto ranking and selection
        let pareto_front = self.find_pareto_front(&mut solutions);

        // Select best solution based on current target
        let optimized_params = self.select_best_solution(&pareto_front, config)?;

        // Calculate improvement
        let after_objectives = self.evaluate_objectives(&optimized_params, config);
        let before_objectives = self.evaluate_objectives(initial_params, config);
        let improvement = [
            after_objectives[0] - before_objectives[0],
            after_objectives[1] - before_objectives[1],
            after_objectives[2] - before_objectives[2],
        ];

        // Record optimization step
        let step = OptimizationStep {
            before,
            after: optimized_params.clone(),
            improvement,
            timestamp: Instant::now(),
        };
        self.optimization_history.push(step);

        // Keep history bounded
        if self.optimization_history.len() > 1000 {
            self.optimization_history.remove(0);
        }

        Ok(optimized_params)
    }

    /// Generate candidate solutions around initial parameters
    fn generate_candidates(
        &self,
        initial: &QuantizationParameters,
        pattern: &Option<String>,
        _config: &AdaptiveQuantConfig,
    ) -> TorshResult<Vec<QuantizationParameters>> {
        let mut candidates = vec![initial.clone()];

        // Generate variations based on current pattern
        let scale_variations = match pattern {
            Some(p) if p == "compute_intensive" => vec![0.8, 0.9, 1.0, 1.1, 1.2],
            Some(p) if p == "memory_bound" => vec![0.9, 1.0, 1.1],
            _ => vec![0.9, 1.0, 1.1],
        };

        let bit_width_variations = match pattern {
            Some(p) if p == "compute_intensive" => vec![6, 8, 10],
            Some(p) if p == "memory_bound" => vec![8, 12, 16],
            _ => vec![8],
        };

        // Generate parameter combinations
        for scale_factor in scale_variations {
            for &bit_width in &bit_width_variations {
                let mut candidate = initial.clone();
                candidate.scale *= scale_factor;
                candidate.bit_width = bit_width.max(4).min(16);

                // Adjust zero point based on scale changes
                if scale_factor < 1.0 {
                    candidate.zero_point =
                        (candidate.zero_point as f32 * scale_factor).round() as i32;
                }
                candidate.zero_point = candidate.zero_point.clamp(-128, 127);

                candidates.push(candidate);
            }
        }

        Ok(candidates)
    }

    /// Evaluate objectives: [accuracy, performance, energy_efficiency]
    fn evaluate_objectives(
        &self,
        params: &QuantizationParameters,
        config: &AdaptiveQuantConfig,
    ) -> [f32; 3] {
        // Accuracy estimation (higher bit width = higher accuracy)
        let accuracy_score = match params.bit_width {
            4 => 0.85,
            8 => 0.92,
            12 => 0.96,
            16 => 0.99,
            _ => 0.90,
        };

        // Performance estimation (lower bit width = higher performance)
        let performance_score = match params.bit_width {
            4 => 1.0,
            8 => 0.8,
            12 => 0.6,
            16 => 0.4,
            _ => 0.7,
        };

        // Energy efficiency (lower bit width = better efficiency)
        let energy_efficiency = match params.bit_width {
            4 => 0.95,
            8 => 0.85,
            12 => 0.70,
            16 => 0.60,
            _ => 0.75,
        };

        // Apply configuration weights
        [
            accuracy_score * config.accuracy_weight,
            performance_score * config.performance_weight,
            energy_efficiency * config.energy_weight,
        ]
    }

    /// Find Pareto-optimal solutions
    fn find_pareto_front(&self, solutions: &mut [ParetoSolution]) -> Vec<ParetoSolution> {
        let mut pareto_front = Vec::new();

        for solution in solutions.iter() {
            let mut is_dominated = false;

            // Check if this solution is dominated by any other
            for other in solutions.iter() {
                if self.dominates(&other.objectives, &solution.objectives) {
                    is_dominated = true;
                    break;
                }
            }

            if !is_dominated {
                pareto_front.push(solution.clone());
            }
        }

        pareto_front
    }

    /// Check if objectives1 dominates objectives2
    fn dominates(&self, obj1: &[f32; 3], obj2: &[f32; 3]) -> bool {
        let mut at_least_one_better = false;
        for i in 0..3 {
            if obj1[i] < obj2[i] {
                return false; // obj1 is worse in at least one objective
            }
            if obj1[i] > obj2[i] {
                at_least_one_better = true;
            }
        }
        at_least_one_better
    }

    /// Select best solution from Pareto front based on weighted preferences
    fn select_best_solution(
        &self,
        pareto_front: &[ParetoSolution],
        config: &AdaptiveQuantConfig,
    ) -> TorshResult<QuantizationParameters> {
        if pareto_front.is_empty() {
            return Ok(QuantizationParameters::default());
        }

        let weights = [
            config.accuracy_weight,
            config.performance_weight,
            config.energy_weight,
        ];
        let mut best_solution = &pareto_front[0];
        let mut best_score = self.calculate_weighted_score(&pareto_front[0].objectives, &weights);

        for solution in pareto_front.iter().skip(1) {
            let score = self.calculate_weighted_score(&solution.objectives, &weights);
            if score > best_score {
                best_score = score;
                best_solution = solution;
            }
        }

        Ok(best_solution.parameters.clone())
    }

    /// Calculate weighted score for solution selection
    fn calculate_weighted_score(&self, objectives: &[f32; 3], weights: &[f32; 3]) -> f32 {
        objectives[0] * weights[0] + objectives[1] * weights[1] + objectives[2] * weights[2]
    }

    /// Update optimization target
    pub fn update_target(&mut self, target: OptimizationTarget) {
        self.current_target = target;
    }

    /// Get optimization history
    pub fn get_optimization_history(&self) -> &[OptimizationStep] {
        &self.optimization_history
    }

    /// Get current Pareto solutions
    pub fn get_pareto_solutions(&self) -> &[ParetoSolution] {
        &self.pareto_solutions
    }

    /// Calculate optimization statistics
    pub fn get_optimization_statistics(&self) -> OptimizationStatistics {
        if self.optimization_history.is_empty() {
            return OptimizationStatistics::default();
        }

        let recent_count = self.optimization_history.len().min(10);
        let recent_steps: Vec<&OptimizationStep> = self
            .optimization_history
            .iter()
            .rev()
            .take(recent_count)
            .collect();

        // Calculate average improvements
        let avg_accuracy_improvement =
            recent_steps.iter().map(|s| s.improvement[0]).sum::<f32>() / recent_count as f32;

        let avg_performance_improvement =
            recent_steps.iter().map(|s| s.improvement[1]).sum::<f32>() / recent_count as f32;

        let avg_energy_improvement =
            recent_steps.iter().map(|s| s.improvement[2]).sum::<f32>() / recent_count as f32;

        OptimizationStatistics {
            total_optimizations: self.optimization_history.len(),
            avg_accuracy_improvement,
            avg_performance_improvement,
            avg_energy_improvement,
            pareto_solutions_count: self.pareto_solutions.len(),
        }
    }

    /// Clear optimization history
    pub fn clear_history(&mut self) {
        self.optimization_history.clear();
        self.pareto_solutions.clear();
    }
}

impl Default for MultiObjectiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStatistics {
    pub total_optimizations: usize,
    pub avg_accuracy_improvement: f32,
    pub avg_performance_improvement: f32,
    pub avg_energy_improvement: f32,
    pub pareto_solutions_count: usize,
}

impl Default for OptimizationStatistics {
    fn default() -> Self {
        Self {
            total_optimizations: 0,
            avg_accuracy_improvement: 0.0,
            avg_performance_improvement: 0.0,
            avg_energy_improvement: 0.0,
            pareto_solutions_count: 0,
        }
    }
}
