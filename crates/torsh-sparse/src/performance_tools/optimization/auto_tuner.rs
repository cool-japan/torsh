//! Auto-tuning functionality for sparse tensor optimization

use crate::{SparseFormat, TorshResult};
use std::collections::HashMap;
use torsh_core::TorshError;

use super::super::core::PerformanceMeasurement;
use super::types::{
    DistributionPattern, InputCharacteristics, OperationType, OptimizationStrategy, TuningResult,
};

/// Automatic performance tuner for sparse tensor operations
///
/// The `AutoTuner` uses machine learning-inspired techniques to automatically
/// select optimal sparse formats and operation parameters based on data
/// characteristics and hardware capabilities.
#[derive(Debug)]
pub struct AutoTuner {
    /// Performance history for learning
    performance_history: Vec<TuningResult>,
    /// Current optimization strategy
    strategy: OptimizationStrategy,
    /// Hardware-specific optimizations
    hardware_optimizations: HashMap<String, f64>,
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoTuner {
    /// Create a new auto-tuner with default optimization strategy
    pub fn new() -> Self {
        Self {
            performance_history: Vec::new(),
            strategy: OptimizationStrategy::Balanced,
            hardware_optimizations: HashMap::new(),
        }
    }

    /// Set the optimization strategy
    pub fn with_strategy(mut self, strategy: OptimizationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Find optimal sparse format for given input characteristics
    pub fn find_optimal_format(
        &mut self,
        characteristics: &InputCharacteristics,
    ) -> TorshResult<TuningResult> {
        // Analyze input characteristics
        let mut candidate_scores = HashMap::new();

        // Score each format based on characteristics
        candidate_scores.insert(SparseFormat::Coo, self.score_coo_format(characteristics));
        candidate_scores.insert(SparseFormat::Csr, self.score_csr_format(characteristics));
        candidate_scores.insert(SparseFormat::Csc, self.score_csc_format(characteristics));

        // Find the best format
        let (best_format, best_score) = candidate_scores
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("score comparison should succeed"))
            .ok_or_else(|| TorshError::InvalidArgument("No valid format found".to_string()))?;

        // Generate reasoning
        let reasoning = self.generate_reasoning(&best_format, characteristics);

        // Calculate confidence based on score margin and historical performance
        let confidence = self.calculate_confidence(best_score, characteristics);

        let result = TuningResult {
            input_characteristics: characteristics.clone(),
            recommended_format: best_format,
            performance_score: best_score,
            confidence,
            reasoning,
        };

        // Add to performance history for learning
        self.performance_history.push(result.clone());

        Ok(result)
    }

    /// Get optimization recommendations based on current system
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Strategy-specific recommendations
        match &self.strategy {
            OptimizationStrategy::Speed => {
                recommendations.push(
                    "Optimizing for maximum speed - consider CSR format for row-wise operations"
                        .to_string(),
                );
                recommendations.push("Enable SIMD optimizations if available".to_string());
            }
            OptimizationStrategy::Memory => {
                recommendations.push(
                    "Optimizing for minimal memory usage - consider format with best compression"
                        .to_string(),
                );
                recommendations
                    .push("Use lazy evaluation for large intermediate results".to_string());
            }
            OptimizationStrategy::Balanced => {
                recommendations.push(
                    "Using balanced optimization - consider workload-specific tuning".to_string(),
                );
            }
            OptimizationStrategy::CacheEfficient => {
                recommendations.push("Optimizing for cache efficiency - prefer formats with sequential access patterns".to_string());
                recommendations.push("Consider matrix reordering for better locality".to_string());
            }
            OptimizationStrategy::Custom {
                speed_weight,
                memory_weight,
                cache_weight,
            } => {
                recommendations.push(format!(
                    "Using custom strategy - speed: {:.1}, memory: {:.1}, cache: {:.1}",
                    speed_weight, memory_weight, cache_weight
                ));
            }
        }

        // Historical performance insights
        if self.performance_history.len() > 5 {
            let avg_confidence: f64 = self
                .performance_history
                .iter()
                .map(|r| r.confidence)
                .sum::<f64>()
                / self.performance_history.len() as f64;

            if avg_confidence < 0.7 {
                recommendations.push(
                    "Consider collecting more performance data for better tuning confidence"
                        .to_string(),
                );
            }
        }

        // Hardware-specific recommendations
        for (hardware_feature, score) in &self.hardware_optimizations {
            if *score > 0.8 {
                recommendations.push(format!(
                    "Excellent {} support detected - leverage this capability",
                    hardware_feature
                ));
            }
        }

        recommendations
    }

    /// Learn from actual performance results to improve future recommendations
    pub fn learn_from_result(
        &mut self,
        actual_performance: &PerformanceMeasurement,
        predicted_result: &TuningResult,
    ) {
        // This is a simplified learning algorithm
        // In practice, this would use more sophisticated ML techniques

        let actual_score = 1.0 / actual_performance.duration.as_secs_f64(); // Higher is better
        let predicted_score = predicted_result.performance_score;

        // Update confidence based on prediction accuracy
        let accuracy = 1.0 - (actual_score - predicted_score).abs() / predicted_score.max(0.001);

        // Adjust future predictions (simplified)
        if accuracy < 0.5 {
            // Poor prediction - consider adjusting strategy
            println!(
                "Warning: Poor prediction accuracy ({:.2}), consider strategy adjustment",
                accuracy
            );
        }

        // Store learning result for future improvements
        // In a real implementation, this would update internal models
    }

    // Scoring functions for different formats

    /// Score COO format based on input characteristics
    fn score_coo_format(&self, characteristics: &InputCharacteristics) -> f64 {
        let mut score: f64 = 0.5; // Base score

        // COO is good for construction and format conversion
        if characteristics
            .operation_types
            .contains(&OperationType::ElementWise)
        {
            score += 0.3;
        }

        // COO is memory-efficient for very sparse matrices
        if characteristics.sparsity > 0.95 {
            score += 0.2;
        }

        // COO handles random patterns well
        if characteristics.distribution_pattern == DistributionPattern::Random {
            score += 0.2;
        }

        // Penalize for operations that require sorted access
        if characteristics
            .operation_types
            .contains(&OperationType::MatrixVector)
        {
            score -= 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    /// Score CSR format based on input characteristics
    fn score_csr_format(&self, characteristics: &InputCharacteristics) -> f64 {
        let mut score: f64 = 0.6; // Higher base score

        // CSR is excellent for row-wise operations
        if characteristics
            .operation_types
            .contains(&OperationType::MatrixVector)
        {
            score += 0.3;
        }

        if characteristics
            .operation_types
            .contains(&OperationType::MatrixMatrix)
        {
            score += 0.2;
        }

        // CSR is good for row-clustered data
        if characteristics.distribution_pattern == DistributionPattern::RowClustered {
            score += 0.3;
        }

        // CSR benefits from iterative solvers
        if characteristics
            .operation_types
            .contains(&OperationType::IterativeSolver)
        {
            score += 0.2;
        }

        // Memory strategy considerations
        match self.strategy {
            OptimizationStrategy::Speed => score += 0.1,
            OptimizationStrategy::Memory => {
                if characteristics.sparsity > 0.9 {
                    score += 0.1;
                } else {
                    score -= 0.1;
                }
            }
            _ => {}
        }

        score.clamp(0.0, 1.0)
    }

    /// Score CSC format based on input characteristics
    fn score_csc_format(&self, characteristics: &InputCharacteristics) -> f64 {
        let mut score: f64 = 0.5; // Base score

        // CSC is good for column-wise operations
        if characteristics
            .operation_types
            .contains(&OperationType::Transpose)
        {
            score += 0.3;
        }

        // CSC is good for column-clustered data
        if characteristics.distribution_pattern == DistributionPattern::ColumnClustered {
            score += 0.3;
        }

        // CSC can be beneficial for certain matrix-matrix operations
        if characteristics
            .operation_types
            .contains(&OperationType::MatrixMatrix)
        {
            score += 0.1;
        }

        // Factorization often benefits from CSC
        if characteristics
            .operation_types
            .contains(&OperationType::Factorization)
        {
            score += 0.2;
        }

        score.clamp(0.0, 1.0)
    }

    /// Generate reasoning for format recommendation
    fn generate_reasoning(
        &self,
        format: &SparseFormat,
        characteristics: &InputCharacteristics,
    ) -> Vec<String> {
        let mut reasoning = Vec::new();

        match format {
            SparseFormat::Coo => {
                reasoning.push("COO format selected for flexible element access".to_string());
                if characteristics.sparsity > 0.95 {
                    reasoning.push("Very high sparsity favors COO's memory efficiency".to_string());
                }
                if characteristics.distribution_pattern == DistributionPattern::Random {
                    reasoning
                        .push("Random distribution pattern is well-suited for COO".to_string());
                }
            }
            SparseFormat::Csr => {
                reasoning.push("CSR format selected for optimal row-wise operations".to_string());
                if characteristics
                    .operation_types
                    .contains(&OperationType::MatrixVector)
                {
                    reasoning
                        .push("Matrix-vector operations are highly optimized in CSR".to_string());
                }
                if characteristics.distribution_pattern == DistributionPattern::RowClustered {
                    reasoning
                        .push("Row-clustered data structure aligns with CSR layout".to_string());
                }
            }
            SparseFormat::Csc => {
                reasoning.push("CSC format selected for column-oriented operations".to_string());
                if characteristics
                    .operation_types
                    .contains(&OperationType::Transpose)
                {
                    reasoning
                        .push("Transpose operations are efficient with CSC storage".to_string());
                }
                if characteristics.distribution_pattern == DistributionPattern::ColumnClustered {
                    reasoning
                        .push("Column-clustered structure benefits from CSC format".to_string());
                }
            }
            _ => {
                reasoning.push(
                    "Alternative sparse format selected based on characteristics".to_string(),
                );
            }
        }

        // Strategy-specific reasoning
        match self.strategy {
            OptimizationStrategy::Speed => {
                reasoning.push(
                    "Speed optimization strategy prioritizes computational efficiency".to_string(),
                );
            }
            OptimizationStrategy::Memory => {
                reasoning
                    .push("Memory optimization strategy minimizes storage overhead".to_string());
            }
            OptimizationStrategy::CacheEfficient => {
                reasoning.push(
                    "Cache-efficient strategy optimizes for memory access patterns".to_string(),
                );
            }
            _ => {}
        }

        reasoning
    }

    /// Calculate confidence in recommendation
    fn calculate_confidence(&self, score: f64, characteristics: &InputCharacteristics) -> f64 {
        let mut confidence = score;

        // Higher confidence for well-characterized inputs
        match characteristics.distribution_pattern {
            DistributionPattern::Random | DistributionPattern::Mixed => confidence *= 0.8,
            _ => confidence *= 1.0,
        }

        // Higher confidence with more operation types specified
        if characteristics.operation_types.len() > 2 {
            confidence *= 1.1;
        } else if characteristics.operation_types.is_empty() {
            confidence *= 0.7;
        }

        // Historical performance affects confidence
        if self.performance_history.len() > 10 {
            let historical_accuracy: f64 = self
                .performance_history
                .iter()
                .map(|r| r.confidence)
                .sum::<f64>()
                / self.performance_history.len() as f64;
            confidence = (confidence + historical_accuracy) / 2.0;
        }

        confidence.clamp(0.0, 1.0)
    }
}
