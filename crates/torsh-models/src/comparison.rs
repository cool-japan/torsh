//! Model comparison and analysis tools
//!
//! This module provides comprehensive tools for comparing machine learning models
//! across various dimensions including performance, accuracy, efficiency, and resource usage.

use crate::benchmark::{BenchmarkConfig, BenchmarkResults, ModelBenchmark};
use crate::validation::{ModelValidator, TaskType, ValidationConfig, ValidationResults};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_nn::Module;
use torsh_tensor::Tensor;

/// Configuration for model comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonConfig {
    /// Models to compare
    pub models: Vec<ModelSpec>,
    /// Comparison dimensions
    pub dimensions: Vec<ComparisonDimension>,
    /// Benchmark configuration
    pub benchmark_config: Option<BenchmarkConfig>,
    /// Validation configuration
    pub validation_config: Option<ValidationConfig>,
    /// Statistical significance testing
    pub statistical_tests: bool,
    /// Generate visualizations
    pub generate_plots: bool,
    /// Output configuration
    pub output_config: OutputConfig,
}

/// Model specification for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Model identifier
    pub id: String,
    /// Model name
    pub name: String,
    /// Model description
    pub description: Option<String>,
    /// Model metadata
    pub metadata: HashMap<String, String>,
    /// Model size in parameters
    pub parameter_count: Option<usize>,
    /// Model size in bytes
    pub model_size_bytes: Option<usize>,
}

/// Dimensions for model comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonDimension {
    /// Model accuracy and performance metrics
    Accuracy,
    /// Inference speed and throughput
    Speed,
    /// Memory usage and efficiency
    Memory,
    /// Model size and complexity
    Size,
    /// Energy consumption
    Energy,
    /// Training time and resources
    Training,
    /// Robustness and reliability
    Robustness,
    /// Interpretability and explainability
    Interpretability,
    /// Custom dimension
    Custom { name: String, description: String },
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Save results to file
    pub save_results: bool,
    /// Output file path
    pub output_path: Option<String>,
    /// Generate HTML report
    pub generate_html: bool,
    /// Generate CSV export
    pub generate_csv: bool,
    /// Include detailed analysis
    pub detailed_analysis: bool,
}

/// Comprehensive comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResults {
    /// Configuration used for comparison
    pub config: ComparisonConfig,
    /// Individual model results
    pub model_results: HashMap<String, ModelComparisonResult>,
    /// Pairwise comparisons
    pub pairwise_comparisons: Vec<PairwiseComparison>,
    /// Rankings for each dimension
    pub rankings: HashMap<String, Vec<ModelRanking>>,
    /// Overall summary and recommendations
    pub summary: ComparisonSummary,
    /// Statistical analysis results
    pub statistical_analysis: Option<StatisticalAnalysis>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Results for individual model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonResult {
    /// Model specification
    pub model_spec: ModelSpec,
    /// Benchmark results
    pub benchmark_results: Option<BenchmarkResults>,
    /// Validation results
    pub validation_results: Option<ValidationResults>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Resource usage metrics
    pub resource_metrics: ResourceMetrics,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Performance metrics for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Inference latency (ms)
    pub latency_ms: f64,
    /// Throughput (samples/sec)
    pub throughput: f64,
    /// Accuracy score
    pub accuracy: f64,
    /// F1 score (if applicable)
    pub f1_score: Option<f64>,
    /// Custom performance metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Model size (bytes)
    pub model_size_bytes: usize,
    /// Parameter count
    pub parameter_count: usize,
    /// FLOPS (floating point operations)
    pub flops: Option<u64>,
    /// Energy consumption (joules)
    pub energy_consumption: Option<f64>,
}

/// Quality metrics for model assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Robustness score
    pub robustness_score: Option<f64>,
    /// Calibration score
    pub calibration_score: Option<f64>,
    /// Fairness metrics
    pub fairness_metrics: Option<HashMap<String, f64>>,
    /// Interpretability score
    pub interpretability_score: Option<f64>,
}

/// Pairwise comparison between two models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseComparison {
    /// First model ID
    pub model_a: String,
    /// Second model ID
    pub model_b: String,
    /// Comparison results by dimension
    pub dimension_comparisons: HashMap<String, DimensionComparison>,
    /// Overall preference
    pub overall_preference: ModelPreference,
    /// Statistical significance
    pub statistical_significance: Option<StatisticalSignificance>,
}

/// Comparison result for a specific dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionComparison {
    /// Dimension name
    pub dimension: String,
    /// Model A score
    pub score_a: f64,
    /// Model B score
    pub score_b: f64,
    /// Relative improvement (B vs A)
    pub relative_improvement: f64,
    /// Winner
    pub winner: ModelPreference,
    /// Confidence level
    pub confidence: f64,
}

/// Model preference in comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelPreference {
    ModelA,
    ModelB,
    Tie,
}

/// Statistical significance testing results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSignificance {
    /// Test statistic
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Is significant at alpha=0.05
    pub is_significant: bool,
    /// Effect size
    pub effect_size: f64,
}

/// Model ranking for a dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRanking {
    /// Model ID
    pub model_id: String,
    /// Rank (1 = best)
    pub rank: usize,
    /// Score for this dimension
    pub score: f64,
    /// Normalized score (0-1)
    pub normalized_score: f64,
}

/// Overall comparison summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Best model overall
    pub best_model_overall: String,
    /// Best model by dimension
    pub best_by_dimension: HashMap<String, String>,
    /// Key insights
    pub insights: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Trade-offs analysis
    pub tradeoffs: Vec<TradeoffAnalysis>,
}

/// Trade-off analysis between dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeoffAnalysis {
    /// Dimension 1
    pub dimension_a: String,
    /// Dimension 2
    pub dimension_b: String,
    /// Correlation coefficient
    pub correlation: f64,
    /// Description of trade-off
    pub description: String,
}

/// Statistical analysis across all models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    /// ANOVA results by dimension
    pub anova_results: HashMap<String, AnovaResult>,
    /// Post-hoc test results
    pub posthoc_results: HashMap<String, PosthocResult>,
    /// Correlation matrix between dimensions
    pub correlation_matrix: HashMap<String, HashMap<String, f64>>,
}

/// ANOVA test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnovaResult {
    /// F-statistic
    pub f_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom
    pub df: (usize, usize),
    /// Is significant
    pub is_significant: bool,
}

/// Post-hoc test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosthocResult {
    /// Pairwise comparisons
    pub pairwise_results: HashMap<String, HashMap<String, StatisticalSignificance>>,
    /// Multiple comparison correction applied
    pub correction_method: String,
}

/// Main model comparison engine
pub struct ModelComparator {
    config: ComparisonConfig,
}

impl ModelComparator {
    /// Create a new model comparator
    pub fn new(config: ComparisonConfig) -> Self {
        Self { config }
    }

    /// Compare multiple models
    pub fn compare_models<M: Module + Clone>(
        &self,
        models: HashMap<String, M>,
        test_dataset: &[(Tensor, Tensor)],
        task_type: TaskType,
    ) -> Result<ComparisonResults> {
        let mut model_results = HashMap::new();

        // Evaluate each model individually
        for (model_id, model) in &models {
            let result = self.evaluate_single_model(model_id, model, test_dataset, &task_type)?;
            model_results.insert(model_id.clone(), result);
        }

        // Perform pairwise comparisons
        let pairwise_comparisons = self.perform_pairwise_comparisons(&model_results)?;

        // Generate rankings
        let rankings = self.generate_rankings(&model_results)?;

        // Create summary
        let summary = self.create_summary(&model_results, &rankings)?;

        // Perform statistical analysis
        let statistical_analysis = if self.config.statistical_tests {
            Some(self.perform_statistical_analysis(&model_results)?)
        } else {
            None
        };

        Ok(ComparisonResults {
            config: self.config.clone(),
            model_results,
            pairwise_comparisons,
            rankings,
            summary,
            statistical_analysis,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Evaluate a single model
    fn evaluate_single_model<M: Module>(
        &self,
        model_id: &str,
        model: &M,
        test_dataset: &[(Tensor, Tensor)],
        task_type: &TaskType,
    ) -> Result<ModelComparisonResult> {
        // Get model specification
        let model_spec = self.get_model_spec(model_id)?;

        // Run benchmarks if configured
        let benchmark_results = if let Some(ref bench_config) = self.config.benchmark_config {
            let mut benchmark = ModelBenchmark::new(bench_config.clone());
            Some(benchmark.benchmark_model(model, model_id)?)
        } else {
            None
        };

        // Run validation if configured
        let validation_results = if let Some(ref val_config) = self.config.validation_config {
            let validator = ModelValidator::new(val_config.clone());
            Some(validator.validate(model, test_dataset, task_type.clone())?)
        } else {
            None
        };

        // Extract metrics
        let performance_metrics =
            self.extract_performance_metrics(&benchmark_results, &validation_results)?;

        let resource_metrics =
            self.extract_resource_metrics(model, &benchmark_results, &model_spec)?;

        let quality_metrics = self.extract_quality_metrics(&validation_results)?;

        Ok(ModelComparisonResult {
            model_spec,
            benchmark_results,
            validation_results,
            performance_metrics,
            resource_metrics,
            quality_metrics,
        })
    }

    /// Get model specification
    fn get_model_spec(&self, model_id: &str) -> Result<ModelSpec> {
        self.config
            .models
            .iter()
            .find(|spec| spec.id == model_id)
            .cloned()
            .ok_or_else(|| {
                TorshError::ComputeError(format!("Model spec not found for ID: {}", model_id))
            })
    }

    /// Extract performance metrics from results
    fn extract_performance_metrics(
        &self,
        benchmark_results: &Option<BenchmarkResults>,
        validation_results: &Option<ValidationResults>,
    ) -> Result<PerformanceMetrics> {
        let mut latency_ms = 0.0;
        let mut throughput = 0.0;
        let mut accuracy = 0.0;
        let mut f1_score = None;

        // Extract from benchmark results
        if let Some(bench) = benchmark_results {
            if let Some(first_metric) = bench.performance_metrics.values().next() {
                latency_ms = first_metric.avg_inference_time_ms;
                throughput = first_metric.throughput_samples_per_sec;
            }
        }

        // Extract from validation results
        if let Some(val) = validation_results {
            if let Some(acc_metric) = val.aggregated_metrics.get("accuracy") {
                accuracy = acc_metric.mean;
            }

            if let Some(f1_metric) = val.aggregated_metrics.get("f1_score_macro") {
                f1_score = Some(f1_metric.mean);
            }
        }

        Ok(PerformanceMetrics {
            latency_ms,
            throughput,
            accuracy,
            f1_score,
            custom_metrics: HashMap::new(),
        })
    }

    /// Extract resource metrics
    fn extract_resource_metrics<M: Module>(
        &self,
        model: &M,
        benchmark_results: &Option<BenchmarkResults>,
        model_spec: &ModelSpec,
    ) -> Result<ResourceMetrics> {
        let parameters = model.parameters();
        let parameter_count = parameters
            .values()
            .map(|p| p.numel())
            .collect::<torsh_core::error::Result<Vec<_>>>()?
            .iter()
            .sum::<usize>();

        let model_size_bytes = model_spec.model_size_bytes.unwrap_or(parameter_count * 4); // Assume f32

        let peak_memory_bytes = benchmark_results
            .as_ref()
            .and_then(|b| b.memory_metrics.as_ref())
            .map(|m| m.peak_memory_bytes)
            .unwrap_or(0);

        Ok(ResourceMetrics {
            peak_memory_bytes,
            model_size_bytes,
            parameter_count,
            flops: None,
            energy_consumption: None,
        })
    }

    /// Extract quality metrics
    fn extract_quality_metrics(
        &self,
        _validation_results: &Option<ValidationResults>,
    ) -> Result<QualityMetrics> {
        // Quality metrics would be computed based on specific tests
        // For now, return default values
        Ok(QualityMetrics {
            robustness_score: None,
            calibration_score: None,
            fairness_metrics: None,
            interpretability_score: None,
        })
    }

    /// Perform pairwise comparisons between models
    fn perform_pairwise_comparisons(
        &self,
        model_results: &HashMap<String, ModelComparisonResult>,
    ) -> Result<Vec<PairwiseComparison>> {
        let mut comparisons = Vec::new();
        let model_ids: Vec<_> = model_results.keys().collect();

        // Compare each pair of models
        for (i, &model_a) in model_ids.iter().enumerate() {
            for &model_b in model_ids.iter().skip(i + 1) {
                let comparison = self.compare_model_pair(
                    model_a,
                    model_b,
                    &model_results[model_a],
                    &model_results[model_b],
                )?;
                comparisons.push(comparison);
            }
        }

        Ok(comparisons)
    }

    /// Compare a pair of models
    fn compare_model_pair(
        &self,
        model_a_id: &str,
        model_b_id: &str,
        result_a: &ModelComparisonResult,
        result_b: &ModelComparisonResult,
    ) -> Result<PairwiseComparison> {
        let mut dimension_comparisons = HashMap::new();

        // Compare each dimension
        for dimension in &self.config.dimensions {
            let dim_name = self.dimension_name(dimension);
            let comparison = self.compare_dimension(dimension, result_a, result_b)?;
            dimension_comparisons.insert(dim_name, comparison);
        }

        // Determine overall preference
        let overall_preference = self.determine_overall_preference(&dimension_comparisons);

        Ok(PairwiseComparison {
            model_a: model_a_id.to_string(),
            model_b: model_b_id.to_string(),
            dimension_comparisons,
            overall_preference,
            statistical_significance: None,
        })
    }

    /// Compare models on a specific dimension
    fn compare_dimension(
        &self,
        dimension: &ComparisonDimension,
        result_a: &ModelComparisonResult,
        result_b: &ModelComparisonResult,
    ) -> Result<DimensionComparison> {
        let (score_a, score_b, higher_is_better) = match dimension {
            ComparisonDimension::Accuracy => (
                result_a.performance_metrics.accuracy,
                result_b.performance_metrics.accuracy,
                true,
            ),
            ComparisonDimension::Speed => (
                result_a.performance_metrics.throughput,
                result_b.performance_metrics.throughput,
                true,
            ),
            ComparisonDimension::Memory => {
                (
                    result_a.resource_metrics.peak_memory_bytes as f64,
                    result_b.resource_metrics.peak_memory_bytes as f64,
                    false,
                ) // Lower memory usage is better
            }
            ComparisonDimension::Size => {
                (
                    result_a.resource_metrics.parameter_count as f64,
                    result_b.resource_metrics.parameter_count as f64,
                    false,
                ) // Smaller model is better
            }
            _ => (0.0, 0.0, true), // Default for other dimensions
        };

        let relative_improvement = if score_a != 0.0 {
            (score_b - score_a) / score_a
        } else {
            0.0
        };

        let winner = if score_a == score_b {
            ModelPreference::Tie
        } else if (score_b > score_a && higher_is_better)
            || (score_b < score_a && !higher_is_better)
        {
            ModelPreference::ModelB
        } else {
            ModelPreference::ModelA
        };

        // Simple confidence calculation
        let confidence = if score_a == score_b {
            0.0
        } else {
            (score_a - score_b).abs() / (score_a + score_b).abs().max(1e-8)
        };

        Ok(DimensionComparison {
            dimension: self.dimension_name(dimension),
            score_a,
            score_b,
            relative_improvement,
            winner,
            confidence,
        })
    }

    /// Get dimension name as string
    fn dimension_name(&self, dimension: &ComparisonDimension) -> String {
        match dimension {
            ComparisonDimension::Accuracy => "accuracy".to_string(),
            ComparisonDimension::Speed => "speed".to_string(),
            ComparisonDimension::Memory => "memory".to_string(),
            ComparisonDimension::Size => "size".to_string(),
            ComparisonDimension::Energy => "energy".to_string(),
            ComparisonDimension::Training => "training".to_string(),
            ComparisonDimension::Robustness => "robustness".to_string(),
            ComparisonDimension::Interpretability => "interpretability".to_string(),
            ComparisonDimension::Custom { name, .. } => name.clone(),
        }
    }

    /// Determine overall preference from dimension comparisons
    fn determine_overall_preference(
        &self,
        dimension_comparisons: &HashMap<String, DimensionComparison>,
    ) -> ModelPreference {
        let mut a_wins = 0;
        let mut b_wins = 0;
        let mut _ties = 0;

        for comparison in dimension_comparisons.values() {
            match comparison.winner {
                ModelPreference::ModelA => a_wins += 1,
                ModelPreference::ModelB => b_wins += 1,
                ModelPreference::Tie => _ties += 1,
            }
        }

        if a_wins > b_wins {
            ModelPreference::ModelA
        } else if b_wins > a_wins {
            ModelPreference::ModelB
        } else {
            ModelPreference::Tie
        }
    }

    /// Generate rankings for each dimension
    fn generate_rankings(
        &self,
        model_results: &HashMap<String, ModelComparisonResult>,
    ) -> Result<HashMap<String, Vec<ModelRanking>>> {
        let mut rankings = HashMap::new();

        for dimension in &self.config.dimensions {
            let dim_name = self.dimension_name(dimension);
            let ranking = self.rank_models_by_dimension(dimension, model_results)?;
            rankings.insert(dim_name, ranking);
        }

        Ok(rankings)
    }

    /// Rank models by a specific dimension
    fn rank_models_by_dimension(
        &self,
        dimension: &ComparisonDimension,
        model_results: &HashMap<String, ModelComparisonResult>,
    ) -> Result<Vec<ModelRanking>> {
        let mut scores: Vec<(String, f64)> = Vec::new();

        for (model_id, result) in model_results {
            let score = match dimension {
                ComparisonDimension::Accuracy => result.performance_metrics.accuracy,
                ComparisonDimension::Speed => result.performance_metrics.throughput,
                ComparisonDimension::Memory => -(result.resource_metrics.peak_memory_bytes as f64), // Negative for ranking
                ComparisonDimension::Size => -(result.resource_metrics.parameter_count as f64), // Negative for ranking
                _ => 0.0,
            };
            scores.push((model_id.clone(), score));
        }

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Normalize scores
        let max_score = scores
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_score = scores.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
        let score_range = max_score - min_score;

        let rankings: Vec<ModelRanking> = scores
            .into_iter()
            .enumerate()
            .map(|(rank, (model_id, score))| {
                let normalized_score = if score_range > 0.0 {
                    (score - min_score) / score_range
                } else {
                    1.0
                };

                ModelRanking {
                    model_id,
                    rank: rank + 1,
                    score,
                    normalized_score,
                }
            })
            .collect();

        Ok(rankings)
    }

    /// Create comparison summary
    fn create_summary(
        &self,
        model_results: &HashMap<String, ModelComparisonResult>,
        rankings: &HashMap<String, Vec<ModelRanking>>,
    ) -> Result<ComparisonSummary> {
        // Find best model overall (based on first ranking)
        let best_model_overall = rankings
            .values()
            .next()
            .and_then(|ranking| ranking.first())
            .map(|r| r.model_id.clone())
            .unwrap_or_default();

        // Find best model by dimension
        let mut best_by_dimension = HashMap::new();
        for (dimension, ranking) in rankings {
            if let Some(best) = ranking.first() {
                best_by_dimension.insert(dimension.clone(), best.model_id.clone());
            }
        }

        // Generate insights
        let insights = self.generate_insights(model_results, rankings)?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(model_results, rankings)?;

        // Analyze trade-offs
        let tradeoffs = self.analyze_tradeoffs(model_results)?;

        Ok(ComparisonSummary {
            best_model_overall,
            best_by_dimension,
            insights,
            recommendations,
            tradeoffs,
        })
    }

    /// Generate insights from comparison results
    fn generate_insights(
        &self,
        model_results: &HashMap<String, ModelComparisonResult>,
        rankings: &HashMap<String, Vec<ModelRanking>>,
    ) -> Result<Vec<String>> {
        let mut insights = Vec::new();

        // Performance spread analysis
        if let Some(accuracy_ranking) = rankings.get("accuracy") {
            if accuracy_ranking.len() > 1 {
                let best_acc = accuracy_ranking[0].score;
                let worst_acc = accuracy_ranking.last().unwrap().score;
                let spread = best_acc - worst_acc;

                insights.push(format!(
                    "Accuracy varies by {:.2}% across models (best: {:.2}%, worst: {:.2}%)",
                    spread * 100.0,
                    best_acc * 100.0,
                    worst_acc * 100.0
                ));
            }
        }

        // Model size analysis
        let param_counts: Vec<usize> = model_results
            .values()
            .map(|r| r.resource_metrics.parameter_count)
            .collect();

        if param_counts.len() > 1 {
            let max_params = *param_counts.iter().max().unwrap();
            let min_params = *param_counts.iter().min().unwrap();
            let ratio = max_params as f64 / min_params as f64;

            insights.push(format!(
                "Model sizes vary by {:.1}x (largest: {}M params, smallest: {}M params)",
                ratio,
                max_params / 1_000_000,
                min_params / 1_000_000
            ));
        }

        // Speed analysis
        if let Some(speed_ranking) = rankings.get("speed") {
            if speed_ranking.len() > 1 {
                let fastest = speed_ranking[0].score;
                let slowest = speed_ranking.last().unwrap().score;
                let speedup = fastest / slowest;

                insights.push(format!(
                    "Fastest model is {:.1}x faster than slowest ({:.1} vs {:.1} samples/sec)",
                    speedup, fastest, slowest
                ));
            }
        }

        Ok(insights)
    }

    /// Generate recommendations
    fn generate_recommendations(
        &self,
        _model_results: &HashMap<String, ModelComparisonResult>,
        rankings: &HashMap<String, Vec<ModelRanking>>,
    ) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        // Best overall recommendation
        if let Some(accuracy_ranking) = rankings.get("accuracy") {
            if let Some(best_model) = accuracy_ranking.first() {
                recommendations.push(format!(
                    "For highest accuracy, use {} ({:.2}% accuracy)",
                    best_model.model_id,
                    best_model.score * 100.0
                ));
            }
        }

        // Speed recommendation
        if let Some(speed_ranking) = rankings.get("speed") {
            if let Some(fastest_model) = speed_ranking.first() {
                recommendations.push(format!(
                    "For fastest inference, use {} ({:.1} samples/sec)",
                    fastest_model.model_id, fastest_model.score
                ));
            }
        }

        // Memory efficiency recommendation
        if let Some(memory_ranking) = rankings.get("memory") {
            if let Some(most_efficient) = memory_ranking.first() {
                recommendations.push(format!(
                    "For memory efficiency, use {} (lowest memory usage)",
                    most_efficient.model_id
                ));
            }
        }

        Ok(recommendations)
    }

    /// Analyze trade-offs between dimensions
    fn analyze_tradeoffs(
        &self,
        model_results: &HashMap<String, ModelComparisonResult>,
    ) -> Result<Vec<TradeoffAnalysis>> {
        let mut tradeoffs = Vec::new();

        // Accuracy vs Speed trade-off
        let accuracy_speed_corr = self.compute_correlation(
            model_results,
            |r| r.performance_metrics.accuracy,
            |r| r.performance_metrics.throughput,
        );

        tradeoffs.push(TradeoffAnalysis {
            dimension_a: "accuracy".to_string(),
            dimension_b: "speed".to_string(),
            correlation: accuracy_speed_corr,
            description: if accuracy_speed_corr < -0.5 {
                "Strong negative correlation: more accurate models tend to be slower".to_string()
            } else if accuracy_speed_corr > 0.5 {
                "Strong positive correlation: more accurate models tend to be faster".to_string()
            } else {
                "Weak correlation between accuracy and speed".to_string()
            },
        });

        // Accuracy vs Size trade-off
        let accuracy_size_corr = self.compute_correlation(
            model_results,
            |r| r.performance_metrics.accuracy,
            |r| -(r.resource_metrics.parameter_count as f64), // Negative for inverse relationship
        );

        tradeoffs.push(TradeoffAnalysis {
            dimension_a: "accuracy".to_string(),
            dimension_b: "size".to_string(),
            correlation: accuracy_size_corr,
            description: if accuracy_size_corr < -0.5 {
                "Strong trade-off: larger models tend to be more accurate".to_string()
            } else {
                "Model size doesn't strongly correlate with accuracy".to_string()
            },
        });

        Ok(tradeoffs)
    }

    /// Compute correlation between two metrics
    fn compute_correlation<F1, F2>(
        &self,
        model_results: &HashMap<String, ModelComparisonResult>,
        extract_x: F1,
        extract_y: F2,
    ) -> f64
    where
        F1: Fn(&ModelComparisonResult) -> f64,
        F2: Fn(&ModelComparisonResult) -> f64,
    {
        let data: Vec<(f64, f64)> = model_results
            .values()
            .map(|r| (extract_x(r), extract_y(r)))
            .collect();

        if data.len() < 2 {
            return 0.0;
        }

        let n = data.len() as f64;
        let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
        let sum_x_sq: f64 = data.iter().map(|(x, _)| x * x).sum();
        let sum_y_sq: f64 = data.iter().map(|(_, y)| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Perform statistical analysis
    fn perform_statistical_analysis(
        &self,
        _model_results: &HashMap<String, ModelComparisonResult>,
    ) -> Result<StatisticalAnalysis> {
        // Statistical analysis would be implemented here
        // For now, return empty results
        Ok(StatisticalAnalysis {
            anova_results: HashMap::new(),
            posthoc_results: HashMap::new(),
            correlation_matrix: HashMap::new(),
        })
    }
}

impl std::fmt::Display for ComparisonResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Model Comparison Results")?;
        writeln!(f, "=======================")?;
        writeln!(f, "Timestamp: {}", self.timestamp)?;
        writeln!(f, "Models compared: {}", self.model_results.len())?;
        writeln!(f)?;

        writeln!(f, "Overall Best Model: {}", self.summary.best_model_overall)?;
        writeln!(f)?;

        writeln!(f, "Best by Dimension:")?;
        for (dimension, model) in &self.summary.best_by_dimension {
            writeln!(f, "  {}: {}", dimension, model)?;
        }
        writeln!(f)?;

        writeln!(f, "Key Insights:")?;
        for insight in &self.summary.insights {
            writeln!(f, "  • {}", insight)?;
        }
        writeln!(f)?;

        writeln!(f, "Recommendations:")?;
        for recommendation in &self.summary.recommendations {
            writeln!(f, "  • {}", recommendation)?;
        }

        Ok(())
    }
}

/// Utility functions for model comparison
pub mod comparison_utils {
    use super::*;

    /// Create a standard comparison configuration
    pub fn create_standard_comparison_config(model_specs: Vec<ModelSpec>) -> ComparisonConfig {
        ComparisonConfig {
            models: model_specs,
            dimensions: vec![
                ComparisonDimension::Accuracy,
                ComparisonDimension::Speed,
                ComparisonDimension::Memory,
                ComparisonDimension::Size,
            ],
            benchmark_config: Some(crate::benchmark::benchmark_utils::quick_benchmark_config()),
            validation_config: Some(crate::validation::validation_utils::create_quick_config()),
            statistical_tests: true,
            generate_plots: false,
            output_config: OutputConfig {
                save_results: true,
                output_path: Some("comparison_results.json".to_string()),
                generate_html: false,
                generate_csv: true,
                detailed_analysis: true,
            },
        }
    }

    /// Create a comprehensive comparison configuration
    pub fn create_comprehensive_comparison_config(model_specs: Vec<ModelSpec>) -> ComparisonConfig {
        ComparisonConfig {
            models: model_specs,
            dimensions: vec![
                ComparisonDimension::Accuracy,
                ComparisonDimension::Speed,
                ComparisonDimension::Memory,
                ComparisonDimension::Size,
                ComparisonDimension::Energy,
                ComparisonDimension::Training,
                ComparisonDimension::Robustness,
                ComparisonDimension::Interpretability,
            ],
            benchmark_config: Some(
                crate::benchmark::benchmark_utils::comprehensive_benchmark_config(),
            ),
            validation_config: Some(
                crate::validation::validation_utils::create_classification_config(5),
            ),
            statistical_tests: true,
            generate_plots: true,
            output_config: OutputConfig {
                save_results: true,
                output_path: Some("comprehensive_comparison_results.json".to_string()),
                generate_html: true,
                generate_csv: true,
                detailed_analysis: true,
            },
        }
    }

    /// Export comparison results to CSV
    pub fn export_to_csv(results: &ComparisonResults, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        // Header
        writeln!(file, "Model,Accuracy,Speed,Memory,Size,Parameters")?;

        // Data rows
        for (model_id, result) in &results.model_results {
            writeln!(
                file,
                "{},{:.4},{:.2},{},{},{}",
                model_id,
                result.performance_metrics.accuracy,
                result.performance_metrics.throughput,
                result.resource_metrics.peak_memory_bytes,
                result.resource_metrics.model_size_bytes,
                result.resource_metrics.parameter_count
            )?;
        }

        Ok(())
    }

    /// Generate HTML report
    pub fn generate_html_report(results: &ComparisonResults, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(
            file,
            "<html><head><title>Model Comparison Report</title></head><body>"
        )?;
        writeln!(file, "<h1>Model Comparison Report</h1>")?;
        writeln!(file, "<p>Generated: {}</p>", results.timestamp)?;

        // Summary table
        writeln!(file, "<h2>Summary</h2>")?;
        writeln!(file, "<table border='1'>")?;
        writeln!(file, "<tr><th>Model</th><th>Accuracy</th><th>Speed</th><th>Memory</th><th>Parameters</th></tr>")?;

        for (model_id, result) in &results.model_results {
            writeln!(
                file,
                "<tr><td>{}</td><td>{:.4}</td><td>{:.2}</td><td>{}</td><td>{}</td></tr>",
                model_id,
                result.performance_metrics.accuracy,
                result.performance_metrics.throughput,
                result.resource_metrics.peak_memory_bytes,
                result.resource_metrics.parameter_count
            )?;
        }

        writeln!(file, "</table>")?;

        // Insights
        writeln!(file, "<h2>Key Insights</h2>")?;
        writeln!(file, "<ul>")?;
        for insight in &results.summary.insights {
            writeln!(file, "<li>{}</li>", insight)?;
        }
        writeln!(file, "</ul>")?;

        writeln!(file, "</body></html>")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_config_creation() {
        let model_specs = vec![ModelSpec {
            id: "model1".to_string(),
            name: "Model 1".to_string(),
            description: None,
            metadata: HashMap::new(),
            parameter_count: Some(1000000),
            model_size_bytes: Some(4000000),
        }];

        let config = comparison_utils::create_standard_comparison_config(model_specs);
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.dimensions.len(), 4);
    }

    #[test]
    fn test_performance_metrics_extraction() {
        let config = ComparisonConfig {
            models: vec![],
            dimensions: vec![],
            benchmark_config: None,
            validation_config: None,
            statistical_tests: false,
            generate_plots: false,
            output_config: OutputConfig {
                save_results: false,
                output_path: None,
                generate_html: false,
                generate_csv: false,
                detailed_analysis: false,
            },
        };

        let comparator = ModelComparator::new(config);
        let metrics = comparator
            .extract_performance_metrics(&None, &None)
            .unwrap();

        assert_eq!(metrics.accuracy, 0.0);
        assert_eq!(metrics.throughput, 0.0);
    }

    #[test]
    fn test_correlation_computation() {
        let config = ComparisonConfig {
            models: vec![],
            dimensions: vec![],
            benchmark_config: None,
            validation_config: None,
            statistical_tests: false,
            generate_plots: false,
            output_config: OutputConfig {
                save_results: false,
                output_path: None,
                generate_html: false,
                generate_csv: false,
                detailed_analysis: false,
            },
        };

        let comparator = ModelComparator::new(config);
        let model_results = HashMap::new();

        let correlation = comparator.compute_correlation(
            &model_results,
            |r| r.performance_metrics.accuracy,
            |r| r.performance_metrics.throughput,
        );

        assert_eq!(correlation, 0.0); // Empty data should return 0
    }

    #[test]
    fn test_dimension_comparison() {
        let config = ComparisonConfig {
            models: vec![],
            dimensions: vec![],
            benchmark_config: None,
            validation_config: None,
            statistical_tests: false,
            generate_plots: false,
            output_config: OutputConfig {
                save_results: false,
                output_path: None,
                generate_html: false,
                generate_csv: false,
                detailed_analysis: false,
            },
        };

        let comparator = ModelComparator::new(config);

        let result_a = ModelComparisonResult {
            model_spec: ModelSpec {
                id: "a".to_string(),
                name: "A".to_string(),
                description: None,
                metadata: HashMap::new(),
                parameter_count: None,
                model_size_bytes: None,
            },
            benchmark_results: None,
            validation_results: None,
            performance_metrics: PerformanceMetrics {
                latency_ms: 100.0,
                throughput: 10.0,
                accuracy: 0.8,
                f1_score: None,
                custom_metrics: HashMap::new(),
            },
            resource_metrics: ResourceMetrics {
                peak_memory_bytes: 1000000,
                model_size_bytes: 4000000,
                parameter_count: 1000000,
                flops: None,
                energy_consumption: None,
            },
            quality_metrics: QualityMetrics {
                robustness_score: None,
                calibration_score: None,
                fairness_metrics: None,
                interpretability_score: None,
            },
        };

        let result_b = ModelComparisonResult {
            performance_metrics: PerformanceMetrics {
                accuracy: 0.9, // Better accuracy
                ..result_a.performance_metrics.clone()
            },
            ..result_a.clone()
        };

        let comparison = comparator
            .compare_dimension(&ComparisonDimension::Accuracy, &result_a, &result_b)
            .unwrap();

        assert_eq!(comparison.score_a, 0.8);
        assert_eq!(comparison.score_b, 0.9);
        assert!(matches!(comparison.winner, ModelPreference::ModelB));
    }
}
