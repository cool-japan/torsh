//! Knowledge base and learning systems for optimization advice

use crate::advisor::config::*;
use crate::JitResult;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Knowledge base and learning system for optimization advice
pub struct LearningSystem {
    knowledge_base: KnowledgeBase,
    historical_data: HistoricalDataStore,
    recommendation_feedback: FeedbackTracker,
    adaptation_engine: AdaptationEngine,
    config: LearningConfig,
}

/// Configuration for the learning system
#[derive(Debug, Clone)]
pub struct LearningConfig {
    pub max_history_size: usize,
    pub learning_rate: f64,
    pub adaptation_threshold: f64,
    pub min_feedback_samples: usize,
    pub enable_pattern_learning: bool,
    pub enable_performance_prediction: bool,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            max_history_size: 10000,
            learning_rate: 0.01,
            adaptation_threshold: 0.1,
            min_feedback_samples: 10,
            enable_pattern_learning: true,
            enable_performance_prediction: true,
        }
    }
}

/// Knowledge base storing optimization patterns and strategies
#[derive(Debug)]
pub struct KnowledgeBase {
    optimization_patterns: HashMap<String, OptimizationPattern>,
    performance_models: HashMap<String, PerformanceModel>,
    best_practices: Vec<BestPractice>,
    failure_cases: Vec<FailureCase>,
}

/// Historical data store for tracking analysis results
#[derive(Debug)]
pub struct HistoricalDataStore {
    analysis_history: VecDeque<AnalysisRecord>,
    performance_history: VecDeque<PerformanceRecord>,
    recommendation_history: VecDeque<RecommendationRecord>,
    max_size: usize,
}

/// Feedback tracking for recommendation quality
#[derive(Debug)]
pub struct FeedbackTracker {
    recommendation_feedback: HashMap<String, Vec<FeedbackEntry>>,
    success_rates: HashMap<String, f64>,
    improvement_metrics: HashMap<String, Vec<f64>>,
}

/// Adaptation engine for improving recommendations over time
#[derive(Debug)]
pub struct AdaptationEngine {
    pattern_weights: HashMap<String, f64>,
    confidence_adjustments: HashMap<String, f64>,
    learning_rate: f64,
}

impl LearningSystem {
    pub fn new(config: LearningConfig) -> Self {
        Self {
            knowledge_base: KnowledgeBase::new(),
            historical_data: HistoricalDataStore::new(config.max_history_size),
            recommendation_feedback: FeedbackTracker::new(),
            adaptation_engine: AdaptationEngine::new(config.learning_rate),
            config,
        }
    }

    pub fn record_analysis(
        &mut self,
        input: &AnalysisInput,
        recommendations: &[OptimizationRecommendation],
    ) {
        let record = AnalysisRecord {
            timestamp: SystemTime::now(),
            input_characteristics: self.extract_input_characteristics(input),
            recommendations_generated: recommendations.len(),
            complexity_score: self.calculate_complexity_score(input),
        };

        self.historical_data.add_analysis_record(record);

        for recommendation in recommendations {
            let rec_record = RecommendationRecord {
                id: recommendation.id.clone(),
                timestamp: SystemTime::now(),
                optimization_type: recommendation.optimization_type.clone(),
                confidence: recommendation.confidence,
                expected_benefit: recommendation.expected_speedup,
                complexity: recommendation.implementation_complexity,
            };
            self.historical_data.add_recommendation_record(rec_record);
        }
    }

    pub fn record_performance(
        &mut self,
        input: &AnalysisInput,
        actual_performance: &ActualPerformanceResult,
    ) {
        let record = PerformanceRecord {
            timestamp: SystemTime::now(),
            input_hash: self.hash_input(input),
            execution_time: actual_performance.execution_time,
            memory_usage: actual_performance.memory_usage,
            throughput: actual_performance.throughput,
            actual_improvement: 1.0, // Default improvement factor since field not available
        };

        self.historical_data.add_performance_record(record);
    }

    pub fn record_feedback(&mut self, recommendation_id: &str, feedback: RecommendationFeedback) {
        let entry = FeedbackEntry {
            timestamp: SystemTime::now(),
            feedback: feedback.clone(),
            implementation_success: true, // Would be provided by user
            actual_improvement: 0.0,      // Would be measured
        };

        self.recommendation_feedback
            .add_feedback(recommendation_id, entry);
        self.update_adaptation_weights(recommendation_id, &feedback);
    }

    pub fn suggest_optimizations(
        &self,
        input: &AnalysisInput,
    ) -> JitResult<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        // Use historical patterns to suggest optimizations
        for pattern in &self.knowledge_base.optimization_patterns {
            if self.pattern_matches_input(pattern.1, input) {
                suggestions.push(OptimizationSuggestion {
                    pattern_name: pattern.0.clone(),
                    confidence: pattern.1.success_rate * self.get_pattern_weight(&pattern.0),
                    estimated_benefit: pattern.1.average_benefit,
                    description: pattern.1.description.clone(),
                });
            }
        }

        // Sort by confidence and return top suggestions
        suggestions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions.truncate(5);

        Ok(suggestions)
    }

    pub fn predict_performance(
        &self,
        input: &AnalysisInput,
        optimization_type: &OptimizationType,
    ) -> JitResult<PerformancePrediction> {
        if !self.config.enable_performance_prediction {
            return Ok(PerformancePrediction::default());
        }

        let similar_cases = self.find_similar_historical_cases(input);
        if similar_cases.is_empty() {
            return Ok(PerformancePrediction::default());
        }

        let avg_improvement = similar_cases
            .iter()
            .map(|case| case.actual_improvement)
            .sum::<f64>()
            / similar_cases.len() as f64;

        let confidence =
            (similar_cases.len() as f64 / self.config.min_feedback_samples as f64).min(1.0);

        Ok(PerformancePrediction {
            expected_improvement: avg_improvement,
            confidence,
            similar_cases_count: similar_cases.len(),
        })
    }

    pub fn learn_from_outcomes(&mut self) -> JitResult<()> {
        if !self.config.enable_pattern_learning {
            return Ok(());
        }

        // Update pattern success rates based on feedback
        for (pattern_name, pattern) in &mut self.knowledge_base.optimization_patterns {
            if let Some(feedback_entries) = self
                .recommendation_feedback
                .recommendation_feedback
                .get(pattern_name)
            {
                if feedback_entries.len() >= self.config.min_feedback_samples {
                    let success_rate = feedback_entries
                        .iter()
                        .map(|entry| {
                            if entry.implementation_success {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>()
                        / feedback_entries.len() as f64;

                    pattern.success_rate = pattern.success_rate * 0.9 + success_rate * 0.1;
                }
            }
        }

        // Update performance models
        self.update_performance_models()?;

        Ok(())
    }

    pub fn get_knowledge_summary(&self) -> KnowledgeSummary {
        KnowledgeSummary {
            total_patterns: self.knowledge_base.optimization_patterns.len(),
            total_analysis_records: self.historical_data.analysis_history.len(),
            total_feedback_entries: self
                .recommendation_feedback
                .recommendation_feedback
                .values()
                .map(|entries| entries.len())
                .sum(),
            average_success_rate: self.calculate_average_success_rate(),
            most_successful_pattern: self.find_most_successful_pattern(),
        }
    }

    pub fn calculate_confidence(&self) -> f64 {
        if self.historical_data.analysis_history.is_empty() {
            return 0.3; // Low confidence with no data
        }

        let base_confidence = 0.6;
        let history_factor = (self.historical_data.analysis_history.len() as f64 / 100.0).min(0.3);
        let feedback_factor = if self.has_sufficient_feedback() {
            0.1
        } else {
            0.0
        };

        base_confidence + history_factor + feedback_factor
    }

    // Helper methods
    fn extract_input_characteristics(&self, input: &AnalysisInput) -> InputCharacteristics {
        InputCharacteristics {
            graph_size: input
                .computation_graph
                .as_ref()
                .map(|g| g.node_count())
                .unwrap_or(0),
            has_gpu: input.system_constraints.has_gpu,
            cpu_cores: input.system_constraints.cpu_cores,
            memory_gb: input.system_constraints.memory_gb,
            target_platform: input.system_constraints.target_platform.clone(),
        }
    }

    fn calculate_complexity_score(&self, input: &AnalysisInput) -> f64 {
        let mut score: f64 = 0.0;

        if let Some(graph) = &input.computation_graph {
            score += (graph.node_count() as f64).log10() * 0.3;
        }

        score += match input.system_constraints.target_platform {
            TargetPlatform::Desktop => 0.0,
            TargetPlatform::Server => 0.1,
            TargetPlatform::Mobile => 0.3,
            TargetPlatform::Embedded => 0.5,
        };

        score.min(1.0)
    }

    fn hash_input(&self, input: &AnalysisInput) -> u64 {
        // Simplified hash function
        let mut hash = 0u64;

        if let Some(graph) = &input.computation_graph {
            hash ^= graph.node_count() as u64;
        }

        hash ^= input.system_constraints.cpu_cores as u64;
        hash ^= if input.system_constraints.has_gpu {
            1
        } else {
            0
        };

        hash
    }

    fn pattern_matches_input(&self, pattern: &OptimizationPattern, input: &AnalysisInput) -> bool {
        // Check if input characteristics match pattern applicability
        if let Some(graph) = &input.computation_graph {
            if pattern.min_graph_size > 0 && graph.node_count() < pattern.min_graph_size {
                return false;
            }
        }

        if pattern.requires_gpu && !input.system_constraints.has_gpu {
            return false;
        }

        true
    }

    fn get_pattern_weight(&self, pattern_name: &str) -> f64 {
        self.adaptation_engine
            .pattern_weights
            .get(pattern_name)
            .cloned()
            .unwrap_or(1.0)
    }

    fn find_similar_historical_cases(&self, _input: &AnalysisInput) -> Vec<&PerformanceRecord> {
        // Simplified implementation - would use more sophisticated similarity matching
        self.historical_data
            .performance_history
            .iter()
            .take(5)
            .collect()
    }

    fn update_adaptation_weights(
        &mut self,
        recommendation_id: &str,
        feedback: &RecommendationFeedback,
    ) {
        let adjustment = match feedback {
            RecommendationFeedback::Excellent => 0.1,
            RecommendationFeedback::Good => 0.05,
            RecommendationFeedback::Fair => 0.0,
            RecommendationFeedback::Poor => -0.05,
            RecommendationFeedback::Failed => -0.1,
        };

        *self
            .adaptation_engine
            .confidence_adjustments
            .entry(recommendation_id.to_string())
            .or_insert(0.0) += adjustment;
    }

    fn update_performance_models(&mut self) -> JitResult<()> {
        // Update performance models based on historical data
        let model_names: Vec<String> = self
            .knowledge_base
            .performance_models
            .keys()
            .cloned()
            .collect();

        for model_name in model_names {
            // Clone the recent data to avoid borrowing conflicts
            let recent_data = self
                .historical_data
                .performance_history
                .iter()
                .take(100)
                .cloned()
                .collect::<Vec<_>>();
            if !recent_data.is_empty() {
                if let Some(model) = self.knowledge_base.performance_models.get_mut(&model_name) {
                    // Convert to references for the update method
                    let recent_data_refs = recent_data.iter().collect::<Vec<&PerformanceRecord>>();
                    model.update_with_data(&recent_data_refs)?;
                }
            }
        }
        Ok(())
    }

    fn get_recent_performance_data(&self, _model_name: &str) -> Option<Vec<&PerformanceRecord>> {
        Some(
            self.historical_data
                .performance_history
                .iter()
                .take(100)
                .collect(),
        )
    }

    fn calculate_average_success_rate(&self) -> f64 {
        if self.knowledge_base.optimization_patterns.is_empty() {
            return 0.0;
        }

        let total_rate = self
            .knowledge_base
            .optimization_patterns
            .values()
            .map(|pattern| pattern.success_rate)
            .sum::<f64>();

        total_rate / self.knowledge_base.optimization_patterns.len() as f64
    }

    fn find_most_successful_pattern(&self) -> Option<String> {
        self.knowledge_base
            .optimization_patterns
            .iter()
            .max_by(|a, b| {
                a.1.success_rate
                    .partial_cmp(&b.1.success_rate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, _)| name.clone())
    }

    fn has_sufficient_feedback(&self) -> bool {
        self.recommendation_feedback
            .recommendation_feedback
            .values()
            .map(|entries| entries.len())
            .sum::<usize>()
            >= self.config.min_feedback_samples
    }
}

impl KnowledgeBase {
    pub fn new() -> Self {
        Self {
            optimization_patterns: HashMap::new(),
            performance_models: HashMap::new(),
            best_practices: Vec::new(),
            failure_cases: Vec::new(),
        }
    }

    pub fn add_pattern(&mut self, name: String, pattern: OptimizationPattern) {
        self.optimization_patterns.insert(name, pattern);
    }

    pub fn add_performance_model(&mut self, name: String, model: PerformanceModel) {
        self.performance_models.insert(name, model);
    }
}

impl HistoricalDataStore {
    pub fn new(max_size: usize) -> Self {
        Self {
            analysis_history: VecDeque::new(),
            performance_history: VecDeque::new(),
            recommendation_history: VecDeque::new(),
            max_size,
        }
    }

    pub fn add_analysis_record(&mut self, record: AnalysisRecord) {
        self.analysis_history.push_back(record);
        if self.analysis_history.len() > self.max_size {
            self.analysis_history.pop_front();
        }
    }

    pub fn add_performance_record(&mut self, record: PerformanceRecord) {
        self.performance_history.push_back(record);
        if self.performance_history.len() > self.max_size {
            self.performance_history.pop_front();
        }
    }

    pub fn add_recommendation_record(&mut self, record: RecommendationRecord) {
        self.recommendation_history.push_back(record);
        if self.recommendation_history.len() > self.max_size {
            self.recommendation_history.pop_front();
        }
    }
}

impl FeedbackTracker {
    pub fn new() -> Self {
        Self {
            recommendation_feedback: HashMap::new(),
            success_rates: HashMap::new(),
            improvement_metrics: HashMap::new(),
        }
    }

    pub fn add_feedback(&mut self, recommendation_id: &str, feedback: FeedbackEntry) {
        self.recommendation_feedback
            .entry(recommendation_id.to_string())
            .or_insert_with(Vec::new)
            .push(feedback);
    }
}

impl AdaptationEngine {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            pattern_weights: HashMap::new(),
            confidence_adjustments: HashMap::new(),
            learning_rate,
        }
    }
}

// Supporting data structures
#[derive(Debug, Clone)]
pub struct OptimizationPattern {
    pub description: String,
    pub success_rate: f64,
    pub average_benefit: f64,
    pub min_graph_size: usize,
    pub requires_gpu: bool,
    pub applicable_platforms: Vec<TargetPlatform>,
}

#[derive(Debug, Clone)]
pub struct PerformanceModel {
    pub name: String,
    pub accuracy: f64,
    pub last_updated: SystemTime,
}

impl PerformanceModel {
    pub fn update_with_data(&mut self, _data: &[&PerformanceRecord]) -> JitResult<()> {
        self.last_updated = SystemTime::now();
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BestPractice {
    pub title: String,
    pub description: String,
    pub category: String,
    pub effectiveness: f64,
}

#[derive(Debug, Clone)]
pub struct FailureCase {
    pub description: String,
    pub root_cause: String,
    pub prevention_strategy: String,
}

#[derive(Debug, Clone)]
pub struct AnalysisRecord {
    pub timestamp: SystemTime,
    pub input_characteristics: InputCharacteristics,
    pub recommendations_generated: usize,
    pub complexity_score: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    pub timestamp: SystemTime,
    pub input_hash: u64,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub throughput: f64,
    pub actual_improvement: f64,
}

#[derive(Debug, Clone)]
pub struct RecommendationRecord {
    pub id: String,
    pub timestamp: SystemTime,
    pub optimization_type: OptimizationType,
    pub confidence: f64,
    pub expected_benefit: f64,
    pub complexity: f64,
}

#[derive(Debug, Clone)]
pub struct FeedbackEntry {
    pub timestamp: SystemTime,
    pub feedback: RecommendationFeedback,
    pub implementation_success: bool,
    pub actual_improvement: f64,
}

#[derive(Debug, Clone)]
pub struct InputCharacteristics {
    pub graph_size: usize,
    pub has_gpu: bool,
    pub cpu_cores: usize,
    pub memory_gb: usize,
    pub target_platform: TargetPlatform,
}

#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    pub pattern_name: String,
    pub confidence: f64,
    pub estimated_benefit: f64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    pub expected_improvement: f64,
    pub confidence: f64,
    pub similar_cases_count: usize,
}

impl Default for PerformancePrediction {
    fn default() -> Self {
        Self {
            expected_improvement: 0.0,
            confidence: 0.0,
            similar_cases_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KnowledgeSummary {
    pub total_patterns: usize,
    pub total_analysis_records: usize,
    pub total_feedback_entries: usize,
    pub average_success_rate: f64,
    pub most_successful_pattern: Option<String>,
}

#[derive(Debug, Clone)]
pub enum RecommendationFeedback {
    Excellent,
    Good,
    Fair,
    Poor,
    Failed,
}

#[derive(Debug, Clone)]
pub struct ActualPerformanceResult {
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub throughput: f64,
}
