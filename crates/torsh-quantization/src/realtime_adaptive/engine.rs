//! Main adaptive quantization engine
//!
//! This module orchestrates all components of the real-time adaptive quantization system,
//! providing the main entry point for adaptive quantization operations.

use crate::TorshResult;
use std::time::{Duration, Instant};
use torsh_tensor::Tensor;

use super::{
    config::{AdaptiveQuantConfig, FeatureExtractor, QuantizationParameters},
    ml_predictor::{MLParameterPredictor, TrainingExample, TrainingResults},
    optimization::{MultiObjectiveOptimizer, OptimizationStatistics},
    pattern_analysis::{PatternStatistics, WorkloadPattern, WorkloadPatternAnalyzer},
    quality_assessment::{QualityAssessor, QualityMetrics},
    results::{
        AdaptationInfo, AdaptiveQuantizationResult, OptimizationRecommendation, QuantizationResult,
        RecommendationPriority, RuntimeStatistics,
    },
};

/// Real-time adaptive quantization engine with ML optimization
#[derive(Debug, Clone)]
pub struct AdaptiveQuantizationEngine {
    /// ML predictor for quantization parameters
    ml_predictor: MLParameterPredictor,
    /// Quality assessment system
    quality_assessor: QualityAssessor,
    /// Workload pattern analyzer
    pattern_analyzer: WorkloadPatternAnalyzer,
    /// Multi-objective optimizer
    optimizer: MultiObjectiveOptimizer,
    /// Feature extractor
    feature_extractor: FeatureExtractor,
    /// Configuration
    config: AdaptiveQuantConfig,
    /// Runtime statistics
    runtime_stats: RuntimeStatistics,
}

impl AdaptiveQuantizationEngine {
    /// Create new adaptive quantization engine
    pub fn new(config: AdaptiveQuantConfig) -> Self {
        let ml_predictor = MLParameterPredictor::new();
        let quality_assessor = QualityAssessor::new();
        let pattern_analyzer = WorkloadPatternAnalyzer::new();
        let optimizer = MultiObjectiveOptimizer::new();
        let feature_extractor = FeatureExtractor::new();

        Self {
            ml_predictor,
            quality_assessor,
            pattern_analyzer,
            optimizer,
            feature_extractor,
            config,
            runtime_stats: RuntimeStatistics::default(),
        }
    }

    /// Perform adaptive quantization
    pub fn adaptive_quantize(
        &mut self,
        tensor: &Tensor,
    ) -> TorshResult<AdaptiveQuantizationResult> {
        let start_time = Instant::now();

        // Extract features from input tensor
        let features = self.feature_extractor.extract_features(tensor)?;

        // Predict optimal quantization parameters using ML
        let predicted_params = if self.config.enable_ml_prediction {
            self.ml_predictor.predict_parameters(&features)?
        } else {
            QuantizationParameters::default()
        };

        // Analyze workload pattern
        let current_pattern = if self.config.enable_pattern_recognition {
            self.pattern_analyzer.analyze_pattern(&features)?
        } else {
            None
        };

        // Perform multi-objective optimization
        let optimized_params = self.optimizer.optimize_parameters(
            &predicted_params,
            &current_pattern,
            &self.config,
        )?;

        // Apply quantization with optimized parameters
        let quantized_result = self.apply_quantization(tensor, &optimized_params)?;

        // Assess quality if enabled
        if self.config.enable_quality_assessment {
            let quality = self.quality_assessor.assess_quality(
                tensor,
                &quantized_result.quantized_tensor,
                &optimized_params,
            )?;

            // Adapt parameters if quality is below threshold
            if quality.perceptual_score < (1.0 - self.config.quality_tolerance) {
                let adapted_params = self.adapt_parameters(&optimized_params, &quality)?;
                let adapted_result = self.apply_quantization(tensor, &adapted_params)?;

                // Update runtime statistics before returning
                self.update_runtime_stats(&adapted_params, start_time.elapsed());

                return Ok(AdaptiveQuantizationResult {
                    quantized_tensor: adapted_result.quantized_tensor,
                    parameters: adapted_params.clone(),
                    quality_metrics: quality,
                    pattern_info: current_pattern,
                    adaptation_info: Some(AdaptationInfo {
                        original_params: optimized_params,
                        adapted_params,
                        quality_improvement: 0.0, // Would calculate actual improvement
                        adaptation_time: start_time.elapsed(),
                    }),
                    runtime_stats: self.runtime_stats.clone(),
                });
            }
        }

        // Update runtime statistics
        self.update_runtime_stats(&optimized_params, start_time.elapsed());

        Ok(AdaptiveQuantizationResult {
            quantized_tensor: quantized_result.quantized_tensor,
            parameters: optimized_params,
            quality_metrics: QualityMetrics::default(),
            pattern_info: current_pattern,
            adaptation_info: None,
            runtime_stats: self.runtime_stats.clone(),
        })
    }

    /// Apply quantization with given parameters
    fn apply_quantization(
        &self,
        tensor: &Tensor,
        params: &QuantizationParameters,
    ) -> TorshResult<QuantizationResult> {
        let data = tensor.data()?;
        let mut quantized_data = Vec::new();

        for &value in data.iter() {
            let quantized = ((value / params.scale) + params.zero_point as f32)
                .round()
                .clamp(
                    -(1 << (params.bit_width - 1)) as f32,
                    ((1 << (params.bit_width - 1)) - 1) as f32,
                );
            quantized_data.push(quantized);
        }

        let quantized_tensor = Tensor::from_data(
            quantized_data,
            tensor.shape().dims().to_vec(),
            torsh_core::DeviceType::Cpu,
        )?;

        Ok(QuantizationResult {
            quantized_tensor,
            scale: params.scale,
            zero_point: params.zero_point,
        })
    }

    /// Adapt parameters based on quality feedback
    fn adapt_parameters(
        &mut self,
        params: &QuantizationParameters,
        quality: &QualityMetrics,
    ) -> TorshResult<QuantizationParameters> {
        let mut adapted = params.clone();

        // Adapt scale based on SNR
        if quality.snr < 20.0 {
            adapted.scale *= 1.0 - self.config.max_adaptation_rate;
        }

        // Adapt bit width based on perceptual quality
        if quality.perceptual_score < 0.8 && adapted.bit_width < 16 {
            adapted.bit_width += 1;
        }

        // Record adaptation for learning
        self.record_adaptation(params, &adapted, quality);

        Ok(adapted)
    }

    /// Record adaptation for continual learning
    fn record_adaptation(
        &mut self,
        _original: &QuantizationParameters,
        _adapted: &QuantizationParameters,
        _quality: &QualityMetrics,
    ) {
        self.runtime_stats.adaptation_events += 1;
    }

    /// Update runtime statistics
    fn update_runtime_stats(&mut self, _params: &QuantizationParameters, _duration: Duration) {
        self.runtime_stats.total_operations += 1;
    }

    /// Get current runtime statistics
    pub fn get_runtime_stats(&self) -> &RuntimeStatistics {
        &self.runtime_stats
    }

    /// Train the ML predictor with new examples
    pub fn train_predictor(
        &mut self,
        examples: &[TrainingExample],
    ) -> TorshResult<TrainingResults> {
        self.ml_predictor.train(examples)
    }

    /// Update pattern analyzer with new patterns
    pub fn update_patterns(&mut self, patterns: Vec<WorkloadPattern>) {
        for pattern in patterns {
            self.pattern_analyzer.add_pattern(pattern);
        }
    }

    /// Get optimization recommendations
    pub fn get_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze recent performance
        if self.runtime_stats.avg_quality < 0.9 {
            recommendations.push(OptimizationRecommendation {
                category: "Quality".to_string(),
                suggestion: "Consider increasing bit width or adjusting scale factors".to_string(),
                priority: RecommendationPriority::High,
                expected_improvement: 0.1,
            });
        }

        // Check adaptation frequency
        if self.runtime_stats.adaptation_events > self.runtime_stats.total_operations / 10 {
            recommendations.push(OptimizationRecommendation {
                category: "Stability".to_string(),
                suggestion:
                    "High adaptation frequency detected; consider more conservative parameters"
                        .to_string(),
                priority: RecommendationPriority::Medium,
                expected_improvement: 0.05,
            });
        }

        // Provide general optimization recommendations for new engines
        if self.runtime_stats.total_operations == 0 {
            recommendations.push(OptimizationRecommendation {
                category: "Initial Setup".to_string(),
                suggestion:
                    "Run calibration with representative data to establish baseline performance"
                        .to_string(),
                priority: RecommendationPriority::Medium,
                expected_improvement: 0.15,
            });
        }

        recommendations
    }

    /// Get pattern statistics
    pub fn get_pattern_statistics(&self) -> PatternStatistics {
        self.pattern_analyzer.get_pattern_statistics()
    }

    /// Get optimization statistics
    pub fn get_optimization_statistics(&self) -> OptimizationStatistics {
        self.optimizer.get_optimization_statistics()
    }

    /// Get current configuration
    pub fn get_config(&self) -> &AdaptiveQuantConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: AdaptiveQuantConfig) {
        self.config = config;
    }

    /// Reset engine state
    pub fn reset(&mut self) {
        self.runtime_stats = RuntimeStatistics::default();
        self.quality_assessor.clear_history();
        self.optimizer.clear_history();
    }
}

impl Default for AdaptiveQuantizationEngine {
    fn default() -> Self {
        Self::new(AdaptiveQuantConfig::default())
    }
}
