//! Real-time Adaptive Quantization with ML-based Optimization
//!
//! This module implements cutting-edge real-time adaptive quantization that uses machine learning
//! to continuously optimize quantization parameters based on runtime patterns, workload characteristics,
//! and quality requirements.
//!
//! ## Modular Architecture (Phase 83 Refactoring)
//!
//! The original 1,695-line monolithic file has been systematically extracted into:
//! - `config` - Configuration types and defaults (185 lines)
//! - `ml_predictor` - ML parameter prediction and neural networks (280 lines)
//! - `feature_extraction` - Comprehensive feature extraction (175 lines)
//! - `quality_assessment` - Quality assessment and metrics (215 lines)
//! - `pattern_analysis` - Workload pattern recognition (280 lines)
//! - `optimization` - Multi-objective optimization (220 lines)
//! - `engine` - Main adaptive quantization engine (185 lines)
//! - `results` - Result types and report generation (250 lines)
//!
//! ## Features
//!
//! - **ML-based Parameter Prediction**: Neural networks predict optimal quantization parameters
//! - **Real-time Quality Assessment**: Continuous quality monitoring and adaptation
//! - **Workload Pattern Recognition**: Identifies and adapts to different computation patterns
//! - **Multi-objective Optimization**: Balances accuracy, performance, and energy consumption
//! - **Predictive Scaling**: Anticipates quantization needs based on input characteristics
//! - **Dynamic Bit-width Allocation**: Adaptive precision assignment based on layer importance
//!
//! ## Usage
//!
//! ```rust,no_run
//! use torsh_quantization::realtime_adaptive::*;
//! use torsh_tensor::tensor_1d;
//!
//! // Create adaptive quantization engine
//! let mut engine = AdaptiveQuantizationEngine::new(AdaptiveQuantConfig::default());
//!
//! // Create test tensor
//! let tensor = tensor_1d(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();
//!
//! // Perform adaptive quantization
//! let result = engine.adaptive_quantize(&tensor).unwrap();
//!
//! // Generate comprehensive report
//! println!("{}", result.generate_report());
//!
//! // Get optimization recommendations
//! let recommendations = engine.get_optimization_recommendations();
//! for rec in recommendations {
//!     println!("📝 {}: {}", rec.category, rec.suggestion);
//! }
//! ```

// Core modules
pub mod config;
pub mod engine;
pub mod enhanced_ml_predictor;
pub mod feature_extraction;
pub mod ml_predictor;
pub mod optimization;
pub mod pattern_analysis;
pub mod quality_assessment;
pub mod results;

// Re-export all types for backward compatibility
pub use config::*;
pub use engine::*;
pub use enhanced_ml_predictor::*;
pub use ml_predictor::*;
pub use optimization::*;
pub use pattern_analysis::*;
pub use quality_assessment::*;
pub use results::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TorshResult;
    use std::time::Instant;
    use torsh_tensor::tensor_1d;

    #[test]
    fn test_adaptive_config_default() {
        let config = AdaptiveQuantConfig::default();

        assert!(config.enable_ml_prediction);
        assert!(config.enable_quality_assessment);
        assert!(config.enable_pattern_recognition);
        assert_eq!(config.update_frequency, 100);
        assert_eq!(config.quality_tolerance, 0.02);
        assert_eq!(config.performance_weight, 0.3);
        assert_eq!(config.energy_weight, 0.3);
        assert_eq!(config.accuracy_weight, 0.4);
        assert_eq!(config.max_adaptation_rate, 0.1);
    }

    #[test]
    fn test_quantization_parameters_default() {
        let params = QuantizationParameters::default();

        assert_eq!(params.scale, 1.0);
        assert_eq!(params.zero_point, 0);
        assert_eq!(params.bit_width, 8);
        assert_eq!(params.scheme, "symmetric");
    }

    #[test]
    fn test_ml_parameter_predictor() -> TorshResult<()> {
        let predictor = MLParameterPredictor::new();
        let features = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        ];

        let params = predictor.predict_parameters(&features)?;
        assert!(params.scale > 0.0);
        assert!(params.bit_width >= 4 && params.bit_width <= 16);
        assert!(params.zero_point >= -128 && params.zero_point <= 127);

        Ok(())
    }

    #[test]
    fn test_predictor_network() -> TorshResult<()> {
        let mut network = PredictorNetwork::new(4, 2, 0.01);
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let target = vec![0.5, 0.6];

        let prediction = network.predict(&input)?;
        assert_eq!(prediction.len(), 2);

        let loss = network.train_step(&input, &target)?;
        assert!(loss >= 0.0);

        Ok(())
    }

    #[test]
    fn test_feature_extraction() -> TorshResult<()> {
        let extractor = FeatureExtractor::new();
        let tensor = tensor_1d(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();

        let features = extractor.extract_features(&tensor)?;
        assert_eq!(features.len(), 16); // Fixed feature dimension

        // Test quick features
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let quick_features = extractor.extract_quick_features(&data);
        assert_eq!(quick_features.len(), 4);

        Ok(())
    }

    #[test]
    fn test_quality_assessment() -> TorshResult<()> {
        let mut assessor = QualityAssessor::new();
        let original = tensor_1d(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();
        let quantized = tensor_1d(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();
        let params = QuantizationParameters::default();

        let quality = assessor.assess_quality(&original, &quantized, &params)?;
        assert!(quality.perceptual_score > 0.9);
        assert!(quality.ssim > 0.9);

        // Test degradation detection
        let degradation = assessor.detect_degradation();
        assert!(!degradation); // Should be false initially

        Ok(())
    }

    #[test]
    fn test_pattern_analysis() -> TorshResult<()> {
        let mut analyzer = WorkloadPatternAnalyzer::new();
        let features = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        ];

        let pattern = analyzer.analyze_pattern(&features)?;
        assert!(pattern.is_some());

        // Test pattern statistics
        let stats = analyzer.get_pattern_statistics();
        assert!(stats.total_patterns > 0);

        Ok(())
    }

    #[test]
    fn test_multi_objective_optimization() -> TorshResult<()> {
        let mut optimizer = MultiObjectiveOptimizer::new();
        let initial_params = QuantizationParameters::default();
        let config = AdaptiveQuantConfig::default();

        let optimized = optimizer.optimize_parameters(&initial_params, &None, &config)?;
        assert!(optimized.scale > 0.0);
        assert!(optimized.bit_width >= 4 && optimized.bit_width <= 16);

        Ok(())
    }

    #[test]
    fn test_adaptive_quantization_engine() -> TorshResult<()> {
        let mut engine = AdaptiveQuantizationEngine::new(AdaptiveQuantConfig::default());
        let tensor = tensor_1d(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();

        let result = engine.adaptive_quantize(&tensor)?;
        assert!(result.parameters.scale > 0.0);
        assert!(result.quality_metrics.perceptual_score >= 0.0);
        assert_eq!(
            result.quantized_tensor.shape().dims(),
            tensor.shape().dims()
        );

        Ok(())
    }

    #[test]
    fn test_runtime_statistics() {
        let mut stats = RuntimeStatistics::default();

        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.adaptation_events, 0);
        assert_eq!(stats.avg_quality, 1.0);

        // Test quality update
        stats.update_avg_quality(0.95);
        assert_eq!(stats.avg_quality, 0.95);

        // Test performance improvements
        stats.add_performance_improvement(0.1);
        assert_eq!(stats.avg_performance_improvement(), 0.1);

        // Test energy savings
        stats.add_energy_savings(0.2);
        assert_eq!(stats.avg_energy_savings(), 0.2);
    }

    #[test]
    fn test_optimization_recommendations() {
        let engine = AdaptiveQuantizationEngine::new(AdaptiveQuantConfig::default());
        let recommendations = engine.get_optimization_recommendations();

        // Should have some recommendations based on initial state
        assert!(!recommendations.is_empty());

        // Check recommendation structure
        for rec in &recommendations {
            assert!(!rec.category.is_empty());
            assert!(!rec.suggestion.is_empty());
            assert!(rec.expected_improvement >= 0.0);
        }
    }

    #[test]
    fn test_training_example() {
        let example = TrainingExample {
            features: vec![0.1, 0.2, 0.3],
            target: vec![1.0, 0.0, 8.0],
            quality_score: 0.95,
            timestamp: Instant::now(),
        };

        assert_eq!(example.features.len(), 3);
        assert_eq!(example.target.len(), 3);
        assert!(example.quality_score > 0.9);
    }

    #[test]
    fn test_ml_predictor_training() -> TorshResult<()> {
        let mut predictor = MLParameterPredictor::new();
        let examples = vec![
            TrainingExample {
                features: vec![0.1; 16],
                target: vec![1.0, 0.0, 8.0],
                quality_score: 0.9,
                timestamp: Instant::now(),
            },
            TrainingExample {
                features: vec![0.2; 16],
                target: vec![0.5, 10.0, 12.0],
                quality_score: 0.8,
                timestamp: Instant::now(),
            },
        ];

        let results = predictor.train(&examples)?;
        assert_eq!(results.examples_processed, 2);
        assert!(results.average_loss >= 0.0);

        Ok(())
    }

    #[test]
    fn test_quality_statistics() -> TorshResult<()> {
        let mut assessor = QualityAssessor::new();
        let original = tensor_1d(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();
        let quantized = tensor_1d(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();
        let params = QuantizationParameters::default();

        // Add some measurements
        for _ in 0..5 {
            assessor.assess_quality(&original, &quantized, &params)?;
        }

        let stats = assessor.get_quality_statistics();
        assert_eq!(stats.sample_count, 5);
        assert!(stats.avg_perceptual_score > 0.0);
        assert!(stats.avg_ssim > 0.0);

        Ok(())
    }

    #[test]
    fn test_pattern_learning() {
        let mut analyzer = WorkloadPatternAnalyzer::new();
        let features = vec![0.8; 16]; // High compute pattern
        let performance = PerformanceProfile {
            avg_execution_time: 10.0,
            memory_usage: 500.0,
            energy_consumption: 30.0,
            cache_efficiency: 0.6,
        };

        analyzer.learn_pattern("custom_pattern".to_string(), features, performance);

        let pattern = analyzer.get_pattern("custom_pattern");
        assert!(pattern.is_some());
        assert_eq!(pattern.unwrap().name, "custom_pattern");
    }

    #[test]
    fn test_constraint_handling() {
        let constraints = ConstraintHandler::default();

        // Test hardware constraints
        assert!(!constraints
            .hardware_constraints
            .supported_bit_widths
            .is_empty());
        assert!(constraints.hardware_constraints.max_memory_bandwidth > 0.0);

        // Test quality constraints
        assert!(constraints.quality_constraints.min_snr > 0.0);
        assert!(constraints.quality_constraints.max_mse > 0.0);

        // Test performance constraints
        assert!(constraints.performance_constraints.max_latency > 0.0);
        assert!(constraints.performance_constraints.min_throughput > 0.0);
    }

    #[test]
    fn test_report_generation() -> TorshResult<()> {
        let mut engine = AdaptiveQuantizationEngine::new(AdaptiveQuantConfig::default());
        let tensor = tensor_1d(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();
        let result = engine.adaptive_quantize(&tensor)?;

        // Test text report
        let report = result.generate_report();
        assert!(report.contains("Adaptive Quantization Report"));
        assert!(report.contains("Quantization Parameters"));
        assert!(report.contains("Quality Metrics"));

        // Test JSON report
        let json_report = result.generate_json_report();
        assert!(json_report.contains("adaptive_quantization_report"));
        assert!(json_report.contains("parameters"));
        assert!(json_report.contains("quality_metrics"));

        // Test CSV format
        let csv_line = result.generate_csv_line();
        assert!(!csv_line.is_empty());

        let csv_header = AdaptiveQuantizationResult::csv_header();
        assert!(csv_header.contains("scale"));
        assert!(csv_header.contains("quality"));

        Ok(())
    }

    #[test]
    fn test_modular_structure_integrity() {
        // Test that all major components can be created and used together

        // Configuration
        let config = AdaptiveQuantConfig::default();
        assert!(config.enable_ml_prediction);

        // ML predictor
        let predictor = MLParameterPredictor::new();

        // Feature extractor
        let extractor = FeatureExtractor::new();
        assert_eq!(extractor.get_feature_dimension(), 16);

        // Quality assessor
        let assessor = QualityAssessor::new();

        // Pattern analyzer
        let analyzer = WorkloadPatternAnalyzer::new();
        assert!(analyzer.get_all_patterns().len() > 0);

        // Optimizer
        let optimizer = MultiObjectiveOptimizer::new();

        // Main engine
        let engine = AdaptiveQuantizationEngine::new(config);
        assert!(engine.get_runtime_stats().total_operations == 0);

        println!("Phase 83 modular structure integrity verified");
    }

    #[test]
    fn test_comprehensive_adaptive_quantization_workflow() -> TorshResult<()> {
        let mut engine = AdaptiveQuantizationEngine::new(AdaptiveQuantConfig::default());

        // Test different tensor patterns
        let test_cases = vec![
            tensor_1d(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap(),
            tensor_1d(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
            tensor_1d(&[0.01, 0.02, 0.03, 0.04, 0.05]).unwrap(),
        ];

        for (i, tensor) in test_cases.iter().enumerate() {
            let result = engine.adaptive_quantize(tensor)?;

            // Verify result consistency
            assert!(result.parameters.scale > 0.0);
            assert!(result.parameters.bit_width >= 4 && result.parameters.bit_width <= 16);
            assert_eq!(
                result.quantized_tensor.shape().dims(),
                tensor.shape().dims()
            );

            // Verify statistics are being tracked
            let stats = engine.get_runtime_stats();
            assert_eq!(stats.total_operations, i + 1);

            println!(
                "Test case {}: Scale={:.4}, Bit-width={}, Pattern={:?}",
                i + 1,
                result.parameters.scale,
                result.parameters.bit_width,
                result.pattern_info
            );
        }

        // Test training with examples
        let training_examples = vec![TrainingExample {
            features: vec![0.5; 16],
            target: vec![1.0, 0.0, 8.0],
            quality_score: 0.95,
            timestamp: Instant::now(),
        }];

        let training_results = engine.train_predictor(&training_examples)?;
        assert!(training_results.examples_processed > 0);

        // Test recommendations
        let recommendations = engine.get_optimization_recommendations();
        assert!(!recommendations.is_empty());

        for rec in recommendations {
            println!(
                "💡 {}: {} (Priority: {:?})",
                rec.category, rec.suggestion, rec.priority
            );
        }

        Ok(())
    }

    #[test]
    fn test_edge_cases_and_error_handling() -> TorshResult<()> {
        // Test with empty tensor
        let empty_tensor = tensor_1d(&[]).unwrap_or_else(|_| tensor_1d(&[0.0]).unwrap());

        let mut engine = AdaptiveQuantizationEngine::new(AdaptiveQuantConfig::default());
        let _result = engine.adaptive_quantize(&empty_tensor)?;

        // Test feature extraction with different sizes
        let extractor = FeatureExtractor::new();
        let small_tensor = tensor_1d(&[0.1]).unwrap();
        let features = extractor.extract_features(&small_tensor)?;
        assert_eq!(features.len(), 16);

        // Test quality assessment with identical tensors
        let mut assessor = QualityAssessor::new();
        let tensor = tensor_1d(&[0.5; 10]).unwrap();
        let quality =
            assessor.assess_quality(&tensor, &tensor, &QuantizationParameters::default())?;
        assert!(quality.perceptual_score > 0.99); // Should be nearly perfect

        Ok(())
    }
}
