//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{OutlierType, SemanticMetricsAnalyzer, SemanticMetricsConfig, SemanticMetricsError, SemanticMetricsResult};

/// Convenience function for simple metrics analysis
pub fn analyze_semantic_metrics(
    similarity_scores: &[f64],
    quality_scores: &[f64],
    confidence_scores: &[f64],
) -> Result<SemanticMetricsResult, SemanticMetricsError> {
    let mut analyzer = SemanticMetricsAnalyzer::default()?;
    analyzer.analyze_metrics(similarity_scores, quality_scores, confidence_scores)
}
/// Convenience function for metrics analysis with custom config
pub fn analyze_semantic_metrics_with_config(
    similarity_scores: &[f64],
    quality_scores: &[f64],
    confidence_scores: &[f64],
    config: SemanticMetricsConfig,
) -> Result<SemanticMetricsResult, SemanticMetricsError> {
    let mut analyzer = SemanticMetricsAnalyzer::new(config)?;
    analyzer.analyze_metrics(similarity_scores, quality_scores, confidence_scores)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_metrics_analyzer_creation() {
        let analyzer = SemanticMetricsAnalyzer::default();
        assert!(analyzer.is_ok());
    }
    #[test]
    fn test_basic_metrics_analysis() {
        let mut analyzer = SemanticMetricsAnalyzer::default().expect("Semantic Metrics Analyzer should succeed");
        let similarity_scores = vec![
            0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.8, 0.6, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9,
            0.6, 0.7, 0.8, 0.9, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.9, 0.7, 0.8, 0.6, 0.9,
        ];
        let quality_scores = vec![0.9; 30];
        let confidence_scores = vec![0.85; 30];
        let result = analyzer
            .analyze_metrics(&similarity_scores, &quality_scores, &confidence_scores);
        assert!(result.is_ok());
        let result = result.expect("operation should succeed");
        assert_eq!(result.summary.sample_count, 30);
        assert!(result.summary.similarity_stats.mean > 0.0);
        assert!(result.summary.similarity_stats.std_dev >= 0.0);
    }
    #[test]
    fn test_insufficient_data_error() {
        let mut analyzer = SemanticMetricsAnalyzer::default().expect("Semantic Metrics Analyzer should succeed");
        let similarity_scores = vec![0.5];
        let quality_scores = vec![0.8];
        let confidence_scores = vec![0.7];
        let result = analyzer
            .analyze_metrics(&similarity_scores, &quality_scores, &confidence_scores);
        assert!(result.is_err());
        match result.unwrap_err() {
            SemanticMetricsError::InsufficientData { .. } => {}
            _ => panic!("Expected InsufficientData error"),
        }
    }
    #[test]
    fn test_mismatched_array_lengths() {
        let mut analyzer = SemanticMetricsAnalyzer::default().expect("Semantic Metrics Analyzer should succeed");
        let similarity_scores = vec![0.5, 0.6, 0.7];
        let quality_scores = vec![0.8, 0.9];
        let confidence_scores = vec![0.7, 0.8, 0.9];
        let result = analyzer
            .analyze_metrics(&similarity_scores, &quality_scores, &confidence_scores);
        assert!(result.is_err());
        match result.unwrap_err() {
            SemanticMetricsError::DataValidationFailed { .. } => {}
            _ => panic!("Expected DataValidationFailed error"),
        }
    }
    #[test]
    fn test_configuration_builder() {
        let config = SemanticMetricsConfig::builder()
            .enable_statistical_analysis(true)
            .enable_distribution_analysis(false)
            .confidence_level(0.99)
            .max_clusters(5)
            .outlier_threshold(3.0)
            .min_sample_size(50)
            .enable_quality_assessment(true)
            .build();
        assert!(config.is_ok());
        let config = config.expect("operation should succeed");
        assert_eq!(config.confidence_level, 0.99);
        assert_eq!(config.max_clusters, 5);
        assert_eq!(config.outlier_threshold, 3.0);
        assert_eq!(config.min_sample_size, 50);
        assert!(config.enable_statistical_analysis);
        assert!(! config.enable_distribution_analysis);
    }
    #[test]
    fn test_invalid_confidence_level() {
        let config = SemanticMetricsConfig::builder().confidence_level(1.5).build();
        assert!(config.is_err());
        match config.unwrap_err() {
            SemanticMetricsError::InvalidConfiguration { .. } => {}
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }
    #[test]
    fn test_basic_statistics_calculation() {
        let analyzer = SemanticMetricsAnalyzer::default().expect("Semantic Metrics Analyzer should succeed");
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = analyzer.calculate_basic_statistics(&data).expect("basic statistics calculation should succeed");
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.range, 4.0);
    }
    #[test]
    fn test_outlier_detection() {
        let analyzer = SemanticMetricsAnalyzer::default().expect("Semantic Metrics Analyzer should succeed");
        let data = vec![0.5, 0.6, 0.7, 0.6, 0.5, 0.6, 0.7, 0.9, 0.1];
        let outliers = analyzer.detect_outliers(&data).expect("outlier detection should succeed");
        assert!(! outliers.is_empty());
        assert!(outliers.iter().any(| o | o.outlier_type == OutlierType::High));
        assert!(outliers.iter().any(| o | o.outlier_type == OutlierType::Low));
    }
    #[test]
    fn test_correlation_calculation() {
        let analyzer = SemanticMetricsAnalyzer::default().expect("Semantic Metrics Analyzer should succeed");
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let correlation = analyzer
            .calculate_pearson_correlation(&data1, &data2)
            .expect("operation should succeed");
        assert!((correlation - 1.0).abs() < 0.001);
    }
    #[test]
    fn test_clustering() {
        let analyzer = SemanticMetricsAnalyzer::default().expect("Semantic Metrics Analyzer should succeed");
        let data = vec![0.1, 0.15, 0.2, 0.8, 0.85, 0.9];
        let clusters = analyzer.perform_similarity_clustering(&data).expect("similarity clustering should succeed");
        assert!(! clusters.is_empty());
        assert!(clusters.len() <= analyzer.config.max_clusters);
    }
    #[test]
    fn test_quality_assessment() {
        let analyzer = SemanticMetricsAnalyzer::default().expect("Semantic Metrics Analyzer should succeed");
        let similarity_scores = vec![
            0.5, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7,
            0.8, 0.5, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9, 0.6,
        ];
        let quality_scores = vec![0.8; 30];
        let confidence_scores = vec![0.85; 30];
        let quality_metrics = analyzer
            .assess_quality_metrics(
                &similarity_scores,
                &quality_scores,
                &confidence_scores,
            );
        assert!(quality_metrics.is_ok());
        let metrics = quality_metrics.expect("operation should succeed");
        assert!(metrics.data_quality.overall_quality_score > 0.0);
        assert!(metrics.reliability_metrics.confidence_in_results > 0.0);
    }
    #[test]
    fn test_convenience_functions() {
        let similarity_scores = vec![
            0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.6,
            0.9, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8,
        ];
        let quality_scores = vec![0.85; 30];
        let confidence_scores = vec![0.9; 30];
        let result = analyze_semantic_metrics(
            &similarity_scores,
            &quality_scores,
            &confidence_scores,
        );
        assert!(result.is_ok());
        let config = SemanticMetricsConfig::builder()
            .enable_statistical_analysis(false)
            .enable_distribution_analysis(false)
            .build()
            .expect("operation should succeed");
        let result = analyze_semantic_metrics_with_config(
            &similarity_scores,
            &quality_scores,
            &confidence_scores,
            config,
        );
        assert!(result.is_ok());
    }
    #[test]
    fn test_historical_data_management() {
        let mut analyzer = SemanticMetricsAnalyzer::default().expect("Semantic Metrics Analyzer should succeed");
        let data = vec![0.5, 0.6, 0.7];
        analyzer.store_historical_data(&data);
        assert_eq!(analyzer.historical_data.len(), 1);
        analyzer.clear_historical_data();
        assert_eq!(analyzer.historical_data.len(), 0);
    }
    #[test]
    fn test_rank_correlation() {
        let analyzer = SemanticMetricsAnalyzer::default().expect("Semantic Metrics Analyzer should succeed");
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let correlation = analyzer.calculate_rank_correlation(&data1, &data2).expect("rank correlation calculation should succeed");
        assert!((correlation - 1.0).abs() < 0.001);
    }
    #[test]
    fn test_empty_data_statistics() {
        let analyzer = SemanticMetricsAnalyzer::default().expect("Semantic Metrics Analyzer should succeed");
        let empty_data: Vec<f64> = vec![];
        let result = analyzer.calculate_basic_statistics(&empty_data);
        assert!(result.is_err());
    }
    #[test]
    fn test_configuration_validation() {
        let config = SemanticMetricsConfig::builder()
            .confidence_level(0.95)
            .max_clusters(5)
            .outlier_threshold(2.0)
            .min_sample_size(10)
            .build();
        assert!(config.is_ok());
        let config = SemanticMetricsConfig::builder().confidence_level(0.0).build();
        assert!(config.is_err());
        let config = SemanticMetricsConfig::builder().max_clusters(0).build();
        assert!(config.is_err());
    }
    #[test]
    fn test_insights_generation() {
        let mut analyzer = SemanticMetricsAnalyzer::default().expect("Semantic Metrics Analyzer should succeed");
        let high_similarity = vec![0.9; 30];
        let high_quality = vec![0.9; 30];
        let high_confidence = vec![0.9; 30];
        let result = analyzer
            .analyze_metrics(&high_similarity, &high_quality, &high_confidence)
            .expect("operation should succeed");
        assert!(! result.summary.insights.is_empty());
        assert!(! result.summary.recommendations.is_empty());
    }
}
