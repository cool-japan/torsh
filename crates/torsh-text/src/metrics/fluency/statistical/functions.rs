//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};

use super::types::{DistributionType, StatisticalAnalyzer, TrendDirection};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_statistical_analyzer_creation() {
        let analyzer = StatisticalAnalyzer::new();
        assert_eq!(analyzer.confidence_level, 0.95);
        assert_eq!(analyzer.bootstrap_samples, 1000);
        assert_eq!(analyzer.outlier_threshold, 2.0);
    }
    #[test]
    fn test_descriptive_statistics() {
        let analyzer = StatisticalAnalyzer::new();
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = analyzer.descriptive_statistics(&data);
        assert!(result.is_ok());
        let stats = result.expect("operation should succeed");
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.range, 4.0);
    }
    #[test]
    fn test_empty_data_handling() {
        let analyzer = StatisticalAnalyzer::new();
        let empty_data = Array1::<f64>::from_vec(vec![]);
        let result = analyzer.descriptive_statistics(&empty_data);
        assert!(result.is_err());
    }
    #[test]
    fn test_quartile_calculation() {
        let analyzer = StatisticalAnalyzer::new();
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = analyzer.calculate_quartiles(&data);
        assert!(result.is_ok());
        let quartiles = result.expect("operation should succeed");
        assert!(quartiles.q1 > 0.0);
        assert!(quartiles.q3 > quartiles.q1);
        assert!(quartiles.iqr > 0.0);
    }
    #[test]
    fn test_correlation_analysis() {
        let analyzer = StatisticalAnalyzer::new();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let result = analyzer.correlation_analysis(&x, &y);
        assert!(result.is_ok());
        let correlation = result.expect("operation should succeed");
        assert!((correlation.pearson_correlation - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_linear_regression() {
        let analyzer = StatisticalAnalyzer::new();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let result = analyzer.linear_regression(&x, &y);
        assert!(result.is_ok());
        let regression = result.expect("operation should succeed");
        assert!((regression.slope - 2.0).abs() < 1e-10);
        assert!(regression.intercept.abs() < 1e-10);
        assert!((regression.r_squared - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_outlier_detection() {
        let analyzer = StatisticalAnalyzer::new();
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        let zscore_outliers = analyzer.detect_zscore_outliers(&data);
        assert!(zscore_outliers.is_ok());
        let iqr_outliers = analyzer.detect_iqr_outliers(&data);
        assert!(iqr_outliers.is_ok());
        let zscore_outliers = zscore_outliers.expect("operation should succeed");
        let iqr_outliers = iqr_outliers.expect("operation should succeed");
        assert!(zscore_outliers.contains(& 5) || iqr_outliers.contains(& 5));
    }
    #[test]
    fn test_bootstrap_confidence_intervals() {
        let analyzer = StatisticalAnalyzer::with_bootstrap_samples(100);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = analyzer.calculate_bootstrap_intervals(&data);
        assert!(result.is_ok());
        let intervals = result.expect("operation should succeed");
        assert!(intervals.contains_key("mean"));
        let (lower, upper) = intervals["mean"];
        assert!(lower <= upper);
        assert!(lower <= 3.0 && 3.0 <= upper);
    }
    #[test]
    fn test_distribution_analysis() {
        let analyzer = StatisticalAnalyzer::new();
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = analyzer.distribution_analysis(&data);
        assert!(result.is_ok());
        let analysis = result.expect("operation should succeed");
        assert_eq!(analysis.distribution_type, DistributionType::Normal);
        assert!(analysis.entropy > 0.0);
    }
    #[test]
    fn test_time_series_analysis() {
        let analyzer = StatisticalAnalyzer::new();
        let data = Array1::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        );
        let result = analyzer.time_series_analysis(&data);
        assert!(result.is_ok());
        let analysis = result.expect("operation should succeed");
        assert!(
            matches!(analysis.trend_analysis.trend_direction, TrendDirection::Increasing)
        );
        assert!(! analysis.seasonality_analysis.has_seasonality);
    }
    #[test]
    fn test_hypothesis_testing() {
        let analyzer = StatisticalAnalyzer::new();
        let data1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let data2 = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = analyzer.hypothesis_testing(&data1, Some(&data2));
        assert!(result.is_ok());
        let tests = result.expect("operation should succeed");
        assert!(tests.t_test_results.one_sample.is_some());
        assert!(tests.t_test_results.two_sample.is_some());
    }
    #[test]
    fn test_information_theory_metrics() {
        let analyzer = StatisticalAnalyzer::new();
        let data1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let data2 = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let result = analyzer.information_theory_metrics(&data1, Some(&data2));
        assert!(result.is_ok());
        let metrics = result.expect("operation should succeed");
        assert!(metrics.entropy > 0.0);
        assert!(metrics.mutual_information >= 0.0);
    }
    #[test]
    fn test_multivariate_analysis() {
        let analyzer = StatisticalAnalyzer::new();
        let data = Array2::<
            f64,
        >::from_shape_vec(
                (5, 3),
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                    14.0, 15.0,
                ],
            )
            .expect("operation should succeed");
        let result = analyzer.multivariate_analysis(&data);
        assert!(result.is_ok());
        let analysis = result.expect("operation should succeed");
        assert_eq!(analysis.principal_component_analysis.optimal_components, 2);
        assert_eq!(analysis.cluster_analysis.optimal_clusters, 3);
    }
    #[test]
    fn test_bayesian_analysis() {
        let analyzer = StatisticalAnalyzer::new();
        let data = Array1::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        );
        let result = analyzer.bayesian_analysis(&data);
        assert!(result.is_ok());
        let analysis = result.expect("operation should succeed");
        assert!(matches!(analysis.posterior_distribution, DistributionType::Normal));
        assert!(analysis.bayes_factor > 0.0);
        assert!(! analysis.credible_intervals.is_empty());
    }
    #[test]
    fn test_rank_calculation() {
        let analyzer = StatisticalAnalyzer::new();
        let data = Array1::from_vec(vec![3.0, 1.0, 4.0, 2.0]);
        let ranks = analyzer.calculate_ranks(&data);
        assert_eq!(ranks[0], 3.0);
        assert_eq!(ranks[1], 1.0);
        assert_eq!(ranks[2], 4.0);
        assert_eq!(ranks[3], 2.0);
    }
    #[test]
    fn test_median_absolute_deviation() {
        let analyzer = StatisticalAnalyzer::new();
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let median = 3.0;
        let result = analyzer.calculate_median_absolute_deviation(&data, median);
        assert!(result.is_ok());
        let mad = result.expect("operation should succeed");
        assert_eq!(mad, 1.0);
    }
}
