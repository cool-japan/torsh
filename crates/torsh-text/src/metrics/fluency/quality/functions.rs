//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{QualityAnalyzer, QualityGrade, QualityMonitoringConfig, QualityStandards, QualityThresholds, QualityWeights};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::fluency::language_model::LanguageModelAnalyzer;
    use crate::metrics::fluency::lexical::LexicalAnalyzer;
    use crate::metrics::fluency::pragmatic::PragmaticAnalyzer;
    use crate::metrics::fluency::prosodic::ProsodicAnalyzer;
    use crate::metrics::fluency::semantic::SemanticAnalyzer;
    use crate::metrics::fluency::syntactic::SyntacticAnalyzer;
    #[test]
    fn test_quality_analyzer_creation() {
        let analyzer = QualityAnalyzer::new();
        assert_eq!(analyzer.weights.language_model_weight, 0.20);
        assert_eq!(analyzer.thresholds.excellent_threshold, 0.90);
        assert!(analyzer.standards.academic_standards.contains_key("clarity"));
    }
    #[test]
    fn test_quality_grade_determination() {
        let analyzer = QualityAnalyzer::new();
        assert!(
            matches!(analyzer.determine_quality_grade(0.95), QualityGrade::Excellent)
        );
        assert!(matches!(analyzer.determine_quality_grade(0.85), QualityGrade::Good));
        assert!(matches!(analyzer.determine_quality_grade(0.75), QualityGrade::Fair));
        assert!(matches!(analyzer.determine_quality_grade(0.65), QualityGrade::Poor));
        assert!(
            matches!(analyzer.determine_quality_grade(0.45), QualityGrade::Critical)
        );
    }
    #[test]
    fn test_dimensional_quality_scores() {
        let analyzer = QualityAnalyzer::new();
        let lm_analyzer = LanguageModelAnalyzer::new();
        let syn_analyzer = SyntacticAnalyzer::new();
        let lex_analyzer = LexicalAnalyzer::new();
        let sem_analyzer = SemanticAnalyzer::new();
        let pros_analyzer = ProsodicAnalyzer::new();
        let prag_analyzer = PragmaticAnalyzer::new();
        let test_text = "This is a well-written test sentence with good structure and flow.";
        let lm_score = lm_analyzer.analyze_language_model_fluency(test_text).ok();
        let syn_score = syn_analyzer.analyze_syntactic_fluency(test_text).ok();
        let lex_score = lex_analyzer.analyze_lexical_fluency(test_text).ok();
        let sem_score = sem_analyzer.analyze_semantic_fluency(test_text).ok();
        let pros_score = pros_analyzer.analyze_prosodic_fluency(test_text).ok();
        let prag_score = prag_analyzer.analyze_pragmatic_fluency(test_text).ok();
        let result = analyzer
            .calculate_dimensional_scores(
                &lm_score,
                &syn_score,
                &lex_score,
                &sem_score,
                &pros_score,
                &prag_score,
            );
        assert!(result.is_ok());
        let dimensional_scores = result.unwrap();
        assert!(dimensional_scores.integration_score >= 0.0);
        assert!(dimensional_scores.integration_score <= 1.0);
    }
    #[test]
    fn test_statistical_quality_score() {
        let analyzer = QualityAnalyzer::new();
        let consistent_scores = vec![0.8, 0.82, 0.78, 0.81, 0.79, 0.83];
        let result = analyzer.calculate_statistical_quality_score(&consistent_scores);
        assert!(result.is_ok());
        let score = result.unwrap();
        assert!(score > 0.7);
        let inconsistent_scores = vec![0.9, 0.3, 0.7, 0.2, 0.8, 0.1];
        let result2 = analyzer.calculate_statistical_quality_score(&inconsistent_scores);
        assert!(result2.is_ok());
        let score2 = result2.unwrap();
        assert!(score2 < score);
    }
    #[test]
    fn test_integration_score() {
        let analyzer = QualityAnalyzer::new();
        let well_integrated_scores = vec![0.8, 0.82, 0.78, 0.81];
        let result = analyzer.calculate_integration_score(&well_integrated_scores);
        assert!(result.is_ok());
        let score = result.unwrap();
        assert!(score > 0.5);
        let poorly_integrated_scores = vec![0.9, 0.2, 0.8, 0.1];
        let result2 = analyzer.calculate_integration_score(&poorly_integrated_scores);
        assert!(result2.is_ok());
        let score2 = result2.unwrap();
        assert!(score2 < score);
    }
    #[test]
    fn test_reading_ease_calculation() {
        let analyzer = QualityAnalyzer::new();
        let simple_text = "This is easy to read. Short sentences work well.";
        let result = analyzer.calculate_reading_ease(simple_text);
        assert!(result.is_ok());
        let ease_score = result.unwrap();
        assert!(ease_score > 0.0);
        assert!(ease_score <= 1.0);
        let complex_text = "Extraordinarily complicated sentences with multisyllabic terminology and convoluted syntactic structures significantly decrease comprehensibility and readability.";
        let result2 = analyzer.calculate_reading_ease(complex_text);
        assert!(result2.is_ok());
        let ease_score2 = result2.unwrap();
        assert!(ease_score2 < ease_score);
    }
    #[test]
    fn test_syllable_count_estimation() {
        let analyzer = QualityAnalyzer::new();
        assert_eq!(analyzer.estimate_syllable_count("cat"), 1);
        assert_eq!(analyzer.estimate_syllable_count("hello"), 2);
        assert_eq!(analyzer.estimate_syllable_count("beautiful"), 3);
        assert_eq!(analyzer.estimate_syllable_count("extraordinary"), 5);
    }
    #[test]
    fn test_sentence_variety_calculation() {
        let analyzer = QualityAnalyzer::new();
        let varied_sentences = vec![
            "Short.", "This is a medium length sentence.",
            "Here we have a much longer sentence with more words and complexity.",
        ];
        let variety_score = analyzer.calculate_sentence_variety(&varied_sentences);
        assert!(variety_score > 0.0);
        let uniform_sentences = vec![
            "Same length here.", "Same length here.", "Same length here.",
        ];
        let uniform_score = analyzer.calculate_sentence_variety(&uniform_sentences);
        assert!(uniform_score < variety_score);
    }
    #[test]
    fn test_cognitive_load_calculation() {
        let analyzer = QualityAnalyzer::new();
        let simple_text = "The cat sat on the mat. It was warm and cozy.";
        let result = analyzer.calculate_cognitive_load(simple_text);
        assert!(result.is_ok());
        let simple_load = result.unwrap();
        let complex_text = "The extraordinary feline positioned itself methodically upon the intricate textile surface, experiencing optimal thermal comfort.";
        let result2 = analyzer.calculate_cognitive_load(complex_text);
        assert!(result2.is_ok());
        let complex_load = result2.unwrap();
        assert!(complex_load > simple_load);
    }
    #[test]
    fn test_interest_maintenance_calculation() {
        let analyzer = QualityAnalyzer::new();
        let varied_text = "Short sentence. This is a medium-length sentence with some complexity. Here we have a much longer sentence that provides detailed information and maintains reader interest through variation.";
        let result = analyzer.calculate_interest_maintenance(varied_text);
        assert!(result.is_ok());
        let interest_score = result.unwrap();
        assert!(interest_score > 0.0);
        assert!(interest_score <= 1.0);
    }
    #[test]
    fn test_comprehensive_quality_assessment() {
        let analyzer = QualityAnalyzer::new();
        let test_text = "This is a comprehensive test of the quality assessment system. It includes multiple sentences with varying complexity levels. The text demonstrates good structure, appropriate vocabulary, and clear communication.";
        let result = analyzer
            .analyze_comprehensive_quality(
                test_text,
                None,
                None,
                None,
                None,
                None,
                None,
            );
        assert!(result.is_ok());
        let assessment = result.unwrap();
        assert!(assessment.overall_quality_score >= 0.0);
        assert!(assessment.overall_quality_score <= 1.0);
        assert!(! matches!(assessment.quality_grade, QualityGrade::Critical));
        assert!(! assessment.quality_report.executive_summary.key_findings.is_empty());
    }
    #[test]
    fn test_quality_weights_configuration() {
        let custom_weights = QualityWeights {
            language_model_weight: 0.3,
            syntactic_weight: 0.2,
            lexical_weight: 0.2,
            semantic_weight: 0.1,
            prosodic_weight: 0.1,
            pragmatic_weight: 0.05,
            statistical_weight: 0.05,
        };
        let analyzer = QualityAnalyzer::new().with_weights(custom_weights.clone());
        assert_eq!(analyzer.weights.language_model_weight, 0.3);
        assert_eq!(analyzer.weights.syntactic_weight, 0.2);
    }
    #[test]
    fn test_quality_thresholds_configuration() {
        let custom_thresholds = QualityThresholds {
            excellent_threshold: 0.95,
            good_threshold: 0.85,
            fair_threshold: 0.75,
            poor_threshold: 0.65,
            critical_threshold: 0.55,
        };
        let analyzer = QualityAnalyzer::new().with_thresholds(custom_thresholds.clone());
        assert_eq!(analyzer.thresholds.excellent_threshold, 0.95);
    }
    #[test]
    fn test_empty_text_handling() {
        let analyzer = QualityAnalyzer::new();
        let result = analyzer.calculate_reading_ease("");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.0);
        let result2 = analyzer.calculate_cognitive_load("");
        assert!(result2.is_ok());
        let result3 = analyzer.calculate_interest_maintenance("");
        assert!(result3.is_ok());
    }
    #[test]
    fn test_quality_standards() {
        let standards = QualityStandards::default();
        assert!(standards.academic_standards.contains_key("clarity"));
        assert!(standards.professional_standards.contains_key("effectiveness"));
        assert!(standards.creative_standards.contains_key("originality"));
        assert!(standards.technical_standards.contains_key("accuracy"));
        assert!(standards.conversational_standards.contains_key("naturalness"));
    }
    #[test]
    fn test_monitoring_configuration() {
        let config = QualityMonitoringConfig::default();
        assert!(config.enable_real_time_monitoring);
        assert!(config.alert_thresholds.contains_key("critical_drop"));
        assert_eq!(config.monitoring_frequency, 100);
        assert_eq!(config.quality_degradation_sensitivity, 0.05);
    }
}
