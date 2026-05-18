//! Dynamic topic modeling extension module.
//!
//! Contains the `Default` implementation for `TopicModeler`, convenience
//! free-functions for quick topic analysis, and the module test suite.

use super::*;

impl Default for TopicModeler {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for simple topic modeling

/// Extract basic topics from text using keyword clustering
pub fn extract_basic_topics(text: &str) -> Result<TopicModelingResult, TopicModelingError> {
    let mut modeler = TopicModeler::new();
    modeler.extract_topics(text)
}

/// Extract topics using TF-IDF approach
pub fn extract_tfidf_topics(
    text: &str,
    num_topics: usize,
) -> Result<TopicModelingResult, TopicModelingError> {
    let config = TopicModelingConfig::new()
        .with_approach(TopicModelingApproach::TfIdf)
        .with_num_topics(num_topics);
    let mut modeler = TopicModeler::with_config(config);
    modeler.extract_topics(text)
}

/// Compare topic similarity between two texts
pub fn compare_topic_similarity_simple(
    text1: &str,
    text2: &str,
) -> Result<f64, TopicModelingError> {
    let mut modeler = TopicModeler::new();
    let topics1 = modeler.extract_topics(text1)?;
    let topics2 = modeler.extract_topics(text2)?;
    modeler.compute_topic_similarity(&topics1, &topics2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_modeler_creation() {
        let modeler = TopicModeler::new();
        assert!(!modeler.predefined_topics.is_empty());
        assert_eq!(modeler.config.num_topics, 8);
    }

    #[test]
    fn test_keyword_topic_extraction() -> Result<(), TopicModelingError> {
        let mut modeler = TopicModeler::new();
        let text = "Computer software development with artificial intelligence and machine learning algorithms";

        let result = modeler.extract_topics(text)?;

        assert!(!result.topics.is_empty());
        assert!(!result.topic_distribution.is_empty());
        assert!(result.diversity_score >= 0.0 && result.diversity_score <= 1.0);

        // Should find technology-related topics
        let has_tech_topic = result.topics.iter().any(|topic| {
            topic.name.contains("Technology")
                || topic
                    .keywords
                    .iter()
                    .any(|kw| kw.word == "computer" || kw.word == "software")
        });
        assert!(has_tech_topic);

        Ok(())
    }

    #[test]
    fn test_tfidf_topic_extraction() -> Result<(), TopicModelingError> {
        let config = TopicModelingConfig::new().with_approach(TopicModelingApproach::TfIdf);
        let mut modeler = TopicModeler::with_config(config);

        let text = "Business market analysis shows financial growth in technology sector with innovation driving investment";

        let result = modeler.extract_topics(text)?;

        assert!(!result.topics.is_empty());
        assert_eq!(result.metadata.approach_used, TopicModelingApproach::TfIdf);

        Ok(())
    }

    #[test]
    fn test_dynamic_topic_modeling() -> Result<(), TopicModelingError> {
        let config = TopicModelingConfig::new()
            .with_approach(TopicModelingApproach::Dynamic)
            .with_dynamic_modeling(true);
        let mut modeler = TopicModeler::with_config(config);

        let text = "Technology advances rapidly. Innovation drives market changes. Business adapts to new trends. Future looks promising for development.";

        let result = modeler.extract_topics(text)?;

        assert!(!result.topics.is_empty());
        assert!(result.topic_evolution.is_some());

        if let Some(evolution) = &result.topic_evolution {
            assert!(!evolution.segment_distributions.is_empty());
        }

        Ok(())
    }

    #[test]
    fn test_topic_similarity() -> Result<(), TopicModelingError> {
        let mut modeler = TopicModeler::new();

        let text1 = "Computer technology software development programming artificial intelligence";
        let text2 = "Technology innovation software engineering machine learning algorithms";
        let text3 = "Medical healthcare treatment patient doctor diagnosis therapy";

        let result1 = modeler.extract_topics(text1)?;
        let result2 = modeler.extract_topics(text2)?;
        let result3 = modeler.extract_topics(text3)?;

        let similarity_tech = modeler.compute_topic_similarity(&result1, &result2)?;
        let similarity_mixed = modeler.compute_topic_similarity(&result1, &result3)?;

        assert!(similarity_tech > similarity_mixed);
        assert!(similarity_tech > 0.3); // Should have decent similarity

        Ok(())
    }

    #[test]
    fn test_hierarchical_topic_modeling() -> Result<(), TopicModelingError> {
        let config = TopicModelingConfig::new()
            .with_approach(TopicModelingApproach::Hierarchical)
            .with_hierarchical_analysis(true);
        let mut modeler = TopicModeler::with_config(config);

        let text = "Science research study analysis with scientific method and laboratory testing";

        let result = modeler.extract_topics(text)?;

        assert!(!result.topics.is_empty());
        assert!(result.hierarchical_structure.is_some());

        Ok(())
    }

    #[test]
    fn test_multiple_topic_comparison() -> Result<(), TopicModelingError> {
        let mut modeler = TopicModeler::new();

        let texts = vec![
            "Technology computer software",
            "Technology innovation development",
            "Medical healthcare treatment",
            "Business market finance",
        ];

        let similarity_matrix = modeler.compare_multiple_topics(&texts)?;

        assert_eq!(similarity_matrix.len(), 4);
        assert_eq!(similarity_matrix[0].len(), 4);

        // Diagonal should be 1.0
        for i in 0..4 {
            assert_eq!(similarity_matrix[i][i], 1.0);
        }

        // Technology texts should be more similar to each other
        assert!(similarity_matrix[0][1] > similarity_matrix[0][2]);

        Ok(())
    }

    #[test]
    fn test_topic_quality_metrics() -> Result<(), TopicModelingError> {
        let mut modeler = TopicModeler::new();
        let text =
            "Research study analysis experiment scientific method laboratory testing validation";

        let result = modeler.extract_topics(text)?;

        let quality = &result.metadata.quality_metrics;
        assert!(quality.coherence >= 0.0 && quality.coherence <= 1.0);
        assert!(quality.diversity >= 0.0 && quality.diversity <= 1.0);
        assert!(quality.interpretability >= 0.0 && quality.interpretability <= 1.0);

        Ok(())
    }

    #[test]
    fn test_convenience_functions() -> Result<(), TopicModelingError> {
        let text = "Technology software development programming algorithms";

        let basic_result = extract_basic_topics(text)?;
        assert!(!basic_result.topics.is_empty());

        let tfidf_result = extract_tfidf_topics(text, 5)?;
        assert!(!tfidf_result.topics.is_empty());
        assert_eq!(
            tfidf_result.metadata.approach_used,
            TopicModelingApproach::TfIdf
        );

        let similarity = compare_topic_similarity_simple(
            "technology software programming",
            "computer development algorithms",
        )?;
        assert!(similarity >= 0.0 && similarity <= 1.0);

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        let mut modeler = TopicModeler::new();

        // Test empty text
        let result = modeler.extract_topics("");
        assert!(matches!(
            result,
            Err(TopicModelingError::InvalidInput { .. })
        ));

        // Test text with insufficient vocabulary
        let result = modeler.extract_topics("a b c");
        assert!(matches!(
            result,
            Err(TopicModelingError::InsufficientVocabulary { .. })
        ));
    }

    #[test]
    fn test_topic_evolution_analysis() -> Result<(), TopicModelingError> {
        let config = TopicModelingConfig::new().with_dynamic_modeling(true);
        let mut modeler = TopicModeler::with_config(config);

        let text = "Technology starts simple. Then innovation accelerates progress. Finally systems become complex. The future holds more advancement.";

        let result = modeler.extract_topics(text)?;

        if let Some(evolution) = &result.topic_evolution {
            assert!(!evolution.segment_distributions.is_empty());
            assert!(!evolution.stability_scores.is_empty());
            assert!(!evolution.evolution_patterns.is_empty());
        }

        Ok(())
    }
}
