//! Convenience utilities example
//!
//! This example demonstrates the high-level convenience functions provided by torsh-text
//! for common text processing tasks. These utilities combine multiple lower-level
//! components for maximum ease of use.

use torsh_text::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Torsh-Text Convenience Utilities Demo\n");

    // =============================================================================
    // Quick Text Processing
    // =============================================================================

    println!("📝 Quick Text Processing");
    println!("========================");

    let processor = QuickTextProcessor::new().with_normalizer(
        TextNormalizer::new()
            .lowercase(true)
            .remove_punctuation(true),
    );

    let sample_text = "Hello, World! This is a SAMPLE text with 123 numbers and @mentions.";
    println!("Original: {}", sample_text);

    let tokens = processor.process(sample_text)?;
    println!("Processed: {:?}", tokens);

    let stats = processor.quick_stats(sample_text)?;
    println!("Quick stats: {:#?}\n", stats);

    // =============================================================================
    // Text Similarity
    // =============================================================================

    println!("🔍 Text Similarity Analysis");
    println!("===========================");

    let text1 = "The quick brown fox jumps over the lazy dog";
    let text2 = "A fast brown fox leaps over the sleepy dog";
    let text3 = "Machine learning is amazing";

    let sim12 = processor.similarity(text1, text2)?;
    let sim13 = processor.similarity(text1, text3)?;

    println!("Text 1: {}", text1);
    println!("Text 2: {}", text2);
    println!("Similarity: {:.3}\n", sim12);

    println!("Text 1: {}", text1);
    println!("Text 3: {}", text3);
    println!("Similarity: {:.3}\n", sim13);

    // =============================================================================
    // Batch Processing
    // =============================================================================

    println!("⚡ Batch Text Processing");
    println!("========================");

    // Create a new processor instance for batch processing
    let batch_processor_instance = QuickTextProcessor::new()
        .with_cleaner(
            TextCleaner::new()
                .remove_urls(true)
                .remove_emails(true)
                .remove_html(true),
        )
        .with_normalizer(TextNormalizer::default());
    let batch_processor = BatchTextProcessor::new(batch_processor_instance, 2);

    let texts = vec![
        "First document about machine learning".to_string(),
        "Second text on artificial intelligence".to_string(),
        "Third article discussing deep learning".to_string(),
        "Fourth paper on neural networks".to_string(),
    ];

    let batch_results = batch_processor.process_batch(&texts)?;
    println!("Batch processing results:");
    for (i, result) in batch_results.iter().enumerate() {
        println!("  Document {}: {:?}", i + 1, result);
    }

    // Similarity matrix
    let sim_matrix = batch_processor.similarity_matrix(&texts)?;
    println!("\nSimilarity Matrix:");
    for (i, row) in sim_matrix.iter().enumerate() {
        print!("Doc {}: ", i + 1);
        for (j, &sim) in row.iter().enumerate() {
            if i != j {
                print!("{:.3} ", sim);
            } else {
                print!("1.000 ");
            }
        }
        println!();
    }
    println!();

    // =============================================================================
    // Language Detection
    // =============================================================================

    println!("🌍 Language Detection");
    println!("=====================");

    let detector = LanguageDetector::new();

    let english_text = "The quick brown fox jumps over the lazy dog";
    let spanish_text = "El zorro marrón rápido salta sobre el perro perezoso";

    if let Some(lang) = detector.detect(english_text) {
        println!("'{}' detected as: {}", english_text, lang);
    }

    if let Some(lang) = detector.detect(spanish_text) {
        println!("'{}' detected as: {}", spanish_text, lang);
    }
    println!();

    // =============================================================================
    // Text Quality Assessment
    // =============================================================================

    println!("📊 Text Quality Assessment");
    println!("==========================");

    let simple_text = "This is easy. Short words. Simple ideas.";
    let complex_text = "The implementation demonstrates sophisticated algorithmic approaches utilizing advanced computational methodologies for comprehensive textual analysis.";

    println!("Simple text: '{}'", simple_text);
    println!(
        "  Readability: {:.1}",
        TextQualityAssessor::readability_score(simple_text)
    );
    println!(
        "  Lexical diversity: {:.3}",
        TextQualityAssessor::lexical_diversity(simple_text)
    );

    println!("\nComplex text: '{}'", complex_text);
    println!(
        "  Readability: {:.1}",
        TextQualityAssessor::readability_score(complex_text)
    );
    println!(
        "  Lexical diversity: {:.3}",
        TextQualityAssessor::lexical_diversity(complex_text)
    );

    // Spam detection
    let potential_spam = "FREE!!! WIN NOW!!! CLICK HERE FOR AMAZING OFFERS!!!";
    let spam_indicators = TextQualityAssessor::spam_indicators(potential_spam);

    println!("\nSpam analysis for: '{}'", potential_spam);
    for (indicator, score) in spam_indicators {
        println!("  {}: {:.3}", indicator, score);
    }
    println!();

    // =============================================================================
    // Complete Pipeline Example
    // =============================================================================

    println!("🔄 Complete Pipeline Example");
    println!("============================");

    let documents = vec![
        "Natural language processing (NLP) is a field of computer science and linguistics."
            .to_string(),
        "Machine learning algorithms can automatically learn patterns from data.".to_string(),
        "Deep learning is a subset of machine learning using neural networks.".to_string(),
        "FREE MONEY!!! CLICK NOW!!! Amazing offers await you!!!".to_string(),
    ];

    println!(
        "Processing {} documents through complete pipeline:\n",
        documents.len()
    );

    for (i, doc) in documents.iter().enumerate() {
        println!("Document {}: '{}'", i + 1, doc);

        // Process text
        let tokens = processor.process(doc)?;
        println!("  Tokens: {} words", tokens.len());

        // Detect language
        if let Some(lang) = detector.detect(doc) {
            println!("  Language: {}", lang);
        }

        // Quality metrics
        let readability = TextQualityAssessor::readability_score(doc);
        let diversity = TextQualityAssessor::lexical_diversity(doc);
        println!(
            "  Readability: {:.1}, Diversity: {:.3}",
            readability, diversity
        );

        // Spam indicators
        let spam_indicators = TextQualityAssessor::spam_indicators(doc);
        let spam_score = spam_indicators.values().sum::<f64>() / spam_indicators.len() as f64;
        println!("  Spam score: {:.3}", spam_score);

        println!();
    }

    println!("✅ Convenience utilities demo completed successfully!");
    Ok(())
}
