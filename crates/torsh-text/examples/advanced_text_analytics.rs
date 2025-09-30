//! Advanced Text Analytics Example
//!
//! This example demonstrates the enhanced SciRS2 integration features
//! including advanced text statistics, complexity analysis, text sampling,
//! and performance monitoring.

use torsh_text::prelude::*;
use torsh_text::{AdvancedTextSampler, AdvancedTextStats, ComplexityAnalyzer, PerformanceMonitor};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Sample text for analysis
    let text = r#"
        Natural language processing (NLP) is a subfield of linguistics, computer science,
        and artificial intelligence concerned with the interactions between computers and
        human language. In particular, it focuses on programming computers to process
        and analyze large amounts of natural language data. The goal is a computer
        capable of understanding the contents of documents, including the contextual
        nuances of the language within them. The technology can then accurately extract
        information and insights contained in the documents as well as categorize and
        organize the documents themselves.

        Challenges in natural language processing frequently involve speech recognition,
        natural language understanding, and natural language generation. Modern NLP
        algorithms are based on machine learning, especially statistical machine learning.
        The paradigm of machine learning is different from that of most prior attempts
        at language processing.
    "#;

    println!("🔍 Advanced Text Analytics Demo");
    println!("================================\n");

    // 1. Compute advanced text statistics
    println!("📊 Advanced Text Statistics:");
    println!("----------------------------");
    let stats = compute_advanced_stats(text)?;
    display_text_stats(&stats);

    // 2. Analyze text complexity
    println!("\n🧠 Text Complexity Analysis:");
    println!("-----------------------------");
    let complexity = ComplexityAnalyzer::analyze_complexity(text)?;
    display_complexity_metrics(&complexity);

    // 3. Demonstrate text sampling with SciRS2 random generation
    println!("\n🎲 Text Sampling with SciRS2 Random Generation:");
    println!("-----------------------------------------------");
    let mut sampler = AdvancedTextSampler::with_seed(123);

    let word_samples = sampler.sample_words(text, 5);
    println!("Random word samples: {:?}", word_samples);

    let sentence_samples = sampler.sample_sentences(text, 2);
    println!("Random sentence samples:");
    for (i, sentence) in sentence_samples.iter().enumerate() {
        println!("  {}. {}", i + 1, sentence);
    }

    // 4. Generate text using Markov chain
    println!("\n📝 Markov Chain Text Generation:");
    println!("--------------------------------");
    let generated_text = sampler.generate_markov_text(text, 20, 2);
    println!("Generated text: {}", generated_text);

    // 5. Performance monitoring
    println!("\n⚡ Performance Monitoring:");
    println!("-------------------------");
    let mut monitor = PerformanceMonitor::new("text_analysis");

    let (word_count, metrics) = monitor.time_operation(text, || text.split_whitespace().count());

    display_performance_metrics(&metrics, word_count);

    // 6. Demonstrate string operations
    println!("\n🔤 String Operations:");
    println!("--------------------");

    let char_freq = char_frequency(text)?;
    let most_frequent_chars: Vec<_> = char_freq
        .iter()
        .filter(|(ch, _)| ch.is_alphabetic())
        .collect();

    let mut sorted_chars = most_frequent_chars;
    sorted_chars.sort_by_key(|(_, count)| std::cmp::Reverse(**count));

    println!("Top 5 most frequent characters:");
    for (ch, count) in sorted_chars.iter().take(5) {
        println!("  '{}': {} occurrences", ch, count);
    }

    // 7. N-gram analysis
    println!("\n📊 N-gram Analysis:");
    println!("-------------------");
    let bigrams = ngram_frequency(text, 2)?;
    let mut sorted_bigrams: Vec<_> = bigrams.iter().collect();
    sorted_bigrams.sort_by_key(|(_, count)| std::cmp::Reverse(**count));

    println!("Top 5 most frequent bigrams:");
    for (bigram, count) in sorted_bigrams.iter().take(5) {
        println!("  '{}': {} occurrences", bigram, count);
    }

    // 8. Text similarity example
    let text1 = "Natural language processing is amazing";
    let text2 = "Language processing with computers is incredible";
    let similarity = cosine_similarity(text1, text2)?;
    println!(
        "\nText similarity between sample sentences: {:.3}",
        similarity
    );

    println!("\n✅ Analysis complete!");
    Ok(())
}

fn display_text_stats(stats: &AdvancedTextStats) {
    println!("  • Character count: {}", stats.char_count);
    println!("  • Word count: {}", stats.word_count);
    println!("  • Sentence count: {}", stats.sentence_count);
    println!("  • Paragraph count: {}", stats.paragraph_count);
    println!("  • Unique words: {}", stats.unique_words);
    println!("  • Lexical diversity: {:.3}", stats.lexical_diversity);
    println!("  • Average word length: {:.2}", stats.average_word_length);
    println!(
        "  • Average sentence length: {:.2}",
        stats.average_sentence_length
    );
    println!("  • Readability score: {:.1}", stats.readability_score);
}

fn display_complexity_metrics(complexity: &ComplexityMetrics) {
    println!(
        "  • Syntactic complexity: {:.3}",
        complexity.syntactic_complexity
    );
    println!(
        "  • Lexical complexity: {:.3}",
        complexity.lexical_complexity
    );
    println!(
        "  • Semantic complexity: {:.3}",
        complexity.semantic_complexity
    );
    println!(
        "  • Overall complexity: {:.3}",
        complexity.overall_complexity
    );
    println!("  • Readability score: {:.1}", complexity.readability_score);

    let complexity_level = match complexity.overall_complexity {
        x if x < 0.3 => "Low",
        x if x < 0.6 => "Medium",
        _ => "High",
    };
    println!("  • Complexity level: {}", complexity_level);
}

fn display_performance_metrics(metrics: &PerformanceMetrics, word_count: usize) {
    println!("  • Operation: {}", metrics.operation);
    println!(
        "  • Duration: {:.3}ms",
        metrics.duration.as_secs_f64() * 1000.0
    );
    println!(
        "  • Characters/sec: {:.0}",
        metrics.throughput_chars_per_sec
    );
    println!("  • Words/sec: {:.0}", metrics.throughput_words_per_sec);
    println!(
        "  • Memory estimate: {} bytes",
        metrics.memory_usage_estimate
    );
    println!("  • Words processed: {}", word_count);
}
