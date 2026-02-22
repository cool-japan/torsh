use std::time::{Duration, Instant};
use torsh_text::prelude::*;
use torsh_text::TextAnalyzer;

/// A simple benchmarking utility for text processing operations
#[derive(Debug)]
pub struct BenchmarkResult {
    pub operation: String,
    pub total_time: Duration,
    pub avg_time_per_item: Duration,
    pub items_per_second: f64,
    pub memory_usage_mb: f64,
}

impl BenchmarkResult {
    pub fn new(operation: String, total_time: Duration, item_count: usize) -> Self {
        let avg_time_per_item = total_time / item_count as u32;
        let items_per_second = item_count as f64 / total_time.as_secs_f64();

        Self {
            operation,
            total_time,
            avg_time_per_item,
            items_per_second,
            memory_usage_mb: 0.0, // Could be enhanced with actual memory tracking
        }
    }

    pub fn print_summary(&self) {
        println!("\n📊 Benchmark Results for: {}", self.operation);
        println!("├─ Total time: {:?}", self.total_time);
        println!("├─ Avg time per item: {:?}", self.avg_time_per_item);
        println!("├─ Items per second: {:.2}", self.items_per_second);
        println!("└─ Memory usage: {:.2} MB", self.memory_usage_mb);
    }
}

/// Benchmark tokenization performance
fn benchmark_tokenization() -> Result<Vec<BenchmarkResult>> {
    println!("🔍 Benchmarking Tokenization Performance...");

    // Sample data
    let texts = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is a fascinating field of artificial intelligence.",
        "Tokenization is the process of breaking down text into smaller units called tokens.",
        "This benchmark measures the performance of different tokenizers on various text inputs.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    ];

    let large_text_set: Vec<String> = (0..1000)
        .map(|i| format!("This is sample text number {} for performance testing.", i))
        .collect();

    let mut results = Vec::new();

    // Benchmark WhitespaceTokenizer
    let whitespace_tokenizer = WhitespaceTokenizer::new();
    let start = Instant::now();
    for text in &large_text_set {
        let _tokens = whitespace_tokenizer.tokenize(text)?;
    }
    let duration = start.elapsed();
    results.push(BenchmarkResult::new(
        "WhitespaceTokenizer".to_string(),
        duration,
        large_text_set.len(),
    ));

    // Benchmark CharTokenizer
    let char_tokenizer = CharTokenizer::new(None);
    let start = Instant::now();
    for text in &large_text_set {
        let _tokens = char_tokenizer.tokenize(text)?;
    }
    let duration = start.elapsed();
    results.push(BenchmarkResult::new(
        "CharTokenizer".to_string(),
        duration,
        large_text_set.len(),
    ));

    // Benchmark BPE Tokenizer (basic)
    let bpe_tokenizer = BPETokenizer::new();
    let start = Instant::now();
    for text in &texts {
        // Use smaller set for BPE as it's more complex
        let _tokens = bpe_tokenizer.tokenize(text)?;
    }
    let duration = start.elapsed();
    results.push(BenchmarkResult::new(
        "BPETokenizer".to_string(),
        duration,
        texts.len(),
    ));

    Ok(results)
}

/// Benchmark text preprocessing performance
fn benchmark_preprocessing() -> Result<Vec<BenchmarkResult>> {
    println!("🧹 Benchmarking Text Preprocessing Performance...");

    let texts = vec![
        "Visit our website at https://example.com for more information!",
        "Contact us at support@company.com or call +1-555-0123.",
        "This text contains <b>HTML tags</b> and &nbsp; entities.",
        "Mixed CASE text with Àccénted chárácters and 123 numbers.",
        "Text with  multiple    spaces   and\t\ttabs\n\nand newlines.",
    ];

    let large_text_set: Vec<String> = texts
        .iter()
        .cycle()
        .take(1000)
        .map(|s| s.to_string())
        .collect();

    let mut results = Vec::new();

    // Benchmark basic normalization
    let normalizer = TextNormalizer::new().lowercase(true).remove_accents(true);

    let start = Instant::now();
    for text in &large_text_set {
        let _normalized = normalizer.normalize(text);
    }
    let duration = start.elapsed();
    results.push(BenchmarkResult::new(
        "Text Normalization".to_string(),
        duration,
        large_text_set.len(),
    ));

    // Benchmark text cleaning
    let cleaner = TextCleaner::new()
        .remove_urls(true)
        .remove_emails(true)
        .remove_html(true);

    let start = Instant::now();
    for text in &large_text_set {
        let _cleaned = cleaner.clean(text);
    }
    let duration = start.elapsed();
    results.push(BenchmarkResult::new(
        "Text Cleaning".to_string(),
        duration,
        large_text_set.len(),
    ));

    // Benchmark full preprocessing pipeline
    let pipeline = TextPreprocessingPipeline::new()
        .with_normalization(TextNormalizer::default())
        .with_cleaning(
            TextCleaner::new()
                .remove_urls(true)
                .remove_emails(true)
                .remove_html(true),
        );

    let start = Instant::now();
    for text in &large_text_set {
        let _processed = pipeline.process_text(text)?;
    }
    let duration = start.elapsed();
    results.push(BenchmarkResult::new(
        "Full Preprocessing Pipeline".to_string(),
        duration,
        large_text_set.len(),
    ));

    Ok(results)
}

/// Benchmark text analysis performance
fn benchmark_analysis() -> Result<Vec<BenchmarkResult>> {
    println!("📈 Benchmarking Text Analysis Performance...");

    let documents = vec![
        "The quick brown fox jumps over the lazy dog. This is a sample document for testing.",
        "Natural language processing involves computational linguistics and machine learning techniques.",
        "Text analysis includes various operations like statistics calculation and similarity measurement.",
        "Performance benchmarking helps identify bottlenecks in text processing pipelines.",
        "Document analysis and text mining are important applications of NLP technology.",
    ];

    let large_doc_set: Vec<String> = documents
        .iter()
        .cycle()
        .take(500)
        .map(|s| s.to_string())
        .collect();

    let mut results = Vec::new();

    // Benchmark text statistics
    let start = Instant::now();
    for doc in &large_doc_set {
        let analyzer = TextAnalyzer::default();
        let _stats = analyzer.analyze(doc)?;
    }
    let duration = start.elapsed();
    results.push(BenchmarkResult::new(
        "Text Statistics".to_string(),
        duration,
        large_doc_set.len(),
    ));

    // Benchmark n-gram extraction
    let extractor = NgramExtractor::new(2); // bigrams
    let start = Instant::now();
    for doc in &large_doc_set {
        let _ngrams = extractor.extract_word_ngrams(doc);
    }
    let duration = start.elapsed();
    results.push(BenchmarkResult::new(
        "N-gram Extraction".to_string(),
        duration,
        large_doc_set.len(),
    ));

    // Benchmark TF-IDF calculation
    let mut tfidf = TfIdfCalculator::new();

    let start = Instant::now();
    let documents_str: Vec<&str> = documents.iter().map(|s| s.as_ref()).collect();
    let _matrix = tfidf.fit_transform(&documents_str)?;
    let duration = start.elapsed();
    results.push(BenchmarkResult::new(
        "TF-IDF Transformation".to_string(),
        duration,
        documents.len(),
    ));

    Ok(results)
}

/// Benchmark vocabulary operations
fn benchmark_vocabulary() -> Result<Vec<BenchmarkResult>> {
    println!("📚 Benchmarking Vocabulary Operations...");

    let texts = vec![
        "hello world test sample",
        "natural language processing machine learning",
        "tokenization vocabulary building text analysis",
        "performance benchmarking optimization speed testing",
        "data science artificial intelligence deep learning",
    ];

    let large_corpus: Vec<String> = texts
        .iter()
        .cycle()
        .take(1000)
        .map(|s| s.to_string())
        .collect();

    let mut results = Vec::new();

    // Benchmark vocabulary building
    let start = Instant::now();
    let _vocab = Vocabulary::from_texts(&large_corpus, 1, Some(2));
    let duration = start.elapsed();
    results.push(BenchmarkResult::new(
        "Vocabulary Building".to_string(),
        duration,
        1, // Single operation
    ));

    // Benchmark vocabulary lookups
    let texts_string: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
    let vocab = Vocabulary::from_texts(&texts_string, 1, None);
    let tokens: Vec<String> = large_corpus
        .iter()
        .flat_map(|text| text.split_whitespace().map(|s| s.to_string()))
        .collect();

    let start = Instant::now();
    for token in &tokens {
        let _id = vocab.token_to_id(token);
    }
    let duration = start.elapsed();
    results.push(BenchmarkResult::new(
        "Vocabulary Lookups".to_string(),
        duration,
        tokens.len(),
    ));

    Ok(results)
}

/// Run comprehensive benchmarks
fn main() -> Result<()> {
    println!("🚀 ToRSh Text Processing Performance Benchmark");
    println!("===============================================");

    // Run all benchmarks
    let tokenization_results = benchmark_tokenization()?;
    let preprocessing_results = benchmark_preprocessing()?;
    let analysis_results = benchmark_analysis()?;
    let vocabulary_results = benchmark_vocabulary()?;

    // Print all results
    println!("\n📋 COMPREHENSIVE RESULTS");
    println!("========================");

    for result in tokenization_results {
        result.print_summary();
    }

    for result in preprocessing_results {
        result.print_summary();
    }

    for result in analysis_results {
        result.print_summary();
    }

    for result in vocabulary_results {
        result.print_summary();
    }

    println!("\n✨ Benchmark completed successfully!");
    println!("📝 Tips for optimization:");
    println!("   • Use caching for repeated operations");
    println!("   • Consider parallel processing for large datasets");
    println!("   • Profile memory usage for memory-intensive operations");
    println!("   • Choose appropriate tokenizers based on your use case");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result_creation() {
        let result =
            BenchmarkResult::new("Test Operation".to_string(), Duration::from_millis(100), 10);

        assert_eq!(result.operation, "Test Operation");
        assert_eq!(result.total_time, Duration::from_millis(100));
        assert_eq!(result.avg_time_per_item, Duration::from_millis(10));
        assert_eq!(result.items_per_second, 100.0);
    }

    #[test]
    fn test_tokenization_benchmark() -> Result<()> {
        let results = benchmark_tokenization()?;
        assert!(!results.is_empty());

        for result in results {
            assert!(result.total_time > Duration::from_nanos(0));
            assert!(result.items_per_second > 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_preprocessing_benchmark() -> Result<()> {
        let results = benchmark_preprocessing()?;
        assert!(!results.is_empty());

        for result in results {
            assert!(result.total_time > Duration::from_nanos(0));
            assert!(result.items_per_second > 0.0);
        }

        Ok(())
    }
}
