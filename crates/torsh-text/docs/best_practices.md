# Best Practices Guide

## Overview

This guide outlines best practices for using the torsh-text crate effectively in production NLP applications. Follow these guidelines to ensure optimal performance, maintainability, and reliability.

## Tokenization Best Practices

### 1. Choose the Right Tokenizer

```rust
// For general text with rich vocabulary
let bpe_tokenizer = BPETokenizer::new();

// For transformer models (BERT-like)
let wordpiece_tokenizer = WordPieceTokenizer::new();

// For character-level modeling
let char_tokenizer = CharTokenizer::new();

// For simple prototyping
let whitespace_tokenizer = WhitespaceTokenizer::new();
```

**Guidelines:**
- Use BPE for most modern language models
- Use WordPiece for BERT-family models
- Use character-level for morphologically rich languages
- Consider subword regularization during training

### 2. Vocabulary Management

```rust
// Always include essential special tokens
let mut vocab = Vocabulary::new();
vocab.add_special_token("pad", "<pad>");
vocab.add_special_token("unk", "<unk>");
vocab.add_special_token("bos", "<s>");
vocab.add_special_token("eos", "</s>");

// Use frequency filtering for large vocabularies
let vocab = Vocabulary::from_texts(&texts, Some(5))?; // min_freq = 5
```

**Guidelines:**
- Always include padding, unknown, beginning, and end tokens
- Use consistent vocabulary across training and inference
- Consider frequency thresholds to manage vocabulary size
- Save vocabulary configurations for reproducibility

### 3. Performance Optimization

```rust
// Enable caching for frequently tokenized text
let mut tokenizer = FastTokenizer::new(base_tokenizer);
tokenizer.enable_caching(10000);

// Use batch processing
let texts = vec!["text1", "text2", "text3"];
let tokens = tokenizer.batch_encode(&texts)?;

// Consider subword regularization only during training
if training_mode {
    let regularizer = SubwordRegularizer::new(0.1);
    tokens = regularizer.regularize(&tokens)?;
}
```

## Dataset Management Best Practices

### 1. Data Organization

```rust
// Use consistent data splits
let (train, temp) = dataset.train_test_split(0.8, Some(42))?;
let (val, test) = temp.train_test_split(0.5, Some(42))?;

// Maintain stratification for classification
let train = dataset.stratified_split("train", 0.8)?;
```

**Guidelines:**
- Use fixed random seeds for reproducible splits
- Maintain class balance in splits when possible
- Keep test sets completely separate until final evaluation
- Document data provenance and preprocessing steps

### 2. Memory Management

```rust
// Use streaming for large datasets
let streaming_processor = StreamingBatchProcessor::new(512, 4);
streaming_processor.process_file("large_file.txt", "output.txt", &pipeline)?;

// Enable caching for frequently accessed data
let dataset = ClassificationDataset::from_csv("data.csv", "text", "label", true)?
    .with_caching(true);
```

**Guidelines:**
- Stream data for datasets larger than available RAM
- Use appropriate batch sizes (typically 16-128 for text)
- Monitor memory usage in production
- Consider data sharding for distributed training

### 3. Data Validation

```rust
// Validate data integrity
fn validate_dataset(dataset: &ClassificationDataset) -> Result<()> {
    let stats = dataset.get_statistics();
    
    // Check for empty texts
    if stats.min_text_length == 0 {
        return Err(anyhow::anyhow!("Found empty texts in dataset"));
    }
    
    // Check label distribution
    if stats.label_distribution.len() < 2 {
        return Err(anyhow::anyhow!("Need at least 2 classes"));
    }
    
    // Check for extreme imbalance
    let counts: Vec<_> = stats.label_distribution.values().collect();
    let max_count = *counts.iter().max().unwrap();
    let min_count = *counts.iter().min().unwrap();
    
    if max_count > min_count * 10 {
        eprintln!("Warning: Significant class imbalance detected");
    }
    
    Ok(())
}
```

## Text Preprocessing Best Practices

### 1. Task-Specific Preprocessing

```rust
// Sentiment Analysis - preserve emoticons and intensity
let sentiment_pipeline = TextPreprocessingPipeline::new()
    .add_normalization(true, false, false) // Light normalization
    .add_cleaning(true, true, false)       // Remove URLs/emails, keep emoticons
    .add_custom_step(Box::new(|text| text.to_lowercase()));

// Named Entity Recognition - preserve casing and structure
let ner_pipeline = TextPreprocessingPipeline::new()
    .add_normalization(true, false, false) // Only unicode normalization
    .add_cleaning(false, false, false);    // Minimal cleaning

// Machine Translation - normalize but preserve meaning
let mt_pipeline = TextPreprocessingPipeline::new()
    .add_normalization(true, true, true)   // Full normalization
    .add_cleaning(true, true, true);       // Clean but preserve structure
```

**Guidelines:**
- Tailor preprocessing to your specific task
- Document preprocessing choices and rationale
- Validate preprocessing effects with ablation studies
- Keep preprocessing consistent between training and inference

### 2. Performance and Scalability

```rust
// Use parallel processing for large corpora
let processor = BatchProcessor::new(1000);
let processed_texts = processor.process_parallel(&texts, &pipeline)?;

// Memory-efficient processing
let pool = MemoryPool::new(1024 * 1024); // 1MB pool
let processed = pool.process_with_reuse(text, &pipeline)?;

// String interning for memory optimization
let mut interner = StringInterner::new();
let interned_id = interner.intern("frequent_string");
```

### 3. Configuration Management

```rust
// Save preprocessing configuration
#[derive(Serialize, Deserialize)]
struct PreprocessingConfig {
    normalize_unicode: bool,
    remove_urls: bool,
    lowercase: bool,
    augmentation_rate: f32,
}

// Use configuration files in production
let config: PreprocessingConfig = serde_json::from_str(&config_json)?;
let pipeline = TextPreprocessingPipeline::from_config(config)?;
```

## Error Handling and Robustness

### 1. Graceful Error Handling

```rust
// Handle tokenization errors gracefully
fn safe_tokenize(tokenizer: &impl Tokenizer, text: &str) -> Vec<String> {
    match tokenizer.tokenize(text) {
        Ok(tokens) => tokens,
        Err(e) => {
            eprintln!("Tokenization failed for text: '{}', error: {}", text, e);
            // Fallback to simple whitespace tokenization
            text.split_whitespace().map(|s| s.to_string()).collect()
        }
    }
}

// Validate inputs
fn validate_text_input(text: &str) -> Result<()> {
    if text.is_empty() {
        return Err(anyhow::anyhow!("Empty text not allowed"));
    }
    
    if text.len() > 1_000_000 {
        return Err(anyhow::anyhow!("Text too long: {} characters", text.len()));
    }
    
    // Check for valid UTF-8
    if !text.is_ascii() && text.chars().any(|c| c.is_control() && c != '\n' && c != '\t') {
        return Err(anyhow::anyhow!("Invalid characters in text"));
    }
    
    Ok(())
}
```

### 2. Input Validation

```rust
// Validate dataset consistency
fn validate_dataset_consistency(datasets: &[&ClassificationDataset]) -> Result<()> {
    if datasets.is_empty() {
        return Err(anyhow::anyhow!("No datasets provided"));
    }
    
    let first_labels: HashSet<_> = datasets[0].get_label_mapping().keys().collect();
    
    for (i, dataset) in datasets.iter().enumerate().skip(1) {
        let labels: HashSet<_> = dataset.get_label_mapping().keys().collect();
        if labels != first_labels {
            return Err(anyhow::anyhow!(
                "Label mismatch between dataset 0 and dataset {}", i
            ));
        }
    }
    
    Ok(())
}
```

## Production Deployment

### 1. Performance Monitoring

```rust
use std::time::Instant;

// Monitor processing times
fn timed_process(text: &str, pipeline: &TextPreprocessingPipeline) -> Result<String> {
    let start = Instant::now();
    let result = pipeline.process(text)?;
    let duration = start.elapsed();
    
    if duration.as_millis() > 100 {
        eprintln!("Slow processing detected: {} ms for {} chars", 
                  duration.as_millis(), text.len());
    }
    
    Ok(result)
}

// Track memory usage
fn monitor_memory_usage<T>(f: impl FnOnce() -> T) -> T {
    let before = get_memory_usage();
    let result = f();
    let after = get_memory_usage();
    
    if after - before > 100_000_000 { // 100MB
        eprintln!("High memory usage detected: {} MB", (after - before) / 1_000_000);
    }
    
    result
}
```

### 2. Caching and Optimization

```rust
// Implement intelligent caching
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub struct ProcessingCache {
    cache: Arc<RwLock<HashMap<String, String>>>,
    max_size: usize,
}

impl ProcessingCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
        }
    }
    
    pub fn get_or_process<F>(&self, key: &str, processor: F) -> Result<String>
    where
        F: FnOnce(&str) -> Result<String>,
    {
        // Check cache first
        if let Ok(cache) = self.cache.read() {
            if let Some(result) = cache.get(key) {
                return Ok(result.clone());
            }
        }
        
        // Process and cache
        let result = processor(key)?;
        
        if let Ok(mut cache) = self.cache.write() {
            if cache.len() >= self.max_size {
                cache.clear(); // Simple eviction strategy
            }
            cache.insert(key.to_string(), result.clone());
        }
        
        Ok(result)
    }
}
```

### 3. Configuration and Versioning

```rust
// Version your preprocessing pipelines
#[derive(Serialize, Deserialize)]
struct VersionedConfig {
    version: String,
    preprocessing: PreprocessingConfig,
    tokenizer: TokenizerConfig,
    created_at: String,
}

// Save complete configuration
fn save_processing_config(config: &VersionedConfig, path: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(config)?;
    std::fs::write(path, json)?;
    Ok(())
}

// Validate config compatibility
fn validate_config_compatibility(
    training_config: &VersionedConfig,
    inference_config: &VersionedConfig,
) -> Result<()> {
    if training_config.version != inference_config.version {
        return Err(anyhow::anyhow!(
            "Version mismatch: training={}, inference={}",
            training_config.version,
            inference_config.version
        ));
    }
    Ok(())
}
```

## Testing and Validation

### 1. Unit Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_preprocessing_deterministic() {
        let pipeline = TextPreprocessingPipeline::new()
            .add_normalization(true, true, true)
            .add_cleaning(true, true, true);
        
        let text = "Test input with URLs https://example.com";
        let result1 = pipeline.process(text).unwrap();
        let result2 = pipeline.process(text).unwrap();
        
        assert_eq!(result1, result2, "Preprocessing should be deterministic");
    }
    
    #[test]
    fn test_tokenization_roundtrip() {
        let tokenizer = BPETokenizer::new();
        // Train tokenizer...
        
        let text = "hello world test";
        let tokens = tokenizer.tokenize(text).unwrap();
        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();
        
        // Verify roundtrip consistency
        assert_eq!(text.replace(" ", ""), decoded.replace(" ", ""));
    }
}
```

### 2. Integration Testing

```rust
#[test]
fn test_full_pipeline_integration() {
    // Test complete pipeline from raw text to model input
    let raw_text = "This is a test with URLs https://example.com and emails test@test.com";
    
    // Preprocessing
    let pipeline = TextPreprocessingPipeline::new()
        .add_cleaning(true, true, true)
        .add_normalization(true, true, true);
    let processed = pipeline.process(raw_text).unwrap();
    
    // Tokenization
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize(&processed).unwrap();
    
    // Validation
    assert!(!tokens.is_empty());
    assert!(!tokens.iter().any(|t| t.contains("http")));
    assert!(!tokens.iter().any(|t| t.contains("@")));
}
```

## Summary

Following these best practices will help you:

1. **Build robust pipelines** that handle edge cases gracefully
2. **Optimize performance** for production workloads
3. **Ensure reproducibility** across different environments
4. **Maintain code quality** with proper testing and validation
5. **Scale effectively** as your data and requirements grow

Remember to always profile your specific use case and adjust these guidelines based on your requirements and constraints.