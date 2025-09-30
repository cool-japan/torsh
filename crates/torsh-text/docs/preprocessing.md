# Text Preprocessing Guide

## Overview

The torsh-text crate provides a comprehensive text preprocessing pipeline that can be customized for different NLP tasks. The preprocessing system includes normalization, cleaning, augmentation, and encoding components.

## Core Components

### Text Normalization

```rust
use torsh_text::utils::TextNormalizer;

let normalizer = TextNormalizer::new()
    .with_unicode_normalization(true)
    .with_accent_removal(true)
    .with_punctuation_normalization(true)
    .with_digit_normalization(true)
    .with_whitespace_normalization(true);

let text = "Héllo  Wörld123!!!";
let normalized = normalizer.normalize(text)?;
// Result: "Hello World 123 !"
```

### Text Cleaning

```rust
use torsh_text::utils::TextCleaner;

let cleaner = TextCleaner::new()
    .remove_urls(true)
    .remove_emails(true)
    .remove_html_tags(true)
    .remove_mentions(true)
    .remove_hashtags(true)
    .remove_special_chars(true);

let text = "Check out https://example.com @user #hashtag <script>alert('xss')</script>";
let cleaned = cleaner.clean(text)?;
// Result: "Check out"
```

### Text Augmentation

```rust
use torsh_text::utils::TextAugmenter;

let augmenter = TextAugmenter::new()
    .with_synonym_replacement(0.1) // Replace 10% of words
    .with_random_insertion(0.1)    // Insert 10% new words
    .with_random_deletion(0.1)     // Delete 10% of words
    .with_random_swap(0.1);        // Swap 10% of word pairs

let text = "This is a sample sentence";
let augmented = augmenter.augment(text)?;
```

### Unified Preprocessing Pipeline

```rust
use torsh_text::utils::TextPreprocessingPipeline;

let mut pipeline = TextPreprocessingPipeline::new();

// Add normalization step
pipeline.add_normalization(true, true, true); // unicode, accents, punctuation

// Add cleaning step  
pipeline.add_cleaning(true, true, false); // urls, emails, html

// Add custom step
pipeline.add_custom_step(Box::new(|text: &str| {
    text.to_lowercase()
}));

// Process text
let processed = pipeline.process("Input TEXT with URLs https://example.com")?;
```

## Advanced Features

### Task-Specific Pipelines

```rust
use torsh_text::utils::{
    TextPreprocessingPipeline,
    SentimentAnalysisPipeline,
    NamedEntityRecognitionPipeline,
    MachineTranslationPipeline
};

// Sentiment analysis preprocessing
let sentiment_pipeline = SentimentAnalysisPipeline::new()
    .preserve_emoticons(true)
    .normalize_punctuation(true)
    .handle_negations(true);

// NER preprocessing
let ner_pipeline = NamedEntityRecognitionPipeline::new()
    .preserve_casing(true)
    .preserve_punctuation(true)
    .tokenize_subwords(false);

// Machine translation preprocessing
let mt_pipeline = MachineTranslationPipeline::new()
    .normalize_unicode(true)
    .preserve_entities(true)
    .handle_code_switching(true);
```

### Batch Processing

```rust
use torsh_text::utils::BatchProcessor;

let processor = BatchProcessor::new(1000); // batch_size

let texts = vec!["text1", "text2", "text3", /* ... thousands more */];
let processed_texts = processor.process_parallel(&texts, &pipeline)?;
```

### Streaming Processing

```rust
use torsh_text::utils::StreamingBatchProcessor;

let processor = StreamingBatchProcessor::new(512, 4); // batch_size=512, num_threads=4

// Process large file without loading everything into memory
processor.process_file("large_text_file.txt", "processed_output.txt", &pipeline)?;
```

## Sequence Processing

### Padding and Truncation

```rust
use torsh_text::utils::{pad_sequence, truncate_sequence, PaddingStrategy, TruncationStrategy};

// Padding
let tokens = vec!["hello", "world"];
let padded = pad_sequence(&tokens, 5, "<pad>", PaddingStrategy::Right)?;
// Result: ["hello", "world", "<pad>", "<pad>", "<pad>"]

// Truncation
let long_tokens = vec!["a", "b", "c", "d", "e", "f"];
let truncated = truncate_sequence(&long_tokens, 4, TruncationStrategy::Right)?;
// Result: ["a", "b", "c", "d"]
```

### Encoding Schemes

```rust
use torsh_text::utils::{one_hot_encode, label_encode};

// One-hot encoding
let labels = vec!["cat", "dog", "cat", "bird"];
let encoded = one_hot_encode(&labels)?;

// Label encoding
let encoded_labels = label_encode(&labels)?;
// Result: [0, 1, 0, 2] (cat=0, dog=1, bird=2)
```

## Memory-Optimized Processing

### String Interning

```rust
use torsh_text::utils::StringInterner;

let mut interner = StringInterner::new();
let interned_id = interner.intern("common_string");

// Reuse interned strings to save memory
let same_id = interner.intern("common_string");
assert_eq!(interned_id, same_id);
```

### Memory Pool

```rust
use torsh_text::utils::MemoryPool;

let pool = MemoryPool::new(1024 * 1024); // 1MB pool
let processed_text = pool.process_with_reuse(text, &pipeline)?;
```

## Configuration and Persistence

### Save/Load Pipeline Configuration

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct PipelineConfig {
    normalize_unicode: bool,
    remove_urls: bool,
    augmentation_rate: f32,
}

// Save configuration
let config = PipelineConfig {
    normalize_unicode: true,
    remove_urls: true,
    augmentation_rate: 0.1,
};
let json = serde_json::to_string(&config)?;

// Load and apply configuration
let loaded_config: PipelineConfig = serde_json::from_str(&json)?;
let pipeline = TextPreprocessingPipeline::from_config(loaded_config)?;
```

## Best Practices

1. **Pipeline Design**:
   - Keep preprocessing steps simple and composable
   - Test each step individually before combining
   - Document the rationale for each preprocessing choice

2. **Performance**:
   - Use batch processing for large datasets
   - Enable parallel processing when possible
   - Profile memory usage for large text corpora

3. **Reproducibility**:
   - Save preprocessing configurations
   - Use deterministic operations when possible
   - Version your preprocessing pipelines

4. **Task-Specific Considerations**:
   - Preserve important features for your task (e.g., casing for NER)
   - Consider the impact of preprocessing on model performance
   - Validate preprocessing effects with ablation studies

5. **Error Handling**:
   - Handle edge cases (empty strings, special characters)
   - Validate input text encoding
   - Log preprocessing statistics for monitoring

6. **Memory Management**:
   - Use streaming for very large datasets
   - Consider memory pools for frequent allocations
   - Monitor memory usage in production systems