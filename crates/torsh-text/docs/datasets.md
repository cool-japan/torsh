# Datasets Guide

## Overview

The torsh-text crate provides comprehensive dataset handling for common NLP tasks including classification, sequence labeling, translation, and language modeling.

## Built-in Datasets

### IMDB Movie Reviews

```rust
use torsh_text::datasets::ImdbDataset;

// Load training data
let train_dataset = ImdbDataset::load("train")?;

// Access samples
for (i, sample) in train_dataset.iter().enumerate().take(5) {
    let (text, label) = sample?;
    println!("Review {}: {} -> {}", i, text, label);
}

// Get dataset info
println!("Size: {}", train_dataset.len());
println!("Classes: {:?}", train_dataset.class_names());
```

### AG News Classification

```rust
use torsh_text::datasets::AgNewsDataset;

let dataset = AgNewsDataset::load("train")?;

// AG News has 4 classes: World, Sports, Business, Sci/Tech
for sample in dataset.iter().take(3) {
    let (text, label) = sample?;
    let class_name = AgNewsDataset::label_to_class_name(label);
    println!("Article: {} -> {}", text, class_name);
}
```

### WikiText Language Modeling

```rust
use torsh_text::datasets::WikiTextDataset;

let dataset = WikiTextDataset::load("wikitext-2", "train")?;

// Configure for language modeling
let lm_dataset = dataset.to_language_modeling(512, 128)?; // seq_len=512, stride=128

for sequence in lm_dataset.iter().take(3) {
    println!("Sequence: {}", sequence?);
}
```

### Multi30k Translation

```rust
use torsh_text::datasets::Multi30kDataset;

let dataset = Multi30kDataset::load("train", "en", "de")?;

for sample in dataset.iter().take(3) {
    let (source, target) = sample?;
    println!("EN: {} -> DE: {}", source, target);
}
```

## Custom Datasets

### Classification Dataset

```rust
use torsh_text::datasets::ClassificationDataset;

// From CSV file
let dataset = ClassificationDataset::from_csv(
    "data.csv", 
    "text", 
    "label", 
    true // has_header
)?;

// From vectors
let texts = vec!["positive text", "negative text"];
let labels = vec!["pos", "neg"];
let dataset = ClassificationDataset::from_data(texts, labels)?;
```

### Sequence Labeling (NER/POS)

```rust
use torsh_text::datasets::SequenceLabelingDataset;

// From CoNLL format
let dataset = SequenceLabelingDataset::from_conll("ner.conll")?;

for sample in dataset.iter().take(3) {
    let (tokens, labels) = sample?;
    println!("Tokens: {:?}", tokens);
    println!("Labels: {:?}", labels);
}
```

### Translation Dataset

```rust
use torsh_text::datasets::TranslationDataset;

// From parallel files
let dataset = TranslationDataset::from_parallel_files("en.txt", "de.txt")?;

// From TSV file
let dataset = TranslationDataset::from_tsv("parallel.tsv", 0, 1)?; // source_col=0, target_col=1
```

## Dataset Utilities

### Data Loading and Caching

```rust
use torsh_text::datasets::DatasetDownloader;

// Download and cache datasets
let downloader = DatasetDownloader::new("./cache")?;
let path = downloader.download_and_extract(
    "https://example.com/dataset.tar.gz",
    "dataset.tar.gz"
)?;
```

### Unified Dataset Interface

```rust
use torsh_text::datasets::{UnifiedDatasetLoader, DatasetConfig};

let config = DatasetConfig {
    name: "imdb".to_string(),
    split: "train".to_string(),
    data_dir: "./data".to_string(),
    ..Default::default()
};

let dataset = UnifiedDatasetLoader::load(config)?;
```

### Dataset Consolidation

```rust
use torsh_text::datasets::ConsolidatedDataset;

// Combine multiple datasets
let datasets = vec![dataset1, dataset2, dataset3];
let consolidated = ConsolidatedDataset::new(datasets);

// Stratified sampling
let sample = consolidated.stratified_sample(1000)?;
```

## Data Processing Pipeline

### Preprocessing Integration

```rust
use torsh_text::utils::TextPreprocessingPipeline;
use torsh_text::datasets::ClassificationDataset;

// Create preprocessing pipeline
let mut pipeline = TextPreprocessingPipeline::new();
pipeline.add_normalization(true, true, true); // unicode, accents, punctuation
pipeline.add_cleaning(true, true, false); // urls, emails, html

// Apply to dataset
let dataset = ClassificationDataset::from_csv("data.csv", "text", "label", true)?;
let processed_dataset = dataset.apply_preprocessing(&pipeline)?;
```

### Batch Processing

```rust
use torsh_text::datasets::BatchProcessor;

let processor = BatchProcessor::new(32); // batch_size=32
let batches = processor.create_batches(&dataset)?;

for batch in batches {
    let (texts, labels) = batch?;
    // Process batch
}
```

## Best Practices

1. **Dataset Selection**:
   - Use built-in datasets for standard benchmarks
   - Create custom datasets for domain-specific tasks
   - Consider data licensing and usage rights

2. **Memory Management**:
   - Use streaming for large datasets
   - Enable caching for frequently accessed data
   - Consider data sharding for distributed training

3. **Data Quality**:
   - Always validate data integrity
   - Handle missing or corrupted samples
   - Implement proper train/validation/test splits

4. **Preprocessing**:
   - Apply consistent preprocessing across splits
   - Save preprocessing configuration for reproducibility
   - Validate preprocessing effects on performance

5. **Performance**:
   - Use parallel data loading when possible
   - Consider memory mapping for very large datasets
   - Profile data loading to identify bottlenecks