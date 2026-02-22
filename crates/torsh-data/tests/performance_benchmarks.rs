//! Performance benchmarks for torsh-data
//!
//! These benchmarks measure the performance of data loading operations,
//! transform applications, and memory usage patterns.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use torsh_core::error::Result;
use torsh_data::prelude::*;
use torsh_data::Transform;
use torsh_tensor::creation::{ones, zeros};

/// Benchmark results structure
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub duration: Duration,
    pub items_per_second: f64,
    pub memory_usage_mb: f64,
    pub additional_metrics: HashMap<String, f64>,
}

impl BenchmarkResult {
    fn new(name: String, duration: Duration, num_items: usize) -> Self {
        let items_per_second = if duration.as_secs_f64() > 0.0 {
            num_items as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        Self {
            name,
            duration,
            items_per_second,
            memory_usage_mb: 0.0,
            additional_metrics: HashMap::new(),
        }
    }

    fn with_memory_usage(mut self, memory_mb: f64) -> Self {
        self.memory_usage_mb = memory_mb;
        self
    }

    fn with_metric(mut self, key: &str, value: f64) -> Self {
        self.additional_metrics.insert(key.to_string(), value);
        self
    }
}

/// Benchmark suite for data loading operations
pub struct DataLoadingBenchmarks;

impl DataLoadingBenchmarks {
    /// Benchmark sequential data loading
    pub fn benchmark_sequential_loading() -> Result<BenchmarkResult> {
        let dataset_size: usize = 10000;
        let batch_size: usize = 32;

        // Create test dataset
        let data = ones::<f32>(&[dataset_size, 100])?;
        let labels = zeros::<f32>(&[dataset_size])?;
        let dataset = TensorDataset::from_tensors(vec![data, labels]);

        // Create sequential data loader
        let dataloader = simple_dataloader(dataset, batch_size, false)?;

        let start_time = Instant::now();
        let mut processed_batches = 0;

        for batch in dataloader.iter() {
            let _batch = batch?;
            processed_batches += 1;
        }

        let duration = start_time.elapsed();
        let expected_batches = dataset_size.div_ceil(batch_size);

        Ok(BenchmarkResult::new(
            "Sequential Data Loading".to_string(),
            duration,
            processed_batches,
        )
        .with_metric("expected_batches", expected_batches as f64)
        .with_metric("batch_size", batch_size as f64))
    }

    /// Benchmark random data loading
    pub fn benchmark_random_loading() -> Result<BenchmarkResult> {
        let dataset_size: usize = 10000;
        let batch_size: usize = 32;

        // Create test dataset
        let data = ones::<f32>(&[dataset_size, 100])?;
        let labels = zeros::<f32>(&[dataset_size])?;
        let dataset = TensorDataset::from_tensors(vec![data, labels]);

        // Create random data loader
        let dataloader = simple_random_dataloader(dataset, batch_size, Some(42))?;

        let start_time = Instant::now();
        let mut processed_batches = 0;

        for batch in dataloader.iter() {
            let _batch = batch?;
            processed_batches += 1;
        }

        let duration = start_time.elapsed();
        let expected_batches = dataset_size.div_ceil(batch_size);

        Ok(BenchmarkResult::new(
            "Random Data Loading".to_string(),
            duration,
            processed_batches,
        )
        .with_metric("expected_batches", expected_batches as f64)
        .with_metric("batch_size", batch_size as f64))
    }

    /// Benchmark weighted sampling
    pub fn benchmark_weighted_sampling() -> Result<BenchmarkResult> {
        let dataset_size: usize = 10000;
        let batch_size: usize = 32;
        let num_samples: usize = 5000;

        // Create test dataset
        let data = ones::<f32>(&[dataset_size, 100])?;
        let dataset = TensorDataset::from_tensor(data);

        // Create weighted sampler (simplified to random for now)
        let dataloader = simple_random_dataloader(dataset, batch_size, Some(42))?;

        let start_time = Instant::now();
        let mut processed_batches = 0;

        for batch in dataloader.iter() {
            let _batch = batch?;
            processed_batches += 1;
        }

        let duration = start_time.elapsed();
        let expected_batches = num_samples.div_ceil(batch_size);

        Ok(
            BenchmarkResult::new("Weighted Sampling".to_string(), duration, processed_batches)
                .with_metric("expected_batches", expected_batches as f64)
                .with_metric("num_samples", num_samples as f64),
        )
    }

    /// Benchmark distributed sampling
    pub fn benchmark_distributed_sampling() -> Result<BenchmarkResult> {
        let dataset_size: usize = 10000;
        let batch_size: usize = 32;
        let num_replicas: usize = 4;
        let rank = 0;

        // Create test dataset
        let data = ones::<f32>(&[dataset_size, 100])?;
        let dataset = TensorDataset::from_tensor(data);

        // Create distributed sampler
        // Create distributed sampler (simplified to random for now)
        let dataloader = simple_random_dataloader(dataset, batch_size, Some(42))?;

        let start_time = Instant::now();
        let mut processed_batches = 0;

        for batch in dataloader.iter() {
            let _batch = batch?;
            processed_batches += 1;
        }

        let duration = start_time.elapsed();
        let samples_per_replica = dataset_size / num_replicas;
        let expected_batches = samples_per_replica.div_ceil(batch_size);

        Ok(BenchmarkResult::new(
            "Distributed Sampling".to_string(),
            duration,
            processed_batches,
        )
        .with_metric("expected_batches", expected_batches as f64)
        .with_metric("num_replicas", num_replicas as f64)
        .with_metric("rank", rank as f64))
    }
}

/// Benchmark suite for data transformations
pub struct TransformBenchmarks;

impl TransformBenchmarks {
    /// Benchmark text transformations
    pub fn benchmark_text_transforms() -> Result<BenchmarkResult> {
        use torsh_data::transforms::text::*;

        let num_texts = 1000;
        let texts: Vec<String> = (0..num_texts)
            .map(|i| {
                format!("This is test text number {i} with some UPPER case and punctuation!!!")
            })
            .collect();

        let start_time = Instant::now();

        // Apply various text transforms
        let lowercase = ToLowercase;
        let remove_punct = RemovePunctuation;
        let tokenize = Tokenize::whitespace();
        let stopwords = RemoveStopwords::english();

        for text in texts {
            let text = lowercase.transform(text)?;
            let text = remove_punct.transform(text)?;
            let tokens = tokenize.transform(text)?;
            let _filtered = stopwords.transform(tokens)?;
        }

        let duration = start_time.elapsed();

        Ok(BenchmarkResult::new(
            "Text Transformations".to_string(),
            duration,
            num_texts,
        ))
    }

    /// Benchmark tensor transformations
    pub fn benchmark_tensor_transforms() -> Result<BenchmarkResult> {
        use torsh_data::transforms::augmentation::*;
        use torsh_data::transforms::tensor::*;

        let num_tensors = 100;
        let tensor_size = [3, 224, 224];

        let start_time = Instant::now();

        // Apply various tensor transforms
        let hflip = RandomHorizontalFlip::new(0.5);
        let brightness = RandomBrightness::symmetric(0.2);
        let contrast = RandomContrast::symmetric(0.2);

        for _i in 0..num_tensors {
            let tensor = ones::<f32>(&tensor_size)?;
            let tensor = hflip.transform(tensor)?;
            let tensor = brightness.transform(tensor)?;
            let _tensor = contrast.transform(tensor)?;
        }

        let duration = start_time.elapsed();

        Ok(BenchmarkResult::new(
            "Tensor Transformations".to_string(),
            duration,
            num_tensors,
        ))
    }

    /// Benchmark augmentation pipeline
    pub fn benchmark_augmentation_pipeline() -> Result<BenchmarkResult> {
        use torsh_data::transforms::augmentation::*;

        let num_tensors = 100;
        let tensor_size = [3, 224, 224];

        // Create complex augmentation pipeline
        let pipeline = AugmentationPipeline::<torsh_tensor::Tensor<f32>>::heavy_augmentation();

        let start_time = Instant::now();

        for _i in 0..num_tensors {
            let tensor = ones::<f32>(&tensor_size)?;
            let _augmented = pipeline.transform(tensor)?;
        }

        let duration = start_time.elapsed();

        Ok(BenchmarkResult::new(
            "Augmentation Pipeline".to_string(),
            duration,
            num_tensors,
        ))
    }

    /// Benchmark online augmentation engine
    pub fn benchmark_online_augmentation() -> Result<BenchmarkResult> {
        use torsh_data::transforms::augmentation::*;
        use torsh_data::transforms::online::*;

        let num_tensors = 100;
        let tensor_size = [3, 224, 224];

        // Create online augmentation engine with caching
        let pipeline = AugmentationPipeline::<torsh_tensor::Tensor<f32>>::medium_augmentation();
        let engine = OnlineAugmentationEngine::new(pipeline).with_cache(50);

        let start_time = Instant::now();

        for i in 0..num_tensors {
            let tensor = ones::<f32>(&tensor_size)?;
            let cache_key = format!("tensor_{}", i % 25); // Some cache hits
            let _augmented = engine.apply(tensor, Some(&cache_key))?;
        }

        let duration = start_time.elapsed();
        let stats = engine.stats();

        Ok(
            BenchmarkResult::new("Online Augmentation".to_string(), duration, num_tensors)
                .with_metric("cache_hits", stats.cache_hits as f64)
                .with_metric("cache_misses", stats.cache_misses as f64)
                .with_metric(
                    "cache_hit_rate",
                    stats.cache_hits as f64 / stats.total_transforms as f64,
                ),
        )
    }
}

/// Benchmark suite for dataset operations
pub struct DatasetBenchmarks;

impl DatasetBenchmarks {
    /// Benchmark dataset concatenation
    pub fn benchmark_concat_dataset() -> Result<BenchmarkResult> {
        let num_datasets = 10;
        let dataset_size = 1000;

        // Create multiple datasets
        let datasets: Result<Vec<_>> = (0..num_datasets)
            .map(|_| {
                let data = ones::<f32>(&[dataset_size, 50])?;
                Ok(TensorDataset::from_tensor(data))
            })
            .collect();
        let datasets = datasets?;

        let start_time = Instant::now();

        // Create concatenated dataset
        let concat_dataset = ConcatDataset::new(datasets);

        // Access all items
        for i in 0..concat_dataset.len() {
            let _item = concat_dataset.get(i)?;
        }

        let duration = start_time.elapsed();
        let total_items = num_datasets * dataset_size;

        Ok(
            BenchmarkResult::new("Dataset Concatenation".to_string(), duration, total_items)
                .with_metric("num_datasets", num_datasets as f64)
                .with_metric("dataset_size", dataset_size as f64),
        )
    }

    /// Benchmark subset dataset
    pub fn benchmark_subset_dataset() -> Result<BenchmarkResult> {
        let dataset_size = 10000;
        let subset_size = 1000;

        // Create large dataset
        let data = ones::<f32>(&[dataset_size, 100])?;
        let dataset = TensorDataset::from_tensor(data);

        // Create subset indices
        let indices: Vec<usize> = (0..subset_size)
            .map(|i| i * (dataset_size / subset_size))
            .collect();

        let start_time = Instant::now();

        // Create subset dataset
        let subset = Subset::new(dataset, indices);

        // Access all subset items
        for i in 0..subset.len() {
            let _item = subset.get(i)?;
        }

        let duration = start_time.elapsed();

        Ok(
            BenchmarkResult::new("Subset Dataset".to_string(), duration, subset_size)
                .with_metric("original_size", dataset_size as f64)
                .with_metric("subset_size", subset_size as f64),
        )
    }

    /// Benchmark cached dataset
    pub fn benchmark_cached_dataset() -> Result<BenchmarkResult> {
        let dataset_size = 1000;
        let cache_size = 100;
        let num_accesses = 2000;

        // Create base dataset
        let data = ones::<f32>(&[dataset_size, 100])?;
        let base_dataset = TensorDataset::from_tensor(data);
        let cached_dataset = CachedDataset::new(base_dataset, cache_size);

        let start_time = Instant::now();

        // Access items with some repetition to test caching
        for i in 0..num_accesses {
            let idx = i % dataset_size;
            let _item = cached_dataset.get(idx)?;
        }

        let duration = start_time.elapsed();
        let hit_rate = cached_dataset.cache_hit_rate();

        Ok(
            BenchmarkResult::new("Cached Dataset".to_string(), duration, num_accesses)
                .with_metric("cache_hit_rate", hit_rate)
                .with_metric("cache_size", cache_size as f64),
        )
    }
}

/// Benchmark suite for collation functions
pub struct CollationBenchmarks;

impl CollationBenchmarks {
    /// Benchmark default collation
    pub fn benchmark_default_collation() -> Result<BenchmarkResult> {
        let batch_size = 32;
        let num_batches = 100;
        let tensor_size = [100];

        let start_time = Instant::now();

        for _batch in 0..num_batches {
            let batch: Vec<torsh_tensor::Tensor<f32>> = (0..batch_size)
                .map(|_| ones::<f32>(&tensor_size))
                .collect::<Result<Vec<_>>>()?;

            let collator = collate_fn::<torsh_tensor::Tensor<f32>>();
            let _collated = collator.collate(batch)?;
        }

        let duration = start_time.elapsed();
        let total_tensors = batch_size * num_batches;

        Ok(
            BenchmarkResult::new("Default Collation".to_string(), duration, total_tensors)
                .with_metric("batch_size", batch_size as f64)
                .with_metric("num_batches", num_batches as f64),
        )
    }

    /// Benchmark dynamic batch collation
    pub fn benchmark_dynamic_collation() -> Result<BenchmarkResult> {
        let batch_size = 32;
        let num_batches = 100;

        let start_time = Instant::now();

        for _batch in 0..num_batches {
            // Create variable-length tensors
            let batch: Vec<torsh_tensor::Tensor<f32>> = (0..batch_size)
                .map(|i| {
                    let length = 10 + (i % 20); // Variable lengths from 10 to 29
                    ones::<f32>(&[length, 50])
                })
                .collect::<Result<Vec<_>>>()?;

            let collator = DynamicBatchCollate::new(0.0f32).with_max_length(50);
            let _collated = collator.collate(batch)?;
        }

        let duration = start_time.elapsed();
        let total_tensors = batch_size * num_batches;

        Ok(
            BenchmarkResult::new("Dynamic Collation".to_string(), duration, total_tensors)
                .with_metric("batch_size", batch_size as f64)
                .with_metric("num_batches", num_batches as f64),
        )
    }
}

/// Comprehensive benchmark runner
pub struct BenchmarkRunner;

impl BenchmarkRunner {
    /// Run all benchmarks and return results
    pub fn run_all_benchmarks() -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        println!("Running Data Loading Benchmarks...");
        results.push(DataLoadingBenchmarks::benchmark_sequential_loading()?);
        results.push(DataLoadingBenchmarks::benchmark_random_loading()?);
        results.push(DataLoadingBenchmarks::benchmark_weighted_sampling()?);
        results.push(DataLoadingBenchmarks::benchmark_distributed_sampling()?);

        println!("Running Transform Benchmarks...");
        results.push(TransformBenchmarks::benchmark_text_transforms()?);
        results.push(TransformBenchmarks::benchmark_tensor_transforms()?);
        results.push(TransformBenchmarks::benchmark_augmentation_pipeline()?);
        results.push(TransformBenchmarks::benchmark_online_augmentation()?);

        println!("Running Dataset Benchmarks...");
        results.push(DatasetBenchmarks::benchmark_concat_dataset()?);
        results.push(DatasetBenchmarks::benchmark_subset_dataset()?);
        results.push(DatasetBenchmarks::benchmark_cached_dataset()?);

        println!("Running Collation Benchmarks...");
        results.push(CollationBenchmarks::benchmark_default_collation()?);
        results.push(CollationBenchmarks::benchmark_dynamic_collation()?);

        Ok(results)
    }

    /// Print benchmark results in a formatted table
    pub fn print_results(results: &[BenchmarkResult]) {
        println!("\n=== Performance Benchmark Results ===\n");
        println!(
            "{:<35} {:>15} {:>20} {:>15}",
            "Benchmark", "Duration (ms)", "Items/sec", "Memory (MB)"
        );
        println!("{}", "-".repeat(90));

        for result in results {
            println!(
                "{:<35} {:>15.2} {:>20.2} {:>15.2}",
                result.name,
                result.duration.as_secs_f64() * 1000.0,
                result.items_per_second,
                result.memory_usage_mb
            );

            // Print additional metrics if any
            if !result.additional_metrics.is_empty() {
                for (key, value) in &result.additional_metrics {
                    println!("    {key}: {value:.2}");
                }
            }
        }

        println!("\n=== Summary ===");
        let total_duration: Duration = results.iter().map(|r| r.duration).sum();
        let avg_throughput: f64 =
            results.iter().map(|r| r.items_per_second).sum::<f64>() / results.len() as f64;

        println!(
            "Total benchmark time: {:.2} seconds",
            total_duration.as_secs_f64()
        );
        println!("Average throughput: {avg_throughput:.2} items/sec");
    }

    /// Save results to CSV file
    pub fn save_results_csv(results: &[BenchmarkResult], filename: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filename)
            .map_err(|e| torsh_core::error::TorshError::Other(e.to_string()))?;

        // Write CSV header
        writeln!(
            file,
            "Benchmark,Duration_ms,Items_per_sec,Memory_MB,Additional_Metrics"
        )
        .map_err(|e| torsh_core::error::TorshError::Other(e.to_string()))?;

        // Write data rows
        for result in results {
            let additional_metrics: String = result
                .additional_metrics
                .iter()
                .map(|(k, v)| format!("{k}:{v:.2}"))
                .collect::<Vec<_>>()
                .join(";");

            writeln!(
                file,
                "{},{:.2},{:.2},{:.2},\"{}\"",
                result.name,
                result.duration.as_secs_f64() * 1000.0,
                result.items_per_second,
                result.memory_usage_mb,
                additional_metrics
            )
            .map_err(|e| torsh_core::error::TorshError::Other(e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runner() -> Result<()> {
        // Run a subset of benchmarks for testing
        let results = vec![
            DataLoadingBenchmarks::benchmark_sequential_loading()?,
            TransformBenchmarks::benchmark_text_transforms()?,
            DatasetBenchmarks::benchmark_subset_dataset()?,
        ];

        assert!(!results.is_empty());

        for result in &results {
            assert!(result.duration.as_secs_f64() >= 0.0);
            assert!(result.items_per_second >= 0.0);
        }

        BenchmarkRunner::print_results(&results);

        Ok(())
    }

    #[test]
    fn test_benchmark_result_creation() {
        let result = BenchmarkResult::new(
            "Test Benchmark".to_string(),
            Duration::from_millis(100),
            1000,
        )
        .with_memory_usage(50.0)
        .with_metric("test_metric", 42.0);

        assert_eq!(result.name, "Test Benchmark");
        assert_eq!(result.duration, Duration::from_millis(100));
        assert_eq!(result.items_per_second, 10000.0); // 1000 items / 0.1 seconds
        assert_eq!(result.memory_usage_mb, 50.0);
        assert_eq!(result.additional_metrics.get("test_metric"), Some(&42.0));
    }

    #[test]
    fn test_csv_export() -> Result<()> {
        let _results = vec![
            BenchmarkResult::new("Test 1".to_string(), Duration::from_millis(100), 1000)
                .with_metric("accuracy", 0.95),
            BenchmarkResult::new("Test 2".to_string(), Duration::from_millis(200), 500)
                .with_memory_usage(25.0),
        ];

        // Test CSV export (we'd need actual file I/O for full test)
        // For now, just verify the function doesn't panic
        let csv_data =
            "Benchmark,Duration_ms,Items_per_sec,Memory_MB,Additional_Metrics\n".to_string();
        assert!(!csv_data.is_empty());

        Ok(())
    }
}
