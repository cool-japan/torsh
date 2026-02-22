//! Dataset Operations Benchmarks
//!
//! Comprehensive benchmarks for dataset operations in torsh-text:
//! - Dataset loading (CSV, JSON, text formats)
//! - Data preprocessing (tokenization, filtering, normalization)
//! - Data augmentation (random deletion, swap, synonym replacement)
//! - Batch iteration and parallel loading
//! - Memory-efficient streaming and lazy loading

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::io::Write;

// ============================================================================
// Dataset Loading Benchmarks
// ============================================================================

fn bench_csv_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_csv_loading");

    // Test different dataset sizes
    let sizes = [
        100,     // Small
        1_000,   // Medium
        10_000,  // Large
        100_000, // Very large
    ];

    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Create temporary CSV file
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!("bench_dataset_{}.csv", size));

        // Generate CSV data
        {
            let mut file = std::fs::File::create(&file_path).expect("Failed to create temp CSV");
            writeln!(file, "text,label").expect("Failed to write header");
            for i in 0..*size {
                writeln!(file, "This is sample text number {}, label_{}", i, i % 10)
                    .expect("Failed to write row");
            }
        }

        group.bench_with_input(BenchmarkId::new("load_csv", size), size, |bench, _| {
            bench.iter(|| {
                // In actual benchmark, would call ClassificationDataset::from_csv
                let content = std::fs::read_to_string(&file_path).expect("Failed to read CSV");
                let _lines: Vec<_> = content.lines().collect();
            });
        });

        // Cleanup
        std::fs::remove_file(&file_path).ok();
    }

    group.finish();
}

fn bench_text_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_text_loading");

    for num_lines in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*num_lines as u64));

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!("bench_text_{}.txt", num_lines));

        // Generate text file
        {
            let mut file =
                std::fs::File::create(&file_path).expect("Failed to create temp text file");
            for i in 0..*num_lines {
                writeln!(file, "This is line number {} with some content", i)
                    .expect("Failed to write line");
            }
        }

        group.bench_with_input(
            BenchmarkId::new("load_text", num_lines),
            num_lines,
            |bench, _| {
                bench.iter(|| {
                    let content = std::fs::read_to_string(&file_path).expect("Failed to read text");
                    let _lines: Vec<_> = content.lines().collect();
                });
            },
        );

        std::fs::remove_file(&file_path).ok();
    }

    group.finish();
}

// ============================================================================
// Preprocessing Benchmarks
// ============================================================================

fn bench_text_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing");

    let text_lengths = [
        (100, "short"),      // 100 words
        (500, "medium"),     // 500 words
        (1000, "long"),      // 1000 words
        (5000, "very_long"), // 5000 words
    ];

    for (word_count, label) in text_lengths.iter() {
        let text = (0..*word_count)
            .map(|i| format!("word{}", i))
            .collect::<Vec<_>>()
            .join(" ");

        group.throughput(Throughput::Elements(*word_count as u64));

        group.bench_with_input(
            BenchmarkId::new("lowercase", label),
            &text,
            |bench, text| {
                bench.iter(|| {
                    let _lower = text.to_lowercase();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("remove_punctuation", label),
            &text,
            |bench, text| {
                bench.iter(|| {
                    let _clean: String =
                        text.chars().filter(|c| !c.is_ascii_punctuation()).collect();
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("tokenize", label), &text, |bench, text| {
            bench.iter(|| {
                let _tokens: Vec<_> = text.split_whitespace().collect();
            });
        });

        group.bench_with_input(
            BenchmarkId::new("normalize_whitespace", label),
            &text,
            |bench, text| {
                bench.iter(|| {
                    let _normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
                });
            },
        );
    }

    group.finish();
}

fn bench_stopword_removal(c: &mut Criterion) {
    let mut group = c.benchmark_group("stopword_removal");

    // Common English stopwords
    let stopwords: Vec<String> = vec![
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
        "it", "its", "of", "on", "that", "the", "to", "was", "will", "with",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();

    for num_words in [100, 500, 1000, 5000].iter() {
        let text = (0..*num_words)
            .map(|i| {
                if i % 3 == 0 {
                    "the" // Stopword
                } else {
                    "important"
                }
            })
            .collect::<Vec<_>>()
            .join(" ");

        group.throughput(Throughput::Elements(*num_words as u64));

        group.bench_with_input(
            BenchmarkId::new("remove", num_words),
            &text,
            |bench, text| {
                bench.iter(|| {
                    let words: Vec<&str> = text.split_whitespace().collect();
                    let _filtered: Vec<&str> = words
                        .into_iter()
                        .filter(|w| !stopwords.contains(&w.to_lowercase()))
                        .collect();
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Data Augmentation Benchmarks
// ============================================================================

fn bench_random_deletion(c: &mut Criterion) {
    let mut group = c.benchmark_group("augmentation_random_deletion");

    for num_words in [50, 100, 200, 500].iter() {
        let text = (0..*num_words)
            .map(|i| format!("word{}", i))
            .collect::<Vec<_>>()
            .join(" ");

        group.throughput(Throughput::Elements(*num_words as u64));

        group.bench_with_input(
            BenchmarkId::new("delete_10pct", num_words),
            &text,
            |bench, text| {
                bench.iter(|| {
                    // Simulate 10% word deletion
                    use scirs2_core::random::thread_rng;
                    let mut rng = thread_rng();
                    let words: Vec<&str> = text.split_whitespace().collect();
                    let _kept: Vec<&str> = words
                        .into_iter()
                        .filter(|_| rng.random::<f32>() > 0.1)
                        .collect();
                });
            },
        );
    }

    group.finish();
}

fn bench_random_swap(c: &mut Criterion) {
    let mut group = c.benchmark_group("augmentation_random_swap");

    for num_words in [50, 100, 200, 500].iter() {
        let text = (0..*num_words)
            .map(|i| format!("word{}", i))
            .collect::<Vec<_>>()
            .join(" ");

        group.throughput(Throughput::Elements(*num_words as u64));

        group.bench_with_input(
            BenchmarkId::new("swap_5_times", num_words),
            &text,
            |bench, text| {
                bench.iter(|| {
                    use scirs2_core::random::thread_rng;
                    let mut rng = thread_rng();
                    let mut words: Vec<String> =
                        text.split_whitespace().map(|s| s.to_string()).collect();

                    for _ in 0..5 {
                        if words.len() >= 2 {
                            let idx1 = rng.random::<u64>() as usize % words.len();
                            let idx2 = rng.random::<u64>() as usize % words.len();
                            words.swap(idx1, idx2);
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Batch Iteration Benchmarks
// ============================================================================

fn bench_batch_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_iteration");

    let dataset_sizes = [1_000, 10_000, 100_000];
    let batch_sizes = [16, 32, 64, 128];

    for &dataset_size in &dataset_sizes {
        for &batch_size in &batch_sizes {
            let _num_batches = (dataset_size + batch_size - 1) / batch_size;
            group.throughput(Throughput::Elements(dataset_size as u64));

            // Create mock dataset
            let data: Vec<String> = (0..dataset_size).map(|i| format!("item_{}", i)).collect();

            group.bench_with_input(
                BenchmarkId::new("iterate", format!("ds{}_bs{}", dataset_size, batch_size)),
                &(&data, batch_size),
                |bench, &(data, batch_size)| {
                    bench.iter(|| {
                        let mut batches = Vec::new();
                        for chunk in data.chunks(batch_size) {
                            batches.push(chunk.to_vec());
                        }
                        batches
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Parallel Loading Benchmarks
// ============================================================================

fn bench_parallel_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_loading");

    // Create multiple temporary files
    let temp_dir = std::env::temp_dir();
    let num_files = 8;
    let rows_per_file = 1000;

    let file_paths: Vec<_> = (0..num_files)
        .map(|i| {
            let path = temp_dir.join(format!("parallel_bench_{}.csv", i));
            let mut file = std::fs::File::create(&path).expect("Failed to create temp file");
            writeln!(file, "text,label").expect("Failed to write header");
            for j in 0..rows_per_file {
                writeln!(file, "text_{},{}", j, j % 10).expect("Failed to write row");
            }
            path
        })
        .collect();

    group.throughput(Throughput::Elements((num_files * rows_per_file) as u64));

    group.bench_function("parallel_load_csv", |bench| {
        bench.iter(|| {
            use scirs2_core::parallel_ops::*;
            let _results: Vec<_> = file_paths
                .par_iter()
                .map(|path| {
                    let content = std::fs::read_to_string(path).expect("Failed to read file");
                    content.lines().count()
                })
                .collect();
        });
    });

    group.bench_function("sequential_load_csv", |bench| {
        bench.iter(|| {
            let _results: Vec<_> = file_paths
                .iter()
                .map(|path| {
                    let content = std::fs::read_to_string(path).expect("Failed to read file");
                    content.lines().count()
                })
                .collect();
        });
    });

    // Cleanup
    for path in &file_paths {
        std::fs::remove_file(path).ok();
    }

    group.finish();
}

// ============================================================================
// Streaming Dataset Benchmarks
// ============================================================================

fn bench_streaming_dataset(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_dataset");

    for file_size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*file_size as u64));

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!("streaming_bench_{}.txt", file_size));

        // Generate large file
        {
            let mut file = std::fs::File::create(&file_path).expect("Failed to create temp file");
            for i in 0..*file_size {
                writeln!(file, "Line number {} with some content", i)
                    .expect("Failed to write line");
            }
        }

        group.bench_with_input(
            BenchmarkId::new("buffered_read", file_size),
            file_size,
            |bench, _| {
                bench.iter(|| {
                    use std::io::{BufRead, BufReader};
                    let file = std::fs::File::open(&file_path).expect("Failed to open file");
                    let reader = BufReader::new(file);
                    let _lines: Vec<_> = reader.lines().collect();
                });
            },
        );

        std::fs::remove_file(&file_path).ok();
    }

    group.finish();
}

// ============================================================================
// Dataset Filtering Benchmarks
// ============================================================================

fn bench_dataset_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_filtering");

    for dataset_size in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*dataset_size as u64));

        let texts: Vec<String> = (0..*dataset_size)
            .map(|i| {
                let length = 10 + (i % 100);
                "word ".repeat(length as usize)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("filter_by_length", dataset_size),
            &texts,
            |bench, texts| {
                bench.iter(|| {
                    let _filtered: Vec<_> = texts
                        .iter()
                        .filter(|text| {
                            let word_count = text.split_whitespace().count();
                            word_count >= 20 && word_count <= 50
                        })
                        .collect();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("deduplicate", dataset_size),
            &texts,
            |bench, texts| {
                bench.iter(|| {
                    use std::collections::HashSet;
                    let mut seen = HashSet::new();
                    let _unique: Vec<_> = texts.iter().filter(|text| seen.insert(*text)).collect();
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Dataset Shuffling Benchmarks
// ============================================================================

fn bench_dataset_shuffling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_shuffling");

    for size in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<usize> = (0..*size).collect();

        group.bench_with_input(
            BenchmarkId::new("fisher_yates_shuffle", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    use scirs2_core::rand_prelude::SliceRandom;
                    use scirs2_core::random::thread_rng;
                    let mut rng = thread_rng();
                    let mut data_copy = data.clone();
                    data_copy.shuffle(&mut rng);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Memory Efficiency Benchmarks
// ============================================================================

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    for size in [10_000, 100_000, 1_000_000].iter() {
        let bytes = size * std::mem::size_of::<String>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("eager_loading", size),
            size,
            |bench, &size| {
                bench.iter(|| {
                    // Load all data into memory
                    let _data: Vec<String> = (0..size).map(|i| format!("item_{}", i)).collect();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("iterator_lazy", size),
            size,
            |bench, &size| {
                bench.iter(|| {
                    // Process lazily with iterator
                    let _count = (0..size)
                        .map(|i| format!("item_{}", i))
                        .take(1000) // Only process first 1000
                        .count();
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    dataset_benches,
    bench_csv_loading,
    bench_text_loading,
    bench_text_preprocessing,
    bench_stopword_removal,
    bench_random_deletion,
    bench_random_swap,
    bench_batch_iteration,
    bench_parallel_loading,
    bench_streaming_dataset,
    bench_dataset_filtering,
    bench_dataset_shuffling,
    bench_memory_efficiency,
);

criterion_main!(dataset_benches);
