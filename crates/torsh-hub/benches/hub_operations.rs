//! Performance benchmarks for critical torsh-hub operations
//!
//! This benchmark suite measures the performance of key operations including:
//! - Version comparison and parsing
//! - Model info serialization
//! - Download configuration creation
//! - URL validation and parsing
//! - Cache operations
//! - Utility functions

use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use torsh_hub::download::validate_url;
use torsh_hub::model_info::*;
use torsh_hub::utils::*;

/// Benchmark version operations
fn bench_version_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("version_operations");

    let v1 = Version::new(1, 2, 3);
    let v2 = Version::new(1, 2, 4);
    let v3 = Version::new(2, 0, 0);

    // Benchmark version comparison
    group.bench_function("version_compare", |b| {
        b.iter(|| {
            let _ = black_box(&v1) < black_box(&v2);
            let _ = black_box(&v2) < black_box(&v3);
            let _ = black_box(&v1) == black_box(&v1);
        });
    });

    // Benchmark version creation
    group.bench_function("version_creation", |b| {
        b.iter(|| {
            let _v = Version::new(black_box(1), black_box(2), black_box(3));
        });
    });

    // Benchmark version to string
    group.bench_function("version_to_string", |b| {
        b.iter(|| {
            let _s = black_box(&v1).to_string();
        });
    });

    group.finish();
}

/// Benchmark model info operations
fn bench_model_info_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_info");

    let model_info = ModelInfo {
        name: "test-model".to_string(),
        description: "A test model for benchmarking performance".to_string(),
        author: "TestAuthor".to_string(),
        version: Version::new(1, 0, 0),
        license: "MIT".to_string(),
        tags: vec![
            "test".to_string(),
            "benchmark".to_string(),
            "performance".to_string(),
        ],
        datasets: vec!["imagenet".to_string(), "coco".to_string()],
        metrics: {
            let mut m = HashMap::new();
            m.insert("accuracy".to_string(), MetricValue::Float(0.95));
            m.insert("f1_score".to_string(), MetricValue::Float(0.93));
            m.insert("latency_ms".to_string(), MetricValue::Float(12.5));
            m
        },
        requirements: Requirements {
            torsh_version: "0.1.0-alpha.2".to_string(),
            dependencies: vec!["scirs2-core".to_string(), "scirs2-neural".to_string()],
            hardware: HardwareRequirements {
                min_gpu_memory_gb: Some(4.0),
                recommended_gpu_memory_gb: Some(8.0),
                min_ram_gb: Some(8.0),
                recommended_ram_gb: Some(16.0),
            },
        },
        files: vec![
            FileInfo {
                path: "model.safetensors".to_string(),
                size_bytes: 1024 * 1024 * 100, // 100 MB
                sha256: "abcdef1234567890".to_string(),
                description: Some("Main model weights".to_string()),
            },
            FileInfo {
                path: "config.json".to_string(),
                size_bytes: 2048,
                sha256: "1234567890abcdef".to_string(),
                description: Some("Model configuration".to_string()),
            },
        ],
        model_card: None,
        version_history: None,
    };

    // Benchmark model info serialization to JSON
    group.bench_function("serialize_to_json", |b| {
        b.iter(|| {
            let _json = serde_json::to_string(black_box(&model_info)).unwrap();
        });
    });

    // Benchmark model info serialization to JSON (pretty)
    group.bench_function("serialize_to_json_pretty", |b| {
        b.iter(|| {
            let _json = serde_json::to_string_pretty(black_box(&model_info)).unwrap();
        });
    });

    // Benchmark validation checks
    group.bench_function("validate_fields", |b| {
        b.iter(|| {
            let _valid = !black_box(&model_info).name.is_empty()
                && !black_box(&model_info).files.is_empty()
                && !black_box(&model_info).tags.is_empty();
        });
    });

    group.finish();
}

/// Benchmark hardware requirements
fn bench_hardware_requirements(c: &mut Criterion) {
    let mut group = c.benchmark_group("hardware_requirements");

    let hw_reqs = HardwareRequirements {
        min_gpu_memory_gb: Some(4.0),
        recommended_gpu_memory_gb: Some(8.0),
        min_ram_gb: Some(8.0),
        recommended_ram_gb: Some(16.0),
    };

    // Benchmark serialization
    group.bench_function("serialize", |b| {
        b.iter(|| {
            let _json = serde_json::to_string(black_box(&hw_reqs)).unwrap();
        });
    });

    // Benchmark GPU memory check
    group.bench_function("check_gpu_requirements", |b| {
        b.iter(|| {
            let available_gpu_gb = black_box(6.0);
            let _meets_min = hw_reqs
                .min_gpu_memory_gb
                .map_or(true, |min| available_gpu_gb >= min);
            let _meets_rec = hw_reqs
                .recommended_gpu_memory_gb
                .map_or(true, |rec| available_gpu_gb >= rec);
        });
    });

    group.finish();
}

/// Benchmark model details operations
fn bench_model_details(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_details");

    let model_details = ModelDetails {
        developed_by: "ToRSh Team".to_string(),
        model_date: "2025-01-01".to_string(),
        model_type: "Vision Transformer".to_string(),
        architecture: "ViT-Base".to_string(),
        paper_url: Some("https://arxiv.org/abs/2010.11929".to_string()),
        citation: Some("@article{dosovitskiy2020vit}".to_string()),
    };

    // Benchmark serialization
    group.bench_function("serialize", |b| {
        b.iter(|| {
            let _json = serde_json::to_string(black_box(&model_details)).unwrap();
        });
    });

    // Benchmark field access
    group.bench_function("field_access", |b| {
        b.iter(|| {
            let _dev = black_box(&model_details.developed_by);
            let _date = black_box(&model_details.model_date);
            let _type = black_box(&model_details.model_type);
        });
    });

    group.finish();
}

/// Benchmark file info operations
fn bench_file_info(c: &mut Criterion) {
    let mut group = c.benchmark_group("file_info");

    let file_info = FileInfo {
        path: "model.safetensors".to_string(),
        size_bytes: 1024 * 1024 * 500, // 500 MB
        sha256: "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890".to_string(),
        description: Some("Main model weights in SafeTensors format".to_string()),
    };

    // Benchmark serialization
    group.bench_function("serialize", |b| {
        b.iter(|| {
            let _json = serde_json::to_string(black_box(&file_info)).unwrap();
        });
    });

    // Benchmark size formatting
    group.bench_function("format_size", |b| {
        b.iter(|| {
            let size_mb = black_box(file_info.size_bytes) as f64 / (1024.0 * 1024.0);
            let _formatted = format!("{:.2} MB", size_mb);
        });
    });

    group.finish();
}

/// Benchmark requirements operations
fn bench_requirements(c: &mut Criterion) {
    let mut group = c.benchmark_group("requirements");

    let requirements = Requirements {
        torsh_version: "0.1.0-alpha.2".to_string(),
        dependencies: vec![
            "scirs2-core@0.1.0-rc.2".to_string(),
            "scirs2-neural@0.1.0-rc.2".to_string(),
            "numrs2@0.1.0-beta.3".to_string(),
        ],
        hardware: HardwareRequirements {
            min_gpu_memory_gb: Some(8.0),
            recommended_gpu_memory_gb: Some(16.0),
            min_ram_gb: Some(16.0),
            recommended_ram_gb: Some(32.0),
        },
    };

    // Benchmark serialization
    group.bench_function("serialize", |b| {
        b.iter(|| {
            let _json = serde_json::to_string(black_box(&requirements)).unwrap();
        });
    });

    // Benchmark dependency check
    group.bench_function("check_dependencies", |b| {
        b.iter(|| {
            let _count = black_box(&requirements).dependencies.len();
            let _has_scirs2 = black_box(&requirements)
                .dependencies
                .iter()
                .any(|d| d.starts_with("scirs2"));
        });
    });

    group.finish();
}

/// Benchmark utility functions
fn bench_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("utils");

    // Benchmark format_size
    group.bench_function("format_size_small", |b| {
        b.iter(|| {
            let _s = format_size(black_box(1024)); // 1 KB
        });
    });

    group.bench_function("format_size_large", |b| {
        b.iter(|| {
            let _s = format_size(black_box(1024 * 1024 * 1024)); // 1 GB
        });
    });

    // Benchmark sanitize_model_name
    group.bench_function("sanitize_model_name_simple", |b| {
        b.iter(|| {
            let _s = sanitize_model_name(black_box("simple-model"));
        });
    });

    group.bench_function("sanitize_model_name_complex", |b| {
        b.iter(|| {
            let _s = sanitize_model_name(black_box("BERT (base) - Uncased v2.0"));
        });
    });

    // Benchmark is_safe_path
    group.bench_function("is_safe_path_safe", |b| {
        b.iter(|| {
            let _safe = is_safe_path(black_box(std::path::Path::new("models/bert/model.onnx")));
        });
    });

    group.bench_function("is_safe_path_unsafe", |b| {
        b.iter(|| {
            let _safe = is_safe_path(black_box(std::path::Path::new("../../../etc/passwd")));
        });
    });

    // Benchmark parse_repo_string
    group.bench_function("parse_repo_string_simple", |b| {
        b.iter(|| {
            let _ = parse_repo_string(black_box("huggingface/bert-base"));
        });
    });

    group.bench_function("parse_repo_string_with_tag", |b| {
        b.iter(|| {
            let _ = parse_repo_string(black_box("openai/gpt2:v1.0"));
        });
    });

    // Benchmark extract_extension
    group.bench_function("extract_extension", |b| {
        b.iter(|| {
            let _ext = extract_extension(black_box("model.safetensors"));
        });
    });

    // Benchmark is_supported_model_format
    group.bench_function("is_supported_format_onnx", |b| {
        b.iter(|| {
            let _supported = is_supported_model_format(black_box("onnx"));
        });
    });

    group.bench_function("is_supported_format_unknown", |b| {
        b.iter(|| {
            let _supported = is_supported_model_format(black_box("txt"));
        });
    });

    // Benchmark estimate_parameters_from_size
    group.bench_function("estimate_parameters", |b| {
        b.iter(|| {
            let _params = estimate_parameters_from_size(black_box(1024 * 1024 * 500));
        });
    });

    // Benchmark format_parameter_count
    group.bench_function("format_parameter_count_millions", |b| {
        b.iter(|| {
            let _s = format_parameter_count(black_box(125_000_000));
        });
    });

    group.bench_function("format_parameter_count_billions", |b| {
        b.iter(|| {
            let _s = format_parameter_count(black_box(7_000_000_000));
        });
    });

    // Benchmark validate_semver
    group.bench_function("validate_semver_valid", |b| {
        b.iter(|| {
            let _ = validate_semver(black_box("1.2.3"));
        });
    });

    group.bench_function("validate_semver_with_prerelease", |b| {
        b.iter(|| {
            let _ = validate_semver(black_box("2.0.0-alpha.1"));
        });
    });

    // Benchmark compare_versions
    group.bench_function("compare_versions", |b| {
        b.iter(|| {
            let _ = compare_versions(black_box("1.2.3"), black_box("2.0.0"));
        });
    });

    group.finish();
}

/// Benchmark URL validation
fn bench_url_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("url_validation");

    // Benchmark valid URL
    group.bench_function("validate_url_valid_http", |b| {
        b.iter(|| {
            let _ = validate_url(black_box("http://example.com/model.onnx"));
        });
    });

    group.bench_function("validate_url_valid_https", |b| {
        b.iter(|| {
            let _ = validate_url(black_box("https://example.com/model.safetensors"));
        });
    });

    // Benchmark invalid URLs
    group.bench_function("validate_url_empty", |b| {
        b.iter(|| {
            let _ = validate_url(black_box(""));
        });
    });

    group.bench_function("validate_url_invalid_protocol", |b| {
        b.iter(|| {
            let _ = validate_url(black_box("file:///local/path"));
        });
    });

    group.bench_function("validate_url_with_spaces", |b| {
        b.iter(|| {
            let _ = validate_url(black_box("https://example.com/model name.onnx"));
        });
    });

    group.finish();
}

/// Benchmark string operations
fn bench_string_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_operations");

    let model_name = "bert-base-uncased";
    let long_name = "very-long-model-name-with-many-hyphens-and-components";

    // Benchmark string allocation
    group.bench_function("clone_short_string", |b| {
        b.iter(|| {
            let _s = black_box(model_name).to_string();
        });
    });

    group.bench_function("clone_long_string", |b| {
        b.iter(|| {
            let _s = black_box(long_name).to_string();
        });
    });

    // Benchmark string splitting
    group.bench_function("split_short_string", |b| {
        b.iter(|| {
            let parts: Vec<&str> = black_box(model_name).split('-').collect();
            let _ = black_box(parts);
        });
    });

    group.bench_function("split_long_string", |b| {
        b.iter(|| {
            let parts: Vec<&str> = black_box(long_name).split('-').collect();
            let _ = black_box(parts);
        });
    });

    // Benchmark string transformation
    group.bench_function("to_lowercase", |b| {
        b.iter(|| {
            let _s = black_box("BERT-BASE-UNCASED").to_lowercase();
        });
    });

    group.bench_function("to_uppercase", |b| {
        b.iter(|| {
            let _s = black_box("bert-base-uncased").to_uppercase();
        });
    });

    group.finish();
}

/// Benchmark hash map operations
fn bench_hashmap_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashmap_operations");

    // Create test data
    let mut small_map = HashMap::new();
    for i in 0..10 {
        small_map.insert(format!("key_{}", i), format!("value_{}", i));
    }

    let mut large_map = HashMap::new();
    for i in 0..1000 {
        large_map.insert(format!("key_{}", i), format!("value_{}", i));
    }

    // Benchmark lookups
    group.bench_function("lookup_small_map", |b| {
        b.iter(|| {
            let _val = black_box(&small_map).get("key_5");
        });
    });

    group.bench_function("lookup_large_map", |b| {
        b.iter(|| {
            let _val = black_box(&large_map).get("key_500");
        });
    });

    // Benchmark insertion
    group.bench_function("insert_small_map", |b| {
        b.iter(|| {
            let mut map = small_map.clone();
            map.insert("new_key".to_string(), "new_value".to_string());
            black_box(map);
        });
    });

    // Benchmark iteration
    group.bench_function("iterate_small_map", |b| {
        b.iter(|| {
            let count = black_box(&small_map).iter().count();
            black_box(count);
        });
    });

    group.bench_function("iterate_large_map", |b| {
        b.iter(|| {
            let count = black_box(&large_map).iter().count();
            black_box(count);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_version_operations,
    bench_model_info_operations,
    bench_hardware_requirements,
    bench_model_details,
    bench_file_info,
    bench_requirements,
    bench_utils,
    bench_url_validation,
    bench_string_operations,
    bench_hashmap_operations
);
criterion_main!(benches);
