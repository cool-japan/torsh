//! Common types for model operations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model information structure
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub format: String,
    pub size: String,
    pub parameters: u64,
    pub layers: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub precision: String,
    pub device: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Result structure for model operations
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelResult {
    pub operation: String,
    pub input_model: String,
    pub output_model: Option<String>,
    pub success: bool,
    pub duration: String,
    pub size_before: Option<String>,
    pub size_after: Option<String>,
    pub metrics: HashMap<String, serde_json::Value>,
    pub errors: Vec<String>,
}

/// Results from model timing benchmark
#[derive(Debug, Serialize)]
pub struct TimingResult {
    pub throughput_fps: f64,
    pub latency_ms: f64,
    pub memory_mb: f64,
    pub warmup_time_ms: f64,
    pub avg_inference_time_ms: f64,
    pub min_inference_time_ms: f64,
    pub max_inference_time_ms: f64,
    pub std_dev_ms: f64,
    pub device_utilization: Option<f64>,
}

/// Format bytes into human-readable format
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}
