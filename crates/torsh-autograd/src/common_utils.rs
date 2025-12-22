// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Common Utilities for Autograd
//!
//! This module provides common utility functions and patterns used across
//! the autograd crate, promoting code reuse and consistency.
//!
//! # Features
//!
//! - **Thread-safe Counters**: Atomic ID generation
//! - **Time Utilities**: Common time-related functions
//! - **Memory Formatting**: Human-readable memory size formatting
//! - **Statistics**: Common statistical calculations
//! - **Collection Utilities**: Helper functions for collections
//! - **String Formatting**: Consistent string formatting

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Thread-safe ID generator
pub struct IdGenerator {
    counter: AtomicU64,
}

impl IdGenerator {
    /// Create a new ID generator (starts at 1)
    pub const fn new() -> Self {
        Self {
            counter: AtomicU64::new(1),
        }
    }

    /// Create a new ID generator with a starting value
    pub const fn with_start(start: u64) -> Self {
        Self {
            counter: AtomicU64::new(start),
        }
    }

    /// Generate next ID
    pub fn next(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Get current value without incrementing
    pub fn current(&self) -> u64 {
        self.counter.load(Ordering::SeqCst)
    }

    /// Reset to 1 (starting value)
    pub fn reset(&self) {
        self.counter.store(1, Ordering::SeqCst);
    }

    /// Reset to specific value
    pub fn reset_to(&self, value: u64) {
        self.counter.store(value, Ordering::SeqCst);
    }
}

impl Default for IdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Format bytes as human-readable string
pub fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

/// Format duration as human-readable string
pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();

    if total_secs < 60 {
        if duration.as_millis() < 1000 {
            format!("{}ms", duration.as_millis())
        } else {
            format!("{:.2}s", duration.as_secs_f64())
        }
    } else if total_secs < 3600 {
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        format!("{}m {}s", mins, secs)
    } else {
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        format!("{}h {}m", hours, mins)
    }
}

/// Format percentage
pub fn format_percentage(value: f64, total: f64) -> String {
    if total == 0.0 {
        "0.00%".to_string()
    } else {
        format!("{:.2}%", (value / total) * 100.0)
    }
}

/// Format number with thousands separator
pub fn format_number(num: usize) -> String {
    let num_str = num.to_string();
    let mut result = String::new();

    for (i, ch) in num_str.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, ch);
    }

    result
}

/// Calculate mean of a slice
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Calculate standard deviation
pub fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean_value = mean(values);
    let variance =
        values.iter().map(|v| (v - mean_value).powi(2)).sum::<f64>() / values.len() as f64;

    variance.sqrt()
}

/// Calculate median
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = sorted.len() / 2;

    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Calculate percentile
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let index = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

/// Calculate min, max, mean, and std dev in one pass
pub fn calculate_statistics(values: &[f64]) -> Statistics {
    if values.is_empty() {
        return Statistics::default();
    }

    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mean_val = mean(values);
    let std_val = std_dev(values);

    Statistics {
        min,
        max,
        mean: mean_val,
        std: std_val,
        count: values.len(),
    }
}

/// Statistical summary
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Statistics {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub count: usize,
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            count: 0,
        }
    }
}

/// Clamp a value between min and max
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Linear interpolation
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Map a value from one range to another
pub fn map_range(value: f64, from_min: f64, from_max: f64, to_min: f64, to_max: f64) -> f64 {
    let normalized = (value - from_min) / (from_max - from_min);
    lerp(to_min, to_max, normalized)
}

/// Check if a float is approximately equal to another
pub fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

/// Truncate string to max length with ellipsis
pub fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let truncated = &s[..max_len.saturating_sub(3)];
        format!("{}...", truncated)
    }
}

/// Generate a unique temporary ID
pub fn generate_temp_id(prefix: &str) -> String {
    use std::time::SystemTime;

    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos();

    format!("{}_{}", prefix, timestamp)
}

/// Round to N decimal places
pub fn round_to(value: f64, decimal_places: u32) -> f64 {
    let multiplier = 10_f64.powi(decimal_places as i32);
    (value * multiplier).round() / multiplier
}

/// Create a progress bar string
pub fn progress_bar(current: usize, total: usize, width: usize) -> String {
    if total == 0 {
        return format!("[{}]", " ".repeat(width));
    }

    let progress = (current as f64 / total as f64).min(1.0);
    let filled = (progress * width as f64).round() as usize;
    let empty = width - filled;

    format!(
        "[{}{}] {}/{}",
        "=".repeat(filled),
        " ".repeat(empty),
        current,
        total
    )
}

/// Moving average calculator
pub struct MovingAverage {
    window_size: usize,
    values: std::collections::VecDeque<f64>,
    sum: f64,
}

impl MovingAverage {
    /// Create a new moving average with given window size
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: std::collections::VecDeque::with_capacity(window_size),
            sum: 0.0,
        }
    }

    /// Add a value and return current average
    pub fn add(&mut self, value: f64) -> f64 {
        if self.values.len() >= self.window_size {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
            }
        }

        self.values.push_back(value);
        self.sum += value;

        self.average()
    }

    /// Get current average
    pub fn average(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }

    /// Reset the moving average
    pub fn reset(&mut self) {
        self.values.clear();
        self.sum = 0.0;
    }

    /// Get number of values currently in window
    pub fn count(&self) -> usize {
        self.values.len()
    }
}

/// Exponential moving average calculator
pub struct ExponentialMovingAverage {
    alpha: f64,
    current: Option<f64>,
}

impl ExponentialMovingAverage {
    /// Create a new EMA with given smoothing factor (0.0 to 1.0)
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            current: None,
        }
    }

    /// Add a value and return current EMA
    pub fn add(&mut self, value: f64) -> f64 {
        match self.current {
            None => {
                self.current = Some(value);
                value
            }
            Some(prev) => {
                let new_value = self.alpha * value + (1.0 - self.alpha) * prev;
                self.current = Some(new_value);
                new_value
            }
        }
    }

    /// Get current EMA value
    pub fn value(&self) -> f64 {
        self.current.unwrap_or(0.0)
    }

    /// Reset the EMA
    pub fn reset(&mut self) {
        self.current = None;
    }
}

/// Rate limiter
pub struct RateLimiter {
    max_count: usize,
    window: Duration,
    timestamps: parking_lot::Mutex<std::collections::VecDeque<std::time::Instant>>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(max_count: usize, window: Duration) -> Self {
        Self {
            max_count,
            window,
            timestamps: parking_lot::Mutex::new(std::collections::VecDeque::new()),
        }
    }

    /// Check if an action is allowed
    pub fn check(&self) -> bool {
        let mut timestamps = self.timestamps.lock();
        let now = std::time::Instant::now();

        // Remove old timestamps
        while let Some(&ts) = timestamps.front() {
            if now.duration_since(ts) > self.window {
                timestamps.pop_front();
            } else {
                break;
            }
        }

        // Check if under limit
        if timestamps.len() < self.max_count {
            timestamps.push_back(now);
            true
        } else {
            false
        }
    }

    /// Reset the rate limiter
    pub fn reset(&self) {
        self.timestamps.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_generator() {
        let gen = IdGenerator::new();
        assert_eq!(gen.next(), 1);
        assert_eq!(gen.next(), 2);
        assert_eq!(gen.next(), 3);
        assert_eq!(gen.current(), 4);

        gen.reset();
        assert_eq!(gen.next(), 1);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert!(format_bytes(1024).contains("KB"));
        assert!(format_bytes(1024 * 1024).contains("MB"));
    }

    #[test]
    fn test_format_duration() {
        assert!(format_duration(Duration::from_millis(500)).contains("ms"));
        assert!(format_duration(Duration::from_secs(30)).contains("s"));
        assert!(format_duration(Duration::from_secs(90)).contains("m"));
    }

    #[test]
    fn test_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(mean(&values), 3.0);
        assert_eq!(median(&values), 3.0);

        let stats = calculate_statistics(&values);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_moving_average() {
        let mut ma = MovingAverage::new(3);

        assert_eq!(ma.add(10.0), 10.0);
        assert_eq!(ma.add(20.0), 15.0);
        assert_eq!(ma.add(30.0), 20.0);
        assert_eq!(ma.add(40.0), 30.0); // Window size reached, oldest value dropped
    }

    #[test]
    fn test_exponential_moving_average() {
        let mut ema = ExponentialMovingAverage::new(0.5);

        assert_eq!(ema.add(10.0), 10.0);
        assert_eq!(ema.add(20.0), 15.0);
        assert_eq!(ema.add(30.0), 22.5);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5, 0, 10), 5);
        assert_eq!(clamp(-5, 0, 10), 0);
        assert_eq!(clamp(15, 0, 10), 10);
    }

    #[test]
    fn test_lerp() {
        assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("hello", 10), "hello");
        assert_eq!(truncate_string("hello world", 8), "hello...");
    }

    #[test]
    fn test_progress_bar() {
        let bar = progress_bar(50, 100, 10);
        assert!(bar.contains("====="));
        assert!(bar.contains("50/100"));
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(3, Duration::from_secs(1));

        assert!(limiter.check());
        assert!(limiter.check());
        assert!(limiter.check());
        assert!(!limiter.check()); // Exceeded limit

        limiter.reset();
        assert!(limiter.check()); // Works after reset
    }
}
