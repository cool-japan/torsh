// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gradient Computation Progress Reporting
//!
//! This module provides comprehensive progress reporting for gradient computations,
//! allowing users to track long-running operations and get estimates of remaining time.
//!
//! # Features
//!
//! - **Real-time Progress Tracking**: Track completion percentage of gradient computation
//! - **Time Estimation**: Estimate remaining time based on historical data
//! - **Cancellation Support**: Allow users to cancel long-running computations
//! - **Callbacks**: Register callbacks for progress updates
//! - **Hierarchical Progress**: Track progress of nested operations
//! - **Statistics**: Detailed statistics about computation progress
//!
//! # Example
//!
//! ```rust,ignore
//! use torsh_autograd::progress_reporting::{ProgressReporter, ProgressConfig};
//!
//! // Create progress reporter
//! let reporter = ProgressReporter::new(ProgressConfig::default());
//!
//! // Start tracking
//! reporter.start("backward_pass", 100);
//!
//! // Update progress
//! for i in 0..100 {
//!     reporter.update(1);
//!     // ... do work ...
//! }
//!
//! // Finish
//! reporter.finish();
//! ```

use crate::error_handling::{AutogradError, AutogradResult};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Progress update callback
pub type ProgressCallback = Arc<dyn Fn(&ProgressUpdate) + Send + Sync>;

/// Progress configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressConfig {
    /// Enable progress reporting
    pub enabled: bool,

    /// Update interval (minimum time between callbacks)
    pub update_interval_ms: u64,

    /// Enable time estimation
    pub enable_time_estimation: bool,

    /// Number of samples for time estimation
    pub estimation_window_size: usize,

    /// Enable hierarchical progress tracking
    pub enable_hierarchical: bool,

    /// Maximum number of progress entries to keep in history
    pub max_history_size: usize,

    /// Print progress to console
    pub print_to_console: bool,

    /// Progress bar width (characters)
    pub progress_bar_width: usize,
}

impl Default for ProgressConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_interval_ms: 100,
            enable_time_estimation: true,
            estimation_window_size: 10,
            enable_hierarchical: true,
            max_history_size: 1000,
            print_to_console: false,
            progress_bar_width: 50,
        }
    }
}

/// Progress update information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressUpdate {
    /// Operation name
    pub operation_name: String,

    /// Current progress (completed items)
    pub current: u64,

    /// Total items to process
    pub total: u64,

    /// Progress percentage (0.0 to 100.0)
    pub percentage: f64,

    /// Elapsed time
    pub elapsed: Duration,

    /// Estimated remaining time (if available)
    pub estimated_remaining: Option<Duration>,

    /// Current rate (items per second)
    pub rate: f64,

    /// Whether operation is complete
    pub is_complete: bool,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ProgressUpdate {
    /// Format as progress bar
    pub fn format_progress_bar(&self, width: usize) -> String {
        let filled = ((self.percentage / 100.0) * width as f64) as usize;
        let empty = width.saturating_sub(filled);

        let bar = format!("[{}{}]", "=".repeat(filled), " ".repeat(empty));

        let time_str = if let Some(remaining) = self.estimated_remaining {
            format!(" ETA: {:.1}s", remaining.as_secs_f64())
        } else {
            String::new()
        };

        format!(
            "{} {:>6.2}% {}/{} @ {:.1} items/s{}",
            bar, self.percentage, self.current, self.total, self.rate, time_str
        )
    }
}

/// Progress statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressStatistics {
    /// Total number of operations tracked
    pub total_operations: usize,

    /// Total items processed
    pub total_items_processed: u64,

    /// Total elapsed time
    pub total_elapsed: Duration,

    /// Average rate (items per second)
    pub average_rate: f64,

    /// Peak rate
    pub peak_rate: f64,

    /// Number of cancelled operations
    pub cancelled_count: usize,

    /// Number of completed operations
    pub completed_count: usize,
}

/// Progress entry for hierarchical tracking
#[derive(Debug)]
struct ProgressEntry {
    /// Operation name
    name: String,

    /// Start time
    start_time: Instant,

    /// Last update time
    last_update_time: Instant,

    /// Current progress
    current: AtomicU64,

    /// Total items
    total: u64,

    /// Recent rates (for time estimation)
    recent_rates: Arc<Mutex<VecDeque<f64>>>,

    /// Whether operation was cancelled
    cancelled: AtomicBool,

    /// Metadata
    metadata: Arc<RwLock<HashMap<String, String>>>,

    /// Parent entry (for hierarchical tracking)
    #[allow(dead_code)]
    parent: Option<Arc<ProgressEntry>>,
}

impl ProgressEntry {
    fn new(name: String, total: u64, parent: Option<Arc<ProgressEntry>>) -> Self {
        Self {
            name,
            start_time: Instant::now(),
            last_update_time: Instant::now(),
            current: AtomicU64::new(0),
            total,
            recent_rates: Arc::new(Mutex::new(VecDeque::new())),
            cancelled: AtomicBool::new(false),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            parent,
        }
    }

    fn get_update(&self) -> ProgressUpdate {
        let current = self.current.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed();
        let percentage = if self.total > 0 {
            (current as f64 / self.total as f64) * 100.0
        } else {
            0.0
        };

        let rate = if elapsed.as_secs_f64() > 0.0 {
            current as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        // Estimate remaining time based on recent rates
        let estimated_remaining = if self.total > current && rate > 0.0 {
            let rates = self.recent_rates.lock();
            let avg_rate = if !rates.is_empty() {
                rates.iter().sum::<f64>() / rates.len() as f64
            } else {
                rate
            };

            if avg_rate > 0.0 {
                let remaining_items = self.total - current;
                let remaining_seconds = remaining_items as f64 / avg_rate;
                Some(Duration::from_secs_f64(remaining_seconds))
            } else {
                None
            }
        } else {
            None
        };

        ProgressUpdate {
            operation_name: self.name.clone(),
            current,
            total: self.total,
            percentage,
            elapsed,
            estimated_remaining,
            rate,
            is_complete: current >= self.total,
            metadata: self.metadata.read().clone(),
        }
    }

    fn update_rate(&self, rate: f64, window_size: usize) {
        let mut rates = self.recent_rates.lock();
        rates.push_back(rate);
        if rates.len() > window_size {
            rates.pop_front();
        }
    }
}

/// Progress reporter for gradient computation
#[derive(Clone)]
pub struct ProgressReporter {
    config: Arc<RwLock<ProgressConfig>>,
    current_entry: Arc<RwLock<Option<Arc<ProgressEntry>>>>,
    entry_stack: Arc<Mutex<Vec<Arc<ProgressEntry>>>>,
    callbacks: Arc<RwLock<Vec<ProgressCallback>>>,
    history: Arc<Mutex<VecDeque<ProgressUpdate>>>,
    statistics: Arc<RwLock<ProgressStatistics>>,
}

impl ProgressReporter {
    /// Create a new progress reporter
    pub fn new(config: ProgressConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            current_entry: Arc::new(RwLock::new(None)),
            entry_stack: Arc::new(Mutex::new(Vec::new())),
            callbacks: Arc::new(RwLock::new(Vec::new())),
            history: Arc::new(Mutex::new(VecDeque::new())),
            statistics: Arc::new(RwLock::new(ProgressStatistics {
                total_operations: 0,
                total_items_processed: 0,
                total_elapsed: Duration::ZERO,
                average_rate: 0.0,
                peak_rate: 0.0,
                cancelled_count: 0,
                completed_count: 0,
            })),
        }
    }

    /// Start tracking a new operation
    pub fn start(&self, name: impl Into<String>, total: u64) {
        let config = self.config.read();
        if !config.enabled {
            return;
        }

        let parent = if config.enable_hierarchical {
            self.current_entry.read().clone()
        } else {
            None
        };

        let entry = Arc::new(ProgressEntry::new(name.into(), total, parent));

        // Push to stack if hierarchical
        if config.enable_hierarchical {
            self.entry_stack.lock().push(entry.clone());
        }

        *self.current_entry.write() = Some(entry.clone());

        // Update statistics
        {
            let mut stats = self.statistics.write();
            stats.total_operations += 1;
        }

        // Trigger initial update
        self.trigger_update(&entry);
    }

    /// Update progress
    pub fn update(&self, delta: u64) {
        let config = self.config.read();
        if !config.enabled {
            return;
        }

        // Check if we should auto-finish (need to check before acquiring lock)
        let should_finish = {
            if let Some(entry) = self.current_entry.read().as_ref() {
                let old_current = entry.current.fetch_add(delta, Ordering::Relaxed);
                let new_current = old_current + delta;

                // Update rate
                let elapsed_since_last = entry.last_update_time.elapsed();
                if elapsed_since_last.as_secs_f64() > 0.0 {
                    let rate = delta as f64 / elapsed_since_last.as_secs_f64();
                    entry.update_rate(rate, config.estimation_window_size);
                }

                // Check if we should trigger update callback
                if elapsed_since_last.as_millis() >= config.update_interval_ms as u128 {
                    self.trigger_update(entry);
                }

                // Update statistics
                {
                    let mut stats = self.statistics.write();
                    stats.total_items_processed += delta;
                    let update = entry.get_update();
                    if update.rate > stats.peak_rate {
                        stats.peak_rate = update.rate;
                    }
                }

                // Check if complete (don't call finish() while holding read lock!)
                new_current >= entry.total
            } else {
                false
            }
        }; // Read lock is released here

        // Auto-finish if complete (now safe to acquire write lock)
        if should_finish {
            self.finish();
        }
    }

    /// Set progress to specific value
    pub fn set(&self, value: u64) {
        if let Some(entry) = self.current_entry.read().as_ref() {
            let old = entry.current.swap(value, Ordering::Relaxed);
            if value != old {
                self.trigger_update(entry);
            }
        }
    }

    /// Set metadata
    pub fn set_metadata(&self, key: impl Into<String>, value: impl Into<String>) {
        if let Some(entry) = self.current_entry.read().as_ref() {
            entry.metadata.write().insert(key.into(), value.into());
        }
    }

    /// Cancel current operation
    pub fn cancel(&self) {
        if let Some(entry) = self.current_entry.read().as_ref() {
            entry.cancelled.store(true, Ordering::Relaxed);
            let mut stats = self.statistics.write();
            stats.cancelled_count += 1;
        }
    }

    /// Check if current operation is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.current_entry
            .read()
            .as_ref()
            .map(|e| e.cancelled.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    /// Finish current operation
    pub fn finish(&self) {
        let config = self.config.read();
        if !config.enabled {
            return;
        }

        // Take the entry and release the write lock immediately
        let entry = self.current_entry.write().take();

        if let Some(entry) = entry {
            // Set to complete
            entry.current.store(entry.total, Ordering::Relaxed);

            // Final update
            self.trigger_update(&entry);

            // Update statistics
            {
                let mut stats = self.statistics.write();
                stats.completed_count += 1;
                stats.total_elapsed += entry.start_time.elapsed();
                if stats.total_items_processed > 0 && stats.total_elapsed.as_secs_f64() > 0.0 {
                    stats.average_rate =
                        stats.total_items_processed as f64 / stats.total_elapsed.as_secs_f64();
                }
            }

            // Pop from stack if hierarchical (now safe to acquire write lock again)
            if config.enable_hierarchical {
                let mut stack = self.entry_stack.lock();
                stack.pop();
                *self.current_entry.write() = stack.last().cloned();
            }
        }
    }

    /// Add progress callback
    pub fn add_callback(&self, callback: ProgressCallback) {
        self.callbacks.write().push(callback);
    }

    /// Clear all callbacks
    pub fn clear_callbacks(&self) {
        self.callbacks.write().clear();
    }

    /// Get current progress
    pub fn current_progress(&self) -> Option<ProgressUpdate> {
        self.current_entry.read().as_ref().map(|e| e.get_update())
    }

    /// Get statistics
    pub fn statistics(&self) -> ProgressStatistics {
        self.statistics.read().clone()
    }

    /// Get history
    pub fn history(&self) -> Vec<ProgressUpdate> {
        self.history.lock().iter().cloned().collect()
    }

    /// Clear history
    pub fn clear_history(&self) {
        self.history.lock().clear();
    }

    /// Enable/disable progress reporting
    pub fn set_enabled(&self, enabled: bool) {
        self.config.write().enabled = enabled;
    }

    /// Check if progress reporting is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.read().enabled
    }

    /// Trigger update callbacks
    fn trigger_update(&self, entry: &Arc<ProgressEntry>) {
        let update = entry.get_update();

        // Add to history
        {
            let config = self.config.read();
            let mut history = self.history.lock();
            history.push_back(update.clone());
            if history.len() > config.max_history_size {
                history.pop_front();
            }

            // Print to console if enabled
            if config.print_to_console {
                println!(
                    "{}: {}",
                    update.operation_name,
                    update.format_progress_bar(config.progress_bar_width)
                );
            }
        }

        // Call callbacks
        let callbacks = self.callbacks.read();
        for callback in callbacks.iter() {
            callback(&update);
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        *self.current_entry.write() = None;
        self.entry_stack.lock().clear();
        self.history.lock().clear();
        *self.statistics.write() = ProgressStatistics {
            total_operations: 0,
            total_items_processed: 0,
            total_elapsed: Duration::ZERO,
            average_rate: 0.0,
            peak_rate: 0.0,
            cancelled_count: 0,
            completed_count: 0,
        };
    }
}

impl Default for ProgressReporter {
    fn default() -> Self {
        Self::new(ProgressConfig::default())
    }
}

/// Global progress reporter instance
static GLOBAL_REPORTER: once_cell::sync::Lazy<ProgressReporter> =
    once_cell::sync::Lazy::new(|| ProgressReporter::default());

/// Get the global progress reporter
pub fn global_reporter() -> &'static ProgressReporter {
    &GLOBAL_REPORTER
}

/// Progress scope guard (RAII)
pub struct ProgressScope {
    reporter: ProgressReporter,
    cancelled: Arc<AtomicBool>,
}

impl ProgressScope {
    /// Create a new progress scope
    pub fn new(reporter: ProgressReporter, name: impl Into<String>, total: u64) -> Self {
        reporter.start(name, total);
        Self {
            reporter,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Update progress
    pub fn update(&self, delta: u64) -> AutogradResult<()> {
        if self.reporter.is_cancelled() || self.cancelled.load(Ordering::Relaxed) {
            return Err(AutogradError::OperationCancelled {
                operation: "Progress tracking".into(),
                partial_completion: self.reporter.current_progress().map(|p| p.percentage),
            });
        }
        self.reporter.update(delta);
        Ok(())
    }

    /// Set progress
    pub fn set(&self, value: u64) -> AutogradResult<()> {
        if self.reporter.is_cancelled() || self.cancelled.load(Ordering::Relaxed) {
            return Err(AutogradError::OperationCancelled {
                operation: "Progress tracking".into(),
                partial_completion: self.reporter.current_progress().map(|p| p.percentage),
            });
        }
        self.reporter.set(value);
        Ok(())
    }

    /// Set metadata
    pub fn set_metadata(&self, key: impl Into<String>, value: impl Into<String>) {
        self.reporter.set_metadata(key, value);
    }

    /// Cancel operation
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.reporter.cancel();
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.reporter.is_cancelled() || self.cancelled.load(Ordering::Relaxed)
    }
}

impl Drop for ProgressScope {
    fn drop(&mut self) {
        self.reporter.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_progress() {
        let reporter = ProgressReporter::new(ProgressConfig::default());
        reporter.start("test_op", 100);

        for i in 0..100 {
            reporter.update(1);
            if let Some(progress) = reporter.current_progress() {
                assert_eq!(progress.current, (i + 1) as u64);
                assert_eq!(progress.total, 100);
            }
        }

        reporter.finish();
        let stats = reporter.statistics();
        assert_eq!(stats.total_items_processed, 100);
        assert_eq!(stats.completed_count, 1);
    }

    #[test]
    fn test_progress_callbacks() {
        let reporter = ProgressReporter::new(ProgressConfig::default());
        let callback_count = Arc::new(AtomicUsize::new(0));
        let count_clone = callback_count.clone();

        reporter.add_callback(Arc::new(move |_update| {
            count_clone.fetch_add(1, Ordering::Relaxed);
        }));

        reporter.start("test_op", 10);
        for _ in 0..10 {
            reporter.update(1);
        }
        reporter.finish();

        assert!(callback_count.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_progress_cancellation() {
        let reporter = ProgressReporter::new(ProgressConfig::default());
        reporter.start("test_op", 100);

        reporter.update(50);
        assert!(!reporter.is_cancelled());

        reporter.cancel();
        assert!(reporter.is_cancelled());

        let stats = reporter.statistics();
        assert_eq!(stats.cancelled_count, 1);
    }

    #[test]
    fn test_progress_scope() {
        let reporter = ProgressReporter::new(ProgressConfig::default());
        {
            let scope = ProgressScope::new(reporter.clone(), "test_scope", 10);
            for _i in 0..10 {
                scope.update(1).unwrap();
            }
        }

        let stats = reporter.statistics();
        assert_eq!(stats.completed_count, 1);
        assert_eq!(stats.total_items_processed, 10);
    }

    #[test]
    fn test_progress_estimation() {
        let mut config = ProgressConfig::default();
        config.enable_time_estimation = true;
        config.update_interval_ms = 0; // No throttling for tests

        let reporter = ProgressReporter::new(config);
        reporter.start("test_op", 100);

        // Simulate work with consistent rate
        for _ in 0..50 {
            reporter.update(1);
            thread::sleep(Duration::from_millis(1));
        }

        if let Some(progress) = reporter.current_progress() {
            assert!(progress.rate > 0.0);
            // Estimation might be available after some updates
        }

        reporter.finish();
    }

    #[test]
    fn test_hierarchical_progress() {
        let mut config = ProgressConfig::default();
        config.enable_hierarchical = true;

        let reporter = ProgressReporter::new(config);

        reporter.start("outer_op", 10);
        for _ in 0..10 {
            reporter.start("inner_op", 5);
            for _ in 0..5 {
                reporter.update(1);
            }
            reporter.finish();
        }
        reporter.finish();

        let stats = reporter.statistics();
        assert_eq!(stats.completed_count, 11); // 1 outer + 10 inner
    }

    #[test]
    fn test_progress_metadata() {
        let reporter = ProgressReporter::new(ProgressConfig::default());
        reporter.start("test_op", 10);
        reporter.set_metadata("layer", "conv1");
        reporter.set_metadata("batch", "0");

        if let Some(progress) = reporter.current_progress() {
            assert_eq!(progress.metadata.get("layer"), Some(&"conv1".to_string()));
            assert_eq!(progress.metadata.get("batch"), Some(&"0".to_string()));
        }

        reporter.finish();
    }

    #[test]
    fn test_progress_history() {
        let mut config = ProgressConfig::default();
        config.max_history_size = 5;

        let reporter = ProgressReporter::new(config);
        reporter.start("test_op", 10);

        for _ in 0..10 {
            reporter.update(1);
        }

        reporter.finish();
        let history = reporter.history();
        assert!(!history.is_empty());
        assert!(history.len() <= 5);
    }

    #[test]
    fn test_progress_bar_formatting() {
        let update = ProgressUpdate {
            operation_name: "test".into(),
            current: 50,
            total: 100,
            percentage: 50.0,
            elapsed: Duration::from_secs(10),
            estimated_remaining: Some(Duration::from_secs(10)),
            rate: 5.0,
            is_complete: false,
            metadata: HashMap::new(),
        };

        let bar = update.format_progress_bar(20);
        assert!(bar.contains("50.00%"));
        assert!(bar.contains("ETA:"));
    }

    #[test]
    fn test_global_reporter() {
        let reporter = global_reporter();
        reporter.reset();

        reporter.start("global_test", 5);
        for _ in 0..5 {
            reporter.update(1);
        }
        reporter.finish();

        let stats = reporter.statistics();
        assert_eq!(stats.total_items_processed, 5);
    }
}
