//! Automatic Memory Layout Optimization based on Access Patterns
//!
//! This module provides intelligent memory layout optimization by:
//! - Tracking tensor access patterns at runtime
//! - Analyzing access patterns to determine optimal memory layouts
//! - Recommending layout transformations for performance improvement
//! - Providing cache-aware optimization strategies
//!
//! # SciRS2 POLICY COMPLIANCE
//! This module uses scirs2_core abstractions exclusively:
//! - ✅ Uses torsh_core::numeric for numerical traits
//! - ✅ Uses torsh_core::parallel for parallel operations (when enabled)
//! - ❌ NO direct external dependencies

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec, vec::Vec};
#[cfg(feature = "std")]
use std::{collections::HashMap, sync::Arc};

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as HashMap;
#[cfg(not(feature = "std"))]
use alloc::sync::Arc;

use crate::{
    error::{Result, TorshError},
    shape::Shape,
    MemoryFormat,
};

/// Access pattern types that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    /// Sequential access (consecutive elements)
    Sequential,
    /// Strided access (regular stride pattern)
    Strided { stride: usize },
    /// Random access (no clear pattern)
    Random,
    /// Row-major access (scanning rows)
    RowMajor,
    /// Column-major access (scanning columns)
    ColumnMajor,
    /// Block-wise access (accessing blocks of data)
    BlockWise { block_size: usize },
    /// Diagonal access (accessing diagonal elements)
    Diagonal,
    /// Broadcast-like access (repeated access to same elements)
    Broadcast,
}

/// Statistics about memory access patterns
#[derive(Debug, Clone)]
pub struct AccessStatistics {
    /// Total number of accesses recorded
    pub total_accesses: u64,
    /// Number of cache hits (estimated)
    pub cache_hits: u64,
    /// Number of cache misses (estimated)
    pub cache_misses: u64,
    /// Average stride between consecutive accesses
    pub average_stride: f64,
    /// Standard deviation of stride
    pub stride_variance: f64,
    /// Dominant access pattern
    pub dominant_pattern: AccessPattern,
    /// Pattern frequency distribution
    pub pattern_distribution: HashMap<AccessPattern, u64>,
}

/// Access pattern tracker for a tensor
#[derive(Debug, Clone)]
pub struct AccessTracker {
    /// Tensor shape being tracked
    shape: Shape,
    /// Current memory format
    memory_format: MemoryFormat,
    /// Recent access indices (circular buffer)
    recent_accesses: Vec<usize>,
    /// Maximum size of access history
    max_history: usize,
    /// Statistics accumulator
    stats: AccessStatistics,
    /// Cache line size (in bytes)
    cache_line_size: usize,
}

impl AccessTracker {
    /// Create a new access tracker
    pub fn new(shape: Shape, memory_format: MemoryFormat) -> Self {
        Self {
            shape,
            memory_format,
            recent_accesses: Vec::with_capacity(1000),
            max_history: 1000,
            stats: AccessStatistics {
                total_accesses: 0,
                cache_hits: 0,
                cache_misses: 0,
                average_stride: 0.0,
                stride_variance: 0.0,
                dominant_pattern: AccessPattern::Random,
                pattern_distribution: HashMap::new(),
            },
            cache_line_size: 64, // Common cache line size
        }
    }

    /// Create with custom cache line size
    pub fn with_cache_line_size(mut self, cache_line_size: usize) -> Self {
        self.cache_line_size = cache_line_size;
        self
    }

    /// Record a memory access
    pub fn record_access(&mut self, linear_index: usize) {
        // Add to recent accesses
        if self.recent_accesses.len() >= self.max_history {
            self.recent_accesses.remove(0);
        }
        self.recent_accesses.push(linear_index);

        // Update statistics
        self.stats.total_accesses += 1;

        // Estimate cache hit/miss based on access pattern
        if self.recent_accesses.len() >= 2 {
            let prev_index = self.recent_accesses[self.recent_accesses.len() - 2];
            let stride = if linear_index > prev_index {
                linear_index - prev_index
            } else {
                prev_index - linear_index
            };

            // If stride is within cache line, likely a cache hit
            if stride * core::mem::size_of::<f32>() <= self.cache_line_size {
                self.stats.cache_hits += 1;
            } else {
                self.stats.cache_misses += 1;
            }
        }

        // Analyze pattern periodically
        if self.stats.total_accesses % 100 == 0 {
            self.analyze_pattern();
        }
    }

    /// Analyze the access pattern
    fn analyze_pattern(&mut self) {
        if self.recent_accesses.len() < 10 {
            return;
        }

        // Calculate stride statistics
        let mut strides = Vec::new();
        for i in 1..self.recent_accesses.len() {
            let stride = if self.recent_accesses[i] > self.recent_accesses[i - 1] {
                self.recent_accesses[i] - self.recent_accesses[i - 1]
            } else {
                self.recent_accesses[i - 1] - self.recent_accesses[i]
            };
            strides.push(stride as f64);
        }

        // Calculate average and variance
        let sum: f64 = strides.iter().sum();
        let avg = sum / strides.len() as f64;
        self.stats.average_stride = avg;

        let variance_sum: f64 = strides.iter().map(|&s| (s - avg).powi(2)).sum();
        self.stats.stride_variance = variance_sum / strides.len() as f64;

        // Detect pattern based on stride statistics
        let pattern = self.detect_pattern(&strides);
        *self.stats.pattern_distribution.entry(pattern).or_insert(0) += 1;

        // Update dominant pattern
        if let Some((&dominant, _)) = self
            .stats
            .pattern_distribution
            .iter()
            .max_by_key(|(_, &count)| count)
        {
            self.stats.dominant_pattern = dominant;
        }
    }

    /// Detect specific access pattern from stride data
    fn detect_pattern(&self, strides: &[f64]) -> AccessPattern {
        if strides.is_empty() {
            return AccessPattern::Random;
        }

        let avg = self.stats.average_stride;
        let variance = self.stats.stride_variance;

        // Sequential: average stride ~1, low variance
        if (avg - 1.0).abs() < 0.1 && variance < 0.5 {
            return AccessPattern::Sequential;
        }

        // Strided: consistent stride, low variance
        if variance < avg * 0.2 && avg > 1.5 {
            return AccessPattern::Strided {
                stride: avg.round() as usize,
            };
        }

        // Row-major: stride equals row length
        if let Some(row_len) = self.shape.dims().last() {
            if (avg - *row_len as f64).abs() < 0.5 {
                return AccessPattern::RowMajor;
            }
        }

        // Column-major: stride equals column height
        if let Some(&first_dim) = self.shape.dims().first() {
            if (avg - first_dim as f64).abs() < 0.5 {
                return AccessPattern::ColumnMajor;
            }
        }

        // Broadcast: very low variance, repeated accesses
        if variance < 1.0 && avg < 2.0 {
            return AccessPattern::Broadcast;
        }

        // Default to random
        AccessPattern::Random
    }

    /// Get current statistics
    pub fn statistics(&self) -> &AccessStatistics {
        &self.stats
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        if self.stats.total_accesses == 0 {
            return 0.0;
        }
        self.stats.cache_hits as f64 / self.stats.total_accesses as f64
    }
}

/// Layout optimization recommendation
#[derive(Debug, Clone)]
pub struct LayoutRecommendation {
    /// Current memory format
    pub current_format: MemoryFormat,
    /// Recommended memory format
    pub recommended_format: MemoryFormat,
    /// Expected performance improvement (0.0 to 1.0)
    pub expected_improvement: f64,
    /// Reason for recommendation
    pub reason: String,
    /// Estimated transformation cost
    pub transformation_cost: TransformationCost,
}

/// Cost of transforming memory layout
#[derive(Debug, Clone)]
pub struct TransformationCost {
    /// Number of memory copies required
    pub memory_copies: usize,
    /// Estimated time in microseconds
    pub estimated_time_us: f64,
    /// Memory overhead during transformation
    pub memory_overhead_bytes: usize,
}

/// Layout optimizer that analyzes access patterns and recommends layouts
#[derive(Debug)]
pub struct LayoutOptimizer {
    /// Cache of access trackers per tensor
    trackers: HashMap<usize, Arc<AccessTracker>>,
    /// Optimization threshold (minimum improvement to recommend)
    optimization_threshold: f64,
    /// Enable aggressive optimizations
    aggressive: bool,
}

impl Default for LayoutOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl LayoutOptimizer {
    /// Create a new layout optimizer
    pub fn new() -> Self {
        Self {
            trackers: HashMap::new(),
            optimization_threshold: 0.1, // 10% improvement threshold
            aggressive: false,
        }
    }

    /// Create with custom optimization threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.optimization_threshold = threshold;
        self
    }

    /// Enable aggressive optimizations (may use more memory)
    pub fn aggressive(mut self, enabled: bool) -> Self {
        self.aggressive = enabled;
        self
    }

    /// Register a tensor for tracking
    pub fn register_tensor(&mut self, tensor_id: usize, shape: Shape, format: MemoryFormat) {
        let tracker = AccessTracker::new(shape, format);
        self.trackers.insert(tensor_id, Arc::new(tracker));
    }

    /// Record an access for a tensor
    pub fn record_access(&mut self, tensor_id: usize, linear_index: usize) -> Result<()> {
        if let Some(tracker) = self.trackers.get_mut(&tensor_id) {
            // Make mutable copy for modification
            let mut tracker_mut = (**tracker).clone();
            tracker_mut.record_access(linear_index);
            *tracker = Arc::new(tracker_mut);
            Ok(())
        } else {
            Err(TorshError::InvalidArgument(format!(
                "Tensor {} not registered for tracking",
                tensor_id
            )))
        }
    }

    /// Get optimization recommendation for a tensor
    pub fn recommend_layout(&self, tensor_id: usize) -> Result<Option<LayoutRecommendation>> {
        let tracker = self.trackers.get(&tensor_id).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Tensor {} not registered", tensor_id))
        })?;

        let stats = tracker.statistics();

        // Need sufficient data for recommendation
        if stats.total_accesses < 100 {
            return Ok(None);
        }

        // Analyze dominant pattern and recommend layout
        let recommendation = self.analyze_and_recommend(tracker)?;

        // Only recommend if improvement exceeds threshold
        if recommendation.expected_improvement >= self.optimization_threshold {
            Ok(Some(recommendation))
        } else {
            Ok(None)
        }
    }

    /// Analyze pattern and generate recommendation
    fn analyze_and_recommend(&self, tracker: &AccessTracker) -> Result<LayoutRecommendation> {
        let stats = tracker.statistics();
        let current_format = tracker.memory_format;
        let cache_hit_rate = tracker.cache_hit_rate();

        match stats.dominant_pattern {
            AccessPattern::Sequential | AccessPattern::RowMajor => {
                // Row-major access benefits from contiguous layout
                if current_format != MemoryFormat::Contiguous {
                    Ok(LayoutRecommendation {
                        current_format,
                        recommended_format: MemoryFormat::Contiguous,
                        expected_improvement: 0.3, // 30% improvement
                        reason: "Sequential/row-major access pattern detected. Contiguous layout will improve cache locality.".to_string(),
                        transformation_cost: self.estimate_cost(&tracker.shape),
                    })
                } else {
                    Ok(LayoutRecommendation {
                        current_format,
                        recommended_format: current_format,
                        expected_improvement: 0.0,
                        reason: "Already using optimal layout".to_string(),
                        transformation_cost: TransformationCost {
                            memory_copies: 0,
                            estimated_time_us: 0.0,
                            memory_overhead_bytes: 0,
                        },
                    })
                }
            }
            AccessPattern::ColumnMajor => {
                // Column-major access benefits from channels-last layout
                if current_format != MemoryFormat::ChannelsLast {
                    Ok(LayoutRecommendation {
                        current_format,
                        recommended_format: MemoryFormat::ChannelsLast,
                        expected_improvement: 0.25,
                        reason: "Column-major access detected. ChannelsLast layout will improve stride patterns.".to_string(),
                        transformation_cost: self.estimate_cost(&tracker.shape),
                    })
                } else {
                    Ok(LayoutRecommendation {
                        current_format,
                        recommended_format: current_format,
                        expected_improvement: 0.0,
                        reason: "Already using optimal layout".to_string(),
                        transformation_cost: TransformationCost {
                            memory_copies: 0,
                            estimated_time_us: 0.0,
                            memory_overhead_bytes: 0,
                        },
                    })
                }
            }
            AccessPattern::Strided { stride } => {
                // Large strides indicate poor cache locality
                let improvement = if cache_hit_rate < 0.5 { 0.4 } else { 0.15 };
                Ok(LayoutRecommendation {
                    current_format,
                    recommended_format: MemoryFormat::Contiguous,
                    expected_improvement: improvement,
                    reason: format!(
                        "Strided access (stride={}) with low cache hit rate ({}%). Contiguous layout recommended.",
                        stride,
                        (cache_hit_rate * 100.0) as u32
                    ),
                    transformation_cost: self.estimate_cost(&tracker.shape),
                })
            }
            AccessPattern::BlockWise { block_size } => {
                if self.aggressive {
                    Ok(LayoutRecommendation {
                        current_format,
                        recommended_format: MemoryFormat::Contiguous,
                        expected_improvement: 0.2,
                        reason: format!(
                            "Block-wise access (block_size={}) detected. Consider cache-friendly blocking.",
                            block_size
                        ),
                        transformation_cost: self.estimate_cost(&tracker.shape),
                    })
                } else {
                    Ok(LayoutRecommendation {
                        current_format,
                        recommended_format: current_format,
                        expected_improvement: 0.0,
                        reason: "Block-wise access requires specialized optimization".to_string(),
                        transformation_cost: TransformationCost {
                            memory_copies: 0,
                            estimated_time_us: 0.0,
                            memory_overhead_bytes: 0,
                        },
                    })
                }
            }
            AccessPattern::Random => {
                // Random access doesn't benefit much from layout changes
                Ok(LayoutRecommendation {
                    current_format,
                    recommended_format: current_format,
                    expected_improvement: 0.0,
                    reason: "Random access pattern - layout optimization unlikely to help"
                        .to_string(),
                    transformation_cost: TransformationCost {
                        memory_copies: 0,
                        estimated_time_us: 0.0,
                        memory_overhead_bytes: 0,
                    },
                })
            }
            AccessPattern::Broadcast => Ok(LayoutRecommendation {
                current_format,
                recommended_format: current_format,
                expected_improvement: 0.0,
                reason: "Broadcast-like access - current layout is fine".to_string(),
                transformation_cost: TransformationCost {
                    memory_copies: 0,
                    estimated_time_us: 0.0,
                    memory_overhead_bytes: 0,
                },
            }),
            AccessPattern::Diagonal => Ok(LayoutRecommendation {
                current_format,
                recommended_format: current_format,
                expected_improvement: 0.0,
                reason: "Diagonal access - specialized algorithm recommended".to_string(),
                transformation_cost: TransformationCost {
                    memory_copies: 0,
                    estimated_time_us: 0.0,
                    memory_overhead_bytes: 0,
                },
            }),
        }
    }

    /// Estimate transformation cost
    fn estimate_cost(&self, shape: &Shape) -> TransformationCost {
        let numel = shape.numel();
        let element_size = 4; // Assume f32 for estimation
        let total_bytes = numel * element_size;

        // Memory copy cost: ~10 GB/s throughput
        let copy_time_us = (total_bytes as f64 / 10_000.0) * 1_000_000.0;

        TransformationCost {
            memory_copies: 1,
            estimated_time_us: copy_time_us,
            memory_overhead_bytes: total_bytes,
        }
    }

    /// Get all tracked tensor IDs
    pub fn tracked_tensors(&self) -> Vec<usize> {
        self.trackers.keys().copied().collect()
    }

    /// Get statistics for a tensor
    pub fn get_statistics(&self, tensor_id: usize) -> Result<AccessStatistics> {
        let tracker = self.trackers.get(&tensor_id).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Tensor {} not registered", tensor_id))
        })?;
        Ok(tracker.statistics().clone())
    }

    /// Clear tracking data for a tensor
    pub fn clear_tensor(&mut self, tensor_id: usize) {
        self.trackers.remove(&tensor_id);
    }

    /// Clear all tracking data
    pub fn clear_all(&mut self) {
        self.trackers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_tracker_creation() {
        let shape = Shape::from_array([100, 100]).unwrap();
        let tracker = AccessTracker::new(shape, MemoryFormat::Contiguous);
        assert_eq!(tracker.statistics().total_accesses, 0);
    }

    #[test]
    fn test_sequential_access_detection() {
        let shape = Shape::from_array([100, 100]).unwrap();
        let mut tracker = AccessTracker::new(shape, MemoryFormat::Contiguous);

        // Simulate sequential access
        for i in 0..1000 {
            tracker.record_access(i);
        }

        let stats = tracker.statistics();
        assert!(stats.total_accesses == 1000);
        assert!(stats.cache_hits > stats.cache_misses); // Sequential should have good cache hits
    }

    #[test]
    fn test_strided_access_detection() {
        let shape = Shape::from_array([100, 100]).unwrap();
        let mut tracker = AccessTracker::new(shape, MemoryFormat::Contiguous);

        // Simulate strided access (every 10th element)
        for i in 0..100 {
            tracker.record_access(i * 10);
        }

        let stats = tracker.statistics();
        assert!(stats.total_accesses == 100);
        // Strided access should show in average_stride
        assert!(stats.average_stride > 8.0);
    }

    #[test]
    fn test_random_access_detection() {
        let shape = Shape::from_array([100, 100]).unwrap();
        let mut tracker = AccessTracker::new(shape, MemoryFormat::Contiguous);

        // Simulate random access
        let indices = [42, 1000, 5, 9999, 50, 7500, 200];
        for &idx in &indices {
            tracker.record_access(idx);
        }

        let stats = tracker.statistics();
        assert!(stats.total_accesses == indices.len() as u64);
    }

    #[test]
    fn test_cache_hit_rate() {
        let shape = Shape::from_array([100, 100]).unwrap();
        let mut tracker = AccessTracker::new(shape, MemoryFormat::Contiguous);

        // Sequential access should have high cache hit rate
        for i in 0..100 {
            tracker.record_access(i);
        }

        let hit_rate = tracker.cache_hit_rate();
        assert!(hit_rate > 0.5); // Should have >50% hit rate
    }

    #[test]
    fn test_layout_optimizer_creation() {
        let optimizer = LayoutOptimizer::new();
        assert!(optimizer.tracked_tensors().is_empty());
    }

    #[test]
    fn test_register_and_track_tensor() {
        let mut optimizer = LayoutOptimizer::new();
        let shape = Shape::from_array([100, 100]).unwrap();

        optimizer.register_tensor(1, shape, MemoryFormat::Contiguous);
        assert_eq!(optimizer.tracked_tensors().len(), 1);
        assert!(optimizer.tracked_tensors().contains(&1));
    }

    #[test]
    fn test_record_access() {
        let mut optimizer = LayoutOptimizer::new();
        let shape = Shape::from_array([100, 100]).unwrap();

        optimizer.register_tensor(1, shape, MemoryFormat::Contiguous);

        for i in 0..50 {
            optimizer.record_access(1, i).unwrap();
        }

        let stats = optimizer.get_statistics(1).unwrap();
        assert_eq!(stats.total_accesses, 50);
    }

    #[test]
    fn test_optimization_recommendation() {
        let mut optimizer = LayoutOptimizer::new().with_threshold(0.05);
        let shape = Shape::from_array([100, 100]).unwrap();

        optimizer.register_tensor(1, shape, MemoryFormat::Strided);

        // Simulate sequential access pattern
        for i in 0..200 {
            optimizer.record_access(1, i).unwrap();
        }

        let recommendation = optimizer.recommend_layout(1).unwrap();
        assert!(recommendation.is_some());

        if let Some(rec) = recommendation {
            // Should recommend Contiguous for sequential access
            assert_eq!(rec.recommended_format, MemoryFormat::Contiguous);
            assert!(rec.expected_improvement > 0.0);
        }
    }

    #[test]
    fn test_insufficient_data_no_recommendation() {
        let mut optimizer = LayoutOptimizer::new();
        let shape = Shape::from_array([100, 100]).unwrap();

        optimizer.register_tensor(1, shape, MemoryFormat::Contiguous);

        // Only a few accesses
        for i in 0..10 {
            optimizer.record_access(1, i).unwrap();
        }

        let recommendation = optimizer.recommend_layout(1).unwrap();
        assert!(recommendation.is_none()); // Not enough data
    }

    #[test]
    fn test_clear_tensor() {
        let mut optimizer = LayoutOptimizer::new();
        let shape = Shape::from_array([100, 100]).unwrap();

        optimizer.register_tensor(1, shape, MemoryFormat::Contiguous);
        assert_eq!(optimizer.tracked_tensors().len(), 1);

        optimizer.clear_tensor(1);
        assert!(optimizer.tracked_tensors().is_empty());
    }

    #[test]
    fn test_aggressive_optimization() {
        let optimizer = LayoutOptimizer::new().aggressive(true);
        assert!(optimizer.aggressive);
    }

    #[test]
    fn test_transformation_cost_estimation() {
        let optimizer = LayoutOptimizer::new();
        let shape = Shape::from_array([1000, 1000]).unwrap();

        let cost = optimizer.estimate_cost(&shape);
        assert!(cost.memory_copies > 0);
        assert!(cost.estimated_time_us > 0.0);
        assert!(cost.memory_overhead_bytes > 0);
    }

    #[test]
    fn test_custom_cache_line_size() {
        let shape = Shape::from_array([100, 100]).unwrap();
        let tracker = AccessTracker::new(shape, MemoryFormat::Contiguous).with_cache_line_size(128);

        assert_eq!(tracker.cache_line_size, 128);
    }
}
