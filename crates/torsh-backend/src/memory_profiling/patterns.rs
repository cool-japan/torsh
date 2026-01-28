// Memory Profiling: Access Patterns Analysis and Optimization
//
// This module provides comprehensive memory access pattern analysis and performance
// optimization recommendations for the ToRSh deep learning framework. It tracks
// access patterns, identifies performance bottlenecks, and generates actionable
// optimization suggestions.

use std::collections::{HashMap, BTreeMap, VecDeque};
use std::time::{Instant, Duration};
use scirs2_core::error::{CoreError, Result};

/// Represents different types of memory access patterns
#[derive(Debug, Clone, PartialEq)]
pub enum AccessPattern {
    /// Sequential access pattern - good cache locality
    Sequential {
        start_addr: usize,
        stride: usize,
        count: usize,
    },
    /// Random access pattern - poor cache locality
    Random {
        addresses: Vec<usize>,
        entropy: f64,
    },
    /// Strided access pattern - predictable but potentially cache-unfriendly
    Strided {
        start_addr: usize,
        stride: usize,
        count: usize,
        regularity: f64,
    },
    /// Temporal clustering - repeated access to same regions
    TemporalCluster {
        hot_regions: Vec<MemoryRegion>,
        access_frequency: f64,
    },
    /// Spatial clustering - access to nearby memory regions
    SpatialCluster {
        center: usize,
        radius: usize,
        density: f64,
    },
    /// Mixed pattern - combination of multiple patterns
    Mixed {
        patterns: Vec<AccessPattern>,
        dominant_pattern: Box<AccessPattern>,
    },
}

/// Memory region definition for pattern analysis
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRegion {
    pub start_addr: usize,
    pub end_addr: usize,
    pub size: usize,
    pub access_count: usize,
    pub last_access: Instant,
    pub access_frequency: f64,
    pub locality_score: f64,
}

impl MemoryRegion {
    pub fn new(start_addr: usize, size: usize) -> Self {
        Self {
            start_addr,
            end_addr: start_addr + size,
            size,
            access_count: 0,
            last_access: Instant::now(),
            access_frequency: 0.0,
            locality_score: 0.0,
        }
    }

    pub fn contains(&self, address: usize) -> bool {
        address >= self.start_addr && address < self.end_addr
    }

    pub fn update_access(&mut self) {
        self.access_count += 1;
        let now = Instant::now();
        let time_since_last = now.duration_since(self.last_access).as_secs_f64();
        self.access_frequency = if time_since_last > 0.0 {
            1.0 / time_since_last
        } else {
            f64::INFINITY
        };
        self.last_access = now;
    }
}

/// Detailed memory access record
#[derive(Debug, Clone)]
pub struct AccessRecord {
    pub address: usize,
    pub size: usize,
    pub timestamp: Instant,
    pub access_type: AccessType,
    pub thread_id: u64,
    pub allocation_id: Option<usize>,
    pub context: AccessContext,
}

/// Type of memory access operation
#[derive(Debug, Clone, PartialEq)]
pub enum AccessType {
    Read,
    Write,
    ReadWrite,
    Prefetch,
    Flush,
    Invalidate,
}

/// Context information for memory access
#[derive(Debug, Clone)]
pub struct AccessContext {
    pub operation: String,
    pub tensor_shape: Option<Vec<usize>>,
    pub data_type: Option<String>,
    pub kernel_name: Option<String>,
    pub call_stack: Vec<String>,
}

/// Performance characteristics of access patterns
#[derive(Debug, Clone)]
pub struct AccessPatternMetrics {
    pub cache_hit_rate: f64,
    pub cache_miss_penalty: Duration,
    pub bandwidth_utilization: f64,
    pub memory_efficiency: f64,
    pub temporal_locality_score: f64,
    pub spatial_locality_score: f64,
    pub prefetch_accuracy: f64,
    pub access_regularity: f64,
}

/// Optimization recommendation based on access patterns
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub priority: RecommendationPriority,
    pub category: OptimizationCategory,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_complexity: ComplexityLevel,
    pub applicable_patterns: Vec<AccessPattern>,
    pub code_suggestions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
    Informational,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationCategory {
    CacheOptimization,
    MemoryLayout,
    AccessPattern,
    Prefetching,
    DataMovement,
    Parallelization,
    Algorithmic,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityLevel {
    Trivial,
    Low,
    Medium,
    High,
    Expert,
}

/// Main patterns analyzer with comprehensive access pattern detection
pub struct AccessPatternsAnalyzer {
    access_history: VecDeque<AccessRecord>,
    memory_regions: Vec<MemoryRegion>,
    pattern_cache: HashMap<String, AccessPattern>,
    metrics_history: VecDeque<AccessPatternMetrics>,
    recommendations: Vec<OptimizationRecommendation>,
    analysis_window: Duration,
    max_history_size: usize,
    cache_line_size: usize,
    page_size: usize,
}

impl AccessPatternsAnalyzer {
    pub fn new() -> Self {
        Self {
            access_history: VecDeque::new(),
            memory_regions: Vec::new(),
            pattern_cache: HashMap::new(),
            metrics_history: VecDeque::new(),
            recommendations: Vec::new(),
            analysis_window: Duration::from_secs(60),
            max_history_size: 100000,
            cache_line_size: 64,
            page_size: 4096,
        }
    }

    pub fn with_config(
        analysis_window: Duration,
        max_history_size: usize,
        cache_line_size: usize,
        page_size: usize,
    ) -> Self {
        Self {
            access_history: VecDeque::new(),
            memory_regions: Vec::new(),
            pattern_cache: HashMap::new(),
            metrics_history: VecDeque::new(),
            recommendations: Vec::new(),
            analysis_window,
            max_history_size,
            cache_line_size,
            page_size,
        }
    }

    /// Record a memory access for pattern analysis
    pub fn record_access(&mut self, access: AccessRecord) -> Result<()> {
        // Update memory regions
        self.update_memory_regions(&access);

        // Add to history
        self.access_history.push_back(access);

        // Maintain history size limit
        if self.access_history.len() > self.max_history_size {
            self.access_history.pop_front();
        }

        // Clean old entries outside analysis window
        self.clean_old_entries();

        Ok(())
    }

    /// Analyze current access patterns and generate insights
    pub fn analyze_patterns(&mut self) -> Result<Vec<AccessPattern>> {
        let recent_accesses = self.get_recent_accesses();

        let mut patterns = Vec::new();

        // Detect sequential patterns
        if let Some(sequential) = self.detect_sequential_pattern(&recent_accesses) {
            patterns.push(sequential);
        }

        // Detect strided patterns
        if let Some(strided) = self.detect_strided_pattern(&recent_accesses) {
            patterns.push(strided);
        }

        // Detect temporal clusters
        if let Some(temporal) = self.detect_temporal_clusters(&recent_accesses) {
            patterns.push(temporal);
        }

        // Detect spatial clusters
        if let Some(spatial) = self.detect_spatial_clusters(&recent_accesses) {
            patterns.push(spatial);
        }

        // Detect random patterns
        if let Some(random) = self.detect_random_pattern(&recent_accesses) {
            patterns.push(random);
        }

        // Cache patterns for future reference
        for (i, pattern) in patterns.iter().enumerate() {
            let key = format!("pattern_{}", i);
            self.pattern_cache.insert(key, pattern.clone());
        }

        Ok(patterns)
    }

    /// Generate performance metrics for current access patterns
    pub fn calculate_metrics(&mut self) -> Result<AccessPatternMetrics> {
        let recent_accesses = self.get_recent_accesses();

        let cache_hit_rate = self.calculate_cache_hit_rate(&recent_accesses);
        let bandwidth_utilization = self.calculate_bandwidth_utilization(&recent_accesses);
        let temporal_locality = self.calculate_temporal_locality(&recent_accesses);
        let spatial_locality = self.calculate_spatial_locality(&recent_accesses);
        let access_regularity = self.calculate_access_regularity(&recent_accesses);

        let metrics = AccessPatternMetrics {
            cache_hit_rate,
            cache_miss_penalty: Duration::from_nanos(100), // Typical L1 cache miss
            bandwidth_utilization,
            memory_efficiency: cache_hit_rate * bandwidth_utilization,
            temporal_locality_score: temporal_locality,
            spatial_locality_score: spatial_locality,
            prefetch_accuracy: 0.8, // Would be calculated based on actual prefetch data
            access_regularity,
        };

        // Store metrics in history
        self.metrics_history.push_back(metrics.clone());
        if self.metrics_history.len() > 1000 {
            self.metrics_history.pop_front();
        }

        Ok(metrics)
    }

    /// Generate optimization recommendations based on access patterns
    pub fn generate_recommendations(&mut self) -> Result<Vec<OptimizationRecommendation>> {
        let patterns = self.analyze_patterns()?;
        let metrics = self.calculate_metrics()?;

        let mut recommendations = Vec::new();

        // Analyze each pattern and generate recommendations
        for pattern in &patterns {
            recommendations.extend(self.recommend_for_pattern(pattern, &metrics)?);
        }

        // Add general recommendations based on metrics
        recommendations.extend(self.recommend_for_metrics(&metrics)?);

        // Sort by priority
        recommendations.sort_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal).reverse());

        self.recommendations = recommendations.clone();
        Ok(recommendations)
    }

    /// Get recent memory access records within the analysis window
    fn get_recent_accesses(&self) -> Vec<&AccessRecord> {
        let cutoff = Instant::now() - self.analysis_window;
        self.access_history
            .iter()
            .filter(|access| access.timestamp >= cutoff)
            .collect()
    }

    /// Update memory region tracking based on new access
    fn update_memory_regions(&mut self, access: &AccessRecord) {
        // Find existing region or create new one
        let mut found_region = false;

        for region in &mut self.memory_regions {
            if region.contains(access.address) {
                region.update_access();
                found_region = true;
                break;
            }
        }

        if !found_region {
            let mut new_region = MemoryRegion::new(access.address, access.size);
            new_region.update_access();
            self.memory_regions.push(new_region);
        }

        // Merge overlapping regions
        self.merge_overlapping_regions();
    }

    /// Merge overlapping memory regions for cleaner analysis
    fn merge_overlapping_regions(&mut self) {
        self.memory_regions.sort_by_key(|r| r.start_addr);

        let mut merged = Vec::new();
        let mut current: Option<MemoryRegion> = None;

        for region in self.memory_regions.drain(..) {
            match current.as_mut() {
                None => current = Some(region),
                Some(curr) => {
                    if curr.end_addr >= region.start_addr {
                        // Merge regions
                        curr.end_addr = curr.end_addr.max(region.end_addr);
                        curr.size = curr.end_addr - curr.start_addr;
                        curr.access_count += region.access_count;
                    } else {
                        merged.push(current.take().expect("current should be present"));
                        current = Some(region);
                    }
                }
            }
        }

        if let Some(last) = current {
            merged.push(last);
        }

        self.memory_regions = merged;
    }

    /// Clean old entries outside the analysis window
    fn clean_old_entries(&mut self) {
        let cutoff = Instant::now() - self.analysis_window;

        while let Some(front) = self.access_history.front() {
            if front.timestamp < cutoff {
                self.access_history.pop_front();
            } else {
                break;
            }
        }
    }

    /// Detect sequential access patterns
    fn detect_sequential_pattern(&self, accesses: &[&AccessRecord]) -> Option<AccessPattern> {
        if accesses.len() < 3 {
            return None;
        }

        let mut sequential_count = 0;
        let mut start_addr = None;
        let mut stride = None;

        for window in accesses.windows(2) {
            let addr1 = window[0].address;
            let addr2 = window[1].address;

            if start_addr.is_none() {
                start_addr = Some(addr1);
                stride = Some(addr2.saturating_sub(addr1));
            } else if let Some(expected_stride) = stride {
                if addr2.saturating_sub(addr1) == expected_stride {
                    sequential_count += 1;
                }
            }
        }

        let sequential_ratio = sequential_count as f64 / (accesses.len() - 1) as f64;

        if sequential_ratio > 0.7 {
            Some(AccessPattern::Sequential {
                start_addr: start_addr.unwrap_or(0),
                stride: stride.unwrap_or(1),
                count: sequential_count + 1,
            })
        } else {
            None
        }
    }

    /// Detect strided access patterns
    fn detect_strided_pattern(&self, accesses: &[&AccessRecord]) -> Option<AccessPattern> {
        if accesses.len() < 4 {
            return None;
        }

        let mut stride_counts: HashMap<usize, usize> = HashMap::new();

        for window in accesses.windows(2) {
            let stride = window[1].address.saturating_sub(window[0].address);
            *stride_counts.entry(stride).or_insert(0) += 1;
        }

        // Find most common stride
        let (most_common_stride, count) = stride_counts
            .iter()
            .max_by_key(|(_, &count)| count)?;

        let regularity = *count as f64 / (accesses.len() - 1) as f64;

        if regularity > 0.5 && *most_common_stride > self.cache_line_size {
            Some(AccessPattern::Strided {
                start_addr: accesses[0].address,
                stride: *most_common_stride,
                count: *count,
                regularity,
            })
        } else {
            None
        }
    }

    /// Detect temporal clustering patterns
    fn detect_temporal_clusters(&self, accesses: &[&AccessRecord]) -> Option<AccessPattern> {
        let mut region_accesses: HashMap<usize, (usize, Instant)> = HashMap::new();

        for access in accesses {
            let region_id = access.address / self.page_size;
            let entry = region_accesses.entry(region_id).or_insert((0, access.timestamp));
            entry.0 += 1;
            if access.timestamp > entry.1 {
                entry.1 = access.timestamp;
            }
        }

        let mut hot_regions = Vec::new();
        let total_accesses = accesses.len();

        for (&region_id, &(access_count, last_access)) in &region_accesses {
            let frequency = access_count as f64 / total_accesses as f64;
            if frequency > 0.1 { // Hot if > 10% of accesses
                hot_regions.push(MemoryRegion {
                    start_addr: region_id * self.page_size,
                    end_addr: (region_id + 1) * self.page_size,
                    size: self.page_size,
                    access_count,
                    last_access,
                    access_frequency: frequency,
                    locality_score: frequency * 2.0, // Weighted by frequency
                });
            }
        }

        if !hot_regions.is_empty() {
            let total_frequency: f64 = hot_regions.iter().map(|r| r.access_frequency).sum();
            Some(AccessPattern::TemporalCluster {
                hot_regions,
                access_frequency: total_frequency,
            })
        } else {
            None
        }
    }

    /// Detect spatial clustering patterns
    fn detect_spatial_clusters(&self, accesses: &[&AccessRecord]) -> Option<AccessPattern> {
        if accesses.is_empty() {
            return None;
        }

        // Calculate center of mass
        let center = accesses.iter().map(|a| a.address).sum::<usize>() / accesses.len();

        // Calculate average distance from center
        let avg_distance: f64 = accesses
            .iter()
            .map(|a| (a.address as i64 - center as i64).abs() as f64)
            .sum::<f64>() / accesses.len() as f64;

        let radius = avg_distance as usize;

        // Count accesses within 2x average distance
        let within_radius = accesses
            .iter()
            .filter(|a| (a.address as i64 - center as i64).abs() as usize <= radius * 2)
            .count();

        let density = within_radius as f64 / accesses.len() as f64;

        if density > 0.6 { // 60% of accesses within 2x radius
            Some(AccessPattern::SpatialCluster {
                center,
                radius,
                density,
            })
        } else {
            None
        }
    }

    /// Detect random access patterns
    fn detect_random_pattern(&self, accesses: &[&AccessRecord]) -> Option<AccessPattern> {
        if accesses.len() < 10 {
            return None;
        }

        // Calculate entropy of access addresses
        let mut address_counts: HashMap<usize, usize> = HashMap::new();
        for access in accesses {
            *address_counts.entry(access.address).or_insert(0) += 1;
        }

        let total = accesses.len() as f64;
        let entropy: f64 = address_counts
            .values()
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.log2()
            })
            .sum();

        let max_entropy = (accesses.len() as f64).log2();
        let normalized_entropy = entropy / max_entropy;

        if normalized_entropy > 0.8 { // High entropy indicates randomness
            Some(AccessPattern::Random {
                addresses: accesses.iter().map(|a| a.address).collect(),
                entropy: normalized_entropy,
            })
        } else {
            None
        }
    }

    /// Calculate cache hit rate based on access patterns
    fn calculate_cache_hit_rate(&self, accesses: &[&AccessRecord]) -> f64 {
        if accesses.is_empty() {
            return 0.0;
        }

        let mut cache_lines: HashMap<usize, Instant> = HashMap::new();
        let cache_size = 32 * 1024; // Assume 32KB L1 cache
        let cache_lines_count = cache_size / self.cache_line_size;
        let mut hits = 0;
        let mut total = 0;

        for access in accesses {
            let cache_line = access.address / self.cache_line_size;
            total += 1;

            if cache_lines.contains_key(&cache_line) {
                hits += 1;
            } else {
                if cache_lines.len() >= cache_lines_count {
                    // Evict oldest
                    if let Some(oldest_line) = cache_lines
                        .iter()
                        .min_by_key(|(_, &timestamp)| timestamp)
                        .map(|(&line, _)| line)
                    {
                        cache_lines.remove(&oldest_line);
                    }
                }
                cache_lines.insert(cache_line, access.timestamp);
            }
        }

        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Calculate bandwidth utilization efficiency
    fn calculate_bandwidth_utilization(&self, accesses: &[&AccessRecord]) -> f64 {
        if accesses.is_empty() {
            return 0.0;
        }

        let total_bytes: usize = accesses.iter().map(|a| a.size).sum();
        let time_span = accesses
            .last()
            .expect("accesses should not be empty")
            .timestamp
            .duration_since(accesses.first().expect("accesses should not be empty").timestamp)
            .as_secs_f64();

        if time_span > 0.0 {
            let bandwidth_used = total_bytes as f64 / time_span; // bytes per second
            let theoretical_max = 50_000_000_000.0; // 50 GB/s theoretical max
            (bandwidth_used / theoretical_max).min(1.0)
        } else {
            0.0
        }
    }

    /// Calculate temporal locality score
    fn calculate_temporal_locality(&self, accesses: &[&AccessRecord]) -> f64 {
        if accesses.len() < 2 {
            return 0.0;
        }

        let mut reuse_distances = Vec::new();
        let mut last_access: HashMap<usize, usize> = HashMap::new();

        for (i, access) in accesses.iter().enumerate() {
            if let Some(&last_idx) = last_access.get(&access.address) {
                reuse_distances.push(i - last_idx);
            }
            last_access.insert(access.address, i);
        }

        if reuse_distances.is_empty() {
            return 0.0;
        }

        let avg_reuse_distance: f64 = reuse_distances.iter().sum::<usize>() as f64 / reuse_distances.len() as f64;
        let max_distance = accesses.len() as f64;

        (max_distance - avg_reuse_distance) / max_distance
    }

    /// Calculate spatial locality score
    fn calculate_spatial_locality(&self, accesses: &[&AccessRecord]) -> f64 {
        if accesses.len() < 2 {
            return 0.0;
        }

        let mut spatial_distances = Vec::new();

        for window in accesses.windows(2) {
            let distance = (window[1].address as i64 - window[0].address as i64).abs() as usize;
            spatial_distances.push(distance);
        }

        let avg_distance: f64 = spatial_distances.iter().sum::<usize>() as f64 / spatial_distances.len() as f64;
        let cache_line_size = self.cache_line_size as f64;

        // Good spatial locality means small distances
        if avg_distance <= cache_line_size {
            1.0
        } else if avg_distance <= cache_line_size * 4.0 {
            1.0 - (avg_distance - cache_line_size) / (cache_line_size * 3.0)
        } else {
            0.0
        }
    }

    /// Calculate access regularity score
    fn calculate_access_regularity(&self, accesses: &[&AccessRecord]) -> f64 {
        if accesses.len() < 3 {
            return 0.0;
        }

        let mut intervals = Vec::new();
        for window in accesses.windows(2) {
            let interval = window[1].timestamp.duration_since(window[0].timestamp).as_nanos();
            intervals.push(interval);
        }

        let mean: f64 = intervals.iter().sum::<u128>() as f64 / intervals.len() as f64;
        let variance: f64 = intervals
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / intervals.len() as f64;

        let coefficient_of_variation = if mean > 0.0 {
            variance.sqrt() / mean
        } else {
            f64::INFINITY
        };

        // Lower CV means more regular
        if coefficient_of_variation < 0.1 {
            1.0
        } else if coefficient_of_variation < 1.0 {
            1.0 - coefficient_of_variation * 0.9
        } else {
            0.1
        }
    }

    /// Generate recommendations for a specific access pattern
    fn recommend_for_pattern(&self, pattern: &AccessPattern, metrics: &AccessPatternMetrics) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        match pattern {
            AccessPattern::Sequential { stride, count, .. } => {
                if *stride <= self.cache_line_size {
                    recommendations.push(OptimizationRecommendation {
                        priority: RecommendationPriority::Low,
                        category: OptimizationCategory::CacheOptimization,
                        description: format!("Sequential access pattern detected with optimal stride ({}). Consider prefetching.", stride),
                        expected_improvement: 1.1,
                        implementation_complexity: ComplexityLevel::Low,
                        applicable_patterns: vec![pattern.clone()],
                        code_suggestions: vec![
                            "Consider using __builtin_prefetch() for upcoming data".to_string(),
                            "Ensure data structures are cache-line aligned".to_string(),
                        ],
                    });
                } else {
                    recommendations.push(OptimizationRecommendation {
                        priority: RecommendationPriority::Medium,
                        category: OptimizationCategory::MemoryLayout,
                        description: format!("Sequential access with large stride ({}). Consider data layout optimization.", stride),
                        expected_improvement: 1.3,
                        implementation_complexity: ComplexityLevel::Medium,
                        applicable_patterns: vec![pattern.clone()],
                        code_suggestions: vec![
                            "Consider restructuring data for better cache line utilization".to_string(),
                            "Use array-of-structures vs structure-of-arrays optimization".to_string(),
                        ],
                    });
                }
            },

            AccessPattern::Strided { stride, regularity, .. } => {
                if *regularity > 0.8 {
                    recommendations.push(OptimizationRecommendation {
                        priority: RecommendationPriority::High,
                        category: OptimizationCategory::Prefetching,
                        description: format!("Highly regular strided pattern (regularity: {:.2}). Excellent candidate for hardware prefetching.", regularity),
                        expected_improvement: 1.4,
                        implementation_complexity: ComplexityLevel::Low,
                        applicable_patterns: vec![pattern.clone()],
                        code_suggestions: vec![
                            "Enable hardware prefetcher hints".to_string(),
                            format!("Use stride-aware prefetching with stride {}", stride),
                        ],
                    });
                }
            },

            AccessPattern::Random { entropy, addresses } => {
                recommendations.push(OptimizationRecommendation {
                    priority: RecommendationPriority::High,
                    category: OptimizationCategory::Algorithmic,
                    description: format!("Random access pattern detected (entropy: {:.2}). Consider algorithmic optimization.", entropy),
                    expected_improvement: 2.0,
                    implementation_complexity: ComplexityLevel::High,
                    applicable_patterns: vec![pattern.clone()],
                    code_suggestions: vec![
                        "Consider sorting data to improve access locality".to_string(),
                        "Use hash tables or spatial data structures for better access patterns".to_string(),
                        "Implement access pattern regularization".to_string(),
                    ],
                });
            },

            AccessPattern::TemporalCluster { hot_regions, access_frequency } => {
                recommendations.push(OptimizationRecommendation {
                    priority: RecommendationPriority::Medium,
                    category: OptimizationCategory::CacheOptimization,
                    description: format!("Temporal clustering detected ({} hot regions, {:.2} frequency).", hot_regions.len(), access_frequency),
                    expected_improvement: 1.25,
                    implementation_complexity: ComplexityLevel::Medium,
                    applicable_patterns: vec![pattern.clone()],
                    code_suggestions: vec![
                        "Keep hot data in faster memory tiers".to_string(),
                        "Use cache warming strategies for hot regions".to_string(),
                    ],
                });
            },

            AccessPattern::SpatialCluster { density, radius, .. } => {
                recommendations.push(OptimizationRecommendation {
                    priority: RecommendationPriority::Medium,
                    category: OptimizationCategory::MemoryLayout,
                    description: format!("Spatial clustering detected (density: {:.2}, radius: {}).", density, radius),
                    expected_improvement: 1.2,
                    implementation_complexity: ComplexityLevel::Medium,
                    applicable_patterns: vec![pattern.clone()],
                    code_suggestions: vec![
                        "Optimize memory layout to improve spatial locality".to_string(),
                        "Use memory pooling for clustered allocations".to_string(),
                    ],
                });
            },

            AccessPattern::Mixed { patterns, .. } => {
                recommendations.push(OptimizationRecommendation {
                    priority: RecommendationPriority::Medium,
                    category: OptimizationCategory::AccessPattern,
                    description: format!("Mixed access pattern with {} sub-patterns.", patterns.len()),
                    expected_improvement: 1.15,
                    implementation_complexity: ComplexityLevel::High,
                    applicable_patterns: vec![pattern.clone()],
                    code_suggestions: vec![
                        "Analyze sub-patterns individually for targeted optimization".to_string(),
                        "Consider adaptive optimization strategies".to_string(),
                    ],
                });
            },
        }

        Ok(recommendations)
    }

    /// Generate recommendations based on overall metrics
    fn recommend_for_metrics(&self, metrics: &AccessPatternMetrics) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        // Cache hit rate recommendations
        if metrics.cache_hit_rate < 0.6 {
            recommendations.push(OptimizationRecommendation {
                priority: RecommendationPriority::Critical,
                category: OptimizationCategory::CacheOptimization,
                description: format!("Poor cache hit rate: {:.2}%. Critical optimization needed.", metrics.cache_hit_rate * 100.0),
                expected_improvement: 3.0,
                implementation_complexity: ComplexityLevel::High,
                applicable_patterns: vec![], // Applies to all patterns
                code_suggestions: vec![
                    "Restructure data layouts for better cache locality".to_string(),
                    "Implement cache-aware algorithms".to_string(),
                    "Consider data compression to fit more in cache".to_string(),
                ],
            });
        } else if metrics.cache_hit_rate < 0.85 {
            recommendations.push(OptimizationRecommendation {
                priority: RecommendationPriority::High,
                category: OptimizationCategory::CacheOptimization,
                description: format!("Moderate cache hit rate: {:.2}%. Room for improvement.", metrics.cache_hit_rate * 100.0),
                expected_improvement: 1.5,
                implementation_complexity: ComplexityLevel::Medium,
                applicable_patterns: vec![],
                code_suggestions: vec![
                    "Fine-tune prefetching strategies".to_string(),
                    "Optimize loop ordering and blocking".to_string(),
                ],
            });
        }

        // Bandwidth utilization recommendations
        if metrics.bandwidth_utilization < 0.3 {
            recommendations.push(OptimizationRecommendation {
                priority: RecommendationPriority::High,
                category: OptimizationCategory::DataMovement,
                description: format!("Low bandwidth utilization: {:.2}%. Consider parallel access.", metrics.bandwidth_utilization * 100.0),
                expected_improvement: 2.5,
                implementation_complexity: ComplexityLevel::Medium,
                applicable_patterns: vec![],
                code_suggestions: vec![
                    "Implement parallel memory access patterns".to_string(),
                    "Use vectorized operations where possible".to_string(),
                    "Consider memory access coalescing".to_string(),
                ],
            });
        }

        // Temporal locality recommendations
        if metrics.temporal_locality_score < 0.4 {
            recommendations.push(OptimizationRecommendation {
                priority: RecommendationPriority::High,
                category: OptimizationCategory::AccessPattern,
                description: format!("Poor temporal locality: {:.2}. Optimize data reuse.", metrics.temporal_locality_score),
                expected_improvement: 1.8,
                implementation_complexity: ComplexityLevel::Medium,
                applicable_patterns: vec![],
                code_suggestions: vec![
                    "Reorganize computations to increase data reuse".to_string(),
                    "Implement loop tiling/blocking".to_string(),
                    "Use data structure pooling".to_string(),
                ],
            });
        }

        // Spatial locality recommendations
        if metrics.spatial_locality_score < 0.5 {
            recommendations.push(OptimizationRecommendation {
                priority: RecommendationPriority::High,
                category: OptimizationCategory::MemoryLayout,
                description: format!("Poor spatial locality: {:.2}. Optimize memory layout.", metrics.spatial_locality_score),
                expected_improvement: 1.6,
                implementation_complexity: ComplexityLevel::Medium,
                applicable_patterns: vec![],
                code_suggestions: vec![
                    "Use structure packing and alignment optimization".to_string(),
                    "Implement cache-conscious data structures".to_string(),
                    "Consider array-of-structures vs structure-of-arrays".to_string(),
                ],
            });
        }

        Ok(recommendations)
    }
}

impl Default for AccessPatternsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_access(address: usize, size: usize, offset_ms: u64) -> AccessRecord {
        AccessRecord {
            address,
            size,
            timestamp: Instant::now() - Duration::from_millis(offset_ms),
            access_type: AccessType::Read,
            thread_id: 1,
            allocation_id: None,
            context: AccessContext {
                operation: "test".to_string(),
                tensor_shape: None,
                data_type: None,
                kernel_name: None,
                call_stack: vec![],
            },
        }
    }

    #[test]
    fn test_sequential_pattern_detection() {
        let mut analyzer = AccessPatternsAnalyzer::new();

        // Create sequential access pattern
        for i in 0..10 {
            let access = create_test_access(i * 64, 64, 100 - i as u64);
            analyzer.record_access(access).unwrap();
        }

        let patterns = analyzer.analyze_patterns().unwrap();
        assert!(!patterns.is_empty());

        match &patterns[0] {
            AccessPattern::Sequential { stride, .. } => {
                assert_eq!(*stride, 64);
            }
            _ => panic!("Expected sequential pattern"),
        }
    }

    #[test]
    fn test_random_pattern_detection() {
        let mut analyzer = AccessPatternsAnalyzer::new();

        // Create random access pattern
        let addresses = vec![1000, 5000, 2000, 8000, 1500, 9000, 3000, 7000, 4000, 6000];
        for (i, &addr) in addresses.iter().enumerate() {
            let access = create_test_access(addr, 64, 100 - i as u64);
            analyzer.record_access(access).unwrap();
        }

        let patterns = analyzer.analyze_patterns().unwrap();

        // Should detect high entropy pattern
        let has_random = patterns.iter().any(|p| matches!(p, AccessPattern::Random { .. }));
        assert!(has_random);
    }

    #[test]
    fn test_metrics_calculation() {
        let mut analyzer = AccessPatternsAnalyzer::new();

        // Create some test accesses
        for i in 0..5 {
            let access = create_test_access(i * 64, 64, 50 - i as u64 * 10);
            analyzer.record_access(access).unwrap();
        }

        let metrics = analyzer.calculate_metrics().unwrap();

        assert!(metrics.cache_hit_rate >= 0.0 && metrics.cache_hit_rate <= 1.0);
        assert!(metrics.bandwidth_utilization >= 0.0);
        assert!(metrics.temporal_locality_score >= 0.0 && metrics.temporal_locality_score <= 1.0);
        assert!(metrics.spatial_locality_score >= 0.0 && metrics.spatial_locality_score <= 1.0);
    }

    #[test]
    fn test_memory_region_tracking() {
        let mut analyzer = AccessPatternsAnalyzer::new();

        // Access same region multiple times
        for _ in 0..5 {
            let access = create_test_access(1000, 64, 10);
            analyzer.record_access(access).unwrap();
        }

        assert_eq!(analyzer.memory_regions.len(), 1);
        assert_eq!(analyzer.memory_regions[0].access_count, 5);
    }

    #[test]
    fn test_recommendation_generation() {
        let mut analyzer = AccessPatternsAnalyzer::new();

        // Create poor access pattern (random)
        let addresses = vec![1000, 5000, 2000, 8000, 1500];
        for (i, &addr) in addresses.iter().enumerate() {
            let access = create_test_access(addr, 64, 50 - i as u64 * 10);
            analyzer.record_access(access).unwrap();
        }

        let recommendations = analyzer.generate_recommendations().unwrap();
        assert!(!recommendations.is_empty());

        // Should have recommendations for random access pattern
        let has_algorithmic = recommendations.iter()
            .any(|r| r.category == OptimizationCategory::Algorithmic);
        assert!(has_algorithmic);
    }

    #[test]
    fn test_temporal_clustering() {
        let mut analyzer = AccessPatternsAnalyzer::new();

        // Create temporal clustering - repeated accesses to same addresses
        let hot_addresses = vec![1000, 2000, 3000];
        for _ in 0..10 {
            for &addr in &hot_addresses {
                let access = create_test_access(addr, 64, 5);
                analyzer.record_access(access).unwrap();
            }
        }

        let patterns = analyzer.analyze_patterns().unwrap();
        let has_temporal = patterns.iter()
            .any(|p| matches!(p, AccessPattern::TemporalCluster { .. }));
        assert!(has_temporal);
    }

    #[test]
    fn test_spatial_clustering() {
        let mut analyzer = AccessPatternsAnalyzer::new();

        // Create spatial clustering - accesses close together
        let base_addr = 10000;
        for i in 0..20 {
            let access = create_test_access(base_addr + i * 8, 8, 20 - i as u64);
            analyzer.record_access(access).unwrap();
        }

        let patterns = analyzer.analyze_patterns().unwrap();
        let has_spatial = patterns.iter()
            .any(|p| matches!(p, AccessPattern::SpatialCluster { .. }));
        assert!(has_spatial);
    }
}