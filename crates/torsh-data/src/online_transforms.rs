//! Online data augmentation system for real-time transform application
//!
//! This module provides advanced online augmentation capabilities for dynamic,
//! adaptive, and performance-aware data transformations during training.
//!
//! # Features
//!
//! - **Real-time augmentation**: OnlineAugmentationEngine for live transform application
//! - **Intelligent caching**: Optional caching system with performance monitoring
//! - **Dynamic strategies**: Epoch-based augmentation strategy switching
//! - **Progressive augmentation**: Gradually increasing augmentation intensity
//! - **Adaptive augmentation**: Performance-based intensity adjustment
//! - **Async processing**: Queue-based augmentation for high-throughput scenarios

use crate::transforms::Transform;
use torsh_core::error::Result;
use torsh_core::error::TorshError;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

#[cfg(feature = "std")]
use scirs2_core::random::{thread_rng, Random, Rng};

#[cfg(not(feature = "std"))]
use scirs2_core::random::{thread_rng, Random, Rng};

/// Online augmentation engine that applies transforms in real-time during data loading
pub struct OnlineAugmentationEngine<T> {
    transform_pipeline: Arc<dyn Transform<T, Output = T> + Send + Sync>,
    cache: Arc<RwLock<HashMap<String, T>>>,
    cache_enabled: bool,
    max_cache_size: usize,
    stats: Arc<RwLock<AugmentationStats>>,
}

/// Statistics for augmentation performance monitoring
#[derive(Debug, Clone)]
pub struct AugmentationStats {
    pub total_transforms: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_time_ms: f64,
    pub average_time_ms: f64,
}

impl Default for AugmentationStats {
    fn default() -> Self {
        Self {
            total_transforms: 0,
            cache_hits: 0,
            cache_misses: 0,
            total_time_ms: 0.0,
            average_time_ms: 0.0,
        }
    }
}

impl<T: Clone + Send + Sync + 'static> OnlineAugmentationEngine<T> {
    /// Create a new online augmentation engine
    pub fn new<P: Transform<T, Output = T> + Send + Sync + 'static>(pipeline: P) -> Self {
        Self {
            transform_pipeline: Arc::new(pipeline),
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_enabled: false,
            max_cache_size: 1000,
            stats: Arc::new(RwLock::new(AugmentationStats::default())),
        }
    }

    /// Enable caching of transformed data
    pub fn with_cache(mut self, max_cache_size: usize) -> Self {
        self.cache_enabled = true;
        self.max_cache_size = max_cache_size;
        self
    }

    /// Apply augmentation with optional caching
    pub fn apply(&self, input: T, cache_key: Option<&str>) -> Result<T> {
        let start_time = Instant::now();

        // Check cache first if enabled
        if self.cache_enabled {
            if let Some(key) = cache_key {
                let cache = self.cache.read().unwrap();
                if let Some(cached_result) = cache.get(key) {
                    self.update_stats(start_time, true);
                    return Ok(cached_result.clone());
                }
            }
        }

        // Apply transformation
        let result = self.transform_pipeline.transform(input)?;

        // Cache result if enabled
        if self.cache_enabled {
            if let Some(key) = cache_key {
                let mut cache = self.cache.write().unwrap();
                if cache.len() < self.max_cache_size {
                    cache.insert(key.to_string(), result.clone());
                }
            }
        }

        self.update_stats(start_time, false);
        Ok(result)
    }

    /// Apply augmentation without caching
    pub fn apply_uncached(&self, input: T) -> Result<T> {
        let start_time = Instant::now();
        let result = self.transform_pipeline.transform(input)?;
        self.update_stats(start_time, false);
        Ok(result)
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        if self.cache_enabled {
            self.cache.write().unwrap().clear();
        }
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        if self.cache_enabled {
            self.cache.read().unwrap().len()
        } else {
            0
        }
    }

    /// Get augmentation statistics
    pub fn stats(&self) -> AugmentationStats {
        self.stats.read().unwrap().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.write().unwrap() = AugmentationStats::default();
    }

    fn update_stats(&self, start_time: Instant, was_cache_hit: bool) {
        let duration = start_time.elapsed().as_secs_f64() * 1000.0;
        let mut stats = self.stats.write().unwrap();

        stats.total_transforms += 1;
        stats.total_time_ms += duration;
        stats.average_time_ms = stats.total_time_ms / stats.total_transforms as f64;

        if was_cache_hit {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
        }
    }
}

/// Dynamic augmentation strategy that can change parameters based on training progress
pub struct DynamicAugmentationStrategy<T> {
    strategies: Vec<StrategyConfig<T>>,
    current_epoch: usize,
    total_epochs: usize,
}

pub struct StrategyConfig<T> {
    epoch_range: (usize, usize),
    pipeline: Arc<dyn Transform<T, Output = T> + Send + Sync>,
    weight: f32,
}

impl<T: Clone + Send + Sync + 'static> DynamicAugmentationStrategy<T> {
    /// Create a new dynamic augmentation strategy
    pub fn new(total_epochs: usize) -> Self {
        Self {
            strategies: Vec::new(),
            current_epoch: 0,
            total_epochs,
        }
    }

    /// Add a strategy for specific epoch range
    pub fn add_strategy<P: Transform<T, Output = T> + Send + Sync + 'static>(
        mut self,
        epoch_start: usize,
        epoch_end: usize,
        pipeline: P,
        weight: f32,
    ) -> Self {
        self.strategies.push(StrategyConfig {
            epoch_range: (epoch_start, epoch_end),
            pipeline: Arc::new(pipeline),
            weight,
        });
        self
    }

    /// Set current epoch
    pub fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    /// Get current active strategies for the current epoch
    fn get_active_strategies(&self) -> Vec<&StrategyConfig<T>> {
        self.strategies
            .iter()
            .filter(|config| {
                self.current_epoch >= config.epoch_range.0
                    && self.current_epoch <= config.epoch_range.1
            })
            .collect()
    }

    /// Get training progress as a value between 0.0 and 1.0
    pub fn get_progress(&self) -> f32 {
        if self.total_epochs == 0 {
            0.0
        } else {
            (self.current_epoch as f32 / self.total_epochs as f32).min(1.0)
        }
    }

    /// Apply augmentation based on current epoch
    pub fn apply(&self, input: T) -> Result<T> {
        let active_strategies = self.get_active_strategies();

        if active_strategies.is_empty() {
            return Ok(input);
        }

        // Select strategy based on weights
        let total_weight: f32 = active_strategies.iter().map(|s| s.weight).sum();
        if total_weight == 0.0 {
            return Ok(input);
        }

        let mut rng = thread_rng();
        let random_value = rng.gen::<f32>() * total_weight;
        let mut cumulative_weight = 0.0;

        for strategy in &active_strategies {
            cumulative_weight += strategy.weight;
            if random_value <= cumulative_weight {
                return strategy.pipeline.transform(input);
            }
        }

        // Fallback to last strategy
        if let Some(last_strategy) = active_strategies.last() {
            last_strategy.pipeline.transform(input)
        } else {
            Ok(input)
        }
    }
}

/// Progressive augmentation that gradually increases intensity
#[derive(Clone)]
pub struct ProgressiveAugmentation<T> {
    light_pipeline: Arc<dyn Transform<T, Output = T> + Send + Sync>,
    medium_pipeline: Arc<dyn Transform<T, Output = T> + Send + Sync>,
    heavy_pipeline: Arc<dyn Transform<T, Output = T> + Send + Sync>,
    current_epoch: usize,
    total_epochs: usize,
    progression_mode: ProgressionMode,
}

#[derive(Clone, Copy)]
pub enum ProgressionMode {
    Linear,
    Exponential,
    StepWise,
}

impl<T: Clone + Send + Sync + 'static> ProgressiveAugmentation<T> {
    /// Create progressive augmentation with different intensity levels
    pub fn new<L, M, H>(light: L, medium: M, heavy: H, total_epochs: usize) -> Self
    where
        L: Transform<T, Output = T> + Send + Sync + 'static,
        M: Transform<T, Output = T> + Send + Sync + 'static,
        H: Transform<T, Output = T> + Send + Sync + 'static,
    {
        Self {
            light_pipeline: Arc::new(light),
            medium_pipeline: Arc::new(medium),
            heavy_pipeline: Arc::new(heavy),
            current_epoch: 0,
            total_epochs,
            progression_mode: ProgressionMode::Linear,
        }
    }

    /// Set progression mode
    pub fn with_progression_mode(mut self, mode: ProgressionMode) -> Self {
        self.progression_mode = mode;
        self
    }

    /// Set current epoch
    pub fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    /// Calculate current intensity based on epoch and progression mode
    fn calculate_intensity(&self) -> f32 {
        if self.total_epochs == 0 {
            return 0.0;
        }

        let progress = self.current_epoch as f32 / self.total_epochs as f32;
        let progress = progress.min(1.0);

        match self.progression_mode {
            ProgressionMode::Linear => progress,
            ProgressionMode::Exponential => progress * progress,
            ProgressionMode::StepWise => {
                if progress < 0.33 {
                    0.0
                } else if progress < 0.66 {
                    0.5
                } else {
                    1.0
                }
            }
        }
    }

    /// Apply progressive augmentation
    pub fn apply(&self, input: T) -> Result<T> {
        let intensity = self.calculate_intensity();

        if intensity < 0.33 {
            self.light_pipeline.transform(input)
        } else if intensity < 0.66 {
            self.medium_pipeline.transform(input)
        } else {
            self.heavy_pipeline.transform(input)
        }
    }
}

/// Adaptive augmentation that adjusts based on model performance
pub struct AdaptiveAugmentation<T> {
    pipelines: Vec<(Arc<dyn Transform<T, Output = T> + Send + Sync>, f32)>, // (pipeline, intensity)
    performance_history: Vec<f32>,
    target_performance: f32,
    adaptation_rate: f32,
    current_intensity: f32,
    min_intensity: f32,
    max_intensity: f32,
}

impl<T: Clone + Send + Sync + 'static> AdaptiveAugmentation<T> {
    /// Create adaptive augmentation system
    pub fn new(target_performance: f32) -> Self {
        Self {
            pipelines: Vec::new(),
            performance_history: Vec::new(),
            target_performance,
            adaptation_rate: 0.1,
            current_intensity: 0.5,
            min_intensity: 0.0,
            max_intensity: 1.0,
        }
    }

    /// Add augmentation pipeline with intensity level
    pub fn add_pipeline<P: Transform<T, Output = T> + Send + Sync + 'static>(
        mut self,
        pipeline: P,
        intensity: f32,
    ) -> Self {
        self.pipelines.push((Arc::new(pipeline), intensity));
        self
    }

    /// Set adaptation parameters
    pub fn with_adaptation_params(
        mut self,
        adaptation_rate: f32,
        min_intensity: f32,
        max_intensity: f32,
    ) -> Self {
        self.adaptation_rate = adaptation_rate;
        self.min_intensity = min_intensity;
        self.max_intensity = max_intensity;
        self
    }

    /// Update with current model performance (e.g., validation accuracy)
    pub fn update_performance(&mut self, performance: f32) {
        self.performance_history.push(performance);

        // Keep only recent history
        if self.performance_history.len() > 10 {
            self.performance_history.remove(0);
        }

        // Adapt intensity based on performance
        if performance < self.target_performance {
            // Performance is low, reduce augmentation intensity
            self.current_intensity -= self.adaptation_rate;
        } else {
            // Performance is good, can increase augmentation intensity
            self.current_intensity += self.adaptation_rate;
        }

        // Clamp intensity
        self.current_intensity = self
            .current_intensity
            .max(self.min_intensity)
            .min(self.max_intensity);
    }

    /// Apply adaptive augmentation
    pub fn apply(&self, input: T) -> Result<T> {
        if self.pipelines.is_empty() {
            return Ok(input);
        }

        // Find pipeline with intensity closest to current intensity
        let mut best_pipeline = &self.pipelines[0].0;
        let mut best_distance = (self.pipelines[0].1 - self.current_intensity).abs();

        for (pipeline, intensity) in &self.pipelines {
            let distance = (intensity - self.current_intensity).abs();
            if distance < best_distance {
                best_distance = distance;
                best_pipeline = pipeline;
            }
        }

        best_pipeline.transform(input)
    }

    /// Get current intensity level
    pub fn current_intensity(&self) -> f32 {
        self.current_intensity
    }

    /// Get recent performance average
    pub fn recent_performance(&self) -> Option<f32> {
        if self.performance_history.is_empty() {
            None
        } else {
            let sum: f32 = self.performance_history.iter().sum();
            Some(sum / self.performance_history.len() as f32)
        }
    }
}

/// Simple message-passing based augmentation queue for async processing
/// (Simplified version without external dependencies)
pub struct AugmentationQueue<T> {
    tasks: Arc<RwLock<Vec<AugmentationTask<T>>>>,
    engine: Arc<OnlineAugmentationEngine<T>>,
    max_queue_size: usize,
}

struct AugmentationTask<T> {
    input: T,
    cache_key: Option<String>,
    #[allow(dead_code)]
    task_id: usize,
}

impl<T: Clone + Send + Sync + 'static> AugmentationQueue<T> {
    /// Create a new augmentation queue
    pub fn new(engine: OnlineAugmentationEngine<T>, max_queue_size: usize) -> Self {
        Self {
            tasks: Arc::new(RwLock::new(Vec::new())),
            engine: Arc::new(engine),
            max_queue_size,
        }
    }

    /// Submit an augmentation task (simplified version)
    pub fn submit(&self, input: T, cache_key: Option<String>) -> Result<T> {
        let tasks = self.tasks.read().unwrap();
        if tasks.len() >= self.max_queue_size {
            return Err(TorshError::InvalidArgument(
                "Augmentation queue is full".to_string(),
            ));
        }
        drop(tasks);

        // For this simplified version, process immediately
        self.engine.apply(input, cache_key.as_deref())
    }

    /// Process pending tasks (placeholder for worker thread processing)
    pub fn process_tasks(&self) -> usize {
        let mut tasks = self.tasks.write().unwrap();
        let processed_count = tasks.len();

        // In a real implementation, this would process tasks asynchronously
        // For now, we just clear the task list
        tasks.clear();

        processed_count
    }

    /// Get queue length
    pub fn queue_length(&self) -> usize {
        self.tasks.read().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::lambda;

    #[test]
    fn test_augmentation_stats_default() {
        let stats = AugmentationStats::default();
        assert_eq!(stats.total_transforms, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.total_time_ms, 0.0);
        assert_eq!(stats.average_time_ms, 0.0);
    }

    #[test]
    fn test_online_augmentation_engine_creation() {
        let transform = lambda(|x: i32| Ok(x * 2));
        let engine = OnlineAugmentationEngine::new(transform);

        assert!(!engine.cache_enabled);
        assert_eq!(engine.max_cache_size, 1000);
        assert_eq!(engine.cache_size(), 0);
    }

    #[test]
    fn test_online_augmentation_engine_with_cache() {
        let transform = lambda(|x: i32| Ok(x * 2));
        let engine = OnlineAugmentationEngine::new(transform).with_cache(500);

        assert!(engine.cache_enabled);
        assert_eq!(engine.max_cache_size, 500);
    }

    #[test]
    fn test_dynamic_augmentation_strategy() {
        let strategy = DynamicAugmentationStrategy::<i32>::new(100);
        assert_eq!(strategy.current_epoch, 0);
        assert_eq!(strategy.total_epochs, 100);
        assert_eq!(strategy.get_progress(), 0.0);
    }

    #[test]
    fn test_dynamic_strategy_progress() {
        let mut strategy = DynamicAugmentationStrategy::<i32>::new(100);
        strategy.set_epoch(25);
        assert_eq!(strategy.get_progress(), 0.25);

        strategy.set_epoch(50);
        assert_eq!(strategy.get_progress(), 0.5);

        strategy.set_epoch(100);
        assert_eq!(strategy.get_progress(), 1.0);
    }

    #[test]
    fn test_progressive_augmentation_intensity() {
        let light = lambda(|x: i32| Ok(x + 1));
        let medium = lambda(|x: i32| Ok(x + 2));
        let heavy = lambda(|x: i32| Ok(x + 3));

        let mut progressive = ProgressiveAugmentation::new(light, medium, heavy, 100);

        // Test different epochs
        progressive.set_epoch(0);
        assert_eq!(progressive.calculate_intensity(), 0.0);

        progressive.set_epoch(25);
        assert_eq!(progressive.calculate_intensity(), 0.25);

        progressive.set_epoch(50);
        assert_eq!(progressive.calculate_intensity(), 0.5);

        progressive.set_epoch(100);
        assert_eq!(progressive.calculate_intensity(), 1.0);
    }

    #[test]
    fn test_progressive_augmentation_step_wise() {
        let light = lambda(|x: i32| Ok(x + 1));
        let medium = lambda(|x: i32| Ok(x + 2));
        let heavy = lambda(|x: i32| Ok(x + 3));

        let mut progressive = ProgressiveAugmentation::new(light, medium, heavy, 100)
            .with_progression_mode(ProgressionMode::StepWise);

        progressive.set_epoch(10);
        assert_eq!(progressive.calculate_intensity(), 0.0);

        progressive.set_epoch(40);
        assert_eq!(progressive.calculate_intensity(), 0.5);

        progressive.set_epoch(80);
        assert_eq!(progressive.calculate_intensity(), 1.0);
    }

    #[test]
    fn test_adaptive_augmentation() {
        let mut adaptive = AdaptiveAugmentation::<i32>::new(0.85);
        assert_eq!(adaptive.current_intensity(), 0.5);
        assert_eq!(adaptive.recent_performance(), None);

        // Update with good performance
        adaptive.update_performance(0.9);
        assert!(adaptive.current_intensity() > 0.5);
        assert_eq!(adaptive.recent_performance(), Some(0.9));

        // Update with poor performance
        adaptive.update_performance(0.7);
        assert!(adaptive.current_intensity() < 0.6);
    }

    #[test]
    fn test_augmentation_queue() {
        let transform = lambda(|x: i32| Ok(x * 2));
        let engine = OnlineAugmentationEngine::new(transform);
        let queue = AugmentationQueue::new(engine, 10);

        assert_eq!(queue.queue_length(), 0);

        let result = queue.submit(5, None).unwrap();
        assert_eq!(result, 10);
    }
}
