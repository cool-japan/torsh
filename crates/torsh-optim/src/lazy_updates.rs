use crate::OptimizerError;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Configuration for lazy parameter updates
#[derive(Debug, Clone)]
pub struct LazyUpdateConfig {
    /// Maximum delay before forcing an update (in milliseconds)
    pub max_delay_ms: u64,
    /// Minimum gradient magnitude to trigger an update
    pub gradient_threshold: f32,
    /// Maximum number of pending updates before forcing a batch update
    pub max_pending_updates: usize,
    /// Whether to use adaptive thresholds based on parameter history
    pub adaptive_threshold: bool,
    /// Update frequency based on parameter importance
    pub importance_based_updates: bool,
    /// Batch size for processing updates
    pub batch_size: usize,
}

impl Default for LazyUpdateConfig {
    fn default() -> Self {
        Self {
            max_delay_ms: 1000, // 1 second max delay
            gradient_threshold: 1e-6,
            max_pending_updates: 1000,
            adaptive_threshold: true,
            importance_based_updates: true,
            batch_size: 100,
        }
    }
}

/// Pending parameter update
#[derive(Debug, Clone)]
pub struct PendingUpdate {
    pub parameter_id: String,
    pub gradient: Vec<f32>,
    pub timestamp: Instant,
    pub priority: UpdatePriority,
    pub accumulated_magnitude: f32,
}

/// Update priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum UpdatePriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Parameter importance tracking
#[derive(Debug, Clone)]
pub struct ParameterImportance {
    pub parameter_id: String,
    pub update_frequency: f32,
    pub average_gradient_magnitude: f32,
    pub variance: f32,
    pub last_significant_update: Instant,
    pub importance_score: f32,
}

impl ParameterImportance {
    pub fn new(parameter_id: String) -> Self {
        Self {
            parameter_id,
            update_frequency: 0.0,
            average_gradient_magnitude: 0.0,
            variance: 0.0,
            last_significant_update: Instant::now(),
            importance_score: 1.0,
        }
    }

    pub fn update(&mut self, gradient_magnitude: f32, is_significant: bool) {
        // Update running averages using exponential moving average
        let alpha = 0.1; // Learning rate for moving averages

        self.average_gradient_magnitude =
            (1.0 - alpha) * self.average_gradient_magnitude + alpha * gradient_magnitude;

        let variance_delta = (gradient_magnitude - self.average_gradient_magnitude).powi(2);
        self.variance = (1.0 - alpha) * self.variance + alpha * variance_delta;

        if is_significant {
            self.last_significant_update = Instant::now();
            self.update_frequency = (1.0 - alpha) * self.update_frequency + alpha;
        } else {
            self.update_frequency = (1.0 - alpha) * self.update_frequency;
        }

        // Calculate importance score based on multiple factors
        let recency_factor =
            1.0 / (1.0 + self.last_significant_update.elapsed().as_secs_f32() / 60.0);
        let magnitude_factor =
            self.average_gradient_magnitude / (1e-6 + self.average_gradient_magnitude);
        let frequency_factor = self.update_frequency;
        let stability_factor = 1.0 / (1.0 + self.variance);

        self.importance_score =
            recency_factor * magnitude_factor * frequency_factor * stability_factor;
    }
}

/// Lazy parameter update manager
pub struct LazyUpdateManager {
    config: LazyUpdateConfig,
    pending_updates: VecDeque<PendingUpdate>,
    parameter_importance: HashMap<String, ParameterImportance>,
    adaptive_thresholds: HashMap<String, f32>,
    last_batch_update: Instant,
    total_updates_processed: usize,
    total_updates_skipped: usize,
}

impl LazyUpdateManager {
    /// Create a new lazy update manager
    pub fn new(config: LazyUpdateConfig) -> Self {
        Self {
            config,
            pending_updates: VecDeque::new(),
            parameter_importance: HashMap::new(),
            adaptive_thresholds: HashMap::new(),
            last_batch_update: Instant::now(),
            total_updates_processed: 0,
            total_updates_skipped: 0,
        }
    }

    /// Submit a gradient for potential lazy update
    pub fn submit_gradient(
        &mut self,
        parameter_id: String,
        gradient: Vec<f32>,
    ) -> LazyUpdateDecision {
        let gradient_magnitude = gradient.iter().map(|&x| x * x).sum::<f32>().sqrt();

        // Get threshold
        let threshold = self.get_threshold(&parameter_id);
        let is_significant = gradient_magnitude > threshold;

        // Get or create parameter importance tracking and update it
        {
            let importance = self
                .parameter_importance
                .entry(parameter_id.clone())
                .or_insert_with(|| ParameterImportance::new(parameter_id.clone()));
            importance.update(gradient_magnitude, is_significant);
        }

        // Now calculate priority with immutable borrow
        let priority = {
            let importance = &self.parameter_importance[&parameter_id];
            self.calculate_priority(&parameter_id, gradient_magnitude, importance)
        };

        // Check if we should update immediately
        if self.should_update_immediately(&parameter_id, gradient_magnitude, priority) {
            self.total_updates_processed += 1;
            return LazyUpdateDecision::UpdateNow;
        }

        // Add to pending updates
        let pending_update = PendingUpdate {
            parameter_id: parameter_id.clone(),
            gradient,
            timestamp: Instant::now(),
            priority,
            accumulated_magnitude: gradient_magnitude,
        };

        self.insert_pending_update(pending_update);

        // Check if we need to process batch updates
        if self.should_process_batch() {
            LazyUpdateDecision::ProcessBatch(self.get_batch_updates())
        } else {
            self.total_updates_skipped += 1;
            LazyUpdateDecision::Defer
        }
    }

    /// Get the next batch of updates to process
    pub fn get_batch_updates(&mut self) -> Vec<PendingUpdate> {
        let mut batch = Vec::new();
        let batch_size = self.config.batch_size.min(self.pending_updates.len());

        // Sort by priority and timestamp
        let mut updates: Vec<_> = self.pending_updates.drain(..).collect();
        updates.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.timestamp.cmp(&b.timestamp))
        });

        // Take the highest priority updates
        for update in updates.into_iter().take(batch_size) {
            batch.push(update);
        }

        self.last_batch_update = Instant::now();
        self.total_updates_processed += batch.len();

        batch
    }

    /// Force processing of all pending updates
    pub fn flush_all_updates(&mut self) -> Vec<PendingUpdate> {
        let updates: Vec<_> = self.pending_updates.drain(..).collect();
        self.total_updates_processed += updates.len();
        updates
    }

    /// Get updates that have exceeded the maximum delay
    pub fn get_expired_updates(&mut self) -> Vec<PendingUpdate> {
        let max_delay = Duration::from_millis(self.config.max_delay_ms);
        let now = Instant::now();

        let mut expired = Vec::new();
        let mut remaining = VecDeque::new();

        while let Some(update) = self.pending_updates.pop_front() {
            if update.timestamp.elapsed() > max_delay {
                expired.push(update);
            } else {
                remaining.push_back(update);
            }
        }

        self.pending_updates = remaining;
        self.total_updates_processed += expired.len();

        expired
    }

    /// Get statistics about lazy updates
    pub fn statistics(&self) -> LazyUpdateStatistics {
        let total_parameters = self.parameter_importance.len();
        let pending_count = self.pending_updates.len();

        let average_importance = if total_parameters > 0 {
            self.parameter_importance
                .values()
                .map(|imp| imp.importance_score)
                .sum::<f32>()
                / total_parameters as f32
        } else {
            0.0
        };

        let high_priority_pending = self
            .pending_updates
            .iter()
            .filter(|update| update.priority >= UpdatePriority::High)
            .count();

        LazyUpdateStatistics {
            total_parameters,
            pending_updates: pending_count,
            high_priority_pending,
            total_processed: self.total_updates_processed,
            total_skipped: self.total_updates_skipped,
            skip_ratio: if self.total_updates_processed + self.total_updates_skipped > 0 {
                self.total_updates_skipped as f32
                    / (self.total_updates_processed + self.total_updates_skipped) as f32
            } else {
                0.0
            },
            average_importance,
        }
    }

    /// Get parameter importance information
    pub fn get_parameter_importance(&self, parameter_id: &str) -> Option<&ParameterImportance> {
        self.parameter_importance.get(parameter_id)
    }

    /// Set custom threshold for a parameter
    pub fn set_parameter_threshold(&mut self, parameter_id: String, threshold: f32) {
        self.adaptive_thresholds.insert(parameter_id, threshold);
    }

    /// Clear statistics
    pub fn reset_statistics(&mut self) {
        self.total_updates_processed = 0;
        self.total_updates_skipped = 0;
    }

    // Private methods

    fn get_threshold(&self, parameter_id: &str) -> f32 {
        if let Some(&custom_threshold) = self.adaptive_thresholds.get(parameter_id) {
            return custom_threshold;
        }

        if !self.config.adaptive_threshold {
            return self.config.gradient_threshold;
        }

        if let Some(importance) = self.parameter_importance.get(parameter_id) {
            // Adaptive threshold based on parameter history
            let base_threshold = self.config.gradient_threshold;
            let variance_factor = (1.0 + importance.variance).sqrt();
            let importance_factor = 1.0 / (1.0 + importance.importance_score);

            base_threshold * variance_factor * importance_factor
        } else {
            self.config.gradient_threshold
        }
    }

    fn calculate_priority(
        &self,
        parameter_id: &str,
        gradient_magnitude: f32,
        importance: &ParameterImportance,
    ) -> UpdatePriority {
        if !self.config.importance_based_updates {
            return UpdatePriority::Medium;
        }

        let threshold = self.get_threshold(parameter_id);
        let magnitude_ratio = gradient_magnitude / threshold;

        let importance_factor = importance.importance_score;
        let time_factor = importance.last_significant_update.elapsed().as_secs_f32() / 60.0; // minutes

        let priority_score = magnitude_ratio * importance_factor * (1.0 + time_factor);

        if priority_score > 10.0 {
            UpdatePriority::Critical
        } else if priority_score > 3.0 {
            UpdatePriority::High
        } else if priority_score > 1.0 {
            UpdatePriority::Medium
        } else {
            UpdatePriority::Low
        }
    }

    fn should_update_immediately(
        &self,
        _parameter_id: &str,
        gradient_magnitude: f32,
        priority: UpdatePriority,
    ) -> bool {
        // Always update critical priority immediately
        if priority == UpdatePriority::Critical {
            return true;
        }

        // Update if gradient is significantly large (10x the normal threshold)
        if gradient_magnitude > self.config.gradient_threshold * 10.0 {
            return true;
        }

        false
    }

    fn should_process_batch(&self) -> bool {
        // Process batch if we have too many pending updates
        if self.pending_updates.len() >= self.config.max_pending_updates {
            return true;
        }

        // Process batch if we have high priority updates
        let high_priority_count = self
            .pending_updates
            .iter()
            .filter(|update| update.priority >= UpdatePriority::High)
            .count();

        if high_priority_count >= self.config.batch_size / 2 {
            return true;
        }

        // Process batch if maximum delay is reached
        let max_delay = Duration::from_millis(self.config.max_delay_ms);
        if let Some(oldest) = self.pending_updates.front() {
            if oldest.timestamp.elapsed() > max_delay {
                return true;
            }
        }

        false
    }

    fn insert_pending_update(&mut self, update: PendingUpdate) {
        // Insert in priority order
        let insert_pos = self
            .pending_updates
            .iter()
            .position(|existing| existing.priority < update.priority)
            .unwrap_or(self.pending_updates.len());

        self.pending_updates.insert(insert_pos, update);
    }
}

/// Decision for lazy update processing
#[derive(Debug)]
pub enum LazyUpdateDecision {
    /// Update the parameter immediately
    UpdateNow,
    /// Defer the update for later
    Defer,
    /// Process a batch of updates
    ProcessBatch(Vec<PendingUpdate>),
}

/// Statistics for lazy update system
#[derive(Debug, Clone)]
pub struct LazyUpdateStatistics {
    pub total_parameters: usize,
    pub pending_updates: usize,
    pub high_priority_pending: usize,
    pub total_processed: usize,
    pub total_skipped: usize,
    pub skip_ratio: f32,
    pub average_importance: f32,
}

impl std::fmt::Display for LazyUpdateStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Lazy Update Statistics:")?;
        writeln!(f, "  Total Parameters: {}", self.total_parameters)?;
        writeln!(f, "  Pending Updates: {}", self.pending_updates)?;
        writeln!(f, "  High Priority Pending: {}", self.high_priority_pending)?;
        writeln!(f, "  Total Processed: {}", self.total_processed)?;
        writeln!(f, "  Total Skipped: {}", self.total_skipped)?;
        writeln!(f, "  Skip Ratio: {:.1}%", self.skip_ratio * 100.0)?;
        writeln!(f, "  Average Importance: {:.3}", self.average_importance)?;
        Ok(())
    }
}

/// Trait for optimizers that support lazy updates
pub trait LazyUpdateSupport {
    /// Apply a single parameter update
    fn apply_update(&mut self, parameter_id: &str, gradient: &[f32]) -> Result<(), OptimizerError>;

    /// Apply a batch of parameter updates
    fn apply_batch_updates(&mut self, updates: &[PendingUpdate]) -> Result<(), OptimizerError> {
        for update in updates {
            self.apply_update(&update.parameter_id, &update.gradient)?;
        }
        Ok(())
    }

    /// Get the current gradient for a parameter
    fn get_parameter_gradient(&self, parameter_id: &str) -> Option<Vec<f32>>;

    /// Get list of all parameter IDs
    fn get_parameter_ids(&self) -> Vec<String>;
}

/// Wrapper optimizer that adds lazy update functionality
pub struct LazyUpdateOptimizer<T> {
    inner: T,
    lazy_manager: LazyUpdateManager,
    enabled: bool,
}

impl<T> LazyUpdateOptimizer<T>
where
    T: LazyUpdateSupport,
{
    /// Create a new lazy update optimizer wrapper
    pub fn new(inner: T, config: LazyUpdateConfig) -> Self {
        Self {
            inner,
            lazy_manager: LazyUpdateManager::new(config),
            enabled: true,
        }
    }

    /// Enable or disable lazy updates
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;

        if !enabled {
            // Process all pending updates immediately
            let pending = self.lazy_manager.flush_all_updates();
            let _ = self.inner.apply_batch_updates(&pending);
        }
    }

    /// Get the inner optimizer
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get the inner optimizer mutably
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Get the lazy update manager
    pub fn lazy_manager(&self) -> &LazyUpdateManager {
        &self.lazy_manager
    }

    /// Get the lazy update manager mutably
    pub fn lazy_manager_mut(&mut self) -> &mut LazyUpdateManager {
        &mut self.lazy_manager
    }

    /// Submit gradients for lazy processing
    pub fn submit_gradients(
        &mut self,
        gradients: HashMap<String, Vec<f32>>,
    ) -> Result<(), OptimizerError> {
        if !self.enabled {
            // Apply all gradients immediately
            for (param_id, gradient) in gradients {
                self.inner.apply_update(&param_id, &gradient)?;
            }
            return Ok(());
        }

        // Process each gradient through the lazy update system
        for (param_id, gradient) in gradients {
            match self.lazy_manager.submit_gradient(param_id, gradient) {
                LazyUpdateDecision::UpdateNow => {
                    // This shouldn't happen in practice since we already processed the gradient
                }
                LazyUpdateDecision::Defer => {
                    // Nothing to do, update is deferred
                }
                LazyUpdateDecision::ProcessBatch(updates) => {
                    self.inner.apply_batch_updates(&updates)?;
                }
            }
        }

        // Check for expired updates
        let expired = self.lazy_manager.get_expired_updates();
        if !expired.is_empty() {
            self.inner.apply_batch_updates(&expired)?;
        }

        Ok(())
    }

    /// Force processing of all pending updates
    pub fn flush_pending_updates(&mut self) -> Result<(), OptimizerError> {
        let pending = self.lazy_manager.flush_all_updates();
        self.inner.apply_batch_updates(&pending)
    }

    /// Get statistics about lazy updates
    pub fn statistics(&self) -> LazyUpdateStatistics {
        self.lazy_manager.statistics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MockOptimizer {
        parameters: HashMap<String, Vec<f32>>,
        gradients: HashMap<String, Vec<f32>>,
        update_count: usize,
    }

    impl MockOptimizer {
        fn new() -> Self {
            let mut parameters = HashMap::new();
            parameters.insert("layer1.weight".to_string(), vec![1.0, 2.0, 3.0]);
            parameters.insert("layer1.bias".to_string(), vec![0.1, 0.2]);

            Self {
                parameters,
                gradients: HashMap::new(),
                update_count: 0,
            }
        }
    }

    impl LazyUpdateSupport for MockOptimizer {
        fn apply_update(
            &mut self,
            parameter_id: &str,
            gradient: &[f32],
        ) -> Result<(), OptimizerError> {
            self.gradients
                .insert(parameter_id.to_string(), gradient.to_vec());
            self.update_count += 1;
            Ok(())
        }

        fn get_parameter_gradient(&self, parameter_id: &str) -> Option<Vec<f32>> {
            self.gradients.get(parameter_id).cloned()
        }

        fn get_parameter_ids(&self) -> Vec<String> {
            self.parameters.keys().cloned().collect()
        }
    }

    #[test]
    fn test_lazy_update_manager() {
        let config = LazyUpdateConfig {
            gradient_threshold: 0.1,
            max_pending_updates: 5,
            ..Default::default()
        };

        let mut manager = LazyUpdateManager::new(config);

        // Submit a small gradient (should be deferred)
        let decision = manager.submit_gradient("param1".to_string(), vec![0.01, 0.02]);
        assert!(matches!(decision, LazyUpdateDecision::Defer));

        // Submit a large gradient (should trigger immediate update)
        let decision = manager.submit_gradient("param2".to_string(), vec![1.0, 2.0]);
        assert!(matches!(decision, LazyUpdateDecision::UpdateNow));

        // Submit multiple small gradients to trigger batch processing
        for i in 0..6 {
            let decision = manager.submit_gradient(format!("param{}", i), vec![0.05, 0.06]);
            if i == 3 {
                // After adding param3, we have 5 pending updates (max_pending_updates),
                // so this should trigger batch processing
                assert!(matches!(decision, LazyUpdateDecision::ProcessBatch(_)));
            } else if i < 3 {
                // Before reaching the batch threshold, updates should be deferred
                assert!(matches!(decision, LazyUpdateDecision::Defer));
            }
            // After i=3, we start accumulating again, so i=4 and i=5 should defer
        }
    }

    #[test]
    fn test_lazy_update_optimizer() {
        let config = LazyUpdateConfig {
            gradient_threshold: 0.1,
            max_pending_updates: 3,
            ..Default::default()
        };

        let optimizer = MockOptimizer::new();
        let mut lazy_optimizer = LazyUpdateOptimizer::new(optimizer, config);

        // Submit gradients
        let mut gradients = HashMap::new();
        gradients.insert("layer1.weight".to_string(), vec![0.01, 0.02, 0.03]);
        gradients.insert("layer1.bias".to_string(), vec![0.05, 0.06]);

        lazy_optimizer.submit_gradients(gradients).unwrap();

        // Check that not all updates were applied immediately
        assert!(lazy_optimizer.inner().update_count < 2);

        // Flush pending updates
        lazy_optimizer.flush_pending_updates().unwrap();

        // Now all updates should be applied
        assert!(lazy_optimizer.inner().update_count >= 2);
    }

    #[test]
    fn test_parameter_importance() {
        let mut importance = ParameterImportance::new("test_param".to_string());

        // Update with various gradient magnitudes
        importance.update(0.5, true);
        assert!(importance.importance_score > 0.0);

        importance.update(0.1, false);
        importance.update(0.8, true);

        // Importance score should reflect the update pattern
        assert!(importance.average_gradient_magnitude > 0.0);
        assert!(importance.update_frequency > 0.0);
    }
}
