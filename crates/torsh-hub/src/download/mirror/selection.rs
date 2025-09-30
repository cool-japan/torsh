//! Mirror Selection Algorithms and Strategies
//!
//! This module provides sophisticated mirror selection algorithms including latency-based,
//! reliability-based, geographic, weighted, adaptive, and machine learning approaches.
//! It manages selection state and provides intelligent mirror ranking for optimal downloads.

use super::types::*;
use super::types::{GeographicCalculator, PerformanceAnalyzer};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use torsh_core::error::{Result, TorshError};

// ================================================================================================
// Mirror Selection State Implementation
// ================================================================================================

impl MirrorSelectionState {
    /// Create a new mirror selection state
    pub fn new() -> Self {
        Self {
            round_robin_index: 0,
            last_benchmark: 0,
            adaptive_weights: MirrorWeights::default(),
            selection_history: Vec::new(),
            ml_model_state: None,
        }
    }

    /// Create a new selection state with custom adaptive weights
    pub fn with_adaptive_weights(weights: MirrorWeights) -> Self {
        Self {
            round_robin_index: 0,
            last_benchmark: 0,
            adaptive_weights: weights,
            selection_history: Vec::new(),
            ml_model_state: None,
        }
    }

    /// Record a selection decision for adaptive learning
    pub fn record_selection(
        &mut self,
        mirror_id: &str,
        strategy: MirrorSelectionStrategy,
        success: bool,
        performance_score: f64,
    ) {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let record = SelectionRecord {
            timestamp: current_time,
            mirror_id: mirror_id.to_string(),
            strategy_used: strategy,
            response_time: None, // Not available in this context
            success,
            performance_score: if success && !performance_score.is_infinite() {
                Some(performance_score)
            } else {
                None
            },
        };

        self.selection_history.push(record);

        // Limit history size to prevent unbounded growth
        if self.selection_history.len() > 1000 {
            self.selection_history.drain(0..100); // Remove oldest 100 entries
        }
    }

    /// Get the next round-robin index
    pub fn get_next_round_robin_index(&mut self, mirror_count: usize) -> usize {
        if mirror_count == 0 {
            return 0;
        }

        let index = self.round_robin_index % mirror_count;
        self.round_robin_index += 1;
        index
    }

    /// Update adaptive weights based on selection history and performance
    pub fn update_adaptive_weights(&mut self) {
        if self.selection_history.len() < 10 {
            return; // Not enough data for adaptation
        }

        // Analyze recent selection history to adjust weights
        let recent_selections: Vec<&SelectionRecord> = self
            .selection_history
            .iter()
            .rev()
            .take(100) // Last 100 selections
            .collect();

        let success_rate = recent_selections.iter().filter(|s| s.success).count() as f64
            / recent_selections.len() as f64;

        // Calculate average response times for successful selections
        let successful_selections: Vec<&SelectionRecord> = recent_selections
            .into_iter()
            .filter(|s| s.success && s.response_time.is_some())
            .collect();

        let avg_response_time = if !successful_selections.is_empty() {
            successful_selections
                .iter()
                .filter_map(|s| s.response_time)
                .map(|rt| rt as f64)
                .sum::<f64>()
                / successful_selections.len() as f64
        } else {
            1000.0 // Default high latency if no successful selections
        };

        // Adjust weights based on success rate and performance trends
        let mut new_weights = self.adaptive_weights.clone();

        if success_rate < 0.8 {
            // Low success rate: increase reliability weight, decrease latency weight
            new_weights.reliability = (new_weights.reliability + 0.05).min(0.6);
            new_weights.latency = (new_weights.latency - 0.02).max(0.1);
        } else if success_rate > 0.95 {
            // High success rate: can optimize for performance
            new_weights.latency = (new_weights.latency + 0.02).min(0.4);
            new_weights.load = (new_weights.load + 0.02).min(0.3);
        }

        if avg_response_time > 2000.0 {
            // High latency: prioritize latency optimization
            new_weights.latency = (new_weights.latency + 0.05).min(0.5);
            new_weights.geographic = (new_weights.geographic + 0.02).min(0.3);
        }

        // Normalize weights to sum to 1.0
        let total_weight = new_weights.latency
            + new_weights.reliability
            + new_weights.load
            + new_weights.geographic
            + new_weights.bandwidth
            + new_weights.provider_quality;

        if total_weight > 0.0 {
            new_weights.latency /= total_weight;
            new_weights.reliability /= total_weight;
            new_weights.load /= total_weight;
            new_weights.geographic /= total_weight;
            new_weights.bandwidth /= total_weight;
            new_weights.provider_quality /= total_weight;
        }

        self.adaptive_weights = new_weights;
    }

    /// Get selection statistics for analysis
    pub fn get_selection_statistics(&self) -> SelectionStatistics {
        if self.selection_history.is_empty() {
            return SelectionStatistics::default();
        }

        let total_selections = self.selection_history.len();
        let successful_selections = self.selection_history.iter().filter(|s| s.success).count();
        let success_rate = successful_selections as f64 / total_selections as f64;

        let successful_response_times: Vec<f64> = self
            .selection_history
            .iter()
            .filter(|s| s.success && s.response_time.is_some())
            .filter_map(|s| s.response_time.map(|rt| rt as f64))
            .collect();

        let avg_response_time_all = if !successful_response_times.is_empty() {
            successful_response_times.iter().sum::<f64>() / successful_response_times.len() as f64
        } else {
            0.0
        };

        // Calculate average performance score for successful selections
        let successful_performance_scores: Vec<f64> = self
            .selection_history
            .iter()
            .filter(|s| s.success && s.performance_score.is_some())
            .filter_map(|s| s.performance_score)
            .collect();
        let avg_performance_score = if !successful_performance_scores.is_empty() {
            successful_performance_scores.iter().sum::<f64>()
                / successful_performance_scores.len() as f64
        } else {
            0.0
        };

        // Count strategy usage
        let mut strategy_counts = std::collections::HashMap::new();
        for record in &self.selection_history {
            let strategy_name = format!("{:?}", record.strategy_used);
            *strategy_counts.entry(strategy_name).or_insert(0) += 1;
        }

        SelectionStatistics {
            total_selections: total_selections as u64,
            successful_selections: successful_selections as u64,
            average_selection_time: Duration::from_millis(avg_response_time_all as u64),
            selection_distribution: strategy_counts.clone(),
            performance_metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("success_rate".to_string(), success_rate);
                metrics.insert("avg_response_time".to_string(), avg_response_time_all);
                metrics
            },
            success_rate,
            avg_performance_score,
            strategy_usage: strategy_counts,
        }
    }

    /// Clear selection history
    pub fn clear_history(&mut self) {
        self.selection_history.clear();
    }

    /// Get recent selection history
    pub fn get_recent_history(&self, limit: usize) -> Vec<&SelectionRecord> {
        self.selection_history.iter().rev().take(limit).collect()
    }

    /// Initialize or update machine learning model state
    pub fn initialize_ml_model(&mut self, feature_count: usize) {
        let mut feature_importance = HashMap::new();
        // Initialize feature importance with default values
        for i in 0..feature_count {
            feature_importance.insert(format!("feature_{}", i), 0.0);
        }

        self.ml_model_state = Some(MLModelState {
            model_accuracy: 0.0,
            training_samples: 0,
            last_training: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            feature_importance,
        });
    }

    /// Update ML model with training data
    pub fn update_ml_model(
        &mut self,
        features: &[f64],
        actual_performance: f64,
        predicted_performance: f64,
    ) {
        if let Some(ref mut ml_state) = self.ml_model_state {
            // Simple gradient descent update
            let learning_rate = 0.01;
            let error = actual_performance - predicted_performance;

            // Update feature importance based on training (simplified approach)
            for (i, &feature) in features.iter().enumerate() {
                let feature_name = format!("feature_{}", i);
                let current_importance = ml_state
                    .feature_importance
                    .get(&feature_name)
                    .unwrap_or(&0.0);
                let new_importance = current_importance + learning_rate * error * feature;
                ml_state
                    .feature_importance
                    .insert(feature_name, new_importance);
            }

            ml_state.training_samples += 1;

            // Update prediction accuracy (exponential moving average)
            let accuracy = 1.0 - (error.abs() / actual_performance.max(1.0));
            if ml_state.model_accuracy == 0.0 {
                ml_state.model_accuracy = accuracy;
            } else {
                ml_state.model_accuracy = 0.9 * ml_state.model_accuracy + 0.1 * accuracy;
            }

            ml_state.last_training = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
    }
}

// ================================================================================================
// Mirror Selection Engine
// ================================================================================================

/// Advanced mirror selection engine with multiple strategies
pub struct MirrorSelector<'a> {
    config: &'a MirrorConfig,
    selection_state: &'a mut MirrorSelectionState,
    geographic_calculator: &'a GeographicCalculator,
    performance_analyzer: &'a PerformanceAnalyzer,
}

impl<'a> MirrorSelector<'a> {
    /// Create a new mirror selector
    pub fn new(
        config: &'a MirrorConfig,
        selection_state: &'a mut MirrorSelectionState,
        geographic_calculator: &'a GeographicCalculator,
        performance_analyzer: &'a PerformanceAnalyzer,
    ) -> Self {
        Self {
            config,
            selection_state,
            geographic_calculator,
            performance_analyzer,
        }
    }

    /// Advanced mirror selection with sophisticated algorithms and optimization
    ///
    /// Selects mirrors based on the configured strategy, filtering out inactive
    /// or unreliable mirrors and applying quality thresholds.
    pub fn select_mirrors(&mut self) -> Result<Vec<MirrorServer>> {
        // Filter active and qualified mirrors
        let active_mirrors = self.get_qualified_mirrors()?;

        if active_mirrors.is_empty() {
            return Err(TorshError::IoError(
                "No active mirrors available that meet quality requirements".to_string(),
            ));
        }

        // Apply selection strategy
        let selected_mirrors = match &self.config.selection_strategy {
            MirrorSelectionStrategy::LowestLatency => self.select_by_lowest_latency(active_mirrors),
            MirrorSelectionStrategy::HighestReliability => {
                self.select_by_highest_reliability(active_mirrors)
            }
            MirrorSelectionStrategy::Geographic => self.select_by_geographic(active_mirrors),
            MirrorSelectionStrategy::LowestLatency => self.select_by_lowest_latency(active_mirrors),
            MirrorSelectionStrategy::Weighted(weights) => {
                self.select_by_weighted_score(active_mirrors, weights)
            }
            MirrorSelectionStrategy::RoundRobin => self.select_by_round_robin(active_mirrors),
            MirrorSelectionStrategy::Adaptive => self.select_adaptively(active_mirrors),
            MirrorSelectionStrategy::MachineLearning(ml_config) => {
                self.select_by_machine_learning(active_mirrors, ml_config)
            }
        };

        Ok(selected_mirrors)
    }

    /// Filter mirrors based on quality requirements
    fn get_qualified_mirrors(&self) -> Result<Vec<MirrorServer>> {
        let qualified_mirrors: Vec<MirrorServer> = self
            .config
            .mirrors
            .iter()
            .filter(|m| {
                m.active
                    && m.consecutive_failures < 5
                    && m.reliability_score >= self.config.min_reliability_score
                    && m.avg_response_time
                        .map_or(true, |time| time <= self.config.max_response_time)
            })
            .cloned()
            .collect();

        Ok(qualified_mirrors)
    }

    /// Select mirrors by lowest latency
    fn select_by_lowest_latency(&self, mut mirrors: Vec<MirrorServer>) -> Vec<MirrorServer> {
        mirrors.sort_by_key(|m| m.avg_response_time.unwrap_or(u64::MAX));
        mirrors
    }

    /// Select mirrors by highest reliability
    fn select_by_highest_reliability(&self, mut mirrors: Vec<MirrorServer>) -> Vec<MirrorServer> {
        mirrors.sort_by(|a, b| {
            b.reliability_score
                .partial_cmp(&a.reliability_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        mirrors
    }

    /// Select mirrors by geographic proximity
    fn select_by_geographic(&self, mut mirrors: Vec<MirrorServer>) -> Vec<MirrorServer> {
        if self.config.enable_geographic_optimization {
            mirrors = self
                .geographic_calculator
                .sort_by_geographic_proximity(mirrors);
        } else {
            // Fallback to country-based sorting
            mirrors.sort_by(|a, b| a.location.country.cmp(&b.location.country));
        }
        mirrors
    }

    /// Select mirrors by lowest current load
    fn select_by_lowest_load(&self, mut mirrors: Vec<MirrorServer>) -> Vec<MirrorServer> {
        mirrors.sort_by(|a, b| {
            let load_a = a.capacity.current_load.unwrap_or(100.0);
            let load_b = b.capacity.current_load.unwrap_or(100.0);
            load_a
                .partial_cmp(&load_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        mirrors
    }

    /// Select mirrors using weighted scoring algorithm
    fn select_by_weighted_score(
        &self,
        mut mirrors: Vec<MirrorServer>,
        weights: &MirrorWeights,
    ) -> Vec<MirrorServer> {
        mirrors.sort_by(|a, b| {
            let score_a = self.calculate_weighted_score(a, weights);
            let score_b = self.calculate_weighted_score(b, weights);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        mirrors
    }

    /// Select mirrors randomly (with reliability bias)
    fn select_randomly(&self, mut mirrors: Vec<MirrorServer>) -> Vec<MirrorServer> {
        // Pseudo-random selection based on reliability scores to ensure determinism
        mirrors.sort_by(|a, b| {
            let rand_a = (a.reliability_score * 1000.0) as u64 % 100;
            let rand_b = (b.reliability_score * 1000.0) as u64 % 100;
            rand_b.cmp(&rand_a)
        });
        mirrors
    }

    /// Select mirrors using round-robin strategy
    fn select_by_round_robin(&mut self, active_mirrors: Vec<MirrorServer>) -> Vec<MirrorServer> {
        if active_mirrors.is_empty() {
            return active_mirrors;
        }

        let index = self
            .selection_state
            .get_next_round_robin_index(active_mirrors.len());
        let selected_mirror = active_mirrors[index].clone();

        // Return selected mirror first, followed by others for failover
        let mut result = vec![selected_mirror.clone()];
        result.extend(
            active_mirrors
                .into_iter()
                .filter(|m| m.id != selected_mirror.id),
        );
        result
    }

    /// Select mirrors using adaptive algorithm
    fn select_adaptively(&mut self, mut mirrors: Vec<MirrorServer>) -> Vec<MirrorServer> {
        // Update adaptive weights based on historical performance
        self.selection_state.update_adaptive_weights();

        // Use updated adaptive weights for selection
        mirrors.sort_by(|a, b| {
            let score_a = self.calculate_weighted_score(a, &self.selection_state.adaptive_weights);
            let score_b = self.calculate_weighted_score(b, &self.selection_state.adaptive_weights);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        mirrors
    }

    /// Select mirrors using machine learning algorithm
    fn select_by_machine_learning(
        &mut self,
        mut mirrors: Vec<MirrorServer>,
        ml_config: &MLConfig,
    ) -> Vec<MirrorServer> {
        if !ml_config.online_learning {
            // Fallback to weighted selection
            return self.select_by_weighted_score(mirrors, &MirrorWeights::default());
        }

        // Initialize ML model if not exists
        if self.selection_state.ml_model_state.is_none() {
            self.selection_state.initialize_ml_model(6); // 6 features
        }

        // Calculate ML-based scores for mirrors
        mirrors.sort_by(|a, b| {
            let score_a = self.calculate_ml_score(a, ml_config);
            let score_b = self.calculate_ml_score(b, ml_config);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        mirrors
    }

    /// Calculate advanced weighted score for a mirror with comprehensive factors
    fn calculate_weighted_score(&self, mirror: &MirrorServer, weights: &MirrorWeights) -> f64 {
        // Latency score (lower is better, normalized to 0-1)
        let latency_score = match mirror.avg_response_time {
            Some(time) => {
                let max_response_time = self.config.max_response_time;
                (max_response_time.saturating_sub(time.min(max_response_time)) as f64)
                    / max_response_time as f64
            }
            None => 0.0,
        };

        // Reliability score (already 0-1)
        let reliability_score = mirror.reliability_score;

        // Load score (lower load is better, normalized to 0-1)
        let load_score = match mirror.capacity.current_load {
            Some(load) => (100.0 - load.min(100.0)) / 100.0,
            None => 0.5, // Neutral score if unknown
        };

        // Geographic proximity score
        let geographic_score = self
            .geographic_calculator
            .calculate_geographic_score(mirror);

        // Bandwidth score (higher bandwidth is better)
        let bandwidth_score = match mirror.capacity.max_bandwidth {
            Some(bandwidth) => {
                // Normalize to reasonable range (1 Gbps = 1000 Mbps)
                (bandwidth as f64 / 1000.0).min(1.0)
            }
            None => 0.5, // Neutral score if unknown
        };

        // Provider quality score
        let provider_score = mirror.provider_info.network_quality.quality_score;

        // Calculate weighted sum
        weights.latency * latency_score
            + weights.reliability * reliability_score
            + weights.load * (load_score as f64)
            + weights.geographic * geographic_score
            + weights.bandwidth * bandwidth_score
            + weights.provider_quality * provider_score
    }

    /// Calculate machine learning-based score for a mirror
    fn calculate_ml_score(&self, mirror: &MirrorServer, ml_config: &MLConfig) -> f64 {
        let ml_state = match &self.selection_state.ml_model_state {
            Some(state) => state,
            None => return 0.5, // Fallback score
        };

        if ml_state.training_samples < 10 {
            // Default minimum samples
            // Not enough training data, fallback to weighted score
            return self.calculate_weighted_score(mirror, &MirrorWeights::default());
        }

        // Extract features for ML prediction
        let features = self.extract_mirror_features(mirror);

        // Calculate prediction using learned weights
        let mut score = 0.0;
        for (i, &feature) in features.iter().enumerate() {
            let feature_name = format!("feature_{}", i);
            if let Some(&importance) = ml_state.feature_importance.get(&feature_name) {
                score += importance * feature;
            }
        }

        // Normalize score to [0, 1] range
        score.max(0.0).min(1.0)
    }

    /// Extract numerical features from a mirror for ML algorithms
    fn extract_mirror_features(&self, mirror: &MirrorServer) -> Vec<f64> {
        vec![
            // Feature 1: Normalized latency (lower is better)
            mirror.avg_response_time.map_or(0.0, |t| {
                1.0 - (t as f64 / self.config.max_response_time as f64)
            }),
            // Feature 2: Reliability score
            mirror.reliability_score,
            // Feature 3: Load score (lower is better)
            mirror
                .capacity
                .current_load
                .map_or(0.5, |load| ((100.0 - load) / 100.0) as f64),
            // Feature 4: Geographic score
            self.geographic_calculator
                .calculate_geographic_score(mirror),
            // Feature 5: Bandwidth score
            mirror
                .capacity
                .max_bandwidth
                .map_or(0.5, |bw| (bw as f64 / 1000.0).min(1.0)),
            // Feature 6: Provider quality score
            mirror.provider_info.network_quality.quality_score,
        ]
    }
}

// ================================================================================================
// Mirror Selection Utilities
// ================================================================================================

impl Default for SelectionStatistics {
    fn default() -> Self {
        Self {
            total_selections: 0,
            successful_selections: 0,
            average_selection_time: std::time::Duration::from_secs(0),
            selection_distribution: std::collections::HashMap::new(),
            performance_metrics: std::collections::HashMap::new(),
            success_rate: 0.0,
            avg_performance_score: 0.0,
            strategy_usage: std::collections::HashMap::new(),
        }
    }
}

/// Validate mirror selection strategy configuration
pub fn validate_selection_strategy(strategy: &MirrorSelectionStrategy) -> Result<()> {
    match strategy {
        MirrorSelectionStrategy::Weighted(weights) => {
            let total = weights.latency
                + weights.reliability
                + weights.load
                + weights.geographic
                + weights.bandwidth
                + weights.provider_quality;

            if (total - 1.0).abs() > 0.01 {
                return Err(TorshError::IoError(format!(
                    "Mirror weights must sum to approximately 1.0, got {}",
                    total
                )));
            }

            // Check individual weights are non-negative
            if weights.latency < 0.0
                || weights.reliability < 0.0
                || weights.load < 0.0
                || weights.geographic < 0.0
                || weights.bandwidth < 0.0
                || weights.provider_quality < 0.0
            {
                return Err(TorshError::IoError(
                    "All mirror weights must be non-negative".to_string(),
                ));
            }
        }
        MirrorSelectionStrategy::MachineLearning(ml_config) => {
            if ml_config.learning_rate <= 0.0 || ml_config.learning_rate > 1.0 {
                return Err(TorshError::IoError(
                    "ML learning rate must be between 0.0 and 1.0".to_string(),
                ));
            }

            if ml_config.min_samples == 0 {
                return Err(TorshError::IoError(
                    "ML minimum samples must be greater than 0".to_string(),
                ));
            }
        }
        _ => {} // Other strategies don't need validation
    }

    Ok(())
}

/// Create optimized weights for different scenarios
pub fn create_optimized_weights(scenario: &str) -> MirrorWeights {
    match scenario {
        "speed" => MirrorWeights {
            latency: 0.5,
            reliability: 0.2,
            load: 0.15,
            geographic: 0.1,
            bandwidth: 0.05,
            provider_quality: 0.0,
        },
        "reliability" => MirrorWeights {
            latency: 0.1,
            reliability: 0.6,
            load: 0.15,
            geographic: 0.05,
            bandwidth: 0.05,
            provider_quality: 0.05,
        },
        "geographic" => MirrorWeights {
            latency: 0.2,
            reliability: 0.2,
            load: 0.1,
            geographic: 0.4,
            bandwidth: 0.05,
            provider_quality: 0.05,
        },
        "balanced" | _ => MirrorWeights::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_mirror(id: &str, latency: Option<u64>, reliability: f64) -> MirrorServer {
        MirrorServer {
            id: id.to_string(),
            base_url: format!("https://{}.example.com", id),
            reliability_score: reliability,
            avg_response_time: latency,
            consecutive_failures: 0,
            location: MirrorLocation {
                country: "US".to_string(),
                region: "California".to_string(),
                city: "San Francisco".to_string(),
                latitude: Some(37.7749),
                longitude: Some(-122.4194),
                provider: "TestProvider".to_string(),
                timezone: Some("America/Los_Angeles".to_string()),
                datacenter: Some("sfo1".to_string()),
            },
            capacity: MirrorCapacity::default(),
            active: true,
            metadata: HashMap::new(),
            priority_weight: 1.0,
            last_successful_connection: None,
            provider_info: ProviderInfo {
                name: "TestProvider".to_string(),
                network_tier: Some("Premium".to_string()),
                cdn_integration: true,
                edge_location: Some("SFO".to_string()),
                network_quality: NetworkQuality::default(),
            },
            performance_history: Vec::new(),
        }
    }

    fn create_test_config() -> MirrorConfig {
        MirrorConfig {
            mirrors: vec![
                create_test_mirror("fast", Some(50), 0.9),
                create_test_mirror("reliable", Some(100), 0.99),
                create_test_mirror("slow", Some(200), 0.8),
            ],
            selection_strategy: MirrorSelectionStrategy::Weighted(MirrorWeights::default()),
            max_mirror_attempts: 3,
            connection_timeout: Duration::from_secs(10),
            enable_auto_discovery: false,
            benchmark_interval: 3600,
            enable_geographic_optimization: true,
            min_reliability_score: 0.7,
            max_response_time: 5000,
            load_balancing: LoadBalancingConfig::default(),
        }
    }

    #[test]
    fn test_mirror_selection_state_creation() {
        let state = MirrorSelectionState::new();
        assert_eq!(state.round_robin_index, 0);
        assert_eq!(state.selection_history.len(), 0);
        assert!(state.ml_model_state.is_none());
    }

    #[test]
    fn test_record_selection() {
        let mut state = MirrorSelectionState::new();

        state.record_selection(
            "test_mirror",
            MirrorSelectionStrategy::LowestLatency,
            true,
            100.0,
        );

        assert_eq!(state.selection_history.len(), 1);
        assert_eq!(state.selection_history[0].mirror_id, "test_mirror");
        assert!(state.selection_history[0].success);
    }

    #[test]
    fn test_round_robin_index() {
        let mut state = MirrorSelectionState::new();

        assert_eq!(state.get_next_round_robin_index(3), 0);
        assert_eq!(state.get_next_round_robin_index(3), 1);
        assert_eq!(state.get_next_round_robin_index(3), 2);
        assert_eq!(state.get_next_round_robin_index(3), 0); // Should wrap around
    }

    #[test]
    fn test_selection_statistics() {
        let mut state = MirrorSelectionState::new();

        // Record some selections
        state.record_selection(
            "mirror1",
            MirrorSelectionStrategy::LowestLatency,
            true,
            50.0,
        );
        state.record_selection(
            "mirror2",
            MirrorSelectionStrategy::LowestLatency,
            false,
            f64::INFINITY,
        );
        state.record_selection(
            "mirror1",
            MirrorSelectionStrategy::HighestReliability,
            true,
            75.0,
        );

        let stats = state.get_selection_statistics();
        assert_eq!(stats.total_selections, 3);
        assert_eq!(stats.successful_selections, 2);
        assert_eq!(stats.success_rate, 2.0 / 3.0);
        assert_eq!(stats.avg_performance_score, 62.5); // (50.0 + 75.0) / 2
    }

    #[test]
    fn test_weighted_score_calculation() {
        let config = create_test_config();
        let mut state = MirrorSelectionState::new();
        let geo_calc = GeographicCalculator::new();
        let perf_analyzer = PerformanceAnalyzer::new();

        let selector = MirrorSelector::new(&config, &mut state, &geo_calc, &perf_analyzer);
        let weights = MirrorWeights::default();
        let mirror = &config.mirrors[0];

        let score = selector.calculate_weighted_score(mirror, &weights);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_lowest_latency_selection() {
        let config = create_test_config();
        let mut state = MirrorSelectionState::new();
        let geo_calc = GeographicCalculator::new();
        let perf_analyzer = PerformanceAnalyzer::new();

        let mut selector = MirrorSelector::new(&config, &mut state, &geo_calc, &perf_analyzer);
        let mirrors = selector.get_qualified_mirrors().unwrap();
        let sorted = selector.select_by_lowest_latency(mirrors);

        // Should be sorted by latency (fast=50, reliable=100, slow=200)
        assert_eq!(sorted[0].id, "fast");
        assert_eq!(sorted[1].id, "reliable");
        assert_eq!(sorted[2].id, "slow");
    }

    #[test]
    fn test_highest_reliability_selection() {
        let config = create_test_config();
        let mut state = MirrorSelectionState::new();
        let geo_calc = GeographicCalculator::new();
        let perf_analyzer = PerformanceAnalyzer::new();

        let mut selector = MirrorSelector::new(&config, &mut state, &geo_calc, &perf_analyzer);
        let mirrors = selector.get_qualified_mirrors().unwrap();
        let sorted = selector.select_by_highest_reliability(mirrors);

        // Should be sorted by reliability (reliable=0.99, fast=0.9, slow=0.8)
        assert_eq!(sorted[0].id, "reliable");
        assert_eq!(sorted[1].id, "fast");
        assert_eq!(sorted[2].id, "slow");
    }

    #[test]
    fn test_round_robin_selection() {
        let config = create_test_config();
        let mut state = MirrorSelectionState::new();
        let geo_calc = GeographicCalculator::new();
        let perf_analyzer = PerformanceAnalyzer::new();

        let mut selector = MirrorSelector::new(&config, &mut state, &geo_calc, &perf_analyzer);
        let mirrors = selector.get_qualified_mirrors().unwrap();

        let selected1 = selector.select_by_round_robin(mirrors.clone());
        let selected2 = selector.select_by_round_robin(mirrors.clone());
        let selected3 = selector.select_by_round_robin(mirrors.clone());

        // Should cycle through mirrors
        assert_ne!(selected1[0].id, selected2[0].id);
        assert_ne!(selected2[0].id, selected3[0].id);
    }

    #[test]
    fn test_mirror_weights_validation() {
        let valid_weights = MirrorWeights::default();
        assert!(
            validate_selection_strategy(&MirrorSelectionStrategy::Weighted(valid_weights)).is_ok()
        );

        let invalid_weights = MirrorWeights {
            latency: 0.5,
            reliability: 0.5,
            load: 0.5, // Sum > 1.0
            geographic: 0.0,
            bandwidth: 0.0,
            provider_quality: 0.0,
        };
        assert!(
            validate_selection_strategy(&MirrorSelectionStrategy::Weighted(invalid_weights))
                .is_err()
        );
    }

    #[test]
    fn test_ml_config_validation() {
        let valid_ml_config = MLConfig {
            model_type: MLModelType::DecisionTree,
            online_learning: true,
            learning_rate: 0.01,
            min_samples: 10,
        };
        assert!(
            validate_selection_strategy(&MirrorSelectionStrategy::MachineLearning(valid_ml_config))
                .is_ok()
        );

        let invalid_ml_config = MLConfig {
            model_type: MLModelType::DecisionTree,
            online_learning: true,
            learning_rate: 1.5, // Invalid learning rate
            min_samples: 10,
        };
        assert!(
            validate_selection_strategy(&MirrorSelectionStrategy::MachineLearning(
                invalid_ml_config
            ))
            .is_err()
        );
    }

    #[test]
    fn test_optimized_weights_creation() {
        let speed_weights = create_optimized_weights("speed");
        assert!(speed_weights.latency > speed_weights.reliability);

        let reliability_weights = create_optimized_weights("reliability");
        assert!(reliability_weights.reliability > reliability_weights.latency);

        let geo_weights = create_optimized_weights("geographic");
        assert!(geo_weights.geographic > geo_weights.latency);

        let balanced_weights = create_optimized_weights("balanced");
        assert_eq!(balanced_weights, MirrorWeights::default());
    }

    #[test]
    fn test_ml_model_initialization() {
        let mut state = MirrorSelectionState::new();
        assert!(state.ml_model_state.is_none());

        state.initialize_ml_model(6);
        assert!(state.ml_model_state.is_some());

        let ml_state = state.ml_model_state.as_ref().unwrap();
        assert_eq!(ml_state.feature_importance.len(), 6);
        assert_eq!(ml_state.training_samples, 0);
    }

    #[test]
    fn test_feature_extraction() {
        let config = create_test_config();
        let mut state = MirrorSelectionState::new();
        let geo_calc = GeographicCalculator::new();
        let perf_analyzer = PerformanceAnalyzer::new();

        let selector = MirrorSelector::new(&config, &mut state, &geo_calc, &perf_analyzer);
        let mirror = &config.mirrors[0];

        let features = selector.extract_mirror_features(mirror);
        assert_eq!(features.len(), 6);

        // All features should be in [0, 1] range
        for &feature in &features {
            assert!(feature >= 0.0 && feature <= 1.0);
        }
    }

    #[test]
    fn test_adaptive_weights_update() {
        let mut state = MirrorSelectionState::new();

        // Record some unsuccessful selections to trigger weight adjustment
        for _ in 0..15 {
            state.record_selection(
                "test",
                MirrorSelectionStrategy::Adaptive,
                false,
                f64::INFINITY,
            );
        }

        let initial_reliability_weight = state.adaptive_weights.reliability;
        state.update_adaptive_weights();

        // Reliability weight should have increased due to low success rate
        assert!(state.adaptive_weights.reliability > initial_reliability_weight);
    }
}
