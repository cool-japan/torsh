//! Main integration orchestration for SciRS2 integration
//!
//! This module provides the central ScirS2Integration coordination layer
//! that orchestrates all components and provides the unified interface.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::memory_profiler::allocation::{HintSeverity, PerformanceHint, PerformanceHintType};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{
    config::{IntegrationStatus, ScirS2IntegrationConfig},
    event_system::{ScirS2Event, ScirS2EventProcessor},
    monitoring::ScirS2MonitoringSystem,
    optimization::{AdvancedScirS2Features, OptimizationMetrics, ScirS2OptimizationEngine},
    pool_management::{PoolStatsAggregator, ScirS2PoolInfo},
    statistics::{AllocatorStatsAggregator, ScirS2AllocatorStats},
};

/// SciRS2 integration state and statistics
///
/// Comprehensive integration layer with the SciRS2 ecosystem providing
/// real-time monitoring, optimization, and advanced memory management capabilities.
pub struct ScirS2Integration {
    /// Whether SciRS2 integration is active
    pub active: bool,

    /// SciRS2 memory allocator statistics
    pub allocator_stats: Arc<RwLock<HashMap<String, ScirS2AllocatorStats>>>,

    /// SciRS2 memory pool information
    pub pool_info: Arc<RwLock<HashMap<String, ScirS2PoolInfo>>>,

    /// Integration configuration
    pub config: ScirS2IntegrationConfig,

    /// Last sync with SciRS2 statistics
    pub last_sync: Option<Instant>,

    /// SciRS2 event callbacks
    pub event_callbacks: Vec<Box<dyn Fn(ScirS2Event) + Send + Sync>>,

    /// Advanced integration features
    advanced_features: Arc<Mutex<AdvancedScirS2Features>>,

    /// Performance optimization engine
    optimization_engine: Arc<Mutex<ScirS2OptimizationEngine>>,

    /// Real-time monitoring system
    monitoring_system: Arc<RwLock<ScirS2MonitoringSystem>>,

    /// Event processing system
    event_processor: Arc<Mutex<ScirS2EventProcessor>>,

    /// Statistics aggregators
    allocator_aggregator: AllocatorStatsAggregator,
    pool_aggregator: PoolStatsAggregator,

    /// Performance metrics
    integration_metrics: IntegrationMetrics,
}

/// Integration performance metrics snapshot
#[derive(Debug, Clone)]
pub struct IntegrationMetricsSnapshot {
    /// Total events processed
    pub total_events_processed: u64,

    /// Total synchronizations performed
    pub total_synchronizations: u64,

    /// Average synchronization time
    pub avg_sync_time: Duration,

    /// Integration uptime
    pub uptime: Duration,

    /// Start time
    pub start_time: Instant,

    /// Error count
    pub error_count: u64,

    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
}

/// Integration performance metrics
#[derive(Debug)]
pub struct IntegrationMetrics {
    /// Total events processed
    pub total_events_processed: AtomicU64,

    /// Total synchronizations performed
    pub total_synchronizations: AtomicU64,

    /// Average synchronization time
    pub avg_sync_time: Arc<RwLock<Duration>>,

    /// Integration uptime
    pub uptime: Arc<RwLock<Duration>>,

    /// Start time
    pub start_time: Instant,

    /// Error count
    pub error_count: AtomicU64,

    /// Success rate (0.0 to 1.0)
    pub success_rate: Arc<RwLock<f64>>,
}

impl ScirS2Integration {
    /// Create a new SciRS2 integration
    pub fn new(config: ScirS2IntegrationConfig) -> Self {
        Self {
            active: false,
            allocator_stats: Arc::new(RwLock::new(HashMap::new())),
            pool_info: Arc::new(RwLock::new(HashMap::new())),
            config,
            last_sync: None,
            event_callbacks: Vec::new(),
            advanced_features: Arc::new(Mutex::new(AdvancedScirS2Features::new())),
            optimization_engine: Arc::new(Mutex::new(ScirS2OptimizationEngine::new())),
            monitoring_system: Arc::new(RwLock::new(ScirS2MonitoringSystem::new())),
            event_processor: Arc::new(Mutex::new(ScirS2EventProcessor::new())),
            allocator_aggregator: AllocatorStatsAggregator::new(),
            pool_aggregator: PoolStatsAggregator::new(),
            integration_metrics: IntegrationMetrics::new(),
        }
    }

    /// Activate SciRS2 integration
    pub fn activate(&mut self) -> Result<(), String> {
        if self.active {
            return Ok(());
        }

        // Validate configuration
        super::config::validate_config(&self.config)?;

        // Initialize SciRS2 connection (placeholder)
        self.active = true;
        // Don't set last_sync here - let sync_statistics set it after first actual sync
        self.integration_metrics.start_time = Instant::now();

        // Initialize monitoring system
        self.initialize_monitoring()?;

        // Start advanced features if enabled
        if self.config.advanced_config.enable_predictive_modeling {
            self.enable_predictive_modeling()?;
        }

        if self.config.advanced_config.enable_automated_optimization {
            self.enable_auto_optimization()?;
        }

        // Initialize event processing
        self.initialize_event_processing();

        Ok(())
    }

    /// Deactivate SciRS2 integration
    pub fn deactivate(&mut self) {
        if !self.active {
            return;
        }

        self.active = false;
        self.last_sync = None;

        // Clean up resources
        self.cleanup_resources();

        // Update metrics
        *self.integration_metrics.uptime.write() = self.integration_metrics.start_time.elapsed();
    }

    /// Synchronize statistics with SciRS2
    pub fn sync_statistics(&mut self) -> Result<(), String> {
        if !self.active {
            return Err("SciRS2 integration not active".to_string());
        }

        let sync_start = Instant::now();
        let now = Instant::now();

        // Check if sync is needed based on interval
        if let Some(last_sync) = self.last_sync {
            if now.duration_since(last_sync) < self.config.sync_interval {
                return Ok(()); // Too early to sync again
            }
        }

        // Perform synchronization
        match self.perform_synchronization() {
            Ok(_) => {
                self.last_sync = Some(now);
                let sync_count = self
                    .integration_metrics
                    .total_synchronizations
                    .fetch_add(1, Ordering::Relaxed)
                    + 1;

                // Update average sync time
                let sync_duration = sync_start.elapsed();
                {
                    let mut avg_time = self.integration_metrics.avg_sync_time.write();
                    let total_sync_time =
                        *avg_time * sync_count.saturating_sub(1) as u32 + sync_duration;
                    *avg_time = total_sync_time / sync_count as u32;
                }

                Ok(())
            }
            Err(e) => {
                self.integration_metrics
                    .error_count
                    .fetch_add(1, Ordering::Relaxed);
                self.update_success_rate();
                Err(e)
            }
        }
    }

    /// Add event callback
    pub fn add_event_callback<F>(&mut self, callback: F)
    where
        F: Fn(ScirS2Event) + Send + Sync + 'static + Clone,
    {
        self.event_callbacks.push(Box::new(callback.clone()));

        // Also add to event processor with signature conversion
        let mut processor = self.event_processor.lock().unwrap();
        let callback_ref = move |event: &ScirS2Event| {
            callback(event.clone());
        };
        processor.add_callback(callback_ref);
    }

    /// Process SciRS2 event
    pub fn process_event(&self, event: ScirS2Event) {
        // Update internal state based on event
        self.update_state_from_event(&event);

        // Process through event processor
        {
            let mut processor = self.event_processor.lock().unwrap();
            processor.process_event(event.clone());
        }

        // Update monitoring system
        {
            let mut monitoring = self.monitoring_system.write();
            monitoring.process_event(&event);
        }

        // Trigger callbacks
        for callback in &self.event_callbacks {
            callback(event.clone());
        }

        // Check for optimization opportunities
        if self.config.enable_optimization_suggestions {
            self.check_optimization_opportunities(&event);
        }

        // Update metrics
        self.integration_metrics
            .total_events_processed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Get allocator statistics
    pub fn get_allocator_stats(&self, allocator_name: &str) -> Option<ScirS2AllocatorStats> {
        let stats_guard = self.allocator_stats.read();
        stats_guard.get(allocator_name).cloned()
    }

    /// Get all allocator statistics
    pub fn get_all_allocator_stats(&self) -> HashMap<String, ScirS2AllocatorStats> {
        self.allocator_stats.read().clone()
    }

    /// Get pool information
    pub fn get_pool_info(&self, pool_id: &str) -> Option<ScirS2PoolInfo> {
        let pool_guard = self.pool_info.read();
        pool_guard.get(pool_id).cloned()
    }

    /// Get all pool information
    pub fn get_all_pools(&self) -> HashMap<String, ScirS2PoolInfo> {
        self.pool_info.read().clone()
    }

    /// Get optimization suggestions
    pub fn get_optimization_suggestions(&self) -> Vec<PerformanceHint> {
        let mut suggestions = Vec::new();

        // Generate suggestions based on allocator performance
        let allocator_stats = self.allocator_stats.read();
        for (name, stats) in allocator_stats.iter() {
            if stats.memory_efficiency < 0.8 {
                suggestions.push(PerformanceHint {
                    hint_type: PerformanceHintType::InefficientSize,
                    severity: HintSeverity::Warning,
                    description: format!(
                        "Allocator '{}' has low memory efficiency: {:.2}",
                        name, stats.memory_efficiency
                    ),
                    suggested_action: "Consider adjusting allocation strategy or pool sizes"
                        .to_string(),
                    impact_estimate: 1.0 - stats.memory_efficiency,
                });
            }

            if stats.allocation_failures > 0 {
                suggestions.push(PerformanceHint {
                    hint_type: PerformanceHintType::ExcessiveAllocations,
                    severity: HintSeverity::Critical,
                    description: format!(
                        "Allocator '{}' has {} allocation failures",
                        name, stats.allocation_failures
                    ),
                    suggested_action:
                        "Investigate memory pressure and consider increasing pool capacity"
                            .to_string(),
                    impact_estimate: 0.5,
                });
            }
        }
        drop(allocator_stats);

        // Generate suggestions based on pool utilization
        let pool_info = self.pool_info.read();
        for (id, pool) in pool_info.iter() {
            if pool.utilization > 0.9 {
                suggestions.push(PerformanceHint {
                    hint_type: PerformanceHintType::SuboptimalMemoryType,
                    severity: HintSeverity::Warning,
                    description: format!(
                        "Pool '{}' has high utilization: {:.1}%",
                        id,
                        pool.utilization * 100.0
                    ),
                    suggested_action:
                        "Consider expanding pool capacity or optimizing allocation patterns"
                            .to_string(),
                    impact_estimate: pool.utilization - 0.8,
                });
            }
        }
        drop(pool_info);

        // Add ML-generated suggestions
        if self.config.advanced_config.enable_automated_optimization {
            if let Ok(engine) = self.optimization_engine.lock() {
                suggestions.extend(engine.get_ml_suggestions());
            }
        }

        suggestions
    }

    /// Check if integration is healthy
    pub fn is_healthy(&self) -> bool {
        // Inactive integrations are not necessarily unhealthy, just not operational
        if !self.active {
            return true; // Fresh/inactive integrations are not unhealthy
        }

        // Check if recent sync was successful
        if let Some(last_sync) = self.last_sync {
            let sync_age = Instant::now().duration_since(last_sync);
            if sync_age > self.config.sync_interval * 3 {
                return false; // Sync is too old
            }
        }

        // Check success rate
        if self.integration_metrics.get_success_rate() < 0.8 {
            return false;
        }

        // Check allocator health
        {
            let allocator_stats = self.allocator_stats.read();
            for stats in allocator_stats.values() {
                if !stats.is_healthy() {
                    return false;
                }
            }
        }

        // Check pool health
        {
            let pool_info = self.pool_info.read();
            for pool in pool_info.values() {
                if !pool.is_healthy() {
                    return false;
                }
            }
        }

        true
    }

    /// Get integration status
    pub fn get_status(&self) -> IntegrationStatus {
        IntegrationStatus {
            active: self.active,
            last_sync: self.last_sync,
            allocator_count: self.allocator_stats.read().len(),
            pool_count: self.pool_info.read().len(),
            health_score: self.calculate_health_score(),
            sync_interval: self.config.sync_interval,
            features_enabled: self.get_enabled_features(),
        }
    }

    /// Get integration metrics
    pub fn get_integration_metrics(&self) -> IntegrationMetricsSnapshot {
        IntegrationMetricsSnapshot {
            total_events_processed: self.integration_metrics.get_total_events_processed(),
            total_synchronizations: self.integration_metrics.get_total_synchronizations(),
            avg_sync_time: self.integration_metrics.get_avg_sync_time(),
            uptime: self.integration_metrics.get_uptime(),
            start_time: self.integration_metrics.start_time,
            error_count: self.integration_metrics.get_error_count(),
            success_rate: self.integration_metrics.get_success_rate(),
        }
    }

    /// Get monitoring dashboard data
    pub fn get_dashboard_data(&self) -> super::monitoring::DashboardData {
        let monitoring = self.monitoring_system.read();
        monitoring.get_dashboard_data()
    }

    /// Force flush all buffers and caches
    pub fn flush_all(&mut self) {
        // Flush event processor buffer
        {
            let mut processor = self.event_processor.lock().unwrap();
            processor.flush_buffer();
        }

        // Force sync if needed
        if self.active {
            let _ = self.sync_statistics();
        }

        // Clean up old monitoring data
        {
            let mut monitoring = self.monitoring_system.write();
            monitoring.cleanup_old_data();
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ScirS2IntegrationConfig) -> Result<(), String> {
        // Validate new configuration
        super::config::validate_config(&config)?;

        // Apply configuration
        self.config = config;

        // Update subsystems
        if self.active {
            self.apply_config_changes()?;
        }

        Ok(())
    }

    /// Get aggregated statistics
    pub fn get_aggregate_statistics(&self) -> AggregateStatistics {
        let allocator_metrics = self.allocator_aggregator.calculate_aggregate_metrics();
        let pool_metrics = self.pool_aggregator.calculate_system_metrics();

        AggregateStatistics {
            allocator_metrics,
            pool_metrics,
            integration_metrics: self.get_integration_metrics(),
            monitoring_active: self.active,
            total_optimization_suggestions: self.get_optimization_suggestions().len(),
        }
    }

    // Private helper methods

    fn initialize_monitoring(&self) -> Result<(), String> {
        let mut monitoring = self.monitoring_system.write();

        // Set up basic alert conditions
        monitoring.add_alert_condition(super::monitoring::AlertCondition {
            id: "memory_efficiency_alert".to_string(),
            metric_name: "memory_efficiency".to_string(),
            threshold: 0.7,
            comparison: super::config::ComparisonType::LessThan,
            severity: super::config::AlertSeverity::Warning,
            description: "Memory efficiency below threshold".to_string(),
            enabled: true,
            cooldown: Duration::from_secs(300),
            last_triggered: None,
        });

        monitoring.add_alert_condition(super::monitoring::AlertCondition {
            id: "allocation_failure_alert".to_string(),
            metric_name: "allocation_failures".to_string(),
            threshold: 10.0,
            comparison: super::config::ComparisonType::GreaterThan,
            severity: super::config::AlertSeverity::Critical,
            description: "High allocation failure rate".to_string(),
            enabled: true,
            cooldown: Duration::from_secs(60),
            last_triggered: None,
        });

        Ok(())
    }

    fn enable_predictive_modeling(&self) -> Result<(), String> {
        let mut features = self.advanced_features.lock().unwrap();
        features.predictive_engine.enable();
        features.initialize_ml_models();
        Ok(())
    }

    fn enable_auto_optimization(&self) -> Result<(), String> {
        let mut features = self.advanced_features.lock().unwrap();
        features.auto_optimization.enable();
        Ok(())
    }

    fn initialize_event_processing(&self) {
        let mut processor = self.event_processor.lock().unwrap();

        // Add high-severity event filter
        processor.add_filter(super::event_system::EventFilter::high_severity_only());

        // Add performance event filter
        processor.add_filter(super::event_system::EventFilter::performance_events());
    }

    fn cleanup_resources(&self) {
        // Clean up monitoring data
        {
            let mut monitoring = self.monitoring_system.write();
            monitoring.cleanup_old_data();
        }

        // Clean up event processor
        {
            let mut processor = self.event_processor.lock().unwrap();
            processor.clear_statistics();
        }
    }

    fn perform_synchronization(&mut self) -> Result<(), String> {
        // Fetch allocator statistics
        self.fetch_allocator_statistics()?;

        // Fetch pool information
        self.fetch_pool_information()?;

        // Update real-time metrics
        self.update_real_time_metrics();

        // Run anomaly detection if enabled
        if self.config.advanced_config.enable_health_monitoring {
            self.run_anomaly_detection();
        }

        // Process optimization queue if enabled
        if self.config.advanced_config.enable_automated_optimization {
            self.process_optimization_queue();
        }

        Ok(())
    }

    fn fetch_allocator_statistics(&mut self) -> Result<(), String> {
        // Create sample statistics for demonstration
        let sample_stats = ScirS2AllocatorStats::new("scirs2_default".to_string());

        {
            let mut allocator_stats = self.allocator_stats.write();
            allocator_stats.insert(sample_stats.name.clone(), sample_stats.clone());
        }
        self.allocator_aggregator.update_stats(sample_stats);

        Ok(())
    }

    fn fetch_pool_information(&mut self) -> Result<(), String> {
        // Create sample pool information
        let sample_pool = ScirS2PoolInfo::new(
            "tensor_pool_1".to_string(),
            "tensor".to_string(),
            1024 * 1024 * 512, // 512MB
        );

        {
            let mut pool_info = self.pool_info.write();
            pool_info.insert(sample_pool.pool_id.clone(), sample_pool.clone());
        }
        self.pool_aggregator.update_pool(sample_pool);

        Ok(())
    }

    fn update_real_time_metrics(&self) {
        let mut monitoring = self.monitoring_system.write();

        // Update real-time metrics
        {
            let allocator_stats = self.allocator_stats.read();
            for (name, stats) in allocator_stats.iter() {
                monitoring.update_metric(format!("{}_efficiency", name), stats.memory_efficiency);
                monitoring.update_metric(
                    format!("{}_failures", name),
                    stats.allocation_failures as f64,
                );
            }
        }

        {
            let pool_info = self.pool_info.read();
            for (id, pool) in pool_info.iter() {
                monitoring.update_metric(format!("{}_utilization", id), pool.utilization);
            }
        }
    }

    fn run_anomaly_detection(&self) {
        if let Ok(mut features) = self.advanced_features.lock() {
            let metrics = self.collect_current_metrics();
            let anomalies = features.anomaly_detector.detect_anomalies(&metrics);

            // Process detected anomalies
            for anomaly in anomalies {
                let event = ScirS2Event::PerformanceDegradation {
                    allocator: "system".to_string(),
                    metric: anomaly.metric_name,
                    degradation_amount: anomaly.score,
                    threshold_exceeded: anomaly.confidence > 0.8,
                };

                // Process anomaly as event (but avoid infinite recursion)
                self.process_anomaly_event(event);
            }
        }
    }

    fn process_optimization_queue(&self) {
        if let Ok(mut features) = self.advanced_features.lock() {
            let results = features.auto_optimization.process_queue();

            // Update optimization metrics
            if let Ok(mut engine) = self.optimization_engine.lock() {
                for result in results {
                    engine.record_optimization_result(result);
                }
            }
        }
    }

    fn update_state_from_event(&self, event: &ScirS2Event) {
        // Update internal state based on the event type
        match event {
            ScirS2Event::Allocation {
                size, allocator, ..
            } => {
                // Update allocator statistics in a thread-safe way
                let mut allocator_stats = self.allocator_stats.write();
                let stats = allocator_stats
                    .entry(allocator.clone())
                    .or_insert_with(|| ScirS2AllocatorStats::new(allocator.clone()));

                stats.total_allocations += 1;
                stats.current_allocated += *size;
                stats.peak_allocated = stats.peak_allocated.max(stats.current_allocated);

                // Update memory efficiency (simple heuristic)
                if stats.peak_allocated > 0 {
                    stats.memory_efficiency =
                        stats.current_allocated as f64 / stats.peak_allocated as f64;
                }
            }
            ScirS2Event::Deallocation { allocator, .. } => {
                // Update allocator statistics for deallocation
                let mut allocator_stats = self.allocator_stats.write();
                if let Some(stats) = allocator_stats.get_mut(allocator) {
                    stats.total_deallocations += 1;
                    // Note: Size information not available in deallocation event
                    // In a full implementation, we'd track allocations separately
                }
            }
            ScirS2Event::PoolUtilizationChange {
                pool_id,
                new_utilization,
                ..
            } => {
                // Update pool utilization in a thread-safe way
                let mut pool_info = self.pool_info.write();
                if let Some(pool) = pool_info.get_mut(pool_id) {
                    pool.utilization = *new_utilization;
                    // Note: ScirS2PoolInfo doesn't have last_updated field
                }
            }
            ScirS2Event::PoolCreated {
                pool_id,
                capacity,
                pool_type,
                ..
            } => {
                // Create new pool info
                let mut pool_info = self.pool_info.write();
                let new_pool = ScirS2PoolInfo::new(pool_id.clone(), pool_type.clone(), *capacity);
                pool_info.insert(pool_id.clone(), new_pool);
            }
            _ => {}
        }
    }

    fn check_optimization_opportunities(&self, _event: &ScirS2Event) {
        // Analyze the event for optimization opportunities
        // This would trigger the optimization engine in a real implementation
        if let Ok(_features) = self.advanced_features.lock() {
            // Add optimization tasks based on event analysis
        }
    }

    fn update_success_rate(&self) {
        let total_sync = self
            .integration_metrics
            .total_synchronizations
            .load(Ordering::Relaxed);
        let error_count = self.integration_metrics.error_count.load(Ordering::Relaxed);
        let total_operations = total_sync + error_count;

        if total_operations > 0 {
            let new_success_rate = total_sync as f64 / total_operations as f64;
            *self.integration_metrics.success_rate.write() = new_success_rate;
        }
    }

    fn calculate_health_score(&self) -> f64 {
        let mut score = 1.0;

        // Consider success rate
        score *= self.integration_metrics.get_success_rate();

        // Consider allocator health
        {
            let allocator_stats = self.allocator_stats.read();
            let healthy_allocators = allocator_stats.values().filter(|s| s.is_healthy()).count();
            if !allocator_stats.is_empty() {
                score *= healthy_allocators as f64 / allocator_stats.len() as f64;
            }
        }

        // Consider pool health
        {
            let pool_info = self.pool_info.read();
            let healthy_pools = pool_info.values().filter(|p| p.is_healthy()).count();
            if !pool_info.is_empty() {
                score *= healthy_pools as f64 / pool_info.len() as f64;
            }
        }

        score
    }

    fn get_enabled_features(&self) -> Vec<String> {
        let mut features = Vec::new();

        if self.config.enable_realtime_sync {
            features.push("real_time_sync".to_string());
        }
        if self.config.enable_event_callbacks {
            features.push("event_callbacks".to_string());
        }
        if self.config.track_allocation_patterns {
            features.push("allocation_patterns".to_string());
        }
        if self.config.enable_optimization_suggestions {
            features.push("optimization_suggestions".to_string());
        }
        if self.config.advanced_config.enable_predictive_modeling {
            features.push("predictive_modeling".to_string());
        }
        if self.config.advanced_config.enable_automated_optimization {
            features.push("automated_optimization".to_string());
        }
        if self.config.advanced_config.enable_health_monitoring {
            features.push("health_monitoring".to_string());
        }
        if self.config.advanced_config.enable_performance_profiling {
            features.push("performance_profiling".to_string());
        }

        features
    }

    fn apply_config_changes(&mut self) -> Result<(), String> {
        // Update monitoring system configuration
        {
            let mut monitoring = self.monitoring_system.write();
            let monitoring_config = super::monitoring::MonitoringConfig {
                enabled: self.config.enable_realtime_sync,
                collection_interval: self.config.sync_interval,
                alert_check_interval: self.config.advanced_config.health_check_interval,
                max_historical_points: 10000,
                enable_aggregation: true,
                alert_throttle_duration: Duration::from_secs(60),
            };
            monitoring.update_config(monitoring_config);
        }

        // Update advanced features
        if self.config.advanced_config.enable_predictive_modeling {
            self.enable_predictive_modeling()?;
        }

        if self.config.advanced_config.enable_automated_optimization {
            self.enable_auto_optimization()?;
        }

        Ok(())
    }

    fn collect_current_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // Collect allocator metrics
        {
            let allocator_stats = self.allocator_stats.read();
            for (name, stats) in allocator_stats.iter() {
                metrics.insert(format!("{}_efficiency", name), stats.memory_efficiency);
                metrics.insert(
                    format!("{}_failures", name),
                    stats.allocation_failures as f64,
                );
                metrics.insert(
                    format!("{}_allocated", name),
                    stats.current_allocated as f64,
                );
            }
        }

        // Collect pool metrics
        {
            let pool_info = self.pool_info.read();
            for (id, pool) in pool_info.iter() {
                metrics.insert(format!("{}_utilization", id), pool.utilization);
            }
        }

        metrics
    }

    fn process_anomaly_event(&self, event: ScirS2Event) {
        // Process anomaly event without triggering recursion
        // This would typically just log or alert, not trigger full event processing
        println!("Anomaly detected: {:?}", event);
    }
}

impl IntegrationMetrics {
    fn new() -> Self {
        Self {
            total_events_processed: AtomicU64::new(0),
            total_synchronizations: AtomicU64::new(0),
            avg_sync_time: Arc::new(RwLock::new(Duration::from_secs(0))),
            uptime: Arc::new(RwLock::new(Duration::from_secs(0))),
            start_time: Instant::now(),
            error_count: AtomicU64::new(0),
            success_rate: Arc::new(RwLock::new(1.0)),
        }
    }

    /// Get total events processed
    pub fn get_total_events_processed(&self) -> u64 {
        self.total_events_processed.load(Ordering::Relaxed)
    }

    /// Get total synchronizations
    pub fn get_total_synchronizations(&self) -> u64 {
        self.total_synchronizations.load(Ordering::Relaxed)
    }

    /// Get error count
    pub fn get_error_count(&self) -> u64 {
        self.error_count.load(Ordering::Relaxed)
    }

    /// Get success rate
    pub fn get_success_rate(&self) -> f64 {
        *self.success_rate.read()
    }

    /// Get average sync time
    pub fn get_avg_sync_time(&self) -> Duration {
        *self.avg_sync_time.read()
    }

    /// Get uptime
    pub fn get_uptime(&self) -> Duration {
        *self.uptime.read()
    }
}

/// Aggregate statistics across all components
#[derive(Debug, Clone)]
pub struct AggregateStatistics {
    pub allocator_metrics: super::statistics::AggregateMetrics,
    pub pool_metrics: super::pool_management::SystemPoolMetrics,
    pub integration_metrics: IntegrationMetricsSnapshot,
    pub monitoring_active: bool,
    pub total_optimization_suggestions: usize,
}

// Extension trait for ScirS2OptimizationEngine
impl ScirS2OptimizationEngine {
    pub fn new() -> Self {
        Self {
            optimization_history: Vec::new(),
            active_optimizations: HashMap::new(),
            optimization_metrics: OptimizationMetrics {
                total_optimizations: 0,
                successful_optimizations: 0,
                average_improvement: 0.0,
                optimization_efficiency: 0.0,
                total_optimization_time: Duration::from_secs(0),
            },
            config: super::optimization::OptimizationConfig::default(),
            baselines: HashMap::new(),
        }
    }

    pub fn get_ml_suggestions(&self) -> Vec<PerformanceHint> {
        // Placeholder for ML-generated suggestions
        Vec::new()
    }

    pub fn record_optimization_result(&mut self, result: super::optimization::OptimizationResult) {
        self.optimization_history.push(result.clone());

        // Update metrics
        self.optimization_metrics.total_optimizations += 1;
        if result.success {
            self.optimization_metrics.successful_optimizations += 1;
        }

        // Update average improvement
        if self.optimization_metrics.total_optimizations > 0 {
            let total_improvement = self
                .optimization_history
                .iter()
                .map(|r| r.performance_improvement)
                .sum::<f64>();
            self.optimization_metrics.average_improvement =
                total_improvement / self.optimization_metrics.total_optimizations as f64;
        }

        // Update efficiency
        self.optimization_metrics.optimization_efficiency =
            self.optimization_metrics.successful_optimizations as f64
                / self.optimization_metrics.total_optimizations.max(1) as f64;
    }
}
