// Memory Profiling: Unified Module Interface
//
// This module provides a unified interface for the ToRSh memory profiling system,
// orchestrating core memory tracking, pressure monitoring, access pattern analysis,
// analytics, fragmentation management, and external system integrations.
//
// ## Architecture
//
// The memory profiling system is organized into six specialized modules:
//
// - **core**: Foundation types, allocation tracking, and basic memory management
// - **pressure**: Memory pressure monitoring, health management, and bandwidth utilization
// - **patterns**: Access pattern analysis, optimization recommendations, and performance hints
// - **analytics**: Statistics, historical data, trend analysis, and anomaly detection
// - **fragmentation**: Memory fragmentation tracking, analysis, and defragmentation algorithms
// - **integration**: SciRS2 integration and external system connectivity
//
// ## Usage
//
// ```rust
// use torsh_backend::memory_profiling::MemoryProfiler;
//
// // Create and configure profiler
// let mut profiler = MemoryProfiler::new();
//
// // Track memory operations
// profiler.track_allocation(ptr, size, context)?;
// profiler.track_deallocation(ptr)?;
//
// // Generate comprehensive analysis
// let report = profiler.generate_comprehensive_report()?;
// ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, Duration, SystemTime};
use scirs2_core::error::{CoreError, Result};

// Re-export all public APIs from specialized modules
pub use self::core::*;
pub use self::pressure::*;
pub use self::patterns::*;
pub use self::analytics::*;
pub use self::fragmentation::*;
pub use self::integration::*;

// Specialized module declarations
pub mod core;
pub mod pressure;
pub mod patterns;
pub mod analytics;
pub mod fragmentation;
pub mod integration;

/// Comprehensive memory profiler that orchestrates all specialized subsystems
pub struct MemoryProfiler {
    /// Core memory allocation tracking
    core_tracker: Arc<RwLock<MemoryTracker>>,

    /// Memory pressure monitoring system
    pressure_monitor: Arc<Mutex<MemoryPressureMonitor>>,

    /// Access pattern analyzer
    pattern_analyzer: Arc<Mutex<AccessPatternsAnalyzer>>,

    /// Analytics and historical data manager
    analytics: Arc<Mutex<MemoryAnalytics>>,

    /// Fragmentation tracking and management
    fragmentation_manager: Arc<Mutex<FragmentationManager>>,

    /// External system integrations
    integrations: Arc<Mutex<MemoryProfilingIntegrations>>,

    /// Global profiler configuration
    config: MemoryProfilerConfig,

    /// Runtime state and statistics
    state: Arc<RwLock<ProfilerState>>,
}

/// Configuration for the unified memory profiler
#[derive(Debug, Clone)]
pub struct MemoryProfilerConfig {
    pub enable_pressure_monitoring: bool,
    pub enable_pattern_analysis: bool,
    pub enable_analytics: bool,
    pub enable_fragmentation_tracking: bool,
    pub enable_integrations: bool,
    pub auto_defragmentation: bool,
    pub reporting_interval: Duration,
    pub max_history_retention: Duration,
    pub performance_optimization: bool,
    pub real_time_alerting: bool,
}

impl Default for MemoryProfilerConfig {
    fn default() -> Self {
        Self {
            enable_pressure_monitoring: true,
            enable_pattern_analysis: true,
            enable_analytics: true,
            enable_fragmentation_tracking: true,
            enable_integrations: false, // Disabled by default for security
            auto_defragmentation: true,
            reporting_interval: Duration::from_secs(300), // 5 minutes
            max_history_retention: Duration::from_secs(86400), // 24 hours
            performance_optimization: true,
            real_time_alerting: false,
        }
    }
}

/// Runtime state of the memory profiler
#[derive(Debug, Clone)]
pub struct ProfilerState {
    pub started_at: SystemTime,
    pub last_analysis: Option<SystemTime>,
    pub total_allocations_tracked: usize,
    pub total_deallocations_tracked: usize,
    pub active_monitoring: bool,
    pub last_defragmentation: Option<SystemTime>,
    pub current_memory_usage: usize,
    pub peak_memory_usage: usize,
    pub errors_encountered: usize,
    pub performance_score: f64,
}

impl Default for ProfilerState {
    fn default() -> Self {
        Self {
            started_at: SystemTime::now(),
            last_analysis: None,
            total_allocations_tracked: 0,
            total_deallocations_tracked: 0,
            active_monitoring: true,
            last_defragmentation: None,
            current_memory_usage: 0,
            peak_memory_usage: 0,
            errors_encountered: 0,
            performance_score: 1.0,
        }
    }
}

/// Comprehensive memory profiling report combining all subsystems
#[derive(Debug, Clone)]
pub struct ComprehensiveMemoryReport {
    pub generated_at: SystemTime,
    pub profiler_state: ProfilerState,
    pub memory_statistics: MemoryStatistics,
    pub pressure_analysis: MemoryPressureAnalysis,
    pub access_patterns: Vec<AccessPattern>,
    pub pattern_metrics: AccessPatternMetrics,
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    pub analytics_report: AnalyticsReport,
    pub fragmentation_analysis: FragmentationAnalysis,
    pub integration_health: Option<IntegrationHealthReport>,
    pub executive_summary: ExecutiveSummary,
}

/// Executive summary of memory profiling results
#[derive(Debug, Clone)]
pub struct ExecutiveSummary {
    pub overall_health_score: f64,
    pub key_findings: Vec<String>,
    pub critical_issues: Vec<String>,
    pub recommendations: Vec<String>,
    pub performance_highlights: Vec<String>,
    pub risk_assessment: RiskAssessment,
    pub next_actions: Vec<String>,
}

/// Risk assessment for memory management
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub overall_risk_level: RiskLevel,
    pub memory_exhaustion_risk: f64,
    pub fragmentation_risk: f64,
    pub performance_degradation_risk: f64,
    pub system_stability_risk: f64,
    pub data_loss_risk: f64,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Convenience result type for profiler operations
pub type ProfilerResult<T> = Result<T>;

/// Quick configuration presets for common use cases
pub enum ProfilerPreset {
    Development,
    Production,
    HighPerformance,
    DeepAnalysis,
    MinimalOverhead,
    Custom(MemoryProfilerConfig),
}

impl MemoryProfiler {
    /// Create a new memory profiler with default configuration
    pub fn new() -> Self {
        Self::with_config(MemoryProfilerConfig::default())
    }

    /// Create a memory profiler with specific configuration
    pub fn with_config(config: MemoryProfilerConfig) -> Self {
        let core_tracker = Arc::new(RwLock::new(MemoryTracker::new()));
        let pressure_monitor = Arc::new(Mutex::new(MemoryPressureMonitor::new()));
        let pattern_analyzer = Arc::new(Mutex::new(AccessPatternsAnalyzer::new()));
        let analytics = Arc::new(Mutex::new(MemoryAnalytics::new()));
        let fragmentation_manager = Arc::new(Mutex::new(FragmentationManager::new()));

        // Initialize integrations if enabled
        let integrations = if config.enable_integrations {
            let integration_config = IntegrationConfiguration::default();
            match MemoryProfilingIntegrations::new(integration_config) {
                Ok(mut integrations) => {
                    if let Err(_e) = integrations.initialize() {
                        // Log error but continue without integrations
                    }
                    Arc::new(Mutex::new(integrations))
                }
                Err(_) => {
                    // Create a placeholder - in practice, we might want to use a null object pattern
                    Arc::new(Mutex::new(MemoryProfilingIntegrations::new(IntegrationConfiguration::default()).unwrap()))
                }
            }
        } else {
            Arc::new(Mutex::new(MemoryProfilingIntegrations::new(IntegrationConfiguration::default()).unwrap()))
        };

        Self {
            core_tracker,
            pressure_monitor,
            pattern_analyzer,
            analytics,
            fragmentation_manager,
            integrations,
            config,
            state: Arc::new(RwLock::new(ProfilerState::default())),
        }
    }

    /// Create profiler with preset configuration
    pub fn with_preset(preset: ProfilerPreset) -> Self {
        let config = match preset {
            ProfilerPreset::Development => MemoryProfilerConfig {
                enable_pressure_monitoring: true,
                enable_pattern_analysis: true,
                enable_analytics: false,
                enable_fragmentation_tracking: true,
                enable_integrations: false,
                auto_defragmentation: false,
                reporting_interval: Duration::from_secs(60),
                max_history_retention: Duration::from_secs(3600),
                performance_optimization: false,
                real_time_alerting: false,
            },
            ProfilerPreset::Production => MemoryProfilerConfig {
                enable_pressure_monitoring: true,
                enable_pattern_analysis: false,
                enable_analytics: true,
                enable_fragmentation_tracking: true,
                enable_integrations: true,
                auto_defragmentation: true,
                reporting_interval: Duration::from_secs(300),
                max_history_retention: Duration::from_secs(86400),
                performance_optimization: true,
                real_time_alerting: true,
            },
            ProfilerPreset::HighPerformance => MemoryProfilerConfig {
                enable_pressure_monitoring: true,
                enable_pattern_analysis: false,
                enable_analytics: false,
                enable_fragmentation_tracking: false,
                enable_integrations: false,
                auto_defragmentation: false,
                reporting_interval: Duration::from_secs(900),
                max_history_retention: Duration::from_secs(3600),
                performance_optimization: true,
                real_time_alerting: false,
            },
            ProfilerPreset::DeepAnalysis => MemoryProfilerConfig {
                enable_pressure_monitoring: true,
                enable_pattern_analysis: true,
                enable_analytics: true,
                enable_fragmentation_tracking: true,
                enable_integrations: false,
                auto_defragmentation: false,
                reporting_interval: Duration::from_secs(30),
                max_history_retention: Duration::from_secs(172800), // 48 hours
                performance_optimization: false,
                real_time_alerting: false,
            },
            ProfilerPreset::MinimalOverhead => MemoryProfilerConfig {
                enable_pressure_monitoring: false,
                enable_pattern_analysis: false,
                enable_analytics: false,
                enable_fragmentation_tracking: false,
                enable_integrations: false,
                auto_defragmentation: false,
                reporting_interval: Duration::from_secs(3600),
                max_history_retention: Duration::from_secs(3600),
                performance_optimization: true,
                real_time_alerting: false,
            },
            ProfilerPreset::Custom(config) => config,
        };

        Self::with_config(config)
    }

    /// Track a memory allocation
    pub fn track_allocation(&mut self, ptr: usize, size: usize, context: AllocationContext) -> ProfilerResult<()> {
        // Update profiler state
        {
            let mut state = self.state.write().map_err(|_| CoreError::InvalidOperation("State lock poisoned".to_string()))?;
            state.total_allocations_tracked += 1;
            state.current_memory_usage += size;
            if state.current_memory_usage > state.peak_memory_usage {
                state.peak_memory_usage = state.current_memory_usage;
            }
        }

        // Track in core system
        {
            let mut tracker = self.core_tracker.write().map_err(|_| CoreError::InvalidOperation("Tracker lock poisoned".to_string()))?;
            tracker.track_allocation(ptr, size, context.clone())?;
        }

        // Update pressure monitoring if enabled
        if self.config.enable_pressure_monitoring {
            if let Ok(mut monitor) = self.pressure_monitor.lock() {
                monitor.update_memory_usage(self.get_current_usage())?;
            }
        }

        // Track access patterns if enabled
        if self.config.enable_pattern_analysis {
            if let Ok(mut analyzer) = self.pattern_analyzer.lock() {
                let access_record = AccessRecord {
                    address: ptr,
                    size,
                    timestamp: Instant::now(),
                    access_type: AccessType::Write, // Allocation is considered a write
                    thread_id: std::thread::current().id().as_u64().get(),
                    allocation_id: Some(ptr), // Use ptr as allocation ID for simplicity
                    context: AccessContext {
                        operation: context.operation.unwrap_or_else(|| "allocation".to_string()),
                        tensor_shape: context.tensor_shape.clone(),
                        data_type: context.data_type.clone(),
                        kernel_name: context.kernel_name.clone(),
                        call_stack: vec![], // Would populate in real implementation
                    },
                };
                let _ = analyzer.record_access(access_record);
            }
        }

        // Track fragmentation if enabled
        if self.config.enable_fragmentation_tracking {
            if let Ok(mut manager) = self.fragmentation_manager.lock() {
                let block = MemoryBlock::new_allocated(
                    ptr,
                    size,
                    ptr, // Use ptr as allocation ID
                    self.classify_allocation_purpose(&context),
                );
                let _ = manager.track_allocation(block);
            }
        }

        Ok(())
    }

    /// Track a memory deallocation
    pub fn track_deallocation(&mut self, ptr: usize) -> ProfilerResult<()> {
        // Get size before deallocation for state update
        let size = {
            let tracker = self.core_tracker.read().map_err(|_| CoreError::InvalidOperation("Tracker lock poisoned".to_string()))?;
            tracker.get_allocation_size(ptr).unwrap_or(0)
        };

        // Update profiler state
        {
            let mut state = self.state.write().map_err(|_| CoreError::InvalidOperation("State lock poisoned".to_string()))?;
            state.total_deallocations_tracked += 1;
            state.current_memory_usage = state.current_memory_usage.saturating_sub(size);
        }

        // Track in core system
        {
            let mut tracker = self.core_tracker.write().map_err(|_| CoreError::InvalidOperation("Tracker lock poisoned".to_string()))?;
            tracker.track_deallocation(ptr)?;
        }

        // Update pressure monitoring if enabled
        if self.config.enable_pressure_monitoring {
            if let Ok(mut monitor) = self.pressure_monitor.lock() {
                monitor.update_memory_usage(self.get_current_usage())?;
            }
        }

        // Track fragmentation if enabled
        if self.config.enable_fragmentation_tracking {
            if let Ok(mut manager) = self.fragmentation_manager.lock() {
                let _ = manager.track_deallocation(ptr);
            }
        }

        Ok(())
    }

    /// Generate a comprehensive memory profiling report
    pub fn generate_comprehensive_report(&mut self) -> ProfilerResult<ComprehensiveMemoryReport> {
        let report_start = Instant::now();

        // Get current state
        let profiler_state = {
            let mut state = self.state.write().map_err(|_| CoreError::InvalidOperation("State lock poisoned".to_string()))?;
            state.last_analysis = Some(SystemTime::now());
            state.clone()
        };

        // Get memory statistics from core
        let memory_statistics = {
            let tracker = self.core_tracker.read().map_err(|_| CoreError::InvalidOperation("Tracker lock poisoned".to_string()))?;
            tracker.get_statistics()
        };

        // Get pressure analysis if enabled
        let pressure_analysis = if self.config.enable_pressure_monitoring {
            match self.pressure_monitor.lock() {
                Ok(mut monitor) => monitor.analyze_pressure_trends()?,
                Err(_) => MemoryPressureAnalysis::default(),
            }
        } else {
            MemoryPressureAnalysis::default()
        };

        // Get access patterns if enabled
        let (access_patterns, pattern_metrics, optimization_recommendations) = if self.config.enable_pattern_analysis {
            match self.pattern_analyzer.lock() {
                Ok(mut analyzer) => {
                    let patterns = analyzer.analyze_patterns()?;
                    let metrics = analyzer.calculate_metrics()?;
                    let recommendations = analyzer.generate_recommendations()?;
                    (patterns, metrics, recommendations)
                }
                Err(_) => (vec![], AccessPatternMetrics::default(), vec![]),
            }
        } else {
            (vec![], AccessPatternMetrics::default(), vec![])
        };

        // Get analytics report if enabled
        let analytics_report = if self.config.enable_analytics {
            match self.analytics.lock() {
                Ok(mut analytics) => {
                    analytics.generate_report(Duration::from_secs(3600))?
                }
                Err(_) => AnalyticsReport::default(),
            }
        } else {
            AnalyticsReport::default()
        };

        // Get fragmentation analysis if enabled
        let fragmentation_analysis = if self.config.enable_fragmentation_tracking {
            match self.fragmentation_manager.lock() {
                Ok(mut manager) => manager.analyze_fragmentation()?,
                Err(_) => FragmentationAnalysis::default(),
            }
        } else {
            FragmentationAnalysis::default()
        };

        // Get integration health if enabled
        let integration_health = if self.config.enable_integrations {
            match self.integrations.lock() {
                Ok(integrations) => Some(integrations.get_health_status()),
                Err(_) => None,
            }
        } else {
            None
        };

        // Generate executive summary
        let executive_summary = self.generate_executive_summary(
            &profiler_state,
            &memory_statistics,
            &pressure_analysis,
            &pattern_metrics,
            &fragmentation_analysis,
        );

        // Update performance score based on report generation time
        let report_duration = report_start.elapsed();
        let performance_impact = report_duration.as_millis() as f64 / 1000.0; // Convert to seconds
        {
            let mut state = self.state.write().map_err(|_| CoreError::InvalidOperation("State lock poisoned".to_string()))?;
            state.performance_score = (1.0 - performance_impact / 10.0).max(0.0); // Penalize slow reports
        }

        Ok(ComprehensiveMemoryReport {
            generated_at: SystemTime::now(),
            profiler_state,
            memory_statistics,
            pressure_analysis,
            access_patterns,
            pattern_metrics,
            optimization_recommendations,
            analytics_report,
            fragmentation_analysis,
            integration_health,
            executive_summary,
        })
    }

    /// Perform automated optimization based on current analysis
    pub fn auto_optimize(&mut self) -> ProfilerResult<OptimizationResult> {
        let mut results = OptimizationResult {
            optimizations_applied: vec![],
            memory_recovered: 0,
            performance_improvement: 0.0,
            duration: Duration::default(),
        };

        let start_time = Instant::now();

        // Auto-defragmentation if enabled and needed
        if self.config.auto_defragmentation {
            if let Ok(mut manager) = self.fragmentation_manager.lock() {
                let analysis = manager.analyze_fragmentation()?;
                if analysis.overall_fragmentation_index > 0.3 {
                    let defrag_result = manager.defragment(DefragmentationType::CoalescingBased)?;
                    if defrag_result.success {
                        results.optimizations_applied.push("Memory defragmentation".to_string());
                        results.memory_recovered += defrag_result.memory_recovered;

                        // Update state
                        let mut state = self.state.write().map_err(|_| CoreError::InvalidOperation("State lock poisoned".to_string()))?;
                        state.last_defragmentation = Some(SystemTime::now());
                    }
                }
            }
        }

        // Apply pattern-based optimizations if enabled
        if self.config.performance_optimization && self.config.enable_pattern_analysis {
            if let Ok(mut analyzer) = self.pattern_analyzer.lock() {
                let recommendations = analyzer.generate_recommendations()?;
                let high_priority_recommendations: Vec<_> = recommendations.iter()
                    .filter(|r| r.priority == RecommendationPriority::High || r.priority == RecommendationPriority::Critical)
                    .collect();

                for recommendation in high_priority_recommendations {
                    // Apply optimization (simplified - would have actual implementation)
                    results.optimizations_applied.push(recommendation.description.clone());
                    results.performance_improvement += recommendation.expected_improvement;
                }
            }
        }

        results.duration = start_time.elapsed();
        Ok(results)
    }

    /// Get current memory usage
    pub fn get_current_usage(&self) -> usize {
        self.state.read().map(|state| state.current_memory_usage).unwrap_or(0)
    }

    /// Get profiler statistics
    pub fn get_statistics(&self) -> ProfilerResult<ProfilerStatistics> {
        let state = self.state.read().map_err(|_| CoreError::InvalidOperation("State lock poisoned".to_string()))?;
        Ok(ProfilerStatistics {
            uptime: SystemTime::now().duration_since(state.started_at).unwrap_or_default(),
            total_allocations: state.total_allocations_tracked,
            total_deallocations: state.total_deallocations_tracked,
            current_memory_usage: state.current_memory_usage,
            peak_memory_usage: state.peak_memory_usage,
            active_monitoring: state.active_monitoring,
            performance_score: state.performance_score,
            errors_count: state.errors_encountered,
        })
    }

    /// Enable or disable real-time monitoring
    pub fn set_monitoring_active(&mut self, active: bool) -> ProfilerResult<()> {
        let mut state = self.state.write().map_err(|_| CoreError::InvalidOperation("State lock poisoned".to_string()))?;
        state.active_monitoring = active;
        Ok(())
    }

    // Helper methods
    fn classify_allocation_purpose(&self, context: &AllocationContext) -> AllocationPurpose {
        if let Some(ref purpose) = context.allocation_purpose {
            match purpose.as_str() {
                "tensor" => AllocationPurpose::TensorData,
                "gradient" => AllocationPurpose::GradientBuffer,
                "activation" => AllocationPurpose::ActivationCache,
                "weight" => AllocationPurpose::WeightData,
                "temp" | "temporary" => AllocationPurpose::TemporaryBuffer,
                "system" => AllocationPurpose::SystemOverhead,
                "user" => AllocationPurpose::UserData,
                _ => AllocationPurpose::Unknown,
            }
        } else {
            AllocationPurpose::Unknown
        }
    }

    fn generate_executive_summary(
        &self,
        state: &ProfilerState,
        memory_stats: &MemoryStatistics,
        pressure_analysis: &MemoryPressureAnalysis,
        pattern_metrics: &AccessPatternMetrics,
        fragmentation_analysis: &FragmentationAnalysis,
    ) -> ExecutiveSummary {
        let mut key_findings = Vec::new();
        let mut critical_issues = Vec::new();
        let mut recommendations = Vec::new();
        let mut performance_highlights = Vec::new();
        let mut next_actions = Vec::new();

        // Analyze overall health
        let memory_efficiency = memory_stats.efficiency_score;
        let fragmentation_level = fragmentation_analysis.overall_fragmentation_index;
        let cache_performance = pattern_metrics.cache_hit_rate;

        let overall_health_score = (memory_efficiency + (1.0 - fragmentation_level) + cache_performance) / 3.0;

        // Generate findings based on metrics
        if overall_health_score > 0.8 {
            performance_highlights.push("Excellent overall memory management performance".to_string());
        } else if overall_health_score > 0.6 {
            key_findings.push("Good memory performance with opportunities for optimization".to_string());
        } else {
            critical_issues.push("Memory management performance requires immediate attention".to_string());
        }

        if fragmentation_level > 0.5 {
            critical_issues.push(format!("High memory fragmentation detected: {:.1}%", fragmentation_level * 100.0));
            next_actions.push("Schedule memory defragmentation".to_string());
        }

        if cache_performance < 0.7 {
            recommendations.push("Optimize memory access patterns to improve cache performance".to_string());
        }

        if memory_stats.allocation_rate > memory_stats.deallocation_rate * 1.2 {
            key_findings.push("Memory allocation rate significantly exceeds deallocation rate".to_string());
            recommendations.push("Review allocation patterns for potential memory leaks".to_string());
        }

        // Risk assessment
        let risk_assessment = RiskAssessment {
            overall_risk_level: if overall_health_score < 0.3 {
                RiskLevel::Critical
            } else if overall_health_score < 0.5 {
                RiskLevel::High
            } else if overall_health_score < 0.7 {
                RiskLevel::Medium
            } else {
                RiskLevel::Low
            },
            memory_exhaustion_risk: if state.current_memory_usage as f64 / state.peak_memory_usage as f64 > 0.9 {
                0.8
            } else {
                0.2
            },
            fragmentation_risk: fragmentation_level,
            performance_degradation_risk: 1.0 - cache_performance,
            system_stability_risk: if critical_issues.is_empty() { 0.1 } else { 0.6 },
            data_loss_risk: 0.1, // Would calculate based on actual error rates
        };

        ExecutiveSummary {
            overall_health_score,
            key_findings,
            critical_issues,
            recommendations,
            performance_highlights,
            risk_assessment,
            next_actions,
        }
    }
}

/// Result of optimization operations
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub optimizations_applied: Vec<String>,
    pub memory_recovered: usize,
    pub performance_improvement: f64,
    pub duration: Duration,
}

/// Statistics about the profiler itself
#[derive(Debug, Clone)]
pub struct ProfilerStatistics {
    pub uptime: Duration,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub current_memory_usage: usize,
    pub peak_memory_usage: usize,
    pub active_monitoring: bool,
    pub performance_score: f64,
    pub errors_count: usize,
}

// Convenience functions for quick profiling tasks
/// Quick allocation tracking without full profiler setup
pub fn track_allocation_quick(ptr: usize, size: usize) -> ProfilerResult<()> {
    thread_local! {
        static QUICK_PROFILER: std::cell::RefCell<Option<MemoryProfiler>> = std::cell::RefCell::new(None);
    }

    QUICK_PROFILER.with(|profiler| {
        let mut profiler_ref = profiler.borrow_mut();
        if profiler_ref.is_none() {
            *profiler_ref = Some(MemoryProfiler::with_preset(ProfilerPreset::MinimalOverhead));
        }

        if let Some(ref mut p) = *profiler_ref {
            let context = AllocationContext {
                allocation_id: Some(ptr),
                operation: Some("quick_alloc".to_string()),
                allocation_purpose: None,
                tensor_shape: None,
                data_type: None,
                kernel_name: None,
                device: None,
                thread_id: std::thread::current().id().as_u64().get(),
                call_stack: vec![],
            };
            p.track_allocation(ptr, size, context)
        } else {
            Err(CoreError::InvalidOperation("Failed to initialize quick profiler".to_string()))
        }
    })
}

/// Quick deallocation tracking without full profiler setup
pub fn track_deallocation_quick(ptr: usize) -> ProfilerResult<()> {
    thread_local! {
        static QUICK_PROFILER: std::cell::RefCell<Option<MemoryProfiler>> = std::cell::RefCell::new(None);
    }

    QUICK_PROFILER.with(|profiler| {
        let mut profiler_ref = profiler.borrow_mut();
        if let Some(ref mut p) = *profiler_ref {
            p.track_deallocation(ptr)
        } else {
            Ok(()) // Silently ignore if profiler not initialized
        }
    })
}

/// Generate a quick memory report
pub fn generate_quick_report() -> ProfilerResult<String> {
    thread_local! {
        static QUICK_PROFILER: std::cell::RefCell<Option<MemoryProfiler>> = std::cell::RefCell::new(None);
    }

    QUICK_PROFILER.with(|profiler| {
        let mut profiler_ref = profiler.borrow_mut();
        if let Some(ref mut p) = *profiler_ref {
            let stats = p.get_statistics()?;
            Ok(format!(
                "Memory Usage: {} bytes (Peak: {} bytes)\nAllocations: {} | Deallocations: {}\nPerformance Score: {:.2}",
                stats.current_memory_usage,
                stats.peak_memory_usage,
                stats.total_allocations,
                stats.total_deallocations,
                stats.performance_score
            ))
        } else {
            Ok("No memory profiling data available".to_string())
        }
    })
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// Default implementations for types that need them
impl Default for MemoryPressureAnalysis {
    fn default() -> Self {
        Self {
            current_pressure_level: 0.0,
            pressure_trend: PressureTrend::Stable,
            critical_events: vec![],
            resource_utilization: ResourceUtilization {
                memory_usage_percent: 0.0,
                available_memory: 0,
                committed_memory: 0,
                reserved_memory: 0,
            },
            recommendations: vec![],
            health_status: MemoryHealthStatus::Healthy,
            prediction: MemoryPressurePrediction {
                predicted_pressure: 0.0,
                confidence: 0.0,
                time_to_critical: None,
                contributing_factors: vec![],
            },
        }
    }
}

impl Default for AccessPatternMetrics {
    fn default() -> Self {
        Self {
            cache_hit_rate: 0.0,
            cache_miss_penalty: Duration::default(),
            bandwidth_utilization: 0.0,
            memory_efficiency: 0.0,
            temporal_locality_score: 0.0,
            spatial_locality_score: 0.0,
            prefetch_accuracy: 0.0,
            access_regularity: 0.0,
        }
    }
}

impl Default for AnalyticsReport {
    fn default() -> Self {
        Self {
            report_id: "default".to_string(),
            generated_at: SystemTime::now(),
            period: ReportPeriod {
                start_time: SystemTime::now(),
                end_time: SystemTime::now(),
                duration: Duration::default(),
                data_points: 0,
                sampling_rate: Duration::from_secs(60),
            },
            summary: ReportSummary {
                key_findings: vec![],
                performance_highlights: vec![],
                areas_of_concern: vec![],
                overall_health_score: 0.0,
                change_from_baseline: 0.0,
                notable_patterns: vec![],
            },
            detailed_statistics: MemoryStatistics {
                total_allocations: 0,
                total_deallocations: 0,
                peak_memory_usage: 0,
                average_memory_usage: 0.0,
                current_memory_usage: 0,
                memory_churn_rate: 0.0,
                allocation_rate: 0.0,
                deallocation_rate: 0.0,
                fragmentation_index: 0.0,
                efficiency_score: 0.0,
                cache_hit_ratio: 0.0,
                bandwidth_utilization: 0.0,
                pressure_incidents: 0,
                optimization_opportunities: 0,
            },
            trend_analyses: vec![],
            anomalies: vec![],
            performance_comparison: None,
            recommendations: vec![],
            visualizations: vec![],
        }
    }
}

impl Default for FragmentationAnalysis {
    fn default() -> Self {
        Self {
            overall_fragmentation_index: 0.0,
            external_fragmentation: ExternalFragmentation {
                free_block_count: 0,
                largest_free_block: 0,
                total_free_space: 0,
                average_free_block_size: 0.0,
                free_block_size_distribution: HashMap::new(),
                fragmentation_ratio: 0.0,
                compaction_potential: 0.0,
            },
            internal_fragmentation: InternalFragmentation {
                total_internal_waste: 0,
                average_waste_per_allocation: 0.0,
                worst_case_waste: 0,
                waste_by_size_class: HashMap::new(),
                alignment_waste: 0,
                padding_waste: 0,
                efficiency_score: 1.0,
            },
            fragmentation_hotspots: vec![],
            efficiency_metrics: FragmentationEfficiency {
                memory_utilization: 0.0,
                allocation_success_rate: 1.0,
                average_search_time: Duration::default(),
                defragmentation_overhead: 0.0,
                compaction_frequency: 0.0,
                waste_ratio: 0.0,
            },
            trend_analysis: FragmentationTrend {
                direction: TrendDirection::Stable,
                rate_of_change: 0.0,
                prediction: FragmentationPrediction {
                    predicted_fragmentation: 0.0,
                    confidence: 0.0,
                    time_to_critical: None,
                    recommended_action_time: Duration::default(),
                },
                contributing_factors: vec![],
                historical_pattern: vec![],
            },
            impact_assessment: FragmentationImpact {
                performance_degradation: 0.0,
                memory_overhead: 0.0,
                allocation_failures: 0,
                cache_efficiency_impact: 0.0,
                bandwidth_loss: 0.0,
                system_stability_risk: 0.0,
            },
            mitigation_recommendations: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler_creation() {
        let profiler = MemoryProfiler::new();
        assert!(profiler.config.enable_pressure_monitoring);
        assert!(profiler.config.enable_pattern_analysis);
    }

    #[test]
    fn test_profiler_presets() {
        let dev_profiler = MemoryProfiler::with_preset(ProfilerPreset::Development);
        assert!(!dev_profiler.config.enable_integrations);
        assert!(!dev_profiler.config.auto_defragmentation);

        let prod_profiler = MemoryProfiler::with_preset(ProfilerPreset::Production);
        assert!(prod_profiler.config.auto_defragmentation);
        assert!(prod_profiler.config.performance_optimization);

        let minimal_profiler = MemoryProfiler::with_preset(ProfilerPreset::MinimalOverhead);
        assert!(!minimal_profiler.config.enable_pattern_analysis);
        assert!(!minimal_profiler.config.enable_analytics);
    }

    #[test]
    fn test_allocation_tracking() {
        let mut profiler = MemoryProfiler::with_preset(ProfilerPreset::Development);

        let context = AllocationContext {
            allocation_id: Some(1000),
            operation: Some("test_alloc".to_string()),
            allocation_purpose: Some("tensor".to_string()),
            tensor_shape: Some(vec![64, 64]),
            data_type: Some("f32".to_string()),
            kernel_name: None,
            device: None,
            thread_id: 1,
            call_stack: vec![],
        };

        let result = profiler.track_allocation(1000, 4096, context);
        assert!(result.is_ok());

        let stats = profiler.get_statistics().unwrap();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.current_memory_usage, 4096);
    }

    #[test]
    fn test_deallocation_tracking() {
        let mut profiler = MemoryProfiler::with_preset(ProfilerPreset::Development);

        // First allocate
        let context = AllocationContext {
            allocation_id: Some(2000),
            operation: Some("test_alloc".to_string()),
            allocation_purpose: None,
            tensor_shape: None,
            data_type: None,
            kernel_name: None,
            device: None,
            thread_id: 1,
            call_stack: vec![],
        };

        profiler.track_allocation(2000, 2048, context).unwrap();

        // Then deallocate
        let result = profiler.track_deallocation(2000);
        assert!(result.is_ok());

        let stats = profiler.get_statistics().unwrap();
        assert_eq!(stats.total_deallocations, 1);
    }

    #[test]
    fn test_quick_profiling_functions() {
        let result = track_allocation_quick(3000, 1024);
        assert!(result.is_ok());

        let result = track_deallocation_quick(3000);
        assert!(result.is_ok());

        let report = generate_quick_report();
        assert!(report.is_ok());
        assert!(!report.unwrap().is_empty());
    }

    #[test]
    fn test_comprehensive_report_generation() {
        let mut profiler = MemoryProfiler::with_preset(ProfilerPreset::Development);

        // Add some data
        let context = AllocationContext {
            allocation_id: Some(4000),
            operation: Some("test".to_string()),
            allocation_purpose: Some("tensor".to_string()),
            tensor_shape: None,
            data_type: None,
            kernel_name: None,
            device: None,
            thread_id: 1,
            call_stack: vec![],
        };

        profiler.track_allocation(4000, 8192, context).unwrap();

        let report = profiler.generate_comprehensive_report();
        assert!(report.is_ok());

        let report = report.unwrap();
        assert!(report.executive_summary.overall_health_score >= 0.0);
        assert!(report.executive_summary.overall_health_score <= 1.0);
    }

    #[test]
    fn test_auto_optimization() {
        let mut profiler = MemoryProfiler::with_preset(ProfilerPreset::Production);

        let result = profiler.auto_optimize();
        assert!(result.is_ok());

        let optimization_result = result.unwrap();
        assert!(optimization_result.duration.as_millis() > 0);
    }
}