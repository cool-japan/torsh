// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Capacity Planning for Autograd Workloads
//!
//! This module provides tools for capacity planning and resource forecasting
//! for autograd workloads, helping optimize infrastructure and predict requirements.
//!
//! # Features
//!
//! - **Workload Profiling**: Analyze historical workload patterns
//! - **Resource Forecasting**: Predict future resource requirements
//! - **Scaling Recommendations**: Suggest optimal resource allocation
//! - **Cost Optimization**: Minimize cost while meeting performance targets
//! - **Trend Analysis**: Identify growth trends and seasonality
//! - **Capacity Alerts**: Alert when approaching capacity limits

use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Capacity planning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityConfig {
    /// Forecast horizon (days)
    pub forecast_horizon_days: u32,

    /// Historical window (days)
    pub historical_window_days: u32,

    /// Safety margin (percentage above forecast)
    pub safety_margin_percent: f64,

    /// Target utilization (0.0-1.0)
    pub target_utilization: f64,

    /// Cost per GB memory per hour
    pub cost_per_gb_hour: f64,

    /// Cost per CPU hour
    pub cost_per_cpu_hour: f64,

    /// Cost per GPU hour
    pub cost_per_gpu_hour: Option<f64>,
}

impl Default for CapacityConfig {
    fn default() -> Self {
        Self {
            forecast_horizon_days: 30,
            historical_window_days: 90,
            safety_margin_percent: 20.0,
            target_utilization: 0.75, // 75% utilization target
            cost_per_gb_hour: 0.001,
            cost_per_cpu_hour: 0.05,
            cost_per_gpu_hour: Some(0.5),
        }
    }
}

/// Capacity planner
pub struct CapacityPlanner {
    config: CapacityConfig,
    workload_history: Arc<RwLock<WorkloadHistory>>,
    resource_usage: Arc<RwLock<ResourceUsageHistory>>,
    forecasts: Arc<RwLock<Vec<CapacityForecast>>>,
}

/// Workload history tracking
#[derive(Debug, Default)]
struct WorkloadHistory {
    /// Workload samples
    samples: VecDeque<WorkloadSample>,

    /// Aggregated daily statistics
    #[allow(dead_code)]
    daily_stats: HashMap<String, DailyWorkloadStats>,
}

/// Single workload sample
#[derive(Debug, Clone)]
struct WorkloadSample {
    timestamp: DateTime<Utc>,
    operations_per_second: f64,
    #[allow(dead_code)]
    avg_operation_duration_ms: f64,
    #[allow(dead_code)]
    peak_operations_per_second: f64,
    #[allow(dead_code)]
    concurrent_operations: u64,
}

/// Daily workload statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyWorkloadStats {
    /// Date
    pub date: String,

    /// Total operations
    pub total_operations: u64,

    /// Average operations per second
    pub avg_ops_per_second: f64,

    /// Peak operations per second
    pub peak_ops_per_second: f64,

    /// Average operation duration
    pub avg_duration_ms: f64,

    /// P95 operation duration
    pub p95_duration_ms: f64,

    /// P99 operation duration
    pub p99_duration_ms: f64,
}

/// Resource usage history
#[derive(Debug, Default)]
struct ResourceUsageHistory {
    /// Memory usage samples
    memory_samples: VecDeque<ResourceSample>,

    /// CPU usage samples
    cpu_samples: VecDeque<ResourceSample>,

    /// GPU usage samples (if applicable)
    gpu_samples: VecDeque<ResourceSample>,
}

/// Resource usage sample
#[derive(Debug, Clone)]
struct ResourceSample {
    timestamp: DateTime<Utc>,
    usage: f64,
    capacity: f64,
}

/// Capacity forecast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityForecast {
    /// Forecast date
    pub date: DateTime<Utc>,

    /// Forecasted memory requirement (GB)
    pub memory_gb: f64,

    /// Forecasted CPU requirement (cores)
    pub cpu_cores: f64,

    /// Forecasted GPU requirement (if applicable)
    pub gpu_count: Option<f64>,

    /// Forecasted operations per second
    pub ops_per_second: f64,

    /// Confidence interval (percentage)
    pub confidence_percent: f64,

    /// Recommended capacity
    pub recommended_capacity: RecommendedCapacity,

    /// Estimated cost per day
    pub estimated_cost_per_day: f64,
}

/// Recommended capacity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedCapacity {
    /// Memory (GB)
    pub memory_gb: f64,

    /// CPU cores
    pub cpu_cores: f64,

    /// GPU count
    pub gpu_count: Option<f64>,

    /// Rationale
    pub rationale: String,
}

/// Scaling recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingRecommendation {
    /// Recommendation type
    pub recommendation_type: ScalingAction,

    /// Urgency level
    pub urgency: UrgencyLevel,

    /// Description
    pub description: String,

    /// Expected benefit
    pub expected_benefit: String,

    /// Estimated cost impact
    pub cost_impact_per_day: f64,

    /// Recommended timeline
    pub recommended_timeline: String,
}

/// Scaling action type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingAction {
    /// Scale up resources
    ScaleUp,

    /// Scale down resources
    ScaleDown,

    /// Maintain current capacity
    Maintain,

    /// Optimize configuration
    Optimize,
}

/// Urgency level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum UrgencyLevel {
    /// Low urgency
    Low,

    /// Medium urgency
    Medium,

    /// High urgency
    High,

    /// Critical urgency
    Critical,
}

impl CapacityPlanner {
    /// Create a new capacity planner
    pub fn new(config: CapacityConfig) -> Self {
        Self {
            config,
            workload_history: Arc::new(RwLock::new(WorkloadHistory::default())),
            resource_usage: Arc::new(RwLock::new(ResourceUsageHistory::default())),
            forecasts: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Record workload sample
    pub fn record_workload(
        &self,
        ops_per_second: f64,
        avg_duration_ms: f64,
        peak_ops: f64,
        concurrent_ops: u64,
    ) {
        let sample = WorkloadSample {
            timestamp: Utc::now(),
            operations_per_second: ops_per_second,
            avg_operation_duration_ms: avg_duration_ms,
            peak_operations_per_second: peak_ops,
            concurrent_operations: concurrent_ops,
        };

        let mut history = self.workload_history.write();
        history.samples.push_back(sample);

        // Cleanup old samples
        let cutoff_time = Utc::now() - Duration::days(self.config.historical_window_days as i64);

        while let Some(sample) = history.samples.front() {
            if sample.timestamp < cutoff_time {
                history.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Record resource usage
    pub fn record_resource_usage(
        &self,
        memory_gb: f64,
        memory_capacity_gb: f64,
        cpu_cores: f64,
        cpu_capacity: f64,
        gpu_usage: Option<(f64, f64)>,
    ) {
        let timestamp = Utc::now();
        let mut usage = self.resource_usage.write();

        usage.memory_samples.push_back(ResourceSample {
            timestamp,
            usage: memory_gb,
            capacity: memory_capacity_gb,
        });

        usage.cpu_samples.push_back(ResourceSample {
            timestamp,
            usage: cpu_cores,
            capacity: cpu_capacity,
        });

        if let Some((gpu_use, gpu_cap)) = gpu_usage {
            usage.gpu_samples.push_back(ResourceSample {
                timestamp,
                usage: gpu_use,
                capacity: gpu_cap,
            });
        }

        // Cleanup old samples
        let cutoff_time = Utc::now() - Duration::days(self.config.historical_window_days as i64);

        let cleanup = |samples: &mut VecDeque<ResourceSample>| {
            while let Some(sample) = samples.front() {
                if sample.timestamp < cutoff_time {
                    samples.pop_front();
                } else {
                    break;
                }
            }
        };

        cleanup(&mut usage.memory_samples);
        cleanup(&mut usage.cpu_samples);
        cleanup(&mut usage.gpu_samples);
    }

    /// Generate capacity forecast
    pub fn generate_forecast(&self) -> Vec<CapacityForecast> {
        let workload = self.workload_history.read();
        let resources = self.resource_usage.read();

        let mut forecasts = Vec::new();

        // Simple linear regression forecast (in production, use more sophisticated models)
        let base_date = Utc::now();

        for day in 0..self.config.forecast_horizon_days {
            let forecast_date = base_date + Duration::days(day as i64);

            // Calculate projected workload
            let growth_rate = self.estimate_growth_rate(&workload);
            let base_ops = self.calculate_average_ops(&workload);
            let forecasted_ops = base_ops * (1.0 + growth_rate).powi(day as i32);

            // Calculate resource requirements
            let memory_gb = self.forecast_memory_requirement(&resources, growth_rate, day);
            let cpu_cores = self.forecast_cpu_requirement(&resources, growth_rate, day);
            let gpu_count = self.forecast_gpu_requirement(&resources, growth_rate, day);

            // Apply safety margin
            let safety_multiplier = 1.0 + (self.config.safety_margin_percent / 100.0);
            let recommended_memory = memory_gb * safety_multiplier;
            let recommended_cpu = cpu_cores * safety_multiplier;
            let recommended_gpu = gpu_count.map(|g| g * safety_multiplier);

            // Estimate cost
            let cost =
                self.estimate_daily_cost(recommended_memory, recommended_cpu, recommended_gpu);

            let forecast = CapacityForecast {
                date: forecast_date,
                memory_gb,
                cpu_cores,
                gpu_count,
                ops_per_second: forecasted_ops,
                confidence_percent: 80.0, // Simplified
                recommended_capacity: RecommendedCapacity {
                    memory_gb: recommended_memory,
                    cpu_cores: recommended_cpu,
                    gpu_count: recommended_gpu,
                    rationale: format!(
                        "Based on {}% growth rate with {}% safety margin",
                        (growth_rate * 100.0) as i32,
                        self.config.safety_margin_percent as i32
                    ),
                },
                estimated_cost_per_day: cost,
            };

            forecasts.push(forecast);
        }

        *self.forecasts.write() = forecasts.clone();
        forecasts
    }

    /// Get scaling recommendations
    pub fn get_scaling_recommendations(&self) -> Vec<ScalingRecommendation> {
        let resources = self.resource_usage.read();
        let mut recommendations = Vec::new();

        // Check current utilization
        let memory_utilization = self.calculate_current_utilization(&resources.memory_samples);
        let cpu_utilization = self.calculate_current_utilization(&resources.cpu_samples);

        // Memory scaling recommendation
        if memory_utilization > 0.9 {
            recommendations.push(ScalingRecommendation {
                recommendation_type: ScalingAction::ScaleUp,
                urgency: UrgencyLevel::Critical,
                description: format!(
                    "Memory utilization at {:.1}% - immediate action required",
                    memory_utilization * 100.0
                ),
                expected_benefit: "Prevent OOM errors and system instability".to_string(),
                cost_impact_per_day: self.config.cost_per_gb_hour * 24.0 * 10.0, // +10GB estimate
                recommended_timeline: "Immediate".to_string(),
            });
        } else if memory_utilization > self.config.target_utilization {
            recommendations.push(ScalingRecommendation {
                recommendation_type: ScalingAction::ScaleUp,
                urgency: UrgencyLevel::Medium,
                description: format!(
                    "Memory utilization at {:.1}% exceeds target of {:.1}%",
                    memory_utilization * 100.0,
                    self.config.target_utilization * 100.0
                ),
                expected_benefit: "Maintain performance and headroom for growth".to_string(),
                cost_impact_per_day: self.config.cost_per_gb_hour * 24.0 * 5.0, // +5GB estimate
                recommended_timeline: "Within 1 week".to_string(),
            });
        } else if memory_utilization < 0.3 {
            recommendations.push(ScalingRecommendation {
                recommendation_type: ScalingAction::ScaleDown,
                urgency: UrgencyLevel::Low,
                description: format!(
                    "Memory utilization at {:.1}% - opportunity for cost savings",
                    memory_utilization * 100.0
                ),
                expected_benefit: "Reduce costs while maintaining adequate capacity".to_string(),
                cost_impact_per_day: -(self.config.cost_per_gb_hour * 24.0 * 5.0), // -5GB estimate
                recommended_timeline: "Within 1 month".to_string(),
            });
        }

        // CPU scaling recommendation
        if cpu_utilization > 0.9 {
            recommendations.push(ScalingRecommendation {
                recommendation_type: ScalingAction::ScaleUp,
                urgency: UrgencyLevel::High,
                description: format!(
                    "CPU utilization at {:.1}% - performance degradation likely",
                    cpu_utilization * 100.0
                ),
                expected_benefit: "Improve training speed and reduce latency".to_string(),
                cost_impact_per_day: self.config.cost_per_cpu_hour * 24.0 * 4.0, // +4 cores estimate
                recommended_timeline: "Within 24 hours".to_string(),
            });
        }

        recommendations
    }

    /// Get capacity forecast for specific date
    pub fn get_forecast_for_date(&self, date: DateTime<Utc>) -> Option<CapacityForecast> {
        let forecasts = self.forecasts.read();

        forecasts
            .iter()
            .min_by_key(|f| {
                let diff = (f.date - date).num_seconds().abs();
                diff
            })
            .cloned()
    }

    /// Get capacity trend analysis
    pub fn get_trend_analysis(&self) -> TrendAnalysis {
        let resources = self.resource_usage.read();

        let memory_trend = self.calculate_trend(&resources.memory_samples);
        let cpu_trend = self.calculate_trend(&resources.cpu_samples);
        let gpu_trend = if !resources.gpu_samples.is_empty() {
            Some(self.calculate_trend(&resources.gpu_samples))
        } else {
            None
        };

        TrendAnalysis {
            memory_trend_percent: memory_trend,
            cpu_trend_percent: cpu_trend,
            gpu_trend_percent: gpu_trend,
            analysis_period_days: self.config.historical_window_days,
            trend_direction: if memory_trend > 5.0 {
                TrendDirection::Increasing
            } else if memory_trend < -5.0 {
                TrendDirection::Decreasing
            } else {
                TrendDirection::Stable
            },
        }
    }

    // Private helper methods

    fn estimate_growth_rate(&self, workload: &WorkloadHistory) -> f64 {
        if workload.samples.len() < 2 {
            return 0.0; // No growth if insufficient data
        }

        // Simple growth rate: compare first half vs second half
        let mid = workload.samples.len() / 2;
        let first_half_avg: f64 = workload
            .samples
            .iter()
            .take(mid)
            .map(|s| s.operations_per_second)
            .sum::<f64>()
            / mid as f64;

        let second_half_avg: f64 = workload
            .samples
            .iter()
            .skip(mid)
            .map(|s| s.operations_per_second)
            .sum::<f64>()
            / (workload.samples.len() - mid) as f64;

        if first_half_avg == 0.0 {
            return 1.0; // 100% growth from zero
        }

        (second_half_avg - first_half_avg) / first_half_avg
    }

    fn calculate_average_ops(&self, workload: &WorkloadHistory) -> f64 {
        if workload.samples.is_empty() {
            return 0.0;
        }

        workload
            .samples
            .iter()
            .map(|s| s.operations_per_second)
            .sum::<f64>()
            / workload.samples.len() as f64
    }

    fn forecast_memory_requirement(
        &self,
        resources: &ResourceUsageHistory,
        growth_rate: f64,
        days: u32,
    ) -> f64 {
        if resources.memory_samples.is_empty() {
            return 1.0; // Default 1GB
        }

        let avg_usage: f64 = resources
            .memory_samples
            .iter()
            .map(|s| s.usage)
            .sum::<f64>()
            / resources.memory_samples.len() as f64;

        avg_usage * (1.0 + growth_rate).powi(days as i32)
    }

    fn forecast_cpu_requirement(
        &self,
        resources: &ResourceUsageHistory,
        growth_rate: f64,
        days: u32,
    ) -> f64 {
        if resources.cpu_samples.is_empty() {
            return 1.0; // Default 1 core
        }

        let avg_usage: f64 = resources.cpu_samples.iter().map(|s| s.usage).sum::<f64>()
            / resources.cpu_samples.len() as f64;

        avg_usage * (1.0 + growth_rate).powi(days as i32)
    }

    fn forecast_gpu_requirement(
        &self,
        resources: &ResourceUsageHistory,
        growth_rate: f64,
        days: u32,
    ) -> Option<f64> {
        if resources.gpu_samples.is_empty() {
            return None;
        }

        let avg_usage: f64 = resources.gpu_samples.iter().map(|s| s.usage).sum::<f64>()
            / resources.gpu_samples.len() as f64;

        Some(avg_usage * (1.0 + growth_rate).powi(days as i32))
    }

    fn estimate_daily_cost(&self, memory_gb: f64, cpu_cores: f64, gpu_count: Option<f64>) -> f64 {
        let mut cost = 0.0;

        cost += memory_gb * self.config.cost_per_gb_hour * 24.0;
        cost += cpu_cores * self.config.cost_per_cpu_hour * 24.0;

        if let (Some(gpus), Some(gpu_cost)) = (gpu_count, self.config.cost_per_gpu_hour) {
            cost += gpus * gpu_cost * 24.0;
        }

        cost
    }

    fn calculate_current_utilization(&self, samples: &VecDeque<ResourceSample>) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        // Use recent samples (last hour)
        let one_hour_ago = Utc::now() - Duration::hours(1);
        let recent_samples: Vec<_> = samples
            .iter()
            .filter(|s| s.timestamp > one_hour_ago)
            .collect();

        if recent_samples.is_empty() {
            return 0.0;
        }

        let avg_usage: f64 =
            recent_samples.iter().map(|s| s.usage).sum::<f64>() / recent_samples.len() as f64;
        let avg_capacity: f64 =
            recent_samples.iter().map(|s| s.capacity).sum::<f64>() / recent_samples.len() as f64;

        if avg_capacity == 0.0 {
            return 0.0;
        }

        avg_usage / avg_capacity
    }

    fn calculate_trend(&self, samples: &VecDeque<ResourceSample>) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mid = samples.len() / 2;
        let first_half_avg: f64 =
            samples.iter().take(mid).map(|s| s.usage).sum::<f64>() / mid as f64;

        let second_half_avg: f64 =
            samples.iter().skip(mid).map(|s| s.usage).sum::<f64>() / (samples.len() - mid) as f64;

        if first_half_avg == 0.0 {
            return 100.0;
        }

        ((second_half_avg - first_half_avg) / first_half_avg) * 100.0
    }
}

/// Trend analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Memory trend (percentage change)
    pub memory_trend_percent: f64,

    /// CPU trend (percentage change)
    pub cpu_trend_percent: f64,

    /// GPU trend (percentage change)
    pub gpu_trend_percent: Option<f64>,

    /// Analysis period (days)
    pub analysis_period_days: u32,

    /// Overall trend direction
    pub trend_direction: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,

    /// Decreasing trend
    Decreasing,

    /// Stable trend
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_recording() {
        let planner = CapacityPlanner::new(CapacityConfig::default());

        planner.record_workload(100.0, 50.0, 150.0, 10);

        let history = planner.workload_history.read();
        assert_eq!(history.samples.len(), 1);
    }

    #[test]
    fn test_resource_recording() {
        let planner = CapacityPlanner::new(CapacityConfig::default());

        planner.record_resource_usage(8.0, 16.0, 4.0, 8.0, Some((1.0, 2.0)));

        let usage = planner.resource_usage.read();
        assert_eq!(usage.memory_samples.len(), 1);
        assert_eq!(usage.cpu_samples.len(), 1);
        assert_eq!(usage.gpu_samples.len(), 1);
    }

    #[test]
    fn test_forecast_generation() {
        let planner = CapacityPlanner::new(CapacityConfig {
            forecast_horizon_days: 7,
            ..Default::default()
        });

        // Record some historical data
        for _ in 0..10 {
            planner.record_workload(100.0, 50.0, 150.0, 10);
            planner.record_resource_usage(8.0, 16.0, 4.0, 8.0, None);
        }

        let forecasts = planner.generate_forecast();
        assert_eq!(forecasts.len(), 7);
    }

    #[test]
    fn test_scaling_recommendations() {
        let planner = CapacityPlanner::new(CapacityConfig::default());

        // Record high utilization
        for _ in 0..10 {
            planner.record_resource_usage(15.0, 16.0, 7.5, 8.0, None); // 93% memory, 93% CPU
        }

        let recommendations = planner.get_scaling_recommendations();
        assert!(!recommendations.is_empty());

        // Should recommend scaling up
        assert!(recommendations
            .iter()
            .any(|r| r.recommendation_type == ScalingAction::ScaleUp));
    }

    #[test]
    fn test_trend_analysis() {
        let planner = CapacityPlanner::new(CapacityConfig::default());

        // Record increasing trend
        for i in 0..20 {
            let usage = 5.0 + (i as f64 * 0.5);
            planner.record_resource_usage(usage, 16.0, 2.0, 8.0, None);
        }

        let trend = planner.get_trend_analysis();
        assert!(trend.memory_trend_percent > 0.0);
        assert_eq!(trend.trend_direction, TrendDirection::Increasing);
    }
}
