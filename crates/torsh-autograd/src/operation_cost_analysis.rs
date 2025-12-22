// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operation Cost Analysis
//!
//! This module provides comprehensive cost analysis for autograd operations,
//! enabling optimization of computational and resource costs.
//!
//! # Features
//!
//! - **Computational Cost**: Analyze time complexity and FLOP requirements
//! - **Memory Cost**: Track memory allocation and access patterns
//! - **Energy Cost**: Estimate power consumption (if supported)
//! - **Financial Cost**: Calculate cloud computing costs
//! - **Cost Attribution**: Attribute costs to specific operations/layers
//! - **Optimization Suggestions**: Identify cost-saving opportunities

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};

/// Cost analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysisConfig {
    /// Enable cost tracking
    pub enabled: bool,

    /// Track computational cost (FLOPs)
    pub track_computational_cost: bool,

    /// Track memory cost
    pub track_memory_cost: bool,

    /// Track energy cost
    pub track_energy_cost: bool,

    /// Cost per GFLOP (for cloud pricing)
    pub cost_per_gflop: f64,

    /// Cost per GB-second of memory
    pub cost_per_gb_second: f64,

    /// Cost per kWh of energy
    pub cost_per_kwh: f64,

    /// Maximum cost history entries
    pub max_history_entries: usize,
}

impl Default for CostAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            track_computational_cost: true,
            track_memory_cost: true,
            track_energy_cost: false,   // Requires hardware support
            cost_per_gflop: 0.0001,     // $0.0001 per GFLOP
            cost_per_gb_second: 0.0001, // $0.0001 per GB-second
            cost_per_kwh: 0.12,         // $0.12 per kWh
            max_history_entries: 10000,
        }
    }
}

/// Operation cost analyzer
pub struct OperationCostAnalyzer {
    config: CostAnalysisConfig,
    operation_costs: Arc<RwLock<HashMap<String, OperationCostStats>>>,
    cost_history: Arc<RwLock<VecDeque<CostEntry>>>,
    aggregated_costs: Arc<RwLock<AggregatedCosts>>,
}

/// Cost statistics for a specific operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationCostStats {
    /// Operation name
    pub operation_name: String,

    /// Number of times executed
    pub execution_count: u64,

    /// Total computational cost (FLOPs)
    pub total_flops: u64,

    /// Average computational cost per execution
    pub avg_flops: f64,

    /// Total memory allocated (bytes)
    pub total_memory_bytes: u64,

    /// Average memory per execution
    pub avg_memory_bytes: f64,

    /// Total execution time (milliseconds)
    pub total_time_ms: f64,

    /// Average execution time per operation
    pub avg_time_ms: f64,

    /// Total financial cost (dollars)
    pub total_cost_usd: f64,

    /// Average cost per execution
    pub avg_cost_usd: f64,

    /// Cost breakdown
    pub cost_breakdown: CostBreakdown,
}

/// Detailed cost breakdown
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CostBreakdown {
    /// Computational cost (compute resources)
    pub computational_cost_usd: f64,

    /// Memory cost (memory allocation)
    pub memory_cost_usd: f64,

    /// Energy cost (power consumption)
    pub energy_cost_usd: f64,

    /// Network cost (data transfer)
    pub network_cost_usd: f64,

    /// Storage cost (disk I/O)
    pub storage_cost_usd: f64,
}

/// Individual cost entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEntry {
    /// Entry ID
    pub id: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Operation name
    pub operation_name: String,

    /// Operation type
    pub operation_type: OperationType,

    /// Computational cost (FLOPs)
    pub flops: u64,

    /// Memory cost (bytes)
    pub memory_bytes: u64,

    /// Execution time (milliseconds)
    pub duration_ms: f64,

    /// Energy consumed (watt-hours)
    pub energy_wh: Option<f64>,

    /// Total cost (USD)
    pub total_cost_usd: f64,

    /// Cost breakdown
    pub breakdown: CostBreakdown,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Operation type for cost analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    /// Matrix multiplication
    MatMul,

    /// Convolution
    Convolution,

    /// Element-wise operations
    ElementWise,

    /// Reduction operations
    Reduction,

    /// Activation functions
    Activation,

    /// Normalization
    Normalization,

    /// Pooling
    Pooling,

    /// Gradient computation
    GradientComputation,

    /// Custom operation
    Custom,
}

/// Aggregated costs across all operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregatedCosts {
    /// Total computational cost (FLOPs)
    pub total_flops: u64,

    /// Total memory allocated (bytes)
    pub total_memory_bytes: u64,

    /// Total execution time (milliseconds)
    pub total_time_ms: f64,

    /// Total financial cost (USD)
    pub total_cost_usd: f64,

    /// Cost breakdown
    pub breakdown: CostBreakdown,

    /// Costs by operation type
    pub costs_by_type: HashMap<String, f64>,

    /// Most expensive operations (top N)
    pub top_expensive_operations: Vec<(String, f64)>,

    /// Period start
    pub period_start: DateTime<Utc>,

    /// Period end
    pub period_end: DateTime<Utc>,
}

impl OperationCostAnalyzer {
    /// Create a new operation cost analyzer
    pub fn new(config: CostAnalysisConfig) -> Self {
        Self {
            config,
            operation_costs: Arc::new(RwLock::new(HashMap::new())),
            cost_history: Arc::new(RwLock::new(VecDeque::new())),
            aggregated_costs: Arc::new(RwLock::new(AggregatedCosts::default())),
        }
    }

    /// Record operation cost
    pub fn record_operation_cost(
        &self,
        operation_name: String,
        operation_type: OperationType,
        flops: u64,
        memory_bytes: u64,
        duration_ms: f64,
        energy_wh: Option<f64>,
    ) {
        if !self.config.enabled {
            return;
        }

        // Calculate costs
        let computational_cost = if self.config.track_computational_cost {
            (flops as f64 / 1_000_000_000.0) * self.config.cost_per_gflop
        } else {
            0.0
        };

        let memory_cost = if self.config.track_memory_cost {
            let gb_seconds = (memory_bytes as f64 / 1_073_741_824.0) * (duration_ms / 1000.0);
            gb_seconds * self.config.cost_per_gb_second
        } else {
            0.0
        };

        let energy_cost = if self.config.track_energy_cost {
            energy_wh
                .map(|wh| (wh / 1000.0) * self.config.cost_per_kwh)
                .unwrap_or(0.0)
        } else {
            0.0
        };

        let total_cost = computational_cost + memory_cost + energy_cost;

        let breakdown = CostBreakdown {
            computational_cost_usd: computational_cost,
            memory_cost_usd: memory_cost,
            energy_cost_usd: energy_cost,
            network_cost_usd: 0.0,
            storage_cost_usd: 0.0,
        };

        // Create cost entry
        let entry = CostEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            operation_name: operation_name.clone(),
            operation_type,
            flops,
            memory_bytes,
            duration_ms,
            energy_wh,
            total_cost_usd: total_cost,
            breakdown,
            metadata: HashMap::new(),
        };

        // Update operation-specific stats
        self.update_operation_stats(&operation_name, &entry);

        // Add to history
        let mut history = self.cost_history.write();
        history.push_back(entry.clone());

        // Enforce max history size
        while history.len() > self.config.max_history_entries {
            history.pop_front();
        }

        drop(history);

        // Update aggregated costs
        self.update_aggregated_costs(&entry);
    }

    /// Get cost statistics for a specific operation
    pub fn get_operation_stats(&self, operation_name: &str) -> Option<OperationCostStats> {
        let costs = self.operation_costs.read();
        costs.get(operation_name).cloned()
    }

    /// Get all operation statistics
    pub fn get_all_operation_stats(&self) -> Vec<OperationCostStats> {
        let costs = self.operation_costs.read();
        costs.values().cloned().collect()
    }

    /// Get aggregated costs
    pub fn get_aggregated_costs(&self) -> AggregatedCosts {
        self.aggregated_costs.read().clone()
    }

    /// Get cost history
    pub fn get_cost_history(&self, limit: Option<usize>) -> Vec<CostEntry> {
        let history = self.cost_history.read();
        let limit = limit.unwrap_or(100);

        history.iter().rev().take(limit).cloned().collect()
    }

    /// Get most expensive operations
    pub fn get_most_expensive_operations(&self, limit: usize) -> Vec<(String, f64)> {
        let costs = self.operation_costs.read();

        let mut operations: Vec<_> = costs
            .iter()
            .map(|(name, stats)| (name.clone(), stats.total_cost_usd))
            .collect();

        operations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        operations.into_iter().take(limit).collect()
    }

    /// Get cost optimization suggestions
    pub fn get_optimization_suggestions(&self) -> Vec<CostOptimizationSuggestion> {
        let costs = self.operation_costs.read();
        let mut suggestions = Vec::new();

        for (op_name, stats) in costs.iter() {
            // High computational cost operations
            if stats.avg_flops > 1_000_000_000.0 {
                suggestions.push(CostOptimizationSuggestion {
                    operation_name: op_name.clone(),
                    suggestion_type: OptimizationType::ReduceComputation,
                    description: format!(
                        "Operation '{}' has high computational cost ({:.2} GFLOPS avg). Consider operation fusion or approximation.",
                        op_name,
                        stats.avg_flops / 1_000_000_000.0
                    ),
                    potential_savings_usd: stats.total_cost_usd * 0.3, // 30% savings estimate
                    priority: if stats.total_cost_usd > 1.0 {
                        OptimizationPriority::High
                    } else {
                        OptimizationPriority::Medium
                    },
                });
            }

            // High memory cost operations
            if stats.avg_memory_bytes > 1_073_741_824.0 {
                // > 1GB
                suggestions.push(CostOptimizationSuggestion {
                    operation_name: op_name.clone(),
                    suggestion_type: OptimizationType::ReduceMemory,
                    description: format!(
                        "Operation '{}' has high memory usage ({:.2} GB avg). Consider gradient checkpointing or memory-efficient implementations.",
                        op_name,
                        stats.avg_memory_bytes as f64 / 1_073_741_824.0
                    ),
                    potential_savings_usd: stats.cost_breakdown.memory_cost_usd * 0.4, // 40% savings estimate
                    priority: OptimizationPriority::Medium,
                });
            }
        }

        suggestions
    }

    /// Generate cost report
    pub fn generate_cost_report(&self) -> String {
        let aggregated = self.aggregated_costs.read();
        let expensive_ops = self.get_most_expensive_operations(10);

        let mut report = String::new();

        report.push_str("=== Autograd Operation Cost Analysis Report ===\n\n");

        // Overall costs
        report.push_str("Overall Costs:\n");
        report.push_str(&format!(
            "  Total Computational: {:.2} GFLOPs\n",
            aggregated.total_flops as f64 / 1_000_000_000.0
        ));
        report.push_str(&format!(
            "  Total Memory: {:.2} GB\n",
            aggregated.total_memory_bytes as f64 / 1_073_741_824.0
        ));
        report.push_str(&format!(
            "  Total Time: {:.2} seconds\n",
            aggregated.total_time_ms / 1000.0
        ));
        report.push_str(&format!(
            "  Total Cost: ${:.4}\n\n",
            aggregated.total_cost_usd
        ));

        // Cost breakdown
        report.push_str("Cost Breakdown:\n");
        report.push_str(&format!(
            "  Computational: ${:.4}\n",
            aggregated.breakdown.computational_cost_usd
        ));
        report.push_str(&format!(
            "  Memory: ${:.4}\n",
            aggregated.breakdown.memory_cost_usd
        ));
        report.push_str(&format!(
            "  Energy: ${:.4}\n\n",
            aggregated.breakdown.energy_cost_usd
        ));

        // Most expensive operations
        report.push_str("Most Expensive Operations:\n");
        for (i, (op_name, cost)) in expensive_ops.iter().enumerate() {
            report.push_str(&format!("  {}. {} - ${:.4}\n", i + 1, op_name, cost));
        }

        report
    }

    // Private helper methods

    fn update_operation_stats(&self, operation_name: &str, entry: &CostEntry) {
        let mut costs = self.operation_costs.write();

        let stats = costs
            .entry(operation_name.to_string())
            .or_insert(OperationCostStats {
                operation_name: operation_name.to_string(),
                execution_count: 0,
                total_flops: 0,
                avg_flops: 0.0,
                total_memory_bytes: 0,
                avg_memory_bytes: 0.0,
                total_time_ms: 0.0,
                avg_time_ms: 0.0,
                total_cost_usd: 0.0,
                avg_cost_usd: 0.0,
                cost_breakdown: CostBreakdown::default(),
            });

        stats.execution_count += 1;
        stats.total_flops += entry.flops;
        stats.total_memory_bytes += entry.memory_bytes;
        stats.total_time_ms += entry.duration_ms;
        stats.total_cost_usd += entry.total_cost_usd;

        // Update averages
        stats.avg_flops = stats.total_flops as f64 / stats.execution_count as f64;
        stats.avg_memory_bytes = stats.total_memory_bytes as f64 / stats.execution_count as f64;
        stats.avg_time_ms = stats.total_time_ms / stats.execution_count as f64;
        stats.avg_cost_usd = stats.total_cost_usd / stats.execution_count as f64;

        // Update cost breakdown
        stats.cost_breakdown.computational_cost_usd += entry.breakdown.computational_cost_usd;
        stats.cost_breakdown.memory_cost_usd += entry.breakdown.memory_cost_usd;
        stats.cost_breakdown.energy_cost_usd += entry.breakdown.energy_cost_usd;
    }

    fn update_aggregated_costs(&self, entry: &CostEntry) {
        let mut aggregated = self.aggregated_costs.write();

        aggregated.total_flops += entry.flops;
        aggregated.total_memory_bytes += entry.memory_bytes;
        aggregated.total_time_ms += entry.duration_ms;
        aggregated.total_cost_usd += entry.total_cost_usd;

        aggregated.breakdown.computational_cost_usd += entry.breakdown.computational_cost_usd;
        aggregated.breakdown.memory_cost_usd += entry.breakdown.memory_cost_usd;
        aggregated.breakdown.energy_cost_usd += entry.breakdown.energy_cost_usd;

        // Update costs by type
        let type_key = format!("{:?}", entry.operation_type);
        *aggregated.costs_by_type.entry(type_key).or_insert(0.0) += entry.total_cost_usd;

        // Update period
        if aggregated.period_start == DateTime::<Utc>::MIN_UTC {
            aggregated.period_start = entry.timestamp;
        }
        aggregated.period_end = entry.timestamp;
    }
}

/// Cost optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationSuggestion {
    /// Operation name
    pub operation_name: String,

    /// Type of optimization
    pub suggestion_type: OptimizationType,

    /// Detailed description
    pub description: String,

    /// Potential cost savings (USD)
    pub potential_savings_usd: f64,

    /// Priority level
    pub priority: OptimizationPriority,
}

/// Optimization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Reduce computational cost
    ReduceComputation,

    /// Reduce memory usage
    ReduceMemory,

    /// Reduce energy consumption
    ReduceEnergy,

    /// Improve caching
    ImproveCaching,

    /// Operation fusion
    OperationFusion,

    /// Use mixed precision
    UseMixedPrecision,
}

/// Optimization priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OptimizationPriority {
    /// Low priority
    Low,

    /// Medium priority
    Medium,

    /// High priority
    High,

    /// Critical priority
    Critical,
}

/// Global operation cost analyzer
static GLOBAL_COST_ANALYZER: OnceLock<Arc<OperationCostAnalyzer>> = OnceLock::new();

/// Get global cost analyzer
pub fn get_global_cost_analyzer() -> Arc<OperationCostAnalyzer> {
    GLOBAL_COST_ANALYZER
        .get_or_init(|| Arc::new(OperationCostAnalyzer::new(CostAnalysisConfig::default())))
        .clone()
}

/// Initialize global cost analyzer
pub fn init_global_cost_analyzer(config: CostAnalysisConfig) {
    let _ = GLOBAL_COST_ANALYZER.set(Arc::new(OperationCostAnalyzer::new(config)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_recording() {
        let analyzer = OperationCostAnalyzer::new(CostAnalysisConfig::default());

        analyzer.record_operation_cost(
            "matmul".to_string(),
            OperationType::MatMul,
            1_000_000_000, // 1 GFLOP
            1024 * 1024,   // 1 MB
            10.0,          // 10 ms
            None,
        );

        let stats = analyzer.get_operation_stats("matmul").unwrap();
        assert_eq!(stats.execution_count, 1);
        assert_eq!(stats.total_flops, 1_000_000_000);
    }

    #[test]
    fn test_aggregated_costs() {
        let analyzer = OperationCostAnalyzer::new(CostAnalysisConfig::default());

        analyzer.record_operation_cost(
            "op1".to_string(),
            OperationType::MatMul,
            1_000_000_000,
            1024 * 1024,
            10.0,
            None,
        );

        analyzer.record_operation_cost(
            "op2".to_string(),
            OperationType::Convolution,
            2_000_000_000,
            2 * 1024 * 1024,
            20.0,
            None,
        );

        let aggregated = analyzer.get_aggregated_costs();
        assert_eq!(aggregated.total_flops, 3_000_000_000);
        assert_eq!(aggregated.total_memory_bytes, 3 * 1024 * 1024);
    }

    #[test]
    fn test_most_expensive_operations() {
        let analyzer = OperationCostAnalyzer::new(CostAnalysisConfig::default());

        analyzer.record_operation_cost(
            "cheap_op".to_string(),
            OperationType::ElementWise,
            100_000,
            1024,
            0.1,
            None,
        );

        analyzer.record_operation_cost(
            "expensive_op".to_string(),
            OperationType::MatMul,
            10_000_000_000,
            1024 * 1024 * 1024,
            100.0,
            None,
        );

        let expensive = analyzer.get_most_expensive_operations(2);
        assert_eq!(expensive.len(), 2);
        assert_eq!(expensive[0].0, "expensive_op");
    }

    #[test]
    fn test_optimization_suggestions() {
        let analyzer = OperationCostAnalyzer::new(CostAnalysisConfig::default());

        // Record high-cost operation
        analyzer.record_operation_cost(
            "heavy_matmul".to_string(),
            OperationType::MatMul,
            10_000_000_000,         // 10 GFLOPs
            2 * 1024 * 1024 * 1024, // 2 GB
            100.0,
            None,
        );

        let suggestions = analyzer.get_optimization_suggestions();
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_cost_report() {
        let analyzer = OperationCostAnalyzer::new(CostAnalysisConfig::default());

        analyzer.record_operation_cost(
            "test_op".to_string(),
            OperationType::MatMul,
            1_000_000_000,
            1024 * 1024,
            10.0,
            None,
        );

        let report = analyzer.generate_cost_report();
        assert!(report.contains("Cost Analysis Report"));
        assert!(report.contains("test_op"));
    }
}
