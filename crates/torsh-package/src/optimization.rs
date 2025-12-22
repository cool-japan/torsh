//! Package optimization utilities
//!
//! This module provides tools for optimizing package size and performance,
//! including resource deduplication, compression analysis, and optimization recommendations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::error::Result;

use crate::package::Package;
use crate::utils::format_file_size;

/// Package optimization report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    /// Original package size in bytes
    pub original_size: u64,
    /// Estimated optimized size in bytes
    pub optimized_size: u64,
    /// Potential size savings in bytes
    pub savings: u64,
    /// Savings percentage
    pub savings_percent: f64,
    /// List of optimization opportunities
    pub opportunities: Vec<OptimizationOpportunity>,
    /// Resource deduplication analysis
    pub deduplication: DeduplicationAnalysis,
    /// Compression analysis
    pub compression: CompressionAnalysis,
}

/// A single optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Description of the opportunity
    pub description: String,
    /// Potential size savings in bytes
    pub potential_savings: u64,
    /// Priority level (1-5, 5 being highest)
    pub priority: u8,
    /// Resources affected
    pub affected_resources: Vec<String>,
}

/// Types of optimizations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Resource deduplication
    Deduplication,
    /// Better compression algorithm selection
    CompressionUpgrade,
    /// Remove unused resources
    RemoveUnused,
    /// Compress uncompressed resources
    AddCompression,
    /// Merge small resources
    MergeSmall,
    /// Split large resources
    SplitLarge,
}

/// Resource deduplication analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationAnalysis {
    /// Total number of resources
    pub total_resources: usize,
    /// Number of unique resources (by hash)
    pub unique_resources: usize,
    /// Number of duplicate resources
    pub duplicate_count: usize,
    /// Duplicate groups (hash -> list of resource names)
    pub duplicate_groups: HashMap<String, Vec<String>>,
    /// Potential savings from deduplication in bytes
    pub potential_savings: u64,
}

/// Compression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionAnalysis {
    /// Resources that would benefit from compression
    pub compressible_resources: Vec<CompressibleResource>,
    /// Resources already well-compressed
    pub well_compressed_count: usize,
    /// Total potential compression savings in bytes
    pub potential_savings: u64,
}

/// A resource that could benefit from compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressibleResource {
    /// Resource name
    pub name: String,
    /// Current size in bytes
    pub current_size: u64,
    /// Estimated compressed size in bytes
    pub estimated_compressed_size: u64,
    /// Potential savings in bytes
    pub potential_savings: u64,
    /// Estimated compression ratio
    pub compression_ratio: f64,
}

/// Package optimizer
pub struct PackageOptimizer {
    /// Minimum size threshold for compression (bytes)
    pub min_compression_size: u64,
    /// Minimum compression ratio to recommend
    pub min_compression_ratio: f64,
    /// Enable resource deduplication
    pub enable_deduplication: bool,
}

impl PackageOptimizer {
    /// Create a new package optimizer with default settings
    pub fn new() -> Self {
        Self {
            min_compression_size: 1024, // 1KB
            min_compression_ratio: 0.7, // 30% savings
            enable_deduplication: true,
        }
    }

    /// Analyze package and generate optimization report
    pub fn analyze(&self, package: &Package) -> Result<OptimizationReport> {
        let original_size = self.calculate_package_size(package);

        // Perform various analyses
        let deduplication = if self.enable_deduplication {
            self.analyze_deduplication(package)
        } else {
            DeduplicationAnalysis {
                total_resources: 0,
                unique_resources: 0,
                duplicate_count: 0,
                duplicate_groups: HashMap::new(),
                potential_savings: 0,
            }
        };

        let compression = self.analyze_compression(package)?;

        // Generate optimization opportunities
        let mut opportunities = Vec::new();

        // Add deduplication opportunities
        if deduplication.duplicate_count > 0 {
            opportunities.push(OptimizationOpportunity {
                optimization_type: OptimizationType::Deduplication,
                description: format!(
                    "Found {} duplicate resources that could be deduplicated",
                    deduplication.duplicate_count
                ),
                potential_savings: deduplication.potential_savings,
                priority: 5,
                affected_resources: deduplication
                    .duplicate_groups
                    .values()
                    .flatten()
                    .cloned()
                    .collect(),
            });
        }

        // Add compression opportunities
        for resource in &compression.compressible_resources {
            if resource.potential_savings > self.min_compression_size {
                opportunities.push(OptimizationOpportunity {
                    optimization_type: OptimizationType::AddCompression,
                    description: format!(
                        "Resource '{}' could be compressed to save {}",
                        resource.name,
                        format_file_size(resource.potential_savings)
                    ),
                    potential_savings: resource.potential_savings,
                    priority: if resource.compression_ratio < 0.5 {
                        4
                    } else {
                        3
                    },
                    affected_resources: vec![resource.name.clone()],
                });
            }
        }

        // Calculate total potential savings
        let total_savings = deduplication.potential_savings + compression.potential_savings;
        let optimized_size = original_size.saturating_sub(total_savings);
        let savings_percent = if original_size > 0 {
            (total_savings as f64 / original_size as f64) * 100.0
        } else {
            0.0
        };

        // Sort opportunities by priority
        opportunities.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(OptimizationReport {
            original_size,
            optimized_size,
            savings: total_savings,
            savings_percent,
            opportunities,
            deduplication,
            compression,
        })
    }

    /// Calculate total package size
    fn calculate_package_size(&self, _package: &Package) -> u64 {
        // TODO: Implement when Package API is available
        // For now, return estimate
        0
    }

    /// Analyze resource deduplication opportunities
    fn analyze_deduplication(&self, _package: &Package) -> DeduplicationAnalysis {
        // TODO: Implement when Package API is available
        let hash_to_resources: HashMap<String, Vec<String>> = HashMap::new();
        let _total_size = 0u64;
        let duplicate_savings = 0u64;

        // Group resources by hash (commented out as Package internals are private)
        // for resource in package.resources() {
        //     let hash = calculate_hash(&resource.data);
        //     let size = resource.data.len() as u64;
        //     total_size += size;
        //
        //     hash_to_resources
        //         .entry(hash)
        //         .or_insert_with(Vec::new)
        //         .push(resource.name.clone());
        // }

        // Find duplicate groups
        let duplicate_groups: HashMap<String, Vec<String>> = hash_to_resources
            .into_iter()
            .filter(|(_, resources)| resources.len() > 1)
            .collect();

        let duplicate_count: usize = duplicate_groups
            .values()
            .map(|v| v.len() - 1) // -1 because we keep one copy
            .sum();

        DeduplicationAnalysis {
            total_resources: 0, // Would be package.resources().len()
            unique_resources: 0,
            duplicate_count,
            duplicate_groups,
            potential_savings: duplicate_savings,
        }
    }

    /// Analyze compression opportunities
    fn analyze_compression(&self, _package: &Package) -> Result<CompressionAnalysis> {
        let compressible_resources = Vec::new();
        let well_compressed_count = 0;
        let total_savings = 0u64;

        // Analyze each resource (commented out as Package internals are private)
        // for resource in package.resources() {
        //     let size = resource.data.len() as u64;
        //
        //     if size < self.min_compression_size {
        //         continue;
        //     }
        //
        //     let compression_ratio = estimate_compression_ratio(&resource.data);
        //
        //     if compression_ratio <= self.min_compression_ratio {
        //         let estimated_compressed = (size as f64 * compression_ratio) as u64;
        //         let savings = size.saturating_sub(estimated_compressed);
        //
        //         compressible_resources.push(CompressibleResource {
        //             name: resource.name.clone(),
        //             current_size: size,
        //             estimated_compressed_size: estimated_compressed,
        //             potential_savings: savings,
        //             compression_ratio,
        //         });
        //
        //         total_savings += savings;
        //     } else {
        //         well_compressed_count += 1;
        //     }
        // }

        Ok(CompressionAnalysis {
            compressible_resources,
            well_compressed_count,
            potential_savings: total_savings,
        })
    }

    /// Apply optimizations to a package
    pub fn optimize(&self, package: &mut Package) -> Result<OptimizationReport> {
        let report = self.analyze(package)?;

        // Apply deduplication
        if self.enable_deduplication && !report.deduplication.duplicate_groups.is_empty() {
            self.apply_deduplication(package, &report.deduplication)?;
        }

        // Apply compression improvements
        // (Would need to implement compression application logic)

        Ok(report)
    }

    /// Apply deduplication optimizations
    fn apply_deduplication(
        &self,
        _package: &mut Package,
        _analysis: &DeduplicationAnalysis,
    ) -> Result<()> {
        // TODO: Implement when Package API supports resource manipulation
        Ok(())
    }
}

impl Default for PackageOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = PackageOptimizer::new();
        assert_eq!(optimizer.min_compression_size, 1024);
        assert!(optimizer.enable_deduplication);
    }

    #[test]
    fn test_optimization_type() {
        let opt_type = OptimizationType::Deduplication;
        assert_eq!(opt_type, OptimizationType::Deduplication);
    }

    #[test]
    fn test_default_optimizer() {
        let optimizer = PackageOptimizer::default();
        assert_eq!(optimizer.min_compression_size, 1024);
    }

    #[test]
    fn test_deduplication_analysis() {
        let analysis = DeduplicationAnalysis {
            total_resources: 10,
            unique_resources: 7,
            duplicate_count: 3,
            duplicate_groups: HashMap::new(),
            potential_savings: 1024,
        };

        assert_eq!(analysis.total_resources, 10);
        assert_eq!(analysis.duplicate_count, 3);
        assert_eq!(analysis.potential_savings, 1024);
    }

    #[test]
    fn test_compressible_resource() {
        let resource = CompressibleResource {
            name: "test.txt".to_string(),
            current_size: 10000,
            estimated_compressed_size: 3000,
            potential_savings: 7000,
            compression_ratio: 0.3,
        };

        assert_eq!(resource.current_size, 10000);
        assert_eq!(resource.potential_savings, 7000);
        assert!(resource.compression_ratio < 0.5);
    }
}
