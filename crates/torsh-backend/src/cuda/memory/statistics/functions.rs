//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::allocation::AllocationStats;

use super::types::{
    CudaMemoryStatisticsManager, DistributionType, EfficiencyTrend, ErrorImpactLevel,
    GlobalMemoryStatistics, MemoryPressureLevel, StatisticsConfig,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_statistics_manager_creation() {
        let config = StatisticsConfig::default();
        let manager = CudaMemoryStatisticsManager::new(config);
        assert!(manager.config.enable_detailed_tracking);
        assert!(manager.config.enable_historical_data);
    }
    #[test]
    fn test_memory_pressure_levels() {
        assert!(MemoryPressureLevel::Critical > MemoryPressureLevel::High);
        assert!(MemoryPressureLevel::High > MemoryPressureLevel::Medium);
        assert!(MemoryPressureLevel::Medium > MemoryPressureLevel::Low);
        assert!(MemoryPressureLevel::Low > MemoryPressureLevel::None);
    }
    #[test]
    fn test_distribution_types() {
        assert_eq!(DistributionType::Normal, DistributionType::Normal);
        assert_ne!(DistributionType::Normal, DistributionType::LogNormal);
    }
    #[test]
    fn test_efficiency_trends() {
        assert_eq!(EfficiencyTrend::Improving, EfficiencyTrend::Improving);
        assert_ne!(EfficiencyTrend::Stable, EfficiencyTrend::Declining);
    }
    #[test]
    fn test_error_impact_levels() {
        assert!(ErrorImpactLevel::Critical > ErrorImpactLevel::High);
        assert!(ErrorImpactLevel::High > ErrorImpactLevel::Moderate);
        assert!(ErrorImpactLevel::Moderate > ErrorImpactLevel::Low);
        assert!(ErrorImpactLevel::Low > ErrorImpactLevel::Minimal);
    }
    #[test]
    fn test_statistics_update() {
        let config = StatisticsConfig::default();
        let manager = CudaMemoryStatisticsManager::new(config);
        let stats = AllocationStats {
            total_allocations: 100,
            current_bytes_allocated: 1024 * 1024,
            ..Default::default()
        };
        manager.update_device_stats(0, &stats);
        let device_stats = manager
            .device_stats
            .read()
            .expect("lock should not be poisoned");
        assert!(device_stats.contains_key(&0));
    }
}
/// Memory usage statistics (alias to GlobalMemoryStatistics)
pub type MemoryUsageStatistics = GlobalMemoryStatistics;
