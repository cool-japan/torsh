//! Utilities for distributed training setup and optimization
//!
//! This module provides helper functions for configuring distributed training
//! environments, optimizing communication parameters, and benchmarking performance.

use std::time::Duration;
use torsh_core::error::Result;

use super::config::{DistributedBackend, DistributedConfig};

/// Detect distributed training environment and create config
pub fn detect_distributed_env() -> DistributedConfig {
    let mut config = DistributedConfig::default();

    // Check for common distributed training environment variables
    if let Ok(world_size) = std::env::var("WORLD_SIZE") {
        if let Ok(size) = world_size.parse::<usize>() {
            config.world_size = size;
        }
    }

    if let Ok(rank) = std::env::var("RANK") {
        if let Ok(r) = rank.parse::<usize>() {
            config.rank = r;
        }
    }

    // Detect backend
    if std::env::var("NCCL_SOCKET_IFNAME").is_ok() {
        config.backend = DistributedBackend::Nccl;
    } else if std::env::var("GLOO_SOCKET_IFNAME").is_ok() {
        config.backend = DistributedBackend::Gloo;
    } else if std::env::var("OMPI_COMM_WORLD_SIZE").is_ok() {
        config.backend = DistributedBackend::Mpi;
    }

    config
}

/// Calculate optimal bucket size based on network bandwidth and model size
pub fn calculate_optimal_bucket_size(
    network_bandwidth_gbps: f64,
    _model_size_mb: f64,
    _world_size: usize,
) -> usize {
    // Simple heuristic for bucket size calculation
    let network_bandwidth_mbps = network_bandwidth_gbps * 1000.0;
    let optimal_latency_ms = 10.0; // Target 10ms per communication

    let optimal_size_mb = (network_bandwidth_mbps * optimal_latency_ms) / 1000.0;
    let optimal_size_bytes = (optimal_size_mb * 1024.0 * 1024.0) as usize;

    // Clamp between reasonable bounds
    optimal_size_bytes.max(1024 * 1024).min(128 * 1024 * 1024) // 1MB to 128MB
}

/// Benchmark communication performance
pub fn benchmark_communication(config: &DistributedConfig, data_size: usize) -> Result<Duration> {
    // Placeholder for communication benchmark
    // In real implementation, this would:
    // 1. Create test data
    // 2. Perform all-reduce operation
    // 3. Measure time

    tracing::info!("Benchmarking communication for {} bytes", data_size);

    // Simulate communication time based on data size and world size
    let base_latency = Duration::from_millis(1);
    let bandwidth_delay = Duration::from_nanos(
        (data_size as u64 * config.world_size as u64) / 100, // Simulate 100 GB/s bandwidth
    );

    Ok(base_latency + bandwidth_delay)
}

/// Detect available network interfaces for distributed communication
pub fn detect_network_interfaces() -> Vec<NetworkInterface> {
    // Placeholder implementation - in real code this would:
    // 1. Query system network interfaces
    // 2. Test connectivity between ranks
    // 3. Measure bandwidth for each interface

    vec![
        NetworkInterface {
            name: "eth0".to_string(),
            ip_address: "192.168.1.100".to_string(),
            bandwidth_mbps: 1000.0,
            is_available: true,
            interface_type: NetworkInterfaceType::Ethernet,
        },
        NetworkInterface {
            name: "ib0".to_string(),
            ip_address: "10.0.0.100".to_string(),
            bandwidth_mbps: 40000.0,
            is_available: false, // Placeholder - would be detected
            interface_type: NetworkInterfaceType::InfiniBand,
        },
    ]
}

/// Information about a network interface
#[derive(Debug, Clone)]
pub struct NetworkInterface {
    /// Interface name (e.g., "eth0", "ib0")
    pub name: String,
    /// IP address assigned to this interface
    pub ip_address: String,
    /// Maximum bandwidth in Mbps
    pub bandwidth_mbps: f64,
    /// Whether this interface is available for use
    pub is_available: bool,
    /// Type of network interface
    pub interface_type: NetworkInterfaceType,
}

/// Type of network interface
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkInterfaceType {
    /// Standard Ethernet
    Ethernet,
    /// InfiniBand for high-performance computing
    InfiniBand,
    /// Wireless network
    Wireless,
    /// Loopback interface
    Loopback,
    /// Unknown or custom interface type
    Unknown,
}

/// Calculate optimal number of communication streams
pub fn calculate_optimal_streams(
    total_bandwidth_gbps: f64,
    per_stream_bandwidth_gbps: f64,
    max_streams: usize,
) -> usize {
    let theoretical_optimal = (total_bandwidth_gbps / per_stream_bandwidth_gbps).ceil() as usize;
    theoretical_optimal.min(max_streams).max(1)
}

/// Estimate communication cost for different patterns
pub fn estimate_communication_cost(
    data_size: usize,
    world_size: usize,
    pattern: super::config::CommunicationPattern,
    bandwidth_gbps: f64,
    latency_ms: f64,
) -> Duration {
    use super::config::CommunicationPattern;

    let bandwidth_bytes_per_ms = (bandwidth_gbps * 1_000_000_000.0) / 1000.0;
    let base_latency = Duration::from_millis(latency_ms as u64);

    match pattern {
        CommunicationPattern::AllReduce => {
            // All-reduce: 2 * (n-1) / n * data transfer + latency * log(n)
            let data_transfer_time = 2.0 * (world_size - 1) as f64 / world_size as f64
                * data_size as f64
                / bandwidth_bytes_per_ms;
            let latency_overhead = latency_ms * (world_size as f64).log2();

            Duration::from_millis((data_transfer_time + latency_overhead) as u64)
        }
        CommunicationPattern::ReduceScatter => {
            // Reduce-scatter: (n-1) / n * data transfer + latency * log(n)
            let data_transfer_time = (world_size - 1) as f64 / world_size as f64
                * data_size as f64
                / bandwidth_bytes_per_ms;
            let latency_overhead = latency_ms * (world_size as f64).log2();

            Duration::from_millis((data_transfer_time + latency_overhead) as u64)
        }
        CommunicationPattern::AllGather => {
            // All-gather: (n-1) / n * data transfer + latency * log(n)
            let data_transfer_time = (world_size - 1) as f64 / world_size as f64
                * data_size as f64
                / bandwidth_bytes_per_ms;
            let latency_overhead = latency_ms * (world_size as f64).log2();

            Duration::from_millis((data_transfer_time + latency_overhead) as u64)
        }
        CommunicationPattern::ParameterServer => {
            // Parameter server: 2 * data transfer + 2 * latency (round trip)
            let data_transfer_time = 2.0 * data_size as f64 / bandwidth_bytes_per_ms;
            let latency_overhead = 2.0 * latency_ms;

            Duration::from_millis((data_transfer_time + latency_overhead) as u64)
        }
        CommunicationPattern::Ring => {
            // Ring: 2 * (n-1) / n * data transfer + latency * 2 * (n-1)
            let data_transfer_time = 2.0 * (world_size - 1) as f64 / world_size as f64
                * data_size as f64
                / bandwidth_bytes_per_ms;
            let latency_overhead = latency_ms * 2.0 * (world_size - 1) as f64;

            Duration::from_millis((data_transfer_time + latency_overhead) as u64)
        }
        CommunicationPattern::Tree => {
            // Tree: data transfer + latency * 2 * log(n)
            let data_transfer_time = data_size as f64 / bandwidth_bytes_per_ms;
            let latency_overhead = latency_ms * 2.0 * (world_size as f64).log2();

            Duration::from_millis((data_transfer_time + latency_overhead) as u64)
        }
    }
    .max(base_latency)
}

/// Determine the best communication pattern for given parameters
pub fn recommend_communication_pattern(
    data_size: usize,
    world_size: usize,
    bandwidth_gbps: f64,
    latency_ms: f64,
) -> super::config::CommunicationPattern {
    use super::config::CommunicationPattern;

    let patterns = [
        CommunicationPattern::AllReduce,
        CommunicationPattern::ReduceScatter,
        CommunicationPattern::AllGather,
        CommunicationPattern::ParameterServer,
        CommunicationPattern::Ring,
        CommunicationPattern::Tree,
    ];

    let mut best_pattern = CommunicationPattern::AllReduce;
    let mut best_cost = Duration::from_secs(u64::MAX);

    for pattern in patterns {
        let cost = estimate_communication_cost(data_size, world_size, pattern, bandwidth_gbps, latency_ms);
        if cost < best_cost {
            best_cost = cost;
            best_pattern = pattern;
        }
    }

    tracing::info!(
        "Recommended communication pattern: {:?} (estimated cost: {:?})",
        best_pattern,
        best_cost
    );

    best_pattern
}

/// Environment configuration helpers
pub mod env {
    use super::*;

    /// Common environment variable names for distributed training
    pub const WORLD_SIZE: &str = "WORLD_SIZE";
    pub const RANK: &str = "RANK";
    pub const LOCAL_RANK: &str = "LOCAL_RANK";
    pub const MASTER_ADDR: &str = "MASTER_ADDR";
    pub const MASTER_PORT: &str = "MASTER_PORT";

    /// NCCL-specific environment variables
    pub const NCCL_SOCKET_IFNAME: &str = "NCCL_SOCKET_IFNAME";
    pub const NCCL_DEBUG: &str = "NCCL_DEBUG";
    pub const NCCL_IB_DISABLE: &str = "NCCL_IB_DISABLE";

    /// Gloo-specific environment variables
    pub const GLOO_SOCKET_IFNAME: &str = "GLOO_SOCKET_IFNAME";

    /// MPI-specific environment variables
    pub const OMPI_COMM_WORLD_SIZE: &str = "OMPI_COMM_WORLD_SIZE";
    pub const OMPI_COMM_WORLD_RANK: &str = "OMPI_COMM_WORLD_RANK";

    /// Get environment variable as usize
    pub fn get_env_usize(name: &str) -> Option<usize> {
        std::env::var(name).ok()?.parse().ok()
    }

    /// Get environment variable as string
    pub fn get_env_string(name: &str) -> Option<String> {
        std::env::var(name).ok()
    }

    /// Set up environment for distributed training
    pub fn setup_distributed_env(
        world_size: usize,
        rank: usize,
        master_addr: &str,
        master_port: u16,
    ) {
        std::env::set_var(WORLD_SIZE, world_size.to_string());
        std::env::set_var(RANK, rank.to_string());
        std::env::set_var(MASTER_ADDR, master_addr);
        std::env::set_var(MASTER_PORT, master_port.to_string());
    }

    /// Validate distributed environment setup
    pub fn validate_distributed_env() -> Result<()> {
        let world_size = get_env_usize(WORLD_SIZE)
            .ok_or_else(|| torsh_core::error::TorshError::AutogradError(
                format!("{} environment variable not set or invalid", WORLD_SIZE)
            ))?;

        let rank = get_env_usize(RANK)
            .ok_or_else(|| torsh_core::error::TorshError::AutogradError(
                format!("{} environment variable not set or invalid", RANK)
            ))?;

        if rank >= world_size {
            return Err(torsh_core::error::TorshError::AutogradError(
                format!("Rank {} must be less than world size {}", rank, world_size)
            ));
        }

        if world_size == 0 {
            return Err(torsh_core::error::TorshError::AutogradError(
                "World size must be greater than 0".to_string()
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_optimal_bucket_size() {
        let bucket_size = calculate_optimal_bucket_size(10.0, 100.0, 4);
        assert!(bucket_size >= 1024 * 1024); // At least 1MB
        assert!(bucket_size <= 128 * 1024 * 1024); // At most 128MB
    }

    #[test]
    fn test_calculate_optimal_streams() {
        assert_eq!(calculate_optimal_streams(40.0, 10.0, 8), 4);
        assert_eq!(calculate_optimal_streams(40.0, 10.0, 2), 2);
        assert_eq!(calculate_optimal_streams(5.0, 10.0, 8), 1);
    }

    #[test]
    fn test_estimate_communication_cost() {
        let cost = estimate_communication_cost(
            1024 * 1024, // 1MB
            4,            // 4 ranks
            super::super::config::CommunicationPattern::AllReduce,
            10.0,         // 10 Gbps
            1.0,          // 1ms latency
        );
        assert!(cost.as_millis() > 0);
    }

    #[test]
    fn test_env_validation() {
        // This test would need to mock environment variables
        // For now, just test that the function exists
        let _ = env::validate_distributed_env();
    }
}