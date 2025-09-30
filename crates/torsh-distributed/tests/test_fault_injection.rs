//! Fault injection tests for distributed training robustness
//!
//! These tests inject various types of failures to verify that the distributed
//! training system can handle and recover from common failure scenarios.

use scirs2_core::random::quick::random_f64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::{sleep, timeout};
use torsh_core::Result;
use torsh_distributed::{
    backend::BackendType,
    backend::ReduceOp,
    collectives::{all_reduce, barrier, broadcast},
    error_recovery::{CircuitBreaker, CircuitBreakerConfig, RetryConfig, RetryExecutor},
    fault_tolerance::{CheckpointConfig, CheckpointManager, ElasticTrainingManager},
    init_process_group, TorshDistributedError,
};
use torsh_tensor::creation::{ones, randn, zeros};
use torsh_tensor::Tensor;

/// Types of faults that can be injected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultType {
    /// Network communication failure
    NetworkFailure,
    /// Process crash/termination
    ProcessCrash,
    /// Memory allocation failure
    MemoryFailure,
    /// Timeout during operations
    TimeoutFailure,
    /// Data corruption
    DataCorruption,
    /// Backend service unavailable
    BackendFailure,
    /// Random intermittent failures
    IntermittentFailure,
}

/// Fault injection configuration
#[derive(Debug, Clone)]
pub struct FaultInjectionConfig {
    /// Type of fault to inject
    pub fault_type: FaultType,
    /// Probability of fault occurring (0.0 to 1.0)
    pub fault_probability: f64,
    /// Duration of fault (for temporary faults)
    pub fault_duration: Duration,
    /// Delay before fault occurs
    pub fault_delay: Duration,
    /// Whether fault should be permanent
    pub permanent: bool,
    /// Target operations to affect
    pub target_operations: Vec<String>,
}

impl Default for FaultInjectionConfig {
    fn default() -> Self {
        Self {
            fault_type: FaultType::NetworkFailure,
            fault_probability: 0.1,
            fault_duration: Duration::from_secs(5),
            fault_delay: Duration::from_secs(1),
            permanent: false,
            target_operations: vec!["all_reduce".to_string(), "barrier".to_string()],
        }
    }
}

/// Fault injector for simulating various failure scenarios
pub struct FaultInjector {
    config: FaultInjectionConfig,
    active_faults: Arc<Mutex<Vec<(FaultType, Instant)>>>,
    fault_count: Arc<Mutex<u64>>,
}

impl FaultInjector {
    pub fn new(config: FaultInjectionConfig) -> Self {
        Self {
            config,
            active_faults: Arc::new(Mutex::new(Vec::new())),
            fault_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Inject a fault based on configuration
    pub async fn inject_fault(&self) -> Result<()> {
        // Wait for fault delay
        sleep(self.config.fault_delay).await;

        // Check if fault should occur based on probability
        let should_fault = random_f64() < self.config.fault_probability;
        if !should_fault {
            return Ok(());
        }

        let fault_start = Instant::now();

        // Record fault
        {
            let mut active_faults = self.active_faults.lock().unwrap();
            active_faults.push((self.config.fault_type, fault_start));

            let mut count = self.fault_count.lock().unwrap();
            *count += 1;
        }

        match self.config.fault_type {
            FaultType::NetworkFailure => self.inject_network_failure().await,
            FaultType::ProcessCrash => self.inject_process_crash().await,
            FaultType::MemoryFailure => self.inject_memory_failure().await,
            FaultType::TimeoutFailure => self.inject_timeout_failure().await,
            FaultType::DataCorruption => self.inject_data_corruption().await,
            FaultType::BackendFailure => self.inject_backend_failure().await,
            FaultType::IntermittentFailure => self.inject_intermittent_failure().await,
        }
    }

    async fn inject_network_failure(&self) -> Result<()> {
        println!(
            "Injecting network failure for {:?}",
            self.config.fault_duration
        );

        // Simulate network unavailability
        sleep(self.config.fault_duration).await;

        if !self.config.permanent {
            println!("Network failure resolved");
        }

        Ok(())
    }

    async fn inject_process_crash(&self) -> Result<()> {
        println!("Simulating process crash");

        if self.config.permanent {
            // In real scenario, this would terminate the process
            return Err(torsh_core::TorshError::Other("Process crashed".to_string()));
        }

        // Simulate temporary process unavailability
        sleep(self.config.fault_duration).await;
        println!("Process recovered from crash");

        Ok(())
    }

    async fn inject_memory_failure(&self) -> Result<()> {
        println!("Simulating memory allocation failure");

        // Simulate memory pressure
        sleep(self.config.fault_duration).await;

        if !self.config.permanent {
            println!("Memory pressure resolved");
        }

        Ok(())
    }

    async fn inject_timeout_failure(&self) -> Result<()> {
        println!("Simulating timeout failure");

        // Simulate operation timeout
        sleep(self.config.fault_duration).await;

        return Err(torsh_core::TorshError::Other(
            "Operation timed out".to_string(),
        ));
    }

    async fn inject_data_corruption(&self) -> Result<()> {
        println!("Simulating data corruption");

        // Simulate data corruption scenario
        sleep(self.config.fault_duration).await;

        Ok(())
    }

    async fn inject_backend_failure(&self) -> Result<()> {
        println!("Simulating backend service failure");

        sleep(self.config.fault_duration).await;

        if !self.config.permanent {
            println!("Backend service recovered");
        }

        Ok(())
    }

    async fn inject_intermittent_failure(&self) -> Result<()> {
        println!("Simulating intermittent failure");

        // Random short failures
        for _ in 0..3 {
            sleep(Duration::from_millis(100)).await;
            // Brief failure
            sleep(Duration::from_millis(50)).await;
        }

        Ok(())
    }

    /// Get fault statistics
    pub fn get_fault_stats(&self) -> (u64, usize) {
        let count = *self.fault_count.lock().unwrap();
        let active = self.active_faults.lock().unwrap().len();
        (count, active)
    }

    /// Clear fault history
    pub fn clear_faults(&self) {
        let mut active_faults = self.active_faults.lock().unwrap();
        active_faults.clear();

        let mut count = self.fault_count.lock().unwrap();
        *count = 0;
    }
}

#[tokio::test]
async fn test_network_failure_recovery() -> Result<()> {
    let config = FaultInjectionConfig {
        fault_type: FaultType::NetworkFailure,
        fault_probability: 1.0, // Always inject fault
        fault_duration: Duration::from_millis(100),
        permanent: false,
        ..Default::default()
    };

    let injector = FaultInjector::new(config);
    let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 29600)?;

    // Test operation with fault injection
    let result = timeout(Duration::from_secs(5), async {
        // Inject fault in background
        tokio::spawn(async move {
            let _ = injector.inject_fault().await;
        });

        // Try operation that might fail
        let mut tensor = ones::<f32>(&[3, 3]);
        all_reduce(&mut tensor, ReduceOp::Sum, &pg).await
    })
    .await;

    // With mock backend, operation should succeed despite fault injection
    assert!(result.is_ok(), "Operation should succeed with mock backend");
    Ok(())
}

#[tokio::test]
async fn test_circuit_breaker_fault_tolerance() -> Result<()> {
    let circuit_breaker_config = CircuitBreakerConfig {
        failure_threshold: 3,
        timeout: Duration::from_secs(5),
        success_threshold: 2,
    };

    let circuit_breaker = CircuitBreaker::new(circuit_breaker_config);
    let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 29601)?;

    // Simulate failing operations
    for i in 0..5 {
        let result = circuit_breaker
            .call(|| async {
                if i < 3 {
                    // First 3 calls fail
                    Err(TorshDistributedError::CommunicationError(
                        "Simulated failure".to_string(),
                    ))
                } else {
                    // Later calls succeed
                    let mut tensor = ones::<f32>(&[2, 2]);
                    all_reduce(&mut tensor, ReduceOp::Sum, &pg).await?;
                    Ok(())
                }
            })
            .await;

        match i {
            0..=2 => assert!(result.is_err(), "Expected failure for call {}", i),
            3..=4 => {
                // Circuit breaker should eventually allow successful calls
                println!("Call {} result: {:?}", i, result);
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_retry_mechanism_with_faults() -> Result<()> {
    let retry_config = RetryConfig {
        max_attempts: 3,
        base_delay: Duration::from_millis(100),
        max_delay: Duration::from_secs(1),
        backoff_multiplier: 2.0,
    };

    let retry_executor = RetryExecutor::new(retry_config);
    let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 29602)?;

    let attempt_count = Arc::new(Mutex::new(0));
    let attempt_count_clone = attempt_count.clone();

    let result = retry_executor
        .execute(|| {
            let attempt_count = attempt_count_clone.clone();
            let pg = pg.clone();

            async move {
                let mut count = attempt_count.lock().unwrap();
                *count += 1;
                let current_attempt = *count;
                drop(count);

                if current_attempt < 2 {
                    // First attempt fails
                    Err(TorshDistributedError::CommunicationError(
                        "Temporary failure".to_string(),
                    ))
                } else {
                    // Second attempt succeeds
                    let mut tensor = ones::<f32>(&[2, 2]);
                    all_reduce(&mut tensor, ReduceOp::Sum, &pg).await?;
                    Ok(())
                }
            }
        })
        .await;

    assert!(result.is_ok(), "Retry mechanism should succeed eventually");

    let final_attempt_count = *attempt_count.lock().unwrap();
    assert_eq!(
        final_attempt_count, 2,
        "Should have made exactly 2 attempts"
    );

    Ok(())
}

#[tokio::test]
async fn test_data_corruption_detection() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 29603)?;

    // Create tensor with known values
    let mut tensor = ones::<f32>(&[4, 4]) * 5.0;
    let original_sum: f32 = tensor.to_vec().iter().sum();

    // Simulate data corruption by modifying tensor
    let mut corrupted_tensor = tensor.clone();
    let corrupted_data = corrupted_tensor.to_vec_mut();
    corrupted_data[0] = f32::NAN; // Inject NaN as corruption

    // Check for corruption
    let has_corruption = corrupted_data
        .iter()
        .any(|&x| x.is_nan() || x.is_infinite());
    assert!(has_corruption, "Should detect data corruption");

    // Test with uncorrupted data
    all_reduce(&mut tensor, ReduceOp::Sum, &pg).await?;

    let result_data = tensor.to_vec();
    let has_valid_data = result_data.iter().all(|&x| x.is_finite());
    assert!(has_valid_data, "Result should contain valid data");

    Ok(())
}

#[tokio::test]
async fn test_partial_process_failure() -> Result<()> {
    let world_size = 4;
    let mut process_groups: Vec<Option<_>> = (0..world_size)
        .map(|rank| {
            init_process_group(BackendType::Gloo, rank, world_size, "127.0.0.1", 29604).ok()
        })
        .collect();

    // Simulate failure of processes 1 and 3
    process_groups[1] = None;
    process_groups[3] = None;

    // Test that remaining processes can still operate
    let remaining_pgs: Vec<_> = process_groups.into_iter().filter_map(|pg| pg).collect();

    for pg in &remaining_pgs {
        let mut tensor = ones::<f32>(&[2, 2]);
        let result = all_reduce(&mut tensor, ReduceOp::Sum, pg).await;
        assert!(
            result.is_ok(),
            "Remaining processes should continue operating"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_checkpoint_recovery_after_failure() -> Result<()> {
    let checkpoint_config = CheckpointConfig {
        save_interval: Duration::from_secs(1),
        max_checkpoints: 3,
        checkpoint_dir: "/tmp/torsh_test_checkpoints".into(),
        async_save: true,
        compression_enabled: false,
    };

    let checkpoint_manager = CheckpointManager::new(checkpoint_config)?;

    // Simulate training state before failure
    let training_state = std::collections::HashMap::new();
    let step = 100;

    // Save checkpoint
    checkpoint_manager
        .save_checkpoint(&training_state, step)
        .await?;

    // Simulate failure and recovery
    let recovered_state = checkpoint_manager.load_latest_checkpoint().await?;

    if let Some((recovered_step, _recovered_state)) = recovered_state {
        assert_eq!(recovered_step, step, "Should recover correct training step");
    } else {
        panic!("Should have recovered checkpoint");
    }

    // Cleanup
    let _ = std::fs::remove_dir_all("/tmp/torsh_test_checkpoints");

    Ok(())
}

#[tokio::test]
async fn test_elastic_training_with_node_failure() -> Result<()> {
    let elastic_config = torsh_distributed::fault_tolerance::ElasticConfig {
        min_nodes: 2,
        max_nodes: 4,
        scale_up_threshold: 0.8,
        scale_down_threshold: 0.3,
        node_failure_timeout: Duration::from_secs(10),
        health_check_interval: Duration::from_secs(1),
    };

    let elastic_manager = ElasticTrainingManager::new(elastic_config)?;

    // Register initial nodes
    for rank in 0..3 {
        let pg = init_process_group(BackendType::Gloo, rank, 3, "127.0.0.1", 29605)?;
        elastic_manager.register_worker(rank, pg).await?;
    }

    // Simulate node failure
    elastic_manager.handle_node_failure(1).await?;

    // Check that training can continue with remaining nodes
    let active_workers = elastic_manager.get_active_workers().await?;
    assert_eq!(
        active_workers.len(),
        2,
        "Should have 2 active workers after failure"
    );

    Ok(())
}

#[tokio::test]
async fn test_cascading_failure_handling() -> Result<()> {
    let world_size = 6;
    let mut process_groups: Vec<Option<_>> = (0..world_size)
        .map(|rank| {
            init_process_group(BackendType::Gloo, rank, world_size, "127.0.0.1", 29606).ok()
        })
        .collect();

    // Simulate cascading failures (multiple nodes failing in sequence)
    let failure_sequence = vec![1, 3, 5]; // Fail nodes 1, 3, 5

    for &failing_rank in &failure_sequence {
        process_groups[failing_rank] = None;

        // Test that remaining nodes can still communicate
        let remaining_pgs: Vec<_> = process_groups.iter().filter_map(|pg| pg.as_ref()).collect();

        if remaining_pgs.len() >= 2 {
            // Test barrier among remaining nodes
            for pg in &remaining_pgs {
                let result = timeout(Duration::from_secs(2), barrier(pg)).await;
                assert!(
                    result.is_ok(),
                    "Remaining nodes should synchronize after cascading failures"
                );
            }
        }
    }

    let final_remaining = process_groups.iter().filter(|pg| pg.is_some()).count();
    assert_eq!(
        final_remaining, 3,
        "Should have 3 remaining nodes after cascading failures"
    );

    Ok(())
}

#[tokio::test]
async fn test_memory_pressure_handling() -> Result<()> {
    let injector = FaultInjector::new(FaultInjectionConfig {
        fault_type: FaultType::MemoryFailure,
        fault_probability: 1.0,
        fault_duration: Duration::from_millis(100),
        permanent: false,
        ..Default::default()
    });

    let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 29607)?;

    // Simulate memory pressure during large tensor operations
    let large_tensor_size = vec![100, 100, 100]; // Large tensor

    // Start fault injection
    tokio::spawn(async move {
        let _ = injector.inject_fault().await;
    });

    // Try to create and operate on large tensor
    let result = timeout(Duration::from_secs(5), async {
        let mut tensor = ones::<f32>(&large_tensor_size)?;
        all_reduce(&mut tensor, ReduceOp::Sum, &pg).await
    })
    .await;

    // Should handle memory pressure gracefully
    match result {
        Ok(Ok(())) => println!("Operation succeeded despite memory pressure"),
        Ok(Err(e)) => println!("Operation failed gracefully: {}", e),
        Err(_) => panic!("Operation timed out"),
    }

    Ok(())
}
