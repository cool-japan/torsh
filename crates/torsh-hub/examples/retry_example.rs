//! Example demonstrating advanced retry logic with exponential backoff
//!
//! This example shows how to use the retry module for robust network operations

use std::time::Duration;
use torsh_core::error::TorshError;
use torsh_hub::retry::{retry_with_backoff, retry_with_backoff_async, CircuitBreaker, RetryConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Hub Retry Logic Example ===\n");

    // Example 1: Basic retry with default config
    println!("1. Basic Retry with Default Config");
    println!("-----------------------------------");

    let mut attempt_count = 0;
    let result = retry_with_backoff(&RetryConfig::default(), || {
        attempt_count += 1;
        println!("  Attempt {}", attempt_count);

        if attempt_count < 3 {
            Err(TorshError::IoError("Temporary network error".to_string()))
        } else {
            Ok("Success!")
        }
    });

    match result {
        Ok(value) => println!("  Result: {}", value),
        Err(e) => println!("  Failed: {}", e),
    }

    // Example 2: Custom retry config with aggressive settings
    println!("\n2. Aggressive Retry Config");
    println!("--------------------------");

    let aggressive_config = RetryConfig::aggressive();
    println!("  Max attempts: {}", aggressive_config.max_attempts);
    println!("  Base delay: {:?}", aggressive_config.base_delay);
    println!(
        "  Backoff multiplier: {}",
        aggressive_config.backoff_multiplier
    );

    // Example 3: Conservative retry config
    println!("\n3. Conservative Retry Config");
    println!("----------------------------");

    let conservative_config = RetryConfig::conservative();
    println!("  Max attempts: {}", conservative_config.max_attempts);
    println!("  Base delay: {:?}", conservative_config.base_delay);
    println!("  Max delay: {:?}", conservative_config.max_delay);

    // Example 4: Circuit breaker pattern
    println!("\n4. Circuit Breaker Pattern");
    println!("--------------------------");

    let mut circuit_breaker = CircuitBreaker::new(3, Duration::from_secs(60));

    // Simulate failures
    for i in 1..=5 {
        if circuit_breaker.allow_request() {
            println!("  Request {}: Allowed", i);
            // Simulate failure
            circuit_breaker.record_failure();
        } else {
            println!("  Request {}: Rejected (circuit open)", i);
        }
    }

    println!("  Circuit state: {:?}", circuit_breaker.state());

    // Example 5: Custom retry configuration
    println!("\n5. Custom Retry Configuration");
    println!("------------------------------");

    let custom_config = RetryConfig {
        max_attempts: 4,
        base_delay: Duration::from_millis(200),
        max_delay: Duration::from_secs(5),
        backoff_multiplier: 1.5,
        use_jitter: true,
        timeout: Some(Duration::from_secs(30)),
        use_circuit_breaker: false,
    };

    println!("  Max attempts: {}", custom_config.max_attempts);
    println!("  Base delay: {:?}", custom_config.base_delay);
    println!("  Use jitter: {}", custom_config.use_jitter);

    // Example 6: Demonstrate exponential backoff
    println!("\n6. Exponential Backoff Demo");
    println!("----------------------------");

    let backoff_config = RetryConfig {
        base_delay: Duration::from_millis(100),
        max_delay: Duration::from_secs(10),
        backoff_multiplier: 2.0,
        use_jitter: false,
        max_attempts: 5,
        ..Default::default()
    };

    let mut demo_attempt = 0;
    let _ = retry_with_backoff(&backoff_config, || {
        demo_attempt += 1;
        let start = std::time::Instant::now();

        let result = if demo_attempt < 5 {
            Err(TorshError::IoError("Demo error".to_string()))
        } else {
            Ok(())
        };

        if demo_attempt > 1 {
            println!(
                "  Attempt {} after ~{}ms delay",
                demo_attempt,
                start.elapsed().as_millis()
            );
        } else {
            println!("  Attempt {} (initial)", demo_attempt);
        }

        result
    });

    println!("\n=== Example Complete ===");
    Ok(())
}

// Async example (requires tokio runtime)
#[allow(dead_code)]
async fn async_retry_example() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::{Arc, Mutex};

    println!("\nAsync Retry Example");
    println!("-------------------");

    let attempts = Arc::new(Mutex::new(0));
    let config = RetryConfig::new(3, Duration::from_millis(100));

    let attempts_clone = attempts.clone();
    let result = retry_with_backoff_async(&config, move || {
        let attempts = attempts_clone.clone();
        async move {
            let mut count = attempts.lock().unwrap();
            *count += 1;
            let current = *count;
            drop(count);

            println!("  Async attempt {}", current);

            if current < 3 {
                Err(TorshError::IoError("Async network error".to_string()))
            } else {
                Ok("Async success!")
            }
        }
    })
    .await;

    match result {
        Ok(value) => println!("  Async result: {}", value),
        Err(e) => println!("  Async failed: {}", e),
    }

    Ok(())
}
