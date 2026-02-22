//! Advanced CUDA Event Coordination Demo
//!
//! This example demonstrates the enhanced event coordination system for
//! operation-level synchronization across multiple CUDA streams, including:
//! - Event pool management for efficient event reuse
//! - Operation coordinator for dependency tracking
//! - Cross-stream barriers for global synchronization
//! - Asynchronous event waiting with callbacks
//! - Deadlock detection in operation dependencies

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;

#[cfg(feature = "cuda")]
use torsh_backend::cuda::{
    CudaBackend, CudaBackendConfig, CudaDevice, CudaStream, CudaEvent,
    event_coordination::{
        EventPool, OperationCoordinator, CrossStreamBarrier, AsyncEventWaiter,
        OperationType, EventPriority, CoordinationMetrics,
    }
};

#[cfg(feature = "cuda")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Advanced CUDA Event Coordination Demo ===\\n");

    // Check if CUDA is available
    if !torsh_backend::cuda::is_available() {
        println!("CUDA is not available on this system.");
        return Ok(());
    }

    // Initialize CUDA backend
    let config = CudaBackendConfig::default();
    let backend = CudaBackend::initialize(config).await?;
    
    println!("✓ CUDA Backend initialized successfully");
    println!("✓ Device: {}", backend.device().name());

    // Demonstrate event pool management
    demo_event_pool().await?;
    
    // Demonstrate operation coordination
    demo_operation_coordination().await?;
    
    // Demonstrate cross-stream barriers
    demo_cross_stream_barriers().await?;
    
    // Demonstrate asynchronous event waiting
    demo_async_event_waiting().await?;
    
    // Demonstrate deadlock detection
    demo_deadlock_detection().await?;
    
    // Demonstrate complex coordination scenarios
    demo_complex_coordination().await?;

    println!("\\n=== Demo Complete ===");
    println!("🎉 Advanced event coordination functionality demonstrated successfully!");

    Ok(())
}

#[cfg(feature = "cuda")]
async fn demo_event_pool() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Event Pool Management Demo ===");
    
    // Create event pool with regular and timing events
    let event_pool = Arc::new(EventPool::new(10, 5)?);
    println!("✓ Created event pool with 10 regular events and 5 timing events");
    
    // Acquire and release events
    let mut acquired_events = Vec::new();
    
    for i in 0..7 {
        let use_timing = i % 3 == 0;  // Every 3rd event uses timing
        let event = event_pool.acquire_event(use_timing)?;
        println!("  • Acquired event {} (timing: {})", i + 1, use_timing);
        acquired_events.push(event);
    }
    
    let (available, timing, in_use) = event_pool.utilization();
    println!("✓ Pool utilization: {} available, {} timing, {} in use", available, timing, in_use);
    
    // Release half the events
    for event in acquired_events.drain(0..3) {
        event_pool.release_event(event);
    }
    
    let (available, timing, in_use) = event_pool.utilization();
    println!("✓ After partial release: {} available, {} timing, {} in use", available, timing, in_use);
    
    // Release remaining events
    for event in acquired_events {
        event_pool.release_event(event);
    }
    
    let (available, timing, in_use) = event_pool.utilization();
    println!("✓ After full release: {} available, {} timing, {} in use", available, timing, in_use);
    
    Ok(())
}

#[cfg(feature = "cuda")]
async fn demo_operation_coordination() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Operation Coordination Demo ===");
    
    let event_pool = Arc::new(EventPool::new(20, 10)?);
    let coordinator = OperationCoordinator::new(Arc::clone(&event_pool));
    
    // Create streams for operations
    let stream1 = CudaStream::new()?;
    let stream2 = CudaStream::new()?;
    let stream3 = CudaStream::new()?;
    
    println!("✓ Created operation coordinator and 3 streams");
    
    // Register operations with dependencies
    let op1 = coordinator.register_operation(
        OperationType::MemoryTransfer,
        EventPriority::High,
        &stream1,
        vec![],  // No dependencies
        "Initial data transfer".to_string(),
    )?;
    
    let op2 = coordinator.register_operation(
        OperationType::Kernel,
        EventPriority::Normal,
        &stream2,
        vec![op1],  // Depends on op1
        "Main computation kernel".to_string(),
    )?;
    
    let op3 = coordinator.register_operation(
        OperationType::Reduction,
        EventPriority::High,
        &stream3,
        vec![op2],  // Depends on op2
        "Result reduction".to_string(),
    )?;
    
    println!("✓ Registered operations with dependency chain:");
    println!("  • Op {} (MemoryTransfer) → Op {} (Kernel) → Op {} (Reduction)", op1, op2, op3);
    
    // Add completion callbacks
    coordinator.add_completion_callback(op1, || {
        println!("  🔄 Data transfer completed");
    });
    
    coordinator.add_completion_callback(op2, || {
        println!("  🔄 Kernel computation completed");
    });
    
    coordinator.add_completion_callback(op3, || {
        println!("  🔄 Reduction completed");
    });
    
    // Execute operations in order
    println!("✓ Executing operations with dependency coordination:");
    
    coordinator.begin_operation(op1, &stream1)?;
    thread::sleep(Duration::from_millis(5)); // Simulate work
    coordinator.complete_operation(op1)?;
    
    coordinator.begin_operation(op2, &stream2)?;
    thread::sleep(Duration::from_millis(10)); // Simulate work
    coordinator.complete_operation(op2)?;
    
    coordinator.begin_operation(op3, &stream3)?;
    thread::sleep(Duration::from_millis(3)); // Simulate work
    coordinator.complete_operation(op3)?;
    
    // Get coordination metrics
    let metrics = coordinator.metrics();
    println!("✓ Coordination metrics:");
    println!("  • Total operations: {}", metrics.total_operations);
    println!("  • Completed operations: {}", metrics.completed_operations);
    
    Ok(())
}

#[cfg(feature = "cuda")]
async fn demo_cross_stream_barriers() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Cross-Stream Barriers Demo ===");
    
    let event_pool = Arc::new(EventPool::new(15, 8)?);
    
    // Create multiple streams for barrier synchronization
    let streams: Vec<Arc<CudaStream>> = (0..4)
        .map(|_| Arc::new(CudaStream::new().unwrap()))
        .collect();
    
    println!("✓ Created {} streams for barrier demonstration", streams.len());
    
    // Create cross-stream barrier
    let barrier = CrossStreamBarrier::new(streams.clone(), Arc::clone(&event_pool))?;
    println!("✓ Created cross-stream barrier");
    
    // Simulate concurrent work on each stream
    let work_handles: Vec<_> = streams.iter().enumerate().map(|(i, stream)| {
        let stream = Arc::clone(stream);
        thread::spawn(move || {
            // Simulate different amounts of work
            let work_duration = Duration::from_millis(5 + (i * 2) as u64);
            thread::sleep(work_duration);
            println!("  • Stream {} completed work ({:?})", stream.id(), work_duration);
        })
    }).collect();
    
    // Wait for all work to complete
    for handle in work_handles {
        handle.join().unwrap();
    }
    
    // Execute barrier synchronization
    println!("✓ Executing barrier synchronization across all streams");
    let sync_duration = barrier.synchronize()?;
    println!("✓ Barrier synchronization completed in {:?}", sync_duration);
    
    // Test waiting on specific stream
    let test_stream = CudaStream::new()?;
    barrier.wait_on_stream(&test_stream)?;
    println!("✓ Successfully waited for barrier on additional stream");
    
    Ok(())
}

#[cfg(feature = "cuda")]
async fn demo_async_event_waiting() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Asynchronous Event Waiting Demo ===");
    
    let async_waiter = AsyncEventWaiter::new();
    println!("✓ Created asynchronous event waiter");
    
    // Create events and set up async waiting
    let event1 = Arc::new(CudaEvent::new()?);
    let event2 = Arc::new(CudaEvent::new_with_timing()?);
    let event3 = Arc::new(CudaEvent::new()?);
    
    let callback_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    
    // Set up async waits with callbacks
    let counter1 = Arc::clone(&callback_counter);
    let wait_id1 = async_waiter.wait_async(Arc::clone(&event1), move || {
        counter1.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        println!("  🔄 Event 1 callback executed");
    });
    
    let counter2 = Arc::clone(&callback_counter);
    let wait_id2 = async_waiter.wait_async(Arc::clone(&event2), move || {
        counter2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        println!("  🔄 Event 2 callback executed");
    });
    
    let counter3 = Arc::clone(&callback_counter);
    let wait_id3 = async_waiter.wait_async(Arc::clone(&event3), move || {
        counter3.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        println!("  🔄 Event 3 callback executed");
    });
    
    println!("✓ Set up async waiting for 3 events with callbacks");
    
    // Simulate events completing at different times
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(10));
        let _ = event1.synchronize(); // Complete event 1
        
        thread::sleep(Duration::from_millis(15));
        let _ = event2.synchronize(); // Complete event 2
        
        thread::sleep(Duration::from_millis(5));
        let _ = event3.synchronize(); // Complete event 3
    });
    
    // Cancel one wait to test cancellation
    thread::sleep(Duration::from_millis(5));
    let cancelled = async_waiter.cancel_wait(wait_id2);
    println!("✓ Cancelled wait for event 2: {}", cancelled);
    
    // Wait for callbacks to execute
    thread::sleep(Duration::from_millis(50));
    
    let final_count = callback_counter.load(std::sync::atomic::Ordering::Relaxed);
    println!("✓ Async event callbacks completed: {} executed", final_count);
    
    Ok(())
}

#[cfg(feature = "cuda")]
async fn demo_deadlock_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Deadlock Detection Demo ===");
    
    let event_pool = Arc::new(EventPool::new(15, 8)?);
    let coordinator = OperationCoordinator::new(Arc::clone(&event_pool));
    
    let stream1 = CudaStream::new()?;
    let stream2 = CudaStream::new()?;
    let stream3 = CudaStream::new()?;
    
    println!("✓ Created coordinator for deadlock detection demo");
    
    // Create operations with circular dependencies (deadlock scenario)
    let op_a = coordinator.register_operation(
        OperationType::Kernel,
        EventPriority::Normal,
        &stream1,
        vec![],
        "Operation A".to_string(),
    )?;
    
    let op_b = coordinator.register_operation(
        OperationType::Kernel,
        EventPriority::Normal,
        &stream2,
        vec![op_a],  // B depends on A
        "Operation B".to_string(),
    )?;
    
    let op_c = coordinator.register_operation(
        OperationType::Kernel,
        EventPriority::Normal,
        &stream3,
        vec![op_b],  // C depends on B
        "Operation C".to_string(),
    )?;
    
    // Now create circular dependency: A depends on C
    let op_a_circular = coordinator.register_operation(
        OperationType::Kernel,
        EventPriority::Normal,
        &stream1,
        vec![op_c],  // A depends on C, creating A → C → B → A cycle
        "Operation A (circular)".to_string(),
    )?;
    
    println!("✓ Created operations with potential circular dependencies:");
    println!("  • Op {} → Op {} → Op {} → Op {}", op_a, op_b, op_c, op_a_circular);
    
    // Detect deadlocks
    let deadlocks = coordinator.detect_deadlocks();
    
    if deadlocks.is_empty() {
        println!("✓ No deadlocks detected (dependency graph is acyclic)");
    } else {
        println!("⚠️  Detected {} deadlock cycle(s):", deadlocks.len());
        for (i, cycle) in deadlocks.iter().enumerate() {
            println!("  • Cycle {}: {:?}", i + 1, cycle);
        }
    }
    
    let metrics = coordinator.metrics();
    println!("✓ Deadlock detection metrics: {} detections performed", metrics.deadlock_detections);
    
    Ok(())
}

#[cfg(feature = "cuda")]
async fn demo_complex_coordination() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Complex Coordination Scenario Demo ===");
    
    let event_pool = Arc::new(EventPool::new(25, 15)?);
    let coordinator = OperationCoordinator::new(Arc::clone(&event_pool));
    let async_waiter = AsyncEventWaiter::new();
    
    // Create multiple streams for complex scenario
    let compute_streams: Vec<_> = (0..3)
        .map(|_| CudaStream::new().unwrap())
        .collect();
    
    let memory_stream = CudaStream::new()?;
    let reduction_stream = CudaStream::new()?;
    
    println!("✓ Created complex coordination scenario with 5 streams");
    
    // Phase 1: Data preparation
    let data_prep_ops: Vec<_> = (0..3).map(|i| {
        coordinator.register_operation(
            OperationType::MemoryTransfer,
            EventPriority::High,
            &memory_stream,
            vec![],
            format!("Data preparation {}", i + 1),
        ).unwrap()
    }).collect();
    
    println!("✓ Phase 1: Registered {} data preparation operations", data_prep_ops.len());
    
    // Phase 2: Parallel computation
    let compute_ops: Vec<_> = compute_streams.iter().enumerate().map(|(i, stream)| {
        coordinator.register_operation(
            OperationType::Kernel,
            EventPriority::Normal,
            stream,
            data_prep_ops.clone(),  // All compute ops depend on all data prep ops
            format!("Parallel computation {}", i + 1),
        ).unwrap()
    }).collect();
    
    println!("✓ Phase 2: Registered {} parallel computation operations", compute_ops.len());
    
    // Phase 3: Result aggregation
    let final_reduction = coordinator.register_operation(
        OperationType::AllReduce,
        EventPriority::Critical,
        &reduction_stream,
        compute_ops.clone(),  // Depends on all compute operations
        "Final result aggregation".to_string(),
    )?;
    
    println!("✓ Phase 3: Registered final aggregation operation");
    
    // Add async callbacks for monitoring
    let completion_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    
    for &op_id in &data_prep_ops {
        let counter = Arc::clone(&completion_counter);
        coordinator.add_completion_callback(op_id, move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            println!("  📊 Data preparation operation {} completed", op_id);
        });
    }
    
    for &op_id in &compute_ops {
        let counter = Arc::clone(&completion_counter);
        coordinator.add_completion_callback(op_id, move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            println!("  🧮 Computation operation {} completed", op_id);
        });
    }
    
    coordinator.add_completion_callback(final_reduction, || {
        println!("  🎯 Final aggregation completed - all operations done!");
    });
    
    // Execute the entire pipeline
    println!("✓ Executing complex coordination pipeline:");
    
    let start_time = Instant::now();
    
    // Phase 1: Execute data preparation
    for &op_id in &data_prep_ops {
        coordinator.begin_operation(op_id, &memory_stream)?;
        thread::sleep(Duration::from_millis(2)); // Simulate work
        coordinator.complete_operation(op_id)?;
    }
    
    // Phase 2: Execute parallel computations
    let compute_handles: Vec<_> = compute_ops.iter().zip(compute_streams.iter()).map(|(&op_id, stream)| {
        let coordinator = &coordinator;
        let stream = stream.clone();
        thread::spawn(move || {
            coordinator.begin_operation(op_id, &stream).unwrap();
            thread::sleep(Duration::from_millis(8)); // Simulate compute work
            coordinator.complete_operation(op_id).unwrap();
        })
    }).collect();
    
    // Wait for all parallel computations
    for handle in compute_handles {
        handle.join().unwrap();
    }
    
    // Phase 3: Execute final reduction
    coordinator.begin_operation(final_reduction, &reduction_stream)?;
    thread::sleep(Duration::from_millis(5)); // Simulate reduction work
    coordinator.complete_operation(final_reduction)?;
    
    let total_duration = start_time.elapsed();
    
    // Get final metrics
    let metrics = coordinator.metrics();
    let final_completions = completion_counter.load(std::sync::atomic::Ordering::Relaxed);
    
    println!("✓ Complex coordination scenario completed in {:?}", total_duration);
    println!("✓ Final metrics:");
    println!("  • Total operations: {}", metrics.total_operations);
    println!("  • Completed operations: {}", metrics.completed_operations);
    println!("  • Callback executions: {}", final_completions);
    
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This demo requires the 'cuda' feature to be enabled.");
    println!("Run with: cargo run --example cuda_event_coordination --features cuda");
}