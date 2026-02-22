//! Advanced CUDA Stream Management Demo
//!
//! This example demonstrates the advanced stream management capabilities 
//! implemented for the ToRSh CUDA backend, including:
//! - Priority-based stream allocation
//! - Workload-aware stream selection
//! - Multi-stream coordination and synchronization
//! - Stream-ordered memory allocation
//! - Performance profiling and metrics

use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use torsh_backend::cuda::{
    CudaBackend, CudaBackendConfig, CudaDevice, CudaStream, CudaEvent,
    stream_advanced::{
        AdvancedStreamPool, StreamOrderedAllocator, MultiStreamCoordinator, StreamProfiler,
        WorkloadType, AllocationStrategy, PoolMetrics,
    }
};

#[cfg(feature = "cuda")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Advanced Stream Management Demo ===\\n");

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

    // Demonstrate advanced stream pool
    demo_advanced_stream_pool().await?;
    
    // Demonstrate stream-ordered memory allocation
    demo_stream_ordered_allocation().await?;
    
    // Demonstrate multi-stream coordination
    demo_multi_stream_coordination().await?;
    
    // Demonstrate stream profiling
    demo_stream_profiling().await?;
    
    // Demonstrate workload-aware optimization
    demo_workload_optimization().await?;

    println!("\\n=== Demo Complete ===");
    println!("🎉 Advanced stream management functionality demonstrated successfully!");

    Ok(())
}

#[cfg(feature = "cuda")]
async fn demo_advanced_stream_pool() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Advanced Stream Pool Demo ===");
    
    // Create advanced stream pool with load-balanced allocation
    let pool = AdvancedStreamPool::new_with_strategy(8, AllocationStrategy::LoadBalanced)?;
    println!("✓ Created advanced stream pool with 8 streams (load-balanced strategy)");
    
    // Test different workload types
    let compute_stream = pool.get_stream_for_workload(WorkloadType::Compute);
    let memory_stream = pool.get_stream_for_workload(WorkloadType::Memory);
    let mixed_stream = pool.get_stream_for_workload(WorkloadType::Mixed);
    let coordination_stream = pool.get_stream_for_workload(WorkloadType::Coordination);
    
    println!("✓ Allocated streams for different workload types:");
    println!("  • Compute: Stream {} (Priority: {:?})", compute_stream.id(), compute_stream.priority());
    println!("  • Memory: Stream {} (Priority: {:?})", memory_stream.id(), memory_stream.priority());
    println!("  • Mixed: Stream {} (Priority: {:?})", mixed_stream.id(), mixed_stream.priority());
    println!("  • Coordination: Stream {} (Priority: {:?})", coordination_stream.id(), coordination_stream.priority());
    
    // Test priority-specific allocation
    let high_priority = pool.get_priority_stream(torsh_backend::cuda::StreamPriority::High);
    let normal_priority = pool.get_priority_stream(torsh_backend::cuda::StreamPriority::Normal);
    let low_priority = pool.get_priority_stream(torsh_backend::cuda::StreamPriority::Low);
    
    println!("✓ Allocated priority-specific streams:");
    println!("  • High Priority: Stream {}", high_priority.id());
    println!("  • Normal Priority: Stream {}", normal_priority.id());
    println!("  • Low Priority: Stream {}", low_priority.id());
    
    // Record workload completion times for optimization
    pool.record_workload_completion(WorkloadType::Compute, Duration::from_millis(15));
    pool.record_workload_completion(WorkloadType::Memory, Duration::from_millis(8));
    pool.record_workload_completion(WorkloadType::Mixed, Duration::from_millis(12));
    
    // Get pool metrics
    let metrics = pool.metrics();
    println!("✓ Pool metrics: {} total allocations", metrics.total_allocations);
    
    // Synchronize specific priority streams
    pool.synchronize_priority(torsh_backend::cuda::StreamPriority::High)?;
    println!("✓ Synchronized high-priority streams");
    
    Ok(())
}

#[cfg(feature = "cuda")]
async fn demo_stream_ordered_allocation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Stream-Ordered Memory Allocation Demo ===");
    
    let mut allocator = StreamOrderedAllocator::new();
    
    // Create streams for different tasks
    let stream1 = CudaStream::new()?;
    let stream2 = CudaStream::new()?;
    let stream3 = CudaStream::new()?;
    
    println!("✓ Created 3 streams for memory allocation demo");
    
    // Allocate memory for different streams
    let alloc1 = allocator.allocate_for_stream(&stream1, 1024 * 1024)?;  // 1MB
    let alloc2 = allocator.allocate_for_stream(&stream2, 2048 * 1024)?;  // 2MB
    let alloc3 = allocator.allocate_for_stream(&stream3, 512 * 1024)?;   // 512KB
    
    println!("✓ Allocated memory for streams:");
    println!("  • Stream {}: {} bytes", stream1.id(), alloc1.size());
    println!("  • Stream {}: {} bytes", stream2.id(), alloc2.size());
    println!("  • Stream {}: {} bytes", stream3.id(), alloc3.size());
    println!("  • Total allocated: {} bytes", allocator.total_allocated());
    
    // Add stream dependencies
    let event1 = Arc::new(CudaEvent::new()?);
    let event2 = Arc::new(CudaEvent::new()?);
    
    allocator.add_stream_dependency(&stream2, event1);
    allocator.add_stream_dependency(&stream3, event2);
    
    println!("✓ Added stream dependencies for coordination");
    
    // Check dependencies
    let deps_satisfied = allocator.dependencies_satisfied(&stream2)?;
    println!("✓ Stream 2 dependencies satisfied: {}", deps_satisfied);
    
    // Free memory for streams
    allocator.free_for_stream(&stream1)?;
    allocator.free_for_stream(&stream2)?;
    allocator.free_for_stream(&stream3)?;
    
    println!("✓ Freed memory for all streams");
    
    Ok(())
}

#[cfg(feature = "cuda")]
async fn demo_multi_stream_coordination() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Multi-Stream Coordination Demo ===");
    
    // Create streams for coordination
    let stream1 = Arc::new(CudaStream::new()?);
    let stream2 = Arc::new(CudaStream::new()?);
    let stream3 = Arc::new(CudaStream::new()?);
    let stream4 = Arc::new(CudaStream::new()?);
    
    let streams = vec![stream1.clone(), stream2.clone(), stream3.clone(), stream4.clone()];
    let mut coordinator = MultiStreamCoordinator::new(streams);
    
    println!("✓ Created multi-stream coordinator with 4 streams");
    
    // Add dependencies between streams
    coordinator.add_dependency(&stream2, &stream1)?;  // stream2 depends on stream1
    coordinator.add_dependency(&stream3, &stream1)?;  // stream3 depends on stream1
    coordinator.add_dependency(&stream4, &stream2)?;  // stream4 depends on stream2
    coordinator.add_dependency(&stream4, &stream3)?;  // stream4 depends on stream3
    
    println!("✓ Created dependency graph:");
    println!("  • Stream {} → Stream {}", stream1.id(), stream2.id());
    println!("  • Stream {} → Stream {}", stream1.id(), stream3.id());
    println!("  • Stream {} → Stream {}", stream2.id(), stream4.id());
    println!("  • Stream {} → Stream {}", stream3.id(), stream4.id());
    
    // Check for cycles (should be false for DAG)
    let has_cycles = coordinator.has_cycles();
    println!("✓ Dependency graph has cycles: {} (should be false)", has_cycles);
    
    // Add completion callbacks
    coordinator.add_completion_callback(&stream1, || {
        println!("  🔄 Stream 1 completion callback executed");
    });
    
    coordinator.add_completion_callback(&stream4, || {
        println!("  🔄 Stream 4 completion callback executed");
    });
    
    println!("✓ Added completion callbacks for streams 1 and 4");
    
    // Create barrier across all streams
    coordinator.create_barrier()?;
    println!("✓ Created synchronization barrier across all streams");
    
    // Execute parallel operations (simulation)
    let operation_count = std::sync::atomic::AtomicUsize::new(0);
    let counter = Arc::new(operation_count);
    
    coordinator.execute_parallel(move |stream| {
        // Simulate some work
        let counter = Arc::clone(&counter);
        let work_duration = match stream.priority() {
            torsh_backend::cuda::StreamPriority::High => Duration::from_millis(5),
            torsh_backend::cuda::StreamPriority::Normal => Duration::from_millis(10),
            torsh_backend::cuda::StreamPriority::Low => Duration::from_millis(15),
        };
        
        std::thread::sleep(work_duration);
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        stream.record_kernel_launch();
        
        Ok(())
    })?;
    
    println!("✓ Executed parallel operations across {} streams", counter.load(std::sync::atomic::Ordering::Relaxed));
    
    // Synchronize all streams
    coordinator.synchronize_all()?;
    println!("✓ Synchronized all coordinated streams");
    
    Ok(())
}

#[cfg(feature = "cuda")]
async fn demo_stream_profiling() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Stream Profiling Demo ===");
    
    let mut profiler = StreamProfiler::new();
    profiler.enable();
    
    println!("✓ Stream profiler enabled");
    
    // Create streams for profiling
    let stream1 = CudaStream::new()?;
    let stream2 = CudaStream::new()?;
    
    // Simulate operations and record them
    let start_time = Instant::now();
    
    // Simulate kernel launches
    profiler.record_operation(&stream1, "matrix_multiply", Duration::from_millis(12));
    profiler.record_kernel_launch(&stream1);
    
    profiler.record_operation(&stream1, "vector_add", Duration::from_millis(3));
    profiler.record_kernel_launch(&stream1);
    
    // Simulate memory transfers
    profiler.record_operation(&stream2, "host_to_device_copy", Duration::from_millis(8));
    profiler.record_memory_transfer(&stream2);
    
    profiler.record_operation(&stream2, "device_to_host_copy", Duration::from_millis(6));
    profiler.record_memory_transfer(&stream2);
    
    println!("✓ Recorded simulated operations:");
    println!("  • Stream {}: 2 kernel launches, 2 operations", stream1.id());
    println!("  • Stream {}: 2 memory transfers, 2 operations", stream2.id());
    
    // Get individual stream reports
    if let Some(report1) = profiler.get_stream_report(&stream1) {
        println!("✓ Stream {} report:", report1.stream_id);
        println!("  • Total time: {:?}", report1.total_time);
        println!("  • Operations: {}", report1.operation_count);
        println!("  • Kernel launches: {}", report1.kernel_launches);
        println!("  • Memory transfers: {}", report1.memory_transfers);
    }
    
    if let Some(report2) = profiler.get_stream_report(&stream2) {
        println!("✓ Stream {} report:", report2.stream_id);
        println!("  • Total time: {:?}", report2.total_time);
        println!("  • Operations: {}", report2.operation_count);
        println!("  • Kernel launches: {}", report2.kernel_launches);
        println!("  • Memory transfers: {}", report2.memory_transfers);
    }
    
    // Get comprehensive report
    let comprehensive = profiler.get_comprehensive_report();
    println!("✓ Comprehensive profiling report:");
    println!("  • Total streams profiled: {}", comprehensive.total_streams);
    println!("  • Profiling enabled: {}", comprehensive.profiling_enabled);
    println!("  • Individual stream reports: {}", comprehensive.streams.len());
    
    Ok(())
}

#[cfg(feature = "cuda")]
async fn demo_workload_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Workload Optimization Demo ===");
    
    // Create pool with workload-aware strategy
    let mut pool = AdvancedStreamPool::new_with_strategy(6, AllocationStrategy::Workload)?;
    println!("✓ Created workload-aware stream pool");
    
    // Simulate different workload patterns
    let workload_types = [
        WorkloadType::Compute,
        WorkloadType::Memory,
        WorkloadType::Mixed,
        WorkloadType::Coordination,
    ];
    
    println!("✓ Simulating workload patterns:");
    
    for workload in workload_types {
        let stream = pool.get_stream_for_workload(workload);
        
        // Simulate workload execution time based on type
        let execution_time = match workload {
            WorkloadType::Compute => Duration::from_millis(20),
            WorkloadType::Memory => Duration::from_millis(10),
            WorkloadType::Mixed => Duration::from_millis(15),
            WorkloadType::Coordination => Duration::from_millis(5),
        };
        
        // Record completion for optimization
        pool.record_workload_completion(workload, execution_time);
        
        println!("  • {:?} workload on Stream {} (Priority: {:?}) - {:?}", 
                 workload, stream.id(), stream.priority(), execution_time);
    }
    
    // Get average times for each workload
    for workload in workload_types {
        if let Some(avg_time) = pool.average_workload_time(workload) {
            println!("✓ Average {:?} workload time: {:?}", workload, avg_time);
        }
    }
    
    // Optimize pool configuration based on usage patterns
    pool.optimize_configuration()?;
    println!("✓ Optimized pool configuration based on workload patterns");
    
    // Test ready stream detection
    let has_ready = pool.has_ready_streams();
    println!("✓ Pool has ready streams: {}", has_ready);
    
    // Test waiting for ready stream with timeout
    if let Ok(Some(ready_stream)) = pool.wait_for_any_ready(Some(Duration::from_millis(100))) {
        println!("✓ Found ready stream: {} within timeout", ready_stream.id());
    } else {
        println!("✓ No streams became ready within timeout (expected for busy streams)");
    }
    
    // Final pool metrics
    let final_metrics = pool.metrics();
    println!("✓ Final pool metrics:");
    println!("  • Total allocations: {}", final_metrics.total_allocations);
    println!("  • Strategy effectiveness entries: {}", final_metrics.strategy_effectiveness.len());
    
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This demo requires the 'cuda' feature to be enabled.");
    println!("Run with: cargo run --example advanced_stream_management --features cuda");
}