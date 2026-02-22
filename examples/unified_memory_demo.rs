//! Unified Memory Demo for ToRSh CUDA Backend
//!
//! This example demonstrates the unified memory functionality implemented
//! for the CUDA backend, allowing seamless data sharing between CPU and GPU.

use std::sync::Arc;

#[cfg(feature = "cuda")]
use torsh_backend::cuda::{CudaBackend, CudaBackendConfig, CudaDevice, UnifiedBuffer, MemoryAdvice};

#[cfg(feature = "cuda")]
use torsh_core::DType;

#[cfg(feature = "cuda")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Unified Memory Demo ===\n");

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

    // Check unified memory support
    let supports_unified = backend.supports_unified_memory()?;
    println!("✓ Unified Memory Support: {}", supports_unified);

    if !supports_unified {
        println!("⚠️  Device does not support unified memory. Demo cannot continue.");
        return Ok(());
    }

    // Demonstrate unified memory allocation
    println!("\n=== Unified Memory Allocation ===");
    let mut buffer = backend.allocate_unified(1024)?;
    println!("✓ Allocated 1024 bytes of unified memory");
    println!("✓ Buffer size: {} bytes", buffer.size());

    // Create a UnifiedBuffer wrapper for easier operations
    let device = Arc::new(CudaDevice::new(0)?);
    let mut typed_buffer = UnifiedBuffer::<f32>::new(device, 256, DType::F32)?;
    println!("✓ Created typed unified buffer with 256 f32 elements");

    // Demonstrate data transfer and access
    println!("\n=== Data Operations ===");
    let test_data: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
    
    // Copy data from host to unified memory
    typed_buffer.copy_from_host(&test_data)?;
    println!("✓ Copied test data to unified memory");

    // Demonstrate memory prefetching
    println!("\n=== Memory Prefetching ===");
    typed_buffer.prefetch_to_device(Some(0))?;
    println!("✓ Prefetched data to GPU device 0");

    typed_buffer.prefetch_to_host()?;
    println!("✓ Prefetched data back to host");

    // Demonstrate memory advice for performance optimization
    println!("\n=== Memory Performance Hints ===");
    
    // Set data as read-mostly for better caching
    typed_buffer.set_memory_advice(MemoryAdvice::SetReadMostly, None)?;
    println!("✓ Set memory as read-mostly");

    // Set preferred location
    typed_buffer.set_preferred_location(0)?;
    println!("✓ Set preferred location to device 0");

    // Indicate device access pattern
    typed_buffer.set_accessed_by(0)?;
    println!("✓ Indicated device 0 will access this memory");

    // Verify data integrity
    println!("\n=== Data Verification ===");
    let mut result_data = vec![0.0f32; 256];
    typed_buffer.copy_to_host(&mut result_data)?;
    
    let data_matches = test_data.iter().zip(result_data.iter())
        .all(|(&a, &b)| (a - b).abs() < 1e-6);
    
    if data_matches {
        println!("✓ Data integrity verified - all values match!");
    } else {
        println!("✗ Data integrity check failed");
    }

    // Demonstrate CPU/GPU performance comparison
    println!("\n=== Performance Benefits ===");
    println!("💡 Unified Memory Benefits:");
    println!("   • Automatic data migration between CPU and GPU");
    println!("   • Reduced explicit memory copy operations");
    println!("   • Simplified memory management");
    println!("   • Optimal data placement with prefetching");
    println!("   • Performance hints for better caching");

    // Advanced features demonstration
    println!("\n=== Advanced Features ===");
    
    // Test different memory advice patterns
    let patterns = [
        (MemoryAdvice::SetPreferredLocation, "Preferred Location"),
        (MemoryAdvice::SetAccessedBy, "Device Access Pattern"),
        (MemoryAdvice::UnsetReadMostly, "Unset Read-Mostly"),
    ];

    for (advice, description) in patterns {
        typed_buffer.set_memory_advice(advice, Some(0))?;
        println!("✓ Applied memory advice: {}", description);
    }

    // Demonstrate large allocation for real workloads
    println!("\n=== Large Buffer Allocation ===");
    let large_buffer = backend.allocate_unified(64 * 1024 * 1024)?; // 64MB
    println!("✓ Allocated large buffer: {} MB", large_buffer.size() / (1024 * 1024));

    // Performance optimization tips
    println!("\n=== Usage Tips ===");
    println!("💡 Best Practices for Unified Memory:");
    println!("   1. Prefetch data before GPU kernels");
    println!("   2. Use memory advice for read-heavy patterns");
    println!("   3. Set preferred locations for frequently accessed data");
    println!("   4. Batch memory operations for better performance");
    println!("   5. Monitor memory usage with profiling tools");

    // Cleanup
    backend.deallocate_unified(buffer)?;
    backend.deallocate_unified(large_buffer)?;
    println!("\n✓ Cleaned up allocated memory");

    println!("\n=== Demo Complete ===");
    println!("🎉 Unified memory functionality demonstrated successfully!");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This demo requires the 'cuda' feature to be enabled.");
    println!("Run with: cargo run --example unified_memory_demo --features cuda");
}