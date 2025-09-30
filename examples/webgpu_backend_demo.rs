//! WebGPU Backend Demo for ToRSh
//!
//! This example demonstrates the capabilities of the WebGPU backend,
//! including device management, buffer operations, compute shaders,
//! and tensor operations across different platforms (web, desktop, mobile).

use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "webgpu")]
use torsh_backend::webgpu::{
    WebGpuBackend, WebGpuBackendBuilder, WebGpuDevice, WebGpuBuffer, WebGpuKernelExecutor,
    WebGpuBackendConfig, AdapterInfo,
};

#[cfg(feature = "webgpu")]
use torsh_backend::{
    BufferDescriptor, BufferUsage, MemoryLocation, BufferHandle,
    BackendBuilder, BackendType,
};

#[cfg(feature = "webgpu")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh WebGPU Backend Demo ===\n");

    // Check WebGPU availability
    if !torsh_backend::webgpu::is_available() {
        println!("❌ WebGPU is not available on this system.");
        println!("Make sure you have:");
        println!("  - A compatible GPU driver");
        println!("  - WebGPU support in your browser (for web targets)");
        println!("  - Vulkan/DirectX 12/Metal support (for native targets)");
        return Ok(());
    }

    println!("✅ WebGPU is available!");

    // Demonstrate adapter enumeration
    demo_adapter_enumeration().await?;
    
    // Demonstrate backend creation and initialization
    demo_backend_initialization().await?;
    
    // Demonstrate device management
    demo_device_management().await?;
    
    // Demonstrate buffer operations
    demo_buffer_operations().await?;
    
    // Demonstrate compute operations
    demo_compute_operations().await?;
    
    // Demonstrate performance benchmarks
    demo_performance_benchmarks().await?;

    println!("\n=== Demo Complete ===");
    println!("🎉 WebGPU backend demonstration completed successfully!");
    println!("🌐 The WebGPU backend enables ToRSh to run on:");
    println!("   • Web browsers with WebGPU support");
    println!("   • Desktop systems with modern GPU drivers");
    println!("   • Mobile devices with compatible GPUs");
    println!("   • Cloud computing environments");

    Ok(())
}

#[cfg(feature = "webgpu")]
async fn demo_adapter_enumeration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== WebGPU Adapter Enumeration ===");
    
    let adapters = torsh_backend::webgpu::enumerate_adapters().await?;
    println!("Found {} WebGPU adapter(s):", adapters.len());
    
    for (i, adapter) in adapters.iter().enumerate() {
        let info = torsh_backend::webgpu::get_adapter_info(adapter);
        println!("  Adapter {}: {}", i, info.name);
        println!("    Type: {:?}", info.device_type);
        println!("    Backend: {:?}", info.backend);
        println!("    Vendor: {}", info.vendor);
        if !info.driver_info.is_empty() {
            println!("    Driver: {}", info.driver_info);
        }
        
        // Get adapter limits and features
        let limits = adapter.limits();
        let features = adapter.features();
        
        println!("    Limits:");
        println!("      Max buffer size: {} MB", limits.max_storage_buffer_binding_size / (1024 * 1024));
        println!("      Max workgroup size: {}x{}x{}", 
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z
        );
        println!("      Max workgroup invocations: {}", limits.max_compute_invocations_per_workgroup);
        
        println!("    Features: {:?}", features);
    }
    
    // Get the best adapter
    let best_adapter = torsh_backend::webgpu::get_best_adapter().await?;
    let best_info = torsh_backend::webgpu::get_adapter_info(&best_adapter);
    println!("\n✅ Selected best adapter: {} ({:?})", best_info.name, best_info.device_type);
    
    Ok(())
}

#[cfg(feature = "webgpu")]
async fn demo_backend_initialization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== WebGPU Backend Initialization ===");
    
    // Create backend with custom configuration
    let config = WebGpuBackendConfig {
        adapter_index: None, // Auto-select best adapter
        power_preference: wgpu::PowerPreference::HighPerformance,
        debug_mode: false,
        max_buffer_size: 1024 * 1024 * 1024, // 1GB
        enable_pipeline_cache: true,
        preferred_workgroup_size: (64, 1, 1),
    };
    
    let mut backend = WebGpuBackend::new(config);
    println!("✅ Created WebGPU backend with custom configuration");
    
    // Initialize the backend
    let start_time = Instant::now();
    backend.initialize().await?;
    let init_time = start_time.elapsed();
    println!("✅ Backend initialized in {:?}", init_time);
    
    // Test backend capabilities
    let capabilities = backend.capabilities();
    println!("\n📊 Backend Capabilities:");
    println!("  Max buffer size: {} MB", capabilities.max_buffer_size / (1024 * 1024));
    println!("  Max compute units: {}", capabilities.max_compute_units);
    println!("  Max workgroup size: {:?}", capabilities.max_workgroup_size);
    println!("  Supported data types: {:?}", capabilities.supported_dtypes);
    println!("  Async support: {}", capabilities.supports_async);
    println!("  Unified memory: {}", capabilities.supports_unified_memory);
    println!("  Sub-buffers: {}", capabilities.supports_sub_buffers);
    println!("  Kernel caching: {}", capabilities.supports_kernel_caching);
    println!("  Memory bandwidth: {:.1} GB/s", capabilities.memory_bandwidth_gbps);
    println!("  Compute throughput: {:.1} GFLOPS", capabilities.compute_throughput_gflops);
    
    // Test performance hints
    let hints = backend.performance_hints();
    println!("\n💡 Performance Hints:");
    println!("  Preferred workgroup size: {:?}", hints.preferred_workgroup_size);
    println!("  Memory alignment: {} bytes", hints.memory_alignment);
    println!("  Prefer vectorized: {}", hints.prefer_vectorized);
    println!("  Prefer async: {}", hints.prefer_async);
    println!("  Optimal batch size: {}", hints.optimal_batch_size);
    println!("  Cache kernels: {}", hints.cache_kernels);
    
    // Test available operations
    println!("\n🔧 Available Operations:");
    let ops = backend.available_ops();
    for (i, op) in ops.iter().enumerate() {
        if i % 4 == 0 { print!("  "); }
        print!("{:<20}", op);
        if (i + 1) % 4 == 0 { println!(); }
    }
    if ops.len() % 4 != 0 { println!(); }
    
    // Shutdown backend
    backend.shutdown().await?;
    println!("✅ Backend shutdown completed");
    
    Ok(())
}

#[cfg(feature = "webgpu")]
async fn demo_device_management() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== WebGPU Device Management ===");
    
    // Create device from best adapter
    let device = WebGpuDevice::from_best_adapter(0).await?;
    let device = Arc::new(device);
    
    println!("✅ Created WebGPU device: {}", device.name());
    println!("  Device ID: {}", device.id().index());
    println!("  Device type: {:?}", device.device_type());
    
    // Get device information
    let info = device.info();
    println!("  Vendor: {}", info.vendor);
    println!("  Driver version: {}", info.driver_version);
    println!("  Compute capability: {}", info.compute_capability);
    println!("  Memory total: {} MB", info.memory_total / (1024 * 1024));
    println!("  Memory free: {} MB", info.memory_free / (1024 * 1024));
    
    // Test device features
    println!("  Features: {:?}", info.features);
    
    // Get adapter information
    let adapter_info = device.adapter_info();
    println!("  Adapter: {} ({:?})", adapter_info.name, adapter_info.device_type);
    
    // Test device limits
    let limits = device.limits();
    println!("  Limits:");
    println!("    Max storage buffer: {} MB", limits.max_storage_buffer_binding_size / (1024 * 1024));
    println!("    Max workgroup storage: {} KB", limits.max_compute_workgroup_storage_size / 1024);
    println!("    Max workgroups per dimension: {}", limits.max_compute_workgroups_per_dimension);
    
    // Test optimal workgroup size calculation
    let optimal_1d = device.optimal_workgroup_size(1024);
    let optimal_2d = device.optimal_workgroup_size(256);
    println!("  Optimal workgroup sizes:");
    println!("    1D (1024 elements): {:?}", optimal_1d);
    println!("    2D (256 elements): {:?}", optimal_2d);
    
    // Test memory tracking
    let memory_usage = device.memory_usage();
    println!("  Memory usage:");
    println!("    Allocated: {} bytes", memory_usage.allocated_bytes);
    println!("    Peak allocated: {} bytes", memory_usage.peak_allocated_bytes);
    println!("    Allocations: {}", memory_usage.allocation_count);
    println!("    Deallocations: {}", memory_usage.deallocation_count);
    
    println!("✅ Device management demo completed");
    Ok(())
}

#[cfg(feature = "webgpu")]
async fn demo_buffer_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== WebGPU Buffer Operations ===");
    
    let device = Arc::new(WebGpuDevice::from_best_adapter(0).await?);
    
    // Create different types of buffers
    let storage_descriptor = BufferDescriptor {
        name: "storage_buffer".to_string(),
        size: 4096,
        usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        memory_location: MemoryLocation::Device,
    };
    
    let uniform_descriptor = BufferDescriptor {
        name: "uniform_buffer".to_string(),
        size: 256,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        memory_location: MemoryLocation::Device,
    };
    
    let staging_descriptor = BufferDescriptor {
        name: "staging_buffer".to_string(),
        size: 1024,
        usage: BufferUsage::MAP_READ | BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        memory_location: MemoryLocation::HostVisible,
    };
    
    let storage_buffer = WebGpuBuffer::new(
        Arc::clone(&device), 
        storage_descriptor, 
        BufferHandle::new(1)
    )?;
    
    let uniform_buffer = WebGpuBuffer::new(
        Arc::clone(&device), 
        uniform_descriptor, 
        BufferHandle::new(2)
    )?;
    
    let staging_buffer = WebGpuBuffer::new(
        Arc::clone(&device), 
        staging_descriptor, 
        BufferHandle::new(3)
    )?;
    
    println!("✅ Created buffers:");
    println!("  Storage buffer: {} bytes", storage_buffer.size());
    println!("  Uniform buffer: {} bytes", uniform_buffer.size());
    println!("  Staging buffer: {} bytes", staging_buffer.size());
    
    // Test buffer with initial data
    let test_data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let data_descriptor = BufferDescriptor {
        name: "data_buffer".to_string(),
        size: (test_data.len() * 4) as u64,
        usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        memory_location: MemoryLocation::Device,
    };
    
    let data_buffer = WebGpuBuffer::with_data(
        Arc::clone(&device),
        data_descriptor,
        BufferHandle::new(4),
        &test_data,
    )?;
    
    println!("✅ Created buffer with initial data: {} elements", test_data.len());
    
    // Test buffer mapping and data transfer
    staging_buffer.map_write(0, Some(256)).await?;
    {
        let mut mapped = staging_buffer.mapped_range_mut(0, Some(256))?;
        for (i, byte) in mapped.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
    }
    staging_buffer.unmap();
    println!("✅ Wrote test pattern to staging buffer");
    
    // Test buffer copy operations
    let mut encoder = device.create_command_encoder(Some("Buffer Copy Test"));
    
    // Copy staging to storage
    storage_buffer.copy_from_buffer(
        &mut encoder,
        &staging_buffer,
        0,
        0,
        256,
    )?;
    
    let command_buffer = encoder.finish();
    device.submit([command_buffer]);
    device.wait_for_completion().await?;
    
    println!("✅ Copied data between buffers");
    
    // Test write/read operations
    let write_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    staging_buffer.write_data(0, &write_data).await?;
    println!("✅ Wrote data to buffer: {:?}", write_data);
    
    let read_data: Vec<f32> = staging_buffer.read_data(0, write_data.len()).await?;
    println!("✅ Read data from buffer: {:?}", read_data);
    
    // Verify data integrity
    let data_matches = write_data.iter().zip(read_data.iter())
        .all(|(a, b)| (a - b).abs() < f32::EPSILON);
    
    if data_matches {
        println!("✅ Data integrity verified - read data matches written data");
    } else {
        println!("❌ Data integrity check failed");
    }
    
    // Test memory tracking after buffer operations
    let final_memory_usage = device.memory_usage();
    println!("📊 Final memory usage:");
    println!("  Allocated: {} bytes", final_memory_usage.allocated_bytes);
    println!("  Peak: {} bytes", final_memory_usage.peak_allocated_bytes);
    println!("  Allocations: {}", final_memory_usage.allocation_count);
    
    println!("✅ Buffer operations demo completed");
    Ok(())
}

#[cfg(feature = "webgpu")]
async fn demo_compute_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== WebGPU Compute Operations ===");
    
    let device = Arc::new(WebGpuDevice::from_best_adapter(0).await?);
    let executor = WebGpuKernelExecutor::new(Arc::clone(&device));
    
    // Create test buffers
    let size = 1024u64;
    let buffer_size = size * 4; // f32 elements
    
    let input_a_descriptor = BufferDescriptor {
        name: "input_a".to_string(),
        size: buffer_size,
        usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
        memory_location: MemoryLocation::Device,
    };
    
    let input_b_descriptor = BufferDescriptor {
        name: "input_b".to_string(),
        size: buffer_size,
        usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
        memory_location: MemoryLocation::Device,
    };
    
    let output_descriptor = BufferDescriptor {
        name: "output".to_string(),
        size: buffer_size,
        usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::MAP_READ,
        memory_location: MemoryLocation::Device,
    };
    
    // Create buffers with test data
    let test_data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let test_data_b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
    
    let input_a = WebGpuBuffer::with_data(
        Arc::clone(&device),
        input_a_descriptor,
        BufferHandle::new(1),
        &test_data_a,
    )?;
    
    let input_b = WebGpuBuffer::with_data(
        Arc::clone(&device),
        input_b_descriptor,
        BufferHandle::new(2),
        &test_data_b,
    )?;
    
    let output = WebGpuBuffer::new(
        Arc::clone(&device),
        output_descriptor,
        BufferHandle::new(3),
    )?;
    
    println!("✅ Created compute buffers with {} elements each", size);
    
    // Test elementwise addition
    println!("\n🧮 Testing elementwise addition...");
    let start_time = Instant::now();
    executor.elementwise_add(&input_a, &input_b, &output).await?;
    executor.synchronize().await?;
    let add_time = start_time.elapsed();
    println!("✅ Elementwise addition completed in {:?}", add_time);
    
    // Verify results (first few elements)
    output.map_read(0, Some(16)).await?; // Read first 4 f32 values
    {
        let mapped = output.mapped_range(0, Some(16))?;
        let results: &[f32] = bytemuck::cast_slice(&mapped);
        println!("  Results (first 4): {:?}", &results[..4.min(results.len())]);
        println!("  Expected: {:?}", &[0.0 + 1.0, 1.0 + 2.0, 2.0 + 3.0, 3.0 + 4.0]);
    }
    output.unmap();
    
    // Test elementwise multiplication
    println!("\n🧮 Testing elementwise multiplication...");
    let start_time = Instant::now();
    executor.elementwise_mul(&input_a, &input_b, &output).await?;
    executor.synchronize().await?;
    let mul_time = start_time.elapsed();
    println!("✅ Elementwise multiplication completed in {:?}", mul_time);
    
    // Test ReLU activation
    println!("\n🧮 Testing ReLU activation...");
    let start_time = Instant::now();
    executor.relu(&input_a, &output).await?;
    executor.synchronize().await?;
    let relu_time = start_time.elapsed();
    println!("✅ ReLU activation completed in {:?}", relu_time);
    
    // Test pipeline cache statistics
    let pipeline_stats = executor.pipeline_stats();
    println!("\n📊 Pipeline Cache Statistics:");
    println!("  Cached pipelines: {}", pipeline_stats.pipeline_count);
    println!("  Cached shaders: {}", pipeline_stats.shader_count);
    println!("  Shader bytes: {} KB", pipeline_stats.shader_bytes / 1024);
    println!("  Total memory usage: {} KB", pipeline_stats.total_memory_usage / 1024);
    
    // Performance summary
    println!("\n📈 Performance Summary:");
    println!("  Elementwise add: {:.2} GFLOPS", size as f64 / add_time.as_secs_f64() / 1e9);
    println!("  Elementwise mul: {:.2} GFLOPS", size as f64 / mul_time.as_secs_f64() / 1e9);
    println!("  ReLU: {:.2} GFLOPS", size as f64 / relu_time.as_secs_f64() / 1e9);
    
    println!("✅ Compute operations demo completed");
    Ok(())
}

#[cfg(feature = "webgpu")]
async fn demo_performance_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== WebGPU Performance Benchmarks ===");
    
    let device = Arc::new(WebGpuDevice::from_best_adapter(0).await?);
    let executor = WebGpuKernelExecutor::new(Arc::clone(&device));
    
    // Test different problem sizes
    let sizes = [1024, 4096, 16384, 65536];
    
    println!("📊 Benchmarking elementwise operations across different sizes:");
    println!("{:<10} {:<15} {:<15} {:<15}", "Size", "Add (ms)", "Mul (ms)", "ReLU (ms)");
    println!("{}", "-".repeat(60));
    
    for &size in &sizes {
        let buffer_size = (size * 4) as u64; // f32 elements
        
        // Create test buffers
        let test_data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let test_data_b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        
        let input_a = WebGpuBuffer::with_data(
            Arc::clone(&device),
            BufferDescriptor {
                name: format!("bench_input_a_{}", size),
                size: buffer_size,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                memory_location: MemoryLocation::Device,
            },
            BufferHandle::new(1),
            &test_data_a,
        )?;
        
        let input_b = WebGpuBuffer::with_data(
            Arc::clone(&device),
            BufferDescriptor {
                name: format!("bench_input_b_{}", size),
                size: buffer_size,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                memory_location: MemoryLocation::Device,
            },
            BufferHandle::new(2),
            &test_data_b,
        )?;
        
        let output = WebGpuBuffer::new(
            Arc::clone(&device),
            BufferDescriptor {
                name: format!("bench_output_{}", size),
                size: buffer_size,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
                memory_location: MemoryLocation::Device,
            },
            BufferHandle::new(3),
        )?;
        
        // Warmup
        executor.elementwise_add(&input_a, &input_b, &output).await?;
        executor.synchronize().await?;
        
        // Benchmark addition
        let start = Instant::now();
        for _ in 0..10 {
            executor.elementwise_add(&input_a, &input_b, &output).await?;
        }
        executor.synchronize().await?;
        let add_time = start.elapsed().as_millis() as f64 / 10.0;
        
        // Benchmark multiplication
        let start = Instant::now();
        for _ in 0..10 {
            executor.elementwise_mul(&input_a, &input_b, &output).await?;
        }
        executor.synchronize().await?;
        let mul_time = start.elapsed().as_millis() as f64 / 10.0;
        
        // Benchmark ReLU
        let start = Instant::now();
        for _ in 0..10 {
            executor.relu(&input_a, &output).await?;
        }
        executor.synchronize().await?;
        let relu_time = start.elapsed().as_millis() as f64 / 10.0;
        
        println!("{:<10} {:<15.2} {:<15.2} {:<15.2}", 
            size, add_time, mul_time, relu_time);
    }
    
    // Memory bandwidth test
    println!("\n📊 Memory Bandwidth Test:");
    let size = 1024 * 1024; // 1M elements
    let buffer_size = (size * 4) as u64;
    
    let src_buffer = WebGpuBuffer::new(
        Arc::clone(&device),
        BufferDescriptor {
            name: "bandwidth_src".to_string(),
            size: buffer_size,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory_location: MemoryLocation::Device,
        },
        BufferHandle::new(1),
    )?;
    
    let dst_buffer = WebGpuBuffer::new(
        Arc::clone(&device),
        BufferDescriptor {
            name: "bandwidth_dst".to_string(),
            size: buffer_size,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
            memory_location: MemoryLocation::Device,
        },
        BufferHandle::new(2),
    )?;
    
    let start = Instant::now();
    let mut encoder = device.create_command_encoder(Some("Bandwidth Test"));
    dst_buffer.copy_from_buffer(&mut encoder, &src_buffer, 0, 0, buffer_size)?;
    let command_buffer = encoder.finish();
    device.submit([command_buffer]);
    device.wait_for_completion().await?;
    let copy_time = start.elapsed();
    
    let bandwidth_gbps = (buffer_size as f64 * 2.0) / copy_time.as_secs_f64() / 1e9; // Read + Write
    println!("  Buffer size: {} MB", buffer_size / (1024 * 1024));
    println!("  Copy time: {:?}", copy_time);
    println!("  Bandwidth: {:.2} GB/s", bandwidth_gbps);
    
    println!("✅ Performance benchmarks completed");
    Ok(())
}

#[cfg(not(feature = "webgpu"))]
fn main() {
    println!("This demo requires the 'webgpu' feature to be enabled.");
    println!("Run with: cargo run --example webgpu_backend_demo --features webgpu");
}