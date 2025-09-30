//! Advanced profiling demo showcasing the new features

use std::{thread, time::Duration};
use torsh_profiler::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Profiling Demo - ToRSh Profiler");
    println!("===========================================\n");

    // Check if --realtime flag is passed
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--realtime" {
        // 5. Real-time Streaming Demo
        demo_realtime_streaming()?;
        return Ok(());
    }

    // 1. Overhead Measurement Demo
    demo_overhead_measurement()?;

    // 2. Custom Export Formats Demo
    demo_custom_export_formats()?;

    // 3. Memory Leak Detection Demo
    demo_memory_leak_detection()?;

    // 4. GPU Synchronization Tracking Demo
    demo_gpu_sync_tracking()?;

    println!("✅ All demos completed successfully!");
    println!("\n💡 Run with --realtime flag to test WebSocket streaming:");
    println!("   cargo run --example advanced_profiling_demo -- --realtime");
    Ok(())
}

fn demo_overhead_measurement() -> TorshResult<()> {
    println!("📊 1. Overhead Measurement Demo");
    println!("-------------------------------");

    // Enable overhead tracking
    set_global_overhead_tracking_enabled(true);
    set_global_stack_traces_enabled(true);

    start_profiling();

    // Simulate some work with profiling
    for i in 0..5 {
        let _scope = ScopeGuard::new(&format!("work_iteration_{i}"));
        thread::sleep(Duration::from_millis(10));

        // Add some events manually
        let profiler = global_profiler();
        let mut prof = profiler.lock();
        prof.add_event(ProfileEvent {
            name: format!("computation_{i}"),
            category: "compute".to_string(),
            start_us: 0,
            duration_us: 5000,
            thread_id: 1,
            operation_count: Some(1000),
            flops: Some(1000000),
            bytes_transferred: Some(1024),
            stack_trace: None,
        });
    }

    stop_profiling();

    // Check overhead statistics
    let overhead_stats = get_global_overhead_stats();
    println!("   Overhead Statistics:");
    println!("   - Add Event Count: {}", overhead_stats.add_event_count);
    println!(
        "   - Add Event Time: {:.2}μs",
        overhead_stats.add_event_time_ns as f64 / 1000.0
    );
    println!(
        "   - Stack Trace Count: {}",
        overhead_stats.stack_trace_count
    );
    println!(
        "   - Stack Trace Time: {:.2}μs",
        overhead_stats.stack_trace_time_ns as f64 / 1000.0
    );
    println!(
        "   - Total Overhead: {:.2}μs",
        overhead_stats.total_overhead_ns as f64 / 1000.0
    );

    // Export with overhead tracking
    export_global_json("/tmp/overhead_demo.json")?;
    println!("   📁 Exported profiling data to /tmp/overhead_demo.json");

    println!();
    Ok(())
}

fn demo_custom_export_formats() -> TorshResult<()> {
    println!("🎨 2. Custom Export Formats Demo");
    println!("-------------------------------");

    start_profiling();

    // Generate some sample data
    for i in 0..3 {
        let _scope = ScopeGuard::with_category(&format!("custom_work_{i}"), "demo");
        thread::sleep(Duration::from_millis(5));
    }

    stop_profiling();

    // Show available custom formats
    let formats = get_global_custom_export_formats();
    println!("   Available custom formats: {formats:?}");

    // Export using different custom formats
    export_global_custom("compact_json", "/tmp/compact_demo.json")?;
    println!("   📁 Exported compact JSON to /tmp/compact_demo.json");

    export_global_custom("performance_csv", "/tmp/performance_demo.csv")?;
    println!("   📁 Exported performance CSV to /tmp/performance_demo.csv");

    export_global_custom("simple_text", "/tmp/simple_demo.txt")?;
    println!("   📁 Exported simple text to /tmp/simple_demo.txt");

    // Register and use a custom format
    let custom_format = CustomExportFormat {
        name: "demo_format".to_string(),
        description: "Demo custom format".to_string(),
        file_extension: "demo".to_string(),
        schema: ExportSchema::Text {
            template: "⏱️  {name} took {duration_us}μs in category '{category}'".to_string(),
            separator: "\n".to_string(),
        },
    };

    register_global_custom_export_format(custom_format);
    export_global_custom("demo_format", "/tmp/custom_demo.demo")?;
    println!("   📁 Exported custom format to /tmp/custom_demo.demo");

    println!();
    Ok(())
}

fn demo_memory_leak_detection() -> TorshResult<()> {
    println!("🔍 3. Memory Leak Detection Demo");
    println!("-------------------------------");

    let mut memory_profiler = MemoryProfiler::new();
    memory_profiler.enable();
    memory_profiler.set_leak_detection_enabled(true);

    println!("   Simulating memory allocations...");

    // Simulate some allocations
    memory_profiler.record_allocation_with_trace(
        0x1000,
        1024,
        Some("allocation_site_1".to_string()),
    )?;
    memory_profiler.record_allocation_with_trace(
        0x2000,
        2048,
        Some("allocation_site_2".to_string()),
    )?;
    memory_profiler.record_allocation_with_trace(0x3000, 512, None)?;
    memory_profiler.record_allocation_with_trace(
        0x4000,
        4096,
        Some("large_allocation".to_string()),
    )?;

    // Deallocate some (simulating partial cleanup)
    memory_profiler.record_deallocation(0x2000)?;

    // Wait a bit to simulate time passing
    thread::sleep(Duration::from_millis(50));

    // Add one more recent allocation
    memory_profiler.record_allocation_with_trace(
        0x5000,
        256,
        Some("recent_allocation".to_string()),
    )?;

    // Detect all leaks
    let all_leaks = memory_profiler.detect_leaks()?;
    println!("   Total potential leaks: {}", all_leaks.leak_count);
    println!(
        "   Total leaked bytes: {} bytes",
        all_leaks.total_leaked_bytes
    );

    // Get largest leaks
    let largest_leaks = memory_profiler.get_largest_leaks(2)?;
    println!("   Largest {} leaks:", largest_leaks.leak_count);
    for (i, leak) in largest_leaks.potential_leaks.iter().enumerate() {
        println!(
            "     {}. Ptr: 0x{:x}, Size: {} bytes, Trace: {:?}",
            i + 1,
            leak.ptr,
            leak.size,
            leak.stack_trace
        );
    }

    // Get old leaks (older than 25ms)
    let old_leaks = memory_profiler.get_leaks_older_than(Duration::from_millis(25))?;
    println!("   Leaks older than 25ms: {}", old_leaks.leak_count);

    println!();
    Ok(())
}

fn demo_gpu_sync_tracking() -> TorshResult<()> {
    println!("🎮 4. GPU Synchronization Tracking Demo");
    println!("--------------------------------------");

    let mut cuda_profiler = CudaProfiler::new(0);
    cuda_profiler.enable()?;
    cuda_profiler.set_sync_tracking_enabled(true);

    println!("   Simulating CUDA operations...");

    // Simulate various CUDA operations
    cuda_profiler.record_kernel_launch(
        "matmul_kernel",
        (256, 256, 1),
        (16, 16, 1),
        2048,
        Duration::from_micros(150),
    )?;

    cuda_profiler.record_memory_copy("HtoD", 1024 * 1024, Duration::from_micros(50))?;
    cuda_profiler.record_device_sync(Duration::from_micros(100))?;
    cuda_profiler.record_stream_sync(1, Duration::from_micros(75))?;
    cuda_profiler.record_event_sync(42, Duration::from_micros(25))?;

    cuda_profiler.record_kernel_launch(
        "reduce_kernel",
        (128, 1, 1),
        (256, 1, 1),
        1024,
        Duration::from_micros(80),
    )?;

    cuda_profiler.record_memory_copy("DtoH", 512 * 1024, Duration::from_micros(30))?;

    // Get synchronization statistics
    let sync_stats = cuda_profiler.get_sync_stats();
    println!("   Synchronization Statistics:");
    println!(
        "   - Device syncs: {} (total: {}μs)",
        sync_stats.device_sync_count, sync_stats.device_sync_time_us
    );
    println!(
        "   - Stream syncs: {} (total: {}μs)",
        sync_stats.stream_sync_count, sync_stats.stream_sync_time_us
    );
    println!(
        "   - Event syncs: {} (total: {}μs)",
        sync_stats.event_sync_count, sync_stats.event_sync_time_us
    );
    println!("   - Total sync time: {}μs", sync_stats.total_sync_time_us);

    // Get recorded events
    let events = cuda_profiler.get_events()?;
    println!("   Total recorded events: {}", events.len());

    for event in events.iter().take(3) {
        println!("     - {}: {}μs", event.name, event.duration_us);
    }

    println!();
    Ok(())
}

fn demo_realtime_streaming() -> TorshResult<()> {
    use std::sync::Arc;

    println!("🔄 5. Real-time Streaming Demo");
    println!("-----------------------------");
    println!("This demo shows WebSocket-based live profiling data streaming");
    println!("with subscription management and 3D visualizations.\n");

    // Create profiler instances
    let profiler = Arc::new(Profiler::new());
    let memory_profiler = Arc::new(MemoryProfiler::new());

    // Configure enhanced dashboard with WebSocket streaming
    let websocket_config = WebSocketConfig {
        port: 8081,
        enabled: true,
        max_connections: 50,
        update_interval_ms: 100, // 10 updates per second
        buffer_size: 2048,
    };

    let dashboard_config = DashboardConfig {
        port: 8080,
        refresh_interval: 1, // Faster refresh for demo
        real_time_updates: true,
        max_data_points: 500,
        enable_stack_traces: false,
        custom_css: None,
        websocket_config,
    };

    // Create and start dashboard
    let dashboard = Arc::new(Dashboard::new(dashboard_config));
    dashboard.start(Arc::clone(&profiler), Arc::clone(&memory_profiler))?;

    println!("✅ Dashboard started at http://localhost:8080");
    println!("✅ WebSocket server started at ws://localhost:8081/ws");
    println!();
    println!("📡 WebSocket Commands (send via WebSocket client):");
    println!("   subscribe:dashboard_updates    - Subscribe to full dashboard updates");
    println!("   subscribe:performance_metrics  - Subscribe to performance metrics only");
    println!("   subscribe:memory_metrics       - Subscribe to memory metrics only");
    println!("   subscribe:visualizations       - Subscribe to 3D visualizations and heatmaps");
    println!("   subscribe:alerts               - Subscribe to real-time alerts");
    println!("   unsubscribe:<type>             - Unsubscribe from specific updates");
    println!("   get_subscriptions              - Get current subscriptions");
    println!("   ping                          - Test connection");
    println!();

    // Start simulation thread to generate profiling data
    let memory_profiler_sim = Arc::clone(&memory_profiler);
    let dashboard_sim = Arc::clone(&dashboard);

    thread::spawn(move || {
        let mut iteration = 0;

        loop {
            iteration += 1;

            // Simulate different types of operations
            for op_type in &[
                "matrix_multiply",
                "convolution",
                "activation",
                "backward_pass",
            ] {
                let start = std::time::Instant::now();

                // Simulate work with varying duration
                let work_duration = match *op_type {
                    "matrix_multiply" => Duration::from_millis(10 + (iteration % 50)),
                    "convolution" => Duration::from_millis(5 + (iteration % 30)),
                    "activation" => Duration::from_millis(1 + (iteration % 10)),
                    "backward_pass" => Duration::from_millis(15 + (iteration % 40)),
                    _ => Duration::from_millis(5),
                };

                thread::sleep(work_duration);

                let elapsed = start.elapsed();

                // Create profile event
                let event = ProfileEvent {
                    name: op_type.to_string(),
                    category: "neural_network".to_string(),
                    start_us: start.elapsed().as_micros() as u64,
                    duration_us: elapsed.as_micros() as u64,
                    thread_id: 1, // Use a fixed thread ID for demo
                    operation_count: Some(1),
                    flops: Some(match *op_type {
                        "matrix_multiply" => 1_000_000 + (iteration * 50_000),
                        "convolution" => 500_000 + (iteration * 25_000),
                        "activation" => 100_000 + (iteration * 5_000),
                        "backward_pass" => 750_000 + (iteration * 30_000),
                        _ => 50_000,
                    }),
                    bytes_transferred: Some(1024 * iteration),
                    stack_trace: None,
                };

                // Use global profiler function since Arc doesn't allow mutable access
                use torsh_profiler::add_event;
                add_event(
                    &event.name,
                    &event.category,
                    event.duration_us,
                    event.thread_id,
                );

                // Simulate memory allocation
                if iteration % 10 == 0 {
                    let ptr = 0x1000 + (iteration * 0x1000);
                    let size = 1024 * iteration;
                    let _ = memory_profiler_sim.record_allocation(ptr as usize, size as usize);
                }
            }

            // Vary iteration timing to create interesting patterns
            thread::sleep(Duration::from_millis(100 + (iteration % 200)));

            if iteration % 20 == 0 {
                println!(
                    "💫 Simulation: {} iterations completed, clients connected: {}",
                    iteration,
                    dashboard_sim
                        .get_websocket_stats()
                        .map(|s| s.connected_clients)
                        .unwrap_or(0)
                );
            }
        }
    });

    // Demonstrate real-time visualization broadcasting
    let dashboard_viz = Arc::clone(&dashboard);
    let profiler_viz = Arc::clone(&profiler);
    thread::spawn(move || {
        let mut counter = 0;
        loop {
            thread::sleep(Duration::from_secs(3));

            // Broadcast 3D landscape every 3 seconds
            let viz_config = VisualizationConfig {
                enable_3d_landscapes: true,
                enable_heatmaps: true,
                grid_resolution: 50,
                color_scheme: VisualizationColorScheme::Viridis,
                animation_speed: 1.5,
            };

            if let Ok(count) = dashboard_viz.broadcast_3d_landscape(&profiler_viz, &viz_config) {
                if count > 0 {
                    println!("📊 Broadcasted 3D landscape to {count} clients");
                }
            }

            // Broadcast heatmap every 6 seconds
            counter += 1;
            if counter % 2 == 0 {
                if let Ok(count) =
                    dashboard_viz.broadcast_heatmap(&profiler_viz, &viz_config, 20, 15)
                {
                    if count > 0 {
                        println!("🔥 Broadcasted heatmap to {count} clients");
                    }
                }
            }
        }
    });

    // Demonstrate alert broadcasting
    let dashboard_alerts = Arc::clone(&dashboard);
    thread::spawn(move || {
        let mut alert_counter = 0;
        loop {
            thread::sleep(Duration::from_secs(15));
            alert_counter += 1;

            let alert = DashboardAlert {
                id: format!("alert_{alert_counter}"),
                severity: if alert_counter % 3 == 0 {
                    DashboardAlertSeverity::Critical
                } else {
                    DashboardAlertSeverity::Warning
                },
                title: format!("Performance Alert #{alert_counter}"),
                message: format!(
                    "Simulated performance issue detected at {}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                ),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                resolved: false,
            };

            if let Ok(count) = dashboard_alerts.broadcast_alert(&alert) {
                if count > 0 {
                    println!("🚨 Broadcasted alert to {} clients: {}", count, alert.title);
                }
            }
        }
    });

    println!("🌐 Connect to WebSocket using tools like:");
    println!("   - Browser console: new WebSocket('ws://localhost:8081')");
    println!("   - wscat: wscat -c ws://localhost:8081");
    println!("   - websocat: websocat ws://localhost:8081");
    println!();
    println!("📱 Open browser to http://localhost:8080 to see the dashboard");
    println!();
    println!("⏹️  Press Ctrl+C to stop the demo...");

    // Keep the main thread alive and show statistics
    let mut stats_counter = 0;
    loop {
        thread::sleep(Duration::from_secs(5));
        stats_counter += 1;

        // Print connection statistics every 5 seconds
        if let Ok(stats) = dashboard.get_websocket_stats() {
            println!(
                "📊 Stats #{}: {} clients connected",
                stats_counter, stats.connected_clients
            );
        }
    }
}
