//! Advanced Visualization Demo - 3D Performance Landscapes and Heatmaps
//!
//! This example demonstrates the new advanced visualization features of the ToRSh profiler,
//! including 3D performance landscapes and interactive heatmaps.

use std::{sync::Arc, thread, time::Duration};
use torsh_profiler::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Advanced Visualization Demo - ToRSh Profiler");
    println!("===============================================\n");

    // Create profiler
    let profiler = Arc::new(Profiler::new());

    println!("📊 Generating sample profiling data...");

    // Simulate multi-threaded operations with varying performance characteristics
    let _profiler_clone = profiler.clone();
    let handle1 = thread::spawn(move || {
        for i in 0..20 {
            let _scope =
                ProfileScope::simple(format!("fast_operation_{i}"), "performance".to_string());
            thread::sleep(Duration::from_millis(1 + i % 5)); // Fast operations
        }
    });

    let _profiler_clone = profiler.clone();
    let handle2 = thread::spawn(move || {
        for i in 0..15 {
            let _scope =
                ProfileScope::simple(format!("medium_operation_{i}"), "performance".to_string());
            thread::sleep(Duration::from_millis(5 + i % 10)); // Medium operations
        }
    });

    let _profiler_clone = profiler.clone();
    let handle3 = thread::spawn(move || {
        for i in 0..10 {
            let _scope =
                ProfileScope::simple(format!("slow_operation_{i}"), "performance".to_string());
            thread::sleep(Duration::from_millis(10 + i % 15)); // Slow operations
        }
    });

    // Wait for all threads to complete
    handle1.join().unwrap();
    handle2.join().unwrap();
    handle3.join().unwrap();

    println!("✅ Generated profiling data from 3 threads with different performance patterns\n");

    // Configure visualization
    let viz_config = VisualizationConfig {
        enable_3d_landscapes: true,
        enable_heatmaps: true,
        grid_resolution: 50,
        color_scheme: VisualizationColorScheme::Thermal,
        animation_speed: 1.0,
    };

    println!("🏔️  Generating 3D Performance Landscape...");
    match generate_3d_landscape(&profiler, Some(viz_config.clone())) {
        Ok(landscape_json) => {
            println!("✅ 3D Landscape generated successfully!");
            println!(
                "   Data points: {}",
                landscape_json.matches("\"x\":").count()
            );

            // Save to file
            std::fs::write("performance_landscape_3d.json", landscape_json)?;
            println!("   Saved to: performance_landscape_3d.json");
        }
        Err(e) => println!("❌ Failed to generate 3D landscape: {e}"),
    }

    println!("\n🔥 Generating Performance Heatmap...");
    match generate_performance_heatmap(&profiler, 50, 30, Some(viz_config.clone())) {
        Ok(heatmap_json) => {
            println!("✅ Heatmap generated successfully!");
            println!("   Grid dimensions: 50x30");
            println!(
                "   Heat cells: {}",
                heatmap_json.matches("\"row\":").count()
            );

            // Save to file
            std::fs::write("performance_heatmap.json", heatmap_json)?;
            println!("   Saved to: performance_heatmap.json");
        }
        Err(e) => println!("❌ Failed to generate heatmap: {e}"),
    }

    // Test different color schemes
    println!("\n🌈 Testing different color schemes...");

    let color_schemes = vec![
        ("Thermal", VisualizationColorScheme::Thermal),
        ("Viridis", VisualizationColorScheme::Viridis),
        ("Plasma", VisualizationColorScheme::Plasma),
        (
            "Custom",
            VisualizationColorScheme::Custom {
                start: [0, 100, 200],
                end: [255, 50, 0],
            },
        ),
    ];

    for (name, scheme) in color_schemes {
        let config = VisualizationConfig {
            color_scheme: scheme,
            ..viz_config.clone()
        };

        match generate_performance_heatmap(&profiler, 20, 15, Some(config)) {
            Ok(_) => println!("   ✅ {name} color scheme"),
            Err(e) => println!("   ❌ {name} color scheme failed: {e}"),
        }
    }

    // Create enhanced dashboard with visualization features
    println!("\n🖥️  Creating enhanced dashboard with visualizations...");
    let dashboard_config = DashboardConfig {
        port: 8080,
        refresh_interval: 5,
        real_time_updates: true,
        max_data_points: 500,
        enable_stack_traces: true,
        custom_css: Some("
            .viz-3d-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 20px; }
            .viz-heatmap-container { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 12px; padding: 20px; }
            .realtime-indicator { width: 12px; height: 12px; background: #00ff00; border-radius: 50%; animation: pulse 1s infinite; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
        ".to_string()),
        websocket_config: WebSocketConfig {
            enabled: true,
            port: 8081,
            max_connections: 50,
            update_interval_ms: 500, // 2 updates per second for smooth visualization
            buffer_size: 2048,
        },
    };

    let _dashboard = create_dashboard_with_config(dashboard_config);
    println!("✅ Enhanced dashboard created with visualization support");
    println!("   Dashboard URL: http://localhost:8080");
    println!("   WebSocket endpoint: ws://localhost:8081");

    // Export dashboard with embedded visualization data
    let memory_profiler = MemoryProfiler::new();
    match export_dashboard_html(&profiler, &memory_profiler, "enhanced_dashboard.html") {
        Ok(()) => {
            println!("✅ Enhanced dashboard exported to: enhanced_dashboard.html");
            println!("   Includes embedded 3D visualization and heatmap data");
        }
        Err(e) => println!("❌ Failed to export dashboard: {e}"),
    }

    println!("\n🎉 Advanced Visualization Demo Complete!");
    println!("Files generated:");
    println!("  - performance_landscape_3d.json (3D landscape data)");
    println!("  - performance_heatmap.json (heatmap data)");
    println!("  - enhanced_dashboard.html (interactive dashboard)");
    println!("\nVisualization Features Demonstrated:");
    println!("  ✅ 3D Performance Landscapes with time, thread, and performance axes");
    println!("  ✅ Advanced Heatmaps with operation vs time analysis");
    println!("  ✅ Multiple color schemes (Thermal, Viridis, Plasma, Custom)");
    println!("  ✅ Real-time WebSocket streaming for live visualization updates");
    println!("  ✅ JSON export for integration with visualization libraries");

    Ok(())
}
