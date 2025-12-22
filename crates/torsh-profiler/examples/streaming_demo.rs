//! Enhanced Real-time Streaming Demo
//!
//! This example demonstrates the advanced streaming capabilities of torsh-profiler
//! including adaptive bitrate streaming, compression, intelligent buffering, and
//! multi-protocol support.

use torsh_profiler::{
    create_high_performance_streaming_engine, create_low_latency_streaming_engine,
    create_streaming_engine, AdaptiveBitrateConfig, AdvancedFeatures, CompressionAlgorithm,
    CompressionConfig, ProfileEvent, ProtocolConfig, StreamingConfig, StreamingProtocol,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Enhanced Real-time Streaming Demo ===\n");

    // Demo 1: Basic Streaming Engine
    demo_basic_streaming().await?;

    // Demo 2: High-Performance Streaming
    demo_high_performance_streaming().await?;

    // Demo 3: Low-Latency Streaming
    demo_low_latency_streaming().await?;

    // Demo 4: Custom Configuration
    demo_custom_configuration().await?;

    // Demo 5: Adaptive Bitrate Streaming
    demo_adaptive_bitrate().await?;

    // Demo 6: Compression Features
    demo_compression_features().await?;

    // Demo 7: Multi-Protocol Streaming
    demo_multi_protocol().await?;

    // Demo 8: Advanced Features
    demo_advanced_features().await?;

    println!("\n=== All Streaming Demos Completed Successfully ===");
    Ok(())
}

/// Demo 1: Basic streaming engine with default configuration
async fn demo_basic_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Demo 1: Basic Streaming Engine ---");

    let engine = create_streaming_engine();

    // Add some sample events
    for i in 0..10 {
        let event = ProfileEvent {
            name: format!("operation_{}", i),
            category: "compute".to_string(),
            start_us: i * 1000,
            duration_us: 500 + (i * 50),
            thread_id: 1,
            operation_count: Some(1000),
            flops: Some(5000),
            bytes_transferred: Some(2048),
            stack_trace: None,
        };
        engine.add_event(event);
    }

    // Get statistics
    let stats = engine.get_stats();
    println!("  Total connections: {}", stats.total_connections);
    println!("  Active connections: {}", stats.active_connections);
    println!("  Total events sent: {}", stats.total_events_sent);
    println!(
        "  Compression ratio: {:.2}%",
        stats.compression_ratio * 100.0
    );

    println!("  ✓ Basic streaming engine created and tested\n");
    Ok(())
}

/// Demo 2: High-performance streaming with maximum throughput
async fn demo_high_performance_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Demo 2: High-Performance Streaming ---");

    let engine = create_high_performance_streaming_engine();

    println!("  Configuration:");
    println!(
        "    Max bitrate: {} events/sec",
        engine.config.adaptive_bitrate.max_bitrate
    );
    println!("    Buffer size: {}", engine.config.buffer_size);
    println!("    Compression level: {}", engine.config.compression.level);
    println!(
        "    Delta compression: {}",
        engine.config.advanced_features.delta_compression
    );

    // Simulate high-volume event stream
    for i in 0..100 {
        let event = ProfileEvent {
            name: format!("high_throughput_op_{}", i),
            category: "performance".to_string(),
            start_us: i * 100,
            duration_us: 50,
            thread_id: ((i % 4) + 1) as usize,
            operation_count: Some(10000),
            flops: Some(50000),
            bytes_transferred: Some(8192),
            stack_trace: None,
        };
        engine.add_event(event);
    }

    let stats = engine.get_stats();
    println!("  Performance metrics:");
    println!("    Events sent: {}", stats.total_events_sent);
    println!("    Bytes sent: {} bytes", stats.total_bytes_sent);
    println!("    Average latency: {} ms", stats.average_latency_ms);

    println!("  ✓ High-performance streaming demonstrated\n");
    Ok(())
}

/// Demo 3: Low-latency streaming optimized for minimal delay
async fn demo_low_latency_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Demo 3: Low-Latency Streaming ---");

    let engine = create_low_latency_streaming_engine();

    println!("  Configuration:");
    println!(
        "    Initial bitrate: {} events/sec",
        engine.config.adaptive_bitrate.initial_bitrate
    );
    println!("    Buffer size: {}", engine.config.buffer_size);
    println!(
        "    Compression enabled: {}",
        engine.config.compression.enabled
    );
    println!(
        "    Target latency: {} ms",
        engine.config.quality.metrics_threshold.latency_ms
    );

    // Simulate real-time critical events
    for i in 0..20 {
        let event = ProfileEvent {
            name: format!("real_time_op_{}", i),
            category: "critical".to_string(),
            start_us: i * 50,
            duration_us: 25,
            thread_id: 1,
            operation_count: Some(100),
            flops: Some(500),
            bytes_transferred: Some(512),
            stack_trace: None,
        };
        engine.add_event(event);
    }

    let stats = engine.get_stats();
    println!("  Latency metrics:");
    println!("    Average latency: {} ms", stats.average_latency_ms);
    println!("    Dropped events: {}", stats.dropped_events);

    println!("  ✓ Low-latency streaming demonstrated\n");
    Ok(())
}

/// Demo 4: Custom streaming configuration
async fn demo_custom_configuration() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Demo 4: Custom Configuration ---");

    let mut config = StreamingConfig::default();

    // Customize adaptive bitrate
    config.adaptive_bitrate = AdaptiveBitrateConfig {
        enabled: true,
        min_bitrate: 50,
        max_bitrate: 500,
        initial_bitrate: 200,
        adaptation_threshold: 0.15,
        adjustment_factor: 1.5,
    };

    // Customize compression
    config.compression = CompressionConfig {
        enabled: true,
        algorithm: CompressionAlgorithm::Zstd,
        level: 8,
        adaptive: true,
        threshold: 2048,
    };

    // Customize quality settings
    config.quality.auto_adjust = true;
    config.quality.metrics_threshold.latency_ms = 75;
    config.quality.metrics_threshold.packet_loss_percent = 0.5;

    let engine = torsh_profiler::streaming::EnhancedStreamingEngine::new(config);

    println!("  Custom configuration applied:");
    println!(
        "    Bitrate range: {} - {} events/sec",
        engine.config.adaptive_bitrate.min_bitrate, engine.config.adaptive_bitrate.max_bitrate
    );
    println!(
        "    Compression: {:?} (level {})",
        engine.config.compression.algorithm, engine.config.compression.level
    );
    println!(
        "    Auto quality adjustment: {}",
        engine.config.quality.auto_adjust
    );

    println!("  ✓ Custom configuration demonstrated\n");
    Ok(())
}

/// Demo 5: Adaptive bitrate streaming
async fn demo_adaptive_bitrate() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Demo 5: Adaptive Bitrate Streaming ---");

    let mut config = StreamingConfig::default();
    config.adaptive_bitrate.enabled = true;
    config.adaptive_bitrate.min_bitrate = 10;
    config.adaptive_bitrate.max_bitrate = 1000;
    config.adaptive_bitrate.initial_bitrate = 100;
    config.adaptive_bitrate.adaptation_threshold = 0.1;
    config.adaptive_bitrate.adjustment_factor = 1.2;

    let engine = torsh_profiler::streaming::EnhancedStreamingEngine::new(config);

    println!("  Adaptive bitrate settings:");
    println!("    Enabled: {}", engine.config.adaptive_bitrate.enabled);
    println!(
        "    Range: {} - {} events/sec",
        engine.config.adaptive_bitrate.min_bitrate, engine.config.adaptive_bitrate.max_bitrate
    );
    println!(
        "    Initial: {} events/sec",
        engine.config.adaptive_bitrate.initial_bitrate
    );
    println!(
        "    Adaptation threshold: {:.1}%",
        engine.config.adaptive_bitrate.adaptation_threshold * 100.0
    );
    println!(
        "    Adjustment factor: {:.1}x",
        engine.config.adaptive_bitrate.adjustment_factor
    );

    println!("  ✓ Adaptive bitrate streaming demonstrated\n");
    Ok(())
}

/// Demo 6: Compression features
async fn demo_compression_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Demo 6: Compression Features ---");

    // Test different compression algorithms
    let algorithms = vec![
        ("None", CompressionAlgorithm::None),
        ("Gzip", CompressionAlgorithm::Gzip),
        ("Zlib", CompressionAlgorithm::Zlib),
        ("Lz4", CompressionAlgorithm::Lz4),
        ("Zstd", CompressionAlgorithm::Zstd),
    ];

    for (name, algorithm) in algorithms {
        let mut config = StreamingConfig::default();
        config.compression = CompressionConfig {
            enabled: name != "None",
            algorithm: algorithm.clone(),
            level: 6,
            adaptive: true,
            threshold: 1024,
        };

        let engine = torsh_profiler::streaming::EnhancedStreamingEngine::new(config);

        println!("  {} compression:", name);
        println!("    Enabled: {}", engine.config.compression.enabled);
        println!("    Level: {}", engine.config.compression.level);
        println!("    Adaptive: {}", engine.config.compression.adaptive);
        println!(
            "    Threshold: {} bytes",
            engine.config.compression.threshold
        );
    }

    println!("  ✓ Compression features demonstrated\n");
    Ok(())
}

/// Demo 7: Multi-protocol streaming support
async fn demo_multi_protocol() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Demo 7: Multi-Protocol Streaming ---");

    let mut config = StreamingConfig::default();
    config.protocols = ProtocolConfig {
        websocket: true,
        sse: true,
        udp: false,
        tcp: true,
        priority: vec![
            StreamingProtocol::WebSocket,
            StreamingProtocol::Tcp,
            StreamingProtocol::ServerSentEvents,
            StreamingProtocol::Udp,
        ],
    };

    let engine = torsh_profiler::streaming::EnhancedStreamingEngine::new(config);

    println!("  Enabled protocols:");
    println!("    WebSocket: {}", engine.config.protocols.websocket);
    println!("    Server-Sent Events: {}", engine.config.protocols.sse);
    println!("    TCP: {}", engine.config.protocols.tcp);
    println!("    UDP: {}", engine.config.protocols.udp);

    println!("  Protocol priority order:");
    for (i, protocol) in engine.config.protocols.priority.iter().enumerate() {
        println!("    {}. {:?}", i + 1, protocol);
    }

    println!("  ✓ Multi-protocol streaming demonstrated\n");
    Ok(())
}

/// Demo 8: Advanced streaming features
async fn demo_advanced_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Demo 8: Advanced Features ---");

    let mut config = StreamingConfig::default();
    config.advanced_features = AdvancedFeatures {
        predictive_buffering: true,
        intelligent_sampling: true,
        deduplication: true,
        delta_compression: true,
        priority_streaming: true,
        load_balancing: true,
    };

    let engine = torsh_profiler::streaming::EnhancedStreamingEngine::new(config);

    println!("  Advanced features enabled:");
    println!(
        "    Predictive buffering: {}",
        engine.config.advanced_features.predictive_buffering
    );
    println!(
        "    Intelligent sampling: {}",
        engine.config.advanced_features.intelligent_sampling
    );
    println!(
        "    Data deduplication: {}",
        engine.config.advanced_features.deduplication
    );
    println!(
        "    Delta compression: {}",
        engine.config.advanced_features.delta_compression
    );
    println!(
        "    Priority streaming: {}",
        engine.config.advanced_features.priority_streaming
    );
    println!(
        "    Load balancing: {}",
        engine.config.advanced_features.load_balancing
    );

    // Test event priority calculation
    let events = vec![
        ("critical_memory_leak", "memory"),
        ("performance_analysis", "performance"),
        ("error_handler", "error"),
        ("debug_trace", "debug"),
        ("normal_operation", "compute"),
    ];

    println!("\n  Event priority classification:");
    for (name, category) in events {
        let event = ProfileEvent {
            name: name.to_string(),
            category: category.to_string(),
            start_us: 0,
            duration_us: 100,
            thread_id: 1,
            operation_count: None,
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        };

        // Event priority is calculated internally, demonstrating the intelligent classification
        engine.add_event(event);
        println!("    {}: {} category", name, category);
    }

    println!("  ✓ Advanced features demonstrated\n");
    Ok(())
}
