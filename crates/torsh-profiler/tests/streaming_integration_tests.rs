//! Comprehensive integration tests for the streaming module
//!
//! These tests verify the streaming configuration and initialization

use torsh_profiler::{
    create_high_performance_streaming_engine, create_low_latency_streaming_engine,
    create_streaming_engine, AdaptiveBitrateConfig, AdvancedFeatures, BufferedEvent,
    CompressionAlgorithm, CompressionConfig, EnhancedStreamingEngine, EventPriority, ProfileEvent,
    ProtocolConfig, QualityConfig, QualityLevel, QualityMetricsThreshold, StreamingConfig,
};

/// Helper function to create a test profile event
fn create_test_event(name: &str, duration_us: u64, category: &str) -> ProfileEvent {
    ProfileEvent {
        name: name.to_string(),
        category: category.to_string(),
        start_us: 0,
        duration_us,
        thread_id: 1,
        operation_count: Some(100),
        flops: Some(1000),
        bytes_transferred: Some(4096),
        stack_trace: None,
    }
}

#[test]
fn test_streaming_config_defaults() {
    let config = StreamingConfig::default();

    // Verify default configuration values
    assert_eq!(config.base_port, 9090);
    assert_eq!(config.max_connections, 100);
    assert_eq!(config.buffer_size, 10000);
    assert!(config.adaptive_bitrate.enabled);
    assert!(config.compression.enabled);
    assert!(config.quality.auto_adjust);
}

#[test]
fn test_adaptive_bitrate_config_defaults() {
    let config = AdaptiveBitrateConfig::default();

    // Verify adaptive bitrate defaults
    assert_eq!(config.min_bitrate, 10);
    assert_eq!(config.max_bitrate, 1000);
    assert_eq!(config.initial_bitrate, 100);
    assert!(config.enabled);
    assert!(config.adaptation_threshold > 0.0);
    assert!(config.adjustment_factor > 1.0);
}

#[test]
fn test_compression_config_defaults() {
    let config = CompressionConfig::default();

    // Verify compression defaults
    assert!(config.enabled);
    assert!(matches!(config.algorithm, CompressionAlgorithm::Zlib));
    assert_eq!(config.level, 6);
    assert!(config.adaptive);
    assert_eq!(config.threshold, 1024);
}

#[test]
fn test_quality_config_defaults() {
    let config = QualityConfig::default();

    // Verify quality configuration defaults
    assert!(!config.levels.is_empty());
    assert!(config.auto_adjust);
    assert_eq!(config.levels.len(), 3); // low, medium, high
}

#[test]
fn test_basic_streaming_engine_creation() {
    let engine = create_streaming_engine();

    // Verify engine was created successfully
    assert_eq!(engine.config.base_port, 9090);
    assert_eq!(engine.config.buffer_size, 10000);
    assert!(engine.config.adaptive_bitrate.enabled);
}

#[test]
fn test_high_performance_streaming_engine_creation() {
    let engine = create_high_performance_streaming_engine();

    // Verify high-performance configuration
    assert_eq!(engine.config.buffer_size, 50000); // Larger buffer for throughput
    assert_eq!(engine.config.adaptive_bitrate.max_bitrate, 2000); // Higher max bitrate
}

#[test]
fn test_low_latency_streaming_engine_creation() {
    let engine = create_low_latency_streaming_engine();

    // Verify low-latency configuration
    assert_eq!(engine.config.buffer_size, 1000); // Smaller buffer for lower latency
    assert_eq!(engine.config.adaptive_bitrate.initial_bitrate, 500); // Moderate initial bitrate
}

#[test]
fn test_custom_streaming_config() {
    let config = StreamingConfig {
        base_port: 8080,
        max_connections: 50,
        buffer_size: 5000,
        adaptive_bitrate: AdaptiveBitrateConfig {
            enabled: true,
            min_bitrate: 50,
            max_bitrate: 500,
            initial_bitrate: 100,
            adaptation_threshold: 0.15,
            adjustment_factor: 1.5,
        },
        compression: CompressionConfig {
            enabled: true,
            algorithm: CompressionAlgorithm::Lz4,
            level: 3,
            adaptive: false,
            threshold: 512,
        },
        quality: QualityConfig {
            levels: vec![QualityLevel::new("custom", 0.8, 100, 400)],
            auto_adjust: false,
            metrics_threshold: QualityMetricsThreshold::default(),
        },
        protocols: ProtocolConfig::default(),
        advanced_features: AdvancedFeatures::default(),
    };

    let engine = EnhancedStreamingEngine::new(config.clone());

    // Verify custom configuration
    assert_eq!(engine.config.base_port, 8080);
    assert_eq!(engine.config.max_connections, 50);
    assert_eq!(engine.config.buffer_size, 5000);
    assert_eq!(engine.config.adaptive_bitrate.min_bitrate, 50);
    assert_eq!(engine.config.adaptive_bitrate.max_bitrate, 500);
    assert!(matches!(
        engine.config.compression.algorithm,
        CompressionAlgorithm::Lz4
    ));
    assert_eq!(engine.config.compression.level, 3);
    assert!(!engine.config.compression.adaptive);
}

#[test]
fn test_compression_algorithms() {
    let algorithms = vec![
        CompressionAlgorithm::None,
        CompressionAlgorithm::Gzip,
        CompressionAlgorithm::Zlib,
        CompressionAlgorithm::Lz4,
        CompressionAlgorithm::Zstd,
    ];

    for algorithm in algorithms {
        let mut config = StreamingConfig::default();
        config.compression.algorithm = algorithm.clone();

        let engine = EnhancedStreamingEngine::new(config);

        // Verify algorithm was set correctly
        match (&engine.config.compression.algorithm, &algorithm) {
            (CompressionAlgorithm::None, CompressionAlgorithm::None) => {}
            (CompressionAlgorithm::Gzip, CompressionAlgorithm::Gzip) => {}
            (CompressionAlgorithm::Zlib, CompressionAlgorithm::Zlib) => {}
            (CompressionAlgorithm::Lz4, CompressionAlgorithm::Lz4) => {}
            (CompressionAlgorithm::Zstd, CompressionAlgorithm::Zstd) => {}
            _ => panic!("Algorithm mismatch"),
        }
    }
}

#[test]
fn test_quality_levels() {
    let levels = vec![
        QualityLevel::new("ultra_low", 0.1, 10, 50),
        QualityLevel::new("low", 0.3, 50, 200),
        QualityLevel::new("medium", 0.5, 100, 500),
        QualityLevel::new("high", 0.8, 500, 2000),
        QualityLevel::new("ultra_high", 1.0, 1000, 5000),
    ];

    for level in &levels {
        // Verify quality level parameters
        assert!(level.sampling_rate > 0.0 && level.sampling_rate <= 1.0);
        assert!(level.min_events_per_second < level.max_events_per_second);
        assert!(!level.name.is_empty());
    }

    // Create engine with custom quality levels
    let mut config = StreamingConfig::default();
    config.quality.levels = levels;

    let engine = EnhancedStreamingEngine::new(config);
    assert_eq!(engine.config.quality.levels.len(), 5);
}

#[test]
fn test_buffered_event_creation() {
    let event = create_test_event("test_op", 1000, "cpu");
    let buffered = BufferedEvent {
        event: event.clone(),
        priority: EventPriority::Normal,
        timestamp: std::time::Instant::now(),
        size_bytes: 512,
        compressed: false,
        category: "cpu".to_string(),
    };

    // Verify buffered event properties
    assert_eq!(buffered.event.name, "test_op");
    assert!(matches!(buffered.priority, EventPriority::Normal));
    assert!(!buffered.compressed);
    assert_eq!(buffered.size_bytes, 512);
    assert_eq!(buffered.category, "cpu");
}

#[test]
fn test_event_priority_levels() {
    let priorities = vec![
        EventPriority::Critical,
        EventPriority::High,
        EventPriority::Normal,
        EventPriority::Low,
    ];

    // Verify priority ordering (enum declared as: Critical, High, Normal, Low)
    // In Rust, earlier enum variants are "less than" later ones
    assert!(EventPriority::Critical < EventPriority::High);
    assert!(EventPriority::High < EventPriority::Normal);
    assert!(EventPriority::Normal < EventPriority::Low);

    // Create events with different priorities
    for priority in priorities {
        let event = create_test_event("test", 100, "test");
        let buffered = BufferedEvent {
            event,
            priority: priority.clone(),
            timestamp: std::time::Instant::now(),
            size_bytes: 100,
            compressed: false,
            category: "test".to_string(),
        };

        match buffered.priority {
            EventPriority::Critical => assert!(matches!(priority, EventPriority::Critical)),
            EventPriority::High => assert!(matches!(priority, EventPriority::High)),
            EventPriority::Normal => assert!(matches!(priority, EventPriority::Normal)),
            EventPriority::Low => assert!(matches!(priority, EventPriority::Low)),
        }
    }
}

#[test]
fn test_streaming_stats_initialization() {
    let engine = create_streaming_engine();
    let stats = engine.get_stats();

    // Verify initial statistics
    assert_eq!(stats.total_events_sent, 0);
    assert_eq!(stats.total_bytes_sent, 0);
    assert_eq!(stats.active_connections, 0);
    assert_eq!(stats.dropped_events, 0);
}

#[test]
fn test_buffer_size_configurations() {
    let buffer_sizes = vec![100, 1000, 5000, 10000, 20000];

    for size in buffer_sizes {
        let mut config = StreamingConfig::default();
        config.buffer_size = size;

        let engine = EnhancedStreamingEngine::new(config);

        // Verify buffer size was configured correctly
        assert_eq!(engine.config.buffer_size, size);
    }
}

#[test]
fn test_compression_threshold() {
    let thresholds = vec![128, 256, 512, 1024, 2048];

    for threshold in thresholds {
        let mut config = StreamingConfig::default();
        config.compression.threshold = threshold;

        let engine = EnhancedStreamingEngine::new(config);

        // Verify compression threshold
        assert_eq!(engine.config.compression.threshold, threshold);
    }
}

#[test]
fn test_adaptive_bitrate_range() {
    let mut config = StreamingConfig::default();
    config.adaptive_bitrate.min_bitrate = 100;
    config.adaptive_bitrate.max_bitrate = 2000;
    config.adaptive_bitrate.initial_bitrate = 500;

    let engine = EnhancedStreamingEngine::new(config);

    // Verify bitrate range
    assert_eq!(engine.config.adaptive_bitrate.min_bitrate, 100);
    assert_eq!(engine.config.adaptive_bitrate.max_bitrate, 2000);
    assert_eq!(engine.config.adaptive_bitrate.initial_bitrate, 500);

    // Verify initial bitrate is within range
    assert!(
        engine.config.adaptive_bitrate.initial_bitrate
            >= engine.config.adaptive_bitrate.min_bitrate
    );
    assert!(
        engine.config.adaptive_bitrate.initial_bitrate
            <= engine.config.adaptive_bitrate.max_bitrate
    );
}

#[test]
fn test_profile_event_in_buffered_context() {
    let event = ProfileEvent {
        name: "matrix_multiply".to_string(),
        category: "gpu".to_string(),
        start_us: 1000,
        duration_us: 5000,
        thread_id: 2,
        operation_count: Some(1000000),
        flops: Some(2000000000),
        bytes_transferred: Some(8388608), // 8 MB
        stack_trace: Some("stack trace here".to_string()),
    };

    let buffered = BufferedEvent {
        event: event.clone(),
        priority: EventPriority::High,
        timestamp: std::time::Instant::now(),
        size_bytes: 2048,
        compressed: true,
        category: "gpu".to_string(),
    };

    // Verify event data preservation
    assert_eq!(buffered.event.name, "matrix_multiply");
    assert_eq!(buffered.event.duration_us, 5000);
    assert_eq!(buffered.event.operation_count, Some(1000000));
    assert_eq!(buffered.event.flops, Some(2000000000));
    assert!(buffered.compressed);
}

#[test]
fn test_multiple_streaming_engines() {
    let engine1 = create_streaming_engine();
    let engine2 = create_high_performance_streaming_engine();
    let engine3 = create_low_latency_streaming_engine();

    // Verify each engine has different configurations
    assert_ne!(engine1.config.buffer_size, engine2.config.buffer_size);
    assert_ne!(engine1.config.buffer_size, engine3.config.buffer_size);
    assert_ne!(engine2.config.buffer_size, engine3.config.buffer_size);

    // Verify bitrate configurations differ
    assert_ne!(
        engine1.config.adaptive_bitrate.max_bitrate,
        engine2.config.adaptive_bitrate.max_bitrate
    );
}
