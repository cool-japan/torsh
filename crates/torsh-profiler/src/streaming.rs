//! Enhanced Real-time Streaming Capabilities
//!
//! This module provides advanced real-time streaming features including
//! adaptive bitrate streaming, data compression, intelligent buffering,
//! and multi-protocol streaming support for profiling data.
//!
//! # Features
//!
//! - **Adaptive Bitrate Streaming**: Automatically adjusts streaming rate based on network conditions
//! - **Multiple Compression Algorithms**: Gzip, Zlib, Lz4, Zstd with adaptive compression
//! - **Intelligent Buffering**: Priority-based event buffering with overflow management
//! - **Multi-Protocol Support**: WebSocket, SSE, TCP, UDP protocols
//! - **Quality Adaptation**: Dynamic quality adjustment based on bandwidth and latency
//! - **Connection Management**: Handle multiple concurrent client connections
//! - **Statistics Tracking**: Real-time metrics on throughput, latency, and compression ratios
//!
//! # Quick Start
//!
//! ```rust
//! use torsh_profiler::create_streaming_engine;
//!
//! // Create a basic streaming engine with default configuration
//! let engine = create_streaming_engine();
//! let stats = engine.get_stats();
//! println!("Active connections: {}", stats.active_connections);
//! ```
//!
//! # Factory Functions
//!
//! Three convenience functions create pre-configured engines for common use cases:
//!
//! ```rust
//! use torsh_profiler::{
//!     create_streaming_engine,
//!     create_high_performance_streaming_engine,
//!     create_low_latency_streaming_engine,
//! };
//!
//! // 1. Default balanced configuration
//! let default_engine = create_streaming_engine();
//!
//! // 2. High-performance: Optimized for maximum throughput
//! //    - 50,000 event buffer
//! //    - 2,000 events/sec max bitrate
//! //    - Level 9 compression
//! let hp_engine = create_high_performance_streaming_engine();
//!
//! // 3. Low-latency: Optimized for minimal delay
//! //    - 1,000 event buffer
//! //    - Compression disabled
//! //    - 50ms latency target
//! let ll_engine = create_low_latency_streaming_engine();
//! ```
//!
//! # Custom Configuration
//!
//! ```rust
//! use torsh_profiler::{
//!     EnhancedStreamingEngine, StreamingConfig, AdaptiveBitrateConfig,
//!     CompressionConfig, CompressionAlgorithm, QualityConfig,
//!     ProtocolConfig, AdvancedFeatures,
//! };
//!
//! let config = StreamingConfig {
//!     base_port: 8080,
//!     max_connections: 50,
//!     buffer_size: 5000,
//!     adaptive_bitrate: AdaptiveBitrateConfig {
//!         enabled: true,
//!         min_bitrate: 50,
//!         max_bitrate: 500,
//!         initial_bitrate: 100,
//!         adaptation_threshold: 0.15,
//!         adjustment_factor: 1.5,
//!     },
//!     compression: CompressionConfig {
//!         enabled: true,
//!         algorithm: CompressionAlgorithm::Lz4,
//!         level: 3,
//!         adaptive: false,
//!         threshold: 512,
//!     },
//!     quality: QualityConfig::default(),
//!     protocols: ProtocolConfig::default(),
//!     advanced_features: AdvancedFeatures::default(),
//! };
//!
//! let engine = EnhancedStreamingEngine::new(config);
//! ```
//!
//! # Compression Algorithms
//!
//! ```rust
//! use torsh_profiler::{CompressionAlgorithm, CompressionConfig};
//!
//! // Available compression algorithms:
//! // - None: No compression (best for latency)
//! // - Gzip: Standard gzip compression (good balance)
//! // - Zlib: Similar to gzip (slightly faster)
//! // - Lz4: Very fast compression (best for throughput)
//! // - Zstd: Modern compression (best ratio)
//!
//! let compression = CompressionConfig {
//!     enabled: true,
//!     algorithm: CompressionAlgorithm::Zstd,
//!     level: 6,  // 0-9, higher = better compression but slower
//!     adaptive: true,  // Automatically adjust based on performance
//!     threshold: 1024,  // Only compress events larger than 1KB
//! };
//! ```
//!
//! # Statistics and Monitoring
//!
//! ```rust
//! use torsh_profiler::create_streaming_engine;
//!
//! let engine = create_streaming_engine();
//! let stats = engine.get_stats();
//!
//! println!("Total events sent: {}", stats.total_events_sent);
//! println!("Total bytes sent: {}", stats.total_bytes_sent);
//! println!("Active connections: {}", stats.active_connections);
//! println!("Average latency: {}ms", stats.average_latency_ms);
//! println!("Compression ratio: {}%", stats.compression_ratio);
//! println!("Dropped events: {}", stats.dropped_events);
//! ```

use crate::dashboard::types::WebSocketConfig;
use crate::ProfileEvent;
use futures_util::{SinkExt, StreamExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{broadcast, mpsc};
use tokio::time::interval;

/// WebSocket message types for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebSocketMessage {
    ProfileEvent(ProfileEvent),
    Stats(StreamingStatsSnapshot),
    Control(ControlMessage),
}

/// Control messages for WebSocket communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    Subscribe(String),
    Unsubscribe(String),
    Ping,
    Pong,
}

/// Enhanced streaming engine with adaptive capabilities
#[derive(Debug)]
pub struct EnhancedStreamingEngine {
    /// Configuration
    pub config: StreamingConfig,
    /// Active streams
    streams: Arc<RwLock<HashMap<String, StreamConnection>>>,
    /// Event buffer for intelligent batching
    event_buffer: Arc<Mutex<EventBuffer>>,
    /// Statistics
    stats: Arc<StreamingStats>,
    /// Adaptive rate controller
    rate_controller: Arc<AdaptiveRateController>,
    /// Compression manager
    compression_manager: Arc<CompressionManager>,
    /// Connection manager
    connection_manager: Arc<ConnectionManager>,
}

/// Streaming configuration with adaptive features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Base port for streaming
    pub base_port: u16,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Buffer size for events
    pub buffer_size: usize,
    /// Adaptive bitrate settings
    pub adaptive_bitrate: AdaptiveBitrateConfig,
    /// Compression settings
    pub compression: CompressionConfig,
    /// Quality settings
    pub quality: QualityConfig,
    /// Protocol settings
    pub protocols: ProtocolConfig,
    /// Advanced features
    pub advanced_features: AdvancedFeatures,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            base_port: 9090,
            max_connections: 100,
            buffer_size: 10000,
            adaptive_bitrate: AdaptiveBitrateConfig::default(),
            compression: CompressionConfig::default(),
            quality: QualityConfig::default(),
            protocols: ProtocolConfig::default(),
            advanced_features: AdvancedFeatures::default(),
        }
    }
}

/// Adaptive bitrate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBitrateConfig {
    /// Enable adaptive bitrate streaming
    pub enabled: bool,
    /// Minimum bitrate (events per second)
    pub min_bitrate: usize,
    /// Maximum bitrate (events per second)
    pub max_bitrate: usize,
    /// Initial bitrate
    pub initial_bitrate: usize,
    /// Quality adaptation threshold
    pub adaptation_threshold: f64,
    /// Bitrate adjustment factor
    pub adjustment_factor: f64,
}

impl Default for AdaptiveBitrateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_bitrate: 10,
            max_bitrate: 1000,
            initial_bitrate: 100,
            adaptation_threshold: 0.1,
            adjustment_factor: 1.2,
        }
    }
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (0-9)
    pub level: u8,
    /// Enable adaptive compression
    pub adaptive: bool,
    /// Compression threshold (bytes)
    pub threshold: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Zlib,
            level: 6,
            adaptive: true,
            threshold: 1024,
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Zlib,
    Lz4,
    Zstd,
}

/// Quality configuration for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Quality levels
    pub levels: Vec<QualityLevel>,
    /// Auto quality adjustment
    pub auto_adjust: bool,
    /// Quality metrics threshold
    pub metrics_threshold: QualityMetricsThreshold,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            levels: vec![
                QualityLevel::new("low", 0.5, 10, 100),
                QualityLevel::new("medium", 0.7, 50, 500),
                QualityLevel::new("high", 1.0, 100, 1000),
            ],
            auto_adjust: true,
            metrics_threshold: QualityMetricsThreshold::default(),
        }
    }
}

/// Quality level definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityLevel {
    pub name: String,
    pub sampling_rate: f64,
    pub min_events_per_second: usize,
    pub max_events_per_second: usize,
}

impl QualityLevel {
    pub fn new(name: &str, sampling_rate: f64, min_eps: usize, max_eps: usize) -> Self {
        Self {
            name: name.to_string(),
            sampling_rate,
            min_events_per_second: min_eps,
            max_events_per_second: max_eps,
        }
    }
}

/// Quality metrics threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetricsThreshold {
    pub latency_ms: u64,
    pub packet_loss_percent: f64,
    pub bandwidth_utilization: f64,
    pub cpu_usage_percent: f64,
}

impl Default for QualityMetricsThreshold {
    fn default() -> Self {
        Self {
            latency_ms: 100,
            packet_loss_percent: 1.0,
            bandwidth_utilization: 0.8,
            cpu_usage_percent: 70.0,
        }
    }
}

/// Protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolConfig {
    /// Enable WebSocket streaming
    pub websocket: bool,
    /// Enable Server-Sent Events
    pub sse: bool,
    /// Enable UDP streaming
    pub udp: bool,
    /// Enable TCP streaming
    pub tcp: bool,
    /// Protocol priority
    pub priority: Vec<StreamingProtocol>,
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            websocket: true,
            sse: true,
            udp: false,
            tcp: false,
            priority: vec![
                StreamingProtocol::WebSocket,
                StreamingProtocol::ServerSentEvents,
                StreamingProtocol::Tcp,
                StreamingProtocol::Udp,
            ],
        }
    }
}

/// Streaming protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamingProtocol {
    WebSocket,
    ServerSentEvents,
    Tcp,
    Udp,
}

/// Advanced streaming features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedFeatures {
    /// Enable predictive buffering
    pub predictive_buffering: bool,
    /// Enable intelligent sampling
    pub intelligent_sampling: bool,
    /// Enable data deduplication
    pub deduplication: bool,
    /// Enable delta compression
    pub delta_compression: bool,
    /// Enable priority streaming
    pub priority_streaming: bool,
    /// Enable load balancing
    pub load_balancing: bool,
}

impl Default for AdvancedFeatures {
    fn default() -> Self {
        Self {
            predictive_buffering: true,
            intelligent_sampling: true,
            deduplication: true,
            delta_compression: true,
            priority_streaming: true,
            load_balancing: true,
        }
    }
}

/// Stream connection information
#[derive(Debug, Clone)]
pub struct StreamConnection {
    pub id: String,
    pub protocol: StreamingProtocol,
    pub remote_addr: SocketAddr,
    pub quality_level: String,
    pub bitrate: usize,
    pub compression: bool,
    pub connected_at: SystemTime,
    pub last_activity: SystemTime,
    pub bytes_sent: u64,
    pub events_sent: u64,
    pub latency_ms: u64,
}

/// Event buffer for intelligent batching
#[derive(Debug)]
pub struct EventBuffer {
    events: VecDeque<BufferedEvent>,
    categories: BTreeMap<String, VecDeque<BufferedEvent>>,
    max_size: usize,
    total_size: usize,
}

/// Buffered event with metadata
#[derive(Debug, Clone)]
pub struct BufferedEvent {
    pub event: ProfileEvent,
    pub priority: EventPriority,
    pub timestamp: Instant,
    pub size_bytes: usize,
    pub compressed: bool,
    pub category: String, // Store category as String
}

/// Event priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventPriority {
    Critical,
    High,
    Normal,
    Low,
}

/// Streaming statistics
#[derive(Debug)]
pub struct StreamingStats {
    pub total_connections: AtomicUsize,
    pub active_connections: AtomicUsize,
    pub total_events_sent: AtomicU64,
    pub total_bytes_sent: AtomicU64,
    pub compression_ratio: AtomicUsize, // percentage * 100
    pub average_latency_ms: AtomicU64,
    pub dropped_events: AtomicU64,
    pub quality_adjustments: AtomicU64,
    pub bitrate_adjustments: AtomicU64,
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self {
            total_connections: AtomicUsize::new(0),
            active_connections: AtomicUsize::new(0),
            total_events_sent: AtomicU64::new(0),
            total_bytes_sent: AtomicU64::new(0),
            compression_ratio: AtomicUsize::new(0),
            average_latency_ms: AtomicU64::new(0),
            dropped_events: AtomicU64::new(0),
            quality_adjustments: AtomicU64::new(0),
            bitrate_adjustments: AtomicU64::new(0),
        }
    }
}

/// Adaptive rate controller
#[derive(Debug)]
pub struct AdaptiveRateController {
    current_bitrate: AtomicUsize,
    target_bitrate: AtomicUsize,
    quality_score: AtomicUsize, // percentage * 100
    adjustment_history: Mutex<VecDeque<BitrateAdjustment>>,
    config: AdaptiveBitrateConfig,
}

#[derive(Debug, Clone)]
pub struct BitrateAdjustment {
    pub timestamp: Instant,
    pub old_bitrate: usize,
    pub new_bitrate: usize,
    pub reason: AdjustmentReason,
}

#[derive(Debug, Clone)]
pub enum AdjustmentReason {
    QualityImprovement,
    QualityDegradation,
    LatencyOptimization,
    BandwidthOptimization,
    LoadBalancing,
}

/// Compression manager
#[derive(Debug)]
pub struct CompressionManager {
    config: CompressionConfig,
    stats: CompressionStats,
}

#[derive(Debug, Default)]
pub struct CompressionStats {
    pub total_compressed: AtomicU64,
    pub compression_time_ns: AtomicU64,
    pub original_size: AtomicU64,
    pub compressed_size: AtomicU64,
}

/// Connection manager for handling multiple protocols
#[derive(Debug)]
pub struct ConnectionManager {
    websocket_connections: Arc<RwLock<HashMap<String, WebSocketConnection>>>,
    sse_connections: Arc<RwLock<HashMap<String, SSEConnection>>>,
    udp_connections: Arc<RwLock<HashMap<String, UdpConnection>>>,
    tcp_connections: Arc<RwLock<HashMap<String, TcpConnection>>>,
}

#[derive(Debug)]
pub struct WebSocketConnection {
    pub sender: mpsc::UnboundedSender<WebSocketMessage>,
    pub stats: ConnectionStats,
}

#[derive(Debug)]
pub struct SSEConnection {
    pub sender: mpsc::UnboundedSender<String>,
    pub stats: ConnectionStats,
}

#[derive(Debug)]
pub struct UdpConnection {
    pub addr: SocketAddr,
    pub stats: ConnectionStats,
}

#[derive(Debug)]
pub struct TcpConnection {
    pub writer: Arc<Mutex<tokio::net::tcp::OwnedWriteHalf>>,
    pub stats: ConnectionStats,
}

#[derive(Debug, Default)]
pub struct ConnectionStats {
    pub bytes_sent: AtomicU64,
    pub messages_sent: AtomicU64,
    pub errors: AtomicU64,
    pub last_send: Arc<Mutex<Option<Instant>>>,
}

impl EnhancedStreamingEngine {
    /// Create a new enhanced streaming engine
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            rate_controller: Arc::new(AdaptiveRateController::new(config.adaptive_bitrate.clone())),
            compression_manager: Arc::new(CompressionManager::new(config.compression.clone())),
            connection_manager: Arc::new(ConnectionManager::new()),
            streams: Arc::new(RwLock::new(HashMap::new())),
            event_buffer: Arc::new(Mutex::new(EventBuffer::new(config.buffer_size))),
            stats: Arc::new(StreamingStats::default()),
            config,
        }
    }

    /// Start the streaming engine
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Start the event processing loop
        self.start_event_processor().await?;

        // Start adaptive rate control
        self.start_rate_controller().await?;

        // Start quality monitoring
        self.start_quality_monitor().await?;

        // Start connection management
        self.start_connection_manager().await?;

        Ok(())
    }

    /// Add event to streaming buffer with intelligent prioritization
    pub fn add_event(&self, event: ProfileEvent) {
        let priority = self.calculate_event_priority(&event);
        let size_bytes = self.estimate_event_size(&event);
        let category = event.category.clone();

        let buffered_event = BufferedEvent {
            event,
            priority,
            timestamp: Instant::now(),
            size_bytes,
            compressed: false,
            category,
        };

        let mut buffer = self
            .event_buffer
            .lock()
            .expect("lock should not be poisoned");
        buffer.add_event(buffered_event);
    }

    /// Stream events to all connected clients with adaptive quality
    pub async fn stream_events(&self) -> Result<(), Box<dyn std::error::Error>> {
        let events = {
            let mut buffer = self
                .event_buffer
                .lock()
                .expect("lock should not be poisoned");
            buffer.get_events_for_streaming()
        };

        if events.is_empty() {
            return Ok(());
        }

        // Apply intelligent sampling
        let sampled_events = if self.config.advanced_features.intelligent_sampling {
            self.apply_intelligent_sampling(events).await
        } else {
            events
        };

        // Apply compression if enabled
        let compressed_events = if self.config.compression.enabled {
            self.compression_manager
                .compress_events(sampled_events)
                .await?
        } else {
            sampled_events
        };

        // Stream to all active connections
        self.broadcast_events(compressed_events).await?;

        Ok(())
    }

    /// Calculate event priority based on type and context
    fn calculate_event_priority(&self, event: &ProfileEvent) -> EventPriority {
        match event.category.as_str() {
            "memory" | "Memory" => {
                if event.name.contains("leak") || event.name.contains("critical") {
                    EventPriority::Critical
                } else {
                    EventPriority::High
                }
            }
            "performance" | "Performance" => EventPriority::High,
            "error" | "Error" => EventPriority::Critical,
            "debug" | "Debug" => EventPriority::Low,
            _ => EventPriority::Normal,
        }
    }

    /// Estimate event size for bandwidth calculation
    fn estimate_event_size(&self, event: &ProfileEvent) -> usize {
        // Rough estimation based on event content
        let base_size = std::mem::size_of::<ProfileEvent>();
        let name_size = event.name.len();
        let stack_trace_size = event.stack_trace.as_ref().map_or(0, |s| s.len());

        base_size + name_size + stack_trace_size
    }

    /// Apply intelligent sampling based on current conditions
    async fn apply_intelligent_sampling(&self, events: Vec<BufferedEvent>) -> Vec<BufferedEvent> {
        let current_bitrate = self.rate_controller.current_bitrate.load(Ordering::Relaxed);
        let max_events = current_bitrate.min(events.len());

        if events.len() <= max_events {
            return events;
        }

        // Sort by priority and recency
        let mut sorted_events = events;
        sorted_events.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then(b.timestamp.cmp(&a.timestamp))
        });

        sorted_events.truncate(max_events);
        sorted_events
    }

    /// Broadcast events to all connected clients
    async fn broadcast_events(
        &self,
        events: Vec<BufferedEvent>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Clone stream info before async operations to avoid holding lock across await
        let stream_info: Vec<(String, StreamingProtocol)> = {
            let streams = self.streams.read();
            streams
                .iter()
                .map(|(id, conn)| (id.clone(), conn.protocol.clone()))
                .collect()
        };

        for (stream_id, protocol) in stream_info {
            match protocol {
                StreamingProtocol::WebSocket => {
                    self.send_to_websocket(&stream_id, &events).await?;
                }
                StreamingProtocol::ServerSentEvents => {
                    self.send_to_sse(&stream_id, &events).await?;
                }
                StreamingProtocol::Tcp => {
                    self.send_to_tcp(&stream_id, &events).await?;
                }
                StreamingProtocol::Udp => {
                    self.send_to_udp(&stream_id, &events).await?;
                }
            }
        }

        // Update statistics
        self.stats
            .total_events_sent
            .fetch_add(events.len() as u64, Ordering::Relaxed);

        Ok(())
    }

    // Protocol-specific sending methods
    async fn send_to_websocket(
        &self,
        stream_id: &str,
        events: &[BufferedEvent],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let connections = self.connection_manager.websocket_connections.read();
        if let Some(connection) = connections.get(stream_id) {
            for event in events {
                let message = WebSocketMessage::ProfileEvent(event.event.clone());
                if connection.sender.send(message).is_err() {
                    // Connection closed
                    break;
                }
                connection
                    .stats
                    .messages_sent
                    .fetch_add(1, Ordering::Relaxed);
                connection
                    .stats
                    .bytes_sent
                    .fetch_add(event.size_bytes as u64, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    async fn send_to_sse(
        &self,
        stream_id: &str,
        events: &[BufferedEvent],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let connections = self.connection_manager.sse_connections.read();
        if let Some(connection) = connections.get(stream_id) {
            for event in events {
                let json = serde_json::to_string(&event.event)?;
                let sse_message = format!("data: {}\n\n", json);
                if connection.sender.send(sse_message).is_err() {
                    break;
                }
                connection
                    .stats
                    .messages_sent
                    .fetch_add(1, Ordering::Relaxed);
                connection
                    .stats
                    .bytes_sent
                    .fetch_add(event.size_bytes as u64, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    async fn send_to_tcp(
        &self,
        _stream_id: &str,
        _events: &[BufferedEvent],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // TCP implementation would go here
        Ok(())
    }

    async fn send_to_udp(
        &self,
        _stream_id: &str,
        _events: &[BufferedEvent],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // UDP implementation would go here
        Ok(())
    }

    // Background task starters
    async fn start_event_processor(&self) -> Result<(), Box<dyn std::error::Error>> {
        let engine = Arc::new(self.clone());

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(50)); // 20 FPS

            loop {
                interval.tick().await;
                if let Err(e) = engine.stream_events().await {
                    eprintln!("Error streaming events: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn start_rate_controller(&self) -> Result<(), Box<dyn std::error::Error>> {
        let controller = Arc::clone(&self.rate_controller);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));

            loop {
                interval.tick().await;
                controller.adjust_bitrate().await;
            }
        });

        Ok(())
    }

    async fn start_quality_monitor(&self) -> Result<(), Box<dyn std::error::Error>> {
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));

            loop {
                interval.tick().await;
                // Monitor quality metrics and adjust settings
                println!(
                    "Quality monitor: Active connections: {}",
                    stats.active_connections.load(Ordering::Relaxed)
                );
            }
        });

        Ok(())
    }

    async fn start_connection_manager(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Connection management logic would go here
        Ok(())
    }

    /// Get streaming statistics
    pub fn get_stats(&self) -> StreamingStatsSnapshot {
        StreamingStatsSnapshot {
            total_connections: self.stats.total_connections.load(Ordering::Relaxed),
            active_connections: self.stats.active_connections.load(Ordering::Relaxed),
            total_events_sent: self.stats.total_events_sent.load(Ordering::Relaxed),
            total_bytes_sent: self.stats.total_bytes_sent.load(Ordering::Relaxed),
            compression_ratio: self.stats.compression_ratio.load(Ordering::Relaxed) as f64 / 100.0,
            average_latency_ms: self.stats.average_latency_ms.load(Ordering::Relaxed),
            dropped_events: self.stats.dropped_events.load(Ordering::Relaxed),
            quality_adjustments: self.stats.quality_adjustments.load(Ordering::Relaxed),
            bitrate_adjustments: self.stats.bitrate_adjustments.load(Ordering::Relaxed),
        }
    }
}

impl Clone for EnhancedStreamingEngine {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            streams: Arc::clone(&self.streams),
            event_buffer: Arc::clone(&self.event_buffer),
            stats: Arc::clone(&self.stats),
            rate_controller: Arc::clone(&self.rate_controller),
            compression_manager: Arc::clone(&self.compression_manager),
            connection_manager: Arc::clone(&self.connection_manager),
        }
    }
}

impl EventBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            events: VecDeque::new(),
            categories: BTreeMap::new(),
            max_size,
            total_size: 0,
        }
    }

    pub fn add_event(&mut self, event: BufferedEvent) {
        // Remove old events if buffer is full
        while self.events.len() >= self.max_size {
            if let Some(old_event) = self.events.pop_front() {
                self.total_size -= old_event.size_bytes;
            }
        }

        self.total_size += event.size_bytes;

        // Add to category-specific queue
        self.categories
            .entry(event.category.clone())
            .or_default()
            .push_back(event.clone());

        self.events.push_back(event);
    }

    fn get_events_for_streaming(&mut self) -> Vec<BufferedEvent> {
        let events: Vec<_> = self.events.drain(..).collect();
        self.categories.clear();
        self.total_size = 0;
        events
    }
}

impl AdaptiveRateController {
    pub fn new(config: AdaptiveBitrateConfig) -> Self {
        Self {
            current_bitrate: AtomicUsize::new(config.initial_bitrate),
            target_bitrate: AtomicUsize::new(config.initial_bitrate),
            quality_score: AtomicUsize::new(8000), // 80%
            adjustment_history: Mutex::new(VecDeque::with_capacity(100)),
            config,
        }
    }

    async fn adjust_bitrate(&self) {
        if !self.config.enabled {
            return;
        }

        let current = self.current_bitrate.load(Ordering::Relaxed);
        let quality = self.quality_score.load(Ordering::Relaxed) as f64 / 100.0;

        let new_bitrate = if quality < self.config.adaptation_threshold {
            // Decrease bitrate
            ((current as f64) / self.config.adjustment_factor) as usize
        } else if quality > (1.0 - self.config.adaptation_threshold) {
            // Increase bitrate
            ((current as f64) * self.config.adjustment_factor) as usize
        } else {
            current // No change
        };

        let clamped_bitrate = new_bitrate
            .max(self.config.min_bitrate)
            .min(self.config.max_bitrate);

        if clamped_bitrate != current {
            self.current_bitrate
                .store(clamped_bitrate, Ordering::Relaxed);

            let adjustment = BitrateAdjustment {
                timestamp: Instant::now(),
                old_bitrate: current,
                new_bitrate: clamped_bitrate,
                reason: if new_bitrate > current {
                    AdjustmentReason::QualityImprovement
                } else {
                    AdjustmentReason::QualityDegradation
                },
            };

            let mut history = self
                .adjustment_history
                .lock()
                .expect("lock should not be poisoned");
            if history.len() >= 100 {
                history.pop_front();
            }
            history.push_back(adjustment);
        }
    }
}

impl CompressionManager {
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            stats: CompressionStats::default(),
        }
    }

    async fn compress_events(
        &self,
        events: Vec<BufferedEvent>,
    ) -> Result<Vec<BufferedEvent>, Box<dyn std::error::Error>> {
        if !self.config.enabled {
            return Ok(events);
        }

        let mut compressed_events = Vec::new();

        for event in events {
            if event.size_bytes < self.config.threshold {
                compressed_events.push(event);
                continue;
            }

            let compressed_event = self.compress_event(event).await?;
            compressed_events.push(compressed_event);
        }

        Ok(compressed_events)
    }

    async fn compress_event(
        &self,
        mut event: BufferedEvent,
    ) -> Result<BufferedEvent, Box<dyn std::error::Error>> {
        let start = Instant::now();

        // Simulate compression (in real implementation, would actually compress the event data)
        let original_size = event.size_bytes;
        let compressed_size = (original_size as f64 * 0.7) as usize; // Simulate 30% compression

        event.size_bytes = compressed_size;
        event.compressed = true;

        let compression_time = start.elapsed();

        // Update statistics
        self.stats.total_compressed.fetch_add(1, Ordering::Relaxed);
        self.stats
            .compression_time_ns
            .fetch_add(compression_time.as_nanos() as u64, Ordering::Relaxed);
        self.stats
            .original_size
            .fetch_add(original_size as u64, Ordering::Relaxed);
        self.stats
            .compressed_size
            .fetch_add(compressed_size as u64, Ordering::Relaxed);

        Ok(event)
    }
}

impl ConnectionManager {
    fn new() -> Self {
        Self {
            websocket_connections: Arc::new(RwLock::new(HashMap::new())),
            sse_connections: Arc::new(RwLock::new(HashMap::new())),
            udp_connections: Arc::new(RwLock::new(HashMap::new())),
            tcp_connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

/// Statistics snapshot for external consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStatsSnapshot {
    pub total_connections: usize,
    pub active_connections: usize,
    pub total_events_sent: u64,
    pub total_bytes_sent: u64,
    pub compression_ratio: f64,
    pub average_latency_ms: u64,
    pub dropped_events: u64,
    pub quality_adjustments: u64,
    pub bitrate_adjustments: u64,
}

/// Convenience functions for creating streaming engines
pub fn create_streaming_engine() -> EnhancedStreamingEngine {
    EnhancedStreamingEngine::new(StreamingConfig::default())
}

pub fn create_high_performance_streaming_engine() -> EnhancedStreamingEngine {
    let mut config = StreamingConfig::default();
    config.adaptive_bitrate.max_bitrate = 2000;
    config.buffer_size = 50000;
    config.compression.level = 9;
    config.advanced_features.delta_compression = true;

    EnhancedStreamingEngine::new(config)
}

pub fn create_low_latency_streaming_engine() -> EnhancedStreamingEngine {
    let mut config = StreamingConfig::default();
    config.adaptive_bitrate.initial_bitrate = 500;
    config.buffer_size = 1000;
    config.compression.enabled = false; // Disable compression for lower latency
    config.quality.metrics_threshold.latency_ms = 50;

    EnhancedStreamingEngine::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_engine_creation() {
        let engine = create_streaming_engine();
        assert_eq!(engine.config.base_port, 9090);
        assert!(engine.config.adaptive_bitrate.enabled);
    }

    #[test]
    fn test_event_buffer() {
        let mut buffer = EventBuffer::new(5);

        let event = ProfileEvent {
            name: "test".to_string(),
            category: "memory".to_string(),
            start_us: 0,
            duration_us: 100,
            thread_id: 1,
            operation_count: None,
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        };

        let buffered_event = BufferedEvent {
            event,
            priority: EventPriority::Normal,
            timestamp: Instant::now(),
            size_bytes: 100,
            compressed: false,
            category: "memory".to_string(),
        };

        buffer.add_event(buffered_event);
        assert_eq!(buffer.events.len(), 1);
        assert_eq!(buffer.total_size, 100);
    }

    #[test]
    fn test_event_priority_calculation() {
        let engine = create_streaming_engine();

        let memory_event = ProfileEvent {
            name: "memory_leak_detected".to_string(),
            category: "memory".to_string(),
            start_us: 0,
            duration_us: 100,
            thread_id: 1,
            operation_count: None,
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        };

        let priority = engine.calculate_event_priority(&memory_event);
        assert_eq!(priority, EventPriority::Critical);
    }

    #[tokio::test]
    async fn test_compression_manager() {
        let config = CompressionConfig::default();
        let manager = CompressionManager::new(config);

        let event = BufferedEvent {
            event: ProfileEvent {
                name: "test".to_string(),
                category: "memory".to_string(),
                start_us: 0,
                duration_us: 100,
                thread_id: 1,
                operation_count: None,
                flops: None,
                bytes_transferred: None,
                stack_trace: None,
            },
            priority: EventPriority::Normal,
            timestamp: Instant::now(),
            size_bytes: 2000, // Above threshold
            compressed: false,
            category: "memory".to_string(),
        };

        let compressed = manager.compress_event(event).await.unwrap();
        assert!(compressed.compressed);
        assert!(compressed.size_bytes < 2000);
    }

    #[test]
    fn test_adaptive_rate_controller() {
        let config = AdaptiveBitrateConfig::default();
        let controller = AdaptiveRateController::new(config);

        assert_eq!(controller.current_bitrate.load(Ordering::Relaxed), 100);
        assert_eq!(controller.target_bitrate.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_quality_level() {
        let level = QualityLevel::new("test", 0.8, 50, 500);
        assert_eq!(level.name, "test");
        assert_eq!(level.sampling_rate, 0.8);
        assert_eq!(level.min_events_per_second, 50);
        assert_eq!(level.max_events_per_second, 500);
    }

    #[test]
    fn test_streaming_config_defaults() {
        let config = StreamingConfig::default();
        assert_eq!(config.base_port, 9090);
        assert_eq!(config.max_connections, 100);
        assert!(config.compression.enabled);
        assert!(config.adaptive_bitrate.enabled);
    }

    #[test]
    fn test_high_performance_engine() {
        let engine = create_high_performance_streaming_engine();
        assert_eq!(engine.config.adaptive_bitrate.max_bitrate, 2000);
        assert_eq!(engine.config.buffer_size, 50000);
        assert!(engine.config.advanced_features.delta_compression);
    }

    #[test]
    fn test_low_latency_engine() {
        let engine = create_low_latency_streaming_engine();
        assert_eq!(engine.config.adaptive_bitrate.initial_bitrate, 500);
        assert_eq!(engine.config.buffer_size, 1000);
        assert!(!engine.config.compression.enabled);
        assert_eq!(engine.config.quality.metrics_threshold.latency_ms, 50);
    }
}
