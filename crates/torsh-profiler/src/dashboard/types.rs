//! Dashboard types and configuration structures
//!
//! This module contains all the data types, configuration structures, and enums
//! used throughout the ToRSh performance dashboard system.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::SystemTime;

// =============================================================================
// Configuration Types
// =============================================================================

/// Configuration for the dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Port to serve the dashboard on
    pub port: u16,
    /// Refresh interval in seconds
    pub refresh_interval: u64,
    /// Whether to enable real-time updates
    pub real_time_updates: bool,
    /// Maximum number of data points to keep in memory
    pub max_data_points: usize,
    /// Whether to enable detailed stack traces
    pub enable_stack_traces: bool,
    /// Custom CSS styling
    pub custom_css: Option<String>,
    /// WebSocket configuration
    pub websocket_config: WebSocketConfig,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            refresh_interval: 5,
            real_time_updates: true,
            max_data_points: 1000,
            enable_stack_traces: false,
            custom_css: None,
            websocket_config: WebSocketConfig::default(),
        }
    }
}

/// WebSocket configuration for real-time streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConfig {
    /// WebSocket server port
    pub port: u16,
    /// Enable WebSocket streaming
    pub enabled: bool,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Stream update interval in milliseconds
    pub update_interval_ms: u64,
    /// Buffer size for WebSocket messages
    pub buffer_size: usize,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            port: 8081,
            enabled: true,
            max_connections: 100,
            update_interval_ms: 100, // 10 updates per second
            buffer_size: 1024,
        }
    }
}

/// Configuration for visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Enable 3D performance landscapes
    pub enable_3d_landscapes: bool,
    /// Enable advanced heatmaps
    pub enable_heatmaps: bool,
    /// 3D grid resolution
    pub grid_resolution: usize,
    /// Color scheme for visualizations
    pub color_scheme: VisualizationColorScheme,
    /// Animation speed for real-time updates
    pub animation_speed: f64,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            enable_3d_landscapes: true,
            enable_heatmaps: true,
            grid_resolution: 50,
            color_scheme: VisualizationColorScheme::Thermal,
            animation_speed: 1.0,
        }
    }
}

// =============================================================================
// Data Structures
// =============================================================================

/// Real-time dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub timestamp: u64,
    pub performance_metrics: PerformanceMetrics,
    pub memory_metrics: MemoryMetrics,
    pub system_metrics: SystemMetrics,
    pub alerts: Vec<DashboardAlert>,
    pub top_operations: Vec<OperationSummary>,
}

/// Performance metrics for the dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_operations: u64,
    pub average_duration_ms: f64,
    pub operations_per_second: f64,
    pub total_flops: u64,
    pub gflops_per_second: f64,
    pub cpu_utilization: f64,
    pub thread_count: usize,
}

/// Memory metrics for the dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub current_usage_mb: f64,
    pub peak_usage_mb: f64,
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub active_allocations: u64,
    pub fragmentation_ratio: f64,
    pub allocation_rate: f64,
}

/// System metrics for the dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub uptime_seconds: u64,
    pub load_average: f64,
    pub available_memory_mb: f64,
    pub disk_usage_percent: f64,
    pub network_io_mbps: f64,
}

/// Operation summary for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationSummary {
    pub name: String,
    pub category: String,
    pub count: u64,
    pub total_duration_ms: f64,
    pub average_duration_ms: f64,
    pub percentage_of_total: f64,
}

// =============================================================================
// Alert System Types
// =============================================================================

/// Dashboard alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardAlert {
    pub id: String,
    pub severity: DashboardAlertSeverity,
    pub title: String,
    pub message: String,
    pub timestamp: u64,
    pub resolved: bool,
}

/// Dashboard alert severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DashboardAlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

// =============================================================================
// WebSocket Types
// =============================================================================

/// WebSocket client connection
#[derive(Clone)]
pub struct WebSocketClient {
    pub id: uuid::Uuid,
    pub addr: SocketAddr,
    pub connected_at: SystemTime,
    pub sender: tokio::sync::mpsc::UnboundedSender<String>,
    pub subscriptions: HashSet<String>,
}

/// WebSocket subscription types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SubscriptionType {
    DashboardUpdates,
    PerformanceMetrics,
    MemoryMetrics,
    SystemMetrics,
    Alerts,
    TopOperations,
    Visualizations,
}

impl From<&str> for SubscriptionType {
    fn from(s: &str) -> Self {
        match s {
            "dashboard_updates" => SubscriptionType::DashboardUpdates,
            "performance_metrics" => SubscriptionType::PerformanceMetrics,
            "memory_metrics" => SubscriptionType::MemoryMetrics,
            "system_metrics" => SubscriptionType::SystemMetrics,
            "alerts" => SubscriptionType::Alerts,
            "top_operations" => SubscriptionType::TopOperations,
            "visualizations" => SubscriptionType::Visualizations,
            _ => SubscriptionType::DashboardUpdates,
        }
    }
}

impl std::fmt::Display for SubscriptionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubscriptionType::DashboardUpdates => write!(f, "dashboard_updates"),
            SubscriptionType::PerformanceMetrics => write!(f, "performance_metrics"),
            SubscriptionType::MemoryMetrics => write!(f, "memory_metrics"),
            SubscriptionType::SystemMetrics => write!(f, "system_metrics"),
            SubscriptionType::Alerts => write!(f, "alerts"),
            SubscriptionType::TopOperations => write!(f, "top_operations"),
            SubscriptionType::Visualizations => write!(f, "visualizations"),
        }
    }
}

// =============================================================================
// Visualization Types
// =============================================================================

/// Color schemes for visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationColorScheme {
    /// Blue to red gradient (cool to hot)
    Thermal,
    /// Viridis color map (perceptually uniform)
    Viridis,
    /// Plasma color map (high contrast)
    Plasma,
    /// Custom RGB gradient
    Custom { start: [u8; 3], end: [u8; 3] },
}

/// 3D Performance Landscape data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePoint3D {
    /// X coordinate (time or operation index)
    pub x: f64,
    /// Y coordinate (thread or category)
    pub y: f64,
    /// Z coordinate (performance metric value)
    pub z: f64,
    /// Optional color intensity
    pub intensity: f64,
    /// Associated metadata
    pub metadata: String,
}

/// Heatmap cell data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapCell {
    /// Row index
    pub row: usize,
    /// Column index
    pub col: usize,
    /// Intensity value (0.0 to 1.0)
    pub intensity: f64,
    /// Optional label
    pub label: Option<String>,
    /// Color in hex format
    pub color: String,
}

// =============================================================================
// Core Dashboard Types
// =============================================================================

/// Dashboard server state
pub struct Dashboard {
    pub(crate) config: DashboardConfig,
    pub(crate) data_history: Arc<std::sync::Mutex<Vec<DashboardData>>>,
    pub(crate) alerts: Arc<std::sync::Mutex<Vec<DashboardAlert>>>,
    pub(crate) running: Arc<std::sync::Mutex<bool>>,
    pub(crate) websocket_clients: Arc<std::sync::Mutex<Vec<WebSocketClient>>>,
}

// =============================================================================
// Utility Types and Constants
// =============================================================================

/// Dashboard response formats
#[derive(Debug, Clone)]
pub enum ResponseFormat {
    Html,
    Json,
    Csv,
    Xml,
}

/// Time window for data aggregation
#[derive(Debug, Clone, Copy)]
pub enum TimeWindow {
    LastMinute,
    LastFiveMinutes,
    LastHour,
    LastDay,
    Custom { start_ms: u64, end_ms: u64 },
}

/// Dashboard component IDs for selective updates
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ComponentId {
    PerformanceChart,
    MemoryGraph,
    SystemStatus,
    AlertPanel,
    OperationsList,
    Heatmap,
    Landscape3D,
}

/// Dashboard theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTheme {
    pub name: String,
    pub primary_color: String,
    pub secondary_color: String,
    pub background_color: String,
    pub text_color: String,
    pub accent_color: String,
    pub font_family: String,
}

impl Default for DashboardTheme {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            primary_color: "#3498db".to_string(),
            secondary_color: "#2ecc71".to_string(),
            background_color: "#f8f9fa".to_string(),
            text_color: "#2c3e50".to_string(),
            accent_color: "#e74c3c".to_string(),
            font_family: "Arial, sans-serif".to_string(),
        }
    }
}

/// Constants for dashboard configuration
pub mod constants {
    /// Default ports
    pub const DEFAULT_DASHBOARD_PORT: u16 = 8080;
    pub const DEFAULT_WEBSOCKET_PORT: u16 = 8081;

    /// Default time intervals (in milliseconds)
    pub const DEFAULT_REFRESH_INTERVAL_MS: u64 = 5000;
    pub const DEFAULT_WEBSOCKET_UPDATE_INTERVAL_MS: u64 = 100;

    /// Memory and performance limits
    pub const DEFAULT_MAX_DATA_POINTS: usize = 1000;
    pub const DEFAULT_MAX_WEBSOCKET_CONNECTIONS: usize = 100;
    pub const DEFAULT_WEBSOCKET_BUFFER_SIZE: usize = 1024;

    /// Grid and visualization constants
    pub const DEFAULT_GRID_RESOLUTION: usize = 50;
    pub const MAX_GRID_RESOLUTION: usize = 200;
    pub const MIN_GRID_RESOLUTION: usize = 10;

    /// Color intensity ranges
    pub const MIN_INTENSITY: f64 = 0.0;
    pub const MAX_INTENSITY: f64 = 1.0;
}
