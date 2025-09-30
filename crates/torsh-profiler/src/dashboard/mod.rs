//! Real-time Performance Dashboard
//!
//! This module provides a comprehensive real-time performance monitoring dashboard that can be
//! accessed via web interface, displaying live profiling data, alerts, analytics, and visualizations.
//!
//! The dashboard is organized into several focused modules:
//! - `types`: Core data structures and configuration types
//! - `metrics`: Performance, memory, and system metrics collection
//! - `websocket`: Real-time WebSocket streaming functionality
//! - `html`: Dashboard web interface generation
//! - `visualizations`: 3D landscapes and heatmap visualizations
//! - `alerts`: Comprehensive alert management system
//!
//! # Quick Start
//!
//! ```rust
//! use torsh_profiler::dashboard::{Dashboard, DashboardConfig, create_dashboard};
//! use torsh_profiler::{Profiler, MemoryProfiler};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create dashboard with default configuration
//!     let dashboard = create_dashboard();
//!
//!     // Or create with custom configuration
//!     let config = DashboardConfig {
//!         port: 9090,
//!         refresh_interval: 10,
//!         real_time_updates: true,
//!         max_data_points: 500,
//!         enable_stack_traces: true,
//!         custom_css: None,
//!         websocket_config: Default::default(),
//!     };
//!     let dashboard = Dashboard::new(config);
//!
//!     // Start the dashboard with profilers
//!     let profiler = Arc::new(Profiler::new());
//!     let memory_profiler = Arc::new(MemoryProfiler::new());
//!     dashboard.start(profiler, memory_profiler)?;
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! ## Real-time Monitoring
//! - Live performance metrics tracking
//! - Memory usage monitoring with leak detection
//! - System resource monitoring
//! - WebSocket streaming for real-time updates
//!
//! ## Advanced Visualizations
//! - 3D performance landscapes showing operation patterns over time
//! - Interactive heatmaps with multiple color schemes
//! - Customizable visualization configurations
//! - Export capabilities for external analysis tools
//!
//! ## Intelligent Alerting
//! - Configurable threshold-based alerts
//! - Automatic alert escalation and resolution
//! - Multiple severity levels (Info, Warning, Critical, Emergency)
//! - Real-time alert broadcasting to connected clients
//!
//! ## Web Interface
//! - Responsive HTML dashboard with modern styling
//! - Customizable themes and CSS
//! - Auto-refreshing data displays
//! - Mobile-friendly responsive design

// Re-export all module functionality
pub mod alerts;
pub mod html;
pub mod metrics;
pub mod types;
pub mod visualizations;
pub mod websocket;

// Re-export core types for convenience
pub use types::*;

// Re-export main dashboard functionality
pub use metrics::{
    collect_memory_metrics, collect_performance_metrics, collect_system_metrics, MetricsCollector,
};

pub use websocket::{
    handle_websocket_client, start_websocket_server, WebSocketManager, WebSocketStats,
};

pub use html::{build_html_document, generate_dashboard_html, DashboardRenderer};

pub use visualizations::{
    generate_3d_landscape, generate_performance_heatmap, PerformanceHeatmap, PerformanceLandscape,
};

pub use alerts::{
    create_alert_context, create_alert_manager, create_alert_manager_with_config, AlertConfig,
    AlertContext, AlertManager,
};

// Main Dashboard imports
use crate::{MemoryProfiler, ProfileEvent, Profiler, TorshResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use torsh_core::TorshError;

// =============================================================================
// Main Dashboard Implementation
// =============================================================================

/// Main Dashboard server that orchestrates all functionality
pub struct Dashboard {
    pub(crate) config: DashboardConfig,
    pub(crate) data_history: Arc<Mutex<Vec<DashboardData>>>,
    pub(crate) alerts: Arc<Mutex<Vec<DashboardAlert>>>,
    pub(crate) running: Arc<Mutex<bool>>,
    pub(crate) websocket_clients: Arc<Mutex<Vec<WebSocketClient>>>,
    pub(crate) alert_manager: Arc<Mutex<AlertManager>>,
    pub(crate) metrics_collector: Arc<MetricsCollector>,
    pub(crate) websocket_manager: Arc<WebSocketManager>,
    pub(crate) dashboard_renderer: Arc<DashboardRenderer>,
}

impl Dashboard {
    /// Create a new dashboard
    pub fn new(config: DashboardConfig) -> Self {
        let alert_config = alerts::AlertConfig::default();
        let alert_manager = AlertManager::new(alert_config);

        Self {
            config: config.clone(),
            data_history: Arc::new(Mutex::new(Vec::new())),
            alerts: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(Mutex::new(false)),
            websocket_clients: Arc::new(Mutex::new(Vec::new())),
            alert_manager: Arc::new(Mutex::new(alert_manager)),
            metrics_collector: Arc::new(MetricsCollector::new()),
            websocket_manager: Arc::new(WebSocketManager::new(config.websocket_config.clone())),
            dashboard_renderer: Arc::new(DashboardRenderer::new()),
        }
    }

    /// Start the dashboard server
    pub fn start(
        &self,
        profiler: Arc<Profiler>,
        memory_profiler: Arc<MemoryProfiler>,
    ) -> TorshResult<()> {
        {
            let mut running = self.running.lock().map_err(|_| {
                TorshError::SynchronizationError("Failed to acquire lock".to_string())
            })?;
            if *running {
                return Err(TorshError::RuntimeError(
                    "Dashboard already running".to_string(),
                ));
            }
            *running = true;
        }

        // Start data collection thread
        let data_history = Arc::clone(&self.data_history);
        let alerts = Arc::clone(&self.alerts);
        let alert_manager = Arc::clone(&self.alert_manager);
        let metrics_collector = Arc::clone(&self.metrics_collector);
        let config = self.config.clone();
        let running = Arc::clone(&self.running);
        let profiler_clone = Arc::clone(&profiler);
        let memory_profiler_clone = Arc::clone(&memory_profiler);

        thread::spawn(move || {
            Self::data_collection_loop(
                data_history,
                alerts,
                alert_manager,
                metrics_collector,
                config,
                running,
                profiler_clone,
                memory_profiler_clone,
            );
        });

        // Start WebSocket server if enabled
        if self.config.websocket_config.enabled {
            let websocket_data_history = Arc::clone(&self.data_history);
            let websocket_clients = Arc::clone(&self.websocket_clients);
            let websocket_manager = Arc::clone(&self.websocket_manager);
            let websocket_running = Arc::clone(&self.running);

            tokio::spawn(async move {
                websocket_manager
                    .start_server(websocket_data_history, websocket_clients, websocket_running)
                    .await;
            });
        }

        println!("Dashboard started on port {}", self.config.port);
        println!("Access dashboard at: http://localhost:{}", self.config.port);

        if self.config.websocket_config.enabled {
            println!(
                "WebSocket server started on port {}",
                self.config.websocket_config.port
            );
            println!(
                "WebSocket endpoint: ws://localhost:{}/ws",
                self.config.websocket_config.port
            );
        }

        Ok(())
    }

    /// Stop the dashboard server
    pub fn stop(&self) -> TorshResult<()> {
        let mut running = self
            .running
            .lock()
            .map_err(|_| TorshError::SynchronizationError("Failed to acquire lock".to_string()))?;
        *running = false;
        println!("Dashboard stopped");
        Ok(())
    }

    /// Get current dashboard data
    pub fn get_current_data(&self) -> TorshResult<Option<DashboardData>> {
        let data_history = self
            .data_history
            .lock()
            .map_err(|_| TorshError::SynchronizationError("Failed to acquire lock".to_string()))?;
        Ok(data_history.last().cloned())
    }

    /// Get data history
    pub fn get_data_history(&self) -> TorshResult<Vec<DashboardData>> {
        let data_history = self
            .data_history
            .lock()
            .map_err(|_| TorshError::SynchronizationError("Failed to acquire lock".to_string()))?;
        Ok(data_history.clone())
    }

    /// Add alert
    pub fn add_alert(&self, alert: DashboardAlert) -> TorshResult<()> {
        // Add to main alerts list
        let mut alerts = self
            .alerts
            .lock()
            .map_err(|_| TorshError::SynchronizationError("Failed to acquire lock".to_string()))?;
        alerts.push(alert.clone());

        // Keep only recent alerts
        if alerts.len() > 100 {
            alerts.remove(0);
        }

        // Also add to alert manager
        if let Ok(mut alert_manager) = self.alert_manager.lock() {
            let _ = alert_manager.add_alert(alert);
        }

        Ok(())
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> TorshResult<Vec<DashboardAlert>> {
        let alerts = self
            .alerts
            .lock()
            .map_err(|_| TorshError::SynchronizationError("Failed to acquire lock".to_string()))?;
        Ok(alerts.iter().filter(|a| !a.resolved).cloned().collect())
    }

    /// Generate dashboard HTML
    pub fn generate_dashboard_html(&self) -> TorshResult<String> {
        let current_data = self.get_current_data()?.unwrap_or_else(create_default_data);
        let active_alerts = self.get_active_alerts()?;
        let theme = DashboardTheme::default();

        self.dashboard_renderer.generate_dashboard_html(
            &current_data,
            &active_alerts,
            &self.config,
            &theme,
        )
    }

    /// Export dashboard data to JSON
    pub fn export_data_json(&self, file_path: &str) -> TorshResult<()> {
        let data_history = self.get_data_history()?;
        let active_alerts = self.get_active_alerts()?;

        let export_data = serde_json::json!({
            "data_history": data_history,
            "active_alerts": active_alerts,
            "config": self.config
        });

        std::fs::write(
            file_path,
            serde_json::to_string_pretty(&export_data).map_err(|e| {
                TorshError::SerializationError(format!("Failed to serialize dashboard data: {e}"))
            })?,
        )
        .map_err(|e| TorshError::IoError(format!("Failed to write dashboard data: {e}")))?;

        Ok(())
    }

    /// Get WebSocket connection statistics
    pub fn get_websocket_stats(&self) -> TorshResult<WebSocketStats> {
        let clients = self
            .websocket_clients
            .lock()
            .map_err(|_| TorshError::SynchronizationError("Failed to acquire lock".to_string()))?;

        Ok(WebSocketStats {
            connected_clients: clients.len(),
            total_connections: clients.len(), // Simplified - would need persistent counter
            uptime_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            subscription_stats: std::collections::HashMap::new(),
            active_clients: clients.len(),
        })
    }

    /// Broadcast visualization data to connected clients
    pub fn broadcast_visualization(
        &self,
        visualization_type: &str,
        data: &str,
    ) -> TorshResult<usize> {
        self.websocket_manager.broadcast_visualization(
            &self.websocket_clients,
            visualization_type,
            data,
        )
    }

    /// Broadcast real-time 3D performance landscape
    pub fn broadcast_3d_landscape(
        &self,
        profiler: &Profiler,
        config: &VisualizationConfig,
    ) -> TorshResult<usize> {
        let mut landscape = PerformanceLandscape::new(config.clone());
        landscape.generate_from_profiler(profiler)?;

        let landscape_data = serde_json::to_string(&landscape.get_points()).map_err(|e| {
            TorshError::SerializationError(format!("Failed to serialize landscape points: {e}"))
        })?;

        self.broadcast_visualization("3d_landscape", &landscape_data)
    }

    /// Broadcast real-time performance heatmap
    pub fn broadcast_heatmap(
        &self,
        profiler: &Profiler,
        config: &VisualizationConfig,
        width: usize,
        height: usize,
    ) -> TorshResult<usize> {
        let mut heatmap = PerformanceHeatmap::new(config.clone(), width, height);
        heatmap.generate_from_profiler(profiler)?;

        let heatmap_data = serde_json::to_string(&heatmap.get_cells()).map_err(|e| {
            TorshError::SerializationError(format!("Failed to serialize heatmap cells: {e}"))
        })?;

        self.broadcast_visualization("heatmap", &heatmap_data)
    }

    /// Broadcast custom alerts to connected clients
    pub fn broadcast_alert(&self, alert: &DashboardAlert) -> TorshResult<usize> {
        if let Ok(alert_manager) = self.alert_manager.lock() {
            alert_manager.broadcast_alert(alert, &self.websocket_clients)
        } else {
            Err(TorshError::SynchronizationError(
                "Failed to acquire alert manager lock".to_string(),
            ))
        }
    }

    /// Data collection loop
    fn data_collection_loop(
        data_history: Arc<Mutex<Vec<DashboardData>>>,
        _alerts: Arc<Mutex<Vec<DashboardAlert>>>,
        alert_manager: Arc<Mutex<AlertManager>>,
        metrics_collector: Arc<MetricsCollector>,
        config: DashboardConfig,
        running: Arc<Mutex<bool>>,
        profiler: Arc<Profiler>,
        memory_profiler: Arc<MemoryProfiler>,
    ) {
        while {
            let is_running = running.lock().map(|r| *r).unwrap_or(false);
            is_running
        } {
            // Collect current metrics
            if let Ok(data) = metrics_collector.collect_dashboard_data(&profiler, &memory_profiler)
            {
                if let Ok(mut history) = data_history.lock() {
                    history.push(data.clone());

                    // Keep only recent data points
                    if history.len() > config.max_data_points {
                        history.remove(0);
                    }
                }

                // Generate alerts based on current metrics
                if let Ok(mut alert_mgr) = alert_manager.lock() {
                    if let Ok(alert_context) = create_alert_context(&profiler, &memory_profiler) {
                        let _ = alert_mgr.generate_alerts(alert_context);
                    }
                }
            }

            // Sleep for refresh interval
            thread::sleep(Duration::from_secs(config.refresh_interval));
        }
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create a new dashboard with default configuration
pub fn create_dashboard() -> Dashboard {
    Dashboard::new(DashboardConfig::default())
}

/// Create a new dashboard with custom configuration
pub fn create_dashboard_with_config(config: DashboardConfig) -> Dashboard {
    Dashboard::new(config)
}

/// Export dashboard HTML to file
pub fn export_dashboard_html(
    profiler: &Profiler,
    memory_profiler: &MemoryProfiler,
    file_path: &str,
) -> TorshResult<()> {
    let dashboard = create_dashboard();
    let html = dashboard.generate_dashboard_html()?;

    std::fs::write(file_path, html)
        .map_err(|e| TorshError::IoError(format!("Failed to write dashboard HTML: {e}")))?;

    Ok(())
}

/// Create default dashboard data for empty state
pub fn create_default_data() -> DashboardData {
    DashboardData {
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        performance_metrics: PerformanceMetrics {
            total_operations: 0,
            average_duration_ms: 0.0,
            operations_per_second: 0.0,
            total_flops: 0,
            gflops_per_second: 0.0,
            cpu_utilization: 0.0,
            thread_count: 0,
        },
        memory_metrics: MemoryMetrics {
            current_usage_mb: 0.0,
            peak_usage_mb: 0.0,
            total_allocations: 0,
            total_deallocations: 0,
            active_allocations: 0,
            fragmentation_ratio: 0.0,
            allocation_rate: 0.0,
        },
        system_metrics: SystemMetrics {
            uptime_seconds: 0,
            load_average: 0.0,
            available_memory_mb: 0.0,
            disk_usage_percent: 0.0,
            network_io_mbps: 0.0,
        },
        alerts: Vec::new(),
        top_operations: Vec::new(),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{memory::MemoryProfiler, ProfileEvent};

    #[test]
    fn test_dashboard_creation() {
        let dashboard = create_dashboard();
        assert_eq!(dashboard.config.port, 8080);
        assert_eq!(dashboard.config.refresh_interval, 5);
    }

    #[test]
    fn test_dashboard_config() {
        let config = DashboardConfig {
            port: 9090,
            refresh_interval: 10,
            real_time_updates: false,
            max_data_points: 500,
            enable_stack_traces: true,
            custom_css: Some("body { background: red; }".to_string()),
            websocket_config: WebSocketConfig::default(),
        };

        let dashboard = create_dashboard_with_config(config.clone());
        assert_eq!(dashboard.config.port, 9090);
        assert_eq!(dashboard.config.refresh_interval, 10);
        assert!(!dashboard.config.real_time_updates);
    }

    #[test]
    fn test_dashboard_alert() {
        let dashboard = create_dashboard();

        let alert = DashboardAlert {
            id: "test_alert".to_string(),
            severity: DashboardAlertSeverity::Warning,
            title: "Test Alert".to_string(),
            message: "This is a test alert".to_string(),
            timestamp: 12345,
            resolved: false,
        };

        dashboard.add_alert(alert).unwrap();
        let active_alerts = dashboard.get_active_alerts().unwrap();
        assert_eq!(active_alerts.len(), 1);
    }

    #[test]
    fn test_dashboard_html_generation() {
        let dashboard = create_dashboard();
        let html = dashboard.generate_dashboard_html().unwrap();

        assert!(html.contains("ToRSh Performance Dashboard"));
        assert!(html.contains("Performance Metrics"));
        assert!(html.contains("Memory Metrics"));
        assert!(html.contains("Active Alerts"));
        assert!(html.contains("Top Operations"));
    }

    #[test]
    fn test_default_data_creation() {
        let data = create_default_data();
        assert_eq!(data.performance_metrics.total_operations, 0);
        assert_eq!(data.memory_metrics.current_usage_mb, 0.0);
        assert_eq!(data.alerts.len(), 0);
        assert_eq!(data.top_operations.len(), 0);
    }

    #[test]
    fn test_websocket_stats() {
        let dashboard = create_dashboard();
        let stats = dashboard.get_websocket_stats().unwrap();
        assert_eq!(stats.connected_clients, 0);
        assert_eq!(stats.total_connections, 0);
    }

    #[test]
    fn test_data_export() {
        let dashboard = create_dashboard();

        // Add some test data
        let alert = DashboardAlert {
            id: "test_alert".to_string(),
            severity: DashboardAlertSeverity::Info,
            title: "Test".to_string(),
            message: "Test message".to_string(),
            timestamp: 12345,
            resolved: false,
        };
        dashboard.add_alert(alert).unwrap();

        // Test export
        let temp_file = "/tmp/dashboard_export_test.json";
        dashboard.export_data_json(temp_file).unwrap();

        // Verify file exists and contains data
        let contents = std::fs::read_to_string(temp_file).unwrap();
        assert!(contents.contains("test_alert"));
        assert!(contents.contains("active_alerts"));
        assert!(contents.contains("config"));

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }
}
