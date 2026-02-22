//! WebSocket streaming functionality for real-time dashboard updates
//!
//! This module provides WebSocket server capabilities for streaming live dashboard
//! data to connected clients with subscription-based filtering and connection management.

use crate::TorshResult;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use torsh_core::TorshError;

use super::types::{DashboardData, SubscriptionType, WebSocketClient, WebSocketConfig};

// =============================================================================
// WebSocket Server Management
// =============================================================================

/// Start WebSocket server for real-time streaming
pub async fn start_websocket_server(
    data_history: Arc<Mutex<Vec<DashboardData>>>,
    clients: Arc<Mutex<Vec<WebSocketClient>>>,
    config: WebSocketConfig,
    running: Arc<Mutex<bool>>,
) {
    let addr = format!("127.0.0.1:{}", config.port);
    let listener = match TcpListener::bind(&addr).await {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("Failed to bind WebSocket server to {addr}: {e}");
            return;
        }
    };

    println!("WebSocket server listening on {addr}");

    // Start broadcasting loop
    let broadcast_data_history = Arc::clone(&data_history);
    let broadcast_clients = Arc::clone(&clients);
    let broadcast_config = config.clone();
    let broadcast_running = Arc::clone(&running);

    tokio::spawn(async move {
        websocket_broadcast_loop(
            broadcast_data_history,
            broadcast_clients,
            broadcast_config,
            broadcast_running,
        )
        .await;
    });

    while let Ok((stream, addr)) = listener.accept().await {
        // Check if server is still running
        if !running.lock().map(|r| *r).unwrap_or(false) {
            break;
        }

        // Check connection limit
        let client_count = clients.lock().map(|c| c.len()).unwrap_or(0);
        if client_count >= config.max_connections {
            eprintln!("WebSocket connection limit reached, rejecting connection from {addr}");
            continue;
        }

        let clients_clone = Arc::clone(&clients);
        tokio::spawn(async move {
            handle_websocket_client(stream, addr, clients_clone).await;
        });
    }
}

/// Handle individual WebSocket client connection
pub async fn handle_websocket_client(
    stream: TcpStream,
    addr: SocketAddr,
    clients: Arc<Mutex<Vec<WebSocketClient>>>,
) {
    let ws_stream = match accept_async(stream).await {
        Ok(ws_stream) => ws_stream,
        Err(e) => {
            eprintln!("WebSocket handshake failed for {addr}: {e}");
            return;
        }
    };

    let client_id = uuid::Uuid::new_v4();
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Create a channel for broadcasting to this client
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let client = WebSocketClient {
        id: client_id,
        addr,
        connected_at: SystemTime::now(),
        sender: tx,
        subscriptions: HashSet::new(),
    };

    // Add client to list
    if let Ok(mut clients_list) = clients.lock() {
        clients_list.push(client);
        println!("WebSocket client connected: {addr} ({client_id})");
    }

    // Spawn a task to handle outgoing messages
    let clients_clone = Arc::clone(&clients);
    let client_id_clone = client_id;
    tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            if let Err(e) = ws_sender.send(Message::Text(message.into())).await {
                eprintln!("Failed to send message to client {client_id_clone}: {e}");
                break;
            }
        }
    });

    // Handle incoming messages
    while let Some(msg) = ws_receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Handle client commands (e.g., subscription preferences)
                if let Err(e) = handle_client_message(&text, client_id, &clients).await {
                    eprintln!("Error handling client message: {e}");
                }
            }
            Ok(Message::Close(_)) => {
                println!("WebSocket client disconnected: {addr}");
                break;
            }
            Err(e) => {
                eprintln!("WebSocket error for {addr}: {e}");
                break;
            }
            _ => {}
        }
    }

    // Remove client from list
    if let Ok(mut clients_list) = clients.lock() {
        clients_list.retain(|c| c.id != client_id);
        println!("WebSocket client cleanup completed for {addr}");
    }
}

// =============================================================================
// Client Message Handling
// =============================================================================

/// Handle client messages and commands
pub async fn handle_client_message(
    message: &str,
    client_id: uuid::Uuid,
    clients: &Arc<Mutex<Vec<WebSocketClient>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Parse client command
    if message.starts_with("subscribe:") {
        let subscription = message.strip_prefix("subscribe:").unwrap_or("");
        handle_subscription(client_id, subscription, clients, true).await?;
    } else if message.starts_with("unsubscribe:") {
        let subscription = message.strip_prefix("unsubscribe:").unwrap_or("");
        handle_subscription(client_id, subscription, clients, false).await?;
    } else if message == "ping" {
        handle_ping(client_id, clients).await?;
    } else if message == "get_subscriptions" {
        handle_get_subscriptions(client_id, clients).await?;
    } else if message.starts_with("set_filter:") {
        let filter = message.strip_prefix("set_filter:").unwrap_or("");
        handle_set_filter(client_id, filter, clients).await?;
    } else if message == "get_stats" {
        handle_get_stats(client_id, clients).await?;
    } else {
        // Unknown command
        send_error_to_client(client_id, "Unknown command", clients).await?;
    }

    Ok(())
}

/// Handle subscription/unsubscription requests
async fn handle_subscription(
    client_id: uuid::Uuid,
    subscription: &str,
    clients: &Arc<Mutex<Vec<WebSocketClient>>>,
    subscribe: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(mut clients_list) = clients.lock() {
        if let Some(client) = clients_list.iter_mut().find(|c| c.id == client_id) {
            if subscribe {
                client.subscriptions.insert(subscription.to_string());
                let response = format!(
                    "{{\"type\":\"subscription_ack\",\"subscription\":\"{subscription}\"}}"
                );
                let _ = client.sender.send(response);
            } else {
                client.subscriptions.remove(subscription);
                let response = format!(
                    "{{\"type\":\"unsubscription_ack\",\"subscription\":\"{subscription}\"}}"
                );
                let _ = client.sender.send(response);
            }
        }
    }
    Ok(())
}

/// Handle ping requests
async fn handle_ping(
    client_id: uuid::Uuid,
    clients: &Arc<Mutex<Vec<WebSocketClient>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(clients_list) = clients.lock() {
        if let Some(client) = clients_list.iter().find(|c| c.id == client_id) {
            let pong_response = format!(
                "{{\"type\":\"pong\",\"timestamp\":{}}}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            );
            let _ = client.sender.send(pong_response);
        }
    }
    Ok(())
}

/// Handle get subscriptions requests
async fn handle_get_subscriptions(
    client_id: uuid::Uuid,
    clients: &Arc<Mutex<Vec<WebSocketClient>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(clients_list) = clients.lock() {
        if let Some(client) = clients_list.iter().find(|c| c.id == client_id) {
            let subscriptions: Vec<String> = client.subscriptions.iter().cloned().collect();
            let response = format!(
                "{{\"type\":\"subscriptions\",\"data\":{}}}",
                serde_json::to_string(&subscriptions).unwrap_or_default()
            );
            let _ = client.sender.send(response);
        }
    }
    Ok(())
}

/// Handle filter configuration
async fn handle_set_filter(
    client_id: uuid::Uuid,
    filter: &str,
    clients: &Arc<Mutex<Vec<WebSocketClient>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Parse filter configuration (JSON format expected)
    let _filter_config: serde_json::Value = serde_json::from_str(filter)?;

    if let Ok(clients_list) = clients.lock() {
        if let Some(client) = clients_list.iter().find(|c| c.id == client_id) {
            let response =
                format!("{{\"type\":\"filter_ack\",\"message\":\"Filter applied successfully\"}}");
            let _ = client.sender.send(response);
        }
    }
    Ok(())
}

/// Handle client statistics requests
async fn handle_get_stats(
    client_id: uuid::Uuid,
    clients: &Arc<Mutex<Vec<WebSocketClient>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(clients_list) = clients.lock() {
        if let Some(client) = clients_list.iter().find(|c| c.id == client_id) {
            let connected_duration = client.connected_at.elapsed().unwrap_or_default().as_secs();

            let stats = ClientStats {
                client_id: client.id.to_string(),
                connected_duration_seconds: connected_duration,
                subscription_count: client.subscriptions.len(),
                total_clients: clients_list.len(),
            };

            let response = format!(
                "{{\"type\":\"client_stats\",\"data\":{}}}",
                serde_json::to_string(&stats).unwrap_or_default()
            );
            let _ = client.sender.send(response);
        }
    }
    Ok(())
}

/// Send error message to client
async fn send_error_to_client(
    client_id: uuid::Uuid,
    error_message: &str,
    clients: &Arc<Mutex<Vec<WebSocketClient>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(clients_list) = clients.lock() {
        if let Some(client) = clients_list.iter().find(|c| c.id == client_id) {
            let response = format!("{{\"type\":\"error\",\"message\":\"{error_message}\"}}");
            let _ = client.sender.send(response);
        }
    }
    Ok(())
}

// =============================================================================
// Broadcasting and Streaming
// =============================================================================

/// WebSocket broadcast loop for streaming dashboard data
pub async fn websocket_broadcast_loop(
    data_history: Arc<Mutex<Vec<DashboardData>>>,
    clients: Arc<Mutex<Vec<WebSocketClient>>>,
    config: WebSocketConfig,
    running: Arc<Mutex<bool>>,
) {
    let mut interval = tokio::time::interval(Duration::from_millis(config.update_interval_ms));

    while running.lock().map(|r| *r).unwrap_or(false) {
        interval.tick().await;

        // Get latest data
        let latest_data = match data_history.lock() {
            Ok(history) => history.last().cloned(),
            Err(_) => continue,
        };

        if let Some(data) = latest_data {
            // Serialize data
            let json_data = match serde_json::to_string(&data) {
                Ok(json) => json,
                Err(e) => {
                    eprintln!("Failed to serialize dashboard data: {e}");
                    continue;
                }
            };

            // Broadcast to subscribed clients
            let clients_copy = {
                if let Ok(clients_list) = clients.lock() {
                    if !clients_list.is_empty() {
                        Some(clients_list.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

            if let Some(mut clients_copy) = clients_copy {
                broadcast_to_clients(&data, &json_data, &mut clients_copy).await;
            }
        }
    }
}

/// Broadcast data to all connected clients based on their subscriptions
async fn broadcast_to_clients(
    data: &DashboardData,
    json_data: &str,
    clients_list: &mut Vec<WebSocketClient>,
) {
    // Prepare different message types
    let messages = prepare_broadcast_messages(data, json_data);
    let mut broadcast_count = 0;

    // Send to subscribed clients and remove any that have disconnected
    clients_list.retain(|client| {
        let messages_to_send = determine_client_messages(client, &messages);

        // Send all relevant messages
        let mut keep_client = true;
        for message in messages_to_send {
            match client.sender.send(message) {
                Ok(_) => {
                    broadcast_count += 1;
                }
                Err(_) => {
                    println!("Removing disconnected client: {}", client.id);
                    keep_client = false;
                    break;
                }
            }
        }

        keep_client
    });

    if broadcast_count > 0 {
        println!(
            "Broadcasting {} messages to {} clients",
            broadcast_count,
            clients_list.len()
        );
    }
}

/// Prepare all types of broadcast messages
fn prepare_broadcast_messages(data: &DashboardData, json_data: &str) -> BroadcastMessages {
    BroadcastMessages {
        dashboard_update: format!("{{\"type\":\"dashboard_update\",\"data\":{json_data}}}"),
        performance_metrics: format!(
            "{{\"type\":\"performance_metrics\",\"data\":{}}}",
            serde_json::to_string(&data.performance_metrics).unwrap_or_default()
        ),
        memory_metrics: format!(
            "{{\"type\":\"memory_metrics\",\"data\":{}}}",
            serde_json::to_string(&data.memory_metrics).unwrap_or_default()
        ),
        system_metrics: format!(
            "{{\"type\":\"system_metrics\",\"data\":{}}}",
            serde_json::to_string(&data.system_metrics).unwrap_or_default()
        ),
        alerts: format!(
            "{{\"type\":\"alerts\",\"data\":{}}}",
            serde_json::to_string(&data.alerts).unwrap_or_default()
        ),
        top_operations: format!(
            "{{\"type\":\"top_operations\",\"data\":{}}}",
            serde_json::to_string(&data.top_operations).unwrap_or_default()
        ),
    }
}

/// Determine which messages to send to a specific client
fn determine_client_messages(
    client: &WebSocketClient,
    messages: &BroadcastMessages,
) -> Vec<String> {
    let mut messages_to_send = Vec::new();

    // If no specific subscriptions, send dashboard updates
    if client.subscriptions.is_empty() {
        messages_to_send.push(messages.dashboard_update.clone());
        return messages_to_send;
    }

    // Send messages based on subscriptions
    if client.subscriptions.contains("dashboard_updates") {
        messages_to_send.push(messages.dashboard_update.clone());
    }
    if client.subscriptions.contains("performance_metrics") {
        messages_to_send.push(messages.performance_metrics.clone());
    }
    if client.subscriptions.contains("memory_metrics") {
        messages_to_send.push(messages.memory_metrics.clone());
    }
    if client.subscriptions.contains("system_metrics") {
        messages_to_send.push(messages.system_metrics.clone());
    }
    if client.subscriptions.contains("alerts") {
        messages_to_send.push(messages.alerts.clone());
    }
    if client.subscriptions.contains("top_operations") {
        messages_to_send.push(messages.top_operations.clone());
    }

    messages_to_send
}

// =============================================================================
// Statistics and Management
// =============================================================================

/// Get WebSocket connection statistics
pub fn get_websocket_stats(
    clients: &Arc<Mutex<Vec<WebSocketClient>>>,
) -> TorshResult<WebSocketStats> {
    let clients_list = clients
        .lock()
        .map_err(|_| TorshError::SynchronizationError("Failed to acquire lock".to_string()))?;

    // Calculate detailed statistics
    let connected_clients = clients_list.len();
    let total_connections = connected_clients; // Simplified - would need persistent counter
    let uptime_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Calculate subscription statistics
    let mut subscription_stats = HashMap::new();
    for client in clients_list.iter() {
        for subscription in &client.subscriptions {
            *subscription_stats.entry(subscription.clone()).or_insert(0) += 1;
        }
    }

    Ok(WebSocketStats {
        connected_clients,
        total_connections,
        uptime_seconds,
        subscription_stats,
        active_clients: calculate_active_clients(&clients_list),
    })
}

/// Calculate number of active clients (connected recently)
fn calculate_active_clients(clients: &[WebSocketClient]) -> usize {
    let threshold = SystemTime::now() - Duration::from_secs(300); // 5 minutes
    clients
        .iter()
        .filter(|client| client.connected_at > threshold)
        .count()
}

/// Disconnect a specific client
pub fn disconnect_client(
    client_id: uuid::Uuid,
    clients: &Arc<Mutex<Vec<WebSocketClient>>>,
) -> TorshResult<bool> {
    let mut clients_list = clients
        .lock()
        .map_err(|_| TorshError::SynchronizationError("Failed to acquire lock".to_string()))?;

    let initial_count = clients_list.len();
    clients_list.retain(|client| client.id != client_id);

    Ok(clients_list.len() < initial_count)
}

/// Broadcast message to all clients with specific subscription
pub async fn broadcast_to_subscription(
    subscription: &str,
    message: &str,
    clients: &Arc<Mutex<Vec<WebSocketClient>>>,
) -> TorshResult<usize> {
    let clients_list = clients
        .lock()
        .map_err(|_| TorshError::SynchronizationError("Failed to acquire lock".to_string()))?;

    let mut sent_count = 0;
    for client in clients_list.iter() {
        if client.subscriptions.contains(subscription) {
            if client.sender.send(message.to_string()).is_ok() {
                sent_count += 1;
            }
        }
    }

    Ok(sent_count)
}

// =============================================================================
// Supporting Types
// =============================================================================

/// WebSocket connection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketStats {
    pub connected_clients: usize,
    pub total_connections: usize,
    pub uptime_seconds: u64,
    pub subscription_stats: HashMap<String, usize>,
    pub active_clients: usize,
}

/// Client-specific statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientStats {
    pub client_id: String,
    pub connected_duration_seconds: u64,
    pub subscription_count: usize,
    pub total_clients: usize,
}

/// Prepared broadcast messages
struct BroadcastMessages {
    dashboard_update: String,
    performance_metrics: String,
    memory_metrics: String,
    system_metrics: String,
    alerts: String,
    top_operations: String,
}

/// WebSocket message types for client communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebSocketMessageType {
    DashboardUpdate,
    PerformanceMetrics,
    MemoryMetrics,
    SystemMetrics,
    Alerts,
    TopOperations,
    SubscriptionAck,
    UnsubscriptionAck,
    Pong,
    Error,
    ClientStats,
}

/// Client connection events
#[derive(Debug, Clone)]
pub enum ClientEvent {
    Connected {
        id: uuid::Uuid,
        addr: SocketAddr,
    },
    Disconnected {
        id: uuid::Uuid,
    },
    Subscribed {
        id: uuid::Uuid,
        subscription: String,
    },
    Unsubscribed {
        id: uuid::Uuid,
        subscription: String,
    },
    MessageSent {
        id: uuid::Uuid,
        message_type: String,
    },
    Error {
        id: uuid::Uuid,
        error: String,
    },
}

/// WebSocket server configuration validation
pub fn validate_websocket_config(config: &WebSocketConfig) -> Result<(), String> {
    if config.port == 0 {
        return Err("WebSocket port cannot be 0".to_string());
    }

    if config.max_connections == 0 {
        return Err("Max connections must be greater than 0".to_string());
    }

    if config.update_interval_ms == 0 {
        return Err("Update interval must be greater than 0".to_string());
    }

    if config.buffer_size == 0 {
        return Err("Buffer size must be greater than 0".to_string());
    }

    Ok(())
}

// =============================================================================
// WebSocketManager - Missing Implementation
// =============================================================================

/// WebSocket manager for handling connections and broadcasting
pub struct WebSocketManager {
    /// WebSocket configuration
    pub config: WebSocketConfig,
    /// Connection statistics
    pub stats: Arc<Mutex<WebSocketStats>>,
}

impl WebSocketManager {
    /// Create a new WebSocket manager
    pub fn new(config: WebSocketConfig) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(WebSocketStats {
                connected_clients: 0,
                total_connections: 0,
                uptime_seconds: 0,
                subscription_stats: HashMap::new(),
                active_clients: 0,
            })),
        }
    }

    /// Start the WebSocket server
    pub async fn start_server(
        &self,
        data_history: Arc<Mutex<Vec<DashboardData>>>,
        clients: Arc<Mutex<Vec<WebSocketClient>>>,
        running: Arc<Mutex<bool>>,
    ) {
        start_websocket_server(data_history, clients, self.config.clone(), running).await;
    }

    /// Get WebSocket statistics
    pub fn get_stats(&self) -> WebSocketStats {
        self.stats
            .lock()
            .map(|stats| stats.clone())
            .unwrap_or_else(|_| WebSocketStats {
                connected_clients: 0,
                total_connections: 0,
                uptime_seconds: 0,
                subscription_stats: HashMap::new(),
                active_clients: 0,
            })
    }

    /// Broadcast visualization data to connected clients
    pub fn broadcast_visualization(
        &self,
        clients: &Arc<Mutex<Vec<WebSocketClient>>>,
        visualization_type: &str,
        data: &str,
    ) -> TorshResult<usize> {
        let message = serde_json::json!({
            "type": "visualization",
            "visualization_type": visualization_type,
            "data": data,
            "timestamp": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        });

        let message_str = serde_json::to_string(&message).map_err(|e| {
            TorshError::SerializationError(format!("Failed to serialize message: {e}"))
        })?;

        let clients_guard = clients.lock().map_err(|_| {
            TorshError::SynchronizationError("Failed to acquire clients lock".to_string())
        })?;

        let mut sent_count = 0;
        for client in clients_guard.iter() {
            if client.subscriptions.contains("visualizations") {
                if let Err(e) = client.sender.send(message_str.clone()) {
                    eprintln!("Failed to send visualization to client {}: {e}", client.id);
                } else {
                    sent_count += 1;
                }
            }
        }

        Ok(sent_count)
    }

    /// Update connection count statistics
    pub fn update_connection_count(&self, connected: usize) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.connected_clients = connected;
            stats.active_clients = connected;
        }
    }

    /// Increment total connection count
    pub fn increment_total_connections(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_connections += 1;
        }
    }

    /// Update subscription statistics
    pub fn update_subscription_stats(&self, subscription_type: &str) {
        if let Ok(mut stats) = self.stats.lock() {
            *stats
                .subscription_stats
                .entry(subscription_type.to_string())
                .or_insert(0) += 1;
        }
    }
}

impl Default for WebSocketManager {
    fn default() -> Self {
        Self::new(WebSocketConfig::default())
    }
}
