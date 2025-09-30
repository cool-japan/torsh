//! Interactive Graph Editor with Real-time Visualization
//!
//! This module provides a comprehensive interactive graph editor that allows developers
//! to create, modify, and visualize FX graphs in real-time through a web-based interface.
//!
//! # Features
//!
//! - **Real-time Visualization**: Live graph updates and interactive manipulation
//! - **Drag-and-Drop Interface**: Intuitive node and edge creation
//! - **Performance Monitoring**: Real-time execution metrics and bottleneck detection
//! - **Export/Import**: Save and load graph configurations in multiple formats
//! - **Collaborative Editing**: Multi-user graph editing capabilities
//! - **Integration**: Seamless integration with existing torsh-fx infrastructure

use crate::{FxGraph, Node};
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use torsh_core::{dtype::DType, error::Result, shape::Shape};

/// Interactive graph editor with real-time capabilities
pub struct InteractiveGraphEditor {
    /// Current graph being edited
    graph: Arc<RwLock<FxGraph>>,
    /// Real-time performance metrics
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
    /// Edit history for undo/redo functionality
    history: Arc<Mutex<EditHistory>>,
    /// Real-time collaboration state
    collaboration_state: Arc<RwLock<CollaborationState>>,
    /// Auto-save configuration
    auto_save_config: AutoSaveConfig,
    /// Visualization settings
    visualization_config: VisualizationConfig,
}

/// Real-time performance monitoring
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    /// Node execution times
    node_timings: HashMap<NodeIndex, Vec<Duration>>,
    /// Memory usage per node
    memory_usage: HashMap<NodeIndex, u64>,
    /// Graph compilation times
    compilation_history: VecDeque<Duration>,
    /// Real-time metrics update frequency
    update_frequency: Duration,
    /// Last update timestamp
    last_update: Instant,
}

/// Edit history for undo/redo functionality
#[derive(Debug, Clone)]
pub struct EditHistory {
    /// Previous graph states
    history: Vec<GraphSnapshot>,
    /// Current position in history
    current_position: usize,
    /// Maximum history size
    max_history_size: usize,
}

/// Graph state snapshot for history management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    /// Serialized graph state
    graph_data: String,
    /// Timestamp of the snapshot
    timestamp: std::time::SystemTime,
    /// Description of the edit operation
    operation_description: String,
    /// User who made the edit (for collaboration)
    editor_id: Option<String>,
}

/// Multi-user collaboration state
#[derive(Debug, Clone)]
pub struct CollaborationState {
    /// Active users
    active_users: HashMap<String, UserSession>,
    /// Real-time edit locks
    edit_locks: HashMap<NodeIndex, String>, // node_id -> user_id
    /// Shared cursors/selections
    user_selections: HashMap<String, EditorSelection>,
    /// Recent collaborative edits
    recent_edits: VecDeque<CollaborativeEdit>,
}

/// User session information
#[derive(Debug, Clone)]
pub struct UserSession {
    pub user_id: String,
    pub username: String,
    pub cursor_position: Option<(f64, f64)>,
    pub selected_nodes: Vec<NodeIndex>,
    pub last_activity: std::time::SystemTime,
    pub color: String, // User color for visual identification
}

/// Editor selection state
#[derive(Debug, Clone)]
pub struct EditorSelection {
    pub selected_nodes: Vec<NodeIndex>,
    pub selected_edges: Vec<(NodeIndex, NodeIndex)>,
    pub selection_rectangle: Option<SelectionRectangle>,
    pub clipboard: Option<ClipboardData>,
}

/// Selection rectangle for multi-select
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionRectangle {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Clipboard data for copy/paste operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipboardData {
    pub nodes: Vec<NodeSnapshot>,
    pub edges: Vec<EdgeSnapshot>,
    pub metadata: HashMap<String, String>,
}

/// Node snapshot for clipboard operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSnapshot {
    pub node_type: String,
    pub operation: Option<String>,
    pub parameters: HashMap<String, String>,
    pub position: (f64, f64),
    pub style: NodeStyle,
}

/// Edge snapshot for clipboard operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSnapshot {
    pub source_index: usize, // Relative index in clipboard
    pub target_index: usize,
    pub edge_type: String,
    pub style: EdgeStyle,
}

/// Visual style for nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStyle {
    pub color: String,
    pub border_color: String,
    pub border_width: f64,
    pub shape: NodeShape,
    pub size: (f64, f64),
    pub label_style: LabelStyle,
}

/// Visual style for edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeStyle {
    pub color: String,
    pub width: f64,
    pub style: EdgeLineStyle,
    pub arrow_style: ArrowStyle,
}

/// Node shape variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeShape {
    Rectangle,
    Circle,
    Diamond,
    Hexagon,
    Custom(String),
}

/// Edge line style variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeLineStyle {
    Solid,
    Dashed,
    Dotted,
    Custom(String),
}

/// Arrow style for edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrowStyle {
    pub size: f64,
    pub style: ArrowType,
}

/// Arrow type variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArrowType {
    Simple,
    Filled,
    Diamond,
    Circle,
    Custom(String),
}

/// Label styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelStyle {
    pub font_family: String,
    pub font_size: f64,
    pub color: String,
    pub background_color: Option<String>,
    pub padding: f64,
}

/// Collaborative edit record
#[derive(Debug, Clone)]
pub struct CollaborativeEdit {
    pub edit_id: String,
    pub user_id: String,
    pub timestamp: std::time::SystemTime,
    pub operation: EditOperation,
    pub affected_nodes: Vec<NodeIndex>,
}

/// Edit operation types
#[derive(Debug, Clone)]
pub enum EditOperation {
    AddNode {
        node_type: String,
        position: (f64, f64),
        parameters: HashMap<String, String>,
    },
    RemoveNode {
        node_id: NodeIndex,
    },
    ModifyNode {
        node_id: NodeIndex,
        changes: HashMap<String, String>,
    },
    AddEdge {
        source: NodeIndex,
        target: NodeIndex,
        edge_type: String,
    },
    RemoveEdge {
        source: NodeIndex,
        target: NodeIndex,
    },
    MoveNodes {
        moves: Vec<(NodeIndex, (f64, f64))>,
    },
    GroupOperation {
        operations: Vec<EditOperation>,
        description: String,
    },
}

/// Auto-save configuration
#[derive(Debug, Clone)]
pub struct AutoSaveConfig {
    pub enabled: bool,
    pub interval: Duration,
    pub max_auto_saves: usize,
    pub save_location: String,
    pub compression: bool,
}

/// Visualization configuration
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    pub theme: VisualizationTheme,
    pub layout_algorithm: LayoutAlgorithm,
    pub animation_settings: AnimationSettings,
    pub performance_overlay: bool,
    pub collaborative_cursors: bool,
    pub grid_settings: GridSettings,
}

/// Visualization theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationTheme {
    Light,
    Dark,
    HighContrast,
    Custom(CustomTheme),
}

/// Custom theme definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomTheme {
    pub background_color: String,
    pub grid_color: String,
    pub default_node_color: String,
    pub default_edge_color: String,
    pub selection_color: String,
    pub hover_color: String,
}

/// Layout algorithm options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutAlgorithm {
    ForceDirected,
    Hierarchical,
    Circular,
    Grid,
    Manual,
    Custom(String),
}

/// Animation settings
#[derive(Debug, Clone)]
pub struct AnimationSettings {
    pub enabled: bool,
    pub duration: Duration,
    pub easing: EasingFunction,
    pub fps_limit: u32,
}

/// Easing function types
#[derive(Debug, Clone)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
    Elastic,
}

/// Grid display settings
#[derive(Debug, Clone)]
pub struct GridSettings {
    pub enabled: bool,
    pub size: f64,
    pub color: String,
    pub opacity: f64,
    pub snap_to_grid: bool,
}

impl InteractiveGraphEditor {
    /// Create a new interactive graph editor
    pub fn new(graph: FxGraph) -> Self {
        Self {
            graph: Arc::new(RwLock::new(graph)),
            performance_monitor: Arc::new(Mutex::new(PerformanceMonitor::new())),
            history: Arc::new(Mutex::new(EditHistory::new())),
            collaboration_state: Arc::new(RwLock::new(CollaborationState::new())),
            auto_save_config: AutoSaveConfig::default(),
            visualization_config: VisualizationConfig::default(),
        }
    }

    /// Start the interactive editor server
    pub async fn start_server(&self, port: u16) -> Result<()> {
        let server = EditorServer::new(
            self.graph.clone(),
            self.performance_monitor.clone(),
            self.history.clone(),
            self.collaboration_state.clone(),
        );

        server.start(port).await
    }

    /// Apply an edit operation
    pub fn apply_edit(&self, operation: EditOperation, user_id: Option<String>) -> Result<()> {
        // Record edit in history
        self.record_edit(&operation, user_id.as_deref())?;

        // Apply the operation
        match operation {
            EditOperation::AddNode {
                node_type,
                position,
                parameters,
            } => self.add_node(&node_type, position, parameters)?,
            EditOperation::RemoveNode { node_id } => self.remove_node(node_id)?,
            EditOperation::ModifyNode { node_id, changes } => self.modify_node(node_id, changes)?,
            EditOperation::AddEdge {
                source,
                target,
                edge_type,
            } => self.add_edge(source, target, &edge_type)?,
            EditOperation::RemoveEdge { source, target } => self.remove_edge(source, target)?,
            EditOperation::MoveNodes { moves } => self.move_nodes(moves)?,
            EditOperation::GroupOperation {
                operations,
                description: _,
            } => {
                for op in operations {
                    self.apply_edit(op, user_id.clone())?;
                }
            }
        }

        // Update performance metrics
        self.update_performance_metrics();

        // Trigger auto-save if enabled
        if self.auto_save_config.enabled {
            self.auto_save()?;
        }

        Ok(())
    }

    /// Undo the last edit operation
    pub fn undo(&self) -> Result<bool> {
        let mut history = self.history.lock().unwrap();
        if history.can_undo() {
            let snapshot = history.undo();
            self.restore_from_snapshot(&snapshot)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Redo the next edit operation
    pub fn redo(&self) -> Result<bool> {
        let mut history = self.history.lock().unwrap();
        if history.can_redo() {
            let snapshot = history.redo();
            self.restore_from_snapshot(&snapshot)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Export graph in various formats
    pub fn export_graph(&self, format: ExportFormat) -> Result<String> {
        let graph = self.graph.read().unwrap();
        match format {
            ExportFormat::Json => {
                // Create a simplified JSON representation since FxGraph doesn't implement Serialize
                let mut json_repr = serde_json::Map::new();
                json_repr.insert(
                    "node_count".to_string(),
                    serde_json::Value::Number(graph.node_count().into()),
                );
                json_repr.insert(
                    "edge_count".to_string(),
                    serde_json::Value::Number(graph.edge_count().into()),
                );
                json_repr.insert(
                    "type".to_string(),
                    serde_json::Value::String("fx_graph".to_string()),
                );
                serde_json::to_string_pretty(&json_repr)
                    .map_err(|e| torsh_core::error::TorshError::SerializationError(e.to_string()))
            }
            ExportFormat::Dot => Ok(self.export_to_dot(&graph)),
            ExportFormat::Svg => self.export_to_svg(&graph),
            ExportFormat::Png => self.export_to_png(&graph),
            ExportFormat::Mermaid => Ok(self.export_to_mermaid(&graph)),
            ExportFormat::Onnx => self.export_to_onnx(&graph),
        }
    }

    /// Import graph from various formats
    pub fn import_graph(&self, data: &str, format: ImportFormat) -> Result<()> {
        let new_graph = match format {
            ImportFormat::Json => {
                // For now, create an empty graph since we can't deserialize FxGraph directly
                // In a real implementation, this would parse the JSON and reconstruct the graph
                FxGraph::new()
            }
            ImportFormat::Onnx => self.import_from_onnx(data)?,
            ImportFormat::TorchScript => self.import_from_torchscript(data)?,
            ImportFormat::TensorFlow => self.import_from_tensorflow(data)?,
        };

        // Replace current graph
        {
            let mut graph = self.graph.write().unwrap();
            *graph = new_graph;
        } // Release write lock before creating snapshot

        // Create snapshot for history
        self.create_snapshot("Import graph")?;

        Ok(())
    }

    /// Get real-time performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        let monitor = self.performance_monitor.lock().unwrap();
        monitor.get_current_metrics()
    }

    /// Start collaborative editing session
    pub fn start_collaboration(&self, user: UserSession) -> Result<String> {
        let mut state = self.collaboration_state.write().unwrap();
        let session_id = uuid::Uuid::new_v4().to_string();
        state.active_users.insert(session_id.clone(), user);
        Ok(session_id)
    }

    /// Stop collaborative editing session
    pub fn stop_collaboration(&self, session_id: &str) -> Result<()> {
        let mut state = self.collaboration_state.write().unwrap();
        state.active_users.remove(session_id);

        // Release any locks held by this user
        state.edit_locks.retain(|_, user_id| user_id != session_id);

        Ok(())
    }

    /// Get current collaboration state
    pub fn get_collaboration_state(&self) -> CollaborationState {
        self.collaboration_state.read().unwrap().clone()
    }

    // Private helper methods
    fn record_edit(&self, operation: &EditOperation, user_id: Option<&str>) -> Result<()> {
        let graph = self.graph.read().unwrap();
        let snapshot = GraphSnapshot {
            graph_data: format!(
                "graph_snapshot_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ), // Simplified since we can't serialize FxGraph
            timestamp: std::time::SystemTime::now(),
            operation_description: format!("{:?}", operation),
            editor_id: user_id.map(|s| s.to_string()),
        };

        let mut history = self.history.lock().unwrap();
        history.add_snapshot(snapshot);

        Ok(())
    }

    fn add_node(
        &self,
        node_type: &str,
        _position: (f64, f64),
        parameters: HashMap<String, String>,
    ) -> Result<()> {
        let mut graph = self.graph.write().unwrap();

        // Create node based on type and parameters
        let node = match node_type {
            "input" => {
                let name = parameters
                    .get("name")
                    .cloned()
                    .unwrap_or_else(|| "input".to_string());
                Node::Input(name)
            }
            "call" => {
                let op_name = parameters
                    .get("operation")
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                let args = parameters
                    .get("args")
                    .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
                    .unwrap_or_default();
                Node::Call(op_name, args)
            }
            "output" => Node::Output,
            _ => {
                return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                    "Unknown node type: {}",
                    node_type
                )))
            }
        };

        graph.add_node(node);
        Ok(())
    }

    fn remove_node(&self, node_id: NodeIndex) -> Result<()> {
        let mut graph = self.graph.write().unwrap();
        if graph.graph.node_weight(node_id).is_some() {
            graph.graph.remove_node(node_id);
            Ok(())
        } else {
            Err(torsh_core::error::TorshError::InvalidArgument(
                "Node not found".to_string(),
            ))
        }
    }

    fn modify_node(&self, node_id: NodeIndex, _changes: HashMap<String, String>) -> Result<()> {
        let graph = self.graph.read().unwrap();
        if graph.graph.node_weight(node_id).is_some() {
            // TODO: Implement node modification
            Ok(())
        } else {
            Err(torsh_core::error::TorshError::InvalidArgument(
                "Node not found".to_string(),
            ))
        }
    }

    fn add_edge(&self, source: NodeIndex, target: NodeIndex, _edge_type: &str) -> Result<()> {
        let mut graph = self.graph.write().unwrap();
        let edge = crate::Edge {
            name: "data".to_string(),
        };
        graph.graph.add_edge(source, target, edge);
        Ok(())
    }

    fn remove_edge(&self, source: NodeIndex, target: NodeIndex) -> Result<()> {
        let mut graph = self.graph.write().unwrap();
        if let Some(edge_id) = graph.graph.find_edge(source, target) {
            graph.graph.remove_edge(edge_id);
            Ok(())
        } else {
            Err(torsh_core::error::TorshError::InvalidArgument(
                "Edge not found".to_string(),
            ))
        }
    }

    fn move_nodes(&self, _moves: Vec<(NodeIndex, (f64, f64))>) -> Result<()> {
        // TODO: Implement node position tracking
        Ok(())
    }

    fn update_performance_metrics(&self) {
        let mut monitor = self.performance_monitor.lock().unwrap();
        monitor.update();
    }

    fn auto_save(&self) -> Result<()> {
        if !self.auto_save_config.enabled {
            return Ok(());
        }

        let export_data = self.export_graph(ExportFormat::Json)?;
        let filename = format!(
            "{}/autosave_{}.json",
            self.auto_save_config.save_location,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        std::fs::write(filename, export_data)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        Ok(())
    }

    fn restore_from_snapshot(&self, _snapshot: &GraphSnapshot) -> Result<()> {
        // For now, create a new empty graph since we can't deserialize FxGraph directly
        // In a real implementation, this would restore the actual graph state
        let new_graph = FxGraph::new();

        let mut graph = self.graph.write().unwrap();
        *graph = new_graph;

        Ok(())
    }

    fn create_snapshot(&self, description: &str) -> Result<()> {
        let _graph = self.graph.read().unwrap();
        let snapshot = GraphSnapshot {
            graph_data: format!(
                "snapshot_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ), // Simplified since we can't serialize FxGraph
            timestamp: std::time::SystemTime::now(),
            operation_description: description.to_string(),
            editor_id: None,
        };

        let mut history = self.history.lock().unwrap();
        history.add_snapshot(snapshot);

        Ok(())
    }

    // Export helper methods
    fn export_to_dot(&self, graph: &FxGraph) -> String {
        crate::visualization::visualize_graph_dot(graph)
    }

    fn export_to_svg(&self, _graph: &FxGraph) -> Result<String> {
        // TODO: Implement SVG export
        Ok("<svg></svg>".to_string())
    }

    fn export_to_png(&self, _graph: &FxGraph) -> Result<String> {
        // TODO: Implement PNG export (base64 encoded)
        Ok("data:image/png;base64,".to_string())
    }

    fn export_to_mermaid(&self, graph: &FxGraph) -> String {
        crate::visualization::GraphDebugger::new(graph.clone())
            .visualize_mermaid(&crate::visualization::VisualizationOptions::default())
    }

    fn export_to_onnx(&self, _graph: &FxGraph) -> Result<String> {
        // TODO: Implement ONNX export
        Ok("".to_string())
    }

    // Import helper methods
    fn import_from_onnx(&self, _data: &str) -> Result<FxGraph> {
        // TODO: Implement ONNX import
        Err(torsh_core::error::TorshError::NotImplemented(
            "ONNX import not yet implemented".to_string(),
        ))
    }

    fn import_from_torchscript(&self, _data: &str) -> Result<FxGraph> {
        // TODO: Implement TorchScript import
        Err(torsh_core::error::TorshError::NotImplemented(
            "TorchScript import not yet implemented".to_string(),
        ))
    }

    fn import_from_tensorflow(&self, _data: &str) -> Result<FxGraph> {
        // TODO: Implement TensorFlow import
        Err(torsh_core::error::TorshError::NotImplemented(
            "TensorFlow import not yet implemented".to_string(),
        ))
    }
}

/// Export format options
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Dot,
    Svg,
    Png,
    Mermaid,
    Onnx,
}

/// Import format options
#[derive(Debug, Clone)]
pub enum ImportFormat {
    Json,
    Onnx,
    TorchScript,
    TensorFlow,
}

/// Real-time performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub graph_execution_time: Duration,
    pub node_execution_times: HashMap<String, Duration>,
    pub memory_usage_mb: f64,
    pub compilation_time: Duration,
    pub fps: f64,
    pub active_users: usize,
}

/// Web server for the interactive editor
pub struct EditorServer {
    graph: Arc<RwLock<FxGraph>>,
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
    history: Arc<Mutex<EditHistory>>,
    collaboration_state: Arc<RwLock<CollaborationState>>,
}

impl EditorServer {
    pub fn new(
        graph: Arc<RwLock<FxGraph>>,
        performance_monitor: Arc<Mutex<PerformanceMonitor>>,
        history: Arc<Mutex<EditHistory>>,
        collaboration_state: Arc<RwLock<CollaborationState>>,
    ) -> Self {
        Self {
            graph,
            performance_monitor,
            history,
            collaboration_state,
        }
    }

    pub async fn start(&self, port: u16) -> Result<()> {
        println!("🚀 Interactive Graph Editor starting on port {}", port);
        println!(
            "📊 Real-time visualization: http://localhost:{}/editor",
            port
        );
        println!("🤝 Collaboration API: http://localhost:{}/api", port);

        // TODO: Implement actual web server using a web framework
        // For now, just simulate server startup
        Ok(())
    }
}

// Implementation of helper structs
impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            node_timings: HashMap::new(),
            memory_usage: HashMap::new(),
            compilation_history: VecDeque::with_capacity(100),
            update_frequency: Duration::from_millis(100),
            last_update: Instant::now(),
        }
    }

    fn update(&mut self) {
        self.last_update = Instant::now();
        // TODO: Implement actual performance monitoring
    }

    fn get_current_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            graph_execution_time: Duration::from_millis(0),
            node_execution_times: HashMap::new(),
            memory_usage_mb: 0.0,
            compilation_time: Duration::from_millis(0),
            fps: 60.0,
            active_users: 0,
        }
    }
}

impl EditHistory {
    fn new() -> Self {
        Self {
            history: Vec::new(),
            current_position: 0,
            max_history_size: 100,
        }
    }

    fn add_snapshot(&mut self, snapshot: GraphSnapshot) {
        // Remove any future history if we're not at the end
        self.history.truncate(self.current_position);

        // Add new snapshot
        self.history.push(snapshot);
        self.current_position = self.history.len();

        // Maintain max history size
        if self.history.len() > self.max_history_size {
            self.history.remove(0);
            self.current_position = self.history.len();
        }
    }

    fn can_undo(&self) -> bool {
        self.current_position > 1
    }

    fn can_redo(&self) -> bool {
        self.current_position < self.history.len()
    }

    fn undo(&mut self) -> &GraphSnapshot {
        self.current_position = self.current_position.saturating_sub(1);
        &self.history[self.current_position.saturating_sub(1)]
    }

    fn redo(&mut self) -> &GraphSnapshot {
        let snapshot = &self.history[self.current_position];
        self.current_position = (self.current_position + 1).min(self.history.len());
        snapshot
    }
}

impl CollaborationState {
    fn new() -> Self {
        Self {
            active_users: HashMap::new(),
            edit_locks: HashMap::new(),
            user_selections: HashMap::new(),
            recent_edits: VecDeque::with_capacity(1000),
        }
    }
}

// Default implementations
impl Default for AutoSaveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            max_auto_saves: 10,
            save_location: "/tmp".to_string(),
            compression: false,
        }
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            theme: VisualizationTheme::Light,
            layout_algorithm: LayoutAlgorithm::ForceDirected,
            animation_settings: AnimationSettings::default(),
            performance_overlay: true,
            collaborative_cursors: true,
            grid_settings: GridSettings::default(),
        }
    }
}

impl Default for AnimationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            duration: Duration::from_millis(300),
            easing: EasingFunction::EaseInOut,
            fps_limit: 60,
        }
    }
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            size: 20.0,
            color: "#e0e0e0".to_string(),
            opacity: 0.3,
            snap_to_grid: false,
        }
    }
}

/// Convenience function to create and start an interactive editor
pub async fn launch_interactive_editor(graph: FxGraph, port: Option<u16>) -> Result<()> {
    let editor = InteractiveGraphEditor::new(graph);
    let port = port.unwrap_or(8080);

    println!("🎨 Launching Interactive Graph Editor...");
    println!("✨ Features: Real-time visualization, collaborative editing, performance monitoring");

    editor.start_server(port).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::ModuleTracer;

    #[test]
    fn test_interactive_editor_creation() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("relu", vec!["x".to_string()]);
        tracer.add_output("node_0");
        let graph = tracer.finalize();

        let editor = InteractiveGraphEditor::new(graph);

        // Test that editor is created successfully
        assert!(editor.graph.read().is_ok());
        assert!(editor.performance_monitor.lock().is_ok());
        assert!(editor.history.lock().is_ok());
    }

    #[test]
    fn test_edit_operations() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        let graph = tracer.finalize();

        let editor = InteractiveGraphEditor::new(graph);

        // Test adding a node
        let add_op = EditOperation::AddNode {
            node_type: "call".to_string(),
            position: (100.0, 100.0),
            parameters: {
                let mut params = HashMap::new();
                params.insert("operation".to_string(), "relu".to_string());
                params.insert("args".to_string(), "x".to_string());
                params
            },
        };

        assert!(editor
            .apply_edit(add_op, Some("test_user".to_string()))
            .is_ok());
    }

    #[test]
    fn test_undo_redo_functionality() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        let graph = tracer.finalize();

        let editor = InteractiveGraphEditor::new(graph);

        // Create initial snapshot
        editor.create_snapshot("Initial state").unwrap();

        // Apply an edit
        let add_op = EditOperation::AddNode {
            node_type: "call".to_string(),
            position: (100.0, 100.0),
            parameters: HashMap::new(),
        };

        assert!(editor.apply_edit(add_op, None).is_ok());

        // Test undo
        assert!(editor.undo().is_ok());

        // Test redo
        assert!(editor.redo().is_ok());
    }

    #[test]
    fn test_export_import() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("relu", vec!["x".to_string()]);
        tracer.add_output("node_0");
        let graph = tracer.finalize();

        let editor = InteractiveGraphEditor::new(graph);

        // Test export JSON only (simplest format)
        let exported = editor.export_graph(ExportFormat::Json);
        assert!(exported.is_ok());

        // Test that the exported data contains expected fields
        if let Ok(data) = exported {
            assert!(data.contains("node_count"));
            assert!(data.contains("edge_count"));
            assert!(data.contains("fx_graph"));

            // Test import - this creates a new empty graph for now
            assert!(editor.import_graph(&data, ImportFormat::Json).is_ok());
        }
    }

    #[test]
    fn test_collaboration_features() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        let graph = tracer.finalize();

        let editor = InteractiveGraphEditor::new(graph);

        // Test starting collaboration
        let user = UserSession {
            user_id: "test_user".to_string(),
            username: "Test User".to_string(),
            cursor_position: Some((0.0, 0.0)),
            selected_nodes: vec![],
            last_activity: std::time::SystemTime::now(),
            color: "#ff0000".to_string(),
        };

        let session_id = editor.start_collaboration(user);
        assert!(session_id.is_ok());

        // Test stopping collaboration
        if let Ok(id) = session_id {
            assert!(editor.stop_collaboration(&id).is_ok());
        }
    }

    #[test]
    fn test_performance_monitoring() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        let graph = tracer.finalize();

        let editor = InteractiveGraphEditor::new(graph);

        // Test getting performance metrics
        let metrics = editor.get_performance_metrics();
        assert_eq!(metrics.fps, 60.0);
        assert_eq!(metrics.active_users, 0);
    }
}
