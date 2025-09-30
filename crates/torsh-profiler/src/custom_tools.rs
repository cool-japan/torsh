//! Custom tool APIs for profiler integration
//!
//! This module provides extensible APIs for integrating custom profiling tools
//! and frameworks into the torsh-profiler ecosystem.

use crate::ProfileEvent;
use anyhow::Result;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Instant, SystemTime};

/// Global registry for custom profiling tools
static CUSTOM_TOOL_REGISTRY: Lazy<Mutex<CustomToolRegistry>> =
    Lazy::new(|| Mutex::new(CustomToolRegistry::new()));

/// Custom tool integration trait
pub trait CustomTool: Send + Sync {
    /// Initialize the custom tool
    fn initialize(&mut self, config: &ToolConfig) -> Result<()>;

    /// Start profiling session
    fn start_session(&mut self, session_id: &str) -> Result<()>;

    /// Record a profiling event
    fn record_event(&mut self, event: &ProfileEvent) -> Result<()>;

    /// Stop profiling session
    fn stop_session(&mut self, session_id: &str) -> Result<()>;

    /// Export collected data
    fn export_data(&self, format: &ExportFormat) -> Result<Vec<u8>>;

    /// Get tool statistics
    fn get_statistics(&self) -> Result<ToolStatistics>;

    /// Cleanup resources
    fn cleanup(&mut self) -> Result<()>;

    /// Get tool name
    fn name(&self) -> &str;

    /// Get tool version
    fn version(&self) -> &str;
}

/// Configuration for custom tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    pub name: String,
    pub version: String,
    pub enabled: bool,
    pub sampling_rate: f64,
    pub buffer_size: usize,
    pub output_path: Option<String>,
    pub custom_options: HashMap<String, String>,
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            name: "custom_tool".to_string(),
            version: "1.0.0".to_string(),
            enabled: true,
            sampling_rate: 1.0,
            buffer_size: 1024,
            output_path: None,
            custom_options: HashMap::new(),
        }
    }
}

/// Export format for custom tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
    Binary,
    Custom(String),
}

/// Tool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolStatistics {
    pub events_recorded: u64,
    pub sessions_completed: u64,
    pub total_duration_ns: u64,
    pub memory_usage_bytes: u64,
    pub error_count: u64,
    pub last_updated: SystemTime,
}

impl Default for ToolStatistics {
    fn default() -> Self {
        Self {
            events_recorded: 0,
            sessions_completed: 0,
            total_duration_ns: 0,
            memory_usage_bytes: 0,
            error_count: 0,
            last_updated: SystemTime::now(),
        }
    }
}

/// Registry for managing custom tools
#[derive(Default)]
pub struct CustomToolRegistry {
    tools: HashMap<String, Box<dyn CustomTool>>,
    configs: HashMap<String, ToolConfig>,
}

impl CustomToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            configs: HashMap::new(),
        }
    }

    /// Register a custom tool
    pub fn register_tool(&mut self, tool: Box<dyn CustomTool>, config: ToolConfig) -> Result<()> {
        let name = tool.name().to_string();
        self.configs.insert(name.clone(), config);
        self.tools.insert(name, tool);
        Ok(())
    }

    /// Unregister a custom tool
    pub fn unregister_tool(&mut self, name: &str) -> Result<()> {
        if let Some(mut tool) = self.tools.remove(name) {
            tool.cleanup()?;
        }
        self.configs.remove(name);
        Ok(())
    }

    /// Get tool by name
    pub fn get_tool(&mut self, name: &str) -> Option<&mut Box<dyn CustomTool>> {
        self.tools.get_mut(name)
    }

    /// Initialize all tools
    pub fn initialize_all(&mut self) -> Result<()> {
        for (name, tool) in &mut self.tools {
            if let Some(config) = self.configs.get(name) {
                tool.initialize(config)?;
            }
        }
        Ok(())
    }

    /// Start session on all tools
    pub fn start_session_all(&mut self, session_id: &str) -> Result<()> {
        for tool in self.tools.values_mut() {
            tool.start_session(session_id)?;
        }
        Ok(())
    }

    /// Record event on all tools
    pub fn record_event_all(&mut self, event: &ProfileEvent) -> Result<()> {
        for tool in self.tools.values_mut() {
            tool.record_event(event)?;
        }
        Ok(())
    }

    /// Stop session on all tools
    pub fn stop_session_all(&mut self, session_id: &str) -> Result<()> {
        for tool in self.tools.values_mut() {
            tool.stop_session(session_id)?;
        }
        Ok(())
    }

    /// Get all tool statistics
    pub fn get_all_statistics(&self) -> HashMap<String, ToolStatistics> {
        let mut stats = HashMap::new();
        for (name, tool) in &self.tools {
            if let Ok(tool_stats) = tool.get_statistics() {
                stats.insert(name.clone(), tool_stats);
            }
        }
        stats
    }

    /// List all registered tools
    pub fn list_tools(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
}

/// Example custom tool implementation
pub struct ExampleCustomTool {
    name: String,
    version: String,
    statistics: ToolStatistics,
    events: Vec<ProfileEvent>,
    active_sessions: Vec<String>,
}

impl ExampleCustomTool {
    pub fn new(name: String, version: String) -> Self {
        Self {
            name,
            version,
            statistics: ToolStatistics::default(),
            events: Vec::new(),
            active_sessions: Vec::new(),
        }
    }
}

impl CustomTool for ExampleCustomTool {
    fn initialize(&mut self, config: &ToolConfig) -> Result<()> {
        println!("Initializing custom tool: {}", config.name);
        Ok(())
    }

    fn start_session(&mut self, session_id: &str) -> Result<()> {
        self.active_sessions.push(session_id.to_string());
        Ok(())
    }

    fn record_event(&mut self, event: &ProfileEvent) -> Result<()> {
        self.events.push(event.clone());
        self.statistics.events_recorded += 1;
        self.statistics.last_updated = SystemTime::now();
        Ok(())
    }

    fn stop_session(&mut self, session_id: &str) -> Result<()> {
        self.active_sessions.retain(|s| s != session_id);
        self.statistics.sessions_completed += 1;
        Ok(())
    }

    fn export_data(&self, format: &ExportFormat) -> Result<Vec<u8>> {
        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&self.events)?;
                Ok(json.into_bytes())
            }
            ExportFormat::Csv => {
                let mut csv = String::new();
                csv.push_str("timestamp,name,duration_ns,thread_id\n");
                for event in &self.events {
                    csv.push_str(&format!(
                        "{},{},{},{}\n",
                        event.start_us,
                        event.name,
                        event.duration_us * 1000, // Convert to ns
                        event.thread_id as u64
                    ));
                }
                Ok(csv.into_bytes())
            }
            _ => Err(anyhow::anyhow!("Unsupported export format")),
        }
    }

    fn get_statistics(&self) -> Result<ToolStatistics> {
        Ok(self.statistics.clone())
    }

    fn cleanup(&mut self) -> Result<()> {
        self.events.clear();
        self.active_sessions.clear();
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }
}

/// External tool bridge for integrating third-party tools
pub struct ExternalToolBridge {
    name: String,
    version: String,
    command: String,
    args: Vec<String>,
    statistics: ToolStatistics,
}

impl ExternalToolBridge {
    pub fn new(name: String, version: String, command: String, args: Vec<String>) -> Self {
        Self {
            name,
            version,
            command,
            args,
            statistics: ToolStatistics::default(),
        }
    }
}

impl CustomTool for ExternalToolBridge {
    fn initialize(&mut self, config: &ToolConfig) -> Result<()> {
        // Initialize external tool process
        std::process::Command::new(&self.command)
            .args(&self.args)
            .arg("--initialize")
            .output()?;
        Ok(())
    }

    fn start_session(&mut self, session_id: &str) -> Result<()> {
        std::process::Command::new(&self.command)
            .args(&self.args)
            .arg("--start-session")
            .arg(session_id)
            .output()?;
        Ok(())
    }

    fn record_event(&mut self, event: &ProfileEvent) -> Result<()> {
        let event_json = serde_json::to_string(event)?;
        std::process::Command::new(&self.command)
            .args(&self.args)
            .arg("--record-event")
            .arg(&event_json)
            .output()?;
        self.statistics.events_recorded += 1;
        Ok(())
    }

    fn stop_session(&mut self, session_id: &str) -> Result<()> {
        std::process::Command::new(&self.command)
            .args(&self.args)
            .arg("--stop-session")
            .arg(session_id)
            .output()?;
        self.statistics.sessions_completed += 1;
        Ok(())
    }

    fn export_data(&self, format: &ExportFormat) -> Result<Vec<u8>> {
        let format_str = match format {
            ExportFormat::Json => "json",
            ExportFormat::Csv => "csv",
            ExportFormat::Binary => "binary",
            ExportFormat::Custom(fmt) => fmt,
        };

        let output = std::process::Command::new(&self.command)
            .args(&self.args)
            .arg("--export")
            .arg(format_str)
            .output()?;

        Ok(output.stdout)
    }

    fn get_statistics(&self) -> Result<ToolStatistics> {
        Ok(self.statistics.clone())
    }

    fn cleanup(&mut self) -> Result<()> {
        std::process::Command::new(&self.command)
            .args(&self.args)
            .arg("--cleanup")
            .output()?;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }
}

/// Public API functions for custom tool integration
/// Register a custom tool with the global registry
pub fn register_custom_tool(tool: Box<dyn CustomTool>, config: ToolConfig) -> Result<()> {
    CUSTOM_TOOL_REGISTRY.lock().register_tool(tool, config)
}

/// Unregister a custom tool from the global registry
pub fn unregister_custom_tool(name: &str) -> Result<()> {
    CUSTOM_TOOL_REGISTRY.lock().unregister_tool(name)
}

/// Initialize all registered custom tools
pub fn initialize_custom_tools() -> Result<()> {
    CUSTOM_TOOL_REGISTRY.lock().initialize_all()
}

/// Start profiling session on all custom tools
pub fn start_custom_tool_session(session_id: &str) -> Result<()> {
    CUSTOM_TOOL_REGISTRY.lock().start_session_all(session_id)
}

/// Record event on all custom tools
pub fn record_custom_tool_event(event: &ProfileEvent) -> Result<()> {
    CUSTOM_TOOL_REGISTRY.lock().record_event_all(event)
}

/// Stop profiling session on all custom tools
pub fn stop_custom_tool_session(session_id: &str) -> Result<()> {
    CUSTOM_TOOL_REGISTRY.lock().stop_session_all(session_id)
}

/// Get statistics from all custom tools
pub fn get_custom_tool_statistics() -> HashMap<String, ToolStatistics> {
    CUSTOM_TOOL_REGISTRY.lock().get_all_statistics()
}

/// List all registered custom tools
pub fn list_custom_tools() -> Vec<String> {
    CUSTOM_TOOL_REGISTRY.lock().list_tools()
}

/// Create an example custom tool
pub fn create_example_tool(name: String, version: String) -> Box<dyn CustomTool> {
    Box::new(ExampleCustomTool::new(name, version))
}

/// Create an external tool bridge
pub fn create_external_tool_bridge(
    name: String,
    version: String,
    command: String,
    args: Vec<String>,
) -> Box<dyn CustomTool> {
    Box::new(ExternalToolBridge::new(name, version, command, args))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_custom_tool_registration() {
        let mut registry = CustomToolRegistry::new();
        let tool = create_example_tool("test_tool".to_string(), "1.0.0".to_string());
        let config = ToolConfig::default();

        assert!(registry.register_tool(tool, config).is_ok());
        assert_eq!(registry.list_tools(), vec!["test_tool"]);
    }

    #[test]
    fn test_tool_lifecycle() {
        let mut registry = CustomToolRegistry::new();
        let tool = create_example_tool("test_tool".to_string(), "1.0.0".to_string());
        let config = ToolConfig::default();

        registry.register_tool(tool, config).unwrap();
        registry.initialize_all().unwrap();
        registry.start_session_all("test_session").unwrap();

        let event = ProfileEvent {
            name: "test_event".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 1,
            thread_id: 0,
            operation_count: Some(1),
            flops: Some(100),
            bytes_transferred: Some(1024),
            stack_trace: Some("test trace".to_string()),
        };

        registry.record_event_all(&event).unwrap();
        registry.stop_session_all("test_session").unwrap();

        let stats = registry.get_all_statistics();
        assert_eq!(stats.get("test_tool").unwrap().events_recorded, 1);
        assert_eq!(stats.get("test_tool").unwrap().sessions_completed, 1);
    }

    #[test]
    fn test_example_tool_export() {
        let mut tool = ExampleCustomTool::new("test".to_string(), "1.0.0".to_string());
        let config = ToolConfig::default();

        tool.initialize(&config).unwrap();
        tool.start_session("test").unwrap();

        let event = ProfileEvent {
            name: "test_event".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 1,
            thread_id: 0,
            operation_count: Some(1),
            flops: Some(100),
            bytes_transferred: Some(1024),
            stack_trace: Some("test trace".to_string()),
        };

        tool.record_event(&event).unwrap();

        let json_data = tool.export_data(&ExportFormat::Json).unwrap();
        assert!(!json_data.is_empty());

        let csv_data = tool.export_data(&ExportFormat::Csv).unwrap();
        assert!(!csv_data.is_empty());

        tool.stop_session("test").unwrap();
        tool.cleanup().unwrap();
    }

    #[test]
    fn test_tool_config_serialization() {
        let config = ToolConfig {
            name: "test_tool".to_string(),
            version: "1.0.0".to_string(),
            enabled: true,
            sampling_rate: 0.5,
            buffer_size: 2048,
            output_path: Some("/tmp/test".to_string()),
            custom_options: {
                let mut opts = HashMap::new();
                opts.insert("key1".to_string(), "value1".to_string());
                opts.insert("key2".to_string(), "value2".to_string());
                opts
            },
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ToolConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.name, deserialized.name);
        assert_eq!(config.version, deserialized.version);
        assert_eq!(config.enabled, deserialized.enabled);
        assert_eq!(config.sampling_rate, deserialized.sampling_rate);
        assert_eq!(config.buffer_size, deserialized.buffer_size);
        assert_eq!(config.output_path, deserialized.output_path);
        assert_eq!(config.custom_options, deserialized.custom_options);
    }

    #[test]
    fn test_global_api_functions() {
        let tool = create_example_tool("global_test".to_string(), "1.0.0".to_string());
        let config = ToolConfig::default();

        assert!(register_custom_tool(tool, config).is_ok());
        assert!(list_custom_tools().contains(&"global_test".to_string()));

        assert!(initialize_custom_tools().is_ok());
        assert!(start_custom_tool_session("global_session").is_ok());

        let event = ProfileEvent {
            name: "global_event".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 1,
            thread_id: 0,
            operation_count: Some(1),
            flops: Some(100),
            bytes_transferred: Some(1024),
            stack_trace: Some("test trace".to_string()),
        };

        assert!(record_custom_tool_event(&event).is_ok());
        assert!(stop_custom_tool_session("global_session").is_ok());

        let stats = get_custom_tool_statistics();
        assert!(stats.contains_key("global_test"));

        assert!(unregister_custom_tool("global_test").is_ok());
    }
}
