//! Plugin System for ToRSh JIT Compilation
//!
//! This module provides a dynamic plugin system that allows loading and
//! registering custom functionality at runtime. Plugins can provide:
//! - Custom operators
//! - Optimization passes
//! - Backend implementations
//! - Type systems
//! - Debug tools

use crate::{custom_ops::CustomOpBuilder, JitError, JitResult};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Plugin interface version for compatibility checking
pub const PLUGIN_API_VERSION: u32 = 1;

/// Plugin metadata
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    /// Plugin name
    pub name: String,

    /// Plugin version
    pub version: String,

    /// Plugin description
    pub description: String,

    /// Plugin author
    pub author: String,

    /// Required API version
    pub api_version: u32,

    /// Plugin dependencies
    pub dependencies: Vec<String>,

    /// Plugin capabilities
    pub capabilities: Vec<PluginCapability>,
}

/// Plugin capabilities
#[derive(Debug, Clone)]
pub enum PluginCapability {
    /// Provides custom operators
    CustomOperators,

    /// Provides optimization passes
    OptimizationPasses,

    /// Provides backend implementations
    BackendImplementation(String),

    /// Provides type systems
    TypeSystem,

    /// Provides debugging tools
    DebuggingTools,

    /// Custom capability
    Custom(String),
}

/// Plugin trait that all plugins must implement
pub trait Plugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Initialize the plugin
    fn initialize(&mut self, context: &PluginContext) -> JitResult<()>;

    /// Register plugin functionality
    fn register(&self, registry: &mut PluginRegistry) -> JitResult<()>;

    /// Cleanup when plugin is unloaded
    fn cleanup(&mut self) -> JitResult<()>;
}

/// Plugin context provided during initialization
#[derive(Debug)]
pub struct PluginContext {
    /// JIT compiler version
    pub jit_version: String,

    /// Available features
    pub features: Vec<String>,

    /// Configuration parameters
    pub config: HashMap<String, String>,
}

/// Dynamic library plugin wrapper
pub struct DynamicPlugin {
    /// Plugin metadata
    metadata: PluginMetadata,

    /// Dynamic library handle (conceptual - would use libloading in real implementation)
    _lib_handle: String,

    /// Plugin instance
    plugin: Box<dyn Plugin>,
}

impl DynamicPlugin {
    /// Load plugin from dynamic library
    pub fn load<P: AsRef<Path>>(path: P) -> JitResult<Self> {
        let path = path.as_ref();

        // In a real implementation, this would use libloading or similar
        // For now, we'll simulate it
        let lib_path = path.to_string_lossy().to_string();

        // Validate file exists and is a valid library
        if !path.exists() {
            return Err(JitError::RuntimeError(format!(
                "Plugin file not found: {}",
                path.display()
            )));
        }

        // Load library symbols (simulated)
        let metadata = Self::load_metadata(&lib_path)?;
        let plugin = Self::create_plugin_instance(&lib_path, &metadata)?;

        Ok(Self {
            metadata,
            _lib_handle: lib_path,
            plugin,
        })
    }

    /// Load plugin metadata from library
    fn load_metadata(lib_path: &str) -> JitResult<PluginMetadata> {
        // Simulated metadata loading
        // In real implementation, would load from library symbols
        let name = Path::new(lib_path)
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("unknown")
            .to_string();

        Ok(PluginMetadata {
            name,
            version: "1.0.0".to_string(),
            description: "Dynamically loaded plugin".to_string(),
            author: "Unknown".to_string(),
            api_version: PLUGIN_API_VERSION,
            dependencies: vec![],
            capabilities: vec![PluginCapability::CustomOperators],
        })
    }

    /// Create plugin instance from library
    fn create_plugin_instance(
        _lib_path: &str,
        metadata: &PluginMetadata,
    ) -> JitResult<Box<dyn Plugin>> {
        // Simulated plugin instantiation
        // In real implementation, would call library constructor
        Ok(Box::new(ExamplePlugin::new(metadata.clone())))
    }

    /// Get plugin metadata
    pub fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    /// Initialize plugin
    pub fn initialize(&mut self, context: &PluginContext) -> JitResult<()> {
        // Check API compatibility
        if self.metadata.api_version != PLUGIN_API_VERSION {
            return Err(JitError::RuntimeError(format!(
                "Plugin API version mismatch: expected {}, got {}",
                PLUGIN_API_VERSION, self.metadata.api_version
            )));
        }

        self.plugin.initialize(context)
    }

    /// Register plugin functionality
    pub fn register(&self, registry: &mut PluginRegistry) -> JitResult<()> {
        self.plugin.register(registry)
    }

    /// Cleanup plugin
    pub fn cleanup(&mut self) -> JitResult<()> {
        self.plugin.cleanup()
    }
}

/// Plugin registry for managing loaded plugins
pub struct PluginRegistry {
    /// Loaded plugins
    plugins: HashMap<String, DynamicPlugin>,

    /// Custom operator builders
    custom_op_builders: Vec<Box<dyn Fn() -> JitResult<CustomOpBuilder> + Send + Sync>>,

    /// Optimization pass factories
    optimization_passes: Vec<Box<dyn Fn() -> JitResult<Box<dyn OptimizationPass>> + Send + Sync>>,

    /// Backend implementations
    backend_impls: HashMap<String, Box<dyn Backend + Send + Sync>>,

    /// Plugin search paths
    search_paths: Vec<PathBuf>,
}

/// Optimization pass trait for plugins
pub trait OptimizationPass: Send + Sync {
    /// Pass name
    fn name(&self) -> &str;

    /// Apply optimization pass
    fn apply(&self, graph: &mut crate::ComputationGraph) -> JitResult<bool>;

    /// Pass dependencies (other passes that must run before this one)
    fn dependencies(&self) -> Vec<String>;
}

/// Backend trait for plugin backends
pub trait Backend: Send + Sync {
    /// Backend name
    fn name(&self) -> &str;

    /// Compile graph to backend-specific representation
    fn compile(&self, graph: &crate::ComputationGraph) -> JitResult<Box<dyn CompiledCode>>;

    /// Check if operation is supported
    fn supports_operation(&self, op: &crate::graph::Operation) -> bool;
}

/// Compiled code trait
pub trait CompiledCode: Send + Sync {
    /// Execute compiled code
    fn execute(&self, inputs: &[crate::TensorRef]) -> JitResult<Vec<crate::TensorRef>>;

    /// Get execution statistics
    fn stats(&self) -> ExecutionStats;
}

/// Execution statistics
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub execution_time: std::time::Duration,
    pub memory_usage: usize,
    pub operations_count: usize,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            custom_op_builders: Vec::new(),
            optimization_passes: Vec::new(),
            backend_impls: HashMap::new(),
            search_paths: vec![
                PathBuf::from("./plugins"),
                PathBuf::from("/usr/local/lib/torsh/plugins"),
                PathBuf::from("~/.torsh/plugins"),
            ],
        }
    }

    /// Add plugin search path
    pub fn add_search_path<P: AsRef<Path>>(&mut self, path: P) {
        self.search_paths.push(path.as_ref().to_path_buf());
    }

    /// Load plugin from file
    pub fn load_plugin<P: AsRef<Path>>(&mut self, path: P) -> JitResult<()> {
        let mut plugin = DynamicPlugin::load(path)?;

        let context = PluginContext {
            jit_version: "0.1.0".to_string(),
            features: vec!["custom_ops".to_string(), "optimization".to_string()],
            config: HashMap::new(),
        };

        plugin.initialize(&context)?;
        plugin.register(self)?;

        let plugin_name = plugin.metadata().name.clone();
        self.plugins.insert(plugin_name, plugin);

        Ok(())
    }

    /// Load all plugins from search paths
    pub fn load_all_plugins(&mut self) -> JitResult<Vec<String>> {
        let mut loaded_plugins = Vec::new();

        for search_path in &self.search_paths.clone() {
            if let Ok(entries) = std::fs::read_dir(search_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if self.is_plugin_file(&path) {
                        match self.load_plugin(&path) {
                            Ok(()) => {
                                if let Some(filename) = path.file_name() {
                                    loaded_plugins.push(filename.to_string_lossy().to_string());
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to load plugin {}: {}", path.display(), e);
                            }
                        }
                    }
                }
            }
        }

        Ok(loaded_plugins)
    }

    /// Check if file is a plugin
    fn is_plugin_file(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            match extension.to_str() {
                Some("so") | Some("dll") | Some("dylib") => true,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Find plugin by name
    pub fn find_plugin(&self, name: &str) -> Option<&DynamicPlugin> {
        self.plugins.get(name)
    }

    /// Unload plugin
    pub fn unload_plugin(&mut self, name: &str) -> JitResult<()> {
        if let Some(mut plugin) = self.plugins.remove(name) {
            plugin.cleanup()?;
        }
        Ok(())
    }

    /// List loaded plugins
    pub fn list_plugins(&self) -> Vec<&PluginMetadata> {
        self.plugins.values().map(|p| p.metadata()).collect()
    }

    /// Register custom operator builder
    pub fn register_custom_op_builder<F>(&mut self, builder: F)
    where
        F: Fn() -> JitResult<CustomOpBuilder> + Send + Sync + 'static,
    {
        self.custom_op_builders.push(Box::new(builder));
    }

    /// Register optimization pass
    pub fn register_optimization_pass<F>(&mut self, factory: F)
    where
        F: Fn() -> JitResult<Box<dyn OptimizationPass>> + Send + Sync + 'static,
    {
        self.optimization_passes.push(Box::new(factory));
    }

    /// Register backend implementation
    pub fn register_backend(&mut self, backend: Box<dyn Backend + Send + Sync>) {
        let name = backend.name().to_string();
        self.backend_impls.insert(name, backend);
    }

    /// Get custom operator builders
    pub fn get_custom_op_builders(
        &self,
    ) -> &[Box<dyn Fn() -> JitResult<CustomOpBuilder> + Send + Sync>] {
        &self.custom_op_builders
    }

    /// Get optimization passes
    pub fn get_optimization_passes(
        &self,
    ) -> &[Box<dyn Fn() -> JitResult<Box<dyn OptimizationPass>> + Send + Sync>] {
        &self.optimization_passes
    }

    /// Get backend implementation
    pub fn get_backend(&self, name: &str) -> Option<&(dyn Backend + Send + Sync)> {
        self.backend_impls.get(name).map(|b| b.as_ref())
    }

    /// List available backends
    pub fn list_backends(&self) -> Vec<&str> {
        self.backend_impls.keys().map(|s| s.as_str()).collect()
    }
}

// Global plugin registry (documentation on accessor function below)
lazy_static::lazy_static! {
    static ref GLOBAL_REGISTRY: Arc<RwLock<PluginRegistry>> =
        Arc::new(RwLock::new(PluginRegistry::new()));
}

/// Get global plugin registry
pub fn global_registry() -> Arc<RwLock<PluginRegistry>> {
    GLOBAL_REGISTRY.clone()
}

/// Load plugin into global registry
pub fn load_plugin<P: AsRef<Path>>(path: P) -> JitResult<()> {
    let binding = global_registry();
    let mut registry = binding
        .write()
        .map_err(|_| JitError::RuntimeError("Failed to acquire registry lock".to_string()))?;
    registry.load_plugin(path)
}

/// Load all plugins from search paths
pub fn load_all_plugins() -> JitResult<Vec<String>> {
    let binding = global_registry();
    let mut registry = binding
        .write()
        .map_err(|_| JitError::RuntimeError("Failed to acquire registry lock".to_string()))?;
    registry.load_all_plugins()
}

/// Plugin manager for high-level plugin operations
pub struct PluginManager {
    registry: Arc<RwLock<PluginRegistry>>,
    auto_load: bool,
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            registry: global_registry(),
            auto_load: true,
        }
    }

    /// Create plugin manager with custom registry
    pub fn with_registry(registry: Arc<RwLock<PluginRegistry>>) -> Self {
        Self {
            registry,
            auto_load: true,
        }
    }

    /// Enable/disable auto-loading plugins
    pub fn set_auto_load(&mut self, auto_load: bool) {
        self.auto_load = auto_load;
    }

    /// Initialize plugin system
    pub fn initialize(&self) -> JitResult<()> {
        if self.auto_load {
            self.load_all_plugins()?;
        }
        Ok(())
    }

    /// Load plugin
    pub fn load_plugin<P: AsRef<Path>>(&self, path: P) -> JitResult<()> {
        let mut registry = self
            .registry
            .write()
            .map_err(|_| JitError::RuntimeError("Failed to acquire registry lock".to_string()))?;
        registry.load_plugin(path)
    }

    /// Load all plugins
    pub fn load_all_plugins(&self) -> JitResult<Vec<String>> {
        let mut registry = self
            .registry
            .write()
            .map_err(|_| JitError::RuntimeError("Failed to acquire registry lock".to_string()))?;
        registry.load_all_plugins()
    }

    /// Unload plugin
    pub fn unload_plugin(&self, name: &str) -> JitResult<()> {
        let mut registry = self
            .registry
            .write()
            .map_err(|_| JitError::RuntimeError("Failed to acquire registry lock".to_string()))?;
        registry.unload_plugin(name)
    }

    /// List plugins
    pub fn list_plugins(&self) -> Vec<PluginMetadata> {
        match self.registry.read() {
            Ok(registry) => registry.list_plugins().into_iter().cloned().collect(),
            Err(_) => vec![],
        }
    }

    /// Get plugin info
    pub fn get_plugin_info(&self, name: &str) -> Option<PluginMetadata> {
        let registry = self.registry.read().ok()?;
        registry.find_plugin(name).map(|p| p.metadata().clone())
    }

    /// Check if plugin is loaded
    pub fn is_plugin_loaded(&self, name: &str) -> bool {
        match self.registry.read() {
            Ok(registry) => registry.find_plugin(name).is_some(),
            Err(_) => false,
        }
    }
}

/// Example plugin implementation
pub struct ExamplePlugin {
    metadata: PluginMetadata,
    initialized: bool,
}

impl ExamplePlugin {
    pub fn new(metadata: PluginMetadata) -> Self {
        Self {
            metadata,
            initialized: false,
        }
    }
}

impl Plugin for ExamplePlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn initialize(&mut self, _context: &PluginContext) -> JitResult<()> {
        self.initialized = true;
        Ok(())
    }

    fn register(&self, registry: &mut PluginRegistry) -> JitResult<()> {
        if !self.initialized {
            return Err(JitError::RuntimeError("Plugin not initialized".to_string()));
        }

        // Register a sample custom operator
        registry.register_custom_op_builder(|| {
            Ok(CustomOpBuilder::new("plugin_add")
                .namespace("example")
                .forward(|inputs| {
                    if inputs.len() != 2 {
                        return Err(JitError::RuntimeError(
                            "plugin_add requires 2 inputs".to_string(),
                        ));
                    }

                    let a = &inputs[0];
                    let b = &inputs[1];
                    let mut result = a.clone();

                    for (i, &val_b) in b.data.iter().enumerate() {
                        if i < result.data.len() {
                            result.data[i] += val_b;
                        }
                    }

                    Ok(vec![result])
                })
                .vectorizable(true)
                .parallelizable(true)
                .elementwise(true))
        });

        Ok(())
    }

    fn cleanup(&mut self) -> JitResult<()> {
        self.initialized = false;
        Ok(())
    }
}

/// Plugin discovery utilities
pub mod discovery {
    use super::*;

    /// Discover plugins in a directory
    pub fn discover_plugins<P: AsRef<Path>>(path: P) -> JitResult<Vec<PathBuf>> {
        let mut plugins = Vec::new();
        let path = path.as_ref();

        if !path.exists() {
            return Ok(plugins);
        }

        for entry in std::fs::read_dir(path)
            .map_err(|e| JitError::RuntimeError(format!("Failed to read directory: {}", e)))?
        {
            let entry = entry
                .map_err(|e| JitError::RuntimeError(format!("Failed to read entry: {}", e)))?;
            let path = entry.path();

            if is_plugin_file(&path) {
                plugins.push(path);
            }
        }

        Ok(plugins)
    }

    /// Check if file is a plugin
    fn is_plugin_file(path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            matches!(extension.to_str(), Some("so") | Some("dll") | Some("dylib"))
        } else {
            false
        }
    }

    /// Validate plugin compatibility
    pub fn validate_plugin(metadata: &PluginMetadata) -> JitResult<()> {
        if metadata.api_version != PLUGIN_API_VERSION {
            return Err(JitError::RuntimeError(format!(
                "Incompatible plugin API version: expected {}, got {}",
                PLUGIN_API_VERSION, metadata.api_version
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_metadata() {
        let metadata = PluginMetadata {
            name: "test_plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Test plugin".to_string(),
            author: "Test Author".to_string(),
            api_version: PLUGIN_API_VERSION,
            dependencies: vec![],
            capabilities: vec![PluginCapability::CustomOperators],
        };

        assert_eq!(metadata.name, "test_plugin");
        assert_eq!(metadata.api_version, PLUGIN_API_VERSION);
    }

    #[test]
    fn test_plugin_registry() {
        let mut registry = PluginRegistry::new();

        // Test initial state
        assert_eq!(registry.list_plugins().len(), 0);
        assert_eq!(registry.list_backends().len(), 0);

        // Test adding search path
        registry.add_search_path("/tmp/plugins");
        assert_eq!(registry.search_paths.len(), 4); // 3 default + 1 added
    }

    #[test]
    fn test_example_plugin() {
        let metadata = PluginMetadata {
            name: "example".to_string(),
            version: "1.0.0".to_string(),
            description: "Example plugin".to_string(),
            author: "Test".to_string(),
            api_version: PLUGIN_API_VERSION,
            dependencies: vec![],
            capabilities: vec![PluginCapability::CustomOperators],
        };

        let mut plugin = ExamplePlugin::new(metadata);
        assert!(!plugin.initialized);

        let context = PluginContext {
            jit_version: "0.1.0".to_string(),
            features: vec![],
            config: HashMap::new(),
        };

        assert!(plugin.initialize(&context).is_ok());
        assert!(plugin.initialized);

        let mut registry = PluginRegistry::new();
        assert!(plugin.register(&mut registry).is_ok());
        assert_eq!(registry.custom_op_builders.len(), 1);
    }

    #[test]
    fn test_plugin_manager() {
        let manager = PluginManager::new();
        assert!(manager.auto_load);

        let plugins = manager.list_plugins();
        assert!(plugins.is_empty()); // No plugins loaded initially
    }
}
