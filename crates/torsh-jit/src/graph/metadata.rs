//! Graph metadata and additional information structures

use std::collections::HashMap;

/// Graph metadata containing additional information about the computation graph
#[derive(Debug, Clone, Default)]
pub struct GraphMetadata {
    /// Graph name
    pub name: String,

    /// Graph version
    pub version: String,

    /// Creator information
    pub creator: String,

    /// Compilation target
    pub target: Option<String>,

    /// Optimization level
    pub optimization_level: OptimizationLevel,

    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

impl GraphMetadata {
    /// Create new metadata with the given name
    pub fn new(name: String) -> Self {
        Self {
            name,
            version: "1.0.0".to_string(),
            creator: "torsh-jit".to_string(),
            target: None,
            optimization_level: OptimizationLevel::Default,
            custom: HashMap::new(),
        }
    }

    /// Set version
    pub fn with_version(mut self, version: String) -> Self {
        self.version = version;
        self
    }

    /// Set creator
    pub fn with_creator(mut self, creator: String) -> Self {
        self.creator = creator;
        self
    }

    /// Set target
    pub fn with_target(mut self, target: String) -> Self {
        self.target = Some(target);
        self
    }

    /// Set optimization level
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Add custom metadata
    pub fn with_custom(mut self, key: String, value: String) -> Self {
        self.custom.insert(key, value);
        self
    }

    /// Get custom metadata value
    pub fn get_custom(&self, key: &str) -> Option<&String> {
        self.custom.get(key)
    }
}

/// Optimization levels for graph compilation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,       // No optimizations
    Basic,      // Basic optimizations (constant folding, dead code elimination)
    Default,    // Default optimization level
    Aggressive, // Aggressive optimizations that may change semantics
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        OptimizationLevel::Default
    }
}
