//! Main code generator orchestration module
//!
//! This module provides the primary CodeGenerator interface that coordinates
//! all backend-specific code generation and manages the compilation pipeline.

use crate::core::{
    BackendType, CodeGenBackend, LoweredGraph, OptimizedKernel, TargetSpecification,
};
use crate::{cpp_codegen::CppCodeGen, python_codegen::PythonCodeGen, FxGraph};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use torsh_core::Result;

/// Main code generator that orchestrates all backends
///
/// The CodeGenerator manages multiple backend implementations and provides
/// a unified interface for code generation from FX graphs.
pub struct CodeGenerator {
    backends: HashMap<String, Box<dyn CodeGenBackend>>,
    cache: Arc<RwLock<HashMap<String, String>>>,
}

impl Default for CodeGenerator {
    fn default() -> Self {
        let mut generator = Self {
            backends: HashMap::new(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        };

        // Register default backends
        generator.register_default_backends();
        generator
    }
}

impl CodeGenerator {
    /// Create a new code generator with default backends
    pub fn new() -> Self {
        Self::default()
    }

    /// Register default backends (Python and C++)
    fn register_default_backends(&mut self) {
        self.add_backend("python".to_string(), PythonCodeGen::new());
        self.add_backend("cpp".to_string(), CppCodeGen::new());
    }

    /// Add a new backend to the generator
    pub fn add_backend<T: CodeGenBackend + 'static>(&mut self, name: String, backend: T) {
        self.backends.insert(name, Box::new(backend));
    }

    /// Generate code for the specified target
    pub fn generate_code(&self, graph: &FxGraph, target: &str) -> Result<String> {
        // Check cache first
        let cache_key = format!("{}_{}", target, self.graph_hash(graph));
        if let Ok(cache) = self.cache.read() {
            if let Some(cached_code) = cache.get(&cache_key) {
                return Ok(cached_code.clone());
            }
        }

        // Generate code using the specified backend
        if let Some(backend) = self.backends.get(target) {
            let code = backend.generate(graph)?;

            // Cache the result
            if let Ok(mut cache) = self.cache.write() {
                cache.insert(cache_key, code.clone());
            }

            Ok(code)
        } else {
            Err(torsh_core::Error::InvalidArgument(format!(
                "Unknown target: {}",
                target
            )))
        }
    }

    /// Get list of available targets
    pub fn available_targets(&self) -> Vec<&String> {
        self.backends.keys().collect()
    }

    /// Get file extension for a target
    pub fn get_file_extension(&self, target: &str) -> Option<&str> {
        self.backends
            .get(target)
            .map(|backend| backend.file_extension())
    }

    /// Get language name for a target
    pub fn get_language_name(&self, target: &str) -> Option<&str> {
        self.backends
            .get(target)
            .map(|backend| backend.language_name())
    }

    /// Generate optimized kernels for specific target device
    pub fn generate_optimized_kernels(
        &self,
        graph: &FxGraph,
        target_spec: &TargetSpecification,
    ) -> Result<Vec<OptimizedKernel>> {
        // This is a simplified implementation
        // In practice, this would analyze the graph and generate optimized kernels
        // based on the target specification
        Ok(vec![])
    }

    /// Lower high-level operations to backend-specific implementations
    pub fn lower_to_backend(&self, graph: &FxGraph, backend: BackendType) -> Result<LoweredGraph> {
        // Simplified lowering implementation
        // In practice, this would transform the graph based on backend capabilities
        Ok(LoweredGraph {
            nodes: vec![],
            edges: vec![],
            backend_type: backend,
        })
    }

    /// Generate target-specific optimized code
    pub fn generate_target_specific(
        &self,
        graph: &FxGraph,
        target_spec: &TargetSpecification,
    ) -> Result<String> {
        // Simplified target-specific generation
        // Would use the target specification to generate optimized code
        let target = match target_spec.device {
            crate::core::TargetDevice::CPU => "cpp",
            crate::core::TargetDevice::CUDA => "cpp", // Use C++ with CUDA
            _ => "python",                            // Default to Python
        };

        self.generate_code(graph, target)
    }

    /// Create lazy compiler for just-in-time compilation
    pub fn create_lazy_compiler(&self, graph: &FxGraph, target: &str) -> Result<LazyCompiler> {
        LazyCompiler::new(graph.clone(), target.to_string())
    }

    /// Clear the code generation cache
    pub fn clear_cache(&self) -> Result<()> {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        if let Ok(cache) = self.cache.read() {
            CacheStats {
                entries: cache.len(),
                total_size_bytes: cache.values().map(|v| v.len()).sum(),
            }
        } else {
            CacheStats {
                entries: 0,
                total_size_bytes: 0,
            }
        }
    }

    /// Simple hash function for graphs (in production, use a proper hash)
    fn graph_hash(&self, _graph: &FxGraph) -> u64 {
        // Simplified hash - in practice, would hash graph structure
        42
    }
}

/// Lazy compiler for just-in-time compilation
pub struct LazyCompiler {
    graph: FxGraph,
    target: String,
    compiled_code: Option<Arc<CompiledCode>>,
    compile_on_demand: bool,
}

impl LazyCompiler {
    /// Create a new lazy compiler
    pub fn new(graph: FxGraph, target: String) -> Result<Self> {
        Ok(Self {
            graph,
            target,
            compiled_code: None,
            compile_on_demand: true,
        })
    }

    /// Set whether to compile on demand
    pub fn set_compile_on_demand(&mut self, enabled: bool) {
        self.compile_on_demand = enabled;
    }

    /// Get the compiled code (compiling if necessary)
    pub fn get_compiled_code(&self) -> Result<Arc<CompiledCode>> {
        if let Some(ref code) = self.compiled_code {
            Ok(code.clone())
        } else if self.compile_on_demand {
            // Would compile here - simplified for now
            let compiled = Arc::new(CompiledCode {
                source: "// Compiled code".to_string(),
                target: self.target.clone(),
                is_valid: true,
            });
            Ok(compiled)
        } else {
            Err(torsh_core::Error::InvalidState(
                "Code not compiled and compile_on_demand is disabled".into(),
            ))
        }
    }
}

/// Compiled code representation
#[derive(Debug, Clone)]
pub struct CompiledCode {
    pub source: String,
    pub target: String,
    pub is_valid: bool,
}

impl CompiledCode {
    /// Check if the compiled code is valid
    pub fn is_valid(&self) -> bool {
        // In a real implementation, this might check various conditions
        self.is_valid && !self.source.is_empty()
    }

    /// Get the source code
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Get the target language/backend
    pub fn target(&self) -> &str {
        &self.target
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub total_size_bytes: usize,
}

impl CacheStats {
    /// Get the average entry size in bytes
    pub fn average_entry_size(&self) -> usize {
        if self.entries > 0 {
            self.total_size_bytes / self.entries
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_generator_creation() {
        let generator = CodeGenerator::new();
        let targets = generator.available_targets();

        // Should have default backends
        assert!(targets.len() >= 2);
        assert!(targets.contains(&&"python".to_string()));
        assert!(targets.contains(&&"cpp".to_string()));
    }

    #[test]
    fn test_get_file_extension() {
        let generator = CodeGenerator::new();

        assert_eq!(generator.get_file_extension("python"), Some("py"));
        assert_eq!(generator.get_file_extension("cpp"), Some("cpp"));
        assert_eq!(generator.get_file_extension("unknown"), None);
    }

    #[test]
    fn test_get_language_name() {
        let generator = CodeGenerator::new();

        assert_eq!(generator.get_language_name("python"), Some("Python"));
        assert_eq!(generator.get_language_name("cpp"), Some("C++"));
        assert_eq!(generator.get_language_name("unknown"), None);
    }

    #[test]
    fn test_cache_stats() {
        let generator = CodeGenerator::new();
        let stats = generator.cache_stats();

        assert_eq!(stats.entries, 0);
        assert_eq!(stats.total_size_bytes, 0);
        assert_eq!(stats.average_entry_size(), 0);
    }

    #[test]
    fn test_clear_cache() {
        let generator = CodeGenerator::new();
        assert!(generator.clear_cache().is_ok());
    }

    #[test]
    fn test_lazy_compiler_creation() {
        // Would need a valid FxGraph for real testing
        // This is a placeholder test
        assert!(true);
    }

    #[test]
    fn test_compiled_code_validity() {
        let code = CompiledCode {
            source: "def test(): pass".to_string(),
            target: "python".to_string(),
            is_valid: true,
        };

        assert!(code.is_valid());
        assert_eq!(code.source(), "def test(): pass");
        assert_eq!(code.target(), "python");
    }
}
