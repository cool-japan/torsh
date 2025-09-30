//! Modern WebGPU shader management for ToRSh
//!
//! This implementation uses wgpu 26.0.1 patterns and focuses exclusively on WGSL
//! for maximum compatibility and performance.

pub mod kernels;

#[cfg(feature = "webgpu")]
use md5;
#[cfg(feature = "webgpu")]
use wgpu;
use crate::webgpu::{WebGpuDevice, WebGpuError, WebGpuResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Modern shader source types (WGSL focused)
#[derive(Debug, Clone)]
pub enum ShaderSource {
    /// WGSL source code (preferred and only fully supported)
    Wgsl(String),
    /// GLSL source code (requires translation to WGSL)
    Glsl(String),
}

/// Compiled shader module with modern wgpu patterns
#[derive(Debug)]
pub struct ShaderModule {
    pub module: wgpu::ShaderModule,
    pub source: ShaderSource,
    pub entry_points: Vec<String>,
    pub size_bytes: usize,
    pub compilation_info: Option<wgpu::CompilationInfo>,
}

impl ShaderModule {
    /// Create a new shader module using modern wgpu 26.0.1 patterns
    pub fn new(
        device: &WebGpuDevice,
        source: ShaderSource,
        label: Option<&str>,
    ) -> WebGpuResult<Self> {
        let (wgsl_source, size_bytes) = match &source {
            ShaderSource::Wgsl(code) => (code.clone(), code.len()),
            ShaderSource::Glsl(glsl_code) => {
                // In a full implementation, you would use naga or similar to translate
                // For now, we'll return an error with guidance
                return Err(WebGpuError::UnsupportedFeature(
                    "GLSL shaders require translation to WGSL. Please convert your GLSL to WGSL or use a translation tool like naga.".to_string(),
                ));
            }
        };

        let module = device.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        // For simplicity, assume 'main' entry point
        let entry_points = vec!["main".to_string()];

        Ok(Self {
            module,
            source,
            entry_points,
            size_bytes,
            compilation_info: None, // Could be populated with module.get_compilation_info() if needed
        })
    }

    /// Get the underlying wgpu shader module
    pub fn wgpu_module(&self) -> &wgpu::ShaderModule {
        &self.module
    }

    /// Get shader source
    pub fn source(&self) -> &ShaderSource {
        &self.source
    }

    /// Get entry points
    pub fn entry_points(&self) -> &[String] {
        &self.entry_points
    }

    /// Get estimated size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }
}

/// Shader cache for compiled modules
#[derive(Debug)]
pub struct ShaderCache {
    cache: RwLock<HashMap<String, Arc<ShaderModule>>>,
}

impl ShaderCache {
    /// Create a new shader cache
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get or compile a shader
    pub fn get_or_compile(
        &self,
        device: &WebGpuDevice,
        key: String,
        source: ShaderSource,
        label: Option<&str>,
    ) -> WebGpuResult<Arc<ShaderModule>> {
        // Try to get from cache first
        {
            let cache = self.cache.read();
            if let Some(module) = cache.get(&key) {
                return Ok(Arc::clone(module));
            }
        }

        // Compile new shader
        let module = ShaderModule::new(device, source, label)?;
        let module_arc = Arc::new(module);

        // Store in cache
        {
            let mut cache = self.cache.write();
            cache.insert(key, Arc::clone(&module_arc));
        }

        Ok(module_arc)
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.write().clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> (usize, usize) {
        let cache = self.cache.read();
        let count = cache.len();
        let total_bytes = cache.values().map(|module| module.size_bytes()).sum();
        (count, total_bytes)
    }
}

/// Shader compiler for different source types
#[derive(Debug)]
pub struct ShaderCompiler {
    cache: Arc<ShaderCache>,
}

impl ShaderCompiler {
    /// Create a new shader compiler
    pub fn new() -> Self {
        Self {
            cache: Arc::new(ShaderCache::new()),
        }
    }

    /// Compile WGSL source
    pub fn compile_wgsl(
        &self,
        device: &WebGpuDevice,
        source: &str,
        label: Option<&str>,
    ) -> WebGpuResult<Arc<ShaderModule>> {
        let key = format!("wgsl_{:x}", md5::compute(source));
        let source = ShaderSource::Wgsl(source.to_string());
        self.cache.get_or_compile(device, key, source, label)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        self.cache.stats()
    }

    /// Clear compiled shaders
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

/// Helper functions for creating common bind group layouts
pub mod layout_helpers {
    use super::*;

    /// Create layout for binary operations (two input buffers, one output)
    pub fn create_binary_op_layout(device: &WebGpuDevice) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Binary Operation Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create layout for unary operations (one input buffer, one output)
    pub fn create_unary_op_layout(device: &WebGpuDevice) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Unary Operation Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
}
