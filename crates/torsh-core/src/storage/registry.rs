//! Registry for backend allocators and storage systems
//!
//! This module provides a centralized registry for managing different allocator
//! implementations and their capabilities across the storage system.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Registry for backend allocators
///
/// The AllocatorRegistry provides centralized management of allocator implementations,
/// allowing runtime discovery and selection of appropriate allocators for different
/// backends and devices.
///
/// Note: This is a simplified version to avoid trait object compatibility issues.
/// In a full implementation, this would use type erasure and dynamic dispatch.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::storage::AllocatorRegistry;
///
/// let mut registry = AllocatorRegistry::new();
/// registry.register("cpu".to_string());
/// registry.register("cuda".to_string());
///
/// assert!(registry.is_registered("cpu"));
/// assert_eq!(registry.list().len(), 2);
/// ```
#[derive(Debug)]
pub struct AllocatorRegistry {
    /// Set of registered allocator names
    allocator_names: HashSet<String>,
    /// Metadata about each allocator
    allocator_metadata: HashMap<String, AllocatorMetadata>,
    /// Default allocator for each backend type
    default_allocators: HashMap<String, String>,
}

impl Default for AllocatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AllocatorRegistry {
    /// Create a new allocator registry
    pub fn new() -> Self {
        Self {
            allocator_names: HashSet::new(),
            allocator_metadata: HashMap::new(),
            default_allocators: HashMap::new(),
        }
    }

    /// Register an allocator for a backend
    ///
    /// # Arguments
    /// * `name` - Unique name for the allocator
    pub fn register(&mut self, name: String) {
        self.allocator_names.insert(name.clone());
        self.allocator_metadata.insert(
            name.clone(),
            AllocatorMetadata {
                name: name.clone(),
                backend_type: "unknown".to_string(),
                supports_async: false,
                supports_numa: false,
                supports_cross_device: false,
                memory_alignment: 1,
                max_allocation_size: None,
                description: format!("Allocator: {name}"),
            },
        );
    }

    /// Register an allocator with metadata
    ///
    /// # Arguments
    /// * `name` - Unique name for the allocator
    /// * `metadata` - Metadata describing the allocator capabilities
    pub fn register_with_metadata(&mut self, name: String, metadata: AllocatorMetadata) {
        self.allocator_names.insert(name.clone());
        self.allocator_metadata.insert(name, metadata);
    }

    /// Check if an allocator is registered
    ///
    /// # Arguments
    /// * `name` - Name of the allocator to check
    ///
    /// # Returns
    /// True if the allocator is registered
    pub fn is_registered(&self, name: &str) -> bool {
        self.allocator_names.contains(name)
    }

    /// List all registered allocators
    ///
    /// # Returns
    /// Vector of allocator names
    pub fn list(&self) -> Vec<&String> {
        self.allocator_names.iter().collect()
    }

    /// Get metadata for an allocator
    ///
    /// # Arguments
    /// * `name` - Name of the allocator
    ///
    /// # Returns
    /// Metadata if the allocator exists
    pub fn get_metadata(&self, name: &str) -> Option<&AllocatorMetadata> {
        self.allocator_metadata.get(name)
    }

    /// Unregister an allocator
    ///
    /// # Arguments
    /// * `name` - Name of the allocator to remove
    ///
    /// # Returns
    /// True if the allocator was found and removed
    pub fn unregister(&mut self, name: &str) -> bool {
        let removed = self.allocator_names.remove(name);
        if removed {
            self.allocator_metadata.remove(name);
            // Remove as default allocator if it was set
            self.default_allocators.retain(|_, v| v != name);
        }
        removed
    }

    /// Set the default allocator for a backend type
    ///
    /// # Arguments
    /// * `backend_type` - Type of backend (e.g., "cpu", "cuda", "metal")
    /// * `allocator_name` - Name of the allocator to use as default
    ///
    /// # Returns
    /// True if the allocator exists and was set as default
    pub fn set_default(&mut self, backend_type: String, allocator_name: String) -> bool {
        if self.is_registered(&allocator_name) {
            self.default_allocators.insert(backend_type, allocator_name);
            true
        } else {
            false
        }
    }

    /// Get the default allocator for a backend type
    ///
    /// # Arguments
    /// * `backend_type` - Type of backend
    ///
    /// # Returns
    /// Name of the default allocator if set
    pub fn get_default(&self, backend_type: &str) -> Option<&String> {
        self.default_allocators.get(backend_type)
    }

    /// Find allocators by capability
    ///
    /// # Arguments
    /// * `capability` - Capability to search for
    ///
    /// # Returns
    /// Vector of allocator names that support the capability
    pub fn find_by_capability(&self, capability: AllocatorCapability) -> Vec<&String> {
        self.allocator_metadata
            .iter()
            .filter(|(_, metadata)| metadata.supports_capability(capability))
            .map(|(name, _)| name)
            .collect()
    }

    /// Find allocators by backend type
    ///
    /// # Arguments
    /// * `backend_type` - Backend type to search for
    ///
    /// # Returns
    /// Vector of allocator names for the backend type
    pub fn find_by_backend(&self, backend_type: &str) -> Vec<&String> {
        self.allocator_metadata
            .iter()
            .filter(|(_, metadata)| metadata.backend_type == backend_type)
            .map(|(name, _)| name)
            .collect()
    }

    /// Get registry statistics
    pub fn statistics(&self) -> RegistryStatistics {
        let backend_counts =
            self.allocator_metadata
                .values()
                .fold(HashMap::new(), |mut acc, metadata| {
                    *acc.entry(metadata.backend_type.clone()).or_insert(0) += 1;
                    acc
                });

        let capability_counts =
            self.allocator_metadata
                .values()
                .fold(HashMap::new(), |mut acc, metadata| {
                    for capability in AllocatorCapability::all() {
                        if metadata.supports_capability(capability) {
                            *acc.entry(capability).or_insert(0) += 1;
                        }
                    }
                    acc
                });

        RegistryStatistics {
            total_allocators: self.allocator_names.len(),
            backend_counts,
            capability_counts,
            default_allocators: self.default_allocators.len(),
        }
    }

    /// Clear all registered allocators
    pub fn clear(&mut self) {
        self.allocator_names.clear();
        self.allocator_metadata.clear();
        self.default_allocators.clear();
    }

    /// Get all allocators sorted by priority
    ///
    /// # Arguments
    /// * `backend_type` - Optional backend type filter
    ///
    /// # Returns
    /// Vector of allocator names sorted by priority (default first)
    pub fn get_prioritized(&self, backend_type: Option<&str>) -> Vec<&String> {
        let mut allocators: Vec<&String> = if let Some(backend) = backend_type {
            self.find_by_backend(backend)
        } else {
            self.list()
        };

        // Sort with default allocator first
        allocators.sort_by(|a, b| {
            let a_is_default = backend_type.and_then(|bt| self.get_default(bt)) == Some(*a);
            let b_is_default = backend_type.and_then(|bt| self.get_default(bt)) == Some(*b);

            match (a_is_default, b_is_default) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.cmp(b),
            }
        });

        allocators
    }
}

/// Metadata describing an allocator's capabilities
#[derive(Debug, Clone)]
pub struct AllocatorMetadata {
    /// Name of the allocator
    pub name: String,
    /// Backend type (e.g., "cpu", "cuda", "metal")
    pub backend_type: String,
    /// Whether the allocator supports async operations
    pub supports_async: bool,
    /// Whether the allocator supports NUMA awareness
    pub supports_numa: bool,
    /// Whether the allocator supports cross-device operations
    pub supports_cross_device: bool,
    /// Required memory alignment in bytes
    pub memory_alignment: usize,
    /// Maximum single allocation size (None for unlimited)
    pub max_allocation_size: Option<usize>,
    /// Human-readable description
    pub description: String,
}

impl AllocatorMetadata {
    /// Create new allocator metadata
    pub fn new(name: String, backend_type: String) -> Self {
        Self {
            name,
            backend_type,
            supports_async: false,
            supports_numa: false,
            supports_cross_device: false,
            memory_alignment: 1,
            max_allocation_size: None,
            description: String::new(),
        }
    }

    /// Set async support
    pub fn with_async(mut self, supports: bool) -> Self {
        self.supports_async = supports;
        self
    }

    /// Set NUMA support
    pub fn with_numa(mut self, supports: bool) -> Self {
        self.supports_numa = supports;
        self
    }

    /// Set cross-device support
    pub fn with_cross_device(mut self, supports: bool) -> Self {
        self.supports_cross_device = supports;
        self
    }

    /// Set memory alignment requirement
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.memory_alignment = alignment;
        self
    }

    /// Set maximum allocation size
    pub fn with_max_allocation(mut self, max_size: usize) -> Self {
        self.max_allocation_size = Some(max_size);
        self
    }

    /// Set description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Check if the allocator supports a specific capability
    pub fn supports_capability(&self, capability: AllocatorCapability) -> bool {
        match capability {
            AllocatorCapability::Async => self.supports_async,
            AllocatorCapability::Numa => self.supports_numa,
            AllocatorCapability::CrossDevice => self.supports_cross_device,
            AllocatorCapability::HighAlignment => self.memory_alignment >= 64,
            AllocatorCapability::LargeAllocations => {
                self.max_allocation_size
                    .is_none_or(|max| max >= 1024 * 1024 * 1024)
                // 1GB threshold
            }
        }
    }

    /// Check compatibility with requirements
    pub fn is_compatible_with(&self, requirements: &AllocatorRequirements) -> bool {
        if let Some(required_backend) = &requirements.backend_type {
            if self.backend_type != *required_backend {
                return false;
            }
        }

        if requirements.requires_async && !self.supports_async {
            return false;
        }

        if requirements.requires_numa && !self.supports_numa {
            return false;
        }

        if requirements.requires_cross_device && !self.supports_cross_device {
            return false;
        }

        if self.memory_alignment < requirements.min_alignment {
            return false;
        }

        if let (Some(max_alloc), Some(required_max)) =
            (self.max_allocation_size, requirements.min_max_allocation)
        {
            if max_alloc < required_max {
                return false;
            }
        }

        true
    }
}

/// Capabilities that an allocator can support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocatorCapability {
    /// Supports asynchronous operations
    Async,
    /// Supports NUMA-aware allocation
    Numa,
    /// Supports cross-device memory operations
    CrossDevice,
    /// Supports high memory alignment (64+ bytes)
    HighAlignment,
    /// Supports large allocations (1GB+)
    LargeAllocations,
}

impl AllocatorCapability {
    /// Get all possible capabilities
    pub fn all() -> Vec<Self> {
        vec![
            Self::Async,
            Self::Numa,
            Self::CrossDevice,
            Self::HighAlignment,
            Self::LargeAllocations,
        ]
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Async => "Async",
            Self::Numa => "NUMA",
            Self::CrossDevice => "Cross-Device",
            Self::HighAlignment => "High Alignment",
            Self::LargeAllocations => "Large Allocations",
        }
    }
}

/// Requirements for allocator selection
#[derive(Debug, Clone, Default)]
pub struct AllocatorRequirements {
    /// Required backend type
    pub backend_type: Option<String>,
    /// Requires async support
    pub requires_async: bool,
    /// Requires NUMA support
    pub requires_numa: bool,
    /// Requires cross-device support
    pub requires_cross_device: bool,
    /// Minimum memory alignment
    pub min_alignment: usize,
    /// Minimum maximum allocation size
    pub min_max_allocation: Option<usize>,
}

impl AllocatorRequirements {
    /// Create new requirements
    pub fn new() -> Self {
        Self::default()
    }

    /// Set backend type requirement
    pub fn with_backend(mut self, backend_type: String) -> Self {
        self.backend_type = Some(backend_type);
        self
    }

    /// Require async support
    pub fn with_async(mut self) -> Self {
        self.requires_async = true;
        self
    }

    /// Require NUMA support
    pub fn with_numa(mut self) -> Self {
        self.requires_numa = true;
        self
    }

    /// Require cross-device support
    pub fn with_cross_device(mut self) -> Self {
        self.requires_cross_device = true;
        self
    }

    /// Set minimum alignment requirement
    pub fn with_min_alignment(mut self, alignment: usize) -> Self {
        self.min_alignment = alignment;
        self
    }

    /// Set minimum maximum allocation size requirement
    pub fn with_min_max_allocation(mut self, size: usize) -> Self {
        self.min_max_allocation = Some(size);
        self
    }
}

/// Statistics about the allocator registry
#[derive(Debug, Clone)]
pub struct RegistryStatistics {
    /// Total number of registered allocators
    pub total_allocators: usize,
    /// Count of allocators by backend type
    pub backend_counts: HashMap<String, usize>,
    /// Count of allocators by capability
    pub capability_counts: HashMap<AllocatorCapability, usize>,
    /// Number of backend types with default allocators
    pub default_allocators: usize,
}

impl std::fmt::Display for RegistryStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Registry(total={}, backends={}, defaults={})",
            self.total_allocators,
            self.backend_counts.len(),
            self.default_allocators
        )
    }
}

/// Global allocator registry instance
static GLOBAL_REGISTRY: RwLock<Option<Arc<RwLock<AllocatorRegistry>>>> = RwLock::new(None);

/// Get the global allocator registry
pub fn global_registry() -> Arc<RwLock<AllocatorRegistry>> {
    let mut global = GLOBAL_REGISTRY.write().unwrap();
    if global.is_none() {
        *global = Some(Arc::new(RwLock::new(AllocatorRegistry::new())));
    }
    global.as_ref().unwrap().clone()
}

/// Initialize the global registry with default allocators
pub fn initialize_global_registry() {
    let registry = global_registry();
    let mut registry = registry.write().unwrap();

    // Register basic allocators
    registry.register_with_metadata(
        "cpu_std".to_string(),
        AllocatorMetadata::new("cpu_std".to_string(), "cpu".to_string())
            .with_description("Standard CPU allocator using system malloc".to_string())
            .with_alignment(16),
    );

    registry.register_with_metadata(
        "cpu_numa".to_string(),
        AllocatorMetadata::new("cpu_numa".to_string(), "cpu".to_string())
            .with_numa(true)
            .with_description("NUMA-aware CPU allocator".to_string())
            .with_alignment(64),
    );

    // Set defaults
    registry.set_default("cpu".to_string(), "cpu_std".to_string());
}

/// Utility functions for registry operations
pub mod utils {
    use super::*;

    /// Find the best allocator for given requirements
    pub fn find_best_allocator(
        registry: &AllocatorRegistry,
        requirements: &AllocatorRequirements,
    ) -> Option<String> {
        // Get compatible allocators
        let compatible: Vec<_> = registry
            .allocator_metadata
            .iter()
            .filter(|(_, metadata)| metadata.is_compatible_with(requirements))
            .collect();

        if compatible.is_empty() {
            return None;
        }

        // Score allocators based on capabilities
        let mut scored: Vec<_> = compatible
            .into_iter()
            .map(|(name, metadata)| {
                let mut score = 0;

                // Prefer allocators with more capabilities
                for capability in AllocatorCapability::all() {
                    if metadata.supports_capability(capability) {
                        score += 1;
                    }
                }

                // Prefer default allocators
                if let Some(backend) = &requirements.backend_type {
                    if registry.get_default(backend) == Some(name) {
                        score += 10;
                    }
                }

                // Prefer exact backend match
                if let Some(required_backend) = &requirements.backend_type {
                    if metadata.backend_type == *required_backend {
                        score += 5;
                    }
                }

                (name.clone(), score)
            })
            .collect();

        // Sort by score (highest first)
        scored.sort_by(|a, b| b.1.cmp(&a.1));

        scored.first().map(|(name, _)| name.clone())
    }

    /// Validate allocator metadata
    pub fn validate_metadata(metadata: &AllocatorMetadata) -> Result<(), String> {
        if metadata.name.is_empty() {
            return Err("Allocator name cannot be empty".to_string());
        }

        if metadata.backend_type.is_empty() {
            return Err("Backend type cannot be empty".to_string());
        }

        if metadata.memory_alignment == 0 || !metadata.memory_alignment.is_power_of_two() {
            return Err("Memory alignment must be a power of two".to_string());
        }

        Ok(())
    }

    /// Get allocator compatibility score
    pub fn compatibility_score(
        metadata: &AllocatorMetadata,
        requirements: &AllocatorRequirements,
    ) -> Option<u32> {
        if !metadata.is_compatible_with(requirements) {
            return None;
        }

        let mut score = 100; // Base compatibility score

        // Bonus for exact backend match
        if let Some(required_backend) = &requirements.backend_type {
            if metadata.backend_type == *required_backend {
                score += 50;
            }
        }

        // Bonus for capabilities
        if requirements.requires_async && metadata.supports_async {
            score += 10;
        }
        if requirements.requires_numa && metadata.supports_numa {
            score += 10;
        }
        if requirements.requires_cross_device && metadata.supports_cross_device {
            score += 10;
        }

        // Bonus for high alignment
        if metadata.memory_alignment >= 64 {
            score += 5;
        }

        Some(score)
    }

    /// Create a summary of registry contents
    pub fn registry_summary(registry: &AllocatorRegistry) -> String {
        let stats = registry.statistics();
        let mut summary = "Allocator Registry Summary:\n".to_string();
        summary.push_str(&format!("  Total allocators: {}\n", stats.total_allocators));

        summary.push_str("  By backend:\n");
        for (backend, count) in &stats.backend_counts {
            summary.push_str(&format!("    {}: {}\n", backend, count));
        }

        summary.push_str("  By capability:\n");
        for (capability, count) in &stats.capability_counts {
            summary.push_str(&format!("    {}: {}\n", capability.name(), count));
        }

        summary.push_str(&format!(
            "  Default allocators: {}\n",
            stats.default_allocators
        ));

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_registry_basic() {
        let mut registry = AllocatorRegistry::new();

        // Initially empty
        assert_eq!(registry.list().len(), 0);
        assert!(!registry.is_registered("test"));

        // Register an allocator
        registry.register("test_allocator".to_string());
        assert!(registry.is_registered("test_allocator"));
        assert_eq!(registry.list().len(), 1);

        // Unregister
        assert!(registry.unregister("test_allocator"));
        assert!(!registry.is_registered("test_allocator"));
        assert_eq!(registry.list().len(), 0);
    }

    #[test]
    fn test_allocator_metadata() {
        let metadata = AllocatorMetadata::new("test".to_string(), "cpu".to_string())
            .with_async(true)
            .with_numa(true)
            .with_alignment(64)
            .with_description("Test allocator".to_string());

        assert!(metadata.supports_capability(AllocatorCapability::Async));
        assert!(metadata.supports_capability(AllocatorCapability::Numa));
        assert!(metadata.supports_capability(AllocatorCapability::HighAlignment));
        assert_eq!(metadata.memory_alignment, 64);
    }

    #[test]
    fn test_allocator_requirements() {
        let requirements = AllocatorRequirements::new()
            .with_backend("cpu".to_string())
            .with_async()
            .with_min_alignment(32);

        let compatible_metadata = AllocatorMetadata::new("test".to_string(), "cpu".to_string())
            .with_async(true)
            .with_alignment(32);

        let incompatible_metadata =
            AllocatorMetadata::new("test".to_string(), "gpu".to_string()).with_alignment(16);

        assert!(compatible_metadata.is_compatible_with(&requirements));
        assert!(!incompatible_metadata.is_compatible_with(&requirements));
    }

    #[test]
    fn test_default_allocators() {
        let mut registry = AllocatorRegistry::new();

        registry.register("cpu_allocator".to_string());
        registry.register("gpu_allocator".to_string());

        // Set defaults
        assert!(registry.set_default("cpu".to_string(), "cpu_allocator".to_string()));
        assert!(registry.set_default("gpu".to_string(), "gpu_allocator".to_string()));

        // Cannot set non-existent allocator as default
        assert!(!registry.set_default("cpu".to_string(), "non_existent".to_string()));

        // Check defaults
        assert_eq!(
            registry.get_default("cpu"),
            Some(&"cpu_allocator".to_string())
        );
        assert_eq!(
            registry.get_default("gpu"),
            Some(&"gpu_allocator".to_string())
        );
        assert_eq!(registry.get_default("metal"), None);
    }

    #[test]
    fn test_find_by_capability() {
        let mut registry = AllocatorRegistry::new();

        let async_metadata =
            AllocatorMetadata::new("async".to_string(), "cpu".to_string()).with_async(true);
        let numa_metadata =
            AllocatorMetadata::new("numa".to_string(), "cpu".to_string()).with_numa(true);

        registry.register_with_metadata("async".to_string(), async_metadata);
        registry.register_with_metadata("numa".to_string(), numa_metadata);

        let async_allocators = registry.find_by_capability(AllocatorCapability::Async);
        assert_eq!(async_allocators.len(), 1);
        assert!(async_allocators.contains(&&"async".to_string()));

        let numa_allocators = registry.find_by_capability(AllocatorCapability::Numa);
        assert_eq!(numa_allocators.len(), 1);
        assert!(numa_allocators.contains(&&"numa".to_string()));
    }

    #[test]
    fn test_find_by_backend() {
        let mut registry = AllocatorRegistry::new();

        let cpu_metadata = AllocatorMetadata::new("cpu1".to_string(), "cpu".to_string());
        let gpu_metadata = AllocatorMetadata::new("gpu1".to_string(), "gpu".to_string());

        registry.register_with_metadata("cpu1".to_string(), cpu_metadata);
        registry.register_with_metadata("gpu1".to_string(), gpu_metadata);

        let cpu_allocators = registry.find_by_backend("cpu");
        assert_eq!(cpu_allocators.len(), 1);
        assert!(cpu_allocators.contains(&&"cpu1".to_string()));

        let gpu_allocators = registry.find_by_backend("gpu");
        assert_eq!(gpu_allocators.len(), 1);
        assert!(gpu_allocators.contains(&&"gpu1".to_string()));
    }

    #[test]
    fn test_registry_statistics() {
        let mut registry = AllocatorRegistry::new();

        let cpu_metadata =
            AllocatorMetadata::new("cpu1".to_string(), "cpu".to_string()).with_async(true);
        let gpu_metadata =
            AllocatorMetadata::new("gpu1".to_string(), "gpu".to_string()).with_numa(true);

        registry.register_with_metadata("cpu1".to_string(), cpu_metadata);
        registry.register_with_metadata("gpu1".to_string(), gpu_metadata);
        registry.set_default("cpu".to_string(), "cpu1".to_string());

        let stats = registry.statistics();
        assert_eq!(stats.total_allocators, 2);
        assert_eq!(stats.backend_counts.get("cpu"), Some(&1));
        assert_eq!(stats.backend_counts.get("gpu"), Some(&1));
        assert_eq!(stats.default_allocators, 1);
    }

    #[test]
    fn test_utils_find_best_allocator() {
        use utils::*;

        let mut registry = AllocatorRegistry::new();

        let good_metadata = AllocatorMetadata::new("good".to_string(), "cpu".to_string())
            .with_async(true)
            .with_numa(true)
            .with_alignment(64);

        let basic_metadata =
            AllocatorMetadata::new("basic".to_string(), "cpu".to_string()).with_alignment(16);

        registry.register_with_metadata("good".to_string(), good_metadata);
        registry.register_with_metadata("basic".to_string(), basic_metadata);
        registry.set_default("cpu".to_string(), "basic".to_string());

        let requirements = AllocatorRequirements::new()
            .with_backend("cpu".to_string())
            .with_async();

        let best = find_best_allocator(&registry, &requirements);
        assert_eq!(best, Some("good".to_string())); // Should pick the one with async support
    }

    #[test]
    fn test_utils_validate_metadata() {
        use utils::*;

        let valid_metadata =
            AllocatorMetadata::new("valid".to_string(), "cpu".to_string()).with_alignment(64);

        let invalid_name = AllocatorMetadata::new("".to_string(), "cpu".to_string());
        let invalid_backend = AllocatorMetadata::new("test".to_string(), "".to_string());
        let invalid_alignment =
            AllocatorMetadata::new("test".to_string(), "cpu".to_string()).with_alignment(0);

        assert!(validate_metadata(&valid_metadata).is_ok());
        assert!(validate_metadata(&invalid_name).is_err());
        assert!(validate_metadata(&invalid_backend).is_err());
        assert!(validate_metadata(&invalid_alignment).is_err());
    }

    #[test]
    fn test_global_registry() {
        initialize_global_registry();

        let registry = global_registry();
        let registry = registry.read().unwrap();

        assert!(registry.is_registered("cpu_std"));
        assert!(registry.is_registered("cpu_numa"));
        assert_eq!(registry.get_default("cpu"), Some(&"cpu_std".to_string()));
    }
}
