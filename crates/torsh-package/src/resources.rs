//! Resource handling for packages

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Type of resource in the package
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// Model weights and state
    Model,
    /// Source code
    Source,
    /// Data files
    Data,
    /// Configuration files
    Config,
    /// Documentation
    Documentation,
    /// License files
    License,
    /// Binary assets
    Binary,
    /// Text assets
    Text,
    /// Metadata
    Metadata,
}

impl ResourceType {
    /// Get file extension for resource type
    pub fn extension(&self) -> &'static str {
        match self {
            ResourceType::Model => "model",
            ResourceType::Source => "rs",
            ResourceType::Data => "data",
            ResourceType::Config => "toml",
            ResourceType::Documentation => "md",
            ResourceType::License => "txt",
            ResourceType::Binary => "bin",
            ResourceType::Text => "txt",
            ResourceType::Metadata => "json",
        }
    }

    /// Get MIME type for resource
    pub fn mime_type(&self) -> &'static str {
        match self {
            ResourceType::Model => "application/octet-stream",
            ResourceType::Source => "text/x-rust",
            ResourceType::Data => "application/octet-stream",
            ResourceType::Config => "text/x-toml",
            ResourceType::Documentation => "text/markdown",
            ResourceType::License => "text/plain",
            ResourceType::Binary => "application/octet-stream",
            ResourceType::Text => "text/plain",
            ResourceType::Metadata => "application/json",
        }
    }

    /// Determine resource type from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "model" | "pth" | "pt" | "torsh" => ResourceType::Model,
            "rs" => ResourceType::Source,
            "json" => ResourceType::Metadata,
            "toml" | "yaml" | "yml" => ResourceType::Config,
            "md" | "rst" => ResourceType::Documentation,
            "txt" | "license" => ResourceType::License,
            "bin" | "dat" => ResourceType::Binary,
            _ => ResourceType::Data,
        }
    }
}

/// A resource in the package
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Resource {
    /// Resource name (unique identifier)
    pub name: String,

    /// Resource type
    pub resource_type: ResourceType,

    /// Resource data
    pub data: Vec<u8>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Resource {
    /// Create a new resource
    pub fn new(name: String, resource_type: ResourceType, data: Vec<u8>) -> Self {
        Self {
            name,
            resource_type,
            data,
            metadata: HashMap::new(),
        }
    }

    /// Create resource from file
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        resource_type: Option<ResourceType>,
    ) -> std::io::Result<Self> {
        let path = path.as_ref();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unnamed")
            .to_string();

        let resource_type = resource_type.unwrap_or_else(|| {
            path.extension()
                .and_then(|e| e.to_str())
                .map(ResourceType::from_extension)
                .unwrap_or(ResourceType::Data)
        });

        let data = std::fs::read(path)?;

        Ok(Self::new(name, resource_type, data))
    }

    /// Get resource size
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Calculate SHA256 hash
    pub fn sha256(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&self.data);
        hex::encode(hasher.finalize())
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Check if resource is compressed
    pub fn is_compressed(&self) -> bool {
        self.metadata.contains_key("compression")
    }

    /// Get compression method
    pub fn compression_method(&self) -> Option<&str> {
        self.get_metadata("compression")
    }
}

/// Resource collection for managing multiple resources
pub struct ResourceCollection {
    resources: HashMap<String, Resource>,
}

impl ResourceCollection {
    /// Create a new resource collection
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
        }
    }

    /// Add a resource
    pub fn add(&mut self, resource: Resource) -> Result<(), String> {
        if self.resources.contains_key(&resource.name) {
            return Err(format!("Resource '{}' already exists", resource.name));
        }

        self.resources.insert(resource.name.clone(), resource);
        Ok(())
    }

    /// Get a resource by name
    pub fn get(&self, name: &str) -> Option<&Resource> {
        self.resources.get(name)
    }

    /// Get mutable resource by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Resource> {
        self.resources.get_mut(name)
    }

    /// Remove a resource
    pub fn remove(&mut self, name: &str) -> Option<Resource> {
        self.resources.remove(name)
    }

    /// List all resource names
    pub fn list(&self) -> Vec<&str> {
        self.resources.keys().map(|s| s.as_str()).collect()
    }

    /// Get resources by type
    pub fn by_type(&self, resource_type: ResourceType) -> Vec<&Resource> {
        self.resources
            .values()
            .filter(|r| r.resource_type == resource_type)
            .collect()
    }

    /// Get total size of all resources
    pub fn total_size(&self) -> usize {
        self.resources.values().map(|r| r.size()).sum()
    }

    /// Clear all resources
    pub fn clear(&mut self) {
        self.resources.clear();
    }

    /// Get number of resources
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }
}

impl Default for ResourceCollection {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource filter for selective operations
pub struct ResourceFilter {
    types: Option<Vec<ResourceType>>,
    name_pattern: Option<String>,
    min_size: Option<usize>,
    max_size: Option<usize>,
}

impl ResourceFilter {
    /// Create a new filter
    pub fn new() -> Self {
        Self {
            types: None,
            name_pattern: None,
            min_size: None,
            max_size: None,
        }
    }

    /// Filter by resource types
    pub fn with_types(mut self, types: Vec<ResourceType>) -> Self {
        self.types = Some(types);
        self
    }

    /// Filter by name pattern
    pub fn with_name_pattern(mut self, pattern: String) -> Self {
        self.name_pattern = Some(pattern);
        self
    }

    /// Filter by size range
    pub fn with_size_range(mut self, min: Option<usize>, max: Option<usize>) -> Self {
        self.min_size = min;
        self.max_size = max;
        self
    }

    /// Check if resource matches filter
    pub fn matches(&self, resource: &Resource) -> bool {
        // Check type filter
        if let Some(types) = &self.types {
            if !types.contains(&resource.resource_type) {
                return false;
            }
        }

        // Check name pattern
        if let Some(pattern) = &self.name_pattern {
            if !resource.name.contains(pattern) {
                return false;
            }
        }

        // Check size range
        let size = resource.size();
        if let Some(min) = self.min_size {
            if size < min {
                return false;
            }
        }
        if let Some(max) = self.max_size {
            if size > max {
                return false;
            }
        }

        true
    }
}

impl Default for ResourceFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_type() {
        assert_eq!(ResourceType::from_extension("rs"), ResourceType::Source);
        assert_eq!(ResourceType::from_extension("model"), ResourceType::Model);
        assert_eq!(ResourceType::from_extension("json"), ResourceType::Metadata);
        assert_eq!(ResourceType::from_extension("unknown"), ResourceType::Data);
    }

    #[test]
    fn test_resource_creation() {
        let resource = Resource::new(
            "test.model".to_string(),
            ResourceType::Model,
            vec![1, 2, 3, 4],
        );

        assert_eq!(resource.name, "test.model");
        assert_eq!(resource.resource_type, ResourceType::Model);
        assert_eq!(resource.size(), 4);
    }

    #[test]
    fn test_resource_collection() {
        let mut collection = ResourceCollection::new();

        let resource1 = Resource::new("res1".to_string(), ResourceType::Model, vec![1, 2, 3]);
        let resource2 = Resource::new("res2".to_string(), ResourceType::Data, vec![4, 5, 6, 7]);

        collection.add(resource1).unwrap();
        collection.add(resource2).unwrap();

        assert_eq!(collection.len(), 2);
        assert_eq!(collection.total_size(), 7);

        let models = collection.by_type(ResourceType::Model);
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn test_resource_filter() {
        let resource = Resource::new("test.model".to_string(), ResourceType::Model, vec![0; 100]);

        let filter = ResourceFilter::new()
            .with_types(vec![ResourceType::Model, ResourceType::Data])
            .with_size_range(Some(50), Some(200));

        assert!(filter.matches(&resource));

        let filter2 = ResourceFilter::new()
            .with_types(vec![ResourceType::Source])
            .with_size_range(Some(50), Some(200));

        assert!(!filter2.matches(&resource));
    }
}
