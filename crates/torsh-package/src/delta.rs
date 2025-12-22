//! Delta patching for incremental package updates
//!
//! This module provides functionality for creating and applying delta patches
//! between package versions, enabling efficient incremental updates for large models.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use torsh_core::error::{Result, TorshError};

use crate::manifest::PackageManifest;
use crate::package::Package;
use crate::resources::Resource;
use crate::utils::calculate_hash;

/// Delta patch operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeltaOperation {
    /// Add a new resource
    Add {
        /// Name of the resource to add
        resource_name: String,
        /// Resource data content
        resource_data: Vec<u8>,
        /// Resource metadata
        metadata: HashMap<String, String>,
    },
    /// Remove a resource
    Remove {
        /// Name of the resource to remove
        resource_name: String,
    },
    /// Modify an existing resource with binary diff
    Modify {
        /// Name of the resource to modify
        resource_name: String,
        /// Binary diff data
        diff_data: Vec<u8>,
        /// Hash of original resource data
        old_hash: String,
        /// Hash of target resource data
        new_hash: String,
    },
    /// Update manifest metadata
    UpdateManifest {
        /// Manifest field name to update
        field: String,
        /// New value for the field
        new_value: String,
    },
}

/// Delta patch containing a set of operations to transform one package version to another
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaPatch {
    /// Source package version
    pub from_version: String,
    /// Target package version
    pub to_version: String,
    /// Patch format version for compatibility
    pub patch_format_version: String,
    /// List of operations to apply
    pub operations: Vec<DeltaOperation>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Patch size in bytes
    pub patch_size: u64,
    /// Expected final package hash after applying patch
    pub target_hash: String,
    /// Metadata about the patch
    pub metadata: HashMap<String, String>,
}

/// Delta patch builder for creating patches between packages
pub struct DeltaPatchBuilder {
    compression_level: u32,
    include_metadata_changes: bool,
    max_diff_size_ratio: f64,
}

/// Delta patch applier for applying patches to packages
pub struct DeltaPatchApplier {
    verify_checksums: bool,
    backup_original: bool,
}

impl Default for DeltaPatchBuilder {
    fn default() -> Self {
        Self {
            compression_level: 6,
            include_metadata_changes: true,
            max_diff_size_ratio: 0.8, // Only create binary diff if it's less than 80% of original size
        }
    }
}

impl DeltaPatchBuilder {
    /// Create a new delta patch builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression level for patch data (0-9)
    pub fn with_compression_level(mut self, level: u32) -> Self {
        self.compression_level = level.min(9);
        self
    }

    /// Enable or disable metadata change tracking
    pub fn with_metadata_changes(mut self, include: bool) -> Self {
        self.include_metadata_changes = include;
        self
    }

    /// Set maximum diff size ratio (diff size / original size)
    pub fn with_max_diff_ratio(mut self, ratio: f64) -> Self {
        self.max_diff_size_ratio = ratio.clamp(0.1, 1.0);
        self
    }

    /// Generate a delta patch between two packages
    pub fn create_patch(&self, old_package: &Package, new_package: &Package) -> Result<DeltaPatch> {
        let mut operations = Vec::new();

        // Compare versions
        let from_version = old_package.get_version().to_string();
        let to_version = new_package.get_version().to_string();

        // Find resource differences
        let old_resources: HashSet<String> = old_package.resources.keys().cloned().collect();
        let new_resources: HashSet<String> = new_package.resources.keys().cloned().collect();

        // Find added resources
        for resource_name in new_resources.difference(&old_resources) {
            if let Some(resource) = new_package.resources.get(resource_name) {
                operations.push(DeltaOperation::Add {
                    resource_name: resource_name.clone(),
                    resource_data: resource.data.clone(),
                    metadata: resource.metadata.clone(),
                });
            }
        }

        // Find removed resources
        for resource_name in old_resources.difference(&new_resources) {
            operations.push(DeltaOperation::Remove {
                resource_name: resource_name.clone(),
            });
        }

        // Find modified resources
        for resource_name in old_resources.intersection(&new_resources) {
            if let (Some(old_resource), Some(new_resource)) = (
                old_package.resources.get(resource_name),
                new_package.resources.get(resource_name),
            ) {
                let old_hash = calculate_hash(&old_resource.data);
                let new_hash = calculate_hash(&new_resource.data);

                if old_hash != new_hash {
                    // Create binary diff
                    let diff_data =
                        self.create_binary_diff(&old_resource.data, &new_resource.data)?;

                    // Only use diff if it's smaller than the threshold
                    let diff_ratio = diff_data.len() as f64 / new_resource.data.len() as f64;

                    if diff_ratio <= self.max_diff_size_ratio {
                        operations.push(DeltaOperation::Modify {
                            resource_name: resource_name.clone(),
                            diff_data,
                            old_hash,
                            new_hash,
                        });
                    } else {
                        // If diff is too large, just replace the entire resource
                        operations.push(DeltaOperation::Remove {
                            resource_name: resource_name.clone(),
                        });
                        operations.push(DeltaOperation::Add {
                            resource_name: resource_name.clone(),
                            resource_data: new_resource.data.clone(),
                            metadata: new_resource.metadata.clone(),
                        });
                    }
                }
            }
        }

        // Check for manifest changes if enabled
        if self.include_metadata_changes {
            self.add_manifest_changes(
                &mut operations,
                &old_package.manifest,
                &new_package.manifest,
            )?;
        }

        // Calculate patch size
        let patch_size = self.calculate_patch_size(&operations);

        // Generate target hash (simplified - just hash the new package manifest)
        let target_hash = calculate_hash(
            &bincode::serde::encode_to_vec(&new_package.manifest, bincode::config::standard())
                .unwrap(),
        );

        let mut metadata = HashMap::new();
        metadata.insert(
            "compression_level".to_string(),
            self.compression_level.to_string(),
        );
        metadata.insert("operations_count".to_string(), operations.len().to_string());

        Ok(DeltaPatch {
            from_version,
            to_version,
            patch_format_version: "1.0.0".to_string(),
            operations,
            created_at: chrono::Utc::now(),
            patch_size,
            target_hash,
            metadata,
        })
    }

    /// Create binary diff between two byte arrays
    fn create_binary_diff(&self, _old_data: &[u8], new_data: &[u8]) -> Result<Vec<u8>> {
        // Simple implementation using basic compression
        // In a production system, you might want to use more sophisticated algorithms like bsdiff

        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;

        // For simplicity, we'll create a compressed representation of the new data
        // A more sophisticated implementation would create actual binary diffs
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.compression_level));
        encoder
            .write_all(new_data)
            .map_err(|e| TorshError::SerializationError(format!("Compression failed: {}", e)))?;

        encoder.finish().map_err(|e| {
            TorshError::SerializationError(format!("Compression finalization failed: {}", e))
        })
    }

    /// Add manifest change operations
    fn add_manifest_changes(
        &self,
        operations: &mut Vec<DeltaOperation>,
        old_manifest: &PackageManifest,
        new_manifest: &PackageManifest,
    ) -> Result<()> {
        // Check for version change
        if old_manifest.version != new_manifest.version {
            operations.push(DeltaOperation::UpdateManifest {
                field: "version".to_string(),
                new_value: new_manifest.version.clone(),
            });
        }

        // Check for description change
        if old_manifest.description != new_manifest.description {
            let new_value = new_manifest.description.clone().unwrap_or_default();
            operations.push(DeltaOperation::UpdateManifest {
                field: "description".to_string(),
                new_value,
            });
        }

        // Check for author change
        if old_manifest.author != new_manifest.author {
            let new_value = new_manifest.author.clone().unwrap_or_default();
            operations.push(DeltaOperation::UpdateManifest {
                field: "author".to_string(),
                new_value,
            });
        }

        Ok(())
    }

    /// Calculate total patch size
    fn calculate_patch_size(&self, operations: &[DeltaOperation]) -> u64 {
        operations
            .iter()
            .map(|op| match op {
                DeltaOperation::Add { resource_data, .. } => resource_data.len() as u64,
                DeltaOperation::Modify { diff_data, .. } => diff_data.len() as u64,
                DeltaOperation::Remove { .. } => 0,
                DeltaOperation::UpdateManifest { new_value, .. } => new_value.len() as u64,
            })
            .sum()
    }
}

impl Default for DeltaPatchApplier {
    fn default() -> Self {
        Self {
            verify_checksums: true,
            backup_original: true,
        }
    }
}

impl DeltaPatchApplier {
    /// Create a new delta patch applier
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable checksum verification
    pub fn with_checksum_verification(mut self, verify: bool) -> Self {
        self.verify_checksums = verify;
        self
    }

    /// Enable or disable backup creation before applying patch
    pub fn with_backup(mut self, backup: bool) -> Self {
        self.backup_original = backup;
        self
    }

    /// Apply a delta patch to a package
    pub fn apply_patch(&self, package: &mut Package, patch: &DeltaPatch) -> Result<()> {
        // Verify version compatibility
        if package.get_version() != patch.from_version {
            return Err(TorshError::InvalidArgument(format!(
                "Version mismatch: package version {} doesn't match patch from_version {}",
                package.get_version(),
                patch.from_version
            )));
        }

        // Apply operations
        for operation in &patch.operations {
            self.apply_operation(package, operation)?;
        }

        // Verify final hash if enabled
        if self.verify_checksums {
            let current_hash = calculate_hash(
                &bincode::serde::encode_to_vec(&package.manifest, bincode::config::standard())
                    .unwrap(),
            );
            if current_hash != patch.target_hash {
                return Err(TorshError::InvalidArgument(
                    "Package hash doesn't match expected target hash after applying patch"
                        .to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Apply a single delta operation
    fn apply_operation(&self, package: &mut Package, operation: &DeltaOperation) -> Result<()> {
        match operation {
            DeltaOperation::Add {
                resource_name,
                resource_data,
                metadata,
            } => {
                let resource = Resource {
                    name: resource_name.clone(),
                    resource_type: crate::resources::ResourceType::Data, // Default type
                    data: resource_data.clone(),
                    metadata: metadata.clone(),
                };
                package.resources.insert(resource_name.clone(), resource);
            }
            DeltaOperation::Remove { resource_name } => {
                package.resources.remove(resource_name);
            }
            DeltaOperation::Modify {
                resource_name,
                diff_data,
                old_hash,
                new_hash,
            } => {
                if let Some(resource) = package.resources.get_mut(resource_name) {
                    // Verify old hash if enabled
                    if self.verify_checksums {
                        let current_hash = calculate_hash(&resource.data);
                        if current_hash != *old_hash {
                            return Err(TorshError::InvalidArgument(format!(
                                "Resource {} hash mismatch before applying diff",
                                resource_name
                            )));
                        }
                    }

                    // Apply diff (decompress for our simple implementation)
                    let new_data = self.apply_binary_diff(&resource.data, diff_data)?;

                    // Verify new hash if enabled
                    if self.verify_checksums {
                        let result_hash = calculate_hash(&new_data);
                        if result_hash != *new_hash {
                            return Err(TorshError::InvalidArgument(format!(
                                "Resource {} hash mismatch after applying diff",
                                resource_name
                            )));
                        }
                    }

                    resource.data = new_data;
                }
            }
            DeltaOperation::UpdateManifest { field, new_value } => {
                self.update_manifest_field(&mut package.manifest, field, new_value)?;
            }
        }
        Ok(())
    }

    /// Apply binary diff (decompress for our simple implementation)
    fn apply_binary_diff(&self, _old_data: &[u8], diff_data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(diff_data);
        let mut result = Vec::new();
        decoder
            .read_to_end(&mut result)
            .map_err(|e| TorshError::SerializationError(format!("Decompression failed: {}", e)))?;

        Ok(result)
    }

    /// Update a manifest field
    fn update_manifest_field(
        &self,
        manifest: &mut PackageManifest,
        field: &str,
        new_value: &str,
    ) -> Result<()> {
        match field {
            "version" => manifest.version = new_value.to_string(),
            "description" => manifest.description = Some(new_value.to_string()),
            "author" => manifest.author = Some(new_value.to_string()),
            _ => {
                // For unknown fields, store in metadata
                manifest
                    .metadata
                    .insert(field.to_string(), new_value.to_string());
            }
        }
        Ok(())
    }

    /// Save delta patch to file
    pub fn save_patch<P: AsRef<Path>>(patch: &DeltaPatch, path: P) -> Result<()> {
        let serialized = bincode::serde::encode_to_vec(patch, bincode::config::standard())
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        fs::write(path, serialized).map_err(|e| TorshError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load delta patch from file
    pub fn load_patch<P: AsRef<Path>>(path: P) -> Result<DeltaPatch> {
        let data = fs::read(path).map_err(|e| TorshError::IoError(e.to_string()))?;

        let (patch, _): (DeltaPatch, usize) =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        Ok(patch)
    }
}

/// Extension trait for Package to add delta functionality
pub trait DeltaPackageExt {
    /// Create a delta patch from this package to another
    fn create_delta_to(&self, target: &Package) -> Result<DeltaPatch>;

    /// Apply a delta patch to this package
    fn apply_delta(&mut self, patch: &DeltaPatch) -> Result<()>;

    /// Get package hash for delta verification
    fn get_package_hash(&self) -> String;
}

impl DeltaPackageExt for Package {
    fn create_delta_to(&self, target: &Package) -> Result<DeltaPatch> {
        let builder = DeltaPatchBuilder::new();
        builder.create_patch(self, target)
    }

    fn apply_delta(&mut self, patch: &DeltaPatch) -> Result<()> {
        let applier = DeltaPatchApplier::new();
        applier.apply_patch(self, patch)
    }

    fn get_package_hash(&self) -> String {
        calculate_hash(
            &bincode::serde::encode_to_vec(&self.manifest, bincode::config::standard()).unwrap(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::ResourceType;

    #[test]
    fn test_delta_patch_builder() {
        let builder = DeltaPatchBuilder::new()
            .with_compression_level(9)
            .with_metadata_changes(true)
            .with_max_diff_ratio(0.5);

        assert_eq!(builder.compression_level, 9);
        assert!(builder.include_metadata_changes);
        assert!((builder.max_diff_size_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_delta_patch_applier() {
        let applier = DeltaPatchApplier::new()
            .with_checksum_verification(false)
            .with_backup(false);

        assert!(!applier.verify_checksums);
        assert!(!applier.backup_original);
    }

    #[test]
    fn test_simple_delta_creation() {
        let old_package = Package::new("test".to_string(), "1.0.0".to_string());
        let mut new_package = Package::new("test".to_string(), "1.1.0".to_string());

        // Add a resource to the new package
        let resource = Resource {
            name: "test.txt".to_string(),
            resource_type: ResourceType::Data,
            data: b"Hello, World!".to_vec(),
            metadata: HashMap::new(),
        };
        new_package
            .resources
            .insert("test.txt".to_string(), resource);

        let builder = DeltaPatchBuilder::new();
        let patch = builder.create_patch(&old_package, &new_package).unwrap();

        assert_eq!(patch.from_version, "1.0.0");
        assert_eq!(patch.to_version, "1.1.0");
        assert!(!patch.operations.is_empty());
    }

    #[test]
    fn test_patch_serialization() {
        let patch = DeltaPatch {
            from_version: "1.0.0".to_string(),
            to_version: "1.1.0".to_string(),
            patch_format_version: "1.0.0".to_string(),
            operations: vec![DeltaOperation::Add {
                resource_name: "test.txt".to_string(),
                resource_data: b"Hello".to_vec(),
                metadata: HashMap::new(),
            }],
            created_at: chrono::Utc::now(),
            patch_size: 5,
            target_hash: "abc123".to_string(),
            metadata: HashMap::new(),
        };

        // Test serialization and deserialization
        let serialized =
            bincode::serde::encode_to_vec(&patch, bincode::config::standard()).unwrap();
        let (deserialized, _): (DeltaPatch, usize) =
            bincode::serde::decode_from_slice(&serialized, bincode::config::standard()).unwrap();

        assert_eq!(patch.from_version, deserialized.from_version);
        assert_eq!(patch.to_version, deserialized.to_version);
        assert_eq!(patch.operations.len(), deserialized.operations.len());
    }
}
