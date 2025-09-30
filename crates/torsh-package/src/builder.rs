//! Package Builder
//!
//! This module provides a convenient builder pattern for creating packages
//! with fluent API and configuration options.

use std::path::Path;
use torsh_core::error::Result;

use crate::package::Package;

/// Package builder for convenient package creation
pub struct PackageBuilder {
    pub(crate) package: Package,
    pub(crate) config: BuilderConfig,
}

/// Configuration for package building
#[derive(Debug, Clone)]
pub struct BuilderConfig {
    /// Include source code in the package
    pub include_source: bool,
    /// Compress the package contents
    pub compress: bool,
    /// Sign the package with a cryptographic signature
    pub sign: bool,
    /// Include dependency information
    pub include_dependencies: bool,
}

impl Default for BuilderConfig {
    fn default() -> Self {
        Self {
            include_source: false,
            compress: true,
            sign: false,
            include_dependencies: true,
        }
    }
}

impl PackageBuilder {
    /// Create a new package builder
    pub fn new(name: String, version: String) -> Self {
        Self {
            package: Package::new(name, version),
            config: BuilderConfig::default(),
        }
    }

    /// Set builder configuration
    pub fn with_config(mut self, config: BuilderConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a module (temporarily disabled - requires torsh-nn)
    #[cfg(feature = "with-nn")]
    pub fn add_module<M: torsh_nn::Module>(mut self, name: &str, module: &M) -> Result<Self> {
        self.package
            .add_module(name, module, self.config.include_source)?;
        Ok(self)
    }

    /// Add a data file
    pub fn add_data_file<P: AsRef<Path>>(mut self, name: &str, path: P) -> Result<Self> {
        self.package.add_data_file(name, path)?;
        Ok(self)
    }

    /// Add metadata
    pub fn add_metadata(mut self, key: &str, value: &str) -> Self {
        self.package
            .manifest_mut()
            .metadata
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Set author
    pub fn author(mut self, author: String) -> Self {
        self.package.manifest_mut().author = Some(author);
        self
    }

    /// Set description
    pub fn description(mut self, description: String) -> Self {
        self.package.manifest_mut().description = Some(description);
        self
    }

    /// Set license
    pub fn license(mut self, license: String) -> Self {
        self.package.manifest_mut().license = Some(license);
        self
    }

    /// Add dependency
    pub fn add_dependency(mut self, name: &str, version: &str) -> Self {
        self.package.add_dependency(name, version);
        self
    }

    /// Build and save the package
    pub fn build<P: AsRef<Path>>(mut self, path: P) -> Result<()> {
        // Apply configuration
        if self.config.include_dependencies {
            self.package
                .add_dependency("torsh-core", env!("CARGO_PKG_VERSION"));
        }

        self.package.save(path)
    }

    /// Get the built package without saving
    pub fn package(self) -> Package {
        self.package
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_package_builder() {
        let builder = PackageBuilder::new("test".to_string(), "1.0.0".to_string())
            .add_metadata("author", "Test Author")
            .add_metadata("description", "Test package")
            .author("Builder Author".to_string())
            .description("Builder description".to_string())
            .license("MIT".to_string());

        let package = builder.package();
        assert!(package.metadata().metadata.contains_key("author"));
        assert_eq!(package.metadata().author.as_deref(), Some("Builder Author"));
        assert_eq!(
            package.metadata().description.as_deref(),
            Some("Builder description")
        );
        assert_eq!(package.metadata().license.as_deref(), Some("MIT"));
    }

    #[test]
    fn test_builder_config() {
        let config = BuilderConfig {
            include_source: true,
            compress: false,
            sign: true,
            include_dependencies: false,
        };

        let builder = PackageBuilder::new("test".to_string(), "1.0.0".to_string())
            .with_config(config.clone());

        assert_eq!(builder.config.include_source, true);
        assert_eq!(builder.config.compress, false);
        assert_eq!(builder.config.sign, true);
        assert_eq!(builder.config.include_dependencies, false);
    }

    #[test]
    fn test_builder_default_config() {
        let config = BuilderConfig::default();
        assert!(!config.include_source);
        assert!(config.compress);
        assert!(!config.sign);
        assert!(config.include_dependencies);
    }

    #[test]
    fn test_fluent_builder_api() {
        let builder = PackageBuilder::new("fluent_test".to_string(), "2.0.0".to_string())
            .author("Fluent Author".to_string())
            .description("Fluent description".to_string())
            .license("Apache-2.0".to_string())
            .add_dependency("tokio", "1.0")
            .add_dependency("serde", "1.0")
            .add_metadata("category", "ml")
            .add_metadata("keywords", "machine-learning,pytorch");

        let package = builder.package();
        let manifest = package.metadata();

        assert_eq!(manifest.name, "fluent_test");
        assert_eq!(manifest.version, "2.0.0");
        assert_eq!(manifest.author.as_deref(), Some("Fluent Author"));
        assert_eq!(manifest.description.as_deref(), Some("Fluent description"));
        assert_eq!(manifest.license.as_deref(), Some("Apache-2.0"));
        assert_eq!(manifest.dependencies.get("tokio"), Some(&"1.0".to_string()));
        assert_eq!(manifest.dependencies.get("serde"), Some(&"1.0".to_string()));
        assert_eq!(manifest.metadata.get("category"), Some(&"ml".to_string()));
        assert_eq!(
            manifest.metadata.get("keywords"),
            Some(&"machine-learning,pytorch".to_string())
        );
    }
}
