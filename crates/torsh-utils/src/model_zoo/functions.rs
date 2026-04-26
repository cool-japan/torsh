//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::types::{ModelInfo, ModelZoo};

/// List all available models
pub fn list_available_models() -> Vec<ModelInfo> {
    let zoo = ModelZoo::new(std::env::temp_dir())
        .expect("Failed to initialize ModelZoo with temp directory");
    zoo.list_models().into_iter().cloned().collect()
}
/// Load a pre-trained model
pub fn load_pretrained(name: &str) -> Result<Box<dyn torsh_nn::Module>> {
    let zoo = ModelZoo::new(std::env::temp_dir())?;
    zoo.load_model(name)
}
/// Print model zoo catalog
pub fn print_model_catalog() {
    let models = list_available_models();
    println!("=== ToRSh Model Zoo ===");
    println!();
    println!(
        "{:<20} {:<15} {:<10} {:<15} {:<40}",
        "Name", "Architecture", "Size (MB)", "Top-1 Acc", "Description"
    );
    println!("{}", "-".repeat(120));
    for model in models {
        let top1 = model
            .metrics
            .get("top1_accuracy")
            .map(|v| format!("{:.2}%", v))
            .unwrap_or_else(|| "N/A".to_string());
        println!(
            "{:<20} {:<15} {:<10.1} {:<15} {:<40}",
            model.name, model.architecture, model.size_mb, top1, model.description
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_zoo::{
        DependencyType, HardwareRequirements, ModelConfig, ModelDependency, ModelSearchQuery,
        ModelVersion, RegistryConfig, RetryConfig, UserPreferences,
    };
    use std::collections::HashMap;
    use tempfile::TempDir;
    #[test]
    fn test_model_zoo() {
        let temp_dir = TempDir::new().expect("Temp Dir should succeed");
        let zoo = ModelZoo::new(temp_dir.path()).expect("operation should succeed");
        let models = zoo.list_models();
        assert!(!models.is_empty());
        let resnet18 = zoo
            .get_model_info("resnet18")
            .expect("model info retrieval should succeed");
        assert_eq!(resnet18.architecture, "ResNet");
        assert_eq!(resnet18.config.num_classes, 1000);
    }
    #[test]
    fn test_model_download() {
        let temp_dir = TempDir::new().expect("Temp Dir should succeed");
        let mut zoo = ModelZoo::new(temp_dir.path()).expect("operation should succeed");
        let dummy_model_content = b"dummy model data";
        let dummy_model_path = temp_dir.path().join("dummy_model.torsh");
        std::fs::write(&dummy_model_path, dummy_model_content).expect("fs should succeed");
        let test_model = ModelInfo {
            name: "test_local_model".to_string(),
            architecture: "TestNet".to_string(),
            version: "1.0".to_string(),
            description: "Local test model".to_string(),
            url: format!("file://{}", dummy_model_path.display()),
            size_mb: 0.001,
            sha256: "test_hash".to_string(),
            author: "test".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            tags: vec!["test".to_string()],
            hardware_requirements: HardwareRequirements::default(),
            metrics: HashMap::new(),
            dependencies: Vec::new(),
            license: "MIT".to_string(),
            frameworks: vec!["torsh".to_string()],
            versions: vec![ModelVersion {
                version: "1.0".to_string(),
                url: format!("file://{}", dummy_model_path.display()),
                size_mb: 0.001,
                sha256: "test_hash".to_string(),
                metrics: HashMap::new(),
                changelog: "Test version".to_string(),
                deprecated: false,
            }],
            config: ModelConfig {
                num_classes: 10,
                input_size: vec![3, 32, 32],
                pretrained: true,
                checkpoint_format: "torsh".to_string(),
            },
        };
        zoo.register_model(test_model);
        let result = zoo.download_model("test_local_model", false);
        match result {
            Ok(path) => {
                assert!(path.exists());
            }
            Err(_) => {}
        }
    }
    #[test]
    fn test_enhanced_model_zoo_features() {
        let temp_dir = TempDir::new().expect("Temp Dir should succeed");
        let config = RegistryConfig {
            mirrors: vec!["https://mirror1.test.com".to_string()],
            retry_config: RetryConfig {
                max_attempts: 2,
                initial_delay_ms: 100,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut zoo =
            ModelZoo::with_config(temp_dir.path(), config).expect("operation should succeed");
        assert!(!zoo.mirror_health.is_empty());
        let deps = zoo
            .resolve_dependencies("resnet18")
            .expect("dependency resolution should succeed");
        assert!(!deps.is_empty());
        assert!(deps.iter().any(|d| d.name == "torsh-vision"));
        let synced = zoo
            .sync_with_huggingface(Some("microsoft"))
            .expect("operation should succeed");
        assert!(!synced.is_empty());
    }
    #[test]
    fn test_model_recommendations() {
        let temp_dir = TempDir::new().expect("Temp Dir should succeed");
        let zoo = ModelZoo::new(temp_dir.path()).expect("operation should succeed");
        let preferences = UserPreferences {
            preferred_architecture: Some("ResNet".to_string()),
            preferred_tasks: vec!["classification".to_string()],
            performance_weight: 0.8,
            efficiency_weight: 0.2,
            hardware_constraints: HardwareRequirements {
                min_ram_gb: 4.0,
                recommended_ram_gb: 8.0,
                gpu_memory_gb: None,
                compute_capabilities: vec![],
                cpu_arch: vec!["x86_64".to_string()],
            },
            ..Default::default()
        };
        let recommendations = zoo.get_recommendations(&preferences, Some(5));
        assert!(!recommendations.is_empty());
        let resnet_models = recommendations
            .iter()
            .filter(|m| m.architecture == "ResNet")
            .count();
        assert!(resnet_models > 0);
    }
    #[test]
    fn test_advanced_search() {
        let temp_dir = TempDir::new().expect("Temp Dir should succeed");
        let zoo = ModelZoo::new(temp_dir.path()).expect("operation should succeed");
        let query = ModelSearchQuery::new()
            .architecture("ResNet")
            .tag("classification")
            .size_range(0.0, 100.0)
            .min_metric("top1_accuracy".to_string(), 70.0)
            .license("MIT");
        let results = zoo.search_models(&query);
        assert!(!results.is_empty());
        for model in &results {
            assert_eq!(model.architecture, "ResNet");
            assert!(model.tags.contains(&"classification".to_string()));
            assert!(model.size_mb <= 100.0);
            assert!(model.license.contains("MIT"));
            if let Some(accuracy) = model.metrics.get("top1_accuracy") {
                assert!(*accuracy >= 70.0);
            }
        }
    }
    #[test]
    fn test_version_management() {
        let temp_dir = TempDir::new().expect("Temp Dir should succeed");
        let zoo = ModelZoo::new(temp_dir.path()).expect("operation should succeed");
        let version_1_0 = zoo.get_model_version("resnet18", "1.0");
        assert!(version_1_0.is_some());
        let version_1_1 = zoo.get_model_version("resnet18", "1.1");
        assert!(version_1_1.is_some());
        let version_info = version_1_1.expect("operation should succeed");
        assert_eq!(version_info.version, "1.1");
        let versions = zoo.get_model_versions("resnet18");
        assert!(versions.is_some());
        let versions = versions.expect("operation should succeed");
        assert!(!versions.is_empty());
    }
    #[test]
    fn test_dependency_types() {
        let runtime_dep = ModelDependency {
            name: "torsh-core".to_string(),
            version_constraint: ">=0.1.0".to_string(),
            optional: false,
            platform_specific: None,
            dependency_type: DependencyType::Runtime,
        };
        assert!(!runtime_dep.optional);
        assert!(matches!(
            runtime_dep.dependency_type,
            DependencyType::Runtime
        ));
        let optional_dep = ModelDependency {
            name: "cuda".to_string(),
            version_constraint: ">=11.0".to_string(),
            optional: true,
            platform_specific: Some("gpu".to_string()),
            dependency_type: DependencyType::Optional,
        };
        assert!(optional_dep.optional);
        assert!(matches!(
            optional_dep.dependency_type,
            DependencyType::Optional
        ));
    }
    #[test]
    fn test_registry_config_defaults() {
        let config = RegistryConfig::default();
        assert_eq!(config.timeout_seconds, 300);
        assert_eq!(config.max_concurrent, 4);
        assert!(config.compression);
        assert!(!config.mirrors.is_empty());
        assert_eq!(config.retry_config.max_attempts, 3);
        assert!(!config.enable_p2p);
        assert!(config.user_agent.contains("ToRSh"));
    }
    #[test]
    fn test_mirror_region_detection() {
        assert_eq!(
            ModelZoo::detect_mirror_region("https://us-west-1.example.com"),
            Some("US".to_string())
        );
        assert_eq!(
            ModelZoo::detect_mirror_region("https://eu-central-1.example.com"),
            Some("EU".to_string())
        );
        assert_eq!(
            ModelZoo::detect_mirror_region("https://asia-pacific.example.com"),
            Some("Asia".to_string())
        );
        assert_eq!(
            ModelZoo::detect_mirror_region("https://generic.example.com"),
            None
        );
    }
}
