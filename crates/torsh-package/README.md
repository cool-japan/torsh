# torsh-package

Model packaging and distribution system for the ToRSh deep learning framework.

## Overview

torsh-package provides functionality similar to PyTorch's `torch.package` for creating self-contained model packages that include code, weights, dependencies, and metadata. This enables easy model distribution, deployment, and reproducibility across different environments.

## Features

- **Self-contained Packages**: Bundle models with code, weights, and dependencies
- **Version Management**: Package versioning with compatibility tracking
- **Resource Management**: Efficient storage and retrieval of model assets
- **Metadata Tracking**: Comprehensive manifest with dependency information
- **Cross-platform**: Platform-independent package format
- **Incremental Updates**: Support for package updates and patches

## Modules

- `exporter`: Package creation and export functionality
- `importer`: Package loading and import functionality
- `manifest`: Package metadata and dependency management
- `resources`: Resource storage and retrieval system
- `version`: Package versioning and compatibility

## Usage

### Creating a Package

```rust
use torsh_package::prelude::*;
use torsh_nn::Module;

// Create package exporter
let config = ExportConfig {
    include_code: true,
    include_weights: true,
    compression: CompressionType::Gzip,
    metadata: PackageMetadata::default(),
};

let mut exporter = PackageExporter::new(config);

// Add model to package
exporter.add_model("my_model", &model)?;

// Add additional resources
exporter.add_resource("config.json", ResourceType::Json, config_data)?;
exporter.add_resource("preprocessing.py", ResourceType::Python, preprocess_code)?;

// Export package
exporter.export_to_file("my_model_package.torsh")?;
```

### Loading a Package

```rust
use torsh_package::prelude::*;

// Load package
let package = Package::load("my_model_package.torsh")?;

// Get model from package
let model = package.get_model("my_model")?;

// Get additional resources
let config = package.get_resource("config.json")?;
let preprocess_code = package.get_resource("preprocessing.py")?;

// Check package metadata
println!("Package: {} v{}", package.name(), package.version());
println!("Created: {}", package.created_at());
println!("Dependencies: {:?}", package.dependencies());
```

### Package Information

```rust
use torsh_package::prelude::*;

// Load package without extracting
let package_info = PackageImporter::inspect("my_model_package.torsh")?;

println!("Package manifest:");
println!("  Name: {}", package_info.manifest.name);
println!("  Version: {}", package_info.manifest.version);
println!("  Models: {:?}", package_info.manifest.models);
println!("  Resources: {:?}", package_info.manifest.resources);
println!("  Size: {} bytes", package_info.size);
```

### Version Management

```rust
use torsh_package::prelude::*;

// Check version compatibility
let package_version = PackageVersion::parse("1.2.3")?;
let required_version = PackageVersion::parse(">=1.0.0,<2.0.0")?;

if package_version.satisfies(&required_version) {
    println!("Package version is compatible");
} else {
    println!("Package version incompatibility detected");
}

// Update package
let mut updater = PackageUpdater::new("my_model_package.torsh")?;
updater.update_model("my_model", &new_model)?;
updater.increment_version(VersionIncrement::Patch)?;
updater.save("my_model_package_v1.2.4.torsh")?;
```

### Resource Types

```rust
use torsh_package::prelude::*;

// Supported resource types
let resources = vec![
    ("model.safetensors", ResourceType::Model),
    ("config.json", ResourceType::Json),
    ("tokenizer.json", ResourceType::Tokenizer),
    ("preprocessing.py", ResourceType::Python),
    ("data.csv", ResourceType::Data),
    ("image.png", ResourceType::Binary),
];

for (name, resource_type) in resources {
    exporter.add_resource(name, resource_type, data)?;
}
```

## Package Format

The torsh-package format uses a structured archive containing:

```
package.torsh
├── manifest.json          # Package metadata and dependency info
├── models/                # Model weights and architectures
│   └── my_model.safetensors
├── code/                  # Python/Rust code files
│   └── preprocessing.py
├── resources/             # Additional resources
│   ├── config.json
│   └── tokenizer.json
└── metadata/              # Version and compatibility info
    └── package.info
```

## Advanced Features

### Incremental Updates

```rust
use torsh_package::prelude::*;

// Create incremental update
let patch = PackagePatch::new("my_model_package.torsh")?
    .update_model("my_model", &updated_model)?
    .add_resource("new_config.json", ResourceType::Json, new_config)?
    .remove_resource("old_file.txt")?;

// Apply patch
patch.save_as_patch("update_v1.2.4.patch")?;

// Apply patch to existing package
let updated_package = Package::load("my_model_package.torsh")?
    .apply_patch("update_v1.2.4.patch")?;
```

### Dependency Management

```rust
use torsh_package::prelude::*;

// Specify dependencies
let dependencies = Dependencies::new()
    .add_torsh_version(">=0.1.0,<0.2.0")?
    .add_python_package("numpy", ">=1.20.0")?
    .add_python_package("transformers", ">=4.20.0")?
    .add_system_requirement("cuda", ">=11.0")?;

exporter.set_dependencies(dependencies)?;
```

## Dependencies

- `torsh-core`: Core types and error handling
- `torsh-nn`: Neural network modules
- `serde`: Serialization support
- `zip`: Archive compression and extraction
- `chrono`: Date and time handling
- `semver`: Semantic versioning

## Performance

torsh-package is optimized for:
- Fast package creation with streaming compression
- Lazy loading of resources to minimize memory usage
- Efficient delta updates for large models
- Parallel compression and decompression

## Compatibility

- **PyTorch**: Compatible with torch.package format (import/export)
- **HuggingFace**: Integration with HuggingFace Hub model format
- **ONNX**: Support for ONNX model packaging
- **MLflow**: Integration with MLflow model registry

## Security

- Package integrity verification with checksums
- Code signing support for trusted packages
- Sandboxed execution environment for untrusted code
- Dependency vulnerability scanning

## Examples

See the `examples/` directory for:
- Complete model packaging workflows
- Package distribution and deployment
- Version management and updates
- Integration with model registries