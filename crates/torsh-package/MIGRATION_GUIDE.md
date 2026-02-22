# PyTorch to ToRSh Package Migration Guide

This guide helps you migrate from PyTorch's `torch.package` format to ToRSh packages, providing equivalent functionality with enhanced features.

## Table of Contents

1. [Overview](#overview)
2. [Feature Comparison](#feature-comparison)
3. [Basic Migration](#basic-migration)
4. [Package Structure](#package-structure)
5. [Code Migration](#code-migration)
6. [Model Migration](#model-migration)
7. [Dependency Handling](#dependency-handling)
8. [Advanced Features](#advanced-features)
9. [Testing Migration](#testing-migration)
10. [Troubleshooting](#troubleshooting)

## Overview

ToRSh packages provide a Rust-native alternative to PyTorch's `torch.package` with additional enterprise features:

- **Type Safety**: Rust's compile-time guarantees
- **Performance**: Native performance without Python overhead
- **Security**: Built-in signing, encryption, and sandboxing
- **Governance**: ML model lineage tracking and compliance
- **Distribution**: CDN, mirrors, and high availability

## Feature Comparison

| Feature | torch.package | torsh-package |
|---------|--------------|---------------|
| **Basic Packaging** | ✅ | ✅ |
| **Model Weights** | ✅ | ✅ |
| **Code Bundling** | ✅ | ✅ |
| **Dependencies** | ✅ | ✅ (Enhanced) |
| **Compression** | Gzip | Gzip, Zstd, LZMA |
| **Signing** | ❌ | ✅ (Ed25519) |
| **Encryption** | ❌ | ✅ (AES-256, ChaCha20) |
| **Sandboxing** | ❌ | ✅ |
| **Lineage Tracking** | ❌ | ✅ |
| **Replication** | ❌ | ✅ |
| **Backup** | ❌ | ✅ |
| **CDN Support** | ❌ | ✅ |
| **Format** | ZIP | ZIP (compatible) |

## Basic Migration

### PyTorch Package

```python
import torch
from torch.package import PackageExporter, PackageImporter

# Create package
with PackageExporter("model.pt") as exp:
    exp.save_pickle("model", "main.pkl", model)
    exp.save_text("config", "config.json", config_json)

# Load package
imp = PackageImporter("model.pt")
model = imp.load_pickle("model", "main.pkl")
```

### ToRSh Package

```rust
use torsh_package::{Package, ResourceType};

// Create package
let mut package = Package::new("model".to_string(), "1.0.0".to_string());

// Add model weights
package.add_resource(
    "model_weights",
    model_data,
    ResourceType::ModelWeights,
)?;

// Add configuration
package.add_resource(
    "config",
    config_json.as_bytes(),
    ResourceType::Config,
)?;

// Save package
package.save("model.torshpkg")?;

// Load package
let package = Package::load("model.torshpkg")?;
let model_data = package.get_resource("model_weights")?;
```

## Package Structure

### PyTorch Package Structure

```
model.pt (ZIP archive)
├── .data/
│   ├── version
│   └── extern_modules
├── main/
│   ├── model.pkl
│   └── __init__.py
└── config/
    └── config.json
```

### ToRSh Package Structure

```
model.torshpkg (ZIP archive)
├── manifest.json          # Package metadata
├── resources/
│   ├── model_weights      # Serialized model
│   ├── config.json        # Configuration
│   └── code/              # Optional code
└── dependencies/          # Dependency information
```

## Code Migration

### Migrating Model Code

**PyTorch (Python):**

```python
# model.py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

# Package it
with PackageExporter("model.pt") as exp:
    exp.intern("torch.**")
    exp.save_module("model")
    exp.save_pickle("model", "model.pkl", model)
```

**ToRSh (Rust):**

```rust
use torsh_nn::{Module, Linear};
use torsh_tensor::Tensor;

pub struct MyModel {
    fc: Linear,
}

impl MyModel {
    pub fn new() -> Self {
        Self {
            fc: Linear::new(784, 10),
        }
    }
}

impl Module for MyModel {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.fc.forward(input)
    }
}

// Package it
let mut package = Package::new("my-model".to_string(), "1.0.0".to_string());

// Serialize model
let model_bytes = oxicode::encode(&model)?;
package.add_resource("model", &model_bytes, ResourceType::ModelWeights)?;

// Add source code
package.add_source_file("model", include_str!("model.rs"))?;

package.save("model.torshpkg")?;
```

## Model Migration

### Step 1: Export from PyTorch

```python
import torch
import numpy as np

# Load PyTorch model
model = MyModel()
model.load_state_dict(torch.load("model.pth"))

# Extract weights
weights = {}
for name, param in model.named_parameters():
    weights[name] = param.detach().cpu().numpy()

# Save as NumPy arrays (for import into Rust)
np.savez("model_weights.npz", **weights)

# Export to ONNX for compatibility
torch.onnx.export(
    model,
    torch.randn(1, 784),
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
)
```

### Step 2: Import into ToRSh

```rust
use torsh_package::{Package, FormatConverter, PackageFormat};

// Option 1: Convert from ONNX
let converter = FormatConverter::new();
let package = converter.convert_from_onnx("model.onnx")?;

// Option 2: Manually construct
let mut package = Package::new("my-model".to_string(), "1.0.0".to_string());

// Load and add weights
let weights_data = std::fs::read("model_weights.npz")?;
package.add_resource("weights", &weights_data, ResourceType::ModelWeights)?;

// Add metadata
package.set_description("Migrated from PyTorch");
package.set_author("Migration Tool");

package.save("model.torshpkg")?;
```

### Step 3: Load in ToRSh

```rust
// Load package
let package = Package::load("model.torshpkg")?;

// Extract weights
let weights_data = package.get_resource("weights")?;

// Reconstruct model
let model = MyModel::new();
model.load_weights(&weights_data)?;

// Use model
let output = model.forward(&input_tensor);
```

## Dependency Handling

### PyTorch Dependencies

```python
# requirements.txt
torch==2.0.0
numpy==1.24.0
pillow==10.0.0

# In package
with PackageExporter("model.pt") as exp:
    exp.extern(["numpy", "PIL"])
```

### ToRSh Dependencies

```rust
use torsh_package::{Package, DependencySpec};

let mut package = Package::new("my-model".to_string(), "1.0.0".to_string());

// Add dependencies
package.add_dependency(DependencySpec {
    name: "torsh-nn".to_string(),
    version_req: "^0.1.0".to_string(),
    optional: false,
    features: vec![],
})?;

package.add_dependency(DependencySpec {
    name: "torsh-vision".to_string(),
    version_req: "^0.1.0".to_string(),
    optional: true,
    features: vec!["image-processing".to_string()],
})?;

package.save("model.torshpkg")?;
```

## Advanced Features

### 1. Package Signing (Not in PyTorch)

```rust
use torsh_package::{PackageSigner, SignatureAlgorithm};

let signer = PackageSigner::new(SignatureAlgorithm::Ed25519);

// Sign package
let signature = signer.sign(&package_data)?;
package.set_signature(signature);

// Verify on load
let package = Package::load("model.torshpkg")?;
signer.verify_package(&package)?;
```

### 2. Encryption (Not in PyTorch)

```rust
use torsh_package::{PackageEncryptor, EncryptionAlgorithm};

let encryptor = PackageEncryptor::new(EncryptionAlgorithm::Aes256Gcm);

// Encrypt sensitive model
let encrypted = encryptor.encrypt_package(&package, "password")?;
encrypted.save("model_encrypted.torshpkg")?;

// Decrypt
let decrypted = encryptor.decrypt_package(&encrypted, "password")?;
```

### 3. Lineage Tracking (Not in PyTorch)

```rust
use torsh_package::{LineageTracker, LineageRelation, ProvenanceInfo};

let mut tracker = LineageTracker::new();

// Record that PyTorch model was migrated
tracker.add_lineage(
    "pytorch-model-v1".to_string(),
    "torsh-model-v1".to_string(),
    LineageRelation::ConvertedFrom,
    "Migrated from PyTorch to ToRSh".to_string(),
)?;

// Track provenance
let provenance = ProvenanceInfo {
    package_id: "torsh-model-v1".to_string(),
    creator: "migration-tool".to_string(),
    creation_time: chrono::Utc::now(),
    source_url: Some("original-pytorch-model-url".to_string()),
    source_commit: None,
    build_environment: [
        ("migration_tool".to_string(), "1.0.0".to_string()),
        ("pytorch_version".to_string(), "2.0.0".to_string()),
    ].iter().cloned().collect(),
    description: "Migrated PyTorch model".to_string(),
};

tracker.record_provenance(provenance);
```

### 4. Compression Options

```rust
use torsh_package::{CompressionAlgorithm, CompressionLevel};

// PyTorch only supports gzip
// ToRSh supports multiple algorithms

// High compression ratio
package.set_compression(CompressionAlgorithm::Lzma, CompressionLevel::Maximum)?;

// Fast compression
package.set_compression(CompressionAlgorithm::Zstd, CompressionLevel::Fast)?;

// Balanced
package.set_compression(CompressionAlgorithm::Gzip, CompressionLevel::Default)?;
```

## Testing Migration

### Automated Migration Test

```rust
use torsh_package::{Package, FormatConverter};

fn test_pytorch_migration(pytorch_package: &str, torsh_package: &str) -> Result<(), Box<dyn Error>> {
    // Step 1: Convert PyTorch package
    let converter = FormatConverter::new();
    let package = converter.convert_from_pytorch(pytorch_package)?;

    // Step 2: Save as ToRSh package
    package.save(torsh_package)?;

    // Step 3: Verify can be loaded
    let loaded = Package::load(torsh_package)?;

    // Step 4: Compare resources
    assert_eq!(package.list_resources().len(), loaded.list_resources().len());

    // Step 5: Verify weights match
    let original_weights = package.get_resource("weights")?;
    let loaded_weights = loaded.get_resource("weights")?;
    assert_eq!(original_weights.data, loaded_weights.data);

    println!("Migration successful!");
    Ok(())
}
```

### Model Output Comparison

```rust
// Load both PyTorch and ToRSh models
let pytorch_output = run_pytorch_model(&pytorch_model, &input);
let torsh_output = run_torsh_model(&torsh_model, &input);

// Compare outputs (with tolerance for floating point)
let diff = (pytorch_output - torsh_output).abs();
assert!(diff < 1e-5, "Model outputs differ significantly");
```

## Troubleshooting

### Issue: Weights Not Loading

**Problem**: Weights from PyTorch don't match ToRSh model structure

**Solution**:
```rust
// Map PyTorch weight names to ToRSh names
let weight_mapping = HashMap::from([
    ("fc1.weight", "layers.0.weight"),
    ("fc1.bias", "layers.0.bias"),
]);

for (pytorch_name, torsh_name) in weight_mapping {
    let weight = pytorch_weights.get(pytorch_name)?;
    torsh_model.set_parameter(torsh_name, weight)?;
}
```

### Issue: Different Tensor Layout

**Problem**: PyTorch uses NCHW, ToRSh might use different layout

**Solution**:
```rust
use torsh_tensor::Tensor;

// Transpose if needed
let pytorch_tensor = load_pytorch_tensor(&data);  // NCHW
let torsh_tensor = pytorch_tensor.permute(&[0, 2, 3, 1])?;  // NHWC
```

### Issue: Missing Dependencies

**Problem**: PyTorch package has external dependencies not in ToRSh

**Solution**:
```rust
// Option 1: Implement equivalent in Rust
// Option 2: Use FFI to call Python (via pyo3)
// Option 3: Find Rust alternative crate

// Add dependency with fallback
package.add_optional_dependency("alternative-crate", "1.0.0")?;
```

### Issue: Custom Operators

**Problem**: PyTorch package uses custom C++/CUDA operators

**Solution**:
```rust
// 1. Reimplement in Rust
// 2. Use torsh-backend-cuda for GPU operations
// 3. Provide fallback CPU implementation

use torsh_backends::Backend;

impl CustomOperator {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        match input.device() {
            Device::Cuda => self.cuda_implementation(input),
            Device::Cpu => self.cpu_implementation(input),
        }
    }
}
```

## Migration Checklist

- [ ] Export PyTorch model weights
- [ ] Convert to ToRSh format
- [ ] Verify weight loading
- [ ] Test model outputs match
- [ ] Add metadata and documentation
- [ ] Implement equivalent operations
- [ ] Handle dependencies
- [ ] Add signing/encryption if needed
- [ ] Set up lineage tracking
- [ ] Configure distribution
- [ ] Update deployment scripts
- [ ] Test in production environment

## Best Practices

1. **Incremental Migration**: Migrate one model at a time
2. **Keep Both Formats**: Maintain PyTorch version during transition
3. **Extensive Testing**: Verify numerical equivalence
4. **Document Differences**: Note any behavior changes
5. **Use Converters**: Leverage ONNX for complex models
6. **Version Control**: Track both formats in git
7. **Automated Testing**: CI/CD for migration validation
8. **Performance Testing**: Ensure no regression
9. **Gradual Rollout**: Deploy to staging first
10. **Rollback Plan**: Keep ability to revert

## Example: Complete Migration

```rust
use torsh_package::*;

fn migrate_complete_model() -> Result<(), Box<dyn Error>> {
    // 1. Load PyTorch exported data
    let weights = load_numpy_weights("pytorch_weights.npz")?;
    let config = load_json_config("config.json")?;

    // 2. Create ToRSh package
    let mut package = Package::new("migrated-model".to_string(), "1.0.0".to_string());
    package.set_author("Migration Tool".to_string());
    package.set_description("Migrated from PyTorch 2.0".to_string());

    // 3. Add resources
    package.add_resource("weights", &weights, ResourceType::ModelWeights)?;
    package.add_resource("config", config.as_bytes(), ResourceType::Config)?;

    // 4. Add dependencies
    package.add_dependency_spec("torsh-nn", "^0.1.0")?;
    package.add_dependency_spec("torsh-vision", "^0.1.0")?;

    // 5. Sign package
    let signer = PackageSigner::new(SignatureAlgorithm::Ed25519);
    let signature = signer.sign(&serialize_package(&package)?)?;
    package.set_signature(signature);

    // 6. Save
    package.save("migrated_model.torshpkg")?;

    // 7. Track lineage
    let mut tracker = LineageTracker::new();
    tracker.add_lineage(
        "pytorch-model".to_string(),
        "migrated-model".to_string(),
        LineageRelation::ConvertedFrom,
        "Migrated from PyTorch".to_string(),
    )?;

    println!("Migration complete!");
    Ok(())
}
```

## Additional Resources

- [ToRSh Package API Documentation](https://docs.rs/torsh-package)
- [PyTorch torch.package Documentation](https://pytorch.org/docs/stable/package.html)
- [ONNX Format Specification](https://onnx.ai/)
- [ToRSh Examples](./examples/)
- [Distribution Guide](./DISTRIBUTION_GUIDE.md)

## Getting Help

If you encounter issues during migration:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review existing [examples](./examples/)
3. Open an issue on GitHub
4. Join the ToRSh community discussions

## Conclusion

Migrating from PyTorch packages to ToRSh provides:
- **Better Performance**: Native Rust implementation
- **Enhanced Security**: Built-in signing and encryption
- **Enterprise Features**: Lineage, compliance, HA
- **Type Safety**: Compile-time guarantees
- **Production Ready**: Monitoring, backup, replication

Follow this guide for a smooth migration path!
