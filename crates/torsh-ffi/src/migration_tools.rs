//! Migration tools for transitioning from other ML frameworks to ToRSh
//!
//! This module provides comprehensive tools to migrate code, models, and data
//! from PyTorch, TensorFlow, JAX, NumPy, and other ML frameworks to ToRSh.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Supported source frameworks for migration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceFramework {
    PyTorch,
    TensorFlow,
    JAX,
    NumPy,
    Keras,
    Scikit,
    Pandas,
    ONNX,
    Other(String),
}

/// Migration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    pub source_framework: SourceFramework,
    pub target_language: String,
    pub preserve_comments: bool,
    pub generate_tests: bool,
    pub optimization_level: OptimizationLevel,
    pub include_documentation: bool,
    pub batch_size: usize,
    pub parallel_processing: bool,
}

/// Optimization level for migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Custom(HashMap<String, bool>),
}

/// Migration result summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationResult {
    pub source_files: Vec<PathBuf>,
    pub target_files: Vec<PathBuf>,
    pub converted_functions: usize,
    pub converted_classes: usize,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub manual_changes_required: Vec<String>,
    pub success_rate: f64,
}

/// Code pattern replacement rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplacementRule {
    pub pattern: String,
    pub replacement: String,
    pub framework: SourceFramework,
    pub language: String,
    pub description: String,
    pub requires_manual_review: bool,
}

/// Main migration tool
pub struct MigrationTool {
    config: MigrationConfig,
    replacement_rules: Vec<ReplacementRule>,
    type_mappings: HashMap<String, String>,
    function_mappings: HashMap<String, String>,
}

impl MigrationTool {
    /// Create a new migration tool
    pub fn new(config: MigrationConfig) -> Self {
        let mut tool = Self {
            config,
            replacement_rules: Vec::new(),
            type_mappings: HashMap::new(),
            function_mappings: HashMap::new(),
        };

        tool.initialize_mappings();
        tool.load_replacement_rules();
        tool
    }

    /// Initialize framework-specific mappings
    fn initialize_mappings(&mut self) {
        match self.config.source_framework {
            SourceFramework::PyTorch => self.init_pytorch_mappings(),
            SourceFramework::TensorFlow => self.init_tensorflow_mappings(),
            SourceFramework::JAX => self.init_jax_mappings(),
            SourceFramework::NumPy => self.init_numpy_mappings(),
            SourceFramework::Keras => self.init_keras_mappings(),
            SourceFramework::Scikit => self.init_scikit_mappings(),
            SourceFramework::Pandas => self.init_pandas_mappings(),
            SourceFramework::ONNX => self.init_onnx_mappings(),
            SourceFramework::Other(_) => self.init_generic_mappings(),
        }
    }

    /// Initialize PyTorch to ToRSh mappings
    fn init_pytorch_mappings(&mut self) {
        // Type mappings
        self.type_mappings
            .insert("torch.Tensor".to_string(), "Tensor".to_string());
        self.type_mappings
            .insert("torch.nn.Module".to_string(), "Module".to_string());
        self.type_mappings
            .insert("torch.optim.Optimizer".to_string(), "Optimizer".to_string());
        self.type_mappings.insert(
            "torch.utils.data.DataLoader".to_string(),
            "DataLoader".to_string(),
        );

        // Function mappings
        self.function_mappings
            .insert("torch.zeros".to_string(), "zeros".to_string());
        self.function_mappings
            .insert("torch.ones".to_string(), "ones".to_string());
        self.function_mappings
            .insert("torch.randn".to_string(), "randn".to_string());
        self.function_mappings
            .insert("torch.rand".to_string(), "rand".to_string());
        self.function_mappings
            .insert("torch.eye".to_string(), "eye".to_string());
        self.function_mappings
            .insert("torch.cat".to_string(), "cat".to_string());
        self.function_mappings
            .insert("torch.stack".to_string(), "stack".to_string());
        self.function_mappings
            .insert("F.relu".to_string(), "relu".to_string());
        self.function_mappings
            .insert("F.softmax".to_string(), "softmax".to_string());
        self.function_mappings
            .insert("F.cross_entropy".to_string(), "cross_entropy".to_string());
        self.function_mappings
            .insert("torch.matmul".to_string(), "matmul".to_string());
        self.function_mappings
            .insert("torch.mm".to_string(), "mm".to_string());
        self.function_mappings
            .insert("torch.bmm".to_string(), "bmm".to_string());
    }

    /// Initialize TensorFlow to ToRSh mappings
    fn init_tensorflow_mappings(&mut self) {
        // Type mappings
        self.type_mappings
            .insert("tf.Tensor".to_string(), "Tensor".to_string());
        self.type_mappings
            .insert("tf.keras.Model".to_string(), "Module".to_string());
        self.type_mappings
            .insert("tf.keras.layers.Layer".to_string(), "Module".to_string());
        self.type_mappings
            .insert("tf.data.Dataset".to_string(), "Dataset".to_string());

        // Function mappings
        self.function_mappings
            .insert("tf.zeros".to_string(), "zeros".to_string());
        self.function_mappings
            .insert("tf.ones".to_string(), "ones".to_string());
        self.function_mappings
            .insert("tf.random.normal".to_string(), "randn".to_string());
        self.function_mappings
            .insert("tf.random.uniform".to_string(), "rand".to_string());
        self.function_mappings
            .insert("tf.eye".to_string(), "eye".to_string());
        self.function_mappings
            .insert("tf.concat".to_string(), "cat".to_string());
        self.function_mappings
            .insert("tf.stack".to_string(), "stack".to_string());
        self.function_mappings
            .insert("tf.nn.relu".to_string(), "relu".to_string());
        self.function_mappings
            .insert("tf.nn.softmax".to_string(), "softmax".to_string());
        self.function_mappings.insert(
            "tf.nn.sparse_softmax_cross_entropy_with_logits".to_string(),
            "cross_entropy".to_string(),
        );
        self.function_mappings
            .insert("tf.linalg.matmul".to_string(), "matmul".to_string());
    }

    /// Initialize JAX to ToRSh mappings
    fn init_jax_mappings(&mut self) {
        // Type mappings
        self.type_mappings
            .insert("jax.Array".to_string(), "Tensor".to_string());
        self.type_mappings
            .insert("flax.linen.Module".to_string(), "Module".to_string());

        // Function mappings
        self.function_mappings
            .insert("jnp.zeros".to_string(), "zeros".to_string());
        self.function_mappings
            .insert("jnp.ones".to_string(), "ones".to_string());
        self.function_mappings
            .insert("jax.random.normal".to_string(), "randn".to_string());
        self.function_mappings
            .insert("jax.random.uniform".to_string(), "rand".to_string());
        self.function_mappings
            .insert("jnp.eye".to_string(), "eye".to_string());
        self.function_mappings
            .insert("jnp.concatenate".to_string(), "cat".to_string());
        self.function_mappings
            .insert("jnp.stack".to_string(), "stack".to_string());
        self.function_mappings
            .insert("jax.nn.relu".to_string(), "relu".to_string());
        self.function_mappings
            .insert("jax.nn.softmax".to_string(), "softmax".to_string());
        self.function_mappings
            .insert("jnp.matmul".to_string(), "matmul".to_string());
    }

    /// Initialize NumPy to ToRSh mappings
    fn init_numpy_mappings(&mut self) {
        // Type mappings
        self.type_mappings
            .insert("np.ndarray".to_string(), "Tensor".to_string());
        self.type_mappings
            .insert("numpy.ndarray".to_string(), "Tensor".to_string());

        // Function mappings
        self.function_mappings
            .insert("np.zeros".to_string(), "zeros".to_string());
        self.function_mappings
            .insert("np.ones".to_string(), "ones".to_string());
        self.function_mappings
            .insert("np.random.randn".to_string(), "randn".to_string());
        self.function_mappings
            .insert("np.random.rand".to_string(), "rand".to_string());
        self.function_mappings
            .insert("np.eye".to_string(), "eye".to_string());
        self.function_mappings
            .insert("np.concatenate".to_string(), "cat".to_string());
        self.function_mappings
            .insert("np.stack".to_string(), "stack".to_string());
        self.function_mappings
            .insert("np.maximum".to_string(), "relu".to_string()); // Approximation
        self.function_mappings
            .insert("np.matmul".to_string(), "matmul".to_string());
        self.function_mappings
            .insert("np.dot".to_string(), "mm".to_string());
    }

    /// Initialize Keras to ToRSh mappings
    fn init_keras_mappings(&mut self) {
        // Type mappings
        self.type_mappings
            .insert("keras.Model".to_string(), "Module".to_string());
        self.type_mappings
            .insert("keras.layers.Layer".to_string(), "Module".to_string());
        self.type_mappings
            .insert("keras.layers.Dense".to_string(), "Linear".to_string());
        self.type_mappings
            .insert("keras.layers.Conv2D".to_string(), "Conv2d".to_string());
        self.type_mappings
            .insert("keras.layers.LSTM".to_string(), "LSTM".to_string());

        // Function mappings
        self.function_mappings
            .insert("keras.activations.relu".to_string(), "relu".to_string());
        self.function_mappings.insert(
            "keras.activations.softmax".to_string(),
            "softmax".to_string(),
        );
        self.function_mappings.insert(
            "keras.losses.categorical_crossentropy".to_string(),
            "cross_entropy".to_string(),
        );
        self.function_mappings
            .insert("keras.optimizers.Adam".to_string(), "Adam".to_string());
        self.function_mappings
            .insert("keras.optimizers.SGD".to_string(), "SGD".to_string());
    }

    /// Initialize Scikit-learn to ToRSh mappings
    fn init_scikit_mappings(&mut self) {
        // Type mappings (approximate mappings since scikit is different paradigm)
        self.type_mappings.insert(
            "sklearn.linear_model.LinearRegression".to_string(),
            "Linear".to_string(),
        );
        self.type_mappings.insert(
            "sklearn.linear_model.LogisticRegression".to_string(),
            "Linear + Sigmoid".to_string(),
        );
        self.type_mappings.insert(
            "sklearn.neural_network.MLPClassifier".to_string(),
            "Sequential".to_string(),
        );

        // Function mappings
        self.function_mappings.insert(
            "sklearn.preprocessing.StandardScaler".to_string(),
            "normalize".to_string(),
        );
        self.function_mappings.insert(
            "sklearn.model_selection.train_test_split".to_string(),
            "DataLoader.split".to_string(),
        );
    }

    /// Initialize Pandas to ToRSh mappings
    fn init_pandas_mappings(&mut self) {
        // Type mappings
        self.type_mappings
            .insert("pd.DataFrame".to_string(), "TensorDataset".to_string());
        self.type_mappings
            .insert("pd.Series".to_string(), "Tensor".to_string());

        // Function mappings
        self.function_mappings
            .insert("pd.concat".to_string(), "cat".to_string());
        self.function_mappings
            .insert("pd.merge".to_string(), "cat".to_string()); // Approximation
        self.function_mappings
            .insert("df.values".to_string(), "tensor_data".to_string());
    }

    /// Initialize ONNX to ToRSh mappings
    fn init_onnx_mappings(&mut self) {
        // Type mappings
        self.type_mappings
            .insert("onnx.TensorProto".to_string(), "Tensor".to_string());
        self.type_mappings
            .insert("onnx.ModelProto".to_string(), "Module".to_string());

        // Function mappings - ONNX operators to ToRSh
        self.function_mappings
            .insert("Add".to_string(), "add".to_string());
        self.function_mappings
            .insert("Sub".to_string(), "sub".to_string());
        self.function_mappings
            .insert("Mul".to_string(), "mul".to_string());
        self.function_mappings
            .insert("Div".to_string(), "div".to_string());
        self.function_mappings
            .insert("MatMul".to_string(), "matmul".to_string());
        self.function_mappings
            .insert("Relu".to_string(), "relu".to_string());
        self.function_mappings
            .insert("Softmax".to_string(), "softmax".to_string());
        self.function_mappings
            .insert("Conv".to_string(), "conv2d".to_string());
    }

    /// Initialize generic mappings
    fn init_generic_mappings(&mut self) {
        // Basic mathematical operations
        self.function_mappings
            .insert("add".to_string(), "add".to_string());
        self.function_mappings
            .insert("subtract".to_string(), "sub".to_string());
        self.function_mappings
            .insert("multiply".to_string(), "mul".to_string());
        self.function_mappings
            .insert("divide".to_string(), "div".to_string());
    }

    /// Load framework-specific replacement rules
    fn load_replacement_rules(&mut self) {
        match self.config.source_framework {
            SourceFramework::PyTorch => self.load_pytorch_rules(),
            SourceFramework::TensorFlow => self.load_tensorflow_rules(),
            SourceFramework::JAX => self.load_jax_rules(),
            SourceFramework::NumPy => self.load_numpy_rules(),
            SourceFramework::Keras => self.load_keras_rules(),
            SourceFramework::Scikit => self.load_scikit_rules(),
            SourceFramework::Pandas => self.load_pandas_rules(),
            SourceFramework::ONNX => self.load_onnx_rules(),
            SourceFramework::Other(_) => self.load_generic_rules(),
        }
    }

    /// Load PyTorch-specific replacement rules
    fn load_pytorch_rules(&mut self) {
        self.replacement_rules.extend(vec![
            ReplacementRule {
                pattern: r"import torch".to_string(),
                replacement: "use torsh::prelude::*;".to_string(),
                framework: SourceFramework::PyTorch,
                language: "python".to_string(),
                description: "Replace PyTorch import with ToRSh prelude".to_string(),
                requires_manual_review: false,
            },
            ReplacementRule {
                pattern: r"torch\.nn\.Module".to_string(),
                replacement: "Module".to_string(),
                framework: SourceFramework::PyTorch,
                language: "python".to_string(),
                description: "Replace torch.nn.Module with ToRSh Module trait".to_string(),
                requires_manual_review: true,
            },
            ReplacementRule {
                pattern: r"\.cuda\(\)".to_string(),
                replacement: ".to_device(DeviceType::Cuda)".to_string(),
                framework: SourceFramework::PyTorch,
                language: "python".to_string(),
                description: "Replace .cuda() with ToRSh device transfer".to_string(),
                requires_manual_review: false,
            },
            ReplacementRule {
                pattern: r"\.backward\(\)".to_string(),
                replacement: ".backward()?".to_string(),
                framework: SourceFramework::PyTorch,
                language: "python".to_string(),
                description: "Replace .backward() with error-handling version".to_string(),
                requires_manual_review: false,
            },
            ReplacementRule {
                pattern: r"optimizer\.zero_grad\(\)".to_string(),
                replacement: "optimizer.zero_grad()?".to_string(),
                framework: SourceFramework::PyTorch,
                language: "python".to_string(),
                description: "Replace optimizer.zero_grad() with error-handling version"
                    .to_string(),
                requires_manual_review: false,
            },
            ReplacementRule {
                pattern: r"F\.(\w+)\(([^)]+)\)".to_string(),
                replacement: "$1($2)".to_string(),
                framework: SourceFramework::PyTorch,
                language: "python".to_string(),
                description: "Replace F.function_name with direct function call".to_string(),
                requires_manual_review: true,
            },
        ]);
    }

    /// Load TensorFlow-specific replacement rules
    fn load_tensorflow_rules(&mut self) {
        self.replacement_rules.extend(vec![
            ReplacementRule {
                pattern: r"import tensorflow as tf".to_string(),
                replacement: "use torsh::prelude::*;".to_string(),
                framework: SourceFramework::TensorFlow,
                language: "python".to_string(),
                description: "Replace TensorFlow import with ToRSh prelude".to_string(),
                requires_manual_review: false,
            },
            ReplacementRule {
                pattern: r"tf\.keras\.Model".to_string(),
                replacement: "Module".to_string(),
                framework: SourceFramework::TensorFlow,
                language: "python".to_string(),
                description: "Replace tf.keras.Model with ToRSh Module trait".to_string(),
                requires_manual_review: true,
            },
            ReplacementRule {
                pattern: r"\.compile\(([^)]+)\)".to_string(),
                replacement: "// Compilation is implicit in ToRSh\n// Original: .compile($1)"
                    .to_string(),
                framework: SourceFramework::TensorFlow,
                language: "python".to_string(),
                description: "Remove .compile() call (not needed in ToRSh)".to_string(),
                requires_manual_review: true,
            },
            ReplacementRule {
                pattern: r"\.fit\(([^)]+)\)".to_string(),
                replacement: "// Replace with training loop\n// Original: .fit($1)".to_string(),
                framework: SourceFramework::TensorFlow,
                language: "python".to_string(),
                description: "Replace .fit() with explicit training loop".to_string(),
                requires_manual_review: true,
            },
        ]);
    }

    /// Load JAX-specific replacement rules
    fn load_jax_rules(&mut self) {
        self.replacement_rules.extend(vec![
            ReplacementRule {
                pattern: r"import jax\.numpy as jnp".to_string(),
                replacement: "use torsh::prelude::*;".to_string(),
                framework: SourceFramework::JAX,
                language: "python".to_string(),
                description: "Replace JAX NumPy import with ToRSh prelude".to_string(),
                requires_manual_review: false,
            },
            ReplacementRule {
                pattern: r"@jax\.jit".to_string(),
                replacement: "// JIT compilation is handled automatically in ToRSh".to_string(),
                framework: SourceFramework::JAX,
                language: "python".to_string(),
                description: "Remove @jax.jit decorator (automatic in ToRSh)".to_string(),
                requires_manual_review: true,
            },
            ReplacementRule {
                pattern: r"jax\.grad\(([^)]+)\)".to_string(),
                replacement:
                    "// Use autograd in ToRSh: tensor.backward()\n// Original: jax.grad($1)"
                        .to_string(),
                framework: SourceFramework::JAX,
                language: "python".to_string(),
                description: "Replace jax.grad with ToRSh autograd".to_string(),
                requires_manual_review: true,
            },
        ]);
    }

    /// Load NumPy-specific replacement rules
    fn load_numpy_rules(&mut self) {
        self.replacement_rules.extend(vec![
            ReplacementRule {
                pattern: r"import numpy as np".to_string(),
                replacement: "use torsh::prelude::*;".to_string(),
                framework: SourceFramework::NumPy,
                language: "python".to_string(),
                description: "Replace NumPy import with ToRSh prelude".to_string(),
                requires_manual_review: false,
            },
            ReplacementRule {
                pattern: r"np\.array\(([^)]+)\)".to_string(),
                replacement: "Tensor::from_data($1)?".to_string(),
                framework: SourceFramework::NumPy,
                language: "python".to_string(),
                description: "Replace np.array with Tensor::from_data".to_string(),
                requires_manual_review: false,
            },
            ReplacementRule {
                pattern: ".shape".to_string(),
                replacement: ".shape().dims()".to_string(),
                framework: SourceFramework::NumPy,
                language: "python".to_string(),
                description: "Replace .shape with ToRSh shape access".to_string(),
                requires_manual_review: false,
            },
        ]);
    }

    /// Load Keras-specific replacement rules
    fn load_keras_rules(&mut self) {
        self.replacement_rules.extend(vec![
            ReplacementRule {
                pattern: r"from tensorflow import keras".to_string(),
                replacement: "use torsh::nn::*;".to_string(),
                framework: SourceFramework::Keras,
                language: "python".to_string(),
                description: "Replace Keras import with ToRSh nn module".to_string(),
                requires_manual_review: false,
            },
            ReplacementRule {
                pattern: r"keras\.layers\.Dense\((\d+)\)".to_string(),
                replacement: "Linear::new(input_size, $1, true)".to_string(),
                framework: SourceFramework::Keras,
                language: "python".to_string(),
                description: "Replace Keras Dense layer with ToRSh Linear".to_string(),
                requires_manual_review: true,
            },
        ]);
    }

    /// Load Scikit-learn-specific replacement rules  
    fn load_scikit_rules(&mut self) {
        self.replacement_rules.extend(vec![ReplacementRule {
            pattern: r"from sklearn import \*".to_string(),
            replacement: "use torsh::prelude::*;".to_string(),
            framework: SourceFramework::Scikit,
            language: "python".to_string(),
            description: "Replace sklearn import with ToRSh prelude".to_string(),
            requires_manual_review: false,
        }]);
    }

    /// Load Pandas-specific replacement rules
    fn load_pandas_rules(&mut self) {
        self.replacement_rules.extend(vec![ReplacementRule {
            pattern: r"import pandas as pd".to_string(),
            replacement: "use torsh::data::*;".to_string(),
            framework: SourceFramework::Pandas,
            language: "python".to_string(),
            description: "Replace Pandas import with ToRSh data module".to_string(),
            requires_manual_review: false,
        }]);
    }

    /// Load ONNX-specific replacement rules
    fn load_onnx_rules(&mut self) {
        self.replacement_rules.extend(vec![ReplacementRule {
            pattern: r"import onnx".to_string(),
            replacement: "use torsh::prelude::*;".to_string(),
            framework: SourceFramework::ONNX,
            language: "python".to_string(),
            description: "Replace ONNX import with ToRSh prelude".to_string(),
            requires_manual_review: false,
        }]);
    }

    /// Load generic replacement rules
    fn load_generic_rules(&mut self) {
        // Basic patterns that apply to multiple frameworks
        self.replacement_rules.extend(vec![ReplacementRule {
            pattern: r"\.detach\(\)".to_string(),
            replacement: ".detach()?".to_string(),
            framework: SourceFramework::Other("generic".to_string()),
            language: "python".to_string(),
            description: "Add error handling to detach operation".to_string(),
            requires_manual_review: false,
        }]);
    }

    /// Migrate a file from source framework to ToRSh
    pub fn migrate_file<P: AsRef<Path>>(
        &self,
        source_path: P,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(&source_path)?;
        let migrated_content = self.migrate_content(&content);
        Ok(migrated_content)
    }

    /// Migrate content string from source framework to ToRSh
    pub fn migrate_content(&self, content: &str) -> String {
        let mut migrated = content.to_string();

        // Apply replacement rules
        for rule in &self.replacement_rules {
            if rule.language == self.config.target_language || rule.language == "python" {
                // Use simple string replacement for now
                // In a full implementation, would use regex with proper parsing
                migrated = migrated.replace(&rule.pattern, &rule.replacement);
            }
        }

        // Apply function mappings
        for (source_func, target_func) in &self.function_mappings {
            migrated = migrated.replace(source_func, target_func);
        }

        // Apply type mappings
        for (source_type, target_type) in &self.type_mappings {
            migrated = migrated.replace(source_type, target_type);
        }

        // Add ToRSh-specific imports if needed
        if !migrated.contains("use torsh") && !migrated.contains("import torsh") {
            migrated = format!("use torsh::prelude::*;\n\n{}", migrated);
        }

        migrated
    }

    /// Migrate an entire directory
    pub fn migrate_directory<P: AsRef<Path>>(
        &self,
        source_dir: P,
        target_dir: P,
    ) -> Result<MigrationResult, Box<dyn std::error::Error>> {
        let source_dir = source_dir.as_ref();
        let target_dir = target_dir.as_ref();

        let mut result = MigrationResult {
            source_files: Vec::new(),
            target_files: Vec::new(),
            converted_functions: 0,
            converted_classes: 0,
            warnings: Vec::new(),
            errors: Vec::new(),
            manual_changes_required: Vec::new(),
            success_rate: 0.0,
        };

        // Create target directory if it doesn't exist
        std::fs::create_dir_all(target_dir)?;

        // Walk through source directory
        for entry in walkdir::WalkDir::new(source_dir) {
            let entry = entry?;
            if entry.file_type().is_file() {
                let source_path = entry.path();

                // Only process relevant file extensions
                if let Some(ext) = source_path.extension() {
                    if matches!(
                        ext.to_str(),
                        Some("py") | Some("pyi") | Some("ipynb") | Some("js") | Some("ts")
                    ) {
                        let relative_path = source_path.strip_prefix(source_dir)?;
                        let target_path = target_dir.join(relative_path);

                        // Create parent directories
                        if let Some(parent) = target_path.parent() {
                            std::fs::create_dir_all(parent)?;
                        }

                        match self.migrate_file(source_path) {
                            Ok(migrated_content) => {
                                std::fs::write(&target_path, migrated_content)?;
                                result.source_files.push(source_path.to_path_buf());
                                result.target_files.push(target_path);
                            }
                            Err(e) => {
                                result.errors.push(format!(
                                    "Failed to migrate {}: {}",
                                    source_path.display(),
                                    e
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Calculate success rate
        let total_attempts = result.source_files.len() + result.errors.len();
        result.success_rate = if total_attempts > 0 {
            result.source_files.len() as f64 / total_attempts as f64
        } else {
            1.0
        };

        // Add migration warnings for manual review
        for rule in &self.replacement_rules {
            if rule.requires_manual_review {
                result.manual_changes_required.push(format!(
                    "Review usage of pattern '{}': {}",
                    rule.pattern, rule.description
                ));
            }
        }

        Ok(result)
    }

    /// Generate migration report
    pub fn generate_migration_report(&self, result: &MigrationResult) -> String {
        let mut report = String::new();

        report.push_str("# ToRSh Migration Report\n\n");
        report.push_str(&format!(
            "**Source Framework:** {:?}\n",
            self.config.source_framework
        ));
        report.push_str(&format!(
            "**Target Language:** {}\n",
            self.config.target_language
        ));
        report.push_str(&format!(
            "**Success Rate:** {:.1}%\n\n",
            result.success_rate * 100.0
        ));

        report.push_str("## Summary\n\n");
        report.push_str(&format!(
            "- **Files Migrated:** {}\n",
            result.source_files.len()
        ));
        report.push_str(&format!(
            "- **Functions Converted:** {}\n",
            result.converted_functions
        ));
        report.push_str(&format!(
            "- **Classes Converted:** {}\n",
            result.converted_classes
        ));
        report.push_str(&format!("- **Warnings:** {}\n", result.warnings.len()));
        report.push_str(&format!("- **Errors:** {}\n", result.errors.len()));

        if !result.manual_changes_required.is_empty() {
            report.push_str("\n## Manual Changes Required\n\n");
            for change in &result.manual_changes_required {
                report.push_str(&format!("- {}\n", change));
            }
        }

        if !result.warnings.is_empty() {
            report.push_str("\n## Warnings\n\n");
            for warning in &result.warnings {
                report.push_str(&format!("- {}\n", warning));
            }
        }

        if !result.errors.is_empty() {
            report.push_str("\n## Errors\n\n");
            for error in &result.errors {
                report.push_str(&format!("- {}\n", error));
            }
        }

        report.push_str("\n## Next Steps\n\n");
        report.push_str("1. Review all migrated files for correctness\n");
        report.push_str("2. Address manual changes required\n");
        report.push_str("3. Run tests to verify functionality\n");
        report.push_str("4. Optimize performance with ToRSh-specific features\n");

        report
    }

    /// Generate migration guide for a specific framework
    pub fn generate_migration_guide(&self) -> String {
        let mut guide = String::new();

        guide.push_str(&format!(
            "# Migrating from {:?} to ToRSh\n\n",
            self.config.source_framework
        ));

        match self.config.source_framework {
            SourceFramework::PyTorch => self.generate_pytorch_guide(&mut guide),
            SourceFramework::TensorFlow => self.generate_tensorflow_guide(&mut guide),
            SourceFramework::JAX => self.generate_jax_guide(&mut guide),
            SourceFramework::NumPy => self.generate_numpy_guide(&mut guide),
            _ => self.generate_generic_guide(&mut guide),
        }

        guide
    }

    fn generate_pytorch_guide(&self, guide: &mut String) {
        guide.push_str("## Key Differences\n\n");
        guide.push_str("- ToRSh uses explicit error handling with `Result` types\n");
        guide.push_str("- Device transfers use `.to_device()` method\n");
        guide.push_str("- Modules implement the `Module` trait\n");
        guide.push_str("- Autograd is built-in and automatic\n\n");

        guide.push_str("## Common Patterns\n\n");
        guide.push_str("### Tensor Creation\n");
        guide.push_str("```python\n");
        guide.push_str("# PyTorch\n");
        guide.push_str("x = torch.zeros(3, 4)\n");
        guide.push_str("y = torch.randn(3, 4)\n\n");
        guide.push_str("# ToRSh\n");
        guide.push_str("let x = zeros(&[3, 4])?;\n");
        guide.push_str("let y = randn(&[3, 4])?;\n");
        guide.push_str("```\n\n");

        guide.push_str("### Neural Network Modules\n");
        guide.push_str("```python\n");
        guide.push_str("# PyTorch\n");
        guide.push_str("class Net(nn.Module):\n");
        guide.push_str("    def __init__(self):\n");
        guide.push_str("        super().__init__()\n");
        guide.push_str("        self.linear = nn.Linear(10, 1)\n\n");
        guide.push_str("# ToRSh\n");
        guide.push_str("struct Net {\n");
        guide.push_str("    linear: Linear,\n");
        guide.push_str("}\n");
        guide.push_str("impl Module for Net { /* ... */ }\n");
        guide.push_str("```\n\n");
    }

    fn generate_tensorflow_guide(&self, guide: &mut String) {
        guide.push_str("## Key Differences\n\n");
        guide.push_str("- No explicit compilation step needed\n");
        guide.push_str("- Training loops are explicit rather than using `.fit()`\n");
        guide.push_str("- Eager execution is the default\n");
        guide.push_str("- Type safety with Rust's type system\n\n");
    }

    fn generate_jax_guide(&self, guide: &mut String) {
        guide.push_str("## Key Differences\n\n");
        guide.push_str("- JIT compilation is automatic\n");
        guide.push_str("- Functional programming patterns are preserved\n");
        guide.push_str("- Automatic differentiation is built-in\n");
        guide.push_str("- PRNG handling is different\n\n");
    }

    fn generate_numpy_guide(&self, guide: &mut String) {
        guide.push_str("## Key Differences\n\n");
        guide.push_str("- GPU acceleration is automatic when available\n");
        guide.push_str("- Type safety prevents common runtime errors\n");
        guide.push_str("- Broadcasting rules are similar but more explicit\n");
        guide.push_str("- Memory management is automatic\n\n");
    }

    fn generate_generic_guide(&self, guide: &mut String) {
        guide.push_str("## General Migration Principles\n\n");
        guide.push_str("- Use explicit error handling with `Result` types\n");
        guide.push_str("- Leverage Rust's type safety features\n");
        guide.push_str("- Take advantage of ToRSh's performance optimizations\n");
        guide.push_str("- Use the prelude for common imports\n\n");
    }
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            source_framework: SourceFramework::PyTorch,
            target_language: "rust".to_string(),
            preserve_comments: true,
            generate_tests: true,
            optimization_level: OptimizationLevel::Basic,
            include_documentation: true,
            batch_size: 100,
            parallel_processing: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_migration() {
        let config = MigrationConfig {
            source_framework: SourceFramework::PyTorch,
            target_language: "rust".to_string(),
            ..Default::default()
        };

        let tool = MigrationTool::new(config);
        let source_code = "import torch\nx = torch.zeros(3, 4)\ny = torch.randn(2, 3)";
        let migrated = tool.migrate_content(source_code);

        assert!(migrated.contains("use torsh::prelude::*"));
        assert!(migrated.contains("zeros"));
        assert!(migrated.contains("randn"));
    }

    #[test]
    fn test_tensorflow_migration() {
        let config = MigrationConfig {
            source_framework: SourceFramework::TensorFlow,
            target_language: "rust".to_string(),
            ..Default::default()
        };

        let tool = MigrationTool::new(config);
        let source_code = "import tensorflow as tf\nx = tf.zeros([3, 4])";
        let migrated = tool.migrate_content(source_code);

        assert!(migrated.contains("use torsh::prelude::*"));
        assert!(migrated.contains("zeros"));
    }

    #[test]
    fn test_numpy_migration() {
        let config = MigrationConfig {
            source_framework: SourceFramework::NumPy,
            target_language: "rust".to_string(),
            ..Default::default()
        };

        let tool = MigrationTool::new(config);
        let source_code = "import numpy as np\nx = np.zeros((3, 4))\ny = x.shape";
        let migrated = tool.migrate_content(source_code);

        assert!(migrated.contains("use torsh::prelude::*"));
        assert!(migrated.contains("zeros"));
        assert!(migrated.contains(".shape().dims()"));
    }

    #[test]
    fn test_migration_guide_generation() {
        let config = MigrationConfig {
            source_framework: SourceFramework::PyTorch,
            ..Default::default()
        };

        let tool = MigrationTool::new(config);
        let guide = tool.generate_migration_guide();

        assert!(guide.contains("Migrating from PyTorch to ToRSh"));
        assert!(guide.contains("Key Differences"));
        assert!(guide.contains("Common Patterns"));
    }
}
