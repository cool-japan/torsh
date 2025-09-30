//! Built-in datasets powered by SciRS2
//!
//! This module provides access to toy datasets, synthetic data generators,
//! and other built-in data sources from the SciRS2 ecosystem.

use crate::error::DataError;
// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{Random, Rng, SeedableRng};
use torsh_core::error::TorshError;
use torsh_tensor::Tensor;
// Direct SciRS2 datasets integration
// use scirs2_datasets::{load_iris, load_boston, make_classification}; // Will be uncommented when API stabilizes

/// Built-in dataset types
#[derive(Debug, Clone)]
pub enum BuiltinDataset {
    Iris,
    Boston,
    Diabetes,
    Wine,
    BreastCancer,
    Digits,
}

/// Synthetic data generation configuration
#[derive(Debug, Clone)]
pub struct SyntheticDataConfig {
    /// Number of samples to generate
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of classes (for classification)
    pub n_classes: Option<usize>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Whether to add noise
    pub noise: Option<f64>,
    /// Feature scaling method
    pub scale: Option<ScalingMethod>,
}

/// Feature scaling methods
#[derive(Debug, Clone)]
pub enum ScalingMethod {
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
}

/// Regression data generation parameters
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    pub n_samples: usize,
    pub n_features: usize,
    pub n_informative: Option<usize>,
    pub noise: Option<f64>,
    pub bias: Option<f64>,
    pub random_state: Option<u64>,
}

/// Classification data generation parameters
#[derive(Debug, Clone)]
pub struct ClassificationConfig {
    pub n_samples: usize,
    pub n_features: usize,
    pub n_classes: usize,
    pub n_informative: Option<usize>,
    pub n_redundant: Option<usize>,
    pub n_clusters_per_class: Option<usize>,
    pub class_sep: Option<f64>,
    pub random_state: Option<u64>,
}

/// Clustering data generation parameters
#[derive(Debug, Clone)]
pub struct ClusteringConfig {
    pub n_samples: usize,
    pub centers: usize,
    pub n_features: Option<usize>,
    pub cluster_std: Option<f64>,
    pub center_box: Option<(f64, f64)>,
    pub random_state: Option<u64>,
}

/// Dataset result containing features and targets
#[derive(Debug, Clone)]
pub struct DatasetResult {
    pub features: Tensor,
    pub targets: Tensor,
    pub feature_names: Option<Vec<String>>,
    pub target_names: Option<Vec<String>>,
    pub description: String,
}

impl Default for SyntheticDataConfig {
    fn default() -> Self {
        Self {
            n_samples: 100,
            n_features: 2,
            n_classes: Some(2),
            seed: None,
            noise: Some(0.1),
            scale: Some(ScalingMethod::StandardScaler),
        }
    }
}

/// Load a built-in dataset
pub fn load_builtin_dataset(dataset: BuiltinDataset) -> Result<DatasetResult, DataError> {
    match dataset {
        BuiltinDataset::Iris => load_iris_dataset(),
        BuiltinDataset::Boston => load_boston_dataset(),
        BuiltinDataset::Diabetes => load_diabetes_dataset(),
        BuiltinDataset::Wine => load_wine_dataset(),
        BuiltinDataset::BreastCancer => load_breast_cancer_dataset(),
        BuiltinDataset::Digits => load_digits_dataset(),
    }
}

/// Generate synthetic regression data
pub fn make_regression(config: RegressionConfig) -> Result<DatasetResult, DataError> {
    // TODO: Use scirs2_datasets::make_regression when API is available
    // For now, implement basic synthetic data generation

    let n_informative = config.n_informative.unwrap_or(config.n_features);
    let noise_std = config.noise.unwrap_or(0.0);

    // Generate random features
    let features_data: Vec<f32> = (0..config.n_samples * config.n_features)
        .map(|_| {
            // ✅ SciRS2 Policy Compliant
            let mut rng = scirs2_core::random::thread_rng();
            rng.gen_range(-1.0..1.0)
        })
        .collect();

    let features = Tensor::from_vec(features_data, &[config.n_samples, config.n_features])?;

    // Generate targets as linear combination of informative features
    let targets_data: Vec<f32> = (0..config.n_samples)
        .map(|i| {
            let mut target = 0.0;
            for j in 0..n_informative {
                if let Ok(feature_vec) = features.to_vec() {
                    let idx = i * config.n_features + j;
                    if let Some(&feature_val) = feature_vec.get(idx) {
                        target += feature_val;
                    }
                }
            }

            // Add noise
            if noise_std > 0.0 {
                // ✅ SciRS2 Policy Compliant
                let mut rng = scirs2_core::random::thread_rng();
                target += rng.gen_range(-noise_std as f32..noise_std as f32);
            }

            target
        })
        .collect();

    let targets = Tensor::from_vec(targets_data, &[config.n_samples])?;

    Ok(DatasetResult {
        features,
        targets,
        feature_names: Some(
            (0..config.n_features)
                .map(|i| format!("feature_{}", i))
                .collect(),
        ),
        target_names: Some(vec!["target".to_string()]),
        description: "Synthetic regression dataset".to_string(),
    })
}

/// Generate synthetic classification data
pub fn make_classification(config: ClassificationConfig) -> Result<DatasetResult, DataError> {
    // TODO: Use scirs2_datasets::make_classification when API is available
    // For now, implement basic synthetic data generation

    // ✅ SciRS2 Policy Compliant - Using scirs2_core::random
    let mut rng = if let Some(seed) = config.random_state {
        scirs2_core::random::StdRng::seed_from_u64(seed)
    } else {
        {
            let mut thread_rng = scirs2_core::random::thread_rng();
            scirs2_core::random::StdRng::from_rng(&mut thread_rng)
        }
    };

    // Generate random features
    let features_data: Vec<f32> = (0..config.n_samples * config.n_features)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let features = Tensor::from_vec(features_data, &[config.n_samples, config.n_features])?;

    // Generate targets
    let targets_data: Vec<f32> = (0..config.n_samples)
        .map(|_| rng.gen_range(0..config.n_classes) as f32)
        .collect();

    let targets = Tensor::from_vec(targets_data, &[config.n_samples])?;

    Ok(DatasetResult {
        features,
        targets,
        feature_names: Some(
            (0..config.n_features)
                .map(|i| format!("feature_{}", i))
                .collect(),
        ),
        target_names: Some(
            (0..config.n_classes)
                .map(|i| format!("class_{}", i))
                .collect(),
        ),
        description: "Synthetic classification dataset".to_string(),
    })
}

/// Generate synthetic clustering data (blobs)
pub fn make_blobs(config: ClusteringConfig) -> Result<DatasetResult, DataError> {
    // TODO: Use scirs2_datasets::make_blobs when API is available

    // ✅ SciRS2 Policy Compliant - Using scirs2_core::random
    let mut rng = if let Some(seed) = config.random_state {
        scirs2_core::random::StdRng::seed_from_u64(seed)
    } else {
        {
            let mut thread_rng = scirs2_core::random::thread_rng();
            scirs2_core::random::StdRng::from_rng(&mut thread_rng)
        }
    };

    let n_features = config.n_features.unwrap_or(2);
    let cluster_std = config.cluster_std.unwrap_or(1.0);

    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..config.centers)
        .map(|_| (0..n_features).map(|_| rng.gen_range(-5.0..5.0)).collect())
        .collect();

    let samples_per_cluster = config.n_samples / config.centers;
    let mut features_data = Vec::new();
    let mut targets_data = Vec::new();

    for (cluster_id, center) in centers.iter().enumerate() {
        for _ in 0..samples_per_cluster {
            // Generate point around cluster center
            for &center_coord in center {
                let noise: f32 = rng.gen_range(-cluster_std as f32..cluster_std as f32);
                features_data.push(center_coord + noise);
            }
            targets_data.push(cluster_id as f32);
        }
    }

    let features = Tensor::from_vec(
        features_data,
        &[samples_per_cluster * config.centers, n_features],
    )?;

    let targets = Tensor::from_vec(targets_data, &[samples_per_cluster * config.centers])?;

    Ok(DatasetResult {
        features,
        targets,
        feature_names: Some((0..n_features).map(|i| format!("feature_{}", i)).collect()),
        target_names: Some(
            (0..config.centers)
                .map(|i| format!("cluster_{}", i))
                .collect(),
        ),
        description: "Synthetic clustering dataset (blobs)".to_string(),
    })
}

// Built-in dataset implementations (placeholders for now)
fn load_iris_dataset() -> Result<DatasetResult, DataError> {
    // TODO: Use scirs2_datasets::load_iris() when available
    // For now, create a minimal iris-like dataset
    make_classification(ClassificationConfig {
        n_samples: 150,
        n_features: 4,
        n_classes: 3,
        n_informative: Some(4),
        random_state: Some(42),
        ..Default::default()
    })
}

fn load_boston_dataset() -> Result<DatasetResult, DataError> {
    // TODO: Use scirs2_datasets::load_boston() when available
    make_regression(RegressionConfig {
        n_samples: 506,
        n_features: 13,
        n_informative: Some(13),
        noise: Some(0.1),
        random_state: Some(42),
        bias: Some(0.0),
    })
}

fn load_diabetes_dataset() -> Result<DatasetResult, DataError> {
    make_regression(RegressionConfig {
        n_samples: 442,
        n_features: 10,
        n_informative: Some(10),
        noise: Some(0.1),
        random_state: Some(42),
        bias: Some(0.0),
    })
}

fn load_wine_dataset() -> Result<DatasetResult, DataError> {
    make_classification(ClassificationConfig {
        n_samples: 178,
        n_features: 13,
        n_classes: 3,
        n_informative: Some(13),
        random_state: Some(42),
        ..Default::default()
    })
}

fn load_breast_cancer_dataset() -> Result<DatasetResult, DataError> {
    make_classification(ClassificationConfig {
        n_samples: 569,
        n_features: 30,
        n_classes: 2,
        n_informative: Some(30),
        random_state: Some(42),
        ..Default::default()
    })
}

fn load_digits_dataset() -> Result<DatasetResult, DataError> {
    make_classification(ClassificationConfig {
        n_samples: 1797,
        n_features: 64,
        n_classes: 10,
        n_informative: Some(64),
        random_state: Some(42),
        ..Default::default()
    })
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            n_samples: 100,
            n_features: 1,
            n_informative: None,
            noise: Some(0.1),
            bias: Some(0.0),
            random_state: None,
        }
    }
}

impl Default for ClassificationConfig {
    fn default() -> Self {
        Self {
            n_samples: 100,
            n_features: 2,
            n_classes: 2,
            n_informative: None,
            n_redundant: None,
            n_clusters_per_class: None,
            class_sep: Some(1.0),
            random_state: None,
        }
    }
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            n_samples: 100,
            centers: 3,
            n_features: Some(2),
            cluster_std: Some(1.0),
            center_box: Some((-10.0, 10.0)),
            random_state: None,
        }
    }
}

/// Dataset registry for managing available datasets
#[derive(Debug, Default)]
pub struct DatasetRegistry {
    builtin_datasets: Vec<BuiltinDataset>,
}

impl DatasetRegistry {
    /// Create a new dataset registry
    pub fn new() -> Self {
        Self {
            builtin_datasets: vec![
                BuiltinDataset::Iris,
                BuiltinDataset::Boston,
                BuiltinDataset::Diabetes,
                BuiltinDataset::Wine,
                BuiltinDataset::BreastCancer,
                BuiltinDataset::Digits,
            ],
        }
    }

    /// List all available built-in datasets
    pub fn list_builtin(&self) -> &[BuiltinDataset] {
        &self.builtin_datasets
    }

    /// Load a dataset by name
    pub fn load_by_name(&self, name: &str) -> Result<DatasetResult, DataError> {
        let dataset = match name.to_lowercase().as_str() {
            "iris" => BuiltinDataset::Iris,
            "boston" => BuiltinDataset::Boston,
            "diabetes" => BuiltinDataset::Diabetes,
            "wine" => BuiltinDataset::Wine,
            "breast_cancer" | "breastcancer" => BuiltinDataset::BreastCancer,
            "digits" => BuiltinDataset::Digits,
            _ => {
                return Err(DataError::dataset(
                    crate::error::DatasetErrorKind::UnsupportedFormat,
                    format!("Unknown dataset: {}", name),
                ))
            }
        };

        load_builtin_dataset(dataset)
    }
}
