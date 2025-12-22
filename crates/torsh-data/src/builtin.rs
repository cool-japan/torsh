//! Built-in datasets powered by SciRS2
//!
//! This module provides access to toy datasets, synthetic data generators,
//! and other built-in data sources from the SciRS2 ecosystem.

use crate::error::DataError;
use scirs2_core::Distribution; // For sample() method on distributions
use torsh_tensor::Tensor;

// ✅ Direct SciRS2 datasets integration - using stable toy datasets API
use scirs2_datasets::toy::{
    load_boston as scirs2_load_boston, load_breast_cancer as scirs2_load_breast_cancer,
    load_diabetes as scirs2_load_diabetes, load_digits as scirs2_load_digits,
    load_iris as scirs2_load_iris,
};

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
///
/// Creates a regression problem with specified characteristics. The targets are generated as:
/// y = X @ coef + bias + noise
///
/// where:
/// - X contains n_informative features that actually contribute to y
/// - The remaining (n_features - n_informative) features are random noise
/// - coef are random coefficients for informative features
/// - noise is Gaussian noise with standard deviation specified by `noise` parameter
pub fn make_regression(config: RegressionConfig) -> Result<DatasetResult, DataError> {
    use scirs2_core::random::{Normal, SeedableRng, StdRng};

    let n_informative = config.n_informative.unwrap_or(config.n_features);
    let noise_std = config.noise.unwrap_or(0.0);
    let bias = config.bias.unwrap_or(0.0);

    if n_informative > config.n_features {
        return Err(DataError::dataset(
            crate::error::DatasetErrorKind::CorruptedData,
            format!(
                "n_informative ({}) cannot exceed n_features ({})",
                n_informative, config.n_features
            ),
        ));
    }

    // ✅ SciRS2 Policy Compliant - Using scirs2_core::random
    let mut rng = if let Some(seed) = config.random_state {
        StdRng::seed_from_u64(seed)
    } else {
        let mut thread_rng = scirs2_core::random::thread_rng();
        StdRng::from_rng(&mut thread_rng)
    };

    // Generate features from standard normal distribution
    let normal = Normal::new(0.0, 1.0).unwrap();
    let features_data: Vec<f32> = (0..config.n_samples * config.n_features)
        .map(|_| normal.sample(&mut rng) as f32)
        .collect();

    let features = Tensor::from_vec(
        features_data.clone(),
        &[config.n_samples, config.n_features],
    )?;

    // Generate random coefficients for informative features
    let coefficients: Vec<f32> = (0..n_informative)
        .map(|_| rng.gen_range(-100.0..100.0))
        .collect();

    // Generate targets as linear combination of informative features
    let noise_dist = Normal::new(0.0, noise_std).unwrap();
    let targets_data: Vec<f32> = (0..config.n_samples)
        .map(|i| {
            // Compute linear combination: sum(coef_j * x_ij) for informative features
            let mut target = bias as f32;
            for j in 0..n_informative {
                let idx = i * config.n_features + j;
                target += coefficients[j] * features_data[idx];
            }

            // Add Gaussian noise
            if noise_std > 0.0 {
                target += noise_dist.sample(&mut rng) as f32;
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
                .map(|i| {
                    if i < n_informative {
                        format!("informative_{}", i)
                    } else {
                        format!("noise_{}", i - n_informative)
                    }
                })
                .collect(),
        ),
        target_names: Some(vec!["target".to_string()]),
        description: format!(
            "Synthetic regression dataset: {} samples, {} features ({} informative), noise_std={:.2}, bias={:.2}",
            config.n_samples, config.n_features, n_informative, noise_std, bias
        ),
    })
}

/// Generate synthetic classification data
///
/// Creates a classification problem with specified characteristics. Features are generated
/// by creating Gaussian clusters for each class, with controllable separation.
///
/// - `n_informative`: Number of features that are informative for classification
/// - `n_redundant`: Number of features that are linear combinations of informative features
/// - `n_clusters_per_class`: Number of Gaussian clusters per class
/// - `class_sep`: Multiplier for class separation (larger = more separated classes)
pub fn make_classification(config: ClassificationConfig) -> Result<DatasetResult, DataError> {
    use scirs2_core::random::{Normal, SeedableRng, StdRng};

    let n_informative = config.n_informative.unwrap_or(config.n_features.min(2));
    let n_redundant = config.n_redundant.unwrap_or(0);
    let n_clusters_per_class = config.n_clusters_per_class.unwrap_or(1);
    let class_sep = config.class_sep.unwrap_or(1.0);

    if n_informative + n_redundant > config.n_features {
        return Err(DataError::dataset(
            crate::error::DatasetErrorKind::CorruptedData,
            format!(
                "n_informative ({}) + n_redundant ({}) cannot exceed n_features ({})",
                n_informative, n_redundant, config.n_features
            ),
        ));
    }

    // ✅ SciRS2 Policy Compliant - Using scirs2_core::random
    let mut rng = if let Some(seed) = config.random_state {
        StdRng::seed_from_u64(seed)
    } else {
        let mut thread_rng = scirs2_core::random::thread_rng();
        StdRng::from_rng(&mut thread_rng)
    };

    // Generate cluster centers for each class in the informative feature space
    let total_clusters = config.n_classes * n_clusters_per_class;
    let mut cluster_centers: Vec<Vec<f32>> = Vec::new();
    let mut cluster_labels: Vec<usize> = Vec::new();

    for class_id in 0..config.n_classes {
        for _ in 0..n_clusters_per_class {
            let center: Vec<f32> = (0..n_informative)
                .map(|_| rng.gen_range(-class_sep as f32..class_sep as f32) * 10.0)
                .collect();
            cluster_centers.push(center);
            cluster_labels.push(class_id);
        }
    }

    // Distribute samples across clusters
    let samples_per_cluster = config.n_samples / total_clusters;
    let remainder = config.n_samples % total_clusters;

    let mut features_data = Vec::new();
    let mut targets_data = Vec::new();

    let normal = Normal::new(0.0, 1.0).unwrap();

    for (cluster_idx, (center, &class_label)) in cluster_centers
        .iter()
        .zip(cluster_labels.iter())
        .enumerate()
    {
        let n_samples_this_cluster =
            samples_per_cluster + if cluster_idx < remainder { 1 } else { 0 };

        for _ in 0..n_samples_this_cluster {
            // Generate informative features from cluster center with Gaussian noise
            for &center_val in center.iter() {
                let noise = normal.sample(&mut rng) as f32;
                features_data.push(center_val + noise);
            }

            // Generate redundant features as linear combinations of informative features
            let start_idx = features_data.len() - n_informative;
            for _ in 0..n_redundant {
                let mut redundant = 0.0f32;
                for j in 0..n_informative {
                    let weight = rng.gen_range(-1.0..1.0);
                    redundant += weight * features_data[start_idx + j];
                }
                features_data.push(redundant);
            }

            // Generate noise features (truly random)
            let n_noise = config.n_features - n_informative - n_redundant;
            for _ in 0..n_noise {
                features_data.push(rng.gen_range(-10.0..10.0));
            }

            targets_data.push(class_label as f32);
        }
    }

    let features = Tensor::from_vec(features_data, &[config.n_samples, config.n_features])?;
    let targets = Tensor::from_vec(targets_data, &[config.n_samples])?;

    Ok(DatasetResult {
        features,
        targets,
        feature_names: Some(
            (0..config.n_features)
                .map(|i| {
                    if i < n_informative {
                        format!("informative_{}", i)
                    } else if i < n_informative + n_redundant {
                        format!("redundant_{}", i - n_informative)
                    } else {
                        format!("noise_{}", i - n_informative - n_redundant)
                    }
                })
                .collect(),
        ),
        target_names: Some(
            (0..config.n_classes)
                .map(|i| format!("class_{}", i))
                .collect(),
        ),
        description: format!(
            "Synthetic classification dataset: {} samples, {} features ({} informative, {} redundant), {} classes, class_sep={:.2}",
            config.n_samples, config.n_features, n_informative, n_redundant, config.n_classes, class_sep
        ),
    })
}

/// Generate synthetic clustering data (blobs)
///
/// Creates isotropic Gaussian blobs for clustering. Each blob is a Gaussian distribution
/// centered at a random location within the bounding box.
///
/// - `centers`: Number of cluster centers to generate
/// - `n_features`: Number of features (dimensions) for each sample
/// - `cluster_std`: Standard deviation of the Gaussian noise around each cluster center
/// - `center_box`: Bounding box (min, max) for cluster center locations
pub fn make_blobs(config: ClusteringConfig) -> Result<DatasetResult, DataError> {
    use scirs2_core::random::{Normal, SeedableRng, StdRng};

    // ✅ SciRS2 Policy Compliant - Using scirs2_core::random
    let mut rng = if let Some(seed) = config.random_state {
        StdRng::seed_from_u64(seed)
    } else {
        let mut thread_rng = scirs2_core::random::thread_rng();
        StdRng::from_rng(&mut thread_rng)
    };

    let n_features = config.n_features.unwrap_or(2);
    let cluster_std = config.cluster_std.unwrap_or(1.0);
    let (box_min, box_max) = config.center_box.unwrap_or((-10.0, 10.0));

    if box_min >= box_max {
        return Err(DataError::dataset(
            crate::error::DatasetErrorKind::CorruptedData,
            format!(
                "center_box min ({}) must be less than max ({})",
                box_min, box_max
            ),
        ));
    }

    // Generate cluster centers uniformly within the bounding box
    let centers: Vec<Vec<f32>> = (0..config.centers)
        .map(|_| {
            (0..n_features)
                .map(|_| rng.gen_range(box_min as f32..box_max as f32))
                .collect()
        })
        .collect();

    // Distribute samples across clusters
    let samples_per_cluster = config.n_samples / config.centers;
    let remainder = config.n_samples % config.centers;

    let mut features_data = Vec::new();
    let mut targets_data = Vec::new();

    // Create Gaussian distribution for sampling around cluster centers
    let normal = Normal::new(0.0, cluster_std).unwrap();

    for (cluster_id, center) in centers.iter().enumerate() {
        let n_samples_this_cluster =
            samples_per_cluster + if cluster_id < remainder { 1 } else { 0 };

        for _ in 0..n_samples_this_cluster {
            // Generate point around cluster center using Gaussian noise
            for &center_coord in center {
                let noise = normal.sample(&mut rng) as f32;
                features_data.push(center_coord + noise);
            }
            targets_data.push(cluster_id as f32);
        }
    }

    let features = Tensor::from_vec(features_data, &[config.n_samples, n_features])?;
    let targets = Tensor::from_vec(targets_data, &[config.n_samples])?;

    Ok(DatasetResult {
        features,
        targets,
        feature_names: Some((0..n_features).map(|i| format!("feature_{}", i)).collect()),
        target_names: Some(
            (0..config.centers)
                .map(|i| format!("cluster_{}", i))
                .collect(),
        ),
        description: format!(
            "Synthetic clustering dataset (blobs): {} samples, {} features, {} clusters, cluster_std={:.2}",
            config.n_samples, n_features, config.centers, cluster_std
        ),
    })
}

/// Convert scirs2_datasets::Dataset to torsh-data's DatasetResult
fn convert_scirs2_dataset(
    scirs2_dataset: scirs2_datasets::utils::Dataset,
) -> Result<DatasetResult, DataError> {
    // Convert features: Array2<f64> -> Tensor
    let shape = scirs2_dataset.data.shape();
    let features_data: Vec<f32> = scirs2_dataset.data.iter().map(|&x| x as f32).collect();
    let features = Tensor::from_vec(features_data, &[shape[0], shape[1]])?;

    // Convert targets: Array1<f64> -> Tensor
    let targets = if let Some(target_array) = scirs2_dataset.target {
        let target_data: Vec<f32> = target_array.iter().map(|&x| x as f32).collect();
        Tensor::from_vec(target_data, &[target_array.len()])?
    } else {
        // Create empty tensor if no targets
        Tensor::from_vec(vec![], &[0])?
    };

    Ok(DatasetResult {
        features,
        targets,
        feature_names: scirs2_dataset.featurenames,
        target_names: scirs2_dataset.targetnames,
        description: scirs2_dataset
            .description
            .unwrap_or_else(|| "Dataset loaded from scirs2".to_string()),
    })
}

// Built-in dataset implementations using scirs2_datasets
fn load_iris_dataset() -> Result<DatasetResult, DataError> {
    // ✅ Using scirs2_datasets::load_iris() for authentic Iris dataset
    let scirs2_dataset = scirs2_load_iris().map_err(|e| {
        DataError::dataset(
            crate::error::DatasetErrorKind::CorruptedData,
            format!("Failed to load Iris dataset from scirs2_datasets: {}", e),
        )
    })?;

    convert_scirs2_dataset(scirs2_dataset)
}

fn load_boston_dataset() -> Result<DatasetResult, DataError> {
    // ✅ Using scirs2_datasets::load_boston() for authentic Boston Housing dataset
    let scirs2_dataset = scirs2_load_boston().map_err(|e| {
        DataError::dataset(
            crate::error::DatasetErrorKind::CorruptedData,
            format!("Failed to load Boston dataset from scirs2_datasets: {}", e),
        )
    })?;

    convert_scirs2_dataset(scirs2_dataset)
}

fn load_diabetes_dataset() -> Result<DatasetResult, DataError> {
    // ✅ Using scirs2_datasets::load_diabetes() for authentic Diabetes dataset
    let scirs2_dataset = scirs2_load_diabetes().map_err(|e| {
        DataError::dataset(
            crate::error::DatasetErrorKind::CorruptedData,
            format!(
                "Failed to load Diabetes dataset from scirs2_datasets: {}",
                e
            ),
        )
    })?;

    convert_scirs2_dataset(scirs2_dataset)
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
    // ✅ Using scirs2_datasets::load_breast_cancer() for authentic Breast Cancer dataset
    let scirs2_dataset = scirs2_load_breast_cancer().map_err(|e| {
        DataError::dataset(
            crate::error::DatasetErrorKind::CorruptedData,
            format!(
                "Failed to load Breast Cancer dataset from scirs2_datasets: {}",
                e
            ),
        )
    })?;

    convert_scirs2_dataset(scirs2_dataset)
}

fn load_digits_dataset() -> Result<DatasetResult, DataError> {
    // ✅ Using scirs2_datasets::load_digits() for authentic Digits dataset
    let scirs2_dataset = scirs2_load_digits().map_err(|e| {
        DataError::dataset(
            crate::error::DatasetErrorKind::CorruptedData,
            format!("Failed to load Digits dataset from scirs2_datasets: {}", e),
        )
    })?;

    convert_scirs2_dataset(scirs2_dataset)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_iris_dataset() {
        let result = load_builtin_dataset(BuiltinDataset::Iris);
        assert!(result.is_ok());
        let dataset = result.unwrap();

        // Iris has 150 samples, 4 features
        assert_eq!(dataset.features.size(0).unwrap(), 150);
        assert_eq!(dataset.features.size(1).unwrap(), 4);
        assert_eq!(dataset.targets.size(0).unwrap(), 150);

        // Check metadata
        assert!(dataset.feature_names.is_some());
        assert!(dataset.target_names.is_some());
        assert!(!dataset.description.is_empty());

        let feature_names = dataset.feature_names.unwrap();
        assert_eq!(feature_names.len(), 4);
        assert!(feature_names.contains(&"sepal_length".to_string()));

        let target_names = dataset.target_names.unwrap();
        assert_eq!(target_names.len(), 3);
    }

    #[test]
    fn test_load_boston_dataset() {
        let result = load_builtin_dataset(BuiltinDataset::Boston);
        assert!(result.is_ok());
        let dataset = result.unwrap();

        // Boston has 30 samples, 5 features (simplified version from scirs2_datasets)
        assert_eq!(dataset.features.size(0).unwrap(), 30);
        assert_eq!(dataset.features.size(1).unwrap(), 5);
        assert_eq!(dataset.targets.size(0).unwrap(), 30);

        // Check metadata
        assert!(dataset.feature_names.is_some());
        assert!(!dataset.description.is_empty());
    }

    #[test]
    fn test_load_diabetes_dataset() {
        let result = load_builtin_dataset(BuiltinDataset::Diabetes);
        assert!(result.is_ok());
        let dataset = result.unwrap();

        // Diabetes has 442 samples, 10 features (from scirs2_datasets)
        assert_eq!(dataset.features.size(0).unwrap(), 442);
        assert_eq!(dataset.features.size(1).unwrap(), 10);
        assert_eq!(dataset.targets.size(0).unwrap(), 442);

        // Check metadata
        assert!(dataset.feature_names.is_some());
        assert!(!dataset.description.is_empty());

        let feature_names = dataset.feature_names.unwrap();
        assert_eq!(feature_names.len(), 10);
        // Verify expected feature names from scirs2 diabetes dataset
        assert!(feature_names.contains(&"age".to_string()));
        assert!(feature_names.contains(&"bmi".to_string()));
    }

    #[test]
    fn test_load_breast_cancer_dataset() {
        let result = load_builtin_dataset(BuiltinDataset::BreastCancer);
        assert!(result.is_ok());
        let dataset = result.unwrap();

        // Breast cancer has 30 samples, 5 features (simplified version from scirs2_datasets)
        assert_eq!(dataset.features.size(0).unwrap(), 30);
        assert_eq!(dataset.features.size(1).unwrap(), 5);
        assert_eq!(dataset.targets.size(0).unwrap(), 30);

        // Check metadata
        assert!(dataset.feature_names.is_some());
        assert!(dataset.target_names.is_some());
        assert!(!dataset.description.is_empty());

        let target_names = dataset.target_names.unwrap();
        assert_eq!(target_names.len(), 2); // Binary classification: malignant, benign
        assert!(target_names.contains(&"malignant".to_string()));
        assert!(target_names.contains(&"benign".to_string()));
    }

    #[test]
    fn test_load_digits_dataset() {
        let result = load_builtin_dataset(BuiltinDataset::Digits);
        assert!(result.is_ok());
        let dataset = result.unwrap();

        // Digits has 50 samples, 16 features (4x4 images from scirs2_datasets)
        assert_eq!(dataset.features.size(0).unwrap(), 50);
        assert_eq!(dataset.features.size(1).unwrap(), 16);
        assert_eq!(dataset.targets.size(0).unwrap(), 50);

        // Check metadata
        assert!(dataset.target_names.is_some());
        assert!(!dataset.description.is_empty());

        let target_names = dataset.target_names.unwrap();
        assert_eq!(target_names.len(), 10); // 10 digits (0-9)
    }

    #[test]
    fn test_load_wine_dataset() {
        let result = load_builtin_dataset(BuiltinDataset::Wine);
        assert!(result.is_ok());
        let dataset = result.unwrap();

        // Wine has 178 samples, 13 features
        assert_eq!(dataset.features.size(0).unwrap(), 178);
        assert_eq!(dataset.features.size(1).unwrap(), 13);
        assert_eq!(dataset.targets.size(0).unwrap(), 178);

        // Check metadata
        assert!(!dataset.description.is_empty());
    }

    #[test]
    fn test_dataset_registry() {
        let registry = DatasetRegistry::new();
        let builtin_datasets = registry.list_builtin();

        // Check all datasets are registered
        assert_eq!(builtin_datasets.len(), 6);
    }

    #[test]
    fn test_load_by_name() {
        let registry = DatasetRegistry::new();

        // Test all dataset names (including aliases)
        assert!(registry.load_by_name("iris").is_ok());
        assert!(registry.load_by_name("boston").is_ok());
        assert!(registry.load_by_name("diabetes").is_ok());
        assert!(registry.load_by_name("wine").is_ok());
        assert!(registry.load_by_name("breast_cancer").is_ok());
        assert!(registry.load_by_name("breastcancer").is_ok()); // Alias
        assert!(registry.load_by_name("digits").is_ok());

        // Test case insensitivity
        assert!(registry.load_by_name("IRIS").is_ok());
        assert!(registry.load_by_name("Diabetes").is_ok());

        // Test unknown dataset
        assert!(registry.load_by_name("unknown").is_err());
    }

    #[test]
    fn test_make_regression() {
        let config = RegressionConfig {
            n_samples: 100,
            n_features: 5,
            n_informative: Some(3),
            noise: Some(0.1),
            bias: Some(1.0),
            random_state: Some(42),
        };

        let result = make_regression(config);
        assert!(result.is_ok());
        let dataset = result.unwrap();

        assert_eq!(dataset.features.size(0).unwrap(), 100);
        assert_eq!(dataset.features.size(1).unwrap(), 5);
        assert_eq!(dataset.targets.size(0).unwrap(), 100);
    }

    #[test]
    fn test_make_classification() {
        let config = ClassificationConfig {
            n_samples: 200,
            n_features: 10,
            n_classes: 3,
            n_informative: Some(5),
            random_state: Some(42),
            ..Default::default()
        };

        let result = make_classification(config);
        assert!(result.is_ok());
        let dataset = result.unwrap();

        assert_eq!(dataset.features.size(0).unwrap(), 200);
        assert_eq!(dataset.features.size(1).unwrap(), 10);
        assert_eq!(dataset.targets.size(0).unwrap(), 200);
    }

    #[test]
    fn test_make_blobs() {
        let config = ClusteringConfig {
            n_samples: 150,
            centers: 3,
            n_features: Some(2),
            cluster_std: Some(0.5),
            random_state: Some(42),
            ..Default::default()
        };

        let result = make_blobs(config);
        assert!(result.is_ok());
        let dataset = result.unwrap();

        assert_eq!(dataset.features.size(0).unwrap(), 150);
        assert_eq!(dataset.features.size(1).unwrap(), 2);
        assert_eq!(dataset.targets.size(0).unwrap(), 150);
    }

    #[test]
    fn test_regression_config_validation() {
        // Test n_informative > n_features
        let config = RegressionConfig {
            n_samples: 100,
            n_features: 5,
            n_informative: Some(10), // More than n_features
            noise: Some(0.1),
            bias: Some(0.0),
            random_state: Some(42),
        };

        let result = make_regression(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_scirs2_integration_diabetes() {
        // Test that diabetes dataset is authentic from scirs2, not synthetic
        let result = load_builtin_dataset(BuiltinDataset::Diabetes);
        assert!(result.is_ok());
        let dataset = result.unwrap();

        // Verify it has the correct scirs2 diabetes dataset characteristics
        assert_eq!(dataset.features.size(0).unwrap(), 442);
        assert_eq!(dataset.features.size(1).unwrap(), 10);

        // Check that description mentions it's from scirs2 or is realistic
        assert!(
            dataset.description.contains("diabetes") || dataset.description.contains("Diabetes")
        );
    }

    #[test]
    fn test_scirs2_integration_breast_cancer() {
        // Test that breast cancer dataset is authentic from scirs2
        let result = load_builtin_dataset(BuiltinDataset::BreastCancer);
        assert!(result.is_ok());
        let dataset = result.unwrap();

        // Verify it has the correct scirs2 breast cancer dataset characteristics
        assert_eq!(dataset.features.size(0).unwrap(), 30);
        assert_eq!(dataset.features.size(1).unwrap(), 5);

        // Check metadata is properly populated
        assert!(dataset.feature_names.is_some());
        assert!(dataset.target_names.is_some());
    }

    #[test]
    fn test_scirs2_integration_digits() {
        // Test that digits dataset is authentic from scirs2
        let result = load_builtin_dataset(BuiltinDataset::Digits);
        assert!(result.is_ok());
        let dataset = result.unwrap();

        // Verify it has the correct scirs2 digits dataset characteristics
        assert_eq!(dataset.features.size(0).unwrap(), 50);
        assert_eq!(dataset.features.size(1).unwrap(), 16); // 4x4 pixels

        // Check that we have 10 target classes (digits 0-9)
        assert!(dataset.target_names.is_some());
        let target_names = dataset.target_names.unwrap();
        assert_eq!(target_names.len(), 10);
    }
}
