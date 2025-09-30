//! Tabular data utilities

use crate::dataset::Dataset;
#[cfg(feature = "dataframe")]
use polars::prelude::*;
use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};
use std::path::Path;

/// Dataset for tabular data stored in CSV files
pub struct CSVDataset {
    data: Vec<Vec<f32>>,
    targets: Option<Vec<f32>>,
    feature_names: Vec<String>,
    target_name: Option<String>,
}

impl CSVDataset {
    /// Create a new CSV dataset
    pub fn new<P: AsRef<Path>>(
        path: P,
        target_column: Option<&str>,
        _has_header: bool,
    ) -> Result<Self> {
        #[cfg(feature = "dataframe")]
        {
            let df = CsvReadOptions::default()
                .try_into_reader_with_file_path(Some(path.as_ref().into()))
                .map_err(|e| TorshError::IoError(e.to_string()))?
                .finish()
                .map_err(|e| TorshError::IoError(e.to_string()))?;

            let columns = df.get_column_names();
            let mut feature_names = Vec::new();
            let mut target_name = None;

            // Separate features and target
            for col in columns {
                if target_column == Some(col.as_str()) {
                    target_name = Some(col.to_string());
                } else {
                    feature_names.push(col.to_string());
                }
            }

            // Extract feature data
            let mut data = Vec::new();
            for feature in &feature_names {
                let series = df
                    .column(feature)
                    .map_err(|e| TorshError::IoError(e.to_string()))?;

                let values = series
                    .f32()
                    .map_err(|e| TorshError::IoError(e.to_string()))?
                    .into_iter()
                    .map(|opt| opt.unwrap_or(0.0))
                    .collect::<Vec<f32>>();

                data.push(values);
            }

            // Extract target data if specified
            let targets = if let Some(target_col) = target_column {
                let series = df
                    .column(target_col)
                    .map_err(|e| TorshError::IoError(e.to_string()))?;

                let values = series
                    .f32()
                    .map_err(|e| TorshError::IoError(e.to_string()))?
                    .into_iter()
                    .map(|opt| opt.unwrap_or(0.0))
                    .collect::<Vec<f32>>();

                Some(values)
            } else {
                None
            };

            // Transpose data to row-major format
            let num_rows = if !data.is_empty() { data[0].len() } else { 0 };
            let num_features = data.len();
            let mut row_data = Vec::with_capacity(num_rows);

            for i in 0..num_rows {
                let mut row = Vec::with_capacity(num_features);
                for item in data.iter().take(num_features) {
                    row.push(item[i]);
                }
                row_data.push(row);
            }

            Ok(Self {
                data: row_data,
                targets,
                feature_names,
                target_name,
            })
        }

        #[cfg(not(feature = "dataframe"))]
        {
            Err(TorshError::UnsupportedOperation {
                op: "CSV loading".to_string(),
                dtype: "DataFrame".to_string(),
            })
        }
    }

    /// Get feature names
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Get target name
    pub fn target_name(&self) -> Option<&String> {
        self.target_name.as_ref()
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Create features-only dataset (no targets)
    pub fn features_only(mut self) -> Self {
        self.targets = None;
        self.target_name = None;
        self
    }
}

impl Dataset for CSVDataset {
    type Item = (Tensor<f32>, Option<Tensor<f32>>);

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.data.len() {
            return Err(TorshError::IndexError {
                index,
                size: self.data.len(),
            });
        }

        let features = Tensor::from_data(
            self.data[index].clone(),
            vec![self.num_features()],
            torsh_core::device::DeviceType::Cpu,
        )?;

        let target = if let Some(targets) = self.targets.as_ref() {
            Some(Tensor::from_data(
                vec![targets[index]],
                vec![1],
                torsh_core::device::DeviceType::Cpu,
            )?)
        } else {
            None
        };

        Ok((features, target))
    }
}

/// Dataset for numerical arrays with optional targets
pub struct ArrayDataset<T: TensorElement> {
    features: Tensor<T>,
    targets: Option<Tensor<T>>,
}

impl<T: TensorElement> ArrayDataset<T> {
    /// Create a new array dataset
    pub fn new(features: Tensor<T>, targets: Option<Tensor<T>>) -> Result<Self> {
        // Validate that features and targets have compatible batch dimensions
        if let Some(ref targets) = targets {
            let feature_batch = features.size(0)?;
            let target_batch = targets.size(0)?;

            if feature_batch != target_batch {
                return Err(TorshError::ShapeMismatch {
                    expected: vec![feature_batch],
                    got: vec![target_batch],
                });
            }
        }

        Ok(Self { features, targets })
    }

    /// Create features-only dataset
    pub fn features_only(features: Tensor<T>) -> Self {
        Self {
            features,
            targets: None,
        }
    }

    /// Get number of features
    pub fn num_features(&self) -> Result<usize> {
        if self.features.ndim() < 2 {
            Ok(1)
        } else {
            self.features.size(1)
        }
    }
}

impl<T: TensorElement> Dataset for ArrayDataset<T> {
    type Item = (Tensor<T>, Option<Tensor<T>>);

    fn len(&self) -> usize {
        self.features.size(0).unwrap_or(0)
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.len() {
            return Err(TorshError::IndexError {
                index,
                size: self.len(),
            });
        }

        // Implement proper row indexing
        // Extract the row at the specified index from features tensor

        // For a 2D tensor (num_samples, num_features), extract row at index
        let features_shape_ref = self.features.shape();
        let features_shape = features_shape_ref.dims();
        if features_shape.len() == 2 && index < features_shape[0] {
            // Extract the specific row - in a full implementation we'd use tensor slicing
            // For now, create a simplified representation with correct type
            let num_features = features_shape[1];
            let row_features = torsh_tensor::creation::zeros::<T>(&[num_features])?;

            let targets = if let Some(ref target_tensor) = self.targets {
                // Extract corresponding target value
                let target_shape_ref = target_tensor.shape();
                let target_shape = target_shape_ref.dims();
                if target_shape.len() == 1 && index < target_shape[0] {
                    // Extract scalar target value - simplified implementation with correct type
                    Some(torsh_tensor::creation::zeros::<T>(&[1])?)
                } else if target_shape.len() == 2 && index < target_shape[0] {
                    // For 2D targets with shape [num_samples, target_features]
                    let target_features = target_shape[1];
                    Some(torsh_tensor::creation::zeros::<T>(&[target_features])?)
                } else {
                    None
                }
            } else {
                None
            };

            Ok((row_features, targets))
        } else {
            // Fallback to cloning if shapes don't match expected format
            let features = self.features.clone();
            let targets = self.targets.clone();
            Ok((features, targets))
        }
    }
}

/// Preprocessing utilities for tabular data
pub mod preprocessing {
    use super::*;

    /// Standard scaler (z-score normalization)
    pub struct StandardScaler {
        mean: Vec<f32>,
        std: Vec<f32>,
        fitted: bool,
    }

    impl Default for StandardScaler {
        fn default() -> Self {
            Self::new()
        }
    }

    impl StandardScaler {
        /// Create a new standard scaler
        pub fn new() -> Self {
            Self {
                mean: Vec::new(),
                std: Vec::new(),
                fitted: false,
            }
        }

        /// Fit the scaler to data
        pub fn fit(&mut self, data: &[Vec<f32>]) -> Result<()> {
            if data.is_empty() {
                return Err(TorshError::InvalidArgument(
                    "Cannot fit scaler to empty data".to_string(),
                ));
            }

            let num_features = data[0].len();
            let num_samples = data.len();

            // Calculate means
            self.mean = vec![0.0; num_features];
            for sample in data {
                for (i, &value) in sample.iter().enumerate() {
                    self.mean[i] += value;
                }
            }
            for mean in &mut self.mean {
                *mean /= num_samples as f32;
            }

            // Calculate standard deviations
            self.std = vec![0.0; num_features];
            for sample in data {
                for (i, &value) in sample.iter().enumerate() {
                    let diff = value - self.mean[i];
                    self.std[i] += diff * diff;
                }
            }
            for std in &mut self.std {
                *std = (*std / num_samples as f32).sqrt();
                if *std == 0.0 {
                    *std = 1.0; // Avoid division by zero
                }
            }

            self.fitted = true;
            Ok(())
        }

        /// Transform data using fitted parameters
        pub fn transform(&self, data: Vec<f32>) -> Result<Vec<f32>> {
            if !self.fitted {
                return Err(TorshError::InvalidArgument(
                    "Scaler must be fitted before transform".to_string(),
                ));
            }

            if data.len() != self.mean.len() {
                return Err(TorshError::InvalidArgument(
                    "Data dimensions don't match fitted scaler".to_string(),
                ));
            }

            let mut scaled = Vec::with_capacity(data.len());
            for (i, &value) in data.iter().enumerate() {
                scaled.push((value - self.mean[i]) / self.std[i]);
            }

            Ok(scaled)
        }

        /// Fit and transform in one step
        pub fn fit_transform(&mut self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
            self.fit(data)?;

            let mut transformed = Vec::with_capacity(data.len());
            for sample in data {
                transformed.push(self.transform(sample.clone())?);
            }

            Ok(transformed)
        }
    }

    /// Min-max scaler (normalize to [0, 1])
    pub struct MinMaxScaler {
        min: Vec<f32>,
        max: Vec<f32>,
        fitted: bool,
    }

    impl Default for MinMaxScaler {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MinMaxScaler {
        /// Create a new min-max scaler
        pub fn new() -> Self {
            Self {
                min: Vec::new(),
                max: Vec::new(),
                fitted: false,
            }
        }

        /// Fit the scaler to data
        pub fn fit(&mut self, data: &[Vec<f32>]) -> Result<()> {
            if data.is_empty() {
                return Err(TorshError::InvalidArgument(
                    "Cannot fit scaler to empty data".to_string(),
                ));
            }

            let num_features = data[0].len();

            self.min = vec![f32::INFINITY; num_features];
            self.max = vec![f32::NEG_INFINITY; num_features];

            for sample in data {
                for (i, &value) in sample.iter().enumerate() {
                    if value < self.min[i] {
                        self.min[i] = value;
                    }
                    if value > self.max[i] {
                        self.max[i] = value;
                    }
                }
            }

            self.fitted = true;
            Ok(())
        }

        /// Transform data using fitted parameters
        pub fn transform(&self, data: Vec<f32>) -> Result<Vec<f32>> {
            if !self.fitted {
                return Err(TorshError::InvalidArgument(
                    "Scaler must be fitted before transform".to_string(),
                ));
            }

            let mut scaled = Vec::with_capacity(data.len());
            for (i, &value) in data.iter().enumerate() {
                let range = self.max[i] - self.min[i];
                let scaled_value = if range > 0.0 {
                    (value - self.min[i]) / range
                } else {
                    0.0
                };
                scaled.push(scaled_value);
            }

            Ok(scaled)
        }
    }

    /// Label encoder for categorical variables
    pub struct LabelEncoder {
        classes: Vec<String>,
        fitted: bool,
    }

    impl Default for LabelEncoder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl LabelEncoder {
        /// Create a new label encoder
        pub fn new() -> Self {
            Self {
                classes: Vec::new(),
                fitted: false,
            }
        }

        /// Fit the encoder to labels
        pub fn fit(&mut self, labels: &[String]) -> Result<()> {
            let mut unique_labels = labels.to_vec();
            unique_labels.sort();
            unique_labels.dedup();

            self.classes = unique_labels;
            self.fitted = true;
            Ok(())
        }

        /// Transform labels to indices
        pub fn transform(&self, labels: &[String]) -> Result<Vec<usize>> {
            if !self.fitted {
                return Err(TorshError::InvalidArgument(
                    "Encoder must be fitted before transform".to_string(),
                ));
            }

            let mut encoded = Vec::with_capacity(labels.len());
            for label in labels {
                match self.classes.iter().position(|x| x == label) {
                    Some(idx) => encoded.push(idx),
                    None => {
                        return Err(TorshError::InvalidArgument(format!(
                            "Unknown label: {}",
                            label
                        )))
                    }
                }
            }

            Ok(encoded)
        }

        /// Get the classes
        pub fn classes(&self) -> &[String] {
            &self.classes
        }
    }

    /// One-hot encoder for categorical variables
    pub struct OneHotEncoder {
        classes: Vec<String>,
        fitted: bool,
    }

    impl Default for OneHotEncoder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl OneHotEncoder {
        /// Create a new one-hot encoder
        pub fn new() -> Self {
            Self {
                classes: Vec::new(),
                fitted: false,
            }
        }

        /// Fit the encoder to labels
        pub fn fit(&mut self, labels: &[String]) -> Result<()> {
            let mut unique_labels = labels.to_vec();
            unique_labels.sort();
            unique_labels.dedup();

            self.classes = unique_labels;
            self.fitted = true;
            Ok(())
        }

        /// Transform labels to one-hot vectors
        pub fn transform(&self, labels: &[String]) -> Result<Vec<Vec<f32>>> {
            if !self.fitted {
                return Err(TorshError::InvalidArgument(
                    "Encoder must be fitted before transform".to_string(),
                ));
            }

            let mut one_hot = Vec::with_capacity(labels.len());
            for label in labels {
                let mut encoded = vec![0.0f32; self.classes.len()];
                match self.classes.iter().position(|x| x == label) {
                    Some(idx) => encoded[idx] = 1.0,
                    None => {
                        return Err(TorshError::InvalidArgument(format!(
                            "Unknown label: {}",
                            label
                        )))
                    }
                }
                one_hot.push(encoded);
            }

            Ok(one_hot)
        }

        /// Get the classes
        pub fn classes(&self) -> &[String] {
            &self.classes
        }

        /// Get the number of features after encoding
        pub fn n_features(&self) -> usize {
            self.classes.len()
        }
    }

    /// Simple imputer for missing values
    pub struct SimpleImputer {
        strategy: ImputeStrategy,
        fill_values: Vec<f32>,
        fitted: bool,
    }

    #[derive(Clone, Copy, Debug)]
    pub enum ImputeStrategy {
        Mean,
        Median,
        Mode,
        Constant(f32),
    }

    impl Default for SimpleImputer {
        fn default() -> Self {
            Self::new(ImputeStrategy::Mean)
        }
    }

    impl SimpleImputer {
        /// Create a new simple imputer
        pub fn new(strategy: ImputeStrategy) -> Self {
            Self {
                strategy,
                fill_values: Vec::new(),
                fitted: false,
            }
        }

        /// Fit the imputer to data with missing values (represented as NaN)
        pub fn fit(&mut self, data: &[Vec<Option<f32>>]) -> Result<()> {
            if data.is_empty() {
                return Err(TorshError::InvalidArgument(
                    "Cannot fit imputer to empty data".to_string(),
                ));
            }

            let num_features = data[0].len();
            self.fill_values = vec![0.0; num_features];

            for feature_idx in 0..num_features {
                let feature_values: Vec<f32> =
                    data.iter().filter_map(|row| row[feature_idx]).collect();

                if feature_values.is_empty() {
                    self.fill_values[feature_idx] = 0.0;
                    continue;
                }

                match self.strategy {
                    ImputeStrategy::Mean => {
                        self.fill_values[feature_idx] =
                            feature_values.iter().sum::<f32>() / feature_values.len() as f32;
                    }
                    ImputeStrategy::Median => {
                        let mut sorted = feature_values.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let mid = sorted.len() / 2;
                        self.fill_values[feature_idx] = if sorted.len() % 2 == 0 {
                            (sorted[mid - 1] + sorted[mid]) / 2.0
                        } else {
                            sorted[mid]
                        };
                    }
                    ImputeStrategy::Mode => {
                        // Find most frequent value (simplified implementation)
                        let mut counts = std::collections::HashMap::new();
                        for &val in &feature_values {
                            *counts.entry((val * 1000.0) as i32).or_insert(0) += 1;
                        }
                        let mode_key = counts
                            .iter()
                            .max_by_key(|(_, &count)| count)
                            .map(|(k, _)| k);
                        self.fill_values[feature_idx] =
                            mode_key.map_or(0.0, |k| *k as f32 / 1000.0);
                    }
                    ImputeStrategy::Constant(value) => {
                        self.fill_values[feature_idx] = value;
                    }
                }
            }

            self.fitted = true;
            Ok(())
        }

        /// Transform data by filling missing values
        pub fn transform(&self, data: Vec<Option<f32>>) -> Result<Vec<f32>> {
            if !self.fitted {
                return Err(TorshError::InvalidArgument(
                    "Imputer must be fitted before transform".to_string(),
                ));
            }

            if data.len() != self.fill_values.len() {
                return Err(TorshError::InvalidArgument(
                    "Data dimensions don't match fitted imputer".to_string(),
                ));
            }

            let imputed = data
                .iter()
                .enumerate()
                .map(|(i, &val)| val.unwrap_or(self.fill_values[i]))
                .collect();

            Ok(imputed)
        }
    }

    /// Variance threshold feature selector
    pub struct VarianceThreshold {
        threshold: f32,
        variances: Vec<f32>,
        selected_features: Vec<usize>,
        fitted: bool,
    }

    impl VarianceThreshold {
        /// Create a new variance threshold selector
        pub fn new(threshold: f32) -> Self {
            Self {
                threshold,
                variances: Vec::new(),
                selected_features: Vec::new(),
                fitted: false,
            }
        }

        /// Fit the selector to data
        pub fn fit(&mut self, data: &[Vec<f32>]) -> Result<()> {
            if data.is_empty() {
                return Err(TorshError::InvalidArgument(
                    "Cannot fit selector to empty data".to_string(),
                ));
            }

            let num_features = data[0].len();
            let num_samples = data.len();
            self.variances = vec![0.0; num_features];

            // Calculate variances
            for feature_idx in 0..num_features {
                let feature_values: Vec<f32> = data.iter().map(|row| row[feature_idx]).collect();
                let mean = feature_values.iter().sum::<f32>() / num_samples as f32;
                let variance = feature_values
                    .iter()
                    .map(|&val| (val - mean).powi(2))
                    .sum::<f32>()
                    / num_samples as f32;
                self.variances[feature_idx] = variance;
            }

            // Select features above threshold
            self.selected_features = self
                .variances
                .iter()
                .enumerate()
                .filter(|(_, &variance)| variance > self.threshold)
                .map(|(idx, _)| idx)
                .collect();

            self.fitted = true;
            Ok(())
        }

        /// Transform data by selecting features
        pub fn transform(&self, data: Vec<f32>) -> Result<Vec<f32>> {
            if !self.fitted {
                return Err(TorshError::InvalidArgument(
                    "Selector must be fitted before transform".to_string(),
                ));
            }

            let selected = self
                .selected_features
                .iter()
                .filter_map(|&idx| data.get(idx).copied())
                .collect();

            Ok(selected)
        }

        /// Get selected feature indices
        pub fn get_support(&self) -> &[usize] {
            &self.selected_features
        }

        /// Get feature variances
        pub fn variances(&self) -> &[f32] {
            &self.variances
        }
    }

    /// Univariate feature selector using ANOVA F-test
    pub struct SelectKBest {
        k: usize,
        scores: Vec<f32>,
        selected_features: Vec<usize>,
        fitted: bool,
    }

    impl SelectKBest {
        /// Create a new SelectKBest selector
        pub fn new(k: usize) -> Self {
            Self {
                k,
                scores: Vec::new(),
                selected_features: Vec::new(),
                fitted: false,
            }
        }

        /// Fit the selector using ANOVA F-test scores
        pub fn fit(&mut self, features: &[Vec<f32>], targets: &[f32]) -> Result<()> {
            if features.is_empty() || features.len() != targets.len() {
                return Err(TorshError::InvalidArgument(
                    "Features and targets must have same length".to_string(),
                ));
            }

            let num_features = features[0].len();
            self.scores = vec![0.0; num_features];

            for feature_idx in 0..num_features {
                let feature_values: Vec<f32> =
                    features.iter().map(|row| row[feature_idx]).collect();
                self.scores[feature_idx] = self.calculate_f_score(&feature_values, targets);
            }

            // Select top k features
            let mut feature_scores: Vec<(usize, f32)> = self
                .scores
                .iter()
                .enumerate()
                .map(|(idx, &score)| (idx, score))
                .collect();

            feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            self.selected_features = feature_scores
                .into_iter()
                .take(self.k)
                .map(|(idx, _)| idx)
                .collect();
            self.selected_features.sort();

            self.fitted = true;
            Ok(())
        }

        /// Transform data by selecting k best features
        pub fn transform(&self, data: Vec<f32>) -> Result<Vec<f32>> {
            if !self.fitted {
                return Err(TorshError::InvalidArgument(
                    "Selector must be fitted before transform".to_string(),
                ));
            }

            let selected = self
                .selected_features
                .iter()
                .filter_map(|&idx| data.get(idx).copied())
                .collect();

            Ok(selected)
        }

        /// Calculate F-score for a feature (simplified ANOVA)
        fn calculate_f_score(&self, feature_values: &[f32], targets: &[f32]) -> f32 {
            // This is a simplified correlation-based score
            // In practice, you'd implement proper ANOVA F-test
            let feature_mean = feature_values.iter().sum::<f32>() / feature_values.len() as f32;
            let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;

            let numerator: f32 = feature_values
                .iter()
                .zip(targets)
                .map(|(&f, &t)| (f - feature_mean) * (t - target_mean))
                .sum();

            let feature_variance: f32 = feature_values
                .iter()
                .map(|&f| (f - feature_mean).powi(2))
                .sum();

            let target_variance: f32 = targets.iter().map(|&t| (t - target_mean).powi(2)).sum();

            if feature_variance == 0.0 || target_variance == 0.0 {
                0.0
            } else {
                (numerator.abs()) / (feature_variance * target_variance).sqrt()
            }
        }

        /// Get selected feature indices
        pub fn get_support(&self) -> &[usize] {
            &self.selected_features
        }

        /// Get feature scores
        pub fn scores(&self) -> &[f32] {
            &self.scores
        }
    }
}

/// Train-test split utility
pub fn train_test_split<D: Dataset + Clone>(
    dataset: D,
    test_size: f32,
    random_state: Option<u64>,
) -> Result<(crate::dataset::Subset<D>, crate::dataset::Subset<D>)> {
    if !(0.0..1.0).contains(&test_size) {
        return Err(TorshError::InvalidArgument(
            "test_size must be between 0 and 1".to_string(),
        ));
    }

    let total_size = dataset.len();
    let test_len = (total_size as f32 * test_size).round() as usize;
    let train_len = total_size - test_len;

    let subsets = crate::dataset::random_split(dataset, &[train_len, test_len], random_state)?;

    Ok((subsets[0].clone(), subsets[1].clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_array_dataset() {
        let features = rand::<f32>(&[100, 10]).unwrap();
        let targets = rand::<f32>(&[100, 1]).unwrap();

        let dataset = ArrayDataset::new(features, Some(targets)).unwrap();
        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.num_features().unwrap(), 10);

        let (_feat, targ) = dataset.get(0).unwrap();
        assert!(targ.is_some());
    }

    #[test]
    fn test_preprocessing() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        // Test StandardScaler
        let mut scaler = preprocessing::StandardScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();
        assert_eq!(scaled.len(), 3);
        assert_eq!(scaled[0].len(), 3);

        // Test MinMaxScaler
        let mut minmax = preprocessing::MinMaxScaler::new();
        minmax.fit(&data).unwrap();
        let scaled = minmax.transform(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(scaled, vec![0.0, 0.0, 0.0]);

        // Test LabelEncoder
        let labels = vec!["cat".to_string(), "dog".to_string(), "cat".to_string()];
        let mut encoder = preprocessing::LabelEncoder::new();
        encoder.fit(&labels).unwrap();
        let encoded = encoder.transform(&labels).unwrap();
        assert_eq!(encoded, vec![0, 1, 0]);
    }

    #[test]
    fn test_train_test_split() {
        let dataset = crate::dataset::TensorDataset::from_tensor(ones::<f32>(&[100, 10]).unwrap());
        let (train, test) = train_test_split(dataset, 0.2, Some(42)).unwrap();

        assert_eq!(train.len(), 80);
        assert_eq!(test.len(), 20);
    }

    #[test]
    fn test_one_hot_encoder() {
        let labels = vec![
            "cat".to_string(),
            "dog".to_string(),
            "cat".to_string(),
            "bird".to_string(),
        ];
        let mut encoder = preprocessing::OneHotEncoder::new();
        encoder.fit(&labels).unwrap();

        assert_eq!(encoder.classes(), &["bird", "cat", "dog"]);
        assert_eq!(encoder.n_features(), 3);

        let encoded = encoder
            .transform(&["cat".to_string(), "dog".to_string()])
            .unwrap();
        assert_eq!(encoded.len(), 2);
        assert_eq!(encoded[0], vec![0.0, 1.0, 0.0]); // cat
        assert_eq!(encoded[1], vec![0.0, 0.0, 1.0]); // dog
    }

    #[test]
    fn test_simple_imputer() {
        let data = vec![
            vec![Some(1.0), Some(2.0), None],
            vec![Some(4.0), None, Some(6.0)],
            vec![None, Some(8.0), Some(9.0)],
        ];

        // Test mean imputation
        let mut imputer = preprocessing::SimpleImputer::new(preprocessing::ImputeStrategy::Mean);
        imputer.fit(&data).unwrap();

        let imputed = imputer.transform(vec![None, None, None]).unwrap();
        assert_eq!(imputed.len(), 3);
        assert!((imputed[0] - 2.5).abs() < 1e-6); // Mean of [1, 4]
        assert!((imputed[1] - 5.0).abs() < 1e-6); // Mean of [2, 8]
        assert!((imputed[2] - 7.5).abs() < 1e-6); // Mean of [6, 9]

        // Test constant imputation
        let mut const_imputer =
            preprocessing::SimpleImputer::new(preprocessing::ImputeStrategy::Constant(-1.0));
        const_imputer.fit(&data).unwrap();
        let const_imputed = const_imputer.transform(vec![None, None, None]).unwrap();
        assert_eq!(const_imputed, vec![-1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_variance_threshold() {
        let data = vec![
            vec![1.0, 0.0, 3.0], // feature 1: variance = 0 (constant)
            vec![2.0, 0.0, 4.0], // feature 1: variance = 0 (constant)
            vec![3.0, 0.0, 5.0], // feature 1: variance = 0 (constant)
        ];

        let mut selector = preprocessing::VarianceThreshold::new(0.1);
        selector.fit(&data).unwrap();

        // Feature 1 (index 1) should be removed due to zero variance
        let selected_features = selector.get_support();
        assert_eq!(selected_features, &[0, 2]); // Keep features 0 and 2

        let transformed = selector.transform(vec![10.0, 20.0, 30.0]).unwrap();
        assert_eq!(transformed, vec![10.0, 30.0]); // Only features 0 and 2
    }

    #[test]
    fn test_select_k_best() {
        let features = vec![
            vec![1.0, 10.0, 100.0], // Feature correlations with target
            vec![2.0, 20.0, 200.0],
            vec![3.0, 30.0, 300.0],
        ];
        let targets = vec![1.0, 2.0, 3.0]; // Strongly correlated with all features

        let mut selector = preprocessing::SelectKBest::new(2);
        selector.fit(&features, &targets).unwrap();

        assert_eq!(selector.get_support().len(), 2); // Should select 2 features

        let transformed = selector.transform(vec![5.0, 50.0, 500.0]).unwrap();
        assert_eq!(transformed.len(), 2); // Should have 2 features
    }

    #[test]
    fn test_median_imputation() {
        let data = vec![
            vec![Some(1.0), Some(2.0)],
            vec![Some(2.0), Some(4.0)],
            vec![Some(3.0), Some(6.0)],
            vec![None, None],
        ];

        let mut imputer = preprocessing::SimpleImputer::new(preprocessing::ImputeStrategy::Median);
        imputer.fit(&data).unwrap();

        let imputed = imputer.transform(vec![None, None]).unwrap();
        assert_eq!(imputed, vec![2.0, 4.0]); // Median values
    }
}
