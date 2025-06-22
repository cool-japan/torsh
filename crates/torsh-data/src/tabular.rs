//! Tabular data utilities

#[cfg(feature = "dataframe")]
use polars::prelude::*;
use torsh_tensor::Tensor;
use torsh_core::{
    error::{Result, TorshError},
    dtype::TensorElement,
};
use crate::dataset::Dataset;

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, boxed::Box};
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
        has_header: bool,
    ) -> Result<Self> {
        #[cfg(feature = "dataframe")]
        {
            let df = LazyFrame::scan_csv(path.as_ref(), ScanArgsCSV::default())
                .map_err(|e| TorshError::IoError(e.to_string()))?
                .collect()
                .map_err(|e| TorshError::IoError(e.to_string()))?;
            
            let columns = df.get_column_names();
            let mut feature_names = Vec::new();
            let mut target_name = None;
            
            // Separate features and target
            for &col in columns {
                if Some(col) == target_column {
                    target_name = Some(col.to_string());
                } else {
                    feature_names.push(col.to_string());
                }
            }
            
            // Extract feature data
            let mut data = Vec::new();
            for feature in &feature_names {
                let series = df.column(feature)
                    .map_err(|e| TorshError::IoError(e.to_string()))?;
                
                let values = series.f32()
                    .map_err(|e| TorshError::IoError(e.to_string()))?
                    .into_iter()
                    .map(|opt| opt.unwrap_or(0.0))
                    .collect::<Vec<f32>>();
                
                data.push(values);
            }
            
            // Extract target data if specified
            let targets = if let Some(target_col) = target_column {
                let series = df.column(target_col)
                    .map_err(|e| TorshError::IoError(e.to_string()))?;
                
                let values = series.f32()
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
                for j in 0..num_features {
                    row.push(data[j][i]);
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
        );
        
        let target = if let Some(ref targets) = self.targets {
            Some(Tensor::from_data(
                vec![targets[index]],
                vec![1],
                torsh_core::device::DeviceType::Cpu,
            ))
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
        
        // TODO: Implement proper indexing when available
        // For now, return clones
        let features = self.features.clone();
        let targets = self.targets.as_ref().map(|t| t.clone());
        
        Ok((features, targets))
    }
}

/// Preprocessing utilities for tabular data
pub mod preprocessing {
    use super::*;
    use crate::transforms::Transform;
    
    /// Standard scaler (z-score normalization)
    pub struct StandardScaler {
        mean: Vec<f32>,
        std: Vec<f32>,
        fitted: bool,
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
                    "Cannot fit scaler to empty data".to_string()
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
                    "Scaler must be fitted before transform".to_string()
                ));
            }
            
            if data.len() != self.mean.len() {
                return Err(TorshError::InvalidArgument(
                    "Data dimensions don't match fitted scaler".to_string()
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
                    "Cannot fit scaler to empty data".to_string()
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
                    "Scaler must be fitted before transform".to_string()
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
            let mut unique_labels = labels.iter().cloned().collect::<Vec<_>>();
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
                    "Encoder must be fitted before transform".to_string()
                ));
            }
            
            let mut encoded = Vec::with_capacity(labels.len());
            for label in labels {
                match self.classes.iter().position(|x| x == label) {
                    Some(idx) => encoded.push(idx),
                    None => return Err(TorshError::InvalidArgument(
                        format!("Unknown label: {}", label)
                    )),
                }
            }
            
            Ok(encoded)
        }
        
        /// Get the classes
        pub fn classes(&self) -> &[String] {
            &self.classes
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
            "test_size must be between 0 and 1".to_string()
        ));
    }
    
    let total_size = dataset.len();
    let test_len = (total_size as f32 * test_size).round() as usize;
    let train_len = total_size - test_len;
    
    let subsets = crate::dataset::random_split(
        dataset,
        &[train_len, test_len],
        random_state,
    )?;
    
    Ok((subsets[0].clone(), subsets[1].clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;
    
    #[test]
    fn test_array_dataset() {
        let features = rand::<f32>(&[100, 10]);
        let targets = rand::<f32>(&[100, 1]);
        
        let dataset = ArrayDataset::new(features, Some(targets)).unwrap();
        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.num_features().unwrap(), 10);
        
        let (feat, targ) = dataset.get(0).unwrap();
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
        let dataset = crate::dataset::TensorDataset::from_tensor(ones::<f32>(&[100, 10]));
        let (train, test) = train_test_split(dataset, 0.2, Some(42)).unwrap();
        
        assert_eq!(train.len(), 80);
        assert_eq!(test.len(), 20);
    }
}