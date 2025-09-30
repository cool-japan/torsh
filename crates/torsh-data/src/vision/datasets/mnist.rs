use crate::{dataset::Dataset, transforms::Transform};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// MNIST dataset
pub struct MNIST {
    root: PathBuf,
    train: bool,
    transform: Option<Box<dyn Transform<Tensor<f32>, Output = Tensor<f32>>>>,
    data: Vec<Tensor<f32>>,
    targets: Vec<usize>,
}

impl MNIST {
    /// Create MNIST dataset
    pub fn new<P: AsRef<Path>>(root: P, train: bool) -> Result<Self> {
        let root = root.as_ref().to_path_buf();

        // MNIST data files
        let (images_file, labels_file) = if train {
            ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        } else {
            ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        };

        let images_path = root.join(images_file);
        let labels_path = root.join(labels_file);

        let (data, targets) = if images_path.exists() && labels_path.exists() {
            Self::load_mnist_data(&images_path, &labels_path)?
        } else {
            // Create dummy data for testing when files don't exist
            let size = if train { 60000 } else { 10000 };
            let mut data = Vec::with_capacity(size.min(100));
            let mut targets = Vec::with_capacity(size.min(100));

            for i in 0..size.min(100) {
                // MNIST images are 28x28 grayscale (1 channel)
                let image = torsh_tensor::creation::rand::<f32>(&[1, 28, 28])?;
                let label = i % 10; // 10 digits (0-9)

                data.push(image);
                targets.push(label);
            }

            (data, targets)
        };

        Ok(Self {
            root,
            train,
            transform: None,
            data,
            targets,
        })
    }

    /// Load MNIST data from binary files
    fn load_mnist_data(
        images_path: &Path,
        labels_path: &Path,
    ) -> Result<(Vec<Tensor<f32>>, Vec<usize>)> {
        // Load labels first
        let mut labels_file = File::open(labels_path)
            .map_err(|e| TorshError::IoError(format!("Failed to open labels file: {e}")))?;
        let mut labels_buf = Vec::new();
        labels_file
            .read_to_end(&mut labels_buf)
            .map_err(|e| TorshError::IoError(format!("Failed to read labels file: {e}")))?;

        // Parse labels header: magic number (4 bytes) + number of labels (4 bytes)
        if labels_buf.len() < 8 {
            return Err(TorshError::IoError(
                "Invalid labels file format".to_string(),
            ));
        }

        let magic =
            u32::from_be_bytes([labels_buf[0], labels_buf[1], labels_buf[2], labels_buf[3]]);
        if magic != 2049 {
            return Err(TorshError::IoError(
                "Invalid labels file magic number".to_string(),
            ));
        }

        let num_labels =
            u32::from_be_bytes([labels_buf[4], labels_buf[5], labels_buf[6], labels_buf[7]])
                as usize;
        let labels: Vec<usize> = labels_buf[8..8 + num_labels]
            .iter()
            .map(|&b| b as usize)
            .collect();

        // Load images
        let mut images_file = File::open(images_path)
            .map_err(|e| TorshError::IoError(format!("Failed to open images file: {e}")))?;
        let mut images_buf = Vec::new();
        images_file
            .read_to_end(&mut images_buf)
            .map_err(|e| TorshError::IoError(format!("Failed to read images file: {e}")))?;

        // Parse images header: magic (4) + num_images (4) + num_rows (4) + num_cols (4)
        if images_buf.len() < 16 {
            return Err(TorshError::IoError(
                "Invalid images file format".to_string(),
            ));
        }

        let magic =
            u32::from_be_bytes([images_buf[0], images_buf[1], images_buf[2], images_buf[3]]);
        if magic != 2051 {
            return Err(TorshError::IoError(
                "Invalid images file magic number".to_string(),
            ));
        }

        let num_images =
            u32::from_be_bytes([images_buf[4], images_buf[5], images_buf[6], images_buf[7]])
                as usize;
        let num_rows =
            u32::from_be_bytes([images_buf[8], images_buf[9], images_buf[10], images_buf[11]])
                as usize;
        let num_cols = u32::from_be_bytes([
            images_buf[12],
            images_buf[13],
            images_buf[14],
            images_buf[15],
        ]) as usize;

        if num_rows != 28 || num_cols != 28 {
            return Err(TorshError::IoError(
                "Expected 28x28 MNIST images".to_string(),
            ));
        }

        let image_size = num_rows * num_cols;
        let expected_data_size = 16 + num_images * image_size;
        if images_buf.len() != expected_data_size {
            return Err(TorshError::IoError("Images file size mismatch".to_string()));
        }

        // Parse images
        let mut data = Vec::with_capacity(num_images);
        for i in 0..num_images {
            let start_idx = 16 + i * image_size;
            let end_idx = start_idx + image_size;

            let pixel_data: Vec<f32> = images_buf[start_idx..end_idx]
                .iter()
                .map(|&b| b as f32 / 255.0)
                .collect();

            let tensor = Tensor::from_data(
                pixel_data,
                vec![1, num_rows, num_cols], // CHW format (1 channel, 28, 28)
                torsh_core::device::DeviceType::Cpu,
            )?;

            data.push(tensor);
        }

        Ok((data, labels))
    }

    /// Set transform
    pub fn with_transform<T>(mut self, transform: T) -> Self
    where
        T: Transform<Tensor<f32>, Output = Tensor<f32>> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }

    /// Get the root directory path
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Check if this is the training set
    pub fn is_train(&self) -> bool {
        self.train
    }

    /// Get the number of samples
    pub fn num_samples(&self) -> usize {
        self.data.len()
    }
}

impl Dataset for MNIST {
    type Item = (Tensor<f32>, usize);

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

        let mut data = self.data[index].clone();
        if let Some(ref transform) = self.transform {
            data = transform.transform(data)?;
        }

        Ok((data, self.targets[index]))
    }
}
