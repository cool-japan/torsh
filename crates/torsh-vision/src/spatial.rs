//! Spatial algorithms integration for computer vision
//!
//! This module integrates scirs2-spatial capabilities to enhance computer vision workflows
//! with advanced geometric algorithms, spatial data structures, and efficient distance computations.

pub mod distance;
pub mod interpolation;
pub mod matching;
pub mod structures;
pub mod transforms;

use crate::{Result, VisionError};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_spatial::distance::{cosine, euclidean, manhattan, EuclideanDistance};
use scirs2_spatial::kdtree::KDTree;
use scirs2_spatial::transform::{RigidTransform, Rotation};
use torsh_tensor::Tensor;

/// Spatial processing configuration
#[derive(Debug, Clone)]
pub struct SpatialConfig {
    /// Distance metric to use for spatial operations
    pub distance_metric: DistanceMetric,
    /// Number of nearest neighbors for spatial queries
    pub k_neighbors: usize,
    /// Whether to use SIMD acceleration
    pub use_simd: bool,
    /// Spatial tolerance for geometric operations
    pub tolerance: f64,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            distance_metric: DistanceMetric::Euclidean,
            k_neighbors: 5,
            use_simd: true,
            tolerance: 1e-8,
        }
    }
}

/// Supported distance metrics
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Chebyshev,
    Minkowski(f64),
}

impl DistanceMetric {
    /// Compute distance between two points
    pub fn compute(&self, a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Result<f64> {
        let distance = match self {
            DistanceMetric::Euclidean => euclidean(a.as_slice().unwrap(), b.as_slice().unwrap()),
            DistanceMetric::Manhattan => manhattan(a.as_slice().unwrap(), b.as_slice().unwrap()),
            DistanceMetric::Cosine => cosine(a.as_slice().unwrap(), b.as_slice().unwrap()),
            _ => euclidean(a.as_slice().unwrap(), b.as_slice().unwrap()), // Fallback to euclidean
        };

        Ok(distance)
    }
}

/// Spatial point representation
#[derive(Debug, Clone)]
pub struct SpatialPoint {
    pub coordinates: Array1<f64>,
    pub data: Option<Tensor>,
    pub id: Option<usize>,
}

impl SpatialPoint {
    /// Create a new spatial point
    pub fn new(coordinates: Array1<f64>) -> Self {
        Self {
            coordinates,
            data: None,
            id: None,
        }
    }

    /// Create a spatial point with associated data
    pub fn with_data(coordinates: Array1<f64>, data: Tensor) -> Self {
        Self {
            coordinates,
            data: Some(data),
            id: None,
        }
    }

    /// Get dimension of the point
    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }
}

/// Feature matching result
#[derive(Debug, Clone)]
pub struct FeatureMatch {
    pub query_idx: usize,
    pub target_idx: usize,
    pub distance: f64,
    pub confidence: f64,
}

impl FeatureMatch {
    /// Create a new feature match
    pub fn new(query_idx: usize, target_idx: usize, distance: f64) -> Self {
        Self {
            query_idx,
            target_idx,
            distance,
            confidence: 1.0 / (1.0 + distance),
        }
    }
}

/// Geometric transformation result
#[derive(Debug, Clone)]
pub struct TransformResult {
    pub rotation: Rotation,
    pub translation: Array1<f64>,
    pub scale: f64,
    pub error: f64,
}

/// Main spatial processor for computer vision
pub struct SpatialProcessor {
    config: SpatialConfig,
    kdtree: Option<KDTree<f32, EuclideanDistance<f32>>>,
}

impl SpatialProcessor {
    /// Create a new spatial processor
    pub fn new(config: SpatialConfig) -> Self {
        Self {
            config,
            kdtree: None,
        }
    }

    /// Build spatial index from feature points
    pub fn build_index(&mut self, points: &Array2<f64>) -> Result<()> {
        // Convert f64 to f32 for KDTree compatibility
        let points_f32 = points.mapv(|x| x as f32);

        let kdtree = KDTree::new(&points_f32)
            .map_err(|e| VisionError::Other(anyhow::anyhow!("Failed to build KDTree: {}", e)))?;

        self.kdtree = Some(kdtree);
        Ok(())
    }

    /// Find nearest neighbors for query points
    pub fn find_neighbors(&self, query: &ArrayView2<f64>) -> Result<Vec<Vec<usize>>> {
        let kdtree = self
            .kdtree
            .as_ref()
            .ok_or_else(|| VisionError::InvalidInput("Spatial index not built".to_string()))?;

        let mut all_neighbors = Vec::new();

        for query_point in query.outer_iter() {
            let (indices, _distances) = kdtree
                .query(
                    &query_point
                        .as_slice()
                        .unwrap()
                        .iter()
                        .map(|&x| x as f32)
                        .collect::<Vec<f32>>(),
                    self.config.k_neighbors,
                )
                .map_err(|e| {
                    VisionError::Other(anyhow::anyhow!("Neighbor search failed: {}", e))
                })?;
            all_neighbors.push(indices);
        }

        Ok(all_neighbors)
    }

    /// Match features between two sets of descriptors
    pub fn match_features(
        &self,
        descriptors1: &Array2<f64>,
        descriptors2: &Array2<f64>,
    ) -> Result<Vec<FeatureMatch>> {
        // Convert f64 to f32 for KDTree compatibility
        let descriptors2_f32 = descriptors2.mapv(|x| x as f32);
        let kdtree = KDTree::new(&descriptors2_f32)
            .map_err(|e| VisionError::Other(anyhow::anyhow!("Failed to build KDTree: {}", e)))?;

        let mut matches = Vec::new();

        for (i, descriptor) in descriptors1.outer_iter().enumerate() {
            let (indices, distances) = kdtree
                .query(
                    &descriptor
                        .as_slice()
                        .unwrap()
                        .iter()
                        .map(|&x| x as f32)
                        .collect::<Vec<f32>>(),
                    2,
                )
                .map_err(|e| {
                    VisionError::Other(anyhow::anyhow!("Feature matching failed: {}", e))
                })?;

            // Apply Lowe's ratio test
            if distances.len() >= 2 && distances[1] > 0.0 {
                let ratio = distances[0] / distances[1];
                if ratio < 0.7 {
                    matches.push(FeatureMatch::new(i, indices[0], distances[0] as f64));
                }
            }
        }

        Ok(matches)
    }

    /// Estimate rigid transformation between two point sets
    pub fn estimate_transform(
        &self,
        source: &Array2<f64>,
        target: &Array2<f64>,
    ) -> Result<TransformResult> {
        // For now, return a placeholder transformation
        // Real implementation would use scirs2_spatial::procrustes
        let rotation = Rotation::identity();
        let translation = Array1::zeros(source.ncols());

        Ok(TransformResult {
            rotation,
            translation,
            scale: 1.0,
            error: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_spatial_processor_creation() {
        let config = SpatialConfig::default();
        let processor = SpatialProcessor::new(config);
        assert!(processor.kdtree.is_none());
    }

    #[test]
    fn test_spatial_point_creation() {
        let coords = Array1::from(vec![1.0, 2.0, 3.0]);
        let point = SpatialPoint::new(coords.clone());
        assert_eq!(point.dimension(), 3);
        assert_eq!(point.coordinates, coords);
    }

    #[test]
    fn test_feature_match_creation() {
        let match_result = FeatureMatch::new(0, 1, 0.5);
        assert_eq!(match_result.query_idx, 0);
        assert_eq!(match_result.target_idx, 1);
        assert_eq!(match_result.distance, 0.5);
        assert!(match_result.confidence > 0.0);
    }

    #[test]
    fn test_distance_metric() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[2.0, 3.0], [4.0, 5.0]]);

        let metric = DistanceMetric::Euclidean;
        let distance = metric.compute(&a.view(), &b.view());
        assert!(distance.is_ok());
    }
}
