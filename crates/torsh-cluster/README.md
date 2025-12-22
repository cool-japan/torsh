# torsh-cluster

Unsupervised learning and clustering algorithms for ToRSh, powered by SciRS2.

## Overview

This crate provides comprehensive clustering and unsupervised learning algorithms with a PyTorch-compatible API. It leverages `scirs2-cluster` for high-performance implementations while maintaining full integration with ToRSh's tensor operations and autograd system.

## Features

- **Partitioning Methods**: K-Means, K-Medoids, Fuzzy C-Means
- **Hierarchical Clustering**: Agglomerative, Divisive, BIRCH
- **Density-Based Methods**: DBSCAN, OPTICS, HDBSCAN
- **Distribution-Based**: Gaussian Mixture Models (GMM), Expectation-Maximization
- **Spectral Methods**: Spectral clustering, Normalized cuts
- **Deep Clustering**: Deep Embedded Clustering (DEC), IDEC
- **Evaluation Metrics**: Silhouette score, Davies-Bouldin index, Calinski-Harabasz
- **Initialization Strategies**: K-means++, Random, Furthest-first
- **GPU Acceleration**: CUDA-accelerated clustering for large datasets

## Usage

### K-Means Clustering

```rust
use torsh_cluster::prelude::*;
use torsh_tensor::prelude::*;

// Create sample data
let data = tensor![[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]];

// Initialize K-Means with 2 clusters
let kmeans = KMeans::new(2)
    .max_iter(100)
    .tolerance(1e-4)
    .init_method(InitMethod::KMeansPlusPlus)
    .random_state(42);

// Fit the model
let result = kmeans.fit(&data)?;

// Get cluster assignments
let labels = result.labels();
println!("Cluster assignments: {:?}", labels);

// Get cluster centers
let centers = result.centers();
println!("Cluster centers: {:?}", centers);

// Predict cluster for new data
let new_point = tensor![[2.0, 3.0]];
let predicted_cluster = kmeans.predict(&new_point)?;
```

### DBSCAN - Density-Based Clustering

```rust
use torsh_cluster::prelude::*;

let data = load_dataset("examples/clustering_data.csv")?;

// Initialize DBSCAN
let dbscan = DBSCAN::new(0.5, 5)  // eps=0.5, min_samples=5
    .metric("euclidean");

// Fit and predict
let labels = dbscan.fit_predict(&data)?;

// Points labeled -1 are considered noise
let noise_points = labels.iter().filter(|&&x| x == -1).count();
println!("Number of noise points: {}", noise_points);

// Get core samples
let core_samples = dbscan.core_sample_indices()?;
```

### Gaussian Mixture Models (GMM)

```rust
use torsh_cluster::prelude::*;

let data = generate_blobs(1000, 3, 5, 1.0, 42)?;  // 1000 samples, 3 features, 5 clusters

// Initialize GMM with 5 components
let gmm = GaussianMixture::new(5)
    .covariance_type(CovarianceType::Full)
    .max_iter(100)
    .n_init(10)
    .init_params(InitParams::KMeans);

// Fit the model
gmm.fit(&data)?;

// Predict cluster probabilities
let probabilities = gmm.predict_proba(&data)?;
println!("Cluster probabilities shape: {:?}", probabilities.shape());

// Get model parameters
let means = gmm.means();
let covariances = gmm.covariances();
let weights = gmm.weights();

// Compute BIC and AIC
let bic = gmm.bic(&data)?;
let aic = gmm.aic(&data)?;
```

### Hierarchical Clustering

```rust
use torsh_cluster::prelude::*;

let data = tensor![[1.0, 2.0], [2.0, 3.0], [10.0, 11.0], [11.0, 12.0]];

// Agglomerative clustering
let hierarchical = AgglomerativeClustering::new(2)
    .linkage(Linkage::Ward)
    .affinity("euclidean");

let labels = hierarchical.fit_predict(&data)?;

// Get dendrogram
let dendrogram = hierarchical.dendrogram();

// Compute cophenetic correlation
let cophenetic_corr = hierarchical.cophenetic_correlation(&data)?;
```

### Spectral Clustering

```rust
use torsh_cluster::prelude::*;

let data = make_moons(200, Some(0.05), Some(42))?;  // Non-convex clusters

// Spectral clustering works well with non-convex shapes
let spectral = SpectralClustering::new(2)
    .affinity(Affinity::NearestNeighbors(10))
    .assign_labels("kmeans")
    .random_state(42);

let labels = spectral.fit_predict(&data)?;

// Use custom affinity matrix
let affinity_matrix = compute_rbf_kernel(&data, gamma=1.0)?;
let spectral_custom = SpectralClustering::new(2)
    .affinity(Affinity::Precomputed(affinity_matrix));

let labels = spectral_custom.fit_predict(&data)?;
```

### Deep Embedded Clustering (DEC)

```rust
use torsh_cluster::prelude::*;
use torsh_nn::prelude::*;

// Define autoencoder for feature learning
let autoencoder = Sequential::new()
    .add(Linear::new(784, 500, true))
    .add(ReLU::new(false))
    .add(Linear::new(500, 500, true))
    .add(ReLU::new(false))
    .add(Linear::new(500, 2000, true))
    .add(ReLU::new(false))
    .add(Linear::new(2000, 10, true));  // 10-dimensional embedding

// Initialize DEC
let dec = DeepEmbeddedClustering::new(10, autoencoder)  // 10 clusters
    .update_interval(140)
    .tolerance(0.001)
    .batch_size(256);

// Pretrain autoencoder
dec.pretrain(&data, epochs=50, learning_rate=0.01)?;

// Cluster
let labels = dec.fit_predict(&data, epochs=100)?;

// Get cluster centers in embedding space
let centers = dec.cluster_centers();
```

### Fuzzy C-Means

```rust
use torsh_cluster::prelude::*;

let data = tensor![[1.0, 2.0], [2.0, 3.0], [6.0, 7.0], [7.0, 8.0]];

// Fuzzy clustering allows soft assignments
let fcm = FuzzyCMeans::new(2)
    .fuzziness(2.0)  // Fuzziness parameter (m)
    .max_iter(100)
    .tolerance(1e-4);

let result = fcm.fit(&data)?;

// Get fuzzy membership matrix (each point belongs to all clusters with different probabilities)
let memberships = result.memberships();
println!("Fuzzy memberships:\n{:?}", memberships);

// Get hard cluster assignments
let labels = result.labels();  // Assigns to cluster with highest membership
```

### OPTICS - Ordering Points To Identify Clustering Structure

```rust
use torsh_cluster::prelude::*;

let data = load_dataset("complex_shapes.csv")?;

// OPTICS can find clusters of varying densities
let optics = OPTICS::new()
    .min_samples(5)
    .max_eps(f32::INFINITY)
    .metric("euclidean")
    .cluster_method("xi");

let labels = optics.fit_predict(&data)?;

// Get reachability plot
let reachability = optics.reachability();
let ordering = optics.ordering();

// Extract clusters with different parameters
let labels_dbscan = optics.extract_dbscan(eps=0.5)?;
```

## Evaluation Metrics

### Clustering Quality

```rust
use torsh_cluster::evaluation::*;

let data = generate_blobs(1000, 2, 3, 1.0, 42)?;
let labels = kmeans.fit_predict(&data)?;

// Silhouette score (-1 to 1, higher is better)
let silhouette = silhouette_score(&data, &labels, "euclidean")?;
println!("Silhouette score: {:.4}", silhouette);

// Davies-Bouldin index (lower is better)
let db_index = davies_bouldin_index(&data, &labels)?;
println!("Davies-Bouldin index: {:.4}", db_index);

// Calinski-Harabasz index (higher is better)
let ch_index = calinski_harabasz_score(&data, &labels)?;
println!("Calinski-Harabasz score: {:.4}", ch_index);

// Dunn index (higher is better)
let dunn = dunn_index(&data, &labels)?;
println!("Dunn index: {:.4}", dunn);
```

### External Validation (when ground truth is available)

```rust
use torsh_cluster::evaluation::*;

let true_labels = tensor![0, 0, 1, 1, 2, 2];
let pred_labels = tensor![0, 0, 1, 2, 2, 1];

// Adjusted Rand Index (-1 to 1, 1 is perfect)
let ari = adjusted_rand_score(&true_labels, &pred_labels)?;

// Normalized Mutual Information (0 to 1, 1 is perfect)
let nmi = normalized_mutual_info_score(&true_labels, &pred_labels)?;

// Fowlkes-Mallows score (0 to 1, 1 is perfect)
let fmi = fowlkes_mallows_score(&true_labels, &pred_labels)?;

// V-measure (0 to 1, 1 is perfect)
let v_measure = v_measure_score(&true_labels, &pred_labels)?;

// Homogeneity and completeness
let (homogeneity, completeness, v_measure) = homogeneity_completeness_v_measure(&true_labels, &pred_labels)?;
```

## Initialization Methods

```rust
use torsh_cluster::initialization::*;

let data = randn(&[1000, 10])?;
let n_clusters = 5;

// K-means++ initialization (smart initialization)
let centers = kmeans_plusplus(&data, n_clusters, Some(42))?;

// Random initialization
let centers = random_init(&data, n_clusters, Some(42))?;

// Furthest-first initialization
let centers = furthest_first(&data, n_clusters)?;

// Use custom initialization
let kmeans = KMeans::new(n_clusters)
    .init_centers(centers);
```

## Advanced Features

### Mini-Batch K-Means for Large Datasets

```rust
use torsh_cluster::prelude::*;

let large_data = randn(&[1_000_000, 100])?;  // 1M samples

// Mini-batch K-Means for scalability
let mb_kmeans = MiniBatchKMeans::new(100)  // 100 clusters
    .batch_size(1000)
    .max_iter(100)
    .reassignment_ratio(0.01);

let labels = mb_kmeans.fit_predict(&large_data)?;
```

### Consensus Clustering

```rust
use torsh_cluster::prelude::*;

let data = generate_blobs(500, 10, 5, 1.0, 42)?;

// Ensemble of clustering algorithms
let consensus = ConsensusCluster::new(5)
    .add_clusterer(KMeans::new(5))
    .add_clusterer(SpectralClustering::new(5))
    .add_clusterer(GaussianMixture::new(5))
    .n_runs(10)
    .aggregation_method("voting");

let labels = consensus.fit_predict(&data)?;
```

### GPU-Accelerated Clustering

```rust
use torsh_cluster::prelude::*;

let data = randn(&[10_000_000, 128])?.to_device("cuda:0")?;

// K-Means on GPU
let kmeans = KMeans::new(1000)
    .device("cuda:0")
    .max_iter(100);

let labels = kmeans.fit_predict(&data)?;
```

## Utilities

### Elbow Method for Optimal K

```rust
use torsh_cluster::utils::*;

let data = generate_blobs(1000, 10, 5, 1.0, 42)?;

// Try different numbers of clusters
let inertias = elbow_method(&data, 2..=10, "kmeans")?;

// Find the elbow point
let optimal_k = find_elbow(&inertias)?;
println!("Optimal number of clusters: {}", optimal_k);
```

### Silhouette Analysis

```rust
use torsh_cluster::utils::*;

// Compute silhouette scores for different k
let silhouette_scores = silhouette_analysis(&data, 2..=10, "kmeans")?;

// Plot silhouette diagram for specific clustering
let labels = kmeans.fit_predict(&data)?;
let silhouette_values = silhouette_samples(&data, &labels)?;
plot_silhouette_diagram(&silhouette_values, &labels)?;
```

## Integration with SciRS2

This crate leverages the SciRS2 ecosystem for:

- High-performance clustering algorithms through `scirs2-cluster`
- Optimized tensor operations via `scirs2-core`
- Statistical functions from `scirs2-stats`
- Evaluation metrics through `scirs2-metrics`
- Linear algebra operations via `scirs2-linalg`

All implementations follow the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md) for consistent APIs and optimal performance.

## Examples

See the `examples/` directory for more detailed examples:

- `kmeans_clustering.rs` - Basic K-Means clustering
- `dbscan_anomaly_detection.rs` - Anomaly detection with DBSCAN
- `gmm_soft_clustering.rs` - Probabilistic clustering with GMM
- `hierarchical_dendrogram.rs` - Hierarchical clustering and visualization
- `spectral_nonconvex.rs` - Spectral clustering on non-convex shapes
- `deep_clustering.rs` - Deep embedded clustering
- `large_scale_clustering.rs` - Mini-batch K-Means for big data

## Performance Tips

1. **Use Mini-Batch K-Means** for datasets with >100k samples
2. **Enable GPU acceleration** for large-scale clustering (>1M samples)
3. **Use K-means++ initialization** for better convergence
4. **Apply PCA** for dimensionality reduction before clustering high-dimensional data
5. **Use parallel features** with `features = ["parallel"]` in Cargo.toml

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.
