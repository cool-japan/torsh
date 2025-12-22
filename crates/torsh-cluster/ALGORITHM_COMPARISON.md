# Clustering Algorithm Comparison Guide

This guide provides a comprehensive comparison of all clustering algorithms available in `torsh-cluster`, helping you choose the right algorithm for your specific use case.

## 📊 Quick Selection Guide

### By Use Case

| Use Case | Recommended Algorithms | Why? |
|----------|----------------------|------|
| **Customer Segmentation** | GMM, K-Means | Probabilistic assignments, interpretable clusters |
| **Image Segmentation** | Spectral, DBSCAN | Handles non-convex shapes, spatial relationships |
| **Anomaly Detection** | DBSCAN, HDBSCAN, OPTICS | Identifies outliers naturally |
| **Text Clustering** | K-Means, Hierarchical | Fast, scalable to high dimensions |
| **Streaming Data** | Online K-Means | Incremental updates, concept drift detection |
| **Varying Density Clusters** | HDBSCAN, OPTICS | Adaptive to density variations |
| **Fixed Number of Clusters** | K-Means, GMM, Spectral | Requires k as input |
| **Unknown Number of Clusters** | DBSCAN, HDBSCAN, OPTICS | Discovers clusters automatically |
| **Hierarchical Structure** | Hierarchical, HDBSCAN | Tree-based cluster relationships |
| **Large Datasets (>100k samples)** | Mini-batch K-Means, Online K-Means | Scalable implementations |

### By Data Characteristics

| Data Characteristic | Best Algorithms | Avoid |
|---------------------|-----------------|-------|
| **Spherical Clusters** | K-Means (all variants), GMM | DBSCAN (may oversegment) |
| **Non-convex Shapes** | DBSCAN, HDBSCAN, Spectral | K-Means, GMM |
| **High Dimensionality** | K-Means, GMM Diagonal | Spectral, DBSCAN (curse of dimensionality) |
| **Noisy Data** | DBSCAN, HDBSCAN, OPTICS | K-Means (sensitive to outliers) |
| **Unbalanced Clusters** | GMM, HDBSCAN | K-Means (assumes balanced) |
| **Streaming/Online** | Online K-Means | Batch algorithms |

## 🔍 Algorithm Details

### 1. K-Means

**Variants:** Lloyd's, Elkan's, Mini-batch

#### Strengths
- ✅ Fast and scalable
- ✅ Simple to understand and interpret
- ✅ Works well with spherical clusters
- ✅ Multiple optimization variants available

#### Weaknesses
- ❌ Requires specifying k beforehand
- ❌ Sensitive to initialization (use K-means++ to mitigate)
- ❌ Assumes spherical clusters of similar size
- ❌ Sensitive to outliers

#### Complexity
- **Time:** O(n·k·d·iter)
- **Space:** O(n·d + k·d)

#### When to Use
- You know the number of clusters
- Data has roughly spherical cluster shapes
- Need fast results on large datasets
- Interpretability is important

#### Example
```rust
use torsh_cluster::algorithms::kmeans::{KMeans, KMeansAlgorithm};
use torsh_cluster::traits::Fit;

// Standard K-Means (Lloyd's algorithm)
let kmeans = KMeans::new(5)
    .max_iters(100)
    .tolerance(1e-4)
    .random_state(42);

// Elkan's algorithm (faster for large k)
let kmeans_elkan = KMeans::new(20)
    .algorithm(KMeansAlgorithm::Elkan)
    .max_iters(100);

// Mini-batch (scalable to millions of samples)
let kmeans_mb = KMeans::new(5)
    .algorithm(KMeansAlgorithm::MiniBatch)
    .max_iters(200);
```

**Performance Tips:**
- Use Elkan's algorithm when k > 10
- Use Mini-batch for datasets > 100k samples
- Initialize with K-means++ (default) for better convergence

---

### 2. Gaussian Mixture Model (GMM)

**Variants:** Full, Diagonal, Spherical covariance

#### Strengths
- ✅ Probabilistic cluster assignments (soft clustering)
- ✅ Models cluster shape with covariance matrices
- ✅ Provides uncertainty estimates
- ✅ Flexible cluster shapes
- ✅ Statistical model with AIC/BIC for model selection

#### Weaknesses
- ❌ Slower than K-Means
- ❌ Can converge to local optima
- ❌ Requires specifying number of components
- ❌ Full covariance is expensive for high dimensions

#### Complexity
- **Time:** O(n·k·d²·iter)
- **Space:** O(n·k + k·d²)

#### When to Use
- Need probabilistic cluster assignments
- Want to model cluster uncertainty
- Clusters have different shapes/sizes
- Need statistical model for inference

#### Example
```rust
use torsh_cluster::algorithms::gaussian_mixture::{GaussianMixture, CovarianceType};

// Full covariance (most flexible)
let gmm_full = GaussianMixture::new(5)
    .covariance_type(CovarianceType::Full)
    .max_iters(100);

// Diagonal (faster, independent features)
let gmm_diag = GaussianMixture::new(5)
    .covariance_type(CovarianceType::Diag)
    .max_iters(100);

// Spherical (fastest, spherical clusters)
let gmm_sph = GaussianMixture::new(5)
    .covariance_type(CovarianceType::Spherical)
    .max_iters(100);

let result = gmm_full.fit(&data)?;
println!("AIC: {}, BIC: {}", result.aic, result.bic);
```

**Performance Tips:**
- Use Diagonal covariance for high-dimensional data
- Use Spherical for spherical clusters (fastest)
- Compare models using BIC/AIC
- Regularization (reg_covar) prevents numerical issues

---

### 3. DBSCAN

#### Strengths
- ✅ Finds arbitrary-shaped clusters
- ✅ Robust to outliers (marks them as noise)
- ✅ No need to specify number of clusters
- ✅ Simple hyperparameters (eps, min_samples)

#### Weaknesses
- ❌ Sensitive to parameter choice
- ❌ Struggles with varying densities
- ❌ Not suitable for high dimensions
- ❌ O(n²) complexity (without spatial indexing)

#### Complexity
- **Time:** O(n²) (O(n log n) with KD-trees)
- **Space:** O(n)

#### When to Use
- Clusters have arbitrary shapes
- Data contains outliers/noise
- Don't know number of clusters
- Clusters have similar densities

#### Example
```rust
use torsh_cluster::algorithms::dbscan::DBSCAN;

let dbscan = DBSCAN::new(0.5, 5)  // eps=0.5, min_samples=5
    .metric("euclidean");

let result = dbscan.fit(&data)?;
println!("Found {} clusters", result.n_clusters);
println!("Noise points: {}", result.noise_points.len());
```

**Parameter Selection:**
- **eps**: Use elbow method or k-distance graph
- **min_samples**: 2·d (d = dimensions) as rule of thumb
- Start with small eps and increase until noise points reduce

---

### 4. HDBSCAN (Hierarchical DBSCAN)

#### Strengths
- ✅ All DBSCAN advantages
- ✅ Handles varying density clusters
- ✅ Builds hierarchical cluster structure
- ✅ More robust parameter selection
- ✅ Automatic cluster extraction

#### Weaknesses
- ❌ More complex than DBSCAN
- ❌ Slower than DBSCAN
- ❌ Still sensitive to min_cluster_size

#### Complexity
- **Time:** O(n² log n)
- **Space:** O(n²)

#### When to Use
- DBSCAN use cases + varying densities
- Need hierarchical cluster structure
- More robust parameter selection needed
- Willing to trade speed for quality

#### Example
```rust
use torsh_cluster::algorithms::dbscan::{HDBSCAN, HDBSCANConfig};

let config = HDBSCANConfig {
    min_cluster_size: 5,
    min_samples: 3,
    ..Default::default()
};
let hdbscan = HDBSCAN::new(config);

let result = hdbscan.fit(&data)?;
println!("Hierarchy levels: {}", result.hierarchy.len());
```

**Parameter Tips:**
- **min_cluster_size**: Minimum meaningful cluster size
- **min_samples**: Controls noise sensitivity (default: min_cluster_size)
- Generally more robust than DBSCAN's eps parameter

---

### 5. OPTICS

#### Strengths
- ✅ Produces reachability plot for analysis
- ✅ Discovers clusters at multiple density scales
- ✅ No need to specify eps
- ✅ Good for exploratory analysis
- ✅ Handles varying density clusters

#### Weaknesses
- ❌ Requires post-processing to extract clusters
- ❌ O(n²) complexity
- ❌ Less intuitive than DBSCAN
- ❌ Cluster extraction has parameters

#### Complexity
- **Time:** O(n²)
- **Space:** O(n)

#### When to Use
- Exploratory data analysis
- Want to visualize cluster structure
- Multiple density scales in data
- Need ordering for visualization

#### Example
```rust
use torsh_cluster::algorithms::optics::OPTICS;

let optics = OPTICS::new(f64::INFINITY, 5)  // max_eps=∞, min_samples=5
    .xi(0.05);  // Extraction threshold

let result = optics.fit(&data)?;
println!("Reachability plot ordering: {:?}", result.ordering);
println!("Found {} clusters", result.n_clusters);
```

**Usage Tips:**
- Analyze reachability plot before extracting clusters
- Xi parameter controls cluster extraction sensitivity
- Use max_eps = ∞ to explore all density scales

---

### 6. Spectral Clustering

#### Strengths
- ✅ Handles non-convex clusters
- ✅ Works well with graph/network data
- ✅ Can use custom similarity measures
- ✅ Theoretically well-founded (graph cuts)

#### Weaknesses
- ❌ Requires specifying number of clusters
- ❌ Expensive: O(n³) due to eigendecomposition
- ❌ Requires full similarity matrix (O(n²) space)
- ❌ Not suitable for large datasets (n > 10k)

#### Complexity
- **Time:** O(n³)
- **Space:** O(n²)

#### When to Use
- Small to medium datasets (< 10k samples)
- Graph or network data
- Non-convex clusters
- Custom similarity measures needed
- Image segmentation

#### Example
```rust
use torsh_cluster::algorithms::spectral::{SpectralClustering, AffinityType};

let spectral = SpectralClustering::new(5)
    .affinity(AffinityType::Rbf)
    .gamma(1.0)
    .n_neighbors(10);

let result = spectral.fit(&data)?;
println!("Spectral embedding shape: {:?}", result.embedding.shape());
```

**Affinity Types:**
- **RBF**: Gaussian kernel (default)
- **KNearest**: K-nearest neighbors graph
- **FullyConnected**: All pairs connected
- **Precomputed**: Provide custom affinity matrix

---

### 7. Hierarchical Clustering

**Linkage Methods:** Single, Complete, Average, Ward

#### Strengths
- ✅ Produces dendrogram (tree structure)
- ✅ Can cut at any level for different k
- ✅ No random initialization
- ✅ Deterministic results
- ✅ Good for small datasets

#### Weaknesses
- ❌ O(n³) complexity (not scalable)
- ❌ Greedy algorithm (can't undo merges)
- ❌ Sensitive to noise and outliers
- ❌ Memory intensive for large n

#### Complexity
- **Time:** O(n³)
- **Space:** O(n²)

#### When to Use
- Small datasets (< 5k samples)
- Need hierarchical structure
- Want to explore different numbers of clusters
- Deterministic results required

#### Example
```rust
use torsh_cluster::algorithms::hierarchical::{AgglomerativeClustering, Linkage};

// Ward linkage (minimizes variance)
let hierarchical = AgglomerativeClustering::new(5)
    .linkage(Linkage::Ward);

// Average linkage (balanced)
let hier_avg = AgglomerativeClustering::new(5)
    .linkage(Linkage::Average);

let result = hierarchical.fit(&data)?;
```

**Linkage Comparison:**
- **Single**: Sensitive to noise, can create chains
- **Complete**: Compact clusters, sensitive to outliers
- **Average**: Balanced approach (recommended)
- **Ward**: Minimizes variance (similar to K-Means)

---

### 8. Online K-Means

#### Strengths
- ✅ Handles streaming data
- ✅ Incremental updates (O(1) per point)
- ✅ Concept drift detection
- ✅ Memory efficient
- ✅ Adaptive learning rate

#### Weaknesses
- ❌ Quality may be lower than batch K-Means
- ❌ Requires good initialization
- ❌ Order of data affects results
- ❌ Still assumes spherical clusters

#### Complexity
- **Time:** O(k·d) per update
- **Space:** O(k·d)

#### When to Use
- Streaming/online data
- Data too large for memory
- Need real-time clustering
- Concept drift in data distribution

#### Example
```rust
use torsh_cluster::algorithms::incremental::{OnlineKMeans, IncrementalClustering};

let mut online_kmeans = OnlineKMeans::new(5)?
    .drift_threshold(0.1);

// Process stream
for batch in data_stream {
    online_kmeans.update_batch(&batch)?;

    if online_kmeans.detect_drift() {
        println!("Concept drift detected!");
    }
}

let result = online_kmeans.get_current_result()?;
```

**Tips:**
- Initialize with batch K-Means on sample if possible
- Monitor drift detection for data distribution changes
- Adjust learning rate based on data stability

---

## 📈 Performance Comparison

### Speed Ranking (Fastest to Slowest)

1. **Mini-batch K-Means** - Sublinear in practice
2. **K-Means Lloyd** - Linear iterations
3. **K-Means Elkan** - Optimized for large k
4. **Online K-Means** - Per-point updates
5. **GMM Spherical** - Fast covariance
6. **GMM Diagonal** - Medium covariance
7. **DBSCAN** - Quadratic without indexing
8. **OPTICS** - Quadratic with ordering
9. **GMM Full** - Full covariance expensive
10. **HDBSCAN** - Hierarchical construction
11. **Spectral** - Eigendecomposition expensive
12. **Hierarchical** - Cubic complexity

### Scalability (Samples)

| Algorithm | Small (< 1k) | Medium (1k-10k) | Large (10k-100k) | Very Large (> 100k) |
|-----------|--------------|-----------------|------------------|---------------------|
| K-Means Lloyd | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| K-Means Elkan | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Mini-batch K-Means | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Online K-Means | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GMM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| DBSCAN | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| HDBSCAN | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| OPTICS | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Spectral | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ❌ |
| Hierarchical | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ❌ |

### Memory Usage Ranking (Lowest to Highest)

1. **Online K-Means** - O(k·d)
2. **K-Means** - O(n·d + k·d)
3. **GMM Spherical/Diagonal** - O(n·k + k·d)
4. **DBSCAN** - O(n)
5. **OPTICS** - O(n)
6. **GMM Full** - O(n·k + k·d²)
7. **HDBSCAN** - O(n²)
8. **Hierarchical** - O(n²)
9. **Spectral** - O(n²)

## 🎯 Decision Tree

```
Start Here
│
├─ Know number of clusters?
│  ├─ YES → Spherical clusters?
│  │  ├─ YES → Fast needed?
│  │  │  ├─ YES → Mini-batch K-Means
│  │  │  └─ NO → K-Means (Elkan for large k)
│  │  └─ NO → Complex shapes?
│  │     ├─ YES → Spectral Clustering
│  │     └─ NO → GMM
│  │
│  └─ NO → Arbitrary shapes?
│     ├─ YES → Varying density?
│     │  ├─ YES → HDBSCAN
│     │  └─ NO → DBSCAN
│     │
│     └─ NO → Need hierarchy?
│        ├─ YES → Hierarchical
│        └─ NO → Explore densities → OPTICS
│
├─ Streaming data?
│  └─ YES → Online K-Means
│
└─ Very large dataset (> 100k)?
   └─ YES → Mini-batch K-Means or Online K-Means
```

## 📚 Algorithm Selection Checklist

Use this checklist to select the best algorithm:

- [ ] **Data Size**: How many samples? (affects scalability)
- [ ] **Dimensions**: How many features? (affects some algorithms)
- [ ] **Cluster Shape**: Spherical or arbitrary?
- [ ] **Number of Clusters**: Known or unknown?
- [ ] **Density**: Uniform or varying?
- [ ] **Outliers**: Many outliers/noise points?
- [ ] **Online vs Batch**: Static or streaming data?
- [ ] **Speed vs Quality**: Performance requirements?
- [ ] **Interpretability**: Need probabilistic assignments?
- [ ] **Hierarchy**: Need hierarchical structure?

## 💡 Best Practices

### General Tips

1. **Always normalize/standardize your data** before clustering
2. **Try multiple algorithms** and compare results
3. **Use evaluation metrics** (Silhouette, ARI, etc.) to compare
4. **Visualize results** when possible (PCA/t-SNE for high dimensions)
5. **Set random seeds** for reproducibility

### K-Means Specific

- Use K-means++ initialization (default)
- Run multiple times with different seeds
- Use Elbow method or Gap Statistic to find optimal k
- Consider Mini-batch for large datasets

### Density-Based

- Plot k-distance graph to select eps (DBSCAN)
- Use HDBSCAN if you're unsure about parameters
- OPTICS for exploratory analysis first
- Consider spatial indexing for large datasets

### GMM Specific

- Start with Diagonal covariance, upgrade to Full if needed
- Use BIC/AIC for model comparison
- Add regularization (reg_covar) for numerical stability
- Try different random initializations

### Performance

- Profile your specific use case with benchmarks
- Use appropriate algorithm variants (Elkan, Mini-batch)
- Consider data dimensionality reduction first
- Use parallel implementations when available

## 🔬 Evaluation Metrics Guide

| Metric | When to Use | Range | Higher is Better? |
|--------|-------------|-------|-------------------|
| **Silhouette Score** | All algorithms | [-1, 1] | ✅ Yes |
| **Calinski-Harabasz Index** | Convex clusters | [0, ∞) | ✅ Yes |
| **Davies-Bouldin Index** | All algorithms | [0, ∞) | ❌ No (lower better) |
| **Adjusted Rand Index** | With ground truth | [-1, 1] | ✅ Yes |
| **NMI** | With ground truth | [0, 1] | ✅ Yes |
| **Gap Statistic** | Finding optimal k | R | ✅ Yes |

## 📖 Further Reading

- [Scikit-learn Clustering Comparison](https://scikit-learn.org/stable/modules/clustering.html)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
- [Spectral Clustering Tutorial](https://arxiv.org/abs/0711.0189)
- [K-means++: The Advantages of Careful Seeding](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)

---

**Need more help?** Check the examples in `examples/` directory or file an issue on GitHub!
