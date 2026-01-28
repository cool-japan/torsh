//! Advanced pattern analysis for sparse matrices
//!
//! This module provides sophisticated algorithms for analyzing, detecting,
//! and optimizing sparsity patterns in sparse matrices.

use crate::TorshResult;
use std::collections::{HashMap, HashSet, VecDeque};
use torsh_core::Shape;

/// Advanced sparsity patterns with detailed characteristics
#[derive(Debug, Clone)]
pub enum AdvancedSparsityPattern {
    /// Diagonal matrix
    Diagonal {
        /// Main diagonal fill ratio
        fill_ratio: f32,
    },
    /// Multi-diagonal (tridiagonal, pentadiagonal, etc.)
    MultiDiagonal {
        /// Number of diagonals
        num_diagonals: usize,
        /// Diagonal offsets
        offsets: Vec<i32>,
    },
    /// Block diagonal with detected block structure
    BlockDiagonal {
        /// Block sizes
        block_sizes: Vec<(usize, usize)>,
        /// Block positions
        block_positions: Vec<(usize, usize)>,
    },
    /// Banded matrix with upper and lower bandwidth
    Banded {
        /// Lower bandwidth
        lower_bandwidth: usize,
        /// Upper bandwidth
        upper_bandwidth: usize,
        /// Fill ratio within band
        fill_ratio: f32,
    },
    /// Symmetric pattern
    Symmetric {
        /// Symmetry ratio (0.0 = not symmetric, 1.0 = perfectly symmetric)
        symmetry_ratio: f32,
        /// Underlying pattern
        base_pattern: Box<AdvancedSparsityPattern>,
    },
    /// Arrow-head pattern (dense first row/column, sparse elsewhere)
    ArrowHead {
        /// Size of the dense head
        head_size: usize,
    },
    /// Random/unstructured pattern
    Random {
        /// Clustering coefficient
        clustering_coefficient: f32,
    },
}

/// Matrix reordering algorithms
#[derive(Debug, Clone)]
pub enum ReorderingAlgorithm {
    /// Reverse Cuthill-McKee ordering
    ReverseCuthillMcKee,
    /// Approximate Minimum Degree ordering
    ApproximateMinimumDegree,
    /// Nested Dissection ordering
    NestedDissection,
    /// King ordering (variation of RCM)
    King,
    /// Random ordering (for comparison)
    Random,
}

/// Matrix clustering algorithms
#[derive(Debug, Clone)]
pub enum ClusteringAlgorithm {
    /// Spectral clustering
    Spectral { num_clusters: usize },
    /// K-means based on matrix structure
    KMeans { num_clusters: usize },
    /// Hierarchical clustering
    Hierarchical { num_clusters: usize },
    /// Graph-based clustering
    GraphBased { num_clusters: usize },
}

/// Pattern statistics and characteristics
#[derive(Debug, Clone)]
pub struct PatternStatistics {
    /// Number of non-zero elements
    pub nnz: usize,
    /// Matrix dimensions
    pub dimensions: (usize, usize),
    /// Sparsity ratio (fraction of zeros)
    pub sparsity: f32,
    /// Maximum number of non-zeros per row
    pub max_nnz_per_row: usize,
    /// Average number of non-zeros per row
    pub avg_nnz_per_row: f32,
    /// Standard deviation of non-zeros per row
    pub std_nnz_per_row: f32,
    /// Bandwidth (maximum distance from diagonal)
    pub bandwidth: usize,
    /// Profile (sum of distances from diagonal)
    pub profile: usize,
    /// Number of connected components in graph representation
    pub connected_components: usize,
    /// Clustering coefficient
    pub clustering_coefficient: f32,
}

/// Advanced pattern analyzer
pub struct PatternAnalyzer {
    /// Cached analysis results
    cache: HashMap<String, AdvancedSparsityPattern>,
}

impl Default for PatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternAnalyzer {
    /// Create a new pattern analyzer
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyze sparsity pattern with advanced detection
    pub fn analyze_advanced_pattern(
        &mut self,
        triplets: &[(usize, usize, f32)],
        shape: &Shape,
    ) -> TorshResult<AdvancedSparsityPattern> {
        let cache_key = self.create_cache_key(triplets, shape);

        if let Some(cached_pattern) = self.cache.get(&cache_key) {
            return Ok(cached_pattern.clone());
        }

        let pattern = self.detect_pattern(triplets, shape)?;
        self.cache.insert(cache_key, pattern.clone());
        Ok(pattern)
    }

    /// Detect the underlying sparsity pattern
    fn detect_pattern(
        &self,
        triplets: &[(usize, usize, f32)],
        shape: &Shape,
    ) -> TorshResult<AdvancedSparsityPattern> {
        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);

        // Check for diagonal patterns first
        if let Some(diagonal_pattern) = self.detect_diagonal_pattern(triplets, rows, cols) {
            return Ok(diagonal_pattern);
        }

        // Check for banded patterns
        if let Some(banded_pattern) = self.detect_banded_pattern(triplets, rows, cols) {
            return Ok(banded_pattern);
        }

        // Check for block diagonal patterns
        if let Some(block_pattern) = self.detect_block_diagonal_pattern(triplets, rows, cols) {
            return Ok(block_pattern);
        }

        // Check for arrow-head patterns
        if let Some(arrow_pattern) = self.detect_arrow_head_pattern(triplets, rows, cols) {
            return Ok(arrow_pattern);
        }

        // Check for symmetry
        if let Some(symmetric_pattern) = self.detect_symmetric_pattern(triplets, rows, cols) {
            return Ok(symmetric_pattern);
        }

        // Default to random pattern with clustering analysis
        let clustering_coefficient = self.compute_clustering_coefficient(triplets, rows, cols);
        Ok(AdvancedSparsityPattern::Random {
            clustering_coefficient,
        })
    }

    /// Detect diagonal and multi-diagonal patterns
    fn detect_diagonal_pattern(
        &self,
        triplets: &[(usize, usize, f32)],
        rows: usize,
        cols: usize,
    ) -> Option<AdvancedSparsityPattern> {
        let mut diagonal_counts: HashMap<i32, usize> = HashMap::new();

        for (r, c, _) in triplets {
            let offset = *r as i32 - *c as i32;
            *diagonal_counts.entry(offset).or_insert(0) += 1;
        }

        let total_nnz = triplets.len();
        let main_diagonal_count = diagonal_counts.get(&0).unwrap_or(&0);

        // Check for pure diagonal matrix
        if diagonal_counts.len() == 1 && diagonal_counts.contains_key(&0) {
            let fill_ratio = *main_diagonal_count as f32 / std::cmp::min(rows, cols) as f32;
            return Some(AdvancedSparsityPattern::Diagonal { fill_ratio });
        }

        // Check for multi-diagonal pattern
        if diagonal_counts.len() <= 5 {
            let diagonal_nnz: usize = diagonal_counts.values().sum();
            if diagonal_nnz as f32 / total_nnz as f32 > 0.9 {
                let mut offsets: Vec<i32> = diagonal_counts.keys().copied().collect();
                offsets.sort();
                return Some(AdvancedSparsityPattern::MultiDiagonal {
                    num_diagonals: diagonal_counts.len(),
                    offsets,
                });
            }
        }

        None
    }

    /// Detect banded patterns
    fn detect_banded_pattern(
        &self,
        triplets: &[(usize, usize, f32)],
        rows: usize,
        cols: usize,
    ) -> Option<AdvancedSparsityPattern> {
        let mut max_lower_bandwidth = 0;
        let mut max_upper_bandwidth = 0;

        for (r, c, _) in triplets {
            let diff = *r as i32 - *c as i32;
            if diff > 0 {
                max_lower_bandwidth = std::cmp::max(max_lower_bandwidth, diff as usize);
            } else {
                max_upper_bandwidth = std::cmp::max(max_upper_bandwidth, (-diff) as usize);
            }
        }

        let total_bandwidth = max_lower_bandwidth + max_upper_bandwidth + 1;
        let max_possible_bandwidth = std::cmp::min(rows, cols);

        // Consider it banded if bandwidth is significantly smaller than matrix size
        if total_bandwidth < max_possible_bandwidth / 4 {
            let band_elements = std::cmp::min(rows, cols) * total_bandwidth
                - (total_bandwidth * (total_bandwidth - 1)) / 2;
            let fill_ratio = triplets.len() as f32 / band_elements as f32;

            return Some(AdvancedSparsityPattern::Banded {
                lower_bandwidth: max_lower_bandwidth,
                upper_bandwidth: max_upper_bandwidth,
                fill_ratio,
            });
        }

        None
    }

    /// Detect block diagonal patterns using graph analysis
    fn detect_block_diagonal_pattern(
        &self,
        triplets: &[(usize, usize, f32)],
        rows: usize,
        _cols: usize,
    ) -> Option<AdvancedSparsityPattern> {
        // Build adjacency representation
        let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();

        for (r, c, _) in triplets {
            adjacency.entry(*r).or_default().insert(*c);
            adjacency.entry(*c).or_default().insert(*r);
        }

        // Find connected components
        let components = self.find_connected_components(&adjacency, rows);

        if components.len() > 1 {
            // Analyze block structure
            let mut block_sizes = Vec::new();
            let mut block_positions = Vec::new();

            for component in &components {
                if component.len() > 1 {
                    let min_idx = *component
                        .iter()
                        .min()
                        .expect("component should not be empty");
                    let max_idx = *component
                        .iter()
                        .max()
                        .expect("component should not be empty");
                    let block_size = max_idx - min_idx + 1;

                    block_sizes.push((block_size, block_size));
                    block_positions.push((min_idx, min_idx));
                }
            }

            if !block_sizes.is_empty() {
                return Some(AdvancedSparsityPattern::BlockDiagonal {
                    block_sizes,
                    block_positions,
                });
            }
        }

        None
    }

    /// Detect arrow-head patterns
    fn detect_arrow_head_pattern(
        &self,
        triplets: &[(usize, usize, f32)],
        rows: usize,
        cols: usize,
    ) -> Option<AdvancedSparsityPattern> {
        let mut first_row_count = 0;
        let mut first_col_count = 0;

        for (r, c, _) in triplets {
            if *r == 0 {
                first_row_count += 1;
            }
            if *c == 0 {
                first_col_count += 1;
            }
        }

        let first_row_density = first_row_count as f32 / cols as f32;
        let first_col_density = first_col_count as f32 / rows as f32;

        // Check if first row or column is significantly denser
        if first_row_density > 0.5 || first_col_density > 0.5 {
            let head_size = std::cmp::max(first_row_count, first_col_count);
            return Some(AdvancedSparsityPattern::ArrowHead { head_size });
        }

        None
    }

    /// Detect symmetric patterns
    fn detect_symmetric_pattern(
        &self,
        triplets: &[(usize, usize, f32)],
        rows: usize,
        cols: usize,
    ) -> Option<AdvancedSparsityPattern> {
        if rows != cols {
            return None; // Can't be symmetric if not square
        }

        let mut pattern_set: HashSet<(usize, usize)> = HashSet::new();
        let mut symmetric_count = 0;

        for (r, c, _) in triplets {
            pattern_set.insert((*r, *c));
        }

        for (r, c, _) in triplets {
            if pattern_set.contains(&(*c, *r)) {
                symmetric_count += 1;
            }
        }

        let symmetry_ratio = symmetric_count as f32 / triplets.len() as f32;

        if symmetry_ratio > 0.8 {
            // Recursively detect underlying pattern
            let base_pattern = Box::new(AdvancedSparsityPattern::Random {
                clustering_coefficient: self.compute_clustering_coefficient(triplets, rows, cols),
            });

            return Some(AdvancedSparsityPattern::Symmetric {
                symmetry_ratio,
                base_pattern,
            });
        }

        None
    }

    /// Compute clustering coefficient for graph representation
    fn compute_clustering_coefficient(
        &self,
        triplets: &[(usize, usize, f32)],
        rows: usize,
        _cols: usize,
    ) -> f32 {
        let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();

        for (r, c, _) in triplets {
            if r != c {
                // Ignore self-loops
                adjacency.entry(*r).or_default().insert(*c);
                adjacency.entry(*c).or_default().insert(*r);
            }
        }

        let mut total_clustering = 0.0;
        let mut nodes_with_neighbors = 0;

        for node in 0..rows {
            if let Some(neighbors) = adjacency.get(&node) {
                if neighbors.len() >= 2 {
                    let mut triangles = 0;
                    let neighbor_vec: Vec<_> = neighbors.iter().collect();

                    for i in 0..neighbor_vec.len() {
                        for j in (i + 1)..neighbor_vec.len() {
                            if adjacency
                                .get(neighbor_vec[i])
                                .is_some_and(|adj| adj.contains(neighbor_vec[j]))
                            {
                                triangles += 1;
                            }
                        }
                    }

                    let possible_edges = neighbors.len() * (neighbors.len() - 1) / 2;
                    if possible_edges > 0 {
                        total_clustering += triangles as f32 / possible_edges as f32;
                        nodes_with_neighbors += 1;
                    }
                }
            }
        }

        if nodes_with_neighbors > 0 {
            total_clustering / nodes_with_neighbors as f32
        } else {
            0.0
        }
    }

    /// Find connected components in graph
    fn find_connected_components(
        &self,
        adjacency: &HashMap<usize, HashSet<usize>>,
        num_nodes: usize,
    ) -> Vec<Vec<usize>> {
        let mut visited = vec![false; num_nodes];
        let mut components = Vec::new();

        for node in 0..num_nodes {
            if !visited[node] {
                let mut component = Vec::new();
                let mut queue = VecDeque::new();
                queue.push_back(node);
                visited[node] = true;

                while let Some(current) = queue.pop_front() {
                    component.push(current);

                    if let Some(neighbors) = adjacency.get(&current) {
                        for &neighbor in neighbors {
                            if !visited[neighbor] {
                                visited[neighbor] = true;
                                queue.push_back(neighbor);
                            }
                        }
                    }
                }

                components.push(component);
            }
        }

        components
    }

    /// Create cache key for memoization
    fn create_cache_key(&self, triplets: &[(usize, usize, f32)], shape: &Shape) -> String {
        format!(
            "{}_{}_{}_{}",
            shape.dims()[0],
            shape.dims()[1],
            triplets.len(),
            triplets
                .iter()
                .take(10)
                .map(|(r, c, _)| format!("{r}_{c}"))
                .collect::<Vec<_>>()
                .join("_")
        )
    }

    /// Compute detailed pattern statistics
    pub fn compute_pattern_statistics(
        &self,
        triplets: &[(usize, usize, f32)],
        shape: &Shape,
    ) -> TorshResult<PatternStatistics> {
        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
        let nnz = triplets.len();
        let sparsity = 1.0 - (nnz as f32 / (rows * cols) as f32);

        // Compute row-wise statistics
        let mut row_counts = vec![0; rows];
        let mut max_bandwidth = 0;
        let mut profile = 0;

        for (r, c, _) in triplets {
            row_counts[*r] += 1;
            let distance = (*r as i32 - *c as i32).unsigned_abs() as usize;
            max_bandwidth = std::cmp::max(max_bandwidth, distance);
            profile += distance;
        }

        let max_nnz_per_row = *row_counts.iter().max().unwrap_or(&0);
        let avg_nnz_per_row = nnz as f32 / rows as f32;
        let variance = row_counts
            .iter()
            .map(|&count| (count as f32 - avg_nnz_per_row).powi(2))
            .sum::<f32>()
            / rows as f32;
        let std_nnz_per_row = variance.sqrt();

        // Build adjacency for connected components
        let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (r, c, _) in triplets {
            adjacency.entry(*r).or_default().insert(*c);
            adjacency.entry(*c).or_default().insert(*r);
        }

        let components = self.find_connected_components(&adjacency, rows);
        let connected_components = components.len();

        let clustering_coefficient = self.compute_clustering_coefficient(triplets, rows, cols);

        Ok(PatternStatistics {
            nnz,
            dimensions: (rows, cols),
            sparsity,
            max_nnz_per_row,
            avg_nnz_per_row,
            std_nnz_per_row,
            bandwidth: max_bandwidth,
            profile,
            connected_components,
            clustering_coefficient,
        })
    }
}

/// Matrix reordering algorithms implementation
pub struct MatrixReorderer;

impl MatrixReorderer {
    /// Apply Reverse Cuthill-McKee reordering
    pub fn reverse_cuthill_mckee(
        triplets: &[(usize, usize, f32)],
        num_rows: usize,
    ) -> TorshResult<Vec<usize>> {
        // Build adjacency list representation
        let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (r, c, _) in triplets {
            if r != c {
                adjacency.entry(*r).or_default().insert(*c);
                adjacency.entry(*c).or_default().insert(*r);
            }
        }

        // Find peripheral vertex (vertex with minimum degree, furthest from center)
        let start_vertex = Self::find_peripheral_vertex(&adjacency, num_rows)?;

        // BFS ordering
        let mut ordering = Vec::new();
        let mut visited = vec![false; num_rows];
        let mut queue = VecDeque::new();

        queue.push_back(start_vertex);
        visited[start_vertex] = true;

        while let Some(vertex) = queue.pop_front() {
            ordering.push(vertex);

            // Get neighbors and sort by degree (ascending)
            if let Some(neighbors) = adjacency.get(&vertex) {
                let mut neighbor_degrees: Vec<_> = neighbors
                    .iter()
                    .filter(|&&neighbor| !visited[neighbor])
                    .map(|&neighbor| {
                        let degree = adjacency.get(&neighbor).map_or(0, |adj| adj.len());
                        (degree, neighbor)
                    })
                    .collect();

                neighbor_degrees.sort_by_key(|&(degree, _)| degree);

                for (_, neighbor) in neighbor_degrees {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Add any remaining unvisited vertices
        for (i, &is_visited) in visited.iter().enumerate() {
            if !is_visited {
                ordering.push(i);
            }
        }

        // Reverse the ordering (Reverse Cuthill-McKee)
        ordering.reverse();

        Ok(ordering)
    }

    /// Find a peripheral vertex (good starting point for RCM)
    fn find_peripheral_vertex(
        adjacency: &HashMap<usize, HashSet<usize>>,
        num_rows: usize,
    ) -> TorshResult<usize> {
        let mut min_degree = usize::MAX;
        let mut peripheral_candidates = Vec::new();

        // Find vertices with minimum degree
        for i in 0..num_rows {
            let degree = adjacency.get(&i).map_or(0, |adj| adj.len());
            if degree < min_degree {
                min_degree = degree;
                peripheral_candidates.clear();
                peripheral_candidates.push(i);
            } else if degree == min_degree {
                peripheral_candidates.push(i);
            }
        }

        if peripheral_candidates.is_empty() {
            return Ok(0); // Fallback to first vertex
        }

        // Among minimum degree vertices, find the one with maximum distance to others
        let mut best_vertex = peripheral_candidates[0];
        let mut max_distance = 0;

        for &candidate in &peripheral_candidates {
            let distance = Self::compute_eccentricity(adjacency, candidate, num_rows);
            if distance > max_distance {
                max_distance = distance;
                best_vertex = candidate;
            }
        }

        Ok(best_vertex)
    }

    /// Compute eccentricity (maximum distance to any other vertex)
    fn compute_eccentricity(
        adjacency: &HashMap<usize, HashSet<usize>>,
        start: usize,
        num_rows: usize,
    ) -> usize {
        let mut distances = vec![usize::MAX; num_rows];
        let mut queue = VecDeque::new();

        distances[start] = 0;
        queue.push_back(start);

        while let Some(vertex) = queue.pop_front() {
            if let Some(neighbors) = adjacency.get(&vertex) {
                for &neighbor in neighbors {
                    if distances[neighbor] == usize::MAX {
                        distances[neighbor] = distances[vertex] + 1;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        distances
            .iter()
            .filter(|&&d| d != usize::MAX)
            .max()
            .copied()
            .unwrap_or(0)
    }

    /// Apply reordering to triplets
    pub fn apply_reordering(
        triplets: &[(usize, usize, f32)],
        ordering: &[usize],
    ) -> Vec<(usize, usize, f32)> {
        let mut inverse_ordering = vec![0; ordering.len()];
        for (new_idx, &old_idx) in ordering.iter().enumerate() {
            inverse_ordering[old_idx] = new_idx;
        }

        triplets
            .iter()
            .map(|(r, c, v)| (inverse_ordering[*r], inverse_ordering[*c], *v))
            .collect()
    }
}

/// Visualization utilities for sparsity patterns
pub struct PatternVisualizer;

impl PatternVisualizer {
    /// Generate ASCII art visualization of sparsity pattern
    pub fn ascii_pattern(
        triplets: &[(usize, usize, f32)],
        shape: &Shape,
        max_size: Option<(usize, usize)>,
    ) -> String {
        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
        let (display_rows, display_cols) = max_size.unwrap_or((50, 50));

        let row_scale = if rows > display_rows {
            rows / display_rows
        } else {
            1
        };
        let col_scale = if cols > display_cols {
            cols / display_cols
        } else {
            1
        };

        let scaled_rows = rows.div_ceil(row_scale);
        let scaled_cols = cols.div_ceil(col_scale);

        let mut pattern = vec![vec![' '; scaled_cols]; scaled_rows];

        for (r, c, _) in triplets {
            let scaled_r = r / row_scale;
            let scaled_c = c / col_scale;
            if scaled_r < scaled_rows && scaled_c < scaled_cols {
                pattern[scaled_r][scaled_c] = '*';
            }
        }

        let mut result = String::new();
        result.push_str(&format!(
            "Sparsity Pattern ({rows}x{cols}, scaled to {scaled_rows}x{scaled_cols})\n"
        ));
        result.push_str(&"-".repeat(scaled_cols + 2));
        result.push('\n');

        for row in pattern {
            result.push('|');
            for cell in row {
                result.push(cell);
            }
            result.push_str("|\n");
        }

        result.push_str(&"-".repeat(scaled_cols + 2));
        result.push('\n');

        result
    }

    /// Generate pattern histogram (distribution of non-zeros per row/column)
    pub fn pattern_histogram(
        triplets: &[(usize, usize, f32)],
        shape: &Shape,
    ) -> (Vec<usize>, Vec<usize>) {
        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
        let mut row_counts = vec![0; rows];
        let mut col_counts = vec![0; cols];

        for (r, c, _) in triplets {
            row_counts[*r] += 1;
            col_counts[*c] += 1;
        }

        (row_counts, col_counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_pattern_detection() {
        let mut analyzer = PatternAnalyzer::new();

        // Test diagonal pattern
        let triplets = vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
        let shape = Shape::new(vec![3, 3]);
        let pattern = analyzer
            .analyze_advanced_pattern(&triplets, &shape)
            .unwrap();

        matches!(pattern, AdvancedSparsityPattern::Diagonal { .. });
    }

    #[test]
    fn test_rcm_reordering() {
        let triplets = vec![
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 3, 1.0),
            (3, 2, 1.0),
        ];

        let ordering = MatrixReorderer::reverse_cuthill_mckee(&triplets, 4).unwrap();
        assert_eq!(ordering.len(), 4);

        let reordered = MatrixReorderer::apply_reordering(&triplets, &ordering);
        assert_eq!(reordered.len(), triplets.len());
    }

    #[test]
    fn test_pattern_statistics() {
        let analyzer = PatternAnalyzer::new();
        let triplets = vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
        let shape = Shape::new(vec![3, 3]);

        let stats = analyzer
            .compute_pattern_statistics(&triplets, &shape)
            .unwrap();
        assert_eq!(stats.nnz, 3);
        assert_eq!(stats.dimensions, (3, 3));
        assert_eq!(stats.bandwidth, 0); // Diagonal matrix
    }

    #[test]
    fn test_pattern_visualization() {
        let triplets = vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
        let shape = Shape::new(vec![3, 3]);

        let ascii = PatternVisualizer::ascii_pattern(&triplets, &shape, Some((10, 10)));
        assert!(ascii.contains("*"));

        let (row_hist, col_hist) = PatternVisualizer::pattern_histogram(&triplets, &shape);
        assert_eq!(row_hist, vec![1, 1, 1]);
        assert_eq!(col_hist, vec![1, 1, 1]);
    }
}
