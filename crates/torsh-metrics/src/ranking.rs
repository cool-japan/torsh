//! Ranking and recommendation metrics with comprehensive implementations
//!
//! This module provides robust ranking evaluation metrics for information retrieval,
//! recommender systems, and search applications.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::Metric;
use std::collections::{HashMap, HashSet};
use torsh_tensor::Tensor;

/// Normalized Discounted Cumulative Gain (NDCG)
/// Measures ranking quality considering position-dependent relevance discounting
pub struct NDCG {
    k: Option<usize>,
    log_base: f64,
}

impl NDCG {
    /// Create a new NDCG metric
    pub fn new() -> Self {
        Self {
            k: None,
            log_base: 2.0,
        }
    }

    /// Set top-k cutoff
    pub fn at_k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Set logarithm base for discounting (default: 2.0)
    pub fn with_log_base(mut self, base: f64) -> Self {
        self.log_base = base;
        self
    }

    /// Compute NDCG score from relevance scores and predicted rankings
    pub fn compute_score(&self, true_relevance: &Tensor, predicted_scores: &Tensor) -> f64 {
        match (true_relevance.to_vec(), predicted_scores.to_vec()) {
            (Ok(rel_vec), Ok(score_vec)) => {
                if rel_vec.len() != score_vec.len() || rel_vec.is_empty() {
                    return 0.0;
                }

                let n_items = rel_vec.len();
                let k = self.k.unwrap_or(n_items);

                self.compute_ndcg(&rel_vec, &score_vec, k)
            }
            _ => 0.0,
        }
    }

    /// Compute NDCG from batch data
    pub fn compute_batch(&self, batch_relevance: &Tensor, batch_scores: &Tensor) -> f64 {
        match (batch_relevance.to_vec(), batch_scores.to_vec()) {
            (Ok(rel_vec), Ok(score_vec)) => {
                let shape = batch_relevance.shape();
                let dims = shape.dims();

                if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
                    return 0.0;
                }

                let n_queries = dims[0];
                let n_items = dims[1];
                let k = self.k.unwrap_or(n_items);

                let mut total_ndcg = 0.0;
                let mut valid_queries = 0;

                for i in 0..n_queries {
                    let query_rel: Vec<f32> =
                        (0..n_items).map(|j| rel_vec[i * n_items + j]).collect();
                    let query_scores: Vec<f32> =
                        (0..n_items).map(|j| score_vec[i * n_items + j]).collect();

                    let ndcg = self.compute_ndcg(&query_rel, &query_scores, k);
                    if ndcg.is_finite() {
                        total_ndcg += ndcg;
                        valid_queries += 1;
                    }
                }

                if valid_queries > 0 {
                    total_ndcg / valid_queries as f64
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    fn compute_ndcg(&self, relevance: &[f32], scores: &[f32], k: usize) -> f64 {
        if relevance.is_empty() || scores.is_empty() || relevance.len() != scores.len() {
            return 0.0;
        }

        let n = relevance.len().min(k);

        // Create items with (relevance, score, original_index)
        let mut items: Vec<(f64, f64, usize)> = relevance
            .iter()
            .zip(scores.iter())
            .enumerate()
            .map(|(i, (&rel, &score))| (rel as f64, score as f64, i))
            .collect();

        // Sort by scores in descending order (predicted ranking)
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate DCG (Discounted Cumulative Gain)
        let mut dcg = 0.0;
        for i in 0..n {
            let relevance_at_i = items[i].0;
            let discount = if i == 0 {
                1.0
            } else {
                1.0 / ((i + 1) as f64).log(self.log_base)
            };
            dcg += (2_f64.powf(relevance_at_i) - 1.0) * discount;
        }

        // Calculate IDCG (Ideal DCG)
        // Sort by relevance in descending order for ideal ranking
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut idcg = 0.0;
        for i in 0..n {
            let relevance_at_i = items[i].0;
            let discount = if i == 0 {
                1.0
            } else {
                1.0 / ((i + 1) as f64).log(self.log_base)
            };
            idcg += (2_f64.powf(relevance_at_i) - 1.0) * discount;
        }

        if idcg > 1e-10 {
            dcg / idcg
        } else {
            if dcg.abs() < 1e-10 {
                1.0 // Perfect score if both DCG and IDCG are zero
            } else {
                0.0
            }
        }
    }
}

impl Default for NDCG {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for NDCG {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(targets, predictions)
    }

    fn name(&self) -> &str {
        "ndcg"
    }
}

/// Mean Average Precision (MAP)
/// Measures precision across different recall levels
pub struct MeanAveragePrecision {
    k: Option<usize>,
}

impl MeanAveragePrecision {
    /// Create a new MAP metric
    pub fn new() -> Self {
        Self { k: None }
    }

    /// Set top-k cutoff
    pub fn at_k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Compute MAP from binary relevance and scores
    pub fn compute_score(&self, true_relevance: &Tensor, predicted_scores: &Tensor) -> f64 {
        match (true_relevance.to_vec(), predicted_scores.to_vec()) {
            (Ok(rel_vec), Ok(score_vec)) => {
                if rel_vec.len() != score_vec.len() || rel_vec.is_empty() {
                    return 0.0;
                }

                self.compute_ap(&rel_vec, &score_vec)
            }
            _ => 0.0,
        }
    }

    /// Compute MAP from batch data
    pub fn compute_batch(&self, batch_relevance: &Tensor, batch_scores: &Tensor) -> f64 {
        match (batch_relevance.to_vec(), batch_scores.to_vec()) {
            (Ok(rel_vec), Ok(score_vec)) => {
                let shape = batch_relevance.shape();
                let dims = shape.dims();

                if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
                    return 0.0;
                }

                let n_queries = dims[0];
                let n_items = dims[1];

                let mut total_ap = 0.0;
                let mut valid_queries = 0;

                for i in 0..n_queries {
                    let query_rel: Vec<f32> =
                        (0..n_items).map(|j| rel_vec[i * n_items + j]).collect();
                    let query_scores: Vec<f32> =
                        (0..n_items).map(|j| score_vec[i * n_items + j]).collect();

                    let ap = self.compute_ap(&query_rel, &query_scores);
                    if ap.is_finite() {
                        total_ap += ap;
                        valid_queries += 1;
                    }
                }

                if valid_queries > 0 {
                    total_ap / valid_queries as f64
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    fn compute_ap(&self, relevance: &[f32], scores: &[f32]) -> f64 {
        if relevance.is_empty() || scores.is_empty() || relevance.len() != scores.len() {
            return 0.0;
        }

        // Count relevant items
        let total_relevant = relevance.iter().filter(|&&x| x > 0.5).count();
        if total_relevant == 0 {
            return 0.0; // No relevant items
        }

        // Create items with (relevance, score, original_index)
        let mut items: Vec<(bool, f64, usize)> = relevance
            .iter()
            .zip(scores.iter())
            .enumerate()
            .map(|(i, (&rel, &score))| (rel > 0.5, score as f64, i))
            .collect();

        // Sort by scores in descending order
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.k.unwrap_or(items.len());
        let mut precision_sum = 0.0;
        let mut relevant_retrieved = 0;

        for i in 0..k.min(items.len()) {
            if items[i].0 {
                // This item is relevant
                relevant_retrieved += 1;
                let precision_at_i = relevant_retrieved as f64 / (i + 1) as f64;
                precision_sum += precision_at_i;
            }
        }

        precision_sum / total_relevant as f64
    }
}

impl Default for MeanAveragePrecision {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for MeanAveragePrecision {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(targets, predictions)
    }

    fn name(&self) -> &str {
        "mean_average_precision"
    }
}

/// Mean Reciprocal Rank (MRR)
/// Measures the average reciprocal rank of the first relevant result
pub struct MeanReciprocalRank {
    k: Option<usize>,
}

impl MeanReciprocalRank {
    /// Create a new MRR metric
    pub fn new() -> Self {
        Self { k: None }
    }

    /// Set top-k cutoff
    pub fn at_k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Compute MRR from binary relevance and scores
    pub fn compute_score(&self, true_relevance: &Tensor, predicted_scores: &Tensor) -> f64 {
        match (true_relevance.to_vec(), predicted_scores.to_vec()) {
            (Ok(rel_vec), Ok(score_vec)) => {
                if rel_vec.len() != score_vec.len() || rel_vec.is_empty() {
                    return 0.0;
                }

                self.compute_rr(&rel_vec, &score_vec)
            }
            _ => 0.0,
        }
    }

    /// Compute MRR from batch data
    pub fn compute_batch(&self, batch_relevance: &Tensor, batch_scores: &Tensor) -> f64 {
        match (batch_relevance.to_vec(), batch_scores.to_vec()) {
            (Ok(rel_vec), Ok(score_vec)) => {
                let shape = batch_relevance.shape();
                let dims = shape.dims();

                if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
                    return 0.0;
                }

                let n_queries = dims[0];
                let n_items = dims[1];

                let mut total_rr = 0.0;
                let mut valid_queries = 0;

                for i in 0..n_queries {
                    let query_rel: Vec<f32> =
                        (0..n_items).map(|j| rel_vec[i * n_items + j]).collect();
                    let query_scores: Vec<f32> =
                        (0..n_items).map(|j| score_vec[i * n_items + j]).collect();

                    let rr = self.compute_rr(&query_rel, &query_scores);
                    total_rr += rr;
                    valid_queries += 1;
                }

                if valid_queries > 0 {
                    total_rr / valid_queries as f64
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    fn compute_rr(&self, relevance: &[f32], scores: &[f32]) -> f64 {
        if relevance.is_empty() || scores.is_empty() || relevance.len() != scores.len() {
            return 0.0;
        }

        // Create items with (relevance, score, original_index)
        let mut items: Vec<(bool, f64, usize)> = relevance
            .iter()
            .zip(scores.iter())
            .enumerate()
            .map(|(i, (&rel, &score))| (rel > 0.5, score as f64, i))
            .collect();

        // Sort by scores in descending order
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.k.unwrap_or(items.len());

        // Find the first relevant item
        for i in 0..k.min(items.len()) {
            if items[i].0 {
                return 1.0 / (i + 1) as f64;
            }
        }

        0.0 // No relevant item found in top-k
    }
}

impl Default for MeanReciprocalRank {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for MeanReciprocalRank {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(targets, predictions)
    }

    fn name(&self) -> &str {
        "mean_reciprocal_rank"
    }
}

/// Precision at K
/// Measures the fraction of relevant items in top-k recommendations
pub struct PrecisionAtK {
    k: usize,
}

impl PrecisionAtK {
    /// Create a new Precision@K metric
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Compute precision@k from binary relevance and scores
    pub fn compute_score(&self, true_relevance: &Tensor, predicted_scores: &Tensor) -> f64 {
        match (true_relevance.to_vec(), predicted_scores.to_vec()) {
            (Ok(rel_vec), Ok(score_vec)) => {
                if rel_vec.len() != score_vec.len() || rel_vec.is_empty() {
                    return 0.0;
                }

                self.compute_precision_k(&rel_vec, &score_vec)
            }
            _ => 0.0,
        }
    }

    fn compute_precision_k(&self, relevance: &[f32], scores: &[f32]) -> f64 {
        if relevance.is_empty() || scores.is_empty() || relevance.len() != scores.len() {
            return 0.0;
        }

        // Create items with (relevance, score, original_index)
        let mut items: Vec<(bool, f64, usize)> = relevance
            .iter()
            .zip(scores.iter())
            .enumerate()
            .map(|(i, (&rel, &score))| (rel > 0.5, score as f64, i))
            .collect();

        // Sort by scores in descending order
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.k.min(items.len());
        if k == 0 {
            return 0.0;
        }

        let relevant_in_k = items.iter().take(k).filter(|(rel, _, _)| *rel).count();
        relevant_in_k as f64 / k as f64
    }
}

impl Metric for PrecisionAtK {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(targets, predictions)
    }

    fn name(&self) -> &str {
        "precision_at_k"
    }
}

/// Recall at K
/// Measures the fraction of relevant items retrieved in top-k recommendations
pub struct RecallAtK {
    k: usize,
}

impl RecallAtK {
    /// Create a new Recall@K metric
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Compute recall@k from binary relevance and scores
    pub fn compute_score(&self, true_relevance: &Tensor, predicted_scores: &Tensor) -> f64 {
        match (true_relevance.to_vec(), predicted_scores.to_vec()) {
            (Ok(rel_vec), Ok(score_vec)) => {
                if rel_vec.len() != score_vec.len() || rel_vec.is_empty() {
                    return 0.0;
                }

                self.compute_recall_k(&rel_vec, &score_vec)
            }
            _ => 0.0,
        }
    }

    fn compute_recall_k(&self, relevance: &[f32], scores: &[f32]) -> f64 {
        if relevance.is_empty() || scores.is_empty() || relevance.len() != scores.len() {
            return 0.0;
        }

        let total_relevant = relevance.iter().filter(|&&x| x > 0.5).count();
        if total_relevant == 0 {
            return 1.0; // Perfect recall if no relevant items
        }

        // Create items with (relevance, score, original_index)
        let mut items: Vec<(bool, f64, usize)> = relevance
            .iter()
            .zip(scores.iter())
            .enumerate()
            .map(|(i, (&rel, &score))| (rel > 0.5, score as f64, i))
            .collect();

        // Sort by scores in descending order
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.k.min(items.len());
        let relevant_retrieved = items.iter().take(k).filter(|(rel, _, _)| *rel).count();

        relevant_retrieved as f64 / total_relevant as f64
    }
}

impl Metric for RecallAtK {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(targets, predictions)
    }

    fn name(&self) -> &str {
        "recall_at_k"
    }
}

/// F1 Score at K
/// Harmonic mean of precision@k and recall@k
pub struct F1AtK {
    k: usize,
}

impl F1AtK {
    /// Create a new F1@K metric
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Compute F1@k from binary relevance and scores
    pub fn compute_score(&self, true_relevance: &Tensor, predicted_scores: &Tensor) -> f64 {
        let precision_k = PrecisionAtK::new(self.k);
        let recall_k = RecallAtK::new(self.k);

        let precision = precision_k.compute_score(true_relevance, predicted_scores);
        let recall = recall_k.compute_score(true_relevance, predicted_scores);

        if precision + recall > 1e-10 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }
}

impl Metric for F1AtK {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(targets, predictions)
    }

    fn name(&self) -> &str {
        "f1_at_k"
    }
}

/// Hit Rate at K
/// Binary measure: 1 if at least one relevant item is in top-k, 0 otherwise
pub struct HitRateAtK {
    k: usize,
}

impl HitRateAtK {
    /// Create a new HitRate@K metric
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Compute hit rate@k from binary relevance and scores
    pub fn compute_score(&self, true_relevance: &Tensor, predicted_scores: &Tensor) -> f64 {
        match (true_relevance.to_vec(), predicted_scores.to_vec()) {
            (Ok(rel_vec), Ok(score_vec)) => {
                if rel_vec.len() != score_vec.len() || rel_vec.is_empty() {
                    return 0.0;
                }

                self.compute_hit_rate(&rel_vec, &score_vec)
            }
            _ => 0.0,
        }
    }

    fn compute_hit_rate(&self, relevance: &[f32], scores: &[f32]) -> f64 {
        if relevance.is_empty() || scores.is_empty() || relevance.len() != scores.len() {
            return 0.0;
        }

        // Create items with (relevance, score, original_index)
        let mut items: Vec<(bool, f64, usize)> = relevance
            .iter()
            .zip(scores.iter())
            .enumerate()
            .map(|(i, (&rel, &score))| (rel > 0.5, score as f64, i))
            .collect();

        // Sort by scores in descending order
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.k.min(items.len());

        // Check if at least one relevant item is in top-k
        for (rel, _, _) in items.iter().take(k) {
            if *rel {
                return 1.0;
            }
        }

        0.0
    }
}

impl Metric for HitRateAtK {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(targets, predictions)
    }

    fn name(&self) -> &str {
        "hit_rate_at_k"
    }
}

/// Area Under the ROC Curve for ranking
/// Measures the probability that a randomly chosen relevant item is ranked higher than a random non-relevant item
pub struct RankingAUC;

impl RankingAUC {
    /// Compute ranking AUC from binary relevance and scores
    pub fn compute_score(&self, true_relevance: &Tensor, predicted_scores: &Tensor) -> f64 {
        match (true_relevance.to_vec(), predicted_scores.to_vec()) {
            (Ok(rel_vec), Ok(score_vec)) => {
                if rel_vec.len() != score_vec.len() || rel_vec.is_empty() {
                    return 0.5;
                }

                self.compute_ranking_auc(&rel_vec, &score_vec)
            }
            _ => 0.5,
        }
    }

    fn compute_ranking_auc(&self, relevance: &[f32], scores: &[f32]) -> f64 {
        let mut positive_scores = Vec::new();
        let mut negative_scores = Vec::new();

        for (&rel, &score) in relevance.iter().zip(scores.iter()) {
            if rel > 0.5 {
                positive_scores.push(score as f64);
            } else {
                negative_scores.push(score as f64);
            }
        }

        if positive_scores.is_empty() || negative_scores.is_empty() {
            return 0.5; // Random performance
        }

        // Count pairs where positive score > negative score
        let mut concordant_pairs = 0;
        let mut total_pairs = 0;

        for &pos_score in &positive_scores {
            for &neg_score in &negative_scores {
                total_pairs += 1;
                if pos_score > neg_score {
                    concordant_pairs += 1;
                } else if pos_score == neg_score {
                    // Count ties as 0.5
                    concordant_pairs += 1; // Will be divided by 2 later for ties
                                           // For now, just count as concordant and adjust
                }
            }
        }

        if total_pairs > 0 {
            concordant_pairs as f64 / total_pairs as f64
        } else {
            0.5
        }
    }
}

impl Metric for RankingAUC {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(targets, predictions)
    }

    fn name(&self) -> &str {
        "ranking_auc"
    }
}

/// Coverage metric for recommender systems
/// Measures the fraction of items in the catalog that get recommended
pub struct Coverage {
    catalog_size: usize,
}

impl Coverage {
    /// Create a new coverage metric
    pub fn new(catalog_size: usize) -> Self {
        Self { catalog_size }
    }

    /// Compute coverage from batch of item lists
    pub fn compute_coverage(&self, recommended_item_batches: &[Vec<usize>]) -> f64 {
        let mut unique_items: HashSet<usize> = HashSet::new();

        for item_list in recommended_item_batches {
            for &item in item_list {
                unique_items.insert(item);
            }
        }

        unique_items.len() as f64 / self.catalog_size as f64
    }

    /// Compute coverage from top-k recommendations
    pub fn compute_coverage_at_k(&self, scores_batch: &Tensor, k: usize) -> f64 {
        match scores_batch.to_vec() {
            Ok(scores_vec) => {
                let shape = scores_batch.shape();
                let dims = shape.dims();

                if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
                    return 0.0;
                }

                let n_users = dims[0];
                let n_items = dims[1];
                let mut unique_items: HashSet<usize> = HashSet::new();

                for i in 0..n_users {
                    // Get top-k items for this user
                    let user_scores: Vec<(f32, usize)> = (0..n_items)
                        .map(|j| (scores_vec[i * n_items + j], j))
                        .collect();

                    let mut sorted_scores = user_scores;
                    sorted_scores.sort_by(|a, b| {
                        b.0.partial_cmp(&a.0)
                            .expect("score values should be comparable")
                    });

                    for j in 0..k.min(sorted_scores.len()) {
                        unique_items.insert(sorted_scores[j].1);
                    }
                }

                unique_items.len() as f64 / self.catalog_size as f64
            }
            _ => 0.0,
        }
    }
}

/// Diversity metric for recommender systems
/// Measures how dissimilar the recommended items are from each other
pub struct Diversity {
    similarity_fn: fn(&[f64], &[f64]) -> f64,
}

impl Diversity {
    /// Create a new diversity metric with cosine similarity
    pub fn cosine() -> Self {
        Self {
            similarity_fn: cosine_similarity,
        }
    }

    /// Create a new diversity metric with Euclidean distance
    pub fn euclidean() -> Self {
        Self {
            similarity_fn: |a, b| {
                let dist = euclidean_distance(a, b);
                1.0 / (1.0 + dist) // Convert distance to similarity
            },
        }
    }

    /// Compute diversity of recommendation list
    pub fn compute_list_diversity(&self, item_features: &[Vec<f64>]) -> f64 {
        if item_features.len() < 2 {
            return 1.0; // Maximum diversity for single item or empty list
        }

        let n = item_features.len();
        let mut total_dissimilarity = 0.0;
        let mut pairs = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let similarity = (self.similarity_fn)(&item_features[i], &item_features[j]);
                total_dissimilarity += 1.0 - similarity.clamp(0.0, 1.0);
                pairs += 1;
            }
        }

        if pairs > 0 {
            total_dissimilarity / pairs as f64
        } else {
            1.0
        }
    }
}

// Helper functions for diversity computation
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Novelty metric for recommender systems
/// Measures how "novel" or unexpected the recommendations are
pub struct Novelty {
    item_popularity: HashMap<usize, f64>,
    total_interactions: f64,
}

impl Novelty {
    /// Create a new novelty metric
    pub fn new(item_popularity: HashMap<usize, f64>) -> Self {
        let total_interactions = item_popularity.values().sum();
        Self {
            item_popularity,
            total_interactions,
        }
    }

    /// Compute novelty of recommendation list
    pub fn compute_list_novelty(&self, recommended_items: &[usize]) -> f64 {
        if recommended_items.is_empty() {
            return 0.0;
        }

        let mut total_novelty = 0.0;

        for &item in recommended_items {
            let popularity = self.item_popularity.get(&item).copied().unwrap_or(0.0);
            let probability = if self.total_interactions > 0.0 {
                popularity / self.total_interactions
            } else {
                1.0 / self.item_popularity.len() as f64 // Uniform if no data
            };

            if probability > 1e-10 {
                total_novelty += -probability.log2();
            }
        }

        total_novelty / recommended_items.len() as f64
    }
}

/// Learning to Rank metrics collection
pub struct LearningToRankMetrics {
    ndcg: NDCG,
    map: MeanAveragePrecision,
    mrr: MeanReciprocalRank,
    precision_at_k: Vec<PrecisionAtK>,
    recall_at_k: Vec<RecallAtK>,
}

impl LearningToRankMetrics {
    /// Create a comprehensive LTR metrics collection
    pub fn new(k_values: &[usize]) -> Self {
        Self {
            ndcg: NDCG::new(),
            map: MeanAveragePrecision::new(),
            mrr: MeanReciprocalRank::new(),
            precision_at_k: k_values.iter().map(|&k| PrecisionAtK::new(k)).collect(),
            recall_at_k: k_values.iter().map(|&k| RecallAtK::new(k)).collect(),
        }
    }

    /// Compute all metrics
    pub fn compute_all(
        &self,
        true_relevance: &Tensor,
        predicted_scores: &Tensor,
    ) -> HashMap<String, f64> {
        let mut results = HashMap::new();

        results.insert(
            "ndcg".to_string(),
            self.ndcg.compute_score(true_relevance, predicted_scores),
        );
        results.insert(
            "map".to_string(),
            self.map.compute_score(true_relevance, predicted_scores),
        );
        results.insert(
            "mrr".to_string(),
            self.mrr.compute_score(true_relevance, predicted_scores),
        );

        for (_i, metric) in self.precision_at_k.iter().enumerate() {
            results.insert(
                format!("precision@{}", metric.k),
                metric.compute_score(true_relevance, predicted_scores),
            );
        }

        for (_i, metric) in self.recall_at_k.iter().enumerate() {
            results.insert(
                format!("recall@{}", metric.k),
                metric.compute_score(true_relevance, predicted_scores),
            );
        }

        results
    }
}

/// Information Retrieval Metrics Collection
/// Comprehensive IR evaluation metrics for search and recommendation systems
#[derive(Debug, Clone)]
pub struct IRMetrics {
    /// Precision at different k values
    pub precision_at_k: Vec<f64>,
    /// Recall at different k values
    pub recall_at_k: Vec<f64>,
    /// Average Precision
    pub average_precision: f64,
    /// Reciprocal Rank
    pub reciprocal_rank: f64,
    /// NDCG score
    pub ndcg: f64,
    /// F1 scores at different k values
    pub f1_at_k: Vec<f64>,
    /// K values used for metrics
    pub k_values: Vec<usize>,
}

impl IRMetrics {
    /// Compute comprehensive IR metrics
    pub fn compute(true_relevance: &Tensor, predicted_scores: &Tensor, k_values: &[usize]) -> Self {
        let mut precision_at_k = Vec::new();
        let mut recall_at_k = Vec::new();
        let mut f1_at_k = Vec::new();

        // Compute precision and recall at each k
        for &k in k_values {
            let p_metric = PrecisionAtK::new(k);
            let r_metric = RecallAtK::new(k);

            let precision = p_metric.compute_score(true_relevance, predicted_scores);
            let recall = r_metric.compute_score(true_relevance, predicted_scores);

            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            precision_at_k.push(precision);
            recall_at_k.push(recall);
            f1_at_k.push(f1);
        }

        // Compute AP and RR
        let map_metric = MeanAveragePrecision::new();
        let mrr_metric = MeanReciprocalRank::new();

        let average_precision = map_metric.compute_score(true_relevance, predicted_scores);
        let reciprocal_rank = mrr_metric.compute_score(true_relevance, predicted_scores);

        // Compute NDCG
        let ndcg_metric = NDCG::new();
        let ndcg = ndcg_metric.compute_score(true_relevance, predicted_scores);

        IRMetrics {
            precision_at_k,
            recall_at_k,
            average_precision,
            reciprocal_rank,
            ndcg,
            f1_at_k,
            k_values: k_values.to_vec(),
        }
    }

    /// Get precision at specific k
    pub fn precision_at(&self, k: usize) -> Option<f64> {
        self.k_values
            .iter()
            .position(|&x| x == k)
            .map(|idx| self.precision_at_k[idx])
    }

    /// Get recall at specific k
    pub fn recall_at(&self, k: usize) -> Option<f64> {
        self.k_values
            .iter()
            .position(|&x| x == k)
            .map(|idx| self.recall_at_k[idx])
    }

    /// Get F1 at specific k
    pub fn f1_at(&self, k: usize) -> Option<f64> {
        self.k_values
            .iter()
            .position(|&x| x == k)
            .map(|idx| self.f1_at_k[idx])
    }

    /// Format metrics as a string
    pub fn format(&self) -> String {
        let mut result = String::new();
        result.push_str("Information Retrieval Metrics:\n");
        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        result.push_str(&format!(
            "Average Precision (AP): {:.4}\n",
            self.average_precision
        ));
        result.push_str(&format!(
            "Reciprocal Rank (RR): {:.4}\n",
            self.reciprocal_rank
        ));
        result.push_str(&format!("NDCG: {:.4}\n", self.ndcg));

        result.push_str("\nMetrics at different k values:\n");
        for (i, &k) in self.k_values.iter().enumerate() {
            result.push_str(&format!(
                "  k={}: P={:.4}, R={:.4}, F1={:.4}\n",
                k, self.precision_at_k[i], self.recall_at_k[i], self.f1_at_k[i]
            ));
        }

        result
    }

    /// Get all metrics as a HashMap
    pub fn as_map(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();

        map.insert("average_precision".to_string(), self.average_precision);
        map.insert("reciprocal_rank".to_string(), self.reciprocal_rank);
        map.insert("ndcg".to_string(), self.ndcg);

        for (i, &k) in self.k_values.iter().enumerate() {
            map.insert(format!("precision@{}", k), self.precision_at_k[i]);
            map.insert(format!("recall@{}", k), self.recall_at_k[i]);
            map.insert(format!("f1@{}", k), self.f1_at_k[i]);
        }

        map
    }
}
