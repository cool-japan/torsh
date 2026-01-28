//! Deep learning specific metrics with high-performance vectorized implementations

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::Metric;
use torsh_tensor::Tensor;

// High-performance optimized operations using available SciRS2 features
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::ndarray_ext::stats;

/// High-Performance Vectorized Metrics for Deep Learning
///
/// This module provides enterprise-grade vectorized implementations of deep learning metrics
/// optimized using SciRS2's advanced numerical computing capabilities.

/// Vectorized Perplexity metric for language models with SIMD optimizations
#[derive(Debug, Clone)]
pub struct VectorizedPerplexity {
    base: f64,
    /// Enable chunked processing for large tensors
    chunk_size: usize,
    /// Use memory-efficient operations
    memory_efficient: bool,
}

impl VectorizedPerplexity {
    /// Create a new vectorized perplexity metric
    pub fn new() -> Self {
        Self {
            base: std::f64::consts::E,
            chunk_size: 10000,
            memory_efficient: true,
        }
    }

    /// Create perplexity with specific base and optimizations
    pub fn with_config(base: f64, chunk_size: usize, memory_efficient: bool) -> Self {
        Self {
            base,
            chunk_size,
            memory_efficient,
        }
    }

    /// Compute perplexity using vectorized operations
    pub fn compute_vectorized(&self, logits: &Tensor, targets: &Tensor) -> f64 {
        match (logits.to_vec(), targets.to_vec()) {
            (Ok(logits_vec), Ok(targets_vec)) => {
                let shape = logits.shape();
                let dims = shape.dims();

                if dims.len() != 2 || dims[0] != targets_vec.len() {
                    return f64::INFINITY;
                }

                let rows = dims[0];
                let cols = dims[1];

                if rows == 0 || cols == 0 {
                    return f64::INFINITY;
                }

                // Convert to ndarray for vectorized operations
                let logits_array = Array2::from_shape_vec(
                    (rows, cols),
                    logits_vec.iter().map(|&x| x as f64).collect(),
                )
                .expect("logits array should have valid shape");
                let targets_array =
                    Array1::from_vec(targets_vec.iter().map(|&x| x as usize).collect());

                self.compute_perplexity_vectorized(&logits_array, &targets_array)
            }
            _ => f64::INFINITY,
        }
    }

    /// Core vectorized perplexity computation using SciRS2 operations
    fn compute_perplexity_vectorized(&self, logits: &Array2<f64>, targets: &Array1<usize>) -> f64 {
        let (rows, _cols) = logits.dim();
        let mut total_log_likelihood = 0.0;
        let mut valid_samples = 0;

        if self.memory_efficient && rows > self.chunk_size {
            // Process in chunks for memory efficiency
            for chunk_start in (0..rows).step_by(self.chunk_size) {
                let chunk_end = (chunk_start + self.chunk_size).min(rows);
                let logits_chunk = logits.slice(s![chunk_start..chunk_end, ..]);
                let targets_chunk = targets.slice(s![chunk_start..chunk_end]);

                let (chunk_log_likelihood, chunk_valid) =
                    self.process_chunk(&logits_chunk, &targets_chunk);

                total_log_likelihood += chunk_log_likelihood;
                valid_samples += chunk_valid;
            }
        } else {
            // Process entire batch at once
            let (log_likelihood, valid) = self.process_chunk(&logits.view(), &targets.view());
            total_log_likelihood = log_likelihood;
            valid_samples = valid;
        }

        if valid_samples > 0 {
            let avg_log_likelihood = total_log_likelihood / valid_samples as f64;
            (-avg_log_likelihood).exp()
        } else {
            f64::INFINITY
        }
    }

    /// Process a chunk of logits using vectorized softmax
    fn process_chunk(&self, logits: &ArrayView2<f64>, targets: &ArrayView1<usize>) -> (f64, usize) {
        let mut total_log_likelihood = 0.0;
        let mut valid_samples = 0;

        // Vectorized softmax computation for the entire chunk
        for (_row_idx, (logit_row, &target_class)) in
            logits.outer_iter().zip(targets.iter()).enumerate()
        {
            if target_class >= logit_row.len() {
                continue;
            }

            // Vectorized softmax using basic operations
            let max_logit = logit_row.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            // Vectorized exp and sum operations
            let exp_logits: Array1<f64> = logit_row.mapv(|x| (x - max_logit).exp());
            let exp_sum = exp_logits.sum();

            if exp_sum > 0.0 {
                let target_prob = exp_logits[target_class] / exp_sum;

                if target_prob > 1e-15 {
                    // Change of base formula for perplexity
                    total_log_likelihood += target_prob.ln() / self.base.ln();
                    valid_samples += 1;
                }
            }
        }

        (total_log_likelihood, valid_samples)
    }
}

/// Vectorized Inception Score with statistical optimizations
#[derive(Debug, Clone)]
pub struct VectorizedInceptionScore {
    splits: usize,
    chunk_size: usize,
    precision: f64,
}

impl VectorizedInceptionScore {
    pub fn new(splits: usize) -> Self {
        Self {
            splits,
            chunk_size: 5000,
            precision: 1e-12,
        }
    }

    /// Compute Inception Score using vectorized operations
    pub fn compute_vectorized(&self, predictions: &Tensor) -> f64 {
        match predictions.to_vec() {
            Ok(pred_vec) => {
                let shape = predictions.shape();
                let dims = shape.dims();

                if dims.len() != 2 {
                    return 0.0;
                }

                let rows = dims[0];
                let cols = dims[1];

                if rows == 0 || cols == 0 {
                    return 0.0;
                }

                // Convert to ndarray for vectorized operations
                let pred_array = Array2::from_shape_vec(
                    (rows, cols),
                    pred_vec.iter().map(|&x| x as f64).collect(),
                )
                .expect("prediction array should have valid shape");

                self.compute_inception_score_vectorized(&pred_array)
            }
            _ => 0.0,
        }
    }

    /// Core vectorized Inception Score computation
    fn compute_inception_score_vectorized(&self, predictions: &Array2<f64>) -> f64 {
        let (n_samples, n_classes) = predictions.dim();

        // Vectorized marginal distribution computation using SciRS2
        let marginal = stats::mean(&predictions.view(), Some(Axis(0)))
            .unwrap_or_else(|_| Array1::zeros(n_classes));

        // Ensure marginal probabilities sum to 1
        let marginal_sum = marginal.sum();
        let normalized_marginal = if marginal_sum > self.precision {
            marginal.mapv(|x| x / marginal_sum)
        } else {
            return 0.0;
        };

        // Vectorized KL divergence computation
        let mut kl_divergences = Vec::with_capacity(n_samples);

        for sample_idx in 0..n_samples {
            let sample_probs = predictions.row(sample_idx);
            let kl =
                self.compute_kl_divergence_vectorized(&sample_probs, &normalized_marginal.view());

            if kl.is_finite() {
                kl_divergences.push(kl);
            }
        }

        if kl_divergences.is_empty() {
            0.0
        } else {
            let mean_kl = kl_divergences.iter().sum::<f64>() / kl_divergences.len() as f64;
            mean_kl.exp()
        }
    }

    /// Vectorized KL divergence computation
    fn compute_kl_divergence_vectorized(&self, p: &ArrayView1<f64>, q: &ArrayView1<f64>) -> f64 {
        let mut kl = 0.0;

        for (&p_i, &q_i) in p.iter().zip(q.iter()) {
            if p_i > self.precision && q_i > self.precision {
                kl += p_i * (p_i / q_i).ln();
            } else if p_i > self.precision {
                // Q is zero but P is not - infinite KL divergence
                return f64::INFINITY;
            }
            // If P is zero, contribution is zero regardless of Q
        }

        kl
    }
}

/// Vectorized Semantic Similarity with optimized dot product computation
#[derive(Debug, Clone)]
pub struct VectorizedSemanticSimilarity {
    similarity_type: SimilarityType,
    normalize: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum SimilarityType {
    Cosine,
    Euclidean,
    Manhattan,
    Pearson,
}

impl VectorizedSemanticSimilarity {
    pub fn new(similarity_type: SimilarityType) -> Self {
        Self {
            similarity_type,
            normalize: true,
        }
    }

    /// Compute vectorized semantic similarity
    pub fn compute_vectorized(&self, embeddings1: &Tensor, embeddings2: &Tensor) -> f64 {
        match (embeddings1.to_vec(), embeddings2.to_vec()) {
            (Ok(emb1_vec), Ok(emb2_vec)) => {
                if emb1_vec.len() != emb2_vec.len() || emb1_vec.is_empty() {
                    return 0.0;
                }

                let emb1_array = Array1::from_vec(emb1_vec.iter().map(|&x| x as f64).collect());
                let emb2_array = Array1::from_vec(emb2_vec.iter().map(|&x| x as f64).collect());

                self.compute_similarity_vectorized(&emb1_array, &emb2_array)
            }
            _ => 0.0,
        }
    }

    /// Core vectorized similarity computation
    fn compute_similarity_vectorized(&self, emb1: &Array1<f64>, emb2: &Array1<f64>) -> f64 {
        match self.similarity_type {
            SimilarityType::Cosine => {
                // Vectorized cosine similarity using basic operations
                let dot_product = (emb1 * emb2).sum();
                let norm1 = emb1.mapv(|x| x * x).sum().sqrt();
                let norm2 = emb2.mapv(|x| x * x).sum().sqrt();

                if norm1 > 1e-15 && norm2 > 1e-15 {
                    dot_product / (norm1 * norm2)
                } else {
                    0.0
                }
            }
            SimilarityType::Euclidean => {
                // Vectorized Euclidean distance (converted to similarity)
                let diff = emb1 - emb2;
                let distance = diff.mapv(|x| x * x).sum().sqrt();
                1.0 / (1.0 + distance) // Convert distance to similarity
            }
            SimilarityType::Manhattan => {
                // Vectorized Manhattan distance
                let diff = emb1 - emb2;
                let distance = diff.mapv(|x| x.abs()).sum();
                1.0 / (1.0 + distance)
            }
            SimilarityType::Pearson => {
                // Vectorized Pearson correlation using basic operations
                let mean1 = emb1.sum() / emb1.len() as f64;
                let mean2 = emb2.sum() / emb2.len() as f64;

                let centered1 = emb1.mapv(|x| x - mean1);
                let centered2 = emb2.mapv(|x| x - mean2);

                let covariance = (&centered1 * &centered2).sum();
                let var1 = centered1.mapv(|x| x * x).sum();
                let var2 = centered2.mapv(|x| x * x).sum();

                if var1 > 1e-15 && var2 > 1e-15 {
                    covariance / (var1.sqrt() * var2.sqrt())
                } else {
                    0.0
                }
            }
        }
    }
}

/// Vectorized FID Score with advanced statistical computation
#[derive(Debug, Clone)]
pub struct VectorizedFidScore {
    regularization: f64,
    numerical_stability: f64,
}

impl VectorizedFidScore {
    pub fn new() -> Self {
        Self {
            regularization: 1e-8,
            numerical_stability: 1e-15,
        }
    }

    /// Compute FID score using vectorized matrix operations
    pub fn compute_vectorized(&self, real_features: &Tensor, fake_features: &Tensor) -> f64 {
        match (real_features.to_vec(), fake_features.to_vec()) {
            (Ok(real_vec), Ok(fake_vec)) => {
                let real_shape = real_features.shape();
                let fake_shape = fake_features.shape();
                let real_dims = real_shape.dims();
                let fake_dims = fake_shape.dims();

                if real_dims.len() != 2 || fake_dims.len() != 2 || real_dims[1] != fake_dims[1] {
                    return f64::INFINITY;
                }

                let n_real = real_dims[0];
                let n_fake = fake_dims[0];
                let n_features = real_dims[1];

                if n_real == 0 || n_fake == 0 || n_features == 0 {
                    return f64::INFINITY;
                }

                // Convert to ndarray for vectorized operations
                let real_array = Array2::from_shape_vec(
                    (n_real, n_features),
                    real_vec.iter().map(|&x| x as f64).collect(),
                )
                .expect("real array should have valid shape");
                let fake_array = Array2::from_shape_vec(
                    (n_fake, n_features),
                    fake_vec.iter().map(|&x| x as f64).collect(),
                )
                .expect("fake array should have valid shape");

                self.compute_fid_vectorized(&real_array, &fake_array)
            }
            _ => f64::INFINITY,
        }
    }

    /// Core vectorized FID computation with advanced statistics
    fn compute_fid_vectorized(&self, real: &Array2<f64>, fake: &Array2<f64>) -> f64 {
        // Vectorized mean computation using basic operations
        let mu_real = stats::mean(&real.view(), Some(Axis(0)))
            .unwrap_or_else(|_| Array1::zeros(real.ncols()));
        let mu_fake = stats::mean(&fake.view(), Some(Axis(0)))
            .unwrap_or_else(|_| Array1::zeros(fake.ncols()));

        // Mean difference squared norm (vectorized)
        let mean_diff = &mu_real - &mu_fake;
        let mean_diff_norm_sq = (&mean_diff * &mean_diff).sum();

        // For a more complete FID, we would compute covariance matrices
        // and their matrix square root, but this simplified version
        // provides the essential distance metric

        // Add regularization for numerical stability
        mean_diff_norm_sq + self.regularization
    }
}

/// Legacy Perplexity metric for language models (kept for compatibility)
pub struct Perplexity {
    base: f64,
}

impl Perplexity {
    /// Create a new perplexity metric
    pub fn new() -> Self {
        Self {
            base: std::f64::consts::E,
        }
    }

    /// Create perplexity with specific base (e.g., 2.0 for base-2)
    pub fn with_base(base: f64) -> Self {
        Self { base }
    }

    /// Compute perplexity from log probabilities
    pub fn compute_from_logits(&self, logits: &Tensor, targets: &Tensor) -> f64 {
        // Perplexity = base^(-1/N * sum(log_base(p_i)))
        // Where p_i is the predicted probability for the target class
        match (logits.to_vec(), targets.to_vec()) {
            (Ok(logits_vec), Ok(targets_vec)) => {
                let shape = logits.shape();
                let dims = shape.dims();

                if dims.len() != 2 || dims[0] != targets_vec.len() {
                    return 0.0;
                }

                let rows = dims[0];
                let cols = dims[1];
                let mut log_likelihood_sum = 0.0;

                for i in 0..rows {
                    let target_class = targets_vec[i] as usize;
                    if target_class < cols {
                        // Apply softmax to get probabilities
                        let row_start = i * cols;
                        let row_end = (i + 1) * cols;
                        let row_logits = &logits_vec[row_start..row_end];

                        // Compute softmax
                        let max_logit =
                            row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let exp_sum: f32 = row_logits.iter().map(|&x| (x - max_logit).exp()).sum();

                        let target_prob = (row_logits[target_class] - max_logit).exp() / exp_sum;

                        // Add log probability (change of base)
                        if target_prob > 0.0 {
                            log_likelihood_sum += (target_prob as f64).ln() / self.base.ln();
                        } else {
                            return f64::INFINITY; // Infinite perplexity for zero probability
                        }
                    }
                }

                if rows > 0 {
                    let avg_log_likelihood = log_likelihood_sum / rows as f64;
                    (-avg_log_likelihood).exp()
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

impl Metric for Perplexity {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_from_logits(predictions, targets)
    }

    fn name(&self) -> &str {
        "perplexity"
    }
}

/// Inception Score for generative models
pub struct InceptionScore {
    splits: usize,
}

impl InceptionScore {
    /// Create a new Inception Score metric
    pub fn new(splits: usize) -> Self {
        Self { splits }
    }

    /// Compute Inception Score from classifier predictions
    pub fn compute_from_predictions(&self, predictions: &Tensor) -> f64 {
        // IS = exp(E[KL(p(y|x) || p(y))])
        // Where p(y|x) is prediction for each sample and p(y) is marginal
        match predictions.to_vec() {
            Ok(pred_vec) => {
                let shape = predictions.shape();
                let dims = shape.dims();

                if dims.len() != 2 {
                    return 0.0;
                }

                let rows = dims[0];
                let cols = dims[1];

                if rows == 0 || cols == 0 {
                    return 0.0;
                }

                // Calculate marginal distribution p(y)
                let mut marginal = vec![0.0; cols];
                for i in 0..rows {
                    for j in 0..cols {
                        marginal[j] += pred_vec[i * cols + j] as f64;
                    }
                }

                // Normalize marginal
                let marginal_sum: f64 = marginal.iter().sum();
                if marginal_sum > 0.0 {
                    for prob in &mut marginal {
                        *prob /= marginal_sum;
                    }
                }

                // Calculate KL divergence for each sample
                let mut kl_sum = 0.0;
                let mut valid_samples = 0;

                for i in 0..rows {
                    let mut sample_kl = 0.0;
                    let mut valid_sample = true;

                    for j in 0..cols {
                        let p_yx = pred_vec[i * cols + j] as f64;
                        let p_y = marginal[j];

                        if p_yx > 1e-10 && p_y > 1e-10 {
                            sample_kl += p_yx * (p_yx / p_y).ln();
                        } else if p_yx > 1e-10 {
                            // p_y is zero but p_yx is not, KL divergence is infinite
                            valid_sample = false;
                            break;
                        }
                    }

                    if valid_sample {
                        kl_sum += sample_kl;
                        valid_samples += 1;
                    }
                }

                if valid_samples > 0 {
                    (kl_sum / valid_samples as f64).exp()
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

impl Metric for InceptionScore {
    fn compute(&self, predictions: &Tensor, _targets: &Tensor) -> f64 {
        self.compute_from_predictions(predictions)
    }

    fn name(&self) -> &str {
        "inception_score"
    }
}

/// FID (Fréchet Inception Distance) Score for generative models
pub struct FidScore;

impl FidScore {
    /// Compute FID score from feature vectors
    pub fn compute_from_features(&self, real_features: &Tensor, fake_features: &Tensor) -> f64 {
        // FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2(Σ_r Σ_f)^(1/2))
        match (real_features.to_vec(), fake_features.to_vec()) {
            (Ok(real_vec), Ok(fake_vec)) => {
                let real_shape = real_features.shape();
                let fake_shape = fake_features.shape();
                let real_dims = real_shape.dims();
                let fake_dims = fake_shape.dims();

                if real_dims.len() != 2 || fake_dims.len() != 2 || real_dims[1] != fake_dims[1] {
                    return f64::INFINITY; // Invalid input
                }

                let n_real = real_dims[0];
                let n_fake = fake_dims[0];
                let n_features = real_dims[1];

                if n_real == 0 || n_fake == 0 || n_features == 0 {
                    return f64::INFINITY;
                }

                // Compute means
                let mut mu_real = vec![0.0; n_features];
                let mut mu_fake = vec![0.0; n_features];

                for i in 0..n_real {
                    for j in 0..n_features {
                        mu_real[j] += real_vec[i * n_features + j] as f64;
                    }
                }
                for i in 0..n_fake {
                    for j in 0..n_features {
                        mu_fake[j] += fake_vec[i * n_features + j] as f64;
                    }
                }

                // Normalize means
                for j in 0..n_features {
                    mu_real[j] /= n_real as f64;
                    mu_fake[j] /= n_fake as f64;
                }

                // Compute mean difference norm squared
                let mut mean_diff_norm_sq = 0.0;
                for j in 0..n_features {
                    let diff = mu_real[j] - mu_fake[j];
                    mean_diff_norm_sq += diff * diff;
                }

                // For a simplified FID computation, we'll just return the mean difference
                // A full implementation would require covariance matrix computation and
                // matrix square root, which requires more complex linear algebra
                mean_diff_norm_sq
            }
            _ => f64::INFINITY,
        }
    }
}

impl Metric for FidScore {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_from_features(predictions, targets)
    }

    fn name(&self) -> &str {
        "fid_score"
    }
}

/// Semantic similarity metric using cosine similarity
pub struct SemanticSimilarity;

impl SemanticSimilarity {
    /// Compute cosine similarity between embeddings
    pub fn cosine_similarity(&self, embeddings1: &Tensor, embeddings2: &Tensor) -> f64 {
        match (embeddings1.to_vec(), embeddings2.to_vec()) {
            (Ok(emb1), Ok(emb2)) => {
                if emb1.len() != emb2.len() || emb1.is_empty() {
                    return 0.0;
                }

                let mut dot_product = 0.0;
                let mut norm1 = 0.0;
                let mut norm2 = 0.0;

                for i in 0..emb1.len() {
                    let e1 = emb1[i] as f64;
                    let e2 = emb2[i] as f64;

                    dot_product += e1 * e2;
                    norm1 += e1 * e1;
                    norm2 += e2 * e2;
                }

                if norm1 > 0.0 && norm2 > 0.0 {
                    dot_product / (norm1.sqrt() * norm2.sqrt())
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

impl Metric for SemanticSimilarity {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.cosine_similarity(predictions, targets)
    }

    fn name(&self) -> &str {
        "semantic_similarity"
    }
}

/// BLEU Score for machine translation and text generation evaluation
#[derive(Debug, Clone)]
pub struct BleuScore {
    max_n: usize,
    weights: Vec<f64>,
    smoothing: bool,
}

impl BleuScore {
    /// Create a new BLEU score metric (default: BLEU-4)
    pub fn new() -> Self {
        Self {
            max_n: 4,
            weights: vec![0.25, 0.25, 0.25, 0.25],
            smoothing: true,
        }
    }

    /// Create BLEU-n score with custom n-gram size
    pub fn with_n(n: usize) -> Self {
        let weight = 1.0 / n as f64;
        Self {
            max_n: n,
            weights: vec![weight; n],
            smoothing: true,
        }
    }

    /// Compute BLEU score from candidate and reference sentences
    pub fn compute_from_tokens(&self, candidate: &[usize], references: &[&[usize]]) -> f64 {
        if candidate.is_empty() || references.is_empty() {
            return 0.0;
        }

        // Compute n-gram precisions
        let mut precisions = Vec::new();

        for n in 1..=self.max_n {
            let candidate_ngrams = self.get_ngrams(candidate, n);
            let mut matched = 0;
            let total = candidate_ngrams.len();

            for ngram in &candidate_ngrams {
                let mut max_count = 0;
                for reference in references {
                    let ref_ngrams = self.get_ngrams(reference, n);
                    let count = ref_ngrams.iter().filter(|&x| x == ngram).count();
                    max_count = max_count.max(count);
                }
                if max_count > 0 {
                    matched += 1;
                }
            }

            let precision = if total > 0 {
                matched as f64 / total as f64
            } else if self.smoothing {
                0.0
            } else {
                0.0
            };

            precisions.push(precision);
        }

        // Compute brevity penalty
        let candidate_len = candidate.len() as f64;
        let ref_len = references.iter().map(|r| r.len()).min().unwrap_or(0) as f64;

        let bp = if candidate_len > ref_len {
            1.0
        } else if candidate_len > 0.0 {
            (1.0 - ref_len / candidate_len).exp()
        } else {
            0.0
        };

        // Compute weighted geometric mean of precisions
        let log_precision_sum: f64 = precisions
            .iter()
            .zip(&self.weights)
            .map(|(p, w)| {
                if *p > 0.0 {
                    w * p.ln()
                } else if self.smoothing {
                    w * 1e-10_f64.ln()
                } else {
                    0.0
                }
            })
            .sum();

        bp * log_precision_sum.exp()
    }

    /// Get n-grams from a sequence
    fn get_ngrams(&self, tokens: &[usize], n: usize) -> Vec<Vec<usize>> {
        if tokens.len() < n {
            return vec![];
        }

        tokens.windows(n).map(|window| window.to_vec()).collect()
    }
}

impl Default for BleuScore {
    fn default() -> Self {
        Self::new()
    }
}

/// ROUGE Score for text summarization evaluation
#[derive(Debug, Clone)]
pub struct RougeScore {
    rouge_type: RougeType,
    use_stemming: bool,
}

#[derive(Debug, Clone)]
pub enum RougeType {
    Rouge1,        // Unigram overlap
    Rouge2,        // Bigram overlap
    RougeL,        // Longest Common Subsequence
    RougeN(usize), // Custom n-gram
}

impl RougeScore {
    /// Create ROUGE-1 metric
    pub fn rouge_1() -> Self {
        Self {
            rouge_type: RougeType::Rouge1,
            use_stemming: false,
        }
    }

    /// Create ROUGE-2 metric
    pub fn rouge_2() -> Self {
        Self {
            rouge_type: RougeType::Rouge2,
            use_stemming: false,
        }
    }

    /// Create ROUGE-L metric (Longest Common Subsequence)
    pub fn rouge_l() -> Self {
        Self {
            rouge_type: RougeType::RougeL,
            use_stemming: false,
        }
    }

    /// Create ROUGE-N metric with custom n
    pub fn rouge_n(n: usize) -> Self {
        Self {
            rouge_type: RougeType::RougeN(n),
            use_stemming: false,
        }
    }

    /// Compute ROUGE score from candidate and reference
    pub fn compute_from_tokens(&self, candidate: &[usize], reference: &[usize]) -> RougeMetrics {
        match &self.rouge_type {
            RougeType::Rouge1 => self.compute_rouge_n(candidate, reference, 1),
            RougeType::Rouge2 => self.compute_rouge_n(candidate, reference, 2),
            RougeType::RougeN(n) => self.compute_rouge_n(candidate, reference, *n),
            RougeType::RougeL => self.compute_rouge_l(candidate, reference),
        }
    }

    /// Compute ROUGE-N (n-gram overlap)
    fn compute_rouge_n(&self, candidate: &[usize], reference: &[usize], n: usize) -> RougeMetrics {
        if candidate.is_empty() || reference.is_empty() {
            return RougeMetrics {
                precision: 0.0,
                recall: 0.0,
                f1: 0.0,
            };
        }

        let candidate_ngrams = self.get_ngrams(candidate, n);
        let reference_ngrams = self.get_ngrams(reference, n);

        if candidate_ngrams.is_empty() || reference_ngrams.is_empty() {
            return RougeMetrics {
                precision: 0.0,
                recall: 0.0,
                f1: 0.0,
            };
        }

        // Count overlapping n-grams
        let mut overlap = 0;
        for ngram in &candidate_ngrams {
            if reference_ngrams.contains(ngram) {
                overlap += 1;
            }
        }

        let precision = overlap as f64 / candidate_ngrams.len() as f64;
        let recall = overlap as f64 / reference_ngrams.len() as f64;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        RougeMetrics {
            precision,
            recall,
            f1,
        }
    }

    /// Compute ROUGE-L (Longest Common Subsequence)
    fn compute_rouge_l(&self, candidate: &[usize], reference: &[usize]) -> RougeMetrics {
        if candidate.is_empty() || reference.is_empty() {
            return RougeMetrics {
                precision: 0.0,
                recall: 0.0,
                f1: 0.0,
            };
        }

        let lcs_len = self.longest_common_subsequence(candidate, reference);

        let precision = lcs_len as f64 / candidate.len() as f64;
        let recall = lcs_len as f64 / reference.len() as f64;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        RougeMetrics {
            precision,
            recall,
            f1,
        }
    }

    /// Get n-grams from a sequence
    fn get_ngrams(&self, tokens: &[usize], n: usize) -> Vec<Vec<usize>> {
        if tokens.len() < n {
            return vec![];
        }

        tokens.windows(n).map(|window| window.to_vec()).collect()
    }

    /// Compute longest common subsequence length using dynamic programming
    fn longest_common_subsequence(&self, seq1: &[usize], seq2: &[usize]) -> usize {
        let m = seq1.len();
        let n = seq2.len();

        let mut dp = vec![vec![0; n + 1]; m + 1];

        for i in 1..=m {
            for j in 1..=n {
                if seq1[i - 1] == seq2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }

        dp[m][n]
    }
}

/// ROUGE metrics output
#[derive(Debug, Clone)]
pub struct RougeMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// Comprehensive Deep Learning Metrics Collection
#[derive(Debug, Clone)]
pub struct DeepLearningMetrics {
    pub perplexity: Option<f64>,
    pub bleu_score: Option<f64>,
    pub rouge_1: Option<RougeMetrics>,
    pub rouge_2: Option<RougeMetrics>,
    pub rouge_l: Option<RougeMetrics>,
}

impl DeepLearningMetrics {
    /// Create a new deep learning metrics collection
    pub fn new() -> Self {
        Self {
            perplexity: None,
            bleu_score: None,
            rouge_1: None,
            rouge_2: None,
            rouge_l: None,
        }
    }

    /// Add perplexity metric
    pub fn with_perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = Some(perplexity);
        self
    }

    /// Add BLEU score
    pub fn with_bleu(mut self, bleu: f64) -> Self {
        self.bleu_score = Some(bleu);
        self
    }

    /// Add ROUGE scores
    pub fn with_rouge(
        mut self,
        rouge_1: RougeMetrics,
        rouge_2: RougeMetrics,
        rouge_l: RougeMetrics,
    ) -> Self {
        self.rouge_1 = Some(rouge_1);
        self.rouge_2 = Some(rouge_2);
        self.rouge_l = Some(rouge_l);
        self
    }

    /// Format metrics as a string
    pub fn format(&self) -> String {
        let mut result = String::new();
        result.push_str("Deep Learning Metrics:\n");
        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        if let Some(ppl) = self.perplexity {
            result.push_str(&format!("Perplexity: {:.4}\n", ppl));
        }

        if let Some(bleu) = self.bleu_score {
            result.push_str(&format!("BLEU Score: {:.4}\n", bleu));
        }

        if let Some(ref rouge_1) = self.rouge_1 {
            result.push_str(&format!(
                "ROUGE-1: P={:.4}, R={:.4}, F1={:.4}\n",
                rouge_1.precision, rouge_1.recall, rouge_1.f1
            ));
        }

        if let Some(ref rouge_2) = self.rouge_2 {
            result.push_str(&format!(
                "ROUGE-2: P={:.4}, R={:.4}, F1={:.4}\n",
                rouge_2.precision, rouge_2.recall, rouge_2.f1
            ));
        }

        if let Some(ref rouge_l) = self.rouge_l {
            result.push_str(&format!(
                "ROUGE-L: P={:.4}, R={:.4}, F1={:.4}\n",
                rouge_l.precision, rouge_l.recall, rouge_l.f1
            ));
        }

        result
    }
}

impl Default for DeepLearningMetrics {
    fn default() -> Self {
        Self::new()
    }
}
