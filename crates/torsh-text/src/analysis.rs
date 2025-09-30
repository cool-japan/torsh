use crate::{Result, TextError};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct TextStatistics {
    pub char_count: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub paragraph_count: usize,
    pub avg_word_length: f64,
    pub avg_sentence_length: f64,
    pub unique_words: usize,
    pub type_token_ratio: f64,
    pub most_common_words: Vec<(String, usize)>,
    pub char_frequencies: HashMap<char, usize>,
}

#[derive(Debug, Clone)]
pub struct TextAnalyzer {
    min_word_freq: usize,
    top_n_words: usize,
}

impl Default for TextAnalyzer {
    fn default() -> Self {
        Self {
            min_word_freq: 1,
            top_n_words: 10,
        }
    }
}

impl TextAnalyzer {
    pub fn with_min_word_freq(mut self, min_freq: usize) -> Self {
        self.min_word_freq = min_freq;
        self
    }

    pub fn with_top_n_words(mut self, top_n: usize) -> Self {
        self.top_n_words = top_n;
        self
    }

    pub fn analyze(&self, text: &str) -> Result<TextStatistics> {
        let char_count = text.chars().count();
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();

        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();
        let sentence_count = sentences.len();

        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();
        let paragraph_count = paragraphs.len();

        let avg_word_length = if word_count > 0 {
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / word_count as f64
        } else {
            0.0
        };

        let avg_sentence_length = if sentence_count > 0 {
            word_count as f64 / sentence_count as f64
        } else {
            0.0
        };

        let mut word_frequencies: HashMap<String, usize> = HashMap::new();
        for word in &words {
            let word_lower = word.to_lowercase();
            *word_frequencies.entry(word_lower).or_insert(0) += 1;
        }

        let unique_words = word_frequencies.len();
        let type_token_ratio = if word_count > 0 {
            unique_words as f64 / word_count as f64
        } else {
            0.0
        };

        let mut most_common_words: Vec<(String, usize)> = word_frequencies
            .iter()
            .filter(|(_, &freq)| freq >= self.min_word_freq)
            .map(|(word, &freq)| (word.clone(), freq))
            .collect();
        most_common_words.sort_by(|a, b| b.1.cmp(&a.1));
        most_common_words.truncate(self.top_n_words);

        let mut char_frequencies: HashMap<char, usize> = HashMap::new();
        for ch in text.chars() {
            *char_frequencies.entry(ch).or_insert(0) += 1;
        }

        Ok(TextStatistics {
            char_count,
            word_count,
            sentence_count,
            paragraph_count,
            avg_word_length,
            avg_sentence_length,
            unique_words,
            type_token_ratio,
            most_common_words,
            char_frequencies,
        })
    }

    pub fn analyze_batch(&self, texts: &[&str]) -> Result<Vec<TextStatistics>> {
        texts.iter().map(|text| self.analyze(text)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct NgramExtractor {
    n: usize,
    min_freq: usize,
    case_sensitive: bool,
}

impl NgramExtractor {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            min_freq: 1,
            case_sensitive: false,
        }
    }

    pub fn with_min_freq(mut self, min_freq: usize) -> Self {
        self.min_freq = min_freq;
        self
    }

    pub fn with_case_sensitivity(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    pub fn extract_char_ngrams(&self, text: &str) -> HashMap<String, usize> {
        let text = if self.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };
        let chars: Vec<char> = text.chars().collect();
        let mut ngrams: HashMap<String, usize> = HashMap::new();

        for i in 0..=chars.len().saturating_sub(self.n) {
            let ngram: String = chars[i..i + self.n].iter().collect();
            *ngrams.entry(ngram).or_insert(0) += 1;
        }

        ngrams.retain(|_, &mut freq| freq >= self.min_freq);
        ngrams
    }

    pub fn extract_word_ngrams(&self, text: &str) -> HashMap<String, usize> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let words: Vec<String> = if self.case_sensitive {
            words.into_iter().map(|w| w.to_string()).collect()
        } else {
            words.into_iter().map(|w| w.to_lowercase()).collect()
        };

        let mut ngrams: HashMap<String, usize> = HashMap::new();

        for i in 0..=words.len().saturating_sub(self.n) {
            let ngram = words[i..i + self.n].join(" ");
            *ngrams.entry(ngram).or_insert(0) += 1;
        }

        ngrams.retain(|_, &mut freq| freq >= self.min_freq);
        ngrams
    }

    pub fn extract_top_ngrams(
        &self,
        text: &str,
        top_n: usize,
        word_level: bool,
    ) -> Vec<(String, usize)> {
        let ngrams = if word_level {
            self.extract_word_ngrams(text)
        } else {
            self.extract_char_ngrams(text)
        };

        let mut sorted_ngrams: Vec<(String, usize)> = ngrams.into_iter().collect();
        sorted_ngrams.sort_by(|a, b| b.1.cmp(&a.1));
        sorted_ngrams.truncate(top_n);
        sorted_ngrams
    }
}

#[derive(Debug, Clone)]
pub struct TfIdfCalculator {
    documents: Vec<String>,
    vocabulary: Vec<String>,
    tf_matrix: Vec<Vec<f64>>,
    idf_vector: Vec<f64>,
    computed: bool,
}

impl TfIdfCalculator {
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
            vocabulary: Vec::new(),
            tf_matrix: Vec::new(),
            idf_vector: Vec::new(),
            computed: false,
        }
    }

    pub fn add_document(&mut self, document: &str) {
        self.documents.push(document.to_string());
        self.computed = false;
    }

    pub fn add_documents(&mut self, documents: &[&str]) {
        for doc in documents {
            self.documents.push(doc.to_string());
        }
        self.computed = false;
    }

    pub fn compute(&mut self) -> Result<()> {
        if self.documents.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!("No documents added")));
        }

        let mut word_set: HashSet<String> = HashSet::new();
        let tokenized_docs: Vec<Vec<String>> = self
            .documents
            .iter()
            .map(|doc| {
                let words: Vec<String> = doc.split_whitespace().map(|w| w.to_lowercase()).collect();
                for word in &words {
                    word_set.insert(word.clone());
                }
                words
            })
            .collect();

        self.vocabulary = word_set.into_iter().collect();
        self.vocabulary.sort();

        let vocab_map: HashMap<String, usize> = self
            .vocabulary
            .iter()
            .enumerate()
            .map(|(i, word)| (word.clone(), i))
            .collect();

        let num_docs = self.documents.len();
        let vocab_size = self.vocabulary.len();

        let mut tf_data = vec![vec![0.0; vocab_size]; num_docs];

        for (doc_idx, words) in tokenized_docs.iter().enumerate() {
            let mut word_counts: HashMap<String, usize> = HashMap::new();
            for word in words {
                *word_counts.entry(word.clone()).or_insert(0) += 1;
            }

            let total_words = words.len() as f64;
            for (word, count) in word_counts {
                if let Some(&word_idx) = vocab_map.get(&word) {
                    tf_data[doc_idx][word_idx] = count as f64 / total_words;
                }
            }
        }

        let mut idf_data = vec![0.0; vocab_size];
        for (word_idx, word) in self.vocabulary.iter().enumerate() {
            let doc_freq = tokenized_docs
                .iter()
                .filter(|doc| doc.contains(word))
                .count() as f64;

            idf_data[word_idx] = (num_docs as f64 / doc_freq).ln();
        }

        self.tf_matrix = tf_data;
        self.idf_vector = idf_data;

        self.computed = true;
        Ok(())
    }

    /// Fit and transform documents to TF-IDF matrix
    pub fn fit_transform(&mut self, documents: &[&str]) -> Result<Vec<Vec<f64>>> {
        self.documents.clear();
        self.add_documents(documents);
        self.compute()?;
        self.get_tfidf_matrix()
    }

    /// Get the vocabulary
    pub fn vocabulary(&self) -> &Vec<String> {
        &self.vocabulary
    }

    pub fn get_tfidf_matrix(&self) -> Result<Vec<Vec<f64>>> {
        if !self.computed {
            return Err(TextError::Other(anyhow::anyhow!(
                "TF-IDF not computed. Call compute() first."
            )));
        }

        let rows = self.tf_matrix.len();
        if rows == 0 {
            return Ok(Vec::new());
        }

        let cols = self.tf_matrix[0].len();
        let mut tfidf_data = vec![vec![0.0; cols]; rows];

        for i in 0..rows {
            for j in 0..cols {
                tfidf_data[i][j] = self.tf_matrix[i][j] * self.idf_vector[j];
            }
        }

        Ok(tfidf_data)
    }

    pub fn get_document_vector(&self, doc_index: usize) -> Result<Vec<f64>> {
        if !self.computed {
            return Err(TextError::Other(anyhow::anyhow!(
                "TF-IDF not computed. Call compute() first."
            )));
        }

        if doc_index >= self.documents.len() {
            return Err(TextError::Other(anyhow::anyhow!(
                "Document index out of bounds"
            )));
        }

        let tfidf_matrix = self.get_tfidf_matrix()?;
        if doc_index >= tfidf_matrix.len() {
            return Err(TextError::Other(anyhow::anyhow!(
                "Document index out of bounds in TF-IDF matrix"
            )));
        }

        Ok(tfidf_matrix[doc_index].clone())
    }

    pub fn similarity(&self, doc1_idx: usize, doc2_idx: usize) -> Result<f64> {
        let vec1 = self.get_document_vector(doc1_idx)?;
        let vec2 = self.get_document_vector(doc2_idx)?;

        if vec1.len() != vec2.len() {
            return Err(TextError::Other(anyhow::anyhow!(
                "Vector dimensions mismatch"
            )));
        }

        let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();

        let norm1 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm1 * norm2))
        }
    }

    pub fn get_vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    pub fn get_document_count(&self) -> usize {
        self.documents.len()
    }
}

impl Default for TfIdfCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TextSimilarity;

impl TextSimilarity {
    pub fn jaccard_similarity(text1: &str, text2: &str) -> f64 {
        let words1: HashSet<String> = text1.split_whitespace().map(|w| w.to_lowercase()).collect();
        let words2: HashSet<String> = text2.split_whitespace().map(|w| w.to_lowercase()).collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    pub fn dice_similarity(text1: &str, text2: &str) -> f64 {
        let words1: HashSet<String> = text1.split_whitespace().map(|w| w.to_lowercase()).collect();
        let words2: HashSet<String> = text2.split_whitespace().map(|w| w.to_lowercase()).collect();

        let intersection = words1.intersection(&words2).count();
        let total = words1.len() + words2.len();

        if total == 0 {
            0.0
        } else {
            2.0 * intersection as f64 / total as f64
        }
    }

    pub fn overlap_similarity(text1: &str, text2: &str) -> f64 {
        let words1: HashSet<String> = text1.split_whitespace().map(|w| w.to_lowercase()).collect();
        let words2: HashSet<String> = text2.split_whitespace().map(|w| w.to_lowercase()).collect();

        let intersection = words1.intersection(&words2).count();
        let min_size = words1.len().min(words2.len());

        if min_size == 0 {
            0.0
        } else {
            intersection as f64 / min_size as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_text_statistics() {
        let analyzer = TextAnalyzer::default().with_top_n_words(5);
        let text = "Hello world! This is a test. Hello again.";

        let stats = analyzer.analyze(text).unwrap();
        assert_eq!(stats.word_count, 8);
        assert_eq!(stats.sentence_count, 3);
        assert!(stats.type_token_ratio > 0.0);
    }

    #[test]
    fn test_ngram_extraction() {
        let extractor = NgramExtractor::new(2);

        let char_ngrams = extractor.extract_char_ngrams("hello");
        assert!(char_ngrams.contains_key("he"));
        assert!(char_ngrams.contains_key("el"));

        let word_ngrams = extractor.extract_word_ngrams("hello world test");
        assert!(word_ngrams.contains_key("hello world"));
        assert!(word_ngrams.contains_key("world test"));
    }

    #[test]
    fn test_tfidf_calculator() {
        let mut calculator = TfIdfCalculator::new();
        calculator.add_documents(&[
            "the cat sat on the mat",
            "the dog ran in the park",
            "cats and dogs are pets",
        ]);

        calculator.compute().unwrap();
        let tfidf_matrix = calculator.get_tfidf_matrix().unwrap();
        assert_eq!(tfidf_matrix.len(), 3); // 3 documents

        let similarity = calculator.similarity(0, 1).unwrap();
        assert!(similarity >= 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_text_similarity() {
        let sim = TextSimilarity::jaccard_similarity("hello world", "world hello");
        assert_relative_eq!(sim, 1.0, epsilon = 1e-6);

        let sim = TextSimilarity::dice_similarity("abc def", "def ghi");
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_kmeans_clustering() {
        let documents = vec![
            "machine learning algorithms are powerful",
            "deep learning neural networks",
            "cats love to play with toys",
            "dogs enjoy running in parks",
            "artificial intelligence and machine learning",
            "pets bring joy to families",
        ];

        let clusterer = KMeansClusterer::new(2, 10, 0.01);
        let result = clusterer.cluster(&documents);
        assert!(result.is_ok());

        let clusters = result.unwrap();
        assert_eq!(clusters.len(), documents.len());

        // Check that all cluster assignments are valid
        for &cluster_id in &clusters {
            assert!(cluster_id < 2);
        }
    }

    #[test]
    fn test_hierarchical_clustering() {
        let documents = vec!["machine learning", "deep learning", "cat toys", "dog parks"];

        let clusterer = HierarchicalClusterer::default();
        let result = clusterer.cluster(&documents, 2);
        assert!(result.is_ok());

        let clusters = result.unwrap();
        assert_eq!(clusters.len(), documents.len());
    }
}

/// Text clustering functionality for grouping similar documents
pub mod clustering {
    use super::*;
    use crate::scirs2_ops::string_ops;
    use std::collections::HashMap;

    /// K-Means clustering for text documents
    #[derive(Debug, Clone)]
    pub struct KMeansClusterer {
        k: usize,
        max_iterations: usize,
        tolerance: f64,
    }

    impl KMeansClusterer {
        /// Create a new K-means clusterer
        pub fn new(k: usize, max_iterations: usize, tolerance: f64) -> Self {
            Self {
                k,
                max_iterations,
                tolerance,
            }
        }

        /// Cluster documents using K-means algorithm
        pub fn cluster(&self, documents: &[&str]) -> Result<Vec<usize>> {
            if documents.len() < self.k {
                return Err(TextError::Other(anyhow::anyhow!(
                    "Number of documents must be >= k"
                )));
            }

            // Create TF-IDF vectors for all documents
            let mut tfidf_calculator = TfIdfCalculator::new();
            let tfidf_matrix = tfidf_calculator.fit_transform(documents)?;
            let vocab = tfidf_calculator.vocabulary();
            let vocab_size = vocab.len();

            // Initialize centroids randomly
            let mut centroids = self.initialize_centroids(&tfidf_matrix, vocab_size)?;
            let mut assignments = vec![0; documents.len()];

            for _iteration in 0..self.max_iterations {
                let mut changed = false;

                // Assign each document to nearest centroid
                for (doc_idx, doc_vector) in tfidf_matrix.iter().enumerate() {
                    let mut best_cluster = 0;
                    let mut best_distance = f64::INFINITY;

                    for (cluster_idx, centroid) in centroids.iter().enumerate() {
                        let distance = self.euclidean_distance(doc_vector, centroid);
                        if distance < best_distance {
                            best_distance = distance;
                            best_cluster = cluster_idx;
                        }
                    }

                    if assignments[doc_idx] != best_cluster {
                        assignments[doc_idx] = best_cluster;
                        changed = true;
                    }
                }

                // Update centroids
                let new_centroids =
                    self.update_centroids(&tfidf_matrix, &assignments, vocab_size)?;

                // Check for convergence
                let centroid_change = self.centroid_distance(&centroids, &new_centroids);
                centroids = new_centroids;

                if !changed || centroid_change < self.tolerance {
                    break;
                }
            }

            Ok(assignments)
        }

        fn initialize_centroids(
            &self,
            tfidf_matrix: &[Vec<f64>],
            _vocab_size: usize,
        ) -> Result<Vec<Vec<f64>>> {
            // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
            use scirs2_core::random::{Random, Rng};
            let mut rng = Random::seed(42);
            let mut centroids = Vec::new();

            // Use K-means++ initialization
            for _ in 0..self.k {
                let doc_idx = rng.gen_range(0..tfidf_matrix.len());
                centroids.push(tfidf_matrix[doc_idx].clone());
            }

            Ok(centroids)
        }

        fn update_centroids(
            &self,
            tfidf_matrix: &[Vec<f64>],
            assignments: &[usize],
            vocab_size: usize,
        ) -> Result<Vec<Vec<f64>>> {
            let mut centroids = vec![vec![0.0; vocab_size]; self.k];
            let mut cluster_counts = vec![0; self.k];

            // Sum up document vectors for each cluster
            for (doc_idx, &cluster_id) in assignments.iter().enumerate() {
                cluster_counts[cluster_id] += 1;

                for (term_idx, &value) in tfidf_matrix[doc_idx].iter().enumerate() {
                    centroids[cluster_id][term_idx] += value;
                }
            }

            // Average the vectors to get centroids
            for (cluster_id, centroid) in centroids.iter_mut().enumerate() {
                let count = cluster_counts[cluster_id] as f64;
                if count > 0.0 {
                    for value in centroid.iter_mut() {
                        *value /= count;
                    }
                }
            }

            Ok(centroids)
        }

        fn euclidean_distance(&self, vec1: &[f64], vec2: &[f64]) -> f64 {
            if vec1.len() != vec2.len() {
                return f64::INFINITY;
            }

            vec1.iter()
                .zip(vec2.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        }

        fn centroid_distance(&self, old_centroids: &[Vec<f64>], new_centroids: &[Vec<f64>]) -> f64 {
            let mut total_distance = 0.0;

            for (old, new) in old_centroids.iter().zip(new_centroids.iter()) {
                total_distance += self.euclidean_distance(old, new);
            }

            total_distance / self.k as f64
        }
    }

    /// Hierarchical clustering for text documents
    #[derive(Debug, Clone)]
    pub struct HierarchicalClusterer {
        linkage: LinkageMethod,
    }

    #[derive(Debug, Clone)]
    pub enum LinkageMethod {
        Single,   // Minimum distance
        Complete, // Maximum distance
        Average,  // Average distance
    }

    impl Default for HierarchicalClusterer {
        fn default() -> Self {
            Self {
                linkage: LinkageMethod::Average,
            }
        }
    }

    impl HierarchicalClusterer {
        pub fn with_linkage(mut self, linkage: LinkageMethod) -> Self {
            self.linkage = linkage;
            self
        }

        /// Cluster documents using hierarchical clustering
        pub fn cluster(&self, documents: &[&str], num_clusters: usize) -> Result<Vec<usize>> {
            if documents.len() < num_clusters {
                return Err(TextError::Other(anyhow::anyhow!(
                    "Number of documents must be >= num_clusters"
                )));
            }

            // Create TF-IDF vectors
            let mut tfidf_calculator = TfIdfCalculator::new();
            let tfidf_matrix = tfidf_calculator.fit_transform(documents)?;

            // Calculate distance matrix
            let distance_matrix = self.calculate_distance_matrix(&tfidf_matrix)?;

            // Perform hierarchical clustering
            let mut clusters: Vec<Vec<usize>> = (0..documents.len()).map(|i| vec![i]).collect();

            while clusters.len() > num_clusters {
                // Find closest pair of clusters
                let (cluster1_idx, cluster2_idx) =
                    self.find_closest_clusters(&clusters, &distance_matrix)?;

                // Merge clusters
                let cluster2 = clusters.remove(cluster2_idx.max(cluster1_idx));
                let cluster1_idx = cluster1_idx.min(cluster2_idx);
                clusters[cluster1_idx].extend(cluster2);
            }

            // Convert to assignment vector
            let mut assignments = vec![0; documents.len()];
            for (cluster_id, cluster) in clusters.iter().enumerate() {
                for &doc_id in cluster {
                    assignments[doc_id] = cluster_id;
                }
            }

            Ok(assignments)
        }

        fn calculate_distance_matrix(&self, tfidf_matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
            let n = tfidf_matrix.len();
            let mut distance_matrix = vec![vec![0.0; n]; n];

            for i in 0..n {
                for j in (i + 1)..n {
                    let distance = self.cosine_distance(&tfidf_matrix[i], &tfidf_matrix[j]);
                    distance_matrix[i][j] = distance;
                    distance_matrix[j][i] = distance;
                }
            }

            Ok(distance_matrix)
        }

        fn cosine_distance(&self, vec1: &[f64], vec2: &[f64]) -> f64 {
            if vec1.len() != vec2.len() {
                return f64::INFINITY;
            }

            let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
            let norm1_sq: f64 = vec1.iter().map(|x| x * x).sum();
            let norm2_sq: f64 = vec2.iter().map(|x| x * x).sum();

            let cosine_sim = if norm1_sq > 0.0 && norm2_sq > 0.0 {
                dot_product / (norm1_sq.sqrt() * norm2_sq.sqrt())
            } else {
                0.0
            };

            1.0 - cosine_sim.max(0.0).min(1.0)
        }

        fn find_closest_clusters(
            &self,
            clusters: &[Vec<usize>],
            distance_matrix: &[Vec<f64>],
        ) -> Result<(usize, usize)> {
            let mut min_distance = f64::INFINITY;
            let mut closest_pair = (0, 1);

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let distance =
                        self.cluster_distance(&clusters[i], &clusters[j], distance_matrix);
                    if distance < min_distance {
                        min_distance = distance;
                        closest_pair = (i, j);
                    }
                }
            }

            Ok(closest_pair)
        }

        fn cluster_distance(
            &self,
            cluster1: &[usize],
            cluster2: &[usize],
            distance_matrix: &[Vec<f64>],
        ) -> f64 {
            match self.linkage {
                LinkageMethod::Single => {
                    // Minimum distance between any two points in different clusters
                    let mut min_dist = f64::INFINITY;
                    for &i in cluster1 {
                        for &j in cluster2 {
                            min_dist = min_dist.min(distance_matrix[i][j]);
                        }
                    }
                    min_dist
                }
                LinkageMethod::Complete => {
                    // Maximum distance between any two points in different clusters
                    let mut max_dist: f64 = 0.0;
                    for &i in cluster1 {
                        for &j in cluster2 {
                            max_dist = max_dist.max(distance_matrix[i][j]);
                        }
                    }
                    max_dist
                }
                LinkageMethod::Average => {
                    // Average distance between all pairs of points in different clusters
                    let mut total_dist = 0.0;
                    let mut count = 0;
                    for &i in cluster1 {
                        for &j in cluster2 {
                            total_dist += distance_matrix[i][j];
                            count += 1;
                        }
                    }
                    if count > 0 {
                        total_dist / count as f64
                    } else {
                        0.0
                    }
                }
            }
        }
    }
}

// Re-export clustering types
pub use clustering::{HierarchicalClusterer, KMeansClusterer, LinkageMethod};
