//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Result;

use super::types::{BatchIterator, DatasetIterator};

/// Base trait for all text datasets
pub trait Dataset {
    type Item;
    /// Get the length of the dataset
    fn len(&self) -> usize;
    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get an item by index
    fn get_item(&self, index: usize) -> Result<Self::Item>;
    /// Iterator over the dataset
    fn iter(&self) -> DatasetIterator<'_, Self>
    where
        Self: Sized,
    {
        DatasetIterator {
            dataset: self,
            index: 0,
        }
    }
}
/// Extension trait for batch iteration
pub trait BatchDataset: Dataset {
    fn batch_iter(&self, batch_size: usize) -> BatchIterator<'_, Self>
    where
        Self: Sized,
    {
        BatchIterator::new(self, batch_size)
    }
}
impl<D: Dataset> BatchDataset for D {}
#[cfg(test)]
mod tests {
    use super::super::types::{
        ConsolidatedDataset, DataAugmentation, DataItem, DatasetUtils, PreprocessingPipeline,
        Stopwords, TextDataset,
    };
    use super::*;
    #[test]
    fn test_dataset_utils_sample() {
        let texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
        ];
        let labels = vec![
            "label1".to_string(),
            "label2".to_string(),
            "label3".to_string(),
        ];
        let dataset = ConsolidatedDataset::new_classification(texts, labels)
            .expect("Failed to create dataset");
        let sampled = DatasetUtils::sample(&dataset, 2).expect("Failed to sample");
        assert_eq!(sampled.len(), 2);
    }
    #[test]
    fn test_dataset_utils_deduplicate() {
        let texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text1".to_string(),
        ];
        let labels = vec![
            "label1".to_string(),
            "label2".to_string(),
            "label1".to_string(),
        ];
        let dataset = ConsolidatedDataset::new_classification(texts, labels)
            .expect("Failed to create dataset");
        let dedup = DatasetUtils::deduplicate(&dataset);
        assert_eq!(dedup.len(), 2);
    }
    #[test]
    fn test_dataset_utils_filter_by_token_count() {
        let texts = vec![
            "short".to_string(),
            "this is longer text".to_string(),
            "medium text".to_string(),
        ];
        let labels = vec![
            "label1".to_string(),
            "label2".to_string(),
            "label3".to_string(),
        ];
        let dataset = ConsolidatedDataset::new_classification(texts, labels)
            .expect("Failed to create dataset");
        let filtered = DatasetUtils::filter_by_token_count(&dataset, Some(2), Some(3));
        assert_eq!(filtered.len(), 1);
    }
    #[test]
    fn test_dataset_utils_statistics() {
        let texts = vec!["ab".to_string(), "abcd".to_string(), "abcdef".to_string()];
        let labels = vec![
            "label1".to_string(),
            "label2".to_string(),
            "label3".to_string(),
        ];
        let dataset = ConsolidatedDataset::new_classification(texts, labels)
            .expect("Failed to create dataset");
        let stats = DatasetUtils::get_statistics(&dataset);
        assert_eq!(stats.total_items, 3);
        assert_eq!(stats.min_text_length, 2);
        assert_eq!(stats.max_text_length, 6);
        assert_eq!(stats.avg_text_length, 4);
    }
    #[test]
    fn test_batch_iterator() {
        let texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
            "text4".to_string(),
            "text5".to_string(),
        ];
        let dataset = TextDataset::from_texts(texts);
        let batches: Vec<_> = dataset.batch_iter(2).collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].as_ref().ok().map(|b| b.len()), Some(2));
        assert_eq!(batches[1].as_ref().ok().map(|b| b.len()), Some(2));
        assert_eq!(batches[2].as_ref().ok().map(|b| b.len()), Some(1));
    }
    #[test]
    fn test_data_augmentation_random_deletion() {
        let text = "this is a test sentence";
        let augmented = DataAugmentation::random_deletion(text, 0.2);
        assert!(!augmented.is_empty());
    }
    #[test]
    fn test_data_augmentation_random_swap() {
        let text = "word1 word2 word3 word4";
        let augmented = DataAugmentation::random_swap(text, 2);
        assert_eq!(
            text.split_whitespace().count(),
            augmented.split_whitespace().count()
        );
    }
    #[test]
    fn test_preprocessing_pipeline() {
        let pipeline = PreprocessingPipeline::new()
            .lowercase()
            .remove_punctuation()
            .normalize_whitespace();
        let text = "Hello,  World!  This   is  a   TEST.";
        let processed = pipeline.process(text);
        assert_eq!(processed, "hello world this is a test");
    }
    #[test]
    fn test_preprocessing_stopwords() {
        let stopwords = Stopwords::english();
        let pipeline = PreprocessingPipeline::new()
            .lowercase()
            .remove_stopwords(stopwords);
        let text = "This is a test with the stopwords";
        let processed = pipeline.process(text);
        assert!(!processed.contains(" is "));
        assert!(!processed.contains(" a "));
    }
    #[test]
    fn test_preprocessing_remove_urls() {
        let pipeline = PreprocessingPipeline::new().remove_urls();
        let text = "Check out https://example.com and http://test.com for more";
        let processed = pipeline.process(text);
        assert!(!processed.contains("https://"));
        assert!(!processed.contains("http://"));
    }
    #[test]
    fn test_preprocessing_remove_html() {
        let pipeline = PreprocessingPipeline::new().remove_html_tags();
        let text = "<p>Hello <strong>world</strong></p>";
        let processed = pipeline.process(text);
        assert!(!processed.contains('<'));
        assert!(!processed.contains('>'));
    }
    #[test]
    fn test_preprocessing_length_constraints() {
        let pipeline = PreprocessingPipeline::new().min_length(5).max_length(20);
        let short = "Hi";
        let good = "Hello World";
        let long = "This is a very long text that exceeds the maximum length";
        assert_eq!(pipeline.process(short), "");
        assert_eq!(pipeline.process(good), "Hello World");
        assert_eq!(pipeline.process(long).len(), 20);
    }
    #[test]
    fn test_shuffle_with_seed() {
        let texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
            "text4".to_string(),
        ];
        let labels = vec![
            "label1".to_string(),
            "label2".to_string(),
            "label3".to_string(),
            "label4".to_string(),
        ];
        let mut dataset1 = ConsolidatedDataset::new_classification(texts.clone(), labels.clone())
            .expect("Failed to create dataset");
        let mut dataset2 = ConsolidatedDataset::new_classification(texts, labels)
            .expect("Failed to create dataset");
        DatasetUtils::shuffle_with_seed(&mut dataset1, 42);
        DatasetUtils::shuffle_with_seed(&mut dataset2, 42);
        for i in 0..dataset1.len() {
            let item1 = dataset1.get_item(i).expect("Failed to get item");
            let item2 = dataset2.get_item(i).expect("Failed to get item");
            match (item1, item2) {
                (
                    DataItem::Classification {
                        text: t1,
                        label: l1,
                    },
                    DataItem::Classification {
                        text: t2,
                        label: l2,
                    },
                ) => {
                    assert_eq!(t1, t2);
                    assert_eq!(l1, l2);
                }
                _ => panic!("Unexpected item type"),
            }
        }
    }
}
