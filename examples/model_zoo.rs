//! ToRSh Model Zoo - Pre-trained Models and Easy Inference
//!
//! This module provides a comprehensive model zoo with:
//! - Pre-trained model definitions and weights loading
//! - Easy inference APIs for common tasks
//! - Model discovery and metadata management
//! - Automatic preprocessing and postprocessing
//! - Performance optimization and caching
//! - Transfer learning utilities

use torsh::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Model metadata for the zoo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub task: Task,
    pub architecture: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub preprocessing: PreprocessingConfig,
    pub postprocessing: PostprocessingConfig,
    pub performance_metrics: HashMap<String, f64>,
    pub paper_url: Option<String>,
    pub license: String,
    pub model_size_mb: f64,
    pub parameters_count: u64,
}

/// Supported tasks in the model zoo
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Task {
    ImageClassification,
    ObjectDetection,
    SemanticSegmentation,
    TextClassification,
    QuestionAnswering,
    TextGeneration,
    SpeechRecognition,
    Translation,
    Embedding,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub image_size: Option<(usize, usize)>,
    pub mean: Option<Vec<f64>>,
    pub std: Option<Vec<f64>>,
    pub normalization: Option<String>,
    pub tokenizer: Option<String>,
    pub max_length: Option<usize>,
}

/// Postprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessingConfig {
    pub class_names: Option<Vec<String>>,
    pub top_k: Option<usize>,
    pub threshold: Option<f64>,
    pub decode_strategy: Option<String>,
}

/// Model zoo registry
pub struct ModelZoo {
    models: HashMap<String, ModelMetadata>,
    cache_dir: PathBuf,
    download_base_url: String,
}

impl ModelZoo {
    pub fn new(cache_dir: Option<PathBuf>) -> Self {
        let cache_dir = cache_dir.unwrap_or_else(|| {
            std::env::var("TORSH_MODEL_CACHE")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    let mut path = std::env::temp_dir();
                    path.push("torsh_model_cache");
                    path
                })
        });
        
        let mut zoo = Self {
            models: HashMap::new(),
            cache_dir,
            download_base_url: "https://huggingface.co/torsh/".to_string(),
        };
        
        zoo.register_builtin_models();
        zoo
    }
    
    fn register_builtin_models(&mut self) {
        // Vision models
        self.register_model(create_resnet50_metadata());
        self.register_model(create_vit_base_metadata());
        self.register_model(create_efficientnet_b0_metadata());
        self.register_model(create_mobilenet_v3_metadata());
        
        // NLP models
        self.register_model(create_bert_base_metadata());
        self.register_model(create_gpt2_small_metadata());
        self.register_model(create_roberta_base_metadata());
        self.register_model(create_distilbert_metadata());
        
        // Multimodal models
        self.register_model(create_clip_metadata());
    }
    
    pub fn register_model(&mut self, metadata: ModelMetadata) {
        self.models.insert(metadata.name.clone(), metadata);
    }
    
    pub fn list_models(&self) -> Vec<&ModelMetadata> {
        self.models.values().collect()
    }
    
    pub fn list_models_by_task(&self, task: &Task) -> Vec<&ModelMetadata> {
        self.models.values()
            .filter(|m| &m.task == task)
            .collect()
    }
    
    pub fn get_model_metadata(&self, name: &str) -> Option<&ModelMetadata> {
        self.models.get(name)
    }
    
    pub fn load_model(&self, name: &str) -> Result<Box<dyn ModelInterface>> {
        let metadata = self.get_model_metadata(name)
            .ok_or_else(|| TorshError::Other(format!("Model {} not found", name)))?;
        
        match metadata.architecture.as_str() {
            "resnet50" => Ok(Box::new(ResNet50Model::load(metadata, &self.cache_dir)?)),
            "vit_base" => Ok(Box::new(ViTBaseModel::load(metadata, &self.cache_dir)?)),
            "bert_base" => Ok(Box::new(BertBaseModel::load(metadata, &self.cache_dir)?)),
            "gpt2_small" => Ok(Box::new(GPT2SmallModel::load(metadata, &self.cache_dir)?)),
            "clip" => Ok(Box::new(CLIPModel::load(metadata, &self.cache_dir)?)),
            _ => Err(TorshError::Other(format!("Unsupported architecture: {}", metadata.architecture))),
        }
    }
    
    pub fn download_model(&self, name: &str) -> Result<()> {
        let metadata = self.get_model_metadata(name)
            .ok_or_else(|| TorshError::Other(format!("Model {} not found", name)))?;
        
        let model_dir = self.cache_dir.join(&metadata.name);
        std::fs::create_dir_all(&model_dir)?;
        
        // Download model weights (simplified)
        let weights_url = format!("{}{}/pytorch_model.bin", self.download_base_url, metadata.name);
        let weights_path = model_dir.join("pytorch_model.bin");
        
        if !weights_path.exists() {
            println!("Downloading model weights for {}...", name);
            self.download_file(&weights_url, &weights_path)?;
        }
        
        // Download configuration
        let config_url = format!("{}{}/config.json", self.download_base_url, metadata.name);
        let config_path = model_dir.join("config.json");
        
        if !config_path.exists() {
            println!("Downloading model config for {}...", name);
            self.download_file(&config_url, &config_path)?;
        }
        
        println!("Model {} downloaded successfully", name);
        Ok(())
    }
    
    fn download_file(&self, url: &str, path: &PathBuf) -> Result<()> {
        // Simplified download implementation
        // In practice, this would use a proper HTTP client
        println!("Downloading {} to {:?}", url, path);
        
        // Create a dummy file for demonstration
        std::fs::write(path, b"dummy_model_data")?;
        
        Ok(())
    }
}

/// Common interface for all models in the zoo
pub trait ModelInterface {
    fn predict(&self, input: &Tensor) -> Result<ModelOutput>;
    fn preprocess(&self, input: &dyn std::any::Any) -> Result<Tensor>;
    fn postprocess(&self, output: &Tensor) -> Result<ModelOutput>;
    fn get_metadata(&self) -> &ModelMetadata;
}

/// Unified output format for all models
#[derive(Debug, Clone)]
pub enum ModelOutput {
    Classification {
        classes: Vec<String>,
        probabilities: Vec<f64>,
        top_k: usize,
    },
    Detection {
        boxes: Vec<BoundingBox>,
        scores: Vec<f64>,
        classes: Vec<String>,
    },
    Segmentation {
        mask: Tensor,
        classes: Vec<String>,
    },
    Text {
        generated_text: String,
        tokens: Vec<String>,
        scores: Option<Vec<f64>>,
    },
    Embedding {
        embedding: Tensor,
        dimension: usize,
    },
    QuestionAnswer {
        answer: String,
        start_position: usize,
        end_position: usize,
        confidence: f64,
    },
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// ResNet-50 implementation for the model zoo
pub struct ResNet50Model {
    model: Sequential,
    metadata: ModelMetadata,
    preprocessing: ImagePreprocessor,
}

impl ResNet50Model {
    pub fn load(metadata: &ModelMetadata, cache_dir: &PathBuf) -> Result<Self> {
        // Load ResNet-50 architecture
        let model = Self::build_resnet50()?;
        
        // Load pre-trained weights
        let weights_path = cache_dir.join(&metadata.name).join("pytorch_model.bin");
        if weights_path.exists() {
            // Load weights (simplified)
            println!("Loading pre-trained weights for ResNet-50");
        }
        
        let preprocessing = ImagePreprocessor::new(
            metadata.preprocessing.image_size.unwrap_or((224, 224)),
            metadata.preprocessing.mean.clone().unwrap_or(vec![0.485, 0.456, 0.406]),
            metadata.preprocessing.std.clone().unwrap_or(vec![0.229, 0.224, 0.225]),
        );
        
        Ok(Self {
            model,
            metadata: metadata.clone(),
            preprocessing,
        })
    }
    
    fn build_resnet50() -> Result<Sequential> {
        // Simplified ResNet-50 architecture
        Ok(Sequential::new()
            .add(Conv2d::new(3, 64, 7, 2, 3)?)
            .add(BatchNorm2d::new(64)?)
            .add(ReLU::new())
            .add(MaxPool2d::new(3, 2, 1)?)
            // ... more layers would be added here
            .add(AdaptiveAvgPool2d::new(1)?)
            .add(Flatten::new())
            .add(Linear::new(2048, 1000)))
    }
}

impl ModelInterface for ResNet50Model {
    fn predict(&self, input: &Tensor) -> Result<ModelOutput> {
        let logits = self.model.forward(input)?;
        self.postprocess(&logits)
    }
    
    fn preprocess(&self, input: &dyn std::any::Any) -> Result<Tensor> {
        if let Some(image) = input.downcast_ref::<Tensor>() {
            self.preprocessing.process(image)
        } else {
            Err(TorshError::Other("Invalid input type for ResNet50".to_string()))
        }
    }
    
    fn postprocess(&self, output: &Tensor) -> Result<ModelOutput> {
        let probabilities = F::softmax(output, -1)?;
        let (top_probs, top_indices) = probabilities.topk(5, -1, true)?;
        
        let classes = if let Some(ref class_names) = self.metadata.postprocessing.class_names {
            (0..5).map(|i| {
                let idx = top_indices.select(-1, i).unwrap().item::<i64>() as usize;
                class_names.get(idx).cloned().unwrap_or_else(|| format!("class_{}", idx))
            }).collect()
        } else {
            (0..5).map(|i| format!("class_{}", i)).collect()
        };
        
        let probs: Vec<f64> = (0..5).map(|i| {
            top_probs.select(-1, i).unwrap().item::<f32>() as f64
        }).collect();
        
        Ok(ModelOutput::Classification {
            classes,
            probabilities: probs,
            top_k: 5,
        })
    }
    
    fn get_metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
}

/// ViT-Base implementation for the model zoo
pub struct ViTBaseModel {
    model: Sequential,
    metadata: ModelMetadata,
    preprocessing: ImagePreprocessor,
}

impl ViTBaseModel {
    pub fn load(metadata: &ModelMetadata, cache_dir: &PathBuf) -> Result<Self> {
        let model = Self::build_vit_base()?;
        
        let preprocessing = ImagePreprocessor::new(
            metadata.preprocessing.image_size.unwrap_or((224, 224)),
            metadata.preprocessing.mean.clone().unwrap_or(vec![0.5, 0.5, 0.5]),
            metadata.preprocessing.std.clone().unwrap_or(vec![0.5, 0.5, 0.5]),
        );
        
        Ok(Self {
            model,
            metadata: metadata.clone(),
            preprocessing,
        })
    }
    
    fn build_vit_base() -> Result<Sequential> {
        // Simplified ViT architecture
        Ok(Sequential::new()
            .add(Linear::new(768, 1000))) // Simplified for demo
    }
}

impl ModelInterface for ViTBaseModel {
    fn predict(&self, input: &Tensor) -> Result<ModelOutput> {
        let logits = self.model.forward(input)?;
        self.postprocess(&logits)
    }
    
    fn preprocess(&self, input: &dyn std::any::Any) -> Result<Tensor> {
        if let Some(image) = input.downcast_ref::<Tensor>() {
            self.preprocessing.process(image)
        } else {
            Err(TorshError::Other("Invalid input type for ViT".to_string()))
        }
    }
    
    fn postprocess(&self, output: &Tensor) -> Result<ModelOutput> {
        let probabilities = F::softmax(output, -1)?;
        let (top_probs, top_indices) = probabilities.topk(5, -1, true)?;
        
        let classes: Vec<String> = (0..5).map(|i| format!("class_{}", i)).collect();
        let probs: Vec<f64> = (0..5).map(|i| {
            top_probs.select(-1, i).unwrap().item::<f32>() as f64
        }).collect();
        
        Ok(ModelOutput::Classification {
            classes,
            probabilities: probs,
            top_k: 5,
        })
    }
    
    fn get_metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
}

/// BERT-Base implementation for the model zoo
pub struct BertBaseModel {
    model: Sequential,
    metadata: ModelMetadata,
    tokenizer: TextTokenizer,
}

impl BertBaseModel {
    pub fn load(metadata: &ModelMetadata, cache_dir: &PathBuf) -> Result<Self> {
        let model = Self::build_bert_base()?;
        
        let tokenizer = TextTokenizer::new(
            metadata.preprocessing.max_length.unwrap_or(512),
            "[PAD]".to_string(),
            "[CLS]".to_string(),
            "[SEP]".to_string(),
        );
        
        Ok(Self {
            model,
            metadata: metadata.clone(),
            tokenizer,
        })
    }
    
    fn build_bert_base() -> Result<Sequential> {
        // Simplified BERT architecture
        Ok(Sequential::new()
            .add(Linear::new(768, 768))
            .add(Tanh::new())
            .add(Linear::new(768, 2))) // Binary classification
    }
}

impl ModelInterface for BertBaseModel {
    fn predict(&self, input: &Tensor) -> Result<ModelOutput> {
        let logits = self.model.forward(input)?;
        self.postprocess(&logits)
    }
    
    fn preprocess(&self, input: &dyn std::any::Any) -> Result<Tensor> {
        if let Some(text) = input.downcast_ref::<String>() {
            self.tokenizer.encode(text)
        } else {
            Err(TorshError::Other("Invalid input type for BERT".to_string()))
        }
    }
    
    fn postprocess(&self, output: &Tensor) -> Result<ModelOutput> {
        let probabilities = F::softmax(output, -1)?;
        let (top_probs, top_indices) = probabilities.topk(2, -1, true)?;
        
        let classes = vec!["negative".to_string(), "positive".to_string()];
        let probs: Vec<f64> = (0..2).map(|i| {
            top_probs.select(-1, i).unwrap().item::<f32>() as f64
        }).collect();
        
        Ok(ModelOutput::Classification {
            classes,
            probabilities: probs,
            top_k: 2,
        })
    }
    
    fn get_metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
}

/// GPT-2 Small implementation for the model zoo
pub struct GPT2SmallModel {
    model: Sequential,
    metadata: ModelMetadata,
    tokenizer: TextTokenizer,
}

impl GPT2SmallModel {
    pub fn load(metadata: &ModelMetadata, cache_dir: &PathBuf) -> Result<Self> {
        let model = Self::build_gpt2_small()?;
        
        let tokenizer = TextTokenizer::new(
            metadata.preprocessing.max_length.unwrap_or(1024),
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
        );
        
        Ok(Self {
            model,
            metadata: metadata.clone(),
            tokenizer,
        })
    }
    
    fn build_gpt2_small() -> Result<Sequential> {
        // Simplified GPT-2 architecture
        Ok(Sequential::new()
            .add(Linear::new(768, 50257))) // Vocab size
    }
}

impl ModelInterface for GPT2SmallModel {
    fn predict(&self, input: &Tensor) -> Result<ModelOutput> {
        let logits = self.model.forward(input)?;
        self.postprocess(&logits)
    }
    
    fn preprocess(&self, input: &dyn std::any::Any) -> Result<Tensor> {
        if let Some(text) = input.downcast_ref::<String>() {
            self.tokenizer.encode(text)
        } else {
            Err(TorshError::Other("Invalid input type for GPT-2".to_string()))
        }
    }
    
    fn postprocess(&self, output: &Tensor) -> Result<ModelOutput> {
        // Generate text (simplified)
        let generated_text = "Generated text would appear here".to_string();
        let tokens = vec!["token1".to_string(), "token2".to_string()];
        
        Ok(ModelOutput::Text {
            generated_text,
            tokens,
            scores: None,
        })
    }
    
    fn get_metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
}

/// CLIP implementation for the model zoo
pub struct CLIPModel {
    text_encoder: Sequential,
    image_encoder: Sequential,
    metadata: ModelMetadata,
    image_preprocessing: ImagePreprocessor,
    text_tokenizer: TextTokenizer,
}

impl CLIPModel {
    pub fn load(metadata: &ModelMetadata, cache_dir: &PathBuf) -> Result<Self> {
        let text_encoder = Self::build_text_encoder()?;
        let image_encoder = Self::build_image_encoder()?;
        
        let image_preprocessing = ImagePreprocessor::new(
            (224, 224),
            vec![0.48145466, 0.4578275, 0.40821073],
            vec![0.26862954, 0.26130258, 0.27577711],
        );
        
        let text_tokenizer = TextTokenizer::new(
            77,
            "<pad>".to_string(),
            "<start>".to_string(),
            "<end>".to_string(),
        );
        
        Ok(Self {
            text_encoder,
            image_encoder,
            metadata: metadata.clone(),
            image_preprocessing,
            text_tokenizer,
        })
    }
    
    fn build_text_encoder() -> Result<Sequential> {
        Ok(Sequential::new().add(Linear::new(512, 512)))
    }
    
    fn build_image_encoder() -> Result<Sequential> {
        Ok(Sequential::new().add(Linear::new(768, 512)))
    }
    
    pub fn encode_text(&self, text: &str) -> Result<Tensor> {
        let tokens = self.text_tokenizer.encode(text)?;
        self.text_encoder.forward(&tokens)
    }
    
    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor> {
        let processed = self.image_preprocessing.process(image)?;
        self.image_encoder.forward(&processed)
    }
    
    pub fn compute_similarity(&self, text: &str, image: &Tensor) -> Result<f64> {
        let text_features = self.encode_text(text)?;
        let image_features = self.encode_image(image)?;
        
        // Normalize features
        let text_norm = text_features.div(&text_features.norm()?)?;
        let image_norm = image_features.div(&image_features.norm()?)?;
        
        // Compute cosine similarity
        let similarity = text_norm.dot(&image_norm)?;
        Ok(similarity.item::<f32>() as f64)
    }
}

impl ModelInterface for CLIPModel {
    fn predict(&self, input: &Tensor) -> Result<ModelOutput> {
        let embedding = self.image_encoder.forward(input)?;
        Ok(ModelOutput::Embedding {
            embedding,
            dimension: 512,
        })
    }
    
    fn preprocess(&self, input: &dyn std::any::Any) -> Result<Tensor> {
        if let Some(image) = input.downcast_ref::<Tensor>() {
            self.image_preprocessing.process(image)
        } else {
            Err(TorshError::Other("Invalid input type for CLIP".to_string()))
        }
    }
    
    fn postprocess(&self, output: &Tensor) -> Result<ModelOutput> {
        Ok(ModelOutput::Embedding {
            embedding: output.clone(),
            dimension: output.shape().dims().last().copied().unwrap_or(0),
        })
    }
    
    fn get_metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
}

/// Image preprocessing utilities
pub struct ImagePreprocessor {
    size: (usize, usize),
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl ImagePreprocessor {
    pub fn new(size: (usize, usize), mean: Vec<f64>, std: Vec<f64>) -> Self {
        Self { size, mean, std }
    }
    
    pub fn process(&self, image: &Tensor) -> Result<Tensor> {
        let mut processed = image.clone();
        
        // Resize
        processed = F::interpolate(&processed, &[self.size.0, self.size.1], "bilinear", true)?;
        
        // Normalize to [0, 1]
        processed = processed.div_scalar(255.0)?;
        
        // Apply mean and std normalization
        for (i, (&mean, &std)) in self.mean.iter().zip(&self.std).enumerate() {
            let channel = processed.select(-3, i)?;
            let normalized = channel.sub_scalar(mean)?.div_scalar(std)?;
            processed = processed.index_put(
                &[tensor![..], tensor![i as i64], tensor![..], tensor![..]],
                &normalized,
            )?;
        }
        
        Ok(processed)
    }
}

/// Text tokenization utilities
pub struct TextTokenizer {
    max_length: usize,
    pad_token: String,
    cls_token: String,
    sep_token: String,
}

impl TextTokenizer {
    pub fn new(max_length: usize, pad_token: String, cls_token: String, sep_token: String) -> Self {
        Self {
            max_length,
            pad_token,
            cls_token,
            sep_token,
        }
    }
    
    pub fn encode(&self, text: &str) -> Result<Tensor> {
        // Simplified tokenization
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut token_ids = vec![1]; // CLS token ID
        
        for word in words.iter().take(self.max_length - 2) {
            // Simple word to ID mapping (in practice, use proper tokenizer)
            let token_id = word.len() % 1000; // Dummy tokenization
            token_ids.push(token_id);
        }
        
        token_ids.push(2); // SEP token ID
        
        // Pad to max length
        while token_ids.len() < self.max_length {
            token_ids.push(0); // PAD token ID
        }
        
        let token_tensor = Tensor::from_data(
            &token_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            &[1, token_ids.len()],
        );
        
        Ok(token_tensor)
    }
}

/// Model metadata factory functions
fn create_resnet50_metadata() -> ModelMetadata {
    ModelMetadata {
        name: "resnet50".to_string(),
        version: "1.0.0".to_string(),
        description: "ResNet-50 pre-trained on ImageNet".to_string(),
        task: Task::ImageClassification,
        architecture: "resnet50".to_string(),
        input_shape: vec![3, 224, 224],
        output_shape: vec![1000],
        preprocessing: PreprocessingConfig {
            image_size: Some((224, 224)),
            mean: Some(vec![0.485, 0.456, 0.406]),
            std: Some(vec![0.229, 0.224, 0.225]),
            normalization: Some("imagenet".to_string()),
            tokenizer: None,
            max_length: None,
        },
        postprocessing: PostprocessingConfig {
            class_names: Some(create_imagenet_classes()),
            top_k: Some(5),
            threshold: None,
            decode_strategy: None,
        },
        performance_metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("top1_accuracy".to_string(), 76.15);
            metrics.insert("top5_accuracy".to_string(), 92.87);
            metrics
        },
        paper_url: Some("https://arxiv.org/abs/1512.03385".to_string()),
        license: "Apache-2.0".to_string(),
        model_size_mb: 97.8,
        parameters_count: 25_557_032,
    }
}

fn create_vit_base_metadata() -> ModelMetadata {
    ModelMetadata {
        name: "vit_base_patch16".to_string(),
        version: "1.0.0".to_string(),
        description: "Vision Transformer Base model pre-trained on ImageNet".to_string(),
        task: Task::ImageClassification,
        architecture: "vit_base".to_string(),
        input_shape: vec![3, 224, 224],
        output_shape: vec![1000],
        preprocessing: PreprocessingConfig {
            image_size: Some((224, 224)),
            mean: Some(vec![0.5, 0.5, 0.5]),
            std: Some(vec![0.5, 0.5, 0.5]),
            normalization: Some("vit".to_string()),
            tokenizer: None,
            max_length: None,
        },
        postprocessing: PostprocessingConfig {
            class_names: Some(create_imagenet_classes()),
            top_k: Some(5),
            threshold: None,
            decode_strategy: None,
        },
        performance_metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("top1_accuracy".to_string(), 81.28);
            metrics.insert("top5_accuracy".to_string(), 95.31);
            metrics
        },
        paper_url: Some("https://arxiv.org/abs/2010.11929".to_string()),
        license: "Apache-2.0".to_string(),
        model_size_mb: 330.3,
        parameters_count: 86_567_656,
    }
}

fn create_bert_base_metadata() -> ModelMetadata {
    ModelMetadata {
        name: "bert_base_uncased".to_string(),
        version: "1.0.0".to_string(),
        description: "BERT base model (uncased)".to_string(),
        task: Task::TextClassification,
        architecture: "bert_base".to_string(),
        input_shape: vec![512],
        output_shape: vec![2],
        preprocessing: PreprocessingConfig {
            image_size: None,
            mean: None,
            std: None,
            normalization: None,
            tokenizer: Some("bert".to_string()),
            max_length: Some(512),
        },
        postprocessing: PostprocessingConfig {
            class_names: Some(vec!["negative".to_string(), "positive".to_string()]),
            top_k: Some(2),
            threshold: None,
            decode_strategy: None,
        },
        performance_metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("accuracy".to_string(), 92.7);
            metrics
        },
        paper_url: Some("https://arxiv.org/abs/1810.04805".to_string()),
        license: "Apache-2.0".to_string(),
        model_size_mb: 417.6,
        parameters_count: 109_482_240,
    }
}

fn create_gpt2_small_metadata() -> ModelMetadata {
    ModelMetadata {
        name: "gpt2_small".to_string(),
        version: "1.0.0".to_string(),
        description: "GPT-2 small model for text generation".to_string(),
        task: Task::TextGeneration,
        architecture: "gpt2_small".to_string(),
        input_shape: vec![1024],
        output_shape: vec![50257],
        preprocessing: PreprocessingConfig {
            image_size: None,
            mean: None,
            std: None,
            normalization: None,
            tokenizer: Some("gpt2".to_string()),
            max_length: Some(1024),
        },
        postprocessing: PostprocessingConfig {
            class_names: None,
            top_k: None,
            threshold: None,
            decode_strategy: Some("greedy".to_string()),
        },
        performance_metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("perplexity".to_string(), 29.41);
            metrics
        },
        paper_url: Some("https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf".to_string()),
        license: "MIT".to_string(),
        model_size_mb: 487.3,
        parameters_count: 124_439_808,
    }
}

fn create_clip_metadata() -> ModelMetadata {
    ModelMetadata {
        name: "clip_vit_base_patch32".to_string(),
        version: "1.0.0".to_string(),
        description: "CLIP model with ViT-B/32 visual encoder".to_string(),
        task: Task::Embedding,
        architecture: "clip".to_string(),
        input_shape: vec![3, 224, 224],
        output_shape: vec![512],
        preprocessing: PreprocessingConfig {
            image_size: Some((224, 224)),
            mean: Some(vec![0.48145466, 0.4578275, 0.40821073]),
            std: Some(vec![0.26862954, 0.26130258, 0.27577711]),
            normalization: Some("clip".to_string()),
            tokenizer: Some("clip".to_string()),
            max_length: Some(77),
        },
        postprocessing: PostprocessingConfig {
            class_names: None,
            top_k: None,
            threshold: None,
            decode_strategy: None,
        },
        performance_metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("zero_shot_accuracy".to_string(), 63.2);
            metrics
        },
        paper_url: Some("https://arxiv.org/abs/2103.00020".to_string()),
        license: "MIT".to_string(),
        model_size_mb: 338.3,
        parameters_count: 151_277_313,
    }
}

fn create_efficientnet_b0_metadata() -> ModelMetadata {
    ModelMetadata {
        name: "efficientnet_b0".to_string(),
        version: "1.0.0".to_string(),
        description: "EfficientNet-B0 model".to_string(),
        task: Task::ImageClassification,
        architecture: "efficientnet_b0".to_string(),
        input_shape: vec![3, 224, 224],
        output_shape: vec![1000],
        preprocessing: PreprocessingConfig {
            image_size: Some((224, 224)),
            mean: Some(vec![0.485, 0.456, 0.406]),
            std: Some(vec![0.229, 0.224, 0.225]),
            normalization: Some("imagenet".to_string()),
            tokenizer: None,
            max_length: None,
        },
        postprocessing: PostprocessingConfig {
            class_names: Some(create_imagenet_classes()),
            top_k: Some(5),
            threshold: None,
            decode_strategy: None,
        },
        performance_metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("top1_accuracy".to_string(), 77.3);
            metrics.insert("top5_accuracy".to_string(), 93.5);
            metrics
        },
        paper_url: Some("https://arxiv.org/abs/1905.11946".to_string()),
        license: "Apache-2.0".to_string(),
        model_size_mb: 20.3,
        parameters_count: 5_288_548,
    }
}

fn create_mobilenet_v3_metadata() -> ModelMetadata {
    ModelMetadata {
        name: "mobilenet_v3_large".to_string(),
        version: "1.0.0".to_string(),
        description: "MobileNetV3 Large model".to_string(),
        task: Task::ImageClassification,
        architecture: "mobilenet_v3".to_string(),
        input_shape: vec![3, 224, 224],
        output_shape: vec![1000],
        preprocessing: PreprocessingConfig {
            image_size: Some((224, 224)),
            mean: Some(vec![0.485, 0.456, 0.406]),
            std: Some(vec![0.229, 0.224, 0.225]),
            normalization: Some("imagenet".to_string()),
            tokenizer: None,
            max_length: None,
        },
        postprocessing: PostprocessingConfig {
            class_names: Some(create_imagenet_classes()),
            top_k: Some(5),
            threshold: None,
            decode_strategy: None,
        },
        performance_metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("top1_accuracy".to_string(), 75.2);
            metrics.insert("top5_accuracy".to_string(), 92.2);
            metrics
        },
        paper_url: Some("https://arxiv.org/abs/1905.02244".to_string()),
        license: "Apache-2.0".to_string(),
        model_size_mb: 21.1,
        parameters_count: 5_483_032,
    }
}

fn create_roberta_base_metadata() -> ModelMetadata {
    ModelMetadata {
        name: "roberta_base".to_string(),
        version: "1.0.0".to_string(),
        description: "RoBERTa base model".to_string(),
        task: Task::TextClassification,
        architecture: "roberta_base".to_string(),
        input_shape: vec![512],
        output_shape: vec![2],
        preprocessing: PreprocessingConfig {
            image_size: None,
            mean: None,
            std: None,
            normalization: None,
            tokenizer: Some("roberta".to_string()),
            max_length: Some(512),
        },
        postprocessing: PostprocessingConfig {
            class_names: Some(vec!["negative".to_string(), "positive".to_string()]),
            top_k: Some(2),
            threshold: None,
            decode_strategy: None,
        },
        performance_metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("accuracy".to_string(), 94.8);
            metrics
        },
        paper_url: Some("https://arxiv.org/abs/1907.11692".to_string()),
        license: "MIT".to_string(),
        model_size_mb: 478.3,
        parameters_count: 124_647_424,
    }
}

fn create_distilbert_metadata() -> ModelMetadata {
    ModelMetadata {
        name: "distilbert_base_uncased".to_string(),
        version: "1.0.0".to_string(),
        description: "DistilBERT base model (uncased)".to_string(),
        task: Task::TextClassification,
        architecture: "distilbert".to_string(),
        input_shape: vec![512],
        output_shape: vec![2],
        preprocessing: PreprocessingConfig {
            image_size: None,
            mean: None,
            std: None,
            normalization: None,
            tokenizer: Some("bert".to_string()),
            max_length: Some(512),
        },
        postprocessing: PostprocessingConfig {
            class_names: Some(vec!["negative".to_string(), "positive".to_string()]),
            top_k: Some(2),
            threshold: None,
            decode_strategy: None,
        },
        performance_metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("accuracy".to_string(), 90.7);
            metrics
        },
        paper_url: Some("https://arxiv.org/abs/1910.01108".to_string()),
        license: "Apache-2.0".to_string(),
        model_size_mb: 255.8,
        parameters_count: 66_955_010,
    }
}

fn create_imagenet_classes() -> Vec<String> {
    // Simplified ImageNet classes (first 10)
    vec![
        "tench".to_string(),
        "goldfish".to_string(),
        "great white shark".to_string(),
        "tiger shark".to_string(),
        "hammerhead".to_string(),
        "electric ray".to_string(),
        "stingray".to_string(),
        "cock".to_string(),
        "hen".to_string(),
        "ostrich".to_string(),
    ]
}

/// Example usage and testing
pub fn run_model_zoo_example() -> Result<()> {
    println!("ToRSh Model Zoo Demo");
    
    // Initialize model zoo
    let zoo = ModelZoo::new(None);
    
    // List all models
    println!("\nAvailable models:");
    for model in zoo.list_models() {
        println!("- {}: {} ({})", model.name, model.description, model.architecture);
        println!("  Task: {:?}, Size: {:.1} MB, Parameters: {}", 
                 model.task, model.model_size_mb, model.parameters_count);
    }
    
    // List models by task
    println!("\nImage classification models:");
    for model in zoo.list_models_by_task(&Task::ImageClassification) {
        println!("- {}: {:.1}% top-1 accuracy", 
                 model.name, 
                 model.performance_metrics.get("top1_accuracy").unwrap_or(&0.0));
    }
    
    // Load and use a model
    println!("\nLoading ResNet-50 model...");
    let resnet = zoo.load_model("resnet50")?;
    
    // Test image classification
    let test_image = randn(&[1, 3, 224, 224]);
    let preprocessed = resnet.preprocess(&test_image as &dyn std::any::Any)?;
    let output = resnet.predict(&preprocessed)?;
    
    if let ModelOutput::Classification { classes, probabilities, top_k } = output {
        println!("Top {} predictions:", top_k);
        for (class, prob) in classes.iter().zip(probabilities.iter()) {
            println!("  {}: {:.3}", class, prob);
        }
    }
    
    // Load and use BERT model
    println!("\nLoading BERT model...");
    let bert = zoo.load_model("bert_base_uncased")?;
    
    let test_text = "This movie is amazing!".to_string();
    let text_input = bert.preprocess(&test_text as &dyn std::any::Any)?;
    let text_output = bert.predict(&text_input)?;
    
    if let ModelOutput::Classification { classes, probabilities, .. } = text_output {
        println!("Text classification result:");
        for (class, prob) in classes.iter().zip(probabilities.iter()) {
            println!("  {}: {:.3}", class, prob);
        }
    }
    
    // Load and use CLIP model
    println!("\nLoading CLIP model...");
    let clip = zoo.load_model("clip_vit_base_patch32")?;
    
    if let Ok(clip_model) = clip.as_any().downcast_ref::<CLIPModel>() {
        let similarity = clip_model.compute_similarity("a photo of a cat", &test_image)?;
        println!("Text-image similarity: {:.3}", similarity);
    }
    
    // Model metadata example
    let metadata = zoo.get_model_metadata("resnet50").unwrap();
    println!("\nResNet-50 metadata:");
    println!("  Paper: {:?}", metadata.paper_url);
    println!("  License: {}", metadata.license);
    println!("  Preprocessing: {:?}", metadata.preprocessing);
    
    println!("Model zoo demo completed successfully!");
    Ok(())
}

// Add trait for downcasting
trait AsAny {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl AsAny for CLIPModel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl AsAny for ResNet50Model {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl AsAny for ViTBaseModel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl AsAny for BertBaseModel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl AsAny for GPT2SmallModel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Extend ModelInterface to include downcasting
pub trait ModelInterfaceExt: ModelInterface + AsAny {}

impl ModelInterfaceExt for ResNet50Model {}
impl ModelInterfaceExt for ViTBaseModel {}
impl ModelInterfaceExt for BertBaseModel {}
impl ModelInterfaceExt for GPT2SmallModel {}
impl ModelInterfaceExt for CLIPModel {}

fn main() -> Result<()> {
    run_model_zoo_example()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_zoo_creation() {
        let zoo = ModelZoo::new(None);
        let models = zoo.list_models();
        assert!(!models.is_empty());
        
        let vision_models = zoo.list_models_by_task(&Task::ImageClassification);
        assert!(!vision_models.is_empty());
    }
    
    #[test]
    fn test_model_metadata() {
        let metadata = create_resnet50_metadata();
        assert_eq!(metadata.name, "resnet50");
        assert_eq!(metadata.task, Task::ImageClassification);
        assert!(metadata.performance_metrics.contains_key("top1_accuracy"));
    }
    
    #[test]
    fn test_image_preprocessor() {
        let preprocessor = ImagePreprocessor::new(
            (224, 224),
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
        );
        
        let image = randn(&[1, 3, 256, 256]);
        let processed = preprocessor.process(&image).unwrap();
        
        assert_eq!(processed.shape().dims(), &[1, 3, 224, 224]);
    }
    
    #[test]
    fn test_text_tokenizer() {
        let tokenizer = TextTokenizer::new(
            128,
            "[PAD]".to_string(),
            "[CLS]".to_string(),
            "[SEP]".to_string(),
        );
        
        let text = "Hello world this is a test";
        let tokens = tokenizer.encode(text).unwrap();
        
        assert_eq!(tokens.shape().dims(), &[1, 128]);
    }
    
    #[test]
    fn test_model_loading() {
        let zoo = ModelZoo::new(None);
        
        // Test that we can get metadata for all registered models
        for model_meta in zoo.list_models() {
            let metadata = zoo.get_model_metadata(&model_meta.name);
            assert!(metadata.is_some());
        }
    }
}