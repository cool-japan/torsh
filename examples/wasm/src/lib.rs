//! ToRSh WASM Deployment Example
//! 
//! This example demonstrates:
//! - Exporting ToRSh models to WebAssembly
//! - Running inference in the browser
//! - JavaScript/Rust interop for ML applications
//! - Optimizing model size for web deployment

use wasm_bindgen::prelude::*;
use torsh_tensor::Tensor;
use torsh_nn::{Module, Linear, Conv2d, ReLU};
use serde::{Serialize, Deserialize};
use std::panic;

// Set up better error messages in browser console
#[wasm_bindgen(start)]
pub fn init() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}

/// Log messages to browser console
macro_rules! log {
    ($($t:tt)*) => {
        web_sys::console::log_1(&format!($($t)*).into());
    };
}

/// Simple CNN for digit classification (MNIST-like)
#[wasm_bindgen]
pub struct DigitClassifier {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    relu: ReLU,
}

#[wasm_bindgen]
impl DigitClassifier {
    /// Create a new digit classifier
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        log!("Creating DigitClassifier model...");
        
        Self {
            // Input: 1x28x28 grayscale image
            conv1: Conv2d::new(1, 16, (3, 3), (1, 1), (1, 1), (1, 1), true, 1),
            conv2: Conv2d::new(16, 32, (3, 3), (2, 2), (1, 1), (1, 1), true, 1), // Stride 2 for downsampling
            fc1: Linear::new(32 * 14 * 14, 128, true),
            fc2: Linear::new(128, 10, true),
            relu: ReLU::new(),
        }
    }
    
    /// Predict digit from image data
    #[wasm_bindgen]
    pub fn predict(&self, image_data: &[f32]) -> Result<Prediction, JsValue> {
        // Validate input
        if image_data.len() != 28 * 28 {
            return Err(JsValue::from_str("Image must be 28x28 pixels"));
        }
        
        // Convert to tensor
        let input = Tensor::from_vec(image_data.to_vec(), &[1, 1, 28, 28])
            .map_err(|e| JsValue::from_str(&format!("Tensor creation error: {:?}", e)))?;
        
        // Forward pass
        let x = self.conv1.forward(&input)
            .map_err(|e| JsValue::from_str(&format!("Conv1 error: {:?}", e)))?;
        let x = self.relu.forward(&x)
            .map_err(|e| JsValue::from_str(&format!("ReLU error: {:?}", e)))?;
        
        let x = self.conv2.forward(&x)
            .map_err(|e| JsValue::from_str(&format!("Conv2 error: {:?}", e)))?;
        let x = self.relu.forward(&x)
            .map_err(|e| JsValue::from_str(&format!("ReLU error: {:?}", e)))?;
        
        // Flatten
        let x = x.view(&[1, 32 * 14 * 14])
            .map_err(|e| JsValue::from_str(&format!("Flatten error: {:?}", e)))?;
        
        let x = self.fc1.forward(&x)
            .map_err(|e| JsValue::from_str(&format!("FC1 error: {:?}", e)))?;
        let x = self.relu.forward(&x)
            .map_err(|e| JsValue::from_str(&format!("ReLU error: {:?}", e)))?;
        
        let logits = self.fc2.forward(&x)
            .map_err(|e| JsValue::from_str(&format!("FC2 error: {:?}", e)))?;
        
        // Apply softmax
        let probs = logits.softmax(1)
            .map_err(|e| JsValue::from_str(&format!("Softmax error: {:?}", e)))?;
        
        // Get prediction
        let probs_vec = probs.to_vec();
        let (digit, confidence) = probs_vec.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &conf)| (idx as u8, conf))
            .unwrap();
        
        Ok(Prediction {
            digit,
            confidence,
            probabilities: probs_vec,
        })
    }
    
    /// Get model info
    #[wasm_bindgen]
    pub fn model_info(&self) -> ModelInfo {
        let param_count: usize = [
            &self.conv1, &self.conv2, &self.fc1, &self.fc2
        ].iter()
            .flat_map(|m| m.parameters())
            .map(|p| p.numel())
            .sum();
        
        ModelInfo {
            name: "DigitClassifier".to_string(),
            version: "1.0.0".to_string(),
            parameters: param_count as u32,
            input_shape: vec![1, 28, 28],
            output_classes: 10,
        }
    }
}

/// Prediction result
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct Prediction {
    pub digit: u8,
    pub confidence: f32,
    probabilities: Vec<f32>,
}

#[wasm_bindgen]
impl Prediction {
    /// Get all class probabilities
    #[wasm_bindgen(getter)]
    pub fn probabilities(&self) -> Vec<f32> {
        self.probabilities.clone()
    }
}

/// Model information
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct ModelInfo {
    name: String,
    version: String,
    pub parameters: u32,
    input_shape: Vec<usize>,
    pub output_classes: u32,
}

#[wasm_bindgen]
impl ModelInfo {
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> String {
        self.version.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn input_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }
}

/// Text sentiment analyzer
#[wasm_bindgen]
pub struct SentimentAnalyzer {
    embedding_dim: usize,
    vocab_size: usize,
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

#[wasm_bindgen]
impl SentimentAnalyzer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        log!("Creating SentimentAnalyzer model...");
        
        let embedding_dim = 50;
        let vocab_size = 10000;
        
        Self {
            embedding_dim,
            vocab_size,
            fc1: Linear::new(embedding_dim * 20, 128, true), // Max sequence length 20
            fc2: Linear::new(128, 64, true),
            fc3: Linear::new(64, 3, true), // Negative, Neutral, Positive
        }
    }
    
    /// Analyze sentiment of text (using pre-tokenized indices)
    #[wasm_bindgen]
    pub fn analyze(&self, token_indices: &[u32]) -> Result<SentimentResult, JsValue> {
        // Simple embedding lookup (in practice, use proper embeddings)
        let max_len = 20;
        let mut embeddings = vec![0.0; self.embedding_dim * max_len];
        
        for (i, &idx) in token_indices.iter().take(max_len).enumerate() {
            // Simple hash-based embedding
            for j in 0..self.embedding_dim {
                embeddings[i * self.embedding_dim + j] = 
                    ((idx as usize * 31 + j) % 100) as f32 / 100.0 - 0.5;
            }
        }
        
        let input = Tensor::from_vec(embeddings, &[1, self.embedding_dim * max_len])
            .map_err(|e| JsValue::from_str(&format!("Tensor error: {:?}", e)))?;
        
        // Forward pass
        let x = self.fc1.forward(&input)
            .map_err(|e| JsValue::from_str(&format!("FC1 error: {:?}", e)))?;
        let x = x.relu()
            .map_err(|e| JsValue::from_str(&format!("ReLU error: {:?}", e)))?;
        
        let x = self.fc2.forward(&x)
            .map_err(|e| JsValue::from_str(&format!("FC2 error: {:?}", e)))?;
        let x = x.relu()
            .map_err(|e| JsValue::from_str(&format!("ReLU error: {:?}", e)))?;
        
        let logits = self.fc3.forward(&x)
            .map_err(|e| JsValue::from_str(&format!("FC3 error: {:?}", e)))?;
        
        let probs = logits.softmax(1)
            .map_err(|e| JsValue::from_str(&format!("Softmax error: {:?}", e)))?;
        
        let probs_vec = probs.to_vec();
        
        Ok(SentimentResult {
            negative: probs_vec[0],
            neutral: probs_vec[1],
            positive: probs_vec[2],
            label: if probs_vec[2] > 0.6 {
                "positive".to_string()
            } else if probs_vec[0] > 0.6 {
                "negative".to_string()
            } else {
                "neutral".to_string()
            },
        })
    }
}

/// Sentiment analysis result
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct SentimentResult {
    pub negative: f32,
    pub neutral: f32,
    pub positive: f32,
    label: String,
}

#[wasm_bindgen]
impl SentimentResult {
    #[wasm_bindgen(getter)]
    pub fn label(&self) -> String {
        self.label.clone()
    }
}

/// Utility functions for data preprocessing
#[wasm_bindgen]
pub struct ImageProcessor;

#[wasm_bindgen]
impl ImageProcessor {
    /// Convert canvas image data to normalized grayscale
    #[wasm_bindgen]
    pub fn process_canvas_data(data: &[u8], width: u32, height: u32) -> Vec<f32> {
        let mut grayscale = Vec::with_capacity((width * height) as usize);
        
        // Convert RGBA to grayscale and normalize
        for i in (0..data.len()).step_by(4) {
            let r = data[i] as f32;
            let g = data[i + 1] as f32;
            let b = data[i + 2] as f32;
            
            // Grayscale conversion
            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            
            // Normalize to [0, 1]
            grayscale.push(gray / 255.0);
        }
        
        grayscale
    }
    
    /// Resize image to target dimensions (simple nearest neighbor)
    #[wasm_bindgen]
    pub fn resize_image(data: &[f32], width: u32, height: u32, 
                       target_width: u32, target_height: u32) -> Vec<f32> {
        let mut resized = Vec::with_capacity((target_width * target_height) as usize);
        
        let x_ratio = width as f32 / target_width as f32;
        let y_ratio = height as f32 / target_height as f32;
        
        for y in 0..target_height {
            for x in 0..target_width {
                let px = (x as f32 * x_ratio) as u32;
                let py = (y as f32 * y_ratio) as u32;
                
                let idx = (py * width + px) as usize;
                resized.push(data[idx]);
            }
        }
        
        resized
    }
}

/// Performance benchmarking utilities
#[wasm_bindgen]
pub struct Benchmark;

#[wasm_bindgen]
impl Benchmark {
    /// Run inference benchmark
    #[wasm_bindgen]
    pub fn inference_speed_test(iterations: u32) -> BenchmarkResult {
        let model = DigitClassifier::new();
        let dummy_input = vec![0.5; 28 * 28];
        
        let start = js_sys::Date::now();
        
        for _ in 0..iterations {
            let _ = model.predict(&dummy_input);
        }
        
        let elapsed = js_sys::Date::now() - start;
        let avg_time = elapsed / iterations as f64;
        
        BenchmarkResult {
            total_time_ms: elapsed,
            iterations,
            avg_inference_ms: avg_time,
            throughput: 1000.0 / avg_time,
        }
    }
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub total_time_ms: f64,
    pub iterations: u32,
    pub avg_inference_ms: f64,
    pub throughput: f64, // inferences per second
}

/// Demo function to show basic usage
#[wasm_bindgen]
pub fn demo() {
    log!("🚀 ToRSh WASM Demo");
    log!("==================");
    
    // Create model
    let classifier = DigitClassifier::new();
    let info = classifier.model_info();
    
    log!("Model: {}", info.name());
    log!("Parameters: {}", info.parameters);
    log!("Input shape: {:?}", info.input_shape());
    
    // Run inference
    let dummy_image = vec![0.0; 28 * 28];
    match classifier.predict(&dummy_image) {
        Ok(prediction) => {
            log!("Predicted digit: {}", prediction.digit);
            log!("Confidence: {:.2}%", prediction.confidence * 100.0);
        }
        Err(e) => {
            log!("Prediction error: {:?}", e);
        }
    }
    
    // Benchmark
    log!("\nRunning benchmark...");
    let bench = Benchmark::inference_speed_test(100);
    log!("Average inference time: {:.2}ms", bench.avg_inference_ms);
    log!("Throughput: {:.0} inferences/sec", bench.throughput);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_digit_classifier() {
        let model = DigitClassifier::new();
        let input = vec![0.5; 28 * 28];
        
        let result = model.predict(&input).unwrap();
        assert!(result.digit < 10);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.probabilities.len(), 10);
    }
    
    #[test]
    fn test_image_processor() {
        let rgba_data = vec![255, 128, 64, 255, 100, 150, 200, 255];
        let grayscale = ImageProcessor::process_canvas_data(&rgba_data, 2, 1);
        
        assert_eq!(grayscale.len(), 2);
        assert!(grayscale[0] > 0.0 && grayscale[0] <= 1.0);
    }
}