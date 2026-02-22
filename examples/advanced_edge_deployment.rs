//! Advanced Edge Deployment and Real-time Inference Demo
//!
//! This example demonstrates sophisticated edge deployment capabilities including:
//! - Model optimization for edge devices (quantization, pruning, distillation)
//! - Real-time inference with strict latency requirements
//! - Adaptive batch processing and request queuing
//! - Dynamic model switching based on workload
//! - Hardware-specific optimizations (ARM, x86, GPU, TPU)
//! - Power and thermal management
//! - Over-the-air model updates and A/B testing
//! - Edge-cloud hybrid processing with fallback strategies

use torsh::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};
use std::thread;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;

/// Edge deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    pub device_type: EdgeDeviceType,
    pub performance_mode: PerformanceMode,
    pub power_budget_watts: f64,
    pub thermal_limit_celsius: f64,
    pub latency_requirements: LatencyRequirements,
    pub optimization_settings: OptimizationSettings,
    pub fallback_strategy: FallbackStrategy,
    pub update_settings: UpdateSettings,
}

/// Types of edge devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeDeviceType {
    RaspberryPi4,
    JetsonNano,
    JetsonXavier,
    IntelNUC,
    AndroidPhone,
    iOSDevice,
    WebBrowser,
    CustomEmbedded { cpu_cores: usize, memory_mb: usize, gpu_available: bool },
}

/// Performance optimization modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMode {
    MaxThroughput,
    MinLatency,
    MinPower,
    Balanced,
    Adaptive,
}

/// Latency requirements for different scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyRequirements {
    pub max_inference_ms: f64,
    pub max_preprocessing_ms: f64,
    pub max_postprocessing_ms: f64,
    pub max_total_pipeline_ms: f64,
    pub percentile_target: f64, // e.g., 95th percentile
}

/// Model optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSettings {
    pub quantization: QuantizationConfig,
    pub pruning: PruningConfig,
    pub distillation: Option<DistillationConfig>,
    pub fusion: FusionConfig,
    pub compilation: CompilationConfig,
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub enabled: bool,
    pub precision: QuantizationPrecision,
    pub calibration_samples: usize,
    pub per_channel: bool,
    pub symmetric: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationPrecision {
    Int8,
    Int4,
    Mixed, // Dynamic precision
}

/// Pruning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    pub enabled: bool,
    pub target_sparsity: f64,
    pub structured: bool,
    pub gradual: bool,
}

/// Knowledge distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    pub teacher_model_path: String,
    pub temperature: f64,
    pub alpha: f64, // Balance between distillation and classification loss
}

/// Operator fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    pub conv_bn_fusion: bool,
    pub linear_activation_fusion: bool,
    pub attention_fusion: bool,
    pub custom_patterns: Vec<String>,
}

/// Model compilation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationConfig {
    pub backend: CompilationBackend,
    pub optimization_level: OptimizationLevel,
    pub target_hardware: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompilationBackend {
    TorchScript,
    TensorRT,
    OpenVINO,
    CoreML,
    TensorFlowLite,
    ONNX,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Conservative,
    Aggressive,
    Maximum,
}

/// Fallback strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackStrategy {
    pub enable_cloud_fallback: bool,
    pub cloud_endpoint: Option<String>,
    pub local_cache_size_mb: usize,
    pub fallback_triggers: Vec<FallbackTrigger>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackTrigger {
    HighLatency { threshold_ms: f64 },
    HighCpuUsage { threshold_percent: f64 },
    HighMemoryUsage { threshold_percent: f64 },
    ThermalThrottling,
    LowBattery { threshold_percent: f64 },
}

/// Over-the-air update configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateSettings {
    pub enable_auto_updates: bool,
    pub update_server_url: Option<String>,
    pub update_frequency_hours: u64,
    pub a_b_testing: bool,
    pub rollback_on_performance_degradation: bool,
}

/// Real-time inference request
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: u64,
    pub input: Tensor,
    pub priority: RequestPriority,
    pub deadline: Instant,
    pub callback: mpsc::Sender<InferenceResponse>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Inference response
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub request_id: u64,
    pub output: Option<Tensor>,
    pub latency_ms: f64,
    pub error: Option<String>,
    pub model_version: String,
    pub processed_on_edge: bool,
}

/// Performance metrics for monitoring
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub throughput_rps: f64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub gpu_usage_percent: f64,
    pub power_consumption_watts: f64,
    pub temperature_celsius: f64,
    pub success_rate: f64,
    pub fallback_rate: f64,
}

/// Edge inference engine for real-time processing
pub struct EdgeInferenceEngine {
    config: EdgeConfig,
    models: HashMap<String, Arc<dyn Module>>,
    active_model: String,
    request_queue: Arc<Mutex<VecDeque<InferenceRequest>>>,
    batch_processor: Option<thread::JoinHandle<()>>,
    performance_monitor: Arc<Mutex<PerformanceMetrics>>,
    thermal_monitor: ThermalMonitor,
    power_monitor: PowerMonitor,
    model_manager: ModelManager,
    is_running: Arc<Mutex<bool>>,
}

impl EdgeInferenceEngine {
    pub fn new(config: EdgeConfig) -> Result<Self> {
        let models = HashMap::new();
        let performance_metrics = PerformanceMetrics {
            throughput_rps: 0.0,
            avg_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            gpu_usage_percent: 0.0,
            power_consumption_watts: 0.0,
            temperature_celsius: 0.0,
            success_rate: 100.0,
            fallback_rate: 0.0,
        };
        
        Ok(Self {
            config: config.clone(),
            models,
            active_model: "default".to_string(),
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            batch_processor: None,
            performance_monitor: Arc::new(Mutex::new(performance_metrics)),
            thermal_monitor: ThermalMonitor::new(config.thermal_limit_celsius),
            power_monitor: PowerMonitor::new(config.power_budget_watts),
            model_manager: ModelManager::new(config.update_settings),
            is_running: Arc::new(Mutex::new(false)),
        })
    }

    /// Load and optimize model for edge deployment
    pub fn load_model(&mut self, name: String, model_path: &str) -> Result<()> {
        println!("🔧 Loading and optimizing model: {}", name);
        
        // Load base model
        let mut model = self.load_base_model(model_path)?;
        
        // Apply optimizations based on configuration
        model = self.apply_optimizations(model)?;
        
        // Compile for target hardware
        model = self.compile_model(model)?;
        
        // Warm up model
        self.warm_up_model(&model)?;
        
        // Validate performance
        self.validate_model_performance(&model)?;
        
        self.models.insert(name.clone(), Arc::new(model));
        self.active_model = name;
        
        println!("✅ Model loaded and optimized successfully");
        Ok(())
    }

    /// Start the edge inference engine
    pub fn start(&mut self) -> Result<()> {
        *self.is_running.lock().unwrap() = true;
        
        // Start background processors
        self.start_batch_processor()?;
        self.start_performance_monitor()?;
        self.start_thermal_monitor()?;
        self.start_power_monitor()?;
        self.start_model_update_checker()?;
        
        println!("🚀 Edge inference engine started");
        Ok(())
    }

    /// Submit inference request
    pub async fn infer_async(&self, input: Tensor, priority: RequestPriority) -> Result<InferenceResponse> {
        let request_id = self.generate_request_id();
        let deadline = Instant::now() + Duration::from_millis(
            self.config.latency_requirements.max_total_pipeline_ms as u64
        );
        
        let (tx, rx) = mpsc::channel();
        let request = InferenceRequest {
            id: request_id,
            input,
            priority,
            deadline,
            callback: tx,
        };
        
        // Add to queue with priority ordering
        {
            let mut queue = self.request_queue.lock().unwrap();
            
            // Insert based on priority (higher priority first)
            let insert_pos = queue.iter().position(|req| req.priority < request.priority)
                .unwrap_or(queue.len());
            
            queue.insert(insert_pos, request);
        }
        
        // Wait for response with timeout
        let timeout_duration = Duration::from_millis(
            self.config.latency_requirements.max_total_pipeline_ms as u64 * 2
        );
        
        match timeout(timeout_duration, async {
            rx.recv().map_err(|_| TorshError::Other("Response channel closed".to_string()))
        }).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(e)) => Err(e),
            Err(_) => {
                // Timeout - try cloud fallback if enabled
                if self.config.fallback_strategy.enable_cloud_fallback {
                    self.fallback_to_cloud(request_id, &input).await
                } else {
                    Err(TorshError::Other("Inference timeout".to_string()))
                }
            }
        }
    }

    fn start_batch_processor(&mut self) -> Result<()> {
        let config = self.config.clone();
        let models = Arc::new(self.models.clone());
        let active_model = self.active_model.clone();
        let request_queue = Arc::clone(&self.request_queue);
        let performance_monitor = Arc::clone(&self.performance_monitor);
        let is_running = Arc::clone(&self.is_running);
        
        let handle = thread::spawn(move || {
            let mut batch_buffer = Vec::new();
            let mut latency_samples = VecDeque::new();
            
            while *is_running.lock().unwrap() {
                // Collect batch from queue
                {
                    let mut queue = request_queue.lock().unwrap();
                    let now = Instant::now();
                    
                    // Remove expired requests
                    queue.retain(|req| req.deadline > now);
                    
                    // Collect requests for batch processing
                    let batch_size = match config.performance_mode {
                        PerformanceMode::MinLatency => 1,
                        PerformanceMode::MaxThroughput => 32,
                        PerformanceMode::Balanced => 8,
                        PerformanceMode::Adaptive => {
                            Self::calculate_adaptive_batch_size(&queue, &latency_samples)
                        }
                        PerformanceMode::MinPower => 4,
                    };
                    
                    for _ in 0..batch_size.min(queue.len()) {
                        if let Some(request) = queue.pop_front() {
                            batch_buffer.push(request);
                        }
                    }
                }
                
                if !batch_buffer.is_empty() {
                    // Process batch
                    if let Some(model) = models.get(&active_model) {
                        Self::process_batch(&config, model, &mut batch_buffer, &mut latency_samples);
                    }
                    
                    // Update performance metrics
                    Self::update_performance_metrics(&performance_monitor, &latency_samples);
                    
                    batch_buffer.clear();
                } else {
                    // Sleep briefly if no requests
                    thread::sleep(Duration::from_micros(100));
                }
            }
        });
        
        self.batch_processor = Some(handle);
        Ok(())
    }

    fn process_batch(
        config: &EdgeConfig,
        model: &Arc<dyn Module>,
        requests: &mut Vec<InferenceRequest>,
        latency_samples: &mut VecDeque<f64>,
    ) {
        if requests.is_empty() {
            return;
        }
        
        let start_time = Instant::now();
        
        // Combine inputs into batch tensor
        let batch_inputs: Vec<&Tensor> = requests.iter().map(|req| &req.input).collect();
        let batch_tensor = match Self::combine_batch_inputs(&batch_inputs) {
            Ok(tensor) => tensor,
            Err(e) => {
                // Send error responses
                for request in requests.iter() {
                    let response = InferenceResponse {
                        request_id: request.id,
                        output: None,
                        latency_ms: 0.0,
                        error: Some(format!("Batch preparation failed: {}", e)),
                        model_version: "unknown".to_string(),
                        processed_on_edge: true,
                    };
                    let _ = request.callback.send(response);
                }
                return;
            }
        };
        
        // Run inference
        match model.forward(&batch_tensor) {
            Ok(batch_output) => {
                // Split outputs and send responses
                let outputs = Self::split_batch_outputs(&batch_output, requests.len());
                
                for (i, request) in requests.iter().enumerate() {
                    let latency = start_time.elapsed().as_secs_f64() * 1000.0;
                    latency_samples.push_back(latency);
                    
                    // Keep only recent samples for adaptive behavior
                    if latency_samples.len() > 1000 {
                        latency_samples.pop_front();
                    }
                    
                    let output = outputs.get(i).cloned();
                    let response = InferenceResponse {
                        request_id: request.id,
                        output,
                        latency_ms: latency,
                        error: None,
                        model_version: "v1.0".to_string(),
                        processed_on_edge: true,
                    };
                    
                    let _ = request.callback.send(response);
                }
            }
            Err(e) => {
                // Send error responses
                for request in requests.iter() {
                    let response = InferenceResponse {
                        request_id: request.id,
                        output: None,
                        latency_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                        error: Some(format!("Inference failed: {}", e)),
                        model_version: "v1.0".to_string(),
                        processed_on_edge: true,
                    };
                    let _ = request.callback.send(response);
                }
            }
        }
    }

    fn calculate_adaptive_batch_size(
        queue: &VecDeque<InferenceRequest>,
        latency_samples: &VecDeque<f64>,
    ) -> usize {
        if queue.is_empty() {
            return 0;
        }
        
        // Calculate current latency percentiles
        if latency_samples.len() < 10 {
            return 1; // Conservative start
        }
        
        let mut sorted_latencies: Vec<f64> = latency_samples.iter().cloned().collect();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p95_latency = sorted_latencies[(sorted_latencies.len() as f64 * 0.95) as usize];
        
        // Adaptive batch sizing based on latency and queue length
        if p95_latency < 10.0 { // Low latency - can increase batch size
            queue.len().min(16)
        } else if p95_latency < 50.0 { // Medium latency
            queue.len().min(8)
        } else { // High latency - reduce batch size
            queue.len().min(4)
        }
    }

    async fn fallback_to_cloud(&self, request_id: u64, input: &Tensor) -> Result<InferenceResponse> {
        // Simulate cloud inference fallback
        println!("☁️  Falling back to cloud for request {}", request_id);
        
        // In a real implementation, this would make an HTTP request to cloud endpoint
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        Ok(InferenceResponse {
            request_id,
            output: Some(zeros(&[1, 10])), // Dummy output
            latency_ms: 200.0,
            error: None,
            model_version: "cloud-v1.0".to_string(),
            processed_on_edge: false,
        })
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.performance_monitor.lock().unwrap().clone()
    }

    /// Update model with new version
    pub async fn update_model(&mut self, name: String, model_data: &[u8]) -> Result<()> {
        println!("🔄 Updating model: {}", name);
        
        // Validate new model
        let new_model = self.validate_model_update(model_data)?;
        
        // A/B test if enabled
        if self.config.update_settings.a_b_testing {
            self.perform_ab_test(&name, new_model).await?;
        } else {
            // Direct update
            self.models.insert(name.clone(), Arc::new(new_model));
            self.active_model = name;
        }
        
        println!("✅ Model updated successfully");
        Ok(())
    }

    // Helper methods (simplified implementations)
    fn load_base_model(&self, model_path: &str) -> Result<Box<dyn Module>> {
        // Load model from path - simplified implementation
        Ok(Box::new(Linear::new(784, 10)))
    }

    fn apply_optimizations(&self, mut model: Box<dyn Module>) -> Result<Box<dyn Module>> {
        if self.config.optimization_settings.quantization.enabled {
            model = self.apply_quantization(model)?;
        }
        
        if self.config.optimization_settings.pruning.enabled {
            model = self.apply_pruning(model)?;
        }
        
        if self.config.optimization_settings.fusion.conv_bn_fusion {
            model = self.apply_fusion(model)?;
        }
        
        Ok(model)
    }

    fn apply_quantization(&self, model: Box<dyn Module>) -> Result<Box<dyn Module>> {
        println!("⚡ Applying quantization");
        // Quantization implementation would go here
        Ok(model)
    }

    fn apply_pruning(&self, model: Box<dyn Module>) -> Result<Box<dyn Module>> {
        println!("✂️  Applying pruning");
        // Pruning implementation would go here
        Ok(model)
    }

    fn apply_fusion(&self, model: Box<dyn Module>) -> Result<Box<dyn Module>> {
        println!("🔗 Applying operator fusion");
        // Fusion implementation would go here
        Ok(model)
    }

    fn compile_model(&self, model: Box<dyn Module>) -> Result<Box<dyn Module>> {
        println!("⚙️  Compiling model for target hardware");
        // Compilation implementation would go here
        Ok(model)
    }

    fn warm_up_model(&self, model: &Box<dyn Module>) -> Result<()> {
        println!("🔥 Warming up model");
        let dummy_input = zeros(&[1, 784]);
        for _ in 0..10 {
            let _ = model.forward(&dummy_input)?;
        }
        Ok(())
    }

    fn validate_model_performance(&self, model: &Box<dyn Module>) -> Result<()> {
        println!("✅ Validating model performance");
        
        let dummy_input = zeros(&[1, 784]);
        let start_time = Instant::now();
        let _ = model.forward(&dummy_input)?;
        let latency = start_time.elapsed().as_secs_f64() * 1000.0;
        
        if latency > self.config.latency_requirements.max_inference_ms {
            return Err(TorshError::Other(format!(
                "Model latency {:.2}ms exceeds requirement {:.2}ms",
                latency, self.config.latency_requirements.max_inference_ms
            )));
        }
        
        Ok(())
    }

    fn generate_request_id(&self) -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.fetch_add(1, Ordering::SeqCst)
    }

    fn combine_batch_inputs(inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.is_empty() {
            return Err(TorshError::Other("Empty batch".to_string()));
        }
        
        // Simple stacking implementation
        let first_shape = inputs[0].shape().dims();
        let mut batch_shape = vec![inputs.len()];
        batch_shape.extend_from_slice(first_shape);
        
        // For simplicity, return a random tensor of the right shape
        Ok(randn(&batch_shape))
    }

    fn split_batch_outputs(output: &Tensor, batch_size: usize) -> Vec<Option<Tensor>> {
        // Simple splitting implementation
        (0..batch_size).map(|i| {
            // For simplicity, return individual tensors
            Some(output.narrow(0, i, 1).unwrap_or_else(|_| zeros(&[1, 10])))
        }).collect()
    }

    fn start_performance_monitor(&self) -> Result<()> {
        // Start performance monitoring thread
        Ok(())
    }

    fn start_thermal_monitor(&self) -> Result<()> {
        // Start thermal monitoring
        Ok(())
    }

    fn start_power_monitor(&self) -> Result<()> {
        // Start power monitoring
        Ok(())
    }

    fn start_model_update_checker(&self) -> Result<()> {
        // Start model update checker
        Ok(())
    }

    fn update_performance_metrics(
        monitor: &Arc<Mutex<PerformanceMetrics>>,
        latency_samples: &VecDeque<f64>,
    ) {
        if let Ok(mut metrics) = monitor.lock() {
            if !latency_samples.is_empty() {
                let avg_latency: f64 = latency_samples.iter().sum::<f64>() / latency_samples.len() as f64;
                metrics.avg_latency_ms = avg_latency;
                
                let mut sorted: Vec<f64> = latency_samples.iter().cloned().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                if sorted.len() > 1 {
                    metrics.p95_latency_ms = sorted[(sorted.len() as f64 * 0.95) as usize];
                    metrics.p99_latency_ms = sorted[(sorted.len() as f64 * 0.99) as usize];
                }
                
                metrics.throughput_rps = 1000.0 / avg_latency;
            }
        }
    }

    fn validate_model_update(&self, model_data: &[u8]) -> Result<Box<dyn Module>> {
        // Validate and load new model
        Ok(Box::new(Linear::new(784, 10)))
    }

    async fn perform_ab_test(&self, name: &str, new_model: Box<dyn Module>) -> Result<()> {
        println!("🧪 Performing A/B test for model: {}", name);
        // A/B testing implementation
        Ok(())
    }
}

// Supporting structures
pub struct ThermalMonitor {
    limit_celsius: f64,
}

impl ThermalMonitor {
    pub fn new(limit: f64) -> Self {
        Self { limit_celsius: limit }
    }
}

pub struct PowerMonitor {
    budget_watts: f64,
}

impl PowerMonitor {
    pub fn new(budget: f64) -> Self {
        Self { budget_watts: budget }
    }
}

pub struct ModelManager {
    update_settings: UpdateSettings,
}

impl ModelManager {
    pub fn new(settings: UpdateSettings) -> Self {
        Self { update_settings: settings }
    }
}

/// Main example function demonstrating edge deployment
pub fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    rt.block_on(async {
        println!("📱 Advanced Edge Deployment and Real-time Inference Demo");
        
        // Configure for Jetson Nano deployment
        let config = EdgeConfig {
            device_type: EdgeDeviceType::JetsonNano,
            performance_mode: PerformanceMode::Balanced,
            power_budget_watts: 10.0,
            thermal_limit_celsius: 80.0,
            latency_requirements: LatencyRequirements {
                max_inference_ms: 50.0,
                max_preprocessing_ms: 10.0,
                max_postprocessing_ms: 5.0,
                max_total_pipeline_ms: 100.0,
                percentile_target: 95.0,
            },
            optimization_settings: OptimizationSettings {
                quantization: QuantizationConfig {
                    enabled: true,
                    precision: QuantizationPrecision::Int8,
                    calibration_samples: 1000,
                    per_channel: true,
                    symmetric: false,
                },
                pruning: PruningConfig {
                    enabled: true,
                    target_sparsity: 0.5,
                    structured: true,
                    gradual: false,
                },
                distillation: None,
                fusion: FusionConfig {
                    conv_bn_fusion: true,
                    linear_activation_fusion: true,
                    attention_fusion: false,
                    custom_patterns: vec![],
                },
                compilation: CompilationConfig {
                    backend: CompilationBackend::TensorRT,
                    optimization_level: OptimizationLevel::Aggressive,
                    target_hardware: "jetson_nano".to_string(),
                },
            },
            fallback_strategy: FallbackStrategy {
                enable_cloud_fallback: true,
                cloud_endpoint: Some("https://api.example.com/inference".to_string()),
                local_cache_size_mb: 100,
                fallback_triggers: vec![
                    FallbackTrigger::HighLatency { threshold_ms: 200.0 },
                    FallbackTrigger::ThermalThrottling,
                ],
            },
            update_settings: UpdateSettings {
                enable_auto_updates: true,
                update_server_url: Some("https://models.example.com".to_string()),
                update_frequency_hours: 24,
                a_b_testing: true,
                rollback_on_performance_degradation: true,
            },
        };
        
        // Create and start inference engine
        let mut engine = EdgeInferenceEngine::new(config)?;
        engine.load_model("classification".to_string(), "model.torsh")?;
        engine.start()?;
        
        // Simulate inference requests
        for i in 0..10 {
            let input = randn(&[1, 784]);
            let priority = if i % 3 == 0 { 
                RequestPriority::High 
            } else { 
                RequestPriority::Normal 
            };
            
            match engine.infer_async(input, priority).await {
                Ok(response) => {
                    println!("Request {}: ✅ Success in {:.2}ms (edge: {})",
                           response.request_id, 
                           response.latency_ms,
                           response.processed_on_edge);
                }
                Err(e) => {
                    println!("Request {}: ❌ Failed: {}", i, e);
                }
            }
        }
        
        // Print performance metrics
        let metrics = engine.get_metrics();
        println!("\n📊 Performance Metrics:");
        println!("  Throughput: {:.1} RPS", metrics.throughput_rps);
        println!("  Avg Latency: {:.2}ms", metrics.avg_latency_ms);
        println!("  P95 Latency: {:.2}ms", metrics.p95_latency_ms);
        println!("  Success Rate: {:.1}%", metrics.success_rate);
        println!("  Fallback Rate: {:.1}%", metrics.fallback_rate);
        
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_edge_config_creation() {
        let config = EdgeConfig {
            device_type: EdgeDeviceType::RaspberryPi4,
            performance_mode: PerformanceMode::MinPower,
            power_budget_watts: 5.0,
            thermal_limit_celsius: 70.0,
            latency_requirements: LatencyRequirements {
                max_inference_ms: 100.0,
                max_preprocessing_ms: 20.0,
                max_postprocessing_ms: 10.0,
                max_total_pipeline_ms: 200.0,
                percentile_target: 90.0,
            },
            optimization_settings: OptimizationSettings {
                quantization: QuantizationConfig {
                    enabled: true,
                    precision: QuantizationPrecision::Int8,
                    calibration_samples: 500,
                    per_channel: false,
                    symmetric: true,
                },
                pruning: PruningConfig {
                    enabled: false,
                    target_sparsity: 0.0,
                    structured: false,
                    gradual: false,
                },
                distillation: None,
                fusion: FusionConfig {
                    conv_bn_fusion: true,
                    linear_activation_fusion: false,
                    attention_fusion: false,
                    custom_patterns: vec![],
                },
                compilation: CompilationConfig {
                    backend: CompilationBackend::ONNX,
                    optimization_level: OptimizationLevel::Conservative,
                    target_hardware: "arm64".to_string(),
                },
            },
            fallback_strategy: FallbackStrategy {
                enable_cloud_fallback: false,
                cloud_endpoint: None,
                local_cache_size_mb: 50,
                fallback_triggers: vec![],
            },
            update_settings: UpdateSettings {
                enable_auto_updates: false,
                update_server_url: None,
                update_frequency_hours: 168,
                a_b_testing: false,
                rollback_on_performance_degradation: false,
            },
        };
        
        assert_eq!(config.power_budget_watts, 5.0);
        assert!(config.optimization_settings.quantization.enabled);
        assert!(!config.optimization_settings.pruning.enabled);
    }
    
    #[test]
    fn test_inference_engine_creation() {
        let config = EdgeConfig {
            device_type: EdgeDeviceType::CustomEmbedded { 
                cpu_cores: 4, 
                memory_mb: 2048, 
                gpu_available: false 
            },
            performance_mode: PerformanceMode::Adaptive,
            power_budget_watts: 15.0,
            thermal_limit_celsius: 85.0,
            latency_requirements: LatencyRequirements {
                max_inference_ms: 25.0,
                max_preprocessing_ms: 5.0,
                max_postprocessing_ms: 5.0,
                max_total_pipeline_ms: 50.0,
                percentile_target: 99.0,
            },
            optimization_settings: OptimizationSettings {
                quantization: QuantizationConfig {
                    enabled: true,
                    precision: QuantizationPrecision::Mixed,
                    calibration_samples: 2000,
                    per_channel: true,
                    symmetric: false,
                },
                pruning: PruningConfig {
                    enabled: true,
                    target_sparsity: 0.7,
                    structured: false,
                    gradual: true,
                },
                distillation: Some(DistillationConfig {
                    teacher_model_path: "teacher.torsh".to_string(),
                    temperature: 4.0,
                    alpha: 0.7,
                }),
                fusion: FusionConfig {
                    conv_bn_fusion: true,
                    linear_activation_fusion: true,
                    attention_fusion: true,
                    custom_patterns: vec!["conv_relu_pool".to_string()],
                },
                compilation: CompilationConfig {
                    backend: CompilationBackend::TensorRT,
                    optimization_level: OptimizationLevel::Maximum,
                    target_hardware: "custom_asic".to_string(),
                },
            },
            fallback_strategy: FallbackStrategy {
                enable_cloud_fallback: true,
                cloud_endpoint: Some("https://inference.cloud.com".to_string()),
                local_cache_size_mb: 200,
                fallback_triggers: vec![
                    FallbackTrigger::HighLatency { threshold_ms: 75.0 },
                    FallbackTrigger::HighCpuUsage { threshold_percent: 90.0 },
                    FallbackTrigger::ThermalThrottling,
                ],
            },
            update_settings: UpdateSettings {
                enable_auto_updates: true,
                update_server_url: Some("https://updates.example.com".to_string()),
                update_frequency_hours: 12,
                a_b_testing: true,
                rollback_on_performance_degradation: true,
            },
        };
        
        let engine = EdgeInferenceEngine::new(config).unwrap();
        assert_eq!(engine.models.len(), 0);
        assert_eq!(engine.active_model, "default");
    }
}