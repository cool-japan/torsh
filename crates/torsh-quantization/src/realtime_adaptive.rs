//! Real-time Adaptive Quantization with ML-based Optimization
//!
//! This module has been refactored into focused submodules while maintaining
//! full backward compatibility. All original exports are preserved.
//!
//! ## Modular Architecture (Phase 83 Refactoring)
//!
//! The original 1,695-line monolithic file has been systematically extracted into:
//! - `realtime_adaptive::config` - Configuration types and defaults (185 lines)
//! - `realtime_adaptive::ml_predictor` - ML parameter prediction and neural networks (280 lines)
//! - `realtime_adaptive::feature_extraction` - Comprehensive feature extraction (175 lines)
//! - `realtime_adaptive::quality_assessment` - Quality assessment and metrics (215 lines)
//! - `realtime_adaptive::pattern_analysis` - Workload pattern recognition (280 lines)
//! - `realtime_adaptive::optimization` - Multi-objective optimization (220 lines)
//! - `realtime_adaptive::engine` - Main adaptive quantization engine (185 lines)
//! - `realtime_adaptive::results` - Result types and report generation (250 lines)
//!
//! ## Features
//!
//! - **ML-based Parameter Prediction**: Neural networks predict optimal quantization parameters
//! - **Real-time Quality Assessment**: Continuous quality monitoring and adaptation
//! - **Workload Pattern Recognition**: Identifies and adapts to different computation patterns
//! - **Multi-objective Optimization**: Balances accuracy, performance, and energy consumption
//! - **Predictive Scaling**: Anticipates quantization needs based on input characteristics
//! - **Dynamic Bit-width Allocation**: Adaptive precision assignment based on layer importance
//!
//! ## Usage
//!
//! ```rust,no_run
//! use torsh_quantization::realtime_adaptive::*;
//! use torsh_tensor::tensor_1d;
//!
//! // Create adaptive quantization engine
//! let mut engine = AdaptiveQuantizationEngine::new(AdaptiveQuantConfig::default());
//!
//! // Create test tensor
//! let tensor = tensor_1d(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();
//!
//! // Perform adaptive quantization
//! let result = engine.adaptive_quantize(&tensor).unwrap();
//!
//! // Generate comprehensive report
//! println!("{}", result.generate_report());
//! ```

// Include the modular implementation
#[path = "realtime_adaptive/mod.rs"]
mod realtime_adaptive_impl;

// Re-export everything from the modular system for backward compatibility
pub use realtime_adaptive_impl::*;
