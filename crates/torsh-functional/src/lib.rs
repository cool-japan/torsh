//! Functional API for ToRSh
//!
//! This module provides functional operations similar to torch.functional,
//! including tensor manipulation, mathematical operations, and utilities.
//!
//! For comprehensive performance optimization guidance, see the separate
//! Performance Tuning Guide documentation.

#![allow(deprecated)]

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

// API patterns and conventions
pub mod api_patterns;

// Neural network functional operations
pub mod activation_lookup;
pub mod activations;
pub mod advanced_nn;
pub mod attention;
pub mod autograd;
pub mod conv;
pub mod dropout;
pub mod loss;
pub mod normalization;
pub mod pooling;
pub mod regularization;

// Tensor operations
pub mod advanced_manipulation;
pub mod broadcast;
pub mod data_ops;
pub mod fusion;
pub mod image;
pub mod interpolation;
pub mod lazy;
pub mod linalg;
pub mod manipulation;
pub mod math;
pub mod numerical;
pub mod optimization;
pub mod parallel;
pub mod profiling;
pub mod quantization;
pub mod random_ops;
pub mod reduction;
pub mod signal;
pub mod sparse;
pub mod special;
pub mod spectral;
pub mod tensor_ops;
pub mod type_promotion;
pub mod utils;
pub mod wavelet;

#[cfg(test)]
pub mod testing;

#[cfg(test)]
pub mod pytorch_correctness;

#[cfg(test)]
pub mod numerical_correctness;

#[cfg(test)]
pub mod property_based_tests;

#[cfg(test)]
pub mod edge_case_tests;

#[cfg(test)]
pub mod platform_tests;

// Re-exports for convenience

// Activation functions
pub use activations::{
    celu, elu, gelu, gumbel_softmax, hardshrink, hardsigmoid, hardsigmoid_v2, hardswish, hardtanh,
    leaky_relu, log_sigmoid, log_softmax, mish, prelu, relu, relu6, rrelu, selu, sigmoid, silu,
    softmax, softmin, softplus, softshrink, softsign, tanh, tanhshrink, threshold,
};

// Loss functions
pub use loss::{
    binary_cross_entropy, binary_cross_entropy_with_logits, contrastive_loss,
    cosine_embedding_loss, cross_entropy, cross_entropy_with_label_smoothing, ctc_loss, focal_loss,
    gaussian_nll_loss, hinge_embedding_loss, kl_div, l1_loss, margin_ranking_loss, mse_loss,
    multi_margin_loss, nll_loss, poisson_nll_loss, smooth_l1_loss, triplet_margin_loss,
    triplet_margin_with_distance_loss, ReductionType,
};

// Convolution operations
pub use conv::{
    conv1d, conv2d, conv3d, conv_output_size, conv_transpose1d, conv_transpose2d, conv_transpose3d,
    conv_transpose_output_size, depthwise_conv2d, fold, separable_conv2d, unfold,
};

// Pooling operations
pub use pooling::{
    adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool1d,
    adaptive_max_pool2d, adaptive_max_pool3d, avg_pool1d, avg_pool2d, avg_pool3d,
    fractional_max_pool2d, global_avg_pool, global_max_pool, learnable_pool2d, lp_pool1d,
    lp_pool2d, max_pool1d, max_pool2d, max_pool3d, max_unpool1d, max_unpool2d, max_unpool3d,
    spatial_pyramid_pool2d, stochastic_pool2d,
};

// Normalization functions
pub use normalization::{
    batch_norm, group_norm, instance_norm, layer_norm, local_response_norm, normalize, weight_norm,
};

// Dropout functions
pub use dropout::{
    alpha_dropout, dropout, dropout1d, dropout2d, dropout3d, feature_alpha_dropout,
    gaussian_dropout,
};

// Attention functions
pub use attention::{
    cross_attention, flash_attention, multi_head_attention, scaled_dot_product_attention,
    self_attention,
};

// Regularization functions
pub use regularization::{
    consistency_penalty, gradient_penalty, r1_gradient_penalty, r2_gradient_penalty,
    spectral_gradient_penalty,
};

// Advanced neural network operations
pub use advanced_nn::{
    // Data augmentation
    cutmix,
    // Neural Architecture Search operations
    darts_operation,
    decode_architecture,
    differentiable_augment,
    encode_architecture,
    // Other functions
    knowledge_distillation_loss,
    label_smoothing,
    mixup,
    mutate_architecture,
    predict_architecture_performance,
    // Normalization
    spectral_norm,
    temperature_scale,
    weight_standardization,
};

// Tensor operations
pub use broadcast::{broadcast_shapes, broadcast_tensors};
pub use linalg::{
    baddbmm, bmm, chain_matmul, cholesky, cond, det, eig, inv, lstsq, lu, matrix_rank, norm,
    pca_lowrank, pinv, qr, solve, svd, svd_lowrank, triangular_solve, NormOrd,
};
pub use manipulation::{
    atleast_1d, atleast_2d, atleast_3d, block_diag, cartesian_prod, chunk, dsplit, hsplit,
    meshgrid, split, tensor_split, tensordot, unravel_index, vsplit, SplitArg, TensorSplitArg,
};
pub use math::{cdist, einsum};
pub use reduction::{unique, unique_consecutive, UniqueResult};
pub use spectral::{istft, stft};
pub use tensor_ops::{
    cosine_similarity, embedding, linear, one_hot, pairwise_distance, pixel_shuffle,
    pixel_unshuffle,
};

// Image processing operations
pub use image::{
    affine_transform, closing, dilation, erosion, gaussian_blur, hsv_to_rgb, laplacian_filter,
    opening, resize, rgb_to_hsv, sobel_filter, InterpolationMode, SobelDirection,
};

// Signal processing
pub use signal::{
    correlate, filtfilt, frame, lfilter, overlap_add, periodogram, welch, window, CorrelationMode,
    PsdScaling, WindowType,
};

// Data operations
pub use data_ops::{
    bincount,
    histogram,
    histogram_with_edges,
    unique as unique_values, // Renamed to avoid conflict with reduction::unique
    value_counts,
};

// Random operations
pub use random_ops::{
    bernoulli, bernoulli_, exponential_, multinomial, normal_, rand, randint, randint_, randn,
    randperm, uniform_,
};

// Type promotion
pub use type_promotion::{
    can_cast_safely, common_dtype_for_operation, ensure_compatible_types, get_type_category,
    get_type_precision, promote_multiple_types, promote_scalar_type, promote_tensors,
    promote_types, reduction_result_type, result_type, TypeCategory,
};

// Operation fusion
pub use fusion::{
    analyze_fusion_opportunities, detect_fusible_patterns, fused_add_mul, fused_add_relu_mul,
    fused_batch_norm, fused_mul_add, fused_relu_add, fused_sigmoid_mul, fused_silu,
    fused_tanh_scale, AdaptiveFusionEngine, FusedOp, FusionOpportunity, FusionPerformance,
    OpFusionEngine, OpSequence,
};

// Special mathematical functions
pub use special::{
    acosh,
    airy_ai,
    asinh,
    atanh,

    bessel_i0,
    bessel_i1,
    bessel_iv,
    // Bessel functions
    bessel_j0,
    bessel_j1,
    bessel_jn,
    bessel_k0,
    bessel_k1,

    bessel_y0,
    bessel_y1,
    bessel_yn,
    beta,
    // Advanced special functions with scirs2-special integration
    betainc,
    dawson,
    digamma,
    // Error functions
    erf,
    erfc,
    erfcinv,

    erfcx,
    erfinv,
    expint,
    expm1,
    fresnel,
    fresnel_c,
    fresnel_s,
    // Gamma functions
    gamma,
    hypergeometric_1f1,
    kelvin_ber,
    lgamma,
    log1p,
    // Statistical functions
    logsumexp,
    multigammaln,

    normal_cdf,
    normal_icdf,

    polygamma,
    // Trigonometric and other special functions
    sinc,
    // Spherical Bessel functions
    spherical_j0,
    spherical_j1,
    spherical_jn,
    spherical_y0,
    spherical_y1,
    spherical_yn,

    voigt_profile,
};

// Wavelet transforms
pub use wavelet::{
    cwt, dwt_1d, dwt_2d, idwt_1d, idwt_2d, wavedec, waverec, WaveletMode, WaveletType,
};

// Interpolation functions
pub use interpolation::{
    barycentric_interp, grid_sample, interp1d, interp2d, lanczos_interp1d, spline1d,
    InterpolationMode as InterpMode,
};

// Numerical methods
pub use numerical::{
    adaptive_quad, bisection, cumtrapz, gaussian_quad, gradient, newton_raphson,
    partial_derivative, second_derivative, simps, trapz, DifferentiationMethod, IntegrationMethod,
};

// Optimization utilities
pub use optimization::{
    adam_optimizer,
    analyze_optimization_problem,
    auto_configure_optimization,
    backtracking_line_search,
    gradient_descent,
    lbfgs_optimizer,
    momentum_gradient_descent,
    wolfe_line_search,
    AdamParams,
    AdaptiveAlgorithmSelector,
    BFGSParams,
    BacktrackingParams,
    GradientDescentParams,
    LineSearchMethod,
    MomentumParams,
    OptimizationAlgorithm,
    // Adaptive algorithm selection
    TensorCharacteristics,
    WolfeParams,
};

// Lazy evaluation
pub use lazy::{
    lazy_ops::{execute, lazy, with_optimization},
    LazyBuilder, LazyContext, LazyOp, LazyTensor,
};

// Advanced tensor manipulation
pub use advanced_manipulation::{
    boolean_index, cat, masked_fill, pad, reshape, slice_with_step, squeeze, unsqueeze,
    where_tensor, PaddingMode,
};

// Quantization and compression
pub use quantization::{
    dynamic_quantize, fake_quantize, gradual_magnitude_prune, lottery_ticket_prune,
    magnitude_prune, quantization_error_analysis, uniform_dequantize, uniform_quantize,
    weight_clustering, QuantizationScheme, QuantizationType,
};

// Sparse operations
pub use sparse::{
    sparse_add, sparse_conv1d, sparse_conv2d, sparse_coo_tensor, sparse_eye, sparse_max,
    sparse_mean, sparse_min, sparse_mm, sparse_mul, sparse_sum, sparse_to_csr, sparse_transpose,
    SparseTensor,
};

// Custom autograd utilities
pub use autograd::{
    apply_custom_function, apply_custom_function_with_context, apply_registered_function,
    get_global_registry, register_custom_function, AutogradContext, AutogradRegistry,
    CustomAutogradFunction, CustomAutogradFunctionWithContext, ExpFunction, ScaledAddFunction,
    SquareFunction,
};

// Performance profiling and benchmarking
pub use profiling::{
    benchmark,
    global_profiler,
    profile_operation,
    run_performance_regression_test,
    BaselineSummary,
    BenchmarkConfig,
    BenchmarkResults,
    OperationMetrics,
    OperationSummary,
    // Performance regression testing
    PerformanceBaseline,
    PerformanceRegressionTester,
    Profiler,
    RegressionTestConfig,
    RegressionTestResult,
    SystemInfo,
};

// Utility functions and patterns
pub use utils::{
    apply_binary_elementwise, apply_conditional_elementwise, apply_elementwise_operation,
    calculate_pooling_output_size, calculate_pooling_output_size_2d,
    calculate_pooling_output_size_3d, create_tensor_like, function_context, safe_for_log, safe_log,
    safe_log_prob, validate_broadcastable_shapes, validate_dimension, validate_elementwise_shapes,
    validate_loss_params, validate_non_empty, validate_pooling_params, validate_positive,
    validate_range, validate_tensor_dims,
};

/// Align tensors to have the same number of dimensions
pub fn align_tensors(tensors: &[Tensor]) -> TorshResult<Vec<Tensor>> {
    if tensors.is_empty() {
        return Ok(vec![]);
    }

    // Find maximum number of dimensions
    let max_dims = tensors.iter().map(|t| t.shape().ndim()).max().unwrap_or(0);

    // Align all tensors
    let aligned: TorshResult<Vec<_>> = tensors
        .iter()
        .map(|t| {
            let current_dims = t.shape().ndim();
            if current_dims < max_dims {
                // Add dimensions of size 1 at the beginning
                let mut new_shape = vec![1; max_dims - current_dims];
                new_shape.extend(t.shape().dims());
                // Convert to i32 for view function
                let new_shape_i32: Vec<i32> = new_shape.iter().map(|&x| x as i32).collect();
                t.view(&new_shape_i32)
            } else {
                Ok(t.clone())
            }
        })
        .collect();

    aligned
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_align_tensors() {
        use crate::align_tensors;
        use torsh_tensor::creation::ones;

        // Test alignment functionality
        let t1 = ones(&[3, 4]).unwrap();
        let t2 = ones(&[4]).unwrap();
        let t3 = ones(&[2, 3, 4]).unwrap();

        let aligned = align_tensors(&[t1, t2, t3]).unwrap();

        // All should have 3 dimensions (max dimensions)
        assert_eq!(aligned[0].shape().ndim(), 3);
        assert_eq!(aligned[1].shape().ndim(), 3);
        assert_eq!(aligned[2].shape().ndim(), 3);

        // Check aligned shapes
        assert_eq!(aligned[0].shape().dims(), &[1, 3, 4]);
        assert_eq!(aligned[1].shape().dims(), &[1, 1, 4]);
        assert_eq!(aligned[2].shape().dims(), &[2, 3, 4]);
    }
}
