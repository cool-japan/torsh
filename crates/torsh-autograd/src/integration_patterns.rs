//! Integration Patterns and Best Practices
//!
//! This module provides comprehensive documentation and examples for integrating
//! with the torsh-autograd system, including SciRS2 integration patterns,
//! performance optimization guidelines, and best practices for robust autograd usage.

use crate::error_handling::{AutogradError, AutogradResult};
use std::fmt;

/// Documentation for integration patterns and best practices
pub struct IntegrationPatterns;

impl IntegrationPatterns {
    /// Get comprehensive integration documentation
    pub fn get_documentation() -> IntegrationDocumentation {
        IntegrationDocumentation::new()
    }

    /// Get specific pattern documentation by category
    pub fn get_pattern_docs(category: PatternCategory) -> PatternDocumentation {
        PatternDocumentation::for_category(category)
    }

    /// Get troubleshooting guide
    pub fn get_troubleshooting_guide() -> TroubleshootingGuide {
        TroubleshootingGuide::new()
    }

    /// Get migration guide
    pub fn get_migration_guide() -> MigrationGuide {
        MigrationGuide::new()
    }
}

/// Categories of integration patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PatternCategory {
    SciRS2Integration,
    PerformanceOptimization,
    ErrorHandling,
    Testing,
    ResourceManagement,
    DistributedTraining,
    CustomOperations,
    DebuggingAndProfiling,
}

impl fmt::Display for PatternCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternCategory::SciRS2Integration => write!(f, "SciRS2 Integration"),
            PatternCategory::PerformanceOptimization => write!(f, "Performance Optimization"),
            PatternCategory::ErrorHandling => write!(f, "Error Handling"),
            PatternCategory::Testing => write!(f, "Testing"),
            PatternCategory::ResourceManagement => write!(f, "Resource Management"),
            PatternCategory::DistributedTraining => write!(f, "Distributed Training"),
            PatternCategory::CustomOperations => write!(f, "Custom Operations"),
            PatternCategory::DebuggingAndProfiling => write!(f, "Debugging and Profiling"),
        }
    }
}

/// Comprehensive integration documentation
pub struct IntegrationDocumentation {
    pub scirs2_patterns: SciRS2IntegrationPatterns,
    pub performance_patterns: PerformancePatterns,
    pub error_handling_patterns: ErrorHandlingPatterns,
    pub testing_patterns: TestingPatterns,
    pub resource_patterns: ResourceManagementPatterns,
    pub distributed_patterns: DistributedTrainingPatterns,
    pub custom_operation_patterns: CustomOperationPatterns,
    pub debugging_patterns: DebuggingPatterns,
}

impl IntegrationDocumentation {
    pub fn new() -> Self {
        Self {
            scirs2_patterns: SciRS2IntegrationPatterns::new(),
            performance_patterns: PerformancePatterns::new(),
            error_handling_patterns: ErrorHandlingPatterns::new(),
            testing_patterns: TestingPatterns::new(),
            resource_patterns: ResourceManagementPatterns::new(),
            distributed_patterns: DistributedTrainingPatterns::new(),
            custom_operation_patterns: CustomOperationPatterns::new(),
            debugging_patterns: DebuggingPatterns::new(),
        }
    }

    /// Print complete documentation to console
    pub fn print_all(&self) {
        println!("# ToRSh Autograd Integration Patterns and Best Practices\n");

        self.scirs2_patterns.print();
        self.performance_patterns.print();
        self.error_handling_patterns.print();
        self.testing_patterns.print();
        self.resource_patterns.print();
        self.distributed_patterns.print();
        self.custom_operation_patterns.print();
        self.debugging_patterns.print();
    }

    /// Export documentation to markdown file
    pub fn export_to_markdown(&self, file_path: &std::path::Path) -> AutogradResult<()> {
        let markdown = self.to_markdown();
        std::fs::write(file_path, markdown).map_err(|e| {
            AutogradError::gradient_computation(
                "documentation_write",
                format!("Failed to write documentation: {}", e),
            )
        })?;
        Ok(())
    }

    /// Convert to markdown format
    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();

        markdown.push_str("# ToRSh Autograd Integration Patterns and Best Practices\n\n");
        markdown.push_str("This document provides comprehensive guidance for integrating with and optimizing the ToRSh autograd system.\n\n");

        markdown.push_str(&self.scirs2_patterns.to_markdown());
        markdown.push_str(&self.performance_patterns.to_markdown());
        markdown.push_str(&self.error_handling_patterns.to_markdown());
        markdown.push_str(&self.testing_patterns.to_markdown());
        markdown.push_str(&self.resource_patterns.to_markdown());
        markdown.push_str(&self.distributed_patterns.to_markdown());
        markdown.push_str(&self.custom_operation_patterns.to_markdown());
        markdown.push_str(&self.debugging_patterns.to_markdown());

        markdown
    }
}

/// SciRS2 integration patterns and best practices
pub struct SciRS2IntegrationPatterns {
    pub patterns: Vec<Pattern>,
}

impl SciRS2IntegrationPatterns {
    pub fn new() -> Self {
        let mut patterns = Vec::new();

        patterns.push(Pattern {
            name: "Basic SciRS2 Integration".to_string(),
            description: "How to integrate with SciRS2 autograd system".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Get the global SciRS2 adapter
let adapter = get_global_adapter();

// Create a gradient tensor
let input_data = vec![1.0, 2.0, 3.0, 4.0];
let input_shape = vec![2, 2];
let tensor = adapter.create_gradient_tensor(&input_data, &input_shape)?;

// Compute gradients
let gradients = adapter.backward(&tensor)?;
"#
            .to_string(),
            best_practices: vec![
                "Always check if SciRS2 is available before using advanced features".to_string(),
                "Use the global adapter for consistent behavior across your application"
                    .to_string(),
                "Handle fallback scenarios gracefully when SciRS2 is unavailable".to_string(),
            ],
            common_pitfalls: vec![
                "Assuming SciRS2 is always available".to_string(),
                "Not handling version compatibility issues".to_string(),
                "Ignoring fallback performance implications".to_string(),
            ],
        });

        patterns.push(Pattern {
            name: "Version Compatibility Checking".to_string(),
            description: "How to check and handle SciRS2 version compatibility".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Check SciRS2 version compatibility
let migration_helper = SciRS2MigrationHelper::new();
let current_version = SciRS2Version::from_string("0.1.0-beta.2")?;

if migration_helper.check_version_compatibility(&current_version)? {
    // Use SciRS2 features
    let adapter = SciRS2AutogradAdapter::new()?;
    // ... continue with SciRS2 operations
} else {
    // Use fallback implementation
    tracing::warn!("SciRS2 version incompatible, using fallback");
    // ... use manual gradient tracking
}
"#
            .to_string(),
            best_practices: vec![
                "Always check version compatibility before using SciRS2 features".to_string(),
                "Implement graceful degradation for unsupported versions".to_string(),
                "Log version compatibility issues for debugging".to_string(),
            ],
            common_pitfalls: vec![
                "Hard-coding version dependencies".to_string(),
                "Not providing fallback for older versions".to_string(),
                "Failing silently on version mismatches".to_string(),
            ],
        });

        patterns.push(Pattern {
            name: "Fallback Implementation".to_string(),
            description: "How to implement robust fallback when SciRS2 is unavailable".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Try SciRS2 first, fallback to manual implementation
let adapter = SciRS2AutogradAdapter::new();
let result = if adapter.is_available() {
    // Use SciRS2 implementation
    adapter.compute_gradient("operation", &input_data, &input_shape)?
} else {
    // Fallback to manual gradient computation
    tracing::info!("Using fallback gradient computation");
    manual_gradient_computation(&input_data, &input_shape)?
};

fn manual_gradient_computation(data: &[f64], shape: &[usize]) -> AutogradResult<Vec<f64>> {
    // Implement manual gradient computation
    Ok(vec![1.0; data.len()]) // Simplified example
}
"#
            .to_string(),
            best_practices: vec![
                "Always provide fallback implementations for critical operations".to_string(),
                "Test fallback paths regularly to ensure they work".to_string(),
                "Document performance differences between SciRS2 and fallback".to_string(),
            ],
            common_pitfalls: vec![
                "Not implementing fallback for all operations".to_string(),
                "Fallback implementations with poor performance".to_string(),
                "Inconsistent behavior between SciRS2 and fallback paths".to_string(),
            ],
        });

        Self { patterns }
    }

    pub fn print(&self) {
        println!("## SciRS2 Integration Patterns\n");
        for pattern in &self.patterns {
            pattern.print();
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        markdown.push_str("## SciRS2 Integration Patterns\n\n");

        for pattern in &self.patterns {
            markdown.push_str(&pattern.to_markdown());
        }

        markdown
    }
}

/// Performance optimization patterns
pub struct PerformancePatterns {
    pub patterns: Vec<Pattern>,
}

impl PerformancePatterns {
    pub fn new() -> Self {
        let mut patterns = Vec::new();

        patterns.push(Pattern {
            name: "Gradient Checkpointing".to_string(),
            description: "Optimize memory usage with gradient checkpointing".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Configure gradient checkpointing
let mut checkpointer = GradientCheckpointer::new();
checkpointer.set_strategy(CheckpointStrategy::Adaptive);

// Use checkpointing in forward pass
let checkpoint_guard = checkpointer.create_checkpoint("layer_1")?;
let intermediate_result = forward_computation(&input)?;
checkpoint_guard.save(&intermediate_result)?;

// Checkpoints will be automatically restored during backward pass
let gradients = backward_computation(&intermediate_result)?;
"#
            .to_string(),
            best_practices: vec![
                "Use adaptive checkpointing for optimal memory-compute tradeoff".to_string(),
                "Checkpoint at layer boundaries for best efficiency".to_string(),
                "Monitor memory usage to tune checkpointing frequency".to_string(),
            ],
            common_pitfalls: vec![
                "Over-checkpointing leading to performance degradation".to_string(),
                "Under-checkpointing causing memory issues".to_string(),
                "Not considering checkpoint overhead in performance calculations".to_string(),
            ],
        });

        patterns.push(Pattern {
            name: "SIMD Optimization".to_string(),
            description: "Leverage SIMD operations for better performance".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Enable SIMD optimizations
let simd_config = SIMDConfig::new()
    .with_auto_vectorization(true)
    .with_target_architecture(TargetArch::Auto);

// Use SIMD-optimized operations
let result = with_simd_optimization(&simd_config, || {
    // Your tensor operations here
    tensor_a.add(&tensor_b)
})?;

// For custom operations, use SIMD primitives directly
use torsh_autograd::simd_ops::*;
let simd_result = simd_dot_product(&vector_a, &vector_b)?;
"#
            .to_string(),
            best_practices: vec![
                "Enable auto-vectorization for compatible operations".to_string(),
                "Use SIMD-optimized primitives for custom operations".to_string(),
                "Profile SIMD performance to ensure benefits".to_string(),
            ],
            common_pitfalls: vec![
                "Assuming all operations benefit from SIMD".to_string(),
                "Not considering data alignment requirements".to_string(),
                "Mixing SIMD and non-SIMD operations inefficiently".to_string(),
            ],
        });

        patterns.push(Pattern {
            name: "Memory Pool Optimization".to_string(),
            description: "Use memory pools for efficient buffer management".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Configure memory pool
let pool_config = MemoryPoolConfig::new()
    .with_initial_size(1024 * 1024) // 1MB
    .with_growth_factor(2.0)
    .with_max_size(1024 * 1024 * 1024); // 1GB

let memory_pool = MemoryPool::with_config(pool_config)?;

// Use pooled memory for temporary buffers
let buffer = memory_pool.allocate(tensor.byte_size())?;
// ... use buffer for computation
// Buffer is automatically returned to pool when dropped
"#
            .to_string(),
            best_practices: vec![
                "Use memory pools for frequently allocated/deallocated buffers".to_string(),
                "Configure pool sizes based on typical workload patterns".to_string(),
                "Monitor pool usage and fragmentation".to_string(),
            ],
            common_pitfalls: vec![
                "Creating too many small pools".to_string(),
                "Not accounting for memory fragmentation".to_string(),
                "Pool sizes that don't match usage patterns".to_string(),
            ],
        });

        Self { patterns }
    }

    pub fn print(&self) {
        println!("## Performance Optimization Patterns\n");
        for pattern in &self.patterns {
            pattern.print();
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        markdown.push_str("## Performance Optimization Patterns\n\n");

        for pattern in &self.patterns {
            markdown.push_str(&pattern.to_markdown());
        }

        markdown
    }
}

/// Error handling patterns
pub struct ErrorHandlingPatterns {
    pub patterns: Vec<Pattern>,
}

impl ErrorHandlingPatterns {
    pub fn new() -> Self {
        let mut patterns = Vec::new();

        patterns.push(Pattern {
            name: "Exception Safety with Transactions".to_string(),
            description: "Use transactions for exception-safe autograd operations".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Create a transaction for exception safety
let executor = get_global_executor();
let transaction = executor.begin_transaction(Some(ExceptionSafetyLevel::Strong))?;

{
    let mut tx = transaction.lock()?;

    // Add operations to transaction
    tx.add_operation(TransactionOperation::new(
        "gradient_computation".to_string(),
        || compute_gradients(&tensor)
    ));

    // Commit all operations atomically
    tx.commit()?;
}
"#
            .to_string(),
            best_practices: vec![
                "Use Strong exception safety for critical operations".to_string(),
                "Group related operations in single transactions".to_string(),
                "Always handle transaction rollback scenarios".to_string(),
            ],
            common_pitfalls: vec![
                "Not handling transaction deadlocks".to_string(),
                "Overly large transactions that reduce concurrency".to_string(),
                "Forgetting to commit or rollback transactions".to_string(),
            ],
        });

        patterns.push(Pattern {
            name: "Graceful Degradation".to_string(),
            description: "Handle unsupported operations gracefully".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Use graceful degradation for unsupported operations
let degradation_manager = get_global_degradation_manager();

let result = degradation_manager.execute_with_degradation("advanced_operation", || {
    // Try advanced operation
    advanced_gradient_computation(&tensor)
})?;

// Register custom fallback for specific operations
degradation_manager.register_degradation_strategy(
    "quantum_gradient".to_string(),
    DegradationStrategy::FallbackImplementation {
        fallback_name: "classical_gradient".to_string(),
        performance_impact: 0.2,
        accuracy_impact: 0.0,
    }
);
"#
            .to_string(),
            best_practices: vec![
                "Register fallbacks for all advanced operations".to_string(),
                "Provide clear error messages with suggested alternatives".to_string(),
                "Monitor degradation events for system health".to_string(),
            ],
            common_pitfalls: vec![
                "Not providing fallbacks for critical operations".to_string(),
                "Degradation strategies that significantly impact performance".to_string(),
                "Silent degradation without user notification".to_string(),
            ],
        });

        patterns.push(Pattern {
            name: "Automatic Error Recovery".to_string(),
            description: "Implement automatic recovery from transient failures".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Configure automatic error recovery
let recovery_system = get_global_recovery();

// Use recovery wrapper for operations
let result = with_error_recovery("gradient_computation", || {
    compute_gradients_with_potential_failure(&tensor)
})?;

// Configure recovery strategies for specific error types
recovery_system.configure_strategy(
    TransientFailureType::MemoryPressure,
    RecoveryStrategy::GracefulDegradation {
        precision_reduction: 0.1,
        simplification_level: 1,
        max_retries: 3,
    }
);
"#
            .to_string(),
            best_practices: vec![
                "Use exponential backoff for transient network failures".to_string(),
                "Implement circuit breakers for unreliable dependencies".to_string(),
                "Log recovery events for system monitoring".to_string(),
            ],
            common_pitfalls: vec![
                "Infinite retry loops without backoff".to_string(),
                "Not distinguishing between transient and permanent failures".to_string(),
                "Recovery strategies that mask underlying issues".to_string(),
            ],
        });

        Self { patterns }
    }

    pub fn print(&self) {
        println!("## Error Handling Patterns\n");
        for pattern in &self.patterns {
            pattern.print();
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        markdown.push_str("## Error Handling Patterns\n\n");

        for pattern in &self.patterns {
            markdown.push_str(&pattern.to_markdown());
        }

        markdown
    }
}

/// Testing patterns and strategies
pub struct TestingPatterns {
    pub patterns: Vec<Pattern>,
}

impl TestingPatterns {
    pub fn new() -> Self {
        let mut patterns = Vec::new();

        patterns.push(Pattern {
            name: "Gradient Verification Testing".to_string(),
            description: "Test gradient correctness with numerical verification".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

#[test]
fn test_gradient_correctness() {
    let verifier = CrossFrameworkVerifier::with_default_tolerance();

    // Test against reference implementation
    let input_data = vec![1.0, 2.0, 3.0];
    let torsh_gradients = compute_torsh_gradients(&input_data)?;
    let reference_gradients = compute_reference_gradients(&input_data)?;

    let result = verifier.compare_gradients(
        "test_operation".to_string(),
        &torsh_gradients,
        &reference_gradients
    )?;

    assert!(result.passed_tolerance);
    assert!(result.correlation_coefficient > 0.99);
}
"#
            .to_string(),
            best_practices: vec![
                "Always test gradients against known reference implementations".to_string(),
                "Use property-based testing for gradient properties".to_string(),
                "Test both forward and backward passes".to_string(),
            ],
            common_pitfalls: vec![
                "Only testing with simple input data".to_string(),
                "Not testing edge cases (zeros, infinities, NaNs)".to_string(),
                "Insufficient tolerance checking for numerical precision".to_string(),
            ],
        });

        patterns.push(Pattern {
            name: "Integration Testing".to_string(),
            description: "Test SciRS2 integration with comprehensive test suites".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

#[test]
fn test_scirs2_integration() {
    let test_suite = run_scirs2_integration_tests()?;

    // Verify test results
    assert!(test_suite.success_rate > 0.95);
    assert!(test_suite.scirs2_available);

    // Check specific categories
    let gradient_tests = &test_suite.test_results_by_category[&TestCategory::GradientComputation];
    assert!(gradient_tests.success_rate > 0.99);

    // Performance verification
    if let Some(ref perf) = test_suite.performance_summary {
        assert!(perf.average_performance_ratio < 2.0); // SciRS2 shouldn't be >2x slower
    }
}
"#
            .to_string(),
            best_practices: vec![
                "Test all integration points systematically".to_string(),
                "Include performance regression testing".to_string(),
                "Test fallback behavior when dependencies are unavailable".to_string(),
            ],
            common_pitfalls: vec![
                "Only testing happy path scenarios".to_string(),
                "Not testing with different hardware configurations".to_string(),
                "Ignoring integration test performance".to_string(),
            ],
        });

        Self { patterns }
    }

    pub fn print(&self) {
        println!("## Testing Patterns\n");
        for pattern in &self.patterns {
            pattern.print();
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        markdown.push_str("## Testing Patterns\n\n");

        for pattern in &self.patterns {
            markdown.push_str(&pattern.to_markdown());
        }

        markdown
    }
}

/// Resource management patterns
pub struct ResourceManagementPatterns {
    pub patterns: Vec<Pattern>,
}

impl ResourceManagementPatterns {
    pub fn new() -> Self {
        let mut patterns = Vec::new();

        patterns.push(Pattern {
            name: "RAII Resource Management".to_string(),
            description: "Use RAII guards for automatic resource cleanup".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Automatic gradient mode management
{
    let _guard = enable_grad(); // Enables gradient computation

    // Gradients are tracked in this scope
    let tensor = create_tensor_requiring_grad(&data)?;
    let result = compute_with_gradients(&tensor)?;

    // Gradient mode automatically restored when guard drops
}

// Resource guards for complex operations
{
    let _resource_guard = AutogradResourceFactory::create_computation_guard()?;

    // Perform complex computation
    let computation_result = complex_autograd_operation()?;

    // Resources automatically cleaned up when guard drops
}
"#
            .to_string(),
            best_practices: vec![
                "Always use RAII guards for resource management".to_string(),
                "Prefer scoped resource management over manual cleanup".to_string(),
                "Use resource factories for consistent resource creation".to_string(),
            ],
            common_pitfalls: vec![
                "Forgetting to create resource guards".to_string(),
                "Manually managing resources that have RAII alternatives".to_string(),
                "Creating guards with too broad or too narrow scope".to_string(),
            ],
        });

        Self { patterns }
    }

    pub fn print(&self) {
        println!("## Resource Management Patterns\n");
        for pattern in &self.patterns {
            pattern.print();
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        markdown.push_str("## Resource Management Patterns\n\n");

        for pattern in &self.patterns {
            markdown.push_str(&pattern.to_markdown());
        }

        markdown
    }
}

/// Distributed training patterns
pub struct DistributedTrainingPatterns {
    pub patterns: Vec<Pattern>,
}

impl DistributedTrainingPatterns {
    pub fn new() -> Self {
        let mut patterns = Vec::new();

        patterns.push(Pattern {
            name: "Gradient Synchronization".to_string(),
            description: "Efficient gradient synchronization across distributed nodes".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Configure distributed gradient synchronization
let sync_config = DistributedSyncConfig::new()
    .with_compression(CompressionType::Quantization)
    .with_synchronization_strategy(SyncStrategy::AllReduce);

let synchronizer = HierarchicalSynchronizer::with_config(sync_config)?;

// Synchronize gradients across nodes
let local_gradients = compute_local_gradients(&batch)?;
let synchronized_gradients = synchronizer.synchronize(local_gradients)?;
"#
            .to_string(),
            best_practices: vec![
                "Use gradient compression to reduce communication overhead".to_string(),
                "Implement hierarchical synchronization for large clusters".to_string(),
                "Monitor synchronization latency and throughput".to_string(),
            ],
            common_pitfalls: vec![
                "Not accounting for network latency in synchronization".to_string(),
                "Over-compressing gradients leading to accuracy loss".to_string(),
                "Synchronizing too frequently, reducing parallelism".to_string(),
            ],
        });

        Self { patterns }
    }

    pub fn print(&self) {
        println!("## Distributed Training Patterns\n");
        for pattern in &self.patterns {
            pattern.print();
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        markdown.push_str("## Distributed Training Patterns\n\n");

        for pattern in &self.patterns {
            markdown.push_str(&pattern.to_markdown());
        }

        markdown
    }
}

/// Custom operation patterns
pub struct CustomOperationPatterns {
    pub patterns: Vec<Pattern>,
}

impl CustomOperationPatterns {
    pub fn new() -> Self {
        let mut patterns = Vec::new();

        patterns.push(Pattern {
            name: "Custom Function Implementation".to_string(),
            description: "Implement custom differentiable functions".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Define custom function with forward and backward passes
struct CustomSigmoid;

impl CustomFunction for CustomSigmoid {
    fn forward(&self, input: &Tensor) -> AutogradResult<Tensor> {
        // Implement custom forward pass
        let output = input.sigmoid();
        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, input: &Tensor) -> AutogradResult<Tensor> {
        // Implement custom backward pass
        let sigmoid_output = input.sigmoid();
        let grad_input = grad_output * &sigmoid_output * &(1.0 - &sigmoid_output);
        Ok(grad_input)
    }
}

// Register and use custom function
let custom_fn = CustomSigmoid;
let result = custom_fn.apply(&input_tensor)?;
"#
            .to_string(),
            best_practices: vec![
                "Always implement both forward and backward passes".to_string(),
                "Test custom functions with gradient checking".to_string(),
                "Optimize custom functions for the target hardware".to_string(),
            ],
            common_pitfalls: vec![
                "Incorrect gradient computation in backward pass".to_string(),
                "Not handling edge cases in custom functions".to_string(),
                "Poor performance compared to built-in operations".to_string(),
            ],
        });

        Self { patterns }
    }

    pub fn print(&self) {
        println!("## Custom Operation Patterns\n");
        for pattern in &self.patterns {
            pattern.print();
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        markdown.push_str("## Custom Operation Patterns\n\n");

        for pattern in &self.patterns {
            markdown.push_str(&pattern.to_markdown());
        }

        markdown
    }
}

/// Debugging and profiling patterns
pub struct DebuggingPatterns {
    pub patterns: Vec<Pattern>,
}

impl DebuggingPatterns {
    pub fn new() -> Self {
        let mut patterns = Vec::new();

        patterns.push(Pattern {
            name: "Performance Profiling".to_string(),
            description: "Profile autograd operations for performance optimization".to_string(),
            code_example: r#"
use torsh_autograd::prelude::*;

// Enable autograd profiling
let profiler = AutogradProfiler::new();
profiler.enable(true);

// Profile specific operations
let _profile_guard = profiler.profile_scope("gradient_computation");

let gradients = {
    let _op_guard = profiler.profile_operation("backward_pass");
    compute_gradients(&tensor)?
};

// Get profiling results
let profile_report = profiler.generate_report();
println!("Total time: {:.2}ms", profile_report.total_time.as_millis());
println!("Memory peak: {:.2}MB", profile_report.peak_memory_usage / 1024.0 / 1024.0);
"#
            .to_string(),
            best_practices: vec![
                "Profile representative workloads, not toy examples".to_string(),
                "Use hierarchical profiling to identify bottlenecks".to_string(),
                "Compare profiles before and after optimizations".to_string(),
            ],
            common_pitfalls: vec![
                "Profiling only debug builds".to_string(),
                "Not considering profiling overhead in measurements".to_string(),
                "Focusing only on time, ignoring memory usage".to_string(),
            ],
        });

        Self { patterns }
    }

    pub fn print(&self) {
        println!("## Debugging and Profiling Patterns\n");
        for pattern in &self.patterns {
            pattern.print();
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        markdown.push_str("## Debugging and Profiling Patterns\n\n");

        for pattern in &self.patterns {
            markdown.push_str(&pattern.to_markdown());
        }

        markdown
    }
}

/// Individual pattern documentation
pub struct Pattern {
    pub name: String,
    pub description: String,
    pub code_example: String,
    pub best_practices: Vec<String>,
    pub common_pitfalls: Vec<String>,
}

impl Pattern {
    pub fn print(&self) {
        println!("### {}\n", self.name);
        println!("{}\n", self.description);

        println!("**Example:**");
        println!("```rust{}", self.code_example);
        println!("```\n");

        if !self.best_practices.is_empty() {
            println!("**Best Practices:**");
            for practice in &self.best_practices {
                println!("- {}", practice);
            }
            println!();
        }

        if !self.common_pitfalls.is_empty() {
            println!("**Common Pitfalls:**");
            for pitfall in &self.common_pitfalls {
                println!("- {}", pitfall);
            }
            println!();
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();

        markdown.push_str(&format!("### {}\n\n", self.name));
        markdown.push_str(&format!("{}\n\n", self.description));

        markdown.push_str("**Example:**\n");
        markdown.push_str(&format!("```rust{}\n```\n\n", self.code_example));

        if !self.best_practices.is_empty() {
            markdown.push_str("**Best Practices:**\n");
            for practice in &self.best_practices {
                markdown.push_str(&format!("- {}\n", practice));
            }
            markdown.push_str("\n");
        }

        if !self.common_pitfalls.is_empty() {
            markdown.push_str("**Common Pitfalls:**\n");
            for pitfall in &self.common_pitfalls {
                markdown.push_str(&format!("- {}\n", pitfall));
            }
            markdown.push_str("\n");
        }

        markdown
    }
}

/// Pattern-specific documentation
pub struct PatternDocumentation {
    pub category: PatternCategory,
    pub patterns: Vec<Pattern>,
}

impl PatternDocumentation {
    pub fn for_category(category: PatternCategory) -> Self {
        let patterns = match category {
            PatternCategory::SciRS2Integration => SciRS2IntegrationPatterns::new().patterns,
            PatternCategory::PerformanceOptimization => PerformancePatterns::new().patterns,
            PatternCategory::ErrorHandling => ErrorHandlingPatterns::new().patterns,
            PatternCategory::Testing => TestingPatterns::new().patterns,
            PatternCategory::ResourceManagement => ResourceManagementPatterns::new().patterns,
            PatternCategory::DistributedTraining => DistributedTrainingPatterns::new().patterns,
            PatternCategory::CustomOperations => CustomOperationPatterns::new().patterns,
            PatternCategory::DebuggingAndProfiling => DebuggingPatterns::new().patterns,
        };

        Self { category, patterns }
    }

    pub fn print(&self) {
        println!("# {} Patterns\n", self.category);
        for pattern in &self.patterns {
            pattern.print();
        }
    }
}

/// Troubleshooting guide for common issues
pub struct TroubleshootingGuide {
    pub issues: Vec<TroubleshootingIssue>,
}

#[derive(Debug, Clone)]
pub struct TroubleshootingIssue {
    pub problem: String,
    pub symptoms: Vec<String>,
    pub causes: Vec<String>,
    pub solutions: Vec<String>,
    pub prevention: Vec<String>,
}

impl TroubleshootingGuide {
    pub fn new() -> Self {
        let mut issues = Vec::new();

        issues.push(TroubleshootingIssue {
            problem: "SciRS2 Integration Failures".to_string(),
            symptoms: vec![
                "SciRS2AutogradAdapter initialization fails".to_string(),
                "Gradient computation returns errors".to_string(),
                "Performance degradation compared to expected".to_string(),
            ],
            causes: vec![
                "SciRS2 version mismatch".to_string(),
                "Missing SciRS2 dependencies".to_string(),
                "Configuration issues".to_string(),
            ],
            solutions: vec![
                "Check SciRS2 version compatibility".to_string(),
                "Update SciRS2 to compatible version".to_string(),
                "Use fallback implementation".to_string(),
                "Review configuration settings".to_string(),
            ],
            prevention: vec![
                "Always check version compatibility before deployment".to_string(),
                "Implement comprehensive integration tests".to_string(),
                "Monitor SciRS2 health in production".to_string(),
            ],
        });

        issues.push(TroubleshootingIssue {
            problem: "Memory Leaks in Gradient Computation".to_string(),
            symptoms: vec![
                "Gradually increasing memory usage".to_string(),
                "Out of memory errors during training".to_string(),
                "Performance degradation over time".to_string(),
            ],
            causes: vec![
                "Unreleased gradient references".to_string(),
                "Circular references in computation graph".to_string(),
                "Inefficient memory pool usage".to_string(),
            ],
            solutions: vec![
                "Use RAII guards for resource management".to_string(),
                "Enable garbage collection for gradients".to_string(),
                "Review computation graph construction".to_string(),
                "Optimize memory pool configuration".to_string(),
            ],
            prevention: vec![
                "Regular memory usage monitoring".to_string(),
                "Proper resource management patterns".to_string(),
                "Automated leak detection in tests".to_string(),
            ],
        });

        Self { issues }
    }

    pub fn print(&self) {
        println!("# Troubleshooting Guide\n");

        for (i, issue) in self.issues.iter().enumerate() {
            println!("## {}. {}\n", i + 1, issue.problem);

            println!("**Symptoms:**");
            for symptom in &issue.symptoms {
                println!("- {}", symptom);
            }
            println!();

            println!("**Common Causes:**");
            for cause in &issue.causes {
                println!("- {}", cause);
            }
            println!();

            println!("**Solutions:**");
            for solution in &issue.solutions {
                println!("- {}", solution);
            }
            println!();

            println!("**Prevention:**");
            for prevention in &issue.prevention {
                println!("- {}", prevention);
            }
            println!();
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        markdown.push_str("# Troubleshooting Guide\n\n");

        for (i, issue) in self.issues.iter().enumerate() {
            markdown.push_str(&format!("## {}. {}\n\n", i + 1, issue.problem));

            markdown.push_str("**Symptoms:**\n");
            for symptom in &issue.symptoms {
                markdown.push_str(&format!("- {}\n", symptom));
            }
            markdown.push_str("\n");

            markdown.push_str("**Common Causes:**\n");
            for cause in &issue.causes {
                markdown.push_str(&format!("- {}\n", cause));
            }
            markdown.push_str("\n");

            markdown.push_str("**Solutions:**\n");
            for solution in &issue.solutions {
                markdown.push_str(&format!("- {}\n", solution));
            }
            markdown.push_str("\n");

            markdown.push_str("**Prevention:**\n");
            for prevention in &issue.prevention {
                markdown.push_str(&format!("- {}\n", prevention));
            }
            markdown.push_str("\n");
        }

        markdown
    }
}

/// Migration guide for different scenarios
pub struct MigrationGuide {
    pub migrations: Vec<MigrationScenario>,
}

#[derive(Debug, Clone)]
pub struct MigrationScenario {
    pub name: String,
    pub description: String,
    pub from_version: String,
    pub to_version: String,
    pub breaking_changes: Vec<String>,
    pub migration_steps: Vec<String>,
    pub code_examples: Vec<String>,
}

impl MigrationGuide {
    pub fn new() -> Self {
        let mut migrations = Vec::new();

        migrations.push(MigrationScenario {
            name: "SciRS2 0.1.0-beta.1 to 0.1.0-beta.2".to_string(),
            description: "Migration from beta.1 to beta.2 with API changes".to_string(),
            from_version: "0.1.0-beta.1".to_string(),
            to_version: "0.1.0-beta.2".to_string(),
            breaking_changes: vec![
                "SciRS2AutogradAdapter constructor signature changed".to_string(),
                "GradientTensor enum variants modified".to_string(),
                "New version compatibility checking required".to_string(),
            ],
            migration_steps: vec![
                "Update SciRS2 dependency to 0.1.0-beta.2".to_string(),
                "Migrate SciRS2AutogradAdapter::new() calls".to_string(),
                "Update GradientTensor usage patterns".to_string(),
                "Add version compatibility checks".to_string(),
                "Test fallback behavior".to_string(),
            ],
            code_examples: vec![
                r#"
// Before (beta.1)
let adapter = SciRS2AutogradAdapter::new();

// After (beta.2)
let adapter = SciRS2AutogradAdapter::new()?;
"#
                .to_string(),
                r#"
// Before (beta.1)
match gradient_tensor {
    GradientTensor::SciRS2(tensor) => { /* ... */ },
    GradientTensor::Manual(data) => { /* ... */ },
}

// After (beta.2)
match gradient_tensor {
    GradientTensor::SciRS2(tensor) => { /* ... */ },
    GradientTensor::Manual(data) => { /* ... */ },
    GradientTensor::Fallback(data) => { /* ... */ }, // New variant
}
"#
                .to_string(),
            ],
        });

        Self { migrations }
    }

    pub fn print(&self) {
        println!("# Migration Guide\n");

        for migration in &self.migrations {
            println!("## {}\n", migration.name);
            println!("{}\n", migration.description);
            println!(
                "**From:** {} **To:** {}\n",
                migration.from_version, migration.to_version
            );

            println!("**Breaking Changes:**");
            for change in &migration.breaking_changes {
                println!("- {}", change);
            }
            println!();

            println!("**Migration Steps:**");
            for (i, step) in migration.migration_steps.iter().enumerate() {
                println!("{}. {}", i + 1, step);
            }
            println!();

            if !migration.code_examples.is_empty() {
                println!("**Code Examples:**");
                for example in &migration.code_examples {
                    println!("```rust{}\n```", example);
                }
                println!();
            }
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        markdown.push_str("# Migration Guide\n\n");

        for migration in &self.migrations {
            markdown.push_str(&format!("## {}\n\n", migration.name));
            markdown.push_str(&format!("{}\n\n", migration.description));
            markdown.push_str(&format!(
                "**From:** {} **To:** {}\n\n",
                migration.from_version, migration.to_version
            ));

            markdown.push_str("**Breaking Changes:**\n");
            for change in &migration.breaking_changes {
                markdown.push_str(&format!("- {}\n", change));
            }
            markdown.push_str("\n");

            markdown.push_str("**Migration Steps:**\n");
            for (i, step) in migration.migration_steps.iter().enumerate() {
                markdown.push_str(&format!("{}. {}\n", i + 1, step));
            }
            markdown.push_str("\n");

            if !migration.code_examples.is_empty() {
                markdown.push_str("**Code Examples:**\n");
                for example in &migration.code_examples {
                    markdown.push_str(&format!("```rust{}\n```\n\n", example));
                }
            }
        }

        markdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_documentation_creation() {
        let docs = IntegrationDocumentation::new();
        assert!(!docs.scirs2_patterns.patterns.is_empty());
        assert!(!docs.performance_patterns.patterns.is_empty());
        assert!(!docs.error_handling_patterns.patterns.is_empty());
    }

    #[test]
    fn test_pattern_category_display() {
        assert_eq!(
            PatternCategory::SciRS2Integration.to_string(),
            "SciRS2 Integration"
        );
        assert_eq!(
            PatternCategory::PerformanceOptimization.to_string(),
            "Performance Optimization"
        );
    }

    #[test]
    fn test_pattern_documentation_for_category() {
        let docs = PatternDocumentation::for_category(PatternCategory::SciRS2Integration);
        assert_eq!(docs.category, PatternCategory::SciRS2Integration);
        assert!(!docs.patterns.is_empty());
    }

    #[test]
    fn test_troubleshooting_guide_creation() {
        let guide = TroubleshootingGuide::new();
        assert!(!guide.issues.is_empty());

        for issue in &guide.issues {
            assert!(!issue.problem.is_empty());
            assert!(!issue.symptoms.is_empty());
            assert!(!issue.solutions.is_empty());
        }
    }

    #[test]
    fn test_migration_guide_creation() {
        let guide = MigrationGuide::new();
        assert!(!guide.migrations.is_empty());

        for migration in &guide.migrations {
            assert!(!migration.name.is_empty());
            assert!(!migration.migration_steps.is_empty());
        }
    }

    #[test]
    fn test_markdown_export() {
        let docs = IntegrationDocumentation::new();
        let markdown = docs.to_markdown();

        assert!(markdown.contains("# ToRSh Autograd Integration Patterns"));
        assert!(markdown.contains("## SciRS2 Integration Patterns"));
        assert!(markdown.contains("## Performance Optimization Patterns"));
    }

    #[test]
    fn test_pattern_markdown_conversion() {
        let pattern = Pattern {
            name: "Test Pattern".to_string(),
            description: "Test description".to_string(),
            code_example: "\nlet x = 1;\n".to_string(),
            best_practices: vec!["Practice 1".to_string()],
            common_pitfalls: vec!["Pitfall 1".to_string()],
        };

        let markdown = pattern.to_markdown();
        assert!(markdown.contains("### Test Pattern"));
        assert!(markdown.contains("Test description"));
        assert!(markdown.contains("```rust"));
        assert!(markdown.contains("**Best Practices:**"));
        assert!(markdown.contains("**Common Pitfalls:**"));
    }
}
