//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;

use super::types::{PassStatistics, XlaComputation, XlaConfig};

/// Trait for XLA optimization passes
pub trait XlaPass {
    /// Name of the pass
    fn name(&self) -> &str;
    /// Run the pass on a computation
    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics>;
    /// Check if the pass should be run (based on config)
    fn should_run(&self, config: &XlaConfig) -> bool {
        config.optimization_level > 0
    }
}
#[cfg(test)]
mod tests {
    use super::super::types::{
        AlgebraicSimplificationPass, CommonSubexpressionEliminationPass, ConstantFoldingPass,
        CopyEliminationPass, DeadCodeEliminationPass, HloOpcode, LayoutOptimizationPass,
        MemoryAllocationOptimizationPass, OperationFusionPass, ParallelizationAnalysisPass,
        XlaBuilder, XlaNodeId, XlaPassManager, XlaTarget,
    };
    use super::*;
    use crate::device::DeviceType;
    use crate::dtype::DType;
    #[test]
    fn test_hlo_opcode_name() {
        assert_eq!(HloOpcode::Add.name(), "add");
        assert_eq!(HloOpcode::Dot.name(), "dot");
        assert_eq!(HloOpcode::Reshape.name(), "reshape");
    }
    #[test]
    fn test_hlo_opcode_properties() {
        assert!(HloOpcode::Add.is_elementwise());
        assert!(!HloOpcode::Dot.is_elementwise());
        assert!(HloOpcode::Reduce.is_reduction());
        assert!(!HloOpcode::Add.is_reduction());
        assert!(HloOpcode::Reshape.is_shape_changing());
        assert!(!HloOpcode::Add.is_shape_changing());
    }
    #[test]
    fn test_xla_builder_parameter() {
        let mut builder = XlaBuilder::new("test");
        let param = builder
            .add_parameter(0, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        assert_eq!(builder.num_nodes(), 1);
        assert_eq!(param, XlaNodeId(0));
    }
    #[test]
    fn test_xla_builder_add() {
        let mut builder = XlaBuilder::new("test");
        let param_a = builder
            .add_parameter(0, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let result = builder
            .add_add(param_a, param_b)
            .expect("add_parameter should succeed");
        assert_eq!(builder.num_nodes(), 3);
        assert_eq!(result, XlaNodeId(2));
    }
    #[test]
    fn test_xla_builder_dot() {
        let mut builder = XlaBuilder::new("matmul");
        let param_a = builder
            .add_parameter(0, &[128, 256], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[256, 512], DType::F32)
            .expect("add_parameter should succeed");
        let result = builder
            .add_dot(param_a, param_b)
            .expect("add_parameter should succeed");
        assert_eq!(builder.num_nodes(), 3);
        let computation = builder.build(result).expect("add_parameter should succeed");
        assert_eq!(
            computation
                .output_shape()
                .expect("add_dot should succeed")
                .dims(),
            &[128, 512]
        );
    }
    #[test]
    fn test_xla_builder_dot_invalid_dims() {
        let mut builder = XlaBuilder::new("matmul");
        let param_a = builder
            .add_parameter(0, &[128, 256], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[128, 512], DType::F32)
            .expect("add_parameter should succeed");
        let result = builder.add_dot(param_a, param_b);
        assert!(result.is_err());
    }
    #[test]
    fn test_xla_builder_reshape() {
        let mut builder = XlaBuilder::new("reshape");
        let param = builder
            .add_parameter(0, &[10, 20, 30], DType::F32)
            .expect("add_parameter should succeed");
        let result = builder
            .add_reshape(param, &[10, 600])
            .expect("add_parameter should succeed");
        assert_eq!(builder.num_nodes(), 2);
        let computation = builder.build(result).expect("add_parameter should succeed");
        assert_eq!(
            computation
                .output_shape()
                .expect("add_reshape should succeed")
                .dims(),
            &[10, 600]
        );
    }
    #[test]
    fn test_xla_builder_reshape_invalid() {
        let mut builder = XlaBuilder::new("reshape");
        let param = builder
            .add_parameter(0, &[10, 20, 30], DType::F32)
            .expect("add_parameter should succeed");
        let result = builder.add_reshape(param, &[10, 100]);
        assert!(result.is_err());
    }
    #[test]
    fn test_xla_builder_transpose() {
        let mut builder = XlaBuilder::new("transpose");
        let param = builder
            .add_parameter(0, &[10, 20, 30], DType::F32)
            .expect("add_parameter should succeed");
        let result = builder
            .add_transpose(param, &[2, 0, 1])
            .expect("add_parameter should succeed");
        let computation = builder.build(result).expect("add_parameter should succeed");
        assert_eq!(
            computation
                .output_shape()
                .expect("add_parameter should succeed")
                .dims(),
            &[30, 10, 20]
        );
    }
    #[test]
    fn test_xla_computation_validate() {
        let mut builder = XlaBuilder::new("test");
        let param_a = builder
            .add_parameter(0, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let result = builder
            .add_add(param_a, param_b)
            .expect("add_parameter should succeed");
        let computation = builder.build(result).expect("add_parameter should succeed");
        assert!(computation.validate().is_ok());
    }
    #[test]
    fn test_xla_computation_operation_counts() {
        let mut builder = XlaBuilder::new("test");
        let param_a = builder
            .add_parameter(0, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let add_result = builder
            .add_add(param_a, param_b)
            .expect("add_parameter should succeed");
        let mul_result = builder
            .add_multiply(add_result, param_b)
            .expect("add_parameter should succeed");
        let computation = builder
            .build(mul_result)
            .expect("add_parameter should succeed");
        let counts = computation.operation_counts();
        assert_eq!(counts.len(), 3);
    }
    #[test]
    fn test_xla_computation_num_parameters() {
        let mut builder = XlaBuilder::new("test");
        builder
            .add_parameter(0, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        builder
            .add_parameter(1, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let param_c = builder
            .add_parameter(2, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let computation = builder
            .build(param_c)
            .expect("add_parameter should succeed");
        assert_eq!(computation.num_parameters(), 3);
    }
    #[test]
    fn test_xla_computation_to_hlo_text() {
        let mut builder = XlaBuilder::new("simple_add");
        let param_a = builder
            .add_parameter(0, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let result = builder
            .add_add(param_a, param_b)
            .expect("add_parameter should succeed");
        let computation = builder.build(result).expect("add_parameter should succeed");
        let hlo_text = computation.to_hlo_text();
        assert!(hlo_text.contains("HloModule simple_add"));
        assert!(hlo_text.contains("ENTRY main"));
        assert!(hlo_text.contains("ROOT"));
    }
    #[test]
    fn test_xla_target_from_device_type() {
        assert_eq!(XlaTarget::from_device_type(DeviceType::Cpu), XlaTarget::Cpu);
        assert_eq!(
            XlaTarget::from_device_type(DeviceType::Cuda(0)),
            XlaTarget::Gpu
        );
        assert_eq!(
            XlaTarget::from_device_type(DeviceType::Metal(0)),
            XlaTarget::Gpu
        );
    }
    #[test]
    fn test_xla_config_default() {
        let config = XlaConfig::default();
        assert_eq!(config.target, XlaTarget::Cpu);
        assert!(config.enable_fusion);
        assert_eq!(config.optimization_level, 2);
    }
    #[test]
    fn test_xla_config_builder() {
        let config = XlaConfig::new(XlaTarget::Gpu)
            .with_fusion(false)
            .with_optimization_level(3);
        assert_eq!(config.target, XlaTarget::Gpu);
        assert!(!config.enable_fusion);
        assert_eq!(config.optimization_level, 3);
    }
    #[test]
    fn test_xla_config_presets() {
        let aggressive = XlaConfig::aggressive();
        assert_eq!(aggressive.optimization_level, 3);
        assert!(aggressive.enable_fusion);
        let conservative = XlaConfig::conservative();
        assert_eq!(conservative.optimization_level, 0);
        assert!(!conservative.enable_fusion);
    }
    #[test]
    fn test_complex_computation() {
        let mut builder = XlaBuilder::new("complex");
        let a = builder
            .add_parameter(0, &[128, 256], DType::F32)
            .expect("add_parameter should succeed");
        let b = builder
            .add_parameter(1, &[256, 512], DType::F32)
            .expect("add_parameter should succeed");
        let c = builder
            .add_parameter(2, &[128, 512], DType::F32)
            .expect("add_parameter should succeed");
        let d = builder
            .add_parameter(3, &[128, 512], DType::F32)
            .expect("add_parameter should succeed");
        let matmul = builder.add_dot(a, b).expect("add_parameter should succeed");
        let mul = builder
            .add_multiply(c, d)
            .expect("add_parameter should succeed");
        let result = builder
            .add_add(matmul, mul)
            .expect("add_parameter should succeed");
        let computation = builder.build(result).expect("add_add should succeed");
        assert_eq!(computation.num_nodes(), 7);
        assert!(computation.validate().is_ok());
    }
    #[test]
    fn test_pass_statistics_creation() {
        let stats = PassStatistics::new();
        assert_eq!(stats.nodes_removed, 0);
        assert_eq!(stats.nodes_added, 0);
        assert_eq!(stats.nodes_modified, 0);
        assert!(!stats.changed);
        assert!(!stats.has_changes());
    }
    #[test]
    fn test_pass_statistics_merge() {
        let mut stats1 = PassStatistics {
            nodes_removed: 2,
            nodes_added: 1,
            nodes_modified: 3,
            changed: true,
        };
        let stats2 = PassStatistics {
            nodes_removed: 1,
            nodes_added: 2,
            nodes_modified: 1,
            changed: false,
        };
        stats1.merge(&stats2);
        assert_eq!(stats1.nodes_removed, 3);
        assert_eq!(stats1.nodes_added, 3);
        assert_eq!(stats1.nodes_modified, 4);
        assert!(stats1.changed);
    }
    #[test]
    fn test_constant_folding_pass_name() {
        let pass = ConstantFoldingPass;
        assert_eq!(pass.name(), "constant-folding");
    }
    #[test]
    fn test_constant_folding_pass_should_run() {
        let pass = ConstantFoldingPass;
        let config = XlaConfig::default();
        assert!(pass.should_run(&config));
        let config = XlaConfig::conservative();
        assert!(!pass.should_run(&config));
    }
    #[test]
    fn test_dead_code_elimination_pass_name() {
        let pass = DeadCodeEliminationPass;
        assert_eq!(pass.name(), "dead-code-elimination");
    }
    #[test]
    fn test_dead_code_elimination_empty_computation() {
        let pass = DeadCodeEliminationPass;
        let mut computation = XlaComputation {
            name: "empty".to_string(),
            nodes: vec![],
            root: XlaNodeId(0),
            config: XlaConfig::default(),
        };
        let stats = pass.run(&mut computation).expect("pass run should succeed");
        assert_eq!(stats.nodes_removed, 0);
        assert!(!stats.changed);
    }
    #[test]
    fn test_dead_code_elimination_no_dead_code() {
        let mut builder = XlaBuilder::new("test");
        let param = builder
            .add_parameter(0, &[10], DType::F32)
            .expect("add_parameter should succeed");
        let computation = builder.build(param).expect("add_parameter should succeed");
        let pass = DeadCodeEliminationPass;
        let mut mut_comp = computation;
        let stats = pass.run(&mut mut_comp).expect("build should succeed");
        assert_eq!(stats.nodes_removed, 0);
        assert!(!stats.changed);
    }
    #[test]
    fn test_cse_pass_name() {
        let pass = CommonSubexpressionEliminationPass;
        assert_eq!(pass.name(), "common-subexpression-elimination");
    }
    #[test]
    fn test_cse_pass_should_run() {
        let pass = CommonSubexpressionEliminationPass;
        let config = XlaConfig::default();
        assert!(pass.should_run(&config));
        let config = XlaConfig::conservative();
        assert!(!pass.should_run(&config));
    }
    #[test]
    fn test_fusion_pass_name() {
        let pass = OperationFusionPass;
        assert_eq!(pass.name(), "operation-fusion");
    }
    #[test]
    fn test_fusion_pass_should_run() {
        let pass = OperationFusionPass;
        let config = XlaConfig::default();
        assert!(pass.should_run(&config));
        let config = XlaConfig::conservative();
        assert!(!pass.should_run(&config));
        let mut config = XlaConfig::default();
        config.enable_fusion = false;
        assert!(!pass.should_run(&config));
    }
    #[test]
    fn test_algebraic_simplification_pass_name() {
        let pass = AlgebraicSimplificationPass;
        assert_eq!(pass.name(), "algebraic-simplification");
    }
    #[test]
    fn test_layout_optimization_pass_name() {
        let pass = LayoutOptimizationPass;
        assert_eq!(pass.name(), "layout-optimization");
    }
    #[test]
    fn test_layout_optimization_pass_should_run() {
        let pass = LayoutOptimizationPass;
        let mut config = XlaConfig::default();
        config.optimization_level = 2;
        assert!(pass.should_run(&config));
        config.optimization_level = 1;
        assert!(!pass.should_run(&config));
    }
    #[test]
    fn test_layout_optimization_pass_detects_opportunities() {
        let mut builder = XlaBuilder::new("layout_test");
        let param_a = builder
            .add_parameter(0, &[128, 256], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[256, 512], DType::F32)
            .expect("add_parameter should succeed");
        let matmul = builder
            .add_dot(param_a, param_b)
            .expect("add_parameter should succeed");
        let transposed = builder
            .add_transpose(matmul, &[1, 0])
            .expect("add_parameter should succeed");
        let mut computation = builder
            .build(transposed)
            .expect("add_parameter should succeed");
        computation.config.optimization_level = 2;
        let pass = LayoutOptimizationPass;
        let stats = pass.run(&mut computation).expect("build should succeed");
        assert!(stats.nodes_modified >= 2);
    }
    #[test]
    fn test_copy_elimination_pass_name() {
        let pass = CopyEliminationPass;
        assert_eq!(pass.name(), "copy-elimination");
    }
    #[test]
    fn test_copy_elimination_pass_should_run() {
        let pass = CopyEliminationPass;
        let mut config = XlaConfig::default();
        config.optimization_level = 1;
        assert!(pass.should_run(&config));
        config.optimization_level = 0;
        assert!(!pass.should_run(&config));
    }
    #[test]
    fn test_copy_elimination_pass_counts_copies() {
        let mut builder = XlaBuilder::new("copy_test");
        let param = builder
            .add_parameter(0, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let copy1 = builder
            .add_copy(param)
            .expect("add_parameter should succeed");
        let copy2 = builder
            .add_copy(copy1)
            .expect("add_parameter should succeed");
        let mut computation = builder.build(copy2).expect("add_parameter should succeed");
        computation.config.optimization_level = 1;
        let pass = CopyEliminationPass;
        let stats = pass.run(&mut computation).expect("build should succeed");
        assert_eq!(stats.nodes_removed, 2);
    }
    #[test]
    fn test_pass_manager_default() {
        let manager = XlaPassManager::default();
        let passes = manager.passes();
        assert_eq!(passes.len(), 9);
        assert!(passes.contains(&"constant-folding"));
        assert!(passes.contains(&"algebraic-simplification"));
        assert!(passes.contains(&"copy-elimination"));
        assert!(passes.contains(&"common-subexpression-elimination"));
        assert!(passes.contains(&"operation-fusion"));
        assert!(passes.contains(&"layout-optimization"));
        assert!(passes.contains(&"memory-allocation-optimization"));
        assert!(passes.contains(&"parallelization-analysis"));
        assert!(passes.contains(&"dead-code-elimination"));
    }
    #[test]
    fn test_pass_manager_new() {
        let manager = XlaPassManager::new();
        assert_eq!(manager.passes().len(), 9);
    }
    #[test]
    fn test_pass_manager_add_pass() {
        let mut manager = XlaPassManager {
            passes: Vec::new(),
            run_until_fixed_point: false,
            max_iterations: 1,
        };
        assert_eq!(manager.passes().len(), 0);
        manager.add_pass(Box::new(ConstantFoldingPass));
        assert_eq!(manager.passes().len(), 1);
        manager.add_pass(Box::new(DeadCodeEliminationPass));
        assert_eq!(manager.passes().len(), 2);
    }
    #[test]
    fn test_pass_manager_with_fixed_point() {
        let manager = XlaPassManager::new().with_fixed_point(false);
        assert!(!manager.run_until_fixed_point);
    }
    #[test]
    fn test_pass_manager_with_max_iterations() {
        let manager = XlaPassManager::new().with_max_iterations(20);
        assert_eq!(manager.max_iterations, 20);
    }
    #[test]
    fn test_pass_manager_run_simple_computation() {
        let mut builder = XlaBuilder::new("simple");
        let param = builder
            .add_parameter(0, &[10], DType::F32)
            .expect("add_parameter should succeed");
        let mut computation = builder.build(param).expect("add_parameter should succeed");
        let manager = XlaPassManager::new();
        let stats = manager
            .run(&mut computation)
            .expect("add_parameter should succeed");
        assert_eq!(stats.nodes_removed, 0);
    }
    #[test]
    fn test_computation_optimize() {
        let mut builder = XlaBuilder::new("test");
        let param_a = builder
            .add_parameter(0, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[10, 20], DType::F32)
            .expect("add_parameter should succeed");
        let result = builder
            .add_add(param_a, param_b)
            .expect("add_parameter should succeed");
        let mut computation = builder.build(result).expect("add_parameter should succeed");
        let stats = computation
            .optimize()
            .expect("add_parameter should succeed");
        assert!(!stats.changed || stats.changed);
    }
    #[test]
    fn test_computation_optimize_with_custom_manager() {
        let mut builder = XlaBuilder::new("test");
        let param = builder
            .add_parameter(0, &[10], DType::F32)
            .expect("add_parameter should succeed");
        let mut computation = builder.build(param).expect("add_parameter should succeed");
        let manager = XlaPassManager::new().with_max_iterations(5);
        let stats = computation
            .optimize_with(&manager)
            .expect("add_parameter should succeed");
        assert_eq!(stats.nodes_removed, 0);
    }
    #[test]
    fn test_computation_run_pass() {
        let mut builder = XlaBuilder::new("test");
        let param = builder
            .add_parameter(0, &[10], DType::F32)
            .expect("add_parameter should succeed");
        let mut computation = builder.build(param).expect("add_parameter should succeed");
        let pass = DeadCodeEliminationPass;
        let stats = computation
            .run_pass(&pass)
            .expect("add_parameter should succeed");
        assert_eq!(stats.nodes_removed, 0);
        assert!(!stats.changed);
    }
    #[test]
    fn test_pass_manager_run_until_fixed_point() {
        let mut builder = XlaBuilder::new("test");
        let param_a = builder
            .add_parameter(0, &[10], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[10], DType::F32)
            .expect("add_parameter should succeed");
        let add1 = builder
            .add_add(param_a, param_b)
            .expect("add_parameter should succeed");
        let add2 = builder
            .add_add(add1, param_b)
            .expect("add_parameter should succeed");
        let mut computation = builder.build(add2).expect("add_parameter should succeed");
        let manager = XlaPassManager::new()
            .with_fixed_point(true)
            .with_max_iterations(10);
        let result = manager.run(&mut computation);
        assert!(result.is_ok());
    }
    #[test]
    fn test_optimization_with_aggressive_config() {
        let mut builder = XlaBuilder::new("aggressive");
        let param_a = builder
            .add_parameter(0, &[128, 256], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[256, 512], DType::F32)
            .expect("add_parameter should succeed");
        let matmul = builder
            .add_dot(param_a, param_b)
            .expect("add_parameter should succeed");
        let mut computation = builder.build(matmul).expect("add_parameter should succeed");
        computation.config = XlaConfig::aggressive();
        let stats = computation.optimize().expect("add_dot should succeed");
        assert!(stats.nodes_removed <= computation.nodes.len());
    }
    #[test]
    fn test_optimization_with_conservative_config() {
        let mut builder = XlaBuilder::new("conservative");
        let param = builder
            .add_parameter(0, &[10], DType::F32)
            .expect("add_parameter should succeed");
        let mut computation = builder.build(param).expect("add_parameter should succeed");
        computation.config = XlaConfig::conservative();
        let manager = XlaPassManager::new();
        let stats = manager.run(&mut computation).expect("build should succeed");
        assert_eq!(stats.nodes_removed, 0);
    }
    #[test]
    fn test_memory_allocation_optimization_pass_name() {
        let pass = MemoryAllocationOptimizationPass;
        assert_eq!(pass.name(), "memory-allocation-optimization");
    }
    #[test]
    fn test_memory_allocation_optimization_pass_should_run() {
        let pass = MemoryAllocationOptimizationPass;
        let mut config = XlaConfig::default();
        config.optimization_level = 1;
        assert!(pass.should_run(&config));
        config.optimization_level = 0;
        assert!(!pass.should_run(&config));
    }
    #[test]
    fn test_memory_allocation_optimization_detects_buffer_reuse() {
        let mut builder = XlaBuilder::new("memory_test");
        let param_a = builder
            .add_parameter(0, &[128, 256], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[128, 256], DType::F32)
            .expect("add_parameter should succeed");
        let add1 = builder
            .add_add(param_a, param_b)
            .expect("add_parameter should succeed");
        let mul = builder
            .add_multiply(add1, param_b)
            .expect("add_parameter should succeed");
        let mut computation = builder.build(mul).expect("add_parameter should succeed");
        computation.config.optimization_level = 1;
        let pass = MemoryAllocationOptimizationPass;
        let stats = pass.run(&mut computation).expect("build should succeed");
        assert!(stats.nodes_modified > 0);
    }
    #[test]
    fn test_memory_allocation_optimization_detects_inplace_ops() {
        let mut builder = XlaBuilder::new("inplace_test");
        let param_a = builder
            .add_parameter(0, &[100, 100], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[100, 100], DType::F32)
            .expect("add_parameter should succeed");
        let add = builder
            .add_add(param_a, param_b)
            .expect("add_parameter should succeed");
        let mul = builder
            .add_multiply(add, param_b)
            .expect("add_parameter should succeed");
        let sub = builder
            .add_subtract(mul, param_b)
            .expect("add_parameter should succeed");
        let mut computation = builder.build(sub).expect("add_add should succeed");
        computation.config.optimization_level = 1;
        let pass = MemoryAllocationOptimizationPass;
        let stats = pass.run(&mut computation).expect("build should succeed");
        assert!(stats.nodes_modified >= 3);
    }
    #[test]
    fn test_parallelization_analysis_pass_name() {
        let pass = ParallelizationAnalysisPass;
        assert_eq!(pass.name(), "parallelization-analysis");
    }
    #[test]
    fn test_parallelization_analysis_pass_should_run() {
        let pass = ParallelizationAnalysisPass;
        let mut config = XlaConfig::default();
        config.optimization_level = 2;
        assert!(pass.should_run(&config));
        config.optimization_level = 1;
        assert!(!pass.should_run(&config));
    }
    #[test]
    fn test_parallelization_analysis_detects_independent_ops() {
        let mut builder = XlaBuilder::new("parallel_test");
        let input = builder
            .add_parameter(0, &[128, 256], DType::F32)
            .expect("add_parameter should succeed");
        let add1 = builder
            .add_add(input, input)
            .expect("add_parameter should succeed");
        let mul1 = builder
            .add_multiply(input, input)
            .expect("add_parameter should succeed");
        let result = builder
            .add_add(add1, mul1)
            .expect("add_parameter should succeed");
        let mut computation = builder.build(result).expect("add_add should succeed");
        computation.config.optimization_level = 2;
        let pass = ParallelizationAnalysisPass;
        let stats = pass.run(&mut computation).expect("build should succeed");
        assert!(stats.nodes_modified > 0);
    }
    #[test]
    fn test_parallelization_analysis_detects_batch_ops() {
        let mut builder = XlaBuilder::new("batch_parallel_test");
        let param_a = builder
            .add_parameter(0, &[32, 128, 256], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[32, 128, 256], DType::F32)
            .expect("add_parameter should succeed");
        let add = builder
            .add_add(param_a, param_a)
            .expect("add_parameter should succeed");
        let mul = builder
            .add_multiply(param_b, param_b)
            .expect("add_parameter should succeed");
        let result = builder.add_add(add, mul).expect("add_add should succeed");
        let mut computation = builder.build(result).expect("add_add should succeed");
        computation.config.optimization_level = 2;
        let pass = ParallelizationAnalysisPass;
        let stats = pass.run(&mut computation).expect("build should succeed");
        assert!(stats.nodes_modified > 0);
    }
    #[test]
    fn test_pass_manager_includes_all_passes() {
        let manager = XlaPassManager::default();
        let passes = manager.passes();
        assert_eq!(passes.len(), 9);
        assert!(passes.contains(&"constant-folding"));
        assert!(passes.contains(&"algebraic-simplification"));
        assert!(passes.contains(&"copy-elimination"));
        assert!(passes.contains(&"common-subexpression-elimination"));
        assert!(passes.contains(&"operation-fusion"));
        assert!(passes.contains(&"layout-optimization"));
        assert!(passes.contains(&"memory-allocation-optimization"));
        assert!(passes.contains(&"parallelization-analysis"));
        assert!(passes.contains(&"dead-code-elimination"));
    }
    #[test]
    fn test_comprehensive_optimization_pipeline() {
        let mut builder = XlaBuilder::new("comprehensive");
        let param_a = builder
            .add_parameter(0, &[128, 256], DType::F32)
            .expect("add_parameter should succeed");
        let param_b = builder
            .add_parameter(1, &[256, 512], DType::F32)
            .expect("add_parameter should succeed");
        let matmul = builder
            .add_dot(param_a, param_b)
            .expect("add_parameter should succeed");
        let copy = builder
            .add_copy(matmul)
            .expect("add_parameter should succeed");
        let add = builder
            .add_add(copy, copy)
            .expect("add_parameter should succeed");
        let transposed = builder
            .add_transpose(add, &[1, 0])
            .expect("add_add should succeed");
        let mut computation = builder.build(transposed).expect("add_add should succeed");
        computation.config = XlaConfig::aggressive();
        let stats = computation
            .optimize()
            .expect("add_transpose should succeed");
        assert!(stats.nodes_removed > 0 || stats.nodes_modified > 0);
    }
}
