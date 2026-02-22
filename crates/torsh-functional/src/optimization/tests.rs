//! Tests for optimization algorithms

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::*;

    #[test]
    fn test_gradient_descent_quadratic() {
        // Minimize f(x) = (x-2)^2 + 1, minimum at x=2
        let objective = |x: &Tensor| -> TorshResult<f32> {
            let data = x.data()?;
            let val = data[0];
            Ok((val - 2.0).powi(2) + 1.0)
        };

        let gradient = |x: &Tensor| -> TorshResult<Tensor> {
            let data = x.data()?;
            let val = data[0];
            let grad_val = 2.0 * (val - 2.0);
            Ok(from_vec(vec![grad_val], &[1], DeviceType::Cpu)?)
        };

        let x0 = from_vec(vec![0.0], &[1], DeviceType::Cpu).unwrap();
        let params = GradientDescentParams {
            learning_rate: 0.1,
            max_iter: 100,
            tolerance: 1e-6,
            line_search: None,
        };

        let (x_opt, _) = gradient_descent(objective, gradient, &x0, Some(params)).unwrap();
        let result = x_opt.data().unwrap()[0];

        assert!(
            (result - 2.0).abs() < 1e-3,
            "Expected result to be close to 2.0, got {}",
            result
        );
    }

    #[test]
    fn test_backtracking_line_search() {
        // Test on f(x) = x^2, gradient = 2x
        let objective = |x: &Tensor| -> TorshResult<f32> {
            let data = x.data()?;
            Ok(data[0].powi(2))
        };

        let gradient = |x: &Tensor| -> TorshResult<Tensor> {
            let data = x.data()?;
            Ok(from_vec(vec![2.0 * data[0]], &[1], DeviceType::Cpu)?)
        };

        let x = from_vec(vec![1.0], &[1], DeviceType::Cpu).unwrap();
        let p = from_vec(vec![-1.0], &[1], DeviceType::Cpu).unwrap(); // Descent direction

        let alpha = backtracking_line_search(objective, gradient, &x, &p, None).unwrap();

        // Should find a reasonable step size
        assert!(alpha > 0.0 && alpha <= 1.0);
    }

    #[test]
    fn test_momentum_gradient_descent() {
        // Test on simple quadratic function
        let objective = |x: &Tensor| -> TorshResult<f32> {
            let data = x.data()?;
            let val = data[0];
            Ok(val.powi(2))
        };

        let gradient = |x: &Tensor| -> TorshResult<Tensor> {
            let data = x.data()?;
            Ok(from_vec(vec![2.0 * data[0]], &[1], DeviceType::Cpu)?)
        };

        let x0 = from_vec(vec![1.0], &[1], DeviceType::Cpu).unwrap();
        let params = MomentumParams {
            learning_rate: 0.1,
            momentum: 0.9,
            max_iter: 50,
            tolerance: 1e-6,
        };

        let (x_opt, _) = momentum_gradient_descent(objective, gradient, &x0, Some(params)).unwrap();
        let result = x_opt.data().unwrap()[0];

        assert!(
            (result - 0.0).abs() < 1e-1,
            "Expected result to be close to 0.0, got {}",
            result
        );
    }

    #[test]
    fn test_tensor_characteristics_analysis() {
        let tensor = from_vec(vec![1.0, 0.0, 3.0, 0.0, 5.0], &[5], DeviceType::Cpu).unwrap();
        let characteristics = TensorCharacteristics::analyze(&tensor).unwrap();

        assert_eq!(characteristics.size, 5);
        assert_eq!(characteristics.sparsity, 0.4); // 2 out of 5 are zero
        assert!(characteristics.condition_number > 1.0);
    }

    #[test]
    fn test_adaptive_algorithm_selector() {
        let selector = AdaptiveAlgorithmSelector::new();

        // Test with sparse tensor characteristics
        let sparse_characteristics = TensorCharacteristics {
            size: 1000,
            condition_number: 100.0,
            sparsity: 0.9,
            numerical_precision: 0.5,
            memory_layout_score: 0.8,
            computational_complexity: 1000.0,
        };

        let algorithm = selector.select_algorithm(&sparse_characteristics);
        assert!(matches!(
            algorithm,
            OptimizationAlgorithm::ConjugateGradient
        ));

        // Test with large, well-conditioned problem
        let large_characteristics = TensorCharacteristics {
            size: 2_000_000,
            condition_number: 50.0,
            sparsity: 0.1,
            numerical_precision: 1.0,
            memory_layout_score: 0.8,
            computational_complexity: 2_000_000.0,
        };

        let algorithm = selector.select_algorithm(&large_characteristics);
        assert!(matches!(algorithm, OptimizationAlgorithm::LBFGS));
    }

    #[test]
    fn test_adaptive_selector_learning() {
        let mut selector = AdaptiveAlgorithmSelector::new();

        // Record poor performance for Adam
        selector.record_performance(&OptimizationAlgorithm::Adam, 0.2);
        selector.record_performance(&OptimizationAlgorithm::Adam, 0.1);

        // Record good performance for MomentumGradientDescent
        selector.record_performance(&OptimizationAlgorithm::MomentumGradientDescent, 0.8);
        selector.record_performance(&OptimizationAlgorithm::MomentumGradientDescent, 0.9);

        let adam_score = selector.get_algorithm_score(&OptimizationAlgorithm::Adam);
        let momentum_score =
            selector.get_algorithm_score(&OptimizationAlgorithm::MomentumGradientDescent);

        assert!(momentum_score > adam_score);
    }

    #[test]
    fn test_optimization_problem_analysis() {
        let objective_values = vec![10.0, 9.0, 8.5, 8.3, 8.29, 8.289];
        let gradient_norms = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1];

        let characteristics = TensorCharacteristics {
            size: 1000,
            condition_number: 200.0,
            sparsity: 0.1,
            numerical_precision: 0.8,
            memory_layout_score: 0.8,
            computational_complexity: 1000.0,
        };

        let (algorithm, recommendation) =
            analyze_optimization_problem(&objective_values, &gradient_norms, &characteristics);

        assert!(matches!(
            algorithm,
            OptimizationAlgorithm::MomentumGradientDescent
        ));
        assert!(recommendation.contains("MomentumGradientDescent"));
    }

    #[test]
    fn test_auto_configure_optimization() {
        let characteristics = TensorCharacteristics {
            size: 10_000,
            condition_number: 50.0,
            sparsity: 0.1,
            numerical_precision: 1.0,
            memory_layout_score: 0.8,
            computational_complexity: 10_000.0,
        };

        let config =
            auto_configure_optimization(&characteristics, &OptimizationAlgorithm::Adam).unwrap();
        assert!(config.contains("AdamParams"));
        assert!(config.contains("learning_rate"));

        let momentum_config = auto_configure_optimization(
            &characteristics,
            &OptimizationAlgorithm::MomentumGradientDescent,
        )
        .unwrap();
        assert!(momentum_config.contains("MomentumParams"));
        assert!(momentum_config.contains("momentum"));
    }
}