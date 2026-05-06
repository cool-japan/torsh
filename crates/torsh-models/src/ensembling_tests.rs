//! Tests for model ensembling utilities.
use super::*;
use torsh_core::DeviceType;
use torsh_tensor::Tensor;

// Simple mock model for testing
struct MockModel {
    bias: f32,
}

impl MockModel {
    fn new(bias: f32) -> Self {
        Self { bias }
    }
}

impl Module for MockModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.add_scalar(self.bias)
    }

    fn parameters(&self) -> HashMap<String, torsh_nn::Parameter> {
        HashMap::new()
    }
    fn named_parameters(&self) -> HashMap<String, torsh_nn::Parameter> {
        HashMap::new()
    }
    fn training(&self) -> bool {
        true
    }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn to_device(&mut self, _device: torsh_core::DeviceType) -> Result<()> {
        Ok(())
    }
}

#[test]
fn test_simple_average_ensemble() {
    let models = vec![
        MockModel::new(0.1),
        MockModel::new(0.2),
        MockModel::new(0.3),
    ];

    let config = ensembling_utils::simple_average_config();
    let mut ensemble = ModelEnsemble::new(models, config);

    let device = DeviceType::Cpu;
    let input = Tensor::ones(&[1, 5], device).unwrap();
    let prediction = ensemble.predict(&input).unwrap();

    assert_eq!(prediction.shape(), input.shape());
}

#[test]
fn test_weighted_average_ensemble() {
    let models = vec![MockModel::new(0.1), MockModel::new(0.2)];

    let weights = vec![0.7, 0.3];
    let config = ensembling_utils::weighted_average_config(weights.clone(), false);
    let mut ensemble = ModelEnsemble::new(models, config);

    assert_eq!(ensemble.get_weights(), &weights);

    let device = DeviceType::Cpu;
    let input = Tensor::ones(&[1, 3], device).unwrap();
    let prediction = ensemble.predict(&input).unwrap();

    assert_eq!(prediction.shape(), input.shape());
}

#[test]
fn test_ensemble_weight_setting() {
    let models = vec![MockModel::new(0.0), MockModel::new(0.0)];
    let config = ensembling_utils::simple_average_config();
    let mut ensemble = ModelEnsemble::new(models, config);

    let new_weights = vec![0.8, 0.2];
    ensemble.set_weights(new_weights).unwrap();

    let weights = ensemble.get_weights();
    assert!((weights[0] - 0.8).abs() < 1e-6);
    assert!((weights[1] - 0.2).abs() < 1e-6);
}

#[test]
fn test_individual_predictions() {
    let models = vec![MockModel::new(1.0), MockModel::new(2.0)];

    let config = ensembling_utils::simple_average_config();
    let ensemble = ModelEnsemble::new(models, config);

    let device = DeviceType::Cpu;
    let input = Tensor::zeros(&[1, 3], device).unwrap();
    let predictions = ensemble.get_individual_predictions(&input).unwrap();

    assert_eq!(predictions.len(), 2);
    assert_eq!(predictions[0].shape(), input.shape());
    assert_eq!(predictions[1].shape(), input.shape());
}

#[test]
fn test_ensemble_config_creation() {
    let config = ensembling_utils::stacking_config(MetaLearnerType::LinearRegression);
    assert!(matches!(config.method, EnsembleMethod::Stacking { .. }));
    assert!(config.diversity_regularization.is_some());

    let online_config = ensembling_utils::online_adaptive_config(0.01);
    assert!(online_config.online_adaptation.is_some());
}

#[test]
fn test_utility_functions() {
    let accuracies = vec![0.8, 0.82, 0.78, 0.85];
    let diversities = vec![0.3, 0.4, 0.35];

    let optimal_size =
        ensembling_utils::calculate_optimal_ensemble_size(&accuracies, &diversities, 0.1);
    assert!(optimal_size >= 1 && optimal_size <= accuracies.len());

    let improvement = ensembling_utils::estimate_ensemble_improvement(
        &accuracies,
        &EnsembleMethod::SimpleAverage,
    );
    assert!(improvement > 0.0);

    let complexity = ensembling_utils::calculate_ensemble_complexity(
        3,
        &EnsembleMethod::Stacking {
            meta_learner_config: MetaLearnerConfig {
                learner_type: MetaLearnerType::LinearRegression,
                config: HashMap::new(),
                include_original_features: false,
            },
            use_cross_validation: true,
        },
    );
    assert!(complexity > 3.0);
}

#[test]
fn test_config_serialization() {
    let config = EnsembleConfig {
        method: EnsembleMethod::WeightedAverage {
            weights: vec![0.5, 0.3, 0.2],
            learnable: true,
        },
        validation_strategy: EnsembleValidationStrategy::KFold { k: 5 },
        diversity_regularization: Some(DiversityRegularization {
            strength: 0.1,
            metric: DiversityMetric::Disagreement,
            encourage: true,
        }),
        performance_weighting: true,
        online_adaptation: None,
    };

    let serialized = serde_json::to_string(&config).unwrap();
    let deserialized: EnsembleConfig = serde_json::from_str(&serialized).unwrap();

    assert!(matches!(
        deserialized.method,
        EnsembleMethod::WeightedAverage { .. }
    ));
    assert!(deserialized.diversity_regularization.is_some());
    assert!(deserialized.performance_weighting);
}

/// Mock that ignores its input and emits a fixed (1, num_classes) probability vector.
struct FixedProbModel {
    probs: Vec<f32>,
}

impl FixedProbModel {
    fn new(probs: Vec<f32>) -> Self {
        Self { probs }
    }
}

impl Module for FixedProbModel {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        let dims = vec![1, self.probs.len()];
        Tensor::from_data(self.probs.clone(), dims, DeviceType::Cpu)
    }

    fn parameters(&self) -> HashMap<String, torsh_nn::Parameter> {
        HashMap::new()
    }
    fn named_parameters(&self) -> HashMap<String, torsh_nn::Parameter> {
        HashMap::new()
    }
    fn training(&self) -> bool {
        true
    }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn to_device(&mut self, _device: torsh_core::DeviceType) -> Result<()> {
        Ok(())
    }
}

fn vote_config_majority() -> EnsembleConfig {
    EnsembleConfig {
        method: EnsembleMethod::MajorityVoting,
        validation_strategy: EnsembleValidationStrategy::HoldOut { split_ratio: 0.2 },
        diversity_regularization: None,
        performance_weighting: false,
        online_adaptation: None,
    }
}

#[test]
fn test_majority_vote_picks_most_common_class() {
    // Three models: two pick class 1, one picks class 0. Expected result: class 1.
    let models = vec![
        FixedProbModel::new(vec![0.9, 0.05, 0.05]),
        FixedProbModel::new(vec![0.1, 0.8, 0.1]),
        FixedProbModel::new(vec![0.2, 0.7, 0.1]),
    ];
    let mut ensemble = ModelEnsemble::new(models, vote_config_majority());
    let device = DeviceType::Cpu;
    let input = Tensor::zeros(&[1, 3], device).expect("input zeros");
    let prediction = ensemble.predict(&input).expect("predict");
    let data = prediction.to_vec().expect("to_vec");
    assert_eq!(data, vec![0.0, 1.0, 0.0]);
}

#[test]
fn test_soft_vote_averages_probabilities() {
    let probs_a = vec![0.6, 0.3, 0.1];
    let probs_b = vec![0.2, 0.5, 0.3];
    let models = vec![
        FixedProbModel::new(probs_a.clone()),
        FixedProbModel::new(probs_b.clone()),
    ];
    let config = ensembling_utils::simple_average_config();
    let mut ensemble = ModelEnsemble::new(models, config);
    let device = DeviceType::Cpu;
    let input = Tensor::zeros(&[1, 3], device).expect("input zeros");
    let prediction = ensemble.predict(&input).expect("predict");
    let data = prediction.to_vec().expect("to_vec");
    let expected: Vec<f32> = probs_a
        .iter()
        .zip(probs_b.iter())
        .map(|(a, b)| (a + b) / 2.0)
        .collect();
    for (got, want) in data.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-6, "got={got}, want={want}");
    }
}

#[test]
fn test_weighted_average_respects_weights() {
    let probs_a = vec![1.0_f32, 0.0, 0.0];
    let probs_b = vec![0.0_f32, 1.0, 0.0];
    let weights = vec![0.75, 0.25];
    let models = vec![
        FixedProbModel::new(probs_a.clone()),
        FixedProbModel::new(probs_b.clone()),
    ];
    let config = ensembling_utils::weighted_average_config(weights.clone(), false);
    let mut ensemble = ModelEnsemble::new(models, config);
    let device = DeviceType::Cpu;
    let input = Tensor::zeros(&[1, 3], device).expect("input zeros");
    let prediction = ensemble.predict(&input).expect("predict");
    let data = prediction.to_vec().expect("to_vec");
    let expected = vec![0.75_f32, 0.25, 0.0];
    for (got, want) in data.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-6, "got={got}, want={want}");
    }
}

#[test]
fn test_bagging_with_n_models_returns_mean() {
    // Bias-only mocks: their forward shifts the input, so the ensemble of N copies
    // collapses to (input + mean(biases)). With biases summing symmetrically, the
    // ensemble output should equal the input itself.
    let biases = [-1.0_f32, 0.0, 1.0];
    let mean_bias: f32 = biases.iter().sum::<f32>() / biases.len() as f32;
    let models: Vec<MockModel> = biases.iter().map(|b| MockModel::new(*b)).collect();
    let config = ensembling_utils::simple_average_config();
    let mut ensemble = ModelEnsemble::new(models, config);
    let device = DeviceType::Cpu;
    let input = Tensor::ones(&[1, 4], device).expect("input ones");
    let prediction = ensemble.predict(&input).expect("predict");
    let data = prediction.to_vec().expect("to_vec");
    let expected_value = 1.0_f32 + mean_bias;
    for got in &data {
        assert!(
            (got - expected_value).abs() < 1e-6,
            "got={got}, want={expected_value}"
        );
    }
}

#[test]
fn test_stacking_passes_base_outputs_as_features() {
    // The internal `concatenate_base_predictions` is the key building block for
    // stacking — verify it produces the concatenated layout.
    let models = vec![
        FixedProbModel::new(vec![0.1, 0.2, 0.7]),
        FixedProbModel::new(vec![0.3, 0.3, 0.4]),
    ];
    let config = ensembling_utils::simple_average_config();
    let ensemble = ModelEnsemble::new(models, config);
    let device = DeviceType::Cpu;
    let input = Tensor::zeros(&[1, 3], device).expect("input zeros");
    let meta = ensemble.concatenate_base_predictions(&input).expect("concat");
    let data = meta.to_vec().expect("to_vec");
    assert_eq!(data, vec![0.1, 0.2, 0.7, 0.3, 0.3, 0.4]);
    assert_eq!(meta.shape().dims(), &[1, 6]);
}

#[test]
fn test_diversity_disagreement_unanimous_is_zero() {
    let models = vec![
        FixedProbModel::new(vec![0.9, 0.05, 0.05]),
        FixedProbModel::new(vec![0.8, 0.1, 0.1]),
        FixedProbModel::new(vec![0.7, 0.15, 0.15]),
    ];
    let config = ensembling_utils::simple_average_config();
    let ensemble = ModelEnsemble::new(models, config);
    let device = DeviceType::Cpu;
    let inputs: Vec<Tensor> = (0..3)
        .map(|_| Tensor::zeros(&[1, 3], device).expect("input zeros"))
        .collect();
    let measures = ensemble.calculate_diversity(&inputs).expect("diversity");
    assert!(measures.disagreement.abs() < 1e-9);
}

#[test]
fn test_diversity_disagreement_full_split() {
    // Two models, predictions diverge on every sample => disagreement = 1.0.
    let models = vec![
        FixedProbModel::new(vec![1.0, 0.0]),
        FixedProbModel::new(vec![0.0, 1.0]),
    ];
    let config = ensembling_utils::simple_average_config();
    let ensemble = ModelEnsemble::new(models, config);
    let device = DeviceType::Cpu;
    let inputs: Vec<Tensor> = (0..4)
        .map(|_| Tensor::zeros(&[1, 2], device).expect("input zeros"))
        .collect();
    let measures = ensemble.calculate_diversity(&inputs).expect("diversity");
    assert!((measures.disagreement - 1.0).abs() < 1e-9);
    // Kappa should be 0 (no chance-corrected agreement) and Q-statistic should
    // also be well-defined.
    assert!(measures.kappa.is_finite());
    assert!(measures.q_statistic.is_finite());
}

#[test]
fn test_holdout_meta_features_layout() {
    let models = vec![
        FixedProbModel::new(vec![0.1, 0.9]),
        FixedProbModel::new(vec![0.4, 0.6]),
    ];
    let config = ensembling_utils::simple_average_config();
    let ensemble = ModelEnsemble::new(models, config);
    let device = DeviceType::Cpu;
    let inputs: Vec<Tensor> = (0..5)
        .map(|_| Tensor::zeros(&[1, 2], device).expect("input zeros"))
        .collect();
    let targets = inputs.clone();
    let meta = ensemble
        .generate_holdout_meta_features(&inputs, &targets, 0.4)
        .expect("holdout");
    // 40% of 5 = 2 samples; each meta-feature is the 4-element concat.
    assert_eq!(meta.len(), 2);
    for tensor in &meta {
        assert_eq!(tensor.shape().dims(), &[1, 4]);
        let data = tensor.to_vec().expect("to_vec");
        assert_eq!(data, vec![0.1, 0.9, 0.4, 0.6]);
    }
}

#[test]
fn test_cv_meta_features_round_trips_all_inputs() {
    let models = vec![
        FixedProbModel::new(vec![0.2, 0.8]),
        FixedProbModel::new(vec![0.5, 0.5]),
    ];
    let config = ensembling_utils::simple_average_config();
    let ensemble = ModelEnsemble::new(models, config);
    let device = DeviceType::Cpu;
    let inputs: Vec<Tensor> = (0..6)
        .map(|_| Tensor::zeros(&[1, 2], device).expect("input zeros"))
        .collect();
    let targets = inputs.clone();
    let meta = ensemble
        .generate_cv_meta_features(&inputs, &targets, 3)
        .expect("cv meta");
    assert_eq!(meta.len(), 6);
    for tensor in &meta {
        assert_eq!(tensor.shape().dims(), &[1, 4]);
    }
}

#[test]
fn test_pearson_correlation_extremes() {
    let a = vec![1.0_f32, 2.0, 3.0, 4.0];
    let b = vec![2.0_f32, 4.0, 6.0, 8.0];
    let r = pearson_correlation(&a, &b);
    assert!((r - 1.0).abs() < 1e-9, "r={r}");

    let c = vec![4.0_f32, 3.0, 2.0, 1.0];
    let r2 = pearson_correlation(&a, &c);
    assert!((r2 + 1.0).abs() < 1e-9, "r2={r2}");

    let d = vec![1.0_f32, 1.0, 1.0, 1.0];
    let r3 = pearson_correlation(&a, &d);
    assert!(r3.abs() < 1e-9, "r3={r3}");
}

#[test]
fn test_train_dynamic_selection_reweights_models() {
    // The first model is "perfect" (bias 0); the second adds a large bias and is
    // therefore worse. After training on a hold-out split, the better model
    // should receive a strictly larger weight.
    let models = vec![MockModel::new(0.0), MockModel::new(10.0)];
    let config = EnsembleConfig {
        method: EnsembleMethod::DynamicSelection {
            selection_method: SelectionMethod::Confidence { threshold: 0.0 },
            validation_split: 0.5,
        },
        validation_strategy: EnsembleValidationStrategy::HoldOut { split_ratio: 0.5 },
        diversity_regularization: None,
        performance_weighting: false,
        online_adaptation: None,
    };
    let mut ensemble = ModelEnsemble::new(models, config);

    let device = DeviceType::Cpu;
    let inputs: Vec<Tensor> = (0..4)
        .map(|_| Tensor::ones(&[1, 3], device).expect("input ones"))
        .collect();
    let targets: Vec<Tensor> = (0..4)
        .map(|_| Tensor::ones(&[1, 3], device).expect("target ones"))
        .collect();

    ensemble.train(&inputs, &targets).expect("train");
    let weights = ensemble.get_weights();
    assert!(
        weights[0] > weights[1],
        "expected better model to receive more weight: {weights:?}"
    );
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "weights should normalize: sum={sum}");
}

#[test]
fn test_online_weight_update_favors_better_model() {
    let models = vec![MockModel::new(0.0), MockModel::new(5.0)];
    let mut ensemble = ModelEnsemble::new(
        models,
        ensembling_utils::online_adaptive_config(0.5),
    );
    // Seed uniform weights manually (the online config starts empty).
    ensemble
        .set_weights(vec![0.5, 0.5])
        .expect("seed weights");
    let device = DeviceType::Cpu;
    let input = Tensor::ones(&[1, 3], device).expect("input");
    let target = Tensor::ones(&[1, 3], device).expect("target");

    if let Some(cfg) = ensemble.config.online_adaptation.clone() {
        ensemble.initialize_online_adaptation(&cfg);
    }

    // Drive several update steps so EMA weight tracking dominates noise.
    let prediction = Tensor::ones(&[1, 3], device).expect("dummy prediction");
    for _ in 0..10 {
        ensemble
            .update_weights(&input, &target, &prediction)
            .expect("online update");
    }
    let weights = ensemble.get_weights();
    assert!(
        weights[0] > weights[1],
        "online update should prefer the lower-loss model: {weights:?}"
    );
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "weights should normalize: sum={sum}");
}

// ---- Tests for the three newly implemented methods ----

#[test]
fn test_train_stacking_ensemble_produces_meta_learner() {
    // After training, self.meta_learner must be populated; stacking_prediction
    // should produce a tensor with the same element count as the OOF meta-features.
    let models = vec![
        FixedProbModel::new(vec![0.2, 0.8]),
        FixedProbModel::new(vec![0.6, 0.4]),
    ];
    let config = ensembling_utils::stacking_config(MetaLearnerType::RidgeRegression { alpha: 1.0 });
    let mut ensemble = ModelEnsemble::new(models, config);
    let device = DeviceType::Cpu;
    let inputs: Vec<Tensor> = (0..10)
        .map(|_| Tensor::zeros(&[1, 2], device).expect("input"))
        .collect();
    let targets: Vec<Tensor> = (0..10)
        .map(|_| Tensor::from_data(vec![0.0_f32, 1.0], vec![1, 2], device).expect("target"))
        .collect();
    ensemble.train(&inputs, &targets).expect("train stacking");

    // After training, stacking_prediction should not error.
    let input = Tensor::zeros(&[1, 2], device).expect("query input");
    let prediction = ensemble.predict(&input).expect("stacking predict");
    // The meta-learner output is a weighted combination: result must be finite.
    let data = prediction.to_vec().expect("to_vec");
    assert!(!data.is_empty());
    assert!(data.iter().all(|v| v.is_finite()), "all outputs must be finite");
}

#[test]
fn test_train_stacking_weights_sum_to_one() {
    // The ridge regression meta-learner trained inside train_stacking_ensemble
    // should produce internally-normalised per-model weights that sum to 1.
    let models = vec![
        FixedProbModel::new(vec![1.0, 0.0]),
        FixedProbModel::new(vec![0.0, 1.0]),
        FixedProbModel::new(vec![0.5, 0.5]),
    ];
    let config = ensembling_utils::stacking_config(MetaLearnerType::RidgeRegression { alpha: 0.5 });
    let mut ensemble = ModelEnsemble::new(models, config);
    let device = DeviceType::Cpu;
    let inputs: Vec<Tensor> = (0..15)
        .map(|_| Tensor::zeros(&[1, 2], device).expect("input"))
        .collect();
    let targets: Vec<Tensor> = (0..15)
        .map(|_| Tensor::from_data(vec![0.4_f32, 0.6], vec![1, 2], device).expect("target"))
        .collect();
    ensemble.train(&inputs, &targets).expect("train stacking");

    // Weights stored on the ensemble should still be non-negative and normalised.
    let weights = ensemble.get_weights();
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "weights sum={sum}");
    assert!(weights.iter().all(|&w| w >= 0.0), "all weights non-negative");
}

#[test]
fn test_train_mixture_of_experts_updates_weights() {
    // After training MoE, the gate weights should differ from the initial
    // uniform distribution (the gating network learned something).
    let n_models = 3;
    let initial_weight = 1.0 / n_models as f64;
    let models = vec![
        MockModel::new(0.0),
        MockModel::new(2.0),
        MockModel::new(-2.0),
    ];
    let config = EnsembleConfig {
        method: EnsembleMethod::MixtureOfExperts {
            gating_network: GatingNetworkConfig {
                hidden_layers: vec![],
                activation: "softmax".to_string(),
                dropout: 0.0,
                temperature: 1.0,
            },
        },
        validation_strategy: EnsembleValidationStrategy::HoldOut { split_ratio: 0.2 },
        diversity_regularization: None,
        performance_weighting: false,
        online_adaptation: None,
    };
    let mut ensemble = ModelEnsemble::new(models, config);
    let device = DeviceType::Cpu;
    // Input features: a 4-dim vector so the gating network has something to learn.
    let inputs: Vec<Tensor> = (0..20)
        .map(|i| {
            let v = if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
            Tensor::from_data(vec![v, v, v, v], vec![1, 4], device).expect("input")
        })
        .collect();
    let targets: Vec<Tensor> = (0..20)
        .map(|_| Tensor::from_data(vec![0.0_f32, 0.0, 0.0, 0.0], vec![1, 4], device).expect("target"))
        .collect();
    ensemble.train(&inputs, &targets).expect("train MoE");

    let weights = ensemble.get_weights();
    // At least one weight should have deviated from the uniform initialisation.
    let changed = weights.iter().any(|&w| (w - initial_weight).abs() > 1e-9);
    assert!(changed, "MoE gating should update weights away from uniform: {weights:?}");
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "weights should sum to 1: sum={sum}");
}

#[test]
fn test_train_mixture_of_experts_favors_low_loss_expert() {
    // Model 0 (bias=0) perfectly matches the ones target; model 1 (bias=5) does not.
    // MoE gating uses the input features to learn the soft gate weights. We must use
    // non-zero constant inputs so that the linear gating can develop a preference
    // (with all-zero inputs, dot products are identically zero and nothing is learned).
    let models = vec![MockModel::new(0.0), MockModel::new(5.0)];
    let config = EnsembleConfig {
        method: EnsembleMethod::MixtureOfExperts {
            gating_network: GatingNetworkConfig {
                hidden_layers: vec![],
                activation: "softmax".to_string(),
                dropout: 0.0,
                temperature: 1.0,
            },
        },
        validation_strategy: EnsembleValidationStrategy::HoldOut { split_ratio: 0.2 },
        diversity_regularization: None,
        performance_weighting: false,
        online_adaptation: None,
    };
    let mut ensemble = ModelEnsemble::new(models, config);
    let device = DeviceType::Cpu;
    // Non-zero constant input: gating network sees a consistent feature vector and
    // can adjust gate weights to prefer the better expert.
    let inputs: Vec<Tensor> = (0..20)
        .map(|_| Tensor::ones(&[1, 3], device).expect("input"))
        .collect();
    // Target: the identity (what model 0 produces perfectly for ones-input with bias=0).
    let targets: Vec<Tensor> = (0..20)
        .map(|_| Tensor::ones(&[1, 3], device).expect("target"))
        .collect();
    ensemble.train(&inputs, &targets).expect("train MoE");

    let weights = ensemble.get_weights();
    assert!(
        weights[0] > weights[1],
        "MoE should upweight the lower-loss expert: {weights:?}"
    );
}

#[test]
fn test_create_meta_learner_returns_valid_module() {
    // create_meta_learner must return a working Module for each learner type.
    let models = vec![FixedProbModel::new(vec![0.5, 0.5])];
    let config = ensembling_utils::simple_average_config();
    let ensemble = ModelEnsemble::new(models, config);

    for learner_type in [
        MetaLearnerType::LinearRegression,
        MetaLearnerType::RidgeRegression { alpha: 0.5 },
    ] {
        let meta_config = MetaLearnerConfig {
            learner_type,
            config: HashMap::new(),
            include_original_features: false,
        };
        let meta = ensemble.create_meta_learner(&meta_config).expect("create_meta_learner");
        // The meta-learner must be callable via forward().
        let device = torsh_core::DeviceType::Cpu;
        let dummy = Tensor::from_data(vec![0.5_f32, 0.5], vec![1, 2], device).expect("dummy");
        let out = meta.forward(&dummy).expect("meta forward");
        let data = out.to_vec().expect("to_vec");
        assert!(!data.is_empty());
        assert!(data.iter().all(|v| v.is_finite()), "meta-learner output must be finite");
    }
}

#[test]
fn test_create_meta_learner_stacking_weights_are_uniform_without_training() {
    // Without explicit training data, create_meta_learner should return a module
    // with uniform weights (equal contribution from each "model" slot).
    let models = vec![
        FixedProbModel::new(vec![1.0, 0.0]),
        FixedProbModel::new(vec![0.0, 1.0]),
    ];
    let config = ensembling_utils::simple_average_config();
    let ensemble = ModelEnsemble::new(models, config);

    let meta_config = MetaLearnerConfig {
        learner_type: MetaLearnerType::RidgeRegression { alpha: 1.0 },
        config: HashMap::new(),
        include_original_features: false,
    };
    let meta = ensemble.create_meta_learner(&meta_config).expect("create_meta_learner");
    // Provide meta-features = [0.5, 0.5, 0.5, 0.5] (two models, 2 classes each).
    let device = torsh_core::DeviceType::Cpu;
    let input = Tensor::from_data(vec![0.5_f32, 0.5, 0.5, 0.5], vec![1, 4], device).expect("input");
    let out = meta.forward(&input).expect("meta forward");
    let data = out.to_vec().expect("to_vec");
    assert!(!data.is_empty());
    assert!(data.iter().all(|v| v.is_finite()));
}
