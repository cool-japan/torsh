//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::{ConvergenceCriteria, Individual, MOAlgorithmType, MultiObjectiveAlgorithm, MultiObjectiveOptimizer};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_multi_objective_optimizer_creation() {
        let optimizer = MultiObjectiveOptimizer::new();
        assert!(! optimizer.algorithms.is_empty());
        assert!(optimizer.algorithms.contains_key("NSGA2"));
        assert!(optimizer.algorithms.contains_key("NSGA3"));
        assert!(optimizer.algorithms.contains_key("SPEA2"));
    }
    #[test]
    fn test_algorithm_configuration() {
        let optimizer = MultiObjectiveOptimizer::new();
        let nsga2 = optimizer.algorithms.get("NSGA2").expect("element retrieval should succeed for valid index");
        assert_eq!(nsga2.algorithm_type, MOAlgorithmType::NSGA2);
        assert_eq!(nsga2.population_size, 100);
        assert_eq!(nsga2.max_generations, 500);
        assert_eq!(nsga2.crossover_probability, 0.9);
        assert_eq!(nsga2.mutation_probability, 0.1);
    }
    #[test]
    fn test_dominance_check() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        let ind1 = Individual {
            genotype: vec![1.0, 2.0],
            phenotype: HashMap::new(),
            objectives: vec![1.0, 2.0],
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            age: 0,
        };
        let ind2 = Individual {
            genotype: vec![2.0, 3.0],
            phenotype: HashMap::new(),
            objectives: vec![2.0, 3.0],
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            age: 0,
        };
        optimizer.population = vec![ind1, ind2];
        assert!(optimizer.dominates(0, 1));
        assert!(! optimizer.dominates(1, 0));
    }
    #[test]
    fn test_population_initialization() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        optimizer.initialize_population(50, 3);
        assert_eq!(optimizer.population.len(), 50);
        for individual in &optimizer.population {
            assert_eq!(individual.genotype.len(), 10);
            assert_eq!(individual.objectives.len(), 3);
            assert_eq!(individual.rank, 0);
            assert_eq!(individual.age, 0);
        }
    }
    #[test]
    fn test_metrics_calculation() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        optimizer.initialize_population(20, 2);
        for individual in &mut optimizer.population {
            individual.objectives[0] = individual.genotype[0].abs();
            individual.objectives[1] = individual.genotype[1].abs();
        }
        optimizer.fast_non_dominated_sort();
        optimizer.update_metrics(1);
        let metrics = &optimizer.performance_metrics;
        assert_eq!(metrics.generations, 1);
        assert!(metrics.hypervolume >= 0.0);
        assert!(metrics.diversity >= 0.0);
        assert!(metrics.solution_count > 0);
    }
    #[test]
    fn test_nsga2_algorithm() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        let objectives = vec!["performance".to_string(), "memory".to_string()];
        let constraints = vec![];
        let result = optimizer.optimize(&objectives, &constraints, "NSGA2");
        assert!(result.is_ok());
    }
    #[test]
    fn test_pareto_solution_extraction() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        optimizer.initialize_population(10, 2);
        for (i, individual) in optimizer.population.iter_mut().enumerate() {
            individual.objectives[0] = i as f64;
            individual.objectives[1] = (10 - i) as f64;
            individual.rank = if i < 5 { 0 } else { 1 };
        }
        let pareto_solutions = optimizer.extract_pareto_front().expect("Pareto front extraction should succeed");
        assert_eq!(pareto_solutions.len(), 5);
        for solution in pareto_solutions {
            assert_eq!(solution.rank, 0);
            assert!(! solution.parameters.is_empty());
            assert!(! solution.objectives.is_empty());
        }
    }
    #[test]
    fn test_convergence_detection() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        let algorithm = optimizer.algorithms.get("NSGA2").expect("element retrieval should succeed for valid index").clone();
        let mut algorithm_short = algorithm.clone();
        algorithm_short.convergence_criteria.max_generations = 5;
        assert!(optimizer.check_convergence(5, & algorithm_short));
        assert!(! optimizer.check_convergence(3, & algorithm_short));
        let mut algorithm_hv = algorithm.clone();
        algorithm_hv.convergence_criteria.target_hypervolume = Some(0.5);
        optimizer.performance_metrics.hypervolume = 0.6;
        assert!(optimizer.check_convergence(10, & algorithm_hv));
        optimizer.performance_metrics.hypervolume = 0.3;
        assert!(! optimizer.check_convergence(10, & algorithm_hv));
    }
    #[test]
    fn test_algorithm_management() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        let initial_count = optimizer.list_algorithms().len();
        let custom_algorithm = MultiObjectiveAlgorithm {
            name: "Custom".to_string(),
            algorithm_type: MOAlgorithmType::Custom,
            parameters: HashMap::new(),
            population_size: 50,
            max_generations: 100,
            convergence_criteria: ConvergenceCriteria::default(),
            crossover_probability: 0.8,
            mutation_probability: 0.2,
            selection_pressure: 1.5,
            elite_ratio: 0.05,
        };
        optimizer.add_algorithm("Custom".to_string(), custom_algorithm);
        assert_eq!(optimizer.list_algorithms().len(), initial_count + 1);
        assert!(optimizer.algorithms.contains_key("Custom"));
        let removed = optimizer.remove_algorithm("Custom");
        assert!(removed.is_some());
        assert_eq!(optimizer.list_algorithms().len(), initial_count);
        assert!(! optimizer.algorithms.contains_key("Custom"));
    }
    #[test]
    fn test_crossover_operation() {
        let optimizer = MultiObjectiveOptimizer::new();
        let parent1 = Individual {
            genotype: vec![0.5, - 0.5, 0.0],
            phenotype: HashMap::new(),
            objectives: vec![1.0, 2.0],
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            age: 0,
        };
        let parent2 = Individual {
            genotype: vec![- 0.5, 0.5, 1.0],
            phenotype: HashMap::new(),
            objectives: vec![2.0, 1.0],
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            age: 0,
        };
        let (child1, child2) = optimizer.simulated_binary_crossover(&parent1, &parent2);
        assert_eq!(child1.genotype.len(), 3);
        assert_eq!(child2.genotype.len(), 3);
        for &gene in &child1.genotype {
            assert!(gene >= - 1.0 && gene <= 1.0);
        }
        for &gene in &child2.genotype {
            assert!(gene >= - 1.0 && gene <= 1.0);
        }
    }
    #[test]
    fn test_tournament_selection() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        optimizer.initialize_population(10, 2);
        for (i, individual) in optimizer.population.iter_mut().enumerate() {
            individual.rank = i / 3;
            individual.crowding_distance = (i % 3) as f64;
        }
        let mut selections = Vec::new();
        for _ in 0..20 {
            selections.push(optimizer.tournament_selection());
        }
        for &selected in &selections {
            assert!(selected < optimizer.population.len());
        }
        assert!(! selections.is_empty());
    }
}
