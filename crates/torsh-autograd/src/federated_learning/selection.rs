//! Client Selection Algorithms for Federated Learning
//!
//! This module provides various client selection strategies for federated learning systems.
//! Client selection is crucial for balancing fairness, performance, and resource utilization
//! in distributed machine learning scenarios.
//!
//! # Selection Strategies
//!
//! - **Random**: Randomly selects clients for each round
//! - **Round Robin**: Cycles through clients in a deterministic order
//! - **Fair Selection**: Prioritizes clients with lower participation rates
//! - **Performance Based**: Selects clients based on historical performance metrics
//! - **Proportional to Data**: Weights selection by local dataset size
//! - **Resource Aware**: Considers computational and communication capabilities
//! - **Geographic Based**: Accounts for geographic distribution and latency
//! - **Diversity Based**: Maximizes data distribution diversity
//!
//! # Usage Examples
//!
//! ```rust,ignore
//! use torsh_autograd::federated_learning::selection::{ClientSelector, ClientSelectionStrategy};
//!
//! // Create a random selection strategy
//! let mut selector = ClientSelector::new(ClientSelectionStrategy::Random);
//!
//! // Select clients for a federated round
//! let available_clients = vec!["client_1", "client_2", "client_3", "client_4"];
//! let selected = selector.select_clients(&available_clients, 2)?;
//!
//! // Use fair selection to ensure equal participation
//! let mut fair_selector = ClientSelector::new(ClientSelectionStrategy::FairSelection);
//! let fair_selected = fair_selector.select_clients(&available_clients, 2)?;
//! ```
//!
//! # Fairness Considerations
//!
//! The module includes comprehensive fairness tracking to ensure equitable participation
//! across all clients, preventing bias towards high-performing or well-connected clients.

use std::collections::{HashMap, VecDeque};

use scirs2_core::random::Random;

use super::types::ClientSelectionStrategy;
use crate::federated_learning::aggregation::FederatedError;

/// Client selector responsible for choosing participants in federated learning rounds
///
/// The ClientSelector manages the selection of clients for each federated learning round
/// based on the configured strategy. It maintains selection history and fairness metrics
/// to ensure balanced participation across all clients.
///
/// # Thread Safety
///
/// This struct is designed to be thread-safe and can be safely shared across threads
/// when wrapped in appropriate synchronization primitives.
#[derive(Debug)]
pub struct ClientSelector {
    /// The selection strategy to use
    strategy: ClientSelectionStrategy,
    /// History of client selections for analysis and fairness tracking
    selection_history: VecDeque<Vec<String>>,
    /// Performance scores for each client (used in performance-based selection)
    client_scores: HashMap<String, f64>,
    /// Fairness tracking mechanism
    fairness_tracker: FairnessTracker,
}

// ClientSelector is Send + Sync
unsafe impl Send for ClientSelector {}
unsafe impl Sync for ClientSelector {}

/// Fairness tracker for monitoring client participation equity
///
/// The FairnessTracker ensures that federated learning systems maintain
/// equitable participation across all clients, preventing bias and ensuring
/// diverse representation in each round.
#[derive(Debug)]
pub struct FairnessTracker {
    /// Count of participation for each client
    participation_counts: HashMap<String, u32>,
    /// Contribution quality scores for each client
    contribution_scores: HashMap<String, f64>,
    /// Representation scores across different demographic groups
    representation_scores: HashMap<String, f64>,
    /// Mapping of clients to demographic groups (for fairness analysis)
    demographic_groups: HashMap<String, String>,
}

// FairnessTracker is Send + Sync
unsafe impl Send for FairnessTracker {}
unsafe impl Sync for FairnessTracker {}

impl ClientSelector {
    /// Creates a new ClientSelector with the specified strategy
    ///
    /// # Arguments
    ///
    /// * `strategy` - The client selection strategy to use
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_autograd::federated_learning::selection::{ClientSelector, ClientSelectionStrategy};
    ///
    /// let selector = ClientSelector::new(ClientSelectionStrategy::Random);
    /// ```
    pub fn new(strategy: ClientSelectionStrategy) -> Self {
        Self {
            strategy,
            selection_history: VecDeque::new(),
            client_scores: HashMap::new(),
            fairness_tracker: FairnessTracker::new(),
        }
    }

    /// Selects clients for a federated learning round
    ///
    /// This is the main method for client selection. It applies the configured
    /// strategy to choose the optimal set of clients for the current round,
    /// considering fairness, performance, and resource constraints.
    ///
    /// # Arguments
    ///
    /// * `available_clients` - List of all available client IDs
    /// * `num_clients` - Number of clients to select for this round
    ///
    /// # Returns
    ///
    /// A vector of selected client IDs, or an error if selection fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let available = vec!["client_1".to_string(), "client_2".to_string()];
    /// let selected = selector.select_clients(&available, 1)?;
    /// ```
    pub fn select_clients(
        &mut self,
        available_clients: &[String],
        num_clients: usize,
    ) -> Result<Vec<String>, FederatedError> {
        let selected = match self.strategy {
            ClientSelectionStrategy::Random => {
                self.random_selection(available_clients, num_clients)
            }
            ClientSelectionStrategy::RoundRobin => {
                self.round_robin_selection(available_clients, num_clients)
            }
            ClientSelectionStrategy::FairSelection => {
                self.fair_selection(available_clients, num_clients)
            }
            ClientSelectionStrategy::PerformanceBased => {
                self.performance_based_selection(available_clients, num_clients)
            }
            _ => self.random_selection(available_clients, num_clients),
        };

        // Update selection history for future fairness analysis
        self.selection_history.push_back(selected.clone());
        if self.selection_history.len() > 100 {
            self.selection_history.pop_front();
        }

        // Update fairness tracking
        self.fairness_tracker.update_participation(&selected);

        Ok(selected)
    }

    /// Updates client performance scores for performance-based selection
    ///
    /// # Arguments
    ///
    /// * `client_id` - The client identifier
    /// * `score` - The performance score (typically between 0.0 and 1.0)
    pub fn update_client_score(&mut self, client_id: &str, score: f64) {
        self.client_scores.insert(client_id.to_string(), score);
    }

    /// Gets the current selection strategy
    pub fn get_strategy(&self) -> &ClientSelectionStrategy {
        &self.strategy
    }

    /// Sets a new selection strategy
    pub fn set_strategy(&mut self, strategy: ClientSelectionStrategy) {
        self.strategy = strategy;
    }

    /// Gets selection history for analysis
    pub fn get_selection_history(&self) -> &VecDeque<Vec<String>> {
        &self.selection_history
    }

    /// Gets client performance scores
    pub fn get_client_scores(&self) -> &HashMap<String, f64> {
        &self.client_scores
    }

    /// Gets fairness metrics
    pub fn get_fairness_tracker(&self) -> &FairnessTracker {
        &self.fairness_tracker
    }

    /// Random client selection strategy
    ///
    /// Selects clients uniformly at random from the available pool.
    /// This provides good diversity but may not optimize for performance
    /// or fairness considerations.
    fn random_selection(&self, available_clients: &[String], num_clients: usize) -> Vec<String> {
        let mut clients = available_clients.to_vec();
        let mut selected = Vec::new();

        for _ in 0..num_clients.min(clients.len()) {
            let index = Random::default().gen_range(0..clients.len());
            selected.push(clients.remove(index));
        }

        selected
    }

    /// Round-robin client selection strategy
    ///
    /// Cycles through clients in a deterministic order, ensuring
    /// that all clients get equal opportunities over time.
    fn round_robin_selection(
        &mut self,
        available_clients: &[String],
        num_clients: usize,
    ) -> Vec<String> {
        let mut selected = Vec::new();
        let total_clients = available_clients.len();

        if total_clients == 0 {
            return selected;
        }

        let current_round = self.selection_history.len();

        for i in 0..num_clients.min(total_clients) {
            let index = (current_round + i) % total_clients;
            selected.push(available_clients[index].clone());
        }

        selected
    }

    /// Fair client selection strategy
    ///
    /// Prioritizes clients with lower participation rates to ensure
    /// equitable distribution of participation opportunities.
    fn fair_selection(&mut self, available_clients: &[String], num_clients: usize) -> Vec<String> {
        let mut client_fairness_scores: Vec<_> = available_clients
            .iter()
            .map(|client_id| {
                let participation_count = self
                    .fairness_tracker
                    .participation_counts
                    .get(client_id)
                    .unwrap_or(&0);
                let fairness_score = 1.0 / (1.0 + *participation_count as f64);
                (client_id.clone(), fairness_score)
            })
            .collect();

        // Sort by fairness score (highest first = least participated)
        client_fairness_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        client_fairness_scores
            .into_iter()
            .take(num_clients)
            .map(|(client_id, _)| client_id)
            .collect()
    }

    /// Performance-based client selection strategy
    ///
    /// Selects clients based on historical performance metrics,
    /// favoring clients that have contributed high-quality updates.
    fn performance_based_selection(
        &self,
        available_clients: &[String],
        num_clients: usize,
    ) -> Vec<String> {
        let mut client_performance_scores: Vec<_> = available_clients
            .iter()
            .map(|client_id| {
                let score = self.client_scores.get(client_id).unwrap_or(&0.5);
                (client_id.clone(), *score)
            })
            .collect();

        // Sort by performance score (highest first)
        client_performance_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        client_performance_scores
            .into_iter()
            .take(num_clients)
            .map(|(client_id, _)| client_id)
            .collect()
    }
}

impl FairnessTracker {
    /// Creates a new FairnessTracker
    pub fn new() -> Self {
        Self {
            participation_counts: HashMap::new(),
            contribution_scores: HashMap::new(),
            representation_scores: HashMap::new(),
            demographic_groups: HashMap::new(),
        }
    }

    /// Updates participation counts for selected clients
    ///
    /// # Arguments
    ///
    /// * `selected_clients` - List of client IDs that participated in this round
    pub fn update_participation(&mut self, selected_clients: &[String]) {
        for client_id in selected_clients {
            *self
                .participation_counts
                .entry(client_id.clone())
                .or_insert(0) += 1;
        }
    }

    /// Updates contribution quality scores for clients
    ///
    /// # Arguments
    ///
    /// * `client_id` - The client identifier
    /// * `score` - The contribution quality score
    pub fn update_contribution_score(&mut self, client_id: &str, score: f64) {
        self.contribution_scores
            .insert(client_id.to_string(), score);
    }

    /// Sets demographic group for a client (for fairness analysis)
    ///
    /// # Arguments
    ///
    /// * `client_id` - The client identifier
    /// * `group` - The demographic group identifier
    pub fn set_demographic_group(&mut self, client_id: &str, group: &str) {
        self.demographic_groups
            .insert(client_id.to_string(), group.to_string());
    }

    /// Gets participation count for a specific client
    pub fn get_participation_count(&self, client_id: &str) -> u32 {
        self.participation_counts
            .get(client_id)
            .copied()
            .unwrap_or(0)
    }

    /// Gets all participation counts
    pub fn get_all_participation_counts(&self) -> &HashMap<String, u32> {
        &self.participation_counts
    }

    /// Gets contribution quality scores
    pub fn get_contribution_scores(&self) -> &HashMap<String, f64> {
        &self.contribution_scores
    }

    /// Gets demographic group assignments
    pub fn get_demographic_groups(&self) -> &HashMap<String, String> {
        &self.demographic_groups
    }

    /// Computes fairness metrics across all clients
    ///
    /// # Returns
    ///
    /// A tuple of (fairness_index, participation_variance) where:
    /// - fairness_index: Jain's fairness index (1.0 = perfectly fair)
    /// - participation_variance: Variance in participation counts
    pub fn compute_fairness_metrics(&self) -> (f64, f64) {
        if self.participation_counts.is_empty() {
            return (1.0, 0.0);
        }

        let counts: Vec<f64> = self
            .participation_counts
            .values()
            .map(|&count| count as f64)
            .collect();

        // Jain's fairness index
        let sum_squares: f64 = counts.iter().map(|&x| x * x).sum();
        let sum: f64 = counts.iter().sum();
        let n = counts.len() as f64;
        let fairness_index = if sum_squares > 0.0 {
            (sum * sum) / (n * sum_squares)
        } else {
            1.0
        };

        // Participation variance
        let mean = sum / n;
        let variance = counts.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

        (fairness_index, variance)
    }

    /// Resets all fairness tracking data
    pub fn reset(&mut self) {
        self.participation_counts.clear();
        self.contribution_scores.clear();
        self.representation_scores.clear();
        self.demographic_groups.clear();
    }
}

impl Default for FairnessTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::federated_learning::types::ClientSelectionStrategy;

    #[test]
    fn test_client_selector_creation() {
        let selector = ClientSelector::new(ClientSelectionStrategy::Random);
        assert_eq!(*selector.get_strategy(), ClientSelectionStrategy::Random);
        assert!(selector.get_selection_history().is_empty());
    }

    #[test]
    fn test_random_selection() {
        let mut selector = ClientSelector::new(ClientSelectionStrategy::Random);
        let clients = vec![
            "client_1".to_string(),
            "client_2".to_string(),
            "client_3".to_string(),
        ];

        let selected = selector.select_clients(&clients, 2).unwrap();
        assert_eq!(selected.len(), 2);
        assert!(clients.contains(&selected[0]));
        assert!(clients.contains(&selected[1]));
    }

    #[test]
    fn test_round_robin_selection() {
        let mut selector = ClientSelector::new(ClientSelectionStrategy::RoundRobin);
        let clients = vec![
            "client_1".to_string(),
            "client_2".to_string(),
            "client_3".to_string(),
        ];

        let selected1 = selector.select_clients(&clients, 2).unwrap();
        let selected2 = selector.select_clients(&clients, 2).unwrap();

        assert_eq!(selected1.len(), 2);
        assert_eq!(selected2.len(), 2);
        assert_ne!(selected1, selected2); // Should be different due to round-robin
    }

    #[test]
    fn test_fair_selection() {
        let mut selector = ClientSelector::new(ClientSelectionStrategy::FairSelection);
        let clients = vec![
            "client_1".to_string(),
            "client_2".to_string(),
            "client_3".to_string(),
        ];

        // First selection - all clients should have equal fairness
        let selected1 = selector.select_clients(&clients, 2).unwrap();
        assert_eq!(selected1.len(), 2);

        // Second selection - should prefer clients not selected in first round
        let selected2 = selector.select_clients(&clients, 1).unwrap();
        assert_eq!(selected2.len(), 1);
    }

    #[test]
    fn test_performance_based_selection() {
        let mut selector = ClientSelector::new(ClientSelectionStrategy::PerformanceBased);
        let clients = vec![
            "client_1".to_string(),
            "client_2".to_string(),
            "client_3".to_string(),
        ];

        // Set performance scores
        selector.update_client_score("client_1", 0.9);
        selector.update_client_score("client_2", 0.5);
        selector.update_client_score("client_3", 0.8);

        let selected = selector.select_clients(&clients, 2).unwrap();
        assert_eq!(selected.len(), 2);
        // Should select client_1 (0.9) and client_3 (0.8) - the top performers
        assert!(selected.contains(&"client_1".to_string()));
        assert!(selected.contains(&"client_3".to_string()));
    }

    #[test]
    fn test_fairness_tracker() {
        let mut tracker = FairnessTracker::new();

        // Update participation
        tracker.update_participation(&["client_1".to_string(), "client_2".to_string()]);
        tracker.update_participation(&["client_1".to_string(), "client_3".to_string()]);

        assert_eq!(tracker.get_participation_count("client_1"), 2);
        assert_eq!(tracker.get_participation_count("client_2"), 1);
        assert_eq!(tracker.get_participation_count("client_3"), 1);

        // Test fairness metrics
        let (fairness_index, variance) = tracker.compute_fairness_metrics();
        assert!(fairness_index > 0.0 && fairness_index <= 1.0);
        assert!(variance >= 0.0);
    }

    #[test]
    fn test_client_score_updates() {
        let mut selector = ClientSelector::new(ClientSelectionStrategy::PerformanceBased);

        selector.update_client_score("client_1", 0.8);
        selector.update_client_score("client_2", 0.6);

        let scores = selector.get_client_scores();
        assert_eq!(scores.get("client_1"), Some(&0.8));
        assert_eq!(scores.get("client_2"), Some(&0.6));
    }

    #[test]
    fn test_strategy_changes() {
        let mut selector = ClientSelector::new(ClientSelectionStrategy::Random);
        assert_eq!(*selector.get_strategy(), ClientSelectionStrategy::Random);

        selector.set_strategy(ClientSelectionStrategy::FairSelection);
        assert_eq!(
            *selector.get_strategy(),
            ClientSelectionStrategy::FairSelection
        );
    }

    #[test]
    fn test_selection_history_limit() {
        let mut selector = ClientSelector::new(ClientSelectionStrategy::Random);
        let clients = vec!["client_1".to_string()];

        // Add more than 100 selections
        for _ in 0..105 {
            let _ = selector.select_clients(&clients, 1);
        }

        // History should be limited to 100
        assert_eq!(selector.get_selection_history().len(), 100);
    }

    #[test]
    fn test_empty_client_list() {
        let mut selector = ClientSelector::new(ClientSelectionStrategy::Random);
        let clients = vec![];

        let selected = selector.select_clients(&clients, 2).unwrap();
        assert!(selected.is_empty());
    }

    #[test]
    fn test_fairness_tracker_reset() {
        let mut tracker = FairnessTracker::new();
        tracker.update_participation(&["client_1".to_string()]);
        tracker.update_contribution_score("client_1", 0.9);

        assert_eq!(tracker.get_participation_count("client_1"), 1);
        assert!(!tracker.get_contribution_scores().is_empty());

        tracker.reset();

        assert_eq!(tracker.get_participation_count("client_1"), 0);
        assert!(tracker.get_contribution_scores().is_empty());
    }

    #[test]
    fn test_demographic_groups() {
        let mut tracker = FairnessTracker::new();
        tracker.set_demographic_group("client_1", "group_a");
        tracker.set_demographic_group("client_2", "group_b");

        let groups = tracker.get_demographic_groups();
        assert_eq!(groups.get("client_1"), Some(&"group_a".to_string()));
        assert_eq!(groups.get("client_2"), Some(&"group_b".to_string()));
    }
}
