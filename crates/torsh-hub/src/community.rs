//! Community features for ToRSh Hub
//!
//! This module provides community-driven features including model ratings,
//! comments, discussions, contributions, and challenges.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use torsh_core::error::{GeneralError, Result, TorshError};
use uuid::Uuid;

/// User identifier
pub type UserId = String;

/// Model identifier  
pub type ModelId = String;

/// Discussion identifier
pub type DiscussionId = String;

/// Challenge identifier
pub type ChallengeId = String;

/// User rating for a model (1-5 stars)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRating {
    pub user_id: UserId,
    pub model_id: ModelId,
    pub rating: u8, // 1-5 stars
    pub review: Option<String>,
    pub timestamp: u64,
    pub helpful_votes: u32,
    pub categories: Vec<RatingCategory>,
}

/// Categories for detailed ratings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RatingCategory {
    Accuracy,
    Performance,
    EaseOfUse,
    Documentation,
    Reliability,
    Novelty,
}

/// Aggregated rating statistics for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRatingStats {
    pub model_id: ModelId,
    pub average_rating: f64,
    pub total_ratings: u32,
    pub rating_distribution: [u32; 5], // counts for 1-5 stars
    pub category_ratings: HashMap<RatingCategory, f64>,
    pub recent_ratings_trend: Vec<f64>, // last 30 days average
}

/// Comment on a model or discussion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    pub id: String,
    pub author_id: UserId,
    pub content: String,
    pub timestamp: u64,
    pub parent_id: Option<String>, // for threaded comments
    pub model_id: Option<ModelId>,
    pub discussion_id: Option<DiscussionId>,
    pub upvotes: u32,
    pub downvotes: u32,
    pub is_edited: bool,
    pub edit_timestamp: Option<u64>,
    pub is_pinned: bool,
    pub is_moderator_comment: bool,
}

/// Discussion thread
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Discussion {
    pub id: DiscussionId,
    pub title: String,
    pub description: String,
    pub author_id: UserId,
    pub created_at: u64,
    pub updated_at: u64,
    pub category: DiscussionCategory,
    pub tags: Vec<String>,
    pub status: DiscussionStatus,
    pub views: u32,
    pub participants: Vec<UserId>,
    pub is_pinned: bool,
    pub is_locked: bool,
    pub related_models: Vec<ModelId>,
}

/// Categories for discussions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscussionCategory {
    General,
    ModelRequests,
    BugReports,
    FeatureRequests,
    Tutorials,
    Research,
    Benchmarks,
    Announcements,
}

/// Status of a discussion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscussionStatus {
    Open,
    Resolved,
    InProgress,
    Closed,
}

/// User contribution to the community
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contribution {
    pub id: String,
    pub contributor_id: UserId,
    pub contribution_type: ContributionType,
    pub title: String,
    pub description: String,
    pub timestamp: u64,
    pub status: ContributionStatus,
    pub impact_score: f64,
    pub related_models: Vec<ModelId>,
    pub metadata: HashMap<String, String>,
}

/// Types of contributions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContributionType {
    ModelUpload,
    DocumentationImprovement,
    BugFix,
    FeatureImplementation,
    Tutorial,
    Benchmark,
    DatasetContribution,
    CodeOptimization,
}

/// Status of a contribution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContributionStatus {
    Submitted,
    UnderReview,
    Approved,
    Rejected,
    NeedsRevision,
}

/// Community challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenge {
    pub id: ChallengeId,
    pub title: String,
    pub description: String,
    pub organizer_id: UserId,
    pub created_at: u64,
    pub start_date: u64,
    pub end_date: u64,
    pub challenge_type: ChallengeType,
    pub prize_pool: Option<String>,
    pub rules: String,
    pub evaluation_criteria: Vec<EvaluationCriteria>,
    pub participants: Vec<ChallengeParticipant>,
    pub submissions: Vec<ChallengeSubmission>,
    pub status: ChallengeStatus,
    pub winner_ids: Vec<UserId>,
    pub tags: Vec<String>,
}

/// Types of challenges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChallengeType {
    ModelAccuracy,
    ModelEfficiency,
    NovelArchitecture,
    BenchmarkImprovement,
    RealWorldApplication,
    OpenProblem,
}

/// Evaluation criteria for challenges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationCriteria {
    pub name: String,
    pub description: String,
    pub weight: f64,
    pub metric_type: MetricType,
}

/// Types of metrics for evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    Accuracy,
    F1Score,
    Latency,
    ThroughputOps,
    MemoryUsage,
    ModelSize,
    Custom,
}

/// Challenge participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeParticipant {
    pub user_id: UserId,
    pub joined_at: u64,
    pub team_name: Option<String>,
    pub team_members: Vec<UserId>,
}

/// Challenge submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeSubmission {
    pub id: String,
    pub participant_id: UserId,
    pub model_id: ModelId,
    pub submitted_at: u64,
    pub description: String,
    pub metrics: HashMap<String, f64>,
    pub code_repository: Option<String>,
    pub paper_link: Option<String>,
    pub is_final: bool,
}

/// Status of a challenge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChallengeStatus {
    Upcoming,
    Active,
    Evaluation,
    Completed,
    Cancelled,
}

/// Manager for community features
pub struct CommunityManager {
    ratings: HashMap<String, ModelRating>,
    rating_stats: HashMap<ModelId, ModelRatingStats>,
    comments: HashMap<String, Comment>,
    discussions: HashMap<DiscussionId, Discussion>,
    contributions: HashMap<String, Contribution>,
    challenges: HashMap<ChallengeId, Challenge>,
    user_profiles: HashMap<UserId, UserProfile>,
}

/// User profile in the community
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: UserId,
    pub username: String,
    pub reputation_score: f64,
    pub contribution_count: u32,
    pub ratings_given: u32,
    pub discussions_started: u32,
    pub challenges_participated: u32,
    pub badges: Vec<Badge>,
    pub joined_at: u64,
    pub last_active: u64,
}

/// User badges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Badge {
    TopContributor,
    HelpfulReviewer,
    ChallengeWinner,
    BugHunter,
    Mentor,
    EarlyAdopter,
    Innovator,
    TeamPlayer,
}

impl CommunityManager {
    /// Create a new community manager
    pub fn new() -> Self {
        Self {
            ratings: HashMap::new(),
            rating_stats: HashMap::new(),
            comments: HashMap::new(),
            discussions: HashMap::new(),
            contributions: HashMap::new(),
            challenges: HashMap::new(),
            user_profiles: HashMap::new(),
        }
    }

    /// Add a rating for a model
    pub fn add_rating(&mut self, mut rating: ModelRating) -> Result<()> {
        if rating.rating < 1 || rating.rating > 5 {
            return Err(TorshError::General(GeneralError::InvalidArgument(
                "Rating must be between 1 and 5".to_string(),
            )));
        }

        let rating_id = format!("{}_{}", rating.user_id, rating.model_id);
        rating.timestamp = current_timestamp();

        // Remove old rating if exists
        if let Some(old_rating) = self.ratings.remove(&rating_id) {
            self.update_rating_stats_remove(&old_rating);
        }

        self.update_rating_stats_add(&rating);
        self.ratings.insert(rating_id, rating);

        Ok(())
    }

    /// Get rating statistics for a model
    pub fn get_rating_stats(&self, model_id: &str) -> Option<&ModelRatingStats> {
        self.rating_stats.get(model_id)
    }

    /// Get all ratings for a model
    pub fn get_model_ratings(&self, model_id: &str) -> Vec<&ModelRating> {
        self.ratings
            .values()
            .filter(|r| r.model_id == model_id)
            .collect()
    }

    /// Add a comment
    pub fn add_comment(&mut self, mut comment: Comment) -> Result<String> {
        if comment.content.trim().is_empty() {
            return Err(TorshError::General(GeneralError::InvalidArgument(
                "Comment content cannot be empty".to_string(),
            )));
        }

        comment.id = Uuid::new_v4().to_string();
        comment.timestamp = current_timestamp();

        let comment_id = comment.id.clone();
        self.comments.insert(comment_id.clone(), comment);

        Ok(comment_id)
    }

    /// Get comments for a model
    pub fn get_model_comments(&self, model_id: &str) -> Vec<&Comment> {
        self.comments
            .values()
            .filter(|c| c.model_id.as_deref() == Some(model_id))
            .collect()
    }

    /// Get comments for a discussion
    pub fn get_discussion_comments(&self, discussion_id: &str) -> Vec<&Comment> {
        self.comments
            .values()
            .filter(|c| c.discussion_id.as_deref() == Some(discussion_id))
            .collect()
    }

    /// Create a new discussion
    pub fn create_discussion(&mut self, mut discussion: Discussion) -> Result<DiscussionId> {
        if discussion.title.trim().is_empty() {
            return Err(TorshError::General(GeneralError::InvalidArgument(
                "Discussion title cannot be empty".to_string(),
            )));
        }

        discussion.id = Uuid::new_v4().to_string();
        discussion.created_at = current_timestamp();
        discussion.updated_at = discussion.created_at;

        let discussion_id = discussion.id.clone();
        self.discussions.insert(discussion_id.clone(), discussion);

        Ok(discussion_id)
    }

    /// Get discussions by category
    pub fn get_discussions_by_category(&self, category: DiscussionCategory) -> Vec<&Discussion> {
        self.discussions
            .values()
            .filter(|d| d.category == category)
            .collect()
    }

    /// Search discussions
    pub fn search_discussions(&self, query: &str, tags: Option<&[String]>) -> Vec<&Discussion> {
        let query_lower = query.to_lowercase();

        self.discussions
            .values()
            .filter(|d| {
                let title_match = d.title.to_lowercase().contains(&query_lower);
                let desc_match = d.description.to_lowercase().contains(&query_lower);
                let tag_match = tags.map_or(true, |search_tags| {
                    search_tags.iter().any(|tag| d.tags.contains(tag))
                });

                (title_match || desc_match) && tag_match
            })
            .collect()
    }

    /// Submit a contribution
    pub fn submit_contribution(&mut self, mut contribution: Contribution) -> Result<String> {
        if contribution.title.trim().is_empty() {
            return Err(TorshError::General(GeneralError::InvalidArgument(
                "Contribution title cannot be empty".to_string(),
            )));
        }

        contribution.id = Uuid::new_v4().to_string();
        contribution.timestamp = current_timestamp();
        contribution.status = ContributionStatus::Submitted;

        let contribution_id = contribution.id.clone();
        self.contributions
            .insert(contribution_id.clone(), contribution);

        Ok(contribution_id)
    }

    /// Get contributions by user
    pub fn get_user_contributions(&self, user_id: &str) -> Vec<&Contribution> {
        self.contributions
            .values()
            .filter(|c| c.contributor_id == user_id)
            .collect()
    }

    /// Create a new challenge
    pub fn create_challenge(&mut self, mut challenge: Challenge) -> Result<ChallengeId> {
        if challenge.title.trim().is_empty() {
            return Err(TorshError::General(GeneralError::InvalidArgument(
                "Challenge title cannot be empty".to_string(),
            )));
        }

        if challenge.end_date <= challenge.start_date {
            return Err(TorshError::General(GeneralError::InvalidArgument(
                "Challenge end date must be after start date".to_string(),
            )));
        }

        challenge.id = Uuid::new_v4().to_string();
        challenge.created_at = current_timestamp();
        challenge.status = if challenge.start_date > current_timestamp() {
            ChallengeStatus::Upcoming
        } else {
            ChallengeStatus::Active
        };

        let challenge_id = challenge.id.clone();
        self.challenges.insert(challenge_id.clone(), challenge);

        Ok(challenge_id)
    }

    /// Join a challenge
    pub fn join_challenge(
        &mut self,
        challenge_id: &str,
        participant: ChallengeParticipant,
    ) -> Result<()> {
        let challenge = self.challenges.get_mut(challenge_id).ok_or_else(|| {
            TorshError::General(GeneralError::RuntimeError(
                "Challenge not found".to_string(),
            ))
        })?;

        if challenge.status != ChallengeStatus::Active
            && challenge.status != ChallengeStatus::Upcoming
        {
            return Err(TorshError::General(GeneralError::InvalidArgument(
                "Challenge is not accepting participants".to_string(),
            )));
        }

        if challenge
            .participants
            .iter()
            .any(|p| p.user_id == participant.user_id)
        {
            return Err(TorshError::General(GeneralError::InvalidArgument(
                "User already participating in challenge".to_string(),
            )));
        }

        challenge.participants.push(participant);
        Ok(())
    }

    /// Submit to a challenge
    pub fn submit_to_challenge(
        &mut self,
        challenge_id: &str,
        submission: ChallengeSubmission,
    ) -> Result<String> {
        let challenge = self.challenges.get_mut(challenge_id).ok_or_else(|| {
            TorshError::General(GeneralError::RuntimeError(
                "Challenge not found".to_string(),
            ))
        })?;

        if challenge.status != ChallengeStatus::Active {
            return Err(TorshError::General(GeneralError::InvalidArgument(
                "Challenge is not accepting submissions".to_string(),
            )));
        }

        if !challenge
            .participants
            .iter()
            .any(|p| p.user_id == submission.participant_id)
        {
            return Err(TorshError::General(GeneralError::InvalidArgument(
                "User is not participating in this challenge".to_string(),
            )));
        }

        challenge.submissions.push(submission.clone());
        Ok(submission.id)
    }

    /// Get active challenges
    pub fn get_active_challenges(&self) -> Vec<&Challenge> {
        self.challenges
            .values()
            .filter(|c| c.status == ChallengeStatus::Active)
            .collect()
    }

    /// Get leaderboard for a challenge
    pub fn get_challenge_leaderboard(
        &self,
        challenge_id: &str,
        metric: &str,
    ) -> Result<Vec<(&ChallengeSubmission, f64)>> {
        let challenge = self.challenges.get(challenge_id).ok_or_else(|| {
            TorshError::General(GeneralError::RuntimeError(
                "Challenge not found".to_string(),
            ))
        })?;

        let mut submissions: Vec<_> = challenge
            .submissions
            .iter()
            .filter_map(|s| s.metrics.get(metric).map(|score| (s, *score)))
            .collect();

        // Sort by score (descending for most metrics)
        submissions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(submissions)
    }

    /// Update user profile
    pub fn update_user_profile(&mut self, profile: UserProfile) {
        self.user_profiles.insert(profile.user_id.clone(), profile);
    }

    /// Get user profile
    pub fn get_user_profile(&self, user_id: &str) -> Option<&UserProfile> {
        self.user_profiles.get(user_id)
    }

    /// Calculate user reputation
    pub fn calculate_user_reputation(&self, user_id: &str) -> f64 {
        let mut reputation = 0.0;

        // Points from ratings given
        let ratings_given = self
            .ratings
            .values()
            .filter(|r| r.user_id == user_id)
            .count();
        reputation += ratings_given as f64 * 1.0;

        // Points from helpful reviews
        let helpful_votes: u32 = self
            .ratings
            .values()
            .filter(|r| r.user_id == user_id)
            .map(|r| r.helpful_votes)
            .sum();
        reputation += helpful_votes as f64 * 2.0;

        // Points from contributions
        let contributions = self.get_user_contributions(user_id);
        for contribution in contributions {
            reputation += match contribution.status {
                ContributionStatus::Approved => contribution.impact_score * 10.0,
                ContributionStatus::UnderReview => contribution.impact_score * 2.0,
                _ => 0.0,
            };
        }

        // Points from discussions
        let discussions_started = self
            .discussions
            .values()
            .filter(|d| d.author_id == user_id)
            .count();
        reputation += discussions_started as f64 * 5.0;

        // Points from challenge participation
        let challenge_participations = self
            .challenges
            .values()
            .filter(|c| c.participants.iter().any(|p| p.user_id == user_id))
            .count();
        reputation += challenge_participations as f64 * 3.0;

        // Bonus for challenge wins
        let challenge_wins = self
            .challenges
            .values()
            .filter(|c| c.winner_ids.contains(&user_id.to_string()))
            .count();
        reputation += challenge_wins as f64 * 50.0;

        reputation
    }

    fn update_rating_stats_add(&mut self, rating: &ModelRating) {
        let stats = self
            .rating_stats
            .entry(rating.model_id.clone())
            .or_insert_with(|| ModelRatingStats {
                model_id: rating.model_id.clone(),
                average_rating: 0.0,
                total_ratings: 0,
                rating_distribution: [0; 5],
                category_ratings: HashMap::new(),
                recent_ratings_trend: Vec::new(),
            });

        let old_total = stats.total_ratings;
        let old_sum = stats.average_rating * old_total as f64;

        stats.total_ratings += 1;
        stats.average_rating = (old_sum + rating.rating as f64) / stats.total_ratings as f64;
        stats.rating_distribution[(rating.rating - 1) as usize] += 1;

        // Update category ratings
        for &category in &rating.categories {
            let category_stats = stats.category_ratings.entry(category).or_insert(0.0);
            *category_stats = (*category_stats + rating.rating as f64) / 2.0; // Simple average for now
        }
    }

    fn update_rating_stats_remove(&mut self, rating: &ModelRating) {
        if let Some(stats) = self.rating_stats.get_mut(&rating.model_id) {
            if stats.total_ratings > 1 {
                let old_sum = stats.average_rating * stats.total_ratings as f64;
                stats.total_ratings -= 1;
                stats.average_rating =
                    (old_sum - rating.rating as f64) / stats.total_ratings as f64;
                stats.rating_distribution[(rating.rating - 1) as usize] =
                    stats.rating_distribution[(rating.rating - 1) as usize].saturating_sub(1);
            } else {
                // Remove stats if this was the last rating
                self.rating_stats.remove(&rating.model_id);
            }
        }
    }
}

impl Default for CommunityManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_rating() {
        let mut manager = CommunityManager::new();

        let rating = ModelRating {
            user_id: "user1".to_string(),
            model_id: "model1".to_string(),
            rating: 5,
            review: Some("Great model!".to_string()),
            timestamp: 0,
            helpful_votes: 0,
            categories: vec![RatingCategory::Accuracy, RatingCategory::Performance],
        };

        assert!(manager.add_rating(rating).is_ok());

        let stats = manager.get_rating_stats("model1").unwrap();
        assert_eq!(stats.average_rating, 5.0);
        assert_eq!(stats.total_ratings, 1);
    }

    #[test]
    fn test_invalid_rating() {
        let mut manager = CommunityManager::new();

        let rating = ModelRating {
            user_id: "user1".to_string(),
            model_id: "model1".to_string(),
            rating: 6, // Invalid rating
            review: None,
            timestamp: 0,
            helpful_votes: 0,
            categories: vec![],
        };

        assert!(manager.add_rating(rating).is_err());
    }

    #[test]
    fn test_discussion_creation() {
        let mut manager = CommunityManager::new();

        let discussion = Discussion {
            id: String::new(), // Will be set by manager
            title: "How to optimize model performance?".to_string(),
            description: "Looking for tips on model optimization".to_string(),
            author_id: "user1".to_string(),
            created_at: 0,
            updated_at: 0,
            category: DiscussionCategory::General,
            tags: vec!["optimization".to_string(), "performance".to_string()],
            status: DiscussionStatus::Open,
            views: 0,
            participants: vec![],
            is_pinned: false,
            is_locked: false,
            related_models: vec![],
        };

        let discussion_id = manager.create_discussion(discussion).unwrap();
        assert!(!discussion_id.is_empty());

        let created_discussion = manager.discussions.get(&discussion_id).unwrap();
        assert_eq!(
            created_discussion.title,
            "How to optimize model performance?"
        );
    }

    #[test]
    fn test_challenge_creation() {
        let mut manager = CommunityManager::new();

        let challenge = Challenge {
            id: String::new(),
            title: "Image Classification Challenge".to_string(),
            description: "Build the best image classifier".to_string(),
            organizer_id: "organizer1".to_string(),
            created_at: 0,
            start_date: current_timestamp() + 86400, // Tomorrow
            end_date: current_timestamp() + 86400 * 30, // 30 days from now
            challenge_type: ChallengeType::ModelAccuracy,
            prize_pool: Some("$10,000".to_string()),
            rules: "Standard classification rules apply".to_string(),
            evaluation_criteria: vec![],
            participants: vec![],
            submissions: vec![],
            status: ChallengeStatus::Upcoming,
            winner_ids: vec![],
            tags: vec!["vision".to_string()],
        };

        let challenge_id = manager.create_challenge(challenge).unwrap();
        assert!(!challenge_id.is_empty());
    }
}
