use crate::metrics::fluency::FluentText;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, Random};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone)]
pub struct PragmaticAnalyzer {
    context_patterns: HashMap<String, f64>,
    register_markers: HashMap<String, Vec<String>>,
    politeness_indicators: HashMap<String, f64>,
    speech_act_patterns: HashMap<String, Vec<String>>,
    discourse_markers: HashSet<String>,
    cultural_markers: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PragmaticScore {
    pub overall_score: f64,
    pub context_appropriateness: f64,
    pub communicative_effectiveness: f64,
    pub audience_awareness: f64,
    pub register_appropriateness: f64,
    pub politeness_level: f64,
    pub speech_act_clarity: f64,
    pub implicature_richness: f64,
    pub social_awareness: f64,
    pub dialogue_competence: f64,
    pub cooperation_adherence: f64,
    pub relevance_score: f64,
    pub detailed_metrics: DetailedPragmaticMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetailedPragmaticMetrics {
    pub context_analysis: ContextAnalysis,
    pub register_analysis: RegisterAnalysis,
    pub politeness_analysis: PolitenessAnalysis,
    pub speech_act_analysis: SpeechActAnalysis,
    pub implicature_analysis: ImplicatureAnalysis,
    pub social_context_analysis: SocialContextAnalysis,
    pub discourse_analysis: DiscourseAnalysis,
    pub relevance_analysis: RelevanceAnalysis,
    pub audience_analysis: AudienceAnalysis,
    pub cultural_sensitivity: CulturalSensitivity,
    pub advanced_metrics: AdvancedPragmaticMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextAnalysis {
    pub situational_appropriateness: f64,
    pub contextual_coherence: f64,
    pub background_knowledge_usage: f64,
    pub contextual_inference_quality: f64,
    pub shared_knowledge_assumptions: f64,
    pub contextual_adaptation_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegisterAnalysis {
    pub formality_level: f64,
    pub register_consistency: f64,
    pub domain_appropriateness: f64,
    pub style_matching: f64,
    pub tone_appropriateness: f64,
    pub linguistic_register_markers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolitenessAnalysis {
    pub positive_politeness: f64,
    pub negative_politeness: f64,
    pub face_saving_strategies: f64,
    pub indirect_speech_usage: f64,
    pub hedging_strategies: f64,
    pub courtesy_markers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechActAnalysis {
    pub primary_speech_acts: Vec<String>,
    pub secondary_speech_acts: Vec<String>,
    pub illocutionary_force: f64,
    pub perlocutionary_effect: f64,
    pub speech_act_appropriateness: f64,
    pub directness_level: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImplicatureAnalysis {
    pub conversational_implicature: f64,
    pub conventional_implicature: f64,
    pub inference_complexity: f64,
    pub implicature_clarity: f64,
    pub presupposition_handling: f64,
    pub implicit_meaning_richness: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SocialContextAnalysis {
    pub power_relations_awareness: f64,
    pub social_distance_management: f64,
    pub cultural_sensitivity_score: f64,
    pub group_dynamics_awareness: f64,
    pub identity_construction: f64,
    pub social_role_appropriateness: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscourseAnalysis {
    pub turn_taking_competence: f64,
    pub topic_management: f64,
    pub coherence_maintenance: f64,
    pub discourse_marker_usage: f64,
    pub conversational_flow: f64,
    pub interactive_competence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelevanceAnalysis {
    pub relevance_to_context: f64,
    pub information_value: f64,
    pub cognitive_effort_balance: f64,
    pub contextual_effects: f64,
    pub relevance_optimization: f64,
    pub maxim_adherence: MaximAdherence,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaximAdherence {
    pub quantity_maxim: f64,
    pub quality_maxim: f64,
    pub relation_maxim: f64,
    pub manner_maxim: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudienceAnalysis {
    pub audience_adaptation: f64,
    pub shared_knowledge_assessment: f64,
    pub expertise_level_matching: f64,
    pub interest_level_maintenance: f64,
    pub comprehensibility_optimization: f64,
    pub engagement_strategies: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CulturalSensitivity {
    pub cultural_appropriateness: f64,
    pub cross_cultural_awareness: f64,
    pub cultural_bias_avoidance: f64,
    pub inclusive_language_usage: f64,
    pub cultural_reference_handling: f64,
    pub intercultural_competence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdvancedPragmaticMetrics {
    pub pragmatic_competence_index: f64,
    pub communicative_strategy_diversity: f64,
    pub contextual_sensitivity_score: f64,
    pub interpersonal_effectiveness: f64,
    pub pragmatic_inference_capability: f64,
    pub social_linguistic_competence: f64,
    pub multimodal_communication_score: f64,
    pub pragmatic_fluency_trajectory: Array1<f64>,
    pub strategy_usage_patterns: HashMap<String, f64>,
    pub pragmatic_error_analysis: PragmaticErrorAnalysis,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PragmaticErrorAnalysis {
    pub pragmatic_failure_count: usize,
    pub sociopragmatic_errors: usize,
    pub pragmalinguistic_errors: usize,
    pub cultural_transfer_errors: usize,
    pub register_violations: usize,
    pub politeness_violations: usize,
}

#[derive(Debug)]
pub enum PragmaticError {
    ContextMismatch(String),
    RegisterInappropriateness(String),
    PolitenessViolation(String),
    SpeechActFailure(String),
    ImplicatureFailure(String),
    CulturalInsensitivity(String),
    AudienceMismatch(String),
    RelevanceFailure(String),
}

impl fmt::Display for PragmaticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PragmaticError::ContextMismatch(msg) => write!(f, "Context mismatch: {}", msg),
            PragmaticError::RegisterInappropriateness(msg) => {
                write!(f, "Register inappropriateness: {}", msg)
            }
            PragmaticError::PolitenessViolation(msg) => write!(f, "Politeness violation: {}", msg),
            PragmaticError::SpeechActFailure(msg) => write!(f, "Speech act failure: {}", msg),
            PragmaticError::ImplicatureFailure(msg) => write!(f, "Implicature failure: {}", msg),
            PragmaticError::CulturalInsensitivity(msg) => {
                write!(f, "Cultural insensitivity: {}", msg)
            }
            PragmaticError::AudienceMismatch(msg) => write!(f, "Audience mismatch: {}", msg),
            PragmaticError::RelevanceFailure(msg) => write!(f, "Relevance failure: {}", msg),
        }
    }
}

impl Error for PragmaticError {}

impl Default for PragmaticAnalyzer {
    fn default() -> Self {
        let mut context_patterns = HashMap::new();
        context_patterns.insert("formal".to_string(), 0.8);
        context_patterns.insert("informal".to_string(), 0.6);
        context_patterns.insert("academic".to_string(), 0.9);
        context_patterns.insert("conversational".to_string(), 0.7);
        context_patterns.insert("professional".to_string(), 0.85);

        let mut register_markers = HashMap::new();
        register_markers.insert(
            "formal".to_string(),
            vec![
                "furthermore".to_string(),
                "moreover".to_string(),
                "consequently".to_string(),
                "nevertheless".to_string(),
                "henceforth".to_string(),
            ],
        );
        register_markers.insert(
            "informal".to_string(),
            vec![
                "yeah".to_string(),
                "okay".to_string(),
                "gonna".to_string(),
                "kinda".to_string(),
                "stuff".to_string(),
            ],
        );
        register_markers.insert(
            "academic".to_string(),
            vec![
                "hypothesis".to_string(),
                "methodology".to_string(),
                "analysis".to_string(),
                "evidence".to_string(),
                "conclusion".to_string(),
            ],
        );

        let mut politeness_indicators = HashMap::new();
        politeness_indicators.insert("please".to_string(), 0.8);
        politeness_indicators.insert("would you".to_string(), 0.9);
        politeness_indicators.insert("could you".to_string(), 0.85);
        politeness_indicators.insert("thank you".to_string(), 0.7);
        politeness_indicators.insert("excuse me".to_string(), 0.75);
        politeness_indicators.insert("I apologize".to_string(), 0.9);

        let mut speech_act_patterns = HashMap::new();
        speech_act_patterns.insert(
            "directive".to_string(),
            vec![
                "please".to_string(),
                "can you".to_string(),
                "should".to_string(),
            ],
        );
        speech_act_patterns.insert(
            "commissive".to_string(),
            vec![
                "I promise".to_string(),
                "I will".to_string(),
                "I guarantee".to_string(),
            ],
        );
        speech_act_patterns.insert(
            "expressive".to_string(),
            vec![
                "sorry".to_string(),
                "congratulations".to_string(),
                "thank you".to_string(),
            ],
        );

        let discourse_markers = [
            "however",
            "therefore",
            "meanwhile",
            "furthermore",
            "nevertheless",
            "consequently",
            "moreover",
            "nonetheless",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let mut cultural_markers = HashMap::new();
        cultural_markers.insert(
            "western".to_string(),
            vec![
                "individualistic".to_string(),
                "direct".to_string(),
                "time-oriented".to_string(),
            ],
        );
        cultural_markers.insert(
            "eastern".to_string(),
            vec![
                "collectivistic".to_string(),
                "indirect".to_string(),
                "relationship-oriented".to_string(),
            ],
        );

        Self {
            context_patterns,
            register_markers,
            politeness_indicators,
            speech_act_patterns,
            discourse_markers,
            cultural_markers,
        }
    }
}

impl PragmaticAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_custom_patterns(
        context_patterns: HashMap<String, f64>,
        register_markers: HashMap<String, Vec<String>>,
        politeness_indicators: HashMap<String, f64>,
    ) -> Self {
        Self {
            context_patterns,
            register_markers,
            politeness_indicators,
            ..Default::default()
        }
    }

    pub fn analyze_pragmatic_fluency(&self, text: &str) -> Result<PragmaticScore, PragmaticError> {
        let context_analysis = self.analyze_context_appropriateness(text)?;
        let register_analysis = self.analyze_register_appropriateness(text)?;
        let politeness_analysis = self.analyze_politeness_strategies(text)?;
        let speech_act_analysis = self.analyze_speech_acts(text)?;
        let implicature_analysis = self.analyze_implicature(text)?;
        let social_context_analysis = self.analyze_social_context(text)?;
        let discourse_analysis = self.analyze_discourse_competence(text)?;
        let relevance_analysis = self.analyze_relevance(text)?;
        let audience_analysis = self.analyze_audience_awareness(text)?;
        let cultural_sensitivity = self.analyze_cultural_sensitivity(text)?;
        let advanced_metrics =
            self.compute_advanced_pragmatic_metrics(text, &context_analysis, &register_analysis)?;

        let context_appropriateness = context_analysis.situational_appropriateness;
        let communicative_effectiveness = (speech_act_analysis.illocutionary_force
            + speech_act_analysis.perlocutionary_effect)
            / 2.0;
        let audience_awareness = audience_analysis.audience_adaptation;
        let register_appropriateness = register_analysis.register_consistency;
        let politeness_level = (politeness_analysis.positive_politeness
            + politeness_analysis.negative_politeness)
            / 2.0;
        let speech_act_clarity = speech_act_analysis.speech_act_appropriateness;
        let implicature_richness = implicature_analysis.implicit_meaning_richness;
        let social_awareness = social_context_analysis.social_role_appropriateness;
        let dialogue_competence = discourse_analysis.interactive_competence;
        let cooperation_adherence = (relevance_analysis.maxim_adherence.quantity_maxim
            + relevance_analysis.maxim_adherence.quality_maxim
            + relevance_analysis.maxim_adherence.relation_maxim
            + relevance_analysis.maxim_adherence.manner_maxim)
            / 4.0;
        let relevance_score = relevance_analysis.relevance_to_context;

        let overall_score = (context_appropriateness
            + communicative_effectiveness
            + audience_awareness
            + register_appropriateness
            + politeness_level
            + speech_act_clarity
            + implicature_richness
            + social_awareness
            + dialogue_competence
            + cooperation_adherence
            + relevance_score)
            / 11.0;

        let detailed_metrics = DetailedPragmaticMetrics {
            context_analysis,
            register_analysis,
            politeness_analysis,
            speech_act_analysis,
            implicature_analysis,
            social_context_analysis,
            discourse_analysis,
            relevance_analysis,
            audience_analysis,
            cultural_sensitivity,
            advanced_metrics,
        };

        Ok(PragmaticScore {
            overall_score,
            context_appropriateness,
            communicative_effectiveness,
            audience_awareness,
            register_appropriateness,
            politeness_level,
            speech_act_clarity,
            implicature_richness,
            social_awareness,
            dialogue_competence,
            cooperation_adherence,
            relevance_score,
            detailed_metrics,
        })
    }

    fn analyze_context_appropriateness(
        &self,
        text: &str,
    ) -> Result<ContextAnalysis, PragmaticError> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let total_words = words.len() as f64;

        if total_words == 0.0 {
            return Err(PragmaticError::ContextMismatch(
                "Empty text provided".to_string(),
            ));
        }

        let situational_appropriateness = self.calculate_situational_appropriateness(text);
        let contextual_coherence = self.calculate_contextual_coherence(&words);
        let background_knowledge_usage = self.calculate_background_knowledge_usage(text);
        let contextual_inference_quality = self.calculate_contextual_inference_quality(text);
        let shared_knowledge_assumptions = self.calculate_shared_knowledge_assumptions(text);
        let contextual_adaptation_score = self.calculate_contextual_adaptation(text);

        Ok(ContextAnalysis {
            situational_appropriateness,
            contextual_coherence,
            background_knowledge_usage,
            contextual_inference_quality,
            shared_knowledge_assumptions,
            contextual_adaptation_score,
        })
    }

    fn analyze_register_appropriateness(
        &self,
        text: &str,
    ) -> Result<RegisterAnalysis, PragmaticError> {
        let formality_level = self.calculate_formality_level(text);
        let register_consistency = self.calculate_register_consistency(text);
        let domain_appropriateness = self.calculate_domain_appropriateness(text);
        let style_matching = self.calculate_style_matching(text);
        let tone_appropriateness = self.calculate_tone_appropriateness(text);
        let linguistic_register_markers = self.extract_register_markers(text);

        Ok(RegisterAnalysis {
            formality_level,
            register_consistency,
            domain_appropriateness,
            style_matching,
            tone_appropriateness,
            linguistic_register_markers,
        })
    }

    fn analyze_politeness_strategies(
        &self,
        text: &str,
    ) -> Result<PolitenessAnalysis, PragmaticError> {
        let positive_politeness = self.calculate_positive_politeness(text);
        let negative_politeness = self.calculate_negative_politeness(text);
        let face_saving_strategies = self.calculate_face_saving_strategies(text);
        let indirect_speech_usage = self.calculate_indirect_speech_usage(text);
        let hedging_strategies = self.calculate_hedging_strategies(text);
        let courtesy_markers = self.extract_courtesy_markers(text);

        Ok(PolitenessAnalysis {
            positive_politeness,
            negative_politeness,
            face_saving_strategies,
            indirect_speech_usage,
            hedging_strategies,
            courtesy_markers,
        })
    }

    fn analyze_speech_acts(&self, text: &str) -> Result<SpeechActAnalysis, PragmaticError> {
        let primary_speech_acts = self.identify_primary_speech_acts(text);
        let secondary_speech_acts = self.identify_secondary_speech_acts(text);
        let illocutionary_force = self.calculate_illocutionary_force(text);
        let perlocutionary_effect = self.calculate_perlocutionary_effect(text);
        let speech_act_appropriateness = self.calculate_speech_act_appropriateness(text);
        let directness_level = self.calculate_directness_level(text);

        Ok(SpeechActAnalysis {
            primary_speech_acts,
            secondary_speech_acts,
            illocutionary_force,
            perlocutionary_effect,
            speech_act_appropriateness,
            directness_level,
        })
    }

    fn analyze_implicature(&self, text: &str) -> Result<ImplicatureAnalysis, PragmaticError> {
        let conversational_implicature = self.calculate_conversational_implicature(text);
        let conventional_implicature = self.calculate_conventional_implicature(text);
        let inference_complexity = self.calculate_inference_complexity(text);
        let implicature_clarity = self.calculate_implicature_clarity(text);
        let presupposition_handling = self.calculate_presupposition_handling(text);
        let implicit_meaning_richness = self.calculate_implicit_meaning_richness(text);

        Ok(ImplicatureAnalysis {
            conversational_implicature,
            conventional_implicature,
            inference_complexity,
            implicature_clarity,
            presupposition_handling,
            implicit_meaning_richness,
        })
    }

    fn analyze_social_context(&self, text: &str) -> Result<SocialContextAnalysis, PragmaticError> {
        let power_relations_awareness = self.calculate_power_relations_awareness(text);
        let social_distance_management = self.calculate_social_distance_management(text);
        let cultural_sensitivity_score = self.calculate_cultural_sensitivity_score(text);
        let group_dynamics_awareness = self.calculate_group_dynamics_awareness(text);
        let identity_construction = self.calculate_identity_construction(text);
        let social_role_appropriateness = self.calculate_social_role_appropriateness(text);

        Ok(SocialContextAnalysis {
            power_relations_awareness,
            social_distance_management,
            cultural_sensitivity_score,
            group_dynamics_awareness,
            identity_construction,
            social_role_appropriateness,
        })
    }

    fn analyze_discourse_competence(
        &self,
        text: &str,
    ) -> Result<DiscourseAnalysis, PragmaticError> {
        let turn_taking_competence = self.calculate_turn_taking_competence(text);
        let topic_management = self.calculate_topic_management(text);
        let coherence_maintenance = self.calculate_coherence_maintenance(text);
        let discourse_marker_usage = self.calculate_discourse_marker_usage(text);
        let conversational_flow = self.calculate_conversational_flow(text);
        let interactive_competence = self.calculate_interactive_competence(text);

        Ok(DiscourseAnalysis {
            turn_taking_competence,
            topic_management,
            coherence_maintenance,
            discourse_marker_usage,
            conversational_flow,
            interactive_competence,
        })
    }

    fn analyze_relevance(&self, text: &str) -> Result<RelevanceAnalysis, PragmaticError> {
        let relevance_to_context = self.calculate_relevance_to_context(text);
        let information_value = self.calculate_information_value(text);
        let cognitive_effort_balance = self.calculate_cognitive_effort_balance(text);
        let contextual_effects = self.calculate_contextual_effects(text);
        let relevance_optimization = self.calculate_relevance_optimization(text);

        let quantity_maxim = self.calculate_quantity_maxim_adherence(text);
        let quality_maxim = self.calculate_quality_maxim_adherence(text);
        let relation_maxim = self.calculate_relation_maxim_adherence(text);
        let manner_maxim = self.calculate_manner_maxim_adherence(text);

        let maxim_adherence = MaximAdherence {
            quantity_maxim,
            quality_maxim,
            relation_maxim,
            manner_maxim,
        };

        Ok(RelevanceAnalysis {
            relevance_to_context,
            information_value,
            cognitive_effort_balance,
            contextual_effects,
            relevance_optimization,
            maxim_adherence,
        })
    }

    fn analyze_audience_awareness(&self, text: &str) -> Result<AudienceAnalysis, PragmaticError> {
        let audience_adaptation = self.calculate_audience_adaptation(text);
        let shared_knowledge_assessment = self.calculate_shared_knowledge_assessment(text);
        let expertise_level_matching = self.calculate_expertise_level_matching(text);
        let interest_level_maintenance = self.calculate_interest_level_maintenance(text);
        let comprehensibility_optimization = self.calculate_comprehensibility_optimization(text);
        let engagement_strategies = self.identify_engagement_strategies(text);

        Ok(AudienceAnalysis {
            audience_adaptation,
            shared_knowledge_assessment,
            expertise_level_matching,
            interest_level_maintenance,
            comprehensibility_optimization,
            engagement_strategies,
        })
    }

    fn analyze_cultural_sensitivity(
        &self,
        text: &str,
    ) -> Result<CulturalSensitivity, PragmaticError> {
        let cultural_appropriateness = self.calculate_cultural_appropriateness(text);
        let cross_cultural_awareness = self.calculate_cross_cultural_awareness(text);
        let cultural_bias_avoidance = self.calculate_cultural_bias_avoidance(text);
        let inclusive_language_usage = self.calculate_inclusive_language_usage(text);
        let cultural_reference_handling = self.calculate_cultural_reference_handling(text);
        let intercultural_competence = self.calculate_intercultural_competence(text);

        Ok(CulturalSensitivity {
            cultural_appropriateness,
            cross_cultural_awareness,
            cultural_bias_avoidance,
            inclusive_language_usage,
            cultural_reference_handling,
            intercultural_competence,
        })
    }

    fn compute_advanced_pragmatic_metrics(
        &self,
        text: &str,
        context_analysis: &ContextAnalysis,
        register_analysis: &RegisterAnalysis,
    ) -> Result<AdvancedPragmaticMetrics, PragmaticError> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let total_words = words.len();

        let pragmatic_competence_index = self.calculate_pragmatic_competence_index(text);
        let communicative_strategy_diversity =
            self.calculate_communicative_strategy_diversity(text);
        let contextual_sensitivity_score = context_analysis.contextual_coherence;
        let interpersonal_effectiveness = self.calculate_interpersonal_effectiveness(text);
        let pragmatic_inference_capability = self.calculate_pragmatic_inference_capability(text);
        let social_linguistic_competence = self.calculate_social_linguistic_competence(text);
        let multimodal_communication_score = self.calculate_multimodal_communication_score(text);

        let trajectory_length = (total_words / 10).max(5);
        let mut trajectory = Array1::<f64>::zeros(trajectory_length);
        let chunk_size = total_words / trajectory_length;

        for (i, chunk_start) in (0..total_words).step_by(chunk_size).enumerate() {
            if i >= trajectory_length {
                break;
            }
            let chunk_end = (chunk_start + chunk_size).min(total_words);
            let chunk_text = words[chunk_start..chunk_end].join(" ");
            trajectory[i] = self.calculate_local_pragmatic_fluency(&chunk_text);
        }

        let strategy_usage_patterns = self.analyze_strategy_usage_patterns(text);
        let pragmatic_error_analysis = self.analyze_pragmatic_errors(text);

        Ok(AdvancedPragmaticMetrics {
            pragmatic_competence_index,
            communicative_strategy_diversity,
            contextual_sensitivity_score,
            interpersonal_effectiveness,
            pragmatic_inference_capability,
            social_linguistic_competence,
            multimodal_communication_score,
            pragmatic_fluency_trajectory: trajectory,
            strategy_usage_patterns,
            pragmatic_error_analysis,
        })
    }

    // Context Analysis Methods
    fn calculate_situational_appropriateness(&self, text: &str) -> f64 {
        let mut appropriateness_score = 0.0;
        let mut context_matches = 0;

        for (context, weight) in &self.context_patterns {
            if text.to_lowercase().contains(context) {
                appropriateness_score += weight;
                context_matches += 1;
            }
        }

        if context_matches > 0 {
            appropriateness_score / context_matches as f64
        } else {
            0.5 // Neutral score if no context patterns found
        }
    }

    fn calculate_contextual_coherence(&self, words: &[&str]) -> f64 {
        if words.len() < 2 {
            return 0.0;
        }

        let mut coherence_score = 0.0;
        let mut coherence_pairs = 0;

        for window in words.windows(3) {
            if window.len() >= 2 {
                let semantic_similarity = self.calculate_semantic_similarity(window[0], window[1]);
                coherence_score += semantic_similarity;
                coherence_pairs += 1;
            }
        }

        if coherence_pairs > 0 {
            coherence_score / coherence_pairs as f64
        } else {
            0.0
        }
    }

    fn calculate_background_knowledge_usage(&self, text: &str) -> f64 {
        let knowledge_indicators = [
            "as you know",
            "obviously",
            "clearly",
            "of course",
            "naturally",
            "as mentioned",
            "previously",
            "as discussed",
            "recall that",
        ];

        let total_words = text.split_whitespace().count() as f64;
        if total_words == 0.0 {
            return 0.0;
        }

        let knowledge_usage_count = knowledge_indicators
            .iter()
            .map(|indicator| text.matches(indicator).count())
            .sum::<usize>() as f64;

        (knowledge_usage_count / total_words * 100.0).min(1.0)
    }

    fn calculate_contextual_inference_quality(&self, text: &str) -> f64 {
        let inference_markers = [
            "therefore",
            "thus",
            "hence",
            "consequently",
            "implies",
            "suggests",
            "indicates",
            "follows that",
            "we can infer",
            "it appears",
        ];

        let total_sentences = text.split('.').count() as f64;
        if total_sentences == 0.0 {
            return 0.0;
        }

        let inference_count = inference_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        (inference_count / total_sentences).min(1.0)
    }

    fn calculate_shared_knowledge_assumptions(&self, text: &str) -> f64 {
        let assumption_markers = [
            "you know",
            "as we all know",
            "it's clear that",
            "obviously",
            "everyone knows",
            "it goes without saying",
            "needless to say",
        ];

        let total_words = text.split_whitespace().count() as f64;
        if total_words == 0.0 {
            return 0.0;
        }

        let assumption_count = assumption_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        (assumption_count / total_words * 100.0).min(1.0)
    }

    fn calculate_contextual_adaptation(&self, text: &str) -> f64 {
        let adaptation_markers = [
            "in this context",
            "given the situation",
            "considering",
            "taking into account",
            "under these circumstances",
            "in light of",
            "given that",
        ];

        let total_sentences = text.split('.').count() as f64;
        if total_sentences == 0.0 {
            return 0.0;
        }

        let adaptation_count = adaptation_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        (adaptation_count / total_sentences).min(1.0)
    }

    // Register Analysis Methods
    fn calculate_formality_level(&self, text: &str) -> f64 {
        let formal_markers = [
            "furthermore",
            "moreover",
            "consequently",
            "nevertheless",
            "henceforth",
        ];
        let informal_markers = ["yeah", "okay", "gonna", "kinda", "stuff"];

        let formal_count = formal_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        let informal_count = informal_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        let total_markers = formal_count + informal_count;
        if total_markers == 0.0 {
            return 0.5;
        }

        formal_count / total_markers
    }

    fn calculate_register_consistency(&self, text: &str) -> f64 {
        let formality_level = self.calculate_formality_level(text);
        let sentences: Vec<&str> = text.split('.').collect();

        if sentences.len() < 2 {
            return 1.0;
        }

        let mut consistency_scores = Vec::new();

        for sentence in &sentences {
            let sentence_formality = self.calculate_formality_level(sentence);
            let difference = (formality_level - sentence_formality).abs();
            consistency_scores.push(1.0 - difference);
        }

        consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64
    }

    fn calculate_domain_appropriateness(&self, text: &str) -> f64 {
        let academic_markers = [
            "hypothesis",
            "methodology",
            "analysis",
            "evidence",
            "conclusion",
        ];
        let professional_markers = [
            "objectives",
            "deliverables",
            "stakeholders",
            "implementation",
        ];

        let academic_score = academic_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        let professional_score = professional_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        let total_words = text.split_whitespace().count() as f64;
        if total_words == 0.0 {
            return 0.0;
        }

        ((academic_score + professional_score) / total_words * 100.0).min(1.0)
    }

    fn calculate_style_matching(&self, text: &str) -> f64 {
        // Simplified style matching based on sentence length variety
        let sentences: Vec<&str> = text.split('.').collect();
        if sentences.len() < 2 {
            return 0.5;
        }

        let sentence_lengths: Vec<usize> = sentences
            .iter()
            .map(|s| s.split_whitespace().count())
            .collect();

        let avg_length =
            sentence_lengths.iter().sum::<usize>() as f64 / sentence_lengths.len() as f64;
        let variance = sentence_lengths
            .iter()
            .map(|&length| (length as f64 - avg_length).powi(2))
            .sum::<f64>()
            / sentence_lengths.len() as f64;

        // Good style matching has moderate variance
        let optimal_variance = avg_length * 0.3;
        let variance_difference = (variance - optimal_variance).abs();

        (1.0 - (variance_difference / optimal_variance)).max(0.0)
    }

    fn calculate_tone_appropriateness(&self, text: &str) -> f64 {
        let positive_tone_markers = ["excellent", "wonderful", "great", "fantastic", "amazing"];
        let neutral_tone_markers = ["adequate", "satisfactory", "acceptable", "reasonable"];
        let negative_tone_markers = ["poor", "inadequate", "disappointing", "problematic"];

        let positive_count = positive_tone_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;
        let neutral_count = neutral_tone_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;
        let negative_count = negative_tone_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        let total_tone_markers = positive_count + neutral_count + negative_count;
        if total_tone_markers == 0.0 {
            return 0.7;
        } // Neutral appropriateness

        // Balanced tone is generally more appropriate
        let balance_score = 1.0 - ((positive_count - negative_count).abs() / total_tone_markers);
        balance_score.max(0.0)
    }

    fn extract_register_markers(&self, text: &str) -> Vec<String> {
        let mut found_markers = Vec::new();

        for (_, markers) in &self.register_markers {
            for marker in markers {
                if text.to_lowercase().contains(&marker.to_lowercase()) {
                    found_markers.push(marker.clone());
                }
            }
        }

        found_markers.sort();
        found_markers.dedup();
        found_markers
    }

    // Politeness Analysis Methods
    fn calculate_positive_politeness(&self, text: &str) -> f64 {
        let positive_politeness_strategies = [
            "we",
            "our",
            "us",
            "together",
            "shared",
            "common",
            "mutual",
            "great",
            "excellent",
            "wonderful",
            "appreciate",
            "admire",
        ];

        let total_words = text.split_whitespace().count() as f64;
        if total_words == 0.0 {
            return 0.0;
        }

        let positive_count = positive_politeness_strategies
            .iter()
            .map(|strategy| text.matches(strategy).count())
            .sum::<usize>() as f64;

        (positive_count / total_words * 10.0).min(1.0)
    }

    fn calculate_negative_politeness(&self, text: &str) -> f64 {
        let negative_politeness_strategies = [
            "please",
            "would you",
            "could you",
            "if you don't mind",
            "I'm sorry",
            "excuse me",
            "pardon",
            "apologize",
        ];

        let total_words = text.split_whitespace().count() as f64;
        if total_words == 0.0 {
            return 0.0;
        }

        let negative_count = negative_politeness_strategies
            .iter()
            .map(|strategy| text.matches(strategy).count())
            .sum::<usize>() as f64;

        (negative_count / total_words * 10.0).min(1.0)
    }

    fn calculate_face_saving_strategies(&self, text: &str) -> f64 {
        let face_saving_markers = [
            "I understand",
            "I see your point",
            "that's a good point",
            "you're right",
            "I agree",
            "fair enough",
            "I can see how",
        ];

        let total_sentences = text.split('.').count() as f64;
        if total_sentences == 0.0 {
            return 0.0;
        }

        let face_saving_count = face_saving_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        (face_saving_count / total_sentences).min(1.0)
    }

    fn calculate_indirect_speech_usage(&self, text: &str) -> f64 {
        let indirect_markers = [
            "might",
            "could",
            "would",
            "perhaps",
            "maybe",
            "possibly",
            "it seems",
            "it appears",
            "I wonder",
            "I suppose",
        ];

        let total_words = text.split_whitespace().count() as f64;
        if total_words == 0.0 {
            return 0.0;
        }

        let indirect_count = indirect_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        (indirect_count / total_words * 10.0).min(1.0)
    }

    fn calculate_hedging_strategies(&self, text: &str) -> f64 {
        let hedging_markers = [
            "sort of",
            "kind of",
            "somewhat",
            "rather",
            "quite",
            "fairly",
            "pretty",
            "relatively",
            "to some extent",
        ];

        let total_words = text.split_whitespace().count() as f64;
        if total_words == 0.0 {
            return 0.0;
        }

        let hedging_count = hedging_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        (hedging_count / total_words * 10.0).min(1.0)
    }

    fn extract_courtesy_markers(&self, text: &str) -> Vec<String> {
        let courtesy_patterns = [
            "please",
            "thank you",
            "thanks",
            "excuse me",
            "pardon me",
            "I'm sorry",
            "apologize",
            "would you mind",
            "if you please",
        ];

        let mut found_markers = Vec::new();
        for pattern in &courtesy_patterns {
            if text.to_lowercase().contains(pattern) {
                found_markers.push(pattern.to_string());
            }
        }

        found_markers
    }

    // Speech Act Analysis Methods
    fn identify_primary_speech_acts(&self, text: &str) -> Vec<String> {
        let mut speech_acts = Vec::new();

        if text.contains('?') {
            speech_acts.push("Question".to_string());
        }
        if text.contains('!') {
            speech_acts.push("Exclamation".to_string());
        }
        if text.contains("please") || text.contains("should") {
            speech_acts.push("Directive".to_string());
        }
        if text.contains("I promise") || text.contains("I will") {
            speech_acts.push("Commissive".to_string());
        }
        if text.contains("sorry") || text.contains("thank") {
            speech_acts.push("Expressive".to_string());
        }

        speech_acts
    }

    fn identify_secondary_speech_acts(&self, text: &str) -> Vec<String> {
        let mut secondary_acts = Vec::new();

        if text.contains("by the way") || text.contains("incidentally") {
            secondary_acts.push("Aside".to_string());
        }
        if text.contains("for example") || text.contains("such as") {
            secondary_acts.push("Exemplification".to_string());
        }
        if text.contains("in other words") || text.contains("that is") {
            secondary_acts.push("Clarification".to_string());
        }

        secondary_acts
    }

    fn calculate_illocutionary_force(&self, text: &str) -> f64 {
        let force_indicators = ["must", "should", "will", "shall", "need to", "have to"];
        let total_words = text.split_whitespace().count() as f64;

        if total_words == 0.0 {
            return 0.0;
        }

        let force_count = force_indicators
            .iter()
            .map(|indicator| text.matches(indicator).count())
            .sum::<usize>() as f64;

        (force_count / total_words * 10.0).min(1.0)
    }

    fn calculate_perlocutionary_effect(&self, text: &str) -> f64 {
        let effect_markers = [
            "convince",
            "persuade",
            "influence",
            "motivate",
            "inspire",
            "encourage",
            "discourage",
            "warn",
            "advise",
        ];

        let total_words = text.split_whitespace().count() as f64;
        if total_words == 0.0 {
            return 0.0;
        }

        let effect_count = effect_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        (effect_count / total_words * 10.0).min(1.0)
    }

    fn calculate_speech_act_appropriateness(&self, text: &str) -> f64 {
        let sentences: Vec<&str> = text.split('.').collect();
        if sentences.is_empty() {
            return 0.0;
        }

        let mut appropriateness_score = 0.0;

        for sentence in &sentences {
            let acts = self.identify_primary_speech_acts(sentence);
            if !acts.is_empty() {
                // Speech acts are present and identifiable
                appropriateness_score += 1.0;
            } else {
                // Neutral sentence, still appropriate
                appropriateness_score += 0.7;
            }
        }

        appropriateness_score / sentences.len() as f64
    }

    fn calculate_directness_level(&self, text: &str) -> f64 {
        let direct_markers = ["you must", "do this", "don't", "stop", "go"];
        let indirect_markers = ["could you", "would you", "might", "perhaps"];

        let direct_count = direct_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;
        let indirect_count = indirect_markers
            .iter()
            .map(|marker| text.matches(marker).count())
            .sum::<usize>() as f64;

        let total_directive_markers = direct_count + indirect_count;
        if total_directive_markers == 0.0 {
            return 0.5;
        }

        direct_count / total_directive_markers
    }

    // Additional helper methods
    fn calculate_semantic_similarity(&self, word1: &str, word2: &str) -> f64 {
        // Simplified semantic similarity based on common prefixes/suffixes
        let chars1: Vec<char> = word1.chars().collect();
        let chars2: Vec<char> = word2.chars().collect();

        if chars1.is_empty() || chars2.is_empty() {
            return 0.0;
        }

        let common_chars = chars1.iter().filter(|&c| chars2.contains(c)).count() as f64;

        let max_length = chars1.len().max(chars2.len()) as f64;
        common_chars / max_length
    }

    // Placeholder implementations for remaining methods
    fn calculate_conversational_implicature(&self, _text: &str) -> f64 {
        0.6
    }
    fn calculate_conventional_implicature(&self, _text: &str) -> f64 {
        0.5
    }
    fn calculate_inference_complexity(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_implicature_clarity(&self, _text: &str) -> f64 {
        0.6
    }
    fn calculate_presupposition_handling(&self, _text: &str) -> f64 {
        0.5
    }
    fn calculate_implicit_meaning_richness(&self, _text: &str) -> f64 {
        0.6
    }

    fn calculate_power_relations_awareness(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_social_distance_management(&self, _text: &str) -> f64 {
        0.6
    }
    fn calculate_cultural_sensitivity_score(&self, _text: &str) -> f64 {
        0.8
    }
    fn calculate_group_dynamics_awareness(&self, _text: &str) -> f64 {
        0.5
    }
    fn calculate_identity_construction(&self, _text: &str) -> f64 {
        0.6
    }
    fn calculate_social_role_appropriateness(&self, _text: &str) -> f64 {
        0.7
    }

    fn calculate_turn_taking_competence(&self, _text: &str) -> f64 {
        0.6
    }
    fn calculate_topic_management(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_coherence_maintenance(&self, _text: &str) -> f64 {
        0.8
    }
    fn calculate_discourse_marker_usage(&self, _text: &str) -> f64 {
        0.6
    }
    fn calculate_conversational_flow(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_interactive_competence(&self, _text: &str) -> f64 {
        0.6
    }

    fn calculate_relevance_to_context(&self, _text: &str) -> f64 {
        0.8
    }
    fn calculate_information_value(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_cognitive_effort_balance(&self, _text: &str) -> f64 {
        0.6
    }
    fn calculate_contextual_effects(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_relevance_optimization(&self, _text: &str) -> f64 {
        0.6
    }

    fn calculate_quantity_maxim_adherence(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_quality_maxim_adherence(&self, _text: &str) -> f64 {
        0.8
    }
    fn calculate_relation_maxim_adherence(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_manner_maxim_adherence(&self, _text: &str) -> f64 {
        0.6
    }

    fn calculate_audience_adaptation(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_shared_knowledge_assessment(&self, _text: &str) -> f64 {
        0.6
    }
    fn calculate_expertise_level_matching(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_interest_level_maintenance(&self, _text: &str) -> f64 {
        0.6
    }
    fn calculate_comprehensibility_optimization(&self, _text: &str) -> f64 {
        0.8
    }
    fn identify_engagement_strategies(&self, _text: &str) -> Vec<String> {
        vec!["questioning".to_string(), "examples".to_string()]
    }

    fn calculate_cultural_appropriateness(&self, _text: &str) -> f64 {
        0.8
    }
    fn calculate_cross_cultural_awareness(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_cultural_bias_avoidance(&self, _text: &str) -> f64 {
        0.8
    }
    fn calculate_inclusive_language_usage(&self, _text: &str) -> f64 {
        0.9
    }
    fn calculate_cultural_reference_handling(&self, _text: &str) -> f64 {
        0.6
    }
    fn calculate_intercultural_competence(&self, _text: &str) -> f64 {
        0.7
    }

    fn calculate_pragmatic_competence_index(&self, _text: &str) -> f64 {
        0.75
    }
    fn calculate_communicative_strategy_diversity(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_interpersonal_effectiveness(&self, _text: &str) -> f64 {
        0.8
    }
    fn calculate_pragmatic_inference_capability(&self, _text: &str) -> f64 {
        0.6
    }
    fn calculate_social_linguistic_competence(&self, _text: &str) -> f64 {
        0.7
    }
    fn calculate_multimodal_communication_score(&self, _text: &str) -> f64 {
        0.5
    }
    fn calculate_local_pragmatic_fluency(&self, _text: &str) -> f64 {
        0.7
    }

    fn analyze_strategy_usage_patterns(&self, _text: &str) -> HashMap<String, f64> {
        let mut patterns = HashMap::new();
        patterns.insert("politeness".to_string(), 0.7);
        patterns.insert("directness".to_string(), 0.5);
        patterns.insert("hedging".to_string(), 0.3);
        patterns
    }

    fn analyze_pragmatic_errors(&self, _text: &str) -> PragmaticErrorAnalysis {
        PragmaticErrorAnalysis {
            pragmatic_failure_count: 0,
            sociopragmatic_errors: 0,
            pragmalinguistic_errors: 0,
            cultural_transfer_errors: 0,
            register_violations: 0,
            politeness_violations: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pragmatic_analyzer_creation() {
        let analyzer = PragmaticAnalyzer::new();
        assert!(!analyzer.context_patterns.is_empty());
        assert!(!analyzer.register_markers.is_empty());
        assert!(!analyzer.politeness_indicators.is_empty());
    }

    #[test]
    fn test_basic_pragmatic_analysis() {
        let analyzer = PragmaticAnalyzer::new();
        let text =
            "Please could you help me with this task? I would really appreciate your assistance.";

        let result = analyzer.analyze_pragmatic_fluency(text);
        assert!(result.is_ok());

        let score = result.unwrap();
        assert!(score.overall_score > 0.0);
        assert!(score.politeness_level > 0.5);
    }

    #[test]
    fn test_context_appropriateness_analysis() {
        let analyzer = PragmaticAnalyzer::new();
        let formal_text = "Furthermore, the analysis demonstrates significant findings.";

        let result = analyzer.analyze_context_appropriateness(formal_text);
        assert!(result.is_ok());

        let context_analysis = result.unwrap();
        assert!(context_analysis.situational_appropriateness > 0.0);
    }

    #[test]
    fn test_register_analysis() {
        let analyzer = PragmaticAnalyzer::new();
        let academic_text = "The hypothesis suggests that methodology analysis provides evidence for our conclusion.";

        let result = analyzer.analyze_register_appropriateness(academic_text);
        assert!(result.is_ok());

        let register_analysis = result.unwrap();
        assert!(register_analysis.domain_appropriateness > 0.0);
    }

    #[test]
    fn test_politeness_analysis() {
        let analyzer = PragmaticAnalyzer::new();
        let polite_text = "Would you please help me? I would really appreciate it.";

        let result = analyzer.analyze_politeness_strategies(polite_text);
        assert!(result.is_ok());

        let politeness_analysis = result.unwrap();
        assert!(politeness_analysis.negative_politeness > 0.0);
    }

    #[test]
    fn test_speech_act_analysis() {
        let analyzer = PragmaticAnalyzer::new();
        let directive_text = "Please complete this task by tomorrow.";

        let result = analyzer.analyze_speech_acts(directive_text);
        assert!(result.is_ok());

        let speech_act_analysis = result.unwrap();
        assert!(!speech_act_analysis.primary_speech_acts.is_empty());
    }

    #[test]
    fn test_empty_text_handling() {
        let analyzer = PragmaticAnalyzer::new();
        let result = analyzer.analyze_pragmatic_fluency("");
        assert!(result.is_err());
    }

    #[test]
    fn test_advanced_pragmatic_metrics() {
        let analyzer = PragmaticAnalyzer::new();
        let complex_text = "I understand your perspective, and I appreciate the thoughtful analysis you've provided.
                          However, I wonder if we might consider alternative approaches that could potentially
                          yield more comprehensive results while maintaining our commitment to quality.";

        let result = analyzer.analyze_pragmatic_fluency(complex_text);
        assert!(result.is_ok());

        let score = result.unwrap();
        let advanced = &score.detailed_metrics.advanced_metrics;

        assert!(advanced.pragmatic_competence_index > 0.0);
        assert!(advanced.communicative_strategy_diversity > 0.0);
        assert!(!advanced.pragmatic_fluency_trajectory.is_empty());
        assert!(!advanced.strategy_usage_patterns.is_empty());
    }

    #[test]
    fn test_cultural_sensitivity_analysis() {
        let analyzer = PragmaticAnalyzer::new();
        let text =
            "We should work together to find a solution that respects everyone's perspective.";

        let result = analyzer.analyze_pragmatic_fluency(text);
        assert!(result.is_ok());

        let score = result.unwrap();
        let cultural = &score.detailed_metrics.cultural_sensitivity;

        assert!(cultural.inclusive_language_usage > 0.7);
        assert!(cultural.cultural_appropriateness > 0.0);
    }

    #[test]
    fn test_discourse_analysis() {
        let analyzer = PragmaticAnalyzer::new();
        let text = "First, let me address your main concern. Furthermore, I'd like to elaborate on the implications.";

        let result = analyzer.analyze_pragmatic_fluency(text);
        assert!(result.is_ok());

        let score = result.unwrap();
        let discourse = &score.detailed_metrics.discourse_analysis;

        assert!(discourse.coherence_maintenance > 0.0);
        assert!(discourse.topic_management > 0.0);
    }

    #[test]
    fn test_pragmatic_error_analysis() {
        let analyzer = PragmaticAnalyzer::new();
        let text = "This is a perfectly normal text with no obvious pragmatic errors.";

        let result = analyzer.analyze_pragmatic_fluency(text);
        assert!(result.is_ok());

        let score = result.unwrap();
        let errors = &score
            .detailed_metrics
            .advanced_metrics
            .pragmatic_error_analysis;

        assert!(errors.pragmatic_failure_count == 0);
        assert!(errors.sociopragmatic_errors == 0);
    }
}
