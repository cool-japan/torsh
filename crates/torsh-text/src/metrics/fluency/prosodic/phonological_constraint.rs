//! Extracted phonological rule, constraint, and pattern sub-analyzers.
//!
//! This sibling module is `#[path]`-included by `phonological.rs` and re-exported
//! verbatim so all public types remain in the original module namespace.

use super::*;

// ──────────────────────────────────────────────────────────────────────────────
// PhonologicalRuleAnalyzer
// ──────────────────────────────────────────────────────────────────────────────

impl PhonologicalRuleAnalyzer {
    pub(super) fn new(config: &PhonologicalAnalysisConfig) -> Self {
        Self {
            active_rules: Self::create_default_rules(),
            rule_contexts: Self::create_rule_contexts(),
            application_frequencies: HashMap::new(),
            rule_interactions: HashMap::new(),
        }
    }

    pub(super) fn analyze(
        &mut self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<PhonologicalRuleMetrics, PhonologicalAnalysisError> {
        let applied_rules = self.identify_applied_rules(text, phonetic_transcription)?;
        let rule_density = applied_rules.len() as f64 / text.len() as f64;
        let rule_diversity = self.calculate_rule_diversity(&applied_rules);
        let rule_interactions = self.analyze_rule_interactions(&applied_rules);

        Ok(PhonologicalRuleMetrics {
            total_rules_applied: applied_rules.len(),
            rule_density,
            rule_diversity,
            applied_rules: applied_rules.clone(),
            rule_interactions,
            assimilation_frequency: self.count_assimilations(&applied_rules),
            deletion_frequency: self.count_deletions(&applied_rules),
            insertion_frequency: self.count_insertions(&applied_rules),
            metathesis_frequency: self.count_metatheses(&applied_rules),
        })
    }

    fn identify_applied_rules(
        &self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<Vec<PhonologicalRule>, PhonologicalAnalysisError> {
        let mut applied_rules = Vec::new();

        // Compare orthographic and phonetic forms to identify rule applications
        let words: Vec<&str> = text.split_whitespace().collect();
        let phonetic_words: Vec<&str> = phonetic_transcription.split(' ').collect();

        for (word, phonetic) in words.iter().zip(phonetic_words.iter()) {
            for rule in &self.active_rules {
                if self.rule_applies(word, phonetic, rule) {
                    applied_rules.push(rule.clone());
                }
            }
        }

        Ok(applied_rules)
    }

    fn rule_applies(&self, orthographic: &str, phonetic: &str, rule: &PhonologicalRule) -> bool {
        // Simplified rule application detection
        match rule.name.as_str() {
            "Final Devoicing" => {
                orthographic.ends_with('d') && phonetic.ends_with('t')
                    || orthographic.ends_with('g') && phonetic.ends_with('k')
                    || orthographic.ends_with('b') && phonetic.ends_with('p')
            }
            "Vowel Reduction" => {
                phonetic.contains('ə') && !orthographic.to_lowercase().contains('ə')
            }
            "Consonant Cluster Simplification" => {
                orthographic.len() > phonetic.split_whitespace().count()
            }
            _ => false,
        }
    }

    fn calculate_rule_diversity(&self, applied_rules: &[PhonologicalRule]) -> f64 {
        let unique_rules: HashSet<String> =
            applied_rules.iter().map(|rule| rule.name.clone()).collect();

        unique_rules.len() as f64 / applied_rules.len().max(1) as f64
    }

    fn analyze_rule_interactions(&self, applied_rules: &[PhonologicalRule]) -> Vec<String> {
        let mut interactions = Vec::new();

        for window in applied_rules.windows(2) {
            if let [rule1, rule2] = window {
                let interaction_key = format!("{}-{}", rule1.name, rule2.name);
                if self
                    .rule_interactions
                    .contains_key(&(rule1.name.clone(), rule2.name.clone()))
                {
                    interactions.push(interaction_key);
                }
            }
        }

        interactions
    }

    fn count_assimilations(&self, applied_rules: &[PhonologicalRule]) -> usize {
        applied_rules
            .iter()
            .filter(|rule| rule.rule_type == "assimilation")
            .count()
    }

    fn count_deletions(&self, applied_rules: &[PhonologicalRule]) -> usize {
        applied_rules
            .iter()
            .filter(|rule| rule.rule_type == "deletion")
            .count()
    }

    fn count_insertions(&self, applied_rules: &[PhonologicalRule]) -> usize {
        applied_rules
            .iter()
            .filter(|rule| rule.rule_type == "insertion")
            .count()
    }

    fn count_metatheses(&self, applied_rules: &[PhonologicalRule]) -> usize {
        applied_rules
            .iter()
            .filter(|rule| rule.rule_type == "metathesis")
            .count()
    }

    fn create_default_rules() -> Vec<PhonologicalRule> {
        vec![
            PhonologicalRule {
                name: "Final Devoicing".to_string(),
                rule_type: "assimilation".to_string(),
                context: "word-final".to_string(),
                structural_change: "[+voice] → [-voice] / _#".to_string(),
                frequency: 0.7,
                language_universal: false,
            },
            PhonologicalRule {
                name: "Vowel Reduction".to_string(),
                rule_type: "reduction".to_string(),
                context: "unstressed".to_string(),
                structural_change: "V → ə / unstressed".to_string(),
                frequency: 0.8,
                language_universal: false,
            },
            PhonologicalRule {
                name: "Consonant Cluster Simplification".to_string(),
                rule_type: "deletion".to_string(),
                context: "complex onset/coda".to_string(),
                structural_change: "CCC → CC".to_string(),
                frequency: 0.4,
                language_universal: true,
            },
        ]
    }

    fn create_rule_contexts() -> HashMap<String, Vec<String>> {
        let mut contexts = HashMap::new();
        contexts.insert(
            "Final Devoicing".to_string(),
            vec!["word-final".to_string(), "syllable-final".to_string()],
        );
        contexts.insert(
            "Vowel Reduction".to_string(),
            vec!["unstressed".to_string(), "function word".to_string()],
        );
        contexts.insert(
            "Assimilation".to_string(),
            vec!["adjacent".to_string(), "within syllable".to_string()],
        );
        contexts
    }

    pub(super) fn update_config(&mut self, config: &PhonologicalAnalysisConfig) {
        if config.include_morphophonological_rules {
            self.active_rules
                .extend(Self::create_morphophonological_rules());
        }
    }

    fn create_morphophonological_rules() -> Vec<PhonologicalRule> {
        vec![PhonologicalRule {
            name: "Past Tense Allomorphy".to_string(),
            rule_type: "morphophonological".to_string(),
            context: "past tense morpheme".to_string(),
            structural_change: "/t/ → /ɪd/ / [+alveolar stop]_".to_string(),
            frequency: 0.9,
            language_universal: false,
        }]
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PhonotacticConstraintAnalyzer
// ──────────────────────────────────────────────────────────────────────────────

impl PhonotacticConstraintAnalyzer {
    pub(super) fn new(_config: &PhonologicalAnalysisConfig) -> Self {
        Self {
            universal_constraints: Self::create_universal_constraints(),
            language_constraints: Self::create_language_constraints(),
            constraint_weights: Self::create_constraint_weights(),
            violation_penalties: Self::create_violation_penalties(),
        }
    }

    pub(super) fn analyze(
        &self,
        phonetic_transcription: &str,
    ) -> Result<PhonotacticConstraintMetrics, PhonologicalAnalysisError> {
        let phonemes: Vec<&str> = phonetic_transcription.split_whitespace().collect();

        let violations = self.evaluate_constraints(&phonemes);
        let violation_density = violations.len() as f64 / phonemes.len() as f64;
        let constraint_adherence = 1.0 - violation_density;

        let universal_violations = violations
            .iter()
            .filter(|v| v.constraint_universality == ConstraintUniversality::Universal)
            .count();

        let language_violations = violations
            .iter()
            .filter(|v| v.constraint_universality == ConstraintUniversality::LanguageSpecific)
            .count();

        Ok(PhonotacticConstraintMetrics {
            total_constraints_evaluated: self.universal_constraints.len()
                + self.language_constraints.len(),
            violations: violations.clone(),
            violation_density,
            constraint_adherence,
            universal_violations,
            language_violations,
            markedness_violations: self.count_markedness_violations(&violations),
            faithfulness_violations: self.count_faithfulness_violations(&violations),
        })
    }

    fn evaluate_constraints(&self, phonemes: &[&str]) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();

        // Evaluate universal constraints
        for constraint in &self.universal_constraints {
            if let Some(violation) = self.check_constraint(phonemes, constraint) {
                violations.push(violation);
            }
        }

        // Evaluate language-specific constraints
        for constraint in &self.language_constraints {
            if let Some(violation) = self.check_constraint(phonemes, constraint) {
                violations.push(violation);
            }
        }

        violations
    }

    fn check_constraint(
        &self,
        phonemes: &[&str],
        constraint: &PhonotacticConstraint,
    ) -> Option<ConstraintViolation> {
        match constraint.constraint_type {
            ConstraintType::Onset => self.check_onset_constraint(phonemes, constraint),
            ConstraintType::Coda => self.check_coda_constraint(phonemes, constraint),
            ConstraintType::Sequence => self.check_sequence_constraint(phonemes, constraint),
            ConstraintType::Sonority => self.check_sonority_constraint(phonemes, constraint),
            _ => None,
        }
    }

    fn check_onset_constraint(
        &self,
        phonemes: &[&str],
        constraint: &PhonotacticConstraint,
    ) -> Option<ConstraintViolation> {
        // Simplified onset constraint checking
        if constraint.name == "No Complex Onsets" {
            for window in phonemes.windows(3) {
                if window.iter().take(2).all(|&p| self.is_consonant(p)) && self.is_vowel(window[2])
                {
                    return Some(ConstraintViolation {
                        constraint_name: constraint.name.clone(),
                        constraint_type: constraint.constraint_type,
                        constraint_universality: constraint.universality,
                        position: 0, // Simplified position
                        severity: constraint.violation_weight,
                        description: format!("Complex onset found: {}-{}", window[0], window[1]),
                    });
                }
            }
        }
        None
    }

    fn check_coda_constraint(
        &self,
        phonemes: &[&str],
        constraint: &PhonotacticConstraint,
    ) -> Option<ConstraintViolation> {
        // Simplified coda constraint checking
        if constraint.name == "No Coda" {
            for window in phonemes.windows(2) {
                if self.is_consonant(window[0])
                    && (window.len() == 1 || self.is_consonant(window[1]))
                {
                    // Check if this consonant is in coda position
                    return Some(ConstraintViolation {
                        constraint_name: constraint.name.clone(),
                        constraint_type: constraint.constraint_type,
                        constraint_universality: constraint.universality,
                        position: 0,
                        severity: constraint.violation_weight,
                        description: format!("Coda consonant found: {}", window[0]),
                    });
                }
            }
        }
        None
    }

    fn check_sequence_constraint(
        &self,
        phonemes: &[&str],
        constraint: &PhonotacticConstraint,
    ) -> Option<ConstraintViolation> {
        // Check for prohibited sequences
        if constraint.pattern == "**" {
            for window in phonemes.windows(2) {
                if window[0] == window[1] {
                    return Some(ConstraintViolation {
                        constraint_name: constraint.name.clone(),
                        constraint_type: constraint.constraint_type,
                        constraint_universality: constraint.universality,
                        position: 0,
                        severity: constraint.violation_weight,
                        description: format!("Geminate found: {}", window[0]),
                    });
                }
            }
        }
        None
    }

    fn check_sonority_constraint(
        &self,
        phonemes: &[&str],
        constraint: &PhonotacticConstraint,
    ) -> Option<ConstraintViolation> {
        // Check sonority sequencing principle
        if constraint.name == "Sonority Sequencing" {
            for window in phonemes.windows(2) {
                let sonority1 = self.get_sonority_level(window[0]);
                let sonority2 = self.get_sonority_level(window[1]);

                if self.is_consonant(window[0])
                    && self.is_consonant(window[1])
                    && sonority1 > sonority2
                {
                    return Some(ConstraintViolation {
                        constraint_name: constraint.name.clone(),
                        constraint_type: constraint.constraint_type,
                        constraint_universality: constraint.universality,
                        position: 0,
                        severity: constraint.violation_weight,
                        description: format!("Sonority violation: {} > {}", window[0], window[1]),
                    });
                }
            }
        }
        None
    }

    fn count_markedness_violations(&self, violations: &[ConstraintViolation]) -> usize {
        violations
            .iter()
            .filter(|v| v.constraint_type == ConstraintType::Markedness)
            .count()
    }

    fn count_faithfulness_violations(&self, violations: &[ConstraintViolation]) -> usize {
        violations
            .iter()
            .filter(|v| v.constraint_type == ConstraintType::Faithfulness)
            .count()
    }

    fn is_consonant(&self, phoneme: &str) -> bool {
        !self.is_vowel(phoneme)
    }

    fn is_vowel(&self, phoneme: &str) -> bool {
        matches!(
            phoneme.to_lowercase().as_str(),
            "a" | "e"
                | "i"
                | "o"
                | "u"
                | "æ"
                | "ɛ"
                | "ɪ"
                | "ɔ"
                | "ʊ"
                | "ə"
                | "ɑ"
                | "ɒ"
                | "ʌ"
                | "ɜ"
                | "ɨ"
                | "ɵ"
                | "ɐ"
                | "ɶ"
                | "ø"
                | "y"
        )
    }

    fn get_sonority_level(&self, phoneme: &str) -> u8 {
        if self.is_vowel(phoneme) {
            5
        } else if matches!(
            phoneme.to_lowercase().as_str(),
            "l" | "r" | "ɫ" | "ɾ" | "ɽ" | "ʀ" | "ʁ"
        ) {
            4
        } else if matches!(
            phoneme.to_lowercase().as_str(),
            "m" | "n" | "ŋ" | "ɲ" | "ɳ" | "ɴ"
        ) {
            3
        } else if matches!(
            phoneme.to_lowercase().as_str(),
            "f" | "v" | "θ" | "ð" | "s" | "z" | "ʃ" | "ʒ" | "h" | "x" | "ɣ"
        ) {
            2
        } else if matches!(
            phoneme.to_lowercase().as_str(),
            "p" | "b" | "t" | "d" | "k" | "g" | "q" | "ɢ" | "ʔ"
        ) {
            1
        } else {
            0
        }
    }

    fn create_universal_constraints() -> Vec<PhonotacticConstraint> {
        vec![
            PhonotacticConstraint {
                name: "Sonority Sequencing".to_string(),
                constraint_type: ConstraintType::Sonority,
                pattern: "sonority rise in onset, fall in coda".to_string(),
                context: "syllable".to_string(),
                violation_weight: 1.0,
                universality: ConstraintUniversality::Universal,
            },
            PhonotacticConstraint {
                name: "No Complex Onsets".to_string(),
                constraint_type: ConstraintType::Onset,
                pattern: "*CC".to_string(),
                context: "syllable-initial".to_string(),
                violation_weight: 0.8,
                universality: ConstraintUniversality::Typological,
            },
        ]
    }

    fn create_language_constraints() -> Vec<PhonotacticConstraint> {
        vec![PhonotacticConstraint {
            name: "No Coda".to_string(),
            constraint_type: ConstraintType::Coda,
            pattern: "*C]σ".to_string(),
            context: "syllable-final".to_string(),
            violation_weight: 0.6,
            universality: ConstraintUniversality::LanguageSpecific,
        }]
    }

    fn create_constraint_weights() -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        weights.insert("Sonority Sequencing".to_string(), 1.0);
        weights.insert("No Complex Onsets".to_string(), 0.8);
        weights.insert("No Coda".to_string(), 0.6);
        weights
    }

    fn create_violation_penalties() -> HashMap<String, f64> {
        let mut penalties = HashMap::new();
        penalties.insert("Sonority Sequencing".to_string(), 2.0);
        penalties.insert("No Complex Onsets".to_string(), 1.5);
        penalties.insert("No Coda".to_string(), 1.0);
        penalties
    }

    pub(super) fn update_config(&mut self, config: &PhonologicalAnalysisConfig) {
        if config.strict_constraint_evaluation {
            for constraint in &mut self.universal_constraints {
                constraint.violation_weight *= 1.2;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PhonologicalPatternMatcher
// ──────────────────────────────────────────────────────────────────────────────

impl PhonologicalPatternMatcher {
    pub(super) fn new(_config: &PhonologicalAnalysisConfig) -> Self {
        Self {
            pattern_database: Self::create_pattern_database(),
            matching_thresholds: Self::create_matching_thresholds(),
            pattern_frequencies: Self::create_pattern_frequencies(),
            context_weights: Self::create_context_weights(),
        }
    }

    pub(super) fn detect_patterns(
        &self,
        phonetic_transcription: &str,
    ) -> Result<Vec<PhonologicalPattern>, PhonologicalAnalysisError> {
        let mut detected_patterns = Vec::new();
        let phonemes: Vec<&str> = phonetic_transcription.split_whitespace().collect();

        for pattern in &self.pattern_database {
            if self.matches_pattern(&phonemes, pattern) {
                detected_patterns.push(pattern.clone());
            }
        }

        Ok(detected_patterns)
    }

    fn matches_pattern(&self, phonemes: &[&str], pattern: &PhonologicalPattern) -> bool {
        // Simplified pattern matching - in practice would be more sophisticated
        match pattern.name.as_str() {
            "CV Syllable" => phonemes
                .windows(2)
                .any(|window| self.is_consonant(window[0]) && self.is_vowel(window[1])),
            "Consonant Cluster" => phonemes
                .windows(2)
                .any(|window| self.is_consonant(window[0]) && self.is_consonant(window[1])),
            "Vowel Hiatus" => phonemes
                .windows(2)
                .any(|window| self.is_vowel(window[0]) && self.is_vowel(window[1])),
            _ => false,
        }
    }

    fn is_consonant(&self, phoneme: &str) -> bool {
        !self.is_vowel(phoneme)
    }

    fn is_vowel(&self, phoneme: &str) -> bool {
        matches!(
            phoneme.to_lowercase().as_str(),
            "a" | "e"
                | "i"
                | "o"
                | "u"
                | "æ"
                | "ɛ"
                | "ɪ"
                | "ɔ"
                | "ʊ"
                | "ə"
                | "ɑ"
                | "ɒ"
                | "ʌ"
                | "ɜ"
                | "ɨ"
                | "ɵ"
                | "ɐ"
                | "ɶ"
                | "ø"
                | "y"
        )
    }

    fn create_pattern_database() -> Vec<PhonologicalPattern> {
        vec![
            PhonologicalPattern {
                name: "CV Syllable".to_string(),
                pattern_type: "syllable".to_string(),
                structural_description: "consonant + vowel".to_string(),
                frequency: 0.6,
                complexity: 1.0,
                language_universal: true,
            },
            PhonologicalPattern {
                name: "Consonant Cluster".to_string(),
                pattern_type: "sequence".to_string(),
                structural_description: "consecutive consonants".to_string(),
                frequency: 0.3,
                complexity: 2.0,
                language_universal: false,
            },
            PhonologicalPattern {
                name: "Vowel Hiatus".to_string(),
                pattern_type: "sequence".to_string(),
                structural_description: "consecutive vowels".to_string(),
                frequency: 0.1,
                complexity: 1.5,
                language_universal: false,
            },
        ]
    }

    fn create_matching_thresholds() -> HashMap<String, f64> {
        let mut thresholds = HashMap::new();
        thresholds.insert("exact".to_string(), 1.0);
        thresholds.insert("close".to_string(), 0.8);
        thresholds.insert("approximate".to_string(), 0.6);
        thresholds
    }

    fn create_pattern_frequencies() -> HashMap<String, f64> {
        let mut frequencies = HashMap::new();
        frequencies.insert("CV Syllable".to_string(), 0.6);
        frequencies.insert("CVC Syllable".to_string(), 0.4);
        frequencies.insert("Consonant Cluster".to_string(), 0.3);
        frequencies.insert("Vowel Hiatus".to_string(), 0.1);
        frequencies
    }

    fn create_context_weights() -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        weights.insert("word-initial".to_string(), 1.2);
        weights.insert("word-medial".to_string(), 1.0);
        weights.insert("word-final".to_string(), 1.1);
        weights
    }

    pub(super) fn update_config(&mut self, config: &PhonologicalAnalysisConfig) {
        if config.extended_pattern_matching {
            self.pattern_database
                .extend(Self::create_extended_patterns());
        }
    }

    fn create_extended_patterns() -> Vec<PhonologicalPattern> {
        vec![PhonologicalPattern {
            name: "CCVC Syllable".to_string(),
            pattern_type: "syllable".to_string(),
            structural_description: "consonant cluster + vowel + consonant".to_string(),
            frequency: 0.2,
            complexity: 2.5,
            language_universal: false,
        }]
    }
}
