/*!
# cuda-attention

Resource-limited attention allocation for agents.

Attention is the bottleneck of intelligence. You can't think about everything
at once. An agent that tries to attend to everything attends to nothing.

This crate implements:
- Attention budget with capacity limits
- Saliency scoring — what deserves attention
- Attention allocation — distributing budget across candidates
- Habituation — familiar stimuli get less attention
- Sudden change detection — novelty demands attention
- Focus modes — narrow (one thing) vs broad (many things)
*/

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Something that might deserve attention
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionCandidate {
    pub id: String,
    pub saliency: f64,      // inherent importance [0, 1]
    pub novelty: f64,       // how new is this [0, 1]
    pub relevance: f64,     // relevance to current goal [0, 1]
    pub urgency: f64,       // time pressure [0, 1]
    pub familiarity: f64,   // how often seen [0, 1] (1 = very familiar)
    pub age: u64,           // how long this candidate has existed
}

impl AttentionCandidate {
    pub fn composite_score(&self) -> f64 {
        // Novelty and relevance boost, familiarity reduces
        let base = self.saliency * 0.3 + self.novelty * 0.25 + self.relevance * 0.25 + self.urgency * 0.2;
        let familiarity_discount = 1.0 - self.familiarity * 0.5; // familiar things get 50% less
        base * familiarity_discount
    }
}

/// Focus mode
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FocusMode {
    Diffuse,     // broad, low threshold — attend to many things
    Normal,      // balanced
    Focused,     // narrow, high threshold — attend to few important things
    Hyperfocus,  // extremely narrow — one thing at a time
}

impl FocusMode {
    /// How many items can be attended simultaneously
    pub fn capacity(&self) -> usize {
        match self {
            FocusMode::Diffuse => 7,
            FocusMode::Normal => 5,
            FocusMode::Focused => 3,
            FocusMode::Hyperfocus => 1,
        }
    }

    /// Minimum saliency threshold for attention
    pub fn threshold(&self) -> f64 {
        match self {
            FocusMode::Diffuse => 0.2,
            FocusMode::Normal => 0.35,
            FocusMode::Focused => 0.5,
            FocusMode::Hyperfocus => 0.7,
        }
    }
}

/// An attention allocation — what the agent is attending to
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionAllocation {
    pub id: String,
    pub budget: f64,        // fraction of total attention [0, 1]
    pub score: f64,
    pub granted: u64,       // timestamp when allocated
    pub duration_ms: u64,   // how long this allocation lasts
}

/// Habituation tracker — familiar stimuli lose attention
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HabituationTracker {
    pub exposure_counts: HashMap<String, u32>,
    pub last_seen: HashMap<String, u64>,
    pub decay_rate: f64,    // how fast exposure count decays
}

impl HabituationTracker {
    pub fn new() -> Self { HabituationTracker { exposure_counts: HashMap::new(), last_seen: HashMap::new(), decay_rate: 0.01 } }

    pub fn expose(&mut self, id: &str) {
        *self.exposure_counts.entry(id.to_string()).or_insert(0) += 1;
        self.last_seen.insert(id.to_string(), now());
    }

    /// Familiarity [0, 1] — higher = more familiar
    pub fn familiarity(&self, id: &str) -> f64 {
        let count = *self.exposure_counts.get(id).unwrap_or(&0);
        (1.0 - (-count as f64 * self.decay_rate).exp()).min(0.95)
    }

    /// Decay old exposures
    pub fn decay(&mut self, current_time: u64, half_life_ms: u64) {
        for (id, last) in self.last_seen.iter_mut() {
            let age = current_time.saturating_sub(*last);
            let factor = 0.5_f64.powf(age as f64 / half_life_ms as f64);
            if let Some(count) = self.exposure_counts.get_mut(id) {
                *count = (*count as f64 * factor) as u32;
                if *count == 0 { *count = 0; }
            }
        }
    }
}

/// Change detector — sudden changes demand attention
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChangeDetector {
    pub history: HashMap<String, VecDeque<f64>>,
    pub history_size: usize,
    pub change_threshold: f64, // how different is "sudden"
}

impl ChangeDetector {
    pub fn new() -> Self { ChangeDetector { history: HashMap::new(), history_size: 10, change_threshold: 0.3 } }

    /// Update with new value, return novelty score [0, 1]
    pub fn update(&mut self, id: &str, value: f64) -> f64 {
        let entry = self.history.entry(id.to_string()).or_insert_with(|| VecDeque::with_capacity(self.history_size));
        let novelty = if entry.len() >= 3 {
            let mean: f64 = entry.iter().sum::<f64>() / entry.len() as f64;
            let std: f64 = {
                let variance: f64 = entry.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / entry.len() as f64;
                variance.sqrt()
            };
            if std < 0.001 { 0.0 } else { ((value - mean).abs() / std / self.change_threshold).min(1.0) }
        } else {
            // Not enough history — everything is novel
            0.5
        };

        if entry.len() >= self.history_size { entry.pop_front(); }
        entry.push_back(value);
        novelty
    }
}

/// The attention engine
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionEngine {
    pub mode: FocusMode,
    pub allocations: HashMap<String, AttentionAllocation>,
    pub habituation: HabituationTracker,
    pub change_detector: ChangeDetector,
    pub total_budget: f64,
    pub attention_log: Vec<AttentionEvent>,
    pub log_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionEvent {
    pub candidate_id: String,
    pub score: f64,
    pub budget_allocated: f64,
    pub mode: FocusMode,
    pub timestamp: u64,
}

impl AttentionEngine {
    pub fn new() -> Self {
        AttentionEngine { mode: FocusMode::Normal, allocations: HashMap::new(), habituation: HabituationTracker::new(), change_detector: ChangeDetector::new(), total_budget: 1.0, attention_log: vec![], log_size: 100 }
    }

    /// Set focus mode
    pub fn set_mode(&mut self, mode: FocusMode) { self.mode = mode; }

    /// Process candidates and allocate attention
    pub fn allocate(&mut self, candidates: Vec<AttentionCandidate>) -> Vec<AttentionAllocation> {
        let capacity = self.mode.capacity();
        let threshold = self.mode.threshold();

        // Score and sort
        let mut scored: Vec<(String, f64)> = candidates.iter()
            .map(|c| {
                let mut c = c.clone();
                c.familiarity = self.habituation.familiarity(&c.id);
                (c.id.clone(), c.composite_score())
            })
            .filter(|(_, score)| *score >= threshold)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top-N
        scored.truncate(capacity);

        // Normalize scores to budget
        let total_score: f64 = scored.iter().map(|(_, s)| s).sum();
        self.allocations.clear();

        let results = if total_score < 0.001 {
            vec![]
        } else {
            scored.iter().map(|(id, score)| {
                let budget = (score / total_score * self.total_budget).clamp(0.01, 1.0);
                let alloc = AttentionAllocation { id: id.clone(), budget, score: *score, granted: now(), duration_ms: 1000 };
                self.allocations.insert(id.clone(), alloc.clone());

                // Update habituation
                self.habituation.expose(id);

                // Log
                self.attention_log.push(AttentionEvent { candidate_id: id.clone(), score: *score, budget_allocated: budget, mode: self.mode, timestamp: now() });
                if self.attention_log.len() > self.log_size { self.attention_log.remove(0); }

                alloc
            }).collect()
        };

        results
    }

    /// Get current attention state
    pub fn attention_summary(&self) -> AttentionSummary {
        let allocated: f64 = self.allocations.values().map(|a| a.budget).sum();
        AttentionSummary { mode: self.mode, capacity: self.mode.capacity(), allocated_count: self.allocations.len(), budget_used: allocated, budget_remaining: self.total_budget - allocated }
    }

    /// Decay habituation over time
    pub fn decay(&mut self, current_time: u64) {
        self.habituation.decay(current_time, 3600_000);
    }
}

#[derive(Clone, Debug)]
pub struct AttentionSummary {
    pub mode: FocusMode,
    pub capacity: usize,
    pub allocated_count: usize,
    pub budget_used: f64,
    pub budget_remaining: f64,
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_score() {
        let c = AttentionCandidate { id: "a".into(), saliency: 0.8, novelty: 0.9, relevance: 0.7, urgency: 0.5, familiarity: 0.0, age: 0 };
        let score = c.composite_score();
        assert!(score > 0.5);
    }

    #[test]
    fn test_familiarity_reduces_score() {
        let mut c1 = AttentionCandidate { id: "a".into(), saliency: 0.8, novelty: 0.5, relevance: 0.5, urgency: 0.5, familiarity: 0.0, age: 0 };
        let mut c2 = AttentionCandidate { id: "a".into(), saliency: 0.8, novelty: 0.5, relevance: 0.5, urgency: 0.5, familiarity: 0.9, age: 0 };
        assert!(c1.composite_score() > c2.composite_score());
    }

    #[test]
    fn test_focus_mode_capacity() {
        assert_eq!(FocusMode::Diffuse.capacity(), 7);
        assert_eq!(FocusMode::Hyperfocus.capacity(), 1);
    }

    #[test]
    fn test_habituation() {
        let mut h = HabituationTracker::new();
        for _ in 0..20 { h.expose("thing"); }
        let fam = h.familiarity("thing");
        assert!(fam > 0.5);
        assert_eq!(h.familiarity("unknown"), 0.0);
    }

    #[test]
    fn test_habituation_decay() {
        let mut h = HabituationTracker::new();
        for _ in 0..10 { h.expose("x"); }
        h.decay(100_000, 50_000);
        assert!(h.familiarity("x") < 1.0);
    }

    #[test]
    fn test_change_detector_stable() {
        let mut cd = ChangeDetector::new();
        for _ in 0..10 { cd.update("sensor", 10.0); }
        let novelty = cd.update("sensor", 10.1); // small change
        assert!(novelty < 0.5);
    }

    #[test]
    fn test_change_detector_sudden() {
        let mut cd = ChangeDetector::new();
        for _ in 0..10 { cd.update("sensor", 10.0); }
        let novelty = cd.update("sensor", 100.0); // big change
        assert!(novelty > 0.5);
    }

    #[test]
    fn test_change_detector_novel() {
        let mut cd = ChangeDetector::new();
        let novelty = cd.update("new_sensor", 42.0);
        assert_eq!(novelty, 0.5); // not enough history
    }

    #[test]
    fn test_attention_allocate() {
        let mut engine = AttentionEngine::new();
        let candidates = vec![
            AttentionCandidate { id: "a".into(), saliency: 0.9, novelty: 0.8, relevance: 0.9, urgency: 0.7, familiarity: 0.0, age: 0 },
            AttentionCandidate { id: "b".into(), saliency: 0.1, novelty: 0.1, relevance: 0.1, urgency: 0.1, familiarity: 0.0, age: 0 },
            AttentionCandidate { id: "c".into(), saliency: 0.7, novelty: 0.6, relevance: 0.7, urgency: 0.5, familiarity: 0.0, age: 0 },
        ];
        let allocs = engine.allocate(candidates);
        assert_eq!(allocs.len(), 2); // normal mode capacity = 5 but only 2 above threshold
    }

    #[test]
    fn test_attention_hyperfocus() {
        let mut engine = AttentionEngine::new();
        engine.set_mode(FocusMode::Hyperfocus);
        let candidates = vec![
            AttentionCandidate { id: "a".into(), saliency: 0.9, novelty: 0.9, relevance: 0.9, urgency: 0.9, familiarity: 0.0, age: 0 },
            AttentionCandidate { id: "b".into(), saliency: 0.8, novelty: 0.8, relevance: 0.8, urgency: 0.8, familiarity: 0.0, age: 0 },
        ];
        let allocs = engine.allocate(candidates);
        assert_eq!(allocs.len(), 1); // hyperfocus = only 1
    }

    #[test]
    fn test_attention_budget_sum() {
        let mut engine = AttentionEngine::new();
        let candidates = vec![
            AttentionCandidate { id: "a".into(), saliency: 0.9, novelty: 0.8, relevance: 0.8, urgency: 0.8, familiarity: 0.0, age: 0 },
            AttentionCandidate { id: "b".into(), saliency: 0.7, novelty: 0.7, relevance: 0.7, urgency: 0.7, familiarity: 0.0, age: 0 },
            AttentionCandidate { id: "c".into(), saliency: 0.5, novelty: 0.6, relevance: 0.6, urgency: 0.5, familiarity: 0.0, age: 0 },
        ];
        let allocs = engine.allocate(candidates);
        let total: f64 = allocs.iter().map(|a| a.budget).sum();
        assert!((total - 1.0).abs() < 0.01); // full budget used
    }

    #[test]
    fn test_attention_summary() {
        let engine = AttentionEngine::new();
        let s = engine.attention_summary();
        assert_eq!(s.mode, FocusMode::Normal);
        assert_eq!(s.allocated_count, 0);
    }
}
