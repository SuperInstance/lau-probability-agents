//! Unified API — AgentBayes struct with full probabilistic reasoning.

use crate::conditioning::{
    bayes_update, sequential_bayes, ConjugateFamily, BayesianUpdate,
};
use crate::convergence::{
    check_convergence_in_probability, check_lp_convergence, empirical_cdf,
    track_convergence, ConvergenceDiagnostics,
};
use crate::central_limit::{clt_confidence_interval, phi, phi_inv, ConfidenceInterval};
use crate::expectation::{
    covariance, correlation, expectation, sample_variance, skewness, kurtosis,
    raw_moment, mgf,
};
use crate::large_deviations::{
    cramer_rate_function, sanov_probability, normal_rate_function, large_deviation_analysis,
    LargeDeviationResult,
};
use crate::markov::{MarkovChain, HMM};
use crate::martingale::{
    Martingale, doob_maximal_inequality, azuma_hoeffding_bound, martingale_from_increments,
};
use crate::measure::{DiscreteMeasure, ContinuousDistribution};
use crate::stochastic_process::{GaussianProcess, PoissonProcess, RenewalProcess};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unified probabilistic reasoning agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBayes {
    /// Agent identifier.
    pub id: String,
    /// Current belief state (discrete).
    pub beliefs: DiscreteMeasure,
    /// Conjugate family for continuous updates.
    pub conjugate: Option<ConjugateFamily>,
    /// History of Bayesian updates.
    pub update_history: Vec<BayesianUpdate>,
    /// Observed data.
    pub observations: Vec<f64>,
    /// Markov chain model.
    pub markov_chain: Option<MarkovChain>,
    /// GP surrogate model.
    #[serde(skip)]
    pub gp_model: Option<GaussianProcessData>,
}

/// Serializable GP data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianProcessData {
    pub train_x: Vec<f64>,
    pub train_y: Vec<f64>,
    pub length_scale: f64,
    pub noise_variance: f64,
}

impl AgentBayes {
    /// Create a new agent with uniform prior over hypotheses.
    pub fn new(id: &str, hypotheses: &[&str]) -> Self {
        Self {
            id: id.to_string(),
            beliefs: DiscreteMeasure::uniform(hypotheses),
            conjugate: None,
            update_history: vec![],
            observations: vec![],
            markov_chain: None,
            gp_model: None,
        }
    }

    /// Create agent with a specific prior.
    pub fn with_prior(id: &str, prior: DiscreteMeasure) -> Self {
        Self {
            id: id.to_string(),
            beliefs: prior,
            conjugate: None,
            update_history: vec![],
            observations: vec![],
            markov_chain: None,
            gp_model: None,
        }
    }

    /// Create agent with conjugate prior for continuous reasoning.
    pub fn with_conjugate(id: &str, conjugate: ConjugateFamily) -> Self {
        Self {
            id: id.to_string(),
            beliefs: DiscreteMeasure::uniform(&["hypothesis"]),
            conjugate: Some(conjugate),
            update_history: vec![],
            observations: vec![],
            markov_chain: None,
            gp_model: None,
        }
    }

    // === Belief Updates ===

    /// Update beliefs with evidence using Bayes' theorem.
    pub fn observe(&mut self, likelihood: &HashMap<String, f64>) -> &BayesianUpdate {
        let update = bayes_update(&self.beliefs, likelihood);
        self.beliefs = update.posterior.clone();
        self.update_history.push(update);
        self.update_history.last().unwrap()
    }

    /// Observe continuous data and update conjugate prior.
    pub fn observe_continuous(&mut self, data: &[f64]) {
        self.observations.extend(data);
        if let Some(ref conj) = self.conjugate {
            self.conjugate = Some(conj.update(data));
        }
    }

    /// Sequential observation of multiple evidence.
    pub fn observe_sequence(&mut self, evidence_stream: &[HashMap<String, f64>]) -> &[BayesianUpdate] {
        let updates = sequential_bayes(&self.beliefs, evidence_stream);
        if let Some(last) = updates.last() {
            self.beliefs = last.posterior.clone();
        }
        self.update_history.extend(updates);
        &self.update_history
    }

    // === Belief Queries ===

    /// Get probability of a hypothesis.
    pub fn belief(&self, hypothesis: &str) -> f64 {
        self.beliefs.prob(hypothesis)
    }

    /// Get the most likely hypothesis.
    pub fn map_hypothesis(&self) -> (&str, f64) {
        self.beliefs
            .weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, &v)| (k.as_str(), v))
            .unwrap_or(("none", 0.0))
    }

    /// Posterior mean (for conjugate models).
    pub fn posterior_mean(&self) -> f64 {
        self.conjugate
            .as_ref()
            .map(|c| c.posterior_mean())
            .unwrap_or_else(|| expectation(&self.observations, |x| x))
    }

    /// Posterior variance (for conjugate models).
    pub fn posterior_variance(&self) -> f64 {
        self.conjugate
            .as_ref()
            .map(|c| c.posterior_variance())
            .unwrap_or_else(|| sample_variance(&self.observations))
    }

    /// Confidence interval for the mean.
    pub fn confidence_interval(&self, confidence: f64) -> ConfidenceInterval {
        let mean = self.posterior_mean();
        let var = self.posterior_variance();
        let n = self.observations.len().max(1);
        clt_confidence_interval(mean, var, n, confidence)
    }

    // === Decision Making ===

    /// Maximum a posteriori decision.
    pub fn decide(&self) -> &str {
        self.map_hypothesis().0
    }

    /// Expected utility of an action.
    pub fn expected_utility<F>(&self, utility_fn: F) -> f64
    where
        F: Fn(&str) -> f64,
    {
        self.beliefs.expectation(utility_fn)
    }

    /// Thompson sampling: sample a hypothesis weighted by belief.
    pub fn thompson_sample(&self, rng: &mut impl FnMut() -> f64) -> &str {
        let r = rng();
        let mut cumsum = 0.0;
        for (hypothesis, &prob) in &self.beliefs.weights {
            cumsum += prob;
            if r < cumsum {
                return hypothesis.as_str();
            }
        }
        self.beliefs.weights.keys().next().map(|s| s.as_str()).unwrap_or("none")
    }

    // === Statistics ===

    /// Compute summary statistics of observations.
    pub fn summary(&self) -> SummaryStatistics {
        let obs = &self.observations;
        SummaryStatistics {
            n: obs.len(),
            mean: expectation(obs, |x| x),
            variance: sample_variance(obs),
            skewness: skewness(obs),
            kurtosis: kurtosis(obs),
        }
    }

    /// Entropy of current beliefs (uncertainty measure).
    pub fn entropy(&self) -> f64 {
        self.beliefs.entropy()
    }

    /// KL divergence from current beliefs to another distribution.
    pub fn kl_to(&self, other: &DiscreteMeasure) -> f64 {
        self.beliefs.kl_divergence(other)
    }

    // === Convergence ===

    /// Check if beliefs have converged (low entropy).
    pub fn has_converged(&self, threshold: f64) -> bool {
        self.entropy() < threshold
    }

    /// Track convergence of observations.
    pub fn track_observation_convergence(&self, expected_mean: f64, expected_var: f64, tolerance: f64) -> ConvergenceDiagnostics {
        track_convergence(&self.observations, expected_mean, expected_var, tolerance)
    }

    // === Large Deviations ===

    /// Compute rate function for observed mean.
    pub fn rate_function(&self, x: f64, dist: &ContinuousDistribution) -> f64 {
        match dist {
            ContinuousDistribution::Normal { mean, variance } => normal_rate_function(x, *mean, *variance),
            _ => 0.0,
        }
    }

    /// Large deviation probability approximation.
    pub fn large_deviation_prob(&self, x: f64, n: usize) -> LargeDeviationResult {
        let mean = self.posterior_mean();
        let var = self.posterior_variance().max(0.01);
        large_deviation_analysis(
            |t| (mean * t + 0.5 * var * t * t).exp(),
            x,
            n,
            (-10.0, 10.0),
            1000,
        )
    }

    // === Martingale ===

    /// Construct a martingale from observations.
    pub fn observation_martingale(&self, initial: f64) -> Martingale {
        let increments: Vec<f64> = self.observations
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        martingale_from_increments(&increments, initial)
    }

    /// Azuma-Hoeffding concentration bound.
    pub fn concentration_bound(&self, t: f64) -> f64 {
        let diffs: Vec<f64> = self.observations
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
        if diffs.is_empty() { return 1.0; }
        azuma_hoeffding_bound(&diffs, t)
    }

    // === Markov Chain ===

    /// Set a Markov chain model.
    pub fn set_markov_chain(&mut self, mc: MarkovChain) {
        self.markov_chain = Some(mc);
    }

    /// Predict next state using Markov chain.
    pub fn predict_next_state(&self, current_state: usize) -> Option<usize> {
        self.markov_chain.as_ref().map(|mc| {
            let row: Vec<f64> = (0..mc.n_states)
                .map(|j| mc.transition_matrix[(current_state, j)])
                .collect();
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
    }

    /// Get stationary distribution of Markov chain.
    pub fn stationary_distribution(&self) -> Option<Vec<f64>> {
        self.markov_chain
            .as_ref()
            .and_then(|mc| mc.stationary_distribution())
            .map(|pi| (0..pi.ncols()).map(|j| pi[(0, j)]).collect())
    }

    // === GP Surrogate ===

    /// Fit a GP surrogate model.
    pub fn fit_gp(&mut self, length_scale: f64, noise_variance: f64) {
        self.gp_model = Some(GaussianProcessData {
            train_x: (0..self.observations.len()).map(|i| i as f64).collect(),
            train_y: self.observations.clone(),
            length_scale,
            noise_variance,
        });
    }

    /// Predict using GP surrogate.
    pub fn gp_predict(&self, x_new: f64) -> Option<(f64, f64)> {
        self.gp_model.as_ref().map(|data| {
            let mut gp = GaussianProcess::new_rbf(data.length_scale, data.noise_variance);
            gp.fit(data.train_x.clone(), data.train_y.clone());
            gp.predict(x_new)
        })
    }

    /// Get number of observations.
    pub fn n_observations(&self) -> usize {
        self.observations.len()
    }

    /// Get number of belief updates.
    pub fn n_updates(&self) -> usize {
        self.update_history.len()
    }
}

/// Summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryStatistics {
    pub n: usize,
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_agent_creation() {
        let agent = AgentBayes::new("test", &["H1", "H2", "H3"]);
        assert_eq!(agent.id, "test");
        assert_relative_eq!(agent.belief("H1"), 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bayesian_update() {
        let mut agent = AgentBayes::new("test", &["fair", "biased"]);
        let mut lik = HashMap::new();
        lik.insert("fair".into(), 0.3);
        lik.insert("biased".into(), 0.8);
        agent.observe(&lik);
        assert!(agent.belief("biased") > agent.belief("fair"));
        assert_eq!(agent.n_updates(), 1);
    }

    #[test]
    fn test_map_hypothesis() {
        let mut agent = AgentBayes::new("test", &["A", "B"]);
        let mut lik = HashMap::new();
        lik.insert("A".into(), 0.1);
        lik.insert("B".into(), 0.9);
        agent.observe(&lik);
        assert_eq!(agent.decide(), "B");
    }

    #[test]
    fn test_conjugate_update() {
        let conj = ConjugateFamily::BetaBinomial { alpha: 1.0, beta: 1.0 };
        let mut agent = AgentBayes::with_conjugate("test", conj);
        agent.observe_continuous(&[1.0, 1.0, 0.0]);
        assert_relative_eq!(agent.posterior_mean(), 3.0 / 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_confidence_interval() {
        let conj = ConjugateFamily::NormalNormal { mu_0: 5.0, sigma_0_sq: 0.1, sigma_sq: 1.0 };
        let agent = AgentBayes::with_conjugate("test", conj);
        let ci = agent.confidence_interval(0.95);
        assert!(ci.lower < 5.0);
        assert!(ci.upper > 5.0);
    }

    #[test]
    fn test_summary_statistics() {
        let conj = ConjugateFamily::BetaBinomial { alpha: 1.0, beta: 1.0 };
        let mut agent = AgentBayes::with_conjugate("test", conj);
        agent.observe_continuous(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let summary = agent.summary();
        assert_eq!(summary.n, 5);
        assert_relative_eq!(summary.mean, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_decreases() {
        let mut agent = AgentBayes::new("test", &["A", "B"]);
        let e0 = agent.entropy();
        let mut lik = HashMap::new();
        lik.insert("A".into(), 0.9);
        lik.insert("B".into(), 0.1);
        agent.observe(&lik);
        let e1 = agent.entropy();
        assert!(e1 < e0);
    }

    #[test]
    fn test_sequential_observations() {
        let mut agent = AgentBayes::new("test", &["H1", "H2"]);
        let e1: HashMap<String, f64> = [("H1".into(), 0.9), ("H2".into(), 0.1)].into();
        let e2: HashMap<String, f64> = [("H1".into(), 0.8), ("H2".into(), 0.2)].into();
        agent.observe_sequence(&[e1, e2]);
        assert_eq!(agent.n_updates(), 2);
        assert!(agent.belief("H1") > 0.8);
    }

    #[test]
    fn test_thompson_sampling() {
        let mut agent = AgentBayes::new("test", &["A", "B"]);
        let mut lik = HashMap::new();
        lik.insert("A".into(), 0.9);
        lik.insert("B".into(), 0.1);
        agent.observe(&lik);
        let mut counter = 0;
        let sample = agent.thompson_sample(&mut || { counter += 1; (counter as f64 * 0.618) % 1.0 });
        assert!(sample == "A" || sample == "B");
    }

    #[test]
    fn test_expected_utility() {
        let mut agent = AgentBayes::new("test", &["A", "B"]);
        let mut lik = HashMap::new();
        lik.insert("A".into(), 0.7);
        lik.insert("B".into(), 0.3);
        agent.observe(&lik);
        let eu = agent.expected_utility(|h| if h == "A" { 10.0 } else { 5.0 });
        assert!(eu > 0.0);
    }

    #[test]
    fn test_has_converged() {
        let mut agent = AgentBayes::new("test", &["A", "B"]);
        assert!(!agent.has_converged(0.1)); // Uniform = max entropy
        let mut lik = HashMap::new();
        lik.insert("A".into(), 0.99);
        lik.insert("B".into(), 0.01);
        agent.observe(&lik);
        agent.observe(&lik);
        agent.observe(&lik);
        assert!(agent.has_converged(0.2));
    }

    #[test]
    fn test_concentration_bound() {
        let conj = ConjugateFamily::BetaBinomial { alpha: 1.0, beta: 1.0 };
        let mut agent = AgentBayes::with_conjugate("test", conj);
        agent.observe_continuous(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let bound = agent.concentration_bound(1.0);
        assert!(bound >= 0.0 && bound <= 2.0);
    }

    #[test]
    fn test_agent_with_prior() {
        let mut w = std::collections::HashMap::new();
        w.insert("X".into(), 0.8);
        w.insert("Y".into(), 0.2);
        let prior = DiscreteMeasure { weights: w };
        let agent = AgentBayes::with_prior("test", prior);
        assert_relative_eq!(agent.belief("X"), 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_large_deviation() {
        let conj = ConjugateFamily::NormalNormal { mu_0: 0.0, sigma_0_sq: 1.0, sigma_sq: 1.0 };
        let agent = AgentBayes::with_conjugate("test", conj);
        let ld = agent.large_deviation_prob(3.0, 100);
        assert!(ld.rate_function_value > 0.0);
    }

    #[test]
    fn test_markov_chain_integration() {
        use nalgebra::DMatrix;
        let mut agent = AgentBayes::new("test", &["s0", "s1"]);
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        agent.set_markov_chain(MarkovChain::new(p));
        let next = agent.predict_next_state(0);
        assert_eq!(next, Some(0)); // 0.7 > 0.3
        let sd = agent.stationary_distribution();
        assert!(sd.is_some());
    }

    #[test]
    fn test_gp_surrogate() {
        let conj = ConjugateFamily::NormalNormal { mu_0: 0.0, sigma_0_sq: 1.0, sigma_sq: 0.1 };
        let mut agent = AgentBayes::with_conjugate("test", conj);
        agent.observe_continuous(&[0.0, 0.5, 1.0, 0.5, 0.0]);
        agent.fit_gp(1.0, 0.01);
        let pred = agent.gp_predict(2.0);
        assert!(pred.is_some());
        let (mean, var) = pred.unwrap();
        assert!(var >= 0.0);
    }
}
