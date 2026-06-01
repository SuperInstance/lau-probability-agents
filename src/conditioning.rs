//! Bayesian conditioning, prior-posterior updates, conjugate families.

use crate::measure::DiscreteMeasure;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of a Bayesian update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianUpdate {
    pub posterior: DiscreteMeasure,
    pub marginal_likelihood: f64,
    pub bayes_factor: f64,
}

/// Perform Bayesian conditioning: P(hypothesis | evidence) ∝ P(evidence | hypothesis) * P(hypothesis).
pub fn bayes_update(
    prior: &DiscreteMeasure,
    likelihood: &HashMap<String, f64>,
) -> BayesianUpdate {
    let mut posterior_weights = HashMap::new();
    let mut marginal = 0.0;

    for hypothesis in prior.support() {
        let prior_prob = prior.prob(hypothesis);
        let lik = likelihood.get(hypothesis).copied().unwrap_or(0.0);
        let unnorm = prior_prob * lik;
        posterior_weights.insert(hypothesis.to_string(), unnorm);
        marginal += unnorm;
    }

    if marginal.abs() > 1e-15 {
        for v in posterior_weights.values_mut() {
            *v /= marginal;
        }
    }

    let bayes_factor = if let (Some(h1), Some(h2)) = (prior.support().first(), prior.support().get(1)) {
        let post1 = posterior_weights.get(*h1).copied().unwrap_or(0.0);
        let post2 = posterior_weights.get(*h2).copied().unwrap_or(0.0);
        let prior1 = prior.prob(*h1);
        let prior2 = prior.prob(*h2);
        if prior2 > 0.0 && post2 > 0.0 {
            (post1 / post2) / (prior1 / prior2)
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    BayesianUpdate {
        posterior: DiscreteMeasure { weights: posterior_weights },
        marginal_likelihood: marginal,
        bayes_factor,
    }
}

/// Sequential Bayesian update with a stream of evidence.
pub fn sequential_bayes(
    prior: &DiscreteMeasure,
    evidence_stream: &[HashMap<String, f64>],
) -> Vec<BayesianUpdate> {
    let mut results = Vec::new();
    let mut current = prior.clone();
    for evidence in evidence_stream {
        let update = bayes_update(&current, evidence);
        current = update.posterior.clone();
        results.push(update);
    }
    results
}

/// Conjugate prior families for common likelihoods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConjugateFamily {
    /// Beta-Binomial: Beta(α, β) prior, Binomial likelihood
    BetaBinomial { alpha: f64, beta: f64 },
    /// Normal-Normal: Normal(μ₀, σ₀²) prior on mean, known variance σ²
    NormalNormal { mu_0: f64, sigma_0_sq: f64, sigma_sq: f64 },
    /// Gamma-Poisson: Gamma(α, β) prior, Poisson likelihood
    GammaPoisson { alpha: f64, beta: f64 },
}

impl ConjugateFamily {
    /// Update the conjugate prior with observed data.
    pub fn update(&self, data: &[f64]) -> Self {
        match self {
            ConjugateFamily::BetaBinomial { alpha, beta } => {
                let successes = data.iter().filter(|&&x| x == 1.0).count() as f64;
                let failures = data.len() as f64 - successes;
                ConjugateFamily::BetaBinomial {
                    alpha: alpha + successes,
                    beta: beta + failures,
                }
            }
            ConjugateFamily::NormalNormal { mu_0, sigma_0_sq, sigma_sq } => {
                let n = data.len() as f64;
                let x_bar = data.iter().sum::<f64>() / n;
                let precision_prior = 1.0 / sigma_0_sq;
                let precision_data = n / sigma_sq;
                let precision_post = precision_prior + precision_data;
                let mu_post = (precision_prior * mu_0 + precision_data * x_bar) / precision_post;
                let sigma_post_sq = 1.0 / precision_post;
                ConjugateFamily::NormalNormal {
                    mu_0: mu_post,
                    sigma_0_sq: sigma_post_sq,
                    sigma_sq: *sigma_sq,
                }
            }
            ConjugateFamily::GammaPoisson { alpha, beta } => {
                let sum: f64 = data.iter().sum();
                let n = data.len() as f64;
                ConjugateFamily::GammaPoisson {
                    alpha: alpha + sum,
                    beta: beta + n,
                }
            }
        }
    }

    /// Posterior mean.
    pub fn posterior_mean(&self) -> f64 {
        match self {
            ConjugateFamily::BetaBinomial { alpha, beta } => alpha / (alpha + beta),
            ConjugateFamily::NormalNormal { mu_0, .. } => *mu_0,
            ConjugateFamily::GammaPoisson { alpha, beta } => alpha / beta,
        }
    }

    /// Posterior variance.
    pub fn posterior_variance(&self) -> f64 {
        match self {
            ConjugateFamily::BetaBinomial { alpha, beta } => {
                (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0))
            }
            ConjugateFamily::NormalNormal { sigma_0_sq, .. } => *sigma_0_sq,
            ConjugateFamily::GammaPoisson { alpha, beta } => alpha / beta.powi(2),
        }
    }

    /// Posterior predictive probability for a new observation.
    pub fn predictive_prob(&self, x: f64) -> f64 {
        match self {
            ConjugateFamily::BetaBinomial { alpha, beta } => {
                if x == 1.0 {
                    alpha / (alpha + beta)
                } else {
                    beta / (alpha + beta)
                }
            }
            ConjugateFamily::NormalNormal { mu_0, sigma_0_sq, sigma_sq } => {
                let pred_var = sigma_0_sq + sigma_sq;
                let pred_sigma = pred_var.sqrt();
                (-(x - mu_0).powi(2) / (2.0 * pred_var)).exp()
                    / (pred_sigma * (2.0 * std::f64::consts::PI).sqrt())
            }
            ConjugateFamily::GammaPoisson { alpha, beta } => {
                // Negative binomial predictive
                let k = x as u64;
                let r = *alpha;
                let p = *beta / (beta + 1.0);
                let ln_pmf = crate::measure::ln_gamma(k as f64 + r)
                    - crate::measure::ln_gamma(k as f64 + 1.0)
                    - crate::measure::ln_gamma(r)
                    + r * p.ln()
                    + (k as f64) * (1.0 - p).ln();
                ln_pmf.exp()
            }
        }
    }
}

/// Likelihood functions for common distributions.
pub mod likelihoods {
    /// Bernoulli likelihood.
    pub fn bernoulli(x: f64, p: f64) -> f64 {
        if x == 1.0 { p } else { 1.0 - p }
    }

    /// Gaussian likelihood.
    pub fn gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
        let var = sigma * sigma;
        (-(x - mu).powi(2) / (2.0 * var)).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
    }

    /// Poisson likelihood.
    pub fn poisson(k: u64, lambda: f64) -> f64 {
        let k_f = k as f64;
        (-lambda + k_f * lambda.ln() - crate::measure::ln_gamma(k_f + 1.0)).exp()
    }

    /// Build a likelihood map for a set of hypotheses given data.
    pub fn likelihood_map(
        hypotheses: &[&str],
        params: &[f64],
        data: f64,
        dist: &str,
    ) -> std::collections::HashMap<String, f64> {
        hypotheses
            .iter()
            .zip(params.iter())
            .map(|(&h, &p)| {
                let lik = match dist {
                    "bernoulli" => bernoulli(data, p),
                    "gaussian" => gaussian(data, 0.0, p),
                    _ => 0.0,
                };
                (h.to_string(), lik)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bayes_simple() {
        let mut w = HashMap::new();
        w.insert("fair".into(), 0.5);
        w.insert("biased".into(), 0.5);
        let prior = DiscreteMeasure { weights: w };

        let mut lik = HashMap::new();
        lik.insert("fair".into(), 0.5);
        lik.insert("biased".into(), 0.9);

        let update = bayes_update(&prior, &lik);
        assert_relative_eq!(update.marginal_likelihood, 0.7, epsilon = 1e-10);
        assert_relative_eq!(update.posterior.prob("biased"), 0.9 / 1.4, epsilon = 1e-10);
    }

    #[test]
    fn test_sequential_bayes() {
        let mut w = HashMap::new();
        w.insert("H1".into(), 0.5);
        w.insert("H2".into(), 0.5);
        let prior = DiscreteMeasure { weights: w };

        let e1: HashMap<String, f64> = [("H1".into(), 0.8), ("H2".into(), 0.2)].into();
        let e2: HashMap<String, f64> = [("H1".into(), 0.7), ("H2".into(), 0.3)].into();

        let results = sequential_bayes(&prior, &[e1, e2]);
        assert_eq!(results.len(), 2);
        // After two updates favoring H1, posterior for H1 should be > prior
        assert!(results[1].posterior.prob("H1") > 0.5);
    }

    #[test]
    fn test_beta_binomial_conjugate() {
        let prior = ConjugateFamily::BetaBinomial { alpha: 1.0, beta: 1.0 };
        assert_relative_eq!(prior.posterior_mean(), 0.5, epsilon = 1e-10);

        let post = prior.update(&[1.0, 1.0, 0.0, 1.0]);
        // 3 successes, 1 failure: Beta(4, 2)
        assert_relative_eq!(post.posterior_mean(), 4.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_normal_conjugate() {
        let prior = ConjugateFamily::NormalNormal {
            mu_0: 0.0,
            sigma_0_sq: 1.0,
            sigma_sq: 1.0,
        };

        let post = prior.update(&[2.0, 3.0, 4.0]);
        // n=3, x_bar=3, precision_prior=1, precision_data=3
        // mu_post = (1*0 + 3*3) / 4 = 9/4
        assert_relative_eq!(post.posterior_mean(), 2.25, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_poisson_conjugate() {
        let prior = ConjugateFamily::GammaPoisson { alpha: 1.0, beta: 1.0 };
        let post = prior.update(&[3.0, 5.0, 2.0]);
        // alpha = 1 + 10 = 11, beta = 1 + 3 = 4
        assert_relative_eq!(post.posterior_mean(), 11.0 / 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predictive_beta_binomial() {
        let post = ConjugateFamily::BetaBinomial { alpha: 3.0, beta: 2.0 };
        assert_relative_eq!(post.predictive_prob(1.0), 0.6, epsilon = 1e-10);
        assert_relative_eq!(post.predictive_prob(0.0), 0.4, epsilon = 1e-10);
    }

    #[test]
    fn test_likelihood_bernoulli() {
        assert_relative_eq!(likelihoods::bernoulli(1.0, 0.7), 0.7, epsilon = 1e-10);
        assert_relative_eq!(likelihoods::bernoulli(0.0, 0.7), 0.3, epsilon = 1e-10);
    }

    #[test]
    fn test_likelihood_gaussian() {
        let p = likelihoods::gaussian(0.0, 0.0, 1.0);
        assert_relative_eq!(p, 1.0 / (2.0 * std::f64::consts::PI).sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_likelihood_poisson() {
        assert_relative_eq!(likelihoods::poisson(0, 1.0), (-1.0f64).exp(), epsilon = 1e-10);
    }
}
