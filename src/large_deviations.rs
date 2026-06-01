//! Cramer's theorem, Sanov's theorem, rate functions, Varadhan's lemma.

use crate::measure::{ContinuousDistribution, ln_gamma};
use serde::{Deserialize, Serialize};

/// Rate function for i.i.d. random variables under Cramer's theorem.
/// I(x) = sup_θ { θx - ln M(θ) } where M is the moment generating function.
pub fn cramer_rate_function<F>(mgf: F, x: f64, theta_range: (f64, f64), n_grid: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let (t_min, t_max) = theta_range;
    let step = (t_max - t_min) / n_grid as f64;
    let mut sup = f64::NEG_INFINITY;

    for i in 0..=n_grid {
        let theta = t_min + i as f64 * step;
        let log_mgf = if let Some(lm) = log_mgf_safe(mgf(theta)) {
            lm
        } else {
            continue;
        };
        let val = theta * x - log_mgf;
        if val > sup {
            sup = val;
        }
    }
    sup.max(0.0)
}

/// Safe logarithm that handles edge cases.
fn log_mgf_safe(mgf_val: f64) -> Option<f64> {
    if mgf_val > 0.0 && mgf_val.is_finite() {
        Some(mgf_val.ln())
    } else {
        None
    }
}

/// Log moment generating function for common distributions.
pub fn log_mgf(dist: &ContinuousDistribution, t: f64) -> f64 {
    match dist {
        ContinuousDistribution::Normal { mean, variance } => {
            mean * t + 0.5 * variance * t * t
        }
        ContinuousDistribution::Exponential { rate } => {
            if t < *rate {
                -(-*rate * t).ln_1p()
            } else {
                f64::INFINITY
            }
        }
        ContinuousDistribution::Gamma { shape, rate } => {
            if t < *rate {
                -shape * (1.0 - t / rate).ln()
            } else {
                f64::INFINITY
            }
        }
        ContinuousDistribution::Uniform { a: _, b: _ } => {
            // MGF of Uniform(0,1) = (e^t - 1) / t
            if t.abs() < 1e-10 {
                0.0
            } else {
                let mt = match dist {
                    ContinuousDistribution::Uniform { a, b } => {
                        let range = b - a;
                        if (t * range).abs() < 1e-10 {
                            a * t + 0.5 * range * t
                        } else {
                            ((t * b).exp() - (t * a).exp()) / (t * range)
                        }
                    }
                    _ => unreachable!(),
                };
                if mt > 0.0 { mt.ln() } else { f64::NEG_INFINITY }
            }
        }
    }
}

/// Cramer's theorem: P(S_n/n ≥ x) ≈ exp(-n * I(x)).
pub fn cramer_probability(mgf_log: &dyn Fn(f64) -> f64, x: f64, n: usize, theta_range: (f64, f64), n_grid: usize) -> f64 {
    let rate = cramer_rate_function(|t| mgf_log(t).exp(), x, theta_range, n_grid);
    (-(n as f64) * rate).exp()
}

/// Sanov's theorem: probability that empirical distribution falls in a set of distributions.
/// P(L_n ∈ A) ≈ exp(-n * inf_{Q ∈ A} D(Q || P))
pub fn sanov_rate(kl_divergence: f64) -> f64 {
    kl_divergence.max(0.0)
}

/// Sanov's probability approximation.
pub fn sanov_probability(kl_divergence: f64, n: usize) -> f64 {
    (-(n as f64) * kl_divergence).exp()
}

/// Relative entropy rate function for multinomial distributions.
pub fn multinomial_rate_function(
    empirical_probs: &[f64],
    true_probs: &[f64],
) -> f64 {
    empirical_probs
        .iter()
        .zip(true_probs.iter())
        .filter(|(&p, _)| p > 0.0)
        .map(|(p, q)| {
            if *q > 0.0 {
                p * (p / q).ln()
            } else {
                f64::INFINITY
            }
        })
        .sum::<f64>()
        .max(0.0)
}

/// Varadhan's lemma: for a function Φ and rate function I,
/// lim_{n→∞} (1/n) ln E[exp(n Φ(X_n))] = sup_x { Φ(x) - I(x) }.
pub fn varadhan_supremum<F, G>(phi: F, rate_fn: G, grid: &[f64]) -> f64
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    grid.iter()
        .map(|&x| phi(x) - rate_fn(x))
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Compute rate function for sample mean of normal distribution.
pub fn normal_rate_function(x: f64, mean: f64, variance: f64) -> f64 {
    (x - mean).powi(2) / (2.0 * variance)
}

/// Compute rate function for sample mean of exponential distribution.
pub fn exponential_rate_function(x: f64, rate: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    let mean = 1.0 / rate;
    x / mean - 1.0 - (x / mean).ln()
}

/// Large deviation principle structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeDeviationResult {
    pub rate_function_value: f64,
    pub log_probability_approx: f64,
    pub theta_optimal: f64,
}

/// Compute the full large deviation result including optimal tilting parameter.
pub fn large_deviation_analysis<F>(
    mgf: F,
    x: f64,
    n: usize,
    theta_range: (f64, f64),
    n_grid: usize,
) -> LargeDeviationResult
where
    F: Fn(f64) -> f64,
{
    let (t_min, t_max) = theta_range;
    let step = (t_max - t_min) / n_grid as f64;
    let mut sup = f64::NEG_INFINITY;
    let mut theta_opt = 0.0;

    for i in 0..=n_grid {
        let theta = t_min + i as f64 * step;
        let mgf_val = mgf(theta);
        if mgf_val <= 0.0 || !mgf_val.is_finite() { continue; }
        let val = theta * x - mgf_val.ln();
        if val > sup {
            sup = val;
            theta_opt = theta;
        }
    }

    let rate = sup.max(0.0);
    LargeDeviationResult {
        rate_function_value: rate,
        log_probability_approx: -(n as f64) * rate,
        theta_optimal: theta_opt,
    }
}

/// Contraction principle: if X_n satisfies LDP with rate I_X,
/// and Y = f(X), then Y_n satisfies LDP with rate I_Y(y) = inf{ I_X(x) : f(x) = y }.
pub fn contraction_principle<F, G>(
    f: F,
    rate_fn: G,
    y: f64,
    x_grid: &[f64],
    tolerance: f64,
) -> f64
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    x_grid
        .iter()
        .filter(|&&x| (f(x) - y).abs() < tolerance)
        .map(|&x| rate_fn(x))
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normal_rate_function() {
        let rate = normal_rate_function(0.0, 0.0, 1.0);
        assert_relative_eq!(rate, 0.0, epsilon = 1e-10);

        let rate2 = normal_rate_function(2.0, 0.0, 1.0);
        assert_relative_eq!(rate2, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cramer_rate_normal() {
        let dist = ContinuousDistribution::Normal { mean: 0.0, variance: 1.0 };
        let rate = cramer_rate_function(
            |t| (0.5 * t * t).exp(),
            2.0,
            (-5.0, 5.0),
            10000,
        );
        assert_relative_eq!(rate, 2.0, epsilon = 0.05);
    }

    #[test]
    fn test_exponential_rate_function() {
        let rate = exponential_rate_function(1.0, 1.0);
        assert_relative_eq!(rate, 0.0, epsilon = 1e-10);

        let rate2 = exponential_rate_function(2.0, 1.0);
        assert!(rate2 > 0.0);
    }

    #[test]
    fn test_sanov_probability() {
        let kl = 0.5;
        let prob = sanov_probability(kl, 100);
        assert_relative_eq!(prob, (-50.0f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_multinomial_rate_function() {
        let emp = vec![0.5, 0.5];
        let true_p = vec![0.5, 0.5];
        assert_relative_eq!(multinomial_rate_function(&emp, &true_p), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multinomial_rate_nonzero() {
        let emp = vec![0.75, 0.25];
        let true_p = vec![0.5, 0.5];
        let rate = multinomial_rate_function(&emp, &true_p);
        assert!(rate > 0.0);
        // KL(0.75 || 0.5) + KL(0.25 || 0.5)
        let expected = 0.75_f64 * (0.75_f64 / 0.5_f64).ln() + 0.25_f64 * (0.25_f64 / 0.5_f64).ln();
        assert_relative_eq!(rate, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_varadhan_supremum() {
        let phi = |x: f64| -x * x;
        let rate_fn = |x: f64| (x - 1.0).powi(2);
        let grid: Vec<f64> = (-50..=50).map(|i| i as f64 / 10.0).collect();
        let sup = varadhan_supremum(phi, rate_fn, &grid);
        // sup { -x² - (x-1)² } = sup { -2x² + 2x - 1 } at x = 0.5
        let expected = -(0.5_f64 * 0.5_f64) - (0.5_f64 - 1.0_f64).powi(2);
        assert_relative_eq!(sup, expected, epsilon = 0.05);
    }

    #[test]
    fn test_large_deviation_analysis() {
        let result = large_deviation_analysis(
            |t| (0.5 * t * t).exp(),
            2.0,
            100,
            (-5.0, 5.0),
            10000,
        );
        assert_relative_eq!(result.rate_function_value, 2.0, epsilon = 0.05);
        assert!(result.theta_optimal > 0.0);
    }

    #[test]
    fn test_contraction_principle() {
        let f = |x: f64| 2.0 * x;
        let rate_fn = |x: f64| x * x;
        let grid: Vec<f64> = (-100..=100).map(|i| i as f64 / 10.0).collect();
        let rate = contraction_principle(f, rate_fn, 2.0, &grid, 0.05);
        // inf { x² : 2x = 2 } = 1² = 1
        assert_relative_eq!(rate, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_log_mgf_normal() {
        let dist = ContinuousDistribution::Normal { mean: 0.0, variance: 1.0 };
        assert_relative_eq!(log_mgf(&dist, 1.0), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_cramer_probability() {
        let prob = cramer_probability(
            &|t| 0.5 * t * t,
            3.0,
            100,
            (-5.0, 5.0),
            10000,
        );
        // Should be very small for large deviation
        assert!(prob < 1e-50);
    }
}
