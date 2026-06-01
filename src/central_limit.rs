//! Central Limit Theorem, Berry-Esseen bounds, Lindeberg condition.

use crate::expectation::sample_variance;
use crate::measure::erf;
use serde::{Deserialize, Serialize};

/// Standard normal CDF.
pub fn phi(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Standard normal PDF.
pub fn phi_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// CLT approximation: given sample mean from n iid observations with given mean and variance,
/// compute P(sample_mean <= x) using normal approximation.
pub fn clt_probability(sample_mean: f64, true_mean: f64, true_var: f64, n: usize) -> f64 {
    let se = (true_var / n as f64).sqrt();
    let z = (sample_mean - true_mean) / se;
    phi(z)
}

/// Compute the CLT standardized Z-scores for a sample.
pub fn standardize_samples(samples: &[f64], true_mean: f64, true_var: f64, n_per_group: usize) -> Vec<f64> {
    samples
        .chunks(n_per_group)
        .map(|chunk| {
            let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
            let se = (true_var / chunk.len() as f64).sqrt();
            (mean - true_mean) / se
        })
        .collect()
}

/// Berry-Esseen bound: sup|F_n(x) - Φ(x)| ≤ C * ρ / (σ³ * √n)
/// where ρ = E[|X - μ|³], C ≈ 0.4748 (best known constant).
pub fn berry_esseen_bound(third_abs_moment: f64, sigma: f64, n: usize) -> f64 {
    const C: f64 = 0.4748;
    if sigma <= 0.0 || n == 0 { return f64::INFINITY; }
    C * third_abs_moment / (sigma.powi(3) * (n as f64).sqrt())
}

/// Compute third absolute central moment from samples.
pub fn third_absolute_moment(samples: &[f64], mean: f64) -> f64 {
    let n = samples.len() as f64;
    samples.iter().map(|&x| (x - mean).abs().powi(3)).sum::<f64>() / n
}

/// Check the Lindeberg condition for a triangular array.
/// Given sequences of variances and third moments, verify Lindeberg's condition.
pub fn lindeberg_condition(
    variances: &[f64],
    epsilons: &[f64],
    max_individual_vars: &[f64],
) -> bool {
    let total_var: f64 = variances.iter().sum();
    if total_var <= 0.0 { return false; }

    for &eps in epsilons {
        let sum: f64 = variances
            .iter()
            .zip(max_individual_vars.iter())
            .map(|(&s2, &max_var)| {
                if max_var > eps * total_var { s2 } else { 0.0 }
            })
            .sum();
        if sum / total_var > eps {
            return false;
        }
    }
    true
}

/// Lyapunov condition (simpler sufficient condition for CLT).
/// Check if E[|X_k - μ_k|^(2+δ)] / s_n^(2+δ) → 0.
pub fn lyapunov_condition(
    central_moments: &[f64],
    total_std: f64,
    delta: f64,
) -> bool {
    if total_std <= 0.0 { return false; }
    let sum: f64 = central_moments.iter().sum();
    let ratio = sum / total_std.powf(2.0 + delta);
    ratio < 0.01 // Threshold for "approximately zero"
}

/// Delta method: if √n(X_n - θ) → N(0, σ²), then
/// √n(g(X_n) - g(θ)) → N(0, [g'(θ)]² σ²).
pub fn delta_method_variance(original_variance: f64, derivative_at_theta: f64, n: usize) -> f64 {
    derivative_at_theta.powi(2) * original_variance / n as f64
}

/// Continuity correction for CLT approximation to discrete distributions.
pub fn clt_continuity_correction(k: f64, mean: f64, variance: f64) -> f64 {
    let se = variance.sqrt();
    phi((k + 0.5 - mean) / se) - phi((k - 0.5 - mean) / se)
}

/// Confidence interval using CLT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence: f64,
}

/// Compute a confidence interval for the mean using CLT.
pub fn clt_confidence_interval(sample_mean: f64, sample_var: f64, n: usize, confidence: f64) -> ConfidenceInterval {
    let alpha = 1.0 - confidence;
    let z = phi_inv(1.0 - alpha / 2.0);
    let se = (sample_var / n as f64).sqrt();
    ConfidenceInterval {
        lower: sample_mean - z * se,
        upper: sample_mean + z * se,
        confidence,
    }
}

/// Inverse of standard normal CDF (quantile function) using rational approximation.
pub fn phi_inv(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if p == 0.5 { return 0.0; }

    // Beasley-Springer-Moro algorithm
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_phi() {
        assert!((phi(0.0) - 0.5).abs() < 1e-8);
        assert!(phi(3.0) > 0.99);
        assert!(phi(-3.0) < 0.01);
    }

    #[test]
    fn test_phi_inv_roundtrip() {
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let p = phi(x);
            let x2 = phi_inv(p);
            assert!((x - x2).abs() < 1e-4);
        }
    }

    #[test]
    fn test_clt_probability() {
        // P(sample_mean <= 0) when true_mean=0, var=1, n=100 → ~0.5
        let p = clt_probability(0.0, 0.0, 1.0, 100);
        assert!((p - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_berry_esseen_bound() {
        // For uniform(0,1): μ=0.5, σ²=1/12, E|X-μ|³ ≈ 0.0625
        let bound = berry_esseen_bound(0.0625, (1.0 / 12.0_f64).sqrt(), 100);
        assert!(bound < 0.5);
        assert!(bound > 0.0);
    }

    #[test]
    fn test_confidence_interval() {
        let ci = clt_confidence_interval(5.0, 4.0, 100, 0.95);
        let z = phi_inv(0.975);
        let expected_margin = z * (4.0_f64 / 100.0_f64).sqrt();
        assert_relative_eq!(ci.lower, 5.0 - expected_margin, epsilon = 1e-6);
        assert_relative_eq!(ci.upper, 5.0 + expected_margin, epsilon = 1e-6);
    }

    #[test]
    fn test_delta_method() {
        let var = delta_method_variance(1.0, 2.0, 100);
        assert_relative_eq!(var, 4.0 / 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_continuity_correction() {
        let p = clt_continuity_correction(5.0, 5.0, 4.0);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn test_standardize_samples() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let z = standardize_samples(&samples, 3.5, 1.0, 3);
        assert_eq!(z.len(), 2);
    }

    #[test]
    fn test_lyapunov_condition() {
        // Small central moments relative to total std → satisfied
        let moments = vec![0.001, 0.001, 0.001];
        assert!(lyapunov_condition(&moments, 10.0, 1.0));
    }

    #[test]
    fn test_phi_pdf() {
        assert_relative_eq!(phi_pdf(0.0), 1.0 / (2.0 * std::f64::consts::PI).sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_third_absolute_moment() {
        let samples = vec![-1.0, 0.0, 1.0];
        let m3 = third_absolute_moment(&samples, 0.0);
        assert_relative_eq!(m3, 2.0 / 3.0, epsilon = 1e-10);
    }
}
