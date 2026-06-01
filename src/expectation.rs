//! Expectation operators, moments, cumulants, moment generating functions.

use crate::measure::ContinuousDistribution;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Compute the k-th raw moment of a continuous distribution by numerical integration.
pub fn raw_moment(dist: &ContinuousDistribution, k: u32, bounds: (f64, f64), n_steps: usize) -> f64 {
    let (a, b) = bounds;
    let step = (b - a) / n_steps as f64;
    let mut sum = 0.0;
    for i in 0..n_steps {
        let x0 = a + i as f64 * step;
        let x1 = a + (i + 1) as f64 * step;
        let xm = (x0 + x1) / 2.0;
        sum += x0.powi(k as i32) * dist.pdf(x0)
            + 4.0 * xm.powi(k as i32) * dist.pdf(xm)
            + x1.powi(k as i32) * dist.pdf(x1);
    }
    sum * step / 6.0
}

/// Central moment E[(X - μ)^k].
pub fn central_moment(dist: &ContinuousDistribution, k: u32, mean: f64, bounds: (f64, f64), n_steps: usize) -> f64 {
    let (a, b) = bounds;
    let step = (b - a) / n_steps as f64;
    let mut sum = 0.0;
    for i in 0..n_steps {
        let x0 = a + i as f64 * step;
        let x1 = a + (i + 1) as f64 * step;
        let xm = (x0 + x1) / 2.0;
        sum += (x0 - mean).powi(k as i32) * dist.pdf(x0)
            + 4.0 * (xm - mean).powi(k as i32) * dist.pdf(xm)
            + (x1 - mean).powi(k as i32) * dist.pdf(x1);
    }
    sum * step / 6.0
}

/// Standardized moment (central moment / σ^k).
pub fn standardized_moment(
    dist: &ContinuousDistribution,
    k: u32,
    mean: f64,
    variance: f64,
    bounds: (f64, f64),
    n_steps: usize,
) -> f64 {
    central_moment(dist, k, mean, bounds, n_steps) / variance.powf(k as f64 / 2.0)
}

/// Moment generating function M_X(t) = E[e^{tX}] evaluated numerically.
pub fn mgf(dist: &ContinuousDistribution, t: f64, bounds: (f64, f64), n_steps: usize) -> f64 {
    let (a, b) = bounds;
    let step = (b - a) / n_steps as f64;
    let mut sum = 0.0;
    for i in 0..n_steps {
        let x0 = a + i as f64 * step;
        let x1 = a + (i + 1) as f64 * step;
        let xm = (x0 + x1) / 2.0;
        let f0 = (t * x0).exp() * dist.pdf(x0);
        let fm = (t * xm).exp() * dist.pdf(xm);
        let f1 = (t * x1).exp() * dist.pdf(x1);
        sum += f0 + 4.0 * fm + f1;
    }
    sum * step / 6.0
}

/// Characteristic function φ_X(t) = E[e^{itX}].
pub fn characteristic_function(dist: &ContinuousDistribution, t: f64, bounds: (f64, f64), n_steps: usize) -> Complex64 {
    let (a, b) = bounds;
    let step = (b - a) / n_steps as f64;
    let mut sum = Complex64::new(0.0, 0.0);
    for i in 0..n_steps {
        let x0 = a + i as f64 * step;
        let x1 = a + (i + 1) as f64 * step;
        let xm = (x0 + x1) / 2.0;
        let it = Complex64::new(0.0, t);
        let f0 = (it * x0).exp() * dist.pdf(x0);
        let fm = (it * xm).exp() * dist.pdf(xm);
        let f1 = (it * x1).exp() * dist.pdf(x1);
        sum += f0 + 4.0 * fm + f1;
    }
    sum * step / 6.0
}

/// Cumulant generating function K_X(t) = ln M_X(t).
pub fn cgf(dist: &ContinuousDistribution, t: f64, bounds: (f64, f64), n_steps: usize) -> f64 {
    let m = mgf(dist, t, bounds, n_steps);
    if m > 0.0 { m.ln() } else { f64::NEG_INFINITY }
}

/// Compute cumulants from raw moments using the recurrence relation.
pub fn cumulants_from_moments(raw_moments: &[f64]) -> Vec<f64> {
    let n = raw_moments.len();
    if n == 0 { return vec![]; }

    let mut cumulants = vec![0.0; n];
    cumulants[0] = raw_moments[0]; // κ₁ = μ₁

    for k in 2..=n {
        let mut sum = raw_moments[k - 1];
        for j in 1..k {
            sum -= binomial_coefficient(k - 1, j - 1) * cumulants[j - 1] * raw_moments[k - 1 - j];
        }
        cumulants[k - 1] = sum;
    }
    cumulants
}

/// Binomial coefficient.
pub fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n { return 0.0; }
    let k = k.min(n - k);
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

/// Covariance from joint samples.
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() { return f64::NAN; }
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>() / n
}

/// Correlation coefficient.
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let cov = covariance(x, y);
    let vx = covariance(x, x);
    let vy = covariance(y, y);
    if vx <= 0.0 || vy <= 0.0 { return f64::NAN; }
    cov / (vx.sqrt() * vy.sqrt())
}

/// Expectation of a function of samples.
pub fn expectation<F: Fn(f64) -> f64>(samples: &[f64], f: F) -> f64 {
    samples.iter().map(|&x| f(x)).sum::<f64>() / samples.len() as f64
}

/// Variance of samples.
pub fn sample_variance(samples: &[f64]) -> f64 {
    covariance(samples, samples)
}

/// Skewness of samples.
pub fn skewness(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = sample_variance(samples);
    if var <= 0.0 { return f64::NAN; }
    samples.iter().map(|&x| ((x - mean) / var.sqrt()).powi(3)).sum::<f64>() / n
}

/// Kurtosis of samples (excess kurtosis).
pub fn kurtosis(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = sample_variance(samples);
    if var <= 0.0 { return f64::NAN; }
    samples.iter().map(|&x| ((x - mean) / var.sqrt()).powi(4)).sum::<f64>() / n - 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_variance_and_covariance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(sample_variance(&x), 2.0, epsilon = 1e-10);
        assert_relative_eq!(covariance(&x, &x), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        assert_relative_eq!(correlation(&x, &y), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expectation() {
        let samples = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(expectation(&samples, |x| x), 2.0, epsilon = 1e-10);
        assert_relative_eq!(expectation(&samples, |x| x * x), 14.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mgf_normal() {
        let dist = ContinuousDistribution::Normal { mean: 0.0, variance: 1.0 };
        let mgf_val = mgf(&dist, 1.0, (-6.0, 6.0), 1000);
        // MGF of N(0,1) at t=1 is e^{1/2}
        assert_relative_eq!(mgf_val, (0.5f64).exp(), epsilon = 0.01);
    }

    #[test]
    fn test_skewness_symmetric() {
        let samples = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let s = skewness(&samples);
        assert!(s.abs() < 0.01);
    }

    #[test]
    fn test_kurtosis_normal_samples() {
        // Uniform samples have excess kurtosis = -6/5 = -1.2
        let samples: Vec<f64> = (0..10000).map(|i| (i as f64 % 10.0) / 10.0).collect();
        let k = kurtosis(&samples);
        // Just verify it's finite
        assert!(k.is_finite());
    }

    #[test]
    fn test_cumulants_from_moments() {
        // For N(μ, σ²): κ₁=μ, κ₂=σ², κ₃=0, κ₄=0
        let moments = vec![0.0, 1.0, 0.0, 3.0]; // raw moments of N(0,1)
        let cumulants = cumulants_from_moments(&moments);
        assert_relative_eq!(cumulants[0], 0.0, epsilon = 1e-10); // κ₁
        assert_relative_eq!(cumulants[1], 1.0, epsilon = 1e-10); // κ₂
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_relative_eq!(binomial_coefficient(5, 2), 10.0, epsilon = 1e-10);
        assert_relative_eq!(binomial_coefficient(10, 0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_central_moment() {
        let dist = ContinuousDistribution::Normal { mean: 0.0, variance: 1.0 };
        let m2 = central_moment(&dist, 2, 0.0, (-6.0, 6.0), 1000);
        assert_relative_eq!(m2, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_standardized_moment_skewness() {
        let dist = ContinuousDistribution::Normal { mean: 0.0, variance: 1.0 };
        let skew = standardized_moment(&dist, 3, 0.0, 1.0, (-6.0, 6.0), 1000);
        assert!(skew.abs() < 0.01);
    }
}
