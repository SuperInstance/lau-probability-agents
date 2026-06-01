//! Probability measures, pushforward, Radon-Nikodym derivatives, absolute continuity.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A discrete probability measure over a finite sample space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteMeasure {
    /// Outcomes and their probabilities
    pub weights: HashMap<String, f64>,
}

impl DiscreteMeasure {
    /// Create a new discrete measure. Normalizes weights to sum to 1.
    pub fn new(weights: HashMap<String, f64>) -> Self {
        let total: f64 = weights.values().sum();
        let normalized = if total.abs() > 1e-15 && total.is_finite() {
            weights.into_iter().map(|(k, v)| (k, v / total)).collect()
        } else {
            weights
        };
        Self { weights: normalized }
    }

    /// Point mass (Dirac measure) at a single outcome.
    pub fn dirac(outcome: &str) -> Self {
        let mut w = HashMap::new();
        w.insert(outcome.to_string(), 1.0);
        Self { weights: w }
    }

    /// Uniform distribution over given outcomes.
    pub fn uniform(outcomes: &[&str]) -> Self {
        if outcomes.is_empty() {
            return Self { weights: HashMap::new() };
        }
        let p = 1.0 / outcomes.len() as f64;
        Self {
            weights: outcomes.iter().map(|&o| (o.to_string(), p)).collect(),
        }
    }

    /// Probability of a specific outcome.
    pub fn prob(&self, outcome: &str) -> f64 {
        *self.weights.get(outcome).unwrap_or(&0.0)
    }

    /// Total probability mass.
    pub fn total_mass(&self) -> f64 {
        self.weights.values().sum()
    }

    /// Check if this is a valid probability measure (sums to 1).
    pub fn is_valid(&self) -> bool {
        let total = self.total_mass();
        (total - 1.0).abs() < 1e-10 && self.weights.values().all(|&p| p >= 0.0)
    }

    /// Support: outcomes with positive probability.
    pub fn support(&self) -> Vec<&str> {
        self.weights
            .iter()
            .filter(|(_, &p)| p > 0.0)
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Pushforward measure through a measurable function.
    pub fn pushforward<F>(&self, f: F) -> Self
    where
        F: Fn(&str) -> String,
    {
        let mut new_weights: HashMap<String, f64> = HashMap::new();
        for (outcome, &prob) in &self.weights {
            let image = f(outcome);
            *new_weights.entry(image).or_insert(0.0) += prob;
        }
        Self { weights: new_weights }
    }

    /// Compute expectation of a function under this measure.
    pub fn expectation<F>(&self, f: F) -> f64
    where
        F: Fn(&str) -> f64,
    {
        self.weights.iter().map(|(o, &p)| p * f(o)).sum()
    }

    /// Shannon entropy.
    pub fn entropy(&self) -> f64 {
        -self.weights
            .values()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// KL divergence D_KL(self || other).
    pub fn kl_divergence(&self, other: &Self) -> f64 {
        self.weights
            .iter()
            .filter(|(_, &p)| p > 0.0)
            .map(|(o, &p)| {
                let q = other.prob(o);
                if q > 0.0 { p * (p / q).ln() } else { f64::INFINITY }
            })
            .sum()
    }
}

/// Check if measure P is absolutely continuous w.r.t. Q.
pub fn is_absolutely_continuous(p: &DiscreteMeasure, q: &DiscreteMeasure) -> bool {
    p.support().iter().all(|&o| q.prob(o) > 0.0)
}

/// Compute the Radon-Nikodym derivative dP/dQ (density ratio) for each outcome.
pub fn radon_nikodym_derivative(p: &DiscreteMeasure, q: &DiscreteMeasure) -> HashMap<String, f64> {
    p.support()
        .iter()
        .map(|&o| {
            let q_val = q.prob(o);
            let ratio = if q_val > 0.0 { p.prob(o) / q_val } else { f64::INFINITY };
            (o.to_string(), ratio)
        })
        .collect()
}

/// A continuous probability distribution represented by its density function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContinuousDistribution {
    Normal { mean: f64, variance: f64 },
    Exponential { rate: f64 },
    Uniform { a: f64, b: f64 },
    Gamma { shape: f64, rate: f64 },
}

impl ContinuousDistribution {
    /// Evaluate the probability density at a point.
    pub fn pdf(&self, x: f64) -> f64 {
        match self {
            ContinuousDistribution::Normal { mean, variance } => {
                let sigma = variance.sqrt();
                (-(x - mean).powi(2) / (2.0 * variance)).exp()
                    / (sigma * (2.0 * std::f64::consts::PI).sqrt())
            }
            ContinuousDistribution::Exponential { rate } => {
                if x >= 0.0 { rate * (-rate * x).exp() } else { 0.0 }
            }
            ContinuousDistribution::Uniform { a, b } => {
                if x >= *a && x <= *b {
                    1.0 / (b - a)
                } else {
                    0.0
                }
            }
            ContinuousDistribution::Gamma { shape, rate } => {
                if x <= 0.0 { return 0.0; }
                let k = *shape;
                let theta = 1.0 / rate;
                let ln_pdf = (k - 1.0) * x.ln() - x / theta - k.ln() - ln_gamma(k) - k * theta.ln();
                ln_pdf.exp()
            }
        }
    }

    /// Cumulative distribution function.
    pub fn cdf(&self, x: f64) -> f64 {
        match self {
            ContinuousDistribution::Normal { mean, variance } => {
                0.5 * (1.0 + erf((x - mean) / (2.0 * variance).sqrt()))
            }
            ContinuousDistribution::Exponential { rate } => {
                if x >= 0.0 { 1.0 - (-rate * x).exp() } else { 0.0 }
            }
            ContinuousDistribution::Uniform { a, b } => {
                if x < *a { 0.0 } else if x > *b { 1.0 } else { (x - a) / (b - a) }
            }
            ContinuousDistribution::Gamma { shape: _, rate: _ } => {
                // Numerical integration approximation via trapezoidal rule
                let n = 200;
                let step = x / n as f64;
                let mut sum = 0.0;
                for i in 0..n {
                    let x0 = i as f64 * step;
                    let x1 = (i + 1) as f64 * step;
                    sum += (self.pdf(x0) + self.pdf(x1)) * step / 2.0;
                }
                sum.min(1.0)
            }
        }
    }

    /// Mean of the distribution.
    pub fn mean(&self) -> f64 {
        match self {
            ContinuousDistribution::Normal { mean, .. } => *mean,
            ContinuousDistribution::Exponential { rate } => 1.0 / rate,
            ContinuousDistribution::Uniform { a, b } => (a + b) / 2.0,
            ContinuousDistribution::Gamma { shape, rate } => shape / rate,
        }
    }

    /// Variance of the distribution.
    pub fn variance(&self) -> f64 {
        match self {
            ContinuousDistribution::Normal { variance, .. } => *variance,
            ContinuousDistribution::Exponential { rate } => 1.0 / rate.powi(2),
            ContinuousDistribution::Uniform { a, b } => (b - a).powi(2) / 12.0,
            ContinuousDistribution::Gamma { shape, rate } => shape / rate.powi(2),
        }
    }
}

/// Log-gamma function (Stirling approximation).
pub fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 { return f64::NAN; }
    if x < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let mut s = c[0];
    for i in 1..9 {
        s += c[i] / (x + i as f64);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + s.ln()
}

/// Error function (Abramowitz & Stegun approximation).
pub fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Product measure of two independent discrete measures.
pub fn product_measure(p: &DiscreteMeasure, q: &DiscreteMeasure) -> DiscreteMeasure {
    let mut weights = HashMap::new();
    for (o1, &p1) in &p.weights {
        for (o2, &p2) in &q.weights {
            let key = format!("({},{})", o1, o2);
            *weights.entry(key).or_insert(0.0) += p1 * p2;
        }
    }
    DiscreteMeasure { weights }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dirac_measure() {
        let d = DiscreteMeasure::dirac("x");
        assert_eq!(d.prob("x"), 1.0);
        assert_eq!(d.prob("y"), 0.0);
        assert!(d.is_valid());
    }

    #[test]
    fn test_uniform_measure() {
        let d = DiscreteMeasure::uniform(&["a", "b", "c"]);
        assert_relative_eq!(d.prob("a"), 1.0 / 3.0, epsilon = 1e-10);
        assert!(d.is_valid());
    }

    #[test]
    fn test_normalized_measure() {
        let mut w = HashMap::new();
        w.insert("a".into(), 2.0);
        w.insert("b".into(), 3.0);
        let d = DiscreteMeasure::new(w);
        assert_relative_eq!(d.prob("a"), 0.4, epsilon = 1e-10);
        assert_relative_eq!(d.prob("b"), 0.6, epsilon = 1e-10);
        assert!(d.is_valid());
    }

    #[test]
    fn test_pushforward() {
        let d = DiscreteMeasure::uniform(&["1", "2", "3"]);
        let pf = d.pushforward(|x| if x == "1" { "even".to_string() } else { "odd".to_string() });
        assert_relative_eq!(pf.prob("even"), 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(pf.prob("odd"), 2.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy() {
        let d = DiscreteMeasure::uniform(&["a", "b"]);
        assert_relative_eq!(d.entropy(), (2.0f64).ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence() {
        let mut w1 = HashMap::new();
        w1.insert("a".into(), 0.5);
        w1.insert("b".into(), 0.5);
        let p = DiscreteMeasure { weights: w1 };

        let mut w2 = HashMap::new();
        w2.insert("a".into(), 0.25);
        w2.insert("b".into(), 0.75);
        let q = DiscreteMeasure { weights: w2 };

        let kl = p.kl_divergence(&q);
        let expected = 0.5_f64 * (0.5_f64 / 0.25_f64).ln() + 0.5_f64 * (0.5_f64 / 0.75_f64).ln();
        assert_relative_eq!(kl, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_absolute_continuity() {
        let mut w1 = HashMap::new();
        w1.insert("a".into(), 0.5);
        w1.insert("b".into(), 0.5);
        let p = DiscreteMeasure { weights: w1 };

        let mut w2 = HashMap::new();
        w2.insert("a".into(), 0.3);
        w2.insert("b".into(), 0.7);
        let q = DiscreteMeasure { weights: w2 };

        assert!(is_absolutely_continuous(&p, &q));

        let mut w3 = HashMap::new();
        w3.insert("a".into(), 1.0);
        let r = DiscreteMeasure { weights: w3 };
        assert!(!is_absolutely_continuous(&p, &r));
    }

    #[test]
    fn test_radon_nikodym() {
        let mut w1 = HashMap::new();
        w1.insert("a".into(), 0.6);
        w1.insert("b".into(), 0.4);
        let p = DiscreteMeasure { weights: w1 };

        let mut w2 = HashMap::new();
        w2.insert("a".into(), 0.3);
        w2.insert("b".into(), 0.7);
        let q = DiscreteMeasure { weights: w2 };

        let rn = radon_nikodym_derivative(&p, &q);
        assert_relative_eq!(rn["a"], 2.0, epsilon = 1e-10);
        assert_relative_eq!(rn["b"], 0.4 / 0.7, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_pdf() {
        let dist = ContinuousDistribution::Normal { mean: 0.0, variance: 1.0 };
        assert_relative_eq!(dist.pdf(0.0), 1.0 / (2.0 * std::f64::consts::PI).sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_exponential_cdf() {
        let dist = ContinuousDistribution::Exponential { rate: 1.0 };
        assert_relative_eq!(dist.cdf(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(dist.cdf(1.0), 1.0 - 1.0 / std::f64::consts::E, epsilon = 1e-10);
    }

    #[test]
    fn test_uniform_distribution() {
        let dist = ContinuousDistribution::Uniform { a: 0.0, b: 1.0 };
        assert_relative_eq!(dist.pdf(0.5), 1.0, epsilon = 1e-10);
        assert_relative_eq!(dist.cdf(0.5), 0.5, epsilon = 1e-10);
        assert_relative_eq!(dist.mean(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(dist.variance(), 1.0 / 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_product_measure() {
        let p = DiscreteMeasure::uniform(&["H", "T"]);
        let prod = product_measure(&p, &p);
        assert_relative_eq!(prod.prob("(H,H)"), 0.25, epsilon = 1e-10);
        assert!(prod.is_valid());
    }

    #[test]
    fn test_ln_gamma() {
        assert_relative_eq!(ln_gamma(1.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ln_gamma(2.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ln_gamma(5.0), (24.0f64).ln(), epsilon = 1e-6);
    }

    #[test]
    fn test_support() {
        let mut w = HashMap::new();
        w.insert("a".into(), 0.5);
        w.insert("b".into(), 0.0);
        w.insert("c".into(), 0.5);
        let d = DiscreteMeasure { weights: w };
        let s = d.support();
        assert_eq!(s.len(), 2);
    }
}
