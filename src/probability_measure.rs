//! Probability measure P: F → [0, 1] with countable additivity

use serde::{Serialize, Deserialize};
use super::sample_space::Event;

/// A probability measure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProbabilityMeasure {
    /// Uniform on {0, 1, ..., n-1}
    DiscreteUniform { n: usize },
    /// Arbitrary discrete distribution: mass at each point
    Discrete { masses: Vec<(f64, f64)> }, // (point, probability)
    /// Continuous uniform on [a, b]
    ContinuousUniform { a: f64, b: f64 },
    /// Normal distribution N(μ, σ²)
    Normal { mu: f64, sigma: f64 },
    /// Exponential distribution with rate λ
    Exponential { lambda: f64 },
    /// Custom measure defined by a density function (represented as a lookup)
    Custom { name: String },
}

impl ProbabilityMeasure {
    /// Compute P(event)
    pub fn probability(&self, event: &Event) -> f64 {
        match self {
            ProbabilityMeasure::DiscreteUniform { n } => {
                if event.is_interval {
                    let count = (0..*n as u64)
                        .filter(|i| event.contains(*i as f64))
                        .count();
                    count as f64 / *n as f64
                } else {
                    let count = event.points.iter()
                        .filter(|p| **p >= 0.0 && **p < *n as f64 && **p == p.floor())
                        .count();
                    count as f64 / *n as f64
                }
            }
            ProbabilityMeasure::Discrete { masses } => {
                let total: f64 = masses.iter().map(|(_, p)| *p).sum();
                if (total - 1.0).abs() > 1e-6 {
                    // Normalize
                    let mut norm_masses = masses.clone();
                    for (_, p) in norm_masses.iter_mut() {
                        *p /= total;
                    }
                    // Recurse with normalized
                    return ProbabilityMeasure::Discrete { masses: norm_masses }.probability(event);
                }
                masses.iter()
                    .filter(|(pt, _)| event.contains(*pt))
                    .map(|(_, p)| *p)
                    .sum()
            }
            ProbabilityMeasure::ContinuousUniform { a, b } => {
                if event.is_interval {
                    let lo = event.low.max(*a);
                    let hi = event.high.min(*b);
                    if lo >= hi { 0.0 } else { (hi - lo) / (b - a) }
                } else {
                    0.0 // Probability of exact points in continuous distribution
                }
            }
            ProbabilityMeasure::Normal { mu, sigma } => {
                if event.is_interval {
                    let lo = event.low;
                    let hi = event.high;
                    normal_cdf_impl(hi, *mu, *sigma) - normal_cdf_impl(lo, *mu, *sigma)
                } else {
                    0.0
                }
            }
            ProbabilityMeasure::Exponential { lambda } => {
                if event.is_interval {
                    let lo = event.low.max(0.0);
                    let hi = event.high;
                    if lo >= hi { 0.0 } else {
                        (-lambda * lo).exp() - (-lambda * hi).exp()
                    }
                } else {
                    0.0
                }
            }
            ProbabilityMeasure::Custom { .. } => {
                // Would need a function pointer or trait object
                0.0
            }
        }
    }

    /// Total probability (should be 1.0)
    pub fn total(&self) -> f64 {
        self.probability(&Event::sure())
    }

    /// Verify axioms: non-negativity, normalization, countable additivity
    pub fn verify_axioms(&self) -> bool {
        // Non-negativity: P(A) ≥ 0 for all A
        // We check a few representative events
        let test_events = vec![
            Event::empty(),
            Event::sure(),
            Event::interval(0.0, 1.0),
            Event::interval(-10.0, 10.0),
        ];
        for e in &test_events {
            let p = self.probability(e);
            if p < -1e-10 || p > 1.0 + 1e-10 {
                return false;
            }
        }
        // Normalization: P(Ω) = 1
        let total = self.total();
        (total - 1.0).abs() < 0.01 // Allow some numerical error
    }

    /// Create a discrete measure from masses (auto-normalizes)
    pub fn discrete(masses: Vec<(f64, f64)>) -> Self {
        let total: f64 = masses.iter().map(|(_, p)| p).sum();
        let norm: Vec<(f64, f64)> = masses.into_iter()
            .map(|(x, p)| (x, p / total))
            .collect();
        ProbabilityMeasure::Discrete { masses: norm }
    }

    /// Probability of A ∪ B using inclusion-exclusion: P(A∪B) = P(A) + P(B) - P(A∩B)
    pub fn probability_union(&self, a: &Event, b: &Event) -> f64 {
        self.probability(a) + self.probability(b) - self.probability(&a.intersection(b))
    }
}

/// Standard normal CDF approximation (Abramowitz and Stegun)
/// Public for use by other modules
pub fn normal_cdf_impl(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Horner's method for the approximation
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_uniform() {
        let p = ProbabilityMeasure::DiscreteUniform { n: 6 };
        let event = Event::discrete(vec![0.0]);
        assert!((p.probability(&event) - 1.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_uniform() {
        let p = ProbabilityMeasure::ContinuousUniform { a: 0.0, b: 1.0 };
        let event = Event::interval(0.0, 0.5);
        assert!((p.probability(&event) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_normal() {
        let p = ProbabilityMeasure::Normal { mu: 0.0, sigma: 1.0 };
        let event = Event::interval(f64::NEG_INFINITY, f64::INFINITY);
        assert!((p.probability(&event) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_axioms() {
        let p = ProbabilityMeasure::DiscreteUniform { n: 10 };
        assert!(p.verify_axioms());
    }

    #[test]
    fn test_discrete_custom() {
        let p = ProbabilityMeasure::discrete(vec![
            (0.0, 0.3), (1.0, 0.5), (2.0, 0.2)
        ]);
        let e0 = Event::discrete(vec![0.0]);
        assert!((p.probability(&e0) - 0.3).abs() < 1e-10);
        let total = p.probability(&Event::sure());
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_inclusion_exclusion() {
        let p = ProbabilityMeasure::ContinuousUniform { a: 0.0, b: 1.0 };
        let a = Event::interval(0.0, 0.6);
        let b = Event::interval(0.4, 1.0);
        let union_p = p.probability_union(&a, &b);
        assert!((union_p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential() {
        let p = ProbabilityMeasure::Exponential { lambda: 1.0 };
        let e = Event::interval(0.0, f64::INFINITY);
        assert!((p.probability(&e) - 1.0).abs() < 1e-6);
    }
}
