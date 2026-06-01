//! Probability space (Ω, F, P) — the foundational triple

use serde::{Serialize, Deserialize};
use super::sample_space::{SampleSpace, Event};
use super::sigma_algebra::SigmaAlgebra;
use super::probability_measure::ProbabilityMeasure;

/// A complete probability space (Ω, F, P)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilitySpace {
    /// Sample space Ω
    pub omega: SampleSpace,
    /// Sigma-algebra F of measurable events
    pub sigma_algebra: SigmaAlgebra,
    /// Probability measure P
    pub measure: ProbabilityMeasure,
}

impl ProbabilitySpace {
    /// Construct a probability space from its three components
    pub fn new(omega: SampleSpace, sigma_algebra: SigmaAlgebra, measure: ProbabilityMeasure) -> Self {
        ProbabilitySpace { omega, sigma_algebra, measure }
    }

    /// Standard uniform probability space on [0, 1] with Borel sigma-algebra
    pub fn uniform_unit() -> Self {
        ProbabilitySpace {
            omega: SampleSpace::continuous(0.0, 1.0),
            sigma_algebra: SigmaAlgebra::borel(),
            measure: ProbabilityMeasure::ContinuousUniform { a: 0.0, b: 1.0 },
        }
    }

    /// Standard normal space (ℝ, B(ℝ), N(0,1))
    pub fn standard_normal() -> Self {
        ProbabilitySpace {
            omega: SampleSpace::real_line(),
            sigma_algebra: SigmaAlgebra::borel(),
            measure: ProbabilityMeasure::Normal { mu: 0.0, sigma: 1.0 },
        }
    }

    /// Fair coin flip space ({H, T}, 2^Ω, Uniform)
    pub fn coin_flip() -> Self {
        ProbabilitySpace {
            omega: SampleSpace::finite(vec![0.0, 1.0]),
            sigma_algebra: SigmaAlgebra::power_set(2),
            measure: ProbabilityMeasure::DiscreteUniform { n: 2 },
        }
    }

    /// Fair n-sided die
    pub fn fair_die(n: usize) -> Self {
        let outcomes: Vec<f64> = (0..n).map(|i| i as f64).collect();
        ProbabilitySpace {
            omega: SampleSpace::finite(outcomes),
            sigma_algebra: SigmaAlgebra::power_set(n),
            measure: ProbabilityMeasure::DiscreteUniform { n },
        }
    }

    /// Compute P(A) for event A
    pub fn probability(&self, event: &Event) -> f64 {
        self.measure.probability(event)
    }

    /// Compute P(A | B) = P(A∩B) / P(B)
    pub fn conditional_probability(&self, a: &Event, b: &Event) -> f64 {
        let pb = self.probability(b);
        if pb.abs() < 1e-15 {
            return 0.0; // Undefined, but return 0 for safety
        }
        let pab = self.probability(&a.intersection(b));
        pab / pb
    }

    /// Check if A and B are independent: P(A∩B) = P(A)P(B)
    pub fn independent(&self, a: &Event, b: &Event) -> bool {
        let pab = self.probability(&a.intersection(b));
        let pa = self.probability(a);
        let pb = self.probability(b);
        (pab - pa * pb).abs() < 1e-10
    }

    /// Law of total probability: P(A) = Σ P(A|Bᵢ)P(Bᵢ)
    pub fn total_probability(&self, a: &Event, partition: &[Event]) -> f64 {
        partition.iter()
            .map(|bi| {
                let pbi = self.probability(bi);
                if pbi > 0.0 {
                    self.conditional_probability(a, bi) * pbi
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Verify this is a valid probability space
    pub fn is_valid(&self) -> bool {
        self.measure.verify_axioms()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_unit() {
        let ps = ProbabilitySpace::uniform_unit();
        let e = Event::interval(0.0, 0.5);
        assert!((ps.probability(&e) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coin_flip() {
        let ps = ProbabilitySpace::coin_flip();
        let heads = Event::discrete(vec![0.0]);
        assert!((ps.probability(&heads) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_conditional() {
        let ps = ProbabilitySpace::uniform_unit();
        let a = Event::interval(0.0, 0.6);
        let b = Event::interval(0.0, 0.3);
        // P(A|B) = P(A∩B)/P(B) = P([0,0.3])/P([0,0.3]) = 1.0
        let cond = ps.conditional_probability(&a, &b);
        assert!((cond - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_independence() {
        let ps = ProbabilitySpace::uniform_unit();
        let a = Event::interval(0.0, 0.5);
        let b = Event::interval(0.25, 0.75);
        assert!(!ps.independent(&a, &b));
        // Independent events on uniform: [0,0.5] and [0,0.5] ∪ ... nope
        // Actually on uniform, no two nontrivial intervals are independent
    }

    #[test]
    fn test_fair_die() {
        let ps = ProbabilitySpace::fair_die(6);
        let one = Event::discrete(vec![0.0]);
        assert!((ps.probability(&one) - 1.0 / 6.0).abs() < 1e-10);
        assert!(ps.is_valid());
    }

    #[test]
    fn test_standard_normal() {
        let ps = ProbabilitySpace::standard_normal();
        let e = Event::interval(-1.96, 1.96);
        let p = ps.probability(&e);
        assert!((p - 0.95).abs() < 0.01);
    }
}
