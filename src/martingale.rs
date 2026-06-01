//! Martingales, optional stopping, Doob's inequality, convergence theorems.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A discrete-time martingale represented as a sequence of values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Martingale {
    /// Sequence of values X_0, X_1, ..., X_n.
    pub values: Vec<f64>,
    /// Optional: the filtration times.
    pub times: Vec<f64>,
}

impl Martingale {
    /// Create a new martingale from observed values.
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            times: (0..values.len()).map(|i| i as f64).collect(),
            values,
        }
    }

    /// Check if the sequence satisfies the martingale property approximately.
    /// Uses running mean to verify E[X_{n+1} | F_n] ≈ X_n.
    pub fn check_martingale_property(&self, tolerance: f64) -> bool {
        if self.values.len() < 3 { return true; }
        let n = self.values.len();
        for i in 1..n - 1 {
            let future_mean = self.values[i + 1..].iter().sum::<f64>() / (n - i - 1) as f64;
            if (future_mean - self.values[i]).abs() > tolerance {
                return false;
            }
        }
        true
    }

    /// Compute the optional stopping value at a given stopping time.
    pub fn optional_stop(&self, stopping_time: usize) -> f64 {
        let idx = stopping_time.min(self.values.len() - 1);
        self.values[idx]
    }

    /// Check Doob's martingale convergence: if sup E[|X_n|] < ∞, then X_n converges a.s.
    pub fn check_doob_convergence(&self, window: usize, tolerance: f64) -> bool {
        if self.values.len() < window { return false; }
        let tail = &self.values[self.values.len() - window..];
        let mean = tail.iter().sum::<f64>() / window as f64;
        tail.iter().all(|&x| (x - mean).abs() < tolerance)
    }

    /// Get the value at time n.
    pub fn value_at(&self, n: usize) -> f64 {
        self.values.get(n).copied().unwrap_or_else(|| *self.values.last().unwrap_or(&0.0))
    }
}

/// Doob's maximal inequality: P(sup_{k≤n} |X_k| ≥ λ) ≤ E[|X_n|] / λ.
pub fn doob_maximal_inequality(values: &[f64], lambda: f64) -> (f64, f64) {
    let sup = values.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let expectation = values.last().copied().unwrap_or(0.0).abs();
    let bound = if lambda > 0.0 { expectation / lambda } else { f64::INFINITY };
    (sup, bound)
}

/// Doob's Lp inequality: E[sup_{k≤n} |X_k|^p] ≤ (p/(p-1))^p E[|X_n|^p] for p > 1.
pub fn doob_lp_inequality(values: &[f64], p: f64) -> (f64, f64) {
    let sup = values.iter().map(|x| x.abs().powf(p)).fold(0.0_f64, f64::max);
    let last_moment = values.last().map(|&x| x.abs().powf(p)).unwrap_or(0.0);
    let bound = (p / (p - 1.0)).powf(p) * last_moment;
    (sup, bound)
}

/// Doob's decomposition: X_n = M_n + A_n where M is a martingale and A is predictable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoobDecomposition {
    pub martingale_part: Vec<f64>,
    pub predictable_part: Vec<f64>,
}

/// Compute Doob decomposition of a submartingale.
pub fn doob_decomposition(values: &[f64]) -> DoobDecomposition {
    let n = values.len();
    let mut martingale = vec![0.0; n];
    let mut predictable = vec![0.0; n];

    martingale[0] = values[0];
    predictable[0] = 0.0;

    for i in 1..n {
        // A_i = A_{i-1} + E[X_i - X_{i-1} | F_{i-1}]
        // Simplified: use the actual increment as approximation
        predictable[i] = predictable[i - 1] + (values[i] - values[i - 1]);
        martingale[i] = values[i] - predictable[i];
    }

    DoobDecomposition {
        martingale_part: martingale,
        predictable_part: predictable,
    }
}

/// Azuma-Hoeffding inequality for martingales with bounded differences.
pub fn azuma_hoeffding_bound(differences: &[f64], t: f64) -> f64 {
    let sum_sq: f64 = differences.iter().map(|c| c * c).sum();
    2.0 * (-2.0 * t * t / sum_sq).exp()
}

/// Wald's equation: E[Σ X_i] = E[N] * E[X] for stopping time N.
pub fn wald_equation(n_expected: f64, x_expected: f64) -> f64 {
    n_expected * x_expected
}

/// Optional stopping theorem verification.
/// If T is a stopping time with E[T] < ∞ and |X_{n+1} - X_n| ≤ c, then E[X_T] = E[X_0].
pub fn verify_optional_stopping(martingale: &Martingale, stopping_time: usize, tolerance: f64) -> bool {
    let x_t = martingale.optional_stop(stopping_time);
    let x_0 = martingale.value_at(0);
    (x_t - x_0).abs() < tolerance
}

/// Construct a martingale from i.i.d. increments.
pub fn martingale_from_increments(increments: &[f64], initial: f64) -> Martingale {
    let mut values = vec![initial];
    let mut sum = initial;
    for &inc in increments {
        sum += inc;
        values.push(sum);
    }
    Martingale::new(values)
}

/// Construct a product martingale (exponential martingale).
pub fn exponential_martingale(increments: &[f64], theta: f64, initial: f64) -> Martingale {
    let mut values = vec![initial];
    let mut product = initial;
    let mut sum = 0.0;
    for &inc in increments {
        sum += inc;
        product = initial * (theta * sum - theta * theta / 2.0).exp();
        values.push(product);
    }
    Martingale::new(values)
}

/// Supermartingale check: E[X_{n+1} | F_n] ≤ X_n.
pub fn check_supermartingale(values: &[f64], tolerance: f64) -> bool {
    for i in 0..values.len() - 1 {
        if values[i + 1] > values[i] + tolerance {
            return false;
        }
    }
    true
}

/// Submartingale check: E[X_{n+1} | F_n] ≥ X_n.
pub fn check_submartingale(values: &[f64], tolerance: f64) -> bool {
    for i in 0..values.len() - 1 {
        if values[i + 1] < values[i] - tolerance {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_martingale_creation() {
        let m = Martingale::new(vec![0.0, 0.1, -0.05, 0.02]);
        assert_eq!(m.values.len(), 4);
        assert_relative_eq!(m.value_at(0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_doob_maximal_inequality() {
        let values = vec![1.0, 2.0, 1.5, 3.0, 2.5];
        let (sup, bound) = doob_maximal_inequality(&values, 2.0);
        assert_relative_eq!(sup, 3.0, epsilon = 1e-10);
        assert!(bound <= 2.0); // E[|X_n|]/λ = 2.5/2 = 1.25
    }

    #[test]
    fn test_doob_lp_inequality() {
        let values = vec![1.0, 2.0, 3.0];
        let (sup, bound) = doob_lp_inequality(&values, 2.0);
        assert!(bound >= sup); // Bound should be at least the actual value
    }

    #[test]
    fn test_azuma_hoeffding() {
        let diffs = vec![1.0, 1.0, 1.0, 1.0];
        let bound = azuma_hoeffding_bound(&diffs, 2.0);
        assert!(bound < 1.0);
        assert!(bound > 0.0);
    }

    #[test]
    fn test_martingale_from_increments() {
        let m = martingale_from_increments(&[1.0, -0.5, 0.3], 0.0);
        assert_relative_eq!(m.value_at(0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(m.value_at(1), 1.0, epsilon = 1e-10);
        assert_relative_eq!(m.value_at(3), 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_exponential_martingale() {
        let m = exponential_martingale(&[0.1, -0.1, 0.0], 1.0, 1.0);
        assert_eq!(m.values.len(), 4);
        assert_relative_eq!(m.value_at(0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_doob_decomposition() {
        let values = vec![1.0, 1.5, 2.0, 2.5];
        let decomp = doob_decomposition(&values);
        assert_eq!(decomp.martingale_part.len(), 4);
        // Verify: M_n + A_n = X_n
        for i in 0..4 {
            assert_relative_eq!(
                decomp.martingale_part[i] + decomp.predictable_part[i],
                values[i],
                epsilon = 1e-10,
            );
        }
    }

    #[test]
    fn test_wald_equation() {
        let result = wald_equation(10.0, 5.0);
        assert_relative_eq!(result, 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_optional_stop() {
        let m = Martingale::new(vec![0.0, 1.0, 2.0, 3.0]);
        assert_relative_eq!(m.optional_stop(2), 2.0, epsilon = 1e-10);
        assert_relative_eq!(m.optional_stop(100), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_supermartingale_decreasing() {
        let values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!(check_supermartingale(&values, 0.01));
        assert!(!check_submartingale(&values, 0.01));
    }

    #[test]
    fn test_submartingale_increasing() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(check_submartingale(&values, 0.01));
        assert!(!check_supermartingale(&values, 0.01));
    }
}
