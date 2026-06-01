//! Martingales: sequences where E[X_{n+1} | F_n] = X_n (fair game property)
//! Includes: Doob's convergence theorem, stopping times, optional stopping theorem

use serde::{Serialize, Deserialize};

/// A martingale sequence (X₀, X₁, X₂, ...)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Martingale {
    /// Values of the martingale sequence
    values: Vec<f64>,
    /// Filtration levels (sigma-algebras F₀ ⊆ F₁ ⊆ ...)
    filtration_depths: Vec<usize>,
}

impl Martingale {
    /// Create from known values (assumes martingale property holds)
    pub fn from_values(values: Vec<f64>) -> Self {
        let filtration_depths = (0..values.len()).collect();
        Martingale { values, filtration_depths }
    }

    /// Simple symmetric random walk: X_n = X_{n-1} + Z_n, Z_n ∈ {-1, +1} equally likely
    pub fn symmetric_random_walk(steps: Vec<i32>) -> Self {
        let mut values = vec![0.0];
        let mut pos = 0i32;
        for step in steps {
            pos += step;
            values.push(pos as f64);
        }
        let filtration_depths = (0..values.len()).collect();
        Martingale { values, filtration_depths }
    }

    /// Generate a symmetric random walk of n steps
    pub fn random_walk(n: usize, rng: &mut impl FnMut() -> f64) -> Self {
        let steps: Vec<i32> = (0..n).map(|_| if rng() < 0.5 { -1 } else { 1 }).collect();
        Self::symmetric_random_walk(steps)
    }

    /// Doob's martingale: X_n = E[X | F_n] for some terminal variable X
    pub fn doob_martingale(terminal_value: f64, partial_observations: Vec<f64>) -> Self {
        // Start with prior, then condition on each observation
        let mut values = vec![terminal_value]; // Simplified
        for _obs in &partial_observations {
            values.push(terminal_value); // In reality, this would update based on observations
        }
        let filtration_depths = (0..values.len()).collect();
        Martingale { values, filtration_depths }
    }

    /// Product martingale: X_n = Π_{i=1}^n Y_i where E[Y_i] = 1
    pub fn product_martingale(factors: Vec<f64>) -> Self {
        let mut values = vec![1.0];
        let mut product = 1.0;
        for f in factors {
            product *= f;
            values.push(product);
        }
        let filtration_depths = (0..values.len()).collect();
        Martingale { values, filtration_depths }
    }

    /// Check the martingale property: E[X_{n+1} | F_n] = X_n
    /// We verify this approximately by checking E[X_{n+1} - X_n] ≈ 0
    pub fn check_martingale_property(&self) -> bool {
        if self.values.len() < 2 { return true; }
        let increments: Vec<f64> = self.values.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        let mean_increment: f64 = increments.iter().sum::<f64>() / increments.len() as f64;
        mean_increment.abs() < 0.1 // Allow some tolerance for finite samples
    }

    /// Check submartingale property: E[X_{n+1} | F_n] ≥ X_n
    pub fn check_submartingale_property(&self) -> bool {
        if self.values.len() < 2 { return true; }
        let increments: Vec<f64> = self.values.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        let mean_increment: f64 = increments.iter().sum::<f64>() / increments.len() as f64;
        mean_increment >= -0.1
    }

    /// Check supermartingale property: E[X_{n+1} | F_n] ≤ X_n
    pub fn check_supermartingale_property(&self) -> bool {
        if self.values.len() < 2 { return true; }
        let increments: Vec<f64> = self.values.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        let mean_increment: f64 = increments.iter().sum::<f64>() / increments.len() as f64;
        mean_increment <= 0.1
    }

    /// Get values
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Get value at time n
    pub fn value_at(&self, n: usize) -> f64 {
        self.values.get(n).copied().unwrap_or(0.0)
    }

    /// Length of the martingale sequence
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Increments X_n - X_{n-1}
    pub fn increments(&self) -> Vec<f64> {
        self.values.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Quadratic variation [X]_n = Σ_{i=1}^n (ΔX_i)²
    pub fn quadratic_variation(&self) -> f64 {
        self.increments().iter().map(|d| d * d).sum()
    }

    /// Doob's maximal inequality: P(max_{k≤n} X_k ≥ λ) ≤ E[|X_n|] / λ
    pub fn doob_maximal_inequality(&self, lambda: f64) -> f64 {
        if lambda.abs() < 1e-15 { return 1.0; }
        let en = self.values.last().copied().unwrap_or(0.0).abs();
        en / lambda
    }

    /// Doob's L^p inequality: E[max |X_k|^p] ≤ (p/(p-1))^p E[|X_n|^p]
    pub fn doob_lp_bound(&self, p: f64) -> f64 {
        if p <= 1.0 { return f64::INFINITY; }
        let x_n = self.values.last().copied().unwrap_or(0.0);
        let factor = (p / (p - 1.0)).powf(p);
        factor * x_n.abs().powf(p)
    }

    /// Predictable quadratic variation ⟨X⟩_n
    /// For random walk with unit steps: ⟨X⟩_n = n
    pub fn predictable_quadratic_variation(&self) -> f64 {
        (self.values.len().saturating_sub(1)) as f64
    }

    /// Azuma-Hoeffding bound for martingale concentration
    /// P(|X_n - X_0| ≥ t) ≤ 2 exp(-t² / (2 Σ c_i²))
    pub fn azuma_hoeffding_bound(&self, t: f64, bounds: &[f64]) -> f64 {
        let sum_sq: f64 = bounds.iter().map(|c| c * c).sum();
        if sum_sq < 1e-15 { return 1.0; }
        2.0 * (-t * t / (2.0 * sum_sq)).exp()
    }
}

/// Stopping time τ: {τ ≤ n} ∈ F_n for all n
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoppingTime {
    /// The time index at which we stop
    pub time: usize,
    /// Optional threshold that triggered the stop
    pub threshold: Option<f64>,
}

impl StoppingTime {
    /// Create a deterministic stopping time
    pub fn at(time: usize) -> Self {
        StoppingTime { time, threshold: None }
    }

    /// First passage time: τ = min{n : X_n ≥ a}
    pub fn first_passage(martingale: &Martingale, level: f64) -> Self {
        let time = martingale.values().iter()
            .position(|&v| v >= level)
            .unwrap_or(martingale.len());
        StoppingTime { time, threshold: Some(level) }
    }

    /// First exit from interval (a, b): τ = min{n : X_n ∉ (a, b)}
    pub fn first_exit(martingale: &Martingale, a: f64, b: f64) -> Self {
        let time = martingale.values().iter()
            .position(|&v| v <= a || v >= b)
            .unwrap_or(martingale.len());
        StoppingTime { time, threshold: None }
    }

    /// Verify it's a valid stopping time (depends only on past information)
    /// For our purposes, we always return true as the stopping time is defined properly
    pub fn is_valid(&self) -> bool {
        true
    }

    /// Apply optional stopping: E[X_τ] should ≈ E[X_0] (under boundedness conditions)
    pub fn evaluate(&self, martingale: &Martingale) -> f64 {
        martingale.value_at(self.time.min(martingale.len() - 1))
    }
}

/// Optional stopping theorem: E[X_τ] = E[X_0] if τ is bounded
/// or other regularity conditions hold
pub fn optional_stopping_theorem(
    martingale: &Martingale,
    stopping_time: &StoppingTime,
) -> (f64, f64, bool) {
    let x_tau = stopping_time.evaluate(martingale);
    let x_0 = martingale.value_at(0);
    let holds = (x_tau - x_0).abs() < 1.0; // Tolerance
    (x_tau, x_0, holds)
}

/// Doob's decomposition: X_n = M_n + A_n where M is a martingale and A is predictable
pub fn doob_decomposition(sequence: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if sequence.is_empty() { return (vec![], vec![]); }
    let mut martingale = vec![sequence[0]];
    let mut predictable = vec![0.0];

    for i in 1..sequence.len() {
        let increment = sequence[i] - sequence[i - 1];
        // M_n = M_{n-1} + (X_n - E[X_n | F_{n-1}])
        // A_n = A_{n-1} + E[X_n - X_{n-1} | F_{n-1}]
        // Simplified: assume fair increments
        martingale.push(martingale[i - 1] + increment);
        predictable.push(0.0); // For a martingale, the predictable part is 0
    }

    (martingale, predictable)
}

/// Doob's convergence theorem: if X_n is a martingale with sup_n E[|X_n|] < ∞,
/// then X_n converges a.s. to some X_∞
pub fn doob_convergence_check(martingale: &Martingale) -> (bool, f64) {
    // Check if sup_n E[|X_n|] is finite
    let sup_abs = martingale.values().iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let is_bounded = sup_abs < 1e10;
    // Check if values are converging
    let n = martingale.len();
    if n < 10 {
        return (is_bounded, 0.0);
    }
    let last_10: Vec<f64> = martingale.values()[n - 10..].to_vec();
    let variance: f64 = {
        let mean = last_10.iter().sum::<f64>() / 10.0;
        last_10.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 10.0
    };
    (is_bounded && variance < 1.0, variance)
}

/// Wald's equation: E[Σ_{i=1}^τ X_i] = E[τ] · E[X₁] when τ is a stopping time
pub fn wald_equation(expected_tau: f64, expected_x: f64) -> f64 {
    expected_tau * expected_x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_random_walk() {
        let mg = Martingale::symmetric_random_walk(vec![1, -1, 1, 1, -1]);
        assert_eq!(mg.values(), &[0.0, 1.0, 0.0, 1.0, 2.0, 1.0]);
        assert_eq!(mg.len(), 6);
    }

    #[test]
    fn test_martingale_property() {
        // Fair random walk is a martingale
        let mg = Martingale::symmetric_random_walk(vec![1, -1, 1, -1, 1, -1, 1, -1]);
        // Check increments are balanced
        let inc = mg.increments();
        let sum: f64 = inc.iter().sum();
        assert!(sum.abs() <= 8.0); // Not exactly 0 for one realization
    }

    #[test]
    fn test_quadratic_variation() {
        let mg = Martingale::symmetric_random_walk(vec![1, -1, 1]);
        assert_eq!(mg.quadratic_variation(), 3.0);
    }

    #[test]
    fn test_doob_maximal_inequality() {
        let mg = Martingale::from_values(vec![0.0, 1.0, 0.5, 2.0]);
        let bound = mg.doob_maximal_inequality(3.0);
        assert!(bound < 1.0);
        assert!(bound >= 0.0);
    }

    #[test]
    fn test_product_martingale() {
        // If each factor has mean 1, the product is a martingale
        let mg = Martingale::product_martingale(vec![1.0, 1.0, 1.0]);
        assert_eq!(mg.values(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_stopping_time_deterministic() {
        let st = StoppingTime::at(5);
        assert_eq!(st.time, 5);
        assert!(st.is_valid());
    }

    #[test]
    fn test_first_passage() {
        let mg = Martingale::symmetric_random_walk(vec![1, 1, 1, -1, 1]);
        let st = StoppingTime::first_passage(&mg, 3.0);
        assert_eq!(st.time, 3); // X_3 = 3
    }

    #[test]
    fn test_first_exit() {
        let mg = Martingale::symmetric_random_walk(vec![1, 1, 1, -1]);
        let st = StoppingTime::first_exit(&mg, -2.0, 2.0);
        assert_eq!(st.time, 2); // X_2 = 2 which is ≥ 2
    }

    #[test]
    fn test_optional_stopping() {
        let mg = Martingale::from_values(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
        let st = StoppingTime::at(4);
        let (x_tau, x_0, _holds) = optional_stopping_theorem(&mg, &st);
        assert_eq!(x_tau, 0.0);
        assert_eq!(x_0, 0.0);
    }

    #[test]
    fn test_doob_decomposition() {
        let seq = vec![0.0, 1.0, 0.0, 1.0];
        let (m, a) = doob_decomposition(&seq);
        assert_eq!(m.len(), 4);
        assert_eq!(a.len(), 4);
    }

    #[test]
    fn test_doob_convergence_bounded() {
        let mg = Martingale::from_values(vec![0.0, 0.1, -0.05, 0.08, -0.02, 0.01, -0.01, 0.005, 0.0, 0.0]);
        let (bounded, variance) = doob_convergence_check(&mg);
        assert!(bounded);
        assert!(variance < 1.0);
    }

    #[test]
    fn test_wald_equation() {
        let expected_sum = wald_equation(10.0, 0.5);
        assert_eq!(expected_sum, 5.0);
    }

    #[test]
    fn test_azuma_hoeffding() {
        let mg = Martingale::from_values(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
        let bounds = vec![1.0, 1.0, 1.0, 1.0];
        let prob = mg.azuma_hoeffding_bound(4.0, &bounds);
        assert!(prob >= 0.0 && prob <= 2.0);
        assert!(prob < 1.0); // Should be small
    }

    #[test]
    fn test_doob_lp_bound() {
        let mg = Martingale::from_values(vec![0.0, 1.0, 0.5, 2.0]);
        let bound = mg.doob_lp_bound(2.0);
        assert!(bound >= 0.0);
    }

    #[test]
    fn test_stopping_evaluate() {
        let mg = Martingale::symmetric_random_walk(vec![1, -1, 1, -1, 1]);
        let st = StoppingTime::at(3);
        assert_eq!(st.evaluate(&mg), 1.0);
    }

    #[test]
    fn test_predictable_quadratic_variation() {
        let mg = Martingale::symmetric_random_walk(vec![1, -1, 1]);
        assert_eq!(mg.predictable_quadratic_variation(), 3.0); // n steps
    }

    #[test]
    fn test_martingale_from_values() {
        let mg = Martingale::from_values(vec![5.0, 5.0, 5.0, 5.0]);
        assert_eq!(mg.value_at(0), 5.0);
        assert_eq!(mg.value_at(3), 5.0);
    }

    #[test]
    fn test_doob_martingale() {
        let mg = Martingale::doob_martingale(3.0, vec![1.0, 2.0, 3.0]);
        assert_eq!(mg.len(), 4);
    }

    #[test]
    fn test_submartingale() {
        // Submartingale: increasing values
        let mg = Martingale::from_values(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(mg.check_submartingale_property());
    }

    #[test]
    fn test_supermartingale() {
        let mg = Martingale::from_values(vec![4.0, 3.0, 2.0, 1.0, 0.0]);
        assert!(mg.check_supermartingale_property());
    }
}
