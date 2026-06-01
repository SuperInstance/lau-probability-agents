//! Gaussian processes, Poisson processes, renewal theory.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Kernel type for Gaussian processes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum KernelType {
    RBF { length_scale: f64 },
    Matern32 { length_scale: f64 },
}

fn eval_kernel(kernel: &KernelType, x: f64, y: f64) -> f64 {
    match kernel {
        KernelType::RBF { length_scale } => {
            (-0.5 * ((x - y) / length_scale).powi(2)).exp()
        }
        KernelType::Matern32 { length_scale } => {
            let r = (x - y).abs() / length_scale;
            let sqrt3 = 3.0_f64.sqrt();
            (1.0 + sqrt3 * r) * (-sqrt3 * r).exp()
        }
    }
}

/// A Gaussian process defined by its mean and covariance functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianProcess {
    /// Kernel type.
    pub kernel: KernelType,
    /// Training inputs.
    pub train_x: Vec<f64>,
    /// Training outputs.
    pub train_y: Vec<f64>,
    /// Noise variance.
    pub noise_variance: f64,
}

impl GaussianProcess {
    /// Create a new GP with RBF kernel.
    pub fn new_rbf(length_scale: f64, noise_variance: f64) -> Self {
        Self {
            kernel: KernelType::RBF { length_scale },
            train_x: vec![],
            train_y: vec![],
            noise_variance,
        }
    }

    /// Create with custom kernel.
    pub fn new(kernel: KernelType, noise_variance: f64) -> Self {
        Self {
            kernel,
            train_x: vec![],
            train_y: vec![],
            noise_variance,
        }
    }

    /// Fit the GP to training data.
    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) {
        self.train_x = x;
        self.train_y = y;
    }

    /// Predict mean and variance at a new point.
    pub fn predict(&self, x_new: f64) -> (f64, f64) {
        let n = self.train_x.len();
        if n == 0 {
            return (0.0, eval_kernel(&self.kernel, x_new, x_new));
        }

        // K(X, X) + σ²I
        let mut k_xx = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                k_xx[(i, j)] = eval_kernel(&self.kernel, self.train_x[i], self.train_x[j]);
            }
            k_xx[(i, i)] += self.noise_variance;
        }

        // k(x_new, X)
        let mut k_xn = DMatrix::zeros(n, 1);
        for i in 0..n {
            k_xn[(i, 0)] = eval_kernel(&self.kernel, self.train_x[i], x_new);
        }

        // y - m(X) (zero mean)
        let mut y_centered = DMatrix::zeros(n, 1);
        for i in 0..n {
            y_centered[(i, 0)] = self.train_y[i];
        }

        // Solve K^{-1} (y - m)
        let lu = k_xx.lu();
        if let Some(alpha) = lu.solve(&y_centered) {
            let mean = (&k_xn.transpose() * &alpha)[(0, 0)];
            let k_nn = eval_kernel(&self.kernel, x_new, x_new);
            if let Some(k_inv_k) = lu.solve(&k_xn) {
                let variance = k_nn - (&k_xn.transpose() * &k_inv_k)[(0, 0)];
                (mean, variance.max(0.0))
            } else {
                (mean, k_nn)
            }
        } else {
            ((0.0), eval_kernel(&self.kernel, x_new, x_new))
        }
    }

    /// Compute the log marginal likelihood.
    pub fn log_marginal_likelihood(&self) -> f64 {
        let n = self.train_x.len();
        if n == 0 { return 0.0; }

        let mut k = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                k[(i, j)] = eval_kernel(&self.kernel, self.train_x[i], self.train_x[j]);
            }
            k[(i, i)] += self.noise_variance;
        }

        let mut y = DMatrix::zeros(n, 1);
        for i in 0..n {
            y[(i, 0)] = self.train_y[i];
        }

        let diag: Vec<f64> = k.diagonal().iter().copied().collect();
        let lu = k.lu();
        if let Some(alpha) = lu.solve(&y) {
            let yty = (&y.transpose() * &alpha)[(0, 0)];
            // Log determinant from diagonal
            let log_det = diag.iter().map(|d| d.abs().ln().max(-30.0)).sum::<f64>();
            -0.5 * yty - 0.5 * log_det - 0.5 * (n as f64) * (2.0 * std::f64::consts::PI).ln()
        } else {
            f64::NEG_INFINITY
        }
    }
}

/// A Poisson process with rate λ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonProcess {
    pub rate: f64,
}

impl PoissonProcess {
    /// Create a new Poisson process with given rate.
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }

    /// Probability of k events in interval of length t.
    pub fn probability(&self, k: usize, t: f64) -> f64 {
        let lambda = self.rate * t;
        let k_f = k as f64;
        (-lambda + k_f * lambda.ln() - crate::measure::ln_gamma(k_f + 1.0)).exp()
    }

    /// Expected number of events in interval of length t.
    pub fn expected_count(&self, t: f64) -> f64 {
        self.rate * t
    }

    /// Variance of count in interval of length t.
    pub fn variance(&self, t: f64) -> f64 {
        self.rate * t
    }

    /// Inter-arrival time distribution (exponential with rate λ).
    pub fn inter_arrival_cdf(&self, t: f64) -> f64 {
        if t >= 0.0 { 1.0 - (-self.rate * t).exp() } else { 0.0 }
    }

    /// Generate arrival times (deterministic pseudo-random simulation).
    pub fn simulate(&self, t_max: f64, rng: &mut impl FnMut() -> f64) -> Vec<f64> {
        let mut arrivals = Vec::new();
        let mut t = 0.0;
        while t < t_max {
            let u = rng();
            let interval = -((1.0 - u).ln()) / self.rate;
            t += interval;
            if t < t_max {
                arrivals.push(t);
            }
        }
        arrivals
    }

    /// Superposition of two independent Poisson processes.
    pub fn superposition(&self, other: &PoissonProcess) -> PoissonProcess {
        PoissonProcess::new(self.rate + other.rate)
    }

    /// Thinning: keep each event with probability p.
    pub fn thinning(&self, p: f64) -> PoissonProcess {
        PoissonProcess::new(self.rate * p)
    }
}

/// Renewal process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenewalProcess {
    /// Mean inter-arrival time.
    pub mean_interarrival: f64,
    /// Inter-arrival time variance.
    pub interarrival_variance: f64,
}

impl RenewalProcess {
    /// Create a new renewal process.
    pub fn new(mean: f64, variance: f64) -> Self {
        Self {
            mean_interarrival: mean,
            interarrival_variance: variance,
        }
    }

    /// Expected number of renewals by time t (approximation).
    pub fn expected_renewals(&self, t: f64) -> f64 {
        t / self.mean_interarrival
    }

    /// Elementary renewal theorem approximation: E[N(t)]/t → 1/μ.
    pub fn renewal_rate(&self) -> f64 {
        1.0 / self.mean_interarrival
    }

    /// Asymptotic variance of N(t).
    pub fn asymptotic_variance(&self, t: f64) -> f64 {
        let mu = self.mean_interarrival;
        let sigma2 = self.interarrival_variance;
        sigma2 * t / (mu.powi(3))
    }

    /// Residual life distribution mean (inspection paradox).
    /// E[residual life] = E[X²] / (2 E[X]) = (μ² + σ²) / (2μ).
    pub fn expected_residual_life(&self) -> f64 {
        let mu = self.mean_interarrival;
        let sigma2 = self.interarrival_variance;
        (mu * mu + sigma2) / (2.0 * mu)
    }

    /// Age distribution mean.
    pub fn expected_age(&self) -> f64 {
        self.expected_residual_life()
    }
}

/// Common kernel functions for Gaussian processes.
pub mod kernels {
    /// RBF (squared exponential) kernel.
    pub fn rbf(x: f64, y: f64, length_scale: f64, variance: f64) -> f64 {
        variance * (-0.5 * ((x - y) / length_scale).powi(2)).exp()
    }

    /// Matérn 3/2 kernel.
    pub fn matern32(x: f64, y: f64, length_scale: f64, variance: f64) -> f64 {
        let r = (x - y).abs() / length_scale;
        let sqrt3 = 3.0_f64.sqrt();
        variance * (1.0 + sqrt3 * r) * (-sqrt3 * r).exp()
    }

    /// Periodic kernel.
    pub fn periodic(x: f64, y: f64, length_scale: f64, period: f64, variance: f64) -> f64 {
        let d = std::f64::consts::PI * (x - y) / period;
        variance * (-2.0 * (d.sin() / length_scale).powi(2)).exp()
    }

    /// Linear kernel.
    pub fn linear(x: f64, y: f64, variance: f64, offset: f64) -> f64 {
        variance * (x - offset) * (y - offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gp_no_data() {
        let gp = GaussianProcess::new_rbf(1.0, 0.1);
        let (mean, var) = gp.predict(0.0);
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
        assert_relative_eq!(var, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gp_with_data() {
        let mut gp = GaussianProcess::new_rbf(1.0, 0.01);
        gp.fit(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.0]);
        let (mean, var) = gp.predict(1.0);
        // Should predict close to 1.0 at x=1.0
        assert!(mean > 0.8);
        // Variance should be lower than prior
        assert!(var < 1.0);
    }

    #[test]
    fn test_poisson_probability() {
        let pp = PoissonProcess::new(2.0);
        // P(N(1) = 2) for λ=2: e^{-2} * 4/2 = 2e^{-2}
        assert_relative_eq!(pp.probability(2, 1.0), 2.0 * (-2.0f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_expected() {
        let pp = PoissonProcess::new(3.0);
        assert_relative_eq!(pp.expected_count(2.0), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_superposition() {
        let p1 = PoissonProcess::new(2.0);
        let p2 = PoissonProcess::new(3.0);
        let sup = p1.superposition(&p2);
        assert_relative_eq!(sup.rate, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_thinning() {
        let pp = PoissonProcess::new(4.0);
        let thinned = pp.thinning(0.5);
        assert_relative_eq!(thinned.rate, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_renewal_process() {
        let rp = RenewalProcess::new(2.0, 1.0);
        assert_relative_eq!(rp.renewal_rate(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(rp.expected_renewals(10.0), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_renewal_residual_life() {
        // For exponential(λ): mean=1/λ, var=1/λ², residual = mean
        let rp = RenewalProcess::new(2.0, 0.25);
        let residual = rp.expected_residual_life();
        // (4 + 0.25) / 4 = 1.0625
        assert_relative_eq!(residual, 1.0625, epsilon = 1e-10);
    }

    #[test]
    fn test_rbf_kernel() {
        let k = kernels::rbf(0.0, 0.0, 1.0, 1.0);
        assert_relative_eq!(k, 1.0, epsilon = 1e-10);

        let k2 = kernels::rbf(0.0, 1.0, 1.0, 1.0);
        assert!(k2 < 1.0 && k2 > 0.0);
    }

    #[test]
    fn test_matern32_kernel() {
        let k = kernels::matern32(0.0, 0.0, 1.0, 1.0);
        assert_relative_eq!(k, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_periodic_kernel() {
        let k = kernels::periodic(0.0, 1.0, 1.0, 1.0, 1.0);
        assert!(k > 0.0 && k <= 1.0);
        // Should be periodic
        let k2 = kernels::periodic(0.0, 2.0, 1.0, 1.0, 1.0);
        assert_relative_eq!(k, k2, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_inter_arrival() {
        let pp = PoissonProcess::new(1.0);
        assert_relative_eq!(pp.inter_arrival_cdf(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(pp.inter_arrival_cdf(1.0), 1.0 - (-1.0f64).exp(), epsilon = 1e-10);
    }
}
