//! Random variables: measurable functions, distributions, CDFs, PDFs

use serde::{Serialize, Deserialize};
use crate::probability_space::{ProbabilityMeasure, Event};

/// A random variable X: Ω → ℝ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RandomVariable {
    /// Constant random variable X = c
    Constant { value: f64 },
    /// Discrete: takes values xᵢ with probabilities pᵢ
    Discrete { values: Vec<f64>, probabilities: Vec<f64> },
    /// Uniform on [a, b]
    Uniform { a: f64, b: f64 },
    /// Normal N(μ, σ²)
    Normal { mu: f64, sigma: f64 },
    /// Exponential with rate λ
    Exponential { lambda: f64 },
    /// Bernoulli(p): 1 with prob p, 0 with prob 1-p
    Bernoulli { p: f64 },
    /// Binomial(n, p): sum of n independent Bernoulli(p)
    Binomial { n: usize, p: f64 },
    /// Poisson(λ)
    Poisson { lambda: f64 },
    /// Sum of two independent random variables
    Sum { a: Box<RandomVariable>, b: Box<RandomVariable> },
    /// Scalar multiple: cX
    Scaled { c: f64, x: Box<RandomVariable> },
}

impl RandomVariable {
    // ---- Factory methods ----

    pub fn constant(value: f64) -> Self { RandomVariable::Constant { value } }
    pub fn discrete(values: Vec<f64>, probabilities: Vec<f64>) -> Self {
        let total: f64 = probabilities.iter().sum();
        let probs: Vec<f64> = if (total - 1.0).abs() > 1e-10 {
            probabilities.iter().map(|p| p / total).collect()
        } else {
            probabilities
        };
        RandomVariable::Discrete { values, probabilities: probs }
    }
    pub fn uniform(a: f64, b: f64) -> Self { RandomVariable::Uniform { a, b } }
    pub fn normal(mu: f64, sigma: f64) -> Self { RandomVariable::Normal { mu, sigma } }
    pub fn exponential(lambda: f64) -> Self { RandomVariable::Exponential { lambda } }
    pub fn bernoulli(p: f64) -> Self { RandomVariable::Bernoulli { p } }
    pub fn binomial(n: usize, p: f64) -> Self { RandomVariable::Binomial { n, p } }
    pub fn poisson(lambda: f64) -> Self { RandomVariable::Poisson { lambda } }

    /// Sum of two independent random variables
    pub fn add(a: RandomVariable, b: RandomVariable) -> Self {
        RandomVariable::Sum { a: Box::new(a), b: Box::new(b) }
    }

    /// Scalar multiplication cX
    pub fn scale(c: f64, x: RandomVariable) -> Self {
        RandomVariable::Scaled { c, x: Box::new(x) }
    }

    // ---- Distribution functions ----

    /// Cumulative distribution function F(x) = P(X ≤ x)
    pub fn cdf(&self, x: f64) -> f64 {
        match self {
            RandomVariable::Constant { value } => {
                if x >= *value { 1.0 } else { 0.0 }
            }
            RandomVariable::Discrete { values, probabilities } => {
                values.iter().zip(probabilities.iter())
                    .filter(|(v, _)| *v <= &x)
                    .map(|(_, p)| *p)
                    .sum()
            }
            RandomVariable::Uniform { a, b } => {
                if x <= *a { 0.0 }
                else if x >= *b { 1.0 }
                else { (x - a) / (b - a) }
            }
            RandomVariable::Normal { mu, sigma } => {
                crate::probability_space::probability_measure::normal_cdf_impl(x, *mu, *sigma)
            }
            RandomVariable::Exponential { lambda } => {
                if x < 0.0 { 0.0 } else { 1.0 - (-lambda * x).exp() }
            }
            RandomVariable::Bernoulli { p } => {
                if x < 0.0 { 0.0 }
                else if x < 1.0 { 1.0 - p }
                else { 1.0 }
            }
            RandomVariable::Binomial { n, p } => {
                let mut cdf = 0.0;
                for k in 0..=*n {
                    if (k as f64) <= x {
                        cdf += binom_pmf(*n, k, *p);
                    }
                }
                cdf
            }
            RandomVariable::Poisson { lambda } => {
                let mut cdf = 0.0;
                let mut k = 0;
                loop {
                    let kf = k as f64;
                    if kf > x { break; }
                    cdf += poisson_pmf(*lambda, k);
                    k += 1;
                    if k > 1000 { break; } // Safety
                }
                cdf
            }
            RandomVariable::Sum { a, b } => {
                // Approximate via numerical integration
                a.convolution_cdf(b, x)
            }
            RandomVariable::Scaled { c, x: inner } => {
                // P(cX ≤ t) = P(X ≤ t/c) if c > 0
                inner.cdf(x / c)
            }
        }
    }

    /// Probability density function f(x)
    pub fn pdf(&self, x: f64) -> f64 {
        match self {
            RandomVariable::Constant { value } => {
                if (x - value).abs() < 1e-12 { f64::INFINITY } else { 0.0 }
            }
            RandomVariable::Discrete { values, probabilities } => {
                values.iter().zip(probabilities.iter())
                    .filter(|(v, _)| (**v - x).abs() < 1e-10)
                    .map(|(_, p)| *p)
                    .sum()
            }
            RandomVariable::Uniform { a, b } => {
                if x >= *a && x <= *b { 1.0 / (b - a) } else { 0.0 }
            }
            RandomVariable::Normal { mu, sigma } => {
                let z = (x - mu) / sigma;
                (1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt())) * (-0.5 * z * z).exp()
            }
            RandomVariable::Exponential { lambda } => {
                if x < 0.0 { 0.0 } else { lambda * (-lambda * x).exp() }
            }
            RandomVariable::Bernoulli { p } => {
                if (x - 0.0).abs() < 1e-10 { 1.0 - p }
                else if (x - 1.0).abs() < 1e-10 { *p }
                else { 0.0 }
            }
            RandomVariable::Binomial { n, p } => {
                binom_pmf(*n, x.round() as usize, *p)
            }
            RandomVariable::Poisson { lambda } => {
                poisson_pmf(*lambda, x.round() as usize)
            }
            RandomVariable::Sum { .. } => {
                // Numerical derivative of CDF
                let h = 1e-5;
                (self.cdf(x + h) - self.cdf(x - h)) / (2.0 * h)
            }
            RandomVariable::Scaled { c, x: inner } => {
                (1.0 / c.abs()) * inner.pdf(x / c)
            }
        }
    }

    /// Numerical convolution: P(X+Y ≤ t) via integration
    fn convolution_cdf(&self, other: &RandomVariable, t: f64) -> f64 {
        // P(X+Y ≤ t) = ∫ F_X(t-y) f_Y(y) dy
        let n_steps = 200;
        let lo = -10.0;
        let hi = t + 5.0;
        let dx = (hi - lo) / n_steps as f64;
        let mut sum = 0.0;
        for i in 0..n_steps {
            let y = lo + (i as f64 + 0.5) * dx;
            let fx = self.cdf(t - y);
            let fy = other.pdf(y);
            sum += fx * fy * dx;
        }
        sum
    }

    /// Expectation E[X]
    pub fn expectation(&self) -> f64 {
        match self {
            RandomVariable::Constant { value } => *value,
            RandomVariable::Discrete { values, probabilities } => {
                values.iter().zip(probabilities.iter())
                    .map(|(v, p)| v * p)
                    .sum()
            }
            RandomVariable::Uniform { a, b } => (a + b) / 2.0,
            RandomVariable::Normal { mu, .. } => *mu,
            RandomVariable::Exponential { lambda } => 1.0 / lambda,
            RandomVariable::Bernoulli { p } => *p,
            RandomVariable::Binomial { n, p } => (*n as f64) * p,
            RandomVariable::Poisson { lambda } => *lambda,
            RandomVariable::Sum { a, b } => a.expectation() + b.expectation(),
            RandomVariable::Scaled { c, x } => c * x.expectation(),
        }
    }

    /// Variance Var(X) = E[X²] - (E[X])²
    pub fn variance(&self) -> f64 {
        match self {
            RandomVariable::Constant { .. } => 0.0,
            RandomVariable::Discrete { values, probabilities } => {
                let mu = self.expectation();
                values.iter().zip(probabilities.iter())
                    .map(|(v, p)| (v - mu).powi(2) * p)
                    .sum()
            }
            RandomVariable::Uniform { a, b } => (b - a).powi(2) / 12.0,
            RandomVariable::Normal { sigma, .. } => sigma.powi(2),
            RandomVariable::Exponential { lambda } => 1.0 / (lambda * lambda),
            RandomVariable::Bernoulli { p } => p * (1.0 - p),
            RandomVariable::Binomial { n, p } => (*n as f64) * p * (1.0 - p),
            RandomVariable::Poisson { lambda } => *lambda,
            RandomVariable::Sum { a, b } => a.variance() + b.variance(), // Assuming independence
            RandomVariable::Scaled { c, x } => c * c * x.variance(),
        }
    }

    /// Standard deviation σ = √Var(X)
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// n-th moment E[X^n]
    pub fn moment(&self, n: u32) -> f64 {
        match self {
            RandomVariable::Constant { value } => value.powi(n as i32),
            RandomVariable::Discrete { values, probabilities } => {
                values.iter().zip(probabilities.iter())
                    .map(|(v, p)| v.powi(n as i32) * p)
                    .sum()
            }
            RandomVariable::Uniform { a, b } => {
                // E[X^n] for Uniform[a,b] = (b^(n+1) - a^(n+1)) / ((n+1)(b-a))
                ((b.powi(n as i32 + 1) - a.powi(n as i32 + 1)) /
                    ((n as f64 + 1.0) * (b - a)))
            }
            RandomVariable::Normal { mu, sigma } => {
                // Central moments of normal
                match n {
                    0 => 1.0,
                    1 => *mu,
                    2 => mu.powi(2) + sigma.powi(2),
                    3 => mu.powi(3) + 3.0 * mu * sigma.powi(2),
                    4 => mu.powi(4) + 6.0 * mu.powi(2) * sigma.powi(2) + 3.0 * sigma.powi(4),
                    _ => {
                        // Numerical integration for higher moments
                        self.numerical_moment(n)
                    }
                }
            }
            RandomVariable::Exponential { lambda } => {
                // E[X^n] = n! / λ^n
                let mut fact = 1.0;
                for i in 1..=n { fact *= i as f64; }
                fact / lambda.powi(n as i32)
            }
            RandomVariable::Bernoulli { p } => p,
            _ => self.numerical_moment(n),
        }
    }

    /// n-th central moment E[(X - μ)^n]
    pub fn central_moment(&self, n: u32) -> f64 {
        let mu = self.expectation();
        match self {
            RandomVariable::Normal { sigma, .. } => {
                match n {
                    0 => 1.0,
                    1 => 0.0,
                    2 => sigma.powi(2),
                    3 => 0.0,
                    4 => 3.0 * sigma.powi(4),
                    _ => {
                        if n % 2 == 1 { 0.0 } else {
                            let k = n / 2;
                            (1..=k).fold(sigma.powi(n as i32), |acc, i| acc * (2 * i - 1) as f64)
                        }
                    }
                }
            }
            _ => {
                // Numerical for other distributions
                self.numerical_central_moment(n, mu)
            }
        }
    }

    /// Numerical computation of E[X^n] via quadrature
    fn numerical_moment(&self, n: u32) -> f64 {
        let steps = 500;
        let lo = -20.0;
        let hi = 20.0;
        let dx = (hi - lo) / steps as f64;
        let mut sum = 0.0;
        for i in 0..steps {
            let x = lo + (i as f64 + 0.5) * dx;
            sum += x.powi(n as i32) * self.pdf(x) * dx;
        }
        sum
    }

    fn numerical_central_moment(&self, n: u32, mu: f64) -> f64 {
        let steps = 500;
        let lo = -20.0;
        let hi = 20.0;
        let dx = (hi - lo) / steps as f64;
        let mut sum = 0.0;
        for i in 0..steps {
            let x = lo + (i as f64 + 0.5) * dx;
            sum += (x - mu).powi(n as i32) * self.pdf(x) * dx;
        }
        sum
    }

    /// Moment generating function M(t) = E[e^{tX}]
    pub fn mgf(&self, t: f64) -> f64 {
        match self {
            RandomVariable::Constant { value } => (t * value).exp(),
            RandomVariable::Normal { mu, sigma } => {
                (t * mu + 0.5 * sigma.powi(2) * t * t).exp()
            }
            RandomVariable::Exponential { lambda } => {
                if t < lambda { lambda / (lambda - t) } else { f64::INFINITY }
            }
            RandomVariable::Bernoulli { p } => {
                (1.0 - p) + p * t.exp()
            }
            RandomVariable::Binomial { n, p } => {
                ((1.0 - p) + p * t.exp()).powi(*n as i32)
            }
            RandomVariable::Poisson { lambda } => {
                (lambda * (t.exp() - 1.0)).exp()
            }
            _ => {
                // Numerical: E[e^{tX}] = ∫ e^{tx} f(x) dx
                let steps = 500;
                let lo = -20.0;
                let hi = 20.0;
                let dx = (hi - lo) / steps as f64;
                let mut sum = 0.0;
                for i in 0..steps {
                    let x = lo + (i as f64 + 0.5) * dx;
                    sum += (t * x).exp() * self.pdf(x) * dx;
                }
                sum
            }
        }
    }

    /// Characteristic function φ(t) = E[e^{itX}]
    pub fn characteristic_function(&self, t: f64) -> (f64, f64) {
        // Returns (real, imag) parts
        match self {
            RandomVariable::Normal { mu, sigma } => {
                let real = (t * mu - 0.5 * sigma.powi(2) * t * t).cos();
                let imag = (t * mu - 0.5 * sigma.powi(2) * t * t).sin();
                (real, imag)
            }
            _ => {
                // Numerical
                let steps = 500;
                let lo = -20.0;
                let hi = 20.0;
                let dx = (hi - lo) / steps as f64;
                let mut real_sum = 0.0;
                let mut imag_sum = 0.0;
                for i in 0..steps {
                    let x = lo + (i as f64 + 0.5) * dx;
                    let pdf = self.pdf(x);
                    real_sum += (t * x).cos() * pdf * dx;
                    imag_sum += (t * x).sin() * pdf * dx;
                }
                (real_sum, imag_sum)
            }
        }
    }

    /// Quantile function Q(p) = inf{x : F(x) ≥ p}
    pub fn quantile(&self, p: f64) -> f64 {
        match self {
            RandomVariable::Uniform { a, b } => a + p * (b - a),
            RandomVariable::Normal { mu, sigma } => {
                // Inverse CDF approximation
                *mu + *sigma * inverse_normal_cdf(p)
            }
            RandomVariable::Exponential { lambda } => {
                -(1.0 - p).ln() / lambda
            }
            _ => {
                // Binary search
                let mut lo = -100.0;
                let mut hi = 100.0;
                for _ in 0..100 {
                    let mid = (lo + hi) / 2.0;
                    if self.cdf(mid) < p {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                (lo + hi) / 2.0
            }
        }
    }

    /// Median = Q(0.5)
    pub fn median(&self) -> f64 {
        self.quantile(0.5)
    }

    /// Skewness = E[(X-μ)³]/σ³
    pub fn skewness(&self) -> f64 {
        let mu = self.expectation();
        let sigma = self.std_dev();
        if sigma.abs() < 1e-15 { return 0.0; }
        match self {
            RandomVariable::Normal { .. } => 0.0,
            RandomVariable::Exponential { .. } => 2.0,
            RandomVariable::Bernoulli { p } => {
                (1.0 - 2.0 * p) / (p * (1.0 - p)).sqrt()
            }
            _ => self.central_moment(3) / sigma.powi(3),
        }
    }

    /// Excess kurtosis = E[(X-μ)⁴]/σ⁴ - 3
    pub fn kurtosis(&self) -> f64 {
        let mu = self.expectation();
        let sigma = self.std_dev();
        if sigma.abs() < 1e-15 { return 0.0; }
        match self {
            RandomVariable::Normal { .. } => 0.0,
            RandomVariable::Exponential { .. } => 6.0,
            RandomVariable::Bernoulli { p } => {
                (1.0 - 3.0 * p * (1.0 - p)) / (p * (1.0 - p))
            }
            _ => self.central_moment(4) / sigma.powi(4) - 3.0,
        }
    }

    /// To probability measure
    pub fn to_measure(&self) -> ProbabilityMeasure {
        match self {
            RandomVariable::Uniform { a, b } => ProbabilityMeasure::ContinuousUniform { a: *a, b: *b },
            RandomVariable::Normal { mu, sigma } => ProbabilityMeasure::Normal { mu: *mu, sigma: *sigma },
            RandomVariable::Exponential { lambda } => ProbabilityMeasure::Exponential { lambda: *lambda },
            RandomVariable::Discrete { values, probabilities } => {
                ProbabilityMeasure::Discrete {
                    masses: values.iter().zip(probabilities.iter()).map(|(v, p)| (*v, *p)).collect(),
                }
            }
            _ => ProbabilityMeasure::Custom { name: "composite".to_string() },
        }
    }
}

/// Binomial PMF
pub fn binom_pmf(n: usize, k: usize, p: f64) -> f64 {
    if k > n { return 0.0; }
    let log_binom = ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k);
    (log_binom + (k as f64) * p.ln() + ((n - k) as f64) * (1.0 - p).ln()).exp()
}

/// Poisson PMF
pub fn poisson_pmf(lambda: f64, k: usize) -> f64 {
    (k as f64 * lambda.ln() - lambda - ln_factorial(k)).exp()
}

/// ln(n!)
pub fn ln_factorial(n: usize) -> f64 {
    if n <= 1 { 0.0 } else {
        // Stirling's approximation for large n
        if n > 20 {
            let n_f = n as f64;
            (n_f * n_f.ln() - n_f + 0.5 * (2.0 * std::f64::consts::PI * n_f).ln())
        } else {
            (1..=n).fold(0.0, |acc, i| acc + (i as f64).ln())
        }
    }
}

/// Rational approximation for inverse normal CDF (Beasley-Springer-Moro)
pub fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if p == 0.5 { return 0.0; }

    let a = [
        -3.969683028665376e1, 2.209460984245205e2,
        -2.759285104469687e2, 1.383577518672690e2,
        -3.066479806614716e1, 2.506628277459239e0,
    ];
    let b = [
        -5.447609879822406e1, 1.615858368580409e2,
        -1.556989798598866e2, 6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3, -3.223964580411365e-1,
        -2.400758277161838e0, -2.549732539343734e0,
        4.374664141464968e0, 2.938163982698783e0,
    ];
    let d = [7.784695709041462e-3, 3.224671290700398e-1,
        2.445134137142996e0, 3.754408661907416e0];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    }
}

/// Public re-export of normal CDF for other modules
pub fn normal_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    crate::probability_space::probability_measure::normal_cdf_impl(x, mu, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_constant_rv() {
        let rv = RandomVariable::constant(42.0);
        assert_relative_eq!(rv.expectation(), 42.0);
        assert_relative_eq!(rv.variance(), 0.0);
        assert_relative_eq!(rv.cdf(41.0), 0.0);
        assert_relative_eq!(rv.cdf(42.0), 1.0);
    }

    #[test]
    fn test_discrete_rv() {
        let rv = RandomVariable::discrete(
            vec![1.0, 2.0, 3.0],
            vec![0.2, 0.5, 0.3],
        );
        assert_relative_eq!(rv.expectation(), 2.1);
        assert_relative_eq!(rv.cdf(1.0), 0.2);
        assert_relative_eq!(rv.cdf(2.0), 0.7);
        assert_relative_eq!(rv.cdf(3.0), 1.0);
    }

    #[test]
    fn test_uniform_rv() {
        let rv = RandomVariable::uniform(0.0, 1.0);
        assert_relative_eq!(rv.expectation(), 0.5);
        assert_relative_eq!(rv.variance(), 1.0 / 12.0, epsilon = 1e-10);
        assert_relative_eq!(rv.cdf(0.0), 0.0);
        assert_relative_eq!(rv.cdf(0.5), 0.5);
        assert_relative_eq!(rv.cdf(1.0), 1.0);
        assert_relative_eq!(rv.pdf(0.5), 1.0);
    }

    #[test]
    fn test_normal_rv() {
        let rv = RandomVariable::normal(0.0, 1.0);
        assert_relative_eq!(rv.expectation(), 0.0);
        assert_relative_eq!(rv.variance(), 1.0);
        assert!((rv.cdf(0.0) - 0.5).abs() < 0.001);
        assert!((rv.pdf(0.0) - 0.3989).abs() < 0.01);
    }

    #[test]
    fn test_exponential_rv() {
        let rv = RandomVariable::exponential(2.0);
        assert_relative_eq!(rv.expectation(), 0.5);
        assert_relative_eq!(rv.variance(), 0.25);
    }

    #[test]
    fn test_bernoulli_rv() {
        let rv = RandomVariable::bernoulli(0.3);
        assert_relative_eq!(rv.expectation(), 0.3);
        assert_relative_eq!(rv.variance(), 0.21);
        assert_eq!(rv.skewness(), (1.0 - 0.6) / (0.3 * 0.7).sqrt());
    }

    #[test]
    fn test_binomial_rv() {
        let rv = RandomVariable::binomial(10, 0.5);
        assert_relative_eq!(rv.expectation(), 5.0);
        assert_relative_eq!(rv.variance(), 2.5);
    }

    #[test]
    fn test_poisson_rv() {
        let rv = RandomVariable::poisson(3.0);
        assert_relative_eq!(rv.expectation(), 3.0);
        assert_relative_eq!(rv.variance(), 3.0);
    }

    #[test]
    fn test_sum_rv() {
        let a = RandomVariable::normal(1.0, 1.0);
        let b = RandomVariable::normal(2.0, 1.0);
        let sum = RandomVariable::add(a, b);
        assert_relative_eq!(sum.expectation(), 3.0);
        assert_relative_eq!(sum.variance(), 2.0); // Independent
    }

    #[test]
    fn test_scaled_rv() {
        let x = RandomVariable::normal(2.0, 1.0);
        let y = RandomVariable::scale(3.0, x);
        assert_relative_eq!(y.expectation(), 6.0);
        assert_relative_eq!(y.variance(), 9.0);
    }

    #[test]
    fn test_mgf_normal() {
        let rv = RandomVariable::normal(0.0, 1.0);
        // M(t) = exp(t²/2)
        let t = 0.5;
        let expected = (0.5 * t * t).exp();
        assert_relative_eq!(rv.mgf(t), expected, epsilon = 1e-6);
    }

    #[test]
    fn test_mgf_exponential() {
        let rv = RandomVariable::exponential(2.0);
        // M(t) = λ/(λ-t) for t < λ
        assert_relative_eq!(rv.mgf(0.5), 2.0 / 1.5, epsilon = 1e-6);
    }

    #[test]
    fn test_quantile_normal() {
        let rv = RandomVariable::normal(0.0, 1.0);
        assert!((rv.quantile(0.5)).abs() < 0.01);
        assert!((rv.quantile(0.975) - 1.96).abs() < 0.05);
    }

    #[test]
    fn test_quantile_exponential() {
        let rv = RandomVariable::exponential(1.0);
        // Q(0.5) = -ln(0.5) = ln(2)
        assert!((rv.quantile(0.5) - 2.0_f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_median() {
        let rv = RandomVariable::normal(5.0, 2.0);
        assert!((rv.median() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_moments_uniform() {
        let rv = RandomVariable::uniform(0.0, 1.0);
        assert_relative_eq!(rv.moment(1), 0.5, epsilon = 1e-6);
        assert_relative_eq!(rv.moment(2), 1.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mgf_bernoulli() {
        let rv = RandomVariable::bernoulli(0.5);
        // M(t) = 0.5 + 0.5*e^t
        let t = 1.0;
        assert_relative_eq!(rv.mgf(t), 0.5 + 0.5 * t.exp(), epsilon = 1e-6);
    }

    #[test]
    fn test_mgf_binomial() {
        let rv = RandomVariable::binomial(5, 0.4);
        let t = 0.3;
        let expected = (0.6 + 0.4 * t.exp()).powi(5);
        assert_relative_eq!(rv.mgf(t), expected, epsilon = 1e-4);
    }

    #[test]
    fn test_mgf_poisson() {
        let rv = RandomVariable::poisson(2.0);
        let t = 0.5;
        let expected = (2.0 * (t.exp() - 1.0)).exp();
        assert_relative_eq!(rv.mgf(t), expected, epsilon = 1e-4);
    }

    #[test]
    fn test_characteristic_function_normal() {
        let rv = RandomVariable::normal(0.0, 1.0);
        let (re, im) = rv.characteristic_function(1.0);
        // φ(t) = exp(-t²/2) for standard normal
        let expected = (-0.5_f64).exp();
        assert!((re - expected).abs() < 0.01);
        assert!(im.abs() < 0.01);
    }

    #[test]
    fn test_kurtosis_normal() {
        let rv = RandomVariable::normal(0.0, 1.0);
        assert_relative_eq!(rv.kurtosis(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_skewness_exponential() {
        let rv = RandomVariable::exponential(1.0);
        assert_relative_eq!(rv.skewness(), 2.0, epsilon = 1e-6);
    }
}
