//! Expectation: Lebesgue integral, moments, moment generating functions

use crate::random_variable::RandomVariable;

/// Compute the Lebesgue integral E[g(X)] = ∫ g(x) f(x) dx via numerical quadrature
pub fn lebesgue_integral(rv: &RandomVariable, g: &dyn Fn(f64) -> f64) -> f64 {
    let steps = 1000;
    let lo = -30.0;
    let hi = 30.0;
    let dx = (hi - lo) / steps as f64;
    let mut sum = 0.0;
    for i in 0..steps {
        let x = lo + (i as f64 + 0.5) * dx;
        sum += g(x) * rv.pdf(x) * dx;
    }
    sum
}

/// Compute E[X] using the Lebesgue integral
pub fn expectation_lebesgue(rv: &RandomVariable) -> f64 {
    lebesgue_integral(rv, &|x| x)
}

/// Compute E[X^n]
pub fn moment(rv: &RandomVariable, n: u32) -> f64 {
    lebesgue_integral(rv, &|x| x.powi(n as i32))
}

/// Compute E[(X - μ)^n] (central moment)
pub fn central_moment(rv: &RandomVariable, n: u32) -> f64 {
    let mu = rv.expectation();
    lebesgue_integral(rv, &|x| (x - mu).powi(n as i32))
}

/// Moment generating function M(t) = E[e^{tX}]
pub fn moment_generating_function(rv: &RandomVariable, t: f64) -> f64 {
    lebesgue_integral(rv, &|x| (t * x).exp())
}

/// Cumulant generating function K(t) = ln M(t)
pub fn cumulant_generating_function(rv: &RandomVariable, t: f64) -> f64 {
    let m = moment_generating_function(rv, t);
    if m > 0.0 { m.ln() } else { f64::NEG_INFINITY }
}

/// n-th cumulant κ_n
/// κ₁ = μ, κ₂ = σ², κ₃ = E[(X-μ)³], etc.
pub fn cumulant(rv: &RandomVariable, n: usize) -> f64 {
    match n {
        1 => rv.expectation(),
        2 => rv.variance(),
        3 => central_moment(rv, 3),
        4 => central_moment(rv, 4) - 3.0 * rv.variance().powi(2),
        _ => {
            // Numerical derivative of cumulant generating function
            let h = 1e-4;
            // κ_n = K^(n)(0)
            // Use finite differences
            numerical_derivative_of_cgf(rv, n, 0.0, h)
        }
    }
}

/// Numerical n-th derivative of cumulant generating function at t=0
fn numerical_derivative_of_cgf(rv: &RandomVariable, n: usize, t: f64, h: f64) -> f64 {
    if n == 0 {
        cumulant_generating_function(rv, t)
    } else {
        let fwd = numerical_derivative_of_cgf(rv, n - 1, t + h, h);
        let bwd = numerical_derivative_of_cgf(rv, n - 1, t - h, h);
        (fwd - bwd) / (2.0 * h)
    }
}

/// Compute variance via law of total variance: Var(X) = E[Var(X|Y)] + Var(E[X|Y])
pub fn law_of_total_variance(
    e_var_given: f64,
    var_e_given: f64,
) -> f64 {
    e_var_given + var_e_given
}

/// Covariance Cov(X, Y) = E[XY] - E[X]E[Y]
pub fn covariance(xy_expectation: f64, ex: f64, ey: f64) -> f64 {
    xy_expectation - ex * ey
}

/// Correlation ρ = Cov(X,Y) / (σ_X σ_Y)
pub fn correlation(cov: f64, sigma_x: f64, sigma_y: f64) -> f64 {
    if sigma_x.abs() < 1e-15 || sigma_y.abs() < 1e-15 {
        return 0.0;
    }
    cov / (sigma_x * sigma_y)
}

/// Change of measure / Radon-Nikodym derivative computation
/// E_Q[X] = E_P[X * dQ/dP]
pub fn change_of_measure(
    rv: &RandomVariable,
    dQ_dP: &dyn Fn(f64) -> f64,
) -> f64 {
    lebesgue_integral(rv, &|x| x * dQ_dP(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lebesgue_expectation() {
        let rv = RandomVariable::uniform(0.0, 1.0);
        let e = expectation_lebesgue(&rv);
        assert_relative_eq!(e, 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_lebesgue_moment() {
        let rv = RandomVariable::uniform(0.0, 1.0);
        let m2 = moment(&rv, 2);
        assert_relative_eq!(m2, 1.0 / 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_central_moment_normal() {
        let rv = RandomVariable::normal(0.0, 1.0);
        let c2 = central_moment(&rv, 2);
        assert_relative_eq!(c2, 1.0, epsilon = 0.05);
        let c4 = central_moment(&rv, 4);
        assert_relative_eq!(c4, 3.0, epsilon = 0.2);
    }

    #[test]
    fn test_mgf_numerical() {
        let rv = RandomVariable::normal(0.0, 1.0);
        let t = 0.5;
        let mgf = moment_generating_function(&rv, t);
        let expected = (0.5 * t * t).exp();
        assert_relative_eq!(mgf, expected, epsilon = 0.05);
    }

    #[test]
    fn test_cumulant_basic() {
        let rv = RandomVariable::normal(0.0, 2.0);
        assert_relative_eq!(cumulant(&rv, 1), 0.0, epsilon = 0.01);
        assert_relative_eq!(cumulant(&rv, 2), 4.0, epsilon = 0.01);
    }

    #[test]
    fn test_covariance() {
        let cov = covariance(6.0, 2.0, 2.0); // E[XY] - E[X]E[Y]
        assert_relative_eq!(cov, 2.0);
    }

    #[test]
    fn test_correlation() {
        let corr = correlation(2.0, 2.0, 2.0);
        assert_relative_eq!(corr, 0.5);
    }

    #[test]
    fn test_law_of_total_variance() {
        let v = law_of_total_variance(3.0, 4.0);
        assert_relative_eq!(v, 7.0);
    }

    #[test]
    fn test_change_of_measure() {
        let rv = RandomVariable::uniform(0.0, 1.0);
        // dQ/dP = 2x (tilts toward higher values)
        let result = change_of_measure(&rv, &|x| 2.0 * x);
        // E[X * 2X] under uniform = 2 * E[X²] = 2/3
        assert_relative_eq!(result, 2.0 / 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_lebesgue_nonlinear() {
        let rv = RandomVariable::uniform(0.0, 1.0);
        let e_x2 = lebesgue_integral(&rv, &|x| x * x);
        assert_relative_eq!(e_x2, 1.0 / 3.0, epsilon = 0.01);
    }
}
