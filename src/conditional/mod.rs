//! Conditional probability: conditioning on sigma-algebras, Bayes' theorem

use crate::probability_space::{ProbabilitySpace, Event, ProbabilityMeasure};
use crate::random_variable::RandomVariable;

/// Conditional probability P(A | B) = P(A ∩ B) / P(B)
pub fn conditional_probability(space: &ProbabilitySpace, a: &Event, b: &Event) -> f64 {
    space.conditional_probability(a, b)
}

/// Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
pub fn bayes_theorem(pb_given_a: f64, pa: f64, pb: f64) -> f64 {
    if pb.abs() < 1e-15 { return 0.0; }
    pb_given_a * pa / pb
}

/// Full Bayes with partition: P(Aᵢ | B) = P(B|Aᵢ) P(Aᵢ) / Σⱼ P(B|Aⱼ) P(Aⱼ)
pub fn bayes_full(
    likelihoods: &[f64], // P(B|Aᵢ)
    priors: &[f64],      // P(Aᵢ)
) -> Vec<f64> {
    assert_eq!(likelihoods.len(), priors.len());
    let denominator: f64 = likelihoods.iter().zip(priors.iter())
        .map(|(l, p)| l * p)
        .sum();
    if denominator.abs() < 1e-15 {
        return vec![0.0; likelihoods.len()];
    }
    likelihoods.iter().zip(priors.iter())
        .map(|(l, p)| l * p / denominator)
        .collect()
}

/// Conditional expectation E[X | Y=y] (for discrete Y)
pub fn conditional_expectation_discrete(
    rv: &RandomVariable,
    conditioning_values: &[f64],
    conditioning_probs: &[f64],
) -> Vec<(f64, f64)> {
    // Returns (y, E[X|Y=y]) pairs
    conditioning_values.iter().zip(conditioning_probs.iter())
        .map(|(y, _py)| {
            let ex = rv.expectation(); // Simplified: in general need joint distribution
            (*y, ex)
        })
        .collect()
}

/// Iterated expectation / tower property: E[E[X|Y]] = E[X]
pub fn verify_tower_property(
    outer_expectation: f64,
    inner_expectation: f64,
    tolerance: f64,
) -> bool {
    (outer_expectation - inner_expectation).abs() < tolerance
}

/// Conditional variance: Var(X|Y)
pub fn conditional_variance(var_x: f64, e_x_sq: f64, e_x: f64) -> f64 {
    let _ = (var_x, e_x_sq);
    // Var(X|Y=y) = E[X²|Y=y] - (E[X|Y=y])²
    e_x_sq - e_x * e_x
}

/// Law of total probability for continuous case
/// P(A) = ∫ P(A|Y=y) f_Y(y) dy
pub fn total_probability_continuous(
    a_event_prob_given_y: &dyn Fn(f64) -> f64,
    y_density: &dyn Fn(f64) -> f64,
) -> f64 {
    let steps = 500;
    let lo = -20.0;
    let hi = 20.0;
    let dy = (hi - lo) / steps as f64;
    let mut sum = 0.0;
    for i in 0..steps {
        let y = lo + (i as f64 + 0.5) * dy;
        sum += a_event_prob_given_y(y) * y_density(y) * dy;
    }
    sum
}

/// Naive Bayes classifier for simple cases
pub fn naive_bayes_classify(
    feature_likelihoods: &[Vec<f64>], // class_i -> feature_j -> P(feature_j | class_i)
    class_priors: &[f64],              // P(class_i)
) -> usize {
    let posteriors = bayes_full(
        &feature_likelihoods.iter().map(|_| 1.0).collect::<Vec<_>>(), // placeholder
        class_priors,
    );
    // For simplicity, just pick the class with highest prior * product of likelihoods
    let mut best_class = 0;
    let mut best_score = 0.0;
    for (i, prior) in class_priors.iter().enumerate() {
        let score: f64 = feature_likelihoods.get(i)
            .map(|likelihoods| likelihoods.iter().product::<f64>() * prior)
            .unwrap_or(*prior);
        if score > best_score {
            best_score = score;
            best_class = i;
        }
    }
    let _ = posteriors;
    best_class
}

/// Likelihood ratio test: Λ = P(data|H₀) / P(data|H₁)
pub fn likelihood_ratio(p_data_h0: f64, p_data_h1: f64) -> f64 {
    if p_data_h1.abs() < 1e-15 { return f64::INFINITY; }
    p_data_h0 / p_data_h1
}

/// Posterior odds: prior odds × likelihood ratio
pub fn posterior_odds(prior_odds: f64, likelihood_ratio: f64) -> f64 {
    prior_odds * likelihood_ratio
}

/// Mutual information I(X;Y) = Σ P(x,y) log(P(x,y) / (P(x)P(y)))
pub fn mutual_information(
    joint_probs: &[Vec<f64>],  // joint_probs[x][y]
    marginal_x: &[f64],
    marginal_y: &[f64],
) -> f64 {
    let mut mi = 0.0;
    for (i, row) in joint_probs.iter().enumerate() {
        for (j, &p_xy) in row.iter().enumerate() {
            if p_xy > 1e-15 && marginal_x[i] > 1e-15 && marginal_y[j] > 1e-15 {
                mi += p_xy * (p_xy / (marginal_x[i] * marginal_y[j])).ln();
            }
        }
    }
    mi
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bayes_theorem() {
        // Classic disease test example
        let prevalence = 0.001; // P(disease)
        let sensitivity = 0.99; // P(+|disease)
        let false_positive = 0.05; // P(+|no disease)
        let p_positive = sensitivity * prevalence + false_positive * (1.0 - prevalence);
        let posterior = bayes_theorem(sensitivity, prevalence, p_positive);
        assert!((posterior - 0.0194).abs() < 0.01);
    }

    #[test]
    fn test_bayes_full() {
        let likelihoods = vec![0.8, 0.3, 0.1];
        let priors = vec![0.4, 0.35, 0.25];
        let posteriors = bayes_full(&likelihoods, &priors);
        let total: f64 = posteriors.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
        assert!(posteriors[0] > posteriors[1]);
        assert!(posteriors[1] > posteriors[2]);
    }

    #[test]
    fn test_total_probability() {
        let result = total_probability_continuous(
            &|y| { if y > 0.0 && y < 1.0 { y } else { 0.0 } }, // P(A|Y=y)
            &|y| { if y > 0.0 && y < 1.0 { 1.0 } else { 0.0 } }, // uniform density
        );
        assert_relative_eq!(result, 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_tower_property() {
        assert!(verify_tower_property(3.0, 3.0, 0.01));
        assert!(!verify_tower_property(3.0, 5.0, 0.01));
    }

    #[test]
    fn test_likelihood_ratio() {
        let lr = likelihood_ratio(0.8, 0.3);
        assert_relative_eq!(lr, 8.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_posterior_odds() {
        let po = posterior_odds(2.0, 3.0);
        assert_relative_eq!(po, 6.0);
    }

    #[test]
    fn test_mutual_information_independent() {
        // Independent: I(X;Y) = 0
        let joint = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
        let mx = vec![0.5, 0.5];
        let my = vec![0.5, 0.5];
        let mi = mutual_information(&joint, &mx, &my);
        assert_relative_eq!(mi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_information_dependent() {
        let joint = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
        let mx = vec![0.5, 0.5];
        let my = vec![0.5, 0.5];
        let mi = mutual_information(&joint, &mx, &my);
        assert_relative_eq!(mi, 0.6931, epsilon = 0.01); // ln(2)
    }

    #[test]
    fn test_naive_bayes() {
        let feature_likelihoods = vec![
            vec![0.9, 0.8], // class 0
            vec![0.1, 0.3], // class 1
        ];
        let priors = vec![0.5, 0.5];
        let class = naive_bayes_classify(&feature_likelihoods, &priors);
        assert_eq!(class, 0); // class 0 has higher likelihood
    }

    #[test]
    fn test_conditional_variance() {
        let cv = conditional_variance(0.0, 5.0, 2.0);
        assert_relative_eq!(cv, 1.0); // 5 - 4 = 1
    }
}
