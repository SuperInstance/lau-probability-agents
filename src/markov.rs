//! Markov chains, transition kernels, ergodicity, detailed balance, mixing times.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Helper to serialize DMatrix as Vec<Vec<f64>>.
mod matrix_serde {
    use nalgebra::DMatrix;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(m: &DMatrix<f64>, s: S) -> Result<S::Ok, S::Error> {
        let rows = m.nrows();
        let cols = m.ncols();
        let data: Vec<Vec<f64>> = (0..rows)
            .map(|i| (0..cols).map(|j| m[(i, j)]).collect())
            .collect();
        data.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<DMatrix<f64>, D::Error> {
        let data: Vec<Vec<f64>> = Vec::deserialize(d)?;
        let rows = data.len();
        if rows == 0 {
            return Ok(DMatrix::zeros(0, 0));
        }
        let cols = data[0].len();
        let flat: Vec<f64> = data.into_iter().flat_map(|r| r).collect();
        Ok(DMatrix::from_vec(rows, cols, flat))
    }
}

/// A finite-state Markov chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovChain {
    /// Number of states.
    pub n_states: usize,
    /// Transition matrix P[i][j] = P(X_{n+1} = j | X_n = i).
    #[serde(with = "matrix_serde")]
    pub transition_matrix: DMatrix<f64>,
    /// State labels.
    pub labels: Vec<String>,
}

impl MarkovChain {
    /// Create a new Markov chain from a transition matrix.
    pub fn new(transition_matrix: DMatrix<f64>) -> Self {
        let n = transition_matrix.nrows();
        let labels = (0..n).map(|i| format!("s{}", i)).collect();
        Self {
            n_states: n,
            transition_matrix,
            labels,
        }
    }

    /// Create with labeled states.
    pub fn with_labels(transition_matrix: DMatrix<f64>, labels: Vec<String>) -> Self {
        let n = transition_matrix.nrows();
        Self {
            n_states: n,
            transition_matrix,
            labels,
        }
    }

    /// Check if the transition matrix is valid (rows sum to 1, non-negative).
    pub fn is_valid(&self) -> bool {
        for i in 0..self.n_states {
            let row_sum: f64 = (0..self.n_states).map(|j| self.transition_matrix[(i, j)]).sum();
            if (row_sum - 1.0).abs() > 1e-8 {
                return false;
            }
            for j in 0..self.n_states {
                if self.transition_matrix[(i, j)] < -1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Compute n-step transition matrix P^n.
    pub fn n_step_transition(&self, n: usize) -> DMatrix<f64> {
        if n == 0 {
            DMatrix::identity(self.n_states, self.n_states)
        } else if n == 1 {
            self.transition_matrix.clone()
        } else {
            let mut result = self.transition_matrix.clone();
            for _ in 1..n {
                result = &result * &self.transition_matrix;
            }
            result
        }
    }

    /// Compute stationary distribution by solving πP = π.
    pub fn stationary_distribution(&self) -> Option<DMatrix<f64>> {
        // Solve (P^T - I)π = 0 with constraint Σπ = 1
        // Use power method instead
        let mut pi = DMatrix::from_element(1, self.n_states, 1.0 / self.n_states as f64);
        for _ in 0..10000 {
            let new_pi = &pi * &self.transition_matrix;
            let diff = (&new_pi - &pi).norm();
            pi = new_pi;
            if diff < 1e-12 {
                return Some(pi);
            }
        }
        // Check convergence
        let new_pi = &pi * &self.transition_matrix;
        if (&new_pi - &pi).norm() < 1e-6 {
            Some(pi)
        } else {
            None
        }
    }

    /// Check if the chain is irreducible (all states communicate).
    pub fn is_irreducible(&self) -> bool {
        // BFS from state 0
        let mut visited = vec![false; self.n_states];
        let mut queue = vec![0];
        visited[0] = true;
        while let Some(s) = queue.pop() {
            for j in 0..self.n_states {
                if self.transition_matrix[(s, j)] > 0.0 && !visited[j] {
                    visited[j] = true;
                    queue.push(j);
                }
            }
        }
        visited.iter().all(|&v| v)
    }

    /// Check if the chain is aperiodic.
    pub fn is_aperiodic(&self) -> bool {
        // Check if any state has self-loop
        for i in 0..self.n_states {
            if self.transition_matrix[(i, i)] > 0.0 {
                return true;
            }
        }
        // More thorough check would compute periods
        false
    }

    /// Check detailed balance: π(i)P(i,j) = π(j)P(j,i).
    pub fn satisfies_detailed_balance(&self) -> bool {
        if let Some(pi) = self.stationary_distribution() {
            for i in 0..self.n_states {
                for j in 0..self.n_states {
                    let lhs = pi[(0, i)] * self.transition_matrix[(i, j)];
                    let rhs = pi[(0, j)] * self.transition_matrix[(j, i)];
                    if (lhs - rhs).abs() > 1e-8 {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Estimate mixing time: smallest n such that ||P^n(i,·) - π||_TV < ε.
    pub fn mixing_time(&self, epsilon: f64, max_steps: usize) -> usize {
        if let Some(pi) = self.stationary_distribution() {
            for n in 1..=max_steps {
                let pn = self.n_step_transition(n);
                let mut max_tv = 0.0;
                for i in 0..self.n_states {
                    let tv: f64 = (0..self.n_states)
                        .map(|j| (pn[(i, j)] - pi[(0, j)]).abs())
                        .sum::<f64>()
                        / 2.0;
                    max_tv = f64::max(max_tv, tv);
                }
                if max_tv < epsilon {
                    return n;
                }
            }
        }
        max_steps
    }

    /// Simulate the Markov chain for n steps.
    pub fn simulate(&self, initial_state: usize, n_steps: usize, rng: &mut impl FnMut() -> f64) -> Vec<usize> {
        let mut states = vec![initial_state];
        let mut current = initial_state;
        for _ in 0..n_steps {
            let r = rng();
            let mut cumsum = 0.0;
            let mut next = current;
            for j in 0..self.n_states {
                cumsum += self.transition_matrix[(current, j)];
                if r < cumsum {
                    next = j;
                    break;
                }
            }
            states.push(next);
            current = next;
        }
        states
    }

    /// Compute expected hitting time from state i to state j.
    pub fn expected_hitting_time(&self, target: usize) -> Option<Vec<f64>> {
        // Solve (I - Q) h = 1 where Q is P with target row/column removed
        if target >= self.n_states { return None; }

        let n = self.n_states;
        let mut other_states: Vec<usize> = (0..n).filter(|&s| s != target).collect();
        let m = other_states.len();
        if m == 0 { return Some(vec![0.0]); }

        let mut q = DMatrix::zeros(m, m);
        let mut ones = DMatrix::zeros(m, 1);

        for (ii, &i) in other_states.iter().enumerate() {
            q[(ii, ii)] = 1.0;
            ones[(ii, 0)] = 1.0;
            for (jj, &j) in other_states.iter().enumerate() {
                q[(ii, jj)] -= self.transition_matrix[(i, j)];
            }
        }

        match q.lu().solve(&ones) {
            Some(h) => {
                let mut result = vec![0.0; n];
                for (ii, &i) in other_states.iter().enumerate() {
                    result[i] = h[(ii, 0)];
                }
                result[target] = 0.0;
                Some(result)
            }
            None => None,
        }
    }
}

/// Hidden Markov Model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HMM {
    pub markov_chain: MarkovChain,
    #[serde(with = "matrix_serde")]
    pub emission_matrix: DMatrix<f64>, // emission_matrix[(state, observation)]
    pub n_observations: usize,
}

impl HMM {
    /// Forward algorithm: compute P(observations | model).
    pub fn forward(&self, observations: &[usize]) -> f64 {
        let n = observations.len();
        let s = self.markov_chain.n_states;
        if n == 0 || s == 0 { return 0.0; }

        let mut alpha = DMatrix::zeros(s, 1);
        let o0 = observations[0].min(self.n_observations - 1);
        for i in 0..s {
            alpha[(i, 0)] = self.emission_matrix[(i, o0)] / s as f64;
        }

        for t in 1..n {
            let ot = observations[t].min(self.n_observations - 1);
            let mut new_alpha = DMatrix::zeros(s, 1);
            for j in 0..s {
                let sum: f64 = (0..s)
                    .map(|i| alpha[(i, 0)] * self.markov_chain.transition_matrix[(i, j)])
                    .sum();
                new_alpha[(j, 0)] = sum * self.emission_matrix[(j, ot)];
            }
            alpha = new_alpha;
        }

        alpha.iter().sum()
    }

    /// Viterbi algorithm: find most likely state sequence.
    pub fn viterbi(&self, observations: &[usize]) -> Vec<usize> {
        let n = observations.len();
        let s = self.markov_chain.n_states;
        if n == 0 { return vec![]; }

        let mut delta = vec![vec![0.0; s]; n];
        let mut psi = vec![vec![0; s]; n];

        // Initialize
        let o0 = observations[0].min(self.n_observations - 1);
        for i in 0..s {
            delta[0][i] = (1.0 / s as f64) * self.emission_matrix[(i, o0)];
        }

        // Recurse
        for t in 1..n {
            let ot = observations[t].min(self.n_observations - 1);
            for j in 0..s {
                let mut best_val = 0.0;
                let mut best_i = 0;
                for i in 0..s {
                    let val = delta[t - 1][i] * self.markov_chain.transition_matrix[(i, j)];
                    if val > best_val {
                        best_val = val;
                        best_i = i;
                    }
                }
                delta[t][j] = best_val * self.emission_matrix[(j, ot)];
                psi[t][j] = best_i;
            }
        }

        // Backtrace
        let mut path = vec![0; n];
        let mut best = 0;
        for i in 1..s {
            if delta[n - 1][i] > delta[n - 1][best] {
                best = i;
            }
        }
        path[n - 1] = best;
        for t in (0..n - 1).rev() {
            path[t] = psi[t + 1][path[t + 1]];
        }
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_valid_markov_chain() {
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let mc = MarkovChain::new(p);
        assert!(mc.is_valid());
    }

    #[test]
    fn test_stationary_distribution() {
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let mc = MarkovChain::new(p);
        let pi = mc.stationary_distribution().unwrap();
        // π = (4/7, 3/7) for detailed balance: 0.3*4/7 = 0.4*3/7
        assert_relative_eq!(pi[(0, 0)], 4.0 / 7.0, epsilon = 1e-6);
        assert_relative_eq!(pi[(0, 1)], 3.0 / 7.0, epsilon = 1e-6);
    }

    #[test]
    fn test_n_step_transition() {
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let mc = MarkovChain::new(p);
        let p2 = mc.n_step_transition(2);
        // P^2[0][0] = 0.7*0.7 + 0.3*0.4 = 0.61
        assert_relative_eq!(p2[(0, 0)], 0.61, epsilon = 1e-10);
    }

    #[test]
    fn test_irreducible() {
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let mc = MarkovChain::new(p);
        assert!(mc.is_irreducible());
    }

    #[test]
    fn test_detailed_balance() {
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let mc = MarkovChain::new(p);
        // π(0)*P(0,1) = 4/7 * 0.3 = 12/70
        // π(1)*P(1,0) = 3/7 * 0.4 = 12/70
        assert!(mc.satisfies_detailed_balance());
    }

    #[test]
    fn test_aperiodic() {
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let mc = MarkovChain::new(p);
        assert!(mc.is_aperiodic());
    }

    #[test]
    fn test_mixing_time() {
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let mc = MarkovChain::new(p);
        let mt = mc.mixing_time(0.01, 1000);
        assert!(mt > 0 && mt < 100);
    }

    #[test]
    fn test_simulate() {
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let mc = MarkovChain::new(p);
        let mut counter = 0;
        let path = mc.simulate(0, 100, &mut || { counter += 1; (counter as f64 * 0.618033988749895) % 1.0 });
        assert_eq!(path.len(), 101);
        assert!(path.iter().all(|&s| s < 2));
    }

    #[test]
    fn test_hitting_time() {
        let p = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let mc = MarkovChain::new(p);
        let ht = mc.expected_hitting_time(1).unwrap();
        assert_relative_eq!(ht[0], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hmm_forward() {
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let emission = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.2, 0.8]);
        let hmm = HMM {
            markov_chain: MarkovChain::new(p),
            emission_matrix: emission,
            n_observations: 2,
        };
        let prob = hmm.forward(&[0, 0]);
        assert!(prob > 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_hmm_viterbi() {
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let emission = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.2, 0.8]);
        let hmm = HMM {
            markov_chain: MarkovChain::new(p),
            emission_matrix: emission,
            n_observations: 2,
        };
        let path = hmm.viterbi(&[0, 1, 0]);
        assert_eq!(path.len(), 3);
    }
}
