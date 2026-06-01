# lau-probability-agents

Probability theory as the foundation for agent reasoning under uncertainty — Bayesian inference, stochastic processes, large deviations, and limit theorems.

## Modules

| Module | Description |
|---|---|
| `measure` | Probability measures, pushforward, Radon-Nikodym derivatives, absolute continuity |
| `conditioning` | Bayesian conditioning, prior-posterior updates, conjugate families |
| `expectation` | Expectation operators, moments, cumulants, moment generating functions |
| `convergence` | Almost sure, in probability, Lp, distribution convergence; Borel-Cantelli lemmas |
| `central_limit` | CLT, Berry-Esseen bounds, Lindeberg condition, confidence intervals |
| `large_deviations` | Cramer's theorem, Sanov's theorem, rate functions, Varadhan's lemma |
| `markov` | Markov chains, transition kernels, ergodicity, detailed balance, mixing times, HMMs |
| `martingale` | Martingales, optional stopping, Doob's inequality, convergence theorems |
| `stochastic_process` | Gaussian processes, Poisson processes, renewal theory |
| `agent_probability` | Unified API — `AgentBayes` struct with full probabilistic reasoning |

## Usage

```rust
use lau_probability_agents::agent_probability::AgentBayes;
use std::collections::HashMap;

let mut agent = AgentBayes::new("my-agent", &["hypothesis_a", "hypothesis_b"]);

let mut likelihood = HashMap::new();
likelihood.insert("hypothesis_a".into(), 0.9);
likelihood.insert("hypothesis_b".into(), 0.1);
agent.observe(&likelihood);

assert_eq!(agent.decide(), "hypothesis_a");
```

## Dependencies

- `nalgebra` — linear algebra
- `num-complex` — complex numbers for characteristic functions
- `serde` — serialization

## License

MIT
