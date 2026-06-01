# lau-probability-agents

> Measure-theoretic probability for agents: sigma-algebras, random variables, martingales, and Bayesian inference.

## What This Does

This crate provides a rigorous probability theory framework built from measure-theoretic foundations. It constructs probability spaces (Ω, F, P) with sigma-algebras, defines random variables as measurable functions, computes expectations via Lebesgue integration, handles conditioning and Bayes' theorem, and implements martingale theory including Doob's theorems, stopping times, and the optional stopping theorem.

Use this when you need mathematically grounded probability — not just sampling and histograms, but the actual measure theory that underpins modern probability, statistics, and stochastic processes.

## The Key Idea

Probability isn't just about dice and coin flips — it's about **measures on sigma-algebras**. A probability space (Ω, F, P) has three layers: a sample space Ω (what could happen), a sigma-algebra F (what you can measure), and a measure P (how likely things are). Random variables are measurable functions X: Ω → ℝ, and expectation is the Lebesgue integral. This crate builds probability from these foundations, which gives you correctness guarantees that ad-hoc approaches can't provide.

## Install

```bash
cargo add lau-probability-agents
```

## Quick Start

```rust
use lau_probability_agents::probability_space::*;

fn main() {
    // Standard uniform space on [0,1]
    let space = ProbabilitySpace::uniform_unit();
    let event = Event::interval(0.0, 0.5);
    println!("P([0, 0.5]) = {:.4}", space.probability(&event)); // 0.5

    // Fair coin flip
    let coin = ProbabilitySpace::coin_flip();
    let heads = Event::discrete(vec![0.0]);
    println!("P(heads) = {:.4}", coin.probability(&heads)); // 0.5

    // Fair 6-sided die
    let die = ProbabilitySpace::fair_die(6);
    let three_or_less = Event::discrete(vec![0.0, 1.0, 2.0]);
    println!("P(≤3) = {:.4}", die.probability(&three_or_less)); // 0.5

    // Check independence
    let a = Event::interval(0.0, 0.5);
    let b = Event::interval(0.25, 0.75);
    println!("Independent? {}", space.independent(&a, &b)); // false
}
```

## API Reference

### Sample Space & Events

#### `SampleSpace`
The set of all possible outcomes Ω.

```rust
let finite = SampleSpace::finite(vec![0.0, 1.0, 2.0]);
let continuous = SampleSpace::continuous(0.0, 1.0);
let reals = SampleSpace::real_line();
let card = finite.cardinality(); // Some(3)
```

#### `Event`
A subset of the sample space — either discrete points or a continuous interval.

```rust
// Discrete event
let discrete = Event::discrete(vec![1.0, 2.0, 3.0]);

// Interval event [a, b]
let interval = Event::interval(0.0, 1.0);

// Special events
let empty = Event::empty();   // ∅
let sure = Event::sure();     // Ω = (-∞, +∞)

// Set operations
let union = a.union(&b);
let intersection = a.intersection(&b);
let complement = a.complement();

// Queries
let contains = interval.contains(0.5); // true
let is_empty = empty.is_empty();       // true
```

### Sigma-Algebra

#### `SigmaAlgebra`
A collection of events closed under complement and countable unions.

```rust
// Trivial sigma-algebra {∅, Ω}
let trivial = SigmaAlgebra::trivial();

// Power set of {0, 1, ..., n-1}
let power = SigmaAlgebra::power_set(3);

// Borel sigma-algebra on ℝ
let borel = SigmaAlgebra::borel();
borel.contains(&Event::interval(0.0, 1.0)); // always true for Borel

// Generated sigma-algebra from a set of events
let sa = SigmaAlgebra::from_events(vec![Event::interval(0.0, 1.0)]);
let closed = sa.closure(); // closes under complements and unions

// Check measurability
sa.contains(&some_event);
```

### Probability Measure

#### `ProbabilityMeasure`
A measure P: F → [0, 1] satisfying Kolmogorov's axioms.

```rust
// Standard distributions
let uniform = ProbabilityMeasure::ContinuousUniform { a: 0.0, b: 1.0 };
let normal = ProbabilityMeasure::Normal { mu: 0.0, sigma: 1.0 };
let discrete = ProbabilityMeasure::DiscreteUniform { n: 6 };
let exp = ProbabilityMeasure::Exponential { lambda: 1.0 };

// Custom discrete
let custom = ProbabilityMeasure::discrete(vec![
    (0.0, 0.3), (1.0, 0.5), (2.0, 0.2)
]);

// Compute P(A)
let p = normal.probability(&Event::interval(-1.96, 1.96)); // ≈ 0.95

// Inclusion-exclusion: P(A ∪ B)
let p_union = uniform.probability_union(&a, &b);

// Verify axioms
assert!(normal.verify_axioms());
```

### Probability Space

#### `ProbabilitySpace`
The complete triple (Ω, F, P).

```rust
// Standard spaces
let unit = ProbabilitySpace::uniform_unit();
let normal = ProbabilitySpace::standard_normal();
let coin = ProbabilitySpace::coin_flip();
let die = ProbabilitySpace::fair_die(6);

// Core operations
let p = space.probability(&event);
let cond = space.conditional_probability(&a, &b);  // P(A|B)
let indep = space.independent(&a, &b);              // P(A∩B) = P(A)P(B)?

// Law of total probability
let total = space.total_probability(&a, &partition);

// Verify validity
assert!(space.is_valid());
```

### Random Variables

#### `RandomVariable`
A measurable function X: Ω → ℝ, supporting numerous distributions and operations.

```rust
// Distributions
let c = RandomVariable::constant(42.0);
let disc = RandomVariable::discrete(vec![1.0, 2.0, 3.0], vec![0.2, 0.5, 0.3]);
let u = RandomVariable::uniform(0.0, 1.0);
let n = RandomVariable::normal(0.0, 1.0);
let exp = RandomVariable::exponential(2.0);
let bern = RandomVariable::bernoulli(0.3);
let binom = RandomVariable::binomial(10, 0.5);
let pois = RandomVariable::poisson(3.0);

// Composition
let sum = RandomVariable::add(n_a, n_b);       // X + Y (independent)
let scaled = RandomVariable::scale(3.0, x);     // cX

// Distribution functions
let cdf_val = n.cdf(1.96);           // P(X ≤ 1.96)
let pdf_val = n.pdf(0.0);            // density at 0
let q75 = n.quantile(0.75);          // 75th percentile
let med = n.median();                // Q(0.5)

// Moments
let mu = n.expectation();            // E[X]
let var = n.variance();              // Var(X)
let std = n.std_dev();               // σ
let m3 = n.moment(3);               // E[X³]
let cm4 = n.central_moment(4);      // E[(X-μ)⁴]
let skew = n.skewness();            // E[(X-μ)³]/σ³
let kurt = n.kurtosis();            // E[(X-μ)⁴]/σ⁴ - 3

// Generating functions
let mgf_val = n.mgf(0.5);                              // M(t) = E[e^{tX}]
let (re, im) = n.characteristic_function(1.0);         // φ(t) = E[e^{itX}]

// Convert to measure
let measure = n.to_measure();
```

### Expectation & Integration

#### `lebesgue_integral`
Compute E[g(X)] via numerical quadrature.

```rust
let integral = lebesgue_integral(&rv, &|x| x * x);  // E[X²]
```

#### `expectation_lebesgue`, `moment`, `central_moment`
Convenience functions for common integrals.

```rust
let e = expectation_lebesgue(&rv);      // E[X]
let m4 = moment(&rv, 4);               // E[X⁴]
let cm3 = central_moment(&rv, 3);      // E[(X-μ)³]
```

#### `moment_generating_function`, `cumulant_generating_function`, `cumulant`

```rust
let mgf = moment_generating_function(&rv, 0.5);
let cgf = cumulant_generating_function(&rv, 0.5);
let k1 = cumulant(&rv, 1);  // mean
let k2 = cumulant(&rv, 2);  // variance
let k3 = cumulant(&rv, 3);  // third cumulant
let k4 = cumulant(&rv, 4);  // excess kurtosis-related
```

#### `covariance`, `correlation`, `law_of_total_variance`

```rust
let cov = covariance(e_xy, e_x, e_y);          // Cov(X,Y) = E[XY] - E[X]E[Y]
let corr = correlation(cov, sigma_x, sigma_y);  // ρ = Cov/(σ_X σ_Y)
let v = law_of_total_variance(e_var_given, var_e_given); // Var(X) = E[Var(X|Y)] + Var(E[X|Y])
```

#### `change_of_measure`
Importance sampling via Radon-Nikodym derivative.

```rust
let e_under_q = change_of_measure(&rv, &|x| dQ_dP(x)); // E_Q[X] = E_P[X · dQ/dP]
```

### Conditional Probability & Bayes

#### `conditional_probability`
P(A|B) = P(A∩B)/P(B).

#### `bayes_theorem`
P(A|B) = P(B|A)·P(A)/P(B).

```rust
let posterior = bayes_theorem(sensitivity, prevalence, p_positive);
```

#### `bayes_full`
Full Bayesian update over a partition.

```rust
let posteriors = bayes_full(&likelihoods, &priors);
// Returns P(Aᵢ|B) for each hypothesis
```

#### `total_probability_continuous`
P(A) = ∫ P(A|Y=y) f_Y(y) dy via numerical integration.

#### `mutual_information`
I(X;Y) = Σ P(x,y) log(P(x,y)/(P(x)P(y))).

#### `naive_bayes_classify`
Simple Naive Bayes classifier.

#### `likelihood_ratio`, `posterior_odds`
Hypothesis testing utilities.

### Martingale Theory

#### `Martingale`
A sequence (X₀, X₁, ...) where E[X_{n+1} | F_n] = X_n.

```rust
// Symmetric random walk
let rw = Martingale::symmetric_random_walk(vec![1, -1, 1, 1, -1]);

// Product martingale
let prod = Martingale::product_martingale(vec![1.0, 1.0, 1.0]);

// Doob's martingale (revelation of information over time)
let doob = Martingale::doob_martingale(terminal_value, observations);

// Properties
let vals = rw.values();
let incs = rw.increments();
let qv = rw.quadratic_variation();          // [X]_n = Σ (ΔX_i)²
let pqv = rw.predictable_quadratic_variation(); // ⟨X⟩_n

// Checks
let is_mg = rw.check_martingale_property();
let is_sub = rw.check_submartingale_property();
let is_super = rw.check_supermartingale_property();

// Doob's inequalities
let bound = rw.doob_maximal_inequality(3.0);  // P(max X_k ≥ λ) ≤ E[|X_n|]/λ
let lp = rw.doob_lp_bound(2.0);               // E[max |X_k|^p] ≤ (p/(p-1))^p E[|X_n|^p]

// Azuma-Hoeffding concentration
let prob = rw.azuma_hoeffding_bound(t, &bounds); // P(|X_n - X_0| ≥ t) ≤ 2 exp(...)
```

#### `StoppingTime`
A stopping time τ: {τ ≤ n} ∈ F_n.

```rust
let fixed = StoppingTime::at(10);
let first_hit = StoppingTime::first_passage(&mg, 5.0);
let exit = StoppingTime::first_exit(&mg, -2.0, 2.0);

let x_tau = fixed.evaluate(&mg);
```

#### `optional_stopping_theorem`
E[X_τ] ≈ E[X_0] under boundedness conditions.

#### `doob_decomposition`
Any adapted process decomposes as X = M + A where M is a martingale and A is predictable.

#### `doob_convergence_check`
If sup_n E[|X_n|] < ∞, the martingale converges almost surely.

#### `wald_equation`
E[Σ_{i=1}^τ X_i] = E[τ] · E[X₁].

## How It Works

**Probability spaces** are built from three components: a sample space (what can happen), a sigma-algebra (what events are measurable), and a probability measure (the probability of each event). The sigma-algebra is closed under complements and countable unions, computed by iterative closure from generating events.

**Random variables** store their distribution parameters and compute CDF, PDF, moments, MGF, and characteristic functions analytically where possible (Normal, Uniform, Exponential, Bernoulli, Binomial, Poisson) and numerically otherwise (quadrature over a finite grid). Sums of independent RVs use numerical convolution.

**Expectation** is computed as the Lebesgue integral ∫ g(x) f(x) dx via Riemann sum approximation over a grid. Moments, central moments, cumulants, and generating functions are all derived from this integral.

**Bayesian inference** implements Bayes' theorem in both simple and partitioned forms, with support for likelihood ratios, posterior odds, Naive Bayes classification, and mutual information computation.

**Martingale theory** implements the key objects and theorems: symmetric random walks, product martingales, Doob martingales, stopping times (first passage, first exit), the optional stopping theorem, Doob's decomposition, Doob's convergence theorem, Doob's maximal and L^p inequalities, the Azuma-Hoeffding bound, and Wald's equation.

## The Math

### Kolmogorov's Axioms

A probability measure P on (Ω, F) satisfies:
1. **Non-negativity**: P(A) ≥ 0 for all A ∈ F
2. **Normalization**: P(Ω) = 1
3. **Countable additivity**: P(∪ A_i) = Σ P(A_i) for disjoint A_i

### Bayes' Theorem

$$P(A_i \mid B) = \frac{P(B \mid A_i) \, P(A_i)}{\sum_j P(B \mid A_j) \, P(A_j)}$$

### Lebesgue Integral

$$\mathbb{E}[g(X)] = \int g(x) \, f(x) \, dx$$

### Moment Generating Function

$$M(t) = \mathbb{E}[e^{tX}]$$

Cumulants: $\kappa_n = M^{(n)}(0) / M(0)$, with $\kappa_1 = \mu$, $\kappa_2 = \sigma^2$.

### Martingale Property

A sequence $(X_n, F_n)$ is a **martingale** if $\mathbb{E}[X_{n+1} \mid F_n] = X_n$ for all n.

### Optional Stopping Theorem

For a martingale $X_n$ and stopping time $\tau$, if $\tau$ is bounded or $X_n$ is uniformly integrable:

$$\mathbb{E}[X_\tau] = \mathbb{E}[X_0]$$

### Doob's Maximal Inequality

For a non-negative submartingale:

$$P\left(\max_{k \leq n} X_k \geq \lambda\right) \leq \frac{\mathbb{E}[X_n]}{\lambda}$$

### Azuma-Hoeffding Bound

For a martingale with bounded differences $|X_k - X_{k-1}| \leq c_k$:

$$P(|X_n - X_0| \geq t) \leq 2 \exp\left(-\frac{t^2}{2 \sum c_k^2}\right)$$

## License

MIT

## References

- **Billingsley, P.** (1995). *Probability and Measure*. Wiley. — Measure-theoretic foundations.
- **Williams, D.** (1991). *Probability with Martingales*. Cambridge. — Martingale theory and Doob's inequalities.
- **Doob, J.L.** (1953). *Stochastic Processes*. Wiley. — Optional stopping theorem.
- **Hoeffding, W.** (1963). "Probability Inequalities for Sums of Bounded Random Variables." *J. Amer. Statist. Assoc.* 58, 13–30.
- **Azuma, K.** (1967). "Weighted Sums of Certain Dependent Random Variables." *Tôhoku Math. J.* 19, 357–367.
