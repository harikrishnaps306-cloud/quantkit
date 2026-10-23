//! Monte Carlo pricing under geometric Brownian motion.
//!
//! * [`european`]: exact terminal-value simulation, with optional antithetic
//!   variates, a standard error and a confidence interval.
//! * [`american_lsm`] / [`american_lsm_detailed`]: Longstaff-Schwartz
//!   least-squares Monte Carlo for American (strictly, Bermudan with `steps`
//!   exercise dates) options, with the exercise policy **trained on one set of
//!   paths and evaluated on an independent set**.
//!
//! # Model
//!
//! Risk-neutral dynamics `dS = (r - q) S dt + sigma S dW`, simulated exactly
//! (`S_{t+dt} = S_t exp((r - q - sigma^2/2) dt + sigma sqrt(dt) Z)`) and
//! discounted at the constant rate `r`.
//!
//! # Reproducibility guarantee
//!
//! Results are a pure function of `(inputs, paths, seed, antithetic, steps)`,
//! the crate version and the versions of the compiler and of the `libm`
//! dependency. They do **not** depend on the number of threads (the `parallel`
//! feature changes speed only: work is split into fixed-size blocks, each with
//! its own random stream derived from the seed and the block index, and block
//! statistics are merged in block order).
//!
//! The code is written to avoid platform dependence (IEEE-exact operations and
//! the pure-Rust `libm` instead of the platform's C library), but that is a
//! design, not a proof. What has actually been checked is listed in
//! `docs/NUMERICS.md`: identical bits on `x86_64-pc-windows-msvc` with two Rust
//! compilers and one `libm` version. Other targets are covered only by CI that
//! is configured but has not been run. A new `libm` or compiler release could
//! change low-order bits, so treat the golden values in the tests as regression
//! locks for a pinned toolchain, not as a guarantee for all future builds.
//!
//! # What the error bars mean
//!
//! * `std_error` is the **sampling** standard error of the reported mean:
//!   the usual `s / sqrt(n)` over independent samples (antithetic pairs are
//!   averaged first and count as one sample, so at least two samples, i.e. two
//!   paths or four antithetic paths, are required). A confidence interval built from
//!   it assumes the central limit theorem applies, which is accurate for
//!   thousands of paths but not for very few paths or extremely deep
//!   out-of-the-money payoffs.
//! * It does **not** include model error (GBM may be wrong), discretisation
//!   error (exercise only at `steps` dates), or, for American options, the
//!   gap between the learned and the optimal exercise policy.
//!
//! References: P. Glasserman, *Monte Carlo Methods in Financial Engineering*
//! (2004); F. Longstaff and E. Schwartz, "Valuing American Options by
//! Simulation: A Simple Least-Squares Approach", *Review of Financial
//! Studies* 14 (2001).

use crate::error::{QuantError, Result};
use crate::linalg::solve3;
use crate::option::{OptionParams, OptionType};

/// A small, fast, seedable pseudo-random generator (xoshiro256++).
///
/// Not cryptographically secure. Use it for simulation only. Normal variates
/// use Box-Muller with the pure-Rust `libm` functions, so streams are
/// identical on every platform.
///
/// # Examples
///
/// ```
/// use quantkit::mc::Rng;
/// let mut a = Rng::new(7);
/// let mut b = Rng::new(7);
/// assert_eq!(a.next_f64(), b.next_f64()); // same seed, same stream
/// let z = a.next_normal();
/// assert!(z.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

/// Maps a raw 64-bit word to a double in the open interval `(0, 1)`; see
/// [`Rng::next_f64`] for the grid.
fn u64_to_open_unit(raw: u64) -> f64 {
    ((raw >> 12) as f64 + 0.5) * (1.0 / 4_503_599_627_370_496.0)
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Creates a generator from a 64-bit seed (expanded with splitmix64).
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        Self {
            s: [
                splitmix64(&mut x),
                splitmix64(&mut x),
                splitmix64(&mut x),
                splitmix64(&mut x),
            ],
            spare: None,
        }
    }

    /// A generator for parallel stream number `stream` derived from `seed`.
    ///
    /// Three different things must not be confused:
    ///
    /// * **Distinct stream identifiers** are a property of the caller's
    ///   labels.
    /// * **Distinct initial states** are what this function provides in
    ///   practice: `(seed, stream)` is hashed to a 64-bit seed that is expanded
    ///   to the four-word state. Two different pairs can in principle hash to
    ///   the same seed (about a `2^-64` chance per pair), and the hash is not
    ///   known to be injective.
    /// * **Non-overlapping sequences** are *not* guaranteed: xoshiro256++ has
    ///   period `2^256 - 1`, so overlap between two random starting points is
    ///   astronomically unlikely, but nothing here proves it cannot happen (a
    ///   jump-ahead construction would).
    ///
    /// **Statistical independence** of the streams is a modelling assumption
    /// that standard tests do not contradict; it is not proved.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::mc::Rng;
    /// let mut a = Rng::for_stream(1, 0);
    /// let mut b = Rng::for_stream(1, 1);
    /// assert_ne!(a.next_u64(), b.next_u64());
    /// ```
    pub fn for_stream(seed: u64, stream: u64) -> Self {
        let mut a = seed;
        let mut b = stream ^ 0xA076_1D64_78BD_642F;
        Self::new(splitmix64(&mut a) ^ splitmix64(&mut b).rotate_left(32))
    }

    /// Creates a generator directly from its four state words, for testing
    /// against published reference vectors.
    ///
    /// The all-zero state is invalid for xoshiro (it would output zeros
    /// forever); it is replaced by the state of `Rng::new(0)`.
    pub fn from_state(state: [u64; 4]) -> Self {
        if state == [0; 4] {
            return Self::new(0);
        }
        Self {
            s: state,
            spare: None,
        }
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform sample in the **open** interval `(0, 1)`.
    ///
    /// The top 52 bits of a draw select one of `2^52` equally spaced
    /// midpoints `(k + 1/2) / 2^52`, so the smallest possible value is `2^-53`
    /// and the largest is `1 - 2^-53`; both are exactly representable, and
    /// neither endpoint can occur. (The discretisation is 52 bits, one less
    /// than the 53 an `f64` can hold in `[0.5, 1)`, because a 53-bit midpoint
    /// grid would round up to exactly `1.0` at its top end.)
    pub fn next_f64(&mut self) -> f64 {
        u64_to_open_unit(self.next_u64())
    }

    /// Standard normal sample (Box-Muller; produces samples in pairs).
    pub fn next_normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let r = libm::sqrt(-2.0 * libm::log(self.next_f64()));
        let (sin, cos) = libm::sincos(std::f64::consts::TAU * self.next_f64());
        self.spare = Some(r * sin);
        r * cos
    }
}

/// Derives a per-item seed from a base seed and an index, for batches where
/// every item needs its own independent random stream.
///
/// # Examples
///
/// ```
/// use quantkit::mc::derive_seed;
/// assert_eq!(derive_seed(1, 5), derive_seed(1, 5));
/// assert_ne!(derive_seed(1, 5), derive_seed(1, 6));
/// ```
pub fn derive_seed(seed: u64, index: u64) -> u64 {
    let mut a = seed;
    let mut b = index ^ 0xE703_7ED1_A0B4_28DB;
    splitmix64(&mut a) ^ splitmix64(&mut b).rotate_left(17)
}

/// Simulation settings.
///
/// # Examples
///
/// ```
/// use quantkit::mc::McConfig;
/// let cfg = McConfig { paths: 50_000, seed: 1, antithetic: true };
/// assert_eq!(cfg.paths, 50_000);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct McConfig {
    /// Number of simulated paths (2 to 10 billion).
    pub paths: usize,
    /// RNG seed. The same seed gives the same result.
    pub seed: u64,
    /// Use antithetic variates (each normal draw `z` is paired with `-z`).
    pub antithetic: bool,
}

impl Default for McConfig {
    fn default() -> Self {
        Self {
            paths: 100_000,
            seed: 42,
            antithetic: true,
        }
    }
}

/// A Monte Carlo estimate with its sampling error.
///
/// # Examples
///
/// ```
/// use quantkit::mc::McResult;
/// let r = McResult { price: 10.0, std_error: 0.1 };
/// let (lo, hi) = r.confidence_interval(1.96);
/// assert!(lo < 10.0 && 10.0 < hi);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct McResult {
    /// Estimated price.
    pub price: f64,
    /// Sampling standard error of the estimate (see the module docs for what
    /// this does and does not cover).
    pub std_error: f64,
}

impl McResult {
    /// `price +/- z * std_error`, e.g. `z = 1.96` for an approximate 95%
    /// interval (normal approximation).
    pub fn confidence_interval(&self, z: f64) -> (f64, f64) {
        (
            self.price - z * self.std_error,
            self.price + z * self.std_error,
        )
    }
}

/// Upper bound on `paths * steps` for [`american_lsm`], which stores every
/// training path.
const MAX_LSM_CELLS: usize = 100_000_000;
/// Largest path count accepted.
const MAX_PATHS: u64 = 10_000_000_000;
/// Samples per independent random-number block.
const BLOCK: usize = 4096;
/// Stream-identifier domains for LSM training paths and evaluation blocks. They
/// are different identifiers, so the two sets of paths start from different
/// states even if both configs carry the same seed (see [`Rng::for_stream`]
/// for what that does and does not guarantee).
const TRAIN_STREAM: u64 = 1 << 62;
const EVAL_STREAM: u64 = 2 << 62;

fn check_config(cfg: &McConfig) -> Result<()> {
    if cfg.paths < 2 {
        return Err(QuantError::InvalidInput {
            name: "paths",
            reason: "need at least 2 paths",
        });
    }
    if cfg.paths as u64 > MAX_PATHS {
        return Err(QuantError::InvalidInput {
            name: "paths",
            reason: "more than 10 billion paths",
        });
    }
    Ok(())
}

fn payoff(kind: OptionType, s: f64, k: f64) -> f64 {
    match kind {
        OptionType::Call => (s - k).max(0.0),
        OptionType::Put => (k - s).max(0.0),
    }
}

/// Running count, mean and sum of squared deviations (Welford), mergeable
/// (Chan et al.) so block results can be combined in a fixed order.
#[derive(Debug, Clone, Copy, Default)]
struct Stats {
    n: f64,
    mean: f64,
    m2: f64,
}

impl Stats {
    fn add(&mut self, x: f64) {
        self.n += 1.0;
        let delta = x - self.mean;
        self.mean += delta / self.n;
        self.m2 += delta * (x - self.mean);
    }
    fn merge(self, o: Stats) -> Stats {
        if o.n == 0.0 {
            return self;
        }
        if self.n == 0.0 {
            return o;
        }
        let n = self.n + o.n;
        let delta = o.mean - self.mean;
        Stats {
            n,
            mean: self.mean + delta * o.n / n,
            m2: self.m2 + o.m2 + delta * delta * self.n * o.n / n,
        }
    }
    fn mean_and_se(self) -> (f64, f64) {
        (self.mean, (self.m2 / (self.n - 1.0) / self.n).sqrt())
    }
}

#[cfg(feature = "parallel")]
fn run_blocks<F: Fn(u64) -> Stats + Sync + Send>(blocks: u64, f: F) -> Vec<Stats> {
    use rayon::prelude::*;
    (0..blocks).into_par_iter().map(f).collect()
}

#[cfg(not(feature = "parallel"))]
fn run_blocks<F: Fn(u64) -> Stats + Sync + Send>(blocks: u64, f: F) -> Vec<Stats> {
    (0..blocks).map(f).collect()
}

/// Splits `samples` into blocks, runs `f(block_index, count)` on each (in
/// parallel with the `parallel` feature) and merges in block order.
fn blocked<F: Fn(u64, usize) -> Stats + Sync + Send>(samples: usize, f: F) -> Stats {
    let blocks = samples.div_ceil(BLOCK) as u64;
    let parts = run_blocks(blocks, |b| {
        let start = b as usize * BLOCK;
        f(b, BLOCK.min(samples - start))
    });
    parts.into_iter().fold(Stats::default(), Stats::merge)
}

/// Monte Carlo price of a European option, simulating the terminal price
/// exactly: `S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)`.
///
/// With antithetic variates the estimate averages the payoffs from `Z` and
/// `-Z`; the standard error is computed over those pair averages, so it is
/// honest about the variance reduction. With an odd `paths` and antithetic
/// sampling, `paths - 1` paths are used. See the [module docs](self) for the
/// reproducibility guarantee and the meaning of the error bar.
///
/// # Errors
///
/// [`QuantError::InvalidInput`] if `paths < 2` or above 10 billion.
///
/// # Examples
///
/// ```
/// use quantkit::{bsm, mc, OptionParams, OptionType};
///
/// let p = OptionParams::builder()
///     .spot(42.0).strike(40.0).rate(0.10).volatility(0.20).expiry(0.5)
///     .build()?;
/// let est = mc::european(&p, OptionType::Call, mc::McConfig::default())?;
/// let exact = bsm::price(&p, OptionType::Call);
/// assert!((est.price - exact).abs() < 4.0 * est.std_error);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn european(p: &OptionParams, kind: OptionType, cfg: McConfig) -> Result<McResult> {
    check_config(&cfg)?;
    let t = p.expiry();
    let sig_sqrt_t = p.volatility() * t.sqrt();
    // Everything is done in *discounted* units, divided by a common scale `c`:
    //   Y = S_T e^{-rT} = S e^{-qT} exp(-sigma^2 T / 2 + sigma sqrt(T) z),
    //   strike' = K e^{-rT}.
    // Simulating S_T directly overflows for large (r - q) T even when the
    // discounted quantities are modest. Dividing by `c` keeps the simulated
    // values O(1) and the variance accumulator far from overflow; the result is
    // multiplied back at the end and rejected if it cannot be represented.
    let ln_spot_disc = libm::log(p.spot()) - p.dividend_yield() * t;
    let ln_strike_disc = libm::log(p.strike()) - p.rate() * t;
    let ln_c = ln_spot_disc.max(ln_strike_disc);
    let ln_y0 = ln_spot_disc - 0.5 * p.volatility() * p.volatility() * t - ln_c;
    let strike_n = libm::exp(ln_strike_disc - ln_c);
    let terminal = |z: f64| libm::exp(ln_y0 + sig_sqrt_t * z);

    let samples = effective_samples(&cfg);
    if samples < 2 {
        return Err(QuantError::InvalidInput {
            name: "paths",
            reason: "need at least 2 independent observations (an antithetic pair counts once, so antithetic sampling needs at least 4 paths)",
        });
    }
    let stats = blocked(samples, |block, count| {
        let mut rng = Rng::for_stream(cfg.seed, block);
        let mut st = Stats::default();
        for _ in 0..count {
            let z = rng.next_normal();
            if cfg.antithetic {
                st.add(
                    0.5 * (payoff(kind, terminal(z), strike_n)
                        + payoff(kind, terminal(-z), strike_n)),
                );
            } else {
                st.add(payoff(kind, terminal(z), strike_n));
            }
        }
        st
    });
    let (mean, se) = stats.mean_and_se();
    let scale = libm::exp(ln_c);
    let (price, std_error) = (mean * scale, se * scale);
    if !(price.is_finite() && std_error.is_finite()) {
        return Err(QuantError::NonFinite {
            name: "monte carlo result",
        });
    }
    Ok(McResult { price, std_error })
}

/// Regression of the continuation value at one exercise date, on the
/// standardised basis `1, z, z^2` with `z = (S/K - mean) / scale`.
#[derive(Debug, Clone, Copy)]
struct StepFit {
    mean: f64,
    scale: f64,
    coef: [f64; 3],
}

impl StepFit {
    fn continuation(&self, s_over_k: f64) -> f64 {
        let z = (s_over_k - self.mean) / self.scale;
        self.coef[0] + z * (self.coef[1] + z * self.coef[2])
    }
}

/// Result of [`american_lsm_detailed`].
///
/// # Which numbers estimate what
///
/// The method first **trains** an exercise policy on one set of paths: the
/// regressions at the interior dates, and a decision at time zero
/// (`exercise_now`, taken when the immediate payoff exceeds the training
/// paths' estimate of the continuation value). Both decisions are fixed
/// before any evaluation path is drawn. The policy is then **evaluated** on
/// independent paths.
///
/// * [`price`](Self::price) / [`std_error`](Self::std_error) describe the fixed
///   trained policy: the immediate payoff (standard error `0`) if
///   `exercise_now`, otherwise the evaluation-path mean of discounted cash
///   flows. Conditional on the trained policy, the evaluation mean is an
///   unbiased estimate of that policy's value, and a feasible stopping rule
///   can be worth no more than the optimal Bermudan value, so in expectation
///   over the evaluation sample `price` does not exceed the Bermudan value
///   (it is not guaranteed to be close to it).
/// * `std_error` is the **sampling** error of the evaluation mean *given* the
///   trained policy. It does not include the variation of the policy across
///   training samples, the gap between the learned and the optimal policy, the
///   restriction to `steps` exercise dates (a Bermudan value is below the
///   American value), or model error, so intervals built from it do not cover
///   those.
/// * [`continuation_value`](Self::continuation_value) is the raw
///   evaluation-path mean of the policy that never exercises at time zero,
///   reported even when `exercise_now`.
/// * [`in_sample_price`](Self::in_sample_price) is the same continuation
///   value measured on the training paths the regressions were fitted to. The
///   fitting tends to make that number optimistic; it is a diagnostic only
///   (compare it with `continuation_value`) and is not an estimate with a
///   valid error bar.
///
/// # Counts
///
/// `train_paths` and `eval_paths` count simulated paths. `train_samples` and
/// `eval_samples` count the **independent observations** the standard errors
/// are computed from: an antithetic pair is averaged first and counts once.
/// Both must be at least two.
///
/// # Examples
///
/// ```
/// use quantkit::{mc, OptionParams, OptionType};
/// let p = OptionParams::builder().spot(36.0).strike(40.0).rate(0.06).volatility(0.2).expiry(1.0).build()?;
/// let cfg = mc::McConfig { paths: 5_000, seed: 3, antithetic: true };
/// let r = mc::american_lsm_detailed(&p, OptionType::Put, 25, cfg, cfg)?;
/// assert!(r.fitted_dates > 0 && r.std_error > 0.0);
/// assert_eq!(r.train_samples, 2_500); // antithetic pairs
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LsmResult {
    /// Value estimate of the fixed trained policy (see the type docs).
    pub price: f64,
    /// Sampling standard error of `price` given the trained policy; `0` when
    /// `exercise_now`.
    pub std_error: f64,
    /// Whether the trained policy exercises immediately (decided from the
    /// training paths only).
    pub exercise_now: bool,
    /// The immediate-exercise payoff, `max(S - K, 0)` or `max(K - S, 0)`.
    pub immediate_exercise_value: f64,
    /// Raw evaluation-path mean of the policy that does not exercise at time
    /// zero.
    pub continuation_value: f64,
    /// Sampling standard error of `continuation_value`.
    pub continuation_std_error: f64,
    /// Continuation value measured on the training paths (optimistic).
    pub in_sample_price: f64,
    /// Standard error of the in-sample mean over its independent observations.
    pub in_sample_std_error: f64,
    /// Simulated training paths.
    pub train_paths: usize,
    /// Independent training observations (antithetic pairs count once).
    pub train_samples: usize,
    /// Simulated evaluation paths.
    pub eval_paths: usize,
    /// Independent evaluation observations (antithetic pairs count once).
    pub eval_samples: usize,
    /// Interior exercise dates with a usable regression. Dates without one
    /// are treated as "never exercise early".
    pub fitted_dates: usize,
    /// Interior dates with in-the-money paths but too few points or a
    /// singular regression.
    pub regression_failures: usize,
}

impl LsmResult {
    /// `max(price, immediate_exercise_value)`, a convenient lower-bounded
    /// number for display.
    ///
    /// This is **not** a statistical estimate: taking a maximum after the
    /// fact is optimistically biased relative to `price`, and `std_error`
    /// does not describe it. Use [`price`](Self::price) for inference.
    pub fn floored_price(&self) -> f64 {
        self.price.max(self.immediate_exercise_value)
    }
}

/// Longstaff-Schwartz price with the exercise policy trained on `cfg` paths
/// and evaluated on independent paths of the same count. Returns the value of
/// the fixed trained policy and its conditional sampling standard error; see
/// [`LsmResult`] and [`american_lsm_detailed`] for exactly what that means.
///
/// # Errors
///
/// [`QuantError::InvalidInput`] if either set of paths yields fewer than two
/// independent observations (`paths < 2`, or `paths < 4` with antithetic
/// sampling), `steps == 0`, or `paths * steps` exceeds 100 million (training
/// paths are held in memory); [`QuantError::NonFinite`] if simulated prices or
/// the result overflow.
///
/// # Examples
///
/// ```
/// use quantkit::{mc, OptionParams, OptionType};
///
/// let p = OptionParams::builder()
///     .spot(36.0).strike(40.0).rate(0.06).volatility(0.2).expiry(1.0)
///     .build()?;
/// let cfg = mc::McConfig { paths: 20_000, seed: 1, antithetic: true };
/// let est = mc::american_lsm(&p, OptionType::Put, 50, cfg)?;
/// assert!(est.price > 4.0 && est.price < 5.0);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn american_lsm(
    p: &OptionParams,
    kind: OptionType,
    steps: usize,
    cfg: McConfig,
) -> Result<McResult> {
    let r = american_lsm_detailed(p, kind, steps, cfg, cfg)?;
    Ok(McResult {
        price: r.price,
        std_error: r.std_error,
    })
}

/// Number of independent observations a configuration yields: antithetic
/// pairs count once.
fn effective_samples(cfg: &McConfig) -> usize {
    if cfg.antithetic {
        cfg.paths / 2
    } else {
        cfg.paths
    }
}

/// Longstaff-Schwartz with separate training and evaluation configurations.
///
/// 1. **Train**: simulate `train.paths` paths (stored in memory), and
///    backward-induct, regressing discounted continuation cash flows of
///    in-the-money paths on `1, z, z^2` at each interior date. The time-zero
///    decision is taken here too: exercise immediately if the immediate payoff
///    exceeds the training paths' continuation estimate.
/// 2. **Evaluate**: simulate `eval.paths` *fresh* paths from random streams
///    with distinct stream identifiers (see [`Rng::for_stream`]; this makes
///    reuse of the training stream extremely unlikely but is not a proof of
///    non-overlap), apply the fixed policy, and average discounted payoffs.
///    Evaluation is streamed, so it needs no path storage.
///
/// See [`LsmResult`] for which field estimates what, what the standard errors
/// condition on, and which sources of error they exclude.
///
/// # Errors
///
/// [`QuantError::InvalidInput`] if either config yields fewer than two
/// independent observations (`paths < 2`; `paths < 4` with antithetic
/// sampling, because a pair is one observation), too many paths, `steps == 0`,
/// or `train.paths * steps` above 100 million; [`QuantError::NonFinite`] if a
/// simulated price or the result is not finite (for example, parameters whose
/// simulated prices exceed the range of `f64`).
///
/// # Examples
///
/// ```
/// use quantkit::{mc, OptionParams, OptionType};
///
/// let p = OptionParams::builder()
///     .spot(40.0).strike(40.0).rate(0.06).volatility(0.2).expiry(1.0)
///     .build()?;
/// let train = mc::McConfig { paths: 20_000, seed: 1, antithetic: true };
/// let eval = mc::McConfig { paths: 100_000, seed: 2, antithetic: true };
/// let r = mc::american_lsm_detailed(&p, OptionType::Put, 50, train, eval)?;
/// assert!(r.price > 0.0 && r.std_error > 0.0);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn american_lsm_detailed(
    p: &OptionParams,
    kind: OptionType,
    steps: usize,
    train: McConfig,
    eval: McConfig,
) -> Result<LsmResult> {
    check_config(&train)?;
    check_config(&eval)?;
    if steps == 0 {
        return Err(QuantError::InvalidInput {
            name: "steps",
            reason: "must be at least 1",
        });
    }
    let train_obs = effective_samples(&train);
    let eval_obs = effective_samples(&eval);
    if train_obs < 2 || eval_obs < 2 {
        return Err(QuantError::InvalidInput {
            name: "paths",
            reason: "need at least 2 independent observations for a standard error (an antithetic pair counts once, so antithetic sampling needs at least 4 paths)",
        });
    }
    let n = if train.antithetic {
        train.paths & !1
    } else {
        train.paths
    };
    match n.checked_mul(steps) {
        Some(cells) if cells <= MAX_LSM_CELLS => {}
        _ => {
            return Err(QuantError::InvalidInput {
                name: "paths",
                reason: "paths * steps exceeds the 100 million cell limit (about 800 MB)",
            })
        }
    }
    let dt = p.expiry() / steps as f64;
    let drift = (p.rate() - p.dividend_yield() - 0.5 * p.volatility() * p.volatility()) * dt;
    let diffusion = p.volatility() * dt.sqrt();
    let disc_step = libm::exp(-p.rate() * dt);
    let k = p.strike();
    let growth = |z: f64| libm::exp(drift + diffusion * z);
    let disc_pow: Vec<f64> = (0..=steps)
        .map(|t| libm::exp(-p.rate() * dt * t as f64))
        .collect();
    let not_finite = |name| QuantError::NonFinite { name };

    // ---- Training -------------------------------------------------------
    let mut rng = Rng::for_stream(train.seed, TRAIN_STREAM);
    let mut s = vec![0.0; n * steps];
    let mut i = 0;
    while i < n {
        let (mut up, mut down) = (p.spot(), p.spot());
        for t in 0..steps {
            let z = rng.next_normal();
            up *= growth(z);
            s[i * steps + t] = up;
            if train.antithetic {
                down *= growth(-z);
                s[(i + 1) * steps + t] = down;
            }
        }
        i += if train.antithetic { 2 } else { 1 };
    }
    if s.iter().any(|v| !v.is_finite()) {
        return Err(not_finite("simulated price"));
    }

    let mut cash: Vec<f64> = (0..n)
        .map(|i| payoff(kind, s[i * steps + steps - 1], k))
        .collect();
    let mut tau = vec![steps; n];
    let mut policy: Vec<Option<StepFit>> = vec![None; steps + 1];
    let (mut fitted_dates, mut regression_failures) = (0usize, 0usize);

    for t in (1..steps).rev() {
        let itm: Vec<usize> = (0..n)
            .filter(|&i| payoff(kind, s[i * steps + t - 1], k) > 0.0)
            .collect();
        if itm.is_empty() {
            continue;
        }
        let xs: Vec<f64> = itm.iter().map(|&i| s[i * steps + t - 1] / k).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        let scale = var.sqrt();
        let usable = itm.len() >= 10 && scale > 1e-12 && scale.is_finite();
        let mut fit = None;
        if usable {
            let (mut a, mut b) = ([[0.0; 3]; 3], [0.0; 3]);
            for (&i, &x) in itm.iter().zip(&xs) {
                let z = (x - mean) / scale;
                let basis = [1.0, z, z * z];
                let y = cash[i] * disc_step.powi((tau[i] - t) as i32);
                for r in 0..3 {
                    b[r] += basis[r] * y;
                    for c in 0..3 {
                        a[r][c] += basis[r] * basis[c];
                    }
                }
            }
            fit = solve3(a, b)
                .map(|coef| StepFit { mean, scale, coef })
                .filter(|f| f.coef.iter().all(|c| c.is_finite()));
        }
        let Some(fit) = fit else {
            regression_failures += 1;
            continue;
        };
        fitted_dates += 1;
        policy[t] = Some(fit);
        for (&i, &x) in itm.iter().zip(&xs) {
            let exercise = payoff(kind, s[i * steps + t - 1], k);
            if exercise > fit.continuation(x) {
                cash[i] = exercise;
                tau[i] = t;
            }
        }
    }

    let path_values: Vec<f64> = (0..n).map(|i| cash[i] * disc_pow[tau[i]]).collect();
    let mut in_sample = Stats::default();
    if train.antithetic {
        for pair in path_values.chunks(2) {
            in_sample.add(0.5 * (pair[0] + pair[1]));
        }
    } else {
        path_values.iter().for_each(|&v| in_sample.add(v));
    }
    let (in_mean, in_se) = in_sample.mean_and_se();
    drop(s);

    // The time-zero decision uses training information only.
    let immediate = payoff(kind, p.spot(), k);
    let exercise_now = immediate > in_mean;

    // ---- Evaluation of the fixed policy on independent paths -------------
    let spot = p.spot();
    let policy_ref = &policy;
    let disc_ref = &disc_pow;
    let run_path = |zs: &[f64], sign: f64| -> f64 {
        let mut price = spot;
        for (t0, &z) in zs.iter().enumerate() {
            let t = t0 + 1;
            price *= growth(sign * z);
            let exercise = payoff(kind, price, k);
            if t == steps {
                return exercise * disc_ref[t];
            }
            if exercise > 0.0 {
                if let Some(fit) = &policy_ref[t] {
                    if exercise > fit.continuation(price / k) {
                        return exercise * disc_ref[t];
                    }
                }
            }
        }
        0.0
    };
    let evaluated = blocked(eval_obs, |block, count| {
        let mut rng = Rng::for_stream(eval.seed, EVAL_STREAM + block);
        let mut zs = vec![0.0; steps];
        let mut st = Stats::default();
        for _ in 0..count {
            for z in zs.iter_mut() {
                *z = rng.next_normal();
            }
            if eval.antithetic {
                st.add(0.5 * (run_path(&zs, 1.0) + run_path(&zs, -1.0)));
            } else {
                st.add(run_path(&zs, 1.0));
            }
        }
        st
    });
    let (out_mean, out_se) = evaluated.mean_and_se();

    let (price, std_error) = if exercise_now {
        (immediate, 0.0)
    } else {
        (out_mean, out_se)
    };
    let all = [
        price, std_error, immediate, out_mean, out_se, in_mean, in_se,
    ];
    if all.iter().any(|v| !v.is_finite()) {
        return Err(not_finite("monte carlo result"));
    }

    Ok(LsmResult {
        price,
        std_error,
        exercise_now,
        immediate_exercise_value: immediate,
        continuation_value: out_mean,
        continuation_std_error: out_se,
        in_sample_price: in_mean,
        in_sample_std_error: in_se,
        train_paths: n,
        train_samples: train_obs,
        eval_paths: if eval.antithetic {
            eval_obs * 2
        } else {
            eval_obs
        },
        eval_samples: eval_obs,
        fitted_dates,
        regression_failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{binomial, bsm, ExerciseStyle};
    use std::collections::HashSet;

    fn params(s: f64, k: f64, r: f64, q: f64, v: f64, t: f64) -> OptionParams {
        OptionParams::builder()
            .spot(s)
            .strike(k)
            .rate(r)
            .dividend_yield(q)
            .volatility(v)
            .expiry(t)
            .build()
            .unwrap()
    }

    // ---- PRNG reference vectors (separate from the statistical checks) -----

    /// xoshiro256++ with state words 1, 2, 3, 4. These ten outputs appear as the
    /// `reference` test of the `rand_xoshiro` crate, which states they were
    /// "produced with the reference implementation"
    /// (xoshiro.di.unimi.it/xoshiro256plusplus.c); the source was read on
    /// 2026-10-23. They also match an independent pure-Python implementation of
    /// the published algorithm (`benchmarks/prng_check.py`).
    #[test]
    fn xoshiro256pp_matches_published_reference_vector() {
        let mut rng = Rng::from_state([1, 2, 3, 4]);
        let expected: [u64; 10] = [
            41943041,
            58720359,
            3588806011781223,
            3591011842654386,
            9228616714210784205,
            9973669472204895162,
            14011001112246962877,
            12406186145184390807,
            15849039046786891736,
            10450023813501588000,
        ];
        for (i, want) in expected.into_iter().enumerate() {
            assert_eq!(rng.next_u64(), want, "output {i}");
        }
    }

    /// splitmix64 from seed 1234567. The values match an independent pure-Python
    /// implementation of the published algorithm; they are also the values used
    /// in `rand_xoshiro`'s `SplitMix64` test, but that source was not re-read.
    #[test]
    fn splitmix64_matches_reference_vector() {
        let mut state = 1234567_u64;
        let expected: [u64; 5] = [
            6457827717110365317,
            3203168211198807973,
            9817491932198370423,
            4593380528125082431,
            16408922859458223821,
        ];
        for (i, want) in expected.into_iter().enumerate() {
            assert_eq!(splitmix64(&mut state), want, "output {i}");
        }
    }

    #[test]
    fn unit_interval_mapping_is_strictly_open_at_both_ends() {
        let lo = u64_to_open_unit(0);
        let hi = u64_to_open_unit(u64::MAX);
        assert!(lo > 0.0 && lo == 0.5 * 2f64.powi(-52), "raw 0 maps to {lo}");
        assert!(
            hi < 1.0 && hi == 1.0 - 0.5 * 2f64.powi(-52),
            "raw MAX maps to {hi}"
        );
        // The 12 discarded low bits do not affect the result.
        assert_eq!(u64_to_open_unit(0xFFF), lo);
        // Monotone in the top 52 bits, with the documented grid spacing 2^-52.
        let a = u64_to_open_unit(1 << 12);
        assert_eq!(a - lo, 2f64.powi(-52));
        // Neither endpoint can reach the Box-Muller logarithm as 0 or 1.
        for raw in [0, 1, u64::MAX - 1, u64::MAX, 1 << 63] {
            let u = u64_to_open_unit(raw);
            assert!(u > 0.0 && u < 1.0, "raw {raw}: {u}");
            assert!(libm::log(u).is_finite() && libm::log(u) < 0.0);
        }
    }

    #[test]
    fn rng_is_deterministic_and_seed_sensitive() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(1);
        let mut c = Rng::new(2);
        let xa: Vec<u64> = (0..5).map(|_| a.next_u64()).collect();
        let xb: Vec<u64> = (0..5).map(|_| b.next_u64()).collect();
        let xc: Vec<u64> = (0..5).map(|_| c.next_u64()).collect();
        assert_eq!(xa, xb);
        assert_ne!(xa, xc);
    }

    /// Evidence, not proof: 64 streams x 2000 outputs share no 64-bit value (a
    /// chance duplicate has probability around 1e-11). This does not establish
    /// that the sequences cannot overlap.
    #[test]
    fn streams_show_no_output_collisions() {
        let mut seen = HashSet::new();
        for stream in 0..64u64 {
            let mut r = Rng::for_stream(42, stream);
            for _ in 0..2000 {
                assert!(
                    seen.insert(r.next_u64()),
                    "duplicate output in stream {stream}"
                );
            }
        }
        // Distinct stream identifiers give distinct initial states, including
        // the training/evaluation identifier domains used by Longstaff-Schwartz.
        let mut states = HashSet::new();
        for id in [TRAIN_STREAM, EVAL_STREAM, EVAL_STREAM + 1, 0, 1, 2] {
            assert!(
                states.insert(Rng::for_stream(7, id).s),
                "identical initial state"
            );
        }
    }

    // Statistical sanity checks (not known-answer vectors).
    #[test]
    fn uniform_and_normal_moments() {
        let mut rng = Rng::new(123);
        let n = 400_000;
        let u: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        assert!(u.iter().all(|&x| x > 0.0 && x < 1.0));
        let mean_u = u.iter().sum::<f64>() / n as f64;
        assert!((mean_u - 0.5).abs() < 5.0 * (1.0 / 12.0_f64 / n as f64).sqrt());

        let z: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let mean = z.iter().sum::<f64>() / n as f64;
        let var = z.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 5.0 / (n as f64).sqrt());
        assert!((var - 1.0).abs() < 5.0 * (2.0 / n as f64).sqrt());
        // Tail mass beyond 1.96 should be about 5%.
        let tail = z.iter().filter(|x| x.abs() > 1.96).count() as f64 / n as f64;
        assert!((tail - 0.05).abs() < 0.003, "tail {tail}");
    }

    #[test]
    fn stats_merge_matches_single_pass() {
        let xs: Vec<f64> = (0..1000)
            .map(|i| ((i * 37 % 101) as f64).sin() * 3.0)
            .collect();
        let mut whole = Stats::default();
        xs.iter().for_each(|&x| whole.add(x));
        let merged = xs
            .chunks(77)
            .map(|c| {
                let mut s = Stats::default();
                c.iter().for_each(|&x| s.add(x));
                s
            })
            .fold(Stats::default(), Stats::merge);
        assert!((whole.mean - merged.mean).abs() < 1e-12);
        assert!((whole.m2 - merged.m2).abs() < 1e-8);
    }

    #[test]
    fn european_agrees_with_bsm() {
        for &(s, k, r, q, v, t) in &[
            (42.0, 40.0, 0.10, 0.0, 0.20, 0.5),
            (100.0, 110.0, 0.03, 0.02, 0.35, 1.0),
            (100.0, 90.0, 0.0, 0.0, 0.25, 2.0),
        ] {
            let p = params(s, k, r, q, v, t);
            for kind in [OptionType::Call, OptionType::Put] {
                let est = european(&p, kind, McConfig::default()).unwrap();
                let exact = bsm::price(&p, kind);
                assert!(est.std_error > 0.0);
                assert!(
                    (est.price - exact).abs() < 4.0 * est.std_error,
                    "{kind:?} {s} {k}: {} vs {exact} (se {})",
                    est.price,
                    est.std_error
                );
            }
        }
    }

    /// Regression lock, NOT an accuracy reference: these values were recorded
    /// from this implementation (seed 42, 50,000 paths, antithetic) after the
    /// open-interval uniform mapping was corrected (see CHANGELOG: the
    /// previous mapping could return exactly 1.0, so the stream of normals, and
    /// hence this value, changed deliberately). The test fails if the random
    /// streams, block layout, merge order or evaluation order change, and it must
    /// pass identically with and without the `parallel` feature, which is how
    /// thread-count independence is checked.
    #[test]
    fn european_bitwise_regression_lock() {
        let p = params(100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        let cfg = McConfig {
            paths: 50_000,
            seed: 42,
            antithetic: true,
        };
        let r = european(&p, OptionType::Call, cfg).unwrap();
        assert_eq!(r.price.to_bits(), GOLDEN_PRICE_BITS, "price {}", r.price);
        assert_eq!(r.std_error.to_bits(), GOLDEN_SE_BITS, "se {}", r.std_error);
    }
    const GOLDEN_PRICE_BITS: u64 = 4622057817873776051;
    const GOLDEN_SE_BITS: u64 = 4586848783294312790;

    /// Regression for the overflow reproduction: with `r = 10`, `q = -10` and
    /// `T = 50` the terminal asset price is about `e^1000` (beyond `f64`), but
    /// the discounted price, about `S e^{-qT} = 1.4e219`, is representable.
    /// The estimate must be finite and consistent with the closed form.
    #[test]
    fn european_survives_terminal_prices_that_overflow() {
        let p = params(100.0, 100.0, 10.0, -10.0, 0.2, 50.0);
        let cfg = McConfig {
            paths: 100,
            seed: 1,
            antithetic: true,
        };
        let est = european(&p, OptionType::Call, cfg).unwrap();
        assert!(est.price.is_finite() && est.std_error.is_finite() && est.std_error > 0.0);
        let exact = bsm::price(&p, OptionType::Call);
        assert!(exact.is_finite() && exact > 1e219);
        assert!(
            (est.price - exact).abs() < 4.0 * est.std_error,
            "{} vs {exact} (se {})",
            est.price,
            est.std_error
        );
        // The put is worth almost nothing; it must be finite and non-negative.
        let put = european(&p, OptionType::Put, cfg).unwrap();
        assert!(put.price.is_finite() && put.price >= 0.0 && put.std_error.is_finite());
    }

    /// When the *result itself* cannot be represented, the answer is an
    /// explicit error, never a `Ok` carrying `NaN` or infinity.
    #[test]
    fn european_reports_unrepresentable_results_as_errors() {
        // sigma sqrt(T) ~ 3e4 makes the normalised terminal value overflow.
        let p = params(100.0, 100.0, 0.0, 0.0, 1000.0, 1000.0);
        let cfg = McConfig {
            paths: 200,
            seed: 3,
            antithetic: true,
        };
        match european(&p, OptionType::Call, cfg) {
            Err(QuantError::NonFinite { .. }) => {}
            Ok(r) => assert!(r.price.is_finite() && r.std_error.is_finite(), "{r:?}"),
            Err(e) => panic!("unexpected error {e}"),
        }
    }

    #[test]
    fn antithetic_reduces_error_and_seed_reproduces() {
        let p = params(100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        let plain = McConfig {
            antithetic: false,
            ..McConfig::default()
        };
        let anti = McConfig::default();
        let a = european(&p, OptionType::Call, anti).unwrap();
        let b = european(&p, OptionType::Call, plain).unwrap();
        assert!(a.std_error < b.std_error);
        assert_eq!(a, european(&p, OptionType::Call, anti).unwrap());
        let other = McConfig { seed: 99, ..anti };
        assert_ne!(
            a.price,
            european(&p, OptionType::Call, other).unwrap().price
        );
    }

    #[test]
    fn rejects_bad_config() {
        let p = params(100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        let bad = McConfig {
            paths: 1,
            ..McConfig::default()
        };
        assert!(european(&p, OptionType::Call, bad).is_err());
        assert!(american_lsm(&p, OptionType::Put, 10, bad).is_err());
        assert!(american_lsm(&p, OptionType::Put, 0, McConfig::default()).is_err());
        let huge = McConfig {
            paths: usize::MAX,
            ..McConfig::default()
        };
        assert!(european(&p, OptionType::Call, huge).is_err());
    }

    /// Two or three antithetic paths give one pair-average, so a sample
    /// standard error does not exist; four give two. This was the reproduction
    /// of `in_sample_std_error = NaN`.
    #[test]
    fn lsm_requires_two_independent_observations() {
        let p = params(100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        let eval = McConfig {
            paths: 100,
            seed: 2,
            antithetic: true,
        };
        for paths in [2, 3] {
            let train = McConfig {
                paths,
                seed: 1,
                antithetic: true,
            };
            assert!(
                matches!(
                    american_lsm_detailed(&p, OptionType::Put, 2, train, eval),
                    Err(QuantError::InvalidInput { name: "paths", .. })
                ),
                "{paths} antithetic training paths"
            );
        }
        let train4 = McConfig {
            paths: 4,
            seed: 1,
            antithetic: true,
        };
        let r = american_lsm_detailed(&p, OptionType::Put, 2, train4, eval).unwrap();
        assert_eq!((r.train_paths, r.train_samples), (4, 2));
        assert!(r.in_sample_std_error.is_finite());
        // The same rule applies to the evaluation sample.
        for paths in [2, 3] {
            let ev = McConfig { paths, ..eval };
            assert!(american_lsm_detailed(&p, OptionType::Put, 2, train4, ev).is_err());
        }
        let ev4 = McConfig { paths: 4, ..eval };
        let r = american_lsm_detailed(&p, OptionType::Put, 2, train4, ev4).unwrap();
        assert_eq!((r.eval_paths, r.eval_samples), (4, 2));
        // Without antithetic sampling two paths are two observations.
        let plain = McConfig {
            paths: 2,
            seed: 1,
            antithetic: false,
        };
        let r = american_lsm_detailed(&p, OptionType::Put, 2, plain, plain).unwrap();
        assert!(r.in_sample_std_error.is_finite() && r.std_error.is_finite());
        assert_eq!((r.train_paths, r.train_samples), (2, 2));
    }

    #[test]
    fn lsm_reports_overflowing_simulations_as_errors() {
        let p = params(100.0, 100.0, 10.0, -10.0, 0.2, 50.0);
        let cfg = McConfig {
            paths: 100,
            seed: 1,
            antithetic: true,
        };
        assert!(matches!(
            american_lsm_detailed(&p, OptionType::Call, 50, cfg, cfg),
            Err(QuantError::NonFinite { .. })
        ));
    }

    #[test]
    fn lsm_put_close_to_binomial() {
        // Longstaff-Schwartz (2001), Table 1 setting: K=40, r=6%, 50 dates/year.
        for &(s, v, t) in &[(36.0, 0.2, 1.0), (40.0, 0.2, 1.0), (44.0, 0.4, 2.0)] {
            let p = params(s, 40.0, 0.06, 0.0, v, t);
            let steps = (50.0 * t) as usize;
            let cfg = McConfig {
                paths: 100_000,
                ..McConfig::default()
            };
            let lsm = american_lsm(&p, OptionType::Put, steps, cfg).unwrap();
            let tree = binomial::price(&p, OptionType::Put, ExerciseStyle::American, 3000).unwrap();
            assert!(
                (lsm.price - tree).abs() < 0.06,
                "S={s} v={v} T={t}: lsm {} vs tree {tree} (se {})",
                lsm.price,
                lsm.std_error
            );
            // A feasible stopping rule cannot be worth more than the optimal one,
            // so up to sampling error it should not exceed the converged tree.
            assert!(lsm.price < tree + 4.0 * lsm.std_error);
        }
    }

    /// Over many independent small samples, the in-sample continuation value
    /// must exceed the independently evaluated one (the regressions were fitted
    /// to the training paths), and the independently evaluated policy value must
    /// not exceed the converged tree value beyond sampling error. Seeds are
    /// fixed, so this is deterministic; it is evidence about this parameter set,
    /// not a theorem about in-sample estimates.
    #[test]
    fn lsm_in_sample_is_optimistic_and_out_of_sample_is_not() {
        let p = params(36.0, 40.0, 0.06, 0.0, 0.2, 1.0);
        let tree = binomial::price(&p, OptionType::Put, ExerciseStyle::American, 3000).unwrap();
        let runs = 200u64;
        let (mut gaps, mut outs) = (Vec::new(), Vec::new());
        for seed in 0..runs {
            let train = McConfig {
                paths: 1_000,
                seed,
                antithetic: false,
            };
            let eval = McConfig {
                paths: 1_000,
                seed: seed + 10_000,
                antithetic: false,
            };
            let r = american_lsm_detailed(&p, OptionType::Put, 50, train, eval).unwrap();
            gaps.push(r.in_sample_price - r.continuation_value);
            outs.push(r.price);
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let se = |v: &[f64]| {
            let m = mean(v);
            (v.iter().map(|x| (x - m).powi(2)).sum::<f64>()
                / (v.len() as f64 - 1.0)
                / v.len() as f64)
                .sqrt()
        };
        let (gap, gap_se) = (mean(&gaps), se(&gaps));
        assert!(gap > 3.0 * gap_se, "optimism gap {gap} +/- {gap_se}");
        let (out, out_se) = (mean(&outs), se(&outs));
        assert!(
            out < tree + 3.0 * out_se,
            "oos {out} +/- {out_se} vs tree {tree}"
        );
    }

    #[test]
    fn lsm_training_and_evaluation_use_different_paths() {
        let p = params(36.0, 40.0, 0.06, 0.0, 0.2, 1.0);
        let same_seed = McConfig {
            paths: 2_000,
            seed: 5,
            antithetic: false,
        };
        let r = american_lsm_detailed(&p, OptionType::Put, 20, same_seed, same_seed).unwrap();
        // If evaluation reused the training paths the two estimates would coincide.
        assert!((r.in_sample_price - r.continuation_value).abs() > 1e-9);
        assert_eq!((r.train_samples, r.eval_samples), (2_000, 2_000));
    }

    /// Deep in the money with a positive rate, immediate exercise is optimal.
    /// The decision comes from the training paths alone: changing only the
    /// evaluation seed must not change it, and the reported price is then the
    /// payoff itself with zero standard error (not `max` of a noisy mean).
    #[test]
    fn lsm_immediate_exercise_is_decided_from_training_only() {
        let p = params(50.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        let train = McConfig {
            paths: 4_000,
            seed: 11,
            antithetic: true,
        };
        let mut previous = None;
        for eval_seed in [1, 2, 3, 4] {
            let eval = McConfig {
                paths: 2_000,
                seed: eval_seed,
                antithetic: true,
            };
            let r = american_lsm_detailed(&p, OptionType::Put, 25, train, eval).unwrap();
            assert!(r.exercise_now);
            assert_eq!(r.price, 50.0);
            assert_eq!(r.std_error, 0.0);
            assert_eq!(r.immediate_exercise_value, 50.0);
            assert_eq!(r.floored_price(), 50.0);
            assert!(r.continuation_std_error > 0.0);
            if let Some(prev) = previous {
                assert_eq!(
                    r.in_sample_price, prev,
                    "training result depends on the evaluation seed"
                );
            }
            previous = Some(r.in_sample_price);
        }
        // And for a call that is far out of the money, the policy continues.
        let otm = params(50.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        let r = american_lsm_detailed(&otm, OptionType::Call, 25, train, train).unwrap();
        assert!(!r.exercise_now && r.immediate_exercise_value == 0.0);
    }

    /// Near the exercise boundary the time-zero decision is noisy: across
    /// training seeds both outcomes occur. The returned price is always either
    /// the exact payoff (when exercising) or the evaluation mean (when not),
    /// never a maximum chosen after seeing the evaluation sample, and averaged
    /// over seeds it does not exceed the converged tree value.
    #[test]
    fn lsm_near_indifference_decision_is_noisy_but_unbiased_high() {
        // K = 40, r = 6%, sigma = 20%, T = 1: the early-exercise boundary is near 32.
        let p = params(32.3, 40.0, 0.06, 0.0, 0.2, 1.0);
        let tree = binomial::price(&p, OptionType::Put, ExerciseStyle::American, 3000).unwrap();
        let (mut now, mut cont, mut prices) = (0, 0, Vec::new());
        for seed in 0..120u64 {
            let train = McConfig {
                paths: 1_000,
                seed,
                antithetic: false,
            };
            let eval = McConfig {
                paths: 1_000,
                seed: seed + 5_000,
                antithetic: false,
            };
            let r = american_lsm_detailed(&p, OptionType::Put, 50, train, eval).unwrap();
            if r.exercise_now {
                now += 1;
                assert_eq!(r.price, r.immediate_exercise_value);
                assert_eq!(r.std_error, 0.0);
            } else {
                cont += 1;
                assert_eq!(r.price, r.continuation_value);
                assert_eq!(r.std_error, r.continuation_std_error);
            }
            prices.push(r.price);
        }
        assert!(
            now > 5 && cont > 5,
            "decision never flipped: exercise {now}, continue {cont}"
        );
        let m = prices.iter().sum::<f64>() / prices.len() as f64;
        let se = (prices.iter().map(|x| (x - m).powi(2)).sum::<f64>()
            / (prices.len() as f64 - 1.0)
            / prices.len() as f64)
            .sqrt();
        assert!(
            m < tree + 3.0 * se,
            "mean policy value {m} +/- {se} vs tree {tree}"
        );
    }

    #[test]
    fn lsm_call_without_dividends_matches_bsm() {
        let p = params(100.0, 100.0, 0.05, 0.0, 0.3, 1.0);
        let cfg = McConfig {
            paths: 100_000,
            ..McConfig::default()
        };
        let lsm = american_lsm(&p, OptionType::Call, 25, cfg).unwrap();
        let exact = bsm::price(&p, OptionType::Call);
        assert!((lsm.price - exact).abs() < 4.0 * lsm.std_error + 0.05);
    }

    #[test]
    fn lsm_is_reproducible() {
        let p = params(70.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        let cfg = McConfig {
            paths: 5_000,
            ..McConfig::default()
        };
        let a = american_lsm(&p, OptionType::Put, 20, cfg).unwrap();
        assert_eq!(a, american_lsm(&p, OptionType::Put, 20, cfg).unwrap());
    }

    #[test]
    fn lsm_reports_regression_failures_instead_of_hiding_them() {
        // Far out of the money: almost no in-the-money paths at any date.
        let p = params(100.0, 20.0, 0.01, 0.0, 0.05, 0.1);
        let cfg = McConfig {
            paths: 200,
            seed: 1,
            antithetic: false,
        };
        let r = american_lsm_detailed(&p, OptionType::Put, 10, cfg, cfg).unwrap();
        assert_eq!(r.fitted_dates, 0);
        assert!(r.price >= 0.0);
    }
}
