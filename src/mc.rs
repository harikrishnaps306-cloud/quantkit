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
//! Results are a pure function of `(inputs, paths, seed, antithetic, steps)`
//! and the crate version. They do **not** depend on the number of threads (the
//! `parallel` feature changes speed only: work is split into fixed-size
//! blocks, each with its own random stream derived from the seed and the block
//! index, and block statistics are merged in block order) and they are
//! bitwise reproducible across platforms, because the random-number code uses
//! only IEEE-exact operations and the pure-Rust `libm` transcendental
//! functions rather than the platform's C library. Changing the crate version
//! may change the numbers.
//!
//! # What the error bars mean
//!
//! * `std_error` is the **sampling** standard error of the reported mean:
//!   the usual `s / sqrt(n)` over independent samples (antithetic pairs are
//!   averaged first and count as one sample). A confidence interval built from
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
    /// Different `(seed, stream)` pairs give streams seeded from independently
    /// hashed 64-bit values. This makes overlaps astronomically unlikely in
    /// practice, but unlike a jump-ahead construction it does not *prove*
    /// that the streams are disjoint.
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

    /// Uniform sample in the open interval `(0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // 53 random bits, shifted by half a step so 0 and 1 are excluded.
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
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
/// Stream-id domain for LSM training paths and evaluation blocks, so the two
/// sets of paths never share a stream even if both configs carry the same seed.
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
    let drift = (p.rate() - p.dividend_yield() - 0.5 * p.volatility() * p.volatility()) * t;
    let diffusion = p.volatility() * t.sqrt();
    let disc = libm::exp(-p.rate() * t);
    let (spot, strike) = (p.spot(), p.strike());
    let terminal = |z: f64| spot * libm::exp(drift + diffusion * z);

    let samples = if cfg.antithetic {
        cfg.paths / 2
    } else {
        cfg.paths
    };
    if samples < 2 {
        return Err(QuantError::InvalidInput {
            name: "paths",
            reason: "too few paths for antithetic sampling",
        });
    }
    let stats = blocked(samples, |block, count| {
        let mut rng = Rng::for_stream(cfg.seed, block);
        let mut st = Stats::default();
        for _ in 0..count {
            let z = rng.next_normal();
            if cfg.antithetic {
                st.add(
                    0.5 * (payoff(kind, terminal(z), strike) + payoff(kind, terminal(-z), strike)),
                );
            } else {
                st.add(payoff(kind, terminal(z), strike));
            }
        }
        st
    });
    let (mean, se) = stats.mean_and_se();
    Ok(McResult {
        price: disc * mean,
        std_error: disc * se,
    })
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
/// # Examples
///
/// ```
/// use quantkit::{mc, OptionParams, OptionType};
/// let p = OptionParams::builder().spot(36.0).strike(40.0).rate(0.06).volatility(0.2).expiry(1.0).build()?;
/// let cfg = mc::McConfig { paths: 5_000, seed: 3, antithetic: true };
/// let r = mc::american_lsm_detailed(&p, OptionType::Put, 25, cfg, cfg)?;
/// assert!(r.fitted_dates > 0 && r.std_error > 0.0);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LsmResult {
    /// Price estimate from the **independent evaluation paths**, floored at
    /// the immediate-exercise value. Because the exercise rule is learned
    /// rather than optimal, this is biased slightly **low** relative to the
    /// Bermudan value (and the Bermudan value is below the American value).
    pub price: f64,
    /// Sampling standard error of the out-of-sample mean (before flooring).
    /// It excludes policy, discretisation and model error.
    pub std_error: f64,
    /// Price estimated on the training paths themselves. This is biased high
    /// by overfitting the regression to those paths; the gap to `price` is a
    /// diagnostic of how noisy the learned policy is.
    pub in_sample_price: f64,
    /// Standard error of the in-sample mean.
    pub in_sample_std_error: f64,
    /// Training paths used.
    pub train_samples: usize,
    /// Independent evaluation samples used (antithetic pairs count once).
    pub eval_samples: usize,
    /// Interior exercise dates with a usable regression. Dates without one
    /// are treated as "never exercise early".
    pub fitted_dates: usize,
    /// Interior dates with in-the-money paths but too few points or a
    /// singular regression.
    pub regression_failures: usize,
}

/// Longstaff-Schwartz price with the exercise policy trained on `cfg` paths
/// and evaluated on independent paths of the same count. Returns the
/// out-of-sample estimate. See [`american_lsm_detailed`] for diagnostics.
///
/// # Errors
///
/// [`QuantError::InvalidInput`] if `paths < 2`, `steps == 0`, or
/// `paths * steps` exceeds 100 million (training paths are held in memory).
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

/// Longstaff-Schwartz with separate training and evaluation configurations.
///
/// 1. **Train**: simulate `train.paths` paths (stored in memory), and
///    backward-induct, regressing discounted continuation cash flows of
///    in-the-money paths on `1, z, z^2` at each interior date.
/// 2. **Evaluate**: simulate `eval.paths` *fresh* paths from random streams
///    that are disjoint by construction from the training stream (even when
///    `train.seed == eval.seed`), apply the learned policy, and average
///    discounted payoffs. Evaluation is streamed, so it needs no path storage.
///
/// The out-of-sample price is a (low-biased) valid estimate of the value of a
/// feasible stopping rule; the in-sample price reuses the regression's own
/// paths and is reported for comparison only.
///
/// # Errors
///
/// [`QuantError::InvalidInput`] if either config has `paths < 2` or too many,
/// `steps == 0`, or `train.paths * steps` exceeds 100 million.
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
    match train.paths.checked_mul(steps) {
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

    // ---- Training -------------------------------------------------------
    let n = if train.antithetic {
        train.paths & !1
    } else {
        train.paths
    };
    if n < 2 {
        return Err(QuantError::InvalidInput {
            name: "paths",
            reason: "too few paths for antithetic sampling",
        });
    }
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
        let usable = itm.len() >= 10 && scale > 1e-12;
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
            fit = solve3(a, b).map(|coef| StepFit { mean, scale, coef });
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

    // ---- Evaluation on independent paths ---------------------------------
    let eval_samples = if eval.antithetic {
        eval.paths / 2
    } else {
        eval.paths
    };
    if eval_samples < 2 {
        return Err(QuantError::InvalidInput {
            name: "paths",
            reason: "too few paths for antithetic sampling",
        });
    }
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
    let evaluated = blocked(eval_samples, |block, count| {
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

    Ok(LsmResult {
        price: out_mean.max(payoff(kind, p.spot(), k)),
        std_error: out_se,
        in_sample_price: in_mean,
        in_sample_std_error: in_se,
        train_samples: n,
        eval_samples,
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

    #[test]
    fn streams_do_not_collide() {
        // 64 streams x 2000 outputs: any duplicate 64-bit output would mean
        // overlapping or identical streams (chance of a fluke is ~1e-11).
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
        // The train and evaluation stream domains are separate too.
        let mut a = Rng::for_stream(7, TRAIN_STREAM);
        let mut b = Rng::for_stream(7, EVAL_STREAM);
        assert_ne!(a.next_u64(), b.next_u64());
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
    /// from this implementation (seed 42, 50,000 paths, antithetic). The test
    /// fails if the random streams, block layout or merge order change, and it
    /// must pass identically with and without the `parallel` feature, which is
    /// how thread-count independence is checked.
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
    const GOLDEN_PRICE_BITS: u64 = 4622057817873776027;
    const GOLDEN_SE_BITS: u64 = 4586848783294312776;

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
            // Low-biased: should not sit far above the continuous-exercise value.
            assert!(lsm.price < tree + 4.0 * lsm.std_error);
        }
    }

    /// Over many independent small samples, the in-sample estimate must
    /// exceed the independent-path estimate (overfitting bias), and the
    /// independent estimate must not exceed the converged tree value beyond
    /// sampling error. Seeds are fixed, so this is deterministic.
    #[test]
    fn lsm_in_sample_is_biased_high_and_out_of_sample_is_not() {
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
            gaps.push(r.in_sample_price - r.price);
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
        assert!(gap > 3.0 * gap_se, "overfit gap {gap} +/- {gap_se}");
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
        assert!((r.in_sample_price - r.price).abs() > 1e-9);
        assert_eq!(r.train_samples, 2_000);
        assert_eq!(r.eval_samples, 2_000);
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
    fn lsm_is_at_least_intrinsic_and_reproducible() {
        let p = params(70.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        let cfg = McConfig {
            paths: 5_000,
            ..McConfig::default()
        };
        let a = american_lsm(&p, OptionType::Put, 20, cfg).unwrap();
        assert!(a.price >= 30.0);
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
