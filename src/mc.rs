//! Monte Carlo pricing under geometric Brownian motion.
//!
//! * [`european`]: exact terminal-value simulation, with optional antithetic
//!   variates, a standard error and a confidence interval.
//! * [`american_lsm`]: Longstaff-Schwartz least-squares Monte Carlo for
//!   American (strictly, Bermudan with `steps` exercise dates) options.
//!
//! Randomness comes from a small built-in generator ([`Rng`], xoshiro256++
//! seeded through splitmix64), so results are **reproducible for a given
//! seed** on every platform and the crate stays dependency-free.
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
/// Not cryptographically secure. Use it for simulation only.
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

impl Rng {
    /// Creates a generator from a 64-bit seed (expanded with splitmix64).
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
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
        let r = (-2.0 * self.next_f64().ln()).sqrt();
        let theta = std::f64::consts::TAU * self.next_f64();
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
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
    /// Number of simulated paths (at least 2).
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
    /// Standard error of the estimate.
    pub std_error: f64,
}

impl McResult {
    /// `price +/- z * std_error`, e.g. `z = 1.96` for a 95% interval.
    pub fn confidence_interval(&self, z: f64) -> (f64, f64) {
        (
            self.price - z * self.std_error,
            self.price + z * self.std_error,
        )
    }
}

/// Upper bound on `paths * steps` for [`american_lsm`], which stores every path.
const MAX_LSM_CELLS: usize = 100_000_000;

fn check_config(cfg: &McConfig) -> Result<()> {
    if cfg.paths < 2 {
        return Err(QuantError::InvalidInput {
            name: "paths",
            reason: "need at least 2 paths",
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

/// Mean and standard error of the mean of `xs`.
fn mean_and_se(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

/// Monte Carlo price of a European option, simulating the terminal price
/// exactly: `S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)`.
///
/// With antithetic variates the estimate averages the payoffs from `Z` and
/// `-Z`; the standard error is computed over those pair averages, so it is
/// honest about the variance reduction. With an odd `paths` and antithetic
/// sampling, `paths - 1` paths are used.
///
/// # Errors
///
/// [`QuantError::InvalidInput`] if `paths < 2`.
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
    let mut rng = Rng::new(cfg.seed);
    let t = p.expiry();
    let drift = (p.rate() - p.dividend_yield() - 0.5 * p.volatility() * p.volatility()) * t;
    let diffusion = p.volatility() * t.sqrt();
    let disc = (-p.rate() * t).exp();
    let terminal = |z: f64| p.spot() * (drift + diffusion * z).exp();

    // Welford's streaming mean and variance: constant memory for any path count.
    let (mut count, mut mean, mut m2) = (0u64, 0.0_f64, 0.0_f64);
    let mut add = |x: f64| {
        count += 1;
        let delta = x - mean;
        mean += delta / count as f64;
        m2 += delta * (x - mean);
    };
    if cfg.antithetic {
        for _ in 0..cfg.paths / 2 {
            let z = rng.next_normal();
            add(0.5
                * (payoff(kind, terminal(z), p.strike()) + payoff(kind, terminal(-z), p.strike())));
        }
    } else {
        for _ in 0..cfg.paths {
            add(payoff(kind, terminal(rng.next_normal()), p.strike()));
        }
    }
    if count < 2 {
        return Err(QuantError::InvalidInput {
            name: "paths",
            reason: "too few paths for antithetic sampling",
        });
    }
    let std_error = (m2 / (count - 1) as f64 / count as f64).sqrt();
    Ok(McResult {
        price: disc * mean,
        std_error: disc * std_error,
    })
}

/// Longstaff-Schwartz Monte Carlo price of an American option with `steps`
/// equally spaced exercise dates.
///
/// At each date the continuation value of in-the-money paths is regressed
/// on `1, x, x^2` with `x = S / K`, and the option is exercised where the
/// intrinsic value exceeds the fitted continuation value. The estimator is
/// biased slightly **low** (the exercise rule is sub-optimal and exercise is
/// restricted to the discrete dates). With antithetic sampling the reported
/// standard error treats paths as independent, which is conservative.
///
/// Memory use is about `8 * paths * steps` bytes.
///
/// # Errors
///
/// [`QuantError::InvalidInput`] if `paths < 2`, `steps == 0`, or
/// `paths * steps` exceeds 100 million (the paths are held in memory).
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
    check_config(&cfg)?;
    if steps == 0 {
        return Err(QuantError::InvalidInput {
            name: "steps",
            reason: "must be at least 1",
        });
    }
    let n = cfg.paths;
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
    let disc = (-p.rate() * dt).exp();
    let k = p.strike();

    // paths[i * steps + (t - 1)] = price of path i at date t = 1..=steps.
    let mut rng = Rng::new(cfg.seed);
    let mut s = vec![0.0; n * steps];
    let mut i = 0;
    while i < n {
        let mut cur = p.spot();
        let mut anti = p.spot();
        let pair = cfg.antithetic && i + 1 < n;
        for t in 0..steps {
            let z = rng.next_normal();
            cur *= (drift + diffusion * z).exp();
            s[i * steps + t] = cur;
            if pair {
                anti *= (drift - diffusion * z).exp();
                s[(i + 1) * steps + t] = anti;
            }
        }
        i += if pair { 2 } else { 1 };
    }

    // cash[i]: cash flow of path i, realised at exercise date tau[i].
    let mut cash: Vec<f64> = (0..n)
        .map(|i| payoff(kind, s[i * steps + steps - 1], k))
        .collect();
    let mut tau = vec![steps; n];

    for t in (1..steps).rev() {
        let mut a = [[0.0; 3]; 3];
        let mut b = [0.0; 3];
        let mut itm: Vec<usize> = Vec::new();
        for i in 0..n {
            let st = s[i * steps + t - 1];
            if payoff(kind, st, k) > 0.0 {
                itm.push(i);
                let x = st / k;
                let basis = [1.0, x, x * x];
                let y = cash[i] * disc.powi((tau[i] - t) as i32);
                for r in 0..3 {
                    b[r] += basis[r] * y;
                    for c in 0..3 {
                        a[r][c] += basis[r] * basis[c];
                    }
                }
            }
        }
        if itm.len() < 3 {
            continue;
        }
        let Some(coef) = solve3(a, b) else { continue };
        for &i in &itm {
            let st = s[i * steps + t - 1];
            let x = st / k;
            let continuation = coef[0] + coef[1] * x + coef[2] * x * x;
            let exercise = payoff(kind, st, k);
            if exercise > continuation {
                cash[i] = exercise;
                tau[i] = t;
            }
        }
    }

    let values: Vec<f64> = (0..n).map(|i| cash[i] * disc.powi(tau[i] as i32)).collect();
    let (m, se) = mean_and_se(&values);
    // An immediate-exercise floor: the option is never worth less than today's payoff.
    let intrinsic = payoff(kind, p.spot(), k);
    Ok(McResult {
        price: m.max(intrinsic),
        std_error: se,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{binomial, bsm, ExerciseStyle};

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
}
