//! The normalised Black function, evaluated stably for out-of-the-money options.
//!
//! For forward `F`, strike `K`, total volatility `s = sigma sqrt(T)` and
//! `x = ln(F / K)`, the Black call price divided by `sqrt(F K)` (and by the
//! discount factor) is
//!
//! ```text
//! b(x, s) = e^{x/2} Phi(x/s + s/2) - e^{-x/2} Phi(x/s - s/2)
//! ```
//!
//! For `x <= 0` this is an out-of-the-money call, and by put-call symmetry
//! the same number is the out-of-the-money *put* at `-x`. In-the-money prices
//! are intrinsic value plus this quantity, so only `x <= 0` is needed.
//!
//! # Why this module exists
//!
//! Evaluating `b` as written subtracts two nearly equal numbers once `|x| / s`
//! is large, so the relative error grows like `eps * d^2 * cancellation`.
//! (Measured against 60-digit arithmetic, the plain formula was off by 1.4e-10
//! at `|d1| = 30`.) With Mills' ratio `M(u) = Phi(-u) / phi(u)` and
//! `h = x / s`, `t = s / 2`,
//!
//! ```text
//! b = P (M(u1) - M(u1 + s)),   P = exp(-(h^2 + t^2) / 2) / sqrt(2 pi),   u1 = -h - t
//! ```
//!
//! and, because `M'(v) = v M(v) - 1`,
//!
//! ```text
//! b = P * integral_{u1}^{u1+s} g(v) dv,   g(v) = 1 - v M(v) > 0.
//! ```
//!
//! The integrand is positive and smooth, so Gauss-Legendre quadrature gives
//! full relative accuracy with no cancellation, and `ln b = ln P + ln(integral)`
//! never underflows. `g(v)` itself is computed without cancellation from the
//! Laplace continued fraction for `M` when `v >= 3`.
//!
//! The ordinary formula is still used when an error estimate says it is
//! accurate to about 2e-14 (the common case), because it is far cheaper.
//!
//! The quantities here are independent of any option's rate, dividend and
//! currency conventions; [`crate::bsm`] and [`crate::implied_vol`] build on them.

use std::sync::OnceLock;

const SQRT_2: f64 = std::f64::consts::SQRT_2;
/// `ln(sqrt(2 pi))`.
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;
/// `sqrt(pi / 2)`.
const SQRT_PI_2: f64 = 1.253_314_137_315_500_3;
/// Beyond this `v`, `g` and `M` use the continued fraction.
const CF_START: f64 = 3.0;
const GL_N: usize = 16;
/// The plain formula is used when `cancellation * curvature` is at most this,
/// which bounds its relative error by about `FAST_LIMIT * eps` (2e-14).
const FAST_LIMIT: f64 = 100.0;

fn cdf(z: f64) -> f64 {
    0.5 * libm::erfc(-z / SQRT_2)
}

/// Backward evaluation of the continued fraction
/// `M(v) = 1 / (v + 1 / (v + 2 / (v + 3 / (v + ...))))`, returning `(Y0, Y1)`
/// where `Y_k = v + (k + 1) / Y_{k+1}`, so `M = 1 / Y0` and `g = 1 / (Y0 Y1)`.
fn cf(v: f64) -> (f64, f64) {
    // The truncation error of the fraction falls roughly like exp(-2 v sqrt(2 N));
    // N = 10 + 400 / v^2 puts it below 1e-30 for every v >= 3.
    let depth = 10 + (400.0 / (v * v)) as usize;
    let mut y = v;
    let mut y1 = v;
    for k in (0..depth).rev() {
        y = v + (k as f64 + 1.0) / y;
        if k == 1 {
            y1 = y;
        }
    }
    (y, y1)
}

/// Mills ratio `Phi(-u) / phi(u)`, accurate for `u >= CF_START`.
fn mills_cf(u: f64) -> f64 {
    1.0 / cf(u).0
}

/// `g(v) = 1 - v M(v)` for any real `v >= -1` (used on `[-1, inf)`).
fn g(v: f64) -> f64 {
    if v >= CF_START {
        let (y0, y1) = cf(v);
        1.0 / (y0 * y1)
    } else {
        // M(v) = sqrt(pi/2) e^{v^2/2} erfc(v / sqrt 2); at most a few for v >= -1.
        1.0 - v * SQRT_PI_2 * libm::exp(0.5 * v * v) * libm::erfc(v / SQRT_2)
    }
}

/// `ln Phi(-u)` for any real `u` (including large positive `u`, where the
/// value underflows an `f64`).
fn ln_phi_neg(u: f64) -> f64 {
    if u > CF_START {
        -0.5 * u * u - LN_SQRT_2PI + libm::log(mills_cf(u))
    } else {
        libm::log(0.5 * libm::erfc(u / SQRT_2))
    }
}

fn gauss_legendre() -> &'static ([f64; GL_N], [f64; GL_N]) {
    static NODES: OnceLock<([f64; GL_N], [f64; GL_N])> = OnceLock::new();
    NODES.get_or_init(|| {
        let n = GL_N as f64;
        let mut x = [0.0; GL_N];
        let mut w = [0.0; GL_N];
        let legendre = |z: f64| {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..GL_N {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) as f64 * z * p2 - j as f64 * p3) / (j + 1) as f64;
            }
            (p1, n * (z * p1 - p2) / (z * z - 1.0))
        };
        for i in 0..GL_N {
            let mut z = libm::cos(std::f64::consts::PI * (i as f64 + 0.75) / (n + 0.5));
            for _ in 0..100 {
                let (p, dp) = legendre(z);
                let next = z - p / dp;
                let done = (next - z).abs() < 1e-16;
                z = next;
                if done {
                    break;
                }
            }
            let (_, dp) = legendre(z);
            x[i] = z;
            w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
        }
        (x, w)
    })
}

/// `integral_a^(a+width) g(v) dv` by Gauss-Legendre on panels whose width grows
/// with their position, matching the `1/v^2` decay of `g`.
///
/// The interval is described by its start and its *width* rather than by two
/// endpoints: for a narrow interval far from zero, `end - start` computed from
/// rounded endpoints would lose most of its digits.
fn integrate_g(a: f64, width: f64) -> f64 {
    let (nodes, weights) = gauss_legendre();
    let mut total = 0.0;
    let mut offset = 0.0;
    while offset < width {
        let end = (offset + (a + offset).max(1.0)).min(width);
        let (half, mid) = (0.5 * (end - offset), a + 0.5 * (end + offset));
        let mut panel = 0.0;
        for (x, w) in nodes.iter().zip(weights) {
            panel += w * g(mid + half * x);
        }
        total += half * panel;
        offset = end;
    }
    total
}

/// `ln b(x, s)` for `x < 0` by the stable representation. Valid for every
/// finite `x < 0` and `s > 0`; never underflows.
fn ln_b_stable(x: f64, s: f64) -> f64 {
    let h = x / s;
    let t = 0.5 * s;
    let u1 = -h - t;
    let u2 = u1 + s;
    if s >= 2.0 && u1 <= 5.0 {
        // The two terms differ by a modest factor here, so subtract them in logs.
        let ln_t1 = 0.5 * x + ln_phi_neg(u1);
        let ln_t2 = -0.5 * x + ln_phi_neg(u2);
        let ratio = libm::exp(ln_t2 - ln_t1);
        if ratio < 1.0 {
            return ln_t1 + libm::log1p(-ratio);
        }
    }
    let ln_p = -0.5 * (h * h + t * t) - LN_SQRT_2PI;
    ln_p + libm::log(integrate_g(u1, s))
}

/// The plain formula, with an error estimate; `None` when it is not trusted
/// to about `2e-14` relative accuracy.
fn b_fast(x: f64, s: f64) -> Option<f64> {
    let d1 = x / s + 0.5 * s;
    let d2 = d1 - s;
    let t1 = libm::exp(0.5 * x) * cdf(d1);
    let t2 = libm::exp(-0.5 * x) * cdf(d2);
    let b = t1 - t2;
    if b.is_nan() || b <= 1e-290 {
        return None;
    }
    let cancellation = (t1 + t2) / b;
    let curvature = 1.0 + 0.5 * (d1 * d1).max(d2 * d2);
    (cancellation * curvature <= FAST_LIMIT).then_some(b)
}

/// The normalised Black function `b(x, s)` for `x <= 0` and `s > 0`: the
/// out-of-the-money option value per unit of `discount * sqrt(F K)`.
///
/// Relative accuracy is about `eps * (1 + (x/s)^2)` over the whole domain,
/// which is the conditioning of the problem itself because `x` and `s` are
/// already rounded `f64` values. The result may be subnormal or `0.0` for
/// extreme tails; use [`ln_normalised_otm`] to keep going past that.
///
/// # Panics
///
/// Does not panic. Arguments outside `x <= 0`, `s > 0` give a meaningless
/// (possibly `NaN`) result.
///
/// # Examples
///
/// ```
/// use quantkit::black::normalised_otm;
/// // At the money b = erf(s / (2 sqrt 2)); with s = 0.2 that is about 0.0797.
/// assert!((normalised_otm(0.0, 0.2) - 0.079_655_674_554_057_6).abs() < 1e-15);
/// // Far out of the money the value is tiny but still accurate.
/// assert!(normalised_otm(-0.6, 0.02) < 1e-190);
/// ```
pub fn normalised_otm(x: f64, s: f64) -> f64 {
    if x == 0.0 {
        return libm::erf(s / (2.0 * SQRT_2));
    }
    b_fast(x, s).unwrap_or_else(|| libm::exp(ln_b_stable(x, s)))
}

/// `ln b(x, s)` for `x <= 0` and `s > 0`, valid even where `b` underflows an
/// `f64`. Same accuracy contract as [`normalised_otm`] (absolute error in the
/// logarithm is the relative error of `b`).
///
/// # Examples
///
/// ```
/// use quantkit::black::{ln_normalised_otm, normalised_otm};
/// let (x, s) = (-0.3, 0.1);
/// assert!((ln_normalised_otm(x, s) - normalised_otm(x, s).ln()).abs() < 1e-12);
/// // b is about e^-450 here: far below the smallest f64, yet the log is fine.
/// assert!(ln_normalised_otm(-2.0, 0.1).is_finite());
/// ```
pub fn ln_normalised_otm(x: f64, s: f64) -> f64 {
    if x == 0.0 {
        return libm::log(libm::erf(s / (2.0 * SQRT_2)));
    }
    match b_fast(x, s) {
        Some(b) => libm::log(b),
        None => ln_b_stable(x, s),
    }
}

/// `ln` of the normalised vega `db/ds = exp(-x^2 / (2 s^2) - s^2 / 8) / sqrt(2 pi)`.
///
/// # Examples
///
/// ```
/// use quantkit::black::ln_normalised_vega;
/// let v = ln_normalised_vega(0.0, 1e-9).exp();
/// assert!((v - 0.398_942_280_401_432_7).abs() < 1e-12);
/// ```
pub fn ln_normalised_vega(x: f64, s: f64) -> f64 {
    -LN_SQRT_2PI - 0.5 * x * x / (s * s) - 0.125 * s * s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_is_exact_for_polynomials() {
        let (x, w) = gauss_legendre();
        assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-15);
        // Exact up to degree 2N - 1 = 31.
        for k in [0, 1, 2, 7, 16, 30, 31] {
            let q: f64 = x.iter().zip(w).map(|(x, w)| w * x.powi(k)).sum();
            let exact = if k % 2 == 1 {
                0.0
            } else {
                2.0 / (k as f64 + 1.0)
            };
            assert!((q - exact).abs() < 2e-15, "degree {k}: {q} vs {exact}");
        }
    }

    #[test]
    fn g_matches_its_definition_and_is_continuous_at_the_switch() {
        assert!((g(0.0) - 1.0).abs() < 1e-15);
        // Large v: g ~ sum (-1)^(k+1) (2k-1)!! / v^(2k) (asymptotic series, independent derivation).
        let v: f64 = 40.0;
        let series = 1.0 / v.powi(2) - 3.0 / v.powi(4) + 15.0 / v.powi(6) - 105.0 / v.powi(8)
            + 945.0 / v.powi(10)
            - 10395.0 / v.powi(12);
        assert!((g(v) - series).abs() / series < 1e-13);
        // The two evaluation methods agree at the switch point.
        let direct = 1.0
            - CF_START
                * SQRT_PI_2
                * libm::exp(0.5 * CF_START * CF_START)
                * libm::erfc(CF_START / SQRT_2);
        assert!((g(CF_START) - direct).abs() / direct < 1e-13);
        assert!(g(-1.0) > 1.0);
    }

    #[test]
    fn stable_and_fast_branches_agree_where_both_are_valid() {
        for &(x, s) in &[
            (-0.05, 0.2),
            (-0.5, 0.5),
            (-1.0, 1.0),
            (-2.0, 3.0),
            (-0.01, 0.1),
        ] {
            let fast = b_fast(x, s).expect("fast path should be trusted here");
            let stable = libm::exp(ln_b_stable(x, s));
            assert!(
                (fast - stable).abs() / stable < 1e-13,
                "x={x} s={s}: {fast} vs {stable}"
            );
        }
    }

    #[test]
    fn at_the_money_uses_erf_and_is_monotone_in_s() {
        let mut prev = 0.0;
        for i in 1..50 {
            let b = normalised_otm(0.0, f64::from(i) * 0.1);
            assert!(b > prev && b < 1.0);
            prev = b;
        }
    }

    #[test]
    fn increases_in_s_and_decreases_in_abs_x() {
        for &x in &[-1e-9, -0.01, -0.5, -5.0, -40.0] {
            let mut prev = 0.0;
            for &s in &[1e-3, 1e-2, 0.1, 0.5, 1.0, 1.99, 2.01, 4.0, 20.0] {
                let lb = ln_normalised_otm(x, s);
                assert!(lb.is_finite());
                assert!(
                    lb >= prev || prev == 0.0 || lb > prev - 1e-12,
                    "x={x} s={s}"
                );
                prev = lb;
            }
        }
        assert!(ln_normalised_otm(-1.0, 0.5) < ln_normalised_otm(-0.5, 0.5));
    }

    #[test]
    fn vega_is_the_derivative_of_b() {
        for &(x, s) in &[(-0.3, 0.4), (-0.01, 0.2), (-2.0, 1.5)] {
            let h = 1e-5 * s;
            let fd = (normalised_otm(x, s + h) - normalised_otm(x, s - h)) / (2.0 * h);
            let v = ln_normalised_vega(x, s).exp();
            assert!((fd - v).abs() / v < 1e-7, "x={x} s={s}: {fd} vs {v}");
        }
    }
}
