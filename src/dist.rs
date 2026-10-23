//! Standard normal distribution helpers.
//!
//! # Why `libm::erfc`
//!
//! The CDF is `Phi(x) = 0.5 * erfc(-x / sqrt(2))`. `erfc` is not in `core`/`std`
//! on stable Rust, and the textbook shortcuts (Abramowitz-Stegun 26.2.17,
//! ~1e-7 accuracy) are too coarse for implied-volatility solving and for
//! far-from-the-money prices. The pure-Rust, dependency-free `libm` crate
//! ports musl's `erfc`, which is accurate to about 1 ulp, including in the
//! tails (relative accuracy, because `erfc` is evaluated directly rather than
//! as `1 - erf`). It is a tiny, widely used crate, so it was preferred to
//! hand-rolled coefficients.

use crate::error::{QuantError, Result};

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const SQRT_2PI: f64 = 2.506_628_274_631_000_7;
const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Standard normal probability density `phi(x)`.
///
/// NaN in gives NaN out.
///
/// # Examples
///
/// ```
/// use quantkit::dist::norm_pdf;
/// assert!((norm_pdf(0.0) - 0.398_942_280_401_432_7).abs() < 1e-15);
/// ```
pub fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal cumulative distribution `Phi(x)`.
///
/// Accurate to a few ulp over the whole real line, with full relative
/// accuracy in the lower tail. NaN in gives NaN out.
///
/// # Examples
///
/// ```
/// use quantkit::dist::norm_cdf;
/// assert_eq!(norm_cdf(0.0), 0.5);
/// assert!((norm_cdf(1.96) - 0.975).abs() < 1e-4);
/// ```
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x * FRAC_1_SQRT_2)
}

/// Inverse of the standard normal CDF (the probit / quantile function).
///
/// Uses Acklam's rational approximation (relative error about 1e-9)
/// followed by one Halley refinement step against [`norm_cdf`], which brings
/// it to near machine precision.
///
/// # Errors
///
/// [`QuantError::NonFinite`] for NaN, and [`QuantError::InvalidInput`]
/// unless `0 < p < 1`.
///
/// # Examples
///
/// ```
/// use quantkit::dist::norm_inv_cdf;
/// let z = norm_inv_cdf(0.975)?;
/// assert!((z - 1.959_963_984_540_054).abs() < 1e-12);
/// assert!(norm_inv_cdf(1.0).is_err());
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn norm_inv_cdf(p: f64) -> Result<f64> {
    if p.is_nan() {
        return Err(QuantError::NonFinite { name: "p" });
    }
    if p <= 0.0 || p >= 1.0 {
        return Err(QuantError::InvalidInput {
            name: "p",
            reason: "must lie strictly between 0 and 1",
        });
    }

    // Peter J. Acklam's coefficients.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;

    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley step.
    let e = norm_cdf(x) - p;
    let u = e * SQRT_2PI * (0.5 * x * x).exp();
    Ok(x - u / (1.0 + 0.5 * x * u))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values below were computed independently in double precision
    // with CPython 3's `0.5 * math.erfc(-x / math.sqrt(2))` and
    // `statistics.NormalDist().inv_cdf(p)`; the four-decimal table entries
    // (Phi(1.96) = 0.9750 etc.) agree with any standard normal table.
    const CDF_REF: [(f64, f64); 6] = [
        (0.0, 0.5),
        (1.0, 0.841_344_746_068_542_8),
        (1.96, 0.975_002_104_851_779_6),
        (-1.645, 0.049_984_905_539_121_355),
        (2.0, 0.977_249_868_051_820_9),
        (-3.0, 0.001_349_898_031_630_095_9),
    ];

    #[test]
    fn cdf_matches_reference() {
        for (x, want) in CDF_REF {
            let got = norm_cdf(x);
            assert!(
                (got - want).abs() <= 1e-15 + 4.0 * f64::EPSILON * want,
                "x={x}"
            );
        }
    }

    #[test]
    fn cdf_symmetry_and_tails() {
        for i in -60..=60 {
            let x = f64::from(i) / 10.0;
            assert!((norm_cdf(x) + norm_cdf(-x) - 1.0).abs() < 1e-15);
        }
        assert_eq!(norm_cdf(f64::INFINITY), 1.0);
        assert_eq!(norm_cdf(f64::NEG_INFINITY), 0.0);
        assert!(norm_cdf(f64::NAN).is_nan());
        // Phi(-10) ~ 7.62e-24: relative accuracy in the tail.
        assert!((norm_cdf(-10.0) / 7.619_853_024_160_595e-24 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pdf_values() {
        assert!((norm_pdf(1.0) - 0.241_970_724_519_143_37).abs() < 1e-15);
        assert_eq!(norm_pdf(2.0), norm_pdf(-2.0));
    }

    #[test]
    fn inv_cdf_matches_reference() {
        let refs = [
            (0.95, 1.644_853_626_951_471_5),
            (0.99, 2.326_347_874_040_840_8),
            (0.975, 1.959_963_984_540_053_6),
            (0.05, -1.644_853_626_951_472_6),
        ];
        for (p, want) in refs {
            assert!((norm_inv_cdf(p).unwrap() - want).abs() < 1e-12, "p={p}");
        }
    }

    #[test]
    fn inv_cdf_round_trip() {
        for p in [
            1e-12,
            1e-6,
            0.001,
            0.02,
            0.3,
            0.5,
            0.7,
            0.98,
            0.999,
            1.0 - 1e-9,
        ] {
            let back = norm_cdf(norm_inv_cdf(p).unwrap());
            assert!(
                (back - p).abs() <= 1e-13 * p.min(1.0 - p).max(1e-3),
                "p={p}"
            );
        }
    }

    #[test]
    fn inv_cdf_rejects_bad_p() {
        for p in [0.0, 1.0, -0.1, 1.5, f64::NAN, f64::INFINITY] {
            assert!(norm_inv_cdf(p).is_err());
        }
    }
}
