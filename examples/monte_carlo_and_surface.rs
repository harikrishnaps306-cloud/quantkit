//! Monte Carlo pricing, a volatility surface built from option prices, and an
//! SVI smile fit.
//!
//! Run with `cargo run --example monte_carlo_and_surface`.

use quantkit::mc::{self, McConfig};
use quantkit::surface::{fit_svi, PriceQuote, VolSurface};
use quantkit::{bsm, OptionParams, OptionType};

fn main() -> Result<(), quantkit::QuantError> {
    // 1. Monte Carlo against the closed form.
    let p = OptionParams::builder()
        .spot(100.0)
        .strike(105.0)
        .rate(0.03)
        .volatility(0.25)
        .expiry(1.0)
        .build()?;
    let est = mc::european(&p, OptionType::Call, McConfig::default())?;
    let (lo, hi) = est.confidence_interval(1.96);
    println!(
        "MC call {:.4} (95% CI {lo:.4}..{hi:.4}), BSM {:.4}",
        est.price,
        bsm::call_price(&p)
    );
    let american = mc::american_lsm(&p, OptionType::Put, 50, McConfig::default())?;
    println!(
        "LSM American put {:.4} +/- {:.4}",
        american.price, american.std_error
    );

    // 2. A surface built from synthetic option prices with a smile.
    let smile = |k: f64| 0.20 + 0.15 * (k / 100.0).ln().powi(2);
    let mut quotes = Vec::new();
    for expiry in [0.25, 1.0] {
        for strike in [80.0, 90.0, 100.0, 110.0, 120.0] {
            let kind = if strike < 100.0 {
                OptionType::Put
            } else {
                OptionType::Call
            };
            let q = OptionParams::builder()
                .spot(100.0)
                .strike(strike)
                .rate(0.02)
                .volatility(smile(strike))
                .expiry(expiry)
                .build()?;
            quotes.push(PriceQuote {
                strike,
                expiry,
                kind,
                price: bsm::price(&q, kind),
            });
        }
    }
    let surface = VolSurface::from_quotes(100.0, 0.02, 0.0, &quotes)?;
    println!(
        "Interpolated vol at K=95, T=0.6: {:.4}",
        surface.vol(95.0, 0.6)?
    );
    println!("Arbitrage clean: {}", surface.check_arbitrage().is_clean());

    // 3. Fit a smooth SVI smile to total-variance points.
    let ks: Vec<f64> = (-8..=8).map(|i| f64::from(i) / 10.0).collect();
    let ws: Vec<f64> = ks
        .iter()
        .map(|&k| 0.04 + 0.3 * (0.2 * k + (k * k + 0.01).sqrt()))
        .collect();
    let fit = fit_svi(&ks, &ws)?;
    println!("SVI fit rmse {:.2e}, params {:?}", fit.rmse, fit.params);
    Ok(())
}
