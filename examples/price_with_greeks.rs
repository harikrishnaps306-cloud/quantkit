//! Prices a European and an American option and prints Greeks and the
//! implied volatility.
//!
//! Run with `cargo run --example price_with_greeks`.

use quantkit::{binomial, bsm, greeks, implied_vol, ExerciseStyle, OptionParams, OptionType};

fn main() -> Result<(), quantkit::QuantError> {
    // Hull's textbook example: S=42, K=40, r=10%, sigma=20%, T=0.5.
    let p = OptionParams::builder()
        .spot(42.0)
        .strike(40.0)
        .rate(0.10)
        .volatility(0.20)
        .expiry(0.5)
        .build()?;

    let call = bsm::price(&p, OptionType::Call);
    let put = bsm::price(&p, OptionType::Put);
    println!("European call: {call:.4}   put: {put:.4}");

    let g = greeks::analytic(&p, OptionType::Call);
    println!(
        "Call Greeks  delta {:.4}  gamma {:.4}  vega {:.4}  theta/yr {:.4}  rho {:.4}",
        g.delta, g.gamma, g.vega, g.theta, g.rho
    );

    let american_put = binomial::price(&p, OptionType::Put, ExerciseStyle::American, 500)?;
    println!("American put (500 steps): {american_put:.4}");

    let iv = implied_vol::solve(p.contract(), OptionType::Call, call)?;
    println!("Implied vol recovered from the call price: {iv:.6}");
    Ok(())
}
