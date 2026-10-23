//! Computes risk metrics on a small, made-up series of daily returns.
//!
//! The returns below are synthetic and only illustrate the API.
//!
//! Run with `cargo run --example var_from_returns`.

use quantkit::risk;

fn main() -> Result<(), quantkit::QuantError> {
    let daily_returns = [
        0.0042, -0.0110, 0.0067, 0.0013, -0.0231, 0.0150, -0.0048, 0.0091, 0.0005, -0.0172, 0.0120,
        -0.0034, 0.0078, -0.0089, 0.0023, 0.0056, -0.0145, 0.0101, -0.0012, 0.0033,
    ];

    println!(
        "95% historical VaR : {:.4}",
        risk::historical_var(&daily_returns, 0.95)?
    );
    println!(
        "95% historical CVaR: {:.4}",
        risk::historical_cvar(&daily_returns, 0.95)?
    );
    println!(
        "95% parametric VaR : {:.4}",
        risk::parametric_var(&daily_returns, 0.95)?
    );
    println!(
        "Sharpe (rf = 0)    : {:.3}",
        risk::sharpe_ratio(&daily_returns, 0.0, 252.0)?
    );
    println!(
        "Sortino (MAR = 0)  : {:.3}",
        risk::sortino_ratio(&daily_returns, 0.0, 252.0)?
    );
    println!(
        "Max drawdown       : {:.4}",
        risk::max_drawdown(&daily_returns)?
    );
    println!(
        "Annualised vol     : {:.4}",
        risk::annualized_volatility(&daily_returns, 252.0)?
    );
    Ok(())
}
