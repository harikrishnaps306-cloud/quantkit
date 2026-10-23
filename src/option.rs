//! Typed, validated option inputs.

use crate::error::{check_range, QuantError, Result};
use crate::limits::*;

/// Call or put.
///
/// # Examples
///
/// ```
/// use quantkit::OptionType;
/// assert_ne!(OptionType::Call, OptionType::Put);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionType {
    /// Right to buy the underlying at the strike.
    Call,
    /// Right to sell the underlying at the strike.
    Put,
}

/// Exercise style.
///
/// # Examples
///
/// ```
/// use quantkit::ExerciseStyle;
/// assert_ne!(ExerciseStyle::European, ExerciseStyle::American);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExerciseStyle {
    /// Exercisable only at expiry.
    European,
    /// Exercisable at any time up to expiry.
    American,
}

/// An option contract plus market inputs, *without* a volatility.
///
/// This is what the implied-volatility solver needs. Build it with
/// [`OptionBuilder::build_contract`]. Fields are validated on construction,
/// so every `Contract` in existence is well-formed.
///
/// # Examples
///
/// ```
/// use quantkit::OptionParams;
///
/// let c = OptionParams::builder()
///     .spot(42.0).strike(40.0).rate(0.10).expiry(0.5)
///     .build_contract()?;
/// assert_eq!(c.strike(), 40.0);
/// let p = c.with_volatility(0.2)?;
/// assert_eq!(p.volatility(), 0.2);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "ContractRaw", into = "ContractRaw")
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contract {
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
}

impl Contract {
    /// Single validation point for every way of constructing a contract.
    fn checked(
        spot: f64,
        strike: f64,
        rate: f64,
        dividend_yield: f64,
        expiry: f64,
    ) -> Result<Self> {
        check_range("spot", spot, MIN_PRICE, MAX_PRICE)?;
        check_range("strike", strike, MIN_PRICE, MAX_PRICE)?;
        check_range("expiry", expiry, MIN_EXPIRY, MAX_EXPIRY)?;
        check_range("rate", rate, -MAX_ABS_RATE, MAX_ABS_RATE)?;
        check_range(
            "dividend_yield",
            dividend_yield,
            -MAX_ABS_RATE,
            MAX_ABS_RATE,
        )?;
        if rate.abs() * expiry > MAX_RATE_TIME || dividend_yield.abs() * expiry > MAX_RATE_TIME {
            return Err(QuantError::InvalidInput {
                name: "rate",
                reason: "|rate| * expiry and |dividend_yield| * expiry must not exceed 500",
            });
        }
        Ok(Self {
            spot,
            strike,
            rate,
            dividend_yield,
            expiry,
        })
    }

    /// Spot price of the underlying (> 0).
    pub fn spot(&self) -> f64 {
        self.spot
    }
    /// Strike price (> 0).
    pub fn strike(&self) -> f64 {
        self.strike
    }
    /// Continuously compounded risk-free rate (may be zero or negative).
    pub fn rate(&self) -> f64 {
        self.rate
    }
    /// Continuous dividend yield (may be zero or negative).
    pub fn dividend_yield(&self) -> f64 {
        self.dividend_yield
    }
    /// Time to expiry in years (> 0).
    pub fn expiry(&self) -> f64 {
        self.expiry
    }

    /// Attaches a volatility, producing full [`OptionParams`].
    ///
    /// # Errors
    ///
    /// Returns an error if `volatility` is not finite and strictly positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::OptionParams;
    /// let c = OptionParams::builder().spot(100.0).strike(100.0).expiry(1.0).build_contract()?;
    /// assert!(c.with_volatility(0.0).is_err());
    /// assert!(c.with_volatility(0.25).is_ok());
    /// # Ok::<(), quantkit::QuantError>(())
    /// ```
    pub fn with_volatility(&self, volatility: f64) -> Result<OptionParams> {
        check_range("volatility", volatility, MIN_VOL, MAX_VOL)?;
        Ok(OptionParams {
            contract: *self,
            volatility,
        })
    }
}

/// Validated inputs for pricing: a [`Contract`] plus a volatility.
///
/// Construct with [`OptionParams::builder`].
///
/// # Examples
///
/// ```
/// use quantkit::OptionParams;
///
/// let p = OptionParams::builder()
///     .spot(42.0).strike(40.0).rate(0.10).volatility(0.20).expiry(0.5)
///     .build()?;
/// assert_eq!(p.spot(), 42.0);
/// assert_eq!(p.dividend_yield(), 0.0);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "OptionParamsRaw", into = "OptionParamsRaw")
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionParams {
    contract: Contract,
    volatility: f64,
}

impl OptionParams {
    /// Starts a builder. `spot`, `strike` and `expiry` are required (and
    /// `volatility` for [`OptionBuilder::build`]); `rate` and
    /// `dividend_yield` default to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::OptionParams;
    /// let b = OptionParams::builder().spot(1.0);
    /// assert!(b.build().is_err()); // strike, volatility, expiry missing
    /// ```
    pub fn builder() -> OptionBuilder {
        OptionBuilder::default()
    }
    /// The contract part (everything except volatility).
    pub fn contract(&self) -> &Contract {
        &self.contract
    }
    /// Spot price of the underlying.
    pub fn spot(&self) -> f64 {
        self.contract.spot
    }
    /// Strike price.
    pub fn strike(&self) -> f64 {
        self.contract.strike
    }
    /// Continuously compounded risk-free rate.
    pub fn rate(&self) -> f64 {
        self.contract.rate
    }
    /// Continuous dividend yield.
    pub fn dividend_yield(&self) -> f64 {
        self.contract.dividend_yield
    }
    /// Annualised volatility (> 0).
    pub fn volatility(&self) -> f64 {
        self.volatility
    }
    /// Time to expiry in years.
    pub fn expiry(&self) -> f64 {
        self.contract.expiry
    }

    /// Copy with a different spot (re-validated). Used for bump-and-reprice.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::OptionParams;
    /// let p = OptionParams::builder().spot(100.0).strike(100.0).volatility(0.2).expiry(1.0).build()?;
    /// assert_eq!(p.with_spot(101.0)?.spot(), 101.0);
    /// assert!(p.with_spot(-1.0).is_err());
    /// # Ok::<(), quantkit::QuantError>(())
    /// ```
    pub fn with_spot(&self, spot: f64) -> Result<Self> {
        let c = &self.contract;
        Ok(Self {
            contract: Contract::checked(spot, c.strike, c.rate, c.dividend_yield, c.expiry)?,
            volatility: self.volatility,
        })
    }

    /// Copy with a different volatility (re-validated).
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::OptionParams;
    /// let p = OptionParams::builder().spot(100.0).strike(100.0).volatility(0.2).expiry(1.0).build()?;
    /// assert_eq!(p.with_vol(0.3)?.volatility(), 0.3);
    /// # Ok::<(), quantkit::QuantError>(())
    /// ```
    pub fn with_vol(&self, volatility: f64) -> Result<Self> {
        self.contract.with_volatility(volatility)
    }

    /// Copy with a different rate (re-validated).
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::OptionParams;
    /// let p = OptionParams::builder().spot(100.0).strike(100.0).volatility(0.2).expiry(1.0).build()?;
    /// assert_eq!(p.with_rate(0.05)?.rate(), 0.05);
    /// # Ok::<(), quantkit::QuantError>(())
    /// ```
    pub fn with_rate(&self, rate: f64) -> Result<Self> {
        let c = &self.contract;
        Ok(Self {
            contract: Contract::checked(c.spot, c.strike, rate, c.dividend_yield, c.expiry)?,
            volatility: self.volatility,
        })
    }

    /// Copy with a different time to expiry (re-validated).
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::OptionParams;
    /// let p = OptionParams::builder().spot(100.0).strike(100.0).volatility(0.2).expiry(1.0).build()?;
    /// assert_eq!(p.with_expiry(0.5)?.expiry(), 0.5);
    /// assert!(p.with_expiry(0.0).is_err());
    /// # Ok::<(), quantkit::QuantError>(())
    /// ```
    pub fn with_expiry(&self, expiry: f64) -> Result<Self> {
        let c = &self.contract;
        Ok(Self {
            contract: Contract::checked(c.spot, c.strike, c.rate, c.dividend_yield, expiry)?,
            volatility: self.volatility,
        })
    }
}

/// Builder for [`OptionParams`] and [`Contract`].
///
/// # Examples
///
/// ```
/// use quantkit::OptionParams;
///
/// let p = OptionParams::builder()
///     .spot(100.0).strike(95.0).rate(0.03).dividend_yield(0.01)
///     .volatility(0.25).expiry(1.0)
///     .build()?;
/// assert_eq!(p.dividend_yield(), 0.01);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct OptionBuilder {
    spot: Option<f64>,
    strike: Option<f64>,
    rate: f64,
    dividend_yield: f64,
    volatility: Option<f64>,
    expiry: Option<f64>,
}

impl OptionBuilder {
    /// Sets the spot price (required, > 0).
    pub fn spot(mut self, v: f64) -> Self {
        self.spot = Some(v);
        self
    }
    /// Sets the strike price (required, > 0).
    pub fn strike(mut self, v: f64) -> Self {
        self.strike = Some(v);
        self
    }
    /// Sets the risk-free rate (default 0).
    pub fn rate(mut self, v: f64) -> Self {
        self.rate = v;
        self
    }
    /// Sets the continuous dividend yield (default 0).
    pub fn dividend_yield(mut self, v: f64) -> Self {
        self.dividend_yield = v;
        self
    }
    /// Sets the volatility (required for [`build`](Self::build), > 0).
    pub fn volatility(mut self, v: f64) -> Self {
        self.volatility = Some(v);
        self
    }
    /// Sets the time to expiry in years (required, > 0).
    pub fn expiry(mut self, v: f64) -> Self {
        self.expiry = Some(v);
        self
    }

    /// Validates everything except volatility and returns a [`Contract`].
    ///
    /// # Errors
    ///
    /// [`QuantError::InvalidInput`] for a missing or out-of-range field,
    /// [`QuantError::NonFinite`] for NaN or infinity.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::OptionParams;
    /// let r = OptionParams::builder().spot(f64::NAN).strike(1.0).expiry(1.0).build_contract();
    /// assert!(r.is_err());
    /// ```
    pub fn build_contract(&self) -> Result<Contract> {
        Contract::checked(
            self.spot.ok_or(missing("spot"))?,
            self.strike.ok_or(missing("strike"))?,
            self.rate,
            self.dividend_yield,
            self.expiry.ok_or(missing("expiry"))?,
        )
    }

    /// Validates all fields and returns [`OptionParams`].
    ///
    /// # Errors
    ///
    /// Same as [`build_contract`](Self::build_contract), plus an error if
    /// the volatility is missing, non-finite or not strictly positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::OptionParams;
    /// let r = OptionParams::builder().spot(100.0).strike(100.0).volatility(0.2).expiry(0.0).build();
    /// assert!(r.is_err()); // zero time to expiry
    /// ```
    pub fn build(&self) -> Result<OptionParams> {
        let contract = self.build_contract()?;
        let vol = self.volatility.ok_or(missing("volatility"))?;
        contract.with_volatility(vol)
    }
}

/// Plain-field mirror of [`Contract`] used for (de)serialization, so that
/// deserialized values go through the same validation as the builder.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct ContractRaw {
    spot: f64,
    strike: f64,
    rate: f64,
    #[serde(default)]
    dividend_yield: f64,
    expiry: f64,
}

#[cfg(feature = "serde")]
impl From<Contract> for ContractRaw {
    fn from(c: Contract) -> Self {
        Self {
            spot: c.spot,
            strike: c.strike,
            rate: c.rate,
            dividend_yield: c.dividend_yield,
            expiry: c.expiry,
        }
    }
}

#[cfg(feature = "serde")]
impl TryFrom<ContractRaw> for Contract {
    type Error = QuantError;
    fn try_from(r: ContractRaw) -> Result<Self> {
        OptionParams::builder()
            .spot(r.spot)
            .strike(r.strike)
            .rate(r.rate)
            .dividend_yield(r.dividend_yield)
            .expiry(r.expiry)
            .build_contract()
    }
}

/// Plain-field mirror of [`OptionParams`] used for (de)serialization.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct OptionParamsRaw {
    spot: f64,
    strike: f64,
    rate: f64,
    #[serde(default)]
    dividend_yield: f64,
    volatility: f64,
    expiry: f64,
}

#[cfg(feature = "serde")]
impl From<OptionParams> for OptionParamsRaw {
    fn from(p: OptionParams) -> Self {
        Self {
            spot: p.spot(),
            strike: p.strike(),
            rate: p.rate(),
            dividend_yield: p.dividend_yield(),
            volatility: p.volatility(),
            expiry: p.expiry(),
        }
    }
}

#[cfg(feature = "serde")]
impl TryFrom<OptionParamsRaw> for OptionParams {
    type Error = QuantError;
    fn try_from(r: OptionParamsRaw) -> Result<Self> {
        OptionParams::builder()
            .spot(r.spot)
            .strike(r.strike)
            .rate(r.rate)
            .dividend_yield(r.dividend_yield)
            .volatility(r.volatility)
            .expiry(r.expiry)
            .build()
    }
}

fn missing(name: &'static str) -> QuantError {
    QuantError::InvalidInput {
        name,
        reason: "not set",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> OptionBuilder {
        OptionParams::builder()
            .spot(100.0)
            .strike(100.0)
            .volatility(0.2)
            .expiry(1.0)
    }

    #[test]
    fn valid_builds() {
        assert!(base().build().is_ok());
    }

    #[test]
    fn rejects_bad_values() {
        assert!(base().spot(0.0).build().is_err());
        assert!(base().spot(-1.0).build().is_err());
        assert!(base().strike(0.0).build().is_err());
        assert!(base().volatility(-0.1).build().is_err());
        assert!(base().volatility(0.0).build().is_err());
        assert!(base().expiry(0.0).build().is_err());
        assert!(base().expiry(-1.0).build().is_err());
    }

    #[test]
    fn rejects_non_finite() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(base().spot(bad).build().is_err());
            assert!(base().strike(bad).build().is_err());
            assert!(base().rate(bad).build().is_err());
            assert!(base().dividend_yield(bad).build().is_err());
            assert!(base().volatility(bad).build().is_err());
            assert!(base().expiry(bad).build().is_err());
        }
    }

    #[test]
    fn missing_fields_error() {
        assert!(OptionParams::builder().build().is_err());
        assert!(base().build_contract().is_ok());
        let no_vol = OptionParams::builder().spot(1.0).strike(1.0).expiry(1.0);
        assert!(no_vol.build().is_err());
        assert!(no_vol.build_contract().is_ok());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip_and_validation() {
        let p = base().rate(0.03).dividend_yield(0.01).build().unwrap();
        let json = serde_json::to_string(&p).unwrap();
        let back: OptionParams = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
        // dividend_yield is optional on input.
        let ok = r#"{"spot":100,"strike":100,"rate":0.05,"volatility":0.2,"expiry":1}"#;
        assert!(serde_json::from_str::<OptionParams>(ok).is_ok());
        // Invalid values are rejected on deserialize, not smuggled in.
        let bad = r#"{"spot":100,"strike":100,"rate":0.05,"volatility":-0.2,"expiry":1}"#;
        assert!(serde_json::from_str::<OptionParams>(bad).is_err());
        let zero_expiry = r#"{"spot":1,"strike":1,"rate":0,"expiry":0}"#;
        assert!(serde_json::from_str::<Contract>(zero_expiry).is_err());
    }

    #[test]
    fn negative_rate_allowed() {
        assert!(base().rate(-0.01).build().is_ok());
    }
}
