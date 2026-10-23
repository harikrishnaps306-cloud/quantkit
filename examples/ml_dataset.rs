//! Generates a reproducible synthetic option-pricing dataset and prints its
//! audit trail.
//!
//! Run with `cargo run --example ml_dataset`.

use quantkit::dataset::{generate, DatasetSpec, Sampling};

fn main() -> Result<(), quantkit::QuantError> {
    let spec = DatasetSpec {
        rows: 1_000,
        seed: 2026,
        sampling: Sampling::LatinHypercube,
        ..DatasetSpec::default()
    };
    let ds = generate(&spec)?;
    let m = &ds.metadata;
    println!(
        "{} rows ({} sampled + {} boundary), quantkit {}",
        ds.rows.len(),
        m.sampled_rows,
        m.boundary_rows,
        m.quantkit_version
    );
    println!("label model : {}", m.label_model);
    println!("conventions : {}", m.conventions);
    println!("checksum    : {:#018x}", m.checksum);
    println!(
        "invariants  : {} rows checked, all passed = {}, max parity error {:.2e}",
        m.invariants.rows_checked,
        m.invariants.all_passed(),
        m.invariants.max_parity_error
    );

    // Same spec, same bytes: that is the reproducibility contract.
    assert_eq!(generate(&spec)?.metadata.checksum, m.checksum);

    // First rows as CSV (use `ds.to_csv()` for the whole table).
    for line in ds.to_csv().lines().take(4) {
        println!("{line}");
    }
    Ok(())
}
