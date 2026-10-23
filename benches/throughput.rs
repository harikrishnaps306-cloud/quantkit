//! Throughput benchmarks for batch, Monte Carlo and dataset generation.
//!
//! Run serial and parallel builds separately and compare:
//!
//! ```text
//! cargo bench --bench throughput
//! cargo bench --bench throughput --features parallel
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use quantkit::detailed::{self, RawOption, RawQuote};
use quantkit::implied_vol::SolverConfig;
use quantkit::mc::{self, McConfig};
use quantkit::{batch, bsm, dataset, OptionParams, OptionType};

fn grid_params(n_target: usize) -> Vec<OptionParams> {
    let side = (n_target as f64).cbrt().ceil() as usize;
    let lin = |lo: f64, hi: f64| -> Vec<f64> {
        (0..side)
            .map(|i| lo + (hi - lo) * i as f64 / (side - 1).max(1) as f64)
            .collect()
    };
    let mut v = batch::grid(
        &lin(70.0, 130.0),
        &lin(80.0, 120.0),
        &lin(0.1, 2.0),
        &[0.2],
        0.03,
        0.01,
    )
    .unwrap();
    v.truncate(n_target);
    v
}

fn bench_batch(c: &mut Criterion) {
    let params = grid_params(100_000);
    let n = params.len() as u64;
    let mut g = c.benchmark_group("batch");
    g.throughput(Throughput::Elements(n));
    g.bench_function("price_100k", |b| {
        b.iter(|| batch::price(black_box(&params), OptionType::Call))
    });
    g.bench_function("greeks_100k", |b| {
        b.iter(|| batch::greeks(black_box(&params), OptionType::Call))
    });

    let raw: Vec<RawOption> = params
        .iter()
        .map(|p| RawOption {
            spot: p.spot(),
            strike: p.strike(),
            rate: p.rate(),
            dividend_yield: p.dividend_yield(),
            volatility: p.volatility(),
            expiry: p.expiry(),
        })
        .collect();
    g.bench_function("detailed_price_100k", |b| {
        b.iter(|| detailed::price(black_box(&raw), OptionType::Call))
    });

    let quotes: Vec<RawQuote> = params
        .iter()
        .take(20_000)
        .map(|p| RawQuote {
            spot: p.spot(),
            strike: p.strike(),
            rate: p.rate(),
            dividend_yield: p.dividend_yield(),
            expiry: p.expiry(),
            price: bsm::price(p, OptionType::Call),
        })
        .collect();
    g.throughput(Throughput::Elements(quotes.len() as u64));
    g.bench_function("detailed_implied_vol_20k", |b| {
        b.iter(|| {
            detailed::implied_vol(
                black_box(&quotes),
                OptionType::Call,
                SolverConfig::default(),
            )
        })
    });
    g.finish();
}

fn bench_mc(c: &mut Criterion) {
    let p = OptionParams::builder()
        .spot(100.0)
        .strike(105.0)
        .rate(0.03)
        .volatility(0.25)
        .expiry(1.0)
        .build()
        .unwrap();
    let mut g = c.benchmark_group("monte_carlo");
    g.sample_size(20);
    let cfg = McConfig {
        paths: 1_000_000,
        seed: 1,
        antithetic: true,
    };
    g.throughput(Throughput::Elements(cfg.paths as u64));
    g.bench_function("european_1M_paths", |b| {
        b.iter(|| mc::european(black_box(&p), OptionType::Call, cfg).unwrap())
    });
    let lsm = McConfig {
        paths: 20_000,
        seed: 1,
        antithetic: true,
    };
    g.bench_function("lsm_20k_train_20k_eval_50_steps", |b| {
        b.iter(|| mc::american_lsm(black_box(&p), OptionType::Put, 50, lsm).unwrap())
    });
    g.finish();
}

fn bench_dataset(c: &mut Criterion) {
    let spec = dataset::DatasetSpec {
        rows: 100_000,
        seed: 1,
        ..dataset::DatasetSpec::default()
    };
    let mut g = c.benchmark_group("dataset");
    g.sample_size(20);
    g.throughput(Throughput::Elements(spec.rows as u64));
    g.bench_function("generate_100k_rows", |b| {
        b.iter(|| dataset::generate(black_box(&spec)).unwrap())
    });
    g.finish();
}

criterion_group!(benches, bench_batch, bench_mc, bench_dataset);
criterion_main!(benches);
