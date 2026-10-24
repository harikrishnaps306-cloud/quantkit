use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quantkit::{binomial, bsm, dist, implied_vol, risk, ExerciseStyle, OptionParams, OptionType};

fn params() -> OptionParams {
    OptionParams::builder()
        .spot(100.0)
        .strike(105.0)
        .rate(0.03)
        .dividend_yield(0.01)
        .volatility(0.25)
        .expiry(0.75)
        .build()
        .unwrap()
}

fn bench_bsm(c: &mut Criterion) {
    let p = params();
    c.bench_function("bsm_call_price", |b| {
        b.iter(|| bsm::price(black_box(&p), OptionType::Call))
    });
}

fn bench_iv(c: &mut Criterion) {
    let p = params();
    let target = bsm::price(&p, OptionType::Call);
    c.bench_function("implied_vol_solve", |b| {
        b.iter(|| implied_vol::solve(black_box(p.contract()), OptionType::Call, black_box(target)))
    });
}

fn bench_binomial(c: &mut Criterion) {
    let p = params();
    let mut group = c.benchmark_group("binomial_american_put");
    for steps in [100usize, 500, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |b, &n| {
            b.iter(|| binomial::price(black_box(&p), OptionType::Put, ExerciseStyle::American, n))
        });
    }
    group.finish();
}

// Volatility 100 with 100+ steps makes `jump * n > 600`, which takes the
// log-domain rollback instead of the ordinary tree.
fn bench_extreme_tree(c: &mut Criterion) {
    let p = OptionParams::builder()
        .spot(100.0)
        .strike(100.0)
        .volatility(100.0)
        .expiry(1.0)
        .build()
        .unwrap();
    let mut group = c.benchmark_group("binomial_extreme_log_domain_call");
    for steps in [100usize, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("european", steps), &steps, |b, &n| {
            b.iter(|| binomial::price(black_box(&p), OptionType::Call, ExerciseStyle::European, n))
        });
        group.bench_with_input(BenchmarkId::new("american", steps), &steps, |b, &n| {
            b.iter(|| binomial::price(black_box(&p), OptionType::Call, ExerciseStyle::American, n))
        });
    }
    group.finish();
}

fn bench_risk(c: &mut Criterion) {
    // Deterministic pseudo-returns (splitmix-style hash), roughly +-3%.
    let returns: Vec<f64> = (0..1_000_000u64)
        .map(|i| {
            let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.06
        })
        .collect();
    let mut group = c.benchmark_group("risk_1M_returns");
    group.sample_size(20);
    group.bench_function("historical_var_99", |b| {
        b.iter(|| risk::historical_var(black_box(&returns), 0.99))
    });
    group.bench_function("historical_cvar_99", |b| {
        b.iter(|| risk::historical_cvar(black_box(&returns), 0.99))
    });
    group.bench_function("parametric_var_99", |b| {
        b.iter(|| risk::parametric_var(black_box(&returns), 0.99))
    });
    group.bench_function("sharpe", |b| {
        b.iter(|| risk::sharpe_ratio(black_box(&returns), 0.0, 252.0))
    });
    group.bench_function("sortino", |b| {
        b.iter(|| risk::sortino_ratio(black_box(&returns), 0.0, 252.0))
    });
    group.bench_function("max_drawdown", |b| {
        b.iter(|| risk::max_drawdown(black_box(&returns)))
    });
    group.bench_function("annualized_volatility", |b| {
        b.iter(|| risk::annualized_volatility(black_box(&returns), 252.0))
    });
    group.finish();
}

fn bench_inv_cdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm_inv_cdf");
    for (name, p) in [
        ("central_0.3", 0.3),
        ("tail_1e-6", 1e-6),
        ("deep_tail_1e-300", 1e-300),
    ] {
        group.bench_function(name, |b| b.iter(|| dist::norm_inv_cdf(black_box(p))));
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bsm,
    bench_iv,
    bench_binomial,
    bench_extreme_tree,
    bench_risk,
    bench_inv_cdf
);
criterion_main!(benches);
