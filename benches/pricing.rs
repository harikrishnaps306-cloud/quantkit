use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quantkit::{binomial, bsm, implied_vol, ExerciseStyle, OptionParams, OptionType};

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

criterion_group!(benches, bench_bsm, bench_iv, bench_binomial);
criterion_main!(benches);
