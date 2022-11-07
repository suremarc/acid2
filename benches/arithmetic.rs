use acid2::F64;
use criterion::black_box;
use criterion::BatchSize;
use criterion::Criterion;

use criterion::criterion_group;
use criterion::criterion_main;
use rand::thread_rng;
use rand::Rng;
use rand_distr::Standard;

fn bench_add(c: &mut Criterion) {
    c.bench_function("batched: 1m additions", |b| {
        b.iter_batched(
            || thread_rng().sample_iter(Standard).take(1000000).collect(),
            |data: Vec<(F64, F64)>| {
                for (x, y) in data {
                    black_box(x + y);
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_mul(c: &mut Criterion) {
    c.bench_function("batched: 1m multiplications", |b| {
        b.iter_batched(
            || thread_rng().sample_iter(Standard).take(1000000).collect(),
            |data: Vec<(F64, F64)>| {
                for (x, y) in data {
                    black_box(x * y);
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_recip(c: &mut Criterion) {
    c.bench_function("batched: 1m reciprocals", |b| {
        b.iter_batched(
            || thread_rng().sample_iter(Standard).take(1000000).collect(),
            |data: Vec<F64>| {
                for x in data {
                    black_box(x.recip());
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_sqrt(c: &mut Criterion) {
    c.bench_function("batched: 1m square roots", |b| {
        b.iter_batched(
            || {
                thread_rng()
                    .sample_iter::<F64, _>(Standard)
                    .filter(|&x| x.significand() % 8 == 1)
                    .take(1000000)
                    .collect()
            },
            |data: Vec<F64>| {
                for x in data {
                    black_box(x.sqrt());
                }
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_add, bench_mul, bench_recip, bench_sqrt);
criterion_main!(benches);
