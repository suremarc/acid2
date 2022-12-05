#![feature(iter_array_chunks)]

use acid2::core::F64;
use acid2::simd::SimdF64;
use criterion::black_box;
use criterion::BatchSize;
use criterion::Criterion;

use criterion::criterion_group;
use criterion::criterion_main;
use rand::thread_rng;
use rand::Rng;
use rand_distr::Standard;

fn bench_add(c: &mut Criterion) {
    c.bench_function("1m additions", |b| {
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
    c.bench_function("1m multiplications", |b| {
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
    c.bench_function("1m reciprocals", |b| {
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
    c.bench_function("1m square roots", |b| {
        b.iter_batched(
            || {
                thread_rng()
                    .sample_iter::<F64, _>(Standard)
                    .filter(|&x| x.significand() % 8 == 1 && x.exponent() % 2 == 0)
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

fn bench_add_simd(c: &mut Criterion) {
    c.bench_function("1m additions, simd", |b| {
        b.iter_batched(
            || {
                thread_rng()
                    .sample_iter(Standard)
                    .take(1000000 / 8)
                    .collect()
            },
            |data: Vec<([F64; 8], [F64; 8])>| {
                for (x, y) in data {
                    black_box(SimdF64::<8>::from_array(x) + SimdF64::<8>::from_array(y));
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_mul_simd(c: &mut Criterion) {
    c.bench_function("1m multiplications, simd", |b| {
        b.iter_batched(
            || {
                thread_rng()
                    .sample_iter(Standard)
                    .take(1000000 / 8)
                    .collect()
            },
            |data: Vec<([F64; 8], [F64; 8])>| {
                for (x, y) in data {
                    black_box(SimdF64::<8>::from_array(x) + SimdF64::<8>::from_array(y));
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_recip_simd(c: &mut Criterion) {
    c.bench_function("1m reciprocals, simd", |b| {
        b.iter_batched(
            || {
                thread_rng()
                    .sample_iter(Standard)
                    .array_chunks()
                    .map(|arr| SimdF64::from_array(arr))
                    .take(1000000 / 8)
                    .collect()
            },
            |data: Vec<SimdF64<8>>| {
                for x in data {
                    black_box(x.recip());
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_sqrt_simd(c: &mut Criterion) {
    c.bench_function("1m square roots, simd", |b| {
        b.iter_batched(
            || {
                thread_rng()
                    .sample_iter::<F64, _>(Standard)
                    .filter(|&x| x.significand() % 8 == 1 && x.exponent() % 2 == 0)
                    .array_chunks()
                    .map(|arr| SimdF64::from_array(arr))
                    .take(1000000 / 8)
                    .collect()
            },
            |data: Vec<SimdF64<8>>| {
                for x in data {
                    black_box(x.sqrt());
                }
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_add, bench_mul, bench_recip, bench_sqrt);
criterion_group!(
    benches_simd,
    bench_add_simd,
    bench_mul_simd,
    bench_recip_simd,
    bench_sqrt_simd
);
criterion_main!(benches, benches_simd);
