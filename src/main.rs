use acid2::F64;

fn main() {
    let mut results = [F64::ZERO; 64];
    for (i, x) in results.iter_mut().enumerate() {
        *x = F64::from(i as u32 + 1) + F64::ONE;
    }
    std::hint::black_box(results);
}
