use acid2::F64;
use rand::{thread_rng, Rng};

fn main() {
    // let mut results = [F64::ZERO; 64];
    // for (i, x) in results.iter_mut().enumerate() {
    //     *x = F64::from(i as u32 + 1) + F64::ONE;
    // }
    let x: F64 = thread_rng().gen();
    println!("{:?}", x);
}
