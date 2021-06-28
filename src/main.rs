const N: i64 = 10000;
const DIMS: [usize; 14] = [4, 6, 8, 10, 14, 20, 27, 38, 52, 71, 98, 135, 186, 256];

pub fn nanotime() -> i64 {
    const NANOSEC: i64 = 1_000_000_000;
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut time) };
    time.tv_sec * NANOSEC + time.tv_nsec
}

///////////////////////////////////////////////////////////////////////////////

fn bench_ndarray() {
    use ndarray::Array;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    let dist = Uniform::new(-1.0f32, 1.0f32);

    for n in &DIMS {
        let a = Array::random((*n,), dist);
        let b = Array::random((*n,), dist);

        let t0 = nanotime();
        for _ in 0..N {
            criterion::black_box(a.dot(&b));
        }
        let dt = (nanotime() - t0) / N;
        println!("ndarray,dot,{},{}", n, dt);
    }

    for n in &DIMS {
        let a = Array::random((*n, *n), dist);
        let b = Array::random((*n,), dist);

        let t0 = nanotime();
        for _ in 0..N {
            criterion::black_box(a.dot(&b));
        }
        let dt = (nanotime() - t0) / N;
        println!("ndarray,gemv,{},{}", n, dt);
    }

    for n in &DIMS {
        let a = Array::random((*n, *n), dist);
        let b = Array::random((*n, *n), dist);

        let t0 = nanotime();
        for _ in 0..N {
            criterion::black_box(a.dot(&b));
        }
        let dt = (nanotime() - t0) / N;
        println!("ndarray,gemm,{},{}", n, dt);
    }

    for n in &DIMS {
        let a = Array::random((*n, 1), dist);
        let b = Array::random((1, *n), dist);

        let t0 = nanotime();
        for _ in 0..N {
            criterion::black_box(a.dot(&b));
        }
        let dt = (nanotime() - t0) / N;
        println!("ndarray,ger,{},{}", n, dt);
    }
}

///////////////////////////////////////////////////////////////////////////////

#[allow(dead_code)]
fn bench_nalgebra() {
    fn random() -> f32 {
        rand::random::<f32>() - 0.5
    }

    use nalgebra as na;
    type Matrix =
        na::Matrix<f32, na::Dynamic, na::Dynamic, na::VecStorage<f32, na::Dynamic, na::Dynamic>>;

    type VectorC = na::DVector<f32>; // na::Matrix<f32, na::Dynamic, na::U1, na::VecStorage<f32, na::Dynamic, na::U1>>;
    type VectorR = na::RowDVector<f32>; // na::Matrix<f32, na::U1, na::Dynamic, na::VecStorage<f32, na::U1, na::Dynamic>>;

    for n in &DIMS {
        let a = &VectorR::from_fn(*n, |_, _| random());
        let b = &VectorC::from_fn(*n, |_, _| random());

        let t0 = nanotime();
        for _ in 0..N {
            criterion::black_box(a * b);
        }
        let dt = (nanotime() - t0) / N;
        println!("nalgebra,dot,{},{}", n, dt);
    }

    for n in &DIMS {
        let a = &Matrix::from_fn(*n, *n, |_, _| random());
        let b = &VectorC::from_fn(*n, |_, _| random());

        let t0 = nanotime();
        for _ in 0..N {
            criterion::black_box(a * b);
        }
        let dt = (nanotime() - t0) / N;
        println!("nalgebra,gemv,{},{}", n, dt);
    }

    for n in &DIMS {
        let a = &Matrix::from_fn(*n, *n, |_, _| random());
        let b = &Matrix::from_fn(*n, *n, |_, _| random());

        let t0 = nanotime();
        for _ in 0..N {
            criterion::black_box(a * b);
        }
        let dt = (nanotime() - t0) / N;
        println!("nalgebra,gemm,{},{}", n, dt);
    }

    for n in &DIMS {
        let a = &VectorC::from_fn(*n, |_, _| random());
        let b = &VectorR::from_fn(*n, |_, _| random());

        let t0 = nanotime();
        for _ in 0..N {
            criterion::black_box(a * b);
        }
        let dt = (nanotime() - t0) / N;
        println!("nalgebra,ger,{},{}", n, dt);
    }
}

fn main() {
    bench_nalgebra();
    bench_ndarray();
}
